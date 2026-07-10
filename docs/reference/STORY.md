# 项目故事线：从 Reasoning-Aware Attention Guidance 到 MLP Readout Attribution

> 本文件是项目的故事线与研究叙事备忘。
> 它用于记录当前研究脉络、阶段性结论，以及 Workspace / J-lens / NLA 等外部论文的接入计划。
>
> 本文件不是执行优先级最高的任务说明。
>

---

## 1. 原始研究目标

本项目最初的目标是建立一种 **Reasoning-Aware Attention Guidance** 方法。

最初假设是：

```text
我们可以发现哪些 question span 对推理真正重要；
如果模型被引导去关注或使用这些 span，
那么推理轨迹会更稳定，错误/幻觉会减少。
```

最初的研究链条是：

```text
span 重要性发现
→ attention / representation guidance
→ 推理轨迹稳定
→ 答案更正确
→ 幻觉减少
```

经过 Sprint 2 到 Sprint 3C 的实验，这个假设被逐步修正。

当前更准确的结论不是：

```text
span importance 没用。
```

而是：

```text
span-level importance 更适合作为诊断信号，
但它不是可以直接用于 answer steering 的控制变量。
```

---

## 2. Sprint 2：span-level 信号存在，但有上限

Sprint 2 主要探索静态 hidden / attention / output-effect 特征能否对重要 span 排序。

主要结果是：

```text
attention 和 response-position output-effect 中确实存在弱但真实的 keyness signal；
但静态 hidden+attention 排序遇到了明显上限。
```

系统可以一定程度地区分 on-path span 和 off-path span，但这个信号还不足以直接支持稳健干预。

解释是：

```text
模型暴露了 span relevance signal，
但 relevance signal 不等于 causal actuator。
```

也就是说，能检测到“某个 span 重要”，不代表能直接增强这个 span 来修正模型答案。

---

## 3. Sprint 3A：attention-bias steering 失败

Sprint 3A 测试的是：

```text
如果提高 answer-position 对 selected span 的 attention，
模型答案是否会更接近 gold answer？
```

干预形式大致是：

```text
attention_score(q, k) += lambda
```

预期是：

```text
oracle span 应该优于 random span。
```

实际结果是：

```text
oracle span 没有稳定优于 random span。
```

当 lambda 增大时，输出分布确实会动，但这种变化不是选择性的，而且会带来 harm。

解释：

```text
让模型“多看”某个 span，
不等于让模型“正确使用”这个 span。
```

attention score 更像 routing 权重，而不是计算变量或答案写入变量。

---

## 4. Sprint 3B：span residual injection 失败

Sprint 3B 将通道从 attention logit 改成 residual injection。

干预形式大致是：

```text
h_answer += beta * ||h_answer|| * unit(span_residual_deviation)
```

预期是：

```text
oracle span residual injection 应该优于 random span residual injection。
```

实际结果是：

```text
oracle 不稳定优于 random。
```

residual channel 的强度足够大，能明显扰动输出，但效果仍然不选择性。

解释：

```text
span representation 不是干净的 causal control variable。
```

一个 span 的 residual 表示中混合了语义、位置、格式、局部上下文和噪声。把它直接注入 answer position，更像是制造大扰动，而不是修复推理。

---

## 5. Sprint 3C-0：初版 activation patching 过于悲观

Sprint 3C-0 从 same-run span injection 转向 correct-vs-wrong activation patching。

核心问题是：

```text
把 correct run 的 activation patch 到 wrong run 的 reasoning-step 位置，
能不能修复 wrong answer？
```

初版结果看起来是负的。

但后来发现 answer proxy 有 bug：

```text
first_token_id(" " + answer)
```

在 Qwen-style tokenizer 下可能取到 leading whitespace token，导致 gold answer 和 wrong answer 的 first token 都退化成同一个空格 token。

因此：

```text
3C-0 初版的 flat null 部分是 instrumentation artifact。
```

这不是科学负结果，而是指标读法不正确。

---

## 6. Sprint 3C-0-Fix：答案读出位出现修复信号

3C-0-Fix 修正了 answer proxy：

```text
在 final answer 数字前一位读取 logits；
使用完整数字答案序列的 logprob；
避免 leading-space token bug；
避免把列表序号误判为答案。
```

修正后结果发生变化：

```text
reasoning-step patch：
  相比 no_patch 略有正向效果；
  但不稳定优于 random donor 或 random position。

answer-readout patch：
  clean_direction 很强；
  稳定优于 random donor；
  但高层 harm 较大，并且不完全 site-specific。
```

解释：

```text
修复信号并不清楚地位于显式 reasoning-step token；
它更集中在 final-answer compression / answer readout 阶段。
```

这一步将研究重点从 visible reasoning-step intervention 转向 answer-readout causal tracing。

---

## 7. Sprint 3C-1：MLP output 是选择性的答案读出写路径

Sprint 3C-1 将 answer-readout residual update 拆成三条路径：

```text
attention_output
mlp_output
residual_output
```

结果是：

```text
attention_output：
  donor-specific，但不 site-specific。

residual_output：
  幅度很大，但 high harm，且非选择性。

mlp_output：
  donor-specific；
  site-specific；
  存在 low-harm regime；
  是目前最清晰的 causal handle。
```

解释：

```text
final-answer readout 位置的 MLP output，
是 3A → 3C 系列中第一个明确的正向因果位点。
```

它也解释了前面为什么失败：

```text
attention-bias 太间接；
span residual injection 太粗；
whole-residual patching 太宽，容易 high harm；
answer-readout MLP output 才是更窄的答案写入路径。
```

---

## 8. Sprint 3C-2：MLP 方向具有机制解释，但 donor-free steering 不稳健

Sprint 3C-2 分析了 MLP readout direction。

主要发现：

```text
MLP readout correct-wrong delta 是稳定、低秩的方向；
在 L24 上，该方向与 gold-vs-wrong unembedding direction 稳定对齐；
gold-unembedding direction 是强、低 harm、first-step 存活的 steering axis。
```

但关键限制是：

```text
gold_unembed 需要知道 gold answer，
因此它只是 oracle upper bound。

mean_delta 和 PC1 等 donor-free 方向较弱，
不能构成稳健的可部署 steering handle。
```

解释：

```text
有效轴本质上是“指向 gold answer token 的方向”。
如果不知道答案，盲目的 donor-free steering 仍然不可靠。
```

这再次回到了本项目反复出现的主题：

```text
最强的信号更像 attribution / detection signal，
而不是 blind steering signal。
```

---

## 9. Sprint 3C-3：转向 MLP Readout Attribution Probe

Sprint 3C-3 应停止 steering，转向诊断。

核心问题是：

```text
在不把 gold answer 作为输入的情况下，
final-answer readout 位置的 MLP output
能否诊断模型正在 commit 到哪个答案，
以及这个 commitment 是否高风险？
```

primary readout 是：

```text
MLP_output @ W_U
```

其中 `W_U` 是模型的 unembedding / lm_head 矩阵。

需要检查的内容包括：

```text
top projected number tokens
projection margin
projection entropy
layer agreement
agreement with model final answer
risk score
correct/wrong AUROC
calibration
high-risk / low-risk buckets
```

gold answer 只能作为 evaluation label，不能作为 feature。

预期贡献是：

```text
MLP readout attribution 提供一个机制化、gold-free 的诊断信号，
用于分析模型当前的答案 commitment 和 answer risk。
```
### Sprint 3C-3 已完成结果

3C-3 已按 attribution / detection probe 完成。核心结果是：

```text
MLP risk AUROC = 0.653
MLP risk AUPRC = 0.638
logistic probe AUROC = 0.623
logistic probe AUPRC = 0.661
high-risk bucket wrong rate = 0.786
low-risk bucket correct rate = 0.692
risk ECE = 0.286
```

解释边界：

```text
MLP risk beats random baseline；
但 MLP risk does not beat final-logits margin baseline。
```

因此当前更精确的结论是：

```text
MLP readout attribution 是机制化诊断信号；
它能提供 gold-free 的 answer-risk evidence；
但它不是优于 final logits 的实用 detector。
```

这意味着 3C-3 的价值主要是解释模型在 answer-readout MLP 处“往哪个答案 token 写”，而不是替代 final logits 做部署级错误检测。

---

## 10. 外部论文接入计划

当前有两条外部 interpretability 线索与本项目相关：

```text
1. Workspace / J-lens
2. NLA activation verbalization
```

它们应该服务于 attribution / detection 转向，而不是重新开启 blind steering。

---

## 11. Workspace / J-lens 的接入方式

Workspace 论文提出，模型内部只有一部分 privileged / verbalizable representations 参与显式推理、报告和调制。

这与当前项目经验吻合：

```text
attention 到某个 span 不够；
只有当 span 信息进入可读出的 answer/workspace-like representation，
它才真正进入可诊断、可归因的状态。
```

J-lens 可以看作比 logit-lens 更原则化的中间激活读出方法：

```text
intermediate activation
→ average Jacobian to final residual / logits
→ readable token-level interpretation
```

在本项目中的使用方式：

```text
3C-3 primary：
  logit-lens / unembedding projection。

3C-3 optional：
  如果存在 Qwen2.5-compatible 的 J-lens 实现，
  或者可以低成本近似，
  则比较 J-lens top-k readout 与 logit-lens top-k readout。
```

注意：

```text
不要让 J-lens 成为 3C-3 的主路径依赖。
```
3C-3R / 3C-4 中，J-lens 应作为优先 sanity check，但不作为主路径依赖。建议优先做小规模、低成本版本：

```text
small-pair finite-difference / approximate J-lens；
对比 J-lens top-k 与 logit-lens top-k；
检查二者是否在 answer-readout MLP attribution 上给出一致或互补读数。
```

如果 J-lens 与 logit-lens 分歧，也不应立即推翻 3C-1/3C-2 的 causal tracing 结果；它首先是读出方法的 sanity check，而不是新的主证据链。

---

## 12. NLA 的接入方式

NLA 提供 activation verbalizer / reconstruction，可以将 activation 翻译成自然语言描述。

与本项目相关的 checkpoint 是：

```text
kitft/nla-qwen2.5-7b-L20-av
kitft/nla-qwen2.5-7b-L20-ar
```

它与本项目相关，是因为：

```text
本项目使用 Qwen2.5-7B-Instruct；
关键层包括 L20；
NLA checkpoint 也对应 Qwen2.5-7B-Instruct 的 L20。
```

潜在用途：

```text
1. 做 L20-only qualitative sanity check；
2. 解码 correct vs wrong 的 MLP readout activation；
3. 解码 patched vs unpatched 的 answer-readout activation；
4. 观察 verbalization 是否提到：
   - gold answer
   - wrong answer
   - arithmetic operation
   - uncertainty
   - misleading recovery
```

重要边界：

```text
NLA 是 optional；
NLA 是 qualitative / auxiliary evidence；
NLA explanation 不是 primary causal evidence；
不要把 L20 NLA 直接套到 L24 activation；
使用前必须做 sanity check。
```
3C-3R 后需要额外明确：

```text
3C-2 最清晰的机制信号在 L24；
当前可用 NLA checkpoint 只有 L20；
NLA 可能 verbalize 的是 residual stream；
而本项目当前对象是 answer-readout 位置的 MLP output。
```

因此：

```text
NLA L20 空结果不能反证 L24 MLP readout 发现；
NLA verbalization 空结果也不能直接反证 MLP-output attribution；
它最多说明“该 L20 / 该 verbalizer / 该 residual 读出未复现文本化证据”。
```

---

## 13. 两篇论文对本项目的影响

它们不会推翻当前路线，而是进一步支持当前转向。

更准确地说：

```text
logit-lens / J-lens：
  强化 MLP readout attribution。

NLA：
  提供 L20 verbalization cross-check。

Workspace theory：
  给出理论框架，解释为什么 MLP readout
  比 raw attention weights 更接近真正的 reasoning anchor。
```

推荐接入方式：

```text
3C-3：
  MLP Readout Attribution Probe
  + optional logit-lens / J-lens / NLA checks。

Sprint 5：
  将 workspace / verbalizability 作为 anchor labeling 新特征。
```

---

## 14. 当前最佳故事线

当前最合理的项目故事线是：

```text
1. span-level relevance 确实存在；
2. 但 span-level blind steering 失败；
3. attention routing 不等于 reasoning control；
4. span residual injection 太粗；
5. correct-run activation repair signal 出现在 answer readout；
6. module-level tracing 将有效 causal write path 定位到 MLP output；
7. direction analysis 说明有效方向指向 gold answer token；
8. 但 gold-free donor-free steering 仍然很弱；
9. 因此项目从 steering 转向 attribution / detection；
10. MLP readout attribution 成为当前最清晰的机制化诊断结果，但不是优于 final logits 的实用 detector。
```

简短版本：

```text
blind steering failed；
answer-readout MLP causal path is real；
gold-free steering is weak；
MLP readout attribution is a mechanistic diagnostic signal, but not a practical detector superior to final logits。
```

中文总结：

```text
盲目干预失败；
答案读出位的 MLP 因果路径成立；
无 gold 的 steering 仍不稳健；
MLP readout attribution 是当前最合理的机制化诊断方向，但不是优于 final logits 的实用 detector。
```

---

## 15. 后续 sprint 安排

近期推荐计划：

```text
Sprint 3C-3：
  MLP Readout Attribution Probe（已完成）。

Sprint 3C-3R / 3C-4：
  story / baseline cleanup；
  small-pair finite-difference / approximate J-lens sanity check；
  NLA L20 sanity check（仅辅助，不反证 L24 MLP readout）。

3C-3 optional extension：
  logit-lens / J-lens comparison；
  NLA L20 sanity check。

后续 Sprint 5：
  workspace / verbalizability 作为 anchor feature。
```

在 attribution / detection 故事稳定之前，不建议重新开启 broad steering。

---

## 16. 边界声明

不得声称：

```text
hallucination reduced
answer accuracy improved
ready for 2000
steering solved
```

除非后续 sprint 明确完成并支持这些评估。

当前证据最多支持：

```text
final-answer readout 位置的 MLP output
是 teacher-forced proxy evaluation 下
answer commitment 的选择性因果写路径。

gold-free donor-free steering 仍然较弱。

MLP readout attribution 是一个有前景的机制化诊断方向；当前证据不支持把它写成优于 final logits 的实用 detector。
```
---

## 17. 为什么最初的 span-guided steering 计划无效

最初的计划不是完全错误。它正确识别了一个重要事实：输入中的某些 span 确实更接近推理依赖、错误敏感性和答案稳定性。但 Sprint 3 的结果说明，这个事实只回答了“哪里重要”，没有回答“答案应该往哪个方向改”。

核心问题是缺少 span-to-direction mapping：

```text
span relevance -> where evidence may be
steering direction -> how the answer state should move
```

两者不是同一个对象。

具体边界如下：

```text
1. span relevance 只能说明“哪里重要”，不能说明“答案应该往哪个方向改”；
2. attention bias 改变的是信息读取比例，不等于执行正确推理；
3. span residual injection 改变的是高维混合状态，不等于 gold answer direction；
4. answer-readout MLP 确实是 causal write path，但仍需要知道 target direction；
5. gold-unembedding direction 有效，但它需要 gold answer，因此只是 oracle upper bound；
6. direct readout / approximate J-lens 没有提供足够强的 gold-free target；
7. 因此原始无监督 steering 主线无效；
8. 新主线必须引入领域监督 probe，学习：
   span / activation / trace -> label direction / correction direction。
```

最初的 Reasoning-Aware Attention Guidance 计划失败的核心原因，不是模型状态无法被干预，而是“推理相关 span”没有自动给出“答案修正方向”。加重一个 span 的 attention 或 residual influence，确实会改变模型内部状态，但这种改变可能增强正确证据，也可能增强错误中间量、格式偏置、数字 token sharpness 或已有错误 commitment。换言之，span 是 evidence，不是 controller。真正缺失的是一个从 evidence 到 direction 的映射函数。Sprint 4 的新主线正是训练这个函数：在网络安全领域中，利用结构化标签和 correct/wrong 对比，训练 direction probe / controller 来预测 answer-readout MLP 的 steering direction。

3C-4A 对这个判断的补充是：direct logit-lens readout 和 approximate J-lens readout 都较弱，approximate J-lens 没有显著改善 3C-3 的 diagnostic result，final-logits margin 仍然更强。因此不能把 gold-free readout 当成可直接替代 gold direction 的控制信号。

---

## 18. Sprint 4 新主线：领域监督 direction probe

Sprint 4 不再把 attention 或 span 本身当作 steering handle。Sprint 4 的核心假设是：span / attention / MLP readout 等 reasoning-aware signals 可以作为 probe 输入，但需要领域监督来学习它们到 answer direction 的映射。

在网络安全领域中，答案空间通常比 GSM8K 数字答案更结构化，例如 MITRE technique、CWE、攻击阶段、告警类别、误报/真阳性、缓解措施或多选项答案。这类标签空间使得“正确方向”更容易定义为 label-unembedding direction 或 correct-wrong MLP delta。因此，Sprint 4 的目标不是恢复无监督 steering，而是训练一个 domain-supervised direction probe：它在冻结 LLM 的前提下，根据当前 span、activation 和 readout features 预测应当增强的 label / direction，再通过低强度 MLP readout steering 或 risk-controlled retry 进行评估。

Sprint 4 的新主线是：

```text
cyber structured labels
-> correct/wrong label contrast
-> reasoning-aware activation/readout features
-> supervised direction-selection probe
-> low-alpha, harm-controlled evaluation
```

边界仍然必须保持清楚：

```text
Sprint 4A 没有训练 probe；
Sprint 4A 没有调用模型；
Sprint 4A 没有执行 steering；
Sprint 4A 没有证明 hallucination reduction；
Sprint 4A 没有证明 answer accuracy improvement；
Sprint 4A 没有说明 cyber probe 已经有效。
```

Sprint 4B 的下一步应是网络安全数据集选择与 domain schema 实现，而不是直接训练或 steering。

---

## 19. 新文献证据与 3C 结果的互证

在 4A reset 之后，对以下文献做了接入评估：Sun et al. 2026（LLM Reasoning as Trajectories，Microsoft）、INSIDE（ICLR'24）、LM-Polygraph、CausalGaze、Causal Tracing of Object Representations（AAAI'26）、Beyond Transcription（ASR，AAAI'26）。

Sun et al. 与本项目形成三处互证：

```text
1. 他们发现 correct/incorrect 轨迹只在最后两步系统性分叉，早期步不分叉
   —— 与我们 3C-0 的发现（reasoning-step 位置无信号、answer-readout 有信号）独立收敛。

2. 他们用 late-step transition 特征（相邻 step 激活差分）做答案前正确性预测，
   AUC 0.852，跑赢 LogitLens baseline（0.765）
   —— 直接解释了我们 3C-3 为什么输给 final logits：
      我们用的是单前向、单位置的静态特征，从未测试 transition 特征族。

3. 他们自己的修正与我们的负结果一致：
   - freeform 无脚手架时正确性预测崩塌到 AUC 0.60（依赖 "Step k:" 格式）；
   - 跨任务迁移掉到 0.64-0.73（correctness 几何必须逐域校准）；
   - gated intervention 只修好 26/90、改坏 14 个（detection >> correction）；
   - 无条件干预有害（掉 1.6%-36%）。
```

INSIDE 提供了我们从未测试的第三族特征（多采样内部一致性 / EigenScore）。LM-Polygraph 提供 baseline 纪律（很多 UQ 方法打不过简单 baseline，必须报增量）。两篇 AAAI'26 因果追踪论文说明 "causal localization → site-informed detection/interception" 正在成为跨模态范式，我们的 3C-1 位点是它在文本推理上的实例。

## 20. 主线修订：从 direction probe 到幻觉检测与门控干预

据此把 Sprint 4 主线从「supervised direction probe / controller」修订为「domain-calibrated hallucination detection + gated closed-form intervention」，计划书见 `CYBER_HALLUCINATION_CONTROL_PLAN.md`（取代 `CYBER_DIRECTION_PROBE_PLAN.md`，原文件保留作历史）。

修订理由：

```text
1. instance-level answer steering 被结构性排除（3C-2：有效方向需要 gold）；
2. 原计划的 vector controller / probe-guided steering 是换皮微调；
3. Sun et al. 证明 transition 特征能跑赢 logits baseline——3C-3 的失败模式可修；
4. 检测路线推理时不需要 gold；干预只保留 closed-form 门控形式。
```

新主线的三层结构与五族特征：

```text
DETECT:  F1 静态位点特征（3C-3）
         F2 anchor-free 轨迹 transition 特征（Sun 配方 + 我们的角色定位机制，
            修他们的 freeform 崩塌——这是相对 Sun et al. 的核心差异化）
         F3 多采样内部一致性（INSIDE）
         F4 exact J-lens 标签投影（见下）
         F5 输出层 baseline 套件（生死门：F1-F4 必须对 F5 有增量 AUROC）
DECIDE:  逐任务族校准 + conformal / selective prediction
INTERVENE(门控可选): 反思 token 注入 / 约束重解码 / 弃权，
         报告 net fix/break 账本；禁止 learned vector、禁止逐题答案 steering。
```

J-lens 的接入方式（回应 §11 与 3C-4A 的遗留）：真 J-lens 全词表不可行，3C-4A 只能用有限差分近似（Case C，对齐弱）。但 cyber 有限标签空间使**精确** J-lens 变得便宜：只需候选标签 k 个 token 的投影，每个标签一次 VJP（k=2-10 次反向传播），无 epsilon、无差分噪声。同时检测生死门可以仲裁 3C-4A 留下的疑团——若 F4 有增量而 F1 没有，说明直接 logit-lens 确实误读中间层（一个可发表的方法学发现）；若 F1≈F4，说明便宜读法够用、3C-4A 的分歧是近似噪声。

NLA 维持 §12 的定位：只对 FLAGGED 高风险样本做 L20 定性 verbalization。注意 L20 不是我们的最强层（对齐信号在 L24），且 verbalizer 训练于 residual 激活、喂 MLP-output 可能 OOD——预期它可能看不到现象，空手而归不构成反证。

## 21. 修订后的 Sprint 4 路线

```text
4B  数据集 + 幻觉操作化定义（编造标识符/错误映射/自信错误分类）+ F5 baseline；
    标签空间用选项字母（单 token；ATT&CK id 会复现 3C-0 的 leading-token 退化）；
    廉价加项：在 label-readout 位置复刻 3C-1 位点验证（无训练）——
    位点不迁移则 F1 动机减弱，尽早知道。
4C  五族特征 bake-off（生死门：对 F5 的增量 AUROC，分组 CV，3 seeds）
    + J-lens vs 直接投影仲裁 + 校准。
4D  （仅生死门通过后）门控干预：net fix/break 账本 + 固定 coverage 错误率。
4E  held-out 任务族 + 消融（anchor-free vs 脚手架；位点 vs 盲选层）+ write-up。
```

全程冻结模型；唯一训练组件是线性探针与校准温度。声明纪律不变：只用 selective-prediction 与 fix/break 术语，不写 unqualified 的 "hallucination reduced"。即使 F1-F4 全部无增量，F5 基础上的 selective-prediction 系统仍是一个诚实可交付的幻觉控制结果——这条路线没有"全输"的分支。
