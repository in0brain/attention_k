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
