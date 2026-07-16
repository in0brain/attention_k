# Pre-registration：内部 hidden-state probe 的条件增量检验

> 本文件是投稿前的**预注册**：thesis、研究问题、O/H 定义、判读规则、前置 gate、
> 统计单位、artifact 控制,全部在跑任何大规模生成之前写死。跑完不许改判读规则,
> 只填结果。版本 v1(待目标 workshop 的 CFP 确认 deadline / page limit / archival)。

## 0. 一句话 thesis

```text
Decodability ≠ Incremental Utility。
一个 supervised hidden-state probe 能把对错分开(decodable),
不等于它在一个 task-adapted、含可见 CoT 的输出侧基线之上还有增量效用。
内部 probe 的价值可能取决于 output-space error observability
(错误在输出置信信号中的可观测性)。
```

## 1. 与现有工作的边界(核实后)

```text
LM-Polygraph (2311.07383 / 2406.15627):统一 UQ 评测框架,大规模方法对比。
  → 占据"统一 benchmark",不占据条件增量问题。
PARALLAX (2605.17028):22 方法 × 12 模型 × 6 数据集,teacher-forced vs
  live-generation,文本表面基线 TxTemb,发现多数 benchmark 有构造 artifact。
  → 明显缩窄"大矩阵"创新;但没把同协议下 O / H / O+H 的 Δ_H 作为主问题。
2605.27016 (Uncertainty relevance):uncertainty 与幻觉的相关性随错误类型/任务变化。
  → 占据"uncertainty ≠ hallucination";没回答"O 已知后 H 是否有条件增量"。
2606.10198 (Density Ridge Selective Prediction):label-scarce selective prediction
  + trajectory geometry detector。→ 不占条件增量问题。
```

我们干净的空间(不被上述占据):

```text
把 hidden-state 幻觉检测评价改写为"相对最强输出侧基线的条件增量问题",
并在同一领域下对照"输出易观测错误(MCQ)"与"高置信 fabrication(H1)",
检验内部信号的增量是否随 error observability 变化。
```

## 2. 研究问题与主量

### RQ1(条件增量)

```text
Δ_H = AUROC(O + H) − AUROC(O)
O = 输出侧最强基线 = max{ F5, full-response-text, F5+text }   (见 §5 阶梯)
H = 预注册的 hidden-state probe(SAPLMA-style,见 §6)
统计:paired grouped-bootstrap(按 source prompt 分组,≥1000 resample)的 95% CI。
```

**判读规则(预注册,三选一,不预设方向):**

```text
CI 全 > 0    → 有稳定条件增量
CI 跨 0      → 无证据支持增量(本方向的主要预期)
CI 全 < 0    → 内部特征显著损害泛化
```

同时报告 AUPRC 增量(类不平衡下必须)。

### RQ2(增量是否取决于 observability)

```text
主检验(within-H1,有统计力):
  按 id-token logprob 把 H1 的 fabrication 分成
  "高置信(low id-logprob 反例:高 logprob)" vs "低置信" 两层,
  检验 Δ_H 是否在高置信层更大(分层各自的 paired bootstrap CI + 层间差)。
粗对照(跨任务,2 点趋势,非 interaction):
  Δ_H(MCQ) vs Δ_H(H1)。
诚实边界:只有两个任务时这是 2 点对照,不是可拟合的 observability→Δ_H 函数;
  within-H1 分层才是有统计力的那一半。
```

## 3. 前置 gate(4D-2 第一步,先于任何 probe bake-off)

必须先证明 H1 确实是"高置信 fabrication 任务",否则核心解释变量没被构造出来。

```text
度量(base-rate 稳健,不能用跨任务裸 AUROC):
  每个任务内 error-vs-correct 的 confidence 分布可分性——
  Cohen's d、分布直方图、AUPRC,而非 AUROC(MCQ) vs AUROC(H1)。
  confidence 用 §5 的输出侧信号(F5:label margin/entropy;H1:id-token logprob)。
通过判据:
  H1 的 fabricated / valid confidence 分布重叠明显 > MCQ 的 wrong / correct。
Outcome 3(如实报告):
  若 H1 的 output-only 可分性仍很强(fabrication 也低 logprob),
  则 H1 未构造出高置信错误设置;此时论文结论转为
  "在我们能构造的两个任务上,输出侧错误都可观测,内部增量在两者都近零",
  不得假装 H1 是高置信设置。
```

## 4. 任务(controlled cross-task,非 matched)

```text
MCQ(CyberMetric,有限标签 {A,B,C,D},已有 4C 数据/特征):
  label = 答案正确性 1[â = a*]。
H1(自由生成 fabricated identifier,4D-0/4D-1 已有数据):
  label(primary,completion-level)= completion 是否含 ≥1 个 fabricated id。
两者 label 定义不同,不是 matched design;称 controlled cross-task comparison。
```

## 5. 输出侧基线阶梯(artifact 控制,受 PARALLAX 启发)

按从弱到强跑,取最强的当 O;并用弱阶梯排除 benchmark shortcut:

```text
1. prompt-only            (只看问题文本)
2. id-string-only         (只看抽出的 identifier 字符串/格式)
3. surface-format-only    (长度、模板、ontology 前缀等表面特征)
4. full-response-text     (完整生成文本的分类器)
5. F5                     (输出层:label margin/entropy、id-token mean/min logprob、
                           id-agreement/self-consistency、verbalized confidence
                           —— 复用 h1_f5_features.py)
6. F5 + full-response-text = O(输出侧最强)
```

**artifact 红线**:若 `id-string-only` 或 `surface-format-only` 已接近 full model,
说明 H1 被字符串/格式模式解决,不能用于声称 hidden-state detection;如实标注并
在该任务上不做增量声明。

## 6. Hidden probe H(预注册,防 post-hoc 选层)

```text
方法:SAPLMA-style —— 在预注册的层/位置/pooling 上取 hidden,套 grouped-CV logistic。
预注册 primary(跑前定死,不许事后挑最优层):
  层:相对深度 0.7·L(H)。
  位置:MCQ = answer-readout token;H1 = identifier-token(mention-level 定位,
        completion 级用 identifier-token 的 mean pooling)。
  pooling:mean over 目标 token 位置。
robustness(附录,不用于挑最优):相对深度 {0.5, 0.7, final} 各报一版,
  但 primary 恒为 0.7·L,增量声明只基于 primary。
可选内部方法(复现不超过 2 天才保留):
  EigenScore(INSIDE,2402.03744):跨采样 mid-layer embedding 协方差谱。
Semantic Entropy(Kuhn/Farquhar):H1 上"语义等价"退化为归一化精确 id 匹配,
  即 = 跨采样 id-agreement(h1_f5_features.id_agreement_rate);
  预期在稳定 fabrication(K 次同一假 id)上失效,作为 error-mode 分析的采样 baseline,
  不当"第三个独立 probe"。
```

## 7. 统计单位与分组

```text
H1 primary unit:completion-level(F5/SAPLMA/SE/CoT 都天然 completion 级)。
H1 secondary unit:mention-level(仅用于 id-token logprob 的 observability 分析)。
两个角色预注册,不混当 primary。
CV 与 bootstrap:一律按 source prompt / question 分组;K 个 completion 绝不当独立样本。
seed:不把"3 classifier seed"当 robustness;用固定预注册 grouped folds +
  paired grouped bootstrap(+ 可选 repeated grouped CV)。更多独立问题 > 更多 seed。
```

## 8. 模型

```text
Qwen2.5-7B:全套(阶梯 + H + 可选 SE/EigenScore + observability 分层)。
Llama-3.1-8B:只跑核心 O / H / O+H,复核"无增量是否 Qwen 特例"。
跨模型协议固定:同相对层深、同 pooling、同 position 定义、同正则搜索空间、
  同 grouped split 原则;不对两模型分别找最优层(防新的 researcher DoF)。
```

## 9. 明确删除(不做,或留 archival)

```text
H1 上的 F1 / F4:F4 是有限标签 exact VJP,H1 无固定标签集,不重定义不能用。删。
raw attention 全缓存:体量 N_layer×N_head×L²,过大且 MVP 无 attention 方法。删。
TruthfulQA:teacher-forced 版有表面 artifact(PARALLAX);archival 再加,且只用
  live-generation protocol(自由生成 + 事后 judge,reference 不进模型输入)。
"calibration"作为核心术语:改用 output-space error observability。
  ECE / Brier / risk-coverage 降为 secondary selective-prediction 交付,不作 AUROC 解释。
```

## 10. 缓存契约(一遍前向,预注册)

```text
每条 trace 一次前向,落盘:
  completion 文本、token logprobs、
  §6 预注册层/位置/pooling 的 hidden、identifier 字符 span→token 位置。
不缓存 raw attention。
复用 h1_f5_features.py:mention_logprob / sequence_logprob / id_agreement /
  verbalized_confidence / looks_like_structured_identifier_list。
```

## 11. 最多允许的结论

```text
在 CyberMetric MCQ(输出易观测错误)与 H1 fabricated-identifier(高置信 fabrication,
经前置 gate 确认)上,对最强输出侧基线 O(F5 + 可见文本),supervised hidden-state
probe H 的条件增量 Δ_H 为 [填],其 95% paired grouped-bootstrap CI 为 [填];
Δ_H 是否随 error observability(within-H1 分层)变化 = [填]。这是关于内部信号
相对输出侧条件增量的结论,不是幻觉减少 / 准确率提升 / intervention 结果。
```
