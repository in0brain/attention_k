# 实验指导书：基于 NLI 语义必要性与问题复原的关键 Span 发现

## 一、实验目标

本实验旨在构建一套关键 span 发现流程，用于判断输入问题中的哪些信息对语义理解、推理过程和最终答案具有不可替代作用。

本实验不直接从 attention 开始，而是先建立两个基础标签：

```text
1. Semantic Necessity
   用 NLI 判断某个 span 被删除或泛化后，原问题和改写问题是否仍然等价。

2. Model Recoverability
   用 mask-recover 判断某个 span 被遮盖后，模型是否还能稳定恢复原问题。
```

后续再用这些标签分析模型是否存在：

```text
忽略关键 span
过度关注非关键 span
缺信息乱补
高置信错误恢复
异常 attention sink
```

最终目标是训练 token / span importance probe，并尝试 attention guidance。

------

# 二、核心方法

## 1. Candidate Span Extraction

从原始问题中抽取候选 span。

候选 span 不限于单个 token，也可以是短语。

优先抽取：

```text
number
entity
operation
relation
comparison
negation
condition
question_target
cyber_security_term
```

示例：

```text
Tom has 3 apples and buys 2 more. How many apples does he have now?
```

可抽取：

```text
Tom
3
apples
buys
2
more
How many
now
```

如果是网络安全问题：

```text
A vulnerability allows remote code execution through unsafe deserialization.
```

可抽取：

```text
vulnerability
remote code execution
unsafe deserialization
```

------

## 2. Ablated Question Construction

对每个 candidate span 构造 ablated question，用于 NLI 判断。

主要消融方式：

```text
delete
generalize
mask
```

前期建议：

```text
NLI semantic necessity 阶段：优先使用 delete / generalize
Mask-recover 阶段：使用 mask
```

示例：

原问题：

```text
Tom has 3 apples and buys 2 more. How many apples does he have now?
```

对 `3` 做 generalize：

```text
Tom has some apples and buys 2 more. How many apples does he have now?
```

对 `Tom` 做 generalize：

```text
Someone has 3 apples and buys 2 more. How many apples does he have now?
```

对 `buys` 做 mask：

```text
Tom has 3 apples and [MASK] 2 more. How many apples does he have now?
```

------

## 3. NLI Semantic Necessity Scoring

对 original question 和 ablated question 做双向 NLI。

比较方向：

```text
original question → ablated question
ablated question → original question
```

标签规则：

| 双向 NLI 结果                                            | 标签             | 含义                 |
| -------------------------------------------------------- | ---------------- | -------------------- |
| 双向 entailment                                          | Equivalent       | 改写后语义基本等价   |
| original entail ablated，但 ablated 不能 entail original | Information Loss | 改写后丢失了精确信息 |
| ablated entail original，但 original 不能 entail ablated | Added Assumption | 改写后加入了额外假设 |
| 双向都不 entail                                          | Non-equivalent   | 改写后语义明显变化   |

解释：

```text
Equivalent:
  该 span 语义必要性低。

Information Loss:
  该 span 携带关键约束或精确信息，语义必要性高。

Added Assumption:
  改写引入了原问题没有的信息，存在误导风险。

Non-equivalent:
  删除或泛化后语义变化明显，语义必要性高。
```

------

## 4. Mask-Recover Model Recoverability

对 candidate span 构造 masked question，让模型只复原问题，不解题。

输入示例：

```text
Tom has [MASK] apples and buys 2 more. How many apples does he have now?
```

要求模型输出：

```text
Recovered Question:
...

Recoverable:
yes / no / uncertain

Confidence:
0-1

Reason:
...
```

注意：

```text
1. 不允许模型解题。
2. 每个 masked question 采样 K 次。
3. 后续用 NLI 或规则判断 recovered question 是否等价于 original question。
```

------

## 5. Recoverability Labeling

根据多次复原结果给 span 打 recoverability 标签。

| 标签                  | 含义                         |
| --------------------- | ---------------------------- |
| Recoverable           | mask 后问题仍可稳定恢复      |
| Partially Recoverable | 大意能恢复，但精确信息不稳定 |
| Non-recoverable       | 无法稳定恢复                 |
| Misleading Recovery   | 模型高置信恢复错误内容       |
|                       |                              |

重点关注：

```text
Non-recoverable:
  说明该 span 不可替代。

Misleading Recovery:
  说明模型容易在缺信息时乱补，是幻觉风险信号。
```

------

## 6. NLI 与 Recoverability 对比

这一阶段是 v0 的核心分析。

需要找四类案例：

| 类型                          | 含义                                             |
| ----------------------------- | ------------------------------------------------ |
| NLI 重要 + Recover 不可恢复   | 人类语义和模型行为一致，是真关键 span            |
| NLI 重要 + Recover 可恢复     | 模型能靠常识或模式补全，但可能存在高置信乱补风险 |
| NLI 不重要 + Recover 不可恢复 | 模型对人类不关键成分过度敏感                     |
| NLI 不重要 + Recover 可恢复   | 稳定低风险 span                                  |

最有研究价值的是中间两类不一致情况。

------

## 7. Incomplete-Question Reasoning

使用同一批 masked questions，让模型直接解题。

Prompt 要求：

```text
If the masked information is necessary and cannot be inferred, say the answer cannot be determined.
Do not guess a specific value unless it is logically implied.
```

需要识别的行为：

| 行为                          | 含义                   |
| ----------------------------- | ---------------------- |
| cannot_determine              | 正确意识到信息不足     |
| symbolic_reasoning            | 用符号表达未知量       |
| guessed_then_solved           | 先猜缺失信息再解题     |
| specific_answer_without_basis | 无依据给出具体答案     |
| wrong_recovery_reasoning      | 错误恢复问题后继续推理 |

高风险情况：

```text
Non-recoverable + specific_answer_without_basis
Non-recoverable + guessed_then_solved
Information Loss + high-confidence wrong recovery
```



# 三、实验阶段

## 阶段 v0：最小闭环实验

### 目标

跑通：

```text
candidate span
→ ablated question
→ NLI
→ masked question
→ recovery
→ label comparison
```

### 样本规模

```text
30–50 条
```

可选数据：

```text
GSM8K
MATH-500
SecQA
CTIBench
CyberMetric
```

前期建议：

```text
先用 GSM8K 跑通流程。
再用 SecQA / CTIBench 做网络安全方向扩展。
```

### 步骤

```text
1. 准备 questions.jsonl
2. 抽取 candidate_spans.jsonl
3. 构造 ablated_questions.jsonl
4. 执行双向 NLI，得到 nli_scores.jsonl
5. 构造 masked_questions.jsonl
6. 执行问题复原，得到 recover_outputs.jsonl
7. 统计复原结果，得到 recover_scores.jsonl
8. 整合 NLI 和 recover 标签，得到 token_labels.jsonl
9. 人工检查典型案例
```

### 验收标准

```text
[OK] 每条 question 能抽出 candidate spans
[OK] 每个 candidate span 至少有一个 ablated question
[OK] 每个 ablated question 有双向 NLI 结果
[OK] 每个 candidate span 至少有一个 masked question
[OK] 每个 masked question 至少有 K 次 recovered question
[OK] 能输出 semantic necessity label
[OK] 能输出 recoverability label
[OK] 能对比 NLI label 与 recoverability label
[OK] 能输出 10 个高质量案例
```

------

## 阶段 v1：批量标签构建

### 目标

扩大样本，构建稳定的 span-level 标签数据。

### 样本规模

```text
300–500 条
```

### 步骤

```text
1. 批量抽取 candidate spans
2. 批量构造 ablated questions
3. 批量 NLI scoring
4. 批量构造 masked questions
5. 批量 question recovery
6. 批量 recoverability scoring
7. 构建 token_labels.jsonl
8. 统计 label distribution
```

### 输出文件

```text
candidate_spans.jsonl
ablated_questions.jsonl
nli_scores.jsonl
masked_questions.jsonl
recover_outputs.jsonl
recover_scores.jsonl
token_labels.jsonl
label_distribution.json
```

### 验收标准

```text
[OK] Equivalent / Information Loss / Non-equivalent 标签分布合理
[OK] Recoverable / Non-recoverable / Misleading Recovery 能区分开
[OK] 人工抽样检查标签基本合理
[OK] 不同 span type 的标签分布有可解释差异
[OK] 能发现 NLI 与 recoverability 不一致案例
```

------

## 阶段 v2：残缺问题直接推理实验

### 目标

验证模型在缺失关键信息时是否会乱补。

### 步骤

```text
1. 使用 masked_questions.jsonl
2. 让模型直接解题
3. 要求模型在无法确定时回答 cannot determine
4. 解析模型行为
5. 构建 hallucination_risk_labels.jsonl
```

### 输出文件

```text
incomplete_reasoning_outputs.jsonl
hallucination_risk_labels.jsonl
```

### 验收标准

```text
[OK] 能识别模型是否承认信息不足
[OK] 能识别模型是否猜测缺失信息
[OK] Non-recoverable span 被 mask 时，高风险行为更多
[OK] 能展示缺信息仍强行给具体答案的案例
```

------

## 阶段 v2.5：Answer Stability / Answer Volatility

目标：  
在 masked question 直接推理实验中，进一步统计模型最终答案是否稳定。因为语义相似或偶然答对并不代表缺失信息真的可恢复，尤其是数字、条件、配置参数、漏洞编号等 span 被遮盖后，模型可能通过猜测得到正确答案，也可能在多次采样中产生明显波动。

核心流程：

```text
masked question
→ direct reasoning with K samples
→ extract final answer
→ compute answer distribution
→ assign answer stability label
```

主要输出：

```
answer_stability_scores.jsonl
```

需要统计：

```
answer_set
most_common_answer
answer_consistency
gold_answer_rate
cannot_determine_rate
symbolic_answer_rate
specific_answer_rate
guess_rate
confidence_mean
confidence_std
high_conf_wrong_rate
answer_stability_label
```

标签类型：

| 标签                    | 含义                         |
| ----------------------- | ---------------------------- |
| Answer-Stable-Correct   | 多次采样答案稳定且正确       |
| Answer-Stable-Wrong     | 多次采样答案稳定但错误       |
| Answer-Volatile         | 多次采样答案波动大           |
| Cannot-Determine-Stable | 稳定回答无法确定             |
| Symbolic-Stable         | 稳定用符号表达未知量         |
| Guessing-Volatile       | 多次猜测缺失信息，答案波动大 |

成功标准：

```
[OK] 能解析每次 direct reasoning 的 final answer
[OK] 能统计同一 masked question 下的答案分布
[OK] 能识别偶然答对但答案波动大的情况
[OK] 能识别稳定答错和高置信错误
[OK] 能区分合理的不确定性表达与无依据乱猜
```

注意：

```
答案偶然正确不能直接说明该 span 可恢复。
如果 span 被 NLI 判为 Information Loss 或 Non-equivalent，但模型仍稳定给出具体答案，需要进一步判断是否存在 shortcut。
如果 span 被 mask 后答案分布高度分散，应标记为 Answer-Volatile。
```

## 阶段 v3：Token Importance Probe

### 目标

训练一个 probe，使其在不实际 mask 的情况下预测 span 重要性。

### 输入

```text
questions.jsonl
candidate_spans.jsonl
nli_scores.jsonl
recover_scores.jsonl
token_labels.jsonl
hallucination_risk_labels.jsonl
```

### 第一版特征

```text
span_type
span_position
span_length
is_number
is_operation_word
is_relation_word
is_question_target
semantic_necessity_label
recoverability_label
```

### 后续白盒特征

```text
hidden state
attention mass
attention rank
gradient sensitivity
```

### baseline

```text
Random baseline
Token type baseline
Number-only baseline
Raw attention baseline
```

### 输出

```text
probe_features.npy
probe_labels.npy
probe_predictions.jsonl
probe_metrics.json
```

------

## 阶段 v4：Attention Guidance

### 目标

验证 span importance 是否能改善模型推理稳定性。

### 对比方法

```text
Baseline
Random Bias
Number Bias
Token Type Bias
Raw Attention Bias
Oracle Semantic Necessity Bias
Oracle Recoverability Bias
Probe-guided Bias
```

### 注入层设置

```text
early
middle
late
early_middle
middle_late
all
```

### 指标

```text
答案准确率
cannot determine 合理率
具体答案乱猜率
幻觉率
attention 是否更集中到关键 span
输出格式是否稳定
```

------

# 四、推荐脚本顺序

```text
01_prepare_data.py
02_extract_candidate_spans.py
03_build_ablated_questions.py
04_run_nli_scoring.py
05_build_masked_questions.py
06_recover_question.py
07_score_recoverability.py
08_build_token_labels.py
09_incomplete_reasoning.py
10_build_probe_features.py
11_train_probe.py
12_attention_guidance.py
13_evaluate.py
14_visualize_cases.py
```

------

# 五、关键输入输出格式

## questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "dataset": "gsm8k",
  "question": "...",
  "gold_answer": "..."
}
```

------

## candidate_spans.jsonl

```json
{
  "id": "gsm8k_0001",
  "question": "...",
  "candidates": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    }
  ]
}
```

------

## ablated_questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_question": "...",
  "ablated_question": "Tom has some apples and buys 2 more. How many apples does he have now?"
}
```

------

## nli_scores.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_to_ablated": "entailment",
  "ablated_to_original": "neutral",
  "semantic_necessity_label": "Information Loss"
}
```

------

## masked_questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "masked_span": "3",
  "span_type": "number",
  "original_question": "...",
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?"
}
```

------

## recover_outputs.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "sample_id": 0,
  "masked_question": "...",
  "recovered_question": "...",
  "recoverable": "uncertain",
  "confidence": 0.42,
  "reason": "The exact number is missing and cannot be uniquely inferred."
}
```

------

## recover_scores.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "recoverability_label": "Non-recoverable",
  "confidence_mean": 0.41,
  "recovery_consistency": 0.22,
  "misleading_recovery": false
}
```

------

## token_labels.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "semantic_necessity_label": "Information Loss",
  "recoverability_label": "Non-recoverable",
  "hallucination_risk": "High",
  "final_importance_label": "High"
}
```

------

# 六、可视化输出

## 1. Span Label 表

展示：

```text
span
span type
original question
ablated question
NLI label
masked question
recovered questions
recoverability label
final label
```

------

## 2. NLI 标签分布图

按 span type 统计：

```text
number
entity
operation
relation
comparison
negation
question_target
cyber_security_term
```

------

## 3. NLI-Recoverability 对比表

展示四类情况：

```text
NLI important + Recover non-recoverable
NLI important + Recover recoverable
NLI unimportant + Recover non-recoverable
NLI unimportant + Recover recoverable
```

------

## 4. 高风险案例表

展示：

```text
masked question
model recovery
confidence
NLI result
reasoning behavior
risk reason
```

------

## 5. Attention Guidance 前后对比图

展示：

```text
原始 attention heatmap
guidance 后 attention heatmap
关键 span 是否获得更多 attention
```

------

# 七、实验优先级

如果时间有限，先做最小闭环：

```text
1. 准备 30–50 条 GSM8K
2. 抽 candidate spans
3. 构造 ablated questions
4. 做双向 NLI
5. 构造 masked questions
6. 让模型只复原问题
7. 比较 NLI label 与 recoverability label
8. 人工检查 20–50 个案例
```

先不要急着做：

```text
probe
attention guidance
hidden states
trajectory
大规模实验
```

第一阶段只需要证明：

```text
不同 span 被删除 / 泛化 / mask 后，语义必要性和模型可恢复性存在差异；
并且 NLI semantic necessity 与 model recoverability 并不完全一致。
```

------

# 八、一句话验收标准

本实验最终要证明：

```text
通过 NLI-based semantic necessity 和 mask-recover model recoverability，可以得到可解释的 span importance 标签；这些标签能够揭示模型与人类语义必要性之间的错配，并可进一步用于 probe 训练和 attention guidance。
```