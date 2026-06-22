# 项目方案：Recoverability-Guided Attention Allocation

## 一、项目目标

本项目旨在研究：输入问题中的哪些 token / span 对问题语义、推理过程和最终答案具有不可替代作用，并进一步利用这些 token / span 的重要性信号指导模型 attention 分配，从而减少缺信息乱补、错误依赖和幻觉风险。

项目的核心思想不是简单地判断某个 token 是否重要，而是区分三类信号：

```text
1. Semantic Necessity
   从人类语义角度看，某个 span 被删除、泛化或替换后，原问题和改写问题是否仍然等价。

2. Model Recoverability
   从模型行为角度看，某个 span 被 mask 后，模型是否还能稳定复原原问题。

3. Reasoning Risk
   从推理行为角度看，当关键信息缺失时，模型是否会承认无法确定，还是会高置信乱补。
```

最终目标是：

```text
NLI semantic necessity
→ mask-recover recoverability
→ incomplete-question reasoning risk
→ token / span importance label
→ token importance probe
→ attention guidance
```

------

## 二、项目阶段

## 阶段 v0：NLI Semantic Necessity

目标：
先不调用大模型复原问题，只通过 NLI 判断某个 candidate span 被删除或泛化后，问题语义是否发生变化。

核心流程：

```text
question
→ candidate span extraction
→ ablated question construction
→ bidirectional NLI
→ semantic necessity label
```

主要输出：

```text
candidate_spans.jsonl
ablated_questions.jsonl
nli_scores.jsonl
```

成功标准：

```text
[OK] 能抽取 candidate spans
[OK] 能构造 ablated questions
[OK] 能完成双向 NLI
[OK] 能得到 Equivalent / Information Loss / Added Assumption / Non-equivalent 标签
[OK] 人工抽查标签基本合理
```

------

## 阶段 v1：Mask-Recover Model Recoverability

目标：
遮盖 candidate span，让模型只复原问题，不解题，判断模型自身能否恢复缺失信息。

核心流程：

```text
candidate span
→ masked question
→ recover original question
→ NLI / rule-based scoring
→ recoverability label
```

主要输出：

```text
masked_questions.jsonl
recover_outputs.jsonl
recover_scores.jsonl
```

成功标准：

```text
[OK] 每个 masked question 至少生成 K 次 recovered question
[OK] 能区分 Recoverable / Partially Recoverable / Non-recoverable / Misleading Recovery
[OK] 能发现 NLI semantic necessity 与 model recoverability 的一致和不一致案例
```

------

## 阶段 v2：Incomplete-Question Reasoning

目标：
直接让模型解残缺问题，观察模型在缺失关键信息时是否会乱猜。

核心流程：

```text
masked question
→ solve with uncertainty instruction
→ parse reasoning behavior
→ hallucination risk label
```

主要输出：

```text
incomplete_reasoning_outputs.jsonl
hallucination_risk_labels.jsonl
```

成功标准：

```text
[OK] 能识别 cannot_determine / symbolic_reasoning / guessed_then_solved / specific_answer_without_basis
[OK] Non-recoverable span 被 mask 时，高风险行为更多
[OK] 能输出缺信息仍强行给答案的典型案例
```

------

## 阶段 v3：Token Importance Probe

目标：
训练一个 probe，使其在不实际 mask 的情况下预测哪些 span 语义必要、难以恢复、容易引发乱补。

核心流程：

```text
token / span features
→ semantic necessity label
→ recoverability label
→ hallucination risk label
→ train probe
```

主要输出：

```text
probe_features.npy
probe_labels.npy
probe_predictions.jsonl
probe_metrics.json
```

必须比较的 baseline：

```text
Random baseline
Token type baseline
Number-only baseline
Raw attention baseline
```

成功标准：

```text
[OK] probe 优于 random baseline
[OK] probe 优于 token type baseline
[OK] probe 不只是学到“数字重要”
[OK] 能识别操作词、关系词、否定词、比较词等关键 span
```

------

## 阶段 v4：Attention Guidance

目标：
验证 recoverability-guided token importance 是否能改善模型推理稳定性，减少缺信息乱补和幻觉。

核心流程：

```text
token importance label / probe score
→ attention bias
→ generation / reasoning
→ compare with baseline
```

需要比较：

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

层注入需要做 sweep：

```text
early
middle
late
early_middle
middle_late
all
```

成功标准：

```text
[OK] Oracle Guidance 优于 Random Bias
[OK] Probe-guided Guidance 优于 Random Bias
[OK] guidance 后 attention 更集中到关键 span
[OK] 不明显破坏原模型正常解题能力
[OK] 部分高风险样本中，模型更愿意承认信息不足
```

------

# 三、推荐项目结构

```text
recover_attention_project/
  README.md
  AGENTS.md
  PROGRESS.md
  requirements.txt
  pyproject.toml

  docs/
    method.md
    experiment_guide.md
    label_schema.md
    prompts.md
    codex_tasks.md

  configs/
    v0_nli_small.yaml
    v1_recover.yaml
    v2_incomplete_reasoning.yaml
    v3_probe.yaml
    v4_attention_guidance.yaml

  data/
    raw/
      gsm8k.jsonl
      math500.jsonl
      secqa.jsonl
      ctibench.jsonl
      cybermetric.jsonl

    processed/
      questions.jsonl
      candidate_spans.jsonl
      ablated_questions.jsonl
      nli_scores.jsonl
      masked_questions.jsonl
      recover_outputs.jsonl
      recover_scores.jsonl
      token_labels.jsonl
      incomplete_reasoning_outputs.jsonl
      hallucination_risk_labels.jsonl
      answer_stability_scores.jsonl

  cache/
    models/
    generations/
    nli_cache/
    hidden_states/
    attentions/

  features/
    semantic_necessity_features.npy
    recoverability_features.npy
    probe_features.npy
    probe_labels.npy
    attention_features.npy

  outputs/
    v0_nli_analysis/
    v1_recover_analysis/
    v2_incomplete_reasoning/
    v3_probe_results/
    v4_attention_guidance/
    figures/
    logs/

  src/
    recover_attention/
      __init__.py
      data_io.py
      schemas.py
      candidate_extraction.py
      question_ablations.py
      nli_scoring.py
      recover_generation.py
      recover_scoring.py
      incomplete_reasoning.py
      label_builder.py
      feature_builder.py
      probe_models.py
      attention_guidance.py
      evaluation.py
      visualization.py
      llm_backends.py
      utils.py
      answer_stability.py

  scripts/
    01_prepare_data.py
    02_extract_candidate_spans.py
    03_build_ablated_questions.py
    04_run_nli_scoring.py
    05_build_masked_questions.py
    06_recover_question.py
    07_score_recoverability.py
    08_build_token_labels.py
    09_incomplete_reasoning.py
    10_score_answer_stability.py
    11_build_probe_features.py
    12_train_probe.py
    13_attention_guidance.py
    14_evaluate.py
    15_visualize_cases.py

  tests/
    test_candidate_extraction.py
    test_question_ablations.py
    test_nli_scoring.py
    test_label_builder.py
```

------

# 四、关键文件说明

## README.md

说明项目目标、环境安装、最小运行命令和数据格式。

建议包含：

```text
项目简介
环境安装
v0 最小实验命令
输入输出说明
当前实验阶段
```

------

## AGENTS.md

给 Codex 使用的项目规则文件。

建议写入：

```text
1. 所有输出统一使用 jsonl。
2. 所有脚本必须支持 --config 或显式输入输出路径。
3. 每次只完成一个阶段，不要一次性实现全部功能。
4. v0 阶段不要实现 probe、attention guidance、hidden states。
5. 新增脚本后必须给出运行命令和检查命令。
6. 修改完代码后必须更新 PROGRESS.md。
```

------

## PROGRESS.md

给 Codex 看的实验进度文件。

固定格式：

```text
# 实验进度记录

## 1. 现有进度

### (1) xxx
已完成内容：
输入：
输出：
运行命令：
检查结果：
遗留问题：

## 2. 需要下一步干的事项

### (1) xxx
目标：
需要修改的文件：
输入：
输出：
实现要求：
验收标准：
运行命令：
禁止事项：
```

规则：

```text
下一步事项完成后，必须移动到“现有进度”。
每次只保留 1–3 个下一步事项。
```

------

## docs/method.md

写方法定义和后续公式。

内容包括：

```text
NLI-based semantic necessity
Mask-recover model recoverability
Incomplete-question reasoning risk
Token / span label taxonomy
Probe training
Attention guidance
```

------

## docs/experiment_guide.md

写实验指导书。

主线必须是：

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity
→ masked question construction
→ model recoverability
→ incomplete-question reasoning
→ token label integration
→ probe
→ attention guidance
```

------

## docs/label_schema.md

定义标签体系。

建议包括：

```text
Semantic Necessity Label:
- Equivalent
- Information Loss
- Added Assumption
- Non-equivalent

Recoverability Label:
- Recoverable
- Partially Recoverable
- Non-recoverable
- Misleading Recovery

Reasoning Behavior Label:
- cannot_determine
- symbolic_reasoning
- guessed_then_solved
- specific_answer_without_basis
- wrong_recovery_reasoning

Hallucination Risk Label:
- Low
- Medium
- High

Final Span Importance:
- Low
- Medium
- High
- Risky
```

------

## docs/prompts.md

集中存放 prompt。

包括：

```text
question recovery prompt
incomplete reasoning prompt
LLM judge prompt
NLI fallback prompt
structured output format
```

------

## docs/codex_tasks.md

给 Codex 拆分任务。

示例：

```text
Task 1: 实现数据准备脚本
Task 2: 实现 candidate span 抽取
Task 3: 实现 ablated question 构造
Task 4: 实现双向 NLI scoring
Task 5: 实现 masked question 构造
Task 6: 实现 question recovery
Task 7: 实现 recoverability scoring
```

------

# 五、核心数据格式

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

说明：
统一使用 `span`，不要使用 `token`，因为很多重要成分是短语，例如：

```text
how many
at least
more than
in total
remote code execution
privilege escalation
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
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
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
  "reason": "The exact number is missing."
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

## answer_stability_scores.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "masked_span": "3",
  "span_type": "number",
  "masked_question": "...",
  "gold_answer": "5",
  "num_samples": 10,
  "answers": ["5", "6", "cannot determine", "x + 2", "5"],
  "answer_set": ["5", "6", "cannot determine", "x + 2"],
  "most_common_answer": "5",
  "answer_consistency": 0.4,
  "gold_answer_rate": 0.2,
  "cannot_determine_rate": 0.2,
  "symbolic_answer_rate": 0.1,
  "specific_answer_rate": 0.7,
  "guess_rate": 0.5,
  "confidence_mean": 0.73,
  "confidence_std": 0.18,
  "high_conf_wrong_rate": 0.3,
  "answer_stability_label": "Answer-Volatile",
  "risk_label": "High"
}
```

---

# 六、最小实验顺序

v0 最小实验只做：

```text
1. 准备 30–50 条数据
2. 抽取 candidate spans
3. 构造 ablated questions
4. 做双向 NLI
5. 得到 semantic necessity label
6. 构造 masked questions
7. 做 question recovery
8. 得到 recoverability label
9. 比较 NLI label 与 recoverability label
10. 输出 10 个高质量案例
```

v0 不做：

```text
probe
attention guidance
hidden states
trajectory analysis
大规模实验
```

------

# 七、当前研究价值判断

前期重点不是证明 attention guidance 一定有效，而是先证明：

```text
1. NLI 能稳定区分语义必要 span。
2. 模型 recoverability 与 NLI semantic necessity 不完全一致。
3. 这种不一致能暴露模型的错误依赖或乱补风险。
```

典型有价值现象：

```text
NLI 认为重要，但模型高置信乱补
→ 潜在幻觉风险。

NLI 认为重要，但 raw attention 不高
→ 模型忽略关键 span。

NLI 认为不重要，但模型恢复困难
→ 模型对非关键 span 过度敏感。

NLI 认为不重要，但 raw attention 很高
→ 异常 attention sink。
```

------

# 八、一句话总结

本项目以 NLI 作为语义必要性的基础标签，以 mask-recover 作为模型可恢复性的行为测试，再用二者的差异发现模型关注与人类语义必要性之间的错配，最终训练 probe 并尝试 attention guidance。