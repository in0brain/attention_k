# Sprint 2I-R: Score Matrix Decomposition and Root-Cause Audit

## 背景

Sprint 2H-C / 2H-D / 2I 已经完成 500-case diagnostic。

当前结论：

```text
1. enriched hidden features 有 instance-level fragility signal；
2. attention features 能进一步提升 classification；
3. hidden+attention 是目前最强分类器；
4. 但 ranking / budget-aware selection 仍然不稳；
5. ready_for_2000_rerun=False；
6. 不进入 Sprint 3A。
```

目前主要问题不是“完全没有信号”，而是：

```text
keyness、fragility、reasoning effect、budget priority 被混成了一个单一 score / bucket。
```

因此，本 sprint 不引入新模型、不重跑 recovery、不做 trajectory / CoT / NLA，而是先做 root-cause audit：

```text
把现有信号矩阵化，拆分 keyness / fragility / budget priority，判断当前失败到底来自公式混杂、指标错位，还是缺少真正 reasoning-level signal。
```

---

## 核心目标

本 sprint 回答以下问题：

```text
1. 当前 hidden+attention score 到底擅长预测什么？
2. keyness 是否已经能被现有信号较好区分？
3. fragility 是否已经能被现有信号较好区分？
4. ranking 失败是因为公式组合不对，还是信号本身不足？
5. 分层判断 / 矩阵组合是否能在不新增模型信号的情况下改善 top-k guidance？
6. 是否确实需要引入 reasoning-level 指标，如 answer-logprob、trajectory、CoT、NLA-lite 或 causal attribution？
```

---

## 严格边界

允许：

```text
读取 2H-B / 2H-C / 2H-D / 2I 的 500-case diagnostic 输出；
读取现有 feature dataset；
读取现有 prediction / score / gate report；
构造 score matrix；
新增 audit scripts；
模拟不同组合公式；
新增 diagnostic metrics；
新增报告和图表。
```

禁止：

```text
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要重跑 recovery；
不要生成 trajectory / CoT；
不要实现 NLA；
不要新增 causal attribution / patching；
不要把 gold solution path / fragility_bucket / risk_strength / drift / label 当作输入 feature；
不要为了通过 gate 修改已有 2H-D / 2I 的 primary method。
```

本 sprint 是 audit，不是新 detector 训练 sprint。

---

## 输入文件

优先读取：

```text
outputs/logs/sprint_2H_instance_signal_500/risk_strength_dataset.jsonl
outputs/logs/sprint_2H_feature_enrichment_500/pre_recovery_feature_dataset.jsonl
outputs/logs/sprint_2H_ordinal_calibration_500/*
outputs/logs/sprint_2I_attention_features_500/*
```

需要尽量对齐以下字段：

```text
masked_id
source_question_id
question
masked_question
span_text
span_type
fragility_bucket
risk_strength
hidden_pre_recovery_enriched scores/features
attention_pre_recovery scores/features
hidden_plus_attention_pre_recovery scores/features
surface_rule scores/features
prediction scores from 2H-D / 2I if available
```

如果某些文件不存在或字段名不同，不要直接失败；先写入 audit warning，并尽量从可用文件恢复。

---

## 任务 1：构造 Span Score Matrix

对每个 candidate span 构造矩阵化表示：

```text
M_i = evidence_source × semantic_dimension
```

其中 semantic dimensions 至少包括：

```text
keyness
fragility
causal_effect
reasoning_effect
budget_risk
```

evidence sources 至少包括：

```text
surface_rule
solution_path_diagnostic
recovery_drift_diagnostic
hidden_pre_recovery
attention_pre_recovery
hidden_plus_attention
current_priority_score
```

注意：

```text
solution_path_diagnostic、recovery_drift_diagnostic、fragility_bucket、risk_strength 只能作为 evaluation / diagnostic label，不可作为 train-time input feature。
```

输出每条 span 的矩阵记录：

```json
{
  "masked_id": "...",
  "source_question_id": "...",
  "span_text": "...",
  "span_type": "...",

  "keyness_signals": {
    "surface_keyness_proxy": null,
    "solution_path_keyness_diagnostic": null,
    "on_path_number": null,
    "off_path_number": null,
    "ambiguous_number": null
  },

  "fragility_signals": {
    "recovery_fragility_diagnostic": null,
    "hidden_fragility_score": null,
    "attention_fragility_score": null,
    "hidden_plus_attention_score": null
  },

  "reasoning_signals": {
    "answer_logprob_delta": null,
    "trajectory_change": null,
    "cot_path_change": null,
    "nla_semantic_role": null
  },

  "budget_signals": {
    "current_priority_score": null,
    "same_question_rank": null,
    "off_path_budget_risk": null
  },

  "labels_for_evaluation_only": {
    "fragility_bucket": null,
    "risk_strength": null
  }
}
```

`reasoning_signals` 目前大概率为空。不要填伪值。
如果为空，需要在报告中明确指出：

```text
当前 pipeline 缺少真正 reasoning-consequence signals。
```

---

## 任务 2：Feature / Label Leakage Audit

检查所有用于公式模拟或模型输入的字段名。

gate-eligible input feature names 禁止包含：

```text
recovered
solution_path
drift
bucket
risk_strength
gold
answer
label
target
trajectory
cot
```

说明：

```text
这些字段可以作为 diagnostic label / evaluation target；
但不能作为 formula input / probe feature。
```

输出：

```text
score_matrix_feature_audit.json
```

如果发现泄漏字段进入 input feature set，必须 fail audit。

---

## 任务 3：Keyness 单独评估

先不要评估总 bucket。
单独评估现有信号是否能判断 keyness。

### Keyness diagnostic target

优先使用：

```text
on_path_number vs off_path_number
```

如果非 number span 没有 reliable keyness label，则先只报告 number subset。

指标：

```text
on_path_vs_off_path_auc
on_path_top_k_coverage
off_path_false_positive_rate
per_question_on_path_rank
per_question_on_path_pairwise_accuracy
```

需要分别评估以下 score：

```text
surface_rule
hidden_pre_recovery
attention_pre_recovery
hidden_plus_attention
current_priority_score
```

必须回答：

```text
现有 hidden/attention signal 能否区分 on-path number 和 off-path number？
如果不能，ranking 失败可能主要来自 keyness 不稳。
```

---

## 任务 4：Fragility 单独评估

在 keyness 相对可靠的 subset 内评估 fragility。

优先 subset：

```text
on_path_number
comparison
negation
condition
```

如果样本数不足，报告 warning，不要强行下结论。

fragility diagnostic target：

```text
exact_recovery / generic_recovery / wrong_or_drifted
bucket_1 / bucket_2 / bucket_3
```

指标：

```text
wrong_vs_exact_auc
generic_or_wrong_vs_exact_auc
bucket3_vs_bucket1_auc
fragility_spearman_within_key_spans
fragility_pairwise_accuracy_within_key_spans
```

必须回答：

```text
在已经关键的 span 内部，现有信号能否区分 stable 和 fragile？
如果不能，说明缺少 fragility / reasoning consequence signal。
```

---

## 任务 5：Budget Priority 单独评估

评估当前 score 是否适合 attention guidance top-k。

指标：

```text
per_question_top1_hit
per_question_top2_hit
per_question_top3_hit
top_10pct_bucket3_precision
top_20pct_bucket3_precision
off_path_budget_share
same_question_pairwise_accuracy
```

重点看：

```text
1. high-score false positives 是否主要是 off-path numbers；
2. low-score false negatives 是否主要是 comparison / negation / condition；
3. hidden+attention 是否只是提升 classification，而没有改善 per-question top-k selection。
```

输出 error breakdown：

```text
false_positive_by_span_type
false_negative_by_span_type
false_positive_by_on_off_path
false_negative_by_bucket
```

---

## 任务 6：Top-k Failure Case Audit

抽取并输出人工检查案例。

至少输出：

```text
30 个 top-k failure cases
30 个 top-k success cases
```

每个 case 包含：

```text
question
span_text
span_type
masked_question
gold/eval-only fragility_bucket
risk_strength
on/off-path diagnostic if available
surface score
hidden score
attention score
hidden+attention score
current rank
expected rank / diagnostic priority
failure_reason_auto_guess
```

自动归因 failure_reason_auto_guess：

```text
off_path_number_overranked
comparison_or_negation_underranked
surface_type_bias
hidden_attention_large_but_irrelevant
label_ambiguous
multiple_reasonable_spans
unknown
```

输出：

```text
topk_failure_cases.jsonl
topk_success_cases.jsonl
```

---

## 任务 7：模拟分层公式

不要训练新模型，先用现有 score 模拟不同公式。

至少比较：

### Formula A: current single score

```text
priority = current_score
```

### Formula B: keyness × fragility

```text
priority = keyness_score * fragility_score
```

### Formula C: keyness gate then fragility

```text
if keyness_score < threshold:
    priority = low_priority
else:
    priority = fragility_score
```

### Formula D: keyness × fragility × attention_delta

```text
priority = keyness_score * fragility_score * attention_score
```

### Formula E: per-question normalized priority

```text
priority = zscore_within_question(raw_priority)
```

or:

```text
priority = softmax_within_question(raw_priority)
```

### Formula F: off-path penalty

```text
priority = raw_priority * (1 - off_path_probability)
```

### Formula G: span-type budget cap

```text
Within each question:
- number spans can occupy at most N slots;
- comparison / negation / condition cannot be suppressed solely by number score;
- off-path number should be downweighted if keyness evidence is weak.
```

不要把 evaluation-only diagnostic labels 直接用于 formula input。
如果需要模拟 oracle upper bound，可以单独标记为：

```text
oracle_diagnostic_only
```

不能与 gate-eligible formulas 混在一起。

---

## 任务 8：比较公式效果

对每种公式报告：

```text
Spearman
pairwise_ordering_accuracy
same_question_pairwise_accuracy
top_10pct_bucket3_precision
top_20pct_bucket3_precision
per_question_top1_hit
per_question_top2_hit
per_question_top3_hit
off_path_budget_share
bucket3_vs_bucket1_auc
```

并和以下 baseline 比较：

```text
surface_rule
hidden_pre_recovery
attention_pre_recovery
hidden_plus_attention
2H-D expected_bucket score if available
2I primary score if available
```

必须做 bootstrap 或 repeated resampling，至少对核心指标：

```text
same_question_pairwise_accuracy
top_10pct_bucket3_precision
off_path_budget_share
```

---

## 任务 9：Root-Cause Decision Table

最终报告必须给出根因判断。

请按以下结构输出：

```text
Case 1: keyness poor, fragility okay
=> 下一步应补 semantic role / NLA-lite / text parser。

Case 2: keyness okay, fragility poor
=> 下一步应补 recovery refinement / answer-logprob / causal attribution / trajectory。

Case 3: keyness okay, fragility okay, priority poor
=> 主要是 formula / per-question ranking 问题，应做分层公式和 budget policy。

Case 4: keyness poor, fragility poor
=> 当前 static signals 不够，应引入 reasoning-level signal。

Case 5: formula simulation significantly improves top-k
=> 先做 formula redesign，不要急着上 NLA / trajectory。

Case 6: formula simulation does not improve top-k
=> 说明缺少新信号，进入 NLA-lite / causal attribution / CoT trajectory 选择阶段。
```

---

## 任务 10：Reasoning Signal Gap Analysis

明确列出当前 pipeline 中哪些 reasoning-related signals 缺失。

至少检查：

```text
answer_logprob_delta
answer_rank_delta
answer_margin_delta
output_distribution_kl
calculation_path_change
cot_step_divergence
trajectory_answer_change
nla_semantic_role
causal_patch_recovery_effect
gradient_attribution_to_answer
```

输出矩阵：

```text
signal_name
currently_available
cost_to_add
expected_help_dimension
recommended_next
```

其中 `expected_help_dimension` 只能从以下集合选择：

```text
keyness
fragility
causal_effect
reasoning_effect
budget_priority
interpretability
```

---

## 输出目录

```text
outputs/logs/sprint_2I_R_score_matrix_audit_500/
```

输出文件：

```text
score_matrix_dataset.jsonl
score_matrix_feature_audit.json
keyness_eval_report.json
fragility_eval_report.json
budget_priority_eval_report.json
topk_failure_cases.jsonl
topk_success_cases.jsonl
formula_simulation_report.json
formula_bootstrap_report.json
reasoning_signal_gap_analysis.json
root_cause_decision_table.json
review_gate_score_matrix_audit.md
```

---

## Review Gate

本 sprint 不以 ready_for_2000_rerun 为主要目标。
默认：

```text
ready_for_2000_rerun=False
```

除非出现非常强的证据：

```text
1. 分层公式在 same-question ranking 上显著超过 hidden+attention；
2. top-k bucket3 precision 显著提升；
3. off-path budget share 明显下降；
4. improvement 在 bootstrap 下稳定；
5. 无任何 leakage。
```

即使公式改善，也不要直接进入 Sprint 3A。
最多建议进入：

```text
formula-redesign validation sprint
```

或：

```text
reasoning-signal sprint
```

---

## 最终报告必须回答

最终 `review_gate_score_matrix_audit.md` 必须清楚回答：

```text
1. 当前失败主要是 keyness 问题、fragility 问题、priority formula 问题，还是缺 reasoning signal？
2. hidden+attention 到底提升了哪一维？
3. attention 为什么提升 classification 但不提升 ranking？
4. off-path number 是否是主要 false positive 来源？
5. comparison / negation 是否是主要 false negative 来源？
6. 分层公式是否能改善 per-question top-k？
7. 是否需要引入 reasoning-level signal？
8. 如果需要，优先是 answer-logprob、CoT trajectory、NLA-lite，还是 causal attribution？
9. 是否建议进入 2000-case rerun？
10. 是否建议进入 Sprint 3A？
```

---

## 预期结论格式

最终请用以下格式给出明确建议：

```text
Root cause:
- ...

Evidence:
- ...

Formula simulation result:
- ...

Reasoning signal gap:
- ...

Recommendation:
- Do not scale / scale with caution / run formula validation / add reasoning signal.

Next sprint candidate:
- ...
```
