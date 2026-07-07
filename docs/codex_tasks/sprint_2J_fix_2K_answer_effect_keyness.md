# Sprint 2J-Fix + 2K: Slot Alignment Repair and Leakage-Safe Answer-Effect Keyness Signal

## 背景

Sprint 2J 已完成 multi-span-per-question reasoning matrix 的代码框架，并首次把评估对象从：

```text
one question -> one scored span
```

推进到：

```text
one question -> multiple candidate spans
```

这是正确方向。

但 2J-B 暴露出两个必须先修复的问题：

```text
1. original forward cache 可能错误复用了 first-span slot_indices；
2. review gate 的 grouped bootstrap 检查过弱，当前 check #7 只是检查 bootstrap 是否存在，而不是判断 improvement 是否稳定。
```

同时，2J 的核心诊断也更清楚了：

```text
hidden / attention static perturbation signals do not provide reliable within-question keyness ranking.
```

因此，本 sprint 分为两部分：

```text
2J-Fix: 修复 multi-span scoring 的 slot alignment 与 gate/report 问题；
2K: 增加 leakage-safe answer-effect / self-output-distribution shift signal，测试 keyness 是否能从模型自身输出变化中恢复。
```

本 sprint 不做 CoT，不做 trajectory，不做 NLA，不做 causal patching，不进入 2000，不进入 Sprint 3A。

---

## 核心目标

本 sprint 回答：

```text
1. 修复 original slot alignment 后，hidden/attention 的 within-question on/off-path AUC 是否仍然低于 surface？
2. 2J-B 的 same-question ranking 结论是否稳定？
3. grouped bootstrap 是否支持任何非 oracle 公式优于 surface-only？
4. answer-effect signal 是否能提供真正的 within-question keyness signal？
5. self-output distribution shift 是否能把 on-path number 排在 off-path number 前面？
6. 是否可以用 answer-effect 为 non-number spans 建立 model-causal keyness diagnostic？
```

---

## 严格边界

允许：

```text
读取 2J-A multi-span matrix；
修复 2J-B scoring 代码；
重跑 2J-B hidden/attention scoring；
在同一次 original/masked forward 中保存 logits-derived answer-effect features；
构造 leakage-safe self-answer/self-distribution features；
构造 diagnostic-only gold-answer features；
评估 same-question ranking；
做 grouped bootstrap；
更新 review reports；
新增 tests。
```

禁止：

```text
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要重跑 recovery；
不要生成 CoT；
不要生成 trajectory；
不要实现 NLA；
不要实现 activation patching / causal attribution；
不要把 gold answer / solution_path / fragility_bucket / risk_strength / recovery drift 当作 gate-eligible input；
不要把 gold-answer logprob delta 用作 formula input；
不要用 oracle label 直接决定 priority score；
不要只因为 top-k coverage 高就宣布通过。
```

默认：

```text
ready_for_2000_rerun=False
do_not_enter_sprint_3A=True
```

---

# Phase 1: 2J-Fix — Slot Alignment Repair

## 问题说明

当前 2J-B 中 original forward 按 question 缓存是合理的，但 original `slot_indices` 不能随 forward result 一起复用。

错误风险：

```text
question q 有多个 spans: s1, s2, s3
original forward 第一次处理 s1 时计算 slot_indices_s1
后续处理 s2/s3 时复用 original result
=> original["slot_indices"] 仍然可能是 s1 的 token indices
```

这会导致：

```text
hidden span pooling 错位；
attention slot mass 错位；
context-to-slot attention 错位；
within-question hidden/attention ranking 不可信。
```

---

## 任务 1：修复 original forward cache 结构

把 original cache 拆成两层：

```text
question-level cached data:
- hidden states
- attention maps
- offsets
- text
- resolved_hidden_state_indices

span-level recomputed data:
- original_slot_indices for current span char range
```

不要缓存绑定某个 span 的 `slot_indices` 作为 question-level reusable state。

建议结构：

```python
OriginalForwardCache = {
    "text": question,
    "offsets": offsets,
    "hidden": hidden_tensor,
    "attention": attention_stack,
    "resolved_hidden_state_indices": resolved_hidden_state_indices,
    "warnings": warnings_without_slot_specific_alignment
}
```

每个 span 单独计算：

```python
original_slot_indices = token_indices_for_char_ranges(
    original_cache["offsets"],
    [[span_char_start, span_char_end]],
    exclude=set(),
)
```

masked forward 可以继续为当前 mask span 计算：

```python
masked_slot_indices = token_indices_for_char_ranges(
    masked_offsets,
    [mask_char_range],
    exclude=set(),
)
```

---

## 任务 2：修改 build_feature_record 接口

当前 `build_feature_record(record, original, masked, layer_indices)` 隐含使用：

```text
original["slot_indices"]
masked["slot_indices"]
```

修复后应显式传入：

```text
original_slot_indices
masked_slot_indices
```

建议接口：

```python
build_feature_record(
    flat_record,
    original_cache,
    masked_result,
    original_slot_indices=...,
    masked_slot_indices=...,
    layer_indices=...
)
```

并在输出 alignment 中记录：

```json
{
  "num_original_slot_tokens": ...,
  "num_masked_slot_tokens": ...,
  "original_slot_token_indices": [...],
  "mask_token_indices": [...],
  "slot_alignment_source": "per_span_recomputed"
}
```

---

## 任务 3：新增 slot alignment tests

必须新增测试覆盖：

```text
同一道题多个 span 的 original_slot_token_indices 不应相同，除非它们确实重叠。
```

测试样例：

```text
question = "Alice bought 3 apples for 2 dollars each."
span A = "3"
span B = "2"
```

断言：

```python
slot_indices_A != slot_indices_B
```

再加一个 phrase + atomic number 测试：

```text
span A = "2"
span B = "2 dollars each"
```

断言：

```python
overlap_group_id exists
slot indices can overlap but must be explicitly explained by overlap relation
```

---

## 任务 4：重跑 2J-B scoring

修复后重跑：

```text
scripts/sprint_2J_multi_span_scoring.py
```

输出目录建议：

```text
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/
```

不要覆盖原始 2J-B 输出。保留修复前后对比。

---

# Phase 2: Gate / Report Fix

## 任务 5：AUC-first report

修改 review report，把以下指标放到主位：

```text
same_question_on_path_vs_off_path_auc
on_path_number_rank_mean
off_path_number_rank_mean
same_question_pairwise_accuracy
```

降低以下指标的叙事权重：

```text
per_question_topk_on_path_coverage
per_question_topk_key_hit
```

原因：

```text
GSM8K 中 on-path numbers base rate 高，coverage 容易被 number flooding 伪造；
on/off-path pairwise AUC 更能检验 true within-question keyness ranking。
```

最终报告必须明确写：

```text
Coverage is base-rate-sensitive and cannot be used alone as evidence of keyness ranking.
```

---

## 任务 6：修复 grouped bootstrap gate

当前 gate #7 不能只检查 `bootstrap_delta_ci95 is not None`。

必须改成真实稳定性判断。

对每个 gate-eligible formula，做 question-grouped bootstrap：

```text
sample source_question_id with replacement
retain all spans within sampled question
compute same_question_on_path_vs_off_path_auc delta against A_surface_only
repeat N=1000
report mean_delta, ci95_low, ci95_high
```

Gate #7 通过条件：

```text
best_non_oracle_formula bootstrap ci95_low > 0
```

如果目标是 off-path budget share，则方向相反：

```text
off_path_budget_share_delta ci95_high < 0
```

不要再使用 placeholder 文案。

输出：

```text
grouped_bootstrap_report.json
```

每个 formula 至少包含：

```json
{
  "formula": "...",
  "baseline": "A_surface_only",
  "metric": "same_question_on_path_vs_off_path_auc",
  "mean_delta": ...,
  "ci95_low": ...,
  "ci95_high": ...,
  "num_questions_with_on_off_pairs": ...,
  "num_bootstrap_samples": 1000,
  "stable_positive": true
}
```

---

## 任务 7：修复 review gate

2J-Fix gate 应至少包含：

```text
1. same-question ranking computable；
2. original slot alignment per-span recomputation passed；
3. feature leakage audit passed；
4. same_question_on_path_vs_off_path_auc reported for all formulas；
5. grouped bootstrap computed for all gate-eligible formulas；
6. best_non_oracle_formula has ci95_low > 0 against surface, or gate fails；
7. ready_for_2000_rerun remains false；
8. do_not_enter_sprint_3A remains true。
```

---

# Phase 3: 2K — Leakage-Safe Answer-Effect Signal

## 核心思想

Hidden/attention 是 perturbation signal：

```text
mask span 后内部状态 / attention 变化多大？
```

但 keyness 更接近 answer-effect：

```text
mask span 后，模型自己的答案分布是否改变？
```

本阶段新增：

```text
self-output-distribution shift under masking
```

作为 gate-eligible keyness / causal-effect proxy。

---

## 任务 8：保存 logits-derived features

在 original/masked forward 中保存 logits-derived summary。

如果当前 2J-B artifacts 未保存 logits，则允许重跑 original/masked forward，但必须复用 2J-A multi-span matrix，不扩大数据。

对每个 span 保存：

```text
original_last_token_logits_topk
masked_last_token_logits_topk
original_last_token_entropy
masked_last_token_entropy
original_top1_token_id
masked_top1_token_id
original_top1_token_logprob
masked_top1_token_logprob
original_topk_token_ids
masked_topk_token_ids
```

注意：

```text
不要保存完整 vocab logits 到 JSONL；
可以保存 top-k 和 summary statistics；
如需完整 logits，保存到 .pt/.npy 并在 manifest 中记录路径。
```

---

## 任务 9：构造 gate-eligible self-answer-effect features

禁止使用 gold answer。

允许使用模型自己的分布变化：

```text
self_last_token_kl_original_to_masked
self_last_token_js_divergence
self_top1_token_changed
self_topk_overlap
self_entropy_delta
self_top1_logprob_delta
self_margin_delta
```

如果实现轻量 answer generation，则新增：

```text
original_greedy_answer_text
masked_greedy_answer_text
self_answer_string_changed
self_numeric_answer_changed
self_answer_edit_distance
```

但 answer generation 不是必须。优先做 logits-level answer-effect。

所有 gate-eligible feature 名必须避免：

```text
gold
answer
label
target
solution_path
risk_strength
bucket
drift
recovered
oracle
```

由于 `answer` 是 banned substring，gate-eligible 字段建议命名为：

```text
self_output_kl
self_output_js
self_output_top1_changed
self_output_topk_overlap
self_output_entropy_delta
self_output_margin_delta
self_output_logprob_delta
```

不要命名为 `answer_logprob_delta`。

---

## 任务 10：diagnostic-only gold-answer effect

可以构造 gold diagnostic，但必须隔离。

允许字段：

```text
diagnostic_gold_final_answer_logprob_delta
diagnostic_gold_answer_rank_delta
diagnostic_gold_correctness_change
```

这些只能出现在：

```text
diagnostic_labels_for_eval_only
oracle_reports
upper_bound_analysis
```

禁止进入：

```text
formula input
probe feature
gate-eligible score
priority score
```

如果没有可靠 gold answer tokenization，跳过 gold diagnostic，不要强行实现。

---

## 任务 11：把 answer-effect 加入 score matrix

新增 score matrix namespace：

```json
{
  "answer_effect_signals": {
    "self_output_kl": null,
    "self_output_js": null,
    "self_output_top1_changed": null,
    "self_output_topk_overlap": null,
    "self_output_entropy_delta": null,
    "self_output_margin_delta": null
  }
}
```

或者为了避免 `answer` substring：

```json
{
  "output_effect_signals": {
    "self_output_kl": null,
    "self_output_js": null,
    "self_output_top1_changed": null,
    "self_output_topk_overlap": null,
    "self_output_entropy_delta": null,
    "self_output_margin_delta": null
  }
}
```

推荐使用 `output_effect_signals`。

---

## 任务 12：新增公式

在原有公式基础上增加：

### J. Output-effect only

```text
priority = self_output_shift_score
```

### K. Keyness × output-effect

```text
priority = keyness_score * self_output_shift_score
```

### L. Output-effect gate then fragility

```text
if self_output_shift_score < threshold:
    priority = low
else:
    priority = hidden_plus_attention_fragility
```

### M. Output-effect × fragility

```text
priority = self_output_shift_score * hidden_plus_attention_fragility
```

### N. Keyness × output-effect × fragility

```text
priority = keyness_score * self_output_shift_score * hidden_plus_attention_fragility
```

所有 J–N 都必须是 gate-eligible 且不使用 gold labels。

---

## 任务 13：2K evaluation

主指标：

```text
same_question_on_path_vs_off_path_auc
on_path_number_rank_mean
off_path_number_rank_mean
same_question_pairwise_accuracy
grouped_bootstrap_delta_vs_surface
grouped_bootstrap_delta_vs_hidden_plus_attention
```

辅助指标：

```text
per_question_top1_key_hit
per_question_top2_key_hit
per_question_top3_key_hit
off_path_budget_share
span_type_budget_distribution
comparison_topk_coverage
negation_topk_coverage
condition_topk_coverage
operation_topk_coverage
question_target_topk_coverage
```

必须单独报告：

```text
number subset:
- on-path vs off-path AUC
- on-path mean rank
- off-path mean rank

non-number subset:
- comparison / negation / condition / operation output-effect distribution
- output-effect top-k selection rate
```

---

## 任务 14：non-number model-causal keyness diagnostic

用 self-output effect 为非数字 span 构造 diagnostic：

```text
model_causal_keyness_high:
    self_output_shift_score >= threshold_high

model_causal_keyness_medium:
    threshold_low <= self_output_shift_score < threshold_high

model_causal_keyness_low:
    self_output_shift_score < threshold_low
```

这不是 gold label，而是 model-causal diagnostic。

报告每类 span 的分布：

```text
comparison
negation
condition
operation
question_target
object
```

目标是回答：

```text
non-number spans 是否能通过 output-effect 获得比 weak semantic keyness 更可靠的 keyness signal？
```

---

# 输出目录

2J-Fix 输出：

```text
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/
```

必须输出：

```text
multi_span_feature_matrix.jsonl
multi_span_score_matrix.jsonl
hidden_attention_feature_report.json
same_question_ranking_report.json
formula_validation_report.json
topk_budget_report.json
grouped_bootstrap_report.json
slot_alignment_audit.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_multi_span_scoring_fixed.md
```

2K 输出：

```text
outputs/logs/sprint_2K_answer_effect_keyness_500/
```

必须输出：

```text
output_effect_feature_matrix.jsonl
output_effect_score_matrix.jsonl
output_effect_feature_audit.json
same_question_output_effect_ranking_report.json
output_effect_formula_validation_report.json
output_effect_grouped_bootstrap_report.json
output_effect_topk_budget_report.json
non_number_model_causal_keyness_report.json
diagnostic_gold_effect_report.json
reasoning_signal_gap_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_answer_effect_keyness.md
```

---

# 2J-Fix Gate

通过条件：

```text
1. original slot alignment per-span recomputation verified by tests；
2. same-question ranking computable；
3. feature leakage audit passed；
4. formula input leakage audit passed；
5. same_question_on_path_vs_off_path_auc is reported as primary metric；
6. grouped bootstrap is real and checks CI direction；
7. no non-oracle formula is declared stable unless ci95_low > 0 vs surface；
8. ready_for_2000_rerun=False；
9. do_not_enter_sprint_3A=True。
```

如果 hidden/attention AUC 仍低于 0.5 after fix，报告应明确写：

```text
hidden/attention perturbation signals are anti-aligned with within-question keyness ranking on this dataset.
```

如果修复后 AUC 回到 0.5 左右，则写：

```text
previous anti-keyness effect was likely inflated by slot alignment error.
```

---

# 2K Gate

2K 不以 2000-ready 为目标。
默认：

```text
ready_for_2000_rerun=False
do_not_enter_sprint_3A=True
```

2K 成功条件：

```text
1. self-output-effect feature leakage audit passed；
2. gold-answer diagnostic features are isolated and not gate-eligible；
3. output-effect-only formula beats surface on same_question_on_path_vs_off_path_auc；
4. grouped bootstrap ci95_low > 0 vs surface；
5. output-effect formula improves or does not worsen off_path_budget_share；
6. output-effect provides useful non-number model-causal keyness diagnostic；
7. failure cases show interpretable relation between high output-effect and reasoning-critical spans。
```

如果 2K 通过，不要直接进入 Sprint 3A。
最多建议：

```text
formula validation sprint with output-effect
```

如果 2K 不通过，建议：

```text
CoT / trajectory / NLA-lite / richer dataset decision sprint
```

---

# 最终报告必须回答

`review_gate_answer_effect_keyness.md` 必须回答：

```text
1. 修复 slot alignment 后，hidden/attention AUC 是否变化？
2. surface-only 的 on/off-path AUC 是多少？
3. hidden-only / attention-only / hidden+attention 的 on/off-path AUC 是多少？
4. output-effect-only 的 on/off-path AUC 是多少？
5. keyness × output-effect 是否优于 keyness × fragility？
6. output-effect × fragility 是否优于 hidden+attention？
7. grouped bootstrap 是否支持 output-effect 改善？
8. off-path number 是否仍然被高排？
9. comparison / negation / condition 是否获得更合理的 keyness signal？
10. 是否仍然缺 CoT / trajectory / NLA？
11. 是否建议扩大到 2000？
12. 是否建议进入 Sprint 3A？
```

---

# 推荐结论格式

最终报告请使用：

```text
Verdict:
- ...

Slot alignment:
- ...

Primary AUC results:
- ...

Grouped bootstrap:
- ...

Output-effect results:
- ...

Non-number keyness:
- ...

Leakage audit:
- ...

Failure cases:
- ...

Root cause:
- ...

Recommendation:
- ...

Next sprint candidate:
- ...
```

---

## 关键提醒

本 sprint 的核心不是证明 hidden/attention 已经可用，而是验证：

```text
Can keyness be recovered from the model's own output change under span masking?
```

如果答案是 yes，后续 attention guidance 才有真正的第一层 keyness gate。
如果答案是 no，再考虑 CoT、trajectory、NLA-lite 或更适合 comparison/negation/condition 的数据集。
