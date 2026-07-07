# Sprint 2K-V: Signal Role Decomposition and Fusion Error Audit

## 背景

Sprint 2J-Fix + 2K 已完成。

当前关键结论：

```text id="yu76tx"
1. 2J-B 原始 hidden/attention anti-keyness 结论是 slot alignment bug artifact；
2. 修复后 attention-only 在 same-question on/off-path AUC 上显著优于 surface；
3. hidden-only 仍然弱；
4. hidden+attention 简单融合低于 attention-only；
5. output-effect last-token signal 单独不稳，但与 fragility 组合后有稳定提升。
```

当前已知结果：

```text id="peh06t"
surface = 0.512
hidden_only = 0.468
attention_only = 0.588
hidden+attention = 0.564
oracle = 0.9995

J_output_effect_only ≈ 0.554, not stable
L/M/N output-effect + fragility ≈ 0.58, stable
```

这说明：

```text id="xzcc94"
attention 可能主要提供 keyness signal；
hidden 可能主要提供 fragility signal；
output-effect 可能提供 causal/keyness 补充；
simple average 会混淆 keyness 与 fragility，因此拉低 attention-only。
```

本 sprint 的目标不是继续调权重，而是做 signal role decomposition：

```text id="n6rrxf"
拆解 attention / hidden / output-effect 分别在解决哪个子问题。
```

---

## 核心目标

本 sprint 回答以下问题：

```text id="lgz83f"
1. hidden+attention 为什么低于 attention-only？
2. hidden 是真的没用，还是只是不适合做 keyness ranking？
3. attention 是否主要负责 keyness？
4. hidden 是否主要负责 fragility？
5. output-effect 是否能增强 attention 的 keyness signal？
6. simple average 的错误主要发生在哪些 span type / question type？
7. 更合理的分层公式是否优于 simple fusion？
8. 当前结果是否支持进入 formula validation sprint？
```

---

## 严格边界

允许：

```text id="xc45pd"
读取 2J-Fix / 2K 已有 feature matrix、score matrix、ranking report、bootstrap report；
构造 signal role decomposition reports；
做 attention/hidden/output-effect 分别评估；
做 conflict pair audit；
做 formula simulation；
做 grouped bootstrap；
输出 failure/success cases；
新增 tests；
更新 PROGRESS.md 和 sprint_2_history.md。
```

禁止：

```text id="llhal8"
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要重跑 recovery；
不要生成 CoT；
不要生成 trajectory；
不要实现 NLA；
不要实现 causal patching；
不要联网下载模型；
不要把 gold answer / solution_path / fragility_bucket / risk_strength / recovery drift 当作 gate-eligible input；
不要把 oracle diagnostic upper bound 混入 formula comparison；
不要用 top-k coverage 替代 same-question AUC 作为主指标；
不要继续用 simple average 作为默认最终融合。
```

默认：

```text id="qu6i7r"
ready_for_2000_rerun=False
do_not_enter_sprint_3A=True
```

---

# Phase 0: Inputs and Output Directory

## 输入

优先读取：

```text id="wilbl8"
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_score_matrix.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/same_question_ranking_report.json
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/grouped_bootstrap_report.json

outputs/logs/sprint_2K_answer_effect_keyness_500/same_question_output_effect_ranking_report.json
outputs/logs/sprint_2K_answer_effect_keyness_500/output_effect_grouped_bootstrap_report.json
outputs/logs/sprint_2K_answer_effect_keyness_500/output_effect_feature_audit.json
outputs/logs/sprint_2K_answer_effect_keyness_500/non_number_model_causal_keyness_report.json
```

如果某些 reports 没有落盘，但 `multi_span_feature_matrix.jsonl` 存在，可以从 feature matrix 重新构造 score matrix 和 formula scores。

不要依赖 PROGRESS.md 中的数字作为唯一数据源。
PROGRESS.md 只能作为 sanity check。

---

## 输出目录

```text id="fw6njo"
outputs/logs/sprint_2K_V_signal_role_decomposition_500/
```

必须输出：

```text id="phjmmr"
signal_role_summary.json
keyness_signal_eval_report.json
fragility_signal_eval_report.json
attention_hidden_conflict_report.json
output_effect_role_report.json
fusion_error_cases.jsonl
fusion_success_cases.jsonl
gated_formula_validation_report.json
gated_formula_bootstrap_report.json
span_type_error_breakdown.json
question_level_case_audit.jsonl
review_gate_signal_role_decomposition.md
```

---

# Phase 1: Reconstruct Core Signals

## 任务 1：构造统一 signal table

为每个 span 构造一行统一记录：

```json id="f0b384"
{
  "source_question_id": "...",
  "span_id": "...",
  "span_text": "...",
  "span_type": "...",

  "labels_for_eval_only": {
    "solution_path_status": "...",
    "weak_semantic_keyness": "...",
    "fragility_bucket_if_available": null,
    "risk_strength_if_available": null
  },

  "signals": {
    "surface_keyness": null,
    "hidden_fragility": null,
    "attention_keyness": null,
    "hidden_plus_attention_simple": null,
    "output_effect": null
  },

  "within_question_ranks": {
    "surface": null,
    "hidden": null,
    "attention": null,
    "hidden_plus_attention": null,
    "output_effect": null
  }
}
```

注意命名：

```text id="w8e2ww"
不要把 gate-eligible feature 命名为 answer / gold / label / target / solution_path 等 forbidden substring。
```

`solution_path_status` 等只保留在 `labels_for_eval_only`。

---

## 任务 2：定义基础信号

至少构造以下基础信号：

### A. surface_keyness

来自：

```text id="nf69cr"
surface_features.surface_keyness_proxy
semantic_keyness_proxy
```

用途：

```text id="x3kmiu"
弱语义先验 baseline。
```

### B. hidden_fragility

来自 hidden features：

```text id="xrndm1"
hidden_delta_relative_norm
question_context_shift_norm
early_mid_late_delta_slope
span_to_mask_similarity
cross_layer_stability
```

用途：

```text id="fv6jx9"
衡量 mask span 后内部状态是否大幅扰动。
```

### C. attention_keyness

来自 attention features：

```text id="hldgtb"
span_attention_in_mass
context_to_slot_attention
operation_to_slot_attention
qfocus_to_slot_attention
number_context_to_slot_attention
original_to_masked_attention_delta
attention_entropy_delta
attention_rank_delta
```

用途：

```text id="ou8312"
衡量该 span 是否在 attention 结构中处于 reasoning-relevant 位置。
```

### D. output_effect

来自 2K output-effect features：

```text id="u4vtmf"
self_output_kl
self_output_js
self_output_top1_changed
self_output_topk_overlap
self_output_entropy_delta
self_output_margin_delta
self_output_logprob_delta
```

用途：

```text id="v65ib8"
衡量 mask span 后模型自身输出分布是否变化。
```

### E. simple_hidden_attention_fusion

当前简单融合 baseline：

```text id="yua7y4"
simple_fusion = mean(hidden_fragility, attention_keyness)
```

用途：

```text id="m4hepf"
验证 simple average 是否拉低 attention-only。
```

---

# Phase 2: Keyness Signal Evaluation

## 任务 3：以 same-question on/off-path AUC 评估 keyness

对以下信号分别计算：

```text id="w1pito"
surface_keyness
hidden_fragility
attention_keyness
simple_hidden_attention_fusion
output_effect
attention_times_output_effect
attention_gate_output_effect
```

主指标：

```text id="r7xvky"
same_question_on_path_vs_off_path_auc
same_question_pairwise_accuracy
on_path_number_rank_mean
off_path_number_rank_mean
num_questions_with_on_off_pairs
num_on_off_pairs
```

所有结果必须按 question-grouped bootstrap 与 surface 比较：

```text id="ye27em"
delta_vs_surface
ci95_low
ci95_high
stable_positive = ci95_low > 0
```

输出：

```text id="jhwc1c"
keyness_signal_eval_report.json
```

报告必须明确回答：

```text id="ppsube"
1. attention 是否稳定优于 surface？
2. hidden 是否低于或接近随机？
3. simple hidden+attention 是否低于 attention-only？
4. output-effect 是否单独稳定？
5. attention × output-effect 是否优于 attention-only？
```

---

# Phase 3: Fragility Signal Evaluation

## 任务 4：单独评估 hidden 是否适合 fragility

不要只用 keyness AUC 判断 hidden。

对 hidden / attention / output-effect 分别评估 fragility 相关指标。

如果 `fragility_bucket_if_available` 可用，计算：

```text id="d8zqzk"
bucket3_vs_bucket1_auc
bucket3_vs_bucket0_auc
fragility_bucket_spearman
risk_strength_spearman
bucket3_topk_precision
bucket3_topk_recall
```

如果 recovery drift labels 可用，计算：

```text id="k7c351"
wrong_vs_exact_auc
generic_or_wrong_vs_exact_auc
drifted_vs_stable_auc
```

对比信号：

```text id="52hjvh"
hidden_fragility
attention_keyness
output_effect
attention_times_output_effect
simple_hidden_attention_fusion
```

输出：

```text id="f9u5la"
fragility_signal_eval_report.json
```

报告必须明确判断：

```text id="snu1ks"
hidden 是否虽然不适合 keyness，但适合 fragility？
```

如果 hidden 在 fragility 指标上明显强于 attention/output-effect，则结论应写：

```text id="ofqi02"
hidden should be used as a second-stage fragility/risk signal, not as a first-stage keyness ranker.
```

如果 hidden 在 fragility 上也弱，则结论应写：

```text id="xrvf77"
hidden signal is currently not useful for either keyness or fragility under this feature construction.
```

---

# Phase 4: Attention-Hidden Conflict Audit

## 任务 5：构造 pair-level conflict table

对每一道题中的所有 on-path number vs off-path number pair，记录：

```json id="u1qojh"
{
  "source_question_id": "...",
  "on_span_id": "...",
  "off_span_id": "...",
  "on_span_text": "...",
  "off_span_text": "...",

  "attention_pair_result": "correct|wrong|tie",
  "hidden_pair_result": "correct|wrong|tie",
  "simple_fusion_pair_result": "correct|wrong|tie",
  "output_effect_pair_result": "correct|wrong|tie",

  "attention_score_on": 0.0,
  "attention_score_off": 0.0,
  "hidden_score_on": 0.0,
  "hidden_score_off": 0.0,
  "fusion_score_on": 0.0,
  "fusion_score_off": 0.0
}
```

---

## 任务 6：统计 conflict 类型

至少统计以下类型：

### Type A: attention_correct_hidden_hurts

```text id="cbj0zw"
attention 排对；
hidden 排错；
simple fusion 排错或变差。
```

解释：

```text id="5i9f01"
hidden 在 keyness ranking 上有害。
```

### Type B: attention_correct_hidden_neutral

```text id="hs8084"
attention 排对；
hidden 打平或弱影响；
fusion 仍排对。
```

解释：

```text id="18826x"
hidden 没有明显破坏 attention。
```

### Type C: attention_wrong_hidden_fixes

```text id="41jcwc"
attention 排错；
hidden 排对；
fusion 排对。
```

解释：

```text id="1w4xu6"
hidden 对 keyness 有补充价值。
```

### Type D: attention_wrong_output_fixes

```text id="cinxtv"
attention 排错；
output-effect 排对；
attention/output fusion 排对。
```

解释：

```text id="sx9j13"
output-effect 对 keyness 有补充价值。
```

### Type E: all_wrong

```text id="wx1g2b"
attention / hidden / output-effect 都排错。
```

解释：

```text id="kxeu93"
需要更强 reasoning signal 或数据问题。
```

输出：

```text id="hpvsag"
attention_hidden_conflict_report.json
```

必须包含：

```text id="lnmh4i"
num_pairs
num_attention_correct
num_hidden_correct
num_fusion_correct
num_output_effect_correct
hidden_help_count
hidden_hurt_count
output_effect_help_count
output_effect_hurt_count
net_hidden_effect
net_output_effect
```

其中：

```text id="dou7ka"
net_hidden_effect = hidden_help_count - hidden_hurt_count
net_output_effect = output_effect_help_count - output_effect_hurt_count
```

---

# Phase 5: Fusion Error Cases

## 任务 7：输出具体 case

输出至少：

```text id="pgy8j3"
30 cases where attention correct but simple fusion wrong
30 cases where attention wrong but output-effect fixes
30 cases where all non-oracle signals fail
30 cases where gated formula succeeds
```

每个 case 包含：

```json id="c505qc"
{
  "source_question_id": "...",
  "question": "...",
  "candidate_spans": [
    {
      "span_text": "...",
      "span_type": "...",
      "solution_path_status_eval_only": "...",
      "surface_score": 0.0,
      "hidden_score": 0.0,
      "attention_score": 0.0,
      "output_effect_score": 0.0,
      "simple_fusion_score": 0.0,
      "gated_formula_score": 0.0,
      "ranks": {}
    }
  ],
  "failure_reason_auto_guess": "...",
  "interpretation": "..."
}
```

failure reason categories：

```text id="zrs5ht"
off_path_number_high_hidden
attention_missed_on_path_number
hidden_overweighted_distractor
output_effect_low_at_prompt_final_token
number_flooding
comparison_or_negation_suppressed
ambiguous_solution_path_number
span_overlap_confusion
all_signals_weak
unknown
```

输出：

```text id="wvjcyj"
fusion_error_cases.jsonl
fusion_success_cases.jsonl
question_level_case_audit.jsonl
```

---

# Phase 6: Gated Formula Validation

## 任务 8：不要再验证 simple average，验证分层公式

必须比较以下公式：

### F0: attention-only

```text id="xjjic3"
priority = attention_keyness
```

### F1: hidden-only

```text id="g6tn95"
priority = hidden_fragility
```

### F2: simple average baseline

```text id="w9879u"
priority = mean(attention_keyness, hidden_fragility)
```

### F3: attention × hidden

```text id="alyn3s"
priority = attention_keyness * hidden_fragility
```

### F4: attention gate then hidden

```text id="jx1teg"
if attention_keyness < threshold:
    priority = low
else:
    priority = hidden_fragility
```

### F5: attention × output-effect

```text id="6pqlp9"
priority = attention_keyness * output_effect
```

### F6: output-effect gate then attention

```text id="u2sfe6"
if output_effect < threshold:
    priority = low
else:
    priority = attention_keyness
```

### F7: attention gate then output-effect

```text id="1xg4hj"
if attention_keyness < threshold:
    priority = low
else:
    priority = output_effect
```

### F8: attention × output-effect × hidden

```text id="zfxuse"
priority = attention_keyness * output_effect * hidden_fragility
```

### F9: two-stage priority

```text id="e0qbrm"
keyness_score = mean(attention_keyness, output_effect)
if keyness_score < threshold:
    priority = low
else:
    priority = keyness_score * hidden_fragility
```

Thresholds must be label-free, for example:

```text id="s1pk95"
within-question median
within-question top-50%
global train-fold median if folds are implemented
```

Do not tune thresholds on evaluation labels unless explicitly marked oracle diagnostic.

---

## 任务 9：公式评估指标

主指标：

```text id="0gzlfx"
same_question_on_path_vs_off_path_auc
same_question_pairwise_accuracy
on_path_number_rank_mean
off_path_number_rank_mean
grouped_bootstrap_delta_vs_attention_only
grouped_bootstrap_delta_vs_surface
```

budget 指标：

```text id="3wv2r9"
off_path_budget_share
top_k_off_path_number_selected_rate
span_type_budget_distribution
comparison_topk_coverage
negation_topk_coverage
condition_topk_coverage
operation_topk_coverage
question_target_topk_coverage
```

fragility-aware 指标：

```text id="v4ytvj"
bucket3_vs_bucket1_auc
bucket3_topk_precision
bucket3_topk_recall
```

输出：

```text id="b4nt0g"
gated_formula_validation_report.json
gated_formula_bootstrap_report.json
```

---

# Phase 7: Span-Type and Error Breakdown

## 任务 10：按 span type 拆解

对每个 span type 报告：

```text id="yv2odg"
number
number_unit
rate
comparison
negation
condition
operation
question_target
object
```

统计：

```text id="ka26ve"
mean attention score
mean hidden score
mean output-effect score
mean simple fusion score
mean gated formula score
top-k selected rate
off-path selected rate if number-like
mean rank
num_cases
```

输出：

```text id="93071t"
span_type_error_breakdown.json
```

必须回答：

```text id="w7d6ch"
1. hidden 是否倾向抬高 off-path number？
2. attention 是否倾向抬高 rate / number_unit / question_target？
3. output-effect 是否对 question_target / comparison 特别敏感？
4. negation 是否样本太少导致结论不稳？
5. object 是否被过度选择？
```

---

# Phase 8: Review Gate

## 2K-V Gate

本 sprint 不以 2000-ready 为目标。

通过条件：

```text id="zsv1d7"
1. keyness_signal_eval_report generated；
2. fragility_signal_eval_report generated；
3. attention_hidden_conflict_report generated；
4. fusion_error_cases generated；
5. gated_formula_validation_report generated；
6. no leakage in gate-eligible features；
7. attention-only remains stable above surface OR report explains why it changed；
8. simple hidden+attention fusion is explicitly compared against attention-only；
9. at least one gated formula is tested against attention-only using grouped bootstrap；
10. ready_for_2000_rerun=False；
11. do_not_enter_sprint_3A=True。
```

Do not require gated formula to pass.
This sprint is a diagnostic role-decomposition sprint.

---

## Success Criteria

Strong success：

```text id="eyqq1n"
1. attention-only stable above surface；
2. hidden hurts keyness but helps fragility；
3. simple fusion worse than attention-only because hidden over-ranks distractors；
4. attention/output-effect gated formula beats attention-only with ci95_low > 0；
5. off_path_budget_share decreases；
6. non-number key spans are not suppressed。
```

Partial success：

```text id="cujd9y"
1. attention-only stable above surface；
2. hidden hurts keyness but fragility role is unclear；
3. gated formula does not beat attention-only；
4. output-effect helps only on subset。
```

Negative result：

```text id="wdjajv"
1. attention-only improvement disappears；
2. output-effect does not help；
3. hidden does not help keyness or fragility；
4. all formulas remain close to random。
```

If negative, recommend:

```text id="g5p9yh"
answer-position output-effect / CoT trajectory / NLA-lite / richer dataset decision sprint.
```

---

# Final Report Requirements

`review_gate_signal_role_decomposition.md` 必须包含：

```text id="v7zll9"
Verdict:
- ...

Signal roles:
- attention:
- hidden:
- output-effect:
- surface:

Keyness evaluation:
- ...

Fragility evaluation:
- ...

Conflict audit:
- ...

Fusion error analysis:
- ...

Gated formula validation:
- ...

Span-type breakdown:
- ...

Leakage audit:
- ...

Root cause:
- ...

Recommendation:
- ...

Next sprint candidate:
- ...
```

---

## 最终报告必须回答的问题

```text id="apwspo"
1. hidden+attention 为什么低于 attention-only？
2. hidden 是 harmful，还是只是用错位置？
3. attention 是否应该作为第一层 keyness gate？
4. output-effect 是否增强 attention keyness？
5. hidden 是否应该保留为第二层 fragility/risk signal？
6. simple average 是否应废弃？
7. 最好的分层公式是什么？
8. 分层公式是否稳定优于 attention-only？
9. off-path number 是否被压下去了？
10. comparison / negation / condition 是否被保留？
11. 是否建议做 answer-position output-effect？
12. 是否建议进入 2000？
13. 是否建议进入 Sprint 3A？
```

---

# PROGRESS.md 更新要求

更新 PROGRESS.md 当前阶段，必须写清：

```text id="klp4vk"
1. 本轮是 signal role decomposition，不是 scale-up；
2. 使用 2J-Fix/2K 4935-span artifacts；
3. 是否确认 attention=keyness、hidden=fragility；
4. simple fusion 为什么拉低 attention-only；
5. gated formula 是否优于 attention-only；
6. ready_for_2000_rerun=False；
7. do_not_enter_sprint_3A=True。
```

同时更新：

```text id="irx33r"
docs/progress/sprint_2_history.md
```

---

## 关键提醒

本 sprint 最重要的不是追求新最高分，而是验证结构假设：

```text id="7nkb2o"
Do not average signals before understanding their roles.
```

如果 attention 是 keyness、hidden 是 fragility，那么它们应该进入不同层级：

```text id="8hkp0i"
keyness gate first；
fragility ranking second；
budget priority last。
```

不要再把 keyness 和 fragility 直接揉成一个未解释的 scalar。

---

# 执行补充（review 后自行补充的修订）

以下 4 点在执行前对本 card 的补充，用于消除潜在漏洞（不改变 card 主体目标）：

1. **Phase 3 具体化（fragility 标签来源）**：2J-Fix multi-span feature matrix 中 `fragility_bucket_if_available` 大多为 null（仅 274/4935 有值），单靠 in-matrix 标签无法稳定评估 hidden 的 fragility 角色。补充做法：把 2J number span 按 `(source_question_id, normalize_number_text(span_text))` join 到 `outputs/logs/sprint_2H_instance_signal_500/risk_strength_dataset.jsonl` 的 fragility_bucket，得到 **382 个带标签的 number span**（bucket 分布 {0:43,1:31,2:82,3:226}），用于计算 hidden/attention/output-effect 的 `bucket3_vs_bucket1_auc`。fragility_bucket 只作 eval-only label，绝不进 gate-eligible 输入。

2. **output-effect 测点 caveat**：2K/2K-V 的 output-effect 均在 prompt 末 token 测量（next-token-after-prompt），而非 answer 位置。因此 output-effect 的任何 null/弱结果都不能否定其 keyness 价值——需要 answer-eliciting 测点复核。此 caveat 写入所有 output-effect 结论。

3. **多重比较 caveat**：F0–F9 共 10 个 gated formula 与 attention-only 比较，存在选择性偏差。要求 stable 的判定用 `ci95_low > 0` 且在报告中写明「共试了 10 个公式」。

4. **小组阈值退化**：within-question median gate 在每题仅 2–3 个 span 时区分度弱；报告统计 gate 实际未区分的题数（`small_group_gate_uninformative`）。

---

# 执行结果摘要（500-case，4935 spans，read-only）

**关键结果与 card 假设部分相反（诚实记录）：**

```text
same-question on/off-path AUC：surface 0.512 / hidden 0.468 / attention 0.588 / simple_fusion 0.564 / attention×output 0.586
fragility bucket3-vs-bucket1 AUC（2H join，382 number spans）：hidden 0.480（≈随机）/ attention 0.715 / output-effect 0.485
conflict pair（1871 on/off pair）：net_hidden_effect=-46（hidden 净拖累 keyness）/ net_output_effect=+16（output-effect 修正 171 个 attention 排错 pair）
gated formula 稳定优于 attention-only：无（gated_beats_attention=[]）
```

- **attention 在 keyness 与 fragility 上都是最强信号**（0.588 / 0.715）；**hidden 在两者上都弱**（0.468 / 0.480）。因此 card 假设「hidden=fragility、应作第二层 fragility ranker」**不成立**——hidden 在 fragility 上也接近随机。
- **simple fusion 低于 attention-only 的原因是 hidden 是噪声（非互补信号）**，而非 keyness/fragility 被混淆。conflict 审计证实 hidden 净拖累（-46）。
- output-effect 有 modest 互补价值（net +16，D 型 171 pair），但 attention×output（0.586）未在 aggregate AUC 上超过 attention-only；单独 output-effect 不稳定（受测点限制）。
- 结论：**保留 attention 作为第一层 keyness gate；不要用 hidden（当前构造下两个角色都弱）；不要 simple-average；先做 answer-position output-effect 复核再决定 output-effect 去留。** 仍不进入 2000 / 3A（oracle 0.9995 vs best 0.588，gap 仍大）。

产物见 `outputs/logs/sprint_2K_V_signal_role_decomposition_500/`；review gate = `review_gate_signal_role_decomposition.md`。
