# Sprint 2J: Multi-Span Reasoning Matrix Construction

## 背景

Sprint 2I-R 已完成 score matrix decomposition and root-cause audit。

2I-R 暴露出一个关键结构问题：

```text id="r2u71w"
当前 500-case diagnostic 实际上是 one-question-one-span：
477 个 question，各自只有 1 条 scored span。
```

因此，当前 pipeline 无法真正评估：

```text id="3xki29"
same-question pairwise ranking
per-question top-k selection
off-path budget share within the same question
span-type budget cap
keyness × fragility formula
```

也就是说，当前系统虽然能做全局分类 / 全局排序，但还不能回答 attention guidance 最核心的问题：

```text id="zv2b3n"
同一道题里，哪些 span 最应该被优先增强 attention？
```

本 sprint 的目标是补上这个地基：把数据结构从 `one question -> one span` 改成 `one question -> multiple candidate spans`。

---

## 核心目标

本 sprint 构造真正的 multi-span-per-question reasoning matrix。

目标不是立即提高 macro_f1，也不是进入 steering，而是建立后续公式验证、NLA-lite、CoT、trajectory、causal attribution 都能共用的数据结构。

本 sprint 需要回答：

```text id="rgghni"
1. 每道题能否稳定抽取 3–10 个 candidate spans？
2. candidate spans 是否覆盖 on-path numbers、off-path numbers、operation、comparison、negation、condition、question target、object/entity？
3. 是否能构造 same-question ranking 所需的 group matrix？
4. surface / hidden / attention 是否能在同一道题内部区分 key spans 和 distractor spans？
5. 分层公式 keyness × fragility 是否优于 single score？
6. 是否仍然缺少 reasoning-level signals，例如 answer-logprob、CoT、trajectory、NLA 或 causal attribution？
```

---

## 严格边界

允许：

```text id="ukna0w"
读取 GSM8K 500-case diagnostic 原始 question / answer / solution；
读取 2H-B / 2H-C / 2H-D / 2I 的既有输出；
构造 multi-span candidate matrix；
抽取 surface / semantic span features；
解析 solution-path number 作为 evaluation-only keyness label；
复用已有 hidden / attention feature extraction 逻辑；
对 candidate spans 运行 original/masked forward；
模拟分层公式；
评估 same-question ranking / top-k / budget metrics；
输出 audit reports。
```

禁止：

```text id="z30ox8"
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要重跑 recovery；
不要生成 CoT；
不要生成 trajectory；
不要实现 NLA；
不要实现 causal attribution / activation patching；
不要把 gold solution path / fragility_bucket / risk_strength / recovery drift 当作 formula input；
不要用 oracle label 直接决定 priority score；
不要把本 sprint 包装成 2000-ready。
```

默认：

```text id="tx0l62"
ready_for_2000_rerun=False
do_not_enter_sprint_3A=True
```

除非后续明确通过独立 validation gate，否则不能改变。

---

## 总体阶段

本 sprint 分为两个阶段：

```text id="b4k7ty"
2J-A: Multi-Span Candidate Matrix
2J-B: Multi-Span Hidden/Attention Scoring and Formula Validation
```

2J-A 必须先通过 coverage audit，才能执行 2J-B。

如果 2J-A 发现 candidate extraction 覆盖不足，不要继续跑 hidden/attention forward。

---

# Phase 2J-A: Multi-Span Candidate Matrix

## 任务 1：读取基础数据

优先读取 500-case diagnostic 中已经使用过的 question set。

输入来源可以包括：

```text id="wp8kjg"
outputs/logs/sprint_2H_instance_signal_500/risk_strength_dataset.jsonl
outputs/logs/sprint_2H_feature_enrichment_500/pre_recovery_feature_dataset.jsonl
outputs/logs/sprint_2I_attention_features_500/attention_feature_dataset.jsonl
```

需要恢复或对齐：

```text id="7n22lt"
source_question_id
question
answer / gold final answer
gold solution text if available
existing sampled span records if available
```

如果同一道题在多个文件中重复出现，只保留一个 canonical question record，并记录 provenance。

---

## 任务 2：抽取候选 span

对每道题抽取多个 candidate spans。

每道题目标数量：

```text id="n1sx15"
min_spans_per_question >= 3
target_spans_per_question = 4–10
max_spans_per_question = 12
```

候选 span 类型至少包括：

### A. Number spans

包括：

```text id="iil9dc"
explicit numbers: 4, 3, 2020, 12.5, 50%
word numbers: three, twelve, half
multiplier words: twice, double, triple
number-unit phrases: 4 apples, 3 dollars each, 5 miles
rate phrases: 3 dollars each, 2 miles per hour
```

数字 span 必须尽量全覆盖，因为后续需要评估：

```text id="vrnuc3"
on-path number > off-path number
```

### B. Operation spans

包括：

```text id="r8nkwk"
bought
sold
gave
left
remaining
total
each
per
spent
cost
earned
lost
shared
divided
increased
decreased
```

### C. Comparison / direction spans

包括：

```text id="pctvnt"
more than
less than
fewer than
older than
younger than
twice as many
half as many
at least
at most
difference between
```

### D. Negation / condition spans

包括：

```text id="s4i15k"
not
without
except
unless
if
after
before
only
remaining
instead
```

### E. Question target spans

包括：

```text id="9ci0pq"
How many
How much
How far
What is the total
How many are left
How much did he spend
```

### F. Object / entity / unit spans

包括：

```text id="2mimk5"
apples
dollars
books
miles
Sam
Alice
Bob
students
tickets
```

Object / entity spans 需要保留一部分作为 low-priority / context candidates，但不能让它们淹没矩阵。

---

## 任务 3：span 去重与合并

需要处理重叠 span。

规则：

```text id="6jrj4l"
1. number-unit phrase 优先于裸 number；
2. rate phrase 优先于裸 number；
3. comparison phrase 优先于单个 comparison word；
4. question target phrase 优先于零散 question words；
5. 严重重叠 span 不重复保留，除非一个是 phrase，一个是 atomic number，且后续需要分别评估。
```

例如：

```text id="2f9xkq"
3 dollars each
```

可以同时保留：

```text id="1ya4gc"
3
3 dollars each
```

但必须标记二者关系：

```text id="u5g7t5"
parent_span_id
child_span_ids
overlap_group_id
```

这样后续评估时可以避免重复计算。

---

## 任务 4：构造 keyness diagnostic labels

### 数字 span

对 number / number-unit / rate span，使用 GSM8K solution path 解析结果作为 evaluation-only keyness diagnostic。

标签：

```text id="ueueta"
on_solution_path_number
off_solution_path_number
ambiguous_number
not_number
```

注意：

```text id="sjz1vi"
solution_path_status 只能作为 evaluation-only label；
不能作为 formula input；
不能作为 model feature；
不能决定 gate-eligible score。
```

### 非数字 span

非数字 span 暂时使用 weak semantic diagnostic：

```text id="4frv90"
comparison_key_candidate
negation_key_candidate
condition_key_candidate
operation_key_candidate
question_target_candidate
object_or_context_candidate
unknown
```

这些也只能作为 diagnostic，不要当作 gold hard label。

---

## 任务 5：构造 multi-span matrix schema

每道题输出一个 grouped record：

```json id="1qgs5n"
{
  "source_question_id": "...",
  "question": "...",
  "gold_answer": "...",
  "gold_solution": "...",
  "candidate_spans": [
    {
      "span_id": "...",
      "span_text": "...",
      "span_type": "...",
      "start_char": 0,
      "end_char": 0,
      "overlap_group_id": "...",
      "parent_span_id": null,
      "child_span_ids": [],
      "surface_features": {},
      "diagnostic_labels_for_eval_only": {
        "solution_path_status": null,
        "weak_semantic_keyness": null
      }
    }
  ]
}
```

同时输出 flattened record，方便后续训练/评估：

```json id="gatryd"
{
  "source_question_id": "...",
  "span_id": "...",
  "span_text": "...",
  "span_type": "...",
  "question": "...",
  "masked_question": "...",
  "surface_features": {},
  "diagnostic_labels_for_eval_only": {}
}
```

必须保留：

```text id="r0lsy8"
source_question_id
```

因为后续所有 ranking metric 都依赖同题分组。

---

## 任务 6：2J-A coverage audit

输出 coverage report。

至少报告：

```text id="dgk26m"
num_questions
num_candidate_spans
mean_spans_per_question
median_spans_per_question
min_spans_per_question
max_spans_per_question
num_questions_with_less_than_3_spans
span_type_distribution
number_span_distribution
questions_with_on_path_and_off_path_number
questions_with_comparison
questions_with_negation
questions_with_condition
questions_with_operation
questions_with_question_target
questions_with_object_or_entity
overlap_group_statistics
```

2J-A pass 条件：

```text id="gxqkkm"
1. 至少 90% questions 有 >=3 个 candidate spans；
2. 平均每题 candidate spans 在 4–10 之间；
3. number spans 覆盖率高；
4. 有足够 on-path/off-path number 对用于 same-question keyness ranking；
5. comparison / negation / condition / operation 不被完全漏掉；
6. 输出 group matrix 和 flat matrix；
7. no leakage feature audit 通过。
```

如果 2J-A 不通过，不要执行 2J-B。

---

# Phase 2J-B: Multi-Span Hidden/Attention Scoring and Formula Validation

## 任务 7：masked question 构造

对每个 candidate span 构造 masked question。

要求：

```text id="cqseqk"
1. 保持原题其他内容不变；
2. 用统一 mask token 或 placeholder 替换 span；
3. 记录 char span 和 token span alignment；
4. 如果 phrase span 与 atomic span 重叠，二者分别构造 masked question；
5. mask 不得泄漏 span_type。
```

输出字段：

```text id="rba73f"
masked_question
mask_text
span_char_start
span_char_end
span_token_indices
mask_token_indices
alignment_status
```

如果 token alignment 失败，记录 warning，不要静默丢弃。

---

## 任务 8：original / masked forward

对每道题：

```text id="npmz95"
original question forward 一次；
每个 candidate span 的 masked question forward 一次。
```

复用 original forward：

```text id="irqlye"
同一 source_question_id 的 original hidden / attention 只缓存一次。
```

保存层：

```text id="t0krnu"
0 / 8 / 16 / 24
```

不要使用 final layer 27 attention，因为 2I 已发现 4-bit eager 下 final layer attention 会出现 all-NaN。

如果 hidden final layer 可用，可以另行保存 hidden，但 attention 层集合必须避开已知 NaN 层。

---

## 任务 9：抽取 hidden features

沿用 2H-C 的 enriched pre-recovery hidden feature 思路。

每个 span 至少抽取：

```text id="9kyvfv"
hidden_delta_l2
hidden_delta_relative_norm
hidden_delta_cosine
span_context_shift_norm
question_context_shift_norm
cross_layer_stability
early_mid_late_delta_slope
max_delta_layer
span_to_question_similarity
span_to_mask_similarity
```

新增 same-question features：

```text id="hd4f5j"
hidden_delta_rank_within_question
hidden_delta_zscore_within_question
hidden_stability_rank_within_question
```

注意：

```text id="h5as3e"
within-question rank 只能用当前 question 内其他 candidate spans；
不能用 diagnostic labels。
```

---

## 任务 10：抽取 attention features

沿用 2I 的 attention feature 思路。

每个 span 至少抽取：

```text id="v1e69f"
span_attention_in_mass
span_attention_out_mass
span_attention_entropy
span_attention_topk_mass
context_to_slot_attention
operation_to_slot_attention
qfocus_to_slot_attention
number_context_to_slot_attention
mask_to_span_attention
span_to_mask_attention
original_to_masked_attention_delta
attention_entropy_delta
attention_rank_delta
```

新增 same-question features：

```text id="evs5l4"
attention_delta_rank_within_question
attention_mass_rank_within_question
attention_entropy_rank_within_question
```

如果某类 context 不存在，例如 operation context，不要造假值，填 null 并记录 missing rate。

---

## 任务 11：构造 score matrix

对每个 candidate span 输出：

```json id="gpx8sa"
{
  "source_question_id": "...",
  "span_id": "...",
  "span_text": "...",
  "span_type": "...",

  "keyness_signals": {
    "surface_keyness_proxy": null,
    "semantic_keyness_proxy": null,
    "within_question_surface_rank": null
  },

  "fragility_signals": {
    "hidden_fragility_score": null,
    "attention_fragility_score": null,
    "hidden_plus_attention_score": null,
    "within_question_hidden_rank": null,
    "within_question_attention_rank": null
  },

  "reasoning_signals": {
    "answer_logprob_delta": null,
    "answer_rank_delta": null,
    "trajectory_change": null,
    "cot_path_change": null,
    "nla_semantic_role": null
  },

  "budget_signals": {
    "priority_score": null,
    "priority_rank_within_question": null,
    "off_path_budget_risk_eval_only": null
  },

  "diagnostic_labels_for_eval_only": {
    "solution_path_status": null,
    "weak_semantic_keyness": null,
    "fragility_bucket_if_available": null,
    "risk_strength_if_available": null
  }
}
```

`reasoning_signals` 在本 sprint 中可以保持 null，但必须保留 schema。

---

## 任务 12：Feature leakage audit

所有 gate-eligible formula input feature names 禁止包含：

```text id="cmhwwh"
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
nla
oracle
```

这些字段允许出现在：

```text id="vx99hx"
diagnostic_labels_for_eval_only
reasoning_signals placeholder
reports
```

但不能进入公式输入或 probe input。

如果泄漏字段进入 formula input，直接 fail。

---

## 任务 13：公式模拟

不要训练复杂模型，先模拟以下公式。

### A. Surface-only baseline

```text id="9neqzx"
priority = surface_keyness_proxy
```

### B. Hidden-only fragility

```text id="yxaagr"
priority = hidden_fragility_score
```

### C. Attention-only fragility

```text id="ojhjwp"
priority = attention_fragility_score
```

### D. Hidden + attention

```text id="42f2ec"
priority = mean(hidden_fragility_score, attention_fragility_score)
```

### E. Keyness × fragility

```text id="cncziu"
priority = keyness_score * fragility_score
```

### F. Keyness gate then fragility

```text id="1aq7vu"
if keyness_score < threshold:
    priority = low_priority
else:
    priority = fragility_score
```

### G. Per-question normalized priority

```text id="lg0ggo"
priority = normalize_within_question(raw_priority)
```

### H. Span-type budget policy

Within each question:

```text id="pwl37o"
1. number spans cannot occupy all top-k slots if comparison / negation / condition has high score;
2. off-path-like surface distractors should be downweighted by non-oracle proxy;
3. object/entity spans should require strong hidden/attention evidence to enter top-k.
```

Do not use `solution_path_status` directly in any gate-eligible formula.

### I. Oracle diagnostic upper bound

Allowed only for analysis:

```text id="7bhuu7"
oracle_priority = solution_path_keyness * fragility_bucket
```

This must be marked:

```text id="6ps3f7"
oracle_diagnostic_only
uses_eval_only_labels=True
not_gate_eligible
```

---

## 任务 14：same-question ranking metrics

This sprint must report group-based metrics.

Core metrics:

```text id="5zxo70"
same_question_pairwise_accuracy
same_question_on_path_vs_off_path_auc
same_question_bucket3_vs_bucket1_pairwise
per_question_top1_key_hit
per_question_top2_key_hit
per_question_top3_key_hit
per_question_top1_bucket3_hit
per_question_top2_bucket3_hit
per_question_top3_bucket3_hit
top_k_off_path_number_selected_rate
off_path_budget_share
span_type_budget_distribution
```

For number subset:

```text id="39afsr"
on_path_number_rank_mean
off_path_number_rank_mean
on_path_number_topk_coverage
off_path_number_false_positive_rate
```

For non-number subset:

```text id="jn43b9"
comparison_topk_coverage
negation_topk_coverage
condition_topk_coverage
operation_topk_coverage
question_target_topk_coverage
```

---

## 任务 15：failure case audit

输出 top-k failure / success cases。

至少输出：

```text id="vgnzlq"
30 top-k failure questions
30 top-k success questions
```

每个 question-level case 包含：

```text id="zvl0vu"
question
candidate_spans
scores_by_formula
ranks_by_formula
diagnostic labels
selected_topk_spans
expected_priority_spans_eval_only
failure_reason_auto_guess
```

failure reason categories:

```text id="di4qq1"
off_path_number_overranked
on_path_number_underranked
comparison_underranked
negation_underranked
condition_underranked
object_overranked
question_target_underranked
span_extraction_error
token_alignment_error
label_ambiguous
multiple_reasonable_spans
missing_reasoning_signal
unknown
```

---

## 任务 16：root-cause report

最终报告必须判断：

```text id="ed01j0"
1. multi-span matrix 是否构造成功？
2. same-question ranking 是否终于可计算？
3. keyness 问题是否主要来自 on/off-path number 区分？
4. fragility 问题是否主要来自 hidden/attention 信号不够？
5. 分层公式是否明显优于 single score？
6. off-path number 是否仍然吃掉 budget？
7. comparison / negation / condition 是否仍然被压低？
8. 是否确实需要 reasoning-level signal？
9. 下一步应该是 formula validation、answer-logprob、NLA-lite、CoT trajectory，还是 causal attribution？
```

---

## 输出目录

2J-A 输出：

```text id="uvt0nb"
outputs/logs/sprint_2J_multi_span_matrix_500/
```

2J-B 输出：

```text id="tq813p"
outputs/logs/sprint_2J_multi_span_scoring_500/
```

2J-A 必须输出：

```text id="vzbi2l"
multi_span_grouped_matrix.jsonl
multi_span_flat_matrix.jsonl
candidate_span_coverage_report.json
span_extraction_audit.json
keyness_label_report.json
surface_ranking_baseline_report.json
review_gate_multi_span_matrix.md
```

2J-B 必须输出：

```text id="mpklwt"
multi_span_feature_matrix.jsonl
multi_span_score_matrix.jsonl
hidden_attention_feature_report.json
same_question_ranking_report.json
formula_validation_report.json
topk_budget_report.json
failure_case_report.jsonl
success_case_report.jsonl
reasoning_signal_gap_report.json
review_gate_multi_span_scoring.md
```

---

## 2J-A Gate

2J-A 通过条件：

```text id="9hpzw3"
1. >=90% questions have at least 3 candidate spans；
2. mean spans per question between 4 and 10；
3. number spans are extracted with high coverage；
4. enough questions contain both on-path and off-path number candidates；
5. non-number candidate types are not empty；
6. grouped and flat matrix outputs are both generated；
7. no leakage in gate-eligible surface features；
8. token/char alignment failure rate is reported and acceptable。
```

如果 2J-A 不通过：

```text id="7wwshm"
do not run hidden/attention scoring；
fix candidate extraction first。
```

---

## 2J-B Gate

2J-B 通过条件：

```text id="ag8k0f"
1. same-question pairwise ranking is computable；
2. at least one non-oracle formula significantly improves over surface-only baseline；
3. keyness × fragility or gated formula improves over single score；
4. off-path budget share decreases or does not worsen；
5. on-path number top-k coverage improves；
6. comparison / negation / condition are not systematically suppressed；
7. bootstrap or grouped resampling confirms improvement stability；
8. no leakage。
```

即使 2J-B 通过，也不要直接进入 Sprint 3A。

最多建议进入：

```text id="py9kh9"
formula validation sprint
```

或者：

```text id="qfqz25"
reasoning-signal sprint
```

---

## 推荐决策规则

最终根据结果选择下一步：

```text id="elmy90"
Case 1:
multi-span matrix works, formula improves ranking
=> next: formula validation sprint.

Case 2:
multi-span matrix works, formula does not improve ranking, keyness errors dominate
=> next: NLA-lite / semantic role explanation.

Case 3:
multi-span matrix works, formula does not improve ranking, fragility errors dominate
=> next: answer-logprob / causal attribution.

Case 4:
multi-span matrix works, hidden/attention cannot distinguish reasoning consequences
=> next: CoT / trajectory / answer-stability sprint.

Case 5:
multi-span matrix fails coverage
=> fix span extraction, do not add new reasoning signals yet.
```

---

## 最终报告格式

最终 `review_gate_multi_span_scoring.md` 必须用以下结构：

```text id="2kpqzq"
Verdict:
- ...

Coverage:
- ...

Same-question ranking:
- ...

Formula comparison:
- ...

Budget analysis:
- ...

Failure cases:
- ...

Reasoning signal gap:
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

本 sprint 的核心不是证明 hidden/attention 一定有效，而是建立正确的评价地基：

```text id="ex43ud"
Attention guidance is a per-question top-k decision.
Therefore the dataset must contain multiple candidate spans per question.
```

如果没有 multi-span matrix，后续任何 NLA / CoT / trajectory / causal attribution 都无法被可靠评估。
