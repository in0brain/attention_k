# Sprint 2K-W: Answer-Position Output-Effect Re-measurement

## 背景

Sprint 2J-Fix + 2K + 2K-V 已完成。

当前结论：

```text id="rke437"
1. 2J-B 的 hidden/attention anti-keyness 结论是 slot alignment bug artifact；
2. 修复后 attention-only 在 same-question on/off-path AUC 上稳定优于 surface；
3. 2K-V 显示 attention 在 keyness 与 joined fragility bucket 上都强于 hidden；
4. hidden 当前构造下主要是噪声，不应进入 simple fusion；
5. output-effect 在 prompt final token 测点上有 modest 互补，但单独不稳定；
6. F0-F9 gated formulas 无一稳定超过 attention-only；
7. oracle 仍远高于 best non-oracle，说明 priority signal 仍不够强。
```

当前关键数字：

```text id="b2rxq8"
keyness AUC:
surface ≈ 0.512
hidden ≈ 0.468
attention-only ≈ 0.588
simple hidden+attention fusion ≈ 0.564
attention × prompt-final output-effect ≈ 0.586
oracle ≈ 0.9995
```

因此，下一步不是进入 steering，也不是继续调 hidden fusion，而是复测 output-effect 的测点。

2K 的 output-effect 当前在：

```text id="vz66ms"
prompt final token
```

即直接在题目结尾看 next-token distribution shift。这个位置可能离真正答案生成太远，低估了 output-effect 的作用。

本 sprint 将 output-effect 移到：

```text id="cm21gl"
answer-eliciting position
```

例如：

```text id="mrt4pa"
Question: <question>
Answer:
```

然后比较 original/masked 在 `Answer:` 后第一个 token 的分布变化。

---

## 核心目标

本 sprint 回答：

```text id="xuk2ik"
1. prompt-final output-effect 是否因为测点太早而被低估？
2. answer-position output-effect 是否比 prompt-final output-effect 更强？
3. answer-position output-effect 是否能稳定优于 surface？
4. answer-position output-effect 是否能稳定接近或超过 attention-only？
5. attention × answer-position output-effect 是否稳定优于 attention-only？
6. answer-position output-effect 是否能改善 off-path number budget？
7. answer-position output-effect 是否对 comparison / negation / condition / question_target 更敏感？
8. 如果 output-effect 仍不强，是否可以确认当前 steering target 应暂时使用 attention-only？
```

---

## 严格边界

允许：

```text id="msfifu"
读取 2J-Fix / 2K / 2K-V 已有 artifacts；
复用 4935-span multi-span matrix；
重跑 original/masked forward，但只用于 answer-position logits；
构造 answer-eliciting prompt；
计算 leakage-safe self-output distribution shift；
对比 prompt-final output-effect 与 answer-position output-effect；
构造 formula scores；
做 same-question ranking；
做 grouped bootstrap；
输出 reports 和 failure/success cases；
更新 PROGRESS.md 和 sprint_2_history.md；
新增 tests。
```

禁止：

```text id="ty2tvr"
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要修改 attention；
不要重跑 recovery；
不要生成 CoT；
不要生成 trajectory；
不要实现 NLA；
不要实现 activation patching / causal attribution；
不要把 gold answer / solution_path / fragility_bucket / risk_strength / recovery drift 当作 gate-eligible input；
不要把 gold-answer logprob delta 作为 feature；
不要用 oracle label 选择公式；
不要只看 top-k coverage 宣布通过；
不要继续使用 hidden+attention simple average 作为主公式。
```

默认：

```text id="z3o46n"
ready_for_2000_rerun=False
do_not_enter_sprint_3A=True
```

---

# Phase 0: Inputs and Output Directory

## 输入

优先读取：

```text id="jz3ggu"
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_score_matrix.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/same_question_ranking_report.json
outputs/logs/sprint_2K_answer_effect_keyness_500/output_effect_feature_matrix.jsonl
outputs/logs/sprint_2K_answer_effect_keyness_500/same_question_output_effect_ranking_report.json
outputs/logs/sprint_2K_V_signal_role_decomposition_500/signal_role_summary.json
outputs/logs/sprint_2K_V_signal_role_decomposition_500/keyness_signal_eval_report.json
outputs/logs/sprint_2K_V_signal_role_decomposition_500/attention_hidden_conflict_report.json
```

如果 summary reports 不存在，但 feature matrix 存在，可以从 feature matrix 重建 score matrix。

不要依赖 PROGRESS.md 作为唯一数据源。
PROGRESS.md 只能作为 sanity check。

---

## 输出目录

```text id="bckov9"
outputs/logs/sprint_2K_W_answer_position_output_effect_500/
```

必须输出：

```text id="gkjyey"
answer_position_feature_matrix.jsonl
answer_position_score_matrix.jsonl
answer_position_feature_audit.json
answer_position_ranking_report.json
answer_position_formula_validation_report.json
answer_position_grouped_bootstrap_report.json
prompt_vs_answer_position_comparison.json
span_type_answer_position_breakdown.json
off_path_budget_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_answer_position_output_effect.md
```

---

# Phase 1: Construct Answer-Eliciting Inputs

## 任务 1：构造 answer-position prompt

对每条原始 question 构造：

```text id="zq94hj"
Question: {question}
Answer:
```

对每条 masked question 构造：

```text id="by2ui3"
Question: {masked_question}
Answer:
```

注意：

```text id="rnqvv0"
original 和 masked 必须使用完全相同的 prompt template；
唯一差异只能是 span 被 [MASK] 替换；
不要加入 gold solution；
不要加入 chain-of-thought；
不要加入 few-shot；
不要加入任何 correct answer hint。
```

---

## 任务 2：prompt template ablation，可选但建议

为了避免单一模板偶然影响，可以支持两种模板：

### Template A: Minimal answer prompt

```text id="dxo9kl"
Question: {question}
Answer:
```

### Template B: Direct concise prompt

```text id="m5gspw"
Solve the problem. Give only the final answer.

Question: {question}
Answer:
```

主结果默认使用 Template A。

Template B 只作为 robustness diagnostic，不作为 primary，除非 Template A 出现明显异常。

输出中记录：

```json id="lk3m4k"
{
  "primary_template": "Question: {question}\\nAnswer:",
  "secondary_template": "...",
  "template_policy": "Template A primary; Template B diagnostic only"
}
```

---

# Phase 2: Forward and Logits Collection

## 任务 3：复用模型 forward，只保存 answer-position logits summary

对每个 span，需要两次 answer-position forward：

```text id="jh7nqv"
original_answer_prompt
masked_answer_prompt
```

记录最后一个输入 token 之后的 next-token logits：

```text id="dvscje"
original_answer_position_last_logits
masked_answer_position_last_logits
```

不要把完整 vocab logits 写入 JSONL。

可以保存：

```text id="txd9v5"
top-k token ids
top-k token logprobs
entropy
margin
top1 token id
top1 token logprob
```

如果需要完整 logits，保存到 `.pt` 或 `.npy`，并在 manifest 里记录路径。

建议输出 manifest：

```text id="lfnvxx"
answer_position_forward_manifest.jsonl
```

每条记录包含：

```json id="sp9l4r"
{
  "span_id": "...",
  "source_question_id": "...",
  "template_id": "A",
  "original_prompt_hash": "...",
  "masked_prompt_hash": "...",
  "original_topk_path": null,
  "masked_topk_path": null,
  "forward_status": "ok|failed",
  "warnings": []
}
```

---

## 任务 4：避免重复 original forward

同一道题的 original answer prompt 相同。

因此：

```text id="s6lfqq"
original answer-position forward 可以按 source_question_id 缓存；
masked answer-position forward 必须按 span_id 单独计算。
```

注意：

```text id="n86m92"
这里不涉及 slot_indices；
本 sprint 只计算 logits-derived output-effect，不需要 hidden/attention slot pooling。
```

---

# Phase 3: Leakage-Safe Answer-Position Output-Effect Features

## 任务 5：复用 2K 的 output-effect feature schema

使用相同命名，避免 `answer` 作为 gate-eligible feature substring。

不要命名为：

```text id="rf0v5q"
answer_logprob_delta
answer_output_kl
gold_answer_effect
```

推荐 namespace：

```json id="kix78z"
{
  "response_position_output_effect": {
    "self_output_kl": ...,
    "self_output_js": ...,
    "self_output_top1_changed": ...,
    "self_output_topk_overlap": ...,
    "self_output_entropy_delta": ...,
    "self_output_margin_delta": ...,
    "self_output_logprob_delta": ...
  }
}
```

或者更严格：

```json id="iokln0"
{
  "resp_pos_output_effect": {
    "self_output_kl": ...,
    "self_output_js": ...,
    "self_output_top1_changed": ...,
    "self_output_topk_overlap": ...,
    "self_output_entropy_delta": ...,
    "self_output_margin_delta": ...,
    "self_output_logprob_delta": ...
  }
}
```

推荐使用：

```text id="lu6mk3"
resp_pos_output_effect
```

因为 `answer` 仍是 banned substring。

---

## 任务 6：feature leakage audit

对所有 gate-eligible feature names 执行 banned substring audit。

禁止：

```text id="itqj2p"
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

注意：

```text id="fcbibk"
输出目录或 report 名可以叫 answer_position；
但 gate-eligible feature names 不要包含 answer。
```

输出：

```text id="o9iizb"
answer_position_feature_audit.json
```

必须包含：

```json id="in178c"
{
  "passed": true,
  "num_feature_names_checked": ...,
  "leaked_input_features": [],
  "forbidden_substrings": [...]
}
```

---

# Phase 4: Score Construction

## 任务 7：构造 response-position output shift score

沿用 2K 的 `output_effect_shift_score`，但输入换成 answer-position logits-derived features。

命名：

```text id="1nap0d"
resp_pos_output_shift
```

不要命名为 `answer_effect_score`。

计算方式可复用：

```text id="uwzxnb"
self_output_js
+ clipped self_output_kl
+ self_output_top1_changed
+ (1 - self_output_topk_overlap)
+ clipped original-top1 logprob drop
```

输出到每条 span：

```json id="fkdoat"
{
  "span_id": "...",
  "signals": {
    "attention_keyness": ...,
    "prompt_final_output_shift": ...,
    "resp_pos_output_shift": ...
  }
}
```

---

## 任务 8：保留 prompt-final output-effect 作为对照

从 2K artifacts 读取：

```text id="fxtv6z"
prompt_final_output_shift
```

与新测得的：

```text id="a1ibwf"
resp_pos_output_shift
```

对比。

同一条 span 应同时有：

```json id="3ayvj1"
{
  "prompt_final_output_shift": ...,
  "resp_pos_output_shift": ...
}
```

如果 prompt-final 数据缺失，则报告为 unavailable，但不要中断主流程。

---

# Phase 5: Formula Simulation

## 任务 9：比较核心公式

必须比较以下公式：

### A. surface baseline

```text id="q426p7"
priority = surface_keyness
```

### B. attention-only

```text id="7esqfe"
priority = attention_keyness
```

这是当前主 baseline。

### C. prompt-final output-effect only

```text id="2cnsq4"
priority = prompt_final_output_shift
```

### D. response-position output-effect only

```text id="6lyzfi"
priority = resp_pos_output_shift
```

### E. attention × prompt-final output-effect

```text id="h74b7j"
priority = attention_keyness * prompt_final_output_shift
```

### F. attention × response-position output-effect

```text id="on1iou"
priority = attention_keyness * resp_pos_output_shift
```

### G. response-position output gate then attention

```text id="yvdvsn"
if resp_pos_output_shift < threshold:
    priority = low
else:
    priority = attention_keyness
```

### H. attention gate then response-position output

```text id="kixw00"
if attention_keyness < threshold:
    priority = low
else:
    priority = resp_pos_output_shift
```

### I. mean(attention, response-position output)

```text id="1lzbx3"
priority = mean(attention_keyness, resp_pos_output_shift)
```

### J. max(attention, response-position output)

```text id="aeu76y"
priority = max(attention_keyness, resp_pos_output_shift)
```

### K. conservative intersection

```text id="3pgqra"
priority = min(attention_keyness, resp_pos_output_shift)
```

Thresholds must be label-free:

```text id="4f7ss5"
within-question median
within-question top-50%
or train-fold median if folds are implemented
```

Do not tune thresholds using solution_path labels.

---

# Phase 6: Evaluation Metrics

## 任务 10：主指标

对所有公式计算：

```text id="dc3ryp"
same_question_on_path_vs_off_path_auc
same_question_pairwise_accuracy
on_path_number_rank_mean
off_path_number_rank_mean
num_questions_with_on_off_pairs
num_on_off_pairs
```

输出：

```text id="1j7api"
answer_position_ranking_report.json
```

---

## 任务 11：grouped bootstrap

对每个 gate-eligible formula 做 question-grouped bootstrap。

必须比较：

```text id="27onfo"
D_response_position_output_only vs C_prompt_final_output_only
D_response_position_output_only vs A_surface
D_response_position_output_only vs B_attention_only

F_attention_x_response_position_output vs B_attention_only
G_response_output_gate_then_attention vs B_attention_only
H_attention_gate_then_response_output vs B_attention_only
I_mean_attention_response_output vs B_attention_only
J_max_attention_response_output vs B_attention_only
K_min_attention_response_output vs B_attention_only
```

每个 bootstrap result 包含：

```json id="fjyb2y"
{
  "formula": "...",
  "baseline": "...",
  "metric": "same_question_on_path_vs_off_path_auc",
  "mean_delta": ...,
  "ci95_low": ...,
  "ci95_high": ...,
  "stable_positive": true,
  "num_bootstrap_samples": 1000,
  "num_questions_with_on_off_pairs": ...
}
```

输出：

```text id="n8g6i6"
answer_position_grouped_bootstrap_report.json
```

---

## 任务 12：budget metrics

计算：

```text id="l73wqn"
off_path_budget_share
top_k_off_path_number_selected_rate
top1_key_hit
top2_key_hit
top3_key_hit
span_type_budget_distribution
comparison_topk_coverage
negation_topk_coverage
condition_topk_coverage
operation_topk_coverage
question_target_topk_coverage
```

输出：

```text id="ynwqdi"
off_path_budget_report.json
```

注意：

```text id="bv0gqg"
top-k coverage 是辅助指标；
不得替代 same-question on/off-path AUC。
```

---

# Phase 7: Prompt-final vs Answer-position Comparison

## 任务 13：直接比较两个测点

输出：

```text id="lca0bd"
prompt_vs_answer_position_comparison.json
```

必须包含：

```json id="kqxmci"
{
  "prompt_final_output_effect": {
    "auc": ...,
    "vs_surface": {...},
    "vs_attention": {...}
  },
  "response_position_output_effect": {
    "auc": ...,
    "vs_surface": {...},
    "vs_attention": {...}
  },
  "delta_response_minus_prompt": {
    "auc_delta": ...,
    "bootstrap_ci95": [...]
  },
  "interpretation": "..."
}
```

必须明确回答：

```text id="mxzjp7"
1. response-position 是否显著强于 prompt-final？
2. prompt-final output-effect 是否确实被低估？
3. response-position 是否接近或超过 attention-only？
```

---

# Phase 8: Span-Type Breakdown

## 任务 14：按 span type 拆解 response-position output-effect

对每类 span 统计：

```text id="ov3w02"
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

每类输出：

```json id="e7qtxp"
{
  "span_type": "...",
  "num_cases": ...,
  "mean_attention_keyness": ...,
  "mean_prompt_final_output_shift": ...,
  "mean_resp_pos_output_shift": ...,
  "resp_minus_prompt_mean_delta": ...,
  "topk_selected_rate_by_resp_pos": ...,
  "topk_selected_rate_by_attention": ...,
  "off_path_frac_if_number_like": ...
}
```

输出：

```text id="o0gkys"
span_type_answer_position_breakdown.json
```

必须回答：

```text id="23oq1n"
1. response-position output-effect 是否对 number_unit / rate 更敏感？
2. 是否对 question_target 更敏感？
3. 是否改善 comparison / condition / negation 的识别？
4. 是否仍然抬高 off-path number？
5. object 是否被过度选择？
```

---

# Phase 9: Failure and Success Cases

## 任务 15：输出 case audit

输出至少：

```text id="dshaxo"
30 cases where response-position output fixes prompt-final output
30 cases where response-position output beats attention
30 cases where attention beats response-position output
30 cases where both attention and response-position output fail
30 cases where attention × response-position output succeeds
```

每个 case 包含：

```json id="jxvnk9"
{
  "source_question_id": "...",
  "question": "...",
  "candidate_spans": [
    {
      "span_text": "...",
      "span_type": "...",
      "solution_path_status_eval_only": "...",
      "attention_score": ...,
      "prompt_final_output_shift": ...,
      "resp_pos_output_shift": ...,
      "formula_scores": {},
      "ranks": {}
    }
  ],
  "case_type": "...",
  "auto_interpretation": "..."
}
```

case_type 包括：

```text id="jsaok2"
resp_pos_fixes_prompt_final
resp_pos_beats_attention
attention_beats_resp_pos
both_fail
attention_resp_pos_fusion_succeeds
off_path_number_overranked
non_number_suppressed
object_overselected
unknown
```

输出：

```text id="jc4o7c"
failure_case_report.jsonl
success_case_report.jsonl
```

---

# Phase 10: Review Gate

## 2K-W Gate

本 sprint 是 diagnostic sprint，不以 2000-ready 为目标。

通过条件：

```text id="8kxogw"
1. answer-position output-effect features generated；
2. feature leakage audit passed；
3. same-question on/off-path AUC reported for all formulas；
4. response-position output-effect compared against prompt-final output-effect；
5. response-position output-effect compared against attention-only；
6. grouped bootstrap computed with question-level resampling；
7. budget metrics generated；
8. span-type breakdown generated；
9. failure/success cases generated；
10. ready_for_2000_rerun=False；
11. do_not_enter_sprint_3A=True。
```

不要要求 response-position output-effect 一定超过 attention-only。
本 sprint 目的是验证测点是否改善。

---

## Success Criteria

### Strong success

```text id="j70koo"
1. response-position output-effect stable > prompt-final output-effect；
2. response-position output-effect stable > surface；
3. attention × response-position output-effect stable > attention-only；
4. off_path_budget_share decreases；
5. non-number key spans are not suppressed。
```

### Partial success

```text id="y4p6tm"
1. response-position output-effect > prompt-final output-effect；
2. response-position output-effect > surface；
3. 但仍未超过 attention-only；
4. attention × response-position output-effect does not stably improve attention-only。
```

### Negative result

```text id="jge5g0"
1. response-position output-effect ≈ prompt-final output-effect；
2. output-effect still not stable above surface；
3. attention-only remains best；
4. fusion does not help。
```

Negative result 不代表项目失败，而是说明：

```text id="tw127o"
当前最强可用 target selector 仍是 attention-only。
```

---

# Final Report Requirements

`review_gate_answer_position_output_effect.md` 必须包含：

```text id="ons20h"
Verdict:
- ...

Input and boundary:
- ...

Feature leakage audit:
- ...

Primary AUC:
- surface:
- attention-only:
- prompt-final output-effect:
- response-position output-effect:
- attention × response-position output-effect:

Prompt-final vs response-position:
- ...

Grouped bootstrap:
- ...

Budget analysis:
- ...

Span-type breakdown:
- ...

Failure/success cases:
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

```text id="t89jni"
1. response-position output-effect 是否强于 prompt-final output-effect？
2. prompt-final output-effect 是否被低估？
3. response-position output-effect 是否稳定优于 surface？
4. response-position output-effect 是否稳定超过 attention-only？
5. attention × response-position output-effect 是否稳定超过 attention-only？
6. response-position output-effect 是否减少 off-path number budget？
7. response-position output-effect 是否改善 comparison / negation / condition / question_target？
8. 如果 output-effect 不如 attention-only，是否建议用 attention-only 作为 steering target？
9. 是否建议继续做 formula validation？
10. 是否建议进入 2000？
11. 是否建议进入 Sprint 3A？
```

---

# PROGRESS.md 更新要求

更新 PROGRESS.md 当前阶段，必须写清：

```text id="x95ryg"
1. 本轮是 answer-position output-effect re-measurement，不是 scale-up；
2. 复用 2J-Fix/2K/2K-V 4935-span matrix；
3. response-position output-effect 是否强于 prompt-final；
4. response-position output-effect 是否超过 attention-only；
5. attention × response-position output-effect 是否超过 attention-only；
6. 当前推荐 steering target 是 attention-only 还是 attention+output-effect；
7. ready_for_2000_rerun=False；
8. do_not_enter_sprint_3A=True。
```

同时更新：

```text id="m7j03u"
docs/progress/sprint_2_history.md
```

---

# 关键提醒

本 sprint 不是为了加入新概念，而是把已有 output-effect 测准。

当前核心假设是：

```text id="096cix"
prompt-final output-effect 可能离答案太远；
answer-position output-effect 可能更接近真正 reasoning consequence。
```

如果该假设成立，后续 steering target 可以从：

```text id="8vbp5q"
attention-only
```

升级为：

```text id="d7q6hd"
attention × response-position output-effect
```

如果该假设不成立，则下一步应以：

```text id="l3p4f0"
attention-only
```

作为 3A-0 attention steering smoke test 的 target selector。

无论结果如何，本 sprint 后仍然不要直接进入正式 3A；最多进入 3A-0 smoke test。
