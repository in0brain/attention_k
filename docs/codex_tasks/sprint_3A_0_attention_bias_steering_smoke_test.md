# Sprint 3A-0: Attention Bias Steering Smoke Test

## 背景

Sprint 2J-Fix、2K、2K-V、2K-W 已完成。

当前结论：

```text
1. multi-span matrix 已经从 one-question-one-span 变成 one-question-many-spans；
2. attention-only 是稳定有效的 within-question keyness signal；
3. hidden 当前构造下是噪声，不进入 steering target；
4. prompt-final output-effect 低估了 output-level consequence；
5. response-position output-effect 明显优于 prompt-final output-effect；
6. output-effect alone 未稳定超过 attention-only；
7. attention × response-position output-effect 稳定优于 attention-only；
8. 当前最合理的 target selector 是 attention + response-position output-effect；
9. 但目前仍只证明了 span ranking / target selection，没有证明 attention intervention 有效；
10. 未证明 answer accuracy improvement，也未证明 hallucination reduction。
```

本 sprint 是从 diagnostic ranking 走向 attention intervention 的最小 smoke test。

本 sprint 不是正式 Sprint 3A，不是完整 attention steering 系统。

---

## 核心问题

本 sprint 回答：

```text
1. 我们能不能在 Qwen forward 中可控地改变目标 span 的 attention mass？
2. 小幅 positive attention bias 是否真的让 selected key spans 被更多读取？
3. attention × response-position output-effect 选出的 span，是否比 random / surface 更适合 boost？
4. oracle on-path span boost 是否提供正向 sanity check？
5. boost key span 是否会系统性破坏原本正确的输出？
6. 当前 intervention 机制是否值得进入正式 3A-1？
```

---

## 本 sprint 的定位

这是：

```text
3A-0 = steering feasibility / smoke test
```

不是：

```text
full Sprint 3A = attention guidance system
```

因此本轮通过不代表：

```text
ready_for_2000_rerun=True
```

也不代表：

```text
hallucination_reduction_proven=True
```

默认：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3A=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

# 严格边界

## 允许

```text
读取 2J-Fix / 2K-W artifacts；
读取 4935-span score matrix；
选择小规模 subset；
实现 inference-time attention logit bias；
只做 positive boost；
记录 before/after attention mass；
记录 before/after answer-position logits；
比较 no steering / random / surface / attention / attention×response-output / oracle；
做 small-scale generation 或 answer-position next-token eval；
输出 steering fidelity、output shift、harm rate、case audit；
新增 tests；
更新 PROGRESS.md 和 sprint_2_history.md。
```

## 禁止

```text
不要扩大到 2000；
不要进入正式 Sprint 3A；
不要做 full-scale steering；
不要做 attention decrease / suppress distractors；
不要 hard mask distractor spans；
不要强制重分配全部 attention；
不要训练模型参数；
不要 LoRA / finetuning；
不要用 gold answer 构造 target selector；
不要用 oracle 作为真实方法；
不要把 oracle result 混入 non-oracle performance；
不要声称 hallucination reduction；
不要声称 answer accuracy improvement，除非本轮明确做了受控生成实验并报告统计；
不要生成 CoT；
不要使用 trajectory / NLA / causal patching；
不要重跑 recovery；
不要把 hidden/simple-average fusion 重新作为主线。
```

---

# Phase 0: Inputs and Output Directory

## 输入 artifacts

优先读取：

```text
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_score_matrix.jsonl

outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_ranking_report.json
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_formula_validation_report.json
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_grouped_bootstrap_report.json
outputs/logs/sprint_2K_W_answer_position_output_effect_500/review_gate_answer_position_output_effect.md
```

如果 2K-W summary reports 没有 commit，但 `answer_position_score_matrix.jsonl` 存在，则从 score matrix 重建 formula scores。

不要依赖 PROGRESS.md 作为唯一数据源。

---

## 输出目录

```text
outputs/logs/sprint_3A_0_attention_bias_steering_smoke_test/
```

必须输出：

```text
steering_subset_manifest.jsonl
target_selector_report.json
attention_bias_config.json
steering_forward_manifest.jsonl
attention_mass_fidelity_report.json
answer_position_output_shift_report.json
steering_generation_report.json
oracle_sanity_report.json
harm_rate_report.json
baseline_comparison_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_attention_bias_smoke_test.md
```

可选输出：

```text
attention_mass_before_after.jsonl
generation_outputs.jsonl
debug_hook_trace.jsonl
```

不要提交过大的 tensor cache。
如果产生大 tensor，只在 manifest 里记录本地路径。

---

# Phase 1: Subset Selection

## 任务 1：选择小规模 subset

本轮只跑小规模：

```text
primary_n = 50
optional_n = 100
```

优先选择满足以下条件的问题：

```text
1. 有 at least one on-path number；
2. 有 at least one off-path number；
3. 有至少 4 个 candidate spans；
4. 2K-W 中 attention × response-position output-effect 有有效 score；
5. 原始模型可以正常 forward；
6. 不包含过长 prompt。
```

建议 subset 结构：

```text
50 questions:
- 30 个 attention×resp_pos selector 排序成功的问题；
- 10 个 attention-only 成功但 fusion 不明显的问题；
- 10 个 selector 失败或 ambiguous 的问题。
```

目的不是 cherry-pick，而是保证 smoke test 覆盖：

```text
success cases
failure cases
ambiguous cases
```

输出：

```text
steering_subset_manifest.jsonl
```

每条记录包含：

```json
{
  "source_question_id": "...",
  "question": "...",
  "num_candidate_spans": 0,
  "has_on_path_number": true,
  "has_off_path_number": true,
  "subset_reason": "fusion_success|attention_only_success|selector_failure|ambiguous",
  "selected_for_primary": true
}
```

---

# Phase 2: Target Selector Construction

## 任务 2：构造 steering target selectors

对每道题构造以下 selector。

### S0: no steering

```text
不加任何 attention bias。
```

### S1: random span boost

```text
从 candidate spans 中随机选 top-k 个 span。
```

要求固定 seed。

用途：

```text
negative control
```

### S2: surface span boost

```text
按 surface_keyness 排序选 top-k。
```

用途：

```text
surface baseline
```

### S3: attention-only span boost

```text
按 attention_keyness 排序选 top-k。
```

用途：

```text
当前 strongest single signal baseline
```

### S4: attention × response-position output-effect boost

```text
按 attention_keyness × resp_pos_output_shift 排序选 top-k。
```

用途：

```text
当前推荐 selector
```

### S5: oracle on-path span boost

```text
用 eval-only solution_path_status 选择 on-path number spans。
```

用途：

```text
upper-bound sanity check only
```

严格要求：

```text
S5 oracle 只能用于 sanity check；
不能作为真实方法；
不能和 non-oracle selector 混合报告为方法提升。
```

---

## 任务 3：top-k 设置

至少测试：

```text
top_k = 1
top_k = 2
top_k = 3
```

如果 top_k 大于可用 span 数，则使用全部 candidate spans。

输出：

```text
target_selector_report.json
```

包含：

```json
{
  "selectors": ["random", "surface", "attention_only", "attention_x_resp_pos", "oracle"],
  "top_k_values": [1, 2, 3],
  "num_questions": 50,
  "oracle_is_eval_only": true,
  "hidden_excluded": true,
  "decrease_disabled": true
}
```

---

# Phase 3: Attention Bias Intervention Design

## 任务 4：只做 positive boost，不做 decrease

本轮只允许：

```text
increase attention to selected key spans
```

禁止：

```text
decrease attention to off-path spans
hard mask distractors
zero out attention
renormalize probability manually
```

理由：

```text
decrease 更容易破坏语义和句法上下文；
第一轮只验证 increase-only feasibility。
```

---

## 任务 5：使用 additive attention logit bias

目标是在 softmax 前修改 attention score：

```text
attention_scores = QK^T / sqrt(d)
attention_scores += causal_mask
attention_scores += guidance_bias
attention_probs = softmax(attention_scores)
```

对 selected span 的 key token 添加正 bias：

```text
guidance_bias[:, query_tokens, selected_key_tokens] += lambda
```

注意：

```text
这是 additive logit bias；
不是直接修改 attention probability；
不是 hard attention replacement。
```

---

## 任务 6：bias strength grid

测试小范围 lambda：

```text
lambda ∈ {0.05, 0.1, 0.2, 0.4}
```

可选：

```text
lambda = 0.8
```

但 0.8 只作为 aggressive diagnostic，不作为 primary。

主设置：

```text
lambda = 0.1 或 0.2
```

---

## 任务 7：layer grid

优先测试：

```text
layers = [16]
layers = [24]
layers = [16, 24]
```

如果模型层数或已用层不同，则读取模型 config 并映射到：

```text
mid layer
late-mid layer
mid + late-mid
```

不要默认使用最后一层。
原因：

```text
最后层 attention 可能不稳定，且更接近输出投影前的局部决策；
先使用中后层 smoke test。
```

---

## 任务 8：query token scope

优先测试三种 query scope：

### Q0: answer-position query

```text
只在 Answer: 后预测 next token 的最后输入位置作为 query。
```

这是 primary query scope。

### Q1: question-target query

```text
question target tokens，例如 how many / how much / total / left / cost。
```

作为 diagnostic。

### Q2: operation query

```text
operation tokens，例如 total / each / per / more than / left / remaining。
```

作为 diagnostic。

第一版 primary 设置：

```text
query_scope = answer_position
```

原因：

```text
2K-W 已经证明 answer-position output-effect 更接近 reasoning consequence。
```

---

## 任务 9：head scope

第一版可以先做：

```text
all heads
```

也就是对指定 layer 的所有 heads 加同样 key bias。

但必须记录：

```text
head_scope = all_heads
```

不要先做复杂 head selection。

如果工程允许，可以附加 diagnostic：

```text
top attention heads only
```

但不作为 primary。

---

# Phase 4: Implementation Requirements

## 任务 10：实现 attention bias hook

新增模块建议：

```text
src/recover_attention/attention_bias_steering.py
```

新增 CLI：

```text
scripts/sprint_3A_0_attention_bias_smoke_test.py
```

模块功能：

```text
1. build_guidance_bias(...)
2. register_attention_bias_hooks(...)
3. remove_attention_bias_hooks(...)
4. run_steered_forward(...)
5. compute_attention_mass_before_after(...)
6. compute_answer_position_output_shift(...)
7. run_3a0_smoke_test(...)
```

---

## 任务 11：优先使用 safe hook

实现策略：

### Option A: attention mask additive bias

如果 Qwen forward 支持传入 additive attention mask 且 shape 可表达：

```text
[batch, heads or 1, query_len, key_len]
```

则优先使用 attention mask bias。

### Option B: monkey patch attention forward

如果 attention mask 不支持 span-specific bias，则 monkey patch attention module，在 attention scores softmax 前加入 guidance bias。

要求：

```text
1. hook 必须可撤销；
2. 每次 forward 后 remove hook；
3. no-steering forward 不应受污染；
4. 多 selector / 多 lambda 之间不能共享残留 bias；
5. debug trace 记录 hook 是否触发。
```

输出：

```text
debug_hook_trace.jsonl
```

每条记录包含：

```json
{
  "source_question_id": "...",
  "selector": "attention_x_resp_pos",
  "lambda": 0.2,
  "layers": [16, 24],
  "hook_registered": true,
  "hook_triggered_layers": [16, 24],
  "hook_removed": true,
  "warnings": []
}
```

---

## 任务 12：token alignment

必须把 selected span char ranges 映射到 answer-position prompt 中的 token indices。

注意：

```text
2J-Fix 的 slot alignment bug 不得复现。
```

要求：

```text
1. original question span char ranges 映射到 response prompt；
2. response prompt 前缀 "Question: " 会改变 char offset；
3. 必须重新计算 response prompt 内的 char ranges；
4. 不能复用旧 prompt 的 slot_indices；
5. 对 multi-token span，bias 应加到所有 key tokens。
```

新增 regression test：

```text
同一道题两个不同 span，在 response prompt 中必须映射到不同 key token indices。
```

---

## 任务 13：保持 no-steering baseline

每个问题必须先跑 no-steering：

```text
original no steering forward
```

再跑 steered variants。

要求：

```text
no-steering output logits
no-steering attention mass
```

作为所有 delta 的 baseline。

---

# Phase 5: Primary Metrics

## 任务 14：steering fidelity

这是本 sprint 最重要的 metric。

回答：

```text
加 bias 后，目标 span 的 attention mass 真的增加了吗？
```

计算：

```text
target_attention_mass_before
target_attention_mass_after
target_attention_mass_delta
non_target_attention_mass_delta
```

按以下维度聚合：

```text
selector
top_k
lambda
layer_config
query_scope
```

输出：

```text
attention_mass_fidelity_report.json
```

通过倾向：

```text
target_attention_mass_delta > 0
target_attention_mass_delta 显著大于 random/noise
non_target_attention_mass_delta 不出现异常大幅崩坏
```

注意：

```text
如果 fidelity 不通过，后续 output/accuracy 指标都不能解释。
```

---

## 任务 15：answer-position output shift

回答：

```text
boost selected spans 后，answer-position next-token distribution 是否发生有意义变化？
```

计算 no-steering vs steered：

```text
steer_output_kl
steer_output_js
steer_top1_changed
steer_topk_overlap
steer_entropy_delta
steer_margin_delta
```

还要记录：

```text
original target selector 的 response-position output-effect
steering 后 output shift
```

输出：

```text
answer_position_output_shift_report.json
```

注意：

```text
这里仍然只是 output distribution change；
不是 answer accuracy proof。
```

---

## 任务 16：oracle sanity

比较：

```text
random boost
surface boost
attention-only boost
attention×resp_pos boost
oracle boost
```

核心 sanity logic：

```text
如果 oracle boost 都不能提高 fidelity 或改善 output signal，
说明 intervention mechanism 本身可能无效。
```

输出：

```text
oracle_sanity_report.json
```

必须包含：

```json
{
  "oracle_attention_mass_delta": ...,
  "predicted_attention_mass_delta": ...,
  "random_attention_mass_delta": ...,
  "oracle_output_shift_summary": {},
  "interpretation": "oracle_effective|oracle_not_effective|inconclusive"
}
```

---

## 任务 17：harm rate

不要只看是否修复错误，也要看是否破坏正确。

如果本轮做 generation，则计算：

```text
original_correct_and_steered_wrong
original_wrong_and_steered_correct
original_correct_and_steered_correct
original_wrong_and_steered_wrong
```

如果本轮不做 full generation，则用 answer-position proxy：

```text
no-steering top1 token changed
margin collapse
entropy explosion
top-k distribution unstable
```

输出：

```text
harm_rate_report.json
```

至少包含：

```json
{
  "selector": "...",
  "lambda": 0.2,
  "harm_proxy_rate": ...,
  "top1_changed_rate": ...,
  "entropy_explosion_rate": ...,
  "margin_collapse_rate": ...
}
```

---

# Phase 6: Optional Generation Evaluation

## 任务 18：小规模 generation eval，可选但建议

如果工程成本可控，在 subset 上做 deterministic generation：

```text
temperature = 0
max_new_tokens = 64 或 128
```

比较：

```text
no steering
random boost
attention×resp_pos boost
oracle boost
```

不要生成 CoT。
prompt 使用：

```text
Question: {question}
Answer:
```

只要求 final answer。

评估：

```text
exact numeric answer match
```

输出：

```text
steering_generation_report.json
generation_outputs.jsonl
```

注意：

```text
generation eval 是 exploratory；
样本小，不得声称最终 answer accuracy improvement。
```

---

# Phase 7: Baseline Comparison

## 任务 19：baseline comparison report

输出：

```text
baseline_comparison_report.json
```

按 selector / lambda / top_k / layer / query_scope 汇总：

```text
steering_fidelity
output_shift
harm_proxy
generation_accuracy_if_available
```

必须包含：

```text
no steering
random
surface
attention-only
attention×resp_pos
oracle
```

推荐重点比较：

```text
attention×resp_pos vs random
attention×resp_pos vs surface
attention×resp_pos vs attention-only
oracle vs no steering
oracle vs predicted
```

---

# Phase 8: Failure and Success Cases

## 任务 20：case audit

输出至少：

```text
20 cases where attention×resp_pos boost increases target attention mass cleanly
20 cases where bias hook changes attention but output barely changes
20 cases where oracle boost helps but predicted boost does not
20 cases where predicted boost beats random/surface
20 cases where steering appears harmful
```

每个 case 包含：

```json
{
  "source_question_id": "...",
  "question": "...",
  "selector": "attention_x_resp_pos",
  "top_k": 2,
  "lambda": 0.2,
  "layers": [16, 24],
  "query_scope": "answer_position",
  "selected_spans": [
    {
      "span_text": "...",
      "span_type": "...",
      "attention_keyness": ...,
      "resp_pos_output_shift": ...,
      "selector_score": ...,
      "solution_path_status_eval_only": "..."
    }
  ],
  "attention_mass_before": ...,
  "attention_mass_after": ...,
  "attention_mass_delta": ...,
  "output_shift": {},
  "no_steering_output": "...",
  "steered_output": "...",
  "case_type": "...",
  "auto_interpretation": "..."
}
```

输出：

```text
failure_case_report.jsonl
success_case_report.jsonl
```

---

# Phase 9: Review Gate

## 3A-0 Gate

本 sprint 是 steering smoke test，不是正式 3A。

通过条件：

```text
1. subset manifest generated；
2. target selectors generated；
3. attention bias hook implemented and removable；
4. no-steering baseline preserved；
5. positive boost only；
6. no decrease / no hard mask；
7. steering fidelity report generated；
8. oracle sanity report generated；
9. harm rate report generated；
10. baseline comparison report generated；
11. failure/success cases generated；
12. ready_for_2000_rerun=False；
13. do_not_enter_full_sprint_3A=True。
```

不要要求 generation accuracy 一定提升。
不要要求 oracle 必须显著提升 accuracy。
本轮主要证明 intervention 可控性。

---

## Success Criteria

### Strong success

```text
1. attention bias hook reliably increases target span attention mass；
2. attention×resp_pos selector has higher fidelity or better output-shift profile than random/surface；
3. oracle boost shows stronger or cleaner effect than random；
4. harm proxy is low at small lambda；
5. attention×resp_pos is at least not worse than attention-only；
6. results support 3A-1 controlled 500-case steering eval。
```

### Partial success

```text
1. hook works and target mass increases；
2. oracle boost works；
3. predicted selector is not clearly better than attention-only/random；
4. no major harm；
5. next step should improve selector/query/layer configuration before 3A-1。
```

### Negative result

```text
1. target attention mass does not increase；
2. hook unreliable or not triggered；
3. oracle boost does not produce meaningful changes；
4. small lambda already causes high harm；
5. attention intervention mechanism should be redesigned before any steering sprint。
```

---

# Final Report Requirements

`review_gate_attention_bias_smoke_test.md` 必须包含：

```text
Verdict:
- ...

Boundary:
- increase-only:
- no decrease:
- no hard mask:
- no training:
- no 2000:
- no full 3A:

Subset:
- num_questions:
- selection policy:

Selectors:
- random:
- surface:
- attention-only:
- attention×resp_pos:
- oracle:

Intervention:
- bias type:
- lambda grid:
- layers:
- query scope:
- head scope:

Steering fidelity:
- ...

Oracle sanity:
- ...

Output shift:
- ...

Harm rate:
- ...

Generation eval if available:
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

## Final report 必须回答的问题

```text
1. attention bias 是否真的改变了目标 span 的 attention mass？
2. hook 是否可靠、可撤销、不会污染 no-steering baseline？
3. attention×resp_pos selector 是否优于 random/surface？
4. attention×resp_pos selector 是否优于 attention-only？
5. oracle boost 是否有效？
6. 如果 oracle boost 无效，是否说明 intervention 机制失败？
7. 小 lambda 是否安全？
8. 是否观察到明显 harm？
9. 是否做了 generation eval？如果做了，是否只是 exploratory？
10. 是否建议进入 3A-1？
11. 是否建议进入 2000？
12. 是否建议进入正式 Sprint 3A？
```

---

# PROGRESS.md 更新要求

更新 PROGRESS.md 当前阶段，必须写清：

```text
1. 本轮是 3A-0 smoke test，不是 full Sprint 3A；
2. 使用小规模 subset；
3. 只做 increase-only positive attention bias；
4. 没有做 decrease / hard mask；
5. attention bias hook 是否可靠；
6. target attention mass 是否真的增加；
7. oracle boost 是否有效；
8. attention×resp_pos 是否优于 random/surface/attention-only；
9. harm rate 是否可接受；
10. 是否建议进入 3A-1；
11. ready_for_2000_rerun=False；
12. do_not_enter_full_sprint_3A=True。
```

同时更新：

```text
docs/progress/sprint_2_history.md
```

如果决定开启 3A-1，再新增单独 card。
不要直接把 3A-0 结果解释成正式 steering 成功。

---

# 关键提醒

本 sprint 的目标不是证明模型 accuracy 提升，而是验证：

```text
Can we safely and controllably steer attention?
```

最关键的判据是：

```text
1. boost 是否真的改变目标 span attention；
2. oracle boost 是否作为 sanity check 有效；
3. predicted selector 是否优于 random/surface；
4. 小幅 boost 是否低伤害。
```

如果这四件事不成立，不要继续推进正式 3A。

如果成立，下一步才是：

```text
Sprint 3A-1: Controlled Attention Guidance on 500 Cases
```

而不是直接 2000-scale rerun。
