# Sprint 3C-0: Correct-vs-Wrong Activation Patching at Reasoning-Step Positions

## 定位

本 sprint 是 3A / 3B 连续负结果之后的 **causal localization sprint**。

它不是 full Sprint 3C，不是 2000-scale rerun，不是训练任务，不是 LoRA / finetuning，不是 hallucination reduction 实验，也不允许声称 answer accuracy improvement。

本轮只回答一个问题：

```text id="vku78b"
如果 final answer-position 的 span-level intervention 无选择性收益，
那么模型真正出错的位置是否更早发生在 reasoning step 中？
```

换句话说，本 sprint 不再继续问：

```text id="oqocq3"
把关键 span 注入最终 Answer: 位置有没有用？
```

而是改问：

```text id="6ve65j"
把正确 run 的 activation patch 到错误 run 的具体 reasoning-step 位置，
能否让错误 run 的后续答案分布更接近 gold？
```

默认结论标记：

```text id="hfq5ep"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 执行前必读

执行前必须先阅读：

```text id="8v4tqq"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_3_history.md

docs/codex_tasks/sprint_3A_1_controlled_attention_guidance_500.md
docs/codex_tasks/sprint_3B_0_representation_level_oracle_intervention_diagnostic.md

src/recover_attention/attention_bias_steering.py
src/recover_attention/representation_intervention.py

scripts/sprint_3A_1_controlled_attention_guidance.py
scripts/sprint_3B_0_representation_level_oracle_intervention.py
```

---

## Preflight：必须先修上一轮遗留问题

在开始 3C-0 实验前，必须先做一个 preflight cleanup，并输出：

```text id="o92ybg"
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/preflight_report.md
```

### Fix 1：修正 PROGRESS.md 顶部 current status

如果 `PROGRESS.md` 顶部仍然是：

```text id="6w5xfu"
## Current Status Update: Sprint 3A-0 Attention Bias Steering Smoke Test
```

必须改成当前状态，例如：

```text id="i1ldh2"
## Current Status Update: Sprint 3B-0 Representation-Level Oracle Intervention Diagnostic
```

或者如果已经正式进入 3C-0：

```text id="dtdtxk"
## Current Status Update: Sprint 3C-0 Correct-vs-Wrong Activation Patching
```

要求：

```text id="i13hc1"
- 不要让 PROGRESS.md 顶部 current status 停留在 3A-0；
- 3A-0、3A-1、3B-0 应放到历史段落；
- 当前阶段必须明确写 3B-0 已完成、3C-0 正在执行或已完成。
```

### Fix 2：补齐 3A-1 / 3B-0 聚合报告的可追踪性

上一轮审查发现，`docs/progress/sprint_3_history.md` 和 `PROGRESS.md` 记录了 3A-1 / 3B-0 的输出文件，但 GitHub 中可能没有提交这些 output reports。

preflight 必须检查以下文件是否存在：

```text id="btoz9b"
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/oracle_sanity_diagnostic_report.json
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/answer_position_output_shift_report_500.json
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/harm_rate_report_500.json
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/baseline_comparison_report_500.json
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/review_gate_controlled_attention_guidance_500.md

outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/representation_intervention_config.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/representation_intervention_fidelity_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/gold_logprob_delta_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/oracle_vs_random_diagnostic_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/selector_comparison_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/harm_rate_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/review_gate_representation_level_oracle_diagnostic.md
```

如果本地存在但未被 Git 追踪，优先提交这些小型聚合报告。

如果 outputs 被 `.gitignore` 忽略，不要强行提交巨大 jsonl；可以新增一个轻量 summary manifest：

```text id="y774ss"
docs/progress/sprint_3_artifact_manifest.md
```

其中记录：

```text id="9wqq1q"
- 文件名
- 是否存在
- 是否提交到 Git
- 若未提交，原因
- 对应 sprint 的关键结论摘要
```

如果本地也不存在，不要伪造结果；在 preflight report 中诚实记录：

```text id="if5b9a"
3A-1 / 3B-0 detailed output reports are missing locally or ignored by Git.
Current conclusions are recoverable from PROGRESS.md and docs/progress/sprint_3_history.md,
but raw aggregate reports are not available for direct audit.
```

### Fix 3：确认 3B-0 结论边界

preflight report 必须明确写：

```text id="n40v60"
3B-0 否定的是：
same-run span residual deviation → final answer-position injection → single-forward gold-token proxy

3B-0 没有否定：
correct-run activation patching
reasoning-step-level patching
value / MLP causal tracing
autoregressive generation steering
```

这条边界必须写清楚，避免后续误读为“representation-level 全部失败”。

---

## 前因后果：为什么要做 3C-0

### 1. 3A-1 说明 attention-logit bias 不产生选择性 answer benefit

3A-1 修正了 3A-0 的 oracle 诊断：

```text id="n9jo4x"
target_attention_mass_delta
→ gold_first_token_logprob_delta
```

并通过 λ sweep 找到了 output shift 可测的 regime。

但结果是：

```text id="bl8d0u"
oracle on-path span 不比 random span 更能提高 gold first-token logprob。
```

解释：

```text id="b9f87w"
把注意力加到正确 span 上，
并不能选择性地让答案更正确；
高 λ 下的 gold-logprob 上升更像是非选择性 sharpening + destabilization。
```

因此，attention-bias steering 不应继续 scale up。

---

### 2. 3B-0 说明 final answer-position residual injection 也不产生选择性 answer benefit

3B-0 换了通道：

```text id="jjg47v"
attention-logit bias
→ answer-position residual injection
```

干预强度已经足够，output JS 明显可测。

但结果仍然是：

```text id="wbv0c4"
oracle residual injection ≈ random residual injection
```

解释：

```text id="d1eet1"
在 final answer-position 注入 selected span representation，
无论 span 是否正确，都主要造成非选择性扰动；
它没有把 span keyness 转化为 causal answer steering。
```

因此，问题不只是 attention 通道失败，而可能是：

```text id="u3q1wv"
final answer-position 太晚；
span-level injection 太粗；
single-forward answer-position proxy 没有触及真正的 reasoning computation。
```

---

### 3. 3C-0 的新假设

新的机制假设是：

```text id="l0n6od"
错误可能不是在最终 Answer: 位置产生的，
而是在更早的 reasoning-step 位置产生的。
```

例如：

```text id="d4h3fs"
数字被读取
数字被组合
运算符被选择
中间结果被生成
最终答案被复制/压缩
```

所以本轮要从：

```text id="t7oet3"
span-level final answer-position intervention
```

转向：

```text id="x9ebk6"
correct-vs-wrong activation patching at reasoning-step positions
```

---

## 本轮核心问题

本 sprint 只回答：

```text id="ci1v20"
在 teacher-forced fixed trace 中，
把 correct run 的 activation patch 到 wrong run 的某个 reasoning-step position，
是否能让 wrong run 的后续 answer logits 更接近 gold？
```

如果有效，说明：

```text id="wojmx0"
模型内部存在可修复错误的 causal state；
只是 3A/3B 干预位置或通道不对。
```

如果无效，说明：

```text id="pnpato"
当前选取的 step position / layer / patch type 仍没有命中 causal bottleneck；
或者 GSM8K span-level guidance 任务本身不适合这种 steering。
```

---

## 严格边界

禁止：

```text id="m0rbyd"
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要 hard mask；
- 不要继续调 attention-bias 或 residual span injection；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 oracle / gold 当作可部署方法；
- 不要覆盖 3A-1 / 3B-0 输出目录；
- 不要把 teacher-forced proxy 结果写成 autoregressive generation 结果。
```

允许：

```text id="df5o62"
- 小规模采样 correct / wrong model traces；
- 使用 GSM8K gold answer 判断 trace 正误；
- 使用 gold answer 做 eval-only metric；
- 使用 correct run activation 作为 diagnostic donor；
- 对 wrong run 做 teacher-forced activation patching；
- 做 layer / position / patch type sweep；
- 做 paired bootstrap；
- 输出 review gate。
```

注意：

```text id="f6cc4l"
本 sprint 可以生成模型自己的 short reasoning trace，
但仅作为实验 artifact；
不得把它写成人类可解释的真实推理证明；
不得声称模型内部 reasoning 已被完全解释。
```

---

## 输入

优先使用：

```text id="kn443z"
data/raw/gsm8k_train_normalized.jsonl
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl

src/recover_attention/attention_bias_steering.py
src/recover_attention/representation_intervention.py
scripts/sprint_3B_0_representation_level_oracle_intervention.py
```

可选读取：

```text id="icbk3t"
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/selector_comparison_report.json
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/review_gate_representation_level_oracle_diagnostic.md
```

但不能依赖 3B-0 output files 作为唯一输入。

---

## 输出目录

```text id="v3tnba"
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/
```

必须输出：

```text id="ykbtgg"
preflight_report.md
trace_sampling_manifest.jsonl
correct_wrong_pair_manifest.jsonl
reasoning_step_position_report.json
activation_patching_config.json
activation_patching_forward_manifest.jsonl
patching_fidelity_report.json
patching_effect_report.json
oracle_patch_vs_control_report.json
layer_position_heatmap_report.json
harm_rate_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_correct_wrong_activation_patching.md
```

同时新增或更新：

```text id="fr342u"
docs/codex_tasks/sprint_3C_0_correct_wrong_activation_patching_reasoning_steps.md
src/recover_attention/activation_patching.py
scripts/sprint_3C_0_correct_wrong_activation_patching.py
tests/test_activation_patching.py
docs/progress/sprint_3_history.md
PROGRESS.md
```

---

## 实验设计

### Step 1：构造 correct / wrong trace pairs

在小规模 question subset 上采样模型输出。

建议：

```text id="py62mx"
primary_questions = 80 到 120
samples_per_question = 4 到 8
temperature = 0.7 或 0.8
max_new_tokens = 128 或 192
```

prompt 建议固定为：

```text id="wz87o6"
Question: {question}
Solve briefly step by step, then give the final answer as: #### <number>
```

解析每个 sample 的 final answer。

对每个 question，寻找：

```text id="lmy4so"
至少 1 条 correct trace
至少 1 条 wrong trace
```

形成 pair：

```text id="f7pbcr"
same_question_id
correct_trace_text
wrong_trace_text
correct_final_answer
wrong_final_answer
gold_answer
```

如果 same-question correct/wrong pairs 不足，允许降级：

```text id="m9zvmn"
primary_n 降到实际可用 pair 数；
不要强行扩大；
不要跨 question 配对。
```

必须报告：

```text id="g07697"
num_questions_sampled
num_traces_generated
num_questions_with_correct_and_wrong
num_pairs
correct_rate
wrong_rate
parse_failure_rate
```

---

### Step 2：识别 reasoning-step positions

对 correct trace 和 wrong trace 分别做 token-level position tagging。

位置类别至少包括：

```text id="n2upzo"
prompt_question_number_token
generated_operator_token
generated_intermediate_number_token
generated_equals_or_result_marker
generated_final_answer_prefix
generated_final_answer_number
final_answer_position
```

其中 primary positions 是：

```text id="wzswu2"
generated_operator_token
generated_intermediate_number_token
generated_equals_or_result_marker
generated_final_answer_number
```

control positions 是：

```text id="ebuxhz"
random_generated_token
final_answer_position
prompt_question_number_token
```

注意：

```text id="tm7qjr"
3A/3B 已经充分测试 final answer-position；
3C-0 必须把 reasoning-step positions 作为 primary，
final answer-position 只能作为 control。
```

---

### Step 3：teacher-forced activation patching

本轮先做 teacher-forced fixed-trace patching，不做 autoregressive generation。

对每个 correct/wrong pair：

```text id="s77iw1"
clean / donor = correct trace full text
corrupt / recipient = wrong trace full text
```

构造完整输入：

```text id="ks1ayb"
Question: {question}
{trace_text}
```

或保留采样时的原始 prompt + completion。

对 donor 和 recipient 分别 forward，保存 target layers 的 hidden states。

然后在 recipient forward 中 patch：

```text id="vsb1wi"
recipient_hidden[layer, recipient_position]
← donor_hidden[layer, donor_position]
```

或者使用 interpolation：

```text id="jnkm3t"
recipient_hidden =
(1 - alpha) * recipient_hidden + alpha * donor_hidden
```

primary 设置：

```text id="o3be2v"
patch_type = residual_replace
alpha = 1.0
layers = [12, 16, 20, 24, 28]
positions = reasoning-step primary positions + controls
```

如果计算量太大，优先：

```text id="kuco5y"
layers = [16, 20, 24]
positions = generated_operator_token / generated_intermediate_number_token / generated_final_answer_number / final_answer_position
```

---

## Patch 位置对齐规则

必须同一 question 内配对，不允许跨 question。

position matching 优先级：

```text id="wufdnr"
1. 同类别 role-to-role 对齐：
   correct generated_intermediate_number_token → wrong generated_intermediate_number_token

2. 若同类别有多个 token：
   使用相同 ordinal index，例如第 1 个 intermediate number 对第 1 个 intermediate number

3. 若 wrong trace 缺少对应类别：
   跳过该 pair 的该 position type，不要乱配

4. 若 correct trace 缺少对应类别：
   跳过该 pair 的该 position type

5. final_answer_position 只作为 control
```

必须输出 position coverage：

```text id="f1a9tv"
position_type
num_pairs_with_match
coverage_rate
```

---

## 指标定义

### 1. Patch fidelity

确认 hook 真的起作用：

```text id="m3skd2"
hook_registered
hook_triggered_layer
hook_removed
patched_position
patch_delta_norm
recipient_hidden_changed_norm
non_target_position_contamination_check
```

要求：

```text id="l7m5lq"
hook_ok_rate = 1.0
patch_delta_norm > 0
non_target_position_contamination_check = pass
```

---

### 2. Answer-directed proxy

核心指标：

```text id="ew8k5v"
gold_first_token_logprob_delta =
logprob(gold_first_token | patched wrong trace)
-
logprob(gold_first_token | unpatched wrong trace)
```

gold answer 只用于 eval-only metric。

---

### 3. Wrong-answer suppression proxy

记录 wrong answer token 的变化：

```text id="ah8dvc"
wrong_first_token_logprob_delta =
logprob(wrong_answer_first_token | patched)
-
logprob(wrong_answer_first_token | unpatched)
```

理想方向：

```text id="62z7ab"
gold_first_token_logprob_delta > 0
wrong_first_token_logprob_delta < 0
```

---

### 4. Clean direction score

定义：

```text id="qqqd8j"
clean_direction_score =
gold_first_token_logprob_delta
-
wrong_first_token_logprob_delta
```

这个指标比只看 gold token 更稳，因为它同时要求：

```text id="tvoc5u"
提高正确答案
压低错误答案
```

---

### 5. Control comparison

必须比较：

```text id="n7qoq3"
correct_activation_patch
random_donor_patch
same_trace_random_position_patch
final_answer_position_patch
no_patch
```

核心 paired comparison：

```text id="a5tbvl"
correct_reasoning_step_patch - random_donor_patch
correct_reasoning_step_patch - final_answer_position_patch
correct_reasoning_step_patch - random_position_patch
```

每个比较都要 question-paired bootstrap：

```text id="hnmjma"
mean
CI95 low
CI95 high
stable_positive = CI95 low > 0
```

---

## Primary success criterion

本 sprint 的 primary success 不是 accuracy improvement。

primary success 是：

```text id="g8zsba"
某些 reasoning-step position × layer 的 correct-run activation patch
在 clean_direction_score 上稳定优于 random/control patch。
```

具体要求：

```text id="spns88"
CI95 low > 0
coverage 足够
harm 可控
效果集中在 reasoning-step positions，而不是所有位置同幅度上升
```

如果所有位置都同幅度上升，仍判定为：

```text id="bufv81"
non-selective perturbation artifact
```

---

## Harm proxy

沿用并扩展：

```text id="y7t9w4"
harm = top1_changed_to_non_gold
    or entropy_delta > 1.0
    or margin_delta < -0.25
    or gold_logprob_delta < -0.5
```

注意：

```text id="q5tz6b"
如果 patch 让输出分布大幅变化但 clean_direction_score 不提升，
不能视为有效 steering。
```

---

## Review gate

`review_gate_correct_wrong_activation_patching.md` 必须逐条回答：

```text id="pl35m9"
1. Preflight 是否完成？
2. PROGRESS.md 顶部 current status 是否已修正？
3. 3A-1 / 3B-0 聚合报告是否存在或是否已有 artifact manifest？
4. 本 sprint 是否仍然非 full 3C、非 2000、非训练？
5. 采样了多少 question？
6. 生成了多少 traces？
7. 构造了多少 same-question correct/wrong pairs？
8. parse failure rate 是多少？
9. 覆盖了哪些 reasoning-step position types？
10. 哪些 layer 被 patch？
11. hook fidelity 是否通过？
12. 是否有任何 position × layer 的 correct patch 稳定优于 random donor patch？
13. 是否有任何 reasoning-step patch 稳定优于 final answer-position patch？
14. clean_direction_score 是否稳定为正？
15. 是否存在非选择性 perturbation artifact？
16. harm 是否可控？
17. 是否允许进入 2000 rerun？
18. 是否允许声称 hallucination reduction / answer accuracy improvement？
19. 下一步应该继续 reasoning-step patching、转 value/MLP tracing，还是暂停 steering？
```

默认最终答案：

```text id="v4xjqd"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 结果解释逻辑

### 情况 A：reasoning-step patch 显著优于 controls

如果出现：

```text id="sib9le"
correct reasoning-step patch > random donor patch
correct reasoning-step patch > final answer-position patch
clean_direction_score stable positive
harm controllable
```

说明：

```text id="tscedg"
3A/3B 失败不是因为模型内部没有可修复状态；
而是因为 final answer-position span-level intervention 没有命中真正 causal bottleneck。
```

下一步：

```text id="xul30f"
Sprint 3C-1: Localize Value/MLP Path at Effective Reasoning Steps
```

---

### 情况 B：correct patch 有效，但 only final answer-position 有效

说明：

```text id="sxsixq"
correct-run activation 确实可修复 wrong run，
但有效位置不在显式 reasoning-step；
可能是答案压缩/读取阶段，而不是中间计算阶段。
```

下一步：

```text id="ghxqzd"
围绕 final-answer compression 做 activation patching，
但不要回到 span-level residual injection。
```

---

### 情况 C：所有 correct patch 都不优于 controls

说明：

```text id="zkpulx"
当前 teacher-forced activation patching 仍未命中 causal mechanism；
可能是 trace alignment 不好、position 粒度不对、或任务不适合。
```

下一步：

```text id="n5c9nr"
转 value/MLP causal tracing，
或暂停 steering，回到 detection / diagnosis。
```

---

## 建议新增模块

```text id="uefaw6"
src/recover_attention/activation_patching.py
```

至少实现：

```text id="tq3bat"
base_forward_with_hidden()
register_activation_patch_hook()
remove_hooks()
patched_forward()
extract_trace_positions()
match_position_roles()
compute_clean_direction_score()
bootstrap_paired_delta()
```

---

## 建议新增脚本

```text id="4p4rul"
scripts/sprint_3C_0_correct_wrong_activation_patching.py
```

建议参数：

```bash id="y9vys3"
conda run -n recover_attention python scripts/sprint_3C_0_correct_wrong_activation_patching.py \
  --primary-questions 100 \
  --samples-per-question 6 \
  --temperature 0.7 \
  --max-new-tokens 160 \
  --layers 16 20 24 \
  --position-types generated_operator_token generated_intermediate_number_token generated_final_answer_number final_answer_position \
  --output-dir outputs/logs/sprint_3C_0_correct_wrong_activation_patching \
  --overwrite
```

---

## 建议新增测试

```text id="j6slgc"
tests/test_activation_patching.py
```

至少覆盖：

```text id="ea7bf0"
1. patch hook 注册、触发、移除；
2. patch 只改变 target position；
3. remove 后 baseline 恢复；
4. position role extraction 能识别 operator / number / final answer marker；
5. role-to-role matching 不跨 question；
6. gold answer 只用于 eval metric；
7. random donor control 不使用 gold；
8. clean_direction_score 计算正确；
9. bootstrap paired delta 输出 mean / CI；
10. preflight 能检测 PROGRESS current status 是否过期。
```

---

## 推荐运行命令

```bash id="mvv91q"
conda run -n recover_attention python -m pytest tests/test_activation_patching.py -q

conda run -n recover_attention python scripts/sprint_3C_0_correct_wrong_activation_patching.py \
  --primary-questions 100 \
  --samples-per-question 6 \
  --temperature 0.7 \
  --max-new-tokens 160 \
  --layers 16 20 24 \
  --position-types generated_operator_token generated_intermediate_number_token generated_final_answer_number final_answer_position \
  --output-dir outputs/logs/sprint_3C_0_correct_wrong_activation_patching \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 最后更新

完成后必须更新：

```text id="rwjzv6"
PROGRESS.md
docs/progress/sprint_3_history.md
```

如果本轮产生的 output reports 较小，应提交聚合报告和 review gate。

必须提交或记录：

```text id="h8p7ww"
preflight_report.md
review_gate_correct_wrong_activation_patching.md
patching_effect_report.json
oracle_patch_vs_control_report.json
layer_position_heatmap_report.json
harm_rate_report.json
```

大体积 trace manifest / forward manifest 可以不提交，但必须在 artifact manifest 中说明位置与是否存在。

最终不得写：

```text id="uwjm58"
hallucination reduced
accuracy improved
ready for 2000
```

除非后续单独做了 autoregressive generation correctness eval 且有统计报告支持。

本 sprint 最多只能声称：

```text id="xae52r"
correct-vs-wrong activation patching reveals / does not reveal a selective causal state
at specific reasoning-step positions under teacher-forced proxy evaluation.
```
