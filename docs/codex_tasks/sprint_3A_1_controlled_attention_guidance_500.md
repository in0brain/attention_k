# Sprint 3A-1: Controlled Attention Guidance on 500 Cases

## 定位

本 sprint **不是** full Sprint 3A，**不是** 2000-scale rerun，**不允许**声称 hallucination reduction 或 answer accuracy improvement（除非 generation eval 有统计报告支持）。

只做 **increase-only positive attention logit bias**（复用 3A-0 hook）。默认：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3A=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

## 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/codex_tasks/sprint_3A_0_attention_bias_steering_smoke_test.md
src/recover_attention/attention_bias_steering.py
tests/test_attention_bias_steering.py
```

## 背景（3A-0 为什么 oracle inconclusive）

3A-0 实测（primary config λ=0.2, layers 16+24, query=answer_position, N=50）：

```text
oracle_attention_mass_delta   = 0.0021
random_attention_mass_delta   = 0.0036   # oracle 甚至低于 random
predicted_attention_mass_delta= 0.0048
oracle_output_shift_js        ≈ 9.3e-6   # 输出几乎不动
random_output_shift_js        ≈ 2.1e-5
top1_changed_rate             ≈ 0.02
```

两个根因：

1. **oracle 指标用错了**：3A-0 的 oracle 判定基于 `target_attention_mass_delta`（有没有把 mass 提到「该 selector 自己选的」span 上）。这对任何 selector（包括 random）都近似恒正，且 oracle 与 random 目标不同 → 二者不可比。oracle 与 random 比 mass-delta 没有意义。
2. **干预太弱**：λ=0.2 / 2 层下 answer-position 输出几乎不动（JS~1e-5），所以所有比较都是在比 0。

## 本轮目标（含补充）

1. 在 eligible 500-case 子集（2K-W 有 232 题满足 ≥4 spans 且含 on+off-path number）上复用 3A-0 attention-bias hook。
2. **[补充] 修正 oracle sanity diagnostic 为 answer-directed（eval-only）**：oracle 是否把 answer-position 分布推向 **gold answer 首 token** 比 random / no-steering 更多。gold answer 只用于 eval-only 指标，绝不进入任何 non-oracle selector。
3. **[补充] intervention-strength regime sweep**：对 λ ∈ {0.2, 0.5, 1.0, 2.0, 4.0}（保持 increase-only）扫描，先确定「输出 shift 可测（JS 明显 >0）」的最小 λ；只有在该 regime 下比较 selector 才有意义。报告每个 λ 的 output-shift 与 harm。
4. 比较 selectors：random / surface / attention_only / attention_x_resp_pos / oracle（oracle 隔离）。
5. 小规模 **deterministic generation eval 子集**（greedy，max_new_tokens 小，prompt="Question:...\nAnswer:" 直出答案，**不生成 CoT**）：no_steering vs attention_x_resp_pos vs oracle 的 answer 正确率、harm（correct→wrong）、help（wrong→correct），带计数与二项显著性；无统计支持不得声称 accuracy improvement。
6. 输出 3A-1 review gate，明确是否允许进入后续 2000 rerun（默认 False）。

## 严格边界

禁止：decrease/suppress；hard mask；训练 / LoRA / finetuning；用 gold answer 构造 non-oracle selector；把 oracle / gold 结果混入 non-oracle method；联网下载模型；声称 hallucination reduction 或 answer accuracy improvement（除非 generation eval 有统计报告支持）；进入 full 3A / 2000。

## 输入

```text
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl   # selector 信号
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl           # span_char_start/end
data/raw/gsm8k_train_normalized.jsonl                                                            # gold answer（eval-only）
src/recover_attention/attention_bias_steering.py                                                # 复用 hook
```

## 输出目录

```text
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/
```

必须输出：

```text
steering_subset_manifest_500.jsonl
target_selector_report_500.json
oracle_sanity_diagnostic_report.json          # answer-directed（eval-only），并解释 3A-0 为何 inconclusive
attention_mass_fidelity_report_500.json
answer_position_output_shift_report_500.json  # 含 λ sweep：找到可测 regime
generation_eval_subset_report.json            # 小规模 greedy 正确率 + 二项显著性
harm_rate_report_500.json
baseline_comparison_report_500.json
failure_case_report_500.jsonl
success_case_report_500.jsonl
review_gate_controlled_attention_guidance_500.md
```

## 关键指标定义（补充）

```text
# answer-directed oracle metric（eval-only）
gold_first_token_logprob_delta = logprob(gold_answer_first_token | steered) - logprob(... | no_steer)
# 在 answer_position 测；gold_answer 来自 GSM8K，仅用于此 eval-only 指标与 generation 正确率。
# oracle_effective 当且仅当：在存在可测 output-shift 的 λ regime 下，
#   oracle 的 gold_first_token_logprob_delta 显著 > random 与 no-steering，
#   且 harm proxy 未爆表。
```

## 3A-1 Review Gate（诚实分层）

```text
机制层（必须过）：
1. hook 可靠（registered / triggered 请求层 / removed，每次 steered forward）；
2. 存在 λ regime 使 answer-position output shift 可测（JS 明显 >0）；

诊断层（回答，不一定过）：
3. answer-directed oracle：可测 regime 下 oracle 的 gold-token logprob delta 是否 > random / no-steer；
4. non-oracle（attention_x_resp_pos）是否 > random / surface（预计弱，因 2K-V 显示 selector AUC≈0.588）；
5. harm proxy 随 λ 的曲线；
6. generation eval（小子集）：no-steer / attention / oracle 正确率 + 二项 CI；

结论层：
7. ready_for_2000_rerun=False（除非 oracle answer-directed 明显有效 且 non-oracle 有统计支持的正效果——本轮预计不满足）；
8. do_not_enter_full_sprint_3A=True。
```

判定逻辑：

```text
若强 oracle steering 也无 answer-directed 正效果（跨所有安全 λ）：
  → attention steering 在当前构造下不改变答案；建议转 representation-level 或换任务，暂停 attention-bias 路线。
若 oracle 有 answer-directed 正效果但 non-oracle 无：
  → 机制成立，瓶颈在 selector 质量（2K-V 已知 attention≈0.588）；下一步提升 selector 而非 scale-up。
两者皆需统计支持；无支持不得声称 accuracy/hallucination 改善。
```

## 最后更新

```text
PROGRESS.md
docs/progress/sprint_2_history.md 或新增 sprint_3_history.md
```
