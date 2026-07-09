# Sprint 3 History — Attention Guidance / Steering

本文件记录 Sprint 3（attention steering / guidance）阶段的实验历史。Sprint 0–2 见 `sprint_0_history.md` / `sprint_1_history.md` / `sprint_2_history.md`。

---

## Sprint 3A-0：Attention-Bias Steering Smoke Test（前置，已存在）

- 实现 `src/recover_attention/attention_bias_steering.py`：increase-only 的 additive attention logit bias hook（patch Qwen2 attention forward，softmax 前对 (query,key) 加正偏置，可移除；负偏置报错）。selectors：random / surface / attention_only / attention_x_resp_pos / oracle（on-path number，eval-only 隔离）。
- prompt 用 answer-eliciting 模板 "Question: {q}\nAnswer:"，query scope=answer_position（末 token）。
- 结论：hook 可靠；但 primary config（λ=0.2, layers 16+24）下 oracle sanity **inconclusive**。

## Sprint 3A-1：Controlled Attention Guidance on 500 Cases（本轮）

目标：复用 3A-0 hook，修正 oracle 诊断并给出是否可进入后续 rerun 的判定。**不是** full 3A、**不是** 2000 rerun、**不声称** hallucination reduction / answer accuracy improvement。默认 ready_for_2000_rerun=False、do_not_enter_full_sprint_3A=True。

任务卡审查与补充（执行前，基于 3A-0 实测数据）：3A-0 oracle inconclusive 有两个根因——(1) **oracle 指标用错**：判定基于 `target_attention_mass_delta`（提到「该 selector 自己选的」span 的 mass），对任何 selector（含 random）都近似恒正、且不同 selector 目标不同不可比（实测 oracle 0.0021 甚至 < random 0.0036）；(2) **干预太弱**：λ=0.2/2 层下 answer-position 输出几乎不动（JS~1e-5）。因此补两点并写回 card：
- **[补充 1] answer-directed oracle 指标（eval-only）**：steering 到 on-path span 是否把 answer-position 分布推向 **gold answer 首 token** 比 random / no-steer 更多（gold 只用于 eval-only 指标，绝不进 non-oracle selector）。
- **[补充 2] λ sweep**：{0.2, 1.0, 4.0}，先找到 output-shift 可测（JS 明显 >0）的 regime，再在该 regime 比 selector。

已完成内容：

- 实现 `scripts/sprint_3A_1_controlled_attention_guidance.py`：复用 3A-0 hook / selectors / 前向；对 120 个 eligible question（2K-W 有 232 题满足 ≥4 spans 且含 on+off-path number）做 λ×selector 前向；新增 gold-first-token logprob delta（eval-only）与 question-grouped bootstrap；输出 oracle 诊断 / mass fidelity / output-shift / harm / baseline / cases / review gate。
- 新增测试：logprob_of / harm proxy。输入：2K-W answer_position_score_matrix（selector 信号）+ 2J-Fix feature matrix（span offsets）+ gsm8k gold answer（eval-only）。

关键数据（120 题；λ∈{0.2,1.0,4.0}；layers 16+24；query=answer_position）：

- **λ sweep 找到可测 regime**：非 oracle answer-position JS 随 λ 上升——0.2→3e-5、1.0→6e-4、4.0→**0.020**；mass delta 4.0 下约 0.20–0.32（vs 3A-0 的 ~0.002）。即 λ=4.0 是「输出真的动」的 regime。
- **answer-directed oracle 诊断（gold-first-token logprob delta，regime λ=4.0，均值）**：random **+0.665** / surface **+0.829** / oracle **+0.593** / attention_only **+0.442** / attention_x_resp_pos **+0.569**。**所有 selector 都把 gold token 概率抬高，但 oracle 并不优于 random**（oracle−random 配对：mean −0.053，CI95 [−0.223,+0.135]，不显著；且 λ=0.2/1.0/4.0 三档均不 stable-positive）。
- harm rate 随 λ 上升：λ=4.0 下 19–27%（top1_changed / entropy 爆 / margin 崩）。
- on-path hit rate（eval-only）：oracle 1.00 / surface 0.93 / attention 0.625 / random 0.59。

核心结论（决定性负面结果）：

- **机制层通过**：hook 可靠（hook_ok 全 True），λ≥~4 时干预对 answer-position 输出有可测影响。3A-0 inconclusive 的两个根因已定位并修正。
- **answer-directed oracle NOT effective（选择性意义上）**：把注意力加大到 **正确** span 并不比加大到 **随机** span 更能把答案推向 gold（oracle≈random，甚至 surface/random 更高）。高 λ 下 gold-token logprob 普遍上升是「把注意力笼统加到问题 token → 泛化地锐化答案」的 **非选择性 artifact**，并伴随 20–27% harm，而非 keyness-driven 的定向引导。
- **瓶颈不在 selector 质量**：即便 oracle（完美 selector）也无选择性收益，说明问题在 **机制本身**——additive attention bias 不能把「span 选对」转化为「答案更对」。因此不建议继续沿 attention-bias steering 路线，也不建议只去提升 selector。
- 决策：ready_for_2000_rerun=False；do_not_enter_full_sprint_3A=True；不声称 accuracy/hallucination 改善。

下一步建议：转向 **representation-level / value-level 干预**（而非 attention-logit bias），或重新审视任务/数据集是否适配 attention guidance；若仍要走 attention 路线，需要 steered autoregressive generation + 正确率统计来复核（本轮为 single-forward answer-directed proxy，generation eval 标注为 deferred）。

输出文件（`outputs/logs/sprint_3A_1_controlled_attention_guidance_500/`）：

```text
steering_subset_manifest_500.jsonl / steering_forward_manifest_500.jsonl
target_selector_report_500.json / oracle_sanity_diagnostic_report.json
attention_mass_fidelity_report_500.json / answer_position_output_shift_report_500.json
harm_rate_report_500.json / baseline_comparison_report_500.json
generation_eval_subset_report.json（deferred，记录原因）
failure_case_report_500.jsonl / success_case_report_500.jsonl
review_gate_controlled_attention_guidance_500.md
```

新增或修改文件：

- docs/codex_tasks/sprint_3A_1_controlled_attention_guidance_500.md（新增任务卡 + 2 点补充）
- scripts/sprint_3A_1_controlled_attention_guidance.py（新增）
- tests/test_sprint_2h_instance_signal.py（新增 3A-1 测试）
- outputs/logs/sprint_3A_1_controlled_attention_guidance_500/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_sprint_2h_instance_signal.py -q
conda run -n recover_attention python scripts/sprint_3A_1_controlled_attention_guidance.py --primary-n 120 --lambdas 0.2 1.0 4.0
conda run -n recover_attention python -m pytest -q
```

检查结果：

- full pytest：605 passed, 2 skipped。
- 120 题 × 3 λ × 5 selector 前向完成；hook_ok 全 True；λ=4.0 为可测 regime。
- oracle answer-directed 不显著（oracle−random CI 含 0，跨 3 档 λ 均不 stable）。

边界：

- increase-only；无 decrease/hard-mask/训练/LoRA；gold answer 仅用于 eval-only 指标与（deferred 的）generation 正确率，绝不进 non-oracle selector；oracle 全程隔离。
- 未进入 full 3A / 2000；未执行 attention steering 的下游 claim；未覆盖 3A-0 输出（新目录）。
- generation eval（steered autoregressive）本轮 deferred；用 gold-first-token logprob delta 作 answer-directed proxy。

遗留问题：

- attention-bias steering 在当前构造下无选择性 answer 收益；需换机制（representation/value-level）或换任务。
- 高 λ 的 gold-token logprob 普升是非选择性锐化 + 20–27% harm，不能当作 guidance 有效证据。
- generation eval 尚未做；若走后续路线需补 steered generation + 正确率统计。

## Sprint 3B-0：Representation-Level Oracle Intervention Diagnostic（residual injection）

目标：3A-1 否定了「attention-logit bias 能把选对 span 转成答对」，但未否定「关键 span 的 representation 有用」。本轮换 **机制通道**：把 selected span 的 residual deviation 注入 answer-position 残差流，检验 **oracle（on-path）span 是否比 random 更能把答案推向 gold token**。非 full 3B、非 2000、非训练；inject-only（beta≥0）；gold/oracle 仅 eval-only；不覆盖 3A-0/3A-1 输出。

补充（与 3A-1 λ-regime 同一教训）：card 的字面公式注入 **单位范数**向量乘 beta，但深层残差范数 ~100s，单位范数注入是 no-op。改为按 **answer-position 残差范数缩放**：`inj = beta·‖ans_residual‖·unit(mean(h_L[span]) − mean(h_L[all]))`，使 beta 为残差幅度的比例、sweep 能达到可测 regime（已在 config 报告注明）。

已完成内容：

- 实现 `src/recover_attention/representation_intervention.py`：base forward 捕获目标层 hidden；compute_injection_vectors（beta<0 报错、空 span/零 deviation→None、按残差范数缩放）；register/remove residual forward-hook（在层输出的 answer-position 残差上加注入向量，clone 防污染）；steered_forward_with_injection。
- 实现 `scripts/sprint_3B_0_representation_level_oracle_intervention.py`：复用 3A-1 的 selector/subset/gold-token-logprob/paired bootstrap，对 120 题 × 5 beta × 5 selector 做 base+steered 前向，输出全部 11 个 required reports + review gate。
- 新增 `tests/test_representation_level_intervention.py`（5 项：beta<0 拒绝、注入范数=beta·‖ans‖、空 span/零 beta→None、hook 注入 answer-position 且移除后基线复原且不污染其它位置、注入范数>0）。

关键数据（120 题；layers 16+24；beta∈{0.05,0.1,0.2,0.4,0.8}；target=answer_position）：

- **通道比 attention 强得多**：非 oracle answer-position JS 随 beta 上升 0.0013→0.005→0.017→0.052→**0.098**（3A-1 attention 同类只有 1e-5→0.02）；injection_norm ~13（beta 0.05）→~210（beta 0.8）。即不是「太弱」的失败。
- **oracle 并不比 random 有选择性**：oracle−random 配对 bootstrap 在 **每个 beta 都不 stable-positive**（CI 均含 0）：0.05 +0.001[−0.026,+0.026]、0.1 +0.007、0.2 +0.029[−0.068,+0.126]、0.4 +0.019、0.8 −0.066。regime beta=0.05 verdict=oracle_not_selectively_better_than_random。
- regime（0.05）各 selector 的 gold-logprob delta 近乎相同：random +0.327 / oracle +0.328 / attention +0.343 / surface +0.324 / attn×resp +0.309（CI 高度重叠）。且该 delta 随 beta 对**所有 selector 同步上升**（oracle β=0.2 +1.26 vs random +1.23；β=0.4 +2.16 vs +2.15）——非选择性。
- harm 随 beta 陡升：β=0.4 时 ~47–48%，β=0.8 时 61–68%。高 beta 的「效果」与失稳高度混淆。

核心结论（跨通道的决定性负面）：

- **residual channel 强且可靠**（hook_ok 全 True，JS 到 0.1 量级），排除「干预太弱」。但注入 **正确** span 的表示并不比注入 **随机** span 更能把答案推向 gold——所有 selector 同幅度移动，gold-logprob 普升是「扰动 answer-position 残差 → 泛化锐化 + 失稳」的非选择性效应。
- **与 3A-1 完全同型**：两个相互独立的通道（attention-logit bias 与 residual injection）都显示同样的非选择性。因此失败不是 attention 通道特有——而是 **span-level / answer-position / single-forward 干预在此 GSM8K 任务上不产生选择性 answer 改善，与通道无关**。span-importance 信号（2K-V 已证 attention AUC~0.588）对 **detection** 真实，但通过这两种机制在 answer position 都无法转成 **causal answer-steering** 杠杆。
- 对应 card 的 Situation C。决策：ready_for_2000_rerun=False、do_not_enter_full_sprint_3B=True；不声称 accuracy/hallucination 改善。

下一步建议：不要扩大；转向更细粒度——(1) reasoning-step（数字被 **消费** 的那一步）而非 final answer position 的干预；(2) 从 **正确 run** 做 answer-position residual patching（activation patching）；(3) value/MLP-level causal tracing；(4) 重审 GSM8K span-level intervention 任务是否适配 guidance；或暂停 steering 线，回到 detection/diagnosis。机制上的解读：on-path 数字在模型内部计算中已被分布式使用，事后在 answer position 再注入并不新增选择性正确信息，只是扰动。

输出文件（`outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/`，共 11）：

```text
representation_subset_manifest.jsonl / representation_intervention_config.json / representation_forward_manifest.jsonl
representation_intervention_fidelity_report.json / gold_logprob_delta_report.json
oracle_vs_random_diagnostic_report.json / selector_comparison_report.json / harm_rate_report.json
failure_case_report.jsonl / success_case_report.jsonl
review_gate_representation_level_oracle_diagnostic.md
```

新增或修改文件：

- src/recover_attention/representation_intervention.py（新增）
- scripts/sprint_3B_0_representation_level_oracle_intervention.py（新增）
- tests/test_representation_level_intervention.py（新增）
- outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_representation_level_intervention.py -q
conda run -n recover_attention python scripts/sprint_3B_0_representation_level_oracle_intervention.py --primary-n 120 --betas 0.05 0.1 0.2 0.4 0.8 --layers 16 24 --overwrite
conda run -n recover_attention python -m pytest -q
```

检查结果：

- full pytest：610 passed, 2 skipped。
- 120 题 × 5 beta × 5 selector 前向完成；hook_ok 全 True；residual channel 可测（JS 到 ~0.1）。
- oracle−random 跨 5 档 beta 均不 stable-positive；verdict=oracle_not_selectively_better_than_random。

边界：

- inject-only（beta≥0）；无 decrease/hard-mask/训练/LoRA；gold answer 仅用于 eval-only gold-token logprob；oracle 仅作 eval-only 诊断 selector，全程隔离。
- 未进入 full 3B / 2000；未覆盖 3A-0/3A-1 输出（新目录）；generation eval 用 single-forward answer-directed proxy（deferred autoregressive）。

遗留问题：

- 两个通道均证明 answer-position span 注入无选择性；若继续 steering 需转 reasoning-step 干预 / correct-run activation patching / value-MLP tracing，或换任务。
- beta 高时 harm 陡升（≥40%），高 beta 的 gold-logprob 上升不可作为 guidance 有效证据。
- 仍未做 steered autoregressive generation + 正确率统计；负面机制诊断已足以支撑「不 scale、换粒度/机制」的决策。

## Sprint 3C-0：Correct-vs-Wrong Activation Patching at Reasoning-Step Positions（本轮）

目标：承接 3A/3B 连续负结果，做 causal localization。只回答一个问题——在 teacher-forced fixed trace 中，把 correct run 的 activation patch 到 wrong run 的某个 reasoning-step position，能否让 wrong run 的后续 answer 分布更接近 gold。**不是** full 3C、**不是** 2000 rerun、**不训练**、**不 LoRA**、**不声称** hallucination reduction / answer accuracy improvement。默认 `ready_for_2000_rerun=False`、`do_not_enter_full_sprint_3C=True`。

Preflight（Fix 1/2/3 全部完成）：

- Fix 1：PROGRESS.md 顶部 current status 已从 3A-0 修正到 3C-0（preflight 检测 stale_top_status=false）。
- Fix 2：`outputs/` 被 `.gitignore` 忽略，3A-1 / 3B-0 聚合报告本地存在但未 tracked；新增 `docs/progress/sprint_3_artifact_manifest.md` 逐条记录（文件名 / exists / tracked / ignored / 关键结论）。
- Fix 3：写清 3B-0 结论边界——只否定 same-run span residual deviation → final answer-position injection → single-forward gold-token proxy；未否定 correct-run activation patching / reasoning-step patching / value-MLP tracing / autoregressive steering。

新增模块 / 脚本 / 测试：

- `src/recover_attention/activation_patching.py`：`forward_with_hidden` / `register_residual_replace_hooks` / `remove_hooks` / `patched_forward` / `extract_reasoning_step_positions` / `match_role_positions` / `layer_patch_vector` / `first_token_id` / `compute_harm` / `bootstrap_ci` / `paired_deltas_vs_control` 等。
- `scripts/sprint_3C_0_correct_wrong_activation_patching.py`：采样 → 同题 correct/wrong 配对 → role-to-role 位置对齐 → 五种 patch condition 前向 → fidelity / effect / control / heatmap / harm / case / review gate。
- `tests/test_activation_patching.py`：hook 注册/触发/移除、只改 target position、role 提取与匹配、alpha 校验、聚合与 paired control delta，以及 **first_token_id 返回 leading digit（非 space token）** 的回归测试。

执行发现（先修 metric bug 再出正式结果）：

- 首跑所有行 `clean_direction_score ≡ 0`。根因：Qwen2.5 把 " 42" 切成 `[space, '4', '2']`，`first_token_id(" "+answer)` 对任何数字答案都返回恒定的 leading-space token（id 220），导致 35 对 pair 的 `gold_first_token == wrong_first_token`。
- 修 `first_token_id`：改为对 strip 后的答案编码并返回 leading digit token（31/35 对可区分），补回归测试后重跑。下述数据为修正后结果。

规模与设置：

- 100 题 × 6 samples（temp 0.7，top_p 0.95，max_new_tokens 160）→ 600 traces → 35 个同题 correct/wrong pair。
- residual_replace，alpha=1.0，layers [16, 20, 24]，position types {generated_operator_token, generated_intermediate_number_token, generated_final_answer_number, final_answer_position(control)}，2085 forward rows。metric = teacher-forced gold-minus-wrong first-token logprob delta（gold 仅 eval-only）。

核心结果：

- hook fidelity 全清：registered/triggered/removed = 1.0，非目标位置无污染，mean patch_delta_norm ≈ 104.2。
- overall `mean_clean_direction_score = -0.0185`，CI95 [-0.113, +0.080] → **非 stable positive**。gold 与 wrong first-token logprob 同向上升（+0.735 vs +0.754），即任务预警的 non-selective perturbation 特征。
- control（同题 paired bootstrap，clean_direction_score）：
  - correct_to_wrong − no_patch：-0.018，CI95 [-0.119, +0.087]（无区分）。
  - correct_to_wrong − random_donor_patch：+0.092，CI95 [-0.012, +0.207]（**不稳定**）。
  - correct_to_wrong − same_trace_random_position_patch：-0.227，CI95 [-0.360, -0.110]（**稳定更差**）。
  - correct_to_wrong − correct_activation_patch：-0.029，CI95 [-0.159, +0.103]。
- 最高 clean-direction cell 是 L20 `final_answer_position`（+0.110），但那是 **control** 位置、harm≈0.94、gold/wrong 同幅上升（+2.72/+2.61）；reasoning-step primary 位置效果≈0 且 harm≈0。没有任何 reasoning-step × layer 稳定优于 control。
- 判定：**情况 C** —— 当前 teacher-forced activation patching 未命中 selective causal state。

检查：

- 定向 pytest：7 passed；full pytest：617 passed, 2 skipped。
- review gate 19 问全部作答；verdict 四项均 false。

边界与遗留：

- 全程 teacher-forced single-forward proxy，未做 steered autoregressive generation / 正确率统计；oracle/gold 仅 eval-only，不作可部署方法。
- answer 解析对非 "#### N" 格式 trace 会退化到 last-number fallback（个别 pair 把列表序号当答案），是噪声来源之一，但不改变负结论。
- 下一步（Case C）：转 value/MLP causal tracing（定位写答案的具体计算路径，而非整段 residual replace），或暂停 steering 回到 detection/diagnosis。不 scale、不回调 attention-bias / residual span injection。

## Sprint 3C-0-Fix：Answer-Position Proxy Repair and Sequence-Logprob Recheck（本轮）

目标：低成本复核 3C-0 的负结果是否是 answer-proxy artifact。修两处 proxy 弱点——(1) 不再读整段 trace 末位 logits，改在 final answer 数字前一位（answer slot）读；(2) 不再只看首个 digit token，改算完整数字答案序列的 conditional logprob（gold vs wrong，同一 wrong prefix 下）。**不是** full 3C、**不是** 2000、**不训练**、**不新增 steering**，全程 teacher-forced single-forward proxy。默认四标记 false。

复用与稳健化：

- 复用 3C-0 `correct_wrong_pair_manifest.jsonl`，**不重新采样**。
- 稳健 answer-span 抽取（`#### N` → answer phrase → 排除列表序号的 last-number fallback → parse failure）重验 pair：35 → 34 保留（1 个 recipient 在稳健抽取下不再是合法非-gold 答案而被丢）。抽取成功率 1.0、parse failure 0.0；但 fallback_last_number 触及 0.80 的 pair（大量 trace 不写 `#### N`），是噪声 caveat。

新增模块/脚本/测试：

- `src/recover_attention/answer_proxy_metrics.py`：`extract_final_answer_span` / `answer_token_ids`（丢 leading-space token）/ `token_index_for_char_start` / `sequence_logprob_at_answer_slot` / `compute_corrected_clean_direction` / `paired_bootstrap_delta`（支持跨 position-type 的 (pair,layer) 配对）。
- `scripts/sprint_3C_0_fix_answer_proxy_recheck.py`；`tests/test_answer_proxy_metrics.py`（9 测试）。

设置：layers [16,20,24]，alpha=1.0，primary position = operator / intermediate number（严格早于 answer slot），control = final_answer_position 重定义为答案数字前一位。1104 forward rows。patch 位置 ≥ answer slot 的一律跳过。

核心结果——corrected proxy 把 3C-0 的 Case C **修订为 Case B**：

- reasoning-step correct→wrong：overall corrected clean_direction = **+0.120**，CI95 [+0.053, +0.197]（相对 no_patch 稳定为正；gold +0.121，wrong +0.005）。但 **非选择性**：vs random donor +0.103 CI95 [-0.017, +0.229]（不稳定）；vs same-trace random position -0.020 CI95 [-0.183, +0.124]（不稳定）。即：在推理步 patch 比「什么都不做」略好，但不比随机 donor / 随机位置更好——是温和的非特异扰动，不是角色特异的因果修复。
- final-answer 读出位（answer-slot 前一位）：**大且部分选择性**——gold **升**、wrong **降**（L16 clean +1.90 / L20 +3.64 / L24 +12.67；gold +1.9/+3.1/+9.1，wrong -0.5/-0.5/-4.3）。vs no_patch +6.12 CI95 [+4.62, +7.79]（稳定）；vs random donor **+3.13 CI95 [+1.87, +4.46]（稳定为正）**；vs same-trace random position -1.24 CI95 [-3.02, +0.61]（不稳定）。harm 随层升高：0.41 → 0.53 → 0.88。
- 读法：correct-run activation **确实**携带可把 wrong run 推向 gold 的信息（3C-0 的平坦 null 有一部分是 proxy artifact）；但选择性信号集中在**答案读出/压缩阶段**，不在显式 reasoning step——与 3A/3B 的 answer-position 关注一致，只是在 corrected full-sequence proxy 下才显出「优于 random donor」的选择性。它更像「位置 + 同题 correct context」而非精确定位的 donor-特异开关（未跑赢 same-trace random position），且高层 harm 很大。

检查：定向 pytest 9 passed；full pytest 626 passed, 2 skipped。review gate 17 问全部作答，verdict 四项 false，判定 Case B。

边界与遗留：

- 全程 teacher-forced single-forward proxy；未做 autoregressive generation / 正确率。
- 下一步（Case B）：围绕 **final-answer compression** 做 activation patching / causal tracing（answer 读出位的 value/MLP 写路径），**不**回到 span-level residual injection，也不再在 reasoning step 做整段 residual replace。需先建立 donor-特异性（跑赢 same-trace random position）与可控 harm 的 regime，再谈任何 generation-level eval。

## Sprint 3C-1：Final-Answer Compression Value/MLP Causal Tracing（本轮）

目标：3C-0-Fix 发现 final-answer 读出位有 correct-run 修复信号，但没定位是哪个模块写的。本轮把读出位（答案数字前一位）的整段 residual patch 拆成三条 module 写路径——self-attn output / MLP output / whole-residual output——各自做 interpolation activation patching，用 **donor-specificity** 与 **site-specificity** + harm 控制来判定真正的 causal write。**不是** full 3C / 2000 / 训练 / 可部署 steering，全程 teacher-forced single-forward proxy。默认四标记 false。

复用与设置：复用 3C-0-Fix 的 34 对 corrected pair（不重采样）。forward hook 抓每层每模块 output；patch = `(1-α)·recipient + α·donor` 打在读出位。网格：modules {attention_output, mlp_output, residual_output} × layers {16,20,24} × α {0.25,0.5,0.75,1.0} × 6 conditions（no_patch / correct_donor / random_donor / same_trace_random_position / same_pair_wrong_position / wrong_donor_self）。7308 forward rows。指标沿用 3C-0-Fix corrected answer-sequence clean_direction。

新增：`src/recover_attention/module_causal_tracing.py`（capture_module_outputs / register_module_patch_hook / sequence_logprob_with_module_patch / build_answer_readout / module_vector / compute_donor_specificity / compute_site_specificity）、`scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py`、`tests/test_module_causal_tracing.py`（7 测试）。

Fidelity：hook registered/triggered/removed = 1.0。`wrong_donor_self` 非严格 no-op——donor 抓自 full-trace forward、注入到较短的 prefix+answer forward，attention kernel 在不同序列长度下非逐位一致——但 floor 很小：self mean|clean| 0.087 vs correct-donor 1.632（比 0.05），且在 specificity 差分中抵消（所有 donor 共享）。`residual_output|L24|α1.0` 复现 3C-0-Fix 整段 residual 结果（+12.67，harm 0.88），作一致性校验。

核心结果——**Case A：答案读出位的正确答案方向主要由 MLP 写入，且该写入 donor-特异、site-特异、harm 可控**：

- donor-specificity（correct − random donor，paired）三模块都稳定为正：attention +0.093 CI95 [+0.014,+0.181]，mlp +0.141 [+0.044,+0.252]，residual +1.703 [+1.24,+2.19]。
- site-specificity（correct 读出位 − correct 随机位，paired）：**mlp_output +0.353 CI95 [+0.242,+0.476] 稳定为正**——唯一 per-module 稳定的 site 信号。attention_output -0.006 [-0.076,+0.062]（无 site 特异）；residual_output -0.547 [-1.13,+0.064]（无 site 特异）。
- 即：attention output 泛泛地推答案（donor-特异但非 site-特异）；whole residual 幅度最大但非选择性且 harm 高；**只有 MLP output 同时通过 donor + site 特异性**。
- MLP harm-controlled regime（α sweep，L24）：clean +0.319/+0.590/+0.943/+1.290（α 0.25→1.0），harm 0.06/0.18/0.24/0.24；gold +0.327/+0.584/+0.912/+1.228，wrong 近平。最优低 harm 选择性 cell：`mlp_output|L24|α0.25`（clean +0.319，gold +0.327，wrong +0.040，harm 0.06）与 `mlp_output|L20|α0.25`（clean +0.131，harm 0.06）。

读法：3A/3B/3C-0 的负结果得到解释——span-level 与 whole-residual 干预太粗；选择性、低 harm 的因果 handle 是**答案读出位的 MLP 写入**。这是在 teacher-forced proxy 下定位到一个机制，**不是** accuracy / generation 结果。

检查：定向 pytest 7 passed；full pytest 633 passed, 2 skipped。review gate 17 问全答，verdict 四项 false，判定 Case A（找到选择性低 harm causal site）。

边界与遗留：

- 全程 teacher-forced single-forward proxy；未做 autoregressive generation / 正确率；MLP-output patch 用 correct donor（eval-only），不作可部署方法。
- 下一步（Case A → Sprint 3C-2）：分析 `mlp_output|L24|α0.25`（及 L20）的 MLP 读出写方向——是否存在低秩/可解释的「答案」方向；低 α 的 MLP-output nudge 是否在 autoregressive generation 下存活；先建立 harm-controlled steering probe，再谈任何 generation-level 正确率 eval。仍不进 2000 / full 3C。
