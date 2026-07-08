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
