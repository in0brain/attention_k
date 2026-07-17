# 实验进度记录：Reasoning-Aware Attention Guidance

本文件只保留当前状态索引。详细 sprint 历史、具体指标、命令、文件清单和边界记录归档在 `docs/progress/*_history.md` 与对应 artifact manifest 中。

## 1. 当前项目状态

当前主线停在 Sprint 4D-2 W0.5-B：条件增量 pipeline 的实现 + ≤20 prompt 真实 smoke 已完成，设计层已冻结关闭。

Sprint 4 当前目的：从早期 span intervention / steering 失败经验转向 cyber-domain hallucination detection baseline，先建立可复核的数据、标签代理、F5 输出层风险基线和后续 feature-family bake-off 边界。4D-2 起主问题改写为**条件增量**：相对最强输出侧基线 O = F5 + 可见文本，hidden-state probe H 是否还有增量。

### 预注册与启动 gate（4D-2 起，代码强制）

设计权威 = `docs/paper/preregistration.md`，冻结指纹见 `docs/paper/preregistration.lock`。

| gate | 含义 | 当前 |
|---|---|---|
| G1 | 目标 workshop CFP 已确认（deadline / page limit / archival） | **未过（provisional）** |
| G2 | preregistration.md 的 sha256 与 lock 一致 | 已过 |
| G3 | ≤20 prompt 真实 model-in-loop smoke 通过 | 已过 |

`stage0_full_generation_allowed = G1 AND G2 AND G3`，当前 **False**。Stage 0（2880 全量前向）仍 No-Go，唯一缺口是 G1。

**G1 目标（2026-07-16 记录）**：EIML3 @ NeurIPS 2026（The 3rd Workshop on Epistemic Intelligence in Machine Learning）。投稿开放 2026-07-29，截止 **2026-08-29 AoE**，通知 09-29，workshop 12-12 巴黎。**archival = false**（NeurIPS workshop 统一非归档，后续可扩展投主会）。选它的理由：EIML 明确关注 hallucination / 过度自信、系统能否识别"知道 vs 不知道"、epistemic signal 能否指导 abstention、以及**经严格压力测试后的负结果**——本工作的条件增量问题（内部表征是否含输出文本尚未暴露的幻觉信息、该增量是否依赖输出可观测性）比"又一个 detector"更贴题，且负结果在此可发表。

**G1 仍为 provisional 的唯一原因：page limit 未公布**（官方 CFP 已列日期/主题/评审安排，但未公布 submission type 与 page limit，须等 07-29 投稿系统开放）。§2 要求 deadline / page limit / archival 三项齐全，故 G1 不得转绿。

事实记录在 `docs/paper/cfp_record.json`。**G1 校验的是该文件的内容，不是它是否存在**：三项事实齐全（page_limit 为正整数、archival 为 bool、deadline 非空）且 `status="confirmed"` 才通过。这防的是"建个占位文件就把 gate 刷绿"，以及"只把 status 改成 confirmed 但事实仍缺"。G1 需要外部事实，不得由脚本或代理自行置位。page limit 一确认，填入 `page_limit` 并把 `status` 改为 `confirmed` 即可。

预注册版本：v2.1（sha `b3ef9228…`，冻结 2026-07-15）→ **v2.2**（sha `ffa722e8…`，冻结 2026-07-16）。v2.2 是唯一一次修订，只改 §7 融合协议（dense 拆成 F5 / H 两个独立 block，三块等权），理由与正当性记录在 preregistration.md 顶部。v2.1 的实现与 hash 保留在 commit `57b7ae9`，其 G3 已随 hash 变更自动失效。

当前结论边界：

- 尚未证明 attention guidance 有效。
- 尚未证明 hallucination reduction。
- 尚未证明 answer accuracy improvement。
- 已训练仅限检测用途的线性 logistic probe。
- 未继续 steering。
- 未进入 full Sprint 3C。
- 未批准 2000-scale rerun。
- **4D-2 尚无任何条件增量结论**：smoke 规模（n=115、正例 10）的 AUROC 只用于接线与协议缺陷诊断，不是研究发现，不进入论文结果。

## 2. 已完成 Sprint 摘要

### Sprint 0-1

- Sprint 0：完成项目骨架、基础 schema、jsonl 数据流和 smoke-test 起点。
- Sprint 1：完成 candidate span、ablation、NLI / recoverability stub 与早期 downstream label plumbing，目标是建立 attention-importance discovery 的最小数据链路。

详细记录：`docs/progress/sprint_0_history.md`、`docs/progress/sprint_1_history.md`。

### Sprint 2

- Sprint 2A-2F：完成 hidden-state cache、representation features、probe dataset、minimal probe baseline、planned-only guidance candidate 和 mini closed-loop report，目标是形成最小 dry-run 闭环。
- Sprint 2G-2K：完成 500/2000 弱标签诊断、多 span matrix、attention / hidden / output-effect 分解和 answer-position output-effect 复核，目标是判断哪些信号适合 keyness / fragility / ranking。

当前 Sprint 2 总结：attention 与 answer-position output-effect 在诊断上有价值，但不足以直接支持 2000-scale rerun 或 attention steering。

详细记录：`docs/progress/sprint_2_history.md`。

### Sprint 3

- Sprint 3A：完成 attention-bias steering smoke 与 controlled attention guidance，目标是验证加性 attention bias 是否能把正确 span 转化为选择性答案改进。
- Sprint 3B：完成 representation-level residual injection diagnostic，目标是检查 span representation 注入是否比 attention bias 更有效。
- Sprint 3C：完成 correct-vs-wrong activation patching、answer-position proxy repair、MLP readout causal tracing、donor-free direction analysis、MLP readout attribution probe 和 approximate J-lens sanity check。

当前 Sprint 3 总结：span-level steering 未形成稳定选择性收益；final-answer readout MLP 是有效机制诊断位置，但当前结果更适合 detection / attribution，不支持继续 blind steering。

详细记录：`docs/progress/sprint_3_history.md` 与 `docs/progress/sprint_3_artifact_manifest.md`。

### Sprint 4

- Sprint 4A：完成 cyber mainline reset，目标是从 unsupervised span steering 转向 cyber-domain hallucination detection 与 gated intervention 研究线。
- Sprint 4B dataset audit：完成 CyberMetric / SecQA / CS-Eval 原始数据审计，目标是选择适合 MCQ label-proxy 的 cyber 数据源。
- Sprint 4B smoke：完成 CyberMetric 小规模 smoke baseline，目标是打通 raw data -> canonical schema -> option-letter proxy -> F5 baseline -> review gate。
- Sprint 4B-1：完成 CyberMetric canonical schema 与 domain label proxy，目标是建立稳定的数据/标签接口。
- Sprint 4B-2：完成 small-model prompt A/B 与 F5 feature plumbing，目标是确定 chat prompt 条件并验证 F5 特征链路。
- Sprint 4B-3：完成 240-question F5 baseline 与 site-transfer diagnostic，目标是得到非训练 F5 baseline bars，并记录 site-transfer gate skipped 状态。
- Sprint 4C：完成 239 条 greedy MCQ 的 F1（L20/L24 direct projection）与 F4（exact finite-label VJP J-lens）捕获、三 seed grouped-CV logistic bake-off 和 grouped-bootstrap increment CI。F5-only CV AUROC 为 0.8343；F5+F1 增量 CI 为 [-0.0429, 0.0202]，F5+F4 为 [0.0000, 0.0000]，均未通过 CI gate。pair mining 由 8 对提升到 17 对，未达 20 对，site-transfer 依规跳过。
- Sprint 4D-0：完成 H1 fabricated-identifier 设计与数据产物：下载/索引完整 CVE/ATT&CK/CWE 本体快照，新增标识符抽取/归一化/存在性判定模块、`h1_sample` schema、480 条 H1 prompt 样本、id 空间密度审计、H1-F5 设计文档和 4D-1 预注册门槛。本轮未调用 causal LM、未生成 completion、未训练 probe、未做 F5/steering。
- Sprint 4D-2 W0.5-A：完成预注册冻结（v2.1）。把 population（completion-level）、O 的单一定义、equivalence margin ε=0.02、observability gate δ=0.15、K=6、hidden 层/位置/pooling、TF-IDF 与融合协议、RQ2 分层规则、启动 gate 全部写死并做 sha256 冻结。
- Sprint 4D-2 W0.5-B：完成条件增量 pipeline 的**实现 + ≤20 prompt 真实 smoke**（不是全量生成）。新增 `conditional_increment.py` 统计与 schema 核心（completion-level 标签与 emission 语义、hidden tuple index、全 identifier hidden pooling、六阶 output ladder、三-block 等权融合、stratified grouped folds、paired/independent grouped bootstrap、RQ2 fold-specific 分层、artifact 红线、G2 校验）与 `scripts/sprint_4D_2_conditional_increment.py`（preflight / smoke_synthetic / verify_hidden / smoke_model）。真实 smoke：20 prompt × 6 traces = 120 条，115 条 eligible，正 10 / 负 105，20 groups；hidden 层核验 block19 → tuple20，forward hook 与 `hidden_states[20]` 数值相等（max_abs_diff 0.00，8-bit backend）。期间发现并修订 v2.2 融合协议（见 §1）。本轮未启动 Sprint 0 全量生成、未跑 MCQ 侧、未跑 Llama、未出任何增量结论。
- Sprint 4D-1：完成 H1 emission/fabrication smoke 修复重跑：上一版 4-bit KV-cache 逐 token 输出因 288/288 乱码作废；修复为 `model.generate` 后又发现 4-bit 长文本生成仍标点退化，最终使用本地 8-bit generation backend 跑完 72 个 train-split 问题、288 条 completion。Route A greedy emission 46/48=0.958，ATT&CK+CWE 主口径 mention-level fabrication 65/775=0.084，completion-level fabrication 33/222=0.149，满足预注册 gate。refusal rate 1/288，has_reasoning_text 288/288，overall degeneration 14/288=0.049，触顶 3/288=0.010。本轮未训练 probe、未捕获 activation、未计算 F5、未继续 steering。

当前 Sprint 4 总结：finite-label cyber MCQ 上 F1/F4 未提供相对 F5 的稳定增量；H1 fabricated-identifier 已通过 emission/base-rate smoke gate；4D-2 的设计已冻结（v2.2）、统计与 schema 核心已实现并通过真实 smoke（G2/G3 绿）。Stage 0 全量生成仍 No-Go：G1 未过，且全量执行路径尚未实现（见 §5）。仍不得直接进入 intervention 或 hallucination-reduction claim。

详细记录：`docs/progress/sprint_4_history.md` 与 `docs/progress/sprint_4_artifact_manifest.md`。

## 3. 当前可运行命令

推荐测试命令：

```bash
conda run -n recover_attention python -m pytest -q
```

Sprint 4D-2 W0.5-B 可运行 stage（`scripts/sprint_4D_2_conditional_increment.py --stage <name>`）：

```text
preflight        启动 gate 报告。G2 不匹配立即停;写 preflight_report.json。
smoke_synthetic  无模型统计接线自检。**不构成 G3**。
verify_hidden    真模型 1 次前向,核验 hidden_states tuple index = block+1。
smoke_model      真模型 ≤20 prompt 端到端;通过后写 smoke_report.json = G3。
stage0_gate      只做 G1+G2+G3 门控检查,未齐则停。
```

`generate / cache / ladder / gate / increment / rq2` **尚未实现**，传入会 `SystemExit`。详细命令、检查和 artifact 清单归档于 Sprint 4 history/manifest。

## 4. 当前关键文件状态

- `PROGRESS.md`：当前状态索引。
- `docs/paper/preregistration.md`：4D-2 设计权威（v2.2，已冻结）。
- `docs/paper/preregistration.lock`：冻结指纹（sha256）、版本历史、G2/G3 判据。
- `docs/codex_tasks/sprint_4D_2_conditional_increment_and_observability.md`：4D-2 现行任务卡（取代旧的 `sprint_4D_2_h1_full_generation_and_f5_baseline.md`）。
- `docs/progress/sprint_0_history.md`：Sprint 0 history。
- `docs/progress/sprint_1_history.md`：Sprint 1 history。
- `docs/progress/sprint_2_history.md`：Sprint 2 history。
- `docs/progress/sprint_3_history.md`：Sprint 3 history。
- `docs/progress/sprint_4_history.md`：Sprint 4 history。
- `docs/progress/sprint_3_artifact_manifest.md`：Sprint 3 artifact manifest。
- `docs/progress/sprint_4_artifact_manifest.md`：Sprint 4 artifact manifest。

## 5. 当前遗留问题

- 真实 attention guidance 尚未验证有效。
- hallucination reduction 与 answer accuracy improvement 尚未验证。
- Sprint 3 的 MLP readout 结果是机制诊断，不是可部署 steering 结果。
- Sprint 4C 的 detector negative 不反证 H1 fabricated-identifier 主线；MCQ 仍是易校准参考任务。
- pair mining 未达到 20 对，3C-1 site-transfer 在 cyber MCQ 上仍未评估。
- Sprint 4D-1 只证明当前 H1 prompt 与本地 Qwen2.5-7B-Instruct 在 8-bit local generation smoke 上有足够 id emission 与非零 fabrication base rate；未证明 F5 可检测，未证明内部特征有效，未证明 intervention 有效。
- 4D-1 暴露出 4-bit backend 对 H1 长文本自由生成不可靠；后续 H1 生成卡必须显式记录/复用已通过 sanity 的 generation backend，不能复用旧 4-bit 长文本结果。
- **G1 未过（provisional）**：目标已定为 EIML3 @ NeurIPS 2026，deadline 与 archival 已知，但 **page limit 未公布**（须等 2026-07-29 投稿系统开放）。§2 要求三项齐全，故 G1 保持红。这是外部事实，不得由脚本/代理自行置位。
- **Stage 0 的执行路径尚未实现**，"设计层关闭"不等于"可以启动全量生成"。缺口：
  - 全量生成（480 prompt × 6 = 2880 traces）——`smoke_model` 硬卡 ≤20 prompt；批量生成、断点续跑、hidden 落盘规模（2880 × 3584 × fp32 ≈ 40GB 量级）均未设计。
  - **MCQ 任务侧完全缺失**——§6 的 gate 是 `D = S_MCQ − S_H1`，需要 CyberMetric 的 O 侧 OOF 分。当前 4D-2 管线只有 H1，没有 MCQ 就无法判定 H1 是否为"高置信设置"。
  - 正式 ladder/gate/increment/rq2 执行路径——统计函数已实现且有单测，但只被 smoke 的接线版 `_run_ladder`（3 折、inner 2、无 bootstrap CI）调用过；正式版需 5 折、n_boot≥1000 的 paired/independent bootstrap、artifact 红线、RQ2 分层 CI。
  - Llama-3.1-8B 核心 O/H/O+H 复核（§9）未跑。
- 4D-2 的 late-fusion（§7 附录 robustness）在 smoke 规模下不可解读（正例 ≈10 时嵌套 stacking 的 meta 会翻号）；实现已由规模对照单测证明健全，但其有效性须在 Stage 0 规模复核。

## 6. 下一步建议

Sprint 4D-2 W0.5-B 已完成、设计层关闭。下一步两项**正交**，可并行：

1. 完成 G1：目标已定（EIML3 @ NeurIPS 2026），只差 page limit。2026-07-29 投稿系统开放后确认页数，填入 `docs/paper/cfp_record.json` 的 `page_limit` 并把 `status` 改为 `confirmed`，G1 即转绿。需用户提供外部事实，代理不得自行置位。
2. 实现 Stage 0 执行路径（见 §5 缺口）。起手须先定范围：MCQ 侧是复用 4B/4C 的 CyberMetric 产物，还是按 4D-2 同协议重跑——这决定 §6 gate 能否成立。

边界：不要自动开始 F1-F4 内部特征、probe training、attention steering、site-transfer patching、2000-scale rerun 或 hallucination-reduction claim；不要在 G1 未过时启动 Stage 0；不要改动已冻结的 preregistration（改设计须 bump version + 重算 hash + 顶部记录变更）；必须先有明确 task card 或用户指令。
