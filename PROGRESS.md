# 实验进度记录：Reasoning-Aware Attention Guidance

本文件只保留当前状态索引。详细 sprint 历史、具体指标、命令、文件清单和边界记录归档在 `docs/progress/*_history.md` 与对应 artifact manifest 中。

## 1. 当前项目状态

当前主线停在 Sprint 4B-3：CyberMetric 240-question F5 baseline 与 site-transfer diagnostic 已完成。

Sprint 4 当前目的：从早期 span intervention / steering 失败经验转向 cyber-domain hallucination detection baseline，先建立可复核的数据、标签代理、F5 输出层风险基线和后续 feature-family bake-off 边界。

当前结论边界：

- 尚未证明 attention guidance 有效。
- 尚未证明 hallucination reduction。
- 尚未证明 answer accuracy improvement。
- 未训练 probe。
- 未继续 steering。
- 未进入 full Sprint 3C。
- 未批准 2000-scale rerun。

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

当前 Sprint 4 总结：F5 输出层 baseline 已可作为 Sprint 4C 的比较基线；site-transfer 未进入评估，因为 gate skipped；F2 trajectory substrate 在当前 chat 条件下不足，应在后续 sprint 降级、替代或重新设计。

详细记录：`docs/progress/sprint_4_history.md` 与 `docs/progress/sprint_4_artifact_manifest.md`。

## 3. 当前可运行命令

推荐测试命令：

```bash
conda run -n recover_attention python -m pytest -q
```

最近 Sprint 4B-3 相关检查已在 history 中归档。当前文件清理仅为文档整理，不需要重新运行实验或 pytest。

## 4. 当前关键文件状态

- `PROGRESS.md`：当前状态索引。
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
- Sprint 4B-3 的 F5 bars 是非训练 baseline，不是 intervention 效果。
- 当前 chat prompt 条件下缺少 reasoning text substrate，F2 trajectory-transition features 需要在 Sprint 4C 重新处理。

## 6. 下一步建议

建议下一步开 Sprint 4C：围绕 Sprint 4B-3 的 F5 baseline，设计 feature-family bake-off 或 reasoning-substrate replacement。

边界：不要自动开始 probe training、attention steering、site-transfer patching、2000-scale rerun 或 hallucination-reduction claim；必须先有明确 task card 或用户指令。
