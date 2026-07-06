# Sprint 2 Final Checkpoint and Visualization Summary

## 0. 任务名称

Sprint 2 Final Checkpoint and Visualization Summary

## 1. 任务定位

本任务是 Sprint 2 结束后的阶段性 checkpoint。

它不是新的实验 sprint。

它不是 Sprint 3。

它不做新的 attention steering。

它只做：

```text id="alhedn"
1. 串行重跑 full pytest，并记录最终测试结果和耗时。
2. 读取 Sprint 2A-real / 2B / 2C / 2D / 2E / 2F 的正式产物。
3. 生成 Sprint 2 阶段性总结 Markdown。
4. 生成 Sprint 2 阶段性可视化图。
5. 明确 dry-run boundary 和 non-claims。
6. 为 Sprint 3A：Attention Steering Interface Design 做准备。
```

---

## 2. 当前项目状态

当前项目为：

```text id="o4zjje"
Reasoning-Aware Attention Guidance
```

Sprint 2 已完成：

```text id="c71l04"
Sprint 2A：Hidden State Cache Baseline
Sprint 2A-real：Real Hidden State Cache Run
Sprint 2B：Representation Feature Extraction
Sprint 2B-fix：Representation Feature Extraction Scope Alignment
Sprint 2C：Probe Dataset Construction
Sprint 2D：Probe Training Baseline
Sprint 2E：Guidance Candidate Manifest Dry Run
Sprint 2F：Mini Closed-loop Report
```

Sprint 2F 已明确：

```text id="k8jg8r"
Sprint 2 formed a dry-run hidden-state-to-guidance-candidate loop.
```

该闭环为：

```text id="d6k1fe"
hidden-state cache
→ representation features
→ probe dataset
→ probe training
→ guidance candidate manifest dry run
→ closed-loop report
```

但这不是：

```text id="g0nqgz"
executed attention-steering loop
```

---

## 3. 本 checkpoint 的目标

本 checkpoint 的目标是生成一个适合阶段汇报、组会、后续 Sprint 3A 设计参考的总结包。

正式输出目录：

```text id="ev4mnl"
outputs/logs/sprint_2_stage_summary/
```

正式输出：

```text id="pxww3u"
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
outputs/logs/sprint_2_stage_summary/figures/sprint_2_pipeline_overview.png
outputs/logs/sprint_2_stage_summary/figures/probe_target_counts.png
outputs/logs/sprint_2_stage_summary/figures/probe_metrics_vs_baseline.png
outputs/logs/sprint_2_stage_summary/figures/guidance_candidate_action_counts.png
outputs/logs/sprint_2_stage_summary/figures/guidance_confidence_counts.png
outputs/logs/sprint_2_stage_summary/figures/sprint_2_boundary_summary.png
```

可选额外输出：

```text id="uzd246"
outputs/logs/sprint_2_stage_summary/figures/*.svg
```

PNG 是必需格式。

SVG 可选。

---

## 4. 非目标

本 checkpoint 不做：

```text id="eu6r3l"
不重新缓存 hidden states
不重新抽取 representation features
不重新构造 probe dataset
不重新训练 probe
不重新生成 guidance candidate manifest
不重新生成 Sprint 2F closed-loop report
不读取 hidden-state tensors
不加载 probe_model.pkl
不调用 HF model
不调用 Ollama
不调用 tokenizer
不调用 NLI model
不注入 attention mask
不修改 transformer attention weights
不重跑 CoT generation
不评估 answer accuracy improvement
不验证 hallucination reduction
不进入 Sprint 3A implementation
```

本 checkpoint 只读已有正式产物。

---

## 5. 必须阅读的文件

开始实现前，先阅读：

```text id="ufwvtp"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_2_history.md

docs/codex_tasks/sprint_2F_mini_closed_loop_report.md

src/recover_attention/closed_loop_report.py
scripts/21_write_sprint_2_closed_loop_report.py
tests/test_closed_loop_report.py
```

如果本 task card 已存在，也阅读：

```text id="xrssfv"
docs/codex_tasks/sprint_2_final_checkpoint_visualization_summary.md
```

---

## 6. 输入文件

本 checkpoint 只读取正式产物。

### 6.1 Sprint 2A-real 输入

```text id="h5dt7v"
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
```

### 6.2 Sprint 2B 输入

```text id="pmqr0j"
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

可选读取：

```text id="ujsmhn"
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
```

只能用于轻量计数和字段检查，不得重新计算 features。

### 6.3 Sprint 2C 输入

```text id="rvq9lw"
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

可选读取：

```text id="owphv2"
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
```

只能用于 target count cross-check。

### 6.4 Sprint 2D 输入

```text id="xdz7e3"
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
```

不得加载：

```text id="vxh5z6"
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
```

只允许检查该文件是否存在。

### 6.5 Sprint 2E 输入

```text id="hj0bwn"
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
```

### 6.6 Sprint 2F 输入

```text id="m16vz1"
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
```

---

## 7. 禁止输入

禁止读取：

```text id="l5kvx7"
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

禁止把以下 legacy/debug 文件作为输入契约：

```text id="cvqz00"
outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl
outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl
outputs/logs/sprint_2B_representation_features/feature_schema.json
outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json
```

如果这些文件存在，可以在 audit 中记录为 present，但不得依赖它们。

---

## 8. 禁止重跑的上游脚本

本 checkpoint 禁止重新运行：

```text id="tw8v5e"
scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
scripts/21_write_sprint_2_closed_loop_report.py
```

说明：

```text id="gwzxbt"
21_write_sprint_2_closed_loop_report.py 已属于 Sprint 2F。
本 checkpoint 读取 2F 输出，不重新生成 2F。
```

本 checkpoint 可以新增并运行：

```text id="nc2k5d"
scripts/22_write_sprint_2_stage_summary.py
```

---

## 9. 允许新增文件

允许新增：

```text id="vlvg2s"
src/recover_attention/stage_summary.py
scripts/22_write_sprint_2_stage_summary.py
tests/test_stage_summary.py
docs/codex_tasks/sprint_2_final_checkpoint_visualization_summary.md
```

---

## 10. 允许修改文件

允许修改：

```text id="b491wj"
PROGRESS.md
docs/progress/sprint_2_history.md
```

---

## 11. 禁止修改文件和目录

禁止修改：

```text id="wud6h6"
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/SKILL.md

data/processed/*

outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*
outputs/logs/sprint_2B_representation_features/*
outputs/logs/sprint_2C_probe_dataset/*
outputs/logs/sprint_2D_probe_training_baseline/*
outputs/logs/sprint_2E_guidance_candidate_dry_run/*
outputs/logs/sprint_2F_mini_closed_loop_report/*

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/probe_dataset.py
src/recover_attention/probe_training.py
src/recover_attention/guidance_candidates.py
src/recover_attention/closed_loop_report.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
scripts/21_write_sprint_2_closed_loop_report.py
```

---

## 12. 工作区状态处理

如果存在 pre-existing AM / untracked 文件：

```text id="jwf4bd"
1. 不主动回滚。
2. 不主动删除。
3. 不主动 git restore。
4. 不主动 git reset。
5. 只在 audit 中记录。
```

特别是如果仍存在：

```text id="p6hz69"
docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md
docs/codex_tasks/sprint_2F_mini_closed_loop_report.md
```

的 AM 状态，或 2E/2F 相关文件的 untracked 状态，不要当作 checkpoint 失败。

但本 checkpoint 新增的文件必须清楚列出。

---

## 13. 串行执行要求

由于 Sprint 2E 曾出现 Windows 并行 `conda run` 临时文件占用问题，本 checkpoint 必须串行执行命令。

禁止并行运行多个：

```text id="v29s66"
conda run ...
```

执行顺序必须是：

```text id="u1k88i"
1. git status --short
2. targeted pytest
3. stage summary generation command
4. full pytest with timing
5. git diff --name-only
6. git status --short
```

如果出现 Windows 临时文件占用：

```text id="x4bx6a"
1. 不立即判断为代码失败。
2. 确认是否并行启动过 conda run。
3. 串行重跑 targeted pytest。
4. 串行重跑 full pytest。
5. 若串行仍失败，再视为真实失败。
```

---

## 14. 可视化要求

本 checkpoint 必须生成以下 PNG 图。

所有图默认输出到：

```text id="wnzl3d"
outputs/logs/sprint_2_stage_summary/figures/
```

### 14.1 sprint_2_pipeline_overview.png

展示 Sprint 2 pipeline：

```text id="mg0760"
2A-real hidden states
→ 2B representation features
→ 2C probe dataset
→ 2D probe training
→ 2E guidance candidates
→ 2F closed-loop report
```

图中必须标明：

```text id="lu8ai9"
dry-run loop
not executed steering
```

### 14.2 probe_target_counts.png

展示 2C target 分布：

```text id="dzy8fp"
risk_positive
positive_anchor
negative
hard_negative_or_weak_positive
```

数据来自：

```text id="yhe49p"
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

### 14.3 probe_metrics_vs_baseline.png

展示 2D probe 与 majority baseline 的对比：

```text id="tf9rae"
probe accuracy
probe macro_f1
majority baseline accuracy
majority baseline macro_f1
```

数据来自：

```text id="igf84j"
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
```

### 14.4 guidance_candidate_action_counts.png

展示 2E candidate action 分布：

```text id="s567oh"
increase_attention_to_original_span
preserve_original_span_attention
review_before_guidance
no_guidance
```

数据来自：

```text id="rwou8u"
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

### 14.5 guidance_confidence_counts.png

展示 2E confidence 分布：

```text id="qil7ys"
high
medium
low
unknown
```

数据来自：

```text id="ddgkmb"
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

### 14.6 sprint_2_boundary_summary.png

展示边界状态：

```text id="pb5eoj"
hidden-state cache completed = true
representation features completed = true
probe dataset completed = true
probe training completed = true
guidance candidate dry run completed = true
attention steering executed = false
hallucination reduction validated = false
answer accuracy improvement validated = false
```

数据来自：

```text id="kqj9hz"
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
```

---

## 15. 可视化实现要求

建议使用：

```text id="nqyv2f"
matplotlib
```

不要使用 seaborn。

图必须可在无 GUI 环境下生成：

```python id="eyp98k"
import matplotlib
matplotlib.use("Agg")
```

不要依赖 notebook。

不要需要人工交互。

不要读取网络资源。

图中不要使用过度复杂样式。

---

## 16. 阶段性总结 Markdown 要求

输出：

```text id="lh5sjt"
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
```

必须包含以下小节。

### 16.1 Title / Metadata

推荐标题：

```text id="mti7ux"
# Sprint 2 Stage Summary
```

必须包含：

```text id="zbsqe8"
日期
项目名
阶段
状态
```

### 16.2 Executive Summary

必须写明：

```text id="q48qsm"
Sprint 2 completed a dry-run hidden-state-to-guidance-candidate loop.
```

同时必须写明：

```text id="qplgqw"
This is not an executed attention-steering loop.
```

### 16.3 Pipeline Summary

列出：

```text id="b0li46"
2A-real
2B
2C
2D
2E
2F
```

每一行包含：

```text id="ucga5b"
stage
goal
input
output
key count
status
```

### 16.4 Key Numbers

必须包含：

```text id="a6mbhz"
2A-real hidden state cache records / cases
2B representation feature records
2C target counts
2D accuracy / macro_f1 / majority baseline
2E candidate action counts
2E confidence counts
2F boundary audit results
full pytest result
full pytest duration if available
```

### 16.5 Figures

必须嵌入相对路径：

```text id="z8b7ns"
figures/sprint_2_pipeline_overview.png
figures/probe_target_counts.png
figures/probe_metrics_vs_baseline.png
figures/guidance_candidate_action_counts.png
figures/guidance_confidence_counts.png
figures/sprint_2_boundary_summary.png
```

### 16.6 Boundary and Non-claims

必须写明：

```text id="s4o8jb"
attention guidance 尚未执行
attention weights 尚未修改
attention mask 尚未注入
CoT 推理尚未在 guidance 下重跑
answer accuracy 尚未验证提升
hallucination reduction 尚未验证
```

### 16.7 Engineering Stability

必须写明：

```text id="jl2z5m"
Sprint 2E 曾出现 Windows 并行 conda run 临时文件占用。
串行重跑后通过。
后续 checkpoint / sprint 默认串行执行 targeted pytest、pipeline command、full pytest。
```

### 16.8 Sprint 3A Readiness

必须写明：

```text id="tceqet"
Sprint 2 可以支持进入 Sprint 3A：Attention Steering Interface Design。
```

但必须同时写明：

```text id="j78o2g"
Sprint 2 不足以支持声称 attention steering effectiveness。
```

推荐 Sprint 3A 起点：

```text id="nz0rlp"
Attention Steering Interface Design
```

---

## 17. Audit JSON 要求

输出：

```text id="kbqxyb"
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
```

建议字段：

```json id="u7m0ky"
{
  "stage": "sprint_2_final_checkpoint",
  "status": "ok",
  "summary_path": "outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md",
  "figure_paths": [],
  "inputs_read": {},
  "outputs_written": {},
  "full_pytest": {
    "command": "conda run -n recover_attention python -m pytest -q",
    "status": "passed",
    "passed": 508,
    "skipped": 2,
    "duration_seconds": null
  },
  "loop_status": {
    "hidden_state_cache": true,
    "representation_features": true,
    "probe_dataset": true,
    "probe_training": true,
    "guidance_candidate_dry_run": true,
    "closed_loop_report": true,
    "executed_attention_steering": false,
    "validated_answer_accuracy_improvement": false,
    "validated_hallucination_reduction": false
  },
  "boundary": {
    "dry_run_only": true,
    "called_hf_model": false,
    "called_ollama": false,
    "read_hidden_state_tensors": false,
    "retrained_probe": false,
    "reran_upstream_pipeline": false,
    "performed_attention_guidance": false
  },
  "windows_stability_note_present": true,
  "serial_execution_required": true,
  "warnings": []
}
```

如果无法自动获取 full pytest duration：

```text id="e9fd7v"
duration_seconds = null
```

并在 summary 中说明未记录耗时。

---

## 18. CLI 要求

新增脚本：

```text id="dnqi36"
scripts/22_write_sprint_2_stage_summary.py
```

推荐 CLI：

```bash id="hlruxu"
conda run -n recover_attention python scripts/22_write_sprint_2_stage_summary.py \
  --output-dir outputs/logs/sprint_2_stage_summary \
  --backend sprint_2_stage_summary_v0 \
  --overwrite
```

可选传入 full pytest 结果：

```bash id="php42o"
--full-pytest-passed 508 \
--full-pytest-skipped 2 \
--full-pytest-duration-seconds <seconds>
```

如果没有记录耗时，不传 `--full-pytest-duration-seconds`。

---

## 19. 实现建议

核心实现放在：

```text id="em7dbh"
src/recover_attention/stage_summary.py
```

建议函数：

```python id="j5psnr"
load_json(...)
read_jsonl_count(...)
collect_2A_real_summary(...)
collect_2B_summary(...)
collect_2C_summary(...)
collect_2D_summary(...)
collect_2E_summary(...)
collect_2F_summary(...)
build_pipeline_overview_figure(...)
build_probe_target_counts_figure(...)
build_probe_metrics_vs_baseline_figure(...)
build_guidance_candidate_action_counts_figure(...)
build_guidance_confidence_counts_figure(...)
build_boundary_summary_figure(...)
build_stage_summary_markdown(...)
build_stage_summary_audit(...)
write_stage_summary(...)
```

---

## 20. 测试要求

新增：

```text id="g6lglv"
tests/test_stage_summary.py
```

至少覆盖：

```text id="sr65xz"
1. 能读取 2A-real report。
2. 能读取 2B report。
3. 能读取 2C report。
4. 能读取 2D report。
5. 能读取 2E report。
6. 能读取 2F report / audit。
7. 能生成 sprint_2_stage_summary.md。
8. 能生成 sprint_2_stage_summary_audit.json。
9. 能生成所有必需 PNG figures。
10. summary 包含 dry-run boundary。
11. summary 明确 attention guidance 未执行。
12. summary 明确 hallucination reduction 未验证。
13. summary 包含 Windows 串行执行说明。
14. summary 包含 Sprint 3A readiness。
15. audit 标记 executed_attention_steering=false。
16. audit 标记 validated_hallucination_reduction=false。
17. 不读取 hidden-state tensors。
18. 不加载 probe_model.pkl。
19. 不重跑 upstream pipeline scripts。
20. CLI smoke test 能生成所有正式输出。
```

---

## 21. 必须运行的命令

Preflight：

```bash id="u7f370"
git status --short
```

Targeted test：

```bash id="rw8kl8"
conda run -n recover_attention python -m pytest tests/test_stage_summary.py -q
```

Generate stage summary：

```bash id="egk9n4"
conda run -n recover_attention python scripts/22_write_sprint_2_stage_summary.py \
  --output-dir outputs/logs/sprint_2_stage_summary \
  --backend sprint_2_stage_summary_v0 \
  --overwrite
```

Full pytest：

```bash id="oqy6vt"
conda run -n recover_attention python -m pytest -q
```

Windows PowerShell full pytest with timing：

```powershell id="mz0c4k"
Measure-Command { conda run -n recover_attention python -m pytest -q }
```

Final check：

```bash id="yz7bo3"
git diff --name-only
git status --short
```

注意：

```text id="atzm7u"
以上命令必须串行执行。
不要并行启动多个 conda run。
```

---

## 22. full pytest 耗时记录方式

如果在 Windows PowerShell 中运行：

```powershell id="orvrjd"
$duration = Measure-Command { conda run -n recover_attention python -m pytest -q }
$duration.TotalSeconds
```

可以把秒数传入：

```bash id="mxy6kt"
--full-pytest-duration-seconds <seconds>
```

如果没有自动写入，不要为了记录耗时重跑多次。

在 summary 中写：

```text id="edk25x"
Full pytest passed: 508 passed, 2 skipped.
Duration: not recorded.
```

或：

```text id="r8a0vr"
Full pytest passed: 508 passed, 2 skipped.
Duration: <seconds> seconds.
```

---

## 23. PROGRESS.md 更新建议

完成后在当前阶段补充：

```text id="n1mw55"
Sprint 2 Final Checkpoint and Visualization Summary 已完成。

已串行执行 full pytest，并生成 Sprint 2 阶段性总结与可视化。

正式输出：
- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
- outputs/logs/sprint_2_stage_summary/figures/*.png

核心结论：
- Sprint 2 已形成 hidden-state cache → representation features → probe dataset → probe training → guidance candidate manifest → closed-loop report 的 dry-run 闭环。
- Attention guidance 尚未执行。
- Hallucination reduction 尚未验证。
- Sprint 3A 可以开始设计 Attention Steering Interface。

工程说明：
- 后续 Windows 本地环境继续串行执行 targeted pytest、pipeline command、full pytest。
```

---

## 24. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text id="o4k6u8"
## Sprint 2 Final Checkpoint and Visualization Summary

### Goal

Produce a final Sprint 2 checkpoint package with full pytest confirmation, stage summary markdown, and visualization figures.

### Inputs

- Sprint 2A-real reports
- Sprint 2B representation feature report
- Sprint 2C probe dataset report
- Sprint 2D probe eval report
- Sprint 2E guidance candidate report
- Sprint 2F closed-loop report and audit

### Outputs

- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
- outputs/logs/sprint_2_stage_summary/figures/*.png

### Notes

- This checkpoint does not rerun upstream pipeline scripts.
- This checkpoint does not train a probe.
- This checkpoint does not perform attention steering.
- This checkpoint does not validate hallucination reduction.
- Full pytest was run serially.
- Windows serial execution requirement remains in effect.

### Next

Sprint 3A：Attention Steering Interface Design.
```

---

## 25. 验收标准

本 checkpoint 完成时必须满足：

```text id="ov6ufh"
1. 生成 sprint_2_stage_summary.md。
2. 生成 sprint_2_stage_summary_audit.json。
3. 生成 sprint_2_pipeline_overview.png。
4. 生成 probe_target_counts.png。
5. 生成 probe_metrics_vs_baseline.png。
6. 生成 guidance_candidate_action_counts.png。
7. 生成 guidance_confidence_counts.png。
8. 生成 sprint_2_boundary_summary.png。
9. 串行运行 targeted pytest。
10. 串行运行 full pytest。
11. 记录 full pytest 结果。
12. 如可行，记录 full pytest 耗时。
13. summary 明确 Sprint 2 是 dry-run loop。
14. summary 明确 attention guidance 尚未执行。
15. summary 明确 hallucination reduction 尚未验证。
16. summary 包含 Windows 串行执行说明。
17. summary 包含 Sprint 3A readiness。
18. audit 标记 executed_attention_steering=false。
19. audit 标记 validated_answer_accuracy_improvement=false。
20. audit 标记 validated_hallucination_reduction=false。
21. 不读取 hidden-state tensors。
22. 不加载 probe_model.pkl。
23. 不重跑 16/17/18/19/20/21。
24. 不修改 Sprint 2A～2F outputs。
25. PROGRESS.md 已更新。
26. docs/progress/sprint_2_history.md 已更新。
```

---

## 26. 下一步

本 checkpoint 完成后，Sprint 2 正式结项。

下一步进入：

```text id="ssz4ma"
Sprint 3A：Attention Steering Interface Design
```

Sprint 3A 才开始设计真实 attention steering interface。

在 Sprint 3A 之前，不得把 Sprint 2 结果表述为：

```text id="s58jwa"
attention guidance 已执行
hallucination reduction 已验证
answer accuracy 已提升
```

正确表述是：

```text id="ogz407"
Sprint 2 established a dry-run signal path from hidden states to planned guidance candidates.
```
