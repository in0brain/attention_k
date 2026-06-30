# Sprint 2F：Mini Closed-loop Report

## 0. Sprint 名称

Sprint 2F — Mini Closed-loop Report

## 1. 当前项目状态

当前项目为：

```text
Reasoning-Aware Attention Guidance
```

Sprint 2 最小闭环为：

```text
Sprint 1R manifest
        ↓
2A hidden state cache
        ↓
2B representation features
        ↓
2C probe dataset
        ↓
2D probe baseline
        ↓
2E guidance candidate manifest dry run
        ↓
2F mini closed-loop report
```

当前 Sprint 2 已完成的阶段包括：

```text
Sprint 2A：Hidden State Cache Baseline
Sprint 2A-real：Real Hidden State Cache Run
Sprint 2B：Representation Feature Extraction
Sprint 2B-fix：Representation Feature Extraction Scope Alignment
Sprint 2C：Probe Dataset Construction
Sprint 2D：Probe Training Baseline
Sprint 2E：Guidance Candidate Manifest Dry Run
```

Sprint 2F 的任务不是继续实验，而是总结 Sprint 2 是否形成了一个最小可审查闭环。

---

## 2. 2F 目标

Sprint 2F 的目标是：

```text
总结 Sprint 2 是否形成最小 hidden-state → probe → guidance candidate 的 dry-run 闭环。
```

正式回答以下 6 个问题：

```text
1. hidden state 是否能稳定缓存？
2. token/span/mask 是否能对齐？
3. wrong_numeric / generic / entity drift 在表征上是否有差异？
4. probe 是否能产生非随机预测？
5. probe 预测能否生成合理 guidance candidate？
6. 哪些问题必须在 Sprint 3 前修？
```

Sprint 2F 的正式产物是：

```text
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
```

---

## 3. 2F 的核心结论边界

Sprint 2F 必须明确写出：

```text
Sprint 2 formed a dry-run hidden-state-to-guidance-candidate loop,
not an executed attention-steering loop.
```

也就是说，Sprint 2 最多证明：

```text
hidden states 可以缓存
representation features 可以抽取
human labels 可以转成 probe targets
probe 可以训练并输出 prediction
prediction 可以转成 planned-only guidance candidate
```

Sprint 2 不能证明：

```text
attention guidance 已执行
transformer attention weights 已修改
attention mask 已注入
CoT 推理已重跑
answer accuracy 已提升
hallucination reduction 已验证
```

这些必须在报告中明确写出，不能模糊表述。

---

## 4. 2E 遗留问题必须处理

Sprint 2F 必须显式处理 Sprint 2E 的两个遗留问题。

### 4.1 Dry-run candidate manifest 不等于 attention guidance

Sprint 2E 只是：

```text
guidance candidate manifest dry run
```

它只表示：

```text
hidden-state signal
→ probe prediction
→ planned-only guidance candidate
```

它不表示：

```text
attention guidance 已执行
hallucination reduction 已验证
answer accuracy 已提升
```

Sprint 2F 报告中必须包含一个独立小节：

```text
Dry-run Boundary and Non-claims
```

该小节必须写明：

```text
- Sprint 2E did not modify model attention.
- Sprint 2E did not inject an attention mask.
- Sprint 2E did not rerun CoT generation.
- Sprint 2E did not evaluate answer accuracy.
- Sprint 2E did not validate hallucination reduction.
- Sprint 2E only generated planned-only guidance candidates.
```

不得在报告中出现以下说法：

```text
attention guidance succeeded
attention steering improved reasoning
hallucination was reduced
answer accuracy improved
closed-loop intervention was validated
```

除非明确加上否定或边界说明。

---

### 4.2 Windows 并行 conda run 临时文件占用问题

Sprint 2E 执行过程中出现过一次 Windows 临时文件占用问题。

已知情况：

```text
并行启动两个 conda run，导致 Windows 临时文件被占用。
```

处理结果：

```text
串行重跑后通过。
```

Sprint 2F 报告中必须将其归类为：

```text
engineering stability issue
```

而不是：

```text
pipeline logic failure
model failure
test design failure
```

Sprint 2F 报告中必须给出后续执行规范：

```text
1. 不并行启动多个 conda run。
2. targeted pytest、pipeline command、full pytest 必须串行执行。
3. 如果 Windows 环境出现临时文件占用错误，先串行重跑验证。
4. 只有串行重跑仍失败，才视为真实测试失败。
```

---

## 5. 非目标

Sprint 2F 不做：

```text
不重新缓存 hidden states
不重新抽取 representation features
不重新构造 probe dataset
不重新训练 probe
不重新生成 guidance candidate manifest
不读取 hidden-state tensors
不调用 HF model
不调用 Ollama
不调用 tokenizer
不调用 NLI model
不修改 attention weights
不注入 attention mask
不重跑 CoT 推理
不评估 answer accuracy
不声称 hallucination reduction
不做 Sprint 3 steering 实验
```

Sprint 2F 是报告 sprint，不是实验 sprint。

---

## 6. 必须阅读的文件

开始实现前，先阅读：

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/progress/sprint_2_history.md

docs/codex_tasks/sprint_2A_hidden_state_cache_baseline.md
docs/codex_tasks/sprint_2A_real_hidden_state_cache_run.md
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
docs/codex_tasks/sprint_2C_probe_dataset_construction.md
docs/codex_tasks/sprint_2D_probe_training_baseline.md
docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md
```

如果本 task card 已存在，也阅读：

```text
docs/codex_tasks/sprint_2F_mini_closed_loop_report.md
```

---

## 7. 输入文件

Sprint 2F 应读取各阶段正式 report 和关键 manifest。

### 7.1 Sprint 2A / 2A-real 输入

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json

outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
```

### 7.2 Sprint 2B 输入

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

2F 可以读取 `representation_features.jsonl` 做轻量统计，但不得重新计算 features。

不得读取：

```text
outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl
outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl
outputs/logs/sprint_2B_representation_features/feature_schema.json
```

这些 legacy/debug files 不是 2F 的输入契约。

### 7.3 Sprint 2C 输入

```text
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

### 7.4 Sprint 2D 输入

```text
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
```

2F 可以读取：

```text
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
```

但只允许检查文件是否存在，不得加载模型，不得重新预测。

### 7.5 Sprint 2E 输入

```text
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

---

## 8. 禁止输入

Sprint 2F 禁止读取：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

Sprint 2F 禁止重新运行：

```text
scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
```

除非用户明确要求重新生成上游产物。

默认 2F 只读上游正式输出。

---

## 9. 输出目录和正式输出

默认输出目录：

```text
outputs/logs/sprint_2F_mini_closed_loop_report/
```

正式输出：

```text
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
```

可选机器可读 audit 文件：

```text
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
```

如果生成 audit JSON，它只能作为检查辅助，不是论文或后续实验输入契约。

---

## 10. 允许新增文件

允许新增：

```text
src/recover_attention/closed_loop_report.py
scripts/21_write_sprint_2_closed_loop_report.py
tests/test_closed_loop_report.py
docs/codex_tasks/sprint_2F_mini_closed_loop_report.md
```

---

## 11. 允许修改文件

允许修改：

```text
PROGRESS.md
docs/progress/sprint_2_history.md
```

---

## 12. 禁止修改文件和目录

禁止修改：

```text
AGENTS.md
README.md
docs/skill/SKILL.md

data/processed/*

outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*
outputs/logs/sprint_2B_representation_features/*
outputs/logs/sprint_2C_probe_dataset/*
outputs/logs/sprint_2D_probe_training_baseline/*
outputs/logs/sprint_2E_guidance_candidate_dry_run/*

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/probe_dataset.py
src/recover_attention/probe_training.py
src/recover_attention/guidance_candidates.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
```

2F 必须把 2A～2E 输出全部视为 frozen upstream artifacts。

---

## 13. 报告结构要求

`sprint_2_minimal_closed_loop_report.md` 必须包含以下一级或二级小节。

### 13.1 Executive Summary

必须回答：

```text
Sprint 2 是否形成了最小闭环？
```

推荐结论格式：

```text
Sprint 2 formed a minimal dry-run hidden-state-to-guidance-candidate loop.

The loop is:
hidden-state cache → representation features → probe dataset → probe baseline → guidance candidate manifest dry run.

This is not yet an executed attention-steering loop.
```

---

### 13.2 Sprint 2 Pipeline Overview

必须列出：

```text
2A / 2A-real：hidden state cache
2B：representation features
2C：probe dataset
2D：probe training baseline
2E：guidance candidate dry run
2F：closed-loop report
```

每个阶段至少写：

```text
goal
input
output
status
key counts
not-done boundary
```

---

### 13.3 Question 1：hidden state 是否能稳定缓存？

必须引用并总结：

```text
2A-real backend
num_cases
num_inputs_total
num_hidden_state_files
layer_indices
output_dir
```

必须回答：

```text
是否稳定缓存？
是否有缺失文件？
是否修改上游输入？
是否需要 Sprint 3 前改进？
```

---

### 13.4 Question 2：token/span/mask 是否能对齐？

必须引用并总结：

```text
single_mask_cases
group_mask_cases
fragment_recovery_outputs
alignment_warning_count
```

必须明确：

```text
当前对齐够支持 Sprint 2 最小闭环。
但 fragment recovery / span overlap nullable 仍是 Sprint 3 前的风险点。
```

---

### 13.5 Question 3：wrong_numeric / generic / entity drift 在表征上是否有差异？

必须基于 2B / 2C / 2D 的统计作保守回答。

可以总结：

```text
representation_features records
position_pool_feature_null_records
probe target counts
feature signal summary
top weighted features
feature group weight summary
layer weight summary
```

必须避免过度结论。

推荐表述：

```text
There are diagnostic signals in the current 20-record sample, but the sample is too small to claim stable generalizable representation separation.
```

不得写成：

```text
wrong_numeric / generic / entity drift 已被可靠区分
```

---

### 13.6 Question 4：probe 是否能产生非随机预测？

必须引用并总结 2D：

```text
num_predictions
cv_strategy
accuracy
macro_f1
weighted_f1
binary_anchor_or_risk_accuracy
majority_baseline_accuracy
majority_baseline_macro_f1
```

必须明确：

```text
probe 产生了高于 majority baseline 的 diagnostic signal。
但由于样本只有 20 条，不能声称泛化性能。
```

---

### 13.7 Question 5：probe 预测能否生成合理 guidance candidate？

必须引用并总结 2E：

```text
num_guidance_candidate_records
candidate_action_counts
predicted_target_counts
confidence_counts
execution_status
dry_run
```

必须明确：

```text
probe predictions can be mapped into planned-only guidance candidates.
```

同时必须写明：

```text
These guidance candidates were not executed.
```

---

### 13.8 Question 6：Sprint 3 前必须修什么？

必须列出至少以下问题：

```text
1. Dry-run boundary:
   Sprint 2 尚未执行真实 attention steering。

2. Evaluation boundary:
   尚未验证 answer accuracy improvement 或 hallucination reduction。

3. Data scale:
   当前只有 20 条 human-reviewed examples，样本太小。

4. Token/span alignment:
   fragment recovery 和 nullable span/mask-position features 需要更稳健处理。

5. Probe robustness:
   需要更多样本、更稳定 split、更严格 held-out evaluation。

6. Guidance design:
   需要定义真实 steering 的 intervention point、attention modification method、controlled generation protocol。

7. Engineering stability:
   Windows 环境不要并行启动多个 conda run；测试和 pipeline command 默认串行。
```

---

### 13.9 Dry-run Boundary and Non-claims

必须包含：

```text
Sprint 2E produced planned-only guidance candidates.
Sprint 2 did not execute attention steering.
Sprint 2 did not modify transformer attention weights.
Sprint 2 did not rerun CoT generation under guidance.
Sprint 2 did not evaluate answer accuracy under guidance.
Sprint 2 did not validate hallucination reduction.
```

---

### 13.10 Windows Execution Stability Note

必须包含：

```text
A Windows temporary file lock occurred once when two conda run commands were launched in parallel.
Serial rerun passed.
This is recorded as an engineering stability issue, not a pipeline logic failure.
Future runs should execute targeted pytest, pipeline command, and full pytest serially.
```

---

### 13.11 Sprint 3 Readiness

必须回答：

```text
Sprint 2 是否足够支持进入 Sprint 3？
```

推荐结论：

```text
Sprint 2 is sufficient to justify designing Sprint 3 attention steering experiments, but not sufficient to claim steering effectiveness.
```

必须列出 Sprint 3 第一阶段建议：

```text
Sprint 3A：Attention Steering Interface Design
Sprint 3B：No-op / dry-run steering executor
Sprint 3C：Controlled attention intervention pilot
Sprint 3D：Answer-level evaluation
```

这只是建议，不要在 2F 中实现。

---

## 14. Markdown 报告格式要求

报告必须是清晰的 Markdown。

必须包含：

```text
title
date
status summary
pipeline table
six-question answers
dry-run boundary
Windows execution stability note
Sprint 3 before-fix list
```

推荐标题：

```text
# Sprint 2 Minimal Closed-loop Report
```

推荐开头：

```text
日期：YYYY-MM-DD
项目：Reasoning-Aware Attention Guidance
阶段：Sprint 2F
状态：Sprint 2 dry-run minimal loop completed
```

日期使用运行当天的本地日期即可。

---

## 15. 可选 audit JSON

如果生成：

```text
sprint_2_minimal_closed_loop_audit.json
```

建议包含：

```json
{
  "sprint": "2F",
  "status": "ok",
  "report_path": "outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md",
  "loop_status": {
    "hidden_state_cache": true,
    "representation_features": true,
    "probe_dataset": true,
    "probe_training": true,
    "guidance_candidate_dry_run": true,
    "executed_attention_steering": false,
    "validated_hallucination_reduction": false
  },
  "required_boundary_statements_present": true,
  "windows_serial_execution_note_present": true,
  "warnings": []
}
```

---

## 16. CLI 要求

新增脚本：

```text
scripts/21_write_sprint_2_closed_loop_report.py
```

推荐 CLI：

```bash
conda run -n recover_attention python scripts/21_write_sprint_2_closed_loop_report.py \
  --output-dir outputs/logs/sprint_2F_mini_closed_loop_report \
  --backend sprint_2_closed_loop_report_v0 \
  --overwrite
```

也可以显式传入输入路径：

```bash
conda run -n recover_attention python scripts/21_write_sprint_2_closed_loop_report.py \
  --hidden-cache-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json \
  --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json \
  --representation-feature-report outputs/logs/sprint_2B_representation_features/representation_feature_report.json \
  --probe-dataset-report outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json \
  --probe-eval-report outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json \
  --guidance-candidate-report outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json \
  --output-dir outputs/logs/sprint_2F_mini_closed_loop_report \
  --backend sprint_2_closed_loop_report_v0 \
  --overwrite
```

不得提供：

```text
--model-path
--device
--train
--steer
--run-generation
--evaluate-accuracy
```

2F 不训练、不 steering、不生成答案。

---

## 17. 实现建议

核心实现放在：

```text
src/recover_attention/closed_loop_report.py
```

建议函数：

```python
load_json(...)
load_jsonl_count(...)
collect_sprint_2A_summary(...)
collect_sprint_2B_summary(...)
collect_sprint_2C_summary(...)
collect_sprint_2D_summary(...)
collect_sprint_2E_summary(...)
build_pipeline_table(...)
build_six_question_answers(...)
build_dry_run_boundary_section(...)
build_windows_stability_section(...)
build_sprint_3_readiness_section(...)
build_markdown_report(...)
build_audit_json(...)
write_report(...)
```

---

## 18. 测试要求

新增：

```text
tests/test_closed_loop_report.py
```

至少覆盖：

```text
1. 能读取 2A-real hidden_state_cache_report.json。
2. 能读取 2A-real token_alignment_report.json。
3. 能读取 2B representation_feature_report.json。
4. 能读取 2C probe_dataset_report.json。
5. 能读取 2D probe_eval_report.json。
6. 能读取 2E guidance_candidate_report.json。
7. 能生成 sprint_2_minimal_closed_loop_report.md。
8. 报告包含 6 个核心问题。
9. 报告包含 dry-run boundary。
10. 报告明确 attention guidance 未执行。
11. 报告明确 hallucination reduction 未验证。
12. 报告不声称 answer accuracy 提升。
13. 报告包含 Windows temporary file / conda run 串行执行说明。
14. 报告把 Windows 问题归类为 engineering stability issue。
15. 报告包含 Sprint 3 前待修问题。
16. 报告不读取 hidden-state tensors。
17. 报告不加载 probe_model.pkl。
18. 报告不重新运行 upstream scripts。
19. audit JSON 若生成，必须标记 executed_attention_steering=false。
20. CLI smoke test 能生成正式 markdown report。
```

---

## 19. 串行执行要求

Sprint 2F 必须明确要求串行执行命令。

不要并行运行：

```text
conda run ...
conda run ...
```

尤其在 Windows 环境下，不要同时启动多个 conda run。

推荐执行顺序：

```text
1. git status --short
2. targeted pytest
3. report generation command
4. full pytest
5. git diff --name-only
6. git status --short
```

如果出现 Windows 临时文件占用：

```text
1. 不立即判断为代码失败。
2. 确认是否并行启动过 conda run。
3. 串行重跑 targeted pytest。
4. 串行重跑 full pytest。
5. 若串行仍失败，再视为真实失败。
```

---

## 20. 必须运行的命令

Preflight：

```bash
git status --short
```

Targeted test：

```bash
conda run -n recover_attention python -m pytest tests/test_closed_loop_report.py -q
```

Run 2F report:

```bash
conda run -n recover_attention python scripts/21_write_sprint_2_closed_loop_report.py \
  --output-dir outputs/logs/sprint_2F_mini_closed_loop_report \
  --backend sprint_2_closed_loop_report_v0 \
  --overwrite
```

Full test：

```bash
conda run -n recover_attention python -m pytest -q
```

Final check：

```bash
git diff --name-only
git status --short
```

可选记录 full pytest 耗时：

Windows PowerShell：

```powershell
Measure-Command { conda run -n recover_attention python -m pytest -q }
```

Linux / macOS：

```bash
time conda run -n recover_attention python -m pytest -q
```

如果记录耗时，可写入 `sprint_2_minimal_closed_loop_report.md` 的 engineering notes。

---

## 21. 验收标准

完成 Sprint 2F 时必须满足：

```text
1. 新增 Sprint 2 mini closed-loop report。
2. 报告读取 2A-real / 2B / 2C / 2D / 2E 正式 report。
3. 报告输出 sprint_2_minimal_closed_loop_report.md。
4. 报告回答 hidden state 是否能稳定缓存。
5. 报告回答 token/span/mask 是否能对齐。
6. 报告回答 wrong_numeric / generic / entity drift 是否有表征差异。
7. 报告回答 probe 是否能产生非随机预测。
8. 报告回答 probe 预测能否生成合理 guidance candidate。
9. 报告列出 Sprint 3 前必须修的问题。
10. 报告明确 Sprint 2 是 dry-run hidden-state-to-guidance-candidate loop。
11. 报告明确 attention guidance 尚未执行。
12. 报告明确 transformer attention weights 尚未修改。
13. 报告明确 CoT 推理尚未重跑。
14. 报告明确 answer accuracy 尚未验证提升。
15. 报告明确 hallucination reduction 尚未验证。
16. 报告不得声称 hallucination reduction。
17. 报告不得声称 attention steering 已完成。
18. 报告记录 Windows 并行 conda run 临时文件占用问题。
19. 报告说明该问题已通过串行重跑解决。
20. 报告将该问题归类为 engineering stability issue。
21. 报告给出后续串行执行规范。
22. 不读取 hidden-state tensors。
23. 不加载 probe_model.pkl。
24. 不重新训练 probe。
25. 不重新生成 guidance candidate manifest。
26. 不修改 Sprint 1Q / 1R / 2A / 2B / 2C / 2D / 2E outputs。
27. targeted pytest passed。
28. full pytest passed。
29. PROGRESS.md 已更新。
30. docs/progress/sprint_2_history.md 已更新。
```

---

## 22. PROGRESS.md 更新建议

完成后在当前阶段补充：

```text
Sprint 2F 已完成：Mini Closed-loop Report。

sprint_2_closed_loop_report_v0 已读取 Sprint 2A-real / 2B / 2C / 2D / 2E 的正式报告与产物，生成 Sprint 2 最小闭环总结。

正式输出：
- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md

核心结论：
- Sprint 2 形成了 hidden-state → representation features → probe dataset → probe training → guidance candidate manifest 的 dry-run 闭环。
- Sprint 2E 只是 planned-only guidance candidate dry run。
- Attention guidance 尚未执行。
- Transformer attention weights 尚未修改。
- CoT 推理尚未在 guidance 下重跑。
- Answer accuracy 尚未验证提升。
- Hallucination reduction 尚未验证。

工程稳定性：
- Sprint 2E 中曾因并行启动两个 conda run 出现 Windows 临时文件占用。
- 串行重跑后通过。
- 后续 sprint 默认串行执行 targeted pytest、pipeline command、full pytest。

下一步建议是 Sprint 3A：Attention Steering Interface Design。
```

---

## 23. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text
## Sprint 2F：Mini Closed-loop Report

### Goal

Summarize whether Sprint 2 formed a minimal hidden-state-to-guidance-candidate dry-run loop.

### Inputs

- Sprint 2A-real hidden state cache report
- Sprint 2A-real token alignment report
- Sprint 2B representation feature report
- Sprint 2C probe dataset report
- Sprint 2D probe eval report
- Sprint 2E guidance candidate report

### Output

- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md

### Core Conclusion

Sprint 2 formed a dry-run loop:

hidden-state cache → representation features → probe dataset → probe training → guidance candidate manifest.

This is not an executed attention-steering loop.

### Non-claims

- No attention guidance was executed.
- No transformer attention weights were modified.
- No CoT generation was rerun under guidance.
- No answer accuracy improvement was evaluated.
- No hallucination reduction was validated.

### Engineering Stability Note

During Sprint 2E, one Windows temporary file lock occurred when two conda run commands were launched in parallel.

Serial rerun passed.

This is recorded as an engineering stability issue, not a pipeline logic failure.

Future runs should execute targeted pytest, pipeline command, and full pytest serially.

### Next

Sprint 3A：Attention Steering Interface Design.
```

---

## 24. 下一步边界

Sprint 2F 完成后，Sprint 2 结束。

下一阶段进入：

```text
Sprint 3A：Attention Steering Interface Design
```

Sprint 3 才开始设计真正的 attention steering。

Sprint 3 之前不得把 Sprint 2 的 dry-run 结果写成：

```text
已执行 attention guidance
已降低 hallucination
已提升 answer accuracy
```

正确表述是：

```text
Sprint 2 established a minimal dry-run signal path from hidden states to planned guidance candidates.
```
