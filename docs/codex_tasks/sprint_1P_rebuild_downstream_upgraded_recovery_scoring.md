# Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring

建议保存路径：

```text id="g7f7qi"
docs/codex_tasks/sprint_1P_rebuild_downstream_upgraded_recovery_scoring.md
```

---

## 1. 目标

本 sprint 使用 Sprint 1O 生成的 upgraded recovery scores 重建 downstream。

当前已经完成：

```text id="7t97n3"
Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs
Sprint 1O：Upgrade Real Recovery Scoring
```

Sprint 1N 已经生成 real downstream baseline：

```text id="l2n875"
outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
```

其中 `recover_scores_real.jsonl` 使用的是：

```text id="jlksb2"
stub_rule_v0
```

Sprint 1O 已经生成 upgraded recovery scores：

```text id="74va7f"
outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
```

其中 `recover_scores_nli_judge.jsonl` 使用的是：

```text id="yjr03f"
nli_recovery_judge_v0
```

本 sprint 的目标是：

```text id="0c6fmh"
使用同一份 semantic_labels_real.jsonl
+
使用 upgraded recover_scores_nli_judge.jsonl
→ 重建 unit_evidence
→ 重建 attention_anchor_labels
→ 重建 intervention_manifest
→ 生成 upgraded_downstream_report
```

---

## 2. 本 sprint 的核心问题

要回答的问题是：

```text id="lwc0oo"
把 recover scoring 从 stub_rule_v0 换成 nli_recovery_judge_v0 后，
downstream 的 unit_evidence / attention_anchor_labels / intervention_manifest 发生了什么变化？
```

重点不是重新生成数据，而是比较：

```text id="f90jkx"
1N baseline downstream
vs
1P upgraded-scoring downstream
```

---

## 3. 本 sprint 做什么

本 sprint 应完成：

```text id="g59apn"
1. 读取 1N 的 semantic_labels_real.jsonl。
2. 读取 1O 的 recover_scores_nli_judge.jsonl。
3. 重建 unit_evidence_upgraded.jsonl。
4. 重建 attention_anchor_labels_upgraded.jsonl。
5. 重建 intervention_manifest_upgraded.jsonl。
6. 生成 upgraded_downstream_report.json。
7. 可选生成 upgraded_downstream_report.md。
8. 更新 PROGRESS.md。
9. 更新 docs/progress/sprint_1_history.md。
```

---

## 4. 本 sprint 不做什么

本 sprint 禁止做：

```text id="z1k6jx"
1. 不重新运行 NLI scoring。
2. 不重新调用 Ollama。
3. 不重新生成 recover_outputs。
4. 不修改 nli_recovery_judge_v0。
5. 不修改 stub_rule_v0。
6. 不修改 semantic_labels.py。
7. 不修改 recover_scoring.py。
8. 不修改 unit_evidence.py。
9. 不修改 attention_anchor_labels.py。
10. 不修改 intervention_manifest.py。
11. 不修改 schema。
12. 不修改 interface docs。
13. 不接入 hidden states。
14. 不接入 attention maps。
15. 不接入 trajectory stability。
16. 不接入 answer stability。
17. 不接入 attention guidance。
18. 不做 probe training。
19. 不覆盖 data/processed baseline。
20. 不覆盖 Sprint 1N / Sprint 1O outputs。
21. 不做目录重构。
```

---

## 5. 输入文件

本 sprint 必须读取以下输入。

### 5.1 1N baseline inputs / comparison targets

```text id="0sf4ns"
outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
```

### 5.2 1O upgraded recovery scoring input

```text id="zbe4g1"
outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
```

---

## 6. 输出目录

新增 isolated output directory：

```text id="u0ov5h"
outputs/logs/sprint_1P_upgraded_downstream/
```

本 sprint 所有新产物都必须写入该目录。

---

## 7. 输出文件

生成：

```text id="cnz7ze"
outputs/logs/sprint_1P_upgraded_downstream/unit_evidence_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/attention_anchor_labels_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/intervention_manifest_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.json
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.md
```

不要生成或覆盖：

```text id="h08njt"
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
outputs/logs/sprint_1O_recovery_scoring/*
```

---

## 8. 允许修改

本 sprint 允许修改：

```text id="oyjiof"
scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
tests/test_rebuild_downstream_upgraded_recovery_scoring.py
PROGRESS.md
docs/progress/sprint_1_history.md
configs/v0_nli_small.yaml
```

如果以下文件不存在，可以新增：

```text id="qoabwy"
scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
tests/test_rebuild_downstream_upgraded_recovery_scoring.py
```

允许由运行命令生成：

```text id="dza8j6"
outputs/logs/sprint_1P_upgraded_downstream/unit_evidence_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/attention_anchor_labels_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/intervention_manifest_upgraded.jsonl
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.json
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.md
```

---

## 9. 禁止修改

本 sprint 禁止修改：

```text id="q8kjmt"
README.md
AGENTS.md
docs/skill/SKILL.md
docs/skill/codex_tasks.md
docs/skill/experiment_guide.md
docs/skill/method.md
docs/skill/label_schema.md
docs/skill/nli_scores_interface.md
docs/skill/semantic_labels_interface.md
docs/skill/masked_questions_interface.md
docs/skill/recover_outputs_interface.md
docs/skill/recover_scores_interface.md
docs/skill/unit_evidence_interface.md
docs/skill/attention_anchor_labels_interface.md
docs/skill/intervention_manifest_interface.md
docs/reference/*
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
src/recover_attention/schemas.py
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
scripts/13_rebuild_downstream_real_signals.py
tests/test_nli_scoring.py
tests/test_semantic_labels.py
tests/test_masked_questions.py
tests/test_recover_generation.py
tests/test_recover_scoring.py
tests/test_unit_evidence.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
tests/test_rebuild_downstream_real_signals.py
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
outputs/logs/sprint_1O_recovery_scoring/*
models/*
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因。

---

## 10. 开始前必须读取

开始前必须读取：

```text id="tmjuvv"
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/skill/codex_tasks.md
configs/v0_nli_small.yaml
src/recover_attention/data_io.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
docs/progress/sprint_1_history.md
```

可以读取但不要修改：

```text id="wou2yb"
docs/skill/unit_evidence_interface.md
docs/skill/attention_anchor_labels_interface.md
docs/skill/intervention_manifest_interface.md
```

不要读取：

```text id="b57uuc"
docs/reference/*
```

除非用户另行明确要求。

---

## 11. Preflight 要求

修改任何文件前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text id="te1fiw"
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1O 已完成。
6. 当前 1N semantic_labels_real.jsonl 是否存在。
7. 当前 1N recover_scores_real.jsonl 是否存在。
8. 当前 1N unit_evidence_real.jsonl 是否存在。
9. 当前 1N attention_anchor_labels_real.jsonl 是否存在。
10. 当前 1N intervention_manifest_real.jsonl 是否存在。
11. 当前 1N real_signal_report.json 是否存在。
12. 当前 1O recover_scores_nli_judge.jsonl 是否存在。
13. 当前 1O recovery_scoring_report.json 是否存在。
14. 1N recover_scores_real record 数量。
15. 1O recover_scores_nli_judge record 数量。
16. 1N 与 1O recover score masked_id 集合是否一致。
17. 1N semantic_labels_real record 数量。
18. 本 sprint 是否会重新运行 NLI。
19. 本 sprint 是否会重新调用 Ollama。
20. 本 sprint 是否会修改 scoring rule。
21. 本 sprint 是否会修改 schema。
22. 本 sprint 是否会覆盖 data/processed/*。
23. 本 sprint 是否会覆盖 1N / 1O outputs。
24. 当前裸 python 路径与版本。
25. 当前 recover_attention 环境 python 路径与版本。
26. 本 sprint 实际运行命令是否全部使用 conda run -n recover_attention python。
27. 本次输出目录。
28. 是否发现冲突。
```

第 18 项必须回答：

```text id="jt3amd"
否。
```

第 19 项必须回答：

```text id="xi7cmf"
否。
```

第 20 项必须回答：

```text id="r4kzmp"
否。
```

第 21 项必须回答：

```text id="ynhwrc"
否。
```

第 22 项必须回答：

```text id="37d7w6"
否。
```

第 23 项必须回答：

```text id="aj8mal"
否。
```

第 26 项必须回答：

```text id="9ya9lo"
是，全部使用 conda run -n recover_attention python。
```

---

## 12. 新增 orchestration script

新增：

```text id="qid9ch"
scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
```

该脚本只做 orchestration 和 comparison report，不新增算法。

它应复用现有模块：

```text id="mdqoo5"
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
```

也可以通过现有 CLI 等价逻辑执行，但推荐直接调用 module function，便于测试。

---

## 13. CLI 参数

`scripts/14_rebuild_downstream_upgraded_recovery_scoring.py` 应支持：

```text id="yroymg"
--semantic-labels
--upgraded-recover-scores
--baseline-recover-scores
--baseline-unit-evidence
--baseline-attention-anchor-labels
--baseline-intervention-manifest
--baseline-report
--output-dir
--unit-evidence-backend
--attention-label-backend
--intervention-type
--intervention-backend
--mask-token
--limit
```

默认值：

```text id="yexbdb"
--semantic-labels outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
--upgraded-recover-scores outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
--baseline-recover-scores outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
--baseline-unit-evidence outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
--baseline-attention-anchor-labels outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
--baseline-intervention-manifest outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
--baseline-report outputs/logs/sprint_1N_real_downstream/real_signal_report.json
--output-dir outputs/logs/sprint_1P_upgraded_downstream
--unit-evidence-backend aggregate_stub_v0
--attention-label-backend early_evidence_rule_stub_v0
--intervention-type mask
--intervention-backend manifest_stub_v0
--mask-token [MASK]
--limit None
```

---

## 14. limit 语义

`--limit` 只用于小样本测试。

要求：

```text id="mrqisa"
1. 如果传入 --limit N，只保留前 N 个 unit_id / masked_id 相关 records。
2. semantic_labels、recover_scores、baseline files 必须按相同 ID 子集过滤。
3. 不改变 library 层默认行为。
4. full run 不应使用 --limit。
```

如果实现 limit 过滤复杂，可以只在测试里覆盖辅助函数，不在正式 full run 使用。

---

## 15. Orchestration pipeline

脚本必须按以下顺序执行。

### Step 1：读取输入

读取：

```text id="y6u1r7"
semantic_labels_real.jsonl
recover_scores_nli_judge.jsonl
recover_scores_real.jsonl
unit_evidence_real.jsonl
attention_anchor_labels_real.jsonl
intervention_manifest_real.jsonl
real_signal_report.json
```

验证：

```text id="c0wmgc"
1. upgraded recover scores 与 baseline recover scores 的 masked_id 集合一致。
2. upgraded recover scores 与 semantic labels 的 unit_id 覆盖关系可用于 unit_evidence build。
3. 输入文件非空。
```

### Step 2：重建 upgraded unit evidence

输入：

```text id="75pp7f"
semantic_labels_real.jsonl
recover_scores_nli_judge.jsonl
```

输出：

```text id="8ilw8b"
unit_evidence_upgraded.jsonl
```

backend：

```text id="n2dplp"
aggregate_stub_v0
```

### Step 3：重建 upgraded attention anchor labels

输入：

```text id="mkpz66"
unit_evidence_upgraded.jsonl
```

输出：

```text id="qjs8ix"
attention_anchor_labels_upgraded.jsonl
```

backend：

```text id="e8xoux"
early_evidence_rule_stub_v0
```

### Step 4：重建 upgraded intervention manifest

输入：

```text id="xqszyr"
attention_anchor_labels_upgraded.jsonl
```

输出：

```text id="n3k35f"
intervention_manifest_upgraded.jsonl
```

backend：

```text id="8wn0ut"
manifest_stub_v0
```

intervention type：

```text id="fgg0ux"
mask
```

### Step 5：生成 comparison report

输出：

```text id="dqld67"
upgraded_downstream_report.json
upgraded_downstream_report.md
```

---

## 16. Report 要求

生成：

```text id="koh3i2"
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.json
```

报告必须至少包含：

```text id="rn51dj"
run_metadata
input_counts
output_counts
recover_score_comparison
unit_evidence_comparison
attention_anchor_label_comparison
intervention_manifest_comparison
changed_cases
sample_records
known_limitations
next_step_recommendation
```

### 16.1 run_metadata

包含：

```text id="jl7h8i"
timestamp
semantic_labels_path
baseline_recover_scores_path
upgraded_recover_scores_path
baseline_unit_evidence_path
baseline_attention_anchor_labels_path
baseline_intervention_manifest_path
output_dir
unit_evidence_backend
attention_label_backend
intervention_backend
intervention_type
limit
```

### 16.2 input_counts

包含：

```text id="hc5chl"
num_semantic_labels
num_baseline_recover_scores
num_upgraded_recover_scores
num_baseline_unit_evidence
num_baseline_attention_anchor_labels
num_baseline_intervention_manifest
```

### 16.3 output_counts

包含：

```text id="t0twcj"
num_unit_evidence_upgraded
num_attention_anchor_labels_upgraded
num_intervention_manifest_upgraded
```

### 16.4 recover_score_comparison

至少包含：

```text id="o9dtm3"
baseline_score_backend_counts
upgraded_score_backend_counts
baseline_recoverability_label_counts
upgraded_recoverability_label_counts
recoverability_label_changed_count
recoverability_label_transition_counts
baseline_recoverability_score_summary
upgraded_recoverability_score_summary
recoverability_score_delta_summary
misleading_recovery_changed_count
```

### 16.5 unit_evidence_comparison

至少包含：

```text id="bqbu2c"
baseline_evidence_backend_counts
upgraded_evidence_backend_counts
unit_evidence_record_count_changed
semantic_evidence_count_consistency
recoverability_evidence_label_transition_counts
```

### 16.6 attention_anchor_label_comparison

至少包含：

```text id="9f7yua"
baseline_attention_anchor_label_counts
upgraded_attention_anchor_label_counts
attention_anchor_label_changed_count
attention_anchor_label_transition_counts
baseline_attention_importance_score_summary
upgraded_attention_importance_score_summary
attention_importance_score_delta_summary
```

### 16.7 intervention_manifest_comparison

至少包含：

```text id="6j95lf"
baseline_manifest_count
upgraded_manifest_count
manifest_count_changed
baseline_intervention_type_counts
upgraded_intervention_type_counts
baseline_attention_anchor_label_counts_in_manifest
upgraded_attention_anchor_label_counts_in_manifest
```

### 16.8 changed_cases

保存最多 20 条变化最大的 case。

每条包含：

```text id="v2y13n"
id
unit_id
masked_id
original_question
masked_question
baseline_recoverability_label
upgraded_recoverability_label
baseline_recoverability_score
upgraded_recoverability_score
recoverability_score_delta
baseline_attention_anchor_label
upgraded_attention_anchor_label
baseline_attention_importance_score
upgraded_attention_importance_score
attention_importance_score_delta
recovered_questions
```

变化排序建议：

```text id="x8fun7"
1. attention_anchor_label changed 优先。
2. recoverability_label changed 次之。
3. abs(attention_importance_score_delta) 大者优先。
4. abs(recoverability_score_delta) 大者优先。
```

### 16.9 sample_records

保存最多 10 条 upgraded records summary：

```text id="jxpkfu"
id
unit_id
masked_id
semantic_necessity_label
upgraded_recoverability_label
upgraded_recoverability_score
upgraded_attention_anchor_label
upgraded_attention_importance_score
```

### 16.10 known_limitations

必须包含：

```text id="f67byv"
1. unit_evidence_upgraded 仍使用 aggregate_stub_v0。
2. attention_anchor_labels_upgraded 仍使用 early_evidence_rule_stub_v0。
3. intervention_manifest_upgraded 仍是 planned_only。
4. nli_recovery_judge_v0 是 question-level NLI judge，不直接验证每个 masked span。
5. 本 sprint 未重新运行 NLI，也未重新调用 Ollama。
6. 本 sprint 未接入 hidden states / attention maps / trajectory stability / attention guidance。
7. 结果只能说明 upgraded recovery scoring 对 downstream labels 的影响，不证明 attention guidance 有效。
```

### 16.11 next_step_recommendation

默认：

```text id="jxtpg9"
Sprint 1Q：Real Signal Quality Review
```

---

## 17. Markdown report 要求

生成：

```text id="qzg1yg"
outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.md
```

内容至少包含：

```text id="avl5wk"
1. Sprint 1P summary
2. Input files
3. Output files
4. Recoverability label distribution comparison
5. Attention anchor label distribution comparison
6. Top changed cases
7. Known limitations
8. Next step
```

不要包含完整 JSONL 内容。

---

## 18. 测试要求

新增：

```text id="vqf9ej"
tests/test_rebuild_downstream_upgraded_recovery_scoring.py
```

测试不能调用 NLI 模型，不能调用 Ollama。

必须用 tmp_path + fake records / monkeypatch。

至少覆盖：

```text id="ifue91"
1. output_dir 下生成预期文件名。
2. report.json 包含必需 top-level keys。
3. upgraded recover scores 与 baseline recover scores 的 masked_id 集合不一致时 raise。
4. recoverability_label_transition_counts 统计正确。
5. attention_anchor_label_transition_counts 统计正确。
6. attention_importance_score_delta_summary 统计正确。
7. changed_cases 最多 20 条。
8. sample_records 最多 10 条。
9. script 不写入 data/processed。
10. script 不覆盖 1N / 1O 输入目录。
11. --limit 能过滤一致 ID 子集。
12. known_limitations 包含 planned_only 和 no attention guidance。
13. next_step_recommendation 是 Sprint 1Q。
14. generated upgraded records 能通过对应 schema validator。
```

如果 orchestration 脚本过大，建议拆出以下纯函数便于测试：

```python id="wckfuz"
build_upgraded_downstream_paths(output_dir: Path) -> dict[str, Path]
summarize_numeric(values: list[float]) -> dict
transition_counts(old_records, new_records, key_field, value_field) -> dict
build_upgraded_downstream_report(...)
select_changed_cases(...)
```

---

## 19. 必须运行命令

所有命令必须使用：

```text id="vkjgsm"
conda run -n recover_attention python ...
```

不要使用裸 `python`。

### 19.1 基础测试

```powershell id="hq5vkh"
conda run -n recover_attention python scripts/sync_interface_fields.py --check

conda run -n recover_attention python -m pytest tests/test_rebuild_downstream_upgraded_recovery_scoring.py -q

conda run -n recover_attention python -m pytest tests/test_schemas.py -q

conda run -n recover_attention python -m pytest -q
```

### 19.2 小样本 dry run

```powershell id="rp360z"
conda run -n recover_attention python scripts/14_rebuild_downstream_upgraded_recovery_scoring.py `
  --semantic-labels outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl `
  --upgraded-recover-scores outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl `
  --baseline-recover-scores outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl `
  --baseline-unit-evidence outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl `
  --baseline-attention-anchor-labels outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl `
  --baseline-intervention-manifest outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl `
  --baseline-report outputs/logs/sprint_1N_real_downstream/real_signal_report.json `
  --output-dir outputs/logs/sprint_1P_upgraded_downstream `
  --unit-evidence-backend aggregate_stub_v0 `
  --attention-label-backend early_evidence_rule_stub_v0 `
  --intervention-type mask `
  --intervention-backend manifest_stub_v0 `
  --mask-token "[MASK]" `
  --limit 10
```

### 19.3 全量 run

如果 dry run 通过，运行不带 limit 的全量版本：

```powershell id="vpm5st"
conda run -n recover_attention python scripts/14_rebuild_downstream_upgraded_recovery_scoring.py `
  --semantic-labels outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl `
  --upgraded-recover-scores outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl `
  --baseline-recover-scores outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl `
  --baseline-unit-evidence outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl `
  --baseline-attention-anchor-labels outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl `
  --baseline-intervention-manifest outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl `
  --baseline-report outputs/logs/sprint_1N_real_downstream/real_signal_report.json `
  --output-dir outputs/logs/sprint_1P_upgraded_downstream `
  --unit-evidence-backend aggregate_stub_v0 `
  --attention-label-backend early_evidence_rule_stub_v0 `
  --intervention-type mask `
  --intervention-backend manifest_stub_v0 `
  --mask-token "[MASK]"
```

如果全量 run 失败，最终回复必须说明：

```text id="qjkkkj"
1. 失败阶段。
2. 错误信息。
3. 已成功生成到哪个阶段。
4. 是否保留 dry run 产物。
5. 是否需要下一个 sprint 修复。
```

---

## 20. configs 更新要求

允许更新：

```text id="o3fqcb"
configs/v0_nli_small.yaml
```

追加：

```yaml id="k1u8hu"
upgraded_downstream:
  output_dir: "outputs/logs/sprint_1P_upgraded_downstream"
  semantic_labels: "outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl"
  upgraded_recover_scores: "outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl"
  baseline_recover_scores: "outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl"
  baseline_unit_evidence: "outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl"
  baseline_attention_anchor_labels: "outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl"
  baseline_intervention_manifest: "outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl"
  baseline_report: "outputs/logs/sprint_1N_real_downstream/real_signal_report.json"
  unit_evidence:
    backend: "aggregate_stub_v0"
  attention_anchor_labels:
    backend: "early_evidence_rule_stub_v0"
  intervention_manifest:
    intervention_type: "mask"
    backend: "manifest_stub_v0"
```

不要让 `scripts/00_smoke_test.py` 自动运行 upgraded downstream。

---

## 21. PROGRESS.md 更新要求

更新：

```text id="xxy122"
PROGRESS.md
```

要求：

```text id="wstxzx"
1. 当前阶段更新为 Sprint 1P 已完成：Rebuild Downstream with Upgraded Recovery Scoring。
2. 已完成 Sprint 摘要中新增 Sprint 1P。
3. 当前可运行命令中新增 scripts/14_rebuild_downstream_upgraded_recovery_scoring.py 命令。
4. 最近一次检查结果中记录：
   upgraded downstream rebuild orchestration: passed
   upgraded unit evidence build: passed
   upgraded attention anchor labels build: passed
   upgraded intervention manifest build: passed
   upgraded downstream comparison report: passed
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中新增：
   scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
   tests/test_rebuild_downstream_upgraded_recovery_scoring.py
   outputs/logs/sprint_1P_upgraded_downstream/*
6. 遗留问题中说明：
   - unit_evidence_upgraded 仍使用 aggregate_stub_v0。
   - attention_anchor_labels_upgraded 仍使用 early_evidence_rule_stub_v0。
   - intervention_manifest_upgraded 仍是 planned_only。
   - 当前仍未接入 hidden states / attention maps / trajectory stability / attention guidance。
   - 本 sprint 只比较 upgraded scoring 对 downstream 的影响，不证明 attention guidance 有效。
7. 下一步建议：
   Sprint 1Q：Real Signal Quality Review
```

如果全量 run 未成功，只完成了 limit 10，则不要写 full run passed，应写：

```text id="j1r2bb"
upgraded downstream small run: passed
full run: failed / skipped，原因是 ...
```

---

## 22. docs/progress/sprint_1_history.md 更新要求

更新：

```text id="z4rw4l"
docs/progress/sprint_1_history.md
```

追加：

```text id="vum3sm"
## Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring
```

内容包括：

```text id="q2q4su"
1. 已完成内容。
2. 新增/修改文件。
3. 输出目录。
4. 输入文件。
5. baseline vs upgraded scoring 对比方式。
6. dry run 是否完成。
7. full run 是否完成。
8. upgraded_downstream_report.json 关键统计摘要。
9. 检查结果。
10. 遗留问题。
11. 下一步建议：Sprint 1Q：Real Signal Quality Review。
```

---

## 23. 验收标准

本 sprint 完成后必须满足：

```text id="jqypjo"
1. 没有重新运行 NLI scoring。
2. 没有重新调用 Ollama。
3. 没有修改 scoring rule。
4. 没有修改 schema。
5. 没有修改 existing backend implementation。
6. 没有覆盖 data/processed/*.jsonl。
7. 没有覆盖 outputs/logs/sprint_1N_real_downstream/*。
8. 没有覆盖 outputs/logs/sprint_1O_recovery_scoring/*。
9. outputs/logs/sprint_1P_upgraded_downstream/ 已生成。
10. unit_evidence_upgraded.jsonl 已生成并通过 unit_evidence schema。
11. attention_anchor_labels_upgraded.jsonl 已生成并通过 attention_anchor_label schema。
12. intervention_manifest_upgraded.jsonl 已生成并通过 intervention_manifest schema。
13. upgraded_downstream_report.json 已生成。
14. upgraded_downstream_report.md 已生成。
15. report 包含 recover_score_comparison。
16. report 包含 attention_anchor_label_comparison。
17. report 包含 changed_cases。
18. report 包含 known limitations。
19. tests/test_rebuild_downstream_upgraded_recovery_scoring.py 通过。
20. tests/test_schemas.py 通过。
21. 全量 pytest 通过。
22. sync_interface_fields.py --check 通过。
23. PROGRESS.md 已更新。
24. docs/progress/sprint_1_history.md 已更新。
```

---

## 24. 完成后回复格式

完成后请按以下格式回复：

```text id="bgqj4z"
1. 本次完成内容
2. 新增/修改文件
3. 输入文件
4. 输出目录
5. baseline vs upgraded scoring 对比摘要
6. dry run / full run 状态
7. upgraded_downstream_report.json 关键统计
8. 检查结果
9. PROGRESS.md 更新摘要
10. docs/progress/sprint_1_history.md 更新摘要
11. 遗留问题
12. 下一步建议
```

下一步建议必须是：

```text id="m7ni5k"
Sprint 1Q：Real Signal Quality Review
```

不要自动开始 Sprint 1Q。
