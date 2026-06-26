# Sprint 1 历史记录

## Sprint 1A：Candidate Span Extraction Framework

已完成内容：

- 实现 rule_based candidate span extraction framework。
- 实现 `extract_candidate_spans`，支持 `language` / `backend` / `max_candidates` 参数。
- 实现 `build_candidate_span_records`，输出 `candidate_spans.jsonl` record 并调用 schema validation。
- 实现 `scripts/02_extract_candidate_spans.py` CLI。
- 生成 `data/processed/candidate_spans.jsonl`。
- 新增 candidate extraction pytest，覆盖英文、中文、offset、cyber security term、max_candidates、少量候选允许通过、span_id 连续编号和 schema validation。

新增或修改文件：

- src/recover_attention/candidate_extraction.py
- scripts/02_extract_candidate_spans.py
- tests/test_candidate_extraction.py
- data/processed/candidate_spans.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/questions.jsonl

输出文件：

- data/processed/candidate_spans.jsonl

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe -c "import sys; print(sys.executable); print(sys.version)"
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/02_extract_candidate_spans.py --input data/processed/questions.jsonl --output data/processed/candidate_spans.jsonl
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- prepare_data 已通过，生成 5 条标准 question records。
- candidate extraction 已通过，生成 `data/processed/candidate_spans.jsonl`。
- `python -m pytest -q` 已通过，结果为 `49 passed`。

candidate 数量统计：

```text
num_questions: 5
total_candidates: 36
avg_candidates_per_question: 7.20
min_candidates_per_question: 7
max_candidates_per_question: 8
questions_with_zero_candidates: 0
span_type_counts: {'comparison': 3, 'number': 10, 'object': 12, 'operation': 6, 'question_target': 5}
max_candidates_setting: 20
language: auto
backend: rule_based
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮验收改用 `D:\conda\Miniconda3\envs\recover_attention\python.exe`。
- rule_based extractor 是最小 baseline，不追求最终抽取质量。
- 本轮未实现 ablation、NLI、masked question construction、question recovery、recoverability scoring、baseline CoT、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1B：Ablation Unit Construction

## Sprint 1B：Ablation Unit Construction

已完成内容：

- 实现 ablation unit construction 最小可运行版本。
- 新增 `validate_ablation_unit_record`，校验 unit 字段、scope / group_type 约束、span_ids 顺序和 offset。
- 实现 `build_ablation_units` 和 `build_ablation_unit_records`。
- 实现 `scripts/03_build_ablation_units.py` CLI。
- 生成 `data/processed/ablation_units.jsonl`。
- 新增 ablation unit pytest，覆盖 single units、group units、budget、连续编号、schema validation、无 candidates 和 CLI smoke test。

新增或修改文件：

- src/recover_attention/ablation_units.py
- scripts/03_build_ablation_units.py
- tests/test_ablation_units.py
- src/recover_attention/schemas.py
- tests/test_schemas.py
- data/processed/ablation_units.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/candidate_spans.jsonl

输出文件：

- data/processed/ablation_units.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import sys; print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/03_build_ablation_units.py --input data/processed/candidate_spans.jsonl --output data/processed/ablation_units.jsonl
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- ablation unit construction 已通过，生成 `data/processed/ablation_units.jsonl`。
- `python -m pytest -q` 已通过，结果为 `65 passed`。

ablation unit 数量统计：

```text
num_questions: 5
num_candidate_spans: 36
num_units: 46
num_single_units: 36
num_group_units: 10
group_type_counts: {'number_set': 5, 'repeated_surface': 5}
questions_with_zero_units: 0
max_group_size: 8
max_group_units: 10
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- 本轮只生成 ablation units，没有生成 `ablated_questions.jsonl`。
- 本轮未实现 ablated question construction、NLI、masked question construction、recovery、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1C：Ablated Question Construction。

## Sprint 1C：Ablated Question Construction

已完成内容：

- 实现 ablated question construction 最小可运行版本。
- 将 `validate_ablated_question_record` 对齐到 unit-level `ablated_questions.jsonl` 接口。
- 实现 `apply_ablation_to_unit`，支持 delete / generalize。
- 实现 `build_ablated_question_records`，返回 records 和 stats。
- 实现 `scripts/04_build_ablated_questions.py` CLI。
- 生成 `data/processed/ablated_questions.jsonl`。
- 新增 question ablation pytest，覆盖 single/group delete、target occurrence、English/Chinese generalize、overlap skip、empty/unchanged skip、schema validation 和 CLI smoke test。

新增或修改文件：

- src/recover_attention/question_ablations.py
- scripts/04_build_ablated_questions.py
- tests/test_question_ablations.py
- src/recover_attention/schemas.py
- tests/test_schemas.py
- data/processed/ablated_questions.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/ablation_units.jsonl

输出文件：

- data/processed/ablated_questions.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import sys; print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/04_build_ablated_questions.py --input data/processed/ablation_units.jsonl --output data/processed/ablated_questions.jsonl
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- ablated question construction 已通过，生成 `data/processed/ablated_questions.jsonl`。
- `python -m pytest -q` 已通过，结果为 `93 passed`。

ablated question 数量统计：

```text
num_input_questions: 5
num_input_units: 46
num_output_ablations: 92
num_skipped_empty: 0
num_skipped_unchanged: 0
num_skipped_overlap: 0
ablation_type_counts: {'delete': 46, 'generalize': 46}
unit_scope_counts: {'group': 20, 'single': 72}
group_type_counts: {'number_set': 10, 'repeated_surface': 10, 'single': 72}
language: auto
ablation_types: ['delete', 'generalize']
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- 本轮只生成 ablated questions，没有生成 `nli_scores.jsonl`。
- 本轮未实现 NLI、masked question construction、recovery、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1D：NLI Semantic Necessity Stub。

## Sprint 1D：NLI Semantic Consistency Scoring Stub

已完成内容：

- 实现 deterministic `stub_v0` 双向 NLI scoring。
- 实现 `detect_language`、`resolve_record_language`、`score_nli_pair_stub`、`score_ablated_question_record` 和 `score_ablated_question_records`。
- 将 `validate_nli_score_record` 对齐到 1D 的 `nli_scores.jsonl` score-only 接口。
- 实现 `scripts/05_run_nli_scoring.py` CLI。
- 生成 `data/processed/nli_scores.jsonl`。
- 新增 NLI scoring pytest，覆盖 generalize/delete、group penalty、bidirectional score、contradiction score、language、unsupported backend/language、schema validation 和 CLI smoke test。

新增或修改文件：

- docs/skill/SKILL.md
- src/recover_attention/nli_scoring.py
- scripts/05_run_nli_scoring.py
- src/recover_attention/schemas.py
- tests/test_nli_scoring.py
- tests/test_schemas.py
- data/processed/nli_scores.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/ablated_questions.jsonl

输出文件：

- data/processed/nli_scores.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import os, sys; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores.jsonl --backend stub_v0 --language auto
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- NLI scoring stub 已通过，生成 `data/processed/nli_scores.jsonl`。
- `python -m pytest -q` 已通过，结果为 `111 passed`。

NLI score 数量统计：

```text
num_input_ablations: 92
num_output_scores: 92
backend: stub_v0
language_setting: auto
language_counts: {'en': 92}
ablation_type_counts: {'delete': 46, 'generalize': 46}
unit_scope_counts: {'group': 20, 'single': 72}
group_type_counts: {'number_set': 10, 'repeated_surface': 10, 'single': 72}
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- `docs/skill/label_schema.md` 中仍保留旧版 NLI score 示例，包含 `semantic_necessity_label`；本轮按 Sprint 1D task card 和 `docs/skill/nli_scores_interface.md` 执行 score-only 接口。
- 本轮只生成 NLI scores，没有生成 semantic necessity labels。
- 本轮未实现 masked question construction、recovery、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1E：Semantic Necessity Label Rule。

## Sprint 1E：Semantic Necessity Label Rule

已完成内容：

- 实现 rule_v0 semantic necessity label rule。
- 新增 `validate_semantic_label_record`，校验 1D 复制字段、forward/backward 方向、rule_v0、label enum、rule parameters、boolean consistency 和 semantic_necessity_score。
- 新增 `src/recover_attention/semantic_labels.py`，提供单条和批量 semantic label 构造函数。
- 新增 `scripts/06_build_semantic_labels.py` CLI。
- 新增 `docs/skill/semantic_labels_interface.md` 并在 `docs/skill/SKILL.md` 中索引。
- 更新 `docs/skill/label_schema.md` 的 Semantic Label Record 摘要。
- 生成 `data/processed/semantic_labels.jsonl`。
- 新增 semantic label pytest，并补充 schema validator pytest。

新增或修改文件：

- docs/skill/semantic_labels_interface.md
- docs/skill/SKILL.md
- docs/skill/label_schema.md
- src/recover_attention/schemas.py
- src/recover_attention/semantic_labels.py
- scripts/06_build_semantic_labels.py
- tests/test_semantic_labels.py
- tests/test_schemas.py
- data/processed/semantic_labels.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/nli_scores.jsonl

输出文件：

- data/processed/semantic_labels.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import sys; print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- semantic label rule 已通过，生成 `data/processed/semantic_labels.jsonl`。
- `python -m pytest -q` 已通过，结果为 `138 passed`。

semantic label 数量统计：

```text
num_input_scores: 92
num_output_labels: 92
backend: rule_v0
rule_parameters: {'equivalent_threshold': 0.7, 'directional_entailment_threshold': 0.5, 'contradiction_threshold': 0.5}
semantic_necessity_label_counts: {'Information Loss': 90, 'Non-equivalent': 2}
is_semantically_necessary_counts: {True: 92}
ablation_type_counts: {'delete': 46, 'generalize': 46}
unit_scope_counts: {'group': 20, 'single': 72}
group_type_counts: {'number_set': 10, 'repeated_surface': 10, 'single': 72}
language_counts: {'en': 92}
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- Sprint 1D 历史记录中提到的 `label_schema.md` 旧 NLI label-in-score 遗留问题已在接口对齐和本轮 1E 中修正；当前 `label_schema.md` 已明确 `nli_scores.jsonl` 为 score-only。
- 本轮未生成 masked_questions、recover_outputs、recover_scores、labels、token_labels 或 attention_anchor_labels。
- 本轮未实现 recovery、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1F：Masked Question Construction for Recoverability。

## Sprint 1F：Masked Question Construction for Recoverability

已完成内容：

- 实现 unit-level masked question construction 最小可运行版本。
- 新增 `src/recover_attention/masked_questions.py`，按 `(id, unit_id)` 聚合 semantic label records。
- 实现 `replace_each_span` mask 构造：unit 内每个 span 各替换为一个 `mask_token`。
- 聚合 delete / generalize semantic sources，并生成同序的 `source_semantic_label_ids`、`source_nli_ids`、`source_ablation_ids`。
- 支持 `--only-necessary` 过滤、overlap unit 跳过统计、自定义 `--mask-token` 和 backend 校验。
- 新增 `scripts/07_build_masked_questions.py` CLI。
- 新增 masked question pytest，覆盖 single/group/repeated_surface、source 聚合、only-necessary、overlap skip、custom mask token、禁止旧字段和 CLI smoke test。
- 1F 收尾修正：`scripts/sync_interface_fields.py` 支持显式 `--check`；batch 构造在过滤和 overlap skip 前先校验同 unit 结构一致性。
- 生成 `data/processed/masked_questions.jsonl`。

新增或修改文件：

- src/recover_attention/masked_questions.py
- scripts/07_build_masked_questions.py
- scripts/sync_interface_fields.py
- tests/test_masked_questions.py
- data/processed/masked_questions.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/semantic_labels.jsonl

输出文件：

- data/processed/masked_questions.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import sys; print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
conda run -n recover_attention python -m pytest tests/test_masked_questions.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- interface required_fields 同步检查已通过。
- masked question construction 已通过，生成 `data/processed/masked_questions.jsonl`。
- `python -m pytest tests/test_masked_questions.py -q` 已通过，结果为 `14 passed`。
- `python -m pytest -q` 已通过，结果为 `182 passed, 2 skipped`。

masked question 数量统计：

```text
num_input_labels: 92
num_units: 46
num_output_masks: 46
num_filtered_not_necessary: 0
num_skipped_overlap: 0
mask_token: [MASK]
mask_backend: unit_mask_v0
mask_strategy: replace_each_span
only_necessary: False
unit_scope_counts: {'group': 10, 'single': 36}
group_type_counts: {'number_set': 5, 'repeated_surface': 5, 'single': 36}
source_count_distribution: {2: 46}
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- `data/processed/*` 是本地生成产物目录，当前被 `.gitignore` 忽略；GitHub 默认不会显示 `masked_questions.jsonl`。
- 本轮未生成 recover_outputs、recover_scores、labels、token_labels 或 attention_anchor_labels。
- 本轮未实现 recovery、recoverability scoring、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1G：Question Recovery（unit-level recover outputs）。

## Sprint 1G 前置接口修正：Recover Output Interface Alignment

已完成内容：

- 将 `recover_outputs.jsonl` 接口从旧 span-level schema 对齐为 unit-level / masked_id-driven schema。
- 新增 `docs/skill/recover_outputs_interface.md`。
- 将 `REQUIRED_FIELDS["recover_output"]` 更新为 unit-level / masked_id-driven 字段集合；后续补丁已进一步扩展为 self-contained 15 字段接口。
- 更新 `validate_recover_output_record`，校验 `masked_id`、unit span metadata、`sample_id >= 0`，并拒绝旧字段 `span_id / span_text / span_type / recoverable / confidence / reason / recoverability_label`。
- 更新 `docs/skill/SKILL.md` 路由和 `docs/skill/label_schema.md` 第 11 节，使 recover output 指向 interface 文档。
- 更新接口一致性测试和 schema 测试。

新增或修改文件：

- docs/skill/recover_outputs_interface.md
- docs/skill/SKILL.md
- docs/skill/label_schema.md
- src/recover_attention/schemas.py
- tests/test_schemas.py
- tests/test_interface_consistency.py
- scripts/sync_interface_fields.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口修正。

输出文件：

- 无 `data/processed/recover_outputs.jsonl` 产物。本轮不运行 recovery。

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_schemas.py tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python -m pytest -q
```

检查结果：

- interface required_fields 同步检查已通过，包含 `recover_outputs_interface.md`。
- `python -m pytest tests/test_schemas.py tests/test_interface_consistency.py -q` 已通过，结果为 `82 passed, 2 skipped`。
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`
- `python -m pytest -q` 已通过，结果为 `189 passed, 2 skipped`。

发现并处理的不一致：

- `src/recover_attention/schemas.py` 中 `recover_output` 仍使用旧 `span_id / recoverable / confidence / reason` 字段，已修正。
- `docs/skill/label_schema.md` 第 11 节仍是旧 span-level recover output 字段表，已改为指向 `recover_outputs_interface.md`。
- `docs/skill/SKILL.md` 未索引 recover output interface，已补充。
- `tests/test_schemas.py` 仍使用旧 span-level recover output fixture，已改为 unit-level fixture。

保留并报告的不一致：

- 历史 task card（例如 `docs/codex_tasks/sprint_0C_schemas.md`）仍包含旧 span-level recover output 示例；本轮不回改历史任务卡。
- `recover_scores.jsonl` 仍是旧 span-level schema；应在 recoverability scoring 前单独做 unit-level 接口修正。

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- 本轮未生成 `data/processed/recover_outputs.jsonl`。
- 本轮未实现 recovery backend、recoverability scoring、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1G：Question Recovery（unit-level recover outputs implementation）。

## Sprint 1G 前置接口修正补丁：Self-contained Recover Output Interface

已完成内容：

- 将 `recover_outputs.jsonl` 从轻量 masked question 绑定调整为 self-contained unit-level record。
- `REQUIRED_FIELDS["recover_output"]` 新增：
  `unit_scope / group_type / original_question / mask_token / mask_backend / mask_strategy`。
- 新增 `ALLOWED_RECOVERY_BACKENDS = {"oracle_stub_v0"}`，并将 `recovery_backend` 收紧为枚举。
- `validate_recover_output_record` 现在校验 unit metadata、mask backend、mask strategy、recovery backend，并检查 `masked_question` 正好为每个 span 新增一个 `mask_token`。
- 更新 `docs/skill/recover_outputs_interface.md` 和 `docs/skill/label_schema.md`，明确 recover output 保留后续 recoverability scoring 所需元数据，避免评分阶段必须 join 回 `masked_questions.jsonl`。
- 更新 schema 测试，新增非法 `recovery_backend` 回归测试。

新增或修改文件：

- docs/skill/recover_outputs_interface.md
- docs/skill/label_schema.md
- src/recover_attention/schemas.py
- tests/test_schemas.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口修正。

输出文件：

- 无 `data/processed/recover_outputs.jsonl` 产物。本轮不运行 recovery。

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_schemas.py tests/test_interface_consistency.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- interface required_fields 同步检查已通过，`recover_outputs_interface.md` 的 `recover_output` block 现在包含 15 个字段。
- `python -m pytest tests/test_schemas.py tests/test_interface_consistency.py -q` 已通过，结果为 `83 passed, 2 skipped`。
- `python -m pytest -q` 已通过，结果为 `190 passed, 2 skipped`。

遗留问题：

- 本轮未生成 `data/processed/recover_outputs.jsonl`。
- 本轮未实现 recovery backend、recoverability scoring、trajectory、attention guidance 或 probe。
- `recover_scores.jsonl` 仍是旧 span-level schema；应在 recoverability scoring 前单独做 unit-level 接口修正。

下一步建议：

- Sprint 1G：Question Recovery（基于 self-contained recover output interface 实现 `oracle_stub_v0`）。

## Sprint 1G：Question Recovery Stub

已完成内容：

- 实现 unit-level question recovery 的最小可运行 stub。
- 新增 `src/recover_attention/recover_generation.py`。
- 新增 `scripts/08_run_recovery.py` CLI。
- 新增 `tests/test_recover_generation.py`。
- 使用 `oracle_stub_v0` 从 `masked_questions.jsonl` 生成 `recover_outputs.jsonl`。
- `oracle_stub_v0` 的行为为 `recovered_question = original_question`，仅用于管线验证，不代表真实模型恢复能力。
- 每条输入先调用 `validate_masked_question_record`，每条输出写入前调用 `validate_recover_output_record`。
- 输出为 self-contained unit-level / masked_id-driven record，不包含旧顶层 span 字段，也不包含 recoverability scoring 字段。

新增或修改文件：

- src/recover_attention/recover_generation.py
- scripts/08_run_recovery.py
- tests/test_recover_generation.py
- data/processed/recover_outputs.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/masked_questions.jsonl

输出文件：

- data/processed/recover_outputs.jsonl

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
conda run -n recover_attention python -m pytest tests/test_recover_generation.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- `scripts/sync_interface_fields.py --check` 已通过。
- `tests/test_interface_consistency.py -q` 已通过，结果为 `20 passed, 2 skipped`。
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`。
- `tests/test_recover_generation.py -q` 已通过，结果为 `11 passed`。
- `python -m pytest -q` 已通过，结果为 `201 passed, 2 skipped`。

recover output 数量统计：

```text
num_input_masks: 46
num_output_recoveries: 46
num_samples: 1
recovery_backend: oracle_stub_v0
unit_scope_counts: {'group': 10, 'single': 36}
group_type_counts: {'number_set': 5, 'repeated_surface': 5, 'single': 36}
mask_backend_counts: {'unit_mask_v0': 46}
mask_strategy_counts: {'replace_each_span': 46}
```

输出内容检查：

```text
num_records: 46
all_fields_match: True
forbidden_present: []
all_oracle_recovered_original: True
sample_id_counts: {0: 46}
recovery_backend_counts: {'oracle_stub_v0': 46}
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- `data/processed/*` 是本地生成产物目录，当前被 `.gitignore` 忽略。
- `recover_scores.jsonl` 仍是旧 span-level schema；进入 recoverability scoring 前需要单独做 unit-level 接口修正。
- 本轮未生成 `recover_scores.jsonl`、`labels.jsonl`、`token_labels.jsonl` 或 `attention_anchor_labels.jsonl`。
- 本轮未实现真实模型 recovery、recoverability scoring、trajectory、attention guidance、hidden states 或 probe。

下一步建议：

- Sprint 1H-prep：Recover Score Interface Alignment。

## Sprint 1H-prep：Recover Score Interface Alignment

已完成内容：

- 将 `recover_score` 从旧 span-level schema 迁移为 unit-level / masked_id-driven schema。
- 新增 `docs/skill/recover_scores_interface.md`。
- 将 `recover_score` 加入 `schemas.INTERFACE_DOCS`，纳入 `scripts/sync_interface_fields.py` 与 `tests/test_interface_consistency.py` 的接口治理。
- 新增 `ALLOWED_RECOVER_SCORE_BACKENDS = {"stub_rule_v0"}`。
- 更新 `REQUIRED_FIELDS["recover_score"]`，使 recover score 保留 recover_outputs 聚合所需的 unit metadata、sample evidence 和 score 字段。
- 更新 `FORBIDDEN_FIELDS["recover_score"]`，显式拒绝旧 span-level 顶层字段、sample-level 字段和 attention/guidance 字段。
- 更新 `validate_recover_score_record`，校验 `recover_score_id`、`masked_id`、unit span metadata、sample 聚合字段、score 范围、backend enum 和 evidence。
- 更新 `docs/skill/label_schema.md` 第 12 节，使其只指向 `recover_scores_interface.md`，不再复制完整字段表。
- 更新 `docs/skill/SKILL.md` 路由和 `scripts/sync_interface_fields.py` 注释。
- 更新 `tests/test_schemas.py` 和 `tests/test_interface_consistency.py`。

新增或修改文件：

- src/recover_attention/schemas.py
- docs/skill/recover_scores_interface.md
- docs/skill/label_schema.md
- docs/skill/SKILL.md
- scripts/sync_interface_fields.py
- tests/test_schemas.py
- tests/test_interface_consistency.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口修正。

输出文件：

- 无 `data/processed/recover_scores.jsonl` 产物。本轮不运行 recoverability scoring。

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py tests/test_schemas.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- `scripts/sync_interface_fields.py --check` 已通过，`recover_scores_interface.md` 的 `recover_score` block 包含 24 个字段。
- `python -m pytest tests/test_interface_consistency.py tests/test_schemas.py -q` 已通过，结果为 `112 passed, 2 skipped`。
- `python -m pytest -q` 已通过，结果为 `230 passed, 2 skipped`。

接口摘要：

```text
recover_score is now unit-level / masked_id-driven.
recover_score consumes recover_outputs.jsonl grouped by masked_id.
old top-level span_id / span_text / span_type are rejected.
sample-level sample_id / recovered_question are rejected.
attention label and guidance fields are rejected.
```

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- 本轮未实现 `src/recover_attention/recover_scoring.py`。
- 本轮未新增 scoring CLI。
- 本轮未生成 `data/processed/recover_scores.jsonl`。
- 本轮未调用真实模型、未生成 hidden states / attention maps、未做 trajectory / probe / attention guidance。

下一步建议：

- Sprint 1H：Recoverability Scoring。

## Sprint 1I：Build Unit Evidence

已完成内容：

- 新增 `src/recover_attention/unit_evidence.py`，实现 `semantic_labels.jsonl + recover_scores.jsonl -> unit_evidence.jsonl` 的 unit-level evidence aggregation。
- 使用 `(id, unit_id)` 作为 join key，semantic labels 按 unit 分组，recover scores 按 unit 建唯一索引。
- 每条 semantic label 输入调用 `validate_semantic_label_record`，每条 recover score 输入调用 `validate_recover_score_record`，每条输出调用 `validate_unit_evidence_record`。
- 实现 fail-fast join：缺失 recover score、额外 recover score、重复 recover score、semantic group 内 metadata 不一致、semantic 与 recover score metadata 不一致都会报错。
- 聚合 `semantic_evidence`，包含 source ids、ablation types、semantic necessity labels/scores/votes、summary score、summary label、backend 和 language settings。
- 聚合 `recoverability_evidence`，包含 recover score id、masked id、backend、recoverability label/score、sample ids、source recovered questions 和 score evidence。
- 新增 `scripts/10_build_unit_evidence.py` CLI。
- 新增 `tests/test_unit_evidence.py`，覆盖正常构造、semantic evidence 聚合、recoverability evidence 聚合、join 错误、禁用字段和 CLI smoke test。
- 生成 `data/processed/unit_evidence.jsonl`。

新增或修改文件：

- src/recover_attention/unit_evidence.py
- scripts/10_build_unit_evidence.py
- tests/test_unit_evidence.py
- data/processed/unit_evidence.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/semantic_labels.jsonl
- data/processed/recover_scores.jsonl

输出文件：

- data/processed/unit_evidence.jsonl

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
conda run -n recover_attention python -m pytest tests/test_unit_evidence.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- `scripts/sync_interface_fields.py --check` 已通过，所有 interface required_fields block in sync。
- `tests/test_interface_consistency.py -q` 已通过，结果为 `29 passed, 2 skipped`。
- `scripts/10_build_unit_evidence.py` 已通过，生成 `data/processed/unit_evidence.jsonl`。
- `tests/test_unit_evidence.py -q` 已通过，结果为 `19 passed`。
- `python -m pytest -q` 已通过，结果为 `287 passed, 2 skipped`。

unit_evidence 数量统计：

```text
num_semantic_labels: 92
num_recover_scores: 46
num_output_unit_evidence: 46
unique_units: 46
evidence_status_counts: {'partial_stub_evidence': 46}
unit_scope_counts: {'group': 10, 'single': 36}
group_type_counts: {'number_set': 5, 'repeated_surface': 5, 'single': 36}
available_signal_types: ['semantic_necessity', 'semantic_recoverability']
missing_signal_types: ['trajectory_stability', 'answer_stability', 'raw_attention_pattern', 'attention_steering_effect']
forbidden_present: []
```

semantic summary label 分布：

```text
{'consistent_semantic_necessity_evidence': 46}
```

recoverability label 分布：

```text
{'Recoverable': 46}
```

遗留问题：

- `unit_evidence` 目前只汇总 semantic necessity 与 semantic recoverability early evidence。
- recoverability 来自 `oracle_stub_v0` + `stub_rule_v0`，只用于管线验证。
- trajectory stability、answer stability、raw attention pattern 和 attention steering effect 尚未接入。
- `unit_evidence` 不是 final attention anchor label。
- 尚未实现 attention_anchor_labels builder。
- 本轮未生成 `attention_anchor_labels.jsonl`。
- 本轮未调用真实模型、未缓存 hidden states / attention maps、未做 trajectory / answer stability / guidance 或 probe。

下一步建议：

- Sprint 1J-prep：Attention Anchor Label Interface Alignment。

## Sprint 1I-prep-a：Unit Evidence Interface Design

已完成内容：

- 新增 `unit_evidence` record type，用作 unit-level evidence aggregation 中间层。
- 新增 `docs/skill/unit_evidence_interface.md`，说明 `unit_evidence.jsonl` 的用途、pipeline 位置、字段来源、ID 规则、signal boundary、禁止内容、validator 和示例。
- 在 `src/recover_attention/schemas.py` 中新增 `REQUIRED_FIELDS["unit_evidence"]`、`FORBIDDEN_FIELDS["unit_evidence"]`、`INTERFACE_DOCS["unit_evidence"]`、`ALLOWED_UNIT_EVIDENCE_BACKENDS`、`ALLOWED_UNIT_EVIDENCE_STATUSES`、`ALLOWED_EVIDENCE_SIGNAL_TYPES` 和 `validate_unit_evidence_record`。
- 更新 `docs/skill/label_schema.md`，新增 Unit Evidence Record 小节并指向 `unit_evidence_interface.md`，不复制完整字段表。
- 更新 `docs/skill/SKILL.md` 文档路由，加入 `unit_evidence_interface.md`。
- 更新 `tests/test_interface_consistency.py`，使 unit evidence interface 纳入 marker / label_schema 一致性检查。
- 更新 `tests/test_schemas.py`，覆盖 valid record、缺字段、禁用字段、ID 规则、backend/status/signal enum、single/group span 约束和 span 顺序。

新增或修改文件：

- docs/skill/unit_evidence_interface.md
- docs/skill/label_schema.md
- docs/skill/SKILL.md
- src/recover_attention/schemas.py
- tests/test_interface_consistency.py
- tests/test_schemas.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口设计。

输出文件：

- 无 `data/processed` 产物。
- 未生成 `data/processed/unit_evidence.jsonl`。

运行命令：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python -m pytest tests/test_schemas.py -q
conda run -n recover_attention python -m pytest -q
```

检查结果：

- `scripts/sync_interface_fields.py --write` 已生成 `unit_evidence_interface.md` 的 `required_fields:unit_evidence` block，共 15 个字段。
- `scripts/sync_interface_fields.py --check` 已通过，所有 interface required_fields block in sync。
- `tests/test_interface_consistency.py -q` 已通过，结果为 `29 passed, 2 skipped`。
- `tests/test_schemas.py -q` 已通过，结果为 `106 passed`。
- `python -m pytest -q` 已通过，结果为 `268 passed, 2 skipped`。

遗留问题：

- `unit_evidence` 目前只有接口和 validator，尚未实现 builder。
- 本轮未新增 `scripts/10_build_unit_evidence.py`。
- 本轮未生成 `data/processed/unit_evidence.jsonl`。
- `unit_evidence` 当前只设计为汇总 semantic/recoverability early evidence。
- trajectory stability、answer stability、raw attention pattern 和 attention steering effect 仍未接入。
- `unit_evidence` 不是 final attention anchor label，不包含 `guidance_action` 或 `guidance_strength`。
- 本轮未调用真实模型、未缓存 hidden states / attention maps、未做 trajectory / answer stability / guidance 或 probe。

下一步建议：

- Sprint 1I：Build Unit Evidence。

## Sprint 1H：Recoverability Scoring

已完成内容：

- 实现 `stub_rule_v0` recoverability scoring 最小可运行版本。
- 新增 `src/recover_attention/recover_scoring.py`，按 `masked_id` 聚合 `recover_outputs.jsonl`。
- 实现 `normalize_question`：仅执行 strip 和连续空白折叠。
- 对每个 recover output 输入 record 调用 `validate_recover_output_record`。
- 每个 `masked_id` 分组生成一条 unit-level recover score record，并在写出前调用 `validate_recover_score_record`。
- 聚合输出 `source_sample_ids`、`recovered_questions`、`num_samples`、`recoverability_score`、`confidence_mean`、`recovery_consistency`、`misleading_recovery`、`recoverability_label` 和 `evidence`。
- 新增 `scripts/09_score_recovery.py` CLI。
- 新增 `tests/test_recover_scoring.py`，覆盖 exact、partial、empty、mismatch、sample 排序、重复 sample、metadata 不一致、unsupported backend、CLI smoke 和 missing input。
- 生成 `data/processed/recover_scores.jsonl`。

新增或修改文件：

- src/recover_attention/recover_scoring.py
- scripts/09_score_recovery.py
- tests/test_recover_scoring.py
- data/processed/recover_scores.jsonl
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/recover_outputs.jsonl

输出文件：

- data/processed/recover_scores.jsonl

运行命令：

```bash
conda run -n recover_attention python -c "import os, sys; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable); print(sys.version)"
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python -m pytest tests/test_recover_scoring.py -q
conda run -n recover_attention python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
conda run -n recover_attention python -m pytest -q
```

检查结果：

- Python 路径：`D:\conda\Miniconda3\envs\recover_attention\python.exe`
- Python 版本：`3.10.20`
- `scripts/sync_interface_fields.py --check` 已通过，所有 interface required_fields block in sync。
- `tests/test_interface_consistency.py -q` 已通过，结果为 `25 passed, 2 skipped`。
- `tests/test_recover_scoring.py -q` 已通过，结果为 `15 passed`。
- `scripts/09_score_recovery.py` 已通过，生成 `data/processed/recover_scores.jsonl`。
- `python -m pytest -q` 已通过，结果为 `246 passed, 2 skipped`。

recover score 数量统计：

```text
num_input_recoveries: 46
num_output_scores: 46
num_masked_ids: 46
score_backend_counts: {'stub_rule_v0': 46}
recovery_backend_counts: {'oracle_stub_v0': 46}
unit_scope_counts: {'group': 10, 'single': 36}
group_type_counts: {'number_set': 5, 'repeated_surface': 5, 'single': 36}
num_misleading_recovery: 0
forbidden_present: []
```

recoverability_label 分布：

```text
{'Recoverable': 46}
```

说明：

- 当前 `Recoverable: 46` 来自 `oracle_stub_v0` recovery output 与 `stub_rule_v0` exact normalized match scorer，只能说明管线闭环可运行。
- 该结果不能解释为真实 recoverability 性能。

遗留问题：

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；本轮使用 `conda run -n recover_attention python ...`。
- `stub_rule_v0` 只做 exact normalized match，不做 semantic similarity。
- 本轮未做真实模型 recovery judging。
- 本轮未调用 OpenAI API 或 Hugging Face 模型。
- 本轮未生成 hidden states / attention maps。
- 本轮未做 trajectory stability、answer stability、attention anchor labeling、attention guidance 或 probe。
- `data/processed/*` 是本地生成产物目录，当前被 `.gitignore` 忽略。

下一步建议：

- Sprint 1I-prep：Unit-to-Anchor Label Interface Design。

## Sprint 1H-prep-fix：Recover Score Governance Doc Cleanup

已完成内容：

- 修正 `docs/skill/label_schema.md` 第 0 节生成边界说明：从“不受 interface doc 管理”的例子中移除 `recover_score`（它已有 interface 文档并登记于 `schemas.INTERFACE_DOCS`），保留 `question / candidate_span / attention_anchor_label`。
- 保持第 0 节三条原则不变：REQUIRED_FIELDS / FORBIDDEN_FIELDS 是顶层字段唯一来源；interface 的 required_fields block 由 `scripts/sync_interface_fields.py` 生成；label_schema.md 是索引/总览，不复制完整字段表。
- 未改动第 12 节 Recover Score Record 的 unit-level / masked_id-driven 方向。
- 在 `tests/test_interface_consistency.py` 新增轻量回归测试 `test_label_schema_out_of_scope_examples_do_not_include_managed_interface_types`：断言 §0 “没有 interface 文档的 record（…）”枚举中，不出现任何已登记在 `schemas.INTERFACE_DOCS` 的 record type。

新增或修改文件：

- docs/skill/label_schema.md
- tests/test_interface_consistency.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做文档治理修补。

输出文件：

- 无 `data/processed` 产物。

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：6 个 interface block 全部 in sync（recover_score = 24 fields，未被手改）。
- tests/test_interface_consistency.py：25 passed, 2 skipped。
- 全量 pytest：231 passed, 2 skipped。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（等价于 `conda run -n recover_attention python ...`；本机 `conda` 不在 PATH）。
- 本轮未实现 recoverability scoring，未生成 `data/processed/recover_scores.jsonl`，未调用模型。

下一步建议：

- Sprint 1H：Recoverability Scoring。

## Sprint 1I-doc-fix：Unit Evidence Interface Post-build Cleanup

已完成内容：

- 修正 `docs/skill/unit_evidence_interface.md` 中 Sprint 1I-prep-a 阶段的过时表述（builder 已在 Sprint 1I 实现）：
  - §2 Pipeline Position 的 “This sprint only defines the interface. It does not implement aggregation.” 改为说明 build 阶段已聚合 `semantic_labels.jsonl` + `recover_scores.jsonl` 生成 `unit_evidence.jsonl`，由 `src/recover_attention/unit_evidence.py` 实现、`scripts/10_build_unit_evidence.py` 运行。
  - §4 Field Sources 的 “This sprint only designs the interface. It does not implement the aggregation stage.” 改为说明这些字段由 build 阶段产出且遵循本接口。
  - §9 Example 的 notes “Example only; builder implementation belongs to a later sprint.” 改为 “Example only; concrete records are produced by scripts/10_build_unit_evidence.py.”
- 未改动 schema、`REQUIRED_FIELDS["unit_evidence"]`、required_fields 生成块、字段结构。
- 保留边界表述不变：`unit_evidence` 不是 final attention anchor label；不含 guidance / trajectory stability / answer stability / raw attention。

修改文件：

- docs/skill/unit_evidence_interface.md
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做文档修补。

输出文件：

- 无 `data/processed` 产物。

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：全部 in sync（`unit_evidence` = 15 fields，未被手改）。
- tests/test_interface_consistency.py：29 passed, 2 skipped。
- 全量 pytest：287 passed, 2 skipped。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（本机 `conda` 不在 PATH）。
- 本轮未实现 attention anchor labeling，未生成 `attention_anchor_labels.jsonl`，未调用模型。

下一步建议：

- Sprint 1J-prep：Attention Anchor Label Interface Alignment。

## Sprint 1J-prep：Attention Anchor Label Interface Alignment

已完成内容：

- 将 `attention_anchor_label` 从旧 span-level schema 对齐为 unit-level（绑定 `id + unit_id`，span 信息保留在 `span_ids` / `spans`）。
- `schemas.py`：
  - 新增 enum `ALLOWED_ATTENTION_LABEL_BACKENDS = {"early_evidence_rule_stub_v0"}`、`ALLOWED_ATTENTION_LABEL_STATUSES = {"partial_evidence_label"}`（保留 `ALLOWED_ATTENTION_ANCHOR_LABELS` 不变）。
  - `REQUIRED_FIELDS["attention_anchor_label"]` 改为 18 字段 unit-level（含 `attention_anchor_label_id` / `unit_evidence_id` / unit metadata / semantic_evidence / recoverability_evidence / signal types / attention_importance_score / attention_anchor_label / label_backend / label_status / evidence）。
  - 新增 `FORBIDDEN_FIELDS["attention_anchor_label"]`，拒绝旧 span-level 顶层字段、sample-level recovery 字段、`guidance_action` / `guidance_strength`、hidden states / attention maps / trajectory / answer stability / raw attention / probe 字段。
  - `INTERFACE_DOCS` 新增 `"attention_anchor_label": "attention_anchor_labels_interface.md"`。
  - 重写 `validate_attention_anchor_label_record` 为 unit-level（复用 unit_evidence 的 span/unit/signal 校验；校验 `attention_anchor_label_id == f"{unit_evidence_id}__anchor_{label_backend}"`；enum/range/signal type 校验）。
- 新增 `docs/skill/attention_anchor_labels_interface.md`（含 `<!-- required_fields:attention_anchor_label -->` 生成块，由 `sync_interface_fields.py --write` 写入 18 字段；Purpose / Pipeline / Field Sources / ID Rule / Unit-level & Label Boundary / Not Included / Validator / Example）。
- `label_schema.md`：§17 降级为指向新 interface 的要点说明（声明旧 span-level 废弃、unit-level、不含 guidance）；§0 out-of-scope 移除 `attention_anchor_label`。
- `SKILL.md`：Document Router 新增 §3.8.9 attention_anchor_labels_interface.md。
- 测试：`test_interface_consistency.py` 的 `LABEL_SCHEMA_SECTIONS` 新增 attention_anchor_label（marker/forbidden 测试经 INTERFACE_DOCS 自动覆盖）；`test_schemas.py` 改写 fixture 为 unit-level 并新增 group fixture + 15 项 unit-level validator 用例。

新增或修改文件：

- docs/skill/attention_anchor_labels_interface.md（新增）
- src/recover_attention/schemas.py
- docs/skill/label_schema.md
- docs/skill/SKILL.md
- tests/test_interface_consistency.py
- tests/test_schemas.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口设计与对齐。

输出文件：

- 无 `data/processed` 产物。

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --write
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_schemas.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：9 个 interface block 全部 in sync（`attention_anchor_label` = 18 fields，由脚本生成）。
- tests/test_interface_consistency.py：33 passed, 2 skipped。
- tests/test_schemas.py：113 passed。
- 全量 pytest：298 passed, 2 skipped。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（本机 `conda` 不在 PATH）。
- `attention_anchor_labels` 目前只有接口与 validator，未实现 builder，未生成 `attention_anchor_labels.jsonl`。
- 当前接口是 unit-level；span/token-level expansion 留待后续。
- guidance_action / guidance_strength 未接入（已 forbidden）；trajectory / answer stability / raw attention 仍未接入。

下一步建议：

- Sprint 1J：Build Attention Anchor Labels。

## Sprint 1J：Build Attention Anchor Labels

已完成内容：

- 实现 `unit_evidence.jsonl → attention_anchor_labels.jsonl` 数据转换阶段（unit-level，每条 unit_evidence 生成一条 attention_anchor_label）。
- 实现 deterministic backend `early_evidence_rule_stub_v0`（只用于管线验证，非真实 attention importance）：
  - `recoverability_risk_score`：Misleading Recovery=1.0 / Non-recoverable=0.75 / Partially Recoverable=0.50 / Recoverable=0.25。
  - `attention_importance_score = 0.6 * semantic_score + 0.4 * recoverability_risk_score`，clamp 到 [0,1]。
  - label rule：Misleading Recovery / misleading_recovery=True → Risky Anchor；>=0.75 Strong；>=0.55 Medium；>=0.35 Weak；否则 Distractor。
  - `label_status = partial_evidence_label`；`attention_anchor_label_id = f"{unit_evidence_id}__anchor_{label_backend}"`。
- 从 unit_evidence 复制 unit metadata 与 early evidence，本阶段创建 label 字段与 evidence trace（含 limitations，明确 partial evidence、无 trajectory/answer/raw attention/guidance）。
- 每条输出写入前调用 `validate_attention_anchor_label_record`；未改 schema / interface / label_schema。

新增或修改文件：

- src/recover_attention/attention_anchor_labels.py（新增）
- scripts/11_build_attention_anchor_labels.py（新增）
- tests/test_attention_anchor_labels.py（新增）
- data/processed/attention_anchor_labels.jsonl（生成）
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/unit_evidence.jsonl

输出文件：

- data/processed/attention_anchor_labels.jsonl

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_attention_anchor_labels.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：全部 in sync。
- tests/test_interface_consistency.py：33 passed, 2 skipped。
- tests/test_attention_anchor_labels.py：26 passed。
- 全量 pytest：324 passed, 2 skipped。

attention anchor label 数量统计：

- num_input_unit_evidence = 46，num_output_attention_anchor_labels = 46。
- attention_anchor_label 分布：{Medium Anchor: 46}。
- unit_scope：{single: 36, group: 10}；group_type：{single: 36, number_set: 5, repeated_surface: 5}。
- num_risky_anchor = 0。

score min / max / mean：

- score_min = 0.55，score_max = 0.61，score_mean ≈ 0.5578。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（本机 `conda` 不在 PATH）。
- `attention_anchor_labels.jsonl` 来自 `early_evidence_rule_stub_v0`，基于 partial early evidence，只用于管线验证，不代表真实 attention importance。
- guidance_action / guidance_strength 未生成（已 forbidden）；尚未实现 attention guidance。
- trajectory stability、answer stability、raw attention pattern、attention steering effect 尚未接入。
- 非阻塞文档残留：`attention_anchor_labels_interface.md` / `label_schema.md` 的 `semantic_evidence` 示例 `summary_label="Information Loss"` 与真实 unit_evidence 的 `summary_label` 词表（consistent_/mixed_/no_semantic_necessity_evidence）不一致；rule 不依赖该取值，validator 不校验该键，故不阻塞，可在后续 doc-fix 顺手修正。

下一步建议：

- Sprint 1K-prep：Guidance Boundary and Intervention Manifest Review。

## Sprint 1K-prep：Guidance Boundary and Intervention Manifest Interface Alignment

已完成内容：

- 设计并注册 unit-level **planned-only** 的 `intervention_manifest` 接口（上游 `attention_anchor_labels.jsonl`）。
- `schemas.py`：
  - 新增 enum：`ALLOWED_INTERVENTION_TYPES={mask,remove,replace}`、`ALLOWED_INTERVENTION_TARGET_SCOPES={unit}`、`ALLOWED_INTERVENTION_BACKENDS={manifest_stub_v0}`、`ALLOWED_INTERVENTION_STATUSES={planned_only}`。
  - 新增 `REQUIRED_FIELDS["intervention_manifest"]`（20 字段，含 intervention_id / attention_anchor_label_id / unit_evidence_id / unit metadata / anchor 信息 / intervention_type / target_scope / intervention_backend / intervention_status / planned_operation / evidence）。
  - 新增 `FORBIDDEN_FIELDS["intervention_manifest"]`，拒绝旧 span-level 顶层字段、guidance 字段、baseline/guided/intervened answer、trajectory/answer stability score、raw_attention_score、hidden states/attention maps 及 `hidden_states_path`/`attentions_path`、probe 字段。
  - `INTERFACE_DOCS` 注册 intervention_manifest。
  - 新增 `validate_intervention_manifest_record`（unit-level + single/group + span 顺序；ID = `f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"`；enum/range/planned_operation dict 校验）。
- 新增 `docs/skill/intervention_manifest_interface.md`（含 `<!-- required_fields:intervention_manifest -->` 生成块，由 sync 写入 20 字段；Purpose / Pipeline / Field Sources / ID Rule / Planned-only & Unit-level Boundary / Not Included / Validator / Example）。
- `label_schema.md`：§14 降级为指向新 interface 的要点（旧 span-level 废弃、unit-level、planned-only、不含 hidden/attention path 与 guidance）。§0 无需改（原本未列 intervention_manifest）。
- `SKILL.md`：Document Router 新增 §3.8.10。
- 测试：`test_interface_consistency.py` 的 `LABEL_SCHEMA_SECTIONS` 新增 intervention_manifest（marker/forbidden 经 INTERFACE_DOCS 自动覆盖）；`test_schemas.py` 新增 unit-level + group fixture 与 18 项 validator 用例。

新增或修改文件：

- docs/skill/intervention_manifest_interface.md（新增）
- src/recover_attention/schemas.py
- docs/skill/label_schema.md
- docs/skill/SKILL.md
- tests/test_interface_consistency.py
- tests/test_schemas.py
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- 无数据输入。本轮只做接口设计与对齐。

输出文件：

- 无 `data/processed` 产物。

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --write
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_schemas.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：10 个 interface block 全部 in sync（`intervention_manifest` = 20 fields，由脚本生成）。
- tests/test_interface_consistency.py：37 passed, 2 skipped。
- tests/test_schemas.py：132 passed。
- 全量 pytest：347 passed, 2 skipped。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（本机 `conda` 不在 PATH）。
- `intervention_manifest` 目前只有接口与 validator，未实现 builder，未生成 `intervention_manifest.jsonl`。
- 当前接口是 unit-level planned-only；hidden_states_path / attentions_path、guidance_action / guidance_strength 均未接入（已 forbidden）；未执行 attention guidance。
- trajectory stability、answer stability、raw attention pattern 仍未接入。

下一步建议：

- Sprint 1K：Build Intervention Manifest。

## Sprint 1K：Build Intervention Manifest

已完成内容：

- 实现 `attention_anchor_labels.jsonl → intervention_manifest.jsonl` 的 planned-only 数据转换阶段（unit-level，每条 anchor label 生成一条 manifest，不筛选 label）。
- 实现 deterministic backend `manifest_stub_v0`（只用于管线验证，非执行结果）：
  - 默认 `intervention_type=mask`、`target_scope=unit`、`intervention_status=planned_only`。
  - `intervention_id = f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"`。
  - `planned_operation`（dict）记录 operation_name / description / target_scope / target_span_ids / target_texts / mask_token / execution_required；不含 intervened_question / guided_answer / hidden_states_path / attentions_path / stability score。
  - `evidence` 记录 source ids / source_files / selection_policy（all_attention_anchor_labels_included_for_pipeline_validation）/ limitations / notes。
- 从 attention_anchor_label 复制 unit metadata 与 anchor 信息；每条输出写入前调用 `validate_intervention_manifest_record`。未改 schema / interface / label_schema。
- CLI 暴露 `--intervention-type`（默认 mask，仅 mask 路径被验收）/ `--backend` / `--mask-token`。

新增或修改文件：

- src/recover_attention/intervention_manifest.py（新增）
- scripts/12_build_intervention_manifest.py（新增）
- tests/test_intervention_manifest.py（新增）
- data/processed/intervention_manifest.jsonl（生成）
- PROGRESS.md
- docs/progress/sprint_1_history.md

输入文件：

- data/processed/attention_anchor_labels.jsonl

输出文件：

- data/processed/intervention_manifest.jsonl

运行命令：

```bash
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/sync_interface_fields.py --check
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_interface_consistency.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe scripts/12_build_intervention_manifest.py --input data/processed/attention_anchor_labels.jsonl --output data/processed/intervention_manifest.jsonl --intervention-type mask --backend manifest_stub_v0 --mask-token "[MASK]"
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest tests/test_intervention_manifest.py -q
D:\conda\Miniconda3\envs\recover_attention\python.exe -m pytest -q
```

检查结果：

- sync_interface_fields --check：全部 in sync。
- tests/test_interface_consistency.py：37 passed, 2 skipped。
- tests/test_intervention_manifest.py：20 passed。
- 全量 pytest：367 passed, 2 skipped。

intervention_manifest 数量统计：

- num_input_attention_anchor_labels = 46，num_output_intervention_manifest = 46。
- intervention_type 分布：{mask: 46}（num_mask=46, num_remove=0, num_replace=0）。
- intervention_status 分布：{planned_only: 46}；target_scope：{unit: 46}。
- attention_anchor_label 分布：{Medium Anchor: 46}；unit_scope：{single: 36, group: 10}。

遗留问题：

- 裸 `python` 当前指向 base conda；本轮经由 `D:\conda\Miniconda3\envs\recover_attention\python.exe` 运行（本机 `conda` 不在 PATH）。
- `intervention_manifest.jsonl` 来自 `manifest_stub_v0`，是 planned-only manifest，只用于管线验证，不代表 intervention 已执行或 attention guidance 已实现。
- 未生成 guidance_action / guidance_strength；未写 hidden_states_path / attentions_path；未执行模型；未做 trajectory / answer stability / raw attention / probe。

下一步建议：

- Sprint 1L-prep：Sprint 1 Boundary Review and Refactor Freeze Plan。
