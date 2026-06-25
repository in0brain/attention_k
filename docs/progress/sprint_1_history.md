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
- 生成 `data/processed/masked_questions.jsonl`。

新增或修改文件：

- src/recover_attention/masked_questions.py
- scripts/07_build_masked_questions.py
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
conda run -n recover_attention python scripts/sync_interface_fields.py
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
- `python -m pytest tests/test_masked_questions.py -q` 已通过，结果为 `12 passed`。
- `python -m pytest -q` 已通过，结果为 `180 passed, 2 skipped`。

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
- Sprint 1F task card 要求 `python scripts/sync_interface_fields.py --check`，但当前脚本只支持 `--write`，无参模式是 check-only；本轮采用无参检查并通过。
- 本轮未生成 recover_outputs、recover_scores、labels、token_labels 或 attention_anchor_labels。
- 本轮未实现 recovery、recoverability scoring、trajectory、attention guidance 或 probe。

下一步建议：

- Sprint 1G：Question Recovery（unit-level recover outputs）。
