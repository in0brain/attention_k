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
