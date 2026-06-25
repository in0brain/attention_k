# 实验进度记录：Reasoning-Aware Attention Guidance

## 1. 当前项目状态

当前主线：

```text
Reasoning-Aware Attention Guidance
```

完整研究闭环：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

当前阶段：

```text
Sprint 1F 已完成。
Sprint 1G 前置接口修正已完成：recover_outputs 已对齐到 self-contained unit-level / masked_id-driven schema。
下一步建议是 Sprint 1G：Question Recovery（unit-level recover outputs implementation）。
```

当前不做：

- 真实模型调用
- hidden states 大规模缓存
- trajectory stability
- attention guidance
- probe training
- 大规模实验

## 2. 已完成 Sprint 摘要

| Sprint | 状态 | 摘要 |
|---|---|---|
| Sprint 0A | 完成 | 项目骨架与文档约束 |
| Sprint 0A-docs | 完成 | docs/skill、docs/reference、docs/codex_tasks 结构 |
| Sprint 0B | 完成 | jsonl 读写、样例数据、smoke test |
| Sprint 0C | 完成 | 基础 schema 校验 |
| Sprint 0D | 完成 | prepare_data |
| Sprint 0E | 完成 | 工程基础验收 |
| Sprint 0F | 完成 | 文档主线对齐 |
| Sprint 0G | 完成 | schema 与 Attention Anchor 标签体系对齐 |
| Sprint 0H | 完成 | PROGRESS.md 瘦身与 Sprint 0 历史归档 |
| Sprint 1A | 完成 | Candidate span extraction framework |
| Sprint 1B | 完成 | Ablation unit construction |
| Sprint 1C | 完成 | Ablated question construction |
| Sprint 1D | 完成 | NLI semantic consistency scoring stub |
| Sprint 1E | 完成 | Semantic necessity label rule |
| Sprint 1F | 完成 | Unit-level masked question construction |
| Sprint 1G-prep | 完成 | Self-contained recover output interface alignment |

详细历史见：

```text
docs/progress/sprint_0_history.md
docs/progress/sprint_1_history.md
```

## 3. 当前可运行命令

```bash
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
conda run -n recover_attention python scripts/02_extract_candidate_spans.py --input data/processed/questions.jsonl --output data/processed/candidate_spans.jsonl
conda run -n recover_attention python scripts/03_build_ablation_units.py --input data/processed/candidate_spans.jsonl --output data/processed/ablation_units.jsonl
conda run -n recover_attention python scripts/04_build_ablated_questions.py --input data/processed/ablation_units.jsonl --output data/processed/ablated_questions.jsonl
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores.jsonl --backend stub_v0 --language auto
conda run -n recover_attention python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
conda run -n recover_attention python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
conda run -n recover_attention python -m pytest -q
```

最近一次检查结果：

```text
pytest: 190 passed, 2 skipped
smoke test: passed
candidate extraction: passed
ablation unit construction: passed
ablated question construction: passed
nli scoring stub: passed
semantic label rule: passed
masked question construction: passed
recover output interface alignment: passed
recover output self-contained interface refinement: passed
```

## 4. 当前关键文件状态

已完成：

- src/recover_attention/data_io.py
- src/recover_attention/schemas.py
- src/recover_attention/prepare_data.py
- src/recover_attention/candidate_extraction.py
- src/recover_attention/ablation_units.py
- src/recover_attention/question_ablations.py
- src/recover_attention/semantic_labels.py
- src/recover_attention/masked_questions.py
- scripts/00_smoke_test.py
- scripts/01_prepare_data.py
- scripts/02_extract_candidate_spans.py
- scripts/03_build_ablation_units.py
- scripts/04_build_ablated_questions.py
- scripts/05_run_nli_scoring.py
- scripts/06_build_semantic_labels.py
- scripts/07_build_masked_questions.py
- tests/test_data_io.py
- tests/test_schemas.py
- tests/test_prepare_data.py
- tests/test_candidate_extraction.py
- tests/test_ablation_units.py
- tests/test_question_ablations.py
- tests/test_nli_scoring.py
- tests/test_semantic_labels.py
- tests/test_masked_questions.py
- data/processed/candidate_spans.jsonl
- data/processed/ablation_units.jsonl
- data/processed/ablated_questions.jsonl
- data/processed/nli_scores.jsonl
- data/processed/semantic_labels.jsonl
- data/processed/masked_questions.jsonl
- docs/skill/semantic_labels_interface.md
- docs/skill/recover_outputs_interface.md
- docs/skill/*
- README.md
- AGENTS.md

下一阶段将新增或修改：

- src/recover_attention/recover_generation.py
- scripts/08_run_recovery.py
- tests/test_recover_generation.py
- data/processed/recover_outputs.jsonl

具体以后续 Sprint 1G task card 为准。

## 5. 当前遗留问题

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；当前验收使用 `conda run -n recover_attention python ...`。
- git 工作区中仍存在此前 sprint 的文档迁移和 schema/test 改动，需要在提交前统一检查。
- 如果 `__pycache__` / `.pyc` 出现在 git status 中，需要确认是否被 git 跟踪；若已被跟踪，应单独处理。
- `data/processed/*` 是本地生成产物目录，当前被 `.gitignore` 忽略；PROGRESS 中列出的 processed jsonl 不代表会提交到 GitHub。
- `recover_outputs.jsonl` 目前只完成 self-contained unit-level / masked_id-driven 接口修正，尚未实现 recovery 生成。
- `recover_scores.jsonl` 仍是旧 span-level schema；进入 recoverability scoring 前需要单独做 unit-level 接口修正。
- 不要从 masked questions 自动扩展到 recovery / recoverability / attention guidance。

## 6. 下一步

下一步建议：

```text
Sprint 1G：Question Recovery（unit-level recover outputs implementation）
```

注意：

```text
不要自动开始 Sprint 1G。
必须先有 Sprint 1G task card 或用户明确指令。
```
