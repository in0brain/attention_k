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
Sprint 1J-prep 已完成：Attention Anchor Label Interface Alignment。
attention_anchor_label 接口已对齐为 unit-level，并纳入 interface governance（INTERFACE_DOCS / sync / 测试）。
下一步建议是 Sprint 1J：Build Attention Anchor Labels。
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
| Sprint 1G | 完成 | Question recovery oracle stub |
| Sprint 1H-prep | 完成 | Recover score interface alignment |
| Sprint 1H-prep-fix | 完成 | Recover score governance 文档残留修补 |
| Sprint 1H | 完成 | Recoverability scoring stub |
| Sprint 1I-prep-a | 完成 | Unit evidence interface design |
| Sprint 1I | 完成 | Build unit evidence |
| Sprint 1I-doc-fix | 完成 | Unit evidence interface post-build cleanup |
| Sprint 1J-prep | 完成 | Attention anchor label interface alignment |

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
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
conda run -n recover_attention python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
conda run -n recover_attention python -m pytest -q
```

最近一次检查结果：

```text
pytest: 298 passed, 2 skipped
smoke test: passed
candidate extraction: passed
ablation unit construction: passed
ablated question construction: passed
nli scoring stub: passed
semantic label rule: passed
masked question construction: passed
recover output interface alignment: passed
recover output self-contained interface refinement: passed
question recovery stub: passed
recover score interface alignment: passed
recover score governance doc cleanup: passed
recoverability scoring stub: passed
unit evidence interface alignment: passed
unit evidence build passed
unit evidence interface post-build cleanup: passed
attention anchor label interface alignment: passed
sync_interface_fields --check: all in sync
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
- src/recover_attention/recover_generation.py
- src/recover_attention/recover_scoring.py
- src/recover_attention/unit_evidence.py
- scripts/00_smoke_test.py
- scripts/01_prepare_data.py
- scripts/02_extract_candidate_spans.py
- scripts/03_build_ablation_units.py
- scripts/04_build_ablated_questions.py
- scripts/05_run_nli_scoring.py
- scripts/06_build_semantic_labels.py
- scripts/07_build_masked_questions.py
- scripts/08_run_recovery.py
- scripts/09_score_recovery.py
- scripts/10_build_unit_evidence.py
- tests/test_data_io.py
- tests/test_schemas.py
- tests/test_prepare_data.py
- tests/test_candidate_extraction.py
- tests/test_ablation_units.py
- tests/test_question_ablations.py
- tests/test_nli_scoring.py
- tests/test_semantic_labels.py
- tests/test_masked_questions.py
- tests/test_recover_generation.py
- tests/test_recover_scoring.py
- tests/test_unit_evidence.py
- data/processed/candidate_spans.jsonl
- data/processed/ablation_units.jsonl
- data/processed/ablated_questions.jsonl
- data/processed/nli_scores.jsonl
- data/processed/semantic_labels.jsonl
- data/processed/masked_questions.jsonl
- data/processed/recover_outputs.jsonl
- data/processed/recover_scores.jsonl
- data/processed/unit_evidence.jsonl
- docs/skill/semantic_labels_interface.md
- docs/skill/recover_outputs_interface.md
- docs/skill/recover_scores_interface.md
- docs/skill/unit_evidence_interface.md
- docs/skill/attention_anchor_labels_interface.md
- docs/skill/*
- README.md
- AGENTS.md

下一阶段可能新增或修改：

- src/recover_attention/attention_anchor_labels.py
- scripts/11_build_attention_anchor_labels.py
- tests/test_attention_anchor_labels.py
- data/processed/attention_anchor_labels.jsonl

具体以后续 Sprint 1J task card 为准。

## 5. 当前遗留问题

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；当前验收使用 `conda run -n recover_attention python ...`。
- 如果 `__pycache__` / `.pyc` 出现在 git status 中，需要确认是否被 git 跟踪；若已被跟踪，应单独处理。
- `data/processed/*` 是本地生成产物目录，当前被 `.gitignore` 忽略；PROGRESS 中列出的 processed jsonl 不代表会提交到 GitHub。
- `recover_outputs.jsonl` 已由 `oracle_stub_v0` 生成；该 backend 只用于管线验证，不代表真实恢复能力。
- `recover_scores.jsonl` 已由 `oracle_stub_v0` recovery output + `stub_rule_v0` exact-match scorer 生成，只用于管线验证。
- `stub_rule_v0` 只做 `strip` 和连续空白折叠后的 exact normalized match，不做真实语义相似度或模型 judge。
- `unit_evidence` 目前只汇总 semantic necessity 与 semantic recoverability early evidence。
- recoverability 来自 `oracle_stub_v0` + `stub_rule_v0`，只用于管线验证。
- trajectory stability、answer stability、raw attention pattern 仍未接入。
- attention steering effect 尚未接入。
- `unit_evidence` 不是 final attention anchor label。
- 尚未实现 attention_anchor_labels builder。
- `unit_evidence_interface.md` 的旧阶段表述（“only defines interface / does not implement aggregation / builder belongs to a later sprint”）已修正为与已实现 builder 对齐；未改 schema / 字段 / 生成块。
- `attention_anchor_labels` 目前只有接口和 validator，尚未实现 builder。
- 当前 attention anchor label interface 是 unit-level（绑定 id + unit_id，span 信息在 span_ids / spans）。
- span/token-level expansion 尚未实现。
- guidance_action / guidance_strength 尚未接入（已在 FORBIDDEN_FIELDS 中显式拒绝）。
- trajectory stability、answer stability、raw attention pattern 仍未接入。
- 本轮未做真实模型 recovery judging、semantic similarity、attention anchor labeling 或 attention guidance。
- recover_score governance 文档残留已修复：label_schema.md §0 不再把 recover_score 误列为“不受 interface doc 管理”；新增回归测试 `test_label_schema_out_of_scope_examples_do_not_include_managed_interface_types` 防止再漂移。
- 不要从 recover score interface 自动扩展到 attention guidance。

## 6. 下一步

下一步建议：

```text
Sprint 1J：Build Attention Anchor Labels
```

注意：

```text
不要自动开始 Sprint 1J。
必须先有 Sprint 1J task card 或用户明确指令。
```
