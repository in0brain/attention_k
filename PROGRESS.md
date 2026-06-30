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
Sprint 2A-real implementation prepared，但真实 4bit hidden-state cache run 被 blocked_by_missing_bitsandbytes 阻断。
hf_local_causal_lm_hidden_states_v0 的本地 HF backend、CLI 参数、local_files_only=True 边界、bitsandbytes 缺失检测和测试已准备完成。
真实输出目录 outputs/logs/sprint_2A_real_hidden_state_cache 未生成成功产物；未 fallback 到 fp16 全量加载，未联网下载模型，未修改 Sprint 1Q / 1R / 2A stub 输出。
下一步建议是安装/修复 bitsandbytes 后重跑 Sprint 2A-real 真实 cache 命令，或由用户明确确认其他低显存本地方案。
```

当前不做：

- 重跑 Ollama
- 重跑 NLI
- 修改 recovery prompt / recovery scoring / masked question construction
- 训练 probe
- 执行 attention guidance
- fallback 到 fp16 全量加载真实 HF 模型
- 联网下载模型
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
| Sprint 1J | 完成 | Build attention anchor labels |
| Sprint 1K-prep | 完成 | Guidance boundary & intervention manifest interface alignment |
| Sprint 1K | 完成 | Build intervention manifest |
| Sprint 1L | 完成 | Plug real bilingual NLI backend |
| Sprint 1M | 完成 | Plug real LLM recovery backend |
| Sprint 1N | 完成 | Rebuild downstream with real NLI and real LLM recovery outputs |
| Sprint 1O | 完成 | Upgrade real recovery scoring |
| Sprint 1P | 完成 | Rebuild downstream with upgraded recovery scoring |
| Sprint 1Q | 完成 | Real Signal Quality Review human labels |
| Sprint 1R | 完成 | Human Review Consolidation & Known Issue Freeze |
| Sprint 2A | 完成 | Hidden State Cache Baseline |
| Sprint 2A-real | 阻塞 | Real hidden-state cache backend prepared；run blocked by missing bitsandbytes |

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
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output outputs/logs/nli_scores_real_auto_small.jsonl --backend hf_nli_auto_v0 --language auto --en-model models/nli/en/roberta-large-mnli --zh-model models/nli/zh/mdeberta-v3-base-xnli --device auto --limit 20
conda run -n recover_attention python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
conda run -n recover_attention python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output outputs/logs/recover_outputs_ollama_small.jsonl --backend ollama_chat_v0 --model qwen3.5:9b --ollama-base-url http://localhost:11434 --num-samples 1 --temperature 0.0 --top-p 1.0 --max-tokens 128 --timeout 120 --seed 42 --limit 10
conda run -n recover_attention python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
conda run -n recover_attention python scripts/09_score_recovery.py --input outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl --output outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl --backend nli_recovery_judge_v0 --nli-backend hf_nli_auto_v0 --language auto --en-model models/nli/en/roberta-large-mnli --zh-model models/nli/zh/mdeberta-v3-base-xnli --device auto --max-length 512 --label-order auto --recoverable-entailment-threshold 0.70 --partial-entailment-threshold 0.50 --contradiction-threshold 0.50 --report-output outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
conda run -n recover_attention python scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
conda run -n recover_attention python scripts/12_build_intervention_manifest.py --input data/processed/attention_anchor_labels.jsonl --output data/processed/intervention_manifest.jsonl --intervention-type mask --backend manifest_stub_v0 --mask-token "[MASK]"
conda run -n recover_attention python scripts/13_rebuild_downstream_real_signals.py --ablated-questions data/processed/ablated_questions.jsonl --output-dir outputs/logs/sprint_1N_real_downstream --nli-backend hf_nli_auto_v0 --language auto --en-model models/nli/en/roberta-large-mnli --zh-model models/nli/zh/mdeberta-v3-base-xnli --recovery-backend ollama_chat_v0 --ollama-model qwen3.5:9b --ollama-base-url http://localhost:11434 --num-samples 1 --temperature 0.0 --top-p 1.0 --max-tokens 128 --timeout 120 --seed 42
conda run -n recover_attention python scripts/14_rebuild_downstream_upgraded_recovery_scoring.py --semantic-labels outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl --upgraded-recover-scores outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl --baseline-recover-scores outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl --baseline-unit-evidence outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl --baseline-attention-anchor-labels outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl --baseline-intervention-manifest outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl --baseline-report outputs/logs/sprint_1N_real_downstream/real_signal_report.json --output-dir outputs/logs/sprint_1P_upgraded_downstream --unit-evidence-backend aggregate_stub_v0 --attention-label-backend early_evidence_rule_stub_v0 --intervention-type mask --intervention-backend manifest_stub_v0 --mask-token "[MASK]"
conda run -n recover_attention python scripts/15_consolidate_human_review.py
conda run -n recover_attention python scripts/16_cache_hidden_states.py --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl --output-dir outputs/logs/sprint_2A_hidden_state_cache_baseline --backend stub_hidden_state_v0 --layer-indices 0 1 2 --hidden-size 8 --mask-token "[MASK]" --overwrite
conda run -n recover_attention python -m pytest -q
```

最近一次检查结果：

```text
pytest: 465 passed, 2 skipped
smoke test: passed
candidate extraction: passed
ablation unit construction: passed
ablated question construction: passed
nli scoring stub: passed
real bilingual NLI backend integration: passed
stub NLI regression: passed
local/remote model loading policy: passed
semantic label rule: passed
masked question construction: passed
recover output interface alignment: passed
recover output self-contained interface refinement: passed
question recovery stub: passed
real LLM recovery backend integration: passed
oracle recovery stub regression: passed
prompt leakage guard: passed
qwen3.5:9b real smoke: passed
ollama_chat_v0 smoke: 10 records, num_empty_recoveries=0
recover score interface alignment: passed
recover score governance doc cleanup: passed
recoverability scoring stub: passed
real recovery scoring upgrade: passed
nli_recovery_judge_v0 small run: passed
nli_recovery_judge_v0 full run: passed, 46 records
stub_rule_v0 regression: passed
recovery scoring report: passed
unit evidence interface alignment: passed
unit evidence build passed
unit evidence interface post-build cleanup: passed
attention anchor label interface alignment: passed
attention anchor label build passed
intervention manifest interface alignment: passed
intervention manifest build passed
real downstream rebuild orchestration: passed
real NLI downstream rebuild: 92 records, backend=hf_nli_auto_v0, language_counts={en: 92}
real LLM recovery downstream rebuild: 46 records, backend=ollama_chat_v0, model=qwen3.5:9b, empty_recovery_count=0
real signal report: exact_match_recovery_count=18, mask_remaining_count=0
upgraded downstream rebuild orchestration: passed
upgraded unit evidence build: passed, 46 records
upgraded attention anchor labels build: passed, 46 records
upgraded intervention manifest build: passed, 46 records
upgraded downstream comparison report: passed
upgraded comparison: recoverability_label_changed_count=15, attention_anchor_label_changed_count=15
human review consolidation: passed, reviewed_count=20, unreviewed_count=0, manifest_count=20, validation_warning_count=0
human review summary: auto_vs_human_recoverability_disagreement_count=3, auto_vs_human_attention_anchor_disagreement_count=4
human review read-only check: labels JSONL / report JSON / human review sheet hashes unchanged
known issues freeze: passed
Sprint 2A manifest: passed, 20 records
hidden state cache baseline: passed, backend=stub_hidden_state_v0, num_cases=20, num_inputs_total=60, num_hidden_state_files=60
hidden state input_type_counts: {original: 20, masked: 20, recovered: 20}
token alignment report: single_mask_cases=17, group_mask_cases=3, fragment_recovery_outputs=8, alignment_warning_count=8
hidden state cache read-only check: Sprint 1Q / 1R input hashes unchanged
real hidden state backend implementation: prepared, backend=hf_local_causal_lm_hidden_states_v0
real hidden state 4bit run: blocked_by_missing_bitsandbytes, no fp16 fallback attempted, no real output manifest generated
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
- src/recover_attention/nli_scoring.py（支持 stub_v0 / hf_nli_en_v0 / hf_nli_zh_v0 / hf_nli_auto_v0）
- src/recover_attention/semantic_labels.py
- src/recover_attention/masked_questions.py
- src/recover_attention/recover_generation.py（支持 oracle_stub_v0 / ollama_chat_v0）
- src/recover_attention/recover_scoring.py
- src/recover_attention/unit_evidence.py
- src/recover_attention/attention_anchor_labels.py
- src/recover_attention/intervention_manifest.py
- src/recover_attention/human_review_consolidation.py
- src/recover_attention/token_alignment.py
- src/recover_attention/hidden_state_cache.py
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
- scripts/11_build_attention_anchor_labels.py
- scripts/12_build_intervention_manifest.py
- scripts/13_rebuild_downstream_real_signals.py
- scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
- scripts/15_consolidate_human_review.py
- scripts/16_cache_hidden_states.py
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
- tests/test_attention_anchor_labels.py
- tests/test_intervention_manifest.py
- tests/test_rebuild_downstream_real_signals.py
- tests/test_rebuild_downstream_upgraded_recovery_scoring.py
- tests/test_human_review_consolidation.py
- tests/test_token_alignment.py
- tests/test_hidden_state_cache.py
- data/processed/candidate_spans.jsonl
- data/processed/ablation_units.jsonl
- data/processed/ablated_questions.jsonl
- data/processed/nli_scores.jsonl
- data/processed/semantic_labels.jsonl
- data/processed/masked_questions.jsonl
- data/processed/recover_outputs.jsonl
- data/processed/recover_scores.jsonl
- data/processed/unit_evidence.jsonl
- data/processed/attention_anchor_labels.jsonl
- data/processed/intervention_manifest.jsonl
- outputs/logs/recover_outputs_stub_check.jsonl
- outputs/logs/recover_outputs_ollama_small.jsonl
- outputs/logs/sprint_1N_real_downstream/nli_scores_real.jsonl
- outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
- outputs/logs/sprint_1N_real_downstream/masked_questions_real.jsonl
- outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl
- outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
- outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
- outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
- outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
- outputs/logs/sprint_1N_real_downstream/real_signal_report.json
- outputs/logs/sprint_1N_real_downstream/real_signal_report.md
- outputs/logs/sprint_1O_recovery_scoring/recover_scores_stub_check.jsonl
- outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge_small.jsonl
- outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report_small.json
- outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
- outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
- outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.md
- outputs/logs/sprint_1P_upgraded_downstream/unit_evidence_upgraded.jsonl
- outputs/logs/sprint_1P_upgraded_downstream/attention_anchor_labels_upgraded.jsonl
- outputs/logs/sprint_1P_upgraded_downstream/intervention_manifest_upgraded.jsonl
- outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.json
- outputs/logs/sprint_1P_upgraded_downstream/upgraded_downstream_report.md
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_guide.md
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
- outputs/logs/sprint_1Q_real_signal_quality_review/upgraded_downstream_report_with_human_fields.json
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_summary.json
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_known_issues.md
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
- outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
- docs/skill/semantic_labels_interface.md
- docs/skill/recover_outputs_interface.md
- docs/skill/recover_scores_interface.md
- docs/skill/unit_evidence_interface.md
- docs/skill/attention_anchor_labels_interface.md
- docs/skill/intervention_manifest_interface.md
- docs/skill/*
- README.md
- AGENTS.md

下一阶段可能新增或修改：

- representation feature extraction（Sprint 2B）

具体以后续 Sprint 2A task card 为准。

## 5. 当前遗留问题

- 裸 `python` 当前指向 base conda：`D:\conda\Miniconda3\python.exe`；当前验收使用 `conda run -n recover_attention python ...`。
- `data/processed/*` 是本地生成产物目录；Sprint 1N 真实 downstream 产物写入 `outputs/logs/sprint_1N_real_downstream/`，未覆盖 processed 主线文件。
- 真实 NLI backend 默认优先使用 `models/nli/en/roberta-large-mnli` 与 `models/nli/zh/mdeberta-v3-base-xnli` 本地模型；只有显式传入 `--allow-download` 时才允许远程模型加载。
- 当前 `data/processed/ablated_questions.jsonl` 没有中文样本；Sprint 1N 全量真实 run 只实际加载英文 NLI 模型，中文 routing 仍由 mock test 覆盖。
- `nli_recovery_judge_v0` 是 question-level NLI judge，不直接验证每个 masked span。
- `unit_evidence_upgraded` 仍使用 `aggregate_stub_v0`。
- `attention_anchor_labels_upgraded` 仍使用 `early_evidence_rule_stub_v0`。
- `intervention_manifest_upgraded` 仍是 `planned_only`，不代表 intervention 已执行。
- 本 sprint 只比较 upgraded scoring 对 downstream labels 的影响，不证明 attention guidance 有效。
- Sprint 1Q human review guide 的表格仍为空；Sprint 1R 使用已填好的 `sprint_1Q_human_review_labels_template.jsonl` 作为结构化来源，只读校验 report JSON，不重新同步或覆盖 report JSON。
- Sprint 1R 只冻结 known issues，不实现 full-question validator、span-aware numeric scorer、entity/unit consistency scorer 或 unit/group budget selector。
- Sprint 2A hidden states 来自 deterministic stub backend，不是真实模型 hidden states。
- Sprint 2A token alignment 是基础 deterministic 对齐，不处理复杂 paraphrase。
- Sprint 2A recovered fragment 输出只记录 warning，不中断 cache；后续在 2B / 2C 判断是否保留。
- Sprint 2A-real 的真实 backend 已实现本地 `local_files_only=True` 加载边界，但当前 `recover_attention` 环境缺少 `bitsandbytes`，4bit 真实 cache run 已按 task card 停止。
- `outputs/logs/sprint_2A_real_hidden_state_cache/` 未生成成功产物；当前没有真实 hidden states cache 可供 2B 消费。
- `docs/skill/nli_scores_interface.md` 仍有旧阶段文字提到 Sprint 1D 只支持 `stub_v0`；Sprint 1N task card 禁止修改 interface docs，本轮以脚本、schema 和测试为准。
- 当前没有接入真实 hidden states / attention maps / trajectory stability / answer stability / raw attention / attention guidance / probe。
- 当前没有声称 attention guidance 有效，也没有声称减少 hallucination。

## 6. 下一步

下一步建议：

```text
安装/修复 bitsandbytes 后重跑 Sprint 2A-real 真实 hidden-state cache。
```

注意：

```text
不要自动 fallback 到 fp16 全量加载。
真实 cache 成功前，不要将 Sprint 2A-real 标记为完成。
```
