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
Sprint 2G-2000 review gate 已完成：Result Review Gate and Final Stage Summary。

Sprint 2G-2000 工程上成功跑通 2000 条 GSM8K weak-labeled dry-run pipeline：
- actual_num_cases = 2000
- real Qwen hidden-state inputs = 6000
- hidden-state cache failure_count = 0
- alignment_warning_count = 0
- adaptive stratified 5-fold probe training completed（min_class_count = 74 >= 5）
- full pytest passed（547 passed, 2 skipped）

但 review gate 判定当前结果暂不适合进入 Sprint 3A implementation：
- risk_positive recall = 0.311，漏检严重（51/74 missed；increase arm 仅 23 条）
- high accuracy（0.9175）主要由 positive_anchor / negative 两个大类驱动（约 80% 数据）
- weak label 与 recovered filler 存在结构性 leakage 风险（recovered filler 按 span type 决定）
- guidance increase arm 只有 23 条，risk-span steering 欠功率

正式产物（只读总结，未重跑 pipeline）：
- outputs/logs/sprint_2G_full_scale_2000/08_review_gate/result_review_gate.{json,md}
- outputs/logs/sprint_2G_full_scale_2000/09_final_stage_summary/sprint_2G_2000_final_stage_summary.{md,json}

当前不建议：
- 不建议现在跑 all-train split（只会放大同一 weak-label leakage 与 risk under-recall 问题）
- 不建议现在进入 Sprint 3A implementation（risk arm 欠功率且部分泄漏）
- 不建议声称 hallucination reduction 或 answer accuracy improvement

下一步建议：
Sprint 2H：Weak Label and Recovery Decoupling Fix（解耦 recovered filler、扩展 risk_positive 规则、加类别权重/阈值校准，先 500 再 2000 重跑，以 risk_positive recall 与 macro_f1 作为 gate）。

Sprint 2G-2000 已完成：Full-scale Weak-labeled 2000-case Dry Run。

source path：data/raw/gsm8k_train_normalized.jsonl
requested_num_cases = 2000，available_num_cases = 7473，actual_num_cases = 2000（seeded_sample, seed=42，无重复/无增强）。
输出目录：outputs/logs/sprint_2G_full_scale_2000/（00_manifest..07_stage_summary）。

核心结论：
- 将 Sprint 2 dry-run diagnostic pipeline 扩展到 2000 条真实 GSM8K question 的 weak-labeled scale，闭环完整：manifest → weak labels → hidden-state cache → representation features → weak probe dataset → probe training → guidance candidate dry run → summary/figures。
- hidden-state cache 使用真实本地 Qwen2.5-7B-Instruct（4-bit, layers 0/8/16/24/27）：num_cases=2000，num_inputs_total=6000，num_hidden_state_files=6000，failure_count=0，alignment 全部 ok。
- weak target 分布（weak_auto）：positive_anchor 1039 / negative 564 / hard_negative_or_weak_positive 323 / risk_positive 74；全部 usable，num_unmapped=0。
- adaptive k-fold：min usable class count=74 ≥ 5 → stratified 5-fold（按 weak probe labels 在生成后决定，不预先固定）。
- weak-labeled probe metrics（诊断用，非 human-supervised validation）：accuracy=0.9175，macro_f1=0.785，weighted_f1=0.909；majority baseline accuracy=0.5195，macro_f1=0.171。
- guidance candidate dry run（planned-only）：candidate_true=1361；actions preserve 1073 / no_guidance 639 / review 265 / increase 23；confidence high 1656 / medium 267 / low 77。全部 dry_run=true、will_modify_attention=false、will_run_model=false、will_change_answer=false。
- 边界：weak-labeled 2000-case dry run，不是 human-reviewed validation；未执行 attention steering；未验证 hallucination reduction；未验证 answer accuracy improvement。20-case human-reviewed Sprint 2 outputs 未被覆盖。

Sprint 2G dataset prep 已完成：Dataset Source Audit and Import Preparation。
本轮未执行 hidden-state cache、representation feature extraction、probe training 或 guidance candidate generation；只做数据源审计与数据集导入/标准化。
核心结论：
- 之前可见最大 JSONL 约 92 行，是 5 条原始 question 的 candidate span / ablation unit / NLI fan-out（distinct source id = 5），不是 92 条独立 source question。
- 仓库原有真正的 raw question source 仅为 5 条玩具样本（data/processed/questions.jsonl、data/examples/questions_small.jsonl）。
- 已导入真实 GSM8K train split 7473 条到 data/raw/gsm8k_train_normalized.jsonl（标准化字段 question_id / source_dataset / source_split / question / answer / metadata），未复制、未重复采样、未数据增强。
- 当前可用 source records = 7473：can_run_500 = true，can_run_2000 = true，can_run_all = true。
- 推荐 Sprint 2G source path：data/raw/gsm8k_train_normalized.jsonl。
- k-fold 不在原始 question dataset 上判断；必须等 weak probe labels 生成后，按 num_folds <= 最小类别数自动决定（5-fold → 3-fold → 2-fold → leave-one-out/holdout），现在不固定 5-fold。
- 不得用 20 条 human-reviewed labels 伪装成 500 / 2000 条 full-scale k-fold 主监督数据。
正式输出：outputs/logs/sprint_2G_dataset_prep/dataset_source_audit.json、dataset_source_audit.md、normalized_dataset_report.json、normalized_dataset_preview.jsonl；完整数据集 data/raw/gsm8k_train_normalized.jsonl。

Sprint 2 final checkpoint 已完成：Stage Summary and Visualization Summary。
sprint_2_stage_summary_v0 已只读汇总 Sprint 2A-real / 2B / 2C / 2D / 2E / 2F 的正式产物，生成 Sprint 2 阶段总结、audit JSON 和 6 张 PNG 可视化。
正式输出为 outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md、outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json 和 outputs/logs/sprint_2_stage_summary/figures/*.png。
核心结论：Sprint 2 形成了 hidden-state cache → representation features → probe dataset → probe training → guidance candidate manifest → closed-loop report 的 dry-run 闭环。
边界说明：Sprint 2E 只是 planned-only guidance candidate dry run；attention guidance 尚未执行；transformer attention weights 尚未修改；CoT 推理尚未在 guidance 下重跑；answer accuracy 尚未验证提升；hallucination reduction 尚未验证。
工程稳定性：Sprint 2E 中曾因并行启动两个 conda run 出现 Windows 临时文件占用，串行重跑后通过；本 checkpoint 串行执行 targeted pytest、stage summary command、full pytest。
工作区状态：pre-existing AM task card state 已观察并保留；未重写 task card；未重跑上游 pipeline scripts 16 / 17 / 18 / 19 / 20 / 21。
下一步建议是 Sprint 3A：Attention Steering Interface Design。
```

当前不做：

- 重跑 Ollama
- 重跑 NLI
- 修改 recovery prompt / recovery scoring / masked question construction
- 重跑 hidden-state cache
- 重训 probe 或扩展 probe training
- 构造 probe train/dev/test split
- 执行 attention guidance
- 联网下载模型
- 大规模实验

## 2. 已完成 Sprint 摘要

| Sprint | 状态 | 摘要 |
|---|---|---|
| Sprint 0A | 完成 | 项目骨架与文档约束 |
| Sprint 0A-docs | 完成 | docs/reasoning-aware-attention-guidance、docs/reference、docs/codex_tasks 结构 |
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
| Sprint 2A-real | 完成 | Real Hidden State Cache Run |
| Sprint 2B | 完成 | Representation Feature Extraction |
| Sprint 2B-fix | 完成 | Representation Feature Extraction Scope Alignment |
| Sprint 2C | 完成 | Probe Dataset Construction |
| Sprint 2D | 完成 | Probe Training Baseline |
| Sprint 2E | 完成 | Guidance Candidate Manifest Dry Run |
| Sprint 2F | 完成 | Mini Closed-loop Report |
| Sprint 2-final-checkpoint | 完成 | Stage Summary and Visualization Summary |

详细历史见：

```text
docs/progress/sprint_0_history.md
docs/progress/sprint_1_history.md
docs/progress/sprint_2_history.md
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
conda run -n recover_attention python scripts/17_extract_representation_features.py --input-manifest outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl --input-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json --metadata outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json --output-dir outputs/logs/sprint_2B_representation_features --backend representation_features_minimal_v0 --overwrite
conda run -n recover_attention python scripts/18_build_probe_dataset.py --features outputs/logs/sprint_2B_representation_features/representation_features.jsonl --feature-report outputs/logs/sprint_2B_representation_features/representation_feature_report.json --output-dir outputs/logs/sprint_2C_probe_dataset --backend probe_dataset_mapping_v0 --overwrite
conda run -n recover_attention python scripts/19_train_probe_baseline.py --dataset outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl --dataset-report outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json --output-dir outputs/logs/sprint_2D_probe_training_baseline --backend probe_training_baseline_v0 --model ridge_classifier_ovr_v0 --cv leave_one_out --seed 42 --overwrite
conda run -n recover_attention python scripts/20_build_guidance_candidate_manifest.py --predictions outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl --eval-report outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json --output-dir outputs/logs/sprint_2E_guidance_candidate_dry_run --backend guidance_candidate_dry_run_v0 --overwrite
conda run -n recover_attention python scripts/21_write_sprint_2_closed_loop_report.py --output-dir outputs/logs/sprint_2F_mini_closed_loop_report --backend sprint_2_closed_loop_report_v0 --overwrite
conda run -n recover_attention python scripts/22_write_sprint_2_stage_summary.py --output-dir outputs/logs/sprint_2_stage_summary --backend sprint_2_stage_summary_v0 --overwrite
conda run -n recover_attention python -m pytest -q
```

最近一次检查结果：

```text
pytest: 515 passed, 2 skipped
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
real hidden state cache run: passed, backend=hf_local_causal_lm_hidden_states_v0, output_dir=outputs/logs/sprint_2A_real_hidden_state_cache
real hidden state cache outputs: hidden_state_manifest.jsonl=60 records, num_cases=20, num_inputs_total=60, num_hidden_state_files=60
representation feature extraction scope alignment: passed
representation feature extraction: passed, backend=representation_features_minimal_v0, output_dir=outputs/logs/sprint_2B_representation_features
representation_features.jsonl: 20 records
representation_feature_report: sprint=2B-fix, num_masked_groups=20, num_recovered_variants=20, num_skipped_groups=0, num_skipped_recovered_variants=0
minimal features: question/span/mask_position cosine distance curves
position pooled features: nullable, position_pool_feature_null_records=8
hidden-state tests discovered: tests/test_hidden_state_cache.py; tests/test_hidden_state_cache_hf.py absent and not a failure
docs/codex_tasks/sprint_2B_representation_feature_extraction.md: pre-existing AM status recorded and scope-aligned
targeted representation feature pytest: 12 passed
sync_interface_fields --check: all in sync
probe dataset construction: passed, backend=probe_dataset_mapping_v0, output_dir=outputs/logs/sprint_2C_probe_dataset
probe_dataset.jsonl: 20 records
probe_dataset_report: num_probe_records=20, num_probe_target_usable=20, num_unmapped=0
probe target counts: risk_positive=7, positive_anchor=3, negative=8, hard_negative_or_weak_positive=2
null representation features retained: records_with_null_position_features=8
targeted probe dataset pytest: 7 passed
probe training baseline: passed, backend=probe_training_baseline_v0, output_dir=outputs/logs/sprint_2D_probe_training_baseline
probe_predictions.jsonl: 20 records
probe_eval_report: status=ok, model=ridge_classifier_ovr_v0, cv=leave_one_out, num_folds=20
probe baseline metrics: accuracy=0.85, macro_f1=0.680952380952381, weighted_f1=0.8285714285714285
binary anchor_or_risk metrics: accuracy=0.9, macro_f1=0.898989898989899
majority baseline: label=negative, accuracy=0.4, macro_f1=0.14285714285714288
feature flattening: num_base_features=99, num_features_with_missing_indicators=198
probe_model.pkl: saved
targeted probe training pytest: 9 passed
guidance candidate dry run: passed, backend=guidance_candidate_dry_run_v0, output_dir=outputs/logs/sprint_2E_guidance_candidate_dry_run
guidance_candidate_manifest.jsonl: 20 records
guidance_candidate_report: status=ok, num_guidance_candidate_records=20, guidance_candidate_true=13, guidance_candidate_false=7
predicted target counts: risk_positive=8, positive_anchor=4, negative=7, hard_negative_or_weak_positive=1
candidate action counts: increase_attention_to_original_span=8, preserve_original_span_attention=4, review_before_guidance=1, no_guidance=7
confidence counts: high=17, medium=3, low=0, unknown=0
guidance correctness diagnostics: prediction_correct=17, prediction_incorrect=3
targeted guidance candidate pytest: 8 passed
mini closed-loop report: passed, backend=sprint_2_closed_loop_report_v0, output_dir=outputs/logs/sprint_2F_mini_closed_loop_report
sprint_2_minimal_closed_loop_report.md: generated
sprint_2_minimal_closed_loop_audit.json: status=ok, hidden_state_cache=true, representation_features=true, probe_dataset=true, probe_training=true, guidance_candidate_dry_run=true
2F boundary audit: executed_attention_steering=false, validated_hallucination_reduction=false, required_boundary_statements_present=true
2F Windows stability note: present, engineering stability issue, serial execution required
2F workspace state note: pre-existing AM task card state observed; pre-existing untracked Sprint 2E files preserved; no upstream pipeline scripts rerun
targeted closed-loop report pytest: 7 passed
Sprint 2 final checkpoint: passed, backend=sprint_2_stage_summary_v0, output_dir=outputs/logs/sprint_2_stage_summary
sprint_2_stage_summary.md: generated
sprint_2_stage_summary_audit.json: status=ok, full_pytest=515 passed / 2 skipped, duration_seconds=10.4918917
stage summary figures: 6 PNGs generated
stage summary boundary audit: executed_attention_steering=false, validated_answer_accuracy_improvement=false, validated_hallucination_reduction=false
stage summary inputs: read formal Sprint 2A-real / 2B / 2C / 2D / 2E / 2F artifacts only
stage summary full pytest: 515 passed, 2 skipped
targeted stage summary pytest: 7 passed
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
- src/recover_attention/representation_features.py
- src/recover_attention/probe_dataset.py
- src/recover_attention/probe_training.py
- src/recover_attention/guidance_candidates.py
- src/recover_attention/closed_loop_report.py
- src/recover_attention/stage_summary.py
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
- scripts/17_extract_representation_features.py
- scripts/18_build_probe_dataset.py
- scripts/19_train_probe_baseline.py
- scripts/20_build_guidance_candidate_manifest.py
- scripts/21_write_sprint_2_closed_loop_report.py
- scripts/22_write_sprint_2_stage_summary.py
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
- tests/test_representation_features.py
- tests/test_probe_dataset.py
- tests/test_probe_training.py
- tests/test_guidance_candidates.py
- tests/test_closed_loop_report.py
- tests/test_stage_summary.py
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
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
- outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
- outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
- outputs/logs/sprint_2B_representation_features/representation_features.jsonl
- outputs/logs/sprint_2B_representation_features/representation_feature_report.json
- outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl（deprecated_extra_outputs）
- outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl（deprecated_extra_outputs）
- outputs/logs/sprint_2B_representation_features/feature_schema.json（deprecated_extra_outputs）
- outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
- outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
- outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
- outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
- outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
- outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json（optional debug output）
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
- outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
- outputs/logs/sprint_2_stage_summary/figures/*.png
- docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
- docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
- docs/reasoning-aware-attention-guidance/recover_scores_interface.md
- docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
- docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
- docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
- docs/reasoning-aware-attention-guidance/*
- README.md
- AGENTS.md

下一阶段可能新增或修改：

- attention steering interface design（Sprint 3A）

具体以后续 Sprint 3A task card 为准。

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
- Sprint 2A recovered fragment 输出只记录 warning，不中断 cache；Sprint 2B 对应 span / mask_position features 可为 null，后续在 2C 判断是否保留或如何标注。
- Sprint 2B 正式输出已收口为 representation_features.jsonl 和 representation_feature_report.json；旧 debug 输出不作为 2C 输入契约。
- Sprint 2B 只抽取 representation features，不构造 probe dataset，不选择 target，不划分 train/dev/test，不生成 guidance candidate manifest。
- Sprint 2C 只构造 probe dataset，不划分 train/dev/test，不训练 probe，不生成 predictions / eval report / model file，不生成 guidance candidate manifest。
- Sprint 2D 只训练最小 probe baseline 并输出诊断指标，不生成 guidance candidate manifest，不执行 attention guidance。
- Sprint 2E 只生成 planned-only guidance candidate manifest，不加载 probe model，不重训 probe，不执行 attention steering。
- Sprint 2F 只总结 Sprint 2 dry-run 闭环，不新增实验，不执行 attention steering，不验证 answer accuracy 或 hallucination reduction。
- Sprint 2 final checkpoint 只生成阶段总结、audit JSON 和可视化 PNG，不新增实验，不重跑上游 pipeline，不读取 hidden-state tensors，不加载 probe_model.pkl。
- 真实 hidden-state cache、2B representation features、2C probe dataset、2D probe baseline、2E guidance candidate dry run、2F closed-loop report 和 Sprint 2 stage summary 只说明最小闭环 dry-run 产物已生成并可审查，不代表 attention guidance 或 hallucination reduction 已验证。
- Sprint 2F 保留并记录了本轮前已有的 `docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md` 和 `docs/codex_tasks/sprint_2F_mini_closed_loop_report.md` 的 AM 状态；未重写 task card。
- Sprint 2F 保留并记录了本轮前已有的 untracked Sprint 2E 文件：`src/recover_attention/guidance_candidates.py`、`scripts/20_build_guidance_candidate_manifest.py`、`tests/test_guidance_candidates.py`。
- Sprint 2 final checkpoint 保留并记录了本轮前已有的 `docs/codex_tasks/sprint_2_final_checkpoint_visualization_summary.md` AM 状态；未重写 task card。
- `docs/reasoning-aware-attention-guidance/nli_scores_interface.md` 仍有旧阶段文字提到 Sprint 1D 只支持 `stub_v0`；Sprint 1N task card 禁止修改 interface docs，本轮以脚本、schema 和测试为准。
- 当前没有接入 attention maps / trajectory stability / answer stability / raw attention / attention guidance。
- 当前没有声称 attention guidance 有效，也没有声称减少 hallucination。

## 6. 下一步

下一步建议：

```text
Sprint 2H：Weak Label and Recovery Decoupling Fix。
- 解耦 recovered question filler 与 span type（统一 neutral filler 或模型生成 recovery）。
- 降低 recovered_cosine label leakage。
- 扩展 / 改进 risk_positive weak label 规则。
- 为 risk_positive 加 class weighting 或 threshold calibration。
- 先 500 条再 2000 条重跑 diagnostic pipeline。
- 以 risk_positive recall 与 macro_f1 作为 gate 指标。
```

注意：

```text
不要自动开始 Sprint 2H / all-train / Sprint 3A；必须先有 task card 或用户明确指令。
Rerun gate（进入 Sprint 3A implementation 前必须满足，阈值待重跑后确认，现在不硬编码 0.7/0.8）：
- risk_positive recall 较 0.311 明显提升；
- risk_positive F1 提升；
- macro_f1 保持稳定；
- confusion matrix 不再把多数 risk_positive 预测成 hard_negative / positive_anchor；
- top features 不再被 recovered filler leakage 主导。
2G-2000 是 weak-labeled diagnostic evidence，不是 human-reviewed validation；
不得把 weak-labeled metrics 说成 attention guidance 有效 / hallucination 减少 / answer accuracy 提升。
```
