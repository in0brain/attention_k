# Sprint 2 历史记录

## Sprint 2A：Hidden State Cache Baseline

已完成内容：

- 实现 `src/recover_attention/token_alignment.py`。
- 实现 single / group `[MASK]` char range 识别。
- 实现 original vs masked 的基础 deterministic 差分对齐。
- 实现 recovered question 模板填充对齐；fragment recovery 只记录 warning，不中断流程。
- 实现 stub tokenizer，输出 `tokens`、`token_ids`、`token_char_ranges`、`seq_len`。
- 实现 `src/recover_attention/hidden_state_cache.py`。
- 实现 `stub_hidden_state_v0`，生成 deterministic 小型 tensor。
- 实现 `scripts/16_cache_hidden_states.py`。
- 生成 Sprint 2A cache 输出目录。
- 新增 token alignment 和 hidden state cache pytest。

输入文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

输出文件：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

新增或修改文件：

- src/recover_attention/token_alignment.py
- src/recover_attention/hidden_state_cache.py
- scripts/16_cache_hidden_states.py
- tests/test_token_alignment.py
- tests/test_hidden_state_cache.py
- PROGRESS.md
- docs/progress/sprint_2_history.md

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_token_alignment.py tests/test_hidden_state_cache.py -q
conda run -n recover_attention python scripts/16_cache_hidden_states.py --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl --output-dir outputs/logs/sprint_2A_hidden_state_cache_baseline --backend stub_hidden_state_v0 --layer-indices 0 1 2 --hidden-size 8 --mask-token "[MASK]" --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

检查结果：

- targeted pytest：16 passed。
- cache command：passed。
- full pytest：459 passed, 2 skipped。
- cache backend：`stub_hidden_state_v0`。
- num_cases：20。
- num_inputs_total：60。
- num_hidden_state_files：60。
- input_type_counts：`{original: 20, masked: 20, recovered: 20}`。
- hidden_size：8。
- layer_indices：`[0, 1, 2]`。
- token alignment：single mask cases 17，group mask cases 3。
- recovered fragment outputs：8。
- alignment_warning_count：8。
- Sprint 1Q / 1R 输入文件 hash 运行前后不变。

遗留问题：

- 当前 hidden states 来自 stub backend，不是真实模型。
- 当前 token alignment 是基础 deterministic 对齐，不处理复杂 paraphrase。
- recovered fragment 输出只记录 warning，后续在 2B / 2C 判断是否保留。
- 真实 HF hidden state backend 未运行；如需运行，必须由用户单独确认。
- 本 sprint 未训练 probe。
- 本 sprint 未做 representation feature extraction。
- 本 sprint 未做 hidden-state distance analysis、PCA / UMAP / t-SNE。
- 本 sprint 未执行 attention guidance。
- 本 sprint 未修改 1Q / 1R 输入文件。

下一步建议：

- Sprint 2B：Representation Feature Extraction。

## Sprint 2B：Representation Feature Extraction

已完成内容：

- 实现 `src/recover_attention/representation_features.py`。
- 实现 `representation_features_v0`。
- 按 `masked_id` 分组逐 case 读取 Sprint 2A-real hidden-state cache。
- 使用 `torch.load(path, map_location="cpu")` 只在 CPU 加载当前 case 需要的 tensor。
- 为每条 input record 生成 input-level summary。
- 为每个 masked case × recovered variant 生成一条 feature record，记录数不写死为 20。
- 提取 global mean pooling 和 last token pooling 的 layer-wise cosine / l2 distance。
- 提取 recovery closure features。
- 提取辅助 span-aware pooling features；span/token overlap 缺失时置为 null 并记录 warning，不中断流程。
- 生成 documentation-only `feature_schema.json`，未接入全局 schema validator。
- 新增 `scripts/17_extract_representation_features.py`。
- 新增 `tests/test_representation_features.py`。

输入文件：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

输出文件：

```text
outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl
outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
outputs/logs/sprint_2B_representation_features/feature_schema.json
```

新增或修改文件：

- src/recover_attention/representation_features.py
- scripts/17_extract_representation_features.py
- tests/test_representation_features.py
- PROGRESS.md
- docs/progress/sprint_2_history.md

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_representation_features.py -q
conda run -n recover_attention python scripts/17_extract_representation_features.py --input-manifest outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl --input-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json --metadata outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json --output-dir outputs/logs/sprint_2B_representation_features --backend representation_features_v0 --eps 1e-8 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

检查结果：

- `tests/test_representation_features.py`：11 passed。
- 2B extraction command：passed。
- backend：`representation_features_v0`。
- output_dir：`outputs/logs/sprint_2B_representation_features`。
- `representation_feature_manifest.jsonl`：20 records。
- `input_representation_summary.jsonl`：60 records。
- `representation_feature_report.json`：`num_masked_groups=20`，`num_recovered_variants=20`，`num_skipped_groups=0`，`num_skipped_recovered_variants=0`。
- source backend：`hf_local_causal_lm_hidden_states_v0`。
- source layer indices：`[0, 8, 16, 24, 27]`。
- span-aware auxiliary features：`span_aware_feature_null_records=8`。
- full pytest：476 passed, 2 skipped。

边界说明：

- 本 sprint 未重跑 HF model forward。
- 本 sprint 未修改 Sprint 1Q / 1R / 2A / 2A-real outputs。
- 本 sprint 未使用 GPU。
- 本 sprint 未构造 probe dataset。
- 本 sprint 未选择 target label。
- 本 sprint 未生成 train/dev/test split。
- 本 sprint 未训练 probe。
- 本 sprint 未执行 attention guidance。
- 本 sprint 未声称 hallucination reduction。

遗留问题：

- 2B features 尚未被整理成 2C probe dataset。
- span-aware features 对 8 条 fragment recovery 相关记录为 nullable，后续 2C 需决定保留策略或标注策略。
- 当前只完成 representation feature extraction，不验证 feature 对 probe 或 guidance 的有效性。

下一步建议：

- Sprint 2C：Probe Dataset Construction。

## Sprint 2B-fix：Representation Feature Extraction Scope Alignment

### Goal

Align Sprint 2B with the minimal Sprint 2 loop.

2B is responsible for:

```text
hidden states → representation features
```

2C is responsible for:

```text
human labels → probe targets / probe dataset
```

### Fixes

- Removed the invalid must-read requirement for `tests/test_hidden_state_cache_hf.py`.
- Confirmed that `tests/test_hidden_state_cache_hf.py` does not exist and its absence is not a failure.
- Recorded that `docs/codex_tasks/sprint_2B_representation_feature_extraction.md` had pre-existing `AM` status.
- Restored the 2B output contract:
  - `representation_features.jsonl`
  - `representation_feature_report.json`
- De-scoped `representation_feature_manifest.jsonl`, `input_representation_summary.jsonl`, and `feature_schema.json` as required outputs.
- Marked legacy/debug files as `deprecated_extra_outputs` in `representation_feature_report.json` when they are present on disk.
- Restored the minimal feature scope:
  - mask position representation
  - span pooled representation
  - question pooled representation
  - original vs masked cosine distance
  - original vs recovered cosine distance
  - masked vs recovered cosine distance
  - layer-wise distance curve

### Inputs

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

### Outputs

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

Deprecated extra outputs still present from the previous 2B run:

```text
outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl
outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl
outputs/logs/sprint_2B_representation_features/feature_schema.json
```

These are not required Sprint 2B outputs and are not 2C input contracts.

### Commands

```bash
git status --short
conda run -n recover_attention python -c "from pathlib import Path; [print(p) for p in sorted(Path('tests').glob('test_*hidden_state*'))]"
conda run -n recover_attention python -m pytest tests/test_representation_features.py -q
conda run -n recover_attention python scripts/17_extract_representation_features.py --input-manifest outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl --input-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json --metadata outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json --output-dir outputs/logs/sprint_2B_representation_features --backend representation_features_minimal_v0 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

### Checks

- hidden-state tests discovered：`tests/test_hidden_state_cache.py`。
- `tests/test_hidden_state_cache_hf.py`：absent and not a failure。
- targeted pytest：12 passed。
- 2B-fix extraction command：passed。
- backend：`representation_features_minimal_v0`。
- `representation_features.jsonl`：20 records。
- `representation_feature_report.json`：`sprint=2B-fix`，`num_masked_groups=20`，`num_recovered_variants=20`，`num_feature_records=20`，`num_skipped_groups=0`，`num_skipped_recovered_variants=0`。
- `position_pool_feature_null_records=8`。
- full pytest：477 passed, 2 skipped。

### Not Done

- No probe dataset.
- No target selection.
- No train/dev/test split.
- No probe training.
- No guidance candidate manifest.
- No attention steering.
- No HF forward.
- No Sprint 1Q / 1R / 2A / 2A-real output modification.

### Next

Sprint 2C：Probe Dataset Construction.

## Sprint 2C：Probe Dataset Construction

### Goal

Construct a probe-ready dataset by mapping Sprint 1Q / 1R human review metadata to probe targets and joining those targets with Sprint 2B representation features.

### Completed

- Implemented `src/recover_attention/probe_dataset.py`.
- Implemented `scripts/18_build_probe_dataset.py`.
- Added `tests/test_probe_dataset.py`.
- Read only the formal Sprint 2B inputs:
  - `outputs/logs/sprint_2B_representation_features/representation_features.jsonl`
  - `outputs/logs/sprint_2B_representation_features/representation_feature_report.json`
- Did not read hidden-state `.pt` tensors.
- Did not read legacy/debug 2B files as inputs:
  - `representation_feature_manifest.jsonl`
  - `input_representation_summary.jsonl`
  - `feature_schema.json`
- Mapped human metadata to probe targets with `probe_dataset_mapping_v0`.
- Preserved null span / mask-position representation features as null with missing indicators.
- Kept one output record per `representation_features.jsonl` record.
- Recorded that `docs/codex_tasks/sprint_2C_probe_dataset_construction.md` had pre-existing `AM` status before this sprint.

### Inputs

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

### Outputs

```text
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

### New Or Modified Files

- src/recover_attention/probe_dataset.py
- scripts/18_build_probe_dataset.py
- tests/test_probe_dataset.py
- PROGRESS.md
- docs/progress/sprint_2_history.md
- outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
- outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json

### Commands

```bash
conda run -n recover_attention python -m pytest tests/test_probe_dataset.py -q
conda run -n recover_attention python scripts/18_build_probe_dataset.py --features outputs/logs/sprint_2B_representation_features/representation_features.jsonl --feature-report outputs/logs/sprint_2B_representation_features/representation_feature_report.json --output-dir outputs/logs/sprint_2C_probe_dataset --backend probe_dataset_mapping_v0 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

### Checks

- Targeted pytest：7 passed.
- Build command：passed.
- backend：`probe_dataset_mapping_v0`.
- output_dir：`outputs/logs/sprint_2C_probe_dataset`.
- `probe_dataset.jsonl`：20 records.
- `probe_dataset_report.json`：`num_probe_records=20`, `num_probe_target_usable=20`, `num_unmapped=0`.
- Target counts：`risk_positive=7`, `positive_anchor=3`, `negative=8`, `hard_negative_or_weak_positive=2`.
- Null feature counts：`records_with_null_position_features=8`.
- Forbidden outputs generated：none.
- Full pytest：484 passed, 2 skipped.

### Not Done

- No train/dev/test split.
- No k-fold split.
- No probe training.
- No probe predictions.
- No probe evaluation report.
- No probe model file.
- No guidance candidate manifest.
- No attention steering.
- No hidden-state tensor read.
- No representation feature re-extraction.
- No Sprint 1Q / 1R / 2A / 2A-real / 2B output modification beyond the new 2C outputs.

### Next

Sprint 2D：Probe Training Baseline.

## Sprint 2D：Probe Training Baseline

### Goal

Train a minimal hidden-state probe baseline from Sprint 2C probe dataset.

### Completed

- Implemented `src/recover_attention/probe_training.py`.
- Implemented `scripts/19_train_probe_baseline.py`.
- Added `tests/test_probe_training.py`.
- Read only the formal Sprint 2C inputs:
  - `outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl`
  - `outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json`
- Flattened `feature_values` scalar features and `feature_arrays` layer-wise features.
- Converted null numeric features to `0.0` with `__is_missing` indicator features.
- Used train-fold z-score standardization for cross-validation.
- Trained a deterministic numpy-based one-vs-rest ridge classifier because `scikit-learn` is not installed in the current environment.
- Ran leave-one-out cross-validation over 20 usable records.
- Saved the final model bundle trained on all usable records.

### Inputs

```text
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

### Outputs

```text
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json
```

`probe_feature_index.json` is an optional debug output and is not a 2E input contract.

### New Or Modified Files

- src/recover_attention/probe_training.py
- scripts/19_train_probe_baseline.py
- tests/test_probe_training.py
- PROGRESS.md
- docs/progress/sprint_2_history.md
- outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
- outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
- outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
- outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json

### Training Setup

- backend: `probe_training_baseline_v0`
- model: `ridge_classifier_ovr_v0`
- cross validation: `leave_one_out`
- target: `probe_target`
- null feature strategy: zero impute + missing indicators
- scaling: train-fold z-score
- seed: 42
- num_base_features: 99
- num_features_with_missing_indicators: 198

### Commands

```bash
conda run -n recover_attention python -m pytest tests/test_probe_training.py -q
conda run -n recover_attention python scripts/19_train_probe_baseline.py --dataset outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl --dataset-report outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json --output-dir outputs/logs/sprint_2D_probe_training_baseline --backend probe_training_baseline_v0 --model ridge_classifier_ovr_v0 --cv leave_one_out --seed 42 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

### Checks

- targeted pytest: 9 passed.
- training command: passed.
- full pytest: 493 passed, 2 skipped.
- num_predictions: 20.
- status: `ok`.
- accuracy: 0.85.
- macro_f1: 0.680952380952381.
- weighted_f1: 0.8285714285714285.
- binary_anchor_or_risk_accuracy: 0.9.
- binary_anchor_or_risk_macro_f1: 0.898989898989899.
- majority baseline label: `negative`.
- majority baseline accuracy: 0.4.
- majority baseline macro_f1: 0.14285714285714288.
- `probe_model.pkl`: saved.

### Notes

- Scores are diagnostic only because the dataset has only 20 human-reviewed records.
- Feature and layer signal summaries are not causal claims.
- No hidden-state tensors were read.
- No Sprint 2B representation features were read as training input.
- No probe dataset was rebuilt.
- No guidance candidate manifest was generated.
- No attention steering was performed.
- No hallucination reduction claim was made.

### Next

Sprint 2E：Guidance Candidate Manifest Dry Run.

## Sprint 2E：Guidance Candidate Manifest Dry Run

### Goal

Convert Sprint 2D probe predictions into planned-only guidance candidate records.

### Completed

- Implemented `src/recover_attention/guidance_candidates.py`.
- Implemented `scripts/20_build_guidance_candidate_manifest.py`.
- Added `tests/test_guidance_candidates.py`.
- Read only the formal Sprint 2D inputs:
  - `outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl`
  - `outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json`
- Did not load `probe_model.pkl`; it was not a 2E input.
- Produced one guidance candidate record per probe prediction, including negative `no_guidance` records.
- Assigned candidate action from `predicted_probe_target`, not from gold labels.
- Added confidence diagnostics from `decision_scores` margin.
- Recorded that `docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md` had pre-existing `AM` status before this sprint.

### Inputs

```text
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
```

### Outputs

```text
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

### New Or Modified Files

- src/recover_attention/guidance_candidates.py
- scripts/20_build_guidance_candidate_manifest.py
- tests/test_guidance_candidates.py
- PROGRESS.md
- docs/progress/sprint_2_history.md
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json

### Candidate Actions

- `risk_positive` -> `increase_attention_to_original_span`
- `positive_anchor` -> `preserve_original_span_attention`
- `hard_negative_or_weak_positive` -> `review_before_guidance`
- `negative` -> `no_guidance`

### Commands

```bash
conda run -n recover_attention python -m pytest tests/test_guidance_candidates.py -q
conda run -n recover_attention python scripts/20_build_guidance_candidate_manifest.py --predictions outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl --eval-report outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json --output-dir outputs/logs/sprint_2E_guidance_candidate_dry_run --backend guidance_candidate_dry_run_v0 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

### Checks

- targeted pytest: 8 passed.
- dry run command: passed.
- full pytest: 501 passed, 2 skipped.
- backend: `guidance_candidate_dry_run_v0`.
- output_dir: `outputs/logs/sprint_2E_guidance_candidate_dry_run`.
- `guidance_candidate_manifest.jsonl`: 20 records.
- `guidance_candidate_report.json`: status `ok`.
- `num_guidance_candidate_records`: 20.
- `guidance_candidate_true`: 13.
- `guidance_candidate_false`: 7.
- predicted target counts: `risk_positive=8`, `positive_anchor=4`, `negative=7`, `hard_negative_or_weak_positive=1`.
- candidate action counts: `increase_attention_to_original_span=8`, `preserve_original_span_attention=4`, `review_before_guidance=1`, `no_guidance=7`.
- confidence counts: `high=17`, `medium=3`, `low=0`, `unknown=0`.
- prediction correctness diagnostics: `correct=17`, `incorrect=3`.

### Notes

- Dry run only: all records have `execution_status=planned_only`, `dry_run=true`, `will_modify_attention=false`, `will_run_model=false`, and `will_change_answer=false`.
- No probe model was loaded.
- No probe was retrained.
- No probe dataset was rebuilt.
- No hidden-state tensors were read.
- No representation features were recomputed.
- No tokenizer, HF model, Ollama, NLI model, or recovery model was called.
- No attention weights were modified.
- No attention steering, CoT generation, answer accuracy evaluation, or hallucination reduction claim was made.

### Next

Sprint 2F：Mini Closed-loop Report.

## Sprint 2F：Mini Closed-loop Report

### Goal

Summarize whether Sprint 2 formed a minimal hidden-state-to-guidance-candidate dry-run loop.

### Completed

- Implemented `src/recover_attention/closed_loop_report.py`.
- Implemented `scripts/21_write_sprint_2_closed_loop_report.py`.
- Added `tests/test_closed_loop_report.py`.
- Read only formal Sprint 2A-real / 2B / 2C / 2D / 2E reports and manifest-style outputs.
- Generated the Sprint 2 minimal closed-loop Markdown report.
- Generated an auxiliary audit JSON for boundary checks.
- Did not read hidden-state `.pt` tensors.
- Did not load `probe_model.pkl`; only file existence was checked.
- Did not rerun upstream pipeline scripts 16 / 17 / 18 / 19 / 20.
- Preserved pre-existing `AM` task card state for Sprint 2E / 2F task cards.
- Preserved pre-existing untracked Sprint 2E implementation files.

### Inputs

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

### Outputs

```text
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
```

### New Or Modified Files

- src/recover_attention/closed_loop_report.py
- scripts/21_write_sprint_2_closed_loop_report.py
- tests/test_closed_loop_report.py
- PROGRESS.md
- docs/progress/sprint_2_history.md
- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
- outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json

### Core Conclusion

Sprint 2 formed a dry-run loop:

```text
hidden-state cache -> representation features -> probe dataset -> probe training -> guidance candidate manifest
```

This is not an executed attention-steering loop.

### Six Question Summary

- Hidden states can be stably cached for the current 20 reviewed cases: 20 cases, 60 inputs, 60 hidden-state files, backend `hf_local_causal_lm_hidden_states_v0`.
- Token/span/mask alignment supports the minimal loop, with 17 single-mask cases, 3 group-mask cases, 8 fragment recovery outputs, and 8 alignment warnings.
- Representation features show diagnostic signals in the current 20-record sample, but the sample is too small to claim stable generalizable separation.
- The probe produced diagnostic signal above majority baseline: accuracy 0.85, macro_f1 0.680952380952381, majority baseline accuracy 0.4.
- Probe predictions can be mapped into planned-only guidance candidates: 20 records, 13 candidate=true, 7 candidate=false.
- Sprint 3 must address dry-run boundary, evaluation boundary, data scale, token/span alignment, probe robustness, guidance design, and Windows serial execution.

### Non-claims

- No attention guidance was executed.
- No transformer attention weights were modified.
- No attention mask was injected.
- No CoT generation was rerun under guidance.
- No answer accuracy improvement was evaluated.
- No hallucination reduction was validated.
- No closed-loop intervention was validated.

### Engineering Stability Note

During Sprint 2E, one Windows temporary file lock occurred when two `conda run` commands were launched in parallel.

Serial rerun passed.

This is recorded as an engineering stability issue, not a pipeline logic failure, model failure, or test design failure.

Future runs should execute targeted pytest, pipeline command, and full pytest serially.

### Workspace State Note

- Pre-existing AM task card state was observed for:
  - `docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md`
  - `docs/codex_tasks/sprint_2F_mini_closed_loop_report.md`
- The task cards were not rewritten.
- Pre-existing untracked Sprint 2E files were preserved:
  - `src/recover_attention/guidance_candidates.py`
  - `scripts/20_build_guidance_candidate_manifest.py`
  - `tests/test_guidance_candidates.py`
- No upstream pipeline scripts were rerun.
- Sprint 2F only generated closed-loop report artifacts.

### Commands

```bash
git status --short
conda run -n recover_attention python -m pytest tests/test_closed_loop_report.py -q
conda run -n recover_attention python scripts/21_write_sprint_2_closed_loop_report.py --output-dir outputs/logs/sprint_2F_mini_closed_loop_report --backend sprint_2_closed_loop_report_v0 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

### Checks

- targeted pytest: 7 passed.
- report generation command: passed.
- full pytest: 508 passed, 2 skipped.
- audit status: `ok`.
- audit loop status: hidden_state_cache=true, representation_features=true, probe_dataset=true, probe_training=true, guidance_candidate_dry_run=true.
- audit boundary: executed_attention_steering=false, validated_hallucination_reduction=false.
- required boundary statements: present.
- Windows serial execution note: present.

### Next

Sprint 3A：Attention Steering Interface Design.

## Sprint 2 Final Checkpoint and Visualization Summary

### Goal

Produce a final Sprint 2 checkpoint package with full pytest confirmation, stage summary Markdown, audit JSON, and visualization figures.

### Completed

- Implemented `src/recover_attention/stage_summary.py`.
- Implemented `scripts/22_write_sprint_2_stage_summary.py`.
- Added `tests/test_stage_summary.py`.
- Read only formal Sprint 2A-real / 2B / 2C / 2D / 2E / 2F artifacts.
- Generated the Sprint 2 stage summary Markdown report.
- Generated the Sprint 2 stage summary audit JSON.
- Generated six required PNG visualization figures.
- Recorded full pytest result and PowerShell measured duration.
- Preserved pre-existing `AM` task card state for `docs/codex_tasks/sprint_2_final_checkpoint_visualization_summary.md`.
- Did not rerun upstream pipeline scripts 16 / 17 / 18 / 19 / 20 / 21.
- Did not read hidden-state `.pt` tensors.
- Did not load `probe_model.pkl`; only file existence was checked.

### Inputs

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md
outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json
```

### Outputs

```text
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary.md
outputs/logs/sprint_2_stage_summary/sprint_2_stage_summary_audit.json
outputs/logs/sprint_2_stage_summary/figures/sprint_2_pipeline_overview.png
outputs/logs/sprint_2_stage_summary/figures/probe_target_counts.png
outputs/logs/sprint_2_stage_summary/figures/probe_metrics_vs_baseline.png
outputs/logs/sprint_2_stage_summary/figures/guidance_candidate_action_counts.png
outputs/logs/sprint_2_stage_summary/figures/guidance_confidence_counts.png
outputs/logs/sprint_2_stage_summary/figures/sprint_2_boundary_summary.png
```

### Commands

```bash
git status --short
conda run -n recover_attention python -m pytest tests/test_stage_summary.py -q
conda run -n recover_attention python scripts/22_write_sprint_2_stage_summary.py --output-dir outputs/logs/sprint_2_stage_summary --backend sprint_2_stage_summary_v0 --overwrite
conda run -n recover_attention python -m pytest -q
conda run -n recover_attention python scripts/22_write_sprint_2_stage_summary.py --output-dir outputs/logs/sprint_2_stage_summary --backend sprint_2_stage_summary_v0 --full-pytest-passed 515 --full-pytest-skipped 2 --full-pytest-duration-seconds 10.4918917 --overwrite
git diff --name-only
git status --short
```

Full pytest timing command:

```powershell
Measure-Command { conda run -n recover_attention python -m pytest -q }
```

### Checks

- targeted pytest: 7 passed.
- stage summary generation command: passed.
- full pytest: 515 passed, 2 skipped.
- PowerShell measured duration: 10.4918917 seconds.
- audit status: `ok`.
- audit backend: `sprint_2_stage_summary_v0`.
- audit loop status: hidden_state_cache=true, representation_features=true, probe_dataset=true, probe_training=true, guidance_candidate_dry_run=true, closed_loop_report=true.
- audit boundary: executed_attention_steering=false, validated_answer_accuracy_improvement=false, validated_hallucination_reduction=false.
- figures generated: six required PNG files.
- output_dir: `outputs/logs/sprint_2_stage_summary`.

### Notes

- This checkpoint does not rerun upstream pipeline scripts.
- This checkpoint does not train a probe.
- This checkpoint does not perform attention steering.
- This checkpoint does not validate answer accuracy improvement.
- This checkpoint does not validate hallucination reduction.
- This checkpoint supports Sprint 3A planning, but it is not evidence that attention steering is effective.
- Windows serial execution requirement remains in effect.

### Next

Sprint 3A：Attention Steering Interface Design.

## Sprint 2A-real：Real Hidden State Cache Run

已完成内容：

- 实现或启用 `hf_local_causal_lm_hidden_states_v0` 的本地 HF causal LM backend 支持。
- `scripts/16_cache_hidden_states.py` 支持真实 backend 参数：
  - `--model-path`
  - `--device`
  - `--device-map`
  - `--dtype`
  - `--max-length`
  - `--batch-size`
  - `--load-in-4bit`
  - `--bnb-4bit-compute-dtype`
  - `--trust-remote-code`
- 真实 backend 加载边界已实现：
  - 必须显式传入本地 `--model-path`。
  - `local_files_only=True`。
  - 不自动下载模型。
  - 不联网。
  - `trust_remote_code` 默认 false。
  - `output_hidden_states=True`。
  - `use_cache=False`。
- 实现 4bit run 的 bitsandbytes 缺失检测。
- 实现 layer index 越界过滤与 warning。
- 新增/更新测试覆盖真实 backend 参数解析、本地模型路径缺失、`local_files_only=True`、4bit 参数解析、bitsandbytes 缺失阻断和 layer index 过滤。
- 使用本地 HF causal LM backend 完成 Sprint 1R manifest 中 20 条 reviewed cases 的真实 hidden-state cache。
- 输出 `hidden_state_manifest.jsonl`、`hidden_state_cache_report.json`、`token_alignment_report.json` 和 `real_run_metadata.json`。
- 保持 Sprint 1Q / 1R / 2A stub 产物只读。

输入文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

预期输出文件：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

实际输出状态：passed。

output_dir：

```text
outputs/logs/sprint_2A_real_hidden_state_cache
```

新增或修改文件：

- src/recover_attention/hidden_state_cache.py
- scripts/16_cache_hidden_states.py
- tests/test_hidden_state_cache.py
- PROGRESS.md
- docs/progress/sprint_2_history.md

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_hidden_state_cache.py -q
conda run -n recover_attention python -m pytest -q
conda run -n recover_attention python scripts/16_cache_hidden_states.py --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl --output-dir outputs/logs/sprint_2A_real_hidden_state_cache --backend hf_local_causal_lm_hidden_states_v0 --model-path D:/models/Qwen2.5-7B-Instruct --layer-indices 0 8 16 24 27 --max-length 512 --batch-size 1 --device-map auto --load-in-4bit --bnb-4bit-compute-dtype float16 --mask-token "[MASK]" --overwrite
git diff --name-only
git status --short
```

检查结果：

- `tests/test_hidden_state_cache.py`：14 passed。
- 全量 pytest：465 passed, 2 skipped。
- 本地模型路径检查：`D:/models/Qwen2.5-7B-Instruct` 存在。
- Sprint 2A stub 输出检查：`hidden_state_manifest.jsonl`、`hidden_state_cache_report.json`、`token_alignment_report.json` 均存在。
- 2A-real real hidden-state cache run：passed。
- backend：`hf_local_causal_lm_hidden_states_v0`。
- output_dir：`outputs/logs/sprint_2A_real_hidden_state_cache`。
- `hidden_state_manifest.jsonl`：60 records。
- `hidden_state_cache_report.json`：`num_cases=20`，`num_inputs_total=60`，`num_hidden_state_files=60`。
- `real_run_metadata.json`：`backend=hf_local_causal_lm_hidden_states_v0`，`num_cases=20`，`num_inputs_total=60`。
- 未修改 Sprint 1Q / 1R / 2A stub 输出。

遗留问题：

- 当前只完成真实 hidden states 缓存，尚未抽取 representation features。
- 当前不分析 hidden-state distance。
- 当前不训练 probe。
- 当前不执行 attention guidance。
- 后续 Sprint 2B 读取真实 cache 进行 representation feature extraction。

下一步建议：

- Sprint 2B：Representation Feature Extraction。

---

## Sprint 2G 前置：Dataset Source Audit and Import Preparation

### Goal

在执行 Sprint 2G full-scale weak-labeled run 之前，先补充并审计数据源，
使后续 500 / 2000 条 weak-labeled k-fold 实验能真正基于同一个 full-scale 数据源构造，
而不是把 92 行 fan-out 或 20 条 human-reviewed labels 当成主监督数据。

本轮未执行 hidden-state cache、representation feature extraction、probe training、
guidance candidate generation，也未重跑 scripts 16/17/18/19/20，未覆盖 20-case Sprint 2 outputs。

### 新增文件

- `src/recover_attention/dataset_audit.py`：数据源审计逻辑（文件分类、fan-out 检测、scale check、k-fold 前置条件）。
- `src/recover_attention/dataset_import.py`：reasoning dataset 导入 / 标准化（GSM8K，本地文件或真实下载，一对一标准化，不复制不增强）。
- `scripts/23a_audit_dataset_sources.py`：数据源审计 CLI（编号避开 Sprint 2G 已占用的 23/24/25/26）。
- `scripts/23b_import_reasoning_dataset.py`：数据集导入 / 标准化 CLI。
- `tests/test_dataset_audit.py`、`tests/test_dataset_import.py`：targeted pytest（共 18 条）。

### 输入

- 本地扫描根：`data`、`outputs/logs`。
- GSM8K train split 原始来源：`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl`（真实公开数据，7473 条）。

### 输出

- `data/raw/gsm8k_train_normalized.jsonl`：标准化后的 GSM8K train，7473 条。
- `outputs/logs/sprint_2G_dataset_prep/dataset_source_audit.json`
- `outputs/logs/sprint_2G_dataset_prep/dataset_source_audit.md`
- `outputs/logs/sprint_2G_dataset_prep/normalized_dataset_report.json`
- `outputs/logs/sprint_2G_dataset_prep/normalized_dataset_preview.jsonl`（前 50 条预览）

每条标准化记录字段：`question_id` / `source_dataset` / `source_split` / `question` / `answer` / `metadata`
（`metadata` 含 `original_id` / `original_answer`（完整 chain-of-thought）/ `normalization_backend`）。
GSM8K 答案抽取规则：以 `####` 分割，最终答案存入 `answer`，完整推理过程保留在 `metadata.original_answer`。

### 运行命令

```bash
conda run -n recover_attention python scripts/23b_import_reasoning_dataset.py \
  --dataset gsm8k --split train \
  --output data/raw/gsm8k_train_normalized.jsonl \
  --report-output outputs/logs/sprint_2G_dataset_prep/normalized_dataset_report.json \
  --preview-output outputs/logs/sprint_2G_dataset_prep/normalized_dataset_preview.jsonl \
  --preview-limit 50 --backend gsm8k_normalize_v0 --overwrite

conda run -n recover_attention python scripts/23a_audit_dataset_sources.py \
  --search-roots data outputs/logs \
  --output-dir outputs/logs/sprint_2G_dataset_prep \
  --backend dataset_source_audit_v0 --overwrite

conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_dataset_import.py -q
conda run -n recover_attention python -m pytest -q
```

注意：本机 `conda` 未在 PATH 上，实际以 `recover_attention` env 的解释器
`D:\conda\Miniconda3\envs\recover_attention\python.exe` 串行执行（等价于上述 conda run）。

### 检查结果

- targeted pytest：18 passed。
- full pytest：533 passed, 2 skipped。
- 审计扫描 63 个候选数据文件。
- 之前最大可见 JSONL ≈ 92 行（`*ablated_questions.jsonl` / `*nli_scores.jsonl` / `*semantic_labels.jsonl`），
  distinct source id 均为 5：这些是 5 条原始 question 的 candidate span / ablation unit / NLI fan-out，不是 92 条独立 question。
- 导入前仓库唯一的 raw question source 是 5 条玩具样本。
- 导入后 `available_num_cases = 7473`，`can_run_500 = true`，`can_run_2000 = true`，`can_run_all = true`。
- 推荐 source path：`data/raw/gsm8k_train_normalized.jsonl`。

### Boundary

- 本轮只做 dataset source audit + dataset import / normalization + scale feasibility + k-fold 前置条件。
- 未执行 hidden-state cache / representation features / probe training / guidance candidates。
- 未执行 Sprint 2G 正式 full-scale pipeline，未重跑 16/17/18/19/20，未覆盖 20-case Sprint 2 outputs。
- 未复制 / 重复采样 / 数据增强凑数；未把 20 条 human-reviewed labels 当成 full-scale 主监督数据。
- k-fold fold 数现在不可判定：必须等 weak probe labels 生成后按最小类别数决定
  （5-fold → 3-fold → 2-fold → leave-one-out / holdout），仅当最小类别数 >= 5 才允许 5-fold。

### Next

- source records >= 2000，可执行 Sprint 2G：优先 2000-case scale run，或先 500-case weak-labeled dry run。
- Sprint 2G 必须以 `data/raw/gsm8k_train_normalized.jsonl` 为单一 full-scale 数据源构造 weak-labeled dataset 与 k-fold。

---

## Sprint 2G-2000：Full-scale Weak-labeled 2000-case Dry Run

### Goal

Scale the Sprint 2 dry-run diagnostic loop from 20 human-reviewed cases to 2000
weak-labeled real GSM8K cases, using `data/raw/gsm8k_train_normalized.jsonl` as
the single full-scale source.

This is a weak-labeled 2000-case dry run. It does not execute attention steering.
It does not validate hallucination reduction. It does not validate answer accuracy
improvement. It is not a human-reviewed 2000-case validation.

### Source path / cases

- source path: `data/raw/gsm8k_train_normalized.jsonl`
- requested_num_cases = 2000, available_num_cases = 7473, actual_num_cases = 2000
- sampling_rule = seeded_sample, seed = 42 (no duplication, no augmentation)
- source_dataset = gsm8k, source_split = train

### New files

- `src/recover_attention/full_scale_manifest.py`, `scripts/24_build_full_scale_manifest.py`, `tests/test_full_scale_manifest.py`
- `src/recover_attention/full_scale_weak_labels.py`, `scripts/25_build_full_scale_weak_labels.py`, `tests/test_full_scale_weak_labels.py`
- `src/recover_attention/weak_probe_dataset.py`, `scripts/26_build_full_scale_weak_probe_dataset.py`, `tests/test_weak_probe_dataset.py`
- `src/recover_attention/full_scale_summary.py`, `scripts/27_write_full_scale_2000_summary.py`, `tests/test_full_scale_summary.py`
- Reused scripts 16 / 17 / 19 / 20 with 2G output paths (no modification).

### Output root

`outputs/logs/sprint_2G_full_scale_2000/` (00_manifest, 01_downstream, 02_hidden_state_cache,
03_representation_features, 04_weak_probe_dataset, 05_probe_training, 06_guidance_candidates,
07_stage_summary/figures). Smoke run output: `outputs/logs/sprint_2G_full_scale_2000_smoke/`.

### Phase results

1. Manifest: 2000 cases (fs2000_000001..fs2000_002000).
2. Weak labels (weak_label_mapping_v0, label_source=weak_auto, human_reviewed=false):
   positive_anchor 1039 / negative 564 / hard_negative_or_weak_positive 323 / risk_positive 74;
   num_unmapped=0; full_scale_2a_manifest.jsonl = 2000 cases. human_* fields are sentinel
   placeholders ("weak_auto_not_human_reviewed"), NOT human labels.
3. Hidden-state cache: real `hf_local_causal_lm_hidden_states_v0` with local
   `D:/models/Qwen2.5-7B-Instruct` (4-bit, layers 0/8/16/24/27, max_length 512, device_map auto).
   num_cases=2000, num_inputs_total=6000, num_hidden_state_files=6000, failure_count=0,
   alignment all ok (alignment_warning_count=0). Smoke (4 cases / 12 inputs) passed first.
   The cache module has no `--resume`; the full run completed in one pass.
4. Representation features: num_feature_records=2000, num_skipped_groups=0.
5. Weak probe dataset: num_probe_records=2000, num_usable_records=2000.
6. Adaptive k-fold: min usable class count = 74 (>= 5) -> stratified_k_fold, num_folds=5.
   Decided AFTER weak probe labels were generated (ladder 5 -> 3 -> 2 -> leave-one-out).
7. Probe training (weak-labeled diagnostic metrics, not human-supervised validation):
   accuracy=0.9175, macro_f1=0.785, weighted_f1=0.909; majority baseline accuracy=0.5195,
   macro_f1=0.171.
8. Guidance candidate dry run (planned-only): candidate_true=1361; actions
   preserve_original_span_attention 1073 / no_guidance 639 / review_before_guidance 265 /
   increase_attention_to_original_span 23; confidence high 1656 / medium 267 / low 77.
   All records dry_run=true, will_modify_attention=false, will_run_model=false, will_change_answer=false.
9. Summary/figures: full_scale_2000_summary.md, full_scale_2000_audit.json, 7 PNGs (PIL fallback).

### 20-case vs 2000-case

- 20-case (human-reviewed, leave-one-out) accuracy=0.85, macro_f1=0.681.
- 2000-case (weak-labeled, 5-fold) accuracy=0.9175, macro_f1=0.785.
- Not directly equivalent; the 2000-case weak-labeled metrics are NOT more reliable than
  the 20-case human-reviewed baseline. The 2000-case weak target is derived from the chosen
  span type, so high probe accuracy reflects recoverability of the weak rule, not validation.

### Runtime / failure / skipped

- failure_count=0, skipped groups=0, skipped recovered variants=0.
- All commands serial (no parallel conda/python). Targeted pytest + full pytest passed
  (547 passed, 2 skipped).

### Boundary

Weak-labeled dry run only. No attention steering executed; no transformer attention weights
modified; no CoT re-generation; answer accuracy improvement not validated; hallucination
reduction not validated; 20-case human-reviewed Sprint 2 outputs not overwritten.

### Next

- Option A: all-train-split (7473) weak-labeled run.
- Option B: Sprint 3A (Attention Steering Interface Design).

---

## Sprint 2G-2000 Review Gate and Final Stage Summary

### Result

Engineering scale-up passed.

Research signal gate did not pass.

This was a read-only review and stage summary: no pipeline phase was re-run, no
all-train, no attention steering, and no 2A-2F or existing 2G-2000 phase outputs
were overwritten. New outputs: `08_review_gate/result_review_gate.{json,md}` and
`09_final_stage_summary/sprint_2G_2000_final_stage_summary.{md,json}`.

### Key Findings

- 2000 GSM8K cases processed
- 6000 real Qwen hidden-state inputs cached
- failure_count = 0
- alignment_warning_count = 0
- stratified 5-fold was valid (min_class_count = 74 >= 5)
- probe accuracy = 0.9175
- macro_f1 = 0.785
- risk_positive recall = 0.311
- risk_positive F1 = 0.474
- risk_positive predicted/increase_attention_to_original_span = 23 / 74 gold (51 missed)
- main confusions: risk_positive -> hard_negative_or_weak_positive (28), risk_positive -> positive_anchor (21)

### Interpretation

High overall accuracy is dominated by positive_anchor and negative classes (~80% of data).

risk_positive is under-recalled; precision 1.0 only reflects an overly conservative model.

The current weak label / recovered filler design introduces structural leakage because
recovered fillers are type-dependent and recovered_cosine features can encode filler
identity rather than independent recovery difficulty. (No direct metadata leakage: the
feature matrix is 99 cosine features, 0 non-cosine metadata features.)

### Decision

Do not run all-train split now.

Do not start Sprint 3A implementation now.

Proceed to Sprint 2H: Weak Label and Recovery Decoupling Fix.

### Rerun gate (before Sprint 3A implementation)

risk_positive recall materially improved over 0.311; risk_positive F1 improved; macro_f1
stable; confusion matrix no longer routes most risk_positive into hard_negative /
positive_anchor; no recovered-filler leakage dominating top features. Absolute thresholds
to be confirmed after rerun (not hardcoded to 0.7/0.8 now).

## Sprint 2H：risk_positive 低召回只读审计

只读审计，未重跑任何 2A-2G 阶段；仅新增审计脚本与审计输出目录。

已完成内容：

- 实现 `scripts/sprint_2H_audit_risk_positive.py`，按 `masked_id` 1:1 join 2G 的 per-sample 预测（`05_probe_training/probe_predictions.jsonl`）、weak probe dataset、weak labels、2A manifest，无 join 错误。
- 复现 2G review gate 核心数字，完全一致：risk_positive support=74，hit=23，missed=51，recall=0.3108；混淆行 hard_negative 28 / positive_anchor 21 / negative 2 / risk_positive 23。
- 分桶审计 risk_positive 漏检原因，回答任务卡问题 A-F。

核心结论：

- 2000 规模 weak label 是 `chosen_span_type` 的确定性函数；gold risk_positive = negation(11) + comparison(63) = 74，`span_type=number` 全部（738 条）被规则强制映射为 positive_anchor，因此 gold risk_positive 中数字类为 0 条。用数字正则扫 74 条 span_text 命中 0 条。
- 「数字类召回更低」假设在本轮既无法成立也无法证伪：数字风险不是被漏检，而是被 label mapping 从 risk 定义中删除。span_type 拆分：comparison recall 0.349（22/63），negation recall 0.091（1/11）。
- Question B/C（wrong_numeric_recovery / misleading_entity_or_unit / critical_number）无法回答：2000 规模 `human_error_type` 全为 sentinel `weak_auto`，`human_semantic_role` 为 `weak_auto_span_type_*`，非人工标注。critical number 被归入 positive_anchor 是规则行为（`span_type_number_critical_value`），非模型错分。
- missed 去向有 span_type 分工：comparison→hard_negative（27/41，模型不敢做 risk 决策）；negation→positive_anchor（9/10，风险 span 被当稳定 anchor，最危险方向）。
- 51 条 missed 中 risk_positive score 排第 2 的有 29 条（56.9%），margin 中位数 −0.30；有可捞回空间，但必须在去泄漏特征上做 calibration 才有意义。
- filler leakage 实锤：recovered filler 是 span_type 的纯函数（number→several、comparison→compared、negation→indeed、object→items、operation→changes、condition→when、question_target→what number），而 label 又是 span_type 纯函数，filler 身份完全编码 label。

输出文件：

```text
outputs/logs/sprint_2H_risk_positive_audit/risk_positive_cases.jsonl（74）
outputs/logs/sprint_2H_risk_positive_audit/risk_positive_hit_cases.jsonl（23）
outputs/logs/sprint_2H_risk_positive_audit/risk_positive_missed_cases.jsonl（51）
outputs/logs/sprint_2H_risk_positive_audit/risk_positive_audit_report.json
outputs/logs/sprint_2H_risk_positive_audit/risk_positive_audit.md
```

新增文件：

- scripts/sprint_2H_audit_risk_positive.py
- outputs/logs/sprint_2H_risk_positive_audit/*

## Sprint 2H-B：Instance-Level Signal Construction

目标：不再扩大 `span_type -> label` 映射表，而是构造 instance-level supervision，让 probe 学习每个具体 span 在具体问题中的 `fragility_bucket` / `risk_strength`。核心边界：主 fragility probe 禁止使用任何 recovered-channel 特征（label 由 recovery 行为派生，用 recovered 特征即读取标签，等价于 2G filler leakage 换皮）。

已完成内容：

- 实现 5 个新模块：`solution_path_numbers.py`（GSM8K gold 解题路径数字解析 + on/off/ambiguous 标注）、`model_recovery.py`（去泄漏 prompt 的模型生成 recovery，替换模板 filler）、`recovery_drift.py`（per-sample drift 分类 + worst-case 聚合）、`risk_strength_targets.py`（fragility_bucket / risk_strength 构造 + 序关系与非 span_type-确定性校验）、`fragility_probe_training.py`（leakage-safe 探针 + 4 组特征 baseline + 序数/排序指标，numpy-only，因环境无 sklearn）。
- 实现 staged 可续跑脚本 `scripts/sprint_2H_instance_signal.py`（select / recovery / drift / targets / probe / gate / analyze / all）。
- 500-case 子集从 2G-2000 池按 seed 42 抽样，复用 2G 已缓存的 original/masked hidden-state 特征；本轮零新增 hidden-state 计算（不缓存 recovered question hidden states）。

关键数据（500 条，真实 run）：

- 解题路径数字普查（500 题）：number span 总计 on_path=1250 / off_path=220（distractor）/ ambiguous=226；implicit path numbers=240；含 distractor 的题=118。percent / `$` / 逗号归一化验证通过（如 on=[2000,20%] off=[25%]）。
- 模型 recovery：qwen3.5:9b，K=3，temperature=0.8，去 span-type 泄漏 prompt（prompt_version=leakage_guarded_recovery_v1），逐样本 seed 记录；500 masked_ids × 3 = 1500 samples。drift 分布：exact 159 / wrong_numeric 117 / generic 114 / ambiguous 60 / object_drift 27 / unit_drift 9 / direction_drift 6 / condition_drift 3 / unrecoverable 5。
- fragility 数据集：trainable 477，排除 ambiguous 23；bucket 分布 0=23 / 1=109 / 2=153 / 3=192；`is_span_type_deterministic=False`（number 落入全部 4 个 bucket：23/5/21/116）；ordering violations=0。
- 探针（macro_f1 / balanced_acc / spearman / bucket3v1_AUC）：
  - hidden_no_recovered（唯一 gate-eligible）：0.318 / 0.355 / 0.256 / 0.690
  - span_type_only baseline：0.339 / 0.402 / 0.374 / 0.858
  - surface_rule baseline：0.378 / 0.406 / 0.387 / 0.849
  - hidden_with_recovered（仅泄漏诊断）：0.371 / 0.388 / 0.279 / 0.725
  - bootstrap（macro_f1 delta）：no_recovered vs span_type_only = −0.021（CI95 −0.051..+0.009，不显著）；vs surface_rule = −0.060（CI95 −0.100..−0.022，显著更差）。
  - per-question top-k(k=1) 覆盖=1.0（因每题仅 1 个 chosen span，覆盖平凡为 1，弱信号）；off-path distractor budget share=0.050。
- review gate：7 项通过 6 项；唯一失败为第 5 项（hidden(no-recovered) 需优于两个 baseline）。`ready_for_2000_rerun=False`。recovered-channel 泄漏诊断：with_recovered − no_recovered = +0.053（弱泄漏，低于 0.1 强泄漏阈值）。

核心结论：

- Instance-level 目标构造在结构上成功：fragility_bucket 不再是 span_type 的纯函数（数字类横跨 4 桶）、ambiguous 被排除、序关系成立、distractor 预算占比仅 5%、解题路径数字有 on/off 拆分。模型生成 recovery 消除了模板 filler leakage（drift 类型多样，不再由 span_type 决定 filler）。
- 但唯一 leakage-free 的表征特征（`*_original_masked_cosine_*`）无法击败 span_type / surface baseline，且在每个指标上都更差；bootstrap 确认对 surface_rule 显著更差。因此本轮不能主张 hidden states 已携带 instance-level fragility 信号：可用的去泄漏特征过薄（仅 original-vs-masked cosine），标签在多数类层面仍与 span_type 强相关（number→多为 bucket 3）。
- Gate 正确判定：不进入 2000-case rerun。下一步优先构造更丰富的 pre-recovery 表征特征（如 span 处 per-layer hidden states、attention entropy），而非扩大规模或调阈值。

输出文件（均在 `outputs/logs/sprint_2H_instance_signal_500/`）：

```text
solution_path_number_report.json / solution_path_number_cases.jsonl / subset_cases.jsonl
model_recovery_outputs.jsonl / model_recovery_report.json
recovery_drift_cases.jsonl / recovery_drift_report.json
risk_strength_dataset.jsonl / risk_strength_report.json
fragility_probe_eval_report.json / fragility_probe_predictions.jsonl
review_gate_instance_signal.json / review_gate_instance_signal.md
```

新增文件：

- src/recover_attention/solution_path_numbers.py
- src/recover_attention/model_recovery.py
- src/recover_attention/recovery_drift.py
- src/recover_attention/risk_strength_targets.py
- src/recover_attention/fragility_probe_training.py
- scripts/sprint_2H_instance_signal.py
- tests/test_sprint_2h_instance_signal.py
- outputs/logs/sprint_2H_instance_signal_500/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_sprint_2h_instance_signal.py -q
conda run -n recover_attention python scripts/sprint_2H_instance_signal.py --stage select
conda run -n recover_attention python scripts/sprint_2H_instance_signal.py --stage recovery
conda run -n recover_attention python scripts/sprint_2H_instance_signal.py --stage analyze
conda run -n recover_attention python -m pytest -q
```

检查结果：

- 2H-B targeted pytest：24 passed。
- full pytest：570 passed, 2 skipped。
- select：subset=500，questions=500，on_path=1250，off_path=220，ambiguous=226。
- recovery：500 masked_ids，1500 samples，backend=ollama_chat_v0，model=qwen3.5:9b，temperature=0.8，K=3。
- targets：trainable=477，excluded_ambiguous=23，is_span_type_deterministic=False，ordering_violations=0。
- probe：hidden_no_recovered macro_f1=0.318 < span_type_only 0.339 < surface_rule 0.378；gate 第 5 项失败。
- gate：6/7 通过，ready_for_2000_rerun=False。

边界：

- 复用 2G-2000 的 original/masked hidden-state 缓存，未重跑 hidden-state cache，未缓存 recovered question hidden states。
- 未执行 attention steering，未 all-train，未进入 Sprint 3A，未覆盖 2A-2G 任何输出。
- 2H-B 是 weak-labeled instance-signal diagnostic，非 human-reviewed validation；未验证 hallucination reduction 或 answer accuracy improvement。

遗留问题：

- 唯一 gate-eligible 特征集（original_masked cosine）过薄，且 2G 中大部分表征判别力位于被正确禁用的 recovered channel；需更丰富的 pre-recovery 特征才能真正检验「hidden states 是否携带 instance-level fragility」。
- fragility_bucket 虽非 span_type 纯函数，但多数类仍与 span_type 强相关（number 多为 bucket 3），使 span_type_only baseline 偏强。
- per-question top-k 覆盖为平凡 1.0（每题 1 个 chosen span）；要做有意义的 top-k / budget 评估需每题多 span。
- `human_error_type` 仍为 sentinel，drift 标签为启发式文本分类的弱标签，非人工校验。
