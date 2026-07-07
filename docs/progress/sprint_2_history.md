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

## Sprint 2H-C：Pre-Recovery Feature Enrichment

目标：不改 label、不调 threshold，而是从 2G 已缓存的原始 hidden-state 张量重新构造更丰富的 pre-recovery 特征，判断模型内部状态是否真的存在 instance-level fragility signal。核心边界：主 gate candidate（`hidden_pre_recovery_enriched`）只用 original + masked 两个 channel，禁止 recovered / solution_path / drift / bucket / risk_strength / gold 任意子串出现在 feature name。

已完成内容：

- 关键发现：2G 的 `.pt` 缓存存储的是完整 `[layers, tokens, hidden]=(5, seq, 3584)` 逐 token 逐层 hidden states（layers 0/8/16/24/27），且 manifest 含 token_char_ranges 与 span/mask char range，可复用 `representation_features.py` 的 `locate_span_token_indices` / `pool_span` / `pool_question` 重新 pooling。2H-B 特征薄的根因：2G 中 span pooling 与 mask_position pooling 完全相同，且只保留了 angle-only cosine 摘要。
- 实现 `src/recover_attention/pre_recovery_features.py`：从 original/masked 张量构造 3 类特征——A. layer-wise original→masked delta（cosine / L2 / relative_norm / slope / range / var / max_layer）；B. within-channel span saliency（span-to-question、span-to-number-context，逐层 + 摘要）；C. cross-layer stability（layer variance、early→late cosine、layer shift norm）。跨层「drift」字段改名为 `layer_shift` 以避开禁用子串。
- 在 `fragility_probe_training.py` 新增 gate candidate feature set `hidden_pre_recovery_enriched`，保留 span_type_only / surface_rule / hidden_no_recovered / hidden_with_recovered；对 gate-eligible 集合断言无 recovered，对 enriched 集合断言无全部禁用子串。
- 新增防泄漏测试（任务 4）：enriched feature_names 不含 recovered / solution_path / drift / bucket / risk_strength / gold，任一出现即测试失败；另加 stub-tensor 抽取测试。
- 实现 `scripts/sprint_2H_feature_enrichment.py`（audit / extract / probe / gate / analyze / all）；复用 2H-B 的 `risk_strength_dataset.jsonl`，不重跑 recovery、不重跑 hidden-state cache、不扩大规模。

关键数据（500 条，复用 2H-B fragility 数据集）：

- audit：hidden_no_recovered 仅 33 个特征，全部为 `*_original_masked_cosine_*`（cosine-only，无 recovered 泄漏）；确认信号薄是 2H-B 未过 gate 的直接原因。
- extract：500 条全部 span_available；enriched 特征 79 个（A/B/C 三族），leakage_free=True；仅 numctx 缺失 14 条（单数字题）。attention 特征在当前 cache 不可用（2G 只缓存 hidden states，无 attention maps），本轮不重跑模型。
- 探针（macro_f1 / balanced_acc / bucket3_recall / spearman / bucket3v1_AUC / pairwise）：
  - hidden_pre_recovery_enriched（gate candidate）：0.426 / 0.434 / 0.786 / 0.353 / 0.807 / 0.665
  - hidden_no_recovered：0.318 / 0.355 / 0.771 / 0.256 / 0.690 / 0.620
  - span_type_only baseline：0.339 / 0.402 / 0.979 / 0.374 / 0.858 / 0.672
  - surface_rule baseline：0.378 / 0.406 / 0.958 / 0.387 / 0.849 / 0.680
  - hidden_with_recovered（仅泄漏诊断）：0.371 / 0.388 / 0.812 / 0.279 / 0.725 / 0.629
  - bootstrap：enriched vs surface_rule macro_f1 delta=+0.048（CI95 +0.008..+0.084，显著）；vs span_type_only delta=+0.087（CI95 +0.047..+0.124，显著）。
- review gate：8 项通过 5 项，ready_for_2000_rerun=False。
  - 通过：#1 macro_f1>surface、#2 balanced_acc>surface、#3 bootstrap CI95 low>0 vs surface、#6 top-k coverage 不下降（1.0=1.0）、#8 feature leakage test。
  - 失败：#4 bucket_3_recall（enriched 0.786 < surface 0.958）——surface baseline 靠「number→bucket3 泛滥」拿到高 bucket3 recall 但拉低 macro，属退化基线 artifact，与最大化 macro_f1 相冲突；#5 排序指标（spearman 0.353<0.387、pairwise 0.665<0.680）——真实但幅度小的排序劣势；#7 off-path budget（enriched 0.054 vs no_recovered 0.050）——+0.004 边际打平。

核心结论：

- 本轮推翻了 2H-B 的负面结论：更丰富的 pre-recovery hidden-state 特征在 macro_f1 与 balanced_accuracy 上显著优于 surface baseline（bootstrap CI95 均 >0），说明模型内部状态确实携带 span_type/surface 之外的 instance-level fragility signal，且其判别力主要来自 delta magnitude（L2/relative norm）与 cross-layer stability，而非 2H-B 的 angle-only cosine。
- 但 enriched probe 并未在所有轴上压制 surface baseline：排序指标（spearman/pairwise）仍略逊，bucket_3 recall 因退化基线而偏低，off-path budget 边际打平。因此按本轮 gate（8 项全过才 ready）判定：不进入 2000-case rerun，不进入 Sprint 3A。
- 下一步方向明确：ordinal-calibrated 探针（用 expected-bucket 排序或序数回归改善 spearman/pairwise）+ 少量针对 bucket-3-vs-rest 的判别特征；attention 级特征需先补一个 targeted attention cache（本轮 boundary 内不重跑模型）。

输出文件（均在 `outputs/logs/sprint_2H_feature_enrichment_500/`）：

```text
current_feature_audit.json / current_feature_audit.md
pre_recovery_feature_dataset.jsonl / pre_recovery_feature_report.json
fragility_probe_enriched_eval_report.json / fragility_probe_enriched_predictions.jsonl
review_gate_feature_enrichment.json / review_gate_feature_enrichment.md
```

新增或修改文件：

- src/recover_attention/pre_recovery_features.py（新增）
- src/recover_attention/fragility_probe_training.py（新增 hidden_pre_recovery_enriched feature set + 禁用子串断言）
- scripts/sprint_2H_feature_enrichment.py（新增）
- tests/test_sprint_2h_instance_signal.py（新增 2H-C 防泄漏与抽取测试）
- outputs/logs/sprint_2H_feature_enrichment_500/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_sprint_2h_instance_signal.py -q
conda run -n recover_attention python scripts/sprint_2H_feature_enrichment.py --stage audit
conda run -n recover_attention python scripts/sprint_2H_feature_enrichment.py --stage extract
conda run -n recover_attention python scripts/sprint_2H_feature_enrichment.py --stage probe
conda run -n recover_attention python scripts/sprint_2H_feature_enrichment.py --stage gate
conda run -n recover_attention python -m pytest -q
```

检查结果：

- 2H-C targeted pytest：26 passed（2H + 2H-C 共用测试文件）。
- full pytest：573 passed, 2 skipped。
- audit：hidden_no_recovered=33 features，only_original_masked=True，cosine_only=True。
- extract：500 records，79 enriched features，span_available=500，leakage_free=True，numctx missing=14。
- probe：enriched macro_f1=0.426 > surface_rule 0.378 > span_type_only 0.339 > hidden_no_recovered 0.318；bootstrap 对两个 baseline 均显著。
- gate：5/8 通过，ready_for_2000_rerun=False（#4 退化基线 artifact、#5 排序小幅劣势、#7 边际打平）。

边界：

- 复用 2G 的 `.pt` hidden-state cache 与 2H-B 的 `risk_strength_dataset.jsonl`；未重跑 recovery、未重跑 hidden-state cache、未扩大规模。
- 主 probe 仅用 original + masked channel；未使用 recovered channel / recovery outputs / gold solution path / fragility label 作为输入特征。
- 未覆盖 2G / 2H-B 任何输出；未 all-train；未进入 Sprint 3A；未执行 attention steering。
- 2H-C 仍为 weak-labeled instance-signal diagnostic，非 human-reviewed validation。

遗留问题：

- enriched probe 的排序指标（spearman/pairwise）仍略逊于 surface baseline；分类（macro_f1/balanced_acc）已显著更优，说明需要 ordinal-calibrated 目标或排序损失。
- bucket_3_recall gate 项与「最大化 macro_f1」内在冲突（surface baseline 靠退化的 number→bucket3 预测拿高 bucket3 recall）；后续 gate 设计应改用 macro/ordinal 综合项，而非单类 recall 对齐退化基线。
- attention-level 特征（span attention mass / entropy / mask-to-span attention）在当前 cache 不可得；如需检验必须先补 targeted attention cache（不在本轮 boundary 内）。
- fragility label 仍为 2H-B 的 weak instance labels（模型 recovery + 启发式 drift 分类），非人工校验。

## Sprint 2H-D：Ordinal Calibration and Budget-Aware Gate Redesign

目标：不改 label、不扩规模，把 2H-C 的 enriched hidden-state signal 通过校准转化为更稳定的 ordinal / budget-aware risk score，并重设 gate 以避免 surface_rule 靠「number→bucket3 泛滥」拿到的假 bucket_3_recall 优势。复用 2H-C 的 `pre_recovery_feature_dataset.jsonl`，不重跑 recovery / hidden-state cache。

数据一致性核验（重要）：开工前用当前（已被修改的）drift 代码，从现有 recovery outputs 重算 fragility labels，与磁盘上 2H-B/2H-C 标签逐条比对——0/500 改变，bucket 分布一致 `{0:23,1:109,2:153,3:192,None:23}`。故 2H-C 数据集与当前代码一致，2H-D 与 2H-C 直接可比，无 stale-label 问题。

已完成内容：

- 实现 `src/recover_attention/ordinal_calibration.py`：三种校准/打分方法——A. expected-bucket（Σ softmax(decision_k/T)·k，温度 T 在 train fold 内网格拟合）；B. ordinal-threshold（三个二分类 P(bucket≥1/2/3) 求和）；C. calibrated regression（ridge risk_strength 回归 + PAV isotonic 单调校准）。全部 per-train-fold 拟合，避免 test-fold 泄漏。另含 budget-aware bucket-3 指标（top-k% precision/recall）与 bootstrap 排序 delta。
- 实现 `scripts/sprint_2H_ordinal_calibration.py`：对 enriched / surface_rule / span_type_only 三个特征集用同一方法逐 fold 产生 OOF 分数并评估（公平：同方法比较，不只给 enriched 加校准）。
- 新增 6 个 2H-D 测试（softmax 归一、expected-bucket 单调、PAV 单调、isotonic 保序、budget_curve、bootstrap delta）。

关键实现要点（关闭三个陷阱）：

- per-fold 校准（温度 / isotonic / 阈值分类器全部只在 train fold 拟合）；
- 对 enriched 与 surface 应用同一打分方法后再比较（否则只给 enriched 校准是不公平优势）；
- 对主排序指标 Spearman 逐方法都做 bootstrap，避免结论悬于一个噪声化的 primary 选择；
- primary method 选择规则为 surface-blind 的「enriched Spearman 最大」（预注册，不偷看 surface）。

关键数据（500 条，复用 2H-C 数据集）：

- 各方法 enriched vs surface 排序（Spearman / pairwise / bucket3v1_AUC）：
  - expected_bucket：0.386/0.682/0.820 vs 0.310/0.645/0.777（enriched 三项全胜）
  - ordinal_threshold（primary）：0.391/0.683/0.833 vs 0.389/0.682/0.859（Spearman/pairwise 打平，AUC 输）
  - reg_calibrated：0.337 vs 0.390（surface 略优）；reg_raw：0.353 vs 0.387（surface 略优）
- 校准提升了 enriched 绝对排序：Spearman 从 2H-C 的 0.353 提升到 0.386（expected_bucket）/0.391（ordinal_threshold），但 surface 也在 ~0.39，属打平。
- enriched vs surface Spearman 逐方法 bootstrap：expected_bucket delta=+0.076（CI95 [+0.003,+0.159]，显著）；ordinal_threshold +0.002（不显著）；reg_calibrated −0.053、reg_raw −0.033（均不显著）。即仅 1/4 方法显著、且 CI 下界紧贴 0。
- budget-aware bucket-3（primary=ordinal_threshold）：top-10% enr precision 0.688 vs surf 0.708（surface 略优）；top-20% enr 0.726 vs surf 0.674（enriched 略优）——混合、无稳定优势；也证实 surface 高整体 bucket_3_recall 来自泛滥而非有限预算下的精准。
- classifier macro_f1 保持 2H-C 水平：enriched 0.426 > surface 0.378 > span_type 0.339（未回退）。coverage 1.0（平凡）；off-path budget 0.055（≈2H-C 0.054，未恶化）。
- review gate（8 项，primary=ordinal_threshold）：通过 5 项（#1 Spearman>surface、#2 pairwise>surface、#4 coverage≥2H-C、#5 budget 未恶化、#7 macro_f1 未回退）；失败 3 项（#3 bucket3v1_AUC 输、#6 top-10% bucket-3 precision 略输、#8 primary 方法 bootstrap CI 下界 <0）。ready_for_2000_rerun=False。

核心结论：

- 校准有效：把 enriched 绝对排序从 0.353 提升到 ≈0.39，与 surface 打平，并保住 2H-C 分类优势（macro_f1 0.426）。存在一个 principled 方法（expected_bucket，从分类器派生，恰是 enriched 优势所在）使 enriched 显著优于 surface（+0.076，CI 显著）。
- 但该排序优势不稳健：仅 4 种打分中的 1 种显著、CI 下界仅 +0.003，budget-aware top-10% bucket-3 打平/略输。按任务门槛「ordinal score 明显超过 surface + budget-aware bucket3 优于 surface」未达标，故不进入 2000-case rerun。
- 诚实性：未把 primary 切到唯一能过的 expected_bucket 去凑 gate（那是 p-hacking）；保留 surface-blind 预注册规则（→ ordinal_threshold → 5/8），并把 expected_bucket 的显著结果作为「promising 但不稳健」透明记录。
- 下一步（任务 fail 分支）：补 targeted attention-map cache（span attention mass / entropy / mask-to-span attention），再考虑 trajectory-level features；expected_bucket 的显著性提示「分类器派生的 ordinal 分」值得在更强特征下复核。

输出文件（均在 `outputs/logs/sprint_2H_ordinal_calibration_500/`）：

```text
ordinal_calibration_report.json / ordinal_calibration_report.md
ordinal_predictions.jsonl
bucket3_budget_curve.json
review_gate_ordinal_calibration.json / review_gate_ordinal_calibration.md
```

新增或修改文件：

- src/recover_attention/ordinal_calibration.py（新增）
- scripts/sprint_2H_ordinal_calibration.py（新增）
- tests/test_sprint_2h_instance_signal.py（新增 2H-D 校准测试）
- outputs/logs/sprint_2H_ordinal_calibration_500/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_sprint_2h_instance_signal.py -q
conda run -n recover_attention python scripts/sprint_2H_ordinal_calibration.py --stage all
conda run -n recover_attention python -m pytest -q
```

检查结果：

- 2H-D targeted pytest：32 passed（2H/2H-C/2H-D 共用测试文件）。
- full pytest：579 passed, 2 skipped。
- provenance：当前 drift 代码重算 fragility labels 与磁盘 0/500 差异。
- primary_method=ordinal_threshold；gate 5/8，ready_for_2000_rerun=False；逐方法 bootstrap 仅 expected_bucket 显著（+0.076，CI [+0.003,+0.159]）。

边界：

- 复用 2H-C `pre_recovery_feature_dataset.jsonl`；未重跑 recovery、未重跑 hidden-state cache、未扩大到 2000。
- 仅用 enriched（original+masked）与 surface / span_type 特征；未使用 recovered channel / gold solution path / drift / bucket / risk_strength 作为输入特征（gate candidate 特征名断言通过）。
- 未覆盖 2G / 2H-B / 2H-C 输出；未 all-train；未进入 Sprint 3A；未执行 attention steering；未手动改标签。

遗留问题：

- enriched 对 surface 的排序优势仅在 expected_bucket 一种打分下显著，且 CI 下界紧贴 0，不稳健；需更强特征（attention/trajectory）复核。
- budget-aware bucket-3 在 top-10% 预算下 enriched 未稳定优于 surface；bucket_3_vs_1 AUC 在 primary 方法下仍输 surface。
- per-question top-k 覆盖仍为平凡 1.0（每题单 span）；有意义的 top-k / budget 评估需每题多 span。
- fragility label 仍为 weak instance labels（模型 recovery + 启发式 drift），非人工校验。

## Sprint 2I：Targeted Attention-Map Feature Cache and Probe

目标：引入 attention-level pre-recovery 特征，判断 attention maps 是否比 hidden-state-only 提供更稳健的 instance-level fragility ranking / budget-aware signal。只在 500-case subset 上运行；新 gate candidate 为 `hidden_plus_attention_pre_recovery`，同时单独报告 `attention_pre_recovery`。复用 2H-C/2H-D 的同一 subset 与 labels。

可行性核验（开工前）：GPU=RTX 5070 Ti Laptop 12.8GB；模型 D:/models/Qwen2.5-7B-Instruct 在盘；2G 用 4-bit（bitsandbytes）+ device_map=auto 装载（bf16 全量 15GB 放不下 12.8GB VRAM，故必须 4-bit）。smoke 确认：4-bit + `attn_implementation="eager"` + `output_attentions=True` 可返回 28 层 [1,28,seq,seq] 且行和为 1，单次 forward ~0.55s。

任务卡审查：发现一个真实漏洞——任务 3C 建议特征名 `question_target_to_span_attention`，但任务 5 禁用子串 `target`（还新增 `answer`/`label`）；直译会导致 leakage test 失败。已改名为 `question_focus`（qfocus）避开禁用词。其余方法学合理，且任务 7 条件 #7「至少两个 scoring method 同向」正好针对 2H-D 的不稳健问题。

已完成内容：

- 实现 `src/recover_attention/attention_features.py`：从 original/masked（绝不 recovered）的 head-averaged attention 计算 4 类 leakage-free 摘要——A. slot mass（in/out/self/rank/in_rel）；B. shape（entropy/top1-3-5 mass/effective edge count）；C. context-to-slot（qfocus / operation / numctx → slot）；E. original→masked delta（含逐层 in_mass/entropy delta）。特征名规避全部禁用子串（含 answer/label/target）。
- 实现 `scripts/sprint_2I_attention_cache.py`：4-bit eager 重跑 original+masked forward（不缓存完整 tensor，只存摘要），用 tokenizer 自身 offset mapping 定位 slot（与 attention seq 维自洽），可续跑。
- 实现 `scripts/sprint_2I_attention_probe.py`：复用 2H-D 的 `run_ordinal_cv`（per-fold 校准、4 种打分、budget-aware bucket-3、bootstrap）评估 5 个特征集，gate 候选与 surface 及 hidden-only enriched 双向比较。
- 在 `fragility_probe_training.py` 注册 `attention_pre_recovery` / `hidden_plus_attention_pre_recovery` 两个 feature set（含扩展禁用子串断言）；新增 3 个 2I 测试。

关键实现修复：最终层（layer 27）在 4-bit eager 下 attention 全为 NaN（layers 0/8/16/24 正常，行和 1.0）；故 attention 层集合改为 [0,8,16,24] 并加 `nan_to_num` 兜底。

关键数据（477 trainable，attention 来自 4-bit 量化模型；context 缺失 qfocus 73 / operation 63 / numctx 14）：

- 分类（macro_f1 / balanced_acc）：hidden_plus_attention 0.437/0.438（最佳）> hidden_only 0.426/0.434 > attention_only 0.397/0.407 > surface 0.378/0.406 > span_type 0.339/0.402。attention 单独已优于 surface；hidden+attention 比 hidden-only 提升 +0.011，bootstrap 显著（macro_f1 delta vs surface +0.060，CI95 [+0.008,+0.117]）。
- 排序（primary=expected_bucket 的 Spearman）：hidden_plus_attention 0.389 vs hidden_only 0.386 = +0.003，bootstrap 不显著（CI95 [−0.046,+0.054]）；即 attention 对排序几乎无增益。cand vs surface Spearman +0.079（CI95 [+0.002,+0.168] 显著），但这基本是 hidden 的贡献。
- 跨方法稳健性：cand 的 Spearman 仅在 expected_bucket 一种打分下 > surface（4 种里 1 种）；任务 7 条件 #7 要求 ≥2 种同向，故失败。
- budget-aware bucket-3：top-10% precision hidden_only 0.729 最优、cand 0.708、surface 0.667；top-20% hidden_only 0.747 > cand 0.663 > surface 0.484。即 attention 未改善预算内 bucket-3 选择，hidden-only 反而最好。
- attention family 有用性（|Spearman| vs bucket）：C_context_to_slot 0.164、E_delta 0.134 最有用；B_shape/A_mass ≈0.079。即「问题上下文/数字对 slot 的注意力」与「masked 后 attention 的重排」信号最强。
- off-path budget：cand 0.056 ≈ hidden 0.057（未恶化）；coverage 1.0（平凡）。

review gate（7 项，primary=expected_bucket）：通过 6 项（#1 Spearman 显著>surface、#2 >hidden_only、#3 top-budget bucket-3>surface、#4 budget 未恶化、#5 coverage 未降、#6 leakage 通过）；失败 #7（≥2 打分方法同向，仅 expected_bucket 一种）。ready_for_2000_rerun=False。

核心结论：

- attention 特征确有真实 instance-level 信号：attention 单独 macro_f1 0.397 已超 surface；hidden+attention 0.437 是目前最强分类器（bootstrap 显著）。最有用的是 context-to-slot 与 original→masked delta 两族——与「脆弱 span 被 masked 后引发 attention 重排」的假设一致。
- 但 attention 未解决 2H-D 标记的真正瓶颈——ordinal ranking 的稳健性：hidden+attention 的排序与 hidden-only 基本相同（+0.003 不显著），且 cand vs surface 的排序优势仍只在 1/4 打分方法下成立（不满足 ≥2 方法）；budget-aware bucket-3 甚至不如 hidden-only。
- 诚实性：未切换 primary、未放宽 ≥2-method 要求去凑 gate。attention 带来的是「分类」提升（enriched 早已擅长），而非「排序/预算稳健性」提升，故按任务门槛不进入 2000-case rerun。
- 下一步（任务 fail 分支）：转向 trajectory-level features（多步生成的 answer / trajectory stability）或 hybrid recovery-guided pipeline；expected_bucket 仍是 2H-D/2I 中唯一稳定使 hidden(+attention) 超过 surface 的打分，值得在新特征下延续。

输出文件（`outputs/logs/sprint_2I_attention_cache_500/` 与 `outputs/logs/sprint_2I_attention_features_500/`）：

```text
attention_cache_report.json
attention_feature_dataset.jsonl / attention_feature_report.json
attention_probe_eval_report.json
attention_ordinal_calibration_report.json / attention_budget_curve.json
review_gate_attention_features.json / review_gate_attention_features.md
```

新增或修改文件：

- src/recover_attention/attention_features.py（新增）
- scripts/sprint_2I_attention_cache.py、scripts/sprint_2I_attention_probe.py（新增）
- src/recover_attention/fragility_probe_training.py（注册 attention / fusion feature set + 扩展禁用子串断言）
- tests/test_sprint_2h_instance_signal.py（新增 2I attention 测试）
- outputs/logs/sprint_2I_attention_cache_500/*、outputs/logs/sprint_2I_attention_features_500/*

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_sprint_2h_instance_signal.py -q
conda run -n recover_attention python scripts/sprint_2I_attention_cache.py
conda run -n recover_attention python scripts/sprint_2I_attention_probe.py
conda run -n recover_attention python -m pytest -q
```

检查结果：

- 2I targeted pytest：35 passed（2H/2H-C/2H-D/2I 共用测试文件）。
- full pytest：582 passed, 2 skipped。
- attention cache：477 records，57 attention features，leakage_free=True，seq_mismatch=0，slot_missing=0，recovered_channel_cached=False，~5.5 forward/s。
- probe：hidden_plus_attention macro_f1=0.437（最佳）；gate 6/7，ready_for_2000_rerun=False（#7 仅 1 种打分方法同向）。

边界：

- 只缓存 original + masked attention，未缓存 recovered question attention / hidden states；未重跑 recovery；未扩大到 2000；未进入 Sprint 3A；未执行 attention steering。
- gate-eligible 特征仅用 original/masked attention；未使用 recovered channel / gold solution path / drift / bucket / risk_strength / answer / label / target 作为输入特征（断言通过）。
- 未覆盖 2G / 2H-B / 2H-C / 2H-D 输出。attention 来自 4-bit 量化模型（与 2G 的 4-bit hidden states 一致），且排除了 NaN 的最终层。

遗留问题：

- attention 提升的是分类（macro_f1）而非 ordinal ranking 稳健性；hidden+attention 排序 ≈ hidden-only；cand vs surface 排序仍只在 expected_bucket 一种打分下显著。
- budget-aware bucket-3 在有限预算下 hidden-only 最优，attention 未带来增益。
- context-to-slot 依赖 qfocus/operation 的正则定位，缺失率 qfocus 15% / operation 13%；更强定位（句法/语义）可能提升 C 族信号。
- per-question top-k 覆盖仍平凡 1.0；trajectory-level 评估需每题多 span 与多步生成信号。
