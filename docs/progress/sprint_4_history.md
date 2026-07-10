# Sprint 4 History - Cyber Hallucination Control

## Sprint 4A - Cybersecurity Direction-Probe Mainline Reset

Goal: reset the project mainline from unsupervised span-guided steering to cyber-domain supervised direction probing. This sprint is documentation, requirements clarification, and code reuse audit only. It does not train a probe, call an LLM, run steering, patch/nudge activations, enter 2000-scale, or continue full Sprint 3C.

Reason for reset:

```text
Sprint 3A / 3B: blind span-level interventions failed to produce selective answer-directed improvement.
Sprint 3C-1 / 3C-2: answer-readout MLP is a valid causal write site, and gold-unembedding direction is effective as an oracle.
Sprint 3C-3 / 3C-4A: gold-free direct readout and approximate J-lens remain weaker than final-logits margin, limiting detection/readout claims.
```

Conclusion:

```text
span relevance is evidence, not a controller.
The missing component is a supervised mapping:
span / activation / trace -> label direction / correction direction.
```

Sprint 4A outputs:

```text
docs/codex_tasks/sprint_4A_cyber_direction_probe_mainline_reset.md
docs/reference/CYBER_DIRECTION_PROBE_PLAN.md
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/reference/STORY.md
docs/progress/sprint_4_artifact_manifest.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/preflight_report.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/review_gate_cyber_direction_probe_mainline_reset.md
```

Boundary flags:

```text
ready_for_2000_rerun=false
do_not_enter_full_sprint_3C=true
hallucination_reduction_proven=false
answer_accuracy_improvement_proven=false
steering_continued=false
domain_supervised_probe_planned=true
```

Next route:

```text
Sprint 4B: Cyber Dataset Selection and Domain Schema Implementation.
Sprint 4C: Cyber Probe Dataset Construction.
Sprint 4D: Probe-Guided MLP Readout Steering.
Sprint 4E: Robustness and Write-up.
```

Test status: `tests/test_dataset_audit.py tests/test_stage_summary.py` passed 14 tests; `tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py` passed 15 tests.

## Post-4A Documentation Guardrail Cleanup - Sprint 4B Ready

Goal: patch the Sprint 4B documentation before execution so the run starts from
dataset/schema/F5 baselines and not from probe training, steering, or Stage 5 by
default. This is a documentation guardrail cleanup, not an experimental result.
No model was called, no data was downloaded, no probe was trained, no steering
was run, and no 4B site-transfer experiment was executed.

Mainline clarification:

```text
CYBER_HALLUCINATION_CONTROL_PLAN.md supersedes CYBER_DIRECTION_PROBE_PLAN.md.
Sprint 4 mainline is domain-calibrated hallucination detection + gated closed-form intervention.
Sprint 4B starts from dataset/schema/F5 baseline, with Stage 5 gated optional.
```

Guardrails added:

```text
parameterized model_path resolution with source reporting;
semantic label id/text retention in candidate_choices;
option_position_bias_report.json requirement;
deterministic option-order and semantic-preservation test requirements;
Stage 5 gate and skipped site_transfer_check_report.json behavior;
review gate questions 18-23.
```

## Merge Guardrail Repair before Sprint 4B Execution

Documentation-only repair after merge conflict resolution. Restored:

```text
- semantic label preservation alongside option-letter readout labels;
- option-position bias audit requirements;
- gated optional Stage 5 site-transfer;
- Review gate items 18-23;
- updated Sprint 4 control plan wording.
```

No experiment was run.

## Sprint 4B Dataset Download and Raw Format Audit

Goal: prepare raw cyber MCQ data sources for Sprint 4B without running the 4B
experiment. This task only downloaded raw/source files, inspected fields and
option/answer formats, and wrote preview/audit artifacts. It did not call a
model, generate completions, train probes, run F5 baselines, do steering, or run
Stage 5 patching.

Outputs:

```text
data/raw/cyber/cybermetric/
data/raw/cyber/secqa/
data/raw/cyber/cs_eval/
outputs/logs/sprint_4B_dataset_download_audit/dataset_source_audit.md
outputs/logs/sprint_4B_dataset_download_audit/raw_file_manifest.json
outputs/logs/sprint_4B_dataset_download_audit/sample_records_preview.jsonl
scripts/sprint_4B_download_and_audit_cyber_datasets.py
```

Raw audit result:

```text
CyberMetric: downloaded 500 / 2000 / 10000 source JSON files; MCQ-like with 4 options and gold solution field; recommended primary source.
SecQA: downloaded v1/v2 dev/val/test CSV files; MCQ-like with 4 options and gold Answer field; small fallback or held-out source.
CS-Eval: downloaded repository metadata/examples/license only; source-inspection-only until a plain MCQ data file is identified.
```

Boundary:

```text
no model call;
no completion generation;
no F5 baseline;
no probe training;
no steering;
no Stage 5 site-transfer or patching.
```

Commands and checks:

```bash
conda run -n recover_attention python scripts/sprint_4B_download_and_audit_cyber_datasets.py --output-dir outputs/logs/sprint_4B_dataset_download_audit
conda run -n recover_attention python -m py_compile scripts/sprint_4B_download_and_audit_cyber_datasets.py
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Check result: audit script completed, syntax check passed, and lightweight
pytest passed 14 tests.

## Sprint 4B CyberMetric Smoke Baseline

Goal: execute the minimal CyberMetric smoke chain without a full primary run, probe training, steering, LoRA/finetuning, full Sprint 3C, or 2000-scale execution.

Setup:
```bash
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py --dataset cybermetric --primary-questions 30 --samples-per-question 2 --temperature 0.7 --max-new-tokens 128 --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke --overwrite
```

Implementation:
- Added `src/recover_attention/cyber_data.py` for CyberMetric raw loading, deterministic option shuffle, canonical MCQ schema, prompt construction, grouped split summary, dataset audit, and option-position bias audit.
- Added `src/recover_attention/domain_label_proxy.py` for option-letter tokenization, answer parsing, label-readout locating, F5 margin/entropy/self-consistency helpers, fixed F5 risk scoring, and gold-feature leakage checks.
- Added `scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py` for the smoke chain. The script uses CPU-side token sampling to avoid CUDA multinomial failures in sampled generation; it does not train a probe.
- Added `tests/test_cyber_data.py` and `tests/test_domain_label_proxy.py`.

Smoke outputs:
```text
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/preflight_report.md
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/dataset_audit_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/label_space_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/option_position_bias_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/cyber_sample_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/trace_sampling_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/f5_baseline_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/correct_wrong_pair_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/site_transfer_check_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/review_gate_cyber_dataset_baseline_site_transfer.md
```

Results:
- Canonical schema: 30 CyberMetric questions.
- Trace sampling: 90 traces (30 greedy + 60 sampled), parse_failure_rate 0.100, sampled wrong_rate 0.315.
- Pair construction: 7 same-question correct/wrong pairs.
- Option-position audit: no severe warning; gold A/B/C/D distribution 6/7/10/7; majority-position baseline accuracy 0.333.
- F5 smoke baseline on 28 scored greedy examples: label-margin risk AUROC/AUPRC 0.741/0.553; label entropy 0.748/0.558; full entropy 0.374/0.244; self-consistency risk 0.823/0.497; fixed non-trained F5 combination 0.776/0.532.
- Stage 5: skipped. Gate failed because `num_questions_with_correct_and_wrong >= 20` was false (7 pairs). `site_transfer_check_report.json` records `status="skipped"` and the skipped reason.
- Rerun verification on 2026-07-09: repeated the same smoke command with `--overwrite`; required artifacts were regenerated locally; Stage 5 remained skipped for the same 7-pair gate failure.

Checks:
```bash
conda run -n recover_attention python -m pytest tests/test_cyber_data.py tests/test_domain_label_proxy.py -q
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Boundary: raw data and smoke outputs remain local/gitignored artifacts. No probe training, steering, nudge, LoRA/finetuning, full Sprint 3C, or 2000-scale run occurred. No hallucination-reduction or accuracy-improvement claim is made.


## Sprint 4B-1 - CyberMetric Canonical Schema and Domain Label Proxy

Goal: establish the data/label interface for the hallucination-control mainline before any model-generation stage. Task card: `docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md`. Strictly no causal LM calls, no completions, no F5, no site-transfer, no probe, no steering; tokenizer-only load of the local Qwen2.5 tokenizer was the sole model asset touched (explicitly allowed by the card).

Implemented:
- `schemas.py`: `cyber_sample` record type + `validate_cyber_sample_record` (field rules for candidate_labels/candidate_choices alignment, gold mapping, label_space, metadata splits).
- `cyber_data.py`: CyberMetric loading/normalization, deterministic per-example option shuffle (sha256-derived per-example seeds; no builtin hash; no in-place mutation), question-hash `group_id` proxy, grouped split, MCQ prompt builder, audits.
- `domain_label_proxy.py`: pure option-letter proxy - `option_token_ids` (whitespace-marker stripping, single-token and collision enforcement), `parse_option_answer` (explicit `Answer: X` > isolated-letter fallback > parse_failure; negative cases like Aes/BERT/CVE/DNS covered), `locate_label_readout_position`, `label_margin`/`label_entropy`/`full_entropy`, `self_consistency_features` (gold-free by construction), recursive gold-leakage checker.
- `scripts/sprint_4B_1_prepare_cybermetric.py`: raw -> canonical -> shuffle -> grouped split -> validation -> manifests -> pre-model audits -> review gate.
- The 4B smoke script was adapted to the new APIs (card compat option 1; compile-checked only).

Results:
- 2000/2000 CyberMetric-2000 records converted, 0 invalid, 0 duplicate example_ids.
- Grouped split train/dev/test = 1400/300/300, group leakage 0 (group_id honestly labeled a question-derived proxy).
- Post-shuffle gold distribution A/B/C/D = 465/503/529/503; fixed-seed reproducible, different-seed order changes verified; gold semantic mapping intact after shuffle.
- Real Qwen2.5 tokenizer check passed: A/B/C/D -> 362/425/356/422, distinct single non-whitespace tokens.
- Smoke manifest: 240 records, split-aware sampling.

Execution bug found and fixed: the tokenizer check was initially skipped with `ValueError: invalid literal for int(): 'input_ids'`. HuggingFace `BatchEncoding` subclasses `UserDict`, not `dict`, so `isinstance(encoded, dict)` in `_tokenizer_ids` fell through and iterated key names as token ids. Fixed via mapping duck-typing; regression test with a UserDict-style fake tokenizer added. Leftover `*.patch`/`tmp_patch/` scratch files (which broke pytest collection) were archived out of the repo root.

Checks: targeted pytest 187 passed; full pytest 708 passed, 2 skipped. All Section-22 entry conditions for Sprint 4B-2 are met.

Boundary: no hallucination-reduction or accuracy claim; gold labels remain eval-only; raw data and outputs stay gitignored with this manifest as the audit record.

Next: Sprint 4B-2 small-model smoke (~32 questions) carrying the three Section-23 carryover items (prompt A/B + degeneration detector; sampling-protection check; F5 cost-tier split in 4B-3).
