# Sprint 4 Artifact Manifest

## Sprint 4A - Cybersecurity Direction-Probe Mainline Reset

Tracked documentation artifacts:

```text
docs/codex_tasks/sprint_4A_cyber_direction_probe_mainline_reset.md
docs/reference/CYBER_DIRECTION_PROBE_PLAN.md
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/reference/STORY.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
PROGRESS.md
```

Gitignored output artifacts:

```text
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/preflight_report.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/review_gate_cyber_direction_probe_mainline_reset.md
```

Sprint 4A conclusion:

```text
The project mainline is reset from unsupervised span-guided steering to cyber-domain supervised direction probing.
Sprint 3 negative results are preserved.
The answer-readout MLP remains a valid causal site.
The missing component is a supervised mapping from reasoning-aware evidence to label/correction direction.
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

Test status: `tests/test_dataset_audit.py tests/test_stage_summary.py` passed 14 tests; `tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py` passed 15 tests.

## Post-4A Documentation Guardrail Cleanup - Sprint 4B Ready

Tracked documentation artifacts updated:

```text
PROGRESS.md
docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

New or newly required Sprint 4B artifacts:

```text
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/option_position_bias_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/site_transfer_check_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/module_patch_fidelity_report.json
```

`option_position_bias_report.json` is required for Stage 1. `site_transfer_check_report.json`
may have `status="skipped"` with explicit `skipped_reason` when the Stage 5 gate
fails. `module_patch_fidelity_report.json` is only required if Stage 5 actually
runs. If `--allow-small-site-transfer` is used, the report must record it and any
result must be labeled exploratory / underpowered diagnostic.

Documentation cleanup boundary:

```text
no new experiment result;
no model call;
no data download;
no probe training;
no steering;
no Stage 5 site-transfer execution.
```

## Sprint 4B Dataset Download and Raw Format Audit

Tracked script:

```text
scripts/sprint_4B_download_and_audit_cyber_datasets.py
```

Local/gitignored raw data artifacts:

```text
data/raw/cyber/cybermetric/CyberMetric-500-v1.json
data/raw/cyber/cybermetric/CyberMetric-2000-v1.json
data/raw/cyber/cybermetric/CyberMetric-10000-v1.json
data/raw/cyber/cybermetric/README.md
data/raw/cyber/secqa/README.md
data/raw/cyber/secqa/data/secqa_v1_dev.csv
data/raw/cyber/secqa/data/secqa_v1_val.csv
data/raw/cyber/secqa/data/secqa_v1_test.csv
data/raw/cyber/secqa/data/secqa_v2_dev.csv
data/raw/cyber/secqa/data/secqa_v2_val.csv
data/raw/cyber/secqa/data/secqa_v2_test.csv
data/raw/cyber/cs_eval/LICENSE
data/raw/cyber/cs_eval/README.md
data/raw/cyber/cs_eval/README_zh.md
data/raw/cyber/cs_eval/dataset_example.md
data/raw/cyber/cs_eval/submission_example.json
data/raw/cyber/cs_eval/github_tree_manifest.json
```

These raw files are local artifacts under `data/raw/cyber/` and are ignored by
Git. They should be regenerated with
`scripts/sprint_4B_download_and_audit_cyber_datasets.py` rather than committed.

Audit output artifacts:

```text
outputs/logs/sprint_4B_dataset_download_audit/dataset_source_audit.md
outputs/logs/sprint_4B_dataset_download_audit/raw_file_manifest.json
outputs/logs/sprint_4B_dataset_download_audit/sample_records_preview.jsonl
```

Audit outputs under `outputs/logs/sprint_4B_dataset_download_audit/` remain
gitignored. The download script is tracked; raw data and generated audit outputs
are local artifacts.

## Sprint 4B CyberMetric smoke baseline artifacts

Tracked implementation artifacts:
```text
src/recover_attention/cyber_data.py
src/recover_attention/domain_label_proxy.py
scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py
tests/test_cyber_data.py
tests/test_domain_label_proxy.py
```

Local/gitignored raw data artifacts:
```text
data/raw/cyber/cybermetric/CyberMetric-500-v1.json
data/raw/cyber/cybermetric/CyberMetric-2000-v1.json
data/raw/cyber/cybermetric/CyberMetric-10000-v1.json
```

Local/gitignored smoke output artifacts:
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
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/high_risk_case_report.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/low_risk_wrong_case_report.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/review_gate_cyber_dataset_baseline_site_transfer.md
```

Artifact boundary: do not commit `data/raw/cyber/` or `outputs/logs/`; both are local artifacts. Stage 5 was skipped in the smoke because only 7 correct/wrong pairs were found, below the 20-pair gate.

Rerun verification: the smoke directory was regenerated locally with `--overwrite`; it remains gitignored and is not a tracked artifact.


## Sprint 4B-1 canonical schema and label proxy artifacts

Directory: `outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/` (gitignored) plus `data/processed/cyber/cybermetric.jsonl` (gitignored).

| file | exists | tracked | note |
| --- | --- | --- | --- |
| preflight_report.md | yes | no | gitignored outputs/ |
| dataset_audit_report.json | yes | no | 2000/2000 converted, 0 leakage |
| label_space_report.json | yes | no | real Qwen tokenizer check passed (A/B/C/D = 362/425/356/422) |
| option_position_bias_pre_model_report.json | yes | no | gold A/B/C/D = 465/503/529/503, no severe imbalance |
| cyber_sample_manifest.jsonl | yes | no | full 2000-record canonical snapshot |
| cyber_sample_smoke_manifest.jsonl | yes | no | 240 records, split-aware |
| review_gate_cybermetric_schema_and_label_proxy.md | yes | no | 30 questions answered |
| data/processed/cyber/cybermetric.jsonl | yes | no | standard pipeline input for 4B-2/4B-3 |

Key conclusion: CyberMetric is converted into a validated canonical MCQ schema with deterministic option randomization, preserved semantic option labels, and a tested option-letter label proxy; ready for the Sprint 4B-2 small-model smoke. A `BatchEncoding`-vs-`dict` duck-typing bug in the tokenizer check was found and fixed with a regression test. No model generation, F5, probe, or steering occurred in this sub-sprint.
