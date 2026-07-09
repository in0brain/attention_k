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

Raw data artifacts:

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

Audit output artifacts:

```text
outputs/logs/sprint_4B_dataset_download_audit/dataset_source_audit.md
outputs/logs/sprint_4B_dataset_download_audit/raw_file_manifest.json
outputs/logs/sprint_4B_dataset_download_audit/sample_records_preview.jsonl
```

Raw data note: `data/raw/` is not ignored by the current `.gitignore`; do not
commit large raw data files unless explicitly intended. `outputs/` remains
gitignored.
