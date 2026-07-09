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
