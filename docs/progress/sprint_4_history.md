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
