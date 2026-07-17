# Sprint 4 Artifact Manifest

## Sprint 4D-2 W0.5-B conditional-increment core and model smoke artifacts

Directory: `outputs/logs/sprint_4D_2_conditional_increment/` (local/gitignored).

| artifact | present | in git | note |
| --- | --- | --- | --- |
| preflight_report.json | yes | no | G1/G2/G3 + hidden index table + frozen constants; `stage0_full_generation_allowed=false` |
| smoke_report.json | yes | no | **G3 record**, v2.2. `passed=true`, `kind=model_smoke`, `schema_version=4D2_completion_cache_v1`, `prereg_sha256=ffa722e8...`, `n_prompts=20`, `n_traces=120` |
| smoke_report_v2.1_archived.json | yes | no | v2.1's G3, archived not overwritten. **Void** — its `prereg_sha256` is b3ef9228 and preflight rejects it automatically |
| smoke_completion_cache.jsonl | yes | no | 120 completion-level records; mentions with token_indices, eligible/label, exclusion reason, F5, surface features |
| smoke_hidden.npz | yes | no | pooled H vectors keyed by trace_id; 3584-dim fp32, eligible completions only |
| verify_hidden_report.json | yes | no | L=28, block 19 -> tuple 20, `numeric_match=true`, `max_abs_diff=0.0`, 8-bit backend |
| smoke_synthetic_report.json | yes | no | statistical wiring self-check. **Does not constitute G3** (`is_g3=false`) |
| smoke_model_v2.2_run.log | yes | no | run log for the v2.2 smoke |

Tracked in git (not artifacts but the sprint's authority records):

| file | note |
| --- | --- |
| docs/paper/preregistration.md | design authority, v2.2 frozen, sha256 `ffa722e8...` |
| docs/paper/preregistration.lock | freeze fingerprint, version history (v2.1 -> v2.2), explicit G2/G3 criteria |
| src/recover_attention/conditional_increment.py | statistical + schema core |
| scripts/sprint_4D_2_conditional_increment.py | preflight / smoke_synthetic / verify_hidden / smoke_model / stage0_gate |
| tests/test_conditional_increment.py | 39 unit tests over the correctness points that would force a 2880-forward redo |

Key conclusions: design frozen at v2.2; G2 and G3 green, G1 red, so Stage 0
remains blocked. The three-block fusion amendment removed the mechanical
O+H -> H degeneration seen under v2.1. Smoke AUROCs (O=0.801 / H=0.785 /
O+H=0.834 at n=115, n_pos=10) are wiring diagnostics and **do not enter paper
results**. `stage0_started=false`, `mcq_side_implemented=false`,
`llama_run=false`, `probe_increment_claimed=false`.

## Sprint 4D-1 H1 emission/fabrication smoke artifacts

Directory: `outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/` (local/gitignored).

Tracked implementation artifacts:

```text
src/recover_attention/h1_data.py
scripts/sprint_4D_1_h1_emission_fabrication_smoke.py
tests/test_h1_emission_smoke.py
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
PROGRESS.md
```

Local/gitignored input artifacts consumed:

```text
data/processed/h1/h1_samples.jsonl
data/raw/ontology/cve/ontology_index.jsonl
data/raw/ontology/attack/ontology_index.jsonl
data/raw/ontology/cwe/ontology_index.jsonl
outputs/logs/sprint_4D_0_h1_data_design/ontology_snapshot_manifest.json
outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json
```

Local/gitignored output artifacts:

| file | status | note |
| --- | --- | --- |
| `preflight_report.md` | present | read/modify plan, 4D-0 prerequisites, model path, boundary checks |
| `sampling_manifest.jsonl` | present | 72 train-split H1 questions, deterministic seed 4242 |
| `h1_generation_manifest.jsonl` | present | 288 completions; no `source_entry_id` field |
| `h1_mention_labels.jsonl` | present | extracted mention-level labels; no `source_entry_id` field |
| `emission_fabrication_report.json` | present | valid rerun: Route A greedy emission 46/48; ATT&CK+CWE mention fabrication 65/775 |
| `refusal_echo_diagnostic.json` | present | refusal 1/288; embedded vs clean prompt echo diagnostics |
| `review_gate_h1_emission_fabrication_smoke.md` | present | `h1_gate_passed=true`; no F5/probe/steering claim |

Generation-backend note: the first 4D-1 output set from the hand-written 4-bit
KV-cache loop is invalidated because the completions were garbled. The script
now uses `model.generate` plus a fail-fast sanity check. 4-bit long-form
generation still failed the sanity probe, fp16 was clean but too slow for the
full smoke, and the valid artifacts above come from the local 8-bit
`model.generate` rerun.

Conclusion boundary: H1 emission/base-rate smoke gate passes for the local
Qwen2.5-7B-Instruct 8-bit generation setup, but this is not an F5, detector,
intervention, hallucination-reduction, or answer-accuracy result.

## Sprint 4D-0 H1 fabricated-identifier data design artifacts

Directory: `outputs/logs/sprint_4D_0_h1_data_design/` (local/gitignored).

Tracked implementation artifacts:

```text
src/recover_attention/h1_identifier.py
src/recover_attention/h1_data.py
scripts/sprint_4D_0_download_ontology_snapshots.py
scripts/sprint_4D_0_build_h1_dataset.py
tests/test_h1_identifier.py
tests/test_h1_data.py
src/recover_attention/schemas.py
.gitignore
```

Local/gitignored raw/processed artifacts:

```text
data/raw/ontology/cve/cvelistV5-main.zip
data/raw/ontology/cve/ontology_index.jsonl
data/raw/ontology/attack/enterprise-attack.json
data/raw/ontology/attack/ontology_index.jsonl
data/raw/ontology/cwe/cwec_latest.xml.zip
data/raw/ontology/cwe/ontology_index.jsonl
data/processed/h1/h1_samples.jsonl
```

Local/gitignored report artifacts:

| file | status | note |
| --- | --- | --- |
| `ontology_snapshot_manifest.json` | present | CVE 364903, ATT&CK 858, CWE 969; URLs, sha256, status rules |
| `id_space_density_report.json` | present | CVE marked auxiliary/weak in dense low-number ranges; ATT&CK+CWE primary |
| `h1_dataset_audit_report.json` | present | 480 prompts; route A/B 360/120; split leakage 0 |
| `h1_f5_design.md` | present | mention/sequence/sampling/verbalized-confidence F5 design + 4D-1 gates |
| `review_gate_h1_data_design.md` | present | 12 review questions answered |

Conclusion boundary: H1 dataset design exists and is tested. No causal LM was
called, no completion was generated, no F5 was computed, no probe was trained,
and no hallucination-reduction, accuracy-improvement, detection-performance, or
emission-viability claim is made.

## Sprint 4C narrowed readout increment artifacts

Directory: `outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer/` (local/gitignored).

| file | status | note |
| --- | --- | --- |
| `preflight_report.md` | present | 4B-3 inputs, F5 bar, model resolution, boundary checks |
| `pair_mining_report.json` / `correct_wrong_pair_manifest.jsonl` | present | 17 pairs; below 20-pair site-transfer gate |
| `site_transfer_check_report.json` | present | skipped for insufficient pairs |
| `readout_feature_manifest.jsonl` | present | 239 gold-free F1/F4/F5 records |
| `readout_increment_report.json` | present | grouped-CV matrix and paired-bootstrap increment CIs |
| `calibration_report.json` | present | F5-only best ECE/risk-coverage diagnostic |
| `review_gate_readout_increment_and_site_transfer.md` | present | `detector_beats_f5=false` |

Conclusion boundary: F1/F4 do not show stable incremental AUROC over F5 on finite-label CyberMetric MCQ. Site transfer remains unmeasured because the pair-quality gate was not met. No hallucination-reduction, answer-accuracy, or intervention claim is made.

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


## Sprint 4B-2 small-model smoke artifacts

Directory: `outputs/logs/sprint_4B_2_small_model_smoke/` (gitignored).

| file | exists | tracked | note |
| --- | --- | --- | --- |
| preflight_report.md | yes | no | worktree/model/dataset spot-check |
| ab_question_manifest.jsonl | yes | no | 32 train-split questions, seed 4242 |
| trace_manifest_cond_raw.jsonl | yes | no | 128 traces, degenerate flags per trace |
| trace_manifest_cond_chat.jsonl | yes | no | 128 traces, all bare-letter single-token |
| prompt_ab_report.json | yes | no | chat 0.0 vs raw 0.1875; winner=chat; admits 4B-3 |
| f5_feature_plumbing_report.json | yes | no | 5 features finite, tier fields, assertion 32/32 |
| option_position_model_bias_report.json | yes | no | no severe warning |
| review_gate_small_model_smoke.md | yes | no | 17 items answered |

Key conclusions: chat prompt eliminates the raw-completion degeneration (0.0 vs 0.094+0.094); two instrumentation bugs (BatchEncoding duck-typing in locate; bare-vs-space option token form) found by dry runs and fixed with regression tests before the official run; chat completions carry zero reasoning text (has_reasoning_text 0.00) - F2 substrate decision deferred to 4B-3 with a gated reasoning-forcing mini-test. No kill bar declared. `probe_trained=false`, `steering_continued=false`.

## Sprint 4B-3 full F5 baseline artifacts

Directory: `outputs\logs\sprint_4B_3_full_f5_baseline_and_site_transfer` (gitignored).

Required artifacts: preflight_report.md, trace_sampling_manifest.jsonl, reasoning_substrate_report.json, f5_baseline_report.json, option_position_bias_report.json, high_risk_case_report.jsonl, low_risk_wrong_case_report.jsonl, correct_wrong_pair_manifest.jsonl, site_transfer_check_report.json, equivalence_spot_check_report.json, review_gate_full_f5_baseline_and_site_transfer.md. module_patch_fidelity_report.json is present only if Stage E executed.
