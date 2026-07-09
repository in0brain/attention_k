# 实验进度记录：Reasoning-Aware Attention Guidance

## Current Status Update: Sprint 4B CyberMetric Smoke Baseline

Sprint 4B smoke is completed on CyberMetric with the minimal chain requested: raw local/gitignored CyberMetric data -> canonical option-letter schema -> label-token proxy -> option-position bias audit -> trace sampling -> F5 output-level baseline -> review gate. This was a smoke run only: no full primary run, no probe training, no steering/nudge, no LoRA/finetuning, no full Sprint 3C, and no 2000-scale run. Gold labels were used only as eval labels, not inference features.

Run setup: `--dataset cybermetric --primary-questions 30 --samples-per-question 2 --temperature 0.7 --max-new-tokens 128 --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke --overwrite`. The default local model fallback `D:/models/Qwen2.5-7B-Instruct` was used. Raw cyber data stayed under `data/raw/cyber/` and remains gitignored; smoke outputs stayed under `outputs/logs/` and remain gitignored.

Smoke result: 30 canonical questions, 90 traces, parse_failure_rate 0.100, sampled wrong_rate 0.315, and 7 same-question correct/wrong pairs. Option-position audit found no severe warning; gold distribution A/B/C/D = 6/7/10/7 and majority-position baseline accuracy = 0.333. Label space passed single non-whitespace token and distinct token-id checks for A/B/C/D.

F5 baseline result on 28 scored greedy examples: label-margin risk AUROC/AUPRC 0.741/0.553; label entropy 0.748/0.558; full entropy 0.374/0.244; self-consistency risk 0.823/0.497; fixed non-trained F5 combination 0.776/0.532. These are smoke-scale estimates only, with wide bootstrap CIs in `f5_baseline_report.json`.

Stage 5 site-transfer was skipped as required because the gate failed: `num_questions_with_correct_and_wrong >= 20` was false (7 pairs). `site_transfer_check_report.json` records `status="skipped"` and the skipped reason. No module patching, steering, or site-transfer diagnostic was run.

Files changed for this smoke implementation: `src/recover_attention/cyber_data.py`, `src/recover_attention/domain_label_proxy.py`, `scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py`, `tests/test_cyber_data.py`, `tests/test_domain_label_proxy.py`, `PROGRESS.md`, `docs/progress/sprint_4_history.md`, and `docs/progress/sprint_4_artifact_manifest.md`.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_cyber_data.py tests/test_domain_label_proxy.py -q
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py --dataset cybermetric --primary-questions 30 --samples-per-question 2 --temperature 0.7 --max-new-tokens 128 --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke --overwrite
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Boundary flags: `probe_trained=false`, `steering_continued=false`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`, `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`.

Next: if continuing to 4C, treat the fixed F5 smoke AUROC as a preliminary baseline only; run a larger non-training primary baseline before any feature-family bake-off. Do not infer cyber semantic signal from option-letter position patterns.

## Current Status Update: Post-4A Mainline Refinement and Sprint 4B Ready

Current mainline: `CYBER_HALLUCINATION_CONTROL_PLAN.md` now supersedes
`CYBER_DIRECTION_PROBE_PLAN.md`. Sprint 4 mainline is domain-calibrated
hallucination detection + gated closed-form intervention, not a supervised
direction-probe/controller mainline. The older 4A direction-probe reset record
is retained below for history only.

Sprint 4B should start from dataset/schema/F5 baseline work, not probe training
or steering. This documentation guardrail cleanup tightens the 4B task card
around parameterized model-path resolution, semantic-label preservation,
option-position bias auditing, and gated optional site-transfer execution.
No model was called, no data was downloaded, no probe was trained, and no
steering or 4B experiment was run.

Merge guardrail repair: after resolving the remote/local merge, the 4B task card
was repaired to restore semantic-label preservation, option-position bias
auditing, and gated optional site-transfer. No model call, data download, probe
training, steering, or 4B experiment was run.

Files changed in this documentation cleanup: `PROGRESS.md`,
`docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md`,
`docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md`,
`docs/progress/sprint_4_history.md`, and
`docs/progress/sprint_4_artifact_manifest.md`.

Required lightweight check:

```bash
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Checks: `tests/test_dataset_audit.py tests/test_stage_summary.py` passed 14 tests.

Next: execute Sprint 4B only from dataset/schema/F5 baseline preflight. Do not
start probe training, steering, or Stage 5 site-transfer unless its gates pass.

Sprint 4B dataset download audit (documentation/data-prep only): downloaded and
inspected raw CyberMetric, SecQA, and CS-Eval source files under `data/raw/cyber/`
and wrote audit artifacts under `outputs/logs/sprint_4B_dataset_download_audit/`.
CyberMetric is the recommended primary raw source for the next 4B smoke because
it is MCQ-like with four options and gold answers; SecQA is a small fallback /
held-out source; CS-Eval was source-inspection-only in this pass. No model call,
completion generation, F5 baseline, probe training, steering, or site-transfer
was run.

Dataset audit commands/checks:

```bash
conda run -n recover_attention python scripts/sprint_4B_download_and_audit_cyber_datasets.py --output-dir outputs/logs/sprint_4B_dataset_download_audit
conda run -n recover_attention python -m py_compile scripts/sprint_4B_download_and_audit_cyber_datasets.py
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Checks: audit script completed; script syntax check passed; lightweight pytest
passed 14 tests.

Raw cyber datasets are local/gitignored artifacts and should be regenerated by
the download audit script rather than committed.

## Current Status Update: Sprint 4A Cybersecurity Direction-Probe Mainline Reset

Sprint 4A resets the project mainline from unsupervised span-guided steering to cyber-domain supervised direction probing. This is a documentation, requirements, and code reuse audit sprint only: no new model calls, no probe training, no steering, no patch/nudge experiment, no full Sprint 3C, and no 2000-scale rerun.

Reason for reset: Sprint 3A-3C show that blind span steering is unstable. Attention-bias steering and span residual injection did not turn span relevance into selective answer improvement. Sprint 3C-1 / 3C-2 preserve the positive mechanism finding: the final-answer readout MLP is a valid causal write site and its effective oracle direction aligns with gold label/token unembedding. Sprint 3C-3 / 3C-4A limit the readout/detection claim: direct MLP readout and approximate J-lens are diagnostic but do not beat final-logits margin and do not justify returning to blind steering.

New mainline: in cybersecurity tasks with structured labels, train or evaluate a domain-supervised direction probe/controller that maps reasoning-aware features to label or correction directions:

```text
span / activation / trace -> correct direction / label direction
```

Files changed: `docs/codex_tasks/sprint_4A_cyber_direction_probe_mainline_reset.md`, `docs/reference/STORY.md`, `docs/reference/CYBER_DIRECTION_PROBE_PLAN.md`, `docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md`, `docs/progress/sprint_4_history.md`, `docs/progress/sprint_4_artifact_manifest.md`, and `PROGRESS.md`. Gitignored outputs were written under `outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/`.

Inputs: Sprint 3C-1 / 3C-2 / 3C-3 / 3C-4A task cards, progress records, artifact manifest, and source modules for causal tracing, MLP readout direction, MLP readout attribution, answer proxy metrics, and approximate J-lens readout.

Outputs: Sprint 4A task card, cyber direction-probe plan, code reuse audit, Sprint 4 history, Sprint 4 artifact manifest, preflight report, and review gate.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
conda run -n recover_attention python -m pytest tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py -q
```

Checks: `tests/test_dataset_audit.py tests/test_stage_summary.py` passed 14 tests; `tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py` passed 15 tests.

Boundary flags: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`, `steering_continued=false`, `domain_supervised_probe_planned=true`.

Next: Sprint 4B Cyber Dataset Selection and Domain Schema Implementation. Do not train a probe until a structured cyber dataset, label space, leakage boundary, and domain answer proxy are defined.

Post-4A plan refinement (2026-07-09, documentation only): after reviewing Sun et al. 2026 (reasoning trajectories), INSIDE, LM-Polygraph, and two AAAI'26 causal-tracing hallucination papers, the Sprint 4 mainline was refined from "supervised direction probe / controller" to "domain-calibrated hallucination detection + gated closed-form intervention". Key changes: the trained vector-controller and probe-guided steering routes are dropped (fine-tuning-adjacent; instance-level answer steering remains structurally blocked per 3C-2); the deliverable is a five-feature-family detection bake-off (static site features, anchor-free trajectory transitions per Sun et al., INSIDE-style multi-sample consistency, exact J-lens label projections — cheap in a finite label space via per-label VJPs — against the mandatory final-logits baseline suite), followed by selective prediction / conformal abstention, with intervention gated behind an incremental-AUROC kill gate. See `docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md` (supersedes `CYBER_DIRECTION_PROBE_PLAN.md`, kept for history) and STORY.md sections 19-21. Sprint 4B scope is unchanged in spirit (dataset + label space + baselines) with two additions: option-letter label space (ATT&CK ids would reproduce the 3C-0 leading-token collision) and a cheap 3C-1 site-transfer check at the label-readout position.

## Current Status Update: Sprint 3C-4A Approximate J-lens Readout Sanity Check

Sprint 3C-4A is completed as a small readout-method sanity check. It reused the 34 Sprint 3C-0-Fix corrected pairs and the Sprint 3C-3 attribution substrate, recaptured answer-readout MLP outputs at layers 20 and 24, and compared direct logit-lens readout (`m @ W_U`) against a directional finite-difference approximate J-lens estimate (`delta_logits / epsilon`) for epsilons 0.01, 0.03, and 0.1. This is not full Workspace J-lens, not steering, not training, not full Sprint 3C, and not a 2000-scale run.

Core result: Case C. Direct and approximate readouts are weakly aligned under this check. At the primary cell `L24|epsilon=0.03`, top-1 match rate is 0.074, top-10 overlap is 0.206, and Spearman over the number-like token subset is 0.122. Approximate J-lens does not improve the 3C-3 diagnostic judgement: direct readout AUROC is 0.653, approximate J-lens AUROC is 0.637 at epsilon 0.03 and 0.648 at epsilon 0.1, while final-logits margin remains stronger at AUROC 0.712.

Interpretation: keep the 3C-1 / 3C-2 mechanistic finding that the final-answer readout MLP site matters, but limit readout/detection claims. Approximate J-lens does not rescue direct unembedding readout into a stronger detector and does not justify returning to steering.

Files changed: `src/recover_attention/approx_j_lens_readout.py`, `scripts/sprint_3C_4A_approx_j_lens_readout_sanity_check.py`, `tests/test_approx_j_lens_readout.py`, `PROGRESS.md`, `docs/progress/sprint_3_history.md`, and `docs/progress/sprint_3_artifact_manifest.md`. Outputs were written under `outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/` and remain gitignored.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_approx_j_lens_readout.py -q
conda run -n recover_attention python scripts/sprint_3C_4A_approx_j_lens_readout_sanity_check.py --input-dir outputs/logs/sprint_3C_3_mlp_readout_attribution_probe --fix-input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck --output-dir outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check --layers 20 24 --epsilons 0.01 0.03 0.1 --top-k 20 --max-pairs 34 --overwrite
conda run -n recover_attention python -m pytest -q
```

Checks: targeted pytest 6 passed; full pytest 657 passed, 2 skipped. Review gate `outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/review_gate_approx_j_lens_readout_sanity_check.md` records `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`, and `steering_continued=false`.

Workspace note: `docs/codex_tasks/sprint_3C_4A_approx_j_lens_readout_sanity_check.md` had pre-existing `AM` status before this sprint (index empty file, working tree populated). It was preserved and not rewritten.

Next: mechanism write-up or detection-only follow-up if needed. Do not run 2000, do not enter full Sprint 3C, and do not resume steering from this result.

## Current Status Update: Sprint 3C-2 MLP Readout Direction Analysis and Donor-Free Nudge

Sprint 3C-2 is completed as an MLP readout-direction analysis. It is not full Sprint 3C, not a 2000-scale rerun, not training, not a deployable steering method, and not evidence of answer accuracy improvement or hallucination reduction. Teacher-forced (plus a minimal first-step) single-forward proxy only.

Question: 3C-1 localized the selective, low-harm answer-readout write to the MLP output. This sprint asks whether that write is a stable direction that (a) aligns with the gold-vs-wrong unembedding and (b) survives as a **donor-free** direction (without the correct donor).

Setup: reused the 34 3C-0-Fix pairs (no re-sampling). Captured MLP output at the answer-readout position for correct/wrong traces; built the correct−wrong delta at layers {20,24}. Donor-free directions constructed **leave-one-out** (mean_delta, pc1_delta, negative, shuffled excluding the eval pair) plus gold_unembed (eval-only, per-pair), random, zero. Nudge: `mlp_output[readout] += α·‖mlp_output[readout]‖·unit(direction)`, α∈{0.05,0.1,0.2,0.4}. 1904 nudge cells. Metric: corrected answer-sequence clean_direction + a first-step (prefix-only) generation-style proxy. New: `src/recover_attention/mlp_readout_direction.py`, `scripts/sprint_3C_2_mlp_readout_direction_analysis.py`, `tests/test_mlp_readout_direction.py` (9 tests).

Core result — **mixed: a clean mechanistic finding, but no robust donor-free direction**:
- Geometry: the correct−wrong MLP delta has a very stable principal axis (PC1 leave-one-out cosine 0.99 @L20, 0.98 @L24; PC1 explained variance ~0.12–0.13, ≈4× uniform). A shared direction exists.
- Unembedding alignment: at **L24** the per-pair delta aligns with the gold-vs-wrong unembedding direction, cosine +0.091 CI95 [+0.064, +0.117] (stable positive); at L20 it does not (~0). So the L24 MLP write is (weakly but stably) toward the gold token's unembedding.
- gold-unembedding direction (**eval-only**, uses the gold answer): a strong, low-harm, first-step-surviving steering axis — L24 clean +0.34→+2.93 across α (gold up, wrong down), harm ≤0.12, stable at every α; beats random by +0.93 CI95 [+0.73, +1.13]. This is an oracle upper bound, not deployable.
- Donor-free mean_delta direction: at L24 stably positive vs no-op (clean +0.13/+0.09/+0.15/+0.31, harm 0.0) and beats random only **marginally** (+0.062 CI95 [+0.0005, +0.130]); it does **not** stably beat its own negation (+0.062 CI95 [-0.029, +0.166]) and barely survives the first-step check (stable only at α0.4). At L20 it fails. PC1 direction does not beat random.
- Control note: the `shuffled` control is **degenerate here** — shuffling the correct/wrong pairing and then averaging returns the same mean direction (`mean_delta − shuffled ≡ 0`), so shuffle is uninformative for a global mean direction (it is only meaningful per-pair). The meaningful controls for a global direction are random and negative.

Reading: the answer-readout MLP write is a real, stable, low-rank direction aligned (at L24) with the gold-token unembedding — mechanistically this is "the MLP nudges the residual toward the gold answer token." But the effective axis essentially **requires knowing the gold answer** (gold_unembed works; a gold-free mean/PC1 direction is only marginally effective and not robust). We do NOT have a deployable donor-free steering handle. This echoes the project's recurring theme: the signal is strong for attribution/detection (which token the MLP pushes toward) but not yet a blind steering control.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_mlp_readout_direction.py -q
conda run -n recover_attention python scripts/sprint_3C_2_mlp_readout_direction_analysis.py \
  --input-dir outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing \
  --fix-input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_2_mlp_readout_direction_analysis \
  --layers 20 24 --alphas 0.05 0.1 0.2 0.4 \
  --directions mean_delta pc1_delta gold_unembed random shuffled negative zero --overwrite
conda run -n recover_attention python -m pytest -q
```

Checks: targeted pytest 9 passed; full pytest 642 passed, 2 skipped. Review gate `outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/review_gate_mlp_readout_direction_analysis.md` (MLP readout carries causal info aligned with gold unembedding, but current extraction is insufficient for clean donor-free steering).

Boundary: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.

Next: either (a) improve gold-free direction extraction (per-layer whitening / a linear probe trained on more pairs / a Fisher-style contrast) to close the gap to gold_unembed, or (b) accept that the effective axis is the gold-unembedding direction and pivot this into an attribution/detection use (report which answer token the readout MLP is being pushed toward) rather than blind steering. Still no 2000 / full 3C / generation-accuracy claim.

## Current Status Update: Sprint 3C-1 Final-Answer Compression Value/MLP Causal Tracing

Sprint 3C-1 is completed as a module-level causal tracing sprint. It is not full Sprint 3C, not a 2000-scale rerun, not training, not a deployable steering method, and not evidence of answer accuracy improvement or hallucination reduction. Teacher-forced single-forward proxy only.

Question: 3C-0-Fix found a correct-run repair signal at the final-answer readout position but could not establish which module writes it. This sprint decomposes the whole-residual patch at the readout position (answer-number prefix token) into per-module writes — self-attention output, MLP output, and whole-residual output — interpolation-patches each with correct-run donor activation, and tests donor-specificity and site-specificity with harm control.

Setup: reused the 3C-0-Fix corrected pairs (34), no re-sampling. Captured per-module outputs via forward hooks; interpolation patch `(1-α)·recipient + α·donor` at the readout position. Grid: modules {attention_output, mlp_output, residual_output} × layers {16,20,24} × α {0.25,0.5,0.75,1.0} × 6 conditions (no_patch, correct_donor, random_donor, same_trace_random_position, same_pair_wrong_position, wrong_donor_self). 7308 forward rows. Metric: corrected answer-sequence clean_direction (3C-0-Fix proxy). New: `src/recover_attention/module_causal_tracing.py`, `scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py`, `tests/test_module_causal_tracing.py` (7 tests).

Fidelity: hooks registered/triggered/removed = 1.0. `wrong_donor_self` is not an exact no-op (donor captured from full-trace forward, injected into shorter prefix+answer forward → attention kernels not bit-identical across lengths), but its floor is small: self mean|clean| 0.087 vs correct-donor 1.632 (ratio 0.05), and it cancels in the specificity contrasts (all donors share it). `residual_output|L24|α1.0` reproduces 3C-0-Fix's whole-residual result (+12.67, harm 0.88) — a consistency check.

Core result — **Case A: the answer direction at the final-answer readout is written primarily by the MLP, and that write is donor-specific, site-specific, and harm-controllable**:
- Donor-specificity (correct − random donor, paired) stable positive for all three modules: attention +0.093 CI95 [+0.014,+0.181], mlp +0.141 [+0.044,+0.252], residual +1.703 [+1.24,+2.19].
- Site-specificity (correct at readout − correct at random position, paired): **mlp_output +0.353 CI95 [+0.242,+0.476] stable positive** — the only per-module stable site signal. attention_output -0.006 [-0.076,+0.062] (not site-specific); residual_output -0.547 [-1.13,+0.064] (not site-specific).
- So attention output moves the answer generically (donor-specific but not site-specific); whole residual is largest but non-selective and high-harm; **only the MLP output passes both donor- and site-specificity.**
- Harm-controlled MLP regime (α sweep, L24): clean +0.319/+0.590/+0.943/+1.290 as α 0.25→1.0, harm 0.06/0.18/0.24/0.24; gold +0.327/+0.584/+0.912/+1.228, wrong ~flat. Best low-harm selective cells: `mlp_output|L24|α0.25` (clean +0.319, gold +0.327, wrong +0.040, harm 0.06) and `mlp_output|L20|α0.25` (clean +0.131, harm 0.06).

Reading: 3A/3B/3C-0 negatives are now explained — span-level and whole-residual interventions are too coarse; the selective, low-harm causal handle is the MLP write at the answer-readout position. This localizes a mechanism under a teacher-forced proxy; it is NOT an accuracy or generation result.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_module_causal_tracing.py -q
conda run -n recover_attention python scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py \
  --input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing \
  --layers 16 20 24 --modules attention_output mlp_output residual_output --alphas 0.25 0.5 0.75 1.0 --overwrite
conda run -n recover_attention python -m pytest -q
```

Checks: targeted pytest 7 passed; full pytest 633 passed, 2 skipped. Review gate `outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/review_gate_final_answer_compression_tracing.md` (Case A, selective low-harm site found).

Boundary: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.

Next (Case A): Sprint 3C-2 — analyze the MLP readout write direction at `mlp_output|L24|α0.25` (and L20): is there a low-rank/interpretable "answer" direction; does a low-harm α-scaled MLP-output nudge survive autoregressive generation; establish a harm-controlled steering probe before any generation-level correctness eval. Still no 2000 / full 3C.

## Current Status Update: Sprint 3C-0-Fix Answer-Position Proxy Recheck

Sprint 3C-0-Fix is completed as a low-cost proxy repair and recheck of Sprint 3C-0. It is not full Sprint 3C, not a 2000-scale rerun, not training, not a new steering mechanism, and not evidence of answer accuracy improvement or hallucination reduction. Teacher-forced single-forward proxy only.

Motivation: 3C-0's `clean_direction` was measured with two proxy weaknesses — logits read at the last trace token (after the answer was already written) and only the answer's first digit token. This sprint reads logits at the **answer slot** (right before the final answer number) and scores the **full numeric answer sequence** conditional logprob, gold vs wrong, under the same wrong prefix.

Setup: reused the 3C-0 `correct_wrong_pair_manifest.jsonl` (no re-sampling). Robust answer-span extraction (`#### N` → answer phrase → fallback last number excluding list ordinals → parse failure) re-validated pairs: 34/35 kept (1 dropped, recipient no longer a valid non-gold answer). Residual-replace patching (alpha=1.0) at layers [16, 20, 24]; primary positions = operator / intermediate number (strictly before the answer slot); control = final_answer_position redefined as the answer-number prefix position. 1104 forward rows. Extraction success 1.0, parse-failure 0.0; but fallback_last_number touched 0.80 of pairs (many traces don't emit `#### N`) — a noise caveat.

New module/script/tests:
- `src/recover_attention/answer_proxy_metrics.py`: `extract_final_answer_span`, `answer_token_ids`, `token_index_for_char_start`, `sequence_logprob_at_answer_slot`, `compute_corrected_clean_direction`, `paired_bootstrap_delta`.
- `scripts/sprint_3C_0_fix_answer_proxy_recheck.py`, `tests/test_answer_proxy_metrics.py` (9 tests).

Core results — the corrected proxy **revises 3C-0's Case C to Case B**:
- Reasoning-step correct→wrong patch: overall corrected clean_direction = **+0.120**, CI95 [+0.053, +0.197] (stable vs no_patch; gold +0.121, wrong +0.005). But **non-selective**: vs random donor +0.103 CI95 [-0.017, +0.229] (not stable); vs same-trace random position -0.020 CI95 [-0.183, +0.124] (not stable). So reasoning-step patching helps a little over doing nothing, but no more than a random donor or random position — mild generic bias, not a role-specific causal fix.
- Final-answer read-out position (answer-slot prefix): **large and partly selective** — gold **up** and wrong **down** (L16 clean +1.90 / L20 +3.64 / L24 +12.67; gold +1.9/+3.1/+9.1, wrong -0.5/-0.5/-4.3). vs no_patch +6.12 CI95 [+4.62, +7.79] (stable); vs random donor **+3.13 CI95 [+1.87, +4.46] (stable positive)**; vs same-trace random position -1.24 CI95 [-3.02, +0.61] (not stable). Harm rises with layer: 0.41 → 0.53 → 0.88.
- Reading: correct-run activation **does** carry answer-fixing information (3C-0's flat null was partly a proxy artifact), but the selective signal lives at the **answer read-out / compression stage**, not explicit reasoning steps — consistent with 3A/3B's answer-position focus, now showing selectivity under the corrected full-sequence proxy. It is "where + same-question correct context" rather than a precisely localized donor-specific switch (same-trace random position not beaten), and it is harm-laden at high layers.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_answer_proxy_metrics.py -q
conda run -n recover_attention python scripts/sprint_3C_0_fix_answer_proxy_recheck.py \
  --input-dir outputs/logs/sprint_3C_0_correct_wrong_activation_patching \
  --output-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck --layers 16 20 24 --overwrite
conda run -n recover_attention python -m pytest -q
```

Checks: targeted pytest 9 passed; full pytest 626 passed, 2 skipped. Review gate `outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/review_gate_answer_proxy_recheck.md` (Case B).

Boundary: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.

Next: per Case B, do activation patching / causal tracing around the **final-answer compression** stage (value/MLP write path at the answer read-out position), not span-level residual injection and not reasoning-step residual replace. Establish donor-specificity (beat same-trace random position) and a harm-controlled regime before considering any generation-level eval.

## Current Status Update: Sprint 3C-0 Correct-vs-Wrong Activation Patching

Sprint 3C-0 is completed as a diagnostic correct-vs-wrong activation patching sprint at reasoning-step positions. It is not full Sprint 3C, not a 2000-scale rerun, not training, and not evidence of answer accuracy improvement or hallucination reduction.

Preflight cleanup (done):
- The stale Sprint 3A-0 top current-status block was already corrected to point at Sprint 3C-0.
- Sprint 3A-0, Sprint 3A-1, and Sprint 3B-0 remain completed historical diagnostics.
- `outputs/` is gitignored, so 3A-1 / 3B-0 aggregate reports exist locally but are untracked; recorded them in `docs/progress/sprint_3_artifact_manifest.md`.
- 3B-0 boundary restated: it negated only same-run span residual deviation → final answer-position injection → single-forward gold-token proxy; it did **not** negate correct-run activation patching, reasoning-step patching, value/MLP tracing, or autoregressive steering.

Setup: 100 questions × 6 samples (temp 0.7, max_new_tokens 160) → 600 traces → 35 same-question correct/wrong pairs. Residual-replace patching (alpha=1.0) at layers [16, 20, 24], role-to-role matched positions {operator, intermediate number, final answer number, final answer position(control)}. 2085 forward rows. Metric: teacher-forced gold-minus-wrong first-token logprob delta (`clean_direction_score`), gold answer eval-only.

Execution finding — metric bug fixed before the reported run:
- First run gave `clean_direction_score ≡ 0` for all rows: Qwen2.5 tokenizes `" 42"` as `[space, '4', '2']`, so `first_token_id(" "+answer)` returned the constant space token (id 220) for every answer, forcing `gold_first_token == wrong_first_token` in all 35 pairs.
- Fixed `first_token_id` to strip the leading space and return the leading digit token (31/35 pairs discriminative), added a regression test, reran. Numbers below are from the corrected run.

Core results (corrected run):
- Hook fidelity clean: registered/triggered/removed = 1.0, non-target contamination check pass, mean patch_delta_norm ≈ 104.2.
- Overall `mean_clean_direction_score = -0.0185`, CI95 `[-0.113, +0.080]` → **not stably positive**.
- gold and wrong first-token logprobs rise together (+0.735 vs +0.754): the non-selective-perturbation signature the task warned about.
- Control comparisons (question-paired bootstrap, clean_direction_score):
  - correct_to_wrong − no_patch: -0.018, CI95 [-0.119, +0.087] (indistinguishable).
  - correct_to_wrong − random_donor_patch: +0.092, CI95 [-0.012, +0.207] (**not stable**).
  - correct_to_wrong − same_trace_random_position_patch: -0.227, CI95 [-0.360, -0.110] (**stably worse**).
  - correct_to_wrong − correct_activation_patch: -0.029, CI95 [-0.159, +0.103].
- Best clean-direction cell is L20 `final_answer_position` (+0.110) — a **control** position with ~0.94 harm and gold/wrong moving equally (+2.72/+2.61); reasoning-step primaries are ~0 with near-zero harm. No reasoning-step position × layer beats controls.
- Verdict: **Case C** — correct-run reasoning-step activation patching does not reveal a selective causal state under this teacher-forced proxy.

Commands:
```bash
conda run -n recover_attention python -m pytest tests/test_activation_patching.py -q
conda run -n recover_attention python scripts/sprint_3C_0_correct_wrong_activation_patching.py \
  --primary-questions 100 --samples-per-question 6 --temperature 0.7 --max-new-tokens 160 \
  --layers 16 20 24 \
  --position-types generated_operator_token generated_intermediate_number_token generated_final_answer_number final_answer_position \
  --output-dir outputs/logs/sprint_3C_0_correct_wrong_activation_patching --overwrite
conda run -n recover_attention python -m pytest -q
```

Checks:
- Targeted pytest: 7 passed. Full pytest: 617 passed, 2 skipped.
- Review gate: `outputs/logs/sprint_3C_0_correct_wrong_activation_patching/review_gate_correct_wrong_activation_patching.md` (19/19 questions answered).

Boundary:
- `ready_for_2000_rerun=false`.
- `do_not_enter_full_sprint_3C=true`.
- `hallucination_reduction_proven=false`.
- `answer_accuracy_improvement_proven=false`.

Next (per Case C):
- Move to value/MLP causal tracing at reasoning steps (localize the write path rather than replacing the whole residual), or pause steering and return to detection/diagnosis. Do not scale up, do not re-tune attention-bias or residual span injection. See "下一步建议" in the session and `docs/progress/sprint_3_history.md`.

## Current Status Update: Sprint 2K-W Answer-Position Output-Effect

Sprint 2K-W is completed as an answer-position output-effect remeasurement sprint, not a 2000-scale rerun and not Sprint 3A.

Completed:
- Added a leakage-audited answer-position output-effect backend and CLI.
- Reran only original/masked answer-position logits over the existing 4935-span 2J-Fix / 2K / 2K-V matrix.
- Generated the required 2K-W reports under `outputs/logs/sprint_2K_W_answer_position_output_effect_500/`.
- Preserved the pre-existing `AM` task card state for `docs/codex_tasks/sprint_2K_W_answer_position_output_effect.md`.

Core results:
- `num_feature_records=4935`, `num_score_records=4935`, feature leakage audit passed.
- AUC: surface 0.5120, attention-only 0.5885, prompt-final output-effect 0.5545, response-position output-effect 0.6371, attention x response-position output-effect 0.6470.
- Response-position output-effect is stably stronger than prompt-final output-effect: delta +0.0803, CI95 [+0.0314, +0.1321].
- Response-position output-effect alone is not stably stronger than attention-only: delta +0.0103, CI95 [-0.0483, +0.0723].
- Attention x response-position output-effect is stably stronger than attention-only: delta +0.0379, CI95 [+0.0090, +0.0709].
- Best AUC formula is `I_mean_attention_response_output` at 0.6534.
- Review gate passed 11/11, but the sprint still records `ready_for_2000_rerun=false` and `do_not_enter_sprint_3A=true`.

Commands:
```bash
conda run -n recover_attention python scripts/sprint_2K_W_answer_position_output_effect.py --output-dir outputs/logs/sprint_2K_W_answer_position_output_effect_500 --overwrite --report-every 50
conda run -n recover_attention python -m pytest tests/test_answer_position_output_effect.py -q
conda run -n recover_attention python -m pytest -q
```

Checks:
- Targeted pytest: 4 passed.
- Full pytest: 599 passed, 2 skipped.

Remaining issues:
- Do not enter 2000 rerun or Sprint 3A directly from this sprint.
- Response-position output-effect improves the measurement point, but the output-only signal is not stably above attention-only.
- Formula validation should stay between 2K-W and any steering implementation.

Next:
- Run a formula-validation / 3A-0 smoke-test sprint only; no full Sprint 3A attention steering yet.

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
Sprint 3B-0 已完成：Representation-Level Oracle Intervention Diagnostic（residual injection，120 题 beta sweep）。

3A-1 否定了 attention-logit bias 通道后，本轮换机制：把 selected span 的 residual deviation 注入 answer-position 残差流（inj=beta·‖ans_residual‖·unit(span_dev)，按残差范数缩放使 sweep 可测），检验 oracle（on-path）span 是否比 random 更能把答案推向 gold token。非 full 3B / 非 2000 / 非训练；gold/oracle 仅 eval-only。

核心结果（跨通道决定性负面）：
- residual channel 比 attention 强得多且可靠：非 oracle answer-position JS 随 beta 0.0013→0.098（3A-1 attention 只有 1e-5→0.02）；排除「干预太弱」。
- 但 oracle 并不比 random 有选择性：oracle−random 配对 bootstrap 在 5 档 beta 全部不 stable-positive（CI 均含 0）；regime beta=0.05 各 selector 的 gold-logprob delta 近乎相同（random +0.327 / oracle +0.328 / ...），且随 beta 对所有 selector 同步上升。harm 随 beta 陡升（β=0.8 达 61–68%）。
- 结论：注入 **正确** span 的表示不比注入 **随机** span 更能改善答案——两个独立通道（attention-bias 与 residual injection）同型失败，说明是 **span-level / answer-position / single-forward 干预与任务不匹配**，与通道无关。span-importance 对 detection 真实（2K-V AUC~0.588），但无法转成 answer-position 的 causal steering 杠杆。对应 card 的 Situation C。

下一步建议：不扩大；转更细粒度——reasoning-step（数字被消费那一步）干预 / 从正确 run 做 answer-position activation patching / value-MLP causal tracing / 重审任务；或暂停 steering 回到 detection。ready_for_2000_rerun=False、do_not_enter_full_sprint_3B=True，不声称 accuracy/hallucination 改善。

正式产物：outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/*（含 review_gate_representation_level_oracle_diagnostic.md）；详见 docs/progress/sprint_3_history.md。

---

Sprint 3A-1 已完成：Controlled Attention Guidance on 500 Cases（increase-only，120 题 λ sweep）。

首次进入 attention steering 的受控诊断（非 full 3A、非 2000 rerun、不声称 hallucination/accuracy 改善）。复用 3A-0 attention-bias hook，修正 3A-0 oracle inconclusive 的两个根因：(1) oracle 指标改为 answer-directed（eval-only：steering 是否把 answer-position 分布推向 gold answer 首 token 比 random 更多），(2) λ sweep{0.2,1.0,4.0} 找到「输出可测」regime（3A-0 λ=0.2 下 JS~1e-5 近似 no-op）。

核心结果（决定性负面）：
- λ=4.0 是可测 regime（非 oracle JS 从 3e-5→0.020；mass delta ~0.2–0.3）。
- answer-directed oracle 诊断（gold-first-token logprob delta，regime λ=4.0）：random +0.665 / surface +0.829 / oracle +0.593 / attention +0.569 / attention_only +0.442。所有 selector 都抬高 gold token 概率，但 **oracle 不优于 random**（oracle−random 配对 mean −0.053，CI [−0.223,+0.135] 不显著；3 档 λ 均不 stable）。harm 20–27%。
- 结论：把注意力加大到 **正确** span 并不比加大到 **随机** span 更能改善答案——高 λ 的 gold-logprob 普升是「泛化锐化」的非选择性 artifact + destabilization，非 keyness-driven guidance。**瓶颈在机制本身（additive attention bias 不能把「选对 span」转成「答对」），不在 selector 质量**。

下一步建议：转向 representation-level / value-level 干预，或重审任务/数据集；不建议继续 attention-bias steering，也不建议只提升 selector。ready_for_2000_rerun=False、do_not_enter_full_sprint_3A=True。generation eval（steered autoregressive）本轮 deferred（用 gold-token logprob delta 作 answer-directed proxy）。

正式产物：outputs/logs/sprint_3A_1_controlled_attention_guidance_500/*（含 review_gate_controlled_attention_guidance_500.md）；详见 docs/progress/sprint_3_history.md。

---

Sprint 2K-V 已完成：Signal Role Decomposition and Fusion Error Audit（read-only，4935 spans）。

本轮是 signal role decomposition（不是 scale-up）：read-only 复用 2J-Fix/2K 的 4935-span artifacts，拆解 attention / hidden / output-effect 各自解决哪个子问题、为什么 simple fusion（0.564）低于 attention-only（0.588）。补充：用 2H risk_strength_dataset join 得 382 个带 bucket 的 number span 做真实 fragility AUC（card 假设 hidden=fragility 的验证）。

核心结果（部分推翻 card 假设，诚实记录）：
- keyness AUC：surface 0.512 / hidden 0.468 / attention 0.588 / simple_fusion 0.564 / attention×output 0.586。simple fusion < attention-only 证实。
- **fragility bucket3-vs-1 AUC（2H join，382 spans）：hidden 0.480（≈随机）/ attention 0.715 / output-effect 0.485。**
- 即 **attention 在 keyness 与 fragility 上都最强；hidden 在两者上都弱**。故「hidden=fragility」不成立——simple fusion 低于 attention-only 的根因是 **hidden 是噪声**（conflict 审计 net_hidden_effect=-46 证实），而非混淆 keyness/fragility。
- output-effect 有 modest 互补（net +16，171 D 型修正），但 attention×output 未超 attention-only；单独 output-effect 不稳定（prompt 末 token 测点限制）。
- gated formula F0–F9 无一稳定优于 attention-only。

是否确认 attention=keyness、hidden=fragility：**否**——attention 兼任 keyness+fragility，hidden 两者都弱。simple fusion 拉低 attention 的原因是 hidden 噪声。gated formula 未优于 attention-only。

下一步建议：保留 attention 作第一层 keyness gate；当前构造下不用 hidden；废弃 simple average；先做 answer-position output-effect 复核。ready_for_2000_rerun=False、do_not_enter_sprint_3A=True（oracle 0.9995 vs best 0.588，gap 仍大）。

正式产物：outputs/logs/sprint_2K_V_signal_role_decomposition_500/*（review_gate_signal_role_decomposition.md 等）

---

Sprint 2J-Fix + 2K 已完成：Slot Alignment Repair 与 Leakage-Safe Answer-Effect Keyness（500-case，4935 spans）。

修复 2J-B 的 slot alignment bug（cached original forward 复用第一个 span 的 slot_indices → 约 90% span 的 original 侧 hidden/attention/delta 特征错位），并新增 leakage-safe self-output-effect（模型自身末 token 分布 shift，非 gold）作为 keyness proxy；改用真实 question-grouped bootstrap、AUC-first。

核心纠正：2J-B 的「hidden/attention anti-keyness（AUC<0.5）」结论是 slot alignment bug artifact。修复后 same-question on/off-path AUC（grouped bootstrap vs surface，ci95_low>0=stable）：
- surface 0.512 / hidden_only 0.468 / attention_only 0.588（best，+0.103 CI[+0.062,+0.146] stable）/ hidden+attention 0.564（+0.075 stable）/ oracle 0.9995。
- 即 attention 确实携带 within-question keyness signal，显著优于 surface；hidden/attention 这条线未死。
2K output-effect：J（单独）0.554（+0.033 not stable，受 prompt 末 token 测点限制）；L/M/N（与 fragility 组合）0.58 且 stable。非数字 model-causal keyness：question_target(0.97)>comparison(0.80)>numbers(~0.78)>...>negation(0.62,n=12)。
gate：2J-Fix PASSED 7/7；2K output-effect-only 不稳健、组合稳定。ready_for_2000_rerun=False、do_not_enter_3A=True（oracle gap 0.99 vs 0.59 仍大）。

下一步建议：output-effect 测点改到 answer-eliciting 位置复核 J-alone；做 attention+output-effect 的 formula-validation sprint；仍不进入 2000/3A。

正式产物：outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/*、outputs/logs/sprint_2K_answer_effect_keyness_500/*

---

Sprint 2I-R 已完成：Score Matrix Decomposition and Root-Cause Audit（500-case read-only audit）。

本轮只读 2H-B / 2H-C / 2H-D / 2I 的 500-case 正式产物，构造 score matrix，拆分 keyness / fragility / budget priority，并做 feature leakage audit、公式模拟、bootstrap、top-k failure/success cases 和 root-cause decision table。未重跑 recovery、hidden-state cache、attention cache、probe training 或 2000-scale pipeline；未执行 attention steering，未进入 Sprint 3A。

2I-R 关键结果（477 score-matrix records）：
- score_matrix_feature_audit：149 个 input feature name 审计通过，leaked_input_features=0；risk_strength / fragility_bucket / solution_path_status 仅保留为 diagnostic/evaluation-only labels。
- reasoning_signal_gap：answer_logprob_delta、trajectory_change、cot_path_change、nla_semantic_role 全部缺失；当前 2H/2I 静态特征无法提供真正 reasoning-path evidence。
- same-question ranking diagnostic：当前 500-case matrix 为 477 个 question 各 1 条 scored span，num_questions_with_multiple_scored_spans=0，因此 same-question pairwise ranking 不可计算；不能宣称当前公式解决 same-question ranking。
- formula simulation：没有非 oracle 公式在 top-k precision 与 off-path budget 上稳定优于 current priority；best_non_oracle_formula 保守保留为 A_current_priority。
- review gate：ready_for_2000_rerun=False，do_not_enter_sprint_3A=True。

正式产物：
- outputs/logs/sprint_2I_R_score_matrix_audit_500/score_matrix_dataset.jsonl
- outputs/logs/sprint_2I_R_score_matrix_audit_500/score_matrix_feature_audit.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/keyness_eval_report.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/fragility_eval_report.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/budget_priority_eval_report.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/formula_simulation_report.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/formula_bootstrap_report.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/reasoning_signal_gap_analysis.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/root_cause_decision_table.json
- outputs/logs/sprint_2I_R_score_matrix_audit_500/topk_failure_cases.jsonl / topk_success_cases.jsonl
- outputs/logs/sprint_2I_R_score_matrix_audit_500/review_gate_score_matrix_audit.md

下一步建议：Sprint 2J 需要补 reasoning-path / answer-stability signals，并先构造 multi-span-per-question 的可排名矩阵；gate 通过前仍不做 2000 rerun，不进入 Sprint 3A。

Sprint 2I 已完成：Targeted Attention-Map Feature Cache and Probe（500-case diagnostic）。

在 2H-D 结论（enriched 分类强、但 ordinal ranking 稳健性不足）之后，2I 引入 attention-level pre-recovery 特征：4-bit eager 重跑 original+masked forward（不含 recovered），提取 leakage-free attention 摘要（slot mass / shape / context-to-slot / original→masked delta），新 gate candidate = hidden_plus_attention_pre_recovery。复用 2H-C/2H-D 的 477 trainable subset 与 labels。

2I 关键结果（477 trainable）：
- 分类：hidden_plus_attention macro_f1=0.437（最佳）> hidden_only 0.426 > attention_only 0.397 > surface 0.378；hidden+attention vs surface bootstrap 显著（+0.060，CI [+0.008,+0.117]）。attention 单独已优于 surface。
- 排序：hidden+attention Spearman 0.389 vs hidden_only 0.386（+0.003，不显著）——attention 对 ranking 几乎无增益。
- budget-aware bucket-3：top-10%/20% precision hidden_only 最优，attention 未带来增益。
- 最有用 attention family：context-to-slot(0.164) 与 original→masked delta(0.134)。
- review gate 6/7，ready_for_2000_rerun=False：唯 #7 失败（cand vs surface 排序仅在 expected_bucket 一种打分下成立，未满足 ≥2 方法同向）。

结论：attention 提升的是分类（enriched 早已擅长），而非 2H-D 标记的排序/预算稳健性；未切换 primary/放宽 gate 去凑通过。下一步转向 trajectory-level features 或 hybrid recovery-guided pipeline；不进入 2000-case rerun，不进入 Sprint 3A。

关键工程点：最终层(27) 在 4-bit eager 下 attention 全 NaN，attention 层集合改用 [0,8,16,24]；attention 来自 4-bit 量化模型（与 2G 一致）。

正式产物：
- outputs/logs/sprint_2I_attention_cache_500/*、outputs/logs/sprint_2I_attention_features_500/*（含 review_gate_attention_features.{json,md}）

Sprint 2H-D 已完成：Ordinal Calibration and Budget-Aware Gate Redesign（500-case diagnostic）。

在 2H-C 证明 enriched pre-recovery 特征在 macro_f1 上显著优于 surface、但排序指标略逊之后，2H-D 用三种校准方法（expected-bucket / ordinal-threshold / calibrated regression，全部 per-train-fold 拟合）把 enriched signal 转成更稳定的 ordinal / budget-aware risk score，并重设 gate 以中和 surface 的 bucket_3 flooding 假优势。复用 2H-C 数据集，不重跑 recovery / hidden-state cache。

数据一致性核验：用当前 drift 代码重算 fragility labels 与磁盘 0/500 差异，2H-C 数据与当前代码一致，2H-D 与 2H-C 直接可比。

2H-D 关键结果（500 条）：
- 校准提升了 enriched 绝对排序（Spearman 0.353→≈0.39），与 surface 打平；classifier macro_f1 仍保持 0.426（未回退）。
- enriched vs surface Spearman 逐方法 bootstrap：仅 expected_bucket 显著（+0.076，CI95 [+0.003,+0.159]）；ordinal_threshold 打平（+0.002）；两种回归打分下 surface 略优。即排序优势真实但不稳健（1/4 方法显著、CI 下界紧贴 0）。
- budget-aware bucket-3：top-10% enriched precision 0.688 vs surface 0.708（略输）、top-20% 0.726 vs 0.674（略优），混合无稳定优势。
- review gate（primary=ordinal_threshold，surface-blind 预注册规则）5/8，ready_for_2000_rerun=False；未把 primary 切到唯一能过的 expected_bucket 去凑 gate（避免 p-hacking）。

下一步建议：按任务 fail 分支，补 targeted attention-map cache（span attention mass / entropy / mask-to-span attention）再考虑 trajectory-level features；gate 通过前不进入 2000-case rerun，不进入 Sprint 3A。

正式产物：
- outputs/logs/sprint_2H_ordinal_calibration_500/*（ordinal_calibration_report、bucket3_budget_curve、review_gate_ordinal_calibration 等）

Sprint 2H-C 已完成：Pre-Recovery Feature Enrichment（500-case diagnostic）。

在 2H-B 证明「唯一 leakage-free 特征集（original_masked cosine）无法击败 surface baseline」之后，2H-C 从 2G 已缓存的完整 hidden-state 张量（[layers, tokens, hidden]=(5, seq, 3584)）重新 pooling，构造更丰富的 pre-recovery 特征（delta magnitude / cross-layer stability / span saliency），主 probe 仍严禁 recovered / solution_path / drift / bucket / risk_strength / gold 任意子串进入特征名。

2H-C 关键结果（500 条，复用 2H-B fragility 数据集，未重跑 recovery / hidden-state cache）：
- enriched 特征 79 个（vs 2H-B 的 33 个 cosine），全部 leakage-free，span_available=500。
- 推翻 2H-B 负面结论：hidden_pre_recovery_enriched macro_f1=0.426，显著优于 surface_rule 0.378 / span_type_only 0.339 / hidden_no_recovered 0.318；bootstrap vs surface delta=+0.048（CI95 +0.008..+0.084），vs span_type delta=+0.087（CI95 +0.047..+0.124），均显著；balanced_acc 0.434>0.406。
- 说明模型内部状态确实携带 span_type/surface 之外的 instance-level fragility signal，判别力主要来自 delta magnitude（L2/relative norm）与 cross-layer stability，而非 angle-only cosine。
- 但 review gate 8 项仅过 5 项，ready_for_2000_rerun=False：#4 bucket_3_recall（enriched 0.786<surface 0.958，surface 靠退化的 number→bucket3 泛滥拿高 recall，与最大化 macro_f1 冲突）、#5 排序指标（spearman 0.353<0.387、pairwise 0.665<0.680，真实但小幅劣势）、#7 off-path budget（0.054 vs 0.050，边际打平）失败。
- attention-level 特征在当前 cache 不可得（2G 只缓存 hidden states，无 attention maps），本轮未重跑模型。

下一步建议：ordinal-calibrated 探针（expected-bucket 排序 / 序数回归改善 spearman/pairwise）+ 少量 bucket-3-vs-rest 判别特征；attention 特征需先补 targeted attention cache。gate 通过前不进入 2000-case rerun，不进入 Sprint 3A。

正式产物：
- outputs/logs/sprint_2H_feature_enrichment_500/*（current_feature_audit、pre_recovery_feature_dataset、fragility_probe_enriched_eval_report、review_gate_feature_enrichment 等）

Sprint 2H-B 已完成：Instance-Level Signal Construction（500-case diagnostic）。

在 Sprint 2H（risk_positive 低召回只读审计）确认「2000 规模 weak label 是 span_type 的确定性函数、recovered filler 也是 span_type 的确定性函数」之后，Sprint 2H-B 不再扩大 span_type -> label 映射，而是构造 instance-level supervision（fragility_bucket / risk_strength），并强制主 fragility probe 禁用一切 recovered-channel 特征（否则等价于 2G filler leakage 换皮）。

2H-B 关键结果（500 条，weak-labeled diagnostic）：
- 结构性成功：fragility_bucket 不再是 span_type 纯函数（number 横跨全部 4 桶 23/5/21/116）；ambiguous number 排除训练；bucket 序关系无违反；解题路径数字 on/off 拆分（on_path=1250, off_path=220, ambiguous=226）；distractor 预算占比仅 0.05；模型生成 recovery（qwen3.5:9b, K=3, temp=0.8, 去 span-type 泄漏 prompt）消除模板 filler leakage。
- 但 review gate 第 5 项失败：唯一 gate-eligible 的 leakage-free 特征集（*_original_masked_cosine_*）macro_f1=0.318，低于 span_type_only baseline 0.339 与 surface_rule baseline 0.378，且 bootstrap 对 surface_rule 显著更差（delta −0.060, CI95 −0.100..−0.022）。
- 结论：本轮不能主张 hidden states 已携带 instance-level fragility 信号；可用去泄漏特征过薄，标签多数类仍与 span_type 强相关。ready_for_2000_rerun=False。
- recovered-channel 泄漏诊断：with_recovered − no_recovered = +0.053（弱泄漏，低于 0.1 强泄漏阈值）。

下一步建议：先构造更丰富的 pre-recovery 表征特征（span 处 per-layer hidden states、attention entropy 等），再判断是否 500→2000 rerun；不要现在扩大规模或只调阈值；不要进入 Sprint 3A。

正式产物：
- outputs/logs/sprint_2H_risk_positive_audit/*（2H 只读审计）
- outputs/logs/sprint_2H_instance_signal_500/*（2H-B 全量诊断产物 + review_gate_instance_signal.{json,md}）

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
| Sprint 2G-prep | 完成 | Dataset Source Audit and Import Preparation |
| Sprint 2G-2000 | 完成 | Full-scale Weak-labeled 2000-case Dry Run |
| Sprint 2G-2000-gate | 完成 | Result Review Gate and Final Stage Summary |
| Sprint 2H | 完成 | risk_positive 低召回只读审计 |
| Sprint 2H-B | 完成 | Instance-Level Signal Construction（gate 第 5 项失败，ready_for_2000_rerun=False） |
| Sprint 2H-C | 完成 | Pre-Recovery Feature Enrichment（enriched 显著优于 surface baseline，但 gate 5/8，ready_for_2000_rerun=False） |
| Sprint 2H-D | 完成 | Ordinal Calibration and Budget-Aware Gate Redesign（排序优势真实但不稳健，gate 5/8，ready_for_2000_rerun=False） |
| Sprint 2I | 完成 | Targeted Attention-Map Feature Cache and Probe（attention 提升分类不提升排序稳健性，gate 6/7，ready_for_2000_rerun=False） |

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
pytest（Sprint 3B-0 后最新）: 610 passed, 2 skipped
3B-0 representation-level oracle intervention: residual channel 强且可测（JS 到 ~0.1）；但 oracle 不比 random 有选择性（oracle−random 5 档 beta 全部 CI 含 0）；跨 attention/residual 两通道同型失败 => span-level answer-position 干预与任务不匹配；ready_for_2000_rerun=False、do_not_enter_full_3B=True
新增：src/recover_attention/representation_intervention.py、scripts/sprint_3B_0_representation_level_oracle_intervention.py、tests/test_representation_level_intervention.py
pytest（Sprint 3A-1 后）: 605 passed, 2 skipped
3A-1 controlled attention guidance: λ=4.0 为可测 regime；answer-directed oracle 不优于 random（oracle−random CI 含 0，3 档 λ 均不显著）；结论=attention-bias steering 无选择性 answer 收益，瓶颈在机制本身，ready_for_2000_rerun=False、do_not_enter_full_3A=True
新增：scripts/sprint_3A_1_controlled_attention_guidance.py、docs/progress/sprint_3_history.md
pytest（Sprint 2K-V 后）: 595 passed, 2 skipped
2K-V signal role decomposition: attention keyness=0.588 & fragility=0.715（均最强）；hidden keyness=0.468 & fragility=0.480（均≈随机）；simple fusion 0.564<attention；hidden 是噪声（net_hidden_effect=-46）；gated formula 无一稳定优于 attention-only
新增：scripts/sprint_2K_V_signal_role_decomposition.py
pytest（Sprint 2J-Fix + 2K 后）: 594 passed, 2 skipped
2J-Fix/2K targeted pytest: 37 passed（2H/2H-C/2H-D/2I/2K 共用测试文件）
2J-Fix gate: PASSED 7/7（slot alignment 修复后 attention_only AUC 0.588 显著>surface 0.512，bootstrap stable）；2K output-effect-only 不稳健、组合稳定
新增：src/recover_attention/answer_effect_features.py、scripts/sprint_2J_fix_2K_scoring.py
pytest（Sprint 2I 后）: 582 passed, 2 skipped
2I targeted pytest: 35 passed（2H/2H-C/2H-D/2I 共用测试文件）
2I attention-features gate: 6/7 checks passed, ready_for_2000_rerun=False（唯 #7 未满足 ≥2 打分方法同向）
2I hidden_plus_attention macro_f1=0.437（最佳）> hidden_only 0.426 > attention_only 0.397 > surface 0.378；但排序 vs hidden_only 仅 +0.003（不显著）
2H-D ordinal-calibration gate: 5/8 checks passed, ready_for_2000_rerun=False（primary=ordinal_threshold；仅 expected_bucket 打分 bootstrap 显著 +0.076）
2H-C feature-enrichment gate: 5/8 checks passed, ready_for_2000_rerun=False
2H-C enriched macro_f1=0.426 > surface_rule 0.378 > span_type_only 0.339 > hidden_no_recovered 0.318（bootstrap 对两 baseline 均显著）
2H-B instance signal gate: 6/7 checks passed, ready_for_2000_rerun=False（第 5 项 hidden probe 未击败 baseline）
以下为 Sprint 2 final checkpoint 历史快照：
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
- src/recover_attention/solution_path_numbers.py（Sprint 2H-B：GSM8K 解题路径数字解析）
- src/recover_attention/model_recovery.py（Sprint 2H-B：去泄漏 prompt 的模型生成 recovery）
- src/recover_attention/recovery_drift.py（Sprint 2H-B：recovery drift 分类与聚合）
- src/recover_attention/risk_strength_targets.py（Sprint 2H-B：fragility_bucket / risk_strength 构造）
- src/recover_attention/fragility_probe_training.py（Sprint 2H-B：leakage-safe 探针 + baseline + 序数指标；Sprint 2H-C：新增 hidden_pre_recovery_enriched feature set）
- src/recover_attention/pre_recovery_features.py（Sprint 2H-C：从 2G .pt 缓存构造 pre-recovery enriched 特征）
- src/recover_attention/ordinal_calibration.py（Sprint 2H-D：expected-bucket / ordinal-threshold / isotonic 校准 + budget-aware 指标）
- src/recover_attention/attention_features.py（Sprint 2I：original/masked attention 摘要特征）
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
- scripts/sprint_2H_audit_risk_positive.py（Sprint 2H：risk_positive 只读审计）
- scripts/sprint_2H_instance_signal.py（Sprint 2H-B：instance-level signal staged pipeline）
- scripts/sprint_2H_feature_enrichment.py（Sprint 2H-C：pre-recovery feature enrichment staged pipeline）
- scripts/sprint_2H_ordinal_calibration.py（Sprint 2H-D：ordinal calibration + budget-aware gate）
- scripts/sprint_2I_attention_cache.py（Sprint 2I：4-bit eager attention 缓存）
- scripts/sprint_2I_attention_probe.py（Sprint 2I：attention probe + gate）
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
- tests/test_sprint_2h_instance_signal.py（Sprint 2H-B：24 tests）
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
- Sprint 2H 只读审计确认：2000 规模 gold risk_positive 中 `span_type=number` 为 0 条（规则强制映射为 positive_anchor），recovered filler 是 span_type 的纯函数；human_error_type 全为 sentinel，wrong_numeric_recovery / critical_number 细分本轮无法用数据回答。
- Sprint 2H-B fragility probe 唯一 gate-eligible 特征集（`*_original_masked_cosine_*`）过薄；2G 大部分表征判别力位于被正确禁用的 recovered channel；需更丰富的 pre-recovery 特征（span per-layer hidden states、attention entropy）才能检验 hidden states 是否携带 instance-level fragility。
- Sprint 2H-B 的 fragility_bucket 虽非 span_type 纯函数，但多数类仍与 span_type 强相关（number 多为 bucket 3），使 span_type_only baseline 偏强；hidden_no_recovered probe 在所有指标上均未击败 baseline，gate 第 5 项失败，ready_for_2000_rerun=False。
- Sprint 2H-B recovered-channel 泄漏诊断为弱泄漏（with_recovered − no_recovered = +0.053，低于 0.1 强泄漏阈值）；drift 标签为启发式文本分类的弱标签，非人工校验；per-question top-k 覆盖平凡为 1.0（每题 1 个 chosen span）。
- Sprint 2H-B 复用 2G-2000 的 original/masked hidden-state 缓存，未重跑 hidden-state cache、未缓存 recovered question hidden states、未覆盖 2A-2G 输出。
- Sprint 2H-C 已证明 enriched pre-recovery 特征（delta magnitude + cross-layer stability）在 macro_f1 / balanced_acc 上显著优于 surface baseline，但排序指标（spearman/pairwise）仍略逊，bucket_3_recall 因退化基线偏低，故 gate 5/8、ready_for_2000_rerun=False。
- Sprint 2H-C 的 bucket_3_recall gate 项与「最大化 macro_f1」内在冲突（surface baseline 靠 number→bucket3 退化预测拿高单类 recall）；后续 gate 应改用 macro/ordinal 综合项而非单类 recall 对齐退化基线。
- attention-level 特征（span attention mass / entropy / mask-to-span attention）在当前 2G cache 不可得；如需检验必须先补 targeted attention cache（不在 2H-C boundary 内，未重跑模型）。
- Sprint 2H-C 复用 2G .pt hidden-state cache 与 2H-B risk_strength_dataset，仅 original+masked channel；未使用 recovered channel / gold solution path / fragility label 作为输入特征；未覆盖 2G/2H-B 输出。
- Sprint 2H-D 校准后 enriched 排序仅在 expected_bucket 一种打分下显著优于 surface（+0.076，CI 下界紧贴 0），ordinal_threshold 打平、两种回归打分下 surface 略优；排序优势真实但不稳健。
- Sprint 2H-D budget-aware bucket-3 在 top-10% 预算下 enriched 未稳定优于 surface（0.688 vs 0.708）；bucket_3_vs_1 AUC 在 primary 方法下仍输 surface；故 gate 5/8、ready_for_2000_rerun=False。
- Sprint 2H-D 未把 primary 切到唯一能过的 expected_bucket 去凑 gate（避免 p-hacking），保留 surface-blind 预注册规则；expected_bucket 显著性作为「promising 但不稳健」记录。
- Sprint 2H-D 复用 2H-C 数据集，未重跑 recovery / hidden-state cache、未扩大到 2000、未手动改标签；开工前核验当前 drift 代码重算 labels 与磁盘 0/500 差异。
- Sprint 2I：attention 特征提升的是分类（hidden+attention macro_f1 0.437 最佳，vs surface bootstrap 显著），但未提升 ordinal ranking 稳健性（vs hidden_only +0.003 不显著；cand vs surface 排序仅 expected_bucket 一种打分显著）；budget-aware bucket-3 在有限预算下 hidden_only 反而最优。gate 6/7、ready_for_2000_rerun=False（#7 未满足 ≥2 打分方法同向）。
- Sprint 2I 只缓存 original+masked attention（4-bit eager，来自量化模型，排除 NaN 的最终层 27），未缓存 recovered attention/hidden、未重跑 recovery、未扩大到 2000、未覆盖既有输出；context-to-slot 依赖正则定位（qfocus 缺失 15% / operation 13%）。

## 6. 下一步

下一步建议：

```text
Sprint 2J（建议）：Trajectory-Level Features 或 Hybrid Recovery-Guided Pipeline（仍 500-case，不扩大规模）。
- 2I 已证明：attention 特征提升分类（hidden+attention macro_f1 0.437 为最佳），但未提升 2H-D 标记的真正瓶颈——ordinal ranking 稳健性（vs hidden_only +0.003 不显著；cand vs surface 仅 expected_bucket 一种打分显著）。
- 结论：pre-recovery 的 hidden + attention 静态表征已到瓶颈；要稳健超过 surface 的排序/预算，需要更强的 instance-level 信号：
  - trajectory-level features：对每个 span 做 masked/原始的多步生成，度量 answer stability / trajectory divergence / self-consistency 方差（推理时可得，不依赖 gold）；
  - 或 hybrid recovery-guided：把 recovery 信号仅用于「标签」的部分转成推理时可复现的一致性特征。
- 评估沿用 2H-D/2I 的 ordinal + budget-aware 框架；gate 仍要求 ≥2 打分方法同向 + budget-aware bucket-3 稳定优于 surface。
- expected_bucket 是 2H-D/2I 中唯一稳定使候选优于 surface 的打分，新特征下优先复核。
- 只有排序/预算在 ≥2 方法下稳健且显著优于 surface 后，才考虑 500 -> 2000 rerun。
```

注意：

```text
不要自动开始 Sprint 2J / all-train / Sprint 3A；必须先有 task card 或用户明确指令。
Attention-features rerun gate（2I 已实现于 review_gate_attention_features.json，当前 6/7）：
- hidden_plus_attention Spearman 显著 > surface（primary=expected_bucket 已通过）；
- Spearman > hidden-only enriched（已通过，但 +0.003 微弱、bootstrap 不显著）；
- top-budget bucket-3 precision/F1 > surface（已通过）；
- off-path budget 不恶化、coverage 不降、leakage 通过（均已通过）；
- ≥2 scoring method 同向支持 candidate > surface（当前失败，仅 expected_bucket 一种）。
2G-2000 / 2H-B / 2H-C / 2H-D / 2I 均为 weak-labeled diagnostic evidence，不是 human-reviewed validation；
不得把 weak-labeled metrics 说成 attention guidance 有效 / hallucination 减少 / answer accuracy 提升。
```
## Current Status - Sprint 2J Multi-Span Reasoning Matrix

Sprint 2J is completed as a 500-case diagnostic run.

Validation:
- Full pytest passed: 592 passed, 2 skipped.

Completed:
- 2J-A built a multi-span-per-question matrix from the existing 500-case GSM8K diagnostic subset.
- 2J-A gate passed: 500 questions, 4935 candidate spans, mean 9.87 spans/question, 8/8 checks passed.
- 2J-B ran the real local Qwen2.5-7B-Instruct 4-bit eager original/masked forward pass for all 4935 candidate spans.
- 2J-B generated hidden/attention features, same-question ranking metrics, formula validation, top-k budget reports, failure/success cases, and reasoning-signal gap report.
- Feature/formula leakage audits passed; eval-only fields such as solution_path_status, fragility_bucket, and risk_strength were not used as non-oracle formula inputs.

Key result:
- 2J-B review gate did not pass: 6/8 checks.
- Best non-oracle formula: C_attention_only.
- Best non-oracle AUC delta vs surface-only: -0.0502.
- Non-oracle formulas reduced off-path budget share in some settings, but did not improve on-path/off-path number ranking over the surface baseline.
- On-path number top-k coverage decreased for the best non-oracle formula.
- Oracle diagnostic upper bound remains high, confirming the gap is signal/formula quality rather than the multi-span substrate itself.

Current decision:
- ready_for_2000_rerun = false.
- do_not_enter_sprint_3A = true.
- Do not run 2000 rerun and do not start attention steering.

Primary outputs:
- outputs/logs/sprint_2J_multi_span_matrix_500/
- outputs/logs/sprint_2J_multi_span_scoring_500/

Next recommended sprint:
- Formula validation or reasoning-signal sprint, especially answer-logprob / semantic-role / trajectory evidence before any steering implementation.
