# Sprint 3 Artifact Manifest

Purpose: `outputs/` is fully ignored by Git (`.gitignore:14 outputs/`), so the
detailed Sprint 3 aggregate reports are not directly auditable from the GitHub
tree. This manifest records, per file, whether it exists locally, whether it is
tracked by Git, and — if not tracked — the reason, plus the key conclusion each
sprint's reports support. It satisfies Sprint 3C-0 preflight Fix 2.

Legend: `exists` = present on the local workstation; `tracked` = committed to
Git; `ignored` = matched by `.gitignore`.

## Sprint 3A-1 — Controlled Attention Guidance (500-subset diagnostic)

Directory: `outputs/logs/sprint_3A_1_controlled_attention_guidance_500/`

| file | exists | tracked | reason if untracked |
| --- | --- | --- | --- |
| oracle_sanity_diagnostic_report.json | yes | no | under gitignored `outputs/` |
| answer_position_output_shift_report_500.json | yes | no | under gitignored `outputs/` |
| harm_rate_report_500.json | yes | no | under gitignored `outputs/` |
| baseline_comparison_report_500.json | yes | no | under gitignored `outputs/` |
| review_gate_controlled_attention_guidance_500.md | yes | no | under gitignored `outputs/` |

Key conclusion (recoverable from `PROGRESS.md` and `docs/progress/sprint_3_history.md`):
Even oracle (correct-span) attention-logit bias does **not** reliably move the
answer toward the gold token at any safe λ. At the measurable-output regime
(λ=4.0) the oracle-minus-random gold-first-token logprob delta is
`-0.05328`, CI95 `[-0.22315, 0.13532]` (not stable positive). Attention-bias
steering was therefore abandoned. `ready_for_2000_rerun=false`.

## Sprint 3B-0 — Representation-Level Oracle Intervention (diagnostic)

Directory: `outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/`

| file | exists | tracked | reason if untracked |
| --- | --- | --- | --- |
| representation_intervention_config.json | yes | no | under gitignored `outputs/` |
| representation_intervention_fidelity_report.json | yes | no | under gitignored `outputs/` |
| gold_logprob_delta_report.json | yes | no | under gitignored `outputs/` |
| oracle_vs_random_diagnostic_report.json | yes | no | under gitignored `outputs/` |
| selector_comparison_report.json | yes | no | under gitignored `outputs/` |
| harm_rate_report.json | yes | no | under gitignored `outputs/` |
| review_gate_representation_level_oracle_diagnostic.md | yes | no | under gitignored `outputs/` |

Key conclusion: Even oracle residual injection at the final answer position is
**not** selectively better than random at any safe β. At the regime β=0.05 the
oracle-minus-random gold-logprob delta is `0.00095`, CI95 `[-0.02640, 0.02593]`
(not stable). Hook fidelity was clean (registered/triggered/removed,
injection_norm>0). `ready_for_2000_rerun=false`.

### 3B-0 conclusion boundary (preflight Fix 3)

3B-0 negated only: same-run span residual deviation → final answer-position
injection → single-forward gold-token proxy.

3B-0 did **not** negate: correct-run activation patching; reasoning-step-level
patching; value/MLP causal tracing; autoregressive generation steering.

## Sprint 3C-0 — Correct-vs-Wrong Activation Patching (this sprint)

Directory: `outputs/logs/sprint_3C_0_correct_wrong_activation_patching/`

Large trace/forward manifests (`trace_sampling_manifest.jsonl`,
`correct_wrong_pair_manifest.jsonl`, `activation_patching_forward_manifest.jsonl`)
exist locally but are not committed (under gitignored `outputs/`). The small
aggregate reports and review gate below are the auditable summary; their numeric
contents are also mirrored into `PROGRESS.md` and
`docs/progress/sprint_3_history.md`.

Aggregate reports: `preflight_report.md`,
`review_gate_correct_wrong_activation_patching.md`,
`patching_effect_report.json`, `oracle_patch_vs_control_report.json`,
`layer_position_heatmap_report.json`, `harm_rate_report.json`,
`patching_fidelity_report.json`.

Metric-fix note (execution finding): the first run produced a degenerate
`clean_direction_score ≡ 0` for all 2085 rows. Root cause: Qwen2.5 tokenizes
`" 42"` as `[space, '4', '2']`, so `first_token_id(" "+answer)` returned the
constant leading-space token (id 220) for every numeric answer, making
`gold_first_token == wrong_first_token` in all 35 pairs. Fixed
`first_token_id` to strip the leading space and return the leading *digit* token
(31/35 pairs then discriminative), added a regression test, and reran.

Key conclusion (corrected run): full hook fidelity (registered/triggered/removed
= 1.0, no contamination, mean patch_delta_norm ≈ 104). Overall
`mean_clean_direction_score = -0.0185`, CI95 `[-0.113, +0.080]` — **not stably
positive**. gold and wrong first-token logprobs rise together
(+0.735 vs +0.754), the non-selective-perturbation signature. No reasoning-step
position × layer beats controls. Verdict = **Case C** under this metric — but see
Sprint 3C-0-Fix below: this first-token/last-position proxy was itself too weak,
and the corrected sequence proxy revises the reading to Case B.
`ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`.

## Sprint 3C-0-Fix — Answer-Position Proxy Repair and Sequence-Logprob Recheck

Directory: `outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/` (under gitignored
`outputs/`; large `corrected_patching_forward_manifest.jsonl` not committed).

Aggregate reports (auditable summary; numbers mirrored in `PROGRESS.md` and
`docs/progress/sprint_3_history.md`): `preflight_proxy_fix_report.md`,
`answer_span_extraction_report.json`, `corrected_pair_manifest.jsonl`,
`corrected_sequence_logprob_report.json`, `corrected_clean_direction_report.json`,
`corrected_control_comparison_report.json`,
`corrected_layer_position_heatmap_report.json`, `corrected_harm_rate_report.json`,
`success_case_report.jsonl`, `failure_case_report.jsonl`,
`review_gate_answer_proxy_recheck.md`. All exist locally, untracked (gitignored).

What changed: reads logits at the answer slot (before the final answer number)
and scores the full numeric answer-sequence logprob (gold vs wrong) under the same
wrong prefix; reuses the 3C-0 pair manifest (no re-sampling); robust answer-span
extraction re-validated 34/35 pairs.

Key conclusion (Case B — revises 3C-0 Case C): correct-run activation does carry
answer-fixing information. Reasoning-step patch: clean_direction +0.120 CI95
`[+0.053, +0.197]` (stable vs no_patch) but **non-selective** (not > random donor,
not > same-trace random position). Final-answer read-out position: gold up / wrong
down, vs random donor **+3.13 CI95 `[+1.87, +4.46]` (stable positive)**, vs
no_patch +6.12 stable, but vs same-trace random position not stable, and harm rises
0.41→0.88 across layers 16→24. The selective signal is at the answer
read-out/compression stage, not explicit reasoning steps.
`ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`,
`hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.

## Sprint 3C-1 — Final-Answer Compression Value/MLP Causal Tracing

Directory: `outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/`
(under gitignored `outputs/`; large `module_patching_forward_manifest.jsonl`,
`module_activation_capture_manifest.jsonl` not committed).

Aggregate reports (auditable; numbers mirrored in `PROGRESS.md` and
`docs/progress/sprint_3_history.md`): `preflight_report.md`,
`module_patching_config.json`, `module_patching_fidelity_report.json`,
`module_patching_effect_report.json`, `module_control_comparison_report.json`,
`donor_specificity_report.json`, `site_specificity_report.json`,
`layer_module_heatmap_report.json`, `harm_control_report.json`,
`success_case_report.jsonl`, `failure_case_report.jsonl`,
`review_gate_final_answer_compression_tracing.md`. All exist locally, untracked.

What it did: decomposed the 3C-0-Fix whole-residual patch at the answer-readout
position into per-module writes (self-attention output / MLP output / residual
output), interpolation-patched each (α∈{0.25,0.5,0.75,1.0}, layers {16,20,24})
with correct-run donor activation, and tested donor- and site-specificity with
harm control. Reused the 34 3C-0-Fix pairs (no re-sampling). Consistency check:
`residual_output|L24|α1.0` reproduces the 3C-0-Fix whole-residual result
(+12.67, harm 0.88).

Key conclusion (Case A — a selective, low-harm causal site exists): the correct
answer direction at the readout is written primarily by the **MLP**. Donor-
specificity (correct − random donor) stable positive for all modules (attention
+0.093, mlp +0.141, residual +1.703). Site-specificity (correct at readout −
correct at random position) stable positive **only for mlp_output**
(+0.353 CI95 `[+0.242, +0.476]`); attention_output (-0.006) and residual_output
(-0.547) are not site-specific. Harm-controlled MLP regime: `mlp_output|L24|α0.25`
clean +0.319 (gold +0.327, wrong +0.040) at harm 0.06; `mlp_output|L20|α0.25`
clean +0.131 at harm 0.06. Whole residual is largest but non-selective and
high-harm; attention output moves the answer generically but not site-specifically.
`ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`,
`hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.
Next: Sprint 3C-2 — MLP readout-direction analysis / harm-controlled steering probe.

## Sprint 3C-2 — MLP Readout Direction Analysis and Donor-Free Nudge

Directory: `outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/` (under
gitignored `outputs/`; large `donor_free_nudge_forward_manifest.jsonl` not committed).

Aggregate reports (auditable; mirrored in `PROGRESS.md` / `sprint_3_history.md`):
`preflight_report.md`, `mlp_direction_config.json`,
`mlp_direction_geometry_report.json`, `mlp_unembedding_alignment_report.json`,
`donor_free_nudge_report.json`, `generation_survival_report.json`,
`harm_control_report.json`, `mlp_readout_delta_manifest.jsonl`,
`donor_free_direction_manifest.jsonl`, `success_case_report.jsonl`,
`failure_case_report.jsonl`, `review_gate_mlp_readout_direction_analysis.md`.
All exist locally, untracked.

What it did: extracted the correct−wrong MLP-output delta at the answer-readout
position (3C-1 site), analysed geometry + gold-vs-wrong unembedding alignment, and
tested leave-one-out donor-free directional nudges (mean/PC1/gold-unembed vs
random/shuffled/negative/zero) with the corrected sequence proxy + a first-step
check. Reused the 34 3C-0-Fix pairs.

Key conclusion (mixed): the MLP readout write is a stable low-rank direction (PC1
LOO cosine 0.98–0.99) that at L24 aligns with the gold-vs-wrong unembedding
(per-pair cosine +0.091 CI95 `[+0.064,+0.117]`, stable). The gold-unembedding
direction (eval-only) is a strong low-harm surviving steering axis (beats random
+0.93). But the donor-free mean_delta direction only marginally beats random
(+0.062 CI95 `[+0.0005,+0.130]`), does not beat its own negation, and barely
survives the first-step check; PC1 does not beat random. Note: the `shuffled`
control is degenerate for a global mean direction (`mean_delta − shuffled ≡ 0`
because shuffle-then-average restores the same mean) — random/negative are the
meaningful controls. So the effective axis essentially requires the gold answer;
no deployable donor-free steering handle was found.
`ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`,
`hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`.
Next: improve gold-free direction extraction (whitening / probe) or pivot the
gold-unembedding alignment into an attribution/detection use.

## Sprint 3C-3 - MLP Readout Attribution Probe

Directory: `outputs/logs/sprint_3C_3_mlp_readout_attribution_probe/` (under gitignored `outputs/`; large `mlp_readout_attribution_manifest.jsonl` is not committed).

Aggregate reports (auditable; numbers mirrored in `PROGRESS.md` and `docs/progress/sprint_3_history.md`): `preflight_report.md`, `mlp_attribution_config.json`, `mlp_unembedding_projection_report.json`, `answer_token_attribution_report.json`, `correct_wrong_detection_report.json`, `risk_score_report.json`, `baseline_comparison_report.json`, `calibration_report.json`, `success_case_report.jsonl`, `failure_case_report.jsonl`, `review_gate_mlp_readout_attribution_probe.md`. All exist locally, untracked because `outputs/` is gitignored.

What it did: reused the 34 3C-0-Fix corrected pairs and built 68 trace-level examples. It captured final-answer readout module outputs at layers {20,24}, projected MLP output to number-like lm_head / unembedding rows, built gold-free attribution features, and evaluated a rule risk score plus a small question-grouped numpy logistic probe. Gold answers are eval-only labels and are not used as features.

Key conclusion: MLP readout attribution provides a modest gold-free diagnostic signal but is not a direct answer decoder. MLP top-number vs parsed model answer top-1 match is 0.162 (top-k 0.426). Rule risk AUROC/AUPRC are 0.653/0.638; logistic probe AUROC/AUPRC are 0.623/0.661. High-risk bucket wrong rate is 0.786 and low-risk bucket correct rate is 0.692. It beats random AUROC (0.481) but does not beat final-logits margin (delta -0.059). The result supports mechanism-level attribution / detection, not steering.

Boundary: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`, `steering_continued=false`.

## Sprint 3C-3R Story artifact move

Tracked documentation artifact:
- `docs/reference/STORY.md` - long-form project story and baseline-boundary narrative.

No new `outputs/` artifacts were produced. Existing 3C-3 output manifests remain under `outputs/logs/sprint_3C_3_mlp_readout_attribution_probe/` and are gitignored / not committed.
