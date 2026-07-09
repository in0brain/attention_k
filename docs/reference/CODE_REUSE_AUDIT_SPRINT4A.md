# Code Reuse Audit for Sprint 4A

Sprint 4 resets the mainline to cyber-domain supervised direction probing. This audit records which Sprint 3 code remains useful, which needs adaptation, and which should remain historical or baseline-only.

## A. Directly Reusable

### `src/recover_attention/module_causal_tracing.py`

Reusable pieces:

```text
capture_module_outputs
module_vector
register_module_patch_hook
sequence_logprob_with_module_patch
build_answer_readout concept
paired_bootstrap_delta
```

Reason:

```text
capture_module_outputs can capture attention_output / mlp_output / residual_output.
module_vector can read a selected layer/position module output.
register_module_patch_hook can serve as a foundation for later controlled patches.
build_answer_readout provides the right design pattern, but the answer parser must be replaced for cyber labels.
```

Boundary:

```text
Do not treat this as permission to run new patching in Sprint 4A.
```

### `src/recover_attention/mlp_readout_direction.py`

Reusable pieces:

```text
extract_mlp_readout
compute_correct_wrong_delta
normalize_direction
mean_direction
pca_direction
cosine
register_mlp_direction_nudge
sequence_logprob_with_direction_nudge
```

Reason:

```text
extract_mlp_readout can capture cyber label-readout MLP outputs once domain readout positions exist.
compute_correct_wrong_delta can construct delta_mlp targets.
normalize_direction / mean_direction / pca_direction / cosine remain generic vector utilities.
register_mlp_direction_nudge can support later probe-guided steering.
sequence_logprob_with_direction_nudge can be adapted to label-sequence scoring.
```

Boundary:

```text
Gold-unembedding directions remain oracle/eval-only unless a supervised direction-selection probe predicts a deployable direction.
```

### `src/recover_attention/mlp_readout_attribution.py`

Reusable pieces:

```text
project_to_unembedding
extract_projection_features pattern
evaluate_correct_wrong_detection
auroc
auprc
question_grouped_cv
calibration_buckets
assert_no_eval_only_features
```

Reason:

```text
project_to_unembedding is generic.
extract_projection_features can be converted from number-token subsets to label-space subsets.
evaluate_correct_wrong_detection / AUROC / AUPRC are reusable.
question_grouped_cv is useful for grouped cyber evaluation.
calibration_buckets remains useful for risk estimates.
assert_no_eval_only_features can be extended to cyber gold-label leakage checks.
```

Required adaptation:

```text
Replace number-token filtering with candidate-label-space projection.
Extend leakage checks from gold_answer to gold_label / gold_label_id / wrong_label metadata.
```

### `src/recover_attention/approx_j_lens_readout.py`

Reusable pieces:

```text
register_mlp_output_perturb_hook
answer_slot_logits_with_mlp_perturb
topk_overlap
spearman_correlation_from_maps
project_vector_to_token_scores
```

Reason:

```text
register_mlp_output_perturb_hook can be reused for direction sanity checks.
answer_slot_logits_with_mlp_perturb can become label-slot logits with domain readout positions.
topk_overlap and spearman_correlation_from_maps remain useful for direct-vs-perturbed readout comparisons.
project_vector_to_token_scores is useful for label-token score maps.
```

Boundary:

```text
Approximate J-lens remains a sanity check, not a proof of deployable steering.
```

## B. Adaptable Reuse

### `src/recover_attention/answer_proxy_metrics.py`

Reusable ideas:

```text
sequence_logprob_at_answer_slot
token_index_for_char_start
bootstrap_ci
paired_bootstrap_delta
```

Must be replaced for cyber-domain labels:

```text
numeric answer parser
HASH_ANSWER_RE
PHRASE_ANSWER_RE
NUMBER_RE
numeric normalization
answer_token_ids assumptions about numeric answers
```

Recommended new module:

```text
src/recover_attention/domain_answer_proxy.py
```

Target responsibilities:

```text
parse structured labels or multiple-choice options;
locate label-readout positions;
score full label sequences;
support multi-token cyber labels;
avoid GSM8K numeric-answer assumptions.
```

## C. Not Recommended as the New Mainline

### `attention_bias_steering.py`

Use as:

```text
negative baseline
historical reference
controlled comparison
```

Reason:

```text
Sprint 3A showed attention-bias steering did not produce selective answer-directed improvement even under oracle span selection.
```

### `representation_intervention.py`

Use as:

```text
negative baseline
historical reference
mechanism comparison
```

Reason:

```text
Sprint 3B showed span residual injection at the answer position was non-selective. It changed model state but did not supply a reliable correction direction.
```

### Old Sprint 3A / 3B Scripts

Use as:

```text
audit trail
negative controls
historical evidence for why blind span steering was stopped
```

Reason:

```text
They test the old mainline. Sprint 4 should not keep iterating blind span-level steering unless used as a clearly labeled baseline.
```

### GSM8K Numeric-Answer Scripts

Use as:

```text
diagnostic baseline
parser reference for what not to assume
historical Sprint 2/3 evidence
```

Reason:

```text
GSM8K numeric answers are not the target interface for cyber-domain structured label prediction. Numeric parsing and arbitrary numeric answer directions should not drive Sprint 4.
```

## Sprint 4A Conclusion

The codebase already contains useful causal tracing, MLP readout, projection, grouped-evaluation, and finite-difference sanity-check machinery. The new mainline should reuse those mechanisms around a different data and label interface:

```text
cyber structured label space
domain answer proxy
direction-selection probe
low-alpha, risk-controlled MLP readout evaluation
```

The old blind attention/span intervention code remains valuable evidence and baseline infrastructure, but it should not be the Sprint 4 mainline.

