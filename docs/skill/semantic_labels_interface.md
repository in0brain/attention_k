# Semantic Labels Interface

This document defines the stable Sprint 1E interface for:

```text
data/processed/semantic_labels.jsonl
```

`semantic_labels.jsonl` is built from Sprint 1D score-only NLI records:

```text
data/processed/nli_scores.jsonl
```

The input contract is defined in:

```text
docs/skill/nli_scores_interface.md
```

---

# 1. Purpose

`semantic_labels.jsonl` stores rule-based semantic necessity labels for each NLI
score record.

It answers:

```text
Did the ablation preserve meaning, remove information, add assumptions, or make
the question non-equivalent?
```

This file is the first stage that may contain `semantic_necessity_label`.
`nli_scores.jsonl` remains score-only and must not contain this label.

---

# 2. Record Schema

Each line in `data/processed/semantic_labels.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:semantic_label -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
semantic_label_id
nli_id
ablation_id
id
unit_id
unit_scope
group_type
span_ids
spans
ablation_type
original_question
ablated_question
nli_backend
language
language_setting
forward
backward
bidirectional_entailment_score
contradiction_score
semantic_label_backend
semantic_necessity_label
semantic_necessity_score
is_semantically_necessary
rule_parameters
decision_reason
```

---

# 3. Field Sources

Fields copied from Sprint 1D / `nli_scores.jsonl`:

```text
nli_id
ablation_id
id
unit_id
unit_scope
group_type
span_ids
spans
ablation_type
original_question
ablated_question
nli_backend
language
language_setting
forward
backward
bidirectional_entailment_score
contradiction_score
```

Fields created by Sprint 1E:

```text
semantic_label_id
semantic_label_backend
semantic_necessity_label
semantic_necessity_score
is_semantically_necessary
rule_parameters
decision_reason
```

---

# 4. ID Convention

`semantic_label_id` is deterministic:

```python
semantic_label_id = f"{nli_id}__sem_{semantic_label_backend}"
```

For Sprint 1E:

```text
semantic_label_backend = rule_v0
```

---

# 5. Backend

Sprint 1E supports only:

```text
rule_v0
```

Unsupported backends must fail with:

```text
Unsupported backend: <backend>
```

This backend does not call a real NLI model, LLM judge, external API, or model
download. It only reads the existing NLI score fields.

---

# 6. Rule Parameters

Default parameters:

```json
{
  "equivalent_threshold": 0.7,
  "directional_entailment_threshold": 0.5,
  "contradiction_threshold": 0.5
}
```

Each threshold must be a number in `[0, 1]`.

---

# 7. Label Enum

Allowed `semantic_necessity_label` values:

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

`is_semantically_necessary` is derived as:

```python
is_semantically_necessary = semantic_necessity_label != "Equivalent"
```

---

# 8. rule_v0 Decision Order

The rule order is fixed:

```python
if contradiction_score >= contradiction_threshold:
    semantic_necessity_label = "Non-equivalent"
    decision_reason = "contradiction above threshold"

elif bidirectional_entailment_score >= equivalent_threshold:
    semantic_necessity_label = "Equivalent"
    decision_reason = "bidirectional entailment above equivalent threshold"

elif (
    forward_entailment >= directional_entailment_threshold
    and backward_entailment < directional_entailment_threshold
):
    semantic_necessity_label = "Information Loss"
    decision_reason = "forward entails ablated but backward does not entail original"

elif (
    forward_entailment < directional_entailment_threshold
    and backward_entailment >= directional_entailment_threshold
):
    semantic_necessity_label = "Added Assumption"
    decision_reason = "backward entails original but forward does not entail ablated"

else:
    semantic_necessity_label = "Non-equivalent"
    decision_reason = "low bidirectional entailment"
```

`forward` means:

```text
original_question -> ablated_question
```

`backward` means:

```text
ablated_question -> original_question
```

---

# 9. Semantic Necessity Score

The numeric score is:

```python
semantic_necessity_score = round(
    max(1.0 - bidirectional_entailment_score, contradiction_score),
    10,
)
```

The value must be in `[0, 1]`.

---

# 10. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_semantic_label_record(record: dict) -> None
```

The validator checks required fields, copied NLI score fields, forward/backward
direction, label enum, rule parameters, boolean consistency, and
`semantic_necessity_score`.

---

# 11. Module Responsibilities

```text
src/recover_attention/semantic_labels.py
  Builds semantic label records from NLI score records with rule_v0.

scripts/06_build_semantic_labels.py
  Reads nli_scores.jsonl, builds semantic_labels.jsonl, validates output, and
  prints summary stats.

src/recover_attention/schemas.py
  Owns validate_semantic_label_record.
```

---

# 12. Downstream Use

The next stage can consume `semantic_labels.jsonl` to construct
`masked_questions.jsonl`.

Sprint 1F should aggregate semantic label records by:

```text
id + unit_id
```

The `delete` and `generalize` semantic sources for the same unit should be
collected into one masked question record. They must not produce duplicate
masked questions.

The masked question schema is defined in:

```text
docs/skill/masked_questions_interface.md
```

This file does not define or contain:

```text
masked question construction
question recovery
recoverability scoring
attention anchor labels
guidance actions
hidden states
attention maps
trajectory analysis
probe training
```
