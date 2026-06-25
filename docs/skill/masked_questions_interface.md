# Masked Questions Interface

This document defines the stable interface for:

```text
data/processed/masked_questions.jsonl
```

`masked_questions.jsonl` stores unit-level masked questions for the later
recoverability stage. It answers:

```text
If one semantic unit is explicitly replaced by [MASK], can the model recover
the missing information?
```

This interface is unit-level and semantic-label-driven. It replaces the old
span-level masked question schema.

---

# 1. Location

```text
data/processed/masked_questions.jsonl
```

---

# 2. Purpose

`masked_questions.jsonl` saves masked inputs constructed from ablation units.
The mask object is the whole unit, not an individual candidate span.

This file contains only masked question inputs. It does not contain recovery
outputs, recoverability labels, attention guidance fields, probe fields, hidden
states, or attention maps.

---

# 3. Upstream and Downstream

Current pipeline position:

```text
semantic_labels.jsonl
-> masked_questions.jsonl
-> recover_outputs.jsonl
-> recover_scores.jsonl
```

Where:

```text
semantic_labels.jsonl:
  Sprint 1E output. Contains semantic necessity information for each ablation
  unit and ablation type.

masked_questions.jsonl:
  Sprint 1F output. Builds one explicit mask input for each unique unit.

recover_outputs.jsonl:
  Later recovery-stage output. Records model recoveries for masked questions.
```

---

# 4. Aggregation Rule

Sprint 1F must aggregate semantic label records by unit.

Aggregation key:

```text
id + unit_id
```

The same unit may have multiple semantic label sources, such as:

```text
delete
generalize
```

These sources must be aggregated into the same masked question record under:

```text
semantic_sources
```

Do not create duplicate masked questions just because `delete` and
`generalize` each produced one semantic label record.

---

# 5. Record Schema

Each line in `data/processed/masked_questions.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:masked_question -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
masked_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
masked_question
mask_token
mask_backend
mask_strategy
source_semantic_label_ids
source_nli_ids
source_ablation_ids
semantic_sources
```

Field meanings:

```text
masked_id:
  Unique ID for the masked question record.

id:
  Original question ID.

unit_id:
  Ablation unit ID being masked.

unit_scope:
  single or group.

group_type:
  single / repeated_surface / number_set / entity_set / object_set /
  cyber_security_term_set.

span_ids:
  List of span_id values included in the unit.

spans:
  Complete span objects included in the unit.

original_question:
  Original question text.

masked_question:
  Question text after replacing all spans in the unit with mask_token.

mask_token:
  Mask marker. Default semantic value is [MASK].

mask_backend:
  Backend used to construct the masked question. Currently only unit_mask_v0
  is allowed.

mask_strategy:
  Masking strategy. Currently only replace_each_span is allowed. It requires
  every span in the unit to add exactly one mask_token.

source_semantic_label_ids:
  Aggregated semantic_label_id values for this unit.

source_nli_ids:
  Aggregated nli_id values for this unit.

source_ablation_ids:
  Aggregated ablation_id values for this unit.

semantic_sources:
  Summary list of the semantic label records aggregated into this masked
  question record.
```

---

# 6. ID Rule

Recommended `masked_id` format:

```python
masked_id = f"{id}__{unit_id}__mask"
```

Example:

```text
gsm8k_0001__unit_001__mask
```

`masked_id` does not include `delete` or `generalize`, because the explicit
mask is constructed for the unit, not for a delete/generalize ablation record.

---

# 7. semantic_sources Schema

`semantic_sources` must be a non-empty list.

Each element must contain at least:

```text
semantic_label_id
nli_id
ablation_id
ablation_type
semantic_necessity_label
semantic_necessity_score
is_semantically_necessary
decision_reason
```

Recommended ordering:

```text
delete
generalize
```

If a unit has only one semantic source, that is allowed, but the record must
still remain schema-valid and preserve the source metadata.

Example:

```json
{
  "semantic_label_id": "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
  "nli_id": "gsm8k_0001__unit_001__delete__nli_stub_v0",
  "ablation_id": "gsm8k_0001__unit_001__delete",
  "ablation_type": "delete",
  "semantic_necessity_label": "Information Loss",
  "semantic_necessity_score": 0.75,
  "is_semantically_necessary": true,
  "decision_reason": "forward entails ablated but backward does not entail original"
}
```

---

# 8. Full Example

```json
{
  "masked_id": "gsm8k_0001__unit_001__mask",
  "id": "gsm8k_0001",
  "unit_id": "unit_001",
  "unit_scope": "single",
  "group_type": "single",
  "span_ids": ["span_001"],
  "spans": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    }
  ],
  "original_question": "Tom has 3 apples and buys 2 more.",
  "masked_question": "Tom has [MASK] apples and buys 2 more.",
  "mask_token": "[MASK]",
  "mask_backend": "unit_mask_v0",
  "mask_strategy": "replace_each_span",
  "source_semantic_label_ids": [
    "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
    "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0"
  ],
  "source_nli_ids": [
    "gsm8k_0001__unit_001__delete__nli_stub_v0",
    "gsm8k_0001__unit_001__generalize__nli_stub_v0"
  ],
  "source_ablation_ids": [
    "gsm8k_0001__unit_001__delete",
    "gsm8k_0001__unit_001__generalize"
  ],
  "semantic_sources": [
    {
      "semantic_label_id": "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
      "nli_id": "gsm8k_0001__unit_001__delete__nli_stub_v0",
      "ablation_id": "gsm8k_0001__unit_001__delete",
      "ablation_type": "delete",
      "semantic_necessity_label": "Information Loss",
      "semantic_necessity_score": 0.75,
      "is_semantically_necessary": true,
      "decision_reason": "forward entails ablated but backward does not entail original"
    },
    {
      "semantic_label_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0",
      "nli_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0",
      "ablation_id": "gsm8k_0001__unit_001__generalize",
      "ablation_type": "generalize",
      "semantic_necessity_label": "Information Loss",
      "semantic_necessity_score": 0.65,
      "is_semantically_necessary": true,
      "decision_reason": "forward entails ablated but backward does not entail original"
    }
  ]
}
```

---

# 9. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_masked_question_record(record: dict) -> None
```

The validator owns schema-level checks for this interface. It rejects the old
top-level span fields:

```text
span_id
span_text
span_type
```

It also checks the required unit fields, span ordering, source ID ordering,
semantic source fields, semantic necessity label enum, semantic necessity score
range, boolean types, and `single` / `group` span count constraints.

Backend and strategy are enforced as closed enums:

```text
mask_backend  must be one of {unit_mask_v0}
mask_strategy must be one of {replace_each_span}
```

When `mask_strategy == "replace_each_span"`, the validator enforces the mask
count instead of mere presence:

```python
added_mask_count = masked_question.count(mask_token) - original_question.count(mask_token)
assert added_mask_count == len(spans)
```

This subtracts any mask_token already present in `original_question`, so a group
unit that only adds one `[MASK]` for two spans is rejected.

---

# 10. Not Included

`masked_questions.jsonl` must not contain:

```text
recovery output
recoverability label
attention anchor label
guidance action
hidden states
attention maps
trajectory analysis
probe label
```
