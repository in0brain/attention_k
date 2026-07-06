# Intervention Manifest Interface

This document defines the stable interface for:

```text
data/processed/intervention_manifest.jsonl
```

`intervention_manifest.jsonl` stores unit-level planned intervention records
derived from `attention_anchor_labels.jsonl`.

---

# 1. Purpose

Each record describes one PLANNED intervention for one ablation unit
(`id + unit_id`), selected from `attention_anchor_labels.jsonl`. It records what
a later stage intends to do; it does not execute anything.

It is NOT:

```text
1. attention guidance result
2. model execution log
3. trajectory stability result
4. answer stability result
5. raw attention analysis
6. hidden-state cache manifest
7. probe training data
```

---

# 2. Pipeline Position

```text
attention_anchor_labels.jsonl
->
intervention_manifest.jsonl
->
future intervention execution / guidance evaluation stages
```

Where:

```text
attention_anchor_labels.jsonl:
  Provides unit-level early attention anchor labels.

intervention_manifest.jsonl:
  Records a planned-only intervention per selected unit.
```

The interface is introduced before the builder. The builder
(`src/recover_attention/intervention_manifest.py`,
`scripts/12_build_intervention_manifest.py`) belongs to a later sprint.

---

# 3. Record Schema

Each line in `data/processed/intervention_manifest.jsonl` must be one JSON
object.

Each record must contain:

<!-- required_fields:intervention_manifest -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
intervention_id
attention_anchor_label_id
unit_evidence_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
attention_importance_score
attention_anchor_label
label_backend
label_status
intervention_type
target_scope
intervention_backend
intervention_status
planned_operation
evidence
```

---

# 4. Field Sources

Copied from `attention_anchor_labels.jsonl`:

```text
attention_anchor_label_id
unit_evidence_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
attention_importance_score
attention_anchor_label
label_backend
label_status
```

Created by the intervention manifest stage:

```text
intervention_id
intervention_type
target_scope
intervention_backend
intervention_status
planned_operation
evidence
```

---

# 5. ID Rule

```python
intervention_id = f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"
```

Current allowed values:

```text
intervention_type:   mask / remove / replace
target_scope:        unit
intervention_backend: manifest_stub_v0
intervention_status: planned_only
```

---

# 6. Planned-only Boundary

```text
1. intervention_manifest is planned-only.
2. It does not execute model inference.
3. It does not apply attention steering.
4. It does not contain baseline / guided answers.
5. It does not contain trajectory or answer stability scores.
6. It does not contain hidden states or attention maps.
7. It does not contain hidden_states_path or attentions_path.
```

---

# 7. Unit-level Boundary

```text
1. This interface is unit-level.
2. It does not use top-level span_id / span_text / span_type.
3. Multiple spans in a group unit remain inside span_ids / spans.
4. Span-level or token-level expansion belongs to a later stage.
```

---

# 8. Not Included

`intervention_manifest.jsonl` must not contain:

```text
guidance_action
guidance_strength
baseline_answer
guided_answer
intervened_answer
trajectory_stability_score
answer_stability_score
raw_attention_score
hidden states
attention maps
hidden_states_path
attentions_path
probe label
probe confidence
```

---

# 9. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_intervention_manifest_record(record: dict) -> None
```

The validator checks required fields, rejects stale span-level fields, guidance
fields, answer fields, stability scores, hidden state / attention map fields and
paths, and probe fields; validates unit span metadata and single/group
constraints; verifies `intervention_id`; enforces the anchor label, label
backend, label status, intervention type, target scope, intervention backend,
and intervention status enums; checks `attention_importance_score` range; and
requires `planned_operation` to be a dict.

---

# 10. Example

```json
{
  "intervention_id": "gsm8k_0001__unit_001__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0__intervention_mask_manifest_stub_v0",
  "attention_anchor_label_id": "gsm8k_0001__unit_001__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0",
  "unit_evidence_id": "gsm8k_0001__unit_001__evidence_aggregate_stub_v0",
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
  "attention_importance_score": 0.55,
  "attention_anchor_label": "Medium Anchor",
  "label_backend": "early_evidence_rule_stub_v0",
  "label_status": "partial_evidence_label",
  "intervention_type": "mask",
  "target_scope": "unit",
  "intervention_backend": "manifest_stub_v0",
  "intervention_status": "planned_only",
  "planned_operation": {
    "description": "Mask all spans in the unit for a future recoverability/guidance experiment.",
    "target_span_ids": ["span_001"]
  },
  "evidence": {
    "notes": "Example only; builder implementation belongs to a later sprint.",
    "source_files": ["data/processed/attention_anchor_labels.jsonl"],
    "limitations": [
      "planned-only manifest; no model execution",
      "no guidance_action / guidance_strength",
      "no trajectory / answer stability, raw attention, or hidden-state cache"
    ]
  }
}
```
