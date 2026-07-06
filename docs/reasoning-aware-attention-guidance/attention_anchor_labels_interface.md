# Attention Anchor Labels Interface

This document defines the stable interface for:

```text
data/processed/attention_anchor_labels.jsonl
```

`attention_anchor_labels.jsonl` stores unit-level attention anchor label records
derived from `unit_evidence.jsonl`.

---

# 1. Purpose

Each record assigns one early attention anchor label to one ablation unit
(`id + unit_id`), based on the evidence summarized in `unit_evidence.jsonl`.

It is NOT:

```text
1. attention guidance result
2. attention steering manifest
3. trajectory stability result
4. answer stability result
5. raw attention analysis
6. probe training data
```

Because the upstream evidence may still be partial, `label_status` makes that
limitation explicit.

---

# 2. Pipeline Position

```text
unit_evidence.jsonl
->
attention_anchor_labels.jsonl
->
future guidance / intervention stages
```

Where:

```text
unit_evidence.jsonl:
  Provides unit-level early evidence (semantic necessity + recoverability).

attention_anchor_labels.jsonl:
  Assigns a unit-level attention anchor label from that evidence.
```

The interface is introduced before the builder. The builder
(`src/recover_attention/attention_anchor_labels.py`,
`scripts/11_build_attention_anchor_labels.py`) belongs to a later sprint.

---

# 3. Record Schema

Each line in `data/processed/attention_anchor_labels.jsonl` must be one JSON
object.

Each record must contain:

<!-- required_fields:attention_anchor_label -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

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
semantic_evidence
recoverability_evidence
available_signal_types
missing_signal_types
attention_importance_score
attention_anchor_label
label_backend
label_status
evidence
```

---

# 4. Field Sources

Copied from `unit_evidence.jsonl`:

```text
unit_evidence_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
semantic_evidence
recoverability_evidence
available_signal_types
missing_signal_types
```

Created by the attention anchor label stage:

```text
attention_anchor_label_id
attention_importance_score
attention_anchor_label
label_backend
label_status
evidence
```

---

# 5. ID Rule

```python
attention_anchor_label_id = f"{unit_evidence_id}__anchor_{label_backend}"
```

Current allowed label backend:

```text
early_evidence_rule_stub_v0
```

Current allowed label status:

```text
partial_evidence_label
```

---

# 6. Unit-level Boundary

```text
1. This interface is unit-level.
2. It does not use top-level span_id / span_text / span_type.
3. Multiple spans in a group unit remain inside span_ids / spans.
4. Span-level or token-level expansion belongs to a later stage.
```

---

# 7. Label Boundary

```text
1. attention_anchor_label is a label decision, not an intervention.
2. guidance_action and guidance_strength are not included.
3. Attention guidance belongs to later stages.
4. Because current evidence may still be partial, label_status must make this
   limitation explicit.
```

The label enum reuses `ALLOWED_ATTENTION_ANCHOR_LABELS`:

```text
Strong Anchor
Medium Anchor
Weak Anchor
Risky Anchor
Distractor
```

---

# 8. Not Included

`attention_anchor_labels.jsonl` must not contain:

```text
guidance_action
guidance_strength
hidden states
attention maps
trajectory analysis
answer stability
raw attention analysis
probe label
```

---

# 9. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_attention_anchor_label_record(record: dict) -> None
```

The validator checks required fields, rejects stale span-level fields,
sample-level recovery fields, guidance fields, and hidden state / attention map
/ trajectory / answer stability / raw attention / probe fields; validates unit
span metadata and single/group constraints; verifies `attention_anchor_label_id`;
enforces the anchor label, label backend, and label status enums; checks
`attention_importance_score` range; and validates available / missing signal
types against the approved signal enum.

---

# 10. Example

```json
{
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
  "semantic_evidence": {
    "source_semantic_label_ids": [
      "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0"
    ],
    "summary_label": "Information Loss",
    "summary_score": 0.75
  },
  "recoverability_evidence": {
    "recover_score_id": "gsm8k_0001__unit_001__mask__score_stub_rule_v0",
    "recoverability_label": "Recoverable",
    "recoverability_score": 1.0
  },
  "available_signal_types": [
    "semantic_necessity",
    "semantic_recoverability"
  ],
  "missing_signal_types": [
    "trajectory_stability",
    "answer_stability",
    "raw_attention_pattern",
    "attention_steering_effect"
  ],
  "attention_importance_score": 0.5,
  "attention_anchor_label": "Weak Anchor",
  "label_backend": "early_evidence_rule_stub_v0",
  "label_status": "partial_evidence_label",
  "evidence": {
    "notes": "Example only; builder implementation belongs to a later sprint.",
    "source_files": [
      "data/processed/unit_evidence.jsonl"
    ],
    "limitations": [
      "evidence is partial: only semantic necessity and recoverability are available",
      "trajectory stability, answer stability, raw attention, and guidance are not included"
    ]
  }
}
```
