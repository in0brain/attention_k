# Unit Evidence Interface

This document defines the stable interface for:

```text
data/processed/unit_evidence.jsonl
```

`unit_evidence.jsonl` stores unit-level evidence summaries before final
attention anchor labeling.

It summarizes what evidence is currently available for an ablation unit. It is
not the final attention anchor label, a guidance action, a trajectory stability
result, an answer stability result, or raw attention analysis.

---

# 1. Purpose

Each record represents one unit-level evidence aggregation record for one
ablation unit:

```text
id + unit_id
```

The record is intended to carry early evidence forward without over-interpreting
recoverability as attention importance.

It is not:

```text
final attention anchor label
guidance action
trajectory stability result
answer stability result
raw attention analysis
```

---

# 2. Pipeline Position

Current intended pipeline position:

```text
semantic_labels.jsonl
+
recover_scores.jsonl
->
unit_evidence.jsonl
->
attention_anchor_labels.jsonl
```

Where:

```text
semantic_labels.jsonl:
  Provides semantic necessity evidence.

recover_scores.jsonl:
  Provides recoverability evidence.

unit_evidence.jsonl:
  Aggregates early evidence at unit level before anchor labeling.
```

This sprint only defines the interface. It does not implement aggregation.

---

# 3. Record Schema

Each line in `data/processed/unit_evidence.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:unit_evidence -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

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
evidence_backend
evidence_status
evidence
```

---

# 4. Field Sources

Copied from `semantic_labels.jsonl` / `recover_scores.jsonl`:

```text
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
```

Aggregated from `semantic_labels.jsonl`:

```text
semantic_evidence
```

Aggregated from `recover_scores.jsonl`:

```text
recoverability_evidence
```

Created by the unit evidence stage:

```text
unit_evidence_id
available_signal_types
missing_signal_types
evidence_backend
evidence_status
evidence
```

This sprint only designs the interface. It does not implement the aggregation
stage.

---

# 5. ID Rule

`unit_evidence_id` binds the evidence record to one unit and one evidence
backend:

```python
unit_evidence_id = f"{id}__{unit_id}__evidence_{evidence_backend}"
```

Current allowed evidence backend:

```text
aggregate_stub_v0
```

Current allowed evidence status:

```text
partial_stub_evidence
```

---

# 6. Signal Boundary

Recoverability is evidence, not final attention importance.

```text
Non-recoverable does not automatically mean Strong Anchor.
Recoverable does not automatically mean unimportant.
Misleading Recovery does not automatically mean suppress.
```

Current evidence can include:

```text
semantic_necessity
semantic_recoverability
```

Known missing future signals can include:

```text
trajectory_stability
answer_stability
raw_attention_pattern
attention_steering_effect
```

---

# 7. Not Included

`unit_evidence.jsonl` must not contain:

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
hidden states
attention maps
trajectory analysis
answer stability
probe label
```

Attention labels and guidance actions belong to later stages.

---

# 8. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_unit_evidence_record(record: dict) -> None
```

The validator checks required fields, rejects stale span-level fields,
sample-level recovery fields, attention-label fields, guidance fields, hidden
state / attention map fields, validates unit span metadata, verifies
`unit_evidence_id`, enforces evidence backend and status enums, and checks that
available / missing signal types use the approved signal enum.

---

# 9. Example

```json
{
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
    "masked_id": "gsm8k_0001__unit_001__mask",
    "recovery_backend": "oracle_stub_v0",
    "score_backend": "stub_rule_v0",
    "recoverability_label": "Recoverable",
    "recoverability_score": 1.0,
    "confidence_mean": 1.0,
    "recovery_consistency": 1.0,
    "misleading_recovery": false
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
  "evidence_backend": "aggregate_stub_v0",
  "evidence_status": "partial_stub_evidence",
  "evidence": {
    "notes": "Example only; builder implementation belongs to a later sprint.",
    "source_files": [
      "data/processed/semantic_labels.jsonl",
      "data/processed/recover_scores.jsonl"
    ],
    "limitations": [
      "recoverability comes from oracle_stub_v0 + stub_rule_v0",
      "trajectory, answer stability, raw attention, and guidance are not included"
    ]
  }
}
```
