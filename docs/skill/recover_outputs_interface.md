# Recover Outputs Interface

This document defines the stable interface for:

```text
data/processed/recover_outputs.jsonl
```

`recover_outputs.jsonl` stores recovery attempts for unit-level masked
questions. It consumes:

```text
data/processed/masked_questions.jsonl
```

This interface is `masked_id`-driven, unit-level, and self-contained enough for
the next scoring stage to read it without joining back to
`masked_questions.jsonl`. It replaces the old span-level recover output schema.

---

# 1. Purpose

Each record represents one recovery sample for one masked question.

The recovery task is:

```text
recover the missing question information
```

It is not:

```text
solve the question
score recoverability
produce attention anchor labels
```

---

# 2. Record Schema

Each line in `data/processed/recover_outputs.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:recover_output -->
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
recovered_question
recovery_backend
sample_id
```

---

# 3. Field Sources

Copied from `masked_questions.jsonl`:

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
```

Created by the recovery stage:

```text
recovered_question
recovery_backend
sample_id
```

`recovered_question` may be an empty string when the backend produces no usable
recovery.

`recovery_backend` must be:

```text
oracle_stub_v0
```

This backend is a pipeline-verification stub. It may use source metadata such as
`original_question` and `spans`, so its outputs must not be interpreted as real
recovery performance.

---

# 4. ID and Grouping Rule

`masked_id` binds the record to one masked question:

```python
masked_id = f"{id}__{unit_id}__mask"
```

Multiple samples for the same masked question are represented by multiple
records with the same `masked_id` and different `sample_id` values.

---

# 5. Not Included

`recover_outputs.jsonl` must not contain:

```text
span_id
span_text
span_type
recoverable
confidence
reason
recoverability_label
attention_anchor_label
guidance_action
hidden states
attention maps
trajectory analysis
probe label
```

Recoverability labels and scores belong to a later scoring stage.

---

# 6. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_recover_output_record(record: dict) -> None
```

The validator checks required fields, rejects stale span-level and scoring
fields, verifies `masked_id`, validates unit span metadata, enforces mask and
recovery backend enums, checks that the masked question adds exactly one
`mask_token` per span, and enforces `sample_id >= 0`.
