# Ablated Questions Interface

## 1. Purpose

This document defines the long-term interface for:

```text
data/processed/ablated_questions.jsonl
```

`ablated_questions.jsonl` is the stable output of Sprint 1C and the stable input for the next NLI-oriented sprint.

Sprint 1C consumes:

```text
data/processed/ablation_units.jsonl
```

The input contract is defined in:

```text
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
```

---

## 2. Record Schema

Each line in `data/processed/ablated_questions.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:ablated_question -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
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
```

---

## 3. Field Sources

Fields copied from Sprint 1B / `ablation_units.jsonl`:

```text
id
unit_id
unit_scope
group_type
span_ids
spans
```

Fields created by Sprint 1C:

```text
ablation_id
ablation_type
original_question
ablated_question
```

`original_question` is copied from the input record's `question` field.

---

## 4. Span Schema

Each item in `spans` must contain:

```text
span_id
text
type
start
end
```

The span offset must satisfy:

```python
original_question[start:end] == text
```

`span_ids` must have the same length and order as `spans[*].span_id`.

---

## 5. Ablation Types

Sprint 1C supports:

```text
delete
generalize
```

Do not write NLI labels, recoverability labels, attention anchor labels, or guidance actions into this file.

---

## 6. ID Convention

`ablation_id` should be deterministic and stable.

Recommended format:

```text
{id}__{unit_id}__{ablation_type}
```

Example:

```text
gsm8k_0001__unit_001__delete
```

---

## 7. Example

```json
{
  "ablation_id": "gsm8k_0001__unit_001__delete",
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
  "ablation_type": "delete",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "ablated_question": "Tom has apples and buys 2 more. How many apples does he have now?"
}
```

---

## 8. Validation Rules

The validator should reject records that violate any of these rules:

```text
1. Required fields are missing.
2. ablation_id, id, unit_id, unit_scope, group_type, ablation_type, original_question, or ablated_question is not a non-empty string.
3. unit_scope is not single or group.
4. span_ids is not a non-empty list[str].
5. spans is not a non-empty list[dict].
6. span_ids length does not equal spans length.
7. span_ids order does not match spans[*].span_id.
8. Any span misses span_id, text, type, start, or end.
9. start or end is not an int.
10. start < 0 or end <= start.
11. original_question[start:end] != span["text"].
12. unit_scope is single but span_ids length is not 1.
13. unit_scope is group but span_ids length is less than 2.
14. group_type is single but unit_scope is not single.
15. group_type is not single but unit_scope is not group.
16. ablation_type is not delete or generalize.
17. ablated_question is identical to original_question.
```

The expected validator name is:

```python
validate_ablated_question_record(record: dict) -> None
```

---

## 9. Builder Contract

The expected builder function is:

```python
build_ablated_question_records(
    ablation_unit_records: list[dict],
    ablation_types: list[str] | None = None,
    language: str = "auto",
) -> tuple[list[dict], dict]
```

Return values:

```text
records:
  Ablated question records.

stats:
  Construction statistics.
```

`stats` must include at least:

```python
{
    "num_input_questions": 0,
    "num_input_units": 0,
    "num_output_ablations": 0,
    "num_skipped_empty": 0,
    "num_skipped_unchanged": 0,
    "num_skipped_overlap": 0,
    "ablation_type_counts": {},
    "unit_scope_counts": {},
    "group_type_counts": {},
}
```

---

## 10. Module Responsibilities

```text
src/recover_attention/question_ablations.py
  Builds ablated question records from ablation unit records.

scripts/04_build_ablated_questions.py
  Reads ablation_units.jsonl, validates input, builds output records, validates output, writes ablated_questions.jsonl, and prints stats.

src/recover_attention/schemas.py
  Owns validate_ablated_question_record.
```

---

## 11. Downstream Use

The next sprint can consume `data/processed/ablated_questions.jsonl` directly for NLI semantic necessity scoring.

Downstream stages should not guess alternate field names or infer missing span offsets. If this interface and an implementation conflict, this interface is the source of truth.

This file does not define:

```text
NLI scoring
semantic necessity labels
masked question construction
question recovery
recoverability scoring
attention guidance
probe training
hidden states
trajectory analysis
```
