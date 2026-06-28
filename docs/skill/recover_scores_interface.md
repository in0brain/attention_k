# Recover Scores Interface

This document defines the stable interface for:

```text
data/processed/recover_scores.jsonl
```

`recover_scores.jsonl` stores recoverability scores aggregated from
unit-level recovery samples. It consumes:

```text
data/processed/recover_outputs.jsonl
```

This interface is `masked_id`-driven, unit-level, and self-contained for later
attention label building. It replaces the old span-level recover score schema.

---

# 1. Purpose

Each record represents one recoverability score for one masked question and one
ablation unit.

The scoring task is:

```text
summarize whether the missing unit was recoverable from recovery samples
```

It is not:

```text
run question recovery
produce attention anchor labels
produce guidance actions
cache hidden states or attention maps
```

---

# 2. Pipeline Position

Current pipeline position:

```text
masked_questions.jsonl
-> recover_outputs.jsonl
-> recover_scores.jsonl
-> attention_anchor_labels.jsonl
```

Where:

```text
recover_outputs.jsonl:
  Sprint 1G output. Contains one recovery sample per masked question and
  sample_id.

recover_scores.jsonl:
  Later scoring-stage output. Aggregates recovery samples by masked_id.
```

---

# 3. Record Schema

Each line in `data/processed/recover_scores.jsonl` must be one JSON object.

Each record must contain:

<!-- required_fields:recover_score -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
recover_score_id
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
recovery_backend
num_samples
source_sample_ids
recovered_questions
recoverability_label
recoverability_score
confidence_mean
recovery_consistency
misleading_recovery
score_backend
evidence
```

---

# 4. Field Sources

Copied or aggregated from `recover_outputs.jsonl`:

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
recovery_backend
```

Created by the scoring stage:

```text
recover_score_id
num_samples
source_sample_ids
recovered_questions
recoverability_label
recoverability_score
confidence_mean
recovery_consistency
misleading_recovery
score_backend
evidence
```

Current allowed scoring backends:

```text
stub_rule_v0
nli_recovery_judge_v0
```

Backend summary:

```text
stub_rule_v0:
  Deterministic pipeline-validation scorer.
  It normalizes original_question and recovered_question by stripping and
  collapsing whitespace, then checks exact equality.

nli_recovery_judge_v0:
  Real recovery scoring backend for existing recovery outputs.
  It compares original_question and recovered_question with bidirectional NLI:
    original_question -> recovered_question
    recovered_question -> original_question
  It uses entailment and contradiction scores to assign the existing
  recoverability labels.
  It does not call Ollama.
  It does not generate new recovery outputs.
  It does not modify recovery generation.
  It does not prove attention guidance effectiveness.
```

---

# 5. ID and Grouping Rule

`masked_id` binds the record to one masked question:

```python
masked_id = f"{id}__{unit_id}__mask"
```

`recover_score_id` binds the score to one scoring backend:

```python
recover_score_id = f"{masked_id}__score_{score_backend}"
```

The scoring stage should group `recover_outputs.jsonl` by:

```text
masked_id
```

Each group becomes one recover score record for one `score_backend`.

---

# 6. Aggregation Boundary

`recover_scores.jsonl` is an aggregated result. It must not represent a single
recovery sample.

Use:

```text
source_sample_ids
recovered_questions
num_samples
```

Do not use top-level:

```text
sample_id
recovered_question
```

---

# 7. Not Included

`recover_scores.jsonl` must not contain:

```text
span_id
span_text
span_type
sample_id
recovered_question
recoverable
confidence
reason
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
hidden states
attention maps
trajectory analysis
probe label
```

Attention labels and guidance actions belong to later stages.

---

# 8. Validator

`src/recover_attention/schemas.py` provides:

```python
validate_recover_score_record(record: dict) -> None
```

The validator checks required fields, rejects stale span-level, sample-level,
and attention-stage fields, verifies `masked_id` and `recover_score_id`,
validates unit span metadata, enforces backend enums, validates sample
aggregation fields, checks score ranges, and enforces `evidence` as a dict or
list.

---

# 9. Example

```json
{
  "recover_score_id": "gsm8k_0001__unit_001__mask__score_stub_rule_v0",
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
  "recovery_backend": "oracle_stub_v0",
  "num_samples": 1,
  "source_sample_ids": [0],
  "recovered_questions": ["Tom has 3 apples and buys 2 more."],
  "recoverability_label": "Recoverable",
  "recoverability_score": 1.0,
  "confidence_mean": 1.0,
  "recovery_consistency": 1.0,
  "misleading_recovery": false,
  "score_backend": "stub_rule_v0",
  "evidence": {
    "notes": "Example only; scoring implementation belongs to a later sprint."
  }
}
```
