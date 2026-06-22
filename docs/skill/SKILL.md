---

name: recover-attention-experiment
description: Use this skill when developing the Recoverability-Guided Attention Allocation experiment. It guides Codex through small, staged implementation tasks for span extraction, ablated question construction, NLI semantic necessity scoring, masked question construction, question recovery, and recoverability scoring. It must prevent premature work on probes, attention guidance, hidden states, trajectory analysis, or large-scale experiments.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Recover Attention Experiment Skill

## 1. Purpose

This skill governs development of the **Recoverability-Guided Attention Allocation** experiment.

The experiment studies which input spans are semantically necessary, difficult for a model to recover, or likely to trigger hallucination-like behavior when missing.

The current development goal is not model performance.
The current goal is to build a stable, inspectable, jsonl-based experimental pipeline.

---

## 2. Always Read First

Before executing any task, read these files in order:

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/codex_tasks/<current_task_card>.md
```

If these files disagree, follow this priority:

```text
User instruction
> Current task card
> AGENTS.md
> PROGRESS.md
> docs/skill/SKILL.md
> docs/reference/*
```

`docs/reference/*` is long-term reference only.
Do not expand the current task based on reference documents unless the user explicitly asks.

---

## 3. Current Scope

Current scope is limited to v0 / v1 pipeline development:

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring
```

Current out-of-scope items:

```text
probe training
attention guidance
hidden states
trajectory analysis
large-scale experiments
GPU inference
model downloading
external API calls
```

Do not implement out-of-scope items unless the current task card explicitly asks for them.

---

## 4. Development Mode

Use small-step agile development.

Each sprint must satisfy:

```text
one goal
few files
clear input
clear output
clear command
clear acceptance criteria
PROGRESS.md update
```

Do not start the next sprint automatically.

Do not modify files outside the current task card unless necessary to keep the project runnable.

If a task card is too broad, stop and report that it should be split.

---

## 5. Data Format Rule

All intermediate experiment artifacts must use `jsonl`.

Each line is one valid JSON object.

Do not use csv, pickle, sqlite, parquet, or binary formats for v0/v1 intermediate artifacts unless the user explicitly asks.

All scripts should print:

```text
input path
output path
number of records read
number of records written
basic validation result
```

---

## 6. Canonical File Flow

The canonical v0/v1 file flow is:

```text
data/examples/questions_small.jsonl
→ data/processed/questions.jsonl
→ data/processed/candidate_spans.jsonl
→ data/processed/ablated_questions.jsonl
→ data/processed/nli_scores.jsonl
→ data/processed/masked_questions.jsonl
→ data/processed/recover_outputs.jsonl
→ data/processed/recover_scores.jsonl
→ data/processed/token_labels.jsonl
```

Do not skip earlier files unless the task card explicitly provides a replacement input.

---

## 7. Canonical Schemas

### 7.1 questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "dataset": "gsm8k",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "gold_answer": "5"
}
```

Required fields:

```text
id
dataset
question
gold_answer
```

---

### 7.2 candidate_spans.jsonl

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "candidates": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    }
  ]
}
```

Required candidate fields:

```text
span_id
text
type
start
end
```

Early supported span types:

```text
number
entity
operation
relation
question_target
```

Later optional span types:

```text
comparison
negation
condition
cyber_security_term
```

---

### 7.3 ablated_questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "ablated_question": "Tom has some apples and buys 2 more. How many apples does he have now?"
}
```

Allowed `ablation_type` values:

```text
delete
generalize
mask
```

For NLI semantic necessity, prefer:

```text
delete
generalize
```

For mask-recover, prefer:

```text
mask
```

---

### 7.4 nli_scores.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_to_ablated": "entailment",
  "ablated_to_original": "neutral",
  "semantic_necessity_label": "Information Loss"
}
```

Allowed NLI directional labels:

```text
entailment
neutral
contradiction
```

Allowed semantic necessity labels:

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

Semantic necessity rule:

```text
original → ablated = entailment
ablated → original = entailment
=> Equivalent
```

```text
original → ablated = entailment
ablated → original != entailment
=> Information Loss
```

```text
original → ablated != entailment
ablated → original = entailment
=> Added Assumption
```

```text
original → ablated != entailment
ablated → original != entailment
=> Non-equivalent
```

Interpretation:

```text
Equivalent: low semantic necessity
Information Loss: high semantic necessity
Added Assumption: rewrite introduced extra assumptions
Non-equivalent: semantic change is substantial
```

---

### 7.5 masked_questions.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?"
}
```

Use `[MASK]` as the default mask token.

---

### 7.6 recover_outputs.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "sample_id": 0,
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
  "recovered_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "recoverable": "yes",
  "confidence": 0.82,
  "reason": "The missing number is likely recoverable from the original context pattern."
}
```

Do not implement recovery unless the current task card asks for it.

---

### 7.7 recover_scores.jsonl

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "recoverability_label": "Non-recoverable",
  "confidence_mean": 0.41,
  "recovery_consistency": 0.22,
  "misleading_recovery": false
}
```

Allowed recoverability labels:

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

---

## 8. Core Method Rules

### 8.1 Candidate Span Extraction

Early implementation should be simple and rule-based.

Prioritize extracting:

```text
numbers
named entities or simple capitalized names
operation words
relation words
question target phrases
```

Do not call an LLM for span extraction in early sprints.

---

### 8.2 Ablated Question Construction

Use deterministic transformations first.

Recommended early behavior:

```text
number → generalize to "some" or "a certain number"
person/entity → generalize to "someone" or "something"
operation/relation → delete or mask
question_target → mask or preserve carefully
```

Do not optimize fluency before the pipeline is runnable.

---

### 8.3 NLI Scoring

Early NLI implementation may use a stub backend if the task card says so.

Do not download or run an NLI model unless the current task card explicitly asks.

Always preserve both directions:

```text
original question → ablated question
ablated question → original question
```

---

### 8.4 Mask-Recover

When implemented later, recovery prompts must instruct the model:

```text
recover the question only
do not solve the problem
do not provide the final answer
return structured fields
```

Each masked question should support multiple samples `K`.

Do not implement this until the task card asks.

---

## 9. Label Comparison Logic

The most valuable later analysis compares semantic necessity with recoverability:

```text
NLI important + Recover non-recoverable
=> true key span

NLI important + Recover recoverable
=> model may infer or guess missing information; inspect for high-confidence wrong recovery

NLI unimportant + Recover non-recoverable
=> model over-sensitive to semantically minor span

NLI unimportant + Recover recoverable
=> stable low-risk span
```

Do not implement analysis tables until the task card asks.

---

## 10. File Ownership Rules

### Human-maintained files

Do not modify unless explicitly asked:

```text
README.md
AGENTS.md
docs/skill/SKILL.md
docs/reference/*
```

### Codex-maintained file

Update after every completed sprint:

```text
PROGRESS.md
```

### Task-card-controlled files

Only modify files listed in the current task card.

---

## 11. Progress Update Rule

After completing a sprint, update `PROGRESS.md` with:

```text
sprint name
completed work
new or modified files
input files
output files
commands run
test/check results
known issues
next recommended sprint
```

Move completed tasks from “next steps” into “current progress” when appropriate.

Do not claim research conclusions without real outputs.

---

## 12. Error Handling Rule

If expected inputs are missing:

1. Do not invent experimental data.
2. Report the missing file.
3. If the task card allows sample creation, create only the specified minimal sample.
4. Otherwise stop and ask for the required input.

If a reference document is missing:

1. Do not invent it.
2. Continue if it is not required for the current task.
3. Record that it was absent.

---

## 13. Acceptance Philosophy

A sprint is complete only when:

```text
the requested files exist
the requested command runs
the output format matches the schema
the tests or smoke checks pass
PROGRESS.md is updated
no next-stage work was started
```

Partial implementation is acceptable only if clearly recorded in `PROGRESS.md`.

---

## 14. Current Next Sprint

The next recommended sprint is:

```text
Sprint 0B: jsonl data I/O + sample data + smoke test
```

Expected task card:

```text
docs/codex_tasks/sprint_0B_data_io.md
```

Do not start Sprint 0B unless the user explicitly asks.
