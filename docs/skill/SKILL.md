---

name: reasoning-aware-attention-guidance
description: Use this skill when developing the Reasoning-Aware Attention Guidance experiment. It guides Codex through small, staged implementation tasks for baseline CoT, candidate span extraction, NLI semantic necessity, semantic recoverability, trajectory stability, attention anchor labeling, oracle attention guidance, and probe-guided attention guidance. It must prevent premature work on model downloading, large-scale inference, hidden-state caching, attention steering, probe training, or paper-level evaluation unless explicitly allowed by the current task card.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Reasoning-Aware Attention Guidance Skill

## 1. Purpose

This skill governs development of the **Reasoning-Aware Attention Guidance** experiment.

The project is not a plain hallucination classifier.

The project is also not merely key span discovery.

The project aims to build a staged experimental pipeline:

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

The final experimental target is to discover and validate attention anchors for reasoning-time attention guidance.

Expected downstream outputs include:

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

NLI and recoverability are auxiliary signals for attention importance discovery. They are not the final goal.

---

## 2. Always Read First

Before executing any task, Codex must read these files in order:

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/codex_tasks/<current_task_card>.md
```

If the current task card explicitly requires additional skill subdocuments, read them after `docs/skill/SKILL.md`.

Common skill subdocuments:

```text
docs/skill/codex_tasks.md
docs/skill/experiment_guide.md
docs/skill/method.md
docs/skill/label_schema.md
docs/skill/prompts.md
```

Do not read all subdocuments by default unless the current task card requires them.

---

## 3. Skill Subdocuments

The following files provide lightweight execution knowledge.

```text
docs/skill/codex_tasks.md
Sprint route, task card rules, execution boundaries, and long-term roadmap.

docs/skill/experiment_guide.md
Experiment flow, stage inputs and outputs, directory conventions, and running boundaries.

docs/skill/method.md
Method concepts, terminology boundaries, signal roles, and common misunderstandings.

docs/skill/label_schema.md
jsonl schemas, field meanings, allowed labels, and schema synchronization notes.

docs/skill/prompts.md
Reusable Codex prompt templates.
```

Default rule:

```text
Read docs/skill/SKILL.md first.
Read a subdocument only when the current task card asks for it or when it is directly relevant.
Do not treat subdocuments as permission to expand scope.
If a subdocument conflicts with the current task card, follow the current task card and report the conflict.
```

---

## 4. Conflict Priority

If instructions conflict, follow this priority:

```text
Current user instruction
> Current task card
> AGENTS.md
> PROGRESS.md
> docs/skill/SKILL.md
> docs/skill/*.md
> docs/skill/prompts.md
> docs/reference/*
```

If Codex finds a conflict, it must:

```text
1. Report the conflict in Preflight.
2. State which instruction it will follow.
3. Record the conflict again in the final “Known Issues” or “遗留问题” section.
```

---

## 5. Reference Documents Rule

`docs/reference/*` contains long-term full reference documents.

Do not use `docs/reference/*` as the default execution source.

Only read `docs/reference/*` when:

```text
1. The current task card explicitly requires it.
2. The current user instruction explicitly asks for it.
3. The current task is impossible to resolve without it and the user has approved using it.
```

Even when reference documents are read, do not implement future-stage content unless the current task card explicitly permits it.

---

## 6. Current Development Philosophy

Use small-step agile development.

Each sprint must satisfy:

```text
one goal
few files
clear input
clear output
clear command
clear tests or checks
PROGRESS.md update
no automatic next sprint
```

If a task card is too broad, stop and report that it should be split.

Do not modify files outside the current task card unless the task would otherwise become unrunnable. If such a modification is necessary, report it before editing whenever possible.

---

## 7. Current High-Level Roadmap

The current staged roadmap is:

```text
Sprint 0:
Engineering foundation and Skill framework.

Sprint 1:
Baseline CoT and reasoning trajectory foundation.

Sprint 2:
Candidate Span and NLI semantic necessity.

Sprint 3:
Mask / Remove Intervention and semantic recoverability.

Sprint 4:
Trajectory stability and answer stability.

Sprint 5:
Attention importance hierarchy and attention anchor labeling.

Sprint 6:
Oracle attention guidance.

Sprint 7:
Probe-guided attention guidance.

Sprint 8:
Hallucination reduction evaluation.
```

Important stage boundaries:

```text
Sprint 0 only builds engineering and documentation foundations.
Sprint 1 may use stub or fixture outputs and should not require real model inference by default.
Sprint 2 may use rule-based span extraction and NLI stub first.
Sprint 3 may use recovery stub first.
Do not implement attention guidance before Sprint 6.
Do not train probes before Sprint 7.
Do not run large-scale experiments unless a task card explicitly allows it.
```

---

## 8. Current Scope Control

Current work must be limited to the current sprint.

Do not start the next sprint automatically.

Do not implement any of the following unless the current task card explicitly asks:

```text
real model downloading
real model inference
external API calls
large-scale data processing
large-scale hidden-state caching
large-scale attention-map caching
trajectory stability scoring
attention steering
probe training
paper-level evaluation
```

Especially do not implement the following during Sprint 0, Sprint 1, Sprint 2, or Sprint 3:

```text
oracle attention guidance
probe-guided attention guidance
```

---

## 9. Data Format Rule

All intermediate experiment artifacts should use:

```text
jsonl
```

Each line must be one valid JSON object.

Do not use the following as primary intermediate formats:

```text
csv
excel
pickle
sqlite
parquet
```

Binary tensor files such as `.pt` are allowed only for hidden states or attention maps when the task card explicitly permits caching.

Large tensors must not be embedded directly into jsonl records. Store paths in manifest files instead.

Detailed schemas are defined in:

```text
docs/skill/label_schema.md
```

Do not duplicate full schema definitions in this file.

---

## 10. Canonical File Flow

The full target pipeline is:

```text
data/examples/questions_small.jsonl
→ data/processed/questions.jsonl
→ data/processed/baseline_cot.jsonl
→ data/processed/baseline_trajectory_manifest.jsonl
→ data/processed/candidate_spans.jsonl
→ data/processed/ablated_questions.jsonl
→ data/processed/nli_scores.jsonl
→ data/processed/masked_questions.jsonl
→ data/processed/recover_outputs.jsonl
→ data/processed/recover_scores.jsonl
→ data/processed/intervention_manifest.jsonl
→ data/processed/trajectory_stability_scores.jsonl
→ data/processed/answer_stability_scores.jsonl
→ data/processed/attention_anchor_labels.jsonl
→ data/processed/oracle_guidance_results.jsonl
→ data/processed/probe_guidance_results.jsonl
→ outputs/evaluation/*
```

Do not skip earlier files unless the current task card explicitly provides a replacement input.

Do not create future-stage files before the corresponding sprint.

---

## 11. Core Method Boundaries

### 11.1 Candidate Span

Candidate span extraction only identifies possible reasoning-relevant spans.

It does not determine final importance.

It does not produce attention anchors.

### 11.2 NLI Semantic Necessity

NLI semantic necessity estimates whether deleting, generalizing, or replacing a span changes the question semantics.

NLI is an auxiliary signal.

It does not replace recoverability, trajectory stability, answer stability, or attention steering.

### 11.3 Semantic Recoverability

Semantic recoverability estimates whether a masked span can be stably recovered from context.

Recovery must recover the question only.

Do not solve the problem inside recovery prompts.

### 11.4 Trajectory Stability

Trajectory stability studies whether span intervention changes the model’s reasoning trajectory.

Do not implement it before the corresponding task card.

### 11.5 Attention Anchor Labeling

Attention anchor labeling combines multiple signals into:

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

Do not decide anchor labels from a single signal alone.

### 11.6 Attention Guidance

Attention guidance is a later validation and intervention stage.

Oracle guidance validates whether known anchors help.

Probe-guided guidance is a later automatic method.

Do not implement either early.

---

## 12. Common Misunderstandings

Avoid the following mistakes:

```text
NLI is the final task.
Recoverability is the final task.
Key span discovery is the final task.
A hallucination classifier is the project goal.
Information Loss automatically means Strong Anchor.
Non-recoverable automatically means Strong Anchor.
Equivalent automatically means Distractor.
High raw attention automatically means important.
Low raw attention automatically means unimportant.
Oracle guidance is the deployable final method.
Probe training can start before anchor labels are stable.
```

Correct principle:

```text
Multiple signals form evidence.
Evidence supports attention_importance_score.
attention_importance_score supports attention_anchor_label.
attention_anchor_label supports guidance_action and guidance_strength.
Attention guidance experiments validate whether those labels are useful.
```

---

## 13. Environment Rules

Default environment:

```text
conda env: recover_attention
python: 3.10
```

Dependency installation must use:

```bash
python -m pip install -r requirements.txt
```

Testing must use:

```bash
python -m pytest -q
```

Do not use:

```bash
pip install ...
pytest -q
```

Do not use:

```text
.venv
```

unless the user explicitly changes the environment plan.

If Python or pip points to the wrong environment, stop and report.

---

## 14. Preflight Rule

Before modifying files, Codex must output a Preflight.

Preflight must include:

```text
1. Files read
2. Files allowed to modify
3. Files forbidden to modify
4. Commands to run
5. Whether docs/reference/* is needed
6. Detected conflicts, if any
```

After Preflight, Codex must pause and wait for user confirmation.

Do not modify files before confirmation.

---

## 15. File Ownership Rules

Human-maintained files:

```text
README.md
AGENTS.md
docs/skill/SKILL.md
docs/skill/*.md
docs/reference/*
```

Codex may modify these only when the current user instruction and current task card explicitly allow it.

Codex-maintained file:

```text
PROGRESS.md
```

Update after every completed sprint.

Task-card-controlled files:

```text
Only modify files listed in the current task card.
```

---

## 16. Progress Update Rule

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

Keep PROGRESS.md concise.
Use PROGRESS.md as the current-state index.
Archive detailed sprint logs under docs/progress/.
Do not let PROGRESS.md grow indefinitely.

Do not claim research conclusions without real outputs.

Do not claim attention guidance improves performance unless an actual guidance evaluation has been run.

---

## 17. Error Handling Rule

If expected inputs are missing:

```text
1. Do not invent experimental data.
2. Report the missing file.
3. If the task card allows sample creation, create only the specified minimal sample.
4. Otherwise stop and ask for the required input.
```

If a command fails:

```text
1. Do not continue to the next stage.
2. Report the failed command.
3. Report the error message.
4. Report the current Python path.
5. Report modified files.
6. Do not expand the modification scope without user approval.
```

If a reference document is missing:

```text
1. Do not invent it.
2. Continue only if it is not required for the current task.
3. Record that it was absent.
```

---

## 18. Acceptance Philosophy

A sprint is complete only when:

```text
the requested files exist
the requested command runs
the output format matches the relevant schema
the tests or smoke checks pass
PROGRESS.md is updated
no next-stage work was started
```

Partial implementation is acceptable only if clearly recorded in `PROGRESS.md`.

---

## 19. Current Next Step

The current next step should be determined from `PROGRESS.md` and the current user instruction.

Do not rely on a hard-coded “Current Next Sprint” in this file.

Do not start any sprint unless the user explicitly asks.
