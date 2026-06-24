---

name: reasoning-aware-attention-guidance
description: Use this skill when working on the Reasoning-Aware Attention Guidance project. This skill is only the entry point and document router. It tells Codex which project documents to read, how to resolve document priority, and where to find task-specific guidance. It does not define the sprint roadmap, canonical pipeline order, schema details, or current next task.
---

# Reasoning-Aware Attention Guidance Skill

## 1. Role of This Skill

This file is the Skill entry point for the project.

It only defines:

```text
1. which documents Codex should read first
2. where different kinds of project knowledge live
3. how to resolve conflicts between documents
4. when to use reference documents
5. what this file must not contain
```

This file does not define:

```text
1. the current sprint roadmap
2. the current next task
3. the canonical pipeline order
4. detailed jsonl schemas
5. method definitions
6. experiment stage details
7. prompt templates
8. implementation requirements
```

Those belong to the routed documents listed below.

---

## 2. Always Read First

Before executing any task, Codex must read these files in order:

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/codex_tasks/<current_task_card>.md
```

The current task card is the direct execution boundary.

If the user gives a direct instruction that functions as the task card, Codex must treat the current user instruction as the direct execution boundary.

---

## 3. Document Router

Use the following documents according to the task.

### 3.1 AGENTS.md

Use for:

```text
Codex behavior rules
Preflight rules
environment rules
file ownership rules
coding style rules
scope control
progress update rules
forbidden actions
```

Do not duplicate AGENTS.md rules here.

---

### 3.2 PROGRESS.md

Use for:

```text
current project state
completed sprint summary
latest runnable commands
current unresolved issues
next recommended sprint
```

PROGRESS.md is a current-state index, not a full historical log.

Detailed progress history should live under:

```text
docs/progress/
```

---

### 3.3 docs/codex_tasks/*.md

Use for:

```text
the current sprint goal
allowed files
forbidden files
required inputs
expected outputs
implementation details
tests and checks
completion format
```

The current task card has priority over general guidance documents.

Do not start a sprint unless the user explicitly asks Codex to execute that sprint or provides a current task card.

---

### 3.4 docs/skill/codex_tasks.md

Use for:

```text
task card conventions
sprint execution rules
Preflight expectations
task splitting principles
scope boundaries
```

This file may contain lightweight planning notes, but it is not the source of truth for the current sprint.

If it conflicts with the current task card, follow the current task card and report the conflict.

---

### 3.5 docs/skill/experiment_guide.md

Use for:

```text
experiment flow
stage input and output relationships
directory conventions
stage boundaries
running order when explicitly needed
```

This file explains the experiment process.

It should not override the current task card.

---

### 3.6 docs/skill/method.md

Use for:

```text
method concepts
terminology
signal roles
conceptual boundaries
common methodological misunderstandings
```

This file explains what concepts mean.

It should not define current implementation scope.

---

### 3.7 docs/skill/label_schema.md

Use for:

```text
jsonl schemas
field names
field meanings
allowed enum values
record examples
schema synchronization notes
```

This file defines data formats.

Code implementation still follows the current task card.

---

### 3.8 docs/skill/prompts.md

Use for:

```text
reusable prompt templates
structured output templates
LLM instruction templates
judge prompt templates
recovery prompt templates
```

This file only provides reusable prompt text.

Do not treat prompt templates as permission to call a model unless the current task card explicitly allows model usage.

---

### 3.8.1 docs/skill/ablation_units_interface.md

Use for:

```text
Ablation unit 的长期接口文档，说明 ablation_units.jsonl schema、字段含义、实例、模块职责和后续消费方式。
```

---

### 3.8.2 docs/skill/ablated_questions_interface.md

Use for:

```text
Ablated question 的长期接口文档，说明 ablated_questions.jsonl schema、字段含义、字段来源、实例、模块职责和后续 NLI 消费方式。
```

---

### 3.8.3 docs/skill/nli_scores_interface.md

Use for:

```text
NLI score 的长期接口文档，说明 nli_scores.jsonl schema、双向 NLI 字段、stub backend、language 参数、字段来源和后续 label rule 消费方式。
```

---

### 3.8.4 docs/skill/semantic_labels_interface.md

Use for:

```text
Semantic label interface for semantic_labels.jsonl, including rule_v0, thresholds, label enum, field sources, validation, and downstream consumption.
```

---

### 3.9 docs/reference/*

Use for:

```text
long-form project plans
full experiment guides
archived design documents
background reference material
```

Reference documents are not read by default.

They are only used when explicitly allowed.

---

## 4. Conflict Priority

Conflict priority is defined in AGENTS.md.

---

## 5. Reference Documents Rule

Do not read `docs/reference/*` by default.

Only read `docs/reference/*` when:

```text
1. the current user instruction explicitly asks for it
2. the current task card explicitly asks for it
3. the current task cannot be completed without it and the user approves using it
```

Even when reference documents are read, do not implement future-stage content unless the current task card explicitly permits it.

---

## 6. Subdocument Reading Rule

Do not read all `docs/skill/*.md` files by default.

Read a skill subdocument only when:

```text
1. the current task card asks for it
2. the current user instruction asks for it
3. it is directly necessary to resolve the current task
```

If a subdocument is missing, report it in Preflight.

Do not invent missing document content.

---

## 7. Current Task Rule

The current task is determined by:

```text
1. the current user instruction
2. the current task card
3. PROGRESS.md current-state index
```

This file must not contain a hard-coded current next sprint.

This file must not contain a fixed sprint roadmap.

This file must not contain a canonical pipeline order.

---

## 8. Scope Rule

Do not use this Skill file to expand the task scope.

If a routed document mentions future work, that future work is not allowed unless the current task card explicitly permits it.

Examples:

```text
A schema document may mention future records.
A method document may mention attention guidance.
A reference plan may mention probe training.
```

These mentions are informational only.

They are not execution permission.

---

## 9. Preflight Routing Checklist

Before modifying files, Codex should use this routing checklist:

```text
1. Read AGENTS.md.
2. Read PROGRESS.md.
3. Read docs/skill/SKILL.md.
4. Read the current task card.
5. Read only the additional documents requested by the task card.
6. Report any missing required documents.
7. Report any conflicts between the task card and other documents.
8. Follow the current task card unless the current user instruction overrides it.
```

The detailed Preflight format is defined in AGENTS.md and/or the current task card.

---

## 10. What This File Must Not Contain

This file must not contain:

```text
1. a hard-coded sprint roadmap
2. a hard-coded current next sprint
3. a canonical file flow
4. detailed schema definitions
5. full method explanations
6. full prompt templates
7. implementation instructions for a specific sprint
8. claims about experimental results
9. instructions to call real models
10. instructions to train probes
11. instructions to run attention guidance
```

If any of the above is needed, put it in the appropriate routed document instead.

---

## 11. Minimal Completion Principle

A sprint is complete only according to the current task card.

This Skill file does not define sprint completion criteria.

Completion criteria should be read from:

```text
docs/codex_tasks/<current_task_card>.md
```

or from the current user instruction when it acts as the task card.
