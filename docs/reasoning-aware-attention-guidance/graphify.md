# Project Knowledge Graph (Graphify)

## 1. Purpose and Boundary

`graphify-out/` contains a generated knowledge graph of this repository's code
and documentation. It supports project navigation, file-relationship review,
and documentation audit.

It is not part of the Reasoning-Aware Attention Guidance experiment pipeline.
It does not create a new dataset, label, model metric, causal result, or
authorization to run a future sprint.

In particular, graph relationships must not be used to claim that attention
guidance is effective, hallucination is reduced, or answer accuracy improved.

## 2. Current Generated Snapshot

The current graph was generated for the repository root on 2026-07-12.

| Item | Recorded value |
| --- | --- |
| Corpus | 259 supported files, about 322,602 words |
| Graph | 3,172 nodes and 7,838 edges |
| Communities | 148 |
| Recorded Graphify token cost | 0 input / 0 output tokens |

The most connected nodes reported in this snapshot are:

1. `ensure_dir()` (75 edges)
2. `validate_recover_score_record()` (43 edges)
3. `read_jsonl()` (40 edges)
4. `validate_intervention_manifest_record()` (39 edges)
5. `validate_attention_anchor_label_record()` (36 edges)

These counts describe repository connectivity, not experimental importance or
causal importance.

## 3. Artifact Locations

```text
graphify-out/graph.html       interactive local graph
graphify-out/GRAPH_REPORT.md  generated audit report
graphify-out/graph.json       machine-readable graph data
graphify-out/manifest.json    scan manifest for incremental updates
graphify-out/cost.json        Graphify run-cost tracker
```

Artifacts are generated outputs. A later Graphify rebuild can replace them;
use the current `GRAPH_REPORT.md` rather than this document for detailed or
time-sensitive findings.

## 4. How to Use It

When a graph already exists, ask a repository question through Graphify query
instead of rebuilding it. Examples include tracing a call path, locating
consumers of a schema validator, or finding links between a task card and its
implementation.

```text
/graphify query "What calls validate_recover_score_record?"
/graphify path "read_jsonl" "validate_attention_anchor_label_record"
/graphify explain "ensure_dir"
```

Use a fresh `/graphify .` build only when the user asks to regenerate the
project graph. Use `/graphify . --update` for a requested incremental refresh.
Both operations must still follow the project's Preflight and file-ownership
rules.

## 5. Integrity and Review Rules

The current graph-health audit reported:

```text
1,069 dangling-endpoint edges
957 collapsed undirected edges
```

The graph remains useful for navigation, but these warnings mean it can be
incomplete or may have merged multiple raw relations into one visual edge.
Treat all graph paths as leads, then verify the referenced code or document.

Graph confidence has the following meaning:

```text
EXTRACTED: explicit in a source file
INFERRED: a reasoned relationship that needs source verification
AMBIGUOUS: uncertain; do not rely on it without direct review
```

The recorded zero token cost covers Graphify's stored tracker only. It does
not establish that all semantic-extraction work was token-free.

## 6. Routing

Read this document when the task concerns:

```text
the repository knowledge graph
graphify-out artifacts
code or document relationship navigation
Graphify graph-health warnings
incremental graph refresh or graph queries
```

For experimental scope, results, schemas, and task execution boundaries,
continue to follow `AGENTS.md`, `PROGRESS.md`, the current task card, and the
other routed documents in `SKILL.md`.
