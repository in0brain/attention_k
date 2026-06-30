"""Sprint 1R human review consolidation helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.schemas import (
    ALLOWED_ATTENTION_ANCHOR_LABELS,
    ALLOWED_RECOVERABILITY_LABELS,
)


HUMAN_REVIEW_FIELDS = [
    "human_review_status",
    "human_recovered_is_full_question",
    "human_masked_info_recovered",
    "human_wrong_information_introduced",
    "human_answer_changed",
    "human_masked_span_is_key",
    "human_recoverability_label",
    "human_attention_anchor_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
    "probe_usage",
    "human_notes",
    "reviewer",
    "review_date",
]

MANIFEST_FIELDS = [
    "masked_id",
    "id",
    "unit_id",
    "original_question",
    "masked_question",
    "recovered_questions",
    "human_recoverability_label",
    "human_attention_anchor_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
    "probe_usage",
]

REVIEWED_STATUS = "reviewed"
ALLOWED_REVIEW_STATUSES = {"todo", "reviewed", "uncertain", "skip"}
ALLOWED_YES_PARTIAL_NO = {"yes", "partial", "no", "uncertain", None}
ALLOWED_GUIDANCE_PRIORITIES = {"high", "medium", "low", "none", None}

DEFAULT_REVIEW_DIR = Path("outputs/logs/sprint_1Q_real_signal_quality_review")
DEFAULT_REVIEW_GUIDE = DEFAULT_REVIEW_DIR / "sprint_1Q_human_review_guide.md"
DEFAULT_LABELS_JSONL = DEFAULT_REVIEW_DIR / "sprint_1Q_human_review_labels_template.jsonl"
DEFAULT_REPORT_JSON = DEFAULT_REVIEW_DIR / "upgraded_downstream_report_with_human_fields.json"
DEFAULT_SUMMARY_JSON = DEFAULT_REVIEW_DIR / "sprint_1Q_human_review_summary.json"
DEFAULT_KNOWN_ISSUES_MD = DEFAULT_REVIEW_DIR / "sprint_1Q_known_issues.md"
DEFAULT_MANIFEST_JSONL = DEFAULT_REVIEW_DIR / "sprint_1Q_to_2A_manifest.jsonl"


def run_human_review_consolidation(
    *,
    review_guide: str | Path = DEFAULT_REVIEW_GUIDE,
    labels_jsonl: str | Path = DEFAULT_LABELS_JSONL,
    report_json: str | Path = DEFAULT_REPORT_JSON,
    summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    known_issues_md: str | Path = DEFAULT_KNOWN_ISSUES_MD,
    manifest_jsonl: str | Path = DEFAULT_MANIFEST_JSONL,
) -> dict[str, Any]:
    """Consolidate Sprint 1Q human labels into Sprint 1R artifacts."""

    labels_path = Path(labels_jsonl)
    report_path = Path(report_json)
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing required human review labels JSONL: {labels_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing required report JSON: {report_path}")

    review_records = read_jsonl(labels_path)
    if not review_records:
        raise ValueError(f"Human review labels JSONL is empty: {labels_path}")

    validate_human_review_records(review_records)
    report = read_json(report_path)
    summary = build_human_review_summary(review_records)
    validation_warnings = validate_report_human_fields_consistency(report, review_records)
    summary["validation_warning_count"] = len(validation_warnings)
    summary["validation_warnings"] = validation_warnings
    manifest_records = build_sprint_2a_manifest(review_records)

    write_json(summary, summary_json)
    ensure_dir(Path(known_issues_md).parent)
    Path(known_issues_md).write_text(build_known_issues_markdown(), encoding="utf-8")
    write_jsonl(manifest_records, manifest_jsonl)

    return {
        "reviewed_count": summary["reviewed_count"],
        "unreviewed_count": summary["unreviewed_count"],
        "manifest_count": len(manifest_records),
        "validation_warning_count": len(validation_warnings),
        "validation_warnings": validation_warnings,
        "summary": summary,
        "input_files": {
            "labels_jsonl": str(labels_path),
            "report_json": str(report_path),
        },
        "output_files": {
            "summary_json": str(summary_json),
            "known_issues_md": str(known_issues_md),
            "manifest_jsonl": str(manifest_jsonl),
        },
    }


def validate_human_review_records(records: list[dict]) -> None:
    """Validate the minimal Sprint 1R human-review fields."""

    seen_masked_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        name = f"human review record line {index}"
        for field in ["masked_id", "id", "unit_id", "original_question", "masked_question"]:
            _require_non_empty_str(record, field, name)
        if record["masked_id"] in seen_masked_ids:
            raise ValueError(f"{name} duplicate masked_id: {record['masked_id']}")
        seen_masked_ids.add(record["masked_id"])

        for field in HUMAN_REVIEW_FIELDS:
            if field not in record:
                raise ValueError(f"{name} missing human field: {field}")

        status = record["human_review_status"]
        if status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(f"{name} invalid human_review_status: {status!r}")

        if status != REVIEWED_STATUS:
            continue

        _require_bool_or_none(record, "human_recovered_is_full_question", name)
        _require_bool_or_none(record, "human_wrong_information_introduced", name)
        _require_bool_or_none(record, "human_answer_changed", name)
        _require_enum(record, "human_masked_info_recovered", ALLOWED_YES_PARTIAL_NO, name)
        _require_enum(record, "human_masked_span_is_key", ALLOWED_YES_PARTIAL_NO, name)
        _require_enum(
            record,
            "human_recoverability_label",
            ALLOWED_RECOVERABILITY_LABELS,
            name,
        )
        _require_enum(
            record,
            "human_attention_anchor_label",
            ALLOWED_ATTENTION_ANCHOR_LABELS,
            name,
        )
        _require_enum(record, "human_guidance_priority", ALLOWED_GUIDANCE_PRIORITIES, name)
        for field in [
            "human_semantic_role",
            "human_error_type",
            "probe_usage",
            "human_notes",
            "reviewer",
            "review_date",
        ]:
            _require_non_empty_str(record, field, name)
        try:
            datetime.strptime(record["review_date"], "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{name} review_date must use YYYY-MM-DD") from exc


def build_human_review_summary(records: list[dict]) -> dict[str, Any]:
    """Build the Sprint 1R human-review summary JSON."""

    reviewed = [record for record in records if record.get("human_review_status") == REVIEWED_STATUS]
    summary = {
        "reviewed_count": len(reviewed),
        "unreviewed_count": len(records) - len(reviewed),
        "total_changed_cases": len(records),
        "human_recoverability_label_counts": _count_non_empty(
            reviewed,
            "human_recoverability_label",
        ),
        "human_attention_anchor_label_counts": _count_non_empty(
            reviewed,
            "human_attention_anchor_label",
        ),
        "human_error_type_counts": _count_non_empty(reviewed, "human_error_type"),
        "probe_usage_counts": _count_non_empty(reviewed, "probe_usage"),
        "human_guidance_priority_counts": _count_non_empty(
            reviewed,
            "human_guidance_priority",
        ),
        "human_semantic_role_counts": _count_non_empty(reviewed, "human_semantic_role"),
        "auto_vs_human_recoverability_disagreement_count": _count_label_disagreements(
            reviewed,
            auto_field="upgraded_recoverability_label",
            human_field="human_recoverability_label",
        ),
        "auto_vs_human_attention_anchor_disagreement_count": _count_label_disagreements(
            reviewed,
            auto_field="upgraded_attention_anchor_label",
            human_field="human_attention_anchor_label",
        ),
        "fragment_recovery_count": _count_error_type(reviewed, "fragment_recovery"),
        "wrong_numeric_recovery_count": _count_error_type(reviewed, "wrong_numeric_recovery"),
        "generic_recovery_count": _count_error_type(reviewed, "generic_recovery"),
        "misleading_entity_or_unit_count": _count_error_type(
            reviewed,
            "misleading_entity_or_unit",
        ),
    }
    return summary


def validate_report_human_fields_consistency(report: dict, review_records: list[dict]) -> list[str]:
    """Return warnings for report/JSONL human-field mismatches without mutating inputs."""

    if not isinstance(report.get("changed_cases"), list):
        raise ValueError("report JSON must contain a changed_cases list")

    by_masked_id = _index_review_records(review_records)
    warnings: list[str] = []
    report_masked_ids: set[str] = set()
    for case in report["changed_cases"]:
        masked_id = case.get("masked_id")
        if not isinstance(masked_id, str) or not masked_id.strip():
            warnings.append("report changed_cases contains a case without non-empty masked_id")
            continue
        report_masked_ids.add(masked_id)
        source = by_masked_id.get(masked_id)
        if source is None:
            warnings.append(f"report changed_cases masked_id not found in JSONL: {masked_id}")
            continue
        for field in HUMAN_REVIEW_FIELDS:
            if case.get(field) != source.get(field):
                warnings.append(
                    f"human field mismatch for masked_id={masked_id}, field={field}: "
                    f"report={case.get(field)!r}, jsonl={source.get(field)!r}"
                )

    for record in review_records:
        masked_id = record["masked_id"]
        if masked_id not in report_masked_ids:
            warnings.append(f"JSONL masked_id not found in report changed_cases: {masked_id}")

    return warnings


def build_sprint_2a_manifest(review_records: list[dict]) -> list[dict]:
    """Build the reviewed-case manifest for Sprint 2A hidden-state caching."""

    manifest_records = []
    for record in review_records:
        if record.get("human_review_status") != REVIEWED_STATUS:
            continue
        manifest_record = {field: record[field] for field in MANIFEST_FIELDS}
        if not isinstance(manifest_record["recovered_questions"], list):
            raise ValueError(f"{record['masked_id']} recovered_questions must be a list")
        manifest_records.append(manifest_record)
    return manifest_records


def build_known_issues_markdown() -> str:
    """Return the Sprint 1R known issues freeze document."""

    return """# Sprint 1Q Known Issues Freeze

This document freezes known issues confirmed during Sprint 1Q human review.
Sprint 1R records these issues only; it does not implement recovery prompt changes,
new recovery scoring logic, hidden-state caching, probe training, or attention guidance.

## Issue 1: LLM Recovery Task Misalignment

The current recovery backend sometimes treats full-question recovery as fill-in-the-blank
generation and returns only a masked fragment, for example:

```text
How many
pages
gave
0
```

The current pipeline expects a complete `recovered_question`.

## Issue 2: Full-Question Recovery Validation Is Needed

Future scoring should add a full-question gate:

```text
if recovered output is only a fragment:
    recoverability_label = Non-recoverable
    error_type = fragment_recovery
```

## Issue 3: Span-Aware Numeric Recovery Check Is Needed

Future scoring should check concrete number consistency:

```text
if masked span is a number
and recovered question contains a different concrete number:
    recoverability_label = Misleading Recovery
    error_type = wrong_numeric_recovery
```

Confirmed examples include `9 -> 8` and `4 -> 5`.

## Issue 4: Generic Critical-Number Recovery Check Is Needed

Future scoring should distinguish generic critical-number recovery such as:

```text
7 -> some
15 -> some
```

These cases should be treated as:

```text
Non-recoverable
generic_recovery
```

They should not be automatically collapsed into `Misleading Recovery`.

## Issue 5: Entity / Unit Consistency Check Is Needed

Future scoring should identify entity or unit drift such as:

```text
pages -> books
pencils -> pens
apples -> bananas
```

These cases should be treated as:

```text
Misleading Recovery
misleading_entity_or_unit
```

## Issue 6: Unit / Group Mask Budget Control Is Needed

Single units and group units are both useful, but generating both for larger runs can
inflate the number of masked questions.

Recommended follow-up:

- Preserve group units for global entity and numeric consistency checks.
- Use single units for local localization, with sampling for low-value single units.
- Add a masked-question budget per original question.
- Lower the priority of duplicate single units after group coverage.

## Deferred Scope

- No Ollama rerun.
- No NLI rerun.
- No recovery prompt rewrite.
- No `recovery_scoring_v2`.
- No full-question recovery validator implementation.
- No span-aware numeric scorer implementation.
- No entity/unit consistency scorer implementation.
- No unit/group mask budget selector implementation.
- No masked-question rebuild.
- No probe training.
- No hidden-state cache.
"""


def read_json(path: str | Path) -> dict:
    json_path = Path(path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {json_path}")
    return data


def write_json(data: dict, path: str | Path) -> None:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _index_review_records(records: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        masked_id = record["masked_id"]
        if masked_id in indexed:
            raise ValueError(f"duplicate masked_id: {masked_id}")
        indexed[masked_id] = record
    return indexed


def _count_non_empty(records: list[dict], field: str) -> dict[str, int]:
    counts = Counter(
        str(record[field])
        for record in records
        if record.get(field) is not None and str(record.get(field)).strip()
    )
    return dict(sorted(counts.items()))


def _count_label_disagreements(records: list[dict], *, auto_field: str, human_field: str) -> int:
    return sum(
        1
        for record in records
        if record.get(auto_field) is not None
        and record.get(human_field) is not None
        and record[auto_field] != record[human_field]
    )


def _count_error_type(records: list[dict], error_type: str) -> int:
    return sum(1 for record in records if record.get("human_error_type") == error_type)


def _require_non_empty_str(record: dict, field: str, name: str) -> None:
    if field not in record:
        raise ValueError(f"{name} missing required field: {field}")
    if not isinstance(record[field], str) or not record[field].strip():
        raise ValueError(f"{name} field {field!r} must be a non-empty str")


def _require_bool_or_none(record: dict, field: str, name: str) -> None:
    value = record[field]
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} field {field!r} must be a bool or null")


def _require_enum(record: dict, field: str, allowed: set[Any], name: str) -> None:
    value = record[field]
    if value not in allowed:
        allowed_values = ", ".join(repr(item) for item in sorted(allowed, key=str))
        raise ValueError(
            f"{name} field {field!r} has invalid value {value!r}; "
            f"allowed values: {allowed_values}"
        )
