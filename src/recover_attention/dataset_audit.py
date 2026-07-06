"""Audit candidate data sources for full-scale weak-labeled experiments.

This module walks a set of search roots, inspects every JSONL / JSON / CSV /
Parquet file it finds, and classifies whether each file can serve as a *raw
question source* for a full-scale (500 / 2000 case) weak-labeled experiment.

It does not modify any data. It only reads and reports. The key distinction it
draws is between:

* raw question sources -- independent ``(question, answer)`` records that can
  seed the Sprint 2G pipeline, and
* derived downstream artifacts -- rows that fan out over candidate spans /
  ablation units / NLI pairs of a *small* number of source questions, and that
  therefore must not be counted as independent cases.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_BACKEND = "dataset_source_audit_v0"
DEFAULT_SEARCH_ROOTS = ("data", "outputs/logs")
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".csv", ".parquet")

TARGET_500 = 500
TARGET_2000 = 2000

# Fields whose presence marks a record as a derived downstream artifact rather
# than a raw source question.
DERIVED_FIELD_MARKERS = frozenset(
    {
        "candidates",
        "units",
        "ablation_id",
        "unit_id",
        "span_ids",
        "spans",
        "original_question",
        "masked_question",
        "masked_id",
        "nli_id",
        "semantic_label_id",
        "recover_score_id",
        "unit_evidence_id",
        "attention_anchor_label_id",
        "intervention_id",
        "feature_id",
        "probe_record_id",
        "hidden_state_path",
        "hidden_states_path",
    }
)

RAW_QUESTION_FIELDS = frozenset({"question"})
RAW_ANSWER_FIELDS = frozenset({"answer", "gold_answer"})


def _read_records(path: Path) -> tuple[list[dict], int, str | None]:
    """Read records from a data file tolerantly.

    Returns ``(records, num_records, read_error)``. ``records`` may be a sample
    capped for very large files, but ``num_records`` always reflects the true
    record count when it can be determined.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return _read_jsonl(path)
        if suffix == ".json":
            return _read_json(path)
        if suffix == ".csv":
            return _read_csv(path)
        if suffix == ".parquet":
            return _read_parquet(path)
    except OSError as exc:
        return [], 0, f"could not read file: {exc}"
    return [], 0, f"unsupported suffix: {suffix}"


def _read_jsonl(path: Path) -> tuple[list[dict], int, str | None]:
    records: list[dict] = []
    num_records = 0
    decode_errors = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            num_records += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    error = f"{decode_errors} undecodable line(s)" if decode_errors else None
    return records, num_records, error


def _read_json(path: Path) -> tuple[list[dict], int, str | None]:
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if isinstance(parsed, list):
        records = [item for item in parsed if isinstance(item, dict)]
        return records, len(parsed), None
    if isinstance(parsed, dict):
        return [parsed], 1, None
    return [], 0, "json root is neither list nor object"


def _read_csv(path: Path) -> tuple[list[dict], int, str | None]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [dict(row) for row in reader]
    return records, len(records), None


def _read_parquet(path: Path) -> tuple[list[dict], int, str | None]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return [], 0, "parquet file present but pyarrow is not installed"
    table = pq.read_table(path)
    records = table.to_pylist()
    return records, table.num_rows, None


def _collect_fields(records: list[dict]) -> list[str]:
    fields: set[str] = set()
    for record in records:
        fields.update(record.keys())
    return sorted(fields)


def _classify_source_type(
    path: Path,
    fields: list[str],
    num_records: int,
    read_error: str | None,
) -> str:
    if read_error and num_records == 0:
        return "unreadable"
    if num_records == 0:
        return "empty"

    name = path.name.lower()
    field_set = set(fields)

    if name.endswith("_report.json") or "report" in name or "manifest" in name:
        # Manifests can still carry question/answer, but they are pipeline
        # bookkeeping rather than a primary question dataset.
        if not (RAW_QUESTION_FIELDS & field_set and RAW_ANSWER_FIELDS & field_set):
            return "report_or_manifest"

    if "human_review" in name or "human" in name:
        return "human_review_labels"

    if field_set & DERIVED_FIELD_MARKERS:
        return "derived_downstream"

    if RAW_QUESTION_FIELDS & field_set and RAW_ANSWER_FIELDS & field_set:
        return "raw_question_source"

    return "other"


_NOT_USABLE_REASONS = {
    "derived_downstream": (
        "derived downstream artifact: rows fan out over candidate spans / "
        "ablation units / NLI pairs of a small number of source questions; "
        "not independent source questions"
    ),
    "report_or_manifest": "report / manifest file, not a question dataset",
    "human_review_labels": (
        "human-reviewed label template: diagnostic baseline / calibration only, "
        "must not be used as full-scale main supervision"
    ),
    "empty": "file contains no records",
    "unreadable": "file could not be read",
    "other": "no (question, answer) record structure detected",
}


def audit_file(path: Path, root: Path) -> dict[str, Any]:
    """Audit a single data file and return its descriptor."""
    records, num_records, read_error = _read_records(path)
    fields = _collect_fields(records)
    field_set = set(fields)
    source_type = _classify_source_type(path, fields, num_records, read_error)
    usable = source_type == "raw_question_source"

    has_question = bool(RAW_QUESTION_FIELDS & field_set)
    has_answer = bool(RAW_ANSWER_FIELDS & field_set)

    descriptor: dict[str, Any] = {
        "path": path.as_posix(),
        "file_type": path.suffix.lower().lstrip("."),
        "num_records": num_records,
        "available_fields": fields,
        "has_question_field": has_question,
        "has_answer_field": has_answer,
        "candidate_source_type": source_type,
        "usable_for_full_scale": usable,
        "reason_if_not_usable": None if usable else _NOT_USABLE_REASONS.get(
            source_type, "not a raw question source"
        ),
    }

    # For derived artifacts, expose the fan-out ratio so the audit can explain
    # why row counts (e.g. 92) overstate the number of real source questions.
    if source_type == "derived_downstream" and records:
        distinct_ids = {
            record.get("id") for record in records if isinstance(record.get("id"), str)
        }
        if distinct_ids:
            descriptor["num_distinct_source_ids"] = len(distinct_ids)
    if read_error:
        descriptor["read_error"] = read_error
    return descriptor


def _iter_data_files(search_roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for suffix in SUPPORTED_SUFFIXES:
            found.extend(root.rglob(f"*{suffix}"))
    # Deduplicate while keeping a stable, deterministic order.
    unique = sorted({path.resolve(): path for path in found if path.is_file()}.values(),
                    key=lambda p: p.as_posix())
    return unique


def _build_fan_out_explanation(
    files: list[dict[str, Any]],
    max_usable_records: int,
    max_usable_path: str | None,
) -> str:
    derived = [f for f in files if f["candidate_source_type"] == "derived_downstream"]
    if not derived:
        return (
            "No derived downstream artifacts were found; row counts reflect raw "
            "record counts directly."
        )
    largest = max(derived, key=lambda f: f["num_records"])
    distinct = largest.get("num_distinct_source_ids")
    distinct_clause = (
        f"only {distinct} distinct source question id(s)"
        if distinct is not None
        else "a small number of distinct source question ids"
    )
    source_clause = (
        f"The largest raw question source is {max_usable_path} with "
        f"{max_usable_records} record(s)."
        if max_usable_path is not None
        else "No raw question source was found at all."
    )
    return (
        f"The largest file by row count is {largest['path']} with "
        f"{largest['num_records']} rows, but it contains {distinct_clause}. "
        "Those rows are downstream fan-out over candidate spans / ablation units "
        "/ NLI pairs, not independent source questions. "
        f"{source_clause} This is why visible JSONL row counts (around 92) do "
        "not represent 92 usable source questions."
    )


def build_kfold_precondition(max_usable_records: int) -> dict[str, Any]:
    """Build the two-layer k-fold feasibility precondition block."""
    return {
        "source_scale_check": {
            "num_source_questions": max_usable_records,
            "can_run_500": max_usable_records >= TARGET_500,
            "can_run_2000": max_usable_records >= TARGET_2000,
            "can_run_all": max_usable_records > 0,
        },
        "future_kfold_requirement": {
            "note": (
                "Future k-fold must be decided after weak probe labels are "
                "generated, not on the raw question dataset."
            ),
            "rule": "num_folds <= min_class_count among usable probe targets",
            "do_not_prefix_5_fold_unless": "min weak-label class count >= 5",
            "auto_downgrade_chain": "5-fold -> 3-fold -> 2-fold -> leave-one-out / holdout fallback",
            "decided_now": False,
            "reason_not_decided_now": (
                "Weak probe labels do not exist yet; per-class counts are unknown "
                "until candidate spans -> ablation units -> masked questions -> "
                "recovery / weak scoring -> weak probe labels has been run."
            ),
        },
    }


def audit_dataset_sources(
    search_roots: list[str | Path],
    backend: str = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Audit all candidate data files under ``search_roots``."""
    roots = [Path(root) for root in search_roots]
    data_files = _iter_data_files(roots)
    files = [audit_file(path, path) for path in data_files]

    raw_sources = [f for f in files if f["usable_for_full_scale"]]
    raw_sources_sorted = sorted(
        raw_sources, key=lambda f: f["num_records"], reverse=True
    )
    max_usable_records = (
        raw_sources_sorted[0]["num_records"] if raw_sources_sorted else 0
    )
    max_usable_path = raw_sources_sorted[0]["path"] if raw_sources_sorted else None

    max_any = max((f["num_records"] for f in files), default=0)
    max_any_path = next(
        (f["path"] for f in files if f["num_records"] == max_any), None
    )

    sources_500 = [f for f in raw_sources_sorted if f["num_records"] >= TARGET_500]
    sources_2000 = [f for f in raw_sources_sorted if f["num_records"] >= TARGET_2000]

    can_run_500 = bool(sources_500)
    can_run_2000 = bool(sources_2000)

    shortfall_reason = None
    if not can_run_500:
        if max_usable_records == 0:
            shortfall_reason = "no local raw question source dataset found"
        else:
            shortfall_reason = (
                f"largest raw question source has only {max_usable_records} "
                f"record(s) (< {TARGET_500}); no local full-scale source dataset found"
            )

    report: dict[str, Any] = {
        "backend": backend,
        "search_roots": [root.as_posix() for root in roots],
        "num_files_scanned": len(files),
        "files": files,
        "raw_question_sources": [
            {"path": f["path"], "num_records": f["num_records"]}
            for f in raw_sources_sorted
        ],
        "max_records_any_file": max_any,
        "max_records_any_file_path": max_any_path,
        "available_num_cases": max_usable_records,
        "max_usable_source_records": max_usable_records,
        "max_usable_source_path": max_usable_path,
        "target_num_cases_500": TARGET_500,
        "target_num_cases_2000": TARGET_2000,
        "can_run_500": can_run_500,
        "can_run_2000": can_run_2000,
        "can_run_all": max_usable_records > 0,
        "shortfall_reason": shortfall_reason,
        "recommended_source_path_500": sources_500[0]["path"] if sources_500 else None,
        "recommended_source_path_2000": (
            sources_2000[0]["path"] if sources_2000 else None
        ),
        "fan_out_explanation": _build_fan_out_explanation(
            files, max_usable_records, max_usable_path
        ),
        "kfold_precondition": build_kfold_precondition(max_usable_records),
    }
    return report


def render_audit_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from an audit report dict."""
    lines: list[str] = []
    lines.append("# Dataset Source Audit")
    lines.append("")
    lines.append(f"- backend: `{report['backend']}`")
    lines.append(f"- search_roots: {', '.join(report['search_roots'])}")
    lines.append(f"- num_files_scanned: {report['num_files_scanned']}")
    lines.append("")
    lines.append("## Scale Feasibility")
    lines.append("")
    lines.append(f"- available_num_cases (max usable source records): "
                 f"**{report['available_num_cases']}**")
    lines.append(f"- max_usable_source_path: "
                 f"`{report['max_usable_source_path']}`")
    lines.append(f"- max_records_any_file: {report['max_records_any_file']} "
                 f"(`{report['max_records_any_file_path']}`)")
    lines.append(f"- target_num_cases_500: {report['target_num_cases_500']}")
    lines.append(f"- target_num_cases_2000: {report['target_num_cases_2000']}")
    lines.append(f"- can_run_500: **{report['can_run_500']}**")
    lines.append(f"- can_run_2000: **{report['can_run_2000']}**")
    lines.append(f"- can_run_all: **{report['can_run_all']}**")
    if report["shortfall_reason"]:
        lines.append(f"- shortfall_reason: {report['shortfall_reason']}")
    lines.append(f"- recommended_source_path_500: "
                 f"`{report['recommended_source_path_500']}`")
    lines.append(f"- recommended_source_path_2000: "
                 f"`{report['recommended_source_path_2000']}`")
    lines.append("")
    lines.append("## Why the largest visible JSONL is only ~92 rows")
    lines.append("")
    lines.append(report["fan_out_explanation"])
    lines.append("")
    lines.append("## Raw Question Sources")
    lines.append("")
    if report["raw_question_sources"]:
        lines.append("| path | num_records |")
        lines.append("| --- | --- |")
        for source in report["raw_question_sources"]:
            lines.append(f"| `{source['path']}` | {source['num_records']} |")
    else:
        lines.append("_No raw question source files were found._")
    lines.append("")
    lines.append("## All Candidate Files")
    lines.append("")
    lines.append("| path | type | records | has_q | has_a | source_type | usable |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for descriptor in report["files"]:
        lines.append(
            f"| `{descriptor['path']}` | {descriptor['file_type']} | "
            f"{descriptor['num_records']} | {descriptor['has_question_field']} | "
            f"{descriptor['has_answer_field']} | "
            f"{descriptor['candidate_source_type']} | "
            f"{descriptor['usable_for_full_scale']} |"
        )
    lines.append("")
    lines.append("## K-fold Feasibility Precondition")
    lines.append("")
    kfold = report["kfold_precondition"]
    scale = kfold["source_scale_check"]
    lines.append("### Layer 1: source scale check")
    lines.append("")
    lines.append(f"- num_source_questions: {scale['num_source_questions']}")
    lines.append(f"- can_run_500: {scale['can_run_500']}")
    lines.append(f"- can_run_2000: {scale['can_run_2000']}")
    lines.append(f"- can_run_all: {scale['can_run_all']}")
    lines.append("")
    future = kfold["future_kfold_requirement"]
    lines.append("### Layer 2: future k-fold requirement")
    lines.append("")
    lines.append(f"- {future['note']}")
    lines.append(f"- rule: `{future['rule']}`")
    lines.append(f"- do_not_prefix_5_fold_unless: {future['do_not_prefix_5_fold_unless']}")
    lines.append(f"- auto_downgrade_chain: `{future['auto_downgrade_chain']}`")
    lines.append(f"- decided_now: {future['decided_now']}")
    lines.append(f"- reason_not_decided_now: {future['reason_not_decided_now']}")
    lines.append("")
    return "\n".join(lines)
