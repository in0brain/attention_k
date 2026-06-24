"""Build ablation units from candidate span records."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_ablation_unit_record,
    validate_candidate_span_record,
)


DEFAULT_MAX_GROUP_SIZE = 8
DEFAULT_MAX_GROUP_UNITS = 10

GROUP_PRIORITY = {
    "repeated_surface": 0,
    "number_set": 1,
    "entity_set": 2,
    "object_set": 3,
    "cyber_security_term_set": 4,
}

TYPE_SET_GROUPS = {
    "number": "number_set",
    "entity": "entity_set",
    "object": "object_set",
    "cyber_security_term": "cyber_security_term_set",
}


def build_ablation_units(
    candidate_span_record: dict,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    max_group_units: int = DEFAULT_MAX_GROUP_UNITS,
) -> dict:
    """Build one ablation unit record from one candidate span record."""
    _validate_group_limits(max_group_size, max_group_units)
    validate_candidate_span_record(candidate_span_record)

    candidates = sorted(
        [dict(candidate) for candidate in candidate_span_record["candidates"]],
        key=lambda candidate: (candidate["start"], candidate["end"], candidate["span_id"]),
    )

    units = _build_single_units(candidates)
    group_units = _build_group_units(candidates, max_group_size, max_group_units)
    units.extend(group_units)
    units = _assign_unit_ids(units)

    record = {
        "id": candidate_span_record["id"],
        "question": candidate_span_record["question"],
        "units": units,
    }
    validate_ablation_unit_record(record)
    return record


def build_ablation_unit_records(
    candidate_span_records: list[dict],
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    max_group_units: int = DEFAULT_MAX_GROUP_UNITS,
) -> list[dict]:
    """Build validated ablation unit records from candidate span records."""
    return [
        build_ablation_units(
            record,
            max_group_size=max_group_size,
            max_group_units=max_group_units,
        )
        for record in candidate_span_records
    ]


def build_ablation_unit_file(
    input_path: str | Path,
    output_path: str | Path,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    max_group_units: int = DEFAULT_MAX_GROUP_UNITS,
) -> list[dict]:
    """Read candidate span records, build ablation unit records, and write JSONL."""
    candidate_span_records = read_jsonl(input_path)
    ablation_unit_records = build_ablation_unit_records(
        candidate_span_records,
        max_group_size=max_group_size,
        max_group_units=max_group_units,
    )
    write_jsonl(ablation_unit_records, output_path)
    return ablation_unit_records


def summarize_ablation_unit_records(
    records: list[dict],
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
    max_group_units: int = DEFAULT_MAX_GROUP_UNITS,
) -> dict:
    """Return ablation unit statistics for CLI output."""
    unit_counts = [len(record["units"]) for record in records]
    units = [unit for record in records for unit in record["units"]]
    single_units = [unit for unit in units if unit["unit_scope"] == "single"]
    group_units = [unit for unit in units if unit["unit_scope"] == "group"]
    group_type_counts = Counter(unit["group_type"] for unit in group_units)

    return {
        "num_questions": len(records),
        "num_candidate_spans": len(single_units),
        "num_units": len(units),
        "num_single_units": len(single_units),
        "num_group_units": len(group_units),
        "group_type_counts": dict(sorted(group_type_counts.items())),
        "questions_with_zero_units": sum(1 for count in unit_counts if count == 0),
        "max_group_size": max_group_size,
        "max_group_units": max_group_units,
    }


def _validate_group_limits(max_group_size: int, max_group_units: int) -> None:
    if not isinstance(max_group_size, int) or isinstance(max_group_size, bool) or max_group_size < 2:
        raise ValueError("max_group_size must be an int >= 2")
    if not isinstance(max_group_units, int) or isinstance(max_group_units, bool) or max_group_units < 0:
        raise ValueError("max_group_units must be an int >= 0")


def _build_single_units(candidates: list[dict]) -> list[dict]:
    return [
        {
            "unit_id": "",
            "unit_scope": "single",
            "group_type": "single",
            "span_ids": [candidate["span_id"]],
            "spans": [dict(candidate)],
            "reason": "single candidate span",
        }
        for candidate in candidates
    ]


def _build_group_units(
    candidates: list[dict],
    max_group_size: int,
    max_group_units: int,
) -> list[dict]:
    group_units: list[dict] = []
    group_units.extend(_build_repeated_surface_groups(candidates, max_group_size))
    group_units.extend(_build_type_set_groups(candidates, max_group_size))

    deduped: dict[tuple[str, ...], dict] = {}
    for unit in sorted(group_units, key=_group_sort_key):
        deduped.setdefault(tuple(unit["span_ids"]), unit)

    sorted_units = sorted(deduped.values(), key=_group_sort_key)
    return sorted_units[:max_group_units]


def _build_repeated_surface_groups(candidates: list[dict], max_group_size: int) -> list[dict]:
    by_text: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_text[candidate["text"]].append(candidate)

    groups = []
    for spans in by_text.values():
        sorted_spans = _sorted_limited_spans(spans, max_group_size)
        if len(sorted_spans) >= 2:
            groups.append(
                _make_group_unit(
                    sorted_spans,
                    "repeated_surface",
                    "same surface text appears multiple times",
                )
            )
    return groups


def _build_type_set_groups(candidates: list[dict], max_group_size: int) -> list[dict]:
    groups = []
    for span_type, group_type in TYPE_SET_GROUPS.items():
        spans = [candidate for candidate in candidates if candidate["type"] == span_type]
        sorted_spans = _sorted_limited_spans(spans, max_group_size)
        if len(sorted_spans) >= 2:
            groups.append(
                _make_group_unit(
                    sorted_spans,
                    group_type,
                    f"all {span_type} spans in the question",
                )
            )
    return groups


def _sorted_limited_spans(spans: list[dict], max_group_size: int) -> list[dict]:
    return sorted(
        [dict(span) for span in spans],
        key=lambda span: (span["start"], span["end"], span["span_id"]),
    )[:max_group_size]


def _make_group_unit(spans: list[dict], group_type: str, reason: str) -> dict:
    return {
        "unit_id": "",
        "unit_scope": "group",
        "group_type": group_type,
        "span_ids": [span["span_id"] for span in spans],
        "spans": [dict(span) for span in spans],
        "reason": reason,
    }


def _group_sort_key(unit: dict) -> tuple[int, int, tuple[str, ...]]:
    first_span_start = unit["spans"][0]["start"] if unit["spans"] else 0
    return (
        GROUP_PRIORITY.get(unit["group_type"], 100),
        first_span_start,
        tuple(unit["span_ids"]),
    )


def _assign_unit_ids(units: list[dict]) -> list[dict]:
    numbered_units = []
    for index, unit in enumerate(units, start=1):
        numbered_unit = dict(unit)
        numbered_unit["unit_id"] = f"unit_{index:03d}"
        numbered_units.append(numbered_unit)
    return numbered_units
