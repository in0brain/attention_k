"""Build unit-level masked question records from semantic label records."""

from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_masked_question_record,
    validate_semantic_label_record,
)


DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_MASK_BACKEND = "unit_mask_v0"
DEFAULT_MASK_STRATEGY = "replace_each_span"
SUPPORTED_MASK_BACKENDS = {"unit_mask_v0"}
SUPPORTED_MASK_STRATEGIES = {"replace_each_span"}

UNIT_STRUCTURE_FIELDS = (
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
)
SEMANTIC_SOURCE_FIELDS = (
    "semantic_label_id",
    "nli_id",
    "ablation_id",
    "ablation_type",
    "semantic_necessity_label",
    "semantic_necessity_score",
    "is_semantically_necessary",
    "decision_reason",
)
SOURCE_ABLATION_ORDER = {"delete": 0, "generalize": 1}


def apply_mask_to_unit(
    question: str,
    unit: dict,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> str:
    """Replace every span in a unit with one mask token."""
    _validate_mask_token(mask_token)
    masked_question = question
    for span in sorted(unit["spans"], key=lambda item: item["start"], reverse=True):
        start = span["start"]
        end = span["end"]
        if start < 0 or end > len(question) or end <= start:
            raise ValueError(
                f"span {span.get('span_id', '<unknown>')} has invalid offsets "
                f"for question length {len(question)}"
            )
        if question[start:end] != span["text"]:
            raise ValueError(
                f"span {span.get('span_id', '<unknown>')} offsets do not match "
                "original_question text"
            )
        masked_question = masked_question[:start] + mask_token + masked_question[end:]
    return masked_question


def build_masked_question_record(
    semantic_label_records_for_unit: list[dict],
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    strategy: str = DEFAULT_MASK_STRATEGY,
) -> dict:
    """Build one validated masked question record for a single unit."""
    _validate_backend(backend)
    _validate_strategy(strategy)
    _validate_mask_token(mask_token)
    if not semantic_label_records_for_unit:
        raise ValueError("semantic_label_records_for_unit must be non-empty")

    for record in semantic_label_records_for_unit:
        validate_semantic_label_record(record)
    _validate_same_unit_structure(semantic_label_records_for_unit)

    base = semantic_label_records_for_unit[0]
    unit = {
        "spans": deepcopy(base["spans"]),
    }
    semantic_sources = [
        _semantic_source(record)
        for record in sorted(
            semantic_label_records_for_unit,
            key=_semantic_source_sort_key,
        )
    ]

    record = {
        "masked_id": f"{base['id']}__{base['unit_id']}__mask",
        "id": base["id"],
        "unit_id": base["unit_id"],
        "unit_scope": base["unit_scope"],
        "group_type": base["group_type"],
        "span_ids": list(base["span_ids"]),
        "spans": deepcopy(base["spans"]),
        "original_question": base["original_question"],
        "masked_question": apply_mask_to_unit(
            base["original_question"],
            unit,
            mask_token=mask_token,
        ),
        "mask_token": mask_token,
        "mask_backend": backend,
        "mask_strategy": strategy,
        "source_semantic_label_ids": [
            source["semantic_label_id"] for source in semantic_sources
        ],
        "source_nli_ids": [source["nli_id"] for source in semantic_sources],
        "source_ablation_ids": [source["ablation_id"] for source in semantic_sources],
        "semantic_sources": semantic_sources,
    }
    validate_masked_question_record(record)
    return record


def build_masked_question_records(
    semantic_label_records: list[dict],
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    strategy: str = DEFAULT_MASK_STRATEGY,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]:
    """Build masked question records and return records plus summary stats."""
    _validate_backend(backend)
    _validate_strategy(strategy)
    _validate_mask_token(mask_token)

    grouped_records: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()
    for record in semantic_label_records:
        validate_semantic_label_record(record)
        key = (record["id"], record["unit_id"])
        grouped_records.setdefault(key, []).append(record)

    stats = _empty_stats(
        semantic_label_records,
        grouped_records,
        mask_token,
        backend,
        strategy,
        only_necessary,
    )
    output_records: list[dict] = []

    for records_for_unit in grouped_records.values():
        if only_necessary and not any(
            record["is_semantically_necessary"] for record in records_for_unit
        ):
            stats["num_filtered_not_necessary"] += 1
            continue
        if _has_overlapping_spans(records_for_unit[0]["spans"]):
            stats["num_skipped_overlap"] += 1
            continue

        masked_record = build_masked_question_record(
            records_for_unit,
            mask_token=mask_token,
            backend=backend,
            strategy=strategy,
        )
        output_records.append(masked_record)
        stats["num_output_masks"] += 1

    stats["unit_scope_counts"] = dict(sorted(stats["unit_scope_counts"].items()))
    stats["group_type_counts"] = dict(sorted(stats["group_type_counts"].items()))
    stats["source_count_distribution"] = dict(
        sorted(stats["source_count_distribution"].items())
    )
    return output_records, stats


def build_masked_question_file(
    input_path: str | Path,
    output_path: str | Path,
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    strategy: str = DEFAULT_MASK_STRATEGY,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]:
    """Read semantic labels, build masked questions, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing input: {input_jsonl}\nPlease run Sprint 1E first."
        )
    semantic_label_records = read_jsonl(input_jsonl)
    records, stats = build_masked_question_records(
        semantic_label_records,
        mask_token=mask_token,
        backend=backend,
        strategy=strategy,
        only_necessary=only_necessary,
    )
    write_jsonl(records, output_path)
    return records, stats


def _empty_stats(
    semantic_label_records: list[dict],
    grouped_records: OrderedDict[tuple[str, str], list[dict]],
    mask_token: str,
    backend: str,
    strategy: str,
    only_necessary: bool,
) -> dict:
    return {
        "num_input_labels": len(semantic_label_records),
        "num_units": len(grouped_records),
        "num_output_masks": 0,
        "num_filtered_not_necessary": 0,
        "num_skipped_overlap": 0,
        "mask_token": mask_token,
        "mask_backend": backend,
        "mask_strategy": strategy,
        "only_necessary": only_necessary,
        "unit_scope_counts": Counter(
            records_for_unit[0]["unit_scope"]
            for records_for_unit in grouped_records.values()
        ),
        "group_type_counts": Counter(
            records_for_unit[0]["group_type"]
            for records_for_unit in grouped_records.values()
        ),
        "source_count_distribution": Counter(
            len(records_for_unit) for records_for_unit in grouped_records.values()
        ),
    }


def _semantic_source(record: dict) -> dict:
    return {field: deepcopy(record[field]) for field in SEMANTIC_SOURCE_FIELDS}


def _semantic_source_sort_key(record: dict) -> tuple[int, str, str]:
    ablation_type = record["ablation_type"]
    return (
        SOURCE_ABLATION_ORDER.get(ablation_type, len(SOURCE_ABLATION_ORDER)),
        ablation_type,
        record["semantic_label_id"],
    )


def _validate_same_unit_structure(records: list[dict]) -> None:
    base = records[0]
    for record in records[1:]:
        for field in UNIT_STRUCTURE_FIELDS:
            if record[field] != base[field]:
                raise ValueError(
                    f"inconsistent unit structure for id={base['id']} "
                    f"unit_id={base['unit_id']}: field '{field}' differs"
                )


def _has_overlapping_spans(spans: list[dict]) -> bool:
    previous_end = -1
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if span["start"] < previous_end:
            return True
        previous_end = max(previous_end, span["end"])
    return False


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_MASK_BACKENDS:
        allowed = ", ".join(sorted(SUPPORTED_MASK_BACKENDS))
        raise ValueError(f"Unsupported backend: {backend}; allowed: {allowed}")


def _validate_strategy(strategy: str) -> None:
    if strategy not in SUPPORTED_MASK_STRATEGIES:
        allowed = ", ".join(sorted(SUPPORTED_MASK_STRATEGIES))
        raise ValueError(f"Unsupported mask_strategy: {strategy}; allowed: {allowed}")


def _validate_mask_token(mask_token: str) -> None:
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError("mask_token must be a non-empty string")
