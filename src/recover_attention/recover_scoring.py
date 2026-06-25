"""Build unit-level recoverability score records from recovery outputs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_recover_output_record,
    validate_recover_score_record,
)


DEFAULT_SCORE_BACKEND = "stub_rule_v0"
SUPPORTED_SCORE_BACKENDS = {"stub_rule_v0"}

RECOVER_SCORE_SOURCE_FIELDS = (
    "masked_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "masked_question",
    "mask_token",
    "mask_backend",
    "mask_strategy",
    "recovery_backend",
)

GROUP_CONSISTENCY_FIELDS = RECOVER_SCORE_SOURCE_FIELDS


def normalize_question(text: str) -> str:
    """Normalize question text for the Sprint 1H exact-match stub rule."""
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    return " ".join(text.strip().split())


def score_recovery_group_stub_rule_v0(recover_output_records: list[dict]) -> dict:
    """Score one masked_id group using deterministic normalized exact match."""
    _validate_non_empty_group(recover_output_records)
    for record in recover_output_records:
        validate_recover_output_record(record)
    _validate_group_consistency(recover_output_records)
    _validate_unique_sample_ids(recover_output_records)

    sorted_records = _sorted_group_records(recover_output_records)
    original_question = sorted_records[0]["original_question"]
    normalized_original = normalize_question(original_question)

    normalized_recoveries = [
        normalize_question(record["recovered_question"]) for record in sorted_records
    ]
    exact_match_count = sum(
        normalized_recovery == normalized_original
        for normalized_recovery in normalized_recoveries
    )
    empty_count = sum(normalized_recovery == "" for normalized_recovery in normalized_recoveries)
    non_empty_mismatch_count = sum(
        normalized_recovery != "" and normalized_recovery != normalized_original
        for normalized_recovery in normalized_recoveries
    )

    num_samples = len(sorted_records)
    recoverability_score = exact_match_count / num_samples
    normalized_counts = Counter(normalized_recoveries)
    recovery_consistency = max(normalized_counts.values()) / num_samples
    misleading_recovery = non_empty_mismatch_count > 0

    if exact_match_count == num_samples:
        recoverability_label = "Recoverable"
    elif exact_match_count > 0:
        recoverability_label = "Partially Recoverable"
    elif non_empty_mismatch_count > 0:
        recoverability_label = "Misleading Recovery"
    else:
        recoverability_label = "Non-recoverable"

    return {
        "num_samples": num_samples,
        "source_sample_ids": [record["sample_id"] for record in sorted_records],
        "recovered_questions": [record["recovered_question"] for record in sorted_records],
        "recoverability_label": recoverability_label,
        "recoverability_score": recoverability_score,
        "confidence_mean": recoverability_score,
        "recovery_consistency": recovery_consistency,
        "misleading_recovery": misleading_recovery,
        "evidence": {
            "rule_name": "stub_rule_v0",
            "normalization": "strip_and_collapse_whitespace",
            "num_exact_matches": exact_match_count,
            "num_empty_recoveries": empty_count,
            "num_non_empty_mismatches": non_empty_mismatch_count,
            "exact_match_ratio": recoverability_score,
            "unique_normalized_recoveries": dict(sorted(normalized_counts.items())),
            "notes": (
                "Deterministic exact-match stub for pipeline validation only; "
                "not a semantic recovery metric."
            ),
        },
    }


def build_recover_score_record(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> dict:
    """Build one validated recover score record for one masked_id group."""
    _validate_score_backend(score_backend)
    _validate_non_empty_group(recover_output_records)

    for record in recover_output_records:
        validate_recover_output_record(record)
    _validate_group_consistency(recover_output_records)
    _validate_unique_sample_ids(recover_output_records)

    first_record = recover_output_records[0]
    score_fields = _score_group(recover_output_records, score_backend)
    recover_score_record = {
        field: deepcopy(first_record[field])
        for field in RECOVER_SCORE_SOURCE_FIELDS
    }
    recover_score_record.update(
        {
            "recover_score_id": f"{first_record['masked_id']}__score_{score_backend}",
            "score_backend": score_backend,
            **score_fields,
        }
    )

    validate_recover_score_record(recover_score_record)
    return recover_score_record


def build_recover_score_records(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> tuple[list[dict], dict]:
    """Group recovery outputs by masked_id and build recover score records."""
    _validate_score_backend(score_backend)

    grouped_records: dict[str, list[dict]] = {}
    for record in recover_output_records:
        validate_recover_output_record(record)
        grouped_records.setdefault(record["masked_id"], []).append(record)

    score_records = [
        build_recover_score_record(records_for_masked_id, score_backend=score_backend)
        for records_for_masked_id in grouped_records.values()
    ]

    stats = _build_stats(recover_output_records, score_records, score_backend)
    return score_records, stats


def build_recover_score_file(
    input_path: str | Path,
    output_path: str | Path,
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> tuple[list[dict], dict]:
    """Read recovery outputs, build recover scores, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing input: {input_jsonl}\nPlease run Sprint 1G first."
        )

    recover_output_records = read_jsonl(input_jsonl)
    records, stats = build_recover_score_records(
        recover_output_records,
        score_backend=score_backend,
    )
    write_jsonl(records, output_path)
    return records, stats


def _score_group(recover_output_records: list[dict], score_backend: str) -> dict:
    if score_backend == "stub_rule_v0":
        return score_recovery_group_stub_rule_v0(recover_output_records)
    _validate_score_backend(score_backend)
    raise AssertionError("unreachable score backend validation state")


def _sorted_group_records(recover_output_records: list[dict]) -> list[dict]:
    return sorted(recover_output_records, key=lambda record: record["sample_id"])


def _validate_score_backend(score_backend: str) -> None:
    if score_backend not in SUPPORTED_SCORE_BACKENDS:
        raise ValueError(f"Unsupported score backend: {score_backend}")


def _validate_non_empty_group(recover_output_records: list[dict]) -> None:
    if not isinstance(recover_output_records, list) or not recover_output_records:
        raise ValueError("recover_output_records must be a non-empty list")


def _validate_group_consistency(recover_output_records: list[dict]) -> None:
    first_record = recover_output_records[0]
    for record_index, record in enumerate(recover_output_records[1:], start=1):
        for field in GROUP_CONSISTENCY_FIELDS:
            if record[field] != first_record[field]:
                raise ValueError(
                    "recover output records for one masked_id must have consistent "
                    f"{field}; mismatch at record index {record_index}"
                )


def _validate_unique_sample_ids(recover_output_records: list[dict]) -> None:
    sample_ids = [record["sample_id"] for record in recover_output_records]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sample_id values for masked_id group: {duplicates}")


def _build_stats(
    recover_output_records: list[dict],
    score_records: list[dict],
    score_backend: str,
) -> dict:
    return {
        "num_input_recoveries": len(recover_output_records),
        "num_output_scores": len(score_records),
        "score_backend": score_backend,
        "recovery_backend_counts": dict(
            sorted(Counter(record["recovery_backend"] for record in recover_output_records).items())
        ),
        "recoverability_label_counts": dict(
            sorted(Counter(record["recoverability_label"] for record in score_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in score_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in score_records).items())
        ),
        "num_misleading_recovery": sum(
            1 for record in score_records if record["misleading_recovery"]
        ),
    }
