"""Build unit-level evidence records from semantic and recoverability signals."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_recover_score_record,
    validate_semantic_label_record,
    validate_unit_evidence_record,
)


DEFAULT_UNIT_EVIDENCE_BACKEND = "aggregate_stub_v0"
SUPPORTED_UNIT_EVIDENCE_BACKENDS = {"aggregate_stub_v0"}
EVIDENCE_STATUS = "partial_stub_evidence"
AVAILABLE_SIGNAL_TYPES = [
    "semantic_necessity",
    "semantic_recoverability",
]
MISSING_SIGNAL_TYPES = [
    "trajectory_stability",
    "answer_stability",
    "raw_attention_pattern",
    "attention_steering_effect",
]

UNIT_METADATA_FIELDS = (
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
)


def build_semantic_evidence_summary(semantic_label_records: list[dict]) -> dict:
    """Aggregate semantic necessity records for one ``(id, unit_id)`` group."""
    _validate_non_empty_records(semantic_label_records, "semantic_label_records")
    for record in semantic_label_records:
        validate_semantic_label_record(record)
    _validate_unit_metadata_consistency(semantic_label_records, context="semantic label group")

    sorted_records = _sort_semantic_label_records(semantic_label_records)
    semantic_scores = [
        record["semantic_necessity_score"] for record in sorted_records
    ]
    necessary_votes = [
        record["is_semantically_necessary"] for record in sorted_records
    ]
    num_semantically_necessary = sum(necessary_votes)
    num_semantic_records = len(sorted_records)

    if num_semantically_necessary == 0:
        summary_label = "no_semantic_necessity_evidence"
    elif num_semantically_necessary == num_semantic_records:
        summary_label = "consistent_semantic_necessity_evidence"
    else:
        summary_label = "mixed_semantic_necessity_evidence"

    return {
        "source_semantic_label_ids": [
            record["semantic_label_id"] for record in sorted_records
        ],
        "source_nli_ids": [record["nli_id"] for record in sorted_records],
        "source_ablation_ids": [
            record["ablation_id"] for record in sorted_records
        ],
        "ablation_types": [record["ablation_type"] for record in sorted_records],
        "semantic_necessity_labels": [
            record["semantic_necessity_label"] for record in sorted_records
        ],
        "semantic_necessity_scores": semantic_scores,
        "is_semantically_necessary_votes": necessary_votes,
        "num_semantic_records": num_semantic_records,
        "num_semantically_necessary": num_semantically_necessary,
        "summary_score": max(semantic_scores),
        "summary_label": summary_label,
        "semantic_label_backend": sorted_records[0]["semantic_label_backend"],
        "nli_backends": sorted({record["nli_backend"] for record in sorted_records}),
        "language_settings": sorted(
            {record["language_setting"] for record in sorted_records}
        ),
    }


def build_recoverability_evidence_summary(recover_score_record: dict) -> dict:
    """Summarize one unit-level recover score record as evidence."""
    validate_recover_score_record(recover_score_record)
    return {
        "recover_score_id": recover_score_record["recover_score_id"],
        "masked_id": recover_score_record["masked_id"],
        "recovery_backend": recover_score_record["recovery_backend"],
        "score_backend": recover_score_record["score_backend"],
        "recoverability_label": recover_score_record["recoverability_label"],
        "recoverability_score": recover_score_record["recoverability_score"],
        "confidence_mean": recover_score_record["confidence_mean"],
        "recovery_consistency": recover_score_record["recovery_consistency"],
        "misleading_recovery": recover_score_record["misleading_recovery"],
        "num_samples": recover_score_record["num_samples"],
        "source_sample_ids": deepcopy(recover_score_record["source_sample_ids"]),
        "source_recovered_questions": deepcopy(
            recover_score_record["recovered_questions"]
        ),
        "score_evidence": deepcopy(recover_score_record["evidence"]),
    }


def build_unit_evidence_record(
    semantic_label_records: list[dict],
    recover_score_record: dict,
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> dict:
    """Build one validated unit evidence record for one ``(id, unit_id)``."""
    _validate_backend(evidence_backend)
    _validate_non_empty_records(semantic_label_records, "semantic_label_records")
    for record in semantic_label_records:
        validate_semantic_label_record(record)
    validate_recover_score_record(recover_score_record)
    _validate_unit_metadata_consistency(semantic_label_records, context="semantic label group")

    sorted_semantic_records = _sort_semantic_label_records(semantic_label_records)
    first_semantic_record = sorted_semantic_records[0]
    _validate_metadata_matches_recover_score(first_semantic_record, recover_score_record)

    semantic_evidence = build_semantic_evidence_summary(sorted_semantic_records)
    recoverability_evidence = build_recoverability_evidence_summary(recover_score_record)

    unit_evidence_record = {
        "unit_evidence_id": (
            f"{first_semantic_record['id']}__{first_semantic_record['unit_id']}"
            f"__evidence_{evidence_backend}"
        ),
        "id": first_semantic_record["id"],
        "unit_id": first_semantic_record["unit_id"],
        "unit_scope": first_semantic_record["unit_scope"],
        "group_type": first_semantic_record["group_type"],
        "span_ids": deepcopy(first_semantic_record["span_ids"]),
        "spans": deepcopy(first_semantic_record["spans"]),
        "original_question": first_semantic_record["original_question"],
        "semantic_evidence": semantic_evidence,
        "recoverability_evidence": recoverability_evidence,
        "available_signal_types": list(AVAILABLE_SIGNAL_TYPES),
        "missing_signal_types": list(MISSING_SIGNAL_TYPES),
        "evidence_backend": evidence_backend,
        "evidence_status": EVIDENCE_STATUS,
        "evidence": {
            "source_files": [
                "data/processed/semantic_labels.jsonl",
                "data/processed/recover_scores.jsonl",
            ],
            "join_key": {
                "id": first_semantic_record["id"],
                "unit_id": first_semantic_record["unit_id"],
            },
            "available_signal_types": list(AVAILABLE_SIGNAL_TYPES),
            "missing_signal_types": list(MISSING_SIGNAL_TYPES),
            "limitations": [
                "recoverability comes from oracle_stub_v0 + stub_rule_v0",
                "no trajectory stability yet",
                "no answer stability yet",
                "no raw attention pattern yet",
                "no attention steering effect yet",
                "unit_evidence is not final attention anchor label",
            ],
            "notes": (
                "Unit-level early evidence aggregation for pipeline validation; "
                "not a final attention importance decision."
            ),
        },
    }

    validate_unit_evidence_record(unit_evidence_record)
    return unit_evidence_record


def build_unit_evidence_records(
    semantic_label_records: list[dict],
    recover_score_records: list[dict],
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> tuple[list[dict], dict]:
    """Join semantic labels with recover scores and build unit evidence records."""
    _validate_backend(evidence_backend)
    semantic_groups = _group_semantic_labels_by_unit(semantic_label_records)
    recover_scores_by_unit = _index_recover_scores_by_unit(recover_score_records)

    semantic_keys = set(semantic_groups)
    recover_keys = set(recover_scores_by_unit)
    missing_recover_scores = sorted(semantic_keys - recover_keys)
    extra_recover_scores = sorted(recover_keys - semantic_keys)
    if missing_recover_scores:
        key = missing_recover_scores[0]
        raise ValueError(
            f"semantic group missing recover_score for id={key[0]}, unit_id={key[1]}"
        )
    if extra_recover_scores:
        key = extra_recover_scores[0]
        raise ValueError(
            f"recover_score has no semantic group for id={key[0]}, unit_id={key[1]}"
        )

    unit_evidence_records = [
        build_unit_evidence_record(
            semantic_groups[key],
            recover_scores_by_unit[key],
            evidence_backend=evidence_backend,
        )
        for key in sorted(semantic_groups)
    ]

    stats = _build_stats(
        semantic_label_records,
        recover_score_records,
        unit_evidence_records,
        evidence_backend,
    )
    return unit_evidence_records, stats


def build_unit_evidence_file(
    semantic_labels_path: str | Path,
    recover_scores_path: str | Path,
    output_path: str | Path,
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> tuple[list[dict], dict]:
    """Read inputs, build unit evidence records, and write JSONL output."""
    semantic_labels_jsonl = Path(semantic_labels_path)
    recover_scores_jsonl = Path(recover_scores_path)
    if not semantic_labels_jsonl.exists():
        raise FileNotFoundError(
            f"Missing semantic labels input: {semantic_labels_jsonl}\n"
            "Please run Sprint 1E first."
        )
    if not recover_scores_jsonl.exists():
        raise FileNotFoundError(
            f"Missing recover scores input: {recover_scores_jsonl}\n"
            "Please run Sprint 1H first."
        )

    semantic_label_records = read_jsonl(semantic_labels_jsonl)
    recover_score_records = read_jsonl(recover_scores_jsonl)
    records, stats = build_unit_evidence_records(
        semantic_label_records,
        recover_score_records,
        evidence_backend=evidence_backend,
    )
    write_jsonl(records, output_path)
    return records, stats


def _group_semantic_labels_by_unit(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped_records: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        validate_semantic_label_record(record)
        grouped_records.setdefault(_unit_key(record), []).append(record)
    return grouped_records


def _index_recover_scores_by_unit(records: list[dict]) -> dict[tuple[str, str], dict]:
    indexed_records: dict[tuple[str, str], dict] = {}
    for record in records:
        validate_recover_score_record(record)
        key = _unit_key(record)
        if key in indexed_records:
            raise ValueError(
                f"duplicate recover_score records for id={key[0]}, unit_id={key[1]}"
            )
        indexed_records[key] = record
    return indexed_records


def _validate_backend(evidence_backend: str) -> None:
    if evidence_backend not in SUPPORTED_UNIT_EVIDENCE_BACKENDS:
        raise ValueError(f"Unsupported unit evidence backend: {evidence_backend}")


def _unit_key(record: dict) -> tuple[str, str]:
    return record["id"], record["unit_id"]


def _validate_non_empty_records(records: list[dict], name: str) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} must be a non-empty list")


def _sort_semantic_label_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda record: (
            record["id"],
            record["unit_id"],
            record["ablation_type"],
            record["semantic_label_id"],
        ),
    )


def _validate_unit_metadata_consistency(records: list[dict], context: str) -> None:
    first_record = records[0]
    id_value = first_record["id"]
    unit_id = first_record["unit_id"]
    for record in records[1:]:
        for field in UNIT_METADATA_FIELDS:
            if record[field] != first_record[field]:
                raise ValueError(
                    f"metadata mismatch for id={id_value}, unit_id={unit_id}, "
                    f"field={field} in {context}"
                )


def _validate_metadata_matches_recover_score(
    semantic_record: dict,
    recover_score_record: dict,
) -> None:
    id_value = semantic_record["id"]
    unit_id = semantic_record["unit_id"]
    for field in UNIT_METADATA_FIELDS:
        if recover_score_record[field] != semantic_record[field]:
            raise ValueError(
                f"metadata mismatch for id={id_value}, unit_id={unit_id}, "
                f"field={field} between semantic_labels and recover_score"
            )


def _build_stats(
    semantic_label_records: list[dict],
    recover_score_records: list[dict],
    unit_evidence_records: list[dict],
    evidence_backend: str,
) -> dict:
    return {
        "num_semantic_labels": len(semantic_label_records),
        "num_recover_scores": len(recover_score_records),
        "num_output_unit_evidence": len(unit_evidence_records),
        "evidence_backend": evidence_backend,
        "available_signal_types": list(AVAILABLE_SIGNAL_TYPES),
        "missing_signal_types": list(MISSING_SIGNAL_TYPES),
        "evidence_status_counts": dict(
            sorted(
                Counter(
                    record["evidence_status"] for record in unit_evidence_records
                ).items()
            )
        ),
        "unit_scope_counts": dict(
            sorted(
                Counter(
                    record["unit_scope"] for record in unit_evidence_records
                ).items()
            )
        ),
        "group_type_counts": dict(
            sorted(
                Counter(
                    record["group_type"] for record in unit_evidence_records
                ).items()
            )
        ),
        "semantic_summary_label_counts": dict(
            sorted(
                Counter(
                    record["semantic_evidence"]["summary_label"]
                    for record in unit_evidence_records
                ).items()
            )
        ),
        "recoverability_label_counts": dict(
            sorted(
                Counter(
                    record["recoverability_evidence"]["recoverability_label"]
                    for record in unit_evidence_records
                ).items()
            )
        ),
        "num_misleading_recovery": sum(
            1
            for record in unit_evidence_records
            if record["recoverability_evidence"]["misleading_recovery"]
        ),
    }
