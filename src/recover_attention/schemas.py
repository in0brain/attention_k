"""Lightweight schema validation helpers for JSONL records."""

from __future__ import annotations

from numbers import Real
from typing import Any


ALLOWED_SPAN_TYPES = {
    "number",
    "entity",
    "operation",
    "relation",
    "question_target",
    "comparison",
    "negation",
    "condition",
    "cyber_security_term",
}

ALLOWED_ABLATION_TYPES = {"delete", "generalize", "mask"}
ALLOWED_NLI_LABELS = {"entailment", "neutral", "contradiction"}
ALLOWED_SEMANTIC_NECESSITY_LABELS = {
    "Equivalent",
    "Information Loss",
    "Added Assumption",
    "Non-equivalent",
}
ALLOWED_RECOVERABLE_VALUES = {"yes", "no", "uncertain"}
ALLOWED_RECOVERABILITY_LABELS = {
    "Recoverable",
    "Partially Recoverable",
    "Non-recoverable",
    "Misleading Recovery",
}


def validate_question_record(record: dict) -> None:
    name = "question record"
    _require_dict(record, name)
    _require_fields(record, ["id", "dataset", "question", "gold_answer"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "dataset", name)
    _require_non_empty_str(record, "question", name)
    _require_non_empty_str(record, "gold_answer", name)


def validate_candidate_span_record(record: dict) -> None:
    name = "candidate span record"
    _require_dict(record, name)
    _require_fields(record, ["id", "question", "candidates"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "question", name)

    candidates = record["candidates"]
    if not isinstance(candidates, list):
        raise ValueError(f"{name} field 'candidates' must be a list")

    for index, candidate in enumerate(candidates):
        candidate_name = f"{name} candidate[{index}]"
        _require_dict(candidate, candidate_name)
        _require_fields(candidate, ["span_id", "text", "type", "start", "end"], candidate_name)
        _require_non_empty_str(candidate, "span_id", candidate_name)
        _require_non_empty_str(candidate, "text", candidate_name)
        _require_enum(candidate, "type", ALLOWED_SPAN_TYPES, candidate_name)
        _require_int(candidate, "start", candidate_name, min_value=0)
        _require_int(candidate, "end", candidate_name)
        if candidate["end"] <= candidate["start"]:
            raise ValueError(f"{candidate_name} field 'end' must be greater than 'start'")


def validate_ablated_question_record(record: dict) -> None:
    name = "ablated question record"
    _require_dict(record, name)
    _require_fields(
        record,
        [
            "id",
            "span_id",
            "span_text",
            "span_type",
            "ablation_type",
            "original_question",
            "ablated_question",
        ],
        name,
    )
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_non_empty_str(record, "span_text", name)
    _require_enum(record, "span_type", ALLOWED_SPAN_TYPES, name)
    _require_enum(record, "ablation_type", ALLOWED_ABLATION_TYPES, name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "ablated_question", name)


def validate_nli_score_record(record: dict) -> None:
    name = "nli score record"
    _require_dict(record, name)
    _require_fields(
        record,
        [
            "id",
            "span_id",
            "span_text",
            "span_type",
            "ablation_type",
            "original_to_ablated",
            "ablated_to_original",
            "semantic_necessity_label",
        ],
        name,
    )
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_non_empty_str(record, "span_text", name)
    _require_enum(record, "span_type", ALLOWED_SPAN_TYPES, name)
    _require_enum(record, "ablation_type", ALLOWED_ABLATION_TYPES, name)
    _require_enum(record, "original_to_ablated", ALLOWED_NLI_LABELS, name)
    _require_enum(record, "ablated_to_original", ALLOWED_NLI_LABELS, name)
    _require_enum(
        record,
        "semantic_necessity_label",
        ALLOWED_SEMANTIC_NECESSITY_LABELS,
        name,
    )


def validate_masked_question_record(record: dict) -> None:
    name = "masked question record"
    _require_dict(record, name)
    _require_fields(
        record,
        [
            "id",
            "span_id",
            "span_text",
            "span_type",
            "original_question",
            "masked_question",
        ],
        name,
    )
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_non_empty_str(record, "span_text", name)
    _require_enum(record, "span_type", ALLOWED_SPAN_TYPES, name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "masked_question", name)


def validate_recover_output_record(record: dict) -> None:
    name = "recover output record"
    _require_dict(record, name)
    _require_fields(
        record,
        [
            "id",
            "span_id",
            "sample_id",
            "masked_question",
            "recovered_question",
            "recoverable",
            "confidence",
            "reason",
        ],
        name,
    )
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_int(record, "sample_id", name, min_value=0)
    _require_non_empty_str(record, "masked_question", name)
    _require_str(record, "recovered_question", name)
    _require_enum(record, "recoverable", ALLOWED_RECOVERABLE_VALUES, name)
    _require_number(record, "confidence", name, min_value=0, max_value=1)
    _require_str(record, "reason", name)


def validate_recover_score_record(record: dict) -> None:
    name = "recover score record"
    _require_dict(record, name)
    _require_fields(
        record,
        [
            "id",
            "span_id",
            "recoverability_label",
            "confidence_mean",
            "recovery_consistency",
            "misleading_recovery",
        ],
        name,
    )
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_enum(record, "recoverability_label", ALLOWED_RECOVERABILITY_LABELS, name)
    _require_number(record, "confidence_mean", name, min_value=0, max_value=1)
    _require_number(record, "recovery_consistency", name, min_value=0, max_value=1)
    _require_bool(record, "misleading_recovery", name)


def _require_dict(record: Any, name: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a dict")


def _require_fields(record: dict, fields: list[str], name: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")


def _require_str(record: dict, field: str, name: str) -> None:
    if not isinstance(record[field], str):
        raise ValueError(f"{name} field '{field}' must be a str")


def _require_non_empty_str(record: dict, field: str, name: str) -> None:
    _require_str(record, field, name)
    if not record[field].strip():
        raise ValueError(f"{name} field '{field}' must be a non-empty str")


def _require_enum(record: dict, field: str, allowed: set[str], name: str) -> None:
    _require_str(record, field, name)
    if record[field] not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"{name} field '{field}' has invalid value {record[field]!r}; "
            f"allowed values: {allowed_values}"
        )


def _require_int(
    record: dict,
    field: str,
    name: str,
    min_value: int | None = None,
) -> None:
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} field '{field}' must be an int")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} field '{field}' must be >= {min_value}")


def _require_number(
    record: dict,
    field: str,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    value = record[field]
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} field '{field}' must be an int or float")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} field '{field}' must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} field '{field}' must be <= {max_value}")


def _require_bool(record: dict, field: str, name: str) -> None:
    if not isinstance(record[field], bool):
        raise ValueError(f"{name} field '{field}' must be a bool")
