"""Prepare normalized question records for the v0/v1 pipeline."""

from __future__ import annotations

from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import validate_question_record


QUESTION_FIELDS = ("id", "dataset", "question", "gold_answer")


def normalize_question_record(record: dict) -> dict:
    """Normalize a raw question record to the standard four-field schema."""
    if not isinstance(record, dict):
        raise ValueError("question record must be a dict")

    missing = [field for field in QUESTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"question record missing required field(s): {', '.join(missing)}")

    normalized = {}
    for field in QUESTION_FIELDS:
        value = record[field]
        normalized[field] = value.strip() if isinstance(value, str) else value

    validate_question_record(normalized)
    return normalized


def prepare_questions(input_path: str | Path, output_path: str | Path) -> list[dict]:
    """Read, normalize, validate, and write question records."""
    records = read_jsonl(input_path)
    normalized_records: list[dict] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        normalized = normalize_question_record(record)
        record_id = normalized["id"]
        if record_id in seen_ids:
            raise ValueError(f"duplicate question id at record {index}: {record_id}")
        seen_ids.add(record_id)
        normalized_records.append(normalized)

    write_jsonl(normalized_records, output_path)
    return normalized_records
