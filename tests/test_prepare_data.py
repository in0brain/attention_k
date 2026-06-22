from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.prepare_data import normalize_question_record, prepare_questions


def raw_question_record(**overrides: object) -> dict:
    record = {
        "id": " gsm8k_0001 ",
        "dataset": " gsm8k ",
        "question": " Tom has 3 apples. ",
        "gold_answer": " 3 ",
        "extra": "remove me",
    }
    record.update(overrides)
    return record


def test_normalize_question_record_returns_standard_fields_only() -> None:
    normalized = normalize_question_record(raw_question_record())

    assert list(normalized.keys()) == ["id", "dataset", "question", "gold_answer"]


def test_normalize_question_record_strips_string_fields() -> None:
    normalized = normalize_question_record(raw_question_record())

    assert normalized == {
        "id": "gsm8k_0001",
        "dataset": "gsm8k",
        "question": "Tom has 3 apples.",
        "gold_answer": "3",
    }


def test_normalize_question_record_missing_field_raises_value_error() -> None:
    record = raw_question_record()
    del record["gold_answer"]

    with pytest.raises(ValueError, match="missing required field"):
        normalize_question_record(record)


def test_normalize_question_record_empty_question_raises_value_error() -> None:
    record = raw_question_record(question="   ")

    with pytest.raises(ValueError, match="non-empty"):
        normalize_question_record(record)


def test_prepare_questions_reads_input_and_writes_output_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "processed" / "questions.jsonl"
    records = [
        raw_question_record(id="q1", question=" First question? "),
        raw_question_record(id="q2", question=" Second question? "),
    ]
    write_jsonl(records, input_path)

    prepared = prepare_questions(input_path, output_path)

    assert output_path.exists()
    assert read_jsonl(output_path) == prepared
    assert len(prepared) == 2


def test_prepare_questions_preserves_input_order(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "questions.jsonl"
    write_jsonl(
        [
            raw_question_record(id="q1"),
            raw_question_record(id="q2"),
            raw_question_record(id="q3"),
        ],
        input_path,
    )

    prepared = prepare_questions(input_path, output_path)

    assert [record["id"] for record in prepared] == ["q1", "q2", "q3"]


def test_prepare_questions_duplicate_id_raises_value_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "questions.jsonl"
    write_jsonl([raw_question_record(id="q1"), raw_question_record(id="q1")], input_path)

    with pytest.raises(ValueError, match="duplicate question id"):
        prepare_questions(input_path, output_path)


def test_prepare_questions_does_not_keep_extra_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "questions.jsonl"
    write_jsonl([raw_question_record(extra="unused")], input_path)

    prepared = prepare_questions(input_path, output_path)

    assert "extra" not in prepared[0]
    assert "extra" not in read_jsonl(output_path)[0]
