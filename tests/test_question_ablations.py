from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.question_ablations import (
    apply_ablation_to_unit,
    build_ablated_question_records,
)
from recover_attention.schemas import validate_ablated_question_record


def make_span(question: str, text: str, span_type: str, span_id: str, occurrence: int = 0) -> dict:
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = question.index(text, search_from)
        search_from = start + len(text)
    return {
        "span_id": span_id,
        "text": text,
        "type": span_type,
        "start": start,
        "end": start + len(text),
    }


def unit(
    question: str,
    spans: list[dict],
    unit_id: str = "unit_001",
    unit_scope: str | None = None,
    group_type: str | None = None,
) -> dict:
    resolved_scope = unit_scope or ("single" if len(spans) == 1 else "group")
    resolved_group_type = group_type or ("single" if len(spans) == 1 else "number_set")
    return {
        "unit_id": unit_id,
        "unit_scope": resolved_scope,
        "group_type": resolved_group_type,
        "span_ids": [span["span_id"] for span in spans],
        "spans": spans,
        "reason": "test unit",
    }


def ablation_record(question: str, units: list[dict]) -> dict:
    return {"id": "q1", "question": question, "units": units}


def test_delete_single_unit_removes_span() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    output = apply_ablation_to_unit(question, target, "delete")

    assert "3" not in output
    assert output == "Tom has apples."


def test_delete_group_unit_removes_all_spans() -> None:
    question = "Tom has 3 apples and buys 2 more."
    target = unit(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
        group_type="number_set",
    )

    output = apply_ablation_to_unit(question, target, "delete")

    assert "3" not in output
    assert "2" not in output


def test_delete_only_target_occurrence() -> None:
    question = "Tom has 3 apples and buys 3 more apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    output = apply_ablation_to_unit(question, target, "delete")

    assert output.count("3") == 1
    assert "buys 3 more" in output


def test_generalize_number_english() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    output = apply_ablation_to_unit(question, target, "generalize", language="en")

    assert "some number" in output


def test_generalize_object_english() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "apples", "object", "span_001")])

    output = apply_ablation_to_unit(question, target, "generalize", language="en")

    assert "some object" in output


def test_generalize_number_chinese() -> None:
    question = "小明有3个苹果。"
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    output = apply_ablation_to_unit(question, target, "generalize", language="zh")

    assert "某个数量" in output


def test_generalize_object_chinese() -> None:
    question = "小明有3个苹果。"
    target = unit(question, [make_span(question, "苹果", "object", "span_001")])

    output = apply_ablation_to_unit(question, target, "generalize", language="zh")

    assert "某个对象" in output


def test_group_generalize_replaces_each_span() -> None:
    question = "Tom has 3 apples and buys 2 more."
    target = unit(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
        group_type="number_set",
    )

    output = apply_ablation_to_unit(question, target, "generalize", language="en")

    assert output.count("some number") == 2


def test_unsupported_ablation_type_raises_value_error() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    with pytest.raises(ValueError, match="Unsupported ablation_type"):
        apply_ablation_to_unit(question, target, "mask")


def test_overlapping_spans_skipped() -> None:
    question = "abcdef"
    target = unit(
        question,
        [
            {"span_id": "span_001", "text": "abc", "type": "object", "start": 0, "end": 3},
            {"span_id": "span_002", "text": "bcd", "type": "object", "start": 1, "end": 4},
        ],
        group_type="object_set",
    )

    records, stats = build_ablated_question_records([ablation_record(question, [target])])

    assert records == []
    assert stats["num_skipped_overlap"] == 2


def test_empty_ablated_question_skipped() -> None:
    question = "3"
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    records, stats = build_ablated_question_records(
        [ablation_record(question, [target])],
        ablation_types=["delete"],
    )

    assert records == []
    assert stats["num_skipped_empty"] == 1


def test_unchanged_ablated_question_skipped() -> None:
    question = "some number"
    target = unit(question, [make_span(question, "some number", "number", "span_001")])

    records, stats = build_ablated_question_records(
        [ablation_record(question, [target])],
        ablation_types=["generalize"],
        language="en",
    )

    assert records == []
    assert stats["num_skipped_unchanged"] == 1


def test_schema_validation_for_built_records() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    records, stats = build_ablated_question_records([ablation_record(question, [target])])

    assert stats["num_output_ablations"] == 2
    assert {record["ablation_type"] for record in records} == {"delete", "generalize"}
    for record in records:
        assert validate_ablated_question_record(record) is None


def test_output_record_preserves_unit_metadata() -> None:
    question = "Tom has 3 apples and buys 2 more."
    target = unit(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
        group_type="number_set",
    )

    records, _stats = build_ablated_question_records(
        [ablation_record(question, [target])],
        ablation_types=["delete"],
    )

    record = records[0]
    assert record["ablation_id"] == "q1__unit_001__delete"
    assert record["unit_id"] == "unit_001"
    assert record["unit_scope"] == "group"
    assert record["group_type"] == "number_set"
    assert record["span_ids"] == ["span_001", "span_002"]
    assert [span["span_id"] for span in record["spans"]] == record["span_ids"]


def test_output_record_has_no_top_level_old_span_fields() -> None:
    question = "Tom has 3 apples."
    target = unit(question, [make_span(question, "3", "number", "span_001")])

    records, _stats = build_ablated_question_records(
        [ablation_record(question, [target])],
        ablation_types=["generalize"],
    )

    record = records[0]
    assert "span_id" not in record
    assert "span_text" not in record
    assert "span_type" not in record
    assert record["span_ids"] == ["span_001"]
    assert record["spans"][0]["span_id"] == "span_001"


def test_cli_smoke_test_builds_ablated_questions(tmp_path: Path) -> None:
    question = "Tom has 3 apples."
    input_path = tmp_path / "ablation_units.jsonl"
    output_path = tmp_path / "ablated_questions.jsonl"
    write_jsonl(
        [
            ablation_record(
                question,
                [unit(question, [make_span(question, "3", "number", "span_001")])],
            )
        ],
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "04_build_ablated_questions.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built ablated questions" in result.stdout
    assert output_path.exists()
    assert {record["ablation_type"] for record in records} == {"delete", "generalize"}
