from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.ablation_units import (
    build_ablation_unit_records,
    build_ablation_units,
)
from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import validate_ablation_unit_record


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


def candidate_record(question: str, candidates: list[dict], record_id: str = "q1") -> dict:
    return {"id": record_id, "question": question, "candidates": candidates}


def group_types(record: dict) -> set[str]:
    return {unit["group_type"] for unit in record["units"] if unit["unit_scope"] == "group"}


def group_units(record: dict) -> list[dict]:
    return [unit for unit in record["units"] if unit["unit_scope"] == "group"]


def first_group(record: dict, group_type: str) -> dict:
    return next(unit for unit in record["units"] if unit["group_type"] == group_type)


def test_single_units_created_for_each_candidate() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "apples", "object", "span_002"),
            make_span(question, "2", "number", "span_003"),
        ],
    )

    output = build_ablation_units(record)
    singles = [unit for unit in output["units"] if unit["unit_scope"] == "single"]

    assert len(singles) == 3


def test_single_unit_schema_is_complete() -> None:
    question = "Tom has 3 apples."
    record = candidate_record(question, [make_span(question, "3", "number", "span_001")])

    output = build_ablation_units(record)
    unit = output["units"][0]

    assert set(unit) == {"unit_id", "unit_scope", "group_type", "span_ids", "spans", "reason"}
    assert unit["unit_scope"] == "single"
    assert unit["group_type"] == "single"
    assert unit["span_ids"] == ["span_001"]
    assert unit["spans"][0] == make_span(question, "3", "number", "span_001")


def test_number_set_group_created() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    unit = first_group(output, "number_set")
    assert unit["unit_scope"] == "group"
    assert unit["group_type"] == "number_set"
    assert unit["span_ids"] == ["span_001", "span_002"]
    assert [span["span_id"] for span in unit["spans"]] == unit["span_ids"]


def test_entity_set_group_created() -> None:
    question = "Alice compares Bob with Carol."
    record = candidate_record(
        question,
        [
            make_span(question, "Alice", "entity", "span_001"),
            make_span(question, "Bob", "entity", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    unit = first_group(output, "entity_set")
    assert unit["unit_scope"] == "group"
    assert unit["group_type"] == "entity_set"
    assert unit["span_ids"] == ["span_001", "span_002"]


def test_object_set_group_created() -> None:
    question = "Tom has apples and books."
    record = candidate_record(
        question,
        [
            make_span(question, "apples", "object", "span_001"),
            make_span(question, "books", "object", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    unit = first_group(output, "object_set")
    assert unit["unit_scope"] == "group"
    assert unit["group_type"] == "object_set"
    assert unit["span_ids"] == ["span_001", "span_002"]


def test_repeated_surface_group_created() -> None:
    question = "Tom has 3 apples and buys 3 more apples."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "3", "number", "span_002", occurrence=1),
        ],
    )

    output = build_ablation_units(record)

    unit = first_group(output, "repeated_surface")
    assert unit["unit_scope"] == "group"
    assert unit["group_type"] == "repeated_surface"
    assert unit["span_ids"] == ["span_001", "span_002"]


def test_cyber_security_term_set_group_created() -> None:
    question = "remote code execution follows unsafe deserialization."
    record = candidate_record(
        question,
        [
            make_span(question, "remote code execution", "cyber_security_term", "span_001"),
            make_span(question, "unsafe deserialization", "cyber_security_term", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    assert "cyber_security_term_set" in group_types(output)


def test_span_ids_order_matches_spans_order() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    for unit in output["units"]:
        assert unit["span_ids"] == [span["span_id"] for span in unit["spans"]]


def test_single_units_come_before_group_units() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
    )

    output = build_ablation_units(record)
    scopes = [unit["unit_scope"] for unit in output["units"]]

    assert scopes == sorted(scopes, key=lambda scope: 0 if scope == "single" else 1)


def test_question_offsets_match_text_for_every_span() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "apples", "object", "span_002"),
            make_span(question, "2", "number", "span_003"),
        ],
    )

    output = build_ablation_units(record)

    for unit in output["units"]:
        for span in unit["spans"]:
            assert output["question"][span["start"] : span["end"]] == span["text"]


def test_max_group_size_limits_group_span_count() -> None:
    question = "Numbers 1 2 3 4 5 6 7 8 9 10"
    candidates = [
        make_span(question, str(number), "number", f"span_{number:03d}")
        for number in range(1, 11)
    ]
    record = candidate_record(question, candidates)

    output = build_ablation_units(record, max_group_size=8)
    number_set = next(unit for unit in output["units"] if unit["group_type"] == "number_set")

    assert len(number_set["spans"]) <= 8


def test_max_group_units_limits_number_of_group_units() -> None:
    question = "Alice Bob 1 1 apples books remote code execution unsafe deserialization"
    candidates = [
        make_span(question, "Alice", "entity", "span_001"),
        make_span(question, "Bob", "entity", "span_002"),
        make_span(question, "1", "number", "span_003"),
        make_span(question, "1", "number", "span_004", occurrence=1),
        make_span(question, "apples", "object", "span_005"),
        make_span(question, "books", "object", "span_006"),
        make_span(question, "remote code execution", "cyber_security_term", "span_007"),
        make_span(question, "unsafe deserialization", "cyber_security_term", "span_008"),
    ]
    record = candidate_record(question, candidates)

    output = build_ablation_units(record, max_group_units=2)

    assert len(group_units(output)) <= 2


def test_unit_ids_are_contiguous() -> None:
    question = "Tom has 3 apples and buys 2 more."
    record = candidate_record(
        question,
        [
            make_span(question, "3", "number", "span_001"),
            make_span(question, "2", "number", "span_002"),
        ],
    )

    output = build_ablation_units(record)

    assert [unit["unit_id"] for unit in output["units"]] == [
        f"unit_{index:03d}" for index in range(1, len(output["units"]) + 1)
    ]


def test_validate_ablation_unit_record_rejects_missing_unit_field() -> None:
    question = "Tom has 3 apples."
    output = build_ablation_units(
        candidate_record(question, [make_span(question, "3", "number", "span_001")])
    )
    del output["units"][0]["reason"]

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "missing required field" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject missing fields")


def test_validate_ablation_unit_record_rejects_offset_mismatch() -> None:
    question = "Tom has 3 apples."
    output = build_ablation_units(
        candidate_record(question, [make_span(question, "3", "number", "span_001")])
    )
    output["units"][0]["spans"][0]["text"] = "4"

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "offsets" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject offset mismatches")


def test_validate_ablation_unit_record_rejects_single_scope_with_multiple_spans() -> None:
    question = "Tom has 3 apples and buys 2 more."
    output = build_ablation_units(
        candidate_record(
            question,
            [
                make_span(question, "3", "number", "span_001"),
                make_span(question, "2", "number", "span_002"),
            ],
        )
    )
    unit = first_group(output, "number_set")
    unit["unit_scope"] = "single"
    unit["group_type"] = "single"

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "exactly one span" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject invalid single units")


def test_validate_ablation_unit_record_rejects_group_scope_with_one_span() -> None:
    question = "Tom has 3 apples."
    output = build_ablation_units(
        candidate_record(question, [make_span(question, "3", "number", "span_001")])
    )
    output["units"][0]["unit_scope"] = "group"
    output["units"][0]["group_type"] = "number_set"

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "at least two spans" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject invalid group units")


def test_validate_ablation_unit_record_rejects_single_group_type_with_group_scope() -> None:
    question = "Tom has 3 apples and buys 2 more."
    output = build_ablation_units(
        candidate_record(
            question,
            [
                make_span(question, "3", "number", "span_001"),
                make_span(question, "2", "number", "span_002"),
            ],
        )
    )
    unit = first_group(output, "number_set")
    unit["group_type"] = "single"

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "group_type 'single'" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject mismatched group_type")


def test_validate_ablation_unit_record_rejects_non_single_group_type_with_single_scope() -> None:
    question = "Tom has 3 apples."
    output = build_ablation_units(
        candidate_record(question, [make_span(question, "3", "number", "span_001")])
    )
    output["units"][0]["group_type"] = "number_set"

    try:
        validate_ablation_unit_record(output)
    except ValueError as exc:
        assert "non-single group_type" in str(exc)
    else:
        raise AssertionError("validate_ablation_unit_record should reject mismatched unit_scope")


def test_build_ablation_unit_records_outputs_schema_valid_records() -> None:
    question = "Tom has 3 apples and buys 2 more."
    records = [
        candidate_record(
            question,
            [
                make_span(question, "3", "number", "span_001"),
                make_span(question, "2", "number", "span_002"),
            ],
        )
    ]

    outputs = build_ablation_unit_records(records)

    assert len(outputs) == 1
    assert validate_ablation_unit_record(outputs[0]) is None


def test_no_candidates_outputs_empty_units_and_valid_schema() -> None:
    record = {"id": "empty_001", "question": "Empty example?", "candidates": []}

    output = build_ablation_units(record)

    assert output == {"id": "empty_001", "question": "Empty example?", "units": []}
    assert validate_ablation_unit_record(output) is None


def test_cli_smoke_test_builds_jsonl_output(tmp_path: Path) -> None:
    question = "Tom has 3 apples."
    input_path = tmp_path / "candidate_spans.jsonl"
    output_path = tmp_path / "ablation_units.jsonl"
    write_jsonl(
        [
            candidate_record(
                question,
                [
                    make_span(question, "3", "number", "span_001"),
                    make_span(question, "apples", "object", "span_002"),
                ],
            )
        ],
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "03_build_ablation_units.py"),
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
    assert "[OK] Built ablation units" in result.stdout
    assert output_path.exists()
    assert records
    assert any(unit["unit_scope"] == "single" for unit in records[0]["units"])
