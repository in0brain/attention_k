from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.candidate_extraction import (
    build_candidate_span_records,
    extract_candidate_spans,
)
from recover_attention.schemas import validate_candidate_span_record


def candidate_pairs(candidates: list[dict]) -> set[tuple[str, str]]:
    return {(candidate["text"], candidate["type"]) for candidate in candidates}


def assert_offsets_match(question: str, candidates: list[dict]) -> None:
    for candidate in candidates:
        assert question[candidate["start"] : candidate["end"]] == candidate["text"]


def test_extract_candidate_spans_english_basic_sample() -> None:
    question = "Tom has 3 apples and buys 2 more. How many apples does he have now?"

    candidates = extract_candidate_spans(question, language="en")
    pairs = candidate_pairs(candidates)

    assert ("3", "number") in pairs
    assert ("2", "number") in pairs
    assert ("apples", "object") in pairs
    assert ("buys", "operation") in pairs
    assert ("How many", "question_target") in pairs
    assert ("more", "comparison") in pairs


def test_extract_candidate_spans_chinese_basic_sample() -> None:
    question = "小明有3个苹果，又买了2个。现在有多少个苹果？"

    candidates = extract_candidate_spans(question, language="zh")
    pairs = candidate_pairs(candidates)

    assert ("3", "number") in pairs
    assert ("2", "number") in pairs
    assert ("苹果", "object") in pairs
    assert ("买", "operation") in pairs
    assert ("多少", "question_target") in pairs


def test_chinese_offsets_are_python_string_offsets() -> None:
    question = "小明有3个苹果，又买了2个。现在有多少个苹果？"

    candidates = extract_candidate_spans(question, language="zh")

    assert_offsets_match(question, candidates)


def test_english_offsets_are_python_string_offsets() -> None:
    question = "Tom has 3 apples and buys 2 more. How many apples does he have now?"

    candidates = extract_candidate_spans(question, language="en")

    assert_offsets_match(question, candidates)


def test_extract_candidate_spans_english_cyber_security_terms() -> None:
    question = "The vulnerability allows remote code execution through unsafe deserialization."

    candidates = extract_candidate_spans(question, language="en")
    pairs = candidate_pairs(candidates)

    assert ("remote code execution", "cyber_security_term") in pairs
    assert ("unsafe deserialization", "cyber_security_term") in pairs


def test_extract_candidate_spans_chinese_cyber_security_terms() -> None:
    question = "该漏洞允许通过不安全反序列化实现远程代码执行。"

    candidates = extract_candidate_spans(question, language="zh")
    pairs = candidate_pairs(candidates)

    assert ("不安全反序列化", "cyber_security_term") in pairs
    assert ("远程代码执行", "cyber_security_term") in pairs


def test_max_candidates_truncates_without_minimum_requirement() -> None:
    question = (
        "Tom has 1 apple, buys 2 books, gives 3 files, removes 4 packets, "
        "and downloads 5 records. How many more items remain?"
    )

    candidates = extract_candidate_spans(question, language="en", max_candidates=5)

    assert len(candidates) <= 5


def test_few_candidates_are_allowed() -> None:
    question = "What now?"

    candidates = extract_candidate_spans(question, language="en")

    assert len(candidates) < 8
    assert ("What", "question_target") in candidate_pairs(candidates)


def test_span_ids_are_contiguous_after_sorting_and_truncation() -> None:
    question = "Tom has 3 apples and buys 2 more. How many apples does he have now?"

    candidates = extract_candidate_spans(question, language="en", max_candidates=6)

    assert [candidate["span_id"] for candidate in candidates] == [
        f"span_{index:03d}" for index in range(1, len(candidates) + 1)
    ]


def test_build_candidate_span_records_outputs_schema_valid_records() -> None:
    question_records = [
        {
            "id": "gsm8k_0001",
            "dataset": "gsm8k",
            "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
            "gold_answer": "5",
        },
        {
            "id": "zh_0001",
            "dataset": "toy_zh",
            "question": "小明有3个苹果，又买了2个。现在有多少个苹果？",
            "gold_answer": "5",
        },
    ]

    records = build_candidate_span_records(question_records, language="auto")

    assert len(records) == 2
    for record in records:
        assert validate_candidate_span_record(record) is None
        assert_offsets_match(record["question"], record["candidates"])


def test_unsupported_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        extract_candidate_spans("What now?", backend="model")
