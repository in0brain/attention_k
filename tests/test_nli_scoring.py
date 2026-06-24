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
from recover_attention.nli_scoring import (
    detect_language,
    score_ablated_question_record,
    score_ablated_question_records,
    score_nli_pair_stub,
)
from recover_attention.schemas import validate_nli_score_record


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


def ablated_question_record(
    ablation_type: str = "generalize",
    unit_scope: str = "single",
    group_type: str = "single",
    spans: list[dict] | None = None,
    original_question: str = "Tom has 3 apples and buys 2 more.",
    ablated_question: str | None = None,
) -> dict:
    resolved_spans = spans or [make_span(original_question, "3", "number", "span_001")]
    resolved_ablated_question = ablated_question
    if resolved_ablated_question is None:
        if ablation_type == "delete":
            resolved_ablated_question = "Tom has apples and buys 2 more."
        else:
            resolved_ablated_question = "Tom has some number apples and buys 2 more."
    return {
        "ablation_id": f"q1__unit_001__{ablation_type}",
        "id": "q1",
        "unit_id": "unit_001",
        "unit_scope": unit_scope,
        "group_type": group_type,
        "span_ids": [span["span_id"] for span in resolved_spans],
        "spans": resolved_spans,
        "ablation_type": ablation_type,
        "original_question": original_question,
        "ablated_question": resolved_ablated_question,
    }


def test_generalize_forward_backward_scores() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert record["forward"]["label"] == "entailment"
    assert record["backward"]["label"] == "neutral"
    assert record["forward"]["scores"]["entailment"] > record["backward"]["scores"]["entailment"]


def test_delete_forward_backward_scores() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert record["forward"]["scores"]["entailment"] > record["backward"]["scores"]["entailment"]
    assert record["backward"]["label"] == "neutral"


def test_group_penalty_reduces_entailment_scores() -> None:
    question = "Tom has 3 apples and buys 2 more."
    single = score_ablated_question_record(ablated_question_record("generalize"))
    group = score_ablated_question_record(
        ablated_question_record(
            "generalize",
            unit_scope="group",
            group_type="number_set",
            spans=[
                make_span(question, "3", "number", "span_001"),
                make_span(question, "2", "number", "span_002"),
            ],
            original_question=question,
            ablated_question="Tom has some number apples and buys some number more.",
        )
    )

    assert group["forward"]["scores"]["entailment"] < single["forward"]["scores"]["entailment"]
    assert group["backward"]["scores"]["entailment"] < single["backward"]["scores"]["entailment"]


def test_bidirectional_entailment_score() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert record["bidirectional_entailment_score"] == min(
        record["forward"]["scores"]["entailment"],
        record["backward"]["scores"]["entailment"],
    )


def test_contradiction_score() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert record["contradiction_score"] == max(
        record["forward"]["scores"]["contradiction"],
        record["backward"]["scores"]["contradiction"],
    )


def test_language_auto_english() -> None:
    record = score_ablated_question_record(ablated_question_record(), language="auto")

    assert record["language"] == "en"
    assert record["language_setting"] == "auto"


def test_language_auto_chinese() -> None:
    question = "小明有3个苹果。"
    record = score_ablated_question_record(
        ablated_question_record(
            spans=[make_span(question, "3", "number", "span_001")],
            original_question=question,
            ablated_question="小明有某个数量个苹果。",
        ),
        language="auto",
    )

    assert record["language"] == "zh"
    assert record["language_setting"] == "auto"


def test_language_forced_zh() -> None:
    record = score_ablated_question_record(ablated_question_record(), language="zh")

    assert record["language"] == "zh"
    assert record["language_setting"] == "zh"


def test_label_fixed_english_enum_for_chinese_input() -> None:
    question = "小明有3个苹果。"
    record = score_ablated_question_record(
        ablated_question_record(
            spans=[make_span(question, "3", "number", "span_001")],
            original_question=question,
            ablated_question="小明有某个数量个苹果。",
        )
    )

    assert record["forward"]["label"] in {"entailment", "neutral", "contradiction"}
    assert record["backward"]["label"] in {"entailment", "neutral", "contradiction"}
    assert "蕴含" not in {record["forward"]["label"], record["backward"]["label"]}


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backend: hf_xnli"):
        score_ablated_question_record(ablated_question_record(), backend="hf_xnli")


def test_unsupported_language_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported language: jp"):
        score_ablated_question_record(ablated_question_record(), language="jp")


def test_score_sum_equals_one() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert abs(sum(record["forward"]["scores"].values()) - 1.0) < 1e-6
    assert abs(sum(record["backward"]["scores"].values()) - 1.0) < 1e-6


def test_schema_validation_for_scored_record() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert validate_nli_score_record(record) is None


def test_scored_record_has_no_semantic_label_or_old_directional_fields() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert "semantic_necessity_label" not in record
    assert "original_to_ablated" not in record
    assert "ablated_to_original" not in record
    assert "forward" in record
    assert "backward" in record


def test_score_ablated_question_records_stats() -> None:
    records, stats = score_ablated_question_records(
        [ablated_question_record("delete"), ablated_question_record("generalize")]
    )

    assert len(records) == 2
    assert stats["num_input_ablations"] == 2
    assert stats["num_output_scores"] == 2
    assert stats["backend"] == "stub_v0"
    assert stats["language_counts"] == {"en": 2}
    assert stats["ablation_type_counts"] == {"delete": 1, "generalize": 1}


def test_score_nli_pair_stub_rejects_unsupported_direction() -> None:
    with pytest.raises(ValueError, match="Unsupported direction"):
        score_nli_pair_stub("a", "b", "delete", 1, "sideways")


def test_detect_language_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="text must be a str"):
        detect_language(123)  # type: ignore[arg-type]


def test_cli_smoke_test_builds_nli_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "ablated_questions.jsonl"
    output_path = tmp_path / "nli_scores.jsonl"
    write_jsonl([ablated_question_record("generalize")], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_run_nli_scoring.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "stub_v0",
            "--language",
            "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built NLI scores" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["nli_backend"] == "stub_v0"
    assert "forward" in records[0]
    assert "backward" in records[0]
    assert "bidirectional_entailment_score" in records[0]
    assert "contradiction_score" in records[0]
