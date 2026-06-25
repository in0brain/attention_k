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
from recover_attention.masked_questions import (
    DEFAULT_MASK_BACKEND,
    DEFAULT_MASK_STRATEGY,
    DEFAULT_MASK_TOKEN,
    apply_mask_to_unit,
    build_masked_question_record,
    build_masked_question_records,
)
from recover_attention.schemas import REQUIRED_FIELDS, validate_masked_question_record


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
FUTURE_STAGE_FIELDS = {
    "recoverable",
    "recovered_question",
    "recoverability_label",
    "confidence",
    "attention_anchor_label",
    "guidance_action",
    "hidden_states_path",
    "attentions_path",
}


def make_span(
    question: str,
    text: str,
    span_type: str,
    span_id: str,
    occurrence: int = 0,
) -> dict:
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


def scores(entailment: float, contradiction: float) -> dict:
    neutral = round(1.0 - entailment - contradiction, 10)
    if neutral < 0:
        raise ValueError("test scores must leave non-negative neutral mass")
    return {
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
    }


def label_from_scores(score_values: dict) -> str:
    return max(
        ("entailment", "neutral", "contradiction"),
        key=lambda label: score_values[label],
    )


def semantic_label_record(
    *,
    question: str = ORIGINAL_QUESTION,
    spans: list[dict] | None = None,
    ablation_type: str = "delete",
    unit_id: str = "unit_001",
    unit_scope: str | None = None,
    group_type: str | None = None,
    necessary: bool = True,
) -> dict:
    resolved_spans = spans or [make_span(question, "3", "number", "span_001")]
    resolved_scope = unit_scope or ("single" if len(resolved_spans) == 1 else "group")
    resolved_group_type = group_type or ("single" if len(resolved_spans) == 1 else "number_set")

    if necessary:
        forward_entailment = 0.65
        backward_entailment = 0.35
        semantic_label = "Information Loss"
    else:
        forward_entailment = 0.80
        backward_entailment = 0.75
        semantic_label = "Equivalent"

    forward_scores = scores(forward_entailment, 0.05)
    backward_scores = scores(backward_entailment, 0.05)
    bidirectional_score = min(
        forward_scores["entailment"],
        backward_scores["entailment"],
    )
    contradiction_score = max(
        forward_scores["contradiction"],
        backward_scores["contradiction"],
    )
    nli_id = f"q1__{unit_id}__{ablation_type}__nli_stub_v0"
    ablated_question = f"{question} [{ablation_type}]"
    return {
        "semantic_label_id": f"{nli_id}__sem_rule_v0",
        "nli_id": nli_id,
        "ablation_id": f"q1__{unit_id}__{ablation_type}",
        "id": "q1",
        "unit_id": unit_id,
        "unit_scope": resolved_scope,
        "group_type": resolved_group_type,
        "span_ids": [span["span_id"] for span in resolved_spans],
        "spans": [dict(span) for span in resolved_spans],
        "ablation_type": ablation_type,
        "original_question": question,
        "ablated_question": ablated_question,
        "nli_backend": "stub_v0",
        "language": "en",
        "language_setting": "auto",
        "forward": {
            "premise": question,
            "hypothesis": ablated_question,
            "label": label_from_scores(forward_scores),
            "scores": forward_scores,
        },
        "backward": {
            "premise": ablated_question,
            "hypothesis": question,
            "label": label_from_scores(backward_scores),
            "scores": backward_scores,
        },
        "bidirectional_entailment_score": bidirectional_score,
        "contradiction_score": contradiction_score,
        "semantic_label_backend": "rule_v0",
        "semantic_necessity_label": semantic_label,
        "semantic_necessity_score": round(
            max(1.0 - bidirectional_score, contradiction_score),
            10,
        ),
        "is_semantically_necessary": necessary,
        "rule_parameters": {
            "equivalent_threshold": 0.70,
            "directional_entailment_threshold": 0.50,
            "contradiction_threshold": 0.50,
        },
        "decision_reason": (
            "forward entails ablated but backward does not entail original"
            if necessary
            else "bidirectional entailment above equivalent threshold"
        ),
    }


def test_apply_mask_to_single_unit() -> None:
    span = make_span(ORIGINAL_QUESTION, "3", "number", "span_001")

    masked = apply_mask_to_unit(ORIGINAL_QUESTION, {"spans": [span]})

    assert masked == "Tom has [MASK] apples and buys 2 more."


def test_group_number_set_replaces_each_span() -> None:
    spans = [
        make_span(ORIGINAL_QUESTION, "3", "number", "span_001"),
        make_span(ORIGINAL_QUESTION, "2", "number", "span_002"),
    ]
    records, stats = build_masked_question_records(
        [
            semantic_label_record(spans=spans, ablation_type="delete", group_type="number_set"),
            semantic_label_record(
                spans=spans,
                ablation_type="generalize",
                group_type="number_set",
            ),
        ]
    )

    assert stats["num_output_masks"] == 1
    assert records[0]["masked_question"] == "Tom has [MASK] apples and buys [MASK] more."
    assert records[0]["masked_question"].count(DEFAULT_MASK_TOKEN) == 2
    assert validate_masked_question_record(records[0]) is None


def test_repeated_surface_group_masks_each_occurrence() -> None:
    question = "Tom sees 3 apples and 3 pears."
    spans = [
        make_span(question, "3", "number", "span_001", occurrence=0),
        make_span(question, "3", "number", "span_002", occurrence=1),
    ]
    records, _stats = build_masked_question_records(
        [
            semantic_label_record(
                question=question,
                spans=spans,
                ablation_type="delete",
                group_type="repeated_surface",
            )
        ]
    )

    assert records[0]["masked_question"] == "Tom sees [MASK] apples and [MASK] pears."


def test_delete_and_generalize_sources_merge_into_one_record() -> None:
    delete_record = semantic_label_record(ablation_type="delete")
    generalize_record = semantic_label_record(ablation_type="generalize")

    records, stats = build_masked_question_records([generalize_record, delete_record])

    assert len(records) == 1
    assert stats["source_count_distribution"] == {2: 1}
    record = records[0]
    assert record["masked_id"] == "q1__unit_001__mask"
    assert record["mask_backend"] == DEFAULT_MASK_BACKEND
    assert record["mask_strategy"] == DEFAULT_MASK_STRATEGY
    assert [source["ablation_type"] for source in record["semantic_sources"]] == [
        "delete",
        "generalize",
    ]
    assert record["source_semantic_label_ids"] == [
        source["semantic_label_id"] for source in record["semantic_sources"]
    ]
    assert record["source_nli_ids"] == [
        source["nli_id"] for source in record["semantic_sources"]
    ]
    assert record["source_ablation_ids"] == [
        source["ablation_id"] for source in record["semantic_sources"]
    ]


def test_build_masked_question_record_validates_output() -> None:
    record = build_masked_question_record([semantic_label_record()])

    assert validate_masked_question_record(record) is None
    assert len(record["spans"]) == 1
    assert DEFAULT_MASK_TOKEN in record["masked_question"]


def test_only_necessary_filters_equivalent_units() -> None:
    records, stats = build_masked_question_records(
        [semantic_label_record(necessary=False)],
        only_necessary=True,
    )

    assert records == []
    assert stats["num_units"] == 1
    assert stats["num_filtered_not_necessary"] == 1
    assert stats["num_output_masks"] == 0


def test_inconsistent_unit_structure_raises_value_error() -> None:
    first_record = semantic_label_record(ablation_type="delete")
    second_record = semantic_label_record(
        spans=[make_span(ORIGINAL_QUESTION, "2", "number", "span_002")],
        ablation_type="generalize",
    )

    with pytest.raises(ValueError, match="inconsistent unit structure"):
        build_masked_question_records([first_record, second_record])


def test_inconsistent_unit_structure_raises_before_only_necessary_filter() -> None:
    first_record = semantic_label_record(ablation_type="delete", necessary=False)
    second_record = semantic_label_record(
        spans=[make_span(ORIGINAL_QUESTION, "2", "number", "span_002")],
        ablation_type="generalize",
        necessary=False,
    )

    with pytest.raises(ValueError, match="inconsistent unit structure"):
        build_masked_question_records(
            [first_record, second_record],
            only_necessary=True,
        )


def test_overlapping_spans_are_skipped() -> None:
    question = "abcdef"
    spans = [
        {"span_id": "span_001", "text": "abc", "type": "object", "start": 0, "end": 3},
        {"span_id": "span_002", "text": "bcd", "type": "object", "start": 1, "end": 4},
    ]

    records, stats = build_masked_question_records(
        [
            semantic_label_record(
                question=question,
                spans=spans,
                unit_scope="group",
                group_type="object_set",
            )
        ]
    )

    assert records == []
    assert stats["num_skipped_overlap"] == 1


def test_custom_mask_token_is_used() -> None:
    records, stats = build_masked_question_records(
        [semantic_label_record()],
        mask_token="___",
    )

    assert records[0]["masked_question"] == "Tom has ___ apples and buys 2 more."
    assert records[0]["mask_token"] == "___"
    assert stats["mask_token"] == "___"


def test_output_has_no_old_or_future_fields() -> None:
    records, _stats = build_masked_question_records([semantic_label_record()])
    record = records[0]

    assert {"span_id", "span_text", "span_type"}.isdisjoint(record.keys())
    assert FUTURE_STAGE_FIELDS.isdisjoint(record.keys())
    assert set(record) == set(REQUIRED_FIELDS["masked_question"])
    assert len(record) == 16


def test_unsupported_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_masked_question_records([semantic_label_record()], backend="manual")


def test_batch_stats_fields_are_complete() -> None:
    records, stats = build_masked_question_records([semantic_label_record()])

    assert len(records) == 1
    assert stats["num_input_labels"] == 1
    assert stats["num_units"] == 1
    assert stats["num_output_masks"] == 1
    assert stats["num_filtered_not_necessary"] == 0
    assert stats["num_skipped_overlap"] == 0
    assert stats["source_count_distribution"] == {1: 1}
    assert stats["unit_scope_counts"] == {"single": 1}
    assert stats["group_type_counts"] == {"single": 1}


def test_cli_smoke_test_builds_masked_questions(tmp_path: Path) -> None:
    input_path = tmp_path / "semantic_labels.jsonl"
    output_path = tmp_path / "masked_questions.jsonl"
    write_jsonl([semantic_label_record()], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "07_build_masked_questions.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mask-token",
            DEFAULT_MASK_TOKEN,
            "--backend",
            DEFAULT_MASK_BACKEND,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built masked questions" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert set(records[0]) == set(REQUIRED_FIELDS["masked_question"])
    assert validate_masked_question_record(records[0]) is None
