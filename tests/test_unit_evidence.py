from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import FORBIDDEN_FIELDS, validate_unit_evidence_record
from recover_attention.unit_evidence import (
    DEFAULT_UNIT_EVIDENCE_BACKEND,
    build_semantic_evidence_summary,
    build_unit_evidence_file,
    build_unit_evidence_record,
    build_unit_evidence_records,
)


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
ABLATED_QUESTION = "Tom has some number apples and buys 2 more."


def span_001() -> dict:
    return {
        "span_id": "span_001",
        "text": "3",
        "type": "number",
        "start": 8,
        "end": 9,
    }


def span_002() -> dict:
    return {
        "span_id": "span_002",
        "text": "2",
        "type": "number",
        "start": 26,
        "end": 27,
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
    question_id: str = "q1",
    unit_id: str = "unit_001",
    unit_scope: str = "single",
    group_type: str = "single",
    spans: list[dict] | None = None,
    ablation_type: str = "generalize",
    forward_entailment: float = 0.65,
    backward_entailment: float = 0.35,
    forward_contradiction: float = 0.05,
    backward_contradiction: float = 0.05,
    original_question: str = ORIGINAL_QUESTION,
) -> dict:
    resolved_spans = [span_001()] if spans is None else deepcopy(spans)
    span_ids = [span["span_id"] for span in resolved_spans]
    forward_scores = scores(forward_entailment, forward_contradiction)
    backward_scores = scores(backward_entailment, backward_contradiction)
    bidirectional_entailment_score = min(
        forward_scores["entailment"],
        backward_scores["entailment"],
    )
    contradiction_score = max(
        forward_scores["contradiction"],
        backward_scores["contradiction"],
    )

    if contradiction_score >= 0.50:
        semantic_necessity_label = "Non-equivalent"
        decision_reason = "contradiction above threshold"
    elif bidirectional_entailment_score >= 0.70:
        semantic_necessity_label = "Equivalent"
        decision_reason = "bidirectional entailment above equivalent threshold"
    elif forward_scores["entailment"] >= 0.50 and backward_scores["entailment"] < 0.50:
        semantic_necessity_label = "Information Loss"
        decision_reason = "forward entails ablated but backward does not entail original"
    elif forward_scores["entailment"] < 0.50 and backward_scores["entailment"] >= 0.50:
        semantic_necessity_label = "Added Assumption"
        decision_reason = "backward entails original but forward does not entail ablated"
    else:
        semantic_necessity_label = "Non-equivalent"
        decision_reason = "low bidirectional entailment"

    nli_id = f"{question_id}__{unit_id}__{ablation_type}__nli_stub_v0"
    return {
        "semantic_label_id": f"{nli_id}__sem_rule_v0",
        "nli_id": nli_id,
        "ablation_id": f"{question_id}__{unit_id}__{ablation_type}",
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": unit_scope,
        "group_type": group_type,
        "span_ids": span_ids,
        "spans": resolved_spans,
        "ablation_type": ablation_type,
        "original_question": original_question,
        "ablated_question": ABLATED_QUESTION,
        "nli_backend": "stub_v0",
        "language": "en",
        "language_setting": "auto",
        "forward": {
            "premise": original_question,
            "hypothesis": ABLATED_QUESTION,
            "label": label_from_scores(forward_scores),
            "scores": forward_scores,
        },
        "backward": {
            "premise": ABLATED_QUESTION,
            "hypothesis": original_question,
            "label": label_from_scores(backward_scores),
            "scores": backward_scores,
        },
        "bidirectional_entailment_score": bidirectional_entailment_score,
        "contradiction_score": contradiction_score,
        "semantic_label_backend": "rule_v0",
        "semantic_necessity_label": semantic_necessity_label,
        "semantic_necessity_score": round(
            max(1.0 - bidirectional_entailment_score, contradiction_score),
            10,
        ),
        "is_semantically_necessary": semantic_necessity_label != "Equivalent",
        "rule_parameters": {
            "equivalent_threshold": 0.70,
            "directional_entailment_threshold": 0.50,
            "contradiction_threshold": 0.50,
        },
        "decision_reason": decision_reason,
    }


def equivalent_semantic_label_record(**kwargs: object) -> dict:
    return semantic_label_record(
        forward_entailment=0.80,
        backward_entailment=0.75,
        **kwargs,
    )


def recover_score_record(
    *,
    question_id: str = "q1",
    unit_id: str = "unit_001",
    unit_scope: str = "single",
    group_type: str = "single",
    spans: list[dict] | None = None,
    original_question: str = ORIGINAL_QUESTION,
    recoverability_label: str = "Recoverable",
    recoverability_score: float = 1.0,
    misleading_recovery: bool = False,
) -> dict:
    resolved_spans = [span_001()] if spans is None else deepcopy(spans)
    span_ids = [span["span_id"] for span in resolved_spans]
    masked_id = f"{question_id}__{unit_id}__mask"
    masked_question = (
        "Tom has [MASK] apples and buys 2 more."
        if len(resolved_spans) == 1
        else "Tom has [MASK] apples and buys [MASK] more."
    )
    return {
        "recover_score_id": f"{masked_id}__score_stub_rule_v0",
        "masked_id": masked_id,
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": unit_scope,
        "group_type": group_type,
        "span_ids": span_ids,
        "spans": resolved_spans,
        "original_question": original_question,
        "masked_question": masked_question,
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovery_backend": "oracle_stub_v0",
        "num_samples": 1,
        "source_sample_ids": [0],
        "recovered_questions": [original_question],
        "recoverability_label": recoverability_label,
        "recoverability_score": recoverability_score,
        "confidence_mean": recoverability_score,
        "recovery_consistency": 1.0,
        "misleading_recovery": misleading_recovery,
        "score_backend": "stub_rule_v0",
        "evidence": {"matched_original_question": recoverability_score == 1.0},
    }


def test_single_unit_builds_valid_unit_evidence_record() -> None:
    record = build_unit_evidence_record(
        [semantic_label_record()],
        recover_score_record(),
    )

    assert record["unit_evidence_id"] == "q1__unit_001__evidence_aggregate_stub_v0"
    assert record["available_signal_types"] == [
        "semantic_necessity",
        "semantic_recoverability",
    ]
    assert record["missing_signal_types"] == [
        "trajectory_stability",
        "answer_stability",
        "raw_attention_pattern",
        "attention_steering_effect",
    ]
    assert record["evidence_status"] == "partial_stub_evidence"
    assert validate_unit_evidence_record(record) is None


def test_multiple_semantic_labels_for_same_unit_aggregate_to_one_record() -> None:
    records, stats = build_unit_evidence_records(
        [
            semantic_label_record(ablation_type="generalize"),
            semantic_label_record(ablation_type="delete"),
        ],
        [recover_score_record()],
    )

    assert len(records) == 1
    assert records[0]["semantic_evidence"]["num_semantic_records"] == 2
    assert stats["num_semantic_labels"] == 2
    assert stats["num_output_unit_evidence"] == 1


def test_multiple_units_output_multiple_unit_evidence_records() -> None:
    semantic_records = [
        semantic_label_record(question_id="q1", unit_id="unit_001"),
        semantic_label_record(question_id="q2", unit_id="unit_002"),
    ]
    recover_records = [
        recover_score_record(question_id="q1", unit_id="unit_001"),
        recover_score_record(question_id="q2", unit_id="unit_002"),
    ]

    records, stats = build_unit_evidence_records(semantic_records, recover_records)

    assert len(records) == 2
    assert {record["unit_evidence_id"] for record in records} == {
        "q1__unit_001__evidence_aggregate_stub_v0",
        "q2__unit_002__evidence_aggregate_stub_v0",
    }
    assert stats["num_output_unit_evidence"] == 2
    for record in records:
        assert validate_unit_evidence_record(record) is None


def test_semantic_evidence_order_is_stable() -> None:
    records = [
        semantic_label_record(ablation_type="generalize"),
        semantic_label_record(ablation_type="delete"),
    ]
    summary = build_semantic_evidence_summary(records)

    assert summary["ablation_types"] == ["delete", "generalize"]
    assert summary["source_semantic_label_ids"] == [
        "q1__unit_001__delete__nli_stub_v0__sem_rule_v0",
        "q1__unit_001__generalize__nli_stub_v0__sem_rule_v0",
    ]


def test_semantic_summary_score_is_max_score() -> None:
    summary = build_semantic_evidence_summary(
        [
            semantic_label_record(ablation_type="delete", backward_entailment=0.45),
            semantic_label_record(ablation_type="generalize", backward_entailment=0.20),
        ]
    )

    assert summary["summary_score"] == max(summary["semantic_necessity_scores"])


def test_all_semantically_necessary_summary_label_is_consistent() -> None:
    summary = build_semantic_evidence_summary(
        [
            semantic_label_record(ablation_type="delete"),
            semantic_label_record(ablation_type="generalize"),
        ]
    )

    assert summary["summary_label"] == "consistent_semantic_necessity_evidence"


def test_mixed_semantic_necessity_summary_label() -> None:
    summary = build_semantic_evidence_summary(
        [
            semantic_label_record(ablation_type="delete"),
            equivalent_semantic_label_record(ablation_type="generalize"),
        ]
    )

    assert summary["summary_label"] == "mixed_semantic_necessity_evidence"


def test_no_semantic_necessity_summary_label() -> None:
    summary = build_semantic_evidence_summary(
        [
            equivalent_semantic_label_record(ablation_type="delete"),
            equivalent_semantic_label_record(ablation_type="generalize"),
        ]
    )

    assert summary["summary_label"] == "no_semantic_necessity_evidence"


def test_recoverability_evidence_copies_score_fields_without_top_level_recovery_samples() -> None:
    record = build_unit_evidence_record(
        [semantic_label_record()],
        recover_score_record(recoverability_label="Partially Recoverable", recoverability_score=0.5),
    )
    recoverability = record["recoverability_evidence"]

    assert recoverability["recover_score_id"] == "q1__unit_001__mask__score_stub_rule_v0"
    assert recoverability["masked_id"] == "q1__unit_001__mask"
    assert recoverability["recoverability_label"] == "Partially Recoverable"
    assert recoverability["recoverability_score"] == 0.5
    assert recoverability["source_recovered_questions"] == [ORIGINAL_QUESTION]
    assert recoverability["score_evidence"] == {"matched_original_question": False}
    assert "recovered_questions" not in record


def test_semantic_group_missing_recover_score_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing recover_score"):
        build_unit_evidence_records([semantic_label_record()], [])


def test_recover_score_missing_semantic_group_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no semantic group"):
        build_unit_evidence_records([], [recover_score_record()])


def test_duplicate_recover_score_for_unit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate recover_score"):
        build_unit_evidence_records(
            [semantic_label_record()],
            [recover_score_record(), recover_score_record()],
        )


def test_semantic_group_metadata_mismatch_raises_value_error() -> None:
    second = semantic_label_record(ablation_type="delete")
    second["original_question"] = "Different question."
    second["forward"]["premise"] = "Different question."
    second["backward"]["hypothesis"] = "Different question."

    with pytest.raises(ValueError, match="field=original_question"):
        build_unit_evidence_records(
            [semantic_label_record(), second],
            [recover_score_record()],
        )


def test_semantic_and_recover_score_metadata_mismatch_raises_value_error() -> None:
    recover_record = recover_score_record(original_question="Different question.")

    with pytest.raises(ValueError, match="field=original_question"):
        build_unit_evidence_records([semantic_label_record()], [recover_record])


def test_unsupported_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported unit evidence backend"):
        build_unit_evidence_records(
            [semantic_label_record()],
            [recover_score_record()],
            evidence_backend="manual",
        )


def test_missing_semantic_labels_path_raises_file_not_found(tmp_path: Path) -> None:
    recover_scores_path = tmp_path / "recover_scores.jsonl"
    write_jsonl([recover_score_record()], recover_scores_path)

    with pytest.raises(FileNotFoundError, match="Please run Sprint 1E first"):
        build_unit_evidence_file(
            tmp_path / "missing_semantic_labels.jsonl",
            recover_scores_path,
            tmp_path / "unit_evidence.jsonl",
        )


def test_missing_recover_scores_path_raises_file_not_found(tmp_path: Path) -> None:
    semantic_labels_path = tmp_path / "semantic_labels.jsonl"
    write_jsonl([semantic_label_record()], semantic_labels_path)

    with pytest.raises(FileNotFoundError, match="Please run Sprint 1H first"):
        build_unit_evidence_file(
            semantic_labels_path,
            tmp_path / "missing_recover_scores.jsonl",
            tmp_path / "unit_evidence.jsonl",
        )


def test_output_record_has_no_forbidden_top_level_fields() -> None:
    record = build_unit_evidence_record(
        [semantic_label_record()],
        recover_score_record(),
    )

    assert set(FORBIDDEN_FIELDS["unit_evidence"]).isdisjoint(record)


def test_cli_smoke_test_builds_unit_evidence(tmp_path: Path) -> None:
    semantic_labels_path = tmp_path / "semantic_labels.jsonl"
    recover_scores_path = tmp_path / "recover_scores.jsonl"
    output_path = tmp_path / "unit_evidence.jsonl"
    write_jsonl([semantic_label_record()], semantic_labels_path)
    write_jsonl([recover_score_record()], recover_scores_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "10_build_unit_evidence.py"),
            "--semantic-labels",
            str(semantic_labels_path),
            "--recover-scores",
            str(recover_scores_path),
            "--output",
            str(output_path),
            "--backend",
            DEFAULT_UNIT_EVIDENCE_BACKEND,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built unit evidence" in result.stdout
    assert "num_output_unit_evidence: 1" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert validate_unit_evidence_record(records[0]) is None
