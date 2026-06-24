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
from recover_attention.schemas import validate_semantic_label_record
from recover_attention.semantic_labels import (
    assign_semantic_necessity_label,
    compute_semantic_necessity_score,
    default_rule_parameters,
    label_nli_score_record,
    label_nli_score_records,
    validate_rule_parameters,
)


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
ABLATED_QUESTION = "Tom has some number apples and buys 2 more."
FUTURE_STAGE_FIELDS = {
    "recoverability_label",
    "recoverable",
    "masked_question",
    "recovered_question",
    "attention_anchor_label",
    "guidance_action",
    "hidden_states_path",
    "attentions_path",
}


def span() -> dict:
    return {
        "span_id": "span_001",
        "text": "3",
        "type": "number",
        "start": ORIGINAL_QUESTION.index("3"),
        "end": ORIGINAL_QUESTION.index("3") + 1,
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


def nli_score_record(
    forward_entailment: float,
    backward_entailment: float,
    forward_contradiction: float = 0.05,
    backward_contradiction: float = 0.05,
    ablation_type: str = "generalize",
) -> dict:
    forward_scores = scores(forward_entailment, forward_contradiction)
    backward_scores = scores(backward_entailment, backward_contradiction)
    return {
        "nli_id": f"q1__unit_001__{ablation_type}__nli_stub_v0",
        "ablation_id": f"q1__unit_001__{ablation_type}",
        "id": "q1",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [span()],
        "ablation_type": ablation_type,
        "original_question": ORIGINAL_QUESTION,
        "ablated_question": ABLATED_QUESTION,
        "nli_backend": "stub_v0",
        "language": "en",
        "language_setting": "auto",
        "forward": {
            "premise": ORIGINAL_QUESTION,
            "hypothesis": ABLATED_QUESTION,
            "label": label_from_scores(forward_scores),
            "scores": forward_scores,
        },
        "backward": {
            "premise": ABLATED_QUESTION,
            "hypothesis": ORIGINAL_QUESTION,
            "label": label_from_scores(backward_scores),
            "scores": backward_scores,
        },
        "bidirectional_entailment_score": min(
            forward_scores["entailment"],
            backward_scores["entailment"],
        ),
        "contradiction_score": max(
            forward_scores["contradiction"],
            backward_scores["contradiction"],
        ),
    }


def test_equivalent_label() -> None:
    record = label_nli_score_record(nli_score_record(0.80, 0.75))

    assert record["semantic_necessity_label"] == "Equivalent"
    assert record["is_semantically_necessary"] is False


def test_information_loss_label() -> None:
    record = label_nli_score_record(nli_score_record(0.65, 0.35))

    assert record["semantic_necessity_label"] == "Information Loss"
    assert record["is_semantically_necessary"] is True


def test_added_assumption_label() -> None:
    record = label_nli_score_record(nli_score_record(0.35, 0.65))

    assert record["semantic_necessity_label"] == "Added Assumption"
    assert record["is_semantically_necessary"] is True


def test_contradiction_high_has_priority() -> None:
    label, reason = assign_semantic_necessity_label(
        nli_score_record(0.35, 0.35, forward_contradiction=0.55)
    )

    assert label == "Non-equivalent"
    assert reason == "contradiction above threshold"


def test_both_low_is_non_equivalent() -> None:
    label, reason = assign_semantic_necessity_label(nli_score_record(0.30, 0.40))

    assert label == "Non-equivalent"
    assert reason == "low bidirectional entailment"


def test_semantic_necessity_score() -> None:
    record = label_nli_score_record(nli_score_record(0.65, 0.35))

    assert record["semantic_necessity_score"] == compute_semantic_necessity_score(
        record["bidirectional_entailment_score"],
        record["contradiction_score"],
    )


def test_semantic_label_id() -> None:
    record = label_nli_score_record(nli_score_record(0.65, 0.35))

    assert record["semantic_label_id"] == f"{record['nli_id']}__sem_rule_v0"


def test_rule_parameters_preserved() -> None:
    parameters = {
        "equivalent_threshold": 0.80,
        "directional_entailment_threshold": 0.60,
        "contradiction_threshold": 0.40,
    }
    record = label_nli_score_record(
        nli_score_record(0.65, 0.35),
        rule_parameters=parameters,
    )

    assert record["rule_parameters"] == parameters


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backend: manual"):
        label_nli_score_record(nli_score_record(0.65, 0.35), backend="manual")


def test_invalid_thresholds_raise() -> None:
    parameters = default_rule_parameters()
    parameters["equivalent_threshold"] = 1.1

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_rule_parameters(parameters)


def test_schema_validation_for_semantic_label() -> None:
    record = label_nli_score_record(nli_score_record(0.65, 0.35))

    assert validate_semantic_label_record(record) is None


def test_output_has_no_future_stage_fields() -> None:
    record = label_nli_score_record(nli_score_record(0.65, 0.35))

    assert FUTURE_STAGE_FIELDS.isdisjoint(record.keys())


def test_batch_stats() -> None:
    records, stats = label_nli_score_records(
        [
            nli_score_record(0.80, 0.75),
            nli_score_record(0.65, 0.35, ablation_type="delete"),
        ]
    )

    assert len(records) == 2
    assert stats["num_input_scores"] == 2
    assert stats["num_output_labels"] == 2
    assert stats["semantic_necessity_label_counts"] == {
        "Equivalent": 1,
        "Information Loss": 1,
    }
    assert stats["is_semantically_necessary_counts"] == {False: 1, True: 1}


def test_cli_smoke_test_builds_semantic_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "nli_scores.jsonl"
    output_path = tmp_path / "semantic_labels.jsonl"
    write_jsonl([nli_score_record(0.65, 0.35)], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "06_build_semantic_labels.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "rule_v0",
            "--equivalent-threshold",
            "0.70",
            "--directional-entailment-threshold",
            "0.50",
            "--contradiction-threshold",
            "0.50",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built semantic labels" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["semantic_label_backend"] == "rule_v0"
    assert "semantic_necessity_label" in records[0]
    assert "semantic_necessity_score" in records[0]
    assert "is_semantically_necessary" in records[0]
