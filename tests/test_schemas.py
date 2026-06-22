from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.schemas import (
    validate_ablated_question_record,
    validate_candidate_span_record,
    validate_masked_question_record,
    validate_nli_score_record,
    validate_question_record,
    validate_recover_output_record,
    validate_recover_score_record,
)


def valid_question_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "dataset": "gsm8k",
        "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "gold_answer": "5",
    }


def valid_candidate_span_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "candidates": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
    }


def valid_ablated_question_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "span_text": "3",
        "span_type": "number",
        "ablation_type": "generalize",
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "ablated_question": "Tom has some apples and buys 2 more. How many apples does he have now?",
    }


def valid_nli_score_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "span_text": "3",
        "span_type": "number",
        "ablation_type": "generalize",
        "original_to_ablated": "entailment",
        "ablated_to_original": "neutral",
        "semantic_necessity_label": "Information Loss",
    }


def valid_masked_question_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "span_text": "3",
        "span_type": "number",
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
    }


def valid_recover_output_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "sample_id": 0,
        "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
        "recovered_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "recoverable": "yes",
        "confidence": 0.82,
        "reason": "The missing number is likely recoverable from the original context pattern.",
    }


def valid_recover_score_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "recoverability_label": "Non-recoverable",
        "confidence_mean": 0.41,
        "recovery_consistency": 0.22,
        "misleading_recovery": False,
    }


def test_valid_question_record_passes() -> None:
    assert validate_question_record(valid_question_record()) is None


def test_question_record_missing_field_raises_value_error() -> None:
    record = valid_question_record()
    del record["gold_answer"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_question_record(record)


def test_question_record_empty_question_raises_value_error() -> None:
    record = valid_question_record()
    record["question"] = ""

    with pytest.raises(ValueError, match="non-empty"):
        validate_question_record(record)


def test_valid_candidate_span_record_passes() -> None:
    assert validate_candidate_span_record(valid_candidate_span_record()) is None


def test_candidate_span_with_invalid_span_type_raises_value_error() -> None:
    record = valid_candidate_span_record()
    record["candidates"][0]["type"] = "invalid"

    with pytest.raises(ValueError, match="invalid value"):
        validate_candidate_span_record(record)


def test_candidate_span_with_end_before_start_raises_value_error() -> None:
    record = valid_candidate_span_record()
    record["candidates"][0]["end"] = 8

    with pytest.raises(ValueError, match="greater than 'start'"):
        validate_candidate_span_record(record)


def test_valid_ablated_question_record_passes() -> None:
    assert validate_ablated_question_record(valid_ablated_question_record()) is None


def test_ablated_question_with_invalid_ablation_type_raises_value_error() -> None:
    record = valid_ablated_question_record()
    record["ablation_type"] = "rewrite"

    with pytest.raises(ValueError, match="invalid value"):
        validate_ablated_question_record(record)


def test_valid_nli_score_record_passes() -> None:
    assert validate_nli_score_record(valid_nli_score_record()) is None


def test_nli_score_with_invalid_semantic_necessity_label_raises_value_error() -> None:
    record = valid_nli_score_record()
    record["semantic_necessity_label"] = "Important"

    with pytest.raises(ValueError, match="invalid value"):
        validate_nli_score_record(record)


def test_valid_masked_question_record_passes() -> None:
    assert validate_masked_question_record(valid_masked_question_record()) is None


def test_valid_recover_output_record_passes() -> None:
    assert validate_recover_output_record(valid_recover_output_record()) is None


def test_recover_output_with_confidence_above_one_raises_value_error() -> None:
    record = valid_recover_output_record()
    record["confidence"] = 1.1

    with pytest.raises(ValueError, match="<= 1"):
        validate_recover_output_record(record)


def test_valid_recover_score_record_passes() -> None:
    assert validate_recover_score_record(valid_recover_score_record()) is None


def test_recover_score_with_invalid_recoverability_label_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["recoverability_label"] = "Unknown"

    with pytest.raises(ValueError, match="invalid value"):
        validate_recover_score_record(record)
