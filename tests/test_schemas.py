from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.schemas import (
    validate_ablated_question_record,
    validate_ablation_unit_record,
    validate_attention_anchor_label_record,
    validate_candidate_span_record,
    validate_cyber_sample_record,
    validate_intervention_manifest_record,
    validate_masked_question_record,
    validate_nli_score_record,
    validate_question_record,
    validate_recover_output_record,
    validate_recover_score_record,
    validate_semantic_label_record,
    validate_unit_evidence_record,
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


def valid_candidate_span_record_with_object() -> dict:
    record = valid_candidate_span_record()
    record["candidates"][0]["type"] = "object"
    return record


def valid_ablation_unit_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "units": [
            {
                "unit_id": "unit_001",
                "unit_scope": "single",
                "group_type": "single",
                "span_ids": ["span_001"],
                "spans": [
                    {
                        "span_id": "span_001",
                        "text": "3",
                        "type": "number",
                        "start": 8,
                        "end": 9,
                    }
                ],
                "reason": "single candidate span",
            }
        ],
    }


def valid_ablated_question_record() -> dict:
    return {
        "ablation_id": "gsm8k_0001__unit_001__generalize",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "ablation_type": "generalize",
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "ablated_question": "Tom has some number apples and buys 2 more. How many apples does he have now?",
    }


def valid_ablated_question_record_with_replace() -> dict:
    record = valid_ablated_question_record()
    record["ablation_type"] = "replace"
    return record


def old_span_level_ablated_question_record() -> dict:
    return {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "span_text": "3",
        "span_type": "number",
        "ablation_type": "generalize",
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "ablated_question": "Tom has some number apples and buys 2 more. How many apples does he have now?",
    }


def valid_nli_score_record() -> dict:
    ablated_record = valid_ablated_question_record()
    return {
        "nli_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0",
        "ablation_id": ablated_record["ablation_id"],
        "id": ablated_record["id"],
        "unit_id": ablated_record["unit_id"],
        "unit_scope": ablated_record["unit_scope"],
        "group_type": ablated_record["group_type"],
        "span_ids": list(ablated_record["span_ids"]),
        "spans": [dict(span) for span in ablated_record["spans"]],
        "ablation_type": ablated_record["ablation_type"],
        "original_question": ablated_record["original_question"],
        "ablated_question": ablated_record["ablated_question"],
        "nli_backend": "stub_v0",
        "language": "en",
        "language_setting": "auto",
        "forward": {
            "premise": ablated_record["original_question"],
            "hypothesis": ablated_record["ablated_question"],
            "label": "entailment",
            "scores": {
                "entailment": 0.75,
                "neutral": 0.20,
                "contradiction": 0.05,
            },
        },
        "backward": {
            "premise": ablated_record["ablated_question"],
            "hypothesis": ablated_record["original_question"],
            "label": "neutral",
            "scores": {
                "entailment": 0.35,
                "neutral": 0.60,
                "contradiction": 0.05,
            },
        },
        "bidirectional_entailment_score": 0.35,
        "contradiction_score": 0.05,
    }


def valid_semantic_label_record() -> dict:
    nli_record = valid_nli_score_record()
    return {
        "semantic_label_id": f"{nli_record['nli_id']}__sem_rule_v0",
        **nli_record,
        "semantic_label_backend": "rule_v0",
        "semantic_necessity_label": "Information Loss",
        "semantic_necessity_score": 0.65,
        "is_semantically_necessary": True,
        "rule_parameters": {
            "equivalent_threshold": 0.70,
            "directional_entailment_threshold": 0.50,
            "contradiction_threshold": 0.50,
        },
        "decision_reason": "forward entails ablated but backward does not entail original",
    }


def valid_masked_question_record() -> dict:
    return {
        "masked_id": "gsm8k_0001__unit_001__mask",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "source_semantic_label_ids": [
            "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
            "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0",
        ],
        "source_nli_ids": [
            "gsm8k_0001__unit_001__delete__nli_stub_v0",
            "gsm8k_0001__unit_001__generalize__nli_stub_v0",
        ],
        "source_ablation_ids": [
            "gsm8k_0001__unit_001__delete",
            "gsm8k_0001__unit_001__generalize",
        ],
        "semantic_sources": [
            {
                "semantic_label_id": "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
                "nli_id": "gsm8k_0001__unit_001__delete__nli_stub_v0",
                "ablation_id": "gsm8k_0001__unit_001__delete",
                "ablation_type": "delete",
                "semantic_necessity_label": "Information Loss",
                "semantic_necessity_score": 0.75,
                "is_semantically_necessary": True,
                "decision_reason": "forward entails ablated but backward does not entail original",
            },
            {
                "semantic_label_id": (
                    "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0"
                ),
                "nli_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0",
                "ablation_id": "gsm8k_0001__unit_001__generalize",
                "ablation_type": "generalize",
                "semantic_necessity_label": "Information Loss",
                "semantic_necessity_score": 0.65,
                "is_semantically_necessary": True,
                "decision_reason": "forward entails ablated but backward does not entail original",
            },
        ],
    }


def valid_group_masked_question_record() -> dict:
    record = valid_masked_question_record()
    record["masked_id"] = "gsm8k_0001__unit_003__mask"
    record["unit_id"] = "unit_003"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_002"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    record["masked_question"] = (
        "Tom has [MASK] apples and buys [MASK] more. How many apples does he have now?"
    )
    return record


def old_span_level_masked_question_record() -> dict:
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
        "masked_id": "gsm8k_0001__unit_001__mask",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovered_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "recovery_backend": "oracle_stub_v0",
        "sample_id": 0,
    }


def old_span_level_recover_output_record() -> dict:
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
        "recover_score_id": "gsm8k_0001__unit_001__mask__score_stub_rule_v0",
        "masked_id": "gsm8k_0001__unit_001__mask",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovery_backend": "oracle_stub_v0",
        "num_samples": 1,
        "source_sample_ids": [0],
        "recovered_questions": [
            "Tom has 3 apples and buys 2 more. How many apples does he have now?"
        ],
        "recoverability_label": "Recoverable",
        "recoverability_score": 1.0,
        "confidence_mean": 1.0,
        "recovery_consistency": 1.0,
        "misleading_recovery": False,
        "score_backend": "stub_rule_v0",
        "evidence": {"matched_original_question": True},
    }


def valid_group_recover_score_record() -> dict:
    record = valid_recover_score_record()
    record["recover_score_id"] = "gsm8k_0001__unit_008__mask__score_stub_rule_v0"
    record["masked_id"] = "gsm8k_0001__unit_008__mask"
    record["unit_id"] = "unit_008"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_002"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    record["masked_question"] = (
        "Tom has [MASK] apples and buys [MASK] more. How many apples does he have now?"
    )
    return record


def valid_unit_evidence_record() -> dict:
    return {
        "unit_evidence_id": "gsm8k_0001__unit_001__evidence_aggregate_stub_v0",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
        "semantic_evidence": {
            "source_semantic_label_ids": [
                "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0"
            ],
            "source_nli_ids": ["gsm8k_0001__unit_001__delete__nli_stub_v0"],
            "semantic_necessity_labels": ["Information Loss"],
            "semantic_necessity_scores": [0.75],
            "is_semantically_necessary_votes": [True],
            "summary_label": "Information Loss",
            "summary_score": 0.75,
        },
        "recoverability_evidence": {
            "recover_score_id": "gsm8k_0001__unit_001__mask__score_stub_rule_v0",
            "masked_id": "gsm8k_0001__unit_001__mask",
            "recovery_backend": "oracle_stub_v0",
            "score_backend": "stub_rule_v0",
            "recoverability_label": "Recoverable",
            "recoverability_score": 1.0,
            "confidence_mean": 1.0,
            "recovery_consistency": 1.0,
            "misleading_recovery": False,
        },
        "available_signal_types": [
            "semantic_necessity",
            "semantic_recoverability",
        ],
        "missing_signal_types": [
            "trajectory_stability",
            "answer_stability",
            "raw_attention_pattern",
            "attention_steering_effect",
        ],
        "evidence_backend": "aggregate_stub_v0",
        "evidence_status": "partial_stub_evidence",
        "evidence": {
            "notes": "synthetic unit evidence fixture",
            "source_files": [
                "data/processed/semantic_labels.jsonl",
                "data/processed/recover_scores.jsonl",
            ],
            "limitations": [
                "recoverability comes from oracle_stub_v0 + stub_rule_v0"
            ],
        },
    }


def valid_group_unit_evidence_record() -> dict:
    record = valid_unit_evidence_record()
    record["unit_evidence_id"] = "gsm8k_0001__unit_008__evidence_aggregate_stub_v0"
    record["unit_id"] = "unit_008"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_002"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    record["recoverability_evidence"] = {
        **record["recoverability_evidence"],
        "recover_score_id": "gsm8k_0001__unit_008__mask__score_stub_rule_v0",
        "masked_id": "gsm8k_0001__unit_008__mask",
    }
    return record


def valid_attention_anchor_label_record() -> dict:
    unit_evidence_id = "gsm8k_0001__unit_001__evidence_aggregate_stub_v0"
    return {
        "attention_anchor_label_id": f"{unit_evidence_id}__anchor_early_evidence_rule_stub_v0",
        "unit_evidence_id": unit_evidence_id,
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}
        ],
        "original_question": "Tom has 3 apples and buys 2 more.",
        "semantic_evidence": {"summary_label": "Information Loss", "summary_score": 0.75},
        "recoverability_evidence": {"recoverability_label": "Recoverable", "recoverability_score": 1.0},
        "available_signal_types": ["semantic_necessity", "semantic_recoverability"],
        "missing_signal_types": [
            "trajectory_stability",
            "answer_stability",
            "raw_attention_pattern",
            "attention_steering_effect",
        ],
        "attention_importance_score": 0.5,
        "attention_anchor_label": "Weak Anchor",
        "label_backend": "early_evidence_rule_stub_v0",
        "label_status": "partial_evidence_label",
        "evidence": {"notes": "Example only; builder implementation belongs to a later sprint."},
    }


def valid_group_attention_anchor_label_record() -> dict:
    record = valid_attention_anchor_label_record()
    record["unit_id"] = "unit_003"
    unit_evidence_id = "gsm8k_0001__unit_003__evidence_aggregate_stub_v0"
    record["unit_evidence_id"] = unit_evidence_id
    record["attention_anchor_label_id"] = (
        f"{unit_evidence_id}__anchor_early_evidence_rule_stub_v0"
    )
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_002"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    return record


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


def test_candidate_span_with_object_span_type_passes() -> None:
    assert validate_candidate_span_record(valid_candidate_span_record_with_object()) is None


def test_candidate_span_with_end_before_start_raises_value_error() -> None:
    record = valid_candidate_span_record()
    record["candidates"][0]["end"] = 8

    with pytest.raises(ValueError, match="greater than 'start'"):
        validate_candidate_span_record(record)


def test_valid_ablation_unit_record_passes() -> None:
    assert validate_ablation_unit_record(valid_ablation_unit_record()) is None


def test_ablation_unit_record_with_empty_units_passes() -> None:
    record = valid_ablation_unit_record()
    record["units"] = []

    assert validate_ablation_unit_record(record) is None


def test_ablation_unit_group_with_one_span_raises_value_error() -> None:
    record = valid_ablation_unit_record()
    record["units"][0]["unit_scope"] = "group"
    record["units"][0]["group_type"] = "number_set"

    with pytest.raises(ValueError, match="at least two spans"):
        validate_ablation_unit_record(record)


def test_ablation_unit_offset_mismatch_raises_value_error() -> None:
    record = valid_ablation_unit_record()
    record["units"][0]["spans"][0]["text"] = "4"

    with pytest.raises(ValueError, match="offsets"):
        validate_ablation_unit_record(record)


def test_valid_ablated_question_record_passes() -> None:
    assert validate_ablated_question_record(valid_ablated_question_record()) is None


def test_ablated_question_missing_ablation_id_raises_value_error() -> None:
    record = valid_ablated_question_record()
    del record["ablation_id"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_ablated_question_record(record)


def test_old_span_level_ablated_question_record_raises_value_error() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        validate_ablated_question_record(old_span_level_ablated_question_record())


def test_ablated_question_with_invalid_ablation_type_raises_value_error() -> None:
    record = valid_ablated_question_record()
    record["ablation_type"] = "rewrite"

    with pytest.raises(ValueError, match="invalid value"):
        validate_ablated_question_record(record)


def test_ablated_question_with_replace_ablation_type_passes() -> None:
    with pytest.raises(ValueError, match="invalid value"):
        validate_ablated_question_record(valid_ablated_question_record_with_replace())


def test_ablated_question_with_offset_mismatch_raises_value_error() -> None:
    record = valid_ablated_question_record()
    record["spans"][0]["text"] = "4"

    with pytest.raises(ValueError, match="offsets"):
        validate_ablated_question_record(record)


def test_ablated_question_with_unchanged_question_raises_value_error() -> None:
    record = valid_ablated_question_record()
    record["ablated_question"] = record["original_question"]

    with pytest.raises(ValueError, match="differ"):
        validate_ablated_question_record(record)


def test_ablated_question_group_with_one_span_raises_value_error() -> None:
    record = valid_ablated_question_record()
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"

    with pytest.raises(ValueError, match="at least two spans"):
        validate_ablated_question_record(record)


def test_valid_nli_score_record_passes() -> None:
    assert validate_nli_score_record(valid_nli_score_record()) is None


def test_nli_score_with_invalid_backend_raises_value_error() -> None:
    record = valid_nli_score_record()
    record["nli_backend"] = "hf_xnli"

    with pytest.raises(ValueError, match="invalid value"):
        validate_nli_score_record(record)


def test_nli_score_with_invalid_bidirectional_score_raises_value_error() -> None:
    record = valid_nli_score_record()
    record["bidirectional_entailment_score"] = 0.99

    with pytest.raises(ValueError, match="bidirectional_entailment_score"):
        validate_nli_score_record(record)


def test_nli_score_with_reversed_direction_texts_raises_value_error() -> None:
    record = valid_nli_score_record()
    record["forward"]["premise"] = record["ablated_question"]
    record["forward"]["hypothesis"] = record["original_question"]
    record["backward"]["premise"] = record["original_question"]
    record["backward"]["hypothesis"] = record["ablated_question"]

    with pytest.raises(ValueError, match="forward premise"):
        validate_nli_score_record(record)


def test_nli_score_with_semantic_necessity_label_raises_value_error() -> None:
    record = valid_nli_score_record()
    record["semantic_necessity_label"] = "Information Loss"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_nli_score_record(record)


def test_old_directional_nli_score_record_raises_value_error() -> None:
    record = {
        "id": "gsm8k_0001",
        "span_id": "span_001",
        "span_text": "3",
        "span_type": "number",
        "ablation_type": "generalize",
        "original_to_ablated": "entailment",
        "ablated_to_original": "neutral",
    }

    with pytest.raises(ValueError, match="forbidden field"):
        validate_nli_score_record(record)


def test_valid_semantic_label_record_passes() -> None:
    assert validate_semantic_label_record(valid_semantic_label_record()) is None


def test_semantic_label_missing_semantic_label_id_raises_value_error() -> None:
    record = valid_semantic_label_record()
    del record["semantic_label_id"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_semantic_label_record(record)


def test_semantic_label_with_invalid_label_raises_value_error() -> None:
    record = valid_semantic_label_record()
    record["semantic_necessity_label"] = "Necessary"

    with pytest.raises(ValueError, match="invalid value"):
        validate_semantic_label_record(record)


def test_semantic_label_with_wrong_score_raises_value_error() -> None:
    record = valid_semantic_label_record()
    record["semantic_necessity_score"] = 0.10

    with pytest.raises(ValueError, match="semantic_necessity_score"):
        validate_semantic_label_record(record)


def test_semantic_label_equivalent_marked_necessary_raises_value_error() -> None:
    record = valid_semantic_label_record()
    record["semantic_necessity_label"] = "Equivalent"
    record["is_semantically_necessary"] = True

    with pytest.raises(ValueError, match="is_semantically_necessary"):
        validate_semantic_label_record(record)


def test_semantic_label_non_equivalent_marked_not_necessary_raises_value_error() -> None:
    record = valid_semantic_label_record()
    record["semantic_necessity_label"] = "Non-equivalent"
    record["is_semantically_necessary"] = False

    with pytest.raises(ValueError, match="is_semantically_necessary"):
        validate_semantic_label_record(record)


def test_valid_masked_question_record_passes() -> None:
    assert validate_masked_question_record(valid_masked_question_record()) is None


def test_valid_group_masked_question_record_passes() -> None:
    assert validate_masked_question_record(valid_group_masked_question_record()) is None


def test_group_masked_question_with_single_mask_raises_value_error() -> None:
    record = valid_group_masked_question_record()
    record["masked_question"] = (
        "Tom has [MASK] apples and buys 2 more. How many apples does he have now?"
    )
    with pytest.raises(ValueError, match="replace_each_span"):
        validate_masked_question_record(record)


def test_masked_question_with_invalid_mask_backend_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["mask_backend"] = "rule_v0"
    with pytest.raises(ValueError, match="mask_backend"):
        validate_masked_question_record(record)


def test_masked_question_with_invalid_mask_strategy_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["mask_strategy"] = "replace_all"
    with pytest.raises(ValueError, match="mask_strategy"):
        validate_masked_question_record(record)


def test_masked_question_added_mask_count_ignores_preexisting_mask_token() -> None:
    record = valid_masked_question_record()
    record["original_question"] = "Given [MASK] context, Tom has 3 apples and buys 2 more."
    record["masked_question"] = "Given [MASK] context, Tom has [MASK] apples and buys 2 more."
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 30, "end": 31}
    ]
    assert validate_masked_question_record(record) is None


def test_masked_question_missing_masked_id_raises_value_error() -> None:
    record = valid_masked_question_record()
    del record["masked_id"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_masked_question_record(record)


def test_old_span_level_masked_question_record_raises_value_error() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        validate_masked_question_record(old_span_level_masked_question_record())


def test_masked_question_with_top_level_span_fields_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["span_id"] = "span_001"
    record["span_text"] = "3"
    record["span_type"] = "number"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_masked_question_record(record)


def test_masked_question_without_mask_token_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["masked_question"] = "Tom has apples and buys 2 more. How many apples does he have now?"

    with pytest.raises(ValueError, match="mask_token"):
        validate_masked_question_record(record)


def test_masked_question_with_unchanged_question_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["original_question"] = record["masked_question"]

    with pytest.raises(ValueError, match="differ"):
        validate_masked_question_record(record)


def test_masked_question_with_span_order_mismatch_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["span_ids"] = ["span_002"]

    with pytest.raises(ValueError, match="span_ids order"):
        validate_masked_question_record(record)


def test_masked_question_with_semantic_source_order_mismatch_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["source_semantic_label_ids"] = list(reversed(record["source_semantic_label_ids"]))

    with pytest.raises(ValueError, match="semantic_label_id order"):
        validate_masked_question_record(record)


def test_masked_question_with_invalid_semantic_necessity_label_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["semantic_sources"][0]["semantic_necessity_label"] = "Necessary"

    with pytest.raises(ValueError, match="invalid value"):
        validate_masked_question_record(record)


def test_masked_question_with_non_bool_semantic_necessity_flag_raises_value_error() -> None:
    record = valid_masked_question_record()
    record["semantic_sources"][0]["is_semantically_necessary"] = "true"

    with pytest.raises(ValueError, match="bool"):
        validate_masked_question_record(record)


def test_valid_recover_output_record_passes() -> None:
    assert validate_recover_output_record(valid_recover_output_record()) is None


def test_recover_output_with_empty_recovered_question_passes() -> None:
    record = valid_recover_output_record()
    record["recovered_question"] = ""

    assert validate_recover_output_record(record) is None


def test_old_span_level_recover_output_record_raises_value_error() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        validate_recover_output_record(old_span_level_recover_output_record())


def test_recover_output_with_negative_sample_id_raises_value_error() -> None:
    record = valid_recover_output_record()
    record["sample_id"] = -1

    with pytest.raises(ValueError, match=">= 0"):
        validate_recover_output_record(record)


def test_recover_output_with_wrong_masked_id_raises_value_error() -> None:
    record = valid_recover_output_record()
    record["masked_id"] = "wrong"

    with pytest.raises(ValueError, match="masked_id"):
        validate_recover_output_record(record)


def test_recover_output_with_invalid_recovery_backend_raises_value_error() -> None:
    record = valid_recover_output_record()
    record["recovery_backend"] = "stub_v0"

    with pytest.raises(ValueError, match="invalid value"):
        validate_recover_output_record(record)


def test_valid_recover_score_record_passes() -> None:
    assert validate_recover_score_record(valid_recover_score_record()) is None


def test_valid_group_recover_score_record_passes() -> None:
    assert validate_recover_score_record(valid_group_recover_score_record()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("span_id", "span_001"),
        ("span_text", "3"),
        ("span_type", "number"),
        ("sample_id", 0),
        ("recovered_question", "Tom has 3 apples and buys 2 more."),
        ("recoverable", "yes"),
        ("confidence", 0.8),
        ("reason", "legacy field"),
        ("attention_anchor_label", "Strong Anchor"),
        ("guidance_action", "boost"),
        ("guidance_strength", 0.7),
    ],
)
def test_recover_score_with_forbidden_top_level_field_raises_value_error(
    field: str,
    value: object,
) -> None:
    record = valid_recover_score_record()
    record[field] = value

    with pytest.raises(ValueError, match="forbidden field"):
        validate_recover_score_record(record)


def test_recover_score_with_wrong_masked_id_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["masked_id"] = "wrong"

    with pytest.raises(ValueError, match="masked_id"):
        validate_recover_score_record(record)


def test_recover_score_with_wrong_recover_score_id_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["recover_score_id"] = "wrong"

    with pytest.raises(ValueError, match="recover_score_id"):
        validate_recover_score_record(record)


def test_recover_score_with_span_length_mismatch_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["span_ids"] = ["span_001", "span_002"]

    with pytest.raises(ValueError, match="same length"):
        validate_recover_score_record(record)


def test_recover_score_with_span_order_mismatch_raises_value_error() -> None:
    record = valid_group_recover_score_record()
    record["span_ids"] = ["span_002", "span_001"]

    with pytest.raises(ValueError, match="span_ids order"):
        validate_recover_score_record(record)


def test_recover_score_single_unit_with_multiple_spans_raises_value_error() -> None:
    record = valid_group_recover_score_record()
    record["recover_score_id"] = "gsm8k_0001__unit_001__mask__score_stub_rule_v0"
    record["masked_id"] = "gsm8k_0001__unit_001__mask"
    record["unit_id"] = "unit_001"
    record["unit_scope"] = "single"
    record["group_type"] = "single"

    with pytest.raises(ValueError, match="exactly one span"):
        validate_recover_score_record(record)


def test_recover_score_group_unit_with_one_span_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["recover_score_id"] = "gsm8k_0001__unit_008__mask__score_stub_rule_v0"
    record["masked_id"] = "gsm8k_0001__unit_008__mask"
    record["unit_id"] = "unit_008"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"

    with pytest.raises(ValueError, match="at least two spans"):
        validate_recover_score_record(record)


def test_recover_score_with_num_samples_zero_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["num_samples"] = 0

    with pytest.raises(ValueError, match=">= 1"):
        validate_recover_score_record(record)


def test_recover_score_with_source_sample_ids_length_mismatch_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["source_sample_ids"] = [0, 1]

    with pytest.raises(ValueError, match="source_sample_ids"):
        validate_recover_score_record(record)


def test_recover_score_with_recovered_questions_length_mismatch_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["recovered_questions"] = []

    with pytest.raises(ValueError, match="recovered_questions"):
        validate_recover_score_record(record)


@pytest.mark.parametrize(
    "field",
    ["recoverability_score", "confidence_mean", "recovery_consistency"],
)
def test_recover_score_with_score_above_one_raises_value_error(field: str) -> None:
    record = valid_recover_score_record()
    record[field] = 1.1

    with pytest.raises(ValueError, match="<= 1"):
        validate_recover_score_record(record)


def test_recover_score_with_unsupported_score_backend_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["score_backend"] = "manual"

    with pytest.raises(ValueError, match="invalid value"):
        validate_recover_score_record(record)


def test_recover_score_with_invalid_recoverability_label_raises_value_error() -> None:
    record = valid_recover_score_record()
    record["recoverability_label"] = "Unknown"

    with pytest.raises(ValueError, match="invalid value"):
        validate_recover_score_record(record)


def test_valid_unit_evidence_record_passes() -> None:
    assert validate_unit_evidence_record(valid_unit_evidence_record()) is None


def test_valid_group_unit_evidence_record_passes() -> None:
    assert validate_unit_evidence_record(valid_group_unit_evidence_record()) is None


def test_unit_evidence_missing_required_field_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    del record["semantic_evidence"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_unit_evidence_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attention_anchor_label", "Strong Anchor"),
        ("guidance_action", "boost"),
        ("span_id", "span_001"),
        ("span_text", "3"),
        ("span_type", "number"),
        ("recovered_question", "Tom has 3 apples and buys 2 more."),
        ("hidden_states", "cache/hidden_states/example.pt"),
        ("attention_maps", "cache/attentions/example.pt"),
    ],
)
def test_unit_evidence_with_forbidden_top_level_field_raises_value_error(
    field: str,
    value: object,
) -> None:
    record = valid_unit_evidence_record()
    record[field] = value

    with pytest.raises(ValueError, match="forbidden field"):
        validate_unit_evidence_record(record)


def test_unit_evidence_with_wrong_unit_evidence_id_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    record["unit_evidence_id"] = "wrong"

    with pytest.raises(ValueError, match="unit_evidence_id"):
        validate_unit_evidence_record(record)


def test_unit_evidence_with_invalid_evidence_backend_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    record["evidence_backend"] = "manual"

    with pytest.raises(ValueError, match="invalid value"):
        validate_unit_evidence_record(record)


def test_unit_evidence_with_invalid_evidence_status_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    record["evidence_status"] = "full_evidence"

    with pytest.raises(ValueError, match="invalid value"):
        validate_unit_evidence_record(record)


def test_unit_evidence_with_invalid_signal_type_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    record["available_signal_types"] = ["semantic_necessity", "semantic_similarity"]

    with pytest.raises(ValueError, match="available_signal_types"):
        validate_unit_evidence_record(record)


def test_unit_evidence_single_unit_with_multiple_spans_raises_value_error() -> None:
    record = valid_group_unit_evidence_record()
    record["unit_evidence_id"] = "gsm8k_0001__unit_001__evidence_aggregate_stub_v0"
    record["unit_id"] = "unit_001"
    record["unit_scope"] = "single"
    record["group_type"] = "single"

    with pytest.raises(ValueError, match="exactly one span"):
        validate_unit_evidence_record(record)


def test_unit_evidence_group_unit_with_one_span_raises_value_error() -> None:
    record = valid_unit_evidence_record()
    record["unit_evidence_id"] = "gsm8k_0001__unit_008__evidence_aggregate_stub_v0"
    record["unit_id"] = "unit_008"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"

    with pytest.raises(ValueError, match="at least two spans"):
        validate_unit_evidence_record(record)


def test_unit_evidence_with_span_order_mismatch_raises_value_error() -> None:
    record = valid_group_unit_evidence_record()
    record["span_ids"] = ["span_002", "span_001"]

    with pytest.raises(ValueError, match="span_ids order"):
        validate_unit_evidence_record(record)


def test_valid_attention_anchor_label_record_passes() -> None:
    assert validate_attention_anchor_label_record(valid_attention_anchor_label_record()) is None


def test_valid_group_attention_anchor_label_record_passes() -> None:
    assert (
        validate_attention_anchor_label_record(valid_group_attention_anchor_label_record())
        is None
    )


def test_attention_anchor_label_record_missing_required_field_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    del record["evidence"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_top_level_span_id_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["span_id"] = "span_001"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_top_level_span_text_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["span_text"] = "3"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_guidance_action_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["guidance_action"] = "boost"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_guidance_strength_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["guidance_strength"] = 0.75

    with pytest.raises(ValueError, match="forbidden field"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_invalid_label_id_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["attention_anchor_label_id"] = "wrong_id"

    with pytest.raises(ValueError, match="attention_anchor_label_id"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_invalid_label_backend_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["label_backend"] = "made_up_backend"

    with pytest.raises(ValueError, match="invalid value"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_invalid_label_status_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["label_status"] = "final_label"

    with pytest.raises(ValueError, match="invalid value"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_invalid_anchor_label_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["attention_anchor_label"] = "Important"

    with pytest.raises(ValueError, match="invalid value"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_importance_score_above_one_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["attention_importance_score"] = 1.1

    with pytest.raises(ValueError, match="<= 1"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_with_invalid_signal_type_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["available_signal_types"] = ["made_up_signal"]

    with pytest.raises(ValueError, match="invalid value"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_single_scope_with_two_spans_raises_value_error() -> None:
    record = valid_attention_anchor_label_record()
    record["span_ids"] = ["span_001", "span_002"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27},
    ]

    with pytest.raises(ValueError, match="single"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_group_scope_with_one_span_raises_value_error() -> None:
    record = valid_group_attention_anchor_label_record()
    record["span_ids"] = ["span_001"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}
    ]

    with pytest.raises(ValueError, match="at least two spans"):
        validate_attention_anchor_label_record(record)


def test_attention_anchor_label_record_span_order_mismatch_raises_value_error() -> None:
    record = valid_group_attention_anchor_label_record()
    record["span_ids"] = ["span_002", "span_001"]

    with pytest.raises(ValueError, match="order must match"):
        validate_attention_anchor_label_record(record)


def valid_intervention_manifest_record() -> dict:
    anchor_id = (
        "gsm8k_0001__unit_001__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0"
    )
    return {
        "intervention_id": f"{anchor_id}__intervention_mask_manifest_stub_v0",
        "attention_anchor_label_id": anchor_id,
        "unit_evidence_id": "gsm8k_0001__unit_001__evidence_aggregate_stub_v0",
        "id": "gsm8k_0001",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [{"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}],
        "original_question": "Tom has 3 apples and buys 2 more.",
        "attention_importance_score": 0.55,
        "attention_anchor_label": "Medium Anchor",
        "label_backend": "early_evidence_rule_stub_v0",
        "label_status": "partial_evidence_label",
        "intervention_type": "mask",
        "target_scope": "unit",
        "intervention_backend": "manifest_stub_v0",
        "intervention_status": "planned_only",
        "planned_operation": {"description": "mask all spans in unit", "target_span_ids": ["span_001"]},
        "evidence": {"notes": "fixture"},
    }


def valid_group_intervention_manifest_record() -> dict:
    record = valid_intervention_manifest_record()
    anchor_id = (
        "gsm8k_0001__unit_009__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0"
    )
    record["attention_anchor_label_id"] = anchor_id
    record["unit_evidence_id"] = "gsm8k_0001__unit_009__evidence_aggregate_stub_v0"
    record["intervention_id"] = f"{anchor_id}__intervention_mask_manifest_stub_v0"
    record["unit_id"] = "unit_009"
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_004"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_004", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    return record


def test_valid_intervention_manifest_record_passes() -> None:
    assert validate_intervention_manifest_record(valid_intervention_manifest_record()) is None


def test_valid_group_intervention_manifest_record_passes() -> None:
    assert (
        validate_intervention_manifest_record(valid_group_intervention_manifest_record()) is None
    )


def test_intervention_manifest_missing_required_field_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    del record["planned_operation"]

    with pytest.raises(ValueError, match="missing required field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_top_level_span_id_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["span_id"] = "span_001"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_guidance_action_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["guidance_action"] = "boost"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_guidance_strength_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["guidance_strength"] = 0.5

    with pytest.raises(ValueError, match="forbidden field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_hidden_states_path_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["hidden_states_path"] = "cache/hidden_states/x.pt"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_attentions_path_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["attentions_path"] = "cache/attentions/x.pt"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_intervention_id_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["intervention_id"] = "wrong_id"

    with pytest.raises(ValueError, match="intervention_id"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_intervention_type_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["intervention_type"] = "rewrite"
    record["intervention_id"] = (
        f"{record['attention_anchor_label_id']}__intervention_rewrite_manifest_stub_v0"
    )

    with pytest.raises(ValueError, match="invalid value"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_target_scope_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["target_scope"] = "token"

    with pytest.raises(ValueError, match="invalid value"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_backend_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["intervention_backend"] = "made_up"
    record["intervention_id"] = (
        f"{record['attention_anchor_label_id']}__intervention_mask_made_up"
    )

    with pytest.raises(ValueError, match="invalid value"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_status_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["intervention_status"] = "executed"

    with pytest.raises(ValueError, match="invalid value"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_importance_score_above_one_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["attention_importance_score"] = 1.1

    with pytest.raises(ValueError, match="<= 1"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_with_invalid_anchor_label_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["attention_anchor_label"] = "Important"

    with pytest.raises(ValueError, match="invalid value"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_single_scope_with_two_spans_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["span_ids"] = ["span_001", "span_004"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_004", "text": "2", "type": "number", "start": 26, "end": 27},
    ]

    with pytest.raises(ValueError, match="single"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_group_scope_with_one_span_raises_value_error() -> None:
    record = valid_group_intervention_manifest_record()
    record["span_ids"] = ["span_001"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}
    ]

    with pytest.raises(ValueError, match="at least two spans"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_span_order_mismatch_raises_value_error() -> None:
    record = valid_group_intervention_manifest_record()
    record["span_ids"] = ["span_004", "span_001"]

    with pytest.raises(ValueError, match="order must match"):
        validate_intervention_manifest_record(record)


def test_intervention_manifest_planned_operation_not_dict_raises_value_error() -> None:
    record = valid_intervention_manifest_record()
    record["planned_operation"] = ["not", "a", "dict"]

    with pytest.raises(ValueError, match="planned_operation"):
        validate_intervention_manifest_record(record)


def valid_cyber_sample_record() -> dict:
    return {
        "example_id": "cybermetric_000001",
        "dataset": "cybermetric",
        "source": "CyberMetric-2000-v1",
        "group_id": "cybermetric_question_abc",
        "task_type": "multiple_choice_qa",
        "input_text": "",
        "question": "Which option is correct?",
        "candidate_labels": ["A", "B", "C", "D"],
        "candidate_choices": [
            {"choice": "A", "label_id": None, "label_text": "One", "original_position": 2},
            {"choice": "B", "label_id": None, "label_text": "Two", "original_position": 0},
            {"choice": "C", "label_id": None, "label_text": "Three", "original_position": 3},
            {"choice": "D", "label_id": None, "label_text": "Four", "original_position": 1},
        ],
        "gold_label": "B",
        "gold_label_id": None,
        "gold_label_text": "Two",
        "label_space": "mcq_option_letter",
        "metadata": {
            "original_index": 1,
            "original_gold_position": 0,
            "shuffled_gold_position": 1,
            "option_shuffle_seed": 42,
            "split_seed": 42,
            "split": "train",
            "raw_category": None,
        },
    }


def test_valid_cyber_sample_record_passes() -> None:
    assert validate_cyber_sample_record(valid_cyber_sample_record()) is None


def test_cyber_sample_missing_required_field_fails() -> None:
    record = valid_cyber_sample_record()
    del record["source"]
    with pytest.raises(ValueError, match="missing required field"):
        validate_cyber_sample_record(record)


def test_cyber_sample_candidate_order_mismatch_fails() -> None:
    record = valid_cyber_sample_record()
    record["candidate_choices"][0]["choice"] = "B"
    with pytest.raises(ValueError, match="order"):
        validate_cyber_sample_record(record)


def test_cyber_sample_gold_label_outside_candidates_fails() -> None:
    record = valid_cyber_sample_record()
    record["gold_label"] = "Z"
    with pytest.raises(ValueError, match="candidate_labels"):
        validate_cyber_sample_record(record)


def test_cyber_sample_gold_text_mismatch_fails() -> None:
    record = valid_cyber_sample_record()
    record["gold_label_text"] = "Wrong"
    with pytest.raises(ValueError, match="gold_label_text"):
        validate_cyber_sample_record(record)


def test_cyber_sample_duplicate_candidate_label_fails() -> None:
    record = valid_cyber_sample_record()
    record["candidate_labels"][1] = "A"
    with pytest.raises(ValueError, match="unique"):
        validate_cyber_sample_record(record)


def test_cyber_sample_invalid_label_space_fails() -> None:
    record = valid_cyber_sample_record()
    record["label_space"] = "semantic_id"
    with pytest.raises(ValueError, match="mcq_option_letter"):
        validate_cyber_sample_record(record)


def test_cyber_sample_invalid_split_fails() -> None:
    record = valid_cyber_sample_record()
    record["metadata"]["split"] = "validation"
    with pytest.raises(ValueError, match="invalid value"):
        validate_cyber_sample_record(record)


def test_cyber_sample_non_integer_original_position_fails() -> None:
    record = valid_cyber_sample_record()
    record["candidate_choices"][0]["original_position"] = "2"
    with pytest.raises(ValueError, match="must be an int"):
        validate_cyber_sample_record(record)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [(("question",), ""), (("candidate_choices", 0, "label_text"), "")],
)
def test_cyber_sample_empty_text_fails(field_path: tuple, value: str) -> None:
    record = deepcopy(valid_cyber_sample_record())
    if len(field_path) == 1:
        record[field_path[0]] = value
    else:
        record[field_path[0]][field_path[1]][field_path[2]] = value
    with pytest.raises(ValueError, match="non-empty"):
        validate_cyber_sample_record(record)
