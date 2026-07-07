from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.answer_position_output_effect import (  # noqa: E402
    build_answer_position_reports,
    build_formula_scores,
    build_response_prompt,
    build_score_matrix,
    feature_leakage_audit,
)
from recover_attention.data_io import write_jsonl  # noqa: E402


def base_records() -> list[dict]:
    rows = []
    for qid in ["q1", "q2"]:
        rows.append(
            {
                "source_question_id": qid,
                "span_id": f"{qid}__on",
                "span_text": "3",
                "span_type": "number",
                "question": "Alice has 3 apples and buys 2 more. How many apples?",
                "masked_question": "Alice has [MASK] apples and buys 2 more. How many apples?",
                "surface_features": {"surface_keyness_proxy": 0.7},
                "attention_features": {
                    "original_to_masked_attention_delta": 0.7,
                    "attention_entropy_delta": 0.2,
                    "attention_rank_delta": 0.1,
                    "span_attention_in_mass": 0.8,
                    "context_to_slot_attention": 0.7,
                },
                "hidden_features": {},
                "output_effect_features": {
                    "self_output_kl": 0.1,
                    "self_output_js": 0.05,
                    "self_output_top1_changed": 0.0,
                    "self_output_topk_overlap": 0.7,
                    "self_output_logprob_delta": -0.1,
                },
                "diagnostic_labels_for_eval_only": {
                    "solution_path_status": "on_solution_path_number",
                    "weak_semantic_keyness": None,
                },
            }
        )
        rows.append(
            {
                "source_question_id": qid,
                "span_id": f"{qid}__off",
                "span_text": "2",
                "span_type": "number",
                "question": "Alice has 3 apples and buys 2 more. How many apples?",
                "masked_question": "Alice has 3 apples and buys [MASK] more. How many apples?",
                "surface_features": {"surface_keyness_proxy": 0.3},
                "attention_features": {
                    "original_to_masked_attention_delta": 0.1,
                    "attention_entropy_delta": 0.0,
                    "attention_rank_delta": 0.0,
                    "span_attention_in_mass": 0.2,
                    "context_to_slot_attention": 0.1,
                },
                "hidden_features": {},
                "output_effect_features": {
                    "self_output_kl": 0.02,
                    "self_output_js": 0.01,
                    "self_output_top1_changed": 0.0,
                    "self_output_topk_overlap": 0.9,
                    "self_output_logprob_delta": -0.01,
                },
                "diagnostic_labels_for_eval_only": {
                    "solution_path_status": "off_solution_path_number",
                    "weak_semantic_keyness": None,
                },
            }
        )
    return rows


def response_records() -> list[dict]:
    rows = []
    for base in base_records():
        on = base["span_id"].endswith("__on")
        rows.append(
            {
                "source_question_id": base["source_question_id"],
                "span_id": base["span_id"],
                "span_text": base["span_text"],
                "span_type": base["span_type"],
                "question": base["question"],
                "masked_question": base["masked_question"],
                "resp_pos_output_effect": {
                    "self_output_kl": 0.3 if on else 0.01,
                    "self_output_js": 0.2 if on else 0.005,
                    "self_output_top1_changed": 1.0 if on else 0.0,
                    "self_output_topk_overlap": 0.2 if on else 0.9,
                    "self_output_logprob_delta": -0.5 if on else -0.01,
                },
                "resp_pos_output_shift": 1.0 if on else 0.0,
                "diagnostic_labels_for_eval_only": base["diagnostic_labels_for_eval_only"],
            }
        )
    return rows


def test_build_response_prompt_has_no_solution_or_cot() -> None:
    prompt = build_response_prompt("What is 2+2?")

    assert prompt == "Question: What is 2+2?\nAnswer:"
    assert "solution" not in prompt.lower()
    assert "chain" not in prompt.lower()


def test_feature_leakage_audit_rejects_answer_feature_name() -> None:
    ok = feature_leakage_audit(response_records())
    assert ok["passed"] is True

    bad_rows = response_records()
    bad_rows[0]["resp_pos_output_effect"]["answer_logprob_delta"] = 1.0
    bad = feature_leakage_audit(bad_rows)
    assert bad["passed"] is False
    assert bad["leaked_input_features"][0]["forbidden_substring"] == "answer"


def test_build_score_matrix_and_formulas_rank_on_path_higher() -> None:
    rows = build_score_matrix(base_records(), response_records())
    formulas = build_formula_scores(rows)

    for qid in ["q1", "q2"]:
        on_id = f"{qid}__on"
        off_id = f"{qid}__off"
        assert formulas["D_response_position_output_only"]["scores"][on_id] > formulas[
            "D_response_position_output_only"
        ]["scores"][off_id]
        assert formulas["F_attention_x_response_position_output"]["scores"][on_id] > formulas[
            "F_attention_x_response_position_output"
        ]["scores"][off_id]


def test_build_answer_position_reports_writes_outputs(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    response_path = tmp_path / "response.jsonl"
    write_jsonl(base_records(), base_path)
    write_jsonl(response_records(), response_path)

    result = build_answer_position_reports(
        base_feature_matrix_path=base_path,
        response_feature_matrix_path=response_path,
        output_dir=tmp_path,
    )

    assert result["review_gate"]["num_checks_passed"] >= 10
    assert result["ranking"]["formulas"]["D_response_position_output_only"][
        "same_question_on_path_vs_off_path_auc"
    ] == 1.0
    for filename in [
        "answer_position_score_matrix.jsonl",
        "answer_position_feature_audit.json",
        "answer_position_ranking_report.json",
        "answer_position_formula_validation_report.json",
        "answer_position_grouped_bootstrap_report.json",
        "prompt_vs_answer_position_comparison.json",
        "span_type_answer_position_breakdown.json",
        "off_path_budget_report.json",
        "failure_case_report.jsonl",
        "success_case_report.jsonl",
        "review_gate_answer_position_output_effect.md",
    ]:
        assert (tmp_path / filename).exists()
