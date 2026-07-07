from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import (  # noqa: E402
    audit_formula_input_feature_names,
    build_multi_span_matrix,
    extract_multi_span_candidates,
    run_2ja_matrix,
)
from recover_attention.multi_span_reasoning_scoring import (  # noqa: E402
    audit_formula_inputs,
    build_score_reports,
    token_indices_for_char_ranges,
)


def sample_questions() -> list[dict]:
    return [
        {
            "source_question_id": "gsm8k_train_test_001",
            "question": "Alice bought 3 apples for 2 dollars each and Bob gave her 5 more apples. How many apples does Alice have?",
            "gold_answer": "8",
            "gold_solution": "Alice has 3+5=<<3+5=8>>8 apples\n#### 8",
            "provenance": ["test"],
        },
        {
            "source_question_id": "gsm8k_train_test_002",
            "question": "Sam had 12 tickets, sold 4 tickets, and kept 2 tickets for later. How many tickets are left?",
            "gold_answer": "8",
            "gold_solution": "Sam sold 4 so 12-4=<<12-4=8>>8 tickets\n#### 8",
            "provenance": ["test"],
        },
    ]


def test_extract_multi_span_candidates_covers_core_families() -> None:
    question = sample_questions()[0]["question"]

    spans = extract_multi_span_candidates(question)
    types = {span["span_type"] for span in spans}
    texts = {span["span_text"] for span in spans}

    assert len(spans) >= 3
    assert "number" in types
    assert "operation" in types
    assert "question_target" in types
    assert "object" in types
    assert "3" in texts
    assert any(span["parent_span_id"] for span in spans)


def test_build_multi_span_matrix_keeps_solution_path_eval_only() -> None:
    grouped, flat, audit = build_multi_span_matrix(sample_questions())

    assert audit["char_alignment_failures"] == 0
    assert len(grouped) == 2
    assert len(flat) >= 6
    first = flat[0]
    assert "source_question_id" in first
    assert "masked_question" in first
    assert "[MASK]" in first["masked_question"]
    assert "solution_path_status" in first["diagnostic_labels_for_eval_only"]
    assert all("solution_path" not in name for name in first["surface_features"])
    assert all("target" not in name for name in first["surface_features"])


def test_formula_feature_leakage_audit_flags_surface_feature_name() -> None:
    rows = [
        {
            "surface_features": {
                "surface_keyness_proxy": 0.1,
                "gold_answer_hint": 1.0,
            }
        }
    ]

    audit = audit_formula_input_feature_names(rows)

    assert audit["passed"] is False
    assert audit["leaked_input_features"][0]["feature_name"] == "gold_answer_hint"


def test_run_2ja_matrix_writes_outputs(tmp_path: Path) -> None:
    subset = []
    raw = []
    for row in sample_questions():
        subset.append(
            {
                "source_question_id": row["source_question_id"],
                "question": row["question"],
                "gold_solution": row["gold_solution"],
                "span_text": "bought",
                "span_type": "operation",
            }
        )
        raw.append(
            {
                "question_id": row["source_question_id"],
                "question": row["question"],
                "answer": row["gold_answer"],
                "metadata": {"original_answer": row["gold_solution"]},
            }
        )

    subset_path = tmp_path / "subset_cases.jsonl"
    raw_path = tmp_path / "gsm8k_train_normalized.jsonl"
    risk_path = tmp_path / "risk_strength_dataset.jsonl"
    write_jsonl(subset, subset_path)
    write_jsonl(raw, raw_path)
    write_jsonl([], risk_path)

    output_dir = tmp_path / "out"
    result = run_2ja_matrix(
        subset_cases_path=subset_path,
        raw_gsm8k_path=raw_path,
        risk_strength_path=risk_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert result["num_questions"] == 2
    assert result["num_candidate_spans"] >= 6
    for filename in [
        "multi_span_grouped_matrix.jsonl",
        "multi_span_flat_matrix.jsonl",
        "candidate_span_coverage_report.json",
        "span_extraction_audit.json",
        "keyness_label_report.json",
        "surface_ranking_baseline_report.json",
        "review_gate_multi_span_matrix.md",
    ]:
        assert (output_dir / filename).exists()

    flat = read_jsonl(output_dir / "multi_span_flat_matrix.jsonl")
    assert all("source_question_id" in row for row in flat)
    coverage = json.loads((output_dir / "candidate_span_coverage_report.json").read_text(encoding="utf-8"))
    assert "gate" in coverage


def test_2jb_token_indices_for_char_ranges() -> None:
    offsets = [[0, 0], [0, 5], [6, 12], [13, 18]]

    assert token_indices_for_char_ranges(offsets, [[1, 4], [13, 18]], exclude=set()) == [1, 3]
    assert token_indices_for_char_ranges(offsets, [[1, 4], [13, 18]], exclude={1}) == [3]


def test_2jb_formula_input_audit_rejects_forbidden_names() -> None:
    assert audit_formula_inputs(formula_input_names=["hidden_fragility_score"])["passed"] is True

    try:
        audit_formula_inputs(formula_input_names=["gold_answer_hint"])
    except AssertionError as exc:
        assert "formula input leakage detected" in str(exc)
    else:
        raise AssertionError("expected leakage audit to fail")


def test_2jb_score_reports_write_required_outputs(tmp_path: Path) -> None:
    records = []
    for row in sample_questions():
        base = {
            "backend": "test",
            "source_question_id": row["source_question_id"],
            "question": row["question"],
            "masked_question": row["question"].replace("3", "[MASK]", 1),
            "mask_text": "[MASK]",
            "alignment": {
                "alignment_status": "ok",
                "num_original_slot_tokens": 1,
                "num_masked_slot_tokens": 1,
                "warnings": [],
            },
        }
        records.append(
            {
                **base,
                "span_id": f"{row['source_question_id']}__span_on",
                "span_text": "3",
                "span_type": "number",
                "span_char_start": 13,
                "span_char_end": 14,
                "surface_features": {"surface_keyness_proxy": 0.6},
                "hidden_features": {
                    "hidden_delta_relative_norm": 0.9,
                    "question_context_shift_norm": 0.7,
                    "early_mid_late_delta_slope": 0.1,
                    "span_to_mask_similarity": 0.1,
                },
                "attention_features": {
                    "original_to_masked_attention_delta": 0.8,
                    "attention_entropy_delta": 0.3,
                    "attention_rank_delta": 0.2,
                    "span_attention_in_mass": 0.7,
                    "context_to_slot_attention": 0.6,
                },
                "diagnostic_labels_for_eval_only": {
                    "solution_path_status": "on_solution_path_number",
                    "weak_semantic_keyness": None,
                },
            }
        )
        records.append(
            {
                **base,
                "span_id": f"{row['source_question_id']}__span_off",
                "span_text": "2",
                "span_type": "number",
                "span_char_start": 28,
                "span_char_end": 29,
                "surface_features": {"surface_keyness_proxy": 0.4},
                "hidden_features": {
                    "hidden_delta_relative_norm": 0.2,
                    "question_context_shift_norm": 0.1,
                    "early_mid_late_delta_slope": 0.0,
                    "span_to_mask_similarity": 0.8,
                },
                "attention_features": {
                    "original_to_masked_attention_delta": 0.1,
                    "attention_entropy_delta": 0.1,
                    "attention_rank_delta": 0.0,
                    "span_attention_in_mass": 0.2,
                    "context_to_slot_attention": 0.1,
                },
                "diagnostic_labels_for_eval_only": {
                    "solution_path_status": "off_solution_path_number",
                    "weak_semantic_keyness": None,
                },
            }
        )

    artifacts = build_score_reports(records, output_dir=tmp_path)

    assert artifacts["ranking_report"]["formulas"]["E_keyness_times_fragility"][
        "same_question_on_path_vs_off_path_auc"
    ] == 1.0
    assert artifacts["formula_report"]["formula_input_leakage_audit"]["passed"] is True
    for filename in [
        "multi_span_score_matrix.jsonl",
        "hidden_attention_feature_report.json",
        "same_question_ranking_report.json",
        "formula_validation_report.json",
        "topk_budget_report.json",
        "failure_case_report.jsonl",
        "success_case_report.jsonl",
        "reasoning_signal_gap_report.json",
        "review_gate_multi_span_scoring.md",
    ]:
        assert (tmp_path / filename).exists()
