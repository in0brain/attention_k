from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.score_matrix_audit import (  # noqa: E402
    audit_input_feature_names,
    build_score_matrix,
    run_score_matrix_audit,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_base_records() -> list[dict]:
    rows = []
    statuses = [
        ("q1", "on_solution_path_number", "number", "12", 3, 0.9),
        ("q1", "off_solution_path_number", "number", "99", 1, 0.1),
        ("q1", "not_a_number", "operation", "sold", 2, 0.5),
        ("q2", "on_solution_path_number", "number", "7", 3, 0.8),
        ("q2", "ambiguous_number", "number", "3", 2, 0.4),
        ("q2", "not_a_number", "object", "pies", 1, 0.2),
    ]
    for idx, (qid, status, span_type, span, bucket, strength) in enumerate(statuses):
        rows.append(
            {
                "masked_id": f"m{idx}",
                "source_question_id": qid,
                "question": f"{qid} question?",
                "span_text": span,
                "span_type": span_type,
                "solution_path_status": status,
                "fragility_bucket": bucket,
                "risk_strength": strength,
                "pre_recovery_features": {
                    "pre_delta_question_relnorm_mean": 0.1 + idx * 0.1,
                    "pre_delta_span_relnorm_mean": 0.2 + idx * 0.1,
                },
                "attention_features": {
                    "attn_delta_slot_in_mass": 0.01 + idx * 0.01,
                    "attn_delta_slot_entropy": 0.02 + idx * 0.01,
                    "attn_orig_qfocus_to_slot": 0.03 + idx * 0.01,
                },
            }
        )
    return rows


def sample_ordinal_predictions(rows: list[dict]) -> list[dict]:
    return [
        {
            "masked_id": row["masked_id"],
            "source_question_id": row["source_question_id"],
            "gold_fragility_bucket": row["fragility_bucket"],
            "score_expected_bucket": float(row["fragility_bucket"]),
            "score_ordinal_threshold": float(row["fragility_bucket"]) + 0.1,
            "score_reg_calibrated": float(row["fragility_bucket"]),
            "score_reg_raw": float(row["fragility_bucket"]),
        }
        for row in rows
    ]


def sample_predictions(rows: list[dict]) -> list[dict]:
    return [
        {
            "masked_id": row["masked_id"],
            "source_question_id": row["source_question_id"],
            "pred_risk_strength": row["risk_strength"],
            "gold_fragility_bucket": row["fragility_bucket"],
        }
        for row in rows
    ]


def test_leakage_audit_rejects_forbidden_input_feature_names() -> None:
    audit = audit_input_feature_names(
        ["pre_delta_question_relnorm_mean", "gold_solution_path_label"]
    )

    assert audit["passed"] is False
    assert audit["leaked_input_features"][0]["feature_name"] == "gold_solution_path_label"


def test_score_matrix_keeps_eval_only_labels_out_of_reasoning_inputs() -> None:
    rows = sample_base_records()
    matrix = build_score_matrix(
        rows,
        ordinal_predictions=sample_ordinal_predictions(rows),
        enriched_predictions=sample_predictions(rows),
        ordinal_primary_method="expected_bucket",
    )

    first = matrix[0]
    assert first["keyness_signals"]["on_path_number"] is True
    assert first["reasoning_signals"]["answer_logprob_delta"] is None
    assert first["labels_for_evaluation_only"]["fragility_bucket"] == 3
    assert "risk_strength" not in first["budget_signals"]
    assert first["hidden_fragility_score"] is not None


def test_run_score_matrix_audit_writes_required_outputs(tmp_path: Path) -> None:
    rows = sample_base_records()
    input_dir = tmp_path / "inputs"
    risk = input_dir / "risk_strength_dataset.jsonl"
    pre = input_dir / "pre_recovery_feature_dataset.jsonl"
    ordinal_predictions = input_dir / "ordinal_predictions.jsonl"
    ordinal_report = input_dir / "ordinal_calibration_report.json"
    attention = input_dir / "attention_feature_dataset.jsonl"
    attention_report = input_dir / "attention_ordinal_calibration_report.json"
    enriched = input_dir / "fragility_probe_enriched_predictions.jsonl"
    hidden = input_dir / "fragility_probe_predictions.jsonl"

    write_jsonl(rows, risk)
    write_jsonl(rows, pre)
    write_jsonl(rows, attention)
    write_jsonl(sample_ordinal_predictions(rows), ordinal_predictions)
    write_jsonl(sample_predictions(rows), enriched)
    write_jsonl(sample_predictions(rows), hidden)
    write_json(ordinal_report, {"primary_method": "expected_bucket"})
    write_json(attention_report, {"primary_method": "expected_bucket"})

    output_dir = tmp_path / "audit"
    result = run_score_matrix_audit(
        risk_strength_path=risk,
        pre_recovery_feature_path=pre,
        ordinal_predictions_path=ordinal_predictions,
        ordinal_report_path=ordinal_report,
        attention_feature_path=attention,
        attention_report_path=attention_report,
        enriched_predictions_path=enriched,
        hidden_predictions_path=hidden,
        output_dir=output_dir,
        overwrite=True,
        bootstrap_samples=10,
    )

    assert result["num_score_matrix_records"] == len(rows)
    assert result["feature_audit_passed"] is True
    for filename in [
        "score_matrix_dataset.jsonl",
        "score_matrix_feature_audit.json",
        "keyness_eval_report.json",
        "fragility_eval_report.json",
        "budget_priority_eval_report.json",
        "topk_failure_cases.jsonl",
        "topk_success_cases.jsonl",
        "formula_simulation_report.json",
        "formula_bootstrap_report.json",
        "reasoning_signal_gap_analysis.json",
        "root_cause_decision_table.json",
        "review_gate_score_matrix_audit.md",
    ]:
        assert (output_dir / filename).exists()

    matrix = read_jsonl(output_dir / "score_matrix_dataset.jsonl")
    assert matrix[0]["formula_scores"]["oracle_diagnostic_only"] is not None
    root = json.loads((output_dir / "root_cause_decision_table.json").read_text(encoding="utf-8"))
    assert root["do_not_enter_sprint_3A"] is True
