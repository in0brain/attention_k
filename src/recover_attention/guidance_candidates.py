"""Sprint 2E guidance candidate manifest dry run."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "guidance_candidate_dry_run_v0"
MANIFEST_FILENAME = "guidance_candidate_manifest.jsonl"
REPORT_FILENAME = "guidance_candidate_report.json"

ACTION_BY_TARGET = {
    "risk_positive": {
        "guidance_candidate": True,
        "candidate_action": "increase_attention_to_original_span",
        "candidate_reason": (
            "probe predicted high-risk recovery drift; original span should be emphasized "
            "in future attention guidance"
        ),
    },
    "positive_anchor": {
        "guidance_candidate": True,
        "candidate_action": "preserve_original_span_attention",
        "candidate_reason": (
            "probe predicted useful anchor; original span should be preserved as candidate "
            "attention anchor"
        ),
    },
    "hard_negative_or_weak_positive": {
        "guidance_candidate": True,
        "candidate_action": "review_before_guidance",
        "candidate_reason": (
            "probe predicted ambiguous weak signal; requires review before real guidance"
        ),
    },
    "negative": {
        "guidance_candidate": False,
        "candidate_action": "no_guidance",
        "candidate_reason": "probe predicted negative; no guidance candidate proposed",
    },
}
CONFIDENCE_LEVELS = ["high", "medium", "low", "unknown"]
FORBIDDEN_OUTPUT_FILENAMES = [
    "attention_steering_results.jsonl",
    "model_outputs_after_guidance.jsonl",
    "answer_accuracy_report.json",
    "hallucination_reduction_report.json",
    "closed_loop_final_report.md",
]


def build_guidance_candidate_manifest(
    *,
    predictions_path: str | Path,
    eval_report_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a planned-only guidance candidate manifest from probe predictions."""

    if backend != BACKEND:
        raise ValueError(f"Unsupported guidance candidate backend {backend!r}; expected {BACKEND!r}")

    predictions_path = Path(predictions_path)
    eval_report_path = Path(eval_report_path)
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite)

    predictions = load_probe_predictions(predictions_path)
    eval_report = load_probe_eval_report(eval_report_path)
    output_files = {
        "guidance_candidate_manifest": output_dir / MANIFEST_FILENAME,
        "guidance_candidate_report": output_dir / REPORT_FILENAME,
    }

    manifest_records = [
        build_guidance_candidate_record(
            prediction,
            source_prediction_path=predictions_path,
            source_eval_report_path=eval_report_path,
            backend=backend,
            source_line_number=line_number,
        )
        for line_number, prediction in enumerate(predictions, start=1)
    ]
    report = build_guidance_candidate_report(
        predictions=predictions,
        manifest_records=manifest_records,
        eval_report=eval_report,
        predictions_path=predictions_path,
        eval_report_path=eval_report_path,
        output_files=output_files,
        backend=backend,
    )

    write_jsonl_strict(manifest_records, output_files["guidance_candidate_manifest"])
    write_json_strict(report, output_files["guidance_candidate_report"])

    return {
        "manifest_records": manifest_records,
        "report": report,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }


def load_probe_predictions(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"probe predictions input is empty: {path}")
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"prediction line {line_number} must be a JSON object")
        for field in [
            "probe_record_id",
            "predicted_probe_target",
            "gold_probe_target",
            "correct",
        ]:
            if field not in record:
                raise ValueError(f"prediction line {line_number} missing required field: {field}")
    return records


def load_probe_eval_report(path: str | Path) -> dict[str, Any]:
    report = load_json(path)
    if report.get("status") != "ok":
        raise ValueError(f"probe eval report status must be ok for Sprint 2E: {path}")
    return report


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON input: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must be an object: {json_path}")
    return data


def candidate_action_from_prediction(predicted_probe_target: str) -> dict[str, Any]:
    action = ACTION_BY_TARGET.get(predicted_probe_target)
    if action is None:
        return {
            "guidance_candidate": False,
            "candidate_action": "no_guidance",
            "candidate_reason": (
                f"unsupported predicted_probe_target {predicted_probe_target!r}; "
                "no guidance candidate proposed"
            ),
            "execution_status": "planned_only",
        }
    result = dict(action)
    result["execution_status"] = "planned_only"
    return result


def compute_score_margin(
    decision_scores: Any,
    predicted_probe_target: str,
) -> dict[str, Any]:
    if not isinstance(decision_scores, dict) or predicted_probe_target not in decision_scores:
        return {
            "probe_predicted_score": None,
            "probe_second_best_score": None,
            "probe_score_margin": None,
            "probe_confidence_level": "unknown",
        }

    numeric_scores: dict[str, float] = {}
    for label, score in decision_scores.items():
        if is_finite_number(score):
            numeric_scores[str(label)] = float(score)
    if predicted_probe_target not in numeric_scores:
        return {
            "probe_predicted_score": None,
            "probe_second_best_score": None,
            "probe_score_margin": None,
            "probe_confidence_level": "unknown",
        }

    predicted_score = numeric_scores[predicted_probe_target]
    other_scores = [
        value for label, value in numeric_scores.items() if label != predicted_probe_target
    ]
    if not other_scores:
        return {
            "probe_predicted_score": sanitize_float(predicted_score),
            "probe_second_best_score": None,
            "probe_score_margin": None,
            "probe_confidence_level": "unknown",
        }
    second_best = max(other_scores)
    margin = predicted_score - second_best
    return {
        "probe_predicted_score": sanitize_float(predicted_score),
        "probe_second_best_score": sanitize_float(second_best),
        "probe_score_margin": sanitize_float(margin),
        "probe_confidence_level": confidence_level_from_margin(margin),
    }


def confidence_level_from_margin(margin: float | None) -> str:
    if margin is None or not math.isfinite(float(margin)):
        return "unknown"
    if margin >= 0.5:
        return "high"
    if margin >= 0.1:
        return "medium"
    return "low"


def build_guidance_candidate_record(
    prediction: dict[str, Any],
    *,
    source_prediction_path: str | Path,
    source_eval_report_path: str | Path,
    backend: str,
    source_line_number: int,
) -> dict[str, Any]:
    predicted_target = str(prediction.get("predicted_probe_target"))
    action = candidate_action_from_prediction(predicted_target)
    score_summary = compute_score_margin(prediction.get("decision_scores"), predicted_target)
    warnings = sorted(
        set(str(warning) for warning in prediction.get("warnings", []) if warning is not None)
    )
    if score_summary["probe_confidence_level"] == "unknown":
        warnings.append("decision_scores missing or incomplete; confidence level set to unknown")

    return sanitize_json_value(
        {
            "guidance_candidate_id": (
                f"{prediction.get('probe_record_id')}__guidance_candidate_{backend}"
            ),
            "probe_record_id": prediction.get("probe_record_id"),
            "feature_id": prediction.get("feature_id"),
            "masked_id": prediction.get("masked_id"),
            "id": prediction.get("id"),
            "unit_id": prediction.get("unit_id"),
            "source_prediction_path": str(source_prediction_path),
            "source_prediction_line": source_line_number,
            "source_eval_report_path": str(source_eval_report_path),
            "gold_probe_target": prediction.get("gold_probe_target"),
            "predicted_probe_target": predicted_target,
            "prediction_correct": bool(prediction.get("correct")),
            "gold_anchor_or_risk_binary": prediction.get("gold_anchor_or_risk_binary"),
            "predicted_anchor_or_risk_binary": prediction.get(
                "predicted_anchor_or_risk_binary"
            ),
            "binary_correct": prediction.get("binary_correct"),
            "decision_scores": prediction.get("decision_scores"),
            **score_summary,
            "guidance_candidate": action["guidance_candidate"],
            "candidate_action": action["candidate_action"],
            "candidate_reason": action["candidate_reason"],
            "execution_status": action["execution_status"],
            "dry_run": True,
            "will_modify_attention": False,
            "will_run_model": False,
            "will_change_answer": False,
            "warnings": warnings,
        }
    )


def build_guidance_candidate_report(
    *,
    predictions: list[dict[str, Any]],
    manifest_records: list[dict[str, Any]],
    eval_report: dict[str, Any],
    predictions_path: Path,
    eval_report_path: Path,
    output_files: dict[str, Path],
    backend: str,
) -> dict[str, Any]:
    candidate_true = sum(1 for record in manifest_records if record["guidance_candidate"])
    candidate_false = len(manifest_records) - candidate_true
    correct_count = sum(1 for prediction in predictions if bool(prediction.get("correct")))
    binary_correct_count = sum(
        1 for prediction in predictions if bool(prediction.get("binary_correct"))
    )
    return sanitize_json_value(
        {
            "sprint": "2E",
            "backend": backend,
            "status": "ok",
            "inputs": {
                "probe_predictions_path": str(predictions_path),
                "probe_eval_report_path": str(eval_report_path),
                "source_2D_backend": eval_report.get("backend"),
                "probe_model_loaded": False,
            },
            "outputs": {
                "guidance_candidate_manifest_path": str(
                    output_files["guidance_candidate_manifest"]
                ),
                "guidance_candidate_report_path": str(
                    output_files["guidance_candidate_report"]
                ),
                "forbidden_outputs_generated": [],
            },
            "source_2D": {
                "status": eval_report.get("status"),
                "model_type": eval_report.get("training", {}).get("model_type"),
                "cv_strategy": eval_report.get("training", {}).get("cv_strategy"),
                "num_folds": eval_report.get("training", {}).get("num_folds"),
                "metrics": eval_report.get("metrics", {}),
            },
            "counts": {
                "num_prediction_records": len(predictions),
                "num_guidance_candidate_records": len(manifest_records),
                "num_guidance_candidate_true": candidate_true,
                "num_guidance_candidate_false": candidate_false,
            },
            "predicted_target_counts": sorted_counter_dict(
                Counter(record["predicted_probe_target"] for record in manifest_records)
            ),
            "candidate_action_counts": sorted_counter_dict(
                Counter(record["candidate_action"] for record in manifest_records)
            ),
            "confidence_counts": {
                level: sum(
                    1
                    for record in manifest_records
                    if record.get("probe_confidence_level") == level
                )
                for level in CONFIDENCE_LEVELS
            },
            "candidate_correctness_by_gold": {
                "num_prediction_correct": correct_count,
                "num_prediction_incorrect": len(predictions) - correct_count,
                "num_binary_prediction_correct": binary_correct_count,
                "num_binary_prediction_incorrect": len(predictions) - binary_correct_count,
                "note": (
                    "Gold labels are used only for retrospective diagnostics, not for "
                    "candidate action assignment."
                ),
            },
            "boundary": boundary_report(),
            "warnings": [],
        }
    )


def boundary_report() -> dict[str, bool]:
    return {
        "dry_run": True,
        "loaded_probe_model": False,
        "retrained_probe": False,
        "rebuilt_probe_dataset": False,
        "read_hidden_state_tensors": False,
        "recomputed_representation_features": False,
        "called_tokenizer": False,
        "called_hf_model": False,
        "called_ollama": False,
        "called_nli_model": False,
        "called_recovery_model": False,
        "modified_attention_weights": False,
        "ran_attention_steering": False,
        "ran_cot_generation": False,
        "evaluated_answer_accuracy": False,
        "claimed_hallucination_reduction": False,
    }


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    if overwrite:
        for filename in [MANIFEST_FILENAME, REPORT_FILENAME]:
            path = output_dir / filename
            if path.exists():
                path.unlink()


def ensure_output_dir_allowed(output_dir: Path) -> None:
    project_root = Path.cwd().resolve()
    resolved = output_dir.resolve()
    forbidden_roots = [
        project_root / "data" / "processed",
        project_root / "outputs" / "logs" / "sprint_1Q_real_signal_quality_review",
        project_root / "outputs" / "logs" / "sprint_1R_human_review_consolidation",
        project_root / "outputs" / "logs" / "sprint_2A_hidden_state_cache_baseline",
        project_root / "outputs" / "logs" / "sprint_2A_real_hidden_state_cache",
        project_root / "outputs" / "logs" / "sprint_2B_representation_features",
        project_root / "outputs" / "logs" / "sprint_2C_probe_dataset",
        project_root / "outputs" / "logs" / "sprint_2D_probe_training_baseline",
    ]
    for forbidden_root in forbidden_roots:
        forbidden_resolved = forbidden_root.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"Refusing to write Sprint 2E outputs under forbidden path: {output_dir}")


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sanitize_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return sanitize_float(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    return value


def write_jsonl_strict(records: list[dict[str, Any]], path: str | Path) -> None:
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json_strict(data: dict[str, Any], path: str | Path) -> None:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}
