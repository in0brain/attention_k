from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.guidance_candidates import (  # noqa: E402
    BACKEND,
    FORBIDDEN_OUTPUT_FILENAMES,
    build_guidance_candidate_manifest,
    candidate_action_from_prediction,
    compute_score_margin,
    confidence_level_from_margin,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "20_build_guidance_candidate_manifest.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("guidance_candidate_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_no_nonfinite_numbers(value) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, list):
        for item in value:
            assert_no_nonfinite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_nonfinite_numbers(item)


def prediction_record(
    index: int,
    predicted: str,
    *,
    gold: str | None = None,
    scores: dict | None = None,
) -> dict:
    gold = gold or predicted
    if scores is None:
        scores = {
            "risk_positive": -0.2,
            "positive_anchor": -0.1,
            "negative": -0.3,
            "hard_negative_or_weak_positive": -0.4,
            predicted: 0.6,
        }
    return {
        "probe_record_id": f"probe_{index:03d}",
        "feature_id": f"case_{index:03d}::recovered::0",
        "masked_id": f"case_{index:03d}",
        "id": f"case_{index:03d}",
        "unit_id": "unit_001",
        "cv_strategy": "leave_one_out",
        "fold_id": index,
        "train_size": 19,
        "test_size": 1,
        "gold_probe_target": gold,
        "predicted_probe_target": predicted,
        "correct": gold == predicted,
        "decision_scores": scores,
        "gold_anchor_or_risk_binary": gold in {"risk_positive", "positive_anchor"},
        "predicted_anchor_or_risk_binary": predicted in {"risk_positive", "positive_anchor"},
        "binary_correct": (gold in {"risk_positive", "positive_anchor"})
        == (predicted in {"risk_positive", "positive_anchor"}),
        "num_features": 10,
        "num_null_features": 0,
        "hidden_state_path": "forbidden_hidden_state_that_must_not_be_read.pt",
        "warnings": [],
    }


def all_prediction_records() -> list[dict]:
    return [
        prediction_record(
            0,
            "risk_positive",
            scores={
                "risk_positive": 0.9,
                "positive_anchor": 0.1,
                "negative": -0.2,
                "hard_negative_or_weak_positive": -0.3,
            },
        ),
        prediction_record(
            1,
            "positive_anchor",
            gold="negative",
            scores={
                "risk_positive": 0.2,
                "positive_anchor": 0.36,
                "negative": 0.25,
                "hard_negative_or_weak_positive": 0.0,
            },
        ),
        prediction_record(
            2,
            "hard_negative_or_weak_positive",
            scores={
                "risk_positive": 0.10,
                "positive_anchor": 0.05,
                "negative": 0.02,
                "hard_negative_or_weak_positive": 0.08,
            },
        ),
        prediction_record(
            3,
            "negative",
            scores=None,
        ),
    ]


def write_source_files(tmp_path: Path, records: list[dict]) -> dict[str, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    predictions_path = input_dir / "probe_predictions.jsonl"
    eval_report_path = input_dir / "probe_eval_report.json"
    write_jsonl(records, predictions_path)
    write_json(
        eval_report_path,
        {
            "sprint": "2D",
            "backend": "probe_training_baseline_v0",
            "status": "ok",
            "training": {
                "model_type": "ridge_classifier_ovr_v0",
                "cv_strategy": "leave_one_out",
                "num_folds": len(records),
            },
            "metrics": {"accuracy": 0.5, "macro_f1": 0.4},
            "outputs": {
                "probe_model_path": str(input_dir / "probe_model.pkl"),
            },
        },
    )
    (input_dir / "probe_model.pkl").write_text("not a pickle and must not be loaded\n", encoding="utf-8")
    return {"predictions": predictions_path, "eval_report": eval_report_path}


def test_candidate_action_mapping() -> None:
    assert (
        candidate_action_from_prediction("risk_positive")["candidate_action"]
        == "increase_attention_to_original_span"
    )
    assert (
        candidate_action_from_prediction("positive_anchor")["candidate_action"]
        == "preserve_original_span_attention"
    )
    assert (
        candidate_action_from_prediction("hard_negative_or_weak_positive")["candidate_action"]
        == "review_before_guidance"
    )
    assert candidate_action_from_prediction("negative")["candidate_action"] == "no_guidance"
    assert candidate_action_from_prediction("negative")["guidance_candidate"] is False
    assert candidate_action_from_prediction("unknown")["guidance_candidate"] is False


def test_score_margin_and_confidence_levels() -> None:
    scores = {"risk_positive": 0.7, "negative": 0.1, "positive_anchor": -0.1}
    summary = compute_score_margin(scores, "risk_positive")

    assert summary["probe_predicted_score"] == 0.7
    assert summary["probe_second_best_score"] == 0.1
    assert summary["probe_score_margin"] == pytest.approx(0.6)
    assert summary["probe_confidence_level"] == "high"
    assert confidence_level_from_margin(0.3) == "medium"
    assert confidence_level_from_margin(0.05) == "low"
    assert compute_score_margin(None, "risk_positive")["probe_confidence_level"] == "unknown"


def test_build_guidance_manifest_uses_predicted_target_not_gold(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, all_prediction_records())
    output_dir = tmp_path / "guidance"

    result = build_guidance_candidate_manifest(
        predictions_path=paths["predictions"],
        eval_report_path=paths["eval_report"],
        output_dir=output_dir,
        backend=BACKEND,
        overwrite=True,
    )
    records = result["manifest_records"]
    report = result["report"]

    assert len(records) == 4
    assert len(read_jsonl(output_dir / "guidance_candidate_manifest.jsonl")) == 4
    assert (output_dir / "guidance_candidate_report.json").exists()
    assert not any((output_dir / filename).exists() for filename in FORBIDDEN_OUTPUT_FILENAMES)
    assert records[0]["candidate_action"] == "increase_attention_to_original_span"
    assert records[1]["gold_probe_target"] == "negative"
    assert records[1]["predicted_probe_target"] == "positive_anchor"
    assert records[1]["candidate_action"] == "preserve_original_span_attention"
    assert records[2]["candidate_action"] == "review_before_guidance"
    assert records[3]["candidate_action"] == "no_guidance"
    assert records[3]["guidance_candidate"] is False
    assert all(record["execution_status"] == "planned_only" for record in records)
    assert all(record["dry_run"] is True for record in records)
    assert all(record["will_modify_attention"] is False for record in records)
    assert all(record["will_run_model"] is False for record in records)
    assert all(record["will_change_answer"] is False for record in records)
    assert report["counts"]["num_guidance_candidate_records"] == 4
    assert report["counts"]["num_guidance_candidate_true"] == 3
    assert report["counts"]["num_guidance_candidate_false"] == 1
    assert report["candidate_action_counts"]["increase_attention_to_original_span"] == 1
    assert report["candidate_action_counts"]["preserve_original_span_attention"] == 1
    assert report["candidate_action_counts"]["review_before_guidance"] == 1
    assert report["candidate_action_counts"]["no_guidance"] == 1
    assert report["predicted_target_counts"]["risk_positive"] == 1
    assert report["confidence_counts"]["high"] == 2
    assert report["confidence_counts"]["medium"] == 1
    assert report["confidence_counts"]["low"] == 1
    assert report["confidence_counts"]["unknown"] == 0
    assert report["candidate_correctness_by_gold"]["num_prediction_correct"] == 3
    assert report["boundary"]["dry_run"] is True
    assert report["boundary"]["loaded_probe_model"] is False
    assert report["boundary"]["retrained_probe"] is False
    assert report["boundary"]["rebuilt_probe_dataset"] is False
    assert report["boundary"]["read_hidden_state_tensors"] is False
    assert report["boundary"]["recomputed_representation_features"] is False
    assert report["boundary"]["modified_attention_weights"] is False
    assert report["boundary"]["ran_attention_steering"] is False
    assert report["boundary"]["evaluated_answer_accuracy"] is False
    assert report["boundary"]["claimed_hallucination_reduction"] is False
    assert_no_nonfinite_numbers(report)
    for record in records:
        assert_no_nonfinite_numbers(record)


def test_unknown_confidence_when_decision_scores_missing_predicted_target(tmp_path: Path) -> None:
    record = prediction_record(
        0,
        "risk_positive",
        scores={"negative": 0.4, "positive_anchor": 0.2},
    )
    paths = write_source_files(tmp_path, [record])

    result = build_guidance_candidate_manifest(
        predictions_path=paths["predictions"],
        eval_report_path=paths["eval_report"],
        output_dir=tmp_path / "guidance",
        backend=BACKEND,
        overwrite=True,
    )

    output_record = result["manifest_records"][0]
    assert output_record["probe_score_margin"] is None
    assert output_record["probe_confidence_level"] == "unknown"
    assert output_record["warnings"]
    assert result["report"]["confidence_counts"]["unknown"] == 1


def test_rejects_non_ok_eval_report(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, all_prediction_records())
    write_json(paths["eval_report"], {"status": "insufficient_data"})

    with pytest.raises(ValueError, match="status must be ok"):
        build_guidance_candidate_manifest(
            predictions_path=paths["predictions"],
            eval_report_path=paths["eval_report"],
            output_dir=tmp_path / "guidance",
            backend=BACKEND,
            overwrite=True,
        )


def test_default_no_overwrite_and_output_dir_guard(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, all_prediction_records())
    output_dir = tmp_path / "guidance"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="--overwrite"):
        build_guidance_candidate_manifest(
            predictions_path=paths["predictions"],
            eval_report_path=paths["eval_report"],
            output_dir=output_dir,
            backend=BACKEND,
        )

    with pytest.raises(ValueError, match="forbidden path"):
        build_guidance_candidate_manifest(
            predictions_path=paths["predictions"],
            eval_report_path=paths["eval_report"],
            output_dir=Path.cwd() / "outputs/logs/sprint_2D_probe_training_baseline",
            backend=BACKEND,
            overwrite=True,
        )


def test_cli_smoke_generates_formal_outputs_only(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, all_prediction_records())
    output_dir = tmp_path / "cli_guidance"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--predictions",
            str(paths["predictions"]),
            "--eval-report",
            str(paths["eval_report"]),
            "--output-dir",
            str(output_dir),
            "--backend",
            BACKEND,
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Built Sprint 2E guidance candidate dry run" in result.stdout
    assert (output_dir / "guidance_candidate_manifest.jsonl").exists()
    assert (output_dir / "guidance_candidate_report.json").exists()
    assert not any((output_dir / filename).exists() for filename in FORBIDDEN_OUTPUT_FILENAMES)


def test_cli_has_no_model_training_or_steering_arguments() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND])

    assert args.backend == BACKEND
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "device")
    assert not hasattr(args, "device_map")
    assert not hasattr(args, "dataset")
    assert not hasattr(args, "features")
    assert not hasattr(args, "hidden_state_dir")
    assert not hasattr(args, "train")
    assert not hasattr(args, "steer")
