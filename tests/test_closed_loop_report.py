from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.closed_loop_report import (  # noqa: E402
    AUDIT_FILENAME,
    BACKEND,
    REPORT_FILENAME,
    build_sprint_2_closed_loop_report,
)
from recover_attention.data_io import write_jsonl  # noqa: E402


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "21_write_sprint_2_closed_loop_report.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("closed_loop_report_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_test_artifacts(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "artifacts"
    paths = {
        "hidden_cache_report": root / "2A_real" / "hidden_state_cache_report.json",
        "alignment_report": root / "2A_real" / "token_alignment_report.json",
        "real_run_metadata": root / "2A_real" / "real_run_metadata.json",
        "representation_features": root / "2B" / "representation_features.jsonl",
        "representation_feature_report": root / "2B" / "representation_feature_report.json",
        "probe_dataset": root / "2C" / "probe_dataset.jsonl",
        "probe_dataset_report": root / "2C" / "probe_dataset_report.json",
        "probe_predictions": root / "2D" / "probe_predictions.jsonl",
        "probe_eval_report": root / "2D" / "probe_eval_report.json",
        "probe_model": root / "2D" / "probe_model.pkl",
        "guidance_candidate_manifest": root / "2E" / "guidance_candidate_manifest.jsonl",
        "guidance_candidate_report": root / "2E" / "guidance_candidate_report.json",
    }
    write_json(
        paths["hidden_cache_report"],
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "status": "ok",
            "output_dir": "outputs/logs/sprint_2A_real_hidden_state_cache",
            "num_cases": 20,
            "num_inputs_total": 60,
            "num_hidden_state_files": 60,
            "layer_indices": [0, 8, 16, 24, 27],
            "failure_count": 0,
        },
    )
    write_json(
        paths["alignment_report"],
        {
            "num_cases": 20,
            "num_single_mask_cases": 17,
            "num_group_mask_cases": 3,
            "num_fragment_recovery_outputs": 8,
            "alignment_warning_count": 8,
        },
    )
    write_json(
        paths["real_run_metadata"],
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "num_cases": 20,
            "num_inputs_total": 60,
            "requested_layer_indices": [0, 8, 16, 24, 27],
            "resolved_layer_indices": [0, 8, 16, 24, 27],
        },
    )
    write_jsonl(
        [
            {"feature_id": "feature_1", "hidden_state_path": "must_not_read.pt"},
            {"feature_id": "feature_2", "hidden_state_path": "must_not_read.pt"},
        ],
        paths["representation_features"],
    )
    write_json(
        paths["representation_feature_report"],
        {
            "sprint": "2B-fix",
            "backend": "representation_features_minimal_v0",
            "status": "ok",
            "counts": {
                "num_feature_records": 2,
                "num_masked_groups": 2,
                "num_recovered_variants": 2,
                "num_skipped_groups": 0,
                "num_skipped_recovered_variants": 0,
            },
            "warning_counts": {
                "missing_span_overlap": 1,
                "missing_mask_position_overlap": 1,
            },
        },
    )
    write_jsonl(
        [
            {"probe_record_id": "probe_1", "probe_target": "risk_positive"},
            {"probe_record_id": "probe_2", "probe_target": "negative"},
        ],
        paths["probe_dataset"],
    )
    write_json(
        paths["probe_dataset_report"],
        {
            "backend": "probe_dataset_mapping_v0",
            "status": "ok",
            "counts": {
                "num_probe_dataset_records": 2,
                "num_probe_target_usable": 2,
                "num_unmapped_records": 0,
            },
            "target_counts": {
                "risk_positive": 1,
                "negative": 1,
                "positive_anchor": 0,
                "hard_negative_or_weak_positive": 0,
            },
            "null_feature_counts": {"records_with_null_position_features": 1},
        },
    )
    write_jsonl(
        [
            {"probe_record_id": "probe_1", "predicted_probe_target": "risk_positive"},
            {"probe_record_id": "probe_2", "predicted_probe_target": "negative"},
        ],
        paths["probe_predictions"],
    )
    write_json(
        paths["probe_eval_report"],
        {
            "backend": "probe_training_baseline_v0",
            "status": "ok",
            "data_summary": {
                "num_records_usable": 2,
                "target_counts": {"risk_positive": 1, "negative": 1},
            },
            "training": {
                "model_type": "ridge_classifier_ovr_v0",
                "cv_strategy": "leave_one_out",
                "num_folds": 2,
            },
            "metrics": {"accuracy": 0.5, "macro_f1": 0.333333, "weighted_f1": 0.5},
            "binary_anchor_or_risk_metrics": {"accuracy": 0.5, "macro_f1": 0.333333},
            "baselines": {"majority_class": {"accuracy": 0.5, "macro_f1": 0.333333}},
            "feature_signal_summary": {
                "top_weighted_features": [{"feature_name": "question_original_masked_cosine_layer_8"}],
                "feature_group_weight_summary": {"question": 0.2, "span": 0.1},
                "layer_weight_summary": {"layer_8": 0.2, "summary_scalar": 0.1},
            },
        },
    )
    paths["probe_model"].parent.mkdir(parents=True, exist_ok=True)
    paths["probe_model"].write_text("not a pickle and must not be loaded\n", encoding="utf-8")
    write_jsonl(
        [
            {"guidance_candidate_id": "candidate_1", "candidate_action": "increase_attention_to_original_span"},
            {"guidance_candidate_id": "candidate_2", "candidate_action": "no_guidance"},
        ],
        paths["guidance_candidate_manifest"],
    )
    write_json(
        paths["guidance_candidate_report"],
        {
            "backend": "guidance_candidate_dry_run_v0",
            "status": "ok",
            "counts": {
                "num_guidance_candidate_records": 2,
                "num_guidance_candidate_true": 1,
                "num_guidance_candidate_false": 1,
            },
            "predicted_target_counts": {"risk_positive": 1, "negative": 1},
            "candidate_action_counts": {
                "increase_attention_to_original_span": 1,
                "no_guidance": 1,
            },
            "confidence_counts": {"high": 1, "medium": 0, "low": 1, "unknown": 0},
            "boundary": {
                "dry_run": True,
                "modified_attention_weights": False,
                "ran_attention_steering": False,
                "claimed_hallucination_reduction": False,
            },
        },
    )
    return paths


def test_build_report_contains_required_sections_and_non_claims(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    result = build_sprint_2_closed_loop_report(
        output_dir=tmp_path / "report",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
        today="2026-06-30",
    )
    report = result["report"]
    audit = result["audit"]

    assert (tmp_path / "report" / REPORT_FILENAME).exists()
    assert (tmp_path / "report" / AUDIT_FILENAME).exists()
    assert "# Sprint 2 Minimal Closed-loop Report" in report
    assert "## Executive Summary" in report
    assert "## Sprint 2 Pipeline Overview" in report
    for index in range(1, 7):
        assert f"## Question {index}:" in report
    assert "## Dry-run Boundary and Non-claims" in report
    assert "Sprint 2E only generated planned-only guidance candidates." in report
    assert "Sprint 2 did not execute attention steering." in report
    assert "Sprint 2 did not validate hallucination reduction." in report
    assert "## Windows Execution Stability Note" in report
    assert "engineering stability issue" in report
    assert "Execute targeted pytest, pipeline command, and full pytest serially." in report
    assert "## Sprint 3 Readiness" in report
    assert "Sprint 3A: Attention Steering Interface Design" in report
    assert "## Workspace State Note" in report
    assert "Pre-existing AM task card state was observed" in report
    assert "No upstream pipeline scripts were rerun" in report
    assert audit["loop_status"]["executed_attention_steering"] is False
    assert audit["loop_status"]["validated_hallucination_reduction"] is False
    assert audit["required_boundary_statements_present"] is True
    assert audit["windows_serial_execution_note_present"] is True


def test_report_does_not_contain_forbidden_positive_claims(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    result = build_sprint_2_closed_loop_report(
        output_dir=tmp_path / "report",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )
    text = result["report"].lower()

    assert "attention guidance succeeded" not in text
    assert "attention steering improved reasoning" not in text
    assert "hallucination was reduced" not in text
    assert "answer accuracy improved" not in text
    assert "closed-loop intervention was validated" not in text


def test_builder_does_not_read_hidden_tensors_or_load_probe_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_test_artifacts(tmp_path)
    hidden_tensor = tmp_path / "artifacts" / "2A_real" / "hidden_states" / "blocked.pt"
    hidden_tensor.parent.mkdir(parents=True, exist_ok=True)
    hidden_tensor.write_text("must not be opened\n", encoding="utf-8")
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self.suffix == ".pt" or self.name == "probe_model.pkl":
            raise AssertionError(f"forbidden file was opened: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    result = build_sprint_2_closed_loop_report(
        output_dir=tmp_path / "report",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )

    assert result["summary"]["stages"]["2D"]["probe_model_exists"] is True
    assert result["audit"]["loop_status"]["hidden_state_cache"] is True


def test_missing_upstream_artifact_generates_best_effort_report(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)
    paths["guidance_candidate_report"].unlink()

    result = build_sprint_2_closed_loop_report(
        output_dir=tmp_path / "report",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )

    assert result["summary"]["status"] == "incomplete_evidence"
    assert str(paths["guidance_candidate_report"]) in result["summary"]["missing_upstream_artifacts"]
    assert "## Missing Upstream Artifacts" in result["report"]
    assert result["audit"]["loop_status"]["guidance_candidate_dry_run"] is False


def test_rejects_forbidden_upstream_output_dir(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    with pytest.raises(ValueError, match="forbidden upstream path"):
        build_sprint_2_closed_loop_report(
            output_dir=Path.cwd() / "outputs/logs/sprint_2E_guidance_candidate_dry_run",
            backend=BACKEND,
            overwrite=True,
            input_paths=paths,
        )


def test_cli_smoke_generates_report_and_audit(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)
    output_dir = tmp_path / "cli_report"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--hidden-cache-report",
            str(paths["hidden_cache_report"]),
            "--alignment-report",
            str(paths["alignment_report"]),
            "--real-run-metadata",
            str(paths["real_run_metadata"]),
            "--representation-features",
            str(paths["representation_features"]),
            "--representation-feature-report",
            str(paths["representation_feature_report"]),
            "--probe-dataset",
            str(paths["probe_dataset"]),
            "--probe-dataset-report",
            str(paths["probe_dataset_report"]),
            "--probe-predictions",
            str(paths["probe_predictions"]),
            "--probe-eval-report",
            str(paths["probe_eval_report"]),
            "--guidance-candidate-manifest",
            str(paths["guidance_candidate_manifest"]),
            "--guidance-candidate-report",
            str(paths["guidance_candidate_report"]),
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

    assert "[OK] Wrote Sprint 2 minimal closed-loop report" in result.stdout
    assert (output_dir / REPORT_FILENAME).exists()
    assert (output_dir / AUDIT_FILENAME).exists()


def test_cli_has_no_model_training_generation_or_steering_arguments() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND])

    assert args.backend == BACKEND
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "device")
    assert not hasattr(args, "train")
    assert not hasattr(args, "steer")
    assert not hasattr(args, "run_generation")
    assert not hasattr(args, "evaluate_accuracy")
