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

from recover_attention.data_io import write_jsonl  # noqa: E402
from recover_attention.stage_summary import (  # noqa: E402
    AUDIT_FILENAME,
    BACKEND,
    FIGURE_FILENAMES,
    SUMMARY_FILENAME,
    write_stage_summary,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "22_write_sprint_2_stage_summary.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("stage_summary_cli", SCRIPT_PATH)
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
        "representation_feature_report": root / "2B" / "representation_feature_report.json",
        "representation_features": root / "2B" / "representation_features.jsonl",
        "probe_dataset_report": root / "2C" / "probe_dataset_report.json",
        "probe_dataset": root / "2C" / "probe_dataset.jsonl",
        "probe_eval_report": root / "2D" / "probe_eval_report.json",
        "probe_predictions": root / "2D" / "probe_predictions.jsonl",
        "probe_model": root / "2D" / "probe_model.pkl",
        "guidance_candidate_report": root / "2E" / "guidance_candidate_report.json",
        "guidance_candidate_manifest": root / "2E" / "guidance_candidate_manifest.jsonl",
        "closed_loop_report": root / "2F" / "sprint_2_minimal_closed_loop_report.md",
        "closed_loop_audit": root / "2F" / "sprint_2_minimal_closed_loop_audit.json",
    }
    write_json(
        paths["hidden_cache_report"],
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "status": "ok",
            "num_cases": 20,
            "num_inputs_total": 60,
            "num_hidden_state_files": 60,
            "layer_indices": [0, 8, 16, 24, 27],
        },
    )
    write_json(
        paths["alignment_report"],
        {
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
            "resolved_layer_indices": [0, 8, 16, 24, 27],
        },
    )
    write_json(
        paths["representation_feature_report"],
        {
            "backend": "representation_features_minimal_v0",
            "status": "ok",
            "counts": {
                "num_feature_records": 20,
                "num_masked_groups": 20,
                "num_recovered_variants": 20,
            },
            "warning_counts": {
                "missing_span_overlap": 8,
                "missing_mask_position_overlap": 8,
            },
        },
    )
    write_jsonl([{"feature_id": "feature_1", "hidden_state_path": "blocked.pt"}], paths["representation_features"])
    write_json(
        paths["probe_dataset_report"],
        {
            "backend": "probe_dataset_mapping_v0",
            "status": "ok",
            "counts": {
                "num_probe_dataset_records": 20,
                "num_probe_target_usable": 20,
            },
            "target_counts": {
                "risk_positive": 7,
                "positive_anchor": 3,
                "negative": 8,
                "hard_negative_or_weak_positive": 2,
            },
        },
    )
    write_jsonl([{"probe_record_id": "probe_1", "probe_target": "risk_positive"}], paths["probe_dataset"])
    write_json(
        paths["probe_eval_report"],
        {
            "backend": "probe_training_baseline_v0",
            "status": "ok",
            "training": {"cv_strategy": "leave_one_out", "num_folds": 20},
            "metrics": {
                "accuracy": 0.85,
                "macro_f1": 0.680952380952381,
                "weighted_f1": 0.8285714285714285,
            },
            "baselines": {
                "majority_class": {
                    "accuracy": 0.4,
                    "macro_f1": 0.14285714285714288,
                }
            },
        },
    )
    write_jsonl([{"probe_record_id": "probe_1", "predicted_probe_target": "risk_positive"}], paths["probe_predictions"])
    paths["probe_model"].parent.mkdir(parents=True, exist_ok=True)
    paths["probe_model"].write_text("not a pickle and must not be loaded\n", encoding="utf-8")
    write_json(
        paths["guidance_candidate_report"],
        {
            "backend": "guidance_candidate_dry_run_v0",
            "status": "ok",
            "counts": {
                "num_guidance_candidate_records": 20,
                "num_guidance_candidate_true": 13,
                "num_guidance_candidate_false": 7,
            },
            "candidate_action_counts": {
                "increase_attention_to_original_span": 8,
                "preserve_original_span_attention": 4,
                "review_before_guidance": 1,
                "no_guidance": 7,
            },
            "confidence_counts": {"high": 17, "medium": 3, "low": 0, "unknown": 0},
            "predicted_target_counts": {
                "risk_positive": 8,
                "positive_anchor": 4,
                "negative": 7,
                "hard_negative_or_weak_positive": 1,
            },
        },
    )
    write_jsonl([{"guidance_candidate_id": "candidate_1"}], paths["guidance_candidate_manifest"])
    paths["closed_loop_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["closed_loop_report"].write_text(
        "# Sprint 2 Minimal Closed-loop Report\n\nSprint 2 did not execute attention steering.\n",
        encoding="utf-8",
    )
    write_json(
        paths["closed_loop_audit"],
        {
            "status": "ok",
            "loop_status": {
                "hidden_state_cache": True,
                "representation_features": True,
                "probe_dataset": True,
                "probe_training": True,
                "guidance_candidate_dry_run": True,
                "executed_attention_steering": False,
                "validated_hallucination_reduction": False,
            },
            "required_boundary_statements_present": True,
            "windows_serial_execution_note_present": True,
        },
    )
    return paths


def test_stage_summary_generates_markdown_audit_and_all_png_figures(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    result = write_stage_summary(
        output_dir=tmp_path / "stage_summary",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
        full_pytest_passed=508,
        full_pytest_skipped=2,
        full_pytest_duration_seconds=7.59,
        today="2026-06-30",
    )
    markdown = result["markdown"]
    audit = result["audit"]
    output_dir = tmp_path / "stage_summary"

    assert (output_dir / SUMMARY_FILENAME).exists()
    assert (output_dir / AUDIT_FILENAME).exists()
    for filename in FIGURE_FILENAMES:
        figure = output_dir / "figures" / filename
        assert figure.exists()
        assert figure.read_bytes().startswith(b"\x89PNG")
        assert figure.stat().st_size > 1000
    assert "# Sprint 2 Stage Summary" in markdown
    assert "Sprint 2 completed a dry-run hidden-state-to-guidance-candidate loop." in markdown
    assert "This is not an executed attention-steering loop." in markdown
    assert "## Pipeline Summary" in markdown
    assert "## Key Numbers" in markdown
    assert "figures/sprint_2_pipeline_overview.png" in markdown
    assert "Attention guidance has not been executed." in markdown
    assert "Hallucination reduction has not been validated." in markdown
    assert "Windows temporary file lock" in markdown
    assert "Sprint 3A: Attention Steering Interface Design" in markdown
    assert "Full pytest passed: 508 passed, 2 skipped." in markdown
    assert "Full pytest duration: 7.59 seconds." in markdown
    assert audit["status"] == "ok"
    assert audit["full_pytest"]["passed"] == 508
    assert audit["full_pytest"]["skipped"] == 2
    assert audit["loop_status"]["executed_attention_steering"] is False
    assert audit["loop_status"]["validated_answer_accuracy_improvement"] is False
    assert audit["loop_status"]["validated_hallucination_reduction"] is False
    assert audit["boundary"]["read_hidden_state_tensors"] is False
    assert audit["boundary"]["loaded_probe_model"] is False
    assert audit["boundary"]["reran_upstream_pipeline"] is False


def test_stage_summary_does_not_contain_forbidden_positive_claims(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    result = write_stage_summary(
        output_dir=tmp_path / "stage_summary",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )
    text = result["markdown"].lower()

    assert "attention guidance succeeded" not in text
    assert "hallucination was reduced" not in text
    assert "answer accuracy improved" not in text
    assert "attention steering improved reasoning" not in text


def test_stage_summary_does_not_read_hidden_tensors_or_load_probe_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = write_stage_summary(
        output_dir=tmp_path / "stage_summary",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )

    assert result["summary"]["probe_model_exists_checked"] is True
    assert result["audit"]["boundary"]["read_hidden_state_tensors"] is False


def test_missing_artifact_generates_incomplete_evidence_audit(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)
    paths["closed_loop_audit"].unlink()

    result = write_stage_summary(
        output_dir=tmp_path / "stage_summary",
        backend=BACKEND,
        overwrite=True,
        input_paths=paths,
    )

    assert result["summary"]["status"] == "incomplete_evidence"
    assert result["audit"]["status"] == "incomplete_evidence"
    assert str(paths["closed_loop_audit"]) in result["audit"]["missing_upstream_artifacts"]


def test_rejects_forbidden_upstream_output_dir(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)

    with pytest.raises(ValueError, match="forbidden upstream path"):
        write_stage_summary(
            output_dir=Path.cwd() / "outputs/logs/sprint_2F_mini_closed_loop_report",
            backend=BACKEND,
            overwrite=True,
            input_paths=paths,
        )


def test_cli_smoke_generates_all_formal_outputs(tmp_path: Path) -> None:
    paths = write_test_artifacts(tmp_path)
    output_dir = tmp_path / "cli_stage_summary"

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
            "--representation-feature-report",
            str(paths["representation_feature_report"]),
            "--representation-features",
            str(paths["representation_features"]),
            "--probe-dataset-report",
            str(paths["probe_dataset_report"]),
            "--probe-dataset",
            str(paths["probe_dataset"]),
            "--probe-eval-report",
            str(paths["probe_eval_report"]),
            "--probe-predictions",
            str(paths["probe_predictions"]),
            "--probe-model",
            str(paths["probe_model"]),
            "--guidance-candidate-report",
            str(paths["guidance_candidate_report"]),
            "--guidance-candidate-manifest",
            str(paths["guidance_candidate_manifest"]),
            "--closed-loop-report",
            str(paths["closed_loop_report"]),
            "--closed-loop-audit",
            str(paths["closed_loop_audit"]),
            "--output-dir",
            str(output_dir),
            "--backend",
            BACKEND,
            "--full-pytest-passed",
            "508",
            "--full-pytest-skipped",
            "2",
            "--full-pytest-duration-seconds",
            "7.59",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Wrote Sprint 2 stage summary and figures" in result.stdout
    assert (output_dir / SUMMARY_FILENAME).exists()
    assert (output_dir / AUDIT_FILENAME).exists()
    for filename in FIGURE_FILENAMES:
        assert (output_dir / "figures" / filename).exists()


def test_cli_has_no_model_training_or_steering_arguments() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND])

    assert args.backend == BACKEND
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "device")
    assert not hasattr(args, "train")
    assert not hasattr(args, "steer")
    assert not hasattr(args, "run_generation")
    assert not hasattr(args, "evaluate_accuracy")
