from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.full_scale_summary import (
    FIGURE_FILENAMES,
    write_full_scale_summary,
)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_root(root: Path) -> None:
    _write(
        root / "00_manifest" / "full_scale_manifest_report.json",
        {
            "requested_num_cases": 2000,
            "available_num_cases": 7473,
            "actual_num_cases": 2000,
            "source_artifact": "data/raw/gsm8k_train_normalized.jsonl",
            "sampling_rule": "seeded_sample",
            "seed": 42,
        },
    )
    _write(
        root / "01_downstream" / "weak_label_report.json",
        {
            "probe_target_counts": {"risk_positive": 400, "positive_anchor": 800, "negative": 500, "hard_negative_or_weak_positive": 300},
            "usable_probe_target_counts": {"risk_positive": 400, "positive_anchor": 800, "negative": 500, "hard_negative_or_weak_positive": 300},
        },
    )
    _write(
        root / "02_hidden_state_cache" / "hidden_state_cache_report.json",
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "num_cases": 2000,
            "num_inputs_total": 6000,
            "num_hidden_state_files": 6000,
            "failure_count": 0,
            "layer_indices": [0, 8, 16, 24, 27],
        },
    )
    _write(root / "02_hidden_state_cache" / "token_alignment_report.json", {"alignment_warning_count": 3})
    _write(root / "02_hidden_state_cache" / "real_run_metadata.json", {"model_name_or_path": "D:/models/Qwen2.5-7B-Instruct"})
    _write(
        root / "03_representation_features" / "representation_feature_report.json",
        {"backend": "representation_features_minimal_v0", "counts": {"num_feature_records": 2000, "num_masked_groups": 2000, "num_skipped_groups": 0}},
    )
    _write(
        root / "04_weak_probe_dataset" / "weak_probe_dataset_report.json",
        {
            "counts": {"num_probe_records": 2000, "num_usable_records": 2000},
            "usable_probe_target_counts": {"risk_positive": 400, "positive_anchor": 800, "negative": 500, "hard_negative_or_weak_positive": 300},
            "adaptive_kfold_decision": {"cv_strategy": "stratified_k_fold", "num_folds": 5, "min_class_count": 300},
        },
    )
    _write(
        root / "05_probe_training" / "probe_eval_report.json",
        {
            "status": "ok",
            "training": {"cv_strategy": "stratified_k_fold", "num_folds": 5},
            "metrics": {"accuracy": 0.62, "macro_f1": 0.55, "weighted_f1": 0.6},
            "baselines": {"majority_class": {"label": "positive_anchor", "accuracy": 0.4}},
        },
    )
    _write(
        root / "06_guidance_candidates" / "guidance_candidate_report.json",
        {
            "candidate_action_counts": {"increase_attention_to_original_span": 400, "preserve_original_span_attention": 800, "no_guidance": 500, "review_before_guidance": 300},
            "confidence_counts": {"high": 100, "medium": 700, "low": 1200, "unknown": 0},
        },
    )


def test_write_full_scale_summary_creates_outputs(tmp_path: Path) -> None:
    root = tmp_path / "sprint_2G_full_scale_2000"
    _build_root(root)
    output_dir = root / "07_stage_summary"

    result = write_full_scale_summary(root_dir=root, output_dir=output_dir)

    assert (output_dir / "full_scale_2000_summary.md").exists()
    assert (output_dir / "full_scale_2000_audit.json").exists()
    for name in FIGURE_FILENAMES:
        assert (output_dir / "figures" / name).exists(), name
        assert result["figure_status"][name] == "ok", result["figure_status"][name]

    audit = result["audit"]
    assert audit["actual_num_cases"] == 2000
    assert audit["boundary"]["executed_attention_steering"] is False
    assert audit["boundary"]["validated_hallucination_reduction"] is False
    assert audit["probe_training"]["num_folds"] == 5

    summary = (output_dir / "full_scale_2000_summary.md").read_text(encoding="utf-8")
    assert "Weak-labeled 2000-case Dry Run" in summary
    assert "does not execute attention steering" in summary


def test_write_full_scale_summary_handles_missing_reports(tmp_path: Path) -> None:
    root = tmp_path / "empty_root"
    root.mkdir()
    output_dir = root / "07_stage_summary"

    result = write_full_scale_summary(root_dir=root, output_dir=output_dir)

    # Even with all phase reports missing, the summary and audit are still written.
    assert (output_dir / "full_scale_2000_summary.md").exists()
    assert result["audit"]["phase_status"]["manifest"] == "missing"
