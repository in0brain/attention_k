from __future__ import annotations

import importlib.util
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.probe_training import (  # noqa: E402
    BACKEND,
    MODEL_TYPE,
    build_feature_matrix,
    flatten_probe_record,
    make_cv_folds,
    select_usable_records,
    standardize_train_and_test,
    train_probe_baseline,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "19_train_probe_baseline.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_probe_baseline_cli", SCRIPT_PATH)
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


def probe_record(
    index: int,
    target: str,
    *,
    usable: bool = True,
    layer_indices: list[int] | None = None,
    nulls: bool = False,
) -> dict:
    layer_indices = [0, 8] if layer_indices is None else layer_indices
    sign = 1.0 if target in {"risk_positive", "positive_anchor"} else -1.0
    value = sign * (index + 1) / 10.0
    feature_values = {
        "question_original_masked_cosine_mean": value,
        "question_original_recovered_cosine_mean": value + 0.01,
        "span_original_masked_cosine_mean": None if nulls else value + 0.02,
    }
    feature_arrays = {
        "question_original_masked_cosine_by_layer": [value, value + 0.1],
        "span_original_masked_cosine_by_layer": None if nulls else [value + 0.2, value + 0.3],
    }
    null_feature_keys = []
    if nulls:
        null_feature_keys = [
            "span_original_masked_cosine_mean",
            "span_original_masked_cosine_by_layer",
        ]
    return {
        "probe_record_id": f"probe_{index:03d}",
        "feature_id": f"case_{index:03d}::recovered::0",
        "masked_id": f"case_{index:03d}",
        "id": f"case_{index:03d}",
        "unit_id": "unit_001",
        "layer_indices": layer_indices,
        "probe_target": target,
        "probe_target_usable": usable,
        "feature_values": feature_values,
        "feature_arrays": feature_arrays,
        "null_feature_keys": null_feature_keys,
        "num_null_features": len(null_feature_keys),
        "has_null_position_features": bool(null_feature_keys),
        "hidden_state_path": "forbidden_hidden_state_that_must_not_be_read.pt",
        "warnings": [],
    }


def balanced_records() -> list[dict]:
    targets = [
        "risk_positive",
        "risk_positive",
        "positive_anchor",
        "positive_anchor",
        "negative",
        "negative",
        "hard_negative_or_weak_positive",
        "hard_negative_or_weak_positive",
    ]
    return [
        probe_record(index, target, nulls=index in {1, 6})
        for index, target in enumerate(targets)
    ]


def write_source_files(tmp_path: Path, records: list[dict]) -> dict[str, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    dataset_path = input_dir / "probe_dataset.jsonl"
    report_path = input_dir / "probe_dataset_report.json"
    write_jsonl(records, dataset_path)
    write_json(
        report_path,
        {
            "sprint": "2C",
            "backend": "probe_dataset_mapping_v0",
            "counts": {
                "num_probe_dataset_records": len(records),
                "num_usable_records": sum(
                    1
                    for record in records
                    if record.get("probe_target_usable")
                    and record.get("probe_target") != "unmapped"
                ),
            },
        },
    )
    (tmp_path / "representation_features.jsonl").write_text(
        "not a 2D input and must not be read\n",
        encoding="utf-8",
    )
    return {"dataset": dataset_path, "report": report_path}


def test_select_usable_records_filters_unmapped_and_unusable() -> None:
    records = [
        probe_record(0, "risk_positive"),
        probe_record(1, "negative", usable=False),
        probe_record(2, "unmapped"),
    ]
    selected = select_usable_records(records)

    assert [record["probe_record_id"] for record in selected] == ["probe_000"]


def test_flatten_feature_values_arrays_and_null_missing_indicators() -> None:
    record = probe_record(0, "risk_positive", nulls=True)
    values, missing = flatten_probe_record(record)

    assert values["question_original_masked_cosine_mean"] != 0.0
    assert missing["question_original_masked_cosine_mean"] is False
    assert values["span_original_masked_cosine_mean"] == 0.0
    assert missing["span_original_masked_cosine_mean"] is True
    assert "question_original_masked_cosine_layer_0" in values
    assert "question_original_masked_cosine_layer_8" in values
    assert values["span_original_masked_cosine_layer_0"] == 0.0
    assert missing["span_original_masked_cosine_layer_0"] is True
    assert missing["span_original_masked_cosine_layer_8"] is True

    matrix_data = build_feature_matrix([record])
    assert matrix_data["matrix"].shape[0] == 1
    assert "span_original_masked_cosine_mean__is_missing" in matrix_data["feature_names"]
    indicator_index = matrix_data["feature_names"].index(
        "span_original_masked_cosine_mean__is_missing"
    )
    assert matrix_data["matrix"][0, indicator_index] == 1.0


def test_standardization_uses_train_fold_statistics_only() -> None:
    train = np.array([[1.0, 3.0], [1.0, 5.0]])
    test = np.array([[10.0, 7.0]])

    train_scaled, test_scaled, mean, std = standardize_train_and_test(train, test)

    assert mean.tolist() == [1.0, 4.0]
    assert std.tolist() == [1.0, 1.0]
    assert train_scaled.tolist() == [[0.0, -1.0], [0.0, 1.0]]
    assert test_scaled.tolist() == [[9.0, 3.0]]


def test_leave_one_out_and_stratified_k_fold_generation() -> None:
    labels = ["a", "a", "b", "b"]

    loo, warnings, cv, num_folds = make_cv_folds(labels, cv="leave_one_out", num_folds=None, seed=42)
    assert cv == "leave_one_out"
    assert num_folds == 4
    assert warnings == []
    assert all(len(test_indices) == 1 for _, test_indices in loo)

    folds, warnings, cv, num_folds = make_cv_folds(
        labels,
        cv="stratified_k_fold",
        num_folds=5,
        seed=42,
    )
    assert cv == "stratified_k_fold"
    assert num_folds == 2
    assert warnings
    assert len(folds) == 2


def test_train_probe_baseline_outputs_predictions_report_and_model(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, balanced_records())
    output_dir = tmp_path / "probe_training"

    result = train_probe_baseline(
        dataset_path=paths["dataset"],
        dataset_report_path=paths["report"],
        output_dir=output_dir,
        backend=BACKEND,
        model=MODEL_TYPE,
        cv="leave_one_out",
        seed=42,
        overwrite=True,
    )
    predictions = read_jsonl(output_dir / "probe_predictions.jsonl")
    report = json.loads((output_dir / "probe_eval_report.json").read_text(encoding="utf-8"))

    assert result["report"]["status"] == "ok"
    assert len(predictions) == 8
    assert all(prediction["test_size"] == 1 for prediction in predictions)
    assert all(prediction["decision_scores"] for prediction in predictions)
    assert {"gold_probe_target", "predicted_probe_target", "correct"} <= set(predictions[0])
    assert report["data_summary"]["num_records_usable"] == 8
    assert report["training"]["model_type"] == MODEL_TYPE
    assert report["training"]["cv_strategy"] == "leave_one_out"
    assert report["training"]["num_folds"] == 8
    assert "accuracy" in report["metrics"]
    assert "macro_f1" in report["metrics"]
    assert "per_class" in report["metrics"]
    assert "majority_class" in report["baselines"]
    assert report["feature_signal_summary"]["top_weighted_features"]
    assert report["feature_signal_summary"]["feature_group_weight_summary"]
    assert report["feature_signal_summary"]["layer_weight_summary"]
    assert report["boundary"]["read_hidden_state_tensors"] is False
    assert report["boundary"]["read_2B_representation_features"] is False
    assert report["boundary"]["generated_guidance_candidates"] is False
    assert report["boundary"]["performed_attention_guidance"] is False
    assert report["small_sample_warning"]
    assert (output_dir / "probe_model.pkl").exists()
    assert (output_dir / "probe_feature_index.json").exists()
    assert not (output_dir / "guidance_candidate_manifest.jsonl").exists()
    assert_no_nonfinite_numbers(report)
    for prediction in predictions:
        assert_no_nonfinite_numbers(prediction)

    with (output_dir / "probe_model.pkl").open("rb") as handle:
        model_bundle = pickle.load(handle)
    assert model_bundle["backend"] == BACKEND
    assert model_bundle["model_type"] == MODEL_TYPE
    assert model_bundle["classes"]
    assert model_bundle["feature_names"]
    assert model_bundle["scaler"]["mean"]
    assert model_bundle["model_parameters"]["weights"]


def test_insufficient_data_report_does_not_train_or_emit_predictions(tmp_path: Path) -> None:
    records = [probe_record(index, "negative") for index in range(4)]
    paths = write_source_files(tmp_path, records)
    output_dir = tmp_path / "probe_training"

    result = train_probe_baseline(
        dataset_path=paths["dataset"],
        dataset_report_path=paths["report"],
        output_dir=output_dir,
        backend=BACKEND,
        model=MODEL_TYPE,
        overwrite=True,
    )
    report = result["report"]

    assert report["status"] == "insufficient_data"
    assert report["data_summary"]["num_records_usable"] == 4
    assert not (output_dir / "probe_predictions.jsonl").exists()
    assert not (output_dir / "probe_model.pkl").exists()
    assert (output_dir / "probe_eval_report.json").exists()


def test_default_no_overwrite_and_output_dir_guard(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, balanced_records())
    output_dir = tmp_path / "probe_training"
    output_dir.mkdir()

    try:
        train_probe_baseline(
            dataset_path=paths["dataset"],
            dataset_report_path=paths["report"],
            output_dir=output_dir,
            backend=BACKEND,
            model=MODEL_TYPE,
        )
    except ValueError as exc:
        assert "--overwrite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected overwrite guard")

    try:
        train_probe_baseline(
            dataset_path=paths["dataset"],
            dataset_report_path=paths["report"],
            output_dir=Path.cwd() / "outputs/logs/sprint_2C_probe_dataset",
            backend=BACKEND,
            model=MODEL_TYPE,
            overwrite=True,
        )
    except ValueError as exc:
        assert "forbidden path" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected forbidden path guard")


def test_cli_smoke_generates_formal_outputs(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, balanced_records())
    output_dir = tmp_path / "cli_probe_training"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset",
            str(paths["dataset"]),
            "--dataset-report",
            str(paths["report"]),
            "--output-dir",
            str(output_dir),
            "--backend",
            BACKEND,
            "--model",
            MODEL_TYPE,
            "--cv",
            "leave_one_out",
            "--seed",
            "42",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Trained Sprint 2D probe baseline" in result.stdout
    assert (output_dir / "probe_predictions.jsonl").exists()
    assert (output_dir / "probe_eval_report.json").exists()
    assert (output_dir / "probe_model.pkl").exists()
    assert not (output_dir / "guidance_candidate_manifest.jsonl").exists()


def test_cli_has_no_hidden_state_or_2b_feature_arguments() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND, "--model", MODEL_TYPE])

    assert args.backend == BACKEND
    assert args.model == MODEL_TYPE
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "device")
    assert not hasattr(args, "device_map")
    assert not hasattr(args, "load_in_4bit")
    assert not hasattr(args, "hidden_state_dir")
    assert not hasattr(args, "features")
    assert not hasattr(args, "feature_report")
