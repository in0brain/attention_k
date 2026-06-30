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
from recover_attention.probe_dataset import (  # noqa: E402
    BACKEND,
    FORBIDDEN_OUTPUT_FILENAMES,
    REQUIRED_FEATURE_ARRAY_KEYS,
    build_probe_dataset,
    map_probe_target,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "18_build_probe_dataset.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_probe_dataset_cli", SCRIPT_PATH)
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


def base_feature_record(
    *,
    feature_id: str = "case_001::recovered::0",
    human_error_type: str = "semantic_equivalent_recovery",
    human_semantic_role: str = "critical_number",
    human_guidance_priority: str = "low",
    human_recoverability_label: str = "Recoverable",
    human_attention_anchor_label: str = "Weak Anchor",
    probe_usage: str = "include",
    with_null_position_features: bool = False,
    nonfinite: bool = False,
) -> dict:
    record = {
        "feature_id": feature_id,
        "masked_id": feature_id.split("::")[0],
        "id": "case_001",
        "unit_id": "unit_001",
        "original_cache_id": f"{feature_id}::original",
        "masked_cache_id": f"{feature_id}::masked",
        "recovered_cache_id": f"{feature_id}::recovered",
        "recovered_input_index": 0,
        "backend": "representation_features_minimal_v0",
        "source_cache_backend": "hf_local_causal_lm_hidden_states_v0",
        "model_name": "local-model",
        "tokenizer_name": "local-tokenizer",
        "layer_indices": [0, 1],
        "hidden_size": 4,
        "hidden_state_path": "missing_tensor_that_must_not_be_read.pt",
        "human_attention_anchor_label": human_attention_anchor_label,
        "human_attention_anchor_label_name": human_attention_anchor_label,
        "human_recoverability_label": human_recoverability_label,
        "human_semantic_role": human_semantic_role,
        "human_guidance_priority": human_guidance_priority,
        "human_error_type": human_error_type,
        "probe_usage": probe_usage,
        "warnings": [],
    }
    for index, key in enumerate(REQUIRED_FEATURE_ARRAY_KEYS, start=1):
        record[key] = [index / 10.0, index / 10.0 + 0.01]
        stem = key.removesuffix("_by_layer")
        record[f"{stem}_mean"] = index / 10.0
        record[f"{stem}_max"] = index / 10.0 + 0.01
        record[f"{stem}_min"] = index / 10.0
        record[f"{stem}_first_layer"] = index / 10.0
        record[f"{stem}_last_layer"] = index / 10.0 + 0.01
        record[f"{stem}_delta_last_minus_first"] = 0.01
    if with_null_position_features:
        record["span_original_masked_cosine_by_layer"] = None
        record["span_original_masked_cosine_mean"] = None
        record["mask_position_original_masked_cosine_by_layer"] = None
        record["mask_position_original_masked_cosine_mean"] = None
    if nonfinite:
        record["question_original_masked_cosine_by_layer"] = [0.1, float("inf")]
        record["question_original_masked_cosine_mean"] = float("nan")
    return record


def write_source_files(tmp_path: Path, records: list[dict]) -> dict[str, Path]:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    features_path = feature_dir / "representation_features.jsonl"
    report_path = feature_dir / "representation_feature_report.json"
    write_jsonl(records, features_path)
    write_json(
        report_path,
        {
            "sprint": "2B-fix",
            "backend": "representation_features_minimal_v0",
            "counts": {
                "num_feature_records": len(records),
                "num_input_records": len(records) * 3,
            },
        },
    )
    for filename in [
        "representation_feature_manifest.jsonl",
        "input_representation_summary.jsonl",
        "feature_schema.json",
    ]:
        (feature_dir / filename).write_text("not valid json and must not be read\n", encoding="utf-8")
    return {"features": features_path, "report": report_path}


def test_target_mapping_rules_cover_required_labels() -> None:
    cases = [
        (
            base_feature_record(human_error_type="wrong_numeric_recovery"),
            "risk_positive",
            "human_error_type_risk_positive",
        ),
        (
            base_feature_record(human_error_type="misleading_entity_or_unit"),
            "risk_positive",
            "human_error_type_risk_positive",
        ),
        (
            base_feature_record(
                human_error_type="generic_recovery",
                human_semantic_role="critical_number",
            ),
            "positive_anchor",
            "generic_recovery_critical_number",
        ),
        (
            base_feature_record(
                human_error_type="generic_recovery",
                human_semantic_role="question_target",
            ),
            "hard_negative_or_weak_positive",
            "generic_recovery_noncritical",
        ),
        (
            base_feature_record(human_error_type="semantic_equivalent_recovery"),
            "negative",
            "semantic_equivalent_recovery",
        ),
        (
            base_feature_record(
                human_error_type="fragment_recovery",
                human_guidance_priority="none",
            ),
            "negative",
            "fragment_recovery_low_priority",
        ),
        (
            base_feature_record(human_error_type="semantically_meaningful_but_recoverable"),
            "hard_negative_or_weak_positive",
            "meaningful_but_recoverable",
        ),
        (
            base_feature_record(human_error_type="recoverable_but_semantically_meaningful_span"),
            "hard_negative_or_weak_positive",
            "meaningful_but_recoverable",
        ),
        (
            base_feature_record(
                human_error_type="other_error",
                human_recoverability_label="Partially Recoverable",
                human_attention_anchor_label="Medium Anchor",
                human_guidance_priority="medium",
            ),
            "hard_negative_or_weak_positive",
            "recoverable_weak_or_medium_anchor",
        ),
    ]

    for record, expected_target, expected_rule in cases:
        mapping = map_probe_target(record)
        assert mapping["probe_target"] == expected_target
        assert mapping["probe_target_usable"] is True
        assert mapping["target_mapping_status"] == "mapped"
        assert mapping["target_mapping_rule"] == expected_rule


def test_missing_unmapped_and_probe_usage_exclude_are_retained() -> None:
    missing_record = base_feature_record()
    missing_record.pop("human_error_type")
    missing = map_probe_target(missing_record)
    assert missing["probe_target"] == "unmapped"
    assert missing["probe_target_usable"] is False
    assert missing["target_mapping_status"] == "missing_human_metadata"

    unmapped = map_probe_target(
        base_feature_record(
            human_error_type="unrecognized",
            human_recoverability_label="Not Recoverable",
            human_attention_anchor_label="Distractor",
            human_guidance_priority="high",
        )
    )
    assert unmapped["probe_target"] == "unmapped"
    assert unmapped["probe_target_usable"] is False
    assert unmapped["target_mapping_status"] == "unmapped"

    excluded = map_probe_target(
        base_feature_record(
            human_error_type="wrong_numeric_recovery",
            probe_usage="exclude",
        )
    )
    assert excluded["probe_target"] == "risk_positive"
    assert excluded["probe_target_usable"] is False
    assert excluded["target_mapping_status"] == "excluded_by_probe_usage"


def test_build_probe_dataset_uses_formal_2b_outputs_and_keeps_null_features(tmp_path: Path) -> None:
    records = [
        base_feature_record(feature_id="case_001::recovered::0"),
        base_feature_record(
            feature_id="case_002::recovered::0",
            human_error_type="wrong_numeric_recovery",
            with_null_position_features=True,
            nonfinite=True,
        ),
        base_feature_record(
            feature_id="case_003::recovered::0",
            human_error_type="unrecognized",
            human_recoverability_label="Not Recoverable",
            human_attention_anchor_label="Distractor",
            human_guidance_priority="high",
        ),
    ]
    paths = write_source_files(tmp_path, records)
    output_dir = tmp_path / "probe_dataset"

    result = build_probe_dataset(
        features_path=paths["features"],
        feature_report_path=paths["report"],
        output_dir=output_dir,
        backend=BACKEND,
        overwrite=True,
    )
    dataset_records = result["dataset_records"]
    report = result["report"]

    assert len(dataset_records) == 3
    assert len(read_jsonl(output_dir / "probe_dataset.jsonl")) == 3
    assert (output_dir / "probe_dataset_report.json").exists()
    assert not any((output_dir / filename).exists() for filename in FORBIDDEN_OUTPUT_FILENAMES)
    assert report["inputs"]["hidden_state_tensors_read"] is False
    assert report["inputs"]["deprecated_2B_inputs_present_but_not_read"]
    assert report["source_2B"]["backend"] == "representation_features_minimal_v0"
    assert report["counts"]["num_probe_dataset_records"] == 3
    assert report["counts"]["num_probe_records"] == 3
    assert report["counts"]["num_usable_records"] == 2
    assert report["counts"]["num_unusable_records"] == 1
    assert report["target_counts"]["negative"] == 1
    assert report["target_counts"]["risk_positive"] == 1
    assert report["target_counts"]["unmapped"] == 1
    assert report["null_feature_counts"]["records_with_null_position_features"] == 1
    assert report["boundary_checks"]["read_hidden_state_tensors"] is False
    assert report["boundary_checks"]["read_legacy_2B_inputs"] is False
    assert report["boundary_checks"]["trained_probe"] is False
    assert report["boundary_checks"]["performed_attention_guidance"] is False
    assert "span_original_masked_cosine_by_layer" in dataset_records[1]["null_feature_keys"]
    assert dataset_records[1]["has_null_position_features"] is True
    assert dataset_records[1]["feature_arrays"]["span_original_masked_cosine_by_layer"] is None
    assert dataset_records[1]["feature_values"]["question_original_masked_cosine_mean"] is None
    assert_no_nonfinite_numbers(json.loads((output_dir / "probe_dataset_report.json").read_text()))
    for line in (output_dir / "probe_dataset.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert_no_nonfinite_numbers(record)
        assert not {"split", "fold_id", "train", "dev", "test"} & set(record)


def test_optional_human_labels_fill_missing_metadata(tmp_path: Path) -> None:
    record = base_feature_record(feature_id="case_001::recovered::0")
    for field in [
        "human_attention_anchor_label",
        "human_attention_anchor_label_name",
        "human_recoverability_label",
        "human_semantic_role",
        "human_guidance_priority",
        "human_error_type",
        "probe_usage",
    ]:
        record.pop(field)
    paths = write_source_files(tmp_path, [record])
    human_labels = tmp_path / "human_labels.jsonl"
    write_jsonl(
        [
            {
                "masked_id": "case_001",
                "human_attention_anchor_label": "Weak Anchor",
                "human_attention_anchor_label_name": "Weak Anchor",
                "human_recoverability_label": "Recoverable",
                "human_semantic_role": "critical_number",
                "human_guidance_priority": "low",
                "human_error_type": "wrong_numeric_recovery",
                "probe_usage": "include",
            }
        ],
        human_labels,
    )

    result = build_probe_dataset(
        features_path=paths["features"],
        feature_report_path=paths["report"],
        output_dir=tmp_path / "probe_dataset",
        backend=BACKEND,
        overwrite=True,
        human_labels_path=human_labels,
    )

    assert result["dataset_records"][0]["probe_target"] == "risk_positive"
    assert result["dataset_records"][0]["probe_target_usable"] is True
    assert result["report"]["counts"]["num_missing_human_metadata"] == 0


def test_default_no_overwrite_and_output_dir_guard(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, [base_feature_record()])
    output_dir = tmp_path / "probe_dataset"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="--overwrite"):
        build_probe_dataset(
            features_path=paths["features"],
            feature_report_path=paths["report"],
            output_dir=output_dir,
            backend=BACKEND,
        )

    with pytest.raises(ValueError, match="forbidden path"):
        build_probe_dataset(
            features_path=paths["features"],
            feature_report_path=paths["report"],
            output_dir=Path.cwd() / "outputs/logs/sprint_2B_representation_features",
            backend=BACKEND,
            overwrite=True,
        )


def test_cli_smoke_generates_only_probe_dataset_outputs(tmp_path: Path) -> None:
    paths = write_source_files(tmp_path, [base_feature_record()])
    output_dir = tmp_path / "cli_probe_dataset"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--features",
            str(paths["features"]),
            "--feature-report",
            str(paths["report"]),
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

    assert "[OK] Built Sprint 2C probe dataset" in result.stdout
    assert (output_dir / "probe_dataset.jsonl").exists()
    assert (output_dir / "probe_dataset_report.json").exists()
    assert not any((output_dir / filename).exists() for filename in FORBIDDEN_OUTPUT_FILENAMES)


def test_cli_has_no_probe_training_or_guidance_arguments() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND])

    assert args.backend == BACKEND
    assert not hasattr(args, "split")
    assert not hasattr(args, "fold_id")
    assert not hasattr(args, "train")
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "guidance")
