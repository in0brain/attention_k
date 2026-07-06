from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.weak_probe_dataset import build_weak_probe_dataset, decide_kfold


def _feature_record(masked_id: str, full_scale_id: str) -> dict:
    record = {
        "feature_id": f"{masked_id}::recovered::0",
        "masked_id": masked_id,
        "id": full_scale_id,
        "unit_id": "unit_000",
        "recovered_input_index": 0,
        "backend": "representation_features_minimal_v0",
        "source_cache_backend": "hf_local_causal_lm_hidden_states_v0",
        "model_name": "D:/models/Qwen2.5-7B-Instruct",
        "tokenizer_name": "D:/models/Qwen2.5-7B-Instruct",
        "layer_indices": [0, 8, 16, 24, 27],
        "hidden_size": 3584,
        "warnings": [],
    }
    for key in [
        "question_original_masked_cosine",
        "question_original_recovered_cosine",
        "question_masked_recovered_cosine",
        "span_original_masked_cosine",
        "span_original_recovered_cosine",
        "span_masked_recovered_cosine",
        "mask_position_original_masked_cosine",
        "mask_position_original_recovered_cosine",
        "mask_position_masked_recovered_cosine",
    ]:
        record[f"{key}_by_layer"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        record[f"{key}_mean"] = 0.3
        record[f"{key}_max"] = 0.5
    return record


def _weak_label(masked_id: str, full_scale_id: str, target: str) -> dict:
    return {
        "full_scale_id": full_scale_id,
        "source_question_id": f"gsm8k_train_{full_scale_id}",
        "masked_id": masked_id,
        "unit_id": "unit_000",
        "question": "q",
        "answer": "1",
        "probe_target": target,
        "probe_target_usable": target != "unmapped",
        "label_source": "weak_auto",
        "label_backend": "weak_label_mapping_v0",
        "label_rule": "span_type_rule",
        "label_confidence": 0.7,
        "human_reviewed": False,
        "chosen_span_type": "number",
        "warnings": [],
    }


def test_decide_kfold_ladder() -> None:
    assert decide_kfold({"a": 5, "b": 9})["num_folds"] == 5
    assert decide_kfold({"a": 4, "b": 9})["num_folds"] == 3
    assert decide_kfold({"a": 2, "b": 9})["num_folds"] == 2
    loo = decide_kfold({"a": 1, "b": 9})
    assert loo["cv_strategy"] == "leave_one_out"
    single = decide_kfold({"a": 7})
    assert single["num_folds"] is None


def test_build_weak_probe_dataset_merges_and_decides(tmp_path: Path) -> None:
    targets = (
        ["risk_positive"] * 6
        + ["positive_anchor"] * 6
        + ["negative"] * 6
        + ["hard_negative_or_weak_positive"] * 6
    )
    features = []
    weak_labels = []
    for index, target in enumerate(targets):
        masked_id = f"fs2000_{index:06d}__unit_000__mask"
        full_scale_id = f"fs2000_{index:06d}"
        features.append(_feature_record(masked_id, full_scale_id))
        weak_labels.append(_weak_label(masked_id, full_scale_id, target))

    features_path = tmp_path / "representation_features.jsonl"
    weak_labels_path = tmp_path / "weak_labels_2000.jsonl"
    feature_report_path = tmp_path / "representation_feature_report.json"
    write_jsonl(features, features_path)
    write_jsonl(weak_labels, weak_labels_path)
    feature_report_path.write_text(
        json.dumps({"backend": "representation_features_minimal_v0"}), encoding="utf-8"
    )

    output_dir = tmp_path / "04_weak_probe_dataset"
    result = build_weak_probe_dataset(
        features_path=features_path,
        feature_report_path=feature_report_path,
        weak_labels_path=weak_labels_path,
        output_dir=output_dir,
    )

    records = read_jsonl(output_dir / "weak_probe_dataset.jsonl")
    assert len(records) == 24
    assert all(record["label_source"] == "weak_auto" for record in records)
    assert all(record["human_reviewed"] is False for record in records)
    assert all("feature_values" in record and "feature_arrays" in record for record in records)
    decision = result["report"]["adaptive_kfold_decision"]
    assert decision["cv_strategy"] == "stratified_k_fold"
    assert decision["num_folds"] == 5
    assert result["report"]["counts"]["num_usable_records"] == 24


def test_build_weak_probe_dataset_skips_unmatched_features(tmp_path: Path) -> None:
    features = [_feature_record("fs2000_000000__unit_000__mask", "fs2000_000000")]
    weak_labels = []  # no matching label
    features_path = tmp_path / "features.jsonl"
    weak_labels_path = tmp_path / "weak.jsonl"
    feature_report_path = tmp_path / "report.json"
    write_jsonl(features, features_path)
    write_jsonl(weak_labels, weak_labels_path)
    feature_report_path.write_text(json.dumps({"backend": "x"}), encoding="utf-8")

    result = build_weak_probe_dataset(
        features_path=features_path,
        feature_report_path=feature_report_path,
        weak_labels_path=weak_labels_path,
        output_dir=tmp_path / "out",
    )
    assert result["report"]["counts"]["num_probe_records"] == 0
    assert result["report"]["counts"]["num_missing_label"] == 1
