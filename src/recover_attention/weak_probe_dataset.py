"""Sprint 2G full-scale weak probe dataset construction.

Merges Sprint 2G representation features (from the real hidden-state cache) with
the weak-auto labels, producing a probe-training-ready dataset. The probe target
comes from the weak label (label_source=weak_auto, human_reviewed=false); it is
never a human label.

It also performs the *adaptive k-fold precondition*: it counts usable probe
target classes and recommends a fold strategy following the downgrade ladder
5 -> 3 -> 2 -> leave-one-out.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl
from recover_attention.probe_dataset import (
    contains_null,
    extract_feature_arrays,
    extract_feature_values,
    has_null_position_features,
    sanitize_json_value,
)

BACKEND = "weak_probe_dataset_mapping_v0"
DATASET_FILENAME = "weak_probe_dataset.jsonl"
REPORT_FILENAME = "weak_probe_dataset_report.json"
LABEL_SOURCE = "weak_auto"

BOUNDARY_STATEMENT = (
    "This is a weak-labeled 2000-case dry run. Probe metrics are weak-labeled "
    "diagnostic metrics, not human-supervised validation metrics. It does not "
    "execute attention steering or validate hallucination reduction."
)


def decide_kfold(usable_class_counts: dict[str, int]) -> dict[str, Any]:
    """Decide the cross-validation strategy from usable class counts.

    Ladder: min_class_count >= 5 -> 5-fold; >= 3 -> 3-fold; >= 2 -> 2-fold;
    otherwise fall back to leave-one-out.
    """
    num_classes = len(usable_class_counts)
    min_class_count = min(usable_class_counts.values()) if usable_class_counts else 0
    warnings: list[str] = []

    if num_classes < 2:
        return {
            "cv_strategy": "holdout_or_leave_one_out_fallback",
            "num_folds": None,
            "min_class_count": min_class_count,
            "num_usable_classes": num_classes,
            "rule": "fewer than 2 usable classes; stratified k-fold impossible",
            "warnings": [
                "fewer than 2 usable probe target classes; probe training will "
                "report insufficient data"
            ],
        }

    if min_class_count >= 5:
        cv_strategy, num_folds = "stratified_k_fold", 5
    elif min_class_count >= 3:
        cv_strategy, num_folds = "stratified_k_fold", 3
        warnings.append("min_class_count in [3,5); downgraded from 5-fold to 3-fold")
    elif min_class_count >= 2:
        cv_strategy, num_folds = "stratified_k_fold", 2
        warnings.append("min_class_count in [2,3); downgraded to 2-fold")
    else:
        return {
            "cv_strategy": "leave_one_out",
            "num_folds": None,
            "min_class_count": min_class_count,
            "num_usable_classes": num_classes,
            "rule": "min_class_count < 2; downgraded to leave-one-out fallback",
            "warnings": ["min_class_count < 2; downgraded to leave-one-out fallback"],
        }

    return {
        "cv_strategy": cv_strategy,
        "num_folds": num_folds,
        "min_class_count": min_class_count,
        "num_usable_classes": num_classes,
        "rule": (
            "num_folds <= min_class_count among usable probe targets; "
            "ladder 5 -> 3 -> 2 -> leave-one-out"
        ),
        "warnings": warnings,
    }


def _index_weak_labels(weak_labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in weak_labels:
        masked_id = record.get("masked_id")
        if isinstance(masked_id, str):
            index[masked_id] = record
    return index


def build_weak_probe_record(
    feature_record: dict[str, Any],
    weak_label: dict[str, Any],
    *,
    source_line_number: int,
    source_features_path: str | Path,
) -> dict[str, Any]:
    feature_arrays = extract_feature_arrays(feature_record)
    feature_values = extract_feature_values(feature_record)
    null_feature_keys = sorted(
        key
        for key, value in {**feature_arrays, **feature_values}.items()
        if contains_null(value)
    )
    warnings = sorted(
        {str(warning) for warning in feature_record.get("warnings", []) if warning is not None}
    )
    warnings.extend(str(w) for w in weak_label.get("warnings", []) if w is not None)
    if null_feature_keys:
        warnings.append("null representation feature(s) retained without imputation")

    return sanitize_json_value(
        {
            "probe_record_id": feature_record.get("feature_id"),
            "feature_id": feature_record.get("feature_id"),
            "full_scale_id": weak_label.get("full_scale_id"),
            "source_question_id": weak_label.get("source_question_id"),
            "masked_id": feature_record.get("masked_id"),
            "id": feature_record.get("id"),
            "unit_id": feature_record.get("unit_id"),
            "recovered_input_index": feature_record.get("recovered_input_index"),
            "source_feature_backend": feature_record.get("backend"),
            "source_cache_backend": feature_record.get("source_cache_backend"),
            "model_name": feature_record.get("model_name"),
            "tokenizer_name": feature_record.get("tokenizer_name"),
            "layer_indices": feature_record.get("layer_indices"),
            "hidden_size": feature_record.get("hidden_size"),
            "source_features_path": str(source_features_path),
            "source_features_line": source_line_number,
            "probe_target": weak_label.get("probe_target"),
            "probe_target_usable": bool(weak_label.get("probe_target_usable")),
            "label_source": LABEL_SOURCE,
            "label_backend": weak_label.get("label_backend"),
            "label_rule": weak_label.get("label_rule"),
            "label_confidence": weak_label.get("label_confidence"),
            "human_reviewed": False,
            "chosen_span_type": weak_label.get("chosen_span_type"),
            "feature_values": feature_values,
            "feature_arrays": feature_arrays,
            "null_feature_keys": null_feature_keys,
            "num_null_features": len(null_feature_keys),
            "has_null_position_features": has_null_position_features(null_feature_keys),
            "warnings": warnings,
        }
    )


def build_weak_probe_dataset(
    *,
    features_path: str | Path,
    feature_report_path: str | Path,
    weak_labels_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge representation features with weak labels into a probe dataset."""
    if backend != BACKEND:
        raise ValueError(f"Unsupported weak probe dataset backend {backend!r}; expected {BACKEND!r}")

    features_path = Path(features_path)
    feature_report_path = Path(feature_report_path)
    weak_labels_path = Path(weak_labels_path)
    output_dir = Path(output_dir)
    dataset_path = output_dir / DATASET_FILENAME
    report_path = output_dir / REPORT_FILENAME

    from recover_attention.full_scale_manifest import ensure_output_dir_allowed

    ensure_output_dir_allowed(output_dir)
    for path in (dataset_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path} (pass overwrite=True to replace)"
            )

    feature_records = read_jsonl(features_path)
    if not feature_records:
        raise ValueError(f"representation features input is empty: {features_path}")
    feature_report = _load_json(feature_report_path)
    weak_labels = read_jsonl(weak_labels_path)
    weak_index = _index_weak_labels(weak_labels)

    dataset_records: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    usable_target_counts: Counter[str] = Counter()
    num_missing_label = 0
    records_with_null = 0

    for line_number, feature_record in enumerate(feature_records, start=1):
        masked_id = feature_record.get("masked_id")
        weak_label = weak_index.get(str(masked_id))
        if weak_label is None:
            num_missing_label += 1
            continue
        record = build_weak_probe_record(
            feature_record,
            weak_label,
            source_line_number=line_number,
            source_features_path=features_path,
        )
        dataset_records.append(record)
        target_counts[record["probe_target"]] += 1
        if record["probe_target_usable"]:
            usable_target_counts[record["probe_target"]] += 1
        if record["num_null_features"]:
            records_with_null += 1

    _write_jsonl_strict(dataset_records, dataset_path)

    kfold_decision = decide_kfold(dict(usable_target_counts))
    warnings: list[str] = list(kfold_decision["warnings"])
    if num_missing_label:
        warnings.append(
            f"{num_missing_label} feature record(s) had no matching weak label "
            "(skipped)"
        )

    report = {
        "backend": backend,
        "status": "ok",
        "inputs": {
            "representation_features_path": features_path.as_posix(),
            "representation_feature_report_path": feature_report_path.as_posix(),
            "weak_labels_path": weak_labels_path.as_posix(),
            "source_feature_backend": feature_report.get("backend"),
        },
        "outputs": {
            "weak_probe_dataset_path": dataset_path.as_posix(),
            "weak_probe_dataset_report_path": report_path.as_posix(),
        },
        "counts": {
            "num_feature_records": len(feature_records),
            "num_probe_records": len(dataset_records),
            "num_usable_records": sum(usable_target_counts.values()),
            "num_unusable_records": len(dataset_records) - sum(usable_target_counts.values()),
            "num_missing_label": num_missing_label,
            "records_with_any_null_feature": records_with_null,
        },
        "probe_target_counts": dict(sorted(target_counts.items())),
        "usable_probe_target_counts": dict(sorted(usable_target_counts.items())),
        "label_source": LABEL_SOURCE,
        "human_reviewed_full_scale": False,
        "adaptive_kfold_decision": kfold_decision,
        "metrics_caveat": (
            "Downstream probe metrics are weak-labeled diagnostic metrics, not "
            "human-supervised validation metrics."
        ),
        "boundary": BOUNDARY_STATEMENT,
        "warnings": warnings,
    }
    ensure_dir(output_dir)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "dataset_records": dataset_records,
        "report": report,
        "kfold_decision": kfold_decision,
        "output_files": {
            "weak_probe_dataset": dataset_path.as_posix(),
            "weak_probe_dataset_report": report_path.as_posix(),
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON input: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must be an object: {json_path}")
    return data


def _write_jsonl_strict(records: list[dict[str, Any]], path: str | Path) -> None:
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
