"""Sprint 2D probe training baseline from Sprint 2C probe dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "probe_training_baseline_v0"
MODEL_TYPE = "ridge_classifier_ovr_v0"
PREDICTIONS_FILENAME = "probe_predictions.jsonl"
REPORT_FILENAME = "probe_eval_report.json"
MODEL_FILENAME = "probe_model.pkl"
FEATURE_INDEX_FILENAME = "probe_feature_index.json"
POSITIVE_BINARY_TARGETS = {"risk_positive", "positive_anchor"}
TARGET_LABELS = [
    "risk_positive",
    "positive_anchor",
    "negative",
    "hard_negative_or_weak_positive",
]
NULL_FEATURE_STRATEGY = "zero_impute_with_missing_indicators"
STANDARDIZATION = "train_fold_zscore"
SMALL_SAMPLE_WARNING = (
    "Only 20 human-reviewed examples are available. Scores are diagnostic, not conclusive."
)


def train_probe_baseline(
    *,
    dataset_path: str | Path,
    dataset_report_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    model: str = MODEL_TYPE,
    cv: str = "leave_one_out",
    num_folds: int | None = None,
    seed: int = 42,
    alpha: float = 1.0,
    top_k_features: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a minimal linear probe baseline."""

    if backend != BACKEND:
        raise ValueError(f"Unsupported probe training backend {backend!r}; expected {BACKEND!r}")
    if model != MODEL_TYPE:
        raise ValueError(f"Unsupported probe model {model!r}; expected {MODEL_TYPE!r}")
    if cv not in {"leave_one_out", "stratified_k_fold"}:
        raise ValueError("cv must be one of: leave_one_out, stratified_k_fold")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if top_k_features <= 0:
        raise ValueError("top_k_features must be > 0")

    dataset_path = Path(dataset_path)
    dataset_report_path = Path(dataset_report_path)
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite)

    records = load_probe_dataset(dataset_path)
    dataset_report = load_json(dataset_report_path)
    usable_records = select_usable_records(records)
    output_files = {
        "probe_predictions": output_dir / PREDICTIONS_FILENAME,
        "probe_eval_report": output_dir / REPORT_FILENAME,
        "probe_model": output_dir / MODEL_FILENAME,
        "probe_feature_index": output_dir / FEATURE_INDEX_FILENAME,
    }

    if len(usable_records) < 4 or len({record["probe_target"] for record in usable_records}) < 2:
        report = build_insufficient_data_report(
            dataset_path=dataset_path,
            dataset_report_path=dataset_report_path,
            output_files=output_files,
            records=records,
            usable_records=usable_records,
            dataset_report=dataset_report,
            backend=backend,
            model=model,
            cv=cv,
            seed=seed,
        )
        write_json_strict(report, output_files["probe_eval_report"])
        return {
            "predictions": [],
            "report": report,
            "model_bundle": None,
            "output_files": {"probe_eval_report": str(output_files["probe_eval_report"])},
        }

    matrix_data = build_feature_matrix(usable_records)
    labels = [record["probe_target"] for record in usable_records]
    classes = ordered_classes(labels)
    folds, cv_warnings, resolved_cv, resolved_num_folds = make_cv_folds(
        labels,
        cv=cv,
        num_folds=num_folds,
        seed=seed,
    )

    predictions: list[dict[str, Any]] = []
    for fold_id, (train_indices, test_indices) in enumerate(folds):
        train_x = matrix_data["matrix"][train_indices, :]
        test_x = matrix_data["matrix"][test_indices, :]
        train_y = [labels[index] for index in train_indices]
        train_scaled, test_scaled, _, _ = standardize_train_and_test(train_x, test_x)
        fold_model = train_ridge_classifier_ovr(train_scaled, train_y, classes, alpha=alpha)
        decision_scores = decision_function(test_scaled, fold_model)
        predicted_indices = np.argmax(decision_scores, axis=1)
        for local_index, record_index in enumerate(test_indices):
            record = usable_records[record_index]
            predicted = classes[int(predicted_indices[local_index])]
            gold = record["probe_target"]
            score_map = {
                label: sanitize_float(float(decision_scores[local_index, class_index]))
                for class_index, label in enumerate(classes)
            }
            predictions.append(
                build_prediction_record(
                    record,
                    cv_strategy=resolved_cv,
                    fold_id=fold_id,
                    train_size=len(train_indices),
                    test_size=len(test_indices),
                    gold=gold,
                    predicted=predicted,
                    decision_scores=score_map,
                    num_features=len(matrix_data["feature_names"]),
                )
            )

    final_scaled, final_mean, final_std = standardize_full_matrix(matrix_data["matrix"])
    final_model = train_ridge_classifier_ovr(final_scaled, labels, classes, alpha=alpha)
    feature_signal_summary = summarize_feature_weights(
        final_model,
        matrix_data["feature_names"],
        top_k=top_k_features,
    )
    metrics = compute_multiclass_metrics(
        [prediction["gold_probe_target"] for prediction in predictions],
        [prediction["predicted_probe_target"] for prediction in predictions],
        classes,
    )
    binary_metrics = compute_binary_anchor_or_risk_metrics(predictions)
    majority_baseline = compute_majority_baseline(labels, classes)
    report = build_eval_report(
        dataset_path=dataset_path,
        dataset_report_path=dataset_report_path,
        output_files=output_files,
        records=records,
        usable_records=usable_records,
        dataset_report=dataset_report,
        backend=backend,
        model=model,
        cv=resolved_cv,
        num_folds=resolved_num_folds,
        seed=seed,
        alpha=alpha,
        classes=classes,
        matrix_data=matrix_data,
        predictions=predictions,
        metrics=metrics,
        binary_metrics=binary_metrics,
        majority_baseline=majority_baseline,
        feature_signal_summary=feature_signal_summary,
        warnings=cv_warnings,
    )
    model_bundle = build_model_bundle(
        backend=backend,
        model=model,
        classes=classes,
        matrix_data=matrix_data,
        final_model=final_model,
        final_mean=final_mean,
        final_std=final_std,
        usable_records=usable_records,
        dataset_path=dataset_path,
        dataset_report_path=dataset_report_path,
        alpha=alpha,
        seed=seed,
    )
    feature_index = {name: index for index, name in enumerate(matrix_data["feature_names"])}

    write_jsonl_strict(predictions, output_files["probe_predictions"])
    write_json_strict(report, output_files["probe_eval_report"])
    write_json_strict(feature_index, output_files["probe_feature_index"])
    save_probe_model(model_bundle, output_files["probe_model"])

    return {
        "predictions": predictions,
        "report": report,
        "model_bundle": model_bundle,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }


def load_probe_dataset(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"probe dataset is empty: {path}")
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"probe dataset line {line_number} must be a JSON object")
        for field in ["probe_record_id", "probe_target", "probe_target_usable"]:
            if field not in record:
                raise ValueError(f"probe dataset line {line_number} missing required field: {field}")
    return records


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON input: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must be an object: {json_path}")
    return data


def select_usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if bool(record.get("probe_target_usable")) and record.get("probe_target") != "unmapped"
    ]


def build_feature_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_feature_names = build_feature_name_space(records)
    feature_names: list[str] = []
    for name in base_feature_names:
        feature_names.append(name)
        feature_names.append(f"{name}__is_missing")

    matrix = np.zeros((len(records), len(feature_names)), dtype=float)
    per_record_null_feature_names: list[list[str]] = []
    for row_index, record in enumerate(records):
        values, missing = flatten_probe_record(record, base_feature_names)
        per_record_null_feature_names.append(sorted(name for name in base_feature_names if missing[name]))
        for base_index, name in enumerate(base_feature_names):
            matrix[row_index, base_index * 2] = values[name]
            matrix[row_index, base_index * 2 + 1] = 1.0 if missing[name] else 0.0
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "base_feature_names": base_feature_names,
        "per_record_null_feature_names": per_record_null_feature_names,
    }


def build_feature_name_space(records: list[dict[str, Any]]) -> list[str]:
    feature_names: set[str] = set()
    for record in records:
        for key in record.get("feature_values", {}):
            feature_names.add(str(key))
        layer_indices = record.get("layer_indices")
        for key, value in record.get("feature_arrays", {}).items():
            for name in expand_array_feature_names(str(key), value, layer_indices):
                feature_names.add(name)
    return sorted(feature_names)


def flatten_probe_record(
    record: dict[str, Any],
    base_feature_names: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, bool]]:
    feature_map: dict[str, Any] = {}
    feature_map.update(record.get("feature_values", {}))
    layer_indices = record.get("layer_indices")
    for key, value in record.get("feature_arrays", {}).items():
        feature_map.update(expand_array_feature_values(str(key), value, layer_indices))

    if base_feature_names is None:
        base_feature_names = sorted(feature_map)

    values: dict[str, float] = {}
    missing: dict[str, bool] = {}
    for name in base_feature_names:
        raw_value = feature_map.get(name)
        if is_missing_numeric(raw_value):
            values[name] = 0.0
            missing[name] = True
        else:
            values[name] = float(raw_value)
            missing[name] = False
    return values, missing


def expand_array_feature_names(
    key: str,
    value: Any,
    layer_indices: Any,
) -> list[str]:
    stem = key.removesuffix("_by_layer")
    if isinstance(value, list):
        layer_names = resolve_layer_names(len(value), layer_indices)
        return [f"{stem}_layer_{layer}" for layer in layer_names]
    if value is None and isinstance(layer_indices, list) and layer_indices:
        return [f"{stem}_layer_{layer}" for layer in layer_indices]
    return [f"{stem}_layer_0"]


def expand_array_feature_values(
    key: str,
    value: Any,
    layer_indices: Any,
) -> dict[str, Any]:
    names = expand_array_feature_names(key, value, layer_indices)
    if isinstance(value, list):
        return {name: value[index] if index < len(value) else None for index, name in enumerate(names)}
    return {name: None for name in names}


def resolve_layer_names(length: int, layer_indices: Any) -> list[Any]:
    if isinstance(layer_indices, list) and len(layer_indices) == length:
        return layer_indices
    return list(range(length))


def is_missing_numeric(value: Any) -> bool:
    if value is None:
        return True
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(parsed)


def make_cv_folds(
    labels: list[str],
    *,
    cv: str,
    num_folds: int | None,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], str, int]:
    if cv == "leave_one_out":
        folds = []
        all_indices = np.arange(len(labels))
        for held_out in all_indices:
            test_indices = np.array([held_out], dtype=int)
            train_indices = np.array([index for index in all_indices if index != held_out], dtype=int)
            folds.append((train_indices, test_indices))
        return folds, [], "leave_one_out", len(folds)

    requested_folds = int(num_folds or 2)
    if requested_folds < 2:
        raise ValueError("num_folds must be >= 2 for stratified_k_fold")
    target_counts = Counter(labels)
    min_class_count = min(target_counts.values())
    warnings: list[str] = []
    resolved_folds = requested_folds
    if requested_folds > min_class_count:
        resolved_folds = min_class_count
        warnings.append(
            f"num_folds reduced from {requested_folds} to {resolved_folds} due to min class count"
        )
    if resolved_folds < 2:
        raise ValueError("stratified_k_fold requires every class to have at least 2 records")
    return (
        make_stratified_k_folds(labels, resolved_folds, seed=seed),
        warnings,
        "stratified_k_fold",
        resolved_folds,
    )


def make_stratified_k_folds(
    labels: list[str],
    num_folds: int,
    *,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)

    fold_tests: list[list[int]] = [[] for _ in range(num_folds)]
    for label in sorted(by_label):
        indices = np.array(by_label[label], dtype=int)
        rng.shuffle(indices)
        for offset, index in enumerate(indices.tolist()):
            fold_tests[offset % num_folds].append(index)

    all_indices = set(range(len(labels)))
    folds = []
    for test_list in fold_tests:
        test_indices = np.array(sorted(test_list), dtype=int)
        train_indices = np.array(sorted(all_indices.difference(test_list)), dtype=int)
        folds.append((train_indices, test_indices))
    return folds


def standardize_train_and_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def standardize_full_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (matrix - mean) / std, mean, std


def train_ridge_classifier_ovr(
    matrix: np.ndarray,
    labels: list[str],
    classes: list[str],
    *,
    alpha: float,
) -> dict[str, Any]:
    x_with_bias = np.column_stack([matrix, np.ones(matrix.shape[0])])
    gram = x_with_bias.T @ x_with_bias
    regularizer = np.eye(gram.shape[0]) * alpha
    regularizer[-1, -1] = 0.0
    inverse = np.linalg.pinv(gram + regularizer)
    weights = []
    for label in classes:
        y = np.array([1.0 if item == label else -1.0 for item in labels], dtype=float)
        weights.append(inverse @ x_with_bias.T @ y)
    weights_array = np.vstack(weights)
    return {
        "classes": classes,
        "weights": weights_array[:, :-1],
        "bias": weights_array[:, -1],
        "alpha": alpha,
    }


def decision_function(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return matrix @ model["weights"].T + model["bias"]


def ordered_classes(labels: list[str]) -> list[str]:
    observed = set(labels)
    ordered = [label for label in TARGET_LABELS if label in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def build_prediction_record(
    record: dict[str, Any],
    *,
    cv_strategy: str,
    fold_id: int,
    train_size: int,
    test_size: int,
    gold: str,
    predicted: str,
    decision_scores: dict[str, float | None],
    num_features: int,
) -> dict[str, Any]:
    gold_binary = gold in POSITIVE_BINARY_TARGETS
    predicted_binary = predicted in POSITIVE_BINARY_TARGETS
    return sanitize_json_value(
        {
            "probe_record_id": record.get("probe_record_id"),
            "feature_id": record.get("feature_id"),
            "masked_id": record.get("masked_id"),
            "id": record.get("id"),
            "unit_id": record.get("unit_id"),
            "cv_strategy": cv_strategy,
            "fold_id": fold_id,
            "train_size": train_size,
            "test_size": test_size,
            "gold_probe_target": gold,
            "predicted_probe_target": predicted,
            "correct": gold == predicted,
            "decision_scores": decision_scores,
            "gold_anchor_or_risk_binary": gold_binary,
            "predicted_anchor_or_risk_binary": predicted_binary,
            "binary_correct": gold_binary == predicted_binary,
            "num_features": num_features,
            "num_null_features": int(record.get("num_null_features", 0)),
            "warnings": record.get("warnings", []),
        }
    )


def compute_multiclass_metrics(
    gold: list[str],
    predicted: list[str],
    classes: list[str],
) -> dict[str, Any]:
    total = len(gold)
    accuracy = sum(1 for g, p in zip(gold, predicted) if g == p) / total if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    weighted_f1_sum = 0.0
    for label in classes:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        support = sum(1 for g in gold if g == label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support
        per_class[label] = {
            "precision": sanitize_float(precision),
            "recall": sanitize_float(recall),
            "f1": sanitize_float(f1),
            "support": support,
        }
    confusion = {
        gold_label: {
            predicted_label: sum(
                1 for g, p in zip(gold, predicted) if g == gold_label and p == predicted_label
            )
            for predicted_label in classes
        }
        for gold_label in classes
    }
    return {
        "accuracy": sanitize_float(accuracy),
        "macro_f1": sanitize_float(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "weighted_f1": sanitize_float(safe_divide(weighted_f1_sum, total)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def compute_binary_anchor_or_risk_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [bool(prediction["gold_anchor_or_risk_binary"]) for prediction in predictions]
    predicted = [bool(prediction["predicted_anchor_or_risk_binary"]) for prediction in predictions]
    classes = [False, True]
    total = len(gold)
    accuracy = sum(1 for g, p in zip(gold, predicted) if g == p) / total if total else 0.0
    f1_values = []
    per_class: dict[str, dict[str, float | int]] = {}
    for label in classes:
        tp = sum(1 for g, p in zip(gold, predicted) if g is label and p is label)
        fp = sum(1 for g, p in zip(gold, predicted) if g is not label and p is label)
        fn = sum(1 for g, p in zip(gold, predicted) if g is label and p is not label)
        support = sum(1 for g in gold if g is label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[str(label).lower()] = {
            "precision": sanitize_float(precision),
            "recall": sanitize_float(recall),
            "f1": sanitize_float(f1),
            "support": support,
        }
    return {
        "accuracy": sanitize_float(accuracy),
        "macro_f1": sanitize_float(sum(f1_values) / len(f1_values)),
        "positive_definition": sorted(POSITIVE_BINARY_TARGETS),
        "per_class": per_class,
    }


def compute_majority_baseline(labels: list[str], classes: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    majority_label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    predicted = [majority_label for _ in labels]
    metrics = compute_multiclass_metrics(labels, predicted, classes)
    return {
        "label": majority_label,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def summarize_feature_weights(
    model: dict[str, Any],
    feature_names: list[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    weights = model["weights"]
    classes = model["classes"]
    abs_mean = np.mean(np.abs(weights), axis=0)
    top_indices = np.argsort(-abs_mean)[:top_k]
    top_weighted_features = []
    for index in top_indices.tolist():
        top_weighted_features.append(
            {
                "feature_name": feature_names[index],
                "weight_abs_mean": sanitize_float(float(abs_mean[index])),
                "per_class_weights": {
                    label: sanitize_float(float(weights[class_index, index]))
                    for class_index, label in enumerate(classes)
                },
            }
        )
    return {
        "top_weighted_features": top_weighted_features,
        "feature_group_weight_summary": summarize_group_weights(feature_names, abs_mean),
        "layer_weight_summary": summarize_layer_weights(feature_names, abs_mean),
    }


def summarize_group_weights(feature_names: list[str], abs_mean: np.ndarray) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(feature_names, abs_mean):
        groups[feature_group_for_name(name)].append(float(value))
    return {key: sanitize_float(float(np.mean(values))) for key, values in sorted(groups.items())}


def summarize_layer_weights(feature_names: list[str], abs_mean: np.ndarray) -> dict[str, float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(feature_names, abs_mean):
        groups[layer_group_for_name(name)].append(float(value))
    return {key: sanitize_float(float(np.mean(values))) for key, values in sorted(groups.items())}


def feature_group_for_name(name: str) -> str:
    if name.endswith("__is_missing"):
        return "missing_indicator"
    if name.startswith("question_"):
        return "question"
    if name.startswith("span_"):
        return "span"
    if name.startswith("mask_position_"):
        return "mask_position"
    return "other"


def layer_group_for_name(name: str) -> str:
    if name.endswith("__is_missing"):
        return "missing_indicator"
    marker = "_layer_"
    if marker in name:
        suffix = name.split(marker, 1)[1]
        layer = suffix.split("__", 1)[0]
        return f"layer_{layer}"
    return "summary_scalar"


def build_eval_report(
    *,
    dataset_path: Path,
    dataset_report_path: Path,
    output_files: dict[str, Path],
    records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    backend: str,
    model: str,
    cv: str,
    num_folds: int,
    seed: int,
    alpha: float,
    classes: list[str],
    matrix_data: dict[str, Any],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    binary_metrics: dict[str, Any],
    majority_baseline: dict[str, Any],
    feature_signal_summary: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    target_counts = Counter(record["probe_target"] for record in usable_records)
    return sanitize_json_value(
        {
            "sprint": "2D",
            "backend": backend,
            "status": "ok",
            "inputs": {
                "probe_dataset_path": str(dataset_path),
                "probe_dataset_report_path": str(dataset_report_path),
                "source_2C_backend": dataset_report.get("backend"),
            },
            "outputs": {
                "probe_predictions_path": str(output_files["probe_predictions"]),
                "probe_eval_report_path": str(output_files["probe_eval_report"]),
                "probe_model_path": str(output_files["probe_model"]),
                "probe_feature_index_path": str(output_files["probe_feature_index"]),
            },
            "data_summary": {
                "num_records_total": len(records),
                "num_records_usable": len(usable_records),
                "num_records_unusable": len(records) - len(usable_records),
                "target_counts": sorted_counter_dict(target_counts),
                "records_with_null_features": sum(
                    1 for record in usable_records if int(record.get("num_null_features", 0)) > 0
                ),
            },
            "training": {
                "model_type": model,
                "cv_strategy": cv,
                "num_folds": num_folds,
                "seed": seed,
                "alpha": alpha,
                "feature_standardization": STANDARDIZATION,
                "null_feature_strategy": NULL_FEATURE_STRATEGY,
                "num_features": len(matrix_data["feature_names"]),
                "num_base_features": len(matrix_data["base_feature_names"]),
                "classes": classes,
            },
            "metrics": metrics,
            "binary_anchor_or_risk_metrics": binary_metrics,
            "baselines": {
                "majority_class": majority_baseline,
            },
            "feature_signal_summary": feature_signal_summary,
            "small_sample_warning": SMALL_SAMPLE_WARNING,
            "boundary": boundary_report(),
            "warnings": warnings,
        }
    )


def build_insufficient_data_report(
    *,
    dataset_path: Path,
    dataset_report_path: Path,
    output_files: dict[str, Path],
    records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    backend: str,
    model: str,
    cv: str,
    seed: int,
) -> dict[str, Any]:
    target_counts = Counter(record.get("probe_target") for record in usable_records)
    return {
        "sprint": "2D",
        "backend": backend,
        "status": "insufficient_data",
        "inputs": {
            "probe_dataset_path": str(dataset_path),
            "probe_dataset_report_path": str(dataset_report_path),
            "source_2C_backend": dataset_report.get("backend"),
        },
        "outputs": {
            "probe_predictions_path": None,
            "probe_eval_report_path": str(output_files["probe_eval_report"]),
            "probe_model_path": None,
        },
        "data_summary": {
            "num_records_total": len(records),
            "num_records_usable": len(usable_records),
            "target_counts": sorted_counter_dict(target_counts),
            "records_with_null_features": sum(
                1 for record in usable_records if int(record.get("num_null_features", 0)) > 0
            ),
        },
        "training": {
            "model_type": model,
            "cv_strategy": cv,
            "seed": seed,
            "feature_standardization": STANDARDIZATION,
            "null_feature_strategy": NULL_FEATURE_STRATEGY,
        },
        "metrics": {},
        "binary_anchor_or_risk_metrics": {},
        "baselines": {},
        "feature_signal_summary": {
            "top_weighted_features": [],
            "feature_group_weight_summary": {},
            "layer_weight_summary": {},
        },
        "small_sample_warning": SMALL_SAMPLE_WARNING,
        "boundary": boundary_report(),
        "warnings": [
            "usable records < 4 or effective class count < 2; probe was not trained",
        ],
    }


def build_model_bundle(
    *,
    backend: str,
    model: str,
    classes: list[str],
    matrix_data: dict[str, Any],
    final_model: dict[str, Any],
    final_mean: np.ndarray,
    final_std: np.ndarray,
    usable_records: list[dict[str, Any]],
    dataset_path: Path,
    dataset_report_path: Path,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "model_type": model,
        "classes": classes,
        "feature_names": matrix_data["feature_names"],
        "scaler": {
            "mean": final_mean.tolist(),
            "std": final_std.tolist(),
        },
        "model_parameters": {
            "weights": final_model["weights"].tolist(),
            "bias": final_model["bias"].tolist(),
            "alpha": alpha,
            "seed": seed,
        },
        "null_feature_strategy": NULL_FEATURE_STRATEGY,
        "feature_standardization": "full_train_zscore",
        "training_record_ids": [record.get("probe_record_id") for record in usable_records],
        "created_from": {
            "probe_dataset_path": str(dataset_path),
            "probe_dataset_report_path": str(dataset_report_path),
        },
    }


def boundary_report() -> dict[str, bool]:
    return {
        "rebuilt_probe_dataset": False,
        "read_hidden_state_tensors": False,
        "recomputed_representation_features": False,
        "read_2B_representation_features": False,
        "generated_guidance_candidates": False,
        "performed_attention_guidance": False,
        "claimed_hallucination_reduction": False,
    }


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    if overwrite:
        for filename in [
            PREDICTIONS_FILENAME,
            REPORT_FILENAME,
            MODEL_FILENAME,
            FEATURE_INDEX_FILENAME,
        ]:
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
    ]
    for forbidden_root in forbidden_roots:
        forbidden_resolved = forbidden_root.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"Refusing to write Sprint 2D outputs under forbidden path: {output_dir}")


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


def save_probe_model(model_bundle: dict[str, Any], path: str | Path) -> None:
    model_path = Path(path)
    ensure_dir(model_path.parent)
    with model_path.open("wb") as handle:
        pickle.dump(model_bundle, handle)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def sanitize_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return sanitize_float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return sanitize_float(float(value))
    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    return value


def sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}
