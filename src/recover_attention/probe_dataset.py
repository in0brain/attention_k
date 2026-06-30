"""Sprint 2C probe dataset construction from representation features."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "probe_dataset_mapping_v0"
DATASET_FILENAME = "probe_dataset.jsonl"
REPORT_FILENAME = "probe_dataset_report.json"

REQUIRED_FEATURE_ARRAY_KEYS = [
    "question_original_masked_cosine_by_layer",
    "question_original_recovered_cosine_by_layer",
    "question_masked_recovered_cosine_by_layer",
    "span_original_masked_cosine_by_layer",
    "span_original_recovered_cosine_by_layer",
    "span_masked_recovered_cosine_by_layer",
    "mask_position_original_masked_cosine_by_layer",
    "mask_position_original_recovered_cosine_by_layer",
    "mask_position_masked_recovered_cosine_by_layer",
]
SCALAR_FEATURE_SUFFIXES = (
    "_mean",
    "_max",
    "_min",
    "_first_layer",
    "_last_layer",
    "_delta_last_minus_first",
)
HUMAN_METADATA_FIELDS = [
    "human_attention_anchor_label",
    "human_attention_anchor_label_name",
    "human_recoverability_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
    "probe_usage",
]
REQUIRED_HUMAN_MAPPING_FIELDS = [
    "human_attention_anchor_label",
    "human_recoverability_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
]
DEPRECATED_2B_INPUT_FILENAMES = [
    "representation_feature_manifest.jsonl",
    "input_representation_summary.jsonl",
    "feature_schema.json",
]
FORBIDDEN_OUTPUT_FILENAMES = [
    "probe_predictions.jsonl",
    "probe_eval_report.json",
    "probe_model.pkl",
    "guidance_candidate_manifest.jsonl",
    "guidance_candidate_report.json",
]


def build_probe_dataset(
    *,
    features_path: str | Path,
    feature_report_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    overwrite: bool = False,
    human_labels_path: str | Path | None = None,
    sprint_1r_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build Sprint 2C probe-ready records from Sprint 2B representation features."""

    if backend != BACKEND:
        raise ValueError(f"Unsupported probe dataset backend {backend!r}; expected {BACKEND!r}")

    features_path = Path(features_path)
    feature_report_path = Path(feature_report_path)
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite)

    feature_records = load_feature_records(features_path)
    feature_report = load_json(feature_report_path)
    human_index = load_optional_metadata_index(human_labels_path)
    manifest_index = load_optional_metadata_index(sprint_1r_manifest_path)

    dataset_records: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    human_error_type_counts: Counter[str] = Counter()
    probe_usage_counts: Counter[str] = Counter()
    null_feature_key_counts: Counter[str] = Counter()
    records_with_any_null_feature = 0
    records_with_null_position_features = 0

    for line_number, feature_record in enumerate(feature_records, start=1):
        merged = merge_optional_metadata(feature_record, human_index, manifest_index)
        dataset_record = build_probe_dataset_record(
            merged,
            source_line_number=line_number,
            source_features_path=features_path,
        )
        dataset_records.append(dataset_record)

        target_counts[dataset_record["probe_target"]] += 1
        status_counts[dataset_record["target_mapping_status"]] += 1
        rule_counts[dataset_record["target_mapping_rule"]] += 1
        human_error_type_counts[normalize_or_missing(merged.get("human_error_type"))] += 1
        probe_usage_counts[normalize_or_missing(merged.get("probe_usage"))] += 1
        if dataset_record["num_null_features"]:
            records_with_any_null_feature += 1
        if dataset_record["has_null_position_features"]:
            records_with_null_position_features += 1
        for key in dataset_record["null_feature_keys"]:
            null_feature_key_counts[key] += 1

    output_files = {
        "probe_dataset": output_dir / DATASET_FILENAME,
        "probe_dataset_report": output_dir / REPORT_FILENAME,
    }
    report = build_report(
        backend=backend,
        features_path=features_path,
        feature_report_path=feature_report_path,
        output_dir=output_dir,
        output_files=output_files,
        feature_report=feature_report,
        dataset_records=dataset_records,
        target_counts=target_counts,
        status_counts=status_counts,
        rule_counts=rule_counts,
        human_error_type_counts=human_error_type_counts,
        probe_usage_counts=probe_usage_counts,
        records_with_any_null_feature=records_with_any_null_feature,
        records_with_null_position_features=records_with_null_position_features,
        null_feature_key_counts=null_feature_key_counts,
        human_labels_path=human_labels_path,
        sprint_1r_manifest_path=sprint_1r_manifest_path,
    )

    write_jsonl_strict(dataset_records, output_files["probe_dataset"])
    write_json_strict(report, output_files["probe_dataset_report"])

    return {
        "dataset_records": dataset_records,
        "report": report,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }


def build_probe_dataset_record(
    feature_record: dict[str, Any],
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
    mapping = map_probe_target(feature_record)
    warnings = sorted(
        set(
            str(warning)
            for warning in feature_record.get("warnings", [])
            if warning is not None
        )
    )
    if null_feature_keys:
        warnings.append("null representation feature(s) retained without imputation")

    return sanitize_json_value(
        {
            "probe_record_id": feature_record.get("feature_id")
            or f"{feature_record.get('masked_id')}::probe::{source_line_number}",
            "feature_id": feature_record.get("feature_id"),
            "masked_id": feature_record.get("masked_id"),
            "id": feature_record.get("id"),
            "unit_id": feature_record.get("unit_id"),
            "recovered_input_index": feature_record.get("recovered_input_index"),
            "original_cache_id": feature_record.get("original_cache_id"),
            "masked_cache_id": feature_record.get("masked_cache_id"),
            "recovered_cache_id": feature_record.get("recovered_cache_id"),
            "source_feature_backend": feature_record.get("backend"),
            "source_cache_backend": feature_record.get("source_cache_backend"),
            "model_name": feature_record.get("model_name"),
            "tokenizer_name": feature_record.get("tokenizer_name"),
            "layer_indices": feature_record.get("layer_indices"),
            "hidden_size": feature_record.get("hidden_size"),
            "source_features_path": str(source_features_path),
            "source_features_line": source_line_number,
            "probe_target": mapping["probe_target"],
            "probe_target_usable": mapping["probe_target_usable"],
            "target_mapping_status": mapping["target_mapping_status"],
            "target_mapping_rule": mapping["target_mapping_rule"],
            "target_mapping_reason": mapping["target_mapping_reason"],
            "human_metadata": extract_human_metadata(feature_record),
            "feature_values": feature_values,
            "feature_arrays": feature_arrays,
            "null_feature_keys": null_feature_keys,
            "num_null_features": len(null_feature_keys),
            "has_null_position_features": has_null_position_features(null_feature_keys),
            "warnings": warnings,
        }
    )


def map_probe_target(record: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [
        field for field in REQUIRED_HUMAN_MAPPING_FIELDS if is_missing(record.get(field))
    ]
    if missing_fields:
        return mapping_result(
            target="unmapped",
            usable=False,
            status="missing_human_metadata",
            rule="missing_human_metadata",
            reason=f"missing required human metadata field(s): {', '.join(missing_fields)}",
        )

    error_type = normalize(record.get("human_error_type"))
    semantic_role = normalize(record.get("human_semantic_role"))
    guidance_priority = normalize(record.get("human_guidance_priority"))
    recoverability = normalize(record.get("human_recoverability_label"))
    anchor = normalize(record.get("human_attention_anchor_label"))

    if error_type in {"wrong_numeric_recovery", "misleading_entity_or_unit"}:
        result = mapping_result(
            target="risk_positive",
            usable=True,
            status="mapped",
            rule="human_error_type_risk_positive",
            reason=f"human_error_type={record.get('human_error_type')!r} maps to risk_positive",
        )
    elif error_type == "generic_recovery" and is_critical_number_role(semantic_role):
        result = mapping_result(
            target="positive_anchor",
            usable=True,
            status="mapped",
            rule="generic_recovery_critical_number",
            reason="generic_recovery on a critical numeric role maps to positive_anchor",
        )
    elif error_type == "generic_recovery":
        result = mapping_result(
            target="hard_negative_or_weak_positive",
            usable=True,
            status="mapped",
            rule="generic_recovery_noncritical",
            reason="generic_recovery outside critical numeric roles maps conservatively",
        )
    elif error_type == "semantic_equivalent_recovery":
        result = mapping_result(
            target="negative",
            usable=True,
            status="mapped",
            rule="semantic_equivalent_recovery",
            reason="semantic_equivalent_recovery maps to negative",
        )
    elif error_type == "fragment_recovery" and guidance_priority in {"low", "none"}:
        result = mapping_result(
            target="negative",
            usable=True,
            status="mapped",
            rule="fragment_recovery_low_priority",
            reason="fragment_recovery with low/none guidance priority maps to negative",
        )
    elif error_type in {
        "semantically_meaningful_but_recoverable",
        "recoverable_but_semantically_meaningful_span",
    }:
        result = mapping_result(
            target="hard_negative_or_weak_positive",
            usable=True,
            status="mapped",
            rule="meaningful_but_recoverable",
            reason=(
                "semantically meaningful but recoverable human_error_type maps to "
                "hard_negative_or_weak_positive"
            ),
        )
    elif (
        recoverability in {"recoverable", "partially_recoverable"}
        and anchor in {"weak_anchor", "medium_anchor"}
        and guidance_priority in {"low", "medium"}
    ):
        result = mapping_result(
            target="hard_negative_or_weak_positive",
            usable=True,
            status="mapped",
            rule="recoverable_weak_or_medium_anchor",
            reason=(
                "recoverable/partially recoverable weak-medium anchor with low/medium "
                "priority maps conservatively"
            ),
        )
    else:
        return mapping_result(
            target="unmapped",
            usable=False,
            status="unmapped",
            rule="no_matching_mapping_rule",
            reason=(
                "human metadata did not match Sprint 2C probe target mapping rules; "
                "record retained as unmapped"
            ),
        )

    return apply_probe_usage_override(result, record.get("probe_usage"))


def apply_probe_usage_override(result: dict[str, Any], probe_usage: Any) -> dict[str, Any]:
    normalized_usage = normalize(probe_usage)
    if normalized_usage == "exclude":
        updated = dict(result)
        updated["probe_target_usable"] = False
        updated["target_mapping_status"] = "excluded_by_probe_usage"
        updated["target_mapping_reason"] = (
            f"{result['target_mapping_reason']}; probe_usage='exclude' marks record unusable"
        )
        return updated
    if normalized_usage == "uncertain":
        updated = dict(result)
        updated["probe_target_usable"] = False
        updated["target_mapping_status"] = "uncertain_probe_usage"
        updated["target_mapping_reason"] = (
            f"{result['target_mapping_reason']}; probe_usage='uncertain' marks record unusable"
        )
        return updated
    return result


def mapping_result(
    *,
    target: str,
    usable: bool,
    status: str,
    rule: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "probe_target": target,
        "probe_target_usable": bool(usable),
        "target_mapping_status": status,
        "target_mapping_rule": rule,
        "target_mapping_reason": reason,
    }


def extract_feature_arrays(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sanitize_json_value(record.get(key))
        for key in REQUIRED_FEATURE_ARRAY_KEYS
        if key in record
    }


def extract_feature_values(record: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in sorted(record):
        if key.endswith(SCALAR_FEATURE_SUFFIXES):
            values[key] = sanitize_json_value(record.get(key))
    return values


def extract_human_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {field: sanitize_json_value(record.get(field)) for field in HUMAN_METADATA_FIELDS}


def merge_optional_metadata(
    feature_record: dict[str, Any],
    human_index: dict[str, dict[str, Any]],
    manifest_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(feature_record)
    lookup_keys = [feature_record.get("masked_id"), feature_record.get("feature_id")]
    for index in [human_index, manifest_index]:
        for key in lookup_keys:
            if key is None or str(key) not in index:
                continue
            for field in HUMAN_METADATA_FIELDS:
                if is_missing(merged.get(field)) and not is_missing(index[str(key)].get(field)):
                    merged[field] = index[str(key)][field]
    return merged


def load_optional_metadata_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = read_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        keys = [
            record.get("masked_id"),
            record.get("feature_id"),
            record.get("id"),
        ]
        for key in keys:
            if key is not None:
                index[str(key)] = record
    return index


def load_feature_records(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"representation features input is empty: {path}")
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"feature line {line_number} must be a JSON object")
        if "feature_id" not in record:
            raise ValueError(f"feature line {line_number} missing required field: feature_id")
        if "masked_id" not in record:
            raise ValueError(f"feature line {line_number} missing required field: masked_id")
    return records


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON input: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must be an object: {json_path}")
    return data


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    if overwrite:
        for filename in [DATASET_FILENAME, REPORT_FILENAME]:
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
    ]
    for forbidden_root in forbidden_roots:
        forbidden_resolved = forbidden_root.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"Refusing to write Sprint 2C outputs under forbidden path: {output_dir}")


def build_report(
    *,
    backend: str,
    features_path: Path,
    feature_report_path: Path,
    output_dir: Path,
    output_files: dict[str, Path],
    feature_report: dict[str, Any],
    dataset_records: list[dict[str, Any]],
    target_counts: Counter[str],
    status_counts: Counter[str],
    rule_counts: Counter[str],
    human_error_type_counts: Counter[str],
    probe_usage_counts: Counter[str],
    records_with_any_null_feature: int,
    records_with_null_position_features: int,
    null_feature_key_counts: Counter[str],
    human_labels_path: str | Path | None,
    sprint_1r_manifest_path: str | Path | None,
) -> dict[str, Any]:
    usable_count = sum(1 for record in dataset_records if record["probe_target_usable"])
    unusable_count = len(dataset_records) - usable_count
    deprecated_inputs_present = [
        str(features_path.parent / filename)
        for filename in DEPRECATED_2B_INPUT_FILENAMES
        if (features_path.parent / filename).exists()
    ]
    return {
        "sprint": "2C",
        "backend": backend,
        "status": "ok",
        "inputs": {
            "representation_features_path": str(features_path),
            "representation_feature_report_path": str(feature_report_path),
            "human_labels_path": str(human_labels_path) if human_labels_path is not None else None,
            "sprint_1r_manifest_path": (
                str(sprint_1r_manifest_path) if sprint_1r_manifest_path is not None else None
            ),
            "deprecated_2B_inputs_present_but_not_read": deprecated_inputs_present,
            "hidden_state_tensors_read": False,
        },
        "outputs": {
            "probe_dataset_path": str(output_files["probe_dataset"]),
            "probe_dataset_report_path": str(output_files["probe_dataset_report"]),
            "forbidden_outputs_generated": [],
        },
        "source_2B": {
            "backend": feature_report.get("backend"),
            "sprint": feature_report.get("sprint"),
            "counts": feature_report.get("counts", {}),
            "num_feature_records": feature_report.get("counts", {}).get("num_feature_records"),
        },
        "counts": {
            "num_probe_dataset_records": len(dataset_records),
            "num_probe_records": len(dataset_records),
            "num_usable_records": usable_count,
            "num_unusable_records": unusable_count,
            "num_probe_target_usable": usable_count,
            "num_probe_target_unusable": unusable_count,
            "num_unmapped": int(target_counts.get("unmapped", 0)),
            "num_missing_human_metadata": int(status_counts.get("missing_human_metadata", 0)),
        },
        "target_counts": sorted_counter_dict(target_counts),
        "target_mapping_status_counts": sorted_counter_dict(status_counts),
        "target_mapping_rule_counts": sorted_counter_dict(rule_counts),
        "human_error_type_counts": sorted_counter_dict(human_error_type_counts),
        "probe_usage_counts": sorted_counter_dict(probe_usage_counts),
        "null_feature_counts": {
            "records_with_any_null_feature": records_with_any_null_feature,
            "records_with_null_position_features": records_with_null_position_features,
            "null_feature_key_counts": sorted_counter_dict(null_feature_key_counts),
        },
        "feature_scope": {
            "representation_feature_extraction": False,
            "probe_dataset_construction": True,
            "target_selection": False,
            "train_dev_test_split": False,
            "k_fold_cross_validation": False,
            "probe_training": False,
            "probe_prediction": False,
            "feature_importance_analysis": False,
            "guidance_candidate_manifest": False,
            "attention_steering": False,
        },
        "boundary_checks": {
            "read_hidden_state_tensors": False,
            "read_legacy_2B_inputs": False,
            "created_train_dev_test_split": False,
            "created_k_fold_split": False,
            "trained_probe": False,
            "generated_probe_predictions": False,
            "generated_probe_eval_report": False,
            "generated_probe_model": False,
            "generated_guidance_candidates": False,
            "performed_attention_guidance": False,
        },
    }


def sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, list):
        return any(contains_null(item) for item in value)
    if isinstance(value, dict):
        return any(contains_null(item) for item in value.values())
    return False


def has_null_position_features(null_feature_keys: list[str]) -> bool:
    return any(key.startswith("span_") or key.startswith("mask_position_") for key in null_feature_keys)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def is_critical_number_role(normalized_role: str) -> bool:
    return normalized_role in {
        "critical_number",
        "key_number",
        "important_number",
        "number",
    }


def normalize_or_missing(value: Any) -> str:
    normalized = normalize(value)
    return normalized if normalized else "__missing__"


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
