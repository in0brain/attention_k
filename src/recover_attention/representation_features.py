"""Sprint 2B representation feature extraction from cached hidden states."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "representation_features_minimal_v0"
LEGACY_BACKEND = "representation_features_v0"
FEATURES_FILENAME = "representation_features.jsonl"
REPORT_FILENAME = "representation_feature_report.json"
DEPRECATED_EXTRA_FILENAMES = [
    "representation_feature_manifest.jsonl",
    "input_representation_summary.jsonl",
    "feature_schema.json",
]

INPUT_TYPES = {"original", "masked", "recovered"}
DISTANCE_PAIRS = {
    "original_masked": ("original", "masked"),
    "original_recovered": ("original", "recovered"),
    "masked_recovered": ("masked", "recovered"),
}
POOLING_NAMES = ["question", "span", "mask_position"]
SUMMARY_SUFFIXES = ["mean", "max", "min", "first_layer", "last_layer", "delta_last_minus_first"]
EPSILON = 1e-8


def extract_representation_features(
    *,
    input_manifest_path: str | Path,
    input_report_path: str | Path,
    alignment_report_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    eps: float = EPSILON,
    overwrite: bool = False,
    tensor_loader: Callable[[str | Path], Any] | None = None,
) -> dict[str, Any]:
    """Extract case-wise representation features from a Sprint 2A cache."""

    if backend not in {BACKEND, LEGACY_BACKEND}:
        raise ValueError(f"Unsupported representation backend {backend!r}; expected {BACKEND!r}")
    backend = BACKEND
    if eps <= 0:
        raise ValueError("eps must be > 0")

    input_manifest_path = Path(input_manifest_path)
    input_report_path = Path(input_report_path)
    alignment_report_path = Path(alignment_report_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)

    prepare_output_dir(output_dir, overwrite=overwrite)

    manifest_records = load_hidden_state_manifest(input_manifest_path)
    source_report = load_json(input_report_path)
    alignment_report = load_json(alignment_report_path)
    metadata = load_json(metadata_path)
    grouped = group_manifest_by_masked_id(manifest_records)
    tensor_loader = tensor_loader or load_tensor_cpu

    input_summaries: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    skipped_groups = 0
    skipped_recovered_variants = 0
    recovered_variant_count = 0

    for masked_id, group_records in grouped.items():
        records_by_type = split_records_by_input_type(group_records)
        original_record = first_by_input_index(records_by_type.get("original", []))
        masked_record = first_by_input_index(records_by_type.get("masked", []))
        recovered_records = sorted(records_by_type.get("recovered", []), key=input_index_key)

        if original_record is None:
            skipped_groups += 1
            add_warning(
                warnings,
                warning_counts,
                "missing_original_input",
                f"masked_id {masked_id!r} has no original input record",
                masked_id=masked_id,
            )
            for record in group_records:
                summary = summarize_unloaded_record(record, ["group skipped: missing original input"])
                input_summaries.append(summary)
            continue
        if masked_record is None:
            skipped_groups += 1
            add_warning(
                warnings,
                warning_counts,
                "missing_masked_input",
                f"masked_id {masked_id!r} has no masked input record",
                masked_id=masked_id,
            )
            for record in group_records:
                summary = summarize_unloaded_record(record, ["group skipped: missing masked input"])
                input_summaries.append(summary)
            continue
        if not recovered_records:
            add_warning(
                warnings,
                warning_counts,
                "missing_recovered_input",
                f"masked_id {masked_id!r} has no recovered input record",
                masked_id=masked_id,
            )

        original = load_and_summarize_input(
            original_record,
            tensor_loader=tensor_loader,
            warning_counts=warning_counts,
        )
        masked = load_and_summarize_input(
            masked_record,
            tensor_loader=tensor_loader,
            warning_counts=warning_counts,
        )
        input_summaries.extend([original["summary"], masked["summary"]])

        if not original["valid"] or not masked["valid"]:
            skipped_groups += 1
            add_warning(
                warnings,
                warning_counts,
                "invalid_required_input_tensor",
                f"masked_id {masked_id!r} has invalid original or masked tensor",
                masked_id=masked_id,
            )
            for recovered_record in recovered_records:
                recovered = load_and_summarize_input(
                    recovered_record,
                    tensor_loader=tensor_loader,
                    warning_counts=warning_counts,
                )
                input_summaries.append(recovered["summary"])
                skipped_recovered_variants += 1
            continue

        for recovered_record in recovered_records:
            recovered_variant_count += 1
            recovered = load_and_summarize_input(
                recovered_record,
                tensor_loader=tensor_loader,
                warning_counts=warning_counts,
            )
            input_summaries.append(recovered["summary"])
            if not recovered["valid"]:
                skipped_recovered_variants += 1
                add_warning(
                    warnings,
                    warning_counts,
                    "invalid_recovered_tensor",
                    f"recovered cache_id {recovered_record.get('cache_id')} is invalid",
                    masked_id=masked_id,
                    cache_id=recovered_record.get("cache_id"),
                )
                continue

            feature_record = build_feature_record(
                original=original,
                masked=masked,
                recovered=recovered,
                backend=backend,
                eps=eps,
                warning_counts=warning_counts,
            )
            if feature_record is None:
                skipped_recovered_variants += 1
                add_warning(
                    warnings,
                    warning_counts,
                    "incompatible_feature_tensors",
                    f"masked_id {masked_id!r} has incompatible pooled tensor shapes",
                    masked_id=masked_id,
                )
                continue
            feature_records.append(feature_record)
            if any_position_pool_null(feature_record):
                null_counts["position_pool_feature_null_records"] += 1
            if any_cosine_null(feature_record):
                null_counts["cosine_near_zero_null_records"] += 1

    output_files = {
        "representation_features": output_dir / FEATURES_FILENAME,
        "representation_feature_report": output_dir / REPORT_FILENAME,
    }
    report = build_report(
        backend=backend,
        input_manifest_path=input_manifest_path,
        input_report_path=input_report_path,
        alignment_report_path=alignment_report_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        output_files=output_files,
        manifest_records=manifest_records,
        grouped=grouped,
        feature_records=feature_records,
        recovered_variant_count=recovered_variant_count,
        skipped_groups=skipped_groups,
        skipped_recovered_variants=skipped_recovered_variants,
        source_report=source_report,
        alignment_report=alignment_report,
        metadata=metadata,
        null_counts=null_counts,
        warning_counts=warning_counts,
        warnings=warnings,
    )

    write_jsonl_strict(feature_records, output_files["representation_features"])
    write_json_strict(report, output_files["representation_feature_report"])

    return {
        "feature_records": feature_records,
        "input_summaries": input_summaries,
        "report": report,
        "output_files": {key: str(path) for key, path in output_files.items()},
    }


def load_hidden_state_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"hidden-state manifest is empty: {path}")
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        missing = [
            field
            for field in [
                "cache_id",
                "masked_id",
                "input_type",
                "input_index",
                "hidden_state_path",
                "hidden_state_shape",
            ]
            if field not in record
        ]
        if missing:
            raise ValueError(
                f"manifest line {line_number} missing required field(s): {', '.join(missing)}"
            )
        if record["input_type"] not in INPUT_TYPES:
            raise ValueError(
                f"manifest line {line_number} has unsupported input_type: {record['input_type']!r}"
            )
    return records


def group_manifest_by_masked_id(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["masked_id"])].append(record)
    return dict(grouped)


def load_tensor_cpu(path: str | Path) -> Any:
    """Load a cached hidden-state tensor on CPU only."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to load hidden-state tensors") from exc

    return torch.load(path, map_location="cpu")


def pool_question(tensor: Any) -> Any:
    return tensor.mean(dim=1)


def pool_span(tensor: Any, token_indices: list[int] | None) -> Any | None:
    if not token_indices:
        return None
    return tensor[:, token_indices, :].mean(dim=1)


def compute_cosine_distance(first: Any, second: Any, *, eps: float = EPSILON) -> list[float | None]:
    values: list[float | None] = []
    for first_layer, second_layer in zip(first, second):
        first_norm = float(first_layer.norm().item())
        second_norm = float(second_layer.norm().item())
        if first_norm <= eps or second_norm <= eps:
            values.append(None)
            continue
        similarity = float((first_layer * second_layer).sum().item() / (first_norm * second_norm))
        similarity = max(-1.0, min(1.0, similarity))
        values.append(sanitize_float(1.0 - similarity))
    return values


def load_and_summarize_input(
    record: dict[str, Any],
    *,
    tensor_loader: Callable[[str | Path], Any],
    warning_counts: Counter[str],
) -> dict[str, Any]:
    warnings: list[str] = []
    tensor_path = Path(record["hidden_state_path"])
    if not tensor_path.exists():
        warning_counts["missing_tensor_file"] += 1
        warnings.append(f"missing tensor file: {tensor_path}")
        return {
            "record": record,
            "summary": build_input_summary(
                record,
                hidden_state_shape=record.get("hidden_state_shape"),
                question_norms=None,
                span_pool_available=False,
                mask_position_available=False,
                span_token_indices=None,
                nonfinite_count=0,
                warnings=warnings,
            ),
            "valid": False,
            "pools": {},
        }

    try:
        tensor = tensor_loader(tensor_path)
    except Exception as exc:  # pragma: no cover - exact torch errors vary by version
        warning_counts["tensor_load_error"] += 1
        warnings.append(f"failed to load tensor {tensor_path}: {exc}")
        return {
            "record": record,
            "summary": build_input_summary(
                record,
                hidden_state_shape=record.get("hidden_state_shape"),
                question_norms=None,
                span_pool_available=False,
                mask_position_available=False,
                span_token_indices=None,
                nonfinite_count=0,
                warnings=warnings,
            ),
            "valid": False,
            "pools": {},
        }

    if not is_valid_tensor_shape(tensor):
        warning_counts["bad_tensor_shape"] += 1
        warnings.append(
            "bad tensor shape: expected [num_layers, seq_len, hidden_size], "
            f"got {list(getattr(tensor, 'shape', []))}"
        )
        return {
            "record": record,
            "summary": build_input_summary(
                record,
                hidden_state_shape=list(getattr(tensor, "shape", record.get("hidden_state_shape"))),
                question_norms=None,
                span_pool_available=False,
                mask_position_available=False,
                span_token_indices=None,
                nonfinite_count=0,
                warnings=warnings,
            ),
            "valid": False,
            "pools": {},
        }

    tensor, nonfinite_count = sanitize_tensor(tensor)
    if nonfinite_count:
        warning_counts["nonfinite_tensor_values"] += 1
        warnings.append(f"tensor contained {nonfinite_count} NaN/Inf value(s); nan_to_num applied")

    span_token_indices, span_warnings = locate_span_token_indices(record)
    warnings.extend(span_warnings)
    for warning in span_warnings:
        if "missing token offsets" in warning:
            warning_counts["missing_token_offsets"] += 1
        elif "missing span overlap" in warning or "missing span char ranges" in warning:
            warning_counts["missing_span_overlap"] += 1
            warning_counts["missing_mask_position_overlap"] += 1

    question = pool_question(tensor)
    span = pool_span(tensor, span_token_indices)
    mask_position = span
    summary = build_input_summary(
        record,
        hidden_state_shape=list(tensor.shape),
        question_norms=norm_by_layer(question),
        span_pool_available=span is not None,
        mask_position_available=mask_position is not None,
        span_token_indices=span_token_indices,
        nonfinite_count=nonfinite_count,
        warnings=warnings,
    )
    return {
        "record": record,
        "summary": summary,
        "valid": True,
        "pools": {
            "question": question,
            "span": span,
            "mask_position": mask_position,
        },
    }


def build_input_summary(
    record: dict[str, Any],
    *,
    hidden_state_shape: list[int] | None,
    question_norms: list[float | None] | None,
    span_pool_available: bool,
    mask_position_available: bool,
    span_token_indices: list[int] | None,
    nonfinite_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "cache_id": record.get("cache_id"),
        "masked_id": record.get("masked_id"),
        "id": record.get("id"),
        "unit_id": record.get("unit_id"),
        "input_type": record.get("input_type"),
        "input_index": record.get("input_index"),
        "backend": record.get("backend"),
        "model_name": record.get("model_name"),
        "tokenizer_name": record.get("tokenizer_name"),
        "layer_indices": record.get("layer_indices", record.get("resolved_layer_indices")),
        "seq_len": record.get("seq_len"),
        "hidden_size": record.get("hidden_size"),
        "hidden_state_shape": hidden_state_shape,
        "hidden_state_path": record.get("hidden_state_path"),
        "question_norm_by_layer": question_norms,
        "span_pool_available": bool(span_pool_available),
        "mask_position_available": bool(mask_position_available),
        "span_token_indices": span_token_indices,
        "span_token_count": len(span_token_indices or []),
        "nonfinite_count": int(nonfinite_count),
        "warnings": warnings,
    }


def summarize_unloaded_record(record: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return build_input_summary(
        record,
        hidden_state_shape=record.get("hidden_state_shape"),
        question_norms=None,
        span_pool_available=False,
        mask_position_available=False,
        span_token_indices=None,
        nonfinite_count=0,
        warnings=warnings,
    )


def build_feature_record(
    *,
    original: dict[str, Any],
    masked: dict[str, Any],
    recovered: dict[str, Any],
    backend: str,
    eps: float,
    warning_counts: Counter[str],
) -> dict[str, Any] | None:
    original_record = original["record"]
    masked_record = masked["record"]
    recovered_record = recovered["record"]

    if not compatible_required_shapes(original, masked, recovered):
        return None

    warnings: list[str] = []
    for loaded in [original, masked, recovered]:
        warnings.extend(loaded["summary"].get("warnings", []))

    feature_record: dict[str, Any] = {
        "feature_id": (
            f"{original_record['masked_id']}::recovered::{recovered_record.get('input_index', 0)}"
        ),
        "masked_id": original_record.get("masked_id"),
        "id": original_record.get("id"),
        "unit_id": original_record.get("unit_id"),
        "original_cache_id": original_record.get("cache_id"),
        "masked_cache_id": masked_record.get("cache_id"),
        "recovered_cache_id": recovered_record.get("cache_id"),
        "recovered_input_index": recovered_record.get("input_index"),
        "backend": backend,
        "source_cache_backend": original_record.get("backend"),
        "model_name": original_record.get("model_name"),
        "tokenizer_name": original_record.get("tokenizer_name"),
        "layer_indices": original_record.get("layer_indices"),
        "hidden_size": original_record.get("hidden_size"),
        "human_attention_anchor_label": original_record.get("human_attention_anchor_label"),
        "human_attention_anchor_label_name": original_record.get(
            "human_attention_anchor_label_name",
            original_record.get("human_attention_anchor_label"),
        ),
        "human_recoverability_label": original_record.get("human_recoverability_label"),
        "human_semantic_role": original_record.get("human_semantic_role"),
        "human_guidance_priority": original_record.get("human_guidance_priority"),
        "human_error_type": original_record.get("human_error_type"),
        "probe_usage": original_record.get("probe_usage"),
        "warnings": sorted(set(warnings)),
    }

    for pooling_name in POOLING_NAMES:
        for pair_name, (left_key, right_key) in DISTANCE_PAIRS.items():
            left_pool = {"original": original, "masked": masked, "recovered": recovered}[left_key][
                "pools"
            ][pooling_name]
            right_pool = {"original": original, "masked": masked, "recovered": recovered}[right_key][
                "pools"
            ][pooling_name]
            if left_pool is None or right_pool is None:
                cosine = None
            else:
                cosine = compute_cosine_distance(left_pool, right_pool, eps=eps)
                if any(value is None for value in cosine):
                    warning_counts["cosine_near_zero"] += 1
                    feature_record["warnings"].append(
                        f"near-zero vector norm encountered for {pooling_name} {pair_name} cosine"
                    )
            add_layerwise_features(
                feature_record,
                f"{pooling_name}_{pair_name}_cosine",
                cosine,
            )

    feature_record["warnings"] = sorted(set(feature_record["warnings"]))
    return feature_record


def add_layerwise_features(
    record: dict[str, Any],
    name: str,
    values: list[float | None] | None,
) -> None:
    record[f"{name}_by_layer"] = values
    aggregates = summarize_layer_values(values)
    for suffix in SUMMARY_SUFFIXES:
        record[f"{name}_{suffix}"] = aggregates[suffix]


def summarize_layer_values(values: list[float | None] | None) -> dict[str, float | None]:
    if values is None:
        return {suffix: None for suffix in SUMMARY_SUFFIXES}
    numeric_values = [value for value in values if value is not None]
    first = values[0] if values else None
    last = values[-1] if values else None
    return {
        "mean": sanitize_float(mean(numeric_values)) if numeric_values else None,
        "max": sanitize_float(max(numeric_values)) if numeric_values else None,
        "min": sanitize_float(min(numeric_values)) if numeric_values else None,
        "first_layer": first,
        "last_layer": last,
        "delta_last_minus_first": sanitize_float(last - first)
        if first is not None and last is not None
        else None,
    }


def locate_span_token_indices(record: dict[str, Any]) -> tuple[list[int] | None, list[str]]:
    token_ranges = record.get("token_char_ranges")
    if not token_ranges:
        return None, ["missing token offsets for span-aware pooling"]

    span_ranges = span_ranges_for_record(record)
    if not span_ranges:
        return None, ["missing span char ranges for span-aware pooling"]

    indices: list[int] = []
    for token_index, token_range in enumerate(token_ranges):
        if not valid_range(token_range):
            continue
        if any(ranges_overlap(token_range, span_range) for span_range in span_ranges):
            indices.append(token_index)
    if not indices:
        return None, ["missing span overlap for span-aware pooling"]
    return indices, []


def span_ranges_for_record(record: dict[str, Any]) -> list[list[int]]:
    input_type = record.get("input_type")
    if input_type == "original":
        return [
            span["original_char_range"]
            for span in record.get("masked_original_spans", [])
            if valid_range(span.get("original_char_range"))
        ]
    if input_type == "masked":
        return [span for span in record.get("mask_char_ranges", []) if valid_range(span)]
    if input_type == "recovered":
        return [
            span["recovered_char_range"]
            for span in record.get("recovered_fill_spans", [])
            if valid_range(span.get("recovered_char_range"))
        ]
    return []


def norm_by_layer(tensor: Any) -> list[float | None]:
    return [sanitize_float(float(value.item())) for value in tensor.norm(dim=1)]


def sanitize_tensor(tensor: Any) -> tuple[Any, int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to sanitize hidden-state tensors") from exc

    nonfinite_mask = ~torch.isfinite(tensor)
    nonfinite_count = int(nonfinite_mask.sum().item())
    if nonfinite_count:
        tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        tensor = tensor.float()
    return tensor, nonfinite_count


def is_valid_tensor_shape(tensor: Any) -> bool:
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 3:
        return False
    return int(shape[0]) > 0 and int(shape[1]) > 0 and int(shape[2]) > 0


def compatible_required_shapes(*loaded_inputs: dict[str, Any]) -> bool:
    shapes = []
    for loaded in loaded_inputs:
        question = loaded["pools"].get("question")
        if question is None:
            return False
        shapes.append(tuple(question.shape))
    return len(set(shapes)) == 1


def split_records_by_input_type(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["input_type"]].append(record)
    return dict(by_type)


def first_by_input_index(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=input_index_key)[0]


def input_index_key(record: dict[str, Any]) -> int:
    try:
        return int(record.get("input_index", 0))
    except (TypeError, ValueError):
        return 0


def any_position_pool_null(record: dict[str, Any]) -> bool:
    return any(
        (
            key.startswith("span_") or key.startswith("mask_position_")
        )
        and key.endswith("_by_layer")
        and value is None
        for key, value in record.items()
    )


def any_cosine_null(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if not (key.endswith("_cosine_by_layer") and isinstance(value, list)):
            continue
        if any(item is None for item in value):
            return True
    return False


def build_report(
    *,
    backend: str,
    input_manifest_path: Path,
    input_report_path: Path,
    alignment_report_path: Path,
    metadata_path: Path,
    output_dir: Path,
    output_files: dict[str, Path],
    manifest_records: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    feature_records: list[dict[str, Any]],
    recovered_variant_count: int,
    skipped_groups: int,
    skipped_recovered_variants: int,
    source_report: dict[str, Any],
    alignment_report: dict[str, Any],
    metadata: dict[str, Any],
    null_counts: Counter[str],
    warning_counts: Counter[str],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    first_record = manifest_records[0] if manifest_records else {}
    deprecated_extra_outputs = [
        str(output_dir / filename)
        for filename in DEPRECATED_EXTRA_FILENAMES
        if (output_dir / filename).exists()
    ]
    return {
        "sprint": "2B-fix",
        "backend": backend,
        "status": "ok",
        "source_cache": {
            "hidden_state_manifest_path": str(input_manifest_path),
            "hidden_state_cache_report_path": str(input_report_path),
            "token_alignment_report_path": str(alignment_report_path),
            "real_run_metadata_path": str(metadata_path),
            "source_cache_backend": source_report.get("backend", metadata.get("backend")),
            "source_model_name": metadata.get("model_name_or_path", first_record.get("model_name")),
            "source_tokenizer_name": metadata.get(
                "tokenizer_name_or_path",
                first_record.get("tokenizer_name"),
            ),
            "source_layer_indices": source_report.get(
                "layer_indices",
                metadata.get("resolved_layer_indices", first_record.get("layer_indices")),
            ),
            "source_num_inputs_total": source_report.get(
                "num_inputs_total",
                len(manifest_records),
            ),
            "source_alignment_warning_count": alignment_report.get("alignment_warning_count"),
        },
        "outputs": {
            "representation_features_path": str(output_files["representation_features"]),
            "representation_feature_report_path": str(
                output_files["representation_feature_report"]
            ),
            "deprecated_extra_outputs": deprecated_extra_outputs,
        },
        "counts": {
            "num_input_records": len(manifest_records),
            "num_masked_groups": len(grouped),
            "num_feature_records": len(feature_records),
            "num_recovered_variants": recovered_variant_count,
            "num_skipped_groups": skipped_groups,
            "num_skipped_recovered_variants": skipped_recovered_variants,
        },
        "feature_scope": {
            "question_pooled_representation": True,
            "span_pooled_representation": True,
            "mask_position_representation": True,
            "cosine_distance": True,
            "layer_wise_distance_curve": True,
            "topology_features": False,
            "trajectory_features": False,
            "probe_dataset": False,
            "probe_training": False,
            "attention_guidance": False,
        },
        "preflight_notes": {
            "missing_tests_test_hidden_state_cache_hf_py": True,
            "missing_tests_test_hidden_state_cache_hf_py_is_failure": False,
            "sprint_2B_task_card_preexisting_AM": True,
        },
        "null_counts": {
            "position_pool_feature_null_records": int(
                null_counts.get("position_pool_feature_null_records", 0)
            ),
            "cosine_near_zero_null_records": int(
                null_counts.get("cosine_near_zero_null_records", 0)
            ),
        },
        "warning_counts": {
            "missing_span_overlap": int(warning_counts.get("missing_span_overlap", 0)),
            "missing_mask_position_overlap": int(
                warning_counts.get("missing_mask_position_overlap", 0)
            ),
            "missing_token_offsets": int(warning_counts.get("missing_token_offsets", 0)),
            "nonfinite_tensor_values": int(warning_counts.get("nonfinite_tensor_values", 0)),
            "bad_tensor_shape": int(warning_counts.get("bad_tensor_shape", 0)),
            "missing_tensor_file": int(warning_counts.get("missing_tensor_file", 0)),
            "cosine_near_zero": int(warning_counts.get("cosine_near_zero", 0)),
            "missing_original_input": int(warning_counts.get("missing_original_input", 0)),
            "missing_masked_input": int(warning_counts.get("missing_masked_input", 0)),
            "missing_recovered_input": int(warning_counts.get("missing_recovered_input", 0)),
            "tensor_load_error": int(warning_counts.get("tensor_load_error", 0)),
            "invalid_required_input_tensor": int(
                warning_counts.get("invalid_required_input_tensor", 0)
            ),
            "invalid_recovered_tensor": int(warning_counts.get("invalid_recovered_tensor", 0)),
            "incompatible_feature_tensors": int(
                warning_counts.get("incompatible_feature_tensors", 0)
            ),
        },
        "warnings": warnings,
    }


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    if overwrite:
        for filename in [
            FEATURES_FILENAME,
            REPORT_FILENAME,
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
    ]
    for forbidden_root in forbidden_roots:
        forbidden_resolved = forbidden_root.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"Refusing to write Sprint 2B outputs under forbidden path: {output_dir}")


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON input: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must be an object: {json_path}")
    return data


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


def add_warning(
    warnings: list[dict[str, Any]],
    warning_counts: Counter[str],
    warning_type: str,
    message: str,
    **metadata: Any,
) -> None:
    warning_counts[warning_type] += 1
    warning = {"warning_type": warning_type, "message": message}
    warning.update(metadata)
    warnings.append(warning)


def valid_range(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
        and value[0] < value[1]
    )


def ranges_overlap(first: list[int], second: list[int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def sanitize_float(value: float) -> float | None:
    try:
        import math
    except ImportError:  # pragma: no cover
        return value
    if not math.isfinite(value):
        return None
    return float(value)
