"""Sprint 2A hidden-state cache baseline using deterministic stub tensors."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.token_alignment import (
    DEFAULT_MASK_TOKEN,
    DEFAULT_TOKENIZER_NAME,
    align_original_to_masked,
    align_recovered_to_masked,
    find_mask_char_ranges,
    tokenize_with_offsets,
)


STUB_BACKEND = "stub_hidden_state_v0"
REAL_HF_BACKEND = "hf_local_causal_lm_hidden_states_v0"
DEFAULT_BACKEND = STUB_BACKEND
DEFAULT_LAYER_INDICES = [0, 1, 2]
DEFAULT_HIDDEN_SIZE = 8
DEFAULT_MODEL_NAME = STUB_BACKEND
DEFAULT_REAL_LAYER_INDICES = [0, 8, 16, 24, 27]
MANIFEST_FILENAME = "hidden_state_manifest.jsonl"
CACHE_REPORT_FILENAME = "hidden_state_cache_report.json"
ALIGNMENT_REPORT_FILENAME = "token_alignment_report.json"
REAL_RUN_METADATA_FILENAME = "real_run_metadata.json"
HIDDEN_STATES_DIRNAME = "hidden_states"

REQUIRED_2A_FIELDS = [
    "masked_id",
    "id",
    "unit_id",
    "original_question",
    "masked_question",
    "recovered_questions",
    "human_recoverability_label",
    "human_attention_anchor_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
    "probe_usage",
]

HUMAN_LABEL_FIELDS = [
    "human_recoverability_label",
    "human_attention_anchor_label",
    "human_semantic_role",
    "human_guidance_priority",
    "human_error_type",
    "probe_usage",
]


def cache_hidden_states_for_manifest(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    backend: str = DEFAULT_BACKEND,
    layer_indices: list[int] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    mask_token: str = DEFAULT_MASK_TOKEN,
    overwrite: bool = False,
    model_path: str | Path | None = None,
    device: str = "cpu",
    device_map: str | None = None,
    dtype: str = "float32",
    max_length: int | None = None,
    batch_size: int = 1,
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Read a Sprint 2A manifest, cache hidden states, and write reports."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Sprint 2A input manifest: {input_path}")
    manifest_records = read_jsonl(input_path)
    validate_2a_manifest_records(manifest_records)

    requested_layers = layer_indices or (
        list(DEFAULT_REAL_LAYER_INDICES)
        if backend == REAL_HF_BACKEND
        else list(DEFAULT_LAYER_INDICES)
    )
    backend_context = None
    if backend == REAL_HF_BACKEND:
        ensure_output_dir_allowed(output_dir)
        backend_context = load_hf_backend(
            model_path=model_path,
            device=device,
            device_map=device_map,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            trust_remote_code=trust_remote_code,
        )

    prepare_output_dir(output_dir, overwrite=overwrite)
    cache_records, cache_report, alignment_report = build_hidden_state_cache_records(
        manifest_records,
        output_dir=output_dir,
        input_path=input_path,
        backend=backend,
        layer_indices=requested_layers,
        hidden_size=hidden_size,
        mask_token=mask_token,
        backend_context=backend_context,
        max_length=max_length,
    )
    write_hidden_state_manifest(cache_records, output_dir / MANIFEST_FILENAME)
    write_hidden_state_cache_report(cache_report, output_dir / CACHE_REPORT_FILENAME)
    write_token_alignment_report(alignment_report, output_dir / ALIGNMENT_REPORT_FILENAME)
    real_run_metadata = None
    if backend == REAL_HF_BACKEND:
        real_run_metadata = build_real_run_metadata(
            cache_report=cache_report,
            model_path=model_path,
            backend_context=backend_context,
            device=device,
            device_map=device_map,
            dtype=dtype,
            max_length=max_length,
            batch_size=batch_size,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            trust_remote_code=trust_remote_code,
            requested_layer_indices=requested_layers,
        )
        write_json(real_run_metadata, output_dir / REAL_RUN_METADATA_FILENAME)
    return {
        "cache_records": cache_records,
        "cache_report": cache_report,
        "token_alignment_report": alignment_report,
        "real_run_metadata": real_run_metadata,
        "output_files": {
            "hidden_state_manifest": str(output_dir / MANIFEST_FILENAME),
            "hidden_state_cache_report": str(output_dir / CACHE_REPORT_FILENAME),
            "token_alignment_report": str(output_dir / ALIGNMENT_REPORT_FILENAME),
            "real_run_metadata": (
                str(output_dir / REAL_RUN_METADATA_FILENAME)
                if backend == REAL_HF_BACKEND
                else None
            ),
            "hidden_states_dir": str(output_dir / HIDDEN_STATES_DIRNAME),
        },
    }


def build_hidden_state_cache_records(
    manifest_records: list[dict],
    *,
    output_dir: str | Path,
    input_path: str | Path,
    backend: str = DEFAULT_BACKEND,
    layer_indices: list[int] | None = None,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend_context: dict[str, Any] | None = None,
    max_length: int | None = None,
) -> tuple[list[dict], dict[str, Any], dict[str, Any]]:
    """Build manifest records and write one `.pt` tensor per model input."""

    if backend not in {STUB_BACKEND, REAL_HF_BACKEND}:
        raise ValueError(
            f"Unsupported backend {backend!r}; supported backends: "
            f"{STUB_BACKEND!r}, {REAL_HF_BACKEND!r}"
        )
    if backend == REAL_HF_BACKEND and backend_context is None:
        backend_context = load_hf_backend(model_path=None)
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")

    resolved_layers = layer_indices or list(DEFAULT_LAYER_INDICES)
    if not resolved_layers:
        raise ValueError("layer_indices must be non-empty")

    output_dir = Path(output_dir)
    hidden_states_dir = output_dir / HIDDEN_STATES_DIRNAME
    ensure_dir(hidden_states_dir)

    cache_records: list[dict] = []
    case_alignment_summaries: list[dict] = []
    records_with_warnings: list[dict] = []
    warning_counts: Counter[str] = Counter()
    input_type_counts: Counter[str] = Counter()
    alignment_status_counts: Counter[str] = Counter()
    seq_lens: list[int] = []

    for case in manifest_records:
        case_summary = build_case_alignment_summary(case, mask_token)
        case_alignment_summaries.append(case_summary)

        for input_spec in build_input_specs(case):
            input_type = input_spec["input_type"]
            input_index = input_spec["input_index"]
            input_text = input_spec["input_text"]
            cache_id = build_cache_id(case["masked_id"], input_type, input_index)
            if backend == STUB_BACKEND:
                token_metadata = tokenize_with_offsets(input_text)
                tensor = build_stub_hidden_state_tensor(
                    token_metadata["token_ids"],
                    layer_indices=resolved_layers,
                    hidden_size=hidden_size,
                )
                record_layer_indices = list(resolved_layers)
                record_requested_layers = list(resolved_layers)
                record_hidden_size = hidden_size
                model_name = DEFAULT_MODEL_NAME
                tokenizer_name = DEFAULT_TOKENIZER_NAME
                layer_warnings: list[dict[str, str]] = []
            else:
                real_result = build_hf_hidden_state_tensor(
                    input_text,
                    backend_context=backend_context,
                    requested_layer_indices=resolved_layers,
                    max_length=max_length,
                )
                token_metadata = real_result["token_metadata"]
                tensor = real_result["tensor"]
                record_layer_indices = real_result["resolved_layer_indices"]
                record_requested_layers = list(resolved_layers)
                record_hidden_size = int(tensor.shape[2])
                model_name = backend_context["model_name"]
                tokenizer_name = backend_context["tokenizer_name"]
                layer_warnings = real_result["warnings"]
            tensor_path = hidden_states_dir / f"{cache_id}.pt"
            save_tensor(tensor, tensor_path)

            alignment = alignment_for_input(case_summary, input_type, input_index)
            warnings = list(alignment["warnings"]) + layer_warnings
            alignment_status = alignment["alignment_status"]
            if layer_warnings and alignment_status == "ok":
                alignment_status = "warning"
            input_type_counts[input_type] += 1
            alignment_status_counts[alignment_status] += 1
            seq_lens.append(token_metadata["seq_len"])
            for warning in warnings:
                warning_type = warning["warning_type"]
                warning_counts[warning_type] += 1
                records_with_warnings.append(
                    {
                        "cache_id": cache_id,
                        "masked_id": case["masked_id"],
                        "input_type": input_type,
                        "input_index": input_index,
                        "warning_type": warning_type,
                        "message": warning["message"],
                    }
                )

            cache_record = {
                "cache_id": cache_id,
                "masked_id": case["masked_id"],
                "id": case["id"],
                "unit_id": case["unit_id"],
                "input_type": input_type,
                "input_index": input_index,
                "input_text": input_text,
                "backend": backend,
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                "tokens": token_metadata["tokens"],
                "token_ids": token_metadata["token_ids"],
                "token_char_ranges": token_metadata["token_char_ranges"],
                "requested_layer_indices": record_requested_layers,
                "resolved_layer_indices": record_layer_indices,
                "layer_indices": record_layer_indices,
                "seq_len": token_metadata["seq_len"],
                "hidden_size": record_hidden_size,
                "hidden_state_shape": list(tensor.shape),
                "hidden_state_path": str(tensor_path),
                "alignment_status": alignment_status,
                "alignment_warnings": [warning["message"] for warning in warnings],
                "mask_char_ranges": case_summary["mask_char_ranges"],
                "masked_original_spans": case_summary["masked_original_spans"],
                "recovered_fill_spans": alignment.get("recovered_fill_spans", []),
            }
            for field in HUMAN_LABEL_FIELDS:
                cache_record[field] = case[field]
            cache_records.append(cache_record)

    report_layer_indices = cache_records[0]["layer_indices"] if cache_records else resolved_layers
    cache_report = build_hidden_state_cache_report(
        input_path=input_path,
        output_dir=output_dir,
        backend=backend,
        manifest_records=manifest_records,
        cache_records=cache_records,
        input_type_counts=input_type_counts,
        alignment_status_counts=alignment_status_counts,
        layer_indices=report_layer_indices,
        hidden_size=hidden_size if backend == STUB_BACKEND else None,
        seq_lens=seq_lens,
    )
    alignment_report = build_token_alignment_report(
        manifest_records=manifest_records,
        case_alignment_summaries=case_alignment_summaries,
        warning_counts=warning_counts,
        records_with_warnings=records_with_warnings,
    )
    return cache_records, cache_report, alignment_report


def write_hidden_state_manifest(records: list[dict], path: str | Path) -> None:
    write_jsonl(records, path)


def write_hidden_state_cache_report(report: dict, path: str | Path) -> None:
    write_json(report, path)


def write_token_alignment_report(report: dict, path: str | Path) -> None:
    write_json(report, path)


def validate_2a_manifest_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Sprint 2A manifest is empty")
    seen_masked_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        name = f"Sprint 2A manifest line {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{name} must be a JSON object")
        missing = [field for field in REQUIRED_2A_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")
        for field in REQUIRED_2A_FIELDS:
            if field == "recovered_questions":
                continue
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"{name} field {field!r} must be a non-empty str")
        if record["masked_id"] in seen_masked_ids:
            raise ValueError(f"{name} duplicate masked_id: {record['masked_id']}")
        seen_masked_ids.add(record["masked_id"])
        recovered = record["recovered_questions"]
        if not isinstance(recovered, list) or not recovered:
            raise ValueError(f"{name} field 'recovered_questions' must be a non-empty list")
        for recovered_index, question in enumerate(recovered):
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"{name} recovered_questions[{recovered_index}] must be a non-empty str"
                )


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_output_dir_allowed(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    cache_report_path = output_dir / CACHE_REPORT_FILENAME
    alignment_report_path = output_dir / ALIGNMENT_REPORT_FILENAME
    hidden_states_dir = output_dir / HIDDEN_STATES_DIRNAME
    existing_outputs = [
        path
        for path in [manifest_path, cache_report_path, alignment_report_path]
        if path.exists()
    ]
    if hidden_states_dir.exists():
        existing_outputs.extend(hidden_states_dir.glob("*.pt"))

    if existing_outputs and not overwrite:
        raise ValueError(
            f"Output directory already contains Sprint 2A outputs; pass --overwrite: {output_dir}"
        )

    ensure_dir(output_dir)
    ensure_dir(hidden_states_dir)
    if overwrite:
        for path in [manifest_path, cache_report_path, alignment_report_path]:
            if path.exists():
                path.unlink()
        for path in hidden_states_dir.glob("*.pt"):
            path.unlink()


def ensure_output_dir_allowed(output_dir: Path) -> None:
    project_root = Path.cwd().resolve()
    resolved = output_dir.resolve()
    forbidden = [
        project_root / "data" / "processed",
        project_root / "outputs" / "logs" / "sprint_1N_real_downstream",
        project_root / "outputs" / "logs" / "sprint_1O_recovery_scoring",
        project_root / "outputs" / "logs" / "sprint_1P_upgraded_downstream",
        project_root / "outputs" / "logs" / "sprint_1Q_real_signal_quality_review",
    ]
    for forbidden_root in forbidden:
        if resolved == forbidden_root.resolve() or resolved.is_relative_to(forbidden_root.resolve()):
            raise ValueError(f"Refusing to write Sprint 2A outputs under forbidden path: {output_dir}")


def build_case_alignment_summary(case: dict, mask_token: str) -> dict[str, Any]:
    mask_info = find_mask_char_ranges(case["masked_question"], mask_token)
    original_alignment = align_original_to_masked(
        case["original_question"],
        case["masked_question"],
        mask_token,
    )
    recovered_alignments = [
        align_recovered_to_masked(case["masked_question"], recovered_question, mask_token)
        for recovered_question in case["recovered_questions"]
    ]
    return {
        "masked_id": case["masked_id"],
        "num_masks": mask_info["num_masks"],
        "mask_char_ranges": mask_info["mask_char_ranges"],
        "mask_warnings": mask_info["warnings"],
        "original_alignment": original_alignment,
        "masked_original_spans": original_alignment["masked_original_spans"],
        "recovered_alignments": recovered_alignments,
    }


def build_input_specs(case: dict) -> list[dict[str, Any]]:
    specs = [
        {
            "input_type": "original",
            "input_index": 0,
            "input_text": case["original_question"],
        },
        {
            "input_type": "masked",
            "input_index": 0,
            "input_text": case["masked_question"],
        },
    ]
    for index, recovered_question in enumerate(case["recovered_questions"]):
        specs.append(
            {
                "input_type": "recovered",
                "input_index": index,
                "input_text": recovered_question,
            }
        )
    return specs


def alignment_for_input(case_summary: dict, input_type: str, input_index: int) -> dict:
    if input_type == "original":
        warnings = list(case_summary["original_alignment"]["warnings"])
        return {
            "alignment_status": "ok" if not warnings else "warning",
            "warnings": warnings,
        }
    if input_type == "masked":
        warnings = list(case_summary["mask_warnings"]) + list(
            case_summary["original_alignment"]["warnings"]
        )
        return {
            "alignment_status": "ok" if not warnings else "warning",
            "warnings": warnings,
        }
    recovered_alignment = case_summary["recovered_alignments"][input_index]
    return {
        "alignment_status": recovered_alignment["alignment_status"],
        "warnings": list(recovered_alignment["warnings"]),
        "recovered_fill_spans": recovered_alignment["recovered_fill_spans"],
    }


def build_stub_hidden_state_tensor(
    token_ids: list[int],
    *,
    layer_indices: list[int],
    hidden_size: int,
) -> Any:
    """Build a deterministic small tensor without calling a model."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to write Sprint 2A .pt cache files") from exc

    seq_len = len(token_ids)
    tensor = torch.zeros((len(layer_indices), seq_len, hidden_size), dtype=torch.float32)
    for layer_position, layer_index in enumerate(layer_indices):
        for token_position, token_id in enumerate(token_ids):
            for hidden_index in range(hidden_size):
                value = (
                    float(layer_index) * 0.1
                    + float(token_position) * 0.01
                    + float(hidden_index) * 0.001
                    + float(token_id % 997) * 0.00001
                )
                tensor[layer_position, token_position, hidden_index] = value
    return tensor


def load_hf_backend(
    *,
    model_path: str | Path | None,
    device: str = "cpu",
    device_map: str | None = None,
    dtype: str = "float32",
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load a local HF causal LM with strict local-only boundaries."""

    if model_path is None:
        raise ValueError("--model-path is required for hf_local_causal_lm_hidden_states_v0")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing local model path for hf_local_causal_lm_hidden_states_v0: {model_path}"
        )
    if not is_module_available("transformers"):
        raise RuntimeError("transformers is required for hf_local_causal_lm_hidden_states_v0")
    if load_in_4bit and not is_module_available("bitsandbytes"):
        raise RuntimeError(
            "blocked_by_missing_bitsandbytes: --load-in-4bit requires bitsandbytes, "
            "but bitsandbytes is not installed. Install it or request a separate lower-memory "
            "configuration; no fp16 fallback was attempted."
        )

    torch = import_torch()
    components = import_transformers_components()
    torch_dtype = resolve_torch_dtype(dtype, torch)
    bnb_compute_dtype = resolve_torch_dtype(bnb_4bit_compute_dtype, torch)
    local_model_path = str(model_path)
    shared_kwargs = {
        "local_files_only": True,
        "trust_remote_code": bool(trust_remote_code),
    }

    tokenizer = components["AutoTokenizer"].from_pretrained(
        local_model_path,
        **shared_kwargs,
    )
    model_kwargs = {
        **shared_kwargs,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if load_in_4bit:
        quantization_config = components["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map or "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        if device_map:
            model_kwargs["device_map"] = device_map

    model = components["AutoModelForCausalLM"].from_pretrained(
        local_model_path,
        **model_kwargs,
    )
    if hasattr(model, "eval"):
        model.eval()
    if not load_in_4bit and not device_map and hasattr(model, "to"):
        resolved_device = resolve_device(device, torch)
        model.to(resolved_device)
    else:
        resolved_device = device

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": str(getattr(model, "name_or_path", local_model_path)),
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", local_model_path)),
        "model_path": local_model_path,
        "device": str(resolved_device),
        "device_map": device_map,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "trust_remote_code": bool(trust_remote_code),
    }


def build_hf_hidden_state_tensor(
    text: str,
    *,
    backend_context: dict[str, Any],
    requested_layer_indices: list[int],
    max_length: int | None,
) -> dict[str, Any]:
    torch = import_torch()
    tokenizer = backend_context["tokenizer"]
    model = backend_context["model"]
    tokenized = tokenize_hf_with_offsets(tokenizer, text, max_length=max_length)
    model_inputs = {
        key: value
        for key, value in tokenized["model_inputs"].items()
        if key != "offset_mapping"
    }
    target_device = infer_model_input_device(model, backend_context["device"], torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
    tensor, resolved_layers, warnings = select_hidden_state_layers(
        outputs.hidden_states,
        requested_layer_indices,
    )
    return {
        "tensor": tensor.cpu(),
        "resolved_layer_indices": resolved_layers,
        "token_metadata": tokenized["token_metadata"],
        "warnings": warnings,
    }


def tokenize_hf_with_offsets(tokenizer: Any, text: str, *, max_length: int | None) -> dict[str, Any]:
    kwargs = {
        "return_tensors": "pt",
        "truncation": max_length is not None,
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(text, **kwargs)
    input_ids = encoded["input_ids"][0].tolist()
    if "offset_mapping" in encoded:
        token_char_ranges = [
            [int(start), int(end)]
            for start, end in encoded["offset_mapping"][0].tolist()
        ]
    else:
        token_char_ranges = []
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return {
        "model_inputs": encoded,
        "token_metadata": {
            "tokenizer_name": str(getattr(tokenizer, "name_or_path", "hf_tokenizer")),
            "tokens": tokens,
            "token_ids": [int(token_id) for token_id in input_ids],
            "token_char_ranges": token_char_ranges,
            "seq_len": len(input_ids),
        },
    }


def select_hidden_state_layers(
    hidden_states: tuple[Any, ...] | list[Any],
    requested_layer_indices: list[int],
) -> tuple[Any, list[int], list[dict[str, str]]]:
    torch = import_torch()
    resolved = [
        layer_index
        for layer_index in requested_layer_indices
        if 0 <= layer_index < len(hidden_states)
    ]
    warnings: list[dict[str, str]] = []
    dropped = [
        layer_index
        for layer_index in requested_layer_indices
        if layer_index not in resolved
    ]
    if dropped:
        warnings.append(
            {
                "warning_type": "layer_index_out_of_range",
                "message": (
                    f"requested layer indices out of range and skipped: {dropped}; "
                    f"available hidden_states={len(hidden_states)}"
                ),
            }
        )
    if not resolved:
        raise ValueError(
            "No requested layer indices are available in model hidden_states; "
            f"requested={requested_layer_indices}, available={len(hidden_states)}"
        )
    selected = [hidden_states[layer_index].detach().squeeze(0) for layer_index in resolved]
    return torch.stack(selected, dim=0), resolved, warnings


def infer_model_input_device(model: Any, configured_device: str, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return resolve_device(configured_device, torch)


def resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_torch_dtype(dtype: str, torch: Any) -> Any:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype {dtype!r}; allowed values: {allowed}")
    return mapping[dtype]


def is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to cache hidden states") from exc
    return torch


def import_transformers_components() -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for hf_local_causal_lm_hidden_states_v0") from exc
    return {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def save_tensor(tensor: Any, path: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to write Sprint 2A .pt cache files") from exc

    ensure_dir(path.parent)
    torch.save(tensor, path)


def build_hidden_state_cache_report(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    backend: str,
    manifest_records: list[dict],
    cache_records: list[dict],
    input_type_counts: Counter[str],
    alignment_status_counts: Counter[str],
    layer_indices: list[int],
    hidden_size: int | None,
    seq_lens: list[int],
) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "backend": backend,
        "num_cases": len(manifest_records),
        "num_inputs_total": len(cache_records),
        "input_type_counts": dict(sorted(input_type_counts.items())),
        "num_hidden_state_files": len(cache_records),
        "layer_indices": list(layer_indices),
        "hidden_size": hidden_size
        if hidden_size is not None
        else (cache_records[0]["hidden_size"] if cache_records else None),
        "seq_len_summary": summarize_numbers(seq_lens),
        "alignment_status_counts": dict(sorted(alignment_status_counts.items())),
        "human_recoverability_label_counts": count_field(
            manifest_records,
            "human_recoverability_label",
        ),
        "human_error_type_counts": count_field(manifest_records, "human_error_type"),
        "probe_usage_counts": count_field(manifest_records, "probe_usage"),
        "failure_count": 0,
        "failures": [],
    }


def build_real_run_metadata(
    *,
    cache_report: dict[str, Any],
    model_path: str | Path | None,
    backend_context: dict[str, Any] | None,
    device: str,
    device_map: str | None,
    dtype: str,
    max_length: int | None,
    batch_size: int,
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str,
    trust_remote_code: bool,
    requested_layer_indices: list[int],
) -> dict[str, Any]:
    resolved_layers = cache_report.get("layer_indices", [])
    return {
        "backend": REAL_HF_BACKEND,
        "model_path": str(model_path),
        "model_name_or_path": backend_context["model_name"] if backend_context else str(model_path),
        "tokenizer_name_or_path": (
            backend_context["tokenizer_name"] if backend_context else str(model_path)
        ),
        "device": device,
        "device_map": device_map,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "max_length": max_length,
        "batch_size": batch_size,
        "trust_remote_code": bool(trust_remote_code),
        "requested_layer_indices": list(requested_layer_indices),
        "resolved_layer_indices": list(resolved_layers),
        "num_cases": cache_report["num_cases"],
        "num_inputs_total": cache_report["num_inputs_total"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_token_alignment_report(
    *,
    manifest_records: list[dict],
    case_alignment_summaries: list[dict],
    warning_counts: Counter[str],
    records_with_warnings: list[dict],
) -> dict[str, Any]:
    mask_counts = [case["num_masks"] for case in case_alignment_summaries]
    recovered_alignments = [
        alignment
        for case in case_alignment_summaries
        for alignment in case["recovered_alignments"]
    ]
    return {
        "num_cases": len(manifest_records),
        "num_masks_total": sum(mask_counts),
        "num_single_mask_cases": sum(1 for count in mask_counts if count == 1),
        "num_group_mask_cases": sum(1 for count in mask_counts if count > 1),
        "num_original_masked_span_alignment_ok": sum(
            1
            for case in case_alignment_summaries
            if case["original_alignment"]["alignment_status"] == "ok"
        ),
        "num_original_masked_span_alignment_failed": sum(
            1
            for case in case_alignment_summaries
            if case["original_alignment"]["alignment_status"] != "ok"
        ),
        "num_recovered_alignment_ok": sum(
            1 for alignment in recovered_alignments if alignment["alignment_status"] == "ok"
        ),
        "num_recovered_alignment_failed": sum(
            1 for alignment in recovered_alignments if alignment["alignment_status"] != "ok"
        ),
        "num_fragment_recovery_outputs": sum(
            1
            for alignment in recovered_alignments
            if alignment["alignment_status"] == "failed_fragment_recovery"
        ),
        "alignment_warning_count": sum(warning_counts.values()),
        "warnings_by_type": dict(sorted(warning_counts.items())),
        "records_with_warnings": records_with_warnings,
    }


def count_field(records: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(record[field] for record in records).items()))


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 10),
    }


def build_cache_id(masked_id: str, input_type: str, input_index: int) -> str:
    return f"{safe_id(masked_id)}__{input_type}__{input_index}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "cache"


def write_json(data: dict, path: str | Path) -> None:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
