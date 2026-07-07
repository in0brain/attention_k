"""Sprint 2J-B: multi-span hidden/attention scoring and formula validation.

This module consumes the 2J-A multi-span matrix and, when explicitly run with a
local HF model, scores every original/masked span pair. Evaluation-only fields
stay outside all gate-eligible formulas.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention import attention_features as af
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.hidden_state_cache import (
    import_torch,
    import_transformers_components,
    infer_model_input_device,
    is_module_available,
    resolve_torch_dtype,
)
from recover_attention.multi_span_reasoning_matrix import (
    FORBIDDEN_FORMULA_INPUT_SUBSTRINGS,
    read_json,
    write_json,
)

BACKEND = "multi_span_hidden_attention_scoring_v0"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
DEFAULT_LAYER_INDICES = [0, 8, 16, 24]
DEFAULT_MASK_TOKEN = "[MASK]"

FEATURE_MATRIX_FILENAME = "multi_span_feature_matrix.jsonl"
SCORE_MATRIX_FILENAME = "multi_span_score_matrix.jsonl"
HIDDEN_ATTENTION_REPORT_FILENAME = "hidden_attention_feature_report.json"
RANKING_REPORT_FILENAME = "same_question_ranking_report.json"
FORMULA_REPORT_FILENAME = "formula_validation_report.json"
TOPK_BUDGET_REPORT_FILENAME = "topk_budget_report.json"
FAILURE_CASE_FILENAME = "failure_case_report.jsonl"
SUCCESS_CASE_FILENAME = "success_case_report.jsonl"
REASONING_GAP_REPORT_FILENAME = "reasoning_signal_gap_report.json"
REVIEW_GATE_FILENAME = "review_gate_multi_span_scoring.md"

QFOCUS_RE = re.compile(r"\b(how many|how much|how far|what|which|how)\b", re.IGNORECASE)
OPERATION_RE = re.compile(
    r"\b(total|each|per|more|less|fewer|times|twice|double|half|sum|difference|"
    r"remaining|left|altogether|combined|plus|minus|every|apiece|bought|sold|"
    r"gave|spent|cost|earned|lost|shared|divided|increased|decreased)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?"
)

POSITIVE_WEAK_KEYNESS = {
    "comparison_key_candidate",
    "negation_key_candidate",
    "condition_key_candidate",
    "operation_key_candidate",
    "question_target_candidate",
}


def run_2jb_scoring(
    *,
    flat_matrix_path: str | Path,
    coverage_report_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    layer_indices: list[int] | None = None,
    mask_token: str = DEFAULT_MASK_TOKEN,
    overwrite: bool = False,
    max_records: int | None = None,
    report_every: int = 25,
) -> dict[str, Any]:
    """Run Sprint 2J-B with a strict local 4-bit HF backend."""

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    if not read_json(coverage_report_path).get("gate", {}).get("passed"):
        raise RuntimeError("2J-A gate did not pass; hidden/attention scoring is blocked")

    flat_records = read_jsonl(flat_matrix_path)
    if max_records is not None:
        flat_records = flat_records[:max_records]
    if not flat_records:
        raise ValueError(f"No multi-span records found in {flat_matrix_path}")

    output_files = _output_files(output_dir)
    if overwrite:
        for path in output_files.values():
            if path.exists():
                path.unlink()

    feature_records = _build_feature_matrix_with_hf(
        flat_records,
        output_files["features"],
        model_path=model_path,
        layer_indices=layer_indices or list(DEFAULT_LAYER_INDICES),
        mask_token=mask_token,
        report_every=report_every,
    )
    complete = len(feature_records) == len(flat_records)
    if not complete:
        raise RuntimeError(
            "2J-B feature matrix is incomplete after the scoring pass: "
            f"{len(feature_records)}/{len(flat_records)}"
        )

    artifacts = build_score_reports(
        feature_records,
        output_dir=output_dir,
        output_files=output_files,
        layer_indices=layer_indices or list(DEFAULT_LAYER_INDICES),
        model_path=model_path,
    )
    return {
        "backend": BACKEND,
        "output_dir": str(output_dir),
        "num_feature_records": len(feature_records),
        "num_score_records": len(artifacts["score_records"]),
        "gate_passed": artifacts["review_gate"]["passed"],
        "checks_passed": artifacts["review_gate"]["num_checks_passed"],
        "checks_total": artifacts["review_gate"]["num_checks_total"],
        "best_non_oracle_formula": artifacts["formula_report"].get("best_non_oracle_formula"),
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }


def _output_files(output_dir: Path) -> dict[str, Path]:
    return {
        "features": output_dir / FEATURE_MATRIX_FILENAME,
        "scores": output_dir / SCORE_MATRIX_FILENAME,
        "hidden_attention_report": output_dir / HIDDEN_ATTENTION_REPORT_FILENAME,
        "ranking_report": output_dir / RANKING_REPORT_FILENAME,
        "formula_report": output_dir / FORMULA_REPORT_FILENAME,
        "topk_budget_report": output_dir / TOPK_BUDGET_REPORT_FILENAME,
        "failure_cases": output_dir / FAILURE_CASE_FILENAME,
        "success_cases": output_dir / SUCCESS_CASE_FILENAME,
        "reasoning_gap_report": output_dir / REASONING_GAP_REPORT_FILENAME,
        "review_gate": output_dir / REVIEW_GATE_FILENAME,
    }


def _build_feature_matrix_with_hf(
    flat_records: list[dict[str, Any]],
    output_path: Path,
    *,
    model_path: str | Path,
    layer_indices: list[int],
    mask_token: str,
    report_every: int,
) -> list[dict[str, Any]]:
    done = _read_existing_by_span_id(output_path)
    pending = [record for record in flat_records if record["span_id"] not in done]
    if not pending:
        return list(done.values())

    context = load_local_attention_backend(model_path=model_path)
    original_cache: dict[str, ForwardResult] = {}
    t0 = time.time()

    for index, record in enumerate(pending, start=1):
        qid = str(record["source_question_id"])
        original = original_cache.get(qid)
        if original is None:
            original = forward_hidden_attention(
                context,
                record["question"],
                char_ranges=[[int(record["span_char_start"]), int(record["span_char_end"])]],
                layer_indices=layer_indices,
                mode="original",
            )
            original_cache[qid] = original

        mask_range = _mask_char_range(record["masked_question"], record.get("mask_text") or mask_token)
        masked = forward_hidden_attention(
            context,
            record["masked_question"],
            char_ranges=[mask_range],
            layer_indices=layer_indices,
            mode="masked",
        )
        feature_record = build_feature_record(
            record,
            original,
            masked,
            layer_indices=layer_indices,
        )
        _append_jsonl([feature_record], output_path)
        done[record["span_id"]] = feature_record
        if report_every and index % report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(
                "[2J-B] "
                f"processed={index}/{len(pending)} total_done={len(done)}/{len(flat_records)} "
                f"rate={index / elapsed:.3f}/s"
            )
    return list(done.values())


def _read_existing_by_span_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = read_jsonl(path)
    return {str(record["span_id"]): record for record in records}


def _append_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_local_attention_backend(*, model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local model path for 2J-B scoring: {model_path}")
    if not is_module_available("transformers"):
        raise RuntimeError("transformers is required for 2J-B hidden/attention scoring")
    if not is_module_available("bitsandbytes"):
        raise RuntimeError(
            "blocked_by_missing_bitsandbytes: 2J-B requires local 4-bit loading via "
            "bitsandbytes. No fp16 fallback, dependency install, or model download was attempted."
        )

    torch = import_torch()
    components = import_transformers_components()
    tokenizer = components["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    quantization_config = components["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_compute_dtype=resolve_torch_dtype("float16", torch),
    )
    model = components["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=resolve_torch_dtype("float16", torch),
    )
    model.eval()
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "model_path": str(model_path),
        "load_in_4bit": True,
        "attn_implementation": "eager",
    }


class ForwardResult(dict):
    """Typed dict-like container used to keep imports simple."""


def forward_hidden_attention(
    context: dict[str, Any],
    text: str,
    *,
    char_ranges: list[list[int]],
    layer_indices: list[int],
    mode: str,
) -> ForwardResult:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(start), int(end)] for start, end in encoded.pop("offset_mapping")[0].tolist()]
    target_device = infer_model_input_device(model, "auto", torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    hidden_tensor, resolved_hidden, hidden_warnings = _select_hidden_layers(
        outputs.hidden_states,
        layer_indices,
        torch,
    )
    attention_stack = af.head_average_selected_layers(outputs.attentions, layer_indices)
    slot_indices = token_indices_for_char_ranges(offsets, char_ranges, exclude=set())
    warnings = list(hidden_warnings)
    if not slot_indices:
        warnings.append(
            {
                "warning_type": f"{mode}_slot_token_alignment_missing",
                "message": f"no tokens overlapped char ranges {char_ranges}",
            }
        )
    return ForwardResult(
        text=text,
        offsets=offsets,
        slot_indices=slot_indices,
        hidden=hidden_tensor.cpu(),
        attention=attention_stack,
        resolved_hidden_state_indices=resolved_hidden,
        warnings=warnings,
    )


def _select_hidden_layers(
    hidden_states: tuple[Any, ...] | list[Any],
    model_layer_indices: list[int],
    torch: Any,
) -> tuple[Any, list[int], list[dict[str, str]]]:
    resolved: list[int] = []
    warnings: list[dict[str, str]] = []
    for layer_index in model_layer_indices:
        hidden_state_index = layer_index + 1
        if hidden_state_index < len(hidden_states):
            resolved.append(hidden_state_index)
        elif layer_index < len(hidden_states):
            resolved.append(layer_index)
            warnings.append(
                {
                    "warning_type": "hidden_layer_index_fallback",
                    "message": (
                        f"used hidden_states[{layer_index}] because hidden_states[{hidden_state_index}] "
                        "was unavailable"
                    ),
                }
            )
        else:
            warnings.append(
                {
                    "warning_type": "hidden_layer_index_out_of_range",
                    "message": (
                        f"requested model layer {layer_index} unavailable; "
                        f"hidden_states length={len(hidden_states)}"
                    ),
                }
            )
    if not resolved:
        raise ValueError(
            f"No requested hidden layers are available: requested={model_layer_indices}, "
            f"hidden_states length={len(hidden_states)}"
        )
    tensors = [hidden_states[index].detach().squeeze(0) for index in resolved]
    return torch.stack(tensors, dim=0), resolved, warnings


def _mask_char_range(masked_question: str, mask_text: str) -> list[int]:
    start = masked_question.find(mask_text)
    if start < 0:
        raise ValueError(f"mask text {mask_text!r} was not found in masked question")
    return [start, start + len(mask_text)]


def token_indices_for_char_ranges(
    offsets: list[list[int]],
    char_ranges: list[list[int]],
    *,
    exclude: set[int],
) -> list[int]:
    indices: list[int] = []
    for token_index, token_range in enumerate(offsets):
        if token_index in exclude or not token_range or len(token_range) != 2:
            continue
        token_start, token_end = token_range
        if token_end <= token_start:
            continue
        for char_start, char_end in char_ranges:
            if token_start < char_end and char_start < token_end:
                indices.append(token_index)
                break
    return indices


def build_feature_record(
    flat_record: dict[str, Any],
    original: ForwardResult,
    masked: ForwardResult,
    *,
    layer_indices: list[int],
) -> dict[str, Any]:
    hidden_features = compute_hidden_features(original, masked)
    context = context_indices_for_question(
        flat_record["question"],
        original["offsets"],
        exclude=set(original.get("slot_indices") or []),
    )
    attention_built = af.build_attention_features(
        original["attention"],
        original.get("slot_indices") or [],
        masked["attention"],
        masked.get("slot_indices") or [],
        layer_indices,
        context_orig=context,
    )
    attention_features = task_attention_feature_view(attention_built["features"])
    warnings = list(original.get("warnings") or []) + list(masked.get("warnings") or [])
    return {
        "backend": BACKEND,
        "source_question_id": flat_record["source_question_id"],
        "span_id": flat_record["span_id"],
        "span_text": flat_record["span_text"],
        "span_type": flat_record["span_type"],
        "question": flat_record["question"],
        "masked_question": flat_record["masked_question"],
        "mask_text": flat_record.get("mask_text", DEFAULT_MASK_TOKEN),
        "span_char_start": flat_record["span_char_start"],
        "span_char_end": flat_record["span_char_end"],
        "surface_features": flat_record.get("surface_features", {}),
        "hidden_features": hidden_features,
        "attention_features": attention_features,
        "raw_attention_feature_values": attention_built["features"],
        "missing_context": attention_built.get("missing", {}),
        "diagnostic_labels_for_eval_only": flat_record.get("diagnostic_labels_for_eval_only", {}),
        "alignment": {
            "alignment_status": "warning" if warnings else "ok",
            "num_original_slot_tokens": len(original.get("slot_indices") or []),
            "num_masked_slot_tokens": len(masked.get("slot_indices") or []),
            "original_slot_token_indices": original.get("slot_indices") or [],
            "mask_token_indices": masked.get("slot_indices") or [],
            "resolved_hidden_state_indices": original.get("resolved_hidden_state_indices") or [],
            "model_layer_indices": list(layer_indices),
            "warnings": warnings,
        },
    }


def compute_hidden_features(original: ForwardResult, masked: ForwardResult) -> dict[str, float | None]:
    torch = import_torch()
    orig_hidden = original["hidden"]
    masked_hidden = masked["hidden"]
    orig_question = orig_hidden.mean(dim=1)
    masked_question = masked_hidden.mean(dim=1)
    orig_span = _pool_slot(orig_hidden, original.get("slot_indices") or [])
    masked_slot = _pool_slot(masked_hidden, masked.get("slot_indices") or [])

    question_l2 = _layer_l2(orig_question, masked_question)
    question_rel = _layer_relative_norm(orig_question, masked_question)
    features: dict[str, float | None] = {
        "question_context_shift_norm": _mean(question_rel),
        "early_mid_late_delta_slope": _slope(question_rel),
        "max_delta_layer": float(_argmax(question_rel)),
    }
    if orig_span is not None and masked_slot is not None:
        span_l2 = _layer_l2(orig_span, masked_slot)
        span_rel = _layer_relative_norm(orig_span, masked_slot)
        span_cos = _layer_cosine(orig_span, masked_slot)
        features.update(
            {
                "hidden_delta_l2": _mean(span_l2),
                "hidden_delta_relative_norm": _mean(span_rel),
                "hidden_delta_cosine": _mean(span_cos),
                "span_context_shift_norm": _mean(span_rel),
                "cross_layer_stability": 1.0 / (1.0 + _variance(span_rel)),
                "span_to_question_similarity": _mean(_layer_cosine(orig_span, orig_question)),
                "span_to_mask_similarity": _mean(_layer_cosine(masked_slot, masked_question)),
            }
        )
    else:
        features.update(
            {
                "hidden_delta_l2": None,
                "hidden_delta_relative_norm": None,
                "hidden_delta_cosine": None,
                "span_context_shift_norm": None,
                "cross_layer_stability": None,
                "span_to_question_similarity": None,
                "span_to_mask_similarity": None,
            }
        )
    # keep torch referenced in this function to surface missing dependency early
    _ = torch
    return features


def _pool_slot(tensor: Any, slot_indices: list[int]) -> Any | None:
    if not slot_indices:
        return None
    return tensor[:, slot_indices, :].mean(dim=1)


def _layer_l2(first: Any, second: Any) -> list[float]:
    return [float((b - a).norm().item()) for a, b in zip(first, second)]


def _layer_relative_norm(first: Any, second: Any) -> list[float]:
    values: list[float] = []
    for a, b in zip(first, second):
        denom = float(a.norm().item())
        values.append(float((b - a).norm().item()) / denom if denom > 1e-8 else 0.0)
    return values


def _layer_cosine(first: Any, second: Any) -> list[float]:
    values: list[float] = []
    for a, b in zip(first, second):
        an = float(a.norm().item())
        bn = float(b.norm().item())
        if an <= 1e-8 or bn <= 1e-8:
            values.append(0.0)
        else:
            values.append(max(-1.0, min(1.0, float((a * b).sum().item()) / (an * bn))))
    return values


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def _variance(values: list[float | None]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return 0.0
    mean = sum(clean) / len(clean)
    return sum((value - mean) ** 2 for value in clean) / len(clean)


def _slope(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n < 2:
        return None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(clean) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, clean)) / denom if denom else 0.0


def _argmax(values: list[float | None]) -> int:
    clean = [(idx, float(value)) for idx, value in enumerate(values) if value is not None]
    if not clean:
        return -1
    return max(clean, key=lambda pair: pair[1])[0]


def context_indices_for_question(
    question: str,
    offsets: list[list[int]],
    *,
    exclude: set[int],
) -> dict[str, list[int]]:
    return {
        "qfocus": token_indices_for_char_ranges(
            offsets,
            [[m.start(), m.end()] for m in QFOCUS_RE.finditer(question)],
            exclude=exclude,
        ),
        "operation": token_indices_for_char_ranges(
            offsets,
            [[m.start(), m.end()] for m in OPERATION_RE.finditer(question)],
            exclude=exclude,
        ),
        "numctx": token_indices_for_char_ranges(
            offsets,
            [[m.start(), m.end()] for m in NUMBER_RE.finditer(question)],
            exclude=exclude,
        ),
    }


def task_attention_feature_view(raw: dict[str, float]) -> dict[str, float | None]:
    context_values = [
        raw.get("attn_orig_qfocus_to_slot"),
        raw.get("attn_orig_operation_to_slot"),
        raw.get("attn_orig_numctx_to_slot"),
    ]
    finite_context = [float(v) for v in context_values if v is not None and math.isfinite(float(v))]
    delta_values = [
        abs(float(raw.get("attn_delta_slot_in_mass", 0.0))),
        abs(float(raw.get("attn_delta_slot_entropy", 0.0))),
        abs(float(raw.get("attn_delta_slot_rank", 0.0))),
    ]
    return {
        "span_attention_in_mass": _finite_or_none(raw.get("attn_orig_slot_in_mass")),
        "span_attention_out_mass": _finite_or_none(raw.get("attn_orig_slot_out_mass")),
        "span_attention_entropy": _finite_or_none(raw.get("attn_orig_slot_entropy")),
        "span_attention_topk_mass": _finite_or_none(raw.get("attn_orig_slot_top3_mass")),
        "context_to_slot_attention": sum(finite_context) / len(finite_context) if finite_context else None,
        "operation_to_slot_attention": _finite_or_none(raw.get("attn_orig_operation_to_slot")),
        "qfocus_to_slot_attention": _finite_or_none(raw.get("attn_orig_qfocus_to_slot")),
        "number_context_to_slot_attention": _finite_or_none(raw.get("attn_orig_numctx_to_slot")),
        "mask_to_span_attention": _finite_or_none(raw.get("attn_masked_slot_self_mass")),
        "span_to_mask_attention": _finite_or_none(raw.get("attn_masked_slot_self_mass")),
        "original_to_masked_attention_delta": sum(delta_values) / len(delta_values),
        "attention_entropy_delta": _finite_or_none(raw.get("attn_delta_slot_entropy")),
        "attention_rank_delta": _finite_or_none(raw.get("attn_delta_slot_rank")),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build_score_reports(
    feature_records: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    output_files: dict[str, Path] | None = None,
    layer_indices: list[int] | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    files = output_files or _output_files(output_dir)
    grouped = _group_by_question(feature_records)
    score_records = build_score_matrix(feature_records)
    formulas = build_formula_scores(score_records)
    _attach_formula_scores(score_records, formulas)
    ranking_report = build_same_question_ranking_report(score_records, formulas)
    formula_report = build_formula_validation_report(ranking_report, formulas, score_records)
    topk_budget_report = build_topk_budget_report(score_records, formulas)
    failure_cases, success_cases = build_failure_success_cases(score_records, formulas)
    feature_report = build_hidden_attention_feature_report(
        feature_records,
        grouped=grouped,
        layer_indices=layer_indices or list(DEFAULT_LAYER_INDICES),
        model_path=model_path,
    )
    gap_report = build_reasoning_signal_gap_report()
    review_gate = build_review_gate_report(
        ranking_report,
        formula_report,
        topk_budget_report,
        feature_report,
    )

    write_jsonl(score_records, files["scores"])
    write_json(feature_report, files["hidden_attention_report"])
    write_json(ranking_report, files["ranking_report"])
    write_json(formula_report, files["formula_report"])
    write_json(topk_budget_report, files["topk_budget_report"])
    write_jsonl(failure_cases, files["failure_cases"])
    write_jsonl(success_cases, files["success_cases"])
    write_json(gap_report, files["reasoning_gap_report"])
    files["review_gate"].write_text(render_review_gate_md(review_gate), encoding="utf-8")

    return {
        "score_records": score_records,
        "ranking_report": ranking_report,
        "formula_report": formula_report,
        "topk_budget_report": topk_budget_report,
        "feature_report": feature_report,
        "reasoning_gap_report": gap_report,
        "review_gate": review_gate,
    }


def _group_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_question_id"])].append(record)
    return dict(grouped)


def build_score_matrix(feature_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in feature_records:
        surface = record.get("surface_features", {})
        hidden = record.get("hidden_features", {})
        attention = record.get("attention_features", {})
        semantic = semantic_keyness_proxy(record.get("span_type"))
        keyness = _safe_float(surface.get("surface_keyness_proxy"), 0.0)
        hidden_raw = hidden_fragility_raw(hidden)
        attention_raw = attention_fragility_raw(attention)
        row = {
            "source_question_id": record["source_question_id"],
            "span_id": record["span_id"],
            "span_text": record["span_text"],
            "span_type": record["span_type"],
            "keyness_signals": {
                "surface_keyness_proxy": keyness,
                "semantic_keyness_proxy": semantic,
                "within_question_surface_rank": None,
            },
            "fragility_signals": {
                "hidden_fragility_score": hidden_raw,
                "attention_fragility_score": attention_raw,
                "hidden_plus_attention_score": None,
                "within_question_hidden_rank": None,
                "within_question_attention_rank": None,
            },
            "reasoning_signals": {
                "answer_logprob_delta": None,
                "answer_rank_delta": None,
                "trajectory_change": None,
                "cot_path_change": None,
                "nla_semantic_role": None,
            },
            "budget_signals": {
                "priority_score": None,
                "priority_rank_within_question": None,
                "off_path_budget_risk_eval_only": None,
            },
            "diagnostic_labels_for_eval_only": record.get("diagnostic_labels_for_eval_only", {}),
            "alignment": record.get("alignment", {}),
        }
        rows.append(row)

    for group in _group_by_question(rows).values():
        _rank_into(group, lambda r: r["keyness_signals"]["surface_keyness_proxy"], "keyness_signals", "within_question_surface_rank")
        _normalize_group_signal(group, "fragility_signals", "hidden_fragility_score")
        _normalize_group_signal(group, "fragility_signals", "attention_fragility_score")
        for row in group:
            hidden_score = row["fragility_signals"]["hidden_fragility_score"]
            attention_score = row["fragility_signals"]["attention_fragility_score"]
            row["fragility_signals"]["hidden_plus_attention_score"] = _mean_numeric([hidden_score, attention_score])
        _rank_into(group, lambda r: r["fragility_signals"]["hidden_fragility_score"], "fragility_signals", "within_question_hidden_rank")
        _rank_into(group, lambda r: r["fragility_signals"]["attention_fragility_score"], "fragility_signals", "within_question_attention_rank")
    audit_formula_inputs(score_records=rows)
    return rows


def hidden_fragility_raw(features: dict[str, Any]) -> float:
    values = [
        features.get("hidden_delta_relative_norm"),
        features.get("question_context_shift_norm"),
        abs(_safe_float(features.get("early_mid_late_delta_slope"), 0.0)),
        1.0 - _safe_float(features.get("span_to_mask_similarity"), 0.0),
    ]
    return _mean_numeric(values)


def attention_fragility_raw(features: dict[str, Any]) -> float:
    values = [
        abs(_safe_float(features.get("original_to_masked_attention_delta"), 0.0)),
        abs(_safe_float(features.get("attention_entropy_delta"), 0.0)),
        abs(_safe_float(features.get("attention_rank_delta"), 0.0)),
        features.get("span_attention_in_mass"),
        features.get("context_to_slot_attention"),
    ]
    return _mean_numeric(values)


def semantic_keyness_proxy(span_type: str | None) -> float:
    mapping = {
        "number": 0.78,
        "number_unit": 0.82,
        "rate": 0.84,
        "operation": 0.72,
        "comparison": 0.76,
        "condition": 0.70,
        "negation": 0.70,
        "question_target": 0.65,
        "object": 0.28,
    }
    return mapping.get(str(span_type), 0.35)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean_numeric(values: list[Any]) -> float:
    clean = [_safe_float(value, default=float("nan")) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else 0.0


def _normalize_group_signal(group: list[dict[str, Any]], namespace: str, key: str) -> None:
    values = [_safe_float(row[namespace].get(key), 0.0) for row in group]
    if not values:
        return
    lo = min(values)
    hi = max(values)
    denom = hi - lo
    for row, value in zip(group, values):
        row[namespace][key] = (value - lo) / denom if denom > 1e-12 else 0.0


def _rank_into(
    group: list[dict[str, Any]],
    getter: Any,
    namespace: str,
    key: str,
) -> None:
    ordered = sorted(group, key=lambda row: getter(row), reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row[namespace][key] = rank


def build_formula_scores(score_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    formulas: dict[str, dict[str, Any]] = {
        "A_surface_only": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "B_hidden_only": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "C_attention_only": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "D_hidden_plus_attention": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "E_keyness_times_fragility": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "F_keyness_gate_then_fragility": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "G_per_question_normalized_priority": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "H_span_type_budget_policy": {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}},
        "I_oracle_diagnostic_upper_bound": {
            "gate_eligible": False,
            "uses_eval_only_labels": True,
            "not_gate_eligible": True,
            "scores": {},
        },
    }
    grouped = _group_by_question(score_records)
    for group in grouped.values():
        raw_g: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in group:
            sid = row["span_id"]
            keyness = keyness_score(row)
            hidden = _safe_float(row["fragility_signals"].get("hidden_fragility_score"), 0.0)
            attention = _safe_float(row["fragility_signals"].get("attention_fragility_score"), 0.0)
            fragility = _mean_numeric([hidden, attention])
            raw_values = {
                "A_surface_only": row["keyness_signals"]["surface_keyness_proxy"],
                "B_hidden_only": hidden,
                "C_attention_only": attention,
                "D_hidden_plus_attention": fragility,
                "E_keyness_times_fragility": keyness * fragility,
                "F_keyness_gate_then_fragility": fragility if keyness >= 0.45 else 0.05 * fragility,
                "G_per_question_normalized_priority": keyness * fragility,
                "H_span_type_budget_policy": span_type_budget_score(row, keyness, fragility),
                "I_oracle_diagnostic_upper_bound": oracle_diagnostic_score(row, fragility),
            }
            for name, value in raw_values.items():
                raw_g[name].append((sid, float(value)))
        normalized_g = _normalize_formula_group(raw_g.get("G_per_question_normalized_priority", []))
        for name, values in raw_g.items():
            if name == "G_per_question_normalized_priority":
                for sid, value in normalized_g.items():
                    formulas[name]["scores"][sid] = value
            else:
                for sid, value in values:
                    formulas[name]["scores"][sid] = value
    return formulas


def keyness_score(row: dict[str, Any]) -> float:
    return _mean_numeric(
        [
            row["keyness_signals"].get("surface_keyness_proxy"),
            row["keyness_signals"].get("semantic_keyness_proxy"),
        ]
    )


def span_type_budget_score(row: dict[str, Any], keyness: float, fragility: float) -> float:
    span_type = str(row.get("span_type"))
    score = keyness * fragility
    if span_type == "object":
        score *= 0.55
    elif span_type in {"comparison", "negation", "condition"}:
        score *= 1.15
    elif span_type in {"number", "number_unit", "rate"}:
        score *= 0.96
    return score


def oracle_diagnostic_score(row: dict[str, Any], fragility: float) -> float:
    labels = row.get("diagnostic_labels_for_eval_only", {})
    sol = labels.get("solution_path_status")
    if sol == "on_solution_path_number":
        key = 1.0
    elif sol == "off_solution_path_number":
        key = 0.0
    elif labels.get("weak_semantic_keyness") in POSITIVE_WEAK_KEYNESS:
        key = 0.75
    else:
        key = 0.2
    return key * fragility


def _normalize_formula_group(values: list[tuple[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(value for _sid, value in values)
    hi = max(value for _sid, value in values)
    denom = hi - lo
    return {
        sid: ((value - lo) / denom if denom > 1e-12 else 0.0)
        for sid, value in values
    }


def _attach_formula_scores(score_records: list[dict[str, Any]], formulas: dict[str, dict[str, Any]]) -> None:
    grouped = _group_by_question(score_records)
    for row in score_records:
        sid = row["span_id"]
        row["formula_scores"] = {
            name: spec["scores"].get(sid)
            for name, spec in formulas.items()
        }
    for group in grouped.values():
        for name in formulas:
            ordered = sorted(group, key=lambda row: row["formula_scores"].get(name, 0.0), reverse=True)
            for rank, row in enumerate(ordered, start=1):
                row.setdefault("formula_ranks", {})[name] = rank
        best = "E_keyness_times_fragility"
        for row in group:
            row["budget_signals"]["priority_score"] = row["formula_scores"][best]
            row["budget_signals"]["priority_rank_within_question"] = row["formula_ranks"][best]
            if is_off_path_number(row):
                row["budget_signals"]["off_path_budget_risk_eval_only"] = row["formula_ranks"][best]


def build_same_question_ranking_report(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "num_questions": len(_group_by_question(score_records)),
        "num_spans": len(score_records),
        "formulas": {},
    }
    for name, spec in formulas.items():
        metrics = formula_metrics(score_records, spec["scores"], top_k=3)
        report["formulas"][name] = metrics
    return report


def formula_metrics(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    top_k: int,
) -> dict[str, Any]:
    grouped = _group_by_question(rows)
    pairwise = []
    on_ranks = []
    off_ranks = []
    top_hits = {1: [], 2: [], 3: []}
    bucket_hits = {1: [], 2: [], 3: []}
    off_path_selected = 0
    selected_total = 0
    on_path_topk = []
    off_path_fp = []
    type_coverage: dict[str, list[bool]] = defaultdict(list)
    span_type_selected: Counter[str] = Counter()

    for group in grouped.values():
        ordered = sorted(group, key=lambda row: scores.get(row["span_id"], 0.0), reverse=True)
        ranks = {row["span_id"]: rank for rank, row in enumerate(ordered, start=1)}
        positives = [row for row in group if is_key_positive(row)]
        bucket3 = [row for row in group if is_bucket3(row)]
        if positives:
            for k in top_hits:
                top_hits[k].append(any(is_key_positive(row) for row in ordered[:k]))
        if bucket3:
            for k in bucket_hits:
                bucket_hits[k].append(any(is_bucket3(row) for row in ordered[:k]))

        on_rows = [row for row in group if is_on_path_number(row)]
        off_rows = [row for row in group if is_off_path_number(row)]
        for on in on_rows:
            on_ranks.append(ranks[on["span_id"]])
        for off in off_rows:
            off_ranks.append(ranks[off["span_id"]])
        for on in on_rows:
            for off in off_rows:
                pairwise.append(_pairwise_value(scores.get(on["span_id"], 0.0), scores.get(off["span_id"], 0.0)))
        if on_rows:
            on_path_topk.append(any(is_on_path_number(row) for row in ordered[:top_k]))
        if off_rows:
            off_path_fp.append(any(is_off_path_number(row) for row in ordered[:top_k]))
        for row in ordered[:top_k]:
            selected_total += 1
            span_type_selected[str(row.get("span_type"))] += 1
            if is_off_path_number(row):
                off_path_selected += 1
        for span_type in ["comparison", "negation", "condition", "operation", "question_target"]:
            candidates = [row for row in group if row.get("span_type") == span_type]
            if candidates:
                type_coverage[span_type].append(any(row.get("span_type") == span_type for row in ordered[:top_k]))

    return {
        "same_question_pairwise_accuracy": _mean_numeric(pairwise),
        "same_question_on_path_vs_off_path_auc": _mean_numeric(pairwise),
        "same_question_bucket3_vs_bucket1_pairwise": bucket3_vs_bucket1_pairwise(rows, scores),
        "per_question_top1_key_hit": _bool_mean(top_hits[1]),
        "per_question_top2_key_hit": _bool_mean(top_hits[2]),
        "per_question_top3_key_hit": _bool_mean(top_hits[3]),
        "per_question_top1_bucket3_hit": _bool_mean(bucket_hits[1]),
        "per_question_top2_bucket3_hit": _bool_mean(bucket_hits[2]),
        "per_question_top3_bucket3_hit": _bool_mean(bucket_hits[3]),
        "top_k_off_path_number_selected_rate": off_path_selected / selected_total if selected_total else 0.0,
        "off_path_budget_share": off_path_selected / selected_total if selected_total else 0.0,
        "span_type_budget_distribution": dict(span_type_selected),
        "on_path_number_rank_mean": _mean_numeric(on_ranks),
        "off_path_number_rank_mean": _mean_numeric(off_ranks),
        "on_path_number_topk_coverage": _bool_mean(on_path_topk),
        "off_path_number_false_positive_rate": _bool_mean(off_path_fp),
        "comparison_topk_coverage": _bool_mean(type_coverage["comparison"]),
        "negation_topk_coverage": _bool_mean(type_coverage["negation"]),
        "condition_topk_coverage": _bool_mean(type_coverage["condition"]),
        "operation_topk_coverage": _bool_mean(type_coverage["operation"]),
        "question_target_topk_coverage": _bool_mean(type_coverage["question_target"]),
        "num_pairwise_on_off_number_pairs": len(pairwise),
        "num_questions_with_key_positive": sum(1 for group in grouped.values() if any(is_key_positive(row) for row in group)),
    }


def _pairwise_value(pos_score: float, neg_score: float) -> float:
    if pos_score > neg_score:
        return 1.0
    if pos_score == neg_score:
        return 0.5
    return 0.0


def _bool_mean(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def bucket3_vs_bucket1_pairwise(rows: list[dict[str, Any]], scores: dict[str, float]) -> float | None:
    values = []
    for group in _group_by_question(rows).values():
        high = [row for row in group if fragility_bucket(row) == 3]
        low = [row for row in group if fragility_bucket(row) == 1]
        for hi in high:
            for lo in low:
                values.append(_pairwise_value(scores.get(hi["span_id"], 0.0), scores.get(lo["span_id"], 0.0)))
    return _mean(values)


def fragility_bucket(row: dict[str, Any]) -> int | None:
    value = row.get("diagnostic_labels_for_eval_only", {}).get("fragility_bucket_if_available")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_on_path_number(row: dict[str, Any]) -> bool:
    return row.get("diagnostic_labels_for_eval_only", {}).get("solution_path_status") == "on_solution_path_number"


def is_off_path_number(row: dict[str, Any]) -> bool:
    return row.get("diagnostic_labels_for_eval_only", {}).get("solution_path_status") == "off_solution_path_number"


def is_bucket3(row: dict[str, Any]) -> bool:
    return fragility_bucket(row) == 3


def is_key_positive(row: dict[str, Any]) -> bool:
    labels = row.get("diagnostic_labels_for_eval_only", {})
    return (
        labels.get("solution_path_status") == "on_solution_path_number"
        or labels.get("weak_semantic_keyness") in POSITIVE_WEAK_KEYNESS
    )


def build_formula_validation_report(
    ranking_report: dict[str, Any],
    formulas: dict[str, dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = ranking_report["formulas"].get("A_surface_only", {})
    baseline_auc = _safe_float(baseline.get("same_question_on_path_vs_off_path_auc"), 0.0)
    formula_rows: dict[str, Any] = {}
    best_name = None
    best_delta = -float("inf")
    for name, spec in formulas.items():
        metrics = ranking_report["formulas"].get(name, {})
        auc = _safe_float(metrics.get("same_question_on_path_vs_off_path_auc"), 0.0)
        delta = auc - baseline_auc
        bootstrap = None
        if spec.get("gate_eligible"):
            bootstrap = grouped_bootstrap_delta(
                score_records,
                formulas,
                name,
                baseline_name="A_surface_only",
            )
            if delta > best_delta and name != "A_surface_only":
                best_delta = delta
                best_name = name
        formula_rows[name] = {
            "gate_eligible": bool(spec.get("gate_eligible")),
            "uses_eval_only_labels": bool(spec.get("uses_eval_only_labels")),
            "same_question_on_path_vs_off_path_auc": auc,
            "delta_vs_surface_only_auc": delta,
            "bootstrap_delta_ci95": bootstrap,
            "per_question_top3_key_hit": metrics.get("per_question_top3_key_hit"),
            "off_path_budget_share": metrics.get("off_path_budget_share"),
        }
    return {
        "backend": BACKEND,
        "baseline_formula": "A_surface_only",
        "best_non_oracle_formula": best_name,
        "best_non_oracle_delta_vs_surface_only_auc": best_delta if best_name else None,
        "formulas": formula_rows,
        "formula_input_leakage_audit": audit_formula_inputs(
            formula_input_names=formula_input_feature_names()
        ),
        "oracle_formula": {
            "name": "I_oracle_diagnostic_upper_bound",
            "oracle_diagnostic_only": True,
            "uses_eval_only_labels": True,
            "not_gate_eligible": True,
        },
    }


def grouped_bootstrap_delta(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
    formula_name: str,
    *,
    baseline_name: str,
    num_samples: int = 1000,
    seed: int = 2302,
) -> dict[str, float] | None:
    grouped = _group_by_question(score_records)
    formula_scores = formulas[formula_name]["scores"]
    baseline_scores = formulas[baseline_name]["scores"]
    question_deltas = []
    for group in grouped.values():
        formula_pairs = []
        baseline_pairs = []
        on_rows = [row for row in group if is_on_path_number(row)]
        off_rows = [row for row in group if is_off_path_number(row)]
        for on in on_rows:
            for off in off_rows:
                formula_pairs.append(
                    _pairwise_value(
                        formula_scores.get(on["span_id"], 0.0),
                        formula_scores.get(off["span_id"], 0.0),
                    )
                )
                baseline_pairs.append(
                    _pairwise_value(
                        baseline_scores.get(on["span_id"], 0.0),
                        baseline_scores.get(off["span_id"], 0.0),
                    )
                )
        if formula_pairs and baseline_pairs:
            question_deltas.append(_mean_numeric(formula_pairs) - _mean_numeric(baseline_pairs))
    if not question_deltas:
        return None

    rng = np.random.default_rng(seed)
    samples = []
    n = len(question_deltas)
    for _ in range(num_samples):
        indices = rng.integers(0, n, size=n)
        samples.append(float(np.mean([question_deltas[int(index)] for index in indices])))
    samples_sorted = sorted(samples)
    low_index = int(0.025 * (len(samples_sorted) - 1))
    high_index = int(0.975 * (len(samples_sorted) - 1))
    delta = float(np.mean(question_deltas))
    return {
        "mean_delta": delta,
        "ci95_low": samples_sorted[low_index],
        "ci95_high": samples_sorted[high_index],
        "num_questions_with_on_off_pairs": n,
        "num_bootstrap_samples": num_samples,
        "seed": seed,
        "method_note": "question-grouped bootstrap over on-path/off-path number pairwise deltas",
    }


def formula_input_feature_names() -> list[str]:
    return [
        "surface_keyness_proxy",
        "semantic_keyness_proxy",
        "within_question_surface_rank",
        "hidden_fragility_score",
        "attention_fragility_score",
        "hidden_plus_attention_score",
        "within_question_hidden_rank",
        "within_question_attention_rank",
        "priority_score",
        "priority_rank_within_question",
    ]


def audit_formula_inputs(
    *,
    score_records: list[dict[str, Any]] | None = None,
    formula_input_names: list[str] | None = None,
) -> dict[str, Any]:
    names = set(formula_input_names or [])
    if score_records:
        names.update(formula_input_feature_names())
    leaked = [
        name
        for name in sorted(names)
        for banned in FORBIDDEN_FORMULA_INPUT_SUBSTRINGS
        if banned in name
    ]
    if leaked:
        raise AssertionError(f"formula input leakage detected: {leaked[:10]}")
    return {
        "passed": True,
        "num_formula_input_names_checked": len(names),
        "forbidden_substrings": list(FORBIDDEN_FORMULA_INPUT_SUBSTRINGS),
        "leaked_input_features": [],
    }


def build_topk_budget_report(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = _group_by_question(score_records)
    for name, spec in formulas.items():
        selected: Counter[str] = Counter()
        off_path = 0
        total = 0
        for group in grouped.values():
            ordered = sorted(group, key=lambda row: spec["scores"].get(row["span_id"], 0.0), reverse=True)
            for row in ordered[:top_k]:
                selected[str(row.get("span_type"))] += 1
                off_path += 1 if is_off_path_number(row) else 0
                total += 1
        out[name] = {
            "top_k": top_k,
            "span_type_budget_distribution": dict(selected),
            "off_path_budget_share": off_path / total if total else 0.0,
            "selected_span_count": total,
        }
    return {
        "backend": BACKEND,
        "formulas": out,
        "policy_note": (
            "Span-type budget policy is diagnostic only; no attention steering or Sprint 3A entry "
            "is performed."
        ),
    }


def build_failure_success_cases(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
    *,
    formula_name: str = "E_keyness_times_fragility",
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = formulas[formula_name]
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for qid, group in _group_by_question(score_records).items():
        ordered = sorted(group, key=lambda row: spec["scores"].get(row["span_id"], 0.0), reverse=True)
        expected = [row for row in group if is_key_positive(row)]
        top = ordered[:top_k]
        success = bool(expected) and any(is_key_positive(row) for row in top)
        case = {
            "source_question_id": qid,
            "question": group[0].get("question"),
            "candidate_spans": [_case_span(row, formulas) for row in group],
            "selected_topk_spans": [_case_span(row, formulas) for row in top],
            "expected_priority_spans_eval_only": [_case_span(row, formulas) for row in expected],
            "scores_by_formula": {
                name: {row["span_id"]: form["scores"].get(row["span_id"]) for row in group}
                for name, form in formulas.items()
            },
            "ranks_by_formula": {
                name: _ranks_for_group(group, form["scores"])
                for name, form in formulas.items()
            },
            "diagnostic_labels": {
                row["span_id"]: row.get("diagnostic_labels_for_eval_only", {})
                for row in group
            },
            "failure_reason_auto_guess": None if success else guess_failure_reason(top, expected),
        }
        if success:
            successes.append(case)
        else:
            failures.append(case)
    failures = sorted(failures, key=lambda case: len(case["expected_priority_spans_eval_only"]), reverse=True)[:30]
    successes = successes[:30]
    return failures, successes


def _case_span(row: dict[str, Any], formulas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "span_id": row["span_id"],
        "span_text": row["span_text"],
        "span_type": row["span_type"],
        "formula_scores": {
            name: spec["scores"].get(row["span_id"])
            for name, spec in formulas.items()
        },
        "diagnostic_labels_for_eval_only": row.get("diagnostic_labels_for_eval_only", {}),
    }


def _ranks_for_group(group: list[dict[str, Any]], scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(group, key=lambda row: scores.get(row["span_id"], 0.0), reverse=True)
    return {row["span_id"]: rank for rank, row in enumerate(ordered, start=1)}


def guess_failure_reason(top: list[dict[str, Any]], expected: list[dict[str, Any]]) -> str:
    if any(is_off_path_number(row) for row in top):
        return "off_path_number_overranked"
    if any(is_on_path_number(row) for row in expected):
        return "on_path_number_underranked"
    expected_types = {row.get("span_type") for row in expected}
    for span_type, reason in [
        ("comparison", "comparison_underranked"),
        ("negation", "negation_underranked"),
        ("condition", "condition_underranked"),
        ("question_target", "question_target_underranked"),
    ]:
        if span_type in expected_types:
            return reason
    if any(row.get("span_type") == "object" for row in top):
        return "object_overranked"
    if not expected:
        return "label_ambiguous"
    return "missing_reasoning_signal"


def build_hidden_attention_feature_report(
    feature_records: list[dict[str, Any]],
    *,
    grouped: dict[str, list[dict[str, Any]]],
    layer_indices: list[int],
    model_path: str | Path | None,
) -> dict[str, Any]:
    hidden_names = sorted({name for row in feature_records for name in row.get("hidden_features", {})})
    attention_names = sorted({name for row in feature_records for name in row.get("attention_features", {})})
    missing_context_counts = Counter()
    warning_counts = Counter()
    for row in feature_records:
        for name, value in row.get("missing_context", {}).items():
            if value:
                missing_context_counts[name] += 1
        for warning in row.get("alignment", {}).get("warnings", []):
            warning_counts[warning.get("warning_type", "unknown_warning")] += 1
    leakage = feature_name_leakage_audit(hidden_names + attention_names)
    return {
        "backend": BACKEND,
        "model_path": str(model_path) if model_path is not None else None,
        "load_in_4bit": True,
        "attn_implementation": "eager",
        "layer_indices": list(layer_indices),
        "attention_final_layer_excluded": True,
        "num_questions": len(grouped),
        "num_feature_records": len(feature_records),
        "hidden_feature_names": hidden_names,
        "attention_feature_names": attention_names,
        "missing_context_counts": dict(missing_context_counts),
        "alignment_warning_counts": dict(warning_counts),
        "feature_name_leakage_audit": leakage,
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }


def feature_name_leakage_audit(names: list[str]) -> dict[str, Any]:
    leaked = [
        {"feature_name": name, "forbidden_substring": token}
        for name in sorted(names)
        for token in FORBIDDEN_FORMULA_INPUT_SUBSTRINGS
        if token in name
    ]
    return {
        "passed": len(leaked) == 0,
        "num_names_checked": len(set(names)),
        "leaked_input_features": leaked,
        "forbidden_substrings": list(FORBIDDEN_FORMULA_INPUT_SUBSTRINGS),
    }


def build_reasoning_signal_gap_report() -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "reasoning_signals_status": {
            "answer_logprob_delta": "missing_not_computed_in_2J",
            "answer_rank_delta": "missing_not_computed_in_2J",
            "trajectory_change": "missing_not_computed_in_2J",
            "cot_path_change": "missing_not_computed_in_2J",
            "nla_semantic_role": "missing_not_computed_in_2J",
            "causal_attribution": "missing_not_computed_in_2J",
        },
        "interpretation": (
            "2J-B validates text/hidden/attention formulas only. If same-question ranking remains weak, "
            "the next sprint should add a reasoning-level signal rather than entering attention steering."
        ),
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }


def build_review_gate_report(
    ranking_report: dict[str, Any],
    formula_report: dict[str, Any],
    topk_budget_report: dict[str, Any],
    feature_report: dict[str, Any],
) -> dict[str, Any]:
    formulas = ranking_report.get("formulas", {})
    surface = formulas.get("A_surface_only", {})
    best_name = formula_report.get("best_non_oracle_formula")
    best = formulas.get(best_name or "", {})
    best_delta = formula_report.get("best_non_oracle_delta_vs_surface_only_auc")
    hidden_plus = formulas.get("D_hidden_plus_attention", {})
    key_frag = formulas.get("E_keyness_times_fragility", {})
    checks = {
        "1_same_question_pairwise_ranking_computable": {
            "pass": any(
                metrics.get("num_pairwise_on_off_number_pairs", 0) > 0
                for metrics in formulas.values()
            ),
            "value": max((metrics.get("num_pairwise_on_off_number_pairs", 0) for metrics in formulas.values()), default=0),
        },
        "2_non_oracle_formula_improves_over_surface": {
            "pass": best_delta is not None and best_delta > 0.01,
            "value": best_delta,
        },
        "3_keyness_times_fragility_or_gated_improves_single_score": {
            "pass": _safe_float(key_frag.get("same_question_on_path_vs_off_path_auc"), 0.0)
            >= _safe_float(hidden_plus.get("same_question_on_path_vs_off_path_auc"), 0.0),
            "value": {
                "keyness_times_fragility": key_frag.get("same_question_on_path_vs_off_path_auc"),
                "hidden_plus_attention": hidden_plus.get("same_question_on_path_vs_off_path_auc"),
            },
        },
        "4_off_path_budget_share_decreases_or_not_worse": {
            "pass": best
            and _safe_float(best.get("off_path_budget_share"), 1.0)
            <= _safe_float(surface.get("off_path_budget_share"), 1.0),
            "value": {
                "surface": surface.get("off_path_budget_share"),
                "best": best.get("off_path_budget_share") if best else None,
            },
        },
        "5_on_path_number_topk_coverage_improves": {
            "pass": best
            and _safe_float(best.get("on_path_number_topk_coverage"), 0.0)
            >= _safe_float(surface.get("on_path_number_topk_coverage"), 0.0),
            "value": {
                "surface": surface.get("on_path_number_topk_coverage"),
                "best": best.get("on_path_number_topk_coverage") if best else None,
            },
        },
        "6_comparison_negation_condition_not_systematically_suppressed": {
            "pass": _non_number_coverage_ok(best),
            "value": {
                "comparison": best.get("comparison_topk_coverage") if best else None,
                "negation": best.get("negation_topk_coverage") if best else None,
                "condition": best.get("condition_topk_coverage") if best else None,
            },
        },
        "7_grouped_resampling_stability_recorded": {
            "pass": all(
                row.get("bootstrap_delta_ci95") is not None
                for name, row in formula_report.get("formulas", {}).items()
                if row.get("gate_eligible") and name != "A_surface_only"
            ),
            "value": "deterministic grouped delta placeholder",
        },
        "8_no_leakage": {
            "pass": bool(feature_report.get("feature_name_leakage_audit", {}).get("passed"))
            and bool(formula_report.get("formula_input_leakage_audit", {}).get("passed")),
            "value": {
                "feature_leakage": feature_report.get("feature_name_leakage_audit"),
                "formula_input_leakage": formula_report.get("formula_input_leakage_audit"),
            },
        },
    }
    passed_count = sum(1 for check in checks.values() if check["pass"])
    return {
        "backend": BACKEND,
        "checks": checks,
        "num_checks_passed": passed_count,
        "num_checks_total": len(checks),
        "passed": passed_count == len(checks),
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
        "best_non_oracle_formula": best_name,
        "recommendation": recommendation_from_gate(checks),
        "topk_budget_report_summary": topk_budget_report.get("formulas", {}),
    }


def _non_number_coverage_ok(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    values = [
        metrics.get("comparison_topk_coverage"),
        metrics.get("negation_topk_coverage"),
        metrics.get("condition_topk_coverage"),
    ]
    present = [value for value in values if value is not None]
    return bool(present) and sum(1 for value in present if value >= 0.05) >= 1


def recommendation_from_gate(checks: dict[str, dict[str, Any]]) -> str:
    if not checks["1_same_question_pairwise_ranking_computable"]["pass"]:
        return "fix multi-span ranking substrate before adding reasoning signals"
    if not checks["2_non_oracle_formula_improves_over_surface"]["pass"]:
        return "run a formula validation or reasoning-signal sprint; do not enter Sprint 3A"
    if not checks["6_comparison_negation_condition_not_systematically_suppressed"]["pass"]:
        return "add semantic reasoning-role signals before steering"
    return "formula validation sprint; do not enter Sprint 3A yet"


def render_review_gate_md(review_gate: dict[str, Any]) -> str:
    lines = [
        "# Sprint 2J-B Multi-Span Scoring Review Gate",
        "",
        "Verdict:",
        f"- passed: {str(review_gate['passed']).lower()}",
        f"- checks: {review_gate['num_checks_passed']}/{review_gate['num_checks_total']}",
        f"- ready_for_2000_rerun: {str(review_gate['ready_for_2000_rerun']).lower()}",
        f"- do_not_enter_sprint_3A: {str(review_gate['do_not_enter_sprint_3A']).lower()}",
        "",
        "Coverage:",
        "- 2J-A gate passed before 2J-B scoring was allowed.",
        "",
        "Same-question ranking:",
        f"- best_non_oracle_formula: {review_gate.get('best_non_oracle_formula')}",
        "",
        "Formula comparison:",
    ]
    for name, check in review_gate["checks"].items():
        lines.append(f"- {name}: pass={str(check['pass']).lower()} value={json.dumps(check['value'], ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "Budget analysis:",
            "- See topk_budget_report.json.",
            "",
            "Failure cases:",
            "- See failure_case_report.jsonl and success_case_report.jsonl.",
            "",
            "Reasoning signal gap:",
            "- answer-logprob, CoT, trajectory, NLA, and causal attribution signals were not computed in Sprint 2J.",
            "",
            "Root cause:",
            "- This sprint evaluates same-question span ranking, not attention steering.",
            "",
            "Recommendation:",
            f"- {review_gate['recommendation']}",
            "",
            "Next sprint candidate:",
            "- formula validation sprint or reasoning-signal sprint; do not enter Sprint 3A directly.",
            "",
        ]
    )
    return "\n".join(lines)
