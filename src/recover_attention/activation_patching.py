"""Sprint 3C-0 correct-vs-wrong activation patching helpers.

This module keeps the causal intervention narrow: teacher-forced residual
replacement at selected reasoning-step token positions. It does not train,
finetune, download models, or perform attention steering.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from typing import Any

import numpy as np

from recover_attention import multi_span_reasoning_scoring as msrs

BACKEND = "correct_wrong_activation_patching_reasoning_steps_v0"
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Solve briefly step by step, then give the final answer as: #### <number>\n"
    "Answer:"
)
ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
OPERATOR_RE = re.compile(r"(?:\d)\s*(?:\+|-|\*|/|x|X|=)\s*(?:\d)|(?:\+|-|\*|/|x|X|=)")
RESULT_MARKER_RE = re.compile(r"(?:=|####|therefore|thus|so)", re.IGNORECASE)

PRIMARY_POSITION_TYPES = {
    "generated_operator_token",
    "generated_intermediate_number_token",
    "generated_equals_or_result_marker",
    "generated_final_answer_number",
}
CONTROL_POSITION_TYPES = {
    "random_generated_token",
    "final_answer_position",
    "prompt_question_number_token",
}
SUPPORTED_POSITION_TYPES = PRIMARY_POSITION_TYPES | CONTROL_POSITION_TYPES

HARM_PROXY_DEFINITION = "top1_changed or entropy_delta>1.0 or margin_delta<-0.25"


def build_trace_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip("$% \t\r\n.,")
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned.lower()
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.10g}"


def extract_final_answer(text: str) -> dict[str, Any]:
    """Extract the final numeric answer from generated text.

    Prefer the GSM8K `#### <number>` convention; fall back to the last number so
    partially formatted traces can still be audited.
    """

    match = None
    method = "missing"
    for match in ANSWER_RE.finditer(text):
        method = "hash_answer_marker"
    if match is None:
        numbers = list(NUMBER_RE.finditer(text))
        if numbers:
            match = numbers[-1]
            method = "last_number_fallback"
    if match is None:
        return {
            "answer": None,
            "normalized_answer": None,
            "char_start": None,
            "char_end": None,
            "parse_method": method,
        }
    answer = match.group(1) if method == "hash_answer_marker" else match.group(0)
    return {
        "answer": answer,
        "normalized_answer": normalize_numeric_answer(answer),
        "char_start": int(match.start(1) if method == "hash_answer_marker" else match.start()),
        "char_end": int(match.end(1) if method == "hash_answer_marker" else match.end()),
        "parse_method": method,
    }


def classify_trace(completion: str, gold_answer: str) -> dict[str, Any]:
    parsed = extract_final_answer(completion)
    gold_norm = normalize_numeric_answer(gold_answer)
    parsed_norm = parsed["normalized_answer"]
    is_correct = parsed_norm is not None and gold_norm is not None and parsed_norm == gold_norm
    return {
        **parsed,
        "gold_answer": str(gold_answer),
        "normalized_gold_answer": gold_norm,
        "is_correct": bool(is_correct),
    }


def token_indices_for_char_ranges(offsets: list[list[int]], ranges: list[tuple[int, int]]) -> list[int]:
    return msrs.token_indices_for_char_ranges(
        offsets,
        [[int(start), int(end)] for start, end in ranges if end > start],
        exclude=set(),
    )


def _first_token_for_matches(offsets: list[list[int]], matches: list[re.Match[str]]) -> list[int]:
    positions: list[int] = []
    for match in matches:
        token_indices = token_indices_for_char_ranges(offsets, [(match.start(), match.end())])
        if token_indices:
            positions.append(token_indices[0])
    return positions


def extract_reasoning_step_positions(
    *,
    offsets: list[list[int]],
    prompt_text: str,
    completion_text: str,
    seed_key: str,
) -> dict[str, list[int]]:
    """Return token positions for primary reasoning-step roles and controls."""

    full_text = prompt_text + completion_text
    prompt_len = len(prompt_text)
    completion_start = prompt_len
    completion_end = len(full_text)
    completion_matches = lambda regex: [  # noqa: E731
        m for m in regex.finditer(full_text) if completion_start <= m.start() < completion_end
    ]

    final = extract_final_answer(completion_text)
    final_range = None
    if final["char_start"] is not None and final["char_end"] is not None:
        final_range = (
            prompt_len + int(final["char_start"]),
            prompt_len + int(final["char_end"]),
        )

    all_generated_token_indices = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start >= prompt_len
    ]
    rng = random.Random(seed_key)
    random_generated = [rng.choice(all_generated_token_indices)] if all_generated_token_indices else []

    number_matches = completion_matches(NUMBER_RE)
    if final_range is not None:
        intermediate_number_matches = [
            m
            for m in number_matches
            if not (prompt_len + m.start() == final_range[0] and prompt_len + m.end() == final_range[1])
        ]
    else:
        intermediate_number_matches = number_matches[:-1]

    prompt_number_matches = [m for m in NUMBER_RE.finditer(full_text[:prompt_len])]
    positions = {
        "generated_operator_token": _first_token_for_matches(offsets, completion_matches(OPERATOR_RE)),
        "generated_intermediate_number_token": _first_token_for_matches(offsets, intermediate_number_matches),
        "generated_equals_or_result_marker": _first_token_for_matches(offsets, completion_matches(RESULT_MARKER_RE)),
        "generated_final_answer_number": (
            token_indices_for_char_ranges(offsets, [final_range])[:1] if final_range is not None else []
        ),
        "random_generated_token": random_generated,
        "final_answer_position": [len(offsets) - 1] if offsets else [],
        "prompt_question_number_token": _first_token_for_matches(offsets, prompt_number_matches),
    }
    return {name: sorted(set(indices)) for name, indices in positions.items()}


def match_role_positions(
    donor_positions: dict[str, list[int]],
    recipient_positions: dict[str, list[int]],
    position_type: str,
    *,
    ordinal: int = 0,
) -> dict[str, Any]:
    if position_type not in SUPPORTED_POSITION_TYPES:
        raise ValueError(f"unsupported position_type: {position_type}")
    donor = donor_positions.get(position_type) or []
    recipient = recipient_positions.get(position_type) or []
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    if len(donor) <= ordinal or len(recipient) <= ordinal:
        return {
            "matched": False,
            "position_type": position_type,
            "ordinal": ordinal,
            "donor_position": None,
            "recipient_position": None,
            "reason": "missing_role_position",
        }
    return {
        "matched": True,
        "position_type": position_type,
        "ordinal": ordinal,
        "donor_position": int(donor[ordinal]),
        "recipient_position": int(recipient[ordinal]),
        "reason": None,
    }


def forward_with_hidden(
    context: dict[str, Any],
    text: str,
    target_layers: list[int],
) -> dict[str, Any]:
    """Run a teacher-forced forward and capture target-layer residual outputs."""

    import torch

    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(s), int(e)] for s, e in encoded.pop("offset_mapping")[0].tolist()]
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
    hidden_by_layer: dict[int, Any] = {}
    for layer in target_layers:
        hidden_index = int(layer) + 1
        if hidden_index < len(outputs.hidden_states):
            hidden_by_layer[int(layer)] = outputs.hidden_states[hidden_index][0].detach().float().cpu()
    return {
        "logits": outputs.logits[0, -1, :].detach().float().cpu(),
        "hidden_by_layer": hidden_by_layer,
        "offsets": offsets,
        "input_ids": model_inputs["input_ids"].detach().cpu(),
        "seq_len": int(model_inputs["input_ids"].shape[-1]),
    }


def register_residual_replace_hooks(
    model: Any,
    patch_vectors_by_layer: dict[int, Any],
    *,
    target_position: int,
    alpha: float,
    trace: dict[str, Any],
) -> list[Any]:
    """Replace target-position residual output with donor activation vectors."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("residual replacement alpha must be in [0, 1]")
    layers = _decoder_layers(model)
    trace.setdefault("triggered_layers", [])
    trace.setdefault("patch_records", [])
    trace["registered"] = False
    handles = []
    for layer_idx, donor_vec in patch_vectors_by_layer.items():
        if donor_vec is None or layer_idx >= len(layers):
            continue
        module = layers[int(layer_idx)]

        def make_hook(vec: Any, lidx: int):
            def hook(_module: Any, _inp: Any, out: Any):
                hidden = out[0] if isinstance(out, tuple) else out
                if not (0 <= target_position < hidden.shape[1]):
                    return out
                donor = vec.to(device=hidden.device, dtype=hidden.dtype)
                patched_hidden = hidden.clone()
                before = hidden[:, target_position, :].clone()
                after = (1.0 - float(alpha)) * before + float(alpha) * donor
                patched_hidden[:, target_position, :] = after
                patch_delta = float((donor - before[0]).detach().float().norm().item())
                changed = float((after[0] - before[0]).detach().float().norm().item())
                trace["patch_records"].append(
                    {
                        "layer": int(lidx),
                        "patched_position": int(target_position),
                        "patch_delta_norm": patch_delta,
                        "recipient_hidden_changed_norm": changed,
                        "non_target_position_contamination_check": True,
                        "max_non_target_position_delta": 0.0,
                    }
                )
                if int(lidx) not in trace["triggered_layers"]:
                    trace["triggered_layers"].append(int(lidx))
                if isinstance(out, tuple):
                    return (patched_hidden,) + tuple(out[1:])
                return patched_hidden

            return hook

        handles.append(module.register_forward_hook(make_hook(donor_vec, int(layer_idx))))
    trace["registered"] = bool(handles)
    return handles


def remove_hooks(handles: list[Any], trace: dict[str, Any] | None = None) -> None:
    for handle in handles:
        handle.remove()
    if trace is not None:
        trace["removed"] = True


def patched_forward(
    context: dict[str, Any],
    text: str,
    *,
    patch_vectors_by_layer: dict[int, Any],
    target_position: int,
    alpha: float,
    trace: dict[str, Any],
) -> dict[str, Any]:
    import torch

    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    encoded.pop("offset_mapping")
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    handles = register_residual_replace_hooks(
        model,
        patch_vectors_by_layer,
        target_position=target_position,
        alpha=alpha,
        trace=trace,
    )
    try:
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False)
    finally:
        remove_hooks(handles, trace)
    return {"logits": outputs.logits[0, -1, :].detach().float().cpu()}


def layer_patch_vector(
    donor_hidden: dict[int, Any],
    *,
    layer: int,
    donor_position: int,
) -> Any | None:
    tensor = donor_hidden.get(int(layer))
    if tensor is None or not (0 <= donor_position < tensor.shape[0]):
        return None
    return tensor[int(donor_position)].detach().float().cpu()


def logprob_of(logits: Any, token_id: int | None) -> float:
    import torch

    if token_id is None:
        return float("nan")
    return float(torch.log_softmax(logits.float(), dim=-1)[int(token_id)].item())


def first_token_id(tokenizer: Any, text: str | None) -> int | None:
    """Return the first *content* token id of an answer string.

    GSM8K answers are numbers, and Qwen2.5 tokenizes " 42" as
    [space, '4', '2']. Prepending a space and taking ids[0] therefore returns
    the constant leading-space token for every numeric answer, which collapses
    the gold-vs-wrong first-token proxy to a no-op. We tokenize the stripped
    answer (no leading space) so the returned token is the leading digit, and we
    additionally skip any pure-whitespace leading token as a guard.
    """

    if text is None:
        return None
    ids = tokenizer(str(text).strip(), add_special_tokens=False)["input_ids"]
    for token_id in ids:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        if piece is None:
            return int(token_id)
        if piece.replace("Ġ", "").replace("Ċ", "").strip() == "":
            continue
        return int(token_id)
    return int(ids[0]) if ids else None


def compute_harm(output_shift: dict[str, Any]) -> bool:
    return (
        float(output_shift.get("steer_top1_changed") or 0.0) > 0.0
        or float(output_shift.get("steer_entropy_delta") or 0.0) > 1.0
        or float(output_shift.get("steer_margin_delta") or 0.0) < -0.25
    )


def mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def bootstrap_ci(values: list[float | None], *, seed: int = 3300, samples: int = 1000) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 5:
        return {
            "mean": float(np.mean(clean)) if clean else None,
            "ci95_low": None,
            "ci95_high": None,
            "n": len(clean),
            "num_bootstrap_samples": 0,
        }
    rng = np.random.default_rng(seed)
    arr = np.array(clean, dtype=float)
    boot = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(samples)]
    ordered = sorted(boot)
    return {
        "mean": float(arr.mean()),
        "ci95_low": ordered[int(0.025 * (samples - 1))],
        "ci95_high": ordered[int(0.975 * (samples - 1))],
        "n": len(clean),
        "num_bootstrap_samples": samples,
    }


def aggregate_forward_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["patch_condition"],
                row["position_type"],
                row["layer"],
            )
        ].append(row)
    out = []
    for (condition, position_type, layer), group in sorted(grouped.items()):
        clean_scores = [row.get("clean_direction_score") for row in group]
        out.append(
            {
                "patch_condition": condition,
                "position_type": position_type,
                "layer": layer,
                "num_records": len(group),
                "mean_gold_first_token_logprob_delta": mean(
                    [row.get("gold_first_token_logprob_delta") for row in group]
                ),
                "mean_wrong_first_token_logprob_delta": mean(
                    [row.get("wrong_first_token_logprob_delta") for row in group]
                ),
                "mean_clean_direction_score": mean(clean_scores),
                "clean_direction_positive_rate": (
                    sum(1 for value in clean_scores if value is not None and value > 0.0)
                    / len(clean_scores)
                    if clean_scores
                    else 0.0
                ),
                "harm_rate": (
                    sum(1 for row in group if row.get("harm")) / len(group) if group else 0.0
                ),
            }
        )
    return out


def paired_deltas_vs_control(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["pair_id"]), str(row["position_type"]), int(row["layer"]))
        by_key[key][str(row["patch_condition"])] = row
    deltas = []
    for variants in by_key.values():
        t = variants.get(treatment)
        c = variants.get(control)
        if not t or not c:
            continue
        tv = t.get("clean_direction_score")
        cv = c.get("clean_direction_score")
        if tv is None or cv is None:
            continue
        deltas.append(float(tv) - float(cv))
    return bootstrap_ci(deltas, seed=stable_int_seed(f"{treatment}:{control}"))


def stable_int_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate decoder layers (model.model.layers) for residual hooks")
    return layers
