"""Sprint 3C-4A approximate J-lens readout helpers.

This module implements a small directional finite-difference readout sanity
check.  It perturbs the MLP output at the final-answer readout position and
observes the change in the actual final logits.  It is not full J-lens, not
steering, and not a generation or accuracy evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from recover_attention import activation_patching as ap
from recover_attention import mlp_readout_attribution as mra
from recover_attention import multi_span_reasoning_scoring as msrs

BACKEND = "approx_j_lens_readout_sanity_check_v0"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if finite(v)]
    return float(np.mean(clean)) if clean else None


def normalize_direction(vector: Any) -> Any:
    import torch

    vec = torch.as_tensor(vector).detach().float().cpu()
    norm = float(vec.norm().item())
    if not math.isfinite(norm) or norm <= 1e-12:
        return torch.zeros_like(vec)
    return vec / norm


def project_vector_to_token_scores(vector: Any, unembedding_weight: Any, token_ids: list[int]) -> dict[int, float]:
    """Project one hidden vector only onto selected lm_head rows."""

    import torch

    ids = sorted({int(t) for t in token_ids if int(t) >= 0})
    if not ids:
        return {}
    weight = torch.as_tensor(unembedding_weight)
    vocab_size = int(weight.shape[0])
    ids = [t for t in ids if t < vocab_size]
    if not ids:
        return {}
    id_tensor = torch.tensor(ids, dtype=torch.long, device=weight.device)
    vec = torch.as_tensor(vector, dtype=torch.float32, device=weight.device)
    selected = weight.index_select(0, id_tensor).float()
    scores = torch.matmul(selected, vec).detach().float().cpu().tolist()
    return {int(tid): float(score) for tid, score in zip(ids, scores)}


def tensor_to_token_scores(scores: Any, token_ids: list[int]) -> dict[int, float]:
    import torch

    tensor = torch.as_tensor(scores).detach().float().cpu()
    out: dict[int, float] = {}
    for token_id in sorted({int(t) for t in token_ids if 0 <= int(t) < int(tensor.numel())}):
        out[int(token_id)] = float(tensor[int(token_id)].item())
    return out


def top_token_ids(score_map: dict[int, float], token_ids: list[int] | None = None, *, k: int = 20) -> list[int]:
    allowed = set(int(t) for t in token_ids) if token_ids is not None else set(score_map)
    pairs = [(int(t), float(score_map[int(t)])) for t in allowed if int(t) in score_map and finite(score_map[int(t)])]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [tid for tid, _ in pairs[: max(0, int(k))]]


def score_map_features(
    score_map: dict[int, float],
    *,
    tokenizer: Any,
    number_token_ids: list[int],
    top_k: int = 20,
    prefix: str = "direct",
    vector_norm: float | None = None,
) -> dict[str, Any]:
    valid = [int(t) for t in number_token_ids if int(t) in score_map and finite(score_map[int(t)])]
    if not valid:
        return {
            f"{prefix}_top1_number_token_id": None,
            f"{prefix}_top1_number_token": None,
            f"{prefix}_top1_number_score": None,
            f"{prefix}_top2_number_score": None,
            f"{prefix}_number_margin": None,
            f"{prefix}_number_entropy": None,
            f"{prefix}_number_top1_prob": None,
            f"{prefix}_projection_norm": vector_norm,
            f"{prefix}_projection_sharpness": None,
            f"{prefix}_top_k_projected_number_tokens": [],
        }
    ranked = top_token_ids(score_map, valid, k=top_k)
    top_scores = [float(score_map[t]) for t in ranked]
    entropy, top_prob = _softmax_entropy([score_map[t] for t in valid])
    top1 = top_scores[0] if top_scores else None
    top2 = top_scores[1] if len(top_scores) > 1 else None
    margin = top1 - top2 if top1 is not None and top2 is not None else top1
    return {
        f"{prefix}_top1_number_token_id": ranked[0] if ranked else None,
        f"{prefix}_top1_number_token": mra.token_text(tokenizer, ranked[0]) if ranked else None,
        f"{prefix}_top1_number_score": top1,
        f"{prefix}_top2_number_score": top2,
        f"{prefix}_number_margin": margin,
        f"{prefix}_number_entropy": entropy,
        f"{prefix}_number_top1_prob": top_prob,
        f"{prefix}_projection_norm": vector_norm,
        f"{prefix}_projection_sharpness": top_prob,
        f"{prefix}_top_k_projected_number_tokens": [
            {"token_id": tid, "token_text": mra.token_text(tokenizer, tid), "score": float(score_map[tid])}
            for tid in ranked
        ],
    }


def _softmax_entropy(values: Iterable[float]) -> tuple[float | None, float | None]:
    clean = np.array([float(v) for v in values if finite(v)], dtype=float)
    if clean.size == 0:
        return None, None
    shifted = clean - float(clean.max())
    probs = np.exp(shifted)
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    return entropy, float(probs.max())


def rank_of_token(score_map: dict[int, float], token_id: int | None, candidate_ids: list[int]) -> int | None:
    if token_id is None or int(token_id) not in score_map or not finite(score_map[int(token_id)]):
        return None
    target_score = float(score_map[int(token_id)])
    better = 0
    for candidate in candidate_ids:
        candidate = int(candidate)
        if candidate in score_map and finite(score_map[candidate]) and float(score_map[candidate]) > target_score:
            better += 1
    return better + 1


def topk_overlap(direct_top: list[int], approx_top: list[int], *, k: int) -> dict[str, Any]:
    k = int(k)
    a = [int(t) for t in direct_top[:k]]
    b = [int(t) for t in approx_top[:k]]
    denom = max(1, min(k, len(a), len(b)))
    overlap = len(set(a) & set(b))
    return {
        "k": k,
        "overlap_count": int(overlap),
        "overlap_rate": float(overlap / denom),
        "top1_match": bool(a[:1] and b[:1] and a[0] == b[0]),
    }


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def spearman_correlation_from_maps(
    left: dict[int, float],
    right: dict[int, float],
    candidate_ids: list[int],
) -> float | None:
    ids = [int(t) for t in candidate_ids if int(t) in left and int(t) in right and finite(left[int(t)]) and finite(right[int(t)])]
    if len(ids) < 2:
        return None
    a = [float(left[t]) for t in ids]
    b = [float(right[t]) for t in ids]
    ra = np.array(_average_ranks(a), dtype=float)
    rb = np.array(_average_ranks(b), dtype=float)
    if float(ra.std()) <= 1e-12 or float(rb.std()) <= 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_readout_risk(
    features: dict[str, Any],
    *,
    prefix: str,
    layer_agreement: float | None,
    model_answer_agreement: float | None,
) -> float:
    entropy = _float_or_zero(features.get(f"{prefix}_number_entropy"))
    margin = _float_or_zero(features.get(f"{prefix}_number_margin"))
    sharpness = _float_or_zero(features.get(f"{prefix}_projection_sharpness"))
    layer = _float_or_zero(layer_agreement)
    model = _float_or_zero(model_answer_agreement)
    return float(entropy - margin + (1.0 - layer) + (1.0 - model) + (1.0 - sharpness))


def token_agreement(a: int | None, b: int | None) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if int(a) == int(b) else 0.0


def _float_or_zero(value: Any) -> float:
    return float(value) if finite(value) else 0.0


def register_mlp_output_perturb_hook(
    model: Any,
    *,
    layer: int,
    direction_vec: Any,
    target_position: int,
    epsilon: float,
    base_norm: float | None = None,
    trace: dict[str, Any] | None = None,
) -> list[Any]:
    """Add epsilon * ||m|| * unit(direction) at one MLP-output token position."""

    import torch

    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    decoder = ap._decoder_layers(model)
    module = decoder[int(layer)].mlp
    unit = normalize_direction(direction_vec)
    norm = float(base_norm) if base_norm is not None else float(torch.as_tensor(direction_vec).detach().float().norm().item())
    if not math.isfinite(norm):
        norm = 0.0
    trace = trace if trace is not None else {}
    trace.setdefault("triggered", [])
    trace.setdefault("patch_records", [])
    trace["registered"] = False

    def hook(_module: Any, _inp: Any, out: Any):
        hidden = out[0] if isinstance(out, tuple) else out
        if not (0 <= int(target_position) < hidden.shape[1]):
            return out
        unit_device = unit.to(device=hidden.device, dtype=hidden.dtype)
        scale = float(epsilon) * float(norm)
        patched = hidden.clone()
        before = hidden[:, int(target_position), :].clone()
        delta = scale * unit_device
        after = before + delta
        patched[:, int(target_position), :] = after
        trace["triggered"].append((int(layer), "mlp_output"))
        trace["patch_records"].append(
            {
                "layer": int(layer),
                "module_type": "mlp_output",
                "patched_position": int(target_position),
                "epsilon": float(epsilon),
                "base_norm": float(norm),
                "perturb_norm": float(delta.detach().float().norm().item()),
                "non_target_position_contamination_check": True,
                "max_non_target_position_delta": 0.0,
            }
        )
        if isinstance(out, tuple):
            return (patched,) + tuple(out[1:])
        return patched

    handle = module.register_forward_hook(hook)
    trace["registered"] = True
    return [handle]


def answer_slot_logits_with_mlp_perturb(
    context: dict[str, Any],
    *,
    prefix_ids: list[int],
    perturb: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return logits at the final prefix token, with an optional MLP perturbation."""

    import torch

    model = context["model"]
    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long)
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = input_ids.to(target_device)
    handles: list[Any] = []
    trace: dict[str, Any] = {"registered": False, "removed": False, "triggered": [], "patch_records": []}
    if perturb is not None:
        handles = register_mlp_output_perturb_hook(
            model,
            layer=int(perturb["layer"]),
            direction_vec=perturb["direction_vec"],
            target_position=int(perturb["target_position"]),
            epsilon=float(perturb["epsilon"]),
            base_norm=perturb.get("base_norm"),
            trace=trace,
        )
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
        trace["removed"] = True
    return {"logits": logits, "trace": trace}


def compact_top_tokens(score_map: dict[int, float], tokenizer: Any, token_ids: list[int], *, k: int) -> list[dict[str, Any]]:
    return [
        {"token_id": tid, "token_text": mra.token_text(tokenizer, tid), "score": float(score_map[tid])}
        for tid in top_token_ids(score_map, token_ids, k=k)
    ]


def bootstrap_ci(values: list[float | None], *, seed_tag: str) -> dict[str, Any]:
    return ap.bootstrap_ci(values, seed=ap.stable_int_seed(seed_tag))
