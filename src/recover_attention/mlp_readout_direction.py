"""Sprint 3C-2: MLP readout direction analysis and donor-free nudge.

Sprint 3C-1 localized the selective, low-harm answer-readout write to the MLP
output at the final-answer readout position. This module (1) extracts the
correct-minus-wrong MLP-output delta direction at that position, (2) analyses its
geometry and alignment to the gold-vs-wrong unembedding direction, and (3)
applies donor-free directional nudges to the MLP output and scores them with the
Sprint 3C-0-Fix corrected answer-sequence proxy.

Boundary: no training, no LoRA, no deployable steering claim, no full Sprint 3C /
2000 rerun, no accuracy or hallucination claim. Teacher-forced (plus a minimal
first-step) single-forward proxy only. Correct-donor deltas and gold-unembedding
directions use gold answers for eval-only analysis, not as deployable methods.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from recover_attention import answer_proxy_metrics as apm
from recover_attention import module_causal_tracing as mct
from recover_attention import multi_span_reasoning_scoring as msrs

BACKEND = "mlp_readout_direction_analysis_v0"


# ------------------------------------------------------------------ extraction


def extract_mlp_readout(context: dict[str, Any], trace_text: str, layers: list[int]) -> dict[str, Any] | None:
    """MLP-output vectors at the answer-readout position for a trace, per layer."""

    capture = mct.capture_module_outputs(context, trace_text, layers, ["mlp_output"])
    readout = mct.build_answer_readout(context, trace_text, capture)
    if readout is None:
        return None
    vectors: dict[int, Any] = {}
    for layer in layers:
        vec = mct.module_vector(capture["captured"], layer=layer, module_type="mlp_output", position=readout["readout_position"])
        if vec is not None:
            vectors[int(layer)] = vec
    if not vectors:
        return None
    return {"vectors": vectors, "readout": readout}


def compute_correct_wrong_delta(correct_vecs: dict[int, Any], wrong_vecs: dict[int, Any], layers: list[int]) -> dict[int, Any]:
    import torch

    deltas: dict[int, Any] = {}
    for layer in layers:
        cv = correct_vecs.get(int(layer))
        wv = wrong_vecs.get(int(layer))
        if cv is not None and wv is not None:
            deltas[int(layer)] = (cv.float() - wv.float())
    return deltas


# ------------------------------------------------------------------ directions


def rotate_derangement(n: int) -> list[int]:
    """Shift-by-one permutation: no index maps to itself for n >= 2.

    Used to build the shuffled-delta control (correct_i paired with a different
    pair's wrong vector), so the control never reuses the same pair's label.
    """

    if n <= 1:
        return list(range(n))
    return [(i + 1) % n for i in range(n)]


def normalize_direction(vec: Any) -> Any:
    """Unit vector; a zero (or non-finite) vector normalizes to zeros (no-op)."""

    import torch

    v = vec.float()
    norm = float(v.norm().item())
    if not math.isfinite(norm) or norm < 1e-12:
        return torch.zeros_like(v)
    return v / norm


def mean_direction(vectors: list[Any]) -> Any:
    import torch

    if not vectors:
        raise ValueError("mean_direction requires at least one vector")
    stacked = torch.stack([v.float() for v in vectors], dim=0)
    return stacked.mean(dim=0)


def pca_direction(vectors: list[Any], *, reference: Any | None = None) -> Any:
    """First principal component of the delta set, optionally sign-aligned."""

    import torch

    stacked = torch.stack([v.float() for v in vectors], dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    matrix = centered.cpu().numpy()
    # economy SVD: right singular vectors are the principal directions.
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    pc1 = torch.tensor(vh[0], dtype=torch.float32)
    if reference is not None:
        pc1 = align_sign(pc1, reference)
    return pc1


def align_sign(direction: Any, reference: Any) -> Any:
    dot = float((direction.float() * reference.float()).sum().item())
    return direction if dot >= 0 else -direction


def explained_variance_ratio(vectors: list[Any], *, k: int = 3) -> list[float]:
    stacked = np.stack([v.float().cpu().numpy() for v in vectors], axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    variances = (s ** 2)
    total = float(variances.sum())
    if total < 1e-12:
        return [0.0] * min(k, len(variances))
    return [float(v / total) for v in variances[:k]]


def unembedding_answer_direction(model: Any, tokenizer: Any, gold_answer: str, wrong_answer: str) -> Any | None:
    """W_U[gold_first_token] - W_U[wrong_first_token] (eval-only reference)."""

    import torch

    embed = model.get_output_embeddings()
    if embed is None:
        return None
    weight = embed.weight  # [vocab, hidden]
    gold_ids = apm.answer_token_ids(tokenizer, gold_answer)
    wrong_ids = apm.answer_token_ids(tokenizer, wrong_answer)
    if not gold_ids or not wrong_ids:
        return None
    gold_row = weight[gold_ids[0]].detach().float().cpu()
    wrong_row = weight[wrong_ids[0]].detach().float().cpu()
    return gold_row - wrong_row


def pairwise_cosine_mean(vectors: list[Any]) -> float | None:
    units = [normalize_direction(v) for v in vectors]
    units = [u for u in units if float(u.norm().item()) > 0]
    if len(units) < 2:
        return None
    matrix = np.stack([u.cpu().numpy() for u in units], axis=0)
    sims = matrix @ matrix.T
    n = len(units)
    off_diag = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(off_diag)


def cosine(a: Any, b: Any) -> float:
    ua = normalize_direction(a)
    ub = normalize_direction(b)
    if float(ua.norm().item()) == 0 or float(ub.norm().item()) == 0:
        return float("nan")
    return float((ua * ub).sum().item())


# ------------------------------------------------------------------ nudge hooks


def register_mlp_direction_nudge(
    model: Any,
    *,
    layer: int,
    unit_direction: Any,
    alpha: float,
    target_position: int,
    trace: dict[str, Any],
) -> list[Any]:
    """Additive nudge on MLP output: out[pos] += alpha * ||out[pos]|| * unit_dir."""

    if alpha < 0.0:
        raise ValueError("alpha must be >= 0")
    decoder = mct._decoder_layers(model)
    module = decoder[int(layer)].mlp
    trace.setdefault("triggered", [])
    trace.setdefault("patch_records", [])
    trace["registered"] = False

    def hook(_module: Any, _inp: Any, out: Any):
        hidden = out[0] if isinstance(out, tuple) else out
        if not (0 <= target_position < hidden.shape[1]):
            return out
        direction = unit_direction.to(device=hidden.device, dtype=hidden.dtype)
        cur = hidden[:, target_position, :]
        norm = cur.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
        patched = hidden.clone()
        add = float(alpha) * norm * direction
        patched[:, target_position, :] = cur + add
        trace["patch_records"].append({"layer": int(layer), "nudge_norm": float(add[0].float().norm().item())})
        trace["triggered"].append((int(layer), "mlp_output"))
        if isinstance(out, tuple):
            return (patched,) + tuple(out[1:])
        return patched

    handle = module.register_forward_hook(hook)
    trace["registered"] = True
    return [handle]


def _forward_with_nudge(context, input_ids, nudge):
    import torch

    model = context["model"]
    device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = input_ids.to(device)
    handles: list[Any] = []
    trace: dict[str, Any] = {"registered": False, "removed": False, "triggered": [], "patch_records": []}
    if nudge is not None and nudge.get("unit_direction") is not None:
        handles = register_mlp_direction_nudge(
            model, layer=int(nudge["layer"]), unit_direction=nudge["unit_direction"],
            alpha=float(nudge["alpha"]), target_position=int(nudge["target_position"]), trace=trace,
        )
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        trace["removed"] = True
    return outputs, trace


def sequence_logprob_with_direction_nudge(
    context: dict[str, Any], *, prefix_ids: list[int], answer_ids: list[int], nudge: dict[str, Any] | None = None, return_slot_logits: bool = False
) -> dict[str, Any]:
    """Teacher-forced answer-sequence logprob under a donor-free MLP nudge."""

    import torch

    if not answer_ids:
        return {"logprob": float("nan"), "per_token": float("nan"), "num_answer_tokens": 0, "trace": {}, "answer_slot_logits": None}
    input_ids = torch.tensor([list(prefix_ids) + list(answer_ids)], dtype=torch.long)
    outputs, trace = _forward_with_nudge(context, input_ids, nudge)
    logits = outputs.logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    n_prefix = len(prefix_ids)
    total = 0.0
    for i, token_id in enumerate(answer_ids):
        total += float(logprobs[n_prefix + i - 1, int(token_id)].item())
    slot = logits[n_prefix - 1].detach().cpu() if return_slot_logits else None
    n = len(answer_ids)
    return {"logprob": total, "per_token": total / n if n else float("nan"), "num_answer_tokens": n, "trace": trace, "answer_slot_logits": slot}


def first_step_logits_with_nudge(context: dict[str, Any], *, prefix_ids: list[int], nudge: dict[str, Any] | None = None) -> Any:
    """Next-token logits at the readout slot (prefix only) for a first-step check."""

    import torch

    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long)
    outputs, _ = _forward_with_nudge(context, input_ids, nudge)
    return outputs.logits[0, -1, :].detach().float().cpu()


# ------------------------------------------------------------------ aggregation


def bootstrap_ci(values: list[float | None], *, seed: int = 3302) -> dict[str, Any]:
    out = apm.bootstrap_ci(values, seed=seed)
    out["stable_positive"] = bool(out.get("ci95_low") is not None and out["ci95_low"] > 0)
    return out


def mean(values: list[float | None]) -> float | None:
    return apm.mean(values)
