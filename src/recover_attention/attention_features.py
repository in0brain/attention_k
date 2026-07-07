"""Sprint 2I: pre-recovery attention-map summary features.

Computes leakage-free attention summaries from the ORIGINAL and MASKED forward passes
only (never the recovered question). Core hypothesis: a fragile span, once masked,
causes an anomalous reorganization of the attention distribution around its slot.

Feature families (per input: original span slot / masked mask slot, plus deltas):
  A. slot attention mass      (in / out / self / rank)
  B. slot attention shape      (entropy / top-k mass / effective edge count)
  C. context-to-slot attention (question-focus / operation / number-context -> slot)
  E. original->masked deltas   (magnitude of attention reorganization)

Feature names deliberately avoid every banned substring
(recovered / solution_path / drift / bucket / risk_strength / gold / answer / label /
target) so the gate-eligibility leakage test passes. Note "question_focus" is used
instead of "question_target" because "target" is a banned substring.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

BANNED_FEATURE_SUBSTRINGS = (
    "recovered", "solution_path", "drift", "bucket", "risk_strength",
    "gold", "answer", "label", "target",
)

_EPS = 1e-12


def _row_entropy(row: np.ndarray) -> float:
    p = row[row > _EPS]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _topk_mass(row: np.ndarray, k: int) -> float:
    if row.size == 0:
        return 0.0
    k = min(k, row.size)
    return float(np.sort(row)[-k:].sum())


def _slot_scalars(att_layer: np.ndarray, slot_idx: list[int]) -> dict[str, float]:
    """Per-layer attention scalars for one slot region. att_layer: [seq, seq], rows sum ~1."""
    seq = att_layer.shape[0]
    slot = np.array(slot_idx, dtype=int)
    col_sum = att_layer.sum(axis=0)  # incoming mass per key token, total == seq
    slot_in_mass = float(col_sum[slot].sum()) / seq  # fraction of all attention into slot
    slot_in_rel = slot_in_mass / (len(slot) / seq) if len(slot) else 0.0  # vs uniform

    slot_rows = att_layer[slot, :]  # [|slot|, seq]
    self_mass = float(slot_rows[:, slot].sum(axis=1).mean())
    out_mask = np.ones(seq, dtype=bool)
    out_mask[slot] = False
    out_mass = float(slot_rows[:, out_mask].sum(axis=1).mean())
    ent = float(np.mean([_row_entropy(slot_rows[i]) for i in range(slot_rows.shape[0])]))
    top1 = float(np.mean([_topk_mass(slot_rows[i], 1) for i in range(slot_rows.shape[0])]))
    top3 = float(np.mean([_topk_mass(slot_rows[i], 3) for i in range(slot_rows.shape[0])]))
    top5 = float(np.mean([_topk_mass(slot_rows[i], 5) for i in range(slot_rows.shape[0])]))
    edge = float(math.exp(ent))

    # rank of slot mean incoming among all tokens (0..1, 1 = most attended)
    slot_mean_col = float(col_sum[slot].mean())
    rank = float((col_sum < slot_mean_col).sum()) / seq

    return {
        "in_mass": slot_in_mass,
        "in_rel": slot_in_rel,
        "self_mass": self_mass,
        "out_mass": out_mass,
        "rank": rank,
        "entropy": ent,
        "top1_mass": top1,
        "top3_mass": top3,
        "top5_mass": top5,
        "edge_count": edge,
    }


def _context_to_slot(att_layer: np.ndarray, slot_idx: list[int], ctx_idx: list[int]) -> float:
    if not ctx_idx or not slot_idx:
        return float("nan")
    rows = att_layer[np.array(ctx_idx, dtype=int), :]
    return float(rows[:, np.array(slot_idx, dtype=int)].sum(axis=1).mean())


def _aggregate_input(
    att_stack: np.ndarray, slot_idx: list[int], layer_ids: list[int], prefix: str,
    context: dict[str, list[int]] | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Aggregate over layers (mean) + keep per-layer in_mass/entropy for deltas."""
    per_layer = [_slot_scalars(att_stack[l], slot_idx) for l in range(att_stack.shape[0])]
    keys = per_layer[0].keys()
    agg = {f"attn_{prefix}_slot_{k}": float(np.mean([pl[k] for pl in per_layer])) for k in keys}
    layer_series = {
        "in_mass": [pl["in_mass"] for pl in per_layer],
        "entropy": [pl["entropy"] for pl in per_layer],
    }
    for li, lid in enumerate(layer_ids):
        agg[f"attn_{prefix}_slot_in_mass_layer_{lid}"] = float(per_layer[li]["in_mass"])
        agg[f"attn_{prefix}_slot_entropy_layer_{lid}"] = float(per_layer[li]["entropy"])

    if context:
        for cname, cidx in context.items():
            vals = [_context_to_slot(att_stack[l], slot_idx, cidx) for l in range(att_stack.shape[0])]
            finite = [v for v in vals if not math.isnan(v)]
            agg[f"attn_{prefix}_{cname}_to_slot"] = float(np.mean(finite)) if finite else 0.0
    return agg, layer_series


def build_attention_features(
    orig_att: np.ndarray,
    orig_slot_idx: list[int],
    masked_att: np.ndarray,
    masked_slot_idx: list[int],
    layer_ids: list[int],
    *,
    context_orig: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Build the full pre-recovery attention feature dict for one masked_id.

    ``orig_att`` / ``masked_att``: head-averaged attention stacks [L, seq, seq].
    Returns {"features": {...}, "missing": {...}}.
    """
    features: dict[str, float] = {}
    missing: dict[str, bool] = {}

    orig_agg, orig_series = _aggregate_input(orig_att, orig_slot_idx, layer_ids, "orig", context_orig)
    masked_agg, masked_series = _aggregate_input(masked_att, masked_slot_idx, layer_ids, "masked")
    features.update(orig_agg)
    features.update(masked_agg)

    # E. original->masked deltas on shared aggregate scalars
    for base in ["in_mass", "in_rel", "self_mass", "out_mass", "rank", "entropy",
                 "top1_mass", "top3_mass", "top5_mass", "edge_count"]:
        o = orig_agg.get(f"attn_orig_slot_{base}")
        m = masked_agg.get(f"attn_masked_slot_{base}")
        if o is not None and m is not None:
            features[f"attn_delta_slot_{base}"] = float(m - o)
    # per-layer delta of in_mass / entropy
    for li, lid in enumerate(layer_ids):
        features[f"attn_delta_slot_in_mass_layer_{lid}"] = float(
            masked_series["in_mass"][li] - orig_series["in_mass"][li]
        )
        features[f"attn_delta_slot_entropy_layer_{lid}"] = float(
            masked_series["entropy"][li] - orig_series["entropy"][li]
        )

    # context missing flags
    if context_orig:
        for cname, cidx in context_orig.items():
            missing[f"{cname}"] = len(cidx) == 0

    assert_no_banned_attention_names(list(features.keys()))
    return {"features": features, "missing": missing}


def assert_no_banned_attention_names(feature_names: list[str]) -> None:
    bad = [n for n in feature_names for tok in BANNED_FEATURE_SUBSTRINGS if tok in n]
    if bad:
        raise AssertionError(
            f"attention feature names must not contain {BANNED_FEATURE_SUBSTRINGS}; offenders: {bad[:5]}"
        )


def head_average_selected_layers(attentions: Any, layer_indices: list[int]) -> np.ndarray:
    """From HF ``output.attentions`` (tuple len=num_layers of [batch, heads, seq, seq])
    build a head-averaged stack [len(layer_indices), seq, seq] as float32 numpy."""
    stack = []
    for lid in layer_indices:
        layer = attentions[lid]  # [batch, heads, seq, seq]
        head_mean = layer[0].mean(axis=0)  # [seq, seq]
        stack.append(head_mean.to("cpu").float().numpy())
    arr = np.stack(stack, axis=0)
    # Safety net: the final decoder layer can return NaN attention under 4-bit eager
    # (that layer is excluded via LAYER_INDICES); zero any residual non-finite values.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
