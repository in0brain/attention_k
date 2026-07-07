"""Sprint 2K: leakage-safe self-output-effect features.

Hidden/attention features are *perturbation* signals (how much does the internal state
move when a span is masked). Keyness is closer to an *output-effect*: how much does the
model's own next-token distribution move when the span is masked. This module computes
that shift from the last-token logits of the original vs masked forward.

Leakage rule: uses ONLY the model's own output distribution. It never reads the gold
answer, solution path, drift, or bucket. Feature names avoid every banned substring
(the word "answer" is banned, so features are named ``self_output_*``, not
``answer_*``).
"""

from __future__ import annotations

import math
from typing import Any

BANNED_FEATURE_SUBSTRINGS = (
    "recovered", "solution_path", "drift", "bucket", "risk_strength",
    "gold", "answer", "label", "target", "trajectory", "cot", "nla", "oracle",
)

_EPS = 1e-12


def _softmax(logits: Any) -> Any:
    import torch

    return torch.softmax(logits.float(), dim=-1)


def _entropy(p: Any) -> float:
    import torch

    pl = p.clamp_min(_EPS)
    return float(-(pl * pl.log()).sum().item())


def _kl(p: Any, q: Any) -> float:
    import torch

    pl = p.clamp_min(_EPS)
    ql = q.clamp_min(_EPS)
    return float((pl * (pl.log() - ql.log())).sum().item())


def compute_output_effect(
    original_logits: Any,
    masked_logits: Any,
    *,
    topk: int = 10,
) -> dict[str, float]:
    """Self-output-distribution shift between original and masked last-token logits.

    Both inputs are 1-D logit vectors over the vocabulary (torch tensors). Returns a
    flat dict of leakage-free scalar features.
    """
    import torch

    p = _softmax(original_logits)
    q = _softmax(masked_logits)
    m = 0.5 * (p + q)

    kl = _kl(p, q)
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    k = min(topk, int(p.numel()))
    p_top = torch.topk(p, k).indices.tolist()
    q_top = torch.topk(q, k).indices.tolist()
    top1_changed = 1.0 if p_top[0] != q_top[0] else 0.0
    topk_overlap = len(set(p_top) & set(q_top)) / float(k)

    ent_p = _entropy(p)
    ent_q = _entropy(q)

    # margin = top1 - top2 probability (higher = more confident); delta masked-original
    p_sorted = torch.topk(p, 2).values.tolist()
    q_sorted = torch.topk(q, 2).values.tolist()
    margin_p = p_sorted[0] - p_sorted[1]
    margin_q = q_sorted[0] - q_sorted[1]

    # how much the original's preferred token lost probability under masking
    orig_top1 = p_top[0]
    logprob_delta = float(math.log(max(float(q[orig_top1]), _EPS)) - math.log(max(float(p[orig_top1]), _EPS)))

    features = {
        "self_output_kl": kl,
        "self_output_js": js,
        "self_output_top1_changed": top1_changed,
        "self_output_topk_overlap": topk_overlap,
        "self_output_entropy_delta": ent_q - ent_p,
        "self_output_margin_delta": margin_q - margin_p,
        "self_output_logprob_delta": logprob_delta,
    }
    assert_no_banned_output_effect_names(list(features.keys()))
    return features


def output_effect_shift_score(features: dict[str, Any]) -> float:
    """A single non-negative 'how much did the output move' magnitude for ranking.

    Larger = the span mattered more to the model's own next-token distribution.
    Uses only leakage-free output-shift components.
    """
    kl = _safe(features.get("self_output_kl"))
    js = _safe(features.get("self_output_js"))
    top1 = _safe(features.get("self_output_top1_changed"))
    overlap = _safe(features.get("self_output_topk_overlap"))
    logprob_drop = max(0.0, -_safe(features.get("self_output_logprob_delta")))
    # js is bounded [0, ln2]; kl unbounded; combine robustly
    return js + 0.5 * min(kl, 5.0) + 0.5 * top1 + 0.5 * (1.0 - overlap) + 0.25 * min(logprob_drop, 5.0)


def _safe(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def assert_no_banned_output_effect_names(feature_names: list[str]) -> None:
    bad = [n for n in feature_names for tok in BANNED_FEATURE_SUBSTRINGS if tok in n]
    if bad:
        raise AssertionError(
            f"output-effect feature names must not contain {BANNED_FEATURE_SUBSTRINGS}; offenders: {bad[:5]}"
        )
