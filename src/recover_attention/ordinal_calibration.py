"""Sprint 2H-D: ordinal calibration and budget-aware scoring of the enriched probe.

2H-C showed the enriched pre-recovery features beat surface on macro_f1/balanced_acc,
but not on ordinal ranking. This module turns the enriched signal into better-ordered
risk scores via three calibration methods, and adds budget-aware bucket-3 metrics that
are immune to the "predict number->bucket3" flooding trick surface_rule exploits.

Methods:
  A. expected-bucket score   : Sigma_k softmax(decision_k / T) * bucket(k), T fit on train
  B. ordinal-threshold score : Sigma_k P(bucket >= k) from 3 binary ridges
  C. calibrated regression    : ridge risk_strength regression + isotonic (PAV) calibration

All calibration is fit on train folds only. Metrics are numpy-only (no sklearn).
This module never reads recovered-channel / gold-path / drift / bucket / risk_strength
as *input features* - it only consumes the gold bucket as a supervision target.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from recover_attention.fragility_probe_training import (
    mann_whitney_auc,
    pairwise_ordering_accuracy,
    spearman_corr,
)

TEMPERATURE_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# --------------------------------------------------------------------------- #
# core numeric helpers
# --------------------------------------------------------------------------- #
def softmax_rows(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_bucket_from_decision(
    decision: np.ndarray, class_values: list[int], temperature: float
) -> np.ndarray:
    """E[bucket] = Sigma_k p_k * value_k, with p = softmax(decision / T)."""
    probs = softmax_rows(decision, temperature)
    values = np.array(class_values, dtype=float)
    return probs @ values


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(
    train_decision: np.ndarray,
    train_buckets: np.ndarray,
    class_values: list[int],
    grid: list[float] = TEMPERATURE_GRID,
) -> float:
    """Pick T maximizing train-fold Spearman(expected_bucket, gold bucket). Train-only."""
    best_t, best_s = 1.0, -2.0
    for t in grid:
        est = expected_bucket_from_decision(train_decision, class_values, t)
        s = spearman_corr(est, train_buckets.astype(float))
        if s is not None and s > best_s:
            best_s, best_t = s, t
    return best_t


def pav_isotonic(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic (non-decreasing) fit on y in x-order."""
    y = y.astype(float).copy()
    n = len(y)
    weights = np.ones(n)
    values = y.copy()
    i = 0
    # standard PAV
    level_val = list(values)
    level_wt = list(weights)
    idx = 0
    stack_val: list[float] = []
    stack_wt: list[float] = []
    stack_len: list[int] = []
    for k in range(n):
        v = level_val[k]
        w = level_wt[k]
        ln = 1
        while stack_val and stack_val[-1] > v:
            pv, pw, pl = stack_val.pop(), stack_wt.pop(), stack_len.pop()
            v = (pv * pw + v * w) / (pw + w)
            w = pw + w
            ln = pl + ln
        stack_val.append(v)
        stack_wt.append(w)
        stack_len.append(ln)
    out = np.empty(n)
    pos = 0
    for v, ln in zip(stack_val, stack_len):
        out[pos:pos + ln] = v
        pos += ln
    return out


def fit_isotonic(train_pred: np.ndarray, train_target: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a monotonic map pred->target on train; return an interpolating callable."""
    order = np.argsort(train_pred, kind="mergesort")
    xs = train_pred[order]
    ys = pav_isotonic(train_target[order])
    # collapse duplicate xs (keep last isotonic value) for np.interp monotonic xp
    uniq_x, uniq_y = [], []
    for x, y in zip(xs, ys):
        if uniq_x and x == uniq_x[-1]:
            uniq_y[-1] = y
        else:
            uniq_x.append(x)
            uniq_y.append(y)
    xp = np.array(uniq_x)
    fp = np.array(uniq_y)

    def _apply(pred: np.ndarray) -> np.ndarray:
        if len(xp) < 2:
            return np.full(len(pred), float(fp[0]) if len(fp) else 0.0)
        return np.interp(pred, xp, fp)

    return _apply


# --------------------------------------------------------------------------- #
# budget-aware bucket-3 metrics
# --------------------------------------------------------------------------- #
def budget_curve(
    scores: np.ndarray, buckets: np.ndarray, fractions: list[float] = (0.1, 0.2)
) -> dict[str, Any]:
    """Bucket-3 precision/recall inside the top-fraction highest-scored spans.

    This is immune to flooding: a baseline that marks *all* numbers bucket-3 cannot
    inflate precision inside a small top-budget slice.
    """
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    total_b3 = int((buckets == 3).sum())
    out: dict[str, Any] = {"total_bucket_3": total_b3, "n": n, "points": {}}
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        top = order[:k]
        top_b3 = int((buckets[top] == 3).sum())
        precision = top_b3 / k
        recall = top_b3 / total_b3 if total_b3 else None
        f1 = (2 * precision * recall / (precision + recall)) if (recall and (precision + recall) > 0) else 0.0
        out["points"][f"top_{int(frac * 100)}pct"] = {
            "k": k,
            "bucket_3_precision": round(precision, 6),
            "bucket_3_recall": round(recall, 6) if recall is not None else None,
            "bucket_3_f1": round(f1, 6),
        }
    return out


def ranking_metrics(scores: np.ndarray, buckets: np.ndarray) -> dict[str, Any]:
    valid = ~np.isnan(scores)
    s = scores[valid]
    b = buckets[valid]
    pos = s[b == 3]
    neg = s[b == 1]
    return {
        "spearman": spearman_corr(s, b.astype(float)),
        "pairwise_ordering_accuracy": pairwise_ordering_accuracy(s, b),
        "bucket_3_vs_bucket_1_auc": mann_whitney_auc(pos, neg),
    }


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ranking_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    buckets: np.ndarray,
    *,
    metric: str = "spearman",
    num_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """CI95 of ranking_metric(a) - ranking_metric(b) over resampled test rows."""
    rng = np.random.default_rng(seed)
    n = len(buckets)

    def _m(sc, idx):
        # compute only the requested metric (avoid the O(n^2) pairwise metric per resample)
        if metric == "spearman":
            return spearman_corr(sc[idx], buckets[idx].astype(float))
        if metric == "bucket_3_vs_bucket_1_auc":
            b = buckets[idx]
            return mann_whitney_auc(sc[idx][b == 3], sc[idx][b == 1])
        return ranking_metrics(sc[idx], buckets[idx])[metric]

    deltas = []
    for _ in range(num_boot):
        idx = rng.integers(0, n, size=n)
        ma, mb = _m(scores_a, idx), _m(scores_b, idx)
        if ma is None or mb is None:
            continue
        deltas.append(ma - mb)
    if not deltas:
        return {"metric": metric, "point": None, "ci95_low": None, "ci95_high": None, "meaningfully_above_noise": False}
    ds = sorted(deltas)
    point = (ranking_metrics(scores_a, buckets)[metric] or 0.0) - (ranking_metrics(scores_b, buckets)[metric] or 0.0)
    return {
        "metric": metric,
        "point": round(point, 6),
        "ci95_low": round(ds[int(0.025 * len(ds))], 6),
        "ci95_high": round(ds[int(0.975 * len(ds))], 6),
        "meaningfully_above_noise": bool(ds[int(0.025 * len(ds))] > 0.0),
        "num_bootstrap": len(ds),
    }
