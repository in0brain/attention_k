"""Gold-free F1/F4 feature math and grouped-CV utilities for Sprint 4C."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from recover_attention import mlp_readout_attribution as mra


def _distribution(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    shifted = array - array.max()
    probs = np.exp(shifted)
    return probs / probs.sum()


def margin_entropy(values: list[float]) -> dict[str, Any]:
    """Return option-only top-two margin and entropy."""
    ordered = sorted((float(value) for value in values), reverse=True)
    if len(ordered) < 2:
        raise ValueError("at least two option values are required")
    probs = _distribution([float(value) for value in values])
    return {
        "margin": float(ordered[0] - ordered[1]),
        "entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()),
        "top_index": int(np.argmax(values)),
        "probabilities": [float(value) for value in probs],
    }


def js_divergence(left: list[float], right: list[float]) -> float:
    p, q = _distribution(left), _distribution(right)
    midpoint = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(np.clip(p / midpoint, 1e-12, None))) + 0.5 * np.sum(q * np.log(np.clip(q / midpoint, 1e-12, None))))


def kl_divergence(left: list[float], right: list[float]) -> float:
    p, q = _distribution(left), _distribution(right)
    return float(np.sum(p * np.log(np.clip(p / q, 1e-12, None))))


def cross_layer_features(layer20: list[float], layer24: list[float], final: list[float], *, prefix: str) -> dict[str, float]:
    low, high = margin_entropy(layer20), margin_entropy(layer24)
    return {
        f"{prefix}_margin_L20": low["margin"],
        f"{prefix}_entropy_L20": low["entropy"],
        f"{prefix}_margin_L24": high["margin"],
        f"{prefix}_entropy_L24": high["entropy"],
        f"{prefix}_layer_disagreement": js_divergence(layer20, layer24),
        f"{prefix}_top_flip": float(low["top_index"] != high["top_index"]),
        f"{prefix}_final_vs_mid_kl": kl_divergence(final, layer24),
    }


def exact_vjp_projection(logit: Any, activation: Any) -> Any:
    """Exact finite-label J-lens scalar: <d logit / d activation, activation>."""
    import torch

    gradient = torch.autograd.grad(logit, activation, retain_graph=True, create_graph=False)[0]
    return (gradient * activation).sum()


def exact_vjp_position_projection(logit: Any, activation: Any, *, position: int) -> Any:
    """Exact J-lens projection at one sequence position of a module output."""
    import torch

    gradient = torch.autograd.grad(logit, activation, retain_graph=True, create_graph=False)[0]
    return (gradient[:, int(position), :] * activation[:, int(position), :]).sum()


def grouped_bootstrap_auroc(rows: list[dict[str, Any]], *, score_key: str, label_key: str, group_key: str, seed: int, iters: int = 500) -> list[float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values: list[float | None] = []
    for _ in range(iters):
        sampled = [grouped[key] for key in rng.choice(keys, size=len(keys), replace=True)]
        flat = [row for group in sampled for row in group]
        values.append(mra.auroc([int(row[label_key]) for row in flat], [float(row[score_key]) for row in flat]))
    return values


def grouped_bootstrap_increment(rows: list[dict[str, Any]], *, baseline_key: str, candidate_key: str, seed: int, iters: int = 500) -> list[float | None]:
    """Paired group bootstrap of candidate AUROC minus baseline AUROC."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    deltas: list[float | None] = []
    for _ in range(iters):
        flat = [row for key in rng.choice(keys, size=len(keys), replace=True) for row in grouped[key]]
        labels = [int(row["wrong_label"]) for row in flat]
        base = mra.auroc(labels, [float(row[baseline_key]) for row in flat])
        candidate = mra.auroc(labels, [float(row[candidate_key]) for row in flat])
        deltas.append(None if base is None or candidate is None else candidate - base)
    return deltas


def evaluate_grouped_cv(rows: list[dict[str, Any]], *, families: dict[str, list[str]], seeds: list[int]) -> dict[str, Any]:
    """Evaluate every family through the same grouped logistic-CV interface."""
    results: dict[str, Any] = {}
    for name, features in families.items():
        mra.assert_no_eval_only_features(features)
        runs = [mra.question_grouped_cv(rows, feature_names=features, label_key="wrong_label", question_key="group_id", seed=int(seed)) for seed in seeds]
        aucs = [run.get("oof_auroc") for run in runs if run.get("oof_auroc") is not None]
        seed_zero = runs[0].get("oof_predictions", []) if runs else []
        bootstrap_rows = [
            {"group_id": str(rows[item["row_index"]]["group_id"]), "wrong_label": item["wrong_label"], "risk_score": item["risk_score"]}
            for item in seed_zero
        ]
        bootstrap = grouped_bootstrap_auroc(bootstrap_rows, score_key="risk_score", label_key="wrong_label", group_key="group_id", seed=33070) if bootstrap_rows else []
        results[name] = {
            "feature_names": features,
            "seed_runs": runs,
            "mean_oof_auroc": float(np.mean(aucs)) if aucs else None,
            "mean_oof_auprc": float(np.mean([run["oof_auprc"] for run in runs if run.get("oof_auprc") is not None])) if aucs else None,
            "oof_auroc_ci95": percentile_ci(bootstrap),
        }
        # Bootstrap uses seed-0 OOF predictions, which are assigned below by the caller.
    return results


def percentile_ci(values: list[float | None]) -> list[float | None]:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not clean:
        return [None, None]
    return [float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))]
