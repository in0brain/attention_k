"""Sprint 3C-3: MLP readout attribution / detection helpers.

This module treats the final-answer readout MLP output as a diagnostic signal:
which number-like token is the MLP output pushing toward, and is that projection
useful for gold-free answer-risk detection?  It deliberately does not patch,
nudge, train a model, or make accuracy / hallucination claims.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

BACKEND = "mlp_readout_attribution_probe_v0"

EVAL_ONLY_FIELD_NAMES = {
    "gold_answer",
    "gold_answer_token_sequence",
    "wrong_answer_token_sequence",
    "is_model_answer_correct",
    "mlp_top_number_matches_gold",
    "mlp_top_number_matches_wrong",
}

FEATURE_NAMES = [
    "mlp_top1_number_score",
    "mlp_top2_number_score",
    "mlp_number_margin",
    "mlp_number_entropy",
    "mlp_projection_norm",
    "mlp_projection_sharpness",
    "mlp_logit_lens_agreement_with_final_answer",
    "mlp_agreement_with_model_generated_answer",
    "mlp_layer20_layer24_agreement",
    "mlp_layer20_layer24_margin_gap",
]

NUMBER_TOKEN_RE = re.compile(r"^[\s_]*(?:[-+]?[\d,]+(?:\.\d+)?%?|\d+/\d+)[\s_]*$")
DIGIT_RE = re.compile(r"\d")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite(value) else default


def mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if _finite(v)]
    return float(np.mean(clean)) if clean else None


def stable_int_seed(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        try:
            return str(tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            return str(token_id)


def is_pure_whitespace_token(text: str | None) -> bool:
    if text is None:
        return False
    cleaned = str(text).replace("Ġ", "").replace("Ċ", "").replace("▁", "")
    return cleaned.strip() == ""


def is_number_like_token_text(text: str | None) -> bool:
    if text is None or is_pure_whitespace_token(text):
        return False
    cleaned = str(text).replace("Ġ", "").replace("Ċ", "").replace("▁", "").strip()
    if not cleaned:
        return False
    return bool(NUMBER_TOKEN_RE.match(cleaned) or (len(cleaned) <= 4 and DIGIT_RE.search(cleaned)))


def filter_number_like_tokens(
    tokenizer: Any,
    token_ids: list[int] | None = None,
    *,
    extra_answer_token_ids: list[int] | None = None,
    max_vocab_scan: int | None = None,
) -> list[int]:
    """Return number-like token ids, never pure whitespace tokens.

    If ``token_ids`` is omitted, the tokenizer vocabulary is scanned. Extra
    answer tokens are appended after filtering so candidate answers remain in the
    projection subset without using gold labels as features.
    """

    candidates: list[int] = []
    if token_ids is not None:
        candidates.extend(int(t) for t in token_ids)
    elif hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        ids = sorted(int(v) for v in vocab.values())
        if max_vocab_scan is not None:
            ids = ids[: int(max_vocab_scan)]
        candidates.extend(ids)
    elif hasattr(tokenizer, "vocab"):
        candidates.extend(int(v) for v in getattr(tokenizer, "vocab").values())
    else:
        raise ValueError("token_ids must be provided when tokenizer vocabulary is unavailable")

    if extra_answer_token_ids:
        candidates.extend(int(t) for t in extra_answer_token_ids)

    out: list[int] = []
    seen: set[int] = set()
    for token_id in candidates:
        if token_id in seen:
            continue
        text = token_text(tokenizer, token_id)
        if is_number_like_token_text(text):
            seen.add(token_id)
            out.append(token_id)
    return out


def project_to_unembedding(vector: Any, unembedding_weight: Any) -> Any:
    """Project one hidden vector onto lm_head rows.

    ``unembedding_weight`` is expected to be shaped ``[vocab, hidden]``.  The
    returned score vector is shaped ``[vocab]``.
    """

    import torch

    vec = torch.as_tensor(vector).float()
    weight = torch.as_tensor(unembedding_weight).float()
    if weight.ndim != 2 or vec.ndim != 1 or weight.shape[1] != vec.shape[0]:
        raise ValueError("expected vector [hidden] and unembedding_weight [vocab, hidden]")
    return torch.matmul(weight, vec)


def _softmax_entropy(scores: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return float("nan"), float("nan")
    shifted = scores - float(np.max(scores))
    probs = np.exp(shifted)
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    top_prob = float(np.max(probs))
    return entropy, top_prob


def extract_projection_features(
    scores: Any,
    *,
    tokenizer: Any,
    number_token_ids: list[int],
    top_k: int = 20,
    prefix: str = "mlp",
    vector_norm: float | None = None,
) -> dict[str, Any]:
    """Gold-free projection features over a number-token subset."""

    import torch

    score_tensor = torch.as_tensor(scores).detach().float().cpu()
    valid_ids = [int(t) for t in number_token_ids if 0 <= int(t) < int(score_tensor.numel())]
    if not valid_ids:
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

    subset = score_tensor[valid_ids]
    k = min(int(top_k), int(subset.numel()))
    top = torch.topk(subset, k)
    top_ids = [valid_ids[int(i)] for i in top.indices.tolist()]
    top_scores = [float(v) for v in top.values.tolist()]
    arr = subset.numpy().astype(float)
    entropy, top_prob = _softmax_entropy(arr)
    top1 = top_scores[0] if top_scores else None
    top2 = top_scores[1] if len(top_scores) > 1 else None
    margin = top1 - top2 if top1 is not None and top2 is not None else top1
    return {
        f"{prefix}_top1_number_token_id": top_ids[0] if top_ids else None,
        f"{prefix}_top1_number_token": token_text(tokenizer, top_ids[0]) if top_ids else None,
        f"{prefix}_top1_number_score": top1,
        f"{prefix}_top2_number_score": top2,
        f"{prefix}_number_margin": margin,
        f"{prefix}_number_entropy": entropy,
        f"{prefix}_number_top1_prob": top_prob,
        f"{prefix}_projection_norm": vector_norm,
        f"{prefix}_projection_sharpness": top_prob,
        f"{prefix}_top_k_projected_number_tokens": [
            {"token_id": tid, "token_text": token_text(tokenizer, tid), "score": score}
            for tid, score in zip(top_ids, top_scores)
        ],
    }


def top_token_agreement(a: int | None, b: int | None) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if int(a) == int(b) else 0.0


def merge_layer_features(layer_features: dict[int, dict[str, Any]], *, primary_layer: int = 24, secondary_layer: int = 20) -> dict[str, Any]:
    primary = dict(layer_features.get(int(primary_layer)) or {})
    secondary = layer_features.get(int(secondary_layer)) or {}
    p_top = primary.get("mlp_top1_number_token_id")
    s_top = secondary.get("mlp_top1_number_token_id")
    primary["mlp_layer20_layer24_agreement"] = top_token_agreement(p_top, s_top)
    primary["mlp_layer20_layer24_margin_gap"] = _as_float(primary.get("mlp_number_margin")) - _as_float(secondary.get("mlp_number_margin"))
    return primary


def compute_mlp_readout_risk(features: dict[str, Any]) -> float:
    """Deterministic gold-free risk score: higher means more likely wrong."""

    entropy = _as_float(features.get("mlp_number_entropy"))
    margin = _as_float(features.get("mlp_number_margin"))
    layer_agreement = _as_float(features.get("mlp_layer20_layer24_agreement"))
    answer_agreement = _as_float(features.get("mlp_agreement_with_model_generated_answer"))
    sharpness = _as_float(features.get("mlp_projection_sharpness"))
    return float(entropy - margin + (1.0 - layer_agreement) + (1.0 - answer_agreement) + (1.0 - sharpness))


def minmax_normalize(values: list[float | None]) -> list[float | None]:
    clean = [float(v) for v in values if _finite(v)]
    if not clean:
        return [None for _ in values]
    lo, hi = min(clean), max(clean)
    if abs(hi - lo) < 1e-12:
        return [0.5 if _finite(v) else None for v in values]
    return [(float(v) - lo) / (hi - lo) if _finite(v) else None for v in values]


def assert_no_eval_only_features(feature_names: list[str]) -> None:
    leaked = sorted(set(feature_names) & EVAL_ONLY_FIELD_NAMES)
    leaked += sorted(name for name in feature_names if name.startswith("gold_") or "gold_answer" in name)
    if leaked:
        raise ValueError(f"eval-only fields leaked into feature list: {sorted(set(leaked))}")


def auroc(labels: list[int], scores: list[float]) -> float | None:
    """Tie-aware AUROC for positive label = 1."""

    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if _finite(s)]
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def auprc(labels: list[int], scores: list[float]) -> float | None:
    pairs = sorted(
        [(int(y), float(s)) for y, s in zip(labels, scores) if _finite(s)],
        key=lambda p: p[1],
        reverse=True,
    )
    total_pos = sum(1 for y, _ in pairs if y == 1)
    if total_pos == 0:
        return None
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for y, _score in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return float(area)


def evaluate_correct_wrong_detection(labels: list[int], scores: list[float]) -> dict[str, Any]:
    """Evaluate risk scores where positive label is wrong / high-risk."""

    clean = [(int(y), float(s)) for y, s in zip(labels, scores) if _finite(s)]
    return {
        "num_examples": len(clean),
        "positive_label": "wrong_answer_case",
        "auroc": auroc([y for y, _ in clean], [s for _, s in clean]),
        "auprc": auprc([y for y, _ in clean], [s for _, s in clean]),
        "positive_rate": float(np.mean([y for y, _ in clean])) if clean else None,
    }


def question_grouped_folds(question_ids: list[str], *, k: int = 5, seed: int = 33070) -> list[tuple[list[int], list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(question_ids):
        groups[str(qid)].append(i)
    unique = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    k = max(2, min(int(k), len(unique)))
    folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(k):
        test_groups = set(unique[fold_idx::k])
        test = [i for g in test_groups for i in groups[g]]
        train = [i for g in unique if g not in test_groups for i in groups[g]]
        folds.append((train, test))
    return folds


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _fit_logistic_numpy(x_train: np.ndarray, y_train: np.ndarray, *, steps: int = 500, lr: float = 0.1, l2: float = 0.01) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=float)
    for _ in range(steps):
        pred = _sigmoid(x_aug @ weights)
        grad = (x_aug.T @ (pred - y_train)) / max(len(y_train), 1)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def _predict_logistic_numpy(x_test: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    return _sigmoid(x_aug @ weights)


def question_grouped_cv(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    label_key: str = "wrong_label",
    question_key: str = "source_question_id",
    k: int = 5,
    seed: int = 33070,
) -> dict[str, Any]:
    """Small numpy logistic probe with question-grouped splits.

    Positive label is wrong / high-risk.  Gold-derived feature names are rejected
    up front.
    """

    assert_no_eval_only_features(feature_names)
    clean_rows = [r for r in rows if r.get(label_key) in {0, 1, False, True}]
    if len(clean_rows) < 6:
        return {"available": False, "reason": "too_few_examples", "feature_names": feature_names}
    labels = np.array([1.0 if bool(r[label_key]) else 0.0 for r in clean_rows], dtype=float)
    if len(set(labels.tolist())) < 2:
        return {"available": False, "reason": "single_class", "feature_names": feature_names}
    features = np.array([[_as_float(r.get(name)) for name in feature_names] for r in clean_rows], dtype=float)
    question_ids = [str(r.get(question_key)) for r in clean_rows]
    folds = question_grouped_folds(question_ids, k=k, seed=seed)
    fold_rows = []
    predictions = np.full(len(clean_rows), np.nan, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            fold_rows.append({"fold": fold_idx, "available": False, "reason": "single_class_in_fold"})
            continue
        train = features[train_idx]
        test = features[test_idx]
        mu = train.mean(axis=0)
        sigma = train.std(axis=0)
        sigma[sigma < 1e-9] = 1.0
        weights = _fit_logistic_numpy((train - mu) / sigma, y_train)
        pred = _predict_logistic_numpy((test - mu) / sigma, weights)
        predictions[test_idx] = pred
        fold_rows.append(
            {
                "fold": fold_idx,
                "available": True,
                "num_train": len(train_idx),
                "num_test": len(test_idx),
                "train_questions": len({question_ids[i] for i in train_idx}),
                "test_questions": len({question_ids[i] for i in test_idx}),
                "auroc": auroc(y_test.astype(int).tolist(), pred.tolist()),
                "auprc": auprc(y_test.astype(int).tolist(), pred.tolist()),
            }
        )
    valid = [(int(y), float(p)) for y, p in zip(labels.astype(int).tolist(), predictions.tolist()) if _finite(p)]
    return {
        "available": bool(valid),
        "backend": "numpy_logistic_probe_v0",
        "feature_names": list(feature_names),
        "label": "wrong_label",
        "question_grouped": True,
        "num_examples": len(clean_rows),
        "num_questions": len(set(question_ids)),
        "folds": fold_rows,
        "oof_auroc": auroc([y for y, _ in valid], [p for _, p in valid]) if valid else None,
        "oof_auprc": auprc([y for y, _ in valid], [p for _, p in valid]) if valid else None,
    }


def calibration_buckets(
    scores: list[float | None],
    labels: list[int],
    *,
    n_buckets: int = 5,
    higher_score_more_risk: bool = True,
) -> dict[str, Any]:
    """Quantile buckets and simple calibration error for wrong-label risk."""

    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if _finite(s)]
    if not pairs:
        return {"num_examples": 0, "buckets": [], "ece": None}
    pairs.sort(key=lambda p: p[0], reverse=not higher_score_more_risk)
    # For reporting high risk first, sort descending after assigning quantiles.
    ordered = sorted(pairs, key=lambda p: p[0], reverse=True)
    chunks = np.array_split(np.array(ordered, dtype=float), min(int(n_buckets), len(ordered)))
    buckets = []
    ece = 0.0
    total = len(ordered)
    for idx, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        bucket_scores = chunk[:, 0]
        bucket_labels = chunk[:, 1]
        mean_score = float(bucket_scores.mean())
        wrong_rate = float(bucket_labels.mean())
        ece += (len(chunk) / total) * abs(mean_score - wrong_rate)
        buckets.append(
            {
                "bucket": int(idx),
                "risk_rank": "high_to_low",
                "num_examples": int(len(chunk)),
                "score_min": float(bucket_scores.min()),
                "score_max": float(bucket_scores.max()),
                "mean_score": mean_score,
                "wrong_rate": wrong_rate,
                "correct_rate": 1.0 - wrong_rate,
            }
        )
    return {"num_examples": total, "buckets": buckets, "ece": float(ece)}


def random_baseline_scores(n: int, *, seed: int = 33071) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.random(int(n))]

