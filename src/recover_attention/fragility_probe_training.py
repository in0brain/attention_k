"""Sprint 2H-B task 5: leakage-safe fragility probe + baselines + gate metrics.

The MAIN probe (``hidden_state_probe_no_recovered_channel``) may only use features
available BEFORE recovery is run: original + masked hidden-state cosine features.
Any feature whose name contains ``recovered`` is forbidden for the gate, because the
fragility labels were derived from recovery behaviour -- feeding recovered-channel
features would let the probe read the label instead of predicting it (this is the
2G recovered-filler leakage in a new guise).

Feature sets:
    span_type_only               -> one-hot(span_type)                     [baseline]
    surface_rule                 -> one-hot(span_type) + inference-time surface stats [baseline]
    hidden_no_recovered          -> *_original_masked_cosine_* only        [GATE-eligible]
    hidden_with_recovered        -> all cosine features                    [leakage diagnostic only]

Only ``hidden_no_recovered`` is eligible for the review gate.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from recover_attention.probe_training import (
    build_feature_name_space,
    decision_function,
    flatten_probe_record,
    make_stratified_k_folds,
    standardize_train_and_test,
    train_ridge_classifier_ovr,
)


RECOVERED_TOKEN = "recovered"
NUMERIC_RE = re.compile(r"\d+(\.\d+)?|\$?\d+|percent|%|half|twice|double|triple", re.IGNORECASE)

FEATURE_SETS = [
    "span_type_only",
    "surface_rule",
    "hidden_no_recovered",
    "hidden_pre_recovery_enriched",
    "hidden_with_recovered",
]
# 2H-C: the enriched pre-recovery set is the new gate candidate; hidden_no_recovered
# is kept for continuity, hidden_with_recovered stays a leakage diagnostic only.
# 2I adds attention feature sets (registered dynamically by the 2I script via
# FEATURE_SETS_2I); the 2I gate candidate is hidden_plus_attention_pre_recovery.
GATE_ELIGIBLE = "hidden_pre_recovery_enriched"
GATE_ELIGIBLE_SETS = {
    "hidden_no_recovered", "hidden_pre_recovery_enriched",
    "attention_pre_recovery", "hidden_plus_attention_pre_recovery",
}
BASELINE_SETS = ["span_type_only", "surface_rule"]
ENRICHED_FEATURE_KEY = "pre_recovery_features"
ATTENTION_FEATURE_KEY = "attention_features"
DICT_FEATURE_KEYS = {
    "hidden_pre_recovery_enriched": [ENRICHED_FEATURE_KEY],
    "attention_pre_recovery": [ATTENTION_FEATURE_KEY],
    "hidden_plus_attention_pre_recovery": [ENRICHED_FEATURE_KEY, ATTENTION_FEATURE_KEY],
}


# --------------------------------------------------------------------------- #
# feature construction
# --------------------------------------------------------------------------- #
def _hidden_base_names(records: list[dict[str, Any]], *, exclude_recovered: bool) -> list[str]:
    names = build_feature_name_space(records)
    if exclude_recovered:
        names = [n for n in names if RECOVERED_TOKEN not in n]
    return names


def _one_hot_span_types(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    span_types = sorted({r.get("span_type", "unknown") for r in records})
    index = {t: i for i, t in enumerate(span_types)}
    matrix = np.zeros((len(records), len(span_types)), dtype=float)
    for row, record in enumerate(records):
        matrix[row, index[record.get("span_type", "unknown")]] = 1.0
    return matrix, [f"span_type={t}" for t in span_types]


def _surface_stats(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for record in records:
        span_text = record.get("span_text", "") or ""
        question = record.get("question", "") or ""
        is_numeric = 1.0 if NUMERIC_RE.search(span_text) else 0.0
        span_char_len = float(len(span_text))
        span_word_len = float(len(span_text.split()))
        q_len = float(len(question)) or 1.0
        pos = question.find(span_text)
        rel_pos = float(pos) / q_len if pos >= 0 else -1.0
        rows.append([is_numeric, span_char_len, span_word_len, float(len(question)), rel_pos])
    names = [
        "surf_is_numeric",
        "surf_span_char_len",
        "surf_span_word_len",
        "surf_question_char_len",
        "surf_span_rel_position",
    ]
    return np.array(rows, dtype=float), names


def build_feature_matrix(records: list[dict[str, Any]], feature_set: str) -> dict[str, Any]:
    """Build a dense feature matrix for a named feature set (no leakage indicators)."""
    if feature_set == "span_type_only":
        matrix, names = _one_hot_span_types(records)
        return {"matrix": matrix, "feature_names": names}

    if feature_set == "surface_rule":
        onehot, onehot_names = _one_hot_span_types(records)
        stats, stats_names = _surface_stats(records)
        return {
            "matrix": np.hstack([onehot, stats]),
            "feature_names": onehot_names + stats_names,
        }

    if feature_set in DICT_FEATURE_KEYS:
        return _build_dict_matrix(records, DICT_FEATURE_KEYS[feature_set])

    exclude = feature_set == "hidden_no_recovered"
    base_names = _hidden_base_names(records, exclude_recovered=exclude)
    matrix = np.zeros((len(records), len(base_names)), dtype=float)
    for row, record in enumerate(records):
        values, _missing = flatten_probe_record(record, base_names)
        for col, name in enumerate(base_names):
            matrix[row, col] = values[name]
    return {"matrix": matrix, "feature_names": base_names}


def _build_dict_matrix(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """Build a dense matrix from one or more flat scalar-feature dicts on each record."""
    names: set[str] = set()
    for record in records:
        for key in keys:
            names.update((record.get(key) or {}).keys())
    feature_names = sorted(names)
    index = {n: c for c, n in enumerate(feature_names)}
    matrix = np.zeros((len(records), len(feature_names)), dtype=float)
    for row, record in enumerate(records):
        merged: dict[str, Any] = {}
        for key in keys:
            merged.update(record.get(key) or {})
        for name, value in merged.items():
            if value is not None and math.isfinite(float(value)):
                matrix[row, index[name]] = float(value)
    return {"matrix": matrix, "feature_names": feature_names}


def _build_enriched_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_dict_matrix(records, [ENRICHED_FEATURE_KEY])


def assert_no_recovered_features(feature_names: list[str]) -> None:
    leaked = [n for n in feature_names if RECOVERED_TOKEN in n]
    if leaked:
        raise AssertionError(
            f"gate-eligible probe must not use recovered-channel features; found {leaked[:5]}"
        )


# --------------------------------------------------------------------------- #
# ridge regression (numpy)
# --------------------------------------------------------------------------- #
def train_ridge_regression(matrix: np.ndarray, targets: np.ndarray, *, alpha: float) -> dict[str, Any]:
    x_bias = np.column_stack([matrix, np.ones(matrix.shape[0])])
    gram = x_bias.T @ x_bias
    reg = np.eye(gram.shape[0]) * alpha
    reg[-1, -1] = 0.0
    weights = np.linalg.pinv(gram + reg) @ x_bias.T @ targets
    return {"weights": weights[:-1], "bias": float(weights[-1]), "alpha": alpha}


def predict_ridge_regression(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return matrix @ model["weights"] + model["bias"]


# --------------------------------------------------------------------------- #
# metrics (numpy only; sklearn is unavailable in this env)
# --------------------------------------------------------------------------- #
def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom == 0:
        return None
    return round(float((ra * rb).sum()) / denom, 6)


def mann_whitney_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float | None:
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None
    combined = np.concatenate([scores_pos, scores_neg])
    ranks = _rankdata(combined)
    rank_pos_sum = float(ranks[: len(scores_pos)].sum())
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 6)


def pairwise_ordering_accuracy(scores: np.ndarray, buckets: np.ndarray) -> float | None:
    concordant = 0.0
    comparable = 0
    n = len(scores)
    for i in range(n):
        for j in range(i + 1, n):
            if buckets[i] == buckets[j]:
                continue
            comparable += 1
            score_diff = scores[i] - scores[j]
            bucket_diff = buckets[i] - buckets[j]
            if score_diff == 0:
                concordant += 0.5
            elif (score_diff > 0) == (bucket_diff > 0):
                concordant += 1.0
    if comparable == 0:
        return None
    return round(concordant / comparable, 6)


def per_class_prf(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[str(label)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return out


def classification_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, Any]:
    prf = per_class_prf(y_true, y_pred, labels)
    present = [label for label in labels if prf[str(label)]["support"] > 0]
    macro_f1 = sum(prf[str(label)]["f1"] for label in present) / len(present) if present else 0.0
    balanced_acc = sum(prf[str(label)]["recall"] for label in present) / len(present) if present else 0.0
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    confusion = {str(a): {str(b): 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        confusion[str(t)][str(p)] += 1
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "per_class": prf,
        "bucket_3_recall": prf.get("3", {}).get("recall"),
        "confusion_matrix": confusion,
    }


# --------------------------------------------------------------------------- #
# CV runner
# --------------------------------------------------------------------------- #
def _safe_num_folds(buckets: list[int], requested: int) -> tuple[int, list[str]]:
    counts = Counter(buckets)
    min_count = min(counts.values())
    warnings: list[str] = []
    folds = requested
    if requested > min_count:
        folds = max(2, min_count)
        warnings.append(f"num_folds reduced from {requested} to {folds} (min bucket count {min_count})")
    return folds, warnings


def run_cv_for_feature_set(
    records: list[dict[str, Any]],
    buckets: list[int],
    risk_strength: list[float],
    feature_set: str,
    *,
    labels: list[int],
    alpha: float,
    num_folds: int,
    seed: int,
) -> dict[str, Any]:
    built = build_feature_matrix(records, feature_set)
    matrix = built["matrix"]
    feature_names = built["feature_names"]
    if feature_set in GATE_ELIGIBLE_SETS:
        assert_no_recovered_features(feature_names)
    if feature_set == "hidden_pre_recovery_enriched":
        from recover_attention.pre_recovery_features import assert_no_banned_feature_names

        assert_no_banned_feature_names(feature_names)
    if feature_set in ("attention_pre_recovery", "hidden_plus_attention_pre_recovery"):
        # 2I expands the banned list with answer / label / target
        from recover_attention.attention_features import assert_no_banned_attention_names

        assert_no_banned_attention_names(feature_names)

    bucket_array = np.array(buckets, dtype=int)
    strength_array = np.array(risk_strength, dtype=float)
    class_labels = [str(label) for label in labels]

    folds, warnings = _safe_num_folds(buckets, num_folds)
    fold_splits = make_stratified_k_folds([str(b) for b in buckets], folds, seed=seed)

    n = len(records)
    oof_pred_bucket = np.full(n, -1, dtype=int)
    oof_reg_score = np.full(n, np.nan, dtype=float)
    oof_decision: list[dict[str, float] | None] = [None] * n

    for train_idx, test_idx in fold_splits:
        train_x, test_x, _, _ = standardize_train_and_test(matrix[train_idx], matrix[test_idx])
        # classification
        clf = train_ridge_classifier_ovr(
            train_x, [str(b) for b in bucket_array[train_idx]], class_labels, alpha=alpha
        )
        scores = decision_function(test_x, clf)
        pred_idx = scores.argmax(axis=1)
        for local, global_idx in enumerate(test_idx):
            oof_pred_bucket[global_idx] = int(class_labels[pred_idx[local]])
            oof_decision[global_idx] = {
                class_labels[c]: float(scores[local, c]) for c in range(len(class_labels))
            }
        # regression on risk_strength
        reg = train_ridge_regression(train_x, strength_array[train_idx], alpha=alpha)
        reg_pred = predict_ridge_regression(test_x, reg)
        for local, global_idx in enumerate(test_idx):
            oof_reg_score[global_idx] = float(reg_pred[local])

    y_true = bucket_array.tolist()
    y_pred = oof_pred_bucket.tolist()
    metrics = classification_metrics(y_true, y_pred, labels)

    # ordinal / ranking metrics from regression score
    valid = ~np.isnan(oof_reg_score)
    spearman = spearman_corr(oof_reg_score[valid], bucket_array[valid].astype(float))
    pairwise = pairwise_ordering_accuracy(oof_reg_score[valid], bucket_array[valid])
    pos = oof_reg_score[(bucket_array == 3) & valid]
    neg = oof_reg_score[(bucket_array == 1) & valid]
    bucket_3_vs_1_auc = mann_whitney_auc(pos, neg)

    metrics.update(
        {
            "spearman_score_vs_bucket": spearman,
            "pairwise_ordering_accuracy": pairwise,
            "bucket_3_vs_bucket_1_auc": bucket_3_vs_1_auc,
            "num_features": matrix.shape[1],
            "num_folds": folds,
            "warnings": warnings,
        }
    )
    return {
        "metrics": metrics,
        "oof_pred_bucket": oof_pred_bucket.tolist(),
        "oof_reg_score": [None if np.isnan(v) else round(float(v), 6) for v in oof_reg_score],
        "oof_decision": oof_decision,
        "feature_names": feature_names,
    }


# --------------------------------------------------------------------------- #
# gate helpers: coverage / budget / bootstrap
# --------------------------------------------------------------------------- #
def per_question_topk_coverage(
    records: list[dict[str, Any]],
    reg_scores: list[float | None],
    *,
    k: int = 1,
) -> dict[str, Any]:
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_q[record.get("source_question_id", record.get("id", str(i)))].append(i)

    covered = 0
    total_on_path = 0
    questions_with_on_path = 0
    for _qid, idxs in by_q.items():
        on_path = [
            i
            for i in idxs
            if records[i].get("span_type") == "number"
            and records[i].get("solution_path_status") == "on_solution_path_number"
        ]
        if not on_path:
            continue
        questions_with_on_path += 1
        ranked = sorted(idxs, key=lambda i: (reg_scores[i] if reg_scores[i] is not None else -1e9), reverse=True)
        topk = set(ranked[:k])
        total_on_path += len(on_path)
        covered += sum(1 for i in on_path if i in topk)

    return {
        "k": k,
        "questions_with_on_path_number": questions_with_on_path,
        "on_path_numbers_total": total_on_path,
        "on_path_numbers_covered_topk": covered,
        "coverage": round(covered / total_on_path, 6) if total_on_path else None,
    }


def off_path_budget_share(records: list[dict[str, Any]], reg_scores: list[float | None]) -> dict[str, Any]:
    total_mass = 0.0
    off_path_mass = 0.0
    for i, record in enumerate(records):
        score = reg_scores[i]
        if score is None:
            continue
        mass = max(0.0, score)
        total_mass += mass
        if record.get("span_type") == "number" and record.get("solution_path_status") == "off_solution_path_number":
            off_path_mass += mass
    return {
        "total_predicted_strength_mass": round(total_mass, 6),
        "off_path_number_mass": round(off_path_mass, 6),
        "off_path_budget_share": round(off_path_mass / total_mass, 6) if total_mass else None,
    }


def bootstrap_delta_ci(
    y_true: list[int],
    pred_main: list[int],
    pred_baseline: list[int],
    labels: list[int],
    *,
    num_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    main_arr = np.array(pred_main)
    base_arr = np.array(pred_baseline)
    deltas = []
    for _ in range(num_boot):
        idx = rng.integers(0, n, size=n)
        m = classification_metrics(y_true_arr[idx].tolist(), main_arr[idx].tolist(), labels)["macro_f1"]
        b = classification_metrics(y_true_arr[idx].tolist(), base_arr[idx].tolist(), labels)["macro_f1"]
        deltas.append(m - b)
    deltas_sorted = sorted(deltas)
    lo = deltas_sorted[int(0.025 * num_boot)]
    hi = deltas_sorted[int(0.975 * num_boot)]
    point = classification_metrics(y_true, pred_main, labels)["macro_f1"] - classification_metrics(
        y_true, pred_baseline, labels
    )["macro_f1"]
    return {
        "macro_f1_delta_point": round(point, 6),
        "ci95_low": round(lo, 6),
        "ci95_high": round(hi, 6),
        "meaningfully_above_noise": bool(lo > 0.0),
        "num_bootstrap": num_boot,
    }
