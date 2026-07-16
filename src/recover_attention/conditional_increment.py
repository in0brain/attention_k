"""Sprint 4D-2 conditional-increment statistical + schema core.

Pure functions that决定 cache schema 与分类管线的正确性。跑错这些会强制重做
2880 条 teacher-forced forward,所以先在此实现 + 单测,再做 model-in-loop smoke。

设计权威:docs/paper/preregistration.md（v2.1, frozen）。
- completion-level population（一 completion 一条主样本）
- hidden tuple index = block+1（HF hidden_states 含 embedding 在 index 0）
- rank-biserial observability gate（max(0, 2·AUROC−1)）
- 增量 = AUROC(O+H) − AUROC(O)，paired grouped bootstrap（同任务同批）
- 跨任务 D = independent grouped bootstrap（两任务各自独立重采样）
- equivalence margin ε=0.02；artifact 红线 = Δ_artifact CI ⊂ [−ε,+ε]
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

PRIMARY_FAMILIES = ("attack", "cwe")
EPS_AUROC = 0.02          # equivalence margin（preregistration §4）
DELTA_RANK_BISERIAL = 0.15  # observability gate（preregistration §6）
DEFAULT_REL_DEPTH = 0.7
RQ2_MIN_PER_CELL = 15
RQ2_LOWPOWER_TOTAL_POS = 30


# ---------------------------------------------------------------- hidden index
def resolve_hidden_index(num_layers: int, rel_depth: float = DEFAULT_REL_DEPTH) -> dict[str, int]:
    """block_index_zero_based = floor(rel·L); hidden_states tuple index = block+1.

    HF hidden_states = (embedding, layer_1, …, layer_L)，index 0 = embedding，
    第 k 个 block 输出 = index k，故 tuple_index = block_index_zero_based + 1。
    Qwen L=28 → block 19, tuple 20；Llama L=32 → block 22, tuple 23。
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    block = int(np.floor(rel_depth * num_layers))
    block = max(0, min(block, num_layers - 1))
    return {"num_layers": num_layers, "rel_depth": rel_depth,
            "block_index_zero_based": block, "hidden_states_tuple_index": block + 1}


# ---------------------------------------------------------- completion schema
def completion_label(mentions: Sequence[dict]) -> dict[str, Any]:
    """一 completion → 一条主样本（preregistration §3）。

    eligible = 至少一个非 echoed 的 ATT&CK/CWE identifier。
    positive = 其中 ≥1 fabricated；negative = 全部 grounded。
    无 primary mention（拒答 / 未生成合法 id）→ emission_failure，不进主 AUROC。
    """
    primary = [m for m in mentions
               if m.get("family") in PRIMARY_FAMILIES
               and m.get("label") in ("grounded", "fabricated")]
    if not primary:
        return {"eligible": False, "label": None, "emission_failure": True,
                "n_primary_mentions": 0}
    label = 1 if any(m["label"] == "fabricated" for m in primary) else 0
    return {"eligible": True, "label": label, "emission_failure": False,
            "n_primary_mentions": len(primary)}


def eligible_identifier_token_positions(mentions: Sequence[dict]) -> list[int]:
    """H 的 pooling 位置：completion 内**所有**非 echoed ATT&CK/CWE identifier 的
    **全部 token** 位置（不是只取第一个）。mention 需带 token_indices 字段。"""
    pos: list[int] = []
    for m in mentions:
        if m.get("family") in PRIMARY_FAMILIES and m.get("label") in ("grounded", "fabricated"):
            pos.extend(m.get("token_indices", []) or [])
    return sorted(set(pos))


# ------------------------------------------------------------------- metrics
def auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    y = np.asarray(list(labels)); s = np.asarray(list(scores), dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def rank_biserial(a: float) -> float:
    """r_rb = 2·AUROC − 1（等价 Cliff's delta；base-rate 不变）。"""
    return 2.0 * a - 1.0


def s_observability(a: float) -> float:
    """S_t = max(0, 2·AUROC − 1)（AUROC<0.5=失效，不翻转成强可观测）。"""
    if a != a:  # nan
        return 0.0
    return max(0.0, 2.0 * a - 1.0)


def equivalence_read(ci_lo: float, ci_hi: float, eps: float = EPS_AUROC) -> str:
    """RQ1 判读（preregistration §4）。"""
    if ci_lo > eps:
        return "increment"
    if ci_hi < -eps:
        return "harmful"
    if ci_lo >= -eps and ci_hi <= eps:
        return "equivalent"
    return "inconclusive"


# --------------------------------------------------------------- grouped CV
def grouped_folds(groups: Sequence, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 group 分组的 CV folds（预注册、可复现）。同 group 的行永不跨 train/test。"""
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[perm[i]]: i % n_splits for i in range(len(uniq))}
    fold = np.array([fold_of[x] for x in g])
    out = []
    for k in range(n_splits):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        out.append((train, test))
    return out


def _risk_scores(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def oof_risk_scores(X, y: Sequence[int], groups: Sequence, make_clf: Callable[[], Any],
                    n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """分组 CV 的 out-of-fold 风险分。每个 fold 内 train-only 拟合（含标准化在 pipeline 内）。"""
    y = np.asarray(y)
    scores = np.full(len(y), np.nan, dtype=float)
    for train, test in grouped_folds(groups, n_splits, seed):
        clf = make_clf()
        clf.fit(X[train] if hasattr(X, "__getitem__") else X, y[train])
        scores[test] = _risk_scores(clf, X[test])
    return scores


# ------------------------------------------------------------- bootstrap CI
def _group_index_map(groups: np.ndarray) -> dict:
    return {gv: np.where(groups == gv)[0] for gv in np.unique(groups)}


def paired_grouped_bootstrap_delta(y: Sequence[int], s_o: Sequence[float], s_oh: Sequence[float],
                                   groups: Sequence, n_boot: int = 1000, seed: int = 0) -> dict:
    """Δ_H = AUROC(O+H) − AUROC(O) 的 paired grouped-bootstrap CI（同任务同批样本）。"""
    y = np.asarray(y); s_o = np.asarray(s_o, float); s_oh = np.asarray(s_oh, float)
    g = np.asarray(groups); uniq = np.unique(g); imap = _group_index_map(g)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([imap[gv] for gv in sampled])
        a_o = auroc(s_o[rows], y[rows]); a_oh = auroc(s_oh[rows], y[rows])
        if a_o == a_o and a_oh == a_oh:
            deltas.append(a_oh - a_o)
    d = np.asarray(deltas)
    point = auroc(s_oh, y) - auroc(s_o, y)
    return {"point": float(point), "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "n_valid": int(d.size)}


def independent_grouped_bootstrap_D(task_mcq: dict, task_h1: dict,
                                    n_boot: int = 1000, seed: int = 0) -> dict:
    """跨任务 D = S_MCQ − S_H1，independent grouped bootstrap（两任务各自独立重采样）。

    每个 task = {"y":…, "score":…, "groups":…}（score 为该任务 O 的 OOF 风险分）。
    S_t = max(0, 2·AUROC(O_t) − 1)。**非 paired**——两任务非同批 prompt。
    """
    def prep(t):
        return (np.asarray(t["y"]), np.asarray(t["score"], float), np.asarray(t["groups"]))
    ym, sm, gm = prep(task_mcq); yh, sh, gh = prep(task_h1)
    um, uh = np.unique(gm), np.unique(gh)
    im, ih = _group_index_map(gm), _group_index_map(gh)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        rm = np.concatenate([im[gv] for gv in rng.choice(um, size=len(um), replace=True)])
        rh = np.concatenate([ih[gv] for gv in rng.choice(uh, size=len(uh), replace=True)])
        s_m = s_observability(auroc(sm[rm], ym[rm]))
        s_h = s_observability(auroc(sh[rh], yh[rh]))
        ds.append(s_m - s_h)
    d = np.asarray(ds)
    point = s_observability(auroc(sm, ym)) - s_observability(auroc(sh, yh))
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    if lo > DELTA_RANK_BISERIAL:
        verdict = "h1_high_confidence"
    elif hi <= 0:
        verdict = "outcome_3"
    else:
        verdict = "uncertain"
    return {"D_point": float(point), "ci_lo": lo, "ci_hi": hi,
            "delta": DELTA_RANK_BISERIAL, "verdict": verdict}


def artifact_redline(delta_lo: float, delta_hi: float, eps: float = EPS_AUROC) -> bool:
    """Δ_artifact = AUROC(full-text) − AUROC(shortcut)；CI ⊂ [−ε,+ε] → shortcut ≈ full-text → 触发。"""
    return (delta_lo >= -eps) and (delta_hi <= eps)


# ----------------------------------------------------------------- RQ2 strata
def rq2_threshold(id_logprob_train: Sequence[float]) -> float:
    """固定分位数（中位数），只在 train fold 拟合。"""
    return float(np.median(np.asarray(id_logprob_train, float)))


def rq2_strata(id_logprob: Sequence[float], threshold: float) -> np.ndarray:
    """≥ 阈值 → high-confidence(True)，< → low-confidence(False)。"""
    return np.asarray(id_logprob, float) >= threshold


def rq2_cell_ok(y_cell: Sequence[int]) -> bool:
    y = np.asarray(y_cell)
    return int((y == 1).sum()) >= RQ2_MIN_PER_CELL and int((y == 0).sum()) >= RQ2_MIN_PER_CELL


def rq2_power_flag(total_positives: int) -> str:
    return "exploratory_low_power" if total_positives < RQ2_LOWPOWER_TOTAL_POS else "adequate"


# -------------------------------------------------------- preregistration gate
def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_preregistration_frozen(prereg_path: str, lock_path: str) -> dict:
    """启动 gate G2：重算 preregistration.md 的 sha256 并与 lock 比对。"""
    cur = sha256_of(prereg_path)
    with open(lock_path, encoding="utf-8") as f:
        locked = f.read().split("sha256:")[1].split()[0].strip()
    return {"current": cur, "locked": locked, "match": cur == locked}
