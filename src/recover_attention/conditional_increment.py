"""Sprint 4D-2 conditional-increment statistical + schema core.

Pure functions that决定 cache schema 与分类管线的正确性。跑错这些会强制重做
2880 条 teacher-forced forward,所以先在此实现 + 单测,再做 model-in-loop smoke。

设计权威:docs/paper/preregistration.md（v2.1, frozen）。
- completion-level population（一 completion 一条主样本）
- hidden tuple index = block+1（HF hidden_states 含 embedding 在 index 0）
- hidden pooling = 全部 eligible identifier token 的 mean（§8）
- output ladder = prompt / id-string / surface / full-text / F5 / F5+text=O（§7）
- 训练协议:同 outer stratified grouped folds、nested C{0.01,0.1,1,10}、
  sparse TF-IDF 与 dense 各自 block 内标准化后等权拼接（§7）
- rank-biserial observability gate（max(0, 2·AUROC−1)）
- 增量 = AUROC(O+H) − AUROC(O)，paired grouped bootstrap（同任务同批）
- 跨任务 D = independent grouped bootstrap（两任务各自独立重采样）
- equivalence margin ε=0.02；artifact 红线 = Δ_artifact CI ⊂ [−ε,+ε]
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PRIMARY_FAMILIES = ("attack", "cwe")
EPS_AUROC = 0.02          # equivalence margin（preregistration §4）
DELTA_RANK_BISERIAL = 0.15  # observability gate（preregistration §6）
DEFAULT_REL_DEPTH = 0.7
RQ2_MIN_PER_CELL = 15
RQ2_LOWPOWER_TOTAL_POS = 30
C_GRID = (0.01, 0.1, 1.0, 10.0)          # §7 nested-CV L2 网格（冻结）
BOOTSTRAP_MIN_VALID_FRAC = 0.8            # 有效重采样轮次下限（低于则停,不静默出 CI）

# §7 冻结的 TF-IDF 协议。跑后不许改。
WORD_TFIDF_PARAMS = dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                         max_features=50000, sublinear_tf=True)
CHAR_TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                         max_features=50000, sublinear_tf=True)


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
def _is_emitted(m: dict) -> bool:
    """非 echoed 的合法 identifier（任意 family）= 模型真的发射了一个 id。"""
    return m.get("label") in ("grounded", "fabricated")


def _is_primary(m: dict) -> bool:
    return m.get("family") in PRIMARY_FAMILIES and _is_emitted(m)


def completion_label(mentions: Sequence[dict], refusal: bool = False) -> dict[str, Any]:
    """一 completion → 一条主样本（preregistration §3）。

    eligible_primary = 至少一个非 echoed 的 ATT&CK/CWE identifier。
      positive = 其中 ≥1 fabricated；negative = 全部 grounded。
    非 eligible 的排除原因分开记（评审 P0）:
      refusal / no_identifier    → emission_failure=True（真的没发射 id）
      only_echoed                → 只复述 prompt 里的 id,也算没发射
      only_cve                   → **发射了** id,只是不在主分析 family 内。
                                   不是 emission failure,否则 end-to-end emission rate 被低估。
    """
    emitted_any = any(_is_emitted(m) for m in mentions)
    primary = [m for m in mentions if _is_primary(m)]
    if primary:
        label = 1 if any(m["label"] == "fabricated" for m in primary) else 0
        return {"eligible": True, "label": label,
                "emitted_any_identifier": True,
                "emission_failure": False,
                "primary_exclusion_reason": None,
                "n_primary_mentions": len(primary)}

    if refusal:
        reason = "refusal"
    elif emitted_any:
        reason = "only_cve"          # 发射了 id,但都不在 attack/cwe
    elif mentions:
        reason = "only_echoed"       # 有 mention,但全是 prompt 里抄的
    else:
        reason = "no_identifier"
    return {"eligible": False, "label": None,
            "emitted_any_identifier": bool(emitted_any),
            "emission_failure": bool(refusal or not emitted_any),
            "primary_exclusion_reason": reason,
            "n_primary_mentions": 0}


def eligible_identifier_token_positions(mentions: Sequence[dict]) -> list[int]:
    """H 的 pooling 位置：completion 内**所有**非 echoed ATT&CK/CWE identifier 的
    **全部 token** 位置（不是只取第一个）。mention 需带 token_indices 字段。"""
    pos: list[int] = []
    for m in mentions:
        if _is_primary(m):
            pos.extend(m.get("token_indices", []) or [])
    return sorted(set(pos))


def pool_hidden_states(hidden: Any, positions: Sequence[int]) -> np.ndarray:
    """H 向量 = 指定 token 位置在目标层 hidden 的 mean pooling（preregistration §8）。

    hidden: (T, D)（单条序列该层的 hidden；BF16/FP16 会被转成 FP32 再算）。
    positions: eligible identifier 的全部 token 位置；重复位置只计一次（去重后取均值,
      避免同一 token 被重复加权）。空位置 = ineligible,直接报错,不许静默出零向量。
    """
    h = np.asarray(hidden, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"hidden must be 2-D (T, D), got shape {h.shape}")
    uniq = sorted(set(int(p) for p in positions))
    if not uniq:
        raise ValueError("no eligible identifier token positions: completion is ineligible for H")
    bad = [p for p in uniq if p < 0 or p >= h.shape[0]]
    if bad:
        raise ValueError(f"token positions out of range for T={h.shape[0]}: {bad}")
    return h[uniq, :].mean(axis=0).astype(np.float32)


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
    """S_t = max(0, 2·AUROC − 1)（AUROC<0.5=失效,不翻转成强可观测）。

    注意:a 为 NaN（单类样本）时 AUROC 未定义,**不是** S=0。调用方必须先丢弃
    该次重采样,不能把"算不出"当"完全不可观测"。这里报错以防误用。
    """
    if a != a:  # nan
        raise ValueError("s_observability got NaN AUROC (single-class sample); "
                         "caller must discard this bootstrap round, not treat it as S=0")
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
    """无标签分层的 group folds。只用于没有 label 的场景;主管线用
    stratified_grouped_folds（正类率低时必须分层,否则单类 fold 会炸）。"""
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


def _folds_have_both_classes(y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> bool:
    for train, test in folds:
        if train.size == 0 or test.size == 0:
            return False
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            return False
    return True


def stratified_grouped_folds(y: Sequence[int], groups: Sequence, n_splits: int = 5,
                             seed: int = 0, max_seed_retries: int = 20
                             ) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 group 分组 + 按 label 分层的 outer folds（评审 P0）。

    H1 正类率低（4D-1 completion-level ≈0.15）,纯随机分 group 很可能出现
    某个 test fold 没有 fabrication（AUROC 未定义）或某个 train fold 单类
    （LogisticRegression 直接报错）。这里:
      1) 用 StratifiedGroupKFold（同 group 不跨 train/test,同时尽量平衡正类）;
      2) 显式校验每个 train/test fold 都有正负两类;
      3) 不满足则换 split seed 重试（确定性序列 seed, seed+1, …）;
      4) 全部重试失败 → 抛错停止,不静默降级。
    """
    y = np.asarray(y)
    g = np.asarray(groups)
    if len(np.unique(y)) < 2:
        raise ValueError("stratified_grouped_folds needs both classes present")
    if len(np.unique(g)) < n_splits:
        raise ValueError(f"need ≥{n_splits} groups for {n_splits}-fold, got {len(np.unique(g))}")
    last_err: Exception | None = None
    for attempt in range(max_seed_retries):
        s = int(seed) + attempt
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
            folds = [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y, g)]
        except ValueError as exc:   # sklearn 自身拒绝的划分
            last_err = exc
            continue
        for train, test in folds:
            assert set(g[train]).isdisjoint(set(g[test])), "group leaked across train/test"
        if _folds_have_both_classes(y, folds):
            return folds
    raise RuntimeError(
        f"no valid stratified grouped split after {max_seed_retries} seeds "
        f"(n={len(y)}, pos={int((y == 1).sum())}, groups={len(np.unique(g))}, "
        f"n_splits={n_splits}); last sklearn error={last_err}. "
        "样本/正例过少 → 停,不降级成单类 fold。"
    )


def _risk_scores(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def oof_risk_scores(X, y: Sequence[int], groups: Sequence, make_clf: Callable[[], Any],
                    n_splits: int = 5, seed: int = 0,
                    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    """分组 CV 的 out-of-fold 风险分。每个 fold 内 train-only 拟合（含标准化在 pipeline 内）。

    folds=None → 用 stratified_grouped_folds。所有 OOF 分数必须有限,否则报错。
    """
    y = np.asarray(y)
    if folds is None:
        folds = stratified_grouped_folds(y, groups, n_splits, seed)
    scores = np.full(len(y), np.nan, dtype=float)
    for train, test in folds:
        clf = make_clf()
        clf.fit(X[train], y[train])
        scores[test] = _risk_scores(clf, X[test])
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"OOF scores contain non-finite values "
                           f"({int((~np.isfinite(scores)).sum())}/{scores.size}); folds 未覆盖全部样本?")
    return scores


# ------------------------------------------------- output-side feature blocks
def _fit_text_block(train_text: Sequence[str], test_text: Sequence[str]) -> tuple[Any, Any, dict]:
    """word TF-IDF + char_wb TF-IDF 拼接。vocabulary **只在 train fold 拟合**（§7）。"""
    word = TfidfVectorizer(**WORD_TFIDF_PARAMS)
    char = TfidfVectorizer(**CHAR_TFIDF_PARAMS)
    tr = sp.hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    te = sp.hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    info = {"word_vocab": len(word.vocabulary_), "char_vocab": len(char.vocabulary_)}
    return tr, te, info


def _fit_dense_block(train_dense: np.ndarray, test_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train-fold z-score（§7）。常数列的 std=0 → 用 1 避免除零。"""
    tr = np.asarray(train_dense, dtype=float)
    te = np.asarray(test_dense, dtype=float)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def _block_equal_weight_scale(train_block) -> float:
    """逐块等权（§7）:按 train fold 的平均行范数归一,使每个 block 贡献相当。

    否则 50000 维 sparse TF-IDF 会把 3584 维 dense hidden 压制掉（或反之）。
    """
    if sp.issparse(train_block):
        norms = np.sqrt(np.asarray(train_block.multiply(train_block).sum(axis=1)).ravel())
    else:
        norms = np.linalg.norm(np.asarray(train_block, dtype=float), axis=1)
    rms = float(np.sqrt((norms ** 2).mean()))
    return 1.0 / rms if rms > 1e-12 else 1.0


def build_fold_design(train: np.ndarray, test: np.ndarray, *,
                      text: Sequence[str] | None = None,
                      dense: np.ndarray | None = None) -> tuple[Any, Any, dict]:
    """按 §7 协议组装一个 fold 的设计矩阵:各 block 内标准化 → 等权 → 拼接。

    text  = full-response-text / prompt / id-string 等文本 block（sparse）。
    dense = F5 / surface / hidden 等稠密 block。
    两者可任一为 None（阶梯的低阶 rung）。所有拟合只用 train。
    """
    if text is None and dense is None:
        raise ValueError("build_fold_design needs at least one of text/dense")
    blocks_tr, blocks_te = [], []
    info: dict[str, Any] = {}
    if text is not None:
        t = np.asarray(text, dtype=object)
        tr, te, tinfo = _fit_text_block(list(t[train]), list(t[test]))
        w = _block_equal_weight_scale(tr)
        blocks_tr.append(tr * w); blocks_te.append(te * w)
        info["text"] = {**tinfo, "block_scale": w, "n_features": tr.shape[1]}
    if dense is not None:
        d = np.asarray(dense, dtype=float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        tr, te = _fit_dense_block(d[train], d[test])
        w = _block_equal_weight_scale(tr)
        blocks_tr.append(sp.csr_matrix(tr * w)); blocks_te.append(sp.csr_matrix(te * w))
        info["dense"] = {"block_scale": w, "n_features": d.shape[1]}
    X_tr = sp.hstack(blocks_tr).tocsr()
    X_te = sp.hstack(blocks_te).tocsr()
    return X_tr, X_te, info


def _select_C(X_tr, y_tr: np.ndarray, groups_tr: np.ndarray, c_grid: Sequence[float],
              inner_splits: int, seed: int) -> float:
    """nested-CV 选 L2 的 C:只在 train/dev 内选,绝不看 test（§7）。"""
    try:
        inner = stratified_grouped_folds(y_tr, groups_tr, n_splits=inner_splits,
                                         seed=seed, max_seed_retries=10)
    except (RuntimeError, ValueError):
        return 1.0    # train fold 太小做不了内层 CV → 用网格中点默认值,并由调用方记录
    best_c, best_auc = 1.0, -np.inf
    for c in c_grid:
        aucs = []
        for itr, ite in inner:
            clf = LogisticRegression(C=float(c), penalty="l2", max_iter=2000, solver="liblinear")
            clf.fit(X_tr[itr], y_tr[itr])
            aucs.append(auroc(_risk_scores(clf, X_tr[ite]), y_tr[ite]))
        m = float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else -np.inf
        if m > best_auc:
            best_auc, best_c = m, float(c)
    return best_c


def block_oof_scores(y: Sequence[int], groups: Sequence,
                     folds: Sequence[tuple[np.ndarray, np.ndarray]], *,
                     text: Sequence[str] | None = None,
                     dense: np.ndarray | None = None,
                     c_grid: Sequence[float] = C_GRID,
                     inner_splits: int = 3, seed: int = 0) -> dict:
    """一个 ladder rung（或 H / O+H）的 OOF 风险分。

    O / H / O+H 走**同一条**代码路径:同 outer folds、同 nested C 网格、同 block 标准化,
    这样"无增量"不会是融合协议不一致造成的 artifact（§7）。
    """
    y = np.asarray(y); g = np.asarray(groups)
    scores = np.full(len(y), np.nan, dtype=float)
    chosen_c, fold_info = [], []
    for train, test in folds:
        X_tr, X_te, info = build_fold_design(train, test, text=text, dense=dense)
        c = _select_C(X_tr, y[train], g[train], c_grid, inner_splits, seed)
        clf = LogisticRegression(C=c, penalty="l2", max_iter=2000, solver="liblinear")
        clf.fit(X_tr, y[train])
        scores[test] = _risk_scores(clf, X_te)
        chosen_c.append(c); fold_info.append(info)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("block_oof_scores produced non-finite OOF values")
    return {"scores": scores, "auroc": auroc(scores, y), "chosen_C": chosen_c,
            "fold_design": fold_info}


# ------------------------------------------------------------- bootstrap CI
def _group_index_map(groups: np.ndarray) -> dict:
    return {gv: np.where(groups == gv)[0] for gv in np.unique(groups)}


def _percentile_ci(d: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def paired_grouped_bootstrap_delta(y: Sequence[int], s_o: Sequence[float], s_oh: Sequence[float],
                                   groups: Sequence, n_boot: int = 1000, seed: int = 0,
                                   min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC) -> dict:
    """Δ_H = AUROC(O+H) − AUROC(O) 的 paired grouped-bootstrap CI（同任务同批样本）。

    单类重采样（AUROC 未定义）→ **丢弃该轮**并计数;有效轮次占比不足 → 抛错,
    不静默用少数轮次出 CI。
    """
    y = np.asarray(y); s_o = np.asarray(s_o, float); s_oh = np.asarray(s_oh, float)
    g = np.asarray(groups); uniq = np.unique(g); imap = _group_index_map(g)
    rng = np.random.default_rng(seed)
    deltas = []
    n_single_class = 0
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([imap[gv] for gv in sampled])
        a_o = auroc(s_o[rows], y[rows]); a_oh = auroc(s_oh[rows], y[rows])
        if a_o != a_o or a_oh != a_oh:
            n_single_class += 1
            continue
        deltas.append(a_oh - a_o)
    d = np.asarray(deltas)
    if d.size == 0:
        raise RuntimeError("paired bootstrap: zero valid rounds (every resample single-class); "
                           "样本/正例过少 → 停")
    if d.size < min_valid_frac * n_boot:
        raise RuntimeError(f"paired bootstrap: only {d.size}/{n_boot} valid rounds "
                           f"(<{min_valid_frac:.0%}); 正例过少,CI 不可信 → 停")
    lo, hi = _percentile_ci(d)
    point = auroc(s_oh, y) - auroc(s_o, y)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "n_boot": int(n_boot), "n_valid": int(d.size),
            "n_discarded_single_class": int(n_single_class)}


def independent_grouped_bootstrap_D(task_mcq: dict, task_h1: dict,
                                    n_boot: int = 1000, seed: int = 0,
                                    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC) -> dict:
    """跨任务 D = S_MCQ − S_H1，independent grouped bootstrap（两任务各自独立重采样）。

    每个 task = {"y":…, "score":…, "groups":…}（score 为该任务 O 的 OOF 风险分）。
    S_t = max(0, 2·AUROC(O_t) − 1)。**非 paired**——两任务非同批 prompt。

    任一任务的 bootstrap sample 单类 → AUROC 未定义 → **丢弃该轮**（不是当 S=0,
    否则正例少时会把"算不出"计成"完全不可观测",系统性压低 S、扭曲 CI）。
    """
    def prep(t):
        return (np.asarray(t["y"]), np.asarray(t["score"], float), np.asarray(t["groups"]))
    ym, sm, gm = prep(task_mcq); yh, sh, gh = prep(task_h1)
    um, uh = np.unique(gm), np.unique(gh)
    im, ih = _group_index_map(gm), _group_index_map(gh)
    rng = np.random.default_rng(seed)
    ds = []
    n_single_class = 0
    for _ in range(n_boot):
        rm = np.concatenate([im[gv] for gv in rng.choice(um, size=len(um), replace=True)])
        rh = np.concatenate([ih[gv] for gv in rng.choice(uh, size=len(uh), replace=True)])
        a_m = auroc(sm[rm], ym[rm]); a_h = auroc(sh[rh], yh[rh])
        if a_m != a_m or a_h != a_h:
            n_single_class += 1
            continue
        ds.append(s_observability(a_m) - s_observability(a_h))
    d = np.asarray(ds)
    if d.size == 0:
        raise RuntimeError("independent bootstrap: zero valid rounds (single-class resamples)")
    if d.size < min_valid_frac * n_boot:
        raise RuntimeError(f"independent bootstrap: only {d.size}/{n_boot} valid rounds "
                           f"(<{min_valid_frac:.0%}); 停")
    lo, hi = _percentile_ci(d)
    point = s_observability(auroc(sm, ym)) - s_observability(auroc(sh, yh))
    if lo > DELTA_RANK_BISERIAL:
        verdict = "h1_high_confidence"
    elif hi <= 0:
        verdict = "outcome_3"
    else:
        verdict = "uncertain"
    return {"D_point": float(point), "ci_lo": lo, "ci_hi": hi,
            "delta": DELTA_RANK_BISERIAL, "verdict": verdict,
            "n_boot": int(n_boot), "n_valid": int(d.size),
            "n_discarded_single_class": int(n_single_class)}


def artifact_redline(delta_lo: float, delta_hi: float, eps: float = EPS_AUROC) -> bool:
    """Δ_artifact = AUROC(full-text) − AUROC(shortcut)；CI ⊂ [−ε,+ε] → shortcut ≈ full-text → 触发。"""
    return (delta_lo >= -eps) and (delta_hi <= eps)


# ----------------------------------------------------------------- RQ2 strata
def rq2_threshold(id_logprob_train: Sequence[float]) -> float:
    """固定分位数（中位数），只在 train fold 拟合（§5）。"""
    return float(np.median(np.asarray(id_logprob_train, float)))


def rq2_strata(id_logprob: Sequence[float], threshold: float) -> np.ndarray:
    """≥ 阈值 → high-confidence(True)，< → low-confidence(False)。"""
    return np.asarray(id_logprob, float) >= threshold


def rq2_fold_strata(id_logprob: Sequence[float],
                    folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    """fold-specific 分层（评审 P0:修 test leakage）。

    §5 要求阈值在**每个 outer fold 的 train 部分**取中位数,再应用到该 fold 的 test。
    在全体样本上取一次中位数 = 用了 test 的信息 → 违反冻结协议。
    每个样本的 stratum 只由它自己所属 fold 的 train 阈值决定。
    """
    lp = np.asarray(id_logprob, float)
    if not np.all(np.isfinite(lp)):
        raise ValueError("rq2_fold_strata: id_logprob contains non-finite values")
    strata = np.full(lp.size, -1, dtype=int)
    thresholds: list[dict] = []
    for k, (train, test) in enumerate(folds):
        thr = rq2_threshold(lp[train])
        strata[test] = (lp[test] >= thr).astype(int)
        thresholds.append({"fold": k, "threshold": float(thr), "n_train": int(train.size),
                           "n_test": int(test.size)})
    if np.any(strata < 0):
        raise RuntimeError(f"rq2_fold_strata: {int((strata < 0).sum())} samples not covered by any test fold")
    return {"strata": strata.astype(bool), "fold_thresholds": thresholds}


def rq2_cell_ok(y_cell: Sequence[int]) -> bool:
    y = np.asarray(y_cell)
    return int((y == 1).sum()) >= RQ2_MIN_PER_CELL and int((y == 0).sum()) >= RQ2_MIN_PER_CELL


def rq2_power_flag(total_positives: int) -> str:
    return "exploratory_low_power" if total_positives < RQ2_LOWPOWER_TOTAL_POS else "adequate"


# ------------------------------------------------------- ladder rung assembly
F5_FEATURE_NAMES = (
    "f5_id_logprob_mean", "f5_id_logprob_min", "f5_id_token_entropy_mean",
    "f5_first_id_token_rank", "f5_id_token_count", "f5_completion_perplexity",
    "f5_lengthnorm_logprob", "f5_id_token_ratio", "f5_completion_token_count",
    "f5_self_consistency_exact", "f5_id_agreement_rate",
    "f5_confidence_high", "f5_confidence_medium", "f5_confidence_low",
)
SURFACE_FEATURE_NAMES = (
    "surface_completion_chars", "surface_completion_lines", "surface_num_mentions",
    "surface_num_attack", "surface_num_cwe", "surface_num_cve",
    "surface_has_subtechnique", "surface_looks_like_list", "surface_first_id_char_pos",
)


def id_string_text(mentions: Sequence[dict]) -> str:
    """ladder rung 2: 只有 identifier 字符串本身（shortcut baseline）。"""
    return " ".join(str(m.get("normalized", "")) for m in mentions if _is_emitted(m))


def surface_format_features(completion: str, mentions: Sequence[dict]) -> dict[str, float]:
    """ladder rung 3: 长度 / 模板 / ontology 前缀等表面格式（shortcut baseline，§7）。

    刻意不含任何 logprob 或 hidden 信息:它存在的意义是,如果它就能解掉任务,
    说明该任务被字符串/格式模式解决（artifact 红线）。
    """
    text = completion or ""
    emitted = [m for m in mentions if _is_emitted(m)]
    starts = [int(m["start"]) for m in emitted if m.get("start") is not None]
    return {
        "surface_completion_chars": float(len(text)),
        "surface_completion_lines": float(len(text.splitlines())),
        "surface_num_mentions": float(len(emitted)),
        "surface_num_attack": float(sum(m.get("family") == "attack" for m in emitted)),
        "surface_num_cwe": float(sum(m.get("family") == "cwe" for m in emitted)),
        "surface_num_cve": float(sum(m.get("family") == "cve" for m in emitted)),
        "surface_has_subtechnique": float(any(m.get("granularity") == "subtechnique" for m in emitted)),
        "surface_looks_like_list": float(sum(1 for ln in text.splitlines()
                                             if ln.strip().startswith(("-", "*"))) >= 3),
        "surface_first_id_char_pos": float(min(starts)) if starts else -1.0,
    }


def _dense_matrix(rows: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(rows), len(names)), dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            v = row.get(name)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            out[i, j] = fv if np.isfinite(fv) else 0.0
    return out


def build_ladder_specs(records: Sequence[dict], hidden: np.ndarray | None = None) -> dict[str, dict]:
    """§7 阶梯 + §8 H 的特征 spec（每个 spec = {"text":…, "dense":…}）。

    rung 1..6 严格按预注册顺序;O 恒 = F5 + full-response-text（不取 max、不按 test 选）。
    H alone 与 O+H 用**同一** block 协议,防"无增量"是融合 artifact（§7 明确要求
    同时报告 AUROC(O) / AUROC(H alone) / AUROC(O+H)）。
    """
    f5 = _dense_matrix(records, F5_FEATURE_NAMES)
    surface = _dense_matrix(records, SURFACE_FEATURE_NAMES)
    prompt_text = [str(r.get("prompt_text", "")) for r in records]
    resp_text = [str(r.get("completion", "")) for r in records]
    id_text = [str(r.get("id_string_text", "")) for r in records]

    specs: dict[str, dict] = {
        "rung1_prompt_only": {"text": prompt_text, "dense": None},
        "rung2_id_string_only": {"text": id_text, "dense": None},
        "rung3_surface_only": {"text": None, "dense": surface},
        "rung4_full_text": {"text": resp_text, "dense": None},
        "rung5_f5": {"text": None, "dense": f5},
        "rung6_O_f5_plus_text": {"text": resp_text, "dense": f5},
    }
    if hidden is not None:
        h = np.asarray(hidden, dtype=float)
        if h.shape[0] != len(records):
            raise ValueError(f"hidden rows {h.shape[0]} != records {len(records)}")
        specs["H_alone"] = {"text": None, "dense": h}
        specs["O_plus_H"] = {"text": resp_text, "dense": np.hstack([f5, h])}
    return specs


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
