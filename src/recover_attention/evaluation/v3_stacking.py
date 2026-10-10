"""Stage 1 v3：五/六 block nested stacking（M1a）+ shared-block 单次构建（M8）。

三条不可退让的性质，均由 tests/test_v3_stacking.py 以**结构性**证据守住：

1. **shared-block 只构建一次。** 每个 `(repeat, outer_fold)` 只调用一次
   `build_base_block_cache`；返回的**同一个** `BaseBlockCache` 实例同时交给
   S*（五块）与 S*+H（六块）。故"五块 vs 六块"的唯一差别真的只是块数，
   base 层不会各选到不同的 C 而产生漂移。
   注意：**不对底层 `LogisticRegression.fit` 的总次数作任何断言** ——
   nested cross-fitting、inner C 选择、full-train→test 本来就会多次拟合。

2. **fold-local 防泄漏。** 所有 transformer（dense 的 z-score/√d、text 的
   TF-IDF vocabulary/IDF）只在当前 train fold 上拟合。实现上直接复用
   Stage 0 已有单测覆盖的 `conditional_increment.build_fold_design`
   （**只读复用，不修改**），从而与 §7 冻结的 TF-IDF 规格一致。

3. **meta 输入是 `decision_function`。** M8 明写：sklearn 的 `decision_function`
   本身已是 log-odds，不得再做 logit。Stage 0 的 `_risk_scores` 优先返回
   `predict_proba[:,1]`，对 AUROC 等价但对 meta 的输入尺度不等价，故此处
   **单独取 decision_function**，不复用 `_risk_scores`。
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.v3_blocks import (
    BaseBlockCache,
    BlockSpec,
    canonical_names,
    validate_blocks,
)
from recover_attention.evaluation.v3_config import (
    PHASE_INNER_CF,
    PHASE_OUTER_FULL,
    V3AnalysisConfig,
)


# ─────────────────────────────────────────────── fold-local 设计矩阵
def _design(block: BlockSpec, train: np.ndarray, test: np.ndarray):
    """fold-local 设计矩阵。transformer 只见 train fold —— 防泄漏的唯一入口。"""
    if block.kind == "text":
        return ci.build_fold_design(train, test, text=block.text)
    return ci.build_fold_design(train, test, dense_blocks=[(block.name, block.dense)])


def _decision(clf, X) -> np.ndarray:
    """meta 输入必须是 log-odds。**不**走 Stage 0 的 _risk_scores（它优先 predict_proba）。"""
    return np.asarray(clf.decision_function(X), dtype=float).ravel()


# ─────────────────────────────────────────────── C 选择（记录 outer/inner）
def _select_C(block: BlockSpec, rows: np.ndarray, y: np.ndarray, groups: np.ndarray,
              cfg: V3AnalysisConfig, *, repeat: int) -> tuple[float, bool]:
    """在给定训练行集合内用 grouped inner CV 选 C。

    返回 (C, fell_back)。tie-break：升序遍历 + 严格大于 → **并列取最小 C**
    （M8 把 Stage 0 的这一实现行为显式冻结为契约）。
    inner CV 不可行 → 回退 cfg.c_fallback 并置 fell_back=True，由调用方记录审计。

    ⚠️ **split RNG 用 `cfg.fold_seed(repeat)`**（§7.3 冻结的唯一 split-seed 来源），
    **不是** `base_random_state` —— 后者是 LogisticRegression 的模型 RNG（M8）。
    第 1 版混用了两者，导致所有 repeat 的 nested C 选择固定用 seed=0，
    与 §7.3 的 `fold-seed = 20260804 + 100·r` 脱节。本版修正。
    """
    try:
        inner = ci.stratified_grouped_folds(
            y[rows], groups[rows], n_splits=cfg.n_inner_folds,
            seed=cfg.fold_seed(repeat), max_seed_retries=cfg.inner_max_seed_retries)
    except (RuntimeError, ValueError):
        return float(cfg.c_fallback), True

    best_c, best_auc = float(cfg.c_fallback), -np.inf
    for c in cfg.c_grid:                                   # 升序
        aucs = []
        for itr, ite in inner:
            gtr, gte = rows[itr], rows[ite]
            X_tr, X_te, _ = _design(block, gtr, gte)
            clf = LogisticRegression(**cfg.base_lr_kwargs(c))
            clf.fit(X_tr, y[gtr])
            aucs.append(ci.auroc(_decision(clf, X_te), y[gte]))
        m = float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else -np.inf
        if m > best_auc:                                   # 严格大于 → 并列保留先出现者
            best_auc, best_c = m, float(c)
    return best_c, False


def _fit_predict(block: BlockSpec, train: np.ndarray, test: np.ndarray,
                 y: np.ndarray, C: float, cfg: V3AnalysisConfig) -> np.ndarray:
    X_tr, X_te, _ = _design(block, train, test)
    clf = LogisticRegression(**cfg.base_lr_kwargs(C))
    clf.fit(X_tr, y[train])
    return _decision(clf, X_te)


# ─────────────────────────────────────────────── base block cache（**每折一次**）
def build_base_block_cache(*, y: np.ndarray, groups: np.ndarray, blocks: Sequence[BlockSpec],
                           repeat: int, outer_fold: int, train: np.ndarray, test: np.ndarray,
                           cfg: V3AnalysisConfig) -> BaseBlockCache:
    """构建一个 `(repeat, outer_fold)` 的全部 base 层产物。

    **每个 (repeat, outer_fold) 只应调用一次**，其返回值同时供 S* 与 S*+H 使用。
    内部按 nested 协议对每个 block 做多次 `LogisticRegression.fit`（inner 交叉拟合
    + C 选择 + outer-full），这是协议要求，不受"只调用一次"约束。
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    cache = BaseBlockCache(repeat=int(repeat), outer_fold=int(outer_fold),
                           train=np.asarray(train), test=np.asarray(test))

    # inner cross-fitting 的划分：所有 block 共用同一份，保证 meta 训练集行对齐
    try:
        inner_folds = ci.stratified_grouped_folds(
            y[train], groups[train], n_splits=cfg.n_inner_folds,
            seed=cfg.fold_seed(repeat), max_seed_retries=cfg.inner_max_seed_retries)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"repeat={repeat} outer_fold={outer_fold}: cannot build inner cross-fitting "
            f"folds for meta training ({exc}). 不静默降级。") from exc

    for block in blocks:
        # ---- phase = inner_cf：产生 meta 的训练分数（train 侧 OOF）----
        meta_tr = np.full(train.size, np.nan, dtype=float)
        cache.chosen_C_inner[block.name] = {}
        for i, (itr, ite) in enumerate(inner_folds):
            gtr, gte = train[itr], train[ite]
            c_i, fell = _select_C(block, gtr, y, groups, cfg, repeat=repeat)
            if fell:
                cache.record_fallback(block=block.name, phase=PHASE_INNER_CF, inner_fold=i)
            cache.chosen_C_inner[block.name][i] = c_i
            meta_tr[ite] = _fit_predict(block, gtr, gte, y, c_i, cfg)
        if not np.all(np.isfinite(meta_tr)):
            raise RuntimeError(f"{block.name}: inner-OOF meta-train scores not fully covered")
        cache.meta_train_scores[block.name] = meta_tr

        # ---- phase = outer_full：整个 train 拟合 → 给 test 打分 ----
        c_o, fell = _select_C(block, train, y, groups, cfg, repeat=repeat)
        if fell:
            cache.record_fallback(block=block.name, phase=PHASE_OUTER_FULL, inner_fold=None)
        cache.chosen_C_outer[block.name] = c_o
        cache.test_scores[block.name] = _fit_predict(block, train, test, y, c_o, cfg)

    return cache


# ─────────────────────────────────────────────── stack（五块 / 六块，同一规格）
def fit_stack(cache: BaseBlockCache, block_names: Sequence[str], y: np.ndarray,
              cfg: V3AnalysisConfig) -> np.ndarray:
    """在给定 cache 上训 meta 并给 outer-test 打分。

    S* = fit_stack(cache, B1..B5)；S*+H = fit_stack(cache, B1..B5, H)。
    **两者共用同一个 cache**，故 base 层完全一致，唯一差别是 meta 的输入列数。
    """
    cache.require_blocks(block_names)
    Xtr = np.column_stack([cache.meta_train_scores[n] for n in block_names])
    Xte = np.column_stack([cache.test_scores[n] for n in block_names])
    meta = LogisticRegression(**cfg.meta_lr_kwargs())
    meta.fit(Xtr, np.asarray(y)[cache.train])
    return _decision(meta, Xte)


# ─────────────────────────────────────────────── 一个 repeat 的完整 OOF
def run_repeat(*, y, groups, blocks: Sequence[BlockSpec], repeat: int,
               cfg: V3AnalysisConfig,
               cache_builder: Callable[..., BaseBlockCache] = build_base_block_cache) -> dict:
    """跑一个 repeat 的完整交叉拟合，返回各量的 OOF 分数（不做任何跨 repeat 平均）。

    `cache_builder` 可注入，供测试统计**高层** cache 构建的调用次数
    （M8：断言每 (repeat, outer_fold) 恰一次，而非底层 fit 次数）。
    """
    cfg.assert_production()                      # 冻结参数不可绕过（P0-3）
    y = np.asarray(y)
    groups = np.asarray(groups)
    blocks, n_rows = validate_blocks(blocks)     # estimand 结构性锁死（P0-2）
    if n_rows != y.size:
        raise ValueError(f"blocks have {n_rows} rows but y has {y.size}")
    names_s, names_sh = canonical_names()        # canonical 顺序来自冻结常量

    folds = ci.stratified_grouped_folds(y, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(repeat),
                                        max_seed_retries=cfg.outer_max_seed_retries)
    n = y.size
    oof = {"S*": np.full(n, np.nan), "S*+H": np.full(n, np.nan)}
    for b in blocks:
        oof[b.name] = np.full(n, np.nan)
    audits, fallbacks = [], []

    for f, (train, test) in enumerate(folds):
        cache = cache_builder(y=y, groups=groups, blocks=blocks, repeat=repeat,
                              outer_fold=f, train=train, test=test, cfg=cfg)
        oof["S*"][test] = fit_stack(cache, names_s, y, cfg)
        oof["S*+H"][test] = fit_stack(cache, names_sh, y, cfg)
        for name, sc in cache.test_scores.items():
            oof[name][test] = sc
        audits.append(cache.audit())
        fallbacks.extend(cache.c_fallback_events)

    for k, v in oof.items():
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"repeat={repeat}: OOF for {k!r} has non-finite entries")
    return {"repeat": int(repeat), "oof": oof, "fold_audits": audits,
            "c_fallback_events": fallbacks, "block_names_s_star": names_s,
            "block_names_s_star_plus_h": names_sh}
