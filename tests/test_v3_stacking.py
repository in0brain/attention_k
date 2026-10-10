"""v3 nested stacking：shared-block 复用（结构性）与 fold-local 防泄漏。

只用合成数据，不读任何真实 feature cache。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import v3_stacking as vs  # noqa: E402
from recover_attention.evaluation.v3_blocks import (  # noqa: E402
    BaseBlockCache,
    BlockSpec,
    validate_blocks as vb_validate,
)
from recover_attention.evaluation.v3_config import (  # noqa: E402
    BLOCK_NAMES_S_STAR_PLUS_H,
    PHASE_INNER_CF,
    PHASE_OUTER_FULL,
    V3AnalysisConfig,
)


# ─────────────────────────────────────────── 合成数据
def synth(n_groups=48, per=4, seed=0, d=3):
    """合成数据规模须撑得住 3×3 nested CV。

    实测 n_groups=30 / 正类率 1/3 时，某些 outer 划分只剩 2 个正 group，
    inner 3 折无法构造 → 实现按 M8 抛错（正确行为），但会误伤测试。
    故取 48 组、正类率 1/2。
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = np.array([1 if g % 2 == 0 else 0 for g in groups], dtype=int)
    n = y.size
    blocks = [
        BlockSpec("B1", "dense", dense=(y * 1.1 + rng.normal(0, 1, (n, d)).T).T),
        BlockSpec("B2", "dense", dense=rng.normal(0, 1, (n, d))),
        BlockSpec("B3", "text",
                  text=[("fabricated bogus id" if v else "grounded real id")
                        + f" filler{rng.integers(0, 4)}" for v in y]),
        BlockSpec("B4", "dense", dense=(y * 0.8 + rng.normal(0, 1, (n, d)).T).T),
        BlockSpec("B5", "dense", dense=rng.normal(0, 1, (n, 2))),
        BlockSpec("H", "dense", dense=(y * 1.3 + rng.normal(0, 1, (n, d)).T).T),
    ]
    return y, groups, blocks


def cfg_small(**over):
    base = dict(n_outer_folds=3, n_inner_folds=3, mode="dry_run", n_repeats=2,
                b_per_batch=40, b_batch=4)
    base.update(over)
    return V3AnalysisConfig(**base)


def cfg_production(**over):
    """合法的 production 配置。

    §7.3 收敛研究已于 2026-08-07 完成（R=3, B_per_batch=1000），
    故 R/B/r_b_source/fallback_declared 现已进 FROZEN_FIELDS，默认值即合法值。
    """
    base = dict(mode="production")
    base.update(over)
    return V3AnalysisConfig(**base)


# ─────────────────────────────────────────── 1. shared-block：结构性证据
def test_base_block_cache_built_exactly_once_per_repeat_and_outer_fold():
    """M8 P0：每个 (repeat, outer_fold) 只调用一次**高层** cache builder。

    刻意**不**断言底层 LogisticRegression.fit 的总次数 —— nested cross-fitting、
    inner C 选择与 outer-full 拟合本来就会多次拟合，写死次数会误杀正确实现。
    """
    y, groups, blocks = synth()
    cfg = cfg_small()
    calls: list[tuple[int, int]] = []

    def counting_builder(**kw):
        calls.append((kw["repeat"], kw["outer_fold"]))
        return vs.build_base_block_cache(**kw)

    vs.run_repeat(y=y, groups=groups, blocks=blocks, repeat=0, cfg=cfg,
                  cache_builder=counting_builder)
    assert len(calls) == cfg.n_outer_folds, calls
    assert sorted(calls) == [(0, f) for f in range(cfg.n_outer_folds)]
    assert len(set(calls)) == len(calls), "同一 (repeat, outer_fold) 被构建了多次"


def test_s_star_and_s_star_plus_h_receive_the_same_cache_instance():
    """S* 与 S*+H 必须收到**同一个** BaseBlockCache 实例（id 相同）。"""
    y, groups, blocks = synth()
    cfg = cfg_small()
    seen: list[int] = []
    real_fit_stack = vs.fit_stack

    def spy_fit_stack(cache, names, y_, c):
        seen.append(id(cache))
        return real_fit_stack(cache, names, y_, c)

    vs.fit_stack = spy_fit_stack
    try:
        vs.run_repeat(y=y, groups=groups, blocks=blocks, repeat=0, cfg=cfg)
    finally:
        vs.fit_stack = real_fit_stack

    # 每折两次 fit_stack（S* 与 S*+H），两者的 cache id 必须相同
    assert len(seen) == 2 * cfg.n_outer_folds
    for i in range(0, len(seen), 2):
        assert seen[i] == seen[i + 1], "S* 与 S*+H 收到了不同的 BaseBlockCache 实例"


def test_fault_injection_separate_caches_doubles_builder_calls():
    """故障注入：若为 S* 与 S*+H 各建一份 cache，builder 调用数翻倍 → 上面的断言失败。"""
    y, groups, blocks = synth()
    cfg = cfg_small()
    calls = []

    def counting_builder(**kw):
        calls.append((kw["repeat"], kw["outer_fold"]))
        return vs.build_base_block_cache(**kw)

    # 模拟错误实现：每折构建两次
    folds = ci.stratified_grouped_folds(y, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(0))
    for f, (tr, te) in enumerate(folds):
        c1 = counting_builder(y=y, groups=groups, blocks=blocks, repeat=0,
                              outer_fold=f, train=tr, test=te, cfg=cfg)
        c2 = counting_builder(y=y, groups=groups, blocks=blocks, repeat=0,
                              outer_fold=f, train=tr, test=te, cfg=cfg)
        assert id(c1) != id(c2)
    assert len(calls) == 2 * cfg.n_outer_folds
    assert len(set(calls)) != len(calls), "错误实现应当出现重复的 (repeat, outer_fold)"


def test_shared_block_numeric_consistency_is_auxiliary_evidence_only():
    """数值一致性（辅助，非唯一证据）：同一 cache 下前 5 block 的 C 与分数当然相等。"""
    y, groups, blocks = synth()
    cfg = cfg_small()
    tr, te = ci.stratified_grouped_folds(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(0))[0]
    cache = vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=0,
                                      outer_fold=0, train=tr, test=te, cfg=cfg)
    s_names = [b.name for b in blocks if b.name != "H"]
    a = vs.fit_stack(cache, s_names, y, cfg)
    b = vs.fit_stack(cache, [b_.name for b_ in blocks], y, cfg)
    assert a.shape == b.shape == te.shape
    for n in s_names:
        assert n in cache.chosen_C_outer and n in cache.chosen_C_inner


# ─────────────────────────────────────────── 2. chosen_C 结构与回退审计
def test_chosen_C_records_outer_and_per_inner_fold_separately():
    """🆕 不得把 chosen_C 压成 dict[name -> float]：outer-full 与各 inner fold 可能不同。"""
    y, groups, blocks = synth()
    cfg = cfg_small()
    tr, te = ci.stratified_grouped_folds(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(0))[0]
    cache = vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=0,
                                      outer_fold=0, train=tr, test=te, cfg=cfg)
    for b in blocks:
        assert isinstance(cache.chosen_C_outer[b.name], float)
        inner = cache.chosen_C_inner[b.name]
        assert set(inner) == set(range(cfg.n_inner_folds))
        assert all(c in cfg.c_grid for c in inner.values())
        assert cache.chosen_C_outer[b.name] in cfg.c_grid
    a = cache.audit()
    assert set(a["chosen_C_inner"]) == {b.name for b in blocks}


def test_c_fallback_event_records_repeat_fold_block_phase_inner_fold():
    """回退审计必须含五个字段，不能只存一个模糊 fold 索引。"""
    cache = BaseBlockCache(repeat=2, outer_fold=1, train=np.arange(4), test=np.arange(4, 6))
    cache.record_fallback(block="B3", phase=PHASE_INNER_CF, inner_fold=1)
    cache.record_fallback(block="B3", phase=PHASE_OUTER_FULL, inner_fold=None)
    assert cache.c_fallback_events[0] == {"repeat": 2, "outer_fold": 1, "block": "B3",
                                          "phase": PHASE_INNER_CF, "inner_fold": 1}
    assert cache.c_fallback_events[1]["phase"] == PHASE_OUTER_FULL
    assert cache.c_fallback_events[1]["inner_fold"] is None
    assert cache.audit()["n_c_fallback"] == 2


def test_select_C_falls_back_to_one_and_reports_it():
    """inner CV 不可行 → 回退 C=1.0 且 fell_back=True（由调用方记录审计）。"""
    y = np.zeros(8, dtype=int)
    y[:1] = 1                       # 只有一个正例 → grouped inner CV 不可行
    groups = np.arange(8)
    blk = BlockSpec("B1", "dense", dense=np.arange(16, dtype=float).reshape(8, 2))
    c, fell = vs._select_C(blk, np.arange(8), y, groups, cfg_small(), repeat=0)
    assert fell is True and c == 1.0


def test_select_C_tie_break_picks_smallest_C():
    """并列时取**最小** C（M8 把 Stage 0 的实现行为冻结为契约）。"""
    y, groups, blocks = synth(seed=2)
    cfg = cfg_small()
    blk = BlockSpec("const", "dense", dense=np.ones((y.size, 2)))   # 无信息 → 各 C 全并列
    c, fell = vs._select_C(blk, np.arange(y.size), y, groups, cfg, repeat=0)
    assert fell is False
    assert c == min(cfg.c_grid), f"并列时应取最小 C，实际 {c}"


# ─────────────────────────────────────────── 3. fold-local 防泄漏
def _transformer_fitted_on(train_rows, text):
    v = ci.TfidfVectorizer(**ci.WORD_TFIDF_PARAMS)
    v.fit([text[i] for i in train_rows])
    return v


def test_text_tfidf_vocabulary_is_fitted_on_train_fold_only():
    """held-out fold 独占词不得进入 train-side vocabulary。"""
    text = [f"common word alpha{i % 3}" for i in range(12)]
    text[10] = "zzunique heldout token"          # 只在 test 出现
    train, test = np.arange(10), np.arange(10, 12)
    X_tr, X_te, info = ci.build_fold_design(train, test, text=text)
    v = _transformer_fitted_on(train, text)
    assert "zzunique" not in v.vocabulary_
    assert X_tr.shape[1] == X_te.shape[1] == info["text"]["n_features"]


def test_dense_standardisation_uses_train_fold_statistics_only():
    """向 held-out 注入极端值不得改变 train 侧的标准化结果。"""
    rng = np.random.default_rng(0)
    dense = rng.normal(0.0, 1.0, (20, 3))
    train, test = np.arange(15), np.arange(15, 20)
    X_a, _, _ = ci.build_fold_design(train, test, dense_blocks=[("b", dense)])
    poisoned = dense.copy()
    poisoned[test] += 1e6                        # 只污染 held-out
    X_b, _, _ = ci.build_fold_design(train, test, dense_blocks=[("b", poisoned)])
    assert np.allclose(X_a.toarray(), X_b.toarray()), \
        "train 侧设计矩阵随 held-out 变化 → transformer 在全数据上拟合了"


def test_poisoning_heldout_does_not_change_inner_oof_meta_train_scores():
    """端到端：污染 held-out fold 不得改变 inner-OOF 的 meta 训练分数。"""
    y, groups, blocks = synth(seed=1)
    cfg = cfg_small()
    tr, te = ci.stratified_grouped_folds(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(0))[0]
    c_clean = vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=0,
                                        outer_fold=0, train=tr, test=te, cfg=cfg)

    poisoned = []
    for b in blocks:
        if b.kind == "dense":
            d = np.asarray(b.dense).copy()
            d[te] += 1e6
            poisoned.append(BlockSpec(b.name, "dense", dense=d))
        else:
            t = list(b.text)
            for i in te:
                t[i] = "zzpoison " + t[i]
            poisoned.append(BlockSpec(b.name, "text", text=t))
    c_dirty = vs.build_base_block_cache(y=y, groups=groups, blocks=poisoned, repeat=0,
                                        outer_fold=0, train=tr, test=te, cfg=cfg)

    for b in blocks:
        assert np.allclose(c_clean.meta_train_scores[b.name],
                           c_dirty.meta_train_scores[b.name]), \
            f"{b.name}: held-out 污染改变了 inner-OOF 训练分数 → 存在泄漏"
        assert c_clean.chosen_C_outer[b.name] == c_dirty.chosen_C_outer[b.name]


def test_fault_injection_global_fit_transformer_would_leak():
    """故障注入：若在**全数据**上拟合 transformer，held-out 污染会改变 train 侧矩阵。"""
    rng = np.random.default_rng(0)
    dense = rng.normal(0.0, 1.0, (20, 3))
    train, test = np.arange(15), np.arange(15, 20)

    def global_fit(d):                       # 错误实现：用全数据的均值/标准差
        mu, sd = d.mean(axis=0), d.std(axis=0)
        return (d - mu) / np.where(sd < 1e-12, 1.0, sd)

    clean = global_fit(dense)[train]
    poisoned = dense.copy()
    poisoned[test] += 1e6
    dirty = global_fit(poisoned)[train]
    assert not np.allclose(clean, dirty), "全数据拟合应当泄漏 —— 该故障注入本身失效了"


# ─────────────────────────────────────────── 4. meta 输入是 decision_function
def test_meta_input_uses_decision_function_not_probability():
    """M8：meta 输入必须是 log-odds。decision_function 的值域应超出 [0,1]。"""
    y, groups, blocks = synth(seed=3)
    cfg = cfg_small()
    tr, te = ci.stratified_grouped_folds(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(0))[0]
    cache = vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=0,
                                      outer_fold=0, train=tr, test=te, cfg=cfg)
    allv = np.concatenate([cache.meta_train_scores[b.name] for b in blocks])
    assert allv.min() < 0.0, "全为正 → 疑似 predict_proba 而非 decision_function"


def test_run_repeat_produces_finite_oof_for_all_quantities():
    y, groups, blocks = synth(seed=4)
    cfg = cfg_small()
    out = vs.run_repeat(y=y, groups=groups, blocks=blocks, repeat=0, cfg=cfg)
    for key in ["S*", "S*+H"] + [b.name for b in blocks]:
        assert np.all(np.isfinite(out["oof"][key])) and out["oof"][key].size == y.size
    assert len(out["fold_audits"]) == cfg.n_outer_folds


def test_production_config_rejects_smoke_spec():
    """折数不再只查下限：必须**恰为** 5/3（预注册值），多了也不行。"""
    with pytest.raises(ValueError, match="frozen field 'n_outer_folds'"):
        cfg_production(n_outer_folds=3).assert_production()
    with pytest.raises(ValueError, match="frozen field 'n_outer_folds'"):
        cfg_production(n_outer_folds=7).assert_production()   # 第 1 版会放行
    with pytest.raises(ValueError, match="frozen field 'n_inner_folds'"):
        cfg_production(n_inner_folds=2).assert_production()
    cfg_production().assert_production()        # 合法 production 规格


# ─────────────────────────────────────────── 5. P0-2：estimand 结构性锁死
def _named(blocks, keep=None, drop=None, extra=None):
    out = [b for b in blocks if drop is None or b.name != drop]
    if keep is not None:
        out = [b for b in out if b.name in keep]
    if extra is not None:
        out = out + [extra]
    return out


def test_validate_blocks_returns_canonical_order_not_caller_order():
    """canonical 顺序来自冻结常量，不从调用方列表推导。"""
    y, groups, blocks = synth()
    shuffled = list(reversed(blocks))
    ordered, n = vb_validate(shuffled)
    assert [b.name for b in ordered] == list(BLOCK_NAMES_S_STAR_PLUS_H)
    assert n == y.size


def test_missing_H_is_rejected_so_s_star_and_s_star_plus_h_cannot_collapse():
    """缺 H 会让 S* 与 S*+H 变成同一个量 —— 必须硬错误，不能静默。"""
    y, groups, blocks = synth()
    with pytest.raises(ValueError, match=r"missing \['H'\]"):
        vb_validate(_named(blocks, drop="H"))


def test_missing_B5_is_rejected():
    y, groups, blocks = synth()
    with pytest.raises(ValueError, match=r"missing \['B5'\]"):
        vb_validate(_named(blocks, drop="B5"))


def test_extra_block_is_rejected():
    y, groups, blocks = synth()
    extra = BlockSpec("B6", "dense", dense=np.zeros((y.size, 2)))
    with pytest.raises(ValueError, match=r"unexpected \['B6'\]"):
        vb_validate(blocks + [extra])


def test_duplicate_block_name_is_rejected():
    y, groups, blocks = synth()
    with pytest.raises(ValueError, match="duplicate block names"):
        vb_validate(blocks + [blocks[0]])


def test_row_count_mismatch_is_rejected():
    y, groups, blocks = synth()
    bad = [BlockSpec("B1", "dense", dense=np.zeros((y.size - 1, 2)))] + list(blocks[1:])
    with pytest.raises(ValueError, match="row count"):
        vb_validate(bad)


def test_non_finite_dense_is_rejected():
    y, groups, blocks = synth()
    d = np.asarray(blocks[0].dense).copy()
    d[0, 0] = np.nan
    bad = [BlockSpec("B1", "dense", dense=d)] + list(blocks[1:])
    with pytest.raises(ValueError, match="non-finite"):
        vb_validate(bad)


def test_run_repeat_rejects_wrong_block_set():
    y, groups, blocks = synth()
    with pytest.raises(ValueError, match="missing"):
        vs.run_repeat(y=y, groups=groups, blocks=_named(blocks, drop="H"),
                      repeat=0, cfg=cfg_small())


# ─────────────────────────────────────────── 6. P1：split RNG 与模型 RNG 分离
def test_select_C_uses_fold_seed_not_base_random_state():
    """spy：_select_C 内的 splitter seed 必须是 fold_seed(repeat)，不是 base_random_state。"""
    y, groups, blocks = synth()
    cfg = cfg_small()
    seeds = []
    real = ci.stratified_grouped_folds

    def spy(*a, **kw):
        seeds.append(kw.get("seed"))
        return real(*a, **kw)

    vs.ci.stratified_grouped_folds = spy
    try:
        tr, te = real(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(1))[0]
        vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=1,
                                  outer_fold=0, train=tr, test=te, cfg=cfg)
    finally:
        vs.ci.stratified_grouped_folds = real

    assert seeds, "spy 未捕获到 splitter 调用"
    assert all(s == cfg.fold_seed(1) for s in seeds), set(seeds)
    assert cfg.base_random_state not in set(seeds),         "splitter 用了 base_random_state —— 模型 RNG 与 split RNG 又混用了"


def test_split_seed_varies_with_repeat():
    y, groups, blocks = synth()
    cfg = cfg_small()
    got = {}
    real = ci.stratified_grouped_folds
    for r in (0, 1):
        seen = []

        def spy(*a, **kw):
            seen.append(kw.get("seed"))
            return real(*a, **kw)

        vs.ci.stratified_grouped_folds = spy
        try:
            tr, te = real(y, groups, cfg.n_outer_folds, seed=cfg.fold_seed(r))[0]
            vs.build_base_block_cache(y=y, groups=groups, blocks=blocks, repeat=r,
                                      outer_fold=0, train=tr, test=te, cfg=cfg)
        finally:
            vs.ci.stratified_grouped_folds = real
        got[r] = set(seen)
    assert got[0] != got[1], "不同 repeat 的 split seed 必须不同"


# ─────────────────────────────────────────── 7. P0-3：冻结参数不可绕过
@pytest.mark.parametrize("field_name,bad", [
    ("n_outer_folds", 7), ("n_inner_folds", 5), ("base_solver", "lbfgs"),
    ("base_max_iter", 500), ("base_random_state", 7), ("meta_C", 0.5),
    ("meta_solver", "liblinear"), ("meta_max_iter", 100), ("eps_ni", 0.05),
    ("alpha", 0.10), ("b_batch", 10), ("c_fallback", 0.5),
    ("fold_seed_base", 1), ("fold_seed_step", 1), ("boot_seed_step", 3),
    ("outer_max_seed_retries", 5), ("base_class_weight", "balanced"),
])
def test_every_frozen_field_is_checked_exactly(field_name, bad):
    with pytest.raises(ValueError, match="frozen field"):
        cfg_production(**{field_name: bad}).assert_production()


def test_valid_production_config_passes():
    cfg_production().assert_production()


def test_convergence_study_values_are_the_production_defaults():
    """§7.4：R=3 / B_per_batch=1000 已由收敛研究确定并冻结，默认值即为它们。"""
    c = V3AnalysisConfig()
    assert c.n_repeats == 3 and c.b_per_batch == 1000
    assert c.b_batch == 20 and c.b_total == 20000
    assert c.r_b_source == "convergence_study" and c.fallback_declared is False
    c.assert_production()


@pytest.mark.parametrize("field_name,bad", [
    ("n_repeats", 5), ("b_per_batch", 10000),
    ("r_b_source", "unset"), ("fallback_declared", True),
])
def test_r_and_b_are_frozen_after_the_convergence_study(field_name, bad):
    with pytest.raises(ValueError, match="frozen field"):
        cfg_production(**{field_name: bad}).assert_production()


def test_fallback_branch_is_closed_now_that_the_study_completed():
    """回退分支在 §7.3 完成后不再适用，报错须指出收敛研究已完成。"""
    with pytest.raises(ValueError, match="frozen field 'r_b_source'"):
        V3AnalysisConfig(mode="production", r_b_source="fallback",
                         fallback_declared=True).assert_production()


def test_provenance_records_r_b_source():
    p = cfg_production().provenance()
    assert p["r_b_source"] == "convergence_study" and p["fallback_declared"] is False
    assert p["n_repeats"] == 3 and p["b_per_batch"] == 1000 and p["b_total"] == 20000
    assert p["fold_seed_rule"] == "20260804 + 100*r"


# ── 冻结轮（2026-08-08）：F / 两个 salt / Stage 1 population ──────────────────
def test_freeze_round_values_are_the_production_defaults():
    """§M2d / §M6 / §7.5：F=5、两个 salt、recall-only population 已是默认并冻结。"""
    c = V3AnalysisConfig()
    assert c.e3_n_blocks == 5
    assert c.e3_block_salt == "recover_attention/v3/E3_vocab_block/2026-08-08"
    assert c.mcq_selection_salt == "recover_attention/v3/mcq_selection/2026-08-08"
    assert c.h1_routes == ("recall",) and c.h1_families == ("attack", "cwe")
    assert c.n_h1_eligible_pool == 1547 and c.n_h1_prompts == 1544
    assert c.n_mcq_questions == 1760 and c.k_traces == 6
    c.assert_production()


@pytest.mark.parametrize("field_name,bad", [
    ("e3_n_blocks", 10),
    ("e3_block_salt", "recover_attention/v3/E3_vocab_block/2026-08-09"),
    ("mcq_selection_salt", "other-salt"),
    ("h1_routes", ("recall", "open_gen")),
    ("h1_families", ("attack", "cwe", "cve")),
    ("n_h1_prompts", 1547),
    ("n_h1_eligible_pool", 1544),
    ("n_mcq_questions", 2000),
    ("k_traces", 3),
])
def test_freeze_round_fields_are_frozen(field_name, bad):
    """换 salt = 换一次分折；改 population = 换 estimand。两者都必须被拦下。"""
    with pytest.raises(ValueError, match="frozen field"):
        cfg_production(**{field_name: bad}).assert_production()


def test_F_must_equal_n_outer_folds_semantic_guard():
    """F=5 的**理由**是块本身充当 outer fold；同时改两个常量也不许绕过。"""
    with pytest.raises(ValueError, match="必须等于 n_outer_folds"):
        V3AnalysisConfig(mode="dry_run", e3_n_blocks=3, n_outer_folds=5)._check_semantics()


def test_open_gen_is_refused_by_a_named_error():
    with pytest.raises(ValueError, match="recall only"):
        V3AnalysisConfig(mode="dry_run", h1_routes=("recall", "open_gen"))._check_semantics()


def test_h1_traces_identity_is_guarded():
    with pytest.raises(ValueError, match="9,264"):
        V3AnalysisConfig(mode="dry_run", n_h1_prompts=1543)._check_semantics()


def test_provenance_records_the_freeze_round_values():
    p = cfg_production().provenance()
    assert p["e3_n_blocks"] == 5 and p["h1_routes"] == ["recall"]
    assert p["n_h1_prompts"] == 1544 and p["h1_traces"] == 9264
    assert p["n_mcq_questions"] == 1760 and p["mcq_traces"] == 10560


def test_run_repeat_enforces_frozen_config():
    """assert_production 由 run_repeat 强制调用 —— 不可绕过。"""
    y, groups, blocks = synth()
    with pytest.raises(ValueError, match="frozen field"):
        vs.run_repeat(y=y, groups=groups, blocks=blocks, repeat=0,
                      cfg=cfg_production(base_solver="lbfgs"))
