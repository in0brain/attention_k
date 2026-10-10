"""v3 bootstrap：抽样计划共享、合并分布、两个方向的同时界、jackknife MC 诊断。

同时界用**预先构造的非对称 fixture** 逐位验算具体数值 ——
不把 `c_α^L ≠ c_α^U` 当作普遍数学性质断言（它对某些数据可以相等）。
只用合成数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention.evaluation import v3_bootstrap as vb  # noqa: E402
from recover_attention.evaluation import v3_repeat_estimand as re_  # noqa: E402
from recover_attention.evaluation.v3_config import (  # noqa: E402
    T_QUANTILE_975_DF19,
    V3AnalysisConfig,
)

from test_v3_stacking import cfg_small, synth  # noqa: E402


# ─────────────────────────────────── 1. resample plan：结构性审计
def test_plan_length_and_batch_assignment():
    groups = np.repeat(np.arange(12), 2)
    cfg = cfg_small(b_batch=4, b_per_batch=5)
    plan = vb.build_resample_plan(groups, cfg)
    assert len(plan) == plan.b_total == cfg.b_batch * cfg.b_per_batch == 20
    seen = [(q, j) for q, j, _rows in plan.iter_draws()]
    assert [q for q, _ in seen] == list(range(20))
    assert [j for _, j in seen] == sorted([j for _, j in seen])
    assert len(plan.chain_digest) == cfg.b_batch


def test_all_repeats_share_the_same_resample_rows_for_a_given_draw():
    """M7：同一个 q 下 R 个 repeat 必须用**同一个** I_q。

    `iter_draws()` 每个 q 只产出一份 rows，由调用方交给全部 repeat —— 结构上
    不可能各自重采样。行索引用完即弃，不驻留（P0-1）。
    """
    groups = np.repeat(np.arange(10), 2)
    cfg = cfg_small(b_batch=2, b_per_batch=4)
    plan = vb.build_resample_plan(groups, cfg)
    assert not hasattr(plan, "rows"), "plan 不得再物化行索引"
    n_seen = 0
    for q, j, rows in plan.iter_draws():
        assert isinstance(rows, np.ndarray) and rows.size == groups.size
        n_seen += 1
    assert n_seen == len(plan)
    a = plan.audit()
    assert set(a["chain_digest"]) == {"0", "1"}
    assert all(len(v) <= plan.n_audit for v in a["audit_digests"].values())


def test_common_random_numbers_prefix_across_grid():
    """§7.3：较小的 B_per_batch 必须是 B_max 档 RNG 流的**前缀**。"""
    groups = np.repeat(np.arange(8), 2)
    cfg = cfg_small(b_batch=3, b_per_batch=2)
    small = vb.build_resample_plan(groups, cfg, b_per_batch=2, b_max=6)
    big = vb.build_resample_plan(groups, cfg, b_per_batch=6, b_max=6)
    s_rows = {}
    for q, j, rows in small.iter_draws():
        s_rows.setdefault(j, []).append(rows.copy())
    b_rows = {}
    for q, j, rows in big.iter_draws():
        b_rows.setdefault(j, []).append(rows.copy())
    for j in range(cfg.b_batch):
        assert len(s_rows[j]) == 2 and len(b_rows[j]) == 6
        for a, b in zip(s_rows[j], b_rows[j]):
            assert np.array_equal(a, b), f"batch {j} 不是前缀 → 网格各档未共享随机数"


def test_plan_rejects_b_max_smaller_than_b_per_batch():
    with pytest.raises(ValueError, match="b_max"):
        vb.build_resample_plan(np.arange(6), cfg_small(), b_per_batch=10, b_max=5)


# ─────────────────────────────────── 2. 合并分布：单一 CI
def test_pooled_ci_comes_from_one_distribution_not_averaged_cis():
    """主 CI 必须来自合并分布；mean(w_j) 与 pooled width 是不同的量。"""
    rng = np.random.default_rng(0)
    v = rng.normal(0.03, 0.01, 400)
    batch_of = np.repeat(np.arange(20), 20)
    pooled = vb.percentile_ci(v)
    bm = vb.batch_means_mc(v, batch_of, 20)
    assert pooled["width"] == pytest.approx(
        np.percentile(v, 97.5) - np.percentile(v, 2.5))
    assert bm["mean_batch_width"] != pytest.approx(pooled["width"], rel=1e-6), \
        "批均宽度与合并宽度恰好相等 → 该对照失去意义"
    assert "NOT the reported width" in bm["note"]


def test_pooled_draws_discards_single_class_rounds():
    y, groups, blocks = synth(seed=6)
    cfg = cfg_small(n_repeats=1, b_batch=2, b_per_batch=3)
    reps = re_.run_all_repeats(y=y, groups=groups, blocks=blocks, cfg=cfg)
    plan = vb.build_resample_plan(groups, cfg)
    d = vb.pooled_draws(y, reps, ["B1"], plan)
    assert d["n_draws"] + d["n_discarded_single_class"] == len(plan)
    assert d["n_draws"] > 0
    assert d["batch_of"].size == d["n_draws"]


# ─────────────────────────────────── 3. 同时界：非对称 fixture 逐位验算
# 手工构造的非对称 fixture：
#   A 右偏且 se 大（15 个近点 + 5 个远的正偏离）
#   B 对称且 se 小
# → T^U 由 A 主导、T^L 由 B 主导，两个方向的临界值实测相差 ~0.79。
ASYMMETRIC = {
    "point": {"A": 0.10, "B": 0.20},
    "draws": {
        "A": 0.10 + np.concatenate([np.linspace(-0.01, 0.01, 15),
                                    np.linspace(0.05, 0.20, 5)]),
        "B": 0.20 + np.linspace(-0.02, 0.02, 20),
    },
}


def test_simultaneous_bounds_match_hand_computed_values():
    """逐位验算 T^L / T^U / c_α / L_k / U_k，而非只断言"不相等"。"""
    point, draws = ASYMMETRIC["point"], ASYMMETRIC["draws"]
    alpha = 0.1
    out = vb.simultaneous_bounds(point, draws, alpha=alpha)

    se = {k: float(np.std(draws[k], ddof=1)) for k in point}
    T_L = np.maximum((point["A"] - draws["A"]) / se["A"],
                     (point["B"] - draws["B"]) / se["B"])
    T_U = np.maximum((draws["A"] - point["A"]) / se["A"],
                     (draws["B"] - point["B"]) / se["B"])
    c_L = float(np.percentile(T_L, 100 * (1 - alpha)))
    c_U = float(np.percentile(T_U, 100 * (1 - alpha)))

    assert out["se"]["A"] == pytest.approx(se["A"])
    assert out["c_alpha_lower"] == pytest.approx(c_L)
    assert out["c_alpha_upper"] == pytest.approx(c_U)
    for k in point:
        assert out["lower"][k] == pytest.approx(point[k] - c_L * se[k])
        assert out["upper"][k] == pytest.approx(point[k] + c_U * se[k])


def test_simultaneous_bounds_pivots_are_not_interchangeable_on_this_fixture():
    """在这份**特意构造的非对称** fixture 上，两个方向的临界值确实不同。

    这是对该 fixture 的数值事实，不是对所有数据成立的数学性质。
    """
    out = vb.simultaneous_bounds(ASYMMETRIC["point"], ASYMMETRIC["draws"], alpha=0.1)
    assert out["c_alpha_lower"] != pytest.approx(out["c_alpha_upper"]), \
        "该 fixture 应当给出不同的上下临界值；若相等则 fixture 失去鉴别力"


def test_simultaneous_bounds_declares_nominal_coverage():
    out = vb.simultaneous_bounds(ASYMMETRIC["point"], ASYMMETRIC["draws"])
    assert "nominal" in out["coverage"]
    assert "NOT a finite-sample guarantee" in out["coverage"]


def test_simultaneous_bounds_drops_degenerate_se():
    point = {"A": 0.1, "B": 0.2}
    draws = {"A": np.linspace(0.05, 0.15, 20), "B": np.full(20, 0.2)}
    out = vb.simultaneous_bounds(point, draws)
    assert out["degenerate_blocks"] == ["B"] and "B" not in out["lower"]


# ─────────────────────────────────── 4. MC 诊断：jackknife primary
def test_jackknife_uses_t_quantile_2093_not_196():
    rng = np.random.default_rng(1)
    v = rng.normal(0.035, 0.004, 400)
    batch_of = np.repeat(np.arange(20), 20)
    j = vb.jackknife_mc(v, batch_of, 20)
    half = (j["mc_band_width"][1] - j["mc_band_width"][0]) / 2
    assert j["t_quantile"] == T_QUANTILE_975_DF19 == 2.093
    assert half == pytest.approx(2.093 * j["se_jack"]["width"])
    assert half != pytest.approx(1.96 * j["se_jack"]["width"])


def test_jackknife_matches_hand_computed_delete_one_batch_variance():
    """逐位验算 delete-one-batch 方差公式。"""
    rng = np.random.default_rng(2)
    v = rng.normal(0.03, 0.005, 200)
    batch_of = np.repeat(np.arange(20), 10)
    j = vb.jackknife_mc(v, batch_of, 20)
    loo = [vb.percentile_ci(v[batch_of != b])["width"] for b in range(20)]
    a = np.asarray(loo)
    want = float(np.sqrt((20 - 1) / 20 * np.sum((a - a.mean()) ** 2)))
    assert j["se_jack"]["width"] == pytest.approx(want)
    assert j["width"] == pytest.approx(vb.percentile_ci(v)["width"])
    assert j["loo_width_mean"] == pytest.approx(float(a.mean()))


def test_band_centre_is_pooled_width_not_loo_mean():
    rng = np.random.default_rng(3)
    v = rng.gamma(2.0, 0.01, 400)
    batch_of = np.repeat(np.arange(20), 20)
    j = vb.jackknife_mc(v, batch_of, 20)
    centre = (j["mc_band_width"][0] + j["mc_band_width"][1]) / 2
    assert centre == pytest.approx(j["width"])


# ─────────────────────────────────── 5. 判读
@pytest.mark.parametrize("band,expect", [
    ([0.030, 0.038], vb.PRECISION_ADEQUATE),
    ([0.041, 0.050], vb.PRECISION_INSUFFICIENT),
    ([0.038, 0.043], vb.PRECISION_BORDERLINE),
    ([0.040, 0.040], vb.PRECISION_BORDERLINE),          # 端点含在带内
])
def test_precision_verdict_three_way(band, expect):
    r = vb.precision_verdict_v3(np.mean(band), band, 0.04)
    assert r["precision"] == expect
    if expect == vb.PRECISION_BORDERLINE:
        assert "must not claim adequate" in r["note"]


def test_certify_non_inferiority_requires_positive_certification():
    """"没有单 block 显著击败 S*" 不构成通过 —— 必须每块的边际下界 > −ε_NI。"""
    rng = np.random.default_rng(4)
    point = {"B1": 0.01, "B2": -0.05}
    draws = {"B1": rng.normal(0.01, 0.004, 500), "B2": rng.normal(-0.05, 0.004, 500)}
    out = vb.certify_non_inferiority(point, draws, eps_ni=0.02)
    assert out["status"] == "NOT CERTIFIED" and out["failed_blocks"] == ["B2"]
    assert out["must_not_write"] == "family-wise certified"
    assert "marginal" in out["primary"]

    ok = vb.certify_non_inferiority(
        {"B1": 0.01}, {"B1": rng.normal(0.01, 0.002, 500)}, eps_ni=0.02)
    assert ok["status"] == "CERTIFIED" and ok["certified"] is True


def test_certify_reports_both_marginal_primary_and_simultaneous_secondary():
    rng = np.random.default_rng(5)
    point = {"B1": 0.01, "B2": 0.005}
    draws = {k: rng.normal(v, 0.003, 400) for k, v in point.items()}
    out = vb.certify_non_inferiority(point, draws, eps_ni=0.02)
    assert set(out["L_marg"]) == {"B1", "B2"} and set(out["L_sim"]) == {"B1", "B2"}
    for k in point:                       # simultaneous 更保守 → 下界不高于边际
        assert out["L_sim"][k] <= out["L_marg"][k] + 1e-9


def test_iva_is_the_sole_authorising_path():
    rng = np.random.default_rng(6)
    point = {"B1": 0.03, "B2": 0.001}
    draws = {k: rng.normal(v, 0.004, 500) for k, v in point.items()}
    out = vb.direct_superiority_iva(point, draws)
    assert out["passed"] is False and out["failed_blocks"] == ["B2"]
    assert "sole authorising path" in out["criterion"]
    assert "no OR of the two paths" in out["note"]


# ─────────────────────────────────── 6. P0-1：流式与显式 rows 参考实现逐位一致
def _reference_materialised_plan(groups, cfg, b_per_batch=None):
    """**测试专用**参考实现：第 1 版的显式 rows 物化写法。

    生产代码已改流式（正式规模下显式物化会 OOM），此处保留仅为证明二者等价。
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    per = int(b_per_batch if b_per_batch is not None else cfg.b_per_batch)
    out = []
    for j in range(cfg.b_batch):
        rng = np.random.default_rng(cfg.batch_seed(j))
        for _ in range(per):
            picked = rng.choice(uniq, size=uniq.size, replace=True)
            out.append((j, np.concatenate([rows_by_group[g] for g in picked])))
    return out


def test_streaming_plan_is_bitwise_identical_to_materialised_reference():
    groups = np.repeat(np.arange(14), 3)
    cfg = cfg_small(b_batch=3, b_per_batch=5)
    ref = _reference_materialised_plan(groups, cfg)
    got = [(j, rows.copy()) for _q, j, rows in vb.build_resample_plan(groups, cfg).iter_draws()]
    assert len(got) == len(ref) == 15
    for (jr, rr), (jg, rg) in zip(ref, got):
        assert jr == jg and np.array_equal(rr, rg)


def test_plan_memory_is_o_n_not_o_n_times_btotal():
    """真实量级合成 benchmark：峰值内存须为 O(n)，不随 B_total 线性增长。

    n≈9270 / groups≈1545（Stage 1 上限量级）。B_total 取秒级可完成的档位，
    并按 B_total=400,000 线性外推 —— **外推值不是实测**，该限制一并记录。
    """
    import tracemalloc

    n_groups, per = 1545, 6
    groups = np.repeat(np.arange(n_groups), per)
    n = groups.size
    assert n == 9270
    cfg = cfg_small(b_batch=4, b_per_batch=50)          # B_total = 200
    plan = vb.build_resample_plan(groups, cfg)

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    peak_extra = 0
    for _q, _j, rows in plan.iter_draws():
        cur, pk = tracemalloc.get_traced_memory()
        peak_extra = max(peak_extra, pk - base)
        del rows
    tracemalloc.stop()

    one_draw_bytes = n * 8
    # 流式：峰值应在"少数几个 draw"量级，取 50 倍 one_draw 作宽松上限
    assert peak_extra < 50 * one_draw_bytes, (
        f"peak {peak_extra/1e6:.1f} MB 远超单 draw 的 {one_draw_bytes/1e6:.2f} MB "
        f"→ 疑似仍在物化 rows")
    materialised_400k = one_draw_bytes * 400_000
    assert materialised_400k > 25e9      # 显式物化在该规模下约 29.7 GB（线性外推，非实测）
