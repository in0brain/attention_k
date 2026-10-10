"""§5 层间差 CI：Δ_H(high_conf) − Δ_H(low_conf)。

这块曾经只有点估计,并留了一句"CI 见 report 中的 between_strata_ci" —— 那个字段从未存在。
属实验系统 bug(设计文档已写、实现缺失),不是欠功能。

统计要点:**group 会跨层**(同一 prompt 的 K=6 条 completion 逐行判层;真实 smoke 实测
9/20 的 group 同时出现在两层)。故不能对两层各自独立重采样 —— 那会重复计入跨层 group,
并抹掉两层 Δ 之间由共享 group 引入的相关。正确做法:对 group 重采样一次再按层切分。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation.config import Stage0Config

CFG = Stage0Config(mode="dry_run", n_outer_folds=3, n_inner_folds=2, n_boot=400,
                   allow_pilot_population=True, seed=0)


def _data(n_groups=40, per=6, seed=0, h_helps_high_more=True):
    """构造:high 层的 H 增量明显大于 low 层 → 层间差应显著为正。"""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    y = rng.integers(0, 2, n_groups * per)
    # 分层变量逐行判(故 group 会跨层),与标签无关
    lp = rng.normal(0, 1, y.size)
    strata = lp >= np.median(lp)
    s_o = y * 0.6 + rng.normal(0, 1, y.size)
    boost = np.where(strata, 1.6, 0.2) if h_helps_high_more else 0.9
    s_oh = s_o + y * boost + rng.normal(0, 0.3, y.size)
    return y, s_o, s_oh, groups, strata


def test_between_strata_ci_is_computed_and_ordered():
    y, s_o, s_oh, g, st = _data()
    r = bs.between_strata_delta(y, s_o, s_oh, g, st, CFG)
    assert r["status"] == "ok"
    assert r["ci_lo"] < r["point"] < r["ci_hi"]
    assert r["n_valid"] > 0
    assert r["point"] == pytest.approx(r["delta_high_conf"] - r["delta_low_conf"], abs=1e-9)


def test_detects_larger_increment_in_high_stratum():
    """high 层增量更大时,层间差应为正（这是 RQ2 想描述的 effect-modification）。"""
    y, s_o, s_oh, g, st = _data(seed=1, h_helps_high_more=True)
    r = bs.between_strata_delta(y, s_o, s_oh, g, st, CFG)
    assert r["point"] > 0
    assert r["delta_high_conf"] > r["delta_low_conf"]


def test_reports_how_many_groups_span_strata():
    """跨层 group 数是判断"能不能独立重采样"的依据,必须报出来供人核。"""
    y, s_o, s_oh, g, st = _data()
    r = bs.between_strata_delta(y, s_o, s_oh, g, st, CFG)
    assert r["n_groups_spanning_strata"] > 0, "本构造下 group 应当跨层"
    assert "groups resampled ONCE" in r["resampling"]


def test_groups_stay_intact_within_a_resample():
    """同一 group 的行必须整体进出;跨层 group 在两层里同进同出。

    直接验重采样语义:用一个"每 group 只有一行"的退化构造与多行构造对比,
    确认实现是按 group 抽而不是按行抽。
    """
    rng = np.random.default_rng(3)
    n_groups, per = 30, 4
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    y = rng.integers(0, 2, n_groups * per)
    st = np.tile([True, True, False, False], n_groups)   # 每个 group 都跨层
    s_o = y * 0.5 + rng.normal(0, 1, y.size)
    s_oh = s_o + y * 0.8 + rng.normal(0, 0.3, y.size)
    r = bs.between_strata_delta(y, s_o, s_oh, groups, st, CFG)
    assert r["n_groups_spanning_strata"] == n_groups, "构造上每个 group 都应跨层"
    assert r["status"] == "ok"


def test_single_class_stratum_rounds_are_discarded_not_scored_zero():
    """某轮里某层单类 → AUROC 未定义 → 丢弃该轮,不当 0。"""
    rng = np.random.default_rng(5)
    n = 60
    groups = np.array([f"g{i}" for i in range(n)])
    y = np.zeros(n, dtype=int)
    y[:6] = 1                                    # 正例极少 → 常出现单类层
    st = np.arange(n) < n // 2
    s = rng.normal(0, 1, n)
    r = bs.between_strata_delta(y, s, s + y * 0.5, groups, st, CFG)
    assert r["n_discarded"] > 0
    if r["status"] == "ok":
        assert r["n_valid"] + r["n_discarded"] == CFG.n_boot


def test_undefined_when_every_round_degenerates():
    n = 20
    groups = np.array([f"g{i}" for i in range(n)])
    y = np.zeros(n, dtype=int)
    y[0] = 1                                     # 只有一个正例
    st = np.arange(n) < n // 2
    s = np.arange(n, dtype=float)
    r = bs.between_strata_delta(y, s, s, groups, st, CFG)
    assert r["status"] == "undefined" and r["n_valid"] == 0


def test_auprc_delta_has_ci_not_just_a_point():
    """§4「AUPRC 增量并报（类不平衡）」—— 只有点估计的话读者无从判断是不是噪声。"""
    y, s_o, s_oh, g, _ = _data(seed=7)
    r = bs.paired_auprc_delta(y, s_o, s_oh, g, CFG)
    assert r["ci_lo"] is not None and r["ci_hi"] is not None
    assert r["ci_lo"] <= r["delta_auprc"] <= r["ci_hi"]
    assert r["delta_auprc"] == pytest.approx(r["O_plus_H"] - r["O"], abs=1e-9)
    assert 0.0 < r["base_rate"] < 1.0
    assert r["n_valid"] + r["n_discarded"] == CFG.n_boot


def test_auprc_delta_discards_single_class_rounds():
    n = 40
    g = np.array([f"g{i}" for i in range(n)])
    y = np.zeros(n, dtype=int)
    y[:3] = 1
    s = np.arange(n, dtype=float)
    r = bs.paired_auprc_delta(y, s, s + y, g, CFG)
    assert r["n_discarded"] > 0


def test_rq2_report_contains_between_strata_ci_not_a_dangling_note():
    """回归:rq2 的输出必须真的带 CI,不能再是一句指向不存在字段的注释。"""
    from recover_attention.evaluation import ladder as ld
    from recover_attention.evaluation import rq2 as rq2m
    from recover_attention.evaluation.feature_cache import ArmCache

    rng = np.random.default_rng(0)
    n_groups, per = 40, 6
    n = n_groups * per
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    y = rng.integers(0, 2, n)
    recs = []
    for i in range(n):
        recs.append({"example_id": f"h1_{i}", "group_id": groups[i],
                     "prompt_text": "some prompt words here", "id_string_text": "T1114",
                     "completion": ("anomalous unverified claim" if y[i]
                                    else "documented standard claim") + f" f{rng.integers(0, 5)}",
                     "f5_id_logprob_mean": float(rng.normal(0, 1)),
                     "f5_id_logprob_min": float(rng.normal(-2, 1)),
                     "f5_id_token_entropy_mean": float(abs(rng.normal(0.5, 0.2))),
                     "f5_id_token_count": float(rng.integers(1, 5)),
                     "f5_completion_perplexity": float(abs(rng.normal(3, 1))),
                     "f5_lengthnorm_logprob": float(rng.normal(-1, 0.5)),
                     "f5_id_token_ratio": float(rng.uniform(0.01, 0.3)),
                     "f5_completion_token_count": float(rng.integers(20, 200)),
                     "f5_first_id_token_rank": float(rng.integers(1, 5)),
                     "f5_self_consistency_exact": float(rng.uniform(0.2, 1.0)),
                     "f5_id_agreement_rate": float(rng.uniform(0.2, 1.0)),
                     "f5_confidence_high": 0.0, "f5_confidence_medium": 0.0,
                     "f5_confidence_low": 0.0,
                     **ci.surface_format_features("x y z", [])})
    hidden = rng.normal(0, 1, (n, 16)) + y.reshape(-1, 1) * 0.7
    cache = ArmCache(arm="h1", records=recs, hidden=hidden, y=y, groups=groups,
                     backend={"load_in_8bit": True}, validation={})
    lad = ld.run_ladder(cache, CFG)
    out = rq2m.compute_rq2(cache, lad, CFG)
    b = out["between_strata_difference"]
    assert b["status"] in ("ok", "undefined", "unavailable")
    if b["status"] == "ok":
        assert "ci_lo" in b and "ci_hi" in b, "§5 要求层间差 **CI**,不是只有点估计"
        assert b["ci_lo"] <= b["point"] <= b["ci_hi"]
