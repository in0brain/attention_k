"""v3 repeat-level estimand（M7）：逐 repeat 算差再平均，禁 ensemble estimand。

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

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import v3_repeat_estimand as re_  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

from test_v3_stacking import cfg_small, synth  # noqa: E402


def _fake_repeats(y, per_repeat_scores):
    """构造 R 份人工 OOF，用于精确验算平均口径。"""
    out = []
    for r, (s, sh, b1) in enumerate(per_repeat_scores):
        out.append({"repeat": r,
                    "oof": {"S*": np.asarray(s, float), "S*+H": np.asarray(sh, float),
                            "B1": np.asarray(b1, float)}})
    return out


def test_delta_h_is_mean_of_per_repeat_auroc_differences():
    """Δ̂_H = (1/R)·Σ_r [AUROC_r(S*+H) − AUROC_r(S*)]，逐位验算。"""
    y = np.array([0, 0, 1, 1, 0, 1])
    rep = _fake_repeats(y, [
        ([0.1, 0.2, 0.7, 0.8, 0.15, 0.9], [0.1, 0.05, 0.75, 0.85, 0.2, 0.95], [0.5] * 6),
        ([0.9, 0.8, 0.3, 0.2, 0.85, 0.1], [0.2, 0.3, 0.6, 0.7, 0.25, 0.8], [0.4] * 6),
    ])
    got = re_.point_estimates(y, rep, ["B1"])
    want = np.mean([
        ci.auroc(r["oof"]["S*+H"], y) - ci.auroc(r["oof"]["S*"], y) for r in rep
    ])
    assert got[re_.DELTA_H] == pytest.approx(want)
    assert got["estimand"] == "repeat_averaged_statistic"


def test_primary_differs_from_ensemble_estimand_on_constructed_data():
    """M7 的核心纪律：先平均分数再算差 **≠** 逐 repeat 算差再平均。

    构造两份方向相反的 OOF，使二者显著不同；断言 primary 取前者。
    """
    # 穷举搜到的构造：per-repeat 差为 +0.3333 / −0.1667 → primary = +0.0833，
    # 而先平均分数再算差 = +0.6111。两者相差 0.53，鉴别力充足。
    y = np.array([0, 0, 0, 1, 1, 1])
    rep = _fake_repeats(y, [
        ([1.0, 0.6, 0.8, 0.0, 0.0, 1.0], [0.2, 0.6, 1.0, 1.0, 1.0, 0.2], [0.0] * 6),
        ([0.2, 0.8, 1.0, 0.6, 1.0, 0.2], [0.6, 0.6, 0.0, 0.4, 0.4, 0.0], [0.0] * 6),
    ])
    primary = re_.point_estimates(y, rep, ["B1"])[re_.DELTA_H]
    ens = re_.ensemble_delta_h(y, rep)
    assert ens["estimand"] == "ensemble"
    assert "APPENDIX ONLY" in ens["warning"]
    assert primary != pytest.approx(ens[re_.DELTA_H]), \
        "构造数据下两种口径应不同；若相等则本测试失去鉴别力"


def test_ensemble_helper_is_labelled_and_never_returned_as_primary():
    y = np.array([0, 1, 0, 1])
    rep = _fake_repeats(y, [([0.1, 0.9, 0.2, 0.8], [0.1, 0.95, 0.15, 0.85], [0.3] * 4)])
    out = re_.point_estimates(y, rep, ["B1"])
    assert "ensemble" not in out["estimand"]
    assert re_.ensemble_delta_h(y, rep)["estimand"] == "ensemble"


def test_delta_k_and_direct_k_use_the_same_repeat_level_pipeline():
    """M1b 的 Δ_k 与 iv-a 的 Δ^direct_k 必须与 Δ_H 同口径（逐 repeat → 平均）。"""
    y = np.array([0, 0, 1, 1, 1, 0])
    rep = _fake_repeats(y, [
        ([0.1, 0.2, 0.8, 0.7, 0.9, 0.05], [0.1, 0.15, 0.85, 0.75, 0.95, 0.02],
         [0.3, 0.4, 0.6, 0.5, 0.7, 0.2]),
        ([0.2, 0.1, 0.7, 0.8, 0.85, 0.15], [0.15, 0.1, 0.9, 0.8, 0.9, 0.05],
         [0.35, 0.3, 0.55, 0.6, 0.65, 0.25]),
    ])
    out = re_.point_estimates(y, rep, ["B1"])
    want_k = np.mean([ci.auroc(r["oof"]["S*"], y) - ci.auroc(r["oof"]["B1"], y) for r in rep])
    want_d = np.mean([ci.auroc(r["oof"]["S*+H"], y) - ci.auroc(r["oof"]["B1"], y) for r in rep])
    assert out[re_.DELTA_K]["B1"] == pytest.approx(want_k)
    assert out[re_.DELTA_DIRECT_K]["B1"] == pytest.approx(want_d)


def test_repeat_statistics_returns_nan_on_single_class_subset():
    """单类子集 → AUROC 未定义 → NaN（由 bootstrap 层丢弃该轮，不当 0）。"""
    y = np.array([0, 0, 1, 1])
    rep = _fake_repeats(y, [([0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 0.85, 0.95], [0.5] * 4)])
    s = re_.repeat_statistics(y, rep[0]["oof"], ["B1"], rows=np.array([0, 1]))
    assert np.isnan(s[re_.DELTA_H])


def test_average_across_repeats_propagates_nan():
    y = np.array([0, 1, 0, 1])
    a = {re_.DELTA_H: 0.1, re_.DELTA_K: {"B1": 0.2}, re_.DELTA_DIRECT_K: {"B1": 0.3},
         "auroc_S*": 0.6, "auroc_S*+H": 0.7, "auroc_block": {"B1": 0.5}}
    b = dict(a, **{re_.DELTA_H: float("nan")})
    out = re_.average_across_repeats([a, b], ["B1"])
    assert np.isnan(out[re_.DELTA_H])
    assert out[re_.DELTA_K]["B1"] == pytest.approx(0.2)


def test_fold_seeds_are_prefix_nested_across_R():
    """R=3 用的必须是 R=5 的前 3 个 fold-seed（§7.3 冻结）。"""
    cfg = V3AnalysisConfig()
    assert [cfg.fold_seed(r) for r in range(3)] == [cfg.fold_seed(r) for r in range(5)][:3]


def test_run_all_repeats_keeps_r_independent_oof_sets():
    """必须保留 R 份彼此独立的 OOF —— 不得在此处就平均掉。"""
    y, groups, blocks = synth(seed=5)
    cfg = cfg_small(n_repeats=2)
    reps = re_.run_all_repeats(y=y, groups=groups, blocks=blocks, cfg=cfg)
    assert len(reps) == 2
    assert reps[0]["oof"]["S*"].shape == reps[1]["oof"]["S*"].shape == y.shape
    assert not np.allclose(reps[0]["oof"]["S*"], reps[1]["oof"]["S*"]), \
        "两个 repeat 用了不同 fold-seed，OOF 不应完全相同"
