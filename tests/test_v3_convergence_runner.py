"""收敛研究 runner 的**前缀复用等价性**。

核心断言：一次 `R_max × B_max` 扫描 + 前缀导出的各档结果，与逐档朴素
`pooled_draws` 的结果**逐位相等**（含随 R 变化的 single-class 丢弃规则）。

只用合成数据；不读 `outputs/**`。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_bootstrap as vb            # noqa: E402
from recover_attention.evaluation import v3_repeat_estimand as re_     # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig    # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_v3_convergence_study", ROOT / "scripts" / "run_v3_convergence_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


# ───────────────────────────────────────────── 合成 repeats（不跑交叉拟合）
def synth_repeats(n=60, n_repeats=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1 if i % 3 == 0 else 0 for i in range(n)], dtype=int)
    groups = np.repeat(np.arange(n // 3), 3)
    reps = []
    for r in range(n_repeats):
        s = y * (0.5 + 0.1 * r) + rng.normal(0, 1, n)
        sh = y * (0.9 + 0.1 * r) + rng.normal(0, 1, n)
        reps.append({"repeat": r, "oof": {"S*": s, "S*+H": sh}})
    return {"y": y, "groups": groups, "n": n}, reps


def cfg(n_repeats, b_per_batch, b_batch=4):
    return V3AnalysisConfig(n_repeats=n_repeats, b_per_batch=b_per_batch,
                            b_batch=b_batch, mode="dry_run")


# ───────────────────────────────────────────── 1. 前缀复用 ≡ 逐档朴素重算
@pytest.mark.parametrize("r_level,b_level", [(1, 3), (2, 3), (4, 3), (4, 1), (4, 2)])
def test_prefix_derived_cell_equals_naive_pooled_draws(r_level, b_level):
    arm, reps = synth_repeats()
    b_max, b_batch = 3, 4

    # ---- 前缀路径：一次 R_max × B_max 扫描，再取前缀 ----
    plan_max = vb.build_resample_plan(arm["groups"], cfg(len(reps), b_max, b_batch),
                                      b_per_batch=b_max, b_max=b_max)
    M, batch_of, pos = runner.delta_matrix(arm, reps, plan_max, len(reps),
                                           progress_every=0)
    got = runner.cell_from_matrix(M, batch_of, pos, r_level, b_level, b_batch)

    # ---- 朴素路径：为该档单独建 plan 并跑 pooled_draws ----
    plan_cell = vb.build_resample_plan(arm["groups"], cfg(r_level, b_level, b_batch),
                                       b_per_batch=b_level, b_max=b_max)
    draws = vb.pooled_draws(arm["y"], reps[:r_level], runner.DELTA_ONLY, plan_cell)
    want = vb.jackknife_mc(draws[re_.DELTA_H], draws["batch_of"], b_batch)

    assert got["n_draws"] == draws["n_draws"]
    assert got["n_discarded_single_class"] == draws["n_discarded_single_class"]
    for k in ("ci_lo", "ci_hi", "width"):
        assert got[k] == pytest.approx(want[k], abs=0.0, rel=0.0), \
            f"{k}: prefix {got[k]!r} != naive {want[k]!r}"
    assert got["se_jack_width"] == pytest.approx(want["se_jack"]["width"],
                                                 abs=0.0, rel=0.0)


def test_prefix_draws_are_the_same_rows_as_naive_plan():
    """前缀取的必须是同一批 draw（共同随机数），否则等价性只是巧合。"""
    arm, reps = synth_repeats()
    b_max, b_batch, b_level = 5, 3, 2
    big = vb.build_resample_plan(arm["groups"], cfg(1, b_max, b_batch),
                                 b_per_batch=b_max, b_max=b_max)
    small = vb.build_resample_plan(arm["groups"], cfg(1, b_level, b_batch),
                                   b_per_batch=b_level, b_max=b_max)
    per_batch_big: dict[int, list] = {}
    for _q, j, rows in big.iter_draws():
        per_batch_big.setdefault(j, []).append(rows.copy())
    per_batch_small: dict[int, list] = {}
    for _q, j, rows in small.iter_draws():
        per_batch_small.setdefault(j, []).append(rows.copy())
    for j in range(b_batch):
        assert len(per_batch_small[j]) == b_level
        for a, b in zip(per_batch_small[j], per_batch_big[j]):
            assert np.array_equal(a, b)


# ───────────────────────────────────────────── 2. 丢弃规则随 R 变化
def test_discard_set_grows_with_R():
    """某 repeat 在某 draw 上单类 → 该 draw 在含它的所有 R 档被丢弃。

    丢弃集合随 R 单调不减；若实现只算一次丢弃集合，此断言会失败。
    """
    arm, reps = synth_repeats(n_repeats=3)
    b_max, b_batch = 4, 3
    plan = vb.build_resample_plan(arm["groups"], cfg(3, b_max, b_batch),
                                  b_per_batch=b_max, b_max=b_max)
    M, batch_of, pos = runner.delta_matrix(arm, reps, plan, 3, progress_every=0)
    M[2, 0] = np.nan                       # 人为让 repeat 2 在 draw 0 上单类
    n1 = runner.cell_from_matrix(M, batch_of, pos, 1, b_max, b_batch)["n_draws"]
    n3 = runner.cell_from_matrix(M, batch_of, pos, 3, b_max, b_batch)["n_draws"]
    assert n3 == n1 - 1, "R=3 应比 R=1 多丢弃那一条"


# ───────────────────────────────────────────── 3. delta-only 不改 Δ_H
def test_block_names_empty_gives_same_delta_H():
    """block_names=[] 只是不算五个 block 的量，Δ_H 本身必须逐位不变。"""
    arm, reps = synth_repeats()
    oof = dict(reps[0]["oof"])
    oof["B1"] = np.linspace(0, 1, arm["n"])
    full = re_.repeat_statistics(arm["y"], oof, ["B1"])
    lean = re_.repeat_statistics(arm["y"], oof, [])
    assert lean[re_.DELTA_H] == full[re_.DELTA_H]
    assert lean[re_.DELTA_K] == {} and lean[re_.DELTA_DIRECT_K] == {}


# ───────────────────────────────────────────── 4. 选定规则
def test_select_picks_smallest_level_whose_tail_is_within_tol():
    # 档位 3 偏差 0.005 > tol=0.002 → 不合格；5/8/12 的尾部全部合格 → 选 5
    cells = {1: {"w": 0.10}, 2: {"w": 0.05}, 3: {"w": 0.045},
             5: {"w": 0.0410}, 8: {"w": 0.0405}, 12: {"w": 0.0400}}
    out = runner.select(cells, 12, ("w",))
    assert out["selected"] == 5 and out["at_grid_max"] is False
    assert out["tol"] == runner.TOL


def test_select_falls_back_to_grid_max_when_nothing_converges():
    cells = {1: {"w": 0.10}, 2: {"w": 0.09}, 12: {"w": 0.02}}
    out = runner.select(cells, 12, ("w",))
    assert out["selected"] == 12 and out["at_grid_max"] is True


def test_frozen_grid_matches_preregistration():
    assert runner.R_GRID == (1, 2, 3, 5, 8, 12)
    assert runner.B_GRID == (1000, 2000, 5000, 10000, 20000)
    assert runner.TOL == 0.002
    assert runner.FALLBACK == {"R": 5, "B_per_batch": 10000}
    assert runner.MAX_HOURS == 12.0 and runner.MAX_PEAK_GB == 8.0
    assert runner.DELTA_ONLY == []
