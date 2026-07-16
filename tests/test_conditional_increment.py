"""W0.5-B unit tests: 4D-2 conditional-increment 统计与 schema 核心。

覆盖会强制重做全量 forward 的正确性点：hidden tuple index、completion-level schema、
grouped folds 不泄漏、paired vs independent bootstrap、rank-biserial gate、
artifact 红线、equivalence 判读、RQ2 分层与最小样本。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci


# ---- hidden index (Qwen 28→20, Llama 32→23) ----
def test_hidden_tuple_index_qwen_llama():
    q = ci.resolve_hidden_index(28)
    assert q["block_index_zero_based"] == 19 and q["hidden_states_tuple_index"] == 20
    la = ci.resolve_hidden_index(32)
    assert la["block_index_zero_based"] == 22 and la["hidden_states_tuple_index"] == 23
    # tuple index = block + 1 恒成立
    for L in (12, 24, 40, 80):
        r = ci.resolve_hidden_index(L)
        assert r["hidden_states_tuple_index"] == r["block_index_zero_based"] + 1


# ---- completion-level schema ----
def _m(family, label, toks):
    return {"family": family, "label": label, "token_indices": toks}


def test_completion_label_positive_negative_and_emission_failure():
    # 首 id grounded、次 id fabricated → completion positive（不能只看第一个）
    pos = ci.completion_label([_m("attack", "grounded", [3]), _m("cwe", "fabricated", [7, 8])])
    assert pos["eligible"] and pos["label"] == 1 and pos["n_primary_mentions"] == 2
    neg = ci.completion_label([_m("attack", "grounded", [3]), _m("cwe", "grounded", [7])])
    assert neg["eligible"] and neg["label"] == 0
    # 无 primary（拒答/无 id / 只有 echoed / 只有 cve 且被排除策略外）→ emission failure
    ef = ci.completion_label([_m("attack", "echoed", [3])])
    assert (not ef["eligible"]) and ef["emission_failure"] and ef["label"] is None


def test_eligible_token_positions_pool_all_not_first():
    mentions = [_m("attack", "grounded", [3, 4]), _m("cwe", "fabricated", [9, 10])]
    pos = ci.eligible_identifier_token_positions(mentions)
    assert pos == [3, 4, 9, 10]  # 全部 eligible id token，不是只取第一个 [3,4]


# ---- metrics ----
def test_auroc_rankbiserial_sobs():
    y = [0, 0, 1, 1]; s = [0.1, 0.2, 0.8, 0.9]
    assert ci.auroc(s, y) == pytest.approx(1.0)
    assert ci.rank_biserial(1.0) == pytest.approx(1.0)
    assert ci.s_observability(0.35) == 0.0        # <0.5 不翻转
    assert ci.s_observability(0.75) == pytest.approx(0.5)
    assert np.isnan(ci.auroc([1, 2, 3], [1, 1, 1]))  # 单类


def test_equivalence_read():
    assert ci.equivalence_read(0.03, 0.05) == "increment"
    assert ci.equivalence_read(-0.01, 0.015) == "equivalent"
    assert ci.equivalence_read(-0.10, -0.03) == "harmful"
    assert ci.equivalence_read(-0.05, 0.05) == "inconclusive"


# ---- grouped folds: 同 group 不跨 train/test ----
def test_grouped_folds_no_leakage():
    groups = np.array([g for g in range(20) for _ in range(3)])  # 20 groups × 3 rows
    for train, test in ci.grouped_folds(groups, n_splits=5, seed=1):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(train) + len(test) == len(groups)


def test_grouped_folds_reproducible():
    g = np.arange(30)
    a = ci.grouped_folds(g, 5, seed=7); b = ci.grouped_folds(g, 5, seed=7)
    for (ta, sa), (tb, sb) in zip(a, b):
        assert np.array_equal(sa, sb)


# ---- oof + paired bootstrap: O+H 应 ≥ O 当 H 携带信号 ----
def _synthetic(n_groups=60, per=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, 2, size=n_groups * per)
    # O 特征弱相关，H 特征强相关
    o = (y * 0.6 + rng.normal(0, 1, y.size)).reshape(-1, 1)
    h = (y * 1.4 + rng.normal(0, 1, y.size)).reshape(-1, 1)
    return y, o, h, groups


def test_paired_bootstrap_increment_positive_when_h_informative():
    y, o, h, groups = _synthetic(seed=3)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    s_o = ci.oof_risk_scores(o, y, groups, mk, n_splits=5, seed=0)
    s_oh = ci.oof_risk_scores(np.hstack([o, h]), y, groups, mk, n_splits=5, seed=0)
    res = ci.paired_grouped_bootstrap_delta(y, s_o, s_oh, groups, n_boot=400, seed=0)
    assert res["ci_lo"] < res["point"] < res["ci_hi"]
    assert res["point"] > 0  # H 有信号，O+H 应更强


def test_paired_bootstrap_equivalent_when_h_is_noise():
    y, o, _h, groups = _synthetic(seed=5)
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 1, y.size).reshape(-1, 1)  # H = 纯噪声
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    s_o = ci.oof_risk_scores(o, y, groups, mk, 5, 0)
    s_oh = ci.oof_risk_scores(np.hstack([o, noise]), y, groups, mk, 5, 0)
    res = ci.paired_grouped_bootstrap_delta(y, s_o, s_oh, groups, n_boot=400, seed=0)
    read = ci.equivalence_read(res["ci_lo"], res["ci_hi"])
    assert read in ("equivalent", "inconclusive")  # 不应误判 increment


# ---- independent cross-task D + gate verdict ----
def test_independent_bootstrap_gate_verdict():
    rng = np.random.default_rng(2)
    # MCQ: 输出高度可分（S 大）；H1: 近随机（S 小）→ D 大 → h1_high_confidence
    n = 200
    ym = rng.integers(0, 2, n); sm = ym * 3.0 + rng.normal(0, 1, n)
    yh = rng.integers(0, 2, n); sh = rng.normal(0, 1, n)
    gm = np.arange(n); gh = np.arange(n)
    res = ci.independent_grouped_bootstrap_D(
        {"y": ym, "score": sm, "groups": gm}, {"y": yh, "score": sh, "groups": gh},
        n_boot=400, seed=0)
    assert res["D_point"] > res["delta"]
    assert res["verdict"] == "h1_high_confidence"


# ---- artifact red-line ----
def test_artifact_redline():
    assert ci.artifact_redline(-0.01, 0.015) is True    # shortcut ≈ full-text
    assert ci.artifact_redline(0.05, 0.12) is False     # full-text 明显更强


# ---- RQ2 strata + power ----
def test_rq2_strata_and_power():
    lp = [-5, -4, -3, -2, -1]
    thr = ci.rq2_threshold(lp)
    assert thr == -3.0
    strata = ci.rq2_strata(lp, thr)
    assert strata.tolist() == [False, False, True, True, True]
    assert ci.rq2_cell_ok([1] * 15 + [0] * 15) is True
    assert ci.rq2_cell_ok([1] * 10 + [0] * 20) is False
    assert ci.rq2_power_flag(29) == "exploratory_low_power"
    assert ci.rq2_power_flag(30) == "adequate"
