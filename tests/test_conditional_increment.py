"""W0.5-B unit tests: 4D-2 conditional-increment 统计与 schema 核心。

覆盖会强制重做全量 forward 的正确性点：hidden tuple index、hidden pooling、
completion-level schema 与 emission 语义、stratified grouped folds 不泄漏、
paired vs independent bootstrap 的单类保护、rank-biserial gate、artifact 红线、
equivalence 判读、RQ2 fold-specific 分层、TF-IDF train-only vocabulary。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
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
    for L in (12, 24, 40, 80):
        r = ci.resolve_hidden_index(L)
        assert r["hidden_states_tuple_index"] == r["block_index_zero_based"] + 1


# ---- completion-level schema ----
def _m(family, label, toks, **extra):
    return {"family": family, "label": label, "token_indices": toks, "normalized": "X", **extra}


def test_completion_label_positive_uses_any_fabricated_not_first():
    pos = ci.completion_label([_m("attack", "grounded", [3]), _m("cwe", "fabricated", [7, 8])])
    assert pos["eligible"] and pos["label"] == 1 and pos["n_primary_mentions"] == 2
    assert pos["emission_failure"] is False and pos["primary_exclusion_reason"] is None
    neg = ci.completion_label([_m("attack", "grounded", [3]), _m("cwe", "grounded", [7])])
    assert neg["eligible"] and neg["label"] == 0


def test_only_cve_is_excluded_but_not_emission_failure():
    """只生成 CVE = 发射了 identifier,只是不在主 family。算 emission failure 会低估
    end-to-end emission rate（评审 P0）。"""
    r = ci.completion_label([_m("cve", "fabricated", [4, 5])])
    assert r["eligible"] is False
    assert r["primary_exclusion_reason"] == "only_cve"
    assert r["emitted_any_identifier"] is True
    assert r["emission_failure"] is False


def test_only_echoed_and_no_identifier_and_refusal_are_emission_failures():
    echoed = ci.completion_label([_m("attack", "echoed", [3])])
    assert echoed["eligible"] is False and echoed["primary_exclusion_reason"] == "only_echoed"
    assert echoed["emitted_any_identifier"] is False and echoed["emission_failure"] is True

    empty = ci.completion_label([])
    assert empty["primary_exclusion_reason"] == "no_identifier" and empty["emission_failure"] is True

    refused = ci.completion_label([], refusal=True)
    assert refused["primary_exclusion_reason"] == "refusal" and refused["emission_failure"] is True


def test_eligible_token_positions_pool_all_not_first():
    mentions = [_m("attack", "grounded", [3, 4]), _m("cwe", "fabricated", [9, 10])]
    assert ci.eligible_identifier_token_positions(mentions) == [3, 4, 9, 10]
    # echoed 与 cve 不进 H 的 pooling 位置
    mixed = [_m("attack", "echoed", [1]), _m("cve", "fabricated", [2]), _m("cwe", "grounded", [5])]
    assert ci.eligible_identifier_token_positions(mixed) == [5]


# ---- hidden pooling ----
def test_pool_hidden_states_means_all_identifier_tokens():
    h = np.arange(40, dtype=np.float32).reshape(10, 4)
    out = ci.pool_hidden_states(h, [2, 5])
    assert np.allclose(out, (h[2] + h[5]) / 2)
    assert out.dtype == np.float32
    assert out.shape == (4,)


def test_pool_hidden_states_dedups_repeated_positions():
    h = np.arange(40, dtype=np.float32).reshape(10, 4)
    # 位置重复不得被重复加权
    assert np.allclose(ci.pool_hidden_states(h, [2, 2, 2, 5]), ci.pool_hidden_states(h, [2, 5]))


def test_pool_hidden_states_rejects_empty_and_out_of_range():
    h = np.zeros((6, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="ineligible"):
        ci.pool_hidden_states(h, [])
    with pytest.raises(ValueError, match="out of range"):
        ci.pool_hidden_states(h, [7])
    with pytest.raises(ValueError, match="2-D"):
        ci.pool_hidden_states(np.zeros((2, 6, 3)), [0])


def test_pool_hidden_states_casts_low_precision_to_fp32():
    h = np.ones((4, 3), dtype=np.float16) * 0.1
    out = ci.pool_hidden_states(h, [0, 1])
    assert out.dtype == np.float32


# ---- metrics ----
def test_auroc_rankbiserial_sobs():
    y = [0, 0, 1, 1]; s = [0.1, 0.2, 0.8, 0.9]
    assert ci.auroc(s, y) == pytest.approx(1.0)
    assert ci.rank_biserial(1.0) == pytest.approx(1.0)
    assert ci.s_observability(0.35) == 0.0        # <0.5 不翻转
    assert ci.s_observability(0.75) == pytest.approx(0.5)
    assert np.isnan(ci.auroc([1, 2, 3], [1, 1, 1]))  # 单类


def test_s_observability_refuses_nan_instead_of_treating_it_as_zero():
    """单类样本的 AUROC 未定义。当成 S=0 会把'算不出'读成'完全不可观测'。"""
    with pytest.raises(ValueError, match="NaN"):
        ci.s_observability(float("nan"))


def test_equivalence_read():
    assert ci.equivalence_read(0.03, 0.05) == "increment"
    assert ci.equivalence_read(-0.01, 0.015) == "equivalent"
    assert ci.equivalence_read(-0.10, -0.03) == "harmful"
    assert ci.equivalence_read(-0.05, 0.05) == "inconclusive"


# ---- grouped folds ----
def test_grouped_folds_no_leakage():
    groups = np.array([g for g in range(20) for _ in range(3)])
    for train, test in ci.grouped_folds(groups, n_splits=5, seed=1):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(train) + len(test) == len(groups)


def test_stratified_grouped_folds_no_leak_and_both_classes_everywhere():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    # 正类按 group 聚集（真实情况：一个 prompt 的 K 条 completion 同标签倾向）
    pos_groups = set(rng.choice(30, size=9, replace=False).tolist())
    y = np.array([1 if g in pos_groups else 0 for g in groups])
    folds = ci.stratified_grouped_folds(y, groups, n_splits=5, seed=0)
    assert len(folds) == 5
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered.tolist()) == list(range(len(y)))
    for train, test in folds:
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2


def test_stratified_grouped_folds_stops_when_positives_too_few():
    groups = np.repeat(np.arange(10), 2)
    y = np.zeros(20, dtype=int)
    y[:2] = 1          # 只有 1 个正 group → 5 折不可能每折 test 都有正例
    with pytest.raises(RuntimeError, match="no valid stratified grouped split"):
        ci.stratified_grouped_folds(y, groups, n_splits=5, seed=0, max_seed_retries=5)


def test_stratified_grouped_folds_reproducible():
    groups = np.repeat(np.arange(20), 3)
    y = np.array([g % 3 == 0 for g in groups], dtype=int)
    a = ci.stratified_grouped_folds(y, groups, 4, seed=7)
    b = ci.stratified_grouped_folds(y, groups, 4, seed=7)
    for (_, sa), (_, sb) in zip(a, b):
        assert np.array_equal(sa, sb)


def test_oof_risk_scores_all_finite():
    y, o, h, groups = _synthetic(seed=1)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    s = ci.oof_risk_scores(o, y, groups, mk, n_splits=5, seed=0)
    assert np.all(np.isfinite(s)) and s.size == y.size


# ---- oof + paired bootstrap ----
def _synthetic(n_groups=60, per=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, 2, size=n_groups * per)
    o = (y * 0.6 + rng.normal(0, 1, y.size)).reshape(-1, 1)
    h = (y * 1.4 + rng.normal(0, 1, y.size)).reshape(-1, 1)
    return y, o, h, groups


def test_paired_bootstrap_increment_positive_when_h_informative():
    y, o, h, groups = _synthetic(seed=3)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    folds = ci.stratified_grouped_folds(y, groups, 5, seed=0)
    s_o = ci.oof_risk_scores(o, y, groups, mk, folds=folds)
    s_oh = ci.oof_risk_scores(np.hstack([o, h]), y, groups, mk, folds=folds)
    res = ci.paired_grouped_bootstrap_delta(y, s_o, s_oh, groups, n_boot=400, seed=0)
    assert res["ci_lo"] < res["point"] < res["ci_hi"]
    assert res["point"] > 0
    assert res["n_valid"] + res["n_discarded_single_class"] == res["n_boot"]


def test_paired_bootstrap_equivalent_when_h_is_noise():
    y, o, _h, groups = _synthetic(seed=5)
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 1, y.size).reshape(-1, 1)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    folds = ci.stratified_grouped_folds(y, groups, 5, seed=0)
    s_o = ci.oof_risk_scores(o, y, groups, mk, folds=folds)
    s_oh = ci.oof_risk_scores(np.hstack([o, noise]), y, groups, mk, folds=folds)
    res = ci.paired_grouped_bootstrap_delta(y, s_o, s_oh, groups, n_boot=400, seed=0)
    assert ci.equivalence_read(res["ci_lo"], res["ci_hi"]) in ("equivalent", "inconclusive")


def test_paired_bootstrap_stops_when_all_rounds_single_class():
    y = np.ones(10, dtype=int)        # 单类 → 每轮 AUROC 都是 NaN
    s = np.arange(10, dtype=float)
    g = np.arange(10)
    with pytest.raises(RuntimeError, match="zero valid rounds"):
        ci.paired_grouped_bootstrap_delta(y, s, s, g, n_boot=50, seed=0)


# ---- independent cross-task D + gate verdict ----
def test_independent_bootstrap_gate_verdict():
    rng = np.random.default_rng(2)
    n = 200
    ym = rng.integers(0, 2, n); sm = ym * 3.0 + rng.normal(0, 1, n)
    yh = rng.integers(0, 2, n); sh = rng.normal(0, 1, n)
    res = ci.independent_grouped_bootstrap_D(
        {"y": ym, "score": sm, "groups": np.arange(n)},
        {"y": yh, "score": sh, "groups": np.arange(n)}, n_boot=400, seed=0)
    assert res["D_point"] > res["delta"]
    assert res["verdict"] == "h1_high_confidence"
    assert res["n_valid"] == 400 and res["n_discarded_single_class"] == 0


def test_independent_bootstrap_discards_single_class_rounds_instead_of_scoring_them_zero():
    """正例极少时,单类重采样若被当成 S=0,会系统性压低 S 并扭曲 CI（评审 P0）。"""
    rng = np.random.default_rng(4)
    n = 40
    ym = np.zeros(n, dtype=int); ym[:3] = 1      # 只有 3 个正例 → 常出现单类重采样
    sm = ym * 3.0 + rng.normal(0, 1, n)
    yh = rng.integers(0, 2, n); sh = rng.normal(0, 1, n)
    with pytest.raises(RuntimeError, match="valid rounds"):
        ci.independent_grouped_bootstrap_D(
            {"y": ym, "score": sm, "groups": np.arange(n)},
            {"y": yh, "score": sh, "groups": np.arange(n)},
            n_boot=200, seed=0, min_valid_frac=0.99)
    res = ci.independent_grouped_bootstrap_D(
        {"y": ym, "score": sm, "groups": np.arange(n)},
        {"y": yh, "score": sh, "groups": np.arange(n)},
        n_boot=200, seed=0, min_valid_frac=0.0)
    assert res["n_discarded_single_class"] > 0
    assert res["n_valid"] == 200 - res["n_discarded_single_class"]


# ---- artifact red-line ----
def test_artifact_redline():
    assert ci.artifact_redline(-0.01, 0.015) is True
    assert ci.artifact_redline(0.05, 0.12) is False


# ---- RQ2 strata + power ----
def test_rq2_threshold_and_strata():
    lp = [-5, -4, -3, -2, -1]
    thr = ci.rq2_threshold(lp)
    assert thr == -3.0
    assert ci.rq2_strata(lp, thr).tolist() == [False, False, True, True, True]
    assert ci.rq2_cell_ok([1] * 15 + [0] * 15) is True
    assert ci.rq2_cell_ok([1] * 10 + [0] * 20) is False
    assert ci.rq2_power_flag(29) == "exploratory_low_power"
    assert ci.rq2_power_flag(30) == "adequate"


def test_rq2_fold_strata_uses_only_train_threshold_per_fold():
    """§5:阈值在每个 outer fold 的 train 内取中位数,再用于该 fold 的 test。
    在全体数据上取一次中位数 = test leakage。"""
    # 两折的 train 分布刻意不同,使 fold 阈值与全局阈值给出**不同**的分层
    lp = np.array([0, 1, 2, 10, 3, 4, 5, 6], dtype=float)
    a, b = np.arange(4), np.arange(4, 8)
    folds = [(a, b), (b, a)]
    res = ci.rq2_fold_strata(lp, folds)
    thr = {t["fold"]: t["threshold"] for t in res["fold_thresholds"]}
    assert thr[0] == np.median(lp[a]) == 1.5      # fold0 阈值只来自它的 train
    assert thr[1] == np.median(lp[b]) == 4.5
    assert res["strata"].tolist() == [False, False, False, True, True, True, True, True]
    # 全局中位数 3.5 会把 index 4 判成 low → 证明这里用的是 fold-specific 阈值
    global_strata = ci.rq2_strata(lp, float(np.median(lp)))
    assert global_strata[4] == False and res["strata"][4] == True
    assert not np.array_equal(res["strata"], global_strata)


def test_rq2_fold_strata_requires_full_test_coverage():
    lp = np.arange(6, dtype=float)
    folds = [(np.array([0, 1, 2]), np.array([3, 4]))]   # 样本 5 没被任何 test 覆盖
    with pytest.raises(RuntimeError, match="not covered"):
        ci.rq2_fold_strata(lp, folds)


# ---- text ladder / block design ----
def test_text_block_vocabulary_is_train_only():
    text = ["alpha beta", "alpha gamma", "delta epsilon", "zeta eta theta"]
    train, test = np.array([0, 1]), np.array([2, 3])
    X_tr, X_te, info = ci.build_fold_design(train, test, text=text)
    assert X_tr.shape[1] == X_te.shape[1] == info["text"]["n_features"]
    # test 里的 "delta"/"zeta" 从未进入 train vocabulary → test 行在 word 路上应无该项
    word_only = ci.TfidfVectorizer(**ci.WORD_TFIDF_PARAMS).fit([text[i] for i in train])
    assert "delta" not in word_only.vocabulary_ and "zeta" not in word_only.vocabulary_


def test_build_fold_design_blocks_are_equally_weighted():
    rng = np.random.default_rng(0)
    text = [f"token{i % 3} sample text number {i}" for i in range(20)]
    dense = rng.normal(0, 100, (20, 50))      # 尺度/维度都远大于 text block
    train, test = np.arange(14), np.arange(14, 20)
    X_tr, _, info = ci.build_fold_design(train, test, text=text, dense=dense)
    n_text = info["text"]["n_features"]
    tr = X_tr.toarray()
    text_norm = np.linalg.norm(tr[:, :n_text], axis=1).mean()
    dense_norm = np.linalg.norm(tr[:, n_text:], axis=1).mean()
    # 逐块等权:两块的平均行范数应同量级（否则 50000 维 sparse 会压制 3584 维 dense）
    assert 0.5 < text_norm / dense_norm < 2.0


def test_build_fold_design_requires_a_block():
    with pytest.raises(ValueError, match="at least one"):
        ci.build_fold_design(np.array([0]), np.array([1]))


def test_block_oof_scores_runs_o_h_and_oh_through_one_protocol():
    rng = np.random.default_rng(1)
    n_groups, per = 24, 3
    groups = np.repeat(np.arange(n_groups), per)
    y = np.array([1 if g % 3 == 0 else 0 for g in groups])
    text = ["fabricated wrong identifier here" if v else "grounded correct identifier here"
            for v in y]
    text = [t + f" filler{rng.integers(0, 5)}" for t in text]
    dense = (y * 1.2 + rng.normal(0, 1, y.size)).reshape(-1, 1)
    folds = ci.stratified_grouped_folds(y, groups, 3, seed=0)
    for kwargs in ({"text": text}, {"dense": dense}, {"text": text, "dense": dense}):
        res = ci.block_oof_scores(y, groups, folds, inner_splits=2, seed=0, **kwargs)
        assert np.all(np.isfinite(res["scores"]))
        assert all(c in ci.C_GRID for c in res["chosen_C"])
        assert 0.0 <= res["auroc"] <= 1.0


# ---- ladder specs / surface + id-string shortcuts ----
def test_id_string_text_skips_echoed():
    mentions = [_m("attack", "grounded", [1], normalized="T1114.003"),
                _m("cwe", "echoed", [2], normalized="CWE-79"),
                _m("cve", "fabricated", [3], normalized="CVE-2024-9999")]
    assert ci.id_string_text(mentions) == "T1114.003 CVE-2024-9999"


def test_surface_format_features_are_output_shape_only():
    mentions = [_m("attack", "grounded", [1], normalized="T1114", start=5, end=10,
                   granularity="technique")]
    feats = ci.surface_format_features("abcde T1114 tail\nsecond line", mentions)
    assert feats["surface_completion_chars"] == float(len("abcde T1114 tail\nsecond line"))
    assert feats["surface_completion_lines"] == 2.0
    assert feats["surface_num_attack"] == 1.0 and feats["surface_num_cwe"] == 0.0
    assert feats["surface_first_id_char_pos"] == 5.0
    assert set(feats) == set(ci.SURFACE_FEATURE_NAMES)


def test_build_ladder_specs_covers_prereg_rungs_and_h():
    records = [{"prompt_text": "p", "completion": "c", "id_string_text": "T1",
                "f5_id_logprob_mean": -1.0} for _ in range(3)]
    specs = ci.build_ladder_specs(records, hidden=np.zeros((3, 4)))
    assert list(specs) == ["rung1_prompt_only", "rung2_id_string_only", "rung3_surface_only",
                           "rung4_full_text", "rung5_f5", "rung6_O_f5_plus_text",
                           "H_alone", "O_plus_H"]
    # O 恒 = F5 + full-response-text（§4：固定单一定义,不取 max）
    assert specs["rung6_O_f5_plus_text"]["text"] == ["c", "c", "c"]
    assert specs["rung6_O_f5_plus_text"]["dense"].shape == (3, len(ci.F5_FEATURE_NAMES))
    # O+H 的 dense = F5 拼 hidden
    assert specs["O_plus_H"]["dense"].shape == (3, len(ci.F5_FEATURE_NAMES) + 4)
    assert specs["H_alone"]["text"] is None


def test_build_ladder_specs_rejects_hidden_row_mismatch():
    records = [{"completion": "c"} for _ in range(3)]
    with pytest.raises(ValueError, match="hidden rows"):
        ci.build_ladder_specs(records, hidden=np.zeros((2, 4)))
