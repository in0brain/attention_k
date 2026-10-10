"""confirmatory analyzer 的**结果契约**测试（预注册 §6「各分支必报」）。

**不解盲、不读真实 label、不写真实产物。** 所有数据由本文件或
`v3_analyze.synthetic_contract_probe()` 合成。

覆盖研究者点名的五项阻断：

    1  E3 真正接线（block-specific label + block-specific H + M2d-gate）
    2  E1 必报项齐全（全部 AUROC / M1b / iv-a / iv-b / T1）
    3  E2 可估时按 E1 协议继续估计
    4  M2c 有结果分支且带三项 caveat
    5  readiness 用**真** end-to-end 结果契约判断，不再只验函数名
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_analysis_paths as vpaths  # noqa: E402
from recover_attention.evaluation import v3_analyze as va  # noqa: E402
from recover_attention.evaluation import v3_grouping as g  # noqa: E402
from recover_attention.evaluation import v3_readiness as vr  # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fx  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

_REAL = _REPO_ROOT / "outputs" / "stage1"


@pytest.fixture(scope="module")
def probe_result():
    """全合成的 end-to-end 结果。与 readiness 用的是**同一个**缓存。"""
    return vr._probe_end_to_end()


def load_entry():
    spec = importlib.util.spec_from_file_location(
        "_v3_entry_contract", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def entry():
    return load_entry()


# ══════════════════════════════════ 2  E1 必报项
def test_every_arm_reports_all_mandatory_aurocs(probe_result):
    """§6：AUROC(S*) / AUROC(H alone) / AUROC(S*+H) 以及**每个**单 block。"""
    for path in vpaths.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = probe_result["arms"][path][arm]
            auroc = a["auroc"]
            for k in ("S*", "S*+H", "H"):
                assert k in auroc, f"{path}/{arm} 缺 AUROC({k})"
            for b in a["block_names_s_star"]:
                assert b in auroc, f"{path}/{arm} 缺单 block AUROC({b})"
            assert auroc["H"] is not None, "AUROC(H alone) 必须真的算出来"
            # path C 移除 B4 —— 不得残留
            if path == vpaths.PATH_C:
                assert "B4" not in a["block_names_s_star"]


def test_every_arm_reports_m1b_iva_ivb_and_t1(probe_result):
    for path in vpaths.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = probe_result["arms"][path][arm]
            ni = a["m1b_non_inferiority"]
            assert ni["status"] in ("CERTIFIED", "NOT CERTIFIED")
            assert set(ni["L_marg"]) == set(a["block_names_s_star"])
            assert "L_sim" in ni                       # secondary 也必须在场
            iva = a["m1b_iv_a_direct_superiority"]
            assert set(iva["L_marg"]) == set(a["block_names_s_star"])
            assert isinstance(iva["passed"], bool)
            assert "sole authorising path" in iva["criterion"]
            ivb = a["m1b_iv_b_consistency"]
            assert ivb["holds"] == (ivb["m1b_certified"] and ivb["second_term_holds"])
            t1 = a["m1c_t1"]
            assert t1["computed"] is True and t1["role"] == "I"
            assert set(t1["role_I_marginal_upper_bounds"]) == set(va.T1_SHORTCUT_BLOCKS)


def test_bootstrap_budget_and_mc_error_are_reported(probe_result):
    for path in vpaths.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = probe_result["arms"][path][arm]
            bb = a["bootstrap_budget"]
            assert bb["B_total"] == bb["B_batch"] * bb["B_per_batch"]
            assert bb["R"] >= 1 and bb["n_valid_draws"] >= 1
            mc = a["ci_width_and_mc_error"]
            assert mc["ci_width"] == a["delta_h_ci"]["width"]
            assert "mc_band_width" in mc["mc_jackknife"]
            assert "se_mc" in mc["mc_batch_means_secondary"]


# ══════════════════════════════════ M1c T1 的口径
def test_t1_uses_the_exact_algebraic_identity_and_the_upper_bound():
    """Δ_task = Δ_k[shortcut] − Δ_k[B3]（恒等式），判据方向是**上界**。"""
    cfg = V3AnalysisConfig()
    rng = np.random.default_rng(0)
    draws = {b: rng.normal(0.03, 0.01, 400) for b in ("B1", "B2", "B3", "B4", "B5")}
    point = {b: float(v.mean()) for b, v in draws.items()}
    t1 = va.t1_diagnostic(point, draws, cfg)

    for sc in va.T1_SHORTCUT_BLOCKS:
        want_draws = draws[sc] - draws["B3"]
        assert t1["point"][sc] == pytest.approx(point[sc] - point["B3"])
        assert t1["role_I_marginal_upper_bounds"][sc] == pytest.approx(
            float(np.percentile(want_draws, 100 * (1 - cfg.alpha))))
        # 上界必须 >= 下界方向的镜像，且**不是**下界
        from recover_attention.evaluation import v3_bootstrap as vb
        assert (t1["role_I_marginal_upper_bounds"][sc]
                > vb.marginal_lower_bound(want_draws, cfg.alpha))


def test_t1_triggers_only_when_the_upper_bound_is_below_eps():
    cfg = V3AnalysisConfig()
    rng = np.random.default_rng(1)
    # Δ_task ≈ 0（极小抖动，非退化）→ 上界 ≈ 0 < eps → 触发
    same = rng.normal(0.05, 0.001, 400)
    jitter = lambda: same + rng.normal(0, 1e-5, 400)      # noqa: E731
    draws = {"B1": jitter(), "B2": jitter(), "B3": jitter(),
             "B4": same, "B5": same}
    point = {b: float(v.mean()) for b, v in draws.items()}
    t1 = va.t1_diagnostic(point, draws, cfg)
    assert t1["triggered"] is True
    assert all(t1["role_I_triggered_per_shortcut"].values())

    # Δ_task 明显为正且远大于 eps → 不触发。
    # 注意方向：Δ_task = Δ_k[shortcut] − Δ_k[B3]，而 Δ_k[b] = AUROC(S*) − AUROC(b)。
    # 所以「shortcut 弱（Δ_k 大）、B3 强（Δ_k 小）」才给出大的正 Δ_task。
    draws2 = {"B1": rng.normal(0.5, 0.001, 400), "B2": rng.normal(0.5, 0.001, 400),
              "B3": rng.normal(0.0, 0.001, 400), "B4": same, "B5": same}
    point2 = {b: float(v.mean()) for b, v in draws2.items()}
    t2 = va.t1_diagnostic(point2, draws2, cfg)
    assert t2["triggered"] is False


def test_t1_role_II_degenerates_softly_without_crashing_analyze():
    """Δ_task 完全退化时，只让角色 II 不可用 —— 不得让解盲后的 analyze() 崩掉。"""
    cfg = V3AnalysisConfig()
    same = np.full(200, 0.05)
    draws = {b: same for b in ("B1", "B2", "B3", "B4", "B5")}
    t1 = va.t1_diagnostic({b: 0.05 for b in draws}, draws, cfg)
    assert t1["computed"] is True
    assert t1["role_II_available"] is False
    assert t1["role_II_simultaneous_upper_bounds"] == {}
    assert "unavailable" in t1["simultaneous_coverage"]
    assert t1["role"] == "I" and t1["triggered"] is True   # 角色 I 仍然给得出结论


def test_t1_declares_role_I_by_default_and_offers_role_II_bounds():
    cfg = V3AnalysisConfig()
    rng = np.random.default_rng(2)
    draws = {b: rng.normal(0.02, 0.01, 300) for b in ("B1", "B2", "B3", "B4", "B5")}
    t1 = va.t1_diagnostic({b: float(v.mean()) for b, v in draws.items()}, draws, cfg)
    assert t1["role"] == "I"
    assert "保守 admission gate" in t1["role_declaration"]
    assert "不得" in t1["role_declaration"]
    assert set(t1["role_II_simultaneous_upper_bounds"]) <= set(va.T1_SHORTCUT_BLOCKS)
    assert "角色 II" in t1["role_II_note"]
    assert "nominal" in t1["simultaneous_coverage"]
    # 同时上界必须**不严于**边际上界（更保守 → 更大）
    for b in t1["role_II_simultaneous_upper_bounds"]:
        assert (t1["role_II_simultaneous_upper_bounds"][b]
                >= t1["role_I_marginal_upper_bounds"][b] - 1e-9)


def test_t1_reports_not_computed_when_a_block_is_missing():
    cfg = V3AnalysisConfig()
    t1 = va.t1_diagnostic({"B1": 0.0, "B2": 0.0}, {"B1": np.zeros(5), "B2": np.zeros(5)},
                          cfg)
    assert t1["computed"] is False and "B3" in t1["reason"]


# ══════════════════════════════════ iv-b 必须写成合取
@pytest.mark.parametrize("certified,ci_lo,holds", [
    (True, 0.05, True), (True, 0.01, False), (False, 0.05, False), (False, 0.0, False)])
def test_iv_b_is_a_conjunction_never_the_second_term_alone(certified, ci_lo, holds):
    out = va.iv_b_consistency({"certified": certified}, {"ci_lo": ci_lo}, 0.02)
    assert out["holds"] is holds
    assert out["conjunction_required"].count("AND") == 1
    assert "不得只写第二项" in out["must_not"]
    assert "不得与 (iv-a) 取并集" in out["must_not"]


def test_result_forbids_the_or_of_iv_a_and_iv_b(probe_result):
    for path in vpaths.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            w = probe_result["arms"][path][arm]["wording_prohibitions"]
            assert "绝不允许" in w["never_allowed"]
            iva = probe_result["arms"][path][arm]["m1b_iv_a_direct_superiority"]
            if not iva["passed"]:
                assert w["iv_a_failed_forbids"] and "超过最佳表层 block" in w["iv_a_failed_forbids"]


def test_wording_prohibitions_track_the_actual_verdicts():
    w = va.wording_prohibitions({"certified": False}, {"passed": False},
                                {"triggered": True})
    assert w["m1b_not_certified_forbids"] and "作废" in w["m1b_not_certified_forbids"]
    assert w["iv_a_failed_forbids"] and w["t1_triggered_forbids"]
    ok = va.wording_prohibitions({"certified": True}, {"passed": True},
                                 {"triggered": False})
    assert ok["m1b_not_certified_forbids"] is None
    assert ok["iv_a_failed_forbids"] is None and ok["t1_triggered_forbids"] is None


# ══════════════════════════════════ 1  E3 真正接线
def _rec(gid, tid, ident, mention_label, sample_type="greedy"):
    return {"trace_id": tid, "group_id": gid, "sample_type": sample_type,
            "sample_index": 0, "example_id": gid,
            "mentions": [{"family": "attack", "normalized": ident,
                          "label": mention_label, "token_indices": [0, 1]}]}


def _ids_per_block(cfg, need=6):
    per = {b: [] for b in range(int(cfg.e3_n_blocks))}
    i = 0
    while min(len(v) for v in per.values()) < need and i < 200000:
        ident = f"T{1000 + i % 9000:04d}"
        per[g.e3_block_of(ident, cfg)].append(ident)
        i += 1
    return per


def test_e3_block_label_is_the_frozen_rule_restricted_to_the_block():
    """块内标签 = 冻结 `completion_label` 限制到块内 primary mention，规则本身不变。"""
    cfg = V3AnalysisConfig()
    per = _ids_per_block(cfg)
    a, b = per[0][0], per[1][0]
    rec = {"mentions": [
        {"family": "attack", "normalized": a, "label": "fabricated"},
        {"family": "attack", "normalized": b, "label": "grounded"}]}
    assert g.e3_block_label(rec, cfg, 0) == {"eligible": True, "label": 1,
                                             "n_primary_mentions_in_block": 1}
    assert g.e3_block_label(rec, cfg, 1) == {"eligible": True, "label": 0,
                                             "n_primary_mentions_in_block": 1}
    # 块内无 primary mention → 不 eligible（与 E1 的 ineligible 处理一致）
    assert g.e3_block_label(rec, cfg, 2)["eligible"] is False
    # echoed 不是 primary
    echo = {"mentions": [{"family": "attack", "normalized": a, "label": "echoed"}]}
    assert g.e3_block_label(echo, cfg, 0)["eligible"] is False
    # CVE 不进 primary
    cve = {"mentions": [{"family": "cve", "normalized": "CVE-2024-1234",
                         "label": "fabricated"}]}
    assert all(not g.e3_block_label(cve, cfg, x)["eligible"]
               for x in range(int(cfg.e3_n_blocks)))


def test_e3_is_estimable_and_wired_on_a_well_formed_sample():
    cfg = V3AnalysisConfig()
    per = _ids_per_block(cfg)
    recs = []
    for b in range(int(cfg.e3_n_blocks)):
        for j in range(4):
            gid = f"p_{b}_{j}"
            recs.append(_rec(gid, f"t_{b}_{j}", per[b][j],
                             "fabricated" if j % 2 == 0 else "grounded"))
    out = g.build_e3_analysis(recs, cfg)
    assert out["estimability"] == g.ESTIMABLE
    assert out["wiring_status"] == "wired" and out["uses_label"] is True
    assert len(out["per_fold"]) == int(cfg.e3_n_blocks)
    assert all(v["n_positive"] > 0 and v["n_negative"] > 0
               for v in out["per_fold"].values())
    assert out["retention"]["mention_retention_rate"] == 1.0
    assert len(out["_folds"]) == int(cfg.e3_n_blocks)


@pytest.mark.parametrize("case", ["empty_fold", "single_class_fold",
                                  "single_class_complement"])
def test_e3_m2d_gate_fires_on_each_degeneracy(case):
    """M2d-gate 三种触发条件各一条负控制。"""
    cfg = V3AnalysisConfig()
    per = _ids_per_block(cfg)
    recs = []
    if case == "empty_fold":
        for b in range(int(cfg.e3_n_blocks) - 1):          # 少一个块 → 空 fold
            for j in range(4):
                recs.append(_rec(f"p_{b}_{j}", f"t_{b}_{j}", per[b][j],
                                 "fabricated" if j % 2 else "grounded"))
        needle = "(a) empty fold"
    elif case == "single_class_fold":
        for b in range(int(cfg.e3_n_blocks)):
            for j in range(4):
                lab = "fabricated" if (b == 0 or j % 2) else "grounded"
                recs.append(_rec(f"p_{b}_{j}", f"t_{b}_{j}", per[b][j], lab))
        needle = "(b) single class"
    else:
        for b in range(int(cfg.e3_n_blocks)):
            for j in range(4):
                lab = "fabricated" if b == 0 else "grounded"
                recs.append(_rec(f"p_{b}_{j}", f"t_{b}_{j}", per[b][j], lab))
        needle = "(c) train complement"
    out = g.build_e3_analysis(recs, cfg)
    assert out["estimability"] == g.NON_ESTIMABLE
    assert any(needle in r for r in out["degenerate_reasons"]), out["degenerate_reasons"]
    # non-estimable 时仍必须报每折 n / 正负例 / retention，且**不含**任何折数据句柄
    assert out["per_fold"] and out["retention"]
    assert "_folds" not in out and "_y" not in out
    assert out["performance_estimates_computed"] is False


def test_e3_gate_message_forbids_rescuing_estimability():
    cfg = V3AnalysisConfig()
    out = g.build_e3_analysis([_rec("p0", "t0", "T1234", "fabricated")], cfg)
    for banned in ("改 F", "换 salt", "重新分块", "合并或丢弃 fold"):
        assert banned in out["gate"]


def test_e3_hidden_comes_from_canonical_repool_and_never_falls_back():
    """缺 block-specific 向量 → 抛错；**不补零、不退回 E1 的 H**。"""
    recs = [{"trace_id": "t", "example_id": "e", "sample_type": "greedy",
             "sample_index": 0, "group_id": "p"}]
    vec = np.arange(4, dtype=np.float32)
    e3 = {"vectors": {("e__greedy_0", 2): vec}}
    got = va._e3_block_hidden(recs, [0], [2], e3, 4)
    assert np.array_equal(got[0], vec)
    with pytest.raises(va.ContractViolation, match="不得"):
        va._e3_block_hidden(recs, [0], [3], e3, 4)


def test_e3_result_is_estimated_and_declares_independence(probe_result):
    for path in vpaths.FROZEN_PATHS:
        e3 = probe_result["estimands"][path]["E3"]["h1"]
        assert e3["estimability"] == g.ESTIMABLE
        assert e3["performance_estimates_computed"] is True
        assert "summary" in e3 and e3["summary"]["estimand"] == "E3"
        assert "block-specific label" in e3["summary"]["label_caliber"]
        assert "canonical repool" in e3["h_source"]
        assert "不要求一致" in e3["independence"]
        assert probe_result["estimands"][path]["E3"]["mcq"]["estimability"] == \
            g.NOT_APPLICABLE


def test_e3_fold_assignment_caliber_is_documented_not_silently_relied_on():
    cfg = V3AnalysisConfig()
    out = g.build_e3_analysis([_rec("p0", "t0", "T1234", "fabricated")], cfg)
    assert "emitted" in out["first_identifier_caliber"]
    assert "echoed" in out["first_identifier_caliber"]
    assert out["e3_block_salt"] == cfg.e3_block_salt


# ══════════════════════════════════ 3  E2 可估时继续估计
def test_e2_non_estimable_reports_structure_only():
    recs, _h = fx.make_h1(n_prompts=6)
    out = g.build_e2_groups(recs, cfg=V3AnalysisConfig())
    assert out["estimability"] == g.NON_ESTIMABLE
    assert out["performance_estimates_computed"] is False
    assert out["groups"] is None
    assert out["component_count"] and "max_component_share" in out


def test_e2_estimable_branch_runs_the_e1_protocol(probe_result):
    for path in vpaths.FROZEN_PATHS:
        e2 = probe_result["estimands"][path]["E2"]["h1"]
        assert e2["estimability"] == g.ESTIMABLE
        assert e2["performance_estimates_computed"] is True
        assert "summary" in e2 and e2["summary"]["estimand"] == "E2"
        assert "与 E1 相同的协议" in e2["protocol"]
        # 同口径 → 必报项一样齐
        for k in va.REQUIRED_ARM_KEYS:
            if k != "path":
                assert k in e2["summary"], k


def test_e2_non_estimable_branch_never_reports_performance(probe_result):
    """契约 validator 必须拒绝『non-estimable 却报了性能量』。"""
    doc = json.loads(json.dumps(probe_result, default=str))
    e2 = doc["estimands"][vpaths.PATH_A]["E2"]["h1"]
    e2["estimability"] = g.NON_ESTIMABLE
    e2["performance_estimates_computed"] = True
    with pytest.raises(va.ContractViolation, match="E2 non-estimable 却报了性能量"):
        va.validate_result_contract(doc)


# ══════════════════════════════════ 4  M2c
def test_m2c_is_label_informed_sensitivity_with_three_caveats(probe_result):
    for path in vpaths.FROZEN_PATHS:
        m = probe_result["estimands"][path]["M2c"]["h1"]
        assert m["is_primary"] is False
        assert m["role"] == "label_informed_sensitivity"
        assert m["uses_label"] is True
        assert len(m["interpretation_caveats"]) == 3
        joined = " ".join(m["interpretation_caveats"])
        assert "label-informed" in joined
        assert "偏倚方向" in joined and "未知" in joined
        assert "grounded identifier 仍跨折" in joined
        assert "summary" in m and m["summary"]["estimand"] == "M2c"
        assert probe_result["estimands"][path]["M2c"]["mcq"]["estimability"] == \
            g.NOT_APPLICABLE


def test_m2c_edges_come_only_from_fabricated_mentions():
    cfg = V3AnalysisConfig()
    shared = "T1234"
    recs = [_rec("p0", "t0", shared, "grounded"),
            _rec("p1", "t1", shared, "grounded")]
    out = g.build_m2c_groups(recs, cfg)
    assert out["n_fabricated_identity_edges"] == 0
    assert out["component_count"] == 2, "grounded 不得连边"

    recs2 = [_rec("p0", "t0", shared, "fabricated"),
             _rec("p1", "t1", shared, "fabricated")]
    out2 = g.build_m2c_groups(recs2, cfg)
    assert out2["n_fabricated_identity_edges"] == 2
    assert out2["component_count"] == 1, "fabricated 必须把两个 prompt 连起来"


# ══════════════════════════════════ 结果契约 validator
def test_contract_accepts_the_probe_result(probe_result):
    assert va.validate_result_contract(probe_result)["ok"] is True


@pytest.mark.parametrize("drop", ["auroc", "m1b_iv_a_direct_superiority",
                                  "m1b_iv_b_consistency", "m1c_t1",
                                  "bootstrap_budget", "ci_width_and_mc_error",
                                  "wording_prohibitions"])
def test_contract_rejects_a_missing_mandatory_arm_key(probe_result, drop):
    doc = json.loads(json.dumps(probe_result, default=str))
    doc["arms"][vpaths.PATH_A]["h1"].pop(drop)
    with pytest.raises(va.ContractViolation, match=drop):
        va.validate_result_contract(doc)


def test_contract_rejects_a_missing_auroc_h():
    doc = json.loads(json.dumps(vr._probe_end_to_end(), default=str))
    doc["arms"][vpaths.PATH_B]["mcq"]["auroc"].pop("H")
    with pytest.raises(va.ContractViolation, match="auroc 缺 H"):
        va.validate_result_contract(doc)


@pytest.mark.parametrize("estimand", ["E1", "E2", "E3", "M2c"])
def test_contract_rejects_a_missing_estimand(probe_result, estimand):
    doc = json.loads(json.dumps(probe_result, default=str))
    doc["estimands"][vpaths.PATH_C].pop(estimand)
    with pytest.raises(va.ContractViolation, match=estimand):
        va.validate_result_contract(doc)


def test_contract_rejects_m2c_without_three_caveats(probe_result):
    doc = json.loads(json.dumps(probe_result, default=str))
    doc["estimands"][vpaths.PATH_A]["M2c"]["h1"]["interpretation_caveats"] = ["only one"]
    with pytest.raises(va.ContractViolation, match="三项 interpretation caveat"):
        va.validate_result_contract(doc)


def test_contract_rejects_a_bad_m1b_status(probe_result):
    doc = json.loads(json.dumps(probe_result, default=str))
    doc["arms"][vpaths.PATH_A]["h1"]["m1b_non_inferiority"]["status"] = "maybe"
    with pytest.raises(va.ContractViolation, match="CERTIFIED/NOT CERTIFIED"):
        va.validate_result_contract(doc)


def test_contract_rejects_a_tampered_path_set(probe_result):
    doc = json.loads(json.dumps(probe_result, default=str))
    doc["frozen_paths"] = list(doc["frozen_paths"]) + ["extra"]
    with pytest.raises(va.ContractViolation, match="frozen_paths"):
        va.validate_result_contract(doc)


def test_contract_rejects_e3_non_estimable_that_reports_performance(probe_result):
    doc = json.loads(json.dumps(probe_result, default=str))
    e3 = doc["estimands"][vpaths.PATH_A]["E3"]["h1"]
    e3["estimability"] = g.NON_ESTIMABLE
    with pytest.raises(va.ContractViolation, match="E3 non-estimable 却报了性能量"):
        va.validate_result_contract(doc)


def test_analyze_result_carries_the_contract_stamp(probe_result):
    assert probe_result["contract"]["ok"] is True
    assert probe_result["schema"] == va.ANALYSIS_SCHEMA


# ══════════════════════════════════ 5  readiness 用真 end-to-end
@pytest.mark.local_artifacts('outputs/stage1/readiness/stage1_finalized_freeze_manifest.json', 'outputs/stage1/readiness/stage1_analysis_input_freeze_manifest.json')
def test_readiness_v3_uses_a_real_end_to_end_not_symbol_names(entry):
    doc = vr.recompute_v3(
        prod_dir=entry.PROD, tail_dir=entry.TAIL_DIR, shadow_dir=entry.SHADOW_DIR,
        repool_dir=entry.REPOOL_DIR, result_dir=entry.RESULT_DIR,
        unblind_log=entry.UNBLIND_LOG, freeze_manifest_path=entry.FREEZE_MANIFEST,
        analysis_input_manifest_path=entry.ANALYSIS_INPUT_MANIFEST,
        entry_source_path=entry.ENTRY_SOURCE)
    by = {c["name"]: c for c in doc["checks"]}
    assert "estimands_wired" not in by, "只验函数名的假阳性检查必须已移除"
    chk = by["estimands_wired_end_to_end"]
    assert chk["ok"] is True
    d = chk["detail"]
    assert d["contract"]["ok"] is True
    assert d["e3_wiring_status"] == "wired"
    assert d["e3_performance_computed"] is True
    assert d["e3_estimability"] == g.ESTIMABLE
    assert d["m2c_caveats"] == 3
    assert "不读真实 label" in d["mode"]
    assert doc["readiness"]["ready_for_unblinding"] is True


def test_readiness_v3_blocks_when_the_contract_fails(entry, monkeypatch):
    """契约不过 → readiness 必须变红，而不是照样 ready。"""
    monkeypatch.setattr(vr, "_probe_end_to_end",
                        lambda: (_ for _ in ()).throw(
                            va.ContractViolation("缺 m1c_t1")))
    doc = vr.recompute_v3(
        prod_dir=entry.PROD, tail_dir=entry.TAIL_DIR, shadow_dir=entry.SHADOW_DIR,
        repool_dir=entry.REPOOL_DIR, result_dir=entry.RESULT_DIR,
        unblind_log=entry.UNBLIND_LOG, freeze_manifest_path=entry.FREEZE_MANIFEST,
        analysis_input_manifest_path=entry.ANALYSIS_INPUT_MANIFEST,
        entry_source_path=entry.ENTRY_SOURCE)
    assert doc["readiness"]["ready_for_unblinding"] is False
    assert "estimands_wired_end_to_end" in doc["readiness"]["blocking_reasons"]


@pytest.mark.pre_unblinding_workspace
@pytest.mark.local_artifacts('outputs/stage1/readiness/stage1_confirmatory_readiness_v3.json')
def test_disk_readiness_v3_matches_a_fresh_recompute(entry):
    doc = vr.recompute_and_compare_v3(
        entry.READINESS_V3, prod_dir=entry.PROD, tail_dir=entry.TAIL_DIR,
        shadow_dir=entry.SHADOW_DIR, repool_dir=entry.REPOOL_DIR,
        result_dir=entry.RESULT_DIR, unblind_log=entry.UNBLIND_LOG,
        freeze_manifest_path=entry.FREEZE_MANIFEST,
        analysis_input_manifest_path=entry.ANALYSIS_INPUT_MANIFEST,
        entry_source_path=entry.ENTRY_SOURCE)
    assert doc["readiness"]["ready_for_unblinding"] is True


# ══════════════════════════════════ 边界：仍未解盲
def test_probe_is_fully_synthetic_and_touches_no_real_artifact():
    probe = va.synthetic_contract_probe()
    for r in probe["h1_records"]:
        assert r["trace_id"].startswith("probe_")
        assert r["schema_version"] == "synthetic_probe_v1"
    assert probe["e3_blocks"]["synthetic"] is True
    assert probe["shadow_overlay"]["synthetic"] is True


@pytest.mark.pre_unblinding_workspace
def test_still_not_unblinded(entry):
    assert not entry.UNBLIND_LOG.exists()
    assert not (entry.RESULT_DIR / entry.RESULT_NAME).exists()
    assert not entry.RESULT_DIR.exists() or not list(entry.RESULT_DIR.glob("*"))
