"""Stage 1 v3 confirmatory 接线的 synthetic-fixture 测试。

**只用 synthetic fixture**：不读取任何真实 finalized 产物、不产生真实分析输出、
不写 unblinding log。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import mcq_conditional_increment as mci             # noqa: E402
from recover_attention.evaluation import v3_adapter_mcq as amcq            # noqa: E402
from recover_attention.evaluation import v3_arm_contrast as contrast       # noqa: E402
from recover_attention.evaluation import v3_grouping as grouping           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_stage1_v3_confirmatory", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
entry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = entry
_spec.loader.exec_module(entry)

K = 6
HD = 8


def _small_cfg(**over):
    """小预算 cfg：接线测试用，不改冻结语义（b_batch/b_per_batch 只影响耗时）。"""
    import dataclasses
    base = V3AnalysisConfig()
    return dataclasses.replace(base, **({"b_batch": 4, "b_per_batch": 5} | over))


# ── MCQ adapter fixture ────────────────────────────────────────────────────
def _mcq_record(i: int, *, eligible=True, letter="A", wrong=0):
    eid = f"cybermetric_{i:06d}"
    r = {"example_id": eid, "group_id": f"cybermetric_question_{i:016x}",
         "population_role": "confirmatory", "run_hash": "r" * 64,
         "schema_version": "stage1_mcq_completion_cache_v1",
         "sample_type": "greedy", "sample_index": 0,
         "eligible_for_primary": bool(eligible),
         "parsed_label": letter if eligible else None,
         "wrong_label": wrong if eligible else None,
         "semantic_output_text": f"option text {i}",
         "prompt_only_text": f"question {i} A) a B) b C) c D) d",
         # 下面这些**不得**进入任何 block
         "raw_completion": f"RAW {i}", "gold_label": "A", "bare_answer": True,
         "response_format": "letter_only"}
    for n in mci.MCQ_SURFACE_FEATURE_NAMES:
        r[n] = float(i % 5)
    for n in mci.MCQ_F5_FEATURE_NAMES:
        r[n] = float(i % 7) / 7.0
    return r


def _mcq_fixture(n=12, n_ineligible=1):
    recs = [_mcq_record(i, letter="ABCD"[i % 4], wrong=i % 2) for i in range(n)]
    for i in range(n_ineligible):
        recs[i] = _mcq_record(i, eligible=False)
    hidden = {r["example_id"]: np.full(HD, float(i)) for i, r in enumerate(recs)}
    return recs, hidden


def test_mcq_adapter_block_shape_and_sources():
    recs, hid = _mcq_fixture(n=12, n_ineligible=0)
    arm = amcq.build_mcq_arm(recs, hid)
    names = [b.name for b in arm["blocks"]]
    assert names == ["B1", "B2", "B3", "B4", "B5", "H"], "恰为 {B1..B5,H}，不得有 B6"
    kinds = {b.name: b.kind for b in arm["blocks"]}
    assert kinds == {"B1": "dense", "B2": "dense", "B3": "text",
                     "B4": "dense", "B5": "text", "H": "dense"}
    by = {b.name: b for b in arm["blocks"]}
    assert by["B1"].dense.shape == (12, 4)                    # A–D one-hot
    assert by["B2"].dense.shape == (12, len(mci.MCQ_SURFACE_FEATURE_NAMES))
    assert by["B4"].dense.shape == (12, len(mci.MCQ_F5_FEATURE_NAMES))
    assert by["B3"].text[0].startswith("option text")          # semantic_output_text
    assert by["B5"].text[0].startswith("question")             # prompt_only_text
    assert arm["label_field"] == "wrong_label" and arm["id_field"] == "example_id"
    assert set(np.unique(arm["y"])) <= {0, 1}


def test_mcq_adapter_uses_exact_frozen_feature_lists():
    assert list(mci.MCQ_SURFACE_FEATURE_NAMES) == [
        "surface_option_chars", "surface_option_words", "surface_option_has_digit",
        "surface_option_has_negation", "surface_option_position_index",
        "surface_option_length_rank"]
    recs, hid = _mcq_fixture(n=8, n_ineligible=0)
    arm = amcq.build_mcq_arm(recs, hid)
    assert arm["block_source_map"]["B2"] == list(mci.MCQ_SURFACE_FEATURE_NAMES)
    assert arm["block_source_map"]["B4"] == list(mci.MCQ_F5_FEATURE_NAMES)


@pytest.mark.parametrize("forbidden", ["raw_completion", "gold_label", "response_format",
                                       "bare_answer", "correctness"])
def test_mcq_adapter_rejects_forbidden_block_sources(forbidden):
    with pytest.raises(ValueError, match="禁用字段"):
        amcq.assert_no_forbidden_source({"B3": forbidden})


def test_mcq_adapter_refuses_non_analyzable_rows():
    """1,751 → 1,750 的真实形态：parse-failure 行不得被送进 adapter。"""
    recs, hid = _mcq_fixture(n=12, n_ineligible=1)
    with pytest.raises(ValueError, match="non-analyzable"):
        amcq.build_mcq_arm(recs, hid)
    kept = amcq.filter_analyzable(recs)
    assert len(kept) == 11
    arm = amcq.build_mcq_arm(kept, hid)
    assert arm["n"] == 11
    dropped = recs[0]["example_id"]
    assert dropped not in {r["example_id"] for r in kept}
    by = {b.name: b for b in arm["blocks"]}
    assert by["H"].dense.shape[0] == 11        # 不进 block / fold / OOF / bootstrap


def test_letter_onehot_parse_failure_is_all_zero_but_not_a_substitute():
    """纯函数行为保留；但它**不是**缺 hidden 的替代编码（上面那条测试锁死入口）。"""
    row = _mcq_record(3, eligible=False)
    assert np.all(mci.letter_onehot([row]) == 0.0)


def test_mcq_adapter_rejects_parse_failure_encoded_as_wrong_label_1():
    recs, hid = _mcq_fixture(n=6, n_ineligible=0)
    recs[0]["wrong_label"] = None
    with pytest.raises(ValueError, match="不是 0/1"):
        amcq.build_mcq_arm(recs, hid)


def test_mcq_adapter_rejects_missing_hidden():
    recs, hid = _mcq_fixture(n=6, n_ineligible=0)
    hid.pop(recs[0]["example_id"])
    with pytest.raises(ValueError, match="缺 hidden"):
        amcq.build_mcq_arm(recs, hid)


# ── E2 / E3 ───────────────────────────────────────────────────────────────
def _tid(i: int) -> str:
    """合法 ATT&CK technique id（T + 4 位数字）。"""
    return f"T{1000 + i:04d}"


def _h1_record(i, gid, idents, label=0):
    return {"trace_id": f"t{i:04d}", "group_id": gid, "label": label,
            "sample_type": "greedy" if i % K == 0 else "sampled",
            "mentions": [{"family": "attack", "normalized": x, "emitted": True}
                         for x in idents]}


def test_e2_and_e3_are_not_applicable_on_mcq():
    for fn, name in ((grouping.build_e2_groups, "E2"),):
        out = fn([], arm="mcq")
        assert out["estimability"] == grouping.NOT_APPLICABLE
        assert out["estimand"] == name
        assert out["performance_estimates_computed"] is False
    out3 = grouping.build_e3_groups([], V3AnalysisConfig(), arm="mcq")
    assert out3["estimability"] == grouping.NOT_APPLICABLE


def test_e2_collapsed_reports_structure_but_no_performance():
    """塌缩时**仍有结构性 accounting 数字**，但没有任何 performance estimate。"""
    recs = [_h1_record(i, f"p{i // K}", ["T1059"]) for i in range(24)]   # 共享 hub
    out = grouping.build_e2_groups(recs, arm="h1")
    assert out["estimability"] == grouping.NON_ESTIMABLE
    assert out["groups"] is None
    for k in ("component_count", "max_component_share", "hub_distribution",
              "n_completions", "n_identities", "n_identity_edges"):
        assert k in out, f"不可估时仍须输出结构性 accounting：缺 {k}"
    assert out["component_count"] >= 1 and out["max_component_share"] > 0
    assert out["hub_distribution"]["identity_degree_max"] >= 2
    assert out["performance_estimates_computed"] is False
    assert out["uses_label"] is False
    forbidden = {"auroc", "delta_h", "ci_lo", "ci_hi", "oof", "score"}
    assert not (forbidden & {k.lower() for k in out}), "不得出现任何性能估计字段"


def test_e2_estimable_when_graph_is_fragmented():
    # 门槛已收紧为 component_count >= 5 * n_outer_folds = 25，故需 >= 25 个独立分量。
    n = 30
    recs = [_h1_record(i, f"p{i}", [f"T{1000 + i:04d}"]) for i in range(n)]
    out = grouping.build_e2_groups(recs, arm="h1")
    assert out["component_count"] == n
    assert out["estimability"] == grouping.ESTIMABLE
    assert out["groups"] is not None and len(out["groups"]) == n


def test_e2_uses_all_identifiers_not_just_first():
    a = [_h1_record(0, "p0", ["T1001", "T1999"]), _h1_record(1, "p1", ["T1002", "T1999"])]
    out = grouping.build_e2_groups(a, arm="h1")
    assert out["n_identities"] == 3            # X1 / X2 / SHARED —— 不只第一个
    assert out["component_count"] == 1         # 经 SHARED 连通


def test_e3_empty_block_is_fail_closed_structure_only():
    """repooling 完成前**只查 fold 是否为空**；单类判定已移除（那要读 E1 label）。"""
    cfg = V3AnalysisConfig()
    recs = [_h1_record(i * K, f"p{i}", [_tid(i)], label=i % 2) for i in range(6)]
    blocks = {grouping.e3_block_of(_tid(i), cfg) for i in range(6)}
    missing = next(b for b in range(cfg.e3_n_blocks) if b not in blocks)
    out = grouping.build_e3_groups(recs, cfg, arm="h1", block_index=missing)
    assert out["fold_structure_status"] == grouping.EMPTY and "空 fold" in out["reason"]
    assert out["class_estimability"] == grouping.NOT_EVALUATED
    assert out["uses_label"] is False


def test_e3_block_assignment_is_deterministic_and_label_free():
    cfg = V3AnalysisConfig()
    assert grouping.e3_block_of("T1059", cfg) == grouping.e3_block_of("T1059", cfg)
    recs = [_h1_record(i * K, f"p{i}", [_tid(i)], label=i % 2) for i in range(10)]
    a = grouping.build_e3_groups(recs, cfg, arm="h1")
    for r in recs:
        r["label"] = 1 - r["label"]
    b = grouping.build_e3_groups(recs, cfg, arm="h1")
    assert a["prompts_per_block"] == b["prompts_per_block"], "分块必须 label-free"
    assert a["uses_label"] is False


# ── 两臂 contrast ──────────────────────────────────────────────────────────
def _arm_scores(n_groups, k, seed, signal):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], k)
    y = (rng.random(n_groups * k) < 0.5).astype(int)
    oof = [y * signal + rng.normal(size=y.size) for _ in range(3)]
    return y, groups, oof


def test_contrast_is_repeat_averaged_not_ensemble():
    y, g, oof = _arm_scores(10, 4, 1, 1.2)
    y2, g2, oof2 = _arm_scores(9, 4, 2, 0.3)
    got = contrast.d_bar(y, oof, y2, oof2)
    per = [contrast.s_per_repeat(y, [oof[r]])[0] - contrast.s_per_repeat(y2, [oof2[r]])[0]
           for r in range(3)]
    assert got == pytest.approx(float(np.mean(per)))
    ens = contrast.s_per_repeat(y, [np.mean(oof, axis=0)])[0] - \
        contrast.s_per_repeat(y2, [np.mean(oof2, axis=0)])[0]
    assert got != pytest.approx(ens), "不得先平均 OOF 再算 AUROC（那是 ensemble estimand）"


def test_contrast_direction_delta_and_provenance():
    cfg = _small_cfg()
    ym, gm, om = _arm_scores(8, 4, 3, 2.5)
    yh, gh, oh = _arm_scores(8, 4, 4, 2.5)
    out = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                              y_h1=yh, groups_h1=gh, oof_h1=oh, cfg=cfg)
    assert out["direction"] == "D = S_MCQ - S_H1"
    assert out["delta"] == 0.15 == contrast.DELTA_OBSERVABILITY
    assert out["contrast_protocol"] == "v3_repeat_averaged_independent_group_bootstrap"
    assert "v2.4 §6" in out["provenance"]["estimand_direction_delta"]
    assert "v3 M7" in out["provenance"]["repeat_and_bootstrap_execution"]
    assert out["paired"] is False
    assert out["precision_criterion_applies"] is False      # 0.04 不适用于 D
    assert out["b_total"] == cfg.b_batch * cfg.b_per_batch
    assert out["batch_seeds"] == [cfg.batch_seed(j) for j in range(cfg.b_batch)]


@pytest.mark.parametrize("verdict,sm,sh", [("h1_high_confidence", 3.0, 0.0),
                                           ("outcome_3", 0.0, 3.0)])
def test_contrast_verdicts(verdict, sm, sh):
    cfg = _small_cfg(b_batch=4, b_per_batch=25)
    ym, gm, om = _arm_scores(14, 4, 5, sm)
    yh, gh, oh = _arm_scores(14, 4, 6, sh)
    out = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                              y_h1=yh, groups_h1=gh, oof_h1=oh, cfg=cfg)
    assert out["verdict"] == verdict


def test_contrast_uncertain_when_ci_straddles():
    cfg = _small_cfg(b_batch=4, b_per_batch=25)
    ym, gm, om = _arm_scores(12, 4, 7, 1.0)
    yh, gh, oh = _arm_scores(12, 4, 8, 1.0)
    out = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                              y_h1=yh, groups_h1=gh, oof_h1=oh, cfg=cfg)
    assert out["verdict"] == "uncertain"


def test_contrast_resampling_is_independent_per_arm():
    """两臂抽样独立：只改 H1 的 group 结构，MCQ 的 draw 不应被强制配对。"""
    cfg = _small_cfg()
    ym, gm, om = _arm_scores(8, 4, 9, 1.5)
    yh1, gh1, oh1 = _arm_scores(8, 4, 10, 1.5)
    yh2, gh2, oh2 = _arm_scores(11, 3, 10, 1.5)          # 不同 group 数 → 不可能 paired
    a = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                            y_h1=yh1, groups_h1=gh1, oof_h1=oh1, cfg=cfg)
    b = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                            y_h1=yh2, groups_h1=gh2, oof_h1=oh2, cfg=cfg)
    assert a["n_valid_draws"] > 0 and b["n_valid_draws"] > 0
    assert a["paired"] is False and b["paired"] is False


def test_contrast_discards_single_class_rounds_not_score_zero():
    cfg = _small_cfg(b_batch=2, b_per_batch=5)
    ym = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    gm = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    om = [np.linspace(0, 1, 8) for _ in range(3)]
    yh, gh, oh = _arm_scores(6, 4, 11, 1.0)
    out = contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om,
                              y_h1=yh, groups_h1=gh, oof_h1=oh, cfg=cfg)
    assert out["n_valid_draws"] + out["n_discarded_single_class"] == out["b_total"]


def test_contrast_rejects_repeat_count_mismatch():
    cfg = _small_cfg()
    ym, gm, om = _arm_scores(6, 4, 12, 1.0)
    yh, gh, oh = _arm_scores(6, 4, 13, 1.0)
    with pytest.raises(ValueError, match="repeat 数不一致"):
        contrast.contrast_d(y_mcq=ym, groups_mcq=gm, oof_mcq=om[:2],
                            y_h1=yh, groups_h1=gh, oof_h1=oh, cfg=cfg)


# ── 入口：调用顺序与 fail-closed ────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _pass_hard_gate(monkeypatch):
    """本文件测的是**硬闸之后**的各项检查；把 readiness 重算 hook 放行。

    硬闸本身（未接线即无条件拒绝）由 tests/test_v3_addendum_hardgate.py 覆盖。
    """
    monkeypatch.setattr(entry, "recompute_readiness", lambda *a, **k: {"ok": True})


def _manifest(tmp_path, prod):
    from recover_attention.evaluation import v3_unblinding_boundary as ub
    mp = tmp_path / "manifest.json"
    if not mp.exists():
        ub.write_stage1_freeze_manifest(mp, prod)
    return mp


def _readiness(tmp_path, prod, *, ready=True, shas=None):
    arts = {n: {"sha256": (shas or {}).get(n, hashlib.sha256((prod / n).read_bytes()).hexdigest())}
            for n in entry.FINALIZED_SIX}
    p = tmp_path / "readiness.json"
    p.write_text(json.dumps({"freeze_chain": {"finalized_artifacts": arts},
                             "readiness": {"ready_for_unblinding": ready,
                                           "blocking_reasons": [] if ready else ["x"]}},
                            ensure_ascii=False), encoding="utf-8")
    return p


def _analysis_inputs(tmp_path, prod):
    """tmp 版的 analysis-input 冻结三件套（tail F5 + canonical repool + manifest）。

    正式 preflight 现在还要核 analysis-input freeze manifest —— 六项 finalized
    管不住 Path B 实际读的 tail F5 与 E3 实际读的 repool 向量。测试必须走同一道门，
    所以在 tmp 里造一份等价结构，而不是绕过去。
    """
    from recover_attention.evaluation import v3_unblinding_boundary as _ub
    tail = tmp_path / "tail"
    tail.mkdir(exist_ok=True)
    for n in _ub.TAIL_INPUT_FILES:
        (tail / n).write_bytes(n.encode())
    rep = tmp_path / "repool"
    (rep / _ub.REPOOL_INPUT_TREE).mkdir(parents=True, exist_ok=True)
    for n in _ub.REPOOL_INPUT_FILES:
        (rep / n).write_bytes(n.encode())
    (rep / _ub.REPOOL_INPUT_TREE / "a.npz").write_bytes(b"x")
    mp = tmp_path / "analysis_input_manifest.json"
    if not mp.exists():
        _ub.write_analysis_input_manifest(mp, prod_dir=prod, tail_dir=tail,
                                          repool_dir=rep)
    return {"analysis_input_manifest_path": mp, "tail_dir": tail, "repool_dir": rep}


def _prod(tmp_path):
    d = tmp_path / "prod"
    d.mkdir()
    for n in entry.FINALIZED_SIX:
        (d / n).write_bytes(n.encode())
    return d


def test_dry_run_never_records_unblinding(monkeypatch):
    called = []
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda *a, **k: called.append(1))
    for argv in (["--arm", "mcq"], ["--arm", "h1", "--dry-run"],
                 ["--arm", "mcq", "--estimand", "E2"]):
        assert entry.main(argv) == 0
    assert called == [], "--dry-run 与默认调用都不得写任何 unblinding 记录"


def test_execute_refused_when_readiness_false(tmp_path):
    prod = _prod(tmp_path)
    with pytest.raises(entry.ExecuteRefused, match="ready_for_unblinding"):
        entry.preflight_execute(prod_dir=prod,
                                readiness_path=_readiness(tmp_path, prod, ready=False),
                                result_dir=tmp_path / "res", unblind_log=tmp_path / "u.jsonl",
                                freeze_manifest_path=_manifest(tmp_path, prod))


def test_execute_refused_when_finalized_sha_drifts(tmp_path):
    prod = _prod(tmp_path)
    rp = _readiness(tmp_path, prod, shas={n: "0" * 64 for n in entry.FINALIZED_SIX})
    with pytest.raises(entry.ExecuteRefused, match="SHA 与 readiness 记录不符"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp,
                                result_dir=tmp_path / "res", unblind_log=tmp_path / "u.jsonl",
                                freeze_manifest_path=_manifest(tmp_path, prod))


def test_execute_refused_when_results_or_log_exist(tmp_path):
    prod = _prod(tmp_path)
    rp = _readiness(tmp_path, prod)
    res = tmp_path / "res"
    res.mkdir()
    (res / "x.json").write_text("{}", encoding="utf-8")
    with pytest.raises(entry.ExecuteRefused, match="真实结果目标已存在"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp, result_dir=res,
                                unblind_log=tmp_path / "u.jsonl",
                                freeze_manifest_path=_manifest(tmp_path, prod))
    log = tmp_path / "u.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    with pytest.raises(entry.ExecuteRefused, match="unblinding log 已存在"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp,
                                result_dir=tmp_path / "res2", unblind_log=log,
                                freeze_manifest_path=_manifest(tmp_path, prod))


def test_execute_call_order_unblinding_before_analysis(tmp_path, monkeypatch):
    """锁死顺序：preflight → record_unblinding → analyze（首次读 label）→ 写结果。"""
    prod = _prod(tmp_path)
    rp = _readiness(tmp_path, prod)
    order = []
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda *a, **k: order.append("record_unblinding"))
    out = entry.run_execute_all(
        cfg=V3AnalysisConfig(),
        analyze_factory=lambda: order.append("build_callback") or (
            lambda: order.append("analyze") or {"ok": True}),
        timestamp="2026-08-17T00:00:00Z", prod_dir=prod,
                                readiness_path=rp, result_dir=tmp_path / "res",
                                unblind_log=tmp_path / "u.jsonl",
                                freeze_manifest_path=_manifest(tmp_path, prod),
                                **_analysis_inputs(tmp_path, prod))
    assert order == ["record_unblinding", "build_callback", "analyze"], (
        "解盲日志必须先于**载入含 label 的 records**，更先于任何性能计算")
    assert Path(out["result_path"]).exists()


def test_unblinding_log_is_kept_when_analysis_fails(tmp_path, monkeypatch):
    prod = _prod(tmp_path)
    rp = _readiness(tmp_path, prod)
    log = tmp_path / "u.jsonl"
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda p, **k: Path(p).write_text("{}\n", encoding="utf-8"))
    def boom():
        raise RuntimeError("analysis exploded")
    with pytest.raises(RuntimeError, match="unblinding log 保留"):
        entry.run_execute_all(cfg=V3AnalysisConfig(), analyze_factory=lambda: boom,
                              timestamp="2026-08-17T00:00:00Z", prod_dir=prod,
                              readiness_path=rp, result_dir=tmp_path / "res",
                              unblind_log=log,
                              freeze_manifest_path=_manifest(tmp_path, prod),
                              **_analysis_inputs(tmp_path, prod))
    assert log.exists(), "已解盲就是已解盲：失败也不得删除日志"
    assert not (tmp_path / "res").exists()


def test_entrypoint_mcq_e2_e3_return_not_applicable():
    from recover_attention.evaluation import v3_synthetic_fixture as _fx
    for est in ("E2", "E3"):
        res = entry.run_dry(arm="mcq", estimand=est,
                            cfg=_fx.synthetic_cfg(b_batch=2, b_per_batch=5))
        assert res["estimability"] == grouping.NOT_APPLICABLE
        assert res["record_unblinding_called"] is False


@pytest.mark.pre_unblinding_workspace
def test_real_analysis_targets_still_absent():
    """本轮不得创建任何真实 analysis 输出或 unblinding log。"""
    assert not entry.RESULT_DIR.exists()
    assert not entry.UNBLIND_LOG.exists()
