"""correction sprint：E2 门槛、E3 状态分离、真实 synthetic dry-run、freeze manifest、
Stage 1 unblinding log、M4 schema/writer。

**只用 synthetic fixture**：不读真实 finalized 产物、不写真实分析输出或 unblinding log。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_grouping as grouping           # noqa: E402
from recover_attention.evaluation import v3_population_declaration as pdec  # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fixture   # noqa: E402
from recover_attention.evaluation import v3_unblinding_boundary as ub      # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_stage1_v3_confirmatory", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
entry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = entry
_spec.loader.exec_module(entry)


@pytest.fixture(autouse=True)
def _pass_hard_gate(monkeypatch):
    """放行 readiness 重算硬闸；本文件测的是它之后的检查。"""
    monkeypatch.setattr(entry, "recompute_readiness", lambda *a, **k: {"ok": True})


def _tid(i):
    return f"T{1000 + i:04d}"


def _rec(i, gid, idents, label=0, greedy=False):
    return {"trace_id": f"t{i:05d}", "group_id": gid, "label": label,
            "sample_type": "greedy" if greedy else "sampled",
            "mentions": [{"family": "attack", "normalized": x, "emitted": True}
                         for x in idents]}


# ══ 1. E2 门槛：> 0.20 且 component_count < 5 * n_outer_folds ═══════════════
def test_e2_min_components_is_five_times_outer_folds():
    cfg = V3AnalysisConfig()
    assert grouping.m2b_min_components(cfg) == 5 * cfg.n_outer_folds == 25
    assert grouping.M2B_MAX_COMPONENT_SHARE == 0.20


def _share_case(n_share_group: int, n_singletons: int):
    """构造：一个大分量占 n_share_group 行，其余各自独立。"""
    recs = [_rec(i, f"hub{i}", ["T9999"], greedy=True) for i in range(n_share_group)]
    recs += [_rec(1000 + i, f"solo{i}", [_tid(i)], greedy=True) for i in range(n_singletons)]
    return recs


def test_e2_share_exactly_at_threshold_is_not_collapsed():
    """share == 0.20 **不**塌缩（严格大于才塌缩）。"""
    recs = _share_case(20, 80)                       # 20/100 = 0.20
    out = grouping.build_e2_groups(recs, arm="h1")
    assert out["max_component_share"] == pytest.approx(0.20)
    assert out["component_count"] >= 25
    assert out["estimability"] == grouping.ESTIMABLE


def test_e2_share_slightly_above_threshold_collapses():
    recs = _share_case(21, 79)                       # 21/100 = 0.21 > 0.20
    out = grouping.build_e2_groups(recs, arm="h1")
    assert out["max_component_share"] > 0.20
    assert out["estimability"] == grouping.NON_ESTIMABLE
    assert "max_component_share" in out["reason"] or "component_count" in out["reason"]


@pytest.mark.parametrize("n_components,collapsed", [(24, True), (25, False)])
def test_e2_component_count_boundary_24_25(n_components, collapsed):
    """component_count < 25 塌缩；== 25 不塌缩。"""
    per = 4
    recs = [_rec(i * per + j, f"p{i}", [_tid(i)], greedy=(j == 0))
            for i in range(n_components) for j in range(per)]
    out = grouping.build_e2_groups(recs, arm="h1")
    assert out["component_count"] == n_components
    assert out["max_component_share"] == pytest.approx(per / (n_components * per))
    assert (out["estimability"] == grouping.NON_ESTIMABLE) is collapsed


def test_e2_thresholds_recorded_in_output():
    recs = _share_case(21, 79)
    th = grouping.build_e2_groups(recs, arm="h1")["thresholds"]
    assert th["max_component_share_strictly_greater_than"] == 0.20
    assert th["min_components"] == 25
    assert th["min_components_rule"] == "5 * cfg.n_outer_folds"


# ══ 2. E3：wiring status vs 统计 estimability 分离 ══════════════════════════
def test_e3_reports_needs_researcher_decision_not_estimability():
    recs, _h = fixture.make_h1(n_prompts=8)
    out = grouping.build_e3_groups(recs, V3AnalysisConfig(), arm="h1")
    assert out["wiring_status"] == grouping.NEEDS_DECISION
    assert out["class_estimability"] == grouping.NOT_EVALUATED   # **不冒充** estimable
    assert out["class_estimability"] not in (grouping.ESTIMABLE, grouping.NON_ESTIMABLE)
    assert "block-specific" in out["blocking_reason"]
    assert out["not_implemented"] == ["block-specific label", "block-specific H pooling"]
    assert "fold_structure_status" in out and "fold_status" not in out


def test_e3_fold_structure_status_still_fail_closed():
    cfg = V3AnalysisConfig()
    recs = [_rec(i * 4, f"p{i}", [_tid(i)], label=0, greedy=True) for i in range(6)]
    used = {grouping.e3_block_of(_tid(i), cfg) for i in range(6)}
    empty = next(b for b in range(cfg.e3_n_blocks) if b not in used)
    out = grouping.build_e3_groups(recs, cfg, arm="h1", block_index=empty)
    assert out["fold_structure_status"] == grouping.EMPTY and "空 fold" in out["reason"]
    assert out["class_estimability"] == grouping.NOT_EVALUATED  # 类别可估性始终未评估


def test_e3_mcq_still_not_applicable():
    out = grouping.build_e3_groups([], V3AnalysisConfig(), arm="mcq")
    assert out["estimability"] == grouping.NOT_APPLICABLE


# ══ 3. dry-run 是真实 synthetic end-to-end ═════════════════════════════════
def test_dry_run_returns_real_numbers_not_metadata():
    cfg = fixture.synthetic_cfg(b_batch=2, b_per_batch=5)
    r = entry.run_dry(arm="mcq", estimand="E1", cfg=cfg)
    a = r["arms"]["mcq"]
    assert a["synthetic"] is True and r["synthetic"] is True
    for k in ("delta_h_point", "delta_h_ci", "mc_jackknife", "precision", "non_inferiority"):
        assert k in a, f"dry-run 必须给真实结果结构，缺 {k}"
    assert isinstance(a["delta_h_point"], float)
    assert a["delta_h_ci"]["width"] >= 0
    assert a["precision"]["precision"] in ("adequate", "insufficient", "borderline")
    assert a["b_total"] == cfg.b_batch * cfg.b_per_batch     # **真实 fixture B**
    assert a["n_valid_draws"] + a["n_discarded_single_class"] == a["b_total"]
    assert r["record_unblinding_called"] is False


def test_dry_run_contrast_only_when_arm_both():
    cfg = fixture.synthetic_cfg(b_batch=2, b_per_batch=5)
    for arm in ("h1", "mcq"):
        r = entry.run_dry(arm=arm, estimand="E1", cfg=cfg)
        assert r["contrast_math_smoke"] is None
        assert "D 只在 --arm both" in r["contrast_note"]
    both = entry.run_dry(arm="both", estimand="E1", cfg=cfg)
    smoke = both["contrast_math_smoke"]
    assert smoke is not None and smoke["synthetic"] is True
    assert smoke["is_end_to_end"] is False           # **数学 smoke，不是 O 接线**
    assert smoke["verdict"] in ("h1_high_confidence", "outcome_3", "uncertain")
    assert set(both["arms"]) == {"h1", "mcq"}


def test_dry_run_writes_nothing_real(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'RESULT_DIR', tmp_path / 'confirmatory')
    monkeypatch.setattr(entry, 'UNBLIND_LOG', tmp_path / 'unblinding.jsonl')
    cfg = fixture.synthetic_cfg(b_batch=2, b_per_batch=5)
    entry.run_dry(arm="both", estimand="E1", cfg=cfg)
    assert not entry.RESULT_DIR.exists()
    assert not entry.UNBLIND_LOG.exists()


def test_synthetic_cfg_is_dry_run_mode_not_production():
    cfg = fixture.synthetic_cfg()
    assert cfg.mode == "dry_run"
    assert V3AnalysisConfig().mode == "production"


# ══ 4. 六项 freeze manifest ════════════════════════════════════════════════
def _prod(tmp_path, tag=b"x"):
    d = tmp_path / "prod"
    d.mkdir(exist_ok=True)
    for n in ub.FINALIZED_SIX:
        (d / n).write_bytes(n.encode() + tag)
    return d


def test_manifest_hash_covers_only_relpath_bytes_sha():
    e = [{"relative_path": "b", "bytes": 2, "sha256": "bb"},
         {"relative_path": "a", "bytes": 1, "sha256": "aa"}]
    same_but_extra = [{**x, "mtime": 12345, "abs_path": "/tmp/x"} for x in e]
    assert ub.freeze_hash_of(sorted(e, key=lambda x: x["relative_path"])) == \
        ub.freeze_hash_of(sorted(same_but_extra, key=lambda x: x["relative_path"]))


def test_manifest_entries_are_sorted_and_hash_is_order_independent(tmp_path):
    prod = _prod(tmp_path)
    m = ub.build_stage1_freeze_manifest(prod)
    paths = [e["relative_path"] for e in m["entries"]]
    assert paths == sorted(paths) and len(paths) == 6
    assert m["freeze_hash"] == ub.freeze_hash_of(list(reversed(m["entries"]))[::-1])


def test_manifest_first_write_is_atomic_then_noop(tmp_path):
    prod = _prod(tmp_path)
    path = tmp_path / "manifest.json"
    a = ub.write_stage1_freeze_manifest(path, prod)
    assert a["written"] is True and path.exists()
    before = path.read_bytes()
    b = ub.write_stage1_freeze_manifest(path, prod)
    assert b["written"] is False and "no-op" in b["reason"]
    assert path.read_bytes() == before, "已存在且一致时必须逐字节不变"


def test_manifest_refuses_overwrite_when_artifacts_changed(tmp_path):
    prod = _prod(tmp_path)
    path = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(path, prod)
    before = path.read_bytes()
    (prod / ub.FINALIZED_SIX[0]).write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="禁止覆盖"):
        ub.write_stage1_freeze_manifest(path, prod)
    assert path.read_bytes() == before


def test_manifest_verify_detects_drift(tmp_path):
    prod = _prod(tmp_path)
    m = ub.build_stage1_freeze_manifest(prod)
    assert ub.verify_stage1_freeze_manifest(m, prod)["ok"] is True
    (prod / ub.FINALIZED_SIX[1]).write_bytes(b"CHANGED")
    v = ub.verify_stage1_freeze_manifest(m, prod)
    assert v["ok"] is False and v["drift"]


def test_manifest_does_not_touch_stage0_builder():
    from recover_attention.evaluation import unblinding as stage0
    import inspect
    sig = list(inspect.signature(stage0.build_freeze_manifest).parameters)
    assert sig == ["feature_dir", "arms"], "Stage 0 builder 签名不得改动"
    assert stage0.LOG_SCHEMA != ub.STAGE1_LOG_SCHEMA


# ══ 5. Stage 1 unblinding log ══════════════════════════════════════════════
def test_stage1_log_schema_and_fields(tmp_path):
    log = tmp_path / "u.jsonl"
    e = ub.record_stage1_unblinding(log, input_freeze_hash="f" * 64, prereg_sha256="p" * 64,
                                    readiness_sha256="r" * 64, config_hash="c" * 64,
                                    timestamp="2026-08-17T00:00:00Z")
    assert e["schema_version"] == "stage1_v3_unblinding_log_v1"
    on_disk = json.loads(log.read_text(encoding="utf-8").strip())
    assert on_disk == e
    for f in ("input_freeze_hash", "prereg_sha256", "readiness_sha256", "config_hash"):
        assert f in on_disk
    assert on_disk["input_freeze_hash"] != on_disk["prereg_sha256"], "两者不得混用同一位"


def test_stage1_log_is_exclusive_create(tmp_path):
    log = tmp_path / "u.jsonl"
    ub.record_stage1_unblinding(log, input_freeze_hash="f" * 64, prereg_sha256="p" * 64,
                                readiness_sha256="r" * 64, config_hash="c" * 64,
                                timestamp="t1")
    before = log.read_bytes()
    with pytest.raises(FileExistsError, match="只允许一次"):
        ub.record_stage1_unblinding(log, input_freeze_hash="f" * 64, prereg_sha256="p" * 64,
                                    readiness_sha256="r" * 64, config_hash="c" * 64,
                                    timestamp="t2")
    assert log.read_bytes() == before


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


def test_entry_passes_manifest_hash_not_prereg_sha(tmp_path, monkeypatch):
    prod = _prod(tmp_path)
    mpath = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(mpath, prod)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    arts = {n: {"sha256": hashlib.sha256((prod / n).read_bytes()).hexdigest()}
            for n in ub.FINALIZED_SIX}
    rp = tmp_path / "readiness.json"
    rp.write_text(json.dumps({"freeze_chain": {"finalized_artifacts": arts},
                              "readiness": {"ready_for_unblinding": True}}), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda p, **k: seen.update(k) or {"ok": True})
    ai = _analysis_inputs(tmp_path, prod)
    entry.run_execute_all(cfg=V3AnalysisConfig(),
                          analyze_factory=lambda: (lambda: {"synthetic": False,
                                                            "ok": True}),
                          timestamp="2026-08-17T00:00:00Z", prod_dir=prod,
                          readiness_path=rp, result_dir=tmp_path / "res",
                          unblind_log=tmp_path / "u.jsonl",
                          freeze_manifest_path=mpath, **ai)
    aman = json.loads(ai["analysis_input_manifest_path"].read_text(encoding="utf-8"))
    assert seen["input_freeze_hash"] == man["freeze_hash"]
    # log v2：两个 freeze hash 同时在场，且互不相同
    assert seen["analysis_input_freeze_hash"] == aman["analysis_input_freeze_hash"]
    assert seen["analysis_input_freeze_hash"] != seen["input_freeze_hash"]
    from recover_attention.evaluation.v3_freeze import check_v3_frozen
    assert seen["prereg_sha256"] == check_v3_frozen().current_sha256
    assert seen["input_freeze_hash"] != seen["prereg_sha256"]


def test_execute_refused_when_manifest_missing_or_drifted(tmp_path):
    prod = _prod(tmp_path)
    arts = {n: {"sha256": hashlib.sha256((prod / n).read_bytes()).hexdigest()}
            for n in ub.FINALIZED_SIX}
    rp = tmp_path / "readiness.json"
    rp.write_text(json.dumps({"freeze_chain": {"finalized_artifacts": arts},
                              "readiness": {"ready_for_unblinding": True}}), encoding="utf-8")
    with pytest.raises(entry.ExecuteRefused, match="freeze manifest 不存在"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp, result_dir=tmp_path / "r1",
                                unblind_log=tmp_path / "u1.jsonl",
                                freeze_manifest_path=tmp_path / "nope.json")
    mpath = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(mpath, prod)
    man = json.loads(mpath.read_text(encoding="utf-8"))
    man["entries"][0]["sha256"] = "0" * 64
    mpath.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(entry.ExecuteRefused, match="freeze manifest 校验失败"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp, result_dir=tmp_path / "r2",
                                unblind_log=tmp_path / "u2.jsonl", freeze_manifest_path=mpath)


# ══ 6. M4 schema / writer（不提前写真实 null 产物）══════════════════════════
def test_m4_declaration_marks_label_fields_pending():
    d = pdec.build_declaration(h1={"finalized": 9264, "analyzable": 8664, "n_positive": 999},
                               mcq={"finalized": 1751, "analyzable": 1750, "n_positive": 111})
    pdec.validate_declaration(d)
    assert d["label_dependent_status"] == pdec.PENDING
    assert d["performance_blind"] is True
    for arm in ("h1", "mcq"):
        for f in pdec.LABEL_DEPENDENT_FIELDS:
            assert d["arms"][arm][f] == pdec.PENDING, "未解盲时不得写真实 label 派生值"
    assert d["arms"]["h1"]["analyzable"] == 8664
    pdec.assert_no_label_leak(d)


def test_m4_declaration_leak_guard_trips():
    d = pdec.build_declaration(h1={"analyzable": 1}, mcq={"analyzable": 1})
    d["arms"]["h1"]["n_positive"] = 123
    with pytest.raises(ValueError, match="含真实 label 派生值"):
        pdec.assert_no_label_leak(d)


def test_m4_writer_roundtrip_synthetic(tmp_path):
    d = pdec.build_declaration(h1={"analyzable": 3}, mcq={"analyzable": 4})
    info = pdec.write_declaration(tmp_path / "decl.json", d)
    assert info["label_dependent_status"] == pdec.PENDING
    back = json.loads((tmp_path / "decl.json").read_text(encoding="utf-8"))
    assert back == d
    assert "runtime_technical_exclusions" in back["attrition_categories"]
    assert "canonicalization_parse_attrition" in back["attrition_categories"]


def test_no_real_declaration_artifact_written_this_round():
    """本轮只接 schema/writer，**不提前写真实 null 产物**。"""
    assert not (_REPO_ROOT / "outputs/stage1/readiness"
                / "stage1_population_declaration.json").exists()
