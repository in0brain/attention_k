"""Path B 换成 tail 实现 + analysis-input 冻结 + 一次性解盲入口的单测。

**不解盲、不读 label、不写真实产物。** 所有可写路径都在 tmp_path 下。

覆盖研究者裁定的七项：

    1  Path B loader 严格读 tail F5 / report / fingerprint
    2  analysis-input freeze manifest 绑定 finalized + tail F5 + canonical repool
    3  readiness v3 存在且 v2 保留不覆盖
    4  unblinding log 同时记录两个 freeze hash
    5  入口只有一次性 --execute-all
    6  旧 shadow 明确 superseded，且**不得**接 label
    7  路径仍然恰好三条，不新增第四条
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

from recover_attention.evaluation import v3_analysis_paths as vpaths  # noqa: E402
from recover_attention.evaluation import v3_analyze as vanalyze  # noqa: E402
from recover_attention.evaluation import v3_readiness as vreadiness  # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow_f5  # noqa: E402
from recover_attention.evaluation import v3_shadow_overlay as vov  # noqa: E402
from recover_attention.evaluation import v3_unblinding_boundary as ub  # noqa: E402

_REAL = _REPO_ROOT / "outputs" / "stage1"


def load_entry():
    spec = importlib.util.spec_from_file_location(
        "_v3_entry", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def entry():
    return load_entry()


# ══════════════════════════════════════════════ 7  路径仍然恰好三条
def test_exactly_three_frozen_paths_no_fourth_variant():
    assert vpaths.FROZEN_PATHS == (vpaths.PATH_A, vpaths.PATH_B, vpaths.PATH_C)
    assert len(vpaths.FROZEN_PATHS) == 3
    with pytest.raises(vpaths.PathViolation):
        vpaths.assert_frozen_paths(vpaths.FROZEN_PATHS + ("tail_stabilized_extra",))


def test_path_b_name_unchanged_only_its_implementation_moved():
    """B 的**名字**没变（否则就是新增第四条）；变的是唯一实现。"""
    assert vpaths.PATH_B == "stable_shadow_sensitivity"
    assert vov.ACCEPTED_OVERLAY_SOURCES == (vov.OVERLAY_SOURCE_TAIL,)
    assert vov.OVERLAY_SOURCE_LEGACY_SHADOW not in vov.ACCEPTED_OVERLAY_SOURCES


# ══════════════════════════════════════════════ 6  旧 shadow superseded
def test_legacy_shadow_marker_requires_all_three_status_labels(tmp_path):
    d = tmp_path / "shadow"
    d.mkdir()
    (d / "h1_shadow_f5.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="未登记为 superseded"):
        vov.assert_legacy_shadow_superseded(d)

    doc = vov.build_legacy_shadow_marker(d, tmp_path / "tail_stabilized")
    assert set(doc["status"]) == set(vov.LEGACY_SHADOW_STATUS)
    (d / vov.LEGACY_SHADOW_MARKER_NAME).write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert vov.assert_legacy_shadow_superseded(d)["status"] == list(
        vov.LEGACY_SHADOW_STATUS)

    for drop in vov.LEGACY_SHADOW_STATUS:
        bad = {**doc, "status": [s for s in doc["status"] if s != drop]}
        (d / vov.LEGACY_SHADOW_MARKER_NAME).write_text(
            json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(vov.OverlayRefused, match="缺状态标签"):
            vov.assert_legacy_shadow_superseded(d)


def test_legacy_shadow_overlay_can_never_reach_labels():
    """即便有人直接调旧 loader，analyze 回调也必须拒绝。"""
    with pytest.raises(vov.OverlayRefused, match="不在被接受集合"):
        vanalyze.make_analyze_callback(
            cfg=None, h1_records=[], h1_hidden={}, mcq_records=[], mcq_hidden={},
            shadow_overlay={"gate_passed": True,
                            "overlay_source": vov.OVERLAY_SOURCE_LEGACY_SHADOW},
            e3_blocks={"canonical_source": "repool"})


@pytest.mark.local_artifacts('outputs/stage1/shadow/shadow_fingerprint.json',
                             'outputs/stage1/shadow/h1_shadow_f5.jsonl',
                             'outputs/stage1/shadow/mcq_shadow_f5.jsonl',
                             'outputs/stage1/shadow/shadow_completion_report.json')
def test_real_legacy_shadow_is_marked_and_its_files_are_retained():
    doc = vov.assert_legacy_shadow_superseded(_REAL / "shadow")
    assert doc["superseded_by"]["overlay_source"] == vov.OVERLAY_SOURCE_TAIL
    retained = set(doc["retained_files"])
    for n in ("h1_shadow_f5.jsonl", "mcq_shadow_f5.jsonl",
              "shadow_completion_report.json", "shadow_fingerprint.json"):
        assert n in retained, "旧 shadow 产物必须保留，不删除"
        assert (_REAL / "shadow" / n).exists()


# ══════════════════════════════════════════════ 2  E3 固定 canonical repool
def test_e3_must_be_canonical_repool_not_the_tail_cache():
    with pytest.raises(vov.OverlayRefused, match="canonical repool"):
        vanalyze.make_analyze_callback(
            cfg=None, h1_records=[], h1_hidden={}, mcq_records=[], mcq_hidden={},
            shadow_overlay={"gate_passed": True,
                            "overlay_source": vov.OVERLAY_SOURCE_TAIL},
            e3_blocks={"canonical_source": "tail_stabilized"})


# ══════════════════════════════════════════════ 1  tail loader 硬门
def _tail_row(arm: str, uid: str) -> dict:
    stable = ({n: (-1.0 if "logprob" in n or "entropy" in n else 1)
               for n in shadow_f5.H1_STABLE_FIELDS} if arm == "h1"
              else {n: 1.0 for n in shadow_f5.MCQ_STABLE_FIELDS})
    if arm == "h1":
        stable["f5_completion_token_count"] = 3
        stable["f5_first_id_token_rank"] = 1
    names = (shadow_f5.H1_FROZEN_PASSTHROUGH if arm == "h1"
             else shadow_f5.MCQ_FROZEN_PASSTHROUGH)
    key = "trace_id" if arm == "h1" else "example_id"
    return {"schema": shadow_f5.SHADOW_SCHEMA, "arm": arm, key: uid,
            "example_id": uid,
            "interpretation": shadow_f5.SENSITIVITY_LABEL,
            "deviation_class": shadow_f5.DEVIATION_LABEL,
            "variant": vov.OVERLAY_SOURCE_TAIL,
            "tail_schema": "stage1_tail_stabilized_v1",
            "not_primary": "x", "stable_fields": stable,
            "frozen_passthrough": {"values": {}, "field_names": list(names),
                                   "source": "frozen（未重建、未修改）"},
            "provenance": {"path": vov.OVERLAY_SOURCE_TAIL}}


@pytest.fixture
def tail_env(tmp_path):
    """最小的 production / repool / tail 三件套。"""
    prod, rep, tail = tmp_path / "prod", tmp_path / "repool", tmp_path / "tail"
    for d in (prod, rep, tail):
        d.mkdir()
    (rep / "h1_e3_block_hidden").mkdir()
    (rep / "h1_e3_block_hidden" / "a.npz").write_bytes(b"x")
    for n in ub.REPOOL_INPUT_FILES:
        (rep / n).write_bytes(n.encode())

    h1_rows = [{"example_id": "e0", "sample_type": "greedy", "sample_index": 0,
                "trace_id": "t0", "id_token_positions_local": [0]}]
    mcq_rows = [{"example_id": "m0", "eligible_for_primary": True}]
    (prod / "h1_completion_records.jsonl").write_text(
        json.dumps(h1_rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    (prod / "mcq_completion_records.jsonl").write_text(
        json.dumps(mcq_rows[0], ensure_ascii=False) + "\n", encoding="utf-8")

    (tail / "h1_tail_f5.jsonl").write_text(
        json.dumps(_tail_row("h1", "t0"), ensure_ascii=False) + "\n", encoding="utf-8")
    (tail / "mcq_tail_f5.jsonl").write_text(
        json.dumps(_tail_row("mcq", "m0"), ensure_ascii=False) + "\n", encoding="utf-8")
    (tail / "tail_completion_report.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8")
    freeze = {"production/h1_completion_records.jsonl":
              hashlib.sha256((prod / "h1_completion_records.jsonl").read_bytes()).hexdigest(),
              "repool/h1_e3_block_hidden__tree_digest":
              vov.tree_digest(rep / "h1_e3_block_hidden")}
    (tail / "tail_fingerprint.json").write_text(
        json.dumps({"variant": vov.OVERLAY_SOURCE_TAIL, "source_freeze": freeze}),
        encoding="utf-8")
    return {"prod": prod, "repool": rep, "tail": tail}


def test_tail_overlay_happy_path(tail_env):
    ov = vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])
    assert ov["overlay_source"] == vov.OVERLAY_SOURCE_TAIL
    assert ov["gate_passed"] is True
    assert ov["counts"] == {"h1": 1, "mcq": 1}
    assert ov["source_freeze_check"]["n_checked"] == 2


def test_tail_overlay_refuses_incomplete_report(tail_env):
    (tail_env["tail"] / "tail_completion_report.json").write_text(
        json.dumps({"complete": False}), encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="complete != true"):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


def test_tail_overlay_refuses_source_freeze_drift(tail_env):
    """源文件或 E3 目录被动过 → 拒绝读取。"""
    (tail_env["prod"] / "h1_completion_records.jsonl").write_text(
        '{"example_id": "e0", "sample_type": "greedy", "sample_index": 0, '
        '"trace_id": "t0", "id_token_positions_local": [0], "x": 1}\n', encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="source_freeze 与现场不一致"):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


def test_tail_overlay_refuses_wrong_variant_rows(tail_env):
    row = {**_tail_row("h1", "t0"), "variant": "full_bf16_shadow"}
    (tail_env["tail"] / "h1_tail_f5.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="variant="):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


def test_tail_overlay_refuses_key_set_mismatch(tail_env):
    row = _tail_row("mcq", "GHOST")
    with (tail_env["tail"] / "mcq_tail_f5.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(vov.OverlayRefused, match="集合不等于 analyzable"):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


def test_tail_overlay_refuses_nonfinite_stable_field(tail_env):
    row = _tail_row("mcq", "m0")
    row["stable_fields"]["f5_label_margin"] = float("nan")
    (tail_env["tail"] / "mcq_tail_f5.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="非有限"):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


def test_tail_overlay_refuses_wrong_fingerprint_variant(tail_env):
    fp = json.loads((tail_env["tail"] / "tail_fingerprint.json").read_text("utf-8"))
    fp["variant"] = "something_else"
    (tail_env["tail"] / "tail_fingerprint.json").write_text(json.dumps(fp),
                                                            encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="variant="):
        vov.load_tail_overlay(tail_env["tail"], tail_env["prod"], tail_env["repool"])


# ══════════════════════════════════════════════ 2  analysis-input manifest
@pytest.fixture
def ai_env(tmp_path):
    prod, tail, rep = tmp_path / "p", tmp_path / "t", tmp_path / "r"
    for d in (prod, tail, rep):
        d.mkdir()
    (rep / ub.REPOOL_INPUT_TREE).mkdir()
    for n in ub.FINALIZED_SIX:
        (prod / n).write_bytes(n.encode())
    for n in ub.TAIL_INPUT_FILES:
        (tail / n).write_bytes(n.encode())
    for n in ub.REPOOL_INPUT_FILES:
        (rep / n).write_bytes(n.encode())
    (rep / ub.REPOOL_INPUT_TREE / "a.npz").write_bytes(b"aa")
    return {"prod_dir": prod, "tail_dir": tail, "repool_dir": rep}


def test_analysis_input_manifest_binds_all_three_families(ai_env):
    man = ub.build_analysis_input_manifest(**ai_env)
    rps = [e["relative_path"] for e in man["entries"]]
    assert man["n_entries"] == len(ub.FINALIZED_SIX) + len(ub.TAIL_INPUT_FILES) \
        + len(ub.REPOOL_INPUT_FILES) + 1
    for n in ub.FINALIZED_SIX:
        assert f"production/{n}" in rps
    for n in ub.TAIL_INPUT_FILES:
        assert f"tail_stabilized/{n}" in rps
    for n in ub.REPOOL_INPUT_FILES:
        assert f"repool/{n}" in rps
    assert f"repool/{ub.REPOOL_INPUT_TREE}/" in rps          # E3 向量目录树摘要
    assert man["path_b_overlay_source"] == vov.OVERLAY_SOURCE_TAIL
    assert man["e3_canonical_source"] == "repool"
    assert rps == sorted(rps)


@pytest.mark.parametrize("mutate", [
    lambda env: (env["tail_dir"] / "h1_tail_f5.jsonl").write_bytes(b"CHANGED"),
    lambda env: (env["prod_dir"] / "h1_hidden.npz").write_bytes(b"CHANGED"),
    lambda env: (env["repool_dir"] / ub.REPOOL_INPUT_TREE / "a.npz").write_bytes(b"bb"),
    lambda env: (env["repool_dir"] / ub.REPOOL_INPUT_TREE / "b.npz").write_bytes(b"cc"),
])
def test_analysis_input_manifest_detects_drift_anywhere(ai_env, mutate):
    man = ub.build_analysis_input_manifest(**ai_env)
    assert ub.verify_analysis_input_manifest(man, **ai_env)["ok"] is True
    mutate(ai_env)
    ver = ub.verify_analysis_input_manifest(man, **ai_env)
    assert ver["ok"] is False and (ver["drift"] or not ver["hash_matches"])


def test_analysis_input_manifest_is_create_only(ai_env, tmp_path):
    p = tmp_path / "ai.json"
    a = ub.write_analysis_input_manifest(p, **ai_env)
    assert a["written"] is True
    b = ub.write_analysis_input_manifest(p, **ai_env)
    assert b["written"] is False                              # 完全一致 → no-op
    (ai_env["tail_dir"] / "h1_tail_f5.jsonl").write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="禁止覆盖"):
        ub.write_analysis_input_manifest(p, **ai_env)


def test_analysis_input_manifest_rejects_malformed_input(ai_env):
    for bad in (None, [], {}, {"schema_version": "x"},
                {"schema_version": ub.ANALYSIS_INPUT_SCHEMA,
                 "hash_caliber": ub.ANALYSIS_INPUT_HASH_CALIBER,
                 "analysis_input_freeze_hash": "zz", "entries": []}):
        ver = ub.verify_analysis_input_manifest(bad, **ai_env)
        assert ver["ok"] is False


def test_analysis_input_hash_differs_from_finalized_hash(ai_env, tmp_path):
    fin = ub.build_stage1_freeze_manifest(ai_env["prod_dir"])
    ana = ub.build_analysis_input_manifest(**ai_env)
    assert fin["freeze_hash"] != ana["analysis_input_freeze_hash"]


# ══════════════════════════════════════════════ 4  unblinding log 双 hash
def test_unblinding_log_records_both_freeze_hashes(tmp_path):
    log = tmp_path / "u.jsonl"
    entry = ub.record_stage1_unblinding(
        log, input_freeze_hash="f" * 64, analysis_input_freeze_hash="a" * 64,
        prereg_sha256="p" * 64, readiness_sha256="r" * 64, config_hash="c" * 64,
        timestamp="2026-08-20T06:00:00Z")
    assert entry["schema_version"] == ub.STAGE1_LOG_SCHEMA_V2
    assert entry["finalized_freeze_hash"] == "f" * 64
    assert entry["analysis_input_freeze_hash"] == "a" * 64
    on_disk = json.loads(log.read_text(encoding="utf-8").strip())
    assert on_disk["analysis_input_freeze_hash"] == "a" * 64
    with pytest.raises(FileExistsError):                      # 一次性解盲仍只允许一次
        ub.record_stage1_unblinding(
            log, input_freeze_hash="f" * 64, analysis_input_freeze_hash="a" * 64,
            prereg_sha256="p" * 64, readiness_sha256="r" * 64, config_hash="c" * 64,
            timestamp="2026-08-20T06:00:01Z")


# ══════════════════════════════════════════════ 5  一次性 --execute-all
def test_entry_has_only_the_single_shot_execute_all(entry):
    src = (_REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py").read_text("utf-8")
    assert "--execute-all" in src
    assert hasattr(entry, "run_execute_all")
    assert not hasattr(entry, "run_execute"), "按 arm/estimand 分次的旧入口必须移除"
    assert "def run_execute(" not in src
    assert entry.RESULT_NAME == "stage1_confirmatory_result.json"


def test_cli_rejects_the_old_execute_flag(entry):
    """`--execute` 不得被 argparse 当成 `--execute-all` 的前缀而静默接受。"""
    src = (_REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py").read_text("utf-8")
    assert "allow_abbrev=False" in src
    with pytest.raises(SystemExit):
        entry.main(["--arm", "both", "--execute"])
    with pytest.raises(SystemExit):
        entry.main(["--arm", "both", "--confirm-unblinding", "--timestamp", "t"])


def test_path_b_marker_names_its_implementation_not_a_generic_label():
    """结果文件只凭 `path_markers` 就必须能把 tail 实现与已废止的 full-BF16 shadow 区分开。"""
    m = vpaths.path_markers(vpaths.PATH_B)
    src = m["b4_source"]
    assert vov.OVERLAY_SOURCE_TAIL in src
    for token in ("两臂对称", "block 19", "外部 float32 lm_head/logsumexp",
                  "frozen passthrough"):
        assert token in src, token
    # 泛称不够：不能只写 "stable-logit shadow overlay"
    assert src != "stable-logit shadow overlay（两臂对称）+ frozen passthrough"
    assert m["marker"] == "protocol_deviation_numerical_sensitivity"


def test_path_b_marker_does_not_claim_bf16_throughout():
    """**不得**声称尾部全程 BF16 —— 真实 bitsandbytes 行为由 fingerprint caveat 说明。"""
    m = vpaths.path_markers(vpaths.PATH_B)
    blob = json.dumps(m, ensure_ascii=False)
    for banned in ("全程 bfloat16", "全程 bf16", "全程 BF16", "全 bf16"):
        assert banned not in m["b4_source"], banned
    assert "b4_source_caveat" in m
    assert "MatMul8bitLt" in m["b4_source_caveat"]
    assert "float16" in m["b4_source_caveat"]
    assert "tail_fingerprint.json" in blob


def test_only_path_b_carries_the_implementation_caveat():
    for p in (vpaths.PATH_A, vpaths.PATH_C):
        assert "b4_source_caveat" not in vpaths.path_markers(p)


@pytest.mark.local_artifacts('outputs/stage1/tail_stabilized/tail_fingerprint.json')
def test_real_tail_fingerprint_still_documents_the_backend_caveat():
    fp = json.loads((_REAL / "tail_stabilized" / "tail_fingerprint.json")
                    .read_text(encoding="utf-8"))
    caveat = fp["backend"]["caveat"]
    assert "MatMul8bitLt" in caveat and "float16" in caveat
    assert "不得" in caveat


# ══════════════════════════════════ 解盲日志先于**载入** label 记录
def test_analyze_callback_is_built_only_after_the_unblinding_log(entry, tmp_path,
                                                                 monkeypatch):
    """构造回调会把含 label 字段的 records 读进内存，因此它必须在写日志之后。"""
    order = []
    monkeypatch.setattr(entry, "preflight_execute",
                        lambda **kw: {"input_freeze_hash": "f" * 64,
                                      "analysis_input_freeze_hash": "a" * 64,
                                      "prereg_sha256": "p" * 64,
                                      "readiness_sha256": "r" * 64})
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda *a, **k: order.append("record_unblinding"))
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    out = entry.run_execute_all(
        cfg=V3AnalysisConfig(),
        analyze_factory=lambda: order.append("load_records") or (
            lambda: order.append("analyze") or {"ok": True}),
        timestamp="2026-08-20T06:00:00Z", result_dir=tmp_path / "res",
        unblind_log=tmp_path / "u.jsonl")
    assert order == ["record_unblinding", "load_records", "analyze"]
    assert Path(out["result_path"]).name == entry.RESULT_NAME


def test_factory_failure_keeps_the_unblinding_log(entry, tmp_path, monkeypatch):
    """工厂本身失败也不得假装尚未解盲。"""
    monkeypatch.setattr(entry, "preflight_execute",
                        lambda **kw: {"input_freeze_hash": "f" * 64,
                                      "analysis_input_freeze_hash": "a" * 64,
                                      "prereg_sha256": "p" * 64,
                                      "readiness_sha256": "r" * 64})
    log = tmp_path / "u.jsonl"
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda p, **k: Path(p).write_text("{}\n", encoding="utf-8"))

    def boom_factory():
        raise RuntimeError("overlay 加载炸了")

    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    with pytest.raises(RuntimeError, match="unblinding log 保留"):
        entry.run_execute_all(cfg=V3AnalysisConfig(), analyze_factory=boom_factory,
                              timestamp="2026-08-20T06:00:00Z",
                              result_dir=tmp_path / "res", unblind_log=log)
    assert log.exists()
    assert not (tmp_path / "res").exists()


def test_entry_no_longer_builds_the_callback_before_the_log(entry):
    src = (_REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py").read_text("utf-8")
    import inspect
    assert "analyze_factory" in inspect.signature(entry.run_execute_all).parameters
    assert "analyze" not in inspect.signature(entry.run_execute_all).parameters
    body = src[src.index("def run_execute_all("):src.index("def json_safe(")]
    # 找的是**实际调用行**，不是 docstring 里的提及
    assert body.index("ub.record_stage1_unblinding") <         body.index("analyze = analyze_factory()")


def test_arm_and_estimand_only_affect_dry_run(entry):
    src = (_REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py").read_text("utf-8")
    assert "只影响 --dry-run" in src
    import inspect
    sig = inspect.signature(entry.run_execute_all)
    assert "arm" not in sig.parameters and "estimand" not in sig.parameters


# ══════════════════════════════════════════════ 3  readiness v3
@pytest.mark.local_artifacts('outputs/stage1/readiness/stage1_confirmatory_readiness_v2.json')
def test_readiness_v3_schema_and_v2_preserved(entry):
    assert vreadiness.READINESS_V3_SCHEMA == "stage1_confirmatory_readiness_v3"
    assert vreadiness.READINESS_SCHEMA == "stage1_confirmatory_readiness_v2"
    assert hasattr(vreadiness, "recompute"), "v2 builder 必须保留，v2 文档仍可复核"
    v2 = _REAL / "readiness" / "stage1_confirmatory_readiness_v2.json"
    assert v2.exists()
    assert json.loads(v2.read_text(encoding="utf-8"))["schema_version"] == \
        vreadiness.READINESS_SCHEMA


@pytest.mark.pre_unblinding_workspace
@pytest.mark.local_artifacts('outputs/stage1/readiness/stage1_confirmatory_readiness_v3.json')
def test_real_readiness_v3_is_ready_and_recomputes_identically(entry):
    """磁盘 v3 必须与现场重算逐项一致 —— 入口不信任磁盘布尔值。"""
    doc = vreadiness.recompute_and_compare_v3(
        entry.READINESS_V3, prod_dir=entry.PROD, tail_dir=entry.TAIL_DIR,
        shadow_dir=entry.SHADOW_DIR, repool_dir=entry.REPOOL_DIR,
        result_dir=entry.RESULT_DIR, unblind_log=entry.UNBLIND_LOG,
        freeze_manifest_path=entry.FREEZE_MANIFEST,
        analysis_input_manifest_path=entry.ANALYSIS_INPUT_MANIFEST,
        entry_source_path=entry.ENTRY_SOURCE)
    assert doc["readiness"]["ready_for_unblinding"] is True
    assert doc["readiness"]["blocking_reasons"] == []
    names = [c["name"] for c in doc["checks"]]
    for required in ("analysis_input_freeze_manifest", "path_b_tail_overlay_gate",
                     "legacy_shadow_superseded_and_not_joined",
                     "e3_canonical_repool_report", "single_shot_execute_all_entry",
                     "unblinding_records_both_freeze_hashes"):
        assert required in names
    assert doc["path_b_implementation"] == vov.OVERLAY_SOURCE_TAIL
    assert doc["e3_canonical_source"] == "repool"


@pytest.mark.local_artifacts('outputs/stage1/readiness/stage1_confirmatory_readiness_v3.json')
def test_readiness_v3_refuses_tampered_disk_boolean(entry, tmp_path):
    disk = json.loads(entry.READINESS_V3.read_text(encoding="utf-8"))
    disk["readiness"]["blocking_reasons"] = ["fabricated"]
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(disk, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(vreadiness.ReadinessMismatch):
        vreadiness.recompute_and_compare_v3(
            p, prod_dir=entry.PROD, tail_dir=entry.TAIL_DIR,
            shadow_dir=entry.SHADOW_DIR, repool_dir=entry.REPOOL_DIR,
            result_dir=entry.RESULT_DIR, unblind_log=entry.UNBLIND_LOG,
            freeze_manifest_path=entry.FREEZE_MANIFEST,
            analysis_input_manifest_path=entry.ANALYSIS_INPUT_MANIFEST,
            entry_source_path=entry.ENTRY_SOURCE)


# ══════════════════════════════════════════════ 未解盲的边界仍然成立
@pytest.mark.pre_unblinding_workspace
def test_nothing_is_unblinded_yet(entry):
    assert not entry.UNBLIND_LOG.exists(), "unblinding log 不得存在"
    assert not entry.RESULT_DIR.exists() or not list(entry.RESULT_DIR.glob("*"))
    assert not (entry.RESULT_DIR / entry.RESULT_NAME).exists()
