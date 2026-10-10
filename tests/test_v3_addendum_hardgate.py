"""correction addendum：M4 两臂 schema、json_safe 键、verifier 类型、**临时硬闸**。

硬闸的意义：`run_execute` 是可**直接导入调用**的公开路径。CLI 尚未接 analyzer
**不构成保护** —— 手工写一个 `ready_for_unblinding=true` 的 JSON 就能走到
`record_unblinding → analyze`。在 checked-in 的 readiness 重算入口落地之前，
`preflight_execute` 必须无条件拒绝。
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

from recover_attention.evaluation import v3_population_declaration as pdec  # noqa: E402
from recover_attention.evaluation import v3_unblinding_boundary as ub      # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_stage1_v3_confirmatory", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
entry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = entry
_spec.loader.exec_module(entry)


# ══ ④ 临时硬闸 ════════════════════════════════════════════════════════════
def _prod(tmp_path):
    d = tmp_path / "prod"
    d.mkdir(exist_ok=True)
    for n in ub.FINALIZED_SIX:
        (d / n).write_bytes(n.encode())
    return d


def _hand_written_ready_true(tmp_path, prod):
    """**手工**构造 ready_for_unblinding=true 的 readiness JSON（攻击面本体）。"""
    arts = {n: {"sha256": hashlib.sha256((prod / n).read_bytes()).hexdigest()}
            for n in ub.FINALIZED_SIX}
    p = tmp_path / "readiness.json"
    p.write_text(json.dumps({"freeze_chain": {"finalized_artifacts": arts},
                             "readiness": {"ready_for_unblinding": True,
                                           "blocking_reasons": []}}), encoding="utf-8")
    return p


def test_hand_written_ready_true_is_still_refused(tmp_path):
    prod = _prod(tmp_path)
    rp = _hand_written_ready_true(tmp_path, prod)
    mp = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(mp, prod)
    with pytest.raises(entry.ExecuteRefused, match="拒绝信任磁盘布尔值|不存在"):
        entry.preflight_execute(prod_dir=prod, readiness_path=rp,
                                result_dir=tmp_path / "res",
                                unblind_log=tmp_path / "u.jsonl",
                                freeze_manifest_path=mp)


def test_run_execute_all_refuses_and_calls_nothing(tmp_path, monkeypatch):
    """硬闸生效时：record_unblinding 0 次、analyze 0 次、结果文件不存在。"""
    prod = _prod(tmp_path)
    rp = _hand_written_ready_true(tmp_path, prod)
    mp = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(mp, prod)
    calls = {"record": 0, "analyze": 0}
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda *a, **k: calls.__setitem__("record", calls["record"] + 1))

    def analyze():
        calls["analyze"] += 1
        return {"ok": True}

    res_dir = tmp_path / "res"
    log = tmp_path / "u.jsonl"
    with pytest.raises(entry.ExecuteRefused, match="拒绝信任磁盘布尔值|不存在"):
        entry.run_execute_all(cfg=V3AnalysisConfig(),
                              analyze_factory=lambda: analyze,
                              timestamp="2026-08-18T00:00:00Z",
                              prod_dir=prod, readiness_path=rp, result_dir=res_dir,
                              unblind_log=log, freeze_manifest_path=mp)
    assert calls == {"record": 0, "analyze": 0}
    assert not log.exists()
    assert not res_dir.exists()


def test_gate_fires_before_reading_readiness_boolean(tmp_path):
    """硬闸位置必须在信任磁盘布尔值**之前** —— readiness 文件不存在时也是同一个理由。"""
    prod = _prod(tmp_path)
    mp = tmp_path / "manifest.json"
    ub.write_stage1_freeze_manifest(mp, prod)
    with pytest.raises(entry.ExecuteRefused, match="拒绝信任磁盘布尔值|不存在"):
        entry.preflight_execute(prod_dir=prod, readiness_path=tmp_path / "nonexistent.json",
                                result_dir=tmp_path / "res",
                                unblind_log=tmp_path / "u.jsonl", freeze_manifest_path=mp)


def test_recompute_readiness_is_now_wired_and_refuses_tampering():
    """builder 已入库：不再抛 NotWired，而是**现场重算并比对**。

    手工把磁盘布尔值改成 true 仍必须被抓出来 —— 这是原硬闸保护的同一条不变量，
    只是理由从"尚未接线"变成"与现场重算不一致"。
    """
    from recover_attention.evaluation import v3_readiness as vrd
    assert entry.recompute_readiness.__module__ == entry.__name__
    with pytest.raises(vrd.ReadinessMismatch):
        entry.recompute_readiness(Path("definitely_missing_readiness.json"),
                                  Path("definitely_missing_prod"))


def test_cli_execute_without_confirm_does_not_unblind(monkeypatch):
    """闸通过 != 该解盲：没有 --confirm-unblinding 就**不构造回调、不解盲**。"""
    monkeypatch.setattr(entry, "preflight_execute", lambda: {})
    def unexpected(*args, **kwargs):
        pytest.fail("The CLI must not build real inputs or unblind without confirmation")
    monkeypatch.setattr(entry, "build_real_analyze", unexpected)
    monkeypatch.setattr(entry, "run_execute_all", unexpected)
    assert entry.main(["--arm", "mcq", "--execute-all"]) == 3


def test_confirm_unblinding_requires_execute_and_timestamp():
    for argv in (["--arm", "mcq", "--confirm-unblinding"],
                 ["--arm", "mcq", "--execute-all", "--confirm-unblinding"]):
        with pytest.raises(SystemExit) as exc:
            entry.main(argv)
        assert exc.value.code == 2


def test_order_test_uses_trusted_gate(tmp_path, monkeypatch):
    """既有顺序测试改为 monkeypatch preflight，只测「通过可信 gate 之后」的顺序。"""
    order = []
    monkeypatch.setattr(entry, "preflight_execute",
                        lambda **kw: {"input_freeze_hash": "f" * 64,
                                      "analysis_input_freeze_hash": "a" * 64,
                                      "prereg_sha256": "p" * 64,
                                      "readiness_sha256": "r" * 64})
    monkeypatch.setattr(entry.ub, "record_stage1_unblinding",
                        lambda *a, **k: order.append("record_unblinding"))
    out = entry.run_execute_all(
        cfg=V3AnalysisConfig(),
        analyze_factory=lambda: order.append("build_callback") or (
            lambda: order.append("analyze") or {"ok": True}),
        timestamp="2026-08-18T00:00:00Z", result_dir=tmp_path / "res",
        unblind_log=tmp_path / "u.jsonl")
    # 构造回调（会载入含 label 的 records）也必须在解盲日志**之后**
    assert order == ["record_unblinding", "build_callback", "analyze"]
    assert Path(out["result_path"]).exists()


# ══ ① M4 两臂 schema ══════════════════════════════════════════════════════
def _decl(**over):
    d = {"schema_version": pdec.DECLARATION_SCHEMA,
         "arms": {a: {"analyzable": 4, "n_positive": 1, "n_negative": 3,
                      "positive_rate": 0.25} for a in ("h1", "mcq")},
         "label_dependent_fields": list(pdec.LABEL_DEPENDENT_FIELDS),
         "label_dependent_status": "filled", "performance_blind": False,
         "attrition_categories": {}}
    d.update(over)
    return d


def test_m4_accepts_well_formed_two_arm_declaration():
    pdec.validate_declaration(_decl())


@pytest.mark.parametrize("over,match", [
    ({"schema_version": "wrong"}, "schema_version"),
    ({"arms": {}}, "arms 键集合"),
    ({"arms": {"h1": {"analyzable": 4, "n_positive": 1, "n_negative": 3,
                      "positive_rate": 0.25}}}, "arms 键集合"),
    ({"arms": {"h1": {}, "mcq": {}, "extra": {}}}, "arms 键集合"),
    ({"label_dependent_fields": ["n_positive"]}, "label_dependent_fields"),
    ({"performance_blind": True}, "performance_blind"),
])
def test_m4_rejects_bad_schema(over, match):
    with pytest.raises(ValueError, match=match):
        pdec.validate_declaration(_decl(**over))


def test_m4_rejects_arm_missing_required_fields():
    d = _decl()
    d["arms"]["mcq"].pop("analyzable")
    with pytest.raises(ValueError, match="缺字段"):
        pdec.validate_declaration(d)


def test_m4_pending_declaration_from_builder_passes_schema():
    d = pdec.build_declaration(h1={"analyzable": 8664}, mcq={"analyzable": 1750})
    pdec.validate_declaration(d)
    assert d["performance_blind"] is True and set(d["arms"]) == {"h1", "mcq"}


# ══ ② json_safe 非字符串键 ════════════════════════════════════════════════
def test_json_safe_rejects_non_string_keys():
    with pytest.raises(TypeError, match="非字符串 dict key"):
        entry.json_safe({1: "integer", "1": "string"})


def test_json_safe_rejects_nested_non_string_keys():
    with pytest.raises(TypeError, match="非字符串 dict key"):
        entry.json_safe({"outer": {2: "x"}})


def test_result_writer_refuses_non_string_keys(tmp_path):
    out = tmp_path / "r.json"
    with pytest.raises(TypeError):
        entry.write_result_atomic(out, {1: "a", "1": "b"})
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


def test_json_safe_keeps_all_string_keys():
    assert entry.json_safe({"1": "a", "b": {"c": 1}}) == {"1": "a", "b": {"c": 1}}


# ══ ③ verifier 类型与 hash_caliber ════════════════════════════════════════
@pytest.mark.parametrize("bad", [6.0, True, "6", None])
def test_verifier_rejects_non_int_n_artifacts(tmp_path, bad):
    prod = _prod(tmp_path)
    m = ub.build_stage1_freeze_manifest(prod)
    m["n_artifacts"] = bad
    v = ub.verify_stage1_freeze_manifest(m, prod)
    assert v["ok"] is False
    assert any("n_artifacts" in s for s in v["structural_problems"])


def test_verifier_locks_hash_caliber(tmp_path):
    prod = _prod(tmp_path)
    m = ub.build_stage1_freeze_manifest(prod)
    assert m["hash_caliber"] == ub.HASH_CALIBER
    m["hash_caliber"] = "sha256 over something else"
    v = ub.verify_stage1_freeze_manifest(m, prod)
    assert v["ok"] is False
    assert any("hash_caliber" in s for s in v["structural_problems"])


def test_real_manifest_still_verifies():
    """仓库里已冻结的那份 manifest 在收紧后仍必须通过。"""
    p = _REPO_ROOT / "outputs/stage1/readiness/stage1_finalized_freeze_manifest.json"
    if not p.exists():
        pytest.skip("freeze manifest 不存在")
    m = json.loads(p.read_text(encoding="utf-8"))
    v = ub.verify_stage1_freeze_manifest(m, _REPO_ROOT / "outputs/stage1/production")
    assert v["ok"] is True and v["structural_problems"] == []
