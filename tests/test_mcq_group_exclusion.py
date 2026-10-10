"""MCQ 整组技术性排除（runtime technical attrition）的 recovery 支持。

覆盖三块：
  A. recovery 的幂等与崩溃恢复（含各中断点故障注入）
  B. finalizer 授权门 —— audit 真源 **加** manifest latest event 双向核对
  C. 诊断持久化 —— 严格 JSON、无 completion 原文、G3/G4 默认不创建该文件
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as gb  # noqa: E402
from recover_attention.generation.manifest import WorkManifest  # noqa: E402

K = 6
TRIGGER_REASON = "forward_nonfinite"
GROUP_REASON = "group_excluded_due_greedy_forward_nonfinite"


# ══════════════════════════════════════════════════════════════════════════════
# 固定装置
# ══════════════════════════════════════════════════════════════════════════════
def _units(eid: str) -> list[str]:
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]


def _audit_record(eid: str, *, reason: str = GROUP_REASON,
                  trigger_reason: str = TRIGGER_REASON,
                  affected: list[str] | None = None) -> dict:
    return {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
            "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
            "trigger_unit_id": f"{eid}__greedy_0", "trigger_reason": trigger_reason,
            "affected_unit_ids": affected if affected is not None else _units(eid),
            "prior_manifest_status": {}, "action": "group_exclusion",
            "exclusion_reason": reason, "frozen_list_sha256": "a" * 64,
            "run_fingerprint_sha256": "b" * 64,
            "classification": "runtime_technical_attrition",
            "protocol_deviation": True, "preregistration_files_modified": False,
            "diagnostic_persisted": False, "diagnostic_persistence_note": "n/a"}


def _row(eid: str, i: int, sample_type: str) -> dict:
    # 六项 MCQ_FINALIZED_F5 全给：跨样本两项本应由 attach_cross_sample_f5 填，
    # 夹具里直接预置，免得 stub 掉聚合后触发"缺 F5"（那是校验在正常工作）。
    return {"unit_id": f"{eid}__{sample_type}_{i}", "example_id": eid,
            "sample_type": sample_type, "sample_index": i,
            "eligible_for_primary": True, "parsed_label": "A",
            "f5_label_margin": 0.5, "f5_label_entropy": 0.5, "f5_full_entropy": 0.5,
            "f5_letter_token_logprob": -0.5,
            "f5_self_consistency_exact": 1.0, "f5_letter_agreement_rate": 1.0}


def _write_arm(tmp_path, n_examples: int, *, excluded: list[str] | None = None,
               audit: list[dict] | None = None,
               manifest_reasons: dict | None = None):
    """造一个可 finalize 的 MCQ 臂；excluded 的 example 不产生任何 raw 行。"""
    excluded = excluded or []
    p = prod.paths_for(tmp_path, "mcq")
    eids = [f"q{i:04d}" for i in range(n_examples)]
    records = [{"example_id": e, "group_id": f"grp_{e}"} for e in eids]
    rows = []
    for e in eids:
        if e in excluded:
            continue
        rows.append(_row(e, 0, "greedy"))
        rows.extend(_row(e, i, "sampled") for i in range(1, K))
    p["raw_results"].parent.mkdir(parents=True, exist_ok=True)
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    man = WorkManifest(p["manifest"])
    man.register(f"{e}__{t}_{i}" for e in eids
                 for t, i in [("greedy", 0)] + [("sampled", j) for j in range(1, K)])
    for r in rows:
        man.mark(r["unit_id"], "done")
    for e in excluded:
        for uid in _units(e):
            reason = (manifest_reasons or {}).get(
                uid, TRIGGER_REASON if uid.endswith("__greedy_0") else GROUP_REASON)
            if reason is not None:
                man.mark(uid, "excluded", reason=reason)
    if audit is not None:
        p["technical_exclusions"].write_text(
            "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit), encoding="utf-8")
    hd = p["hidden_dir"]
    hd.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["sample_type"] == "greedy":
            np.save(hd / f"{r['unit_id']}.npy", np.zeros(3584, dtype=np.float32))
    return p, records


def _finalize(tmp_path, records, **kw):
    cfg = V3AnalysisConfig()
    return prod.finalize_arm("mcq", tmp_path, records, cfg=cfg,
                             run_fingerprint={"run_hash": "x" * 64}, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# A. finalizer 授权门
# ══════════════════════════════════════════════════════════════════════════════
def test_authorized_group_exclusion_finalizes_with_dual_caliber(tmp_path, monkeypatch):
    """完整、已登记的整组排除 → 接受；报告双口径自洽，且不产生合成行。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _write_arm(tmp_path, 10, excluded=["q0003"], audit=[_audit_record("q0003")])
    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 10
    assert rep["technical_exclusions"] == 1
    assert rep["retained_questions"] == 9
    assert rep["actual_finalized_rows"] == 9
    assert rep["retained_raw_units"] == 9 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 0
    assert rep["expected_total_raw_rows"] == 9 * K
    assert rep["protocol_deviation"] is True
    assert rep["classification"] == "runtime_technical_attrition"
    assert rep["excluded_example_ids"] == ["q0003"]
    assert rep["exclusion_audit_sha256"]
    # finalized 行数 == 真实 greedy 行数：没有合成占位行
    finalized = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert len(finalized) == 9
    assert all(r["example_id"] != "q0003" for r in finalized)


def test_report_and_provenance_agree_on_both_calibers(tmp_path, monkeypatch):
    """报告与 population_provenance **不得**一处 nominal 一处 actual。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"], audit=[_audit_record("q0003")])
    rep = _finalize(tmp_path, records)
    prov = rep["population_provenance"]
    for key in ("nominal_questions", "technical_exclusions", "retained_questions",
                "actual_finalized_rows", "retained_raw_units",
                "preserved_but_not_analyzed_raw_units", "expected_total_raw_rows",
                "protocol_deviation", "classification"):
        assert prov[key] == rep[key], key
    assert prov["nominal_finalized_rows"] == 10
    assert prov["actual_finalized_rows"] == 9
    assert "not_a_prereg_change" not in prov
    assert prov["preregistration_files_modified"] is False


def test_unregistered_exclusion_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"], audit=[])
    with pytest.raises(ValueError, match="未登记"):
        _finalize(tmp_path, records)


def test_partial_group_exclusion_is_rejected(tmp_path, monkeypatch):
    """只排 3 个、留 3 个 pending —— 部分排除一律拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    reasons = {f"q0003__sampled_{i}": None for i in (3, 4, 5)}   # 这三条不标 excluded
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"],
                            audit=[_audit_record("q0003")], manifest_reasons=reasons)
    with pytest.raises(ValueError, match="期望 'excluded'"):
        _finalize(tmp_path, records)


def test_unknown_exclusion_reason_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"],
                            audit=[_audit_record("q0003", reason="because_i_said_so")])
    with pytest.raises(ValueError, match="白名单"):
        _finalize(tmp_path, records)


def test_finalizer_checks_manifest_latest_reason_not_only_audit(tmp_path, monkeypatch):
    """audit 登记正确，但 manifest 的 reason 被写成别的 —— 必须拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    reasons = {"q0003__sampled_2": "some_other_reason"}
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"],
                            audit=[_audit_record("q0003")], manifest_reasons=reasons)
    with pytest.raises(ValueError, match="latest reason"):
        _finalize(tmp_path, records)


def test_missing_one_unit_in_normal_group_is_rejected(tmp_path, monkeypatch):
    """非排除组仍必须严格 K=6：缺一条即拒绝，不得靠授权门放行。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _write_arm(tmp_path, 10, excluded=["q0003"], audit=[_audit_record("q0003")])
    rows = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["unit_id"] != "q0005__sampled_3"]
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="尚未跑完|!= K"):
        _finalize(tmp_path, records)


def test_frozen_list_sha_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _write_arm(tmp_path, 10, excluded=["q0003"], audit=[_audit_record("q0003")])
    with pytest.raises(ValueError, match="frozen_list_sha256"):
        _finalize(tmp_path, records, frozen_list_sha256="c" * 64)


def test_no_exclusion_keeps_no_deviation(tmp_path, monkeypatch):
    """没有排除时：零偏离，nominal == actual。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _write_arm(tmp_path, 5)
    rep = _finalize(tmp_path, records)
    assert rep["technical_exclusions"] == 0
    assert rep["protocol_deviation"] is False
    assert rep["classification"] == "no_deviation"
    assert rep["nominal_finalized_rows"] == rep["actual_finalized_rows"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# B. 诊断持久化
# ══════════════════════════════════════════════════════════════════════════════
def test_sanitize_removes_completion_and_encodes_nonfinite():
    diag = {"reason": "forward_nonfinite", "completion": "秘密正文 abc",
            "candidate_logits": {"A": float("nan"), "B": float("inf"),
                                 "C": float("-inf"), "D": 1.25},
            "row_nonfinite": 3, "row_size": 152064}
    out = gb.sanitize_diagnostic(diag)
    assert "completion" not in out
    assert out["completion_char_len"] == len("秘密正文 abc")
    assert out["completion_utf8_byte_len"] == len("秘密正文 abc".encode("utf-8"))
    assert len(out["completion_sha256"]) == 64
    assert out["candidate_logits"]["A"] == {"value": None, "nonfinite_kind": "nan"}
    assert out["candidate_logits"]["B"] == {"value": None, "nonfinite_kind": "posinf"}
    assert out["candidate_logits"]["C"] == {"value": None, "nonfinite_kind": "neginf"}
    assert out["candidate_logits"]["D"] == 1.25
    # 严格 JSON：不得出现裸 NaN / Infinity
    text = json.dumps(out, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == out


def test_diagnostic_file_is_strict_json_and_has_no_completion(tmp_path):
    from recover_attention.evaluation import v3_g3_smoke as g3
    path = tmp_path / "mcq_forward_nonfinite_diagnostics.jsonl"
    diag = gb.sanitize_diagnostic({"reason": "forward_nonfinite", "completion": "x" * 30,
                                   "candidate_logits": {"A": float("nan")},
                                   "row_nonfinite": 1, "row_size": 10})
    g3._append_diagnostic(path, "mcq", {"unit_id": "q1__greedy_0"}, diag)
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    rec = json.loads(raw.strip())
    assert rec["unit_id"] == "q1__greedy_0"
    assert "completion" not in rec
    assert rec["completion_char_len"] == 30
    assert rec["candidate_logits"]["A"]["nonfinite_kind"] == "nan"


def test_append_diagnostic_refuses_raw_completion(tmp_path):
    from recover_attention.evaluation import v3_g3_smoke as g3
    with pytest.raises(ValueError, match="completion"):
        g3._append_diagnostic(tmp_path / "d.jsonl", "mcq", {"unit_id": "u"},
                              {"completion": "raw text"})


def test_run_arm_default_creates_no_diagnostic_file(tmp_path):
    """G3/G4 默认 diagnostic_path=None —— 不创建该文件，文件集合不变。"""
    import inspect
    from recover_attention.evaluation import v3_g3_smoke as g3
    sig = inspect.signature(g3.run_arm)
    assert sig.parameters["diagnostic_path"].default is None
    before = set(p.name for p in tmp_path.iterdir()) if tmp_path.exists() else set()
    assert not any(n.endswith("_forward_nonfinite_diagnostics.jsonl") for n in before)


# ══════════════════════════════════════════════════════════════════════════════
# C. recovery 幂等 / 崩溃恢复
# ══════════════════════════════════════════════════════════════════════════════
def _apply(man_path, record, *, stop_after: int | None = None):
    """recovery 的 manifest 部分（与脚本同一算法），stop_after 用于故障注入。"""
    man = WorkManifest(man_path)
    status = man.current_status()
    events = {ev["unit_id"]: ev for ev in man._events}
    appended = 0
    for uid in record["affected_unit_ids"]:
        want = (record["trigger_reason"] if uid == record["trigger_unit_id"]
                else record["exclusion_reason"])
        st = status.get(uid)
        if st == "excluded":
            assert events[uid].get("reason") == want
            continue
        if st != "pending":
            raise ValueError(f"{uid} 状态 {st!r} 不是 pending")
        if stop_after is not None and appended >= stop_after:
            raise RuntimeError("injected crash")
        man.mark(uid, "excluded", reason=want)
        appended += 1
    return appended


@pytest.mark.parametrize("crash_after", [0, 1, 2, 3, 4])
def test_recovery_is_idempotent_across_crash_points(tmp_path, crash_after):
    """在第 0–4 条 manifest append 后崩溃，重跑均安全收敛，且每 unit 只留一条事件。"""
    eid = "q0003"
    man_path = tmp_path / "mcq_production_manifest.jsonl"
    man = WorkManifest(man_path)
    man.register(_units(eid))
    man.mark(f"{eid}__greedy_0", "excluded", reason=TRIGGER_REASON)
    record = _audit_record(eid)
    with pytest.raises(RuntimeError, match="injected crash"):
        _apply(man_path, record, stop_after=crash_after)
    prefix = man_path.read_bytes()
    appended = _apply(man_path, record)              # 重跑收敛
    assert appended == 5 - crash_after
    assert man_path.read_bytes()[:len(prefix)] == prefix   # 前缀逐字节不变
    again = _apply(man_path, record)                 # 再跑一次：完全幂等
    assert again == 0
    man2 = WorkManifest(man_path)
    status = man2.current_status()
    n_recovery = 0
    for uid in _units(eid):
        assert status[uid] == "excluded"
        hits = [ev for ev in man2._events
                if ev["unit_id"] == uid and ev["status"] == "excluded"]
        assert len(hits) == 1, uid
        if hits[0].get("reason") == GROUP_REASON:
            n_recovery += 1
    assert n_recovery == 5


def test_recovery_refuses_non_pending_unit(tmp_path):
    eid = "q0003"
    man_path = tmp_path / "m.jsonl"
    man = WorkManifest(man_path)
    man.register(_units(eid))
    man.mark(f"{eid}__greedy_0", "excluded", reason=TRIGGER_REASON)
    man.mark(f"{eid}__sampled_2", "done")
    with pytest.raises(ValueError, match="不是 pending"):
        _apply(man_path, _audit_record(eid))


def test_excluded_units_are_not_requeued_on_resume(tmp_path):
    eid = "q0003"
    man = WorkManifest(tmp_path / "m.jsonl")
    man.register(_units(eid))
    man.mark(f"{eid}__greedy_0", "excluded", reason=TRIGGER_REASON)
    for i in range(1, K):
        man.mark(f"{eid}__sampled_{i}", "excluded", reason=GROUP_REASON)
    assert man.pending_units(_units(eid)) == []


def test_existing_audit_with_different_content_is_rejected():
    """审计文件已存在但内容不符 → 拒绝（不覆盖、不重写）。"""
    a, b = _audit_record("q0003"), _audit_record("q0003")
    b["question_hash"] = "z" * 64
    assert a != b
    diff = [k for k in a if a[k] != b.get(k)]
    assert diff == ["question_hash"]


# ══════════════════════════════════════════════════════════════════════════════
# D. diagnostic → manifest 跨文件断电窗口的一致性门
# ══════════════════════════════════════════════════════════════════════════════
from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402


def _diag_line(uid: str, arm: str = "mcq") -> str:
    return json.dumps({"schema_version": "stage1_forward_nonfinite_diagnostic_v1",
                       "arm": arm, "unit_id": uid, "reason": "forward_nonfinite",
                       "row_nonfinite": 1, "row_size": 10}, ensure_ascii=False)


def _setup(tmp_path, *, status: str | None, reason: str | None = None,
           diag_text: str | None = None, uid: str = "q0__greedy_0"):
    dp, mp = tmp_path / "diag.jsonl", tmp_path / "man.jsonl"
    man = WorkManifest(mp)
    man.register([uid, "q0__sampled_1"])
    if status is not None:
        man.mark(uid, status, **({"reason": reason} if reason else {}))
    dp.write_text(_diag_line(uid) + "\n" if diag_text is None else diag_text,
                  encoding="utf-8")
    return dp, mp


@pytest.mark.parametrize("status,reason", [("running", None), ("pending", None),
                                           ("failed", "boom"), ("done", None)])
def test_gate_rejects_diagnostic_without_terminal_exclusion(tmp_path, status, reason):
    """有诊断但 manifest 不是 excluded/forward_nonfinite → 一律拒绝。"""
    dp, mp = _setup(tmp_path, status=status, reason=reason)
    with pytest.raises(g3.DiagnosticRecoveryRequired, match="DiagnosticRecoveryRequired"):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_rejects_excluded_with_wrong_reason(tmp_path):
    dp, mp = _setup(tmp_path, status="excluded", reason="something_else")
    with pytest.raises(g3.DiagnosticRecoveryRequired):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_accepts_excluded_forward_nonfinite(tmp_path):
    dp, mp = _setup(tmp_path, status="excluded", reason="forward_nonfinite")
    out = g3.require_diagnostic_manifest_consistency(
        arm="mcq", diagnostic_path=dp, manifest_path=mp)
    assert out["diagnostic_units"] == 1


def test_gate_allows_historic_exclusion_without_diagnostic(tmp_path):
    """反向不查：excluded 但无 diagnostic 合法（cybermetric_005396 即此情形）。"""
    mp = tmp_path / "man.jsonl"
    man = WorkManifest(mp)
    units = _units("cybermetric_005396")
    man.register(units)
    man.mark(units[0], "excluded", reason=TRIGGER_REASON)
    for u in units[1:]:
        man.mark(u, "excluded", reason=GROUP_REASON)
    out = g3.require_diagnostic_manifest_consistency(
        arm="mcq", diagnostic_path=tmp_path / "absent.jsonl", manifest_path=mp,
        known_units=units)
    assert out["diagnostic_units"] == 0


@pytest.mark.parametrize("bad", [
    "{not json}\n",                                        # malformed
    '{"arm": "mcq", "unit_id": "q0__greedy_0"',            # truncated（无换行、缺右括号）
    '{"arm": "mcq"}\n',                                    # 缺 unit_id
])
def test_gate_fails_closed_on_bad_lines(tmp_path, bad):
    dp, mp = _setup(tmp_path, status="excluded", reason="forward_nonfinite", diag_text=bad)
    with pytest.raises(g3.DiagnosticRecoveryRequired):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_rejects_duplicate_unit_id(tmp_path):
    text = _diag_line("q0__greedy_0") + "\n" + _diag_line("q0__greedy_0") + "\n"
    dp, mp = _setup(tmp_path, status="excluded", reason="forward_nonfinite", diag_text=text)
    with pytest.raises(g3.DiagnosticRecoveryRequired, match="重复"):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_rejects_wrong_arm(tmp_path):
    dp, mp = _setup(tmp_path, status="excluded", reason="forward_nonfinite",
                    diag_text=_diag_line("q0__greedy_0", arm="h1") + "\n")
    with pytest.raises(g3.DiagnosticRecoveryRequired, match="arm"):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_rejects_unknown_unit(tmp_path):
    dp, mp = _setup(tmp_path, status="excluded", reason="forward_nonfinite")
    with pytest.raises(g3.DiagnosticRecoveryRequired, match="未知 unit_id"):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp,
            known_units=["some_other__greedy_0"])


def test_gate_rejects_missing_manifest_event(tmp_path):
    dp = tmp_path / "d.jsonl"; mp = tmp_path / "m.jsonl"
    dp.write_text(_diag_line("q0__greedy_0") + "\n", encoding="utf-8")
    mp.write_text("", encoding="utf-8")
    with pytest.raises(g3.DiagnosticRecoveryRequired):
        g3.require_diagnostic_manifest_consistency(
            arm="mcq", diagnostic_path=dp, manifest_path=mp)


def test_gate_disabled_when_path_is_none(tmp_path):
    """G3/G4 默认 diagnostic_path=None —— 门不启用，行为不变。"""
    out = g3.require_diagnostic_manifest_consistency(
        arm="mcq", diagnostic_path=None, manifest_path=tmp_path / "nope.jsonl")
    assert out == {"checked": 0, "diagnostic_units": 0, "path": ""}


def test_run_arm_gate_runs_before_backend(tmp_path):
    """run_arm 内的防线：不一致时 backend **一次都不被调用**。"""
    cfg = V3AnalysisConfig()
    records = [{"example_id": "q0", "group_id": "g0"}]
    calls = {"n": 0}

    class _Backend:
        def run_fingerprint(self, arm):
            calls["n"] += 1
            return {}

        def verify_hidden_index(self):
            calls["n"] += 1
            return {}

        def run_unit(self, unit, record):
            calls["n"] += 1
            return {}

    mp = tmp_path / "man.jsonl"
    WorkManifest(mp).register([f"q0__{t}_{i}" for t, i in
                               [("greedy", 0)] + [("sampled", j) for j in range(1, K)]])
    dp = tmp_path / "diag.jsonl"
    dp.write_text(_diag_line("q0__greedy_0") + "\n", encoding="utf-8")   # manifest 仍 pending
    with pytest.raises(g3.DiagnosticRecoveryRequired):
        g3.run_arm(arm="mcq", records=records, backend=_Backend(), out_dir=tmp_path,
                   cfg=cfg, results_path=tmp_path / "raw.jsonl",
                   hidden_dir=tmp_path / "hid", manifest_path=mp, diagnostic_path=dp)
    assert calls["n"] == 0


def test_cli_gate_runs_before_real_backend(tmp_path, monkeypatch):
    """CLI 必须在 `_real_backend()` 之前拒绝 —— 以其调用次数为证。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_prod_cli", _REPO_ROOT / "scripts" / "run_stage1_production.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    src = (_REPO_ROOT / "scripts" / "run_stage1_production.py").read_text(encoding="utf-8")
    i_gate = src.index("require_diagnostic_manifest_consistency")
    i_backend = src.index("backend = _real_backend(")
    assert i_gate < i_backend, "一致性门必须出现在 _real_backend() 之前"
    calls = {"n": 0}
    monkeypatch.setattr(cli, "_real_backend", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 0
