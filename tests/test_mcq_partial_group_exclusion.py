"""形态 B：sampled 失败 → 5 done+raw / 1 excluded，整组不进 finalization。

与形态 A（greedy 失败 → 6 excluded / 0 raw）并列，是**仅有的**两种被授权形态。
重点：五条 raw/hidden 永久保留但绝不进入 F5 与 finalized；
      diagnostic 按**记录级** SHA 授权（全文件 SHA 会随未来追加而变，不能作依据）。
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import tempfile
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
PARTIAL_REASON = "partial_group_excluded_due_sampled_forward_nonfinite"


def _units(eid: str) -> list[str]:
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]


def _row(eid: str, i: int, sample_type: str) -> dict:
    return {"unit_id": f"{eid}__{sample_type}_{i}", "example_id": eid,
            "sample_type": sample_type, "sample_index": i,
            "eligible_for_primary": True, "parsed_label": "A",
            "f5_label_margin": 0.5, "f5_label_entropy": 0.5, "f5_full_entropy": 0.5,
            "f5_letter_token_logprob": -0.5,
            "f5_self_consistency_exact": 1.0, "f5_letter_agreement_rate": 1.0}


def _diag_bytes(uid: str, arm: str = "mcq", *, row_size: int = 152064) -> bytes:
    return json.dumps({"schema_version": "stage1_forward_nonfinite_diagnostic_v1",
                       "arm": arm, "unit_id": uid, "reason": "forward_nonfinite",
                       "row_nonfinite": row_size, "row_size": row_size},
                      ensure_ascii=False).encode("utf-8")


def _partial_audit(eid: str, diag_sha: str) -> dict:
    trig = f"{eid}__sampled_5"
    return {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
            "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
            "action": prod.ACTION_PARTIAL_GROUP,
            "trigger_unit_id": trig, "trigger_sample_type": "sampled",
            "trigger_sample_index": 5, "trigger_reason": "forward_nonfinite",
            "manifest_pattern": "5_done_1_excluded",
            "preserved_raw_unit_ids": _units(eid)[:5],
            "excluded_unit_ids": [trig],
            "raw_preserved_not_analyzed": True,
            "exclusion_reason": PARTIAL_REASON,
            "diagnostic_unit_id": trig, "diagnostic_record_sha256": diag_sha,
            "diagnostics_prefix_bytes_at_recovery": 0,
            "diagnostics_prefix_sha256_at_recovery": "0" * 64,
            "diagnostic_persisted": True,
            "frozen_list_sha256": "a" * 64, "run_fingerprint_sha256": "b" * 64,
            "classification": "runtime_technical_attrition",
            "protocol_deviation": True, "preregistration_files_modified": False}


def _write_partial(tmp_path, n=10, peid="q0003", *, drop_raw=None, extra_raw=False,
                   mark_done_as_excluded=False, audit_over=None, diag_lines=None):
    p = prod.paths_for(tmp_path, "mcq")
    eids = [f"q{i:04d}" for i in range(n)]
    records = [{"example_id": e, "group_id": f"grp_{e}"} for e in eids]
    rows = []
    for e in eids:
        keep = range(K) if e != peid else range(K - 1)     # 形态 B 少一条 sampled_5
        for i in keep:
            rows.append(_row(e, i, "greedy" if i == 0 else "sampled"))
    if drop_raw:
        rows = [r for r in rows if r["unit_id"] != drop_raw]
    if extra_raw:
        rows.append(_row(peid, 5, "sampled"))
    p["raw_results"].parent.mkdir(parents=True, exist_ok=True)
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    man = WorkManifest(p["manifest"])
    man.register(f"{e}__{t}_{i}" for e in eids
                 for t, i in [("greedy", 0)] + [("sampled", j) for j in range(1, K)])
    for r in rows:
        man.mark(r["unit_id"], "done")
    if drop_raw:
        # manifest 仍标 done，只是 raw 行没了 —— 这才是"raw 集合与登记不符"，
        # 而不是更前置的 manifest 状态不符。
        man.mark(drop_raw, "done")
    trig = f"{peid}__sampled_5"
    man.mark(trig, "excluded", reason="forward_nonfinite")
    if mark_done_as_excluded:
        man.mark(f"{peid}__sampled_1", "excluded", reason=PARTIAL_REASON)
    lines = diag_lines if diag_lines is not None else [_diag_bytes(trig)]
    p["diagnostics"].write_bytes(b"".join(x + b"\r\n" for x in lines))
    rec = _partial_audit(peid, hashlib.sha256(_diag_bytes(trig)).hexdigest())
    rec.update(audit_over or {})
    p["technical_exclusions"].write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    hd = p["hidden_dir"]
    hd.mkdir(parents=True, exist_ok=True)
    for r in rows:
        np.save(hd / f"{r['unit_id']}.npy", np.zeros(3584, dtype=np.float32))
    return p, records, rec


def _finalize(tmp_path, records, **kw):
    return prod.finalize_arm("mcq", tmp_path, records, cfg=V3AnalysisConfig(),
                             run_fingerprint={"run_hash": "x" * 64}, **kw)


# ── 授权通过 + 保留但不分析 ─────────────────────────────────────────────────
def test_form_b_finalizes_and_preserves_raw(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records, _ = _write_partial(tmp_path)
    rep = _finalize(tmp_path, records)
    assert rep["technical_exclusions"] == 1
    assert rep["retained_questions"] == 9
    assert rep["actual_finalized_rows"] == 9
    assert rep["retained_raw_units"] == 9 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 5
    assert rep["expected_total_raw_rows"] == 9 * K + 5
    assert rep["analysis_uses"] == "retained_raw_units"
    assert rep["raw_integrity_uses"] == "expected_total_raw_rows"
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in raw if r["example_id"] == "q0003") == 5      # 物理保留
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert len(fin) == 9 and all(r["example_id"] != "q0003" for r in fin)
    groups = collections.Counter(r["example_id"] for r in seen["rows"])
    assert "q0003" not in groups and len(groups) == 9 and set(groups.values()) == {K}
    for uid in _units("q0003")[:5]:                                    # hidden 也保留
        assert (p["hidden_dir"] / f"{uid}.npy").exists()


def test_f5_only_sees_complete_groups(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    _, records, _ = _write_partial(tmp_path, n=12)
    _finalize(tmp_path, records)
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 11}, "F5 只能收到完整 K=6 的 group，绝无 K=5"


# ── 拒绝路径 ────────────────────────────────────────────────────────────────
def test_rejects_preserved_marked_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records, _ = _write_partial(tmp_path, mark_done_as_excluded=True)
    with pytest.raises(ValueError, match="期望 'done'"):
        _finalize(tmp_path, records)


@pytest.mark.parametrize("kw", [{"drop_raw": "q0003__sampled_2"}, {"extra_raw": True}])
def test_rejects_raw_set_mismatch(tmp_path, monkeypatch, kw):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records, _ = _write_partial(tmp_path, **kw)
    with pytest.raises(ValueError, match="raw 集合与 preserved_raw_unit_ids 不符|尚未跑完"):
        _finalize(tmp_path, records)


def test_rejects_wrong_trigger_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records, _ = _write_partial(tmp_path, audit_over={"trigger_reason": "other"})
    with pytest.raises(ValueError, match="白名单"):
        _finalize(tmp_path, records)


def test_rejects_unknown_action(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records, _ = _write_partial(tmp_path, audit_over={"action": "just_drop_it"})
    with pytest.raises(ValueError, match="action"):
        _finalize(tmp_path, records)


def test_rejects_unregistered_partial_group(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records, _ = _write_partial(tmp_path)
    p["technical_exclusions"].unlink()
    with pytest.raises(ValueError, match="未登记"):
        _finalize(tmp_path, records)


# ── diagnostic 记录级 SHA ───────────────────────────────────────────────────
def test_survives_unrelated_diagnostic_append(tmp_path, monkeypatch):
    """diagnostics 追加无关合法记录后，既有形态 B 授权**仍有效**。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records, _ = _write_partial(tmp_path)
    with open(p["diagnostics"], "ab") as fh:
        fh.write(_diag_bytes("q0007__sampled_2") + b"\r\n")
    assert _finalize(tmp_path, records)["technical_exclusions"] == 1


def test_rejects_when_target_record_bytes_change(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records, _ = _write_partial(tmp_path)
    p["diagnostics"].write_bytes(
        _diag_bytes("q0003__sampled_5", row_size=2) + b"\r\n")
    with pytest.raises(ValueError, match="字节已变"):
        _finalize(tmp_path, records)


def test_rejects_duplicate_diagnostic_unit_id(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    d = _diag_bytes("q0003__sampled_5")
    _, records, _ = _write_partial(tmp_path, diag_lines=[d, d])
    with pytest.raises(ValueError, match="重复"):
        _finalize(tmp_path, records)


def test_rejects_missing_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records, _ = _write_partial(tmp_path, diag_lines=[])
    with pytest.raises(ValueError, match="不存在"):
        _finalize(tmp_path, records)


def test_record_sha_caliber_excludes_line_terminator():
    """record SHA 口径：**不含**行终止符，故 LF 与 CRLF 得到同一值。"""
    line = _diag_bytes("q0__sampled_5")
    got = []
    for term in (b"\n", b"\r\n"):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        Path(path).write_bytes(line + term)
        got.append(prod.diagnostic_record_index(Path(path)))
        os.unlink(path)
    assert got[0] == got[1] == {"q0__sampled_5": hashlib.sha256(line).hexdigest()}
    assert prod.DIAGNOSTIC_RECORD_SHA_CALIBER == "line_bytes_excluding_line_terminator"
