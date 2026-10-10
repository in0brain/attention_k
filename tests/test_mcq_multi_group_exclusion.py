"""多组技术性排除的组合：**两个形态 A + 一个形态 B**。

验证计数、过滤与 F5 分组在多组共存时仍正确，并覆盖形态 A 的
`diagnostic_persisted` 两种取值：
  - false（诊断持久化补丁之前的历史排除）→ **合法**，跳过诊断校验
  - true （补丁之后）→ 必须验证唯一 diagnostic 行、record SHA、unit_id、trigger reason
"""

from __future__ import annotations

import collections
import hashlib
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
TRIG = "forward_nonfinite"
A_REASON = "group_excluded_due_greedy_forward_nonfinite"
B_REASON = "partial_group_excluded_due_sampled_forward_nonfinite"
#: sampled_{j+1..K-1} 的 manifest 事件 reason（与组级 B_REASON 层级不同，不可混用）。
POST_REASON = "group_excluded_due_sampled_forward_nonfinite"


def _units(eid):
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]


def _row(eid, i, st):
    return {"unit_id": f"{eid}__{st}_{i}", "example_id": eid, "sample_type": st,
            "sample_index": i, "eligible_for_primary": True, "parsed_label": "A",
            "f5_label_margin": 0.5, "f5_label_entropy": 0.5, "f5_full_entropy": 0.5,
            "f5_letter_token_logprob": -0.5,
            "f5_self_consistency_exact": 1.0, "f5_letter_agreement_rate": 1.0}


def _diag_bytes(uid):
    return json.dumps({"schema_version": "stage1_forward_nonfinite_diagnostic_v1",
                       "arm": "mcq", "unit_id": uid, "reason": TRIG,
                       "row_nonfinite": 152064, "row_size": 152064},
                      ensure_ascii=False).encode("utf-8")


def _a_record(eid, *, persisted, diag_sha=None):
    rec = {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
           "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
           "action": prod.ACTION_FULL_GROUP,
           "trigger_unit_id": f"{eid}__greedy_0", "trigger_reason": TRIG,
           "affected_unit_ids": _units(eid), "prior_manifest_status": {},
           "exclusion_reason": A_REASON,
           "frozen_list_sha256": "a" * 64, "run_fingerprint_sha256": "b" * 64,
           "classification": "runtime_technical_attrition",
           "protocol_deviation": True, "preregistration_files_modified": False,
           "diagnostic_persisted": persisted, "diagnostic_persistence_note": "n/a"}
    if persisted:
        rec.update({"diagnostic_unit_id": f"{eid}__greedy_0",
                    "diagnostic_record_sha256": diag_sha,
                    "diagnostic_record_sha256_caliber": prod.DIAGNOSTIC_RECORD_SHA_CALIBER,
                    "diagnostics_prefix_bytes_at_recovery": 0,
                    "diagnostics_prefix_sha256_at_recovery": "0" * 64})
    return rec


def _b_record(eid, diag_sha):
    trig = f"{eid}__sampled_5"
    return {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
            "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
            "action": prod.ACTION_PARTIAL_GROUP,
            "trigger_unit_id": trig, "trigger_sample_type": "sampled",
            "trigger_sample_index": 5, "trigger_reason": TRIG,
            "manifest_pattern": "5_done_1_excluded",
            "preserved_raw_unit_ids": _units(eid)[:5], "excluded_unit_ids": [trig],
            "raw_preserved_not_analyzed": True, "exclusion_reason": B_REASON,
            "diagnostic_unit_id": trig, "diagnostic_record_sha256": diag_sha,
            "diagnostic_record_sha256_caliber": prod.DIAGNOSTIC_RECORD_SHA_CALIBER,
            "diagnostics_prefix_bytes_at_recovery": 0,
            "diagnostics_prefix_sha256_at_recovery": "0" * 64,
            "diagnostic_persisted": True,
            "frozen_list_sha256": "a" * 64, "run_fingerprint_sha256": "b" * 64,
            "classification": "runtime_technical_attrition",
            "protocol_deviation": True, "preregistration_files_modified": False}


def _build(tmp_path, n=20, a_old="q0002", a_new="q0005", b="q0008",
           *, a_new_persisted=True, tamper_diag=False, drop_diag=False):
    """两个形态 A（一个 persisted=false、一个 true）+ 一个形态 B。"""
    p = prod.paths_for(tmp_path, "mcq")
    eids = [f"q{i:04d}" for i in range(n)]
    records = [{"example_id": e, "group_id": f"grp_{e}"} for e in eids]
    rows = []
    for e in eids:
        if e in (a_old, a_new):
            continue                                   # 形态 A：raw 0 条
        keep = range(K) if e != b else range(K - 1)     # 形态 B：少一条 sampled_5
        for i in keep:
            rows.append(_row(e, i, "greedy" if i == 0 else "sampled"))
    p["raw_results"].parent.mkdir(parents=True, exist_ok=True)
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    man = WorkManifest(p["manifest"])
    man.register(u for e in eids for u in _units(e))
    for r in rows:
        man.mark(r["unit_id"], "done")
    for e in (a_old, a_new):
        man.mark(f"{e}__greedy_0", "excluded", reason=TRIG)
        for u in _units(e)[1:]:
            man.mark(u, "excluded", reason=A_REASON)
    man.mark(f"{b}__sampled_5", "excluded", reason=TRIG)

    lines = [] if drop_diag else [_diag_bytes(f"{a_new}__greedy_0"),
                                  _diag_bytes(f"{b}__sampled_5")]
    p["diagnostics"].write_bytes(b"".join(x + b"\r\n" for x in lines))
    a_sha = hashlib.sha256(_diag_bytes(f"{a_new}__greedy_0")).hexdigest()
    if tamper_diag:
        a_sha = "f" * 64
    audit = [_a_record(a_old, persisted=False),
             _a_record(a_new, persisted=a_new_persisted, diag_sha=a_sha),
             _b_record(b, hashlib.sha256(_diag_bytes(f"{b}__sampled_5")).hexdigest())]
    p["technical_exclusions"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit), encoding="utf-8")
    hd = p["hidden_dir"]
    hd.mkdir(parents=True, exist_ok=True)
    for r in rows:
        np.save(hd / f"{r['unit_id']}.npy", np.zeros(3584, dtype=np.float32))
    return p, records


def _finalize(tmp_path, records):
    return prod.finalize_arm("mcq", tmp_path, records, cfg=V3AnalysisConfig(),
                             run_fingerprint={"run_hash": "x" * 64})


def test_two_form_a_plus_one_form_b(tmp_path, monkeypatch):
    """多组共存：计数、过滤、F5 分组全部正确。"""
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, n=20)
    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 20
    assert rep["technical_exclusions"] == 3
    assert rep["retained_questions"] == 17
    assert rep["actual_finalized_rows"] == 17
    assert rep["retained_raw_units"] == 17 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 5      # 只有形态 B 贡献
    assert rep["expected_total_raw_rows"] == 17 * K + 5
    assert rep["excluded_example_ids"] == ["q0002", "q0005", "q0008"]
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert len(fin) == 17
    assert not ({r["example_id"] for r in fin} & {"q0002", "q0005", "q0008"})
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 17}, "F5 只能收到 17 个完整 K=6 group"
    # 形态 B 的五条 raw/hidden 物理保留
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in raw if r["example_id"] == "q0008") == 5
    for uid in _units("q0008")[:5]:
        assert (p["hidden_dir"] / f"{uid}.npy").exists()


def test_form_a_without_persisted_diagnostic_is_still_valid(tmp_path, monkeypatch):
    """diagnostic_persisted=false 的历史形态 A 仍合法（不因缺诊断被否定）。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, n=20, drop_diag=False)
    assert _finalize(tmp_path, records)["technical_exclusions"] == 3


def test_form_a_with_persisted_diagnostic_requires_matching_record(tmp_path, monkeypatch):
    """persisted=true 但 record SHA 对不上 → 拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, n=20, tamper_diag=True)
    with pytest.raises(ValueError, match="字节已变"):
        _finalize(tmp_path, records)


def test_form_a_with_persisted_diagnostic_requires_record_present(tmp_path, monkeypatch):
    """persisted=true 但 diagnostics 里没有该记录 → 拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, n=20, drop_diag=True)
    with pytest.raises(ValueError, match="不存在"):
        _finalize(tmp_path, records)


def test_form_a_persisted_requires_diagnostic_fields(tmp_path, monkeypatch):
    """persisted=true 却缺 diagnostic_unit_id / record SHA 字段 → 拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _build(tmp_path, n=20)
    recs = [json.loads(x) for x in
            p["technical_exclusions"].read_text(encoding="utf-8").splitlines()]
    for r in recs:
        if r["example_id"] == "q0005":
            r.pop("diagnostic_record_sha256")
    p["technical_exclusions"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    with pytest.raises(ValueError, match="缺字段"):
        _finalize(tmp_path, records)


def test_two_form_a_plus_two_partial_j1_and_j5(tmp_path, monkeypatch):
    """两个形态 A + **两个**泛化 partial（j=1 与 j=5）共存时的过滤、F5 与三口径。"""
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, n=20)                       # a_old/a_new/b(j=5) 三组
    # 追加第四组：q0012 的 sampled_1 失败（j=1）—— 只保留 greedy_0。
    eid, j = "q0012", 1
    trig, suffix = f"{eid}__sampled_{j}", [f"{eid}__sampled_{i}" for i in range(j + 1, K)]
    rows = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["example_id"] != eid or r["unit_id"] == f"{eid}__greedy_0"]
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    man = WorkManifest(p["manifest"])
    man.mark(trig, "excluded", reason=TRIG)
    for uid in suffix:
        man.mark(uid, "excluded", reason=POST_REASON)
    for uid in suffix + [trig]:
        (p["hidden_dir"] / f"{uid}.npy").unlink(missing_ok=True)
    with open(p["diagnostics"], "ab") as fh:
        fh.write(_diag_bytes(trig) + b"\r\n")
    rec = {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
           "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
           "action": prod.ACTION_PARTIAL_GROUP, "trigger_unit_id": trig,
           "trigger_sample_type": "sampled", "trigger_sample_index": j,
           "trigger_reason": TRIG, "manifest_pattern": "1_done_1_excluded_4_post_excluded",
           "preserved_raw_unit_ids": [f"{eid}__greedy_0"],
           "trigger_excluded_unit_id": trig,
           "post_trigger_excluded_unit_ids": suffix,
           "post_trigger_exclusion_reason": POST_REASON,
           "raw_preserved_not_analyzed": True, "exclusion_reason": B_REASON,
           "diagnostic_unit_id": trig,
           "diagnostic_record_sha256": hashlib.sha256(_diag_bytes(trig)).hexdigest(),
           "diagnostic_record_sha256_caliber": prod.DIAGNOSTIC_RECORD_SHA_CALIBER,
           "diagnostics_prefix_bytes_at_recovery": 0,
           "diagnostics_prefix_sha256_at_recovery": "0" * 64,
           "diagnostic_persisted": True,
           "frozen_list_sha256": "a" * 64, "run_fingerprint_sha256": "b" * 64,
           "classification": "runtime_technical_attrition",
           "protocol_deviation": True, "preregistration_files_modified": False}
    with open(p["technical_exclusions"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 20
    assert rep["technical_exclusions"] == 4
    assert rep["retained_questions"] == 16
    assert rep["actual_finalized_rows"] == 16
    assert rep["retained_raw_units"] == 16 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 6          # 5（j=5）+ 1（j=1）
    assert rep["expected_total_raw_rows"] == 16 * K + 6
    assert rep["excluded_example_ids"] == ["q0002", "q0005", "q0008", "q0012"]
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 16}, "F5 只能收到 16 个完整 K=6 group"
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert not ({r["example_id"] for r in fin} & {"q0002", "q0005", "q0008", "q0012"})
    assert (p["hidden_dir"] / f"{eid}__greedy_0.npy").exists()       # j=1 的一条物理保留


def test_multi_group_rejects_stray_excluded(tmp_path, monkeypatch):
    """多组共存时，未登记的 excluded 仍被拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _build(tmp_path, n=20)
    WorkManifest(p["manifest"]).mark("q0011__sampled_3", "excluded", reason=TRIG)
    with pytest.raises(ValueError, match="未登记"):
        _finalize(tmp_path, records)
