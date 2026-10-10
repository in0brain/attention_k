"""预授权恢复工具 `scripts/recover_forward_nonfinite.py`（仅 MCQ，fail-closed，两阶段）。

覆盖四条硬约束：
  1. 仅 MCQ —— CLI 不接受其它 arm，也不接受 example_id / 强制开关
  2. hidden 不变量按 **requires_hidden** 判定：ineligible 行没有 hidden 是**合法终态**
  3. 两阶段 —— 任一候选被拒时 audit / manifest / raw / hidden **逐字节不变**
  4. 崩溃恢复 —— 「已有本工具创建且逐字段一致的 audit、manifest 未收敛」是合法候选
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

from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.generation.manifest import WorkManifest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "recover_forward_nonfinite", _REPO_ROOT / "scripts" / "recover_forward_nonfinite.py")
rfn = importlib.util.module_from_spec(_spec)
# 必须先登记到 sys.modules：@dataclass 会按 cls.__module__ 回查模块命名空间。
sys.modules[_spec.name] = rfn
_spec.loader.exec_module(rfn)

K = 6
N = 10


def _qhash(eid: str) -> str:
    return hashlib.sha256(eid.encode()).hexdigest()


def _gid(eid: str) -> str:
    return f"cybermetric_question_{_qhash(eid)[:16]}"


def _eid(i: int) -> str:
    return f"cybermetric_{i:06d}"


def _row(eid, i, st, *, eligible=True):
    return {"unit_id": f"{eid}__{st}_{i}", "example_id": eid, "sample_type": st,
            "sample_index": i, "eligible_for_primary": eligible, "parsed_label": "A"}


def _diag(uid, eid, *, row_nonfinite=152064, row_size=152064, hidden_nonfinite=0,
          logits_all_nan=True, group_id=None):
    logits = {c: ({"value": None, "nonfinite_kind": "nan"} if logits_all_nan or c != "A"
                  else {"value": 1.25}) for c in "ABCD"}
    return {"schema_version": "stage1_forward_nonfinite_diagnostic_v1", "arm": "mcq",
            "unit_id": uid, "example_id": eid,
            "group_id": group_id if group_id is not None else _gid(eid),
            "reason": "forward_nonfinite", "sample_type": uid.split("__")[1].split("_")[0],
            "candidate_logits": logits, "row_nonfinite": row_nonfinite, "row_size": row_size,
            "hidden_nonfinite": hidden_nonfinite, "hidden_size": 3584,
            "letter_token_id": 356, "letter_token_form": "space"}


def _build(tmp_path, *, failures=None, ineligible=None, diag_over=None,
           drop_diag=(), extra_pending_status=None):
    """failures = {example_id: j}（j=0 表示 greedy 失败 → 形态 A）。"""
    failures = failures or {}
    ineligible = ineligible or set()
    p = prod.paths_for(tmp_path, "mcq")
    p["raw_results"].parent.mkdir(parents=True, exist_ok=True)
    eids = [_eid(i) for i in range(N)]
    frozen = tmp_path / "mcq_primary_questions.jsonl"
    frozen.write_text("".join(
        json.dumps({"original_index": i, "question_hash": _qhash(_eid(i))}) + "\n"
        for i in range(N)), encoding="utf-8")
    p["fingerprint"].write_text(json.dumps({"run_hash": "f" * 64}), encoding="utf-8")

    rows, man_marks = [], []
    for e in eids:
        if e in failures and failures[e] == 0:                 # 形态 A：raw 0 条
            man_marks.append((f"{e}__greedy_0", "excluded", "forward_nonfinite"))
            continue
        keep = range(K) if e not in failures else range(failures[e])
        for i in keep:
            rows.append(_row(e, i, "greedy" if i == 0 else "sampled",
                             eligible=f"{e}__{'greedy' if i == 0 else 'sampled'}_{i}"
                             not in ineligible))
        if e in failures:
            man_marks.append((f"{e}__sampled_{failures[e]}", "excluded", "forward_nonfinite"))
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    man = WorkManifest(p["manifest"])
    man.register(f"{e}__{t}_{i}" for e in eids
                 for t, i in [("greedy", 0)] + [("sampled", j) for j in range(1, K)])
    for r in rows:
        man.mark(r["unit_id"], "done")
    for uid, st, reason in man_marks:
        man.mark(uid, st, reason=reason)
    if extra_pending_status:
        for uid, st, reason in extra_pending_status:
            man.mark(uid, st, **({"reason": reason} if reason else {}))

    lines = []
    for e, j in failures.items():
        uid = f"{e}__greedy_0" if j == 0 else f"{e}__sampled_{j}"
        if uid in drop_diag:
            continue
        d = _diag(uid, e)
        d.update((diag_over or {}).get(e, {}))
        lines.append(json.dumps(d, ensure_ascii=False).encode("utf-8"))
    p["diagnostics"].write_bytes(b"".join(x + b"\r\n" for x in lines))

    hd = p["hidden_dir"]
    hd.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if prod.requires_hidden("mcq", r):
            np.save(hd / f"{r['unit_id']}.npy", np.zeros(8, dtype=np.float32))
    return p, frozen


def _snapshot(p):
    out = {}
    for k in ("technical_exclusions", "manifest", "raw_results", "diagnostics"):
        out[k] = p[k].read_bytes() if p[k].exists() else None
    out["hidden"] = sorted(f.name for f in p["hidden_dir"].glob("*.npy"))
    return out


def _run(tmp_path, frozen, *, apply=False):
    return rfn.run_recovery(out_dir=tmp_path, frozen_list=frozen, apply=apply)


# ── 仅 MCQ ─────────────────────────────────────────────────────────────────
def test_cli_rejects_non_mcq_arm():
    with pytest.raises(SystemExit) as e:
        rfn.main(["--arm", "h1"])
    assert e.value.code != 0


def test_cli_has_no_example_id_or_force_switches():
    opts = set()
    for a in rfn.main.__globals__["argparse"].ArgumentParser().parse_known_args.__self__ \
            .__class__.__mro__[:1]:                     # 占位，真正断言见下
        opts |= set()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["mcq"], required=True)
    ap.add_argument("--apply", action="store_true")
    allowed = {"-h", "--help", "--arm", "--apply"}
    src = (_REPO_ROOT / "scripts" / "recover_forward_nonfinite.py").read_text(encoding="utf-8")
    declared = {line.split('"')[1] for line in src.splitlines()
                if "ap.add_argument(" in line and '"' in line}
    assert declared <= allowed - {"-h", "--help"}, f"CLI 多出开关：{declared}"
    for banned in ("example_id", "--force", "--skip", "--yes", "--unit-id", "--reason"):
        assert f'"{banned}"' not in src.split("def main(")[1], f"CLI 不得接受 {banned}"


# ── dry-run 零写入 ─────────────────────────────────────────────────────────
def test_dry_run_writes_nothing(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 1})
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=False)
    assert res["applied"] is False and len(res["candidates"]) == 1
    assert _snapshot(p) == before


# ── 形态 A ─────────────────────────────────────────────────────────────────
def test_form_a_greedy_requires_all_sampled_pending(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 0})
    res = _run(tmp_path, frozen, apply=True)
    assert res["audit_appended"] == 1 and res["manifest_events"] == 5
    st = WorkManifest(p["manifest"]).current_status()
    assert all(st[f"{_eid(3)}__sampled_{i}"] == "excluded" for i in range(1, K))
    rec = [json.loads(x) for x in
           p["technical_exclusions"].read_text(encoding="utf-8").splitlines()][0]
    assert rec["action"] == prod.ACTION_FULL_GROUP
    assert rec["created_by"] == rfn.CREATED_BY


def test_form_a_rejects_when_a_sampled_is_not_pending(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 0},
                       extra_pending_status=[(f"{_eid(3)}__sampled_2", "done", None)])
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="全为 pending"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_form_a_rejects_when_raw_present(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 0})
    with open(p["raw_results"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row(_eid(3), 1, "sampled")) + "\n")
    with pytest.raises(rfn.RecoveryRejected, match="形态 A"):
        _run(tmp_path, frozen, apply=True)


# ── generalized partial：j = 1..5 ──────────────────────────────────────────
@pytest.mark.parametrize("j", [1, 2, 3, 4, 5])
def test_partial_j(tmp_path, j):
    p, frozen = _build(tmp_path, failures={_eid(4): j})
    res = _run(tmp_path, frozen, apply=True)
    assert res["audit_appended"] == 1
    assert res["manifest_events"] == K - 1 - j
    rec = [json.loads(x) for x in
           p["technical_exclusions"].read_text(encoding="utf-8").splitlines()][0]
    assert rec["action"] == prod.ACTION_PARTIAL_GROUP
    assert rec["trigger_sample_index"] == j
    assert isinstance(rec["trigger_excluded_unit_id"], str)      # **标量**
    assert len(rec["preserved_raw_unit_ids"]) == j
    assert len(rec["post_trigger_excluded_unit_ids"]) == K - 1 - j
    assert rec["post_trigger_exclusion_reason"] in rfn.AUTHORIZED_POST_TRIGGER_REASONS
    st = WorkManifest(p["manifest"]).current_status()
    assert all(st[u] == "excluded" for u in rfn._split(_eid(4), j)[2])


def test_partial_rejects_suffix_not_pending(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1},
                       extra_pending_status=[(f"{_eid(4)}__sampled_3", "done", None)])
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="全为 pending"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_partial_rejects_raw_prefix_mismatch(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    rows = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["unit_id"] != f"{_eid(4)}__sampled_1"]
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(rfn.RecoveryRejected, match="raw 集合与 preserved 前缀不符"):
        _run(tmp_path, frozen, apply=True)


def test_partial_rejects_preserved_not_done(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2},
                       extra_pending_status=[(f"{_eid(4)}__sampled_1", "pending", None)])
    with pytest.raises(rfn.RecoveryRejected, match="preserved 前缀非全 done"):
        _run(tmp_path, frozen, apply=True)


# ── 诊断数值判据（A3）──────────────────────────────────────────────────────
def test_rejects_partial_nan_row(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1},
                       diag_over={_eid(4): {"row_nonfinite": 5}})
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="非\\*\\*整行\\*\\*非有限"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_rejects_hidden_nonfinite(tmp_path):
    _p, frozen = _build(tmp_path, failures={_eid(4): 1},
                        diag_over={_eid(4): {"hidden_nonfinite": 3}})
    with pytest.raises(rfn.RecoveryRejected, match="hidden 含非有限值"):
        _run(tmp_path, frozen, apply=True)


def test_rejects_when_not_all_candidate_logits_nonfinite(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    d = [json.loads(x) for x in
         p["diagnostics"].read_text(encoding="utf-8").splitlines()]
    d[0]["candidate_logits"]["A"] = {"value": 1.5}
    p["diagnostics"].write_bytes(json.dumps(d[0], ensure_ascii=False).encode() + b"\r\n")
    with pytest.raises(rfn.RecoveryRejected, match="四项全为非有限"):
        _run(tmp_path, frozen, apply=True)


def test_rejects_missing_diagnostic(tmp_path):
    _p, frozen = _build(tmp_path, failures={_eid(4): 1},
                        drop_diag=(f"{_eid(4)}__sampled_1",))
    with pytest.raises(rfn.RecoveryRejected, match="不存在该 unit 的记录"):
        _run(tmp_path, frozen, apply=True)


def test_rejects_duplicate_diagnostic(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    line = p["diagnostics"].read_bytes()
    p["diagnostics"].write_bytes(line + line)
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="diagnostics 解析失败"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_rejects_group_id_mismatch_with_frozen_list(tmp_path):
    _p, frozen = _build(tmp_path, failures={_eid(4): 1},
                        diag_over={_eid(4): {"group_id": "cybermetric_question_deadbeefdeadbeef"}})
    with pytest.raises(rfn.RecoveryRejected, match="与冻结名单派生值"):
        _run(tmp_path, frozen, apply=True)


# ── 全局前置（C1 / C3）─────────────────────────────────────────────────────
def test_rejects_when_preregistration_not_frozen(tmp_path, monkeypatch):
    _p, frozen = _build(tmp_path, failures={_eid(4): 1})
    monkeypatch.setattr(rfn, "_check_freeze", lambda: False)
    with pytest.raises(rfn.RecoveryRejected, match="预注册冻结校验未通过"):
        _run(tmp_path, frozen, apply=True)


def test_rejects_when_manifest_has_running_or_failed(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1},
                       extra_pending_status=[(f"{_eid(8)}__sampled_2", "failed", None)])
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="failed 单元"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


# ── 两阶段：任一被拒 → 合法候选也不写 ──────────────────────────────────────
def test_phase_one_rejection_blocks_all_writes(tmp_path):
    """一个合法候选 + 一个非法候选 → **零写入**，不先处理合法的那个。"""
    p, frozen = _build(tmp_path, failures={_eid(3): 1, _eid(6): 2},
                       diag_over={_eid(6): {"hidden_nonfinite": 1}})
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected) as exc:
        _run(tmp_path, frozen, apply=True)
    assert any("hidden 含非有限值" in r for r in exc.value.reasons)
    assert _snapshot(p) == before, "被拒时 audit/manifest/raw/hidden 必须逐字节不变"


def test_multiple_candidates_all_valid_are_all_written(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(2): 0, _eid(5): 1, _eid(8): 3})
    res = _run(tmp_path, frozen, apply=True)
    assert res["audit_appended"] == 3
    assert res["manifest_events"] == 5 + 4 + 2
    recs = [json.loads(x) for x in
            p["technical_exclusions"].read_text(encoding="utf-8").splitlines()]
    assert {r["example_id"] for r in recs} == {_eid(2), _eid(5), _eid(8)}
    assert res["verification"]["counts"]["pending"] == 0


# ── 幂等与崩溃恢复 ─────────────────────────────────────────────────────────
def test_apply_is_idempotent(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    _run(tmp_path, frozen, apply=True)
    after = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["audit_appended"] == 0 and res["manifest_events"] == 0
    assert _snapshot(p) == after


def test_crash_recovery_audit_written_manifest_not_converged(tmp_path):
    """audit 已在、manifest 只补了一半 → 合法「中断恢复候选」，补齐剩余。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _base = rfn.discover_and_validate(tmp_path, frozen)
    rfn._append_audit(p["technical_exclusions"], cands[0].record)
    man = WorkManifest(p["manifest"])
    man.mark(f"{_eid(4)}__sampled_2", "excluded", reason=rfn.POST_TRIGGER_REASON)
    res = _run(tmp_path, frozen, apply=True)
    assert res["audit_appended"] == 0
    assert res["manifest_events"] == 3            # 剩下 sampled_3..5
    st = WorkManifest(p["manifest"]).current_status()
    assert all(st[f"{_eid(4)}__sampled_{i}"] == "excluded" for i in range(1, K))


def test_crash_recovery_reuses_stored_prefix_snapshot(tmp_path):
    """`*_at_recovery` 沿用已落盘历史值，**不按当前 diagnostics 重算**。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    stored = dict(cands[0].record)
    rfn._append_audit(p["technical_exclusions"], stored)
    with open(p["diagnostics"], "ab") as fh:          # 之后又追加了别的诊断
        fh.write(json.dumps(_diag(f"{_eid(9)}__sampled_2", _eid(9)),
                            ensure_ascii=False).encode() + b"\r\n")
    cands2, _b2 = rfn.discover_and_validate(tmp_path, frozen)
    mine = [c for c in cands2 if c.example_id == _eid(4)][0]
    assert mine.kind == "resume"
    assert mine.record["diagnostics_prefix_bytes_at_recovery"] == \
        stored["diagnostics_prefix_bytes_at_recovery"]
    assert mine.record["diagnostics_prefix_sha256_at_recovery"] == \
        stored["diagnostics_prefix_sha256_at_recovery"]


def test_rejects_existing_audit_with_field_mismatch(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    rec = dict(cands[0].record)
    rec["question_hash"] = "0" * 64
    rfn._append_audit(p["technical_exclusions"], rec)
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="逐字段不一致"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_rejects_existing_audit_with_wrong_action(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    rec = dict(cands[0].record)
    rec["action"] = prod.ACTION_FULL_GROUP
    rfn._append_audit(p["technical_exclusions"], rec)
    before = _snapshot(p)
    # 由**复用的** finalizer 记录级校验先行拦下（partial 记录冒充 full_group 缺
    # affected_unit_ids）。两处语义同源，所以不会出现「恢复工具放过、finalize 才拒」。
    with pytest.raises(rfn.RecoveryRejected, match="缺字段 affected_unit_ids|action="):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_rejects_audit_not_created_by_this_tool(tmp_path):
    """历史逐例记录（无 created_by）不由本工具接管，交人工处理。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    rec = dict(cands[0].record)
    rec.pop("created_by")
    rfn._append_audit(p["technical_exclusions"], rec)
    with pytest.raises(rfn.RecoveryRejected, match="非本工具创建"):
        _run(tmp_path, frozen, apply=True)


def _converge_partial(p, eid, j, *, drop_created_by=True):
    """写入一条 partial audit 并把 manifest 补到完全收敛，返回该记录。"""
    cands, _b = rfn.discover_and_validate(p["manifest"].parent, p["_frozen"])
    rec = dict(next(c for c in cands if c.example_id == eid).record)
    if drop_created_by:
        rec.pop("created_by")                     # 冒充历史逐例记录
    rfn._append_audit(p["technical_exclusions"], rec)
    man = WorkManifest(p["manifest"])
    for u in rfn._split(eid, j)[2]:
        man.mark(u, "excluded", reason=rfn.POST_TRIGGER_REASON)
    return rec


def _converge_full(p, eid, *, persisted):
    """写入一条形态 A audit（persisted 可选）并把 manifest 补到完全收敛。"""
    rec = {"schema_version": rfn.SCHEMA, "arm": "mcq", "example_id": eid,
           "question_hash": _qhash(eid), "group_id": _gid(eid),
           "action": prod.ACTION_FULL_GROUP,
           "trigger_unit_id": f"{eid}__greedy_0", "trigger_reason": "forward_nonfinite",
           "affected_unit_ids": rfn._units(eid), "prior_manifest_status": {},
           "exclusion_reason": rfn.FULL_GROUP_REASON,
           "classification": "runtime_technical_attrition", "protocol_deviation": True,
           "preregistration_files_modified": False,
           "diagnostic_persisted": bool(persisted),
           "diagnostic_persistence_note": "n/a"}
    if persisted:
        idx = prod.diagnostic_record_index(p["diagnostics"])
        rec.update({"diagnostic_unit_id": f"{eid}__greedy_0",
                    "diagnostic_record_sha256": idx[f"{eid}__greedy_0"],
                    "diagnostic_record_sha256_caliber": prod.DIAGNOSTIC_RECORD_SHA_CALIBER})
    rfn._append_audit(p["technical_exclusions"], rec)
    man = WorkManifest(p["manifest"])
    for u in rfn._units(eid)[1:]:
        man.mark(u, "excluded", reason=rfn.FULL_GROUP_REASON)
    return rec


# ── converged_history：必须按 finalizer 授权语义机械验证，而非「无 pending 即跳过」──
def test_converged_history_form_a_without_persisted_diagnostic_passes(tmp_path):
    """首次失败那种 diagnostic_persisted=false 的形态 A —— 按 legacy 规则接受。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 0},
                       drop_diag=(f"{_eid(4)}__greedy_0",))
    _converge_full(p, _eid(4), persisted=False)
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == [] and res["converged_history"] == 1
    assert _snapshot(p) == before


def test_converged_history_form_a_with_persisted_diagnostic_passes(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 0})
    _converge_full(p, _eid(4), persisted=True)
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == [] and res["converged_history"] == 1
    assert _snapshot(p) == before


def test_converged_history_partial_passes(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == [] and res["converged_history"] == 1
    assert _snapshot(p) == before


@pytest.mark.parametrize("tamper", ["status", "reason"])
def test_converged_history_rejects_manifest_tampering(tmp_path, tamper):
    """manifest 的 status 或 reason 被改 → 拒绝（旧实现会误跳过）。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    man = WorkManifest(p["manifest"])
    if tamper == "status":
        man.mark(f"{_eid(4)}__sampled_4", "done")
    else:
        man.mark(f"{_eid(4)}__sampled_4", "excluded", reason="something_else")
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="未按授权语义完整收敛"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_converged_history_rejects_raw_set_tampering(tmp_path):
    """raw 集合被改（多出一条 preserved 之外的行）→ 拒绝。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    with open(p["raw_results"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row(_eid(4), 3, "sampled")) + "\n")
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="未按授权语义完整收敛"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_converged_history_rejects_persisted_diagnostic_removed(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    p["diagnostics"].write_bytes(b"")
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="未按授权语义完整收敛"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_converged_history_rejects_persisted_diagnostic_sha_changed(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    d = json.loads(p["diagnostics"].read_text(encoding="utf-8").splitlines()[0])
    d["row_nonfinite"] = d["row_size"] = 4096          # 仍是「整行」，但字节已变
    p["diagnostics"].write_bytes(json.dumps(d, ensure_ascii=False).encode() + b"\r\n")
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="未按授权语义完整收敛|字节已变"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_converged_history_rejects_duplicate_diagnostic(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 2})
    p["_frozen"] = frozen
    _converge_partial(p, _eid(4), 2)
    blob = p["diagnostics"].read_bytes()
    p["diagnostics"].write_bytes(blob + blob)
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="diagnostics 解析失败"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


def test_converged_history_without_created_by_is_skipped(tmp_path):
    """历史逐例记录（无 created_by）且该组**已收敛** → 跳过，不拒绝、不重写。

    真实 MCQ 臂就是这个状态：7 条历史记录都不是本工具写的，其中首次失败
    （cybermetric_005396）甚至没有 diagnostic。本工具必须对它们完全无动作。
    """
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    rec = dict(cands[0].record)
    rec.pop("created_by")                       # 冒充历史逐例记录
    rfn._append_audit(p["technical_exclusions"], rec)
    man = WorkManifest(p["manifest"])           # 手工补齐 manifest → 该组已收敛
    for i in range(2, K):
        man.mark(f"{_eid(4)}__sampled_{i}", "excluded", reason=rfn.POST_TRIGGER_REASON)
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == []
    assert _snapshot(p) == before


def test_converged_history_without_diagnostic_is_skipped(tmp_path):
    """已收敛的历史组即使**没有 diagnostic** 也只是跳过（对应首次失败那条）。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 0},
                       drop_diag=(f"{_eid(4)}__greedy_0",))
    rfn._append_audit(p["technical_exclusions"], {          # 复刻历史形态 A 记录的形状
        "schema_version": rfn.SCHEMA, "arm": "mcq", "example_id": _eid(4),
        "question_hash": _qhash(_eid(4)), "group_id": _gid(_eid(4)),
        "action": prod.ACTION_FULL_GROUP,
        "trigger_unit_id": f"{_eid(4)}__greedy_0", "trigger_reason": "forward_nonfinite",
        "affected_unit_ids": rfn._units(_eid(4)), "prior_manifest_status": {},
        "exclusion_reason": rfn.FULL_GROUP_REASON,
        "classification": "runtime_technical_attrition", "protocol_deviation": True,
        "preregistration_files_modified": False,
        "diagnostic_persisted": False,
        "diagnostic_persistence_note": "触发发生在诊断持久化补丁之前，未落盘。"})
    man = WorkManifest(p["manifest"])
    for i in range(1, K):
        man.mark(f"{_eid(4)}__sampled_{i}", "excluded", reason=rfn.FULL_GROUP_REASON)
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == []
    assert _snapshot(p) == before


def test_rejects_duplicate_audit_records(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    cands, _b = rfn.discover_and_validate(tmp_path, frozen)
    rfn._append_audit(p["technical_exclusions"], cands[0].record)
    blob = p["technical_exclusions"].read_bytes()
    with open(p["technical_exclusions"], "ab") as fh:
        fh.write(blob)
    before = _snapshot(p)
    with pytest.raises(rfn.RecoveryRejected, match="重复|审计真源中有 2 条"):
        _run(tmp_path, frozen, apply=True)
    assert _snapshot(p) == before


# ── hidden 不变量 ──────────────────────────────────────────────────────────
def test_ineligible_row_without_hidden_is_legal(tmp_path):
    """eligible_for_primary=false 且**没有** hidden —— 合法终态，不得判为丢失。"""
    p, frozen = _build(tmp_path, failures={_eid(4): 1},
                       ineligible={f"{_eid(7)}__sampled_2"})
    assert not (p["hidden_dir"] / f"{_eid(7)}__sampled_2.npy").exists()
    res = _run(tmp_path, frozen, apply=True)
    v = res["verification"]
    assert v["n_hidden"] == v["n_raw"] - 1          # 少一个 hidden 是合法的
    assert v["counts"]["pending"] == 0


def test_rejects_missing_hidden_for_eligible_row(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    (p["hidden_dir"] / f"{_eid(7)}__sampled_2.npy").unlink()
    with pytest.raises(RuntimeError, match="hidden 集合与应有集合不符"):
        _run(tmp_path, frozen, apply=True)


def test_rejects_orphan_hidden(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(4): 1})
    np.save(p["hidden_dir"] / "cybermetric_009999__greedy_0.npy", np.zeros(8, dtype=np.float32))
    with pytest.raises(RuntimeError, match="hidden 集合与应有集合不符"):
        _run(tmp_path, frozen, apply=True)


# ── 恢复后口径 ─────────────────────────────────────────────────────────────
def test_provenance_and_closure_after_apply(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(2): 0, _eid(5): 2})
    v = _run(tmp_path, frozen, apply=True)["verification"]
    pv = v["provenance"]
    assert pv["technical_exclusions"] == 2
    assert pv["retained_questions"] == N - 2
    assert pv["retained_raw_units"] == (N - 2) * K
    assert pv["preserved_but_not_analyzed_raw_units"] == 2      # 仅 j=2 那组
    assert pv["expected_total_raw_rows"] == (N - 2) * K + 2
    assert v["counts"]["done"] + v["counts"]["pending"] + v["counts"]["excluded"] == N * K
    # 产物可被既有 finalizer 授权门接受
    au = prod.load_authorized_exclusions("mcq", tmp_path)
    assert set(au) == {_eid(2), _eid(5)}


def test_no_candidates_is_a_clean_noop(tmp_path):
    p, frozen = _build(tmp_path, failures={})
    before = _snapshot(p)
    res = _run(tmp_path, frozen, apply=True)
    assert res["candidates"] == [] and res["audit_appended"] == 0
    assert _snapshot(p) == before


def test_tool_never_writes_raw_or_diagnostics(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 1, _eid(6): 0})
    raw_before = p["raw_results"].read_bytes()
    diag_before = p["diagnostics"].read_bytes()
    hidden_before = sorted(f.name for f in p["hidden_dir"].glob("*.npy"))
    _run(tmp_path, frozen, apply=True)
    assert p["raw_results"].read_bytes() == raw_before
    assert p["diagnostics"].read_bytes() == diag_before
    assert sorted(f.name for f in p["hidden_dir"].glob("*.npy")) == hidden_before


def test_audit_lines_are_strict_json_utf8(tmp_path):
    p, frozen = _build(tmp_path, failures={_eid(3): 1, _eid(6): 0})
    _run(tmp_path, frozen, apply=True)
    blob = p["technical_exclusions"].read_bytes()
    blob.decode("utf-8")
    n = 0
    for chunk in blob.split(b"\n"):
        line = chunk[:-1] if chunk.endswith(b"\r") else chunk
        if line.strip():
            json.loads(line.decode("utf-8"))          # 严格 JSON，单行
            n += 1
    assert n == 2
