"""预授权的 `ForwardNonFiniteError` 恢复工具（**仅 MCQ**，fail-closed，两阶段写入）。

生成进程行为**完全不变**：失败仍在 unit 边界停、CLI 非零、不重试、不改冻结参数。
本工具是**独立命令行脚本，只在生成停止后由人手动运行**，不被生成进程调用，
不加载模型、不生成任何 unit、不碰 raw / hidden。

预授权免除的是**逐例裁决**，不是校验：每一例都要逐条通过下面的 fail-closed 判据，
任一条不同即拒绝并回到人工裁决。

    形态判定由 sample_type 决定，**不可人为指定**：
      greedy_0 失败   → 形态 A（group_exclusion）：该题 raw 恰好 0 条，
                        且 sampled_1..5 **全部仍为 pending**
      sampled_j 失败  → generalized partial：
                        preserved = greedy_0 + sampled_1..j-1 全 done，
                                    且 raw 集合**精确等于** preserved
                        trigger   = sampled_j
                        suffix    = sampled_{j+1}..sampled_5 **全部仍为 pending**

待处理集合完全由**产物本身**推导，人不能指定要排除谁：CLI 不接受
example_id / unit_id / reason / j / action，也没有任何强制或跳过校验的开关。

**两阶段**：Phase 1 发现并完整验证全部候选，**零写入**；只有全部通过才进入 Phase 2
逐例写入。任一候选被拒 → audit / manifest / raw / hidden 逐字节不变，
**不先处理合法候选**。

**仅支持 MCQ**：形态判定、`n_questions × K` 对账与 finalizer 授权都是 MCQ 专用。
H1 若需要另开 sprint —— 本工具不声称支持 H1。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation.v3_freeze import check_v3_frozen        # noqa: E402
from recover_attention.evaluation.v3_production import (                  # noqa: E402
    ACTION_FULL_GROUP, ACTION_PARTIAL_GROUP, AUTHORIZED_POST_TRIGGER_REASONS,
    DIAGNOSTIC_RECORD_SHA_CALIBER, diagnostic_record_index,
    load_authorized_exclusions, paths_for, requires_hidden)
# **刻意复用 finalizer 的私有授权验证函数**：历史组是否已正确收敛，必须按 finalize
# 时的同一套语义机械判定。若在此另写一套「等价」检查，两处语义早晚漂移，
# 恢复工具就会放过 finalizer 会拒的状态。
from recover_attention.evaluation.v3_production import \
    _verify_group_exclusion as verify_group_exclusion                     # noqa: E402
from recover_attention.generation.manifest import WorkManifest, write_atomic  # noqa: E402

ARM = "mcq"
K = 6
TRIGGER_REASON = "forward_nonfinite"
FULL_GROUP_REASON = "group_excluded_due_greedy_forward_nonfinite"
PARTIAL_REASON = "partial_group_excluded_due_sampled_forward_nonfinite"
POST_TRIGGER_REASON = "group_excluded_due_sampled_forward_nonfinite"
SCHEMA = "stage1_technical_exclusion_v1"
#: 本工具写出的记录带此标记，便于把「工具创建的记录」与历史逐例记录区分开。
CREATED_BY = "recover_forward_nonfinite_v1"

DEFAULT_OUT_DIR = ROOT / "outputs/stage1/production"
DEFAULT_FROZEN_LIST = ROOT / "outputs/stage1/samples/mcq_primary_questions.jsonl"

FULL_GROUP_KEYS = {
    "schema_version", "arm", "example_id", "question_hash", "group_id", "action",
    "trigger_unit_id", "trigger_reason", "affected_unit_ids", "prior_manifest_status",
    "exclusion_reason", "diagnostic_unit_id", "diagnostic_record_sha256",
    "diagnostic_record_sha256_caliber", "diagnostics_prefix_bytes_at_recovery",
    "diagnostics_prefix_sha256_at_recovery", "diagnostic_persisted",
    "frozen_list_sha256", "run_fingerprint_sha256", "classification",
    "protocol_deviation", "preregistration_files_modified", "created_by",
}
PARTIAL_KEYS = {
    "schema_version", "arm", "example_id", "question_hash", "group_id", "action",
    "trigger_unit_id", "trigger_sample_type", "trigger_sample_index", "trigger_reason",
    "manifest_pattern", "preserved_raw_unit_ids", "trigger_excluded_unit_id",
    "post_trigger_excluded_unit_ids", "post_trigger_exclusion_reason",
    "raw_preserved_not_analyzed", "exclusion_reason", "diagnostic_unit_id",
    "diagnostic_record_sha256", "diagnostic_record_sha256_caliber",
    "diagnostics_prefix_bytes_at_recovery", "diagnostics_prefix_sha256_at_recovery",
    "diagnostic_persisted", "frozen_list_sha256", "run_fingerprint_sha256",
    "classification", "protocol_deviation", "preregistration_files_modified",
    "created_by",
}


class RecoveryRejected(RuntimeError):
    """Phase 1 拒绝：**零写入**，回到人工裁决。`.reasons` 列出全部拒绝理由。"""

    def __init__(self, reasons: list[str]) -> None:
        # 拒绝理由进入异常消息本身：调用方/测试无需再翻 `.reasons` 才知道被拒原因。
        super().__init__(f"{len(reasons)} 项拒绝；未写入任何文件 —— " + "；".join(reasons))
        self.reasons = list(reasons)


@dataclass
class Candidate:
    unit_id: str
    example_id: str
    action: str
    kind: str                       # "new"（尚无 audit）｜ "resume"（audit 已在，manifest 未收敛）
    j: int | None                   # partial 的 sampled_j；形态 A 为 None
    record: dict = field(default_factory=dict)
    pending_marks: list[tuple[str, str]] = field(default_factory=list)   # (unit_id, reason)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for chunk in path.read_bytes().split(b"\n"):
        line = chunk[:-1] if chunk.endswith(b"\r") else chunk
        if line.strip():
            out.append(json.loads(line.decode("utf-8")))
    return out


def _check_freeze() -> bool:
    """预注册冻结门。测试可 monkeypatch 本函数以覆盖 C1 的拒绝路径。"""
    return bool(check_v3_frozen().match)


def _units(eid: str) -> list[str]:
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]


def _split(eid: str, j: int) -> tuple[list[str], str, list[str]]:
    u = _units(eid)
    return u[:j], u[j], u[j + 1:]


def _frozen_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[f"cybermetric_{int(row['original_index']):06d}"] = row["question_hash"]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1：发现 + 完整验证（**零写入**）
# ══════════════════════════════════════════════════════════════════════════════
def _validate_diagnostic(uid: str, diag: dict | None, eid: str, gid: str,
                         rej: list[str]) -> bool:
    """A2–A4 的数值与身份判据。任一不同即拒。"""
    if diag is None:
        rej.append(f"{uid}: diagnostics 中不存在该 unit 的记录")
        return False
    ok = True
    if diag.get("reason") != TRIGGER_REASON:
        rej.append(f"{uid}: diagnostic reason={diag.get('reason')!r} 非 {TRIGGER_REASON}")
        ok = False
    row_n, row_size = diag.get("row_nonfinite"), diag.get("row_size")
    if not isinstance(row_n, int) or not isinstance(row_size, int) or row_size <= 0:
        rej.append(f"{uid}: row_nonfinite/row_size 非正整数")
        ok = False
    elif row_n != row_size:
        rej.append(f"{uid}: 非**整行**非有限（row_nonfinite {row_n} != row_size {row_size}）"
                   f" —— 形态不同，回到人工裁决")
        ok = False
    if diag.get("hidden_nonfinite") != 0:
        rej.append(f"{uid}: hidden 含非有限值（hidden_nonfinite={diag.get('hidden_nonfinite')}）"
                   f" —— 形态不同，回到人工裁决")
        ok = False
    logits = diag.get("candidate_logits")
    if not isinstance(logits, dict) or not logits:
        rej.append(f"{uid}: candidate_logits 缺失或非 dict")
        ok = False
    elif not all(isinstance(v, dict) and v.get("value") is None and v.get("nonfinite_kind")
                 for v in logits.values()):
        rej.append(f"{uid}: candidate_logits 并非四项全为非有限 —— 形态不同，回到人工裁决")
        ok = False
    if diag.get("example_id") != eid:
        rej.append(f"{uid}: diagnostic example_id={diag.get('example_id')!r} 与 unit_id 不符")
        ok = False
    if diag.get("group_id") != gid:
        rej.append(f"{uid}: diagnostic group_id={diag.get('group_id')!r} 与冻结名单派生值 {gid!r} 不符")
        ok = False
    return ok


def _build_record(*, action: str, eid: str, uid: str, j: int | None, qhash: str, gid: str,
                  rec_sha: str, prefix_bytes: int, prefix_sha: str, status: dict[str, str],
                  frozen_sha: str, fp_sha: str) -> dict:
    base = {
        "schema_version": SCHEMA, "arm": ARM, "example_id": eid,
        "question_hash": qhash, "group_id": gid, "action": action,
        "trigger_unit_id": uid, "trigger_reason": TRIGGER_REASON,
        "diagnostic_unit_id": uid, "diagnostic_record_sha256": rec_sha,
        "diagnostic_record_sha256_caliber": DIAGNOSTIC_RECORD_SHA_CALIBER,
        "diagnostics_prefix_bytes_at_recovery": prefix_bytes,
        "diagnostics_prefix_sha256_at_recovery": prefix_sha,
        "diagnostic_persisted": True,
        "frozen_list_sha256": frozen_sha, "run_fingerprint_sha256": fp_sha,
        "classification": "runtime_technical_attrition",
        "protocol_deviation": True, "preregistration_files_modified": False,
        "created_by": CREATED_BY,
    }
    if action == ACTION_FULL_GROUP:
        units = _units(eid)
        base.update({
            "affected_unit_ids": units,
            "prior_manifest_status": {u: status.get(u) for u in units},
            "exclusion_reason": FULL_GROUP_REASON,
        })
        return base
    preserved, trigger, suffix = _split(eid, int(j))
    base.update({
        "trigger_sample_type": "sampled", "trigger_sample_index": int(j),
        "manifest_pattern": f"{j}_done_1_excluded_{len(suffix)}_post_excluded",
        "preserved_raw_unit_ids": preserved,
        "trigger_excluded_unit_id": trigger,          # **标量字符串**
        "post_trigger_excluded_unit_ids": suffix,
        "post_trigger_exclusion_reason": POST_TRIGGER_REASON,
        "raw_preserved_not_analyzed": True,
        "exclusion_reason": PARTIAL_REASON,
    })
    return base


def discover_and_validate(out_dir: Path, frozen_list: Path) -> tuple[list[Candidate], dict]:
    """Phase 1。返回 (候选列表, 起点 SHA 快照)。任一拒绝即抛 `RecoveryRejected`。"""
    p = paths_for(out_dir, ARM)
    rej: list[str] = []

    # ── C1 预注册冻结 ────────────────────────────────────────────────────────
    if not _check_freeze():
        raise RecoveryRejected(["预注册冻结校验未通过（check_v3_frozen().match 为假）"])

    baseline = {k: (_sha_file(p[k]) if p[k].exists() else None)
                for k in ("technical_exclusions", "manifest", "raw_results", "diagnostics")}
    baseline_bytes = {k: (p[k].stat().st_size if p[k].exists() else 0)
                      for k in baseline}

    man = WorkManifest(p["manifest"])
    status = man.current_status()
    events = {ev["unit_id"]: ev for ev in man._events}            # noqa: SLF001
    frozen = _frozen_index(frozen_list)
    n_expected = len(frozen) * K
    if len(status) != n_expected:
        rej.append(f"manifest 注册 {len(status)} 个 unit，冻结名单推出 {n_expected}；拒绝")
    # ── C3 无 running / failed ──────────────────────────────────────────────
    for bad in ("running", "failed"):
        n = sum(1 for v in status.values() if v == bad)
        if n:
            rej.append(f"manifest 存在 {n} 个 {bad} 单元；先人工处理后再恢复")

    raw_rows = _read_jsonl(p["raw_results"])
    raw_by_example: dict[str, set[str]] = {}
    for r in raw_rows:
        raw_by_example.setdefault(r["example_id"], set()).add(r["unit_id"])
    diag_by_unit = {d["unit_id"]: d for d in _read_jsonl(p["diagnostics"])}
    try:
        diag_index = diagnostic_record_index(p["diagnostics"])
    except ValueError as exc:                    # 重复 unit_id / 缺 unit_id / 坏行
        raise RecoveryRejected([f"diagnostics 解析失败：{exc}"]) from exc
    # 记录级白名单与必填字段：直接用 finalizer 的同一函数，不另写一套。
    try:
        load_authorized_exclusions(ARM, out_dir)
    except ValueError as exc:
        raise RecoveryRejected([f"审计真源记录级校验失败：{exc}"]) from exc
    audit = _read_jsonl(p["technical_exclusions"])
    audit_by_example: dict[str, list[dict]] = {}
    for a in audit:
        audit_by_example.setdefault(a["example_id"], []).append(a)
    frozen_sha = _sha_file(frozen_list)
    fp_sha = _sha_file(p["fingerprint"])

    triggers = sorted(u for u, v in status.items()
                      if v == "excluded" and events[u].get("reason") == TRIGGER_REASON)
    cands: list[Candidate] = []
    n_converged = 0                      # 已按授权语义验证通过的历史组（跳过、不重写）
    for uid in triggers:
        eid, tail = uid.split("__", 1)
        prior = audit_by_example.get(eid, [])
        if len(prior) > 1:
            rej.append(f"{eid}: 审计真源中有 {len(prior)} 条记录；拒绝")
            continue
        # ── B 形态判定（由 sample_type 决定）—— 先做**纯结构**部分 ──────────
        if tail == "greedy_0":
            action, j = ACTION_FULL_GROUP, None
            targets = _units(eid)[1:]
            want_reason = {u: FULL_GROUP_REASON for u in targets}
        elif tail.startswith("sampled_"):
            j = int(tail.rsplit("_", 1)[1])
            if not 1 <= j <= K - 1:
                rej.append(f"{uid}: sample_index {j} 超出 1..{K - 1}；拒绝")
                continue
            action = ACTION_PARTIAL_GROUP
            _pre0, _trig0, targets = _split(eid, j)
            want_reason = {u: POST_TRIGGER_REASON for u in targets}
        else:
            rej.append(f"{uid}: 无法解析 sample_type；拒绝")
            continue

        # ── 已有 audit 且无 pending recovery target → 必须**机械验证已完整收敛** ──
        # 「无 pending」**不等于**「已正确收敛」：manifest 状态/reason 被改、raw 集合
        # 错误、persisted diagnostic 被篡改，都可能表现为「无 pending」。这里按
        # finalizer 的同一套授权语义验证（含 diagnostic_persisted=false 的 legacy 规则）；
        # 不要求 created_by，也不重新裁决历史裁定 —— audit 是授权真源，只做一致性验证。
        if prior and not any(status.get(u) == "pending" for u in targets):
            try:
                verify_group_exclusion(ARM, prior[0], events, K,
                                       raw_by_example.get(eid, set()), diag_index)
            except (ValueError, KeyError) as exc:
                rej.append(f"{eid}: 既有 audit 但该组未按授权语义完整收敛：{exc}")
                continue
            n_converged += 1
            continue                     # converged_history：跳过且**不重写**

        # ── C2 冻结名单 ────────────────────────────────────────────────────
        qhash = frozen.get(eid)
        if qhash is None:
            rej.append(f"{eid}: 不在冻结名单中；拒绝")
            continue
        gid = f"cybermetric_question_{qhash[:16]}"
        # ── A2–A4 诊断 ─────────────────────────────────────────────────────
        if not _validate_diagnostic(uid, diag_by_unit.get(uid), eid, gid, rej):
            continue
        rec_sha = diag_index.get(uid)

        # ── B 形态判定的产物侧判据 ─────────────────────────────────────────
        if action == ACTION_FULL_GROUP:
            if raw_by_example.get(eid):
                rej.append(f"{eid}: greedy 失败（形态 A）但 raw 中有 "
                           f"{len(raw_by_example[eid])} 条；拒绝")
                continue
        else:
            preserved = _split(eid, int(j))[0]
            bad_pre = [u for u in preserved if status.get(u) != "done"]
            if bad_pre:
                rej.append(f"{eid}: preserved 前缀非全 done（{bad_pre[:3]}）；拒绝")
                continue
            if raw_by_example.get(eid, set()) != set(preserved):
                got = raw_by_example.get(eid, set())
                rej.append(f"{eid}: raw 集合与 preserved 前缀不符"
                           f"（缺 {sorted(set(preserved) - got)[:3]}，"
                           f"多 {sorted(got - set(preserved))[:3]}）；拒绝")
                continue

        # ── 新候选 vs 中断恢复候选 ─────────────────────────────────────────
        if not prior:
            kind = "new"
            not_pending = [u for u in targets if status.get(u) != "pending"]
            if not_pending:
                rej.append(f"{eid}: 首次恢复要求目标 unit 全为 pending，"
                           f"实为 {[(u, status.get(u)) for u in not_pending][:3]}；拒绝")
                continue
            blob = p["diagnostics"].read_bytes()
            prefix_bytes, prefix_sha = len(blob), hashlib.sha256(blob).hexdigest()
        else:
            kind = "resume"
            got = prior[0]
            if got.get("created_by") != CREATED_BY:
                rej.append(f"{eid}: 既有审计记录非本工具创建"
                           f"（created_by={got.get('created_by')!r}）；交由人工处理")
                continue
            if got.get("action") != action:
                rej.append(f"{eid}: 既有审计记录 action={got.get('action')!r}，"
                           f"按当前形态应为 {action!r}；拒绝")
                continue
            # `*_at_recovery` 是**时点快照**，沿用已落盘历史值，不按当前文件重算
            prefix_bytes = got.get("diagnostics_prefix_bytes_at_recovery")
            prefix_sha = got.get("diagnostics_prefix_sha256_at_recovery")
            # 目标 unit 只能是 pending，或**已按预期 reason 收敛**
            bad = [(u, status.get(u), events.get(u, {}).get("reason"))
                   for u in targets
                   if not (status.get(u) == "pending"
                           or (status.get(u) == "excluded"
                               and events[u].get("reason") == want_reason[u]))]
            if bad:
                rej.append(f"{eid}: 中断恢复时目标 unit 状态异常 {bad[:3]}；拒绝")
                continue

        record = _build_record(action=action, eid=eid, uid=uid, j=j, qhash=qhash, gid=gid,
                               rec_sha=rec_sha, prefix_bytes=prefix_bytes,
                               prefix_sha=prefix_sha, status=status,
                               frozen_sha=frozen_sha, fp_sha=fp_sha)
        want_keys = FULL_GROUP_KEYS if action == ACTION_FULL_GROUP else PARTIAL_KEYS
        missing, extra = want_keys - set(record), set(record) - want_keys
        if missing or extra:
            rej.append(f"{eid}: 生成的记录字段不符 missing={sorted(missing)} extra={sorted(extra)}")
            continue
        if kind == "resume" and prior[0] != record:
            diff = sorted(k for k in set(record) | set(prior[0])
                          if record.get(k) != prior[0].get(k))
            rej.append(f"{eid}: 既有审计记录与重建结果逐字段不一致，差异 {diff}；"
                       f"**不覆盖、不重写**，交由人工处理")
            continue

        marks = [(u, want_reason[u]) for u in targets if status.get(u) == "pending"]
        cands.append(Candidate(unit_id=uid, example_id=eid, action=action, kind=kind,
                               j=j, record=record, pending_marks=marks))

    if rej:
        raise RecoveryRejected(rej)
    return cands, {"sha": baseline, "bytes": baseline_bytes,
                   "converged_history": n_converged}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2：逐例写入（只有 Phase 1 全部通过才执行）
# ══════════════════════════════════════════════════════════════════════════════
def _append_audit(path: Path, record: dict) -> bool:
    existing = _read_jsonl(path)
    if any(r.get("example_id") == record["example_id"] for r in existing):
        return False                                   # 幂等：Phase 1 已确认逐字段一致
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         allow_nan=False).encode("utf-8") + b"\r\n"
    if not existing:
        write_atomic(path, payload.decode("utf-8").replace("\r\n", "\n"))
    else:
        before = path.read_bytes()
        with open(path, "ab") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        after = path.read_bytes()
        if not after.startswith(before) or after[len(before):] != payload:
            raise RuntimeError("append 后字节与预期不符；停")
    return True


def _apply(out_dir: Path, cands: list[Candidate]) -> dict:
    p = paths_for(out_dir, ARM)
    n_audit = n_marks = 0
    for c in cands:
        if _append_audit(p["technical_exclusions"], c.record):
            n_audit += 1
        man = WorkManifest(p["manifest"])
        cur = man.current_status()
        for uid, reason in c.pending_marks:
            if cur.get(uid) == "pending":
                man.mark(uid, "excluded", reason=reason)
                n_marks += 1
    return {"audit_appended": n_audit, "manifest_events": n_marks}


# ══════════════════════════════════════════════════════════════════════════════
# 恢复后强制门
# ══════════════════════════════════════════════════════════════════════════════
def verify_after(out_dir: Path, frozen_list: Path, baseline: dict) -> dict:
    from recover_attention.evaluation import v3_g3_smoke as g3
    from recover_attention.evaluation import v3_production as prod

    p = paths_for(out_dir, ARM)
    problems: list[str] = []
    # append-only 前缀
    for key in ("technical_exclusions", "manifest"):
        n = baseline["bytes"][key]
        if not n:
            continue                       # 起点为空/不存在，无前缀可比（首次写入）
        b = p[key].read_bytes()
        if hashlib.sha256(b[:n]).hexdigest() != baseline["sha"][key]:
            problems.append(f"{p[key].name} 的前 {n} B 前缀已变")
    for key in ("raw_results", "diagnostics"):
        now = _sha_file(p[key]) if p[key].exists() else None
        if now != baseline["sha"][key]:
            problems.append(f"{p[key].name} 被修改（本工具绝不写它）")

    gate = g3.require_diagnostic_manifest_consistency(
        arm=ARM, diagnostic_path=p["diagnostics"], manifest_path=p["manifest"])

    raw_rows = _read_jsonl(p["raw_results"])
    man = WorkManifest(p["manifest"])
    status = man.current_status()
    n_done = sum(1 for v in status.values() if v == "done")
    if n_done != len(raw_rows):
        problems.append(f"n_done {n_done} != n_rows {len(raw_rows)}")
    # hidden 不变量：**hidden 集合 == raw 中 requires_hidden 为真的集合**；
    # eligible_for_primary=false 没有 hidden 是**合法终态**，不是数据丢失。
    hidden_ids = {f.stem for f in p["hidden_dir"].glob("*.npy")}
    need_hidden = {r["unit_id"] for r in raw_rows if requires_hidden(ARM, r)}
    if hidden_ids != need_hidden:
        problems.append(
            f"hidden 集合与应有集合不符（缺 {sorted(need_hidden - hidden_ids)[:3]}，"
            f"多 {sorted(hidden_ids - need_hidden)[:3]}）")

    frozen = _frozen_index(frozen_list)
    au = prod.load_authorized_exclusions(ARM, out_dir)
    preserved = sum(len(r.get("preserved_raw_unit_ids", [])) for r in au.values()
                    if r["action"] == ACTION_PARTIAL_GROUP)
    prov = prod.mcq_population_provenance(
        nominal_questions=len(frozen), technical_exclusions=len(au),
        k_traces=K, preserved_not_analyzed=preserved)
    counts = {k: sum(1 for v in status.values() if v == k)
              for k in ("done", "excluded", "pending", "running", "failed")}
    total = counts["done"] + counts["pending"] + counts["excluded"]
    if total != len(frozen) * K:
        problems.append(f"done+pending+excluded {total} != {len(frozen)} × {K}")
    if counts["running"] or counts["failed"]:
        problems.append(f"仍有 running/failed：{counts}")
    if problems:
        raise RuntimeError("恢复后一致性核验未通过：" + "；".join(problems))
    return {"gate": gate, "counts": counts, "provenance": prov,
            "n_raw": len(raw_rows), "n_hidden": len(hidden_ids)}


def run_recovery(*, out_dir: Path, frozen_list: Path, apply: bool) -> dict:
    cands, baseline = discover_and_validate(out_dir, frozen_list)
    plan = [{"unit_id": c.unit_id, "action": c.action, "kind": c.kind, "j": c.j,
             "manifest_marks": [u for u, _r in c.pending_marks]} for c in cands]
    converged = baseline["converged_history"]
    if not apply:
        return {"applied": False, "candidates": plan, "converged_history": converged}
    written = _apply(out_dir, cands)
    return {"applied": True, "candidates": plan, "converged_history": converged, **written,
            "verification": verify_after(out_dir, frozen_list, baseline)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 仅 MCQ：形态判定、n_questions × K 对账与 finalizer 授权都是 MCQ 专用。
    ap.add_argument("--arm", choices=[ARM], required=True)
    ap.add_argument("--apply", action="store_true",
                    help="缺省为 dry-run：只读报告，不写任何文件")
    args = ap.parse_args(argv)
    try:
        res = run_recovery(out_dir=DEFAULT_OUT_DIR, frozen_list=DEFAULT_FROZEN_LIST,
                           apply=args.apply)
    except RecoveryRejected as exc:
        print(f"[REJECT] {exc}")
        for r in exc.reasons:
            print(f"  - {r}")
        print("[REJECT] **零写入**；回到人工裁决。")
        return 2
    mode = "APPLY" if res["applied"] else "DRY-RUN"
    print(f"[{mode}] 候选 {len(res['candidates'])} 例 ｜ "
          f"已按授权语义验证通过的历史组 {res['converged_history']} 个（跳过、不重写）")
    for c in res["candidates"]:
        print(f"  {c['unit_id']:<34} {c['action']:<40} kind={c['kind']:<6} "
              f"j={c['j']} marks={len(c['manifest_marks'])}")
    if not res["applied"]:
        print("[DRY-RUN] 未写入任何文件。加 --apply 执行。")
        return 0
    v = res["verification"]
    print(f"[APPLY ] audit +{res['audit_appended']} 条 ｜ manifest +{res['manifest_events']} 事件")
    print(f"[verify] 门 {v['gate']}")
    print(f"[verify] 五态 {v['counts']} ｜ raw {v['n_raw']} ｜ hidden {v['n_hidden']}")
    pv = v["provenance"]
    print(f"[verify] technical_exclusions {pv['technical_exclusions']} ｜ "
          f"retained_raw_units {pv['retained_raw_units']} ｜ "
          f"preserved {pv['preserved_but_not_analyzed_raw_units']} ｜ "
          f"expected_total_raw_rows {pv['expected_total_raw_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
