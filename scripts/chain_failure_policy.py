r"""接续链的**失败策略**与**完成判据**（单独成模块，好在合成故障下验证）。

用户 2026-09-15 的有限授权：**只有** `forward_nonfinite` 允许自动接续，别的一律停止。
每条判断都要有结构化证据 —— 不看退出码、不看日志关键词、不把 PID 消失当正常结束。

七条限定与实现的对应：

    1 三方核对   本次运行**新增**的段日志 stopped_reason == unit_failure:forward_nonfinite
                 且本次**新增恰好一条**诊断（前缀逐条校验，不按长度切片）
                 且 manifest 该 unit 最新状态 == excluded/forward_nonfinite
                 三者指向**同一个 unit_id** 才算数。
    2 只跳过     接续 = 跳过已终结的 unit（manifest 终态本就不重入队）。
                 本模块**从不写 manifest**。并复核已登记的本臂排除 unit
                 **仍然存在且仍是** excluded/forward_nonfinite —— 缺失或 reason 变了都停。
    3 全链上限   计数持久化在 chain_autoresume.json，含 unit_id 列表与逐条证据；
                 **跨臂累计、重启不清零**。同一 unit 重复出现 → 停。
                 无新增终态进展 → 停。累计 5 次 → 停。
    4 其余全停   CUDA OOM、其他异常、指纹冲突、诊断落盘不完整、**退出原因无法确认**
                 （例如被接管的进程 PID 消失却没留下段日志）—— 一律停。
    5 完成判据   绑定**完整计划名单**（冻结名单 × K 规则）：
                 manifest 的 unit 集合 == 计划集合；raw 的**唯一** unit 集合 == done 集合；
                 raw 无重复行；done + excluded == 计划总数且全部终态。
                 excluded 正确计入，但**不让缺失 unit 与重复 raw 相互抵消**。
    6 不扩排除   本模块不写审计、不标授权、不改任何实验设置。
                 排除的**授权**由 apply_mcq_*_exclusion.py 人工执行；未登记时
                 finalize 会自己拒绝 —— 那是设计中的闸，不绕过。
    7 合成故障   见 tests/test_chain_failure_policy.py 与
                 scripts/diagnostics/verify_chain_failure_policy.py。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: 唯一允许自动接续的结构化停止原因。
RESUMABLE_STOPPED_REASON = "unit_failure:forward_nonfinite"
#: 允许自动接续的 manifest 终态与 reason。
RESUMABLE_MANIFEST_STATUS = "excluded"
RESUMABLE_MANIFEST_REASON = "forward_nonfinite"
#: **明确的正常分段结束**。除这三种外，任何 stopped_reason 都不算"正常退出"。
NORMAL_SEGMENT_REASONS = ("completed", "max_new_units", "max_wall_seconds")
#: **整条 H1 → MCQ 链**累计上限（不是每臂）。
MAX_AUTORESUMES_PER_CHAIN = 5


# ══════════════════════════════════════════════════════════════════════════════
# manifest / 诊断 / 段日志的只读视图
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ManifestState:
    total: int
    done: int
    excluded: int
    failed: int
    nonterminal: int
    status: dict[str, str] = field(default_factory=dict)
    reason: dict[str, str | None] = field(default_factory=dict)

    @property
    def all_terminal(self) -> bool:
        return self.total > 0 and self.nonterminal == 0

    @property
    def done_units(self) -> set[str]:
        return {u for u, s in self.status.items() if s == "done"}

    @property
    def excluded_units(self) -> set[str]:
        return {u for u, s in self.status.items() if s == "excluded"}


def manifest_state(path: Path) -> ManifestState:
    """每个 unit 的**最新**事件即其当前状态（append-only 语义）。"""
    status: dict[str, str] = {}
    reason: dict[str, str | None] = {}
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = rec.get("unit_id")
                if uid:
                    status[uid] = rec.get("status")
                    reason[uid] = rec.get("reason")
    done = sum(1 for v in status.values() if v == "done")
    exc = sum(1 for v in status.values() if v == "excluded")
    fail = sum(1 for v in status.values() if v == "failed")
    return ManifestState(total=len(status), done=done, excluded=exc, failed=fail,
                         nonterminal=len(status) - done - exc,
                         status=status, reason=reason)


def diagnostic_unit_ids(path: Path) -> list[str]:
    """诊断文件里的 unit_id（顺序即追加顺序）。缺 unit_id 的行视为落盘不完整。"""
    out: list[str] = []
    if not Path(path).exists():
        return out
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec.get("unit_id")
            if not uid:
                raise ValueError("诊断行缺 unit_id —— 落盘不完整，拒绝自动接续")
            out.append(uid)
    return out


def segment_entries(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in
            Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def raw_unit_ids(path: Path) -> list[str]:
    """raw 结果行的 unit_id，**按出现顺序**（要用它查重复行）。"""
    out: list[str] = []
    if not Path(path).exists():
        return out
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line).get("unit_id"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 限定 4：本次运行的证据边界
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RunBoundary:
    """一次运行**开始之前**的快照。用来把"本次新增"与历史严格分开。"""
    n_segments: int
    diagnostics: list[str]
    mstate: ManifestState


def snapshot(*, manifest: Path, diagnostics: Path, segment_log: Path) -> RunBoundary:
    return RunBoundary(n_segments=len(segment_entries(segment_log)),
                       diagnostics=diagnostic_unit_ids(diagnostics),
                       mstate=manifest_state(manifest))


def new_diagnostics(before: list[str], after: list[str]) -> list[str]:
    """**逐条**核对历史前缀一致，再取新增。不按旧长度切片就当新增。"""
    if len(after) < len(before) or after[:len(before)] != before:
        raise ValueError(
            f"诊断历史前缀被改动（before {len(before)} 条 / after {len(after)} 条）"
            f" —— 证据不可信，拒绝自动接续")
    return after[len(before):]


# ══════════════════════════════════════════════════════════════════════════════
# 限定 5：完成判据绑定完整计划名单
# ══════════════════════════════════════════════════════════════════════════════
def planned_unit_ids(example_ids, k: int) -> set[str]:
    """K = 1 greedy + (k−1) sampled —— 与 `v3_g3_smoke.units_for` 同一条规则。"""
    out: set[str] = set()
    for eid in example_ids:
        out.add(f"{eid}__greedy_0")
        for i in range(1, k):
            out.add(f"{eid}__sampled_{i}")
    return out


def arm_complete(mstate: ManifestState, raw_ids: list[str],
                 planned: set[str]) -> tuple[bool, str]:
    """完成 = 计划名单、manifest 终态、raw 唯一集合三者严格吻合。

    **不拿 raw 行数与计划总数比**（excluded 本来就没有结果行），
    也**不让缺失 unit 与重复 raw 相互抵消** —— 集合比对，不是数量比对。
    """
    if not planned:
        return False, "计划名单为空 —— 无法判定完成"
    manifest_units = set(mstate.status)
    missing = planned - manifest_units
    stray = manifest_units - planned
    if missing or stray:
        return False, (f"manifest 的 unit 集合与计划名单不符："
                       f"缺 {len(missing)} 个 {sorted(missing)[:3]}，"
                       f"多 {len(stray)} 个 {sorted(stray)[:3]}")
    if not mstate.all_terminal:
        return False, f"仍有 {mstate.nonterminal} 个 unit 未进入终态"
    if mstate.done + mstate.excluded != len(planned):
        return False, (f"终态计数不自洽：done {mstate.done} + excluded "
                       f"{mstate.excluded} != 计划 {len(planned)}")
    dup = len(raw_ids) - len(set(raw_ids))
    if dup:
        seen, dups = set(), set()
        for u in raw_ids:
            (dups if u in seen else seen).add(u)
        return False, f"raw 有 {dup} 条重复行，例如 {sorted(dups)[:3]}"
    if set(raw_ids) != mstate.done_units:
        only_raw = set(raw_ids) - mstate.done_units
        only_done = mstate.done_units - set(raw_ids)
        return False, (f"raw 的唯一 unit 集合与 done 集合不符："
                       f"只在 raw {sorted(only_raw)[:3]}，只在 done {sorted(only_done)[:3]}")
    return True, (f"计划 {len(planned)} 个 unit 全部终态；done {mstate.done} / "
                  f"excluded {mstate.excluded}；raw 唯一集合 == done 集合，无重复行")


# ══════════════════════════════════════════════════════════════════════════════
# 限定 1 + 4：一次运行的退出该怎么归类
# ══════════════════════════════════════════════════════════════════════════════
#: 归类结果
OUTCOME_NORMAL = "normal_segment_end"       # 明确的正常分段结束，可进入下一步
OUTCOME_NONFINITE = "forward_nonfinite"     # 走有限授权的接续判据
OUTCOME_OTHER = "other_failure"             # 明确的其他失败 → 停
OUTCOME_UNKNOWN = "unconfirmed_exit"        # 退出原因无法确认 → 停


def classify_exit(*, segments_new: list[dict], returncode: int | None) -> tuple[str, str]:
    """只凭**本次新增的段日志**判断退出原因。

    `returncode=None` 表示"接管的既有进程"（接续器没拿到退出码）——
    此时更不能把 PID 消失当正常结束：没有新段日志就是 `unconfirmed_exit`。
    """
    if len(segments_new) == 0:
        return OUTCOME_UNKNOWN, ("本次运行没有留下任何段日志 —— 进程可能被杀"
                                 "（OOM / 驱动复位 / 外部终止）。PID 消失不等于正常结束")
    if len(segments_new) > 1:
        return OUTCOME_UNKNOWN, f"本次运行留下 {len(segments_new)} 条段日志（应为 1）"
    seg = segments_new[0]
    reason = str(seg.get("stopped_reason", ""))
    n_fail = int(seg.get("n_failures", 0) or 0)
    if reason in NORMAL_SEGMENT_REASONS and n_fail == 0:
        if returncode not in (None, 0):
            return OUTCOME_UNKNOWN, (f"段日志说 {reason!r} 但退出码是 {returncode} —— "
                                     f"两者矛盾，无法确认")
        return OUTCOME_NORMAL, f"明确的正常分段结束：{reason}"
    if reason == RESUMABLE_STOPPED_REASON:
        if returncode == 0:
            return OUTCOME_UNKNOWN, "段日志说 forward_nonfinite 但退出码是 0 —— 矛盾"
        return OUTCOME_NONFINITE, reason
    return OUTCOME_OTHER, f"明确的其他失败：stopped_reason={reason!r} n_failures={n_fail}"


# ══════════════════════════════════════════════════════════════════════════════
# 限定 1–4：能不能自动接续
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class AutoResumeDecision:
    allow: bool
    why: str
    unit_id: str | None = None
    evidence: dict = field(default_factory=dict)


def decide_autoresume(*, segments_new: list[dict], mstate_before: ManifestState,
                      mstate_after: ManifestState, new_diag: list[str],
                      already_resumed: list[str],
                      returncode: int | None = None,
                      max_total: int = MAX_AUTORESUMES_PER_CHAIN) -> AutoResumeDecision:
    """三方证据齐备才放行。任一条不成立 → 停止（返回 allow=False 并说明理由）。

    `new_diag` 必须由 `new_diagnostics()` 产出（它已校验历史前缀未被改动）。
    """
    kind, detail = classify_exit(segments_new=segments_new, returncode=returncode)
    if kind != OUTCOME_NONFINITE:
        return AutoResumeDecision(False, f"退出归类为 {kind}：{detail} —— 停止")

    seg = segments_new[0]
    if int(seg.get("n_failures", 0)) != 1:
        return AutoResumeDecision(
            False, f"本段失败 unit 数 {seg.get('n_failures')} != 1 —— 停止")

    # 限定 4：本段若还有**其他** failed 事件（非终态、会被重新入队），不得放行
    if mstate_after.failed > mstate_before.failed:
        return AutoResumeDecision(
            False, f"本段新增 {mstate_after.failed - mstate_before.failed} 个 failed 事件 —— 停止")

    # 限定 1：本次运行必须**新增恰好一条**诊断
    if len(new_diag) != 1:
        return AutoResumeDecision(
            False, f"本次运行新增诊断 {len(new_diag)} 条（应为 1）—— 诊断落盘不完整，停止",
            evidence={"new_diagnostics": new_diag})
    uid = new_diag[0]

    # 限定 1：manifest 该 unit 的最新状态必须就是 excluded / forward_nonfinite
    st, rs = mstate_after.status.get(uid), mstate_after.reason.get(uid)
    if st != RESUMABLE_MANIFEST_STATUS or rs != RESUMABLE_MANIFEST_REASON:
        return AutoResumeDecision(
            False, f"manifest 中 {uid} 是 status={st!r} reason={rs!r}，"
                   f"不是 excluded/forward_nonfinite —— 停止", unit_id=uid)

    # 限定 3：同一 unit 重复出现 → 停
    if uid in already_resumed:
        return AutoResumeDecision(
            False, f"{uid} 此前已自动接续过一次，重复出现同一事件 —— 停止", unit_id=uid)

    # 限定 3：必须有**新增终态进展**
    if not (mstate_after.excluded > mstate_before.excluded
            or mstate_after.done > mstate_before.done):
        return AutoResumeDecision(
            False, "本段没有新增终态进展（done / excluded 都没涨）—— 停止", unit_id=uid)

    # 限定 3：全链累计上限
    if len(already_resumed) >= max_total:
        return AutoResumeDecision(
            False, f"全链自动接续已达上限 {max_total} 次 —— 停止", unit_id=uid)

    return AutoResumeDecision(
        True,
        f"三方证据一致（本次新增的段日志 / 诊断 / manifest 均指向 {uid}）；"
        f"该 unit 已是 excluded 终态，接续只会跳过它，不重试",
        unit_id=uid,
        evidence={"stopped_reason": seg.get("stopped_reason"),
                  "new_diagnostic_unit": uid,
                  "manifest_status": st, "manifest_reason": rs,
                  "excluded_before_after": [mstate_before.excluded, mstate_after.excluded],
                  "done_before_after": [mstate_before.done, mstate_after.done],
                  "failed_before_after": [mstate_before.failed, mstate_after.failed],
                  "autoresumes_used": len(already_resumed), "max_total": max_total})


# ══════════════════════════════════════════════════════════════════════════════
# 限定 2 + 3：持久化计数 / 接续后的复核
# ══════════════════════════════════════════════════════════════════════════════
def load_counter(path: Path) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"schema": "stage1_chain_autoresume_v1", "units": [], "events": []}


def record_autoresume(path: Path, counter: dict, *, arm: str, unit_id: str,
                      when: str, evidence: dict) -> dict:
    """append-only：只加不减。**不写 manifest、不动 excluded 状态。**"""
    counter["units"] = counter.get("units", []) + [unit_id]
    counter["events"] = counter.get("events", []) + [
        {"t": when, "arm": arm, "unit_id": unit_id, "evidence": evidence}]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(counter, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return counter


def units_for_arm(counter: dict, arm: str) -> list[str]:
    """计数是**全链累计**（H1 + MCQ）；核对某臂的 manifest 时只取属于该臂的。"""
    return [e["unit_id"] for e in counter.get("events", []) if e.get("arm") == arm]


def verify_still_excluded(mstate: ManifestState, unit_ids: list[str]) -> tuple[bool, str]:
    """限定 2 + 3：本臂**已登记**的排除 unit 必须**仍然存在**且仍是
    excluded/forward_nonfinite。缺失或 reason 变了 —— 都停。

    调用方已用 `units_for_arm` 筛过所属臂，所以"查不到"就是真问题，不是跨臂噪声。
    """
    missing = [u for u in unit_ids if u not in mstate.status]
    if missing:
        return False, (f"已登记的排除 unit 在 manifest 里消失了：{missing[:5]}"
                       f"（共 {len(missing)} 个）—— 停止")
    bad = [(u, mstate.status[u], mstate.reason.get(u)) for u in unit_ids
           if mstate.status[u] != RESUMABLE_MANIFEST_STATUS
           or mstate.reason.get(u) != RESUMABLE_MANIFEST_REASON]
    if bad:
        return False, (f"已登记的排除 unit 状态被改动：{bad[:3]} —— "
                       f"停止（不得清除 excluded，也不得改 reason）")
    return True, f"{len(unit_ids)} 个已登记排除 unit 仍存在且仍是 excluded/forward_nonfinite"
