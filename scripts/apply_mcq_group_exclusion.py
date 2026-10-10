"""形态 A 的整组技术性排除：greedy 失败 → 该 question 整组不进 finalized population。

**显式 case 表**，只处理表中具名的 example_id —— **不接受任意 example_id 参数**。
每个 case 只做两件事，都不加载模型、不生成任何 unit：

    1. 向 append-only 审计真源 `mcq_technical_exclusions.jsonl` 追加/校验该组记录
    2. 依审计真源，把该 question 仍为 pending 的 unit 追加标记为 excluded

**幂等且可崩溃恢复**：审计记录已存在则逐字段比对（不符即停）；manifest 依当前 latest 状态
只补缺失的事件。每个 unit 最多留下一条 excluded 事件；已有的事件与前缀逐字节不变。

这**不是**预注册允许的排除，也**不是** criterion correction：
是运行期技术减员（runtime technical attrition），属 protocol deviation。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("D:/Projects/recover_attention_project")
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation.v3_production import (  # noqa: E402
    ACTION_FULL_GROUP, DIAGNOSTIC_RECORD_SHA_CALIBER, diagnostic_record_index, paths_for)
from recover_attention.generation.manifest import WorkManifest, write_atomic  # noqa: E402

OUT_DIR = ROOT / "outputs/stage1/production"
FROZEN_LIST = ROOT / "outputs/stage1/samples/mcq_primary_questions.jsonl"

ARM = "mcq"
TRIGGER_REASON = "forward_nonfinite"
EXCLUSION_REASON = "group_excluded_due_greedy_forward_nonfinite"

#: **显式 case 表** —— 每条都由研究者单独裁决后具名加入，不开放通配。
CASES: list[dict] = [
    {
        "example_id": "cybermetric_005396",
        "question_hash": "0ae20d5773c83547822dd6b0822908be5995c1db65c80313b0fdf69ddf757b9e",
        "group_id": "cybermetric_question_0ae20d5773c83547",
        "diagnostic_persisted": False,
        "diagnostic_persistence_note": (
            "本次触发发生在诊断持久化补丁之前，numeric diagnostic 只存在于内存中的 "
            "run.failures，未落盘；成因无法从产物追溯，不作猜测。"),
    },
    {
        "example_id": "cybermetric_003809",
        "question_hash": "98e40db830741d061f33ada4f653e8c7f05f0ecbb0762d5d4dfef663c77721e6",
        "group_id": "cybermetric_question_98e40db830741d06",
        "diagnostic_persisted": True,
        "diagnostic_persistence_note": (
            "诊断持久化补丁生效后触发，sanitized diagnostic 已 append 到 "
            "mcq_forward_nonfinite_diagnostics.jsonl；按 record-level SHA 授权。"),
    },
]

BASE_KEYS = {
    "schema_version", "arm", "example_id", "question_hash", "group_id",
    "trigger_unit_id", "trigger_reason", "affected_unit_ids", "prior_manifest_status",
    "action", "exclusion_reason", "frozen_list_sha256", "run_fingerprint_sha256",
    "classification", "protocol_deviation", "preregistration_files_modified",
    "diagnostic_persisted", "diagnostic_persistence_note",
}
DIAG_KEYS = {"diagnostic_unit_id", "diagnostic_record_sha256",
             "diagnostic_record_sha256_caliber",
             "diagnostics_prefix_bytes_at_recovery",
             "diagnostics_prefix_sha256_at_recovery"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"[STOP] {msg}")


def affected(eid: str) -> list[str]:
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, 6)]


def build_record(case: dict, p: dict) -> dict:
    eid = case["example_id"]
    trig = f"{eid}__greedy_0"
    rec = {
        "schema_version": "stage1_technical_exclusion_v1",
        "arm": ARM,
        "example_id": eid,
        "question_hash": case["question_hash"],
        "group_id": case["group_id"],
        "trigger_unit_id": trig,
        "trigger_reason": TRIGGER_REASON,
        "affected_unit_ids": affected(eid),
        "prior_manifest_status": {trig: "excluded",
                                  **{u: "pending" for u in affected(eid)[1:]}},
        "action": ACTION_FULL_GROUP,
        "exclusion_reason": EXCLUSION_REASON,
        "frozen_list_sha256": sha256_file(FROZEN_LIST),
        "run_fingerprint_sha256": sha256_file(p["fingerprint"]),
        "classification": "runtime_technical_attrition",
        "protocol_deviation": True,
        "preregistration_files_modified": False,
        "diagnostic_persisted": bool(case["diagnostic_persisted"]),
        "diagnostic_persistence_note": case["diagnostic_persistence_note"],
    }
    if case["diagnostic_persisted"]:
        idx = diagnostic_record_index(p["diagnostics"])
        rec_sha = idx.get(trig)
        if rec_sha is None:
            fail(f"{eid}: diagnostics 中找不到 {trig} 的记录 —— 不得在无诊断的情况下声明 persisted")
        # `*_at_recovery` 是**时点快照**：diagnostics 为 append-only，日后追加会让当前 prefix
        # 变化。复跑时沿用已落盘记录里的历史值，否则幂等比对会误判不符。
        prior = next((r for r in _read_records(p["technical_exclusions"])
                      if r.get("example_id") == eid), None)
        if prior is not None:
            prefix_bytes = prior["diagnostics_prefix_bytes_at_recovery"]
            prefix_sha = prior["diagnostics_prefix_sha256_at_recovery"]
        else:
            blob = p["diagnostics"].read_bytes()
            prefix_bytes, prefix_sha = len(blob), hashlib.sha256(blob).hexdigest()
        rec.update({
            "diagnostic_unit_id": trig,
            "diagnostic_record_sha256": rec_sha,
            "diagnostic_record_sha256_caliber": DIAGNOSTIC_RECORD_SHA_CALIBER,
            "diagnostics_prefix_bytes_at_recovery": prefix_bytes,
            "diagnostics_prefix_sha256_at_recovery": prefix_sha,
        })
    return rec


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for chunk in path.read_bytes().split(b"\n"):
        line = chunk[:-1] if chunk.endswith(b"\r") else chunk
        if line.strip():
            out.append(json.loads(line.decode("utf-8")))
    return out


def ensure_audit(path: Path, record: dict) -> bool:
    """不存在则写；已存在则逐字段比对。返回是否真的写入。

    首条用 write_atomic 创建；后续条用**二进制 append**（UTF-8 + CRLF，flush + fsync），
    这样旧记录的原始字节前缀逐字节不变。
    """
    want = BASE_KEYS | (DIAG_KEYS if record["diagnostic_persisted"] else set())
    existing = _read_records(path)
    same = [r for r in existing if r.get("example_id") == record["example_id"]]
    if same:
        if len(same) > 1:
            fail(f"{record['example_id']} 在审计真源中出现 {len(same)} 次 —— 停")
        got = same[0]
        missing, extra = want - set(got), set(got) - want
        if missing or extra:
            fail(f"审计记录字段不符 missing={sorted(missing)} extra={sorted(extra)}")
        if got != record:
            diff = sorted(k for k in want if got.get(k) != record.get(k))
            fail(f"{record['example_id']} 的既有记录与预期不符，差异字段 {diff} —— 不覆盖、不重写")
        print(f"[audit ] {record['example_id']} 记录已存在且逐字段一致 —— no-op")
        return False
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if not existing:
        write_atomic(path, payload + "\n")
        print(f"[audit ] 已原子创建 {path.name}（首条）")
    else:
        before = path.read_bytes()
        with open(path, "ab") as handle:
            handle.write(payload.encode("utf-8") + b"\r\n")
            handle.flush()
            os.fsync(handle.fileno())
        after = path.read_bytes()
        if not after.startswith(before):
            fail("append 后旧前缀发生变化 —— 停")
        print(f"[audit ] 已二进制 append 一条（{record['example_id']}）")
    got = [r for r in _read_records(path) if r["example_id"] == record["example_id"]]
    if len(got) != 1 or got[0] != record:
        fail("回读记录与内存记录不等价 —— 停")
    print(f"[audit ] {record['example_id']} 回读逐字段比对通过 ✓")
    return True


def ensure_manifest(man_path: Path, record: dict) -> int:
    """依审计真源补齐 manifest：只追加仍为 pending 的目标 unit。"""
    man = WorkManifest(man_path)
    status = man.current_status()
    events = {ev["unit_id"]: ev for ev in man._events}          # noqa: SLF001
    appended = 0
    for uid in record["affected_unit_ids"]:
        want = (record["trigger_reason"] if uid == record["trigger_unit_id"]
                else record["exclusion_reason"])
        st = status.get(uid)
        if st == "excluded":
            got = events[uid].get("reason")
            if got != want:
                fail(f"{uid} 已是 excluded 但 reason={got!r}，期望 {want!r} —— 停")
            continue                          # 幂等：已正确排除，跳过
        if uid == record["trigger_unit_id"]:
            fail(f"{uid} 是 trigger unit，应已由生成过程标为 excluded，实为 {st!r} —— 停")
        if st != "pending":
            fail(f"{uid} 当前状态 {st!r} 不是 pending —— 拒绝改写非 pending 状态")
        man.mark(uid, "excluded", reason=want)
        appended += 1
    return appended


def verify_converged(man_path: Path, record: dict) -> None:
    man = WorkManifest(man_path)
    status = man.current_status()
    n_recovery = 0
    for uid in record["affected_unit_ids"]:
        if status.get(uid) != "excluded":
            fail(f"收敛校验失败：{uid} 状态为 {status.get(uid)!r}")
        hits = [ev for ev in man._events                        # noqa: SLF001
                if ev["unit_id"] == uid and ev.get("status") == "excluded"]
        if len(hits) != 1:
            fail(f"收敛校验失败：{uid} 有 {len(hits)} 条 excluded 事件，应恰好 1 条")
        if hits[0].get("reason") == record["exclusion_reason"]:
            n_recovery += 1
    if n_recovery != 5:
        fail(f"收敛校验失败：{record['example_id']} 的 recovery 事件应恰好 5 条，实为 {n_recovery}")


def main() -> int:
    p = paths_for(OUT_DIR, ARM)
    total = 0
    for case in CASES:
        rec = build_record(case, p)
        rows = [r for r in _read_records(p["raw_results"])
                if r["example_id"] == case["example_id"]]
        if rows:
            fail(f"{case['example_id']} 为形态 A，但 raw 中有 {len(rows)} 条 —— 停")
        ensure_audit(p["technical_exclusions"], rec)
        n = ensure_manifest(p["manifest"], rec)
        total += n
        verify_converged(p["manifest"], rec)
        print(f"[case  ] {case['example_id']} 新增 {n} 条 recovery 事件；六个 unit 全 excluded ✓")

    man = WorkManifest(p["manifest"])
    status = man.current_status()
    counts = {k: sum(1 for v in status.values() if v == k)
              for k in ("pending", "running", "done", "failed", "excluded")}
    print(f"[verify] 本次共新增 {total} 条事件 ｜ manifest 五态 {counts}")
    print(f"[audit ] {len(_read_records(p['technical_exclusions']))} 条 ｜ "
          f"sha256 {sha256_file(p['technical_exclusions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
