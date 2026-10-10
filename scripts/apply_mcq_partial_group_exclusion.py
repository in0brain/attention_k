"""泛化的整组技术性排除（sampled_j 失败）：该题整组不进 F5 / finalization。

**显式 case 表**，只处理表中具名的 example_id —— **不接受任意 example_id 参数**。

    preserved = greedy_0 + sampled_1..j-1   j 条 done + raw（永久保留，但不被分析）
    trigger   = sampled_j                   excluded / forward_nonfinite
    suffix    = sampled_{j+1}..sampled_5    excluded / group_excluded_due_sampled_forward_nonfinite

两个 reason 字符串**层级不同，不可混用**：审计记录的 `exclusion_reason` 是组级分类
（`partial_group_...`），`post_trigger_exclusion_reason` 是 suffix 的**逐 unit manifest
事件 reason**（`group_excluded_due_sampled_...`），后者显式写进审计记录，finalizer 从
记录读取而非取代码常量 —— 否则日后改常量会把既有审计记录重新解释一遍。

`cybermetric_000278`（j=5）是**历史旧式记录**：字段形状为 `excluded_unit_ids`、无
post-trigger unit。本脚本对它**只校验、绝不重写**，其字节保持逐字节不变。

本脚本不加载模型、不生成任何 unit、不碰 raw/hidden。写入采用二进制 append
（`json.dumps(..., allow_nan=False)` + UTF-8 + CRLF，flush + os.fsync），写后回读校验。
幂等且可崩溃恢复：审计记录已存在则逐字段比对（不符即停）；manifest 只补仍为 pending
的 suffix unit。

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
    ACTION_PARTIAL_GROUP, AUTHORIZED_POST_TRIGGER_REASONS,
    DIAGNOSTIC_RECORD_SHA_CALIBER, diagnostic_record_index, paths_for)
from recover_attention.generation.manifest import WorkManifest  # noqa: E402

OUT_DIR = ROOT / "outputs/stage1/production"
FROZEN_LIST = ROOT / "outputs/stage1/samples/mcq_primary_questions.jsonl"

ARM = "mcq"
K = 6
TRIGGER_REASON = "forward_nonfinite"
EXCLUSION_REASON = "partial_group_excluded_due_sampled_forward_nonfinite"
POST_TRIGGER_REASON = "group_excluded_due_sampled_forward_nonfinite"

#: **显式 case 表** —— 每条都由研究者单独裁决后具名加入，不开放通配。
#:   shape="legacy"      旧式 `excluded_unit_ids` 记录，只校验不重写（仅 j=K-1 合法）
#:   shape="generalized" 新式 `trigger_excluded_unit_id` + post_trigger_* 三字段
CASES: list[dict] = [
    {"example_id": "cybermetric_000278", "j": 5, "shape": "legacy"},
    {"example_id": "cybermetric_005478", "j": 1, "shape": "generalized"},
    {"example_id": "cybermetric_005228", "j": 1, "shape": "generalized"},
    {"example_id": "cybermetric_003232", "j": 2, "shape": "generalized"},
    {"example_id": "cybermetric_002391", "j": 1, "shape": "generalized"},
]

GENERALIZED_KEYS = {
    "schema_version", "arm", "example_id", "question_hash", "group_id", "action",
    "trigger_unit_id", "trigger_sample_type", "trigger_sample_index", "trigger_reason",
    "manifest_pattern", "preserved_raw_unit_ids", "trigger_excluded_unit_id",
    "post_trigger_excluded_unit_ids", "post_trigger_exclusion_reason",
    "raw_preserved_not_analyzed", "exclusion_reason",
    "diagnostic_unit_id", "diagnostic_record_sha256", "diagnostic_record_sha256_caliber",
    "diagnostics_prefix_bytes_at_recovery", "diagnostics_prefix_sha256_at_recovery",
    "diagnostic_persisted", "frozen_list_sha256", "run_fingerprint_sha256",
    "classification", "protocol_deviation", "preregistration_files_modified",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"[STOP] {msg}")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for chunk in path.read_bytes().split(b"\n"):
        line = chunk[:-1] if chunk.endswith(b"\r") else chunk
        if line.strip():
            out.append(json.loads(line.decode("utf-8")))
    return out


def split_units(eid: str, j: int) -> tuple[list[str], str, list[str]]:
    units = [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]
    return units[:j], units[j], units[j + 1:]


def question_hash_of(eid: str) -> str:
    for line in FROZEN_LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if f"cybermetric_{int(row['original_index']):06d}" == eid:
            return row["question_hash"]
    fail(f"冻结名单中找不到 {eid}")
    raise AssertionError                                   # unreachable


# ══════════════════════════════════════════════════════════════════════════════
# 前置校验：只**核对**，绝不修改
# ══════════════════════════════════════════════════════════════════════════════
def verify_preconditions(case: dict, p: dict, man: WorkManifest) -> None:
    eid, j = case["example_id"], case["j"]
    if not 1 <= j <= K - 1:
        fail(f"{eid}: j={j} 超出 sampled_j 通则允许的 1..{K - 1}")
    if case["shape"] == "legacy" and j != K - 1:
        fail(f"{eid}: 旧式记录只在 j={K - 1} 时合法，实为 j={j}")
    preserved, trigger, suffix = split_units(eid, j)
    status = man.current_status()
    events = {ev["unit_id"]: ev for ev in man._events}      # noqa: SLF001
    for uid in preserved:
        if status.get(uid) != "done":
            fail(f"{uid} 状态 {status.get(uid)!r} 不是 done —— preserved prefix 必须全部 done")
    if status.get(trigger) != "excluded":
        fail(f"{trigger} 状态 {status.get(trigger)!r} 不是 excluded")
    if events[trigger].get("reason") != TRIGGER_REASON:
        fail(f"{trigger} 的 reason={events[trigger].get('reason')!r}，期望 {TRIGGER_REASON!r}")
    for uid in suffix:
        st = status.get(uid)
        if st not in ("pending", "excluded"):
            fail(f"{uid} 当前状态 {st!r} 既非 pending 也非 excluded —— 拒绝改写")
        if st == "excluded" and events[uid].get("reason") != POST_TRIGGER_REASON:
            fail(f"{uid} 已 excluded 但 reason={events[uid].get('reason')!r}，"
                 f"期望 {POST_TRIGGER_REASON!r}")
    raw_units = {r["unit_id"] for r in read_records(p["raw_results"])
                 if r["example_id"] == eid}
    if raw_units != set(preserved):
        fail(f"{eid}: raw 集合与 preserved prefix 不符"
             f"（缺 {sorted(set(preserved) - raw_units)[:3]}，"
             f"多 {sorted(raw_units - set(preserved))[:3]}）")
    idx = diagnostic_record_index(p["diagnostics"])
    if trigger not in idx:
        fail(f"{eid}: diagnostics 中找不到 {trigger} 的记录 —— 不得在无诊断的情况下授权")
    # question_hash / group_id 的**真源是冻结名单**，不取自 diagnostic；
    # diagnostic 里的 group_id 只用来交叉核对，不一致即停。
    qhash = question_hash_of(eid)
    gid = f"cybermetric_question_{qhash[:16]}"
    diag = next(r for r in read_records(p["diagnostics"]) if r["unit_id"] == trigger)
    if diag.get("group_id") != gid:
        fail(f"{eid}: diagnostic 的 group_id={diag.get('group_id')!r} 与冻结名单派生值 {gid!r} 不符")
    if diag.get("example_id") != eid:
        fail(f"{eid}: diagnostic 的 example_id={diag.get('example_id')!r} 不符")
    print(f"[pre   ] {eid} j={j} 前置校验通过："
          f"{len(preserved)} done+raw / 1 excluded / {len(suffix)} post（**未做任何修改**）")
    print(f"[pre   ]   冻结名单 question_hash {qhash[:16]}… ｜ group_id {gid}"
          f" ｜ 与 diagnostic 交叉核对一致")


# ══════════════════════════════════════════════════════════════════════════════
# 审计真源
# ══════════════════════════════════════════════════════════════════════════════
def build_generalized_record(case: dict, p: dict) -> dict:
    eid, j = case["example_id"], case["j"]
    preserved, trigger, suffix = split_units(eid, j)
    qhash = question_hash_of(eid)
    rec_sha = diagnostic_record_index(p["diagnostics"])[trigger]
    # `*_at_recovery` 是**时点快照**：diagnostics 为 append-only，日后追加会让当前 prefix
    # 变化。复跑时沿用已落盘记录里的历史值，否则幂等比对会误判不符。
    prior = next((r for r in read_records(p["technical_exclusions"])
                  if r.get("example_id") == eid), None)
    if prior is not None:
        prefix_bytes = prior["diagnostics_prefix_bytes_at_recovery"]
        prefix_sha = prior["diagnostics_prefix_sha256_at_recovery"]
    else:
        blob = p["diagnostics"].read_bytes()
        prefix_bytes, prefix_sha = len(blob), hashlib.sha256(blob).hexdigest()
    return {
        "schema_version": "stage1_technical_exclusion_v1",
        "arm": ARM,
        "example_id": eid,
        "question_hash": qhash,
        "group_id": f"cybermetric_question_{qhash[:16]}",
        "action": ACTION_PARTIAL_GROUP,
        "trigger_unit_id": trigger,
        "trigger_sample_type": "sampled",
        "trigger_sample_index": j,
        "trigger_reason": TRIGGER_REASON,
        "manifest_pattern": f"{j}_done_1_excluded_{len(suffix)}_post_excluded",
        "preserved_raw_unit_ids": preserved,
        "trigger_excluded_unit_id": trigger,               # **标量字符串**，不是列表
        "post_trigger_excluded_unit_ids": suffix,
        "post_trigger_exclusion_reason": POST_TRIGGER_REASON,
        "raw_preserved_not_analyzed": True,
        "exclusion_reason": EXCLUSION_REASON,
        "diagnostic_unit_id": trigger,
        "diagnostic_record_sha256": rec_sha,
        "diagnostic_record_sha256_caliber": DIAGNOSTIC_RECORD_SHA_CALIBER,
        "diagnostics_prefix_bytes_at_recovery": prefix_bytes,
        "diagnostics_prefix_sha256_at_recovery": prefix_sha,
        "diagnostic_persisted": True,
        "frozen_list_sha256": sha256_file(FROZEN_LIST),
        "run_fingerprint_sha256": sha256_file(p["fingerprint"]),
        "classification": "runtime_technical_attrition",
        "protocol_deviation": True,
        "preregistration_files_modified": False,
    }


def verify_legacy_record(case: dict, p: dict) -> None:
    """旧式记录：**只校验、绝不重写**，其字节逐字节不变。"""
    eid, j = case["example_id"], case["j"]
    preserved, trigger, suffix = split_units(eid, j)
    got = [r for r in read_records(p["technical_exclusions"]) if r.get("example_id") == eid]
    if len(got) != 1:
        fail(f"{eid}: 审计真源中出现 {len(got)} 次，应恰好 1 次")
    rec = got[0]
    checks = {
        "action": (rec.get("action"), ACTION_PARTIAL_GROUP),
        "trigger_unit_id": (rec.get("trigger_unit_id"), trigger),
        "excluded_unit_ids": (rec.get("excluded_unit_ids"), [trigger]),
        "preserved_raw_unit_ids": (rec.get("preserved_raw_unit_ids"), preserved),
        "exclusion_reason": (rec.get("exclusion_reason"), EXCLUSION_REASON),
        "trigger_reason": (rec.get("trigger_reason"), TRIGGER_REASON),
        "diagnostic_unit_id": (rec.get("diagnostic_unit_id"), trigger),
        "diagnostic_record_sha256": (rec.get("diagnostic_record_sha256"),
                                     diagnostic_record_index(p["diagnostics"])[trigger]),
    }
    for field, (a, b) in checks.items():
        if a != b:
            fail(f"{eid}: 旧式记录字段 {field} 为 {a!r}，期望 {b!r}")
    if suffix:
        fail(f"{eid}: 旧式记录不应有 post-trigger unit，但 j={j} 推出 {len(suffix)} 个")
    for field in ("trigger_excluded_unit_id", "post_trigger_excluded_unit_ids",
                  "post_trigger_exclusion_reason"):
        if field in rec:
            fail(f"{eid}: 旧式记录不应含新式字段 {field}")
    print(f"[audit ] {eid} 旧式记录校验通过 —— **未重写**（保持逐字节不变）")


def append_record(path: Path, record: dict) -> bool:
    """二进制 append 一条严格 JSON 单行（UTF-8 + CRLF）。返回是否真的写入。"""
    eid = record["example_id"]
    missing, extra = GENERALIZED_KEYS - set(record), set(record) - GENERALIZED_KEYS
    if missing or extra:
        fail(f"{eid}: 记录字段不符 missing={sorted(missing)} extra={sorted(extra)}")
    if not isinstance(record["trigger_excluded_unit_id"], str):
        fail(f"{eid}: trigger_excluded_unit_id 必须是标量字符串")
    if record["post_trigger_exclusion_reason"] not in AUTHORIZED_POST_TRIGGER_REASONS:
        fail(f"{eid}: post_trigger_exclusion_reason 不在白名单 {AUTHORIZED_POST_TRIGGER_REASONS}")
    existing = read_records(path)
    same = [r for r in existing if r.get("example_id") == eid]
    if same:
        if len(same) > 1:
            fail(f"{eid} 在审计真源中出现 {len(same)} 次 —— 停")
        if same[0] != record:
            diff = sorted(k for k in set(record) | set(same[0])
                          if record.get(k) != same[0].get(k))
            fail(f"{eid} 的既有记录与预期不符，差异字段 {diff} —— 不覆盖、不重写，人工介入")
        print(f"[audit ] {eid} 记录已存在且逐字段一致 —— no-op")
        return False
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         allow_nan=False).encode("utf-8") + b"\r\n"
    before = path.read_bytes() if path.exists() else b""
    with open(path, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    after = path.read_bytes()
    if not after.startswith(before):
        fail("append 后旧前缀发生变化 —— 停")
    if after[len(before):] != payload:
        fail("append 写入的字节与预期不符 —— 停")
    print(f"[audit ] {eid} 已二进制 append 一条（{len(payload)} B，UTF-8 + CRLF）")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# manifest：只补仍为 pending 的 suffix unit
# ══════════════════════════════════════════════════════════════════════════════
def ensure_manifest(man_path: Path, case: dict) -> int:
    eid, j = case["example_id"], case["j"]
    _preserved, _trigger, suffix = split_units(eid, j)
    man = WorkManifest(man_path)
    status = man.current_status()
    events = {ev["unit_id"]: ev for ev in man._events}      # noqa: SLF001
    appended = 0
    for uid in suffix:
        st = status.get(uid)
        if st == "excluded":
            if events[uid].get("reason") != POST_TRIGGER_REASON:
                fail(f"{uid} 已是 excluded 但 reason 不符 —— 停")
            continue                                       # 幂等：已正确排除，跳过
        if st != "pending":
            fail(f"{uid} 当前状态 {st!r} 不是 pending —— 拒绝改写非 pending 状态")
        man.mark(uid, "excluded", reason=POST_TRIGGER_REASON)
        appended += 1
    return appended


def verify_converged(man_path: Path, case: dict) -> None:
    eid, j = case["example_id"], case["j"]
    preserved, trigger, suffix = split_units(eid, j)
    man = WorkManifest(man_path)
    status = man.current_status()
    for uid in preserved:
        if status.get(uid) != "done":
            fail(f"收敛校验失败：{uid} 状态为 {status.get(uid)!r}，应为 done")
    if status.get(trigger) != "excluded":
        fail(f"收敛校验失败：{trigger} 状态为 {status.get(trigger)!r}")
    for uid in suffix:
        if status.get(uid) != "excluded":
            fail(f"收敛校验失败：{uid} 状态为 {status.get(uid)!r}")
        hits = [ev for ev in man._events                   # noqa: SLF001
                if ev["unit_id"] == uid and ev.get("status") == "excluded"]
        if len(hits) != 1:
            fail(f"收敛校验失败：{uid} 有 {len(hits)} 条 excluded 事件，应恰好 1 条")
        if hits[0].get("reason") != POST_TRIGGER_REASON:
            fail(f"收敛校验失败：{uid} 的 reason 为 {hits[0].get('reason')!r}")


def main() -> int:
    p = paths_for(OUT_DIR, ARM)
    total = 0
    for case in CASES:
        man = WorkManifest(p["manifest"])
        verify_preconditions(case, p, man)
        if case["shape"] == "legacy":
            verify_legacy_record(case, p)
        else:
            append_record(p["technical_exclusions"], build_generalized_record(case, p))
        n = ensure_manifest(p["manifest"], case)
        total += n
        verify_converged(p["manifest"], case)
        print(f"[case  ] {case['example_id']} 新增 {n} 条 manifest 事件；整组形态已收敛 ✓")

    # ── 回读校验 ────────────────────────────────────────────────────────────
    recs = read_records(p["technical_exclusions"])
    ids = [r["example_id"] for r in recs]
    if len(ids) != len(set(ids)):
        fail(f"审计真源存在重复 example_id：{sorted({x for x in ids if ids.count(x) > 1})}")
    idx = diagnostic_record_index(p["diagnostics"])
    for r in recs:
        uid = r.get("diagnostic_unit_id")
        if r.get("diagnostic_persisted") and idx.get(uid) != r.get("diagnostic_record_sha256"):
            fail(f"{r['example_id']} 的 diagnostic_record_sha256 与 diagnostics 现状不符")
    man = WorkManifest(p["manifest"])
    status = man.current_status()
    counts = {k: sum(1 for v in status.values() if v == k)
              for k in ("pending", "running", "done", "failed", "excluded")}
    print(f"\n[verify] 本次共新增 {total} 条 manifest 事件 ｜ 五态 {counts}")
    print(f"[verify] 审计真源 {len(recs)} 条 ｜ sha256 {sha256_file(p['technical_exclusions'])}")
    for i, r in enumerate(recs, 1):
        if r["action"] != ACTION_PARTIAL_GROUP:
            shape = "full_group"                           # 形态 A，不适用 legacy/generalized
        else:
            shape = "legacy" if "excluded_unit_ids" in r else "generalized"
        print(f"[verify]   #{i} {r['example_id']} action={r['action']} shape={shape} "
              f"j={r.get('trigger_sample_index', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
