r"""Llama 跨模型复核的**生成 + finalize 验收**（只读，不改任何产物）。

范围边界（重要）：本脚本**只做验收**。它不跑任何跨模型性能评估、不算 AUROC、
不导入 evaluation 的分析入口。当前授权覆盖的是生成与 finalize；性能评估方案
未经批准前不执行，只交付"验收结果 + 分析就绪情况"。

逐项核对：
    1  计划名单        manifest 终态 vs 计划 unit 名单（集合比对，不比行数）
    2  K 完整性        每个 group 恰好 K 条；不足/超出都列出来
    3  重复与缺失      raw 的 unit_id 唯一性；manifest done 与 raw 行的双向差集
    4  排除原因        excluded 全部有结构化原因，且每条都在 manifest 有登记
    5  hidden 对齐     eligible 记录必须有 hidden；hidden 不得有孤儿；维度/dtype/有限性
    6  实际样本损失    计划 → 完成 → eligible → 可分析，逐级列出损失与占比
    7  protocol_deviation  预算登记、**解析器版本**、标记一并列出，不省略
                           解析口径按实际沿革写，不照抄预算登记里的"未改"
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs/stage1_llama_v3/production"
REG = ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
REPORT = ROOT / "outputs/stage1_llama_v3/acceptance_report.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def check_arm(arm: str, checks: list) -> dict:
    import chain_failure_policy as pol
    from recover_attention.evaluation import v3_production as prod

    def ck(name, ok, detail=""):
        checks.append({"arm": arm, "check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)
        return ok

    p = prod.paths_for(OUT, arm)
    raw = _jsonl(p["raw_results"])
    mstate = pol.manifest_state(p["manifest"])
    out: dict = {"arm": arm}

    # ── 1 计划名单 ─────────────────────────────────────────────────────────
    from run_llama_production_chain import _planned
    planned = set(_planned(arm))
    # ManifestState 是 dataclass，不是 dict —— 状态在 .status 里
    done = mstate.done_units
    excluded = {u for u, st in mstate.status.items() if st not in ("done", None)
                and st != "pending" and st != "running"}
    terminal = done | excluded
    ck("manifest 终态集合 == 计划名单",
       terminal == planned,
       f"planned={len(planned)} terminal={len(terminal)} "
       f"缺={len(planned - terminal)} 多={len(terminal - planned)}")
    out.update(n_planned=len(planned), n_done=len(done), n_excluded=len(excluded))

    # ── 2 K 完整性 ─────────────────────────────────────────────────────────
    by_group = collections.Counter(r["group_id"] for r in raw)
    ks = collections.Counter(by_group.values())
    kmax = max(ks) if ks else 0
    bad_groups = {g: n for g, n in by_group.items() if n != kmax}
    ck(f"每个 group 恰好 K={kmax} 条", not bad_groups,
       f"组数={len(by_group)}  K 分布={dict(ks)}"
       + (f"  异常组 {list(bad_groups.items())[:5]}" if bad_groups else ""))
    out.update(n_groups=len(by_group), k_distribution={str(k): v for k, v in ks.items()},
               groups_with_wrong_k=len(bad_groups))

    # ── 3 重复与缺失 ───────────────────────────────────────────────────────
    ids = [r["unit_id"] for r in raw]
    dup = [u for u, c in collections.Counter(ids).items() if c > 1]
    ck("raw unit_id 唯一", not dup, f"{len(ids)} 行 / {len(set(ids))} 唯一")
    ck("manifest done 与 raw 行双向一致",
       set(ids) == done,
       f"done 无 raw={len(done - set(ids))}  raw 无 done={len(set(ids) - done)}")
    out.update(n_raw_rows=len(ids), n_duplicate_unit_ids=len(dup))

    # ── 4 排除原因 ─────────────────────────────────────────────────────────
    reasons = collections.Counter(
        f"{mstate.status.get(u)}:{mstate.reason.get(u)}" for u in excluded)
    no_reason = [u for u in excluded if not mstate.reason.get(u)]
    ck("全部 excluded 都有结构化原因", not no_reason,
       f"{dict(reasons) if reasons else '无 excluded'}"
       + (f"  缺原因 {no_reason[:5]}" if no_reason else ""))
    out["exclusion_reasons"] = dict(reasons)
    out["excluded_without_reason"] = len(no_reason)

    hd = p["hidden_dir"]
    hfiles = {q.stem for q in hd.iterdir() if q.suffix == ".npy"} if hd.exists() else set()

    # ── 5 hidden 对齐 ──────────────────────────────────────────────────────
    #
    # ⚠️ **两臂的分析单元不同，不能用同一套比对。** 第一版就是在这里栽的：
    #   H1   分析单元 = trace（样本级）。finalized 记录 9264 行，键 trace_id，
    #        取值空间与 raw 的 unit_id 相同
    #   MCQ  分析单元 = **题**。raw 10560 条是 1760 题 × K=6 的样本级记录，
    #        finalized 只有 1760 行，键 example_id —— 拿它去比 unit_id 当然全不匹配
    # 正确做法：按 finalization 报告**自述**的 `hidden_alignment_key` 对齐，
    # 并且比对**分析真正消费的** {arm}_hidden.npz，而不是 {arm}_unit_hidden/ 目录。
    rep_path = OUT / f"{arm}_finalization_report.json"
    frep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    recs = _jsonl(OUT / f"{arm}_completion_records.jsonl")
    npz_path = OUT / f"{arm}_hidden.npz"
    if not recs or not npz_path.exists():
        ck("completion_records 与 {arm}_hidden.npz 已生成（finalize 产物）", False,
           "尚未 finalize")
        out["finalized"] = False
    else:
        key = frep.get("hidden_alignment_key")
        ck("finalization 报告声明了 hidden 对齐键", bool(key), f"key={key}")
        ck("该键在 completion record 里存在", bool(key) and key in recs[0], f"key={key}")

        elig = [r for r in recs
                if bool(r.get("eligible", r.get("eligible_for_primary")))]
        rec_ids = {str(r[key]) for r in recs}
        elig_ids = {str(r[key]) for r in elig}
        ck("completion record 的对齐键唯一", len(rec_ids) == len(recs),
           f"{len(recs)} 行 / {len(rec_ids)} 唯一")

        with np.load(npz_path) as z:
            hkeys = set(z.files)
            sample_keys = sorted(hkeys)[:200]
            dims, dtypes, bad = set(), set(), []
            for k in sample_keys:
                v = z[k]
                dims.add(v.shape); dtypes.add(str(v.dtype))
                if not np.all(np.isfinite(v)) or not np.any(v):
                    bad.append(k)
        ck("hidden.npz 的键集合 == 可分析记录的对齐键集合",
           hkeys == elig_ids,
           f"npz={len(hkeys)}  可分析={len(elig_ids)}  "
           f"npz 多={len(hkeys - elig_ids)}  可分析缺={len(elig_ids - hkeys)}")
        ck("hidden 数量与 finalization 报告自述一致",
           len(hkeys) == frep.get("n_hidden_vectors"),
           f"npz={len(hkeys)}  报告={frep.get('n_hidden_vectors')}")
        ck(f"hidden 抽样 {len(sample_keys)} 个：维度/dtype 一致、有限、非全零",
           not bad and len(dims) <= 1 and len(dtypes) <= 1,
           f"dims={dims} dtypes={dtypes}" + (f" 异常={bad[:3]}" if bad else ""))

        # 每个分析单元的 group 必须在 raw 里；K 完整性上面已按 raw 查过
        rec_groups = {str(r["group_id"]) for r in recs}
        raw_groups = {str(r["group_id"]) for r in raw}
        ck("completion record 的 group 全部出现在 raw 里",
           rec_groups <= raw_groups,
           f"record 组={len(rec_groups)}  raw 组={len(raw_groups)}  "
           f"不在 raw 的={len(rec_groups - raw_groups)}")

        # 样本级 hidden 目录单独查孤儿（H1 的分析输入就是它逐条聚成的 npz）
        orphan = hfiles - set(ids)
        ck("样本级 hidden 目录无孤儿（都对得上 raw unit）", not orphan,
           f"目录文件={len(hfiles)}  孤儿={len(orphan)}")

        out.update(n_records=len(recs), n_eligible=len(elig),
                   hidden_alignment_key=key, n_hidden_in_npz=len(hkeys),
                   n_unit_hidden_files=len(hfiles), n_hidden_orphan=len(orphan),
                   hidden_sample={"n": len(sample_keys),
                                  "dims": [list(d) for d in dims],
                                  "dtypes": sorted(dtypes), "bad": bad[:5]})

    # ── 6 实际样本损失（逐级）─────────────────────────────────────────────
    ladder = [("计划 unit（样本级）", len(planned)), ("manifest done", len(done)),
              ("raw 行（样本级）", len(ids))]
    if recs:
        # MCQ 的 finalized 是**题级**，从 10560 变 1760 是 K=6 聚合，**不是样本丢失**
        ladder += [(f"finalized 记录（{'样本级' if len(recs) == len(ids) else '题级聚合'}）",
                    len(recs)),
                   ("可分析（eligible）", out.get("n_eligible", 0))]
    print("  样本损失逐级：")
    prev = ladder[0][1]
    loss = []
    for name, n in ladder:
        d = n - prev
        print(f"      {name:<16} {n:>6}" + (f"   ({d:+d})" if d else ""))
        loss.append({"stage": name, "n": n, "delta_vs_prev": d,
                     "pct_of_planned": round(100.0 * n / max(len(planned), 1), 3)})
        prev = n
    out["sample_ladder"] = loss

    rep = OUT / f"{arm}_finalization_report.json"
    out["finalization_report"] = (
        {"exists": True, "sha256": _sha(rep),
         **{k: v for k, v in json.loads(rep.read_text(encoding="utf-8")).items()
            if not isinstance(v, (list, dict))}}
        if rep.exists() else {"exists": False})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["h1", "mcq"])
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    checks: list = []
    arms = {}
    for arm in args.arms:
        print(f"═══ {arm}", flush=True)
        arms[arm] = check_arm(arm, checks)
        print(flush=True)

    # ── 7 protocol_deviation：一并列出，不省略 ────────────────────────────
    reg = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    PARSER_CALIBER = "采用**修复后的解析器**，正式生产期间未再修改；跨模型比较同时登记**预算**与**解析版本**差异"
    dev = {
        "mcq_budget": {
            "frozen_max_new_tokens": reg.get("frozen_max_new_tokens"),
            "registered_max_new_tokens": reg.get("registered_max_new_tokens"),
            "status": reg.get("status"),
            "registration_sha256": _sha(REG) if REG.exists() else None,
            "claim_limits": reg.get("claim_limits", []),
        },
        # ⚠️ **不能直接引用预算登记里的"未改"**：那份文件日期是 2026-09-14，
        # 那句话只对**当轮（预算校准）**成立。此后 L3G 做了 MCQ 答案提取的最小修复、
        # L3H 收紧解析规则（第二次修订）、L3I 重提取特征并重建 gate 证据，
        # L3J 才启动生产。照抄"未改"会漏掉这三步。
        "parser_caliber": PARSER_CALIBER,
        "parser_caliber_evidence": [
            "L3G MCQ 答案提取与定位的最小修复（2026-09-15，离线）",
            "L3H 收紧 MCQ 解析规则（2026-09-15，离线，第二次修订）",
            "L3I 解析修复后的特征重提取与 gate 证据重建（2026-09-15）",
            "L3J 生产准备与正式生成启动（2026-09-15）",
            "docs/progress/stage1_llama_replication_history.md",
        ],
        "registration_parser_note_scope": (
            "budget512_registration.json 里的 "
            "\"domain_label_proxy.parse_option_answer（未改）\" "
            "**只描述该登记当轮**（2026-09-14 的预算校准），不涵盖此后的 L3G/L3H 修订"),
        "cross_model_registered_differences": ["生成预算（16 → 512）", "解析器版本"],
        "held_fixed": reg.get("held_fixed"),
        "population_role": "cross_model_replication_non_confirmatory",
        "must_not_claim": [
            "**不得**称解析口径全程未改 —— 生产前经 L3G/L3H 两次修订，"
            "L3I 重提取特征并重建 gate 证据",
            "生成完成不等于方法复现成功",
            "finalize 通过不等于方法复现成功",
            "**不是**原冻结配置下的严格同参数复现 —— MCQ 的生成上限由冻结的 16 "
            "扩展到登记的 512，属同一研究方法下透明登记的预算扩展",
        ],
        "analysis_authorization": (
            "本轮授权覆盖生成与 finalize。跨模型性能评估只执行原先已批准的方案；"
            "当前未获新增分析授权，故只交付验收结果与分析就绪情况，不擅自新增分析方案。"),
    }
    print("═══ protocol_deviation")
    print(f"  MCQ 生成上限  冻结 {dev['mcq_budget']['frozen_max_new_tokens']} "
          f"→ 登记 {dev['mcq_budget']['registered_max_new_tokens']}"
          f"（{dev['mcq_budget']['status']}）")
    print(f"  解析口径      {dev['parser_caliber']}")
    print(f"                依据 {'; '.join(dev['parser_caliber_evidence'][:4])}")
    for c in dev["must_not_claim"]:
        print(f"  ⚠️ {c}")

    ok = all(c["ok"] for c in checks)
    out = {"schema": "llama_production_acceptance_v1",
           "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "verdict": "PASS" if ok else "FAIL",
           "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["ok"]),
           "checks": checks, "arms": arms, "protocol_deviation": dev,
           "scope": "只做生成与 finalize 的验收；**不执行任何跨模型性能评估**"}
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
                      encoding="utf-8")
    print(f"\n[{out['verdict']}] {len(checks) - out['n_failed']}/{len(checks)} 项通过")
    print(f"[out] {REPORT}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
