"""Stage 0 production generation + feature cache（W0.5-F）。

    python scripts/run_stage0_generation.py --arm h1  --dry-run
    python scripts/run_stage0_generation.py --arm mcq --dry-run
    python scripts/run_stage0_generation.py --arm h1            # 需 G1+G2+G3 全绿

**本轮范围:只实现,不启动真实 2880/1760 生成。** 真跑受 stage0 门控:G1 未过即停。

产物契约(必须正好是 run_stage0_analysis.py 读得懂的格式):
    outputs/stage0/features/h1_completion_records.jsonl   + h1_hidden.npz
    outputs/stage0/features/mcq_completion_records.jsonl  + mcq_hidden.npz
    outputs/stage0/generation/{arm}_generated_manifest.jsonl   (resume 日志)
    outputs/stage0/generation/{arm}_run_fingerprint.json

规模事实:H 是每 completion 一个 pooled 向量(3584 维 fp32 = 14 KB) →
H1 2880 ≈ 41 MB、MCQ 1760 ≈ 25 MB。npz 直存,无需分片。§11 不缓存 raw attention/全层。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.generation import fingerprint as fp  # noqa: E402
from recover_attention.generation.backends import ForwardNonFiniteError  # noqa: E402
from recover_attention.generation.manifest import (  # noqa: E402
    STATUS_DONE,
    STATUS_EXCLUDED,
    STATUS_FAILED,
    STATUS_RUNNING,
    WorkManifest,
    write_atomic,
)

ARM_H1, ARM_MCQ = "h1", "mcq"
CACHE_SCHEMA = {ARM_H1: "stage0_h1_completion_cache_v1",
                ARM_MCQ: "stage0_mcq_completion_cache_v1"}
K_TRACES = 6                     # §3 冻结:1 greedy + 5 sampled


# --------------------------------------------------------------------- gating
def require_stage0_gate(args) -> dict:
    """真跑前的**科学启动闸**（v2.4 §2）：G2 + G3 + G4。

    **不含 G1**：v2.4 把场地事实（CFP / page limit / archival）从启动条件里拆出去了 ——
    它决定能不能投、投哪里，不决定测到了什么。G1 现在只挡投稿（见 stage0_preflight）。
    """
    from recover_attention import stage0_preflight as pf

    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit(f"STOP (G2): preregistration hash mismatch\n  current={g2['current']}\n"
                         f"  locked ={g2['locked']}")
    smoke_dir = Path(args.smoke_dir)
    g3_h1 = (smoke_dir / "smoke_report_h1.json").exists()
    g3_mcq = (smoke_dir / "smoke_report_mcq.json").exists()
    g4 = pf.check_g4(Path(args.output_root) / "stage0_g4_quality_check.json", g2["current"])
    allowed = bool(g2["match"] and g3_h1 and g3_mcq and g4["ok"])
    if not allowed:
        raise SystemExit(
            "STAGE 0 BLOCKED — the scientific launch gate requires G2+G3+G4 (v2.4 §2).\n"
            f"  G2 (prereg)  : {g2['match']}\n"
            f"  G3 (smoke)   : h1={g3_h1} mcq={g3_mcq}\n"
            f"  G4 (quality) : {g4['ok']}  {g4.get('reason') or g4.get('checks')}\n"
            "  (G1/CFP is a submission gate under v2.4 and does not block the experiment.)\n"
            "Use --dry-run to exercise the pipeline without generating.")
    return {"G2": g2, "G3": {"h1": g3_h1, "mcq": g3_mcq}, "G4": g4}


# --------------------------------------------------------------- work units
def h1_units(records: list[dict]) -> list[dict]:
    """H1: 每 prompt K=6 条 trace（1 greedy + 5 sampled）。"""
    units = []
    for r in records:
        for i in range(K_TRACES):
            st = "greedy" if i == 0 else "sampled"
            units.append({"unit_id": f"{r['example_id']}__{st}_{i}",
                          "example_id": r["example_id"], "sample_type": st, "sample_index": i})
    return units


def mcq_units(records: list[dict]) -> list[dict]:
    """MCQ: population = 每题一条 greedy;另 5 条 sampled **仅**供 F5 self-consistency（§3 v2.3）。"""
    units = []
    for r in records:
        for i in range(K_TRACES):
            st = "greedy" if i == 0 else "sampled"
            units.append({"unit_id": f"{r['example_id']}__{st}_{i}",
                          "example_id": r["example_id"], "sample_type": st, "sample_index": i,
                          "in_population": i == 0})
    return units


# ------------------------------------------------------------------ dry-run
class _FakeBackend:
    """--dry-run 的假后端：形状与字段齐全，数值无意义。

    存在的意义是让 resume/指纹/产物契约在**没有 GPU、没有真数据**时也能被测到。
    它绝不能产出可解读的数值,故 hidden 是纯噪声、completion 是模板。
    """

    def __init__(self, arm: str, hidden_dim: int = 3584, seed: int = 0):
        self.arm = arm
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)
        self.backend = {"load_in_8bit": True, "device_map": "auto",
                        "attn_implementation": "eager", "local_files_only": True}

    def run_unit(self, unit: dict, record: dict) -> dict:
        r = self.rng
        if self.arm == ARM_MCQ:
            letter = "ABCD"[int(r.integers(0, 4))]
            return {"raw_completion": letter, "parsed_label": letter,
                    "hidden": r.normal(0, 1, self.hidden_dim).astype(np.float32),
                    "f5_label_margin": float(r.normal(5, 3)),
                    "f5_label_entropy": float(abs(r.normal(0.3, 0.2))),
                    "f5_full_entropy": float(abs(r.normal(0.3, 0.2))),
                    "f5_letter_token_logprob": float(-abs(r.normal(0.2, 0.2)))}
        return {"completion": f"dry-run completion for {unit['unit_id']} T1114.003",
                "hidden": r.normal(0, 1, self.hidden_dim).astype(np.float32),
                "f5_id_logprob_mean": float(r.normal(-1, 0.5))}


# ----------------------------------------------------------------- pipeline
def generate_arm(arm: str, records: list[dict], backend: Any, out_gen: Path, out_feat: Path,
                 run_fp: dict, resume: bool = True,
                 progress_every: int = 100) -> dict:
    """跑一臂:登记 units → 续跑未完成的 → 落 feature cache。

    崩溃安全:每条 unit 先写 running 再写 done/failed,状态是 append-only 且 fsync。
    重跑时 running 视为未完成重新入队 —— 上次被杀留下的 running 不重跑就会丢数据。
    """
    units = (h1_units if arm == ARM_H1 else mcq_units)(records)
    by_id = {r["example_id"]: r for r in records}
    man = WorkManifest(out_gen / f"{arm}_generated_manifest.jsonl")
    man.register(u["unit_id"] for u in units)

    all_ids = [u["unit_id"] for u in units]
    todo = man.pending_units(all_ids) if resume else all_ids
    unit_by_id = {u["unit_id"]: u for u in units}
    print(f"[gen:{arm}] units={len(units)} todo={len(todo)} "
          f"(resume={resume}; already done={len(units) - len(todo)})")

    results_path = out_gen / f"{arm}_unit_results.jsonl"
    hid_dir = out_gen / f"{arm}_unit_hidden"
    hid_dir.mkdir(parents=True, exist_ok=True)
    done_payloads: dict[str, dict] = {}
    if results_path.exists():
        for line in open(results_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break            # 尾部损坏 → 丢弃,append-only 的损坏只会在末尾
            done_payloads[row["unit_id"]] = row

    # 一致性自愈:manifest 说 done、但结果或 hidden 不在盘上 → 重新入队。
    # 这是 resume 最阴险的失败模式:它把数据丢失伪装成"已完成"。
    # （早期实现把 hidden 攒在内存里最后才 savez,中断后 38 个 done 的 unit 向量全丢,
    #   而 manifest 照样说 done → n_records=12 但 n_hidden=6。）
    def _requires_hidden(uid: str) -> bool:
        payload = done_payloads.get(uid)
        if payload is None:
            return True
        if arm == ARM_H1:
            return bool(payload.get("id_token_positions_local"))
        # MCQ:parse_failure(eligible_for_primary=False)在 §7.1 下**不进 population**,
        # 从不计算 F5/hidden——这是正确的终态,不是数据丢失。此前这里恒 True,把每一条
        # parse_failure 都当成"该有 hidden 却没有"的孤儿,每次 resume 都重新入队、重跑、
        # 再次判定孤儿,自愈判定永远不收敛(真实撞见过:cybermetric_000718 的 greedy/
        # sampled_4 连续两次 resume 被重新排队,GPU 时间白费,直到这次才发现)。
        return bool(payload.get("eligible_for_primary"))

    def _durable(uid: str) -> bool:
        if uid not in done_payloads:
            return False
        return (not _requires_hidden(uid)) or (hid_dir / f"{_safe(uid)}.npy").exists()

    st = man.current_status()
    orphaned = [u for u in all_ids if st.get(u) == STATUS_DONE and not _durable(u)]
    if orphaned:
        print(f"[gen:{arm}] {len(orphaned)} units marked done but missing durable output "
              f"-> re-queued (e.g. {orphaned[:2]})")
        todo = sorted(set(todo) | set(orphaned))

    n_ok = n_fail = n_excl = 0
    for i, uid in enumerate(todo, 1):
        unit = unit_by_id[uid]
        rec = by_id[unit["example_id"]]
        man.mark(uid, STATUS_RUNNING, attempt=man.attempts(uid) + 1)
        try:
            res = backend.run_unit(unit, rec)
            h = res.pop("hidden", None)
            # **先把 hidden 落盘,再写 done** —— 顺序反了就会出现"done 但向量不存在"
            if h is not None:
                _save_npy_atomic(hid_dir / f"{_safe(uid)}.npy", np.asarray(h, dtype=np.float32))
            payload = {"unit_id": uid, **{k: v for k, v in unit.items() if k != "unit_id"}, **res}
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=_jd) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done_payloads[uid] = payload
            man.mark(uid, STATUS_DONE)
            n_ok += 1
        except ForwardNonFiniteError as exc:
            # 终态排除,不是"待重试的失败"——数值层面的事重跑不会变好。不写 results 行、
            # 不写 hidden;诊断判据整份存进 manifest 事件(不经 repr(exc)[:300] 截断)。
            man.mark(uid, STATUS_EXCLUDED, reason=exc.diagnostic["reason"],
                     diagnostic=exc.diagnostic)
            n_excl += 1
        except Exception as exc:                       # noqa: BLE001
            man.mark(uid, STATUS_FAILED, error=repr(exc)[:300])
            n_fail += 1
        if i % progress_every == 0 or i == len(todo):
            print(f"[gen:{arm}] {i}/{len(todo)} ok={n_ok} failed={n_fail} excluded={n_excl}")

    hidden_store = {p.stem: np.load(p) for p in sorted(hid_dir.glob("*.npy"))}
    summary = man.summary(all_ids)
    if not summary["complete"]:
        return {"arm": arm, "status": "incomplete", "manifest": summary,
                "note": "rerun to resume; feature cache is NOT written until all units are done"}

    st_final = man.current_status()
    excluded_events = {u: man.last_event(u) for u in all_ids if st_final.get(u) == STATUS_EXCLUDED}
    cache = _build_feature_cache(arm, records, done_payloads, hidden_store, run_fp, out_feat,
                                 excluded_events)
    return {"arm": arm, "status": "complete", "manifest": summary, "feature_cache": cache}


def _safe(unit_id: str) -> str:
    """unit_id → 文件名安全的形式（unit_id 里有 '__'，无路径分隔符，直接可用）。"""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in unit_id)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """原子写单条 hidden：写临时文件 → fsync → rename。

    per-unit 落盘（而非最后统一 savez）是 resume 正确性的前提:npz 不支持追加,攒在内存里
    最后写 = 中断即全丢,而 manifest 却说 done。2880 × 14 KB 的小文件完全可接受。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _jd(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def _build_feature_cache(arm: str, records: list[dict], payloads: dict[str, dict],
                         hidden_store: dict[str, np.ndarray], run_fp: dict,
                         out_feat: Path, excluded_events: dict[str, dict] | None = None) -> dict:
    """组装 run_stage0_analysis.py 读得懂的 feature cache。

    这是两条 pipeline 的**契约点**:字段名/文件名必须与 evaluation.feature_cache.load_arm
    对得上,否则生成完了分析入口读不进去。tests/test_stage0_generation.py 直接跑通这条链
    来钉住它。

    excluded_events:manifest 里终态 excluded 的 unit_id -> 事件(含 reason/diagnostic)。
    MCQ 的主样本是每题一条 greedy(§3 v2.3);greedy 若落进 excluded,该题必须从 primary
    population **可见地**消失,而不是 payloads.get(...) 返回 None 就悄悄 continue —— 决策
    (a,2026-07-21 确认):排除,不用 sampled 顶替(顶替会改写"主样本=greedy"这条协议定义,
    而且"这题可以替、那题不替"本身比直接排除更难向审稿人解释)。
    """
    excluded_events = excluded_events or {}
    out_feat.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    hidden: dict[str, np.ndarray] = {}
    excluded_primary: list[dict] = []
    by_group = {r["example_id"]: r["group_id"] for r in records}

    # 跨样本 F5（self-consistency / agreement）必须在这里算,且必须用**全部** K 个 unit
    # 聚合(而不是只用最终进 population 的行) —— MCQ 的 population 只含 greedy,但一致性
    # 要看它与 5 条 sampled 手足的吻合度;H1 则是每条 completion 各自一条主样本,但同一
    # prompt 的 K 条互为彼此的一致性参照。attach_cross_sample_f5 原地修改 dict,而
    # payloads 的 value 与下面构造 rows 时使用的是同一批对象,故必须先调用这个函数,
    # 再从 payloads 取值 —— 顺序反了,rows 里拿到的还是修改前的（缺失）版本。
    # 此前这里从未调用过此函数:两列被 _dense_matrix 静默填 0/常数,不改变任何检查的
    # 通过与否(F5 completeness 检查本该抓住这个,却因为“连续量”判定漏了信号量太弱的列),
    # 但会在下游削弱 O 的基线、抬高 Δ_H。
    from recover_attention.generation.backends import attach_cross_sample_f5
    attach_cross_sample_f5(list(payloads.values()), arm)

    if arm == ARM_H1:
        # §3:H1 的主样本是**每条 completion**,不是每题一条。K=6 → 480×6 = 2880 traces,
        # 同一 prompt 的 6 条共享 group(§3:"K 条 completion 共享 group,不当独立样本")。
        # 只取 greedy 会把 population 从 ~2760 砍到 480 —— 而且是静默的:形状对、
        # 分析照跑,只是有效样本量塌掉,Δ_H 的 CI 宽到装不进 ±ε,该臂被判 inconclusive,
        # 原因却是代码 bug 不是真实精度限制。
        for uid in sorted(payloads):
            p = payloads[uid]
            eid = p["example_id"]
            row = {"schema_version": CACHE_SCHEMA[arm], "trace_id": uid,
                   "example_id": eid, "group_id": by_group[eid],
                   "population_role": "confirmatory", "run_hash": run_fp["run_hash"],
                   **{k: v for k, v in p.items() if k not in ("unit_id", "example_id")}}
            rows.append(row)
            if _safe(uid) in hidden_store:
                hidden[uid] = hidden_store[_safe(uid)]
    else:
        # §3 v2.3:MCQ population = 每题一条 greedy;K−1 条 sampled **仅**供 F5
        # self-consistency / letter-agreement,不进 population。
        for r in records:
            eid = r["example_id"]
            greedy = payloads.get(f"{eid}__greedy_0")
            if greedy is None:
                # complete=True 时到这里只有一种可能:greedy 落进了终态 excluded
                # （不会是 pending/running/failed,那些状态下 summary["complete"] 为 False,
                #   generate_arm 根本不会走到这里）。仍显式取事件、标注找不到时的兜底原因,
                # 不让"消失"这件事在任何情况下悄悄发生。
                ev = excluded_events.get(f"{eid}__greedy_0") or {}
                excluded_primary.append({
                    "example_id": eid, "group_id": r["group_id"],
                    "reason": ev.get("reason", "missing_greedy_unaccounted"),
                    "diagnostic": ev.get("diagnostic"),
                })
                continue
            row = {"schema_version": CACHE_SCHEMA[arm], "example_id": eid,
                   "group_id": r["group_id"], "population_role": "confirmatory",
                   "run_hash": run_fp["run_hash"],
                   **{k: v for k, v in greedy.items() if k not in ("unit_id",)}}
            rows.append(row)
            key = _safe(f"{eid}__greedy_0")
            if key in hidden_store:
                hidden[eid] = hidden_store[key]

    # 契约:population 里的每条记录都必须有 hidden。少一条就说明 resume 漏了单元 ——
    # 与其让分析入口后面报"missing hidden",不如在生成端当场停。
    id_field = "trace_id" if arm == ARM_H1 else "example_id"
    if arm == ARM_H1:
        missing = [r[id_field] for r in rows
                   if r.get("id_token_positions_local") and r[id_field] not in hidden]
    else:
        # MCQ parse_failure(eligible_for_primary=False)按 §7.1 从不计算 hidden —— 这是
        # 正确终态,不是数据丢失,同一理由与上面 _requires_hidden 的修法对齐;缺 hidden
        # 只在 eligible 记录上才是需要当场停的信号。
        missing = [r[id_field] for r in rows
                   if r.get("eligible_for_primary") and r[id_field] not in hidden]
    if missing:
        raise RuntimeError(
            f"{arm}: {len(missing)}/{len(rows)} population records have no hidden vector "
            f"(e.g. {missing[:3]}). The manifest and the durable output disagree — do not "
            f"write a feature cache from this state.")

    write_atomic(out_feat / f"{arm}_completion_records.jsonl",
                 "".join(json.dumps(r, ensure_ascii=False, default=_jd) + "\n" for r in rows))
    np.savez_compressed(out_feat / f"{arm}_hidden.npz", **hidden)
    write_atomic(out_feat / f"{arm}_run_fingerprint.json",
                 json.dumps(run_fp, ensure_ascii=False, indent=2, default=_jd))
    mb = sum(v.nbytes for v in hidden.values()) / 1e6
    result = {"n_records": len(rows), "n_hidden": len(hidden),
             "hidden_mb_uncompressed": round(mb, 2),
             "records_path": str(out_feat / f"{arm}_completion_records.jsonl"),
             "hidden_path": str(out_feat / f"{arm}_hidden.npz")}

    if arm == ARM_MCQ:
        n_sampled_excluded = sum(
            1 for ev in excluded_events.values()
            if (ev.get("diagnostic") or {}).get("sample_type") == "sampled")
        # 写进 records 文件的 rows 里,还有一类跟 forward_nonfinite 无关、§7.1 早就定义好
        # 的排除:parse_failure(eligible_for_primary=False,模型输出解析不到唯一选项)。
        # 这类行**会**被写进 mcq_completion_records.jsonl(与 H1 同规则),但不进 confirmatory
        # 分析——evaluation.feature_cache.load_arm 在读取时按 eligible_for_primary 过滤。
        # 所以"写了多少行"和"多少行真正喂进分析"是两个数,报告里不能混成一个,
        # 否则"analyzable"这个词本身就是误导。
        n_parse_failure = sum(1 for r in rows if not r.get("eligible_for_primary"))
        exclusion_report = {
            "schema_version": "stage0_mcq_primary_exclusion_report_v1",
            "decision": ("a: exclude — do not backfill primary with a sampled trace "
                        "(2026-07-21)"),
            "n_nominal_questions": len(records),
            "n_primary_records_written": len(rows),
            "n_primary_records_eligible": len(rows) - n_parse_failure,
            "n_primary_records_parse_failure": n_parse_failure,
            "n_excluded_primary_forward_nonfinite": len(excluded_primary),
            "excluded_primary": excluded_primary,
            "n_sampled_traces_excluded_or_degraded": n_sampled_excluded,
            "note": ("n_primary_records_written = rows in mcq_completion_records.jsonl "
                     "(includes parse_failure rows, which load_arm filters out by "
                     "eligible_for_primary at analysis time — see §7.1). "
                     "n_primary_records_eligible is the count that actually feeds AUROC. "
                     "sampled traces excluded here only shrink the sibling set that "
                     "attach_cross_sample_f5 averages over for self-consistency/agreement; "
                     "they do not remove a question from the primary population unless the "
                     "greedy trace itself is excluded (see excluded_primary above)."),
        }
        write_atomic(out_feat / f"{arm}_primary_exclusion_report.json",
                     json.dumps(exclusion_report, ensure_ascii=False, indent=2, default=_jd))
        result["primary_exclusion_report"] = str(out_feat / f"{arm}_primary_exclusion_report.json")
        result["n_excluded_primary"] = len(excluded_primary)

    return result


def _load_production_records(args) -> list[dict]:
    """真跑的输入记录。MCQ 只取 fresh 1760（burned 240 是 exploratory，v2.3 已排除）。"""
    from recover_attention import mcq_conditional_increment as mci
    from recover_attention.data_io import read_jsonl

    if args.arm == ARM_H1:
        recs = read_jsonl(Path(args.h1_samples))
        if len(recs) != H1_PROMPTS_EXPECTED:
            raise SystemExit(f"expected {H1_PROMPTS_EXPECTED} H1 prompts, got {len(recs)}")
        return recs

    pool = read_jsonl(Path(args.mcq_pool))
    burned = {json.loads(l)["example_id"]
              for l in open(args.burned_traces, encoding="utf-8")}
    split = mci.split_fresh_confirmatory([r["example_id"] for r in pool], sorted(burned))
    fresh = set(split["fresh_ids"])
    if len(fresh) != MCQ_FRESH_EXPECTED:
        raise SystemExit(f"expected {MCQ_FRESH_EXPECTED} fresh MCQ ids, got {len(fresh)}")
    tok_prompt = None
    out = []
    for r in pool:
        if r["example_id"] not in fresh:
            continue
        out.append({**r, "rendered_prompt": _render_mcq_prompt(r, args)})
    return out


def _render_mcq_prompt(record: dict, args) -> str:
    """与 4B-2 A/B 胜出的 chat 条件**同一个**构造器（prompt_style='chat'）。

    必须复用 cyber_data.build_mcq_chat_messages,不能自己拼 —— 手拼时用
    record["input_text"] 会得到空串(该字段为空),fallback 到 question 就**不含选项**,
    模型看不见 A/B/C/D,实测 2/2 parse_failure。prompt 条件也是 4B-2 A/B 选出来的,
    换一个就不再是同一个实验条件。
    """
    global _MCQ_TOKENIZER
    from recover_attention import cyber_data as cd
    if _MCQ_TOKENIZER is None:
        from recover_attention.hidden_state_cache import import_transformers_components
        _MCQ_TOKENIZER = import_transformers_components()["AutoTokenizer"].from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False)
    return _MCQ_TOKENIZER.apply_chat_template(
        cd.build_mcq_chat_messages(record), tokenize=False, add_generation_prompt=True)


_MCQ_TOKENIZER = None
H1_PROMPTS_EXPECTED = 480
MCQ_FRESH_EXPECTED = 1760


def _production(args, out_gen: Path, out_feat: Path) -> None:
    """真跑。到这里说明 G1+G2+G3 已全绿。"""
    records = _load_production_records(args)
    print(f"[gen:{args.arm}] production records: {len(records)}")

    backend = _make_backend(args)
    hv = backend.verify_hidden_index()
    print(f"[gen:{args.arm}] hidden index {hv}")
    if not hv["numeric_match"]:
        raise SystemExit("STOP: forward hook != hidden_states[tuple_idx] (off-by-one?)")

    run_fp = backend.run_fingerprint(args.arm)
    res = generate_arm(args.arm, records, backend, out_gen, out_feat, run_fp,
                       resume=not args.no_resume, progress_every=50)
    res["mode"] = "production"
    res["hidden_verification"] = hv
    write_atomic(out_gen / f"{args.arm}_generation_report.json",
                 json.dumps(res, ensure_ascii=False, indent=2, default=_jd))
    print(f"[gen:{args.arm}] status={res['status']} manifest={res['manifest']}")
    if res["status"] == "complete":
        print(f"[gen:{args.arm}] feature cache: {res['feature_cache']}")


def _make_backend(args):
    """按 arm 造真后端（供 _production / _replication / _preflight 复用同一条路径）。"""
    from recover_attention.generation import backends as bk
    if args.arm == ARM_H1:
        from recover_attention.h1_identifier import build_ontology_index
        return bk.H1Backend(args.model_path, build_ontology_index(Path(args.ontology_dir)),
                            seed=args.seed, max_new_tokens=args.max_new_tokens)
    return bk.MCQBackend(args.model_path, seed=args.seed, max_new_tokens=args.mcq_max_new_tokens)


def _preflight(args) -> None:
    """换机器/换模型前的 ~2min 自检：加载模型 + 核验 §8 层位数值一致 + 打印 backend。

    不生成任何数据、不写任何产物。numeric_match=False（层位 off-by-one）→ 非零退出，
    让调用方在启动 30h 生成之前就知道环境有问题。
    """
    print(f"[preflight] loading model: {args.model_path}")
    backend = _make_backend(args)
    hv = backend.verify_hidden_index()
    print(f"[preflight] arm={args.arm}")
    print(f"[preflight] num_layers={hv['num_layers']} rel_depth={hv['rel_depth']} "
          f"block={hv['block_index_zero_based']} tuple_index={hv['hidden_states_tuple_index']}")
    print(f"[preflight] forward-hook == hidden_states[tuple]: numeric_match={hv['numeric_match']} "
          f"max_abs_diff={hv['max_abs_diff']:.2e}")
    print(f"[preflight] backend={backend.backend}")
    if not hv["numeric_match"]:
        raise SystemExit("[preflight] FAIL: forward hook != hidden_states[tuple_idx] "
                         "(off-by-one?). Do NOT launch generation.")
    print("[preflight] OK — layer index verified. Safe to launch generation.")


def _replication(args, out_gen: Path, out_feat: Path) -> None:
    """§9 跨模型复核。**非 confirmatory。**

    与 _production 共用**同一条**生成机制（generate_arm / _build_feature_cache），只在两点
    不同：(1) 跳过 Qwen 专属的 G3/G4 启动闸——它们的 sha256 与 smoke 只为 Qwen 挣得，拿它们
    给 Llama 放行是错的；(2) 仍校验 G2（设计冻结未变，§9 要求"若跑则须同协议"）。
    产物写独立 --output-root，run_fingerprint 天然带 Llama 的 model_hash/层位，不会与 Qwen
    的 confirmatory cache 混淆。此路径不建 freeze manifest、不写 unblinding log——那套是
    confirmatory 专用且一次性已用尽。
    """
    if args.arm != ARM_H1:
        raise SystemExit("--cross-model-replication 仅用于 H1 臂（§9 决策 a：第二模型只跑 H1）")
    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit(f"STOP (G2): preregistration hash mismatch — 设计已漂移，"
                         f"复核必须在同一冻结设计下进行\n  current={g2['current']}\n"
                         f"  locked ={g2['locked']}")
    print(f"[replication] §9 cross-model H1 replication (NOT confirmatory). "
          f"G2 ok; G3/G4 intentionally skipped (Qwen-specific).")
    print(f"[replication] model={args.model_path}  output-root={args.output_root}")

    records = _load_production_records(args)
    print(f"[replication:h1] records: {len(records)}")
    backend = _make_backend(args)
    hv = backend.verify_hidden_index()
    print(f"[replication:h1] hidden index {hv}")
    if not hv["numeric_match"]:
        raise SystemExit("STOP: forward hook != hidden_states[tuple_idx] (off-by-one?)")

    run_fp = backend.run_fingerprint(args.arm)
    run_fp["run_role"] = "cross_model_replication_non_confirmatory"
    run_fp["preregistration_sha256"] = g2["current"]
    res = generate_arm(args.arm, records, backend, out_gen, out_feat, run_fp,
                       resume=not args.no_resume, progress_every=50)
    res["mode"] = "cross_model_replication"
    res["is_confirmatory"] = False
    res["hidden_verification"] = hv
    write_atomic(out_gen / f"{args.arm}_generation_report.json",
                 json.dumps(res, ensure_ascii=False, indent=2, default=_jd))
    print(f"[replication:h1] status={res['status']} manifest={res['manifest']}")
    if res["status"] == "complete":
        print(f"[replication:h1] feature cache: {res['feature_cache']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[ARM_H1, ARM_MCQ], required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="假后端跑通 resume/指纹/产物契约;不调模型、不生成真数据")
    ap.add_argument("--dry-run-units", type=int, default=12,
                    help="dry-run 用多少个 prompt（默认 12，仅为速度）")
    ap.add_argument("--output-root", default="outputs/stage0")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--cfp-record", default="docs/paper/cfp_record.json")
    ap.add_argument("--smoke-dir", default="outputs/logs/sprint_4D_2_conditional_increment")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--h1-samples", default="data/processed/h1/h1_samples.jsonl")
    ap.add_argument("--mcq-pool", default="data/processed/cyber/cybermetric.jsonl")
    ap.add_argument("--burned-traces",
                    default="outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/"
                            "trace_sampling_manifest.jsonl")
    ap.add_argument("--ontology-dir", default="data/raw/ontology")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--mcq-max-new-tokens", type=int, default=16)
    ap.add_argument("--preflight", action="store_true",
                    help="只加载模型、核验 §8 层位(forward-hook == hidden_states[tuple])、"
                         "打印 backend/dtype 后退出;不生成。用于换机器/换模型前的 ~2min 环境自检")
    ap.add_argument("--cross-model-replication", action="store_true",
                    help="§9 跨模型复核（**非** confirmatory）：跳过 Qwen 的 G3/G4 启动闸"
                         "（它们只为 Qwen 挣得，不能给别的模型放行），保留 G2 设计冻结校验，"
                         "强制 arm=h1，产物打 replication 标记、写独立目录。confirmatory 的"
                         "一次性揭盲已用尽，此路径绝不触碰 freeze/unblinding。")
    args = ap.parse_args()

    root = Path(args.output_root)
    out_gen = root / ("dry_run/generation" if args.dry_run else "generation")
    out_feat = root / ("dry_run/features" if args.dry_run else "features")

    if args.preflight:
        return _preflight(args)

    if not args.dry_run:
        if args.cross_model_replication:
            return _replication(args, out_gen, out_feat)
        require_stage0_gate(args)       # G1+G2+G3 缺任一即停
        return _production(args, out_gen, out_feat)

    # ---- dry run ----
    n = args.dry_run_units
    records = [{"example_id": f"{args.arm}_{i:05d}", "group_id": f"{args.arm}_g{i:05d}",
                "question_text": f"synthetic question {i}"} for i in range(n)]
    backend = _FakeBackend(args.arm, seed=args.seed)
    run_fp = fp.run_fingerprint(
        model_fp={"model_path": "DRY_RUN", "model_hash": "dry-run", "backend": backend.backend},
        tokenizer_fp={"tokenizer_hash": "dry-run"},
        arm=args.arm, hidden_index=ci.resolve_hidden_index(28),
        gen_params={"K": K_TRACES, "max_new_tokens": 384, "temperature": 0.7, "top_p": 0.95})
    res = generate_arm(args.arm, records, backend, out_gen, out_feat, run_fp,
                       resume=not args.no_resume, progress_every=max(1, n))
    res["mode"] = "dry_run"
    res["data_is_synthetic"] = True
    write_atomic(out_gen / f"{args.arm}_generation_report.json",
                 json.dumps(res, ensure_ascii=False, indent=2, default=_jd))
    print(f"[gen:{args.arm}] status={res['status']} manifest={res['manifest']}")
    if res["status"] == "complete":
        print(f"[gen:{args.arm}] feature cache: {res['feature_cache']}")


if __name__ == "__main__":
    main()
