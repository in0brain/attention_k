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
from recover_attention.generation.manifest import (  # noqa: E402
    STATUS_DONE,
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
    """真跑前的门控。G1+G2+G3 缺任一即停 —— production generation 是 Stage 0 的一部分。"""
    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit(f"STOP (G2): preregistration hash mismatch\n  current={g2['current']}\n"
                         f"  locked ={g2['locked']}")
    g1 = ci.check_cfp_confirmed(args.cfp_record)
    smoke_dir = Path(args.smoke_dir)
    g3_h1 = (smoke_dir / "smoke_report_h1.json").exists()
    g3_mcq = (smoke_dir / "smoke_report_mcq.json").exists()
    allowed = bool(g1["ok"] and g2["match"] and g3_h1 and g3_mcq)
    if not allowed:
        raise SystemExit(
            "STAGE 0 BLOCKED — production generation requires G1+G2+G3 all green.\n"
            f"  G1 (CFP)   : {g1['ok']}  status={g1.get('status')} missing={g1.get('missing')}\n"
            f"  G2 (prereg): {g2['match']}\n"
            f"  G3 (smoke) : h1={g3_h1} mcq={g3_mcq}\n"
            "Use --dry-run to exercise the pipeline without generating.")
    return {"G1": g1, "G2": g2, "G3": {"h1": g3_h1, "mcq": g3_mcq}}


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
    def _durable(uid: str) -> bool:
        if uid not in done_payloads:
            return False
        return (hid_dir / f"{_safe(uid)}.npy").exists()

    st = man.current_status()
    orphaned = [u for u in all_ids if st.get(u) == STATUS_DONE and not _durable(u)]
    if orphaned:
        print(f"[gen:{arm}] {len(orphaned)} units marked done but missing durable output "
              f"-> re-queued (e.g. {orphaned[:2]})")
        todo = sorted(set(todo) | set(orphaned))

    n_ok = n_fail = 0
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
        except Exception as exc:                       # noqa: BLE001
            man.mark(uid, STATUS_FAILED, error=repr(exc)[:300])
            n_fail += 1
        if i % progress_every == 0 or i == len(todo):
            print(f"[gen:{arm}] {i}/{len(todo)} ok={n_ok} failed={n_fail}")

    hidden_store = {p.stem: np.load(p) for p in sorted(hid_dir.glob("*.npy"))}
    summary = man.summary(all_ids)
    if not summary["complete"]:
        return {"arm": arm, "status": "incomplete", "manifest": summary,
                "note": "rerun to resume; feature cache is NOT written until all units are done"}

    cache = _build_feature_cache(arm, records, done_payloads, hidden_store, run_fp, out_feat)
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
                         out_feat: Path) -> dict:
    """组装 run_stage0_analysis.py 读得懂的 feature cache。

    这是两条 pipeline 的**契约点**:字段名/文件名必须与 evaluation.feature_cache.load_arm
    对得上,否则生成完了分析入口读不进去。tests/test_stage0_generation.py 直接跑通这条链
    来钉住它。
    """
    out_feat.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    hidden: dict[str, np.ndarray] = {}
    for r in records:
        eid = r["example_id"]
        greedy = payloads.get(f"{eid}__greedy_0")
        if greedy is None:
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
    missing = [r["example_id"] for r in rows if r["example_id"] not in hidden]
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
    return {"n_records": len(rows), "n_hidden": len(hidden),
            "hidden_mb_uncompressed": round(mb, 2),
            "records_path": str(out_feat / f"{arm}_completion_records.jsonl"),
            "hidden_path": str(out_feat / f"{arm}_hidden.npz")}


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.output_root)
    out_gen = root / ("dry_run/generation" if args.dry_run else "generation")
    out_feat = root / ("dry_run/features" if args.dry_run else "features")

    if not args.dry_run:
        gates = require_stage0_gate(args)
        print(f"[gen] gates green: {json.dumps({k: (v if isinstance(v, dict) else v) for k, v in gates.items()}, default=str)[:120]}")
        raise SystemExit(
            "production generation is not wired to a real backend in W0.5-F by design: "
            "this sprint implements the pipeline only and must not launch the real "
            "2880/1760 run. Wire the real backend in the Stage 0 launch commit.")

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
