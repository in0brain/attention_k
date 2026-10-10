r"""生成长度检查：在校准题单上跑一次 Llama MCQ，检查候选预算是否满足**事前登记**的
运行质量要求。

**这不是 gate，也不是 G3。** 判据只看完整性、异常/非有限值与截断率；
**不看正确率、不看解析成功率**。解析结果如实记录并报告，但不参与选值。

走的就是 G3 的那条路径：`v3_g3_smoke.build_arm_records` → `units_for` → `run_arm`
（`MCQBackend.run_unit`，含 teacher-forced 前向、hidden、F5、非有限值检测）。
**不另写提示、生成或解析逻辑。** 唯一的差别是题单与解码预算。

用法：
    python scripts/diagnostics/run_mcq_budget_check.py
    python scripts/diagnostics/run_mcq_budget_check.py --smoke 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool      # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.evaluation.v3_freeze import require_v3_frozen  # noqa: E402
from recover_attention.evaluation.v3_g4_quality import G4Criteria    # noqa: E402
from recover_attention.generation import backends as bk              # noqa: E402
from recover_attention.generation.manifest import write_atomic       # noqa: E402

MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR = ROOT / "outputs/stage1_llama_v3/mcq_budget512_check"
REGISTRATION = ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
GATE_TAG = "budget512"


class _GenerationRecorder:
    """包住 model.generate，记录**真实**新生成 token 序列。与 G4 的 token 计数器同法，
    纯观测：不改生成行为，也不改 backends 的任何一行。"""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.last: dict | None = None
        self._orig = ctx["model"].generate

        def recording(**kw):
            ids = self._orig(**kw)
            plen = int(kw["input_ids"].shape[-1])
            new = [int(t) for t in ids[0][plen:].tolist()]
            self.last = {"prompt_len": plen, "generated_ids_raw": new,
                         "n_generated_tokens": len(new),
                         "last_generated_token_id": (new[-1] if new else None),
                         "prompt_input_ids_head": [int(t) for t in
                                                   kw["input_ids"][0][:4].tolist()],
                         "prompt_n_bos_128000": int(
                             (kw["input_ids"][0] == 128000).sum().item()),
                         "generate_kwargs": {k: v for k, v in kw.items()
                                             if k not in ("input_ids", "attention_mask")},
                         }
            return ids

        ctx["model"].generate = recording

    def restore(self) -> None:
        self.ctx["model"].generate = self._orig


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    """仓库内给相对路径，仓库外（自检临时目录）如实给绝对路径。"""
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def versions() -> dict:
    import accelerate, bitsandbytes, torch, transformers   # noqa: PLC0415,E401
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__,
            "python": sys.version.split()[0], "host": platform.node()}


def _progress(payload: dict) -> None:
    rate = payload["elapsed_s"] / max(payload["done"], 1)
    print(f"[gen  ] {payload['done']}/{payload['total']} elapsed={payload['elapsed_s']:.0f}s "
          f"{rate:.1f}s/trace eta={(payload['total'] - payload['done']) * rate / 60:.1f}min "
          f"failures={payload['failures']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 题（自检用）")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--registration", type=Path, default=REGISTRATION)
    ap.add_argument("--pace-budget-hours", type=float, default=4.0)
    ap.add_argument("--pace-check-after", type=int, default=20)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze = require_v3_frozen(ROOT / "docs/paper/preregistration_v3.md",
                               ROOT / "docs/paper/preregistration_v3.lock")
    print(f"[G2   ] match={freeze.match} sha={freeze.current_sha256[:12]}…")

    # ── 登记：预算与判据都必须是**事前**写下的，本脚本只读取，不生成 ──────
    reg_path = Path(args.registration)
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    budget = int(reg["registered_max_new_tokens"])
    crit = reg["budget_criteria"]
    if crit.get("declared_before_run") is not True:
        raise SystemExit("[stop ] 登记文件未标 declared_before_run=true")
    thr = float(crit["max_touched_max_new_tokens_rate"])
    if thr != G4Criteria().max_touched_max_new_tokens_rate:
        raise SystemExit("[stop ] 登记的截断率阈值与 G4Criteria 的冻结值不一致 —— "
                         "本轮沿用既有判据，不新设阈值")
    if reg["frozen_max_new_tokens"] != cfg.mcq_max_new_tokens:
        raise SystemExit("[stop ] 登记的 frozen_max_new_tokens 与 v3_config 不一致")
    print(f"[reg  ] budget={budget}（冻结值 {cfg.mcq_max_new_tokens}）  截断率阈值={thr}  "
          f"sha={_sha256_file(reg_path)[:12]}…")

    # ── 校准题单：只读已核验的名单，本脚本不重新选题 ──────────────────────
    list_path = out_dir / "mcq_budget_check_questions.jsonl"
    list_meta = json.loads((out_dir / "mcq_budget_check_questions_meta.json")
                           .read_text(encoding="utf-8"))
    if _sha256_file(list_path) != list_meta["list_sha256"]:
        raise SystemExit("[stop ] 校准题单的 sha256 与其 meta 不符")
    if list_meta["list_sha256"] != reg["calibration_set"]["list_sha256"]:
        raise SystemExit("[stop ] 校准题单与登记文件记录的不是同一份")
    gate_rows = [json.loads(x) for x in
                 list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.smoke:
        gate_rows = gate_rows[:args.smoke]
    print(f"[list ] {len(gate_rows)} 题  chain={list_meta['chain_sha256'][:12]}…")

    # ── 后端与记录 ───────────────────────────────────────────────────────
    backend = bk.MCQBackend(MODEL, seed=cfg.generation_seed, max_new_tokens=budget,
                            temperature=0.7, top_p=0.95)
    tok = backend.ctx["tokenizer"]
    geometry = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=MODEL)
    print(f"[geom ] block={geometry.block_index} tuple={geometry.tuple_index} "
          f"dim={geometry.hidden_dim} source={geometry.source}")

    records = g3.build_arm_records(g3.ARM_MCQ, gate_rows, cfg=cfg, tokenizer=tok)
    recorder = _GenerationRecorder(backend.ctx)
    obs: dict[str, dict] = {}

    def after_unit(unit, row):
        if recorder.last is None:
            raise ValueError(f"{unit['unit_id']}: 未捕获到生成记录")
        cap = recorder.last
        recorder.last = None
        eos = tok.eos_token_id
        cap["reached_max_new_tokens"] = cap["n_generated_tokens"] >= budget
        cap["hit_passed_eos"] = (cap["last_generated_token_id"] is not None
                                 and cap["last_generated_token_id"]
                                 == (eos if isinstance(eos, int) else None))
        cap["other_eos_ids_present_in_output"] = sorted(
            {i for i in cap["generated_ids_raw"] if i in (128001, 128008, 128009)})
        cap["decoded_keep_special_tokens"] = tok.decode(cap["generated_ids_raw"],
                                                        skip_special_tokens=False)
        obs[unit["unit_id"]] = cap
        row["num_generated_tokens"] = cap["n_generated_tokens"]
        row["reached_max_new_tokens"] = cap["reached_max_new_tokens"]

    print(f"[mode ] REAL  model={MODEL}  records={len(records)} "
          f"units={len(records) * cfg.k_traces}")
    t0 = time.time()
    try:
        run = g3.run_arm(arm=g3.ARM_MCQ, records=records, backend=backend,
                         out_dir=out_dir, cfg=cfg, progress=_progress,
                         progress_every=args.pace_check_after,
                         pace_check_after=args.pace_check_after,
                         pace_budget_hours=args.pace_budget_hours,
                         gate=GATE_TAG, after_unit=after_unit)
    except g3.PaceBudgetExceeded as exc:
        print(f"[STOP ] {exc}", file=sys.stderr)
        return 3
    finally:
        recorder.restore()
    elapsed = time.time() - t0

    write_atomic(out_dir / "mcq_budget_check_generation_observations.jsonl",
                 "\n".join(json.dumps({"unit_id": k, **v}, ensure_ascii=False)
                           for k, v in obs.items()) + "\n")

    # ── 判据（只应用登记里事前写下的三条）────────────────────────────────
    rows = run.rows
    n, n_expected = len(rows), len(records) * cfg.k_traces
    non_finite = sum(int(np.count_nonzero(~np.isfinite(v))) for v in run.hidden.values())
    touched = [r for r in rows if r.get("reached_max_new_tokens")]
    missing_count = [r for r in rows if not isinstance(r.get("num_generated_tokens"), int)]
    rate = round(len(touched) / n, 4) if n else 1.0
    checks = {
        "every_unit_produced_a_row": n == n_expected,
        "no_unhandled_exception": not run.failures,
        "no_non_finite_values": non_finite == 0,
        "exact_token_count_recorded": not missing_count,
        "touched_max_new_tokens_rate": rate,
        "touched_max_new_tokens_rate_ok": rate <= thr,
    }
    boolean = {k: v for k, v in checks.items() if isinstance(v, bool)}
    passed = all(boolean.values())

    # ── 只作报告、不参与选值的观察量 ─────────────────────────────────────
    lens = sorted(r["num_generated_tokens"] for r in rows
                  if isinstance(r.get("num_generated_tokens"), int))

    def pct(p: float):
        return lens[min(len(lens) - 1, int(p * (len(lens) - 1)))] if lens else None

    observed_only = {
        "_not_used_for_budget_decision": (
            "以下全部**只作报告**：解析口径、eligibility 与正确率都不参与预算判定"),
        "n_generated_tokens": {"min": lens[0] if lens else None, "p50": pct(0.5),
                               "p90": pct(0.9), "p99": pct(0.99),
                               "max": lens[-1] if lens else None,
                               "mean": round(sum(lens) / len(lens), 1) if lens else None},
        "hit_passed_eos": sum(1 for v in obs.values() if v["hit_passed_eos"]),
        "parse_status_counts": {s: sum(1 for r in rows if r.get("parse_status") == s)
                                for s in sorted({r.get("parse_status") for r in rows})},
        "n_eligible_for_primary": sum(1 for r in rows if r.get("eligible_for_primary")),
        "by_sample_type": {
            st: {"n": sum(1 for r in rows if r.get("sample_type") == st),
                 "touched": sum(1 for r in rows if r.get("sample_type") == st
                                and r.get("reached_max_new_tokens")),
                 "p50_tokens": None}
            for st in ("greedy", "sampled")},
    }
    for st in ("greedy", "sampled"):
        s = sorted(r["num_generated_tokens"] for r in rows
                   if r.get("sample_type") == st
                   and isinstance(r.get("num_generated_tokens"), int))
        observed_only["by_sample_type"][st]["p50_tokens"] = (
            s[len(s) // 2] if s else None)

    report = {
        "schema": "stage1_mcq_budget_check_report_v1",
        "date": "2026-09-14",
        "card": "docs/codex_tasks/sprint_stage1_llama_budget512_check_and_g3.md",
        "what_this_is": ("生成长度检查。**不是 gate、不是 G3、不是复现结论。**"
                         "通过仅表示本批输出在该预算下满足既有截断率要求，"
                         "**不**证明研究方法在 Llama 上复现成功，"
                         "**不**表示 512 是最小可用预算。"),
        "registration": {"path": _rel(reg_path),
                         "sha256": _sha256_file(reg_path),
                         "registered_max_new_tokens": budget,
                         "frozen_max_new_tokens": cfg.mcq_max_new_tokens},
        "calibration_set": {"path": _rel(list_path),
                            "sha256": list_meta["list_sha256"],
                            "chain_sha256": list_meta["chain_sha256"],
                            "n_questions": len(gate_rows),
                            "disjoint_from": list_meta["verification"]["disjoint_from"]},
        "prereg": freeze.as_dict(),
        "environment": {**VERSIONS, "backend": dict(bk.BACKEND_8BIT),
                        "model_path": MODEL, "interpreter": sys.executable},
        "geometry": {"rel_depth": geometry.rel_depth, "block": geometry.block_index,
                     "tuple": geometry.tuple_index, "hidden_dim": geometry.hidden_dim,
                     "source": geometry.source},
        "fingerprint": run.fingerprint,
        "hidden_verification": run.hidden_verification,
        "criteria_declared_before_run": crit,
        "checks": checks,
        "failed_checks": sorted(k for k, v in boolean.items() if not v),
        "passed": passed,
        "n_rows": n, "n_expected": n_expected,
        "n_touched_max_new_tokens": len(touched),
        "n_non_finite_values": non_finite,
        "failures": run.failures,
        "elapsed_seconds": round(elapsed, 1),
        "s_per_trace": round(elapsed / n, 2) if n else None,
        "pace": run.pace,
        "observed_only": observed_only,
        "smoke": args.smoke or None,
        "claim_limits": reg["claim_limits"],
    }
    report_path = out_dir / "mcq_budget_check_report.json"
    write_atomic(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"\n[budget] touched={len(touched)}/{n} rate={rate} <= {thr} ? "
          f"{checks['touched_max_new_tokens_rate_ok']}")
    print(f"[budget] passed={passed}  failed={report['failed_checks']}")
    print(f"[obs  ] tokens min/p50/p90/max="
          f"{observed_only['n_generated_tokens']['min']}/"
          f"{observed_only['n_generated_tokens']['p50']}/"
          f"{observed_only['n_generated_tokens']['p90']}/"
          f"{observed_only['n_generated_tokens']['max']}  "
          f"eos={observed_only['hit_passed_eos']}  "
          f"eligible={observed_only['n_eligible_for_primary']}（只作报告）")
    print(f"[time ] {elapsed:.0f}s  {report['s_per_trace']}s/trace")
    print(f"[out  ] {report_path}")
    return 0 if passed else 1


VERSIONS = versions()

if __name__ == "__main__":
    raise SystemExit(main())
