"""Stage 1 的 G4 性能盲质量检查（v3.1）。每次只跑一臂。

    python scripts/run_stage1_g4_quality.py --arm h1 --dry-run
    python scripts/run_stage1_g4_quality.py --arm h1
    python scripts/run_stage1_g4_quality.py --arm mcq
    python scripts/run_stage1_g4_quality.py --summarise

判据写死在 `v3_g4_quality.CRITERIA`。任一冻结门槛失败 → `passed=false` 并停止；
**不得**换 G4 样本、调整阈值或修改生成参数后重跑。

精确 token 数：用记录型代理包住 `ctx["model"].generate`，取
`ids.shape[-1] - input_ids.shape[-1]`。零重复实现、不改 Stage 0 一行，
也不依赖 decode→re-encode 的往返一致性。**不使用字符长度近似。**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation import v3_g4_quality as g4  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation.manifest import write_atomic  # noqa: E402


class _TokenCountRecorder:
    """包住 model.generate，记录**真实**新生成 token 数。不改 Stage 0 一行。"""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.last: int | None = None
        self._orig = ctx["model"].generate

        def recording(**kw):
            ids = self._orig(**kw)
            self.last = int(ids.shape[-1] - kw["input_ids"].shape[-1])
            return ids

        ctx["model"].generate = recording

    def restore(self) -> None:
        self.ctx["model"].generate = self._orig


def _dry_backend(arm: str, cfg: V3AnalysisConfig, geometry=None, *, max_new_tokens: int):
    """dry-run 假后端：schema 完整、内容合成，绝不加载模型。"""
    from types import SimpleNamespace

    geometry = geometry or repool.QWEN_GEOMETRY

    max_new = int(max_new_tokens)

    class Fake:
        def run_fingerprint(self, a):
            return {"schema": "stage0_fingerprint_v1", "arm": a,
                    "model": {"model_path": "<dry-run>",
                              "num_hidden_layers": geometry.num_layers,
                              "hidden_size": geometry.hidden_dim,
                              "backend": dict(g3.gb.BACKEND_8BIT), "model_hash": "dry" * 16},
                    "tokenizer": {"class": "Qwen2Tokenizer", "tokenizer_hash": "dry" * 16},
                    "hidden_index": {"block_index_zero_based": geometry.block_index,
                                     "hidden_states_tuple_index": geometry.tuple_index},
                    "generation_params": {"K": cfg.k_traces, "max_new_tokens": max_new,
                                          "temperature": 0.7, "top_p": 0.95,
                                          "top_k": g3.gb._RealBackendBase.QWEN_PROTOCOL_TOP_K,
                                          "repetition_penalty":
                                              g3.gb._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY,
                                          "seed": cfg.generation_seed,
                                          "schema": g3.GENERATION_PARAMS_SCHEMA,
                                          "sampling_params_source": "dry-run stub"},
                    "run_hash": f"dryrun_{a}"}

        def verify_hidden_index(self):
            return {"block_index_zero_based": geometry.block_index,
                    "hidden_states_tuple_index": geometry.tuple_index,
                    "numeric_match": True, "max_abs_diff": 0.0}

        def run_unit(self, unit, record):
            rng = np.random.default_rng(abs(hash(unit["unit_id"])) % (2 ** 31))
            i = unit["sample_index"]
            row = {"example_id": record["example_id"], "group_id": record.get("group_id", "g"),
                   "prompt_hash": f"dry_{unit['unit_id']}", "unit_hash": f"dry_{unit['unit_id']}u",
                   "seed": cfg.generation_seed, "sample_index": i,
                   "hidden": rng.normal(0, 1, geometry.hidden_dim).astype(np.float32),
                   "num_generated_tokens": 12}
            if arm == g4.ARM_H1:
                row.update({"completion": "The identifier is T1059.",
                            "family": record.get("family"), "route": "recall", "refusal": False,
                            "mentions": [{"normalized": "T1059", "start": 0, "end": 5,
                                          "emitted": True, "token_indices": [0]}],
                            "id_token_positions_local": [0], "hidden_pooled_n_tokens": 1,
                            "label": "grounded" if i % 2 else "fabricated",
                            "f5_id_mean_logprob": float(rng.normal()),
                            "f5_id_min_logprob": float(rng.normal()),
                            "f5_seq_mean_logprob": float(rng.normal())})
            else:
                row.update({"raw_completion": "A", "parsed_label": "A",
                            "eligible_for_primary": True, "wrong_label": i % 2,
                            "readout_to_prompt_end_distance": 0,
                            "f5_label_margin": float(rng.normal()),
                            "f5_label_entropy": float(abs(rng.normal())),
                            "f5_full_entropy": float(abs(rng.normal())),
                            "f5_letter_token_logprob": float(-abs(rng.normal()))})
            return row

    return Fake(), SimpleNamespace(
        apply_chat_template=lambda msgs, **kw: json.dumps(msgs, ensure_ascii=False))


def _real_backend(arm: str, args, cfg: V3AnalysisConfig, max_new_tokens: int):
    from recover_attention.generation import backends as gb
    from recover_attention.h1_identifier import build_ontology_index

    # 预算由调用方算好后显式传入 —— 生成、指纹与判据必须用**同一个数**，
    # 这里再从 cfg 取一次就会在有登记时悄悄跑回冻结值。
    common = dict(seed=cfg.generation_seed, temperature=0.7, top_p=0.95)
    if arm == g4.ARM_H1:
        index = build_ontology_index(Path(args.ontology_dir))
        return gb.H1Backend(args.model_path, index,
                            max_new_tokens=max_new_tokens, **common)
    return gb.MCQBackend(args.model_path, max_new_tokens=max_new_tokens, **common)


def _progress(payload: dict) -> None:
    rate = payload["elapsed_s"] / max(payload["done"], 1)
    print(f"[{payload['arm']}] {payload['done']}/{payload['total']} "
          f"elapsed={payload['elapsed_s']:.0f}s {rate:.1f}s/trace "
          f"eta={(payload['total'] - payload['done']) * rate / 60:.1f}min "
          f"failures={payload['failures']}", flush=True)


def _load_g3_fingerprint(g3_dir: Path, arm: str,
                         budget_registration=None) -> tuple[dict, dict]:
    report = json.loads((g3_dir / f"{arm}_g3_smoke_report.json").read_text(encoding="utf-8"))
    summary = json.loads((g3_dir / "g3_summary.json").read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise SystemExit("G3 尚未通过，不得进入 G4")
    if budget_registration is not None and not summary.get("real_g3_passed"):
        # 只在走扩展预算的新链上强制。旧 Qwen 的汇总产出于这个键存在之前（值为 None），
        # 无条件强制等于改变旧路径行为 —— 那不是本轮的范围。
        raise SystemExit("G3 汇总不是真实运行的通过（real_g3_passed != true），不得进入 G4")
    # 上游证据必须与本次登记同源：拿一份登记去认领另一批 G3 产物 = 新旧证据混用。
    summary_reg = (summary.get("budget_registration") or {}).get("sha256")
    mine = budget_registration.sha256 if budget_registration else None
    if summary_reg != mine:
        raise SystemExit(
            f"G3 汇总记录的预算登记（{summary_reg}）与本次 G4 传入的（{mine}）不是同一份 —— "
            f"拒绝把上游证据接到不同预算的运行上")
    provenance = {
        "g3_summary_passed": summary["passed"],
        "g3_h1_eligible_rate_note": summary.get("h1_eligible_rate_note"),
        "g3_h1_eligible_margin": 0.0,
        "g3_margin_note": ("G3 的 H1 eligible_margin = 0.000（恰好达标、零余量）。"
                           "作为 provenance 同报，**不得据此提高门槛、换样本或重跑**。"),
        "g3_production_fingerprint_frozen": summary.get(
            "production_fingerprint_is_now_frozen"),
    }
    return report["fingerprint"], provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(g4.ARMS))
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--samples-dir", type=Path, default=Path("outputs/stage1/samples"))
    ap.add_argument("--g3-dir", type=Path, default=Path("outputs/stage1/g3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage1/g4"))
    ap.add_argument("--prereg", type=Path, default=Path("docs/paper/preregistration_v3.md"))
    ap.add_argument("--lock", type=Path, default=Path("docs/paper/preregistration_v3.lock"))
    ap.add_argument("--ontology-dir", type=Path, default=Path("data/raw/ontology"))
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--budget-registration", type=Path, default=None,
                    help="已登记的分臂解码预算。**只要本臂的 max_new_tokens 与 "
                         "v3_config 的冻结值不同就必须给**，否则拒绝运行。")
    ap.add_argument("--pace-budget-hours", type=float, default=2.0)
    ap.add_argument("--pace-check-after", type=int, default=20)
    repool.add_geometry_cli_args(ap)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    geometry = repool.geometry_from_cli(args, model_path=args.model_path)
    criteria = g4.G4Criteria(hidden_dim=geometry.hidden_dim,
                             block_index=geometry.block_index,
                             tuple_index=geometry.tuple_index)
    print(f"[geom ] L={geometry.num_layers} block={geometry.block_index} "
          f"tuple={geometry.tuple_index} dim={geometry.hidden_dim} "
          f"source={geometry.source}")

    budget_registration = None
    if args.budget_registration is not None:
        try:
            budget_registration = g3.load_budget_registration(
                args.budget_registration, cfg=cfg, model_path=args.model_path)
        except ValueError as exc:
            print(f"[stop ] {exc}", file=sys.stderr)
            return 2
        print(f"[budget] {budget_registration.arm} 登记预算 "
              f"{budget_registration.registered_max_new_tokens}"
              f"（冻结值 {budget_registration.frozen_max_new_tokens}）  "
              f"sha={budget_registration.sha256[:12]}…  "
              f"长度检查 passed={budget_registration.length_check.get('passed')}")

    if args.summarise:
        return _summarise(args)
    if not args.arm:
        ap.error("需要 --arm 或 --summarise")

    # 本臂实际预算：未登记取冻结值，已登记取登记值。生成、指纹与判据**用同一个数**。
    arm_budget = g3.expected_max_new_tokens(args.arm, cfg, budget_registration)
    if arm_budget != g3.expected_max_new_tokens(args.arm, cfg):
        print(f"[budget] {args.arm} 本次实际 max_new_tokens={arm_budget}"
              f"（冻结值 {g3.expected_max_new_tokens(args.arm, cfg)}）")

    verification = g3.verify_inputs(samples_dir=args.samples_dir, prereg=args.prereg,
                                    lock=args.lock, cfg=cfg, gate="G4")
    g2 = verification["g2"]
    print(f"[G2   ] match={g2['match']} version={g2['locked_version']} "
          f"hash_mode={g2['hash_mode']} sha={g2['current_sha256'][:12]}…")
    print(f"[lists] manifest_sha={verification['sample_manifest_sha256'][:12]}… "
          f"chain={verification['n_lists_verified']}/6 "
          f"g4 counts={verification['gate_counts']} "
          f"primary_read={verification['primary_lists_read']}")

    g3_fp, g3_prov = _load_g3_fingerprint(args.g3_dir, args.arm,
                                          budget_registration)
    print(f"[G3   ] production fingerprint loaded; H1 eligible_margin(G3)="
          f"{g3_prov['g3_h1_eligible_margin']}")

    gate_rows = verification.pop("_gate_rows")[args.arm]

    recorder = None
    if args.dry_run:
        backend, tokenizer = _dry_backend(args.arm, cfg, geometry=geometry,
                                          max_new_tokens=arm_budget)
        print("[mode ] DRY-RUN（假后端，不加载模型）")
    else:
        backend = _real_backend(args.arm, args, cfg, arm_budget)
        tokenizer = backend.ctx["tokenizer"]
        recorder = _TokenCountRecorder(backend.ctx)
        print(f"[mode ] REAL model={args.model_path}")

    records = g3.build_arm_records(args.arm, gate_rows, cfg=cfg, tokenizer=tokenizer)

    def after_unit(unit, row):
        if recorder is not None:
            if recorder.last is None:
                raise ValueError(f"{unit['unit_id']}: 未捕获到生成 token 数")
            row["num_generated_tokens"] = recorder.last
            recorder.last = None

    try:
        run = g3.run_arm(arm=args.arm, records=records, backend=backend,
                         out_dir=args.out_dir, cfg=cfg, progress=_progress,
                         progress_every=args.pace_check_after,
                         pace_check_after=args.pace_check_after,
                         pace_budget_hours=args.pace_budget_hours,
                         gate="g4", after_unit=after_unit)
    except g3.PaceBudgetExceeded as exc:
        print(f"[STOP ] {exc}", file=sys.stderr)
        return 3
    finally:
        if recorder is not None:
            recorder.restore()

    # group_id 回填：新跑的行已由 run_arm 挂载；旧行按冻结名单在内存回填，不改写落盘文件。
    n_filled = g4.backfill_group_id(run, records)
    if n_filled:
        print(f"[group] 从冻结名单回填 group_id {n_filled} 行（仅内存，未改写 unit_results）")

    evaluation = g4.check_arm(args.arm, run, cfg=cfg, criteria=criteria,
                              budget_registration=budget_registration)
    fp_check = g4.matches_g3_production_fingerprint(
        args.arm, run.fingerprint, g3_fp, cfg=cfg,
        budget_registration=budget_registration)
    report = g4.build_report(arm=args.arm, run=run, verification=verification,
                             criteria=criteria,
                             evaluation=evaluation, fingerprint_check=fp_check,
                             cfg=cfg, g3_provenance=g3_prov)
    report["dry_run"] = bool(args.dry_run)
    out = args.out_dir / f"{args.arm}_g4_quality_report.json"
    g4.write_report(out, report)

    degen = evaluation["stats"]["degeneration"]
    print(f"[degen] raw={degen['raw_pattern_degeneration_rate']} "
          f"exempt={degen['structured_list_false_positive_rate']} "
          f"adjusted={degen['adjusted_degeneration_rate']} (gate) "
          f"touched={degen['touched_max_new_tokens_rate']}")
    print(f"[margin] {json.dumps(report['margins'], ensure_ascii=False)}")
    print(f"[G4   ] arm={args.arm} passed={report['passed']}")
    if report["failed_checks"] or not fp_check["ok"]:
        print(f"[FAIL ] checks={report['failed_checks']} fp={fp_check['mismatched_keys']}",
              file=sys.stderr)
    print(f"[out  ] {out}")
    return 0 if report["passed"] else 1


def _shared_criteria(reports: dict) -> dict:
    """汇总只报**两臂实际使用**的判据，且两臂必须完全一致。

    先前这里读的是模块默认 `g4.CRITERIA` —— 跨模型运行时它与实际用的判据不同，
    汇总会声称用了 Qwen 的几何。两臂判据不一致也必须当场停：那意味着两臂
    跑在不同几何上，backend invariant 已经破了。
    """
    seen = {arm: r.get("criteria") for arm, r in reports.items()}
    missing = sorted(a for a, c in seen.items() if not c)
    if missing:
        raise SystemExit(f"[stop ] 以下臂的报告缺 criteria：{missing}")
    arms = sorted(seen)
    first = seen[arms[0]]
    for arm in arms[1:]:
        if seen[arm] != first:
            diff = sorted(k for k in set(first) | set(seen[arm])
                          if first.get(k) != seen[arm].get(k))
            raise SystemExit(
                f"[stop ] 两臂 G4 判据不一致，字段 {diff} —— 拒绝汇总")
    return first


def _summarise(args) -> int:
    reports = {}
    for arm in g4.ARMS:
        path = args.out_dir / f"{arm}_g4_quality_report.json"
        if not path.exists():
            print(f"[stop ] 缺少 {path}", file=sys.stderr)
            return 2
        reports[arm] = json.loads(path.read_text(encoding="utf-8"))
    passed = all(r["passed"] for r in reports.values())
    summary = {
        "schema_version": "stage1_g4_summary_v1", "gate": "G4",
        "gate_only": True, "performance_blind": True,
        "did_not_compute": list(g4.DID_NOT_COMPUTE),
        "arms_passed": {arm: r["passed"] for arm, r in reports.items()},
        "criteria": _shared_criteria(reports),
        "degeneration": {arm: r["stats"]["degeneration"] for arm, r in reports.items()},
        "margins": {arm: r["margins"] for arm, r in reports.items()},
        "fingerprint_matches_g3_production": {
            arm: r["fingerprint_matches_g3_production"]["ok"] for arm, r in reports.items()},
        "g3_provenance": reports[g4.ARM_H1]["g3_provenance"],
        "passed": passed,
    }
    out = args.out_dir / "g4_summary.json"
    write_atomic(out, json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[G4   ] overall passed={passed} arms={summary['arms_passed']}")
    print(f"[out  ] {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
