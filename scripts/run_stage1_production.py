"""Stage 1 正式生成入口（v3.1）。分臂、分段、可安全续跑。

PowerShell 下统一用 `-X utf8`，不要用 Bash 风格的 `VAR=值 命令` 前缀：

    python.exe -X utf8 scripts/run_stage1_production.py --arm h1 --dry-run --dry-run-prompts 4 --max-new-units 12
    python.exe -X utf8 scripts/run_stage1_production.py --arm h1 --max-new-units 500
    python.exe -X utf8 scripts/run_stage1_production.py --arm h1 --finalize

启动前强制 G2 / G3 / G4 / G5 与指纹一致；任一不过即停，**不创建任何文件**。
分段只在 unit 边界停止。**不因 OOM 或速率问题更改模型、量化、样本、seed、阈值或生成参数。**
只输出运行质量；**不计算** AUROC / AUPRC / Δ / CI / ladder / probe。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool
from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation.manifest import WorkManifest, write_atomic  # noqa: E402


def _dry_backend(arm: str, cfg: V3AnalysisConfig, geometry=None, *, max_new_tokens: int):
    """假后端：schema 完整、内容合成，绝不加载模型。仅 --dry-run 可用。"""
    from types import SimpleNamespace

    geometry = geometry or repool.QWEN_GEOMETRY

    import numpy as np

    max_new = int(max_new_tokens)

    class Fake:
        def run_fingerprint(self, a):
            return {"schema": "stage0_fingerprint_v1", "arm": a,
                    "model": {"model_path": "<dry-run>", "backend": dict(g3.gb.BACKEND_8BIT),
                              "model_hash": "dry" * 16},
                    "tokenizer": {"tokenizer_hash": "dry" * 16},
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
            row = {"example_id": record["example_id"],
                   "prompt_hash": f"dry_{unit['unit_id']}",
                   "unit_hash": f"dry_{unit['unit_id']}u",
                   "seed": cfg.generation_seed, "sample_index": i,
                   "num_generated_tokens": 12,
                   "hidden": rng.normal(0, 1, geometry.hidden_dim).astype(np.float32)}
            if arm == prod.ARM_H1:
                row.update({"completion": f"The identifier is T10{i}9.",
                            "family": record.get("family"), "route": "recall",
                            "refusal": False, "label": i % 2,
                            # mention 必须带 family + label，否则 ci._is_primary 判 False，
                            # attach_cross_sample_f5 拿不到 primary id → 跨样本两列为 None。
                            "mentions": [{"normalized": f"T10{i}9", "family": "attack",
                                          "label": "grounded" if i % 2 else "fabricated",
                                          "start": 0, "end": 5, "emitted": True,
                                          "token_indices": [0]}],
                            "id_token_positions_local": [0], "hidden_pooled_n_tokens": 1,
                            **{k: float(rng.normal()) for k in prod.H1_FINALIZED_F5
                               if k not in ("f5_self_consistency_exact",
                                            "f5_id_agreement_rate")}})
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

    common = dict(seed=cfg.generation_seed, temperature=0.7, top_p=0.95)
    if arm == prod.ARM_H1:
        index = build_ontology_index(Path(args.ontology_dir))
        return gb.H1Backend(args.model_path, index,
                            max_new_tokens=max_new_tokens, **common)
    return gb.MCQBackend(args.model_path, max_new_tokens=max_new_tokens, **common)


def _progress(payload: dict) -> None:
    rate = payload["elapsed_s"] / max(payload["done"], 1)
    print(f"[{payload['arm']}] {payload['done']}/{payload['total']} "
          f"elapsed={payload['elapsed_s']:.0f}s {rate:.1f}s/unit "
          f"failures={payload['failures']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(prod.ARMS), required=True)
    ap.add_argument("--dry-run", action="store_true", help="假后端；不加载模型")
    ap.add_argument("--dry-run-prompts", type=int, default=None,
                    help="**仅 --dry-run 可用**：把 population 截成 N 个 prompt，"
                         "以便两段之后能合法 finalize")
    ap.add_argument("--finalize", action="store_true",
                    help="一臂完整跑完后，从全部 raw 行确定性重建 finalized 产物")
    ap.add_argument("--max-new-units", type=int, default=None)
    ap.add_argument("--max-wall-seconds", type=float, default=None)
    ap.add_argument("--samples-dir", type=Path, default=Path("outputs/stage1/samples"))
    ap.add_argument("--g3-dir", type=Path, default=Path("outputs/stage1/g3"))
    ap.add_argument("--g4-dir", type=Path, default=Path("outputs/stage1/g4"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage1/production"))
    ap.add_argument("--prereg", type=Path, default=Path("docs/paper/preregistration_v3.md"))
    ap.add_argument("--lock", type=Path, default=Path("docs/paper/preregistration_v3.lock"))
    ap.add_argument("--ontology-dir", type=Path, default=Path("data/raw/ontology"))
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--budget-registration", type=Path, default=None,
                    help="已登记的分臂解码预算。**只要本臂的 max_new_tokens 与 "
                         "v3_config 的冻结值不同就必须给**，否则输入校验会拒绝。")
    ap.add_argument("--population-role", default=prod.ROLE_CONFIRMATORY,
                    choices=list(prod.ALLOWED_POPULATION_ROLES),
                    help="写进 finalized 记录与报告的角色。默认 confirmatory；"
                         "跨模型复现必须显式传 cross_model_replication_non_confirmatory")
    repool.add_geometry_cli_args(ap)
    args = ap.parse_args()

    if args.dry_run_prompts is not None and not args.dry_run:
        print("[refuse] --dry-run-prompts 只允许与 --dry-run 同用：正式模式不得截断 population",
              file=sys.stderr)
        return 2

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    geometry = repool.geometry_from_cli(args, model_path=args.model_path)
    print(f"[geom ] L={geometry.num_layers} block={geometry.block_index} "
          f"tuple={geometry.tuple_index} dim={geometry.hidden_dim} "
          f"source={geometry.source}")
    print(f"[role ] population_role={args.population_role}")

    # 已登记的分臂解码预算：未给即取冻结值，行为与引入登记机制之前逐字相同。
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
              f"sha={budget_registration.sha256[:12]}…")
    # 生成、指纹与校验用**同一个数**。
    arm_budget = g3.expected_max_new_tokens(args.arm, cfg, budget_registration)

    # ── 解析口径闸：MCQ 侧不得把**旧解析**的 gate 证据接进新链 ──────────────
    # 只在走扩展预算的新链上强制（budget_registration 非空）。旧 Qwen 的 gate 报告
    # 产出于 evidence_sources 这个键存在之前，无条件强制会改变旧路径行为。
    if budget_registration is not None and args.arm == prod.ARM_MCQ:
        import hashlib as _hl
        want = _hl.sha256(
            (_REPO_ROOT / "src/recover_attention/domain_label_proxy.py").read_bytes()
        ).hexdigest()
        for gate_dir, name in ((args.g3_dir, "mcq_g3_smoke_report.json"),
                               (args.g4_dir, "mcq_g4_quality_report.json")):
            rep = json.loads((gate_dir / name).read_text(encoding="utf-8"))
            got = ((rep.get("evidence_sources") or {}).get("current_parsing") or {}
                   ).get("parser_sha256")
            if got != want:
                print(f"[stop ] {gate_dir / name} 记录的解析实现 {got} != 当前 {want} —— "
                      f"旧解析的 MCQ gate 证据不得接进新链（口径不同，产物不可混用）",
                      file=sys.stderr)
                return 2
        print(f"[parse] MCQ gate 证据的解析实现与当前一致 sha={want[:12]}…")

    verification = prod.verify_production_inputs(
        samples_dir=args.samples_dir, g3_dir=args.g3_dir, g4_dir=args.g4_dir,
        prereg=args.prereg, lock=args.lock, cfg=cfg, arm=args.arm)
    g2 = verification["g2"]
    print(f"[G2   ] match={g2['match']} version={g2['locked_version']} "
          f"hash_mode={g2['hash_mode']}")
    print(f"[G3/G4] passed=True/True arms={verification['g4_arms_passed']}")
    print(f"[G5   ] n_primary={verification['n_primary']} "
          f"chain={verification['primary_chain_sha256'][:12]}… "
          f"uniqueness={verification['uniqueness']} disjoint_from_gates=True")

    primary_rows = verification.pop("_primary_rows")

    # paths_for 只拼路径、无副作用，上移到加载模型之前，好让一致性门在 backend 之前跑。
    paths = prod.paths_for(args.out_dir, args.arm)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # **断电中间态的门**：diagnostic 已落盘但 manifest 事件没跟上时，绝不加载模型。
    # unknown unit 这里用 manifest 现状的 unit 全集核；run_arm 内再用 units_for 复核一次。
    try:
        known = list(WorkManifest(paths["manifest"]).current_status())
        diag = g3.require_diagnostic_manifest_consistency(
            arm=args.arm, diagnostic_path=paths["diagnostics"],
            manifest_path=paths["manifest"], known_units=known or None)
    except g3.DiagnosticRecoveryRequired as exc:
        print(f"[STOP ] {exc}", file=sys.stderr)
        return 1
    if diag["diagnostic_units"]:
        print(f"[diag ] forward-nonfinite 诊断 {diag['diagnostic_units']} 条，"
              f"与 manifest 终态一致")

    if args.dry_run:
        backend, tokenizer = _dry_backend(args.arm, cfg, geometry=geometry,
                                          max_new_tokens=arm_budget)
        print("[mode ] DRY-RUN（假后端，不加载模型）")
    else:
        backend = _real_backend(args.arm, args, cfg, arm_budget)
        tokenizer = backend.ctx["tokenizer"]
        print(f"[mode ] REAL model={args.model_path}")

    records = g3.build_arm_records(args.arm, primary_rows, cfg=cfg, tokenizer=tokenizer)
    if args.dry_run_prompts is not None:
        records = records[:args.dry_run_prompts]
        print(f"[dry  ] population 截为 {len(records)} 个 prompt（仅 dry-run）")

    fingerprint = backend.run_fingerprint(args.arm)
    fp_check = prod.check_fingerprint(args.arm, fingerprint,
                                      verification["g3_production_fingerprint"], cfg=cfg,
                                      budget_registration=budget_registration)
    if not fp_check["ok"] and not args.dry_run:
        print(f"[STOP ] 指纹与 G3 生产指纹不一致：{fp_check['mismatched_keys']}", file=sys.stderr)
        return 1
    bound = prod.bind_run_fingerprint(paths["fingerprint"], fingerprint, args.arm, cfg=cfg,
                                      budget_registration=budget_registration)
    print(f"[fp   ] first_segment={bound['created']} matches_previous={bound['matches_previous_segment']}")

    audit = prod.audit_resume_state(args.arm, args.out_dir)
    print(f"[resume] done={audit.n_done} rows={audit.n_rows} hidden={audit.n_hidden_files} "
          f"repaired_done={len(audit.repaired_done)}")

    if args.finalize:
        report = prod.finalize_arm(args.arm, args.out_dir, records, cfg=cfg,
                                   run_fingerprint=fingerprint,
                                   population_role=args.population_role)
        print(f"[final] records={report['n_finalized_records']} "
              f"analyzable={report['n_analyzable_records']} "
              f"hidden={report['n_hidden_vectors']} key={report['hidden_alignment_key']}")
        print(f"[out  ] {paths['records']}")
        return 0

    started = time.time()
    run = g3.run_arm(arm=args.arm, records=records, backend=backend,
                     out_dir=args.out_dir, cfg=cfg, progress=_progress,
                     progress_every=20, durable_fsync=True, stop_on_failure=True,
                     max_new_units=args.max_new_units,
                     max_wall_seconds=args.max_wall_seconds,
                     results_path=paths["raw_results"],
                     hidden_dir=paths["hidden_dir"],
                     manifest_path=paths["manifest"],
                     diagnostic_path=paths["diagnostics"])

    segment = {
        "arm": args.arm, "dry_run": bool(args.dry_run),
        "n_units_total": run.n_units,
        "n_generated_this_segment": run.n_generated_this_process,
        "elapsed_seconds_this_segment": round(run.seconds, 1),
        "seconds_per_generated_unit": (round(run.seconds / run.n_generated_this_process, 2)
                                       if run.n_generated_this_process else None),
        "stopped_reason": run.stopped_reason or "completed",
        "n_failures": len(run.failures),
        "wall_seconds_including_setup": round(time.time() - started, 1),
    }
    with open(paths["segment_log"], "a", encoding="utf-8") as handle:   # append-only 历史
        handle.write(json.dumps(segment, ensure_ascii=False) + "\n")
    write_atomic(paths["progress"], json.dumps({                        # 当前快照
        "schema_version": "stage1_production_progress_v1",
        "performance_blind": True, "did_not_compute": list(prod.DID_NOT_COMPUTE),
        "latest_segment": segment,
        "note": "本文件是当前快照；完整分段历史见 *_segment_log.jsonl，不得以快照代替历史",
    }, ensure_ascii=False, indent=2))

    print(f"[seg  ] generated={segment['n_generated_this_segment']} "
          f"reason={segment['stopped_reason']} "
          f"rate={segment['seconds_per_generated_unit']}s/unit "
          f"failures={segment['n_failures']}")
    print(f"[out  ] {paths['raw_results']}")

    # 正式生成不容忍静默失败：失败即非零退出，供调用方/脚本据此停止推进。
    failed_stop = str(segment["stopped_reason"]).startswith("unit_failure:")
    if failed_stop or segment["n_failures"]:
        print(f"[STOP ] 本段存在失败 unit（reason={segment['stopped_reason']}，"
              f"n_failures={segment['n_failures']}）；已在 unit 边界停止，未尝试后续 unit。"
              f" 不自动重试、不修改任何冻结参数。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
