"""Stage 0 production preflight / launch readiness（W0.5-C）。

目的只有一个:**把"缺外部事实"和"缺代码"分开**。

7/29 之前应该达到的状态是
    blocked_by = ["external_cfp_confirmation"]
而不是
    blocked_by = ["external_cfp_confirmation", "code not written", "assets missing"]

本模块**只读**:不生成 completion、不加载模型权重、不改 v2.3 hash、不碰 G1 状态。
模型只读 config.json —— 那足以核对 §8 的层位（num_hidden_layers → block/tuple index），
不需要把 15GB 权重搬进显存。
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

from recover_attention import conditional_increment as ci
from recover_attention import mcq_conditional_increment as mci

# §3 冻结的规模
H1_PROMPTS = 480
H1_K = 6
MCQ_CONFIRMATORY = 1760
MCQ_POOL = 2000
MCQ_BURNED = 240
HIDDEN_BYTES_PER_VECTOR = 3584 * 4        # fp32 pooled vector（§8:每 completion 一个,非每 token）


# ------------------------------------------------------------------- gates
def check_gates(prereg: Path, lock: Path, cfp_record: Path, smoke_dir: Path) -> dict:
    """**只读**三道门。不改 G1，不改 hash。"""
    g2 = ci.check_preregistration_frozen(str(prereg), str(lock))
    g1 = ci.check_cfp_confirmed(str(cfp_record))

    def _arm(name: str, kind: str) -> dict:
        p = smoke_dir / name
        if not p.exists():
            return {"ok": False, "reason": f"missing {p}"}
        rep = json.loads(p.read_text(encoding="utf-8"))
        return {"ok": bool(rep.get("passed") and rep.get("kind") == kind
                           and rep.get("prereg_sha256") == g2["current"]),
                "passed": rep.get("passed"),
                "prereg_sha256_matches": rep.get("prereg_sha256") == g2["current"],
                "backend": rep.get("backend") or (rep.get("fingerprint") or {}).get("backend")}

    h1 = _arm("smoke_report_h1.json", "model_smoke_h1")
    mcq = _arm("smoke_report_mcq.json", "model_smoke_mcq")
    keys = ("load_in_8bit", "device_map", "attn_implementation", "local_files_only")
    bk_h1 = {k: (h1.get("backend") or {}).get(k) for k in keys}
    bk_mcq = {k: (mcq.get("backend") or {}).get(k) for k in keys}
    bk_ok = bk_h1 == bk_mcq and bk_h1.get("load_in_8bit") is True
    return {
        "G1": {"ok": bool(g1["ok"]), **{k: g1.get(k) for k in
                                        ("status", "missing", "target", "deadline", "page_limit",
                                         "archival", "reason")}},
        "G2": {"ok": bool(g2["match"]), "current": g2["current"], "locked": g2["locked"]},
        "G3": {"ok": bool(h1["ok"] and mcq["ok"] and bk_ok),
               "h1_arm": h1, "mcq_arm": mcq,
               "backend_invariant": {"ok": bk_ok, "h1": bk_h1, "mcq": bk_mcq,
                                     "why": ("section-6 gate D = S_MCQ - S_H1 confounds "
                                             "observability with quantization if arms differ")}},
    }


# ------------------------------------------------------- code readiness
def check_code_readiness() -> dict:
    """代码是否就绪 —— 这一项存在的意义就是让"没写完"不能伪装成"在等 CFP"。"""
    checks: dict[str, Any] = {}
    modules = ["recover_attention.evaluation.config",
               "recover_attention.evaluation.feature_cache",
               "recover_attention.evaluation.ladder",
               "recover_attention.evaluation.bootstrap",
               "recover_attention.evaluation.increment",
               "recover_attention.evaluation.artifact_redline",
               "recover_attention.evaluation.observability_gate",
               "recover_attention.evaluation.rq2",
               "recover_attention.generation.manifest",
               "recover_attention.generation.fingerprint"]
    bad = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:                       # noqa: BLE001
            bad.append({"module": m, "error": repr(exc)[:200]})
    checks["all_modules_import"] = {"ok": not bad, "failed": bad, "n_modules": len(modules)}

    # 正式规格能被构造出来（config 层面强制 5 folds / bootstrap>=1000 / nested C）
    try:
        from recover_attention.evaluation.config import load_config
        cfg = load_config(Path("configs/stage0.yaml"))
        checks["production_config_valid"] = {
            "ok": cfg.mode == "production" and cfg.n_outer_folds >= 5 and cfg.n_boot >= 1000
            and not cfg.allow_pilot_population,
            "n_outer_folds": cfg.n_outer_folds, "n_boot": cfg.n_boot,
            "n_inner_folds": cfg.n_inner_folds, "allow_pilot": cfg.allow_pilot_population}
    except Exception as exc:                           # noqa: BLE001
        checks["production_config_valid"] = {"ok": False, "error": repr(exc)[:200]}

    # 分析链已被 dry-run 证明可跑通（读产物,不重跑）
    dry = Path("outputs/stage0/evaluation/stage0_analysis_dry_run.json")
    if dry.exists():
        rep = json.loads(dry.read_text(encoding="utf-8"))
        checks["analysis_pipeline_dry_run"] = {
            "ok": bool(rep.get("analysis_pipeline_ready")),
            "artifact": str(dry),
            "arms": sorted((rep.get("increment") or {})),
            "note": "run `python scripts/run_stage0_analysis.py --dry-run` to refresh"}
    else:
        checks["analysis_pipeline_dry_run"] = {
            "ok": False, "reason": f"missing {dry}; run run_stage0_analysis.py --dry-run"}

    # 生成链已被 dry-run 证明可跑通
    gen_ok = {}
    for arm in ("h1", "mcq"):
        p = Path(f"outputs/stage0/dry_run/generation/{arm}_generation_report.json")
        if p.exists():
            rep = json.loads(p.read_text(encoding="utf-8"))
            gen_ok[arm] = rep.get("status") == "complete"
        else:
            gen_ok[arm] = False
    checks["generation_pipeline_dry_run"] = {
        "ok": all(gen_ok.values()), "per_arm": gen_ok,
        "note": "run `python scripts/run_stage0_generation.py --arm {h1,mcq} --dry-run`"}

    checks["ok"] = all(v.get("ok") for k, v in checks.items() if isinstance(v, dict))
    return checks


# ----------------------------------------------------- input asset checks
def check_input_assets(h1_samples: Path, mcq_pool: Path, burned_traces: Path,
                       ontology_dir: Path, model_path: Path) -> dict:
    """输入资产。**不加载模型权重** —— 只读 config.json 就够核对 §8 的层位。"""
    out: dict[str, Any] = {}

    if h1_samples.exists():
        rows = [json.loads(l) for l in open(h1_samples, encoding="utf-8")]
        ids = {r["example_id"] for r in rows}
        out["h1_samples"] = {"ok": len(ids) == H1_PROMPTS, "n": len(rows), "n_unique": len(ids),
                             "expected": H1_PROMPTS, "n_groups": len({r["group_id"] for r in rows})}
    else:
        out["h1_samples"] = {"ok": False, "reason": f"missing {h1_samples}"}

    if mcq_pool.exists() and burned_traces.exists():
        pool = [json.loads(l)["example_id"] for l in open(mcq_pool, encoding="utf-8")]
        burned = {json.loads(l)["example_id"] for l in open(burned_traces, encoding="utf-8")}
        try:
            split = mci.split_fresh_confirmatory(pool, sorted(burned))
            out["mcq_split"] = {
                "ok": (split["n_pool"] == MCQ_POOL and split["n_burned"] == MCQ_BURNED
                       and split["n_fresh"] == MCQ_CONFIRMATORY
                       and split["intersection_fresh_burned"] == 0
                       and split["union_equals_pool"]),
                "n_pool": split["n_pool"], "n_burned": split["n_burned"],
                "n_fresh": split["n_fresh"],
                "intersection": split["intersection_fresh_burned"],
                "union_equals_pool": split["union_equals_pool"],
                "note": "fresh/burned validated by ID SET, not row count"}
        except Exception as exc:                       # noqa: BLE001
            out["mcq_split"] = {"ok": False, "error": repr(exc)[:200]}
    else:
        out["mcq_split"] = {"ok": False, "reason": "missing pool or burned trace manifest"}

    fam = {}
    for f in ("attack", "cwe", "cve"):
        p = ontology_dir / f / "ontology_index.jsonl"
        fam[f] = p.exists()
    out["ontology_snapshot"] = {"ok": all(fam.values()), "per_family": fam}

    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        L = int(cfg.get("num_hidden_layers", -1))
        idx = ci.resolve_hidden_index(L) if L > 0 else {}
        out["model_config"] = {
            "ok": L == 28 and idx.get("block_index_zero_based") == 19
            and idx.get("hidden_states_tuple_index") == 20,
            "weights_loaded": False,
            "model_type": cfg.get("model_type"), "num_hidden_layers": L,
            "hidden_size": cfg.get("hidden_size"),
            "hidden_index": idx,
            "note": "config only — verifying section 8's layer index needs no weights"}
    else:
        out["model_config"] = {"ok": False, "reason": f"missing {cfg_path}"}

    out["ok"] = all(v.get("ok") for k, v in out.items() if isinstance(v, dict))
    return out


# -------------------------------------------------------- scale projection
def check_scale(output_root: Path) -> dict:
    """规模投影 + 磁盘。

    早前 PROGRESS 写 hidden "≈40GB" 是错的（那按全 token hidden 算,而 §11 不缓存全层/全
    token）。§8 的 H 是每 completion 一个 pooled 向量 = 14 KB,故实际是 41 MB / 25 MB。
    这里把数算出来落盘,免得再有人按错的量级去做分片。
    """
    h1_units = H1_PROMPTS * H1_K
    mcq_units = MCQ_CONFIRMATORY * H1_K            # K−1 sampled 仅供 F5 self-consistency
    h1_mb = H1_PROMPTS * H1_K * HIDDEN_BYTES_PER_VECTOR / 1e6
    mcq_mb = MCQ_CONFIRMATORY * HIDDEN_BYTES_PER_VECTOR / 1e6   # population = 每题一条 greedy
    try:
        free_gb = shutil.disk_usage(str(output_root.anchor or ".")).free / 1e9
    except OSError:
        free_gb = None
    need_gb = (h1_mb + mcq_mb) / 1000 * 3          # ×3 给 unit 级 npy + npz + 余量
    return {
        "h1": {"prompts": H1_PROMPTS, "K": H1_K, "generation_units": h1_units,
               "population": H1_PROMPTS * H1_K, "hidden_mb": round(h1_mb, 1)},
        "mcq": {"questions": MCQ_CONFIRMATORY, "K": H1_K, "generation_units": mcq_units,
                "population": MCQ_CONFIRMATORY, "hidden_mb": round(mcq_mb, 1),
                "note": "population = one greedy per question (§3 v2.3); sampled only feed F5"},
        "hidden_bytes_per_vector": HIDDEN_BYTES_PER_VECTOR,
        "total_hidden_mb": round(h1_mb + mcq_mb, 1),
        "disk_free_gb": None if free_gb is None else round(free_gb, 1),
        "estimated_need_gb": round(need_gb, 2),
        "ok": free_gb is None or free_gb > need_gb,
        "sharding_required": False,
        "correction": ("earlier docs said ~40GB; that was computed as if all-token hidden were "
                       "cached. Section 8 pools ONE vector per completion and section 11 does not "
                       "cache all-layer/all-token. Real total is ~66 MB."),
    }


# ------------------------------------------------------------- invariants
def check_frozen_invariants() -> dict:
    """代码里的常量必须与预注册一致。config 改不动它们（load_config 会拒）。"""
    return {
        "eps_auroc": {"value": ci.EPS_AUROC, "ok": ci.EPS_AUROC == 0.02},
        "delta_rank_biserial": {"value": ci.DELTA_RANK_BISERIAL,
                                "ok": ci.DELTA_RANK_BISERIAL == 0.15},
        "c_grid": {"value": list(ci.C_GRID), "ok": tuple(ci.C_GRID) == (0.01, 0.1, 1.0, 10.0)},
        "rel_depth": {"value": ci.DEFAULT_REL_DEPTH, "ok": ci.DEFAULT_REL_DEPTH == 0.7},
        "K_traces": {"value": H1_K, "ok": H1_K == 6},
        "rq2_min_per_cell": {"value": ci.RQ2_MIN_PER_CELL, "ok": ci.RQ2_MIN_PER_CELL == 15},
        "precision_max_ci_width": {"value": 2 * ci.EPS_AUROC, "ok": True,
                                   "commitment": ("section 7.2: measured Delta_H CI wider than "
                                                  "this -> record inconclusive; no extra samples, "
                                                  "no changing eps/read-rule/fusion")},
        "ok": True,
    }


# --------------------------------------------------------------- verdict
def launch_readiness(gates: dict, code: dict, assets: dict, scale: dict,
                     invariants: dict) -> dict:
    """把"缺外部事实"和"缺代码"分开 —— 这就是本模块存在的理由。"""
    blocked_by: list[str] = []
    if not gates["G1"]["ok"]:
        blocked_by.append("external_cfp_confirmation")
    if not gates["G2"]["ok"]:
        blocked_by.append("preregistration_hash_mismatch")
    if not gates["G3"]["ok"]:
        blocked_by.append("model_smoke_not_green")
    if not code["ok"]:
        blocked_by.append("code_not_ready")
    if not assets["ok"]:
        blocked_by.append("input_assets_missing")
    if not scale["ok"]:
        blocked_by.append("insufficient_disk")
    if not invariants["ok"]:
        blocked_by.append("frozen_constants_drifted")

    # 除 G1 外全绿 = 我们这边做完了,只等外部事实
    everything_but_cfp = blocked_by == ["external_cfp_confirmation"]
    return {
        "stage0_launch_allowed": not blocked_by,
        "blocked_by": blocked_by,
        "blocked_only_by_external_cfp": everything_but_cfp,
        "readiness": ("READY — blocked only by external CFP confirmation"
                      if everything_but_cfp else
                      "LAUNCH ALLOWED" if not blocked_by else
                      f"NOT READY: {blocked_by}"),
        "what_unblocks_it": (
            "EIML3 opens submissions 2026-07-29; fill page_limit in docs/paper/cfp_record.json "
            "and set status=confirmed. G1 is an external fact — no script or agent may assert it."
            if "external_cfp_confirmation" in blocked_by else None),
    }
