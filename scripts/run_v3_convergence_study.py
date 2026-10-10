"""§7.3 收敛研究：确定 R 与 B_per_batch 两个数值超参。

⚠️ **本研究不是性能盲的**（R 的判据含点估计），且它跑在**已揭盲的 Stage 0 数据**上、
用 **v3 的新 estimand**（S* 含 B1 identifier identity 与 B5 prompt）。因此：

    产物中的 Δ̂_H **不得**作为任何结果被引用、不得用于 N 的选择、
    不得写进论文。它只用来定 R 与 B_per_batch。

设计全部来自 preregistration_v3 §7.3，**预冻结**，运行只填数不改规则：

    R 扫描  R ∈ {1,2,3,5,8,12}，B_per_batch = B_max = 20000
            D_R(g) = max(|ΔΔ̂_H|, |ΔL|, |ΔU|)   与 R_max 比较
    B 扫描  B_per_batch ∈ {1000,…,20000}，R = R_max
            D_B(g) = max(|ΔL|, |ΔU|, |Δwidth|)  —— **不查点估计**
    选定    g* = min{ g : ∀g'≥g, D(g') < tol }，tol = 0.002
    跑完整网格，无 early stop；一律与最大档比较，不比相邻档。

**两项等价优化（只动 runner，不改 estimand / 预注册 / v3 核心）**：

  1. §7.3 只需要 Δ_H → `repeat_statistics(..., block_names=[])`，每单位只算
     S* 与 S*+H 两个 AUROC，不算五个 block（7 → 2）。
  2. R/B 网格天然**前缀嵌套** → 只做**一次** (R_max, B_max) 的 per-repeat draw
     扫描；R 各档沿 repeat 轴取前缀均值，B 各档取每批 draw 的前缀。
     单位数由朴素逐档的 16.7M 降到 4.8M。

等价性由 `tests/test_v3_convergence_runner.py` 证明：前缀导出的各档结果与逐档朴素
`pooled_draws` 逐位相等（含随 R 变化的 single-class 丢弃规则）。

Stage 0 cache 全程**只读**。中间缓存写仓库外的 scratch 目录，
`outputs/` 下只写那唯一一份 JSON（全部档位齐备后才原子写出）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention.evaluation import v3_adapter as ad            # noqa: E402
from recover_attention.evaluation import v3_bootstrap as vb          # noqa: E402
from recover_attention.evaluation import v3_repeat_estimand as re_   # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

# ── §7.3 预冻结网格 ────────────────────────────────────────────────────
R_GRID = (1, 2, 3, 5, 8, 12)
B_GRID = (1000, 2000, 5000, 10000, 20000)
R_MAX = max(R_GRID)
B_MAX = max(B_GRID)
TOL = 0.002
FALLBACK = {"R": 5, "B_per_batch": 10000}

OUT_JSON = PROJECT_ROOT / "outputs/logs/stage0_development_convergence_R_Bperbatch.json"
FEATURES = PROJECT_ROOT / "outputs/stage0/features"

MAX_HOURS = 12.0
MAX_PEAK_GB = 8.0

DELTA_ONLY: list[str] = []      # block_names=[] → 每单位只算 S* 与 S*+H


def prereg_sha() -> str:
    return hashlib.sha256(
        (PROJECT_ROOT / "docs/paper/preregistration_v3.md").read_bytes()).hexdigest()


def scratch_dir() -> Path:
    env = os.environ.get("V3_SCRATCH")
    d = Path(env) if env else Path(os.environ.get("TEMP", "/tmp")) / "v3_convergence_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def peak_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def load_arm() -> dict:
    recs = []
    with open(FEATURES / "h1_completion_records.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("eligible"):
                recs.append(r)
    npz = np.load(FEATURES / "h1_hidden.npz")
    hidden = {t: npz[t] for t in (r["trace_id"] for r in recs)}
    arm = ad.build_arm(recs, hidden)
    print(f"[data] eligible={arm['n']} groups={len(set(arm['groups']))} "
          f"pos={int(arm['y'].sum())}", flush=True)
    return arm


def cfg_for(n_repeats: int, b_per_batch: int) -> V3AnalysisConfig:
    return V3AnalysisConfig(n_repeats=n_repeats, b_per_batch=b_per_batch,
                            mode="production", r_b_source="convergence_study")


# ── 阶段 A：交叉拟合（R_max 次，前缀复用，可断点续跑）────────────────
def stage_a(arm: dict, sha: str, n_repeats: int = R_MAX) -> list[dict]:
    cache = scratch_dir()
    fp = hashlib.sha256(
        f"{sha}|{arm['n']}|{len(set(arm['groups']))}".encode()).hexdigest()[:16]
    out = []
    for r in range(n_repeats):
        f = cache / f"repeat_{r:02d}_{fp}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out.append({"repeat": r, "oof": {k: z[k] for k in z.files}})
            print(f"[A] repeat {r}: cached", flush=True)
            continue
        t0 = time.time()
        res = re_.run_repeat(y=arm["y"], groups=arm["groups"], blocks=arm["blocks"],
                             repeat=r, cfg=cfg_for(max(n_repeats, 1), B_MAX))
        np.savez_compressed(f, **res["oof"])
        out.append({"repeat": r, "oof": res["oof"]})
        print(f"[A] repeat {r}: {time.time() - t0:.1f}s -> {f.name}", flush=True)
    return out


# ── 阶段 B：一次 R_max × B_max 扫描，网格各档由前缀导出 ────────────────
def delta_matrix(arm, repeats, plan, n_repeats, progress_every: int = 20000):
    """Δ_r^(q) 矩阵（NaN = 该 repeat 在该 draw 上单类）。12×400k float64 ≈ 38 MB。"""
    Q = len(plan)
    M = np.full((n_repeats, Q), np.nan, dtype=float)
    batch_of = np.empty(Q, dtype=int)
    pos_in_batch = np.empty(Q, dtype=int)
    counter: dict[int, int] = {}
    y = arm["y"]
    oofs = [rep["oof"] for rep in repeats[:n_repeats]]
    t0 = time.time()
    for q, j, rows in plan.iter_draws():
        for r, oof in enumerate(oofs):
            M[r, q] = re_.repeat_statistics(y, oof, DELTA_ONLY, rows=rows)[re_.DELTA_H]
        batch_of[q] = j
        pos_in_batch[q] = counter.get(j, 0)
        counter[j] = pos_in_batch[q] + 1
        if progress_every and (q + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"    [scan] {q + 1}/{Q}  {el / 60:.1f} min  "
                  f"eta {(Q - q - 1) * el / (q + 1) / 60:.1f} min", flush=True)
    return M, batch_of, pos_in_batch


def cell_from_matrix(M, batch_of, pos_in_batch, r_level, b_level, b_batch):
    """由矩阵前缀导出一个 (R, B_per_batch) 档位。

    丢弃规则与逐档朴素路径一致：**该档所用的 repeat 中任一为 NaN 即丢弃该 draw**，
    故丢弃集合随 R 变化 —— 这一点必须复现，不能只算一次。
    """
    sel = np.flatnonzero(pos_in_batch < b_level)
    sub = M[:r_level, sel]
    keep = np.all(np.isfinite(sub), axis=0)
    dbar = sub[:, keep].mean(axis=0)
    if dbar.size == 0:
        raise RuntimeError(f"R={r_level} B={b_level}: zero valid draws")
    jk = vb.jackknife_mc(dbar, batch_of[sel][keep], b_batch)
    return {"R": int(r_level), "B_per_batch": int(b_level),
            "B_total": int(b_batch * b_level),
            "ci_lo": jk["ci_lo"], "ci_hi": jk["ci_hi"], "width": jk["width"],
            "se_jack_width": jk["se_jack"]["width"],
            "mc_band_width": jk["mc_band_width"],
            "n_draws": int(dbar.size),
            "n_discarded_single_class": int(sel.size - dbar.size)}


def select(cells: dict, ref_key, keys) -> dict:
    ref = cells[ref_key]
    D = {g: max(abs(c[k] - ref[k]) for k in keys) for g, c in cells.items()}
    levels = sorted(cells)
    chosen = levels[-1]
    for i, g in enumerate(levels):
        if all(D[h] < TOL for h in levels[i:]):
            chosen = g
            break
    return {"D": D, "selected": chosen, "at_grid_max": chosen == levels[-1], "tol": TOL}


def _guard(t_start: float) -> None:
    h = (time.time() - t_start) / 3600
    if h > MAX_HOURS:
        raise RuntimeError(f"elapsed {h:.1f} h > {MAX_HOURS} h — 停并报告")
    pk = peak_gb()
    if pk == pk and pk > MAX_PEAK_GB:
        raise RuntimeError(f"peak RSS {pk:.1f} GB > {MAX_PEAK_GB} GB — 停并报告")


def run_pilot(arm, sha, pilot_draws: int) -> dict:
    t0 = time.time()
    reps = stage_a(arm, sha, n_repeats=1)
    t_repeat = time.time() - t0
    pd = max(1, pilot_draws // 20)
    plan = vb.build_resample_plan(arm["groups"], cfg_for(1, pd),
                                  b_per_batch=pd, b_max=B_MAX)
    t1 = time.time()
    M, _b, _p = delta_matrix(arm, reps, plan, 1, progress_every=0)
    t_scan = time.time() - t1
    per_unit = t_scan / max(1, M.size)

    total_units = R_MAX * 20 * B_MAX          # 前缀复用后只需一次扫描
    est_a_h = t_repeat * R_MAX / 3600
    est_b_h = total_units * per_unit / 3600
    est_h = est_a_h + est_b_h
    pk = peak_gb()
    ok = bool(est_h <= MAX_HOURS and (pk != pk or pk <= MAX_PEAK_GB))
    print("\n" + "=" * 68)
    print(f"[pilot] 1 repeat 交叉拟合            {t_repeat:8.1f} s")
    print(f"[pilot] {M.size} 单位扫描（2 AUROC/单位）   {t_scan:8.1f} s"
          f"  → {per_unit * 1e3:.3f} ms/单位")
    print(f"[pilot] 前缀复用后单位数 = {R_MAX}×20×{B_MAX} = {total_units / 1e6:.1f}M")
    print(f"[pilot] 外推 阶段A {est_a_h:.2f} h + 阶段B {est_b_h:.2f} h = **{est_h:.2f} h**")
    print(f"[pilot] 当前 RSS {pk:.2f} GB")
    print(f"[pilot] 阈值 ≤{MAX_HOURS}h 且 ≤{MAX_PEAK_GB}GB → "
          f"{'可直接开全网格' if ok else '**超阈值 → 启用预注册回退值**'}")
    print("=" * 68, flush=True)
    verdict = {"est_hours": est_h, "est_stage_a_hours": est_a_h,
               "est_stage_b_hours": est_b_h, "ms_per_unit": per_unit * 1e3,
               "peak_gb": pk, "proceed": ok}
    (scratch_dir() / "pilot_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8")
    return verdict


def run_full(arm, sha) -> dict:
    t_start = time.time()
    repeats = stage_a(arm, sha)
    cfg_max = cfg_for(R_MAX, B_MAX)
    plan = vb.build_resample_plan(arm["groups"], cfg_max,
                                  b_per_batch=B_MAX, b_max=B_MAX)
    print(f"[B] single pass: R_max={R_MAX} × B_total={len(plan)} …", flush=True)
    M, batch_of, pos_in_batch = delta_matrix(arm, repeats, plan, R_MAX)
    _guard(t_start)

    cells_R, cells_B = {}, {}
    for r in R_GRID:
        c = cell_from_matrix(M, batch_of, pos_in_batch, r, B_MAX, cfg_max.b_batch)
        c["delta_H_point"] = re_.point_estimates(
            arm["y"], repeats[:r], DELTA_ONLY)[re_.DELTA_H]
        cells_R[r] = c
        print(f"[B] R={r:2d} B={B_MAX}: width={c['width']:.4f} n={c['n_draws']}",
              flush=True)
    for b in B_GRID:
        c = (cells_R[R_MAX] if b == B_MAX
             else cell_from_matrix(M, batch_of, pos_in_batch, R_MAX, b, cfg_max.b_batch))
        c.setdefault("delta_H_point", cells_R[R_MAX]["delta_H_point"])
        cells_B[b] = c
        print(f"[B] R={R_MAX} B={b:5d}: width={c['width']:.4f} n={c['n_draws']}",
              flush=True)
    _guard(t_start)

    if set(cells_R) != set(R_GRID) or set(cells_B) != set(B_GRID):
        raise RuntimeError("网格有缺档 —— 不写半成品 JSON")

    sel_R = select(cells_R, R_MAX, ("delta_H_point", "ci_lo", "ci_hi"))
    sel_B = select(cells_B, B_MAX, ("ci_lo", "ci_hi", "width"))
    sha_end = prereg_sha()
    if sha_end != sha:
        raise RuntimeError(f"prereg_v3 SHA changed mid-run: {sha} -> {sha_end}")

    report = {
        "schema_version": "stage0_development_convergence_v1",
        "performance_blind": False,
        "scope": "development-only hyperparameter selection (R, B_per_batch)",
        "must_not_be_used_for": ["N selection", "any reported result or claim",
                                 "any Stage 1 conclusion"],
        "warning": ("Runs on ALREADY-UNBLINDED Stage 0 data under the v3 estimand. "
                    "Delta_H values here are development artefacts, not results."),
        "prereg_v3_sha256": sha,
        "frozen_design": {"R_grid": list(R_GRID), "B_grid": list(B_GRID),
                          "R_max": R_MAX, "B_max": B_MAX, "tol": TOL,
                          "compare_against": "grid maximum (not adjacent level)",
                          "R_checks": ["delta_H_point", "ci_lo", "ci_hi"],
                          "B_checks": ["ci_lo", "ci_hi", "width"],
                          "B_note": "B_per_batch does not enter the point estimate",
                          "fallback": FALLBACK},
        "computation": {
            "method": "single R_max x B_max pass; grid levels derived by prefix reuse",
            "equivalence": "tests/test_v3_convergence_runner.py proves prefix-derived "
                           "cells equal naive per-cell pooled_draws bitwise",
            "delta_only": "repeat_statistics called with block_names=[] (2 AUROCs/unit)"},
        "data": {"source": "outputs/stage0/features (READ-ONLY)",
                 "n": arm["n"], "n_groups": len(set(arm["groups"])),
                 "n_positive": int(arm["y"].sum())},
        "plan_audit": plan.audit(),
        "R_sweep": {str(k): v for k, v in cells_R.items()},
        "B_sweep": {str(k): v for k, v in cells_B.items()},
        "selection": {"R": {**sel_R, "D": {str(k): v for k, v in sel_R["D"].items()}},
                      "B_per_batch": {**sel_B,
                                      "D": {str(k): v for k, v in sel_B["D"].items()}}},
        "chosen": {"R": sel_R["selected"], "B_per_batch": sel_B["selected"],
                   "r_b_source": "convergence_study", "fallback_used": False},
        "total_seconds": round(time.time() - t_start, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OUT_JSON)
    print(f"\n[done] R* = {sel_R['selected']}  B_per_batch* = {sel_B['selected']}")
    print(f"[done] -> {OUT_JSON}", flush=True)
    return report


def write_fallback_report(arm, sha, verdict: dict) -> dict:
    """§7.3 冻结的回退分支：收敛研究因算力/时间无法完成时启用，**必须显式声明**。"""
    report = {
        "schema_version": "stage0_development_convergence_v1",
        "performance_blind": False,
        "scope": "development-only hyperparameter selection (R, B_per_batch)",
        "must_not_be_used_for": ["N selection", "any reported result or claim",
                                 "any Stage 1 conclusion"],
        "prereg_v3_sha256": sha,
        "frozen_design": {"R_grid": list(R_GRID), "B_grid": list(B_GRID),
                          "tol": TOL, "fallback": FALLBACK},
        "data": {"source": "outputs/stage0/features (READ-ONLY)",
                 "n": arm["n"], "n_groups": len(set(arm["groups"]))},
        "convergence_study_completed": False,
        "reason": "compute/time budget exceeded (see pilot_extrapolation)",
        "pilot_extrapolation": verdict,
        "chosen": {"R": FALLBACK["R"], "B_per_batch": FALLBACK["B_per_batch"],
                   "r_b_source": "fallback", "fallback_used": True,
                   "declaration": "§7.3 frozen fallback values used because the "
                                  "convergence study could not be completed; this is "
                                  "NOT a convergence result and must not be reported "
                                  "as one."},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OUT_JSON)
    print(f"[fallback] R={FALLBACK['R']} B_per_batch={FALLBACK['B_per_batch']} "
          f"(declared) -> {OUT_JSON}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="pilot 后自动决定：达标则跑全网格，否则启用预注册回退值")
    ap.add_argument("--pilot-draws", type=int, default=2000)
    args = ap.parse_args()
    if not (args.pilot or args.full or args.auto):
        ap.error("need --pilot / --full / --auto")

    sha = prereg_sha()
    print(f"[cfg] prereg_v3 sha256 = {sha}", flush=True)
    print(f"[cfg] scratch = {scratch_dir()}", flush=True)
    arm = load_arm()

    if args.full:
        run_full(arm, sha)
        return

    verdict = run_pilot(arm, sha, args.pilot_draws)
    if not args.auto:
        return
    if verdict["proceed"]:
        run_full(arm, sha)
    else:
        write_fallback_report(arm, sha, verdict)


if __name__ == "__main__":
    main()
