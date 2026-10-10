r"""任意两个探针条件之间的**直接配对比较**（只读已保存的折外预测，不重新拟合模型）。

起因（2026-09-19）：报告里把两个参照方向混用了 —— 表格那一列写的是
「放宽后 Δ_H 的变化」（应为 `han_l1_wide − han_l1`），填进去的却是
`han_l1_wide − ref_l2_wide_grid`。两者不是一回事，必须分别算、分别报。

纪律：
  · **不重新拟合模型**：只读 `oof_{arm}_{condition}.npz`
  · **不用两个区间相减构造区间**：每一次重采样上同时算两个条件的 Δ_H，
    存它们的**差**，再对这组差取分位数
  · **不硬编码任何期望值**：如需与外部数字核对，用 `--cross-check` 传入，
    脚本只报告是否一致，不参与计算
  · 重采样计划与主路径一致：cluster/group bootstrap，`cfg.batch_seed(j)`，
    B = `cfg.b_total`；任一条件在某轮出现非有限值 → **整轮丢弃**，保持配对

estimand 与主路径同构：逐 repeat 算 `AUROC(S*+H) − AUROC(S*)`，再跨 repeat 平均。

用法：
    python scripts/posthoc_pairwise_delta.py --arm h1 --a han_l1_wide --b han_l1
    python scripts/posthoc_pairwise_delta.py --arm mcq --a han_l1_wide --b han_l1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recover_attention import conditional_increment as ci             # noqa: E402
from recover_attention.evaluation import v3_bootstrap as vb           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig   # noqa: E402

OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(arm: str, cond: str) -> tuple[dict, Path]:
    p = OUT_DIR / f"oof_{arm}_{cond}.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, p


def _delta_per_repeat(d: dict, y: np.ndarray, n_rep: int, rows) -> list[float]:
    """逐 repeat 的 Δ_H（AUROC(S*+H) − AUROC(S*)），在给定行子集上。"""
    ys = y[rows]
    return [ci.auroc(d[f"r{r}_S*+H"][rows], ys) - ci.auroc(d[f"r{r}_S*"][rows], ys)
            for r in range(n_rep)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("h1", "mcq"), required=True)
    ap.add_argument("--a", required=True, help="比较方向的**被减数**")
    ap.add_argument("--b", required=True, help="比较方向的**减数**（参照）")
    ap.add_argument("--cross-check", type=float, default=None,
                    help="外部报出的点估计，仅用于核对；**不参与计算**")
    ap.add_argument("--cross-check-ci", type=float, nargs=2, default=None,
                    help="外部报出的区间上下界，仅用于核对")
    ap.add_argument("--tol", type=float, default=1e-9, help="核对容差")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    t0 = time.perf_counter()
    A, pa = _load(args.arm, args.a)
    B, pb = _load(args.arm, args.b)

    # ── 对齐核对：不对齐就没有"配对"可言 ──────────────────────────────────
    align = {}
    for k in ("sample_id", "group_id", "y"):
        same = bool(np.array_equal(A[k], B[k]))
        align[k] = same
        if not same:
            raise RuntimeError(f"{k} 在两个条件之间不一致 —— 无法配对比较")
    for r in range(cfg.n_repeats):
        k = f"r{r}_fold"
        align[k] = bool(np.array_equal(A[k], B[k]))
        if not align[k]:
            raise RuntimeError(f"{k} 不一致 —— 外层折归属必须相同")
    # S* 是共享的，必须逐位相同（否则"只换 H"这件事就不成立）
    s_star_max_diff = max(
        float(np.max(np.abs(A[f"r{r}_S*"] - B[f"r{r}_S*"]))) for r in range(cfg.n_repeats))
    if s_star_max_diff != 0.0:
        raise RuntimeError(f"S* 在两个条件间不是逐位相同（最大差 {s_star_max_diff:.3e}）")

    y, groups = A["y"].astype(int), A["group_id"]
    n = y.size
    all_rows = np.arange(n)
    pt_a = float(np.mean(_delta_per_repeat(A, y, cfg.n_repeats, all_rows)))
    pt_b = float(np.mean(_delta_per_repeat(B, y, cfg.n_repeats, all_rows)))
    point = pt_a - pt_b

    # ── 配对 bootstrap：**同一批重采样行**同时喂两个条件，存它们的差 ───────
    plan = vb.build_resample_plan(groups, cfg)
    diffs: list[float] = []
    n_disc = 0
    for _q, _j, rows in plan.iter_draws():
        da = _delta_per_repeat(A, y, cfg.n_repeats, rows)
        db = _delta_per_repeat(B, y, cfg.n_repeats, rows)
        if not (np.all(np.isfinite(da)) and np.all(np.isfinite(db))):
            n_disc += 1                       # 整轮丢弃，保持配对
            continue
        diffs.append(float(np.mean(da)) - float(np.mean(db)))
    ci_d = vb.percentile_ci(np.asarray(diffs, dtype=float), cfg.alpha)

    print(f"[{args.arm}] {args.a} − {args.b}")
    print(f"  Δ_H({args.a}) = {pt_a:+.10f}   Δ_H({args.b}) = {pt_b:+.10f}")
    print(f"  增量之差       = {point:+.10f}")
    print(f"  95% 配对区间   = [{ci_d['ci_lo']:+.10f}, {ci_d['ci_hi']:+.10f}]")
    print(f"  B={plan.b_total}  丢弃单类 {n_disc}  有效 {len(diffs)}")

    cross = None
    if args.cross_check is not None:
        d_pt = abs(point - args.cross_check)
        cross = {"external_point": args.cross_check, "abs_diff_point": d_pt,
                 "point_matches": d_pt <= args.tol, "tol": args.tol}
        if args.cross_check_ci is not None:
            lo, hi = args.cross_check_ci
            cross.update(external_ci=[lo, hi],
                         abs_diff_ci_lo=abs(ci_d["ci_lo"] - lo),
                         abs_diff_ci_hi=abs(ci_d["ci_hi"] - hi),
                         ci_matches=(abs(ci_d["ci_lo"] - lo) <= args.tol
                                     and abs(ci_d["ci_hi"] - hi) <= args.tol))
        print(f"  与外部数字核对：{json.dumps(cross, ensure_ascii=False)}")

    out = {
        "schema": "posthoc_pairwise_delta_v1",
        "arm": args.arm,
        "direction": f"{args.a} - {args.b}",
        "minuend": args.a, "subtrahend_reference": args.b,
        "delta_H_minuend": pt_a, "delta_H_reference": pt_b,
        "difference_point": point,
        "paired_ci": {"ci_lo": ci_d["ci_lo"], "ci_hi": ci_d["ci_hi"],
                      "width": ci_d["ci_hi"] - ci_d["ci_lo"], "alpha": cfg.alpha},
        "method": ("每一次重采样上同时计算两个条件的 Δ_H，存其**差**，再取分位数；"
                   "**不是**两个独立区间相减"),
        "estimand": "逐 repeat 的 AUROC(S*+H) − AUROC(S*)，再跨 repeat 平均",
        "resample": {"unit": "cluster/group bootstrap（同主路径）",
                     "b_total": plan.b_total, "b_batch": cfg.b_batch,
                     "b_per_batch": cfg.b_per_batch,
                     "batch_seeds": [cfg.batch_seed(j) for j in range(cfg.b_batch)],
                     "discarded_single_class": n_disc, "n_effective": len(diffs)},
        "alignment_checks": {**align, "s_star_max_abs_diff": s_star_max_diff,
                             "n": int(n), "n_positive": int(y.sum()),
                             "n_groups": int(len(set(map(str, groups))))},
        "inputs": {args.a: {"path": str(pa.relative_to(ROOT)), "sha256": _sha(pa)},
                   args.b: {"path": str(pb.relative_to(ROOT)), "sha256": _sha(pb)}},
        "seeds": {"fold_seed_expr": "20260804 + 100*r",
                  "fold_seeds": [cfg.fold_seed(r) for r in range(cfg.n_repeats)],
                  "base_random_state": cfg.base_random_state},
        "command": f"python scripts/posthoc_pairwise_delta.py --arm {args.arm} "
                   f"--a {args.a} --b {args.b}",
        "refit_models": False,
        "cross_check": cross,
        "secs": round(time.perf_counter() - t0, 1),
    }
    f = OUT_DIR / f"pairwise_{args.arm}_{args.a}_minus_{args.b}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n",
                 encoding="utf-8")
    print(f"[out] {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
