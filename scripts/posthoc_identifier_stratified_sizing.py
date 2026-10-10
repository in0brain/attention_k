"""POST-HOC PLANNING：identifier-stratified sizing —— 下一版该怎么设计？

**这不是结果分析，是设计规划分析。** 不参与主结果，不改动、不重跑、不替换
`stage0_analysis_report.json`，也不改变 identification audit 的任何判定。
它只回答一个 future-work 的定量问题：

    若从一开始就按 fabricated identifier 分组设计，需要多少个独立 identifier-group
    （或把最大分量占比压到多少）才能把 Δ_H 的 CI 宽度压回 ≤ 0.04（§7.2 上限）？

────────────────────────────────────────────────────────────────────────
起点（来自 identification audit 发现二，strict grouping = prompt ∪ fabricated-id）

    groups        = 277
    max component = 200 records（占 2016 条的 9.9%）
    Δ_H point     = +0.044
    CI            = [−0.013, +0.093]，width = 0.107   → read=inconclusive
    §7.2 上限     = 0.040

CI 之所以从 A 分组的 0.040 涨到 0.107，有两个来源，必须分开量化：
    (i)  独立单元变少      358 prompt-group → 277 分量
    (ii) 单元极不均匀      最大分量独占 9.9%，一次重采样抽中与否就大幅摆动统计量
只报 (i) 会给出错误的扩样建议 —— 若不同时控制 (ii)，扩样收益会被方差地板吃掉。

────────────────────────────────────────────────────────────────────────
两个设计杠杆，分别模拟

杠杆 1｜分量数 k（保持现有分量大小分布）
    从现有 277 个分量中**有放回**抽 k 个（k 可大于 277），复用固定的 OOF 分数，
    重算 Δ 的 percentile CI。这是标准的 bootstrap 样本量外推：假设新分量与已观测分量
    同分布。拟合 width = a·k^(−b) 并解出 width = 0.04 所需的 k。
    **b 本身是诊断量**：理论上 b ≈ 0.5；若实测 b 明显小于 0.5，说明存在方差地板，
    单靠扩样达不到目标 —— 这正是杠杆 2 要处理的。

杠杆 2｜最大分量占比上限 c（保持总样本量 n）
    把超过 c·n 的分量切成若干块，模拟"设计上限制 identifier 复用"的未来数据。
    **这是规划用的反事实，不是对本数据的重新分析** —— 对现有数据真的切开分量会重新
    打开它当初被建出来所要封堵的泄漏通道。它回答的是"若未来设计使 identifier 不聚集，
    同样的 n 能达到多少精度"。

────────────────────────────────────────────────────────────────────────
不做的事
    不改 Δ_H 的点估计（sizing 只关心宽度，不关心位置）
    不用本分析论证"其实有增量"或"其实没有增量"
    不 bump 预注册版本
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import ladder as ld  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_H1, load_arm  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "leak", PROJECT_ROOT / "scripts" / "posthoc_h1_identifier_leakage_diagnostic.py")
leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak)

TARGET_WIDTH = 0.04          # §7.2 max_ci_width
MIN_VALID_FRAC = 0.8


def ci_width_at_k(y, s_o, s_oh, groups, k, n_boot, seed):
    """重采样 k 个 group（可 k > 现有分量数）得到 Δ 的 percentile CI 宽度。

    不复用 ci.paired_grouped_bootstrap_delta —— 那个函数把重采样规模写死为
    len(uniq)（它是 confirmatory 分析用的冻结实现，不该为 sizing 改动）。
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    imap = {gv: np.where(g == gv)[0] for gv in uniq}
    rng = np.random.default_rng(seed)
    deltas, n_bad = [], 0
    for _ in range(n_boot):
        rows = np.concatenate([imap[gv] for gv in rng.choice(uniq, size=k, replace=True)])
        a_o, a_oh = ci.auroc(s_o[rows], y[rows]), ci.auroc(s_oh[rows], y[rows])
        if a_o != a_o or a_oh != a_oh:
            n_bad += 1
            continue
        deltas.append(a_oh - a_o)
    d = np.asarray(deltas)
    if d.size < MIN_VALID_FRAC * n_boot:
        return None
    lo, hi = ci._percentile_ci(d)
    return {"k": int(k), "ci_lo": float(lo), "ci_hi": float(hi),
            "width": float(hi - lo), "n_valid": int(d.size), "n_discarded": int(n_bad)}


def cap_components(groups, max_share, seed=0):
    """把超过 max_share·n 的分量切成若干块（规划反事实，见文件头）。"""
    g = np.asarray(groups).copy().astype(object)
    n = g.size
    cap = max(1, int(np.floor(max_share * n)))
    rng = np.random.default_rng(seed)
    out = g.copy()
    for gv in np.unique(g):
        idx = np.where(g == gv)[0]
        if idx.size <= cap:
            continue
        idx = rng.permutation(idx)
        for j, start in enumerate(range(0, idx.size, cap)):
            out[idx[start:start + cap]] = f"{gv}__split{j}"
    return out


def fit_power_law(ks, widths):
    """log(width) = log a − b·log k。返回 (a, b) 与解出 target 所需的 k。"""
    lk, lw = np.log(np.asarray(ks, float)), np.log(np.asarray(widths, float))
    b_neg, log_a = np.polyfit(lk, lw, 1)
    b = -float(b_neg)
    a = float(np.exp(log_a))
    k_needed = float((a / TARGET_WIDTH) ** (1.0 / b)) if b > 0 else float("inf")
    return a, b, k_needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/stage0/features")
    ap.add_argument("--output", default="outputs/stage0/evaluation/"
                                        "posthoc_identifier_stratified_sizing.json")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--timestamp", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    config = Stage0Config(n_outer_folds=cfg.n_outer_folds, n_inner_folds=cfg.n_inner_folds,
                          n_boot=cfg.n_boot, seed=cfg.seed, mode=cfg.mode,
                          allow_pilot_population=False)
    in_dir = Path(args.input_dir)
    cache = load_arm(ARM_H1, in_dir / "h1_completion_records.jsonl", in_dir / "h1_hidden.npz",
                     config, label_field="label", id_field="trace_id")

    groups_strict, comp_stats = leak.component_groups(cache.records, link_prompt=True,
                                                      link_fabricated=True)
    print(f"[sizing] strict grouping: {comp_stats['n_components']} components, "
          f"largest {comp_stats['largest_component']} records "
          f"({comp_stats['largest_component_share']:.1%})")

    print("[sizing] running ladder once under strict grouping to obtain OOF scores ...")
    cache_strict = replace(cache, groups=groups_strict)
    lad = ld.run_ladder(cache_strict, config)
    y = cache_strict.y
    s_o = lad["scores"][cache_strict.spec["o_rung"]]
    s_oh = lad["scores"]["O_plus_H"]
    point = float(ci.auroc(s_oh, y) - ci.auroc(s_o, y))
    n_now = int(len(np.unique(groups_strict)))
    print(f"[sizing] Delta_H point (fixed, not re-estimated): {point:+.4f}")

    # ---------------- 杠杆 1：分量数 ----------------
    ks = [100, 150, 200, n_now, 400, 600, 900, 1400, 2000, 3000, 4500]
    lever1 = []
    for k in ks:
        ws = []
        for s in range(args.n_seeds):
            r = ci_width_at_k(y, s_o, s_oh, groups_strict, k, args.n_boot, seed=1000 + s)
            if r:
                ws.append(r["width"])
        if ws:
            lever1.append({"k": k, "width_mean": float(np.mean(ws)),
                           "width_sd": float(np.std(ws)), "n_seeds": len(ws)})
            print(f"[sizing]   k={k:5d}  width={np.mean(ws):.4f}")
    a, b, k_needed = fit_power_law([r["k"] for r in lever1],
                                   [r["width_mean"] for r in lever1])
    print(f"[sizing] fitted width = {a:.4f} * k^-{b:.3f}   (theory b=0.5)")
    print(f"[sizing] k needed for width<={TARGET_WIDTH}: {k_needed:.0f} "
          f"({k_needed / n_now:.1f}x current)")

    # ---------------- 杠杆 2：最大分量占比 ----------------
    lever2 = []
    for share in (comp_stats["largest_component_share"], 0.05, 0.03, 0.02, 0.01, 0.005):
        gg = cap_components(groups_strict, share)
        k_eff = int(len(np.unique(gg)))
        ws = []
        for s in range(args.n_seeds):
            r = ci_width_at_k(y, s_o, s_oh, gg, k_eff, args.n_boot, seed=2000 + s)
            if r:
                ws.append(r["width"])
        if ws:
            lever2.append({"max_share": round(float(share), 4), "n_groups": k_eff,
                           "width_mean": float(np.mean(ws))})
            print(f"[sizing]   max_share<={share:.3%}  groups={k_eff:5d}  "
                  f"width={np.mean(ws):.4f}")

    meets = [r for r in lever2 if r["width_mean"] <= TARGET_WIDTH]
    report = {
        "schema_version": "posthoc_identifier_stratified_sizing_v1",
        "status": ("POST-HOC PLANNING ANALYSIS — not a result. Does not modify or supersede "
                  "stage0_analysis_report.json, and is not used to argue for or against an "
                  "increment. It sizes a future design."),
        "timestamp": args.timestamp,
        "target_ci_width": TARGET_WIDTH,
        "current_strict_grouping": {
            "n_groups": n_now,
            "largest_component_records": comp_stats["largest_component"],
            "largest_component_share": comp_stats["largest_component_share"],
            "n_records": int(y.size),
            "n_positive": int((y == 1).sum()),
            "delta_H_point": point,
            "ci_width": lever1[[r["k"] for r in lever1].index(n_now)]["width_mean"],
        },
        "lever1_group_count": {
            "method": ("resample k groups with replacement from the observed components, "
                      "OOF scores held fixed; assumes new groups are drawn from the same "
                      "size/effect distribution"),
            "curve": lever1,
            "fit": {"a": a, "b": b, "b_theory": 0.5,
                    "note": ("b < 0.5 indicates a variance floor from component-size "
                            "heterogeneity: scaling up alone will underdeliver")},
            "k_needed_for_target": k_needed,
            "multiple_of_current": k_needed / n_now,
        },
        "lever2_max_component_share": {
            "method": ("split components exceeding the cap; a PLANNING counterfactual for a "
                      "design that limits identifier reuse, NOT a re-analysis of this data "
                      "(actually splitting would reopen the leakage the components prevent)"),
            "curve": lever2,
            "smallest_share_meeting_target": (meets[0]["max_share"] if meets else None),
        },
        "recommendation": None,
    }

    if meets:
        rec = (f"at the current n={y.size}, capping the largest identifier component at "
               f"<={meets[0]['max_share']:.1%} of records ({meets[0]['n_groups']} groups) "
               f"already reaches width {meets[0]['width_mean']:.4f} <= {TARGET_WIDTH}")
    else:
        rec = (f"capping alone does not reach the target at n={y.size}; the group count must "
               f"also grow, to roughly {k_needed:.0f} independent identifier groups "
               f"({k_needed / n_now:.1f}x current)")
    report["recommendation"] = rec
    print(f"\n[sizing] RECOMMENDATION: {rec}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"[sizing] report -> {out}")


if __name__ == "__main__":
    main()
