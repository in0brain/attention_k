"""Read-only Stage 1 diagnostics: frozen comparisons or aligned OOF exports.

No model fitting, generation, bootstrap, or inference is performed. Summary
mode uses an audited result and mandatory SHA manifest. OOF mode is exploratory
and plots each repeat independently; it never ensembles or pools repeats.
See docs/progress/stage1_diagnostic_figures_history.md for the input contract.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recover_attention.data_io import (file_sha256, read_json, read_jsonl,
                                       save_plot_pair, write_jsonl)
from recover_attention.schemas import (validate_stage1_diagnostic_oof,
                                       validate_stage1_diagnostic_result)

PRIMARY = "frozen_implementation_primary"
NO_B4 = "no_B4_dependency_sensitivity"
BLUE, GREEN, VERMILION, GREY = "#0072B2", "#009E73", "#D55E00", "#777777"
COLORS = {"S*": GREY, "H": GREEN, "S*+H": BLUE}
BLOCKS = {"B1": "答案身份", "B2": "表面形式", "B3": "文本内容",
          "B4": "生成与采样信号", "B5": "提示上下文"}


def configure_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                 if name in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
                         "font.size": 10, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})


def load_summary(source: Path, manifest: Path):
    metadata = read_json(manifest)
    expected = metadata.get("result", {}).get("sha256")
    actual = file_sha256(source)
    if expected != actual:
        raise ValueError(f"{source}: SHA does not match {manifest}")
    result = read_json(source)
    validate_stage1_diagnostic_result(result)
    return result, actual


def build_block_bounds(result):
    """Plot arms[primary][arm].m1b_{non_inferiority,iv_a_direct_superiority}.

    Each dot is a marginal one-sided lower bound, NOT a two-sided interval.
    Non-inferiority bounds are shifted by the stored eps_NI so that both
    columns display distance to their own decision boundary.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.8))
    for i, arm in enumerate(("h1", "mcq")):
        row = result["arms"][PRIMARY][arm]
        blocks = row["block_names_s_star"]
        for j, (key, color) in enumerate((("m1b_non_inferiority", GREY),
                                          ("m1b_iv_a_direct_superiority", BLUE))):
            offset = row[key]["eps_ni"] if j == 0 else 0
            values = [row[key]["L_marg"][block] + offset for block in blocks]
            ax = axes[i, j]
            ax.axvline(0, color=VERMILION, linestyle="--", lw=1.2)
            ax.hlines(range(len(blocks)), 0, values, color=color, alpha=0.5)
            ax.scatter(values, range(len(blocks)), color=color, s=40, zorder=3)
            for y, value in enumerate(values):
                ax.annotate(f"{value:+.4f}", (value, y), xytext=(6, 5),
                            textcoords="offset points", fontsize=8)
            ax.set_yticks(range(len(blocks)), [f"{b} {BLOCKS[b]}" for b in blocks])
            ax.invert_yaxis()
            ax.set_xlim(min(-0.025, min(values) - 0.025), max(values) + 0.085)
            ax.set_ylim(len(blocks) - 0.5, -0.65)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{arm.upper()}：" + ("组合基线相对单块" if j == 0 else "融合模型相对单块"))
            ax.set_xlabel("单侧下界 + 非劣容忍度" if j == 0 else "单侧下界 − 优效门槛（0）")
    fig.tight_layout(h_pad=2)
    return fig


def build_b4_comparison(result):
    """Read arms[A/C][arm].auroc and delta_h_point; no new significance test."""
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.3))
    for i, arm in enumerate(("h1", "mcq")):
        rows = [result["arms"][path][arm] for path in (PRIMARY, NO_B4)]
        all_scores = []
        for model, color in COLORS.items():
            vals = [row["auroc"][model] for row in rows]
            all_scores.extend(vals)
            marker = {"S*": "s", "H": "^", "S*+H": "o"}[model]
            axes[i, 0].plot([0, 1], vals, marker=marker, color=color, label=model,
                            lw=1.5, markersize=7, markerfacecolor="white" if model == "S*" else color)
            for x, value in enumerate(vals):
                offset = {"S*": (-28, -16), "H": (28, -16), "S*+H": (0, 12)}[model]
                axes[i, 0].annotate(f"{value:.4f}", (x, value), xytext=offset,
                                    textcoords="offset points", ha="center", fontsize=8, color=color)
        deltas = [row["delta_h_point"] for row in rows]
        axes[i, 1].plot([0, 1], deltas, "o-", color=VERMILION, lw=2, label="delta_h")
        for x, value in enumerate(deltas):
            axes[i, 1].annotate(f"{value:+.4f}", (x, value), xytext=(0, 10),
                                textcoords="offset points", ha="center")
        pad = max(0.01, (max(all_scores) - min(all_scores)) * .18)
        axes[i, 0].set_ylim(max(0, min(all_scores) - pad), min(1.02, max(all_scores) + pad))
        lo, hi = min(0, min(deltas)), max(0, max(deltas))
        pad_delta = max(0.001, (hi - lo) * .3)
        axes[i, 1].set_ylim(lo - pad_delta if lo < 0 else 0, hi + pad_delta)
        axes[i, 0].set_ylabel("AUROC")
        axes[i, 1].set_ylabel(r"$\Delta_H$")
        axes[i, 0].set_title(f"{arm.upper()}：绝对检测性能")
        axes[i, 1].set_title(f"{arm.upper()}：条件增量")
        for ax in axes[i]:
            ax.set_xticks([0, 1], ["A：包含 B4", "C：移除 B4"])
            ax.set_xlim(-0.25, 1.25)
            ax.grid(axis="y", alpha=0.2)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    delta_handles, _ = axes[0, 1].get_legend_handles_labels()
    labels = [r"$S^*$：组合基线", r"$H$：隐藏状态",
              r"$S^*{+}H$：融合模型", r"$\Delta_H$：条件增量"]
    fig.legend(handles + delta_handles, labels, loc="center left",
               bbox_to_anchor=(0.80, 0.5), frameon=False, labelspacing=1.4,
               handlelength=2.4, handletextpad=0.7)
    fig.tight_layout(rect=(0, 0, 0.80, 1), h_pad=2)
    return fig


def spearman_matrix(scores):
    """Pairwise rank correlation, with None for insufficient/constant columns."""
    n, width = scores.shape
    out = [[None for _ in range(width)] for _ in range(width)]
    if n < 2:
        return out
    ranks = np.stack([rankdata(scores[:, j]) for j in range(width)], axis=1)
    for a in range(width):
        for b in range(width):
            if np.ptp(ranks[:, a]) and np.ptp(ranks[:, b]):
                out[a][b] = float(np.corrcoef(ranks[:, a], ranks[:, b])[0, 1])
    return out


def describe_repeat(rows, threshold=None):
    """One repeat only. Scores are probabilities of error (y=1).

    Signed movement is (2*y-1)*(p_fused-p_baseline). It is not a change in
    accuracy or AUROC, and no inferential intervals are computed from rows.
    """
    repeats = {r["repeat"] for r in rows}
    if len(repeats) != 1:
        raise ValueError("describe_repeat requires exactly one repeat")
    names = [n for n in ("B1", "B2", "B3", "B4", "B5", "S*", "H", "S*+H")
             if n in rows[0]["scores"]]
    y = np.array([r["y"] for r in rows])
    matrix = np.array([[r["scores"][n] for n in names] for r in rows])
    delta = matrix[:, names.index("S*+H")] - matrix[:, names.index("S*")]
    signed = (2 * y - 1) * delta
    correlations = {}
    for label, mask in (("all", np.ones(len(y), dtype=bool)), ("correct", y == 0), ("error", y == 1)):
        correlations[label] = {"n": int(mask.sum()), "rho": spearman_matrix(matrix[mask])}
    stats = {"kind": "oof_repeat_diagnostics", "role": "posthoc_descriptive",
             "data_role": rows[0]["data_role"], "repeat": rows[0]["repeat"],
             "model": rows[0]["model"], "arm": rows[0]["arm"], "path": rows[0]["path"],
             "run_id": rows[0]["run_id"], "n": len(rows),
             "n_groups": len({r["group_id"] for r in rows}), "predictors": names,
             "correlations": correlations,
             "score_movement": {"toward_true_class": int((signed > 0).sum()),
                                "away_from_true_class": int((signed < 0).sum()),
                                "unchanged": int((signed == 0).sum())}}
    if threshold is not None:
        old = matrix[:, names.index("S*")] >= threshold
        new = matrix[:, names.index("S*+H")] >= threshold
        old_ok, new_ok = old == y, new == y
        stats["threshold"] = float(threshold)
        stats["transitions"] = {
            "both_correct": int((old_ok & new_ok).sum()),
            "corrected": int((~old_ok & new_ok).sum()),
            "new_error": int((old_ok & ~new_ok).sum()),
            "both_wrong": int((~old_ok & ~new_ok).sum()),
        }
    return stats


def oof_title(rows):
    first = rows[0]
    prefix = "合成测试数据；不代表实验结果" if first["data_role"] == "synthetic" else "探索性描述"
    return f"{prefix} | {first['model']} / {first['arm']} / repeat {first['repeat']}"


def build_score_shift(rows):
    """Use scores[S*,S*+H] and y; display all samples without subsampling."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    densities = []
    count_cmap = LinearSegmentedColormap.from_list("visible_counts", ["#AAAAAA", "#111111"])
    for label, ax in enumerate(axes):
        chosen = [r for r in rows if r["y"] == label]
        x = np.array([r["scores"]["S*"] for r in chosen])
        delta = np.array([r["scores"]["S*+H"] - r["scores"]["S*"] for r in chosen])
        density = ax.hexbin(x, delta, gridsize=25, mincnt=1, bins="log", cmap=count_cmap, extent=(0, 1, -1, 1))
        densities.append(density)
        ax.axhline(0, color=BLUE, lw=1)
        ax.set(xlim=(0, 1), ylim=(-1, 1), xlabel="组合基线的错误概率",
               ylabel="融合错误概率 − 基线错误概率")
        ax.set_title(("真实正确：向下为有利变化" if label == 0 else "真实错误：向上为有利变化") + f"\nn={len(chosen)}")
    maximum = max(2, max(float(d.get_array().max()) for d in densities))
    norm = LogNorm(vmin=1, vmax=maximum)
    ticks = sorted({1, int(maximum), *[10 ** p for p in range(1, 6) if 10 ** p < maximum]})
    for ax, density in zip(axes, densities):
        density.set_norm(norm)
        colorbar = fig.colorbar(density, ax=ax, label="格内样本数（共享对数色标）", ticks=ticks)
        colorbar.ax.set_yticklabels([str(tick) for tick in ticks])
        colorbar.ax.minorticks_off()
    fig.suptitle(oof_title(rows), fontsize=11)
    fig.text(0.5, 0.02, "分数方向变化不是准确率变化；每个 repeat 单独展示，不合并重复预测，不据样本数推断显著性。",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.065, 1, 0.92))
    return fig


def build_correlation(rows, stats):
    """Read per-repeat Spearman statistics, both pooled and within true labels."""
    names = stats["predictors"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    cmap = LinearSegmentedColormap.from_list("oi_correlation", [VERMILION, "#FFFFFF", BLUE])
    cmap.set_bad("#E6E6E6")
    for ax, (key, title) in zip(axes, (("all", "全部样本"), ("correct", "真实正确"), ("error", "真实错误"))):
        record = stats["correlations"][key]
        values = np.array(record["rho"], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for a in range(len(names)):
            for b in range(len(names)):
                value = values[a, b]
                ax.text(b, a, "NA" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=7 if len(names) > 4 else 10,
                        color="white" if np.isfinite(value) and abs(value) > 0.65 else "#222222")
        ax.set_title(f"{title} (n={record['n']})")
    fig.suptitle(oof_title(rows), fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.80, bottom=0.22, wspace=0.27)
    cax = fig.add_axes([0.94, 0.28, 0.012, 0.43])
    fig.colorbar(im, cax=cax, label="Spearman ρ")
    fig.text(0.48, 0.035, "类内相关用于区分标签混合的影响；NA 表示常数或样本不足。预测相关不等于表示冗余或因果关系。",
             ha="center", fontsize=9)
    return fig


def build_transitions(rows, stats):
    """Fixed user-supplied threshold only, never selected on the test labels."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = list(stats["transitions"].values())
    ax.bar(range(4), values, color=[GREY, GREEN, VERMILION, GREY])
    ax.set_xticks(range(4), ["均判断正确", "融合纠正", "融合新增错误", "均判断错误"])
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.025, str(value), ha="center")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("样本数")
    ax.set_title(f"固定阈值 t={stats['threshold']:g}；p(error) ≥ t 判为错误")
    fig.suptitle(oof_title(rows), fontsize=10)
    fig.text(0.5, 0.025, "描述性阈值诊断；未经校准或代价验证，不等同于部署收益；不替代 AUROC 结果。", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    summary = commands.add_parser("summary", help="Plot audited frozen comparisons")
    summary.add_argument("--result", type=Path, required=True)
    summary.add_argument("--source-manifest", type=Path, required=True)
    oof = commands.add_parser("oof", help="Plot per-sample exported OOF probabilities")
    oof.add_argument("--input", type=Path, required=True)
    oof.add_argument("--repeat", type=int, help="Plot one repeat; default: each repeat separately")
    oof.add_argument("--threshold", type=float, help="Optional prechosen diagnostic threshold; never optimized")
    for child in (summary, oof):
        child.add_argument("--output-dir", type=Path, required=True,
                           help="New dedicated directory; must not exist")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir}: use a new output directory; no overwrite")
    configure_style()
    audit = []
    jobs = []
    sources = []
    if args.mode == "summary":
        result, digest = load_summary(args.result, args.source_manifest)
        sources = [args.result, args.source_manifest]
        jobs = [("block_comparison_bounds", lambda: build_block_bounds(result)),
                ("b4_dependency_comparison", lambda: build_b4_comparison(result))]
        audit.append({"kind": "frozen_summary_source", "role": "descriptive_visualization_of_frozen_statistics",
                      "source_sha256": digest,
                      "source_fields": "arms[path][arm]: auroc, delta_h_point, m1b_non_inferiority, m1b_iv_a_direct_superiority",
                      "plotted_paths": [PRIMARY, NO_B4]})
        for path in (PRIMARY, NO_B4):
            for arm in ("h1", "mcq"):
                row = result["arms"][path][arm]
                audit.append({"kind": "frozen_summary_values", "path": path, "arm": arm,
                              **{key: row[key] for key in ("n", "n_groups", "auroc", "delta_h_point",
                                  "m1b_non_inferiority", "m1b_iv_a_direct_superiority")}})
        print(f"[input] {args.result} sha256={digest} arms=h1,mcq paths=A,C")
    else:
        if args.threshold is not None and (not math.isfinite(args.threshold) or not 0 < args.threshold < 1):
            raise ValueError("--threshold must be finite and strictly between 0 and 1")
        records = read_jsonl(args.input)
        try:
            validate_stage1_diagnostic_oof(records)
        except ValueError as exc:
            raise ValueError(f"{args.input}: {exc}") from exc
        sources = [args.input]
        repeats = sorted({r["repeat"] for r in records})
        if args.repeat is not None:
            if args.repeat not in repeats:
                raise ValueError(f"repeat {args.repeat} not present in input")
            repeats = [args.repeat]
        for repeat in repeats:
            rows = sorted((r for r in records if r["repeat"] == repeat), key=lambda r: r["sample_id"])
            stats = describe_repeat(rows, args.threshold)
            audit.append(stats)
            jobs.extend([(f"repeat_{repeat}_score_shift", lambda rows=rows: build_score_shift(rows)),
                         (f"repeat_{repeat}_correlation", lambda rows=rows, stats=stats: build_correlation(rows, stats))])
            if args.threshold is not None:
                jobs.append((f"repeat_{repeat}_threshold_transitions", lambda rows=rows, stats=stats: build_transitions(rows, stats)))
        print(f"[input] {args.input} rows={len(records)} repeats={repeats} role={records[0]['data_role']}")
    hashes = {str(path.resolve()): file_sha256(path) for path in sources}
    # All validation completes before producing the first output.
    for stem, build in jobs:
        fig = build()
        try:
            paths = save_plot_pair(fig, args.output_dir, stem)
        finally:
            plt.close(fig)
        audit.append({"kind": "figure", "stem": stem,
                      "files": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in paths]})
        print(f"[figure] {paths[0]} | {paths[1]}")
    if hashes != {str(path.resolve()): file_sha256(path) for path in sources}:
        raise RuntimeError("input changed while plotting; do not use this output directory")
    audit.insert(0, {"kind": "provenance", "mode": args.mode, "input_sha256": hashes,
                     "script_sha256": file_sha256(Path(__file__)),
                     "limitations": "No refitting or confirmatory inference. OOF provenance is producer-declared; independently audit the exporter."})
    write_jsonl(audit, args.output_dir / "figure_audit.jsonl")
    print(f"[done] figures={len(jobs)} audit={args.output_dir / 'figure_audit.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
