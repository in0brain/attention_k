"""Build the Stage 1 E1 paper figure from the frozen confirmatory result.

The script reads every plotted statistic from the confirmatory JSON.  It does
not recompute an estimand and does not contain a second copy of the reported
numbers.  The output is a vector PDF for LaTeX plus a PNG for visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PATHS = (
    ("A", "frozen_implementation_primary"),
    ("B", "stable_shadow_sensitivity"),
    ("C", "no_B4_dependency_sensitivity"),
)
ARMS = ("h1", "mcq")

# Okabe--Ito colour-blind-safe palette.  Marker shape remains the primary
# distinction so the figure is also readable in greyscale.
GREY = "#777777"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILION = "#D55E00"
GRID = "#D9D9D9"
INK = "#222222"
PATH_C_SHADE = "#F4F4F4"


class E1Row(NamedTuple):
    path: str
    arm: str
    s_star: float
    hidden: float
    fused: float
    delta: float
    ci_lo: float
    ci_hi: float
    precision: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_source(result_path: Path, source_manifest: Path | None) -> None:
    if source_manifest is None:
        return
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected = manifest.get("result", {}).get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("source manifest lacks result.sha256")
    actual = _sha256(result_path)
    if actual != expected:
        raise ValueError(
            "confirmatory result SHA does not match the paper source manifest: "
            f"expected={expected}, actual={actual}"
        )


def load_e1_rows(
    result_path: Path,
    source_manifest: Path | None = None,
) -> list[E1Row]:
    """Load and validate the six plotted E1 rows without recomputing results."""

    _verify_source(result_path, source_manifest)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("contract", {}).get("ok") is not True:
        raise ValueError("confirmatory result contract.ok is not true")
    expected_paths = [key for _, key in PATHS]
    if result.get("frozen_paths") != expected_paths:
        raise ValueError("confirmatory frozen_paths do not match the registered A/B/C order")

    rows: list[E1Row] = []
    for path_label, path_key in PATHS:
        for arm in ARMS:
            record = result["arms"][path_key][arm]
            auroc = record["auroc"]
            ci = record["delta_h_ci"]
            row = E1Row(
                path=path_label,
                arm=arm.upper(),
                s_star=float(auroc["S*"]),
                hidden=float(auroc["H"]),
                fused=float(auroc["S*+H"]),
                delta=float(record["delta_h_point"]),
                ci_lo=float(ci["ci_lo"]),
                ci_hi=float(ci["ci_hi"]),
                precision=str(record["precision"]["precision"]),
            )
            _validate_row(row)
            rows.append(row)
    return rows


def _validate_row(row: E1Row) -> None:
    numeric = row.s_star, row.hidden, row.fused, row.delta, row.ci_lo, row.ci_hi
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"non-finite E1 value in {row.path}/{row.arm}")
    if not all(0.0 <= value <= 1.0 for value in (row.s_star, row.hidden, row.fused)):
        raise ValueError(f"AUROC outside [0,1] in {row.path}/{row.arm}")
    if not row.ci_lo <= row.delta <= row.ci_hi:
        raise ValueError(f"delta point is outside its CI in {row.path}/{row.arm}")
    if not math.isclose(row.fused - row.s_star, row.delta, abs_tol=1e-12):
        raise ValueError(f"delta does not equal AUROC(S*+H)-AUROC(S*) in {row.path}/{row.arm}")


def _configure_fonts() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
    family = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.0,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": "#555555",
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_figure(rows: list[E1Row]):
    """Return the two-panel figure with a genuine horizontal broken axis."""

    if len(rows) != 6:
        raise ValueError(f"expected six E1 rows, got {len(rows)}")
    _configure_fonts()

    fig = plt.figure(figsize=(11.2, 4.75), dpi=160)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.78, 1.18, 0.48),
        left=0.105,
        right=0.985,
        bottom=0.16,
        top=0.80,
        wspace=0.075,
    )
    ax_auroc = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1], sharey=ax_auroc)
    ax_far = fig.add_subplot(grid[0, 2], sharey=ax_auroc)
    axes = (ax_auroc, ax_delta, ax_far)
    y_positions = list(reversed(range(len(rows))))

    for ax in axes:
        ax.set_ylim(-0.5, 5.5)
        ax.axhspan(-0.5, 1.5, color=PATH_C_SHADE, zorder=0)
        for boundary in (1.5, 3.5):
            ax.axhline(boundary, color=GRID, lw=0.8, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.65, alpha=0.8, zorder=0)

    # Left panel: a small vertical dodge retains one categorical row while
    # preventing nearly identical AUROC markers from erasing one another.
    marker_specs = (
        ("s_star", 0.11, "o", GREY, "none", r"组合基线 $S^*$"),
        ("hidden", 0.00, "^", GREEN, "none", r"隐藏状态 $H$"),
        ("fused", -0.11, "o", BLUE, BLUE, r"融合模型 $S^*+H$"),
    )
    for row, y in zip(rows, y_positions):
        for field, offset, marker, edge, face, _ in marker_specs:
            ax_auroc.scatter(
                getattr(row, field),
                y + offset,
                marker=marker,
                s=64,
                facecolors=face,
                edgecolors=edge,
                linewidths=1.5,
                zorder=4,
            )

    ax_auroc.set_xlim(0.49, 1.005)
    ax_auroc.set_xticks((0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    ax_auroc.set_xlabel("AUROC")
    ax_auroc.set_title("三点 AUROC", fontsize=11, pad=10)
    ax_auroc.set_yticks(y_positions)
    ax_auroc.set_yticklabels([f"{row.path} / {row.arm}" for row in rows], fontsize=9.5)
    ax_auroc.tick_params(axis="y", length=0, pad=8)

    c_mcq = rows[-1]
    annotation_x = min(c_mcq.hidden, c_mcq.fused)
    ax_auroc.annotate(
        r"$S^*+H < H$",
        xy=(annotation_x, y_positions[-1] - 0.04),
        xytext=(-55, 24),
        textcoords="offset points",
        fontsize=8.2,
        color=VERMILION,
        arrowprops={"arrowstyle": "-", "color": VERMILION, "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": VERMILION, "lw": 0.7},
        zorder=6,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markeredgecolor=edge,
            markerfacecolor=face,
            markeredgewidth=1.5,
            markersize=7.5,
            label=label,
        )
        for _, _, marker, edge, face, label in marker_specs
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.43, 0.965),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
        fontsize=9.2,
    )

    # Right panel: five small increments use the left axis and the C/MCQ
    # diagnostic occupies the separate high-range axis.
    ax_delta.set_xlim(-0.0025, 0.045)
    ax_delta.set_xticks((0.0, 0.01, 0.02, 0.03, 0.04))
    ax_far.set_xlim(0.215, 0.300)
    ax_far.set_xticks((0.22, 0.25, 0.28))
    for row, y in zip(rows, y_positions):
        for ax, lo, hi in ((ax_delta, -0.0025, 0.045), (ax_far, 0.215, 0.300)):
            visible_lo = max(row.ci_lo, lo)
            visible_hi = min(row.ci_hi, hi)
            if visible_lo <= visible_hi:
                ax.plot([visible_lo, visible_hi], [y, y], color=BLUE, lw=1.8, zorder=3)
                ax.plot([visible_lo, visible_lo], [y - 0.07, y + 0.07], color=BLUE, lw=1.2)
                ax.plot([visible_hi, visible_hi], [y - 0.07, y + 0.07], color=BLUE, lw=1.2)
            if lo <= row.delta <= hi:
                ax.scatter(row.delta, y, s=46, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)

        if row.delta <= 0.045:
            ax_delta.annotate(
                f"{row.delta:+.4f}",
                (row.delta, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.6,
                color="#444444",
            )
        else:
            label = f"{row.delta:+.4f}  精度不足" if row.precision != "adequate" else f"{row.delta:+.4f}"
            ax_far.annotate(
                label,
                (row.delta, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
                color=VERMILION if row.precision != "adequate" else "#444444",
            )

    ax_delta.axvline(0.0, color="#888888", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax_delta.set_xlabel(r"条件增量 $\Delta_H$")
    ax_far.set_xlabel("")
    fig.text(0.78, 0.823, r"条件增量 $\Delta_H$（95% CI）", ha="center", fontsize=11)

    for ax in (ax_delta, ax_far):
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax_delta.spines["right"].set_visible(False)
    ax_far.spines["left"].set_visible(False)
    ax_far.tick_params(axis="y", left=False)

    # Conventional diagonal marks indicate one horizontal-axis break.
    kwargs = {"color": "#555555", "clip_on": False, "lw": 1.0}
    size = 0.018
    ax_delta.plot((1 - size, 1 + size), (-size, +size), transform=ax_delta.transAxes, **kwargs)
    ax_far.plot((-size, +size), (-size, +size), transform=ax_far.transAxes, **kwargs)

    fig.text(
        0.985,
        0.055,
        "灰色区域为路径 C（移除 B4）的依赖性敏感性分析。",
        ha="right",
        fontsize=7.6,
        color="#555555",
    )
    return fig


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (output_dir / f"{stem}.pdf", output_dir / f"{stem}.png")
    formats = ("pdf", "png")
    metadata = (
        {"Creator": "make_stage1_e1_figure.py", "CreationDate": None, "ModDate": None},
        {"Software": "make_stage1_e1_figure.py"},
    )
    for target, fmt, meta in zip(targets, formats, metadata):
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            fig.savefig(temporary, format=fmt, bbox_inches="tight", metadata=meta)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    plt.close(fig)
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="figure_02_e1_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_e1_rows(args.result, args.source_manifest)
    figure = build_figure(rows)
    pdf_path, png_path = save_figure(figure, args.output_dir, args.stem)
    print(f"[figure] rows={len(rows)} source_sha256={_sha256(args.result)}")
    print(f"[figure] pdf={pdf_path} png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
