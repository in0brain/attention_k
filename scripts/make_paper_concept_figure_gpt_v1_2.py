"""Draw the v1.2 observable-sufficiency concept figure.

The figure is conceptual and contains no empirical result. It separates the
population sufficiency claim from a finite-sample, method-specific increment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILION = "#D55E00"
GRAY = "#666666"
LIGHT_BLUE = "#EAF3F8"
LIGHT_GREEN = "#E7F4EF"
LIGHT_ORANGE = "#FBEDE7"


def configure_font() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return "DejaVu Sans"


def box(ax, xy, width, height, text, *, edge=GRAY, face="white", size=8.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#222222",
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, style="-", width=1.25):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        color=color,
        linestyle=style,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_wide(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.17, 0.94, "模型内部计算", ha="center", fontsize=9.3, weight="bold")
    box(ax, (0.04, 0.64), 0.26, 0.17, "内部状态  H\n高维、需额外访问", edge=SKY, face=LIGHT_BLUE, size=9.0)
    box(ax, (0.04, 0.32), 0.26, 0.17, "内部读出  H_m=g_m(H)\nprobe / 几何统计 / 其他方法", edge=BLUE, face="white", size=8.0)
    arrow(ax, (0.17, 0.64), (0.17, 0.49), color=BLUE)

    ax.text(0.49, 0.94, "模型原生行为接口", ha="center", fontsize=9.3, weight="bold")
    box(ax, (0.36, 0.64), 0.26, 0.17, "可观测信息  S\nlogits · 概率 · 熵 · 文本", edge=GREEN, face=LIGHT_GREEN, size=8.8)
    arrow(ax, (0.30, 0.725), (0.36, 0.725), color=GREEN)
    box(ax, (0.35, 0.32), 0.28, 0.17, "理论问题\n给定 S 后，H 还有信息吗？", edge=GRAY, face="white", size=8.5)
    arrow(ax, (0.49, 0.64), (0.49, 0.49), color=GREEN)

    ax.text(0.82, 0.94, "有限样本操作化", ha="center", fontsize=9.3, weight="bold")
    box(ax, (0.69, 0.64), 0.27, 0.17, "Delta_m = Perf(S+H_m)\n- Perf(S)\n同切分 · 同预测器族 · 折外", edge=BLUE, face=LIGHT_BLUE, size=7.1)
    arrow(ax, (0.63, 0.405), (0.69, 0.69), color=BLUE)
    arrow(ax, (0.62, 0.725), (0.69, 0.725), color=GREEN)

    box(ax, (0.68, 0.37), 0.28, 0.13, "增量越过实质界限且审计通过\n→ 方法特定的剩余预测价值", edge=GREEN, face="white", size=7.5)
    box(ax, (0.68, 0.17), 0.28, 0.13, "区间宽 / 条件失效\n→ 不确定，不等于充分", edge=VERMILION, face=LIGHT_ORANGE, size=7.5)
    arrow(ax, (0.825, 0.64), (0.825, 0.50), color=GREEN)
    arrow(ax, (0.86, 0.64), (0.86, 0.30), color=VERMILION, style="--")

    box(
        ax,
        (0.04, 0.08),
        0.58,
        0.13,
        "识别条件：C1 可观测边界 · C2 基线强度 · C3 捷径\nC4 泛化单位 · C5 数值有效性 · C6 分析冻结",
        edge=GRAY,
        face="#F5F5F5",
        size=7.5,
    )
    arrow(ax, (0.62, 0.145), (0.68, 0.225), color=GRAY, style="--")
    ax.text(
        0.49,
        0.015,
        "H_m 单独低于 S 不等于零增量；未检出增量不等于已证明条件充分",
        ha="center",
        fontsize=8.2,
        color=VERMILION,
        weight="bold",
    )
    save(fig, output_dir, "fig0_observable_sufficiency_wide_gpt_v1_2")


def draw_single(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 5.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.08, 0.82), 0.36, 0.11, "内部状态 H", edge=SKY, face=LIGHT_BLUE, size=9.0)
    box(ax, (0.56, 0.82), 0.36, 0.11, "可观测信息 S", edge=GREEN, face=LIGHT_GREEN, size=9.0)
    arrow(ax, (0.44, 0.875), (0.56, 0.875), color=GREEN)
    box(ax, (0.08, 0.65), 0.36, 0.10, "内部读出 H_m\n方法特定", edge=BLUE, size=8.2)
    arrow(ax, (0.26, 0.82), (0.26, 0.75), color=BLUE)
    box(ax, (0.14, 0.48), 0.72, 0.105, "理论：给定 S 后，H 还有信息吗？\n经验：Delta_m = Perf(S+H_m) - Perf(S)", edge=GRAY, size=7.9)
    arrow(ax, (0.26, 0.65), (0.41, 0.585), color=BLUE)
    arrow(ax, (0.74, 0.82), (0.59, 0.585), color=GREEN)
    box(ax, (0.08, 0.27), 0.84, 0.145, "C1 可观测边界 · C2 基线强度 · C3 捷径\nC4 泛化单位 · C5 数值有效性 · C6 分析冻结", edge=GRAY, face="#F5F5F5", size=7.4)
    arrow(ax, (0.50, 0.48), (0.50, 0.415), color=GRAY)
    box(ax, (0.08, 0.10), 0.38, 0.10, "审计通过且有精度\n→ 剩余预测价值", edge=GREEN, size=7.3)
    box(ax, (0.54, 0.10), 0.38, 0.10, "条件或精度不足\n→ 不确定", edge=VERMILION, face=LIGHT_ORANGE, size=7.3)
    arrow(ax, (0.38, 0.27), (0.27, 0.20), color=GREEN)
    arrow(ax, (0.62, 0.27), (0.73, 0.20), color=VERMILION, style="--")
    ax.text(0.5, 0.035, "未检出增量不自动证明充分", ha="center", fontsize=8.2, color=VERMILION, weight="bold")
    save(fig, output_dir, "fig0_observable_sufficiency_single_gpt_v1_2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage0/figures"),
    )
    parser.add_argument("--layout", choices=("wide", "single", "all"), default="all")
    args = parser.parse_args()
    font = configure_font()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    if args.layout in ("wide", "all"):
        draw_wide(args.output_dir)
    if args.layout in ("single", "all"):
        draw_single(args.output_dir)
    print(f"[concept-figure-v1.2] font={font}")
    print(f"[concept-figure-v1.2] output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
