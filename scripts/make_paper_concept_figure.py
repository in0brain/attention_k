"""Draw the paper's identification-conditions concept figure.

This figure contains no empirical result. It visualizes only frozen method
definitions from docs/paper/framing.md and identification_audit.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILION = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#F4F4F4"
LIGHT_BLUE = "#EAF3F8"
LIGHT_PURPLE = "#F2EEF7"


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


def draw_box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    edge=GRAY,
    face="white",
    fontsize=8,
    linewidth=1.1,
    radius=0.012,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
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
        fontsize=fontsize,
        color="#222222",
        linespacing=1.25,
    )
    return patch


def draw_arrow(ax, start, end, *, color=GRAY, linestyle="-", linewidth=1.2):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    return arrow


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_wide(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.05, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        r"禁止直接跳跃：$\Delta_H>0$  $\nRightarrow$  模型“知道”、幻觉减少或干预有效",
        ha="center",
        va="top",
        fontsize=8.6,
        color=VERMILION,
        weight="bold",
    )

    ax.text(0.13, 0.865, "A  条件增量估计", ha="center", fontsize=9, weight="bold")
    ax.text(0.50, 0.865, "B  操作性条件审计", ha="center", fontsize=9, weight="bold")
    ax.text(0.86, 0.865, "C  结论许可", ha="center", fontsize=9, weight="bold")

    draw_box(ax, (0.025, 0.67), 0.085, 0.085, r"$O$", edge=GREEN, face="#E7F4EF", fontsize=11)
    draw_box(ax, (0.025, 0.53), 0.085, 0.085, r"$H$", edge=SKY, face="#EAF6FB", fontsize=11)
    draw_box(ax, (0.145, 0.67), 0.09, 0.085, r"$O+H$", edge=BLUE, face=LIGHT_BLUE, fontsize=10)
    draw_box(
        ax,
        (0.055, 0.35),
        0.18,
        0.10,
        r"$\Delta_H=S(O+H)-S(O)$",
        edge=BLUE,
        face="white",
        fontsize=8.5,
    )
    ax.text(0.125, 0.49, r"$H$ 单独报告", fontsize=7.2, ha="center", color=GRAY)
    draw_arrow(ax, (0.11, 0.712), (0.145, 0.712), color=GREEN)
    draw_arrow(ax, (0.11, 0.572), (0.145, 0.69), color=SKY)
    draw_arrow(ax, (0.19, 0.67), (0.17, 0.45), color=BLUE)
    draw_arrow(ax, (0.075, 0.67), (0.10, 0.45), color=GREEN)
    ax.plot([0.075, 0.12], [0.53, 0.49], color=SKY, linestyle="--", linewidth=1)

    ax.add_patch(
        FancyBboxPatch(
            (0.285, 0.52),
            0.39,
            0.28,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=LIGHT_BLUE,
            edgecolor=BLUE,
            linewidth=1.0,
        )
    )
    ax.text(0.48, 0.77, "Tier I｜识别层", ha="center", fontsize=8.3, weight="bold", color=BLUE)
    tier1 = [
        ((0.305, 0.64), "C1  输出范围已指定\n冻结单一 $O$"),
        ((0.495, 0.64), "C2  $O$ 不被支配\n检查输出阶梯"),
        ((0.305, 0.54), "C3  完整输出优于捷径\n单侧非优越性"),
        ((0.495, 0.54), "C4  分组阻断标签来源\n决定对象不得跨折"),
    ]
    for xy, text in tier1:
        draw_box(ax, xy, 0.16, 0.075, text, face="white", fontsize=6.9)

    ax.add_patch(
        FancyBboxPatch(
            (0.285, 0.27),
            0.39,
            0.19,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=LIGHT_PURPLE,
            edgecolor=GRAY,
            linewidth=1.0,
        )
    )
    ax.text(0.48, 0.43, "Tier II｜有效性与完整性", ha="center", fontsize=8.2, weight="bold")
    draw_box(ax, (0.305, 0.30), 0.16, 0.085, "C5  数值与后端有效\n两臂后端一致", face="white", fontsize=6.9)
    draw_box(ax, (0.495, 0.30), 0.16, 0.085, "C6  冻结且只揭盲一次\n哈希锁与审计日志", face="white", fontsize=6.9)
    draw_arrow(ax, (0.235, 0.40), (0.285, 0.52), color=BLUE)

    diamond = Polygon(
        [(0.72, 0.59), (0.77, 0.68), (0.82, 0.59), (0.77, 0.50)],
        closed=True,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.1,
    )
    ax.add_patch(diamond)
    ax.text(0.77, 0.59, "最低门槛\n通过？", ha="center", va="center", fontsize=7.2)
    draw_arrow(ax, (0.675, 0.66), (0.72, 0.62), color=GRAY)
    draw_arrow(ax, (0.675, 0.36), (0.72, 0.56), color=GRAY)

    draw_box(
        ax,
        (0.845, 0.65),
        0.14,
        0.13,
        "允许的有限读法\n相对于预注册 $O$\n的条件增量",
        edge=GREEN,
        face="white",
        fontsize=7.2,
    )
    draw_box(
        ax,
        (0.845, 0.43),
        0.14,
        0.12,
        "条件已知失效\n结论降级为\n受限描述",
        edge=VERMILION,
        face="white",
        fontsize=7.0,
    )
    draw_box(
        ax,
        (0.845, 0.22),
        0.14,
        0.12,
        "条件或精度不足\n不确定\n≠ 无增量",
        edge=GRAY,
        face="white",
        fontsize=7.0,
    )
    draw_arrow(ax, (0.82, 0.62), (0.845, 0.70), color=GREEN)
    draw_arrow(ax, (0.80, 0.54), (0.845, 0.49), color=VERMILION, linestyle="--")
    draw_arrow(ax, (0.77, 0.50), (0.845, 0.30), color=GRAY, linestyle="--")
    ax.text(0.5, 0.08, "C1–C6 是本协议的操作性条件", ha="center", fontsize=8.3, weight="bold")

    save(fig, output_dir, "fig0_identification_conditions_wide_gpt_v1")


def draw_single(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.985,
        r"$\Delta_H>0$ 不蕴含模型“知道”或干预有效",
        ha="center",
        va="top",
        fontsize=7.8,
        color=VERMILION,
        weight="bold",
    )
    draw_box(ax, (0.12, 0.82), 0.76, 0.095, r"$\Delta_H=S(O+H)-S(O)$" + "\n" + r"$H$ 单独报告但不进入差值", edge=BLUE, face=LIGHT_BLUE, fontsize=8.2)
    draw_arrow(ax, (0.50, 0.82), (0.50, 0.76), color=BLUE)
    draw_box(ax, (0.08, 0.50), 0.84, 0.25, "Tier I｜识别层\n\nC1  输出范围已指定　　C2  $O$ 不被支配\n\nC3  完整输出优于捷径　C4  分组阻断标签来源", edge=BLUE, face=LIGHT_BLUE, fontsize=7.2)
    draw_arrow(ax, (0.50, 0.50), (0.50, 0.45), color=GRAY)
    draw_box(ax, (0.08, 0.31), 0.84, 0.13, "Tier II｜有效性与完整性\n\nC5  数值与后端有效　　C6  冻结且只揭盲一次", edge=GRAY, face=LIGHT_PURPLE, fontsize=7.2)
    draw_arrow(ax, (0.50, 0.31), (0.50, 0.26), color=GRAY)
    draw_box(
        ax,
        (0.10, 0.17),
        0.80,
        0.08,
        "全部通过：仅允许\n“相对于预注册 $O$ 的条件增量”",
        edge=GREEN,
        fontsize=6.7,
    )
    draw_box(
        ax,
        (0.10, 0.07),
        0.80,
        0.07,
        "失效或精度不足：结论降级 / 不确定\n（≠ 无增量）",
        edge=VERMILION,
        fontsize=6.3,
    )
    ax.text(0.5, 0.025, "对可审计内部证据主张必要而非充分", ha="center", fontsize=8, weight="bold")
    save(fig, output_dir, "fig0_identification_conditions_single_gpt_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage0/figures"),
        help="Directory for PDF and PNG files.",
    )
    parser.add_argument(
        "--layout",
        choices=("wide", "single", "all"),
        default="all",
        help="Layout variant to render.",
    )
    args = parser.parse_args()
    font = configure_font()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    if args.layout in ("wide", "all"):
        draw_wide(args.output_dir)
    if args.layout in ("single", "all"):
        draw_single(args.output_dir)
    print(f"[concept-figure] font={font}")
    print(f"[concept-figure] output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
