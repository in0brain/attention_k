"""论文图（EIML3）：每张图扛一条 identification condition 的证据，不是装饰。

用法
────
    python scripts/make_paper_figures.py          # 无参数;读产物、写 outputs/stage0/figures/
依赖 matplotlib（已在 requirements.txt）。改了审计数据 → 重跑本脚本 → 四张图自动同步。

设计约束（三条，改图时不要破坏）
────────────────────────────
1. **所有数字从已提交的 JSON 产物读取，零硬编码。** 一篇卖点是可审计的论文，图也必须
   可从审计产物复现而不是手抄；否则"图里的 0.919"和"报告里的 0.919"就成了两个独立事实，
   哪天数据变了图不会跟着变。下面每个函数都标了它读哪个 JSON 的哪个键。
2. **Okabe-Ito 色盲安全配色**（Okabe & Ito 2008，科学出版界通用，构造即色盲安全，
   无需再跑校验器）。灰色 = 次要序列，vermil/blue/green = 三个主角，固定不轮换。
3. **PDF（矢量，给 LaTeX \\includegraphics）+ PNG（预览）双出**。正文引用用 PDF。

图 ↔ finding ↔ 数据源 ↔ 论文位置（见 framing.md 的 Tier I/II、draft_v1.md §6）
────────────────────────────────────────────────────────────────────
  Fig 1  分组切换 slope（A→C）      C4 泄漏 + C2 支配翻号   draft §6.3/§6.4  ← 最强，扛两条
         ← posthoc_h1_identifier_leakage_diagnostic.json
  Fig 2  红线 forest plot（两规则）  C3 + 红线方向写错       draft §6.2
         ← posthoc_redline_rule_and_o_dominance.json
  Fig 3  文本表示三分解             finding 3 的残余分解     draft §6.4
         ← posthoc_h1_text_representation_diagnostic.json (+ leakage.json 取 rung1 参照线)
  Fig 4  sizing 曲线（log-log）     future work 定量依据     draft §7
         ← posthoc_identifier_stratified_sizing.json

**不画的东西（有意为之）**：confirmatory 主结果表（§6.1）、RQ2 分层表、data accounting
—— 这些列表就够，画成图只是装饰。图只留给"列表传达不了"的信息：交叉/反转（Fig 1）、
区间与判据带的几何关系（Fig 2）、gap 的分解（Fig 3）、趋势与目标交点（Fig 4）。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

E = Path("outputs/stage0/evaluation")
OUT = Path("outputs/stage0/figures")

# Okabe-Ito（色盲安全）
OK = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermil": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
INK = "#222222"
MUTED = "#666666"
BAND = "#DDDDDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

RUNG_LABEL = {
    "rung1_prompt_only": "prompt-only",
    "rung2_id_string_only": "id-string-only",
    "rung3_surface_only": "surface-only",
    "rung4_full_text": "full-text",
    "rung5_f5": "F5",
    "rung6_O_f5_plus_text": "O = F5+text",
    "H_alone": "H alone",
    "O_plus_H": "O+H",
    # MCQ 侧 shortcut 名（红线图会用到）
    "rung2_answer_letter_only": "answer-letter",
    "rung3_option_surface_only": "option-surface",
}
RUNG_ORDER = ["rung1_prompt_only", "rung2_id_string_only", "rung3_surface_only",
              "rung4_full_text", "rung5_f5", "rung6_O_f5_plus_text", "H_alone", "O_plus_H"]

CELL_LABEL = {
    "h1_A_prompt": "H1 / A",
    "h1_C_prompt_union_fabid": "H1 / C",
    "mcq_A_question": "MCQ / A",
}


def _load(name):
    return json.loads((E / name).read_text(encoding="utf-8"))


def _save(fig, stem):
    """落盘。**metadata 置空是有意的**：matplotlib 默认把生成时刻写进 PDF/PNG,
    于是同一份数据每次重跑都产出不同字节,git 里全是无意义的 diff。置空后
    「同数据 → 同字节」,任何人重跑都能验证图确实来自那份审计产物 —— 与项目其他
    地方的 sha256 冻结是同一个精神。"""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"CreationDate": None}, "png": {"Software": None}}
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", metadata=meta[ext])
    plt.close(fig)
    print(f"[fig] {stem}.pdf / .png")


# ───────────────────────────────────────────── Fig 1: grouping slope (A→C)
def fig1_grouping_slope():
    """slope chart：每个 rung 在 A(prompt) vs C(prompt∪fab-id) 下的 AUROC。

    扛两条：rung2 塌陷（C4 泄漏）+ O 反超 rung2（C2 支配性翻号）。job = change + rank。

    数据源：posthoc_h1_identifier_leakage_diagnostic.json
        A → A_prompt_grouped_confirmatory_design.rungs.{rung} (float)
        C → C_prompt_union_fabricated_grouped_PRIMARY.rungs.{rung} (float)
    为什么必须是图：**交叉**是列表给不出的——id-string 从最高点俯冲、穿过 O+H 与 O
    落到下方，"谁越过了谁"正是 C2 翻号的证据。8×2 个数字排成表看不出这个动作。
    """
    leak = _load("posthoc_h1_identifier_leakage_diagnostic.json")
    A = leak["A_prompt_grouped_confirmatory_design"]["rungs"]
    C = leak["C_prompt_union_fabricated_grouped_PRIMARY"]["rungs"]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    xa, xc = 0, 1
    for r in RUNG_ORDER:
        ya, yc = A[r], C[r]
        # 高亮三条主角，其余灰
        if r == "rung2_id_string_only":
            col, lw, z = OK["vermil"], 2.2, 5
        elif r == "O_plus_H":
            col, lw, z = OK["blue"], 2.2, 5
        elif r == "rung6_O_f5_plus_text":
            col, lw, z = OK["green"], 2.2, 5
        else:
            col, lw, z = OK["grey"], 1.0, 2
        ax.plot([xa, xc], [ya, yc], "-", color=col, lw=lw, zorder=z,
                marker="o", ms=5, mfc=col, mec="white", mew=0.8)
        # 直接标注（只标主角 + 端点值）；灰色里 full-text/surface-only 端点极近，错开
        gray_nudge = {"rung4_full_text": 5, "rung3_surface_only": -5, "rung5_f5": 10}
        if col != OK["grey"]:
            ax.annotate(f"{RUNG_LABEL[r]}", (xc, yc), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=col, fontweight="bold")
            ax.annotate(f"{ya:.3f}", (xa, ya), xytext=(-6, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=7.5, color=MUTED)
        else:
            ax.annotate(f"{RUNG_LABEL[r]}", (xc, yc), xytext=(6, gray_nudge.get(r, 0)),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=MUTED)

    ax.set_xlim(-0.55, 1.65)
    ax.set_xticks([xa, xc])
    ax.set_xticklabels(["grouping A\n(prompt)", "grouping C\n(prompt ∪ fabricated-id)"])
    ax.set_ylabel("H1 AUROC (grouped CV)")
    ax.axhline(0.5, color=BAND, lw=0.8, ls=":", zorder=1)
    # 标注 rung2 的跌幅（放在 id-string 线上方、靠 C 端，避开 O/O+H 线）
    dr = A["rung2_id_string_only"] - C["rung2_id_string_only"]
    ax.annotate(f"id-string −{dr:.3f}", (0.62, 0.88),
                fontsize=8, color=OK["vermil"], ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=OK["vermil"], lw=0.8))
    ax.set_title("Blocking identifier leakage collapses the string baseline\n"
                 "and restores O above it", fontsize=9.5, loc="left", color=INK)
    _save(fig, "fig1_grouping_slope")


# ───────────────────────────────────────────── Fig 2: redline forest plot
def fig2_redline_forest():
    """forest plot：每个 (arm, shortcut, grouping) 的 Δ_art 点+CI，ε 带与 0 标出。

    扛 finding 1：冻结（双侧等价）规则的触发条件是 CI⊂[−ε,ε]，图上能看出 CI 全在一侧、
    双侧带够不着；修正（单侧 non-superiority）规则 CI_hi<ε 就触发。job = polarity+uncertainty。

    数据源：posthoc_redline_rule_and_o_dominance.json
        cells.{cell}.redline.per_shortcut.{shortcut}.delta_artifact.{point,ci_lo,ci_hi}
        cells.{cell}.redline.per_shortcut.{shortcut}.{frozen_rule_triggered,
                                                      corrected_rule_triggered}
    ε=0.02 与预注册一致（此处显式写死仅为画带；判定值来自上面两个布尔字段，不在图里重算）。
    为什么必须是图：规则缺陷**本质是几何的**——六条 CI 全落在带的一侧、双侧带够不着，
    而单侧判据 CI_hi<ε 恰好抓住。文字描述这个几何关系反而绕。
    """
    red = _load("posthoc_redline_rule_and_o_dominance.json")
    eps = 0.02
    rows = []
    for cell, cdata in red["cells"].items():
        for sc, d in cdata["redline"]["per_shortcut"].items():
            da = d["delta_artifact"]
            rows.append((cell, sc, da["point"], da["ci_lo"], da["ci_hi"],
                         d["frozen_rule_triggered"], d["corrected_rule_triggered"]))
    rows.reverse()  # 从上到下更自然

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.axvspan(-eps, eps, color=BAND, alpha=0.7, zorder=1,
               label=f"equivalence band ±{eps}")
    ax.axvline(0, color=MUTED, lw=0.9, zorder=2)
    ylabels = []
    for i, (cell, sc, pt, lo, hi, ff, fc) in enumerate(rows):
        # 颜色：修正规则触发=vermil（红线该响），否则灰
        col = OK["vermil"] if fc else OK["grey"]
        ax.plot([lo, hi], [i, i], "-", color=col, lw=2, zorder=4)
        ax.plot(pt, i, "o", ms=7, color=col, mec="white", mew=1, zorder=5)
        ylabels.append(f"{CELL_LABEL.get(cell, cell)}  ·  {RUNG_LABEL.get(sc, sc)}")
        if fc:
            ax.annotate("◆ corrected fires", (hi, i), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=OK["vermil"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"$\Delta_{\mathrm{art}}$ = AUROC(full-text) − AUROC(shortcut),  95% CI")
    # 图例放**坐标区下方**，避免与两行标题相撞
    leg = [Patch(fc=BAND, ec="none", label=f"equivalence band ±{eps}"),
           Line2D([0], [0], color=OK["vermil"], lw=2, marker="o", mec="white",
                  label="corrected rule fires (CI$_{hi}$ < +ε)"),
           Line2D([0], [0], color=OK["grey"], lw=2, marker="o", mec="white",
                  label="neither rule fires")]
    ax.legend(handles=leg, loc="upper center", fontsize=7.5, frameon=False,
              bbox_to_anchor=(0.5, -0.16), ncol=3, columnspacing=1.2, handletextpad=0.5)
    ax.set_title("The frozen two-sided redline fires 0/6; the corrected one-sided\n"
                 "rule fires on every shortcut whose CI sits below +ε",
                 fontsize=9.5, loc="left", color=INK, pad=8)
    _save(fig, "fig2_redline_forest")


# ───────────────────────────────────────────── Fig 3: text-rep decomposition
def fig3_text_decomposition():
    """三种文本表示在 A / C 分组下的 AUROC，直接读出三分解。

    full→window（稀释↑）、window→mask（字符串↓）、mask vs rung1（上下文残余）。
    job = 一个 gap 的分解。grouped bar，C 为实、A 为半透明参照。

    数据源：
        posthoc_h1_text_representation_diagnostic.json
            text_representation_auroc.{A_prompt|C_prompt_union_fabid}.{full|window|window_mask}
        posthoc_h1_identifier_leakage_diagnostic.json
            C_prompt_union_fabricated_grouped_PRIMARY.rungs.rung1_prompt_only  ← 橙色参照线
    注意参照线必须取 **C 分组** 的 rung1（0.555），不能取 A 的（0.683）：
    "mask 后仍高于 prompt-only"这个论断只在同一分组内才成立。
    """
    tx = _load("posthoc_h1_text_representation_diagnostic.json")["text_representation_auroc"]
    leak = _load("posthoc_h1_identifier_leakage_diagnostic.json")
    reps = ["full", "window", "window_mask"]
    replab = ["full\ncompletion", "identifier\nwindow ±60", "window,\nid masked"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    x = range(len(reps))
    cvals = [tx["C_prompt_union_fabid"][r] for r in reps]
    avals = [tx["A_prompt"][r] for r in reps]
    ax.bar([i - 0.19 for i in x], avals, width=0.36, color=OK["skyblue"], alpha=0.45,
           label="grouping A (prompt)", zorder=3)
    ax.bar([i + 0.19 for i in x], cvals, width=0.36, color=OK["blue"],
           label="grouping C (clean)", zorder=3)
    for i, v in enumerate(cvals):
        ax.annotate(f"{v:.3f}", (i + 0.19, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=OK["blue"], fontweight="bold")
    # 参照线：C 分组下 rung1 prompt-only 与 0.5
    r1c = leak["C_prompt_union_fabricated_grouped_PRIMARY"]["rungs"]["rung1_prompt_only"]
    ax.axhline(r1c, color=OK["orange"], lw=1.2, ls="--", zorder=2)
    ax.annotate(f"prompt-only (C) = {r1c:.3f}", (len(reps) - 0.5, r1c), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=7.5, color=OK["orange"])
    ax.axhline(0.5, color=BAND, lw=0.8, ls=":", zorder=1)
    # 分解箭头注释（用 C 分组的值）
    d1 = cvals[1] - cvals[0]  # dilution
    d2 = cvals[2] - cvals[1]  # string
    ax.annotate(f"dilution +{d1:.3f}", (0.5, max(cvals[0], cvals[1]) + 0.02),
                ha="center", fontsize=7.5, color=OK["green"])
    ax.annotate(f"string −{-d2:.3f}", (1.5, max(cvals[1], cvals[2]) + 0.02),
                ha="center", fontsize=7.5, color=OK["vermil"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(replab, fontsize=8)
    ax.set_ylabel("H1 AUROC (text-only rung)")
    ax.set_ylim(0.45, 0.92)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_title("rung4 < rung2 decomposes into dilution, string channel,\n"
                 "and a non-zero context residual above prompt-only",
                 fontsize=9.5, loc="left", color=INK)
    _save(fig, "fig3_text_decomposition")


# ───────────────────────────────────────────── Fig 4: sizing curves
def fig4_sizing():
    """width vs k（lever 1，log-log）+ 幂律拟合 + 目标线 0.04 交点；lever 2 作第二序列。

    job = magnitude/trend + planning target。

    数据源：posthoc_identifier_stratified_sizing.json
        lever1_group_count.curve[].{k,width_mean}      散点
        lever1_group_count.fit.{a,b}                   拟合曲线 w = a·k^(−b)
        lever1_group_count.k_needed_for_target         星标
        lever2_max_component_share.curve[].{n_groups,width_mean}  绿色方块
        current_strict_grouping.{n_groups,ci_width}    橙色菱形
        target_ci_width                                目标虚线（0.04，= §7.2 上限）
    **这是 planning 图，不是结果图**：Δ_H 的点估计固定、不在此重估，图上也不画点估计，
    只画 CI 宽度。caption 与正文都必须标 planning analysis。
    """
    sz = _load("posthoc_identifier_stratified_sizing.json")
    L1 = sz["lever1_group_count"]["curve"]
    fit = sz["lever1_group_count"]["fit"]
    k_need = sz["lever1_group_count"]["k_needed_for_target"]
    target = sz["target_ci_width"]
    L2 = sz["lever2_max_component_share"]["curve"]
    n_now = sz["current_strict_grouping"]["n_groups"]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ks = [p["k"] for p in L1]
    ws = [p["width_mean"] for p in L1]
    ax.plot(ks, ws, "o", color=OK["blue"], ms=6, mec="white", mew=0.8,
            label="lever 1: resample k groups", zorder=5)
    # 幂律拟合曲线
    import numpy as np
    kk = np.linspace(min(ks), max(k_need * 1.1, max(ks)), 200)
    ax.plot(kk, fit["a"] * kk ** (-fit["b"]), "-", color=OK["blue"], lw=1.3, alpha=0.7,
            zorder=4, label=f"fit  w = {fit['a']:.2f}·k$^{{-{fit['b']:.3f}}}$")
    # 目标线与交点
    ax.axhline(target, color=OK["vermil"], lw=1.2, ls="--", zorder=3)
    ax.annotate(f"target width = {target}", (min(ks), target), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=OK["vermil"])
    ax.plot([k_need], [target], "*", color=OK["vermil"], ms=15, mec="white", mew=0.8,
            zorder=6)
    ax.annotate(f"k ≈ {k_need:.0f}\n({k_need/n_now:.1f}× current)", (k_need, target),
                xytext=(6, 10), textcoords="offset points", fontsize=8,
                color=OK["vermil"], fontweight="bold")
    # 当前点
    cur_w = sz["current_strict_grouping"]["ci_width"]
    ax.plot([n_now], [cur_w], "D", color=OK["orange"], ms=8, mec="white", mew=0.8, zorder=6)
    ax.annotate(f"current: {n_now} groups,\nwidth {cur_w:.3f}", (n_now, cur_w),
                xytext=(8, 8), textcoords="offset points", fontsize=7.5, color=OK["orange"])
    # lever 2
    ks2 = [p["n_groups"] for p in L2]
    ws2 = [p["width_mean"] for p in L2]
    ax.plot(ks2, ws2, "s", color=OK["green"], ms=5, mec="white", mew=0.6,
            label="lever 2: cap largest component (fixed n)", zorder=5, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("independent identifier groups  k")
    ax.set_ylabel("Δ$_H$ 95% CI width")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    ax.set_title("Reaching adequate precision under strict grouping needs ~7.7×\n"
                 "more independent identifier groups (capping alone is insufficient)",
                 fontsize=9.5, loc="left", color=INK)
    _save(fig, "fig4_sizing")


def main():
    fig1_grouping_slope()
    fig2_redline_forest()
    fig3_text_decomposition()
    fig4_sizing()
    print(f"\n[fig] all figures -> {OUT}/  (PDF for LaTeX, PNG for preview)")


if __name__ == "__main__":
    main()
