"""RQ1 条件增量：Δ_H = AUROC(O+H) − AUROC(O)，以及 §7 强制的三者同报。

§7 原文:"报告 AUROC(O)、AUROC(H alone)、AUROC(O+H) 三者——若 H alone 强但 O+H≈O,
则增量被融合掩盖,须讨论,不得直接判'无增量'。"
故本模块不只产出 Δ_H,还检出**融合掩盖**这一形态并显式标记。
"""

from __future__ import annotations

from typing import Any

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation.config import Stage0Config
from recover_attention.evaluation.feature_cache import ArmCache


def _fusion_masking_flag(auroc_o: float, auroc_h: float, auroc_oh: float,
                         eps: float) -> dict:
    """检出"H alone 强但 O+H≈O"(增量被融合掩盖)与"O+H 显著低于 O"(融合有害)。

    v2.1 的两-block 融合就出过后一种:O+H(0.721) < H(0.785) < O(0.843) —— 那是协议缺陷,
    不是"内部状态无增量"。v2.2 三-block 修掉后 O+H(0.834) >= O(0.801)。
    这类形态必须自动标出来,否则容易被误读成负结果。
    """
    masked = (auroc_h > auroc_o + eps) and (abs(auroc_oh - auroc_o) <= eps)
    harmful = auroc_oh < auroc_o - eps
    collapsed_to_h = abs(auroc_oh - auroc_h) <= eps and auroc_oh < auroc_o - eps
    return {
        "increment_possibly_masked_by_fusion": bool(masked),
        "fusion_harmful_oh_below_o": bool(harmful),
        "oh_collapsed_toward_h_alone": bool(collapsed_to_h),
        "note": ("若 masked=True:H alone 强而 O+H≈O → 增量可能被融合掩盖,§7 要求讨论,"
                 "不得直接判'无增量'。若 collapsed=True:O+H 塌向 H alone 且低于 O → "
                 "先查融合协议(v2.1 的两-block 就是这个形态),不要归因于 hidden。"),
    }


def compute_increment(cache: ArmCache, ladder: dict[str, Any],
                      config: Stage0Config) -> dict[str, Any]:
    """一臂的 RQ1 主结果。"""
    scores = ladder["scores"]
    o_rung = cache.spec["o_rung"]
    s_o, s_h, s_oh = scores[o_rung], scores["H_alone"], scores["O_plus_H"]
    a_o = ci.auroc(s_o, cache.y)
    a_h = ci.auroc(s_h, cache.y)
    a_oh = ci.auroc(s_oh, cache.y)

    delta = bs.paired_delta(cache.y, s_o, s_oh, cache.groups, config)
    # AUPRC 增量并报（§4：类不平衡）。**带 CI** —— 只有点估计的话，正例率 15~20% 下
    # 读者无从判断增量是不是噪声，而"类不平衡"正是 §4 要求同报 AUPRC 的理由。
    auprc = bs.paired_auprc_delta(cache.y, s_o, s_oh, cache.groups, config)
    auprc["H_alone"] = _auprc(cache.y, s_h)

    # §7 附录：late fusion robustness（非 primary）
    late = _late_fusion(cache, config)

    return {
        "arm": cache.arm,
        "n": int(cache.y.size), "n_positive": int((cache.y == 1).sum()),
        # §7 强制三者同报
        "auroc": {"O": a_o, "H_alone": a_h, "O_plus_H": a_oh},
        "auprc": auprc,
        "delta_H": {"point": delta["point"], "ci_lo": delta["ci_lo"], "ci_hi": delta["ci_hi"],
                    "n_boot": delta["n_boot"], "n_valid": delta["n_valid"],
                    "n_discarded_single_class": delta["n_discarded_single_class"]},
        "read": delta["read"],
        "read_before_precision_check": delta.get("read_before_precision_check"),
        "precision": delta["precision"],
        "eps": config.eps_auroc,
        "fusion_diagnostics": _fusion_masking_flag(a_o, a_h, a_oh, config.eps_auroc),
        "late_fusion_appendix": late,
        "o_definition": f"{o_rung} (F5 + canonical output text; §4 fixed, no max-taking)",
    }


def _auprc(y, s) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _late_fusion(cache: ArmCache, config: Stage0Config) -> dict:
    """§7 附录:分数级融合(2 维 meta-logistic),防"高维拼接不公平"的质疑。

    primary 恒为三-block early fusion;本项只作 robustness。
    meta 严格嵌套在 outer fold 内(inner cross-fitting),实现与单测在
    conditional_increment.late_fusion_oof_scores。

    该实现在**小样本下会翻号**(meta 训练集来自噪声更大的 inner 基模型),已由规模对照单测
    证明:n=90/n_pos=30 → 0.475,n=1200/n_pos=402 → 1.000。故此处附带样本量,
    并在样本不足时明确标为不可解读,而不是把一个 0.25 的 AUROC 当结论。
    """
    specs = cache.build_ladder_specs()
    try:
        folds = ci.stratified_grouped_folds(cache.y, cache.groups,
                                            n_splits=config.n_outer_folds, seed=config.seed)
        res = ci.late_fusion_oof_scores(
            cache.y, cache.groups, folds,
            spec_o=specs[cache.spec["o_rung"]], spec_h=specs["H_alone"],
            c_grid=config.c_grid, inner_splits=config.n_inner_folds, seed=config.seed)
    except (RuntimeError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)[:300], "is_primary": False}
    n_pos = int((cache.y == 1).sum())
    # 经验阈值来自规模对照:n_pos≈30 时嵌套 stacking 不稳定,n_pos≈400 时正常
    reliable = n_pos >= 100
    return {
        "status": "ok", "is_primary": False,
        "auroc_late_fusion": res["auroc"],
        "n": int(cache.y.size), "n_positive": n_pos,
        "interpretable": bool(reliable),
        "note": (None if reliable else
                 f"n_pos={n_pos}: nested stacking is unstable at this scale (meta trains on "
                 f"noisier inner-CV base scores and can learn negative weights). Scale control: "
                 f"n_pos=30 -> 0.475, n_pos=402 -> 1.000. Do not read this number as a result."),
        "reading": ("early 与 late 都无增量 → 负结果更强;仅 early 失败 → 是融合结构问题,"
                    "不得归因于 hidden state（§7）"),
    }
