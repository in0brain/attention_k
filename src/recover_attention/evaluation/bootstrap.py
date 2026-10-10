"""正式 bootstrap 包装：强制 n_boot>=1000，并执行 §7.2 的精度承诺。

纯统计仍在 conditional_increment（已有单测）。本模块加的是**正式规格的强制**与
**精度判读**——后者是 v2.3 写死的硬承诺,必须由代码执行,不能靠人自觉。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.config import PRODUCTION_MIN_BOOT, Stage0Config

PRECISION_ADEQUATE = "adequate"
PRECISION_INSUFFICIENT = "insufficient"


def precision_verdict(ci_lo: float, ci_hi: float, config: Stage0Config) -> dict:
    """§7.2 的硬承诺:实际 CI 宽度装不进 [−ε,+ε] → 该臂如实记 inconclusive。

    冻结原文:"Stage 0 后必须回报各臂 Δ_H 的**实际** CI 宽度。实际宽度 > 0.04 →
    该臂如实记 inconclusive,**不得**事后追加样本、改 ε、换判读规则或改融合协议。
    扩容只此一次(v2.3),不再有下一次。"

    先验 sizing(MCQ n=1760 → 87%、H1 480×K=6 → 95%)是**设计时依据,不是已兑现的功效**:
    rho=0.9051 借自 4C 的 F1(7 维、旧协议),用于 H(3584 维、v2.2 三-block)。本函数就是
    那条承诺的兑现点。
    """
    width = float(ci_hi - ci_lo)
    ok = width <= config.precision_max_ci_width
    return {
        "ci_width": width,
        "max_ci_width": config.precision_max_ci_width,
        "precision": PRECISION_ADEQUATE if ok else PRECISION_INSUFFICIENT,
        "prior_sizing_was_an_estimate_not_achieved_power": True,
        "note": (None if ok else
                 f"CI width {width:.4f} > {config.precision_max_ci_width} — cannot fit inside "
                 f"[-eps,+eps]; this arm records inconclusive. Per section 7.2: no extra samples, "
                 f"no changing eps, no changing the read rule, no changing the fusion protocol."),
    }


def paired_delta(y: Sequence[int], s_o: Sequence[float], s_oh: Sequence[float],
                 groups: Sequence, config: Stage0Config) -> dict:
    """Δ_H = AUROC(O+H) − AUROC(O) 的 paired grouped-bootstrap（同任务同批样本）。"""
    if config.n_boot < PRODUCTION_MIN_BOOT and config.mode == "production":
        raise ValueError(f"production requires n_boot>={PRODUCTION_MIN_BOOT}, got {config.n_boot}")
    res = ci.paired_grouped_bootstrap_delta(y, s_o, s_oh, groups,
                                            n_boot=config.n_boot, seed=config.seed)
    res["read"] = ci.equivalence_read(res["ci_lo"], res["ci_hi"], eps=config.eps_auroc)
    res["precision"] = precision_verdict(res["ci_lo"], res["ci_hi"], config)
    # 精度不足 → 判读一律降为 inconclusive,不许拿窄不下来的 CI 去宣称 equivalent
    if res["precision"]["precision"] == PRECISION_INSUFFICIENT:
        res["read_before_precision_check"] = res["read"]
        res["read"] = "inconclusive"
    return res


def paired_auprc_delta(y: Sequence[int], s_o: Sequence[float], s_oh: Sequence[float],
                       groups: Sequence, config: Stage0Config) -> dict:
    """§4「AUPRC 增量并报（类不平衡）」——**带 CI**。

    只报点估计的话,正例率 15~20% 下读者无从判断增量是不是噪声,而类不平衡正是 §4 要求
    同报 AUPRC 的理由。重采样与 Δ_AUROC 同构:paired grouped bootstrap,单类轮丢弃。
    """
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    s_o = np.asarray(s_o, float)
    s_oh = np.asarray(s_oh, float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    imap = {gv: np.where(g == gv)[0] for gv in uniq}
    rng = np.random.default_rng(config.seed)
    diffs: list[float] = []
    n_discard = 0
    for _ in range(config.n_boot):
        rows = np.concatenate([imap[gv] for gv in
                               rng.choice(uniq, size=len(uniq), replace=True)])
        yr = y[rows]
        if len(np.unique(yr)) < 2:
            n_discard += 1
            continue
        diffs.append(float(average_precision_score(yr, s_oh[rows]))
                     - float(average_precision_score(yr, s_o[rows])))
    point_o = float(average_precision_score(y, s_o))
    point_oh = float(average_precision_score(y, s_oh))
    out = {"O": point_o, "O_plus_H": point_oh, "delta_auprc": point_oh - point_o,
           "base_rate": float((y == 1).mean()),
           "n_boot": int(config.n_boot), "n_valid": len(diffs), "n_discarded": n_discard}
    if diffs:
        d = np.asarray(diffs)
        out["ci_lo"], out["ci_hi"] = _percentile_ci(d)
    else:
        out["ci_lo"] = out["ci_hi"] = None
        out["reason"] = "every resample was single-class"
    return out


def between_strata_delta(y: Sequence[int], s_o: Sequence[float], s_oh: Sequence[float],
                         groups: Sequence, strata: Sequence[bool],
                         config: Stage0Config) -> dict:
    """§5 要求的**层间差 CI**：Δ_H(high_conf) − Δ_H(low_conf)。

    重采样方式不是想当然的"两层各自独立 bootstrap"。**group 会跨层**:同一 prompt 的
    K=6 条 completion 由 fold 阈值逐行判层,实测 9/20 的 group 同时出现在两层。
    独立重采样会把跨层 group 重复计入,并抹掉两层 Δ 之间由共享 group 引入的相关 →
    CI 宽度失真。

    正确做法(本函数):**对 group 重采样一次**,再用预先算好的 stratum 标签把重采样后的
    行切成两层,同一轮内算 Δ_high 与 Δ_low 之差。这样跨层 group 在两层里同进同出,
    组内相关被正确保留。

    任一层在某轮里单类(AUROC 未定义)→ 丢弃该轮并计数,不当 0 处理。
    """
    y = np.asarray(y)
    s_o = np.asarray(s_o, float)
    s_oh = np.asarray(s_oh, float)
    g = np.asarray(groups)
    st = np.asarray(strata, dtype=bool)
    uniq = np.unique(g)
    imap = {gv: np.where(g == gv)[0] for gv in uniq}
    spanning = int(sum(1 for gv in uniq if len(np.unique(st[imap[gv]])) > 1))

    rng = np.random.default_rng(config.seed)
    diffs: list[float] = []
    n_discard = 0
    for _ in range(config.n_boot):
        rows = np.concatenate([imap[gv] for gv in
                               rng.choice(uniq, size=len(uniq), replace=True)])
        hi = rows[st[rows]]
        lo = rows[~st[rows]]
        if hi.size == 0 or lo.size == 0:
            n_discard += 1
            continue
        a = [ci.auroc(s_o[hi], y[hi]), ci.auroc(s_oh[hi], y[hi]),
             ci.auroc(s_o[lo], y[lo]), ci.auroc(s_oh[lo], y[lo])]
        if any(v != v for v in a):          # 任一层单类 → 该轮无定义
            n_discard += 1
            continue
        diffs.append((a[1] - a[0]) - (a[3] - a[2]))

    if not diffs:
        return {"status": "undefined", "n_valid": 0, "n_discarded": n_discard,
                "n_groups_spanning_strata": spanning,
                "reason": "every resample left a single-class stratum; too few positives"}
    d = np.asarray(diffs)
    lo_ci, hi_ci = _percentile_ci(d)
    point_hi = ci.auroc(s_oh[st], y[st]) - ci.auroc(s_o[st], y[st])
    point_lo = ci.auroc(s_oh[~st], y[~st]) - ci.auroc(s_o[~st], y[~st])
    return {
        "status": "ok",
        "point": float(point_hi - point_lo),
        "delta_high_conf": float(point_hi), "delta_low_conf": float(point_lo),
        "ci_lo": lo_ci, "ci_hi": hi_ci,
        "n_boot": int(config.n_boot), "n_valid": int(d.size), "n_discarded": int(n_discard),
        "n_groups_spanning_strata": spanning,
        "resampling": ("groups resampled ONCE then split by stratum label — strata are not "
                       "independent because groups span them (measured: "
                       f"{spanning}/{len(uniq)} groups)"),
        "positioning": ("descriptive effect-modification (§5); the stratifier is itself part of O "
                        "and this must be stated in the paper"),
    }


def _percentile_ci(d: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def independent_D(task_mcq: dict, task_h1: dict, config: Stage0Config) -> dict:
    """跨任务 D = S_MCQ − S_H1（independent grouped bootstrap，**非** paired）。"""
    if config.n_boot < PRODUCTION_MIN_BOOT and config.mode == "production":
        raise ValueError(f"production requires n_boot>={PRODUCTION_MIN_BOOT}, got {config.n_boot}")
    return ci.independent_grouped_bootstrap_D(task_mcq, task_h1,
                                              n_boot=config.n_boot, seed=config.seed)
