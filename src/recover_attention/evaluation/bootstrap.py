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


def independent_D(task_mcq: dict, task_h1: dict, config: Stage0Config) -> dict:
    """跨任务 D = S_MCQ − S_H1（independent grouped bootstrap，**非** paired）。"""
    if config.n_boot < PRODUCTION_MIN_BOOT and config.mode == "production":
        raise ValueError(f"production requires n_boot>={PRODUCTION_MIN_BOOT}, got {config.n_boot}")
    return ci.independent_grouped_bootstrap_D(task_mcq, task_h1,
                                              n_boot=config.n_boot, seed=config.seed)
