"""输出侧阶梯 + H + O+H 的 OOF 风险分（§7 正式规格）。

与 smoke 版的差别只在规格,不在协议:5 folds / inner>=3 / nested C{0.01,0.1,1,10}。
O / H / O+H 走**同一条** block_oof_scores 路径、**同一组** outer folds ——
这是 §7 明写的防融合-artifact 要求,也是"无增量"结论能成立的前提。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.config import Stage0Config
from recover_attention.evaluation.feature_cache import ArmCache


def run_ladder(cache: ArmCache, config: Stage0Config) -> dict[str, Any]:
    """跑完一臂的全部 rung（含 H_alone 与 O_plus_H），返回 OOF 分数与 AUROC。

    **同一组 outer folds 贯穿所有 rung** —— 分开生成 folds 会让 rung 之间不可比。
    """
    folds = ci.stratified_grouped_folds(cache.y, cache.groups,
                                        n_splits=config.n_outer_folds, seed=config.seed)
    specs = cache.build_ladder_specs()
    scores: dict[str, np.ndarray] = {}
    rungs: dict[str, Any] = {}
    for name, spec in specs.items():
        res = ci.block_oof_scores(cache.y, cache.groups, folds,
                                  text=spec["text"], dense_blocks=spec["dense_blocks"],
                                  c_grid=config.c_grid, inner_splits=config.n_inner_folds,
                                  seed=config.seed)
        scores[name] = res["scores"]
        rungs[name] = {"auroc": res["auroc"], "chosen_C": res["chosen_C"],
                       "fold_design": res["fold_design"][0]}
    return {
        "arm": cache.arm,
        "n": int(cache.y.size), "n_positive": int((cache.y == 1).sum()),
        "n_groups": int(len(np.unique(cache.groups))),
        "n_outer_folds": config.n_outer_folds,
        "n_inner_folds": config.n_inner_folds,
        "c_grid": list(config.c_grid),
        "folds": [(tr.tolist(), te.tolist()) for tr, te in folds],
        "rungs": rungs,
        "scores": scores,
        "o_rung": cache.spec["o_rung"],
        "same_outer_folds_for_all_rungs": True,
    }


def ladder_table(ladder: dict[str, Any]) -> dict[str, float]:
    """展示用:各 rung 的 AUROC。O 恒 = F5 + canonical text（§4 固定单一定义，不取 max）。"""
    return {name: r["auroc"] for name, r in ladder["rungs"].items()}
