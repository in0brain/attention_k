"""Stage 1 v3：repeat-level estimand（M7）。

**核心纪律**：逐 repeat 计算 AUROC 差，再跨 repeat 平均。

    Δ_r  = AUROC(s_r(S*+H)) − AUROC(s_r(S*))
    Δ̂_H  = (1/R) · Σ_r Δ_r

❌ **不得**先逐样本平均 R 份 OOF 分数再算一次 AUROC 差 —— 那定义的是一个
   **R 成员 ensemble 预测器**的增量，是**另一个 estimand**。本模块提供
   `ensemble_delta_h` 仅供附录对照，其返回值带 `estimand="ensemble"` 标记，
   **不得**进入 M1b、(iv-a) 或主 Δ_H。

同口径的三组逐块统计量（M1b / M1b(iv-a)）：
    Δ_{k,r}        = AUROC_r(S*)   − AUROC_r(B_k)        非劣性认证用
    Δ^direct_{k,r} = AUROC_r(S*+H) − AUROC_r(B_k)        (iv-a) 唯一授权路径用
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.v3_blocks import BlockSpec
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_stacking import build_base_block_cache, run_repeat

DELTA_H = "delta_H"
DELTA_K = "delta_k"                 # M1b：S* − B_k
DELTA_DIRECT_K = "delta_direct_k"   # iv-a：S*+H − B_k


def repeat_statistics(y, oof: dict, block_names: Sequence[str], rows=None) -> dict:
    """在给定行子集上计算**一个 repeat** 的全部统计量。

    `rows=None` 表示全样本（点估计）；bootstrap 时传入该次重采样的行索引。
    单类子集下 AUROC 未定义 → 相应统计量为 NaN，由调用方丢弃该轮（不当 0）。
    """
    y = np.asarray(y)
    idx = np.arange(y.size) if rows is None else np.asarray(rows)
    ys = y[idx]
    a_s = ci.auroc(oof["S*"][idx], ys)
    a_sh = ci.auroc(oof["S*+H"][idx], ys)
    # 每个 block 的 AUROC **只算一次**再复用（第 1 版算了三次，正式规模下是 3 倍开销）
    a_b = {k: ci.auroc(oof[k][idx], ys) for k in block_names}
    return {
        DELTA_H: a_sh - a_s,
        DELTA_K: {k: a_s - a_b[k] for k in block_names},
        DELTA_DIRECT_K: {k: a_sh - a_b[k] for k in block_names},
        "auroc_S*": a_s, "auroc_S*+H": a_sh,
        "auroc_block": a_b,
    }


def average_across_repeats(per_repeat: Sequence[dict], block_names: Sequence[str]) -> dict:
    """跨 repeat 平均**统计量**（不是平均分数）。任一 repeat 为 NaN → 该量为 NaN。"""
    def _m(vals):
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)) if np.all(np.isfinite(arr)) else float("nan")

    return {
        DELTA_H: _m([s[DELTA_H] for s in per_repeat]),
        DELTA_K: {k: _m([s[DELTA_K][k] for s in per_repeat]) for k in block_names},
        DELTA_DIRECT_K: {k: _m([s[DELTA_DIRECT_K][k] for s in per_repeat])
                         for k in block_names},
        "auroc_S*": _m([s["auroc_S*"] for s in per_repeat]),
        "auroc_S*+H": _m([s["auroc_S*+H"] for s in per_repeat]),
        "auroc_block": {k: _m([s["auroc_block"][k] for s in per_repeat]) for k in block_names},
        "n_repeats": len(per_repeat),
    }


def run_all_repeats(*, y, groups, blocks: Sequence[BlockSpec], cfg: V3AnalysisConfig,
                    cache_builder=build_base_block_cache) -> list[dict]:
    """跑 R 次重复交叉拟合，**保留 R 份彼此独立的 OOF 分数**。

    fold-seed 按前缀嵌套（cfg.fold_seed(r)），故 R=3 用的就是 R=5 的前 3 个。
    """
    return [run_repeat(y=y, groups=groups, blocks=blocks, repeat=r, cfg=cfg,
                       cache_builder=cache_builder)
            for r in range(cfg.n_repeats)]


def point_estimates(y, repeats: Sequence[dict], block_names: Sequence[str]) -> dict:
    """Δ̂_H 与逐块点估计：**逐 repeat 算差 → 跨 repeat 平均**。"""
    per = [repeat_statistics(y, r["oof"], block_names) for r in repeats]
    out = average_across_repeats(per, block_names)
    out["per_repeat"] = per
    out["estimand"] = "repeat_averaged_statistic"
    return out


def ensemble_delta_h(y, repeats: Sequence[dict]) -> dict:
    """⚠️ **附录对照，另一个 estimand。**

    先逐样本平均 R 份 OOF 分数，再算一次 AUROC 差 —— 这测的是一个
    **R 成员 ensemble 预测器**的增量，**不是** primary 的稳健化版本。
    返回值带 `estimand="ensemble"` 标记；**不得**进入 M1b、(iv-a) 或主 Δ_H。
    """
    y = np.asarray(y)
    s = np.mean([r["oof"]["S*"] for r in repeats], axis=0)
    sh = np.mean([r["oof"]["S*+H"] for r in repeats], axis=0)
    return {"estimand": "ensemble",
            "warning": "APPENDIX ONLY — different estimand; must not replace primary",
            DELTA_H: ci.auroc(sh, y) - ci.auroc(s, y)}
