"""§5 RQ2：observability 的 effect-modification（描述性）。

分层对象 = 全部 eligible completions（同时含 pos 与 neg,不只分 fabrication,否则单类层
AUROC 无法计算）。分层变量 = O 侧的 id-token logprob。
阈值 = **每个 outer fold 的 train 中位数**,应用到该 fold 的 test —— 全体取一次中位数
是 test leakage（v2.1 曾犯,已修）。
最小样本:每层 pos>=15 且 neg>=15,否则记 insufficient。
全体正例 < 30 → 整体标 exploratory / low-power。
定位:描述性 effect-modification,非因果;且分层变量本身 ∈ O,须在文中声明。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation.config import Stage0Config
from recover_attention.evaluation.feature_cache import ArmCache

# 分层变量:各臂 O 侧的 id-token logprob（H1 = identifier span 的 mean;MCQ = letter token）
STRATIFIER = {"h1": "f5_id_logprob_mean", "mcq": "f5_letter_token_logprob"}


def compute_rq2(cache: ArmCache, ladder: dict[str, Any], config: Stage0Config) -> dict[str, Any]:
    field = STRATIFIER[cache.arm]
    lp = np.array([float(r.get(field) or 0.0) for r in cache.records])
    folds = [(np.array(tr), np.array(te)) for tr, te in ladder["folds"]]
    fs = ci.rq2_fold_strata(lp, folds)
    hi = fs["strata"]

    scores = ladder["scores"]
    s_o = scores[cache.spec["o_rung"]]
    s_oh = scores["O_plus_H"]
    total_pos = int((cache.y == 1).sum())

    out: dict[str, Any] = {
        "arm": cache.arm,
        "stratifier": field,
        "threshold_policy": ("per-outer-fold train median of id-logprob, applied to that fold's "
                             "test (§5). A single global median would leak test information."),
        "fold_thresholds": fs["fold_thresholds"],
        "total_positives": total_pos,
        "power_flag": ci.rq2_power_flag(total_pos),
        "min_per_cell": config.rq2_min_per_cell,
        "positioning": ("descriptive effect-modification, not causal; the stratifier itself is "
                        "part of O and this must be stated in the paper (§5)"),
        "strata": {},
    }
    per_stratum_delta: dict[str, dict] = {}
    for name, mask in (("high_conf", hi), ("low_conf", ~hi)):
        yc = cache.y[mask]
        cell = {"n": int(mask.sum()), "n_pos": int((yc == 1).sum()), "n_neg": int((yc == 0).sum())}
        if not ci.rq2_cell_ok(yc):
            out["strata"][name] = {**cell, "status": "insufficient",
                                   "why": f"needs pos>={config.rq2_min_per_cell} and "
                                          f"neg>={config.rq2_min_per_cell} (§5)"}
            continue
        try:
            d = bs.paired_delta(yc, s_o[mask], s_oh[mask], cache.groups[mask], config)
        except RuntimeError as exc:
            out["strata"][name] = {**cell, "status": "bootstrap_stopped", "reason": str(exc)}
            continue
        out["strata"][name] = {
            **cell, "status": "ok",
            "auroc_O": ci.auroc(s_o[mask], yc),
            "auroc_H_alone": ci.auroc(scores["H_alone"][mask], yc),
            "auroc_O_plus_H": ci.auroc(s_oh[mask], yc),
            "delta_H": {"point": d["point"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]},
            "read": d["read"], "precision": d["precision"],
        }
        per_stratum_delta[name] = d

    # 层间差（§5 要求"+ 层间差 CI"）
    if len(per_stratum_delta) == 2:
        out["between_strata_difference"] = {
            "point": per_stratum_delta["high_conf"]["point"] - per_stratum_delta["low_conf"]["point"],
            "note": ("两层的 Δ_H 之差。两层样本不重叠,故非 paired;此处只报点估计,"
                     "CI 需 independent bootstrap —— 见 report 中的 between_strata_ci"),
        }
    return out
