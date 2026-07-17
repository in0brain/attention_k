"""§7 artifact 红线：Δ_artifact = AUROC(full-text) − AUROC(shortcut)。

冻结原文:"shortcut ∈ {id-string-only, surface-format-only};若 grouped-bootstrap
CI95(Δ_artifact) ⊂ [−0.02,+0.02]（= ε）→ shortcut 与 full-text 实质等价 → 触发红线:
该任务被字符串/格式模式解决,不做 hidden 增量声明。"

顺序很重要:红线在 hidden 增量**声明**之前。触发 → 该臂不出增量结论。

v2.3 的相关性:v2.2 下 MCQ 的 rung2(字母) 与 rung4(full-text) 是同一个东西,红线会因构造
必然触发、判定无意义。v2.3 的 semantic output canonicalization 让 rung4 = 被选中选项的
文本、rung2 = 字母身份,两者分离,红线在 MCQ 上才真正可解释。
"""

from __future__ import annotations

from typing import Any

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation.config import Stage0Config
from recover_attention.evaluation.feature_cache import ArmCache


def compute_artifact_redline(cache: ArmCache, ladder: dict[str, Any],
                             config: Stage0Config) -> dict[str, Any]:
    scores = ladder["scores"]
    full_rung = cache.spec["full_text_rung"]
    s_full = scores[full_rung]
    a_full = ci.auroc(s_full, cache.y)

    per_shortcut: dict[str, Any] = {}
    triggered_by = []
    for sc in cache.spec["shortcut_rungs"]:
        s_sc = scores[sc]
        d = ci.paired_grouped_bootstrap_delta(cache.y, s_sc, s_full, cache.groups,
                                              n_boot=config.n_boot, seed=config.seed)
        # 注意方向:Δ_artifact = AUROC(full-text) − AUROC(shortcut)
        # paired_grouped_bootstrap_delta(y, s_o, s_oh) 返回 AUROC(s_oh) − AUROC(s_o),
        # 故传 (shortcut, full) 得到的正是 full − shortcut。
        fired = ci.artifact_redline(d["ci_lo"], d["ci_hi"], eps=config.eps_auroc)
        per_shortcut[sc] = {
            "auroc_shortcut": ci.auroc(s_sc, cache.y),
            "delta_artifact": {"point": d["point"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                               "n_valid": d["n_valid"]},
            "redline_triggered": bool(fired),
            "meaning": ("shortcut 与 full-text 实质等价 → 该任务被字符串/格式模式解决"
                        if fired else "full-text 相对 shortcut 有实质差异"),
        }
        if fired:
            triggered_by.append(sc)

    return {
        "arm": cache.arm,
        "auroc_full_text": a_full,
        "full_text_rung": full_rung,
        "per_shortcut": per_shortcut,
        "redline_triggered": bool(triggered_by),
        "triggered_by": triggered_by,
        "consequence": ("该臂被字符串/格式模式解决 → **不做 hidden 增量声明**（§7）"
                        if triggered_by else "未触发 → 可继续 hidden 增量判读"),
        "order_note": "artifact 红线在 hidden 增量声明之前（§7）",
    }
