"""§6 observability gate：D = S_MCQ − S_H1，independent grouped bootstrap，δ=0.15。

顺序（§6 明写）:generate → ladder → **gate** → increment → rq2。
"前置"指"在 hidden 增量结论之前",不指"在 ladder 之前" —— gate 需要各任务 O 的 OOF 分数。

前提（v2.3 后由代码强制）:两臂同 backend。否则 D = observability 差异 + 量化差异,
不可解释。见 feature_cache.check_backend_invariant。
"""

from __future__ import annotations

from typing import Any

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation.config import Stage0Config
from recover_attention.evaluation.feature_cache import ArmCache

OUTCOME_3 = "outcome_3"
H1_HIGH_CONF = "h1_high_confidence"
UNCERTAIN = "uncertain"


def compute_gate(mcq: ArmCache, mcq_ladder: dict[str, Any],
                 h1: ArmCache, h1_ladder: dict[str, Any],
                 config: Stage0Config, backend_invariant: dict) -> dict[str, Any]:
    """S_t = max(0, 2·AUROC(O_t) − 1);D = S_MCQ − S_H1。"""
    if not backend_invariant.get("ok"):
        raise ValueError(
            f"observability gate refused: arms do not share one backend — D would confound "
            f"observability with quantization. {backend_invariant}")

    s_mcq = mcq_ladder["scores"][mcq.spec["o_rung"]]
    s_h1 = h1_ladder["scores"][h1.spec["o_rung"]]
    a_mcq = ci.auroc(s_mcq, mcq.y)
    a_h1 = ci.auroc(s_h1, h1.y)

    res = bs.independent_D(
        {"y": mcq.y, "score": s_mcq, "groups": mcq.groups},
        {"y": h1.y, "score": s_h1, "groups": h1.groups}, config)

    verdict = res["verdict"]
    return {
        "auroc_O_mcq": a_mcq, "auroc_O_h1": a_h1,
        "S_mcq": ci.s_observability(a_mcq), "S_h1": ci.s_observability(a_h1),
        "D": {"point": res["D_point"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
              "n_boot": res["n_boot"], "n_valid": res["n_valid"],
              "n_discarded_single_class": res["n_discarded_single_class"]},
        "delta": config.delta_rank_biserial,
        "verdict": verdict,
        "h1_is_high_confidence_setting": verdict == H1_HIGH_CONF,
        "reading": {
            H1_HIGH_CONF: "CI(D) 全 > δ → H1 显著更不易由输出信号观测",
            UNCERTAIN: "CI 跨 δ / 0 → observability 差异不确定",
            OUTCOME_3: ("CI 全 ≤ 0 → Outcome 3:H1 output-only 仍强可分 → H1 未构造高置信错误。"
                        "结论须转为'两个可构造任务上输出错误都可观测,内部增量在两者都近零',"
                        "**不得假装 H1 是高置信设置**"),
        }[verdict],
        "paired_not_used": ("independent bootstrap:两任务非同批 prompt、无天然配对;"
                            "paired 只用于同任务同批样本的 O vs O+H"),
        "backend_invariant": backend_invariant,
        "order": "generate -> ladder -> gate -> increment -> rq2",
    }
