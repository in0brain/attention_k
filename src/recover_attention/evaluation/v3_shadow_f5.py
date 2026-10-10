"""两臂 stable-logit shadow F5 —— **performance-blind，不解盲，不读 label**。

命名口径（强制）：产物只能称 `stable-logit shadow sensitivity` /
`protocol-deviation numerical sensitivity`。**不得**称 primary / confirmatory /
"修正后的 B4" / 预注册结果。原 14 项 H1 B4 仍是 **frozen implementation primary**，
保留不覆盖不作废；其中 8 项不能按名义科学含义单独解释。

重建范围（其余字段**不重新发明**，只引用冻结值并记录来源）：

    H1  重建 8 项 / 引用 6 项
    MCQ 重建 4 项 / 引用 2 项

两臂必须使用**同一个** `stable_logits` 原语。MCQ 的四项直接调用
`domain_label_proxy` 的原定义 —— 只换输入数值（float32 稳定 logits），不改公式。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .. import domain_label_proxy as dlp
from ..generation.stable_logits import STABLE_PROJECTION_SCHEMA

SHADOW_SCHEMA = "stage1_stable_logit_shadow_f5_v1"

#: 只能这样称呼；测试会断言这两个词出现在产物里。
SENSITIVITY_LABEL = "stable_logit_shadow_sensitivity"
DEVIATION_LABEL = "protocol_deviation_numerical_sensitivity"

H1_STABLE_FIELDS = (
    "f5_completion_token_count", "f5_lengthnorm_logprob", "f5_completion_perplexity",
    "f5_id_token_ratio", "f5_id_logprob_mean", "f5_id_logprob_min",
    "f5_id_token_entropy_mean", "f5_first_id_token_rank",
)
H1_FROZEN_PASSTHROUGH = (
    "f5_id_token_count", "f5_self_consistency_exact", "f5_id_agreement_rate",
    "f5_confidence_high", "f5_confidence_medium", "f5_confidence_low",
)
MCQ_STABLE_FIELDS = (
    "f5_label_margin", "f5_label_entropy", "f5_full_entropy", "f5_letter_token_logprob",
)
MCQ_FROZEN_PASSTHROUGH = ("f5_self_consistency_exact", "f5_letter_agreement_rate")

#: `sequence_logprob_features` 的指数保护阈值，shadow 侧沿用同一条件。
PPL_EXP_GUARD = 700.0


class ShadowAcceptanceError(RuntimeError):
    """硬验收失败 —— 停机，不降级、不静默修补。"""


def _finite(value: Any) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ShadowAcceptanceError(f"shadow 字段非有限：{value!r}")
    return v


# ------------------------------------------------------------------- H1
def h1_shadow_row(row: Mapping[str, Any], *, n_tok: int,
                  target_logprob: np.ndarray, entropy: np.ndarray,
                  rank: np.ndarray, id_positions: Sequence[int]) -> dict:
    """重建 H1 的 8 项。**不做任何过滤** —— 全部 completion token 都参与。

    `target_logprob` / `entropy` / `rank` 必须是**全部** n_tok 个 completion 位置的
    逐行结果，顺序与 completion token 顺序一致。
    """
    lp = np.asarray(target_logprob, dtype=np.float64)
    ent = np.asarray(entropy, dtype=np.float64)
    rk = np.asarray(rank, dtype=np.int64)
    pos = sorted({int(p) for p in id_positions})

    if lp.shape != (n_tok,) or ent.shape != (n_tok,) or rk.shape != (n_tok,):
        raise ShadowAcceptanceError(
            f"逐行结果长度与 completion token 数不符："
            f"{lp.shape}/{ent.shape}/{rk.shape} vs n_tok={n_tok}")
    if not pos:
        raise ShadowAcceptanceError("identifier positions 为空 —— 该 trace 不应进入 shadow")
    if pos[0] < 0 or pos[-1] >= n_tok:
        raise ShadowAcceptanceError(
            f"identifier positions 越界：min {pos[0]} max {pos[-1]} vs n_tok={n_tok}")

    lengthnorm = _finite(lp.mean())
    ppl = math.exp(-lengthnorm) if lengthnorm < PPL_EXP_GUARD else None
    out = {
        "f5_completion_token_count": int(n_tok),
        "f5_lengthnorm_logprob": lengthnorm,
        "f5_completion_perplexity": _finite(ppl) if ppl is not None else None,
        "f5_id_token_ratio": _finite(len(pos) / n_tok),
        "f5_id_logprob_mean": _finite(lp[pos].mean()),
        "f5_id_logprob_min": _finite(lp[pos].min()),
        "f5_id_token_entropy_mean": _finite(ent[pos].mean()),
        "f5_first_id_token_rank": int(rk[pos[0]]),
    }
    assert_h1_acceptance(out, row, n_tok=n_tok, n_id=len(pos))
    return out


def assert_h1_acceptance(shadow: Mapping[str, Any], row: Mapping[str, Any], *,
                         n_tok: int, n_id: int) -> None:
    """硬验收 —— 写进代码，不只写文档。任一不过即停机。"""
    if shadow["f5_completion_token_count"] != int(n_tok):
        raise ShadowAcceptanceError(
            f"shadow count {shadow['f5_completion_token_count']} != 真实 token 数 {n_tok}")
    ratio = shadow["f5_id_token_ratio"]
    if not (0.0 <= ratio <= 1.0):
        raise ShadowAcceptanceError(f"f5_id_token_ratio 越界：{ratio}")
    if not math.isclose(ratio, n_id / n_tok, rel_tol=1e-12, abs_tol=0.0):
        raise ShadowAcceptanceError("f5_id_token_ratio 与 n_id/n_tok 不一致")
    if shadow["f5_first_id_token_rank"] < 1:
        raise ShadowAcceptanceError("rank 必须 >= 1")
    for name in H1_STABLE_FIELDS:
        v = shadow[name]
        if v is None:
            if name != "f5_completion_perplexity":
                raise ShadowAcceptanceError(f"{name} 不得为 None")
            continue
        if not math.isfinite(float(v)):
            raise ShadowAcceptanceError(f"{name} 非有限：{v!r}")
    frozen_id_count = row.get("f5_id_token_count")
    if frozen_id_count is not None and int(frozen_id_count) != n_id:
        raise ShadowAcceptanceError(
            f"identifier 计数与冻结值不符：{frozen_id_count} vs {n_id}")


# ------------------------------------------------------------------ MCQ
def mcq_shadow_row(row: Mapping[str, Any], *, option_logits: Mapping[str, float],
                   full_logits_row: Any, letter_logprob: float) -> dict:
    """重建 MCQ 的 4 项。

    公式**完全复用** `domain_label_proxy` 的原定义（`label_margin` / `label_entropy` /
    `full_entropy`），只把输入换成 float32 稳定路径的数值。
    """
    opts = dict(option_logits)
    if len(opts) < 2:
        raise ShadowAcceptanceError("候选 logit 少于 2 个，margin/entropy 无从定义")
    if any(not math.isfinite(float(v)) for v in opts.values()):
        raise ShadowAcceptanceError(f"候选 logit 含非有限值：{opts!r}")
    try:
        out = {
            "f5_label_margin": _finite(dlp.label_margin(opts)),
            "f5_label_entropy": _finite(dlp.label_entropy(opts)),
            "f5_full_entropy": _finite(dlp.full_entropy(full_logits_row)),
            "f5_letter_token_logprob": _finite(letter_logprob),
        }
    except ValueError as exc:
        # `domain_label_proxy` 自己也 fail-closed，但抛的是 ValueError；
        # 统一成一种停机类型，便于上游 stop_on_failure 一致处理。原因原样带出。
        raise ShadowAcceptanceError(f"domain_label_proxy 拒绝输入：{exc}") from exc
    assert_mcq_acceptance(out, row, opts)
    return out


def assert_mcq_acceptance(shadow: Mapping[str, Any], row: Mapping[str, Any],
                          option_logits: Mapping[str, float]) -> None:
    if len(option_logits) < 2:
        raise ShadowAcceptanceError("候选 logit 少于 2 个，margin/entropy 无从定义")
    for name in MCQ_STABLE_FIELDS:
        if not math.isfinite(float(shadow[name])):
            raise ShadowAcceptanceError(f"{name} 非有限：{shadow[name]!r}")
    if shadow["f5_label_margin"] < 0:
        raise ShadowAcceptanceError("label_margin 定义为 top1−top2，不得为负")
    if shadow["f5_label_entropy"] < 0 or shadow["f5_full_entropy"] < 0:
        raise ShadowAcceptanceError("entropy 不得为负")
    if shadow["f5_letter_token_logprob"] > 0:
        raise ShadowAcceptanceError("logprob 不得为正")


# --------------------------------------------------- 冻结字段引用（不重建）
def frozen_passthrough(row: Mapping[str, Any], arm: str) -> dict:
    """引用冻结值并**记录来源** —— 这些字段不重新发明。"""
    names = H1_FROZEN_PASSTHROUGH if arm == "h1" else MCQ_FROZEN_PASSTHROUGH
    return {"values": {n: row.get(n) for n in names},
            "source": f"frozen {arm}_completion_records.jsonl (未重建、未修改)",
            "field_names": list(names)}


def shadow_record(row: Mapping[str, Any], arm: str, stable: Mapping[str, Any], *,
                  provenance: Mapping[str, Any]) -> dict:
    """一条 shadow 记录。命名口径与来源都写进记录本体。"""
    key = "trace_id" if arm == "h1" else "example_id"
    ident = (provenance.get("unit_id") if arm == "h1" else row.get("example_id"))
    return {
        "schema": SHADOW_SCHEMA,
        "arm": arm,
        key: ident,
        "example_id": row.get("example_id"),
        "interpretation": SENSITIVITY_LABEL,
        "deviation_class": DEVIATION_LABEL,
        "not_primary": ("原冻结 B4 仍是 frozen implementation primary；"
                        "本记录**不得**冒充 primary / confirmatory / 预注册结果"),
        "stable_fields": dict(stable),
        "frozen_passthrough": frozen_passthrough(row, arm),
        "projection_schema": STABLE_PROJECTION_SCHEMA,
        "provenance": dict(provenance),
    }
