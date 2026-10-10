"""两臂 observability contrast D（v3 执行口径，v2.4 §6 estimand）。

**estimand / 方向 / 阈值继承 v2.4 §6**：

    S_t = max(0, 2·AUROC(O_t) − 1)      D = S_MCQ − S_H1      δ = 0.15
    O_H1  = H1 F5 + completion full text
    O_MCQ = MCQ F5 + semantic_output_text
    判读  CI lower > 0.15 → h1_high_confidence（H1 更不易由输出侧观测，gate 通过）
          CI upper ≤ 0   → outcome_3
          其余           → uncertain

**repeat 与 bootstrap 执行口径继承 v3 M7**（不是旧函数默认值的逐字复刻）：

    每臂 R=3 次独立 grouped cross-fitting；逐 repeat 算 S_{t,r} 与 D_r，
    再 D̄ = mean_r D_r。**不得**先平均三个 repeat 的 OOF 分数再算 AUROC
    —— 那是 M7 明确区分的 ensemble estimand。
    B_total = 20,000 = B_batch(20) × B_per_batch(1,000)，批种子沿用 cfg.batch_seed(j)。
    每个 draw：先独立抽 H1 group，再独立抽 MCQ group；
    **各臂抽到的 group 在该臂 R 个 repeat 之间共用**；逐 repeat 算 D_r^(q) 再 mean_r。
    CI 取合并后的 20,000 个 D̄^(q) 的 2.5/97.5 百分位。

⚠️ **不得**把 M7 的 `0.04` 精度判据移植到 D —— 那不是 D gate 的判据。
⚠️ D 是**输出侧 observability contrast**，不是 `Δ_H(MCQ) − Δ_H(H1)`，也不得用 S* 代替 O。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from recover_attention.conditional_increment import auroc, s_observability
from recover_attention.evaluation.v3_config import V3AnalysisConfig

DELTA_OBSERVABILITY = 0.15                 # v2.4 §6 冻结阈值
CONTRAST_PROTOCOL = "v3_repeat_averaged_independent_group_bootstrap"
PROVENANCE = {
    "estimand_direction_delta": "inherited from v2.4 §6 (D = S_MCQ − S_H1, δ = 0.15)",
    "repeat_and_bootstrap_execution": "inherited from v3 M7 (R=3 repeat-averaged; "
                                      "B_total = B_batch × B_per_batch = 20 × 1000)",
    "not_v2_4_gate_verbatim": "旧函数的 n_boot=1000 / seed=0 / single-OOF 只是默认值，"
                              "不是冻结协议正文；此处不复刻",
}
VERDICT_H1 = "h1_high_confidence"
VERDICT_OUTCOME_3 = "outcome_3"
VERDICT_UNCERTAIN = "uncertain"
#: 有效 draw 占比下限；低于则停，不静默出 CI。
MIN_VALID_FRAC = 0.8


def _rows_by_group(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(groups)
    return uniq, {g: np.flatnonzero(groups == g) for g in uniq}


def _digest(picked) -> str:
    return hashlib.sha256(",".join(str(g) for g in picked).encode("utf-8")).hexdigest()[:16]


def s_per_repeat(y, oof_per_repeat: Sequence[np.ndarray], rows=None) -> list[float]:
    """逐 repeat 的 S_t = max(0, 2·AUROC(O_t,r) − 1)。单类 → NaN（调用方丢弃该轮）。"""
    y = np.asarray(y)
    idx = np.arange(y.size) if rows is None else np.asarray(rows)
    ys = y[idx]
    out = []
    for s in oof_per_repeat:
        a = auroc(np.asarray(s, dtype=float)[idx], ys)
        out.append(float("nan") if a != a else s_observability(a))
    return out


def d_bar(y_mcq, oof_mcq: Sequence[np.ndarray], y_h1, oof_h1: Sequence[np.ndarray],
          rows_mcq=None, rows_h1=None) -> float:
    """D̄ = mean_r ( S_MCQ,r − S_H1,r )。任一 repeat 单类 → NaN。"""
    sm = s_per_repeat(y_mcq, oof_mcq, rows_mcq)
    sh = s_per_repeat(y_h1, oof_h1, rows_h1)
    if len(sm) != len(sh):
        raise ValueError(f"两臂 repeat 数不一致：MCQ {len(sm)} vs H1 {len(sh)}")
    d = [a - b for a, b in zip(sm, sh)]
    return float(np.mean(d)) if np.all(np.isfinite(d)) else float("nan")


def contrast_d(*, y_mcq, groups_mcq, oof_mcq: Sequence[np.ndarray],
               y_h1, groups_h1, oof_h1: Sequence[np.ndarray],
               cfg: V3AnalysisConfig, delta: float = DELTA_OBSERVABILITY) -> dict:
    """两臂**独立** grouped bootstrap 的 D，repeat-averaged，合并分布取 CI。

    `oof_*` 是该臂 R 份彼此独立的 O 的 OOF 风险分（**不是** S*，不是 Δ_H）。
    """
    y_mcq, y_h1 = np.asarray(y_mcq), np.asarray(y_h1)
    gm, gh = np.asarray(groups_mcq), np.asarray(groups_h1)
    if len(oof_mcq) != len(oof_h1):
        raise ValueError(f"两臂 repeat 数不一致：MCQ {len(oof_mcq)} vs H1 {len(oof_h1)}")
    if len(oof_mcq) != int(cfg.n_repeats):
        raise ValueError(f"repeat 数 {len(oof_mcq)} != cfg.n_repeats {cfg.n_repeats}")

    point = d_bar(y_mcq, oof_mcq, y_h1, oof_h1)
    if not np.isfinite(point):
        raise RuntimeError("contrast_d: 点估计为 NaN（全样本上出现单类）")

    um, im = _rows_by_group(gm)
    uh, ih = _rows_by_group(gh)
    draws: list[float] = []
    batch_of: list[int] = []
    n_discarded = 0
    audit: dict[str, list[str]] = {}
    for j in range(int(cfg.b_batch)):
        rng = np.random.default_rng(cfg.batch_seed(j))
        kept: list[str] = []
        for _ in range(int(cfg.b_per_batch)):
            # 每个 draw：**先独立抽 H1 group，再独立抽 MCQ group**
            ph = rng.choice(uh, size=uh.size, replace=True)
            pm = rng.choice(um, size=um.size, replace=True)
            rows_h = np.concatenate([ih[g] for g in ph])
            rows_m = np.concatenate([im[g] for g in pm])
            if len(kept) < int(cfg.n_audit_digests_per_batch):
                kept.append(_digest(ph) + "|" + _digest(pm))
            # 各臂抽到的 group 在该臂 R 个 repeat 之间**共用**
            v = d_bar(y_mcq, oof_mcq, y_h1, oof_h1, rows_mcq=rows_m, rows_h1=rows_h)
            if not np.isfinite(v):
                n_discarded += 1
                continue
            draws.append(v)
            batch_of.append(j)
        audit[str(j)] = kept

    b_total = int(cfg.b_batch) * int(cfg.b_per_batch)
    d = np.asarray(draws, dtype=float)
    if d.size == 0:
        raise RuntimeError("contrast_d: 零个有效 draw（重采样全单类）")
    if d.size < MIN_VALID_FRAC * b_total:
        raise RuntimeError(f"contrast_d: 仅 {d.size}/{b_total} 个有效 draw "
                           f"(<{MIN_VALID_FRAC:.0%})；停")
    lo = float(np.percentile(d, 2.5))
    hi = float(np.percentile(d, 97.5))
    if lo > delta:
        verdict = VERDICT_H1
    elif hi <= 0:
        verdict = VERDICT_OUTCOME_3
    else:
        verdict = VERDICT_UNCERTAIN
    return {
        "contrast_protocol": CONTRAST_PROTOCOL, "provenance": dict(PROVENANCE),
        "D_point": point, "ci_lo": lo, "ci_hi": hi, "width": hi - lo,
        "delta": float(delta), "verdict": verdict,
        "direction": "D = S_MCQ - S_H1",
        "n_repeats": int(cfg.n_repeats), "b_batch": int(cfg.b_batch),
        "b_per_batch": int(cfg.b_per_batch), "b_total": b_total,
        "n_valid_draws": int(d.size), "n_discarded_single_class": int(n_discarded),
        "batch_of": np.asarray(batch_of, dtype=int),
        "batch_seeds": [cfg.batch_seed(j) for j in range(int(cfg.b_batch))],
        "audit_digests": audit,
        "paired": False,
        "precision_criterion_applies": False,
        "note": "M7 的 0.04 精度判据**不适用于 D**；D 的判读只用 δ=0.15 与 CI 方向",
    }
