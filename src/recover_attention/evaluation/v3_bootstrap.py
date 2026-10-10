"""Stage 1 v3：合并 grouped-bootstrap、simultaneous bounds、MC 诊断（M3 / M7）。

四条规格：

1. **同一 group 抽样跨 repeat 共用**（M7）。第 q 次重采样只抽一次 group，
   R 个 repeat 用**同一个** I_q 各算统计量，再跨 repeat 平均得 Δ̄^(q)。
   由 `BootstrapResamplePlan` 显式承载并可审计（每个 q 记 group-index digest）。

2. **主 CI 来自单一合并分布**（M7）。B_total = B_batch × B_per_batch 个 Δ̄^(q)
   合并成一个分布取 2.5/97.5 百分位。❌ 不产生 B_batch 个独立主 CI，
   ❌ 不对多个 CI 或多个宽度求平均。批只是计算与诊断手段。

3. **simultaneous bounds 的两个方向枢轴不同**（M3）：
       T_q^L = max_k (Δ̂_k − Δ_k^(q)) / ŝe_k   →  L_k = Δ̂_k − c_α^L·ŝe_k
       T_q^U = max_k (Δ_k^(q) − Δ̂_k) / ŝe_k   →  U_k = Δ̂_k + c_α^U·ŝe_k
   一般 c_α^L ≠ c_α^U。覆盖率是 **nominal**，依赖 grouped-bootstrap 有效性与
   常规正则条件，**不是**有限样本严格保证。

4. **MC 诊断 primary = delete-one-batch jackknife**，batch-means 为 secondary；
   带宽用 t₍₀.₉₇₅,₁₉₎ = 2.093（df=19，正态分位会低估）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from recover_attention.evaluation.v3_config import T_QUANTILE_975_DF19, V3AnalysisConfig
from recover_attention.evaluation.v3_repeat_estimand import (
    DELTA_DIRECT_K,
    DELTA_H,
    DELTA_K,
    repeat_statistics,
)

PRECISION_ADEQUATE = "adequate"
PRECISION_INSUFFICIENT = "insufficient"
PRECISION_BORDERLINE = "borderline"


# ───────────────────────────────────────────── resample plan（流式 / 可重放）
@dataclass
class BootstrapResamplePlan:
    """B_total 次 group 重采样的**可重放**计划 —— 不物化任何行索引。

    ⚠️ 第 1 版把每个 draw 的完整行索引存进 `rows: list[np.ndarray]`。按正式规模：

        n=2016,  B_total=200,000  → 仅 int64 行索引约 3.23 GB
        n=9270,  B_total=400,000  → 约 29.7 GB

    尚未计入 Python 对象、digest、OOF 与模型内存 → 必 OOM。**本版改为流式**：
    plan 只保存冻结 seed 与审计信息，`iter_draws()` 逐 q 生成 I_q，
    调用方算完标量立即释放；常驻内存只有最终的标量 draws。

    **共同随机数**：批 j 的 RNG 由 `cfg.batch_seed(j)` 唯一决定，
    较小的 `b_per_batch` 取该流的**前缀**（§7.3 冻结条款）。

    **审计（O(1) 内存）**：每批逐条留存前 `n_audit` 个 draw 的 group digest，
    其余折进该批的**滚动链哈希**，故 400k draw 不产生 400k 个 Python 字符串。
    """

    b_batch: int
    b_per_batch: int
    uniq_groups: np.ndarray
    batch_seeds: list[int]
    n_audit: int = 8
    audit_digests: dict[int, list[str]] = field(default_factory=dict)
    chain_digest: dict[int, str] = field(default_factory=dict)
    _rows_by_group: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return int(self.b_batch * self.b_per_batch)

    @property
    def b_total(self) -> int:
        return len(self)

    def iter_draws(self):
        """逐 draw 产出 `(q, batch_j, rows)`。行索引**用完即弃**，不驻留。"""
        q = 0
        for j in range(self.b_batch):
            rng = np.random.default_rng(self.batch_seeds[j])
            chain = hashlib.sha256()
            kept: list[str] = []
            for _ in range(self.b_per_batch):
                picked = rng.choice(self.uniq_groups, size=self.uniq_groups.size, replace=True)
                d = _digest(picked)
                chain.update(d.encode("utf-8"))
                if len(kept) < self.n_audit:
                    kept.append(d)
                yield q, j, np.concatenate([self._rows_by_group[g] for g in picked])
                q += 1
            self.audit_digests[j] = kept
            self.chain_digest[j] = chain.hexdigest()[:16]

    def audit(self) -> dict:
        return {"b_batch": self.b_batch, "b_per_batch": self.b_per_batch,
                "b_total": self.b_total, "batch_seeds": list(self.batch_seeds),
                "n_audit_per_batch": self.n_audit,
                "audit_digests": {str(k): v for k, v in self.audit_digests.items()},
                "chain_digest": {str(k): v for k, v in self.chain_digest.items()},
                "note": "streaming plan: row indices are generated on demand and never stored"}


def _digest(sampled_groups) -> str:
    return hashlib.sha256(
        ",".join(str(g) for g in sampled_groups).encode("utf-8")).hexdigest()[:16]


def build_resample_plan(groups, cfg: V3AnalysisConfig, *,
                        b_per_batch: int | None = None,
                        b_max: int | None = None) -> BootstrapResamplePlan:
    """构建**流式** plan。`b_max` 仅作合法性校验：前缀语义由"取 RNG 流前 p 项"天然保证。"""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    per = int(b_per_batch if b_per_batch is not None else cfg.b_per_batch)
    if b_max is not None and int(b_max) < per:
        raise ValueError(f"b_max={b_max} < b_per_batch={per}")
    return BootstrapResamplePlan(
        b_batch=int(cfg.b_batch), b_per_batch=per, uniq_groups=uniq,
        batch_seeds=[cfg.batch_seed(j) for j in range(cfg.b_batch)],
        n_audit=int(cfg.n_audit_digests_per_batch),
        _rows_by_group={g: np.flatnonzero(groups == g) for g in uniq})


# ───────────────────────────────────────────── 合并分布（流式消费）
def pooled_draws(y, repeats, block_names, plan: BootstrapResamplePlan) -> dict:
    """流式消费 plan：每个 q 的 I_q 由 R 个 repeat 共用 → 跨 repeat 平均 → 立即释放。

    任一 repeat 在该 q 上出现单类（AUROC 未定义）→ **丢弃该 q**，不计 0。
    常驻内存只有最终标量 draws。
    """
    dh: list[float] = []
    dk = {k: [] for k in block_names}
    dd = {k: [] for k in block_names}
    kept_batch: list[int] = []
    n_discarded = 0

    for _q, j, rows in plan.iter_draws():
        per = [repeat_statistics(y, r["oof"], block_names, rows=rows) for r in repeats]
        del rows                                    # 尽早释放行索引
        vals_h = [s[DELTA_H] for s in per]
        ok = bool(np.all(np.isfinite(vals_h)))
        if ok:
            for k in block_names:
                if not (np.all(np.isfinite([s[DELTA_K][k] for s in per]))
                        and np.all(np.isfinite([s[DELTA_DIRECT_K][k] for s in per]))):
                    ok = False
                    break
        if not ok:
            n_discarded += 1
            continue
        dh.append(float(np.mean(vals_h)))
        for k in block_names:
            dk[k].append(float(np.mean([s[DELTA_K][k] for s in per])))
            dd[k].append(float(np.mean([s[DELTA_DIRECT_K][k] for s in per])))
        kept_batch.append(j)

    draws = {DELTA_H: np.asarray(dh),
             DELTA_K: {k: np.asarray(v) for k, v in dk.items()},
             DELTA_DIRECT_K: {k: np.asarray(v) for k, v in dd.items()},
             "batch_of": np.asarray(kept_batch, dtype=int),
             "n_draws": len(dh), "n_discarded_single_class": int(n_discarded),
             "plan_audit": plan.audit()}
    if draws["n_draws"] == 0:
        raise RuntimeError("pooled_draws: zero valid bootstrap draws (all single-class)")
    return draws


def percentile_ci(values, alpha: float = 0.05) -> dict:
    v = np.asarray(values, dtype=float)
    lo = float(np.percentile(v, 100 * alpha / 2))
    hi = float(np.percentile(v, 100 * (1 - alpha / 2)))
    return {"ci_lo": lo, "ci_hi": hi, "width": hi - lo}


def marginal_lower_bound(values, alpha: float = 0.05) -> float:
    """边际单侧 (1−α) 下界 = 合并分布的 α 分位（M1b primary）。"""
    return float(np.percentile(np.asarray(values, dtype=float), 100 * alpha))


# ───────────────────────────────────────────── simultaneous bounds（两个方向）
def simultaneous_bounds(point: dict, draws: dict, alpha: float = 0.05) -> dict:
    """max-statistic 同时单侧界。**上下界枢轴不同，不可共用。**

    覆盖率为 **nominal** —— 依赖 grouped-bootstrap 有效性与常规正则条件，
    不是有限样本严格不等式。
    """
    names = [k for k in point]
    se = {k: float(np.std(np.asarray(draws[k], dtype=float), ddof=1)) for k in names}
    usable = [k for k in names if se[k] >= 1e-8]
    degenerate = [k for k in names if se[k] < 1e-8]
    if not usable:
        raise RuntimeError("simultaneous_bounds: all ŝe_k degenerate (<1e-8)")

    n_q = len(np.asarray(draws[usable[0]]))
    T_L = np.full(n_q, -np.inf)
    T_U = np.full(n_q, -np.inf)
    for k in usable:
        d = np.asarray(draws[k], dtype=float)
        T_L = np.maximum(T_L, (point[k] - d) / se[k])
        T_U = np.maximum(T_U, (d - point[k]) / se[k])
    c_L = float(np.percentile(T_L, 100 * (1 - alpha)))
    c_U = float(np.percentile(T_U, 100 * (1 - alpha)))
    return {
        "alpha": alpha, "m": len(usable), "se": se,
        "c_alpha_lower": c_L, "c_alpha_upper": c_U,
        "lower": {k: point[k] - c_L * se[k] for k in usable},
        "upper": {k: point[k] + c_U * se[k] for k in usable},
        "degenerate_blocks": degenerate,
        "coverage": "nominal (asymptotic; depends on grouped-bootstrap validity "
                    "and regularity conditions) — NOT a finite-sample guarantee",
    }


# ───────────────────────────────────────────── MC 诊断
def jackknife_mc(values, batch_of, b_batch: int, alpha_quantile: float = 0.05) -> dict:
    """delete-one-batch jackknife（**primary**）：直接估 pooled 分位数的 MC 误差。"""
    v = np.asarray(values, dtype=float)
    b = np.asarray(batch_of, dtype=int)
    full = percentile_ci(v, alpha_quantile)
    loo = {"ci_lo": [], "ci_hi": [], "width": []}
    used = 0
    for j in range(b_batch):
        keep = v[b != j]
        if keep.size < 2:
            continue
        c = percentile_ci(keep, alpha_quantile)
        for key in loo:
            loo[key].append(c[key])
        used += 1
    if used < 2:
        raise RuntimeError("jackknife_mc: fewer than 2 usable batches")

    se = {}
    for key, arr in loo.items():
        a = np.asarray(arr, dtype=float)
        se[key] = float(np.sqrt((used - 1) / used * np.sum((a - a.mean()) ** 2)))
    w = full["width"]
    half = T_QUANTILE_975_DF19 * se["width"]
    return {"method": "delete_one_batch_jackknife", "n_batches_used": used,
            "width": w, "ci_lo": full["ci_lo"], "ci_hi": full["ci_hi"],
            "se_jack": se, "t_quantile": T_QUANTILE_975_DF19,
            "mc_band_width": [w - half, w + half],
            "loo_width_mean": float(np.mean(loo["width"]))}


def batch_means_mc(values, batch_of, b_batch: int, alpha_quantile: float = 0.05) -> dict:
    """batch-means（**secondary**，并报作对照）。注意 mean(w_j) ≠ pooled width。"""
    v = np.asarray(values, dtype=float)
    b = np.asarray(batch_of, dtype=int)
    per = [percentile_ci(v[b == j], alpha_quantile) for j in range(b_batch)
           if np.count_nonzero(b == j) >= 2]
    if len(per) < 2:
        raise RuntimeError("batch_means_mc: fewer than 2 usable batches")
    se = {key: float(np.std([p[key] for p in per], ddof=1) / np.sqrt(len(per)))
          for key in ("ci_lo", "ci_hi", "width")}
    return {"method": "batch_means", "n_batches_used": len(per), "se_mc": se,
            "mean_batch_width": float(np.mean([p["width"] for p in per])),
            "note": "mean(w_j) is NOT the reported width; judgement uses pooled width only"}


def precision_verdict_v3(width: float, mc_band: Sequence[float],
                         max_ci_width: float) -> dict:
    """M7 精度判定：按 **MC 误差带的端点**三分，跑前写死。"""
    lo, hi = float(mc_band[0]), float(mc_band[1])
    if hi < max_ci_width:
        verdict = PRECISION_ADEQUATE
    elif lo > max_ci_width:
        verdict = PRECISION_INSUFFICIENT
    else:
        verdict = PRECISION_BORDERLINE
    return {"precision": verdict, "width": float(width), "mc_band": [lo, hi],
            "max_ci_width": float(max_ci_width),
            "note": None if verdict != PRECISION_BORDERLINE else
                    "0.04 falls inside the MC band — must not claim adequate or "
                    "insufficient; no enlarging B_batch/B_per_batch or adding samples"}


def certify_non_inferiority(point_delta_k: dict, draws_delta_k: dict, eps_ni: float,
                            alpha: float = 0.05) -> dict:
    """M1b：simultaneous **non-inferiority 认证**。

    PRIMARY   = 逐块**边际**单侧 (1−α) 下界（IUT，无多重性校正）
    SECONDARY = family-wise simultaneous 下界（更保守的敏感性分析）
    "没有单 block 显著击败 S*" **不构成通过** —— 必须正面认证。
    """
    names = list(point_delta_k)
    marg = {k: marginal_lower_bound(draws_delta_k[k], alpha) for k in names}
    sim = simultaneous_bounds(point_delta_k, draws_delta_k, alpha)
    failed = [k for k in names if marg[k] <= -eps_ni]
    return {
        "eps_ni": float(eps_ni),
        "primary": "marginal one-sided lower bounds (IUT; no multiplicity correction)",
        "L_marg": marg, "L_sim": sim["lower"],
        "certified": len(failed) == 0,
        "status": "CERTIFIED" if not failed else "NOT CERTIFIED",
        "failed_blocks": failed,
        "wording": ("S* is non-inferior to each of the selected blocks "
                    "(marginal one-sided 95%, margin eps_NI, IUT overall alpha<=0.05 "
                    "*nominal*)" if not failed else
                    "NOT CERTIFIED — the core claim that Delta_H is an increment over a "
                    "strong surface baseline does not hold"),
        "must_not_write": "family-wise certified",
    }


def direct_superiority_iva(point_direct_k: dict, draws_direct_k: dict,
                           alpha: float = 0.05) -> dict:
    """M1b(iv-a)：**唯一授权**「S*+H 超过每一个单 block」的判据。

    逐块边际单侧 (1−α) 下界 > 0，且对全部 block 成立（IUT）。
    """
    names = list(point_direct_k)
    marg = {k: marginal_lower_bound(draws_direct_k[k], alpha) for k in names}
    failed = [k for k in names if marg[k] <= 0.0]
    return {"criterion": "iv-a (PRIMARY, sole authorising path)",
            "L_marg": marg, "passed": len(failed) == 0, "failed_blocks": failed,
            "note": "iv-b is a sensitivity/consistency check only and never "
                    "authorises the claim on its own; no OR of the two paths"}
