"""真实 confirmatory analyze callback —— **唯一被允许读 label / 算性能的地方**。

本模块只**构造**回调；调用它才会第一次读 label。执行入口
`run_stage1_v3_confirmatory.run_execute_all` 保证的顺序是：

    preflight → record_unblinding → analyze_factory() → analyze()（首次读 label）→ 写结果

三条分析路径在解盲前写死（`v3_analysis_paths.FROZEN_PATHS`），解盲后不得增删。

**结果契约（预注册 §6「各分支必报」，由 `validate_result_contract` 强制）**

    每条路径 × 每臂：
      AUROC(S*) / AUROC(H alone) / AUROC(S*+H) / 各单 block B_k
      M1b 非劣性认证：L_k^marg（primary）+ L_k^sim（secondary）+ CERTIFIED/NOT CERTIFIED
      M1b(iv-a)  逐块直接检验（**唯一授权路径**）
      M1b(iv-b)  合取一致性检查（sensitivity，**不单独授权任何表述**）
      M1c T1     Δ_task 的单侧**上界** + 角色声明（角色 I 默认 / 角色 II 需同时界）
      Δ̂_H、CI、precision、CI 宽度及其 MC 误差
      R / B_per_batch / B_batch / B_total 的实际取值
    每条路径：
      E2 可估性判定（non-estimable → 只报结构，不报任何 AUROC）
      E3 判定与结果（M2d-gate；non-estimable → 只报每折 n / 正负例 / mention retention）
      M2c label-informed sensitivity + 三项 caveat
      contrast D
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import v3_adapter as adapter
from . import v3_adapter_mcq as amcq
from . import v3_analysis_paths as paths
from . import v3_arm_contrast as contrast
from . import v3_bootstrap as vb
from . import v3_e3_repool as repool
from . import v3_grouping as grouping
from . import v3_population_declaration as pop
from . import v3_repeat_estimand as vre
from . import v3_shadow_f5 as shadow_f5
from . import v3_shadow_overlay as overlay
from . import v3_stacking as vs
from .v3_blocks import BlockSpec
from .v3_config import BLOCK_NAME_H, V3AnalysisConfig

ANALYSIS_SCHEMA = "stage1_v3_confirmatory_analysis_v2"

#: 跨模型复现的结果 schema —— 与 confirmatory **分开命名**，防止两类结果被同一个
#: 下游读取器不加区分地消费。
REPLICATION_ANALYSIS_SCHEMA = "stage1_v3_cross_model_replication_analysis_v1"

ROLE_CONFIRMATORY = "confirmatory"
ROLE_CROSS_MODEL_REPLICATION = "cross_model_replication_non_confirmatory"
ALLOWED_ANALYSIS_ROLES: tuple[str, ...] = (ROLE_CONFIRMATORY,
                                           ROLE_CROSS_MODEL_REPLICATION)

#: RQ2 的 frozen primary 状态 —— 由 F5 数值完整性审计裁定，写死在这里。
RQ2_FROZEN_STATUS = "not_interpretable_due_to_invalid_frozen_stratifier"

#: RQ2 在跨模型复现中的状态。RQ2 **不在** v3 预注册内，且现行 analyzer 的 RQ2
#: 假设是 Qwen 专有的 —— 因此本角色下既不计算也不报告，**更不得**写成 Llama 发现。
RQ2_REPLICATION_STATUS = "not_in_stage1_v3_cross_model_replication_scope"

#: M1c T1：canonical 全文 block 与两个 shortcut（M1c 逐字冻结）。
T1_CANONICAL_BLOCK = "B3"
T1_SHORTCUT_BLOCKS: tuple[str, ...] = ("B1", "B2")

#: §6 每臂必报键；`validate_result_contract` 逐项核。
REQUIRED_ARM_KEYS: tuple[str, ...] = (
    "path", "n", "auroc", "delta_h_point", "delta_h_ci", "precision",
    "ci_width_and_mc_error", "m1b_non_inferiority", "m1b_iv_a_direct_superiority",
    "m1b_iv_b_consistency", "m1c_t1", "bootstrap_budget", "wording_prohibitions",
)
REQUIRED_AUROC_KEYS: tuple[str, ...] = ("S*", "S*+H", "H")
REQUIRED_PATH_KEYS: tuple[str, ...] = ("arms", "contrast_D", "estimands")


class ContractViolation(RuntimeError):
    """正式结果缺少预注册 §6 的必报项 —— 停机，不发布半个结果。"""


# ────────────────────────────────────────────────────────────── 小工具
def _marginal_upper_bound(values, alpha: float = 0.05) -> float:
    """边际单侧 (1−α) **上界** = 合并分布的 (1−α) 分位。

    `v3_bootstrap.marginal_lower_bound` 的镜像。M1c 的判据方向是上界，
    **不得**拿下界凑数（M3(1) 明写两个方向枢轴不同、不可共用）。
    """
    return float(np.percentile(np.asarray(values, dtype=float), 100 * (1 - alpha)))


def _subset(d: Mapping[str, Any], names: Sequence[str]) -> dict:
    return {k: d[k] for k in names if k in d}


def _finite(x) -> float | None:
    v = float(x)
    return v if np.isfinite(v) else None


# ────────────────────────────────────────────────────── M1c T1 诊断
def t1_diagnostic(point_delta_k: Mapping[str, float],
                  draws_delta_k: Mapping[str, np.ndarray], cfg: V3AnalysisConfig) -> dict:
    """M1c T1：Δ_task = AUROC(B3) − AUROC(shortcut)，shortcut ∈ {B1, B2}。

    **代数恒等式**（不是近似）：Δ_task = Δ_k[shortcut] − Δ_k[B3]，因为两者共有
    减数 AUROC(S*)。跨 repeat 平均是线性的，故逐 draw 相减与"先算 Δ_task 再平均"
    逐元素相同 —— 因此不需要另跑一轮 bootstrap。

    判据：单侧 **上界** < +ε → T1 触发。
      角色 I（默认，M5）：**边际**上界，保守 admission gate，不做多重性校正；
                          据此只能说「按预注册的保守准入准则，我们不作 X 表述」。
      角色 II（若要主张"该任务是表层可解的"这一**发现**）：必须改用 M3(1) 的
                          **同时**上界（max-statistic T^U，m=2）并明示切换角色。
    """
    missing = [b for b in (T1_CANONICAL_BLOCK,) + T1_SHORTCUT_BLOCKS
               if b not in point_delta_k]
    if missing:
        return {"computed": False, "reason": f"缺 block {missing}，无法构造 Δ_task",
                "canonical_block": T1_CANONICAL_BLOCK,
                "shortcut_blocks": list(T1_SHORTCUT_BLOCKS)}

    point = {b: float(point_delta_k[b] - point_delta_k[T1_CANONICAL_BLOCK])
             for b in T1_SHORTCUT_BLOCKS}
    draws = {b: (np.asarray(draws_delta_k[b], dtype=float)
                 - np.asarray(draws_delta_k[T1_CANONICAL_BLOCK], dtype=float))
             for b in T1_SHORTCUT_BLOCKS}
    eps = float(cfg.eps_auroc)
    u_marg = {b: _marginal_upper_bound(draws[b], cfg.alpha) for b in T1_SHORTCUT_BLOCKS}
    triggered = {b: bool(u_marg[b] < eps) for b in T1_SHORTCUT_BLOCKS}
    # 角色 II 的同时上界可能退化（Δ_task 的 bootstrap 标准差 < 1e-8，例如两个 block
    # 的分数完全相同）。那时**只**让角色 II 不可用并如实记录 —— 绝不让整个
    # analyze() 在解盲之后崩掉，也不拿边际界冒充同时界。
    try:
        sim = vb.simultaneous_bounds(point, draws, cfg.alpha)
        u_sim: dict = dict(sim["upper"])
        sim_coverage = sim["coverage"]
        sim_degenerate = list(sim["degenerate_blocks"])
        sim_available = True
    except RuntimeError as exc:
        u_sim, sim_available = {}, False
        sim_degenerate = list(T1_SHORTCUT_BLOCKS)
        sim_coverage = f"unavailable: {exc}"
    triggered_sim = {b: bool(u_sim[b] < eps) for b in u_sim}

    any_marg = any(triggered.values())
    return {
        "computed": True,
        "definition": "Delta_task = AUROC(B3 canonical/full text) - AUROC(shortcut)",
        "canonical_block": T1_CANONICAL_BLOCK,
        "shortcut_blocks": list(T1_SHORTCUT_BLOCKS),
        "algebraic_identity": "Delta_task = delta_k[shortcut] - delta_k[B3] (exact)",
        "eps": eps,
        "point": point,
        "role_I_marginal_upper_bounds": u_marg,
        "role_I_triggered_per_shortcut": triggered,
        "triggered": any_marg,
        "role": "I",
        "role_declaration": (
            "角色 I｜保守 admission gate（M5）：不做多重性校正；据此我们**放弃**机制性"
            "表述，但**不得**声称在全局 alpha=.05 下证明了该任务无效"),
        "role_II_simultaneous_upper_bounds": u_sim,
        "role_II_triggered_per_shortcut": triggered_sim,
        "role_II_available": sim_available,
        "role_II_degenerate_blocks": sim_degenerate,
        "role_II_note": (
            "若要把「该任务是表层可解的」写成**发现**，须切换到角色 II 并引用上面的"
            "M3(1) 同时上界（max-statistic T^U，m=2，nominal 覆盖）"),
        "consequence_if_triggered": (
            "禁**机制性**表述（语义理解 / 内部觉察 / 模型知道自己在编造）；"
            "**不禁** E1 的 benchmark-conditional Delta_H 的报告（M1c）"),
        "simultaneous_coverage": sim_coverage,
    }


def iv_b_consistency(non_inferiority: Mapping[str, Any], ci_dh: Mapping[str, float],
                     eps_ni: float) -> dict:
    """M1b(iv-b)：**合取**一致性检查 —— sensitivity，不单独授权任何表述。

    必须写成「M1b CERTIFIED **且** Δ_H 的 CI 下界 > ε_NI」这个合取；
    只写第二项是预注册明令撤回的写法。
    """
    certified = bool(non_inferiority.get("certified"))
    ci_lo = float(ci_dh["ci_lo"])
    second = bool(ci_lo > float(eps_ni))
    return {
        "criterion": "iv-b (SENSITIVITY / consistency check only)",
        "conjunction_required": "M1b CERTIFIED AND Delta_H CI lower bound > eps_NI",
        "m1b_certified": certified,
        "delta_h_ci_lo": ci_lo,
        "eps_ni": float(eps_ni),
        "second_term_holds": second,
        "holds": bool(certified and second),
        "must_not": ("不得只写第二项；不得与 (iv-a) 取并集作正面主张 —— "
                     "两条路径的并集会重新引入择优触发与 look-elsewhere"),
    }


def wording_prohibitions(non_inf: Mapping[str, Any], iv_a: Mapping[str, Any],
                         t1: Mapping[str, Any]) -> dict:
    """§6 的三条禁令，按本次结果实例化写进产物。"""
    out = {
        "m1b_not_certified_forbids": None,
        "iv_a_failed_forbids": None,
        "t1_triggered_forbids": None,
        "never_allowed": ("(iv-a) 通过 OR (iv-b) 通过 → 作正面主张 —— **绝不允许**"),
    }
    if not non_inf.get("certified"):
        out["m1b_not_certified_forbids"] = (
            "NOT CERTIFIED：核心主张「Δ_H 是相对强表层基线的增量」**作废**；"
            "禁「通过强表层基线」「超出输出可观测」类表述；"
            "Δ_H 只能限定为「相对该特定 S* 构造的增量」")
    if not iv_a.get("passed"):
        out["iv_a_failed_forbids"] = (
            "(iv-a) 未通过：**禁**「超过最佳表层 block」类表述 —— 无论 (iv-b) 是否成立")
    if t1.get("triggered"):
        out["t1_triggered_forbids"] = (
            "T1 触发：禁**机制性**表述；**不禁** E1 的报告")
    return out


# ─────────────────────────────────────────────── 单臂 × 单路径（E1 口径）
def _summarise(y, groups, blocks: Sequence[BlockSpec], cfg: V3AnalysisConfig, path: str,
               *, reps: Sequence[dict] | None = None,
               estimand: str = "E1", label: str | None = None,
               progress=None, progress_context: Mapping[str, Any] | None = None) -> dict:
    """单臂单路径：repeats → 点估计 → bootstrap → 全部 §6 必报项。

    `reps` 可注入（E3 用固定折的 repeats）；否则按 path 的标准折构造。

    `progress` 为可选的**纯进度**回调（阶段/编号/耗时/错误状态，无任何统计量）；
    省略即完全不调用，行为与加日志之前一致。
    """
    ctx = {**(dict(progress_context) if progress_context else {}),
           "estimand": estimand}
    if reps is None:
        reps = paths.run_all_repeats_for_path(y=y, groups=groups, blocks=blocks,
                                              cfg=cfg, path=path,
                                              progress=progress,
                                              progress_context=ctx)
    names_s, names_sh = paths.names_for_path(path)
    # AUROC 必报项含 H alone → 点估计用**六块**名单；bootstrap 仍只跑五块，
    # 因为 M1b / iv-a 的族就是那 5 个单 block（H 不是 S* 的组成块）。
    point = vre.point_estimates(y, reps, names_sh)
    plan = vb.build_resample_plan(groups, cfg)
    paths.emit_progress(progress, "bootstrap_start", path=path,
                        b_total=cfg.b_total, b_batch=cfg.b_batch, **ctx)
    _t_boot = time.time()
    draws = vb.pooled_draws(y, reps, names_s, plan)
    paths.emit_progress(progress, "bootstrap_done", path=path,
                        elapsed_s=round(time.time() - _t_boot, 3), status="ok", **ctx)

    dh = np.asarray(draws[vre.DELTA_H], dtype=float)
    ci_dh = vb.percentile_ci(dh, cfg.alpha)
    jack = vb.jackknife_mc(dh, draws["batch_of"], cfg.b_batch, cfg.alpha)
    bm = vb.batch_means_mc(dh, draws["batch_of"], cfg.b_batch, cfg.alpha)
    verdict = vb.precision_verdict_v3(ci_dh["width"], jack["mc_band_width"],
                                      cfg.precision_max_ci_width)

    dk_point = _subset(point[vre.DELTA_K], names_s)
    dd_point = _subset(point[vre.DELTA_DIRECT_K], names_s)
    dk_draws = {k: np.asarray(v) for k, v in draws[vre.DELTA_K].items()}
    dd_draws = {k: np.asarray(v) for k, v in draws[vre.DELTA_DIRECT_K].items()}

    non_inf = vb.certify_non_inferiority(dk_point, dk_draws, cfg.eps_ni, cfg.alpha)
    iv_a = vb.direct_superiority_iva(dd_point, dd_draws, cfg.alpha)
    iv_b = iv_b_consistency(non_inf, ci_dh, cfg.eps_ni)
    t1 = t1_diagnostic(dk_point, dk_draws, cfg)

    auroc = {"S*": _finite(point["auroc_S*"]), "S*+H": _finite(point["auroc_S*+H"])}
    for k in names_sh:
        auroc[k] = _finite(point["auroc_block"][k])
    if BLOCK_NAME_H not in auroc:
        auroc[BLOCK_NAME_H] = None

    return {
        "path": path, "estimand": estimand, "n": int(np.asarray(y).size),
        "label_caliber": label or "frozen completion_label (E1)",
        "block_names_s_star": names_s, "block_names_s_star_plus_h": names_sh,
        "n_groups": int(len(set(map(str, np.asarray(groups).tolist())))),
        "auroc": auroc,
        "delta_h_point": _finite(point[vre.DELTA_H]), "delta_h_ci": ci_dh,
        "delta_k_point": dk_point, "delta_direct_k_point": dd_point,
        "m1b_non_inferiority": non_inf,
        "m1b_simultaneous_secondary": vb.simultaneous_bounds(dk_point, dk_draws, cfg.alpha),
        "m1b_iv_a_direct_superiority": iv_a,
        "m1b_iv_b_consistency": iv_b,
        "m1c_t1": t1,
        "wording_prohibitions": wording_prohibitions(non_inf, iv_a, t1),
        "precision": verdict,
        "ci_width_and_mc_error": {
            "ci_width": ci_dh["width"],
            "mc_jackknife": {k: jack[k] for k in
                             ("se_jack", "mc_band_width", "t_quantile")},
            "mc_batch_means_secondary": {"se_mc": bm["se_mc"]},
            "borderline_rule": "若 0.04 落在误差带内 → borderline（见 precision.verdict）",
        },
        "bootstrap_budget": {
            "R": int(cfg.n_repeats), "B_per_batch": int(cfg.b_per_batch),
            "B_batch": int(cfg.b_batch), "B_total": int(cfg.b_total),
            "n_valid_draws": int(draws["n_draws"]),
            "n_discarded_single_class": int(draws["n_discarded_single_class"]),
        },
        "_oof_per_repeat": [r["oof"] for r in reps],
        "_repeats": reps,
    }


def _strip_internal(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if not k.startswith("_")}


# ───────────────────────────────────────── E3：固定折（词表块即 outer fold）
def run_fixed_fold_repeats(*, y, groups, blocks: Sequence[BlockSpec], folds,
                           cfg: V3AnalysisConfig, path: str,
                           cache_builder=vs.build_base_block_cache) -> list[dict]:
    """用**给定**的 outer folds 跑 R 个 repeat。

    E3 的词表块**本身就是 outer fold**（M2d，F = n_outer_folds = 5），
    因此不能走 `stratified_grouped_folds`。除折来源外，其余与
    `v3_analysis_paths.run_repeat_for_path` 逐行同构：同一 `build_base_block_cache`
    与 `fit_stack`，repeat 仍驱动 inner CV 的 seed。
    """
    blocks = paths.blocks_for_path(blocks, path)
    names_s, names_sh = paths.names_for_path(path)
    y = np.asarray(y)
    out = []
    for r in range(cfg.n_repeats):
        n = y.size
        oof = {"S*": np.full(n, np.nan), "S*+H": np.full(n, np.nan)}
        for b in blocks:
            oof[b.name] = np.full(n, np.nan)
        audits = []
        for f, (train, test) in enumerate(folds):
            cache = cache_builder(y=y, groups=np.asarray(groups), blocks=blocks,
                                  repeat=r, outer_fold=f, train=train, test=test, cfg=cfg)
            oof["S*"][test] = vs.fit_stack(cache, names_s, y, cfg)
            oof["S*+H"][test] = vs.fit_stack(cache, names_sh, y, cfg)
            for name, sc in cache.test_scores.items():
                oof[name][test] = sc
            audits.append(cache.audit())
        out.append({"repeat": r, "oof": oof, "audits": audits, "path": path,
                    "outer_folds": "fixed (E3 vocabulary blocks)",
                    "block_names_s_star": names_s,
                    "block_names_s_star_plus_h": names_sh})
    return out


def _e3_block_hidden(records: Sequence[dict], rows, fold_of, e3_blocks: Mapping[str, Any],
                     dim: int) -> np.ndarray:
    """block-specific H：从 canonical repool 的 `(unit_id, block)` 向量取。

    E3 的 H 是「只在该块内 mention 上 pool」的向量 —— 正是 repool cache 的内容。
    缺任何一条即抛错，**不补零、不回退 E1 的 H**。
    """
    vectors = e3_blocks["vectors"]
    out = np.empty((len(rows), dim), dtype=float)
    for j, (i, b) in enumerate(zip(rows, np.asarray(fold_of).tolist())):
        uid = repool.derive_unit_id(records[i])
        key = (uid, int(b))
        if key not in vectors:
            raise ContractViolation(
                f"E3 缺 block-specific H：{key} 不在 canonical repool cache 里；"
                f"**不得**补零或退回 E1 的 H")
        out[j] = np.asarray(vectors[key], dtype=float)
    return out


# ─────────────────────────────────────────────────────── 结果契约校验
def validate_result_contract(result: Mapping[str, Any]) -> dict:
    """逐项核 §6「各分支必报」。缺任何一项 → `ContractViolation`。"""
    problems: list[str] = []
    got_schema = result.get("schema")
    role = result.get("population_role", ROLE_CONFIRMATORY)
    want_schema = (REPLICATION_ANALYSIS_SCHEMA
                   if role == ROLE_CROSS_MODEL_REPLICATION else ANALYSIS_SCHEMA)
    if got_schema != want_schema:
        problems.append(f"schema={got_schema!r} != {want_schema} (role={role})")
    if role == ROLE_CROSS_MODEL_REPLICATION:
        # 复现结果必须自证三件事，否则它可能被误读成 confirmatory。
        if result.get("is_confirmatory") is not False:
            problems.append("复现结果的 is_confirmatory 必须为 False")
        rq2 = result.get("rq2") or {}
        if rq2.get("status") != RQ2_REPLICATION_STATUS or rq2.get("computed") is not False:
            problems.append(
                f"复现角色下 rq2 必须是 status={RQ2_REPLICATION_STATUS} / computed=False")
        ni = result.get("numeric_integrity") or {}
        if ni.get("status") not in paths.NUMERIC_STATES:
            problems.append("复现结果必须带 numeric_integrity.status（三态之一）")
    fp = tuple(result.get("frozen_paths") or ())
    if fp != tuple(paths.FROZEN_PATHS):
        problems.append(f"frozen_paths={fp} != {tuple(paths.FROZEN_PATHS)}")
    for key in REQUIRED_PATH_KEYS:
        if key not in result:
            problems.append(f"缺顶层键 {key}")

    for path in paths.FROZEN_PATHS:
        arms = (result.get("arms") or {}).get(path)
        if not arms:
            problems.append(f"{path}: 缺 arms")
            continue
        for arm in ("h1", "mcq"):
            a = arms.get(arm)
            if not a:
                problems.append(f"{path}/{arm}: 缺臂结果")
                continue
            for k in REQUIRED_ARM_KEYS:
                if k not in a:
                    problems.append(f"{path}/{arm}: 缺 {k}")
            for k in REQUIRED_AUROC_KEYS:
                if k not in (a.get("auroc") or {}):
                    problems.append(f"{path}/{arm}: auroc 缺 {k}")
            for k in a.get("block_names_s_star", []):
                if k not in (a.get("auroc") or {}):
                    problems.append(f"{path}/{arm}: auroc 缺单 block {k}")
            ni = a.get("m1b_non_inferiority") or {}
            if ni.get("status") not in ("CERTIFIED", "NOT CERTIFIED"):
                problems.append(f"{path}/{arm}: M1b status 必须是 CERTIFIED/NOT CERTIFIED")
            for k in ("L_marg", "L_sim"):
                if k not in ni:
                    problems.append(f"{path}/{arm}: M1b 缺 {k}")
            iva = a.get("m1b_iv_a_direct_superiority") or {}
            if "passed" not in iva or "L_marg" not in iva:
                problems.append(f"{path}/{arm}: 缺 (iv-a) 的 passed / L_marg")
            ivb = a.get("m1b_iv_b_consistency") or {}
            if "holds" not in ivb or "m1b_certified" not in ivb:
                problems.append(f"{path}/{arm}: (iv-b) 必须写成合取（缺 holds 或 certified）")
            t1 = a.get("m1c_t1") or {}
            if "role" not in t1 or "role_declaration" not in t1:
                problems.append(f"{path}/{arm}: T1 缺角色声明")
            if t1.get("computed") and "role_I_marginal_upper_bounds" not in t1:
                problems.append(f"{path}/{arm}: T1 缺单侧上界")
            bb = a.get("bootstrap_budget") or {}
            for k in ("R", "B_per_batch", "B_batch", "B_total"):
                if k not in bb:
                    problems.append(f"{path}/{arm}: bootstrap_budget 缺 {k}")

        est = (result.get("estimands") or {}).get(path)
        if not est:
            problems.append(f"{path}: 缺 estimands")
            continue
        for name in ("E1", "E2", "E3", "M2c"):
            if name not in est:
                problems.append(f"{path}: estimands 缺 {name}")
        e2 = (est.get("E2") or {}).get("h1") or {}
        if e2.get("estimability") == grouping.NON_ESTIMABLE:
            if e2.get("performance_estimates_computed"):
                problems.append(f"{path}: E2 non-estimable 却报了性能量")
        elif e2.get("estimability") == grouping.ESTIMABLE and "summary" not in e2:
            problems.append(f"{path}: E2 estimable 却没有按 E1 口径估计")
        e3 = (est.get("E3") or {}).get("h1") or {}
        if e3.get("estimability") == grouping.NON_ESTIMABLE:
            if e3.get("performance_estimates_computed"):
                problems.append(f"{path}: E3 non-estimable 却报了性能量")
            for k in ("per_fold", "retention"):
                if k not in e3:
                    problems.append(f"{path}: E3 non-estimable 必须报 {k}")
        elif e3.get("estimability") == grouping.ESTIMABLE and "summary" not in e3:
            problems.append(f"{path}: E3 estimable 却没有结果")
        m2c = (est.get("M2c") or {}).get("h1") or {}
        caveats = m2c.get("interpretation_caveats") or []
        if len(caveats) != 3:
            problems.append(f"{path}: M2c 必须带三项 interpretation caveat")
        if m2c.get("is_primary") is not False:
            problems.append(f"{path}: M2c 必须显式声明 is_primary=False")
        if path not in (result.get("contrast_D") or {}):
            problems.append(f"{path}: 缺 contrast_D")

    if problems:
        raise ContractViolation("正式结果不满足预注册 §6 必报项：\n  "
                                + "\n  ".join(problems))
    return {"ok": True, "n_paths": len(paths.FROZEN_PATHS),
            "checked": list(REQUIRED_ARM_KEYS)}


# ─────────────────────────────────────────────────────────── 回调构造
def make_analyze_callback(*, cfg: V3AnalysisConfig, h1_records: Sequence[dict],
                          h1_hidden: Mapping[str, Any], mcq_records: Sequence[dict],
                          mcq_hidden: Mapping[str, Any],
                          shadow_overlay: Mapping[str, Any] | None,
                          e3_blocks: Mapping[str, Any] | None,
                          label_free_check: bool = True,
                          population_role: str = ROLE_CONFIRMATORY,
                          result_schema: str | None = None,
                          numeric_integrity_status: str | None = None,
                          numeric_integrity_evidence: Mapping[str, Any] | None = None,
                          run_fingerprint: Mapping[str, Any] | None = None,
                          observed_counts: Mapping[str, Any] | None = None,
                          progress=None) -> Callable[[], dict]:
    """构造真实回调。**返回的函数被调用之前，不读任何 label。**

    默认参数 = Qwen confirmatory 的既有行为，逐项不变。跨模型复现必须显式传
    `population_role` / `result_schema` / `numeric_integrity_status` ——
    尤其最后一项：**省略即 `not_audited`，绝不会默认成"已通过"**。
    """
    paths.assert_frozen_paths(paths.FROZEN_PATHS)
    if population_role not in ALLOWED_ANALYSIS_ROLES:
        raise ValueError(f"population_role={population_role!r} 不在白名单 "
                         f"{list(ALLOWED_ANALYSIS_ROLES)} 内")
    is_replication = population_role == ROLE_CROSS_MODEL_REPLICATION
    if numeric_integrity_status is None:
        # confirmatory 沿用 Qwen 已裁定的 compromised；复现角色下**必须显式给**，
        # 未给就是 not_audited —— 不允许"没审计"悄悄等价于"没问题"。
        numeric_status = (paths.NUMERIC_NOT_AUDITED if is_replication
                          else paths.DEFAULT_NUMERIC_STATUS)
    else:
        numeric_status = paths.assert_numeric_status(numeric_integrity_status)
    schema = result_schema or (REPLICATION_ANALYSIS_SCHEMA if is_replication
                               else ANALYSIS_SCHEMA)
    if is_replication and schema == ANALYSIS_SCHEMA:
        raise ValueError(
            "跨模型复现结果不得使用 confirmatory 的 schema —— 两类结果必须能被下游"
            "区分开")
    if shadow_overlay is None or shadow_overlay.get("gate_passed") is not True:
        raise overlay.OverlayRefused("Path B overlay 未过硬门 —— 拒绝构造 analyze 回调")
    src = shadow_overlay.get("overlay_source")
    if src not in overlay.ACCEPTED_OVERLAY_SOURCES:
        raise overlay.OverlayRefused(
            f"Path B overlay 来源 {src!r} 不在被接受集合 "
            f"{list(overlay.ACCEPTED_OVERLAY_SOURCES)} 内 —— 解盲前已裁定 Path B 的唯一实现"
            f"为 {overlay.OVERLAY_SOURCE_TAIL}；旧 full-BF16 shadow 是 "
            f"superseded_before_unblinding / engineering_historical_evidence_only / "
            f"not_joined_with_labels，**不得**接 label，也不得与 tail 同时分析")
    if e3_blocks is None:
        raise overlay.OverlayRefused("E3 blocks 未加载 —— 拒绝构造 analyze 回调")
    if e3_blocks.get("canonical_source") != "repool":
        raise overlay.OverlayRefused(
            f"E3 正式输入必须是 canonical repool，收到 "
            f"{e3_blocks.get('canonical_source')!r} —— E3 是独立 estimand，"
            f"**不随 Path B 改变**；tail 的 E3 cache 只作全量交叉验证证据")

    def analyze() -> dict:
        out: dict[str, Any] = {
            "schema": schema,
            "population_role": population_role,
            "is_confirmatory": not is_replication,
            "numeric_integrity": {
                "status": numeric_status,
                "b4_scope": "H1 B4 的 logit-derived F5 字段",
                "evidence": (dict(numeric_integrity_evidence)
                             if numeric_integrity_evidence else None),
                "fail_closed_note": ("缺失审计记为 not_audited，**不**等价于通过；"
                                     "Qwen 的 compromised 裁定不自动继承给其他模型"),
            },
            "frozen_paths": list(paths.FROZEN_PATHS),
            "path_markers": {p: paths.path_markers(p, numeric_status)
                             for p in paths.FROZEN_PATHS},
            "arms": {}, "contrast_D": {}, "estimands": {},
        }
        if run_fingerprint is not None:
            out["run_fingerprint"] = dict(run_fingerprint)
        if observed_counts is not None:
            out["observed_counts"] = dict(observed_counts)
        bundles: dict[str, dict] = {}
        # E2 / M2c 的分组构造与路径无关，只算一次。
        e2_audit = grouping.build_e2_groups(list(h1_records), arm="h1", cfg=cfg)
        m2c = grouping.build_m2c_groups(list(h1_records), cfg, arm="h1")
        e3_plan = grouping.build_e3_analysis(list(h1_records), cfg, arm="h1")

        for path in paths.FROZEN_PATHS:
            h1_recs = (overlay.apply_overlay(h1_records, shadow_overlay, "h1")
                       if path == paths.PATH_B else list(h1_records))
            mcq_recs = (overlay.apply_overlay(mcq_records, shadow_overlay, "mcq")
                        if path == paths.PATH_B else list(mcq_records))
            bh = adapter.build_arm(h1_recs, dict(h1_hidden))
            bm = amcq.build_mcq_arm(amcq.filter_analyzable(mcq_recs), dict(mcq_hidden))
            sh = _summarise(bh["y"], bh["groups"], bh["blocks"], cfg, path,
                            progress=progress, progress_context={"arm": "h1"})
            sm = _summarise(bm["y"], bm["groups"], bm["blocks"], cfg, path,
                            progress=progress, progress_context={"arm": "mcq"})
            bundles[path] = {"h1": (bh, sh), "mcq": (bm, sm)}
            out["arms"][path] = {"h1": _strip_internal(sh), "mcq": _strip_internal(sm)}

            # ---- D：每条路径各算一份，标记各不相同 -------------------------
            d = contrast.contrast_d(
                y_mcq=bm["y"], groups_mcq=bm["groups"],
                oof_mcq=[o["S*+H"] for o in sm["_oof_per_repeat"]],
                y_h1=bh["y"], groups_h1=bh["groups"],
                oof_h1=[o["S*+H"] for o in sh["_oof_per_repeat"]], cfg=cfg)
            # Path A 的 D 标记随**本次**数值审计生成；B/C 与冻结 B4 数值无关，保持原样。
            if path == paths.PATH_A:
                dm = paths.d_markers(numeric_status)
                d["marker"] = dm["marker"]
                d["interpretation_limit"] = dm["interpretation_limit"]
                d["numeric_integrity_status"] = dm["numeric_integrity_status"]
            else:
                d["marker"] = {
                    paths.PATH_B: "interpretable_sensitivity_symmetric_overlay",
                    paths.PATH_C: "dependency_diagnostic_only",
                }[path]
                d["interpretation_limit"] = {
                    paths.PATH_B: "两臂使用同一 tail overlay，可作解释性 sensitivity",
                    paths.PATH_C: "只作 B4 依赖性诊断",
                }[path]
            out["contrast_D"][path] = d

            # ---- E1 / E2 / E3 / M2c（按预注册身份输出）--------------------
            out["estimands"][path] = {
                "E1": {"role": "PRIMARY", "grouping": "A (prompt-disjoint)",
                       "h1": {"n": int(bh["y"].size),
                              "delta_h_point": sh["delta_h_point"],
                              "delta_h_ci": sh["delta_h_ci"]},
                       "mcq": {"n": int(bm["y"].size),
                               "delta_h_point": sm["delta_h_point"],
                               "delta_h_ci": sm["delta_h_ci"]}},
                "E2": {"h1": _e2_branch(e2_audit, bh, cfg, path),
                       "mcq": grouping.not_applicable("mcq", "E2")},
                "E3": {"h1": _e3_branch(e3_plan, h1_recs, bh, e3_blocks, cfg, path),
                       "mcq": grouping.not_applicable("mcq", "E3")},
                "M2c": {"h1": _m2c_branch(m2c, bh, cfg, path),
                        "mcq": grouping.not_applicable("mcq", "M2c")},
            }

        # ---- RQ2 状态分支 ---------------------------------------------
        if is_replication:
            # RQ2 不在 v3 预注册内，且 analyzer 现行 RQ2 状态含 Qwen 专有假设。
            # 保留字段只为 schema 兼容，**不得**把它写成 Llama 的复现发现。
            out["rq2"] = {
                "status": RQ2_REPLICATION_STATUS,
                "computed": False,
                "reason": ("RQ2 不在 preregistration_v3 内，现行 RQ2 状态又含 Qwen "
                           "专有假设；本次跨模型复现**不计算、不报告**。"
                           "若日后需要，须另行定义为独立的 posthoc sensitivity"),
                "retained_for": "schema 兼容",
            }
        else:
            out["rq2"] = {
                "frozen_primary": {
                    "status": RQ2_FROZEN_STATUS,
                    "computed": False,
                    "reason": ("frozen stratifier 依赖受非有限值静默过滤影响的 H1 F5 字段；"
                               "**不静默换 stratifier**，也不输出具有名义含义的 frozen RQ2 结论"),
                },
                "stable_shadow": {
                    "status": "protocol_deviation_sensitivity",
                    "is_frozen_primary": False,
                    "note": "可作 sensitivity；明确**不是** frozen primary",
                },
            }

        # ---- M4 population declaration ---------------------------------
        bh_a = bundles[paths.PATH_A]["h1"][0]
        bm_a = bundles[paths.PATH_A]["mcq"][0]
        out["population_declaration"] = pop.build_declaration(
            h1={"analyzable": int(bh_a["y"].size),
                "n_positive": int((bh_a["y"] == 1).sum()),
                "n_negative": int((bh_a["y"] == 0).sum()),
                "positive_rate": float((bh_a["y"] == 1).mean())},
            mcq={"analyzable": int(bm_a["y"].size),
                 "n_positive": int((bm_a["y"] == 1).sum()),
                 "n_negative": int((bm_a["y"] == 0).sum()),
                 "positive_rate": float((bm_a["y"] == 1).mean())},
            include_label_dependent=True)
        pop.validate_declaration(out["population_declaration"])
        out["contract"] = validate_result_contract(out)
        return out

    analyze.frozen_paths = tuple(paths.FROZEN_PATHS)          # type: ignore[attr-defined]
    analyze.is_real_analyze = True                             # type: ignore[attr-defined]
    return analyze


# ─────────────────────────────────────────────────────── estimand 分支
def _e2_branch(audit: Mapping[str, Any], bundle: Mapping[str, Any],
               cfg: V3AnalysisConfig, path: str) -> dict:
    """E2：non-estimable → 只报结构；estimable → 按**与 E1 相同的协议**估计。"""
    out = {k: v for k, v in audit.items() if k != "groups"}
    if audit.get("estimability") != grouping.ESTIMABLE:
        out["performance_estimates_computed"] = False
        out["stop_note"] = ("M2b：连通图塌缩 → **不构造 CV、不报任何 AUROC/Δ_H**，"
                            "只报可估性判定与连通结构统计")
        return out
    s = _summarise(bundle["y"], np.asarray(audit["groups"]), bundle["blocks"], cfg, path,
                   estimand="E2", label="frozen completion_label (E2 同 E1 标签)")
    out["performance_estimates_computed"] = True
    out["summary"] = _strip_internal(s)
    out["protocol"] = "与 E1 相同的协议（M2b）：同 R / 同 bootstrap / 同判据"
    return out


def _e3_branch(plan: Mapping[str, Any], records: Sequence[dict],
               bundle: Mapping[str, Any], e3_blocks: Mapping[str, Any],
               cfg: V3AnalysisConfig, path: str) -> dict:
    """E3：M2d-gate 不过 → 只报每折 n / 正负例 / retention；过 → 固定折上按 E1 口径估计。"""
    out = {k: v for k, v in plan.items() if not k.startswith("_")}
    if plan.get("estimability") != grouping.ESTIMABLE:
        out["performance_estimates_computed"] = False
        out["stop_note"] = ("M2d-gate：NON-ESTIMABLE → 只报每折样本数、正/负例数、"
                            "mention retention，**不报任何 AUROC / Δ_H**")
        return out

    rows = list(plan["_rows"])
    y = np.asarray(plan["_y"], dtype=int)
    fold_of = np.asarray(plan["_fold_of"], dtype=int)
    groups = np.asarray([str(records[i]["group_id"]) for i in rows])
    h_dim = next(b.dense.shape[1] for b in bundle["blocks"] if b.name == BLOCK_NAME_H)
    hidden = _e3_block_hidden(records, rows, fold_of, e3_blocks, h_dim)

    blocks = []
    for b in bundle["blocks"]:
        if b.name == BLOCK_NAME_H:
            blocks.append(BlockSpec(BLOCK_NAME_H, "dense", dense=hidden))
        elif b.kind == "dense":
            blocks.append(BlockSpec(b.name, "dense", dense=np.asarray(b.dense)[rows]))
        else:
            blocks.append(BlockSpec(b.name, "text",
                                    text=[b.text[i] for i in rows]))
    reps = run_fixed_fold_repeats(y=y, groups=groups, blocks=blocks,
                                  folds=plan["_folds"], cfg=cfg, path=path)
    s = _summarise(y, groups, blocks, cfg, path, reps=reps, estimand="E3",
                   label="block-specific label (M2d：冻结规则限制到块内 primary mention)")
    out["performance_estimates_computed"] = True
    out["summary"] = _strip_internal(s)
    out["h_source"] = ("canonical repool 的 block-specific pooled vector "
                       "(unit_id, block)；**未**回退 E1 的 H")
    out["independence"] = "E3 的判读独立于 E1，**不要求一致** —— 标签定义不同（M2d）"
    out["n_dropped_records"] = int(len(records) - len(rows))
    return out


def _m2c_branch(m2c: Mapping[str, Any], bundle: Mapping[str, Any],
                cfg: V3AnalysisConfig, path: str) -> dict:
    """M2c：C 分组上的 label-informed sensitivity，带三项 caveat。"""
    out = {k: v for k, v in m2c.items() if k != "groups"}
    s = _summarise(bundle["y"], np.asarray(m2c["groups"]), bundle["blocks"], cfg, path,
                   estimand="M2c", label="frozen completion_label；分组为 label-informed C")
    out["performance_estimates_computed"] = True
    out["summary"] = _strip_internal(s)
    return out


# ────────────────────────────────────────────── 合成契约探针（**非真实数据**）
def synthetic_contract_probe(seed: int = 7, n_prompts: int = 30, k: int = 3) -> dict:
    """构造一份**完全合成**的输入，用于 readiness 跑真正的 end-to-end 契约检查。

    数据全部由本函数生成，**不读任何真实产物、不接触真实 label**。
    刻意让 E2 与 E3 都**可估**，从而把两条 estimable 分支也真跑一遍：
    每个 prompt 用**独有**的 identifier（→ 连通分量数 = n_prompts ≥ 5·n_outer_folds，
    最大分量占比 = k/n ≪ 0.20），而 identifier 所属块按 prompt 轮转（→ 5 个 E3 折都非空
    且两类齐全）。
    """
    from . import v3_synthetic_fixture as fx
    rng = np.random.default_rng(seed)
    # 探针只用来验**契约**，不用来验精度：预算压到最小，
    # 但 R / B_batch / B_per_batch 仍是真实字段，契约检查照样能核到。
    cfg = fx.synthetic_cfg(b_batch=2, b_per_batch=10, n_repeats=2)
    n_blocks = int(cfg.e3_n_blocks)

    # 为每个 block 预备若干 identifier，保证 5 个块都非空且两类齐全
    per_block: dict[int, list[str]] = {b: [] for b in range(n_blocks)}
    i = 0
    need = -(-n_prompts // n_blocks) + 2          # 每块要够给该块的每个 prompt 一个独有 id
    while min(len(v) for v in per_block.values()) < need and i < 200000:
        ident = f"T{1000 + i % 9000:04d}"
        per_block[grouping.e3_block_of(ident, cfg)].append(ident)
        i += 1

    recs, hidden = [], {}
    hd = 8
    for p in range(n_prompts):
        gid = f"probe_prompt_{p:04d}"
        blk = p % n_blocks
        ident = per_block[blk][(p // n_blocks) % len(per_block[blk])]
        for j in range(k):
            tid = f"probe_trace_{p:04d}_{j}"
            fab = ((p // n_blocks) + j) % 2 == 0
            lab = 1 if fab else 0
            r = {"trace_id": tid, "group_id": gid, "label": lab,
                 "example_id": f"probe_ex_{p:04d}", "family": "attack", "route": "recall",
                 "sample_type": "greedy" if j == 0 else "sampled", "sample_index": j,
                 "population_role": "confirmatory", "run_hash": "s" * 64,
                 "schema_version": "synthetic_probe_v1",
                 "id_string_text": ident,
                 "completion": f"probe completion {p}-{j} {ident}",
                 "prompt_text": f"probe prompt {p}",
                 "id_token_positions_local": [0, 1],
                 "mentions": [{"family": "attack", "normalized": ident,
                               "label": "fabricated" if fab else "grounded",
                               "token_indices": [0, 1]}]}
            for nm in fx.ci.SURFACE_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            for nm in fx.ci.F5_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            recs.append(r)
            hidden[tid] = rng.normal(size=hd) + 1.1 * lab
    mcq_recs, mcq_hidden = fx.make_mcq(n_q=24, seed=seed + 1)

    vectors = {}
    for r in recs:
        uid = repool.derive_unit_id(r)
        for b in range(n_blocks):
            if grouping.e3_block_label(r, cfg, b)["eligible"]:
                vectors[(uid, b)] = (hidden[r["trace_id"]]
                                     + 0.05 * rng.normal(size=hd)).astype(np.float32)
    return {
        "cfg": cfg,
        "h1_records": recs, "h1_hidden": hidden,
        "mcq_records": mcq_recs, "mcq_hidden": mcq_hidden,
        "shadow_overlay": {"gate_passed": True,
                           "overlay_source": overlay.OVERLAY_SOURCE_TAIL,
                           "by_arm": {
                               "h1": {r["trace_id"]: {"stable_fields": {
                                   n: float(rng.normal())
                                   for n in shadow_f5.H1_STABLE_FIELDS}} for r in recs},
                               "mcq": {r["example_id"]: {"stable_fields": {
                                   n: float(rng.normal())
                                   for n in shadow_f5.MCQ_STABLE_FIELDS}}
                                   for r in mcq_recs}},
                           "synthetic": True},
        "e3_blocks": {"vectors": vectors, "canonical_source": "repool",
                      "n_traces": len(recs), "n_pairs": len(vectors),
                      "grouping_role": "synthetic probe", "synthetic": True},
    }
