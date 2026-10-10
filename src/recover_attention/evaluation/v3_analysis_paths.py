"""三条分析路径 —— **在首次解盲前写死，解盲后不得临时增删**。

    A  frozen_implementation_primary
       完整执行原冻结 B4，保留预注册执行记录；
       但报告必须标记 `h1_b4_numeric_integrity_compromised`，
       **不得**按名义 B4 科学含义单独解释。

    B  stable_shadow_sensitivity
       两臂以 shadow `stable_fields` 覆盖对应 logit-derived B4；
       passthrough 字段继续取冻结值；
       population / split / repeat / bootstrap seeds / 模型与判读规则**全部不变**。

    C  no_B4_dependency_sensitivity
       两臂同时移除整个 B4，其他块不变；
       **只回答结论是否依赖 B4**，不得用它证明原 B4 有效。

**禁止再增加第四种变体。** `assert_frozen_paths()` 会在任何执行入口处强制这一点。
"""

from __future__ import annotations

from typing import Sequence

import time
from typing import Mapping

import numpy as np

from .. import conditional_increment as ci
from . import v3_stacking as vs
from .v3_blocks import BlockSpec
from .v3_config import BLOCK_NAME_H, BLOCK_NAMES_S_STAR, V3AnalysisConfig

PATH_A = "frozen_implementation_primary"
PATH_B = "stable_shadow_sensitivity"
PATH_C = "no_B4_dependency_sensitivity"

# ── B4 数值完整性状态 ────────────────────────────────────────────────────────
# Qwen 的 `h1_b4_numeric_integrity_compromised` 是**那次审计的结论**，不是常量。
# 另一个模型的 B4 是否同样受非有限值静默过滤影响，必须由**该次**审计决定。
# 因此三态齐备，且缺失审计**只能**落到 NOT_AUDITED —— 不得默认为通过。
NUMERIC_NOT_AUDITED = "not_audited"
NUMERIC_VERIFIED_CLEAN = "verified_clean"
NUMERIC_COMPROMISED = "compromised"
NUMERIC_STATES: tuple[str, ...] = (NUMERIC_NOT_AUDITED, NUMERIC_VERIFIED_CLEAN,
                                   NUMERIC_COMPROMISED)

#: Qwen Stage 1 confirmatory 的既有裁定 —— 保持默认，解释文字一字不变。
DEFAULT_NUMERIC_STATUS = NUMERIC_COMPROMISED

_PATH_A_MARKER_BY_STATUS = {
    NUMERIC_COMPROMISED: "h1_b4_numeric_integrity_compromised",
    NUMERIC_NOT_AUDITED: "h1_b4_numeric_integrity_not_audited",
    NUMERIC_VERIFIED_CLEAN: "h1_b4_numeric_integrity_verified_clean",
}
_PATH_A_LIMIT_BY_STATUS = {
    NUMERIC_COMPROMISED: (
        "保留预注册执行记录；**不得**按名义 B4 科学含义单独解释 —— "
        "H1 的 8 项 logit-derived F5 受非有限值静默过滤影响"),
    NUMERIC_NOT_AUDITED: (
        "保留预注册执行记录；本次运行的 H1 B4 数值完整性**尚未审计** —— "
        "**不得**按名义 B4 科学含义解释，也**不得**因未发现问题而当作已通过"),
    NUMERIC_VERIFIED_CLEAN: (
        "保留预注册执行记录；本次运行的 H1 B4 数值完整性审计**未发现**非有限值"
        "静默过滤。该结论只覆盖被审计的字段与本次运行，**不外推**到其他模型或后端"),
}
#: 跨臂 D 的标记：两臂对非有限值的处置是否对称，同样由本次审计决定。
_D_MARKER_BY_STATUS = {
    NUMERIC_COMPROMISED: "asymmetric_numeric_handling_compromised",
    NUMERIC_NOT_AUDITED: "asymmetric_numeric_handling_not_audited",
    NUMERIC_VERIFIED_CLEAN: "symmetric_numeric_handling_verified_clean",
}
_D_LIMIT_BY_STATUS = {
    NUMERIC_COMPROMISED: (
        "照冻结实现计算并留审计记录；两臂对非有限值的处置不对称 —— "
        "MCQ 硬停排除、H1 静默过滤 —— **不作名义跨臂机制解释**"),
    NUMERIC_NOT_AUDITED: (
        "照冻结实现计算并留审计记录；本次运行**尚未**审计两臂的非有限值处置是否对称，"
        "**不作名义跨臂机制解释**"),
    NUMERIC_VERIFIED_CLEAN: (
        "照冻结实现计算并留审计记录；本次审计**未发现**两臂非有限值处置不对称。"
        "该结论只覆盖本次运行，**不**据此主张跨臂机制解释成立"),
}


def assert_numeric_status(status: str | None) -> str:
    """fail-closed：`None` / 未知值一律落到 `not_audited`，**永不**落到 clean。"""
    if status is None:
        return NUMERIC_NOT_AUDITED
    if status not in NUMERIC_STATES:
        raise PathViolation(
            f"未知的 B4 数值完整性状态 {status!r}；合法值为 {list(NUMERIC_STATES)}。"
            f"缺失审计请显式传 {NUMERIC_NOT_AUDITED!r}，**不得**当作通过")
    return status


def d_markers(numeric_status: str | None = DEFAULT_NUMERIC_STATUS) -> dict:
    """路径 A 下跨臂 D 的标记与解释边界（B/C 的标记与数值审计无关，见调用方）。"""
    st = assert_numeric_status(numeric_status)
    return {"marker": _D_MARKER_BY_STATUS[st],
            "interpretation_limit": _D_LIMIT_BY_STATUS[st],
            "numeric_integrity_status": st}

#: 冻结的三条路径。顺序即报告顺序。
FROZEN_PATHS: tuple[str, ...] = (PATH_A, PATH_B, PATH_C)

PATH_SEMANTICS = {
    PATH_A: {
        "role": "primary",
        "b4_source": "frozen F5 fields (原冻结实现，未改)",
        "mandatory_marker": "h1_b4_numeric_integrity_compromised",
        "interpretation_limit": (
            "保留预注册执行记录；**不得**按名义 B4 科学含义单独解释 —— "
            "H1 的 8 项 logit-derived F5 受非有限值静默过滤影响"),
    },
    PATH_B: {
        "role": "sensitivity",
        # **写出实现**，不用泛称：只凭结果文件里的 `path_markers` 就必须能把它与
        # 已废止的 full-BF16 shadow 区分开。
        # 注意口径：**不声称尾部全程 BF16** —— bitsandbytes 的 MatMul8bitLt 会把激活
        # 内部降回 float16 做量化，真实行为由 tail_fingerprint 的 caveat 说明。
        "b4_source": (
            "tail_stabilized_single_forward stable-logit overlay"
            "（两臂对称；block 19 前冻结 H/E3 锚点逐元素一致；"
            "外部 float32 lm_head/logsumexp）+ frozen passthrough"),
        "b4_source_caveat": (
            "**不得**据此声称尾部全程 bfloat16：block 19 之后仅把残差流 cast 成 bf16，"
            "bitsandbytes MatMul8bitLt 仍会把激活内部降回 float16 做量化；"
            "真实后端行为以 tail_fingerprint.json 的 backend caveat 为准"),
        "mandatory_marker": "protocol_deviation_numerical_sensitivity",
        "interpretation_limit": (
            "**不是** frozen primary，也**不是**真值；population / split / repeat / "
            "bootstrap seeds / 模型与判读规则与 A 完全相同，只换 B4 的 logit-derived 数值"),
    },
    PATH_C: {
        "role": "diagnostic",
        "b4_source": "B4 整块移除（两臂同时）",
        "mandatory_marker": "dependency_diagnostic_only",
        "interpretation_limit": (
            "只回答『结论是否依赖 B4』；**不得**用它证明原 B4 有效"),
    },
}

#: 路径 C 的块集：B4 整块移除，其余不变。
PATH_C_S_STAR: tuple[str, ...] = tuple(n for n in BLOCK_NAMES_S_STAR if n != "B4")
PATH_C_S_STAR_PLUS_H: tuple[str, ...] = PATH_C_S_STAR + (BLOCK_NAME_H,)


class PathViolation(RuntimeError):
    """路径集合被改动 —— 停机。"""


def assert_frozen_paths(paths: Sequence[str]) -> None:
    """任何执行入口都必须先过这一关：路径集合与顺序都不得变。"""
    if tuple(paths) != FROZEN_PATHS:
        raise PathViolation(
            f"分析路径集合已冻结为 {FROZEN_PATHS}，收到 {tuple(paths)}；"
            f"解盲前写死，解盲后不得临时增删")


def blocks_for_path(blocks: Sequence[BlockSpec], path: str) -> list[BlockSpec]:
    """A / B 用完整六块；C 去掉 B4。A 与 B 的**块结构相同**，差别只在 B4 的数值。"""
    if path not in FROZEN_PATHS:
        raise PathViolation(f"未知路径 {path!r}")
    if path == PATH_C:
        return [b for b in blocks if b.name != "B4"]
    return list(blocks)


def names_for_path(path: str) -> tuple[list[str], list[str]]:
    if path == PATH_C:
        return list(PATH_C_S_STAR), list(PATH_C_S_STAR_PLUS_H)
    return list(BLOCK_NAMES_S_STAR), list(BLOCK_NAMES_S_STAR_PLUS_H := tuple(
        BLOCK_NAMES_S_STAR) + (BLOCK_NAME_H,))


def emit_progress(progress, event: str, **fields) -> None:
    """纯进度事件。**只有计数、编号、时间与错误状态，绝无任何统计量。**

    `progress=None`（默认）时一次都不调用 —— 关闭日志即回到原行为。
    本函数不消耗任何随机数、不改变计算顺序，因此不影响结果；
    回调自身抛错**不吞**，否则日志故障会伪装成分析故障。
    """
    if progress is None:
        return
    payload = {"event": event, "ts": time.time(), **fields}
    banned = {"auroc", "delta", "delta_h", "ci", "ci_lo", "ci_hi", "oof", "score",
              "scores", "y", "label", "labels", "pred", "prediction", "increment",
              "ladder", "width", "point"}
    leaked = sorted(k for k in payload if k.lower() in banned)
    if leaked:                       # 结构性护栏：性能量不得经由进度日志泄漏
        raise PathViolation(f"进度事件不得携带性能相关字段：{leaked}")
    progress(payload)


def run_repeat_for_path(*, y, groups, blocks: Sequence[BlockSpec], repeat: int,
                        cfg: V3AnalysisConfig, path: str,
                        cache_builder=vs.build_base_block_cache,
                        progress=None, progress_context: Mapping | None = None) -> dict:
    """按路径跑一次 repeat。

    复用 `v3_stacking` 的**公开原语**（fold 划分 / cache / fit_stack），
    只把块名单换成该路径的。给定 canonical 块集时，本函数必须与
    `v3_stacking.run_repeat` **逐元素相同** —— 由 `tests/test_v3_analysis_paths.py`
    的等价性测试锁死，防止实现分叉。
    """
    blocks = blocks_for_path(blocks, path)
    names_s, names_sh = names_for_path(path)
    have = {b.name for b in blocks}
    missing = set(names_sh) - have
    if missing:
        raise PathViolation(f"路径 {path} 缺块：{sorted(missing)}")

    n_rows = {b.n_rows for b in blocks}
    if len(n_rows) != 1 or n_rows.pop() != y.size:
        raise PathViolation("blocks 行数与 y 不一致")

    folds = ci.stratified_grouped_folds(y, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(repeat),
                                        max_seed_retries=cfg.outer_max_seed_retries)
    n = y.size
    oof = {"S*": np.full(n, np.nan), "S*+H": np.full(n, np.nan)}
    for b in blocks:
        oof[b.name] = np.full(n, np.nan)
    audits = []
    ctx = dict(progress_context or {})
    for f, (train, test) in enumerate(folds):
        t0 = time.time()
        emit_progress(progress, "outer_fold_start", path=path, repeat=int(repeat),
                      outer_fold=int(f), n_outer_folds=len(folds),
                      n_train=int(np.size(train)), n_test=int(np.size(test)), **ctx)
        cache = cache_builder(y=y, groups=groups, blocks=blocks, repeat=repeat,
                              outer_fold=f, train=train, test=test, cfg=cfg)
        oof["S*"][test] = vs.fit_stack(cache, names_s, y, cfg)
        oof["S*+H"][test] = vs.fit_stack(cache, names_sh, y, cfg)
        for name, sc in cache.test_scores.items():
            oof[name][test] = sc
        audits.append(cache.audit())
        emit_progress(progress, "outer_fold_done", path=path, repeat=int(repeat),
                      outer_fold=int(f), n_outer_folds=len(folds),
                      elapsed_s=round(time.time() - t0, 3), status="ok", **ctx)
    return {"repeat": int(repeat), "oof": oof, "audits": audits,
            "path": path, "block_names_s_star": names_s,
            "block_names_s_star_plus_h": names_sh}


def run_all_repeats_for_path(*, y, groups, blocks: Sequence[BlockSpec],
                             cfg: V3AnalysisConfig, path: str,
                             cache_builder=vs.build_base_block_cache,
                             progress=None,
                             progress_context: Mapping | None = None) -> list[dict]:
    return [run_repeat_for_path(y=y, groups=groups, blocks=blocks, repeat=r,
                                cfg=cfg, path=path, cache_builder=cache_builder,
                                progress=progress, progress_context=progress_context)
            for r in range(cfg.n_repeats)]


def path_markers(path: str,
                 numeric_status: str | None = DEFAULT_NUMERIC_STATUS) -> dict:
    """报告必须携带的标记与解释边界。

    `b4_source` 写的是**实现**而非泛称 —— 结果文件只凭这里就能把 Path B 的
    tail 实现与已废止的 full-BF16 shadow 区分开。

    `numeric_status` 只影响 **Path A** 的 marker 与解释边界（B 是对称 overlay、
    C 已整块移除 B4，两者都不依赖冻结 B4 的数值完整性）。省略即沿用 Qwen 的
    既有裁定 `compromised`，文字与参数化之前逐字相同。
    """
    sem = PATH_SEMANTICS[path]
    st = assert_numeric_status(numeric_status)
    out = {"path": path, "role": sem["role"], "b4_source": sem["b4_source"],
           "marker": sem["mandatory_marker"],
           "interpretation_limit": sem["interpretation_limit"]}
    if path == PATH_A:
        out["marker"] = _PATH_A_MARKER_BY_STATUS[st]
        out["interpretation_limit"] = _PATH_A_LIMIT_BY_STATUS[st]
        out["numeric_integrity_status"] = st
    if "b4_source_caveat" in sem:
        out["b4_source_caveat"] = sem["b4_source_caveat"]
    return out
