"""Stage 1 v3 的 block 输入契约与 base 层产物容器。

**范围边界（本 sprint 明确不做）**：本模块只定义 `BlockSpec` 的**输入接口**，
不构造 B1–B5。Stage 1 的 `records → {B1..B5}` adapter 留待后续独立 sprint。
因此本 sprint 的产物**尚不能直接运行 Stage 1 正式分析**——它是经合成数据
验证的统计核心。

`BaseBlockCache` 是「shared block 只构建一次」这一 P0 要求的载体：
同一个 `(repeat, outer_fold)` 只调用一次 `build_base_block_cache`，
返回的**同一个实例**同时交给 S*（五块）与 S*+H（六块）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from recover_attention.evaluation.v3_config import (
    BLOCK_NAMES_S_STAR,
    BLOCK_NAMES_S_STAR_PLUS_H,
)


@dataclass(frozen=True)
class BlockSpec:
    """一个 block 的输入契约。

    `dense` / `text` 是**未做任何 fold 相关变换**的原始表示；
    train-fold 内的标准化与 TF-IDF 拟合由 v3_stacking 负责（防泄漏）。
    """

    name: str
    kind: str                                   # "dense" | "text"
    dense: np.ndarray | None = None             # (n, d)
    text: list[str] | None = None               # len == n

    def __post_init__(self) -> None:
        if self.kind not in ("dense", "text"):
            raise ValueError(f"{self.name}: kind must be dense|text, got {self.kind!r}")
        if self.kind == "dense":
            if self.dense is None:
                raise ValueError(f"{self.name}: kind=dense requires dense array")
            if np.asarray(self.dense).ndim != 2:
                raise ValueError(f"{self.name}: dense must be 2-D, got "
                                 f"{np.asarray(self.dense).ndim}-D")
        else:
            if self.text is None:
                raise ValueError(f"{self.name}: kind=text requires text list")

    @property
    def n_rows(self) -> int:
        return int(len(self.text) if self.kind == "text" else np.asarray(self.dense).shape[0])


def validate_blocks(blocks: Sequence[BlockSpec]) -> tuple[list[BlockSpec], int]:
    """**结构性锁死 estimand**：精确要求 {B1..B5, H}，并返回 canonical 顺序。

    第 1 版从调用方列表推导 `names_s = [b for b in blocks if b.name != "H"]`，
    于是缺 B5、缺 H（此时 S\\* 与 S\\*+H 变成同一个量）、多一个 B6、重名、
    顺序错都会**静默改变 estimand**。本函数把这些全部变成硬错误，
    且 canonical 顺序由 `v3_config` 的冻结常量给出，**不从调用方列表推导**。
    """
    names = [b.name for b in blocks]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate block names {dup}; estimand 必须由唯一 block 构成")

    want = set(BLOCK_NAMES_S_STAR_PLUS_H)
    got = set(names)
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        raise ValueError(
            f"block set mismatch — missing {missing}, unexpected {extra}. "
            f"S* = {list(BLOCK_NAMES_S_STAR)}, S*+H = {list(BLOCK_NAMES_S_STAR_PLUS_H)}；"
            f"缺 H 会让 S* 与 S*+H 变成同一个量，缺/多任何一块都改变 estimand。")

    by_name = {b.name: b for b in blocks}
    n_rows = {b.name: b.n_rows for b in blocks}
    if len(set(n_rows.values())) != 1:
        raise ValueError(f"blocks disagree on row count: {n_rows}")
    n = next(iter(n_rows.values()))

    for b in blocks:
        if b.kind == "dense":
            arr = np.asarray(b.dense)
            if not np.all(np.isfinite(arr)):
                bad = int(np.count_nonzero(~np.isfinite(arr)))
                raise ValueError(f"{b.name}: dense contains {bad} non-finite values")
        else:
            if len(b.text) != n:
                raise ValueError(f"{b.name}: text length {len(b.text)} != n_rows {n}")

    # canonical 顺序来自冻结常量
    return [by_name[name] for name in BLOCK_NAMES_S_STAR_PLUS_H], int(n)


def canonical_names() -> tuple[list[str], list[str]]:
    """(S* 的五块, S*+H 的六块)，均取自冻结常量。"""
    return list(BLOCK_NAMES_S_STAR), list(BLOCK_NAMES_S_STAR_PLUS_H)


@dataclass
class BaseBlockCache:
    """一个 `(repeat, outer_fold)` 的全部 base 层产物。**S\\* 与 S\\*+H 共享同一实例。**

    字段对应 M8 的四项要求：
      meta_train_scores  各 block 的 inner-OOF 分数（decision_function），用于训 meta
      test_scores        各 block 在 outer-test 上的分数（decision_function）
      chosen_C_outer     outer-full fit 选中的 C（每 block 一个）
      chosen_C_inner     每个 inner fold 各自选中的 C（可与 outer 不同 —— 这正是
                         🆕 不把 chosen_C 压成 dict[name -> float] 的原因）
      c_fallback_events  inner CV 不可行时的回退审计，逐条含
                         repeat / outer_fold / block / phase / inner_fold
    """

    repeat: int
    outer_fold: int
    train: np.ndarray
    test: np.ndarray
    meta_train_scores: dict[str, np.ndarray] = field(default_factory=dict)
    test_scores: dict[str, np.ndarray] = field(default_factory=dict)
    chosen_C_outer: dict[str, float] = field(default_factory=dict)
    chosen_C_inner: dict[str, dict[int, float]] = field(default_factory=dict)
    c_fallback_events: list[dict] = field(default_factory=list)

    def record_fallback(self, *, block: str, phase: str, inner_fold: int | None) -> None:
        self.c_fallback_events.append({
            "repeat": int(self.repeat), "outer_fold": int(self.outer_fold),
            "block": block, "phase": phase,
            "inner_fold": None if inner_fold is None else int(inner_fold),
        })

    def audit(self) -> dict:
        """落盘用的审计摘要。"""
        return {
            "repeat": int(self.repeat), "outer_fold": int(self.outer_fold),
            "n_train": int(self.train.size), "n_test": int(self.test.size),
            "blocks": sorted(self.test_scores),
            "chosen_C_outer": dict(self.chosen_C_outer),
            "chosen_C_inner": {b: dict(v) for b, v in self.chosen_C_inner.items()},
            "n_c_fallback": len(self.c_fallback_events),
            "c_fallback_events": list(self.c_fallback_events),
        }

    def require_blocks(self, names) -> None:
        missing = [n for n in names if n not in self.test_scores or n not in self.meta_train_scores]
        if missing:
            raise KeyError(f"BaseBlockCache missing blocks {missing} "
                           f"(have {sorted(self.test_scores)})")
