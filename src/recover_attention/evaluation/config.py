"""Stage 0 分析配置。冻结常量来自 preregistration v2.3，不可由 config 覆盖。

区分两类参数:
  **冻结常量**（ε / δ / C 网格 / rel_depth / RQ2 最小样本）—— 来自预注册,config 改不动。
      若 config 试图覆盖 → 报错。跑后改判读规则正是 §2 要防的事。
  **运行参数**（n_splits / n_boot / seed / 路径）—— 有冻结下限,config 只能在合法范围内选。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from recover_attention import conditional_increment as ci

# §4/§7 的正式协议下限。smoke 用 3 折/inner 2/少量 bootstrap 是**接线**规格,不是论文规格。
PRODUCTION_MIN_OUTER_FOLDS = 5
PRODUCTION_MIN_BOOT = 1000
PRODUCTION_MIN_INNER_FOLDS = 3

# 这些来自预注册,config **不得**覆盖
FROZEN_KEYS = ("eps_auroc", "delta_rank_biserial", "c_grid", "rel_depth",
               "rq2_min_per_cell", "rq2_lowpower_total_pos", "precision_max_ci_width")


@dataclass(frozen=True)
class Stage0Config:
    """Stage 0 分析的运行配置。frozen 字段镜像预注册,只读。"""

    # ---- 运行参数（可配置，但有冻结下限）----
    n_outer_folds: int = PRODUCTION_MIN_OUTER_FOLDS
    n_inner_folds: int = PRODUCTION_MIN_INNER_FOLDS
    n_boot: int = PRODUCTION_MIN_BOOT
    seed: int = 0
    mode: str = "production"            # production | dry_run
    allow_pilot_population: bool = False  # 只有 dry_run 可以为 True

    # ---- 冻结常量（镜像 preregistration v2.3，不可覆盖）----
    eps_auroc: float = ci.EPS_AUROC
    delta_rank_biserial: float = ci.DELTA_RANK_BISERIAL
    c_grid: tuple[float, ...] = ci.C_GRID
    rel_depth: float = ci.DEFAULT_REL_DEPTH
    rq2_min_per_cell: int = ci.RQ2_MIN_PER_CELL
    rq2_lowpower_total_pos: int = ci.RQ2_LOWPOWER_TOTAL_POS
    # §7.2 的硬承诺:实际 Δ_H CI 宽度 > 2ε 装不进 [−ε,+ε] → 该臂如实记 inconclusive
    precision_max_ci_width: float = 2 * ci.EPS_AUROC

    def __post_init__(self) -> None:
        if self.mode not in ("production", "dry_run"):
            raise ValueError(f"mode must be production|dry_run, got {self.mode!r}")
        if self.mode == "production":
            if self.n_outer_folds < PRODUCTION_MIN_OUTER_FOLDS:
                raise ValueError(f"production needs >={PRODUCTION_MIN_OUTER_FOLDS} outer folds "
                                 f"(§4/§7), got {self.n_outer_folds}")
            if self.n_boot < PRODUCTION_MIN_BOOT:
                raise ValueError(f"production needs >={PRODUCTION_MIN_BOOT} bootstrap rounds "
                                 f"(§4), got {self.n_boot}")
            if self.n_inner_folds < PRODUCTION_MIN_INNER_FOLDS:
                raise ValueError(f"production needs >={PRODUCTION_MIN_INNER_FOLDS} inner folds, "
                                 f"got {self.n_inner_folds}")
            if self.allow_pilot_population:
                raise ValueError("production must not admit pilot_smoke records; "
                                 "allow_pilot_population is dry-run only")
        # 冻结常量必须与预注册一致 —— 防止有人从 config 悄悄改判读规则
        if self.eps_auroc != ci.EPS_AUROC:
            raise ValueError(f"eps is frozen at {ci.EPS_AUROC} (§4); config cannot change it")
        if self.delta_rank_biserial != ci.DELTA_RANK_BISERIAL:
            raise ValueError(f"delta is frozen at {ci.DELTA_RANK_BISERIAL} (§6)")
        if tuple(self.c_grid) != tuple(ci.C_GRID):
            raise ValueError(f"C grid is frozen at {ci.C_GRID} (§7)")

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["c_grid"] = list(self.c_grid)
        d["frozen_keys"] = list(FROZEN_KEYS)
        d["frozen_source"] = "docs/paper/preregistration.md v2.3"
        return d


def load_config(path: str | Path | None = None, **overrides: Any) -> Stage0Config:
    """从 yaml 读运行参数。yaml **不得**含冻结常量 —— 出现即报错。"""
    data: dict[str, Any] = {}
    if path is not None:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = dict((raw.get("stage0_analysis") or {}))
    data.update(overrides)
    bad = sorted(set(data) & set(FROZEN_KEYS))
    if bad:
        raise ValueError(
            f"config must not set frozen preregistration constants: {bad}. "
            f"These live in preregistration v2.3; changing them post-freeze is exactly what §2 "
            f"forbids. Bump the version + rehash if a change is genuinely intended.")
    known = {f for f in Stage0Config.__dataclass_fields__ if f not in FROZEN_KEYS}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}; allowed: {sorted(known)}")
    return Stage0Config(**data)
