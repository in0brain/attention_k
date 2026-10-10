"""Stage 1 v3：`records → {B1..B5, H}` 的 feature adapter（M1a）。

**职责边界（关键）**：本模块只做 **fold 无关**的原始表示 —— 取字段、拼字符串、
按 `trace_id` 对齐 hidden。所有 **fold 相关**的变换（TF-IDF 的 vocabulary/IDF 拟合、
dense 的 z-score/√d）都留在 `v3_stacking._design` 里，由 `build_fold_design` 在
train fold 内完成。**防泄漏的责任只在一处**，adapter 不碰。

块映射（源字段全部来自 Stage 0 已冻结的 record schema）：

| block | kind  | 来源                                   | Stage 0 对应 |
|-------|-------|----------------------------------------|--------------|
| B1    | text  | `id_string_text`                       | rung2        |
| B2    | dense | `SURFACE_FEATURE_NAMES`（9）           | rung3        |
| B3    | text  | `completion`                           | rung4        |
| B4    | dense | `F5_FEATURE_NAMES`（14）               | rung5        |
| B5    | text  | `prompt_text` + 固定 family/route 标记 | rung1 的推广 |
| H     | dense | `h1_hidden.npz`，按 `trace_id` 取      | H            |

**B5 为什么是 text 且必须含 prompt**（v3 §11.7）：H 在 identifier token 上 pooling，
本身编码 prompt 上下文；若 S\\* 把 prompt 排除在外，`Δ_H` 会混入**输入信息增量**。
Stage 0 实测 `rung1_prompt_only = 0.6829`（A 分组），prompt 单独就有可观可分性。
B3 是**模型输出**，B5 是**模型输入**，两者语义不同但同为 text block。**不新增 B6。**

**不静默填 0**：缺列、`None`、非有限值一律抛错 —— Stage 0 曾因 `_dense_matrix`
静默填 0 而让两列跨样本 F5 全缺仍通过形状检查（见 PROGRESS「H1 全量验收记录」）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.v3_blocks import BlockSpec, validate_blocks

# 预注册的固定枚举 —— **不从数据推导**，避免类别集合随样本漂移。
FAMILY_ENUM: tuple[str, ...] = ("attack", "cwe", "cve")
ROUTE_ENUM: tuple[str, ...] = ("recall", "open_gen")

# B5 的标记模板：固定前缀 + 枚举值，使 TF-IDF 能把它当作稳定 token 学到。
FAMILY_TOKEN = "b5family_{}"
ROUTE_TOKEN = "b5route_{}"


def _require(records: Sequence[dict], field: str) -> None:
    missing = [i for i, r in enumerate(records) if field not in r or r[field] is None]
    if missing:
        raise ValueError(
            f"records missing field {field!r} at rows {missing[:10]}"
            f"{'…' if len(missing) > 10 else ''} (n_missing={len(missing)}). "
            f"不静默填 0 —— Stage 0 曾因静默填 0 让两列 F5 全缺仍通过形状检查。")


def _text_field(records: Sequence[dict], field: str) -> list[str]:
    _require(records, field)
    out = []
    for i, r in enumerate(records):
        v = r[field]
        if not isinstance(v, str):
            raise ValueError(f"row {i}: {field!r} must be str, got {type(v).__name__}")
        out.append(v)
    return out


def _dense_from_names(records: Sequence[dict], names: Sequence[str],
                      block: str) -> np.ndarray:
    for n in names:
        _require(records, n)
    mat = np.asarray([[float(r[n]) for n in names] for r in records], dtype=float)
    if not np.all(np.isfinite(mat)):
        bad = np.argwhere(~np.isfinite(mat))
        raise ValueError(
            f"{block}: {len(bad)} non-finite values, first at "
            f"row={int(bad[0][0])} column={names[int(bad[0][1])]!r}")
    return mat


def build_b5_text(records: Sequence[dict]) -> list[str]:
    """B5 = prompt_text + 固定 family/route 标记（v3 §11.7）。

    标记来自**预注册的固定枚举**；出现枚举外的取值 → 抛错，不静默归入 other，
    否则类别空间会随样本漂移而改变 estimand。
    """
    prompts = _text_field(records, "prompt_text")
    _require(records, "family")
    _require(records, "route")
    out = []
    for i, (r, p) in enumerate(zip(records, prompts)):
        fam, rt = r["family"], r["route"]
        if fam not in FAMILY_ENUM:
            raise ValueError(f"row {i}: family={fam!r} not in frozen enum {FAMILY_ENUM}")
        if rt not in ROUTE_ENUM:
            raise ValueError(f"row {i}: route={rt!r} not in frozen enum {ROUTE_ENUM}")
        out.append(f"{FAMILY_TOKEN.format(fam)} {ROUTE_TOKEN.format(rt)} {p}")
    return out


def align_hidden(records: Sequence[dict], hidden_by_id: dict, *,
                 id_field: str = "trace_id") -> np.ndarray:
    """按 `trace_id` 对齐 hidden。缺失或维度不一致 → 抛错。"""
    _require(records, id_field)
    missing = [r[id_field] for r in records if r[id_field] not in hidden_by_id]
    if missing:
        raise ValueError(f"hidden missing for {len(missing)} records, "
                         f"first: {missing[:5]}")
    rows = [np.asarray(hidden_by_id[r[id_field]], dtype=float).ravel() for r in records]
    dims = {v.size for v in rows}
    if len(dims) != 1:
        raise ValueError(f"hidden rows disagree on dimension: {sorted(dims)}")
    mat = np.vstack(rows)
    if not np.all(np.isfinite(mat)):
        n_bad = int(np.count_nonzero(~np.isfinite(mat)))
        raise ValueError(f"H: {n_bad} non-finite values in aligned hidden matrix")
    return mat


def build_blocks(records: Sequence[dict], hidden_by_id: dict, *,
                 id_field: str = "trace_id") -> list[BlockSpec]:
    """构建 `{B1..B5, H}` 六块，返回 canonical 顺序（经 `validate_blocks` 校验）。

    只做 fold 无关的表示；fold 内的 TF-IDF / 标准化由 v3_stacking 负责。
    """
    if not records:
        raise ValueError("empty records")
    blocks = [
        BlockSpec("B1", "text", text=_text_field(records, "id_string_text")),
        BlockSpec("B2", "dense",
                  dense=_dense_from_names(records, ci.SURFACE_FEATURE_NAMES, "B2")),
        BlockSpec("B3", "text", text=_text_field(records, "completion")),
        BlockSpec("B4", "dense",
                  dense=_dense_from_names(records, ci.F5_FEATURE_NAMES, "B4")),
        BlockSpec("B5", "text", text=build_b5_text(records)),
        BlockSpec("H", "dense", dense=align_hidden(records, hidden_by_id,
                                                   id_field=id_field)),
    ]
    ordered, _n = validate_blocks(blocks)
    return ordered


def build_arm(records: Sequence[dict], hidden_by_id: dict, *,
              label_field: str = "label", group_field: str = "group_id",
              id_field: str = "trace_id") -> dict:
    """一站式：blocks + y + groups，并做长度一致性校验。"""
    _require(records, label_field)
    _require(records, group_field)
    y = np.asarray([int(r[label_field]) for r in records], dtype=int)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"{label_field} must be binary 0/1, got {sorted(set(y.tolist()))}")
    groups = np.asarray([r[group_field] for r in records])
    blocks = build_blocks(records, hidden_by_id, id_field=id_field)
    return {"blocks": blocks, "y": y, "groups": groups, "n": int(y.size),
            "block_source_map": {
                "B1": "id_string_text", "B2": list(ci.SURFACE_FEATURE_NAMES),
                "B3": "completion", "B4": list(ci.F5_FEATURE_NAMES),
                "B5": "prompt_text + frozen family/route markers",
                "H": f"hidden aligned by {id_field}"}}


REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "id_string_text", "completion", "prompt_text", "family", "route",
    "trace_id", "group_id", "label",
) + tuple(ci.SURFACE_FEATURE_NAMES) + tuple(ci.F5_FEATURE_NAMES)
