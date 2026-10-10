"""MCQ 侧 adapter：finalized records → `{B1..B5, H} + y + groups`（研究者裁决后写死）。

冻结映射（**不得由实现者改动**）：

    B1  dense  letter_onehot(records)            parsed_label 的 A–D one-hot，**不用 gold**
    B2  dense  MCQ_SURFACE_FEATURE_NAMES（精确 6 项）
    B3  text   semantic_output_text              被选中选项文本
    B4  dense  MCQ_F5_FEATURE_NAMES（精确 6 项）
    B5  text   prompt_only_text                  题干 + 全部选项，**不含模型选择**
    H   dense  按 example_id 对齐 hidden
    y          wrong_label（1 = incorrect，0 = correct）
    groups     group_id（E1 = question-disjoint）

`raw_completion` / gold / correctness / 额外 response-format 字段**不得进入任何 block**；
**不新增 B6**；`parsed_label` 只作 B1 的身份特征，**不作 y**。

⚠️ **non-analyzable 行必须在进入本模块之前过滤掉**（顶层入口负责）。
parse-failure 行的 `letter_onehot` 虽然是全零，但那是纯函数行为，
**不是**"缺 hidden 时的替代编码"——把这类行送进分析是错误的，这里直接抛错。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recover_attention import mcq_conditional_increment as mci
from recover_attention.evaluation.v3_blocks import BlockSpec, validate_blocks
from recover_attention.evaluation.v3_production import requires_hidden

ARM = "mcq"
#: 绝不允许出现在任何 block 里的字段（gold / 正确性 / 原始输出 / response-format）。
FORBIDDEN_IN_BLOCKS = ("raw_completion", "gold_label", "gold", "is_correct", "correct",
                       "correctness", "response_format", "bare_answer")
TEXT_FIELD_B3 = "semantic_output_text"
TEXT_FIELD_B5 = "prompt_only_text"
LABEL_FIELD = "wrong_label"
GROUP_FIELD = "group_id"
ID_FIELD = "example_id"


def filter_analyzable(records: Sequence[dict]) -> list[dict]:
    """按**既有** `requires_hidden` 规则筛出分析集。顶层入口在调 adapter 前必须先调它。"""
    return [r for r in records if requires_hidden(ARM, r)]


def _dense(records: Sequence[dict], names: Sequence[str], block: str) -> np.ndarray:
    out = np.empty((len(records), len(names)), dtype=float)
    for i, r in enumerate(records):
        for j, n in enumerate(names):
            v = r.get(n)
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{block}: {r.get(ID_FIELD)} 的 {n} 缺失或非数值：{v!r}")
            out[i, j] = float(v)
    if not np.all(np.isfinite(out)):
        bad = int(np.count_nonzero(~np.isfinite(out)))
        raise ValueError(f"{block}: dense 含 {bad} 个非有限值")
    return out


def _text(records: Sequence[dict], field: str) -> list[str]:
    out = []
    for r in records:
        v = r.get(field)
        if v is None:
            raise ValueError(f"{field} 缺失于 {r.get(ID_FIELD)}")
        out.append(str(v))
    return out


def assert_no_forbidden_source(block_source_map: dict) -> None:
    """block 来源映射里不得出现 gold / correctness / raw_completion / response-format。"""
    flat = " ".join(
        str(v) for v in block_source_map.values()
        if not isinstance(v, (list, tuple))) + " " + " ".join(
        str(x) for v in block_source_map.values() if isinstance(v, (list, tuple)) for x in v)
    hit = sorted({f for f in FORBIDDEN_IN_BLOCKS if f in flat})
    if hit:
        raise ValueError(f"block 来源含禁用字段 {hit}；gold / correctness / 原始输出 "
                         f"不得进入任何 block")


def align_hidden(records: Sequence[dict], hidden_by_id: dict) -> np.ndarray:
    missing = [r[ID_FIELD] for r in records if r[ID_FIELD] not in hidden_by_id]
    if missing:
        raise ValueError(f"{len(missing)} 条记录缺 hidden（例：{missing[:3]}）；"
                         f"analyzable 行必须都有 hidden")
    mat = np.asarray([np.asarray(hidden_by_id[r[ID_FIELD]], dtype=float) for r in records])
    if mat.ndim != 2:
        raise ValueError(f"hidden 对齐后维度异常：{mat.shape}")
    return mat


def build_mcq_blocks(records: Sequence[dict], hidden_by_id: dict) -> list[BlockSpec]:
    """构建 `{B1..B5, H}` 六块并返回 canonical 顺序（经 `validate_blocks` 校验）。"""
    if not records:
        raise ValueError("empty records")
    blocks = [
        BlockSpec("B1", "dense", dense=mci.letter_onehot(records)),
        BlockSpec("B2", "dense",
                  dense=_dense(records, mci.MCQ_SURFACE_FEATURE_NAMES, "B2")),
        BlockSpec("B3", "text", text=_text(records, TEXT_FIELD_B3)),
        BlockSpec("B4", "dense", dense=_dense(records, mci.MCQ_F5_FEATURE_NAMES, "B4")),
        BlockSpec("B5", "text", text=_text(records, TEXT_FIELD_B5)),
        BlockSpec("H", "dense", dense=align_hidden(records, hidden_by_id)),
    ]
    ordered, _n = validate_blocks(blocks)
    return ordered


def build_mcq_arm(records: Sequence[dict], hidden_by_id: dict) -> dict:
    """一站式：blocks + y + groups。**要求调用方已过滤 non-analyzable 行。**"""
    if not records:
        raise ValueError("empty records")
    bad = [r.get(ID_FIELD) for r in records if not requires_hidden(ARM, r)]
    if bad:
        raise ValueError(
            f"{len(bad)} 条 non-analyzable 记录被送入 adapter（例：{bad[:3]}）；"
            f"顶层入口必须先按 requires_hidden 过滤。parse-failure 行的全零 one-hot "
            f"**不是**缺 hidden 的替代编码")
    ys = []
    for r in records:
        v = r.get(LABEL_FIELD)
        if v is None or isinstance(v, bool) or v not in (0, 1):
            raise ValueError(f"{r.get(ID_FIELD)} 的 {LABEL_FIELD}={v!r} 不是 0/1；"
                             f"parse failure 不得被编码成 {LABEL_FIELD}=1")
        ys.append(int(v))
    y = np.asarray(ys, dtype=int)
    groups = np.asarray([r[GROUP_FIELD] for r in records])
    if any(not g for g in groups):
        raise ValueError(f"{GROUP_FIELD} 存在空值")
    blocks = build_mcq_blocks(records, hidden_by_id)
    source_map = {
        "B1": "parsed_label one-hot (letter_onehot)",
        "B2": list(mci.MCQ_SURFACE_FEATURE_NAMES),
        "B3": TEXT_FIELD_B3,
        "B4": list(mci.MCQ_F5_FEATURE_NAMES),
        "B5": TEXT_FIELD_B5,
        "H": f"hidden aligned by {ID_FIELD}",
    }
    assert_no_forbidden_source(source_map)
    return {"blocks": blocks, "y": y, "groups": groups, "n": int(y.size),
            "block_source_map": source_map, "arm": ARM,
            "label_field": LABEL_FIELD, "group_field": GROUP_FIELD, "id_field": ID_FIELD,
            "label_semantics": "1 = incorrect, 0 = correct（与 H1 的正类=错误同向）"}
