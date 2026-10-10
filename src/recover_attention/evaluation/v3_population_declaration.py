"""M4 —— 两臂合并的分析样本口径声明（schema + writer）。

本轮**只接 schema 与 writer**：所有依赖 label 的字段（如 `n_positive`）留空并标
`pending_unblinding`，**不提前写真实 null 产物**；真实值在解盲后随正式结果写出。

两类减员**分开记账，不得合并**：
    runtime technical exclusions      ForwardNonFiniteError，整组不进 F5/finalization
    canonicalization/parse attrition  requires_hidden=false，保留 finalized 行但不进分析
"""

from __future__ import annotations

import json
from pathlib import Path

DECLARATION_SCHEMA = "stage1_population_declaration_v1"
PENDING = "pending_unblinding"
#: 依赖 label、必须解盲后才能填的字段。
LABEL_DEPENDENT_FIELDS = ("n_positive", "n_negative", "positive_rate")
#: 分析集规模的 **schema 真源字段名**（沿用既有命名，不改成 n_analyzable）。
ANALYZABLE_FIELD = "analyzable"
#: 声明必须**恰好**覆盖这两臂。
REQUIRED_ARMS = ("h1", "mcq")
RATE_TOL = 1e-9


def build_declaration(*, h1: dict, mcq: dict, include_label_dependent: bool = False) -> dict:
    """构造两臂合并声明。`include_label_dependent=False` 时 label 相关字段标 pending。

    `h1` / `mcq` 为各臂的 population 计数字典（不含任何 label 派生量）。
    """
    def arm_block(d: dict) -> dict:
        out = {k: v for k, v in d.items() if k not in LABEL_DEPENDENT_FIELDS}
        for f in LABEL_DEPENDENT_FIELDS:
            out[f] = d.get(f) if include_label_dependent else PENDING
        return out

    out = {
        "schema_version": DECLARATION_SCHEMA,
        "arms": {"h1": arm_block(h1), "mcq": arm_block(mcq)},
        "attrition_categories": {
            "runtime_technical_exclusions":
                "ForwardNonFiniteError；整组不进 F5 / finalization",
            "canonicalization_parse_attrition":
                "requires_hidden=false；保留 finalized 行但不进入依赖 hidden 的分析",
            "note": "两类**分开记账，不得合并**",
        },
        "label_dependent_fields": list(LABEL_DEPENDENT_FIELDS),
        "label_dependent_status": "filled" if include_label_dependent else PENDING,
        "performance_blind": not include_label_dependent,
    }
    validate_declaration(out)                  # builder 侧先校验
    return out


def validate_declaration(declaration: dict) -> None:
    """**唯一**的校验入口，由 builder 与 writer 双双调用 —— 构造后被篡改也拦得住。

    pending：三个 label 派生字段必须全为 `pending_unblinding`。
    filled ：必须满足恒等式
        n_positive + n_negative == analyzable
        positive_rate == n_positive / analyzable
    `analyzable == 0` 时**不做除法**：要求两个计数均为 0，`positive_rate` 记 None
    并给出 `undefined_rate_reason`。
    """
    if not isinstance(declaration, dict):
        raise ValueError("declaration 必须是 dict")
    if declaration.get("schema_version") != DECLARATION_SCHEMA:
        raise ValueError(f"schema_version={declaration.get('schema_version')!r} "
                         f"!= {DECLARATION_SCHEMA}")
    status = declaration.get("label_dependent_status")
    if status not in (PENDING, "filled"):
        raise ValueError(f"label_dependent_status 非法：{status!r}")
    if declaration.get("label_dependent_fields") != list(LABEL_DEPENDENT_FIELDS):
        raise ValueError(f"label_dependent_fields 必须精确等于 {list(LABEL_DEPENDENT_FIELDS)}")
    want_blind = (status == PENDING)
    if declaration.get("performance_blind") is not want_blind:
        raise ValueError(f"performance_blind={declaration.get('performance_blind')!r} "
                         f"与 label_dependent_status={status!r} 不一致")
    arms = declaration.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(REQUIRED_ARMS):
        raise ValueError(f"arms 键集合必须恰为 {sorted(REQUIRED_ARMS)}，"
                         f"得到 {sorted(arms) if isinstance(arms, dict) else arms!r}")
    for arm, blk in arms.items():
        if not isinstance(blk, dict):
            raise ValueError(f"{arm}: 臂内容必须是 dict")
        missing = [f for f in (ANALYZABLE_FIELD, *LABEL_DEPENDENT_FIELDS) if f not in blk]
        if missing:
            raise ValueError(f"{arm}: 缺字段 {missing}")
        if status == PENDING:
            bad = [f for f in LABEL_DEPENDENT_FIELDS if blk.get(f) != PENDING]
            if bad:
                raise ValueError(f"{arm}: 未解盲声明含真实 label 派生值 {bad}")
            continue
        n = blk.get(ANALYZABLE_FIELD)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"{arm}: {ANALYZABLE_FIELD}={n!r} 必须是非负整数")
        pos, neg, rate = (blk.get(f) for f in LABEL_DEPENDENT_FIELDS)
        for name, v in (("n_positive", pos), ("n_negative", neg)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{arm}: {name}={v!r} 必须是非负整数（filled 下不得为 None）")
        if pos + neg != n:
            raise ValueError(
                f"{arm}: n_positive + n_negative = {pos + neg} != {ANALYZABLE_FIELD} {n}")
        if n == 0:
            if rate is not None:
                raise ValueError(f"{arm}: analyzable == 0 时 positive_rate 必须为 None，"
                                 f"得到 {rate!r}")
            if blk.get("undefined_rate_reason") != "analyzable == 0":
                raise ValueError(f"{arm}: analyzable == 0 时须写 "
                                 f"undefined_rate_reason='analyzable == 0'")
            continue
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError(f"{arm}: positive_rate={rate!r} 必须是数值")
        if not (0.0 <= float(rate) <= 1.0):
            raise ValueError(f"{arm}: positive_rate={rate} 不在 [0,1]")
        if abs(float(rate) - pos / n) > RATE_TOL:
            raise ValueError(f"{arm}: positive_rate={rate} != n_positive/{ANALYZABLE_FIELD} "
                             f"= {pos / n}")


def write_declaration(path: Path, declaration: dict) -> dict:
    """原子写。`pending_unblinding` 状态下也可写，但那只是 schema 占位，不是真实产物。"""
    import os
    import tempfile
    validate_declaration(declaration)          # writer 侧再校验一次，防构造后篡改
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(declaration, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"path": str(path), "bytes": path.stat().st_size,
            "label_dependent_status": declaration["label_dependent_status"]}


def assert_no_label_leak(declaration: dict) -> None:
    """未解盲时，label 相关字段必须全为 `pending_unblinding`（`validate_declaration` 的子集）。"""
    if declaration.get("label_dependent_status") == PENDING:
        bad = [(arm, f) for arm, blk in declaration["arms"].items()
               for f in LABEL_DEPENDENT_FIELDS if blk.get(f) != PENDING]
        if bad:
            raise ValueError(f"未解盲声明含真实 label 派生值：{bad}")
