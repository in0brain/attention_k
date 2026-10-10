"""stable-shadow overlay 与 E3 正式 reader —— **只在分析内存中构造**。

命名口径（强制）：overlay 产出的是 `stable_logit_shadow_sensitivity` /
`protocol_deviation_numerical_sensitivity`。**不是真值，也不是修正后的 primary**。
原冻结 completion records **一律不修改**。

读取前的硬门（任一不过即拒绝，不降级）：

    shadow_completion_report.complete == true
    repool completion report.complete == true
    H1 trace_id 集合精确等于 analyzable 集合
    MCQ example_id 集合精确等于 analyzable 集合
    schema / fingerprint / source SHA 全部吻合
    stable_fields 全有限
    frozen_passthrough 字段名与来源精确匹配
    无重复、缺失、多余 key
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import v3_e3_repool as repool
from . import v3_production as prod
from . import v3_shadow_f5 as shadow_f5
from .v3_config import V3AnalysisConfig

OVERLAY_SCHEMA = "stage1_shadow_overlay_v1"

#: Path B 的**唯一**被接受的 overlay 来源。解盲前裁定：B 的实现从
#: full-BF16 shadow 改为 tail-stabilized 单次前向。
OVERLAY_SOURCE_TAIL = "tail_stabilized_single_forward"
#: 旧 full-BF16 shadow 的来源标签 —— **永不**出现在 `ACCEPTED_OVERLAY_SOURCES` 里。
OVERLAY_SOURCE_LEGACY_SHADOW = "full_bf16_shadow"
ACCEPTED_OVERLAY_SOURCES: tuple[str, ...] = (OVERLAY_SOURCE_TAIL,)

#: 旧 shadow 的处置标签。三个都必须出现在 supersede 标记文件里。
LEGACY_SHADOW_STATUS = (
    "superseded_before_unblinding",
    "engineering_historical_evidence_only",
    "not_joined_with_labels",
)
LEGACY_SHADOW_MARKER_NAME = "SUPERSEDED.json"
LEGACY_MARKER_SCHEMA = "stage1_legacy_shadow_superseded_v1"


class OverlayRefused(RuntimeError):
    """硬门未通过 —— 拒绝读取，不降级、不部分加载。"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(directory: Path) -> str:
    """与 tail fingerprint 的 `source_freeze` 同一配方。"""
    import os
    h = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        blob = (Path(directory) / name).read_bytes()
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(blob).digest())
    return h.hexdigest()


def build_legacy_shadow_marker(shadow_dir: Path, tail_dir: Path) -> dict:
    """旧 full-BF16 shadow 的 supersede 标记内容。**不删除任何旧产物。**"""
    shadow_dir, tail_dir = Path(shadow_dir), Path(tail_dir)
    return {
        "schema": LEGACY_MARKER_SCHEMA,
        "status": list(LEGACY_SHADOW_STATUS),
        "superseded_by": {
            "path": str(tail_dir.name),
            "overlay_source": OVERLAY_SOURCE_TAIL,
            "reason": ("Path B 的唯一实现改为 tail-stabilized 单次前向："
                       "block 19 之前逐位保持冻结口径（8,664 条锚点全量证明），"
                       "cast 之后走外部 float32 投影且**不过滤 token**"),
        },
        "selection_basis": ("label-blind 的数值完整性与锚点保持；"
                            "**不是**依据性能结果挑路径"),
        "must_not": ("**不得**与 label 连接、**不得**与 tail 同时进入分析 —— "
                     "同时分析两者会重新形成结果树"),
        "retained_files": sorted(p.name for p in shadow_dir.iterdir() if p.is_file()
                                 and p.name != LEGACY_SHADOW_MARKER_NAME),
    }


def assert_legacy_shadow_superseded(shadow_dir: Path) -> dict:
    """标记必须存在且三个状态标签齐全，否则 readiness v3 不放行。"""
    p = Path(shadow_dir) / LEGACY_SHADOW_MARKER_NAME
    if not p.exists():
        raise OverlayRefused(f"旧 shadow 未登记为 superseded：{p} 不存在")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != LEGACY_MARKER_SCHEMA:
        raise OverlayRefused(f"supersede 标记 schema 不符：{doc.get('schema')!r}")
    missing = [s for s in LEGACY_SHADOW_STATUS if s not in (doc.get("status") or [])]
    if missing:
        raise OverlayRefused(f"supersede 标记缺状态标签：{missing}")
    if (doc.get("superseded_by") or {}).get("overlay_source") != OVERLAY_SOURCE_TAIL:
        raise OverlayRefused("supersede 标记未指向 tail 实现")
    return doc


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_bytes().split(b"\n") if x.strip()]


def _require_report(path: Path, what: str) -> dict:
    if not path.exists():
        raise OverlayRefused(
            f"{what} completion report 缺失 —— 按契约下游**必须拒绝读取**：{path}")
    rep = json.loads(path.read_text(encoding="utf-8"))
    if rep.get("complete") is not True:
        raise OverlayRefused(f"{what} completion report complete != true：{path}")
    return rep


def _key_of(row, arm: str) -> str:
    """H1 用 `trace_id`（adapter 对齐用的就是它），缺失时回退到 `derive_unit_id`；
    实测真实记录上两者**逐条相同**（200/200）。MCQ 用 `example_id`。"""
    if arm == "mcq":
        return row["example_id"]
    tid = row.get("trace_id")
    return tid if tid else repool.derive_unit_id(row)


# ------------------------------------------------------------------ 硬门
def load_shadow_overlay(shadow_dir: Path, prod_dir: Path) -> dict:
    """**已被取代（superseded_before_unblinding）** —— 旧 full-BF16 shadow 的 loader。

    保留供工程史与交叉核对，产物一律不删除；但它返回的 overlay 带
    `overlay_source = full_bf16_shadow`，**不在** `ACCEPTED_OVERLAY_SOURCES` 里，
    因此 `make_analyze_callback` 会拒绝把它接到 label 上。Path B 的正式实现是
    `load_tail_overlay`。**不得同时分析两者**——那会重新形成结果树。
    """
    shadow_dir, prod_dir = Path(shadow_dir), Path(prod_dir)
    report = _require_report(shadow_dir / "shadow_completion_report.json", "shadow")
    fp_path = shadow_dir / "shadow_fingerprint.json"
    if not fp_path.exists():
        raise OverlayRefused(f"shadow fingerprint 缺失：{fp_path}")
    fingerprint = json.loads(fp_path.read_text(encoding="utf-8"))

    # -- source SHA 必须与当前 production 一致 -----------------------------
    for name, want in (fingerprint.get("source_shas") or {}).items():
        cur = prod_dir / name
        if not cur.exists():
            raise OverlayRefused(f"fingerprint 指向的源文件不存在：{cur}")
        if _sha(cur) != want:
            raise OverlayRefused(f"源文件 SHA 漂移：{name}")

    # -- 记录集合必须精确等于 analyzable -----------------------------------
    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for arm, fname, key in (("h1", "h1_shadow_f5.jsonl", "trace_id"),
                            ("mcq", "mcq_shadow_f5.jsonl", "example_id")):
        recs = [json.loads(x) for x in
                (prod_dir / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
                if x.strip()]
        analyzable = [r for r in recs if prod.requires_hidden(arm, r)]
        want_ids = {_key_of(r, arm) for r in analyzable}

        rows = _jsonl(shadow_dir / fname)
        ids = [r.get(key) for r in rows]
        if len(ids) != len(set(ids)):
            raise OverlayRefused(f"{arm} shadow 存在重复 {key}")
        got = set(ids)
        if got != want_ids:
            raise OverlayRefused(
                f"{arm} shadow {key} 集合不等于 analyzable："
                f"缺 {len(want_ids - got)} 条，多 {len(got - want_ids)} 条")

        stable_names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1"
                        else shadow_f5.MCQ_STABLE_FIELDS)
        frozen_names = (shadow_f5.H1_FROZEN_PASSTHROUGH if arm == "h1"
                        else shadow_f5.MCQ_FROZEN_PASSTHROUGH)
        by_id: dict[str, dict] = {}
        for r in rows:
            if r.get("schema") != shadow_f5.SHADOW_SCHEMA:
                raise OverlayRefused(f"{arm} shadow schema 不符：{r.get('schema')}")
            if r.get("interpretation") != shadow_f5.SENSITIVITY_LABEL:
                raise OverlayRefused(f"{arm} shadow 命名口径不符")
            sf = r.get("stable_fields") or {}
            if set(sf) != set(stable_names):
                raise OverlayRefused(
                    f"{arm} {r.get(key)}: stable_fields 键集合不符（缺失/多余）")
            for k, v in sf.items():
                if v is None:
                    if not (arm == "h1" and k == "f5_completion_perplexity"):
                        raise OverlayRefused(f"{arm} {r.get(key)}: {k} 为 None")
                    continue
                if not math.isfinite(float(v)):
                    raise OverlayRefused(f"{arm} {r.get(key)}: {k} 非有限")
            fpass = r.get("frozen_passthrough") or {}
            if list(fpass.get("field_names") or []) != list(frozen_names):
                raise OverlayRefused(f"{arm} {r.get(key)}: frozen_passthrough 字段名不符")
            if "未重建、未修改" not in str(fpass.get("source", "")):
                raise OverlayRefused(f"{arm} {r.get(key)}: frozen_passthrough 来源标注不符")
            by_id[r[key]] = r
        out[arm] = by_id
        counts[arm] = len(by_id)

    return {
        "schema": OVERLAY_SCHEMA,
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": shadow_f5.DEVIATION_LABEL,
        "not_truth": ("overlay **不是**真值，也**不是**修正后的 primary；"
                      "只作 stable-logit shadow sensitivity"),
        "by_arm": out,
        "counts": counts,
        "shadow_report": report,
        "shadow_fingerprint": fingerprint,
        "gate_passed": True,
        # **不在** ACCEPTED_OVERLAY_SOURCES 里 —— 即便有人调用了这个函数，
        # `make_analyze_callback` 也会拒绝把它接到 label 上。
        "overlay_source": OVERLAY_SOURCE_LEGACY_SHADOW,
        "status": list(LEGACY_SHADOW_STATUS),
    }


# ------------------------------------------------------- Path B：tail 实现
def _validate_f5_rows(rows: list[dict], *, arm: str, key: str, want_ids: set,
                      schema: str, require_variant: str | None) -> dict[str, dict]:
    """两种实现共用的逐行硬门。集合必须**精确**等于 analyzable。"""
    ids = [r.get(key) for r in rows]
    if len(ids) != len(set(ids)):
        raise OverlayRefused(f"{arm} overlay 存在重复 {key}")
    got = set(ids)
    if got != want_ids:
        raise OverlayRefused(
            f"{arm} overlay {key} 集合不等于 analyzable："
            f"缺 {len(want_ids - got)} 条，多 {len(got - want_ids)} 条")

    stable_names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1"
                    else shadow_f5.MCQ_STABLE_FIELDS)
    frozen_names = (shadow_f5.H1_FROZEN_PASSTHROUGH if arm == "h1"
                    else shadow_f5.MCQ_FROZEN_PASSTHROUGH)
    by_id: dict[str, dict] = {}
    for r in rows:
        if r.get("schema") != schema:
            raise OverlayRefused(f"{arm} overlay schema 不符：{r.get('schema')}")
        if r.get("interpretation") != shadow_f5.SENSITIVITY_LABEL:
            raise OverlayRefused(f"{arm} overlay 命名口径不符")
        if r.get("deviation_class") != shadow_f5.DEVIATION_LABEL:
            raise OverlayRefused(f"{arm} overlay deviation_class 不符")
        if require_variant is not None and r.get("variant") != require_variant:
            raise OverlayRefused(
                f"{arm} {r.get(key)}: variant={r.get('variant')!r} != {require_variant!r}")
        sf = r.get("stable_fields") or {}
        if set(sf) != set(stable_names):
            raise OverlayRefused(
                f"{arm} {r.get(key)}: stable_fields 键集合不符（缺失/多余）")
        for k, v in sf.items():
            if v is None:
                if not (arm == "h1" and k == "f5_completion_perplexity"):
                    raise OverlayRefused(f"{arm} {r.get(key)}: {k} 为 None")
                continue
            if not math.isfinite(float(v)):
                raise OverlayRefused(f"{arm} {r.get(key)}: {k} 非有限")
        fpass = r.get("frozen_passthrough") or {}
        if list(fpass.get("field_names") or []) != list(frozen_names):
            raise OverlayRefused(f"{arm} {r.get(key)}: frozen_passthrough 字段名不符")
        if "未重建、未修改" not in str(fpass.get("source", "")):
            raise OverlayRefused(f"{arm} {r.get(key)}: frozen_passthrough 来源标注不符")
        by_id[r[key]] = r
    return by_id


def verify_tail_source_freeze(fingerprint: Mapping[str, Any], *, prod_dir: Path,
                              repool_dir: Path) -> dict:
    """核对 tail fingerprint 的 12 项 `source_freeze` 与当前磁盘一致。

    键带 `production/` / `repool/` 前缀；`__tree_digest` 结尾的是目录摘要。
    """
    freeze = fingerprint.get("source_freeze") or {}
    if not freeze:
        raise OverlayRefused("tail fingerprint 缺 source_freeze")
    roots = {"production": Path(prod_dir), "repool": Path(repool_dir)}
    drift = []
    for key, want in sorted(freeze.items()):
        root_name, _, rel = key.partition("/")
        root = roots.get(root_name)
        if root is None:
            drift.append(f"{key}: 未知根目录")
            continue
        if rel.endswith("__tree_digest"):
            d = root / rel[: -len("__tree_digest")]
            got = tree_digest(d) if d.is_dir() else "<missing>"
        else:
            f = root / rel
            got = _sha(f) if f.exists() else "<missing>"
        if got != want:
            drift.append(f"{key}: 冻结={want[:16]} 现场={str(got)[:16]}")
    if drift:
        raise OverlayRefused("tail fingerprint 的 source_freeze 与现场不一致：\n  "
                             + "\n  ".join(drift))
    return {"n_checked": len(freeze), "drift": []}


def load_tail_overlay(tail_dir: Path, prod_dir: Path, repool_dir: Path) -> dict:
    """**Path B 的正式 loader** —— 严格读取 tail F5 / report / fingerprint。

    硬门：

        tail_completion_report.complete == true
        tail_fingerprint 的 12 项 source_freeze 与现场逐项一致
        两臂 F5 键集合**精确**等于 analyzable
        每行 schema / interpretation / deviation_class / variant 全部吻合
        stable_fields 键集合精确、全有限
        frozen_passthrough 字段名与来源精确匹配
        无重复、缺失、多余 key
    """
    tail_dir, prod_dir, repool_dir = Path(tail_dir), Path(prod_dir), Path(repool_dir)
    report = _require_report(tail_dir / "tail_completion_report.json", "tail_stabilized")
    fp_path = tail_dir / "tail_fingerprint.json"
    if not fp_path.exists():
        raise OverlayRefused(f"tail fingerprint 缺失：{fp_path}")
    fingerprint = json.loads(fp_path.read_text(encoding="utf-8"))
    if fingerprint.get("variant") != OVERLAY_SOURCE_TAIL:
        raise OverlayRefused(
            f"tail fingerprint variant={fingerprint.get('variant')!r} "
            f"!= {OVERLAY_SOURCE_TAIL!r}")
    freeze_check = verify_tail_source_freeze(fingerprint, prod_dir=prod_dir,
                                             repool_dir=repool_dir)

    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for arm, fname, key in (("h1", "h1_tail_f5.jsonl", "trace_id"),
                            ("mcq", "mcq_tail_f5.jsonl", "example_id")):
        recs = [json.loads(x) for x in
                (prod_dir / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
                if x.strip()]
        analyzable = [r for r in recs if prod.requires_hidden(arm, r)]
        want_ids = {_key_of(r, arm) for r in analyzable}
        by_id = _validate_f5_rows(_jsonl(tail_dir / fname), arm=arm, key=key,
                                  want_ids=want_ids, schema=shadow_f5.SHADOW_SCHEMA,
                                  require_variant=OVERLAY_SOURCE_TAIL)
        out[arm] = by_id
        counts[arm] = len(by_id)

    return {
        "schema": OVERLAY_SCHEMA,
        "overlay_source": OVERLAY_SOURCE_TAIL,
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": shadow_f5.DEVIATION_LABEL,
        "not_truth": ("overlay **不是**真值，也**不是**修正后的 primary；"
                      "只作 stable-logit shadow sensitivity"),
        "by_arm": out,
        "counts": counts,
        "tail_report": report,
        "tail_fingerprint": fingerprint,
        "source_freeze_check": freeze_check,
        "gate_passed": True,
    }


def apply_overlay(records: Sequence[Mapping[str, Any]], overlay: Mapping[str, Any],
                  arm: str) -> list[dict]:
    """在**内存**里用 shadow stable_fields 覆盖对应 logit-derived B4 字段。

    passthrough 字段继续取冻结值 —— 因此这里只改 stable 的那几项，其余原样透传。
    原 records **不被修改**（返回新 dict）。
    """
    by_id = overlay["by_arm"][arm]
    stable_names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1"
                    else shadow_f5.MCQ_STABLE_FIELDS)
    out = []
    for r in records:
        key = _key_of(r, arm)
        sh = by_id.get(key)
        if sh is None:
            raise OverlayRefused(f"{arm}: {key} 无对应 shadow 记录")
        new = dict(r)
        for name in stable_names:
            v = sh["stable_fields"][name]
            if v is None:                      # 只有 H1 的 perplexity 允许 None
                continue
            new[name] = v
        out.append(new)
    return out


# ------------------------------------------------------------- E3 reader
def load_e3_blocks(repool_dir: Path, records: Sequence[Mapping[str, Any]],
                   cfg: V3AnalysisConfig,
                   geometry: "repool.ModelGeometry | None" = None) -> dict:
    """正式 E3 reader：先校验 completion report，再加载 pooled 向量。

    E3 **不是** primary grouping —— 按预注册身份输出，只作 §2 E3 的分块。
    """
    repool_dir = Path(repool_dir)
    geo = geometry or repool.QWEN_GEOMETRY
    report = _require_report(repool_dir / "e3_repool_report.json", "e3_repool")
    manifest = {e["unit_id"]: e for e in
                _jsonl(repool_dir / "e3_repool_manifest.jsonl")
                if e.get("state") == "done"}
    d = repool_dir / "h1_e3_block_hidden"

    vectors: dict[tuple[str, int], np.ndarray] = {}
    pairs = 0
    for r in records:
        uid = _key_of(r, "h1")
        assignment = repool.block_assignment(r, cfg)
        ev = manifest.get(uid)
        if ev is None:
            raise OverlayRefused(f"E3 manifest 缺 {uid}")
        if sorted(assignment) != list(ev["block_ids"]):
            raise OverlayRefused(f"E3 {uid}: block_ids 与源记录现场重建不符")
        if repool.position_digest(assignment) != ev["position_digest"]:
            raise OverlayRefused(f"E3 {uid}: position digest 不符")
        path = d / f"{uid}.npz"
        with np.load(path, allow_pickle=False) as z:
            for b in assignment:
                v = np.asarray(z[f"block_{b}"])
                if v.dtype != np.float32 or v.shape != (geo.hidden_dim,) \
                        or not bool(np.isfinite(v).all()):
                    raise OverlayRefused(
                        f"E3 {uid}/block_{b}: 向量不满足 "
                        f"float32/({geo.hidden_dim},)/finite")
                vectors[(uid, int(b))] = v
                pairs += 1
    return {"vectors": vectors, "n_traces": len(records), "n_pairs": pairs,
            "report": report,
            "canonical_source": "repool",
            "geometry": geo.as_dict(),
            # 路径与计数都取自**本次运行**，不再写死 Qwen 的目录与 8,664/9,916 ——
            # 那两个数是 Qwen 的结果，对另一个模型没有先验值。
            "canonical_note": (
                f"E3 正式输入固定为本次运行的 canonical repool 目录（{repool_dir}）"
                f" —— E3 是独立 estimand，**不随 Path B 的实现改变**。"
                f"本次加载 {len(records)} traces / {pairs} vectors。"
                f"tail 的 E3 cache 只作全量交叉验证证据，**不作第二份分析输入**；"
                f"两者是否逐元素相同须由该次交叉校验实测给出，不由本说明断言"),
            "grouping_role": "E3 按预注册身份输出；**不作 primary grouping**"}
