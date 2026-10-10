"""v3 feature adapter：records → {B1..B5, H}。

**只用合成 fixture。** 测试套件不依赖 `outputs/**` —— 真实 Stage 0 文件的
字段存在性只做过一次性只读 smoke（记录在
`docs/progress/stage1_v3_analysis_history.md`），不进永久测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import v3_adapter as ad  # noqa: E402
from recover_attention.evaluation.v3_config import BLOCK_NAMES_S_STAR_PLUS_H  # noqa: E402


# ───────────────────────────────────────────── 合成 fixture
def make_records(n=12, d_hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    recs, hidden = [], {}
    for i in range(n):
        tid = f"t{i:04d}"
        fam = ad.FAMILY_ENUM[i % len(ad.FAMILY_ENUM)]
        rt = ad.ROUTE_ENUM[i % len(ad.ROUTE_ENUM)]
        r = {
            "trace_id": tid, "group_id": f"g{i // 2:03d}", "label": int(i % 2),
            "id_string_text": f"T{1000 + i} CWE-{i}",
            "completion": f"completion text number {i} alpha beta",
            "prompt_text": f"prompt question {i} gamma delta",
            "family": fam, "route": rt,
        }
        for name in ci.SURFACE_FEATURE_NAMES:
            r[name] = float(rng.normal())
        for name in ci.F5_FEATURE_NAMES:
            r[name] = float(rng.normal())
        recs.append(r)
        hidden[tid] = rng.normal(size=d_hidden)
    return recs, hidden


# ───────────────────────────────────────────── 1. 六块齐全 + canonical 顺序
def test_build_blocks_returns_six_blocks_in_canonical_order():
    recs, hid = make_records()
    blocks = ad.build_blocks(recs, hid)
    assert [b.name for b in blocks] == list(BLOCK_NAMES_S_STAR_PLUS_H)
    assert all(b.n_rows == len(recs) for b in blocks)


def test_block_kinds_match_m1a():
    recs, hid = make_records()
    kinds = {b.name: b.kind for b in ad.build_blocks(recs, hid)}
    assert kinds == {"B1": "text", "B2": "dense", "B3": "text",
                     "B4": "dense", "B5": "text", "H": "dense"}


def test_build_arm_aligns_y_groups_and_blocks():
    recs, hid = make_records(n=10)
    arm = ad.build_arm(recs, hid)
    assert arm["n"] == 10 and arm["y"].size == 10 and arm["groups"].size == 10
    assert set(np.unique(arm["y"])) <= {0, 1}
    assert [b.name for b in arm["blocks"]] == list(BLOCK_NAMES_S_STAR_PLUS_H)
    assert arm["block_source_map"]["B5"].startswith("prompt_text +")


# ───────────────────────────────────────────── 2. 列名与顺序恒等于冻结常量
def test_b2_and_b4_column_order_is_exactly_the_frozen_names():
    """顺序漂移会静默改变 estimand —— 逐列对照原始字段值。"""
    recs, hid = make_records(n=6)
    by = {b.name: b for b in ad.build_blocks(recs, hid)}
    for name, names in (("B2", ci.SURFACE_FEATURE_NAMES), ("B4", ci.F5_FEATURE_NAMES)):
        mat = np.asarray(by[name].dense)
        assert mat.shape == (6, len(names))
        for j, col in enumerate(names):
            assert np.allclose(mat[:, j], [r[col] for r in recs]), \
                f"{name} 第 {j} 列不是 {col!r} —— 列顺序漂移"


# ───────────────────────────────────────────── 3. B5：text，含 prompt 与固定标记
def test_b5_is_text_and_contains_prompt_and_frozen_markers():
    """v3 §11.7：B5 必须含 prompt 文本，否则 Δ_H 会混入输入信息增量。"""
    recs, hid = make_records(n=6)
    b5 = {b.name: b for b in ad.build_blocks(recs, hid)}["B5"]
    assert b5.kind == "text"
    for r, t in zip(recs, b5.text):
        assert r["prompt_text"] in t, "B5 未包含 prompt 文本"
        assert ad.FAMILY_TOKEN.format(r["family"]) in t
        assert ad.ROUTE_TOKEN.format(r["route"]) in t


def test_b5_markers_come_from_frozen_enum_not_from_data():
    recs, hid = make_records(n=4)
    recs[0]["family"] = "unknown_family"
    with pytest.raises(ValueError, match="not in frozen enum"):
        ad.build_blocks(recs, hid)
    recs[0]["family"] = "attack"
    recs[1]["route"] = "weird_route"
    with pytest.raises(ValueError, match="not in frozen enum"):
        ad.build_blocks(recs, hid)


def test_b5_differs_from_b3_input_vs_output():
    """B3 = 模型输出，B5 = 模型输入；两者不得相同。"""
    recs, hid = make_records(n=5)
    by = {b.name: b for b in ad.build_blocks(recs, hid)}
    assert by["B3"].text != by["B5"].text


def test_no_b6_is_produced():
    recs, hid = make_records()
    assert len(ad.build_blocks(recs, hid)) == 6


# ───────────────────────────────────────────── 4. B1 沿用 Stage 0 语义
def test_b1_uses_id_string_text_field_verbatim():
    recs, hid = make_records(n=5)
    b1 = {b.name: b for b in ad.build_blocks(recs, hid)}["B1"]
    assert b1.text == [r["id_string_text"] for r in recs]


def test_id_string_text_semantics_skip_echoed():
    """B1 的来源字段由 Stage 0 的 ci.id_string_text 生成，echoed 不入。"""
    mentions = [{"family": "attack", "label": "grounded", "normalized": "T1114"},
                {"family": "cwe", "label": "echoed", "normalized": "CWE-79"},
                {"family": "cve", "label": "fabricated", "normalized": "CVE-2024-1"}]
    assert ci.id_string_text(mentions) == "T1114 CVE-2024-1"


# ───────────────────────────────────────────── 5. H 对齐
def test_hidden_aligned_by_trace_id():
    recs, hid = make_records(n=6, d_hidden=5)
    H = {b.name: b for b in ad.build_blocks(recs, hid)}["H"]
    mat = np.asarray(H.dense)
    assert mat.shape == (6, 5)
    for i, r in enumerate(recs):
        assert np.allclose(mat[i], hid[r["trace_id"]])


def test_hidden_missing_trace_raises():
    recs, hid = make_records(n=5)
    hid.pop(recs[2]["trace_id"])
    with pytest.raises(ValueError, match="hidden missing"):
        ad.build_blocks(recs, hid)


def test_hidden_dimension_mismatch_raises():
    recs, hid = make_records(n=4, d_hidden=6)
    hid[recs[1]["trace_id"]] = np.zeros(5)
    with pytest.raises(ValueError, match="disagree on dimension"):
        ad.build_blocks(recs, hid)


def test_hidden_non_finite_raises():
    recs, hid = make_records(n=4, d_hidden=3)
    hid[recs[0]["trace_id"]] = np.array([1.0, np.inf, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        ad.build_blocks(recs, hid)


# ───────────────────────────────────────────── 6. 不静默填 0
@pytest.mark.parametrize("field", ["id_string_text", "completion", "prompt_text",
                                   "family", "route", "trace_id"])
def test_missing_text_field_raises_instead_of_filling(field):
    recs, hid = make_records(n=4)
    del recs[1][field]
    with pytest.raises(ValueError, match="missing field"):
        ad.build_blocks(recs, hid)


def test_none_value_raises():
    recs, hid = make_records(n=4)
    recs[2]["completion"] = None
    with pytest.raises(ValueError, match="missing field"):
        ad.build_blocks(recs, hid)


def test_missing_dense_column_raises():
    recs, hid = make_records(n=4)
    del recs[0][ci.F5_FEATURE_NAMES[3]]
    with pytest.raises(ValueError, match="missing field"):
        ad.build_blocks(recs, hid)


def test_non_finite_dense_value_raises_with_column_name():
    recs, hid = make_records(n=4)
    col = ci.SURFACE_FEATURE_NAMES[2]
    recs[1][col] = float("nan")
    with pytest.raises(ValueError, match=f"column='{col}'"):
        ad.build_blocks(recs, hid)


def test_non_str_text_field_raises():
    recs, hid = make_records(n=3)
    recs[0]["completion"] = 42
    with pytest.raises(ValueError, match="must be str"):
        ad.build_blocks(recs, hid)


def test_non_binary_label_raises():
    recs, hid = make_records(n=4)
    recs[0]["label"] = 2
    with pytest.raises(ValueError, match="binary"):
        ad.build_arm(recs, hid)


def test_empty_records_raises():
    with pytest.raises(ValueError, match="empty records"):
        ad.build_blocks([], {})


# ───────────────────────────────────────────── 7. 与统计核心接口贯通
def test_adapter_output_feeds_run_repeat():
    """端到端接线（合成数据）：adapter 的产物可直接喂给 v3 统计核心。"""
    from recover_attention.evaluation import v3_stacking as vs
    from recover_attention.evaluation.v3_config import V3AnalysisConfig

    rng = np.random.default_rng(0)
    n_groups, per = 24, 4
    recs, hid = [], {}
    for g in range(n_groups):
        for k in range(per):
            i = g * per + k
            tid = f"t{i:05d}"
            lab = int(g % 2 == 0)
            recs.append({
                "trace_id": tid, "group_id": f"g{g:03d}", "label": lab,
                "id_string_text": f"T{1000 + lab} extra{i % 3}",
                "completion": ("fabricated bogus id" if lab else "grounded real id")
                              + f" w{i % 4}",
                "prompt_text": f"prompt {i % 5} topic",
                "family": ad.FAMILY_ENUM[g % 3], "route": ad.ROUTE_ENUM[g % 2],
                **{nm: float(rng.normal()) for nm in ci.SURFACE_FEATURE_NAMES},
                **{nm: float(lab * 0.8 + rng.normal()) for nm in ci.F5_FEATURE_NAMES},
            })
            hid[tid] = lab * 1.2 + rng.normal(size=6)

    arm = ad.build_arm(recs, hid)
    cfg = V3AnalysisConfig(n_outer_folds=3, n_inner_folds=3, mode="dry_run",
                           n_repeats=1, b_per_batch=10, b_batch=2)
    out = vs.run_repeat(y=arm["y"], groups=arm["groups"], blocks=arm["blocks"],
                        repeat=0, cfg=cfg)
    for key in ("S*", "S*+H", "B1", "B5", "H"):
        assert np.all(np.isfinite(out["oof"][key]))


def test_required_record_fields_listed():
    for f in ("id_string_text", "completion", "prompt_text", "family", "route"):
        assert f in ad.REQUIRED_RECORD_FIELDS
    assert set(ci.F5_FEATURE_NAMES) <= set(ad.REQUIRED_RECORD_FIELDS)
    assert set(ci.SURFACE_FEATURE_NAMES) <= set(ad.REQUIRED_RECORD_FIELDS)
