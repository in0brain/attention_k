"""stable-logit shadow F5 的 schema 与硬验收单测。合成 fixture，**不加载模型**。

锁定研究者裁定的边界：
  * H1 重建 8 项 / 引用 6 项；MCQ 重建 4 项 / 引用 2 项
  * **不得**静默过滤任何 token：count == 真实 token 数
  * 0 <= ratio <= 1；全部值有限；identifier positions 合法
  * MCQ 四项复用 domain_label_proxy 原定义，只换输入数值
  * 命名口径：sensitivity / protocol-deviation，**不得**冒充 primary
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention.conditional_increment import F5_FEATURE_NAMES  # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow  # noqa: E402
from recover_attention.mcq_conditional_increment import MCQ_F5_FEATURE_NAMES  # noqa: E402


def h1_inputs(n_tok=10, id_pos=(2, 3, 7)):
    rng = np.random.default_rng(20260818)
    return {
        "n_tok": n_tok,
        "target_logprob": -np.abs(rng.normal(1.0, 0.5, n_tok)),
        "entropy": np.abs(rng.normal(2.0, 0.3, n_tok)),
        "rank": rng.integers(1, 500, n_tok),
        "id_positions": list(id_pos),
    }


def h1_row(n_id=3):
    return {"example_id": "ex_a", "sample_type": "greedy", "sample_index": 0,
            "f5_id_token_count": n_id, "f5_self_consistency_exact": 0.5,
            "f5_id_agreement_rate": 0.25, "f5_confidence_high": 0,
            "f5_confidence_medium": 1, "f5_confidence_low": 0}


# ------------------------------------------------------------ 字段划分
def test_h1_field_partition_is_eight_plus_six():
    assert len(shadow.H1_STABLE_FIELDS) == 8
    assert len(shadow.H1_FROZEN_PASSTHROUGH) == 6
    union = set(shadow.H1_STABLE_FIELDS) | set(shadow.H1_FROZEN_PASSTHROUGH)
    assert union == set(F5_FEATURE_NAMES)                 # 恰好覆盖 B4 的 14 项
    assert not (set(shadow.H1_STABLE_FIELDS) & set(shadow.H1_FROZEN_PASSTHROUGH))


def test_mcq_field_partition_is_four_plus_two():
    assert len(shadow.MCQ_STABLE_FIELDS) == 4
    assert len(shadow.MCQ_FROZEN_PASSTHROUGH) == 2
    union = set(shadow.MCQ_STABLE_FIELDS) | set(shadow.MCQ_FROZEN_PASSTHROUGH)
    assert union == set(MCQ_F5_FEATURE_NAMES)
    assert not (set(shadow.MCQ_STABLE_FIELDS) & set(shadow.MCQ_FROZEN_PASSTHROUGH))


# ---------------------------------------------------------------- H1 重建
def test_h1_shadow_uses_every_token_no_silent_filtering():
    inp = h1_inputs(n_tok=10)
    out = shadow.h1_shadow_row(h1_row(), **inp)
    assert out["f5_completion_token_count"] == 10          # == 真实 token 数
    assert math.isclose(out["f5_lengthnorm_logprob"],
                        float(inp["target_logprob"].mean()), rel_tol=1e-12)
    assert math.isclose(out["f5_id_token_ratio"], 3 / 10, rel_tol=1e-12)
    assert 0.0 <= out["f5_id_token_ratio"] <= 1.0


def test_h1_identifier_aggregates_use_only_identifier_positions():
    inp = h1_inputs(n_tok=10, id_pos=(2, 3, 7))
    out = shadow.h1_shadow_row(h1_row(), **inp)
    lp, ent, rk = inp["target_logprob"], inp["entropy"], inp["rank"]
    assert math.isclose(out["f5_id_logprob_mean"], float(lp[[2, 3, 7]].mean()), rel_tol=1e-12)
    assert math.isclose(out["f5_id_logprob_min"], float(lp[[2, 3, 7]].min()), rel_tol=1e-12)
    assert math.isclose(out["f5_id_token_entropy_mean"], float(ent[[2, 3, 7]].mean()),
                        rel_tol=1e-12)
    assert out["f5_first_id_token_rank"] == int(rk[2])     # **第一个** identifier 位置


def test_h1_perplexity_matches_exp_neg_lengthnorm():
    out = shadow.h1_shadow_row(h1_row(), **h1_inputs())
    assert math.isclose(math.log(out["f5_completion_perplexity"]),
                        -out["f5_lengthnorm_logprob"], abs_tol=1e-9)


def test_h1_all_stable_fields_finite():
    out = shadow.h1_shadow_row(h1_row(), **h1_inputs())
    for name in shadow.H1_STABLE_FIELDS:
        assert out[name] is not None and math.isfinite(float(out[name]))


# ---------------------------------------------------------------- H1 硬验收
def test_h1_rejects_length_mismatch():
    inp = h1_inputs(n_tok=10)
    inp["n_tok"] = 9
    with pytest.raises(shadow.ShadowAcceptanceError, match="长度与 completion token 数不符"):
        shadow.h1_shadow_row(h1_row(), **inp)


def test_h1_rejects_out_of_range_identifier_positions():
    inp = h1_inputs(n_tok=10, id_pos=(2, 99))
    with pytest.raises(shadow.ShadowAcceptanceError, match="越界"):
        shadow.h1_shadow_row(h1_row(n_id=2), **inp)


def test_h1_rejects_empty_identifier_positions():
    inp = h1_inputs()
    inp["id_positions"] = []
    with pytest.raises(shadow.ShadowAcceptanceError, match="为空"):
        shadow.h1_shadow_row(h1_row(), **inp)


def test_h1_rejects_nonfinite_values():
    inp = h1_inputs()
    inp["target_logprob"] = inp["target_logprob"].copy()
    inp["target_logprob"][0] = float("nan")
    with pytest.raises(shadow.ShadowAcceptanceError, match="非有限"):
        shadow.h1_shadow_row(h1_row(), **inp)


def test_h1_rejects_identifier_count_mismatch_with_frozen():
    inp = h1_inputs(id_pos=(2, 3, 7))
    with pytest.raises(shadow.ShadowAcceptanceError, match="identifier 计数与冻结值不符"):
        shadow.h1_shadow_row(h1_row(n_id=5), **inp)


def test_h1_ratio_can_never_exceed_one_by_construction():
    inp = h1_inputs(n_tok=4, id_pos=(0, 1, 2, 3))
    out = shadow.h1_shadow_row(h1_row(n_id=4), **inp)
    assert out["f5_id_token_ratio"] == 1.0                 # 上界正好取到，不越界


def test_h1_acceptance_rejects_count_not_equal_to_true_tokens():
    bad = {n: 1.0 for n in shadow.H1_STABLE_FIELDS}
    bad["f5_completion_token_count"] = 7
    bad["f5_id_token_ratio"] = 0.5
    bad["f5_first_id_token_rank"] = 1
    with pytest.raises(shadow.ShadowAcceptanceError, match="真实 token 数"):
        shadow.assert_h1_acceptance(bad, h1_row(), n_tok=10, n_id=5)


# --------------------------------------------------------------- MCQ 重建
def test_mcq_shadow_reuses_domain_label_proxy_definitions():
    opts = {"A": 2.0, "B": 1.0, "C": -1.0, "D": 0.5}
    full = np.array([2.0, 1.0, -1.0, 0.5, 0.0], dtype=np.float64)
    out = shadow.mcq_shadow_row({"example_id": "m1"}, option_logits=opts,
                                full_logits_row=full, letter_logprob=-0.3)
    assert math.isclose(out["f5_label_margin"], dlp.label_margin(opts), rel_tol=1e-12)
    assert math.isclose(out["f5_label_entropy"], dlp.label_entropy(opts), rel_tol=1e-12)
    assert math.isclose(out["f5_full_entropy"], dlp.full_entropy(full), rel_tol=1e-12)
    assert out["f5_letter_token_logprob"] == -0.3


def test_mcq_rejects_nonfinite_and_impossible_values():
    full = np.array([1.0, 0.0], dtype=np.float64)
    with pytest.raises(shadow.ShadowAcceptanceError):
        shadow.mcq_shadow_row({}, option_logits={"A": float("nan"), "B": 1.0},
                              full_logits_row=full, letter_logprob=-0.1)
    with pytest.raises(shadow.ShadowAcceptanceError, match="logprob 不得为正"):
        shadow.mcq_shadow_row({}, option_logits={"A": 1.0, "B": 0.0},
                              full_logits_row=full, letter_logprob=0.5)


def test_mcq_rejects_fewer_than_two_candidates():
    with pytest.raises(shadow.ShadowAcceptanceError):
        shadow.mcq_shadow_row({}, option_logits={"A": 1.0},
                              full_logits_row=np.array([1.0]), letter_logprob=-0.1)


# ------------------------------------------------------------ 引用与命名
def test_frozen_passthrough_records_source_and_does_not_recompute():
    row = h1_row()
    out = shadow.frozen_passthrough(row, "h1")
    assert out["values"]["f5_id_token_count"] == row["f5_id_token_count"]
    assert set(out["field_names"]) == set(shadow.H1_FROZEN_PASSTHROUGH)
    assert "未重建、未修改" in out["source"]


def test_shadow_record_carries_naming_discipline():
    stable = shadow.h1_shadow_row(h1_row(), **h1_inputs())
    rec = shadow.shadow_record(h1_row(), "h1", stable,
                               provenance={"unit_id": "ex_a__greedy_0"})
    assert rec["interpretation"] == shadow.SENSITIVITY_LABEL
    assert rec["deviation_class"] == shadow.DEVIATION_LABEL
    assert "不得" in rec["not_primary"]
    assert rec["trace_id"] == "ex_a__greedy_0"
    assert rec["projection_schema"] == "stable_logit_projection_f32_v1"


def test_module_never_claims_primary_status():
    src = (_REPO_ROOT / "src" / "recover_attention" / "evaluation"
           / "v3_shadow_f5.py").read_text(encoding="utf-8")
    assert shadow.SENSITIVITY_LABEL in src and shadow.DEVIATION_LABEL in src
    for banned in ('"primary"', '"confirmatory"', "corrected_b4", "replaces_frozen"):
        assert banned not in src
