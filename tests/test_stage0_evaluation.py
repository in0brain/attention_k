"""W0.5-E：production evaluation pipeline 的护栏测试。

重点不是"能跑",而是**跑错时会不会拦住**:
  - config 不得从 yaml 悄悄改冻结常量
  - production 不得降到 smoke 规格（3 折 / 少量 bootstrap）
  - production 不得吃 pilot_smoke 记录
  - F5 静默填 0 必须被拒
  - 两臂 backend 不一致必须拒发 gate
  - §7.2 精度承诺:CI 宽度 > 2ε → 判读降为 inconclusive
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import conditional_increment as ci
from recover_attention.evaluation import bootstrap as bs
from recover_attention.evaluation import feature_cache as fc
from recover_attention.evaluation.config import (
    PRODUCTION_MIN_BOOT,
    PRODUCTION_MIN_OUTER_FOLDS,
    Stage0Config,
    load_config,
)


# ---- config：冻结常量不可从 yaml 改 ----
def test_config_rejects_frozen_constant_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stage0_analysis:\n  eps_auroc: 0.10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen preregistration constants"):
        load_config(p)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stage0_analysis:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_config_mirrors_preregistration_constants():
    c = Stage0Config()
    assert c.eps_auroc == ci.EPS_AUROC
    assert c.delta_rank_biserial == ci.DELTA_RANK_BISERIAL
    assert tuple(c.c_grid) == tuple(ci.C_GRID)
    assert c.precision_max_ci_width == 2 * ci.EPS_AUROC


def test_production_refuses_smoke_spec():
    with pytest.raises(ValueError, match="outer folds"):
        Stage0Config(n_outer_folds=3)
    with pytest.raises(ValueError, match="bootstrap"):
        Stage0Config(n_boot=500)
    with pytest.raises(ValueError, match="inner folds"):
        Stage0Config(n_inner_folds=2)


def test_production_refuses_pilot_population():
    with pytest.raises(ValueError, match="pilot_smoke"):
        Stage0Config(allow_pilot_population=True)
    # dry_run 才可以
    c = Stage0Config(mode="dry_run", n_outer_folds=3, n_boot=100, allow_pilot_population=True)
    assert c.allow_pilot_population is True


def test_real_stage0_yaml_is_production_spec():
    c = load_config(ROOT / "configs" / "stage0.yaml")
    assert c.mode == "production"
    assert c.n_outer_folds >= PRODUCTION_MIN_OUTER_FOLDS
    assert c.n_boot >= PRODUCTION_MIN_BOOT
    assert c.allow_pilot_population is False


# ---- §7.2 精度承诺 ----
def test_precision_verdict_flags_wide_ci():
    c = Stage0Config()
    ok = bs.precision_verdict(-0.01, 0.01, c)
    assert ok["precision"] == bs.PRECISION_ADEQUATE and ok["ci_width"] == pytest.approx(0.02)
    bad = bs.precision_verdict(-0.04, 0.04, c)
    assert bad["precision"] == bs.PRECISION_INSUFFICIENT
    assert "no changing eps" in bad["note"]


def test_wide_ci_downgrades_read_to_inconclusive():
    """CI 装不进 [−ε,+ε] 就不得宣称 equivalent —— 哪怕点估计接近 0。"""
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([1 if i % 4 == 0 else 0 for i in range(n)])
    s = rng.normal(0, 1, n) + y * 0.5
    c = Stage0Config(mode="dry_run", n_outer_folds=3, n_inner_folds=2, n_boot=300,
                     allow_pilot_population=True)
    res = bs.paired_delta(y, s, s + rng.normal(0, 0.6, n), np.arange(n), c)
    if res["precision"]["precision"] == bs.PRECISION_INSUFFICIENT:
        assert res["read"] == "inconclusive"
        assert res["read_before_precision_check"] is not None


# ---- backend invariant（分析入口的第二道关）----
def _arm(backend):
    return fc.ArmCache(arm="h1", records=[], hidden=np.zeros((1, 2)), y=np.array([0, 1]),
                       groups=np.array(["a", "b"]), backend=backend, validation={})


EIGHT = {"load_in_8bit": True, "device_map": "auto",
         "attn_implementation": "eager", "local_files_only": True}


def test_check_backend_invariant_blocks_mixed_quantization():
    four = {**EIGHT, "load_in_8bit": False, "load_in_4bit": True}
    r = fc.check_backend_invariant({"h1": _arm(EIGHT), "mcq": _arm(four)})
    assert r["ok"] is False and r["same_backend_all_arms"] is False


def test_check_backend_invariant_passes_when_identical():
    r = fc.check_backend_invariant({"h1": _arm(dict(EIGHT)), "mcq": _arm(dict(EIGHT))})
    assert r["ok"] is True


def test_gate_refuses_when_backend_invariant_violated():
    from recover_attention.evaluation import observability_gate as og
    with pytest.raises(ValueError, match="do not share one backend"):
        og.compute_gate(_arm(EIGHT), {}, _arm(EIGHT), {}, Stage0Config(),
                        {"ok": False, "reason": "mixed"})


# ---- F5 completeness ----
def test_f5_completeness_detects_silently_zero_columns():
    """真实发生过:MCQ 的 3 列 19/19 全 0,而 d_F=6、检查全绿。"""
    recs = [{"f5_a": 1.0 + i, "f5_b": None} for i in range(5)]
    r = fc._f5_completeness(recs, ("f5_a", "f5_b"), ())
    assert r["ok"] is False
    assert r["empty_continuous_columns"] == ["f5_b"]
    assert r["nonzero_per_column"]["f5_b"] == 0
    assert r["d_F"] == 2, "d_F 只反映列数,不保证列有值 —— 这正是坑所在"


def test_f5_completeness_detects_constant_column():
    recs = [{"f5_a": 1.0 + i, "f5_c": 7.0} for i in range(5)]
    r = fc._f5_completeness(recs, ("f5_a", "f5_c"), ())
    assert r["ok"] is False and r["constant_continuous_columns"] == ["f5_c"]


def test_f5_completeness_allows_sparse_indicators_to_be_all_zero():
    """H1 的 verbalized-confidence 是稀疏指示量,整列 0 属正常,不得误报。"""
    recs = [{"f5_a": 1.0 + i, "f5_confidence_high": 0.0} for i in range(5)]
    r = fc._f5_completeness(recs, ("f5_a", "f5_confidence_high"), ("f5_confidence_",))
    assert r["ok"] is True
