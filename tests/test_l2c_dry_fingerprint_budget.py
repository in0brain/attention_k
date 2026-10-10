r"""G3 模拟后端的解码预算必须来自**本次运行实际生效的值**，不得写死。

**假后端；不加载模型、不跑 GPU、不碰真实 outputs/。**

修复前：`_FakeBackend.run_fingerprint` 把 `max_new_tokens` 写死成 384，不分臂。
于是 MCQ 臂的模拟指纹恒与冻结值 16 不符，dry 汇总每次都报
`generation_params.max_new_tokens[mcq]` —— 那是**假后端自己造出来的**不一致，
与真实生成路径无关（真实路径用同一个参数写指纹并调用生成）。

这里钉住：两臂各取自己的冻结默认值、CLI 覆盖能透传、按臂构造的桩不给别的臂出指纹，
以及修好之后模拟汇总**仍然**不算真实 gate 通过。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location(
        "_l2c_g3_entry", _REPO_ROOT / "scripts" / "run_stage1_g3_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg():
    return V3AnalysisConfig()


def _budget(entry, cfg, arm, **kw):
    be = entry._FakeBackend(arm, cfg, **kw)
    return be.run_fingerprint(arm)["generation_params"]["max_new_tokens"]


def test_each_arm_gets_its_own_frozen_budget(entry, cfg):
    assert _budget(entry, cfg, g3.ARM_H1) == cfg.h1_max_new_tokens == 384
    assert _budget(entry, cfg, g3.ARM_MCQ) == cfg.mcq_max_new_tokens == 16


def test_mcq_is_no_longer_hardcoded_to_the_h1_budget(entry, cfg):
    """这条就是修复前失败的那一条。"""
    assert _budget(entry, cfg, g3.ARM_MCQ) != _budget(entry, cfg, g3.ARM_H1)
    assert _budget(entry, cfg, g3.ARM_MCQ) != 384


def test_cli_override_reaches_the_dry_fingerprint(entry, cfg):
    """CLI 覆盖走的是同一个入参 —— 假后端不再自带写死的数字。"""
    assert _budget(entry, cfg, g3.ARM_H1, max_new_tokens=256) == 256
    assert _budget(entry, cfg, g3.ARM_MCQ, max_new_tokens=8) == 8


def test_a_per_arm_stub_refuses_to_fingerprint_another_arm(entry, cfg):
    be = entry._FakeBackend(g3.ARM_MCQ, cfg)
    with pytest.raises(ValueError, match="按臂构造"):
        be.run_fingerprint(g3.ARM_H1)


def test_backend_invariant_is_clean_for_a_simulated_pair(entry, cfg):
    """两臂模拟指纹应当通过 backend invariant —— 不再有假的长度不一致。"""
    runs = {arm: type("R", (), {
        "fingerprint": entry._FakeBackend(arm, cfg).run_fingerprint(arm)})()
        for arm in g3.ARMS}
    out = g3.backend_invariant(runs, cfg)
    assert out["ok"], out["mismatched_keys"]
    by_arm = out["arm_specific_generation_params"]["by_arm"]
    assert by_arm[g3.ARM_H1] == {"expected": 384, "actual": 384, "ok": True}
    assert by_arm[g3.ARM_MCQ] == {"expected": 16, "actual": 16, "ok": True}


def test_a_wrong_budget_is_still_caught(entry, cfg):
    """故障注入：修复没有把这项检查变松 —— 错的预算照样被抓。"""
    runs = {arm: type("R", (), {
        "fingerprint": entry._FakeBackend(arm, cfg).run_fingerprint(arm)})()
        for arm in g3.ARMS}
    runs[g3.ARM_MCQ].fingerprint["generation_params"]["max_new_tokens"] = 384
    out = g3.backend_invariant(runs, cfg)
    assert not out["ok"]
    assert "generation_params.max_new_tokens[mcq]" in out["mismatched_keys"]


def test_simulated_artifacts_still_cannot_be_real_gate_evidence(entry):
    """修好长度**不等于**模拟产物可以当真实 gate 证据。

    汇总里 `real_g3_passed` 与 `production_fingerprint_is_now_frozen` 由
    `dry_run_per_arm` 决定，与 `max_new_tokens` 无关 —— 这条拒绝没有被本次修复削弱。
    """
    src = (_REPO_ROOT / "scripts" / "run_stage1_g3_smoke.py").read_text(encoding="utf-8")
    assert 'refusals.append("dry_run_present:' in src
    assert "real_g3_passed = bool(gate_passed and not any_dry)" in src
    assert '"production_fingerprint_is_now_frozen": real_g3_passed,' in src
