"""backend invariant：两臂必须同一推理后端，否则 G3 拿不到。

§6 的 gate 是 D = S_MCQ − S_H1。量化改变 logits(实测 4bit vs 8bit 的 f5_label_margin
最大差 10.6),两臂 backend 不同 → S 之差分不清是 observability 差异还是量化差异。
这条不写在 preregistration.md(不 bump version),但必须由**代码**强制 —— lock 是散文,
会悄悄漂移。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "sprint_4D_2_conditional_increment",
    ROOT / "scripts" / "sprint_4D_2_conditional_increment.py")
sci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sci)

EIGHT_BIT = {"load_in_8bit": True, "device_map": "auto",
             "attn_implementation": "eager", "local_files_only": True}
FOUR_BIT = {"load_in_4bit": True, "load_in_8bit": False, "device_map": "auto",
            "attn_implementation": "eager", "local_files_only": True}


def _write_arms(tmp_path, h1_backend, mcq_backend):
    (tmp_path / "smoke_report_h1.json").write_text(
        json.dumps({"backend": h1_backend}), encoding="utf-8")
    (tmp_path / "smoke_report_mcq.json").write_text(
        json.dumps({"fingerprint": {"backend": mcq_backend}, "backend": mcq_backend}),
        encoding="utf-8")


def test_same_8bit_backend_passes(tmp_path):
    _write_arms(tmp_path, EIGHT_BIT, dict(EIGHT_BIT))
    r = sci._check_backend_invariant(tmp_path)
    assert r["ok"] is True and r["same_backend_both_arms"] is True


def test_mixed_quantization_blocks_g3(tmp_path):
    """4-bit MCQ + 8-bit H1 = 正是 4C 的情形。必须拦住。"""
    _write_arms(tmp_path, EIGHT_BIT, FOUR_BIT)
    r = sci._check_backend_invariant(tmp_path)
    assert r["ok"] is False
    assert r["same_backend_both_arms"] is False


def test_same_but_not_8bit_still_blocks(tmp_path):
    """两臂一致但不是 8-bit 也不行：8-bit 是被 H1 长文本退化逼定的(4D-1)。"""
    both_four = dict(FOUR_BIT)
    _write_arms(tmp_path, both_four, dict(both_four))
    r = sci._check_backend_invariant(tmp_path)
    assert r["ok"] is False


def test_missing_backend_fingerprint_blocks(tmp_path):
    (tmp_path / "smoke_report_h1.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "smoke_report_mcq.json").write_text(json.dumps({}), encoding="utf-8")
    r = sci._check_backend_invariant(tmp_path)
    assert r["ok"] is False and "missing" in r["reason"]


def test_g3_requires_backend_invariant(tmp_path):
    """即便两臂 smoke 都 passed，backend 不一致仍不得授予 G3。"""
    sha = "deadbeef"
    common = {"passed": True, "prereg_sha256": sha, "n_prompts": 20,
              "hidden_verification": {"numeric_match": True}}
    (tmp_path / "smoke_report_h1.json").write_text(json.dumps({
        **common, "kind": "model_smoke_h1",
        "schema_version": sci.SMOKE_SCHEMA_VERSION, "backend": EIGHT_BIT}), encoding="utf-8")
    (tmp_path / "smoke_report_mcq.json").write_text(json.dumps({
        **common, "kind": "model_smoke_mcq",
        "schema_version": sci.MCQ_SMOKE_SCHEMA_VERSION, "backend": FOUR_BIT}), encoding="utf-8")
    g3 = sci._read_g3(tmp_path, sha)
    assert g3["h1_arm"]["ok"] is True and g3["mcq_arm"]["ok"] is True   # 两臂各自都过
    assert g3["backend_invariant"]["ok"] is False
    assert g3["ok"] is False, "backend 不一致时 G3 必须拒绝,哪怕两臂 smoke 都 passed"


def test_real_reports_share_one_backend():
    """对真实产物跑一遍：当前两臂必须同为 8-bit。"""
    out = ROOT / "outputs/logs/sprint_4D_2_conditional_increment"
    if not (out / "smoke_report_mcq.json").exists():
        pytest.skip("MCQ smoke report not produced yet")
    r = sci._check_backend_invariant(out)
    assert r["ok"] is True, f"两臂 backend 不一致: {r}"
    assert r["h1"]["load_in_8bit"] is True and r["mcq"]["load_in_8bit"] is True
