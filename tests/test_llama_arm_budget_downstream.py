r"""已登记的分臂解码预算在**下游校验**里的行为。

**不加载模型、不跑 GPU、不写真实 outputs/。**

钉住四类：
  1. 合法组合 H1=384 / MCQ=512（MCQ 臂已登记）→ 两臂汇总、G4 指纹、生产指纹均通过
  2. 旧 Qwen 的 384/16（无登记）→ **仍然通过**，行为与引入登记机制之前逐字相同
  3. 拒绝：未登记的扩展预算 / 登记漂移（沙改文件）/ 错误预算 / 登记指向别的臂
  4. 拒绝：新旧证据混用 —— G4 的指纹是 512 而它引用的 G3 生产指纹还是 16

以及 `load_budget_registration` 本身的"不能只凭一个 JSON 放行"：
登记必须与**一次已通过的生成长度检查**逐项对上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation import v3_g4_quality as g4  # noqa: E402
from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
LIST_REL = "outputs/stage1_llama_v3/mcq_budget512_check/mcq_budget_check_questions.jsonl"


@pytest.fixture
def cfg():
    return V3AnalysisConfig()


def _fp(arm, max_new, *, schema=True):
    gen = {"K": 6, "temperature": 0.7, "top_p": 0.95, "seed": 4242,
           "top_k": 20, "repetition_penalty": 1.05, "max_new_tokens": max_new}
    if schema:
        gen["schema"] = g3.GENERATION_PARAMS_SCHEMA
    return {"model": {"m": 1}, "tokenizer": {"t": 1}, "hidden_index": {"h": 1},
            "generation_params": gen, "run_hash": f"x_{arm}"}


def _runs(h1_budget, mcq_budget):
    return {g3.ARM_H1: g3.ArmRun(arm=g3.ARM_H1, fingerprint=_fp("h1", h1_budget)),
            g3.ARM_MCQ: g3.ArmRun(arm=g3.ARM_MCQ, fingerprint=_fp("mcq", mcq_budget))}


# ── 一份**自洽的**登记 + 配套的通过长度检查，落在 tmp 里 ──────────────────
def _make_registration(tmp_path: Path, *, budget=512, frozen=16, arm="mcq",
                       model_path=MODEL, list_sha=None, check_passed=True,
                       check_budget=None, root_offset=("outputs", "x")) -> Path:
    """在 tmp 下复刻真实的目录形状：<root>/outputs/stage1_llama_v3/。"""
    root = tmp_path
    reg_dir = root / "outputs" / "stage1_llama_v3"
    cal_dir = reg_dir / "mcq_budget512_check"
    cal_dir.mkdir(parents=True, exist_ok=True)

    list_path = cal_dir / "mcq_budget_check_questions.jsonl"
    list_path.write_text('{"question_hash":"abc"}\n', encoding="utf-8")
    real_sha = g3._sha256_file(list_path)
    declared_sha = list_sha or real_sha

    cal_rel = "outputs/stage1_llama_v3/mcq_budget512_check/mcq_budget_check_questions.jsonl"
    reg = {"schema": g3.BUDGET_REGISTRATION_SCHEMA, "arm": arm,
           "model_path": model_path,
           "frozen_max_new_tokens": frozen, "registered_max_new_tokens": budget,
           "status": "候选，未经证明", "reason": "测试用", "claim_limits": [],
           "calibration_set": {"path": cal_rel, "list_sha256": declared_sha}}
    reg_path = reg_dir / "budget512_registration.json"
    reg_path.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")

    report = {"passed": check_passed, "n_rows": 120,
              "checks": {"touched_max_new_tokens_rate": 0.0},
              "registration": {"sha256": g3._sha256_file(reg_path),
                               "registered_max_new_tokens": check_budget or budget},
              "calibration_set": {"sha256": declared_sha}}
    (cal_dir / "mcq_budget_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return reg_path


@pytest.fixture
def reg(tmp_path, cfg):
    """登记与其证据全在 tmp 下的真实目录形状里，不碰真实 outputs/。"""
    return g3.load_budget_registration(_make_registration(tmp_path),
                                       cfg=cfg, model_path=MODEL)


# ══════════════════════════════════════════════════════════════════════════
# 1. 合法组合 384 / 512
# ══════════════════════════════════════════════════════════════════════════
def test_expected_budget_per_arm(cfg, reg):
    assert g3.expected_max_new_tokens(g3.ARM_H1, cfg, reg) == 384
    assert g3.expected_max_new_tokens(g3.ARM_MCQ, cfg, reg) == 512


def test_backend_invariant_accepts_384_512(cfg, reg):
    inv = g3.backend_invariant(_runs(384, 512), cfg, budget_registration=reg)
    assert inv["ok"] is True, inv["mismatched_keys"]
    by_arm = inv["arm_specific_generation_params"]["by_arm"]
    assert by_arm[g3.ARM_H1] == {"expected": 384, "actual": 384, "ok": True}
    assert by_arm[g3.ARM_MCQ] == {"expected": 512, "actual": 512, "ok": True}
    assert inv["arm_specific_generation_params"]["budget_registration"]["sha256"] == reg.sha256


def test_g4_and_production_accept_384_512(cfg, reg):
    for arm, budget in ((g3.ARM_H1, 384), (g3.ARM_MCQ, 512)):
        chk = g4.matches_g3_production_fingerprint(
            arm, _fp(arm, budget), _fp(arm, budget), cfg=cfg, budget_registration=reg)
        assert chk["ok"] is True, (arm, chk["mismatched_keys"])
        chk = prod.check_fingerprint(arm, _fp(arm, budget), _fp(arm, budget),
                                     cfg=cfg, budget_registration=reg)
        assert chk["ok"] is True, (arm, chk["mismatched_keys"])


# ══════════════════════════════════════════════════════════════════════════
# 2. 旧 Qwen 的 384 / 16：无登记，仍然通过
# ══════════════════════════════════════════════════════════════════════════
def test_legacy_qwen_384_16_still_passes(cfg):
    assert g3.backend_invariant(_runs(384, 16), cfg)["ok"] is True
    for arm, budget in ((g3.ARM_H1, 384), (g3.ARM_MCQ, 16)):
        assert g4.matches_g3_production_fingerprint(
            arm, _fp(arm, budget), _fp(arm, budget), cfg=cfg)["ok"] is True
        assert prod.check_fingerprint(arm, _fp(arm, budget), _fp(arm, budget),
                                      cfg=cfg)["ok"] is True


def test_legacy_path_still_rejects_512_without_registration(cfg):
    inv = g3.backend_invariant(_runs(384, 512), cfg)
    assert inv["ok"] is False
    assert "generation_params.max_new_tokens[mcq]" in inv["mismatched_keys"]


# ══════════════════════════════════════════════════════════════════════════
# 3. 拒绝：错误预算 / 登记指向别的臂
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("h1, mcq, bad_arm", [
    (384, 16, "mcq"),      # 登记了 512 却跑出 16
    (384, 1024, "mcq"),    # 跑出一个没登记过的数
    (512, 512, "h1"),      # H1 臂没登记，却用了 512
])
def test_wrong_budget_is_rejected(cfg, reg, h1, mcq, bad_arm):
    inv = g3.backend_invariant(_runs(h1, mcq), cfg, budget_registration=reg)
    assert inv["ok"] is False
    assert f"generation_params.max_new_tokens[{bad_arm}]" in inv["mismatched_keys"]


def test_registration_for_the_other_arm_does_not_licence_this_one(cfg, reg):
    """登记的是 MCQ 臂 —— H1 臂仍按冻结值比对。"""
    chk = g4.matches_g3_production_fingerprint(
        g3.ARM_H1, _fp("h1", 512), _fp("h1", 512), cfg=cfg, budget_registration=reg)
    assert chk["ok"] is False
    assert chk["max_new_tokens"]["expected"] == 384


# ══════════════════════════════════════════════════════════════════════════
# 4. 拒绝：新旧证据混用
# ══════════════════════════════════════════════════════════════════════════
def test_g4_512_cannot_inherit_a_16_token_g3_fingerprint(cfg, reg):
    chk = g4.matches_g3_production_fingerprint(
        g3.ARM_MCQ, _fp("mcq", 512), _fp("mcq", 16), cfg=cfg, budget_registration=reg)
    assert chk["ok"] is False
    assert "g3.generation_params.max_new_tokens[mcq]" in chk["mismatched_keys"]


def test_production_512_cannot_inherit_a_16_token_reference(cfg, reg):
    chk = prod.check_fingerprint(g3.ARM_MCQ, _fp("mcq", 512), _fp("mcq", 16),
                                 cfg=cfg, budget_registration=reg)
    assert chk["ok"] is False
    assert "reference.generation_params.max_new_tokens[mcq]" in chk["mismatched_keys"]


def test_missing_schema_tag_is_still_incomparable(cfg, reg):
    """缺字段 ≠ 与任意取值兼容 —— 这条老规则没被预算登记削弱。"""
    runs = {g3.ARM_H1: g3.ArmRun(arm=g3.ARM_H1, fingerprint=_fp("h1", 384)),
            g3.ARM_MCQ: g3.ArmRun(arm=g3.ARM_MCQ,
                                  fingerprint=_fp("mcq", 512, schema=False))}
    inv = g3.backend_invariant(runs, cfg, budget_registration=reg)
    assert inv["ok"] is False
    assert any("schema" in str(k) for k in inv["mismatched_keys"])


# ══════════════════════════════════════════════════════════════════════════
# 5. load_budget_registration：不能只凭一个 JSON 放行
# ══════════════════════════════════════════════════════════════════════════
def test_drifted_registration_is_rejected(tmp_path, cfg):
    """登记文件在长度检查跑完之后被改过 → 报告引用的 sha256 对不上。"""
    path = _make_registration(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["reason"] = "偷偷改一个字段"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="不是同一份"):
        g3.load_budget_registration(path, cfg=cfg, model_path=MODEL)


def test_failed_length_check_does_not_licence_a_budget(tmp_path, cfg):
    path = _make_registration(tmp_path, check_passed=False)
    with pytest.raises(ValueError, match="passed"):
        g3.load_budget_registration(path, cfg=cfg, model_path=MODEL)


def test_length_check_for_a_different_budget_is_rejected(tmp_path, cfg):
    path = _make_registration(tmp_path, budget=512, check_budget=1024)
    with pytest.raises(ValueError, match="预算与本登记不一致"):
        g3.load_budget_registration(path, cfg=cfg, model_path=MODEL)


def test_calibration_list_hash_mismatch_is_rejected(tmp_path, cfg):
    path = _make_registration(tmp_path, list_sha="0" * 64)
    with pytest.raises(ValueError, match="校准题单"):
        g3.load_budget_registration(path, cfg=cfg, model_path=MODEL)


def test_wrong_model_is_rejected(tmp_path, cfg):
    path = _make_registration(tmp_path)
    with pytest.raises(ValueError, match="model_path"):
        g3.load_budget_registration(path, cfg=cfg,
                                    model_path="D:/models/Qwen2.5-7B-Instruct")


def test_registration_must_name_the_real_frozen_value(tmp_path, cfg):
    """登记里写错冻结值 = 试图把"改冻结值"伪装成"从冻结值扩展"。"""
    path = _make_registration(tmp_path, frozen=384)
    with pytest.raises(ValueError, match="frozen_max_new_tokens"):
        g3.load_budget_registration(path, cfg=cfg, model_path=MODEL)


def test_real_registration_on_disk_loads_and_links_to_a_passed_check(cfg):
    """真实落盘的那份登记：必须能载入，且确实连到一次通过的生成长度检查。"""
    real = _REPO_ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
    if not real.exists():
        pytest.skip("真实登记文件不在（本测试只在本机产物齐备时运行）")
    loaded = g3.load_budget_registration(real, cfg=cfg, model_path=MODEL)
    assert loaded.arm == g3.ARM_MCQ
    assert (loaded.frozen_max_new_tokens, loaded.registered_max_new_tokens) == (16, 512)
    assert loaded.length_check["passed"] is True
    assert loaded.length_check["touched_max_new_tokens_rate"] == 0.0
