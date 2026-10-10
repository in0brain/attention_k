r"""生成长度扩展的**事前登记闸**。

**不加载模型、不跑 GPU、不碰真实 outputs/。**

钉住四件事：
  1. 预算 == 冻结值 → 闸返回 None，行为与加这道闸之前逐字相同（Qwen 路径不受影响）
  2. 预算 != 冻结值且**没有**登记 → 拒绝运行（不接受只在命令行上临时改一个数字）
  3. 登记与本次运行不符（臂 / 预算 / 冻结值 / 模型）→ 拒绝运行
  4. 登记相符 → 放行，并把登记内容与哈希带进产物

以及：**未传登记时**下游的长度一致性检查一字未动 —— `backend_invariant` 仍按
v3_config 的冻结值逐臂精确比对，扩展预算的指纹会被它拒绝。传入已核验的登记后
该臂改比登记值（见 `tests/test_llama_arm_budget_downstream.py`），**仍是精确比对**。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location(
        "_budget_g3_entry", _REPO_ROOT / "scripts" / "run_stage1_g3_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg():
    return V3AnalysisConfig()


def _args(*, mcq=None, h1=None, registration=None, model_path=MODEL):
    c = V3AnalysisConfig()
    return SimpleNamespace(
        max_new_tokens=(c.h1_max_new_tokens if h1 is None else h1),
        mcq_max_new_tokens=(c.mcq_max_new_tokens if mcq is None else mcq),
        budget_registration=registration, model_path=model_path)


def _registration(tmp_path: Path, *, budget=512, frozen=16, arm="mcq",
                  model_path=MODEL, **over) -> Path:
    """一份**自洽**的登记：含校准题单与配套的、通过的生成长度检查。

    闸自 2026-09-14 起复用共享的 `load_budget_registration`，后者要求登记与
    一次已通过的长度检查逐项对上 —— 只有一个孤立 JSON 不再放行任意预算。
    """
    reg_dir = tmp_path / "outputs" / "stage1_llama_v3"
    cal_dir = reg_dir / "mcq_budget512_check"
    cal_dir.mkdir(parents=True, exist_ok=True)
    list_path = cal_dir / "mcq_budget_check_questions.jsonl"
    list_path.write_text('{"question_hash":"abc"}\n', encoding="utf-8")
    list_sha = g3._sha256_file(list_path)

    doc = {"schema": "stage1_generation_budget_registration_v1",
           "arm": arm, "model_path": model_path,
           "frozen_max_new_tokens": frozen, "registered_max_new_tokens": budget,
           "status": "候选，未经证明", "reason": "测试用", "claim_limits": [],
           "calibration_set": {
               "path": "outputs/stage1_llama_v3/mcq_budget512_check/"
                       "mcq_budget_check_questions.jsonl",
               "list_sha256": list_sha}}
    doc.update(over)
    path = reg_dir / "reg.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    (cal_dir / "mcq_budget_check_report.json").write_text(json.dumps({
        "passed": True, "n_rows": 120,
        "checks": {"touched_max_new_tokens_rate": 0.0},
        "registration": {"sha256": g3._sha256_file(path),
                         "registered_max_new_tokens": budget},
        "calibration_set": {"sha256": list_sha}}, ensure_ascii=False), encoding="utf-8")
    return path


# ── 1. 冻结预算：闸不介入 ────────────────────────────────────────────────
def test_frozen_budget_needs_no_registration(entry, cfg):
    assert entry._budget_extension(g3.ARM_MCQ, _args(), cfg) is None
    assert entry._budget_extension(g3.ARM_H1, _args(), cfg) is None


def test_frozen_budget_ignores_a_registration_that_is_present(entry, cfg, tmp_path):
    """预算没变就是没变 —— 有登记文件也不会凭空变成"扩展运行"。"""
    args = _args(registration=_registration(tmp_path))
    assert entry._budget_extension(g3.ARM_MCQ, args, cfg) is None


# ── 2. 非冻结预算 + 无登记 → 拒绝 ───────────────────────────────────────
def test_extended_budget_without_registration_is_refused(entry, cfg):
    with pytest.raises(SystemExit) as exc:
        entry._budget_extension(g3.ARM_MCQ, _args(mcq=512), cfg)
    assert "budget-registration" in str(exc.value)


def test_h1_arm_is_guarded_too(entry, cfg):
    with pytest.raises(SystemExit):
        entry._budget_extension(g3.ARM_H1, _args(h1=1024), cfg)


# ── 3. 登记与运行不符 → 拒绝 ────────────────────────────────────────────
@pytest.mark.parametrize("kw, needle", [
    ({"over": {"schema": "something_else"}}, "schema"),
    ({"over": {"frozen_max_new_tokens": 384}}, "frozen_max_new_tokens"),
    ({"model_path": "D:/models/Qwen2.5-7B-Instruct"}, "model_path"),
])
def test_registration_must_match_this_run(entry, cfg, tmp_path, kw, needle):
    over = kw.pop("over", {})
    args = _args(mcq=512, registration=_registration(tmp_path, **kw, **over))
    with pytest.raises(SystemExit) as exc:
        entry._budget_extension(g3.ARM_MCQ, args, cfg)
    assert needle in str(exc.value)


def test_registration_for_another_arm_is_refused(entry, cfg, tmp_path):
    """登记本身自洽（H1 臂、冻结值 384），但不能拿来给 MCQ 臂放行。"""
    args = _args(mcq=512, registration=_registration(tmp_path, arm="h1",
                                                     frozen=384, budget=512))
    with pytest.raises(SystemExit) as exc:
        entry._budget_extension(g3.ARM_MCQ, args, cfg)
    assert "不能用于" in str(exc.value)


def test_budget_that_differs_from_the_registered_one_is_refused(entry, cfg, tmp_path):
    """登记的是 128，本次却跑 512 —— 登记不覆盖这个数。"""
    args = _args(mcq=512, registration=_registration(tmp_path, budget=128))
    with pytest.raises(SystemExit) as exc:
        entry._budget_extension(g3.ARM_MCQ, args, cfg)
    assert "登记的预算" in str(exc.value)


def test_a_lone_json_without_a_passed_length_check_is_refused(entry, cfg, tmp_path):
    """**不能只凭一个登记文件放行任意预算** —— 必须连到一次已通过的长度检查。"""
    path = _registration(tmp_path)
    (path.parent / "mcq_budget512_check" / "mcq_budget_check_report.json").unlink()
    with pytest.raises(SystemExit) as exc:
        entry._budget_extension(g3.ARM_MCQ, _args(mcq=512, registration=path), cfg)
    assert "生成长度检查" in str(exc.value)


# ── 4. 相符 → 放行并留痕 ────────────────────────────────────────────────
def test_matching_registration_is_recorded(entry, cfg, tmp_path):
    path = _registration(tmp_path)
    stamp = entry._budget_extension(g3.ARM_MCQ, _args(mcq=512, registration=path), cfg)
    assert stamp["registered"] is True
    assert stamp["effective_max_new_tokens"] == 512
    assert stamp["frozen_max_new_tokens"] == cfg.mcq_max_new_tokens == 16
    assert stamp["registration_sha256"] == __import__("hashlib").sha256(
        path.read_bytes()).hexdigest()
    # 登记必须带着它的证据走：报告里看得见是哪一次长度检查放行了这个预算
    assert stamp["length_check"]["passed"] is True
    # 声明边界必须随产物走，不能只留在命令行里
    assert "扩展验证" in stamp["note"] and "严格同参数复现" in stamp["note"]


# ── 下游长度一致性检查未被删除 / 未被放宽 ───────────────────────────────
def _fp(arm, max_new):
    return {"model": {"m": 1}, "tokenizer": {"t": 1}, "hidden_index": {"h": 1},
            "generation_params": {"K": 6, "temperature": 0.7, "top_p": 0.95,
                                  "seed": 4242, "top_k": 20, "repetition_penalty": 1.05,
                                  "schema": g3.GENERATION_PARAMS_SCHEMA,
                                  "max_new_tokens": max_new},
            "run_hash": f"x_{arm}"}


def test_backend_invariant_still_rejects_the_extended_budget(cfg):
    """扩展预算的指纹**仍然**过不了两臂汇总 —— 这是预期行为，不是缺陷。"""
    runs = {g3.ARM_H1: g3.ArmRun(arm=g3.ARM_H1, fingerprint=_fp("h1", 384)),
            g3.ARM_MCQ: g3.ArmRun(arm=g3.ARM_MCQ, fingerprint=_fp("mcq", 512))}
    inv = g3.backend_invariant(runs, cfg)
    assert inv["ok"] is False
    assert "generation_params.max_new_tokens[mcq]" in inv["mismatched_keys"]
    by_arm = inv["arm_specific_generation_params"]["by_arm"]
    assert by_arm[g3.ARM_MCQ] == {"expected": 16, "actual": 512, "ok": False}


def test_backend_invariant_unchanged_for_the_frozen_pair(cfg):
    runs = {g3.ARM_H1: g3.ArmRun(arm=g3.ARM_H1, fingerprint=_fp("h1", 384)),
            g3.ARM_MCQ: g3.ArmRun(arm=g3.ARM_MCQ, fingerprint=_fp("mcq", 16))}
    assert g3.backend_invariant(runs, cfg)["ok"] is True
