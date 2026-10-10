"""v2.4：gate 解耦 + 性能盲 G4 + 一次性揭盲。

三条新约束的价值全在于**它们拦得住**:
  - G4 报告里出现任何性能字段 → 抛错(否则等于冻结前偷看结果)
  - 分析第二次揭盲 → 必须显式标 rerun 并给理由,无法隐形
  - 产物在冻结后被动过 → 分析必须停
以及 gate 解耦本身:G1 红不再阻塞 Stage 0,但仍阻塞投稿。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import stage0_preflight as pf
from recover_attention.evaluation import unblinding as ub
from recover_attention.generation import quality_check as qc

OK = {"ok": True}


def _gates(g1=True, g2=True, g3=True):
    return {"G1": {"ok": g1, "status": "confirmed" if g1 else "provisional",
                   "missing": [] if g1 else ["page_limit"]},
            "G2": {"ok": g2, "current": "x", "locked": "x"},
            "G3": {"ok": g3, "h1_arm": OK, "mcq_arm": OK, "backend_invariant": OK}}


# ---- gate 解耦（v2.4 的核心）----
def test_cfp_no_longer_blocks_stage0_but_still_blocks_submission():
    """G1 是场地事实:决定能报告多少,不决定测到了什么(§2)。"""
    v = pf.launch_readiness(_gates(g1=False), OK, OK, OK, OK, g4=OK)
    assert v["stage0_launch_allowed"] is True, "v2.4 起 CFP 不再阻塞实验"
    assert v["stage0_blocked_by"] == []
    assert v["submission_allowed"] is False
    assert v["submission_blocked_by"] == ["external_cfp_confirmation"]


def test_scientific_gates_still_block_stage0():
    for patch, expected in ((_gates(g2=False), "preregistration_hash_mismatch"),
                            (_gates(g3=False), "model_smoke_not_green")):
        v = pf.launch_readiness(patch, OK, OK, OK, OK, g4=OK)
        assert v["stage0_launch_allowed"] is False
        assert expected in v["stage0_blocked_by"]


def test_missing_g4_blocks_stage0():
    """G4 未跑 → 不许启动全量。"""
    v = pf.launch_readiness(_gates(g1=False), OK, OK, OK, OK, g4=None)
    assert v["stage0_launch_allowed"] is False
    assert "production_quality_check_not_passed" in v["stage0_blocked_by"]


def test_g1_still_content_validated_not_existence():
    """解耦不等于放松:G1 仍校验内容,仍不得由占位文件刷绿。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = Path(d) / "cfp.json"
        rec.write_text(json.dumps({"status": "confirmed", "deadline": "d",
                                   "archival": False, "page_limit": None}), encoding="utf-8")
        g = pf.check_gates(ROOT / "docs/paper/preregistration.md",
                           ROOT / "docs/paper/preregistration.lock", rec, Path(d))
        assert g["G1"]["ok"] is False and "page_limit" in g["G1"]["missing"]


# ---- G4 性能盲（代码强制）----
def test_quality_report_rejects_performance_fields():
    for bad in ({"auroc": 0.9}, {"per_arm": {"h1": {"delta_h": 0.01}}},
                {"x": [{"ci_lo": 0.0}]}, {"ladder": {}}, {"oof_scores": []}):
        with pytest.raises(qc.PerformanceLeakInQualityCheck):
            qc.assert_no_performance_fields(bad)


def test_quality_report_allows_operational_fields():
    qc.assert_no_performance_fields({
        "per_arm": {"h1": {"stats": {"eligible_rate": 0.9, "n_positive": 5,
                                     "f5_completeness": {"nonzero_per_column": {"a": 3}}}}},
        "backend_invariant": {"ok": True}})


def test_quality_check_module_does_not_import_evaluation():
    """性能盲的结构性保证:G4 模块不接触分析实现。"""
    import ast
    src = (ROOT / "src/recover_attention/generation/quality_check.py").read_text(encoding="utf-8")
    tree = ast.parse(src)                     # 查 import 语句，不查注释/docstring
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = ("evaluation", "sklearn", "roc_auc", "average_precision",
              "conditional_increment")
    leaked = [m for m in imported if any(b in m for b in banned)]
    assert not leaked, f"G4 不得导入分析实现: {leaked}"


def test_h1_quality_check_flags_low_eligible_rate():
    rows = [{"trace_id": f"t{i}", "example_id": "e", "group_id": "g", "eligible": i < 3,
             "label": i % 2, "prompt_hash": "p", "unit_hash": "u",
             "primary_exclusion_reason": None if i < 3 else "no_identifier",
             "emission_failure": i >= 3,
             "hidden_pooled_n_tokens": 1, "id_token_positions_local": [1],
             "f5_id_logprob_mean": -1.0 - i} for i in range(10)]
    hidden = {f"t{i}": np.ones(4) for i in range(3)}
    r = qc.check_h1_batch(rows, hidden, ["f5_id_logprob_mean"])
    assert r["checks"]["eligible_rate_ok"] is False      # 0.3 < 0.80
    assert r["ok"] is False
    qc.assert_no_performance_fields(r)


def test_every_launch_entrypoint_uses_the_v24_gate_not_g1():
    """v2.4 迁移必须覆盖**所有**启动入口，不能只改 preflight。

    回归:启动 H1 全量时被 run_stage0_generation.py 自己的 require_stage0_gate 拦下,
    因为那里还写着 v2.3 的 "G1+G2+G3"。纵深防御拦住了,但那是迁移不完整。
    """
    import ast
    src = (ROOT / "scripts" / "run_stage0_generation.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "require_stage0_gate")
    body = ast.get_source_segment(src, fn)
    assert "check_cfp_confirmed" not in body, \
        "科学启动闸不得校验 CFP —— v2.4 已把它拆成投稿闸"
    assert "check_g4" in body, "科学启动闸必须包含 G4"


def test_h1_eligible_rate_ignores_cve_family_by_design():
    """§3:主 population 是 ATT&CK/CWE。CVE 族的题模型正确地答 CVE → only_cve,
    那是**设计使然**,不是质量缺陷。把 CVE 题算进分母会惩罚正确的模型输出。

    回归:G4 首跑样本是 12 attack + 8 cve,48 条 only_cve 全来自那 8 个 CVE 题,
    整体 rate 0.60 触发失败,而 attack 题的 eligible 率其实是 100%。
    """
    def _row(i, fam, eligible):
        return {"trace_id": f"t{i}", "example_id": f"e{i}", "group_id": f"g{i}",
                "family": fam, "eligible": eligible, "label": i % 2,
                "prompt_hash": "p", "unit_hash": "u",
                "primary_exclusion_reason": None if eligible else "only_cve",
                "emission_failure": False, "hidden_pooled_n_tokens": 1,
                "id_token_positions_local": [1], "f5_id_logprob_mean": -1.0 - i}

    rows = ([_row(i, "attack", True) for i in range(6)]
            + [_row(i + 6, "cwe", True) for i in range(6)]
            + [_row(i + 12, "cve", False) for i in range(8)])   # CVE 全 ineligible
    hidden = {f"t{i}": np.ones(4) for i in range(12)}
    r = qc.check_h1_batch(rows, hidden, ["f5_id_logprob_mean"])
    assert r["stats"]["eligible_rate_all_traces"] == pytest.approx(12 / 20)   # 0.60
    assert r["stats"]["eligible_rate_among_capable"] == 1.0                   # attack+cwe 全过
    assert r["checks"]["eligible_rate_ok"] is True, "不得因 CVE 题被排除而判失败"
    assert r["checks"]["sample_covers_both_primary_families"] is True


def test_h1_quality_check_requires_both_primary_families_in_sample():
    """质检样本漏掉一个主 family（如全无 cwe）→ 该 family 根本没被验到。"""
    rows = [{"trace_id": f"t{i}", "example_id": f"e{i}", "group_id": f"g{i}",
             "family": "attack", "eligible": True, "label": i % 2,
             "prompt_hash": "p", "unit_hash": "u", "primary_exclusion_reason": None,
             "emission_failure": False, "hidden_pooled_n_tokens": 1,
             "id_token_positions_local": [1], "f5_id_logprob_mean": -1.0 - i}
            for i in range(6)]
    r = qc.check_h1_batch(rows, {f"t{i}": np.ones(4) for i in range(6)},
                          ["f5_id_logprob_mean"])
    assert r["checks"]["sample_covers_both_primary_families"] is False
    assert r["ok"] is False


def test_mcq_quality_check_flags_parse_failures_and_f5_gap():
    rows = [{"example_id": f"e{i}", "group_id": f"g{i}", "parsed_label": "A" if i < 5 else None,
             "canonicalization_status": "ok" if i < 5 else "parse_failure",
             "wrong_label": i % 2, "prompt_hash": "p", "unit_hash": "u",
             "response_format": "bare_label", "semantic_output_text": "text",
             "readout_to_prompt_end_distance": 0,
             "f5_label_margin": 0.0} for i in range(10)]     # 整列 0 → F5 缺失
    hidden = {f"e{i}": np.ones(4) for i in range(5)}
    r = qc.check_mcq_batch(rows, hidden, ["f5_label_margin"])
    assert r["checks"]["parse_failure_rate_ok"] is False           # 0.5 > 0.10
    assert r["checks"]["f5_all_continuous_columns_populated"] is False
    assert r["ok"] is False


def test_build_g4_report_requires_shared_backend():
    per_arm = {"h1": {"ok": True}, "mcq": {"ok": True}}
    hv = {"h1": {"numeric_match": True}, "mcq": {"numeric_match": True}}
    same = {"model": {"backend": {"load_in_8bit": True, "device_map": "auto",
                                  "attn_implementation": "eager", "local_files_only": True}}}
    rep = qc.build_g4_report(per_arm, hv, {"h1": same, "mcq": same}, "sha")
    assert rep["passed"] is True and rep["backend_invariant"]["ok"] is True

    four = {"model": {"backend": {"load_in_8bit": False, "device_map": "auto",
                                  "attn_implementation": "eager", "local_files_only": True}}}
    rep2 = qc.build_g4_report(per_arm, hv, {"h1": same, "mcq": four}, "sha")
    assert rep2["passed"] is False and rep2["backend_invariant"]["ok"] is False


# ---- §2.2 一次性揭盲 ----
def test_second_unblinding_must_be_declared_a_rerun(tmp_path):
    log = tmp_path / "unblinding.jsonl"
    first = ub.record_unblinding(log, freeze_hash="f", config_hash="c", timestamp="t1")
    assert first["role"] == "confirmatory" and first["is_rerun"] is False

    with pytest.raises(ValueError, match="already unblinded"):
        ub.record_unblinding(log, freeze_hash="f", config_hash="c", timestamp="t2")

    with pytest.raises(ValueError, match="must state a reason"):
        ub.record_unblinding(log, freeze_hash="f", config_hash="c", timestamp="t2",
                             is_rerun=True)

    second = ub.record_unblinding(log, freeze_hash="f", config_hash="c", timestamp="t2",
                                  is_rerun=True, reason="bug in plotting code")
    assert second["role"] == "rerun_1" and second["reason"] == "bug in plotting code"
    assert ub.confirmatory_entry(log)["role"] == "confirmatory"


def test_freeze_manifest_detects_post_freeze_edits(tmp_path):
    rec = tmp_path / "h1_completion_records.jsonl"
    hid = tmp_path / "h1_hidden.npz"
    rec.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    np.savez_compressed(hid, x=np.ones(3))
    arms = {"h1": {"records": rec, "hidden": hid}}

    man = ub.build_freeze_manifest(tmp_path, arms)
    assert man["arms"]["h1"]["n_records"] == 2
    assert ub.verify_freeze(man)["ok"] is True

    rec.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")   # 冻结后追加
    v = ub.verify_freeze(man)
    assert v["ok"] is False and v["drift"][0]["issue"] == "sha256 changed"


def test_freeze_hash_changes_with_content(tmp_path):
    rec = tmp_path / "r.jsonl"; hid = tmp_path / "h.npz"
    rec.write_text('{"a": 1}\n', encoding="utf-8"); np.savez_compressed(hid, x=np.ones(3))
    a = ub.build_freeze_manifest(tmp_path, {"h1": {"records": rec, "hidden": hid}})
    rec.write_text('{"a": 2}\n', encoding="utf-8")
    b = ub.build_freeze_manifest(tmp_path, {"h1": {"records": rec, "hidden": hid}})
    assert a["freeze_hash"] != b["freeze_hash"]


# ---- §2.2 在**生产入口**上真的接线了吗（回归） ----
#
# 前四次同类事故的形状完全一致:模块存在、文档写清、单测通过,但生产路径漏调
# （attach_cross_sample_f5 / wrong_label / _requires_hidden / 以及这里的 freeze+unblinding）。
# 上面那些测试只证明 unblinding.py 这个**模块**是对的,证明不了 run_stage0_analysis.py
# **调用**了它。以下三条专门钉住调用点。
import importlib.util  # noqa: E402

_ana_spec = importlib.util.spec_from_file_location(
    "run_stage0_analysis", ROOT / "scripts" / "run_stage0_analysis.py")
ana = importlib.util.module_from_spec(_ana_spec)
_ana_spec.loader.exec_module(ana)


class _Args:
    """production() / freeze() 需要的最小参数集。"""

    def __init__(self, tmp_path, **kw):
        self.input_dir = str(tmp_path / "feat")
        self.output_dir = str(tmp_path / "eval")
        self.freeze_manifest = str(tmp_path / "eval" / "freeze.json")
        self.unblinding_log = str(tmp_path / "eval" / "unblinding.jsonl")
        self.config = "configs/stage0.yaml"
        self.prereg = str(ROOT / "docs" / "paper" / "preregistration.md")
        self.lock = str(ROOT / "docs" / "paper" / "preregistration.lock")
        self.seed = 0
        self.timestamp = "2026-07-21T22:00:00Z"
        self.rerun = False
        self.reason = None
        self.refreeze = False
        for k, v in kw.items():
            setattr(self, k, v)


def _fake_arm_artifacts(tmp_path):
    """两臂的最小产物（内容无意义，只用于哈希/存在性）。"""
    feat = tmp_path / "feat"
    feat.mkdir(parents=True, exist_ok=True)
    for arm in ("h1", "mcq"):
        (feat / f"{arm}_completion_records.jsonl").write_text(
            '{"example_id": "a"}\n', encoding="utf-8")
        np.savez_compressed(feat / f"{arm}_hidden.npz", a=np.ones(4, dtype=np.float32))
        (feat / f"{arm}_run_fingerprint.json").write_text(
            json.dumps({"run_hash": f"{arm}hash"}), encoding="utf-8")
    return feat


def test_production_refuses_to_run_without_a_freeze(tmp_path):
    """揭盲前必须先冻结。没有 freeze manifest 就分析 = 产物还能在结果底下改。"""
    _fake_arm_artifacts(tmp_path)
    with pytest.raises(SystemExit, match="no freeze manifest"):
        ana.production(_Args(tmp_path))


def test_production_stops_when_artifacts_drifted_after_freeze(tmp_path):
    """冻结后产物被动过 → 分析必须停,而不是照跑出一个"预注册的"结果。"""
    feat = _fake_arm_artifacts(tmp_path)
    ana.freeze(_Args(tmp_path))
    (feat / "h1_completion_records.jsonl").write_text(
        '{"example_id": "a"}\n{"example_id": "b"}\n', encoding="utf-8")   # 冻结后追加
    with pytest.raises(SystemExit, match="no longer be the preregistered one"):
        ana.production(_Args(tmp_path))


def test_freeze_refuses_to_silently_overwrite_an_existing_manifest(tmp_path):
    """重新冻结会把审计链悄悄挪到新数据上 —— 必须显式 --refreeze。"""
    _fake_arm_artifacts(tmp_path)
    ana.freeze(_Args(tmp_path))
    with pytest.raises(SystemExit, match="already exists"):
        ana.freeze(_Args(tmp_path))
    man = ana.freeze(_Args(tmp_path, refreeze=True))     # 显式声明则放行
    assert "freeze_hash" in man
    assert list((tmp_path / "eval").glob("freeze.superseded_*.json")), "旧 manifest 必须留档"
