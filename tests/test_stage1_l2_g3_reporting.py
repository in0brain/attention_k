"""Stage1-L2：G3 报告真实性与汇总拒绝条件。

**纯离线：合成报告 + 假后端，不加载模型、不跑 GPU、不碰真实 outputs/。**

覆盖 task card §7 的六条：

    1  默认 Qwen 的判定与报告几何仍为 28/19/20/3584
    2  Llama 的判定、单臂报告与汇总几何均为 32/22/23/4096
    3  缺 criteria、两臂不一致、混入 dry-run 时，真实汇总拒绝通过
    4  用**不同于 96/120** 的合成计数验证汇总据实报告
    5  Llama 报告不宣称经历旧 Qwen 的历史纠正事件
    6  不变的非劣/性能分析边界、无指标日志约束和原有拒绝路径

设计要点：先证明正常路径通过，再逐条注入缺陷证明它被抓住（故障注入）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

_REAL_OUTPUTS = _REPO_ROOT / "outputs"


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "_l2_g3_entry", _REPO_ROOT / "scripts" / "run_stage1_g3_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def entry():
    return _load_entry()


@pytest.fixture(scope="module")
def cfg():
    return V3AnalysisConfig()


QWEN_CRIT = g3.G3Criteria()
LLAMA_GEO = repool.ModelGeometry.from_num_layers(32, 4096)
LLAMA_CRIT = g3.G3Criteria(hidden_dim=LLAMA_GEO.hidden_dim,
                           block_index=LLAMA_GEO.block_index,
                           tuple_index=LLAMA_GEO.tuple_index)


# ═════════════════════════════════════════════ 真实产物守卫
def _tree_state(root: Path) -> dict:
    if not root.exists():
        return {"exists": False}
    n = total = newest = 0
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            n += 1
            total += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return {"exists": True, "n_files": n, "total_bytes": total, "newest": newest}


@pytest.fixture(scope="module", autouse=True)
def _real_outputs_untouched():
    watched = [_REAL_OUTPUTS / d for d in ("stage1", "stage1_llama_v3", "stage0")]
    before = {str(d): _tree_state(d) for d in watched}
    yield
    after = {str(d): _tree_state(d) for d in watched}
    changed = [k for k in after if after[k] != before[k]]
    assert not changed, f"真实 outputs/ 被本测试文件改动了：{changed}"


# ═════════════════════════════════════════════ §7.1 / §7.2 几何贯通
def _fake_run(arm: str, *, geo: repool.ModelGeometry, n_traces: int = 120):
    """构造一个最小 ArmRun：只用于让 build_report 走通，不做数值主张。"""
    fp = {"model": {"model_path": "<synthetic>", "num_hidden_layers": geo.num_layers,
                    "hidden_size": geo.hidden_dim},
          "tokenizer": {"tokenizer_hash": "t" * 8},
          "hidden_index": {"num_layers": geo.num_layers,
                           "block_index_zero_based": geo.block_index,
                           "hidden_states_tuple_index": geo.tuple_index},
          "generation_params": {"K": 6, "temperature": 0.7, "top_p": 0.95, "seed": 4242,
                                "max_new_tokens": 384 if arm == g3.ARM_H1 else 16}}
    run = g3.ArmRun(arm=arm, fingerprint=fp)
    run.n_units = n_traces
    run.hidden_verification = {"block_index_zero_based": geo.block_index,
                               "hidden_states_tuple_index": geo.tuple_index,
                               "numeric_match": True, "max_abs_diff": 0.0}
    return run


def _verification() -> dict:
    return {"g2": {"current_sha256": "s" * 64, "hash_mode": "text-lf-utf8",
                   "locked_version": "v3.1"},
            "sample_manifest_sha256": "m" * 64,
            "chain_sha256_verified": True,
            "primary_lists_read": False}


def _evaluation() -> dict:
    return {"checks": {"n_prompts_ok": True, "_detail": {"n_rows": 120}},
            "failed_checks": [], "passed": True}


def test_default_report_geometry_is_still_qwen(cfg):
    """§7.1 省略 criteria 时，报告的几何必须仍是 28/19/20/3584。"""
    rep = g3.build_report(arm=g3.ARM_H1, run=_fake_run(g3.ARM_H1, geo=repool.QWEN_GEOMETRY),
                          verification=_verification(), evaluation=_evaluation(),
                          cfg=cfg, n_records=20)
    c = rep["criteria"]
    assert (c["hidden_dim"], c["block_index_zero_based"],
            c["hidden_states_tuple_index"]) == (3584, 19, 20)


def test_report_records_the_criteria_actually_used_not_the_module_default(cfg):
    """§7.2 传入 Llama criteria → 报告必须写 32→22/23/4096，不得回落 Qwen 默认。

    这是本 sprint 修的核心缺陷：先前 build_report 写死 CRITERIA.as_dict()，
    判定用 Llama 几何而报告声称用了 Qwen 几何。
    """
    rep = g3.build_report(arm=g3.ARM_H1, run=_fake_run(g3.ARM_H1, geo=LLAMA_GEO),
                          verification=_verification(), evaluation=_evaluation(),
                          cfg=cfg, n_records=20, criteria=LLAMA_CRIT)
    c = rep["criteria"]
    assert (c["hidden_dim"], c["block_index_zero_based"],
            c["hidden_states_tuple_index"]) == (4096, 22, 23)
    # 报告的几何必须与判定用的几何逐项一致
    assert c == LLAMA_CRIT.as_dict()
    # 且与同一份报告里的 hidden_verification 自洽
    hv = rep["hidden_verification"]
    assert hv["block_index_zero_based"] == c["block_index_zero_based"]
    assert hv["hidden_states_tuple_index"] == c["hidden_states_tuple_index"]


@pytest.mark.parametrize("crit,geo", [(QWEN_CRIT, repool.QWEN_GEOMETRY),
                                      (LLAMA_CRIT, LLAMA_GEO)])
def test_evaluate_and_report_agree_on_geometry(crit, geo, cfg):
    """判定与报告不能各用一套几何。"""
    run = _fake_run(g3.ARM_H1, geo=geo)
    ev = g3.evaluate_arm(run, n_records=20, criteria=crit)
    rep = g3.build_report(arm=g3.ARM_H1, run=run, verification=_verification(),
                          evaluation=ev, cfg=cfg, n_records=20, criteria=crit)
    assert rep["criteria"] == crit.as_dict()
    assert ev["checks"]["hidden_index_matches_frozen"] is True


# ═════════════════════════════════════════════ 汇总：本轮实测 + 拒绝条件
def _write_reports(out_dir: Path, *, crit_h1: dict, crit_mcq: dict,
                   dry_h1=False, dry_mcq=False,
                   h1_eligible=104, h1_rows=120, mcq_fail=3, mcq_rows=120,
                   passed=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = {"num_layers": 32, "block_index_zero_based": crit_h1["block_index_zero_based"],
           "hidden_states_tuple_index": crit_h1["hidden_states_tuple_index"]}
    common_fp = {"model": {"model_path": "<synthetic>"},
                 "tokenizer": {"tokenizer_hash": "t" * 8},
                 "hidden_index": geo}
    specs = {
        g3.ARM_H1: dict(criteria=crit_h1, dry_run=dry_h1, max_new=384,
                        checks={"eligible_rate": round(h1_eligible / h1_rows, 4),
                                "eligible_rate_ok": h1_eligible / h1_rows >= 0.80},
                        detail={"_detail": {"n_eligible": h1_eligible, "n_rows": h1_rows}}),
        g3.ARM_MCQ: dict(criteria=crit_mcq, dry_run=dry_mcq, max_new=16,
                         checks={"parse_failure_rate": round(mcq_fail / mcq_rows, 4),
                                 "parse_failure_rate_ok": mcq_fail / mcq_rows <= 0.10},
                         detail={"_detail": {"n_parse_failure": mcq_fail,
                                             "n_parsed": mcq_rows - mcq_fail,
                                             "n_rows": mcq_rows}}),
    }
    for arm, s in specs.items():
        fp = json.loads(json.dumps(common_fp))
        fp["generation_params"] = {"K": 6, "temperature": 0.7, "top_p": 0.95,
                                   "seed": 4242, "max_new_tokens": s["max_new"]}
        (out_dir / f"{arm}_g3_smoke_report.json").write_text(json.dumps({
            "arm": arm, "passed": passed, "dry_run": s["dry_run"],
            "criteria": s["criteria"], "fingerprint": fp,
            "checks": s["checks"], "detail": s["detail"],
            "n_prompts": 20, "n_traces": 120, "traces_per_prompt": 6,
            "n_rows_written": 120, "n_failures": 0,
            "hidden_verification": {"numeric_match": True, "max_abs_diff": 0.0},
            "elapsed_seconds": 1234.5,
        }, ensure_ascii=False), encoding="utf-8")


class _Args:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)


def test_summary_reports_this_rounds_real_counts(entry, cfg, tmp_path):
    """§7.4 用**不同于 96/120** 的合成计数，验证汇总据实报告。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict(),
                   h1_eligible=104, h1_rows=120, mcq_fail=3, mcq_rows=120)
    rc = entry._summarise(_Args(out), cfg, LLAMA_CRIT)
    assert rc == 0
    s = json.loads((out / "g3_summary.json").read_text(encoding="utf-8"))

    h1 = s["thresholds"]["h1"]
    assert (h1["numerator"], h1["denominator"]) == (104, 120)
    assert h1["value"] == pytest.approx(104 / 120, abs=1e-4)
    assert h1["exactly_at_threshold"] is False and h1["margin"] > 0
    mcq = s["thresholds"]["mcq"]
    assert (mcq["numerator"], mcq["denominator"]) == (3, 120)
    assert s["counts"]["h1"]["n_prompts"] == 20
    assert s["mode"] == "real" and s["real_g3_passed"] is True


def test_summary_flags_a_zero_margin_instead_of_calling_it_stable(entry, cfg, tmp_path):
    """恰好达标必须显式标出，不得写成"稳定通过"。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict(),
                   h1_eligible=96, h1_rows=120)
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 0
    h1 = json.loads((out / "g3_summary.json").read_text("utf-8"))["thresholds"]["h1"]
    assert h1["margin"] == 0.0 and h1["exactly_at_threshold"] is True
    assert "零余量" in h1["margin_caveat"]


def test_summary_geometry_is_the_llama_one(entry, cfg, tmp_path):
    """§7.2 汇总的几何也必须是 32→22/23/4096。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    entry._summarise(_Args(out), cfg, LLAMA_CRIT)
    g = json.loads((out / "g3_summary.json").read_text("utf-8"))["geometry"]
    assert g == {"hidden_dim": 4096, "block_index_zero_based": 22,
                 "hidden_states_tuple_index": 23}


def test_summary_does_not_inherit_qwen_history(entry, cfg, tmp_path):
    """§7.5 Llama 汇总不得宣称经历旧 Qwen 的历史纠正事件。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    entry._summarise(_Args(out), cfg, LLAMA_CRIT)
    text = (out / "g3_summary.json").read_text("utf-8")
    for banned in ("criterion_correction", "superseded_artifact",
                   "g3_summary_misspecified_check_20260808",
                   "fb231a4e", "96/120", "0.800"):
        assert banned not in text, f"汇总里出现了旧 Qwen 的历史痕迹：{banned}"
    s = json.loads(text)
    # 通用规则可以保留（两臂 max_new_tokens 不同），但只作规则、不带历史数值
    rule = s["arm_specific_max_new_tokens_rule"]
    assert rule["field"] == "generation_params.max_new_tokens"
    assert "superseded" not in json.dumps(rule, ensure_ascii=False)
    assert "本轮" in s["history_inheritance"]


@pytest.mark.parametrize("missing_arm", ["h1", "mcq"])
def test_summary_refuses_when_criteria_missing(entry, cfg, tmp_path, missing_arm):
    """§7.3 缺 criteria → 拒绝汇总（无法确认判定用的几何）。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    path = out / f"{missing_arm}_g3_smoke_report.json"
    doc = json.loads(path.read_text("utf-8"))
    doc.pop("criteria")
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 2
    assert not (out / "g3_summary.json").exists(), "拒绝时不得写出汇总"


def test_summary_refuses_when_arms_disagree_on_geometry(entry, cfg, tmp_path):
    """§7.3 两臂 criteria 不一致 → 拒绝（两臂跑在不同几何上）。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=QWEN_CRIT.as_dict())
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 2
    assert not (out / "g3_summary.json").exists()


def test_summary_refuses_when_cli_geometry_differs_from_the_reports(entry, cfg, tmp_path):
    """操作者传错几何 → 拒绝，不张冠李戴。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    assert entry._summarise(_Args(out), cfg, QWEN_CRIT) == 2
    assert not (out / "g3_summary.json").exists()


@pytest.mark.parametrize("dry_h1,dry_mcq", [(True, False), (False, True), (True, True)])
def test_dry_run_cannot_be_summarised_as_a_real_g3_pass(entry, cfg, tmp_path,
                                                        dry_h1, dry_mcq):
    """§7.3 混入 dry-run → 不得写成真实通过，也不得标生产指纹已冻结。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict(),
                   dry_h1=dry_h1, dry_mcq=dry_mcq)
    entry._summarise(_Args(out), cfg, LLAMA_CRIT)
    s = json.loads((out / "g3_summary.json").read_text("utf-8"))
    assert s["mode"] == "simulated"
    assert s["real_g3_passed"] is False
    assert s["production_fingerprint_is_now_frozen"] is False
    assert any("dry_run" in r for r in s["refusals"])


def test_summary_refuses_to_silently_overwrite_a_different_existing_summary(
        entry, cfg, tmp_path):
    """不静默覆盖已有 G3 结果；内容相同则 no-op。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 0
    first = (out / "g3_summary.json").read_bytes()
    # 完全相同的输入 → no-op，字节不变
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 0
    assert (out / "g3_summary.json").read_bytes() == first
    # 换了计数 → 拒绝覆盖，旧文件字节不变
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict(),
                   h1_eligible=110)
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 2
    assert (out / "g3_summary.json").read_bytes() == first


def test_summary_missing_arm_report_stops_before_anything_else(entry, cfg, tmp_path):
    """原有拒绝路径不变：缺任一臂报告即停。"""
    out = tmp_path / "g3"
    out.mkdir(parents=True)
    assert entry._summarise(_Args(out), cfg, LLAMA_CRIT) == 2


# ═════════════════════════════════════════════ §7.6 既有边界未被削弱
def test_performance_blind_boundary_is_unchanged(entry, cfg, tmp_path):
    """汇总仍是 gate_only / performance_blind，且不含任何性能量。"""
    out = tmp_path / "g3"
    _write_reports(out, crit_h1=LLAMA_CRIT.as_dict(), crit_mcq=LLAMA_CRIT.as_dict())
    entry._summarise(_Args(out), cfg, LLAMA_CRIT)
    s = json.loads((out / "g3_summary.json").read_text("utf-8"))
    assert s["gate_only"] is True and s["performance_blind"] is True
    assert set(g3.DID_NOT_COMPUTE) <= set(s["did_not_compute"])
    text = json.dumps(s, ensure_ascii=False).lower()
    for banned in ("auroc", "auprc", "delta_h", "ladder", "increment"):
        assert banned not in text or banned in json.dumps(
            s["did_not_compute"], ensure_ascii=False).lower()


def test_g3_criteria_thresholds_were_not_touched():
    """本 sprint 只改报告真实性与拒绝条件，**不改任何 gate 阈值**。"""
    c = g3.G3Criteria()
    assert c.n_prompts == 20 and c.traces_per_prompt == 6 and c.n_traces == 120
    assert c.h1_min_eligible_rate == 0.80
    assert c.mcq_max_parse_failure_rate == 0.10
    assert c.as_dict()["frozen_before_run"] is True


def test_backend_invariant_semantics_unchanged():
    """max_new_tokens 仍按臂分别精确匹配，不被并入两臂相同的那组。"""
    assert g3.FINGERPRINT_KEYS == ("model", "tokenizer", "hidden_index")
    assert g3.INVARIANT_GENERATION_PARAMS == ("K", "temperature", "top_p", "seed")
    assert g3.ARM_SPECIFIC_GENERATION_PARAMS == ("max_new_tokens",)


def test_entry_source_sha256_is_recordable():
    """卡 §6 要求逐文件 SHA-256：这些文件未被 Git 跟踪，git diff 不代表改动。"""
    for rel in ("scripts/run_stage1_g3_smoke.py",
                "src/recover_attention/evaluation/v3_g3_smoke.py"):
        digest = hashlib.sha256((_REPO_ROOT / rel).read_bytes()).hexdigest()
        assert len(digest) == 64
