"""Stage 1 正式生成入口的回归测试。

不加载模型、不读 `data/**`；除真实 prereg/lock（G2 接线）外全部合成。
重点：crash-point / orphan 的四种状态、eligible 条件下的 hidden、两层产物、分段与续跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation.manifest import WorkManifest  # noqa: E402

REAL_PREREG = _REPO_ROOT / "docs" / "paper" / "preregistration_v3.md"
REAL_LOCK = _REPO_ROOT / "docs" / "paper" / "preregistration_v3.lock"


def _cfg(**over) -> V3AnalysisConfig:
    base = dict(mode="dry_run", k_traces=3)
    base.update(over)
    return V3AnalysisConfig(**base)


# ── hidden 的 eligible 条件 ──────────────────────────────────────────────────
def test_requires_hidden_is_conditional_on_eligibility():
    assert prod.requires_hidden("h1", {"id_token_positions_local": [0]}) is True
    assert prod.requires_hidden("h1", {"id_token_positions_local": []}) is False
    assert prod.requires_hidden("h1", {}) is False
    assert prod.requires_hidden("mcq", {"eligible_for_primary": True}) is True
    assert prod.requires_hidden("mcq", {"eligible_for_primary": False}) is False


# ── crash-point / orphan 四种状态 ────────────────────────────────────────────
def _seed_raw(tmp_path: Path, arm: str, rows: list[dict], *, hidden_for: list[str],
              done: list[str]) -> Path:
    out = tmp_path / "prod"
    p = prod.paths_for(out, arm)
    p["hidden_dir"].mkdir(parents=True, exist_ok=True)
    p["raw_results"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for uid in hidden_for:
        np.save(p["hidden_dir"] / f"{uid}.npy", np.ones(4, dtype=np.float32))
    man = WorkManifest(p["manifest"])
    man.register(r["unit_id"] for r in rows)
    for uid in done:
        man.mark(uid, "done")
    return out


def _h1_row(uid: str, *, eligible: bool = True) -> dict:
    return {"unit_id": uid, "example_id": uid.split("__")[0], "sample_type": "greedy",
            "id_token_positions_local": [0] if eligible else []}


def test_eligible_done_without_hidden_is_refused(tmp_path):
    out = _seed_raw(tmp_path, "h1", [_h1_row("e0__greedy_0")],
                    hidden_for=[], done=["e0__greedy_0"])
    with pytest.raises(ValueError, match="eligible 却没有 hidden"):
        prod.audit_resume_state("h1", out)


def test_ineligible_done_without_hidden_is_allowed(tmp_path):
    """ineligible 行没有 hidden 是**合法终态**，不是数据丢失。"""
    out = _seed_raw(tmp_path, "h1", [_h1_row("e0__greedy_0", eligible=False)],
                    hidden_for=[], done=["e0__greedy_0"])
    audit = prod.audit_resume_state("h1", out)
    assert audit.n_done == 1 and audit.n_rows == 1 and audit.n_hidden_files == 0


def test_mcq_parse_failure_without_hidden_is_allowed(tmp_path):
    row = {"unit_id": "m0__greedy_0", "example_id": "m0", "sample_type": "greedy",
           "eligible_for_primary": False}
    out = _seed_raw(tmp_path, "mcq", [row], hidden_for=[], done=["m0__greedy_0"])
    assert prod.audit_resume_state("mcq", out).n_done == 1


def test_manifest_done_but_result_missing_is_refused(tmp_path):
    out = _seed_raw(tmp_path, "h1", [], hidden_for=[], done=[])
    man = WorkManifest(prod.paths_for(out, "h1")["manifest"])
    man.register(["e0__greedy_0"])
    man.mark("e0__greedy_0", "done")
    with pytest.raises(ValueError, match="raw 结果缺行"):
        prod.audit_resume_state("h1", out)


def test_orphan_hidden_without_result_is_refused(tmp_path):
    out = _seed_raw(tmp_path, "h1", [_h1_row("e0__greedy_0")],
                    hidden_for=["e0__greedy_0", "e1__greedy_0"], done=["e0__greedy_0"])
    with pytest.raises(ValueError, match="orphan hidden"):
        prod.audit_resume_state("h1", out)


def test_duplicate_unit_id_in_raw_is_refused(tmp_path):
    rows = [_h1_row("e0__greedy_0"), _h1_row("e0__greedy_0")]
    out = _seed_raw(tmp_path, "h1", rows, hidden_for=["e0__greedy_0"], done=[])
    with pytest.raises(ValueError, match="重复 unit_id"):
        prod.audit_resume_state("h1", out)


def test_result_and_hidden_present_but_not_marked_done_is_repaired_without_regeneration(tmp_path):
    """崩在 manifest 标记前 → 校验一致后**只补记 done**，不重新调用模型。"""
    out = _seed_raw(tmp_path, "h1", [_h1_row("e0__greedy_0")],
                    hidden_for=["e0__greedy_0"], done=[])
    audit = prod.audit_resume_state("h1", out)
    assert audit.repaired_done == ["e0__greedy_0"] and audit.n_done == 1
    again = prod.audit_resume_state("h1", out)      # 幂等
    assert again.repaired_done == [] and again.n_done == 1


def test_result_without_required_hidden_and_not_done_is_refused(tmp_path):
    out = _seed_raw(tmp_path, "h1", [_h1_row("e0__greedy_0")], hidden_for=[], done=[])
    with pytest.raises(ValueError, match="缺所需 hidden"):
        prod.audit_resume_state("h1", out)


# ── run fingerprint：首段创建，后续段只读比较 ────────────────────────────────
def _fp(arm: str, **over) -> dict:
    fp = {"model": {"model_hash": "m"}, "tokenizer": {"tokenizer_hash": "t"},
          "hidden_index": {"block_index_zero_based": 19},
          "generation_params": {"K": 6, "temperature": 0.7, "top_p": 0.95, "seed": 4242,
                                "max_new_tokens": 384 if arm == "h1" else 16}}
    fp.update(over)
    return fp


def test_run_fingerprint_is_created_once_then_only_compared(tmp_path):
    path = tmp_path / "h1_run_fingerprint.json"
    first = prod.bind_run_fingerprint(path, _fp("h1"), "h1", cfg=V3AnalysisConfig())
    assert first["created"] is True
    before = path.read_bytes()
    second = prod.bind_run_fingerprint(path, _fp("h1"), "h1", cfg=V3AnalysisConfig())
    assert second["created"] is False and second["matches_previous_segment"] is True
    assert path.read_bytes() == before, "后续段不得覆盖重写首段指纹"


@pytest.mark.parametrize("patch", [
    {"model": {"model_hash": "other"}},
    {"tokenizer": {"tokenizer_hash": "other"}},
    {"hidden_index": {"block_index_zero_based": 15}},
])
def test_fingerprint_drift_across_segments_is_refused(tmp_path, patch):
    path = tmp_path / "h1_run_fingerprint.json"
    prod.bind_run_fingerprint(path, _fp("h1"), "h1", cfg=V3AnalysisConfig())
    with pytest.raises(ValueError, match="与首段不一致"):
        prod.bind_run_fingerprint(path, _fp("h1", **patch), "h1", cfg=V3AnalysisConfig())


def test_fingerprint_seed_and_max_new_tokens_drift_are_refused(tmp_path):
    path = tmp_path / "h1_run_fingerprint.json"
    prod.bind_run_fingerprint(path, _fp("h1"), "h1", cfg=V3AnalysisConfig())
    drifted = _fp("h1")
    drifted["generation_params"] = {**drifted["generation_params"], "seed": 999}
    with pytest.raises(ValueError, match="generation_params.seed"):
        prod.bind_run_fingerprint(path, drifted, "h1", cfg=V3AnalysisConfig())

    wrong = _fp("h1")
    wrong["generation_params"] = {**wrong["generation_params"], "max_new_tokens": 512}
    with pytest.raises(ValueError, match=r"max_new_tokens\[h1\]"):
        prod.bind_run_fingerprint(path, wrong, "h1", cfg=V3AnalysisConfig())


def test_check_fingerprint_accepts_the_frozen_per_arm_values():
    cfg = V3AnalysisConfig()
    for arm in ("h1", "mcq"):
        assert prod.check_fingerprint(arm, _fp(arm), _fp(arm), cfg=cfg)["ok"] is True


# ── 分段：只在 unit 边界停 ───────────────────────────────────────────────────
class _Fake:
    def __init__(self, arm="h1", hidden_dim=8):
        self.arm, self.hidden_dim = arm, hidden_dim

    def run_fingerprint(self, arm):
        return _fp(arm)

    def verify_hidden_index(self):
        return {"block_index_zero_based": 19, "hidden_states_tuple_index": 20,
                "numeric_match": True}

    def run_unit(self, unit, record):
        return {"example_id": record["example_id"], "prompt_hash": "p",
                "unit_hash": "u", "seed": 4242, "sample_index": unit["sample_index"],
                "completion": "T1059", "label": unit["sample_index"] % 2,
                "mentions": [{"normalized": "T1059", "family": "attack",
                              "label": "grounded", "emitted": True}],
                "id_token_positions_local": [0], "hidden_pooled_n_tokens": 1,
                "hidden": np.ones(self.hidden_dim, dtype=np.float32),
                **{k: 0.5 for k in prod.H1_FINALIZED_F5
                   if k not in ("f5_self_consistency_exact", "f5_id_agreement_rate")}}


def _records(n=4):
    return [{"example_id": f"e{i}", "group_id": f"g{i}", "family": "attack"}
            for i in range(n)]


def _run(out, records, cfg, **over):
    p = prod.paths_for(out, "h1")
    kw = dict(arm="h1", records=records, backend=_Fake(), out_dir=out, cfg=cfg,
              durable_fsync=True, results_path=p["raw_results"],
              hidden_dir=p["hidden_dir"], manifest_path=p["manifest"])
    kw.update(over)
    return g3.run_arm(**kw)


def test_max_new_units_stops_on_a_unit_boundary(tmp_path):
    cfg = _cfg()
    records = _records(4)                       # 4 × K=3 = 12 units
    run = _run(tmp_path / "p", records, cfg, max_new_units=5)
    assert run.n_generated_this_process == 5 and run.stopped_reason == "max_new_units"
    rows = [json.loads(l) for l in
            prod.paths_for(tmp_path / "p", "h1")["raw_results"].read_text(
                encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 5, "分段必须停在 unit 边界，不留半条"


def test_max_wall_seconds_stops_and_is_resumable(tmp_path):
    cfg = _cfg()
    records = _records(4)
    first = _run(tmp_path / "p", records, cfg, max_wall_seconds=-1)   # 立即触发
    assert first.stopped_reason == "max_wall_seconds"
    second = _run(tmp_path / "p", records, cfg)
    assert second.n_generated_this_process == 12 - first.n_generated_this_process


def test_resume_does_not_regenerate_and_prefix_is_byte_stable(tmp_path):
    import hashlib
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    _run(out, records, cfg, max_new_units=6)
    path = prod.paths_for(out, "h1")["raw_results"]
    prefix = hashlib.sha256(path.read_bytes()).hexdigest()

    run2 = _run(out, records, cfg)
    assert run2.n_generated_this_process == 6
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    assert hashlib.sha256(b"\n".join(lines[:6]) + b"\n").hexdigest() == prefix
    assert len([l for l in lines if l.strip()]) == 12


def test_durable_fsync_defaults_to_off_and_only_adds_the_two_extra_layers(tmp_path,
                                                                          monkeypatch):
    """默认 durable_fsync=False 保持 G3/G4 既有行为。

    ⚠️ **不能断言"完全不调 fsync"**：`WorkManifest._append` 本来就每条 fsync
    （Stage 0 既有且必要的行为 —— 没有它 resume 就是骗人的）。
    `durable_fsync` 额外覆盖的是 **result 行**与 **hidden** 两层。
    """
    seen = []
    real_save = g3._save_hidden
    monkeypatch.setattr(g3, "_save_hidden",
                        lambda path, vec, *, durable: (seen.append(durable),
                                                       real_save(path, vec, durable=durable))[1])

    counts = {}
    real_fsync = g3.os.fsync
    for label, durable in (("default", False), ("production", True)):
        calls = []
        monkeypatch.setattr(g3.os, "fsync", lambda fd, _c=calls: (_c.append(fd),
                                                                  real_fsync(fd))[1])
        g3.run_arm(arm="h1", records=_records(1), backend=_Fake(),
                   out_dir=tmp_path / label, cfg=_cfg(), durable_fsync=durable)
        counts[label] = len(calls)

    assert seen == [False, False, False, True, True, True], "hidden 层的 durable 标志须逐条透传"
    assert counts["production"] > counts["default"], (
        "durable_fsync=True 必须比默认路径多出 result 行的 fsync")


# ── finalization ─────────────────────────────────────────────────────────────
def test_finalize_refuses_before_the_arm_is_complete(tmp_path):
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    _run(out, records, cfg, max_new_units=6)
    with pytest.raises(ValueError, match="尚未跑完，不得 finalize"):
        prod.finalize_arm("h1", out, records, cfg=cfg, run_fingerprint=_fp("h1"))


def test_finalize_builds_records_and_persists_cross_sample_f5(tmp_path):
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    _run(out, records, cfg)
    report = prod.finalize_arm("h1", out, records, cfg=cfg, run_fingerprint=_fp("h1"))
    p = prod.paths_for(out, "h1")

    assert report["n_finalized_records"] == 12 and report["hidden_alignment_key"] == "trace_id"
    finalized = [json.loads(l) for l in p["records"].read_text(encoding="utf-8").splitlines()]
    assert all({"trace_id", "group_id", "population_role", "run_hash"} <= set(r)
               for r in finalized)
    assert all(r["population_role"] == "confirmatory" for r in finalized)
    assert all(r["f5_self_consistency_exact"] is not None for r in finalized)

    raw = [json.loads(l) for l in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    assert all("f5_self_consistency_exact" not in r for r in raw), "raw 永不回写"


def test_finalize_refuses_when_a_prompt_lacks_k_units(tmp_path):
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    _run(out, records, cfg)
    p = prod.paths_for(out, "h1")
    rows = [l for l in p["raw_results"].read_text(encoding="utf-8").splitlines() if l.strip()]
    p["raw_results"].write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="尚未跑完|unit 数 !="):
        prod.finalize_arm("h1", out, records, cfg=cfg, run_fingerprint=_fp("h1"))


def test_finalize_refuses_when_a_frozen_f5_column_is_missing(tmp_path):
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    _run(out, records, cfg)
    p = prod.paths_for(out, "h1")
    rows = [json.loads(l) for l in
            p["raw_results"].read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        row.pop("f5_id_token_entropy_mean", None)
    p["raw_results"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="f5_id_token_entropy_mean"):
        prod.finalize_arm("h1", out, records, cfg=cfg, run_fingerprint=_fp("h1"))


def test_h1_frozen_f5_list_is_the_full_fourteen():
    assert len(prod.H1_FINALIZED_F5) == 14
    assert "f5_self_consistency_exact" in prod.H1_FINALIZED_F5
    assert "f5_id_agreement_rate" in prod.H1_FINALIZED_F5


def test_mcq_finalized_f5_is_the_full_six():
    assert prod.MCQ_FINALIZED_F5 == (
        "f5_label_margin", "f5_label_entropy", "f5_full_entropy",
        "f5_letter_token_logprob", "f5_self_consistency_exact", "f5_letter_agreement_rate")


def test_mcq_population_provenance_is_recorded():
    """provenance 改为**双口径**：nominal 与 actual 必须同时出现。

    旧契约只有单一 `finalized_rows=1760` 与 `not_a_prereg_change`；
    发生 runtime technical attrition 后前者会与实际行数矛盾、后者与
    `protocol_deviation=true` 语义冲突，故两者都已移除。
    """
    p = prod.MCQ_POPULATION_PROVENANCE          # 无排除时的 nominal 视图
    assert p["nominal_questions"] == 1760
    assert p["nominal_finalized_rows"] == 1760
    assert p["technical_exclusions"] == 0
    assert p["retained_questions"] == 1760
    assert p["actual_finalized_rows"] == 1760
    assert p["retained_raw_units"] == 1760 * 6
    assert p["preserved_but_not_analyzed_raw_units"] == 0
    assert p["expected_total_raw_rows"] == 1760 * 6
    assert p["protocol_deviation"] is False
    assert p["classification"] == "no_deviation"
    assert p["hidden_alignment"].startswith("example_id")
    assert p["preregistration_files_modified"] is False
    assert "not_a_prereg_change" not in p
    assert "v2.3/v2.4" in p["inherited_from"]


def test_mcq_population_provenance_reflects_attrition():
    p = prod.mcq_population_provenance(nominal_questions=1760, technical_exclusions=1,
                                       k_traces=6)
    assert (p["nominal_questions"], p["nominal_finalized_rows"]) == (1760, 1760)
    assert (p["retained_questions"], p["actual_finalized_rows"]) == (1759, 1759)
    assert p["retained_raw_units"] == 1759 * 6 == 10554
    assert p["expected_total_raw_rows"] == 10554
    assert p["protocol_deviation"] is True
    assert p["classification"] == "runtime_technical_attrition"


# ── gate 未通过即拒绝 ────────────────────────────────────────────────────────
def _gate_dirs(tmp_path: Path, *, g3_passed=True, g4_passed=True, arms=(True, True)):
    g3d, g4d = tmp_path / "g3", tmp_path / "g4"
    g3d.mkdir(parents=True, exist_ok=True)
    g4d.mkdir(parents=True, exist_ok=True)
    (g3d / "g3_summary.json").write_text(json.dumps({"passed": g3_passed}), encoding="utf-8")
    (g3d / "h1_g3_smoke_report.json").write_text(
        json.dumps({"fingerprint": _fp("h1")}), encoding="utf-8")
    (g4d / "g4_summary.json").write_text(json.dumps(
        {"passed": g4_passed, "arms_passed": {"h1": arms[0], "mcq": arms[1]}}),
        encoding="utf-8")
    return g3d, g4d


def test_production_refuses_when_g3_not_passed(tmp_path):
    g3d, g4d = _gate_dirs(tmp_path, g3_passed=False)
    with pytest.raises(ValueError, match="G3 未通过"):
        prod.verify_production_inputs(
            samples_dir=tmp_path, g3_dir=g3d, g4_dir=g4d, prereg=REAL_PREREG,
            lock=REAL_LOCK, cfg=V3AnalysisConfig(), arm="h1")


def test_production_refuses_when_g4_not_passed(tmp_path):
    g3d, g4d = _gate_dirs(tmp_path, g4_passed=False)
    with pytest.raises(ValueError, match="G4 未通过"):
        prod.verify_production_inputs(
            samples_dir=tmp_path, g3_dir=g3d, g4_dir=g4d, prereg=REAL_PREREG,
            lock=REAL_LOCK, cfg=V3AnalysisConfig(), arm="h1")


def test_production_refuses_when_one_g4_arm_failed(tmp_path):
    g3d, g4d = _gate_dirs(tmp_path, arms=(True, False))
    with pytest.raises(ValueError, match="G4 两臂未全部通过"):
        prod.verify_production_inputs(
            samples_dir=tmp_path, g3_dir=g3d, g4_dir=g4d, prereg=REAL_PREREG,
            lock=REAL_LOCK, cfg=V3AnalysisConfig(), arm="h1")


def test_production_refuses_on_tampered_prereg(tmp_path):
    g3d, g4d = _gate_dirs(tmp_path)
    bad = tmp_path / "tampered.md"
    bad.write_bytes(REAL_PREREG.read_bytes() + b"\nsneaky\n")
    with pytest.raises(ValueError, match="G2 失败"):
        prod.verify_production_inputs(
            samples_dir=tmp_path, g3_dir=g3d, g4_dir=g4d, prereg=bad,
            lock=REAL_LOCK, cfg=V3AnalysisConfig(), arm="h1")


# ── stop_on_failure：正式生成不容忍静默失败 ─────────────────────────────────
class _FailingBackend(_Fake):
    """在第 `fail_at` 次 run_unit 调用时抛指定异常；记录总调用次数。"""

    def __init__(self, *, fail_at: int, exc: Exception, **kw):
        super().__init__(**kw)
        self.fail_at = fail_at
        self.exc = exc
        self.calls = 0

    def run_unit(self, unit, record):
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.exc
        return super().run_unit(unit, record)


def _oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")


def _nonfinite() -> Exception:
    return g3.gb.ForwardNonFiniteError("non-finite option logits",
                                       diagnostic={"reason": "forward_nonfinite"})


@pytest.mark.parametrize("exc_factory,kind", [
    (_oom, "RuntimeError"),
    (lambda: ValueError("boom"), "ValueError"),
])
def test_production_stops_at_the_first_failure_and_calls_backend_no_further(
        tmp_path, exc_factory, kind):
    """production：第一处失败即停，**后续 unit 的 backend 未被调用**。"""
    cfg = _cfg()
    records = _records(4)                       # 4 × K=3 = 12 units
    backend = _FailingBackend(fail_at=5, exc=exc_factory())
    p = prod.paths_for(tmp_path / "p", "h1")
    run = g3.run_arm(arm="h1", records=records, backend=backend, out_dir=tmp_path / "p",
                     cfg=cfg, durable_fsync=True, stop_on_failure=True,
                     results_path=p["raw_results"], hidden_dir=p["hidden_dir"],
                     manifest_path=p["manifest"])
    assert backend.calls == 5, "失败之后不得再调用 backend"
    assert run.stopped_reason == f"unit_failure:{kind}"
    assert run.n_generated_this_process == 4
    assert len(run.failures) == 1 and run.failures[0]["kind"] == kind


def test_production_stops_immediately_on_forward_nonfinite(tmp_path):
    cfg = _cfg()
    records = _records(4)
    backend = _FailingBackend(fail_at=3, exc=_nonfinite())
    p = prod.paths_for(tmp_path / "p", "h1")
    run = g3.run_arm(arm="h1", records=records, backend=backend, out_dir=tmp_path / "p",
                     cfg=cfg, durable_fsync=True, stop_on_failure=True,
                     results_path=p["raw_results"], hidden_dir=p["hidden_dir"],
                     manifest_path=p["manifest"])
    assert backend.calls == 3
    assert run.stopped_reason == "unit_failure:forward_nonfinite"


def test_g3_g4_default_keeps_going_on_failure(tmp_path):
    """G3/G4 默认 stop_on_failure=False：记录后继续，行为逐项不变。"""
    cfg = _cfg()
    records = _records(4)
    backend = _FailingBackend(fail_at=5, exc=_oom())
    run = g3.run_arm(arm="h1", records=records, backend=backend,
                     out_dir=tmp_path / "g3", cfg=cfg)
    assert backend.calls == 12, "默认路径必须把全部 unit 试完"
    assert run.stopped_reason is None
    assert run.n_generated_this_process == 11 and len(run.failures) == 1


def test_failed_unit_is_not_terminal_and_is_requeued_next_run(tmp_path):
    """普通 failed：**未标 done**，下一次经明确授权的运行从该 unit 重试。"""
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    p = prod.paths_for(out, "h1")
    common = dict(arm="h1", records=records, out_dir=out, cfg=cfg, durable_fsync=True,
                  stop_on_failure=True, results_path=p["raw_results"],
                  hidden_dir=p["hidden_dir"], manifest_path=p["manifest"])
    failing = _FailingBackend(fail_at=5, exc=_oom())
    g3.run_arm(backend=failing, **common)

    man = WorkManifest(p["manifest"])
    status = man.current_status()
    failed_uid = [u for u, st in status.items() if st == "failed"]
    assert len(failed_uid) == 1
    assert status[failed_uid[0]] != "done"
    assert failed_uid[0] in man.pending_units(status), "failed 不是终态，须重新入队"

    second = g3.run_arm(backend=_Fake(), **common)      # 下一次授权运行
    assert failed_uid[0] in {r["unit_id"] for r in second.rows}, "应从该 unit 重试"
    assert second.n_generated_this_process == 8


def test_forward_nonfinite_unit_is_terminal_and_not_retried(tmp_path):
    """forward_nonfinite：标 **excluded**（终态），下一次运行**不得**重试。"""
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    p = prod.paths_for(out, "h1")
    common = dict(arm="h1", records=records, out_dir=out, cfg=cfg, durable_fsync=True,
                  stop_on_failure=True, results_path=p["raw_results"],
                  hidden_dir=p["hidden_dir"], manifest_path=p["manifest"])
    g3.run_arm(backend=_FailingBackend(fail_at=3, exc=_nonfinite()), **common)

    man = WorkManifest(p["manifest"])
    status = man.current_status()
    excluded = [u for u, st in status.items() if st == "excluded"]
    assert len(excluded) == 1
    assert excluded[0] not in man.pending_units(status), "excluded 是终态，不得重新入队"

    second = g3.run_arm(backend=_Fake(), **common)
    assert excluded[0] not in {r["unit_id"] for r in second.rows}, "不得重试 excluded unit"


def test_prefix_of_successfully_written_rows_is_unchanged_after_a_failure(tmp_path):
    import hashlib
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    p = prod.paths_for(out, "h1")
    common = dict(arm="h1", records=records, out_dir=out, cfg=cfg, durable_fsync=True,
                  stop_on_failure=True, results_path=p["raw_results"],
                  hidden_dir=p["hidden_dir"], manifest_path=p["manifest"])
    g3.run_arm(backend=_FailingBackend(fail_at=5, exc=_oom()), **common)
    raw = p["raw_results"].read_bytes()
    assert len([l for l in raw.split(b"\n") if l.strip()]) == 4
    before = hashlib.sha256(raw).hexdigest()

    g3.run_arm(backend=_Fake(), **common)
    after = p["raw_results"].read_bytes()
    off = -1
    for _ in range(4):
        off = after.index(b"\n", off + 1)
    assert hashlib.sha256(after[:off + 1]).hexdigest() == before, "已落盘前缀必须逐字节不变"


def test_excluded_unit_without_a_result_row_does_not_break_resume(tmp_path):
    """excluded 是"有结论、无数据"的终态：续跑读回时不得把它当成 done 缺行而报错。

    这是补 stop_on_failure 时由测试逮到的真缺陷 —— 先前 run_arm 对任何"不在 todo
    却没有结果行"的 unit 一律抛错，而 excluded 本来就没有行。
    """
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    p = prod.paths_for(out, "h1")
    common = dict(arm="h1", records=records, out_dir=out, cfg=cfg, durable_fsync=True,
                  stop_on_failure=True, results_path=p["raw_results"],
                  hidden_dir=p["hidden_dir"], manifest_path=p["manifest"])
    g3.run_arm(backend=_FailingBackend(fail_at=3, exc=_nonfinite()), **common)

    second = g3.run_arm(backend=_Fake(), **common)          # 不得抛错
    assert second.n_generated_this_process == 9             # 12 − 2 已成功 − 1 excluded
    assert len(second.rows) == 11                           # excluded 那条没有行
    assert prod.audit_resume_state("h1", out, repair=False).n_rows == 11


def test_audit_resume_state_tolerates_excluded_units(tmp_path):
    cfg = _cfg()
    records = _records(4)
    out = tmp_path / "p"
    p = prod.paths_for(out, "h1")
    g3.run_arm(arm="h1", records=records, backend=_FailingBackend(fail_at=3, exc=_nonfinite()),
               out_dir=out, cfg=cfg, durable_fsync=True, stop_on_failure=True,
               results_path=p["raw_results"], hidden_dir=p["hidden_dir"],
               manifest_path=p["manifest"])
    audit = prod.audit_resume_state("h1", out, repair=False)
    assert audit.n_rows == 2 and audit.orphan_hidden == []
