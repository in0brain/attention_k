"""Stage 1 G3 smoke 的回归测试。

**不加载模型、不读 `data/**`。** 全部用假后端与合成名单；
只有两项例外读仓库里真实的 prereg/lock（G2 接线），不读任何数据。
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
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.evaluation.v3_freeze import V3_LOCK_VERSION, sha256_text_lf  # noqa: E402
from recover_attention.evaluation.v3_sample_builder import (  # noqa: E402
    MANIFEST_NAME,
    chain_digest,
)


def _cfg(**over) -> V3AnalysisConfig:
    base = dict(mode="dry_run", n_g3_per_arm=2, k_traces=3)
    base.update(over)
    return V3AnalysisConfig(**base)


# ── 合成的 samples 目录 ──────────────────────────────────────────────────────
def _h1_row(i: int, gate: str | None = "G3") -> dict:
    row = {"example_id": f"e{i}", "route": "recall", "family": "attack",
           "question_text": f"question {i}?", "source_entry_id": f"T{1000 + i}",
           "group_id": f"g{i}", "entry_status": "active",
           "g5_prompt_hash": f"h{i}", "label_space": "open_identifier"}
    if gate:
        row.update({"gate": gate, "gate_only": True, "previously_burned": True})
    return row


def _mcq_row(i: int, gate: str | None = "G3") -> dict:
    row = {"question_hash": f"q{i}", "source": "CyberMetric-10000-v1.json",
           "original_index": i, "question": f"mcq question {i}?",
           "answers": {"A": "a", "B": "b", "C": "c", "D": "d"},
           "gold_original_label": "A", "rank": i}
    if gate:
        row.update({"gate": gate, "gate_only": True, "previously_burned": False})
    return row


@pytest.fixture()
def samples(tmp_path: Path):
    """两臂各 2 条 G3；正式集与 G4 也建出来，用于验证"chain hash 六份都查"。"""
    directory = tmp_path / "samples"
    directory.mkdir()
    lists = {
        "h1_primary": ("h1_primary_entries.jsonl", [_h1_row(i, None) for i in range(5)]),
        "h1_G3": ("h1_g3_entries.jsonl", [_h1_row(100 + i) for i in range(2)]),
        "h1_G4": ("h1_g4_entries.jsonl", [_h1_row(200 + i, "G4") for i in range(2)]),
        "mcq_primary": ("mcq_primary_questions.jsonl", [_mcq_row(i, None) for i in range(5)]),
        "mcq_G3": ("mcq_g3_questions.jsonl", [_mcq_row(100 + i) for i in range(2)]),
        "mcq_G4": ("mcq_g4_questions.jsonl", [_mcq_row(200 + i, "G4") for i in range(2)]),
    }
    key_of = {"h1": "source_entry_id", "mcq": "question_hash"}
    chain = {}
    for name, (filename, rows) in lists.items():
        (directory / filename).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        chain[name] = chain_digest(r[key_of[name.split("_")[0]]] for r in rows)

    prereg = _REPO_ROOT / "docs" / "paper" / "preregistration_v3.md"
    (directory / MANIFEST_NAME).write_text(json.dumps({
        "prereg": {"sha256": sha256_text_lf(prereg), "lock_version": V3_LOCK_VERSION},
        "chain_sha256": chain,
    }, ensure_ascii=False), encoding="utf-8")
    return directory


REAL = {"prereg": _REPO_ROOT / "docs" / "paper" / "preregistration_v3.md",
        "lock": _REPO_ROOT / "docs" / "paper" / "preregistration_v3.lock"}


def _verify(samples_dir: Path, cfg=None):
    return g3.verify_inputs(samples_dir=samples_dir, cfg=cfg or _cfg(), **REAL)


# ── 判据必须跑前写死 ─────────────────────────────────────────────────────────
def test_criteria_are_module_constants_frozen_before_any_run():
    c = g3.CRITERIA
    assert (c.n_prompts, c.traces_per_prompt, c.n_traces) == (20, 6, 120)
    assert c.hidden_dim == 3584
    assert c.h1_min_eligible_rate == 0.80
    assert c.mcq_max_parse_failure_rate == 0.10
    assert set(c.required_unit_fields) == {"prompt_hash", "unit_hash", "seed", "sample_index"}
    assert c.as_dict()["frozen_before_run"] is True


def test_criteria_dataclass_is_immutable():
    with pytest.raises(Exception):
        g3.CRITERIA.h1_min_eligible_rate = 0.1     # type: ignore[misc]


# ── 启动前校验 ───────────────────────────────────────────────────────────────
def test_verify_inputs_passes_and_reports_all_six_chain_hashes(samples):
    out = _verify(samples)
    assert out["g2"]["match"] is True and out["g2"]["locked_version"] == V3_LOCK_VERSION
    assert out["n_lists_verified"] == 6
    assert out["g3_counts"] == {"h1": 2, "mcq": 2}
    assert out["primary_lists_read"] is False and out["primary_lists_written"] is False


def test_verify_inputs_only_hands_back_gate_rows_not_primary(samples):
    """结构性保证：返回值里根本没有正式集的行，调用方无从据此读到它。"""
    out = _verify(samples)
    assert set(out["_gate_rows"]) == {"h1", "mcq"}
    for rows in out["_gate_rows"].values():
        assert all(r["gate"] == "G3" for r in rows)


def test_g2_failure_stops_before_anything_else(samples, tmp_path):
    bad = tmp_path / "tampered.md"
    bad.write_bytes(REAL["prereg"].read_bytes() + b"\nsneaky edit\n")
    with pytest.raises(ValueError, match="G2 失败"):
        g3.verify_inputs(samples_dir=samples, prereg=bad, lock=REAL["lock"], cfg=_cfg())


def test_manifest_prereg_hash_mismatch_is_rejected(samples):
    path = samples / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prereg"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prereg sha256 与当前不一致"):
        _verify(samples)


def test_chain_hash_mismatch_is_rejected(samples):
    rows = [_h1_row(999)]
    (samples / "h1_g3_entries.jsonl").write_text(
        json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chain hash 与 manifest 不符"):
        _verify(samples)


def test_wrong_g3_count_is_rejected(samples):
    with pytest.raises(ValueError, match=r"G3 名单 2 条 != 20"):
        _verify(samples, cfg=_cfg(n_g3_per_arm=20))


@pytest.mark.parametrize("field_name,value,pattern", [
    ("gate_only", False, "gate_only != true"),
    ("gate", "G4", "gate != 'G3'"),
    ("previously_burned", False, "previously_burned=true"),
])
def test_gate_flags_are_enforced(samples, field_name, value, pattern):
    rows = [json.loads(l) for l in
            (samples / "h1_g3_entries.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0][field_name] = value
    (samples / "h1_g3_entries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    payload = json.loads((samples / MANIFEST_NAME).read_text(encoding="utf-8"))
    payload["chain_sha256"]["h1_G3"] = chain_digest(r["source_entry_id"] for r in rows)
    (samples / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        _verify(samples)


# ── 假后端与运行 ─────────────────────────────────────────────────────────────
class FakeBackend:
    def __init__(self, arm, *, hidden_dim=3584, drop_hidden_for=(), non_finite_for=(),
                 raise_for=(), omit_field=None, ineligible_for=(), block_index=19):
        self.arm = arm
        self.hidden_dim = hidden_dim
        self.drop_hidden_for = set(drop_hidden_for)
        self.non_finite_for = set(non_finite_for)
        self.raise_for = set(raise_for)
        self.omit_field = omit_field
        self.ineligible_for = set(ineligible_for)
        self.block_index = block_index

    def run_fingerprint(self, arm):
        return {"arm": arm, "model": {"model_hash": "m" * 8, "backend": dict(g3.gb.BACKEND_8BIT)},
                "tokenizer": {"tokenizer_hash": "t" * 8},
                "hidden_index": {"block_index_zero_based": 19, "hidden_states_tuple_index": 20},
                "generation_params": {"K": 3, "temperature": 0.7, "top_p": 0.95, "seed": 4242,
                                      "max_new_tokens": 384 if arm == g3.ARM_H1 else 16},
                "run_hash": f"rh_{arm}"}

    def verify_hidden_index(self):
        return {"block_index_zero_based": self.block_index,
                "hidden_states_tuple_index": self.block_index + 1,
                "numeric_match": True, "max_abs_diff": 0.0}

    def run_unit(self, unit, record):
        uid = unit["unit_id"]
        if uid in self.raise_for:
            raise RuntimeError("boom")
        row = {"example_id": record["example_id"], "prompt_hash": f"p_{uid}",
               "unit_hash": f"u_{uid}", "seed": 4242, "sample_index": unit["sample_index"]}
        if self.omit_field:
            row.pop(self.omit_field, None)
        vec = np.ones(self.hidden_dim, dtype=np.float32)
        if uid in self.non_finite_for:
            vec[0] = np.nan
        if uid not in self.drop_hidden_for:
            row["hidden"] = vec
        if self.arm == g3.ARM_H1:
            eligible = uid not in self.ineligible_for
            row.update({"completion": "T1059", "refusal": False,
                        "mentions": [{"normalized": "T1059", "start": 0, "end": 5,
                                      "emitted": True, "token_indices": [0]}],
                        "id_token_positions_local": [0] if eligible else [],
                        "hidden_pooled_n_tokens": 1 if eligible else 0})
        else:
            eligible = uid not in self.ineligible_for
            row.update({"parsed_label": "A", "eligible_for_primary": eligible,
                        "f5_label_margin": 1.0, "f5_label_entropy": 0.5,
                        "f5_full_entropy": 2.0, "f5_letter_token_logprob": -0.1})
        return row


def _records(arm, n=2):
    if arm == g3.ARM_H1:
        return [_h1_row(i) | {"example_id": f"e{i}"} for i in range(n)]
    return [{"example_id": f"m{i}", "rendered_prompt": f"p{i}"} for i in range(n)]


def _run(arm, tmp_path, backend=None, cfg=None, n=2):
    cfg = cfg or _cfg()
    recs = _records(arm, n)
    return recs, g3.run_arm(arm=arm, records=recs, backend=backend or FakeBackend(arm),
                            out_dir=tmp_path / "g3", cfg=cfg)


def test_units_are_one_greedy_plus_k_minus_one_sampled():
    units = g3.units_for(_records(g3.ARM_H1, 2), 6)
    assert len(units) == 12
    per = [u for u in units if u["example_id"] == "e0"]
    assert [u["sample_type"] for u in per] == ["greedy"] + ["sampled"] * 5
    assert [u["sample_index"] for u in per] == [0, 1, 2, 3, 4, 5]


def test_clean_run_passes_all_checks(tmp_path):
    for arm in g3.ARMS:
        recs, run = _run(arm, tmp_path / arm)
        result = g3.evaluate_arm(run, n_records=len(recs),
                                 criteria=g3.G3Criteria(n_prompts=2, traces_per_prompt=3,
                                                        n_traces=6))
        assert result["passed"] is True, result["failed_checks"]


@pytest.mark.parametrize("arm", list(g3.ARMS))
@pytest.mark.parametrize("kwargs,expected", [
    ({"hidden_dim": 128}, "hidden_dim_ok"),
    ({"non_finite_for": ["e0__greedy_0", "m0__greedy_0"]}, "no_non_finite_values"),
    ({"raise_for": ["e0__greedy_0", "m0__greedy_0"]}, "no_unhandled_exception"),
    ({"omit_field": "prompt_hash"}, "unit_fields_present"),
    ({"block_index": 15}, "hidden_index_matches_frozen"),
])
def test_each_failure_mode_flips_passed_to_false(tmp_path, arm, kwargs, expected):
    recs, run = _run(arm, tmp_path / f"{arm}_{expected}", backend=FakeBackend(arm, **kwargs))
    result = g3.evaluate_arm(run, n_records=len(recs),
                             criteria=g3.G3Criteria(n_prompts=2, traces_per_prompt=3,
                                                    n_traces=6))
    assert result["passed"] is False
    assert expected in result["failed_checks"]


def test_h1_eligible_rate_threshold_is_applied(tmp_path):
    ineligible = [f"e0__sampled_{i}" for i in (1, 2)]      # 6 条里 2 条不合格 → 0.667 < 0.80
    recs, run = _run(g3.ARM_H1, tmp_path, backend=FakeBackend(g3.ARM_H1,
                                                              ineligible_for=ineligible))
    result = g3.evaluate_arm(run, n_records=len(recs),
                             criteria=g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6))
    assert result["checks"]["eligible_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert "eligible_rate_ok" in result["failed_checks"] and result["passed"] is False


def test_mcq_parse_failure_threshold_is_applied(tmp_path):
    recs, run = _run(g3.ARM_MCQ, tmp_path,
                     backend=FakeBackend(g3.ARM_MCQ, ineligible_for=["m0__greedy_0"]))
    result = g3.evaluate_arm(run, n_records=len(recs),
                             criteria=g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6))
    assert result["checks"]["parse_failure_rate"] == pytest.approx(1 / 6, abs=1e-3)
    assert "parse_failure_rate_ok" in result["failed_checks"]


def test_eligible_trace_without_hidden_is_caught(tmp_path):
    recs, run = _run(g3.ARM_H1, tmp_path,
                     backend=FakeBackend(g3.ARM_H1, drop_hidden_for=["e0__greedy_0"]))
    result = g3.evaluate_arm(run, n_records=len(recs),
                             criteria=g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6))
    assert "every_eligible_trace_has_hidden" in result["failed_checks"]


# ── resume ───────────────────────────────────────────────────────────────────
def test_resume_reads_back_completed_rows_so_the_verdict_is_unchanged(tmp_path):
    """断点续跑必须与一次跑完得到**同一份判定输入** —— 否则会在一切正常时判 false。"""
    crit = g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6)
    recs, first = _run(g3.ARM_H1, tmp_path)
    assert g3.evaluate_arm(first, n_records=len(recs), criteria=crit)["passed"] is True

    second = g3.run_arm(arm=g3.ARM_H1, records=recs, backend=FakeBackend(g3.ARM_H1),
                        out_dir=tmp_path / "g3", cfg=_cfg())
    assert len(second.rows) == 6 and len(second.hidden) == 6
    assert g3.evaluate_arm(second, n_records=len(recs), criteria=crit)["passed"] is True


def test_resume_refuses_when_manifest_says_done_but_the_row_is_missing(tmp_path):
    """把数据丢失伪装成'已完成'是 resume 最阴险的失败模式 —— 必须炸，不得静默跳过。"""
    recs, _ = _run(g3.ARM_H1, tmp_path)
    (tmp_path / "g3" / "h1_g3_unit_results.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="伪装成"):
        g3.run_arm(arm=g3.ARM_H1, records=recs, backend=FakeBackend(g3.ARM_H1),
                   out_dir=tmp_path / "g3", cfg=_cfg())


def test_manifest_is_append_only(tmp_path):
    _run(g3.ARM_H1, tmp_path)
    path = tmp_path / "g3" / "h1_g3_generated_manifest.jsonl"
    before = path.read_text(encoding="utf-8")
    _run(g3.ARM_H1, tmp_path)
    assert path.read_text(encoding="utf-8").startswith(before), "manifest 必须是 append-only"


# ── backend invariant ────────────────────────────────────────────────────────
def test_backend_invariant_ok_when_arms_match(tmp_path):
    runs = {arm: _run(arm, tmp_path / arm)[1] for arm in g3.ARMS}
    inv = g3.backend_invariant(runs)
    assert inv["ok"] is True and inv["mismatched_keys"] == []
    assert inv["_run_hashes_differ_by_arm_is_expected"] is True


@pytest.mark.parametrize("key,patch", [
    ("model", {"model_hash": "different"}),
    ("tokenizer", {"tokenizer_hash": "different"}),
    ("hidden_index", {"block_index_zero_based": 15}),
    ("generation_params.seed", {"seed": 999}),
])
def test_backend_invariant_catches_each_kind_of_drift(tmp_path, key, patch):
    runs = {arm: _run(arm, tmp_path / arm)[1] for arm in g3.ARMS}
    target = key.split(".")[0]
    runs[g3.ARM_MCQ].fingerprint[target] = {**runs[g3.ARM_MCQ].fingerprint[target], **patch}
    inv = g3.backend_invariant(runs)
    assert inv["ok"] is False and key in inv["mismatched_keys"]


def test_backend_invariant_catches_a_precision_change(tmp_path):
    """换精度（8-bit → 4-bit）必须被拦下 —— 这正是 §6 的 D 会被量化混淆的情形。"""
    runs = {arm: _run(arm, tmp_path / arm)[1] for arm in g3.ARMS}
    fp = runs[g3.ARM_MCQ].fingerprint
    fp["model"] = {**fp["model"], "backend": {**fp["model"]["backend"],
                                              "load_in_8bit": False, "load_in_4bit": True}}
    assert g3.backend_invariant(runs)["ok"] is False


# ── 报告 ─────────────────────────────────────────────────────────────────────
def test_report_contains_every_required_field(tmp_path, samples):
    verification = _verify(samples)
    verification.pop("_gate_rows")
    recs, run = _run(g3.ARM_H1, tmp_path)
    crit = g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6)
    report = g3.build_report(arm=g3.ARM_H1, run=run, verification=verification,
                             evaluation=g3.evaluate_arm(run, n_records=len(recs), criteria=crit),
                             cfg=_cfg(), n_records=len(recs))
    for key in ("prereg_sha256", "lock_version", "hash_mode", "sample_manifest_sha256",
                "fingerprint", "n_prompts", "n_traces", "gate_only", "performance_blind",
                "passed", "criteria", "did_not_compute"):
        assert key in report, key
    assert report["gate_only"] is True and report["performance_blind"] is True
    assert report["lock_version"] == V3_LOCK_VERSION
    assert report["prereg_sha256"] == sha256_text_lf(REAL["prereg"])


def test_report_is_performance_blind(tmp_path, samples):
    verification = _verify(samples)
    verification.pop("_gate_rows")
    recs, run = _run(g3.ARM_H1, tmp_path)
    crit = g3.G3Criteria(n_prompts=2, traces_per_prompt=3, n_traces=6)
    report = g3.build_report(arm=g3.ARM_H1, run=run, verification=verification,
                             evaluation=g3.evaluate_arm(run, n_records=len(recs), criteria=crit),
                             cfg=_cfg(), n_records=len(recs))
    # `did_not_compute` 与 `criteria` 的说明文本本来就要点名这些量 —— 扫描时排除它们，
    # 剩下的报告主体里一个都不许出现。
    body = {k: v for k, v in report.items() if k not in ("did_not_compute", "criteria")}
    blob = json.dumps(body, ensure_ascii=False).lower()
    for banned in ("auroc", "auprc", "delta_h", "ci_lo", "ci_hi", "ladder", "probe"):
        assert banned not in blob, banned
    assert "AUROC" in report["did_not_compute"]


def test_provenance_records_both_execution_level_seeds():
    p = V3AnalysisConfig().provenance()
    assert p["mcq_shuffle_seed"] == 42 and p["generation_seed"] == 4242
    assert p["h1_gate_selection_salt"].endswith("2026-08-08")


@pytest.mark.parametrize("field_name,bad", [("mcq_shuffle_seed", 7), ("generation_seed", 1)])
def test_execution_level_seeds_are_frozen(field_name, bad):
    with pytest.raises(ValueError, match="frozen field"):
        V3AnalysisConfig(mode="production", **{field_name: bad}).assert_production()


# ── pace guard ───────────────────────────────────────────────────────────────
class _SlowBackend(FakeBackend):
    def run_unit(self, unit, record):
        import time as _t
        _t.sleep(0.02)
        return super().run_unit(unit, record)


def test_pace_guard_stops_when_the_projection_exceeds_the_budget(tmp_path):
    """前 N 条实测后外推超预算 → 停下来报告，且已完成的 unit 全部落盘可续跑。"""
    recs = _records(g3.ARM_H1, 2)
    with pytest.raises(g3.PaceBudgetExceeded) as exc:
        g3.run_arm(arm=g3.ARM_H1, records=recs, backend=_SlowBackend(g3.ARM_H1),
                   out_dir=tmp_path / "g3", cfg=_cfg(), pace_check_after=2,
                   pace_budget_hours=1e-9)
    assert exc.value.pace["over_budget"] is True
    assert exc.value.pace["measured_on_units"] == 2
    results = (tmp_path / "g3" / "h1_g3_unit_results.jsonl").read_text(encoding="utf-8")
    assert len(results.strip().splitlines()) == 2, "中断前完成的 unit 必须已落盘"


def test_pace_guard_does_not_fire_within_budget(tmp_path):
    recs = _records(g3.ARM_H1, 2)
    run = g3.run_arm(arm=g3.ARM_H1, records=recs, backend=FakeBackend(g3.ARM_H1),
                     out_dir=tmp_path / "g3", cfg=_cfg(), pace_check_after=2,
                     pace_budget_hours=2.0)
    assert run.pace["over_budget"] is False and len(run.rows) == 6


# ── invariant 口径修正（2026-08-08）：max_new_tokens 按臂精确匹配，不是忽略 ────
def _runs_with_max_new(tmp_path, h1_value, mcq_value):
    runs = {arm: _run(arm, tmp_path / arm)[1] for arm in g3.ARMS}
    for arm, value in ((g3.ARM_H1, h1_value), (g3.ARM_MCQ, mcq_value)):
        fp = runs[arm].fingerprint
        fp["generation_params"] = {**fp["generation_params"], "max_new_tokens": value}
    return runs


def test_invariant_accepts_the_only_legal_max_new_tokens_combination(tmp_path):
    """H1=384 / MCQ=16 是冻结设计里唯一合法的组合 —— 不同不代表漂移。"""
    inv = g3.backend_invariant(_runs_with_max_new(tmp_path, 384, 16))
    assert inv["ok"] is True and inv["mismatched_keys"] == []
    by_arm = inv["arm_specific_generation_params"]["by_arm"]
    assert by_arm["h1"] == {"expected": 384, "actual": 384, "ok": True}
    assert by_arm["mcq"] == {"expected": 16, "actual": 16, "ok": True}


def test_invariant_rejects_h1_max_new_tokens_that_is_not_384(tmp_path):
    inv = g3.backend_invariant(_runs_with_max_new(tmp_path, 512, 16))
    assert inv["ok"] is False
    assert "generation_params.max_new_tokens[h1]" in inv["mismatched_keys"]
    assert inv["arm_specific_generation_params"]["by_arm"]["h1"]["ok"] is False


def test_invariant_rejects_mcq_max_new_tokens_that_is_not_16(tmp_path):
    inv = g3.backend_invariant(_runs_with_max_new(tmp_path, 384, 32))
    assert inv["ok"] is False
    assert "generation_params.max_new_tokens[mcq]" in inv["mismatched_keys"]


def test_invariant_rejects_swapped_max_new_tokens(tmp_path):
    """两臂的值对调 —— 集合相同但归属错了，必须拒。"""
    inv = g3.backend_invariant(_runs_with_max_new(tmp_path, 16, 384))
    assert inv["ok"] is False
    assert {"generation_params.max_new_tokens[h1]",
            "generation_params.max_new_tokens[mcq]"} <= set(inv["mismatched_keys"])


@pytest.mark.parametrize("param,value", [
    ("K", 99), ("temperature", 0.9), ("top_p", 0.5), ("seed", 999),
])
def test_invariant_still_rejects_drift_in_the_common_generation_params(tmp_path, param, value):
    """放宽只针对 max_new_tokens：采样协议参数的任何不一致仍然拒。"""
    runs = _runs_with_max_new(tmp_path, 384, 16)
    fp = runs[g3.ARM_MCQ].fingerprint
    fp["generation_params"] = {**fp["generation_params"], param: value}
    inv = g3.backend_invariant(runs)
    assert inv["ok"] is False and f"generation_params.{param}" in inv["mismatched_keys"]


def test_invariant_still_rejects_model_tokenizer_and_hidden_index_drift(tmp_path):
    for key, patch in (("model", {"model_hash": "x"}), ("tokenizer", {"tokenizer_hash": "x"}),
                       ("hidden_index", {"block_index_zero_based": 15})):
        runs = _runs_with_max_new(tmp_path / key, 384, 16)
        runs[g3.ARM_MCQ].fingerprint[key] = {**runs[g3.ARM_MCQ].fingerprint[key], **patch}
        inv = g3.backend_invariant(runs)
        assert inv["ok"] is False and key in inv["mismatched_keys"]


def test_max_new_tokens_by_arm_is_frozen_in_config():
    c = V3AnalysisConfig()
    assert (c.h1_max_new_tokens, c.mcq_max_new_tokens) == (384, 16)
    assert c.provenance()["max_new_tokens_by_arm"] == {"h1": 384, "mcq": 16}
    for field_name, bad in (("h1_max_new_tokens", 512), ("mcq_max_new_tokens", 32)):
        with pytest.raises(ValueError, match="frozen field"):
            V3AnalysisConfig(mode="production", **{field_name: bad}).assert_production()
