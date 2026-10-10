"""F5 数值完整性审计的单测。合成 fixture，**不加载模型、不碰真实产物**。

覆盖研究者逐条裁定的边界：
  * 主分母 = analyzable；finalized 只作 secondary，两者不得混用
  * 恒等式用**预先写死**的容差，不用浮点相等
  * ratio > 1 只记红旗，脚本**不**裁定可用性
  * perplexity is None 的两因分别归因
  * ⑤⑥ 只声称"没有有限值进入聚合"，**不**声称"原始值全部非有限"
  * rank 的**两种**退化：target 非有限 → 精确写 1；row 含非有限 competitor → 低估
  * 组 II 长度分层的确定性、含首尾、去重、补位数、选择集合 SHA
  * 安全边界：硬门 / 独占创建 / os.link 原子发布 / 只读不留产物
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import h1_f5_features as f5f  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

CFG = V3AnalysisConfig()


def load_audit():
    spec = importlib.util.spec_from_file_location(
        "_f5_audit", _REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def aud(tmp_path, monkeypatch):
    mod = load_audit()
    prod = tmp_path / "production"
    audit = tmp_path / "audit"
    prod.mkdir()
    audit.mkdir()
    monkeypatch.setattr(mod, "PROD", prod)
    monkeypatch.setattr(mod, "AUDIT", audit)
    monkeypatch.setattr(mod, "AUDIT_OUT", audit / "f5_numeric_integrity_audit.json")
    monkeypatch.setattr(mod, "TELEMETRY_OUT", audit / "f5_audit_gpu_telemetry.csv")
    return mod


# ------------------------------------------------------------- fixtures
def row(uid="ex_a", *, n_id=2, count=10, ratio=None, lengthnorm=-1.5,
        ppl=None, id_mean=-1.2, id_min=-2.0, entropy=0.5, rank=7,
        completion=None, positions=None):
    pos = positions if positions is not None else list(range(n_id))
    if ratio is None and count:
        ratio = len(pos) / count
    if ppl is None and lengthnorm is not None and lengthnorm < 700:
        ppl = math.exp(-lengthnorm)
    return {
        "example_id": uid, "sample_type": "greedy", "sample_index": 0,
        "prompt_text": "q", "completion": completion or " ".join(f"w{i}" for i in range(20)),
        "id_token_positions_local": pos,
        "f5_id_token_count": len(pos),
        "f5_completion_token_count": count,
        "f5_lengthnorm_logprob": lengthnorm,
        "f5_completion_perplexity": ppl,
        "f5_id_token_ratio": ratio,
        "f5_id_logprob_mean": id_mean,
        "f5_id_logprob_min": id_min,
        "f5_id_token_entropy_mean": entropy,
        "f5_first_id_token_rank": rank,
    }


# ------------------------------------------------------------ 分母作用域
def test_primary_and_secondary_denominators_are_reported_separately(aud):
    rows = [row(f"ex_{i}") for i in range(5)]
    out = aud.audit_1a(rows, CFG)
    assert out["denominator"] == 5
    assert out["evidence_class"] == "1a_frozen_record_arithmetic"
    assert "纯冻结历史事实" in out["evidence_caliber"]


def test_scope_block_names_analyzable_as_primary(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    assert "primary_denominator" in src
    assert "n_finalized_secondary_accounting_only" in src
    assert "不得混用" in src


# ------------------------------------------------------------- 恒等式容差
def test_ratio_identity_uses_declared_tolerance_not_float_equality(aud):
    exact = row("ex_ok", n_id=2, count=10)
    # 制造一个 1e-12 相对量级的扰动：浮点相等会判失败，isclose(1e-9) 应通过
    nudged = dict(exact)
    nudged["f5_id_token_ratio"] = exact["f5_id_token_ratio"] * (1 + 1e-12)
    out = aud.audit_1a([exact, nudged], CFG)["identity_ratio_times_count_eq_id_count"]
    assert out["pass"] == 2 and out["fail"] == 0
    assert aud.RATIO_REL_TOL == 1e-9


def test_ratio_identity_detects_real_violation(aud):
    bad = row("ex_bad", n_id=2, count=10)
    bad["f5_id_token_ratio"] = 0.9                      # 0.9*10 = 9 != 2
    out = aud.audit_1a([bad], CFG)["identity_ratio_times_count_eq_id_count"]
    assert out["fail"] == 1 and out["fail_unit_ids_first_50"] == ["ex_bad__greedy_0"]


def test_ratio_identity_skips_rows_with_missing_values(aud):
    r = row("ex_none")
    r["f5_id_token_ratio"] = None
    out = aud.audit_1a([r], CFG)["identity_ratio_times_count_eq_id_count"]
    assert out["not_evaluable_missing_value"] == 1 and out["pass"] == 0 and out["fail"] == 0


def test_perplexity_identity_compares_in_log_space(aud):
    r = row("ex_ppl", lengthnorm=-3.0)
    out = aud.audit_1a([r], CFG)["identity_log_perplexity_eq_neg_lengthnorm"]
    assert out["pass"] == 1
    assert "取对数" in aud.audit_1a([r], CFG)["tolerances"]["note"]


def test_perplexity_identity_detects_violation(aud):
    r = row("ex_pplbad", lengthnorm=-3.0)
    r["f5_completion_perplexity"] = 1.0                 # log(1)=0 != 3.0
    out = aud.audit_1a([r], CFG)["identity_log_perplexity_eq_neg_lengthnorm"]
    assert out["fail"] == 1


def test_id_token_count_identity_is_exact_int_comparison(aud):
    good = row("ex_g", positions=[0, 1, 2])
    bad = row("ex_b", positions=[0, 1, 2])
    bad["f5_id_token_count"] = 3.0                      # float 不算精确 int 相等
    out = aud.audit_1a([good, bad], CFG)["identity_id_token_count_eq_n_positions"]
    assert out["pass"] == 1 and out["fail"] == 1
    assert out["comparison"] == "exact int equality"


# ------------------------------------------------------------------ 红旗
def test_ratio_gt_1_is_a_red_flag_not_a_verdict(aud):
    r = row("ex_over", n_id=8, count=4)                 # ratio = 2.0
    out = aud.audit_1a([r], CFG)["red_flag_ratio_gt_1"]
    assert out["count"] == 1
    assert out["verdict"].startswith("NOT_DECIDED_BY_SCRIPT")
    assert "研究者裁决" in out["verdict"]


def test_script_never_declares_features_unusable(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    assert "decides_feature_usability" in src
    for banned in ("feature_unusable", "not_usable_for_confirmatory", "drop_from_b4"):
        assert banned not in src


# --------------------------------------------------- 聚合量 None 的措辞
def test_none_aggregate_claim_is_bounded_to_what_the_record_proves(aud):
    r = row("ex_nomean")
    r["f5_id_logprob_mean"] = None
    out = aud.audit_1a([r], CFG)["no_finite_identifier_logprob_entered_aggregate"]
    assert out["count"] == 1
    assert "没有有限的 identifier-token 数值进入聚合" in out["provable_from_frozen_record"]
    assert "全部非有限" in out["NOT_provable_from_frozen_record"]
    assert "缺键" in out["NOT_provable_from_frozen_record"]


def test_missing_key_and_nonfinite_are_indistinguishable_in_frozen_aggregate():
    """`.get()` 缺键与非有限值经 finite_float 后落到**同一**分支——这是措辞收紧的根据。"""
    missing = f5f.mention_logprob_features({}, {}, {}, [0, 1])
    nonfin = f5f.mention_logprob_features(
        {0: float("nan"), 1: float("nan")},
        {0: float("nan"), 1: float("nan")}, {0: 3}, [0, 1])
    assert missing["f5_id_logprob_mean"] is None
    assert nonfin["f5_id_logprob_mean"] is None         # 冻结值里完全同形


def test_observable_lower_bound_caveat_present(aud):
    out = aud.audit_1a([row()], CFG)
    assert "下界" in out["observable_lower_bound_caveat"]


# --------------------------------------------------- perplexity None 两因
def test_perplexity_none_attribution_splits_two_causes(aud):
    a = row("ex_a")
    a["f5_lengthnorm_logprob"] = None
    a["f5_completion_perplexity"] = None
    b = row("ex_b")
    b["f5_lengthnorm_logprob"] = 900.0                  # >= 700 保护条件
    b["f5_completion_perplexity"] = None
    out = aud.audit_1a([a, b], CFG)["perplexity_none_attribution"]
    assert out["cause_a_lengthnorm_is_none_no_finite_logprob"] == 1
    assert out["cause_b_exponent_guard_lengthnorm_ge_700"] == 1
    assert out["cause_c_unexplained"] == 0
    assert out["guard_threshold"] == aud.PPL_EXP_GUARD


# ------------------------------------------------------------- rank 退化
def test_rank_degeneracy_a_nan_target_yields_exactly_one():
    """target logprob 为 NaN → 所有比较 False → rank 精确写成 1。"""
    row_vals = np.array([-1.0, -2.0, float("nan"), -0.5], dtype=np.float64)
    tid = 2
    rank = int((row_vals > row_vals[tid]).sum()) + 1
    assert rank == 1                                    # 最自信的取值


def test_rank_degeneracy_b_nonfinite_competitor_underestimates_rank():
    """target 有限但 competitor 含 NaN → competitor 不计入 → rank 被低估。"""
    clean = np.array([-1.0, -2.0, -0.1, -0.2], dtype=np.float64)
    tid = 1
    true_rank = int((clean > clean[tid]).sum()) + 1     # 3 个更大 -> 4
    dirty = clean.copy()
    dirty[0] = float("nan")                             # 一个更大的 competitor 变 NaN
    got = int((dirty > dirty[tid]).sum()) + 1
    assert true_rank == 4 and got == 3                  # 被低估


def test_rank_validity_condition_and_direction_claims(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    assert "全有限**时才可称数值有效" in src
    assert "更小、更像高排名" in src
    assert "不得**声称它必然把 fabrication classifier" in src


def test_rank_state_labels_are_the_three_declared_ones(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    for label in ("row_all_finite", "target_nonfinite", "target_finite_row_has_nonfinite"):
        assert label in src


# ------------------------------------------------------------- 行分类工具
def test_row_class_three_way(aud):
    assert aud._row_class([1.0, 2.0]) == "finite"
    assert aud._row_class([1.0, float("nan")]) == "partial_nonfinite"
    assert aud._row_class([float("nan"), float("inf")]) == "all_nonfinite"


def test_scalar_class_distinguishes_inf_sign(aud):
    assert aud._scalar_class(1.0) == "finite"
    assert aud._scalar_class(float("nan")) == "nan"
    assert aud._scalar_class(float("inf")) == "posinf"
    assert aud._scalar_class(float("-inf")) == "neginf"


# ------------------------------------------------------------ 组 II 选取
class FakeTok:
    def __call__(self, text, add_special_tokens=True, **kw):
        return {"input_ids": list(range(len(text.split())))}


def _corpus(n=500):
    out = []
    for i in range(n):
        length = 5 + (i * 7) % 200
        out.append(row(f"ex_{i:04d}", completion=" ".join(["w"] * length)))
    return out


def test_group_ii_is_deterministic_and_covers_full_length_range(aud):
    rows = _corpus()
    tok = FakeTok()
    a = aud.select_group_ii(rows, tok, set())
    b = aud.select_group_ii(list(reversed(rows)), tok, set())
    assert a["unit_ids"] == b["unit_ids"]                        # 与输入顺序无关
    assert a["n_selected"] == aud.GROUP_II_SIZE
    lens = sorted(len(r["completion"].split()) for r in rows)
    assert a["n_tok_min"] == lens[0] and a["n_tok_max"] == lens[-1]   # 含首尾


def test_group_ii_dedups_against_group_i_and_records_fill_count(aud):
    rows = _corpus()
    tok = FakeTok()
    plain = aud.select_group_ii(rows, tok, set())
    excluded = set(plain["unit_ids"][:10])
    out = aud.select_group_ii(rows, tok, excluded)
    assert not (set(out["unit_ids"]) & excluded)
    assert out["n_selected"] == aud.GROUP_II_SIZE
    assert out["n_fill_steps"] > 0
    assert out["excluded_group_i_size"] == len(excluded)


def test_group_ii_records_algorithm_version_and_selection_sha(aud):
    out = aud.select_group_ii(_corpus(), FakeTok(), set())
    assert out["selection_algorithm_version"] == aud.SELECTION_ALGORITHM_VERSION
    assert out["sort_key"] == "(n_tok, unit_id)"
    assert len(out["selection_set_sha256"]) == 64


# ------------------------------------------------------------- 安全边界
def test_hard_gate_refuses_existing_output(aud):
    aud.AUDIT_OUT.write_bytes(b"{}")
    with pytest.raises(aud.ScopeViolation, match="拒绝覆盖"):
        aud.assert_outputs_absent([aud.AUDIT_OUT])
    assert aud.AUDIT_OUT.read_bytes() == b"{}"


def test_publish_refuses_existing_target_and_stale_tmp(aud):
    aud.AUDIT_OUT.write_bytes(b"{}")
    with pytest.raises(aud.ScopeViolation, match="拒绝覆盖"):
        aud.publish_json_exclusive(aud.AUDIT_OUT, {"a": 1})
    aud.AUDIT_OUT.unlink()
    tmp = aud.AUDIT_OUT.with_suffix(".json.tmp")
    tmp.write_bytes(b"residue")
    with pytest.raises(aud.ScopeViolation, match="陈旧临时文件"):
        aud.publish_json_exclusive(aud.AUDIT_OUT, {"a": 1})
    assert tmp.read_bytes() == b"residue"


def test_publish_never_falls_back_to_os_replace(aud, monkeypatch):
    monkeypatch.setattr(aud.os, "link",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("no link")))
    monkeypatch.setattr(aud.os, "replace",
                        lambda *a, **kw: pytest.fail("不得退回 os.replace"))
    with pytest.raises(aud.ScopeViolation, match="fail-closed"):
        aud.publish_json_exclusive(aud.AUDIT_OUT, {"a": 1})
    assert not aud.AUDIT_OUT.exists()


def test_publish_succeeds_and_leaves_no_tmp(aud):
    aud.publish_json_exclusive(aud.AUDIT_OUT, {"a": 1})
    assert json.loads(aud.AUDIT_OUT.read_text("utf-8")) == {"a": 1}
    assert not aud.AUDIT_OUT.with_suffix(".json.tmp").exists()


def test_telemetry_is_exclusive_create(aud):
    aud.TELEMETRY_OUT.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        aud.start_telemetry(aud.TELEMETRY_OUT)
    assert aud.TELEMETRY_OUT.read_bytes() == b"keep me"


def test_scope_check_flags_leaked_production_artifacts(aud):
    seg = aud.PROD / "h1_e3_seg001_x.csv"
    seg.write_bytes(b"leak")
    with pytest.raises(aud.ScopeViolation):
        aud.assert_scope_clean()
    assert seg.read_bytes() == b"leak"


def test_audit_persists_no_completion_text_or_vocab_values(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    assert "不持久化" in src
    # diagnose_unit 的返回键里不得出现原文/逐 vocab 字段
    for banned in ('"completion"', '"logits"', '"log_softmax"', '"prompt"'):
        assert f"        {banned}:" not in src


def test_boundaries_block_declares_all_five_prohibitions(aud):
    src = (_REPO_ROOT / "scripts" / "audit_f5_numeric_integrity.py").read_text("utf-8")
    for key in ("performance_blind", "joined_with_label", "modified_frozen_records",
                "unblinded", "decides_remediation", "decides_feature_usability"):
        assert key in src


def test_abc_meanings_do_not_assert_original_runtime_state(aud):
    assert "不证明" in aud.ABC_MEANING["A"]
    assert "来源仍未确定" in aud.ABC_MEANING["B"]
    assert "停止作统一解释" in aud.ABC_MEANING["C"]


def test_summarize_group_reports_counts_and_unit_ids(aud):
    res = [
        {"unit_id": "u1", "abc_class": "A", "n_anomalous_rows": 0,
         "identifier_positions_on_anomalous_rows": [],
         "first_identifier_rank_state": "row_all_finite",
         "sequence_level_comparison": {"f5_id_token_ratio": {"match": True}},
         "identifier_position_comparison": {"f5_first_id_token_rank": {"match": False}}},
        {"unit_id": "u2", "abc_class": "B", "n_anomalous_rows": 3,
         "identifier_positions_on_anomalous_rows": [2],
         "first_identifier_rank_state": "target_nonfinite",
         "sequence_level_comparison": {"f5_id_token_ratio": {"match": True}},
         "identifier_position_comparison": {"f5_first_id_token_rank": {"match": False}}},
    ]
    out = aud.summarize_group("g", res)
    assert out["abc_counts"] == {"A": 1, "B": 1, "C": 0}
    assert out["abc_unit_ids"]["A"] == ["u1"] and out["abc_unit_ids"]["B"] == ["u2"]
    assert out["field_mismatch_counts_vs_frozen"]["f5_first_id_token_rank"] == 2
    assert out["n_units_with_any_anomalous_row"] == 1
    assert out["n_units_with_identifier_on_anomalous_row"] == 1


def test_field_scope_is_eight_plus_one_reference(aud):
    assert len(aud.FIELDS_SEQUENCE) == 4 and len(aud.FIELDS_IDENTIFIER) == 4
    assert len(aud.FIELDS_AUDITED) == 8
    assert aud.FIELD_REFERENCE == "f5_id_token_count"
    from recover_attention.conditional_increment import F5_FEATURE_NAMES
    assert len(F5_FEATURE_NAMES) == 14
    assert set(aud.FIELDS_AUDITED) < set(F5_FEATURE_NAMES)     # 真子集，不是 B4 本身
