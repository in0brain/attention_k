"""Stage 1 G4 的回归测试 —— 重点是 Stage 0 从未真正强制的两条门槛的**故障注入**。

不加载模型、不读 `data/**`；全部合成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_g4_quality as g4  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.evaluation.v3_g3_smoke import ArmRun  # noqa: E402

CFG = V3AnalysisConfig()
H1_MAX, MCQ_MAX = CFG.h1_max_new_tokens, CFG.mcq_max_new_tokens

CLEAN_H1 = "The identifier is T1059 and it refers to command interpreters."
CLEAN_MCQ = "A"
SINGLE_CHAR_RUN = "!" * 40
SUBSTRING_LOOP = ".49221" * 8      # 子串需 6-40 字符：.4922 只有 5 字符，不会触发规则
IDENTIFIER_LIST = "\n".join(["1. T1059 command interpreter",
                             "2. T1078 valid accounts",
                             "3. T1105 ingress tool transfer",
                             "4. T1027 obfuscated files",
                             "5. T1053 scheduled task"])


# ── 判据跑前写死 ─────────────────────────────────────────────────────────────
def test_criteria_are_frozen_module_constants():
    c = g4.CRITERIA
    assert (c.h1_min_eligible_rate, c.mcq_max_parse_failure_rate) == (0.80, 0.10)
    assert (c.max_degeneration_rate, c.max_touched_max_new_tokens_rate) == (0.10, 0.10)
    assert (c.n_prompts_per_arm, c.n_traces_per_arm) == (40, 240)
    assert c.as_dict()["frozen_before_run"] is True
    assert set(c.as_dict()["enforced_here_unlike_stage0"]) == {
        "max_degeneration_rate", "max_touched_max_new_tokens_rate"}
    with pytest.raises(Exception):
        g4.CRITERIA.max_degeneration_rate = 0.5     # type: ignore[misc]


def test_rule_sets_are_frozen_per_arm():
    assert g4.H1_DEGENERATION_RULES == ("single_char_run", "substring_loop")
    assert "truncated_failure" not in g4.H1_DEGENERATION_RULES
    assert g4.MCQ_DEGENERATION_RULES == ("single_char_run", "substring_loop",
                                         "truncated_failure")


# ── C1：H1 不使用 truncated_failure ──────────────────────────────────────────
def test_h1_long_completion_without_loops_is_not_degenerate():
    """H1 达到 0.9×384 但无字符/子串循环 → **不得**判 degeneration（C1 的核心）。"""
    long_text = " ".join(f"technique T{1000 + i} relates to step {i}." for i in range(200))
    verdict = g4.classify_row(g4.ARM_H1,
                              {"completion": long_text, "num_generated_tokens": 350},
                              max_new_tokens=H1_MAX)
    assert 350 >= 0.9 * H1_MAX                       # 确实落在旧 rule 3 会触发的区间
    assert verdict["degenerate_for_gate"] is False
    assert verdict["matched_rules"] == []
    assert verdict["touched_max_new_tokens"] is False   # 350 < 384，长度问题另算


def test_h1_touched_max_new_tokens_is_measured_separately():
    verdict = g4.classify_row(g4.ARM_H1,
                              {"completion": CLEAN_H1, "num_generated_tokens": H1_MAX},
                              max_new_tokens=H1_MAX)
    assert verdict["touched_max_new_tokens"] is True
    assert verdict["degenerate_for_gate"] is False, "撞上限本身不是退化"


def test_mcq_still_uses_truncated_failure_with_exact_tokens():
    """MCQ 保留三规则：解析失败 + 精确 token 数达 0.9×16 → truncated_failure。"""
    verdict = g4.classify_row(g4.ARM_MCQ,
                              {"raw_completion": "no letter here at all",
                               "num_generated_tokens": MCQ_MAX},
                              max_new_tokens=MCQ_MAX)
    assert "truncated_failure" in verdict["matched_rules"]
    assert verdict["degenerate_for_gate"] is True


def test_exact_token_count_is_required_not_char_approximation():
    with pytest.raises(ValueError, match="不接受字符长度近似"):
        g4.classify_row(g4.ARM_H1, {"completion": CLEAN_H1, "unit_id": "u"},
                        max_new_tokens=H1_MAX)


# ── C2：structured-list 豁免，透明报告 ───────────────────────────────────────
def test_identifier_list_only_substring_loop_is_exempted():
    verdict = g4.classify_h1_degeneration(IDENTIFIER_LIST + SUBSTRING_LOOP)
    assert "substring_loop" in verdict["matched_rules"]
    assert verdict["structured_list_like"] is True
    assert verdict["structured_list_exempted"] is True
    assert verdict["raw_degenerate"] is True
    assert verdict["degenerate_for_gate"] is False


def test_identifier_list_that_also_has_single_char_run_still_fails():
    """豁免只作用于"仅由列表结构触发的 substring_loop" —— 同时命中字符循环仍判退化。"""
    verdict = g4.classify_h1_degeneration(IDENTIFIER_LIST + SUBSTRING_LOOP + SINGLE_CHAR_RUN)
    assert "single_char_run" in verdict["matched_rules"]
    assert verdict["structured_list_like"] is True
    assert verdict["structured_list_exempted"] is False
    assert verdict["degenerate_for_gate"] is True


def test_substring_loop_without_list_structure_is_not_exempted():
    verdict = g4.classify_h1_degeneration(SUBSTRING_LOOP)
    assert verdict["structured_list_like"] is False
    assert verdict["degenerate_for_gate"] is True


def test_mcq_does_not_use_the_structured_list_exemption():
    verdict = g4.classify_mcq_degeneration(IDENTIFIER_LIST + SUBSTRING_LOOP,
                                           num_generated_tokens=4, max_new_tokens=MCQ_MAX)
    assert verdict["structured_list_exempted"] is False
    assert verdict["degenerate_for_gate"] is True


def test_report_can_reconstruct_raw_from_adjusted_plus_exempted():
    rows = ([{"completion": IDENTIFIER_LIST + SUBSTRING_LOOP, "num_generated_tokens": 50}
             for _ in range(3)]
            + [{"completion": SINGLE_CHAR_RUN, "num_generated_tokens": 50} for _ in range(2)]
            + [{"completion": CLEAN_H1, "num_generated_tokens": 20} for _ in range(5)])
    stats = g4.degeneration_stats(g4.ARM_H1, rows, max_new_tokens=H1_MAX)
    assert stats["raw_pattern_degeneration_count"] == 5
    assert stats["structured_list_false_positive_count"] == 3
    assert stats["adjusted_degeneration_count"] == 2
    assert (stats["raw_pattern_degeneration_count"]
            == stats["adjusted_degeneration_count"]
            + stats["structured_list_false_positive_count"])
    assert stats["gate_uses"] == "adjusted_degeneration_rate"
    # 注意：`"!"*40` 同时命中 substring_loop（6 个 "!" 重复 ≥5 次），故直方图是 3+2=5。
    # 这正是"豁免只作用于**仅**由列表触发的 substring_loop"这条规则要区分的情形：
    # 那 2 条同时有 single_char_run，不豁免。
    assert stats["matched_rules_histogram"]["substring_loop"] == 5
    assert stats["matched_rules_histogram"]["single_char_run"] == 2


# ── 故障注入：两条 Stage 0 从未强制的门槛 ────────────────────────────────────
def _h1_run(rows: list[dict], *, n_units: int | None = None) -> ArmRun:
    run = ArmRun(arm=g4.ARM_H1, n_units=n_units if n_units is not None else len(rows))
    for i, row in enumerate(rows):
        row.setdefault("unit_id", f"u{i}")
        row.setdefault("example_id", f"e{i % 40}")

        row.setdefault("prompt_hash", f"p{i}")
        row.setdefault("unit_hash", f"h{i}")
        row.setdefault("family", "attack" if i % 2 else "cwe")
        row.setdefault("id_token_positions_local", [0])
        row.setdefault("hidden_pooled_n_tokens", 1)
        row.setdefault("label", i % 2)                     # **二元整数**，非字符串
        row.setdefault("f5_id_logprob_mean", float(i) - 5.0)
        row.setdefault("f5_id_logprob_min", float(i) - 7.0)
        row.setdefault("f5_id_token_entropy_mean", float(i) * 0.1)
        row.setdefault("f5_completion_perplexity", 1.0 + i * 0.05)
        row.setdefault("f5_lengthnorm_logprob", -0.5 - i * 0.01)
        row.setdefault("group_id", f"g{i % 40}")
        run.rows.append(row)
        run.hidden[row["unit_id"]] = np.ones(3584, dtype=np.float32)
    run.fingerprint = {"generation_params": {"max_new_tokens": H1_MAX}}
    run.hidden_verification = {"numeric_match": True, "block_index_zero_based": 19,
                               "hidden_states_tuple_index": 20}
    return run


def _rows_with_degeneration(n_total: int, n_degenerate: int) -> list[dict]:
    # 每行必须是**独立 dict**：`[{...}] * n` 是同一对象的 n 个引用，
    # setdefault 只对第一份生效，全部行会静默塌成同一条。
    return [{"completion": SINGLE_CHAR_RUN if i < n_degenerate else CLEAN_H1,
             "num_generated_tokens": 20} for i in range(n_total)]


def _rows_with_touched(n_total: int, n_touched: int) -> list[dict]:
    return [{"completion": CLEAN_H1,
             "num_generated_tokens": H1_MAX if i < n_touched else 20}
            for i in range(n_total)]


def test_degeneration_rate_above_threshold_fails():
    """11/100 = 0.11 > 0.10 → 必须失败。Stage 0 这条从未接进 passed。"""
    run = _h1_run(_rows_with_degeneration(100, 11))
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                          criteria=g4.G4Criteria(n_traces_per_arm=100))
    assert result["stats"]["degeneration"]["adjusted_degeneration_rate"] == 0.11
    assert result["ok"] is False
    assert "degeneration_rate_ok" in result["failed_checks"]


def test_degeneration_rate_exactly_at_threshold_passes():
    """恰好 0.10 → 通过（边界是 ≤，不是 <）。"""
    run = _h1_run(_rows_with_degeneration(100, 10))
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                          criteria=g4.G4Criteria(n_traces_per_arm=100))
    assert result["stats"]["degeneration"]["adjusted_degeneration_rate"] == 0.10
    assert "degeneration_rate_ok" not in result["failed_checks"]


def test_touched_max_new_tokens_rate_above_threshold_fails():
    """11/100 撞上限 → 必须失败。Stage 0 这条连报都没有。"""
    run = _h1_run(_rows_with_touched(100, 11))
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                          criteria=g4.G4Criteria(n_traces_per_arm=100))
    assert result["stats"]["degeneration"]["touched_max_new_tokens_rate"] == 0.11
    assert result["ok"] is False
    assert "touched_max_new_tokens_rate_ok" in result["failed_checks"]


def test_touched_max_new_tokens_rate_exactly_at_threshold_passes():
    run = _h1_run(_rows_with_touched(100, 10))
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                          criteria=g4.G4Criteria(n_traces_per_arm=100))
    assert result["stats"]["degeneration"]["touched_max_new_tokens_rate"] == 0.10
    assert "touched_max_new_tokens_rate_ok" not in result["failed_checks"]


def test_margins_do_not_change_the_verdict():
    """余量字段只报不判：把余量算出来不改变任何 check 的布尔值。"""
    for n_degenerate in (0, 5, 10, 11):
        run = _h1_run(_rows_with_degeneration(100, n_degenerate))
        result = g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                              criteria=g4.G4Criteria(n_traces_per_arm=100))
        before = dict(result["checks"])
        margin = g4.margins(g4.ARM_H1, result)
        assert result["checks"] == before, "计算余量不得改动 checks"
        expected_ok = n_degenerate <= 10
        assert ("degeneration_rate_ok" not in result["failed_checks"]) is expected_ok
        assert margin["degeneration_margin"] == pytest.approx(0.10 - n_degenerate / 100, abs=1e-9)
        assert "不参与 passed" in margin["note"]


def test_margins_report_all_four_fields():
    run = _h1_run(_rows_with_degeneration(100, 3))
    h1 = g4.margins(g4.ARM_H1, g4.check_arm(g4.ARM_H1, run, cfg=CFG,
                                            criteria=g4.G4Criteria(n_traces_per_arm=100)))
    assert set(h1) >= {"h1_eligible_margin", "degeneration_margin", "max_tokens_margin"}


# ── 完整性检查 ───────────────────────────────────────────────────────────────
def test_missing_family_coverage_fails():
    rows = _rows_with_degeneration(40, 0)
    run = _h1_run(rows)
    for row in run.rows:
        row["family"] = "attack"
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "covers_both_families" in result["failed_checks"]


def test_single_label_class_fails():
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        row["label"] = 0
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "both_label_classes_present" in result["failed_checks"]


def test_constant_f5_column_fails():
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        row["f5_id_logprob_mean"] = 1.0
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "f5_not_constant" in result["failed_checks"]


def test_non_finite_hidden_fails():
    run = _h1_run(_rows_with_degeneration(40, 0))
    run.hidden["u0"][0] = np.nan
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "hidden_all_finite" in result["failed_checks"]


def test_wrong_max_new_tokens_fails():
    run = _h1_run(_rows_with_degeneration(40, 0))
    run.fingerprint = {"generation_params": {"max_new_tokens": 512}}
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "max_new_tokens_exact" in result["failed_checks"]


# ── 与 G3 生产指纹一致 ───────────────────────────────────────────────────────
def _fp(arm: str, **over) -> dict:
    fp = {"model": {"model_hash": "m"}, "tokenizer": {"tokenizer_hash": "t"},
          "hidden_index": {"block_index_zero_based": 19},
          "generation_params": {"K": 6, "temperature": 0.7, "top_p": 0.95, "seed": 4242,
                                "max_new_tokens": H1_MAX if arm == g4.ARM_H1 else MCQ_MAX}}
    fp.update(over)
    return fp


def test_fingerprint_matching_g3_passes():
    check = g4.matches_g3_production_fingerprint(
        g4.ARM_H1, _fp(g4.ARM_H1), _fp(g4.ARM_H1), cfg=CFG)
    assert check["ok"] is True and check["mismatched_keys"] == []


@pytest.mark.parametrize("patch,expected", [
    ({"model": {"model_hash": "other"}}, "model"),
    ({"tokenizer": {"tokenizer_hash": "other"}}, "tokenizer"),
    ({"hidden_index": {"block_index_zero_based": 15}}, "hidden_index"),
])
def test_fingerprint_drift_from_g3_fails(patch, expected):
    check = g4.matches_g3_production_fingerprint(
        g4.ARM_H1, _fp(g4.ARM_H1, **patch), _fp(g4.ARM_H1), cfg=CFG)
    assert check["ok"] is False and expected in check["mismatched_keys"]


def test_fingerprint_seed_drift_from_g3_fails():
    drifted = _fp(g4.ARM_H1)
    drifted["generation_params"] = {**drifted["generation_params"], "seed": 999}
    check = g4.matches_g3_production_fingerprint(g4.ARM_H1, drifted, _fp(g4.ARM_H1), cfg=CFG)
    assert check["ok"] is False and "generation_params.seed" in check["mismatched_keys"]


def test_fingerprint_arm_max_new_tokens_must_be_exact():
    drifted = _fp(g4.ARM_MCQ)
    drifted["generation_params"] = {**drifted["generation_params"], "max_new_tokens": 32}
    check = g4.matches_g3_production_fingerprint(g4.ARM_MCQ, drifted, _fp(g4.ARM_MCQ), cfg=CFG)
    assert check["ok"] is False
    assert "generation_params.max_new_tokens[mcq]" in check["mismatched_keys"]


# ── 检查器修正（2026-08-08）：三处字段定义 ───────────────────────────────────
def test_f5_columns_match_stage0_exactly_five():
    """列名以 Stage 0 为准，不得自造。Stage 0 = run_stage0_quality_check.py:37-38。"""
    assert g4.H1_F5_CONTINUOUS == (
        "f5_id_logprob_mean", "f5_id_logprob_min", "f5_id_token_entropy_mean",
        "f5_completion_perplexity", "f5_lengthnorm_logprob")
    assert len(g4.H1_F5_CONTINUOUS) == 5
    for invented in ("f5_id_mean_logprob", "f5_id_min_logprob", "f5_seq_mean_logprob"):
        assert invented not in g4.H1_F5_CONTINUOUS, "自造列名不得再出现"


def test_f5_entropy_column_is_actually_checked():
    """五列里的 f5_id_token_entropy_mean 必须真的参与检查（先前整列缺失）。"""
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        del row["f5_id_token_entropy_mean"]
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "f5_no_missing" in result["failed_checks"]
    assert "f5_id_token_entropy_mean" in result["stats"]["f5_missing"]


def test_label_is_binary_integer_not_string():
    """H1 的 label 是 0/1 整数；两类都在即通过。"""
    run = _h1_run(_rows_with_degeneration(40, 0))
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "both_label_classes_present" not in result["failed_checks"]
    assert set(result["stats"]["label_distribution"]) == {"0", "1"}


def test_label_distribution_is_reported_for_audit():
    """真实分布高度不平衡（实测 213:8）；报告须给出计数，供后续判读。"""
    rows = _rows_with_degeneration(40, 0)
    run = _h1_run(rows)
    for i, row in enumerate(run.rows):
        row["label"] = 1 if i < 3 else 0
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert result["stats"]["label_distribution"] == {"0": 37, "1": 3}
    assert "both_label_classes_present" not in result["failed_checks"]


def test_group_id_is_still_required_by_schema():
    """group_id **不从要求中删除** —— Stage 0 的 G4 与 v3 adapter 都需要它。"""
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        del row["group_id"]
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "schema_fields_present" in result["failed_checks"]


def test_backfill_group_id_from_frozen_records_only_touches_memory():
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        del row["group_id"]
    records = [{"example_id": f"e{i}", "group_id": f"frozen_g{i}"} for i in range(40)]
    filled = g4.backfill_group_id(run, records)
    assert filled == 40
    assert all(r["group_id"].startswith("frozen_g") for r in run.rows)
    result = g4.check_arm(g4.ARM_H1, run, cfg=CFG, criteria=g4.G4Criteria(n_traces_per_arm=40))
    assert "schema_fields_present" not in result["failed_checks"]


def test_backfill_is_deterministic_by_example_id():
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        del row["group_id"]
    records = [{"example_id": f"e{i}", "group_id": f"frozen_g{i}"} for i in range(40)]
    g4.backfill_group_id(run, records)
    for row in run.rows:
        assert row["group_id"] == f"frozen_g{row['example_id'][1:]}"


def test_backfill_refuses_when_the_frozen_list_lacks_the_example():
    run = _h1_run(_rows_with_degeneration(40, 0))
    for row in run.rows:
        del row["group_id"]
    with pytest.raises(ValueError, match="无法确定性回填"):
        g4.backfill_group_id(run, [{"example_id": "nope", "group_id": "g"}])


def test_backfill_does_not_overwrite_an_existing_group_id():
    run = _h1_run(_rows_with_degeneration(40, 0))
    records = [{"example_id": f"e{i}", "group_id": "SHOULD_NOT_WIN"} for i in range(40)]
    assert g4.backfill_group_id(run, records) == 0
    assert all(r["group_id"].startswith("g") for r in run.rows)


def test_criterion_correction_block_is_complete():
    cc = g4.CRITERION_CORRECTION
    assert cc["model_outputs_rerun"] is False
    assert cc["performance_values_used_for_correction"] is False
    assert cc["thresholds_changed"] is False and cc["samples_changed"] is False
    assert cc["generation_params_changed"] is False
    assert len(cc["corrected_fields"]) == 3
    assert cc["superseded_artifact_sha256"].startswith("40ff3e8c")
