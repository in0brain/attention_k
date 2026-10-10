"""Stage 1 的 G4 性能盲质量检查（启动 gate G4，v3.1）。

**判据在本模块顶部写死，运行前即已固定。** 任一冻结门槛失败 → `passed=false` 并停止；
**不得**换 G4 样本、调整阈值或修改生成参数后重跑。

为什么另起 v3 入口而不复用 `generation/quality_check.py`（实测，非推测）：

    quality_check.py:25  MAX_DEGENERATION_RATE = 0.10
                         → 只出现在 build_g4_report 的 `thresholds` 字典里，**从未参与比较**
    quality_check.py:26  MAX_TOUCHED_MAX_TOKENS_RATE = 0.10
                         → **全文再无第二处引用**，连 thresholds 都没报
    passed = all(per_arm ok) and all(numeric_match) and backend_same
             per_arm ok = all(checks.values())，两者都不在 checks 里

即：一个只报不判、另一个连报都没有。Stage 1 **实算实强制**，不照抄后声称已检查。
Stage 0 的 `quality_check.py` 本轮一行未改，其既有产物与结论不受影响。

**performance-blind**：不计算 AUROC / AUPRC / Δ_H / CI / ladder / probe /
任何正式样本统计。（`label` / `wrong_label` 是该样本的**真值**，不是"探针表现如何"，
与 Stage 0 的 G4 同一处理。）
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention import domain_label_proxy as dlp
from recover_attention import h1_f5_features as f5f
from recover_attention import mcq_conditional_increment as mci
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_g3_smoke import expected_max_new_tokens
from recover_attention.evaluation.v3_g3_smoke import (
    ARM_H1,
    ARM_MCQ,
    ARMS,
    DID_NOT_COMPUTE,
    ArmRun,
    write_report,
)

SCHEMA_VERSION = "stage1_g4_quality_report_v1"


# ══════════════════════════════════════════════════════════════════════════════
# 判据（**跑前写死**，2026-08-08 用户裁决）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class G4Criteria:
    n_prompts_per_arm: int = 40
    traces_per_prompt: int = 6
    n_traces_per_arm: int = 240
    # 三项模型几何：默认 = Qwen 冻结值，跨模型由 CLI 显式覆盖（同 G3Criteria）。
    hidden_dim: int = 3584
    block_index: int = 19
    tuple_index: int = 20
    h1_min_eligible_rate: float = 0.80
    mcq_max_parse_failure_rate: float = 0.10
    max_degeneration_rate: float = 0.10
    max_touched_max_new_tokens_rate: float = 0.10

    def as_dict(self) -> dict:
        return {
            "frozen_before_run": True,
            "n_prompts_per_arm": self.n_prompts_per_arm,
            "traces_per_prompt": self.traces_per_prompt,
            "n_traces_per_arm": self.n_traces_per_arm,
            "hidden_dim": self.hidden_dim,
            "block_index_zero_based": self.block_index,
            "hidden_states_tuple_index": self.tuple_index,
            "h1_min_eligible_rate": self.h1_min_eligible_rate,
            "mcq_max_parse_failure_rate": self.mcq_max_parse_failure_rate,
            "max_degeneration_rate": self.max_degeneration_rate,
            "max_touched_max_new_tokens_rate": self.max_touched_max_new_tokens_rate,
            "boundary_is_inclusive": "阈值均为 ≤ / ≥，恰好等于阈值判**通过**",
            "enforced_here_unlike_stage0": [
                "max_degeneration_rate", "max_touched_max_new_tokens_rate"],
            "on_failure": "passed=false 并停止；不得换样本、调阈值或改生成参数后重跑",
        }


CRITERIA = G4Criteria()

#: 2026-08-08 的检查器修正（**语义未变，改的是实现读错的字段**）。
#: 与 G3 的 `max_new_tokens` 同类：不改阈值、不改样本、不改生成参数、不重跑模型。
CRITERION_CORRECTION = {
    "corrected_fields": [
        {"field": "H1_F5_CONTINUOUS",
         "was": ["f5_id_mean_logprob", "f5_id_min_logprob", "f5_seq_mean_logprob"],
         "now": list(("f5_id_logprob_mean", "f5_id_logprob_min", "f5_id_token_entropy_mean",
                      "f5_completion_perplexity", "f5_lengthnorm_logprob")),
         "reason": "先前三列是凭印象自造的列名，生成行里根本不存在",
         "evidence": "与 Stage 0 逐字一致：scripts/run_stage0_quality_check.py:37-38（五列）"},
        {"field": "both_label_classes_present",
         "was": 'label ∈ {"grounded", "fabricated"}（字符串）',
         "now": "label ∈ {0, 1}（二元整数）",
         "reason": "H1 的 label 是二元整数，字符串集合取交恒为空 → 误判两类缺失",
         "evidence": "实测 eligible 行 label 分布 {0: 213, 1: 8}，两类均在"},
        {"field": "schema_fields_present / group_id",
         "was": "拟从 schema 要求中删除 group_id",
         "now": "**保留要求**；改为在保存新行前从冻结输入记录确定性挂载，"
                "既有行按 example_id 在读取时回填（只改内存，不改写落盘行）",
         "reason": "Stage 0 的 G4 显式从输入记录复制 group_id，v3 adapter 也强制需要它；"
                   "缺的是运行编排，不是要求本身",
         "evidence": "两个 backend 的 run_unit 均不产出 group_id（实测 0/240）"},
    ],
    "model_outputs_rerun": False,
    "performance_values_used_for_correction": False,
    "thresholds_changed": False,
    "samples_changed": False,
    "generation_params_changed": False,
    "superseded_artifact": "h1_g4_quality_report_misspecified_check_20260808.json",
    "superseded_artifact_sha256":
        "40ff3e8cbb8280a61f25abdeab89fb72bec25f6d23ac38de918563ab27966896",
    "superseded_verdict": ("passed=false, failed_checks=['both_label_classes_present', "
                           "'f5_no_missing', 'schema_fields_present']"),
    "note": "先前的失败报告保留归档，未静默抹掉",
}

#: H1 的退化候选规则（**不含 truncated_failure**）。
#: H1 是自由文本，`parse_option_answer` 对它没有语义：rule 3 会把"生成较长"误判为退化，
#: 且与 `touched_max_new_tokens` 重复计数。长度/预算问题单独由后者度量。
H1_DEGENERATION_RULES = ("single_char_run", "substring_loop")
#: MCQ 用完整三规则（它有合法选项字母），且用**精确 token 数**而非字符长度近似。
MCQ_DEGENERATION_RULES = ("single_char_run", "substring_loop", "truncated_failure")


# ══════════════════════════════════════════════════════════════════════════════
# 退化与预算
# ══════════════════════════════════════════════════════════════════════════════
def classify_h1_degeneration(completion: str) -> dict:
    """H1 的门控退化定义（2026-08-08 冻结）：

    ```
    degenerate_for_gate = single_char_run
                          OR (substring_loop AND NOT looks_like_structured_identifier_list)
    ```

    structured-list 豁免**只作用于"仅由 identifier 列表结构触发的 substring_loop"**：
    同时命中 `single_char_run` 时仍判退化。豁免样本**不静默删除** —— 报告里保留
    raw / exempted / adjusted 三个计数，可还原 `raw = adjusted + exempted`。
    """
    raw = dlp.detect_degeneration(completion, valid_labels=list(mci.MCQ_LETTERS),
                                  max_new_tokens=10 ** 9,      # 让 rule 3 恒不触发
                                  num_generated_tokens=0)
    matched = [r for r in raw["matched_rules"] if r in H1_DEGENERATION_RULES]
    single = "single_char_run" in matched
    loop = "substring_loop" in matched
    structured = f5f.looks_like_structured_identifier_list(completion)
    exempted = bool(loop and structured and not single)
    return {
        "matched_rules": matched,
        "raw_degenerate": bool(matched),
        "structured_list_like": bool(structured),
        "structured_list_exempted": exempted,
        "degenerate_for_gate": bool(single or (loop and not structured)),
    }


def classify_mcq_degeneration(completion: str, *, num_generated_tokens: int,
                              max_new_tokens: int) -> dict:
    """MCQ 用完整三规则；**不使用 structured-list 豁免**；用精确 token 数。"""
    raw = dlp.detect_degeneration(completion, valid_labels=list(mci.MCQ_LETTERS),
                                  max_new_tokens=max_new_tokens,
                                  num_generated_tokens=num_generated_tokens)
    return {
        "matched_rules": list(raw["matched_rules"]),
        "raw_degenerate": bool(raw["degenerate"]),
        "structured_list_like": False,
        "structured_list_exempted": False,
        "degenerate_for_gate": bool(raw["degenerate"]),
    }


def classify_row(arm: str, row: dict, *, max_new_tokens: int) -> dict:
    """逐行给出退化判定与"是否撞到生成上限"。

    `touched_max_new_tokens` 用**精确**新生成 token 数（由 runner 的记录型代理捕获），
    定义为 `num_generated_tokens >= max_new_tokens`，与 degeneration 里 `≥0.9·max`
    的截断规则是**两个不同的量**。
    """
    completion = row.get("completion") or row.get("raw_completion") or ""
    n_tokens = row.get("num_generated_tokens")
    if n_tokens is None:
        raise ValueError(
            f"{row.get('unit_id')} 缺 num_generated_tokens —— G4 要求精确 token 数，"
            f"不接受字符长度近似")
    if arm == ARM_H1:
        verdict = classify_h1_degeneration(completion)
    else:
        verdict = classify_mcq_degeneration(completion, num_generated_tokens=int(n_tokens),
                                            max_new_tokens=max_new_tokens)
    verdict["num_generated_tokens"] = int(n_tokens)
    verdict["touched_max_new_tokens"] = bool(int(n_tokens) >= max_new_tokens)
    return verdict


def degeneration_stats(arm: str, rows: list[dict], *, max_new_tokens: int) -> dict:
    """全部计数与比率。**不静默删除豁免样本**：raw = adjusted + exempted（仅 loop 情形）。"""
    verdicts = [classify_row(arm, row, max_new_tokens=max_new_tokens) for row in rows]
    n = len(verdicts) or 1
    raw = sum(v["raw_degenerate"] for v in verdicts)
    exempted = sum(v["structured_list_exempted"] for v in verdicts)
    adjusted = sum(v["degenerate_for_gate"] for v in verdicts)
    touched = sum(v["touched_max_new_tokens"] for v in verdicts)

    rule_hist: dict[str, int] = {}
    for verdict in verdicts:
        for rule in verdict["matched_rules"]:
            rule_hist[rule] = rule_hist.get(rule, 0) + 1

    if raw != adjusted + exempted:
        raise ValueError(
            f"{arm}: raw({raw}) != adjusted({adjusted}) + exempted({exempted}) —— "
            f"豁免账目对不上，报告将无法还原")

    return {
        "n_rows": len(verdicts),
        "rules_used": list(H1_DEGENERATION_RULES if arm == ARM_H1 else MCQ_DEGENERATION_RULES),
        "raw_pattern_degeneration_count": raw,
        "raw_pattern_degeneration_rate": round(raw / n, 4),
        "structured_list_false_positive_count": exempted,
        "structured_list_false_positive_rate": round(exempted / n, 4),
        "adjusted_degeneration_count": adjusted,
        "adjusted_degeneration_rate": round(adjusted / n, 4),
        "touched_max_new_tokens_count": touched,
        "touched_max_new_tokens_rate": round(touched / n, 4),
        "matched_rules_histogram": dict(sorted(rule_hist.items())),
        "max_new_tokens": max_new_tokens,
        "reconstruction": "raw = adjusted + exempted（豁免仅作用于 substring_loop 情形）",
        "gate_uses": "adjusted_degeneration_rate",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 完整性检查
# ══════════════════════════════════════════════════════════════════════════════
def _finite_column(rows: list[dict], name: str) -> list[float] | None:
    values = []
    for row in rows:
        value = row.get(name)
        if value is None or not np.isfinite(float(value)):
            return None
        values.append(float(value))
    return values


def _f5_checks(rows: list[dict], columns: tuple[str, ...], stats: dict) -> dict:
    missing, all_zero, constant = [], [], []
    for name in columns:
        values = _finite_column(rows, name)
        if values is None:
            missing.append(name)
            continue
        if all(v == 0.0 for v in values):
            all_zero.append(name)
        if len(set(values)) == 1:
            constant.append(name)
    stats["f5_columns_checked"] = list(columns)
    stats["f5_missing"] = missing
    stats["f5_all_zero"] = all_zero
    stats["f5_constant"] = constant
    return {"f5_no_missing": not missing, "f5_not_all_zero": not all_zero,
            "f5_not_constant": not constant}


#: 与 Stage 0 **逐字一致**的五列（`scripts/run_stage0_quality_check.py:37-38`）。
#: 先前写成 f5_id_mean_logprob / f5_id_min_logprob / f5_seq_mean_logprob 是**凭印象编的列名**，
#: 生成行里根本不存在这三列 → 误判 f5_no_missing。列名以 Stage 0 的定义为准，不得自造。
H1_F5_CONTINUOUS = ("f5_id_logprob_mean", "f5_id_logprob_min", "f5_id_token_entropy_mean",
                    "f5_completion_perplexity", "f5_lengthnorm_logprob")
MCQ_F5_CONTINUOUS = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy",
                     "f5_letter_token_logprob")

#: H1 的 `label` 是**二元整数** 0/1（实测 {0: 213, 1: 8}），不是 grounded/fabricated 字符串。
H1_LABEL_CLASSES = (0, 1)


def backfill_group_id(run: ArmRun, records: list[dict]) -> int:
    """按 `example_id` 从**冻结的 gate 名单**回填 `group_id` —— 只改内存，不改写原始行。

    两个 backend 的 `run_unit` 都不产出 `group_id`；新跑的行已由 `run_arm` 在写盘前挂载，
    但**先前已落盘的行**没有。此处按冻结输入记录在读取后回填，
    **不重写 `*_unit_results.jsonl`**（那会破坏 append-only 与既有 SHA）。

    回填是确定性的：同一 `example_id` 只对应一个 `group_id`；对不上即抛错，不猜。
    """
    by_id = {r["example_id"]: r.get("group_id") for r in records}
    filled = 0
    for row in run.rows:
        if row.get("group_id") is not None:
            continue
        group_id = by_id.get(row["example_id"])
        if group_id is None:
            raise ValueError(
                f"{row['unit_id']}: 冻结名单里没有 example_id={row['example_id']!r} 的 "
                f"group_id，无法确定性回填")
        row["group_id"] = group_id
        filled += 1
    return filled


def check_arm(arm: str, run: ArmRun, *, cfg: V3AnalysisConfig,
              criteria: G4Criteria = CRITERIA,
              budget_registration=None) -> dict:
    rows = run.rows
    # 本臂**应当**出现的解码预算：未登记取冻结值，已登记取登记值。
    # `touched_max_new_tokens` 与 `max_new_tokens_exact` 都用它 —— 用错这个数，
    # 退化统计会把每一条都算成撞上限，指纹检查也会整臂报错。
    max_new = expected_max_new_tokens(arm, cfg, budget_registration)
    degen = degeneration_stats(arm, rows, max_new_tokens=max_new)
    stats: dict[str, Any] = {"n_rows": len(rows), "n_units": run.n_units,
                             "degeneration": degen}
    checks: dict[str, Any] = {}

    checks["n_traces_ok"] = run.n_units == criteria.n_traces_per_arm
    checks["every_unit_produced_a_row"] = len(rows) == run.n_units
    checks["no_unhandled_exception"] = not [f for f in run.failures
                                            if f["kind"] != "forward_nonfinite"]
    checks["no_forward_nonfinite"] = not any(f["kind"] == "forward_nonfinite"
                                             for f in run.failures)

    # ── 冻结门槛：退化与预算（Stage 0 从未真正强制的两条）──────────────────
    checks["degeneration_rate_ok"] = (
        degen["adjusted_degeneration_rate"] <= criteria.max_degeneration_rate)
    checks["touched_max_new_tokens_rate_ok"] = (
        degen["touched_max_new_tokens_rate"] <= criteria.max_touched_max_new_tokens_rate)

    if arm == ARM_H1:
        eligible = [r for r in rows if r.get("id_token_positions_local")]
        rate = len(eligible) / (len(rows) or 1)
        stats["eligible_rate"] = round(rate, 4)
        stats["n_eligible"] = len(eligible)
        checks["eligible_rate_ok"] = rate >= criteria.h1_min_eligible_rate
        families = {r.get("family") for r in rows}
        stats["families"] = sorted(f for f in families if f)
        checks["covers_both_families"] = {"attack", "cwe"} <= families
        labels = [r.get("label") for r in eligible]
        classes = {int(v) for v in labels if v is not None}
        stats["label_distribution"] = {
            str(c): sum(1 for v in labels if v is not None and int(v) == c)
            for c in sorted(classes)}
        checks["both_label_classes_present"] = classes >= set(H1_LABEL_CLASSES)
        checks["every_eligible_has_hidden"] = all(r["unit_id"] in run.hidden for r in eligible)
        checks["pooling_is_all_identifier_tokens"] = all(
            r.get("hidden_pooled_n_tokens") == len(set(r["id_token_positions_local"]))
            for r in eligible)
        checks.update(_f5_checks(eligible, H1_F5_CONTINUOUS, stats))
        checks["schema_fields_present"] = all(
            {"example_id", "group_id", "prompt_hash", "unit_hash", "completion"} <= set(r)
            for r in rows)
        scored = eligible
    else:
        parsed = [r for r in rows if r.get("eligible_for_primary")]
        rate = (len(rows) - len(parsed)) / (len(rows) or 1)
        stats["parse_failure_rate"] = round(rate, 4)
        stats["n_parsed"] = len(parsed)
        checks["parse_failure_rate_ok"] = rate <= criteria.mcq_max_parse_failure_rate
        checks["both_label_classes_present"] = len(
            {r.get("wrong_label") for r in parsed} & {0, 1}) == 2
        checks["every_parsed_has_hidden"] = all(r["unit_id"] in run.hidden for r in parsed)
        checks["readout_position_ok"] = all(
            r.get("readout_to_prompt_end_distance") is not None for r in parsed)
        checks.update(_f5_checks(parsed, MCQ_F5_CONTINUOUS, stats))
        checks["schema_fields_present"] = all(
            {"example_id", "parsed_label", "prompt_hash", "unit_hash",
             "raw_completion"} <= set(r) for r in rows)
        scored = parsed

    dims = {int(v.size) for v in run.hidden.values()}
    stats["hidden_dims_seen"] = sorted(dims)
    checks["hidden_dim_ok"] = dims == {criteria.hidden_dim} if dims else False
    checks["hidden_all_finite"] = all(bool(np.all(np.isfinite(v)))
                                      for v in run.hidden.values())
    hv = run.hidden_verification
    checks["hidden_index_numeric_match"] = bool(hv.get("numeric_match"))
    checks["hidden_index_matches_frozen"] = (
        hv.get("block_index_zero_based") == criteria.block_index
        and hv.get("hidden_states_tuple_index") == criteria.tuple_index)
    checks["max_new_tokens_exact"] = (
        run.fingerprint.get("generation_params", {}).get("max_new_tokens") == max_new)
    stats["n_scored_rows"] = len(scored)

    boolean = {k: v for k, v in checks.items() if isinstance(v, bool)}
    return {"arm": arm, "checks": checks, "stats": stats,
            "failed_checks": sorted(k for k, v in boolean.items() if not v),
            "ok": all(boolean.values())}


def margins(arm: str, evaluation: dict, criteria: G4Criteria = CRITERIA) -> dict:
    """**只报不判**：余量字段不参与 `passed`，也不得据此提高门槛或重跑。"""
    stats = evaluation["stats"]
    degen = stats["degeneration"]
    out = {
        "degeneration_margin": round(
            criteria.max_degeneration_rate - degen["adjusted_degeneration_rate"], 4),
        "max_tokens_margin": round(
            criteria.max_touched_max_new_tokens_rate - degen["touched_max_new_tokens_rate"], 4),
        "note": "报告用，不参与 passed；不得据余量提高门槛、换样本或重跑",
    }
    if arm == ARM_H1:
        out["h1_eligible_margin"] = round(
            stats["eligible_rate"] - criteria.h1_min_eligible_rate, 4)
    else:
        out["mcq_parse_margin"] = round(
            criteria.mcq_max_parse_failure_rate - stats["parse_failure_rate"], 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 与 G3 生产指纹的一致性（三段式，沿用已裁决口径）
# ══════════════════════════════════════════════════════════════════════════════
from recover_attention.evaluation.v3_g3_smoke import (  # noqa: E402
    FINGERPRINT_KEYS,
    INVARIANT_GENERATION_PARAMS,
    RECORDED_SAMPLING_PARAMS,
    _all_schema_tagged,
    generation_params_schema_mismatch,
)


def matches_g3_production_fingerprint(arm: str, fingerprint: dict, g3_fingerprint: dict,
                                      *, cfg: V3AnalysisConfig,
                                      budget_registration=None) -> dict:
    """G4 的指纹必须与 **G3 冻结的生产指纹**一致（三段式，非整块比较）。"""
    mismatches = []
    for key in FINGERPRINT_KEYS:
        if json.dumps(fingerprint.get(key), sort_keys=True) != \
           json.dumps(g3_fingerprint.get(key), sort_keys=True):
            mismatches.append(key)
    for param in INVARIANT_GENERATION_PARAMS:
        if fingerprint.get("generation_params", {}).get(param) != \
           g3_fingerprint.get("generation_params", {}).get(param):
            mismatches.append(f"generation_params.{param}")

    # 新旧口径不得混用：旧指纹没有 top_k / repetition_penalty，其实际生效值无从得知，
    # **不能**当成"与任意取值兼容"。双方同口径时才逐键比这两个采样参数。
    schema_issues = generation_params_schema_mismatch(fingerprint, g3_fingerprint)
    mismatches.extend(schema_issues)
    if not schema_issues and _all_schema_tagged(fingerprint, g3_fingerprint):
        for param in RECORDED_SAMPLING_PARAMS:
            if fingerprint.get("generation_params", {}).get(param) != \
               g3_fingerprint.get("generation_params", {}).get(param):
                mismatches.append(f"generation_params.{param}")

    expected = expected_max_new_tokens(arm, cfg, budget_registration)
    actual = fingerprint.get("generation_params", {}).get("max_new_tokens")
    if actual != expected:
        mismatches.append(f"generation_params.max_new_tokens[{arm}]")
    # G3 的生产指纹必须**也**是这个预算 —— 否则 G4 与 G3 跑在两个不同的预算上，
    # 而上面那一段只查了 G4 自己。缺这一条就能把旧 16-token 的 G3 证据接到
    # 新 512 的 G4 上。
    g3_actual = g3_fingerprint.get("generation_params", {}).get("max_new_tokens")
    if g3_actual != expected:
        mismatches.append(f"g3.generation_params.max_new_tokens[{arm}]")
    return {"ok": not mismatches, "mismatched_keys": mismatches,
            "max_new_tokens": {"expected": expected, "actual": actual,
                               "g3_actual": g3_actual},
            "budget_registration": (budget_registration.as_dict()
                                    if budget_registration else None)}


def build_report(*, arm: str, run: ArmRun, verification: dict, evaluation: dict,
                 fingerprint_check: dict, cfg: V3AnalysisConfig,
                 g3_provenance: dict, criteria: G4Criteria = CRITERIA) -> dict:
    # 报告必须记录**本次实际使用**的判据，而不是模块默认值 —— 跨模型运行时
    # 两者会不同，写默认值等于让报告说谎。
    g2 = verification["g2"]
    passed = evaluation["ok"] and fingerprint_check["ok"]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "G4",
        "arm": arm,
        "gate_only": True,
        "previously_burned": arm == ARM_H1,
        "performance_blind": True,
        "did_not_compute": list(DID_NOT_COMPUTE),
        "prereg_sha256": g2["current_sha256"],
        "hash_mode": g2["hash_mode"],
        "lock_version": g2["locked_version"],
        "sample_manifest_sha256": verification["sample_manifest_sha256"],
        "primary_lists_read": verification["primary_lists_read"],
        "criteria": criteria.as_dict(),
        "criterion_correction": CRITERION_CORRECTION,
        "n_prompts": verification["gate_counts"][arm],
        "n_traces": run.n_units,
        "checks": evaluation["checks"],
        "stats": evaluation["stats"],
        "failed_checks": evaluation["failed_checks"],
        "margins": margins(arm, evaluation, criteria),
        "g3_provenance": g3_provenance,
        "fingerprint_matches_g3_production": fingerprint_check,
        "fingerprint": run.fingerprint,
        "hidden_verification": run.hidden_verification,
        "provenance": cfg.provenance(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "generation_timing": {
            "n_units_total": run.n_units,
            "n_units_generated_this_process": run.n_generated_this_process,
            "elapsed_seconds_this_process": round(run.seconds, 1),
            "seconds_per_generated_unit": (round(run.seconds / run.n_generated_this_process, 2)
                                           if run.n_generated_this_process else None),
            "note": ("耗时与速率**只覆盖本进程新生成的 unit**；跨进程续跑的总时长与"
                     "分段速率见 docs/progress/stage1_v3_analysis_history.md，"
                     "不得用本字段的分母冒充全部 unit"),
        },
        "n_failures": len(run.failures),
        "failures": run.failures[:20],
        "passed": passed,
    }


__all__ = ["CRITERIA", "G4Criteria", "SCHEMA_VERSION", "ARMS", "ARM_H1", "ARM_MCQ",
           "classify_h1_degeneration", "classify_mcq_degeneration", "classify_row",
           "degeneration_stats", "check_arm", "margins", "build_report",
           "matches_g3_production_fingerprint", "write_report",
           "H1_DEGENERATION_RULES", "MCQ_DEGENERATION_RULES"]
