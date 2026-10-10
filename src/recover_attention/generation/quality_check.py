"""G4：production quality check（§2.1，**性能盲**）。

只回答一个问题:**生成管线在真实数据上能不能产出可用记录?**
不回答:记录说明了什么。

性能盲由代码强制,不是承诺:
  - 本模块**不导入** evaluation 包,也不导入任何 AUROC/AUPRC/bootstrap 实现;
  - 报告 schema 由 assert_no_performance_fields 校验,出现性能字段即抛错;
  - 唯一接触标签的地方是"正负例各有多少"(判断样本可用性所需),不做任何按标签的
    判别性度量。
理由:G4 若能看性能,就等于在冻结前先偷看结果、再据此决定要不要重跑 —— 那正是
预注册要防的事(§2.1)。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# §2.1 冻结的运行质量阈值（与标签/性能无关）
H1_MIN_ELIGIBLE_RATE = 0.80
MCQ_MAX_PARSE_FAILURE_RATE = 0.10
MAX_DEGENERATION_RATE = 0.10
MAX_TOUCHED_MAX_TOKENS_RATE = 0.10

# 报告里绝不允许出现的字段（性能盲的强制点）
FORBIDDEN_PERFORMANCE_SUBSTRINGS = (
    "auroc", "auprc", "delta_h", "delta_auroc", "s_observability", "rank_biserial",
    "increment", "ladder", "bootstrap_ci", "ci_lo", "ci_hi", "gate_verdict",
    "risk_score", "oof",
)


class PerformanceLeakInQualityCheck(AssertionError):
    """G4 报告里出现了性能指标 —— 那等于冻结前偷看结果。"""


def assert_no_performance_fields(report: Any, path: str = "") -> None:
    """递归校验:G4 报告不得含任何性能字段。

    这是 §2.1 "性能盲" 的强制点。名字里带 auroc/increment/ci_lo… 的键一律拒绝。
    """
    if isinstance(report, dict):
        for k, v in report.items():
            key = str(k).lower()
            for bad in FORBIDDEN_PERFORMANCE_SUBSTRINGS:
                if bad in key:
                    raise PerformanceLeakInQualityCheck(
                        f"performance field {path}.{k!r} in a G4 quality report; §2.1 forbids "
                        f"computing or viewing performance before Stage 0 is frozen")
            assert_no_performance_fields(v, f"{path}.{k}")
    elif isinstance(report, (list, tuple)):
        for i, v in enumerate(report):
            assert_no_performance_fields(v, f"{path}[{i}]")


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def check_h1_batch(rows: Sequence[dict], hidden: dict[str, np.ndarray],
                   f5_continuous: Sequence[str]) -> dict:
    """H1 小批的运行质量。**不看标签的判别性**，只看能否产出可用记录。"""
    n = len(rows)
    eligible = [r for r in rows if r.get("eligible")]
    checks: dict[str, Any] = {}
    stats: dict[str, Any] = {"n_traces": n, "n_eligible": len(eligible)}

    # eligible rate 必须只对**可能 eligible 的题**算。§3 的主 population 是 ATT&CK/CWE;
    # CVE 族的题模型正确地答 CVE → only_cve → 按设计不进主 population,那是**正确行为**,
    # 不是质量缺陷。把 CVE 题算进分母会让阈值惩罚正确的模型输出。
    # (G4 首跑就栽在这里:样本 12 attack + 8 cve,48 条 only_cve 全来自那 8 个 CVE 题,
    #  整体 rate 0.60 触发失败,而 attack 题的 eligible 率是 12/12 = 100%。)
    capable = [r for r in rows if r.get("family") in ("attack", "cwe")]
    cap_eligible = [r for r in capable if r.get("eligible")]
    stats["eligible_rate_all_traces"] = _rate(len(eligible), n)
    stats["n_eligible_capable_traces"] = len(capable)
    stats["eligible_rate_among_capable"] = _rate(len(cap_eligible), len(capable))
    stats["family_mix"] = _counter(r.get("family") for r in rows)
    stats["note_cve"] = ("CVE-family prompts are excluded from the primary population by "
                         "design (§3: ATT&CK/CWE only); they are not a quality failure")
    checks["eligible_rate_ok"] = (
        (stats["eligible_rate_among_capable"] or 0) >= H1_MIN_ELIGIBLE_RATE
        if capable else False)
    checks["sample_covers_both_primary_families"] = (
        {"attack", "cwe"} <= set(str(r.get("family")) for r in rows))

    reasons: dict[str, int] = {}
    for r in rows:
        why = r.get("primary_exclusion_reason")
        if why:
            reasons[why] = reasons.get(why, 0) + 1
    stats["primary_exclusion_reasons"] = reasons
    stats["emission_failure_rate"] = _rate(sum(bool(r.get("emission_failure")) for r in rows), n)

    # 样本可用性:两类都要有(否则后续 fold 无法分层)。这是可用性,不是性能。
    labels = [r["label"] for r in eligible if r.get("label") is not None]
    stats["n_positive"] = int(sum(labels))
    stats["n_negative"] = int(len(labels) - sum(labels))
    checks["both_classes_present"] = stats["n_positive"] > 0 and stats["n_negative"] > 0

    checks["every_eligible_has_hidden"] = all(r["trace_id"] in hidden for r in eligible)
    dims = {int(v.shape[0]) for v in hidden.values()}
    checks["hidden_dim_consistent"] = len(dims) <= 1
    checks["hidden_all_finite"] = all(bool(np.all(np.isfinite(v))) for v in hidden.values())
    stats["hidden_dim"] = sorted(dims)
    checks["hidden_pooled_all_identifier_tokens"] = all(
        r.get("hidden_pooled_n_tokens") == len(set(r.get("id_token_positions_local", [])))
        for r in eligible)

    checks.update(_f5_checks(eligible, f5_continuous, stats))
    checks["schema_fields_present"] = all(
        {"trace_id", "example_id", "group_id", "eligible", "prompt_hash", "unit_hash"} <= set(r)
        for r in rows)
    return {"arm": "h1", "checks": checks, "stats": stats,
            "ok": all(v is True for v in checks.values())}


def check_mcq_batch(rows: Sequence[dict], hidden: dict[str, np.ndarray],
                    f5_continuous: Sequence[str]) -> dict:
    """MCQ 小批的运行质量。"""
    n = len(rows)
    ok_rows = [r for r in rows if r.get("canonicalization_status") == "ok"]
    checks: dict[str, Any] = {}
    stats: dict[str, Any] = {"n_questions": n, "n_canonicalized": len(ok_rows)}

    fails = sum(1 for r in rows if r.get("canonicalization_status") != "ok")
    stats["parse_failure_rate"] = _rate(fails, n)
    checks["parse_failure_rate_ok"] = (stats["parse_failure_rate"] or 0) <= MCQ_MAX_PARSE_FAILURE_RATE

    stats["response_format"] = _counter(r.get("response_format") for r in rows)
    stats["letter_distribution"] = _counter(r.get("parsed_label") for r in ok_rows)

    labels = [r["wrong_label"] for r in ok_rows if r.get("wrong_label") is not None]
    stats["n_positive"] = int(sum(labels))
    stats["n_negative"] = int(len(labels) - sum(labels))
    checks["both_classes_present"] = stats["n_positive"] > 0 and stats["n_negative"] > 0

    checks["every_eligible_has_hidden"] = all(r["example_id"] in hidden for r in ok_rows)
    dims = {int(v.shape[0]) for v in hidden.values()}
    checks["hidden_dim_consistent"] = len(dims) <= 1
    checks["hidden_all_finite"] = all(bool(np.all(np.isfinite(v))) for v in hidden.values())
    stats["hidden_dim"] = sorted(dims)
    # §8:answer-letter token 自身 —— 读出位必须在 prompt 末位
    checks["readout_at_prompt_end"] = all(
        r.get("readout_to_prompt_end_distance") == 0 for r in ok_rows)
    checks["semantic_output_present"] = all(r.get("semantic_output_text") for r in ok_rows)

    checks.update(_f5_checks(ok_rows, f5_continuous, stats))
    checks["schema_fields_present"] = all(
        {"example_id", "group_id", "parsed_label", "canonicalization_status",
         "prompt_hash", "unit_hash"} <= set(r) for r in rows)
    return {"arm": "mcq", "checks": checks, "stats": stats,
            "ok": all(v is True for v in checks.values())}


def _counter(vals) -> dict:
    out: dict[str, int] = {}
    for v in vals:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _f5_checks(rows: Sequence[dict], f5_continuous: Sequence[str], stats: dict) -> dict:
    """F5 完整性:每个连续量列必须真的有值。

    缺失会被下游 _dense_matrix 静默填 0 —— 形状与 d_F 不变、其余检查照绿,列却是空的。
    这个洞在 MCQ 侧真实发生过(三列 19/19 全 0)。
    """
    nonzero, distinct, missing = {}, {}, {}
    for name in f5_continuous:
        vals = [r.get(name) for r in rows]
        fin = [float(v) for v in vals
               if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        nonzero[name] = int(sum(1 for v in fin if v != 0.0))
        distinct[name] = int(len(set(fin)))
        missing[name] = int(sum(1 for v in vals if v is None))
    stats["f5_completeness"] = {"nonzero_per_column": nonzero,
                                "distinct_per_column": distinct,
                                "missing_per_column": missing}
    return {"f5_all_continuous_columns_populated": all(v > 0 for v in nonzero.values()),
            "f5_no_constant_continuous_column": all(v > 1 for v in distinct.values())}


def build_g4_report(per_arm: dict[str, dict], hidden_verification: dict[str, dict],
                    fingerprints: dict[str, dict], prereg_sha: str) -> dict:
    """汇总 G4。落盘前强制校验性能盲。"""
    keys = ("load_in_8bit", "device_map", "attn_implementation", "local_files_only")
    backends = {a: {k: (fingerprints[a].get("model", {}).get("backend", {}) or {}).get(k)
                    for k in keys} for a in fingerprints}
    vals = list(backends.values())
    backend_same = bool(vals) and all(v == vals[0] for v in vals) and \
        vals[0].get("load_in_8bit") is True

    report = {
        "schema_version": "stage0_g4_quality_check_v1",
        "gate": "G4",
        "performance_blind": True,
        "did_not_compute": ["AUROC", "AUPRC", "Delta_H", "S", "D", "ladder rungs",
                            "any discriminative metric"],
        "why_blind": ("§2.1: a quality check that can see performance is a peek at the results "
                      "before freezing, and would let a bad-looking batch justify a rerun"),
        "prereg_sha256": prereg_sha,
        "per_arm": per_arm,
        "hidden_verification": hidden_verification,
        "backend_invariant": {"ok": backend_same, "per_arm": backends},
        "thresholds": {"h1_min_eligible_rate": H1_MIN_ELIGIBLE_RATE,
                       "mcq_max_parse_failure_rate": MCQ_MAX_PARSE_FAILURE_RATE,
                       "max_degeneration_rate": MAX_DEGENERATION_RATE},
        "passed": bool(all(v["ok"] for v in per_arm.values())
                       and all(h.get("numeric_match") for h in hidden_verification.values())
                       and backend_same),
    }
    assert_no_performance_fields(report)      # 性能盲的强制点
    return report
