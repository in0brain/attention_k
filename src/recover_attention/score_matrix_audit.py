"""Sprint 2I-R: score matrix decomposition and root-cause audit.

This module is deliberately read-only with respect to upstream sprint artifacts. It
builds a per-span audit matrix from existing 2H/2I outputs, separates formula inputs
from evaluation-only labels, and simulates simple score decompositions without
training a new model or rerunning recovery/attention/hidden-state stages.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl

BACKEND = "score_matrix_audit_v0"

FORBIDDEN_INPUT_FEATURE_SUBSTRINGS = (
    "recovered",
    "solution_path",
    "drift",
    "bucket",
    "risk_strength",
    "gold",
    "answer",
    "label",
    "target",
    "trajectory",
    "cot",
)

REASONING_SIGNAL_FIELDS = (
    "answer_logprob_delta",
    "trajectory_change",
    "cot_path_change",
    "nla_semantic_role",
)

NUMBER_RE = re.compile(
    r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?"
    r"|percent|%|half|twice|double|triple",
    re.IGNORECASE,
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _round_or_none(value: Any, ndigits: int = 6) -> float | None:
    value = _finite_float(value)
    return round(value, ndigits) if value is not None else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _minmax(values: list[float | None]) -> list[float | None]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [None for _ in values]
    lo, hi = min(finite), max(finite)
    if hi == lo:
        return [0.5 if v is not None else None for v in values]
    return [((v - lo) / (hi - lo)) if v is not None else None for v in values]


def _rank_desc(records: list[dict[str, Any]], score_key: str, output_key: str) -> None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    for rows in by_question.values():
        rows.sort(
            key=lambda r: (
                -(_finite_float(r.get(score_key), -float("inf")) or -float("inf")),
                str(r.get("masked_id")),
            )
        )
        last_score: float | None = None
        last_rank = 0
        for idx, row in enumerate(rows, start=1):
            score = _finite_float(row.get(score_key))
            if score != last_score:
                last_rank = idx
                last_score = score
            row[output_key] = last_rank if score is not None else None


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(values: list[float], targets: list[float]) -> float | None:
    pairs = [
        (float(v), float(t))
        for v, t in zip(values, targets)
        if v is not None and t is not None and math.isfinite(float(v)) and math.isfinite(float(t))
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    rx = _rankdata(list(xs))
    ry = _rankdata(list(ys))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    if denx == 0 or deny == 0:
        return None
    return round(num / (denx * deny), 6)


def auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    pos = [float(v) for v in pos_scores if v is not None and math.isfinite(float(v))]
    neg = [float(v) for v in neg_scores if v is not None and math.isfinite(float(v))]
    if not pos or not neg:
        return None
    combined = pos + neg
    ranks = _rankdata(combined)
    rank_pos_sum = sum(ranks[: len(pos)])
    n_pos, n_neg = len(pos), len(neg)
    return round((rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 6)


def pairwise_ordering(
    records: list[dict[str, Any]],
    score_key: str,
    target_key: str,
    *,
    same_question_only: bool = False,
) -> float | None:
    groups: list[list[dict[str, Any]]]
    if same_question_only:
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_question[str(record.get("source_question_id"))].append(record)
        groups = list(by_question.values())
    else:
        groups = [records]

    comparable = 0
    correct = 0.0
    for rows in groups:
        for i in range(len(rows)):
            si = _finite_float(rows[i].get(score_key))
            ti = _finite_float(rows[i].get(target_key))
            if si is None or ti is None:
                continue
            for j in range(i + 1, len(rows)):
                sj = _finite_float(rows[j].get(score_key))
                tj = _finite_float(rows[j].get(target_key))
                if sj is None or tj is None or ti == tj:
                    continue
                comparable += 1
                score_diff = si - sj
                target_diff = ti - tj
                if score_diff == 0:
                    correct += 0.5
                elif (score_diff > 0) == (target_diff > 0):
                    correct += 1.0
    if comparable == 0:
        return None
    return round(correct / comparable, 6)


def _surface_keyness_proxy(record: dict[str, Any]) -> float:
    span_type = str(record.get("span_type") or "unknown")
    span_text = str(record.get("span_text") or "")
    base = {
        "number": 0.86,
        "operation": 0.70,
        "comparison": 0.68,
        "condition": 0.64,
        "negation": 0.60,
        "question_target": 0.44,
        "object": 0.30,
    }.get(span_type, 0.35)
    if NUMBER_RE.search(span_text):
        base = max(base, 0.82)
    if not span_text.strip():
        base *= 0.75
    if len(span_text.split()) > 4:
        base *= 0.92
    return round(_clip01(base), 6)


def _solution_path_keyness(status: str | None) -> float | None:
    if status == "on_solution_path_number":
        return 1.0
    if status == "off_solution_path_number":
        return 0.0
    if status == "ambiguous_number":
        return 0.5
    return None


def _pre_recovery_raw(record: dict[str, Any]) -> float | None:
    features = record.get("pre_recovery_features") or {}
    keys = [
        "pre_delta_question_relnorm_mean",
        "pre_delta_question_relnorm_max",
        "pre_delta_question_l2_mean",
        "pre_delta_span_relnorm_mean",
        "pre_delta_span_relnorm_max",
        "pre_delta_span_l2_mean",
        "pre_span_context_shift_norm",
        "pre_question_layer_shift_norm",
    ]
    vals = [_finite_float(features.get(k)) for k in keys]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(abs(v) for v in vals) / len(vals)


def _attention_raw(record: dict[str, Any]) -> float | None:
    features = record.get("attention_features") or {}
    keys = [
        "attn_delta_slot_in_mass",
        "attn_delta_slot_in_rel",
        "attn_delta_slot_entropy",
        "attn_delta_slot_rank",
        "attn_delta_slot_top3_mass",
        "attn_delta_slot_edge_count",
        "attn_orig_numctx_to_slot",
        "attn_orig_operation_to_slot",
        "attn_orig_qfocus_to_slot",
    ]
    vals = [_finite_float(features.get(k)) for k in keys]
    vals = [abs(v) for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _prediction_index(records: list[dict[str, Any]], key: str = "masked_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in records if row.get(key) is not None}


def _ordinal_score_field(primary_method: str | None) -> str:
    return {
        "expected_bucket": "score_expected_bucket",
        "ordinal_threshold": "score_ordinal_threshold",
        "reg_calibrated": "score_reg_calibrated",
        "reg_raw": "score_reg_raw",
    }.get(str(primary_method or ""), "score_expected_bucket")


def collect_input_feature_names(records: list[dict[str, Any]]) -> list[str]:
    names = {
        "surf_is_numeric",
        "surf_span_char_len",
        "surf_span_word_len",
        "surf_question_char_len",
        "surf_span_rel_position",
        "surf_type_number",
        "surf_type_operation",
        "surf_type_condition",
        "surf_type_comparison",
        "surf_type_negation",
        "surf_type_object_or_query",
        "hidden_pre_recovery_pred_strength",
        "ordinal_priority_score",
    }
    for record in records:
        names.update((record.get("pre_recovery_features") or {}).keys())
        names.update((record.get("attention_features") or {}).keys())
    return sorted(names)


def audit_input_feature_names(feature_names: list[str]) -> dict[str, Any]:
    hits = []
    for name in feature_names:
        lower = name.lower()
        matched = [tok for tok in FORBIDDEN_INPUT_FEATURE_SUBSTRINGS if tok in lower]
        if matched:
            hits.append({"feature_name": name, "matched_substrings": matched})
    return {
        "backend": BACKEND,
        "forbidden_input_feature_substrings": list(FORBIDDEN_INPUT_FEATURE_SUBSTRINGS),
        "num_input_features_audited": len(feature_names),
        "leaked_input_features": hits,
        "passed": len(hits) == 0,
        "diagnostic_eval_only_fields": [
            "solution_path_keyness_diagnostic",
            "on_path_number",
            "off_path_number",
            "ambiguous_number",
            "recovery_fragility_diagnostic",
            "fragility_bucket",
            "risk_strength",
            "solution_path_status",
        ],
        "note": (
            "Forbidden substrings are disallowed only in formula input feature names. "
            "Recovery, bucket, risk, and solution-path fields are retained as diagnostic "
            "or evaluation-only labels and are not used by eligible formulas."
        ),
    }


def build_score_matrix(
    base_records: list[dict[str, Any]],
    *,
    ordinal_predictions: list[dict[str, Any]] | None = None,
    enriched_predictions: list[dict[str, Any]] | None = None,
    hidden_predictions: list[dict[str, Any]] | None = None,
    ordinal_primary_method: str | None = None,
) -> list[dict[str, Any]]:
    ordinal_by_id = _prediction_index(ordinal_predictions or [])
    enriched_by_id = _prediction_index(enriched_predictions or [])
    hidden_by_id = _prediction_index(hidden_predictions or [])
    ordinal_field = _ordinal_score_field(ordinal_primary_method)

    draft: list[dict[str, Any]] = []
    for idx, record in enumerate(base_records):
        masked_id = str(record.get("masked_id"))
        status = record.get("solution_path_status")
        ordinal = ordinal_by_id.get(masked_id, {})
        enriched = enriched_by_id.get(masked_id, {})
        hidden = hidden_by_id.get(masked_id, {})

        hidden_raw = _finite_float(enriched.get("pred_risk_strength"))
        if hidden_raw is None:
            hidden_raw = _finite_float(hidden.get("pred_risk_strength"))
        if hidden_raw is None:
            hidden_raw = _pre_recovery_raw(record)

        current_raw = _finite_float(ordinal.get(ordinal_field))
        if current_raw is None:
            current_raw = _finite_float(ordinal.get("score_expected_bucket"))
        if current_raw is None:
            current_raw = hidden_raw

        draft.append(
            {
                "audit_record_id": f"score_matrix_{idx:06d}",
                "masked_id": masked_id,
                "source_question_id": record.get("source_question_id"),
                "question": record.get("question"),
                "span_text": record.get("span_text"),
                "span_type": record.get("span_type"),
                "_current_raw": current_raw,
                "_hidden_raw": hidden_raw,
                "_attention_raw": _attention_raw(record),
                "surface_keyness_proxy": _surface_keyness_proxy(record),
                "solution_path_keyness_diagnostic": _solution_path_keyness(status),
                "on_path_number": status == "on_solution_path_number",
                "off_path_number": status == "off_solution_path_number",
                "ambiguous_number": status == "ambiguous_number",
                "recovery_fragility_diagnostic": _finite_float(record.get("risk_strength")),
                "fragility_bucket": record.get("fragility_bucket"),
                "risk_strength": _finite_float(record.get("risk_strength")),
                "solution_path_status": status,
                "ordinal_primary_method": ordinal_primary_method,
                "ordinal_score_field": ordinal_field,
                "ordinal_prediction_available": bool(ordinal),
                "hidden_prediction_available": bool(enriched or hidden),
            }
        )

    for raw_key, out_key in [
        ("_current_raw", "current_priority_score"),
        ("_hidden_raw", "hidden_fragility_score"),
        ("_attention_raw", "attention_fragility_score"),
    ]:
        scaled = _minmax([_finite_float(r.get(raw_key)) for r in draft])
        for record, score in zip(draft, scaled):
            record[out_key] = _round_or_none(score)

    for record in draft:
        h = _finite_float(record.get("hidden_fragility_score"), 0.0) or 0.0
        a = _finite_float(record.get("attention_fragility_score"), 0.0) or 0.0
        record["hidden_plus_attention_score"] = round((h + a) / 2.0, 6)

    _rank_desc(draft, "current_priority_score", "same_question_rank")
    for record in draft:
        current = _finite_float(record.get("current_priority_score"), 0.0) or 0.0
        record["off_path_budget_risk"] = round(current if record["off_path_number"] else 0.0, 6)
        record["keyness_signals"] = {
            "surface_keyness_proxy": record["surface_keyness_proxy"],
            "solution_path_keyness_diagnostic": record["solution_path_keyness_diagnostic"],
            "on_path_number": record["on_path_number"],
            "off_path_number": record["off_path_number"],
            "ambiguous_number": record["ambiguous_number"],
        }
        record["fragility_signals"] = {
            "recovery_fragility_diagnostic": record["recovery_fragility_diagnostic"],
            "hidden_fragility_score": record["hidden_fragility_score"],
            "attention_fragility_score": record["attention_fragility_score"],
            "hidden_plus_attention_score": record["hidden_plus_attention_score"],
        }
        record["reasoning_signals"] = {name: None for name in REASONING_SIGNAL_FIELDS}
        record["budget_signals"] = {
            "current_priority_score": record["current_priority_score"],
            "same_question_rank": record["same_question_rank"],
            "off_path_budget_risk": record["off_path_budget_risk"],
        }
        record["labels_for_evaluation_only"] = {
            "fragility_bucket": record["fragility_bucket"],
            "risk_strength": record["risk_strength"],
            "solution_path_status": record["solution_path_status"],
        }
        for raw_key in ["_current_raw", "_hidden_raw", "_attention_raw"]:
            record.pop(raw_key, None)
    return draft


def add_formula_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    attention_values = [
        _finite_float(r.get("attention_fragility_score"), 0.0) or 0.0 for r in records
    ]
    attention_median = sorted(attention_values)[len(attention_values) // 2] if attention_values else 0.0
    for record in records:
        keyness = _finite_float(record.get("surface_keyness_proxy"), 0.0) or 0.0
        hidden = _finite_float(record.get("hidden_fragility_score"), 0.0) or 0.0
        attention = _finite_float(record.get("attention_fragility_score"), 0.0) or 0.0
        current = _finite_float(record.get("current_priority_score"), 0.0) or 0.0
        bucket = _finite_float(record.get("fragility_bucket"), 0.0) or 0.0
        solution_key = _finite_float(record.get("solution_path_keyness_diagnostic"), 0.0)
        b_score = keyness * hidden
        c_score = hidden if keyness >= 0.55 else hidden * 0.25
        d_score = keyness * hidden * (0.5 + attention)
        f_penalty = 0.75 if record.get("span_type") == "number" and attention < attention_median else 1.0
        record["formula_scores"] = {
            "A_current_priority": round(current, 6),
            "B_keyness_times_fragility": round(b_score, 6),
            "C_keyness_gate_then_fragility": round(c_score, 6),
            "D_keyness_fragility_attention": round(d_score, 6),
            "F_offpath_proxy_penalty": round(b_score * f_penalty, 6),
            "surface_rule_baseline": round(keyness, 6),
            "hidden_pre_recovery_baseline": round(hidden, 6),
            "attention_pre_recovery_baseline": round(attention, 6),
            "hidden_plus_attention_baseline": round(
                _finite_float(record.get("hidden_plus_attention_score"), 0.0) or 0.0,
                6,
            ),
            "oracle_diagnostic_only": round(
                (solution_key if solution_key is not None else 0.0) * (bucket / 3.0),
                6,
            ),
        }

    _add_per_question_normalized_formula(records)
    _add_span_type_budget_cap_formula(records)

    return {
        "attention_median_for_proxy_penalty": round(attention_median, 6),
        "formulas": {
            "A_current_priority": {
                "description": "Current available per-record priority score from 2H-D ordinal predictions.",
                "uses_eval_only_labels": False,
            },
            "B_keyness_times_fragility": {
                "description": "surface_keyness_proxy * hidden_fragility_score",
                "uses_eval_only_labels": False,
            },
            "C_keyness_gate_then_fragility": {
                "description": "hidden fragility gated by surface keyness >= 0.55",
                "uses_eval_only_labels": False,
            },
            "D_keyness_fragility_attention": {
                "description": "surface keyness * hidden fragility * attention-delta modifier",
                "uses_eval_only_labels": False,
            },
            "E_per_question_normalized_priority": {
                "description": "per-question min-max normalized B score",
                "uses_eval_only_labels": False,
            },
            "F_offpath_proxy_penalty": {
                "description": "B score with a surface/attention-only number-span penalty; does not use off-path labels",
                "uses_eval_only_labels": False,
            },
            "G_span_type_budget_cap": {
                "description": "B score with per-question number-span cap based on surface span type and B rank",
                "uses_eval_only_labels": False,
            },
            "oracle_diagnostic_only": {
                "description": "Uses solution-path diagnostic and fragility bucket for upper-bound diagnosis only.",
                "uses_eval_only_labels": True,
            },
        },
    }


def _add_per_question_normalized_formula(records: list[dict[str, Any]]) -> None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    for rows in by_question.values():
        values = [
            _finite_float(r["formula_scores"].get("B_keyness_times_fragility"), 0.0)
            for r in rows
        ]
        scaled = _minmax(values)
        for row, score in zip(rows, scaled):
            row["formula_scores"]["E_per_question_normalized_priority"] = _round_or_none(score)


def _add_span_type_budget_cap_formula(records: list[dict[str, Any]]) -> None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    for rows in by_question.values():
        numbers = [r for r in rows if r.get("span_type") == "number"]
        numbers.sort(
            key=lambda r: -(_finite_float(r["formula_scores"].get("B_keyness_times_fragility"), 0.0) or 0.0)
        )
        number_rank = {id(row): idx + 1 for idx, row in enumerate(numbers)}
        for row in rows:
            b_score = _finite_float(row["formula_scores"].get("B_keyness_times_fragility"), 0.0) or 0.0
            if row.get("span_type") != "number":
                multiplier = 1.0
            else:
                rank = number_rank.get(id(row), 99)
                multiplier = 1.0 if rank == 1 else (0.70 if rank == 2 else 0.45)
            row["formula_scores"]["G_span_type_budget_cap"] = round(b_score * multiplier, 6)


def _score_values(records: list[dict[str, Any]], score_name: str) -> list[float | None]:
    out = []
    for record in records:
        if score_name in record:
            out.append(_finite_float(record.get(score_name)))
        else:
            out.append(_finite_float((record.get("formula_scores") or {}).get(score_name)))
    return out


def keyness_eval(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    subset = [
        r for r in records
        if r.get("solution_path_status") in {"on_solution_path_number", "off_solution_path_number"}
    ]
    out: dict[str, Any] = {
        "target": "on_path_number_vs_off_path_number",
        "num_number_eval_records": len(subset),
        "num_on_path": sum(1 for r in subset if r.get("on_path_number")),
        "num_off_path": sum(1 for r in subset if r.get("off_path_number")),
        "metrics_by_score": {},
    }
    for score_name in score_names:
        values = _score_values(subset, score_name)
        pos = [v for v, r in zip(values, subset) if r.get("on_path_number") and v is not None]
        neg = [v for v, r in zip(values, subset) if r.get("off_path_number") and v is not None]
        out["metrics_by_score"][score_name] = {
            "auc_on_vs_off": auc(pos, neg),
            "pairwise_on_gt_off_same_question": _on_off_pairwise(subset, score_name),
            "top1_on_path_coverage": _topk_on_path_coverage(subset, score_name, 1),
            "top2_on_path_coverage": _topk_on_path_coverage(subset, score_name, 2),
            "off_path_false_positive_rate_top20pct": _off_path_false_positive_rate(subset, score_name, 0.20),
            "mean_on_path_rank_among_number_spans": _mean_on_path_rank(subset, score_name),
        }
    return out


def _on_off_pairwise(records: list[dict[str, Any]], score_name: str) -> float | None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    comparable = 0
    correct = 0.0
    for rows in by_question.values():
        ons = [r for r in rows if r.get("on_path_number")]
        offs = [r for r in rows if r.get("off_path_number")]
        for on in ons:
            on_score = _score_values([on], score_name)[0]
            if on_score is None:
                continue
            for off in offs:
                off_score = _score_values([off], score_name)[0]
                if off_score is None:
                    continue
                comparable += 1
                if on_score == off_score:
                    correct += 0.5
                elif on_score > off_score:
                    correct += 1.0
    if comparable == 0:
        return None
    return round(correct / comparable, 6)


def _topk_on_path_coverage(records: list[dict[str, Any]], score_name: str, k: int) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    total = 0
    covered = 0
    for rows in by_question.values():
        if not any(r.get("on_path_number") for r in rows):
            continue
        total += 1
        ranked = sorted(
            rows,
            key=lambda r: -(_score_values([r], score_name)[0] if _score_values([r], score_name)[0] is not None else -1e12),
        )
        if any(r.get("on_path_number") for r in ranked[:k]):
            covered += 1
    return {"k": k, "questions": total, "covered": covered, "coverage": round(covered / total, 6) if total else None}


def _off_path_false_positive_rate(records: list[dict[str, Any]], score_name: str, fraction: float) -> float | None:
    scored = [
        (r, _score_values([r], score_name)[0])
        for r in records
        if _score_values([r], score_name)[0] is not None
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: -item[1])
    k = max(1, int(round(len(scored) * fraction)))
    top = scored[:k]
    return round(sum(1 for r, _ in top if r.get("off_path_number")) / k, 6)


def _mean_on_path_rank(records: list[dict[str, Any]], score_name: str) -> float | None:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    ranks = []
    for rows in by_question.values():
        ranked = sorted(
            rows,
            key=lambda r: -(_score_values([r], score_name)[0] if _score_values([r], score_name)[0] is not None else -1e12),
        )
        for idx, row in enumerate(ranked, start=1):
            if row.get("on_path_number"):
                ranks.append(idx)
    if not ranks:
        return None
    return round(sum(ranks) / len(ranks), 6)


def fragility_eval(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    subset = [r for r in records if _finite_float(r.get("fragility_bucket")) is not None]
    out: dict[str, Any] = {
        "target": "fragility_bucket",
        "num_eval_records": len(subset),
        "bucket_counts": dict(Counter(str(r.get("fragility_bucket")) for r in subset)),
        "metrics_by_score": {},
        "subset_metrics": {},
    }
    for score_name in score_names:
        values = _score_values(subset, score_name)
        buckets = [_finite_float(r.get("fragility_bucket")) for r in subset]
        out["metrics_by_score"][score_name] = {
            "spearman_vs_bucket": spearman([v for v in values], [b for b in buckets]),
            "wrong_vs_exact_auc_bucket3_vs_bucket1": auc(
                [v for v, b in zip(values, buckets) if b == 3 and v is not None],
                [v for v, b in zip(values, buckets) if b == 1 and v is not None],
            ),
            "generic_or_wrong_vs_exact_auc_bucket23_vs_bucket1": auc(
                [v for v, b in zip(values, buckets) if b is not None and b >= 2 and v is not None],
                [v for v, b in zip(values, buckets) if b == 1 and v is not None],
            ),
            "bucket3_vs_bucket1_auc": auc(
                [v for v, b in zip(values, buckets) if b == 3 and v is not None],
                [v for v, b in zip(values, buckets) if b == 1 and v is not None],
            ),
            "same_question_pairwise_bucket": pairwise_ordering(subset, score_name, "fragility_bucket", same_question_only=True),
            "global_pairwise_bucket": pairwise_ordering(subset, score_name, "fragility_bucket"),
        }

    for span_type in ["number", "comparison", "negation", "condition"]:
        rows = [r for r in subset if r.get("span_type") == span_type]
        if len(rows) < 5:
            out["subset_metrics"][span_type] = {"num_records": len(rows), "status": "too_few_records"}
            continue
        out["subset_metrics"][span_type] = {
            "num_records": len(rows),
            "metrics_by_score": {
                score_name: {
                    "spearman_vs_bucket": spearman(
                        [v for v in _score_values(rows, score_name)],
                        [_finite_float(r.get("fragility_bucket")) for r in rows],
                    ),
                    "bucket3_vs_bucket1_auc": auc(
                        [v for v, r in zip(_score_values(rows, score_name), rows) if r.get("fragility_bucket") == 3 and v is not None],
                        [v for v, r in zip(_score_values(rows, score_name), rows) if r.get("fragility_bucket") == 1 and v is not None],
                    ),
                }
                for score_name in score_names
            },
        }
    return out


def budget_priority_eval(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    out = {
        "target": "budget_priority_for_high_fragility_without_off_path_flooding",
        "num_records": len(records),
        "metrics_by_score": {},
    }
    for score_name in score_names:
        out["metrics_by_score"][score_name] = budget_metrics_for_score(records, score_name)
    return out


def budget_metrics_for_score(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    return {
        "per_question_top1_bucket3_hit": _per_question_topk_bucket_hit(records, score_name, 1),
        "per_question_top2_bucket3_hit": _per_question_topk_bucket_hit(records, score_name, 2),
        "per_question_top3_bucket3_hit": _per_question_topk_bucket_hit(records, score_name, 3),
        "top10pct_bucket3_precision": _top_bucket_precision(records, score_name, 0.10),
        "top20pct_bucket3_precision": _top_bucket_precision(records, score_name, 0.20),
        "top10_records_bucket3_precision": _top_bucket_precision(records, score_name, None, k=10),
        "top20_records_bucket3_precision": _top_bucket_precision(records, score_name, None, k=20),
        "off_path_budget_share": _off_path_budget_share(records, score_name),
        "same_question_pairwise_bucket": pairwise_ordering(records, score_name, "fragility_bucket", same_question_only=True),
        "error_breakdown": _budget_error_breakdown(records, score_name),
    }


def _per_question_topk_bucket_hit(records: list[dict[str, Any]], score_name: str, k: int) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    total = 0
    hit = 0
    for rows in by_question.values():
        if not any(r.get("fragility_bucket") == 3 for r in rows):
            continue
        total += 1
        ranked = sorted(
            rows,
            key=lambda r: -(_score_values([r], score_name)[0] if _score_values([r], score_name)[0] is not None else -1e12),
        )
        if any(r.get("fragility_bucket") == 3 for r in ranked[:k]):
            hit += 1
    return {"k": k, "questions_with_bucket3": total, "hits": hit, "hit_rate": round(hit / total, 6) if total else None}


def _top_bucket_precision(
    records: list[dict[str, Any]], score_name: str, fraction: float | None, *, k: int | None = None
) -> dict[str, Any]:
    scored = [
        (r, _score_values([r], score_name)[0])
        for r in records
        if _score_values([r], score_name)[0] is not None
    ]
    if not scored:
        return {"k": 0, "bucket3_precision": None, "bucket3_count": 0}
    scored.sort(key=lambda item: -item[1])
    if k is None:
        assert fraction is not None
        k = max(1, int(round(len(scored) * fraction)))
    k = min(k, len(scored))
    top = scored[:k]
    count = sum(1 for r, _ in top if r.get("fragility_bucket") == 3)
    return {"k": k, "bucket3_count": count, "bucket3_precision": round(count / k, 6)}


def _off_path_budget_share(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    total = 0.0
    off_path = 0.0
    for record in records:
        score = _score_values([record], score_name)[0]
        if score is None:
            continue
        score = max(0.0, float(score))
        total += score
        if record.get("off_path_number"):
            off_path += score
    return {
        "total_score_mass": round(total, 6),
        "off_path_number_score_mass": round(off_path, 6),
        "off_path_budget_share": round(off_path / total, 6) if total > 0 else None,
    }


def _budget_error_breakdown(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    scored = [
        (r, _score_values([r], score_name)[0])
        for r in records
        if _score_values([r], score_name)[0] is not None
    ]
    scored.sort(key=lambda item: -item[1])
    k = max(1, int(round(len(scored) * 0.20))) if scored else 0
    top = scored[:k]
    return {
        "top20pct_k": k,
        "top20pct_off_path_number": sum(1 for r, _ in top if r.get("off_path_number")),
        "top20pct_bucket0_or_1": sum(1 for r, _ in top if r.get("fragility_bucket") in {0, 1}),
        "top20pct_not_a_number": sum(1 for r, _ in top if r.get("solution_path_status") == "not_a_number"),
        "top20pct_span_type_counts": dict(Counter(str(r.get("span_type")) for r, _ in top)),
    }


def bootstrap_formula_deltas(
    records: list[dict[str, Any]],
    formula_names: list[str],
    *,
    baseline: str = "A_current_priority",
    num_bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("source_question_id"))].append(record)
    qids = sorted(by_question)
    if not qids:
        return {"num_bootstrap": 0, "baseline": baseline, "formula_deltas": {}}

    def metric_pack(rows: list[dict[str, Any]], score: str) -> dict[str, float | None]:
        m = budget_metrics_for_score(rows, score)
        return {
            "same_question_pairwise_bucket": m["same_question_pairwise_bucket"],
            "top10pct_bucket3_precision": m["top10pct_bucket3_precision"]["bucket3_precision"],
            "off_path_budget_share": m["off_path_budget_share"]["off_path_budget_share"],
        }

    deltas: dict[str, dict[str, list[float]]] = {
        name: {
            "same_question_pairwise_bucket": [],
            "top10pct_bucket3_precision": [],
            "off_path_budget_share": [],
        }
        for name in formula_names
        if name != baseline
    }
    for _ in range(num_bootstrap):
        sampled = []
        for _ in qids:
            sampled.extend(by_question[rng.choice(qids)])
        base = metric_pack(sampled, baseline)
        for formula in deltas:
            cand = metric_pack(sampled, formula)
            for metric in deltas[formula]:
                bv = base[metric]
                cv = cand[metric]
                if bv is None or cv is None:
                    continue
                deltas[formula][metric].append(float(cv) - float(bv))

    return {
        "baseline": baseline,
        "num_bootstrap": num_bootstrap,
        "formula_deltas": {
            formula: {metric: _summarize_bootstrap(vals) for metric, vals in metrics.items()}
            for formula, metrics in deltas.items()
        },
        "stability_rule": (
            "For ranking and precision, CI95 low > 0 indicates stable improvement. "
            "For off_path_budget_share, CI95 high < 0 indicates stable reduction."
        ),
    }


def _summarize_bootstrap(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"point": None, "ci95_low": None, "ci95_high": None, "stable_improvement": False}
    vals = sorted(values)
    low = vals[int(0.025 * (len(vals) - 1))]
    high = vals[int(0.975 * (len(vals) - 1))]
    point = sum(vals) / len(vals)
    return {
        "point": round(point, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "stable_positive_delta": bool(low > 0),
        "stable_negative_delta": bool(high < 0),
    }


def select_best_formula(formula_report: dict[str, Any]) -> str:
    metrics = formula_report["metrics_by_formula"]
    current = metrics.get("A_current_priority")
    if not current:
        return "A_current_priority"
    current_pair = current.get("same_question_pairwise_bucket")
    current_top10 = current["top10pct_bucket3_precision"]["bucket3_precision"]
    current_off = current["off_path_budget_share"]["off_path_budget_share"]
    candidates = [
        name
        for name in metrics
        if name not in {"A_current_priority", "oracle_diagnostic_only"}
    ]
    if not candidates:
        return "A_current_priority"

    def key(name: str) -> tuple[float, float, float]:
        row = metrics[name]
        pairwise = row.get("same_question_pairwise_bucket")
        top10 = row["top10pct_bucket3_precision"]["bucket3_precision"]
        offshare = row["off_path_budget_share"]["off_path_budget_share"]
        return (
            pairwise if pairwise is not None else -1.0,
            top10 if top10 is not None else -1.0,
            -(offshare if offshare is not None else 1.0),
        )

    def improves_current(name: str) -> bool:
        row = metrics[name]
        pairwise = row.get("same_question_pairwise_bucket")
        top10 = row["top10pct_bucket3_precision"]["bucket3_precision"]
        offshare = row["off_path_budget_share"]["off_path_budget_share"]
        pair_not_worse = (
            current_pair is None
            or pairwise is None
            or pairwise >= current_pair
        )
        top_not_worse = top10 is not None and current_top10 is not None and top10 >= current_top10
        off_not_worse = offshare is not None and current_off is not None and offshare <= current_off
        strictly_better = (
            (pairwise is not None and current_pair is not None and pairwise > current_pair)
            or (top10 is not None and current_top10 is not None and top10 > current_top10)
            or (offshare is not None and current_off is not None and offshare < current_off)
        )
        return pair_not_worse and top_not_worse and off_not_worse and strictly_better

    viable = [name for name in candidates if improves_current(name)]
    if not viable:
        return "A_current_priority"
    return max(viable, key=key)


def formula_simulation_report(records: list[dict[str, Any]], formula_metadata: dict[str, Any]) -> dict[str, Any]:
    formula_names = list(formula_metadata["formulas"].keys())
    metrics = {
        name: budget_metrics_for_score(records, name)
        for name in formula_names
    }
    by_question: dict[str, int] = defaultdict(int)
    for record in records:
        by_question[str(record.get("source_question_id"))] += 1
    report = {
        "backend": BACKEND,
        "baseline": "A_current_priority",
        "formula_metadata": formula_metadata,
        "metrics_by_formula": metrics,
        "same_question_rank_diagnostic": {
            "num_questions": len(by_question),
            "num_questions_with_multiple_scored_spans": sum(1 for count in by_question.values() if count > 1),
            "same_question_pairwise_available": any(count > 1 for count in by_question.values()),
            "interpretation": (
                "Current 500-case score-matrix inputs contain one scored span per question, "
                "so same-question pairwise ranking metrics are unavailable in this audit."
            ),
        },
        "baselines_compared": [
            "surface_rule_baseline",
            "hidden_pre_recovery_baseline",
            "attention_pre_recovery_baseline",
            "hidden_plus_attention_baseline",
            "A_current_priority",
        ],
        "unavailable_baselines": [
            "2I primary per-record OOF score is not present in existing 2I outputs; only aggregate 2I reports are available."
        ],
    }
    report["best_non_oracle_formula"] = select_best_formula(report)
    return report


def reasoning_signal_gap(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        field: sum(1 for r in records if (r.get("reasoning_signals") or {}).get(field) is not None)
        for field in REASONING_SIGNAL_FIELDS
    }
    return {
        "backend": BACKEND,
        "num_records": len(records),
        "reasoning_signal_non_null_counts": counts,
        "all_reasoning_signals_missing": all(count == 0 for count in counts.values()),
        "missing_signal_fields": list(REASONING_SIGNAL_FIELDS),
        "interpretation": (
            "Current 2H/2I outputs contain span surface, hidden-state perturbation, and attention-map summaries, "
            "but no answer-logprob, trajectory-change, CoT-path, or NLA semantic-role measurements. "
            "Ranking failures therefore cannot be resolved with true reasoning-path evidence in this audit."
        ),
        "recommended_next_signal_sprint": (
            "Add reasoning-path or answer-stability features before any 2000 rerun or Sprint 3A steering attempt."
        ),
    }


def topk_case_exports(
    records: list[dict[str, Any]],
    *,
    score_name: str,
    limit: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        records,
        key=lambda r: -(_score_values([r], score_name)[0] if _score_values([r], score_name)[0] is not None else -1e12),
    )
    failures = []
    successes = []
    for idx, record in enumerate(ranked, start=1):
        score = _score_values([record], score_name)[0]
        case = _case_record(record, score_name, score, idx)
        if _is_success_case(record) and len(successes) < limit:
            successes.append(case)
        if _is_failure_case(record) and len(failures) < limit:
            failures.append(case)
        if len(successes) >= limit and len(failures) >= limit:
            break
    if len(failures) < limit:
        missed = [
            r for r in reversed(ranked)
            if r.get("fragility_bucket") == 3 or r.get("on_path_number")
        ]
        for record in missed:
            if len(failures) >= limit:
                break
            score = _score_values([record], score_name)[0]
            failures.append(
                _case_record(record, score_name, score, ranked.index(record) + 1, reason_override="important_span_ranked_low")
            )
    return failures[:limit], successes[:limit]


def _is_success_case(record: dict[str, Any]) -> bool:
    return bool(record.get("fragility_bucket") == 3 or record.get("on_path_number"))


def _is_failure_case(record: dict[str, Any]) -> bool:
    return bool(
        record.get("off_path_number")
        or record.get("fragility_bucket") in {0, 1}
        or record.get("solution_path_status") == "ambiguous_number"
    )


def _case_record(
    record: dict[str, Any],
    score_name: str,
    score: float | None,
    rank: int,
    *,
    reason_override: str | None = None,
) -> dict[str, Any]:
    reason = reason_override
    if reason is None:
        if record.get("off_path_number"):
            reason = "off_path_number_ranked_high"
        elif record.get("fragility_bucket") in {0, 1}:
            reason = "low_fragility_bucket_ranked_high"
        elif record.get("solution_path_status") == "ambiguous_number":
            reason = "ambiguous_number_ranked_high"
        else:
            reason = "high_fragility_or_on_path_ranked_high"
    return {
        "masked_id": record.get("masked_id"),
        "source_question_id": record.get("source_question_id"),
        "question": record.get("question"),
        "span_text": record.get("span_text"),
        "span_type": record.get("span_type"),
        "score_name": score_name,
        "score": _round_or_none(score),
        "rank": rank,
        "expected": {
            "fragility_bucket": record.get("fragility_bucket"),
            "solution_path_status": record.get("solution_path_status"),
            "risk_strength": record.get("risk_strength"),
        },
        "scores": {
            "surface_keyness_proxy": record.get("surface_keyness_proxy"),
            "hidden_fragility_score": record.get("hidden_fragility_score"),
            "attention_fragility_score": record.get("attention_fragility_score"),
            "hidden_plus_attention_score": record.get("hidden_plus_attention_score"),
            "current_priority_score": record.get("current_priority_score"),
            score_name: _round_or_none(score),
        },
        "failure_reason_auto_guess": reason,
    }


def root_cause_decision_table(
    *,
    feature_audit: dict[str, Any],
    keyness_report: dict[str, Any],
    fragility_report: dict[str, Any],
    formula_report: dict[str, Any],
    bootstrap_report: dict[str, Any],
    reasoning_gap_report: dict[str, Any],
) -> dict[str, Any]:
    best = formula_report["best_non_oracle_formula"]
    best_metrics = formula_report["metrics_by_formula"][best]
    current_metrics = formula_report["metrics_by_formula"]["A_current_priority"]
    best_pair = best_metrics.get("same_question_pairwise_bucket")
    current_pair = current_metrics.get("same_question_pairwise_bucket")
    boot_best = bootstrap_report["formula_deltas"].get(best, {})
    stable_pair = (boot_best.get("same_question_pairwise_bucket") or {}).get("stable_positive_delta") is True
    stable_top = (boot_best.get("top10pct_bucket3_precision") or {}).get("stable_positive_delta") is True
    stable_off = (boot_best.get("off_path_budget_share") or {}).get("stable_negative_delta") is True
    ready = bool(
        feature_audit.get("passed")
        and stable_pair
        and stable_top
        and stable_off
        and best_pair is not None
        and current_pair is not None
        and best_pair > current_pair + 0.05
    )
    decisions = [
        {
            "hypothesis": "same-question ranking can be diagnosed from the current 500-case matrix",
            "evidence": formula_report.get("same_question_rank_diagnostic", {}),
            "verdict": (
                "reject"
                if not formula_report.get("same_question_rank_diagnostic", {}).get("same_question_pairwise_available")
                else "supported"
            ),
            "next_action": "build a multi-span-per-question weak-labeled matrix before claiming same-question ranking gains",
        },
        {
            "hypothesis": "ranking failure is caused by feature leakage",
            "evidence": f"leaked_input_features={len(feature_audit.get('leaked_input_features', []))}",
            "verdict": "reject" if feature_audit.get("passed") else "confirm",
            "next_action": "block formula use until leakage is removed" if not feature_audit.get("passed") else "continue audit",
        },
        {
            "hypothesis": "surface keyness alone can identify solution-path numbers",
            "evidence": keyness_report["metrics_by_score"].get("surface_keyness_proxy", {}),
            "verdict": "partial",
            "next_action": "do not treat surface keyness as sufficient for 2000 rerun",
        },
        {
            "hypothesis": "hidden/attention fragility is enough for budget ranking",
            "evidence": {
                "best_formula": best,
                "best_same_question_pairwise": best_pair,
                "current_same_question_pairwise": current_pair,
                "bootstrap_pairwise_stable": stable_pair,
            },
            "verdict": "partial" if stable_pair else "not_stable",
            "next_action": "validate reasoning-path signals before scale-up",
        },
        {
            "hypothesis": "missing reasoning-path signals are a root cause",
            "evidence": reasoning_gap_report["reasoning_signal_non_null_counts"],
            "verdict": "supported" if reasoning_gap_report["all_reasoning_signals_missing"] else "inconclusive",
            "next_action": reasoning_gap_report["recommended_next_signal_sprint"],
        },
    ]
    return {
        "backend": BACKEND,
        "best_non_oracle_formula": best,
        "ready_for_2000_rerun": ready,
        "do_not_enter_sprint_3A": True,
        "decision_rows": decisions,
        "gate_rule": (
            "ready_for_2000_rerun may become true only when leakage audit passes and bootstrap-stable "
            "improvements are observed for same-question ranking, top-k bucket-3 precision, and off-path budget share."
        ),
        "recommended_next_step": (
            "Sprint 2J reasoning-path / answer-stability feature sprint"
            if not ready
            else "formula-validation sprint before any 2000 rerun; still do not enter Sprint 3A directly"
        ),
    }


def render_review_markdown(
    *,
    root_cause: dict[str, Any],
    keyness_report: dict[str, Any],
    fragility_report: dict[str, Any],
    budget_report: dict[str, Any],
    formula_report: dict[str, Any],
    bootstrap_report: dict[str, Any],
    reasoning_gap_report: dict[str, Any],
) -> str:
    best = root_cause["best_non_oracle_formula"]
    best_metrics = formula_report["metrics_by_formula"][best]
    current_metrics = formula_report["metrics_by_formula"]["A_current_priority"]
    lines = [
        "# Sprint 2I-R Score Matrix Audit",
        "",
        "## Verdict",
        "",
        f"- ready_for_2000_rerun: `{str(root_cause['ready_for_2000_rerun']).lower()}`",
        "- do_not_enter_sprint_3A: `true`",
        f"- best_non_oracle_formula: `{best}`",
        f"- recommended_next_step: {root_cause['recommended_next_step']}",
        "",
        "## Ten Audit Questions",
        "",
        "1. What does the current score contain? Current per-record priority uses existing 2H-D ordinal predictions; 2I has only aggregate per-method reports, not per-record OOF scores.",
        f"2. Does keyness separate on-path from off-path numbers? Surface keyness AUC: `{keyness_report['metrics_by_score'].get('surface_keyness_proxy', {}).get('auc_on_vs_off')}`.",
        f"3. Does fragility rank severe buckets? Hidden-plus-attention bucket3-vs-bucket1 AUC: `{fragility_report['metrics_by_score'].get('hidden_plus_attention_score', {}).get('bucket3_vs_bucket1_auc')}`.",
        f"4. Does budget priority improve same-question ordering? Current `{current_metrics.get('same_question_pairwise_bucket')}`, best `{best_metrics.get('same_question_pairwise_bucket')}`; availability `{formula_report.get('same_question_rank_diagnostic', {}).get('same_question_pairwise_available')}`.",
        f"5. Is off-path budget share controlled? Current `{current_metrics['off_path_budget_share']['off_path_budget_share']}`, best `{best_metrics['off_path_budget_share']['off_path_budget_share']}`.",
        f"6. Are reasoning-path signals available? `{not reasoning_gap_report['all_reasoning_signals_missing']}`; missing fields: `{', '.join(reasoning_gap_report['missing_signal_fields'])}`.",
        f"7. Which formula is strongest? `{best}` under the local non-oracle metrics.",
        "8. Is the formula bootstrap-stable? See `formula_bootstrap_report.json`; all required stability gates must pass before scale-up.",
        "9. Is there input-feature leakage? See `score_matrix_feature_audit.json`; eval-only labels are separated from eligible formula inputs.",
        f"10. Should the project rerun 2000 or enter Sprint 3A? `{str(root_cause['ready_for_2000_rerun']).lower()}` for 2000; never enter Sprint 3A directly from this audit.",
        "",
        "## Root Cause Table",
        "",
    ]
    for row in root_cause["decision_rows"]:
        lines.append(f"- `{row['verdict']}`: {row['hypothesis']} -> {row['next_action']}")
    lines.extend(
        [
            "",
        "## Formula Snapshot",
        "",
        f"- same-question diagnostic: {formula_report.get('same_question_rank_diagnostic', {}).get('interpretation')}",
        "",
        "| formula | same_question_pairwise | top10pct_bucket3_precision | off_path_budget_share |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in formula_report["metrics_by_formula"].items():
        lines.append(
            "| {name} | {pair} | {top} | {off} |".format(
                name=name,
                pair=metrics.get("same_question_pairwise_bucket"),
                top=metrics["top10pct_bucket3_precision"]["bucket3_precision"],
                off=metrics["off_path_budget_share"]["off_path_budget_share"],
            )
        )
    lines.extend(
        [
            "",
            "## Bootstrap",
            "",
            f"- baseline: `{bootstrap_report['baseline']}`",
            f"- num_bootstrap: `{bootstrap_report['num_bootstrap']}`",
            "- Stability rule: positive CI95 for pairwise/top-k and negative CI95 for off-path budget share.",
            "",
            "## Boundary",
            "",
            "- No recovery rerun.",
            "- No hidden-state cache rerun.",
            "- No attention cache rerun.",
            "- No probe training.",
            "- No attention steering.",
            "- No 2000-scale rerun.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_score_matrix_audit(
    *,
    risk_strength_path: str | Path,
    pre_recovery_feature_path: str | Path,
    ordinal_predictions_path: str | Path,
    ordinal_report_path: str | Path,
    attention_feature_path: str | Path,
    attention_report_path: str | Path,
    output_dir: str | Path,
    enriched_predictions_path: str | Path | None = None,
    hidden_predictions_path: str | Path | None = None,
    backend: str = BACKEND,
    overwrite: bool = False,
    bootstrap_samples: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    if backend != BACKEND:
        raise ValueError(f"Unsupported backend {backend!r}; expected {BACKEND!r}")
    out_dir = Path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {out_dir}")
    ensure_dir(out_dir)

    risk_records = read_jsonl(risk_strength_path)
    pre_records = read_jsonl(pre_recovery_feature_path)
    attention_records = read_jsonl(attention_feature_path)
    ordinal_predictions = read_jsonl(ordinal_predictions_path)
    ordinal_report = read_json(ordinal_report_path)
    attention_report = read_json(attention_report_path)
    enriched_predictions = read_jsonl(enriched_predictions_path) if enriched_predictions_path and Path(enriched_predictions_path).exists() else []
    hidden_predictions = read_jsonl(hidden_predictions_path) if hidden_predictions_path and Path(hidden_predictions_path).exists() else []

    # Prefer the richest 2I rows, but retain the required 2H inputs in provenance.
    base_records = attention_records if attention_records else pre_records
    if not base_records:
        base_records = risk_records

    input_feature_names = collect_input_feature_names(base_records)
    feature_audit = audit_input_feature_names(input_feature_names)
    feature_audit.update(
        {
            "input_paths": {
                "risk_strength_dataset": str(risk_strength_path),
                "pre_recovery_feature_dataset": str(pre_recovery_feature_path),
                "ordinal_predictions": str(ordinal_predictions_path),
                "attention_feature_dataset": str(attention_feature_path),
            },
            "num_risk_records": len(risk_records),
            "num_pre_recovery_records": len(pre_records),
            "num_attention_records": len(attention_records),
            "num_ordinal_prediction_records": len(ordinal_predictions),
            "attention_report_primary_method": attention_report.get("primary_method"),
            "ordinal_report_primary_method": ordinal_report.get("primary_method"),
        }
    )

    matrix_records = build_score_matrix(
        base_records,
        ordinal_predictions=ordinal_predictions,
        enriched_predictions=enriched_predictions,
        hidden_predictions=hidden_predictions,
        ordinal_primary_method=ordinal_report.get("primary_method"),
    )
    formula_metadata = add_formula_scores(matrix_records)

    base_score_names = [
        "surface_keyness_proxy",
        "hidden_fragility_score",
        "attention_fragility_score",
        "hidden_plus_attention_score",
        "current_priority_score",
    ]
    formula_names = [
        "A_current_priority",
        "B_keyness_times_fragility",
        "C_keyness_gate_then_fragility",
        "D_keyness_fragility_attention",
        "E_per_question_normalized_priority",
        "F_offpath_proxy_penalty",
        "G_span_type_budget_cap",
        "surface_rule_baseline",
        "hidden_pre_recovery_baseline",
        "attention_pre_recovery_baseline",
        "hidden_plus_attention_baseline",
        "oracle_diagnostic_only",
    ]
    keyness_report = keyness_eval(matrix_records, base_score_names + formula_names)
    fragility_report = fragility_eval(matrix_records, base_score_names + formula_names)
    budget_report = budget_priority_eval(matrix_records, formula_names)
    formula_report = formula_simulation_report(matrix_records, formula_metadata)
    bootstrap_report = bootstrap_formula_deltas(
        matrix_records,
        formula_names=[
            "A_current_priority",
            "B_keyness_times_fragility",
            "C_keyness_gate_then_fragility",
            "D_keyness_fragility_attention",
            "E_per_question_normalized_priority",
            "F_offpath_proxy_penalty",
            "G_span_type_budget_cap",
        ],
        num_bootstrap=bootstrap_samples,
        seed=seed,
    )
    reasoning_gap_report = reasoning_signal_gap(matrix_records)
    root_cause = root_cause_decision_table(
        feature_audit=feature_audit,
        keyness_report=keyness_report,
        fragility_report=fragility_report,
        formula_report=formula_report,
        bootstrap_report=bootstrap_report,
        reasoning_gap_report=reasoning_gap_report,
    )
    failures, successes = topk_case_exports(
        matrix_records,
        score_name=formula_report["best_non_oracle_formula"],
        limit=30,
    )

    write_jsonl(matrix_records, out_dir / "score_matrix_dataset.jsonl")
    write_json(feature_audit, out_dir / "score_matrix_feature_audit.json")
    write_json(keyness_report, out_dir / "keyness_eval_report.json")
    write_json(fragility_report, out_dir / "fragility_eval_report.json")
    write_json(budget_report, out_dir / "budget_priority_eval_report.json")
    write_jsonl(failures, out_dir / "topk_failure_cases.jsonl")
    write_jsonl(successes, out_dir / "topk_success_cases.jsonl")
    write_json(formula_report, out_dir / "formula_simulation_report.json")
    write_json(bootstrap_report, out_dir / "formula_bootstrap_report.json")
    write_json(reasoning_gap_report, out_dir / "reasoning_signal_gap_analysis.json")
    write_json(root_cause, out_dir / "root_cause_decision_table.json")
    (out_dir / "review_gate_score_matrix_audit.md").write_text(
        render_review_markdown(
            root_cause=root_cause,
            keyness_report=keyness_report,
            fragility_report=fragility_report,
            budget_report=budget_report,
            formula_report=formula_report,
            bootstrap_report=bootstrap_report,
            reasoning_gap_report=reasoning_gap_report,
        ),
        encoding="utf-8",
    )

    return {
        "backend": BACKEND,
        "output_dir": str(out_dir),
        "num_score_matrix_records": len(matrix_records),
        "feature_audit_passed": feature_audit["passed"],
        "best_non_oracle_formula": formula_report["best_non_oracle_formula"],
        "ready_for_2000_rerun": root_cause["ready_for_2000_rerun"],
        "topk_failure_cases": len(failures),
        "topk_success_cases": len(successes),
        "outputs": {
            "score_matrix_dataset": str(out_dir / "score_matrix_dataset.jsonl"),
            "score_matrix_feature_audit": str(out_dir / "score_matrix_feature_audit.json"),
            "keyness_eval_report": str(out_dir / "keyness_eval_report.json"),
            "fragility_eval_report": str(out_dir / "fragility_eval_report.json"),
            "budget_priority_eval_report": str(out_dir / "budget_priority_eval_report.json"),
            "topk_failure_cases": str(out_dir / "topk_failure_cases.jsonl"),
            "topk_success_cases": str(out_dir / "topk_success_cases.jsonl"),
            "formula_simulation_report": str(out_dir / "formula_simulation_report.json"),
            "formula_bootstrap_report": str(out_dir / "formula_bootstrap_report.json"),
            "reasoning_signal_gap_analysis": str(out_dir / "reasoning_signal_gap_analysis.json"),
            "root_cause_decision_table": str(out_dir / "root_cause_decision_table.json"),
            "review_gate_score_matrix_audit": str(out_dir / "review_gate_score_matrix_audit.md"),
        },
    }
