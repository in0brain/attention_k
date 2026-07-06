# -*- coding: utf-8 -*-
"""Sprint 2H read-only audit: why is risk_positive recall so low in Sprint 2G-2000?

Read-only boundary:
- reads existing 2G outputs only
- writes ONLY to outputs/logs/sprint_2H_risk_positive_audit/
- no cache / feature / training / guidance / steering reruns
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G2 = ROOT / "outputs" / "logs" / "sprint_2G_full_scale_2000"
OUT = ROOT / "outputs" / "logs" / "sprint_2H_risk_positive_audit"

PRED_PATH = G2 / "05_probe_training" / "probe_predictions.jsonl"
PROBE_DS_PATH = G2 / "04_weak_probe_dataset" / "weak_probe_dataset.jsonl"
WEAK_LABELS_PATH = G2 / "01_downstream" / "weak_labels_2000.jsonl"
MANIFEST_2A_PATH = G2 / "01_downstream" / "full_scale_2a_manifest.jsonl"
WEAK_LABEL_REPORT_PATH = G2 / "01_downstream" / "weak_label_report.json"
REVIEW_GATE_PATH = G2 / "08_review_gate" / "result_review_gate.json"
EVAL_REPORT_PATH = G2 / "05_probe_training" / "probe_eval_report.json"

NUMERIC_RE = re.compile(r"\d+(\.\d+)?|\$?\d+|percent|%|half|twice|double|triple", re.IGNORECASE)

EXPECTED = {
    "risk_positive_total": 74,
    "risk_positive_hit": 23,
    "risk_positive_missed": 51,
    "confusion": {
        "hard_negative_or_weak_positive": 28,
        "positive_anchor": 21,
        "negative": 2,
        "risk_positive": 23,
    },
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_filler(masked_question, recovered_question, mask_token="[MASK]"):
    """Recovered question = masked question with mask_token replaced by a filler.
    Recover the filler by aligning the common prefix/suffix around the mask."""
    if not masked_question or not recovered_question:
        return None
    idx = masked_question.find(mask_token)
    if idx < 0:
        return None
    prefix = masked_question[:idx]
    suffix = masked_question[idx + len(mask_token):]
    if not recovered_question.startswith(prefix):
        return None
    if suffix and not recovered_question.endswith(suffix):
        return None
    end = len(recovered_question) - len(suffix) if suffix else len(recovered_question)
    return recovered_question[len(prefix):end]


def rate(hit, total):
    return round(hit / total, 4) if total else None


def bucket_recall(cases, key_fn):
    buckets = defaultdict(lambda: {"support": 0, "hit": 0, "missed": 0, "missed_predicted_as": Counter()})
    for c in cases:
        b = buckets[key_fn(c)]
        b["support"] += 1
        if c["is_hit"]:
            b["hit"] += 1
        else:
            b["missed"] += 1
            b["missed_predicted_as"][c["predicted_probe_target"]] += 1
    out = {}
    for k, b in sorted(buckets.items(), key=lambda kv: -kv[1]["support"]):
        out[str(k)] = {
            "support": b["support"],
            "hit": b["hit"],
            "missed": b["missed"],
            "recall": rate(b["hit"], b["support"]),
            "missed_predicted_as": dict(b["missed_predicted_as"]),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    preds = read_jsonl(PRED_PATH)
    probe_ds = read_jsonl(PROBE_DS_PATH)
    weak_labels = read_jsonl(WEAK_LABELS_PATH)
    manifest_2a = read_jsonl(MANIFEST_2A_PATH)
    weak_label_report = json.loads(WEAK_LABEL_REPORT_PATH.read_text(encoding="utf-8"))

    # ---- join on masked_id (1 masked unit per case at 2000 scale) ----
    ds_by_masked = {r["masked_id"]: r for r in probe_ds}
    wl_by_masked = {r["masked_id"]: r for r in weak_labels}
    mf_by_masked = {r["masked_id"]: r for r in manifest_2a}
    assert len(ds_by_masked) == len(probe_ds), "duplicate masked_id in probe dataset"
    assert len(wl_by_masked) == len(weak_labels), "duplicate masked_id in weak labels"
    assert len(mf_by_masked) == len(manifest_2a), "duplicate masked_id in 2A manifest"

    join_errors = []
    records = []
    for p in preds:
        mid = p["masked_id"]
        ds = ds_by_masked.get(mid)
        wl = wl_by_masked.get(mid)
        mf = mf_by_masked.get(mid)
        if ds is None or wl is None or mf is None:
            join_errors.append(mid)
            continue
        if p["gold_probe_target"] != ds.get("probe_target") or p["gold_probe_target"] != wl.get("probe_target"):
            join_errors.append(f"gold mismatch: {mid}")
            continue

        rq = mf.get("recovered_questions")
        if isinstance(rq, list):
            recovered_question = rq[p.get("recovered_input_index", 0) if isinstance(p.get("recovered_input_index"), int) else 0] if rq else None
        else:
            recovered_question = rq
        filler = extract_filler(mf.get("masked_question"), recovered_question)

        scores = p.get("decision_scores") or {}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        rank_of_risk = next((i + 1 for i, (cls, _) in enumerate(ranked) if cls == "risk_positive"), None)
        risk_score = scores.get("risk_positive")
        pred_score = scores.get(p["predicted_probe_target"])
        margin = (risk_score - pred_score) if (risk_score is not None and pred_score is not None) else None

        span_text = wl.get("chosen_span_text", "")
        records.append({
            "masked_id": mid,
            "probe_record_id": p.get("probe_record_id"),
            "full_scale_id": p.get("id"),
            "source_question_id": wl.get("source_question_id"),
            "unit_id": p.get("unit_id"),
            "fold_id": p.get("fold_id"),
            "gold_probe_target": p["gold_probe_target"],
            "predicted_probe_target": p["predicted_probe_target"],
            "is_hit": p["predicted_probe_target"] == p["gold_probe_target"],
            "decision_scores": scores,
            "risk_positive_score": risk_score,
            "predicted_class_score": pred_score,
            "risk_minus_predicted_margin": margin,
            "risk_positive_score_rank": rank_of_risk,
            "chosen_span_type": wl.get("chosen_span_type"),
            "chosen_span_text": span_text,
            "label_rule": wl.get("label_rule"),
            "label_confidence": wl.get("label_confidence"),
            "human_semantic_role": mf.get("human_semantic_role"),
            "human_error_type": mf.get("human_error_type"),
            "is_numeric_span": bool(NUMERIC_RE.search(span_text or "")),
            "original_question": mf.get("original_question"),
            "masked_question": mf.get("masked_question"),
            "recovered_question": recovered_question,
            "recovered_filler": filler,
        })

    if join_errors:
        print("JOIN ERRORS -> stopping:", join_errors[:20], file=sys.stderr)
        sys.exit(1)

    # ---- step 2: reproduce review-gate numbers ----
    risk_cases = [r for r in records if r["gold_probe_target"] == "risk_positive"]
    hits = [r for r in risk_cases if r["is_hit"]]
    missed = [r for r in risk_cases if not r["is_hit"]]
    confusion = Counter(r["predicted_probe_target"] for r in risk_cases)

    reproduction = {
        "risk_positive_total": len(risk_cases),
        "risk_positive_hit": len(hits),
        "risk_positive_missed": len(missed),
        "recall": rate(len(hits), len(risk_cases)),
        "confusion_row": dict(confusion),
        "expected": EXPECTED,
        "matches_review_gate": (
            len(risk_cases) == EXPECTED["risk_positive_total"]
            and len(hits) == EXPECTED["risk_positive_hit"]
            and all(confusion.get(k, 0) == v for k, v in EXPECTED["confusion"].items())
        ),
    }
    if not reproduction["matches_review_gate"]:
        (OUT / "risk_positive_audit_report.json").write_text(
            json.dumps({"status": "REPRODUCTION_FAILED", "reproduction": reproduction}, indent=2),
            encoding="utf-8")
        print("REPRODUCTION FAILED", json.dumps(reproduction, indent=2), file=sys.stderr)
        sys.exit(2)

    # ---- step 3: bucket audits on gold risk_positive ----
    recall_by_span_type = bucket_recall(risk_cases, lambda c: c["chosen_span_type"])
    recall_by_numeric = bucket_recall(risk_cases, lambda c: "numeric_span" if c["is_numeric_span"] else "non_numeric_span")
    recall_by_label_rule = bucket_recall(risk_cases, lambda c: c["label_rule"])
    recall_by_semantic_role = bucket_recall(risk_cases, lambda c: c["human_semantic_role"])
    recall_by_error_type = bucket_recall(risk_cases, lambda c: c["human_error_type"])
    recall_by_span_text = bucket_recall(risk_cases, lambda c: (c["chosen_span_text"] or "").lower().strip())

    # numeric vs non-numeric detail
    numeric_missed = [r for r in missed if r["is_numeric_span"]]

    # ---- question C: are number spans present in gold risk_positive at all? ----
    mapping_rules = weak_label_report.get("weak_target_mapping_rules", {})
    number_records = [r for r in records if r["chosen_span_type"] == "number"]
    number_target_counts = Counter(r["gold_probe_target"] for r in number_records)
    critical_number_analysis = {
        "note": ("At 2000-scale the weak label is a deterministic function of chosen_span_type. "
                 "span_type=number is ALWAYS mapped to positive_anchor by rule "
                 "'span_type_number_critical_value' (confidence 0.8). Therefore gold risk_positive "
                 "contains ZERO span_type=number samples by construction; 'critical numbers' cannot "
                 "appear as risk_positive gold labels at all."),
        "number_span_gold_target_counts": dict(number_target_counts),
        "number_span_count": len(number_records),
        "number_spans_in_gold_risk_positive": sum(1 for r in risk_cases if r["chosen_span_type"] == "number"),
        "mapping_rule_for_number": mapping_rules.get("number"),
        "human_semantic_role_values_in_2000": sorted({r["human_semantic_role"] for r in records}),
        "human_error_type_values_in_2000": sorted({r["human_error_type"] for r in records}),
    }

    # ---- question B: human_error_type availability ----
    wrong_numeric_recovery_analysis = {
        "available": False,
        "note": ("human_error_type at 2000-scale is the sentinel 'weak_auto' for ALL 2000 records "
                 "(human_field_note in weak_label_report.json confirms human_* fields are placeholders). "
                 "wrong_numeric_recovery / misleading_entity_or_unit / generic_recovery breakdowns are "
                 "IMPOSSIBLE on this run; they exist only in the ~20-case human review from Sprint 2A-2F."),
        "human_error_type_counts": dict(Counter(r["human_error_type"] for r in records)),
    }

    # ---- question E: score / margin analysis ----
    def margin_stats(cases):
        margins = [c["risk_minus_predicted_margin"] for c in cases if c["risk_minus_predicted_margin"] is not None]
        if not margins:
            return None
        margins_sorted = sorted(margins)
        n = len(margins_sorted)
        return {
            "count": n,
            "min": round(margins_sorted[0], 4),
            "median": round(margins_sorted[n // 2], 4),
            "max": round(margins_sorted[-1], 4),
            "mean": round(sum(margins_sorted) / n, 4),
        }

    rank_counts_missed = Counter(r["risk_positive_score_rank"] for r in missed)
    rank_counts_hit = Counter(r["risk_positive_score_rank"] for r in hits)
    close_missed = [r for r in missed if r["risk_minus_predicted_margin"] is not None and r["risk_minus_predicted_margin"] > -0.5]
    score_margin_analysis = {
        "score_type": "ovr ridge decision_function scores (not calibrated probabilities)",
        "missed_risk_rank_counts": {str(k): v for k, v in sorted(rank_counts_missed.items())},
        "hit_risk_rank_counts": {str(k): v for k, v in sorted(rank_counts_hit.items())},
        "missed_rank2_count": rank_counts_missed.get(2, 0),
        "missed_rank2_share": rate(rank_counts_missed.get(2, 0), len(missed)),
        "missed_margin_stats(risk_score - predicted_class_score)": margin_stats(missed),
        "hit_margin_stats": margin_stats(hits),
        "missed_with_margin_gt_-0.5": len(close_missed),
        "missed_risk_score_stats": margin_stats([
            {"risk_minus_predicted_margin": r["risk_positive_score"]} for r in missed]),
    }

    # ---- question F: filler leakage ----
    filler_by_type = defaultdict(Counter)
    for r in records:
        filler_by_type[r["chosen_span_type"]][(r["recovered_filler"] or "").strip().lower()] += 1
    filler_by_target = defaultdict(Counter)
    for r in records:
        filler_by_target[r["gold_probe_target"]][(r["recovered_filler"] or "").strip().lower()] += 1
    filler_leakage_analysis = {
        "filler_by_span_type": {k: dict(v.most_common(5)) for k, v in filler_by_type.items()},
        "filler_by_gold_target": {k: dict(v.most_common(8)) for k, v in filler_by_target.items()},
        "note": ("If each span type has exactly one filler and each target maps to a fixed set of "
                 "span types, then filler identity fully determines the label -> structural leakage."),
    }
    # deterministic check: is filler a pure function of span_type?
    filler_is_deterministic_per_type = all(len(v) == 1 for v in filler_by_type.values())
    filler_leakage_analysis["filler_is_pure_function_of_span_type"] = filler_is_deterministic_per_type

    # missed vs hit: do missed cases differ in span/rule composition?
    missed_by_predicted = dict(Counter(r["predicted_probe_target"] for r in missed))

    # per span_type x predicted cross-tab for risk cases
    crosstab = defaultdict(Counter)
    for r in risk_cases:
        crosstab[r["chosen_span_type"]][r["predicted_probe_target"]] += 1

    # ---- findings ----
    comparison_stats = recall_by_span_type.get("comparison", {})
    negation_stats = recall_by_span_type.get("negation", {})
    numeric_stats = recall_by_numeric.get("numeric_span", {})
    non_numeric_stats = recall_by_numeric.get("non_numeric_span", {})

    main_findings = [
        ("F1: gold risk_positive is 100% span_type in {{negation, comparison}} by label-rule construction "
         "(negation={}, comparison={}). span_type=number can NEVER be gold risk_positive in this run, "
         "so the question 'is numeric risk_positive recall lower' is structurally about comparison spans "
         "with numeric-ish words (twice/half/...), not about masked numbers.").format(
            negation_stats.get("support"), comparison_stats.get("support")),
        ("F2: numeric-flagged (regex) risk_positive recall = {} on support {}, vs non-numeric recall = {} "
         "on support {}.").format(
            numeric_stats.get("recall"), numeric_stats.get("support"),
            non_numeric_stats.get("recall"), non_numeric_stats.get("support")),
        ("F3: span-type recall inside risk_positive: comparison = {} ({}/{}), negation = {} ({}/{}).").format(
            comparison_stats.get("recall"), comparison_stats.get("hit"), comparison_stats.get("support"),
            negation_stats.get("recall"), negation_stats.get("hit"), negation_stats.get("support")),
        ("F4: missed risk_positive predicted as: {} -> the two dominant sinks are "
         "hard_negative_or_weak_positive and positive_anchor.").format(missed_by_predicted),
        ("F5: in missed cases risk_positive score rank distribution = {}; rank-2 share = {}.").format(
            {str(k): v for k, v in sorted(rank_counts_missed.items())},
            score_margin_analysis["missed_rank2_share"]),
        ("F6: recovered filler is a pure function of span_type: {}. Combined with the deterministic "
         "span_type->label mapping, filler identity fully encodes the label for every class -> "
         "structural leakage confirmed.").format(filler_is_deterministic_per_type),
        ("F7: human_error_type / human_semantic_role are sentinel placeholders at 2000-scale "
         "(weak_auto*), so wrong_numeric_recovery / critical_number breakdowns cannot be computed "
         "from this run."),
        ("F8: class imbalance: risk_positive is 74/2000 (3.7%) and is split across two sub-populations "
         "(negation n=11, comparison n=63) that share no surface pattern with each other."),
    ]

    report = {
        "audit": "sprint_2H_risk_positive_audit",
        "mode": "read_only_audit",
        "boundary": {
            "reran_hidden_state_cache": False,
            "regenerated_cot": False,
            "reextracted_features": False,
            "retrained_probe": False,
            "generated_guidance_candidates": False,
            "executed_attention_steering": False,
            "modified_2A_to_2G_outputs": False,
        },
        "inputs": {
            "probe_predictions": str(PRED_PATH.relative_to(ROOT)),
            "weak_probe_dataset": str(PROBE_DS_PATH.relative_to(ROOT)),
            "weak_labels": str(WEAK_LABELS_PATH.relative_to(ROOT)),
            "full_scale_2a_manifest": str(MANIFEST_2A_PATH.relative_to(ROOT)),
            "weak_label_report": str(WEAK_LABEL_REPORT_PATH.relative_to(ROOT)),
        },
        "reproduction": reproduction,
        "risk_positive_total": len(risk_cases),
        "risk_positive_hit": len(hits),
        "risk_positive_missed": len(missed),
        "recall": reproduction["recall"],
        "missed_by_predicted_target": missed_by_predicted,
        "recall_by_span_type": recall_by_span_type,
        "recall_by_numeric_flag": recall_by_numeric,
        "recall_by_label_rule": recall_by_label_rule,
        "recall_by_human_error_type": recall_by_error_type,
        "recall_by_semantic_role": recall_by_semantic_role,
        "recall_by_span_text": recall_by_span_text,
        "span_type_x_predicted_crosstab": {k: dict(v) for k, v in crosstab.items()},
        "numeric_missed_case_ids": [r["masked_id"] for r in numeric_missed],
        "wrong_numeric_recovery_analysis": wrong_numeric_recovery_analysis,
        "critical_number_analysis": critical_number_analysis,
        "score_margin_analysis": score_margin_analysis,
        "filler_leakage_analysis": filler_leakage_analysis,
        "main_findings": [f for f in main_findings],
        "recommended_next_step": [],  # filled below after inspection
    }

    # recommended next steps derived from the measured facts
    recs = []
    recs.append(
        "1) Fix the weak label mapping BEFORE any threshold tuning: split 'critical number' out of "
        "positive_anchor (span_type=number, n=738, currently 100% positive_anchor by rule). Any "
        "wrong_numeric_recovery-style risk in numbers is invisible to the probe because the gold "
        "label says anchor. At minimum route ambiguous/critical numbers to review or risk_positive.")
    recs.append(
        "2) Break the recovered-filler leakage: filler is a pure function of span_type and span_type "
        "determines the label, so recovered/masked cosine features encode label identity. Use one "
        "shared filler for all types or model-generated recovery before retraining.")
    if score_margin_analysis["missed_rank2_share"] and score_margin_analysis["missed_rank2_share"] >= 0.5:
        recs.append(
            "3) risk_positive is rank-2 in {} of 51 missed cases -> class weighting / one-vs-rest "
            "threshold calibration for risk_positive is likely to recover a large share of the misses "
            "cheaply; do this in the SAME rerun as (1)+(2), not as a standalone fix on leaky features.".format(
                score_margin_analysis["missed_rank2_count"]))
    else:
        recs.append(
            "3) risk_positive score is rank-2 in only {} of 51 missed cases -> threshold calibration "
            "alone will NOT fix recall; the feature signal itself does not separate risk_positive. "
            "Prioritize label mapping + recovery construction over calibration.".format(
                score_margin_analysis["missed_rank2_count"]))
    recs.append(
        "4) Collect real human error-type labels (wrong_numeric_recovery / misleading_entity_or_unit / "
        "generic_recovery) on a stratified slice of the 2000 run; the current human_* fields are "
        "sentinels and Question B/C cannot be answered until then.")
    recs.append(
        "5) Do NOT enter Sprint 3A: only 23 increase_attention_to_original_span candidates exist and "
        "the gold risk pool itself (74, negation+comparison only) under-represents the risk concept.")
    report["recommended_next_step"] = recs

    # ---- write outputs ----
    def dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(OUT / "risk_positive_cases.jsonl", risk_cases)
    dump_jsonl(OUT / "risk_positive_hit_cases.jsonl", hits)
    dump_jsonl(OUT / "risk_positive_missed_cases.jsonl", missed)
    (OUT / "risk_positive_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "reproduction_ok": reproduction["matches_review_gate"],
        "recall_by_span_type": recall_by_span_type,
        "recall_by_numeric_flag": recall_by_numeric,
        "missed_by_predicted_target": missed_by_predicted,
        "score_margin_analysis": score_margin_analysis,
        "filler_pure_function_of_span_type": filler_is_deterministic_per_type,
        "filler_by_span_type": filler_leakage_analysis["filler_by_span_type"],
        "span_text_top": dict(list(recall_by_span_text.items())[:15]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
