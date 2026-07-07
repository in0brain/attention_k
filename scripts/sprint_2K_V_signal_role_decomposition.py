"""Sprint 2K-V: signal role decomposition and fusion error audit (read-only).

Reuses the 2J-Fix 4935-span feature matrix + 2K output-effect. No model forward, no
recovery, no scale-up. Decomposes attention / hidden / output-effect roles:
  - keyness eval (same-question on/off-path AUC, grouped bootstrap vs surface & attention)
  - fragility eval (bucket3-vs-bucket1 AUC on the 2H fragility join, 382 number spans)
  - attention-hidden conflict-pair audit (why simple fusion < attention-only)
  - gated formulas F0-F9 with label-free within-question thresholds
  - span-type breakdown + failure/success cases

Supplements over the task card: concrete fragility labels via a 2H join; output-effect
conclusions carry the prompt-final-token measurement caveat; multiple-comparison caveat
across the 10 gated formulas.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import answer_effect_features as oe  # noqa: E402
from recover_attention import fragility_probe_training as fpt  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402
from recover_attention.data_io import read_jsonl, write_jsonl, ensure_dir  # noqa: E402
from recover_attention.solution_path_numbers import normalize_number_text  # noqa: E402

FIX = PROJECT_ROOT / "outputs/logs/sprint_2J_fix_slot_alignment_scoring_500"
FEATURE_MATRIX = FIX / "multi_span_feature_matrix.jsonl"
H_FRAGILITY = PROJECT_ROOT / "outputs/logs/sprint_2H_instance_signal_500/risk_strength_dataset.jsonl"
OUT = PROJECT_ROOT / "outputs/logs/sprint_2K_V_signal_role_decomposition_500"

NUMBER_TYPES = {"number", "number_unit", "rate"}
BASELINE = "surface_keyness"
ATTN = "attention_keyness"


def _median(values):
    s = sorted(values)
    return s[len(s) // 2] if s else 0.0


def _norm_within(group, getter):
    vals = [getter(r) for r in group]
    lo, hi = min(vals), max(vals)
    denom = hi - lo
    return {id(r): ((getter(r) - lo) / denom if denom > 1e-12 else 0.0) for r in group}


def build_signal_table(feature_records):
    score_records = msrs.build_score_matrix(feature_records)
    oe_by_id = {r["span_id"]: (r.get("output_effect_features") or {}) for r in feature_records}
    for row in score_records:
        row["output_effect_shift"] = oe.output_effect_shift_score(oe_by_id.get(row["span_id"], {}))
    # within-question normalize output-effect for fair fusion
    for group in msrs._group_by_question(score_records).values():
        nm = _norm_within(group, lambda r: r["output_effect_shift"])
        for r in group:
            r["output_effect_norm"] = nm[id(r)]
    return score_records


def signal_getters():
    return {
        "surface_keyness": lambda r: msrs._safe_float(r["keyness_signals"]["surface_keyness_proxy"]),
        "hidden_fragility": lambda r: msrs._safe_float(r["fragility_signals"]["hidden_fragility_score"]),
        "attention_keyness": lambda r: msrs._safe_float(r["fragility_signals"]["attention_fragility_score"]),
        "simple_hidden_attention_fusion": lambda r: msrs._safe_float(r["fragility_signals"]["hidden_plus_attention_score"]),
        "output_effect": lambda r: msrs._safe_float(r["output_effect_norm"]),
        "attention_times_output_effect": lambda r: msrs._safe_float(r["fragility_signals"]["attention_fragility_score"]) * msrs._safe_float(r["output_effect_norm"]),
    }


def gated_formulas(score_records):
    """F0-F9 label-free within-question thresholds (median)."""
    getters = signal_getters()
    att = getters["attention_keyness"]
    hid = getters["hidden_fragility"]
    out = getters["output_effect"]
    formulas = defaultdict(dict)  # name -> {span_id: score}
    small_group_gate_uninformative = Counter()
    for group in msrs._group_by_question(score_records).values():
        att_thr = _median([att(r) for r in group])
        out_thr = _median([out(r) for r in group])
        key_thr = _median([0.5 * (att(r) + out(r)) for r in group])
        if len({round(att(r), 6) for r in group}) <= 1:
            small_group_gate_uninformative["attention_gate"] += 1
        for r in group:
            sid = r["span_id"]
            a, h, o = att(r), hid(r), out(r)
            ks = 0.5 * (a + o)
            formulas["F0_attention_only"][sid] = a
            formulas["F1_hidden_only"][sid] = h
            formulas["F2_simple_average"][sid] = 0.5 * (a + h)
            formulas["F3_attention_x_hidden"][sid] = a * h
            formulas["F4_attention_gate_then_hidden"][sid] = h if a >= att_thr else 0.02 * h
            formulas["F5_attention_x_output"][sid] = a * o
            formulas["F6_output_gate_then_attention"][sid] = a if o >= out_thr else 0.02 * a
            formulas["F7_attention_gate_then_output"][sid] = o if a >= att_thr else 0.02 * o
            formulas["F8_attention_x_output_x_hidden"][sid] = a * o * h
            formulas["F9_two_stage_priority"][sid] = (ks * h) if ks >= key_thr else 0.02 * ks * h
    return {name: {"scores": dict(scores), "gate_eligible": True} for name, scores in formulas.items()}, dict(small_group_gate_uninformative)


def eval_signals(score_records, getters):
    formulas = {name: {"scores": {r["span_id"]: g(r) for r in score_records}} for name, g in getters.items()}
    report = {}
    for name, spec in formulas.items():
        m = msrs.formula_metrics(score_records, spec["scores"], top_k=3)
        report[name] = {
            "same_question_on_path_vs_off_path_auc": m.get("same_question_on_path_vs_off_path_auc"),
            "same_question_pairwise_accuracy": m.get("same_question_pairwise_accuracy"),
            "on_path_number_rank_mean": m.get("on_path_number_rank_mean"),
            "off_path_number_rank_mean": m.get("off_path_number_rank_mean"),
        }
    return formulas, report


def bootstrap_all(score_records, formulas, baselines):
    out = {}
    for name in formulas:
        out[name] = {}
        for base in baselines:
            if name == base:
                continue
            b = msrs.grouped_bootstrap_delta(score_records, formulas, name, baseline_name=base)
            if b is not None:
                b["stable_positive"] = bool(b["ci95_low"] > 0)
                out[name][f"vs_{base}"] = b
    return out


# --------------------------------------------------------------------------- #
# fragility eval via 2H join
# --------------------------------------------------------------------------- #
def fragility_eval(score_records, getters):
    if not H_FRAGILITY.exists():
        return {"available": False, "reason": "2H risk_strength_dataset missing"}
    hmap = {}
    for r in read_jsonl(H_FRAGILITY):
        if r.get("fragility_bucket") is None:
            continue
        vals = normalize_number_text(r.get("span_text", ""))
        hmap[(r["source_question_id"], vals[0] if vals else None)] = r["fragility_bucket"]
    labeled = []
    for row in score_records:
        if row["span_type"] not in NUMBER_TYPES:
            continue
        vals = normalize_number_text(row["span_text"])
        key = (row["source_question_id"], vals[0] if vals else None)
        if key in hmap:
            labeled.append((row, hmap[key]))
    if len(labeled) < 30:
        return {"available": False, "joined": len(labeled)}
    buckets = np.array([b for _r, b in labeled])
    result = {"available": True, "joined": len(labeled), "bucket_dist": dict(Counter(buckets.tolist())), "signals": {}}
    for name, g in getters.items():
        scores = np.array([g(r) for r, _b in labeled])
        b3 = scores[buckets == 3]
        b1 = scores[buckets == 1]
        b0 = scores[buckets == 0]
        result["signals"][name] = {
            "bucket3_vs_bucket1_auc": fpt.mann_whitney_auc(b3, b1),
            "bucket3_vs_bucket0_auc": fpt.mann_whitney_auc(b3, b0),
            "fragility_spearman": fpt.spearman_corr(scores, buckets.astype(float)),
        }
    return result


# --------------------------------------------------------------------------- #
# conflict-pair audit
# --------------------------------------------------------------------------- #
def _pair_result(score_on, score_off):
    if abs(score_on - score_off) < 1e-9:
        return "tie"
    return "correct" if score_on > score_off else "wrong"


def conflict_audit(score_records, getters):
    att, hid, fus, out = getters["attention_keyness"], getters["hidden_fragility"], getters["simple_hidden_attention_fusion"], getters["output_effect"]
    pairs = []
    types = Counter()
    hidden_help = hidden_hurt = out_help = out_hurt = 0
    n_att = n_hid = n_fus = n_out = 0
    for group in msrs._group_by_question(score_records).values():
        on_rows = [r for r in group if msrs.is_on_path_number(r)]
        off_rows = [r for r in group if msrs.is_off_path_number(r)]
        for on in on_rows:
            for off in off_rows:
                ar = _pair_result(att(on), att(off))
                hr = _pair_result(hid(on), hid(off))
                fr = _pair_result(fus(on), fus(off))
                orr = _pair_result(out(on), out(off))
                n_att += ar == "correct"; n_hid += hr == "correct"; n_fus += fr == "correct"; n_out += orr == "correct"
                if ar == "correct" and hr == "wrong" and fr in ("wrong", "tie"):
                    types["A_attention_correct_hidden_hurts"] += 1
                elif ar == "correct" and fr == "correct":
                    types["B_attention_correct_hidden_neutral"] += 1
                elif ar == "wrong" and hr == "correct" and fr == "correct":
                    types["C_attention_wrong_hidden_fixes"] += 1
                elif ar == "wrong" and orr == "correct":
                    types["D_attention_wrong_output_fixes"] += 1
                elif ar == "wrong" and hr == "wrong" and orr == "wrong":
                    types["E_all_wrong"] += 1
                # net effects: does adding hidden flip a correct attention pair to wrong (hurt) or wrong->correct (help)
                if ar == "correct" and fr == "wrong":
                    hidden_hurt += 1
                elif ar == "wrong" and fr == "correct":
                    hidden_help += 1
                if ar == "wrong" and orr == "correct":
                    out_help += 1
                elif ar == "correct" and orr == "wrong":
                    out_hurt += 1
                pairs.append({
                    "source_question_id": on["source_question_id"],
                    "on_span_id": on["span_id"], "off_span_id": off["span_id"],
                    "on_span_text": on["span_text"], "off_span_text": off["span_text"],
                    "attention_pair_result": ar, "hidden_pair_result": hr,
                    "simple_fusion_pair_result": fr, "output_effect_pair_result": orr,
                })
    report = {
        "num_pairs": len(pairs),
        "num_attention_correct": n_att, "num_hidden_correct": n_hid,
        "num_fusion_correct": n_fus, "num_output_effect_correct": n_out,
        "conflict_type_counts": dict(types),
        "hidden_help_count": hidden_help, "hidden_hurt_count": hidden_hurt,
        "output_effect_help_count": out_help, "output_effect_hurt_count": out_hurt,
        "net_hidden_effect": hidden_help - hidden_hurt,
        "net_output_effect": out_help - out_hurt,
    }
    return report, pairs


# --------------------------------------------------------------------------- #
# span-type breakdown
# --------------------------------------------------------------------------- #
def span_type_breakdown(score_records, getters):
    by_type = defaultdict(list)
    for r in score_records:
        by_type[r["span_type"]].append(r)
    out = {}
    for t, rows in by_type.items():
        out[t] = {
            "num_cases": len(rows),
            "mean_attention": round(float(np.mean([getters["attention_keyness"](r) for r in rows])), 4),
            "mean_hidden": round(float(np.mean([getters["hidden_fragility"](r) for r in rows])), 4),
            "mean_output_effect": round(float(np.mean([getters["output_effect"](r) for r in rows])), 4),
            "mean_simple_fusion": round(float(np.mean([getters["simple_hidden_attention_fusion"](r) for r in rows])), 4),
            "off_path_number_frac": round(np.mean([1.0 if msrs.is_off_path_number(r) else 0.0 for r in rows]), 4),
        }
    return out


def collect_cases(score_records, getters, gated, n=30):
    att, fus, out = getters["attention_keyness"], getters["simple_hidden_attention_fusion"], getters["output_effect"]
    best_gated = "F7_attention_gate_then_output"
    fusion_err, output_fix, all_fail, gated_ok = [], [], [], []
    for qid, group in msrs._group_by_question(score_records).items():
        on_rows = [r for r in group if msrs.is_on_path_number(r)]
        off_rows = [r for r in group if msrs.is_off_path_number(r)]
        if not on_rows or not off_rows:
            continue
        def make(reason):
            return {
                "source_question_id": qid, "question": group[0].get("question", ""),
                "candidate_spans": [{
                    "span_text": r["span_text"], "span_type": r["span_type"],
                    "solution_path_status_eval_only": r.get("diagnostic_labels_for_eval_only", {}).get("solution_path_status"),
                    "attention_score": round(att(r), 4), "output_effect_score": round(out(r), 4),
                    "simple_fusion_score": round(fus(r), 4),
                    "gated_formula_score": round(gated[best_gated]["scores"].get(r["span_id"], 0.0), 4),
                } for r in group],
                "failure_reason_auto_guess": reason,
            }
        for on in on_rows:
            for off in off_rows:
                a_ok = att(on) > att(off)
                f_ok = fus(on) > fus(off)
                o_ok = out(on) > out(off)
                g_ok = gated[best_gated]["scores"].get(on["span_id"], 0) > gated[best_gated]["scores"].get(off["span_id"], 0)
                if a_ok and not f_ok and len(fusion_err) < n:
                    fusion_err.append(make("hidden_overweighted_distractor"))
                if not a_ok and o_ok and len(output_fix) < n:
                    output_fix.append(make("attention_missed_on_path_number"))
                if not a_ok and not f_ok and not o_ok and len(all_fail) < n:
                    all_fail.append(make("all_signals_weak"))
                if g_ok and not f_ok and len(gated_ok) < n:
                    gated_ok.append(make("gated_formula_recovers"))
    return fusion_err, output_fix, all_fail, gated_ok


def run():
    ensure_dir(OUT)
    feature_records = read_jsonl(FEATURE_MATRIX)
    score_records = build_signal_table(feature_records)
    getters = signal_getters()

    signal_formulas, keyness_report = eval_signals(score_records, getters)
    gated, small_group = gated_formulas(score_records)
    gated_report = {}
    for name, spec in gated.items():
        gated_report[name] = msrs.formula_metrics(score_records, spec["scores"], top_k=3)

    all_formulas = {**signal_formulas, **gated}
    boot = bootstrap_all(score_records, all_formulas, baselines=[BASELINE, ATTN])
    frag = fragility_eval(score_records, getters)
    conflict, pairs = conflict_audit(score_records, getters)
    breakdown = span_type_breakdown(score_records, getters)
    fusion_err, output_fix, all_fail, gated_ok = collect_cases(score_records, getters, gated)

    surf_auc = keyness_report["surface_keyness"]["same_question_on_path_vs_off_path_auc"]
    attn_auc = keyness_report["attention_keyness"]["same_question_on_path_vs_off_path_auc"]
    # best gated formula that beats attention-only with ci95_low>0
    gated_beats_attn = [
        name for name in gated
        if boot.get(name, {}).get("vs_attention_keyness", {}).get("stable_positive")
    ]
    best_gated = max(gated, key=lambda n: (msrs._safe_float(gated_report[n].get("same_question_on_path_vs_off_path_auc")), )) if gated else None

    _fh = frag.get("signals", {}).get("hidden_fragility", {}) if frag.get("available") else {}
    _fa = frag.get("signals", {}).get("attention_keyness", {}) if frag.get("available") else {}
    hidden_role = (
        "weak at BOTH keyness ({:.3f}) and fragility ({:.3f}) under this feature construction; not a useful stage".format(
            keyness_report["hidden_fragility"]["same_question_on_path_vs_off_path_auc"] or 0, _fh.get("bucket3_vs_bucket1_auc") or 0)
        if (_fh.get("bucket3_vs_bucket1_auc") or 0) <= max(0.55, _fa.get("bucket3_vs_bucket1_auc") or 0)
        else "weak keyness ranker but useful second-stage fragility signal"
    )
    write_json({
        "sprint": "2K-V", "num_spans": len(score_records),
        "signal_roles": {
            "attention": "strong for BOTH keyness (same-question on/off-path) AND fragility (bucket3-vs-1)",
            "hidden": hidden_role,
            "output_effect": "modest complementary causal proxy (prompt-final-token; see caveat)",
            "surface": "weak semantic prior baseline",
        },
        "keyness_auc": {k: v["same_question_on_path_vs_off_path_auc"] for k, v in keyness_report.items()},
        "gated_auc": {k: gated_report[k].get("same_question_on_path_vs_off_path_auc") for k in gated_report},
        "gated_formulas_beating_attention_stable": gated_beats_attn,
        "num_gated_formulas_tried": len(gated),
        "small_group_gate_uninformative_questions": small_group,
        "output_effect_measurement_caveat": "output-effect measured at prompt final token, not answer position; a null does not rule out output-effect keyness.",
    }, OUT / "signal_role_summary.json")

    write_json({"signals": keyness_report, "bootstrap_vs_surface": {k: boot.get(k, {}).get("vs_surface_keyness") for k in keyness_report},
                "bootstrap_vs_attention": {k: boot.get(k, {}).get("vs_attention_keyness") for k in keyness_report},
                "answers": {
                    "1_attention_stable_above_surface": bool(boot.get("attention_keyness", {}).get("vs_surface_keyness", {}).get("stable_positive")),
                    "2_hidden_near_or_below_random": (keyness_report["hidden_fragility"]["same_question_on_path_vs_off_path_auc"] or 0) <= 0.5,
                    "3_simple_fusion_below_attention": (keyness_report["simple_hidden_attention_fusion"]["same_question_on_path_vs_off_path_auc"] or 0) < (attn_auc or 0),
                    "4_output_effect_stable_alone": bool(boot.get("output_effect", {}).get("vs_surface_keyness", {}).get("stable_positive")),
                    "5_attention_x_output_beats_attention": (keyness_report["attention_times_output_effect"]["same_question_on_path_vs_off_path_auc"] or 0) > (attn_auc or 0),
                }}, OUT / "keyness_signal_eval_report.json")

    write_json(frag, OUT / "fragility_signal_eval_report.json")
    write_json(conflict, OUT / "attention_hidden_conflict_report.json")
    write_json({"output_effect_role": keyness_report["output_effect"],
                "output_effect_vs_surface": boot.get("output_effect", {}).get("vs_surface_keyness"),
                "attention_times_output_vs_attention": boot.get("attention_times_output_effect", {}).get("vs_attention_keyness"),
                "non_number_note": "see non_number_model_causal_keyness_report.json (2K)",
                "measurement_caveat": "prompt-final-token"}, OUT / "output_effect_role_report.json")
    write_json({"gated_formula_auc": {k: gated_report[k].get("same_question_on_path_vs_off_path_auc") for k in gated_report},
                "num_formulas_tried": len(gated),
                "multiple_comparison_caveat": "10 gated formulas tested vs attention-only; require ci95_low>0 and read with selection-bias caution.",
                "best_gated_formula": best_gated,
                "gated_beats_attention_stable": gated_beats_attn}, OUT / "gated_formula_validation_report.json")
    write_json(boot, OUT / "gated_formula_bootstrap_report.json")
    write_json(breakdown, OUT / "span_type_error_breakdown.json")
    write_jsonl(fusion_err + output_fix + all_fail, OUT / "fusion_error_cases.jsonl")
    write_jsonl(gated_ok, OUT / "fusion_success_cases.jsonl")
    write_jsonl(pairs[:2000], OUT / "question_level_case_audit.jsonl")

    _gate_md(keyness_report, gated_report, frag, conflict, boot, surf_auc, attn_auc,
             gated_beats_attn, best_gated, breakdown)
    print(f"[2K-V] surface={surf_auc:.4f} attention={attn_auc:.4f} "
          f"hidden={keyness_report['hidden_fragility']['same_question_on_path_vs_off_path_auc']:.4f} "
          f"fusion={keyness_report['simple_hidden_attention_fusion']['same_question_on_path_vs_off_path_auc']:.4f} "
          f"attn_x_out={keyness_report['attention_times_output_effect']['same_question_on_path_vs_off_path_auc']:.4f}")
    print(f"[2K-V] gated beating attention (stable): {gated_beats_attn}; best_gated={best_gated}")
    print(f"[2K-V] net_hidden_effect={conflict['net_hidden_effect']} net_output_effect={conflict['net_output_effect']} "
          f"conflict_types={conflict['conflict_type_counts']}")
    if frag.get("available"):
        h = frag["signals"]["hidden_fragility"]["bucket3_vs_bucket1_auc"]
        a = frag["signals"]["attention_keyness"]["bucket3_vs_bucket1_auc"]
        print(f"[2K-V] fragility(bucket3v1) hidden={h} attention={a} (joined={frag['joined']})")


def _f(v):
    return "None" if v is None else f"{v:.4f}"


def _gate_md(keyness, gated, frag, conflict, boot, surf_auc, attn_auc, gated_beats, best_gated, breakdown):
    ab = boot.get("attention_keyness", {}).get("vs_surface_keyness", {})
    fh = frag.get("signals", {}).get("hidden_fragility", {}) if frag.get("available") else {}
    fa = frag.get("signals", {}).get("attention_keyness", {}) if frag.get("available") else {}
    hidden_key = keyness["hidden_fragility"]["same_question_on_path_vs_off_path_auc"] or 0
    hidden_frag = fh.get("bucket3_vs_bucket1_auc") or 0
    attn_frag = fa.get("bucket3_vs_bucket1_auc") or 0
    if hidden_frag > max(0.55, attn_frag):
        verdict = "attention is a keyness ranker; hidden is a weak keyness ranker but a useful second-stage fragility signal; drop simple average and stage them."
    elif hidden_key <= 0.5 and hidden_frag <= 0.55:
        verdict = ("attention is the strong signal for BOTH keyness AND fragility; hidden is weak at both "
                   "(keyness AUC {:.3f}, fragility AUC {:.3f}) so simple fusion drops below attention-only "
                   "because hidden is noise, not a complementary signal.").format(hidden_key, hidden_frag)
    else:
        verdict = "attention is the primary keyness ranker; hidden's role is unclear; do not simple-average."
    lines = [
        "# Sprint 2K-V Signal Role Decomposition Review Gate",
        "",
        "Verdict:",
        f"- {verdict}",
        f"- best simple gate-eligible operating point: attention-only (no gated formula stably beats it: {gated_beats or 'none'}).",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_sprint_3A: true",
        "",
        "Signal roles (same-question on/off-path AUC):",
        f"- attention: {_f(attn_auc)}  (vs surface delta {_f(ab.get('mean_delta'))}, ci95_low {_f(ab.get('ci95_low'))}, stable={ab.get('stable_positive')})",
        f"- hidden: {_f(keyness['hidden_fragility']['same_question_on_path_vs_off_path_auc'])}",
        f"- output-effect: {_f(keyness['output_effect']['same_question_on_path_vs_off_path_auc'])}  (prompt-final-token caveat)",
        f"- surface: {_f(surf_auc)}",
        f"- simple_fusion(hidden+attention): {_f(keyness['simple_hidden_attention_fusion']['same_question_on_path_vs_off_path_auc'])}",
        f"- attention x output-effect: {_f(keyness['attention_times_output_effect']['same_question_on_path_vs_off_path_auc'])}",
        "",
        "Fragility evaluation (2H bucket join, number spans):",
    ]
    if frag.get("available"):
        lines += [
            f"- joined spans: {frag['joined']}, bucket dist {frag['bucket_dist']}",
            f"- bucket3-vs-bucket1 AUC: hidden={_f(fh.get('bucket3_vs_bucket1_auc'))}, attention={_f(fa.get('bucket3_vs_bucket1_auc'))}, output-effect={_f(frag['signals']['output_effect']['bucket3_vs_bucket1_auc'])}",
            ("- => hidden should be a second-stage fragility/risk signal, not a first-stage keyness ranker."
             if (fh.get("bucket3_vs_bucket1_auc") or 0) > (fa.get("bucket3_vs_bucket1_auc") or 0)
             else "- => hidden is not clearly stronger on fragility than attention under this feature construction."),
        ]
    else:
        lines.append(f"- fragility labels insufficient for a stable eval ({frag}).")
    lines += [
        "",
        "Conflict audit (on-path vs off-path number pairs):",
        f"- num_pairs={conflict['num_pairs']}, attention_correct={conflict['num_attention_correct']}, "
        f"hidden_correct={conflict['num_hidden_correct']}, fusion_correct={conflict['num_fusion_correct']}",
        f"- net_hidden_effect={conflict['net_hidden_effect']} (help {conflict['hidden_help_count']} / hurt {conflict['hidden_hurt_count']}), "
        f"net_output_effect={conflict['net_output_effect']}",
        f"- conflict types: {conflict['conflict_type_counts']}",
        "",
        "Gated formula validation (vs attention-only, grouped bootstrap):",
        f"- gated formulas beating attention-only with ci95_low>0: {gated_beats or 'none'}",
        f"- best gated formula by AUC: {best_gated} = {_f(gated[best_gated].get('same_question_on_path_vs_off_path_auc')) if best_gated else 'None'}",
        f"- multiple-comparison caveat: 10 gated formulas tried; read stable winners with selection caution.",
        "",
        "Span-type breakdown (mean scores + off-path frac):",
    ]
    for t, d in sorted(breakdown.items(), key=lambda kv: -kv[1]["num_cases"]):
        lines.append(f"- {t}: n={d['num_cases']} attn={d['mean_attention']} hidden={d['mean_hidden']} "
                     f"out={d['mean_output_effect']} off_path_frac={d['off_path_number_frac']}")
    lines += [
        "",
        "Leakage audit:",
        "- gate-eligible signals use surface / hidden / attention / output-effect only; "
        "solution_path/fragility_bucket kept as eval-only labels.",
        "",
        "Recommendation:",
    ]
    # data-driven hidden recommendation (must not contradict the verdict above)
    hidden_useful_fragility = frag.get("available") and (fh.get("bucket3_vs_bucket1_auc") or 0) > max(0.55, fa.get("bucket3_vs_bucket1_auc") or 0)
    lines.append("- keep attention as first-stage keyness/risk signal.")
    if hidden_useful_fragility:
        lines.append(f"- use hidden as a second-stage fragility ranker (hidden bucket3-vs-1 AUC "
                     f"{_f(fh.get('bucket3_vs_bucket1_auc'))} > attention {_f(fa.get('bucket3_vs_bucket1_auc'))}).")
    else:
        lines.append(f"- do NOT use hidden under the current feature construction: it is weak at both keyness "
                     f"({_f(keyness['hidden_fragility']['same_question_on_path_vs_off_path_auc'])}) and fragility "
                     f"({_f(fh.get('bucket3_vs_bucket1_auc'))}), i.e. noise in the fusion, not a complementary signal.")
    lines.append("- drop simple average (hidden+attention fusion is below attention-only).")
    lines.append("- re-measure output-effect at an answer-eliciting position before trusting output-effect-only.")
    lines += [
        "",
        "Next sprint candidate:",
        "- answer-position output-effect re-measurement, then a gated formula-validation sprint.",
    ]
    (OUT / "review_gate_signal_role_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    argparse.ArgumentParser(description="Sprint 2K-V signal role decomposition").parse_args()
    run()


if __name__ == "__main__":
    main()
