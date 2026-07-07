"""Sprint 2J-Fix + 2K: slot-alignment repair + leakage-safe output-effect keyness.

Fixes the 2J-B bug where the cached original forward reused the FIRST span's
slot_indices for every span in a question (corrupting all original-side hidden/
attention/delta features). Here the original forward is cached once per question and
the original slot indices are RECOMPUTED per span from the question offsets.

Adds (2K) leakage-safe self-output-effect features from the last-token logits of the
same forwards, plus formulas J-N, and evaluates same-question on/off-path AUC with a
real question-grouped bootstrap. Reuses the 2J-B module's pure helpers; does not
overwrite 2J-B outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import answer_effect_features as oe  # noqa: E402
from recover_attention import attention_features as af  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import read_jsonl, write_jsonl, ensure_dir  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402
from recover_attention.hidden_state_cache import infer_model_input_device  # noqa: E402

FLAT_MATRIX = PROJECT_ROOT / "outputs/logs/sprint_2J_multi_span_matrix_500/multi_span_flat_matrix.jsonl"
FIX_DIR = PROJECT_ROOT / "outputs/logs/sprint_2J_fix_slot_alignment_scoring_500"
K_DIR = PROJECT_ROOT / "outputs/logs/sprint_2K_answer_effect_keyness_500"
FEATURE_MATRIX = FIX_DIR / "multi_span_feature_matrix.jsonl"
MODEL_PATH = r"D:/models/Qwen2.5-7B-Instruct"
LAYERS = list(msrs.DEFAULT_LAYER_INDICES)  # [0, 8, 16, 24]


# --------------------------------------------------------------------------- #
# forward with last-token logits (reuses module internals)
# --------------------------------------------------------------------------- #
def forward_with_logits(context, text, layer_indices):
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(s), int(e)] for s, e in enc.pop("offset_mapping")[0].tolist()]
    device = infer_model_input_device(model, "auto", torch)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=False)
    hidden, resolved, warns = msrs._select_hidden_layers(out.hidden_states, layer_indices, torch)
    attention = af.head_average_selected_layers(out.attentions, layer_indices)
    last_logits = out.logits[0, -1, :].detach().float().cpu()
    return msrs.ForwardResult(
        text=text, offsets=offsets, slot_indices=[], hidden=hidden.cpu(),
        attention=attention, resolved_hidden_state_indices=resolved,
        last_logits=last_logits, warnings=list(warns),
    )


def build_feature_record_fixed(flat, original, masked, original_slot_indices, masked_slot_indices, layers):
    # per-span views so the reused pure helpers see the CORRECT slot indices
    orig_view = msrs.ForwardResult(original)
    orig_view["slot_indices"] = original_slot_indices
    masked_view = msrs.ForwardResult(masked)
    masked_view["slot_indices"] = masked_slot_indices

    hidden_features = msrs.compute_hidden_features(orig_view, masked_view)
    context = msrs.context_indices_for_question(
        flat["question"], original["offsets"], exclude=set(original_slot_indices)
    )
    attention_built = af.build_attention_features(
        original["attention"], original_slot_indices,
        masked["attention"], masked_slot_indices, layers, context_orig=context,
    )
    output_effect = oe.compute_output_effect(original["last_logits"], masked["last_logits"])
    warnings = []
    if not original_slot_indices:
        warnings.append({"warning_type": "original_slot_missing", "message": flat["span_id"]})
    if not masked_slot_indices:
        warnings.append({"warning_type": "masked_slot_missing", "message": flat["span_id"]})
    return {
        "backend": "multi_span_fix_2k_v1",
        "source_question_id": flat["source_question_id"],
        "span_id": flat["span_id"],
        "span_text": flat["span_text"],
        "span_type": flat["span_type"],
        "question": flat["question"],
        "masked_question": flat["masked_question"],
        "mask_text": flat.get("mask_text", msrs.DEFAULT_MASK_TOKEN),
        "span_char_start": flat["span_char_start"],
        "span_char_end": flat["span_char_end"],
        "surface_features": flat.get("surface_features", {}),
        "hidden_features": hidden_features,
        "attention_features": msrs.task_attention_feature_view(attention_built["features"]),
        "output_effect_features": output_effect,
        "missing_context": attention_built.get("missing", {}),
        "diagnostic_labels_for_eval_only": flat.get("diagnostic_labels_for_eval_only", {}),
        "alignment": {
            "alignment_status": "warning" if warnings else "ok",
            "num_original_slot_tokens": len(original_slot_indices),
            "num_masked_slot_tokens": len(masked_slot_indices),
            "original_slot_token_indices": original_slot_indices,
            "mask_token_indices": masked_slot_indices,
            "slot_alignment_source": "per_span_recomputed",
            "warnings": warnings,
        },
    }


def run_forward(limit, report_every=50):
    flat = read_jsonl(FLAT_MATRIX)
    if limit:
        flat = flat[:limit]
    done = {}
    if FEATURE_MATRIX.exists():
        done = {r["span_id"]: r for r in read_jsonl(FEATURE_MATRIX)}
    pending = [r for r in flat if r["span_id"] not in done]
    print(f"[2J-Fix/2K] spans={len(flat)} done={len(done)} pending={len(pending)}")
    if not pending:
        return list(done.values())

    context = msrs.load_local_attention_backend(model_path=MODEL_PATH)
    original_cache = {}
    ensure_dir(FEATURE_MATRIX.parent)
    t0 = time.time()
    for i, flat_rec in enumerate(pending, start=1):
        qid = str(flat_rec["source_question_id"])
        original = original_cache.get(qid)
        if original is None:
            original = forward_with_logits(context, flat_rec["question"], LAYERS)
            original_cache[qid] = original
        # per-span original slot indices (THE FIX)
        original_slot = msrs.token_indices_for_char_ranges(
            original["offsets"],
            [[int(flat_rec["span_char_start"]), int(flat_rec["span_char_end"])]],
            exclude=set(),
        )
        mask_range = msrs._mask_char_range(flat_rec["masked_question"], flat_rec.get("mask_text") or msrs.DEFAULT_MASK_TOKEN)
        masked = forward_with_logits(context, flat_rec["masked_question"], LAYERS)
        masked_slot = msrs.token_indices_for_char_ranges(masked["offsets"], [mask_range], exclude=set())
        rec = build_feature_record_fixed(flat_rec, original, masked, original_slot, masked_slot, LAYERS)
        with FEATURE_MATRIX.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[flat_rec["span_id"]] = rec
        if report_every and i % report_every == 0:
            print(f"[2J-Fix/2K] {i}/{len(pending)} rate={i/max(time.time()-t0,1e-9):.2f}/s")
    return list(done.values())


# --------------------------------------------------------------------------- #
# scoring: reuse module score matrix, inject output-effect, extend formulas J-N
# --------------------------------------------------------------------------- #
def _inject_output_effect(score_rows, feature_records):
    by_id = {r["span_id"]: r for r in feature_records}
    for row in score_rows:
        feats = (by_id.get(row["span_id"]) or {}).get("output_effect_features", {}) or {}
        row["output_effect_signals"] = feats
        row["output_effect_shift"] = oe.output_effect_shift_score(feats)


def build_formula_scores_extended(score_records):
    formulas = msrs.build_formula_scores(score_records)  # A-I
    for name in ["J_output_effect_only", "K_keyness_times_output_effect",
                 "L_output_gate_then_fragility", "M_output_effect_times_fragility",
                 "N_keyness_output_effect_fragility"]:
        formulas[name] = {"gate_eligible": True, "uses_eval_only_labels": False, "scores": {}}
    grouped = defaultdict(list)
    for row in score_records:
        grouped[row["source_question_id"]].append(row)
    for group in grouped.values():
        # label-free within-question median gate threshold for output-effect
        shifts = sorted(r.get("output_effect_shift", 0.0) for r in group)
        gate_thr = shifts[len(shifts) // 2] if shifts else 0.0
        for row in group:
            oe_shift = float(row.get("output_effect_shift", 0.0))
            keyness = msrs.keyness_score(row)
            hidden = msrs._safe_float(row["fragility_signals"].get("hidden_fragility_score"), 0.0)
            attention = msrs._safe_float(row["fragility_signals"].get("attention_fragility_score"), 0.0)
            fragility = msrs._mean_numeric([hidden, attention])
            formulas["J_output_effect_only"]["scores"][row["span_id"]] = oe_shift
            formulas["K_keyness_times_output_effect"]["scores"][row["span_id"]] = keyness * oe_shift
            formulas["L_output_gate_then_fragility"]["scores"][row["span_id"]] = fragility if oe_shift >= gate_thr else 0.05 * fragility
            formulas["M_output_effect_times_fragility"]["scores"][row["span_id"]] = oe_shift * fragility
            formulas["N_keyness_output_effect_fragility"]["scores"][row["span_id"]] = keyness * oe_shift * fragility
    return formulas


GATE_ELIGIBLE = [
    "A_surface_only", "B_hidden_only", "C_attention_only", "D_hidden_plus_attention",
    "E_keyness_times_fragility", "F_keyness_gate_then_fragility", "G_per_question_normalized_priority",
    "H_span_type_budget_policy", "J_output_effect_only", "K_keyness_times_output_effect",
    "L_output_gate_then_fragility", "M_output_effect_times_fragility", "N_keyness_output_effect_fragility",
]


def auc_of(ranking_report, name):
    return ranking_report["formulas"].get(name, {}).get("same_question_on_path_vs_off_path_auc")


def build_reports(feature_records):
    score_records = msrs.build_score_matrix(feature_records)
    _inject_output_effect(score_records, feature_records)
    formulas = build_formula_scores_extended(score_records)
    ranking = msrs.build_same_question_ranking_report(score_records, formulas)

    # real grouped bootstrap for every gate-eligible formula vs surface
    bootstrap = {}
    for name in GATE_ELIGIBLE:
        if name == "A_surface_only":
            continue
        b = msrs.grouped_bootstrap_delta(score_records, formulas, name, baseline_name="A_surface_only")
        if b is not None:
            b["stable_positive"] = bool(b["ci95_low"] > 0)
            bootstrap[name] = b

    best = max(
        (n for n in GATE_ELIGIBLE if n != "A_surface_only"),
        key=lambda n: (auc_of(ranking, n) or -1),
    )
    surf_auc = auc_of(ranking, "A_surface_only")
    return score_records, formulas, ranking, bootstrap, best, surf_auc


def write_fix_outputs(score_records, formulas, ranking, bootstrap, best, surf_auc):
    ensure_dir(FIX_DIR)
    write_json(ranking, FIX_DIR / "same_question_ranking_report.json")
    write_json(bootstrap, FIX_DIR / "grouped_bootstrap_report.json")
    hidden_aucs = {n: auc_of(ranking, n) for n in ["A_surface_only", "B_hidden_only", "C_attention_only", "D_hidden_plus_attention"]}
    audit = {
        "sprint": "2J-Fix",
        "slot_alignment_source": "per_span_recomputed",
        "num_spans": len(score_records),
        "hidden_attention_auc_after_fix": hidden_aucs,
        "surface_auc": surf_auc,
        "interpretation": (
            "hidden/attention perturbation signals remain anti-aligned with within-question keyness"
            if (hidden_aucs["D_hidden_plus_attention"] or 0) < 0.5 else
            "previous anti-keyness effect was likely inflated by slot alignment error"
        ),
    }
    write_json(audit, FIX_DIR / "slot_alignment_audit.json")
    checks = {
        "1_same_question_ranking_computable": ranking["formulas"]["A_surface_only"]["num_pairwise_on_off_number_pairs"] > 0,
        "2_slot_alignment_per_span": True,
        "5_auc_primary_metric_reported": all(auc_of(ranking, n) is not None for n in GATE_ELIGIBLE),
        "6_grouped_bootstrap_real": all("ci95_low" in v for v in bootstrap.values()),
        "7_best_stable_vs_surface": bool(bootstrap.get(best, {}).get("ci95_low", -1) > 0),
        "8_ready_for_2000_rerun_false": True,
        "9_do_not_enter_3A": True,
    }
    md = _fix_md(hidden_aucs, surf_auc, best, bootstrap, checks, ranking)
    (FIX_DIR / "review_gate_multi_span_scoring_fixed.md").write_text(md, encoding="utf-8")
    return audit, checks


def write_k_outputs(score_records, formulas, ranking, bootstrap, best, surf_auc):
    ensure_dir(K_DIR)
    aucs = {n: auc_of(ranking, n) for n in GATE_ELIGIBLE}
    write_json({"formulas_auc": aucs, "num_spans": len(score_records)},
               K_DIR / "same_question_output_effect_ranking_report.json")
    write_json({"gate_eligible_bootstrap_vs_surface": bootstrap, "best_non_oracle_formula": best},
               K_DIR / "output_effect_grouped_bootstrap_report.json")
    # leakage audit of output-effect feature names
    oe_names = sorted({k for r in score_records for k in (r.get("output_effect_signals") or {}).keys()})
    oe.assert_no_banned_output_effect_names(oe_names)
    write_json({"output_effect_feature_names": oe_names, "leakage_free": True,
                "forbidden_substrings": list(oe.BANNED_FEATURE_SUBSTRINGS)},
               K_DIR / "output_effect_feature_audit.json")
    # non-number model-causal keyness diagnostic (self-output shift terciles by span_type)
    diag = _non_number_diag(score_records)
    write_json(diag, K_DIR / "non_number_model_causal_keyness_report.json")
    write_json({
        "self_output_effect_computed": True, "gold_answer_diagnostic": "skipped_optional",
        "cot": "not_computed", "trajectory": "not_computed", "nla": "not_computed",
        "causal_attribution": "not_computed",
    }, K_DIR / "reasoning_signal_gap_report.json")
    md = _k_md(aucs, surf_auc, best, bootstrap, diag)
    (K_DIR / "review_gate_answer_effect_keyness.md").write_text(md, encoding="utf-8")


def _non_number_diag(score_records):
    by_type = defaultdict(list)
    for r in score_records:
        by_type[r["span_type"]].append(float(r.get("output_effect_shift", 0.0)))
    out = {}
    for t, vals in by_type.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        out[t] = {
            "n": n,
            "mean_output_effect_shift": round(float(np.mean(vals)), 5) if vals else None,
            "median": round(vals_sorted[n // 2], 5) if n else None,
        }
    return out


def _fmt(v):
    return "None" if v is None else f"{v:.4f}"


def _fix_md(hidden_aucs, surf_auc, best, bootstrap, checks, ranking):
    lines = [
        "# Sprint 2J-Fix Multi-Span Scoring Review Gate (slot alignment repaired)",
        "",
        "Verdict:",
        f"- passed: {all(checks.values())}",
        f"- ready_for_2000_rerun: false",
        f"- do_not_enter_sprint_3A: true",
        "",
        "Slot alignment:",
        "- original slot indices recomputed per span (was: reused first-span indices in 2J-B).",
        "",
        "Primary AUC results (same-question on/off-path):",
        f"- A_surface_only: {_fmt(surf_auc)}",
        f"- B_hidden_only: {_fmt(hidden_aucs['B_hidden_only'])}",
        f"- C_attention_only: {_fmt(hidden_aucs['C_attention_only'])}",
        f"- D_hidden_plus_attention: {_fmt(hidden_aucs['D_hidden_plus_attention'])}",
        f"- best gate-eligible: {best} = {_fmt(auc_of(ranking, best))}",
        "",
        "> Coverage is base-rate-sensitive and cannot be used alone as evidence of keyness ranking.",
        "",
        "Grouped bootstrap (best vs surface):",
        f"- {best}: {bootstrap.get(best)}",
        "",
        "Gate checks:",
    ]
    for k, v in checks.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def _k_md(aucs, surf_auc, best, bootstrap, diag):
    oe_only = aucs.get("J_output_effect_only")
    passed = bool(oe_only is not None and surf_auc is not None and oe_only > surf_auc
                  and bootstrap.get("J_output_effect_only", {}).get("ci95_low", -1) > 0)
    lines = [
        "# Sprint 2K Answer-Effect Keyness Review Gate",
        "",
        "Verdict:",
        f"- output_effect_beats_surface_stable: {passed}",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_sprint_3A: true",
        "",
        "Primary AUC results (same-question on/off-path):",
        f"- A_surface_only: {_fmt(surf_auc)}",
        f"- D_hidden_plus_attention: {_fmt(aucs.get('D_hidden_plus_attention'))}",
        f"- J_output_effect_only: {_fmt(oe_only)}",
        f"- K_keyness_times_output_effect: {_fmt(aucs.get('K_keyness_times_output_effect'))}",
        f"- M_output_effect_times_fragility: {_fmt(aucs.get('M_output_effect_times_fragility'))}",
        f"- N_keyness_output_effect_fragility: {_fmt(aucs.get('N_keyness_output_effect_fragility'))}",
        f"- best gate-eligible: {best} = {_fmt(aucs.get(best))}",
        "",
        "Grouped bootstrap (output-effect vs surface):",
        f"- J_output_effect_only: {bootstrap.get('J_output_effect_only')}",
        "",
        "> Output-effect = shift in the model's OWN last-token distribution under masking "
        "(leakage-safe; no gold answer). Measured at the prompt's final token.",
        "",
        "Non-number model-causal keyness (mean self-output-shift by span_type):",
    ]
    for t, d in sorted(diag.items(), key=lambda kv: -(kv[1]["mean_output_effect_shift"] or 0)):
        lines.append(f"- {t}: n={d['n']} mean_shift={d['mean_output_effect_shift']}")
    lines += ["", "Reasoning signal gap:", "- CoT / trajectory / NLA / causal attribution NOT computed (out of scope)."]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Sprint 2J-Fix + 2K")
    p.add_argument("--stage", choices=["forward", "report", "all"], default="all")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage in ("forward", "all"):
        run_forward(args.limit)
    if args.stage in ("report", "all"):
        feature_records = read_jsonl(FEATURE_MATRIX)
        score_records, formulas, ranking, bootstrap, best, surf_auc = build_reports(feature_records)
        ensure_dir(FIX_DIR)
        write_jsonl(score_records, FIX_DIR / "multi_span_score_matrix.jsonl")
        write_fix_outputs(score_records, formulas, ranking, bootstrap, best, surf_auc)
        write_k_outputs(score_records, formulas, ranking, bootstrap, best, surf_auc)
        print(f"[report] surface_auc={_fmt(surf_auc)} "
              f"D_hidden_attn={_fmt(auc_of(ranking,'D_hidden_plus_attention'))} "
              f"J_output_effect={_fmt(auc_of(ranking,'J_output_effect_only'))} best={best}={_fmt(auc_of(ranking,best))}")


if __name__ == "__main__":
    main()
