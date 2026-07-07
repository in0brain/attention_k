"""Sprint 2I stage 2: evaluate attention feature sets with the 2H-D ordinal/budget framework.

Reuses run_ordinal_cv from the 2H-D script (per-fold calibration, all four scoring
methods, budget-aware bucket-3 curves) across:
    span_type_only, surface_rule, hidden_pre_recovery_enriched,
    attention_pre_recovery, hidden_plus_attention_pre_recovery (gate candidate).

Gate (task 7) requires the gate candidate to beat surface_rule AND hidden-only enriched,
with agreement across >=2 scoring methods (directly targeting 2H-D's fragility).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import attention_features as af  # noqa: E402
from recover_attention import fragility_probe_training as fpt  # noqa: E402
from recover_attention import ordinal_calibration as oc  # noqa: E402


def _load_2hd_module():
    spec = importlib.util.spec_from_file_location(
        "sprint_2H_ordinal_calibration", PROJECT_ROOT / "scripts" / "sprint_2H_ordinal_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2HD = _load_2hd_module()
METHODS = S2HD.METHODS

FEAT_OUT = PROJECT_ROOT / "outputs" / "logs" / "sprint_2I_attention_features_500"
DATASET = FEAT_OUT / "attention_feature_dataset.jsonl"
PROBE_EVAL = FEAT_OUT / "attention_probe_eval_report.json"
ORDINAL_REPORT = FEAT_OUT / "attention_ordinal_calibration_report.json"
BUDGET_CURVE = FEAT_OUT / "attention_budget_curve.json"
GATE_JSON = FEAT_OUT / "review_gate_attention_features.json"
GATE_MD = FEAT_OUT / "review_gate_attention_features.md"

FEATURE_SETS_2I = [
    "span_type_only",
    "surface_rule",
    "hidden_pre_recovery_enriched",
    "attention_pre_recovery",
    "hidden_plus_attention_pre_recovery",
]
GATE_CANDIDATE = "hidden_plus_attention_pre_recovery"
BUCKET_LABELS = [0, 1, 2, 3]
BUDGET_TOL = 0.02
MACRO_TOL = 0.02


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _feature_family(name: str) -> str:
    if name.startswith("attn_delta_"):
        return "E_delta"
    if "_to_slot" in name:
        return "C_context_to_slot"
    if any(k in name for k in ["entropy", "top1", "top3", "top5", "edge_count"]):
        return "B_shape"
    if any(k in name for k in ["in_mass", "out_mass", "self_mass", "rank", "in_rel"]):
        return "A_mass"
    return "other"


def _family_usefulness(records, buckets):
    """Univariate |Spearman(feature, bucket)| grouped by attention family."""
    names = sorted({k for r in records for k in (r.get("attention_features") or {}).keys()})
    b = np.array(buckets, dtype=float)
    by_family: dict[str, list[float]] = {}
    for name in names:
        col = np.array([float((r.get("attention_features") or {}).get(name, 0.0)) for r in records])
        s = oc.spearman_corr(col, b)
        if s is None:
            continue
        by_family.setdefault(_feature_family(name), []).append(abs(s))
    return {fam: round(float(np.mean(v)), 4) for fam, v in sorted(by_family.items(), key=lambda kv: -np.mean(kv[1]))}


def stage_run(alpha, num_folds, seed, top_k):
    records = [r for r in read_jsonl(DATASET) if r["fragility_bucket"] is not None]
    buckets = [r["fragility_bucket"] for r in records]
    strength = [r["risk_strength"] for r in records]

    # explicit leakage guard on the gate-eligible attention sets
    for fs in ("attention_pre_recovery", GATE_CANDIDATE):
        built = fpt.build_feature_matrix(records, fs)
        af.assert_no_banned_attention_names(built["feature_names"])
        fpt.assert_no_recovered_features(built["feature_names"])

    results = {
        fs: S2HD.run_ordinal_cv(records, buckets, strength, fs, alpha=alpha, num_folds=num_folds, seed=seed)
        for fs in FEATURE_SETS_2I
    }
    cand = results[GATE_CANDIDATE]
    surf = results["surface_rule"]
    henr = results["hidden_pre_recovery_enriched"]
    attn = results["attention_pre_recovery"]
    bucket_arr = np.array(buckets)

    def sp(res, m):
        v = res["methods"][m]["spearman"]
        return v if v is not None else -2.0

    primary = max(METHODS, key=lambda m: sp(cand, m))

    cand_primary = np.array(cand["oof"][primary])
    surf_primary = np.array(surf["oof"][primary])
    henr_primary = np.array(henr["oof"][primary])

    cov = fpt.per_question_topk_coverage(records, cand["oof"][primary], k=top_k)
    budget = fpt.off_path_budget_share(records, cand["oof"][primary])
    henr_budget = fpt.off_path_budget_share(records, henr["oof"][primary])

    boot_vs_surf = oc.bootstrap_ranking_delta(cand_primary, surf_primary, bucket_arr, metric="spearman", seed=seed)
    boot_vs_henr = oc.bootstrap_ranking_delta(cand_primary, henr_primary, bucket_arr, metric="spearman", seed=seed)
    boot_macro_vs_surf = fpt.bootstrap_delta_ci(buckets, cand["oof"]["pred_bucket"], surf["oof"]["pred_bucket"], BUCKET_LABELS)

    # >=2 scoring methods agree that candidate out-ranks surface
    methods_cand_gt_surf = [m for m in METHODS if sp(cand, m) > sp(surf, m)]

    report = {
        "sprint": "2I",
        "num_trainable": len(records),
        "gate_candidate": GATE_CANDIDATE,
        "primary_method": primary,
        "primary_method_selection_rule": "surface-blind: argmax over methods of gate-candidate Spearman",
        "feature_set_results": {
            fs: {
                "classification_macro_f1": results[fs]["classification"]["macro_f1"],
                "classification_balanced_accuracy": results[fs]["classification"]["balanced_accuracy"],
                "classification_bucket_3_recall": results[fs]["classification"]["bucket_3_recall"],
                "methods": {m: {k: v for k, v in results[fs]["methods"][m].items() if k != "budget_curve"} for m in METHODS},
                "budget_curve_primary": results[fs]["methods"][primary]["budget_curve"],
            }
            for fs in FEATURE_SETS_2I
        },
        "primary_coverage_candidate": cov,
        "primary_off_path_budget_candidate": budget,
        "off_path_budget_hidden_enriched": henr_budget,
        "bootstrap_spearman_candidate_vs_surface": boot_vs_surf,
        "bootstrap_spearman_candidate_vs_hidden_enriched": boot_vs_henr,
        "bootstrap_macro_f1_candidate_vs_surface": boot_macro_vs_surf,
        "methods_with_candidate_spearman_gt_surface": methods_cand_gt_surf,
        "attention_family_usefulness": _family_usefulness(records, buckets),
    }
    report["answers"] = _answers(report, results, primary)
    write_json(report, ORDINAL_REPORT)
    write_json({
        "sprint": "2I",
        "feature_set_classification": {fs: results[fs]["classification"] for fs in FEATURE_SETS_2I},
        "gate_candidate": GATE_CANDIDATE,
        "primary_method": primary,
    }, PROBE_EVAL)
    write_json({
        "primary_method": primary,
        "budget_curves": {fs: {m: results[fs]["methods"][m]["budget_curve"] for m in METHODS} for fs in FEATURE_SETS_2I},
    }, BUDGET_CURVE)

    _gate(report, results, primary)
    print(f"[2I] primary={primary}; cand spearman={cand['methods'][primary]['spearman']} "
          f"surf={surf['methods'][primary]['spearman']} henr={henr['methods'][primary]['spearman']}; "
          f"attn_alone spearman={attn['methods'][primary]['spearman']}")


def _answers(report, results, primary):
    fs = report["feature_set_results"]
    cand = fs[GATE_CANDIDATE]
    surf = fs["surface_rule"]
    henr = fs["hidden_pre_recovery_enriched"]
    attn = fs["attention_pre_recovery"]
    cand_top10 = cand["budget_curve_primary"]["points"]["top_10pct"]
    surf_top10 = surf["budget_curve_primary"]["points"]["top_10pct"]
    return {
        "1_attention_alone_beats_surface": {
            "attn_macro_f1": attn["classification_macro_f1"], "surface_macro_f1": surf["classification_macro_f1"],
            "attn_spearman": attn["methods"][primary]["spearman"], "surface_spearman": surf["methods"][primary]["spearman"],
        },
        "2_hidden_plus_attention_beats_hidden_only": {
            "cand_macro_f1": cand["classification_macro_f1"], "hidden_macro_f1": henr["classification_macro_f1"],
            "cand_spearman": cand["methods"][primary]["spearman"], "hidden_spearman": henr["methods"][primary]["spearman"],
        },
        "3_what_improves": "compare cand vs hidden_enriched: macro_f1 delta {:+.3f}, spearman delta {:+.3f}, top10 precision delta {:+.3f}".format(
            cand["classification_macro_f1"] - henr["classification_macro_f1"],
            (cand["methods"][primary]["spearman"] or 0) - (henr["methods"][primary]["spearman"] or 0),
            cand_top10["bucket_3_precision"] - henr["budget_curve_primary"]["points"]["top_10pct"]["bucket_3_precision"],
        ),
        "4_bucket3_top_budget": {"cand_top10": cand_top10, "surface_top10": surf_top10},
        "5_offpath_budget": {"candidate": report["primary_off_path_budget_candidate"]["off_path_budget_share"],
                             "hidden_enriched": report["off_path_budget_hidden_enriched"]["off_path_budget_share"]},
        "6_most_useful_family": report["attention_family_usefulness"],
        "7_recommend_2000_rerun": None,
        "8_next_if_fail": "trajectory-level features (answer/trajectory stability over multi-step generation) or hybrid recovery-guided pipeline",
    }


def _gate(report, results, primary):
    fs = report["feature_set_results"]
    cand = fs[GATE_CANDIDATE]
    surf = fs["surface_rule"]
    henr = fs["hidden_pre_recovery_enriched"]
    cm = cand["methods"][primary]
    sm = surf["methods"][primary]
    hm = henr["methods"][primary]
    boot_surf = report["bootstrap_spearman_candidate_vs_surface"]
    cand_top10 = cand["budget_curve_primary"]["points"]["top_10pct"]
    cand_top20 = cand["budget_curve_primary"]["points"]["top_20pct"]
    surf_top10 = surf["budget_curve_primary"]["points"]["top_10pct"]
    surf_top20 = surf["budget_curve_primary"]["points"]["top_20pct"]

    def gt(a, b):
        return a is not None and b is not None and a > b

    budget_ok = (report["primary_off_path_budget_candidate"]["off_path_budget_share"] or 0) <= (
        report["off_path_budget_hidden_enriched"]["off_path_budget_share"] or 0) + BUDGET_TOL
    cov_ok = (report["primary_coverage_candidate"]["coverage"] or 0) >= 1.0
    top_budget_ok = gt(cand_top10["bucket_3_precision"], surf_top10["bucket_3_precision"]) or \
        gt(cand_top10["bucket_3_f1"], surf_top10["bucket_3_f1"]) or \
        gt(cand_top20["bucket_3_precision"], surf_top20["bucket_3_precision"]) or \
        gt(cand_top20["bucket_3_f1"], surf_top20["bucket_3_f1"])

    checks = {
        "1_spearman_significantly_gt_surface": {"pass": bool(boot_surf["ci95_low"] is not None and boot_surf["ci95_low"] > 0),
                                                "detail": boot_surf},
        "2_spearman_gt_hidden_only": {"pass": gt(cm["spearman"], hm["spearman"]),
                                      "detail": {"candidate": cm["spearman"], "hidden_only": hm["spearman"]}},
        "3_top_budget_bucket3_gt_surface": {"pass": top_budget_ok,
                                            "detail": {"cand_top10": cand_top10, "surface_top10": surf_top10, "cand_top20": cand_top20, "surface_top20": surf_top20}},
        "4_offpath_budget_not_worse": {"pass": budget_ok,
                                       "detail": {"candidate": report["primary_off_path_budget_candidate"]["off_path_budget_share"], "hidden_enriched": report["off_path_budget_hidden_enriched"]["off_path_budget_share"]}},
        "5_coverage_not_decreased": {"pass": cov_ok, "detail": report["primary_coverage_candidate"]},
        "6_feature_leakage_test": {"pass": True, "detail": "attention + fusion feature names passed banned-substring assertion in build"},
        "7_at_least_two_methods_agree": {"pass": len(report["methods_with_candidate_spearman_gt_surface"]) >= 2,
                                         "detail": report["methods_with_candidate_spearman_gt_surface"]},
    }
    num_pass = sum(1 for c in checks.values() if c["pass"])
    ready = all(c["pass"] for c in checks.values())

    gate = {
        "sprint": "2I",
        "gate": "attention_features_review_gate",
        "primary_method": primary,
        "gate_candidate": GATE_CANDIDATE,
        "checks": checks,
        "num_checks_passed": num_pass,
        "num_checks_total": len(checks),
        "ready_for_2000_rerun": ready,
        "next_step": "Sprint 2J: 2000-case hidden+attention pre-recovery rerun" if ready
        else "do not scale; turn to trajectory-level features or hybrid recovery-guided pipeline",
    }
    report["answers"]["7_recommend_2000_rerun"] = ready
    write_json(report, ORDINAL_REPORT)
    write_json(gate, GATE_JSON)
    _gate_md(gate, report)
    print(f"[gate] {num_pass}/{len(checks)} checks passed; ready_for_2000_rerun={ready}")


def _gate_md(gate, report):
    fs = report["feature_set_results"]
    primary = gate["primary_method"]
    labels = {
        "1_spearman_significantly_gt_surface": "Spearman significantly > surface (bootstrap CI low>0)",
        "2_spearman_gt_hidden_only": "Spearman > hidden-only enriched",
        "3_top_budget_bucket3_gt_surface": "top-budget bucket-3 precision/F1 > surface",
        "4_offpath_budget_not_worse": "off-path budget not worse",
        "5_coverage_not_decreased": "top-k coverage not decreased",
        "6_feature_leakage_test": "feature leakage test passes",
        "7_at_least_two_methods_agree": ">=2 scoring methods agree",
    }
    lines = [
        "# Sprint 2I Attention-Features Review Gate",
        "",
        f"Gate candidate: **{gate['gate_candidate']}**, primary method: **{primary}**",
        "",
        f"**Checks passed: {gate['num_checks_passed']}/{gate['num_checks_total']}** — "
        f"ready_for_2000_rerun: **{gate['ready_for_2000_rerun']}**",
        "",
        "| # | Check | Pass |",
        "|---|-------|------|",
    ]
    for k, c in gate["checks"].items():
        lines.append(f"| {k.split('_')[0]} | {labels.get(k, k)} | {'✅' if c['pass'] else '❌'} |")
    lines += [
        "",
        "## Classification + ranking by feature set (primary method)",
        "",
        "| feature set | macro_f1 | bal_acc | spearman | pairwise | b3v1_auc |",
        "|---|---|---|---|---|---|",
    ]
    for name in FEATURE_SETS_2I:
        m = fs[name]
        pm = m["methods"][primary]
        lines.append(f"| {name} | {m['classification_macro_f1']} | {m['classification_balanced_accuracy']} | "
                     f"{pm['spearman']} | {pm['pairwise_ordering_accuracy']} | {pm['bucket_3_vs_bucket_1_auc']} |")
    lines += ["", "## Attention family usefulness (|Spearman| vs bucket)", "",
              f"{report['attention_family_usefulness']}", "",
              f"**Next step:** {gate['next_step']}", ""]
    GATE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 2I attention probe + gate")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    stage_run(args.alpha, args.num_folds, args.seed, args.top_k)


if __name__ == "__main__":
    main()
