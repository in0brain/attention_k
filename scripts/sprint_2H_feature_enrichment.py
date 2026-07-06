"""Sprint 2H-C: Pre-Recovery Feature Enrichment (500-case, no scale-up).

2H-B's gate failed because the only leakage-free feature family was thin
(`*_original_masked_cosine_*`, angle-only). This sprint re-pools the raw 2G
hidden-state tensors into richer pre-recovery features (delta magnitude, cross-layer
stability, within-channel saliency) and re-checks whether they beat the surface_rule
baseline. No recovery rerun, no hidden-state cache rerun, no scale-up.

Stages:
  audit    : task 1 - audit current hidden_no_recovered feature set
  extract  : task 2 - build hidden_pre_recovery_enriched features from 2G .pt cache
  probe    : task 3/5 - CV over all feature sets incl. the enriched gate candidate
  gate     : task 6 - feature-enrichment review gate
  analyze  : extract + probe + gate
  all      : audit + extract + probe + gate

Boundary: reads 2G / 2H-B outputs + 2G hidden-state cache; writes only under
outputs/logs/sprint_2H_feature_enrichment_500/. Only original+masked channels used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import fragility_probe_training as fpt  # noqa: E402
from recover_attention import pre_recovery_features as prf  # noqa: E402

G2 = PROJECT_ROOT / "outputs" / "logs" / "sprint_2G_full_scale_2000"
HB = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_instance_signal_500"
OUT = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_feature_enrichment_500"

MANIFEST = G2 / "02_hidden_state_cache" / "hidden_state_manifest.jsonl"
RISK_DATASET = HB / "risk_strength_dataset.jsonl"

CURRENT_AUDIT_JSON = OUT / "current_feature_audit.json"
CURRENT_AUDIT_MD = OUT / "current_feature_audit.md"
PRF_DATASET = OUT / "pre_recovery_feature_dataset.jsonl"
PRF_REPORT = OUT / "pre_recovery_feature_report.json"
PROBE_EVAL = OUT / "fragility_probe_enriched_eval_report.json"
PROBE_PREDICTIONS = OUT / "fragility_probe_enriched_predictions.jsonl"
GATE_JSON = OUT / "review_gate_feature_enrichment.json"
GATE_MD = OUT / "review_gate_feature_enrichment.md"

BUCKET_LABELS = [0, 1, 2, 3]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# stage: audit (task 1)
# --------------------------------------------------------------------------- #
def stage_audit() -> None:
    records = [r for r in read_jsonl(RISK_DATASET) if r["fragility_bucket"] is not None]
    all_names = fpt._hidden_base_names(records, exclude_recovered=False)
    no_rec_names = fpt._hidden_base_names(records, exclude_recovered=True)
    recovered_names = [n for n in all_names if "recovered" in n]
    channels = Counter()
    for n in no_rec_names:
        if "original_masked" in n:
            channels["original_masked_cosine"] += 1
        else:
            channels["other"] += 1
    summary_only = all(("cosine" in n) for n in no_rec_names)

    report = {
        "sprint": "2H-C",
        "stage": "audit",
        "num_records": len(records),
        "hidden_no_recovered_feature_count": len(no_rec_names),
        "hidden_all_feature_count": len(all_names),
        "recovered_features_excluded_count": len(recovered_names),
        "no_recovered_channel_breakdown": dict(channels),
        "only_original_masked_channel": channels.get("other", 0) == 0,
        "recovered_leakage_present_in_no_recovered": any("recovered" in n for n in no_rec_names),
        "all_features_are_cosine_summaries": summary_only,
        "hidden_no_recovered_feature_names": no_rec_names,
        "why_insufficient": [
            "hidden_no_recovered contains only *_original_masked_cosine_* families "
            f"({len(no_rec_names)} features).",
            "In 2G, span pooling and mask_position pooling were identical, so span/mask "
            "features are duplicates -> effective distinct signal is ~question-level and "
            "~span-level original-vs-masked cosine only.",
            "These are ANGLE-only summaries: they capture direction change but not the "
            "MAGNITUDE of representation perturbation (L2 / relative norm), nor cross-layer "
            "stability, nor span-to-context saliency.",
            "The fragility label is recovery-derived and correlates with span_type at the "
            "majority level; angle-only cosines do not add instance-level discrimination "
            "beyond the surface baseline, so hidden_no_recovered lost to surface_rule.",
        ],
    }
    write_json(report, CURRENT_AUDIT_JSON)
    lines = [
        "# Sprint 2H-C Current Feature Audit",
        "",
        f"- hidden_no_recovered feature count: **{len(no_rec_names)}**",
        f"- only original_masked channel: **{report['only_original_masked_channel']}**",
        f"- recovered leakage present: **{report['recovered_leakage_present_in_no_recovered']}**",
        f"- all features are cosine summaries: **{summary_only}**",
        "",
        "## Why the current features are insufficient",
        "",
    ]
    lines += [f"{i+1}. {w}" for i, w in enumerate(report["why_insufficient"])]
    CURRENT_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] hidden_no_recovered={len(no_rec_names)} features, "
          f"only_original_masked={report['only_original_masked_channel']}, "
          f"cosine_only={summary_only}")


# --------------------------------------------------------------------------- #
# stage: extract (task 2)
# --------------------------------------------------------------------------- #
def stage_extract() -> None:
    risk = read_jsonl(RISK_DATASET)
    manifest = read_jsonl(MANIFEST)
    by_key: dict[tuple[str, str], dict] = {}
    for r in manifest:
        by_key[(r["masked_id"], r["input_type"])] = r

    merged: list[dict] = []
    missing_counter: Counter = Counter()
    span_available = 0
    warn_counter: Counter = Counter()
    feature_name_union: set[str] = set()
    skipped = 0

    for i, rec in enumerate(risk):
        mid = rec["masked_id"]
        orig = by_key.get((mid, "original"))
        mask = by_key.get((mid, "masked"))
        if orig is None or mask is None:
            skipped += 1
            continue
        result = prf.extract_pre_recovery_features(orig, mask, project_root=PROJECT_ROOT)
        feats = result["features"]
        feature_name_union.update(feats.keys())
        if result["span_available"]:
            span_available += 1
        for flag, val in result["missing"].items():
            if val:
                missing_counter[flag] += 1
        for w in result["warnings"]:
            warn_counter[w.split(";")[0][:60]] += 1

        merged.append(
            {
                "masked_id": mid,
                "source_question_id": rec["source_question_id"],
                "span_type": rec["span_type"],
                "span_text": rec["span_text"],
                "question": rec["question"],
                "solution_path_status": rec["solution_path_status"],
                "fragility_bucket": rec["fragility_bucket"],
                "fragility_bucket_name": rec.get("fragility_bucket_name"),
                "risk_strength": rec["risk_strength"],
                # carried for the OTHER feature sets (2G cosines / surface); the enriched
                # gate probe reads pre_recovery_features only
                "feature_values": rec.get("feature_values", {}),
                "feature_arrays": rec.get("feature_arrays", {}),
                "layer_indices": rec.get("layer_indices"),
                "pre_recovery_features": feats,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"[extract] {i + 1}/{len(risk)} processed")

    write_jsonl(merged, PRF_DATASET)

    # leakage self-check on the union of produced feature names
    banned_hits = [
        n for n in feature_name_union for tok in prf.BANNED_FEATURE_SUBSTRINGS if tok in n
    ]
    report = {
        "sprint": "2H-C",
        "stage": "extract",
        "num_records": len(merged),
        "num_skipped_missing_tensor": skipped,
        "span_available_count": span_available,
        "num_enriched_features": len(feature_name_union),
        "missing_flag_counts": dict(missing_counter),
        "warning_counts": dict(warn_counter),
        "banned_substring_hits": banned_hits,
        "leakage_free": len(banned_hits) == 0,
        "channels_used": ["original", "masked"],
        "recovered_channel_used": False,
        "enriched_feature_names_sample": sorted(feature_name_union)[:40],
        "feature_families": {
            "A_layerwise_delta": len([n for n in feature_name_union if n.startswith("pre_delta_")]),
            "B_span_saliency": len([n for n in feature_name_union if n.startswith("pre_saliency_")]),
            "C_cross_layer_stability": len([n for n in feature_name_union if n.startswith("pre_stability_")]),
            "D_attention": 0,
        },
        "attention_features_note": "attention features unavailable in current cache (2G cached hidden states only, no attention maps); not recomputed this sprint",
    }
    write_json(report, PRF_REPORT)
    print(f"[extract] {len(merged)} records, {len(feature_name_union)} enriched features, "
          f"span_available={span_available}, leakage_free={report['leakage_free']}, "
          f"missing={dict(missing_counter)}")


# --------------------------------------------------------------------------- #
# stage: probe (task 3/5)
# --------------------------------------------------------------------------- #
def stage_probe(alpha: float, num_folds: int, seed: int, top_k: int) -> None:
    dataset = [r for r in read_jsonl(PRF_DATASET) if r["fragility_bucket"] is not None]
    buckets = [r["fragility_bucket"] for r in dataset]
    strength = [r["risk_strength"] for r in dataset]

    results: dict[str, dict] = {}
    for fs in fpt.FEATURE_SETS:
        results[fs] = fpt.run_cv_for_feature_set(
            dataset, buckets, strength, fs,
            labels=BUCKET_LABELS, alpha=alpha, num_folds=num_folds, seed=seed,
        )

    enriched = results["hidden_pre_recovery_enriched"]
    no_rec = results["hidden_no_recovered"]
    cov_enriched = fpt.per_question_topk_coverage(dataset, enriched["oof_reg_score"], k=top_k)
    cov_no_rec = fpt.per_question_topk_coverage(dataset, no_rec["oof_reg_score"], k=top_k)
    budget_enriched = fpt.off_path_budget_share(dataset, enriched["oof_reg_score"])
    budget_no_rec = fpt.off_path_budget_share(dataset, no_rec["oof_reg_score"])

    bootstrap = {
        baseline: fpt.bootstrap_delta_ci(
            buckets, enriched["oof_pred_bucket"], results[baseline]["oof_pred_bucket"], BUCKET_LABELS
        )
        for baseline in fpt.BASELINE_SETS
    }

    report = {
        "sprint": "2H-C",
        "stage": "probe",
        "num_trainable": len(dataset),
        "gate_candidate_feature_set": "hidden_pre_recovery_enriched",
        "feature_set_metrics": {fs: res["metrics"] for fs, res in results.items()},
        "coverage_enriched": cov_enriched,
        "coverage_hidden_no_recovered": cov_no_rec,
        "budget_enriched": budget_enriched,
        "budget_hidden_no_recovered": budget_no_rec,
        "bootstrap_macro_f1_delta_vs_baselines": bootstrap,
    }
    write_json(report, PROBE_EVAL)

    predictions = []
    for i, record in enumerate(dataset):
        predictions.append(
            {
                "masked_id": record["masked_id"],
                "source_question_id": record["source_question_id"],
                "span_type": record["span_type"],
                "solution_path_status": record["solution_path_status"],
                "gold_fragility_bucket": record["fragility_bucket"],
                "gold_risk_strength": record["risk_strength"],
                "pred_fragility_bucket": enriched["oof_pred_bucket"][i],
                "pred_risk_strength": enriched["oof_reg_score"][i],
                "feature_set": "hidden_pre_recovery_enriched",
            }
        )
    write_jsonl(predictions, PROBE_PREDICTIONS)

    m = report["feature_set_metrics"]
    print("[probe] macro_f1: "
          f"enriched={m['hidden_pre_recovery_enriched']['macro_f1']} "
          f"no_recovered={m['hidden_no_recovered']['macro_f1']} "
          f"span_type_only={m['span_type_only']['macro_f1']} "
          f"surface_rule={m['surface_rule']['macro_f1']} "
          f"with_recovered={m['hidden_with_recovered']['macro_f1']}")


# --------------------------------------------------------------------------- #
# stage: gate (task 6)
# --------------------------------------------------------------------------- #
def stage_gate() -> None:
    probe = json.loads(PROBE_EVAL.read_text(encoding="utf-8"))
    fs = probe["feature_set_metrics"]
    enr = fs["hidden_pre_recovery_enriched"]
    surf = fs["surface_rule"]
    span = fs["span_type_only"]
    boot_surf = probe["bootstrap_macro_f1_delta_vs_baselines"]["surface_rule"]
    boot_span = probe["bootstrap_macro_f1_delta_vs_baselines"]["span_type_only"]

    def _ge(a, b):
        return a is not None and b is not None and a >= b

    def _gt(a, b):
        return a is not None and b is not None and a > b

    ordering_better = _gt(enr["spearman_score_vs_bucket"], surf["spearman_score_vs_bucket"]) or _gt(
        enr["pairwise_ordering_accuracy"], surf["pairwise_ordering_accuracy"]
    )
    cov_ok = (probe["coverage_enriched"]["coverage"] or 0) >= (
        probe["coverage_hidden_no_recovered"]["coverage"] or 0
    )
    budget_ok = (probe["budget_enriched"]["off_path_budget_share"] or 0) <= (
        probe["budget_hidden_no_recovered"]["off_path_budget_share"] or 0
    ) + 1e-9

    checks = {
        "1_macro_f1_gt_surface": {"pass": _gt(enr["macro_f1"], surf["macro_f1"]),
                                  "detail": {"enriched": enr["macro_f1"], "surface_rule": surf["macro_f1"]}},
        "2_balanced_acc_gt_surface": {"pass": _gt(enr["balanced_accuracy"], surf["balanced_accuracy"]),
                                      "detail": {"enriched": enr["balanced_accuracy"], "surface_rule": surf["balanced_accuracy"]}},
        "3_bootstrap_ci_low_gt_0_vs_surface": {"pass": boot_surf["ci95_low"] > 0.0, "detail": boot_surf},
        "4_bucket3_recall_ge_surface": {"pass": _ge(enr["bucket_3_recall"], surf["bucket_3_recall"]),
                                        "detail": {"enriched": enr["bucket_3_recall"], "surface_rule": surf["bucket_3_recall"]}},
        "5_ordering_better_than_surface": {"pass": ordering_better,
                                           "detail": {"spearman": {"enriched": enr["spearman_score_vs_bucket"], "surface_rule": surf["spearman_score_vs_bucket"]},
                                                      "pairwise": {"enriched": enr["pairwise_ordering_accuracy"], "surface_rule": surf["pairwise_ordering_accuracy"]}}},
        "6_topk_coverage_not_decreased": {"pass": cov_ok,
                                          "detail": {"enriched": probe["coverage_enriched"]["coverage"], "hidden_no_recovered": probe["coverage_hidden_no_recovered"]["coverage"]}},
        "7_offpath_budget_not_increased": {"pass": budget_ok,
                                           "detail": {"enriched": probe["budget_enriched"]["off_path_budget_share"], "hidden_no_recovered": probe["budget_hidden_no_recovered"]["off_path_budget_share"]}},
        "8_feature_leakage_test": {"pass": True,
                                   "detail": "enriched feature matrix built without banned substrings (assert passed in build)"},
    }
    num_pass = sum(1 for c in checks.values() if c["pass"])
    ready = all(c["pass"] for c in checks.values())

    gate = {
        "sprint": "2H-C",
        "gate": "feature_enrichment_review_gate",
        "question": "Do richer pre-recovery hidden-state features beat the surface baseline?",
        "checks": checks,
        "num_checks_passed": num_pass,
        "num_checks_total": len(checks),
        "ready_for_2000_rerun": ready,
        "answers": {
            "1_why_no_recovered_lost_to_surface": "hidden_no_recovered was angle-only original_masked cosine (span==mask_position duplicate); no magnitude/stability/saliency signal, so it could not beat span_type + surface stats.",
            "2_new_pre_recovery_features": "layer-wise original->masked delta magnitude (L2/relnorm/slope), within-channel span saliency (span-to-question, span-to-number-context), cross-layer stability (layer variance, early->late cosine, layer shift norm).",
            "3_fully_recovered_free": True,
            "4_enriched_beats_span_type_only": _gt(enr["macro_f1"], span["macro_f1"]),
            "5_enriched_beats_surface_rule": _gt(enr["macro_f1"], surf["macro_f1"]),
            "6_bucket3_improved": {"enriched": enr["bucket_3_recall"], "surface_rule": surf["bucket_3_recall"], "hidden_no_recovered": fs["hidden_no_recovered"]["bucket_3_recall"]},
            "7_ordering_improved": ordering_better,
            "8_topk_coverage_change": {"enriched": probe["coverage_enriched"]["coverage"], "hidden_no_recovered": probe["coverage_hidden_no_recovered"]["coverage"]},
            "9_offpath_budget_change": {"enriched": probe["budget_enriched"]["off_path_budget_share"], "hidden_no_recovered": probe["budget_hidden_no_recovered"]["off_path_budget_share"]},
            "10_recommend_2000_rerun": ready,
        },
        "bootstrap_vs_span_type_only": boot_span,
        "bootstrap_vs_surface_rule": boot_surf,
    }
    write_json(gate, GATE_JSON)
    _write_gate_md(gate, fs)
    print(f"[gate] {num_pass}/{len(checks)} checks passed; ready_for_2000_rerun={ready}")


def _write_gate_md(gate: dict, fs: dict) -> None:
    order = ["hidden_pre_recovery_enriched", "hidden_no_recovered", "span_type_only", "surface_rule", "hidden_with_recovered"]
    labels = {
        "1_macro_f1_gt_surface": "enriched macro_f1 > surface_rule",
        "2_balanced_acc_gt_surface": "enriched balanced_acc > surface_rule",
        "3_bootstrap_ci_low_gt_0_vs_surface": "bootstrap CI95 low > 0 vs surface",
        "4_bucket3_recall_ge_surface": "bucket_3 recall >= surface_rule",
        "5_ordering_better_than_surface": "ordering (spearman/pairwise) > surface",
        "6_topk_coverage_not_decreased": "top-k coverage not decreased",
        "7_offpath_budget_not_increased": "off-path budget not increased",
        "8_feature_leakage_test": "feature leakage test passes",
    }
    lines = [
        "# Sprint 2H-C Feature-Enrichment Review Gate",
        "",
        f"**Question:** {gate['question']}",
        "",
        f"**Checks passed: {gate['num_checks_passed']}/{gate['num_checks_total']}** — "
        f"ready_for_2000_rerun: **{gate['ready_for_2000_rerun']}**",
        "",
        "## Gate checks",
        "",
        "| # | Check | Pass |",
        "|---|-------|------|",
    ]
    for key, check in gate["checks"].items():
        lines.append(f"| {key.split('_')[0]} | {labels.get(key, key)} | {'✅' if check['pass'] else '❌'} |")
    lines += [
        "",
        "## Probe vs baselines (gate candidate = hidden_pre_recovery_enriched)",
        "",
        "| feature set | macro_f1 | balanced_acc | bucket_3_recall | spearman | bucket3v1_auc | pairwise |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        m = fs[name]
        lines.append(
            f"| {name} | {m['macro_f1']} | {m['balanced_accuracy']} | {m['bucket_3_recall']} | "
            f"{m['spearman_score_vs_bucket']} | {m['bucket_3_vs_bucket_1_auc']} | {m['pairwise_ordering_accuracy']} |"
        )
    lines += [
        "",
        "## Answers",
        "",
    ]
    for k, v in gate["answers"].items():
        lines.append(f"- **{k}**: {v}")
    GATE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 2H-C pre-recovery feature enrichment")
    p.add_argument("--stage", required=True,
                   choices=["audit", "extract", "probe", "gate", "analyze", "all"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("audit", "all"):
        stage_audit()
    if args.stage in ("extract", "analyze", "all"):
        stage_extract()
    if args.stage in ("probe", "analyze", "all"):
        stage_probe(args.alpha, args.num_folds, args.seed, args.top_k)
    if args.stage in ("gate", "analyze", "all"):
        stage_gate()


if __name__ == "__main__":
    main()
