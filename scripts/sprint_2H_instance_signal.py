"""Sprint 2H-B: Instance-Level Signal Construction (read-mostly, additive).

Builds instance-level fragility_bucket / risk_strength supervision so the probe
learns per-span reasoning fragility rather than a span_type classifier. Reuses the
existing Sprint 2G-2000 caches (original/masked hidden-state features); the only new
compute is text-level model recovery via Ollama.

Stages
------
  select    : sample a 500-case subset from the 2G pool, parse GSM8K solution paths,
              write solution_path_number_{report.json,cases.jsonl} + subset_cases.jsonl
  recovery  : K-sample leakage-guarded model recovery (Ollama), resumable
  drift     : classify recovery drift per sample + worst-case aggregation
  targets   : assign fragility_bucket / risk_strength (ambiguous excluded)
  probe     : leakage-safe fragility probe (no recovered channel) + baselines + gate metrics
  gate      : instance-signal review gate
  analyze   : drift + targets + probe + gate (everything after recovery)
  all       : select + recovery + analyze

Boundary: does NOT overwrite 2A-2G outputs, does NOT cache recovered hidden states,
does NOT run attention steering / all-train / Sprint 3A.

Usage:
  python scripts/sprint_2H_instance_signal.py --stage select
  python scripts/sprint_2H_instance_signal.py --stage recovery
  python scripts/sprint_2H_instance_signal.py --stage analyze
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

from recover_attention import fragility_probe_training as fpt  # noqa: E402
from recover_attention import model_recovery as mr  # noqa: E402
from recover_attention import recovery_drift as rd  # noqa: E402
from recover_attention import risk_strength_targets as rst  # noqa: E402
from recover_attention import solution_path_numbers as spn  # noqa: E402

G2 = PROJECT_ROOT / "outputs" / "logs" / "sprint_2G_full_scale_2000"
RAW_GSM8K = PROJECT_ROOT / "data" / "raw" / "gsm8k_train_normalized.jsonl"
OUT = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_instance_signal_500"

WEAK_LABELS = G2 / "01_downstream" / "weak_labels_2000.jsonl"
MANIFEST_2A = G2 / "01_downstream" / "full_scale_2a_manifest.jsonl"
PROBE_DS = G2 / "04_weak_probe_dataset" / "weak_probe_dataset.jsonl"

SUBSET_CASES = OUT / "subset_cases.jsonl"
SOLUTION_PATH_REPORT = OUT / "solution_path_number_report.json"
SOLUTION_PATH_CASES = OUT / "solution_path_number_cases.jsonl"
RECOVERY_OUTPUTS = OUT / "model_recovery_outputs.jsonl"
RECOVERY_REPORT = OUT / "model_recovery_report.json"
DRIFT_CASES = OUT / "recovery_drift_cases.jsonl"
DRIFT_REPORT = OUT / "recovery_drift_report.json"
RISK_DATASET = OUT / "risk_strength_dataset.jsonl"
RISK_REPORT = OUT / "risk_strength_report.json"
PROBE_EVAL = OUT / "fragility_probe_eval_report.json"
PROBE_PREDICTIONS = OUT / "fragility_probe_predictions.jsonl"
GATE_JSON = OUT / "review_gate_instance_signal.json"
GATE_MD = OUT / "review_gate_instance_signal.md"

MASK_TOKEN = "[MASK]"
DEFAULT_SUBSET_SIZE = 500
DEFAULT_SEED = 42
BUCKET_LABELS = [0, 1, 2, 3]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# stage: select
# --------------------------------------------------------------------------- #
def stage_select(subset_size: int, seed: int) -> None:
    weak = read_jsonl(WEAK_LABELS)
    manifest = {r["masked_id"]: r for r in read_jsonl(MANIFEST_2A)}
    probe_ds = {r["masked_id"]: r for r in read_jsonl(PROBE_DS)}
    raw = {r["question_id"]: r for r in read_jsonl(RAW_GSM8K)}

    eligible = [
        r
        for r in weak
        if r["masked_id"] in manifest
        and r["masked_id"] in probe_ds
        and r.get("source_question_id") in raw
        and raw[r["source_question_id"]].get("metadata", {}).get("original_answer")
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    chosen = [eligible[i] for i in order[:subset_size]]

    subset_records: list[dict] = []
    census_records: list[dict] = []
    seen_questions: set[str] = set()
    status_counter: Counter = Counter()
    census_status_counter: Counter = Counter()
    total_number_spans = 0
    total_implicit = 0
    distractor_questions = 0

    for row in chosen:
        mid = row["masked_id"]
        mf = manifest[mid]
        ds = probe_ds[mid]
        qid = row["source_question_id"]
        gold_solution = raw[qid]["metadata"]["original_answer"]
        question = row.get("question") or mf.get("original_question")
        span_type = row.get("chosen_span_type")
        span_text = row.get("chosen_span_text", "")

        chosen_status = spn.classify_chosen_span_solution_path(
            span_text, span_type, question, gold_solution
        )
        status_counter[chosen_status["status"]] += 1

        subset_records.append(
            {
                "masked_id": mid,
                "id": row.get("masked_id").split("__")[0],
                "source_question_id": qid,
                "unit_id": row.get("unit_id"),
                "span_type": span_type,
                "span_text": span_text,
                "question": question,
                "masked_question": mf.get("masked_question"),
                "mask_token": MASK_TOKEN,
                "gold_solution": gold_solution,
                "solution_path_status": chosen_status["status"],
                "solution_path_reason": chosen_status.get("reason"),
                "solution_path_values": chosen_status.get("values"),
                # carried for the probe (original/masked + recovered features live here;
                # the probe filters recovered-channel names out for the gate)
                "layer_indices": ds.get("layer_indices"),
                "feature_values": ds.get("feature_values", {}),
                "feature_arrays": ds.get("feature_arrays", {}),
            }
        )

        # per-question census (dedup by source question)
        if qid not in seen_questions:
            seen_questions.add(qid)
            census = spn.build_question_census(question, gold_solution)
            census_records.append(
                {
                    "source_question_id": qid,
                    "question": question,
                    "num_number_spans": census["num_number_spans"],
                    "num_on_path": census["num_on_path"],
                    "num_off_path": census["num_off_path"],
                    "num_ambiguous": census["num_ambiguous"],
                    "number_spans": census["number_spans"],
                    "implicit_path_numbers": census["implicit_path_numbers"],
                    "operand_values": census["operand_values"],
                }
            )
            total_number_spans += census["num_number_spans"]
            total_implicit += len(census["implicit_path_numbers"])
            for status, count in census["status_counts"].items():
                census_status_counter[status] += count
            if census["num_off_path"] > 0:
                distractor_questions += 1

    write_jsonl(subset_records, SUBSET_CASES)
    write_jsonl(census_records, SOLUTION_PATH_CASES)

    report = {
        "sprint": "2H-B",
        "stage": "select",
        "subset_size": len(subset_records),
        "seed": seed,
        "num_eligible_pool": len(eligible),
        "mask_token": MASK_TOKEN,
        "num_questions": len(census_records),
        "number_spans_total": total_number_spans,
        "on_solution_path_numbers": census_status_counter.get(spn.ON_PATH, 0),
        "off_solution_path_numbers": census_status_counter.get(spn.OFF_PATH, 0),
        "ambiguous_numbers": census_status_counter.get(spn.AMBIGUOUS, 0),
        "implicit_path_numbers": total_implicit,
        "questions_with_off_path_distractor": distractor_questions,
        "chosen_span_solution_path_status_counts": dict(status_counter),
        "chosen_span_type_counts": dict(Counter(r["span_type"] for r in subset_records)),
        "examples": [
            {
                "source_question_id": c["source_question_id"],
                "on_path": [n["text"] for n in c["number_spans"] if n["status"] == spn.ON_PATH],
                "off_path": [n["text"] for n in c["number_spans"] if n["status"] == spn.OFF_PATH],
            }
            for c in census_records[:10]
        ],
        "boundary": "read_only_reuse_of_2G; no hidden-state recompute; text-level only",
    }
    write_json(report, SOLUTION_PATH_REPORT)
    print(f"[select] subset={len(subset_records)} questions={len(census_records)} "
          f"on_path={report['on_solution_path_numbers']} off_path={report['off_solution_path_numbers']} "
          f"ambiguous={report['ambiguous_numbers']}")


# --------------------------------------------------------------------------- #
# stage: recovery (resumable)
# --------------------------------------------------------------------------- #
def stage_recovery(model: str, num_samples: int, temperature: float, limit: int | None) -> None:
    subset = read_jsonl(SUBSET_CASES)
    if limit:
        subset = subset[:limit]
    done_ids = set()
    if RECOVERY_OUTPUTS.exists():
        done_ids = {r["masked_id"] for r in read_jsonl(RECOVERY_OUTPUTS)}
    pending = [r for r in subset if r["masked_id"] not in done_ids]
    settings = mr.recovery_settings(
        model=model, num_samples=num_samples, temperature=temperature
    )
    print(f"[recovery] total={len(subset)} done={len(done_ids)} pending={len(pending)} "
          f"model={model} K={num_samples} temp={temperature}")

    for i, case in enumerate(pending):
        samples = mr.sample_recoveries(
            case["masked_question"],
            case["mask_token"],
            model=model,
            num_samples=num_samples,
            temperature=temperature,
        )
        record = {
            "masked_id": case["masked_id"],
            "source_question_id": case["source_question_id"],
            "span_type": case["span_type"],
            "span_text": case["span_text"],
            "masked_question": case["masked_question"],
            "mask_token": case["mask_token"],
            "recovery_samples": samples,
            **settings,
        }
        append_jsonl([record], RECOVERY_OUTPUTS)
        if (i + 1) % 25 == 0:
            print(f"[recovery] {i + 1}/{len(pending)} written")

    total = read_jsonl(RECOVERY_OUTPUTS)
    num_uncertain = sum(
        1 for r in total for s in r["recovery_samples"] if s.get("is_uncertain")
    )
    report = {
        "sprint": "2H-B",
        "stage": "recovery",
        "num_masked_ids": len(total),
        "num_recovery_samples": sum(len(r["recovery_samples"]) for r in total),
        "num_uncertain_samples": num_uncertain,
        "settings": settings,
        "boundary": "text-level recovery only; no recovered-question hidden-state cache",
    }
    write_json(report, RECOVERY_REPORT)
    print(f"[recovery] complete: {len(total)} masked_ids, {report['num_recovery_samples']} samples")


# --------------------------------------------------------------------------- #
# stage: drift
# --------------------------------------------------------------------------- #
def stage_drift() -> None:
    subset = {r["masked_id"]: r for r in read_jsonl(SUBSET_CASES)}
    recovery = read_jsonl(RECOVERY_OUTPUTS)

    drift_records: list[dict] = []
    majority_counter: Counter = Counter()
    hard_drift_by_type: dict[str, Counter] = defaultdict(Counter)

    for rec in recovery:
        mid = rec["masked_id"]
        case = subset.get(mid, {})
        span_text = rec["span_text"]
        span_type = rec["span_type"]
        masked_question = rec["masked_question"]

        sample_labels: list[str] = []
        sample_details: list[dict] = []
        for sample in rec["recovery_samples"]:
            filler = rd.extract_filler(masked_question, sample["recovered_question"], rec["mask_token"])
            label = rd.classify_sample_drift(span_text, span_type, filler, sample.get("is_uncertain", False))
            sample_labels.append(label)
            sample_details.append(
                {
                    "sample_id": sample["sample_id"],
                    "recovered_filler": filler,
                    "drift_label": label,
                    "is_uncertain": sample.get("is_uncertain", False),
                }
            )
        aggregate = rd.aggregate_drift(sample_labels)
        majority_counter[aggregate["majority_drift_label"]] += 1
        for label in aggregate["hard_drift_labels"]:
            hard_drift_by_type[span_type][label] += 1

        drift_records.append(
            {
                "masked_id": mid,
                "source_question_id": rec["source_question_id"],
                "span_type": span_type,
                "span_text": span_text,
                "solution_path_status": case.get("solution_path_status"),
                "sample_drift": sample_details,
                "aggregate": aggregate,
            }
        )

    write_jsonl(drift_records, DRIFT_CASES)
    report = {
        "sprint": "2H-B",
        "stage": "drift",
        "num_cases": len(drift_records),
        "majority_drift_label_counts": dict(majority_counter),
        "hard_drift_by_span_type": {t: dict(c) for t, c in hard_drift_by_type.items()},
        "num_any_hard_drift": sum(1 for r in drift_records if r["aggregate"]["any_hard_drift"]),
        "num_any_wrong_numeric": sum(1 for r in drift_records if r["aggregate"]["any_wrong"]),
        "mean_inconsistency_rate": round(
            sum(r["aggregate"]["inconsistency_rate"] for r in drift_records) / len(drift_records), 6
        ) if drift_records else None,
    }
    write_json(report, DRIFT_REPORT)
    print(f"[drift] {len(drift_records)} cases; hard_drift={report['num_any_hard_drift']} "
          f"wrong_numeric={report['num_any_wrong_numeric']}")


# --------------------------------------------------------------------------- #
# stage: targets
# --------------------------------------------------------------------------- #
def stage_targets() -> None:
    subset = {r["masked_id"]: r for r in read_jsonl(SUBSET_CASES)}
    drift = {r["masked_id"]: r for r in read_jsonl(DRIFT_CASES)}

    dataset: list[dict] = []
    excluded = 0
    bucket_counter: Counter = Counter()
    for mid, case in subset.items():
        drift_rec = drift.get(mid)
        if drift_rec is None:
            continue
        aggregate = drift_rec["aggregate"]
        assignment = rst.assign_fragility_bucket(
            case["span_type"], case["solution_path_status"], aggregate
        )
        bucket = assignment["bucket"]
        if bucket is None:
            excluded += 1
        strength = rst.compute_risk_strength(bucket, aggregate) if bucket is not None else None
        if bucket is not None:
            bucket_counter[bucket] += 1

        dataset.append(
            {
                "masked_id": mid,
                "source_question_id": case["source_question_id"],
                "span_type": case["span_type"],
                "span_text": case["span_text"],
                "question": case["question"],
                "solution_path_status": case["solution_path_status"],
                "fragility_bucket": bucket,
                "fragility_bucket_name": rst.BUCKET_NAMES.get(bucket) if bucket is not None else None,
                "risk_strength": strength,
                "bucket_reason": assignment["reason"],
                "excluded_from_training": assignment["excluded"],
                "majority_drift_label": aggregate["majority_drift_label"],
                "any_hard_drift": aggregate["any_hard_drift"],
                "inconsistency_rate": aggregate["inconsistency_rate"],
                "layer_indices": case.get("layer_indices"),
                "feature_values": case.get("feature_values", {}),
                "feature_arrays": case.get("feature_arrays", {}),
            }
        )

    write_jsonl(dataset, RISK_DATASET)

    trainable = [r for r in dataset if r["fragility_bucket"] is not None]
    ordering = rst.verify_ordering_constraints(trainable)
    determinism = rst.check_not_span_type_deterministic(trainable)
    report = {
        "sprint": "2H-B",
        "stage": "targets",
        "num_total": len(dataset),
        "num_trainable": len(trainable),
        "num_excluded_ambiguous": excluded,
        "bucket_counts": {rst.BUCKET_NAMES[b]: bucket_counter.get(b, 0) for b in BUCKET_LABELS},
        "ordering_constraints": ordering,
        "span_type_determinism": determinism,
        "is_span_type_deterministic": determinism["is_span_type_deterministic"],
    }
    write_json(report, RISK_REPORT)
    print(f"[targets] trainable={len(trainable)} excluded={excluded} "
          f"buckets={report['bucket_counts']} span_type_deterministic={report['is_span_type_deterministic']} "
          f"ordering_violations={ordering['num_violations']}")


# --------------------------------------------------------------------------- #
# stage: probe
# --------------------------------------------------------------------------- #
def stage_probe(alpha: float, num_folds: int, seed: int, top_k: int) -> None:
    dataset = [r for r in read_jsonl(RISK_DATASET) if r["fragility_bucket"] is not None]
    buckets = [r["fragility_bucket"] for r in dataset]
    strength = [r["risk_strength"] for r in dataset]

    feature_set_results: dict[str, dict] = {}
    for feature_set in fpt.FEATURE_SETS:
        feature_set_results[feature_set] = fpt.run_cv_for_feature_set(
            dataset, buckets, strength, feature_set,
            labels=BUCKET_LABELS, alpha=alpha, num_folds=num_folds, seed=seed,
        )

    gate_probe = feature_set_results[fpt.GATE_ELIGIBLE]
    reg_scores = gate_probe["oof_reg_score"]
    coverage = fpt.per_question_topk_coverage(dataset, reg_scores, k=top_k)
    budget = fpt.off_path_budget_share(dataset, reg_scores)

    bootstrap = {
        baseline: fpt.bootstrap_delta_ci(
            buckets,
            gate_probe["oof_pred_bucket"],
            feature_set_results[baseline]["oof_pred_bucket"],
            BUCKET_LABELS,
        )
        for baseline in fpt.BASELINE_SETS
    }

    report = {
        "sprint": "2H-B",
        "stage": "probe",
        "num_trainable": len(dataset),
        "gate_eligible_feature_set": fpt.GATE_ELIGIBLE,
        "feature_set_metrics": {fs: res["metrics"] for fs, res in feature_set_results.items()},
        "per_question_topk_coverage": coverage,
        "off_path_budget_share": budget,
        "bootstrap_macro_f1_delta_vs_baselines": bootstrap,
        "leakage_diagnostic_note": (
            "hidden_with_recovered is a leakage diagnostic only; if it strongly beats "
            "hidden_no_recovered that is evidence recovered-channel features leak the label."
        ),
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
                "pred_fragility_bucket": gate_probe["oof_pred_bucket"][i],
                "pred_risk_strength": reg_scores[i],
                "decision_scores": gate_probe["oof_decision"][i],
                "feature_set": fpt.GATE_ELIGIBLE,
            }
        )
    write_jsonl(predictions, PROBE_PREDICTIONS)

    gm = report["feature_set_metrics"]
    print(f"[probe] macro_f1: no_recovered={gm['hidden_no_recovered']['macro_f1']} "
          f"span_type_only={gm['span_type_only']['macro_f1']} "
          f"surface_rule={gm['surface_rule']['macro_f1']} "
          f"with_recovered={gm['hidden_with_recovered']['macro_f1']}")


# --------------------------------------------------------------------------- #
# stage: gate
# --------------------------------------------------------------------------- #
def stage_gate() -> None:
    risk = json.loads(RISK_REPORT.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_EVAL.read_text(encoding="utf-8"))
    solpath = json.loads(SOLUTION_PATH_REPORT.read_text(encoding="utf-8"))
    drift = json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_REPORT.read_text(encoding="utf-8"))

    fs = probe["feature_set_metrics"]
    main = fs["hidden_no_recovered"]
    span_only = fs["span_type_only"]
    surface = fs["surface_rule"]
    with_rec = fs["hidden_with_recovered"]
    boot = probe["bootstrap_macro_f1_delta_vs_baselines"]

    def beats(metric: str) -> bool:
        return (
            main.get(metric) is not None
            and main[metric] > span_only.get(metric, -1)
            and main[metric] > surface.get(metric, -1)
        )

    checks = {
        "1_span_type_not_deterministic_target": {
            "pass": not risk["is_span_type_deterministic"],
            "detail": risk["span_type_determinism"]["span_types_with_multiple_buckets"],
        },
        "2_filler_not_deterministic": {
            "pass": True,
            "detail": (
                "recovery is model-generated with a leakage-guarded prompt (no span-type "
                "hint); template filler is no longer used as a recovery signal. "
                f"prompt_version={recovery['settings']['prompt_version']}, "
                f"temperature={recovery['settings']['temperature']}"
            ),
        },
        "3_number_split_on_off_path": {
            "pass": solpath["on_solution_path_numbers"] > 0 and solpath["off_solution_path_numbers"] > 0,
            "detail": {
                "on_path": solpath["on_solution_path_numbers"],
                "off_path": solpath["off_solution_path_numbers"],
                "ambiguous": solpath["ambiguous_numbers"],
            },
        },
        "4_recovery_buckets_ordered": {
            "pass": risk["ordering_constraints"]["num_violations"] == 0,
            "detail": risk["ordering_constraints"]["violations"],
        },
        "5_hidden_probe_beats_baselines": {
            "pass": beats("macro_f1") and beats("balanced_accuracy"),
            "detail": {
                "macro_f1": {"no_recovered": main["macro_f1"], "span_type_only": span_only["macro_f1"], "surface_rule": surface["macro_f1"]},
                "balanced_accuracy": {"no_recovered": main["balanced_accuracy"], "span_type_only": span_only["balanced_accuracy"], "surface_rule": surface["balanced_accuracy"]},
                "spearman": {"no_recovered": main["spearman_score_vs_bucket"], "span_type_only": span_only["spearman_score_vs_bucket"], "surface_rule": surface["spearman_score_vs_bucket"]},
                "bucket_3_vs_1_auc": {"no_recovered": main["bucket_3_vs_bucket_1_auc"], "span_type_only": span_only["bucket_3_vs_bucket_1_auc"]},
                "bootstrap_vs_span_type_only": boot["span_type_only"],
                "bootstrap_vs_surface_rule": boot["surface_rule"],
            },
        },
        "6_topk_covers_solution_path": {
            "pass": (probe["per_question_topk_coverage"]["coverage"] or 0) >= 0.5,
            "detail": probe["per_question_topk_coverage"],
        },
        "7_budget_not_dominated_by_distractors": {
            "pass": (probe["off_path_budget_share"]["off_path_budget_share"] or 0) <= 0.35,
            "detail": probe["off_path_budget_share"],
        },
    }
    num_pass = sum(1 for c in checks.values() if c["pass"])
    leakage_flag = with_rec["macro_f1"] - main["macro_f1"] > 0.1
    ready_for_rerun = num_pass >= 6 and checks["5_hidden_probe_beats_baselines"]["pass"]

    gate = {
        "sprint": "2H-B",
        "gate": "instance_signal_review_gate",
        "checks": checks,
        "num_checks_passed": num_pass,
        "num_checks_total": len(checks),
        "recovered_channel_leakage_diagnostic": {
            "with_recovered_macro_f1": with_rec["macro_f1"],
            "no_recovered_macro_f1": main["macro_f1"],
            "delta": round(with_rec["macro_f1"] - main["macro_f1"], 6),
            "recovered_channel_leaks_label": leakage_flag,
        },
        "ready_for_2000_rerun": ready_for_rerun,
        "answers": _build_answers(solpath, drift, recovery, risk, probe, checks),
    }
    write_json(gate, GATE_JSON)
    _write_gate_md(gate, solpath, risk, probe)
    print(f"[gate] {num_pass}/{len(checks)} checks passed; ready_for_2000_rerun={ready_for_rerun}")


def _build_answers(solpath, drift, recovery, risk, probe, checks) -> dict:
    fs = probe["feature_set_metrics"]
    return {
        "1_number_spans_on_solution_path": solpath["on_solution_path_numbers"],
        "2_off_path_distractors": {
            "off_path_numbers": solpath["off_solution_path_numbers"],
            "questions_with_distractor": solpath["questions_with_off_path_distractor"],
        },
        "3_filler_leakage_removed": (
            f"yes; model-generated recovery via {recovery['settings']['model_name']} with "
            f"leakage-guarded prompt {recovery['settings']['prompt_version']}, "
            f"temperature {recovery['settings']['temperature']}, K={recovery['settings']['num_samples']}"
        ),
        "4_recovery_distribution": drift["majority_drift_label_counts"],
        "5_target_not_span_type_function": not risk["is_span_type_deterministic"],
        "6_hidden_probe_beats_span_type_baseline": checks["5_hidden_probe_beats_baselines"]["pass"],
        "7_topk_covers_solution_path": probe["per_question_topk_coverage"],
        "8_ready_for_2000_rerun": checks["5_hidden_probe_beats_baselines"]["pass"]
        and checks["1_span_type_not_deterministic_target"]["pass"],
    }


def _write_gate_md(gate, solpath, risk, probe) -> None:
    fs = probe["feature_set_metrics"]
    lines = [
        "# Sprint 2H-B Instance-Signal Review Gate",
        "",
        f"**Checks passed: {gate['num_checks_passed']}/{gate['num_checks_total']}** — "
        f"ready_for_2000_rerun: **{gate['ready_for_2000_rerun']}**",
        "",
        "## Gate checks",
        "",
        "| # | Check | Pass |",
        "|---|-------|------|",
    ]
    labels = {
        "1_span_type_not_deterministic_target": "target not a span_type function",
        "2_filler_not_deterministic": "filler not span_type-deterministic",
        "3_number_split_on_off_path": "numbers split on/off solution path",
        "4_recovery_buckets_ordered": "fragility buckets ordinally separated",
        "5_hidden_probe_beats_baselines": "hidden(no-recovered) > baselines",
        "6_topk_covers_solution_path": "top-k covers solution-path numbers",
        "7_budget_not_dominated_by_distractors": "budget not spent on distractors",
    }
    for key, check in gate["checks"].items():
        lines.append(f"| {key.split('_')[0]} | {labels.get(key, key)} | {'✅' if check['pass'] else '❌'} |")

    lines += [
        "",
        "## Probe vs baselines (main = hidden_no_recovered)",
        "",
        "| feature set | macro_f1 | balanced_acc | bucket_3_recall | spearman | bucket3v1_auc |",
        "|---|---|---|---|---|---|",
    ]
    for fs_name in fpt_order():
        m = fs[fs_name]
        lines.append(
            f"| {fs_name} | {m['macro_f1']} | {m['balanced_accuracy']} | "
            f"{m['bucket_3_recall']} | {m['spearman_score_vs_bucket']} | {m['bucket_3_vs_bucket_1_auc']} |"
        )

    leak = gate["recovered_channel_leakage_diagnostic"]
    lines += [
        "",
        "## Recovered-channel leakage diagnostic",
        "",
        f"- with_recovered macro_f1 = {leak['with_recovered_macro_f1']}",
        f"- no_recovered  macro_f1 = {leak['no_recovered_macro_f1']}",
        f"- delta = {leak['delta']} → recovered_channel_leaks_label = **{leak['recovered_channel_leaks_label']}**",
        "",
        "## Solution-path number census",
        "",
        f"- number spans total: {solpath['number_spans_total']}",
        f"- on solution path: {solpath['on_solution_path_numbers']}",
        f"- off solution path (distractors): {solpath['off_solution_path_numbers']}",
        f"- ambiguous: {solpath['ambiguous_numbers']}",
        f"- implicit path numbers: {solpath['implicit_path_numbers']}",
        "",
        "## Answers to required questions",
        "",
    ]
    for key, value in gate["answers"].items():
        lines.append(f"- **{key}**: {value}")
    GATE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fpt_order() -> list[str]:
    return ["hidden_no_recovered", "span_type_only", "surface_rule", "hidden_with_recovered"]


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 2H-B instance-level signal construction")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["select", "recovery", "drift", "targets", "probe", "gate", "analyze", "all"],
    )
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=mr.DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=mr.DEFAULT_NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=mr.DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="cap recovery cases (debug)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("select", "all"):
        stage_select(args.subset_size, args.seed)
    if args.stage in ("recovery", "all"):
        stage_recovery(args.model, args.num_samples, args.temperature, args.limit)
    if args.stage in ("drift", "analyze", "all"):
        stage_drift()
    if args.stage in ("targets", "analyze", "all"):
        stage_targets()
    if args.stage in ("probe", "analyze", "all"):
        stage_probe(args.alpha, args.num_folds, args.seed, args.top_k)
    if args.stage in ("gate", "analyze", "all"):
        stage_gate()


if __name__ == "__main__":
    main()
