from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import sprint_4D_1_h1_emission_fabrication_smoke as d1  # noqa: E402
from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention import h1_data as hd  # noqa: E402
from recover_attention import h1_f5_features as f5  # noqa: E402
from recover_attention import mlp_readout_attribution as mra  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text  # noqa: E402
from recover_attention.h1_identifier import (  # noqa: E402
    assert_no_h1_gold_label_leakage,
    build_ontology_index,
    extract_identifiers,
    label_completion,
)
from recover_attention.schemas import validate_h1_sample_record  # noqa: E402


DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4D_2_h1_full_generation_and_f5_baseline"
PRIMARY_FAMILIES = {"attack", "cwe"}
FEATURE_GROUPS = {
    "mention_logprob": [
        "f5_id_logprob_mean",
        "f5_id_logprob_min",
        "f5_first_id_token_rank",
        "f5_id_token_entropy_mean",
    ],
    "sequence": [
        "f5_completion_perplexity",
        "f5_lengthnorm_logprob",
        "f5_id_token_ratio",
    ],
    "sampling": [
        "f5_ksample_exact_id_agreement",
        "f5_self_consistency_exact",
    ],
    "verbalized_confidence": [
        "f5_confidence_high",
        "f5_confidence_medium",
        "f5_confidence_low",
    ],
}
FEATURE_GROUPS["f5_full"] = sorted({name for names in FEATURE_GROUPS.values() for name in names})
SINGLE_FEATURE_RISK_SIGN = {
    "f5_id_logprob_mean": -1.0,
    "f5_id_logprob_min": -1.0,
    "f5_first_id_token_rank": 1.0,
    "f5_id_token_entropy_mean": 1.0,
    "f5_completion_perplexity": 1.0,
    "f5_lengthnorm_logprob": -1.0,
    "f5_id_token_ratio": 1.0,
    "f5_ksample_exact_id_agreement": -1.0,
    "f5_self_consistency_exact": -1.0,
    "f5_confidence_high": -1.0,
    "f5_confidence_medium": 0.0,
    "f5_confidence_low": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--ontology-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--density-report", type=Path, required=True)
    parser.add_argument("--samples-per-question", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_git(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return {
        "cmd": args,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_jsonl(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def trace_id(record: dict, sample_type: str, sample_index: int) -> str:
    return f"{record['example_id']}__{sample_type}_{sample_index}"


def trace_specs(samples_per_question: int) -> list[tuple[str, int, bool]]:
    return [("greedy", 0, False)] + [
        ("sampled", index + 1, True) for index in range(int(samples_per_question))
    ]


def cve_is_high_sequence(normalized_id: str, density_report: dict) -> bool:
    return d1.cve_is_high_sequence(normalized_id, density_report)


def preflight(
    args: argparse.Namespace,
    *,
    model_path: Path | None,
    model_source: str,
    fingerprint_report: dict,
) -> dict[str, Any]:
    git_status = run_git(["git", "status", "--porcelain"])
    tracked_check = run_git(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "scripts/sprint_4D_1_h1_emission_fabrication_smoke.py",
            "tests/test_h1_emission_smoke.py",
        ]
    )
    git_head = run_git(["git", "rev-parse", "--short", "HEAD"])
    required = {
        "samples_jsonl": args.samples_jsonl.exists(),
        "ontology_dir": args.ontology_dir.exists(),
        "snapshot_manifest": args.snapshot_manifest.exists(),
        "density_report": args.density_report.exists(),
        "4d1_emission_report": (
            PROJECT_ROOT
            / "outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/emission_fabrication_report.json"
        ).exists(),
        "model_path": model_path is not None and model_path.exists(),
    }
    stop_reasons = []
    if tracked_check["returncode"] != 0:
        stop_reasons.append("4D-1 generator/test files are not both tracked by git")
    if not all(required.values()):
        stop_reasons.append(f"missing required input(s): {[key for key, ok in required.items() if not ok]}")
    report = {
        "task": "Sprint 4D-2 H1 full generation + H1-F5 baseline",
        "git_head": git_head,
        "git_status_porcelain": git_status,
        "tracked_4d1_generator_assertion": tracked_check,
        "generate_completion_source": {
            "module": "scripts/sprint_4D_1_h1_emission_fabrication_smoke.py",
            "backend_required": "local_8bit_transformers_model_generate",
            "forbidden": ["4bit_long_form_generation", "fp16_full_backend", "manual_kv_cache_loop", "cpu_side_sampling_fallback"],
        },
        "required_inputs_exist": required,
        "ontology_fingerprints": fingerprint_report,
        "model_path": str(model_path) if model_path else None,
        "model_path_source": model_source,
        "boundary_flags": {
            "internal_features_touched": False,
            "intermediate_activation_captured": False,
            "probe_trained": "F5_linear_logistic_detection_only",
            "steering_continued": False,
            "hallucination_reduction_proven": False,
            "answer_accuracy_improvement_proven": False,
        },
        "stop_reasons": stop_reasons,
        "can_continue": not stop_reasons,
    }
    lines = ["# Sprint 4D-2 Preflight", ""]
    for key, value in report.items():
        lines.append(f"- {key}: `{json.dumps(value, ensure_ascii=False)}`")
    write_text("\n".join(lines) + "\n", args.output_dir / "preflight_report.md")
    if stop_reasons:
        raise SystemExit(f"preflight stop: {stop_reasons}")
    return report


def select_all_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda row: (row["split"], row["route"], row["family"], row["example_id"]))


def generate_full_manifest(
    context: dict[str, Any],
    selected: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    path = args.output_dir / "h1_generation_manifest.jsonl"
    existing_rows = load_existing_jsonl(path)
    existing = {row["trace_id"]: row for row in existing_rows}
    total = len(selected) * (1 + int(args.samples_per_question))
    print(f"[4D-2] generation resume state: {len(existing)}/{total} traces already present")

    done = len(existing)
    start_time = time.time()
    for question_index, record in enumerate(selected):
        prompt = d1.build_prompt(context, record)
        for sample_type, sample_index, do_sample in trace_specs(args.samples_per_question):
            tid = trace_id(record, sample_type, sample_index)
            if tid in existing:
                continue
            seed = d1.stable_seed(args.seed, record["example_id"], sample_type, sample_index)
            completion, num_tokens, touched = d1.generate_completion(
                context,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
            )
            row = d1.completion_row(
                record,
                prompt=prompt,
                completion=completion,
                num_generated_tokens=num_tokens,
                touched_max_new_tokens=touched,
                sample_index=sample_index,
                sample_type=sample_type,
                args=args,
            )
            row.update(
                {
                    "prompt": prompt,
                    "refusal": d1.is_refusal(completion),
                    "has_reasoning_text": d1.has_reasoning_text(completion),
                    "degeneration": dlp.detect_degeneration(
                        completion,
                        valid_labels=["A", "B"],
                        max_new_tokens=args.max_new_tokens,
                        num_generated_tokens=num_tokens,
                    ),
                    "generation_backend": "local_8bit_model_generate",
                }
            )
            row["degenerate"] = bool(row["degeneration"]["degenerate"])
            row["structured_list_false_positive"] = f5.looks_like_structured_identifier_list(completion)
            row["true_degeneration_estimate"] = bool(row["degenerate"] and not row["structured_list_false_positive"])
            assert_no_h1_gold_label_leakage(row)
            append_jsonl(path, row)
            existing[tid] = row
            done += 1
            elapsed = max(time.time() - start_time, 1e-9)
            rate = (done - len(existing_rows)) / elapsed
            remaining = (total - done) / rate if rate > 0 else None
            print(
                "[4D-2] generated "
                f"{done}/{total} q={question_index + 1}/{len(selected)} {tid} "
                f"tokens={num_tokens} eta_s={int(remaining) if remaining else 'NA'}"
            )
    return list(existing.values())


def build_mention_labels(
    traces: list[dict],
    index: Any,
    density_report: dict,
    output_dir: Path,
) -> tuple[list[dict], dict[str, dict]]:
    mention_rows: list[dict] = []
    labels_by_trace: dict[str, dict] = {}
    for trace in traces:
        labels = label_completion(trace["completion"], trace["prompt"], index)
        labels_by_trace[trace["trace_id"]] = labels
        mention_rows.extend(d1.mention_label_rows(trace, labels, density_report))
    write_jsonl(mention_rows, output_dir / "h1_mention_labels.jsonl")
    return mention_rows, labels_by_trace


def teacher_forced_f5(
    context: dict[str, Any],
    trace: dict,
    mention_token_positions: list[int],
) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    full_text = trace["prompt"] + trace["completion"]
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    device = msrs.infer_model_input_device(model, "auto", torch)
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    input_ids = encoded["input_ids"][0]
    prompt_len = len(trace["prompt"])
    completion_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if index > 0 and int(end) > prompt_len and int(end) > int(start)
    ]
    if not completion_positions:
        return {
            "offsets": offsets,
            "completion_positions": [],
            "token_logprobs": {},
            "token_entropies": {},
            "first_token_ranks": {},
            "completion_token_logprobs": [],
        }
    with torch.no_grad():
        output = model(**encoded, use_cache=False)
    pos_tensor = torch.tensor([pos - 1 for pos in completion_positions], dtype=torch.long, device=device)
    target_tensor = input_ids[torch.tensor(completion_positions, dtype=torch.long, device=device)]
    logits = output.logits[0].index_select(0, pos_tensor).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    row_index = torch.arange(log_probs.shape[0], device=device)
    selected = log_probs[row_index, target_tensor]
    probs = torch.exp(log_probs)
    entropies = -(probs * log_probs).sum(dim=-1)

    token_logprobs = {
        int(position): float(value)
        for position, value in zip(completion_positions, selected.detach().cpu().tolist())
    }
    token_entropies = {
        int(position): float(value)
        for position, value in zip(completion_positions, entropies.detach().cpu().tolist())
    }
    position_to_local = {position: local for local, position in enumerate(completion_positions)}
    first_token_ranks: dict[int, int] = {}
    for position in sorted(set(int(pos) for pos in mention_token_positions)):
        local = position_to_local.get(position)
        if local is None:
            continue
        actual_lp = log_probs[local, target_tensor[local]]
        first_token_ranks[int(position)] = int((log_probs[local] > actual_lp).sum().detach().cpu().item()) + 1
    return {
        "offsets": offsets,
        "completion_positions": completion_positions,
        "token_logprobs": token_logprobs,
        "token_entropies": token_entropies,
        "first_token_ranks": first_token_ranks,
        "completion_token_logprobs": [token_logprobs[position] for position in completion_positions],
    }


def sampling_context(traces: list[dict], labels_by_trace: dict[str, dict]) -> dict[str, dict[str, Any]]:
    per_example_trace_sets: dict[str, list[str]] = defaultdict(list)
    per_example_ids: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        if trace["sample_type"] != "sampled":
            continue
        ids = sorted(
            {
                row["normalized"]
                for row in labels_by_trace[trace["trace_id"]]["mentions"]
                if row["label"] != "echoed"
            }
        )
        joined = "|".join(ids)
        per_example_trace_sets[trace["example_id"]].append(joined)
        per_example_ids[trace["example_id"]].extend(ids)
    out: dict[str, dict[str, Any]] = {}
    for example_id in set(per_example_trace_sets) | set(per_example_ids):
        consistency = f5.exact_consistency(per_example_trace_sets.get(example_id, []))
        out[example_id] = {
            "trace_set_consistency": consistency["f5_self_consistency_exact"],
            "ids": per_example_ids.get(example_id, []),
            "num_sampled_traces": len(per_example_trace_sets.get(example_id, [])),
        }
    return out


def build_f5_features(
    context: dict[str, Any],
    traces: list[dict],
    mention_rows: list[dict],
    labels_by_trace: dict[str, dict],
    output_dir: Path,
) -> list[dict]:
    mentions_by_trace: dict[str, list[dict]] = defaultdict(list)
    for row in mention_rows:
        if row["label"] != "echoed":
            mentions_by_trace[row["trace_id"]].append(row)
    sample_ctx = sampling_context(traces, labels_by_trace)
    feature_rows: list[dict] = []
    total = len(traces)
    for index, trace in enumerate(sorted(traces, key=lambda row: row["trace_id"])):
        mentions = mentions_by_trace.get(trace["trace_id"], [])
        if not mentions:
            continue
        prompt_len = len(trace["prompt"])
        full_text_offsets_probe = trace["prompt"] + trace["completion"]
        # First pass tokenizes only to map mention spans. The teacher-forced
        # helper reuses the same tokenizer settings for the forward pass.
        offset_encoded = context["tokenizer"](
            full_text_offsets_probe,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = offset_encoded["offset_mapping"]
        mention_positions: dict[int, list[int]] = {}
        all_id_positions: list[int] = []
        for mention_index, mention in enumerate(mentions):
            positions = f5.token_indices_for_char_span(
                offsets,
                prompt_len + int(mention["start"]),
                prompt_len + int(mention["end"]),
            )
            mention_positions[mention_index] = positions
            all_id_positions.extend(positions[:1])
        tf = teacher_forced_f5(context, trace, all_id_positions)
        union_id_positions = sorted({pos for positions in mention_positions.values() for pos in positions})
        sequence = f5.sequence_logprob_features(
            tf["completion_token_logprobs"],
            id_token_count=len(union_id_positions),
        )
        confidence = f5.extract_verbalized_confidence(trace["completion"])
        example_sampling = sample_ctx.get(trace["example_id"], {})
        for mention_index, mention in enumerate(mentions):
            positions = mention_positions[mention_index]
            mention_features = f5.mention_logprob_features(
                tf["token_logprobs"],
                tf["token_entropies"],
                tf["first_token_ranks"],
                positions,
            )
            row = {
                "trace_id": trace["trace_id"],
                "example_id": trace["example_id"],
                "group_id": trace["group_id"],
                "split": trace["split"],
                "route": trace["route"],
                "family": trace["family"],
                "sample_type": trace["sample_type"],
                "sample_index": trace["sample_index"],
                "mention_index": mention["mention_index"],
                "mention_family": mention["mention_family"],
                "raw": mention["raw"],
                "normalized": mention["normalized"],
                "mention_start": mention["start"],
                "mention_end": mention["end"],
                "label": mention["label"],
                "fabricated_label": int(mention["label"] == "fabricated"),
                "primary_included": f5.included_in_primary_detection(mention),
                "cve_high_sequence": mention["cve_high_sequence"],
                "token_positions": positions,
                **mention_features,
                **sequence,
                "f5_ksample_exact_id_agreement": f5.id_agreement_rate(
                    mention["normalized"],
                    example_sampling.get("ids", []),
                ),
                "f5_self_consistency_exact": example_sampling.get("trace_set_consistency"),
                "f5_num_sampled_traces_for_question": example_sampling.get("num_sampled_traces", 0),
                **confidence,
            }
            assert_no_h1_gold_label_leakage(row)
            feature_rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"[4D-2] teacher-forced F5 {index + 1}/{total} traces")
    write_jsonl(feature_rows, output_dir / "h1_f5_feature_manifest.jsonl")
    return feature_rows


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def metric_ci(
    rows: list[dict],
    predictions: dict[int, float],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> list[float | None]:
    import numpy as np

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index in predictions:
            groups[str(row["group_id"])].append(index)
    group_ids = sorted(groups)
    if not group_ids:
        return [None, None]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        idx = [row_index for group in sampled_groups for row_index in groups[str(group)]]
        labels = [int(rows[i]["fabricated_label"]) for i in idx]
        scores = [float(predictions[i]) for i in idx]
        value = mra.auroc(labels, scores) if metric == "auroc" else mra.auprc(labels, scores)
        if value is not None:
            values.append(float(value))
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate_cv_family(
    rows: list[dict],
    feature_names: list[str],
    *,
    seeds: list[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    seed_runs = []
    for seed in seeds:
        result = mra.question_grouped_cv(
            rows,
            feature_names=feature_names,
            label_key="fabricated_label",
            question_key="group_id",
            k=5,
            seed=seed,
        )
        seed_runs.append(result)
    aurocs = [run.get("oof_auroc") for run in seed_runs if run.get("oof_auroc") is not None]
    auprcs = [run.get("oof_auprc") for run in seed_runs if run.get("oof_auprc") is not None]
    first_predictions = {
        int(item["row_index"]): float(item["risk_score"])
        for item in seed_runs[0].get("oof_predictions", [])
    } if seed_runs else {}
    return {
        "backend": "question_grouped_numpy_logistic_cv",
        "feature_names": feature_names,
        "num_examples": len(rows),
        "positive_rate": rate(sum(int(row["fabricated_label"]) for row in rows), len(rows)),
        "mean_oof_auroc": mra.mean(aurocs),
        "mean_oof_auprc": mra.mean(auprcs),
        "auroc_ci95": metric_ci(
            rows,
            first_predictions,
            metric="auroc",
            bootstrap_samples=bootstrap_samples,
            seed=93001,
        ),
        "auprc_ci95": metric_ci(
            rows,
            first_predictions,
            metric="auprc",
            bootstrap_samples=bootstrap_samples,
            seed=93002,
        ),
        "seed_runs": seed_runs,
    }


def single_feature_report(rows: list[dict]) -> dict[str, Any]:
    labels = [int(row["fabricated_label"]) for row in rows]
    report: dict[str, Any] = {}
    for name in FEATURE_GROUPS["f5_full"]:
        sign = SINGLE_FEATURE_RISK_SIGN.get(name, 1.0)
        if sign == 0.0:
            scores = [float(f5.finite_float(row.get(name)) or 0.0) for row in rows]
        else:
            scores = [sign * float(f5.finite_float(row.get(name)) or 0.0) for row in rows]
        report[name] = {
            "risk_orientation": sign,
            "auroc": mra.auroc(labels, scores),
            "auprc": mra.auprc(labels, scores),
        }
    return report


def stratum_report(rows: list[dict], seeds: list[int], bootstrap_samples: int) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for key, predicate in {
        "family:attack": lambda row: row["mention_family"] == "attack",
        "family:cwe": lambda row: row["mention_family"] == "cwe",
        "route:recall": lambda row: row["route"] == "recall",
        "route:open_gen": lambda row: row["route"] == "open_gen",
    }.items():
        subset = [row for row in rows if predicate(row)]
        labels = {int(row["fabricated_label"]) for row in subset}
        if len(subset) < 8 or len(labels) < 2:
            strata[key] = {
                "available": False,
                "num_examples": len(subset),
                "positive_count": sum(int(row["fabricated_label"]) for row in subset),
                "reason": "too_few_examples_or_single_class",
            }
        else:
            strata[key] = evaluate_cv_family(
                subset,
                FEATURE_GROUPS["f5_full"],
                seeds=seeds,
                bootstrap_samples=bootstrap_samples,
            )
    return strata


def calibration_report(rows: list[dict], f5_full: dict) -> dict[str, Any]:
    predictions = f5_full["seed_runs"][0].get("oof_predictions", []) if f5_full.get("seed_runs") else []
    by_index = {int(item["row_index"]): float(item["risk_score"]) for item in predictions}
    paired = [
        (by_index[index], int(row["fabricated_label"]), row)
        for index, row in enumerate(rows)
        if index in by_index
    ]
    scores = [score for score, _label, _row in paired]
    labels = [label for _score, label, _row in paired]
    ordered_low_risk = sorted(paired, key=lambda item: item[0])
    curve = []
    for coverage in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        count = max(1, round(len(ordered_low_risk) * coverage)) if ordered_low_risk else 0
        accepted = ordered_low_risk[:count]
        curve.append(
            {
                "coverage": coverage,
                "accepted_examples": count,
                "fabrication_through_rate": rate(sum(label for _score, label, _row in accepted), len(accepted)),
            }
        )
    aurc = None
    if curve:
        values = [item["fabrication_through_rate"] for item in curve if item["fabrication_through_rate"] is not None]
        aurc = sum(values) / len(values) if values else None
    return {
        "feature_names": f5_full.get("feature_names"),
        "ece": mra.calibration_buckets(scores, labels).get("ece") if scores else None,
        "calibration_buckets": mra.calibration_buckets(scores, labels) if scores else {"num_examples": 0, "buckets": [], "ece": None},
        "risk_coverage_curve": curve,
        "aurc_discrete_mean": aurc,
        "fixed_80pct_coverage_fabrication_through_rate": next(
            (item["fabrication_through_rate"] for item in curve if item["coverage"] == 0.8),
            None,
        ),
    }


def summarize_generation(traces: list[dict], labels_by_trace: dict[str, dict]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in ("attack", "cwe", "cve"):
        subset = [row for row in traces if row["family"] == family]
        by_family[family] = {
            "n": len(subset),
            "touched_max_new_tokens": {
                "num": sum(bool(row["touched_max_new_tokens"]) for row in subset),
                "den": len(subset),
                "rate": rate(sum(bool(row["touched_max_new_tokens"]) for row in subset), len(subset)),
            },
        }
    return {
        "num_traces": len(traces),
        "num_questions": len({row["example_id"] for row in traces}),
        "emission": {
            "num": sum(labels_by_trace[row["trace_id"]]["num_mentions"] > 0 for row in traces),
            "den": len(traces),
            "rate": rate(sum(labels_by_trace[row["trace_id"]]["num_mentions"] > 0 for row in traces), len(traces)),
        },
        "h1_positive": {
            "num": sum(labels_by_trace[row["trace_id"]]["h1_positive"] for row in traces),
            "den": len(traces),
            "rate": rate(sum(labels_by_trace[row["trace_id"]]["h1_positive"] for row in traces), len(traces)),
        },
        "raw_degeneration": {
            "num": sum(bool(row.get("degenerate")) for row in traces),
            "den": len(traces),
            "rate": rate(sum(bool(row.get("degenerate")) for row in traces), len(traces)),
        },
        "true_degeneration_estimate": {
            "num": sum(bool(row.get("true_degeneration_estimate")) for row in traces),
            "den": len(traces),
            "rate": rate(sum(bool(row.get("true_degeneration_estimate")) for row in traces), len(traces)),
        },
        "touched_by_family": by_family,
    }


def build_baseline_report(
    traces: list[dict],
    mention_rows: list[dict],
    feature_rows: list[dict],
    labels_by_trace: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = [row for row in feature_rows if row["primary_included"]]
    cve_rows = [row for row in feature_rows if row["mention_family"] == "cve"]
    report = {
        "task": "Sprint 4D-2 H1-F5 baseline",
        "backend": {
            "generation": "local_8bit_model_generate",
            "f5_logprob": "teacher_forcing_single_forward",
            "used_generation_output_scores": False,
            "used_4bit_long_form_generation": False,
            "used_manual_kv_cache_loop": False,
            "used_cpu_side_sampling_fallback": False,
            "internal_features_touched": False,
        },
        "generation_summary": summarize_generation(traces, labels_by_trace),
        "mention_summary": {
            "total_mentions": len(mention_rows),
            "feature_rows_non_echo": len(feature_rows),
            "echo_mentions_excluded_from_features": sum(row["label"] == "echoed" for row in mention_rows),
            "primary_examples": len(primary),
            "primary_positive_count": sum(int(row["fabricated_label"]) for row in primary),
            "primary_base_rate": rate(sum(int(row["fabricated_label"]) for row in primary), len(primary)),
            "cve_feature_rows": len(cve_rows),
            "cve_positive_count": sum(int(row["fabricated_label"]) for row in cve_rows),
            "cve_high_sequence_rows": sum(bool(row["cve_high_sequence"]) for row in cve_rows),
        },
        "single_feature": single_feature_report(primary) if primary else {},
        "families": {},
        "strata": {},
        "mcq_4c_reference": {"f5_only_cv_auroc": 0.8343},
        "boundary_flags": {
            "probe_trained": "F5_linear_logistic_detection_only",
            "intermediate_activation_captured": False,
            "f1_f4_touched": False,
            "steering_continued": False,
            "hallucination_reduction_proven": False,
            "answer_accuracy_improvement_proven": False,
        },
    }
    if len(primary) >= 8 and len({row["fabricated_label"] for row in primary}) == 2:
        for family_name, feature_names in FEATURE_GROUPS.items():
            report["families"][family_name] = evaluate_cv_family(
                primary,
                feature_names,
                seeds=args.cv_seeds,
                bootstrap_samples=args.bootstrap_samples,
            )
        report["strata"] = stratum_report(primary, args.cv_seeds, args.bootstrap_samples)
    else:
        report["families"]["f5_full"] = {
            "available": False,
            "reason": "too_few_primary_examples_or_single_class",
        }
    calibration = calibration_report(primary, report["families"].get("f5_full", {}))
    return report, calibration


def write_case_reports(feature_rows: list[dict], baseline_report: dict, output_dir: Path) -> None:
    f5_full = baseline_report.get("families", {}).get("f5_full", {})
    predictions = f5_full.get("seed_runs", [{}])[0].get("oof_predictions", []) if f5_full.get("seed_runs") else []
    by_index = {int(item["row_index"]): float(item["risk_score"]) for item in predictions}
    primary = [row for row in feature_rows if row["primary_included"]]
    paired = [
        {**row, "f5_full_oof_risk_score": by_index[index]}
        for index, row in enumerate(primary)
        if index in by_index and int(row["fabricated_label"]) == 1
    ]
    high = sorted(paired, key=lambda row: row["f5_full_oof_risk_score"], reverse=True)[:25]
    low = sorted(paired, key=lambda row: row["f5_full_oof_risk_score"])[:25]
    write_jsonl(high, output_dir / "high_risk_fabricated_case_report.jsonl")
    write_jsonl(low, output_dir / "low_risk_fabricated_case_report.jsonl")


def review_gate(report: dict, calibration: dict, elapsed: float) -> str:
    full = report.get("families", {}).get("f5_full", {})
    gen = report["generation_summary"]
    mention = report["mention_summary"]
    h1_auroc = full.get("mean_oof_auroc")
    delta = h1_auroc - 0.8343 if h1_auroc is not None else None
    recommendation = (
        "draft Sprint 4D-3 F1-F4 H1 increment check against this F5-full bar"
        if h1_auroc is not None and h1_auroc > 0.5 and full.get("mean_oof_auprc", 0) > (mention.get("primary_base_rate") or 0)
        else "treat H1-F5 as weak/uncertain; 4D-3 may still be useful, but use stricter CI and class-imbalance caution"
    )
    return f"""# Sprint 4D-2 Review Gate: H1 full generation + H1-F5 baseline

0. Backend: local 8-bit `model.generate` = yes; 4D-1 fixed path + sanity gate = yes; 4-bit/fp16/manual KV/CPU-side sampling = no; F5 logprob = teacher-forcing single forward, not generation `output_scores`.
1. Full generation: traces `{gen['num_traces']}`, questions `{gen['num_questions']}`, elapsed seconds `{elapsed:.1f}`, touched by family `{gen['touched_by_family']}`.
2. Primary mention set: ATT&CK/CWE non-echo n `{mention['primary_examples']}`, fabricated n `{mention['primary_positive_count']}`, base rate `{mention['primary_base_rate']}`.
3. Strongest single F5 feature: `{best_single_feature(report)}`.
4. H1-F5-full grouped-CV AUROC `{full.get('mean_oof_auroc')}` CI `{full.get('auroc_ci95')}`; AUPRC `{full.get('mean_oof_auprc')}` CI `{full.get('auprc_ci95')}`; base rate `{mention['primary_base_rate']}`.
5. 4C MCQ F5-only AUROC reference `0.8343`; H1 delta `{delta}`.
6. Stratified F5-full: `{report.get('strata')}`.
7. CVE separate: rows `{mention['cve_feature_rows']}`, fabricated `{mention['cve_positive_count']}`, high-sequence rows `{mention['cve_high_sequence_rows']}`; excluded from primary because CVE density makes existence labels weaker.
8. Calibration: ECE `{calibration.get('ece')}`, AURC discrete mean `{calibration.get('aurc_discrete_mean')}`, 80% coverage fabrication-through `{calibration.get('fixed_80pct_coverage_fabrication_through_rate')}`.
9. Degeneration: raw `{gen['raw_degeneration']}`, list-adjusted true estimate `{gen['true_degeneration_estimate']}`.
10. F1-F4 / intermediate activations / nonlinear model touched: no / no / no.
11. Gold id use: eval-only source id never enters prompt or F5 feature rows; leakage assertions passed.
12. Hallucination reduction / accuracy / intervention claim: no.
13. Next recommendation: {recommendation}.
"""


def best_single_feature(report: dict) -> dict[str, Any] | None:
    items = report.get("single_feature", {})
    if not items:
        return None
    name, value = max(items.items(), key=lambda item: item[1].get("auprc") or -1.0)
    return {"name": name, **value}


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    start = time.time()

    snapshot_manifest = read_json(args.snapshot_manifest)
    density_report = read_json(args.density_report)
    index = build_ontology_index(args.ontology_dir)
    fingerprint_report = d1.validate_snapshot_fingerprints(index, snapshot_manifest)
    model_path, model_source = d1.resolve_model_path(args.model_path)
    preflight(args, model_path=model_path, model_source=model_source, fingerprint_report=fingerprint_report)
    if model_path is None:
        raise SystemExit("model path unavailable")

    records = read_jsonl(args.samples_jsonl)
    for record in records:
        validate_h1_sample_record(record)
    selected = select_all_records(records)
    print(f"[4D-2] selected full H1 records: {len(selected)}")

    context = d1.load_generation_backend(model_path)
    print(f"[4D-2] model loaded from {model_path} (source={model_source}); backend=8-bit model.generate")
    d1._sanity_check_generation(context, selected, args)

    traces = generate_full_manifest(context, selected, args)
    expected = len(selected) * (1 + int(args.samples_per_question))
    if len(traces) != expected:
        raise SystemExit(f"generation incomplete: have {len(traces)}, expected {expected}; rerun will resume")
    traces = sorted(load_existing_jsonl(args.output_dir / "h1_generation_manifest.jsonl"), key=lambda row: row["trace_id"])
    mention_rows, labels_by_trace = build_mention_labels(traces, index, density_report, args.output_dir)
    feature_rows = build_f5_features(context, traces, mention_rows, labels_by_trace, args.output_dir)
    report, calibration = build_baseline_report(traces, mention_rows, feature_rows, labels_by_trace, args)
    write_json(report, args.output_dir / "h1_f5_baseline_report.json")
    write_json(calibration, args.output_dir / "h1_calibration_report.json")
    write_case_reports(feature_rows, report, args.output_dir)
    elapsed = time.time() - start
    write_text(review_gate(report, calibration, elapsed), args.output_dir / "review_gate_h1_full_generation_and_f5_baseline.md")
    print(
        "[4D-2] complete "
        f"traces={len(traces)} primary_n={report['mention_summary']['primary_examples']} "
        f"f5_auroc={report.get('families', {}).get('f5_full', {}).get('mean_oof_auroc')}"
    )


if __name__ == "__main__":
    main()
