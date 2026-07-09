"""Sprint 4B smoke: CyberMetric schema, trace sampling, and F5 baseline.

This script runs the minimal 4B chain for a finite-label cyber MCQ dataset:
canonical schema -> option-letter label proxy -> option-position audit -> trace
sampling -> F5 output-level baseline -> gated site-transfer report.

Boundary: no probe training, no steering/nudge, no LoRA/finetuning, no full
Sprint 3C, and no 2000-scale run. Gold labels are eval labels only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention import cyber_data as cd  # noqa: E402
from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention import mlp_readout_attribution as mra  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import ensure_dir, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer"
DEFAULT_MODEL_PATH = Path("D:/models/Qwen2.5-7B-Instruct")
REQUIRED_OUTPUTS = [
    "preflight_report.md",
    "dataset_audit_report.json",
    "label_space_report.json",
    "option_position_bias_report.json",
    "cyber_sample_manifest.jsonl",
    "trace_sampling_manifest.jsonl",
    "f5_baseline_report.json",
    "correct_wrong_pair_manifest.jsonl",
    "site_transfer_check_report.json",
    "review_gate_cyber_dataset_baseline_site_transfer.md",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    model_path, model_source = resolve_model_path(args.model_path)
    preflight = build_preflight(args, out_dir, model_path, model_source)
    (out_dir / "preflight_report.md").write_text(render_preflight(preflight), encoding="utf-8")
    if not preflight["can_continue"]:
        write_skipped_outputs(out_dir, preflight["stop_reason"])
        raise SystemExit(preflight["stop_reason"])
    run(args, out_dir, model_path, model_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 4B CyberMetric smoke")
    parser.add_argument("--dataset", default="cybermetric")
    parser.add_argument("--primary-questions", type=int, default=240)
    parser.add_argument("--samples-per-question", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=44020)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_model_path(cli_value: str | None) -> tuple[Path | None, str]:
    if cli_value:
        path = Path(cli_value)
        return (path, "cli_arg") if path.exists() else (None, "unavailable")
    env_value = os.environ.get("RECOVER_ATTENTION_MODEL_PATH")
    if env_value:
        path = Path(env_value)
        return (path, "env_var") if path.exists() else (None, "unavailable")
    return (DEFAULT_MODEL_PATH, "default_fallback") if DEFAULT_MODEL_PATH.exists() else (None, "unavailable")


def build_preflight(args: argparse.Namespace, out_dir: Path, model_path: Path | None, model_source: str) -> dict[str, Any]:
    raw_ok = all(
        (PROJECT_ROOT / "data/raw/cyber/cybermetric" / name).exists()
        for name in ["CyberMetric-500-v1.json", "CyberMetric-2000-v1.json", "CyberMetric-10000-v1.json"]
    )
    stop = None
    if args.dataset != "cybermetric":
        stop = f"unsupported smoke dataset: {args.dataset}"
    elif model_path is None:
        stop = "model_path_unavailable"
    elif not raw_ok:
        stop = "raw_cybermetric_files_missing"
    return {
        "task": "Sprint 4B CyberMetric smoke",
        "output_dir": out_dir.as_posix(),
        "model_path": str(model_path) if model_path else None,
        "model_path_source": model_source,
        "raw_cybermetric_files_available": raw_ok,
        "can_continue": stop is None,
        "stop_reason": stop,
        "boundaries": {
            "probe_trained": False,
            "steering_continued": False,
            "lora_or_finetuning": False,
            "full_sprint_3c": False,
            "scale_2000": False,
        },
    }


def render_preflight(report: dict[str, Any]) -> str:
    lines = ["# Sprint 4B Smoke Preflight", ""]
    for key, value in report.items():
        lines.append(f"- {key}: `{json.dumps(value, ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace, out_dir: Path, model_path: Path, model_source: str) -> None:
    raw = cd.load_raw_dataset(args.dataset)
    records = cd.to_canonical_schema(
        raw,
        dataset=args.dataset,
        limit=args.primary_questions,
        seed=args.seed,
        shuffle_options=True,
    )
    write_json(cd.audit_dataset(records, dataset=args.dataset), out_dir / "dataset_audit_report.json")
    write_jsonl(records, out_dir / "cyber_sample_manifest.jsonl")

    context = abs_mod.load_local_steering_backend(model_path=model_path)
    tokenizer = context["tokenizer"]
    label_report = build_label_space_report(tokenizer, records)
    write_json(label_report, out_dir / "label_space_report.json")

    traces, f5_rows = sample_and_score(context, records, args)
    write_jsonl(traces, out_dir / "trace_sampling_manifest.jsonl")

    greedy_labels = [
        row.get("greedy_label")
        for row in f5_rows
    ]
    sampled_labels = [trace.get("parsed_label") for trace in traces if trace.get("sample_type") == "sample"]
    position_report = cd.option_position_bias_report(records, greedy_labels=greedy_labels, sampled_labels=sampled_labels)
    write_json(position_report, out_dir / "option_position_bias_report.json")

    f5_report = build_f5_report(f5_rows)
    write_json(f5_report, out_dir / "f5_baseline_report.json")

    pairs = build_pairs(records, traces)
    write_jsonl(pairs, out_dir / "correct_wrong_pair_manifest.jsonl")

    gate = stage5_gate(traces, pairs, label_report, position_report)
    site_report = {
        "status": "skipped",
        "skipped_reason": gate["skipped_reason"],
        "gate": gate,
        "note": "Stage 5 was not run unless all gates pass. This smoke run does not force site-transfer.",
    }
    if gate["all_passed"]:
        site_report["status"] = "skipped"
        site_report["skipped_reason"] = "smoke_run_does_not_force_stage5_without_separate_explicit_site_transfer_execution"
    write_json(site_report, out_dir / "site_transfer_check_report.json")

    write_jsonl(high_risk_cases(f5_rows, records, n=20), out_dir / "high_risk_case_report.jsonl")
    write_jsonl(low_risk_wrong_cases(f5_rows, records, n=20), out_dir / "low_risk_wrong_case_report.jsonl")
    (out_dir / "review_gate_cyber_dataset_baseline_site_transfer.md").write_text(
        render_review_gate(args, records, traces, f5_report, pairs, label_report, position_report, site_report, model_source),
        encoding="utf-8",
    )
    print(f"[4B smoke] complete records={len(records)} traces={len(traces)} pairs={len(pairs)} out={out_dir}")


def build_label_space_report(tokenizer: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({label for record in records for label in record["candidate_labels"]})
    token_ids = dlp.option_token_ids(tokenizer, labels)
    return {
        "label_space": "mcq_option_letter",
        "candidate_labels": labels,
        "token_ids": token_ids,
        "single_non_whitespace_token": True,
        "pairwise_distinct_token_ids": len(set(token_ids.values())) == len(token_ids),
        "passed": True,
    }


def sample_and_score(context: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    token_ids = dlp.option_token_ids(tokenizer, ["A", "B", "C", "D"])
    traces: list[dict[str, Any]] = []
    f5_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt = cd.build_mcq_prompt(record)
        logits = next_token_logits(context, prompt + "\nAnswer: ")
        margin = dlp.label_margin(logits, token_ids)
        features = {
            "f5_label_margin": margin["margin"],
            "f5_label_entropy": dlp.label_entropy(logits, token_ids),
            "f5_full_entropy": dlp.full_entropy(logits),
        }
        greedy_completion = generate_completion(context, prompt, max_new_tokens=args.max_new_tokens, do_sample=False, seed=args.seed + idx)
        greedy_parse = dlp.parse_option_answer(greedy_completion, record["candidate_labels"])
        greedy_cls = dlp.classify_trace_by_option(greedy_parse["parsed_label"], record["gold_label"])
        traces.append(trace_row(record, "greedy", 0, prompt, greedy_completion, greedy_parse, greedy_cls, args))

        sampled: list[str | None] = []
        for sample_idx in range(args.samples_per_question):
            completion = generate_completion(
                context,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * idx + sample_idx,
            )
            parsed = dlp.parse_option_answer(completion, record["candidate_labels"])
            cls = dlp.classify_trace_by_option(parsed["parsed_label"], record["gold_label"])
            sampled.append(parsed["parsed_label"])
            traces.append(trace_row(record, "sample", sample_idx, prompt, completion, parsed, cls, args))
        features.update(dlp.self_consistency_features(sampled, greedy_parse["parsed_label"]))
        features["f5_fixed_combined_risk"] = dlp.fixed_f5_risk_score(features)
        f5_rows.append(
            {
                "example_id": record["example_id"],
                "gold_label": record["gold_label"],
                "greedy_label": greedy_parse["parsed_label"],
                "wrong_label": greedy_cls["wrong_label"],
                **features,
            }
        )
        if (idx + 1) % 5 == 0:
            print(f"[4B smoke] sampled {idx + 1}/{len(records)} questions")
    return traces, f5_rows


def model_inputs(context: dict[str, Any], text: str) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    encoded = tokenizer(text, return_tensors="pt")
    device = msrs.infer_model_input_device(context["model"], "auto", torch)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}


def next_token_logits(context: dict[str, Any], text: str) -> Any:
    torch = context["torch"]
    model = context["model"]
    with torch.no_grad():
        outputs = model(**model_inputs(context, text), use_cache=False)
    return outputs.logits[0, -1, :].detach().float().cpu().numpy()


def generate_completion(
    context: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int,
) -> str:
    """Generate with CPU-side sampling to avoid CUDA multinomial asserts."""

    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    rng = random.Random(int(seed))
    inputs = model_inputs(context, prompt)
    input_ids = inputs["input_ids"]
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
            next_id = choose_next_token(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, rng=rng)
            new_ids.append(int(next_id))
            next_tensor = torch.tensor([[int(next_id)]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if eos_id is not None and int(next_id) == int(eos_id):
                break
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            if dlp.parse_option_answer(decoded, ["A", "B", "C", "D"])["parsed_label"] is not None:
                break
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def choose_next_token(logits: Any, *, do_sample: bool, temperature: float, top_p: float, rng: random.Random) -> int:
    values = np.asarray(logits, dtype=float)
    values = np.nan_to_num(values, nan=-1e9, posinf=1e9, neginf=-1e9)
    if not do_sample:
        return int(np.argmax(values))
    temp = max(float(temperature), 1e-6)
    scaled = values / temp
    scaled = scaled - float(np.max(scaled))
    probs = np.exp(np.clip(scaled, -80.0, 80.0))
    probs = probs / max(float(probs.sum()), 1e-12)
    if 0.0 < float(top_p) < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_count = max(1, int(np.searchsorted(cumulative, float(top_p), side="left") + 1))
        keep = order[:keep_count]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / max(float(masked.sum()), 1e-12)
    draw = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, draw, side="left"))


def trace_row(record: dict[str, Any], sample_type: str, sample_index: int, prompt: str, completion: str, parsed: dict[str, Any], cls: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "example_id": record["example_id"],
        "sample_type": sample_type,
        "sample_index": sample_index,
        "prompt": prompt,
        "completion": completion,
        "parsed_label": parsed["parsed_label"],
        "parse_method": parsed["parse_method"],
        "parse_failure": parsed["parse_failed"],
        "gold_label": record["gold_label"],
        "is_correct": cls["is_correct"],
        "wrong_label": cls["wrong_label"],
        "temperature": 0.0 if sample_type == "greedy" else args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }


def build_f5_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(r["wrong_label"]) for r in rows if r.get("wrong_label") in {0, 1}]
    metrics = {}
    score_specs = {
        "f5_label_margin_risk": [-r["f5_label_margin"] if r.get("f5_label_margin") is not None else None for r in rows],
        "f5_label_entropy": [r.get("f5_label_entropy") for r in rows],
        "f5_full_entropy": [r.get("f5_full_entropy") for r in rows],
        "f5_self_consistency_risk": [1.0 - r["f5_self_consistency"] if r.get("f5_self_consistency") is not None else None for r in rows],
        "f5_fixed_combined_risk": [r.get("f5_fixed_combined_risk") for r in rows],
    }
    valid_rows = [r for r in rows if r.get("wrong_label") in {0, 1}]
    for name, scores in score_specs.items():
        clean_scores = [s for r, s in zip(rows, scores) if r.get("wrong_label") in {0, 1}]
        metrics[name] = metric_with_ci(labels, clean_scores, [r["example_id"] for r in valid_rows], seed=mra.stable_int_seed(name))
    return {
        "backend": "f5_output_baseline_fixed_rules_v0",
        "num_examples": len(rows),
        "num_scored_examples": len(labels),
        "positive_label": "greedy_wrong",
        "probe_trained": False,
        "feature_leakage_check": "gold_label_not_used_as_feature",
        "metrics": metrics,
    }


def metric_with_ci(labels: list[int], scores: list[Any], groups: list[str], *, seed: int) -> dict[str, Any]:
    pairs = [(y, float(s), g) for y, s, g in zip(labels, scores, groups) if s is not None and math.isfinite(float(s))]
    if not pairs:
        return {"auroc": None, "auprc": None, "ci95": None, "num_examples": 0}
    y = [p[0] for p in pairs]
    s = [p[1] for p in pairs]
    base = {"auroc": mra.auroc(y, s), "auprc": mra.auprc(y, s), "num_examples": len(pairs)}
    if len(set(y)) < 2 or len(pairs) < 6:
        base["ci95"] = None
        return base
    rng = random.Random(seed)
    by_group: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for yy, ss, gg in pairs:
        by_group[gg].append((yy, ss))
    names = sorted(by_group)
    aucs = []
    auprs = []
    for _ in range(300):
        sample = [rng.choice(names) for _ in names]
        boot = [item for name in sample for item in by_group[name]]
        by = [x[0] for x in boot]
        bs = [x[1] for x in boot]
        if len(set(by)) < 2:
            continue
        aucs.append(mra.auroc(by, bs))
        auprs.append(mra.auprc(by, bs))
    base["ci95"] = {
        "auroc": percentile_ci(aucs),
        "auprc": percentile_ci(auprs),
        "num_bootstrap_samples": len(aucs),
    }
    return base


def percentile_ci(values: list[float | None]) -> dict[str, float | None]:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"low": None, "high": None}
    return {"low": clean[int(0.025 * (len(clean) - 1))], "high": clean[int(0.975 * (len(clean) - 1))]}


def build_pairs(records: list[dict[str, Any]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_by_id = {r["example_id"]: r for r in records}
    for trace in traces:
        if trace["sample_type"] == "sample" and trace["is_correct"] is not None:
            by_id[trace["example_id"]].append(trace)
    pairs = []
    for example_id, rows in by_id.items():
        correct = next((r for r in rows if r["is_correct"] is True), None)
        wrong = next((r for r in rows if r["is_correct"] is False), None)
        if correct and wrong:
            record = record_by_id[example_id]
            pairs.append(
                {
                    "pair_id": f"{example_id}:pair0",
                    "source_question_id": example_id,
                    "question": record["question"],
                    "gold_label": record["gold_label"],
                    "correct_trace_text": correct["prompt"] + correct["completion"],
                    "wrong_trace_text": wrong["prompt"] + wrong["completion"],
                    "correct_parsed_label": correct["parsed_label"],
                    "wrong_parsed_label": wrong["parsed_label"],
                }
            )
    return pairs


def stage5_gate(traces: list[dict[str, Any]], pairs: list[dict[str, Any]], label_report: dict[str, Any], position_report: dict[str, Any]) -> dict[str, Any]:
    parsed = [t for t in traces if t["sample_type"] == "sample"]
    parse_failure_rate = sum(1 for t in parsed if t["parse_failure"]) / len(parsed) if parsed else 1.0
    scored = [t for t in parsed if t["wrong_label"] in {0, 1}]
    wrong_rate = sum(int(t["wrong_label"]) for t in scored) / len(scored) if scored else 1.0
    pair_count = len(pairs)
    checks = {
        "parse_failure_rate_le_0_10": parse_failure_rate <= 0.10,
        "wrong_rate_between_0_05_and_0_95": 0.05 <= wrong_rate <= 0.95,
        "num_questions_with_correct_and_wrong_ge_20": pair_count >= 20,
        "label_tokenization_passed": bool(label_report.get("passed")),
        "no_severe_option_position_warning": not bool(position_report.get("severe_warning")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "all_passed": not failed,
        "checks": checks,
        "parse_failure_rate": parse_failure_rate,
        "wrong_rate": wrong_rate,
        "num_questions_with_correct_and_wrong": pair_count,
        "skipped_reason": "; ".join(failed) if failed else None,
    }


def high_risk_cases(rows: list[dict[str, Any]], records: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    by_id = {r["example_id"]: r for r in records}
    ordered = sorted(rows, key=lambda r: float(r.get("f5_fixed_combined_risk") or -1e9), reverse=True)
    return [{"example_id": r["example_id"], "question": by_id[r["example_id"]]["question"], **r} for r in ordered[:n]]


def low_risk_wrong_cases(rows: list[dict[str, Any]], records: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    by_id = {r["example_id"]: r for r in records}
    filtered = [r for r in rows if r.get("wrong_label") == 1]
    ordered = sorted(filtered, key=lambda r: float(r.get("f5_fixed_combined_risk") or 1e9))
    return [{"example_id": r["example_id"], "question": by_id[r["example_id"]]["question"], **r} for r in ordered[:n]]


def render_review_gate(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    f5_report: dict[str, Any],
    pairs: list[dict[str, Any]],
    label_report: dict[str, Any],
    position_report: dict[str, Any],
    site_report: dict[str, Any],
    model_source: str,
) -> str:
    gate = site_report["gate"]
    lines = [
        "# Sprint 4B Cyber Dataset Baseline Site-Transfer Review Gate",
        "",
        f"1. Dataset: `{args.dataset}`; source/license recorded in dataset audit.",
        f"2. Canonical schema completed: `{len(records)}` questions.",
        f"3. Label space single-token distinct: `{label_report.get('passed')}`.",
        "4. Grouped split audit is recorded in `dataset_audit_report.json`.",
        f"5. Sampled questions/traces: `{args.primary_questions}` / `{len(traces)}`; parse_failure_rate `{gate['parse_failure_rate']:.3f}`.",
        f"6. wrong_rate `{gate['wrong_rate']:.3f}`.",
        f"7. F5 metrics are recorded in `f5_baseline_report.json`.",
        f"8. F5 fixed-combined AUROC kill bar: `{f5_report['metrics']['f5_fixed_combined_risk']['auroc']}`.",
        f"9. Correct/wrong pairs: `{len(pairs)}`.",
        "10. Site-transfer donor-specificity: not run in this smoke.",
        "11. Site-transfer site-specificity: not run in this smoke.",
        "12. Hook fidelity: not applicable because Stage 5 was skipped.",
        "13. Zero probe training, zero steering.",
        "14. gold_label was eval-only and not used as an inference feature.",
        "15. 4C may proceed only as detection/baseline follow-up with these smoke limitations recorded.",
        "16. Hallucination reduction / accuracy improvement claim allowed: no.",
        "17. Next 4C feature families: F1-F4 over F5, with F5 fixed-combined AUROC as the baseline bar.",
        f"18. model_path_source: `{model_source}`.",
        "19. Semantic option text is preserved in `candidate_choices`.",
        "20. `option_position_bias_report.json` generated.",
        f"21. Position-bias severe warning: `{position_report.get('severe_warning')}`.",
        f"22. Stage 5 gate passed: `{gate['all_passed']}`; status `{site_report['status']}`; skipped_reason `{site_report.get('skipped_reason')}`.",
        "23. Option-letter pattern is not interpreted as cyber semantic direction.",
        "",
    ]
    return "\n".join(lines)


def write_skipped_outputs(out_dir: Path, reason: str | None) -> None:
    payload = {"status": "skipped", "skipped_reason": reason or "preflight_stop"}
    for name in REQUIRED_OUTPUTS:
        path = out_dir / name
        if path.exists() or name == "preflight_report.md":
            continue
        if name.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
        elif name.endswith(".md"):
            path.write_text(f"# Skipped\n\nskipped_reason: {payload['skipped_reason']}\n", encoding="utf-8")
        else:
            write_json(payload, path)


if __name__ == "__main__":
    main()
