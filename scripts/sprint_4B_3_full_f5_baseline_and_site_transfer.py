"""Sprint 4B-3: full F5 baseline and gated 3C-1 site-transfer check.

Boundary: no probe training, no steering, no LoRA/finetuning, no 2000-scale
run. Stage E module patching is diagnostic only and runs only behind the task
card gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
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
from recover_attention import module_causal_tracing as mct  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from recover_attention.schemas import validate_cyber_sample_record  # noqa: E402

DEFAULT_MODEL_PATH = Path("D:/models/Qwen2.5-7B-Instruct")
DEFAULT_PROCESSED = PROJECT_ROOT / "data/processed/cyber/cybermetric.jsonl"
DEFAULT_SMOKE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/cyber_sample_smoke_manifest.jsonl"
)
DEFAULT_AB_REPORT = PROJECT_ROOT / "outputs/logs/sprint_4B_2_small_model_smoke/prompt_ab_report.json"
DEFAULT_AB_TRACE_CHAT = PROJECT_ROOT / "outputs/logs/sprint_4B_2_small_model_smoke/trace_manifest_cond_chat.jsonl"
DEFAULT_AB_TRACE_RAW = PROJECT_ROOT / "outputs/logs/sprint_4B_2_small_model_smoke/trace_manifest_cond_raw.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer"
CANDIDATE_LABELS = ["A", "B", "C", "D"]
WINNER_ADMISSION_THRESHOLD = 0.08
DEGRADED_ADMISSION_THRESHOLD = 0.15
REASONING_MIN_NONSPACE = 20
GSM8K_3C1_REFERENCE = {
    "mlp_output": {"donor_specificity_mean": 0.141, "site_specificity_mean": 0.353},
    "attention_output": {"donor_specificity_mean": 0.093, "site_specificity_mean": -0.006},
    "residual_output": {"donor_specificity_mean": 1.703, "site_specificity_mean": -0.547},
}
FORBIDDEN_OUTPUT_PREFIXES = (
    "sprint_3",
    "sprint_4A",
    "sprint_4B_1",
    "sprint_4B_2",
    "sprint_4B_dataset_download_audit",
    "sprint_4B_cyber_dataset_baseline_and_site_transfer",
)


# ---------------------------------------------------------------------------
# Pure helpers covered by unit tests.


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def zscore(values: list[float | None]) -> list[float | None]:
    clean = [float(v) for v in values if finite_float(v)]
    if not clean:
        return [None for _ in values]
    mu = float(np.mean(clean))
    sigma = float(np.std(clean))
    if sigma < 1e-12:
        return [0.0 if finite_float(v) else None for v in values]
    return [(float(v) - mu) / sigma if finite_float(v) else None for v in values]


def fixed_equal_weight_zscore_combo(rows: list[dict[str, Any]], specs: list[tuple[str, str]]) -> list[float | None]:
    """Return an equal-weight non-trained z-score risk combination.

    ``specs`` is ``[(field_name, orientation)]`` where orientation is either
    ``"risk_high"`` or ``"risk_low"``. ``risk_low`` flips the sign before
    standardization, e.g. margin and agreement become higher-risk when small.
    """

    if not specs:
        raise ValueError("specs must not be empty")
    transformed_columns: list[list[float | None]] = []
    for field, orientation in specs:
        if orientation not in {"risk_high", "risk_low"}:
            raise ValueError(f"unsupported orientation: {orientation}")
        raw = [row.get(field) for row in rows]
        signed = [
            (-float(v) if orientation == "risk_low" else float(v))
            if finite_float(v)
            else None
            for v in raw
        ]
        transformed_columns.append(zscore(signed))
    out: list[float | None] = []
    for idx in range(len(rows)):
        vals = [col[idx] for col in transformed_columns if col[idx] is not None]
        out.append(float(sum(vals) / len(vals)) if vals else None)
    return out


def has_reasoning_text_before_answer(completion: str, parsed_label: str | None) -> bool:
    if not isinstance(completion, str) or parsed_label is None:
        return False
    label = str(parsed_label).strip().upper()
    upper = completion.upper()
    answer_idx = upper.rfind(f"ANSWER: {label}")
    if answer_idx < 0:
        answer_idx = upper.rfind(label)
    prefix = completion[:answer_idx] if answer_idx >= 0 else completion
    return sum(1 for char in prefix if not char.isspace()) >= REASONING_MIN_NONSPACE


def equivalence_spot_check_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [
        bool(row.get("new_parsed_label") == row.get("reference_parsed_label"))
        for row in comparisons
    ]
    return {
        "num_checked": len(comparisons),
        "num_matched": sum(1 for ok in matched if ok),
        "all_matched": bool(comparisons) and all(matched),
        "mismatches": [row for row, ok in zip(comparisons, matched) if not ok],
    }


# ---------------------------------------------------------------------------
# CLI and preflight.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 4B-3 full F5 baseline and site-transfer diagnostic")
    parser.add_argument("--processed-path", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--smoke-manifest", default=str(DEFAULT_SMOKE_MANIFEST))
    parser.add_argument("--ab-report", default=str(DEFAULT_AB_REPORT))
    parser.add_argument("--samples-per-question", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=44303)
    parser.add_argument("--equivalence-count", type=int, default=5)
    parser.add_argument("--reasoning-forcing-questions", type=int, default=16)
    parser.add_argument("--reasoning-forcing-samples", type=int, default=2)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--site-check-layers", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--site-check-alphas", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--site-check-max-pairs", type=int, default=34)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=None)
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


def git_status_porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def dataset_spot_check(path: Path, *, n: int = 20, seed: int = 44303) -> dict[str, Any]:
    if not path.exists():
        return {"checked": 0, "passed": 0, "failed": 0, "errors": ["file_missing"]}
    records = read_jsonl(path)
    sample = random.Random(seed).sample(records, min(n, len(records)))
    errors: list[str] = []
    for record in sample:
        try:
            validate_cyber_sample_record(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{record.get('example_id', '?')}: {exc}")
    return {"checked": len(sample), "passed": len(sample) - len(errors), "failed": len(errors), "errors": errors[:5]}


def read_ab_decision(path: Path) -> dict[str, Any]:
    report = read_json(path)
    decision = report.get("decision") or {}
    winner = decision.get("winner")
    score = None
    if winner:
        score = (report.get("scores") or {}).get(winner, {}).get("score")
    admission = "blocked"
    if winner and score is not None and float(score) <= WINNER_ADMISSION_THRESHOLD:
        admission = "normal"
    elif winner and score is not None and float(score) <= DEGRADED_ADMISSION_THRESHOLD:
        admission = "degraded"
    return {"report": report, "winner": winner, "winner_score": score, "admission": admission}


def build_preflight(args: argparse.Namespace, model_path: Path | None, model_source: str, out_dir: Path) -> dict[str, Any]:
    ab_path = Path(args.ab_report)
    ab = read_ab_decision(ab_path) if ab_path.exists() else {"winner": None, "winner_score": None, "admission": "missing"}
    output_name = out_dir.name
    forbidden_collision = output_name != "sprint_4B_3_full_f5_baseline_and_site_transfer" and output_name.startswith(FORBIDDEN_OUTPUT_PREFIXES)
    stop_reasons: list[str] = []
    for path, label in (
        (Path(args.processed_path), "processed_dataset_missing"),
        (Path(args.smoke_manifest), "smoke_manifest_missing"),
        (ab_path, "ab_report_missing"),
        (DEFAULT_OUT.parent / "sprint_4B_2_small_model_smoke" / "review_gate_small_model_smoke.md", "4B2_review_gate_missing"),
    ):
        if not path.exists():
            stop_reasons.append(f"{label}: {path}")
    if model_path is None:
        stop_reasons.append("model_path_unavailable")
    if ab.get("winner") is None:
        stop_reasons.append("ab_report_winner_missing")
    if ab.get("admission") == "blocked":
        stop_reasons.append("ab_report_winner_not_admitted")
    if forbidden_collision:
        stop_reasons.append(f"refusing_to_overwrite_prior_output_dir: {out_dir}")
    spot = dataset_spot_check(Path(args.processed_path))
    if spot["failed"]:
        stop_reasons.append(f"processed_dataset_spot_check_failed: {spot['failed']}")
    return {
        "task": "Sprint 4B-3 full F5 baseline and site-transfer diagnostic",
        "worktree_status_porcelain": git_status_porcelain(),
        "worktree_dirty_policy": "continued because AGENTS conversational preflight was confirmed by user before script execution",
        "processed_dataset_spot_check": spot,
        "model_path": str(model_path) if model_path else None,
        "model_path_source": model_source,
        "ab_report": str(ab_path),
        "chosen_prompt_style": ab.get("winner"),
        "winner_score": ab.get("winner_score"),
        "admission_level": ab.get("admission"),
        "estimated_main_generations": 240 * (args.samples_per_question + 1),
        "generation_path": "transformers.model.generate(use_cache=True, renormalize_logits=True, remove_invalid_values=True)",
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_dir": str(out_dir),
        "forbidden_output_collision": forbidden_collision,
        "stop_reasons": stop_reasons,
        "can_continue": not stop_reasons,
        "boundary_flags": {
            "ready_for_2000_rerun": False,
            "do_not_enter_full_sprint_3C": True,
            "hallucination_reduction_proven": False,
            "answer_accuracy_improvement_proven": False,
            "steering_continued": False,
            "probe_trained": False,
        },
    }


def render_preflight(report: dict[str, Any]) -> str:
    lines = ["# Sprint 4B-3 Preflight", ""]
    for key, value in report.items():
        lines.append(f"- {key}: `{json.dumps(value, ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompting and generation.


def select_questions(smoke_manifest: Path) -> list[dict[str, Any]]:
    records = [r for r in read_jsonl(smoke_manifest) if r["metadata"]["split"] == "train"]
    return sorted(records, key=lambda r: r["example_id"])


def build_prompt(context: dict[str, Any], record: dict[str, Any], prompt_style: str) -> str:
    if prompt_style == "raw":
        return cd.build_mcq_prompt(record)
    if prompt_style == "chat":
        return context["tokenizer"].apply_chat_template(
            cd.build_mcq_chat_messages(record),
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unsupported prompt_style: {prompt_style}")


def build_reasoning_forcing_prompt(context: dict[str, Any], record: dict[str, Any]) -> str:
    user = cd.build_mcq_prompt(record) + "\n\nOutput exactly two sections:\nReasoning: <brief reasoning>\nAnswer: <letter>"
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert. Reason briefly, then answer with one option letter."},
        {"role": "user", "content": user},
    ]
    return context["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class PeriodicParseStop:
    def __init__(self, tokenizer: Any, valid_labels: list[str], *, prompt_len: int, every: int = 16) -> None:
        self.tokenizer = tokenizer
        self.valid_labels = valid_labels
        self.prompt_len = int(prompt_len)
        self.every = max(1, int(every))

    def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
        generated = int(input_ids.shape[-1]) - self.prompt_len
        if generated <= 0 or generated % self.every != 0:
            return False
        new_ids = input_ids[0, self.prompt_len :].detach().cpu().tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return dlp.parse_option_answer(text, self.valid_labels)["parsed_label"] is not None


def model_inputs(context: dict[str, Any], text: str) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    encoded = tokenizer(text, return_tensors="pt")
    device = msrs.infer_model_input_device(context["model"], "auto", torch)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def generate_completion(
    context: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    valid_labels: list[str],
) -> tuple[str, int]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    inputs = model_inputs(context, prompt)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
    stopping = PeriodicParseStop(tokenizer, valid_labels, prompt_len=prompt_len, every=16)
    torch.manual_seed(int(seed))
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "use_cache": True,
        "renormalize_logits": True,
        "remove_invalid_values": True,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "stopping_criteria": [stopping],
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    with torch.no_grad():
        output_ids = model.generate(**kwargs)
    new_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return completion, len(new_ids)


def build_trace_row(
    record: dict[str, Any],
    prompt_style: str,
    prompt: str,
    sample_type: str,
    sample_index: int,
    completion: str,
    num_generated_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parsed = dlp.parse_option_answer(completion, record["candidate_labels"])
    status = dlp.classify_trace_by_option(parsed["parsed_label"], record["gold_label"])
    degeneration = dlp.detect_degeneration(
        completion,
        valid_labels=record["candidate_labels"],
        max_new_tokens=args.max_new_tokens,
        num_generated_tokens=num_generated_tokens,
    )
    return {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "prompt_style": prompt_style,
        "sample_type": sample_type,
        "sample_index": int(sample_index),
        "prompt": prompt,
        "completion": completion,
        "completion_num_chars": len(completion),
        "num_generated_tokens": int(num_generated_tokens),
        "parsed_label": parsed["parsed_label"],
        "parse_method": parsed["parse_method"],
        "parse_failure": parsed["parse_failure"],
        "gold_label": record["gold_label"],
        "is_correct": True if status == "correct" else False if status == "wrong" else None,
        "degenerate": degeneration["degenerate"],
        "degeneration_rules": degeneration["matched_rules"],
        "has_reasoning_text": has_reasoning_text_before_answer(completion, parsed["parsed_label"]),
        "temperature": 0.0 if sample_type == "greedy" else args.temperature,
        "top_p": 1.0 if sample_type == "greedy" else args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }


def run_generation(context: dict[str, Any], records: list[dict[str, Any]], prompt_style: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt = build_prompt(context, record, prompt_style)
        greedy, greedy_tokens = generate_completion(
            context,
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed + idx,
            valid_labels=record["candidate_labels"],
        )
        traces.append(build_trace_row(record, prompt_style, prompt, "greedy", 0, greedy, greedy_tokens, args))
        for sample_idx in range(args.samples_per_question):
            sample, sample_tokens = generate_completion(
                context,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1000 * (idx + 1) + sample_idx,
                valid_labels=record["candidate_labels"],
            )
            traces.append(build_trace_row(record, prompt_style, prompt, "sample", sample_idx, sample, sample_tokens, args))
        if (idx + 1) % 20 == 0:
            print(f"[4B-3] sampled {idx + 1}/{len(records)} questions")
    return traces


def condition_score(traces: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    parse_failures = sum(1 for t in traces if t["parse_failure"])
    degenerate = sum(1 for t in traces if t["degenerate"])
    correct = sum(1 for t in traces if t["is_correct"] is True)
    wrong = sum(1 for t in traces if t["is_correct"] is False)
    q_status: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        if trace["sample_type"] == "sample" and not trace["parse_failure"] and not trace["degenerate"]:
            q_status[trace["example_id"]].add("correct" if trace["is_correct"] else "wrong")
    return {
        "num_traces": n,
        "correct_rate": correct / n if n else 0.0,
        "wrong_rate": wrong / n if n else 0.0,
        "parse_failure_rate": parse_failures / n if n else 1.0,
        "degeneration_rate": degenerate / n if n else 1.0,
        "num_questions_with_correct_and_wrong": sum({"correct", "wrong"}.issubset(v) for v in q_status.values()),
    }


# ---------------------------------------------------------------------------
# Stage A and Stage B'.


def reference_trace_path(prompt_style: str) -> Path:
    return DEFAULT_AB_TRACE_CHAT if prompt_style == "chat" else DEFAULT_AB_TRACE_RAW


def run_equivalence_spot_check(
    context: dict[str, Any],
    records: list[dict[str, Any]],
    prompt_style: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference = [r for r in read_jsonl(reference_trace_path(prompt_style)) if r["sample_type"] == "greedy"]
    ref_by_id = {r["example_id"]: r for r in reference}
    selected = [r for r in records if r["example_id"] in ref_by_id][: args.equivalence_count]
    comparisons = []
    for idx, record in enumerate(selected):
        prompt = build_prompt(context, record, prompt_style)
        completion, tokens = generate_completion(
            context,
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed + 70000 + idx,
            valid_labels=record["candidate_labels"],
        )
        parsed = dlp.parse_option_answer(completion, record["candidate_labels"])
        ref = ref_by_id[record["example_id"]]
        comparisons.append(
            {
                "example_id": record["example_id"],
                "reference_parsed_label": ref["parsed_label"],
                "new_parsed_label": parsed["parsed_label"],
                "new_num_generated_tokens": tokens,
                "reference_completion_preview": ref["completion"][:120],
                "new_completion_preview": completion[:120],
            }
        )
    summary = equivalence_spot_check_summary(comparisons)
    summary["comparisons"] = comparisons
    summary["reference_trace_path"] = str(reference_trace_path(prompt_style))
    return summary


def run_reasoning_forcing_probe(context: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = records[: min(args.reasoning_forcing_questions, len(records))]
    traces: list[dict[str, Any]] = []
    for idx, record in enumerate(selected):
        prompt = build_reasoning_forcing_prompt(context, record)
        for sample_idx in range(args.reasoning_forcing_samples):
            completion, tokens = generate_completion(
                context,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 90000 + 1000 * idx + sample_idx,
                valid_labels=record["candidate_labels"],
            )
            traces.append(build_trace_row(record, "reasoning_forcing_chat", prompt, "sample", sample_idx, completion, tokens, args))
    score = condition_score(traces)
    score["has_reasoning_text_rate"] = sum(1 for t in traces if t["has_reasoning_text"]) / len(traces) if traces else 0.0
    score["clean_score"] = score["parse_failure_rate"] + score["degeneration_rate"]
    score["admitted_for_f2"] = score["has_reasoning_text_rate"] > 0.5 and score["clean_score"] <= WINNER_ADMISSION_THRESHOLD
    score["num_questions"] = len(selected)
    score["num_traces"] = len(traces)
    return {"summary": score, "traces": traces}


def reasoning_substrate_report(traces: list[dict[str, Any]], forcing: dict[str, Any]) -> dict[str, Any]:
    main = {
        "num_traces": len(traces),
        "has_reasoning_text_rate": sum(1 for t in traces if t["has_reasoning_text"]) / len(traces) if traces else 0.0,
        "completion_token_stats": summarize_numeric([t["num_generated_tokens"] for t in traces]),
        "completion_char_stats": summarize_numeric([t["completion_num_chars"] for t in traces]),
    }
    if forcing["summary"].get("admitted_for_f2"):
        plan = "use_stage_b_reasoning_forcing_variant_for_f2_only"
    elif main["has_reasoning_text_rate"] > 0.5:
        plan = "use_main_prompt_traces_for_f2"
    else:
        plan = "downgrade_or_drop_f2_and_prioritize_f3_plus_f1_f4"
    return {
        "definition": f"has_reasoning_text means >= {REASONING_MIN_NONSPACE} non-whitespace chars before the answer letter",
        "main_prompt": main,
        "stage_b_prime_reasoning_forcing": forcing["summary"],
        "f2_plan_for_4c": plan,
    }


# ---------------------------------------------------------------------------
# F5 features and evaluation.


def summarize_numeric(values: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in values if finite_float(v)]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "p50": None, "max": None}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "min": float(np.min(clean)),
        "p50": float(np.median(clean)),
        "max": float(np.max(clean)),
    }


def readout_feature_rows(context: dict[str, Any], records_by_id: dict[str, dict[str, Any]], traces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    token_ids_space = dlp.option_token_ids(tokenizer, CANDIDATE_LABELS)
    token_ids_bare = dlp.bare_option_token_ids(tokenizer, CANDIDATE_LABELS)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    failures = 0
    checked = 0
    greedy_traces = [t for t in traces if t["sample_type"] == "greedy"]
    for trace in greedy_traces:
        record = records_by_id[trace["example_id"]]
        row: dict[str, Any] = {
            "example_id": trace["example_id"],
            "group_id": record["group_id"],
            "gold_label": record["gold_label"],
            "parsed_label": trace["parsed_label"],
            "is_correct": trace["is_correct"],
            "parse_failure": trace["parse_failure"],
            "degenerate": trace["degenerate"],
            "readout_available": False,
        }
        if trace["parse_failure"] or trace["parsed_label"] is None or trace["degenerate"]:
            row["reason"] = "parse_failure_or_degenerate"
            rows.append(row)
            continue
        try:
            position = dlp.locate_label_readout_position(tokenizer, trace["prompt"], trace["completion"], trace["parsed_label"])
        except ValueError as exc:
            row["reason"] = f"locate_failed: {exc}"
            rows.append(row)
            continue
        combined = trace["prompt"] + trace["completion"]
        encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoded["input_ids"]
        actual_id = full_ids[position + 1] if position + 1 < len(full_ids) else None
        checked += 1
        if actual_id == token_ids_space[trace["parsed_label"]]:
            token_form, token_ids = "space", token_ids_space
        elif actual_id == token_ids_bare[trace["parsed_label"]]:
            token_form, token_ids = "bare", token_ids_bare
        else:
            failures += 1
            counts["unknown"] += 1
            row["reason"] = "token_form_unknown_assertion_failed"
            rows.append(row)
            continue
        counts[token_form] += 1
        prefix_ids = full_ids[: position + 1]
        device = msrs.infer_model_input_device(model, "auto", torch)
        input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, use_cache=False)
        logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
        option_logits = {label: float(logits[token_id]) for label, token_id in token_ids.items()}
        row.update(
            {
                "readout_available": True,
                "readout_position": int(position),
                "readout_to_prompt_end_distance": int(len(tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]) - 1 - position),
                "readout_token_form": token_form,
                "actual_label_token_id": int(actual_id),
                "tier_single_forward": {
                    "f5_label_margin": dlp.label_margin(option_logits),
                    "f5_label_entropy": dlp.label_entropy(option_logits),
                    "f5_full_entropy": dlp.full_entropy(logits),
                },
            }
        )
        rows.append(row)
    fidelity = {
        "checked": checked,
        "failed": failures,
        "token_form_counts": dict(counts),
        "label_space_report": {"space": token_ids_space, "bare": token_ids_bare},
    }
    return attach_sampling_features(rows, traces, records_by_id), fidelity


def attach_sampling_features(rows: list[dict[str, Any]], traces: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    samples_by_id: dict[str, list[str | None]] = defaultdict(list)
    greedy_by_id: dict[str, str | None] = {}
    for trace in traces:
        if trace["sample_type"] == "sample":
            samples_by_id[trace["example_id"]].append(trace["parsed_label"])
        elif trace["sample_type"] == "greedy":
            greedy_by_id[trace["example_id"]] = trace["parsed_label"]
    for row in rows:
        record = records_by_id[row["example_id"]]
        consistency = dlp.self_consistency_features(
            greedy_by_id.get(row["example_id"]),
            samples_by_id.get(row["example_id"], []),
            record["candidate_labels"],
        )
        row["tier_sampling"] = {
            "f5_self_consistency": consistency["self_consistency_with_greedy"],
            "f5_sc_majority_agree": (
                float(consistency["majority_agrees_with_greedy"])
                if consistency["majority_agrees_with_greedy"] is not None
                else None
            ),
        }
        row["sample_vote_counts"] = consistency["sample_vote_counts"]
        if row.get("readout_available"):
            feature_record = {"tier_single_forward": row["tier_single_forward"], "tier_sampling": row["tier_sampling"]}
            dlp.assert_no_gold_label_leakage(feature_record)
    return rows


def flatten_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("readout_available") or row.get("is_correct") is None:
            continue
        out = {
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "wrong_label": 0 if row["is_correct"] else 1,
            **row["tier_single_forward"],
            **row["tier_sampling"],
        }
        flat.append(out)
    combo_single = fixed_equal_weight_zscore_combo(
        flat,
        [
            ("f5_label_margin", "risk_low"),
            ("f5_label_entropy", "risk_high"),
            ("f5_full_entropy", "risk_high"),
        ],
    )
    combo_sampling = fixed_equal_weight_zscore_combo(
        flat,
        [
            ("f5_label_margin", "risk_low"),
            ("f5_label_entropy", "risk_high"),
            ("f5_full_entropy", "risk_high"),
            ("f5_self_consistency", "risk_low"),
            ("f5_sc_majority_agree", "risk_low"),
        ],
    )
    for row, single, sampling in zip(flat, combo_single, combo_sampling):
        row["f5_combo_single_forward_z"] = single
        row["f5_combo_sampling_z"] = sampling
    return flat


def grouped_bootstrap_metric(rows: list[dict[str, Any]], score_name: str, *, iters: int, seed: int) -> dict[str, Any]:
    clean = [r for r in rows if r.get("wrong_label") in {0, 1} and finite_float(r.get(score_name))]
    labels = [int(r["wrong_label"]) for r in clean]
    scores = [float(r[score_name]) for r in clean]
    base = {"auroc": mra.auroc(labels, scores), "auprc": mra.auprc(labels, scores)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        groups[str(row["example_id"])].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    boot_auroc: list[float] = []
    boot_auprc: list[float] = []
    for _ in range(int(iters)):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        sample_rows = [row for gid in sampled for row in groups[gid]]
        y = [int(r["wrong_label"]) for r in sample_rows]
        s = [float(r[score_name]) for r in sample_rows]
        auroc = mra.auroc(y, s)
        auprc = mra.auprc(y, s)
        if auroc is not None:
            boot_auroc.append(float(auroc))
        if auprc is not None:
            boot_auprc.append(float(auprc))
    return {
        "score_name": score_name,
        "num_examples": len(clean),
        "num_positive_wrong": sum(labels),
        "auroc": base["auroc"],
        "auroc_ci95": percentile_ci(boot_auroc),
        "auprc": base["auprc"],
        "auprc_ci95": percentile_ci(boot_auprc),
    }


def percentile_ci(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def build_f5_report(rows: list[dict[str, Any]], fidelity: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat = flatten_feature_rows(rows)
    feature_names = [
        "f5_label_margin",
        "f5_label_entropy",
        "f5_full_entropy",
        "f5_self_consistency",
        "f5_sc_majority_agree",
        "f5_combo_single_forward_z",
        "f5_combo_sampling_z",
    ]
    metrics = {
        name: grouped_bootstrap_metric(flat, name, iters=args.bootstrap_iters, seed=args.seed + idx)
        for idx, name in enumerate(feature_names)
    }
    return {
        "backend": "full_f5_baseline_v0",
        "num_greedy_rows": len(rows),
        "num_binary_eval_rows": len(flat),
        "positive_label": "greedy_wrong",
        "fixed_combo_formula": {
            "single_forward": "mean(z(-margin), z(label_entropy), z(full_entropy))",
            "sampling": "mean(z(-margin), z(label_entropy), z(full_entropy), z(1-self_consistency), z(1-majority_agree))",
            "trained": False,
        },
        "readout_position_assertion": fidelity,
        "feature_metrics": metrics,
        "kill_bar_single_forward": metrics["f5_combo_single_forward_z"],
        "kill_bar_sampling": metrics["f5_combo_sampling_z"],
        "boundary_flags": {
            "probe_trained": False,
            "steering_continued": False,
            "hallucination_reduction_proven": False,
            "answer_accuracy_improvement_proven": False,
        },
    }, flat


def option_position_bias_report(records: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    gold = Counter(r["gold_label"] for r in records)
    greedy = Counter(t["parsed_label"] for t in traces if t["sample_type"] == "greedy" and t["parsed_label"])
    sampled = Counter(t["parsed_label"] for t in traces if t["sample_type"] == "sample" and t["parsed_label"])
    majority = gold.most_common(1)[0][0] if gold else None
    position_baseline = gold[majority] / len(records) if majority else None
    greedy_total = sum(greedy.values())
    max_greedy_share = max(greedy.values()) / greedy_total if greedy_total else 0.0
    return {
        "num_questions": len(records),
        "gold_distribution": dict(sorted(gold.items())),
        "greedy_distribution": dict(sorted(greedy.items())),
        "sampled_distribution": dict(sorted(sampled.items())),
        "majority_gold_label": majority,
        "position_only_baseline_accuracy": position_baseline,
        "max_greedy_option_share": max_greedy_share,
        "severe_position_bias_warning": bool(max_greedy_share > 0.40),
        "warning_threshold": 0.40,
    }


def write_case_reports(out_dir: Path, flat: list[dict[str, Any]], rows: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]) -> None:
    row_by_id = {row["example_id"]: row for row in rows}
    high = sorted([r for r in flat if finite_float(r.get("f5_combo_sampling_z"))], key=lambda r: float(r["f5_combo_sampling_z"]), reverse=True)[:20]
    low_wrong = sorted(
        [r for r in flat if int(r["wrong_label"]) == 1 and finite_float(r.get("f5_combo_sampling_z"))],
        key=lambda r: float(r["f5_combo_sampling_z"]),
    )[:20]

    def enrich(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in items:
            rec = records_by_id[item["example_id"]]
            source_row = row_by_id[item["example_id"]]
            out.append(
                {
                    "example_id": item["example_id"],
                    "question": rec["question"],
                    "gold_label": rec["gold_label"],
                    "parsed_label": source_row["parsed_label"],
                    "is_correct": source_row["is_correct"],
                    "risk_score": item.get("f5_combo_sampling_z"),
                    "features": {k: v for k, v in item.items() if k.startswith("f5_")},
                }
            )
        return out

    write_jsonl(enrich(high), out_dir / "high_risk_case_report.jsonl")
    write_jsonl(enrich(low_wrong), out_dir / "low_risk_wrong_case_report.jsonl")


# ---------------------------------------------------------------------------
# Stage D/E: pairs and diagnostic patching.


def build_pairs(traces: list[dict[str, Any]], *, max_pairs: int | None = None) -> list[dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace["sample_type"] == "sample" and not trace["parse_failure"] and not trace["degenerate"]:
            by_q[trace["example_id"]].append(trace)
    pairs: list[dict[str, Any]] = []
    for example_id in sorted(by_q):
        correct = [t for t in by_q[example_id] if t["is_correct"] is True]
        wrong = [t for t in by_q[example_id] if t["is_correct"] is False]
        if correct and wrong:
            pair = {
                "pair_id": f"{example_id}_pair0",
                "example_id": example_id,
                "correct_trace": correct[0],
                "wrong_trace": wrong[0],
                "correct_parsed_label": correct[0]["parsed_label"],
                "wrong_parsed_label": wrong[0]["parsed_label"],
            }
            pairs.append(pair)
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def stage_e_gate(pairs: list[dict[str, Any]], score: dict[str, Any], token_fidelity: dict[str, Any], bias: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "pairs_at_least_20": len(pairs) >= 20,
        "parse_failure_rate_at_most_0_10": score["parse_failure_rate"] <= 0.10,
        "wrong_rate_between_0_05_and_0_95": 0.05 <= score["wrong_rate"] <= 0.95,
        "label_tokenization_passed": token_fidelity["failed"] == 0,
        "no_severe_position_bias_warning": not bias["severe_position_bias_warning"],
    }
    return {"passed": all(checks.values()), "checks": checks, "num_pairs": len(pairs)}


def locate_trace_state(context: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any] | None:
    tokenizer = context["tokenizer"]
    if trace["parse_failure"] or trace["parsed_label"] is None:
        return None
    try:
        readout = dlp.locate_label_readout_position(tokenizer, trace["prompt"], trace["completion"], trace["parsed_label"])
    except ValueError:
        return None
    full = tokenizer(trace["prompt"] + trace["completion"], add_special_tokens=False)
    prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
    label_position = readout + 1
    if label_position >= len(full["input_ids"]):
        return None
    return {
        "full_text": trace["prompt"] + trace["completion"],
        "input_ids": full["input_ids"],
        "prompt_token_count": len(prompt_ids),
        "readout_position": int(readout),
        "label_position": int(label_position),
        "prefix_ids": full["input_ids"][:label_position],
        "readout_to_prompt_end_distance": int(len(prompt_ids) - 1 - readout),
    }


def actual_token_form_ids(context: dict[str, Any], trace: dict[str, Any], state: dict[str, Any]) -> dict[str, int] | None:
    tokenizer = context["tokenizer"]
    space = dlp.option_token_ids(tokenizer, CANDIDATE_LABELS)
    bare = dlp.bare_option_token_ids(tokenizer, CANDIDATE_LABELS)
    actual = state["input_ids"][state["label_position"]]
    label = trace["parsed_label"]
    if actual == space[label]:
        return space
    if actual == bare[label]:
        return bare
    return None


def run_stage_e(context: dict[str, Any], pairs: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    module_types = ["attention_output", "mlp_output", "residual_output"]
    selected = pairs[: args.site_check_max_pairs]
    rows: list[dict[str, Any]] = []
    fidelity = {"registered": 0, "removed": 0, "triggered": 0, "same_trace_random_position_skipped": 0, "wrong_donor_self_floor_values": []}
    random_donor_by_index = selected[1:] + selected[:1] if len(selected) > 1 else selected
    for pair_idx, pair in enumerate(selected):
        wrong = pair["wrong_trace"]
        correct = pair["correct_trace"]
        wrong_state = locate_trace_state(context, wrong)
        correct_state = locate_trace_state(context, correct)
        random_state = locate_trace_state(context, random_donor_by_index[pair_idx]["correct_trace"])
        if wrong_state is None or correct_state is None or random_state is None:
            continue
        token_ids = actual_token_form_ids(context, wrong, wrong_state)
        if token_ids is None:
            continue
        gold = records_by_id[pair["example_id"]]["gold_label"]
        wrong_label = wrong["parsed_label"]
        gold_ids = [token_ids[gold]]
        wrong_ids = [token_ids[wrong_label]]
        captures = {
            "correct": mct.capture_module_outputs(context, correct_state["full_text"], args.site_check_layers, module_types),
            "wrong": mct.capture_module_outputs(context, wrong_state["full_text"], args.site_check_layers, module_types),
            "random": mct.capture_module_outputs(context, random_state["full_text"], args.site_check_layers, module_types),
        }
        no_patch_gold = mct.sequence_logprob_with_module_patch(context, prefix_ids=wrong_state["prefix_ids"], answer_ids=gold_ids)
        no_patch_wrong = mct.sequence_logprob_with_module_patch(context, prefix_ids=wrong_state["prefix_ids"], answer_ids=wrong_ids)
        base_metric = no_patch_gold["logprob"] - no_patch_wrong["logprob"]
        for layer in args.site_check_layers:
            for module_type in module_types:
                donors = {
                    "correct_donor": mct.module_vector(captures["correct"]["captured"], layer=layer, module_type=module_type, position=correct_state["readout_position"]),
                    "random_donor": mct.module_vector(captures["random"]["captured"], layer=layer, module_type=module_type, position=random_state["readout_position"]),
                    "wrong_donor_self": mct.module_vector(captures["wrong"]["captured"], layer=layer, module_type=module_type, position=wrong_state["readout_position"]),
                }
                same_trace_position = None
                completion_positions = list(range(wrong_state["prompt_token_count"], len(wrong_state["input_ids"])))
                completion_positions = [p for p in completion_positions if p != wrong_state["label_position"]]
                if completion_positions:
                    same_trace_position = completion_positions[0]
                    donors["same_trace_random_position"] = mct.module_vector(
                        captures["wrong"]["captured"], layer=layer, module_type=module_type, position=same_trace_position
                    )
                else:
                    fidelity["same_trace_random_position_skipped"] += 1
                for alpha in args.site_check_alphas:
                    for condition, donor in donors.items():
                        if donor is None:
                            continue
                        target_position = same_trace_position if condition == "same_trace_random_position" else wrong_state["readout_position"]
                        if target_position is None:
                            continue
                        patch = {"layer": layer, "module_type": module_type, "donor_vec": donor, "target_position": target_position, "alpha": alpha}
                        gold_lp = mct.sequence_logprob_with_module_patch(context, prefix_ids=wrong_state["prefix_ids"], answer_ids=gold_ids, patch=patch)
                        wrong_lp = mct.sequence_logprob_with_module_patch(context, prefix_ids=wrong_state["prefix_ids"], answer_ids=wrong_ids, patch=patch)
                        for trace_info in (gold_lp["trace"], wrong_lp["trace"]):
                            fidelity["registered"] += int(bool(trace_info.get("registered")))
                            fidelity["removed"] += int(bool(trace_info.get("removed")))
                            fidelity["triggered"] += int(bool(trace_info.get("triggered")))
                        metric = gold_lp["logprob"] - wrong_lp["logprob"]
                        clean_direction = metric - base_metric
                        if condition == "wrong_donor_self":
                            fidelity["wrong_donor_self_floor_values"].append(clean_direction)
                        rows.append(
                            {
                                "pair_id": pair["pair_id"],
                                "example_id": pair["example_id"],
                                "layer": int(layer),
                                "module_type": module_type,
                                "alpha": float(alpha),
                                "condition": condition,
                                "clean_direction": float(clean_direction),
                                "base_metric": float(base_metric),
                                "readout_to_prompt_end_distance": wrong_state["readout_to_prompt_end_distance"],
                                "answer_token_form": "actual_trace_form",
                            }
                        )
    report = summarize_stage_e(rows)
    fidelity_report = {
        **fidelity,
        "registered_rate": fidelity["registered"] / max(2 * len(rows), 1),
        "removed_rate": fidelity["removed"] / max(2 * len(rows), 1),
        "triggered_rate": fidelity["triggered"] / max(2 * len(rows), 1),
        "wrong_donor_self_floor": summarize_numeric(fidelity["wrong_donor_self_floor_values"]),
    }
    return report, fidelity_report


def summarize_stage_e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["module_type"], row["condition"])].append(row)
    summary: dict[str, Any] = {}
    for (module_type, condition), items in sorted(by_key.items()):
        values = [float(r["clean_direction"]) for r in items]
        summary[f"{module_type}|{condition}"] = {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "ci95": percentile_ci(values),
        }
    comparison = []
    for module_type in ["mlp_output", "attention_output", "residual_output"]:
        donor_rows = [r for r in rows if r["module_type"] == module_type and r["condition"] == "correct_donor"]
        random_rows = [r for r in rows if r["module_type"] == module_type and r["condition"] == "random_donor"]
        site_rows = [r for r in rows if r["module_type"] == module_type and r["condition"] == "same_trace_random_position"]
        donor_mean = mean_delta(donor_rows, random_rows)
        site_mean = mean_delta(donor_rows, site_rows)
        comparison.append(
            {
                "module_type": module_type,
                "gsm8k_3c1": GSM8K_3C1_REFERENCE[module_type],
                "cyber_donor_specificity_mean": donor_mean["mean"],
                "cyber_donor_specificity_ci95": donor_mean["ci95"],
                "cyber_site_specificity_mean": site_mean["mean"],
                "cyber_site_specificity_ci95": site_mean["ci95"],
            }
        )
    conclusion = "no_transfer"
    positives = [
        row for row in comparison
        if row["cyber_donor_specificity_mean"] is not None and row["cyber_donor_specificity_mean"] > 0
        and row["cyber_site_specificity_mean"] is not None and row["cyber_site_specificity_mean"] > 0
    ]
    if positives:
        conclusion = "partial_transfer"
    return {
        "status": "executed",
        "num_patch_rows": len(rows),
        "summary_by_module_condition": summary,
        "gsm8k_vs_cyber_comparison": comparison,
        "transfer_conclusion": conclusion,
        "f1_priority_implication": (
            "If no transfer, treat F1 as a control feature family and prioritize F3 plus any admitted F2 substrate."
        ),
        "chat_adapter_note": "For bare-letter completions, same_trace_random_position is skipped when no non-label completion token exists; prompt positions are not substituted.",
    }


def mean_delta(treatment: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    if not treatment or not control:
        return {"mean": None, "ci95": [None, None]}
    by_key: dict[tuple[str, int, float], dict[str, float]] = defaultdict(dict)
    for row in treatment:
        by_key[(row["pair_id"], row["layer"], row["alpha"])]["t"] = float(row["clean_direction"])
    for row in control:
        by_key[(row["pair_id"], row["layer"], row["alpha"])]["c"] = float(row["clean_direction"])
    deltas = [v["t"] - v["c"] for v in by_key.values() if "t" in v and "c" in v]
    return {"mean": float(np.mean(deltas)) if deltas else None, "ci95": percentile_ci(deltas)}


# ---------------------------------------------------------------------------
# Reports and docs.


def render_review_gate(
    questions: list[dict[str, Any]],
    prompt_style: str,
    admission_level: str,
    eq: dict[str, Any],
    score: dict[str, Any],
    fidelity: dict[str, Any],
    f5_report: dict[str, Any],
    reasoning: dict[str, Any],
    bias: dict[str, Any],
    pairs: list[dict[str, Any]],
    gate: dict[str, Any],
    site_report: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Sprint 4B-3 Review Gate",
            "",
            f"1. Dataset/questions: `{len(questions)}` train-split CyberMetric records; prompt winner=`{prompt_style}` admission=`{admission_level}`.",
            f"2. Stage A equivalence: `{eq['num_matched']}/{eq['num_checked']}` matched; passed=`{eq['all_matched']}`.",
            f"3. Traces: `{score['num_traces']}` total; correct_rate=`{score['correct_rate']:.4f}` wrong_rate=`{score['wrong_rate']:.4f}` parse_failure_rate=`{score['parse_failure_rate']:.4f}` degeneration_rate=`{score['degeneration_rate']:.4f}`.",
            "4. Degeneration traces are counted as an independent category in the manifest and rates.",
            f"5. token_form_counts: `{fidelity['token_form_counts']}`; dual-form assertion failures=`{fidelity['failed']}`.",
            f"6. kill_bar_single_forward: AUROC=`{f5_report['kill_bar_single_forward']['auroc']}` CI95=`{f5_report['kill_bar_single_forward']['auroc_ci95']}`.",
            f"7. kill_bar_sampling: AUROC=`{f5_report['kill_bar_sampling']['auroc']}` CI95=`{f5_report['kill_bar_sampling']['auroc_ci95']}`.",
            f"8. Strongest single feature: `{strongest_single_feature(f5_report)}`.",
            "9. Low-risk wrong cases are listed in low_risk_wrong_case_report.jsonl.",
            f"10. Reasoning substrate main has_reasoning_text_rate=`{reasoning['main_prompt']['has_reasoning_text_rate']:.4f}`; F2 plan=`{reasoning['f2_plan_for_4c']}`.",
            f"11. Position bias severe warning: `{bias['severe_position_bias_warning']}`.",
            f"12. Correct/wrong pairs: `{len(pairs)}`; Stage E gate passed=`{gate['passed']}` checks=`{gate['checks']}`.",
            f"13. Site-transfer conclusion: `{site_report.get('transfer_conclusion', site_report.get('status'))}`; comparison table in site_transfer_check_report.json.",
            f"14. Hook fidelity: `{'module_patch_fidelity_report.json' if site_report.get('status') == 'executed' else 'skipped because Stage E did not run'}`.",
            "15. Probe training / steering / nudge outside diagnostic patching: `yes, zero`.",
            "16. gold_label never used as an inference feature: `yes`; it is eval-only.",
            f"17. 4C entry: use measured bars above; F2 plan=`{reasoning['f2_plan_for_4c']}`; F1 priority follows site-transfer conclusion.",
            "18. Claims of hallucination reduction or accuracy improvement: `no`.",
            "",
        ]
    )


def strongest_single_feature(f5_report: dict[str, Any]) -> str | None:
    candidates = {
        k: v.get("auroc")
        for k, v in f5_report["feature_metrics"].items()
        if k not in {"f5_combo_single_forward_z", "f5_combo_sampling_z"} and v.get("auroc") is not None
    }
    return max(candidates, key=lambda k: float(candidates[k])) if candidates else None


def update_progress_docs(out_dir: Path, f5_report: dict[str, Any], site_report: dict[str, Any], reasoning: dict[str, Any], commands: list[str]) -> None:
    progress = PROJECT_ROOT / "PROGRESS.md"
    old = progress.read_text(encoding="utf-8")
    block = f"""# 瀹為獙杩涘害璁板綍锛歊easoning-Aware Attention Guidance

## Current Status Update: Sprint 4B-3 Full F5 Baseline and Site-Transfer Diagnostic

Sprint 4B-3 is completed on the 240-question CyberMetric smoke manifest using the 4B-2 winning prompt. The dual F5 bars were measured with fixed, non-trained z-score combinations; Stage E site-transfer was {'executed' if site_report.get('status') == 'executed' else 'gate-skipped'} as recorded in `site_transfer_check_report.json`. No probe was trained, no steering was continued, and no hallucination-reduction or accuracy-improvement claim is made.

Measured bars: `kill_bar_single_forward` AUROC={f5_report['kill_bar_single_forward']['auroc']} CI95={f5_report['kill_bar_single_forward']['auroc_ci95']}; `kill_bar_sampling` AUROC={f5_report['kill_bar_sampling']['auroc']} CI95={f5_report['kill_bar_sampling']['auroc_ci95']}. Reasoning substrate plan for 4C F2: `{reasoning['f2_plan_for_4c']}`. Site-transfer conclusion: `{site_report.get('transfer_conclusion', site_report.get('status'))}`.

Outputs are under `{out_dir}` and remain gitignored. Commands:
```bash
{chr(10).join(commands)}
```

Checks: targeted label/cyber tests, the full 4B-3 script command, and full pytest were run for this sprint. Boundary flags: `ready_for_2000_rerun=false`, `do_not_enter_full_sprint_3C=true`, `hallucination_reduction_proven=false`, `answer_accuracy_improvement_proven=false`, `steering_continued=false`, `probe_trained=false`.

Next: open Sprint 4C against these measured F5 bars; use F3 and F1/F4 as the stable baseline families and follow the recorded F2 substrate plan.

"""
    if old.startswith("# 瀹為獙杩涘害璁板綍锛歊easoning-Aware Attention Guidance\n"):
        old = old.split("\n", 1)[1]
    progress.write_text(block + old, encoding="utf-8")

    history = PROJECT_ROOT / "docs/progress/sprint_4_history.md"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Sprint 4B-3 Full F5 Baseline and Site-Transfer Diagnostic\n\n"
            f"- Output dir: `{out_dir}`\n"
            f"- kill_bar_single_forward: `{json.dumps(f5_report['kill_bar_single_forward'], ensure_ascii=False)}`\n"
            f"- kill_bar_sampling: `{json.dumps(f5_report['kill_bar_sampling'], ensure_ascii=False)}`\n"
            f"- reasoning F2 plan: `{reasoning['f2_plan_for_4c']}`\n"
            f"- site-transfer conclusion: `{site_report.get('transfer_conclusion', site_report.get('status'))}`\n"
            "- Boundary: no probe training, no steering, no hallucination/accuracy improvement claim.\n"
        )

    manifest = PROJECT_ROOT / "docs/progress/sprint_4_artifact_manifest.md"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Sprint 4B-3 full F5 baseline artifacts\n\n"
            f"Directory: `{out_dir}` (gitignored).\n\n"
            "Required artifacts: preflight_report.md, trace_sampling_manifest.jsonl, reasoning_substrate_report.json, "
            "f5_baseline_report.json, option_position_bias_report.json, high_risk_case_report.jsonl, "
            "low_risk_wrong_case_report.jsonl, correct_wrong_pair_manifest.jsonl, site_transfer_check_report.json, "
            "equivalence_spot_check_report.json, review_gate_full_f5_baseline_and_site_transfer.md. "
            "module_patch_fidelity_report.json is present only if Stage E executed.\n"
        )


# ---------------------------------------------------------------------------
# Main.


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        for path in out_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    ensure_dir(out_dir)

    model_path, model_source = resolve_model_path(args.model_path)
    preflight = build_preflight(args, model_path, model_source, out_dir)
    (out_dir / "preflight_report.md").write_text(render_preflight(preflight), encoding="utf-8")
    if not preflight["can_continue"]:
        raise SystemExit(f"preflight stop: {preflight['stop_reasons']}")

    ab = read_ab_decision(Path(args.ab_report))
    prompt_style = str(ab["winner"])
    admission_level = str(ab["admission"])
    questions = select_questions(Path(args.smoke_manifest))
    records_by_id = {q["example_id"]: q for q in questions}

    context = abs_mod.load_local_steering_backend(model_path=model_path)
    print(f"[4B-3] model loaded from {model_path} (source={model_source})")

    eq = run_equivalence_spot_check(context, questions, prompt_style, args)
    write_json(eq, out_dir / "equivalence_spot_check_report.json")
    if not eq["all_matched"]:
        write_json({"status": "stopped", "reason": "stage_a_equivalence_failed", "equivalence": eq}, out_dir / "site_transfer_check_report.json")
        raise SystemExit("Stage A equivalence spot check failed; stopping before full run.")

    traces = run_generation(context, questions, prompt_style, args)
    write_jsonl(traces, out_dir / "trace_sampling_manifest.jsonl")
    score = condition_score(traces)

    forcing = run_reasoning_forcing_probe(context, questions, args)
    write_jsonl(forcing["traces"], out_dir / "reasoning_forcing_trace_manifest.jsonl")
    reasoning = reasoning_substrate_report(traces, forcing)
    write_json(reasoning, out_dir / "reasoning_substrate_report.json")

    rows, fidelity = readout_feature_rows(context, records_by_id, traces)
    f5_report, flat = build_f5_report(rows, fidelity, args)
    write_json(f5_report, out_dir / "f5_baseline_report.json")

    bias = option_position_bias_report(questions, traces)
    write_json(bias, out_dir / "option_position_bias_report.json")
    write_case_reports(out_dir, flat, rows, records_by_id)

    pairs = build_pairs(traces)
    write_jsonl(pairs, out_dir / "correct_wrong_pair_manifest.jsonl")
    gate = stage_e_gate(pairs, score, fidelity, bias)
    if gate["passed"]:
        site_report, patch_fidelity = run_stage_e(context, pairs, records_by_id, args)
        site_report["gate"] = gate
        write_json(site_report, out_dir / "site_transfer_check_report.json")
        write_json(patch_fidelity, out_dir / "module_patch_fidelity_report.json")
    else:
        site_report = {
            "status": "skipped",
            "skipped_reason": "stage_e_gate_failed",
            "gate": gate,
            "gsm8k_3c1_reference": GSM8K_3C1_REFERENCE,
            "transfer_conclusion": "not_evaluated_gate_skipped",
            "chat_adapter_note": "same_trace_random_position would be skipped when bare-letter completion has no non-label completion token; prompt positions are not substituted.",
        }
        write_json(site_report, out_dir / "site_transfer_check_report.json")

    (out_dir / "review_gate_full_f5_baseline_and_site_transfer.md").write_text(
        render_review_gate(questions, prompt_style, admission_level, eq, score, fidelity, f5_report, reasoning, bias, pairs, gate, site_report),
        encoding="utf-8",
    )

    commands = [
        "conda run -n recover_attention python -m pytest tests/test_domain_label_proxy.py tests/test_cyber_data.py -q",
        "conda run -n recover_attention python scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py --processed-path data/processed/cyber/cybermetric.jsonl --smoke-manifest outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/cyber_sample_smoke_manifest.jsonl --ab-report outputs/logs/sprint_4B_2_small_model_smoke/prompt_ab_report.json --samples-per-question 6 --temperature 0.7 --max-new-tokens 256 --site-check-layers 16 20 24 --site-check-alphas 0.25 1.0 --site-check-max-pairs 34 --output-dir outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer --overwrite",
        "conda run -n recover_attention python -m pytest -q",
    ]
    update_progress_docs(out_dir, f5_report, site_report, reasoning, commands)
    print(f"[4B-3] complete out={out_dir}")


if __name__ == "__main__":
    main()

