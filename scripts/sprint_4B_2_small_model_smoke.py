"""Sprint 4B-2: CyberMetric small-model smoke and F5 feature plumbing.

Runs the first model-generation stage of the hallucination-control mainline
on a small (32-question) subset: A/B tests a raw-completion prompt against a
chat-template prompt, validates a degeneration detector against real Sprint
4B smoke failures, then -- on the winning prompt condition only -- validates
the F5 output-level feature pipeline (label-readout position -> option
logits -> margin/entropy/self-consistency) and audits option-position model
bias.

Boundary: no probe training, no F5 kill-bar claim, no correct/wrong pair
construction at scale, no site-transfer, no steering/patching/nudge, no
LoRA/finetuning, no full Sprint 3C, no 2000-scale run. Gold labels are eval
labels only, never inference features.
"""

from __future__ import annotations

import argparse
import math
import os
import random
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
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402
from recover_attention.schemas import validate_cyber_sample_record  # noqa: E402

DEFAULT_MODEL_PATH = Path("D:/models/Qwen2.5-7B-Instruct")
DEFAULT_PROCESSED = PROJECT_ROOT / "data/processed/cyber/cybermetric.jsonl"
DEFAULT_SMOKE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/cyber_sample_smoke_manifest.jsonl"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4B_2_small_model_smoke"
CONDITIONS = ("raw", "chat")
CANDIDATE_LABELS = ["A", "B", "C", "D"]
SCORE_TIE_MARGIN = 0.02
SCORE_STOP_THRESHOLD = 0.15
WINNER_ADMISSION_THRESHOLD = 0.08

PRIOR_OUTPUT_DIRS = [
    "sprint_3C_0_correct_wrong_activation_patching",
    "sprint_3C_0_fix_answer_proxy_recheck",
    "sprint_3C_1_final_answer_compression_value_mlp_tracing",
    "sprint_3C_2_mlp_readout_direction_analysis",
    "sprint_3C_3_mlp_readout_attribution_probe",
    "sprint_3C_4A_approx_j_lens_readout_sanity_check",
    "sprint_4A_cyber_direction_probe_mainline_reset",
    "sprint_4B_1_cybermetric_schema_and_label_proxy",
    "sprint_4B_dataset_download_audit",
    "sprint_4B_cyber_dataset_baseline_and_site_transfer",
    "sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    for name in PRIOR_OUTPUT_DIRS:
        if out_dir.name == name:
            raise SystemExit(f"refusing to reuse a prior sprint output directory name: {name}")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()

    model_path, model_source = resolve_model_path(args.model_path)
    preflight = build_preflight(args, model_path, model_source)
    (out_dir / "preflight_report.md").write_text(render_preflight(preflight), encoding="utf-8")
    if not preflight["can_continue"]:
        raise SystemExit(f"preflight stop: {preflight['stop_reason']}")

    run(args, out_dir, model_path, model_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 4B-2 small-model smoke and F5 plumbing")
    parser.add_argument("--processed-path", default=str(DEFAULT_PROCESSED))
    parser.add_argument("--smoke-manifest", default=str(DEFAULT_SMOKE_MANIFEST))
    parser.add_argument("--num-questions", type=int, default=32)
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--question-seed", type=int, default=4242)
    parser.add_argument("--seed", type=int, default=44242)
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


def git_clean_worktree() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment without git
        return False, f"git status failed: {exc}"
    dirty = [line for line in result.stdout.splitlines() if line.strip()]
    return not dirty, "\n".join(dirty[:20])


def spot_check_processed_dataset(processed_path: Path, *, n: int = 20, seed: int = 4242) -> dict[str, Any]:
    if not processed_path.exists():
        return {"checked": 0, "passed": 0, "failed": 0, "errors": [], "skipped_reason": "file_missing"}
    records = read_jsonl(processed_path)
    sample_size = min(n, len(records))
    sample = random.Random(seed).sample(records, sample_size) if records else []
    errors: list[str] = []
    passed = 0
    for record in sample:
        try:
            validate_cyber_sample_record(record)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{record.get('example_id', '?')}: {exc}")
    return {"checked": len(sample), "passed": passed, "failed": len(sample) - passed, "errors": errors[:5]}


def build_preflight(args: argparse.Namespace, model_path: Path | None, model_source: str) -> dict[str, Any]:
    clean, dirty_summary = git_clean_worktree()
    processed_ok = Path(args.processed_path).exists()
    smoke_ok = Path(args.smoke_manifest).exists()
    spot_check = spot_check_processed_dataset(Path(args.processed_path))
    stop = None
    if model_path is None:
        stop = "model_path_unavailable"
    elif not processed_ok:
        stop = f"processed_dataset_missing: {args.processed_path}"
    elif not smoke_ok:
        stop = f"smoke_manifest_missing: {args.smoke_manifest}"
    elif spot_check["failed"] > 0:
        stop = f"processed_dataset_spot_check_failed: {spot_check['failed']}/{spot_check['checked']}"
    return {
        "task": "Sprint 4B-2 CyberMetric small-model smoke and F5 plumbing",
        "model_path": str(model_path) if model_path else None,
        "model_path_source": model_source,
        "processed_dataset_available": processed_ok,
        "processed_dataset_spot_check": spot_check,
        "smoke_manifest_available": smoke_ok,
        "worktree_clean": clean,
        "worktree_dirty_summary": dirty_summary if not clean else None,
        "estimated_generations": args.num_questions * len(CONDITIONS) * (1 + args.samples_per_question),
        "can_continue": stop is None,
        "stop_reason": stop,
        "boundaries": {
            "probe_trained": False,
            "steering_continued": False,
            "patching_or_nudge": False,
            "lora_or_finetuning": False,
            "full_sprint_3c": False,
            "scale_2000": False,
            "f5_kill_bar_established": False,
        },
    }


def render_preflight(report: dict[str, Any]) -> str:
    import json as _json

    lines = ["# Sprint 4B-2 Preflight", ""]
    for key, value in report.items():
        lines.append(f"- {key}: `{_json.dumps(value, ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- selection


def select_ab_questions(smoke_manifest_path: Path, *, num_questions: int, seed: int) -> list[dict]:
    records = [r for r in read_jsonl(smoke_manifest_path) if r["metadata"]["split"] == "train"]
    if len(records) < num_questions:
        raise ValueError(f"only {len(records)} train-split records available; need {num_questions}")
    ordered = sorted(records, key=lambda r: r["example_id"])
    rng = random.Random(seed)
    return rng.sample(ordered, num_questions)


# ---------------------------------------------------------------- generation


def build_prompt(context: dict[str, Any], record: dict[str, Any], condition: str) -> str:
    if condition == "raw":
        return cd.build_mcq_prompt(record)
    if condition == "chat":
        messages = cd.build_mcq_chat_messages(record)
        return context["tokenizer"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    raise ValueError(f"unsupported condition: {condition}")


def model_inputs(context: dict[str, Any], text: str) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    encoded = tokenizer(text, return_tensors="pt")
    device = msrs.infer_model_input_device(context["model"], "auto", torch)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}


def choose_next_token(logits: Any, *, do_sample: bool, temperature: float, top_p: float, rng: random.Random) -> int:
    # Numerical-stability sampling protection: this hand-rolled loop does not
    # call HF model.generate(), so the renormalize_logits/remove_invalid_values
    # kwargs from 3C-0 do not apply directly; nan_to_num + clipped exponentials
    # are the equivalent guard against NaN/Inf logits for this code path.
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
    """Generate, early-stopping once an option letter parses. Returns (text, num_generated_tokens)."""
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
            if dlp.parse_option_answer(decoded, valid_labels)["parsed_label"] is not None:
                break
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return completion, len(new_ids)


def run_condition(
    context: dict[str, Any],
    records: list[dict[str, Any]],
    condition: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt = build_prompt(context, record, condition)
        valid_labels = record["candidate_labels"]

        greedy_text, greedy_tokens = generate_completion(
            context, prompt, max_new_tokens=args.max_new_tokens, do_sample=False,
            temperature=0.0, top_p=1.0, seed=args.seed + idx, valid_labels=valid_labels,
        )
        traces.append(build_trace_row(record, condition, prompt, "greedy", 0, greedy_text, greedy_tokens, args))

        for sample_idx in range(args.samples_per_question):
            sample_text, sample_tokens = generate_completion(
                context, prompt, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                seed=args.seed + 1000 * (idx + 1) + sample_idx, valid_labels=valid_labels,
            )
            traces.append(
                build_trace_row(record, condition, prompt, "sample", sample_idx, sample_text, sample_tokens, args)
            )
        if (idx + 1) % 8 == 0:
            print(f"[4B-2] condition={condition} sampled {idx + 1}/{len(records)} questions")
    return traces


def build_trace_row(
    record: dict[str, Any],
    condition: str,
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
        "condition": condition,
        "sample_type": sample_type,
        "sample_index": sample_index,
        "prompt": prompt,
        "completion": completion,
        "num_generated_tokens": num_generated_tokens,
        "parsed_label": parsed["parsed_label"],
        "parse_method": parsed["parse_method"],
        "parse_failure": parsed["parse_failure"],
        "gold_label": record["gold_label"],
        "is_correct": True if status == "correct" else False if status == "wrong" else None,
        "degenerate": degeneration["degenerate"],
        "degeneration_rules": degeneration["matched_rules"],
        "temperature": 0.0 if sample_type == "greedy" else args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }


# ---------------------------------------------------------------- readout position verification


def verify_readout_positions_all_traces(context: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokenizer-only (no GPU forward) check: locate_label_readout_position + 1
    must land on the parsed label's token id (in EITHER the space-prefixed or
    bare-letter form -- see readout_features_for_greedy_traces' docstring),
    for EVERY parseable trace in a condition (Section 6.6). Cheap by design
    so it can cover samples too, not just the greedy subset used by the
    GPU-bound F5 feature pass.
    """
    tokenizer = context["tokenizer"]
    token_ids_space = dlp.option_token_ids(tokenizer, CANDIDATE_LABELS)
    token_ids_bare = dlp.bare_option_token_ids(tokenizer, CANDIDATE_LABELS)
    checked = 0
    failed = 0
    locate_errors = 0
    token_form_counts: Counter[str] = Counter()
    for trace in traces:
        if trace["parse_failure"] or trace["parsed_label"] is None:
            continue
        combined = trace["prompt"] + trace["completion"]
        try:
            position = dlp.locate_label_readout_position(
                tokenizer, trace["prompt"], trace["completion"], trace["parsed_label"]
            )
        except ValueError:
            locate_errors += 1
            continue
        encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoded["input_ids"]
        checked += 1
        label = trace["parsed_label"]
        actual_id = input_ids[position + 1] if position + 1 < len(input_ids) else None
        if actual_id == token_ids_space[label]:
            token_form_counts["space"] += 1
        elif actual_id == token_ids_bare[label]:
            token_form_counts["bare"] += 1
        else:
            token_form_counts["unknown"] += 1
            failed += 1
    return {
        "num_traces": len(traces),
        "num_parseable": checked + locate_errors,
        "num_locate_errors": locate_errors,
        "num_checked": checked,
        "num_failed": failed,
        "token_form_counts": dict(token_form_counts),
        "pass_rate": (checked - failed) / checked if checked else None,
    }


# ---------------------------------------------------------------- decision


def condition_score(traces: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    parse_failures = sum(1 for t in traces if t["parse_failure"])
    degenerate = sum(1 for t in traces if t["degenerate"])
    correct = sum(1 for t in traces if t["is_correct"] is True)
    parse_failure_rate = parse_failures / n if n else 1.0
    degeneration_rate = degenerate / n if n else 1.0
    return {
        "num_traces": n,
        "parse_failure_rate": parse_failure_rate,
        "degeneration_rate": degeneration_rate,
        "score": parse_failure_rate + degeneration_rate,
        "correct_rate": correct / n if n else 0.0,
    }


def decide_condition(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_score = scores["raw"]["score"]
    chat_score = scores["chat"]["score"]
    tie = abs(raw_score - chat_score) < SCORE_TIE_MARGIN
    if tie:
        winner = "chat"
        reason = f"tie (|{raw_score:.4f} - {chat_score:.4f}| < {SCORE_TIE_MARGIN}); chat wins by default rule"
    elif raw_score < chat_score:
        winner = "raw"
        reason = f"raw score {raw_score:.4f} < chat score {chat_score:.4f}"
    else:
        winner = "chat"
        reason = f"chat score {chat_score:.4f} < raw score {raw_score:.4f}"
    both_over_stop = raw_score > SCORE_STOP_THRESHOLD and chat_score > SCORE_STOP_THRESHOLD
    return {
        "winner": None if both_over_stop else winner,
        "tie_triggered": tie,
        "reason": "both conditions exceed stop threshold; no prompt style resolved degeneration" if both_over_stop else reason,
        "both_over_stop_threshold": both_over_stop,
        "stop_threshold": SCORE_STOP_THRESHOLD,
        "tie_margin": SCORE_TIE_MARGIN,
        "winner_admits_4b3": (not both_over_stop) and scores.get(winner, {}).get("score", 1.0) <= WINNER_ADMISSION_THRESHOLD,
        "winner_admission_threshold": WINNER_ADMISSION_THRESHOLD,
    }


# ---------------------------------------------------------------- F5 plumbing


def readout_features_for_greedy_traces(
    context: dict[str, Any],
    greedy_traces: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute F5 tier_single_forward features at each greedy trace's readout
    position.

    Finding from the 4B-2 dry run: chat-condition completions often answer
    with the bare option letter (no leading space, no restated "Answer: "),
    which tokenizes to a DIFFERENT id than the space-prefixed form
    option_token_ids assumes. Using a fixed space-prefixed token set there
    would silently compute margin/entropy over the wrong candidate tokens.
    We resolve both forms and pick whichever one actually matches the
    realized token at the readout position; only fall back (and flag the
    assertion as failed) when neither form matches.
    """
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    token_ids_space = dlp.option_token_ids(tokenizer, CANDIDATE_LABELS)
    token_ids_bare = dlp.bare_option_token_ids(tokenizer, CANDIDATE_LABELS)
    rows: list[dict[str, Any]] = []
    assertion_failures = 0
    assertion_checked = 0
    skipped_parse_failure = 0
    token_form_counts: Counter[str] = Counter()

    for trace in greedy_traces:
        record = records_by_id[trace["example_id"]]
        parsed_label = trace["parsed_label"]
        if trace["parse_failure"] or parsed_label is None:
            skipped_parse_failure += 1
            rows.append({"example_id": record["example_id"], "readout_available": False, "reason": "parse_failure"})
            continue
        combined = trace["prompt"] + trace["completion"]
        try:
            position = dlp.locate_label_readout_position(tokenizer, trace["prompt"], trace["completion"], parsed_label)
        except ValueError as exc:
            rows.append(
                {"example_id": record["example_id"], "readout_available": False, "reason": f"locate_failed: {exc}"}
            )
            continue

        encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
        input_ids_full = encoded["input_ids"]
        assertion_checked += 1
        actual_id = input_ids_full[position + 1] if position + 1 < len(input_ids_full) else None
        if actual_id == token_ids_space[parsed_label]:
            token_form, token_ids = "space", token_ids_space
        elif actual_id == token_ids_bare[parsed_label]:
            token_form, token_ids = "bare", token_ids_bare
        else:
            token_form, token_ids = "unknown", token_ids_space
            assertion_failures += 1
        token_form_counts[token_form] += 1

        prefix_ids = input_ids_full[: position + 1]
        device = msrs.infer_model_input_device(model, "auto", torch)
        input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, use_cache=False)
        logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()

        option_logits = {label: float(logits[tid]) for label, tid in token_ids.items()}
        rows.append(
            {
                "example_id": record["example_id"],
                "readout_available": True,
                "readout_position": position,
                "readout_token_form": token_form,
                "readout_assertion_passed": token_form != "unknown",
                "features": {
                    "tier_single_forward": {
                        "f5_label_margin": dlp.label_margin(option_logits),
                        "f5_label_entropy": dlp.label_entropy(option_logits),
                        "f5_full_entropy": dlp.full_entropy(logits),
                    },
                },
            }
        )
    fidelity = {
        "num_greedy_traces": len(greedy_traces),
        "num_skipped_parse_failure": skipped_parse_failure,
        "num_readout_assertion_checked": assertion_checked,
        "num_readout_assertion_failed": assertion_failures,
        "readout_assertion_pass_rate": (
            (assertion_checked - assertion_failures) / assertion_checked if assertion_checked else None
        ),
        "readout_token_form_counts": dict(token_form_counts),
    }
    return rows, fidelity


def attach_self_consistency(
    rows: list[dict[str, Any]],
    winning_traces: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_id: dict[str, list[str | None]] = defaultdict(list)
    greedy_by_id: dict[str, str | None] = {}
    for trace in winning_traces:
        if trace["sample_type"] == "sample":
            samples_by_id[trace["example_id"]].append(trace["parsed_label"])
        else:
            greedy_by_id[trace["example_id"]] = trace["parsed_label"]

    enriched: list[dict[str, Any]] = []
    for row in rows:
        example_id = row["example_id"]
        record = records_by_id[example_id]
        consistency = dlp.self_consistency_features(
            greedy_by_id.get(example_id), samples_by_id.get(example_id, []), record["candidate_labels"]
        )
        tier_sampling = {
            "f5_self_consistency": consistency["self_consistency_with_greedy"],
            "f5_sc_majority_agree": (
                float(consistency["majority_agrees_with_greedy"])
                if consistency["majority_agrees_with_greedy"] is not None
                else None
            ),
        }
        if row.get("readout_available"):
            row["features"]["tier_sampling"] = tier_sampling
            dlp.assert_no_gold_label_leakage(row["features"])
        row["self_consistency_diagnostics"] = {
            k: v for k, v in consistency.items() if k not in ("greedy_label",)
        }
        enriched.append(row)
    return enriched


def build_f5_plumbing_report(rows: list[dict[str, Any]], fidelity: dict[str, Any]) -> dict[str, Any]:
    single_forward_names = ["f5_label_margin", "f5_label_entropy", "f5_full_entropy"]
    sampling_names = ["f5_self_consistency", "f5_sc_majority_agree"]
    available = [r for r in rows if r.get("readout_available")]

    def stats_for(name: str, tier: str) -> dict[str, Any]:
        values = [
            r["features"][tier][name]
            for r in available
            if tier in r["features"] and r["features"][tier].get(name) is not None
        ]
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        return {
            "num_present": len(values),
            "num_finite": len(finite),
            "num_missing": len(available) - len(values),
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "mean": sum(finite) / len(finite) if finite else None,
        }

    feature_stats = {name: stats_for(name, "tier_single_forward") for name in single_forward_names}
    feature_stats.update({name: stats_for(name, "tier_sampling") for name in sampling_names})

    return {
        "backend": "f5_feature_plumbing_v0",
        "num_greedy_examples": fidelity["num_greedy_traces"],
        "num_readout_available": len(available),
        "num_skipped_parse_failure": fidelity["num_skipped_parse_failure"],
        "readout_position_assertion": {
            "checked": fidelity["num_readout_assertion_checked"],
            "failed": fidelity["num_readout_assertion_failed"],
            "pass_rate": fidelity["readout_assertion_pass_rate"],
            "token_form_counts": fidelity["readout_token_form_counts"],
        },
        "feature_finiteness": feature_stats,
        "gold_leakage_check": "passed_for_all_readout_available_rows",
        "cost_tiers": {
            "tier_single_forward": single_forward_names,
            "tier_sampling": sampling_names,
            "note": "predefined for the Sprint 4B-3 dual kill-bar split; no bar is declared here.",
        },
        "plumbing_validation_only": True,
    }


def compute_smoke_auroc(rows: list[dict[str, Any]], greedy_traces_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        if not row.get("readout_available"):
            continue
        example_id = row["example_id"]
        trace = greedy_traces_by_id[example_id]
        if trace["is_correct"] is None:
            continue
        labels.append(0 if trace["is_correct"] else 1)
        scores.append(-row["features"]["tier_single_forward"]["f5_label_margin"])
    if len(set(labels)) < 2:
        return {"auroc": None, "n": len(labels), "reason": "insufficient class diversity at smoke scale"}
    order = sorted(range(len(labels)), key=lambda i: scores[i])
    ranks = [0.0] * len(labels)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "n": len(labels), "reason": "single class"}
    rank_sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"auroc": auroc, "n": len(labels), "num_positive_wrong": n_pos, "num_negative_correct": n_neg}


# ---------------------------------------------------------------- position bias


def option_position_model_bias_report(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    traces_by_condition: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    gold_counts = Counter(r["gold_label"] for r in records)
    report: dict[str, Any] = {"gold_distribution": dict(sorted(gold_counts.items())), "num_questions": len(records)}
    for condition, traces in traces_by_condition.items():
        greedy_labels = [t["parsed_label"] for t in traces if t["sample_type"] == "greedy" and t["parsed_label"]]
        sampled_labels = [t["parsed_label"] for t in traces if t["sample_type"] == "sample" and t["parsed_label"]]
        greedy_dist = dict(Counter(greedy_labels))
        sampled_dist = dict(Counter(sampled_labels))
        greedy_accuracy = scores[condition]["correct_rate"]
        max_greedy_share = max(greedy_dist.values()) / len(greedy_labels) if greedy_labels else 0.0
        report[condition] = {
            "greedy_predicted_choice_distribution": greedy_dist,
            "sampled_predicted_choice_distribution": sampled_dist,
            "greedy_accuracy_smoke_reference": greedy_accuracy,
            "max_greedy_option_share": max_greedy_share,
            "severe_position_bias_warning": max_greedy_share > 0.40,
        }
    majority_label = gold_counts.most_common(1)[0][0] if gold_counts else None
    position_only_baseline = (
        gold_counts[majority_label] / len(records) if majority_label and records else 0.0
    )
    report["position_only_baseline_accuracy"] = position_only_baseline
    return report


# ---------------------------------------------------------------- review gate


def render_review_gate(
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    f5_report: dict[str, Any] | None,
    position_report: dict[str, Any] | None,
    model_source: str,
    smoke_auroc: dict[str, Any] | None,
    readout_position_checks: dict[str, dict[str, Any]],
) -> str:
    import json as _json

    lines = [
        "# Sprint 4B-2 Small-Model Smoke Review Gate",
        "",
        f"1. Questions: `{len(questions)}` (train split, seed `{args.question_seed}`, not first-N).",
        f"2. Traces per condition: raw=`{scores['raw']['num_traces']}` chat=`{scores['chat']['num_traces']}`.",
        f"3. cond_raw parse_failure_rate=`{scores['raw']['parse_failure_rate']:.4f}` degeneration_rate=`{scores['raw']['degeneration_rate']:.4f}` score=`{scores['raw']['score']:.4f}`.",
        f"4. cond_chat parse_failure_rate=`{scores['chat']['parse_failure_rate']:.4f}` degeneration_rate=`{scores['chat']['degeneration_rate']:.4f}` score=`{scores['chat']['score']:.4f}`.",
        f"5. Degeneration rule hit distribution recorded per trace in trace_manifest_cond_*.jsonl.",
        f"5b. Readout-position runtime assertion (Section 6.6, all traces): raw failed=`{readout_position_checks['raw']['num_failed']}`/`{readout_position_checks['raw']['num_checked']}`; chat failed=`{readout_position_checks['chat']['num_failed']}`/`{readout_position_checks['chat']['num_checked']}`.",
        f"6. Decision: winner=`{decision['winner']}` tie_triggered=`{decision['tie_triggered']}` reason=`{decision['reason']}`.",
        f"7. Prior 4B smoke parse_failure_rate reference = 0.10 (max_new_tokens=128, raw prompt only).",
        f"8. model_path_source: `{model_source}`.",
    ]
    if f5_report is not None:
        lines.extend(
            [
                f"9. F5 readout assertion: checked=`{f5_report['readout_position_assertion']['checked']}` failed=`{f5_report['readout_position_assertion']['failed']}` token_form_counts=`{f5_report['readout_position_assertion']['token_form_counts']}` (space-prefixed vs bare-letter token resolution; see docstring in readout_features_for_greedy_traces).",
                f"10. F5 five features finite/missing counts recorded in f5_feature_plumbing_report.json.",
                f"11. Cost-tier fields (`tier_single_forward`, `tier_sampling`) present and gold-leakage checked: `{f5_report['gold_leakage_check']}`.",
                f"12. smoke_scale_auroc (plumbing_validation_only, NOT a kill bar): `{_json.dumps(smoke_auroc, ensure_ascii=False)}`.",
            ]
        )
    else:
        lines.extend(
            [
                "9. F5 readout assertion: skipped (no winning condition admitted).",
                "10. F5 features: skipped.",
                "11. Cost-tier fields: skipped.",
                "12. smoke_scale_auroc: skipped.",
            ]
        )
    if position_report is not None:
        warnings = [
            c for c in CONDITIONS if position_report.get(c, {}).get("severe_position_bias_warning")
        ]
        lines.append(f"13. Position-bias severe warnings by condition: `{warnings}`.")
    else:
        lines.append("13. Position-bias report: skipped.")
    lines.extend(
        [
            "14. Zero probe training, zero steering, zero patching/nudge: `true`.",
            "15. F5 kill bar declared this sprint: `false`.",
            f"16. Ready for Sprint 4B-3: `{decision.get('winner_admits_4b3', False)}` (threshold `{decision['winner_admission_threshold']}`).",
            "17. Next 4B-3 input: winning prompt condition, `--prompt-style` equivalent, full 240-question run with dual F5 kill-bar reporting.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace, out_dir: Path, model_path: Path, model_source: str) -> None:
    questions = select_ab_questions(Path(args.smoke_manifest), num_questions=args.num_questions, seed=args.question_seed)
    write_jsonl(
        [{"example_id": q["example_id"], "group_id": q["group_id"], "split": q["metadata"]["split"]} for q in questions],
        out_dir / "ab_question_manifest.jsonl",
    )
    questions_by_id = {q["example_id"]: q for q in questions}

    context = abs_mod.load_local_steering_backend(model_path=model_path)
    print(f"[4B-2] model loaded from {model_path} (source={model_source})")

    traces_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        traces = run_condition(context, questions, condition, args)
        traces_by_condition[condition] = traces
        write_jsonl(traces, out_dir / f"trace_manifest_cond_{condition}.jsonl")

    scores = {c: condition_score(traces_by_condition[c]) for c in CONDITIONS}
    decision = decide_condition(scores)
    # Section 6.6: cheap tokenizer-only readout-position assertion over EVERY
    # trace (greedy + samples) in each condition, not just the F5-plumbing
    # greedy subset. Always computed (not gated on which condition wins) so
    # the raw condition also gets this correctness signal for free.
    readout_position_checks = {
        c: verify_readout_positions_all_traces(context, traces_by_condition[c]) for c in CONDITIONS
    }
    write_json(
        {"scores": scores, "decision": decision, "readout_position_checks": readout_position_checks},
        out_dir / "prompt_ab_report.json",
    )
    print(f"[4B-2] decision: {decision}")

    f5_report = None
    position_report = None
    smoke_auroc = None
    if decision["winner"] is not None:
        winner = decision["winner"]
        winning_traces = traces_by_condition[winner]
        greedy_traces = [t for t in winning_traces if t["sample_type"] == "greedy"]
        greedy_traces_by_id = {t["example_id"]: t for t in greedy_traces}

        rows, fidelity = readout_features_for_greedy_traces(context, greedy_traces, questions_by_id)
        rows = attach_self_consistency(rows, winning_traces, questions_by_id)
        f5_report = build_f5_plumbing_report(rows, fidelity)
        smoke_auroc = compute_smoke_auroc(rows, greedy_traces_by_id)
        f5_report["smoke_scale_auroc_reference"] = smoke_auroc
        write_json(f5_report, out_dir / "f5_feature_plumbing_report.json")

        position_report = option_position_model_bias_report(questions, scores, traces_by_condition)
        write_json(position_report, out_dir / "option_position_model_bias_report.json")
    else:
        write_json({"status": "skipped", "reason": decision["reason"]}, out_dir / "f5_feature_plumbing_report.json")
        write_json({"status": "skipped", "reason": decision["reason"]}, out_dir / "option_position_model_bias_report.json")

    (out_dir / "review_gate_small_model_smoke.md").write_text(
        render_review_gate(
            args, questions, scores, decision, f5_report, position_report,
            model_source, smoke_auroc, readout_position_checks,
        ),
        encoding="utf-8",
    )
    print(f"[4B-2] complete out={out_dir}")


if __name__ == "__main__":
    main()
