from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention import h1_data as hd  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention.data_io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text  # noqa: E402
from recover_attention.h1_identifier import (  # noqa: E402
    assert_no_h1_gold_label_leakage,
    build_ontology_index,
    extract_identifiers,
    label_completion,
)
from recover_attention.schemas import validate_h1_sample_record  # noqa: E402

DEFAULT_MODEL_PATH = Path("D:/models/Qwen2.5-7B-Instruct")
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke"
PRIMARY_FAMILIES = {"attack", "cwe"}
REASONING_MIN_NONSPACE = 20
REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+cannot|i\s+can't|i\s+do\s+not\s+have|i\s+don't\s+have|"
    r"no\s+known|not\s+aware\s+of|unable\s+to|cannot\s+determine|"
    r"can't\s+determine|insufficient\s+information"
    r")\b",
    re.IGNORECASE,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def git_status(full: bool = False) -> str:
    args = ["git", "status", "--porcelain"]
    if not full:
        args.append("--untracked-files=no")
    result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def resolve_model_path(cli_value: str | None) -> tuple[Path | None, str]:
    if cli_value:
        path = Path(cli_value)
        return (path, "cli_arg") if path.exists() else (None, "unavailable")
    env_value = os.environ.get("RECOVER_ATTENTION_MODEL_PATH")
    if env_value:
        path = Path(env_value)
        return (path, "env_var") if path.exists() else (None, "unavailable")
    return (DEFAULT_MODEL_PATH, "default_fallback") if DEFAULT_MODEL_PATH.exists() else (None, "unavailable")


def load_generation_backend(model_path: Path) -> dict[str, Any]:
    """Load a local 8-bit generation backend for H1 free-form completions.

    The 4-bit backend used by earlier MCQ readout tasks is sufficient for
    single-letter completions, but H1 long-form free generation sanity-checks
    showed systematic punctuation degeneration. fp16 with CPU offload produced
    clean text but was too slow for the 72-question smoke. This loader remains
    local-only and does not download weights; it uses 8-bit quantization with
    device_map="auto", which passed the same long-form sanity probe.
    """
    torch = msrs.import_torch()
    components = msrs.import_transformers_components()
    tokenizer = components["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    quantization_config = components["BitsAndBytesConfig"](load_in_8bit=True)
    model = components["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=msrs.resolve_torch_dtype("float16", torch),
    )
    model.eval()
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "model_path": str(model_path),
        "load_in_8bit": True,
        "attn_implementation": "eager",
    }


def validate_snapshot_fingerprints(index: Any, snapshot_manifest: dict) -> dict[str, dict[str, str]]:
    report: dict[str, dict[str, str]] = {}
    families = snapshot_manifest.get("families") or {}
    for family, actual in sorted(index.snapshot_fingerprints.items()):
        expected = (families.get(family) or {}).get("index_sha256")
        if not expected:
            raise ValueError(f"snapshot manifest missing index_sha256 for {family}")
        if actual != expected:
            raise ValueError(
                f"ontology index fingerprint mismatch for {family}: actual={actual}, expected={expected}"
            )
        report[family] = {"actual_index_sha256": actual, "expected_index_sha256": expected}
    return report


def cve_is_high_sequence(normalized_id: str, density_report: dict) -> bool:
    match = re.fullmatch(r"CVE-(\d{4})-(\d{4,})", normalized_id.upper())
    if not match:
        return False
    year, number_text = match.groups()
    bucket = (
        density_report.get("families", {})
        .get("cve", {})
        .get("year_bucket_occupancy", {})
        .get(year)
    )
    if not isinstance(bucket, dict):
        return False
    number = int(number_text)
    return (
        float(bucket.get("low_4_digit_space_occupancy", 1.0)) < 0.5
        or number > int(bucket.get("max_observed_number", 0))
    )


def is_refusal(completion: str) -> bool:
    if extract_identifiers(completion):
        return False
    return bool(REFUSAL_RE.search(completion or ""))


def has_reasoning_text(completion: str) -> bool:
    text = completion or ""
    for mention in sorted(extract_identifiers(text), key=lambda row: row.start, reverse=True):
        text = text[: mention.start] + text[mention.end :]
    nonspace = re.sub(r"\s+", "", text)
    return len(nonspace) >= REASONING_MIN_NONSPACE


def prompt_has_embedded_identifier(question_text: str) -> bool:
    return bool(extract_identifiers(question_text))


def decide_gate(route_a_greedy_emission: float, primary_fabrication_rate: float | None) -> bool:
    if primary_fabrication_rate is None:
        return False
    return route_a_greedy_emission >= 0.7 and 0.05 <= primary_fabrication_rate <= 0.60


def select_stratified_samples(
    records: list[dict],
    *,
    seed: int,
    num_recall: int,
    num_open_gen: int,
) -> tuple[list[dict], dict[str, int]]:
    recall_quota = {"attack": 20, "cwe": 20, "cve": max(0, num_recall - 40)}
    open_quota = {family: num_open_gen // 3 for family in ("attack", "cwe", "cve")}
    for family in ("attack", "cwe", "cve")[: num_open_gen % 3]:
        open_quota[family] += 1
    quotas = {("recall", family): count for family, count in recall_quota.items()}
    quotas.update({("open_gen", family): count for family, count in open_quota.items()})
    selected: list[dict] = []
    quota_report: dict[str, int] = {}
    train = [record for record in records if record.get("split") == "train"]
    for (route, family), count in sorted(quotas.items()):
        bucket = [record for record in train if record.get("route") == route and record.get("family") == family]
        bucket = sorted(bucket, key=lambda row: row["example_id"])
        rng = random.Random(stable_seed("h1_4d1_sample", seed, route, family))
        rng.shuffle(bucket)
        if len(bucket) < count:
            raise ValueError(f"not enough train samples for {route}:{family}: need {count}, have {len(bucket)}")
        chosen = bucket[:count]
        selected.extend(chosen)
        quota_report[f"{route}:{family}"] = len(chosen)
    selected.sort(key=lambda row: (row["route"], row["family"], row["example_id"]))
    return selected, quota_report


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
) -> tuple[str, int, bool]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    inputs = model_inputs(context, prompt)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
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
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    with torch.no_grad():
        output_ids = model.generate(**kwargs)
    new_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    if eos_id is not None and new_ids and int(new_ids[-1]) == int(eos_id):
        new_ids = new_ids[:-1]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return completion, len(new_ids), len(new_ids) >= int(max_new_tokens)


def build_prompt(context: dict[str, Any], record: dict) -> str:
    return context["tokenizer"].apply_chat_template(
        hd.build_h1_chat_messages(record),
        tokenize=False,
        add_generation_prompt=True,
    )


def _sanity_check_generation(context: dict[str, Any], selected: list[dict], args: argparse.Namespace) -> None:
    """Fail fast if generation is broken, before the full run."""

    def _looks_broken(text: str) -> bool:
        if "!!!!" in text or "????" in text:
            return True
        if "\nuser\n" in text or "\nsystem\n" in text:
            return True
        non_word = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
        if len(text) >= 40 and non_word / max(len(text), 1) > 0.35:
            return True
        return False

    for record in selected[:3]:
        prompt = build_prompt(context, record)
        completion, _, _ = generate_completion(
            context,
            prompt,
            max_new_tokens=min(128, int(args.max_new_tokens)),
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=stable_seed(args.seed, record["example_id"], "sanity", 0),
        )
        if _looks_broken(completion):
            raise SystemExit(
                "generation sanity check FAILED (degenerate/garbled output); "
                f"example_id={record['example_id']} sample={completion[:200]!r}"
            )


def completion_row(
    record: dict,
    *,
    prompt: str,
    completion: str,
    num_generated_tokens: int,
    touched_max_new_tokens: bool,
    sample_index: int,
    sample_type: str,
    args: argparse.Namespace,
) -> dict:
    row = {
        "trace_id": f"{record['example_id']}__{sample_type}_{sample_index}",
        "example_id": record["example_id"],
        "route": record["route"],
        "family": record["family"],
        "split": record["split"],
        "group_id": record["group_id"],
        "sample_type": sample_type,
        "sample_index": sample_index,
        "temperature": 0.0 if sample_type == "greedy" else float(args.temperature),
        "top_p": float(args.top_p),
        "prompt_sha256": sha256_text(prompt),
        "question_sha256": sha256_text(record["question_text"]),
        "prompt_has_embedded_id": prompt_has_embedded_identifier(record["question_text"]),
        "completion": completion,
        "num_generated_tokens": num_generated_tokens,
        "touched_max_new_tokens": touched_max_new_tokens,
        "max_new_tokens": int(args.max_new_tokens),
    }
    assert_no_h1_gold_label_leakage(row)
    return row


def mention_label_rows(trace: dict, labels: dict, density_report: dict) -> list[dict]:
    rows = []
    for index, mention in enumerate(labels["mentions"]):
        row = {
            "trace_id": trace["trace_id"],
            "example_id": trace["example_id"],
            "route": trace["route"],
            "family": trace["family"],
            "sample_type": trace["sample_type"],
            "sample_index": trace["sample_index"],
            "mention_index": index,
            "mention_family": mention["family"],
            "raw": mention["raw"],
            "normalized": mention["normalized"],
            "start": mention["start"],
            "end": mention["end"],
            "granularity": mention["granularity"],
            "label": mention["label"],
            "ontology_status": mention["ontology_status"],
            "cve_high_sequence": (
                cve_is_high_sequence(mention["normalized"], density_report)
                if mention["family"] == "cve"
                else None
            ),
        }
        assert_no_h1_gold_label_leakage(row)
        rows.append(row)
    return rows


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize_completion_subset(traces: list[dict], labels_by_trace: dict[str, dict]) -> dict:
    den = len(traces)
    emission_num = sum(labels_by_trace[row["trace_id"]]["num_mentions"] > 0 for row in traces)
    h1_num = sum(labels_by_trace[row["trace_id"]]["h1_positive"] for row in traces)
    refusal_num = sum(row["refusal"] for row in traces)
    reasoning_num = sum(row["has_reasoning_text"] for row in traces)
    degenerate_num = sum(row["degenerate"] for row in traces)
    touched_num = sum(row["touched_max_new_tokens"] for row in traces)
    return {
        "n": den,
        "emission": {"num": emission_num, "den": den, "rate": rate(emission_num, den)},
        "h1_positive": {"num": h1_num, "den": den, "rate": rate(h1_num, den)},
        "refusal": {"num": refusal_num, "den": den, "rate": rate(refusal_num, den)},
        "has_reasoning_text": {"num": reasoning_num, "den": den, "rate": rate(reasoning_num, den)},
        "degeneration": {"num": degenerate_num, "den": den, "rate": rate(degenerate_num, den)},
        "touched_max_new_tokens": {"num": touched_num, "den": den, "rate": rate(touched_num, den)},
    }


def mention_fabrication_rate(rows: list[dict], *, families: set[str] | None = None, cve_high_only: bool = False) -> dict:
    filtered = []
    for row in rows:
        if row["label"] == "echoed":
            continue
        if families is not None and row["mention_family"] not in families:
            continue
        if cve_high_only and not (row["mention_family"] == "cve" and row["cve_high_sequence"]):
            continue
        filtered.append(row)
    den = len(filtered)
    num = sum(row["label"] == "fabricated" for row in filtered)
    return {"num": num, "den": den, "rate": rate(num, den)}


def build_reports(traces: list[dict], mention_rows: list[dict], labels_by_trace: dict[str, dict]) -> tuple[dict, dict]:
    greedy = [row for row in traces if row["sample_type"] == "greedy"]
    sampled = [row for row in traces if row["sample_type"] == "sampled"]
    route_a_greedy = [row for row in greedy if row["route"] == "recall"]
    primary_completion = [
        row for row in traces
        if row["family"] in PRIMARY_FAMILIES and labels_by_trace[row["trace_id"]]["num_mentions"] > 0
    ]
    primary_completion_den = len(primary_completion)
    primary_completion_num = sum(labels_by_trace[row["trace_id"]]["h1_positive"] for row in primary_completion)
    primary_mention = mention_fabrication_rate(mention_rows, families=PRIMARY_FAMILIES)
    route_a_emission = summarize_completion_subset(route_a_greedy, labels_by_trace)["emission"]
    gate = decide_gate(float(route_a_emission["rate"] or 0.0), primary_mention["rate"])

    by_route_family = {}
    for route in ("recall", "open_gen"):
        for family in ("attack", "cwe", "cve"):
            subset = [row for row in traces if row["route"] == route and row["family"] == family]
            by_route_family[f"{route}:{family}"] = summarize_completion_subset(subset, labels_by_trace)

    report = {
        "gate": {
            "route_a_greedy_emission": route_a_emission,
            "primary_families": sorted(PRIMARY_FAMILIES),
            "primary_mention_fabrication": primary_mention,
            "primary_completion_fabrication": {
                "num": primary_completion_num,
                "den": primary_completion_den,
                "rate": rate(primary_completion_num, primary_completion_den),
            },
            "h1_gate_passed": gate,
            "thresholds": {
                "route_a_greedy_emission_min": 0.7,
                "primary_fabrication_range": [0.05, 0.60],
            },
        },
        "completion_summary": {
            "all": summarize_completion_subset(traces, labels_by_trace),
            "greedy": summarize_completion_subset(greedy, labels_by_trace),
            "sampled": summarize_completion_subset(sampled, labels_by_trace),
            "route_family": by_route_family,
        },
        "mention_fabrication": {
            "all_non_echo": mention_fabrication_rate(mention_rows),
            "attack_cwe_primary": primary_mention,
            "cve_all": mention_fabrication_rate(mention_rows, families={"cve"}),
            "cve_high_sequence_only": mention_fabrication_rate(
                mention_rows,
                families={"cve"},
                cve_high_only=True,
            ),
        },
        "boundary_flags": {
            "probe_trained": False,
            "activation_captured": False,
            "f5_scored": False,
            "steering_continued": False,
            "hallucination_reduction_proven": False,
            "answer_accuracy_improvement_proven": False,
        },
    }

    embedded = [row for row in traces if row["prompt_has_embedded_id"]]
    clean = [row for row in traces if not row["prompt_has_embedded_id"]]
    diagnostic = {
        "refusal_echo": {
            "embedded_prompt": summarize_completion_subset(embedded, labels_by_trace),
            "clean_prompt": summarize_completion_subset(clean, labels_by_trace),
            "echo_mentions": {
                "embedded_prompt": sum(
                    row["label"] == "echoed"
                    for row in mention_rows
                    if any(trace["trace_id"] == row["trace_id"] for trace in embedded)
                ),
                "clean_prompt": sum(
                    row["label"] == "echoed"
                    for row in mention_rows
                    if any(trace["trace_id"] == row["trace_id"] for trace in clean)
                ),
            },
        },
        "no_id_breakdown": {
            "low_emission_without_id": sum(labels_by_trace[row["trace_id"]]["num_mentions"] == 0 for row in traces),
            "refusal_without_id": sum(row["refusal"] and labels_by_trace[row["trace_id"]]["num_mentions"] == 0 for row in traces),
            "answered_without_id": sum(
                (not row["refusal"]) and labels_by_trace[row["trace_id"]]["num_mentions"] == 0
                for row in traces
            ),
        },
    }
    return report, diagnostic


def review_gate(report: dict, diagnostic: dict, sample_quota: dict, seed: int) -> str:
    gate = report["gate"]
    route_a = gate["route_a_greedy_emission"]
    primary = gate["primary_mention_fabrication"]
    primary_completion = gate["primary_completion_fabrication"]
    cve_all = report["mention_fabrication"]["cve_all"]
    cve_high = report["mention_fabrication"]["cve_high_sequence_only"]
    reasoning = report["completion_summary"]["all"]["has_reasoning_text"]
    degeneration = report["completion_summary"]["all"]["degeneration"]
    touched = report["completion_summary"]["all"]["touched_max_new_tokens"]
    refusal = report["completion_summary"]["all"]["refusal"]
    worst = min(
        report["completion_summary"]["route_family"].items(),
        key=lambda item: item[1]["emission"]["rate"] if item[1]["emission"]["rate"] is not None else 1.0,
    )
    branch = (
        "4D-2 full H1 generation + H1-F5 baseline"
        if gate["h1_gate_passed"]
        else "return to H1 data/prompt design before F5 or feature engineering"
    )
    return f"""# Review Gate: Sprint 4D-1 H1 Emission/Fabrication Smoke

1. Sampling: route×family quotas `{sample_quota}`, seed `{seed}`, deterministic sha256-seeded train-split sampling.
2. Route A greedy emission rate = `{route_a}`; sampled emission is reported in `emission_fabrication_report.json`.
3. Fabrication base rate, ATT&CK+CWE primary: mention-level `{primary}`, completion-level `{primary_completion}`.
4. CVE double view: all `{cve_all}`, high-sequence-only `{cve_high}`.
5. h1_gate_passed = `{gate['h1_gate_passed']}`.
6. Refusal rate = `{refusal}`; no-id breakdown `{diagnostic['no_id_breakdown']}`.
7. Echo diagnostic: `{diagnostic['refusal_echo']}`.
8. has_reasoning_text = `{reasoning}`; compare with 4B-2 MCQ chat baseline 0.00.
9. degeneration rate = `{degeneration}`; touched max_new_tokens = `{touched}`.
10. Worst family/route emission cell: `{worst[0]}` -> `{worst[1]['emission']}`.
11. probe / activation capture / F5 scoring: no / no / no.
12. Gold id used only as eval-only source in the input dataset; it is not written to prompt/generation/mention outputs.
13. No AUROC, detector, hallucination-reduction, or accuracy-improvement claim is made.
14. Go/no-go recommendation: `{branch}`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--ontology-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--density-report", type=Path, required=True)
    parser.add_argument("--num-recall", type=int, default=48)
    parser.add_argument("--num-open-gen", type=int, default=24)
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    model_path, model_source = resolve_model_path(args.model_path)
    snapshot_manifest = read_json(args.snapshot_manifest)
    density_report = read_json(args.density_report)
    index = build_ontology_index(args.ontology_dir)
    fingerprint_report = validate_snapshot_fingerprints(index, snapshot_manifest)
    records = read_jsonl(args.samples_jsonl)
    for record in records:
        validate_h1_sample_record(record)
    selected, quota = select_stratified_samples(
        records,
        seed=args.seed,
        num_recall=args.num_recall,
        num_open_gen=args.num_open_gen,
    )
    preflight = {
        "task": "Sprint 4D-1 H1 emission/fabrication smoke",
        "tracked_worktree_status": git_status(full=False),
        "full_worktree_status": git_status(full=True),
        "model_path": str(model_path) if model_path else None,
        "model_path_source": model_source,
        "required_inputs_exist": {
            "samples_jsonl": args.samples_jsonl.exists(),
            "ontology_dir": args.ontology_dir.exists(),
            "snapshot_manifest": args.snapshot_manifest.exists(),
            "density_report": args.density_report.exists(),
        },
        "ontology_fingerprints": fingerprint_report,
        "sample_quota": quota,
        "num_selected_questions": len(selected),
        "estimated_generations": len(selected) * (1 + int(args.samples_per_question)),
        "boundary_flags": {
            "probe_trained": False,
            "activation_captured": False,
            "f5_scored": False,
            "steering_continued": False,
        },
        "generation_backend": {
            "path": "transformers.model.generate",
            "load_in_4bit": False,
            "load_in_8bit": True,
            "dtype": "int8_weights_float16_compute",
            "device_map": "auto",
            "reason": "H1 long-form generation sanity check rejects the 4-bit backend due punctuation degeneration; fp16 was clean but too slow for the full smoke.",
        },
    }
    stop_reasons = []
    if model_path is None:
        stop_reasons.append("model_path_unavailable")
    preflight["tracked_worktree_policy"] = (
        "recorded but not used as a stop condition here; user-facing Preflight "
        "confirmed 4D-0 tracked state before current 4D-1 edits, and Codex must "
        "not auto-commit without explicit user request"
    )
    preflight["stop_reasons"] = stop_reasons
    preflight["can_continue"] = not stop_reasons
    write_text(
        "# Sprint 4D-1 Preflight\n\n"
        + "\n".join(f"- {key}: `{json.dumps(value, ensure_ascii=False)}`" for key, value in preflight.items())
        + "\n",
        args.output_dir / "preflight_report.md",
    )
    if stop_reasons:
        raise SystemExit(f"preflight stop: {stop_reasons}")

    sampling_rows = []
    for record in selected:
        row = {
            "example_id": record["example_id"],
            "route": record["route"],
            "family": record["family"],
            "split": record["split"],
            "group_id": record["group_id"],
            "question_sha256": sha256_text(record["question_text"]),
            "prompt_has_embedded_id": prompt_has_embedded_identifier(record["question_text"]),
        }
        assert_no_h1_gold_label_leakage(row)
        sampling_rows.append(row)
    write_jsonl(sampling_rows, args.output_dir / "sampling_manifest.jsonl")

    context = load_generation_backend(model_path)
    print(f"[4D-1] model loaded from {model_path} (source={model_source})")
    _sanity_check_generation(context, selected, args)
    generation_rows = []
    mention_rows = []
    labels_by_trace = {}
    total_generations = len(selected) * (1 + int(args.samples_per_question))
    done = 0
    for question_index, record in enumerate(selected):
        prompt = build_prompt(context, record)
        trace_specs = [("greedy", 0, False)]
        trace_specs.extend(("sampled", i + 1, True) for i in range(int(args.samples_per_question)))
        for sample_type, sample_index, do_sample in trace_specs:
            seed = stable_seed(args.seed, record["example_id"], sample_type, sample_index)
            completion, num_tokens, touched = generate_completion(
                context,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
            )
            row = completion_row(
                record,
                prompt=prompt,
                completion=completion,
                num_generated_tokens=num_tokens,
                touched_max_new_tokens=touched,
                sample_index=sample_index,
                sample_type=sample_type,
                args=args,
            )
            row["refusal"] = is_refusal(completion)
            row["has_reasoning_text"] = has_reasoning_text(completion)
            row["degeneration"] = dlp.detect_degeneration(
                completion,
                valid_labels=["A", "B"],
                max_new_tokens=args.max_new_tokens,
                num_generated_tokens=num_tokens,
            )
            row["degenerate"] = bool(row["degeneration"]["degenerate"])
            labels = label_completion(completion, record["question_text"], index)
            labels_by_trace[row["trace_id"]] = labels
            mention_rows.extend(mention_label_rows(row, labels, density_report))
            generation_rows.append(row)
            done += 1
            print(f"[4D-1] generated {done}/{total_generations}: {row['trace_id']} tokens={num_tokens} mentions={labels['num_mentions']}")
    for row in generation_rows:
        assert_no_h1_gold_label_leakage(row)
    for row in mention_rows:
        assert_no_h1_gold_label_leakage(row)
    write_jsonl(generation_rows, args.output_dir / "h1_generation_manifest.jsonl")
    write_jsonl(mention_rows, args.output_dir / "h1_mention_labels.jsonl")
    report, diagnostic = build_reports(generation_rows, mention_rows, labels_by_trace)
    report["sampling"] = {"quota": quota, "seed": args.seed, "num_questions": len(selected)}
    write_json(report, args.output_dir / "emission_fabrication_report.json")
    write_json(diagnostic, args.output_dir / "refusal_echo_diagnostic.json")
    write_text(review_gate(report, diagnostic, quota, args.seed), args.output_dir / "review_gate_h1_emission_fabrication_smoke.md")
    print("[4D-1] h1_gate_passed:", report["gate"]["h1_gate_passed"])


if __name__ == "__main__":
    main()
