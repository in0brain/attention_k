"""Sprint 3C-0: correct-vs-wrong activation patching at reasoning-step positions.

This diagnostic samples small GSM8K traces, pairs correct and wrong completions
for the same question, and teacher-forces residual replacement from the correct
trace into the wrong trace at matched reasoning-step token positions.

Boundary: no training, no LoRA, no full 3C/2000 run, no hallucination or answer
accuracy claims. The metric is a single-forward answer-directed proxy.
"""

from __future__ import annotations

import argparse
import json
import math
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

from recover_attention import activation_patching as ap  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
DEFAULT_GSM = PROJECT_ROOT / "data/raw/gsm8k_train_normalized.jsonl"
DEFAULT_FEATURES = PROJECT_ROOT / "outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl"
DEFAULT_SCORES = PROJECT_ROOT / "outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
TASK_CARD = PROJECT_ROOT / "docs/codex_tasks/sprint_3C_0_correct_wrong_activation_patching_reasoning_steps.md"

PREVIOUS_3A1_ARTIFACTS = [
    "outputs/logs/sprint_3A_1_controlled_attention_guidance_500/oracle_sanity_diagnostic_report.json",
    "outputs/logs/sprint_3A_1_controlled_attention_guidance_500/answer_position_output_shift_report_500.json",
    "outputs/logs/sprint_3A_1_controlled_attention_guidance_500/harm_rate_report_500.json",
    "outputs/logs/sprint_3A_1_controlled_attention_guidance_500/baseline_comparison_report_500.json",
    "outputs/logs/sprint_3A_1_controlled_attention_guidance_500/review_gate_controlled_attention_guidance_500.md",
]
PREVIOUS_3B0_ARTIFACTS = [
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/representation_intervention_config.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/representation_intervention_fidelity_report.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/gold_logprob_delta_report.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/oracle_vs_random_diagnostic_report.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/selector_comparison_report.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/harm_rate_report.json",
    "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/review_gate_representation_level_oracle_diagnostic.md",
]

PATCH_CONDITIONS = [
    "no_patch",
    "correct_to_wrong",
    "correct_activation_patch",
    "same_trace_random_position_patch",
    "random_donor_patch",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    if args.overwrite:
        clear_output_dir(out_dir)
    write_preflight_report(out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-0 correct-vs-wrong activation patching")
    parser.add_argument("--primary-questions", type=int, default=100)
    parser.add_argument("--samples-per-question", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument(
        "--position-types",
        nargs="+",
        default=[
            "generated_operator_token",
            "generated_intermediate_number_token",
            "generated_final_answer_number",
            "final_answer_position",
        ],
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source-questions", default=str(DEFAULT_GSM))
    parser.add_argument("--feature-matrix", default=str(DEFAULT_FEATURES))
    parser.add_argument("--answer-position-score-matrix", default=str(DEFAULT_SCORES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=33030)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clear_output_dir(out_dir: Path) -> None:
    for path in out_dir.glob("*"):
        if path.is_file():
            path.unlink()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    for position_type in args.position_types:
        if position_type not in ap.SUPPORTED_POSITION_TYPES:
            raise ValueError(f"unsupported --position-types value: {position_type}")
    if args.alpha < 0 or args.alpha > 1:
        raise ValueError("--alpha must be in [0, 1]")

    question_rows = read_jsonl(args.source_questions)
    gold_by_qid = {
        str(row["question_id"]): str(row.get("answer", "")).strip()
        for row in question_rows
    }
    eligible_questions = select_eligible_questions(
        feature_matrix_path=args.feature_matrix,
        score_matrix_path=args.answer_position_score_matrix,
        primary_questions=args.primary_questions,
    )
    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    print(
        "[3C-0] "
        f"eligible_questions={len(eligible_questions)} samples_per_question={args.samples_per_question} "
        f"layers={args.layers} positions={args.position_types} model loaded"
    )

    trace_rows = sample_traces(
        context=context,
        questions=eligible_questions,
        gold_by_qid=gold_by_qid,
        samples_per_question=args.samples_per_question,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        report_every=args.report_every,
    )
    write_jsonl(trace_rows, out_dir / "trace_sampling_manifest.jsonl")

    pair_rows = build_correct_wrong_pairs(trace_rows, args.position_types, args.seed)
    write_jsonl(pair_rows, out_dir / "correct_wrong_pair_manifest.jsonl")
    if not pair_rows:
        write_empty_reports(args, out_dir, trace_rows)
        print("[3C-0] no same-question correct/wrong pairs found; wrote incomplete-evidence reports")
        return

    pair_state = prepare_pair_state(
        context=context,
        pairs=pair_rows,
        layers=args.layers,
        position_types=args.position_types,
        seed=args.seed,
        report_every=args.report_every,
    )
    position_report = build_position_report(pair_rows, pair_state, args.position_types)
    write_json(position_report, out_dir / "reasoning_step_position_report.json")
    config = build_config(args, len(eligible_questions), len(trace_rows), len(pair_rows))
    write_json(config, out_dir / "activation_patching_config.json")

    forward_rows = run_patching_forwards(
        context=context,
        tokenizer=tokenizer,
        pair_state=pair_state,
        layers=args.layers,
        position_types=args.position_types,
        alpha=args.alpha,
        report_every=args.report_every,
    )
    write_jsonl(forward_rows, out_dir / "activation_patching_forward_manifest.jsonl")
    reports = build_reports(forward_rows, pair_rows, position_report)
    for filename, payload in reports.items():
        if filename.endswith(".json"):
            write_json(payload, out_dir / filename)
        elif filename.endswith(".jsonl"):
            write_jsonl(payload, out_dir / filename)
        else:
            (out_dir / filename).write_text(str(payload), encoding="utf-8")
    print(
        "[3C-0] complete: "
        f"traces={len(trace_rows)} pairs={len(pair_rows)} patch_rows={len(forward_rows)} "
        f"output_dir={out_dir}"
    )


def select_eligible_questions(
    *,
    feature_matrix_path: str | Path,
    score_matrix_path: str | Path,
    primary_questions: int,
) -> list[dict[str, Any]]:
    records = abs_mod.load_steering_source_records(
        feature_matrix_path=feature_matrix_path,
        answer_position_score_matrix_path=score_matrix_path,
    )
    subset = abs_mod.build_steering_subset(records, primary_n=primary_questions)
    return subset[:primary_questions]


def sample_traces(
    *,
    context: dict[str, Any],
    questions: list[dict[str, Any]],
    gold_by_qid: dict[str, str],
    samples_per_question: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    report_every: int,
) -> list[dict[str, Any]]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for q_index, item in enumerate(questions, start=1):
        qid = str(item["source_question_id"])
        question = str(item["question"])
        gold_answer = gold_by_qid.get(qid, "")
        prompt = ap.build_trace_prompt(question)
        encoded = tokenizer(prompt, return_tensors="pt")
        target_device = infer_device(context)
        model_inputs = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        input_len = int(model_inputs["input_ids"].shape[-1])
        for sample_idx in range(samples_per_question):
            local_seed = seed + (q_index * 1000) + sample_idx
            torch.manual_seed(local_seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    renormalize_logits=True,
                    remove_invalid_values=True,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            completion_ids = generated[0, input_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            parsed = ap.classify_trace(completion, gold_answer)
            full_text = prompt + completion
            rows.append(
                {
                    "source_question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "sample_id": f"{qid}__sample_{sample_idx:02d}",
                    "sample_index": sample_idx,
                    "prompt": prompt,
                    "completion": completion,
                    "trace_text": full_text,
                    "extracted_answer": parsed["answer"],
                    "normalized_extracted_answer": parsed["normalized_answer"],
                    "normalized_gold_answer": parsed["normalized_gold_answer"],
                    "parse_method": parsed["parse_method"],
                    "is_correct": parsed["is_correct"],
                    "generation": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "seed": local_seed,
                    },
                }
            )
        if report_every and q_index % report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-0] sampled_questions={q_index}/{len(questions)} rate={q_index/elapsed:.3f}/s")
    return rows


def build_correct_wrong_pairs(
    trace_rows: list[dict[str, Any]],
    position_types: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[str(row["source_question_id"])].append(row)
    pairs = []
    for qid, rows in grouped.items():
        correct = [row for row in rows if row["is_correct"]]
        wrong = [row for row in rows if not row["is_correct"] and row["normalized_extracted_answer"] is not None]
        if not correct or not wrong:
            continue
        donor = correct[0]
        recipient = wrong[0]
        pair_id = f"{qid}__cw_pair_00"
        pairs.append(
            {
                "pair_id": pair_id,
                "source_question_id": qid,
                "question": donor["question"],
                "gold_answer": donor["gold_answer"],
                "correct_sample_id": donor["sample_id"],
                "wrong_sample_id": recipient["sample_id"],
                "correct_answer": donor["extracted_answer"],
                "wrong_answer": recipient["extracted_answer"],
                "correct_trace_text": donor["trace_text"],
                "wrong_trace_text": recipient["trace_text"],
                "correct_completion": donor["completion"],
                "wrong_completion": recipient["completion"],
                "position_types_requested": list(position_types),
                "pairing_policy": "first_correct_first_wrong_same_question",
                "seed": seed,
            }
        )
    return pairs


def prepare_pair_state(
    *,
    context: dict[str, Any],
    pairs: list[dict[str, Any]],
    layers: list[int],
    position_types: list[str],
    seed: int,
    report_every: int,
) -> list[dict[str, Any]]:
    state = []
    t0 = time.time()
    for index, pair in enumerate(pairs, start=1):
        correct_base = ap.forward_with_hidden(context, pair["correct_trace_text"], layers)
        wrong_base = ap.forward_with_hidden(context, pair["wrong_trace_text"], layers)
        correct_prompt = ap.build_trace_prompt(pair["question"])
        wrong_prompt = correct_prompt
        correct_positions = ap.extract_reasoning_step_positions(
            offsets=correct_base["offsets"],
            prompt_text=correct_prompt,
            completion_text=pair["correct_completion"],
            seed_key=f"{seed}:{pair['pair_id']}:correct",
        )
        wrong_positions = ap.extract_reasoning_step_positions(
            offsets=wrong_base["offsets"],
            prompt_text=wrong_prompt,
            completion_text=pair["wrong_completion"],
            seed_key=f"{seed}:{pair['pair_id']}:wrong",
        )
        matches = {
            position_type: ap.match_role_positions(correct_positions, wrong_positions, position_type)
            for position_type in position_types
        }
        state.append(
            {
                "pair": pair,
                "correct_base": correct_base,
                "wrong_base": wrong_base,
                "correct_positions": correct_positions,
                "wrong_positions": wrong_positions,
                "matches": matches,
            }
        )
        if report_every and index % report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-0] prepared_pairs={index}/{len(pairs)} rate={index/elapsed:.3f}/s")
    return state


def run_patching_forwards(
    *,
    context: dict[str, Any],
    tokenizer: Any,
    pair_state: list[dict[str, Any]],
    layers: list[int],
    position_types: list[str],
    alpha: float,
    report_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    for idx, item in enumerate(pair_state):
        pair = item["pair"]
        random_item = pair_state[(idx + 1) % len(pair_state)] if len(pair_state) > 1 else item
        wrong_base = item["wrong_base"]
        correct_base = item["correct_base"]
        wrong_gold_base, wrong_wrong_base = base_logprobs(tokenizer, wrong_base["logits"], pair)
        correct_gold_base, correct_wrong_base = base_logprobs(tokenizer, correct_base["logits"], pair)
        for position_type in position_types:
            match = item["matches"][position_type]
            if not match["matched"]:
                continue
            donor_pos = int(match["donor_position"])
            recipient_pos = int(match["recipient_position"])
            random_donor_pos = choose_random_donor_position(item["correct_positions"], donor_pos)
            random_donor_item = random_item
            random_match = random_donor_item["matches"].get(position_type) or {}
            cross_donor_pos = random_match.get("donor_position")
            if cross_donor_pos is None:
                cross_donor_pos = donor_pos
                random_donor_item = item
            for layer in layers:
                rows.append(
                    no_patch_row(
                        pair=pair,
                        position_type=position_type,
                        layer=layer,
                        base_gold_lp=wrong_gold_base,
                        base_wrong_lp=wrong_wrong_base,
                    )
                )
                for condition in [
                    "correct_to_wrong",
                    "correct_activation_patch",
                    "same_trace_random_position_patch",
                    "random_donor_patch",
                ]:
                    if condition == "correct_to_wrong":
                        donor_hidden = item["correct_base"]["hidden_by_layer"]
                        target_text = pair["wrong_trace_text"]
                        target_base = item["wrong_base"]
                        target_pos = recipient_pos
                        donor_position = donor_pos
                        base_gold_lp = wrong_gold_base
                        base_wrong_lp = wrong_wrong_base
                    elif condition == "correct_activation_patch":
                        donor_hidden = item["correct_base"]["hidden_by_layer"]
                        target_text = pair["correct_trace_text"]
                        target_base = item["correct_base"]
                        target_pos = donor_pos
                        donor_position = donor_pos
                        base_gold_lp = correct_gold_base
                        base_wrong_lp = correct_wrong_base
                    elif condition == "same_trace_random_position_patch":
                        donor_hidden = item["correct_base"]["hidden_by_layer"]
                        target_text = pair["wrong_trace_text"]
                        target_base = item["wrong_base"]
                        target_pos = recipient_pos
                        donor_position = random_donor_pos
                        base_gold_lp = wrong_gold_base
                        base_wrong_lp = wrong_wrong_base
                    else:
                        donor_hidden = random_donor_item["correct_base"]["hidden_by_layer"]
                        target_text = pair["wrong_trace_text"]
                        target_base = item["wrong_base"]
                        target_pos = recipient_pos
                        donor_position = int(cross_donor_pos)
                        base_gold_lp = wrong_gold_base
                        base_wrong_lp = wrong_wrong_base

                    donor_vec = ap.layer_patch_vector(
                        donor_hidden,
                        layer=layer,
                        donor_position=int(donor_position),
                    )
                    if donor_vec is None:
                        continue
                    trace = {"registered": False, "removed": False, "triggered_layers": [], "patch_records": []}
                    patched = ap.patched_forward(
                        context,
                        target_text,
                        patch_vectors_by_layer={int(layer): donor_vec},
                        target_position=int(target_pos),
                        alpha=alpha,
                        trace=trace,
                    )
                    rows.append(
                        forward_row(
                            tokenizer=tokenizer,
                            pair=pair,
                            condition=condition,
                            position_type=position_type,
                            layer=layer,
                            donor_position=int(donor_position),
                            recipient_position=int(target_pos),
                            base_logits=target_base["logits"],
                            patched_logits=patched["logits"],
                            base_gold_lp=base_gold_lp,
                            base_wrong_lp=base_wrong_lp,
                            trace=trace,
                            alpha=alpha,
                        )
                    )
                    done += 1
                    if report_every and done % report_every == 0:
                        elapsed = max(time.time() - t0, 1e-9)
                        print(f"[3C-0] patched_forwards={done} rate={done/elapsed:.3f}/s")
    return rows


def choose_random_donor_position(positions: dict[str, list[int]], fallback: int) -> int:
    candidates = []
    for name, values in positions.items():
        if name == "final_answer_position":
            continue
        candidates.extend(values)
    unique = sorted({int(v) for v in candidates if int(v) != int(fallback)})
    return unique[0] if unique else int(fallback)


def base_logprobs(tokenizer: Any, logits: Any, pair: dict[str, Any]) -> tuple[float, float]:
    gold_tok = ap.first_token_id(tokenizer, pair["gold_answer"])
    wrong_tok = ap.first_token_id(tokenizer, pair["wrong_answer"])
    return ap.logprob_of(logits, gold_tok), ap.logprob_of(logits, wrong_tok)


def no_patch_row(
    *,
    pair: dict[str, Any],
    position_type: str,
    layer: int,
    base_gold_lp: float,
    base_wrong_lp: float,
) -> dict[str, Any]:
    return {
        "backend": ap.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "patch_condition": "no_patch",
        "patch_type": "none",
        "position_type": position_type,
        "layer": int(layer),
        "donor_position": None,
        "recipient_position": None,
        "gold_first_token_logprob_delta": 0.0,
        "wrong_first_token_logprob_delta": 0.0,
        "clean_direction_score": 0.0,
        "base_gold_first_token_logprob": base_gold_lp,
        "base_wrong_first_token_logprob": base_wrong_lp,
        "patched_gold_first_token_logprob": base_gold_lp,
        "patched_wrong_first_token_logprob": base_wrong_lp,
        "output_shift": {
            "steer_output_js": 0.0,
            "steer_top1_changed": 0.0,
            "steer_entropy_delta": 0.0,
            "steer_margin_delta": 0.0,
        },
        "harm": False,
        "hook": {
            "hook_registered": False,
            "hook_triggered_layer": None,
            "hook_removed": True,
            "patch_delta_norm": 0.0,
            "recipient_hidden_changed_norm": 0.0,
            "non_target_position_contamination_check": True,
        },
    }


def forward_row(
    *,
    tokenizer: Any,
    pair: dict[str, Any],
    condition: str,
    position_type: str,
    layer: int,
    donor_position: int,
    recipient_position: int,
    base_logits: Any,
    patched_logits: Any,
    base_gold_lp: float,
    base_wrong_lp: float,
    trace: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    shift = abs_mod.compute_answer_position_output_shift(base_logits, patched_logits)
    gold_tok = ap.first_token_id(tokenizer, pair["gold_answer"])
    wrong_tok = ap.first_token_id(tokenizer, pair["wrong_answer"])
    patched_gold_lp = ap.logprob_of(patched_logits, gold_tok)
    patched_wrong_lp = ap.logprob_of(patched_logits, wrong_tok)
    gold_delta = patched_gold_lp - base_gold_lp if finite_pair(patched_gold_lp, base_gold_lp) else None
    wrong_delta = patched_wrong_lp - base_wrong_lp if finite_pair(patched_wrong_lp, base_wrong_lp) else None
    clean_direction = gold_delta - wrong_delta if gold_delta is not None and wrong_delta is not None else None
    record = (trace.get("patch_records") or [{}])[-1]
    return {
        "backend": ap.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "patch_condition": condition,
        "patch_type": "residual_replace",
        "alpha": alpha,
        "position_type": position_type,
        "layer": int(layer),
        "donor_position": int(donor_position),
        "recipient_position": int(recipient_position),
        "gold_answer": pair["gold_answer"],
        "wrong_answer": pair["wrong_answer"],
        "gold_first_token_logprob_delta": gold_delta,
        "wrong_first_token_logprob_delta": wrong_delta,
        "clean_direction_score": clean_direction,
        "base_gold_first_token_logprob": base_gold_lp,
        "base_wrong_first_token_logprob": base_wrong_lp,
        "patched_gold_first_token_logprob": patched_gold_lp,
        "patched_wrong_first_token_logprob": patched_wrong_lp,
        "output_shift": shift,
        "harm": ap.compute_harm(shift),
        "hook": {
            "hook_registered": bool(trace.get("registered")),
            "hook_triggered_layer": int(layer) if int(layer) in set(trace.get("triggered_layers") or []) else None,
            "hook_removed": bool(trace.get("removed")),
            "patch_delta_norm": record.get("patch_delta_norm"),
            "recipient_hidden_changed_norm": record.get("recipient_hidden_changed_norm"),
            "non_target_position_contamination_check": record.get("non_target_position_contamination_check"),
            "max_non_target_position_delta": record.get("max_non_target_position_delta"),
        },
    }


def finite_pair(a: float, b: float) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b))


def build_position_report(
    pairs: list[dict[str, Any]],
    pair_state: list[dict[str, Any]],
    position_types: list[str],
) -> dict[str, Any]:
    match_counts = Counter()
    donor_counts = defaultdict(list)
    recipient_counts = defaultdict(list)
    per_pair = []
    for item in pair_state:
        row = {"pair_id": item["pair"]["pair_id"], "positions": {}}
        for position_type in position_types:
            donor_n = len(item["correct_positions"].get(position_type) or [])
            rec_n = len(item["wrong_positions"].get(position_type) or [])
            donor_counts[position_type].append(donor_n)
            recipient_counts[position_type].append(rec_n)
            matched = bool(item["matches"].get(position_type, {}).get("matched"))
            match_counts[(position_type, matched)] += 1
            row["positions"][position_type] = {
                "donor_count": donor_n,
                "recipient_count": rec_n,
                "matched": matched,
                "donor_position": item["matches"].get(position_type, {}).get("donor_position"),
                "recipient_position": item["matches"].get(position_type, {}).get("recipient_position"),
            }
        per_pair.append(row)
    return {
        "backend": ap.BACKEND,
        "num_pairs": len(pairs),
        "position_types": list(position_types),
        "matching_policy": "same-question role-to-role ordinal index 0",
        "aggregate_by_position_type": [
            {
                "position_type": pt,
                "pairs_with_match": match_counts[(pt, True)],
                "pairs_without_match": match_counts[(pt, False)],
                "mean_donor_positions": ap.mean(donor_counts[pt]) or 0.0,
                "mean_recipient_positions": ap.mean(recipient_counts[pt]) or 0.0,
            }
            for pt in position_types
        ],
        "per_pair": per_pair,
    }


def build_config(
    args: argparse.Namespace,
    num_eligible_questions: int,
    num_trace_records: int,
    num_pairs: int,
) -> dict[str, Any]:
    return {
        "backend": ap.BACKEND,
        "model_path": str(args.model_path),
        "source_questions": str(args.source_questions),
        "feature_matrix": str(args.feature_matrix),
        "answer_position_score_matrix": str(args.answer_position_score_matrix),
        "primary_questions_requested": args.primary_questions,
        "num_eligible_questions_sampled": num_eligible_questions,
        "samples_per_question": args.samples_per_question,
        "num_trace_records": num_trace_records,
        "num_correct_wrong_pairs": num_pairs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "layers": list(args.layers),
        "position_types": list(args.position_types),
        "patch_type": "residual_replace",
        "alpha": args.alpha,
        "controls": list(PATCH_CONDITIONS),
        "training": False,
        "lora": False,
        "full_sprint_3c": False,
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
    }


def build_reports(
    forward_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    position_report: dict[str, Any],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    fidelity = build_fidelity_report(forward_rows)
    effect = build_effect_report(forward_rows)
    control = build_control_report(forward_rows)
    heatmap = build_heatmap_report(forward_rows)
    harm = build_harm_report(forward_rows)
    success, failure = build_case_reports(forward_rows, pair_rows)
    review_gate = render_review_gate(
        pair_rows=pair_rows,
        forward_rows=forward_rows,
        fidelity=fidelity,
        effect=effect,
        control=control,
        heatmap=heatmap,
        harm=harm,
        position_report=position_report,
    )
    reports["patching_fidelity_report.json"] = fidelity
    reports["patching_effect_report.json"] = effect
    reports["oracle_patch_vs_control_report.json"] = control
    reports["layer_position_heatmap_report.json"] = heatmap
    reports["harm_rate_report.json"] = harm
    reports["success_case_report.jsonl"] = success
    reports["failure_case_report.jsonl"] = failure
    reports["review_gate_correct_wrong_activation_patching.md"] = review_gate
    return reports


def build_fidelity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patched = [row for row in rows if row["patch_condition"] != "no_patch"]
    return {
        "backend": ap.BACKEND,
        "metric": "residual_replace_hook_fidelity",
        "num_patched_records": len(patched),
        "hook_registered_rate": rate(patched, lambda row: row["hook"].get("hook_registered")),
        "hook_removed_rate": rate(patched, lambda row: row["hook"].get("hook_removed")),
        "hook_triggered_requested_layer_rate": rate(
            patched,
            lambda row: row["hook"].get("hook_triggered_layer") == row["layer"],
        ),
        "non_target_position_contamination_check_rate": rate(
            patched,
            lambda row: row["hook"].get("non_target_position_contamination_check"),
        ),
        "mean_patch_delta_norm": ap.mean([row["hook"].get("patch_delta_norm") for row in patched]),
        "mean_recipient_hidden_changed_norm": ap.mean(
            [row["hook"].get("recipient_hidden_changed_norm") for row in patched]
        ),
        "all_hooks_reliable": all(
            row["hook"].get("hook_registered")
            and row["hook"].get("hook_removed")
            and row["hook"].get("hook_triggered_layer") == row["layer"]
            and row["hook"].get("non_target_position_contamination_check")
            for row in patched
        )
        if patched
        else False,
    }


def build_effect_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = ap.aggregate_forward_rows(rows)
    treatment = [row for row in rows if row["patch_condition"] == "correct_to_wrong"]
    return {
        "backend": ap.BACKEND,
        "metric": "single_forward_gold_minus_wrong_token_logprob_delta",
        "aggregate_by_condition_position_layer": aggregate,
        "correct_to_wrong_overall": {
            "num_records": len(treatment),
            "mean_gold_first_token_logprob_delta": ap.mean(
                [row.get("gold_first_token_logprob_delta") for row in treatment]
            ),
            "mean_wrong_first_token_logprob_delta": ap.mean(
                [row.get("wrong_first_token_logprob_delta") for row in treatment]
            ),
            "mean_clean_direction_score": ap.mean([row.get("clean_direction_score") for row in treatment]),
            "clean_direction_bootstrap": ap.bootstrap_ci(
                [row.get("clean_direction_score") for row in treatment],
                seed=33031,
            ),
        },
        "answer_accuracy_improvement_proven": False,
        "hallucination_reduction_proven": False,
        "proxy_note": "This is a teacher-forced next-token proxy after the trace, not full answer accuracy.",
    }


def build_control_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for control in [
        "no_patch",
        "correct_activation_patch",
        "same_trace_random_position_patch",
        "random_donor_patch",
    ]:
        comparisons[f"correct_to_wrong_minus_{control}"] = ap.paired_deltas_vs_control(
            rows,
            treatment="correct_to_wrong",
            control=control,
        )
    stable = comparisons["correct_to_wrong_minus_random_donor_patch"]
    return {
        "backend": ap.BACKEND,
        "comparisons": comparisons,
        "oracle_patch_beats_random_donor_stably": bool(
            stable.get("ci95_low") is not None and stable.get("ci95_low") > 0
        ),
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
    }


def build_heatmap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    treatment = [row for row in rows if row["patch_condition"] == "correct_to_wrong"]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in treatment:
        grouped[(int(row["layer"]), str(row["position_type"]))].append(row)
    heatmap = []
    for (layer, position_type), group in sorted(grouped.items()):
        heatmap.append(
            {
                "layer": layer,
                "position_type": position_type,
                "num_records": len(group),
                "mean_clean_direction_score": ap.mean([row.get("clean_direction_score") for row in group]),
                "mean_gold_first_token_logprob_delta": ap.mean(
                    [row.get("gold_first_token_logprob_delta") for row in group]
                ),
                "mean_wrong_first_token_logprob_delta": ap.mean(
                    [row.get("wrong_first_token_logprob_delta") for row in group]
                ),
                "harm_rate": rate(group, lambda row: row.get("harm")),
            }
        )
    best = max(
        heatmap,
        key=lambda row: row["mean_clean_direction_score"]
        if row["mean_clean_direction_score"] is not None
        else -float("inf"),
        default=None,
    )
    return {
        "backend": ap.BACKEND,
        "heatmap": heatmap,
        "best_layer_position_by_clean_direction": best,
    }


def build_harm_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["patch_condition"], row["position_type"], int(row["layer"]))].append(row)
    return {
        "backend": ap.BACKEND,
        "harm_proxy_definition": ap.HARM_PROXY_DEFINITION,
        "aggregate_by_condition_position_layer": [
            {
                "patch_condition": condition,
                "position_type": position_type,
                "layer": layer,
                "num_records": len(group),
                "harm_rate": rate(group, lambda row: row.get("harm")),
                "top1_changed_rate": rate(
                    group,
                    lambda row: (row.get("output_shift") or {}).get("steer_top1_changed", 0.0) > 0.0,
                ),
                "mean_entropy_delta": ap.mean(
                    [(row.get("output_shift") or {}).get("steer_entropy_delta") for row in group]
                ),
                "mean_margin_delta": ap.mean(
                    [(row.get("output_shift") or {}).get("steer_margin_delta") for row in group]
                ),
            }
            for (condition, position_type, layer), group in sorted(grouped.items())
        ],
    }


def build_case_reports(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    treatment = [row for row in rows if row["patch_condition"] == "correct_to_wrong"]
    ordered_success = sorted(
        [row for row in treatment if (row.get("clean_direction_score") or -float("inf")) > 0],
        key=lambda row: row.get("clean_direction_score") or -float("inf"),
        reverse=True,
    )
    ordered_failure = sorted(
        [row for row in treatment if (row.get("clean_direction_score") or 0.0) <= 0 or row.get("harm")],
        key=lambda row: (row.get("clean_direction_score") or 0.0),
    )
    return (
        [case_row(row, pair_by_id, "positive_clean_direction") for row in ordered_success[:30]],
        [case_row(row, pair_by_id, "non_positive_or_harmful_clean_direction") for row in ordered_failure[:30]],
    )


def case_row(row: dict[str, Any], pair_by_id: dict[str, dict[str, Any]], case_type: str) -> dict[str, Any]:
    pair = pair_by_id.get(row["pair_id"], {})
    return {
        "case_type": case_type,
        "pair_id": row["pair_id"],
        "source_question_id": row["source_question_id"],
        "question": pair.get("question"),
        "gold_answer": pair.get("gold_answer"),
        "wrong_answer": pair.get("wrong_answer"),
        "correct_sample_id": pair.get("correct_sample_id"),
        "wrong_sample_id": pair.get("wrong_sample_id"),
        "position_type": row["position_type"],
        "layer": row["layer"],
        "clean_direction_score": row.get("clean_direction_score"),
        "gold_first_token_logprob_delta": row.get("gold_first_token_logprob_delta"),
        "wrong_first_token_logprob_delta": row.get("wrong_first_token_logprob_delta"),
        "harm": row.get("harm"),
        "hook": row.get("hook"),
    }


def render_review_gate(
    *,
    pair_rows: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
    fidelity: dict[str, Any],
    effect: dict[str, Any],
    control: dict[str, Any],
    heatmap: dict[str, Any],
    harm: dict[str, Any],
    position_report: dict[str, Any],
) -> str:
    overall = effect["correct_to_wrong_overall"]
    bootstrap = overall["clean_direction_bootstrap"]
    stable_positive = bool(bootstrap.get("ci95_low") is not None and bootstrap["ci95_low"] > 0)
    beats_random = control["oracle_patch_beats_random_donor_stably"]
    best = heatmap.get("best_layer_position_by_clean_direction") or {}
    checks = [
        ("same_question_correct_wrong_pairs_exist", len(pair_rows) > 0, f"pairs={len(pair_rows)}"),
        ("reasoning_step_positions_matched", any(r["pairs_with_match"] > 0 for r in position_report["aggregate_by_position_type"]), "see reasoning_step_position_report.json"),
        ("residual_replace_hook_reliable", fidelity["all_hooks_reliable"], "registered/triggered/removed/no contamination"),
        ("correct_to_wrong_patch_forward_rows_exist", any(r["patch_condition"] == "correct_to_wrong" for r in forward_rows), f"rows={len(forward_rows)}"),
        ("clean_direction_stably_positive", stable_positive, json.dumps(bootstrap)),
        ("correct_patch_beats_random_donor", beats_random, json.dumps(control["comparisons"].get("correct_to_wrong_minus_random_donor_patch"))),
        ("harm_report_generated", True, "see harm_rate_report.json"),
        ("no_training_or_full_3c", True, "diagnostic only"),
        ("no_accuracy_or_hallucination_claim", True, "single-forward proxy only"),
    ]
    lines = [
        "# Sprint 3C-0 Correct-vs-Wrong Activation Patching Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "",
        "Scope:",
        f"- correct/wrong same-question pairs: {len(pair_rows)}",
        f"- forward rows: {len(forward_rows)}",
        "- intervention: residual_replace, alpha=1.0 unless configured otherwise",
        "- metric: teacher-forced gold-minus-wrong first-token logprob delta",
        "",
        "Main effect:",
        f"- mean_gold_first_token_logprob_delta: {overall['mean_gold_first_token_logprob_delta']}",
        f"- mean_wrong_first_token_logprob_delta: {overall['mean_wrong_first_token_logprob_delta']}",
        f"- mean_clean_direction_score: {overall['mean_clean_direction_score']}",
        f"- clean_direction_bootstrap: {json.dumps(bootstrap, ensure_ascii=False)}",
        "",
        "Best layer/position:",
        f"- {json.dumps(best, ensure_ascii=False)}",
        "",
        "Control comparison:",
        f"- oracle_patch_beats_random_donor_stably: {str(beats_random).lower()}",
        f"- comparisons: {json.dumps(control['comparisons'], ensure_ascii=False)}",
        "",
        "Fidelity:",
        f"- all_hooks_reliable: {str(fidelity['all_hooks_reliable']).lower()}",
        f"- hook_registered_rate: {fidelity['hook_registered_rate']}",
        f"- hook_removed_rate: {fidelity['hook_removed_rate']}",
        f"- hook_triggered_requested_layer_rate: {fidelity['hook_triggered_requested_layer_rate']}",
        "",
        "Harm:",
        "- See harm_rate_report.json.",
        "",
        "Required review questions:",
    ]
    review_answers = [
        ("Did we form same-question correct/wrong pairs?", len(pair_rows) > 0),
        ("Did the donor come from a correct trace?", True),
        ("Did the recipient come from a wrong trace?", True),
        ("Did we avoid cross-question correct/wrong pairing?", True),
        ("Did reasoning-step positions resolve?", any(r["pairs_with_match"] > 0 for r in position_report["aggregate_by_position_type"])),
        ("Did role-to-role ordinal matching drive the primary patch?", True),
        ("Did hooks register?", fidelity["hook_registered_rate"] == 1.0),
        ("Did hooks trigger requested layers?", fidelity["hook_triggered_requested_layer_rate"] == 1.0),
        ("Were hooks removed?", fidelity["hook_removed_rate"] == 1.0),
        ("Was non-target contamination checked?", fidelity["non_target_position_contamination_check_rate"] == 1.0),
        ("Did correct-to-wrong patches raise gold token logprob?", (overall["mean_gold_first_token_logprob_delta"] or 0) > 0),
        ("Did correct-to-wrong patches suppress wrong token logprob?", (overall["mean_wrong_first_token_logprob_delta"] or 0) < 0),
        ("Was clean direction stable by bootstrap?", stable_positive),
        ("Did correct-to-wrong beat random donor?", beats_random),
        ("Did final-answer-position control get reported if requested?", any(r["position_type"] == "final_answer_position" for r in forward_rows)),
        ("Was harm monitored?", True),
        ("Was generation accuracy evaluation performed?", False),
        ("Is 2000 rerun allowed from this result?", False),
        ("Is hallucination reduction proven?", False),
    ]
    for question, answer in review_answers:
        lines.append(f"- {question} {str(answer).lower()}")
    lines.extend(
        [
            "",
            "Checks:",
        ]
    )
    for name, passed, detail in checks:
        lines.append(f"- {name}: {str(passed).lower()} ({detail})")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This sprint tests whether correct-run activation at reasoning-step positions has a selective single-forward causal signal.",
            "- It does not evaluate steered autoregressive generation, answer accuracy, or hallucination reduction.",
            "- Full Sprint 3C / 2000-scale remains blocked unless a later review explicitly promotes the mechanism.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_empty_reports(args: argparse.Namespace, out_dir: Path, trace_rows: list[dict[str, Any]]) -> None:
    config = build_config(args, args.primary_questions, len(trace_rows), 0)
    config["incomplete_evidence_reason"] = "no same-question correct/wrong pairs were sampled"
    write_json(config, out_dir / "activation_patching_config.json")
    write_json(
        {
            "backend": ap.BACKEND,
            "num_pairs": 0,
            "position_types": list(args.position_types),
            "incomplete_evidence_reason": "no same-question correct/wrong pairs were sampled",
        },
        out_dir / "reasoning_step_position_report.json",
    )
    write_jsonl([], out_dir / "activation_patching_forward_manifest.jsonl")
    for name in [
        "patching_fidelity_report.json",
        "patching_effect_report.json",
        "oracle_patch_vs_control_report.json",
        "layer_position_heatmap_report.json",
        "harm_rate_report.json",
    ]:
        write_json(
            {
                "backend": ap.BACKEND,
                "incomplete_evidence_reason": "no same-question correct/wrong pairs were sampled",
                "ready_for_2000_rerun": False,
                "do_not_enter_full_sprint_3C": True,
            },
            out_dir / name,
        )
    write_jsonl([], out_dir / "failure_case_report.jsonl")
    write_jsonl([], out_dir / "success_case_report.jsonl")
    (out_dir / "review_gate_correct_wrong_activation_patching.md").write_text(
        "# Sprint 3C-0 Correct-vs-Wrong Activation Patching Review Gate\n\n"
        "Verdict:\n"
        "- ready_for_2000_rerun: false\n"
        "- do_not_enter_full_sprint_3C: true\n"
        "- hallucination_reduction_proven: false\n"
        "- answer_accuracy_improvement_proven: false\n\n"
        "Incomplete evidence: no same-question correct/wrong pairs were sampled.\n",
        encoding="utf-8",
    )


def write_preflight_report(out_dir: Path) -> None:
    artifact_rows = [artifact_status(path) for path in PREVIOUS_3A1_ARTIFACTS + PREVIOUS_3B0_ARTIFACTS]
    task_status = git_status_for_path(TASK_CARD)
    progress_text = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    top_blocks = "\n\n".join(progress_text.split("\n\n", 2)[:2])
    stale_top_status = "Current Status Update: Sprint 3A-0" in top_blocks
    lines = [
        "# Sprint 3C-0 Preflight Report",
        "",
        "Progress cleanup:",
        f"- stale_top_status_3A0_observed_after_cleanup: {str(stale_top_status).lower()}",
        "- PROGRESS.md current status should point to Sprint 3C-0 before the experiment run.",
        "",
        "Pre-existing task-card state:",
        f"- {TASK_CARD.relative_to(PROJECT_ROOT)} git_status: {task_status or 'clean_or_untracked_not_seen'}",
        "",
        "3A-1 / 3B-0 detailed artifact audit:",
    ]
    for row in artifact_rows:
        lines.append(
            f"- {row['path']}: exists={str(row['exists']).lower()} "
            f"tracked={str(row['tracked']).lower()} ignored={str(row['ignored']).lower()}"
        )
    lines.extend(
        [
            "",
            "Recovered conclusions boundary:",
            "- 3B-0 negated only same-run span residual deviation -> final answer-position injection -> single-forward gold-token proxy.",
            "- 3B-0 did not negate correct-run activation patching, reasoning-step-level patching, value/MLP tracing, or autoregressive generation steering.",
            "",
            "3C-0 boundary:",
            "- no training, no LoRA, no full Sprint 3C, no 2000 rerun.",
            "- ready_for_2000_rerun=false.",
            "- do_not_enter_full_sprint_3C=true.",
            "- hallucination_reduction_proven=false.",
            "- answer_accuracy_improvement_proven=false.",
            "",
        ]
    )
    (out_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def artifact_status(path: str) -> dict[str, Any]:
    full = PROJECT_ROOT / path
    return {
        "path": path,
        "exists": full.exists(),
        "tracked": git_bool(["ls-files", "--error-unmatch", "--", path]),
        "ignored": git_bool(["check-ignore", "-q", "--", path]),
    }


def git_status_for_path(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", str(path.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def git_bool(args: list[str]) -> bool:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=False)
    except Exception:
        return False
    return result.returncode == 0


def rate(rows: list[dict[str, Any]], pred: Any) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if pred(row)) / len(rows))


def infer_device(context: dict[str, Any]) -> Any:
    from recover_attention import multi_span_reasoning_scoring as msrs

    return msrs.infer_model_input_device(context["model"], "auto", context["torch"])


if __name__ == "__main__":
    main()
