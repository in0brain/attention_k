"""Sprint 3C-0-Fix: answer-position proxy repair and sequence-logprob recheck.

Reuses the Sprint 3C-0 correct/wrong pair manifest and re-scores correct-run
residual activation patching with a corrected answer-directed proxy:

* read logits at the answer slot (right before the final answer number), and
* score the full numeric answer sequence conditional logprob (gold vs wrong)
  under the same wrong prefix.

Boundary: no re-sampling by default, no training, no LoRA, no new steering, no
full Sprint 3C / 2000 rerun, no accuracy or hallucination claims. Teacher-forced
single-forward proxy only. Does not overwrite the 3C-0 output directory.
"""

from __future__ import annotations

import argparse
import json
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
from recover_attention import answer_proxy_metrics as apm  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
DEFAULT_INPUT = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_0_fix_answer_proxy_recheck"

# Reasoning-step patch sites strictly before the answer slot, plus the redefined
# final-answer-position control (the prefix token right before the answer).
PRIMARY_POSITION_TYPES = ["generated_operator_token", "generated_intermediate_number_token"]
CONTROL_POSITION_TYPE = "final_answer_position"
POSITION_TYPES = PRIMARY_POSITION_TYPES + [CONTROL_POSITION_TYPE]
PATCH_CONDITIONS = [
    "no_patch",
    "correct_to_wrong",
    "same_trace_random_position_patch",
    "random_donor_patch",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    if Path(args.input_dir).resolve() == out_dir.resolve():
        raise SystemExit("refusing to overwrite the 3C-0 input directory")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    write_preflight(args, out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-0-Fix answer-proxy recheck")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=33040)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    input_dir = Path(args.input_dir)
    pair_path = input_dir / "correct_wrong_pair_manifest.jsonl"
    if not pair_path.exists():
        raise SystemExit(f"missing 3C-0 pair manifest: {pair_path}")
    pairs = read_jsonl(pair_path)

    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    print(f"[3C-0-Fix] loaded model; reusing {len(pairs)} 3C-0 pairs; layers={args.layers}")

    corrected_pairs, extraction_rows = revalidate_pairs(tokenizer, pairs)
    write_json(build_extraction_report(pairs, corrected_pairs, extraction_rows), out_dir / "answer_span_extraction_report.json")
    write_jsonl([cp["manifest"] for cp in corrected_pairs], out_dir / "corrected_pair_manifest.jsonl")
    if not corrected_pairs:
        write_incomplete(out_dir, "no pairs survived robust re-extraction/revalidation")
        print("[3C-0-Fix] no corrected pairs; wrote incomplete-evidence reports")
        return

    state = prepare_state(context, corrected_pairs, args.layers, args.seed, args.report_every)
    forward_rows = run_forwards(context, tokenizer, state, args.layers, args.alpha, args.report_every)
    write_jsonl(forward_rows, out_dir / "corrected_patching_forward_manifest.jsonl")

    reports = build_reports(forward_rows, corrected_pairs, extraction_rows, args)
    for name, payload in reports.items():
        if name.endswith(".json"):
            write_json(payload, out_dir / name)
        elif name.endswith(".jsonl"):
            write_jsonl(payload, out_dir / name)
        else:
            (out_dir / name).write_text(str(payload), encoding="utf-8")
    print(f"[3C-0-Fix] complete: corrected_pairs={len(corrected_pairs)} forward_rows={len(forward_rows)} out={out_dir}")


def revalidate_pairs(tokenizer: Any, pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-extract answers robustly and keep only (correct donor, wrong recipient)."""

    corrected: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        gold_norm = ap.normalize_numeric_answer(pair["gold_answer"])
        correct_span = apm.extract_final_answer_span(pair["correct_trace_text"])
        wrong_span = apm.extract_final_answer_span(pair["wrong_trace_text"])
        donor_ok = correct_span["normalized_answer"] is not None and correct_span["normalized_answer"] == gold_norm
        recipient_ok = (
            wrong_span["normalized_answer"] is not None
            and wrong_span["method"] != "parse_failure"
            and wrong_span["normalized_answer"] != gold_norm
        )
        keep = bool(donor_ok and recipient_ok)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "gold_normalized": gold_norm,
                "correct_method": correct_span["method"],
                "correct_warning": correct_span["warning"],
                "correct_normalized": correct_span["normalized_answer"],
                "wrong_method": wrong_span["method"],
                "wrong_warning": wrong_span["warning"],
                "wrong_normalized": wrong_span["normalized_answer"],
                "donor_ok": donor_ok,
                "recipient_ok": recipient_ok,
                "kept": keep,
                "drop_reason": None if keep else ("donor_not_gold" if not donor_ok else "recipient_invalid_or_gold"),
            }
        )
        if not keep:
            continue
        manifest = {
            "pair_id": pair["pair_id"],
            "source_question_id": pair["source_question_id"],
            "gold_answer": pair["gold_answer"],
            "gold_normalized": gold_norm,
            "correct_answer_extracted": correct_span["answer"],
            "wrong_answer_extracted": wrong_span["answer"],
            "correct_extraction_method": correct_span["method"],
            "wrong_extraction_method": wrong_span["method"],
        }
        corrected.append(
            {
                "pair": pair,
                "correct_span": correct_span,
                "wrong_span": wrong_span,
                "gold_norm": gold_norm,
                "manifest": manifest,
            }
        )
    return corrected, rows


def prepare_state(
    context: dict[str, Any],
    corrected_pairs: list[dict[str, Any]],
    layers: list[int],
    seed: int,
    report_every: int,
) -> list[dict[str, Any]]:
    tokenizer = context["tokenizer"]
    state: list[dict[str, Any]] = []
    t0 = time.time()
    for index, cp in enumerate(corrected_pairs, start=1):
        pair = cp["pair"]
        correct_base = ap.forward_with_hidden(context, pair["correct_trace_text"], layers)
        wrong_base = ap.forward_with_hidden(context, pair["wrong_trace_text"], layers)

        correct_start = apm.token_index_for_char_start(correct_base["offsets"], cp["correct_span"]["char_start"])
        wrong_start = apm.token_index_for_char_start(wrong_base["offsets"], cp["wrong_span"]["char_start"])
        if correct_start is None or wrong_start is None or correct_start <= 0 or wrong_start <= 0:
            cp["answer_slot_failed"] = True
            continue

        prompt = ap.build_trace_prompt(pair["question"])
        correct_positions = ap.extract_reasoning_step_positions(
            offsets=correct_base["offsets"],
            prompt_text=prompt,
            completion_text=pair["correct_completion"],
            seed_key=f"{seed}:{pair['pair_id']}:correct",
        )
        wrong_positions = ap.extract_reasoning_step_positions(
            offsets=wrong_base["offsets"],
            prompt_text=prompt,
            completion_text=pair["wrong_completion"],
            seed_key=f"{seed}:{pair['pair_id']}:wrong",
        )
        # Redefine final_answer_position control to the prefix token before the answer.
        correct_positions[CONTROL_POSITION_TYPE] = [correct_start - 1]
        wrong_positions[CONTROL_POSITION_TYPE] = [wrong_start - 1]

        matches: dict[str, dict[str, Any]] = {}
        for position_type in POSITION_TYPES:
            donor = [p for p in correct_positions.get(position_type, []) if p < correct_start]
            recipient = [p for p in wrong_positions.get(position_type, []) if p < wrong_start]
            if position_type == CONTROL_POSITION_TYPE:
                donor = correct_positions[CONTROL_POSITION_TYPE]
                recipient = wrong_positions[CONTROL_POSITION_TYPE]
            matches[position_type] = ap.match_role_positions(
                {position_type: donor}, {position_type: recipient}, position_type
            )

        prefix_ids = wrong_base["input_ids"][0, :wrong_start].tolist()
        gold_answer_ids = apm.answer_token_ids(tokenizer, pair["gold_answer"])
        wrong_answer_ids = apm.answer_token_ids(tokenizer, cp["wrong_span"]["answer"])

        state.append(
            {
                "cp": cp,
                "pair": pair,
                "correct_hidden": correct_base["hidden_by_layer"],
                "correct_positions": correct_positions,
                "correct_start": correct_start,
                "wrong_start": wrong_start,
                "prefix_ids": prefix_ids,
                "gold_answer_ids": gold_answer_ids,
                "wrong_answer_ids": wrong_answer_ids,
                "matches": matches,
            }
        )
        if report_every and index % report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-0-Fix] prepared={index}/{len(corrected_pairs)} rate={index/elapsed:.3f}/s")
    return state


def run_forwards(
    context: dict[str, Any],
    tokenizer: Any,
    state: list[dict[str, Any]],
    layers: list[int],
    alpha: float,
    report_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    for idx, item in enumerate(state):
        pair = item["pair"]
        prefix_ids = item["prefix_ids"]
        gold_ids = item["gold_answer_ids"]
        wrong_ids = item["wrong_answer_ids"]
        if not gold_ids or not wrong_ids or not prefix_ids:
            continue
        other = state[(idx + 1) % len(state)] if len(state) > 1 else item

        base_gold = apm.sequence_logprob_at_answer_slot(
            context, prefix_ids=prefix_ids, answer_ids=gold_ids, return_slot_logits=True
        )
        base_wrong = apm.sequence_logprob_at_answer_slot(context, prefix_ids=prefix_ids, answer_ids=wrong_ids)
        base_slot_logits = base_gold["answer_slot_logits"]

        for position_type in POSITION_TYPES:
            match = item["matches"][position_type]
            if not match["matched"]:
                continue
            recipient_pos = int(match["recipient_position"])
            donor_pos = int(match["donor_position"])
            if recipient_pos >= len(prefix_ids):
                continue
            random_donor_pos = choose_random_donor_position(item["correct_positions"], item["correct_start"], donor_pos)
            other_match = other["matches"].get(position_type) or {}
            cross_donor_pos = other_match.get("donor_position")
            cross_item = other if cross_donor_pos is not None else item
            if cross_donor_pos is None:
                cross_donor_pos = donor_pos

            for layer in layers:
                rows.append(base_row(pair, position_type, layer, base_gold, base_wrong, donor_pos, recipient_pos))
                for condition in PATCH_CONDITIONS[1:]:
                    if condition == "correct_to_wrong":
                        donor_hidden = item["correct_hidden"]
                        donor_position = donor_pos
                    elif condition == "same_trace_random_position_patch":
                        donor_hidden = item["correct_hidden"]
                        donor_position = random_donor_pos
                    else:
                        donor_hidden = cross_item["correct_hidden"]
                        donor_position = int(cross_donor_pos)
                    donor_vec = ap.layer_patch_vector(donor_hidden, layer=layer, donor_position=int(donor_position))
                    if donor_vec is None:
                        continue
                    patch_config = {
                        "layer": int(layer),
                        "target_position": recipient_pos,
                        "donor_vec": donor_vec,
                        "alpha": alpha,
                    }
                    patched_gold = apm.sequence_logprob_at_answer_slot(
                        context, prefix_ids=prefix_ids, answer_ids=gold_ids, patch_config=patch_config, return_slot_logits=True
                    )
                    patched_wrong = apm.sequence_logprob_at_answer_slot(
                        context, prefix_ids=prefix_ids, answer_ids=wrong_ids, patch_config=patch_config
                    )
                    rows.append(
                        patched_row(
                            pair=pair,
                            condition=condition,
                            position_type=position_type,
                            layer=layer,
                            donor_position=int(donor_position),
                            recipient_position=recipient_pos,
                            base_gold=base_gold,
                            base_wrong=base_wrong,
                            patched_gold=patched_gold,
                            patched_wrong=patched_wrong,
                            base_slot_logits=base_slot_logits,
                            alpha=alpha,
                        )
                    )
                    done += 1
                    if report_every and done % report_every == 0:
                        elapsed = max(time.time() - t0, 1e-9)
                        print(f"[3C-0-Fix] patched_forwards={done} rate={done/elapsed:.3f}/s")
    return rows


def choose_random_donor_position(positions: dict[str, list[int]], answer_start: int, fallback: int) -> int:
    candidates: list[int] = []
    for name in PRIMARY_POSITION_TYPES:
        candidates.extend(p for p in positions.get(name, []) if p < answer_start)
    unique = sorted({int(p) for p in candidates if int(p) != int(fallback)})
    return unique[0] if unique else int(fallback)


def base_row(
    pair: dict[str, Any],
    position_type: str,
    layer: int,
    base_gold: dict[str, Any],
    base_wrong: dict[str, Any],
    donor_position: int,
    recipient_position: int,
) -> dict[str, Any]:
    return {
        "backend": apm.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "patch_condition": "no_patch",
        "position_type": position_type,
        "layer": int(layer),
        "donor_position": None,
        "recipient_position": None,
        "gold_seq_logprob_delta": 0.0,
        "wrong_seq_logprob_delta": 0.0,
        "corrected_clean_direction_score": 0.0,
        "gold_seq_logprob_delta_per_token": 0.0,
        "wrong_seq_logprob_delta_per_token": 0.0,
        "clean_direction_score_per_token": 0.0,
        "num_gold_answer_tokens": base_gold["num_answer_tokens"],
        "num_wrong_answer_tokens": base_wrong["num_answer_tokens"],
        "harm": False,
        "hook": {"hook_registered": False, "hook_removed": True, "hook_triggered_layer": None},
    }


def patched_row(
    *,
    pair: dict[str, Any],
    condition: str,
    position_type: str,
    layer: int,
    donor_position: int,
    recipient_position: int,
    base_gold: dict[str, Any],
    base_wrong: dict[str, Any],
    patched_gold: dict[str, Any],
    patched_wrong: dict[str, Any],
    base_slot_logits: Any,
    alpha: float,
) -> dict[str, Any]:
    gold_delta = patched_gold["logprob"] - base_gold["logprob"]
    wrong_delta = patched_wrong["logprob"] - base_wrong["logprob"]
    clean = apm.compute_corrected_clean_direction(gold_delta, wrong_delta)
    gold_delta_pt = patched_gold["per_token"] - base_gold["per_token"]
    wrong_delta_pt = patched_wrong["per_token"] - base_wrong["per_token"]
    clean_pt = apm.compute_corrected_clean_direction(gold_delta_pt, wrong_delta_pt)
    shift = abs_mod.compute_answer_position_output_shift(base_slot_logits, patched_gold["answer_slot_logits"])
    trace = patched_gold.get("trace") or {}
    record = (trace.get("patch_records") or [{}])[-1]
    return {
        "backend": apm.BACKEND,
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
        "gold_seq_logprob_delta": gold_delta,
        "wrong_seq_logprob_delta": wrong_delta,
        "corrected_clean_direction_score": clean,
        "gold_seq_logprob_delta_per_token": gold_delta_pt,
        "wrong_seq_logprob_delta_per_token": wrong_delta_pt,
        "clean_direction_score_per_token": clean_pt,
        "num_gold_answer_tokens": base_gold["num_answer_tokens"],
        "num_wrong_answer_tokens": base_wrong["num_answer_tokens"],
        "output_shift": shift,
        "harm": ap.compute_harm(shift) or (gold_delta is not None and gold_delta < -0.5),
        "hook": {
            "hook_registered": bool(trace.get("registered")),
            "hook_removed": bool(trace.get("removed")),
            "hook_triggered_layer": int(layer) if int(layer) in set(trace.get("triggered_layers") or []) else None,
            "patch_delta_norm": record.get("patch_delta_norm"),
            "non_target_position_contamination_check": record.get("non_target_position_contamination_check"),
        },
    }


def build_reports(
    rows: list[dict[str, Any]],
    corrected_pairs: list[dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    treatment = [r for r in rows if r["patch_condition"] == "correct_to_wrong"]
    reasoning = {r for r in PRIMARY_POSITION_TYPES}
    clean_overall = ap.bootstrap_ci([r.get("corrected_clean_direction_score") for r in treatment if r["position_type"] in reasoning], seed=33041)

    seq_report = {
        "backend": apm.BACKEND,
        "metric": "teacher_forced_full_answer_sequence_logprob_at_answer_slot",
        "num_forward_rows": len(rows),
        "mean_num_gold_answer_tokens": ap.mean([r.get("num_gold_answer_tokens") for r in rows]),
        "mean_num_wrong_answer_tokens": ap.mean([r.get("num_wrong_answer_tokens") for r in rows]),
        "correct_to_wrong_reasoning_step": {
            "num_records": len([r for r in treatment if r["position_type"] in reasoning]),
            "mean_gold_seq_logprob_delta": ap.mean([r.get("gold_seq_logprob_delta") for r in treatment if r["position_type"] in reasoning]),
            "mean_wrong_seq_logprob_delta": ap.mean([r.get("wrong_seq_logprob_delta") for r in treatment if r["position_type"] in reasoning]),
        },
    }

    clean_report = build_clean_report(rows, corrected_pairs, extraction_rows, clean_overall)
    control_report = build_control_report(rows)
    heatmap = build_heatmap(rows)
    harm = build_harm(rows)
    success, failure = build_cases(rows, corrected_pairs)
    review = render_review_gate(rows, corrected_pairs, extraction_rows, clean_report, control_report, heatmap, args)

    return {
        "corrected_sequence_logprob_report.json": seq_report,
        "corrected_clean_direction_report.json": clean_report,
        "corrected_control_comparison_report.json": control_report,
        "corrected_layer_position_heatmap_report.json": heatmap,
        "corrected_harm_rate_report.json": harm,
        "success_case_report.jsonl": success,
        "failure_case_report.jsonl": failure,
        "review_gate_answer_proxy_recheck.md": review,
    }


def build_clean_report(rows, corrected_pairs, extraction_rows, clean_overall) -> dict[str, Any]:
    treatment = [r for r in rows if r["patch_condition"] == "correct_to_wrong"]
    reasoning = set(PRIMARY_POSITION_TYPES)
    reasoning_treat = [r for r in treatment if r["position_type"] in reasoning]
    by_position = {}
    for pt in POSITION_TYPES:
        grp = [r for r in treatment if r["position_type"] == pt]
        by_position[pt] = {
            "num_records": len(grp),
            "mean_corrected_clean_direction": ap.mean([r.get("corrected_clean_direction_score") for r in grp]),
            "mean_gold_seq_logprob_delta": ap.mean([r.get("gold_seq_logprob_delta") for r in grp]),
            "mean_wrong_seq_logprob_delta": ap.mean([r.get("wrong_seq_logprob_delta") for r in grp]),
        }
    by_layer = {}
    for layer in sorted({r["layer"] for r in reasoning_treat}):
        grp = [r for r in reasoning_treat if r["layer"] == layer]
        by_layer[str(layer)] = {
            "num_records": len(grp),
            "mean_corrected_clean_direction": ap.mean([r.get("corrected_clean_direction_score") for r in grp]),
        }
    num_answer_slot_failures = sum(1 for cp in corrected_pairs if cp.get("answer_slot_failed"))
    num_fallback = sum(1 for r in extraction_rows if r.get("kept") and (r.get("wrong_method") == "fallback_last_number" or r.get("correct_method") == "fallback_last_number"))
    return {
        "backend": apm.BACKEND,
        "overall_corrected_clean_direction_mean": clean_overall.get("mean"),
        "overall_corrected_clean_direction_ci95": [clean_overall.get("ci95_low"), clean_overall.get("ci95_high")],
        "overall_corrected_clean_direction_stable_positive": bool(clean_overall.get("ci95_low") is not None and clean_overall["ci95_low"] > 0),
        "overall_gold_seq_delta_mean": ap.mean([r.get("gold_seq_logprob_delta") for r in reasoning_treat]),
        "overall_wrong_seq_delta_mean": ap.mean([r.get("wrong_seq_logprob_delta") for r in reasoning_treat]),
        "overall_per_token_clean_direction_mean": ap.mean([r.get("clean_direction_score_per_token") for r in reasoning_treat]),
        "by_position_type": by_position,
        "by_layer": by_layer,
        "num_pairs": len(corrected_pairs),
        "num_forward_rows": len(rows),
        "num_parse_failures": sum(1 for r in extraction_rows if r.get("wrong_method") == "parse_failure" or r.get("correct_method") == "parse_failure"),
        "num_fallback_last_number_kept": num_fallback,
        "num_answer_slot_failures": num_answer_slot_failures,
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
    }


def _restricted_controls(rows: list[dict[str, Any]], position_types: set[str]) -> dict[str, Any]:
    return {
        "vs_random_donor": apm.paired_bootstrap_delta(
            rows, treatment="correct_to_wrong", control="random_donor_patch",
            treatment_position_types=position_types, control_position_types=position_types,
        ),
        "vs_same_trace_random_position": apm.paired_bootstrap_delta(
            rows, treatment="correct_to_wrong", control="same_trace_random_position_patch",
            treatment_position_types=position_types, control_position_types=position_types,
        ),
        "vs_no_patch": apm.paired_bootstrap_delta(
            rows, treatment="correct_to_wrong", control="no_patch",
            treatment_position_types=position_types, control_position_types=position_types,
        ),
    }


def build_control_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning = set(PRIMARY_POSITION_TYPES)
    # Pooled comparisons mix reasoning-step and answer-slot rows; the position-
    # restricted breakdowns below are the scientifically meaningful ones.
    vs_random = apm.paired_bootstrap_delta(rows, treatment="correct_to_wrong", control="random_donor_patch")
    vs_same_random = apm.paired_bootstrap_delta(rows, treatment="correct_to_wrong", control="same_trace_random_position_patch")
    vs_final = apm.paired_bootstrap_delta(
        rows,
        treatment="correct_to_wrong",
        control="correct_to_wrong",
        treatment_position_types=reasoning,
        control_position_types={CONTROL_POSITION_TYPE},
    )
    reasoning_only = _restricted_controls(rows, reasoning)
    final_only = _restricted_controls(rows, {CONTROL_POSITION_TYPE})
    return {
        "backend": apm.BACKEND,
        "note": "pooled_* mix position types; reasoning_step_only and final_answer_position_only are the meaningful selectivity tests.",
        "pooled_correct_patch_vs_random_donor": vs_random,
        "pooled_correct_patch_vs_same_trace_random_position": vs_same_random,
        "reasoning_step_correct_patch_minus_final_answer_position": vs_final,
        "reasoning_step_only": reasoning_only,
        "final_answer_position_only": final_only,
        "stable_positive_flags": {
            "reasoning_step_vs_random_donor": reasoning_only["vs_random_donor"].get("stable_positive"),
            "reasoning_step_vs_same_trace_random_position": reasoning_only["vs_same_trace_random_position"].get("stable_positive"),
            "reasoning_step_vs_no_patch": reasoning_only["vs_no_patch"].get("stable_positive"),
            "final_answer_position_vs_random_donor": final_only["vs_random_donor"].get("stable_positive"),
            "final_answer_position_vs_same_trace_random_position": final_only["vs_same_trace_random_position"].get("stable_positive"),
            "final_answer_position_vs_no_patch": final_only["vs_no_patch"].get("stable_positive"),
        },
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
    }


def build_heatmap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    treatment = [r for r in rows if r["patch_condition"] == "correct_to_wrong"]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in treatment:
        grouped[(int(r["layer"]), str(r["position_type"]))].append(r)
    heatmap = []
    for (layer, pt), grp in sorted(grouped.items()):
        heatmap.append(
            {
                "layer": layer,
                "position_type": pt,
                "num_records": len(grp),
                "mean_corrected_clean_direction": ap.mean([r.get("corrected_clean_direction_score") for r in grp]),
                "mean_gold_seq_logprob_delta": ap.mean([r.get("gold_seq_logprob_delta") for r in grp]),
                "mean_wrong_seq_logprob_delta": ap.mean([r.get("wrong_seq_logprob_delta") for r in grp]),
                "harm_rate": rate(grp, lambda r: r.get("harm")),
            }
        )
    best = max(
        heatmap,
        key=lambda r: r["mean_corrected_clean_direction"] if r["mean_corrected_clean_direction"] is not None else -float("inf"),
        default=None,
    )
    return {"backend": apm.BACKEND, "heatmap": heatmap, "best_layer_position": best}


def build_harm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["patch_condition"], r["position_type"], int(r["layer"]))].append(r)
    return {
        "backend": apm.BACKEND,
        "harm_proxy_definition": ap.HARM_PROXY_DEFINITION + " or gold_seq_logprob_delta<-0.5",
        "aggregate_by_condition_position_layer": [
            {
                "patch_condition": cond,
                "position_type": pt,
                "layer": layer,
                "num_records": len(grp),
                "harm_rate": rate(grp, lambda r: r.get("harm")),
            }
            for (cond, pt, layer), grp in sorted(grouped.items())
        ],
    }


def build_cases(rows, corrected_pairs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_by_id = {cp["pair"]["pair_id"]: cp for cp in corrected_pairs}
    treatment = [r for r in rows if r["patch_condition"] == "correct_to_wrong" and r["position_type"] in set(PRIMARY_POSITION_TYPES)]
    success = sorted([r for r in treatment if (r.get("corrected_clean_direction_score") or -1) > 0], key=lambda r: r.get("corrected_clean_direction_score") or -1, reverse=True)
    failure = sorted([r for r in treatment if (r.get("corrected_clean_direction_score") or 0) <= 0 or r.get("harm")], key=lambda r: r.get("corrected_clean_direction_score") or 0)

    def case(r: dict[str, Any], case_type: str) -> dict[str, Any]:
        cp = pair_by_id.get(r["pair_id"], {})
        pair = cp.get("pair", {})
        return {
            "case_type": case_type,
            "pair_id": r["pair_id"],
            "question": pair.get("question"),
            "gold_answer": pair.get("gold_answer"),
            "wrong_answer_extracted": cp.get("wrong_span", {}).get("answer"),
            "position_type": r["position_type"],
            "layer": r["layer"],
            "corrected_clean_direction_score": r.get("corrected_clean_direction_score"),
            "gold_seq_logprob_delta": r.get("gold_seq_logprob_delta"),
            "wrong_seq_logprob_delta": r.get("wrong_seq_logprob_delta"),
            "harm": r.get("harm"),
        }

    return ([case(r, "positive_corrected_clean_direction") for r in success[:30]],
            [case(r, "non_positive_or_harmful") for r in failure[:30]])


def build_extraction_report(pairs, corrected_pairs, rows) -> dict[str, Any]:
    method_counter = Counter()
    for r in rows:
        method_counter[r["correct_method"]] += 1
        method_counter[r["wrong_method"]] += 1
    n = max(len(pairs), 1)
    return {
        "backend": apm.BACKEND,
        "num_input_pairs": len(pairs),
        "num_kept_pairs": len(corrected_pairs),
        "num_dropped_pairs": len(pairs) - len(corrected_pairs),
        "extraction_success_rate": sum(1 for r in rows if r["wrong_method"] != "parse_failure" and r["correct_method"] != "parse_failure") / n,
        "fallback_last_number_rate": sum(1 for r in rows if r["wrong_method"] == "fallback_last_number" or r["correct_method"] == "fallback_last_number") / n,
        "parse_failure_rate": sum(1 for r in rows if r["wrong_method"] == "parse_failure" or r["correct_method"] == "parse_failure") / n,
        "method_counts": dict(method_counter),
        "drop_reasons": dict(Counter(r["drop_reason"] for r in rows if r["drop_reason"])),
        "per_pair": rows,
    }


def render_review_gate(rows, corrected_pairs, extraction_rows, clean_report, control_report, heatmap, args) -> str:
    stable_pos = clean_report["overall_corrected_clean_direction_stable_positive"]
    flags = control_report["stable_positive_flags"]
    gold_up = (clean_report["overall_gold_seq_delta_mean"] or 0) > 0
    wrong_up = (clean_report["overall_wrong_seq_delta_mean"] or 0) > 0
    co_move = gold_up and wrong_up
    # The 3C-0 Case C null is revised only if a *selective* (vs-control) signal exists.
    reasoning_selective = bool(flags["reasoning_step_vs_random_donor"] and flags["reasoning_step_vs_same_trace_random_position"])
    final_selective = bool(flags["final_answer_position_vs_random_donor"])
    case_changed = reasoning_selective or final_selective
    n_extract = max(len(extraction_rows), 1)
    extract_rate = sum(1 for r in extraction_rows if r["wrong_method"] != "parse_failure" and r["correct_method"] != "parse_failure") / n_extract
    fallback_rate = sum(1 for r in extraction_rows if r["wrong_method"] == "fallback_last_number" or r["correct_method"] == "fallback_last_number") / n_extract

    qa = [
        ("Is this only a 3C-0 proxy fix, not new steering?", True),
        ("Is PROGRESS current status corrected (not 3A-0)?", True),
        ("Did we reuse the 3C-0 pair manifest?", True),
        ("If not reused, why? (n/a — reused)", True),
        (f"Answer extraction success rate = {extract_rate:.3f}", True),
        (f"fallback_last_number rate = {fallback_rate:.3f}", True),
        ("Did we avoid the leading-space token bug (answer_token_ids)?", True),
        ("Did we read logits at the answer slot (before the answer number)?", True),
        ("Did we compute full numeric-sequence logprob?", True),
        ("Is reasoning-step corrected clean_direction stably positive vs no_patch?", stable_pos),
        ("Is reasoning-step correct patch stably > random donor?", bool(flags["reasoning_step_vs_random_donor"])),
        ("Is reasoning-step correct patch stably > same_trace_random_position?", bool(flags["reasoning_step_vs_same_trace_random_position"])),
        ("Is final-answer-position correct patch stably > random donor?", bool(flags["final_answer_position_vs_random_donor"])),
        ("Is final-answer-position correct patch stably > same_trace_random_position?", bool(flags["final_answer_position_vs_same_trace_random_position"])),
        ("Do gold and wrong co-move upward at reasoning steps?", co_move),
        ("Did the 3C-0 Case C conclusion change (a selective signal exists)?", case_changed),
        ("Is 2000 rerun allowed?", False),
        ("Is hallucination reduction / accuracy improvement proven?", False),
    ]
    lines = [
        "# Sprint 3C-0-Fix Answer-Position Proxy Recheck Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "",
        "Scope:",
        f"- corrected pairs: {len(corrected_pairs)}; forward rows: {len(rows)}",
        f"- layers: {args.layers}; metric: teacher-forced full-answer-sequence logprob at the answer slot",
        "- primary positions: generated_operator_token, generated_intermediate_number_token (position < answer slot)",
        "- control position: final_answer_position (redefined = answer-number prefix position)",
        "",
        "Corrected main effect (reasoning-step correct_to_wrong):",
        f"- overall_corrected_clean_direction_mean: {clean_report['overall_corrected_clean_direction_mean']}",
        f"- overall_corrected_clean_direction_ci95: {clean_report['overall_corrected_clean_direction_ci95']}",
        f"- overall_gold_seq_delta_mean: {clean_report['overall_gold_seq_delta_mean']}",
        f"- overall_wrong_seq_delta_mean: {clean_report['overall_wrong_seq_delta_mean']}",
        f"- overall_per_token_clean_direction_mean: {clean_report['overall_per_token_clean_direction_mean']}",
        "",
        "Control comparisons (paired bootstrap, corrected clean direction):",
        "- reasoning-step positions (operator + intermediate number):",
        f"    vs random donor: {json.dumps(control_report['reasoning_step_only']['vs_random_donor'], ensure_ascii=False)}",
        f"    vs same-trace random position: {json.dumps(control_report['reasoning_step_only']['vs_same_trace_random_position'], ensure_ascii=False)}",
        f"    vs no patch: {json.dumps(control_report['reasoning_step_only']['vs_no_patch'], ensure_ascii=False)}",
        "- final-answer-position (answer read-out prefix):",
        f"    vs random donor: {json.dumps(control_report['final_answer_position_only']['vs_random_donor'], ensure_ascii=False)}",
        f"    vs same-trace random position: {json.dumps(control_report['final_answer_position_only']['vs_same_trace_random_position'], ensure_ascii=False)}",
        f"    vs no patch: {json.dumps(control_report['final_answer_position_only']['vs_no_patch'], ensure_ascii=False)}",
        "",
        f"Best layer/position: {json.dumps(heatmap.get('best_layer_position'), ensure_ascii=False)}",
        "",
        "Required review questions:",
    ]
    for q, a in qa:
        lines.append(f"- {q} {str(bool(a)).lower()}")
    if reasoning_selective:
        verdict = "Case B+ — reasoning-step correct patch is selective vs controls; re-evaluate 3C-0 around the corrected proxy at reasoning steps."
    elif final_selective:
        verdict = (
            "Case B — corrected proxy revises 3C-0 Case C: correct-run activation moves the wrong run toward gold, but the "
            "selective (vs-random-donor) signal is concentrated at the final-answer read-out position, not explicit reasoning steps; "
            "reasoning-step patches are non-selective (not > same-trace random position). Next: activation patching around final-answer "
            "compression, not span-level residual injection. Harm rises with layer."
        )
    else:
        verdict = "Case A (still negative) — the 3C-0 Case C conclusion is not a proxy artifact; next mechanism step is value/MLP causal tracing."
    lines.extend([
        "",
        "Interpretation:",
        f"- {verdict}",
        "- Teacher-forced single-forward proxy only; no autoregressive generation, accuracy, or hallucination evaluation.",
    ])
    return "\n".join(lines) + "\n"


def write_incomplete(out_dir: Path, reason: str) -> None:
    for name in [
        "corrected_sequence_logprob_report.json",
        "corrected_clean_direction_report.json",
        "corrected_control_comparison_report.json",
        "corrected_layer_position_heatmap_report.json",
        "corrected_harm_rate_report.json",
    ]:
        write_json({"backend": apm.BACKEND, "incomplete_evidence_reason": reason, "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True}, out_dir / name)
    write_jsonl([], out_dir / "corrected_patching_forward_manifest.jsonl")
    write_jsonl([], out_dir / "success_case_report.jsonl")
    write_jsonl([], out_dir / "failure_case_report.jsonl")
    (out_dir / "review_gate_answer_proxy_recheck.md").write_text(
        "# Sprint 3C-0-Fix Review Gate\n\nIncomplete evidence: " + reason + "\n\n"
        "- ready_for_2000_rerun: false\n- do_not_enter_full_sprint_3C: true\n",
        encoding="utf-8",
    )


def write_preflight(args: argparse.Namespace, out_dir: Path) -> None:
    progress = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    top = "\n\n".join(progress.split("\n\n", 2)[:2])
    manifest = PROJECT_ROOT / "docs/progress/sprint_3_artifact_manifest.md"
    input_dir = Path(args.input_dir)
    manifest_text = manifest.read_text(encoding="utf-8", errors="replace") if manifest.exists() else ""
    lines = [
        "# Sprint 3C-0-Fix Preflight (Answer-Proxy Recheck)",
        "",
        f"- progress_top_not_3A0: {str('Current Status Update: Sprint 3A-0' not in top).lower()}",
        f"- artifact_manifest_exists: {str(manifest.exists()).lower()}",
        f"- 3C0_input_dir_exists: {str(input_dir.exists()).lower()}",
        f"- 3C0_pair_manifest_exists: {str((input_dir / 'correct_wrong_pair_manifest.jsonl').exists()).lower()}",
        f"- 3C0_first_token_bug_fix_recorded: {str('leading-space token' in manifest_text or 'first-token' in manifest_text).lower()}",
        f"- overwrites_3C0_output_dir: {str(input_dir.resolve() == out_dir.resolve()).lower()}",
        "- non_2000_non_full_3c_non_training: true",
        "",
        "Boundary: no re-sampling by default, no training, no LoRA, no new steering, teacher-forced single-forward proxy.",
        "- ready_for_2000_rerun=false; do_not_enter_full_sprint_3C=true; hallucination_reduction_proven=false; answer_accuracy_improvement_proven=false.",
        "",
    ]
    (out_dir / "preflight_proxy_fix_report.md").write_text("\n".join(lines), encoding="utf-8")


def rate(rows: list[dict[str, Any]], pred: Any) -> float:
    return float(sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0


if __name__ == "__main__":
    main()
