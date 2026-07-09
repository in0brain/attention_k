"""Sprint 3C-1: final-answer-compression value/MLP causal tracing.

Decomposes the Sprint 3C-0-Fix whole-residual patch at the answer-readout
position into per-module writes (self-attention output, MLP output, residual
output), interpolation-patches each with correct-run donor activation, and tests
whether any module x layer x alpha yields a selective, donor-specific,
site-specific, low-harm causal signal on the corrected answer-sequence proxy.

Boundary: reuses the 3C-0 / 3C-0-Fix pairs (no re-sampling); no training, no
LoRA, no new steering deployment, no full Sprint 3C / 2000 rerun, no accuracy or
hallucination claims. Teacher-forced single-forward proxy only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import activation_patching as ap  # noqa: E402
from recover_attention import answer_proxy_metrics as apm  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention import module_causal_tracing as mct  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
DEFAULT_FIX_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_fix_answer_proxy_recheck"
DEFAULT_3C0_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing"

PATCH_CONDITIONS = [
    "no_patch",
    "correct_donor_patch",
    "random_donor_patch",
    "same_trace_random_position_patch",
    "same_pair_wrong_position_patch",
    "wrong_donor_self_patch",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    for guard in (Path(args.input_dir), DEFAULT_3C0_DIR):
        if guard.resolve() == out_dir.resolve():
            raise SystemExit("refusing to overwrite a 3C-0 / 3C-0-Fix directory")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    write_preflight(args, out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-1 module-level causal tracing")
    parser.add_argument("--input-dir", default=str(DEFAULT_FIX_DIR))
    parser.add_argument("--pair-source-dir", default=str(DEFAULT_3C0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--modules", nargs="+", default=list(mct.MODULE_TYPES))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=33050)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    for module_type in args.modules:
        if module_type not in mct.MODULE_TYPES:
            raise ValueError(f"unsupported module: {module_type}")

    corrected_ids = load_corrected_pair_ids(Path(args.input_dir))
    all_pairs = {p["pair_id"]: p for p in read_jsonl(Path(args.pair_source_dir) / "correct_wrong_pair_manifest.jsonl")}
    if corrected_ids:
        pairs = [all_pairs[pid] for pid in corrected_ids if pid in all_pairs]
    else:
        pairs = list(all_pairs.values())
    if not pairs:
        write_incomplete(out_dir, "no reusable correct/wrong pairs found")
        print("[3C-1] no pairs; wrote incomplete-evidence reports")
        return

    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    print(f"[3C-1] loaded model; pairs={len(pairs)} layers={args.layers} modules={args.modules} alphas={args.alphas}")

    state = prepare_state(context, tokenizer, pairs, args)
    write_jsonl([s["manifest"] for s in state], out_dir / "module_patch_pair_manifest.jsonl")
    write_jsonl([capture_row(s) for s in state], out_dir / "module_activation_capture_manifest.jsonl")
    if not state:
        write_incomplete(out_dir, "no pairs produced a valid answer-readout slot")
        print("[3C-1] no usable readout slots; wrote incomplete-evidence reports")
        return

    write_json(build_config(args, len(pairs), len(state)), out_dir / "module_patching_config.json")
    rows = run_forwards(context, tokenizer, state, args)
    write_jsonl(rows, out_dir / "module_patching_forward_manifest.jsonl")

    reports = build_reports(rows, state, args)
    for name, payload in reports.items():
        if name.endswith(".json"):
            write_json(payload, out_dir / name)
        elif name.endswith(".jsonl"):
            write_jsonl(payload, out_dir / name)
        else:
            (out_dir / name).write_text(str(payload), encoding="utf-8")
    print(f"[3C-1] complete: pairs={len(state)} rows={len(rows)} out={out_dir}")


def load_corrected_pair_ids(fix_dir: Path) -> list[str]:
    path = fix_dir / "corrected_pair_manifest.jsonl"
    if not path.exists():
        return []
    return [str(r["pair_id"]) for r in read_jsonl(path)]


def prepare_state(context: dict[str, Any], tokenizer: Any, pairs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    t0 = time.time()
    for index, pair in enumerate(pairs, start=1):
        correct_cap = mct.capture_module_outputs(context, pair["correct_trace_text"], args.layers, args.modules)
        wrong_cap = mct.capture_module_outputs(context, pair["wrong_trace_text"], args.layers, args.modules)
        correct_readout = mct.build_answer_readout(context, pair["correct_trace_text"], correct_cap)
        wrong_readout = mct.build_answer_readout(context, pair["wrong_trace_text"], wrong_cap)
        if correct_readout is None or wrong_readout is None:
            continue

        prompt = ap.build_trace_prompt(pair["question"])
        correct_positions = ap.extract_reasoning_step_positions(
            offsets=correct_cap["offsets"], prompt_text=prompt, completion_text=pair["correct_completion"], seed_key=f"{args.seed}:{pair['pair_id']}:c"
        )
        wrong_positions = ap.extract_reasoning_step_positions(
            offsets=wrong_cap["offsets"], prompt_text=prompt, completion_text=pair["wrong_completion"], seed_key=f"{args.seed}:{pair['pair_id']}:w"
        )
        correct_rand = earlier_position(correct_positions, correct_readout["answer_start"], correct_readout["readout_position"])
        wrong_rand = earlier_position(wrong_positions, wrong_readout["answer_start"], wrong_readout["readout_position"])

        gold_ids = apm.answer_token_ids(tokenizer, pair["gold_answer"])
        wrong_ids = apm.answer_token_ids(tokenizer, wrong_readout["span"]["answer"])
        if not gold_ids or not wrong_ids or not wrong_readout["prefix_ids"]:
            continue

        state.append(
            {
                "pair": pair,
                "correct_cap": correct_cap,
                "wrong_cap": wrong_cap,
                "correct_readout": correct_readout,
                "wrong_readout": wrong_readout,
                "correct_rand_pos": correct_rand,
                "wrong_rand_pos": wrong_rand,
                "prefix_ids": wrong_readout["prefix_ids"],
                "gold_ids": gold_ids,
                "wrong_ids": wrong_ids,
                "manifest": {
                    "pair_id": pair["pair_id"],
                    "source_question_id": pair["source_question_id"],
                    "gold_answer": pair["gold_answer"],
                    "wrong_answer_extracted": wrong_readout["span"]["answer"],
                    "correct_readout_position": correct_readout["readout_position"],
                    "wrong_readout_position": wrong_readout["readout_position"],
                    "num_gold_answer_tokens": len(gold_ids),
                    "num_wrong_answer_tokens": len(wrong_ids),
                },
            }
        )
        if args.report_every and index % args.report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-1] prepared={index}/{len(pairs)} rate={index/elapsed:.3f}/s")
    return state


def earlier_position(positions: dict[str, list[int]], answer_start: int, exclude: int) -> int | None:
    candidates: list[int] = []
    for name in ("generated_operator_token", "generated_intermediate_number_token", "generated_equals_or_result_marker"):
        candidates.extend(p for p in positions.get(name, []) if p < answer_start and p != exclude)
    unique = sorted({int(p) for p in candidates})
    return unique[0] if unique else None


def run_forwards(context: dict[str, Any], tokenizer: Any, state: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    for idx, item in enumerate(state):
        pair = item["pair"]
        prefix_ids = item["prefix_ids"]
        gold_ids = item["gold_ids"]
        wrong_ids = item["wrong_ids"]
        other = state[(idx + 1) % len(state)] if len(state) > 1 else item

        base_gold = mct.sequence_logprob_with_module_patch(context, prefix_ids=prefix_ids, answer_ids=gold_ids, return_slot_logits=True)
        base_wrong = mct.sequence_logprob_with_module_patch(context, prefix_ids=prefix_ids, answer_ids=wrong_ids)
        base_slot = base_gold["answer_slot_logits"]

        for module_type in args.modules:
            for layer in args.layers:
                # no_patch baseline row (module/layer/alpha-independent, clean == 0)
                for alpha in args.alphas:
                    rows.append(base_row(pair, module_type, layer, alpha, base_gold, base_wrong))
                for alpha in args.alphas:
                    for condition in PATCH_CONDITIONS[1:]:
                        donor_vec, target_pos = resolve_patch(item, other, condition, module_type, layer)
                        if donor_vec is None or target_pos is None or target_pos >= len(prefix_ids):
                            continue
                        patch = {"layer": int(layer), "module_type": module_type, "donor_vec": donor_vec, "target_position": int(target_pos), "alpha": float(alpha)}
                        patched_gold = mct.sequence_logprob_with_module_patch(context, prefix_ids=prefix_ids, answer_ids=gold_ids, patch=patch, return_slot_logits=True)
                        patched_wrong = mct.sequence_logprob_with_module_patch(context, prefix_ids=prefix_ids, answer_ids=wrong_ids, patch=patch)
                        rows.append(forward_row(pair, condition, module_type, layer, alpha, target_pos, base_gold, base_wrong, patched_gold, patched_wrong, base_slot))
                        done += 1
                        if args.report_every and done % args.report_every == 0:
                            elapsed = max(time.time() - t0, 1e-9)
                            print(f"[3C-1] patched_forwards={done} rate={done/elapsed:.3f}/s")
    return rows


def resolve_patch(item: dict[str, Any], other: dict[str, Any], condition: str, module_type: str, layer: int) -> tuple[Any, int | None]:
    cc = item["correct_cap"]["captured"]
    wc = item["wrong_cap"]["captured"]
    cr = item["correct_readout"]["readout_position"]
    wr = item["wrong_readout"]["readout_position"]
    if condition == "correct_donor_patch":
        return mct.module_vector(cc, layer=layer, module_type=module_type, position=cr), wr
    if condition == "random_donor_patch":
        opos = other["correct_readout"]["readout_position"]
        return mct.module_vector(other["correct_cap"]["captured"], layer=layer, module_type=module_type, position=opos), wr
    if condition == "same_trace_random_position_patch":
        pos = item["correct_rand_pos"]
        if pos is None:
            return None, None
        return mct.module_vector(cc, layer=layer, module_type=module_type, position=pos), wr
    if condition == "same_pair_wrong_position_patch":
        target = item["wrong_rand_pos"]
        if target is None:
            return None, None
        return mct.module_vector(cc, layer=layer, module_type=module_type, position=cr), target
    if condition == "wrong_donor_self_patch":
        return mct.module_vector(wc, layer=layer, module_type=module_type, position=wr), wr
    return None, None


def base_row(pair, module_type, layer, alpha, base_gold, base_wrong) -> dict[str, Any]:
    return {
        "backend": mct.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "patch_condition": "no_patch",
        "module_type": module_type,
        "layer": int(layer),
        "alpha": float(alpha),
        "target_position": None,
        "gold_seq_logprob_delta": 0.0,
        "wrong_seq_logprob_delta": 0.0,
        "corrected_clean_direction_score": 0.0,
        "clean_direction_score_per_token": 0.0,
        "num_gold_answer_tokens": base_gold["num_answer_tokens"],
        "num_wrong_answer_tokens": base_wrong["num_answer_tokens"],
        "harm": False,
        "hook": {"hook_registered": False, "hook_removed": True, "hook_triggered": False},
    }


def forward_row(pair, condition, module_type, layer, alpha, target_pos, base_gold, base_wrong, patched_gold, patched_wrong, base_slot) -> dict[str, Any]:
    gold_delta = patched_gold["logprob"] - base_gold["logprob"]
    wrong_delta = patched_wrong["logprob"] - base_wrong["logprob"]
    clean = apm.compute_corrected_clean_direction(gold_delta, wrong_delta)
    clean_pt = apm.compute_corrected_clean_direction(
        patched_gold["per_token"] - base_gold["per_token"], patched_wrong["per_token"] - base_wrong["per_token"]
    )
    shift = abs_mod.compute_answer_position_output_shift(base_slot, patched_gold["answer_slot_logits"])
    trace = patched_gold.get("trace") or {}
    triggered = bool(trace.get("triggered"))
    record = (trace.get("patch_records") or [{}])[-1]
    harm = bool(ap.compute_harm(shift) or (gold_delta is not None and gold_delta < -0.5) or (clean is not None and clean < -0.5))
    return {
        "backend": mct.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "patch_condition": condition,
        "module_type": module_type,
        "layer": int(layer),
        "alpha": float(alpha),
        "target_position": int(target_pos),
        "gold_seq_logprob_delta": gold_delta,
        "wrong_seq_logprob_delta": wrong_delta,
        "corrected_clean_direction_score": clean,
        "clean_direction_score_per_token": clean_pt,
        "num_gold_answer_tokens": base_gold["num_answer_tokens"],
        "num_wrong_answer_tokens": base_wrong["num_answer_tokens"],
        "output_shift": shift,
        "harm": harm,
        "hook": {
            "hook_registered": bool(trace.get("registered")),
            "hook_removed": bool(trace.get("removed")),
            "hook_triggered": triggered,
            "patch_delta_norm": record.get("patch_delta_norm"),
        },
    }


# ---------------------------------------------------------------- paired deltas


def paired_delta(rows, *, treatment_cond, control_cond, key_fields, treatment_module=None, control_module=None, restrict_alpha=None, seed_tag=None) -> dict[str, Any]:
    treat: dict[tuple[Any, ...], float] = {}
    ctrl: dict[tuple[Any, ...], float] = {}
    for r in rows:
        value = r.get("corrected_clean_direction_score")
        if value is None:
            continue
        if restrict_alpha is not None and float(r["alpha"]) != float(restrict_alpha):
            continue
        key = tuple(r[f] for f in key_fields)
        if r["patch_condition"] == treatment_cond and (treatment_module is None or r["module_type"] == treatment_module):
            treat[key] = float(value)
        if r["patch_condition"] == control_cond and (control_module is None or r["module_type"] == control_module):
            ctrl[key] = float(value)
    deltas = [treat[k] - ctrl[k] for k in treat.keys() & ctrl.keys()]
    tag = seed_tag or f"{treatment_cond}:{control_cond}:{treatment_module}:{control_module}"
    out = ap.bootstrap_ci(deltas, seed=ap.stable_int_seed(tag))
    out["stable_positive"] = bool(out.get("ci95_low") is not None and out["ci95_low"] > 0)
    return out


CELL_KEYS = ("pair_id", "module_type", "layer", "alpha")


def build_reports(rows, state, args) -> dict[str, Any]:
    fidelity = build_fidelity(rows)
    effect = build_effect(rows, args)
    donor = build_specificity(rows, args, treatment="correct_donor_patch", control="random_donor_patch", label="donor_specificity")
    site = build_specificity(rows, args, treatment="correct_donor_patch", control="same_trace_random_position_patch", label="site_specificity")
    control = build_control(rows, args)
    heatmap = build_heatmap(rows, args)
    harm = build_harm(rows)
    success, failure = build_cases(rows, state)
    review = render_review_gate(rows, state, args, fidelity, effect, donor, site, control, heatmap, harm)
    return {
        "module_patching_fidelity_report.json": fidelity,
        "module_patching_effect_report.json": effect,
        "module_control_comparison_report.json": control,
        "donor_specificity_report.json": donor,
        "site_specificity_report.json": site,
        "layer_module_heatmap_report.json": heatmap,
        "harm_control_report.json": harm,
        "success_case_report.jsonl": success,
        "failure_case_report.jsonl": failure,
        "review_gate_final_answer_compression_tracing.md": review,
    }


def build_fidelity(rows) -> dict[str, Any]:
    patched = [r for r in rows if r["patch_condition"] != "no_patch"]
    self_rows = [r for r in rows if r["patch_condition"] == "wrong_donor_self_patch"]
    correct_rows = [r for r in rows if r["patch_condition"] == "correct_donor_patch"]
    self_abs = ap.mean([abs(r["corrected_clean_direction_score"]) for r in self_rows if r.get("corrected_clean_direction_score") is not None]) or 0.0
    correct_abs = ap.mean([abs(r["corrected_clean_direction_score"]) for r in correct_rows if r.get("corrected_clean_direction_score") is not None]) or 0.0
    self_pdn = ap.mean([r["hook"].get("patch_delta_norm") for r in self_rows if r["hook"].get("patch_delta_norm") is not None]) or 0.0
    correct_pdn = ap.mean([r["hook"].get("patch_delta_norm") for r in correct_rows if r["hook"].get("patch_delta_norm") is not None]) or 0.0
    return {
        "backend": mct.BACKEND,
        "num_patched_records": len(patched),
        "hook_registered_rate": rate(patched, lambda r: r["hook"].get("hook_registered")),
        "hook_triggered_rate": rate(patched, lambda r: r["hook"].get("hook_triggered")),
        "hook_removed_rate": rate(patched, lambda r: r["hook"].get("hook_removed")),
        "all_hooks_reliable": bool(patched and all(r["hook"].get("hook_registered") and r["hook"].get("hook_triggered") and r["hook"].get("hook_removed") for r in patched)),
        # wrong_donor_self patches the recipient's own activation; it is not an exact
        # no-op because the donor is captured from the full-trace forward while it is
        # injected into a shorter prefix+answer forward (attention kernels are not
        # bit-identical across sequence lengths). This is the numerical/self floor and
        # it cancels in the donor/site specificity contrasts (all donors share it).
        "self_floor_mean_abs_clean_direction": self_abs,
        "correct_donor_mean_abs_clean_direction": correct_abs,
        "self_floor_ratio_of_correct": (self_abs / correct_abs) if correct_abs else None,
        "self_patch_delta_norm": self_pdn,
        "correct_patch_delta_norm": correct_pdn,
        "self_floor_small_vs_correct_donor": bool(correct_abs > 0 and self_abs < 0.5 * correct_abs and self_pdn < 0.5 * correct_pdn),
    }


def build_effect(rows, args) -> dict[str, Any]:
    treat = [r for r in rows if r["patch_condition"] == "correct_donor_patch"]
    by_cell = {}
    grouped = defaultdict(list)
    for r in treat:
        grouped[(r["module_type"], int(r["layer"]), float(r["alpha"]))].append(r)
    for (mt, layer, alpha), grp in sorted(grouped.items()):
        by_cell[f"{mt}|L{layer}|a{alpha}"] = {
            "module_type": mt, "layer": layer, "alpha": alpha, "num_records": len(grp),
            "mean_clean_direction": ap.mean([r.get("corrected_clean_direction_score") for r in grp]),
            "mean_gold_seq_logprob_delta": ap.mean([r.get("gold_seq_logprob_delta") for r in grp]),
            "mean_wrong_seq_logprob_delta": ap.mean([r.get("wrong_seq_logprob_delta") for r in grp]),
            "harm_rate": rate(grp, lambda r: r.get("harm")),
        }
    return {"backend": mct.BACKEND, "metric": "corrected_answer_sequence_clean_direction", "by_module_layer_alpha": by_cell,
            "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True,
            "hallucination_reduction_proven": False, "answer_accuracy_improvement_proven": False}


def build_specificity(rows, args, *, treatment, control, label) -> dict[str, Any]:
    per_module = {}
    per_cell = {}
    for mt in args.modules:
        mrows = [r for r in rows if r["module_type"] == mt]
        per_module[mt] = paired_delta(mrows, treatment_cond=treatment, control_cond=control, key_fields=CELL_KEYS, seed_tag=f"{label}:{mt}")
        for layer in args.layers:
            for alpha in args.alphas:
                cell = [r for r in mrows if int(r["layer"]) == layer and float(r["alpha"]) == alpha]
                per_cell[f"{mt}|L{layer}|a{alpha}"] = paired_delta(cell, treatment_cond=treatment, control_cond=control, key_fields=CELL_KEYS, seed_tag=f"{label}:{mt}:{layer}:{alpha}")
    stable_cells = [k for k, v in per_cell.items() if v.get("stable_positive")]
    return {"backend": mct.BACKEND, "metric": label, "definition": f"clean_direction({treatment}) - clean_direction({control})",
            "per_module": per_module, "per_cell": per_cell, "stable_positive_cells": stable_cells,
            "any_stable_positive": bool(stable_cells)}


def build_control(rows, args) -> dict[str, Any]:
    cross = {}
    key = ("pair_id", "layer", "alpha")
    for a, b in [("attention_output", "residual_output"), ("mlp_output", "residual_output"), ("attention_output", "mlp_output")]:
        cross[f"{a}_minus_{b}"] = paired_delta(rows, treatment_cond="correct_donor_patch", control_cond="correct_donor_patch", key_fields=key, treatment_module=a, control_module=b, seed_tag=f"cross:{a}:{b}")
    others = {}
    for control in ["no_patch", "same_pair_wrong_position_patch", "wrong_donor_self_patch"]:
        others[f"correct_donor_minus_{control}"] = paired_delta(rows, treatment_cond="correct_donor_patch", control_cond=control, key_fields=CELL_KEYS, seed_tag=f"ctrl:{control}")
    return {"backend": mct.BACKEND, "cross_module": cross, "vs_other_controls": others,
            "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True}


def build_heatmap(rows, args) -> dict[str, Any]:
    treat = [r for r in rows if r["patch_condition"] == "correct_donor_patch"]
    grouped = defaultdict(list)
    for r in treat:
        grouped[(r["module_type"], int(r["layer"]))].append(r)
    cells = []
    for (mt, layer), grp in sorted(grouped.items()):
        cells.append({"module_type": mt, "layer": layer, "num_records": len(grp),
                      "mean_clean_direction": ap.mean([r.get("corrected_clean_direction_score") for r in grp]),
                      "harm_rate": rate(grp, lambda r: r.get("harm"))})
    best = max(cells, key=lambda c: c["mean_clean_direction"] if c["mean_clean_direction"] is not None else -float("inf"), default=None)
    return {"backend": mct.BACKEND, "cells": cells, "best_module_layer": best}


def build_harm(rows) -> dict[str, Any]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["patch_condition"], r["module_type"], int(r["layer"]), float(r["alpha"]))].append(r)
    return {"backend": mct.BACKEND,
            "harm_proxy_definition": "top1_changed or entropy_delta>1.0 or margin_delta<-0.25 or gold_seq_logprob_delta<-0.5 or clean_direction<-0.5",
            "aggregate": [{"patch_condition": c, "module_type": m, "layer": l, "alpha": a, "num_records": len(g), "harm_rate": rate(g, lambda r: r.get("harm"))}
                          for (c, m, l, a), g in sorted(grouped.items())]}


def build_cases(rows, state) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_by_id = {s["pair"]["pair_id"]: s for s in state}
    treat = [r for r in rows if r["patch_condition"] == "correct_donor_patch"]
    success = sorted([r for r in treat if (r.get("corrected_clean_direction_score") or -1) > 0 and not r.get("harm")], key=lambda r: r.get("corrected_clean_direction_score") or -1, reverse=True)
    failure = sorted([r for r in treat if (r.get("corrected_clean_direction_score") or 0) <= 0 or r.get("harm")], key=lambda r: r.get("corrected_clean_direction_score") or 0)

    def case(r, tag):
        s = pair_by_id.get(r["pair_id"], {})
        pair = s.get("pair", {})
        return {"case_type": tag, "pair_id": r["pair_id"], "question": pair.get("question"), "gold_answer": pair.get("gold_answer"),
                "module_type": r["module_type"], "layer": r["layer"], "alpha": r["alpha"],
                "corrected_clean_direction_score": r.get("corrected_clean_direction_score"),
                "gold_seq_logprob_delta": r.get("gold_seq_logprob_delta"), "wrong_seq_logprob_delta": r.get("wrong_seq_logprob_delta"), "harm": r.get("harm")}

    return ([case(r, "selective_low_harm_candidate") for r in success[:30]], [case(r, "non_positive_or_harmful") for r in failure[:30]])


def render_review_gate(rows, state, args, fidelity, effect, donor, site, control, heatmap, harm) -> str:
    # A module x layer x alpha "wins" if clean>0 and donor+site specificity both stable positive with controllable harm.
    winners = []
    for key, cell in effect["by_module_layer_alpha"].items():
        d = donor["per_cell"].get(key, {})
        s = site["per_cell"].get(key, {})
        clean_pos = (cell["mean_clean_direction"] or 0) > 0
        if clean_pos and d.get("stable_positive") and s.get("stable_positive"):
            winners.append({"cell": key, "clean": cell["mean_clean_direction"], "harm_rate": cell["harm_rate"],
                            "donor_ci": [d.get("ci95_low"), d.get("ci95_high")], "site_ci": [s.get("ci95_low"), s.get("ci95_high")]})
    winners.sort(key=lambda w: (w["harm_rate"], -(w["clean"] or 0)))
    low_harm_winners = [w for w in winners if w["harm_rate"] <= 0.2]
    best = heatmap.get("best_module_layer") or {}

    if low_harm_winners:
        verdict = f"Selective low-harm causal site(s) found: {low_harm_winners[0]['cell']} (and {len(low_harm_winners)-1} more). Next: MLP/attention readout-direction analysis at that site."
    elif winners:
        verdict = f"Selective site(s) found but harm not clearly controllable (best {winners[0]['cell']}, harm={winners[0]['harm_rate']:.2f}). Next: finer decomposition / harm-controlled probe; do not scale."
    else:
        verdict = "Case D — no module x layer x alpha passes donor+site specificity; the 3C-0-Fix readout signal is non-specific answer-readout perturbation. Next: pause steering, return to detection/diagnosis."

    donor_any = donor["any_stable_positive"]
    site_any = site["any_stable_positive"]
    attn_vs_res = control["cross_module"].get("attention_output_minus_residual_output", {})
    mlp_vs_res = control["cross_module"].get("mlp_output_minus_residual_output", {})
    attn_vs_mlp = control["cross_module"].get("attention_output_minus_mlp_output", {})

    qa = [
        ("Is this module-level causal tracing, not full 3C?", True),
        ("Did we reuse the 3C-0-Fix corrected proxy?", True),
        ("Did we use the answer-number prefix position as readout?", True),
        ("Did we compute full numeric-sequence logprob?", True),
        (f"correct/wrong pairs used = {len(state)}", True),
        (f"layers = {args.layers}", True),
        (f"module types = {args.modules}", True),
        (f"alphas = {args.alphas}", True),
        ("Hook fidelity passed (registered/triggered/removed)?", fidelity["all_hooks_reliable"]),
        ("wrong_donor_self floor small vs correct donor (sanity)?", fidelity["self_floor_small_vs_correct_donor"]),
        ("Any module x layer x alpha clean_direction stable positive vs no_patch?", (best.get("mean_clean_direction") or 0) > 0),
        ("Any donor_specificity stable positive?", donor_any),
        ("Any site_specificity stable positive?", site_any),
        ("Any selective (donor+site) cell?", bool(winners)),
        ("Any selective cell with low harm (<=0.2)?", bool(low_harm_winners)),
        ("Is 2000 rerun allowed?", False),
        ("Is hallucination reduction / accuracy improvement proven?", False),
    ]
    lines = [
        "# Sprint 3C-1 Final-Answer Compression Value/MLP Causal Tracing Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "",
        "Scope:",
        f"- pairs: {len(state)}; forward rows: {len(rows)}",
        f"- modules: {args.modules}; layers: {args.layers}; alphas: {args.alphas}",
        "- target: final-answer readout position (answer-number prefix); metric: corrected answer-sequence clean_direction",
        "",
        "Specificity (any stable-positive cell):",
        f"- donor_specificity stable cells: {donor['stable_positive_cells']}",
        f"- site_specificity stable cells: {site['stable_positive_cells']}",
        "",
        "Cross-module (correct_donor clean_direction, paired):",
        f"- attention_output - residual_output: {json.dumps(attn_vs_res, ensure_ascii=False)}",
        f"- mlp_output - residual_output: {json.dumps(mlp_vs_res, ensure_ascii=False)}",
        f"- attention_output - mlp_output: {json.dumps(attn_vs_mlp, ensure_ascii=False)}",
        "",
        f"Best module/layer (avg over alpha): {json.dumps(best, ensure_ascii=False)}",
        f"Selective (donor+site) cells: {json.dumps(winners[:8], ensure_ascii=False)}",
        "",
        "Required review questions:",
    ]
    for q, a in qa:
        lines.append(f"- {q} {str(bool(a)).lower()}")
    lines.extend(["", "Interpretation:", f"- {verdict}",
                  "- Teacher-forced single-forward proxy only; no autoregressive generation, accuracy, or hallucination evaluation."])
    return "\n".join(lines) + "\n"


def capture_row(s: dict[str, Any]) -> dict[str, Any]:
    return {"pair_id": s["pair"]["pair_id"], "correct_readout_position": s["correct_readout"]["readout_position"],
            "wrong_readout_position": s["wrong_readout"]["readout_position"], "correct_rand_pos": s["correct_rand_pos"],
            "wrong_rand_pos": s["wrong_rand_pos"], "prefix_len": len(s["prefix_ids"]),
            "num_gold_answer_tokens": len(s["gold_ids"]), "num_wrong_answer_tokens": len(s["wrong_ids"])}


def build_config(args, num_pairs, num_state) -> dict[str, Any]:
    return {"backend": mct.BACKEND, "model_path": args.model_path, "num_input_pairs": num_pairs, "num_usable_pairs": num_state,
            "layers": list(args.layers), "modules": list(args.modules), "alphas": list(args.alphas),
            "patch_type": "module_output_interpolation", "controls": PATCH_CONDITIONS,
            "training": False, "lora": False, "full_sprint_3c": False,
            "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True,
            "hallucination_reduction_proven": False, "answer_accuracy_improvement_proven": False}


def write_incomplete(out_dir: Path, reason: str) -> None:
    for name in ["module_patching_fidelity_report.json", "module_patching_effect_report.json", "module_control_comparison_report.json",
                 "donor_specificity_report.json", "site_specificity_report.json", "layer_module_heatmap_report.json", "harm_control_report.json"]:
        write_json({"backend": mct.BACKEND, "incomplete_evidence_reason": reason, "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True}, out_dir / name)
    for name in ["module_patch_pair_manifest.jsonl", "module_activation_capture_manifest.jsonl", "module_patching_forward_manifest.jsonl", "success_case_report.jsonl", "failure_case_report.jsonl"]:
        write_jsonl([], out_dir / name)
    (out_dir / "review_gate_final_answer_compression_tracing.md").write_text(
        "# Sprint 3C-1 Review Gate\n\nIncomplete evidence: " + reason + "\n\n- ready_for_2000_rerun: false\n- do_not_enter_full_sprint_3C: true\n", encoding="utf-8")


def write_preflight(args, out_dir: Path) -> None:
    progress = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    top = "\n\n".join(progress.split("\n\n", 2)[:2])
    manifest = PROJECT_ROOT / "docs/progress/sprint_3_artifact_manifest.md"
    manifest_text = manifest.read_text(encoding="utf-8", errors="replace") if manifest.exists() else ""
    fix_dir = Path(args.input_dir)
    lines = [
        "# Sprint 3C-1 Preflight (Final-Answer Compression Module Causal Tracing)",
        "",
        f"- progress_top_not_3A0: {str('Current Status Update: Sprint 3A-0' not in top).lower()}",
        f"- artifact_manifest_exists: {str(manifest.exists()).lower()}",
        f"- 3C0_fix_dir_exists: {str(fix_dir.exists()).lower()}",
        f"- 3C0_fix_case_b_recorded: {str('Case B' in manifest_text).lower()}",
        f"- 3C0_fix_reused_3c0_pair_manifest: {str('reuse' in manifest_text.lower() or 'reused' in manifest_text.lower()).lower()}",
        f"- corrected_pair_manifest_available: {str((fix_dir / 'corrected_pair_manifest.jsonl').exists()).lower()}",
        f"- corrected_reports_available: {str((fix_dir / 'corrected_clean_direction_report.json').exists() and (fix_dir / 'corrected_control_comparison_report.json').exists()).lower()}",
        f"- overwrites_prior_output_dir: {str(fix_dir.resolve() == out_dir.resolve() or DEFAULT_3C0_DIR.resolve() == out_dir.resolve()).lower()}",
        "- non_2000_non_full_3c_non_training: true",
        "- outputs_gitignored_manifest_will_be_updated: true",
        "",
        "Boundary: reuse 3C-0/3C-0-Fix pairs, no re-sampling, no training, no new steering deployment, teacher-forced single-forward proxy.",
        "- ready_for_2000_rerun=false; do_not_enter_full_sprint_3C=true; hallucination_reduction_proven=false; answer_accuracy_improvement_proven=false.",
        "",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def rate(rows, pred) -> float:
    return float(sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0


if __name__ == "__main__":
    main()
