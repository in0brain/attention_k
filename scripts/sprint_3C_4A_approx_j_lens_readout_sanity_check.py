"""Sprint 3C-4A: approximate J-lens readout sanity check.

Compares direct MLP-output logit-lens scores (m @ W_U) with a directional
finite-difference estimate of how that same MLP output affects the actual final
logits.  This is a small readout-method sanity check only: no steering, no
training, no full Sprint 3C, no 2000-scale run, and no accuracy / hallucination
claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import activation_patching as ap  # noqa: E402
from recover_attention import answer_proxy_metrics as apm  # noqa: E402
from recover_attention import approx_j_lens_readout as aj  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention import mlp_readout_attribution as mra  # noqa: E402
from recover_attention import module_causal_tracing as mct  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import read_json, write_json  # noqa: E402

DEFAULT_3C3_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_3_mlp_readout_attribution_probe"
DEFAULT_FIX_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_fix_answer_proxy_recheck"
DEFAULT_3C0_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
TASK_CARD = "docs/codex_tasks/sprint_3C_4A_approx_j_lens_readout_sanity_check.md"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    for guard in (Path(args.input_dir), Path(args.fix_input_dir), Path(args.pair_source_dir)):
        if out_dir.resolve() == guard.resolve():
            raise SystemExit(f"refusing to overwrite prior sprint output directory: {guard}")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    write_preflight(args, out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-4A approximate J-lens readout sanity check")
    parser.add_argument("--input-dir", default=str(DEFAULT_3C3_DIR))
    parser.add_argument("--fix-input-dir", default=str(DEFAULT_FIX_DIR))
    parser.add_argument("--pair-source-dir", default=str(DEFAULT_3C0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24])
    parser.add_argument("--primary-layer", type=int, default=24)
    parser.add_argument("--secondary-layer", type=int, default=20)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=34)
    parser.add_argument("--random-candidate-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33080)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    fix_by_id = {str(r["pair_id"]): r for r in read_jsonl(Path(args.fix_input_dir) / "corrected_pair_manifest.jsonl")}
    pair_ids = list(fix_by_id)
    all_pairs = {str(p["pair_id"]): p for p in read_jsonl(Path(args.pair_source_dir) / "correct_wrong_pair_manifest.jsonl")}
    pairs = [all_pairs[pid] for pid in pair_ids if pid in all_pairs]
    if args.max_pairs:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        write_incomplete(out_dir, "no reusable 3C-0-Fix corrected pairs found")
        return

    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    model = context["model"]
    unembedding_weight = model.get_output_embeddings().weight.detach()
    answer_ids = answer_subset_tokens(tokenizer, pairs, fix_by_id)
    number_token_ids = mra.filter_number_like_tokens(tokenizer, extra_answer_token_ids=answer_ids)
    print(
        f"[3C-4A] loaded model; pairs={len(pairs)} traces={2 * len(pairs)} "
        f"layers={args.layers} epsilons={args.epsilons} number_tokens={len(number_token_ids)}"
    )

    examples, forward_rows, comparison_rows = collect_examples(
        context=context,
        tokenizer=tokenizer,
        unembedding_weight=unembedding_weight,
        number_token_ids=number_token_ids,
        pairs=pairs,
        fix_by_id=fix_by_id,
        args=args,
    )
    write_jsonl(forward_rows, out_dir / "approx_j_lens_forward_manifest.jsonl")
    write_jsonl(comparison_rows, out_dir / "direct_vs_approx_readout_manifest.jsonl")

    if len(examples) < 6 or not comparison_rows:
        write_incomplete(out_dir, f"only {len(examples)} usable examples and {len(comparison_rows)} comparison rows")
        return

    reports = build_reports(examples, comparison_rows, args)
    write_json(build_config(args, pairs, examples, number_token_ids), out_dir / "approx_j_lens_config.json")
    write_json(reports["topk"], out_dir / "topk_overlap_report.json")
    write_json(reports["rank"], out_dir / "rank_correlation_report.json")
    write_json(reports["ordering"], out_dir / "token_ordering_report.json")
    write_json(reports["diagnostic"], out_dir / "diagnostic_comparison_report.json")
    write_json(reports["epsilon"], out_dir / "epsilon_stability_report.json")
    write_jsonl(reports["success_cases"], out_dir / "success_case_report.jsonl")
    write_jsonl(reports["failure_cases"], out_dir / "failure_case_report.jsonl")
    (out_dir / "review_gate_approx_j_lens_readout_sanity_check.md").write_text(
        reports["review_gate"], encoding="utf-8"
    )
    print(f"[3C-4A] complete: examples={len(examples)} comparisons={len(comparison_rows)} out={out_dir}")


def answer_subset_tokens(tokenizer: Any, pairs: list[dict[str, Any]], fix_by_id: dict[str, dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for pair in pairs:
        fix = fix_by_id.get(str(pair["pair_id"]), {})
        for value in [
            pair.get("gold_answer"),
            pair.get("correct_answer"),
            pair.get("wrong_answer"),
            fix.get("correct_answer_extracted"),
            fix.get("wrong_answer_extracted"),
        ]:
            out.extend(apm.answer_token_ids(tokenizer, value))
    return sorted({int(t) for t in out})


def collect_examples(
    *,
    context: dict[str, Any],
    tokenizer: Any,
    unembedding_weight: Any,
    number_token_ids: list[int],
    pairs: list[dict[str, Any]],
    fix_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for pair_index, pair in enumerate(pairs, start=1):
        fix = fix_by_id.get(str(pair["pair_id"]), {})
        for trace_kind, trace_key in [("correct_trace", "correct_trace_text"), ("wrong_trace", "wrong_trace_text")]:
            result = build_trace_example(
                context=context,
                tokenizer=tokenizer,
                unembedding_weight=unembedding_weight,
                number_token_ids=number_token_ids,
                pair=pair,
                fix=fix,
                trace_kind=trace_kind,
                trace_text=pair.get(trace_key) or "",
                args=args,
            )
            if result is None:
                continue
            example, f_rows, c_rows = result
            examples.append(example)
            forward_rows.extend(f_rows)
            comparison_rows.extend(c_rows)
        if args.report_every and pair_index % args.report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-4A] processed_pairs={pair_index}/{len(pairs)} rate={pair_index/elapsed:.3f}/s")
    return examples, forward_rows, comparison_rows


def build_trace_example(
    *,
    context: dict[str, Any],
    tokenizer: Any,
    unembedding_weight: Any,
    number_token_ids: list[int],
    pair: dict[str, Any],
    fix: dict[str, Any],
    trace_kind: str,
    trace_text: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    capture = mct.capture_module_outputs(context, trace_text, args.layers, ["mlp_output"])
    readout = mct.build_answer_readout(context, trace_text, capture)
    if readout is None:
        return None
    prefix_ids = readout["prefix_ids"]
    if not prefix_ids:
        return None

    model_answer = readout["span"]["answer"]
    model_answer_norm = ap.normalize_numeric_answer(model_answer)
    gold_norm = ap.normalize_numeric_answer(pair.get("gold_answer"))
    wrong_eval = fix.get("wrong_answer_extracted") or pair.get("wrong_answer")
    model_answer_ids = apm.answer_token_ids(tokenizer, model_answer)
    gold_ids = apm.answer_token_ids(tokenizer, pair.get("gold_answer"))
    wrong_ids = apm.answer_token_ids(tokenizer, wrong_eval)
    answer_token_ids = sorted({int(t) for t in model_answer_ids + gold_ids + wrong_ids})
    score_token_ids = sorted(set(number_token_ids) | set(answer_token_ids))

    base = aj.answer_slot_logits_with_mlp_perturb(context, prefix_ids=prefix_ids)
    base_logits = base["logits"]
    final_score_map = aj.tensor_to_token_scores(base_logits, score_token_ids)
    final_features = aj.score_map_features(
        final_score_map, tokenizer=tokenizer, number_token_ids=score_token_ids, top_k=args.top_k, prefix="final_logits"
    )

    example = {
        "example_id": f"{pair['pair_id']}::{trace_kind}",
        "backend": aj.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair.get("source_question_id"),
        "trace_kind": trace_kind,
        "question": pair.get("question"),
        "gold_answer_eval_only": pair.get("gold_answer"),
        "wrong_answer_eval_only": wrong_eval,
        "model_final_answer": model_answer,
        "model_final_answer_normalized": model_answer_norm,
        "wrong_label": int(not (model_answer_norm is not None and gold_norm is not None and model_answer_norm == gold_norm)),
        "answer_readout_position": readout["readout_position"],
        "prefix_len": len(prefix_ids),
        "model_answer_first_token_id": model_answer_ids[0] if model_answer_ids else None,
        "gold_first_token_id_eval_only": gold_ids[0] if gold_ids else None,
        "wrong_first_token_id_eval_only": wrong_ids[0] if wrong_ids else None,
        "final_logits_features": final_features,
        "direct_layer_features": {},
        "approx_layer_features_by_epsilon": defaultdict(dict),
    }
    forward_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    final_top = aj.top_token_ids(final_score_map, score_token_ids, k=args.top_k)
    random_ids = deterministic_random_number_tokens(
        score_token_ids,
        count=args.random_candidate_count,
        seed_key=f"{pair['pair_id']}:{trace_kind}:{args.seed}",
    )

    for layer in args.layers:
        vec = mct.module_vector(capture["captured"], layer=int(layer), module_type="mlp_output", position=readout["readout_position"])
        if vec is None:
            continue
        mlp_norm = float(vec.float().norm().item())
        direct_score_ids = sorted(set(score_token_ids) | set(random_ids))
        direct_map = aj.project_vector_to_token_scores(vec, unembedding_weight, direct_score_ids)
        direct_features = aj.score_map_features(
            direct_map,
            tokenizer=tokenizer,
            number_token_ids=score_token_ids,
            top_k=args.top_k,
            prefix="direct",
            vector_norm=mlp_norm,
        )
        example["direct_layer_features"][str(layer)] = direct_features
        direct_top = aj.top_token_ids(direct_map, score_token_ids, k=args.top_k)

        for epsilon in args.epsilons:
            patched = aj.answer_slot_logits_with_mlp_perturb(
                context,
                prefix_ids=prefix_ids,
                perturb={
                    "layer": int(layer),
                    "direction_vec": vec,
                    "target_position": int(readout["readout_position"]),
                    "epsilon": float(epsilon),
                    "base_norm": mlp_norm,
                },
            )
            delta_logits = patched["logits"] - base_logits
            approx_scores_tensor = delta_logits / max(float(epsilon), 1e-12)
            approx_score_ids = sorted(set(score_token_ids) | set(direct_top) | set(final_top) | set(random_ids))
            approx_map = aj.tensor_to_token_scores(approx_scores_tensor, approx_score_ids)
            approx_features = aj.score_map_features(
                approx_map,
                tokenizer=tokenizer,
                number_token_ids=score_token_ids,
                top_k=args.top_k,
                prefix="approx_j",
                vector_norm=mlp_norm,
            )
            example["approx_layer_features_by_epsilon"][str(epsilon)][str(layer)] = approx_features
            approx_top = aj.top_token_ids(approx_map, score_token_ids, k=args.top_k)
            candidate_ids = sorted(set(direct_top) | set(approx_top) | set(final_top) | set(answer_token_ids) | set(random_ids))
            direct_candidate_map = ensure_scores_for_candidates(direct_map, vec, unembedding_weight, candidate_ids)
            approx_candidate_map = aj.tensor_to_token_scores(approx_scores_tensor, candidate_ids)
            final_candidate_map = aj.tensor_to_token_scores(base_logits, candidate_ids)
            comparison_rows.append(
                build_comparison_row(
                    example=example,
                    layer=int(layer),
                    epsilon=float(epsilon),
                    tokenizer=tokenizer,
                    number_token_ids=score_token_ids,
                    direct_map=direct_map,
                    approx_map=approx_map,
                    final_map=final_score_map,
                    direct_top=direct_top,
                    approx_top=approx_top,
                    final_top=final_top,
                    candidate_ids=candidate_ids,
                    candidate_direct_map=direct_candidate_map,
                    candidate_approx_map=approx_candidate_map,
                    candidate_final_map=final_candidate_map,
                    top_k=args.top_k,
                )
            )
            forward_rows.append(
                {
                    "backend": aj.BACKEND,
                    "pair_id": pair["pair_id"],
                    "source_question_id": pair.get("source_question_id"),
                    "trace_kind": trace_kind,
                    "layer": int(layer),
                    "epsilon": float(epsilon),
                    "direction_type": "mlp_self_direction",
                    "answer_readout_position": int(readout["readout_position"]),
                    "prefix_len": len(prefix_ids),
                    "mlp_output_norm": mlp_norm,
                    "delta_logits_norm": float(delta_logits.float().norm().item()),
                    "base_final_top1_number_token_id": final_top[0] if final_top else None,
                    "direct_top1_number_token_id": direct_top[0] if direct_top else None,
                    "approx_j_top1_number_token_id": approx_top[0] if approx_top else None,
                    "hook_trace": patched["trace"],
                }
            )
    return normalize_example(example), forward_rows, comparison_rows


def ensure_scores_for_candidates(score_map: dict[int, float], vector: Any, unembedding_weight: Any, candidate_ids: list[int]) -> dict[int, float]:
    missing = [int(t) for t in candidate_ids if int(t) not in score_map]
    if not missing:
        return dict(score_map)
    out = dict(score_map)
    out.update(aj.project_vector_to_token_scores(vector, unembedding_weight, missing))
    return out


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    # Convert defaultdicts before JSON serialization and report construction.
    approx = example.get("approx_layer_features_by_epsilon") or {}
    example["approx_layer_features_by_epsilon"] = {str(k): dict(v) for k, v in approx.items()}
    return example


def deterministic_random_number_tokens(token_ids: list[int], *, count: int, seed_key: str) -> list[int]:
    ids = sorted({int(t) for t in token_ids})
    if not ids or count <= 0:
        return []
    rng = np.random.default_rng(ap.stable_int_seed(seed_key))
    size = min(int(count), len(ids))
    return [int(ids[i]) for i in rng.choice(len(ids), size=size, replace=False).tolist()]


def build_comparison_row(
    *,
    example: dict[str, Any],
    layer: int,
    epsilon: float,
    tokenizer: Any,
    number_token_ids: list[int],
    direct_map: dict[int, float],
    approx_map: dict[int, float],
    final_map: dict[int, float],
    direct_top: list[int],
    approx_top: list[int],
    final_top: list[int],
    candidate_ids: list[int],
    candidate_direct_map: dict[int, float],
    candidate_approx_map: dict[int, float],
    candidate_final_map: dict[int, float],
    top_k: int,
) -> dict[str, Any]:
    overlap = {str(k): aj.topk_overlap(direct_top, approx_top, k=k) for k in [1, 3, 5, 10, top_k]}
    spearman_all = aj.spearman_correlation_from_maps(direct_map, approx_map, number_token_ids)
    spearman_candidates = aj.spearman_correlation_from_maps(direct_map, approx_map, candidate_ids)
    model_id = example.get("model_answer_first_token_id")
    gold_id = example.get("gold_first_token_id_eval_only")
    wrong_id = example.get("wrong_first_token_id_eval_only")
    candidate_scores = []
    for token_id in candidate_ids:
        candidate_scores.append(
            {
                "token_id": int(token_id),
                "token_text": mra.token_text(tokenizer, int(token_id)),
                "direct_score": candidate_direct_map.get(int(token_id)),
                "approx_j_score": candidate_approx_map.get(int(token_id)),
                "final_logits_score": candidate_final_map.get(int(token_id)),
            }
        )
    return {
        "backend": aj.BACKEND,
        "example_id": example["example_id"],
        "pair_id": example["pair_id"],
        "source_question_id": example.get("source_question_id"),
        "trace_kind": example["trace_kind"],
        "wrong_label": int(example["wrong_label"]),
        "layer": int(layer),
        "epsilon": float(epsilon),
        "direction_type": "mlp_self_direction",
        "direct_top_k_number_tokens": [
            {"token_id": tid, "token_text": mra.token_text(tokenizer, tid), "score": direct_map.get(tid)}
            for tid in direct_top[:top_k]
        ],
        "approx_j_top_k_number_tokens": [
            {"token_id": tid, "token_text": mra.token_text(tokenizer, tid), "score": approx_map.get(tid)}
            for tid in approx_top[:top_k]
        ],
        "final_logits_top_k_number_tokens": [
            {"token_id": tid, "token_text": mra.token_text(tokenizer, tid), "score": final_map.get(tid)}
            for tid in final_top[:top_k]
        ],
        "topk_overlap": overlap,
        "rank_correlation_spearman_all_number_tokens": spearman_all,
        "rank_correlation_spearman_candidate_subset": spearman_candidates,
        "token_ordering": {
            "model_answer_first_token_id": model_id,
            "gold_first_token_id_eval_only": gold_id,
            "wrong_first_token_id_eval_only": wrong_id,
            "direct_model_rank": aj.rank_of_token(direct_map, model_id, number_token_ids),
            "approx_j_model_rank": aj.rank_of_token(approx_map, model_id, number_token_ids),
            "final_logits_model_rank": aj.rank_of_token(final_map, model_id, number_token_ids),
            "direct_gold_rank_eval_only": aj.rank_of_token(direct_map, gold_id, number_token_ids),
            "approx_j_gold_rank_eval_only": aj.rank_of_token(approx_map, gold_id, number_token_ids),
            "final_logits_gold_rank_eval_only": aj.rank_of_token(final_map, gold_id, number_token_ids),
            "direct_wrong_rank_eval_only": aj.rank_of_token(direct_map, wrong_id, number_token_ids),
            "approx_j_wrong_rank_eval_only": aj.rank_of_token(approx_map, wrong_id, number_token_ids),
            "final_logits_wrong_rank_eval_only": aj.rank_of_token(final_map, wrong_id, number_token_ids),
        },
        "candidate_token_scores": candidate_scores,
    }


def build_reports(examples: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    topk = build_topk_overlap_report(comparison_rows)
    rank = build_rank_correlation_report(comparison_rows)
    ordering = build_token_ordering_report(comparison_rows)
    diagnostic = build_diagnostic_comparison_report(examples, args)
    epsilon = build_epsilon_stability_report(comparison_rows)
    success, failure = build_cases(comparison_rows, examples, args)
    review = render_review_gate(examples, args, topk, rank, ordering, diagnostic, epsilon)
    return {
        "topk": topk,
        "rank": rank,
        "ordering": ordering,
        "diagnostic": diagnostic,
        "epsilon": epsilon,
        "success_cases": success,
        "failure_cases": failure,
        "review_gate": review,
    }


def build_topk_overlap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), float(row["epsilon"]))].append(row)
    per_cell = {}
    for (layer, epsilon), group in sorted(grouped.items()):
        cell = {"layer": layer, "epsilon": epsilon, "num_rows": len(group)}
        for k in [1, 3, 5, 10, 20]:
            vals = [((r.get("topk_overlap") or {}).get(str(k)) or {}).get("overlap_rate") for r in group]
            top1 = [((r.get("topk_overlap") or {}).get(str(k)) or {}).get("top1_match") for r in group]
            cell[f"top{k}_overlap_rate_mean"] = aj.mean(vals)
            if k == 1:
                cell["top1_match_rate"] = sum(1 for v in top1 if v) / len(group) if group else 0.0
        per_cell[f"L{layer}|eps{epsilon}"] = cell
    return {"backend": aj.BACKEND, "per_layer_epsilon": per_cell}


def build_rank_correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), float(row["epsilon"]))].append(row)
    per_cell = {}
    for (layer, epsilon), group in sorted(grouped.items()):
        all_scores = [r.get("rank_correlation_spearman_all_number_tokens") for r in group]
        cand_scores = [r.get("rank_correlation_spearman_candidate_subset") for r in group]
        per_cell[f"L{layer}|eps{epsilon}"] = {
            "layer": layer,
            "epsilon": epsilon,
            "num_rows": len(group),
            "spearman_all_number_tokens": aj.bootstrap_ci(all_scores, seed_tag=f"spearman-all:{layer}:{epsilon}"),
            "spearman_candidate_subset": aj.bootstrap_ci(cand_scores, seed_tag=f"spearman-cand:{layer}:{epsilon}"),
        }
    return {"backend": aj.BACKEND, "per_layer_epsilon": per_cell}


def build_token_ordering_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), float(row["epsilon"]))].append(row)
    per_cell = {}
    for (layer, epsilon), group in sorted(grouped.items()):
        ordering = [r.get("token_ordering") or {} for r in group]
        per_cell[f"L{layer}|eps{epsilon}"] = {
            "layer": layer,
            "epsilon": epsilon,
            "num_rows": len(group),
            "direct_model_rank_mean": aj.mean([o.get("direct_model_rank") for o in ordering]),
            "approx_j_model_rank_mean": aj.mean([o.get("approx_j_model_rank") for o in ordering]),
            "final_logits_model_rank_mean": aj.mean([o.get("final_logits_model_rank") for o in ordering]),
            "direct_gold_rank_eval_only_mean": aj.mean([o.get("direct_gold_rank_eval_only") for o in ordering]),
            "approx_j_gold_rank_eval_only_mean": aj.mean([o.get("approx_j_gold_rank_eval_only") for o in ordering]),
            "final_logits_gold_rank_eval_only_mean": aj.mean([o.get("final_logits_gold_rank_eval_only") for o in ordering]),
            "direct_gold_beats_wrong_eval_only_rate": rank_better_rate(ordering, "direct_gold_rank_eval_only", "direct_wrong_rank_eval_only"),
            "approx_j_gold_beats_wrong_eval_only_rate": rank_better_rate(ordering, "approx_j_gold_rank_eval_only", "approx_j_wrong_rank_eval_only"),
            "final_logits_gold_beats_wrong_eval_only_rate": rank_better_rate(ordering, "final_logits_gold_rank_eval_only", "final_logits_wrong_rank_eval_only"),
        }
    return {
        "backend": aj.BACKEND,
        "note": "gold/wrong rank fields are eval-only diagnostics, not feature inputs.",
        "per_layer_epsilon": per_cell,
    }


def rank_better_rate(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    hits = []
    for row in rows:
        left = row.get(left_key)
        right = row.get(right_key)
        if left is None or right is None:
            continue
        hits.append(1.0 if int(left) < int(right) else 0.0)
    return aj.mean(hits)


def build_diagnostic_comparison_report(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    direct_scores = []
    final_margins = []
    labels = []
    rows_by_epsilon: dict[str, list[float | None]] = {str(e): [] for e in args.epsilons}
    feature_rows = []
    for example in examples:
        labels.append(int(example["wrong_label"]))
        final_margins.append((example.get("final_logits_features") or {}).get("final_logits_number_margin"))
        direct_features = (example.get("direct_layer_features") or {}).get(str(args.primary_layer)) or {}
        direct_secondary = (example.get("direct_layer_features") or {}).get(str(args.secondary_layer)) or {}
        direct_layer_agreement = aj.token_agreement(
            direct_features.get("direct_top1_number_token_id"),
            direct_secondary.get("direct_top1_number_token_id"),
        )
        model_id = example.get("model_answer_first_token_id")
        direct_model_agreement = aj.token_agreement(direct_features.get("direct_top1_number_token_id"), model_id)
        direct_risk = aj.compute_readout_risk(
            direct_features,
            prefix="direct",
            layer_agreement=direct_layer_agreement,
            model_answer_agreement=direct_model_agreement,
        )
        direct_scores.append(direct_risk)
        for epsilon in args.epsilons:
            approx_by_layer = (example.get("approx_layer_features_by_epsilon") or {}).get(str(float(epsilon))) or {}
            primary = approx_by_layer.get(str(args.primary_layer)) or {}
            secondary = approx_by_layer.get(str(args.secondary_layer)) or {}
            layer_agreement = aj.token_agreement(
                primary.get("approx_j_top1_number_token_id"),
                secondary.get("approx_j_top1_number_token_id"),
            )
            model_agreement = aj.token_agreement(primary.get("approx_j_top1_number_token_id"), model_id)
            risk = aj.compute_readout_risk(
                primary,
                prefix="approx_j",
                layer_agreement=layer_agreement,
                model_answer_agreement=model_agreement,
            )
            rows_by_epsilon[str(float(epsilon))].append(risk)
        feature_rows.append(
            {
                "example_id": example["example_id"],
                "wrong_label": int(example["wrong_label"]),
                "direct_readout_risk": direct_risk,
                "final_logits_number_margin": (example.get("final_logits_features") or {}).get("final_logits_number_margin"),
            }
        )
    final_risk = [1.0 - v if v is not None else None for v in mra.minmax_normalize(final_margins)]
    random_scores = mra.random_baseline_scores(len(labels), seed=args.seed)
    direct_eval = mra.evaluate_correct_wrong_detection(labels, direct_scores)
    final_eval = mra.evaluate_correct_wrong_detection(labels, final_risk)
    random_eval = mra.evaluate_correct_wrong_detection(labels, random_scores)

    per_epsilon = {}
    for epsilon, scores in rows_by_epsilon.items():
        approx_eval = mra.evaluate_correct_wrong_detection(labels, scores)
        per_epsilon[epsilon] = {
            "epsilon": float(epsilon),
            "direct_readout_AUROC": direct_eval.get("auroc"),
            "direct_readout_AUPRC": direct_eval.get("auprc"),
            "approx_j_readout_AUROC": approx_eval.get("auroc"),
            "approx_j_readout_AUPRC": approx_eval.get("auprc"),
            "final_logits_margin_AUROC": final_eval.get("auroc"),
            "final_logits_margin_AUPRC": final_eval.get("auprc"),
            "random_AUROC": random_eval.get("auroc"),
            "random_AUPRC": random_eval.get("auprc"),
            "auroc_delta_approx_minus_direct": delta(approx_eval.get("auroc"), direct_eval.get("auroc")),
            "auroc_delta_approx_minus_final_logits": delta(approx_eval.get("auroc"), final_eval.get("auroc")),
        }
    return {
        "backend": aj.BACKEND,
        "positive_label": "wrong_answer_case",
        "num_examples": len(examples),
        "primary_layer": int(args.primary_layer),
        "secondary_layer": int(args.secondary_layer),
        "per_epsilon": per_epsilon,
        "direct_readout_detection": direct_eval,
        "final_logits_margin_baseline": final_eval,
        "random_baseline": random_eval,
        "feature_boundary": "direct/approx risk features are gold-free; gold/wrong token ranks are eval-only.",
        "approx_j_changes_3C3_judgment": infer_judgment_change(per_epsilon),
        "note": "Approximate J-lens is not required to beat final logits; this report checks whether it changes the direct readout interpretation.",
    }


def delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def infer_judgment_change(per_epsilon: dict[str, dict[str, Any]]) -> bool:
    deltas = [row.get("auroc_delta_approx_minus_direct") for row in per_epsilon.values()]
    clean = [float(d) for d in deltas if aj.finite(d)]
    return bool(clean and max(clean) > 0.05)


def build_epsilon_stability_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_example_layer = defaultdict(list)
    for row in rows:
        by_example_layer[(row["example_id"], int(row["layer"]))].append(row)
    pair_rows = []
    for (example_id, layer), group in by_example_layer.items():
        by_eps = {float(r["epsilon"]): r for r in group}
        for eps_a, eps_b in combinations(sorted(by_eps), 2):
            a = by_eps[eps_a]
            b = by_eps[eps_b]
            top_a = [int(t["token_id"]) for t in a.get("approx_j_top_k_number_tokens", [])]
            top_b = [int(t["token_id"]) for t in b.get("approx_j_top_k_number_tokens", [])]
            score_a = {int(t["token_id"]): t.get("approx_j_score") for t in a.get("candidate_token_scores", [])}
            score_b = {int(t["token_id"]): t.get("approx_j_score") for t in b.get("candidate_token_scores", [])}
            candidates = sorted(set(score_a) & set(score_b))
            pair_rows.append(
                {
                    "example_id": example_id,
                    "layer": layer,
                    "epsilon_a": eps_a,
                    "epsilon_b": eps_b,
                    "top1_same": bool(top_a[:1] and top_b[:1] and top_a[0] == top_b[0]),
                    "top5_overlap_rate": aj.topk_overlap(top_a, top_b, k=5)["overlap_rate"],
                    "spearman_candidate_subset": aj.spearman_correlation_from_maps(score_a, score_b, candidates),
                }
            )
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(int(row["layer"]), float(row["epsilon_a"]), float(row["epsilon_b"]))].append(row)
    per_pair = {}
    for (layer, eps_a, eps_b), group in sorted(grouped.items()):
        per_pair[f"L{layer}|eps{eps_a}_vs_{eps_b}"] = {
            "layer": layer,
            "epsilon_a": eps_a,
            "epsilon_b": eps_b,
            "num_rows": len(group),
            "top1_same_rate": sum(1 for r in group if r["top1_same"]) / len(group) if group else 0.0,
            "top5_overlap_rate_mean": aj.mean([r.get("top5_overlap_rate") for r in group]),
            "spearman_candidate_subset": aj.bootstrap_ci(
                [r.get("spearman_candidate_subset") for r in group],
                seed_tag=f"eps-stability:{layer}:{eps_a}:{eps_b}",
            ),
        }
    return {"backend": aj.BACKEND, "per_layer_epsilon_pair": per_pair}


def build_cases(rows: list[dict[str, Any]], examples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    example_by_id = {e["example_id"]: e for e in examples}
    primary_eps = min(args.epsilons, key=lambda e: abs(float(e) - 0.03))
    candidates = [
        r for r in rows
        if int(r["layer"]) == int(args.primary_layer) and abs(float(r["epsilon"]) - float(primary_eps)) < 1e-12
    ]
    def score(row: dict[str, Any]) -> float:
        spearman = row.get("rank_correlation_spearman_all_number_tokens")
        overlap = ((row.get("topk_overlap") or {}).get("10") or {}).get("overlap_rate")
        return (float(spearman) if aj.finite(spearman) else -1.0) + (float(overlap) if aj.finite(overlap) else 0.0)
    successes = sorted(candidates, key=score, reverse=True)[:30]
    failures = sorted(candidates, key=score)[:30]
    return [case_row(r, example_by_id, "high_direct_approx_agreement") for r in successes], [
        case_row(r, example_by_id, "low_direct_approx_agreement") for r in failures
    ]


def case_row(row: dict[str, Any], example_by_id: dict[str, dict[str, Any]], case_type: str) -> dict[str, Any]:
    example = example_by_id.get(row["example_id"], {})
    return {
        "case_type": case_type,
        "example_id": row["example_id"],
        "pair_id": row["pair_id"],
        "trace_kind": row["trace_kind"],
        "question": example.get("question"),
        "gold_answer_eval_only": example.get("gold_answer_eval_only"),
        "model_final_answer": example.get("model_final_answer"),
        "wrong_label": row.get("wrong_label"),
        "layer": row.get("layer"),
        "epsilon": row.get("epsilon"),
        "spearman_all_number_tokens": row.get("rank_correlation_spearman_all_number_tokens"),
        "top10_overlap_rate": ((row.get("topk_overlap") or {}).get("10") or {}).get("overlap_rate"),
        "direct_top1": (row.get("direct_top_k_number_tokens") or [{}])[0],
        "approx_j_top1": (row.get("approx_j_top_k_number_tokens") or [{}])[0],
        "final_logits_top1": (row.get("final_logits_top_k_number_tokens") or [{}])[0],
    }


def render_review_gate(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    topk: dict[str, Any],
    rank: dict[str, Any],
    ordering: dict[str, Any],
    diagnostic: dict[str, Any],
    epsilon: dict[str, Any],
) -> str:
    primary_eps = min(args.epsilons, key=lambda e: abs(float(e) - 0.03))
    primary_key = f"L{args.primary_layer}|eps{float(primary_eps)}"
    top_cell = (topk.get("per_layer_epsilon") or {}).get(primary_key) or {}
    rank_cell = (rank.get("per_layer_epsilon") or {}).get(primary_key) or {}
    spearman_mean = ((rank_cell.get("spearman_all_number_tokens") or {}).get("mean"))
    top10 = top_cell.get("top10_overlap_rate_mean")
    diag_cell = (diagnostic.get("per_epsilon") or {}).get(str(float(primary_eps))) or {}
    approx_delta = diag_cell.get("auroc_delta_approx_minus_direct")
    approx_better = approx_delta is not None and approx_delta > 0.05
    direct_usable = (spearman_mean is not None and spearman_mean >= 0.4) or (top10 is not None and top10 >= 0.3)
    both_weak = not direct_usable and not approx_better
    if direct_usable and not approx_better:
        verdict = "Case A: direct logit-lens and directional approximate J-lens broadly agree; direct unembedding remains an acceptable low-cost readout approximation for small diagnostic use."
    elif approx_better:
        verdict = "Case B: directional approximate J-lens improves over direct readout; future detection-only work should prefer approximate J-lens features."
    elif both_weak:
        verdict = "Case C: both direct and approximate readouts are weak under this check; keep the mechanistic MLP site result but limit detection claims."
    else:
        verdict = "Mixed: approximate J-lens does not clearly settle the direct-readout reliability question."

    questions = [
        ("Is this only a readout sanity check?", True),
        ("Did it avoid steering, nudge, patching claims, training, and LoRA?", True),
        ("Did it reuse the 3C-3 / 3C-0-Fix corrected-pair substrate?", len(examples) > 0),
        ("Were layers 20 and 24 covered?", {20, 24}.issubset(set(int(l) for l in args.layers))),
        ("Was an epsilon sweep run?", len(args.epsilons) >= 2),
        ("Is this labelled directional approximate J-lens, not full J-lens?", True),
        ("Were direct logit-lens and approximate J-lens top-k overlap measured?", bool(top_cell)),
        ("Were rank correlations measured?", bool(rank_cell)),
        ("Were gold/wrong token orderings kept eval-only?", True),
        ("Does approximate J-lens change the 3C-3 judgement?", bool(diagnostic.get("approx_j_changes_3C3_judgment"))),
        ("Is approximate J-lens clearly better than direct readout?", bool(approx_better)),
        ("Is final-logits margin kept as a baseline rather than a target to beat?", True),
        ("Is the result ready for 2000 rerun?", False),
        ("Should full Sprint 3C be entered?", False),
        ("Is hallucination reduction proven?", False),
        ("Is answer accuracy improvement proven?", False),
        ("Was the 3C-3 output directory left untouched?", True),
        ("Should the artifact manifest record this gitignored output directory?", True),
    ]
    lines = [
        "# Sprint 3C-4A Approximate J-lens Readout Sanity Check Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "- steering_continued: false",
        f"- interpretation: {verdict}",
        "",
        "Primary cell:",
        f"- cell: {primary_key}",
        f"- top1_match_rate: {top_cell.get('top1_match_rate')}",
        f"- top10_overlap_rate_mean: {top10}",
        f"- spearman_all_number_tokens_mean: {spearman_mean}",
        f"- approx_minus_direct_AUROC: {approx_delta}",
        "",
        "Required review questions:",
    ]
    for question, answer in questions:
        lines.append(f"- {question} {str(bool(answer)).lower()}")
    lines.extend(
        [
            "",
            "Boundary:",
            "- This is a finite-difference readout-method check. It does not reproduce full Workspace J-lens.",
            "- Gold and wrong answer token ranks are eval-only diagnostics and are not used as feature inputs.",
            "- No claim is made about answer accuracy improvement or hallucination reduction.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_config(args: argparse.Namespace, pairs: list[dict[str, Any]], examples: list[dict[str, Any]], number_token_ids: list[int]) -> dict[str, Any]:
    return {
        "backend": aj.BACKEND,
        "model_path": args.model_path,
        "input_dir": args.input_dir,
        "fix_input_dir": args.fix_input_dir,
        "pair_source_dir": args.pair_source_dir,
        "output_dir": args.output_dir,
        "num_pairs": len(pairs),
        "num_trace_examples": len(examples),
        "layers": list(args.layers),
        "primary_layer": int(args.primary_layer),
        "secondary_layer": int(args.secondary_layer),
        "epsilons": [float(e) for e in args.epsilons],
        "top_k": int(args.top_k),
        "num_number_like_tokens": len(number_token_ids),
        "method": "directional approximate J-lens: delta_logits / epsilon from mlp_output += epsilon * ||m|| * unit(m)",
        "full_j_lens_reproduced": False,
        "training": False,
        "steering_or_nudge": False,
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
        "steering_continued": False,
    }


def write_preflight(args: argparse.Namespace, out_dir: Path) -> None:
    input_dir = Path(args.input_dir)
    fix_dir = Path(args.fix_input_dir)
    story = PROJECT_ROOT / "docs/reference/STORY.md"
    story_text = story.read_text(encoding="utf-8", errors="replace") if story.exists() else ""
    progress_text = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    required_3c3 = [
        "mlp_readout_attribution_manifest.jsonl",
        "mlp_attribution_config.json",
        "mlp_unembedding_projection_report.json",
        "answer_token_attribution_report.json",
        "correct_wrong_detection_report.json",
        "risk_score_report.json",
        "baseline_comparison_report.json",
        "calibration_report.json",
        "review_gate_mlp_readout_attribution_probe.md",
    ]
    task_status = run_git(["status", "--short", "--", TASK_CARD])
    outputs_ignored = run_git(
        ["check-ignore", "-q", "outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/preflight_report.md"],
        check=False,
    ).returncode == 0
    lines = [
        "# Sprint 3C-4A Preflight",
        "",
        "Checks:",
        f"- progress_records_3C3R: {str('Sprint 3C-3R' in progress_text).lower()}",
        f"- story_exists: {str(story.exists()).lower()}",
        f"- story_records_3C3_completed: {str('3C-3' in story_text and 'MLP' in story_text).lower()}",
        f"- story_records_mlp_risk_beats_random: {str('random' in story_text.lower()).lower()}",
        f"- story_records_mlp_not_final_logits_superior: {str('final-logits' in story_text.lower() or 'final logits' in story_text.lower()).lower()}",
        f"- story_records_nla_l20_boundary: {str('NLA' in story_text and 'L20' in story_text).lower()}",
        f"- story_records_j_lens_priority_sanity_check: {str('J-lens' in story_text and 'sanity check' in story_text.lower()).lower()}",
        f"- input_dir_exists: {str(input_dir.exists()).lower()}",
        f"- all_required_3C3_reports_exist: {str(all((input_dir / name).exists() for name in required_3c3)).lower()}",
        f"- corrected_pair_manifest_exists: {str((fix_dir / 'corrected_pair_manifest.jsonl').exists()).lower()}",
        f"- will_not_overwrite_3C3_output_dir: {str(out_dir.resolve() != input_dir.resolve()).lower()}",
        "- will_not_do_steering_patching_or_nudge_claims: true",
        "- will_not_do_2000_full_3C_or_training: true",
        f"- outputs_gitignored: {str(outputs_ignored).lower()}",
        "- artifact_manifest_will_be_updated: true",
        f"- task_card_pre_existing_AM_status_observed: {str(task_status.strip().startswith('AM')).lower()}",
        "",
        "Boundary:",
        "- finite-difference directional approximate J-lens only; not full J-lens.",
        "- ready_for_2000_rerun=false; do_not_enter_full_sprint_3C=true.",
        "- hallucination_reduction_proven=false; answer_accuracy_improvement_proven=false; steering_continued=false.",
        "",
        "Current task card git status:",
        "```text",
        task_status.strip() or "(not modified)",
        "```",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incomplete(out_dir: Path, reason: str) -> None:
    payload = {
        "backend": aj.BACKEND,
        "incomplete_evidence_reason": reason,
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
    }
    for name in [
        "approx_j_lens_config.json",
        "topk_overlap_report.json",
        "rank_correlation_report.json",
        "token_ordering_report.json",
        "diagnostic_comparison_report.json",
        "epsilon_stability_report.json",
    ]:
        write_json(payload, out_dir / name)
    for name in [
        "approx_j_lens_forward_manifest.jsonl",
        "direct_vs_approx_readout_manifest.jsonl",
        "failure_case_report.jsonl",
        "success_case_report.jsonl",
    ]:
        if not (out_dir / name).exists():
            write_jsonl([], out_dir / name)
    (out_dir / "review_gate_approx_j_lens_readout_sanity_check.md").write_text(
        "# Sprint 3C-4A Review Gate\n\nIncomplete evidence: "
        + reason
        + "\n\n- ready_for_2000_rerun: false\n- do_not_enter_full_sprint_3C: true\n",
        encoding="utf-8",
    )


def run_git(args: list[str], *, check: bool = True):
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if check:
        return (result.stdout or "") + (result.stderr or "")
    return result


if __name__ == "__main__":
    main()
