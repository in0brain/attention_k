"""Sprint 3C-2: MLP readout direction analysis and donor-free nudge.

Extracts the correct-minus-wrong MLP-output delta direction at the final-answer
readout position (the Sprint 3C-1 causal site), analyses its geometry and
alignment to the gold-vs-wrong unembedding direction, then applies leave-one-out
donor-free directional nudges (mean / PC1 / gold-unembed vs random / shuffled /
negative / zero controls) and scores them with the Sprint 3C-0-Fix corrected
answer-sequence proxy plus a minimal first-step generation-style check.

Boundary: reuse 3C-0/3C-1 pairs (no re-sampling); no training, no LoRA, no
deployable steering claim, no full Sprint 3C / 2000 rerun, no accuracy or
hallucination claim. Teacher-forced (+ first-step) single-forward proxy only.
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
from recover_attention import mlp_readout_direction as mrd  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

DEFAULT_3C1_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing"
DEFAULT_FIX_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_fix_answer_proxy_recheck"
DEFAULT_3C0_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_2_mlp_readout_direction_analysis"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"

DIRECTION_TYPES = ["mean_delta", "pc1_delta", "gold_unembed", "random", "shuffled", "negative", "zero"]
DONOR_FREE = {"mean_delta", "pc1_delta"}  # the directions we claim as donor-free steering candidates


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    for guard in (Path(args.input_dir), DEFAULT_3C0_DIR, DEFAULT_FIX_DIR):
        if guard.resolve() == out_dir.resolve():
            raise SystemExit("refusing to overwrite a prior sprint directory")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    write_preflight(args, out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-2 MLP readout direction analysis")
    parser.add_argument("--input-dir", default=str(DEFAULT_3C1_DIR))
    parser.add_argument("--fix-input-dir", default=str(DEFAULT_FIX_DIR))
    parser.add_argument("--pair-source-dir", default=str(DEFAULT_3C0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--directions", nargs="+", default=DIRECTION_TYPES)
    parser.add_argument("--seed", type=int, default=33060)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    import torch

    pair_ids = load_pair_ids(Path(args.fix_input_dir), Path(args.input_dir))
    all_pairs = {p["pair_id"]: p for p in read_jsonl(Path(args.pair_source_dir) / "correct_wrong_pair_manifest.jsonl")}
    pairs = [all_pairs[pid] for pid in pair_ids if pid in all_pairs] or list(all_pairs.values())
    if not pairs:
        write_incomplete(out_dir, "no reusable correct/wrong pairs found")
        return

    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    print(f"[3C-2] loaded model; pairs={len(pairs)} layers={args.layers} alphas={args.alphas}")

    state = collect_state(context, tokenizer, pairs, args)
    write_jsonl([delta_row(s, args.layers) for s in state], out_dir / "mlp_readout_delta_manifest.jsonl")
    if len(state) < 5:
        write_incomplete(out_dir, f"only {len(state)} usable pairs; need >=5")
        return

    geometry = build_geometry(state, args.layers)
    write_json(geometry, out_dir / "mlp_direction_geometry_report.json")
    alignment = build_unembedding_alignment(state, args.layers)
    write_json(alignment, out_dir / "mlp_unembedding_alignment_report.json")

    directions = build_loo_directions(state, args, torch)
    write_jsonl(direction_manifest(state, directions, args), out_dir / "donor_free_direction_manifest.jsonl")

    rows, gen_rows = run_nudge_forwards(context, tokenizer, state, directions, args)
    write_jsonl(rows, out_dir / "donor_free_nudge_forward_manifest.jsonl")

    nudge_report = build_nudge_report(rows, args)
    survival = build_survival_report(gen_rows, args)
    harm = build_harm(rows, args)
    success, failure = build_cases(rows, state)
    review = render_review_gate(state, args, geometry, alignment, nudge_report, survival)

    write_json(build_config(args, len(pairs), len(state)), out_dir / "mlp_direction_config.json")
    write_json(nudge_report, out_dir / "donor_free_nudge_report.json")
    write_json(survival, out_dir / "generation_survival_report.json")
    write_json(harm, out_dir / "harm_control_report.json")
    write_jsonl(success, out_dir / "success_case_report.jsonl")
    write_jsonl(failure, out_dir / "failure_case_report.jsonl")
    (out_dir / "review_gate_mlp_readout_direction_analysis.md").write_text(review, encoding="utf-8")
    print(f"[3C-2] complete: pairs={len(state)} nudge_rows={len(rows)} out={out_dir}")


def load_pair_ids(fix_dir: Path, c1_dir: Path) -> list[str]:
    for path in (fix_dir / "corrected_pair_manifest.jsonl", c1_dir / "module_patch_pair_manifest.jsonl"):
        if path.exists():
            return [str(r["pair_id"]) for r in read_jsonl(path)]
    return []


def collect_state(context, tokenizer, pairs, args) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    t0 = time.time()
    for index, pair in enumerate(pairs, start=1):
        correct = mrd.extract_mlp_readout(context, pair["correct_trace_text"], args.layers)
        wrong = mrd.extract_mlp_readout(context, pair["wrong_trace_text"], args.layers)
        if correct is None or wrong is None:
            continue
        gold_ids = apm.answer_token_ids(tokenizer, pair["gold_answer"])
        wrong_ids = apm.answer_token_ids(tokenizer, wrong["readout"]["span"]["answer"])
        prefix_ids = wrong["readout"]["prefix_ids"]
        if not gold_ids or not wrong_ids or not prefix_ids:
            continue
        unembed = mrd.unembedding_answer_direction(context["model"], tokenizer, pair["gold_answer"], wrong["readout"]["span"]["answer"])
        deltas = mrd.compute_correct_wrong_delta(correct["vectors"], wrong["vectors"], args.layers)
        if not deltas:
            continue
        state.append({
            "pair": pair,
            "correct_vecs": correct["vectors"],
            "wrong_vecs": wrong["vectors"],
            "deltas": deltas,
            "unembed_dir": unembed,
            "prefix_ids": prefix_ids,
            "gold_ids": gold_ids,
            "wrong_ids": wrong_ids,
            "wrong_readout_pos": wrong["readout"]["readout_position"],
        })
        if args.report_every and index % args.report_every == 0:
            print(f"[3C-2] collected={index}/{len(pairs)} rate={index/max(time.time()-t0,1e-9):.3f}/s")
    return state


def build_geometry(state, layers) -> dict[str, Any]:
    per_layer = {}
    for layer in layers:
        deltas = [s["deltas"][layer] for s in state if layer in s["deltas"]]
        shuffled = shuffled_deltas(state, layer)
        per_layer[str(layer)] = {
            "num_deltas": len(deltas),
            "mean_delta_norm": mrd.mean([float(d.norm().item()) for d in deltas]),
            "pairwise_cosine_mean_true": mrd.pairwise_cosine_mean(deltas),
            "pairwise_cosine_mean_shuffled": mrd.pairwise_cosine_mean(shuffled) if shuffled else None,
            "explained_variance_ratio_top3": mrd.explained_variance_ratio(deltas) if len(deltas) >= 3 else None,
            "pc1_loo_stability_mean_cosine": pc1_loo_stability(deltas),
        }
    return {"backend": mrd.BACKEND, "per_layer": per_layer,
            "note": "pairwise_cosine_mean_true >> shuffled and low-rank explained variance indicate a stable shared direction."}


def pc1_loo_stability(deltas) -> float | None:
    if len(deltas) < 4:
        return None
    full = mrd.pca_direction(deltas)
    sims = []
    for i in range(len(deltas)):
        loo = mrd.pca_direction([d for j, d in enumerate(deltas) if j != i], reference=full)
        sims.append(abs(mrd.cosine(loo, full)))
    return mrd.mean(sims)


def shuffled_deltas(state, layer):
    items = [s for s in state if layer in s["correct_vecs"] and layer in s["wrong_vecs"]]
    if len(items) < 2:
        return []
    perm = mrd.rotate_derangement(len(items))
    return [items[i]["correct_vecs"][layer].float() - items[perm[i]]["wrong_vecs"][layer].float() for i in range(len(items))]


def build_unembedding_alignment(state, layers) -> dict[str, Any]:
    per_layer = {}
    for layer in layers:
        cosines = [mrd.cosine(s["deltas"][layer], s["unembed_dir"]) for s in state if layer in s["deltas"] and s["unembed_dir"] is not None]
        cosines = [c for c in cosines if c == c]  # drop NaN
        boot = mrd.bootstrap_ci(cosines, seed=ap.stable_int_seed(f"align:{layer}"))
        mean_delta = mrd.mean_direction([s["deltas"][layer] for s in state if layer in s["deltas"]])
        mean_unembed = mrd.mean_direction([s["unembed_dir"] for s in state if s["unembed_dir"] is not None])
        per_layer[str(layer)] = {
            "num_pairs": len(cosines),
            "per_pair_cosine_mean": boot.get("mean"),
            "per_pair_cosine_ci95": [boot.get("ci95_low"), boot.get("ci95_high")],
            "per_pair_cosine_stable_positive": boot.get("stable_positive"),
            "mean_delta_vs_mean_unembed_cosine": mrd.cosine(mean_delta, mean_unembed),
        }
    return {"backend": mrd.BACKEND, "per_layer": per_layer,
            "note": "gold-vs-wrong unembedding alignment is an eval-only interpretive metric; W_U rows approximate logit-space, so cosine is a rough readout-direction proxy."}


def build_loo_directions(state, args, torch) -> dict[str, dict[int, dict[str, Any]]]:
    """directions[pair_index][layer][direction_type] = unit vector."""

    rng = __import__("numpy").random.default_rng(args.seed)
    random_by_layer = {}
    for layer in args.layers:
        dim = next((s["deltas"][layer].shape[0] for s in state if layer in s["deltas"]), None)
        random_by_layer[layer] = mrd.normalize_direction(torch.tensor(rng.standard_normal(dim), dtype=torch.float32)) if dim else None

    directions: dict[int, dict[int, dict[str, Any]]] = {}
    for i, s in enumerate(state):
        directions[i] = {}
        for layer in args.layers:
            if layer not in s["deltas"]:
                continue
            others = [t["deltas"][layer] for j, t in enumerate(state) if j != i and layer in t["deltas"]]
            if not others:
                continue
            mean_i = mrd.mean_direction(others)
            pc1_i = mrd.pca_direction(others, reference=mean_i) if len(others) >= 3 else mean_i
            shuffled_all = shuffled_deltas([t for j, t in enumerate(state) if j != i], layer)
            shuffled_i = mrd.mean_direction(shuffled_all) if shuffled_all else mean_i
            directions[i][layer] = {
                "mean_delta": mrd.normalize_direction(mean_i),
                "pc1_delta": mrd.normalize_direction(pc1_i),
                "negative": mrd.normalize_direction(-mean_i),
                "shuffled": mrd.normalize_direction(shuffled_i),
                "gold_unembed": mrd.normalize_direction(s["unembed_dir"]) if s["unembed_dir"] is not None else None,
                "random": random_by_layer.get(layer),
                "zero": torch.zeros_like(mean_i),
            }
    return directions


def run_nudge_forwards(context, tokenizer, state, directions, args):
    rows: list[dict[str, Any]] = []
    gen_rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    for i, s in enumerate(state):
        pair = s["pair"]
        prefix_ids, gold_ids, wrong_ids = s["prefix_ids"], s["gold_ids"], s["wrong_ids"]
        base_gold = mrd.sequence_logprob_with_direction_nudge(context, prefix_ids=prefix_ids, answer_ids=gold_ids, return_slot_logits=True)
        base_wrong = mrd.sequence_logprob_with_direction_nudge(context, prefix_ids=prefix_ids, answer_ids=wrong_ids)
        base_slot = base_gold["answer_slot_logits"]
        base_first = mrd.first_step_logits_with_nudge(context, prefix_ids=prefix_ids)
        gold_first, wrong_first = s["gold_ids"][0], s["wrong_ids"][0]
        base_first_gold = ap.logprob_of(base_first, gold_first)
        base_first_wrong = ap.logprob_of(base_first, wrong_first)

        for layer in args.layers:
            dmap = directions.get(i, {}).get(layer)
            if not dmap:
                continue
            for direction_type in args.directions:
                unit = dmap.get(direction_type)
                if unit is None:
                    continue
                for alpha in args.alphas:
                    nudge = {"layer": int(layer), "unit_direction": unit, "alpha": float(alpha), "target_position": int(s["wrong_readout_pos"])}
                    pg = mrd.sequence_logprob_with_direction_nudge(context, prefix_ids=prefix_ids, answer_ids=gold_ids, nudge=nudge, return_slot_logits=True)
                    pw = mrd.sequence_logprob_with_direction_nudge(context, prefix_ids=prefix_ids, answer_ids=wrong_ids, nudge=nudge)
                    rows.append(nudge_row(pair, direction_type, layer, alpha, base_gold, base_wrong, pg, pw, base_slot))
                    pf = mrd.first_step_logits_with_nudge(context, prefix_ids=prefix_ids, nudge=nudge)
                    gen_rows.append(gen_row(pair, direction_type, layer, alpha, base_first_gold, base_first_wrong, pf, gold_first, wrong_first))
                    done += 1
                    if args.report_every and done % args.report_every == 0:
                        print(f"[3C-2] nudge_forwards={done} rate={done/max(time.time()-t0,1e-9):.3f}/s")
    return rows, gen_rows


def nudge_row(pair, direction_type, layer, alpha, base_gold, base_wrong, pg, pw, base_slot):
    gold_delta = pg["logprob"] - base_gold["logprob"]
    wrong_delta = pw["logprob"] - base_wrong["logprob"]
    clean = apm.compute_corrected_clean_direction(gold_delta, wrong_delta)
    shift = abs_mod.compute_answer_position_output_shift(base_slot, pg["answer_slot_logits"])
    harm = bool(ap.compute_harm(shift) or (gold_delta is not None and gold_delta < -0.5) or (clean is not None and clean < -0.5))
    return {"backend": mrd.BACKEND, "pair_id": pair["pair_id"], "direction_type": direction_type, "layer": int(layer), "alpha": float(alpha),
            "gold_seq_logprob_delta": gold_delta, "wrong_seq_logprob_delta": wrong_delta,
            "corrected_clean_direction_score": clean, "output_shift": shift, "harm": harm}


def gen_row(pair, direction_type, layer, alpha, base_gold, base_wrong, patched_logits, gold_first, wrong_first):
    pg = ap.logprob_of(patched_logits, gold_first)
    pw = ap.logprob_of(patched_logits, wrong_first)
    gd = pg - base_gold if _finite(pg, base_gold) else None
    wd = pw - base_wrong if _finite(pw, base_wrong) else None
    clean = apm.compute_corrected_clean_direction(gd, wd)
    import torch
    top1 = int(torch.tensor(patched_logits).argmax().item())
    return {"backend": mrd.BACKEND, "pair_id": pair["pair_id"], "direction_type": direction_type, "layer": int(layer), "alpha": float(alpha),
            "first_step_gold_logprob_delta": gd, "first_step_wrong_logprob_delta": wd, "first_step_clean_direction": clean,
            "top1_is_gold_first": bool(top1 == gold_first)}


def _finite(a, b):
    import math
    return math.isfinite(float(a)) and math.isfinite(float(b))


def paired_delta(rows, *, treatment, control, key_fields, score_key, seed_tag):
    t, c = {}, {}
    for r in rows:
        v = r.get(score_key)
        if v is None:
            continue
        key = tuple(r[f] for f in key_fields)
        if r["direction_type"] == treatment:
            t[key] = float(v)
        if r["direction_type"] == control:
            c[key] = float(v)
    deltas = [t[k] - c[k] for k in t.keys() & c.keys()]
    out = ap.bootstrap_ci(deltas, seed=ap.stable_int_seed(seed_tag))
    out["stable_positive"] = bool(out.get("ci95_low") is not None and out["ci95_low"] > 0)
    return out


def build_nudge_report(rows, args) -> dict[str, Any]:
    by_cell = {}
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["direction_type"], int(r["layer"]), float(r["alpha"]))].append(r)
    for (dt, layer, alpha), grp in sorted(grouped.items()):
        by_cell[f"{dt}|L{layer}|a{alpha}"] = {
            "direction_type": dt, "layer": layer, "alpha": alpha, "num_records": len(grp),
            "mean_clean_direction": mrd.mean([r["corrected_clean_direction_score"] for r in grp]),
            "mean_gold_seq_delta": mrd.mean([r["gold_seq_logprob_delta"] for r in grp]),
            "mean_wrong_seq_delta": mrd.mean([r["wrong_seq_logprob_delta"] for r in grp]),
            "harm_rate": _rate(grp, lambda r: r.get("harm")),
            "clean_direction_bootstrap": mrd.bootstrap_ci([r["corrected_clean_direction_score"] for r in grp], seed=ap.stable_int_seed(f"cell:{dt}:{layer}:{alpha}")),
        }
    key = ("pair_id", "layer", "alpha")
    comparisons = {}
    for treat, ctrl in [("mean_delta", "random"), ("pc1_delta", "random"), ("mean_delta", "shuffled"), ("mean_delta", "negative"), ("gold_unembed", "random")]:
        comparisons[f"{treat}_minus_{ctrl}"] = paired_delta(rows, treatment=treat, control=ctrl, key_fields=key, score_key="corrected_clean_direction_score", seed_tag=f"cmp:{treat}:{ctrl}")
    return {"backend": mrd.BACKEND, "metric": "corrected_answer_sequence_clean_direction",
            "by_direction_layer_alpha": by_cell, "control_comparisons": comparisons,
            "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True,
            "hallucination_reduction_proven": False, "answer_accuracy_improvement_proven": False}


def build_survival_report(gen_rows, args) -> dict[str, Any]:
    grouped = defaultdict(list)
    for r in gen_rows:
        grouped[(r["direction_type"], int(r["layer"]), float(r["alpha"]))].append(r)
    by_cell = {}
    for (dt, layer, alpha), grp in sorted(grouped.items()):
        by_cell[f"{dt}|L{layer}|a{alpha}"] = {
            "direction_type": dt, "layer": layer, "alpha": alpha, "num_records": len(grp),
            "mean_first_step_clean_direction": mrd.mean([r["first_step_clean_direction"] for r in grp]),
            "first_step_clean_bootstrap": mrd.bootstrap_ci([r["first_step_clean_direction"] for r in grp], seed=ap.stable_int_seed(f"gen:{dt}:{layer}:{alpha}")),
            "top1_gold_rate": _rate(grp, lambda r: r.get("top1_is_gold_first")),
        }
    return {"backend": mrd.BACKEND, "note": "first-step generation-style proxy; not accuracy or generation success.",
            "by_direction_layer_alpha": by_cell}


def build_harm(rows, args) -> dict[str, Any]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["direction_type"], int(r["layer"]), float(r["alpha"]))].append(r)
    return {"backend": mrd.BACKEND,
            "harm_proxy_definition": "top1_changed or entropy_delta>1.0 or margin_delta<-0.25 or gold_seq_delta<-0.5 or clean_direction<-0.5",
            "aggregate": [{"direction_type": dt, "layer": l, "alpha": a, "num_records": len(g), "harm_rate": _rate(g, lambda r: r.get("harm"))}
                          for (dt, l, a), g in sorted(grouped.items())]}


def build_cases(rows, state):
    pair_by_id = {s["pair"]["pair_id"]: s for s in state}
    donor_free = [r for r in rows if r["direction_type"] in DONOR_FREE]
    success = sorted([r for r in donor_free if (r.get("corrected_clean_direction_score") or -1) > 0 and not r.get("harm")], key=lambda r: r.get("corrected_clean_direction_score") or -1, reverse=True)
    failure = sorted([r for r in donor_free if (r.get("corrected_clean_direction_score") or 0) <= 0 or r.get("harm")], key=lambda r: r.get("corrected_clean_direction_score") or 0)

    def case(r, tag):
        pair = pair_by_id.get(r["pair_id"], {}).get("pair", {})
        return {"case_type": tag, "pair_id": r["pair_id"], "question": pair.get("question"), "gold_answer": pair.get("gold_answer"),
                "direction_type": r["direction_type"], "layer": r["layer"], "alpha": r["alpha"],
                "corrected_clean_direction_score": r.get("corrected_clean_direction_score"), "harm": r.get("harm")}

    return ([case(r, "donor_free_selective_low_harm") for r in success[:30]], [case(r, "non_positive_or_harmful") for r in failure[:30]])


def delta_row(s, layers):
    return {"pair_id": s["pair"]["pair_id"], "gold_answer": s["pair"]["gold_answer"],
            "wrong_answer_extracted": s["pair"].get("wrong_answer"),
            "delta_norm_by_layer": {str(l): float(s["deltas"][l].norm().item()) for l in layers if l in s["deltas"]},
            "wrong_readout_position": s["wrong_readout_pos"]}


def direction_manifest(state, directions, args):
    out = []
    for i, s in enumerate(state):
        for layer in args.layers:
            dmap = directions.get(i, {}).get(layer)
            if not dmap:
                continue
            row = {"pair_id": s["pair"]["pair_id"], "layer": layer}
            for dt, unit in dmap.items():
                if unit is None:
                    row[f"cos_{dt}_vs_mean_delta"] = None
                    continue
                row[f"cos_{dt}_vs_mean_delta"] = mrd.cosine(unit, dmap["mean_delta"]) if dmap.get("mean_delta") is not None else None
            out.append(row)
    return out


def render_review_gate(state, args, geometry, alignment, nudge_report, survival) -> str:
    cmp = nudge_report["control_comparisons"]
    def flag(name):
        return bool(cmp.get(name, {}).get("stable_positive"))
    mean_beats_random = flag("mean_delta_minus_random")
    pc1_beats_random = flag("pc1_delta_minus_random")
    gold_beats_random = flag("gold_unembed_minus_random")
    mean_beats_shuffled = flag("mean_delta_minus_shuffled")
    mean_beats_negative = flag("mean_delta_minus_negative")

    # low-harm donor-free winner: a mean/pc1 cell with clean>0 (stable) and harm<=0.2
    winners = []
    for key, cell in nudge_report["by_direction_layer_alpha"].items():
        if cell["direction_type"] in DONOR_FREE:
            b = cell["clean_direction_bootstrap"]
            if (b.get("ci95_low") is not None and b["ci95_low"] > 0) and cell["harm_rate"] <= 0.2:
                winners.append({"cell": key, "clean": cell["mean_clean_direction"], "harm": cell["harm_rate"], "ci95_low": b["ci95_low"]})
    winners.sort(key=lambda w: (w["harm"], -(w["clean"] or 0)))

    survivors = []
    for key, cell in survival["by_direction_layer_alpha"].items():
        if cell["direction_type"] in DONOR_FREE:
            b = cell["first_step_clean_bootstrap"]
            if b.get("ci95_low") is not None and b["ci95_low"] > 0:
                survivors.append(key)

    # High-dim residual deltas have low raw pairwise cosine even when they share a
    # stable direction; the faithful low-rank evidence is PC1 LOO stability plus
    # true deltas being more self-aligned than the shuffled control.
    low_rank = any(
        (v.get("pc1_loo_stability_mean_cosine") or 0) > 0.5
        and (v.get("pairwise_cosine_mean_true") or -1) > (v.get("pairwise_cosine_mean_shuffled") or -1)
        for v in geometry["per_layer"].values()
    )
    aligned = any(v.get("per_pair_cosine_stable_positive") for v in alignment["per_layer"].values())

    donor_free_works = bool(winners) and (mean_beats_random or pc1_beats_random) and mean_beats_shuffled and mean_beats_negative
    if donor_free_works and survivors:
        verdict = f"Donor-free MLP readout direction works and survives a first-step check (winners: {[w['cell'] for w in winners[:4]]}). Next: low-harm generation-level eval design (still not accuracy-proven here)."
    elif donor_free_works:
        verdict = f"Donor-free MLP readout direction is selective and low-harm under the teacher-forced proxy (winners: {[w['cell'] for w in winners[:4]]}) but the first-step check is weak. Next: strengthen the direction / probe generation."
    elif mean_beats_random or pc1_beats_random:
        verdict = "MLP readout carries causal information and beats random, but not all specificity/harm gates pass; current direction extraction is insufficient for clean donor-free steering. Next: refine direction (per-layer, whitening) before any generation eval."
    else:
        verdict = "Donor-free directions do not beat controls; the 3C-1 signal is not captured by a simple linear readout direction. Next: reconsider extraction or pause steering."

    qa = [
        ("Is this only MLP readout direction analysis, not full 3C?", True),
        ("Did we reuse the 3C-1 MLP readout causal site?", True),
        ("Did we use the corrected answer-sequence proxy?", True),
        (f"correct/wrong pairs used = {len(state)}", True),
        (f"layers analysed = {args.layers}", True),
        ("Is the correct-wrong MLP delta low-rank / shared-direction?", low_rank),
        ("Is the mean/PC1 direction stable (LOO)?", any((v.get("pc1_loo_stability_mean_cosine") or 0) > 0.5 for v in geometry["per_layer"].values())),
        ("Is the delta aligned with gold-vs-wrong unembedding?", aligned),
        ("Does donor-free mean direction beat random?", mean_beats_random),
        ("Does donor-free PC1 direction beat random?", pc1_beats_random),
        ("Does gold-vs-wrong unembedding direction beat random?", gold_beats_random),
        ("Does mean direction beat shuffled and negative controls?", bool(mean_beats_shuffled and mean_beats_negative)),
        ("Is there a low-harm alpha regime (<=0.2) for a donor-free direction?", bool(winners)),
        ("Does the signal survive the first-step generation-style check?", bool(survivors)),
        ("Is this still a teacher-forced / first-step proxy only?", True),
        ("Is 2000 rerun allowed?", False),
        ("Is hallucination reduction / accuracy improvement proven?", False),
    ]
    lines = [
        "# Sprint 3C-2 MLP Readout Direction Analysis Review Gate", "",
        "Verdict:", "- ready_for_2000_rerun: false", "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false", "- answer_accuracy_improvement_proven: false", "",
        "Scope:", f"- pairs: {len(state)}; layers: {args.layers}; alphas: {args.alphas}; directions: {args.directions}",
        "- target: MLP output at final-answer readout position; metric: corrected answer-sequence clean_direction (+ first-step proxy)", "",
        "Geometry:", f"- per_layer: {json.dumps(geometry['per_layer'], ensure_ascii=False)}", "",
        "Unembedding alignment (eval-only):", f"- per_layer: {json.dumps(alignment['per_layer'], ensure_ascii=False)}", "",
        "Donor-free control comparisons (paired, corrected clean_direction):",
    ]
    for name, res in cmp.items():
        lines.append(f"- {name}: mean={res.get('mean')} ci95=[{res.get('ci95_low')},{res.get('ci95_high')}] stable_positive={res.get('stable_positive')}")
    lines += ["", f"Low-harm donor-free winners: {json.dumps(winners[:8], ensure_ascii=False)}",
              f"First-step survivors (donor-free): {survivors}", "", "Required review questions:"]
    for q, a in qa:
        lines.append(f"- {q} {str(bool(a)).lower()}")
    lines += ["", "Interpretation:", f"- {verdict}",
              "- Teacher-forced (+ first-step) single-forward proxy only; no autoregressive generation, accuracy, or hallucination evaluation.",
              "- gold-unembedding and correct-donor deltas use gold answers for eval-only analysis, not as deployable methods."]
    return "\n".join(lines) + "\n"


def build_config(args, num_pairs, num_state):
    return {"backend": mrd.BACKEND, "model_path": args.model_path, "num_input_pairs": num_pairs, "num_usable_pairs": num_state,
            "layers": list(args.layers), "alphas": list(args.alphas), "directions": list(args.directions),
            "donor_free_directions": sorted(DONOR_FREE), "direction_construction": "leave-one-out (mean/pc1/shuffled built excluding the eval pair)",
            "nudge": "mlp_output[readout] += alpha * ||mlp_output[readout]|| * unit(direction)",
            "training": False, "lora": False, "full_sprint_3c": False,
            "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True,
            "hallucination_reduction_proven": False, "answer_accuracy_improvement_proven": False}


def write_incomplete(out_dir, reason):
    for name in ["mlp_direction_geometry_report.json", "mlp_unembedding_alignment_report.json", "donor_free_nudge_report.json",
                 "generation_survival_report.json", "harm_control_report.json"]:
        write_json({"backend": mrd.BACKEND, "incomplete_evidence_reason": reason, "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3C": True}, out_dir / name)
    for name in ["mlp_readout_delta_manifest.jsonl", "donor_free_direction_manifest.jsonl", "donor_free_nudge_forward_manifest.jsonl", "success_case_report.jsonl", "failure_case_report.jsonl"]:
        write_jsonl([], out_dir / name)
    (out_dir / "review_gate_mlp_readout_direction_analysis.md").write_text(
        "# Sprint 3C-2 Review Gate\n\nIncomplete evidence: " + reason + "\n\n- ready_for_2000_rerun: false\n- do_not_enter_full_sprint_3C: true\n", encoding="utf-8")


def write_preflight(args, out_dir):
    progress = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    top = "\n\n".join(progress.split("\n\n", 2)[:2])
    manifest = PROJECT_ROOT / "docs/progress/sprint_3_artifact_manifest.md"
    manifest_text = manifest.read_text(encoding="utf-8", errors="replace") if manifest.exists() else ""
    c1_dir = Path(args.input_dir)
    tracked = _git_tracked("src/recover_attention/module_causal_tracing.py")
    lines = [
        "# Sprint 3C-2 Preflight (MLP Readout Direction Analysis)", "",
        f"- progress_top_not_3A0: {str('Current Status Update: Sprint 3A-0' not in top).lower()}",
        f"- 3C1_case_a_recorded: {str('Case A' in manifest_text).lower()}",
        f"- 3C1_low_harm_cells_recorded: {str('mlp_output|L24' in manifest_text or 'mlp_output' in manifest_text).lower()}",
        f"- 3C1_output_dir_exists: {str(c1_dir.exists()).lower()}",
        f"- 3C_series_tracked_committed (module_causal_tracing.py tracked): {str(tracked).lower()}",
        f"- overwrites_prior_output_dir: {str(c1_dir.resolve() == out_dir.resolve()).lower()}",
        "- non_2000_non_full_3c_non_training: true", "",
        "Boundary: reuse 3C pairs, no re-sampling, no training, no deployable steering claim, teacher-forced (+first-step) proxy.",
        "- ready_for_2000_rerun=false; do_not_enter_full_sprint_3C=true; hallucination_reduction_proven=false; answer_accuracy_improvement_proven=false.", "",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def _git_tracked(path: str) -> bool:
    import subprocess
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", path], cwd=PROJECT_ROOT, capture_output=True, check=False)
        return r.returncode == 0
    except Exception:
        return False


def _rate(rows, pred):
    return float(sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0


if __name__ == "__main__":
    main()
