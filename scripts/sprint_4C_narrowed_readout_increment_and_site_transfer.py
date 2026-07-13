"""Sprint 4C: F1/F4 finite-label readout bake-off and diagnostic site transfer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import sprint_4B_3_full_f5_baseline_and_site_transfer as b3  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402
from recover_attention import mlp_readout_attribution as mra  # noqa: E402
from recover_attention import readout_increment as ri  # noqa: E402
from recover_attention.data_io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_text  # noqa: E402

LABELS = ["A", "B", "C", "D"]
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f5-input-dir", default="outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer")
    parser.add_argument("--processed-path", default="data/processed/cyber/cybermetric.jsonl")
    parser.add_argument("--model-path")
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 24])
    parser.add_argument("--mine-temperature", type=float, default=1.1)
    parser.add_argument("--mine-samples-per-question", type=int, default=12)
    parser.add_argument("--site-check-layers", nargs="+", type=int, default=[16, 20, 24])
    parser.add_argument("--site-check-alphas", nargs="+", type=float, default=[0.25, 1.0])
    parser.add_argument("--site-check-max-pairs", type=int, default=34)
    parser.add_argument("--cv-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=44031)
    return parser.parse_args()


def resolve_model_path(cli: str | None) -> tuple[Path | None, str]:
    for value, source in ((cli, "cli"), (os.environ.get("RECOVER_ATTENTION_MODEL_PATH"), "environment"), ("D:/models/Qwen2.5-7B-Instruct", "fallback")):
        if value and Path(value).exists():
            return Path(value), source
    return None, "unavailable"


def option_ids_for_trace(tokenizer: Any, trace: dict[str, Any], position: int) -> tuple[str, dict[str, int]] | None:
    encoded = tokenizer(trace["prompt"] + trace["completion"], add_special_tokens=False)["input_ids"]
    actual = encoded[position + 1] if position + 1 < len(encoded) else None
    for form, ids in (("space", dlp.option_token_ids(tokenizer, LABELS)), ("bare", dlp.bare_option_token_ids(tokenizer, LABELS))):
        if actual == ids[trace["parsed_label"]]:
            return form, ids
    return None


def capture_readout(context: dict[str, Any], trace: dict[str, Any], layers: list[int]) -> dict[str, Any] | None:
    """Capture direct MLP projections and exact VJP projections at the actual label readout."""
    import torch

    if trace.get("parse_failure") or trace.get("degenerate") or not trace.get("parsed_label"):
        return None
    tokenizer, model = context["tokenizer"], context["model"]
    try:
        position = dlp.locate_label_readout_position(tokenizer, trace["prompt"], trace["completion"], trace["parsed_label"])
    except ValueError:
        return None
    form_and_ids = option_ids_for_trace(tokenizer, trace, position)
    if form_and_ids is None:
        return None
    token_form, ids = form_and_ids
    full_ids = tokenizer(trace["prompt"] + trace["completion"], add_special_tokens=False)["input_ids"]
    prefix_ids = full_ids[: position + 1]
    device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    decoder = model.model.layers
    captured: dict[int, Any] = {}
    handles = []

    def hook_for(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor.retain_grad()
            captured[layer] = tensor
        return hook

    for layer in layers:
        handles.append(decoder[int(layer)].mlp.register_forward_hook(hook_for(int(layer))))
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, use_cache=False)
        final_logits = output.logits[0, -1]
        unembedding = model.get_output_embeddings().weight
        direct: dict[int, list[float]] = {}
        exact: dict[int, list[float]] = {}
        for layer in layers:
            activation = captured.get(int(layer))
            if activation is None:
                return None
            vector = activation[0, -1]
            direct[int(layer)] = [float(torch.dot(vector.float(), unembedding[int(ids[label])].float()).detach().cpu()) for label in LABELS]
            exact[int(layer)] = [float(ri.exact_vjp_position_projection(final_logits[int(ids[label])], activation, position=-1).detach().cpu()) for label in LABELS]
        final = [float(final_logits[int(ids[label])].detach().cpu()) for label in LABELS]
        full_probs = torch.softmax(final_logits.float(), dim=-1)
        full_entropy = float(-(full_probs * torch.log(torch.clamp(full_probs, min=1e-12))).sum().detach().cpu())
    finally:
        for handle in handles:
            handle.remove()
    f5 = ri.margin_entropy(final)
    return {
        "readout_position": int(position), "readout_token_form": token_form,
        "readout_to_prompt_end_distance": int(len(tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]) - 1 - position),
        "f5_label_margin": f5["margin"], "f5_label_entropy": f5["entropy"], "f5_full_entropy": full_entropy,
        **ri.cross_layer_features(direct[20], direct[24], final, prefix="f1"),
        **ri.cross_layer_features(exact[20], exact[24], final, prefix="f4"),
    }


def mine_pairs(context: dict[str, Any], records: dict[str, dict[str, Any]], f5_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = read_jsonl(f5_dir / "correct_wrong_pair_manifest.jsonl")
    candidates = []
    for path in (f5_dir / "high_risk_case_report.jsonl", f5_dir / "low_risk_wrong_case_report.jsonl"):
        candidates.extend(read_jsonl(path))
    ids = list(dict.fromkeys(row["example_id"] for row in candidates))[:60]
    traces = [trace for pair in original for trace in (pair["correct_trace"], pair["wrong_trace"])]
    for index, example_id in enumerate(ids):
        record = records[example_id]
        prompt = b3.build_prompt(context, record, "chat")
        for sample_idx in range(args.mine_samples_per_question):
            completion, count = b3.generate_completion(context, prompt, max_new_tokens=256, do_sample=True, temperature=args.mine_temperature, top_p=0.95, seed=args.seed + index * 100 + sample_idx, valid_labels=record["candidate_labels"])
            traces.append(b3.build_trace_row(record, "chat", prompt, "sample", sample_idx + 100, completion, count, argparse.Namespace(max_new_tokens=256, temperature=args.mine_temperature, top_p=0.95)))
        if (index + 1) % 10 == 0:
            print(f"[4C] pair mining sampled {index + 1}/{len(ids)} candidates")
    pairs = b3.build_pairs(traces)
    return pairs, {"candidate_questions": len(ids), "additional_samples": len(ids) * args.mine_samples_per_question, "num_pairs": len(pairs), "insufficient_pairs": len(pairs) < 20}


def evaluate(rows: list[dict[str, Any]], seeds: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    families = {
        "A_F5_alone": ["f5_label_margin", "f5_label_entropy", "f5_full_entropy"],
        "B_F1_alone": [key for key in rows[0] if key.startswith("f1_")],
        "B_F4_alone": [key for key in rows[0] if key.startswith("f4_")],
    }
    families.update({"C_F5_F1": families["A_F5_alone"] + families["B_F1_alone"], "C_F5_F4": families["A_F5_alone"] + families["B_F4_alone"], "C_F5_F1_F4": families["A_F5_alone"] + families["B_F1_alone"] + families["B_F4_alone"]})
    report = ri.evaluate_grouped_cv(rows, families=families, seeds=seeds)
    baseline = report["A_F5_alone"]["mean_oof_auroc"]
    base_predictions = {item["row_index"]: item["risk_score"] for item in report["A_F5_alone"]["seed_runs"][0].get("oof_predictions", [])}
    for name, value in report.items():
        value["increment_over_f5"] = None if baseline is None or value["mean_oof_auroc"] is None else value["mean_oof_auroc"] - baseline
        candidate_predictions = {item["row_index"]: item["risk_score"] for item in value["seed_runs"][0].get("oof_predictions", [])}
        paired = [{"group_id": rows[index]["group_id"], "wrong_label": rows[index]["wrong_label"], "base": base_predictions[index], "candidate": candidate_predictions[index]} for index in base_predictions.keys() & candidate_predictions.keys()]
        value["increment_ci95"] = ri.percentile_ci(ri.grouped_bootstrap_increment(paired, baseline_key="base", candidate_key="candidate", seed=34000)) if paired else [None, None]
        value["increment_ci95_lower_gt_zero"] = bool(value["increment_ci95"][0] is not None and value["increment_ci95"][0] > 0)
    combined = [report[name] for name in ("C_F5_F1", "C_F5_F4", "C_F5_F1_F4")]
    detector_beats = any(item["increment_ci95_lower_gt_zero"] for item in combined)
    f1, f4 = report["C_F5_F1"], report["C_F5_F4"]
    arbitration = "F4_increment_only_readout_method_matters" if f4["increment_ci95_lower_gt_zero"] and not f1["increment_ci95_lower_gt_zero"] else "F1_at_least_F4_or_both_no_increment"
    return report, {"detector_beats_f5": detector_beats, "j_lens_arbitration": arbitration, "baseline_cv_auroc": baseline}


def calibration_report(result: dict[str, Any]) -> dict[str, Any]:
    predictions = result["seed_runs"][0].get("oof_predictions", [])
    scores = [float(item["risk_score"]) for item in predictions]
    labels = [int(item["wrong_label"]) for item in predictions]
    ordered = sorted(zip(scores, labels))
    curve = []
    for coverage in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        count = max(1, round(len(ordered) * coverage))
        accepted = ordered[:count]
        curve.append({"coverage": coverage, "accepted_examples": count, "selective_wrong_rate": float(np.mean([label for _score, label in accepted]))})
    return {"best_feature_names": result["feature_names"], "best_cv_auroc": result["mean_oof_auroc"], "calibration": mra.calibration_buckets(scores, labels), "risk_coverage_curve": curve, "note": "OOF grouped-CV logistic predictions only; no intervention is performed."}


def main() -> None:
    args = parse_args(); out_dir = Path(args.output_dir); f5_dir = Path(args.f5_input_dir)
    ensure_dir(out_dir)
    required = [f5_dir / name for name in ("trace_sampling_manifest.jsonl", "f5_baseline_report.json", "correct_wrong_pair_manifest.jsonl")]
    if any(not path.exists() for path in required):
        raise SystemExit(f"missing required 4B-3 input(s): {[str(path) for path in required if not path.exists()]}")
    model_path, source = resolve_model_path(args.model_path)
    if model_path is None:
        raise SystemExit("model path unavailable: --model-path > RECOVER_ATTENTION_MODEL_PATH > fallback")
    f5_report = read_json(f5_dir / "f5_baseline_report.json")
    print(f"[4C] input={f5_dir} F5={f5_report['kill_bar_single_forward']['auroc']:.4f} model={model_path} source={source}")
    records = {row["example_id"]: row for row in read_jsonl(args.processed_path)}
    context = abs_mod.load_local_steering_backend(model_path=model_path)
    pairs, mining = mine_pairs(context, records, f5_dir, args)
    write_json(mining, out_dir / "pair_mining_report.json"); write_jsonl(pairs, out_dir / "correct_wrong_pair_manifest.jsonl")
    if len(pairs) >= 20:
        site, fidelity = b3.run_stage_e(context, pairs[:args.site_check_max_pairs], records, args)
        site.update({"status": "evaluated", "gsm8k_3c1_reference": b3.GSM8K_3C1_REFERENCE})
        write_json(site, out_dir / "site_transfer_check_report.json"); write_json(fidelity, out_dir / "module_patch_fidelity_report.json")
    else:
        site = {"status": "skipped", "skipped_reason": "insufficient_pairs", "num_pairs": len(pairs), "transfer_conclusion": "not_evaluated_gate_skipped"}
        write_json(site, out_dir / "site_transfer_check_report.json")
    traces = [row for row in read_jsonl(f5_dir / "trace_sampling_manifest.jsonl") if row["sample_type"] == "greedy"]
    rows, counts = [], Counter()
    for index, trace in enumerate(traces):
        features = capture_readout(context, trace, args.layers)
        if features is None:
            counts["unavailable"] += 1; continue
        dlp.assert_no_gold_label_leakage(features)
        rows.append({"example_id": trace["example_id"], "group_id": trace["group_id"], "wrong_label": int(trace["is_correct"] is False), **features})
        counts[features["readout_token_form"]] += 1
        if (index + 1) % 20 == 0: print(f"[4C] captured {index + 1}/{len(traces)}")
    write_jsonl(rows, out_dir / "readout_feature_manifest.jsonl")
    report, decision = evaluate(rows, args.cv_seeds)
    report.update({"f5_single_forward_reference": f5_report["kill_bar_single_forward"], "num_feature_rows": len(rows), "token_form_counts": dict(counts), **decision, "probe_type": "linear_logistic_detection_only"})
    write_json(report, out_dir / "readout_increment_report.json")
    best = max((item for item in report.values() if isinstance(item, dict) and "mean_oof_auroc" in item), key=lambda item: item["mean_oof_auroc"] or -1)
    calibration = calibration_report(best)
    write_json(calibration, out_dir / "calibration_report.json")
    for name in ("high_risk_case_report.jsonl", "low_risk_wrong_case_report.jsonl"):
        write_jsonl(read_jsonl(f5_dir / name), out_dir / name)
    review = f"# Sprint 4C Review Gate\n\n- F5 raw reference AUROC: {f5_report['kill_bar_single_forward']['auroc']:.4f}\n- F5 grouped-CV logistic AUROC: {decision['baseline_cv_auroc']}\n- F1-alone / F4-alone AUROC: {report['B_F1_alone']['mean_oof_auroc']} / {report['B_F4_alone']['mean_oof_auroc']}\n- F5+F1 increment CI: {report['C_F5_F1']['increment_ci95']}\n- F5+F4 increment CI: {report['C_F5_F4']['increment_ci95']}\n- F5+F1+F4 increment CI: {report['C_F5_F1_F4']['increment_ci95']}\n- pairs after mining: {len(pairs)}\n- site transfer: {site['status']}\n- detector_beats_f5: {decision['detector_beats_f5']}\n- J-lens arbitration: {decision['j_lens_arbitration']}\n- probe is linear logistic detection only: yes\n- gold label used as inference feature: no\n- hallucination reduction / accuracy improvement claimed: no\n"
    write_text(review, out_dir / "review_gate_readout_increment_and_site_transfer.md")
    print(f"[4C] complete rows={len(rows)} pairs={len(pairs)} out={out_dir}")


if __name__ == "__main__":
    main()
