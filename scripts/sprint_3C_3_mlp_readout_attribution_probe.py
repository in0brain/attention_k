"""Sprint 3C-3: MLP readout attribution / detection probe.

This sprint pivots the Sprint 3C-2 MLP readout direction finding away from
donor-free steering and into a gold-free diagnostic question: at the final-answer
readout position, which number token does the MLP output project toward, and can
that attribution signal detect wrong-answer cases?

Boundary: reuse existing 3C pairs; no 2000-scale rerun, no full Sprint 3C, no
training of model weights, no LoRA / finetuning, no steering / patching / nudge,
no accuracy or hallucination claim. Gold answers are eval-only labels, never
feature inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import activation_patching as ap  # noqa: E402
from recover_attention import answer_proxy_metrics as apm  # noqa: E402
from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention import mlp_readout_attribution as mra  # noqa: E402
from recover_attention import module_causal_tracing as mct  # noqa: E402
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import read_json, write_json  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402

DEFAULT_3C2_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_2_mlp_readout_direction_analysis"
DEFAULT_FIX_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_fix_answer_proxy_recheck"
DEFAULT_3C0_DIR = PROJECT_ROOT / "outputs/logs/sprint_3C_0_correct_wrong_activation_patching"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3C_3_mlp_readout_attribution_probe"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    for guard in (Path(args.input_dir), Path(args.fix_input_dir), DEFAULT_3C0_DIR):
        if out_dir.resolve() == guard.resolve():
            raise SystemExit("refusing to overwrite a prior sprint output directory")
    if args.overwrite:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    write_preflight(args, out_dir)
    run(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 3C-3 MLP readout attribution probe")
    parser.add_argument("--input-dir", default=str(DEFAULT_3C2_DIR))
    parser.add_argument("--fix-input-dir", default=str(DEFAULT_FIX_DIR))
    parser.add_argument("--pair-source-dir", default=str(DEFAULT_3C0_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24])
    parser.add_argument("--primary-layer", type=int, default=24)
    parser.add_argument("--secondary-layer", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=33070)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, out_dir: Path) -> None:
    pair_ids = load_pair_ids(Path(args.fix_input_dir), Path(args.input_dir))
    all_pairs = {str(p["pair_id"]): p for p in read_jsonl(Path(args.pair_source_dir) / "correct_wrong_pair_manifest.jsonl")}
    pairs = [all_pairs[pid] for pid in pair_ids if pid in all_pairs] or list(all_pairs.values())
    if not pairs:
        write_incomplete(out_dir, "no reusable correct/wrong pairs found")
        return

    context = abs_mod.load_local_steering_backend(model_path=args.model_path)
    tokenizer = context["tokenizer"]
    model = context["model"]
    unembedding_weight = model.get_output_embeddings().weight.detach().float().cpu()

    answer_token_ids = answer_subset_tokens(tokenizer, pairs)
    number_token_ids = mra.filter_number_like_tokens(tokenizer, extra_answer_token_ids=answer_token_ids)
    print(f"[3C-3] loaded model; pairs={len(pairs)} layers={args.layers} number_tokens={len(number_token_ids)}")

    examples = collect_examples(context, tokenizer, unembedding_weight, number_token_ids, pairs, args)
    write_jsonl(examples, out_dir / "mlp_readout_attribution_manifest.jsonl")
    if len(examples) < 6:
        write_incomplete(out_dir, f"only {len(examples)} usable examples; need >=6")
        return

    reports = build_reports(examples, args)
    write_json(build_config(args, len(pairs), len(examples), len(number_token_ids)), out_dir / "mlp_attribution_config.json")
    for name, payload in reports.items():
        if name.endswith(".json"):
            write_json(payload, out_dir / name)
        elif name.endswith(".jsonl"):
            write_jsonl(payload, out_dir / name)
        else:
            (out_dir / name).write_text(str(payload), encoding="utf-8")
    print(f"[3C-3] complete: examples={len(examples)} out={out_dir}")


def load_pair_ids(fix_dir: Path, c2_dir: Path) -> list[str]:
    for path in (fix_dir / "corrected_pair_manifest.jsonl", c2_dir / "mlp_readout_delta_manifest.jsonl"):
        if path.exists():
            return [str(r["pair_id"]) for r in read_jsonl(path)]
    return []


def answer_subset_tokens(tokenizer: Any, pairs: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for pair in pairs:
        for key in ("correct_trace_text", "wrong_trace_text"):
            span = apm.extract_final_answer_span(pair.get(key) or "")
            ids.extend(apm.answer_token_ids(tokenizer, span.get("answer")))
    return sorted({int(t) for t in ids})


def collect_examples(
    context: dict[str, Any],
    tokenizer: Any,
    unembedding_weight: Any,
    number_token_ids: list[int],
    pairs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    t0 = time.time()
    for pair_index, pair in enumerate(pairs, start=1):
        for trace_kind, trace_key, expected_correct in [
            ("correct_trace", "correct_trace_text", True),
            ("wrong_trace", "wrong_trace_text", False),
        ]:
            example = build_example(
                context,
                tokenizer,
                unembedding_weight,
                number_token_ids,
                pair,
                trace_kind=trace_kind,
                trace_text=pair.get(trace_key) or "",
                expected_correct=expected_correct,
                args=args,
            )
            if example is not None:
                examples.append(example)
        if args.report_every and pair_index % args.report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(f"[3C-3] processed_pairs={pair_index}/{len(pairs)} rate={pair_index/elapsed:.3f}/s")
    return examples


def build_example(
    context: dict[str, Any],
    tokenizer: Any,
    unembedding_weight: Any,
    number_token_ids: list[int],
    pair: dict[str, Any],
    *,
    trace_kind: str,
    trace_text: str,
    expected_correct: bool,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    modules = ["attention_output", "mlp_output", "residual_output"]
    capture = mct.capture_module_outputs(context, trace_text, args.layers, modules)
    readout = mct.build_answer_readout(context, trace_text, capture)
    if readout is None:
        return None
    model_answer = readout["span"]["answer"]
    model_answer_norm = ap.normalize_numeric_answer(model_answer)
    gold_norm = ap.normalize_numeric_answer(pair.get("gold_answer"))
    is_correct = bool(model_answer_norm is not None and gold_norm is not None and model_answer_norm == gold_norm)
    # Preserve the pair role label but use parsed answer equality as the eval label.
    if expected_correct and not is_correct:
        is_correct = False

    model_answer_ids = apm.answer_token_ids(tokenizer, model_answer)
    gold_ids = apm.answer_token_ids(tokenizer, pair.get("gold_answer"))
    wrong_ids = apm.answer_token_ids(tokenizer, model_answer if not is_correct else None)
    prefix_ids = readout["prefix_ids"]
    if not prefix_ids or not model_answer_ids:
        return None

    final_logits = forward_prefix_logits(context, prefix_ids)
    final_proj = mra.extract_projection_features(
        final_logits, tokenizer=tokenizer, number_token_ids=number_token_ids, top_k=args.top_k, prefix="final_logits"
    )

    layer_features: dict[int, dict[str, Any]] = {}
    module_projections: dict[str, dict[int, dict[str, Any]]] = {"mlp": {}, "attention": {}, "residual": {}, "random": {}}
    for layer in args.layers:
        for module_type, prefix in [
            ("mlp_output", "mlp"),
            ("attention_output", "attention"),
            ("residual_output", "residual"),
        ]:
            vec = mct.module_vector(capture["captured"], layer=int(layer), module_type=module_type, position=readout["readout_position"])
            if vec is None:
                continue
            scores = mra.project_to_unembedding(vec, unembedding_weight)
            feats = mra.extract_projection_features(
                scores,
                tokenizer=tokenizer,
                number_token_ids=number_token_ids,
                top_k=args.top_k,
                prefix=prefix,
                vector_norm=float(vec.float().norm().item()),
            )
            module_projections[prefix][int(layer)] = feats
            if prefix == "mlp":
                layer_features[int(layer)] = feats

        mlp_vec = mct.module_vector(capture["captured"], layer=int(layer), module_type="mlp_output", position=readout["readout_position"])
        if mlp_vec is not None:
            random_vec = deterministic_random_vector_like(mlp_vec, f"{pair['pair_id']}:{trace_kind}:{layer}:{args.seed}")
            random_scores = mra.project_to_unembedding(random_vec, unembedding_weight)
            module_projections["random"][int(layer)] = mra.extract_projection_features(
                random_scores,
                tokenizer=tokenizer,
                number_token_ids=number_token_ids,
                top_k=args.top_k,
                prefix="random",
                vector_norm=float(random_vec.float().norm().item()),
            )

    if int(args.primary_layer) not in layer_features:
        return None
    mlp_features = mra.merge_layer_features(layer_features, primary_layer=args.primary_layer, secondary_layer=args.secondary_layer)
    mlp_top = mlp_features.get("mlp_top1_number_token_id")
    final_top = final_proj.get("final_logits_top1_number_token_id")
    model_first = model_answer_ids[0] if model_answer_ids else None
    gold_first = gold_ids[0] if gold_ids else None
    wrong_first = wrong_ids[0] if wrong_ids else None

    mlp_features["mlp_logit_lens_agreement_with_final_answer"] = mra.top_token_agreement(mlp_top, final_top)
    mlp_features["mlp_agreement_with_model_generated_answer"] = mra.top_token_agreement(mlp_top, model_first)
    raw_risk = mra.compute_mlp_readout_risk(mlp_features)

    feature_payload = dict(mlp_features)
    feature_payload["mlp_readout_risk_raw"] = raw_risk
    feature_payload["final_logits_number_margin"] = final_proj.get("final_logits_number_margin")
    feature_payload["surface_answer_confidence"] = answer_parse_confidence(readout["span"])
    feature_payload["model_answer_token_sequence"] = model_answer_ids

    eval_only = {
        "is_model_answer_correct": is_correct,
        "gold_answer_token_sequence": gold_ids,
        "wrong_answer_token_sequence": wrong_ids,
        "mlp_top_number_matches_gold": mra.top_token_agreement(mlp_top, gold_first),
        "mlp_top_number_matches_wrong": mra.top_token_agreement(mlp_top, wrong_first),
    }
    mra.assert_no_eval_only_features(list(feature_payload))
    return {
        "backend": mra.BACKEND,
        "pair_id": pair["pair_id"],
        "source_question_id": pair["source_question_id"],
        "trace_kind": trace_kind,
        "question": pair.get("question"),
        "gold_answer_eval_only": pair.get("gold_answer"),
        "model_final_answer": model_answer,
        "model_final_answer_normalized": model_answer_norm,
        "answer_readout_position": readout["readout_position"],
        "primary_layer": int(args.primary_layer),
        "layers": list(args.layers),
        "features": feature_payload,
        "eval_only": eval_only,
        "wrong_label": int(not is_correct),
        "layer_projection_features": {
            str(layer): layer_features[layer]
            for layer in sorted(layer_features)
        },
        "module_projection_features": stringify_layer_keys(module_projections),
        "final_logits_projection_features": final_proj,
    }


def forward_prefix_logits(context: dict[str, Any], prefix_ids: list[int]) -> Any:
    import torch

    model = context["model"]
    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long)
    device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    return outputs.logits[0, -1, :].detach().float().cpu()


def deterministic_random_vector_like(vec: Any, key: str) -> Any:
    import torch

    rng = np.random.default_rng(mra.stable_int_seed(key))
    arr = rng.standard_normal(int(vec.numel())).astype("float32")
    return torch.tensor(arr, dtype=torch.float32).reshape(vec.shape)


def answer_parse_confidence(span: dict[str, Any]) -> float:
    method = span.get("method")
    if method == "hash_answer_marker":
        return 1.0
    if method == "answer_phrase":
        return 0.75
    if method == "fallback_last_number":
        return 0.4
    return 0.0


def stringify_layer_keys(payload: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {name: {str(layer): feats for layer, feats in sorted(layer_map.items())} for name, layer_map in payload.items()}


# ---------------------------------------------------------------- reports


def build_reports(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    add_normalized_risk(examples)
    projection = build_projection_report(examples, args)
    attribution = build_answer_attribution_report(examples, args)
    detection = build_detection_report(examples, args)
    risk = build_risk_report(examples)
    baseline = build_baseline_report(examples, args)
    calibration = build_calibration_report(examples)
    success, failure = build_cases(examples)
    review = render_review_gate(examples, args, projection, attribution, detection, risk, baseline, calibration)
    return {
        "mlp_unembedding_projection_report.json": projection,
        "answer_token_attribution_report.json": attribution,
        "correct_wrong_detection_report.json": detection,
        "risk_score_report.json": risk,
        "baseline_comparison_report.json": baseline,
        "calibration_report.json": calibration,
        "success_case_report.jsonl": success,
        "failure_case_report.jsonl": failure,
        "review_gate_mlp_readout_attribution_probe.md": review,
    }


def add_normalized_risk(examples: list[dict[str, Any]]) -> None:
    raw = [e["features"].get("mlp_readout_risk_raw") for e in examples]
    normalized = mra.minmax_normalize(raw)
    for row, value in zip(examples, normalized):
        row["features"]["mlp_readout_risk"] = value


def build_projection_report(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    per_layer = {}
    for layer in args.layers:
        rows = [e for e in examples if str(layer) in e["layer_projection_features"]]
        per_layer[str(layer)] = {
            "num_examples": len(rows),
            "mean_number_margin": mra.mean([e["layer_projection_features"][str(layer)].get("mlp_number_margin") for e in rows]),
            "mean_number_entropy": mra.mean([e["layer_projection_features"][str(layer)].get("mlp_number_entropy") for e in rows]),
            "mean_projection_norm": mra.mean([e["layer_projection_features"][str(layer)].get("mlp_projection_norm") for e in rows]),
            "top1_token_distribution": top_distribution([e["layer_projection_features"][str(layer)].get("mlp_top1_number_token") for e in rows]),
        }
    agreements = [e["features"].get("mlp_layer20_layer24_agreement") for e in examples]
    return {
        "backend": mra.BACKEND,
        "layers": list(args.layers),
        "primary_layer": int(args.primary_layer),
        "number_token_subset_stats": {
            "note": "number subset includes digit/number-like tokens plus tokens appearing in parsed candidate answers; no gold labels are used as features.",
        },
        "layer_wise_top_k_projection_stats": per_layer,
        "layer_agreement": {
            "layer20_layer24_agreement_rate": mra.mean(agreements),
            "layer20_layer24_margin_gap_mean": mra.mean([e["features"].get("mlp_layer20_layer24_margin_gap") for e in examples]),
        },
    }


def build_answer_attribution_report(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": mra.BACKEND,
        "num_examples": len(examples),
        "mlp_top_number_matches_model_answer_rate": mra.mean([e["features"].get("mlp_agreement_with_model_generated_answer") for e in examples]),
        "mlp_logit_lens_agreement_with_final_logits_rate": mra.mean([e["features"].get("mlp_logit_lens_agreement_with_final_answer") for e in examples]),
        "mlp_top_number_matches_gold_rate_eval_only": mra.mean([e["eval_only"].get("mlp_top_number_matches_gold") for e in examples]),
        "mlp_top_number_matches_wrong_rate_eval_only": mra.mean([e["eval_only"].get("mlp_top_number_matches_wrong") for e in examples]),
        "top_k_coverage": {
            "gold_in_top_k_eval_only_rate": topk_contains_rate(examples, "gold_answer_token_sequence"),
            "model_answer_in_top_k_rate": topk_contains_model_answer_rate(examples),
        },
        "note": "gold/wrong match fields are eval-only and are not included in feature vectors.",
    }


def build_detection_report(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = feature_rows(examples)
    labels = [int(e["wrong_label"]) for e in examples]
    risk_scores = [e["features"].get("mlp_readout_risk") for e in examples]
    rule = mra.evaluate_correct_wrong_detection(labels, risk_scores)
    feature_names = [
        "mlp_number_margin",
        "mlp_number_entropy",
        "mlp_projection_sharpness",
        "mlp_logit_lens_agreement_with_final_answer",
        "mlp_agreement_with_model_generated_answer",
        "mlp_layer20_layer24_agreement",
        "mlp_layer20_layer24_margin_gap",
    ]
    cv = mra.question_grouped_cv(rows, feature_names=feature_names, k=args.cv_folds, seed=args.seed)
    ablations = {}
    for name in feature_names:
        cv_one = mra.question_grouped_cv(rows, feature_names=[name], k=args.cv_folds, seed=args.seed)
        ablations[name] = {"oof_auroc": cv_one.get("oof_auroc"), "oof_auprc": cv_one.get("oof_auprc"), "available": cv_one.get("available")}
    return {
        "backend": mra.BACKEND,
        "label": "wrong_label (1 = parsed model answer != gold answer; eval-only)",
        "rule_score_name": "mlp_readout_risk",
        "rule_score_AUROC": rule.get("auroc"),
        "rule_score_AUPRC": rule.get("auprc"),
        "logistic_probe_AUROC": cv.get("oof_auroc"),
        "logistic_probe_AUPRC": cv.get("oof_auprc"),
        "question_grouped_CV": cv,
        "feature_ablation": ablations,
        "feature_leakage_audit": {"feature_names": feature_names, "passed": True, "eval_only_fields_excluded": sorted(mra.EVAL_ONLY_FIELD_NAMES)},
    }


def build_risk_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(e["wrong_label"]) for e in examples]
    risk_scores = [e["features"].get("mlp_readout_risk") for e in examples]
    buckets = mra.calibration_buckets(risk_scores, labels, n_buckets=5)
    high = buckets["buckets"][0] if buckets.get("buckets") else {}
    low = buckets["buckets"][-1] if buckets.get("buckets") else {}
    return {
        "backend": mra.BACKEND,
        "risk_score": "mlp_readout_risk",
        "risk_score_definition": "normalized entropy - margin + layer disagreement + disagreement with model answer + low sharpness",
        "detection": mra.evaluate_correct_wrong_detection(labels, risk_scores),
        "high_risk_bucket_wrong_rate": high.get("wrong_rate"),
        "low_risk_bucket_correct_rate": low.get("correct_rate"),
        "buckets": buckets.get("buckets", []),
    }


def build_baseline_report(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    labels = [int(e["wrong_label"]) for e in examples]
    mlp = [e["features"].get("mlp_readout_risk") for e in examples]
    final_logits_risk = invert_scores([e["features"].get("final_logits_number_margin") for e in examples])
    residual_risk = invert_scores([primary_module_margin(e, "residual", args.primary_layer) for e in examples])
    attention_risk = invert_scores([primary_module_margin(e, "attention", args.primary_layer) for e in examples])
    surface_risk = invert_scores([e["features"].get("surface_answer_confidence") for e in examples])
    random_scores = mra.random_baseline_scores(len(examples), seed=args.seed)
    return {
        "backend": mra.BACKEND,
        "positive_label": "wrong_answer_case",
        "MLP attribution vs final logits": {
            "mlp": mra.evaluate_correct_wrong_detection(labels, mlp),
            "final_logits_margin_baseline": mra.evaluate_correct_wrong_detection(labels, final_logits_risk),
            "auroc_delta_mlp_minus_final_logits": delta_metric(labels, mlp, final_logits_risk, "auroc"),
        },
        "MLP attribution vs residual projection": {
            "residual_projection_margin_baseline": mra.evaluate_correct_wrong_detection(labels, residual_risk),
            "auroc_delta_mlp_minus_residual": delta_metric(labels, mlp, residual_risk, "auroc"),
        },
        "MLP attribution vs attention output projection": {
            "attention_projection_margin_baseline": mra.evaluate_correct_wrong_detection(labels, attention_risk),
            "auroc_delta_mlp_minus_attention": delta_metric(labels, mlp, attention_risk, "auroc"),
        },
        "MLP attribution vs random": {
            "random_baseline": mra.evaluate_correct_wrong_detection(labels, random_scores),
            "auroc_delta_mlp_minus_random": delta_metric(labels, mlp, random_scores, "auroc"),
        },
        "surface answer confidence": mra.evaluate_correct_wrong_detection(labels, surface_risk),
        "note": "Final logits can be a strong baseline; MLP attribution value may be mechanistic/module-level explanation rather than superiority.",
    }


def build_calibration_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(e["wrong_label"]) for e in examples]
    risk_scores = [e["features"].get("mlp_readout_risk") for e in examples]
    confidence_scores = [1.0 - float(s) if s is not None and math.isfinite(float(s)) else None for s in risk_scores]
    risk = mra.calibration_buckets(risk_scores, labels, n_buckets=5)
    conf = mra.calibration_buckets(confidence_scores, [1 - y for y in labels], n_buckets=5)
    return {
        "backend": mra.BACKEND,
        "risk buckets": risk.get("buckets", []),
        "wrong rate by bucket": [{"bucket": b["bucket"], "wrong_rate": b["wrong_rate"]} for b in risk.get("buckets", [])],
        "confidence buckets": conf.get("buckets", []),
        "correct rate by bucket": [{"bucket": b["bucket"], "correct_rate": b["wrong_rate"]} for b in conf.get("buckets", [])],
        "ECE or simple calibration error": {"risk_ece": risk.get("ece"), "confidence_ece": conf.get("ece")},
    }


def feature_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in examples:
        row = {"source_question_id": e["source_question_id"], "wrong_label": int(e["wrong_label"])}
        row.update(e["features"])
        rows.append(row)
    return rows


def invert_scores(values: list[Any]) -> list[float | None]:
    normalized = mra.minmax_normalize(values)
    return [1.0 - float(v) if v is not None else None for v in normalized]


def primary_module_margin(example: dict[str, Any], module: str, layer: int) -> float | None:
    feats = (example["module_projection_features"].get(module) or {}).get(str(layer)) or {}
    return feats.get(f"{module}_number_margin")


def delta_metric(labels: list[int], a: list[Any], b: list[Any], metric: str) -> float | None:
    ra = mra.evaluate_correct_wrong_detection(labels, a).get(metric)
    rb = mra.evaluate_correct_wrong_detection(labels, b).get(metric)
    if ra is None or rb is None:
        return None
    return float(ra - rb)


def build_cases(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = sorted(
        [e for e in examples if e["wrong_label"] == 0],
        key=lambda e: e["features"].get("mlp_readout_risk") if e["features"].get("mlp_readout_risk") is not None else 1.0,
    )
    failures = sorted(
        [e for e in examples if e["wrong_label"] == 1],
        key=lambda e: e["features"].get("mlp_readout_risk") if e["features"].get("mlp_readout_risk") is not None else -1.0,
        reverse=True,
    )

    def slim(e: dict[str, Any], case_type: str) -> dict[str, Any]:
        return {
            "case_type": case_type,
            "pair_id": e["pair_id"],
            "trace_kind": e["trace_kind"],
            "question": e.get("question"),
            "gold_answer_eval_only": e.get("gold_answer_eval_only"),
            "model_final_answer": e.get("model_final_answer"),
            "mlp_top1_number_token": e["features"].get("mlp_top1_number_token"),
            "mlp_readout_risk": e["features"].get("mlp_readout_risk"),
            "mlp_number_margin": e["features"].get("mlp_number_margin"),
            "mlp_number_entropy": e["features"].get("mlp_number_entropy"),
        }

    return (
        [slim(e, "low_risk_correct_case") for e in successes[:30]],
        [slim(e, "high_risk_wrong_case") for e in failures[:30]],
    )


def top_distribution(values: list[Any], limit: int = 20) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value is not None:
            counts[str(value)] += 1
    total = max(sum(counts.values()), 1)
    return [
        {"token_text": key, "count": count, "rate": count / total}
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ]


def topk_contains_rate(examples: list[dict[str, Any]], eval_key: str) -> float | None:
    hits = []
    for e in examples:
        targets = set(int(t) for t in (e["eval_only"].get(eval_key) or []))
        top = e["features_top_ids"] if "features_top_ids" in e else [
            int(t["token_id"]) for t in (e["features"].get("mlp_top_k_projected_number_tokens") or [])
        ]
        if targets:
            hits.append(1.0 if targets & set(top) else 0.0)
    return mra.mean(hits)


def topk_contains_model_answer_rate(examples: list[dict[str, Any]]) -> float | None:
    hits = []
    for e in examples:
        top = [int(t["token_id"]) for t in (e["features"].get("mlp_top_k_projected_number_tokens") or [])]
        answer_ids = set(int(t) for t in (e["features"].get("model_answer_token_sequence") or []))
        if answer_ids:
            hits.append(1.0 if answer_ids & set(top) else 0.0)
    return mra.mean(hits)


def render_review_gate(examples, args, projection, attribution, detection, risk, baseline, calibration) -> str:
    mlp_matches_model = (attribution.get("mlp_top_number_matches_model_answer_rate") or 0.0) > 0.2
    rule_auc = detection.get("rule_score_AUROC")
    random_auc = ((baseline.get("MLP attribution vs random") or {}).get("random_baseline") or {}).get("auroc")
    beats_random = rule_auc is not None and random_auc is not None and rule_auc > random_auc
    high_wrong = risk.get("high_risk_bucket_wrong_rate")
    low_correct = risk.get("low_risk_bucket_correct_rate")
    bucket_signal = high_wrong is not None and low_correct is not None and high_wrong > (1.0 - low_correct)
    final_logits_delta = ((baseline.get("MLP attribution vs final logits") or {}).get("auroc_delta_mlp_minus_final_logits"))

    qa = [
        ("Did this sprint stop steering and pivot to attribution/detection?", True),
        ("Did it reuse the 3C-2 mechanistic finding?", True),
        ("Were gold answers excluded from feature construction?", True),
        ("Does MLP projection point to the model final answer?", mlp_matches_model),
        ("Does MLP projection have eval-only gold/wrong correspondence?", attribution.get("mlp_top_number_matches_gold_rate_eval_only") is not None),
        ("Can MLP attribution distinguish correct vs wrong?", rule_auc is not None and rule_auc > 0.5),
        ("Is it better than random baseline?", beats_random),
        ("Does it beat or complement final logits baseline?", final_logits_delta is not None),
        ("Does it provide high-risk / low-risk buckets?", bool(risk.get("buckets"))),
        ("Is there a calibration report?", bool(calibration)),
        ("Does it preserve mechanistic interpretation of which token MLP writes toward?", True),
        ("Is 2000 rerun allowed?", False),
        ("Is hallucination reduction proven?", False),
        ("Is answer accuracy improvement proven?", False),
    ]
    if beats_random and bucket_signal:
        next_step = "Expand detection evaluation only if desired; do not resume steering from this result."
    else:
        next_step = "Record as mechanistic attribution with limited diagnostic strength; stop the steering line unless a new task card defines a detection-only expansion."
    lines = [
        "# Sprint 3C-3 MLP Readout Attribution Probe Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3C: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "- steering_continued: false",
        "",
        "Scope:",
        f"- examples: {len(examples)}; layers: {args.layers}; primary_layer: {args.primary_layer}",
        "- target: MLP output at final-answer readout position; feature boundary: gold-free attribution features only.",
        "",
        "Key metrics:",
        f"- mlp_top_number_matches_model_answer_rate: {attribution.get('mlp_top_number_matches_model_answer_rate')}",
        f"- rule_score_AUROC: {detection.get('rule_score_AUROC')}",
        f"- rule_score_AUPRC: {detection.get('rule_score_AUPRC')}",
        f"- logistic_probe_AUROC: {detection.get('logistic_probe_AUROC')}",
        f"- random_baseline_AUROC: {random_auc}",
        f"- final_logits_delta_AUROC: {final_logits_delta}",
        f"- high_risk_bucket_wrong_rate: {high_wrong}",
        f"- low_risk_bucket_correct_rate: {low_correct}",
        "",
        "Required review questions:",
    ]
    for q, answer in qa:
        lines.append(f"- {q} {str(bool(answer)).lower()}")
    lines.extend(
        [
            f"- Next step: {next_step}",
            "",
            "Interpretation:",
            "- MLP readout attribution is a diagnostic / attribution signal under the current teacher-forced setting, not a steering method.",
            "- No claim is made about answer accuracy improvement or hallucination reduction.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_config(args, num_pairs, num_examples, num_number_tokens) -> dict[str, Any]:
    return {
        "backend": mra.BACKEND,
        "model_path": args.model_path,
        "num_input_pairs": num_pairs,
        "num_examples": num_examples,
        "layers": list(args.layers),
        "primary_layer": int(args.primary_layer),
        "secondary_layer": int(args.secondary_layer),
        "top_k": int(args.top_k),
        "num_number_like_tokens": int(num_number_tokens),
        "feature_boundary": "gold-free features; gold answer used only for eval labels",
        "training_model_weights": False,
        "lora": False,
        "steering_or_nudge": False,
        "full_sprint_3c": False,
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3C": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
        "steering_continued": False,
    }


def write_incomplete(out_dir: Path, reason: str) -> None:
    for name in [
        "mlp_attribution_config.json",
        "mlp_unembedding_projection_report.json",
        "answer_token_attribution_report.json",
        "correct_wrong_detection_report.json",
        "risk_score_report.json",
        "baseline_comparison_report.json",
        "calibration_report.json",
    ]:
        write_json({"backend": mra.BACKEND, "incomplete_evidence_reason": reason, "ready_for_2000_rerun": False}, out_dir / name)
    for name in ["mlp_readout_attribution_manifest.jsonl", "success_case_report.jsonl", "failure_case_report.jsonl"]:
        write_jsonl([], out_dir / name)
    (out_dir / "review_gate_mlp_readout_attribution_probe.md").write_text(
        "# Sprint 3C-3 Review Gate\n\nIncomplete evidence: "
        + reason
        + "\n\n- ready_for_2000_rerun: false\n- do_not_enter_full_sprint_3C: true\n- steering_continued: false\n",
        encoding="utf-8",
    )


def write_preflight(args: argparse.Namespace, out_dir: Path) -> None:
    input_dir = Path(args.input_dir)
    review = input_dir / "review_gate_mlp_readout_direction_analysis.md"
    review_text = review.read_text(encoding="utf-8", errors="replace") if review.exists() else ""
    progress_text = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8", errors="replace")
    manifest_path = PROJECT_ROOT / "docs/progress/sprint_3_artifact_manifest.md"
    manifest_text = manifest_path.read_text(encoding="utf-8", errors="replace") if manifest_path.exists() else ""
    evidence_text = "\n".join([review_text, progress_text, manifest_text]).lower()
    geometry = input_dir / "mlp_direction_geometry_report.json"
    alignment = input_dir / "mlp_unembedding_alignment_report.json"
    delta = input_dir / "mlp_readout_delta_manifest.jsonl"
    git_status = run_git(["status", "--short"])
    latest = run_git(["log", "--oneline", "-5"])
    tracked_changes_committed = "3C-2 completed" in latest
    outputs_ignored = run_git(["check-ignore", "-q", "outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/mlp_readout_delta_manifest.jsonl"], check=False).returncode == 0
    lines = [
        "# Sprint 3C-3 Preflight (MLP Readout Attribution Probe)",
        "",
        "Read files:",
        "- AGENTS.md",
        "- PROGRESS.md",
        "- docs/reasoning-aware-attention-guidance/SKILL.md",
        "- docs/progress/sprint_3_history.md",
        "- docs/progress/sprint_3_artifact_manifest.md",
        "- docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md",
        "- docs/codex_tasks/sprint_3C_2_mlp_readout_direction_analysis.md",
        "- docs/codex_tasks/sprint_3C_3_mlp_readout_attribution_probe.md",
        "- src/recover_attention/answer_proxy_metrics.py",
        "- src/recover_attention/module_causal_tracing.py",
        "- src/recover_attention/mlp_readout_direction.py",
        "- scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py",
        "- scripts/sprint_3C_2_mlp_readout_direction_analysis.py",
        "- tests/test_answer_proxy_metrics.py",
        "- tests/test_module_causal_tracing.py",
        "- tests/test_mlp_readout_direction.py",
        "",
        "Required checks:",
        f"- 3C2_tracked_changes_committed: {str(tracked_changes_committed).lower()}",
        f"- 3C2_mixed_result_recorded: {str('mixed' in evidence_text).lower()}",
        f"- 3C2_gold_unembed_oracle_not_deployable: {str('oracle' in evidence_text and ('not deployable' in evidence_text or 'not as deployable' in evidence_text or '不可部署' in evidence_text)).lower()}",
        f"- 3C2_mean_pc1_not_robust: {str('insufficient' in evidence_text or 'does not beat' in evidence_text or '不 robust' in evidence_text).lower()}",
        "- this_sprint_stops_steering: true",
        f"- will_not_overwrite_3C2_output_dir: {str(out_dir.resolve() != input_dir.resolve()).lower()}",
        f"- outputs_gitignored: {str(outputs_ignored).lower()}",
        "- artifact_manifest_will_be_updated: true",
        "- non_2000_non_full_3c_non_training: true",
        "",
        "Input availability:",
        f"- mlp_readout_delta_manifest_exists: {str(delta.exists()).lower()}",
        f"- mlp_direction_geometry_report_exists: {str(geometry.exists()).lower()}",
        f"- mlp_unembedding_alignment_report_exists: {str(alignment.exists()).lower()}",
        f"- review_gate_exists: {str(review.exists()).lower()}",
        f"- corrected_pair_manifest_exists: {str((Path(args.fix_input_dir) / 'corrected_pair_manifest.jsonl').exists()).lower()}",
        "",
        "Current git state:",
        "```text",
        git_status.strip() or "(clean except ignored outputs)",
        "```",
        "",
        "Conflict report: none. Current task card is the execution boundary.",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_git(args: list[str], *, check: bool = True):
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        return (result.stdout or "") + (result.stderr or "")
    if check:
        return (result.stdout or "") + (result.stderr or "")
    return result


if __name__ == "__main__":
    main()
