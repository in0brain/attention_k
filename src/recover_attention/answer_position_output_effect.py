"""Sprint 2K-W: response-position self-output-effect measurement.

The gate-eligible signal is the model's own next-token distribution shift after
adding an answer-eliciting prompt. Gold answers and solution paths are never used
to build features or formula scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention import answer_effect_features as oe
from recover_attention import multi_span_reasoning_scoring as msrs
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.hidden_state_cache import (
    import_torch,
    import_transformers_components,
    infer_model_input_device,
    is_module_available,
    resolve_torch_dtype,
)
from recover_attention.multi_span_reasoning_matrix import write_json

BACKEND = "answer_position_output_effect_v0"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
PRIMARY_TEMPLATE_ID = "A"
TEMPLATES = {
    "A": "Question: {question}\nAnswer:",
    "B": "Solve the problem. Give only the final answer.\n\nQuestion: {question}\nAnswer:",
}
FORBIDDEN_FEATURE_SUBSTRINGS = tuple(oe.BANNED_FEATURE_SUBSTRINGS)

FEATURE_MATRIX_FILENAME = "answer_position_feature_matrix.jsonl"
SCORE_MATRIX_FILENAME = "answer_position_score_matrix.jsonl"
FEATURE_AUDIT_FILENAME = "answer_position_feature_audit.json"
RANKING_REPORT_FILENAME = "answer_position_ranking_report.json"
FORMULA_REPORT_FILENAME = "answer_position_formula_validation_report.json"
BOOTSTRAP_REPORT_FILENAME = "answer_position_grouped_bootstrap_report.json"
COMPARISON_REPORT_FILENAME = "prompt_vs_answer_position_comparison.json"
SPAN_TYPE_BREAKDOWN_FILENAME = "span_type_answer_position_breakdown.json"
BUDGET_REPORT_FILENAME = "off_path_budget_report.json"
FAILURE_CASES_FILENAME = "failure_case_report.jsonl"
SUCCESS_CASES_FILENAME = "success_case_report.jsonl"
REVIEW_GATE_FILENAME = "review_gate_answer_position_output_effect.md"
FORWARD_MANIFEST_FILENAME = "answer_position_forward_manifest.jsonl"


def build_response_prompt(question: str, *, template_id: str = PRIMARY_TEMPLATE_ID) -> str:
    if template_id not in TEMPLATES:
        allowed = ", ".join(sorted(TEMPLATES))
        raise ValueError(f"unsupported template_id {template_id!r}; allowed: {allowed}")
    return TEMPLATES[template_id].format(question=question)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_local_logits_backend(*, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local model path for 2K-W: {model_path}")
    if not is_module_available("transformers"):
        raise RuntimeError("transformers is required for 2K-W response-position output-effect")
    if not is_module_available("bitsandbytes"):
        raise RuntimeError(
            "blocked_by_missing_bitsandbytes: 2K-W requires local 4-bit loading via "
            "bitsandbytes. No fp16 fallback, dependency install, or model download was attempted."
        )
    torch = import_torch()
    components = import_transformers_components()
    tokenizer = components["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    qcfg = components["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_compute_dtype=resolve_torch_dtype("float16", torch),
    )
    model = components["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=qcfg,
        device_map="auto",
        torch_dtype=resolve_torch_dtype("float16", torch),
    )
    model.eval()
    return {"torch": torch, "tokenizer": tokenizer, "model": model, "model_path": str(model_path)}


def forward_last_logits(context: dict[str, Any], prompt: str) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(prompt, return_tensors="pt")
    device = infer_model_input_device(model, "auto", torch)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :].detach().float().cpu()
    return {"last_logits": logits, "summary": summarize_logits(logits)}


def summarize_logits(logits: Any, *, topk: int = 10) -> dict[str, Any]:
    torch = import_torch()
    probs = torch.softmax(logits.float(), dim=-1)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    k = min(topk, int(probs.numel()))
    top = torch.topk(probs, k)
    top_ids = [int(i) for i in top.indices.tolist()]
    top_probs = [float(v) for v in top.values.tolist()]
    top_logprobs = [float(logprobs[i]) for i in top_ids]
    entropy = float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum().item())
    top2 = torch.topk(probs, min(2, int(probs.numel()))).values.tolist()
    margin = float(top2[0] - top2[1]) if len(top2) > 1 else float(top2[0])
    return {
        "topk_token_ids": top_ids,
        "topk_token_probs": top_probs,
        "topk_token_logprobs": top_logprobs,
        "entropy": entropy,
        "margin": margin,
        "top1_token_id": top_ids[0] if top_ids else None,
        "top1_token_logprob": top_logprobs[0] if top_logprobs else None,
    }


def run_answer_position_forward(
    *,
    input_feature_matrix_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    template_id: str = PRIMARY_TEMPLATE_ID,
    overwrite: bool = False,
    limit: int | None = None,
    report_every: int = 50,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    feature_path = output_dir / FEATURE_MATRIX_FILENAME
    manifest_path = output_dir / FORWARD_MANIFEST_FILENAME
    if overwrite:
        for path in [feature_path, manifest_path]:
            if path.exists():
                path.unlink()

    base_records = read_jsonl(input_feature_matrix_path)
    if limit is not None:
        base_records = base_records[:limit]
    done = _read_existing_by_span_id(feature_path)
    pending = [record for record in base_records if record["span_id"] not in done]
    if not pending:
        return list(done.values())

    context = load_local_logits_backend(model_path=model_path)
    original_cache: dict[tuple[str, str], dict[str, Any]] = {}
    t0 = time.time()
    for index, record in enumerate(pending, start=1):
        qid = str(record["source_question_id"])
        original_prompt = build_response_prompt(record["question"], template_id=template_id)
        masked_prompt = build_response_prompt(record["masked_question"], template_id=template_id)
        cache_key = (qid, template_id)
        original = original_cache.get(cache_key)
        if original is None:
            original = forward_last_logits(context, original_prompt)
            original_cache[cache_key] = original
        masked = forward_last_logits(context, masked_prompt)
        features = oe.compute_output_effect(original["last_logits"], masked["last_logits"])
        feature_record = build_feature_record(
            record,
            features,
            original_summary=original["summary"],
            masked_summary=masked["summary"],
            original_prompt=original_prompt,
            masked_prompt=masked_prompt,
            template_id=template_id,
        )
        manifest_record = build_forward_manifest_record(feature_record)
        _append_jsonl([feature_record], feature_path)
        _append_jsonl([manifest_record], manifest_path)
        done[record["span_id"]] = feature_record
        if report_every and index % report_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(
                "[2K-W] "
                f"processed={index}/{len(pending)} total_done={len(done)}/{len(base_records)} "
                f"rate={index / elapsed:.3f}/s"
            )
    return list(done.values())


def _read_existing_by_span_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["span_id"]): row for row in read_jsonl(path)}


def _append_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_feature_record(
    base_record: dict[str, Any],
    features: dict[str, float],
    *,
    original_summary: dict[str, Any],
    masked_summary: dict[str, Any],
    original_prompt: str,
    masked_prompt: str,
    template_id: str,
) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "source_question_id": base_record["source_question_id"],
        "span_id": base_record["span_id"],
        "span_text": base_record["span_text"],
        "span_type": base_record["span_type"],
        "question": base_record["question"],
        "masked_question": base_record["masked_question"],
        "template_id": template_id,
        "primary_template": TEMPLATES[PRIMARY_TEMPLATE_ID],
        "secondary_template": TEMPLATES.get("B"),
        "template_policy": "Template A primary; Template B diagnostic only",
        "original_prompt_hash": stable_hash(original_prompt),
        "masked_prompt_hash": stable_hash(masked_prompt),
        "resp_pos_output_effect": features,
        "resp_pos_output_shift": oe.output_effect_shift_score(features),
        "original_answer_position_summary": original_summary,
        "masked_answer_position_summary": masked_summary,
        "surface_features": base_record.get("surface_features", {}),
        "attention_features": base_record.get("attention_features", {}),
        "hidden_features": base_record.get("hidden_features", {}),
        "prompt_final_output_effect": base_record.get("output_effect_features", {}),
        "diagnostic_labels_for_eval_only": base_record.get("diagnostic_labels_for_eval_only", {}),
    }


def build_forward_manifest_record(feature_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": feature_record["span_id"],
        "source_question_id": feature_record["source_question_id"],
        "template_id": feature_record["template_id"],
        "original_prompt_hash": feature_record["original_prompt_hash"],
        "masked_prompt_hash": feature_record["masked_prompt_hash"],
        "original_topk_path": None,
        "masked_topk_path": None,
        "forward_status": "ok",
        "warnings": [],
    }


def feature_leakage_audit(feature_records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in feature_records for name in row.get("resp_pos_output_effect", {})})
    leaked = [
        {"feature_name": name, "forbidden_substring": token}
        for name in names
        for token in FORBIDDEN_FEATURE_SUBSTRINGS
        if token in name
    ]
    return {
        "passed": len(leaked) == 0,
        "num_feature_names_checked": len(names),
        "feature_names": names,
        "leaked_input_features": leaked,
        "forbidden_substrings": list(FORBIDDEN_FEATURE_SUBSTRINGS),
    }


def build_answer_position_reports(
    *,
    base_feature_matrix_path: str | Path,
    response_feature_matrix_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    base_records = read_jsonl(base_feature_matrix_path)
    response_records = read_jsonl(response_feature_matrix_path)
    score_records = build_score_matrix(base_records, response_records)
    formulas = build_formula_scores(score_records)
    ranking = build_ranking_report(score_records, formulas)
    bootstrap = build_bootstrap_report(score_records, formulas)
    budget = build_budget_report(score_records, formulas)
    comparison = build_prompt_vs_response_comparison(ranking, bootstrap)
    breakdown = build_span_type_breakdown(score_records, formulas)
    failures, successes = build_case_reports(score_records, formulas)
    audit = feature_leakage_audit(response_records)
    formula_report = build_formula_validation_report(ranking, bootstrap)
    review_gate = build_review_gate(
        audit=audit,
        ranking=ranking,
        bootstrap=bootstrap,
        budget=budget,
        breakdown=breakdown,
        failures=failures,
        successes=successes,
    )

    write_jsonl(score_records, output_dir / SCORE_MATRIX_FILENAME)
    write_json(audit, output_dir / FEATURE_AUDIT_FILENAME)
    write_json(ranking, output_dir / RANKING_REPORT_FILENAME)
    write_json(formula_report, output_dir / FORMULA_REPORT_FILENAME)
    write_json(bootstrap, output_dir / BOOTSTRAP_REPORT_FILENAME)
    write_json(comparison, output_dir / COMPARISON_REPORT_FILENAME)
    write_json(breakdown, output_dir / SPAN_TYPE_BREAKDOWN_FILENAME)
    write_json(budget, output_dir / BUDGET_REPORT_FILENAME)
    write_jsonl(failures, output_dir / FAILURE_CASES_FILENAME)
    write_jsonl(successes, output_dir / SUCCESS_CASES_FILENAME)
    (output_dir / REVIEW_GATE_FILENAME).write_text(render_review_gate(review_gate, ranking, bootstrap), encoding="utf-8")
    return {
        "score_records": score_records,
        "ranking": ranking,
        "bootstrap": bootstrap,
        "formula_report": formula_report,
        "review_gate": review_gate,
        "comparison": comparison,
    }


def build_score_matrix(
    base_records: list[dict[str, Any]],
    response_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response_by_id = {row["span_id"]: row for row in response_records}
    rows = []
    for base in base_records:
        response = response_by_id.get(base["span_id"])
        if response is None:
            continue
        prompt_features = base.get("output_effect_features", {}) or {}
        resp_features = response.get("resp_pos_output_effect", {}) or {}
        row = {
            "source_question_id": base["source_question_id"],
            "span_id": base["span_id"],
            "span_text": base["span_text"],
            "span_type": base["span_type"],
            "question": base.get("question", ""),
            "signals": {
                "surface_keyness": _safe(base.get("surface_features", {}).get("surface_keyness_proxy")),
                "attention_keyness": 0.0,
                "prompt_final_output_shift": oe.output_effect_shift_score(prompt_features),
                "resp_pos_output_shift": oe.output_effect_shift_score(resp_features),
            },
            "raw_output_effect": {
                "prompt_final_output_effect": prompt_features,
                "resp_pos_output_effect": resp_features,
            },
            "diagnostic_labels_for_eval_only": base.get("diagnostic_labels_for_eval_only", {}),
        }
        rows.append(row)

    score_view = msrs.build_score_matrix(base_records)
    attention_by_id = {
        row["span_id"]: _safe(row.get("fragility_signals", {}).get("attention_fragility_score"))
        for row in score_view
    }
    for row in rows:
        row["signals"]["attention_keyness"] = attention_by_id.get(row["span_id"], 0.0)

    for group in msrs._group_by_question(rows).values():
        _add_group_normalized(group, "prompt_final_output_shift", "prompt_final_output_shift_norm")
        _add_group_normalized(group, "resp_pos_output_shift", "resp_pos_output_shift_norm")
    return rows


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _add_group_normalized(group: list[dict[str, Any]], source_key: str, dest_key: str) -> None:
    values = [row["signals"][source_key] for row in group]
    lo = min(values) if values else 0.0
    hi = max(values) if values else 0.0
    denom = hi - lo
    for row in group:
        row["signals"][dest_key] = (row["signals"][source_key] - lo) / denom if denom > 1e-12 else 0.0


def build_formula_scores(score_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    formulas = {
        "A_surface": {"scores": {}, "gate_eligible": True},
        "B_attention_only": {"scores": {}, "gate_eligible": True},
        "C_prompt_final_output_only": {"scores": {}, "gate_eligible": True},
        "D_response_position_output_only": {"scores": {}, "gate_eligible": True},
        "E_attention_x_prompt_final_output": {"scores": {}, "gate_eligible": True},
        "F_attention_x_response_position_output": {"scores": {}, "gate_eligible": True},
        "G_response_output_gate_then_attention": {"scores": {}, "gate_eligible": True},
        "H_attention_gate_then_response_output": {"scores": {}, "gate_eligible": True},
        "I_mean_attention_response_output": {"scores": {}, "gate_eligible": True},
        "J_max_attention_response_output": {"scores": {}, "gate_eligible": True},
        "K_min_attention_response_output": {"scores": {}, "gate_eligible": True},
    }
    for group in msrs._group_by_question(score_records).values():
        resp_values = sorted(row["signals"]["resp_pos_output_shift_norm"] for row in group)
        attn_values = sorted(row["signals"]["attention_keyness"] for row in group)
        resp_median = resp_values[len(resp_values) // 2] if resp_values else 0.0
        attn_median = attn_values[len(attn_values) // 2] if attn_values else 0.0
        for row in group:
            sid = row["span_id"]
            s = row["signals"]
            attn = s["attention_keyness"]
            prompt = s["prompt_final_output_shift_norm"]
            resp = s["resp_pos_output_shift_norm"]
            formulas["A_surface"]["scores"][sid] = s["surface_keyness"]
            formulas["B_attention_only"]["scores"][sid] = attn
            formulas["C_prompt_final_output_only"]["scores"][sid] = prompt
            formulas["D_response_position_output_only"]["scores"][sid] = resp
            formulas["E_attention_x_prompt_final_output"]["scores"][sid] = attn * prompt
            formulas["F_attention_x_response_position_output"]["scores"][sid] = attn * resp
            formulas["G_response_output_gate_then_attention"]["scores"][sid] = attn if resp >= resp_median else 0.02 * attn
            formulas["H_attention_gate_then_response_output"]["scores"][sid] = resp if attn >= attn_median else 0.02 * resp
            formulas["I_mean_attention_response_output"]["scores"][sid] = 0.5 * (attn + resp)
            formulas["J_max_attention_response_output"]["scores"][sid] = max(attn, resp)
            formulas["K_min_attention_response_output"]["scores"][sid] = min(attn, resp)
    return formulas


def build_ranking_report(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "backend": BACKEND,
        "num_questions": len(msrs._group_by_question(score_records)),
        "num_spans": len(score_records),
        "formulas": {},
    }
    for name, spec in formulas.items():
        report["formulas"][name] = msrs.formula_metrics(score_records, spec["scores"], top_k=3)
    return report


def build_bootstrap_report(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparisons = [
        ("D_response_position_output_only", "C_prompt_final_output_only"),
        ("D_response_position_output_only", "A_surface"),
        ("D_response_position_output_only", "B_attention_only"),
        ("F_attention_x_response_position_output", "B_attention_only"),
        ("G_response_output_gate_then_attention", "B_attention_only"),
        ("H_attention_gate_then_response_output", "B_attention_only"),
        ("I_mean_attention_response_output", "B_attention_only"),
        ("J_max_attention_response_output", "B_attention_only"),
        ("K_min_attention_response_output", "B_attention_only"),
    ]
    results = []
    by_key = {}
    for formula, baseline in comparisons:
        result = msrs.grouped_bootstrap_delta(score_records, formulas, formula, baseline_name=baseline)
        if result is None:
            continue
        result = {
            "formula": formula,
            "baseline": baseline,
            "metric": "same_question_on_path_vs_off_path_auc",
            **result,
            "stable_positive": bool(result["ci95_low"] > 0),
        }
        results.append(result)
        by_key[f"{formula}__vs__{baseline}"] = result
    return {"backend": BACKEND, "comparisons": results, "by_key": by_key}


def build_formula_validation_report(
    ranking: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    formulas = ranking["formulas"]
    aucs = {
        name: metrics.get("same_question_on_path_vs_off_path_auc")
        for name, metrics in formulas.items()
    }
    best = max(aucs, key=lambda name: _safe(aucs[name], -1.0))
    return {
        "backend": BACKEND,
        "primary_metric": "same_question_on_path_vs_off_path_auc",
        "formula_auc": aucs,
        "best_formula_by_auc": best,
        "response_position_vs_prompt_final": bootstrap["by_key"].get(
            "D_response_position_output_only__vs__C_prompt_final_output_only"
        ),
        "response_position_vs_attention": bootstrap["by_key"].get(
            "D_response_position_output_only__vs__B_attention_only"
        ),
        "attention_x_response_vs_attention": bootstrap["by_key"].get(
            "F_attention_x_response_position_output__vs__B_attention_only"
        ),
        "interpretation": interpret_formula_result(aucs, bootstrap),
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }


def interpret_formula_result(aucs: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    resp_vs_prompt = bootstrap["by_key"].get("D_response_position_output_only__vs__C_prompt_final_output_only", {})
    resp_vs_attn = bootstrap["by_key"].get("D_response_position_output_only__vs__B_attention_only", {})
    fusion_vs_attn = bootstrap["by_key"].get("F_attention_x_response_position_output__vs__B_attention_only", {})
    if resp_vs_prompt.get("stable_positive") and resp_vs_attn.get("stable_positive"):
        if fusion_vs_attn.get("stable_positive"):
            return "strong_success"
        return "partial_success_response_position_improves_but_fusion_not_stable"
    if not resp_vs_prompt.get("stable_positive") and not resp_vs_attn.get("stable_positive"):
        return "negative_result_attention_only_remains_primary"
    return "mixed_result"


def build_budget_report(score_records: list[dict[str, Any]], formulas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "note": "top-k coverage is auxiliary and does not replace same-question on/off-path AUC",
        "formulas": {
            name: _budget_metrics(score_records, spec["scores"])
            for name, spec in formulas.items()
        },
    }


def _budget_metrics(rows: list[dict[str, Any]], scores: dict[str, float], *, top_k: int = 3) -> dict[str, Any]:
    metrics = msrs.formula_metrics(rows, scores, top_k=top_k)
    return {
        key: metrics.get(key)
        for key in [
            "off_path_budget_share",
            "top_k_off_path_number_selected_rate",
            "per_question_top1_key_hit",
            "per_question_top2_key_hit",
            "per_question_top3_key_hit",
            "span_type_budget_distribution",
            "comparison_topk_coverage",
            "negation_topk_coverage",
            "condition_topk_coverage",
            "operation_topk_coverage",
            "question_target_topk_coverage",
        ]
    }


def build_prompt_vs_response_comparison(ranking: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    formulas = ranking["formulas"]
    prompt = formulas["C_prompt_final_output_only"]
    response = formulas["D_response_position_output_only"]
    attention = formulas["B_attention_only"]
    surface = formulas["A_surface"]
    delta = bootstrap["by_key"].get("D_response_position_output_only__vs__C_prompt_final_output_only")
    return {
        "prompt_final_output_effect": {
            "auc": prompt.get("same_question_on_path_vs_off_path_auc"),
            "vs_surface": _auc_delta(prompt, surface),
            "vs_attention": _auc_delta(prompt, attention),
        },
        "response_position_output_effect": {
            "auc": response.get("same_question_on_path_vs_off_path_auc"),
            "vs_surface": _auc_delta(response, surface),
            "vs_attention": _auc_delta(response, attention),
        },
        "delta_response_minus_prompt": {
            "auc_delta": _safe(response.get("same_question_on_path_vs_off_path_auc"))
            - _safe(prompt.get("same_question_on_path_vs_off_path_auc")),
            "bootstrap_ci95": [delta.get("ci95_low"), delta.get("ci95_high")] if delta else None,
            "stable_positive": delta.get("stable_positive") if delta else None,
        },
        "answers": {
            "response_position_stronger_than_prompt_final": bool(delta and delta.get("stable_positive")),
            "prompt_final_underestimated": bool(delta and delta.get("stable_positive")),
            "response_position_near_or_above_attention": _safe(response.get("same_question_on_path_vs_off_path_auc"))
            >= _safe(attention.get("same_question_on_path_vs_off_path_auc")) - 0.01,
        },
        "interpretation": "response-position is evaluated at the first token after an answer-eliciting prompt",
    }


def _auc_delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    return {
        "auc_delta": _safe(first.get("same_question_on_path_vs_off_path_auc"))
        - _safe(second.get("same_question_on_path_vs_off_path_auc"))
    }


def build_span_type_breakdown(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_resp = _selected_span_ids(score_records, formulas["D_response_position_output_only"]["scores"])
    selected_attention = _selected_span_ids(score_records, formulas["B_attention_only"]["scores"])
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_records:
        by_type[str(row["span_type"])].append(row)
    rows = []
    for span_type, group in sorted(by_type.items()):
        number_like = span_type in {"number", "number_unit", "rate"}
        rows.append(
            {
                "span_type": span_type,
                "num_cases": len(group),
                "mean_attention_keyness": _mean([r["signals"]["attention_keyness"] for r in group]),
                "mean_prompt_final_output_shift": _mean([r["signals"]["prompt_final_output_shift_norm"] for r in group]),
                "mean_resp_pos_output_shift": _mean([r["signals"]["resp_pos_output_shift_norm"] for r in group]),
                "resp_minus_prompt_mean_delta": _mean(
                    [
                        r["signals"]["resp_pos_output_shift_norm"] - r["signals"]["prompt_final_output_shift_norm"]
                        for r in group
                    ]
                ),
                "topk_selected_rate_by_resp_pos": sum(r["span_id"] in selected_resp for r in group) / len(group),
                "topk_selected_rate_by_attention": sum(r["span_id"] in selected_attention for r in group) / len(group),
                "off_path_frac_if_number_like": (
                    sum(msrs.is_off_path_number(r) for r in group) / len(group) if number_like else None
                ),
            }
        )
    return {"backend": BACKEND, "span_types": rows, "answers": interpret_span_breakdown(rows)}


def _selected_span_ids(rows: list[dict[str, Any]], scores: dict[str, float], *, top_k: int = 3) -> set[str]:
    selected = set()
    for group in msrs._group_by_question(rows).values():
        ordered = sorted(group, key=lambda row: scores.get(row["span_id"], 0.0), reverse=True)
        selected.update(row["span_id"] for row in ordered[:top_k])
    return selected


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def interpret_span_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = {row["span_type"]: row for row in rows}
    def delta(span_type: str) -> float | None:
        row = by_type.get(span_type)
        return None if row is None else row.get("resp_minus_prompt_mean_delta")
    return {
        "number_unit_or_rate_more_sensitive": any((delta(t) or 0.0) > 0 for t in ["number_unit", "rate"]),
        "question_target_more_sensitive": (delta("question_target") or 0.0) > 0,
        "comparison_condition_negation_improved": any((delta(t) or 0.0) > 0 for t in ["comparison", "condition", "negation"]),
        "off_path_number_still_risk": any(
            row["span_type"] in {"number", "number_unit", "rate"}
            and (row.get("off_path_frac_if_number_like") or 0.0) > 0.1
            and row["topk_selected_rate_by_resp_pos"] > 0
            for row in rows
        ),
        "object_overselected": (by_type.get("object", {}).get("topk_selected_rate_by_resp_pos") or 0.0)
        > (by_type.get("object", {}).get("topk_selected_rate_by_attention") or 0.0),
    }


def build_case_reports(
    score_records: list[dict[str, Any]],
    formulas: dict[str, dict[str, Any]],
    *,
    per_type: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_scores = formulas["C_prompt_final_output_only"]["scores"]
    resp_scores = formulas["D_response_position_output_only"]["scores"]
    attn_scores = formulas["B_attention_only"]["scores"]
    fusion_scores = formulas["F_attention_x_response_position_output"]["scores"]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, group in msrs._group_by_question(score_records).items():
        on_rows = [row for row in group if msrs.is_on_path_number(row)]
        off_rows = [row for row in group if msrs.is_off_path_number(row)]
        if not on_rows or not off_rows:
            continue
        prompt_ok = _question_pair_success(on_rows, off_rows, prompt_scores)
        resp_ok = _question_pair_success(on_rows, off_rows, resp_scores)
        attn_ok = _question_pair_success(on_rows, off_rows, attn_scores)
        fusion_ok = _question_pair_success(on_rows, off_rows, fusion_scores)
        if resp_ok and not prompt_ok:
            buckets["resp_pos_fixes_prompt_final"].append(_case(qid, group, "resp_pos_fixes_prompt_final", formulas))
        if resp_ok and not attn_ok:
            buckets["resp_pos_beats_attention"].append(_case(qid, group, "resp_pos_beats_attention", formulas))
        if attn_ok and not resp_ok:
            buckets["attention_beats_resp_pos"].append(_case(qid, group, "attention_beats_resp_pos", formulas))
        if not attn_ok and not resp_ok:
            buckets["both_fail"].append(_case(qid, group, "both_fail", formulas))
        if fusion_ok and not attn_ok:
            buckets["attention_resp_pos_fusion_succeeds"].append(_case(qid, group, "attention_resp_pos_fusion_succeeds", formulas))

    successes = []
    failures = []
    for case_type, cases in buckets.items():
        target = successes if case_type in {
            "resp_pos_fixes_prompt_final",
            "resp_pos_beats_attention",
            "attention_resp_pos_fusion_succeeds",
        } else failures
        target.extend(cases[:per_type])
    return failures, successes


def _question_pair_success(on_rows: list[dict[str, Any]], off_rows: list[dict[str, Any]], scores: dict[str, float]) -> bool:
    values = []
    for on in on_rows:
        for off in off_rows:
            values.append(msrs._pairwise_value(scores.get(on["span_id"], 0.0), scores.get(off["span_id"], 0.0)))
    return bool(values) and _mean(values) >= 0.5


def _case(qid: str, group: list[dict[str, Any]], case_type: str, formulas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranks = {
        name: msrs._ranks_for_group(group, spec["scores"])
        for name, spec in formulas.items()
    }
    return {
        "source_question_id": qid,
        "question": group[0].get("question", ""),
        "case_type": case_type,
        "auto_interpretation": case_type,
        "candidate_spans": [
            {
                "span_text": row["span_text"],
                "span_type": row["span_type"],
                "solution_path_status_eval_only": row.get("diagnostic_labels_for_eval_only", {}).get("solution_path_status"),
                "attention_score": row["signals"]["attention_keyness"],
                "prompt_final_output_shift": row["signals"]["prompt_final_output_shift_norm"],
                "resp_pos_output_shift": row["signals"]["resp_pos_output_shift_norm"],
                "formula_scores": {
                    name: spec["scores"].get(row["span_id"])
                    for name, spec in formulas.items()
                },
                "ranks": {
                    name: ranks[name].get(row["span_id"])
                    for name in formulas
                },
            }
            for row in group
        ],
    }


def build_review_gate(
    *,
    audit: dict[str, Any],
    ranking: dict[str, Any],
    bootstrap: dict[str, Any],
    budget: dict[str, Any],
    breakdown: dict[str, Any],
    failures: list[dict[str, Any]],
    successes: list[dict[str, Any]],
) -> dict[str, Any]:
    formulas = ranking.get("formulas", {})
    checks = {
        "1_features_generated": {"pass": ranking.get("num_spans", 0) > 0, "value": ranking.get("num_spans")},
        "2_feature_leakage_audit_passed": {"pass": bool(audit.get("passed")), "value": audit},
        "3_auc_reported_for_all_formulas": {
            "pass": all(m.get("same_question_on_path_vs_off_path_auc") is not None for m in formulas.values()),
            "value": list(formulas),
        },
        "4_response_compared_against_prompt_final": {
            "pass": "D_response_position_output_only__vs__C_prompt_final_output_only" in bootstrap.get("by_key", {}),
            "value": bootstrap.get("by_key", {}).get("D_response_position_output_only__vs__C_prompt_final_output_only"),
        },
        "5_response_compared_against_attention": {
            "pass": "D_response_position_output_only__vs__B_attention_only" in bootstrap.get("by_key", {}),
            "value": bootstrap.get("by_key", {}).get("D_response_position_output_only__vs__B_attention_only"),
        },
        "6_grouped_bootstrap_computed": {"pass": len(bootstrap.get("comparisons", [])) >= 9, "value": len(bootstrap.get("comparisons", []))},
        "7_budget_metrics_generated": {"pass": bool(budget.get("formulas")), "value": len(budget.get("formulas", {}))},
        "8_span_type_breakdown_generated": {"pass": bool(breakdown.get("span_types")), "value": len(breakdown.get("span_types", []))},
        "9_failure_success_cases_generated": {"pass": bool(failures or successes), "value": {"failures": len(failures), "successes": len(successes)}},
        "10_ready_for_2000_rerun_false": {"pass": True, "value": False},
        "11_do_not_enter_sprint_3A_true": {"pass": True, "value": True},
    }
    passed = sum(1 for check in checks.values() if check["pass"])
    return {
        "backend": BACKEND,
        "checks": checks,
        "num_checks_passed": passed,
        "num_checks_total": len(checks),
        "passed": passed == len(checks),
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }


def render_review_gate(review_gate: dict[str, Any], ranking: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    formulas = ranking["formulas"]
    def auc(name: str) -> Any:
        return formulas[name].get("same_question_on_path_vs_off_path_auc")
    resp_vs_prompt = bootstrap["by_key"].get("D_response_position_output_only__vs__C_prompt_final_output_only", {})
    resp_vs_attn = bootstrap["by_key"].get("D_response_position_output_only__vs__B_attention_only", {})
    fusion_vs_attn = bootstrap["by_key"].get("F_attention_x_response_position_output__vs__B_attention_only", {})
    recommendation = (
        "attention+response-position output-effect"
        if fusion_vs_attn.get("stable_positive")
        else "attention-only"
    )
    lines = [
        "# Sprint 2K-W Answer-Position Output-Effect Review Gate",
        "",
        "Verdict:",
        f"- passed: {str(review_gate['passed']).lower()} ({review_gate['num_checks_passed']}/{review_gate['num_checks_total']})",
        f"- ready_for_2000_rerun: {str(review_gate['ready_for_2000_rerun']).lower()}",
        f"- do_not_enter_sprint_3A: {str(review_gate['do_not_enter_sprint_3A']).lower()}",
        "",
        "Input and boundary:",
        "- Reused 2J-Fix / 2K / 2K-V 4935-span artifacts.",
        "- Reran only original/masked answer-position logits; no recovery, CoT, trajectory, NLA, causal attribution, or attention steering.",
        "",
        "Feature leakage audit:",
        f"- passed: {str(review_gate['checks']['2_feature_leakage_audit_passed']['pass']).lower()}",
        "",
        "Primary AUC:",
        f"- surface: {_fmt(auc('A_surface'))}",
        f"- attention-only: {_fmt(auc('B_attention_only'))}",
        f"- prompt-final output-effect: {_fmt(auc('C_prompt_final_output_only'))}",
        f"- response-position output-effect: {_fmt(auc('D_response_position_output_only'))}",
        f"- attention x response-position output-effect: {_fmt(auc('F_attention_x_response_position_output'))}",
        "",
        "Prompt-final vs response-position:",
        f"- response-position vs prompt-final: {resp_vs_prompt}",
        f"- response-position vs attention-only: {resp_vs_attn}",
        "",
        "Grouped bootstrap:",
        f"- attention x response-position vs attention-only: {fusion_vs_attn}",
        "",
        "Budget analysis:",
        "- See off_path_budget_report.json.",
        "",
        "Span-type breakdown:",
        "- See span_type_answer_position_breakdown.json.",
        "",
        "Failure/success cases:",
        "- See failure_case_report.jsonl and success_case_report.jsonl.",
        "",
        "Root cause:",
        "- This sprint tests whether the output-effect measurement point was too early at prompt-final position.",
        "",
        "Recommendation:",
        f"- Current steering target recommendation: {recommendation}.",
        "",
        "Next sprint candidate:",
        "- formula validation or 3A-0 smoke test only; do not enter full Sprint 3A directly.",
        "",
        "Required final questions:",
        f"- response-position stronger than prompt-final: {bool(resp_vs_prompt.get('stable_positive'))}",
        f"- prompt-final underestimated: {bool(resp_vs_prompt.get('stable_positive'))}",
        f"- response-position stable above surface: {bool(bootstrap['by_key'].get('D_response_position_output_only__vs__A_surface', {}).get('stable_positive'))}",
        f"- response-position stable above attention-only: {bool(resp_vs_attn.get('stable_positive'))}",
        f"- attention x response-position stable above attention-only: {bool(fusion_vs_attn.get('stable_positive'))}",
        f"- recommend 2000: false",
        f"- recommend Sprint 3A: false",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def run_2kw(
    *,
    base_feature_matrix_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    template_id: str = PRIMARY_TEMPLATE_ID,
    overwrite: bool = False,
    limit: int | None = None,
    report_every: int = 50,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    features = run_answer_position_forward(
        input_feature_matrix_path=base_feature_matrix_path,
        output_dir=output_dir,
        model_path=model_path,
        template_id=template_id,
        overwrite=overwrite,
        limit=limit,
        report_every=report_every,
    )
    artifacts = build_answer_position_reports(
        base_feature_matrix_path=base_feature_matrix_path,
        response_feature_matrix_path=output_dir / FEATURE_MATRIX_FILENAME,
        output_dir=output_dir,
    )
    ranking = artifacts["ranking"]
    formula_report = artifacts["formula_report"]
    return {
        "backend": BACKEND,
        "output_dir": str(output_dir),
        "num_feature_records": len(features),
        "num_score_records": len(artifacts["score_records"]),
        "review_gate_passed": artifacts["review_gate"]["passed"],
        "checks_passed": artifacts["review_gate"]["num_checks_passed"],
        "checks_total": artifacts["review_gate"]["num_checks_total"],
        "best_formula_by_auc": formula_report["best_formula_by_auc"],
        "attention_auc": ranking["formulas"]["B_attention_only"]["same_question_on_path_vs_off_path_auc"],
        "prompt_final_auc": ranking["formulas"]["C_prompt_final_output_only"]["same_question_on_path_vs_off_path_auc"],
        "response_position_auc": ranking["formulas"]["D_response_position_output_only"]["same_question_on_path_vs_off_path_auc"],
        "attention_x_response_auc": ranking["formulas"]["F_attention_x_response_position_output"]["same_question_on_path_vs_off_path_auc"],
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }

