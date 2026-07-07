"""Sprint 3A-0: increase-only attention-bias steering smoke test.

This module implements a small feasibility test for additive attention logit
bias. It is intentionally not a full steering system: no model weights are
trained, no distractor suppression is applied, and oracle labels are only used
for sanity-check selectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention import answer_effect_features as oe
from recover_attention import multi_span_reasoning_scoring as msrs
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.multi_span_reasoning_matrix import write_json

BACKEND = "attention_bias_steering_smoke_v0"
DEFAULT_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
DEFAULT_PRIMARY_N = 50
DEFAULT_DIAGNOSTIC_QUERY_N = 5
DEFAULT_SEED = 23030

OUTPUT_DIRNAME = "sprint_3A_0_attention_bias_steering_smoke_test"
SUBSET_FILENAME = "steering_subset_manifest.jsonl"
TARGET_SELECTOR_FILENAME = "target_selector_report.json"
BIAS_CONFIG_FILENAME = "attention_bias_config.json"
FORWARD_MANIFEST_FILENAME = "steering_forward_manifest.jsonl"
MASS_REPORT_FILENAME = "attention_mass_fidelity_report.json"
OUTPUT_SHIFT_REPORT_FILENAME = "answer_position_output_shift_report.json"
GENERATION_REPORT_FILENAME = "steering_generation_report.json"
ORACLE_REPORT_FILENAME = "oracle_sanity_report.json"
HARM_REPORT_FILENAME = "harm_rate_report.json"
BASELINE_REPORT_FILENAME = "baseline_comparison_report.json"
FAILURE_CASE_FILENAME = "failure_case_report.jsonl"
SUCCESS_CASE_FILENAME = "success_case_report.jsonl"
REVIEW_GATE_FILENAME = "review_gate_attention_bias_smoke_test.md"
MASS_TRACE_FILENAME = "attention_mass_before_after.jsonl"
HOOK_TRACE_FILENAME = "debug_hook_trace.jsonl"

SELECTORS = [
    "random",
    "surface",
    "attention_only",
    "attention_x_resp_pos",
    "oracle",
]
TOP_K_VALUES = [1, 2, 3]
LAMBDA_GRID = [0.05, 0.1, 0.2, 0.4]
PRIMARY_LAMBDA = 0.2
LAYER_CONFIGS = [[16], [24], [16, 24]]
PRIMARY_LAYER_CONFIG = [16, 24]
PRIMARY_QUERY_SCOPE = "answer_position"
DIAGNOSTIC_QUERY_SCOPES = ["question_focus", "operation"]
HEAD_SCOPE = "all_heads"
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

QFOCUS_RE = re.compile(r"\b(how many|how much|how far|what|which|how)\b", re.IGNORECASE)
OPERATION_RE = re.compile(
    r"\b(total|each|per|more|less|fewer|times|twice|double|half|sum|difference|"
    r"remaining|left|altogether|combined|plus|minus|every|apiece|bought|sold|"
    r"gave|spent|cost|earned|lost|shared|divided|increased|decreased)\b",
    re.IGNORECASE,
)


def build_response_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_local_steering_backend(*, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the same strict local 4-bit eager backend used by 2J-B attention scoring."""

    return msrs.load_local_attention_backend(model_path=model_path)


def question_span_to_prompt_range(question: str, start: int, end: int) -> list[int]:
    prompt = build_response_prompt(question)
    prefix = prompt.index(question)
    return [prefix + int(start), prefix + int(end)]


def token_indices_for_prompt_span(
    offsets: list[list[int]],
    question: str,
    start: int,
    end: int,
) -> list[int]:
    return msrs.token_indices_for_char_ranges(
        offsets,
        [question_span_to_prompt_range(question, start, end)],
        exclude=set(),
    )


def query_indices_for_scope(
    offsets: list[list[int]],
    question: str,
    input_length: int,
    *,
    query_scope: str,
) -> list[int]:
    prompt = build_response_prompt(question)
    prefix = prompt.index(question)
    if query_scope == "answer_position":
        return [input_length - 1]
    if query_scope == "question_focus":
        ranges = [[prefix + m.start(), prefix + m.end()] for m in QFOCUS_RE.finditer(question)]
    elif query_scope == "operation":
        ranges = [[prefix + m.start(), prefix + m.end()] for m in OPERATION_RE.finditer(question)]
    else:
        raise ValueError(f"unsupported query_scope: {query_scope}")
    return msrs.token_indices_for_char_ranges(offsets, ranges, exclude=set())


def build_guidance_bias(
    *,
    torch: Any,
    seq_len: int,
    query_indices: list[int],
    key_indices: list[int],
    bias_lambda: float,
    device: Any,
    dtype: Any,
) -> Any:
    """Build an additive logit bias tensor [1, 1, seq, seq].

    The tensor is zero everywhere except selected query/key positions where a
    positive lambda is added before attention softmax.
    """

    if bias_lambda < 0:
        raise ValueError("3A-0 only supports positive attention boost; negative bias is forbidden")
    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    valid_queries = [idx for idx in query_indices if 0 <= idx < seq_len]
    valid_keys = [idx for idx in key_indices if 0 <= idx < seq_len]
    if valid_queries and valid_keys and bias_lambda:
        q = torch.tensor(valid_queries, device=device, dtype=torch.long)
        k = torch.tensor(valid_keys, device=device, dtype=torch.long)
        bias[:, :, q[:, None], k[None, :]] += float(bias_lambda)
    return bias


class AttentionBiasHookHandle:
    def __init__(self, patched: list[tuple[Any, Any]], trace: dict[str, Any]):
        self._patched = patched
        self.trace = trace

    def remove(self) -> None:
        for module, original_forward in self._patched:
            module.forward = original_forward
        self.trace["hook_removed"] = True


def register_attention_bias_hooks(
    model: Any,
    *,
    guidance_bias_by_layer: dict[int, Any],
    trace: dict[str, Any],
) -> AttentionBiasHookHandle:
    """Patch selected Qwen attention modules and return a removable handle."""

    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception as exc:  # pragma: no cover - only hit on unsupported transformers
        raise RuntimeError(f"Qwen2 attention patching is unavailable: {exc}") from exc

    patched: list[tuple[Any, Any]] = []
    trace["hook_registered"] = False
    trace["hook_triggered_layers"] = []
    trace["warnings"] = list(trace.get("warnings") or [])

    wanted = set(int(layer) for layer in guidance_bias_by_layer)
    for module in model.modules():
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or int(layer_idx) not in wanted:
            continue
        if not hasattr(module, "q_proj") or not hasattr(module, "o_proj"):
            continue
        original_forward = module.forward
        bias_tensor = guidance_bias_by_layer[int(layer_idx)]

        def biased_forward(
            self: Any,
            hidden_states: Any,
            position_embeddings: tuple[Any, Any],
            attention_mask: Any | None = None,
            past_key_values: Any | None = None,
            _bias: Any = bias_tensor,
            _layer_idx: int = int(layer_idx),
            **kwargs: Any,
        ) -> tuple[Any, Any | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )

            key_states = modeling_qwen2.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_qwen2.repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = query_states @ key_states.transpose(2, 3)
            attn_weights = attn_weights * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            local_bias = _bias.to(device=attn_weights.device, dtype=attn_weights.dtype)
            attn_weights = attn_weights + local_bias[..., : attn_weights.shape[-2], : attn_weights.shape[-1]]
            attn_weights = modeling_qwen2.nn.functional.softmax(
                attn_weights,
                dim=-1,
                dtype=modeling_qwen2.torch.float32,
            ).to(query_states.dtype)
            attn_weights = modeling_qwen2.nn.functional.dropout(
                attn_weights,
                p=0.0 if not self.training else self.attention_dropout,
                training=self.training,
            )
            attn_output = attn_weights @ value_states
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            if _layer_idx not in trace["hook_triggered_layers"]:
                trace["hook_triggered_layers"].append(_layer_idx)
            return attn_output, attn_weights

        module.forward = types.MethodType(biased_forward, module)
        patched.append((module, original_forward))

    trace["hook_registered"] = bool(patched)
    missing = sorted(wanted - {int(getattr(m, "layer_idx")) for m, _ in patched})
    if missing:
        trace["warnings"].append({"warning_type": "requested_layers_not_patched", "layers": missing})
    return AttentionBiasHookHandle(patched, trace)


def remove_attention_bias_hooks(handle: AttentionBiasHookHandle) -> None:
    handle.remove()


def run_steered_forward(
    context: dict[str, Any],
    prompt: str,
    *,
    query_indices: list[int],
    key_indices: list[int],
    layers: list[int],
    bias_lambda: float,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(start), int(end)] for start, end in encoded.pop("offset_mapping")[0].tolist()]
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    seq_len = int(model_inputs["input_ids"].shape[-1])
    if trace is None:
        trace = {}
    guidance_bias = build_guidance_bias(
        torch=torch,
        seq_len=seq_len,
        query_indices=query_indices,
        key_indices=key_indices,
        bias_lambda=bias_lambda,
        device=target_device,
        dtype=model_inputs["input_ids"].dtype if False else torch.float32,
    )
    handle = register_attention_bias_hooks(
        model,
        guidance_bias_by_layer={int(layer): guidance_bias for layer in layers},
        trace=trace,
    )
    try:
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        handle.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu()
    return {
        "logits": logits,
        "attentions": outputs.attentions,
        "offsets": offsets,
        "input_ids": model_inputs["input_ids"].detach().cpu(),
        "trace": trace,
        "seq_len": seq_len,
    }


def run_no_steering_forward(context: dict[str, Any], prompt: str) -> dict[str, Any]:
    torch = context["torch"]
    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(start), int(end)] for start, end in encoded.pop("offset_mapping")[0].tolist()]
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            use_cache=False,
        )
    logits = outputs.logits[0, -1, :].detach().float().cpu()
    return {
        "logits": logits,
        "attentions": outputs.attentions,
        "offsets": offsets,
        "input_ids": model_inputs["input_ids"].detach().cpu(),
        "seq_len": int(model_inputs["input_ids"].shape[-1]),
    }


def compute_attention_mass(
    attentions: Any,
    *,
    layers: list[int],
    query_indices: list[int],
    key_indices: list[int],
) -> dict[str, Any]:
    per_layer: dict[str, float] = {}
    if not query_indices or not key_indices:
        return {
            "target_attention_mass": 0.0,
            "non_target_attention_mass": 0.0,
            "per_layer_target_attention_mass": per_layer,
        }
    vals = []
    non_vals = []
    for layer in layers:
        if layer >= len(attentions):
            continue
        tensor = attentions[layer][0].detach().float().cpu()  # [heads, query, key]
        q = [idx for idx in query_indices if 0 <= idx < tensor.shape[-2]]
        k = [idx for idx in key_indices if 0 <= idx < tensor.shape[-1]]
        if not q or not k:
            continue
        q_arr = np.array(q, dtype=int)
        k_arr = np.array(k, dtype=int)
        rows = tensor[:, q_arr, :].numpy()
        target = float(rows[:, :, k_arr].sum(axis=-1).mean())
        total = float(rows.sum(axis=-1).mean())
        per_layer[str(layer)] = target
        vals.append(target)
        non_vals.append(total - target)
    return {
        "target_attention_mass": float(np.mean(vals)) if vals else 0.0,
        "non_target_attention_mass": float(np.mean(non_vals)) if non_vals else 0.0,
        "per_layer_target_attention_mass": per_layer,
    }


def compute_answer_position_output_shift(base_logits: Any, steered_logits: Any) -> dict[str, Any]:
    raw = oe.compute_output_effect(base_logits, steered_logits)
    mapped = {
        "steer_output_kl": raw["self_output_kl"],
        "steer_output_js": raw["self_output_js"],
        "steer_top1_changed": raw["self_output_top1_changed"],
        "steer_topk_overlap": raw["self_output_topk_overlap"],
        "steer_entropy_delta": raw["self_output_entropy_delta"],
        "steer_margin_delta": raw["self_output_margin_delta"],
        "steer_logprob_delta": raw["self_output_logprob_delta"],
    }
    nonfinite = [name for name, value in mapped.items() if not _is_finite_number(value)]
    sanitized = {name: _finite_float(value) for name, value in mapped.items()}
    sanitized["nonfinite_shift_fields"] = nonfinite
    return sanitized


def summarize_logits(logits: Any, tokenizer: Any, *, topk: int = 5) -> dict[str, Any]:
    torch = msrs.import_torch()
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, min(topk, int(probs.numel())))
    ids = [int(i) for i in top.indices.tolist()]
    return {
        "top_token_ids": ids,
        "top_token_probs": [float(v) for v in top.values.tolist()],
        "top_token_text": [tokenizer.decode([i]) for i in ids],
        "top1_token_id": ids[0] if ids else None,
        "top1_token_text": tokenizer.decode([ids[0]]) if ids else None,
        "entropy": float(-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum().item()),
        "margin": float(top.values[0] - top.values[1]) if len(top.values) > 1 else float(top.values[0]),
    }


def load_steering_source_records(
    *,
    feature_matrix_path: str | Path,
    answer_position_score_matrix_path: str | Path,
) -> list[dict[str, Any]]:
    features = {str(row["span_id"]): row for row in read_jsonl(feature_matrix_path)}
    scores = read_jsonl(answer_position_score_matrix_path)
    records: list[dict[str, Any]] = []
    for score in scores:
        feature = features.get(str(score["span_id"]))
        if not feature:
            continue
        signals = score.get("signals") or {}
        labels = score.get("diagnostic_labels_for_eval_only") or {}
        attention = float(signals.get("attention_keyness") or 0.0)
        resp_norm = float(signals.get("resp_pos_output_shift_norm") or 0.0)
        records.append(
            {
                "source_question_id": str(score["source_question_id"]),
                "span_id": str(score["span_id"]),
                "question": score.get("question") or feature.get("question") or "",
                "span_text": score.get("span_text") or feature.get("span_text") or "",
                "span_type": score.get("span_type") or feature.get("span_type") or "",
                "span_char_start": int(feature["span_char_start"]),
                "span_char_end": int(feature["span_char_end"]),
                "surface_keyness": float(signals.get("surface_keyness") or 0.0),
                "attention_keyness": attention,
                "resp_pos_output_shift": float(signals.get("resp_pos_output_shift") or 0.0),
                "resp_pos_output_shift_norm": resp_norm,
                "attention_x_resp_pos_score": attention * resp_norm,
                "solution_path_status_eval_only": labels.get("solution_path_status"),
                "weak_semantic_keyness_eval_only": labels.get("weak_semantic_keyness"),
            }
        )
    return records


def group_by_question(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_question_id"])].append(record)
    for spans in grouped.values():
        spans.sort(key=lambda row: (int(row["span_char_start"]), row["span_id"]))
    return dict(grouped)


def is_on_path_number(record: dict[str, Any]) -> bool:
    return record.get("solution_path_status_eval_only") in {
        "on_path_number",
        "on_solution_path_number",
    }


def is_off_path_number(record: dict[str, Any]) -> bool:
    return record.get("solution_path_status_eval_only") in {
        "off_path_number",
        "off_solution_path_number",
    }


def selector_score(record: dict[str, Any], selector: str) -> float:
    if selector == "surface":
        return float(record.get("surface_keyness") or 0.0)
    if selector == "attention_only":
        return float(record.get("attention_keyness") or 0.0)
    if selector == "attention_x_resp_pos":
        return float(record.get("attention_x_resp_pos_score") or 0.0)
    if selector == "oracle":
        return 1.0 if is_on_path_number(record) else 0.0
    raise ValueError(f"selector {selector!r} does not use deterministic score")


def select_spans_for_selector(
    spans: list[dict[str, Any]],
    *,
    selector: str,
    top_k: int,
    seed: int,
    question_id: str,
) -> list[dict[str, Any]]:
    k = min(max(int(top_k), 0), len(spans))
    if k == 0:
        return []
    if selector == "random":
        rng = random.Random(f"{seed}:{question_id}:{top_k}")
        shuffled = list(spans)
        rng.shuffle(shuffled)
        return shuffled[:k]
    if selector == "oracle":
        on_path = [row for row in spans if is_on_path_number(row)]
        pool = on_path if on_path else spans
        ranked = sorted(
            pool,
            key=lambda row: (
                selector_score(row, "oracle"),
                float(row.get("attention_keyness") or 0.0),
                float(row.get("surface_keyness") or 0.0),
            ),
            reverse=True,
        )
        return ranked[: min(k, len(ranked))]
    ranked = sorted(
        spans,
        key=lambda row: (selector_score(row, selector), -int(row["span_char_start"])),
        reverse=True,
    )
    return ranked[:k]


def build_steering_subset(
    records: list[dict[str, Any]],
    *,
    primary_n: int = DEFAULT_PRIMARY_N,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    grouped = group_by_question(records)
    eligible: list[dict[str, Any]] = []
    for qid, spans in grouped.items():
        if len(spans) < 4:
            continue
        has_on = any(is_on_path_number(span) for span in spans)
        has_off = any(is_off_path_number(span) for span in spans)
        if not has_on or not has_off:
            continue
        fusion_top = select_spans_for_selector(
            spans,
            selector="attention_x_resp_pos",
            top_k=1,
            seed=seed,
            question_id=qid,
        )
        attention_top = select_spans_for_selector(
            spans,
            selector="attention_only",
            top_k=1,
            seed=seed,
            question_id=qid,
        )
        fusion_success = bool(fusion_top and is_on_path_number(fusion_top[0]))
        attention_success = bool(attention_top and is_on_path_number(attention_top[0]))
        fusion_failure = bool(fusion_top and is_off_path_number(fusion_top[0]))
        if fusion_success:
            reason = "fusion_success"
        elif attention_success:
            reason = "attention_only_success"
        elif fusion_failure:
            reason = "selector_failure"
        else:
            reason = "ambiguous"
        eligible.append(
            {
                "source_question_id": qid,
                "question": spans[0]["question"],
                "num_candidate_spans": len(spans),
                "has_on_path_number": has_on,
                "has_off_path_number": has_off,
                "subset_reason": reason,
                "selected_for_primary": False,
            }
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(eligible, key=lambda row: row["source_question_id"]):
        buckets[item["subset_reason"]].append(item)
    selected: list[dict[str, Any]] = []
    quotas = [
        ("fusion_success", 30),
        ("attention_only_success", 10),
        ("selector_failure", 5),
        ("ambiguous", 5),
    ]
    for reason, quota in quotas:
        selected.extend(buckets.get(reason, [])[:quota])
    if len(selected) < primary_n:
        selected_ids = {row["source_question_id"] for row in selected}
        for item in sorted(eligible, key=lambda row: (row["subset_reason"], row["source_question_id"])):
            if item["source_question_id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["source_question_id"])
            if len(selected) >= primary_n:
                break
    selected = selected[:primary_n]
    for item in selected:
        item["selected_for_primary"] = True
    return selected


def build_smoke_configurations(*, diagnostic_query_n: int = DEFAULT_DIAGNOSTIC_QUERY_N) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def add(selector: str, top_k: int, lam: float, layers: list[int], query_scope: str, n_limit: int | None) -> None:
        cfg = {
            "selector": selector,
            "top_k": int(top_k),
            "lambda": float(lam),
            "layers": list(layers),
            "layer_config": "+".join(str(layer) for layer in layers),
            "query_scope": query_scope,
            "head_scope": HEAD_SCOPE,
            "question_limit": n_limit,
        }
        if cfg not in configs:
            configs.append(cfg)

    for selector in SELECTORS:
        for top_k in TOP_K_VALUES:
            add(selector, top_k, PRIMARY_LAMBDA, PRIMARY_LAYER_CONFIG, PRIMARY_QUERY_SCOPE, None)
    for selector in ["random", "attention_x_resp_pos", "oracle"]:
        for lam in LAMBDA_GRID:
            add(selector, 2, lam, PRIMARY_LAYER_CONFIG, PRIMARY_QUERY_SCOPE, None)
    for selector in ["attention_only", "attention_x_resp_pos", "oracle"]:
        for layers in LAYER_CONFIGS:
            add(selector, 2, PRIMARY_LAMBDA, layers, PRIMARY_QUERY_SCOPE, None)
    for scope in DIAGNOSTIC_QUERY_SCOPES:
        add("attention_x_resp_pos", 2, PRIMARY_LAMBDA, PRIMARY_LAYER_CONFIG, scope, diagnostic_query_n)
        add("oracle", 2, PRIMARY_LAMBDA, PRIMARY_LAYER_CONFIG, scope, diagnostic_query_n)
    return configs


def build_target_selector_report(
    *,
    subset: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    topk_summary: dict[str, dict[str, Any]] = {}
    for selector in SELECTORS:
        selector_rows = []
        for top_k in TOP_K_VALUES:
            hits = 0
            total = 0
            for item in subset:
                qid = item["source_question_id"]
                selected = select_spans_for_selector(
                    grouped[qid],
                    selector=selector,
                    top_k=top_k,
                    seed=DEFAULT_SEED,
                    question_id=qid,
                )
                if any(is_on_path_number(span) for span in selected):
                    hits += 1
                total += 1
            selector_rows.append(
                {
                    "top_k": top_k,
                    "question_hit_rate_eval_only": hits / total if total else 0.0,
                }
            )
        topk_summary[selector] = {"topk_eval_only": selector_rows}
    return {
        "backend": BACKEND,
        "selectors": SELECTORS,
        "top_k_values": TOP_K_VALUES,
        "num_questions": len(subset),
        "oracle_is_eval_only": True,
        "hidden_excluded": True,
        "decrease_disabled": True,
        "head_scope": HEAD_SCOPE,
        "grid_policy": "sparse_smoke_grid_v0",
        "num_forward_configurations_per_full_question": sum(1 for c in configs if c["question_limit"] is None),
        "diagnostic_query_scope_question_limit": DEFAULT_DIAGNOSTIC_QUERY_N,
        "selector_eval_only_summary": topk_summary,
    }


def build_attention_bias_config(configs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "bias_type": "additive_attention_logit_bias",
        "increase_only": True,
        "decrease_disabled": True,
        "hard_mask_disabled": True,
        "renormalize_probability_manually": False,
        "lambda_grid": LAMBDA_GRID,
        "primary_lambda": PRIMARY_LAMBDA,
        "layer_grid": LAYER_CONFIGS,
        "primary_layers": PRIMARY_LAYER_CONFIG,
        "query_scope_grid": [PRIMARY_QUERY_SCOPE] + DIAGNOSTIC_QUERY_SCOPES,
        "primary_query_scope": PRIMARY_QUERY_SCOPE,
        "head_scope": HEAD_SCOPE,
        "grid_policy": "sparse_smoke_grid_v0",
        "num_configurations": len(configs),
    }


def run_3a0_smoke_test(
    *,
    feature_matrix_path: str | Path,
    answer_position_score_matrix_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    primary_n: int = DEFAULT_PRIMARY_N,
    diagnostic_query_n: int = DEFAULT_DIAGNOSTIC_QUERY_N,
    overwrite: bool = False,
    report_every: int = 50,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    if overwrite:
        for name in [
            SUBSET_FILENAME,
            TARGET_SELECTOR_FILENAME,
            BIAS_CONFIG_FILENAME,
            FORWARD_MANIFEST_FILENAME,
            MASS_REPORT_FILENAME,
            OUTPUT_SHIFT_REPORT_FILENAME,
            GENERATION_REPORT_FILENAME,
            ORACLE_REPORT_FILENAME,
            HARM_REPORT_FILENAME,
            BASELINE_REPORT_FILENAME,
            FAILURE_CASE_FILENAME,
            SUCCESS_CASE_FILENAME,
            REVIEW_GATE_FILENAME,
            MASS_TRACE_FILENAME,
            HOOK_TRACE_FILENAME,
        ]:
            path = output_dir / name
            if path.exists():
                path.unlink()

    records = load_steering_source_records(
        feature_matrix_path=feature_matrix_path,
        answer_position_score_matrix_path=answer_position_score_matrix_path,
    )
    grouped = group_by_question(records)
    subset = build_steering_subset(records, primary_n=primary_n)
    if not subset:
        raise RuntimeError("No eligible 3A-0 steering subset could be selected")
    configs = build_smoke_configurations(diagnostic_query_n=diagnostic_query_n)
    write_jsonl(subset, output_dir / SUBSET_FILENAME)
    target_report = build_target_selector_report(subset=subset, grouped=grouped, configs=configs)
    bias_config = build_attention_bias_config(configs)
    write_json(target_report, output_dir / TARGET_SELECTOR_FILENAME)
    write_json(bias_config, output_dir / BIAS_CONFIG_FILENAME)

    context = load_local_steering_backend(model_path=model_path)
    tokenizer = context["tokenizer"]
    base_cache: dict[str, dict[str, Any]] = {}
    forward_records: list[dict[str, Any]] = []
    hook_records: list[dict[str, Any]] = []
    mass_records: list[dict[str, Any]] = []

    t0 = time.time()
    completed = 0
    for q_index, item in enumerate(subset):
        qid = item["source_question_id"]
        spans = grouped[qid]
        prompt = build_response_prompt(item["question"])
        base = run_no_steering_forward(context, prompt)
        base_summary = summarize_logits(base["logits"], tokenizer)
        base_cache[qid] = base
        query_cache: dict[str, list[int]] = {}
        for scope in [PRIMARY_QUERY_SCOPE] + DIAGNOSTIC_QUERY_SCOPES:
            query_cache[scope] = query_indices_for_scope(
                base["offsets"],
                item["question"],
                base["seq_len"],
                query_scope=scope,
            )

        for config in configs:
            if config["question_limit"] is not None and q_index >= int(config["question_limit"]):
                continue
            selected = select_spans_for_selector(
                spans,
                selector=config["selector"],
                top_k=config["top_k"],
                seed=DEFAULT_SEED,
                question_id=qid,
            )
            key_indices: list[int] = []
            selected_details = []
            for span in selected:
                span_tokens = token_indices_for_prompt_span(
                    base["offsets"],
                    item["question"],
                    int(span["span_char_start"]),
                    int(span["span_char_end"]),
                )
                key_indices.extend(span_tokens)
                selected_details.append(
                    {
                        "span_id": span["span_id"],
                        "span_text": span["span_text"],
                        "span_type": span["span_type"],
                        "span_char_start": span["span_char_start"],
                        "span_char_end": span["span_char_end"],
                        "attention_keyness": span["attention_keyness"],
                        "resp_pos_output_shift": span["resp_pos_output_shift"],
                        "resp_pos_output_shift_norm": span["resp_pos_output_shift_norm"],
                        "selector_score": selector_score(span, config["selector"])
                        if config["selector"] != "random"
                        else None,
                        "solution_path_status_eval_only": span["solution_path_status_eval_only"],
                        "key_token_indices": span_tokens,
                    }
                )
            key_indices = sorted(set(key_indices))
            query_indices = query_cache.get(config["query_scope"], [])
            before_mass = compute_attention_mass(
                base["attentions"],
                layers=config["layers"],
                query_indices=query_indices,
                key_indices=key_indices,
            )
            trace = {
                "source_question_id": qid,
                "selector": config["selector"],
                "top_k": config["top_k"],
                "lambda": config["lambda"],
                "layers": config["layers"],
                "query_scope": config["query_scope"],
                "hook_removed": False,
                "warnings": [],
            }
            if not query_indices:
                trace["warnings"].append(
                    {
                        "warning_type": "empty_query_indices",
                        "query_scope": config["query_scope"],
                    }
                )
            if not key_indices:
                trace["warnings"].append({"warning_type": "empty_key_indices"})
            steered = run_steered_forward(
                context,
                prompt,
                query_indices=query_indices,
                key_indices=key_indices,
                layers=config["layers"],
                bias_lambda=config["lambda"],
                trace=trace,
            )
            after_mass = compute_attention_mass(
                steered["attentions"],
                layers=config["layers"],
                query_indices=query_indices,
                key_indices=key_indices,
            )
            shift = compute_answer_position_output_shift(base["logits"], steered["logits"])
            steered_summary = summarize_logits(steered["logits"], tokenizer)
            record = {
                "backend": BACKEND,
                "source_question_id": qid,
                "question": item["question"],
                "selector": config["selector"],
                "top_k": config["top_k"],
                "lambda": config["lambda"],
                "layers": config["layers"],
                "layer_config": config["layer_config"],
                "query_scope": config["query_scope"],
                "head_scope": HEAD_SCOPE,
                "selected_spans": selected_details,
                "query_token_indices": query_indices,
                "selected_key_token_indices": key_indices,
                "target_attention_mass_before": before_mass["target_attention_mass"],
                "target_attention_mass_after": after_mass["target_attention_mass"],
                "target_attention_mass_delta": (
                    after_mass["target_attention_mass"] - before_mass["target_attention_mass"]
                ),
                "non_target_attention_mass_delta": (
                    after_mass["non_target_attention_mass"] - before_mass["non_target_attention_mass"]
                ),
                "per_layer_target_attention_mass_before": before_mass["per_layer_target_attention_mass"],
                "per_layer_target_attention_mass_after": after_mass["per_layer_target_attention_mass"],
                "output_shift": shift,
                "no_steering_output": base_summary,
                "steered_output": steered_summary,
                "hook": {
                    "hook_registered": trace.get("hook_registered", False),
                    "hook_triggered_layers": sorted(trace.get("hook_triggered_layers") or []),
                    "hook_removed": trace.get("hook_removed", False),
                    "warnings": trace.get("warnings") or [],
                },
                "oracle_is_eval_only": config["selector"] == "oracle",
            }
            forward_records.append(record)
            mass_records.append(record)
            hook_records.append(
                {
                    "source_question_id": qid,
                    "selector": config["selector"],
                    "top_k": config["top_k"],
                    "lambda": config["lambda"],
                    "layers": config["layers"],
                    "query_scope": config["query_scope"],
                    "hook_registered": trace.get("hook_registered", False),
                    "hook_triggered_layers": sorted(trace.get("hook_triggered_layers") or []),
                    "hook_removed": trace.get("hook_removed", False),
                    "warnings": trace.get("warnings") or [],
                }
            )
            completed += 1
            if report_every and completed % report_every == 0:
                elapsed = max(time.time() - t0, 1e-9)
                print(f"[3A-0] forwards={completed} rate={completed / elapsed:.3f}/s")

    write_jsonl(forward_records, output_dir / FORWARD_MANIFEST_FILENAME)
    write_jsonl(mass_records, output_dir / MASS_TRACE_FILENAME)
    write_jsonl(hook_records, output_dir / HOOK_TRACE_FILENAME)

    reports = build_reports(
        subset=subset,
        configs=configs,
        forward_records=forward_records,
        target_report=target_report,
        bias_config=bias_config,
    )
    write_json(reports["attention_mass_fidelity_report"], output_dir / MASS_REPORT_FILENAME)
    write_json(reports["answer_position_output_shift_report"], output_dir / OUTPUT_SHIFT_REPORT_FILENAME)
    write_json(reports["steering_generation_report"], output_dir / GENERATION_REPORT_FILENAME)
    write_json(reports["oracle_sanity_report"], output_dir / ORACLE_REPORT_FILENAME)
    write_json(reports["harm_rate_report"], output_dir / HARM_REPORT_FILENAME)
    write_json(reports["baseline_comparison_report"], output_dir / BASELINE_REPORT_FILENAME)
    write_jsonl(reports["failure_cases"], output_dir / FAILURE_CASE_FILENAME)
    write_jsonl(reports["success_cases"], output_dir / SUCCESS_CASE_FILENAME)
    review_gate = build_review_gate(
        subset=subset,
        forward_records=forward_records,
        reports=reports,
        output_dir=output_dir,
    )
    (output_dir / REVIEW_GATE_FILENAME).write_text(
        render_review_gate(review_gate, reports),
        encoding="utf-8",
    )
    return {
        "backend": BACKEND,
        "output_dir": str(output_dir),
        "num_questions": len(subset),
        "num_forward_records": len(forward_records),
        "hook_reliable": review_gate["checks"]["attention_bias_hook_reliable"]["passed"],
        "target_mass_increased": review_gate["checks"]["target_attention_mass_increased"]["passed"],
        "oracle_effective": reports["oracle_sanity_report"]["interpretation"],
        "review_gate_passed": review_gate["passed"],
        "checks_passed": review_gate["num_checks_passed"],
        "checks_total": review_gate["num_checks_total"],
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3A": True,
    }


def build_reports(
    *,
    subset: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    forward_records: list[dict[str, Any]],
    target_report: dict[str, Any],
    bias_config: dict[str, Any],
) -> dict[str, Any]:
    mass_report = build_attention_mass_fidelity_report(forward_records)
    output_report = build_output_shift_report(forward_records)
    harm_report = build_harm_rate_report(forward_records)
    oracle_report = build_oracle_sanity_report(forward_records)
    baseline_report = build_baseline_comparison_report(forward_records, mass_report, output_report, harm_report)
    generation_report = {
        "backend": BACKEND,
        "generation_eval_performed": False,
        "answer_position_proxy_performed": True,
        "reason": "3A-0 default uses answer-position next-token distribution shift only; no CoT or full generation.",
        "temperature": None,
        "max_new_tokens": None,
        "exploratory_only": True,
    }
    success_cases, failure_cases = build_case_reports(forward_records)
    return {
        "target_selector_report": target_report,
        "attention_bias_config": bias_config,
        "attention_mass_fidelity_report": mass_report,
        "answer_position_output_shift_report": output_report,
        "steering_generation_report": generation_report,
        "oracle_sanity_report": oracle_report,
        "harm_rate_report": harm_report,
        "baseline_comparison_report": baseline_report,
        "success_cases": success_cases,
        "failure_cases": failure_cases,
    }


def _group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["selector"],
        record["top_k"],
        record["lambda"],
        record["layer_config"],
        record["query_scope"],
    )


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, default: float = 0.0) -> float:
    return float(value) if _is_finite_number(value) else default


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.mean(finite)) if finite else 0.0


def _positive_rate(values: list[float]) -> float:
    return float(sum(1 for value in values if value > 0.0) / len(values)) if values else 0.0


def aggregate_by_config(forward_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in forward_records:
        grouped[_group_key(record)].append(record)
    rows = []
    for key, records in grouped.items():
        selector, top_k, lam, layer_config, query_scope = key
        deltas = [float(row["target_attention_mass_delta"]) for row in records]
        js = [float(row["output_shift"]["steer_output_js"]) for row in records]
        top1 = [float(row["output_shift"]["steer_top1_changed"]) for row in records]
        entropy = [float(row["output_shift"]["steer_entropy_delta"]) for row in records]
        margin = [float(row["output_shift"]["steer_margin_delta"]) for row in records]
        rows.append(
            {
                "selector": selector,
                "top_k": top_k,
                "lambda": lam,
                "layer_config": layer_config,
                "query_scope": query_scope,
                "num_records": len(records),
                "mean_target_attention_mass_delta": _mean(deltas),
                "target_attention_mass_positive_rate": _positive_rate(deltas),
                "mean_output_js": _mean(js),
                "top1_changed_rate": _mean(top1),
                "mean_entropy_delta": _mean(entropy),
                "mean_margin_delta": _mean(margin),
            }
        )
    return sorted(rows, key=lambda row: (row["query_scope"], row["selector"], row["top_k"], row["lambda"], row["layer_config"]))


def build_attention_mass_fidelity_report(forward_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = aggregate_by_config(forward_records)
    primary = [
        row
        for row in rows
        if row["top_k"] == 2
        and row["lambda"] == PRIMARY_LAMBDA
        and row["layer_config"] == "16+24"
        and row["query_scope"] == PRIMARY_QUERY_SCOPE
    ]
    by_selector = {row["selector"]: row for row in primary}
    s4 = by_selector.get("attention_x_resp_pos", {})
    random_row = by_selector.get("random", {})
    surface = by_selector.get("surface", {})
    attention = by_selector.get("attention_only", {})
    return {
        "backend": BACKEND,
        "metric": "target_attention_mass_delta",
        "aggregate_by_config": rows,
        "primary_config": {
            "top_k": 2,
            "lambda": PRIMARY_LAMBDA,
            "layers": PRIMARY_LAYER_CONFIG,
            "query_scope": PRIMARY_QUERY_SCOPE,
        },
        "primary_comparison": {
            "attention_x_resp_pos_mean_delta": s4.get("mean_target_attention_mass_delta"),
            "random_mean_delta": random_row.get("mean_target_attention_mass_delta"),
            "surface_mean_delta": surface.get("mean_target_attention_mass_delta"),
            "attention_only_mean_delta": attention.get("mean_target_attention_mass_delta"),
            "attention_x_resp_pos_beats_random": (
                s4.get("mean_target_attention_mass_delta", 0.0)
                > random_row.get("mean_target_attention_mass_delta", 0.0)
            ),
            "attention_x_resp_pos_beats_surface": (
                s4.get("mean_target_attention_mass_delta", 0.0)
                > surface.get("mean_target_attention_mass_delta", 0.0)
            ),
            "attention_x_resp_pos_not_worse_than_attention_only": (
                s4.get("mean_target_attention_mass_delta", -1.0)
                >= attention.get("mean_target_attention_mass_delta", 0.0) - 1e-9
            ),
        },
    }


def build_output_shift_report(forward_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = aggregate_by_config(forward_records)
    nonfinite_counter: Counter[str] = Counter()
    for record in forward_records:
        nonfinite_counter.update(record.get("output_shift", {}).get("nonfinite_shift_fields") or [])
    return {
        "backend": BACKEND,
        "metric": "answer_position_next_token_distribution_shift",
        "aggregate_by_config": rows,
        "nonfinite_shift_field_counts": dict(nonfinite_counter),
        "note": "Output shift is a next-token distribution proxy, not answer accuracy proof.",
    }


def _record_harm(record: dict[str, Any]) -> dict[str, bool]:
    shift = record["output_shift"]
    return {
        "top1_changed": float(shift["steer_top1_changed"]) > 0.0,
        "entropy_explosion": float(shift["steer_entropy_delta"]) > 1.0,
        "margin_collapse": float(shift["steer_margin_delta"]) < -0.25,
    }


def build_harm_rate_report(forward_records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in forward_records:
        grouped[_group_key(record)].append(record)
    rows = []
    for key, records in grouped.items():
        selector, top_k, lam, layer_config, query_scope = key
        flags = [_record_harm(record) for record in records]
        harm = [
            flag["top1_changed"] or flag["entropy_explosion"] or flag["margin_collapse"]
            for flag in flags
        ]
        rows.append(
            {
                "selector": selector,
                "top_k": top_k,
                "lambda": lam,
                "layer_config": layer_config,
                "query_scope": query_scope,
                "num_records": len(records),
                "harm_proxy_rate": float(sum(harm) / len(harm)) if harm else 0.0,
                "top1_changed_rate": float(sum(f["top1_changed"] for f in flags) / len(flags)) if flags else 0.0,
                "entropy_explosion_rate": float(sum(f["entropy_explosion"] for f in flags) / len(flags)) if flags else 0.0,
                "margin_collapse_rate": float(sum(f["margin_collapse"] for f in flags) / len(flags)) if flags else 0.0,
            }
        )
    return {
        "backend": BACKEND,
        "harm_proxy_definition": "top1_changed or entropy_delta>1.0 or margin_delta<-0.25",
        "aggregate_by_config": sorted(rows, key=lambda row: (row["query_scope"], row["selector"], row["top_k"], row["lambda"], row["layer_config"])),
    }


def build_oracle_sanity_report(forward_records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row
        for row in forward_records
        if row["top_k"] == 2
        and row["lambda"] == PRIMARY_LAMBDA
        and row["layer_config"] == "16+24"
        and row["query_scope"] == PRIMARY_QUERY_SCOPE
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped[row["selector"]].append(row)

    def mean_delta(selector: str) -> float:
        return _mean([float(row["target_attention_mass_delta"]) for row in grouped.get(selector, [])])

    def output_summary(selector: str) -> dict[str, float]:
        rows = grouped.get(selector, [])
        return {
            "mean_output_js": _mean([float(row["output_shift"]["steer_output_js"]) for row in rows]),
            "top1_changed_rate": _mean([float(row["output_shift"]["steer_top1_changed"]) for row in rows]),
        }

    oracle_delta = mean_delta("oracle")
    random_delta = mean_delta("random")
    predicted_delta = mean_delta("attention_x_resp_pos")
    if oracle_delta > max(random_delta, 0.0):
        interpretation = "oracle_effective"
    elif oracle_delta <= 0.0:
        interpretation = "oracle_not_effective"
    else:
        interpretation = "inconclusive"
    return {
        "backend": BACKEND,
        "oracle_attention_mass_delta": oracle_delta,
        "predicted_attention_mass_delta": predicted_delta,
        "random_attention_mass_delta": random_delta,
        "oracle_output_shift_summary": output_summary("oracle"),
        "predicted_output_shift_summary": output_summary("attention_x_resp_pos"),
        "random_output_shift_summary": output_summary("random"),
        "oracle_is_eval_only": True,
        "interpretation": interpretation,
    }


def build_baseline_comparison_report(
    forward_records: list[dict[str, Any]],
    mass_report: dict[str, Any],
    output_report: dict[str, Any],
    harm_report: dict[str, Any],
) -> dict[str, Any]:
    primary_rows = [
        row
        for row in mass_report["aggregate_by_config"]
        if row["top_k"] == 2
        and row["lambda"] == PRIMARY_LAMBDA
        and row["layer_config"] == "16+24"
        and row["query_scope"] == PRIMARY_QUERY_SCOPE
    ]
    return {
        "backend": BACKEND,
        "includes_no_steering_baseline": True,
        "selectors": ["no_steering"] + SELECTORS,
        "primary_config_summary": primary_rows,
        "comparison_focus": [
            "attention_x_resp_pos vs random",
            "attention_x_resp_pos vs surface",
            "attention_x_resp_pos vs attention_only",
            "oracle vs random",
            "oracle vs predicted",
        ],
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3A": True,
    }


def build_case_reports(forward_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_s4 = [
        row
        for row in forward_records
        if row["selector"] == "attention_x_resp_pos"
        and row["top_k"] == 2
        and row["lambda"] == PRIMARY_LAMBDA
        and row["layer_config"] == "16+24"
        and row["query_scope"] == PRIMARY_QUERY_SCOPE
    ]
    successes = sorted(primary_s4, key=lambda row: row["target_attention_mass_delta"], reverse=True)
    failures = sorted(primary_s4, key=lambda row: (row["target_attention_mass_delta"], -row["output_shift"]["steer_output_js"]))
    success_cases = [case_record(row, "clean_attention_mass_increase") for row in successes[:30]]
    failure_cases = [case_record(row, "weak_or_harmful_predicted_boost") for row in failures[:30]]
    harmful = [row for row in primary_s4 if any(_record_harm(row).values())]
    for row in harmful[:20]:
        failure_cases.append(case_record(row, "harm_proxy_triggered"))
    return success_cases, failure_cases


def case_record(record: dict[str, Any], case_type: str) -> dict[str, Any]:
    return {
        "source_question_id": record["source_question_id"],
        "question": record["question"],
        "selector": record["selector"],
        "top_k": record["top_k"],
        "lambda": record["lambda"],
        "layers": record["layers"],
        "query_scope": record["query_scope"],
        "selected_spans": record["selected_spans"],
        "attention_mass_before": record["target_attention_mass_before"],
        "attention_mass_after": record["target_attention_mass_after"],
        "attention_mass_delta": record["target_attention_mass_delta"],
        "output_shift": record["output_shift"],
        "no_steering_output": record["no_steering_output"].get("top1_token_text"),
        "steered_output": record["steered_output"].get("top1_token_text"),
        "case_type": case_type,
        "auto_interpretation": _case_interpretation(record, case_type),
    }


def _case_interpretation(record: dict[str, Any], case_type: str) -> str:
    if case_type == "clean_attention_mass_increase":
        return "predicted boost increased target attention mass under the primary config"
    if any(_record_harm(record).values()):
        return "attention mass changed but at least one harm proxy was triggered"
    return "predicted boost had weak target-mass effect or weak output movement"


def build_review_gate(
    *,
    subset: list[dict[str, Any]],
    forward_records: list[dict[str, Any]],
    reports: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    primary_rows = [
        row
        for row in reports["attention_mass_fidelity_report"]["aggregate_by_config"]
        if row["top_k"] == 2
        and row["lambda"] == PRIMARY_LAMBDA
        and row["layer_config"] == "16+24"
        and row["query_scope"] == PRIMARY_QUERY_SCOPE
    ]
    primary_s4 = next((row for row in primary_rows if row["selector"] == "attention_x_resp_pos"), {})
    hooks_ok = all(
        row.get("hook", {}).get("hook_registered")
        and row.get("hook", {}).get("hook_removed")
        and set(row.get("hook", {}).get("hook_triggered_layers") or []) == set(row["layers"])
        for row in forward_records
    )
    checks = {
        "subset_manifest_generated": {
            "passed": bool(subset) and (output_dir / SUBSET_FILENAME).exists(),
            "detail": f"num_questions={len(subset)}",
        },
        "target_selectors_generated": {
            "passed": (output_dir / TARGET_SELECTOR_FILENAME).exists(),
            "detail": "random/surface/attention/attention_x_resp_pos/oracle",
        },
        "attention_bias_hook_reliable": {
            "passed": hooks_ok,
            "detail": "registered, triggered requested layers, and removed for every steered forward",
        },
        "no_steering_baseline_preserved": {
            "passed": True,
            "detail": "one no-steering forward was cached per question before steered variants",
        },
        "positive_boost_only": {
            "passed": True,
            "detail": "all configured lambdas are positive additive logit biases",
        },
        "no_decrease_no_hard_mask": {
            "passed": True,
            "detail": "no suppressor, hard mask, or probability replacement implemented",
        },
        "target_attention_mass_increased": {
            "passed": float(primary_s4.get("mean_target_attention_mass_delta") or 0.0) > 0.0,
            "detail": f"primary_s4_delta={primary_s4.get('mean_target_attention_mass_delta')}",
        },
        "oracle_sanity_report_generated": {
            "passed": (output_dir / ORACLE_REPORT_FILENAME).exists()
            and reports["oracle_sanity_report"]["interpretation"] != "oracle_not_effective",
            "detail": reports["oracle_sanity_report"]["interpretation"],
        },
        "harm_rate_report_generated": {
            "passed": (output_dir / HARM_REPORT_FILENAME).exists(),
            "detail": "harm proxy report generated",
        },
        "baseline_comparison_report_generated": {
            "passed": (output_dir / BASELINE_REPORT_FILENAME).exists(),
            "detail": "baseline comparison generated",
        },
        "failure_success_cases_generated": {
            "passed": bool(reports["failure_cases"]) and bool(reports["success_cases"]),
            "detail": f"failure={len(reports['failure_cases'])}, success={len(reports['success_cases'])}",
        },
        "ready_for_2000_rerun_false": {
            "passed": True,
            "detail": "forced false by 3A-0 boundary",
        },
        "do_not_enter_full_sprint_3A_true": {
            "passed": True,
            "detail": "forced true by 3A-0 boundary",
        },
    }
    num_passed = sum(1 for check in checks.values() if check["passed"])
    return {
        "backend": BACKEND,
        "passed": num_passed == len(checks),
        "checks": checks,
        "num_checks_passed": num_passed,
        "num_checks_total": len(checks),
        "ready_for_2000_rerun": False,
        "do_not_enter_full_sprint_3A": True,
        "hallucination_reduction_proven": False,
        "answer_accuracy_improvement_proven": False,
    }


def render_review_gate(review_gate: dict[str, Any], reports: dict[str, Any]) -> str:
    mass = reports["attention_mass_fidelity_report"]
    oracle = reports["oracle_sanity_report"]
    primary = mass["primary_comparison"]
    checks = "\n".join(
        f"- {name}: {check['passed']} ({check['detail']})"
        for name, check in review_gate["checks"].items()
    )
    return f"""# Sprint 3A-0 Attention Bias Steering Smoke Test Review Gate

Verdict:
- passed: {review_gate['passed']} ({review_gate['num_checks_passed']}/{review_gate['num_checks_total']})
- ready_for_2000_rerun: false
- do_not_enter_full_sprint_3A: true
- hallucination_reduction_proven: false
- answer_accuracy_improvement_proven: false

Boundary:
- increase-only: true
- no decrease: true
- no hard mask: true
- no training: true
- no 2000: true
- no full 3A: true

Intervention:
- bias type: additive attention logit bias
- lambda grid: {LAMBDA_GRID}
- primary lambda: {PRIMARY_LAMBDA}
- layers: {LAYER_CONFIGS}
- primary query scope: {PRIMARY_QUERY_SCOPE}
- head scope: {HEAD_SCOPE}

Steering fidelity:
- attention_x_resp_pos mean target mass delta: {primary.get('attention_x_resp_pos_mean_delta')}
- random mean target mass delta: {primary.get('random_mean_delta')}
- surface mean target mass delta: {primary.get('surface_mean_delta')}
- attention-only mean target mass delta: {primary.get('attention_only_mean_delta')}

Oracle sanity:
- interpretation: {oracle['interpretation']}
- oracle_attention_mass_delta: {oracle['oracle_attention_mass_delta']}
- predicted_attention_mass_delta: {oracle['predicted_attention_mass_delta']}
- random_attention_mass_delta: {oracle['random_attention_mass_delta']}

Output shift:
- See answer_position_output_shift_report.json.

Harm rate:
- See harm_rate_report.json.

Generation eval if available:
- generation_eval_performed: false
- answer-position next-token proxy was used instead.

Failure/success cases:
- See failure_case_report.jsonl and success_case_report.jsonl.

Checks:
{checks}

Required final questions:
- attention bias changed target span attention mass: {review_gate['checks']['target_attention_mass_increased']['passed']}
- hook reliable/removable/no baseline contamination: {review_gate['checks']['attention_bias_hook_reliable']['passed']}
- attention_x_resp_pos better than random: {primary.get('attention_x_resp_pos_beats_random')}
- attention_x_resp_pos better than surface: {primary.get('attention_x_resp_pos_beats_surface')}
- attention_x_resp_pos not worse than attention-only: {primary.get('attention_x_resp_pos_not_worse_than_attention_only')}
- oracle boost effective: {oracle['interpretation']}
- small lambda safe: see harm_rate_report.json
- generation eval performed: false
- recommend 3A-1: {'true' if review_gate['passed'] else 'false'}
- recommend 2000: false
- recommend full Sprint 3A: false

Recommendation:
- 3A-0 can only justify a controlled 3A-1 smoke/eval if the hook and fidelity checks pass.
- Do not claim answer accuracy improvement or hallucination reduction from this run.
"""
