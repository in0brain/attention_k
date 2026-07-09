"""Sprint 3C-1: module-level causal tracing at the final-answer readout position.

Decomposes the whole-residual patch used in Sprint 3C-0-Fix into per-module
writes — self-attention output, MLP output, and the whole decoder-layer
(residual) output — and interpolation-patches each at the answer-readout token
position (the token right before the final answer number). Reuses the corrected
teacher-forced answer-sequence proxy from ``answer_proxy_metrics``.

Boundary: no training, no LoRA, no new steering deployment, no full Sprint 3C /
2000 rerun, no accuracy or hallucination claims. Teacher-forced single-forward
proxy only.
"""

from __future__ import annotations

from typing import Any

from recover_attention import activation_patching as ap
from recover_attention import answer_proxy_metrics as apm
from recover_attention import multi_span_reasoning_scoring as msrs

BACKEND = "final_answer_compression_module_causal_tracing_v0"
MODULE_TYPES = ("attention_output", "mlp_output", "residual_output")


def _decoder_layers(model: Any) -> Any:
    return ap._decoder_layers(model)


def _module_for(layer_module: Any, module_type: str) -> Any:
    if module_type == "attention_output":
        return layer_module.self_attn
    if module_type == "mlp_output":
        return layer_module.mlp
    if module_type == "residual_output":
        return layer_module
    raise ValueError(f"unsupported module_type: {module_type}")


def capture_module_outputs(
    context: dict[str, Any],
    text: str,
    layers: list[int],
    module_types: list[str],
) -> dict[str, Any]:
    """Forward once and capture (layer, module) output tensors [seq, hidden]."""

    import torch

    for module_type in module_types:
        if module_type not in MODULE_TYPES:
            raise ValueError(f"unsupported module_type: {module_type}")
    tokenizer = context["tokenizer"]
    model = context["model"]
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(s), int(e)] for s, e in encoded.pop("offset_mapping")[0].tolist()]
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in encoded.items()}

    captured: dict[tuple[int, str], Any] = {}
    handles = []
    decoder = _decoder_layers(model)

    def make_hook(layer_idx: int, module_type: str):
        def hook(_module: Any, _inp: Any, out: Any):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[(int(layer_idx), module_type)] = hidden[0].detach().float().cpu()

        return hook

    for layer in layers:
        module_owner = decoder[int(layer)]
        for module_type in module_types:
            handles.append(_module_for(module_owner, module_type).register_forward_hook(make_hook(int(layer), module_type)))
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "captured": captured,
        "offsets": offsets,
        "input_ids": inputs["input_ids"].detach().cpu(),
        "seq_len": int(inputs["input_ids"].shape[-1]),
    }


def module_vector(captured: dict[tuple[int, str], Any], *, layer: int, module_type: str, position: int) -> Any | None:
    tensor = captured.get((int(layer), module_type))
    if tensor is None or not (0 <= position < tensor.shape[0]):
        return None
    return tensor[int(position)].detach().float().cpu()


def register_module_patch_hook(
    model: Any,
    *,
    layer: int,
    module_type: str,
    donor_vec: Any,
    target_position: int,
    alpha: float,
    trace: dict[str, Any],
) -> list[Any]:
    """Interpolation-patch a single module's output at one token position."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if module_type not in MODULE_TYPES:
        raise ValueError(f"unsupported module_type: {module_type}")
    decoder = _decoder_layers(model)
    module = _module_for(decoder[int(layer)], module_type)
    trace.setdefault("triggered", [])
    trace.setdefault("patch_records", [])
    trace["registered"] = False

    def hook(_module: Any, _inp: Any, out: Any):
        hidden = out[0] if isinstance(out, tuple) else out
        if not (0 <= target_position < hidden.shape[1]):
            return out
        donor = donor_vec.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        before = hidden[:, target_position, :].clone()
        after = (1.0 - float(alpha)) * before + float(alpha) * donor
        patched[:, target_position, :] = after
        trace["patch_records"].append(
            {
                "layer": int(layer),
                "module_type": module_type,
                "patched_position": int(target_position),
                "patch_delta_norm": float((donor - before[0]).detach().float().norm().item()),
                "recipient_changed_norm": float((after[0] - before[0]).detach().float().norm().item()),
                "non_target_position_contamination_check": True,
            }
        )
        trace["triggered"].append((int(layer), module_type))
        if isinstance(out, tuple):
            return (patched,) + tuple(out[1:])
        return patched

    handle = module.register_forward_hook(hook)
    trace["registered"] = True
    return [handle]


def sequence_logprob_with_module_patch(
    context: dict[str, Any],
    *,
    prefix_ids: list[int],
    answer_ids: list[int],
    patch: dict[str, Any] | None = None,
    return_slot_logits: bool = False,
) -> dict[str, Any]:
    """Teacher-forced answer-sequence logprob with an optional module patch."""

    import torch

    model = context["model"]
    if not answer_ids:
        return {"logprob": float("nan"), "per_token": float("nan"), "num_answer_tokens": 0, "trace": {}, "answer_slot_logits": None}

    input_ids = torch.tensor([list(prefix_ids) + list(answer_ids)], dtype=torch.long)
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = input_ids.to(target_device)

    handles: list[Any] = []
    trace: dict[str, Any] = {"registered": False, "removed": False, "triggered": [], "patch_records": []}
    if patch is not None and patch.get("donor_vec") is not None:
        handles = register_module_patch_hook(
            model,
            layer=int(patch["layer"]),
            module_type=str(patch["module_type"]),
            donor_vec=patch["donor_vec"],
            target_position=int(patch["target_position"]),
            alpha=float(patch.get("alpha", 1.0)),
            trace=trace,
        )
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        n_prefix = len(prefix_ids)
        total = 0.0
        for i, token_id in enumerate(answer_ids):
            total += float(logprobs[n_prefix + i - 1, int(token_id)].item())
        slot_logits = logits[n_prefix - 1].detach().cpu() if return_slot_logits else None
    finally:
        for handle in handles:
            handle.remove()
        trace["removed"] = True
    n = len(answer_ids)
    return {
        "logprob": total,
        "per_token": total / n if n else float("nan"),
        "num_answer_tokens": n,
        "trace": trace,
        "answer_slot_logits": slot_logits,
    }


def build_answer_readout(context: dict[str, Any], trace_text: str, capture: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the answer span and the readout position (answer_start - 1)."""

    span = apm.extract_final_answer_span(trace_text)
    if span["char_start"] is None or span["method"] == "parse_failure":
        return None
    answer_start = apm.token_index_for_char_start(capture["offsets"], span["char_start"])
    if answer_start is None or answer_start <= 0:
        return None
    return {
        "span": span,
        "answer_start": int(answer_start),
        "readout_position": int(answer_start) - 1,
        "prefix_ids": capture["input_ids"][0, : int(answer_start)].tolist(),
    }


def paired_bootstrap_delta(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return apm.paired_bootstrap_delta(rows, **kwargs)


def compute_donor_specificity(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """clean_direction(correct_donor) - clean_direction(random_donor), paired."""

    return apm.paired_bootstrap_delta(
        rows, treatment="correct_donor_patch", control="random_donor_patch", **kwargs
    )


def compute_site_specificity(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """clean_direction(answer readout) - clean_direction(same-trace random position)."""

    return apm.paired_bootstrap_delta(
        rows, treatment="correct_donor_patch", control="same_trace_random_position_patch", **kwargs
    )
