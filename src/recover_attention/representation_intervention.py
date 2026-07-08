"""Sprint 3B-0: representation-level (residual) intervention hook.

3A-1 showed additive attention-logit bias does not convert "correct span selected"
into "answer more correct" (oracle no better than random). This module tests a
different channel: inject the selected span's residual-stream *deviation* into the
answer-position residual at chosen layers, and check whether the ORACLE (on-path)
span moves the answer toward the gold token more than a RANDOM span.

Primary intervention (Method A, conservative form):
    inj_L = beta * ||h_L[answer_pos]|| * unit( mean(h_L[span_tokens]) - mean(h_L[all]) )
    h_L[answer_pos] += inj_L                         (added at layer L output)

The `||h_L[answer_pos]||` scaling makes beta a fraction of the residual magnitude, so
the beta sweep reaches a measurable regime (a unit-norm injection is a no-op against a
~100-norm residual). This scaling is documented in the config report.

Boundary: no training, no LoRA, increase/inject only (beta >= 0), gold answer used only
for eval-only metrics, oracle used only as an eval-only diagnostic selector.
"""

from __future__ import annotations

from typing import Any

_EPS = 1e-8


def base_forward_with_hidden(context: dict[str, Any], prompt: str, target_layers: list[int]) -> dict[str, Any]:
    """Run a no-intervention forward, returning last-token logits + per-target-layer
    hidden states (output of layer L = hidden_states[L+1]) on cpu."""
    import torch

    from recover_attention import multi_span_reasoning_scoring as msrs

    tokenizer = context["tokenizer"]
    model = context["model"]
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = [[int(s), int(e)] for s, e in enc.pop("offset_mapping")[0].tolist()]
    device = msrs.infer_model_input_device(model, "auto", torch)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple length num_layers+1
    base_hidden = {}
    for layer in target_layers:
        idx = layer + 1
        if idx < len(hs):
            base_hidden[layer] = hs[idx][0].detach().float().cpu()
    return {
        "logits": out.logits[0, -1, :].detach().float().cpu(),
        "base_hidden": base_hidden,
        "offsets": offsets,
        "seq_len": int(inputs["input_ids"].shape[-1]),
    }


def compute_injection_vectors(
    base_hidden: dict[int, Any],
    span_token_indices: list[int],
    target_layers: list[int],
    answer_pos: int,
    beta: float,
) -> dict[int, Any]:
    """Return {layer: injection tensor} = beta * ||ans_residual|| * unit(span_dev).

    span_dev = mean(h_L[span_tokens]) - mean(h_L[all tokens]). Returns None for a layer
    when the span is empty or its deviation is degenerate.
    """
    if beta < 0:
        raise ValueError("3B-0 residual injection only supports beta >= 0 (inject/increase only)")
    inj: dict[int, Any] = {}
    for layer in target_layers:
        h = base_hidden.get(layer)
        if h is None or not span_token_indices:
            inj[layer] = None
            continue
        idx = [i for i in span_token_indices if 0 <= i < h.shape[0]]
        if not idx:
            inj[layer] = None
            continue
        span_repr = h[idx].mean(dim=0)
        mean_repr = h.mean(dim=0)
        vec = span_repr - mean_repr
        vec_norm = float(vec.norm().item())
        if vec_norm < _EPS or beta == 0.0:
            inj[layer] = None
            continue
        ans_norm = float(h[answer_pos].norm().item()) if 0 <= answer_pos < h.shape[0] else 0.0
        inj[layer] = (beta * ans_norm / vec_norm) * vec
    return inj


def register_residual_injection_hooks(
    model: Any,
    injections: dict[int, Any],
    answer_pos: int,
    trace: dict[str, Any],
) -> list[Any]:
    """Register forward hooks on decoder layers that add the injection to the answer-
    position residual at each target layer's output. Returns removable handles."""
    layers = _decoder_layers(model)
    trace.setdefault("triggered_layers", [])
    trace["registered"] = False
    handles = []
    for layer_idx, vec in injections.items():
        if vec is None or layer_idx >= len(layers):
            continue
        module = layers[layer_idx]

        def make_hook(v: Any, lidx: int):
            def hook(_module: Any, _inp: Any, out: Any):
                hidden = out[0] if isinstance(out, tuple) else out
                add = v.to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden.clone()
                if 0 <= answer_pos < hidden.shape[1]:
                    hidden[:, answer_pos, :] = hidden[:, answer_pos, :] + add
                    if lidx not in trace["triggered_layers"]:
                        trace["triggered_layers"].append(lidx)
                if isinstance(out, tuple):
                    return (hidden,) + tuple(out[1:])
                return hidden
            return hook

        handles.append(module.register_forward_hook(make_hook(vec, layer_idx)))
    trace["registered"] = len(handles) > 0
    return handles


def remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()


def steered_forward_with_injection(
    context: dict[str, Any],
    prompt: str,
    injections: dict[int, Any],
    answer_pos: int,
    trace: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from recover_attention import multi_span_reasoning_scoring as msrs

    tokenizer = context["tokenizer"]
    model = context["model"]
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    enc.pop("offset_mapping")
    device = msrs.infer_model_input_device(model, "auto", torch)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    handles = register_residual_injection_hooks(model, injections, answer_pos, trace)
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
    finally:
        remove_hooks(handles)
        trace["removed"] = True
    return {"logits": out.logits[0, -1, :].detach().float().cpu()}


def injection_total_norm(injections: dict[int, Any]) -> float:
    total = 0.0
    for vec in injections.values():
        if vec is not None:
            total += float(vec.norm().item())
    return total


def _decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate decoder layers (model.model.layers) for residual hooks")
    return layers
