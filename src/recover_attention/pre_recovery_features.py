"""Sprint 2H-C: pre-recovery hidden-state feature enrichment.

2H-B's only leakage-free feature family was ``*_original_masked_cosine_*`` (angle-only
summaries; span and mask_position pooling were identical). This module re-pools the raw
2G hidden-state tensors (full ``[layers, tokens, hidden]`` cache) into richer features
that are still available BEFORE recovery is run:

  A. layer-wise original->masked delta magnitude (L2 / relative norm / cosine / slope)
  B. within-channel span saliency (span-to-question, span-to-number-context)
  C. cross-layer stability (per-layer variance, early->late cosine, layer shift norm)

Hard rule: only the ``original`` and ``masked`` channels are read. The ``recovered``
channel, recovery outputs, gold solution path, and any label component are never used
as inputs. Feature names deliberately avoid the banned substrings
(recovered / solution_path / drift / bucket / risk_strength / gold) so the gate-eligibility
leakage test passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from recover_attention.representation_features import (
    locate_span_token_indices,
    pool_question,
    pool_span,
    sanitize_tensor,
)

ARABIC_NUMBER_PATTERN = re.compile(
    r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?"
)

# Substrings forbidden in any gate-eligible feature name (task 4).
BANNED_FEATURE_SUBSTRINGS = ("recovered", "solution_path", "drift", "bucket", "risk_strength", "gold")

EPS = 1e-8


def load_tensor_cpu(path: str | Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# small vector / layer-curve helpers (operate on torch tensors [L, H])
# --------------------------------------------------------------------------- #
def _cos(a: Any, b: Any) -> float:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na <= EPS or nb <= EPS:
        return 0.0
    return max(-1.0, min(1.0, float((a * b).sum().item()) / (na * nb)))


def _layer_cosine(first: Any, second: Any) -> list[float]:
    return [_cos(first[l], second[l]) for l in range(first.shape[0])]


def _layer_l2(first: Any, second: Any) -> list[float]:
    return [float((first[l] - second[l]).norm().item()) for l in range(first.shape[0])]


def _layer_relnorm(first: Any, second: Any) -> list[float]:
    out = []
    for l in range(first.shape[0]):
        denom = float(first[l].norm().item())
        out.append(float((second[l] - first[l]).norm().item()) / denom if denom > EPS else 0.0)
    return out


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (v - my) for x, v in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _range(values: list[float]) -> float:
    return (max(values) - min(values)) if values else 0.0


def _var(values: list[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i]) if values else -1


def _layer_variance(vectors: Any) -> float:
    """Mean over layers of ||v_l - v_mean||^2, normalized by ||v_mean||^2."""
    mean_vec = vectors.mean(dim=0)
    denom = float((mean_vec * mean_vec).sum().item())
    if denom <= EPS:
        return 0.0
    total = 0.0
    for l in range(vectors.shape[0]):
        diff = vectors[l] - mean_vec
        total += float((diff * diff).sum().item())
    return (total / vectors.shape[0]) / denom


def _layer_shift_norm(vectors: Any) -> float:
    """Relative L2 shift between first and last layer (renamed from 'drift' to avoid
    the banned feature-name substring)."""
    first = vectors[0]
    last = vectors[-1]
    denom = float(first.norm().item())
    return float((last - first).norm().item()) / denom if denom > EPS else 0.0


# --------------------------------------------------------------------------- #
# number-context token localization (original channel only)
# --------------------------------------------------------------------------- #
def _token_indices_for_char_ranges(token_char_ranges, char_ranges, exclude: set[int]) -> list[int]:
    indices = []
    for token_index, token_range in enumerate(token_char_ranges or []):
        if token_index in exclude:
            continue
        if not token_range or len(token_range) != 2:
            continue
        ts, te = token_range
        for cs, ce in char_ranges:
            if ts < ce and cs < te:
                indices.append(token_index)
                break
    return indices


def _number_context_indices(record: dict[str, Any], span_indices: set[int]) -> list[int]:
    text = record.get("input_text", "") or ""
    char_ranges = [[m.start(), m.end()] for m in ARABIC_NUMBER_PATTERN.finditer(text)]
    if not char_ranges:
        return []
    return _token_indices_for_char_ranges(record.get("token_char_ranges"), char_ranges, span_indices)


# --------------------------------------------------------------------------- #
# main extraction
# --------------------------------------------------------------------------- #
def extract_pre_recovery_features(
    original_record: dict[str, Any],
    masked_record: dict[str, Any],
    *,
    tensor_loader: Callable[[str | Path], Any] = load_tensor_cpu,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Compute enriched pre-recovery features for one masked_id.

    Returns {"features": {name: float}, "missing": {flag: bool}, "warnings": [...]}.
    Only original + masked channels are read.
    """
    warnings: list[str] = []

    def _resolve(path: str) -> Path:
        p = Path(path)
        if not p.is_absolute() and project_root is not None:
            p = project_root / p
        return p

    orig_t = sanitize_tensor(tensor_loader(_resolve(original_record["hidden_state_path"])))[0]
    mask_t = sanitize_tensor(tensor_loader(_resolve(masked_record["hidden_state_path"])))[0]

    orig_span_idx, ow = locate_span_token_indices(original_record)
    mask_span_idx, mw = locate_span_token_indices(masked_record)
    warnings.extend(ow + mw)

    orig_q = pool_question(orig_t)                       # [L, H]
    mask_q = pool_question(mask_t)
    orig_s = pool_span(orig_t, orig_span_idx)            # [L, H] or None
    mask_s = pool_span(mask_t, mask_span_idx)

    span_available = orig_s is not None and mask_s is not None
    features: dict[str, float] = {}
    missing: dict[str, bool] = {}

    # ---- A. layer-wise original->masked delta (question pooling always available) ----
    def _delta_block(prefix: str, first: Any, second: Any) -> None:
        cos = _layer_cosine(first, second)
        l2 = _layer_l2(first, second)
        rel = _layer_relnorm(first, second)
        n = first.shape[0]
        for l in range(n):
            features[f"pre_delta_{prefix}_cosine_layer_{l}"] = cos[l]
            features[f"pre_delta_{prefix}_l2_layer_{l}"] = l2[l]
            features[f"pre_delta_{prefix}_relnorm_layer_{l}"] = rel[l]
        features[f"pre_delta_{prefix}_relnorm_slope"] = _slope(rel)
        features[f"pre_delta_{prefix}_relnorm_range"] = _range(rel)
        features[f"pre_delta_{prefix}_relnorm_var"] = _var(rel)
        features[f"pre_delta_{prefix}_relnorm_mean"] = sum(rel) / n
        features[f"pre_delta_{prefix}_relnorm_max"] = max(rel)
        features[f"pre_delta_{prefix}_l2_mean"] = sum(l2) / n
        features[f"pre_delta_{prefix}_l2_slope"] = _slope(l2)
        features[f"pre_delta_{prefix}_max_layer"] = float(_argmax(rel))
        features[f"pre_delta_{prefix}_early_late_relnorm_diff"] = rel[-1] - rel[0]

    _delta_block("question", orig_q, mask_q)
    if span_available:
        _delta_block("span", orig_s, mask_s)
    else:
        missing["span_delta"] = True
        warnings.append("span pooling unavailable; span delta features skipped")

    # ---- B. within-channel span saliency ----
    if span_available:
        s2q_orig = _layer_cosine(orig_s, orig_q)
        s2q_mask = _layer_cosine(mask_s, mask_q)
        n = orig_s.shape[0]
        for l in range(n):
            features[f"pre_saliency_span_to_question_orig_layer_{l}"] = s2q_orig[l]
            features[f"pre_saliency_span_to_question_masked_layer_{l}"] = s2q_mask[l]
        features["pre_saliency_span_to_question_orig_mean"] = sum(s2q_orig) / n
        features["pre_saliency_span_to_question_masked_mean"] = sum(s2q_mask) / n
        features["pre_saliency_span_to_question_orig_minus_masked_mean"] = (
            sum(s2q_orig) / n - sum(s2q_mask) / n
        )

        # number context (original channel), excluding the span tokens themselves
        numctx_idx = _number_context_indices(original_record, set(orig_span_idx or []))
        if numctx_idx:
            numctx_vec = orig_t[:, numctx_idx, :].mean(dim=1)
            s2n = _layer_cosine(orig_s, numctx_vec)
            for l in range(orig_s.shape[0]):
                features[f"pre_saliency_span_to_numctx_orig_layer_{l}"] = s2n[l]
            features["pre_saliency_span_to_numctx_orig_mean"] = sum(s2n) / len(s2n)
            missing["numctx"] = False
        else:
            missing["numctx"] = True
    else:
        missing["saliency"] = True

    # ---- C. cross-layer stability (per channel) ----
    features["pre_stability_question_layervar_orig"] = _layer_variance(orig_q)
    features["pre_stability_question_layervar_masked"] = _layer_variance(mask_q)
    features["pre_stability_question_early_late_cos_orig"] = _cos(orig_q[0], orig_q[-1])
    features["pre_stability_question_early_late_cos_masked"] = _cos(mask_q[0], mask_q[-1])
    features["pre_stability_question_layershift_orig"] = _layer_shift_norm(orig_q)
    features["pre_stability_question_layershift_masked"] = _layer_shift_norm(mask_q)
    if span_available:
        features["pre_stability_span_layervar_orig"] = _layer_variance(orig_s)
        features["pre_stability_span_layervar_masked"] = _layer_variance(mask_s)
        features["pre_stability_span_early_late_cos_orig"] = _cos(orig_s[0], orig_s[-1])
        features["pre_stability_span_early_late_cos_masked"] = _cos(mask_s[0], mask_s[-1])
        features["pre_stability_span_layershift_orig"] = _layer_shift_norm(orig_s)
        features["pre_stability_span_layershift_masked"] = _layer_shift_norm(mask_s)

    assert_no_banned_feature_names(list(features.keys()))
    return {"features": features, "missing": missing, "warnings": warnings, "span_available": span_available}


def assert_no_banned_feature_names(feature_names: list[str]) -> None:
    """Fail loudly if any feature name contains a banned (label-leaking) substring."""
    bad = [n for n in feature_names for token in BANNED_FEATURE_SUBSTRINGS if token in n]
    if bad:
        raise AssertionError(
            f"pre-recovery feature names must not contain {BANNED_FEATURE_SUBSTRINGS}; offenders: {bad[:5]}"
        )
