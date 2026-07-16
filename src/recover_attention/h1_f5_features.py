"""Pure helpers for Sprint 4D-2 H1-F5 output-layer features.

The functions here deliberately operate on already-computed token logprobs,
offsets, ranks, and mention strings.  They do not load models, inspect gold
source identifiers, or capture intermediate activations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable


PRIMARY_FAMILIES = {"attack", "cwe"}
CONFIDENCE_PATTERNS = {
    "high": re.compile(r"\b(certainly|definitely|confident|clearly|exactly|known as)\b", re.I),
    "medium": re.compile(r"\b(likely|probably|typically|commonly|appears to|seems to)\b", re.I),
    "low": re.compile(r"\b(possibly|perhaps|may|might|could be|i think|not sure)\b", re.I),
}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if finite_float(v) is not None]
    return sum(clean) / len(clean) if clean else None


def token_indices_for_char_span(
    offsets: list[tuple[int, int]] | list[list[int]],
    start: int,
    end: int,
) -> list[int]:
    """Map a character span to token indices using overlap with offsets.

    Offsets are expected in the same coordinate system as ``start``/``end``.
    Zero-length special-token offsets are ignored.  Boundary-touching tokens do
    not count as overlap.
    """

    if end <= start:
        raise ValueError(f"invalid char span: start={start}, end={end}")
    indices: list[int] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError(f"offset at index {index} must have two values")
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            indices.append(index)
    return indices


def mention_logprob_features(
    token_logprobs: dict[int, float],
    token_entropies: dict[int, float],
    first_token_ranks: dict[int, int],
    token_indices: list[int],
) -> dict[str, float | int | None]:
    """Summarize teacher-forced logprobs over one identifier span."""

    lps = [token_logprobs.get(int(index)) for index in token_indices]
    entropies = [token_entropies.get(int(index)) for index in token_indices]
    clean_lps = [float(value) for value in lps if finite_float(value) is not None]
    return {
        "f5_id_token_count": len(token_indices),
        "f5_id_logprob_mean": mean(clean_lps),
        "f5_id_logprob_min": min(clean_lps) if clean_lps else None,
        "f5_first_id_token_rank": first_token_ranks.get(int(token_indices[0])) if token_indices else None,
        "f5_id_token_entropy_mean": mean(entropies),
    }


def sequence_logprob_features(
    completion_token_logprobs: list[float],
    *,
    id_token_count: int,
) -> dict[str, float | None]:
    """Completion-level F5 features inherited by each mention row."""

    clean = [float(value) for value in completion_token_logprobs if finite_float(value) is not None]
    lengthnorm = mean(clean)
    perplexity = math.exp(-lengthnorm) if lengthnorm is not None and lengthnorm < 700 else None
    return {
        "f5_completion_token_count": len(clean),
        "f5_completion_perplexity": perplexity,
        "f5_lengthnorm_logprob": lengthnorm,
        "f5_id_token_ratio": id_token_count / len(clean) if clean else None,
    }


def exact_consistency(ids: Iterable[str]) -> dict[str, float | int | str | None]:
    """Exact-string self-consistency over sampled identifier sets."""

    clean = [str(item) for item in ids if isinstance(item, str) and item.strip()]
    if not clean:
        return {
            "num_values": 0,
            "mode_value": None,
            "mode_count": 0,
            "f5_self_consistency_exact": None,
        }
    counts = Counter(clean)
    mode_value, mode_count = counts.most_common(1)[0]
    return {
        "num_values": len(clean),
        "mode_value": mode_value,
        "mode_count": int(mode_count),
        "f5_self_consistency_exact": mode_count / len(clean),
    }


def id_agreement_rate(target_id: str, ids: Iterable[str]) -> float | None:
    clean = [str(item) for item in ids if isinstance(item, str) and item.strip()]
    if not clean:
        return None
    return sum(item == target_id for item in clean) / len(clean)


def extract_verbalized_confidence(text: str) -> dict[str, int | str | None]:
    """Small diagnostic regex extractor, intentionally label-agnostic."""

    matches: list[tuple[str, str]] = []
    for level, pattern in CONFIDENCE_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            matches.append((level, match.group(0)))
    first = matches[0] if matches else None
    return {
        "f5_confidence_phrase": first[1] if first else None,
        "f5_confidence_high": int(any(level == "high" for level, _ in matches)),
        "f5_confidence_medium": int(any(level == "medium" for level, _ in matches)),
        "f5_confidence_low": int(any(level == "low" for level, _ in matches)),
    }


def included_in_primary_detection(row: dict[str, Any]) -> bool:
    """Primary 4D-2 label set: non-echo ATT&CK/CWE mentions only."""

    return (
        row.get("mention_family") in PRIMARY_FAMILIES
        and row.get("label") in {"grounded", "fabricated"}
    )


def looks_like_structured_identifier_list(text: str) -> bool:
    """Detect list-shaped outputs that can trip substring-cycle degeneration.

    This is used only to separate true backend degeneration from a known false
    positive pattern where repeated identifier-list formatting triggers the
    shared degeneration heuristic.
    """

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    list_like = 0
    for line in lines:
        if re.match(r"^(?:[-*]|\d+[.)])\s+", line):
            list_like += 1
        elif re.match(r"^(?:CVE-\d{4}-\d{4,}|CWE-\d{1,5}|T\d{4}(?:\.\d{3})?)\b", line, re.I):
            list_like += 1
    return list_like >= 4 and list_like / len(lines) >= 0.5


def safe_numeric_feature_row(row: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    """Return finite numeric feature values, replacing missing values with 0."""

    out: dict[str, float] = {}
    for name in feature_names:
        value = finite_float(row.get(name))
        out[name] = float(value) if value is not None else 0.0
    return out
