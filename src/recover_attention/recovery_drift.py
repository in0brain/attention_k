"""Sprint 2H-B task 3: drift classification of recovery samples.

For each recovery sample we recover the model's filler for the masked slot and
classify how it drifted from the original span. Multi-sample aggregation uses
worst-case evidence (one wrong recovery is enough to prove fragility).

Drift labels:
    exact_recovery, generic_recovery, wrong_numeric_recovery,
    unit_drift, object_drift, direction_drift, condition_drift,
    unrecoverable, ambiguous
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from recover_attention.solution_path_numbers import normalize_number_text


EXACT = "exact_recovery"
GENERIC = "generic_recovery"
WRONG_NUMERIC = "wrong_numeric_recovery"
UNIT_DRIFT = "unit_drift"
OBJECT_DRIFT = "object_drift"
DIRECTION_DRIFT = "direction_drift"
CONDITION_DRIFT = "condition_drift"
UNRECOVERABLE = "unrecoverable"
AMBIGUOUS = "ambiguous"

DRIFT_LABELS = [
    EXACT, GENERIC, WRONG_NUMERIC, UNIT_DRIFT,
    OBJECT_DRIFT, DIRECTION_DRIFT, CONDITION_DRIFT, UNRECOVERABLE, AMBIGUOUS,
]

# Drift labels that count as "hard" fragility evidence (force bucket 3).
HARD_DRIFT_LABELS = {WRONG_NUMERIC, DIRECTION_DRIFT, UNIT_DRIFT, OBJECT_DRIFT, CONDITION_DRIFT}

GENERIC_QUANTITY_WORDS = {
    "some", "several", "a few", "few", "many", "a number", "a number of",
    "a couple", "couple", "multiple", "various", "numerous", "a lot", "lots",
    "a handful", "handful", "certain", "n", "x", "y", "a", "an", "the",
    "several of", "some of",
}

# Direction groups for comparison / negation spans.
DIRECTION_GROUPS = {
    "more": {"more", "more than", "greater", "greater than", "larger", "bigger", "over", "above", "at least"},
    "less": {"less", "less than", "fewer", "fewer than", "smaller", "lower", "under", "below", "at most"},
    "equal": {"equal", "equal to", "same", "as many", "as much"},
}
NEGATION_WORDS = {"not", "no", "never", "without", "none", "cannot", "n't", "neither", "nor"}

UNIT_TOKENS = {
    "mile", "miles", "km", "kilometer", "kilometers", "meter", "meters", "m",
    "hour", "hours", "hr", "minute", "minutes", "second", "seconds", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "kg", "gram",
    "grams", "pound", "pounds", "lb", "lbs", "ounce", "ounces", "liter", "liters",
    "gallon", "gallons", "cup", "cups", "dollar", "dollars", "cent", "cents",
    "percent", "%", "$", "foot", "feet", "inch", "inches", "yard", "yards",
    "per",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%.,'-]+", (text or "").lower())


def _norm_token(token: str) -> str:
    return token.strip().strip(".,!?;:'\"").lower()


def extract_filler(masked_question: str, recovered_question: str, mask_token: str) -> str | None:
    """Recover the model's filler for the masked slot via word-level prefix/suffix trim.

    Tolerant to minor rephrasing outside the slot: trims the longest matching word
    prefix/suffix shared with the masked context and returns the middle.
    Returns None if alignment is impossible.
    """
    if not masked_question or recovered_question is None:
        return None
    idx = masked_question.find(mask_token)
    if idx < 0:
        return None

    prefix = masked_question[:idx]
    suffix = masked_question[idx + len(mask_token):]

    # Fast path: exact char alignment.
    if recovered_question.startswith(prefix) and (not suffix or recovered_question.endswith(suffix)):
        end = len(recovered_question) - len(suffix) if suffix else len(recovered_question)
        return recovered_question[len(prefix):end].strip()

    # Robust path: word-level LCP / LCS trimming.
    pw = _words(prefix)
    sw = _words(suffix)
    rw_tokens = re.findall(r"\S+", recovered_question)
    rw_norm = [_norm_token(t) for t in rw_tokens]
    pw_norm = [_norm_token(t) for t in pw]
    sw_norm = [_norm_token(t) for t in sw]

    i = 0
    while i < len(pw_norm) and i < len(rw_norm) and pw_norm[i] == rw_norm[i]:
        i += 1
    j = 0
    while j < len(sw_norm) and j < len(rw_norm) - i and sw_norm[-1 - j] == rw_norm[-1 - j]:
        j += 1

    middle = rw_tokens[i:len(rw_tokens) - j] if j > 0 else rw_tokens[i:]
    if not middle or i + j > len(rw_tokens):
        return None
    return " ".join(middle).strip()


def _contains_unit(text: str) -> set[str]:
    return {tok for tok in _words(text) if tok in UNIT_TOKENS} | (
        {"%"} if "%" in (text or "") else set()
    ) | ({"$"} if "$" in (text or "") else set())


def _direction_of(text: str) -> str | None:
    norm = (text or "").strip().lower()
    for group, members in DIRECTION_GROUPS.items():
        if norm in members:
            return group
    for group, members in DIRECTION_GROUPS.items():
        if any(re.search(rf"\b{re.escape(m)}\b", norm) for m in members):
            return group
    return None


def _has_negation(text: str) -> bool:
    toks = set(_words(text))
    return bool(toks & NEGATION_WORDS) or "n't" in (text or "").lower()


def _is_generic_quantity(text: str) -> bool:
    norm = (text or "").strip().lower().strip(".")
    return norm in GENERIC_QUANTITY_WORDS


def _singular(word: str) -> str:
    w = word.strip().lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 2:
        return w[:-2]
    if w.endswith("s") and len(w) > 1:
        return w[:-1]
    return w


def _plural_singular_match(a: str, b: str) -> bool:
    return _singular(a) == _singular(b)


def _looks_like_rephrase(filler: str, span_text: str) -> bool:
    """A filler far longer than the original span means the model rewrote the
    question instead of filling the slot; such a filler is not usable drift
    evidence and should not be scored as a concrete drift."""
    filler_words = len(filler.split())
    span_words = max(1, len((span_text or "").split()))
    return filler_words > max(8, 4 * span_words)


def classify_sample_drift(
    span_text: str,
    span_type: str,
    filler: str | None,
    is_uncertain: bool,
) -> str:
    """Classify a single recovery sample's drift relative to the original span."""
    if is_uncertain:
        return UNRECOVERABLE
    if filler is None:
        return AMBIGUOUS
    filler_norm = filler.strip()
    if not filler_norm:
        return UNRECOVERABLE

    # Model left a mask token in, or rephrased the whole question: not usable
    # drift evidence -> ambiguous (excluded from hard-drift escalation).
    if "[mask]" in filler_norm.lower() or "mask]" in filler_norm.lower():
        return AMBIGUOUS
    if _looks_like_rephrase(filler_norm, span_text):
        return AMBIGUOUS

    span_norm = (span_text or "").strip()

    # Exact surface match (case-insensitive) short-circuits for any type.
    if filler_norm.lower() == span_norm.lower():
        return EXACT
    # singular/plural variants of the same token count as exact recovery
    if len(filler_norm.split()) == 1 and len(span_norm.split()) == 1 and _plural_singular_match(
        filler_norm, span_norm
    ):
        return EXACT

    if span_type == "number":
        span_values = normalize_number_text(span_norm)
        filler_values = normalize_number_text(filler_norm)
        if span_values and filler_values:
            if any(abs(a - b) <= 1e-6 for a in span_values for b in filler_values):
                return EXACT
            return WRONG_NUMERIC
        if _is_generic_quantity(filler_norm):
            return GENERIC
        # non-numeric, non-generic filler for a number slot: precision lost
        return GENERIC

    if span_type == "comparison":
        span_dir = _direction_of(span_norm)
        filler_dir = _direction_of(filler_norm)
        if span_dir and filler_dir:
            return EXACT if span_dir == filler_dir else DIRECTION_DRIFT
        if filler_dir is None and span_dir is not None:
            return GENERIC
        return GENERIC

    if span_type == "negation":
        span_neg = _has_negation(span_norm)
        filler_neg = _has_negation(filler_norm)
        if span_neg and not filler_neg:
            return DIRECTION_DRIFT
        if span_neg == filler_neg:
            return EXACT
        return DIRECTION_DRIFT

    # unit-bearing drift check applies to object/entity/relation/operation/condition
    span_units = _contains_unit(span_norm)
    filler_units = _contains_unit(filler_norm)
    if span_units and span_units != filler_units:
        return UNIT_DRIFT

    if span_type in {"object", "entity"}:
        if _is_generic_quantity(filler_norm) or filler_norm.lower() in {"items", "things", "objects", "stuff"}:
            return GENERIC
        # determiner/expansion overlap (span ⊆ filler or filler ⊆ span) is not a
        # concrete object substitution -> under-recovery, not hard drift.
        span_words = set(_words(span_norm))
        filler_words = set(_words(filler_norm))
        if span_words and filler_words and (span_words <= filler_words or filler_words <= span_words):
            return GENERIC
        return OBJECT_DRIFT

    if span_type == "condition":
        return CONDITION_DRIFT

    # operation / question_target / relation / other
    if _is_generic_quantity(filler_norm):
        return GENERIC
    return GENERIC


def aggregate_drift(sample_drift_labels: list[str]) -> dict[str, Any]:
    """Aggregate per-sample drift labels with worst-case-oriented summary fields."""
    counts = Counter(sample_drift_labels)
    total = len(sample_drift_labels)
    num_exact = counts.get(EXACT, 0)
    num_generic = counts.get(GENERIC, 0)
    num_wrong = counts.get(WRONG_NUMERIC, 0)
    num_unrecoverable = counts.get(UNRECOVERABLE, 0)
    num_direction = counts.get(DIRECTION_DRIFT, 0)
    num_unit = counts.get(UNIT_DRIFT, 0)
    num_object = counts.get(OBJECT_DRIFT, 0)
    num_condition = counts.get(CONDITION_DRIFT, 0)

    majority_label = counts.most_common(1)[0][0] if counts else UNRECOVERABLE
    hard_drift = {label for label in sample_drift_labels if label in HARD_DRIFT_LABELS}
    any_drift = any(label != EXACT for label in sample_drift_labels)
    top_count = counts.most_common(1)[0][1] if counts else 0
    inconsistency_rate = round(1.0 - (top_count / total), 6) if total else 0.0

    return {
        "majority_drift_label": majority_label,
        "any_wrong": num_wrong > 0,
        "any_direction_drift": num_direction > 0,
        "any_unit_drift": num_unit > 0,
        "any_object_drift": num_object > 0,
        "any_condition_drift": num_condition > 0,
        "any_hard_drift": len(hard_drift) > 0,
        "hard_drift_labels": sorted(hard_drift),
        "any_drift": any_drift,
        "inconsistency_rate": inconsistency_rate,
        "num_exact": num_exact,
        "num_generic": num_generic,
        "num_wrong": num_wrong,
        "num_unrecoverable": num_unrecoverable,
        "num_direction_drift": num_direction,
        "num_unit_drift": num_unit,
        "num_object_drift": num_object,
        "num_condition_drift": num_condition,
        "num_samples": total,
        "label_counts": dict(counts),
    }
