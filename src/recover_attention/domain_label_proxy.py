"""Pure option-letter proxy helpers for finite-label cyber MCQ tasks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

FORBIDDEN_INFERENCE_FEATURE_FIELDS = {
    "gold_label",
    "gold_label_id",
    "gold_label_text",
    "correct_option",
    "answer_key",
    "target_label",
}
_EXPLICIT_ANSWER_RE = re.compile(r"\b(?:final\s+)?answer\s*:\s*([A-Z])\b", re.IGNORECASE)
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", re.IGNORECASE)
_TRAILING_PUNCTUATION_RE = re.compile(r"^[\s.,;:!?()\[\]{}]*$")
_TOKEN_WHITESPACE_MARKERS = ("Ġ", "▁", "Ċ", "ĉ", "▏")
_SINGLE_CHAR_RUN_RE = re.compile(r"(.)\1{29,}", re.S)
_SUBSTRING_LOOP_RE = re.compile(r"(.{6,40}?)\1{4,}", re.S)
DEFAULT_CHAR_PER_TOKEN_RATIO = 3.0


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    # HuggingFace returns BatchEncoding (a UserDict, NOT a dict subclass), so an
    # isinstance(encoded, dict) check misses it and iterating yields key names.
    # Duck-type on mapping behaviour instead.
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _token_piece_content(piece: str | None) -> str:
    if piece is None:
        return ""
    normalized = str(piece)
    for marker in _TOKEN_WHITESPACE_MARKERS:
        normalized = normalized.replace(marker, "")
    return normalized.strip()


def _resolve_option_token_ids(tokenizer: Any, labels: list[str], *, text_form: str) -> dict[str, int]:
    if len(labels) != len(set(labels)):
        raise ValueError(f"option labels must be unique: {labels!r}")
    result: dict[str, int] = {}
    for raw_label in labels:
        label = str(raw_label).strip().upper()
        if len(label) != 1:
            raise ValueError(f"option label must be one character: {raw_label!r}")
        probe_text = f" {label}" if text_form == "space" else label
        token_ids = _tokenizer_ids(tokenizer, probe_text)
        valid_ids = [
            token_id
            for token_id in token_ids
            if _token_piece_content(tokenizer.convert_ids_to_tokens(token_id))
        ]
        if len(valid_ids) != 1:
            raise ValueError(
                f"option label {label!r} ({text_form} form) must map to one non-whitespace token; "
                f"token_ids={token_ids}, valid_token_ids={valid_ids}"
            )
        result[label] = valid_ids[0]
    collisions = {
        token_id: sorted(label for label, candidate in result.items() if candidate == token_id)
        for token_id in set(result.values())
        if sum(candidate == token_id for candidate in result.values()) > 1
    }
    if collisions:
        raise ValueError(f"option label token-id collision ({text_form} form): labels={result}, collisions={collisions}")
    return result


def option_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Resolve each label to one distinct non-whitespace tokenizer token.

    Uses the leading-space form (" A", " B", ...), matching the "Answer: X"
    convention where the letter follows a space. See bare_option_token_ids
    for completions that emit the letter with no preceding space (e.g. a
    chat-turn that answers with just the letter, no restated prefix).
    """
    return _resolve_option_token_ids(tokenizer, labels, text_form="space")


def bare_option_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Resolve each label to its token id with NO leading-space prefix.

    Some completion styles emit the option letter as the very first token of
    a turn (observed on the Sprint 4B-2 chat-template condition: the model
    treats "Answer: <letter>" in the prompt as a format instruction and
    answers with the bare letter directly, never restating "Answer: "), which
    tokenizes to a different id than the " <letter>" form. Same validation as
    option_token_ids, without the space prefix.
    """
    return _resolve_option_token_ids(tokenizer, labels, text_form="bare")


def _valid_label_set(valid_labels: list[str]) -> tuple[list[str], set[str]]:
    normalized = [str(label).strip().upper() for label in valid_labels]
    if len(normalized) < 2 or len(normalized) != len(set(normalized)):
        raise ValueError("valid_labels must contain at least two unique labels")
    if any(len(label) != 1 for label in normalized):
        raise ValueError("valid_labels must contain single-character labels")
    return normalized, set(normalized)


def _trailing_isolated_matches(text: str, valid: set[str]) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in _ISOLATED_LETTER_RE.finditer(text):
        label = match.group(1).upper()
        if label in valid and _TRAILING_PUNCTUATION_RE.fullmatch(text[match.end() :]):
            matches.append(match)
    return matches


def parse_option_answer(text: str, valid_labels: list[str]) -> dict:
    """Parse an explicit answer marker, then a final isolated option letter."""
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    _, valid = _valid_label_set(valid_labels)
    explicit = [
        match for match in _EXPLICIT_ANSWER_RE.finditer(text)
        if match.group(1).upper() in valid
    ]
    if explicit:
        label = explicit[-1].group(1).upper()
        return {
            "parsed_label": label,
            "parse_method": "explicit_answer_marker",
            "parse_failure": False,
        }
    trailing = _trailing_isolated_matches(text, valid)
    if trailing:
        return {
            "parsed_label": trailing[-1].group(1).upper(),
            "parse_method": "last_isolated_label",
            "parse_failure": False,
        }
    return {
        "parsed_label": None,
        "parse_method": "parse_failure",
        "parse_failure": True,
    }


def detect_degeneration(
    completion: str,
    *,
    valid_labels: list[str],
    max_new_tokens: int,
    num_generated_tokens: int | None = None,
    char_per_token_ratio: float = DEFAULT_CHAR_PER_TOKEN_RATIO,
) -> dict:
    """Flag a completion as degenerate generation (loop/repeat/gibberish runaway).

    Three independent rules, any one match sets ``degenerate=True``:

    1. single_char_run: one character repeated >= 30 times consecutively
       (e.g. "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!").
    2. substring_loop: a substring of length 6-40 repeated >= 5 times
       consecutively (e.g. ".4922.4922.4922.4922.4922").
    3. truncated_failure: no option letter could be parsed AND the completion
       ran to (or near) the generation budget. When ``num_generated_tokens``
       is supplied (the caller controls the generation loop and knows the
       exact count) that is used directly -- this is the precise signal and
       is preferred. Without it, a character-length heuristic is used instead
       (``len(completion) >= 0.9 * max_new_tokens * char_per_token_ratio``);
       this is approximate because the chars-per-token ratio for gibberish
       completions varies widely (~2.3-6.1 observed on real Sprint 4B smoke
       failures) so the fallback can miss short completions that hit EOS
       mid-gibberish without exhausting the budget. Such cases still count
       toward parse_failure_rate even when not flagged degenerate here.
    """

    single_char_match = _SINGLE_CHAR_RUN_RE.search(completion)
    substring_match = _SUBSTRING_LOOP_RE.search(completion)
    parsed_failed = parse_option_answer(completion, valid_labels)["parse_failure"]
    if num_generated_tokens is not None:
        truncated = parsed_failed and num_generated_tokens >= 0.9 * max_new_tokens
    else:
        truncated = parsed_failed and len(completion) >= 0.9 * max_new_tokens * char_per_token_ratio
    matched_rules = []
    if single_char_match:
        matched_rules.append("single_char_run")
    if substring_match:
        matched_rules.append("substring_loop")
    if truncated:
        matched_rules.append("truncated_failure")
    return {
        "degenerate": bool(matched_rules),
        "matched_rules": matched_rules,
        "single_char_run_match": single_char_match.group(0)[:12] if single_char_match else None,
        "substring_loop_match": substring_match.group(1) if substring_match else None,
        "truncated_failure_basis": (
            "token_count" if num_generated_tokens is not None else "char_length_heuristic"
        ),
    }


def locate_label_readout_position(
    tokenizer: Any,
    prompt: str,
    completion: str,
    parsed_label: str,
) -> int:
    """Return the token position immediately before the parsed answer label."""
    label = str(parsed_label).strip().upper()
    if not label:
        raise ValueError("parsed_label must be non-empty")
    parsed = parse_option_answer(completion, [label, "_"])
    if parsed["parsed_label"] != label:
        raise ValueError(f"parsed label {label!r} cannot be located in completion")
    explicit = [
        match for match in _EXPLICIT_ANSWER_RE.finditer(completion)
        if match.group(1).upper() == label
    ]
    if explicit:
        candidates = [match.start(1) for match in explicit]
    else:
        candidates = [match.start(1) for match in _trailing_isolated_matches(completion, {label})]
    if len(candidates) != 1:
        raise ValueError(
            f"parsed label {label!r} must have exactly one answer occurrence; "
            f"found {len(candidates)}"
        )
    combined = prompt + completion
    encoded = tokenizer(
        combined,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # HuggingFace BatchEncoding is a UserDict, not a dict subclass; the same
    # duck-typing fix as _tokenizer_ids applies here (see its docstring).
    offsets = (
        encoded.get("offset_mapping")
        if isinstance(encoded, dict) or hasattr(encoded, "get")
        else None
    )
    if offsets is None:
        raise ValueError("tokenizer must provide offset_mapping")
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    label_start = len(prompt) + candidates[0]
    token_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(start) and int(start) <= label_start < int(end)
    ]
    if len(token_positions) != 1:
        raise ValueError(
            f"answer label must map to exactly one token position; found {token_positions}"
        )
    label_position = token_positions[0]
    if label_position == 0:
        raise ValueError("answer label has no preceding readout position")
    return label_position - 1


def _finite_values(values: Iterable[Any], *, name: str, minimum: int) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) < minimum:
        raise ValueError(f"{name} requires at least {minimum} values")
    if any(not math.isfinite(value) for value in converted):
        raise ValueError(f"{name} values must be finite real numbers")
    return converted


def label_margin(option_logits: dict[str, float]) -> float:
    """Return top-1 minus top-2 over candidate option logits."""
    if not isinstance(option_logits, dict):
        raise ValueError("option_logits must be a dict")
    ordered = sorted(
        _finite_values(option_logits.values(), name="label_margin", minimum=2),
        reverse=True,
    )
    return ordered[0] - ordered[1]


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def label_entropy(option_logits: dict[str, float]) -> float:
    """Compute entropy after normalizing only candidate option logits."""
    if not isinstance(option_logits, dict):
        raise ValueError("option_logits must be a dict")
    values = _finite_values(option_logits.values(), name="label_entropy", minimum=2)
    probabilities = _softmax(values)
    return -sum(probability * math.log(probability) for probability in probabilities)


def _flatten_logits(logits: Any) -> list[float]:
    if hasattr(logits, "detach") and hasattr(logits, "reshape"):
        logits = logits.detach().float().cpu().reshape(-1).tolist()
    elif hasattr(logits, "reshape") and hasattr(logits, "tolist"):
        logits = logits.reshape(-1).tolist()
    elif hasattr(logits, "tolist"):
        logits = logits.tolist()

    def walk(value: Any) -> Iterable[Any]:
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)
        else:
            yield value

    return _finite_values(walk(logits), name="full_entropy", minimum=2)


def full_entropy(logits: Any) -> float:
    """Compute numerically stable full-vocabulary softmax entropy."""
    probabilities = _softmax(_flatten_logits(logits))
    return -sum(probability * math.log(probability) for probability in probabilities)


def self_consistency_features(
    greedy_label: str | None,
    sampled_labels: list[str | None],
    valid_labels: list[str],
) -> dict:
    """Summarize parsed samples without accepting any evaluation label."""
    ordered_labels, valid = _valid_label_set(valid_labels)
    normalized_greedy = None if greedy_label is None else str(greedy_label).upper()
    if normalized_greedy is not None and normalized_greedy not in valid:
        raise ValueError(f"greedy_label {greedy_label!r} is outside valid_labels")
    normalized_samples = [
        None if label is None else str(label).upper() for label in sampled_labels
    ]
    invalid = [label for label in normalized_samples if label is not None and label not in valid]
    if invalid:
        raise ValueError(f"sampled label(s) outside valid_labels: {invalid}")
    counts = Counter(label for label in normalized_samples if label is not None)
    majority_label = None
    if counts:
        majority_label = max(
            ordered_labels,
            key=lambda label: (counts.get(label, 0), -ordered_labels.index(label)),
        )
    parsed_count = sum(counts.values())
    sample_count = len(normalized_samples)
    agreement = (
        counts.get(normalized_greedy, 0) / parsed_count
        if normalized_greedy is not None and parsed_count
        else None
    )
    return {
        "num_samples": sample_count,
        "num_parsed_samples": parsed_count,
        "parse_failure_count": sample_count - parsed_count,
        "parse_failure_rate": (
            (sample_count - parsed_count) / sample_count if sample_count else 0.0
        ),
        "greedy_label": normalized_greedy,
        "sample_vote_counts": {
            label: counts.get(label, 0) for label in ordered_labels
        },
        "sample_majority_label": majority_label,
        "self_consistency_with_greedy": agreement,
        "majority_agrees_with_greedy": (
            majority_label == normalized_greedy
            if majority_label is not None and normalized_greedy is not None
            else None
        ),
    }



def fixed_f5_risk_score(features: dict[str, Any]) -> float | None:
    """Compatibility fixed combination for the superseded Sprint 4B smoke."""
    raw_values = [
        -float(features["f5_label_margin"])
        if features.get("f5_label_margin") is not None
        else None,
        float(features["f5_label_entropy"])
        if features.get("f5_label_entropy") is not None
        else None,
        1.0 - float(features["f5_self_consistency"])
        if features.get("f5_self_consistency") is not None
        else None,
        1.0 - float(features["f5_sc_majority_agree"])
        if features.get("f5_sc_majority_agree") is not None
        else None,
    ]
    values = [value for value in raw_values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def assert_no_gold_label_leakage(feature_record: dict) -> None:
    """Reject evaluation-label keys anywhere inside a feature record."""
    if not isinstance(feature_record, dict):
        raise ValueError("feature_record must be a dict")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).casefold()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized_key in FORBIDDEN_INFERENCE_FEATURE_FIELDS:
                    raise ValueError(f"forbidden inference feature field at {child_path}")
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(feature_record, "")


def classify_trace_by_option(parsed_label: str | None, gold_label: str) -> str:
    """Classify a parsed trace for evaluation only."""
    if parsed_label is None:
        return "parse_failure"
    return (
        "correct"
        if str(parsed_label).strip().upper() == str(gold_label).strip().upper()
        else "wrong"
    )


def assert_no_gold_feature_leakage(feature_names: list[str]) -> None:
    """Compatibility wrapper for the superseded Sprint 4B smoke script."""
    assert_no_gold_label_leakage({name: None for name in feature_names})
