"""Deterministic NLI semantic consistency scoring stub."""

from __future__ import annotations

from collections import Counter
import re

from recover_attention.schemas import (
    validate_ablated_question_record,
    validate_nli_score_record,
)


SUPPORTED_BACKENDS = {"stub_v0"}
SUPPORTED_LANGUAGES = {"auto", "en", "zh"}
SUPPORTED_ABLATION_TYPES = {"delete", "generalize"}
SUPPORTED_DIRECTIONS = {"forward", "backward"}
NLI_LABELS = ("entailment", "neutral", "contradiction")
LABEL_PRIORITY = {"entailment": 3, "neutral": 2, "contradiction": 1}

BASE_STUB_SCORES = {
    "generalize": {
        "forward": {"entailment": 0.75, "neutral": 0.20, "contradiction": 0.05},
        "backward": {"entailment": 0.35, "neutral": 0.60, "contradiction": 0.05},
    },
    "delete": {
        "forward": {"entailment": 0.55, "neutral": 0.40, "contradiction": 0.05},
        "backward": {"entailment": 0.25, "neutral": 0.70, "contradiction": 0.05},
    },
}


def detect_language(text: str) -> str:
    """Return zh when text contains CJK unified ideographs, otherwise en."""
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def resolve_record_language(record: dict, language: str = "auto") -> str:
    """Resolve one record's language from the requested setting."""
    _validate_language(language)
    if language in {"en", "zh"}:
        return language

    combined_text = f"{record.get('original_question', '')}\n{record.get('ablated_question', '')}"
    return detect_language(combined_text)


def score_nli_pair_stub(
    premise: str,
    hypothesis: str,
    ablation_type: str,
    num_spans: int,
    direction: str,
) -> dict:
    """Score one NLI direction with deterministic stub_v0 rules."""
    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("premise must be a non-empty str")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty str")
    _validate_ablation_type(ablation_type)
    _validate_direction(direction)
    _validate_num_spans(num_spans)

    scores = dict(BASE_STUB_SCORES[ablation_type][direction])
    scores = _apply_group_penalty(scores, num_spans)
    label = _label_from_scores(scores)
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "scores": scores,
    }


def score_ablated_question_record(
    record: dict,
    backend: str = "stub_v0",
    language: str = "auto",
) -> dict:
    """Build one validated NLI score record from one ablated question record."""
    validate_ablated_question_record(record)
    _validate_backend(backend)
    resolved_language = resolve_record_language(record, language)
    num_spans = len(record["span_ids"])

    forward = score_nli_pair_stub(
        premise=record["original_question"],
        hypothesis=record["ablated_question"],
        ablation_type=record["ablation_type"],
        num_spans=num_spans,
        direction="forward",
    )
    backward = score_nli_pair_stub(
        premise=record["ablated_question"],
        hypothesis=record["original_question"],
        ablation_type=record["ablation_type"],
        num_spans=num_spans,
        direction="backward",
    )

    nli_record = {
        "nli_id": f"{record['ablation_id']}__nli_{backend}",
        "ablation_id": record["ablation_id"],
        "id": record["id"],
        "unit_id": record["unit_id"],
        "unit_scope": record["unit_scope"],
        "group_type": record["group_type"],
        "span_ids": list(record["span_ids"]),
        "spans": [dict(span) for span in record["spans"]],
        "ablation_type": record["ablation_type"],
        "original_question": record["original_question"],
        "ablated_question": record["ablated_question"],
        "nli_backend": backend,
        "language": resolved_language,
        "language_setting": language,
        "forward": forward,
        "backward": backward,
        "bidirectional_entailment_score": min(
            forward["scores"]["entailment"],
            backward["scores"]["entailment"],
        ),
        "contradiction_score": max(
            forward["scores"]["contradiction"],
            backward["scores"]["contradiction"],
        ),
    }
    validate_nli_score_record(nli_record)
    return nli_record


def score_ablated_question_records(
    records: list[dict],
    backend: str = "stub_v0",
    language: str = "auto",
) -> tuple[list[dict], dict]:
    """Score ablated question records and return records plus summary stats."""
    _validate_backend(backend)
    _validate_language(language)

    scored_records = [
        score_ablated_question_record(record, backend=backend, language=language)
        for record in records
    ]
    stats = {
        "num_input_ablations": len(records),
        "num_output_scores": len(scored_records),
        "backend": backend,
        "language_setting": language,
        "language_counts": dict(
            sorted(Counter(record["language"] for record in scored_records).items())
        ),
        "ablation_type_counts": dict(
            sorted(Counter(record["ablation_type"] for record in scored_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in scored_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in scored_records).items())
        ),
    }
    return scored_records, stats


def _apply_group_penalty(scores: dict[str, float], num_spans: int) -> dict[str, float]:
    penalty = min(0.20, 0.05 * (num_spans - 1)) if num_spans > 1 else 0.0
    adjusted = dict(scores)
    adjusted["entailment"] -= penalty
    adjusted["neutral"] += penalty
    return _rounded_scores(adjusted)


def _rounded_scores(scores: dict[str, float]) -> dict[str, float]:
    rounded = {label: round(float(scores[label]), 10) for label in NLI_LABELS}
    total = sum(rounded.values())
    if abs(total - 1.0) > 1e-10:
        rounded["neutral"] = round(rounded["neutral"] + (1.0 - total), 10)
    return rounded


def _label_from_scores(scores: dict[str, float]) -> str:
    return max(NLI_LABELS, key=lambda label: (scores[label], LABEL_PRIORITY[label]))


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")


def _validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")


def _validate_ablation_type(ablation_type: str) -> None:
    if ablation_type not in SUPPORTED_ABLATION_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_ABLATION_TYPES))
        raise ValueError(f"Unsupported ablation_type: {ablation_type}; allowed: {allowed}")


def _validate_direction(direction: str) -> None:
    if direction not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"Unsupported direction: {direction}; allowed: {allowed}")


def _validate_num_spans(num_spans: int) -> None:
    if not isinstance(num_spans, int) or isinstance(num_spans, bool) or num_spans < 1:
        raise ValueError("num_spans must be an int >= 1")
