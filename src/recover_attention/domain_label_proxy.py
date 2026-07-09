"""Domain label proxy helpers for finite-label cyber MCQ tasks."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

LABEL_RE = re.compile(r"\b(?:Answer|Final answer)\s*:\s*([A-Z])\b", re.IGNORECASE)
LETTER_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", re.IGNORECASE)

EVAL_ONLY_FIELD_NAMES = {
    "gold_label",
    "gold_label_id",
    "gold_label_text",
    "is_correct",
    "wrong_label",
}


def _is_whitespace_token(piece: str | None) -> bool:
    if piece is None:
        return False
    return piece.replace("臓", "").replace("膴", "").replace("鈻?", "").strip() == ""


def option_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Map option letters to single non-whitespace token ids."""

    out: dict[str, int] = {}
    for label in labels:
        ids = tokenizer(str(label).strip(), add_special_tokens=False)["input_ids"]
        clean: list[int] = []
        for token_id in ids:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if _is_whitespace_token(piece):
                continue
            clean.append(int(token_id))
        if len(clean) != 1:
            raise ValueError(f"option label {label!r} is not a single non-whitespace token: {clean}")
        out[str(label)] = clean[0]
    if len(set(out.values())) != len(out):
        raise ValueError(f"option labels do not have distinct token ids: {out}")
    return out


def parse_option_answer(text: str, candidate_labels: list[str] | None = None) -> dict[str, Any]:
    """Parse an option-letter answer without forcing a label on failure."""

    labels = {str(x).upper() for x in (candidate_labels or ["A", "B", "C", "D"])}
    for match in LABEL_RE.finditer(text):
        label = match.group(1).upper()
        if label in labels:
            return {"parsed_label": label, "parse_method": "answer_prefix", "parse_failed": False}
    last = None
    for match in LETTER_RE.finditer(text):
        label = match.group(1).upper()
        if label in labels:
            last = label
    if last is not None:
        return {"parsed_label": last, "parse_method": "last_standalone_letter", "parse_failed": False}
    return {"parsed_label": None, "parse_method": "parse_failure", "parse_failed": True}


def locate_label_readout_position(tokenizer: Any, text: str, label: str) -> dict[str, Any]:
    """Locate the token slot immediately before a label occurrence after Answer:"""

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [[int(s), int(e)] for s, e in encoded["offset_mapping"]]
    pattern = re.compile(r"(?:Answer|Final answer)\s*:\s*" + re.escape(label), re.IGNORECASE)
    match = None
    for match in pattern.finditer(text):
        pass
    if match is None:
        return {"found": False, "label_token_position": None, "readout_position": None}
    label_start = match.end() - len(label)
    label_token_position = None
    for idx, (start, end) in enumerate(offsets):
        if end > start and start <= label_start < end:
            label_token_position = idx
            break
    if label_token_position is None or label_token_position <= 0:
        return {"found": False, "label_token_position": label_token_position, "readout_position": None}
    return {
        "found": True,
        "label_token_position": int(label_token_position),
        "readout_position": int(label_token_position) - 1,
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12)


def label_distribution(logits: Any, token_ids: dict[str, int]) -> dict[str, float]:
    arr = np.asarray([float(logits[int(tid)]) for tid in token_ids.values()], dtype=float)
    probs = _softmax(arr)
    return {label: float(prob) for label, prob in zip(token_ids.keys(), probs)}


def label_margin(logits: Any, token_ids: dict[str, int]) -> dict[str, Any]:
    scores = {label: float(logits[int(token_id)]) for label, token_id in token_ids.items()}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top1 = ordered[0] if ordered else (None, None)
    top2 = ordered[1] if len(ordered) > 1 else (None, None)
    return {
        "top1_label": top1[0],
        "top2_label": top2[0],
        "top1_score": top1[1],
        "top2_score": top2[1],
        "margin": (top1[1] - top2[1]) if top1[1] is not None and top2[1] is not None else None,
    }


def label_entropy(logits: Any, token_ids: dict[str, int]) -> float:
    arr = np.asarray([float(logits[int(tid)]) for tid in token_ids.values()], dtype=float)
    probs = _softmax(arr)
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())


def full_entropy(logits: Any) -> float:
    arr = np.asarray(logits, dtype=float)
    probs = _softmax(arr)
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())


def self_consistency_features(sampled_labels: list[str | None], greedy_label: str | None) -> dict[str, Any]:
    clean = [x for x in sampled_labels if x is not None]
    if not clean or greedy_label is None:
        return {"f5_self_consistency": None, "f5_sc_majority_agree": None, "sample_majority_label": None}
    counts = {label: clean.count(label) for label in sorted(set(clean))}
    majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "f5_self_consistency": clean.count(greedy_label) / len(clean),
        "f5_sc_majority_agree": 1.0 if majority == greedy_label else 0.0,
        "sample_majority_label": majority,
    }


def fixed_f5_risk_score(features: dict[str, Any]) -> float | None:
    values = [
        -float(features["f5_label_margin"]) if features.get("f5_label_margin") is not None else None,
        float(features["f5_label_entropy"]) if features.get("f5_label_entropy") is not None else None,
        1.0 - float(features["f5_self_consistency"]) if features.get("f5_self_consistency") is not None else None,
        1.0 - float(features["f5_sc_majority_agree"]) if features.get("f5_sc_majority_agree") is not None else None,
    ]
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else None


def assert_no_gold_feature_leakage(feature_names: list[str]) -> None:
    leaked = sorted(set(feature_names) & EVAL_ONLY_FIELD_NAMES)
    leaked += sorted(name for name in feature_names if name.startswith("gold_") or "gold_label" in name)
    if leaked:
        raise ValueError(f"eval-only fields leaked into feature list: {sorted(set(leaked))}")


def classify_trace_by_option(parsed_label: str | None, gold_label: str) -> dict[str, Any]:
    if parsed_label is None:
        return {"is_correct": None, "wrong_label": None}
    is_correct = str(parsed_label).upper() == str(gold_label).upper()
    return {"is_correct": bool(is_correct), "wrong_label": 0 if is_correct else 1}
