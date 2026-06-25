"""Build early attention anchor labels from unit evidence records.

This stage applies the deterministic ``early_evidence_rule_stub_v0`` rule to each
``unit_evidence`` record and emits one ``attention_anchor_label`` record. The
output is for pipeline validation only and is based on partial early evidence
(semantic necessity + recoverability); it is not a real attention importance
decision and does not produce guidance actions.

Authoritative schema / validator:
    src/recover_attention/schemas.py
      REQUIRED_FIELDS["attention_anchor_label"]
      FORBIDDEN_FIELDS["attention_anchor_label"]
      validate_attention_anchor_label_record
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Real
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    ALLOWED_RECOVERABILITY_LABELS,
    validate_attention_anchor_label_record,
    validate_unit_evidence_record,
)


DEFAULT_ATTENTION_LABEL_BACKEND = "early_evidence_rule_stub_v0"
SUPPORTED_ATTENTION_LABEL_BACKENDS = {"early_evidence_rule_stub_v0"}
DEFAULT_ATTENTION_LABEL_STATUS = "partial_evidence_label"

SEMANTIC_WEIGHT = 0.6
RECOVERABILITY_WEIGHT = 0.4

# Early risk score per recoverability label. This is an early heuristic only;
# Non-recoverable does not mean Strong Anchor and Recoverable does not mean
# unimportant.
RECOVERABILITY_RISK_SCORES = {
    "Misleading Recovery": 1.0,
    "Non-recoverable": 0.75,
    "Partially Recoverable": 0.50,
    "Recoverable": 0.25,
}

# Fields copied verbatim from the upstream unit_evidence record.
UNIT_EVIDENCE_COPY_FIELDS = (
    "unit_evidence_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "semantic_evidence",
    "recoverability_evidence",
    "available_signal_types",
    "missing_signal_types",
)

LABEL_THRESHOLDS = {
    "strong_anchor": 0.75,
    "medium_anchor": 0.55,
    "weak_anchor": 0.35,
}

LIMITATIONS = [
    "label is based on partial early evidence only.",
    "trajectory stability is not included.",
    "answer stability is not included.",
    "raw attention pattern is not included.",
    "attention steering effect is not included.",
    "no guidance_action or guidance_strength is produced.",
    "recoverability comes from oracle_stub_v0 + stub_rule_v0 in current pipeline.",
]


def compute_recoverability_risk_score(recoverability_label: str) -> float:
    """Map a recoverability label to an early risk score in [0, 1]."""
    _validate_recoverability_label(recoverability_label)
    return RECOVERABILITY_RISK_SCORES[recoverability_label]


def score_attention_anchor_stub_rule_v0(unit_evidence_record: dict) -> dict:
    """Compute the early score, label, and decision trace for one unit evidence record."""
    semantic_score = _validate_semantic_score(unit_evidence_record)

    recoverability_evidence = unit_evidence_record["recoverability_evidence"]
    if not isinstance(recoverability_evidence, dict):
        raise ValueError("recoverability_evidence must be a dict")
    if "recoverability_label" not in recoverability_evidence:
        raise ValueError("recoverability_evidence missing 'recoverability_label'")
    recoverability_label = recoverability_evidence["recoverability_label"]
    _validate_recoverability_label(recoverability_label)
    misleading_recovery = bool(recoverability_evidence.get("misleading_recovery", False))

    recoverability_risk_score = compute_recoverability_risk_score(recoverability_label)
    attention_importance_score = _clamp_score(
        SEMANTIC_WEIGHT * semantic_score + RECOVERABILITY_WEIGHT * recoverability_risk_score
    )
    attention_anchor_label, label_reason = _decide_label(
        attention_importance_score,
        recoverability_label,
        misleading_recovery,
    )

    return {
        "semantic_score": semantic_score,
        "recoverability_label": recoverability_label,
        "misleading_recovery": misleading_recovery,
        "recoverability_risk_score": recoverability_risk_score,
        "attention_importance_score": attention_importance_score,
        "attention_anchor_label": attention_anchor_label,
        "label_reason": label_reason,
    }


def build_attention_anchor_label_record(
    unit_evidence_record: dict,
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> dict:
    """Build one validated attention anchor label record from one unit evidence record."""
    _validate_label_backend(label_backend)
    validate_unit_evidence_record(unit_evidence_record)

    scored = score_attention_anchor_stub_rule_v0(unit_evidence_record)
    unit_evidence_id = unit_evidence_record["unit_evidence_id"]

    record = {field: deepcopy(unit_evidence_record[field]) for field in UNIT_EVIDENCE_COPY_FIELDS}
    record["attention_anchor_label_id"] = f"{unit_evidence_id}__anchor_{label_backend}"
    record["attention_importance_score"] = scored["attention_importance_score"]
    record["attention_anchor_label"] = scored["attention_anchor_label"]
    record["label_backend"] = label_backend
    record["label_status"] = DEFAULT_ATTENTION_LABEL_STATUS
    record["evidence"] = {
        "rule_name": label_backend,
        "score_formula": (
            f"{SEMANTIC_WEIGHT} * semantic_score + "
            f"{RECOVERABILITY_WEIGHT} * recoverability_risk_score"
        ),
        "semantic_score": scored["semantic_score"],
        "recoverability_label": scored["recoverability_label"],
        "recoverability_risk_score": scored["recoverability_risk_score"],
        "attention_importance_score": scored["attention_importance_score"],
        "label_thresholds": dict(LABEL_THRESHOLDS),
        "label_reason": scored["label_reason"],
        "source_unit_evidence_id": unit_evidence_id,
        "source_files": ["data/processed/unit_evidence.jsonl"],
        "available_signal_types": deepcopy(unit_evidence_record["available_signal_types"]),
        "missing_signal_types": deepcopy(unit_evidence_record["missing_signal_types"]),
        "limitations": list(LIMITATIONS),
        "notes": (
            "Early attention anchor label from partial evidence stub; "
            "not a final attention importance decision and not an intervention."
        ),
    }

    validate_attention_anchor_label_record(record)
    return record


def build_attention_anchor_label_records(
    unit_evidence_records: list[dict],
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> tuple[list[dict], dict]:
    """Build attention anchor label records and summary statistics."""
    _validate_label_backend(label_backend)
    labeled_records = [
        build_attention_anchor_label_record(record, label_backend=label_backend)
        for record in unit_evidence_records
    ]
    stats = _build_stats(unit_evidence_records, labeled_records, label_backend)
    return labeled_records, stats


def build_attention_anchor_label_file(
    input_path: str | Path,
    output_path: str | Path,
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> tuple[list[dict], dict]:
    """Read unit evidence, build attention anchor labels, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing unit evidence input: {input_jsonl}\nPlease run Sprint 1I first."
        )

    unit_evidence_records = read_jsonl(input_jsonl)
    records, stats = build_attention_anchor_label_records(
        unit_evidence_records,
        label_backend=label_backend,
    )
    write_jsonl(records, output_path)
    return records, stats


def _decide_label(
    attention_importance_score: float,
    recoverability_label: str,
    misleading_recovery: bool,
) -> tuple[str, str]:
    if recoverability_label == "Misleading Recovery" or misleading_recovery:
        return "Risky Anchor", "misleading recovery flagged as early risk"
    if attention_importance_score >= LABEL_THRESHOLDS["strong_anchor"]:
        return "Strong Anchor", "attention_importance_score >= strong_anchor threshold"
    if attention_importance_score >= LABEL_THRESHOLDS["medium_anchor"]:
        return "Medium Anchor", "attention_importance_score >= medium_anchor threshold"
    if attention_importance_score >= LABEL_THRESHOLDS["weak_anchor"]:
        return "Weak Anchor", "attention_importance_score >= weak_anchor threshold"
    return "Distractor", "attention_importance_score below weak_anchor threshold"


def _validate_label_backend(label_backend: str) -> None:
    if label_backend not in SUPPORTED_ATTENTION_LABEL_BACKENDS:
        raise ValueError(f"Unsupported attention label backend: {label_backend}")


def _validate_semantic_score(unit_evidence_record: dict) -> float:
    semantic_evidence = unit_evidence_record.get("semantic_evidence")
    if not isinstance(semantic_evidence, dict):
        raise ValueError("semantic_evidence must be a dict")
    if "summary_score" not in semantic_evidence:
        raise ValueError("semantic_evidence missing 'summary_score'")
    semantic_score = semantic_evidence["summary_score"]
    if not isinstance(semantic_score, Real) or isinstance(semantic_score, bool):
        raise ValueError("semantic_evidence['summary_score'] must be an int or float")
    if semantic_score < 0 or semantic_score > 1:
        raise ValueError("semantic_evidence['summary_score'] must be between 0 and 1")
    return float(semantic_score)


def _validate_recoverability_label(recoverability_label: str) -> None:
    if recoverability_label not in ALLOWED_RECOVERABILITY_LABELS:
        allowed_values = ", ".join(sorted(ALLOWED_RECOVERABILITY_LABELS))
        raise ValueError(
            f"invalid recoverability_label {recoverability_label!r}; "
            f"allowed values: {allowed_values}"
        )


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, round(float(score), 10)))


def _build_stats(
    unit_evidence_records: list[dict],
    labeled_records: list[dict],
    label_backend: str,
) -> dict:
    scores = [record["attention_importance_score"] for record in labeled_records]
    available_signal_counter: Counter = Counter()
    missing_signal_counter: Counter = Counter()
    for record in labeled_records:
        available_signal_counter.update(record["available_signal_types"])
        missing_signal_counter.update(record["missing_signal_types"])

    return {
        "num_input_unit_evidence": len(unit_evidence_records),
        "num_output_attention_anchor_labels": len(labeled_records),
        "label_backend": label_backend,
        "label_status_counts": dict(
            sorted(Counter(record["label_status"] for record in labeled_records).items())
        ),
        "attention_anchor_label_counts": dict(
            sorted(Counter(record["attention_anchor_label"] for record in labeled_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in labeled_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in labeled_records).items())
        ),
        "available_signal_type_counts": dict(sorted(available_signal_counter.items())),
        "missing_signal_type_counts": dict(sorted(missing_signal_counter.items())),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 10) if scores else None,
        "num_risky_anchor": sum(
            1 for record in labeled_records if record["attention_anchor_label"] == "Risky Anchor"
        ),
    }
