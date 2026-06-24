"""Rule-based semantic necessity labels built from NLI score records."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Real

from recover_attention.schemas import (
    validate_nli_score_record,
    validate_semantic_label_record,
)


DEFAULT_EQUIVALENT_THRESHOLD = 0.70
DEFAULT_DIRECTIONAL_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_CONTRADICTION_THRESHOLD = 0.50
SUPPORTED_SEMANTIC_LABEL_BACKENDS = {"rule_v0"}
SEMANTIC_NECESSITY_LABELS = {
    "Equivalent",
    "Information Loss",
    "Added Assumption",
    "Non-equivalent",
}
REQUIRED_RULE_PARAMETERS = (
    "equivalent_threshold",
    "directional_entailment_threshold",
    "contradiction_threshold",
)
NLI_SCORE_FIELDS = (
    "nli_id",
    "ablation_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "ablation_type",
    "original_question",
    "ablated_question",
    "nli_backend",
    "language",
    "language_setting",
    "forward",
    "backward",
    "bidirectional_entailment_score",
    "contradiction_score",
)


def default_rule_parameters() -> dict:
    """Return the default rule_v0 threshold parameters."""
    return {
        "equivalent_threshold": DEFAULT_EQUIVALENT_THRESHOLD,
        "directional_entailment_threshold": DEFAULT_DIRECTIONAL_ENTAILMENT_THRESHOLD,
        "contradiction_threshold": DEFAULT_CONTRADICTION_THRESHOLD,
    }


def validate_rule_parameters(rule_parameters: dict) -> None:
    """Validate rule_v0 threshold parameters."""
    if not isinstance(rule_parameters, dict):
        raise ValueError("rule_parameters must be a dict")

    missing = [
        parameter for parameter in REQUIRED_RULE_PARAMETERS if parameter not in rule_parameters
    ]
    if missing:
        raise ValueError(f"rule_parameters missing required field(s): {', '.join(missing)}")

    for parameter in REQUIRED_RULE_PARAMETERS:
        value = rule_parameters[parameter]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"rule_parameters field '{parameter}' must be an int or float")
        if value < 0 or value > 1:
            raise ValueError(f"rule_parameters field '{parameter}' must be between 0 and 1")


def assign_semantic_necessity_label(
    nli_score_record: dict,
    rule_parameters: dict | None = None,
) -> tuple[str, str]:
    """Assign a semantic necessity label and decision reason with rule_v0."""
    validate_nli_score_record(nli_score_record)
    resolved_parameters = _resolve_rule_parameters(rule_parameters)

    forward_entailment = nli_score_record["forward"]["scores"]["entailment"]
    backward_entailment = nli_score_record["backward"]["scores"]["entailment"]
    bidirectional_entailment_score = nli_score_record["bidirectional_entailment_score"]
    contradiction_score = nli_score_record["contradiction_score"]

    contradiction_threshold = resolved_parameters["contradiction_threshold"]
    equivalent_threshold = resolved_parameters["equivalent_threshold"]
    directional_threshold = resolved_parameters["directional_entailment_threshold"]

    if contradiction_score >= contradiction_threshold:
        return "Non-equivalent", "contradiction above threshold"
    if bidirectional_entailment_score >= equivalent_threshold:
        return "Equivalent", "bidirectional entailment above equivalent threshold"
    if forward_entailment >= directional_threshold and backward_entailment < directional_threshold:
        return (
            "Information Loss",
            "forward entails ablated but backward does not entail original",
        )
    if forward_entailment < directional_threshold and backward_entailment >= directional_threshold:
        return (
            "Added Assumption",
            "backward entails original but forward does not entail ablated",
        )
    return "Non-equivalent", "low bidirectional entailment"


def compute_semantic_necessity_score(
    bidirectional_entailment_score: float,
    contradiction_score: float,
) -> float:
    """Compute the numeric semantic necessity score."""
    _validate_score("bidirectional_entailment_score", bidirectional_entailment_score)
    _validate_score("contradiction_score", contradiction_score)
    return round(max(1.0 - bidirectional_entailment_score, contradiction_score), 10)


def label_nli_score_record(
    record: dict,
    backend: str = "rule_v0",
    rule_parameters: dict | None = None,
) -> dict:
    """Build one validated semantic label record from one NLI score record."""
    validate_nli_score_record(record)
    _validate_backend(backend)
    resolved_parameters = _resolve_rule_parameters(rule_parameters)

    semantic_necessity_label, decision_reason = assign_semantic_necessity_label(
        record,
        rule_parameters=resolved_parameters,
    )
    semantic_record = {field: deepcopy(record[field]) for field in NLI_SCORE_FIELDS}
    semantic_record.update(
        {
            "semantic_label_id": f"{record['nli_id']}__sem_{backend}",
            "semantic_label_backend": backend,
            "semantic_necessity_label": semantic_necessity_label,
            "semantic_necessity_score": compute_semantic_necessity_score(
                record["bidirectional_entailment_score"],
                record["contradiction_score"],
            ),
            "is_semantically_necessary": semantic_necessity_label != "Equivalent",
            "rule_parameters": dict(resolved_parameters),
            "decision_reason": decision_reason,
        }
    )
    ordered_record = {
        "semantic_label_id": semantic_record["semantic_label_id"],
        **{field: semantic_record[field] for field in NLI_SCORE_FIELDS},
        "semantic_label_backend": semantic_record["semantic_label_backend"],
        "semantic_necessity_label": semantic_record["semantic_necessity_label"],
        "semantic_necessity_score": semantic_record["semantic_necessity_score"],
        "is_semantically_necessary": semantic_record["is_semantically_necessary"],
        "rule_parameters": semantic_record["rule_parameters"],
        "decision_reason": semantic_record["decision_reason"],
    }
    validate_semantic_label_record(ordered_record)
    return ordered_record


def label_nli_score_records(
    records: list[dict],
    backend: str = "rule_v0",
    rule_parameters: dict | None = None,
) -> tuple[list[dict], dict]:
    """Label NLI score records and return semantic labels plus summary stats."""
    _validate_backend(backend)
    resolved_parameters = _resolve_rule_parameters(rule_parameters)
    labeled_records = [
        label_nli_score_record(
            record,
            backend=backend,
            rule_parameters=resolved_parameters,
        )
        for record in records
    ]
    stats = {
        "num_input_scores": len(records),
        "num_output_labels": len(labeled_records),
        "backend": backend,
        "rule_parameters": dict(resolved_parameters),
        "semantic_necessity_label_counts": dict(
            sorted(
                Counter(
                    record["semantic_necessity_label"] for record in labeled_records
                ).items()
            )
        ),
        "is_semantically_necessary_counts": dict(
            sorted(
                Counter(
                    record["is_semantically_necessary"] for record in labeled_records
                ).items()
            )
        ),
        "ablation_type_counts": dict(
            sorted(Counter(record["ablation_type"] for record in labeled_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in labeled_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in labeled_records).items())
        ),
        "language_counts": dict(
            sorted(Counter(record["language"] for record in labeled_records).items())
        ),
    }
    return labeled_records, stats


def _resolve_rule_parameters(rule_parameters: dict | None) -> dict:
    resolved_parameters = (
        default_rule_parameters() if rule_parameters is None else dict(rule_parameters)
    )
    validate_rule_parameters(resolved_parameters)
    return resolved_parameters


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_SEMANTIC_LABEL_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")


def _validate_score(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int or float")
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
