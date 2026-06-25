"""Lightweight schema validation helpers for JSONL records."""

from __future__ import annotations

from numbers import Real
from typing import Any


ALLOWED_SPAN_TYPES = {
    "number",
    "entity",
    "object",
    "operation",
    "relation",
    "question_target",
    "comparison",
    "negation",
    "condition",
    "cyber_security_term",
}

ALLOWED_ABLATION_TYPES = {"delete", "generalize", "replace", "mask"}
ALLOWED_ABLATED_QUESTION_ABLATION_TYPES = {"delete", "generalize"}
ALLOWED_NLI_LABELS = {"entailment", "neutral", "contradiction"}
ALLOWED_NLI_BACKENDS = {"stub_v0"}
ALLOWED_NLI_LANGUAGES = {"en", "zh"}
ALLOWED_NLI_LANGUAGE_SETTINGS = {"auto", "en", "zh"}
ALLOWED_SEMANTIC_NECESSITY_LABELS = {
    "Equivalent",
    "Information Loss",
    "Added Assumption",
    "Non-equivalent",
}
ALLOWED_SEMANTIC_LABEL_BACKENDS = {"rule_v0"}
REQUIRED_SEMANTIC_RULE_PARAMETERS = {
    "equivalent_threshold",
    "directional_entailment_threshold",
    "contradiction_threshold",
}
ALLOWED_RECOVERABLE_VALUES = {"yes", "no", "uncertain"}
ALLOWED_RECOVERABILITY_LABELS = {
    "Recoverable",
    "Partially Recoverable",
    "Non-recoverable",
    "Misleading Recovery",
}
ALLOWED_ATTENTION_ANCHOR_LABELS = {
    "Strong Anchor",
    "Medium Anchor",
    "Weak Anchor",
    "Risky Anchor",
    "Distractor",
}
ALLOWED_GUIDANCE_ACTIONS = {
    "boost",
    "suppress",
    "keep",
    "review",
}
ALLOWED_ABLATION_UNIT_SCOPES = {"single", "group"}
ALLOWED_ABLATION_UNIT_GROUP_TYPES = {
    "single",
    "repeated_surface",
    "number_set",
    "entity_set",
    "object_set",
    "cyber_security_term_set",
}
ALLOWED_MASK_BACKENDS = {"unit_mask_v0"}
ALLOWED_MASK_STRATEGIES = {"replace_each_span"}
ALLOWED_RECOVERY_BACKENDS = {"oracle_stub_v0"}

# Single source of truth for top-level record fields.
# Interface docs and label_schema.md are checked against these by
# tests/test_interface_consistency.py. Update here first.
REQUIRED_FIELDS = {
    "question": ["id", "dataset", "question", "gold_answer"],
    "candidate_span": ["id", "question", "candidates"],
    "ablation_unit": ["id", "question", "units"],
    "ablated_question": [
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
    ],
    "nli_score": [
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
    ],
    "semantic_label": [
        "semantic_label_id",
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
        "semantic_label_backend",
        "semantic_necessity_label",
        "semantic_necessity_score",
        "is_semantically_necessary",
        "rule_parameters",
        "decision_reason",
    ],
    "masked_question": [
        "masked_id",
        "id",
        "unit_id",
        "unit_scope",
        "group_type",
        "span_ids",
        "spans",
        "original_question",
        "masked_question",
        "mask_token",
        "mask_backend",
        "mask_strategy",
        "source_semantic_label_ids",
        "source_nli_ids",
        "source_ablation_ids",
        "semantic_sources",
    ],
    "recover_output": [
        "masked_id",
        "id",
        "unit_id",
        "unit_scope",
        "group_type",
        "span_ids",
        "spans",
        "original_question",
        "masked_question",
        "mask_token",
        "mask_backend",
        "mask_strategy",
        "recovered_question",
        "recovery_backend",
        "sample_id",
    ],
    "recover_score": [
        "id",
        "span_id",
        "recoverability_label",
        "confidence_mean",
        "recovery_consistency",
        "misleading_recovery",
    ],
    "attention_anchor_label": [
        "id",
        "span_id",
        "span_text",
        "span_type",
        "attention_importance_score",
        "attention_anchor_label",
        "guidance_action",
        "guidance_strength",
        "evidence",
    ],
}

# Top-level fields that must NOT appear (stale or future-stage fields).
FORBIDDEN_FIELDS = {
    "ablated_question": ["span_id", "span_text", "span_type"],
    "nli_score": ["original_to_ablated", "ablated_to_original", "semantic_necessity_label"],
    "masked_question": ["span_id", "span_text", "span_type"],
    "recover_output": [
        "span_id",
        "span_text",
        "span_type",
        "recoverable",
        "confidence",
        "reason",
        "recoverability_label",
    ],
}

# Binds each record type to the interface doc whose `required_fields` marker
# block is generated from REQUIRED_FIELDS. Shared by
# scripts/sync_interface_fields.py (generator) and
# tests/test_interface_consistency.py (gate). Record types not listed here have
# no interface doc and are not managed by the generator.
INTERFACE_DOCS = {
    "ablation_unit": "ablation_units_interface.md",
    "ablated_question": "ablated_questions_interface.md",
    "nli_score": "nli_scores_interface.md",
    "semantic_label": "semantic_labels_interface.md",
    "masked_question": "masked_questions_interface.md",
    "recover_output": "recover_outputs_interface.md",
}


def validate_question_record(record: dict) -> None:
    name = "question record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["question"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "dataset", name)
    _require_non_empty_str(record, "question", name)
    _require_non_empty_str(record, "gold_answer", name)


def validate_candidate_span_record(record: dict) -> None:
    name = "candidate span record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["candidate_span"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "question", name)

    candidates = record["candidates"]
    if not isinstance(candidates, list):
        raise ValueError(f"{name} field 'candidates' must be a list")

    for index, candidate in enumerate(candidates):
        candidate_name = f"{name} candidate[{index}]"
        _require_dict(candidate, candidate_name)
        _require_fields(candidate, ["span_id", "text", "type", "start", "end"], candidate_name)
        _require_non_empty_str(candidate, "span_id", candidate_name)
        _require_non_empty_str(candidate, "text", candidate_name)
        _require_enum(candidate, "type", ALLOWED_SPAN_TYPES, candidate_name)
        _require_int(candidate, "start", candidate_name, min_value=0)
        _require_int(candidate, "end", candidate_name)
        if candidate["end"] <= candidate["start"]:
            raise ValueError(f"{candidate_name} field 'end' must be greater than 'start'")


def validate_ablation_unit_record(record: dict) -> None:
    name = "ablation unit record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["ablation_unit"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "question", name)

    units = record["units"]
    if not isinstance(units, list):
        raise ValueError(f"{name} field 'units' must be a list")

    for index, unit in enumerate(units):
        unit_name = f"{name} unit[{index}]"
        _require_dict(unit, unit_name)
        _require_fields(
            unit,
            ["unit_id", "unit_scope", "group_type", "span_ids", "spans", "reason"],
            unit_name,
        )
        _require_non_empty_str(unit, "unit_id", unit_name)
        _require_enum(unit, "unit_scope", ALLOWED_ABLATION_UNIT_SCOPES, unit_name)
        _require_enum(unit, "group_type", ALLOWED_ABLATION_UNIT_GROUP_TYPES, unit_name)
        _require_non_empty_str(unit, "reason", unit_name)

        span_ids = unit["span_ids"]
        spans = unit["spans"]
        if not isinstance(span_ids, list) or not span_ids:
            raise ValueError(f"{unit_name} field 'span_ids' must be a non-empty list")
        if not all(isinstance(span_id, str) and span_id.strip() for span_id in span_ids):
            raise ValueError(f"{unit_name} field 'span_ids' must contain non-empty str values")
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"{unit_name} field 'spans' must be a non-empty list")
        if len(spans) != len(span_ids):
            raise ValueError(f"{unit_name} field 'spans' must have the same length as 'span_ids'")

        if unit["unit_scope"] == "single" and len(span_ids) != 1:
            raise ValueError(f"{unit_name} with unit_scope 'single' must contain exactly one span")
        if unit["unit_scope"] == "group" and len(span_ids) < 2:
            raise ValueError(f"{unit_name} with unit_scope 'group' must contain at least two spans")
        if unit["group_type"] == "single" and unit["unit_scope"] != "single":
            raise ValueError(f"{unit_name} with group_type 'single' must have unit_scope 'single'")
        if unit["group_type"] != "single" and unit["unit_scope"] != "group":
            raise ValueError(f"{unit_name} with non-single group_type must have unit_scope 'group'")

        for span_index, span in enumerate(spans):
            span_name = f"{unit_name} span[{span_index}]"
            _require_dict(span, span_name)
            _require_fields(span, ["span_id", "text", "type", "start", "end"], span_name)
            _require_non_empty_str(span, "span_id", span_name)
            _require_non_empty_str(span, "text", span_name)
            _require_enum(span, "type", ALLOWED_SPAN_TYPES, span_name)
            _require_int(span, "start", span_name, min_value=0)
            _require_int(span, "end", span_name)
            if span["end"] <= span["start"]:
                raise ValueError(f"{span_name} field 'end' must be greater than 'start'")
            if span["span_id"] != span_ids[span_index]:
                raise ValueError(f"{unit_name} span_ids order must match spans order")
            if record["question"][span["start"] : span["end"]] != span["text"]:
                raise ValueError(f"{span_name} offsets must match question text")


def validate_ablated_question_record(record: dict) -> None:
    name = "ablated question record"
    _require_dict(record, name)
    _reject_fields(record, FORBIDDEN_FIELDS["ablated_question"], name)
    _require_fields(record, REQUIRED_FIELDS["ablated_question"], name)
    _require_non_empty_str(record, "ablation_id", name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "unit_id", name)
    _require_enum(record, "unit_scope", ALLOWED_ABLATION_UNIT_SCOPES, name)
    _require_non_empty_str(record, "group_type", name)
    _require_enum(record, "ablation_type", ALLOWED_ABLATED_QUESTION_ABLATION_TYPES, name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "ablated_question", name)

    span_ids = record["span_ids"]
    spans = record["spans"]
    if not isinstance(span_ids, list) or not span_ids:
        raise ValueError(f"{name} field 'span_ids' must be a non-empty list")
    if not all(isinstance(span_id, str) and span_id.strip() for span_id in span_ids):
        raise ValueError(f"{name} field 'span_ids' must contain non-empty str values")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{name} field 'spans' must be a non-empty list")
    if len(spans) != len(span_ids):
        raise ValueError(f"{name} field 'spans' must have the same length as 'span_ids'")

    if record["unit_scope"] == "single" and len(span_ids) != 1:
        raise ValueError(f"{name} with unit_scope 'single' must contain exactly one span")
    if record["unit_scope"] == "group" and len(span_ids) < 2:
        raise ValueError(f"{name} with unit_scope 'group' must contain at least two spans")
    if record["group_type"] == "single" and record["unit_scope"] != "single":
        raise ValueError(f"{name} with group_type 'single' must have unit_scope 'single'")
    if record["group_type"] != "single" and record["unit_scope"] != "group":
        raise ValueError(f"{name} with non-single group_type must have unit_scope 'group'")
    if record["ablated_question"] == record["original_question"]:
        raise ValueError(f"{name} field 'ablated_question' must differ from 'original_question'")

    for span_index, span in enumerate(spans):
        span_name = f"{name} span[{span_index}]"
        _require_dict(span, span_name)
        _require_fields(span, ["span_id", "text", "type", "start", "end"], span_name)
        _require_non_empty_str(span, "span_id", span_name)
        _require_non_empty_str(span, "text", span_name)
        _require_enum(span, "type", ALLOWED_SPAN_TYPES, span_name)
        _require_int(span, "start", span_name, min_value=0)
        _require_int(span, "end", span_name)
        if span["end"] <= span["start"]:
            raise ValueError(f"{span_name} field 'end' must be greater than 'start'")
        if span["span_id"] != span_ids[span_index]:
            raise ValueError(f"{name} span_ids order must match spans order")
        if record["original_question"][span["start"] : span["end"]] != span["text"]:
            raise ValueError(f"{span_name} offsets must match original_question text")


def validate_nli_score_record(record: dict) -> None:
    name = "nli score record"
    _require_dict(record, name)
    _reject_fields(record, FORBIDDEN_FIELDS["nli_score"], name)
    _require_fields(record, REQUIRED_FIELDS["nli_score"], name)
    _require_non_empty_str(record, "nli_id", name)
    _require_non_empty_str(record, "ablation_id", name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "unit_id", name)
    _require_enum(record, "unit_scope", ALLOWED_ABLATION_UNIT_SCOPES, name)
    _require_non_empty_str(record, "group_type", name)
    _require_enum(record, "ablation_type", ALLOWED_ABLATED_QUESTION_ABLATION_TYPES, name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "ablated_question", name)
    _require_enum(record, "nli_backend", ALLOWED_NLI_BACKENDS, name)
    _require_enum(record, "language", ALLOWED_NLI_LANGUAGES, name)
    _require_enum(record, "language_setting", ALLOWED_NLI_LANGUAGE_SETTINGS, name)
    _require_number(record, "bidirectional_entailment_score", name, min_value=0, max_value=1)
    _require_number(record, "contradiction_score", name, min_value=0, max_value=1)

    span_ids = record["span_ids"]
    spans = record["spans"]
    if not isinstance(span_ids, list) or not span_ids:
        raise ValueError(f"{name} field 'span_ids' must be a non-empty list")
    if not all(isinstance(span_id, str) and span_id.strip() for span_id in span_ids):
        raise ValueError(f"{name} field 'span_ids' must contain non-empty str values")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{name} field 'spans' must be a non-empty list")
    if len(spans) != len(span_ids):
        raise ValueError(f"{name} field 'spans' must have the same length as 'span_ids'")

    for span_index, span in enumerate(spans):
        span_name = f"{name} span[{span_index}]"
        _require_dict(span, span_name)
        _require_fields(span, ["span_id", "text", "type", "start", "end"], span_name)
        _require_non_empty_str(span, "span_id", span_name)
        _require_non_empty_str(span, "text", span_name)
        _require_enum(span, "type", ALLOWED_SPAN_TYPES, span_name)
        _require_int(span, "start", span_name, min_value=0)
        _require_int(span, "end", span_name)
        if span["end"] <= span["start"]:
            raise ValueError(f"{span_name} field 'end' must be greater than 'start'")
        if span["span_id"] != span_ids[span_index]:
            raise ValueError(f"{name} span_ids order must match spans order")

    _validate_nli_direction(record["forward"], "forward", name)
    _validate_nli_direction(record["backward"], "backward", name)
    _validate_nli_direction_texts(record, name)

    forward_entailment = record["forward"]["scores"]["entailment"]
    backward_entailment = record["backward"]["scores"]["entailment"]
    expected_bidirectional = min(forward_entailment, backward_entailment)
    if abs(record["bidirectional_entailment_score"] - expected_bidirectional) >= 1e-6:
        raise ValueError(
            f"{name} field 'bidirectional_entailment_score' must equal "
            "min(forward entailment, backward entailment)"
        )

    forward_contradiction = record["forward"]["scores"]["contradiction"]
    backward_contradiction = record["backward"]["scores"]["contradiction"]
    expected_contradiction = max(forward_contradiction, backward_contradiction)
    if abs(record["contradiction_score"] - expected_contradiction) >= 1e-6:
        raise ValueError(
            f"{name} field 'contradiction_score' must equal "
            "max(forward contradiction, backward contradiction)"
        )


def validate_semantic_label_record(record: dict) -> None:
    name = "semantic label record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["semantic_label"], name)
    _require_non_empty_str(record, "semantic_label_id", name)
    _require_enum(record, "semantic_label_backend", ALLOWED_SEMANTIC_LABEL_BACKENDS, name)
    expected_semantic_label_id = (
        f"{record['nli_id']}__sem_{record['semantic_label_backend']}"
    )
    if record["semantic_label_id"] != expected_semantic_label_id:
        raise ValueError(
            f"{name} field 'semantic_label_id' must equal "
            "f'{nli_id}__sem_{semantic_label_backend}'"
        )

    nli_record = {
        field: record[field]
        for field in [
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
        ]
    }
    validate_nli_score_record(nli_record)

    _require_enum(
        record,
        "semantic_necessity_label",
        ALLOWED_SEMANTIC_NECESSITY_LABELS,
        name,
    )
    _require_number(record, "semantic_necessity_score", name, min_value=0, max_value=1)
    _require_bool(record, "is_semantically_necessary", name)
    _require_dict(record["rule_parameters"], f"{name} rule_parameters")
    _require_non_empty_str(record, "decision_reason", name)

    expected_is_necessary = record["semantic_necessity_label"] != "Equivalent"
    if record["is_semantically_necessary"] != expected_is_necessary:
        raise ValueError(
            f"{name} field 'is_semantically_necessary' must be false only for Equivalent"
        )

    missing_parameters = [
        parameter
        for parameter in sorted(REQUIRED_SEMANTIC_RULE_PARAMETERS)
        if parameter not in record["rule_parameters"]
    ]
    if missing_parameters:
        raise ValueError(
            f"{name} rule_parameters missing required field(s): "
            f"{', '.join(missing_parameters)}"
        )
    for parameter in sorted(REQUIRED_SEMANTIC_RULE_PARAMETERS):
        _require_number(
            record["rule_parameters"],
            parameter,
            f"{name} rule_parameters",
            min_value=0,
            max_value=1,
        )

    expected_score = round(
        max(1.0 - record["bidirectional_entailment_score"], record["contradiction_score"]),
        10,
    )
    if abs(record["semantic_necessity_score"] - expected_score) >= 1e-6:
        raise ValueError(
            f"{name} field 'semantic_necessity_score' must equal "
            "max(1 - bidirectional_entailment_score, contradiction_score)"
        )


def validate_masked_question_record(record: dict) -> None:
    name = "masked question record"
    _require_dict(record, name)
    _reject_fields(record, FORBIDDEN_FIELDS["masked_question"], name)
    _require_fields(record, REQUIRED_FIELDS["masked_question"], name)
    _require_non_empty_str(record, "masked_id", name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "unit_id", name)
    _require_enum(record, "unit_scope", ALLOWED_ABLATION_UNIT_SCOPES, name)
    _require_non_empty_str(record, "group_type", name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "masked_question", name)
    if record["masked_question"] == record["original_question"]:
        raise ValueError(f"{name} field 'masked_question' must differ from 'original_question'")
    _require_non_empty_str(record, "mask_token", name)
    _require_enum(record, "mask_backend", ALLOWED_MASK_BACKENDS, name)
    _require_enum(record, "mask_strategy", ALLOWED_MASK_STRATEGIES, name)

    span_ids = _require_non_empty_str_list(record, "span_ids", name)
    spans = record["spans"]
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{name} field 'spans' must be a non-empty list")
    if len(spans) != len(span_ids):
        raise ValueError(f"{name} field 'spans' must have the same length as 'span_ids'")

    if record["unit_scope"] == "single" and len(span_ids) != 1:
        raise ValueError(f"{name} with unit_scope 'single' must contain exactly one span")
    if record["unit_scope"] == "group" and len(span_ids) < 2:
        raise ValueError(f"{name} with unit_scope 'group' must contain at least two spans")

    if record["mask_strategy"] == "replace_each_span":
        mask_token = record["mask_token"]
        added_mask_count = record["masked_question"].count(mask_token) - record[
            "original_question"
        ].count(mask_token)
        if added_mask_count != len(span_ids):
            raise ValueError(
                f"{name} with mask_strategy 'replace_each_span' must add exactly "
                f"{len(span_ids)} mask_token(s), but added {added_mask_count}"
            )

    for span_index, span in enumerate(spans):
        span_name = f"{name} span[{span_index}]"
        _require_dict(span, span_name)
        _require_fields(span, ["span_id", "text", "type", "start", "end"], span_name)
        _require_non_empty_str(span, "span_id", span_name)
        _require_non_empty_str(span, "text", span_name)
        _require_enum(span, "type", ALLOWED_SPAN_TYPES, span_name)
        _require_int(span, "start", span_name, min_value=0)
        _require_int(span, "end", span_name)
        if span["end"] <= span["start"]:
            raise ValueError(f"{span_name} field 'end' must be greater than 'start'")
        if span["span_id"] != span_ids[span_index]:
            raise ValueError(f"{name} span_ids order must match spans order")

    source_semantic_label_ids = _require_non_empty_str_list(
        record,
        "source_semantic_label_ids",
        name,
    )
    source_nli_ids = _require_non_empty_str_list(record, "source_nli_ids", name)
    source_ablation_ids = _require_non_empty_str_list(record, "source_ablation_ids", name)

    semantic_sources = record["semantic_sources"]
    if not isinstance(semantic_sources, list) or not semantic_sources:
        raise ValueError(f"{name} field 'semantic_sources' must be a non-empty list")
    if len(semantic_sources) != len(source_semantic_label_ids):
        raise ValueError(
            f"{name} field 'semantic_sources' must have the same length as "
            "'source_semantic_label_ids'"
        )
    if len(semantic_sources) != len(source_nli_ids):
        raise ValueError(
            f"{name} field 'semantic_sources' must have the same length as 'source_nli_ids'"
        )
    if len(semantic_sources) != len(source_ablation_ids):
        raise ValueError(
            f"{name} field 'semantic_sources' must have the same length as "
            "'source_ablation_ids'"
        )

    for source_index, semantic_source in enumerate(semantic_sources):
        source_name = f"{name} semantic_sources[{source_index}]"
        _require_dict(semantic_source, source_name)
        _require_fields(
            semantic_source,
            [
                "semantic_label_id",
                "nli_id",
                "ablation_id",
                "ablation_type",
                "semantic_necessity_label",
                "semantic_necessity_score",
                "is_semantically_necessary",
                "decision_reason",
            ],
            source_name,
        )
        _require_non_empty_str(semantic_source, "semantic_label_id", source_name)
        _require_non_empty_str(semantic_source, "nli_id", source_name)
        _require_non_empty_str(semantic_source, "ablation_id", source_name)
        _require_enum(
            semantic_source,
            "ablation_type",
            ALLOWED_ABLATED_QUESTION_ABLATION_TYPES,
            source_name,
        )
        _require_enum(
            semantic_source,
            "semantic_necessity_label",
            ALLOWED_SEMANTIC_NECESSITY_LABELS,
            source_name,
        )
        _require_number(
            semantic_source,
            "semantic_necessity_score",
            source_name,
            min_value=0,
            max_value=1,
        )
        _require_bool(semantic_source, "is_semantically_necessary", source_name)
        _require_non_empty_str(semantic_source, "decision_reason", source_name)

        if semantic_source["semantic_label_id"] != source_semantic_label_ids[source_index]:
            raise ValueError(
                f"{name} semantic_sources semantic_label_id order must match "
                "source_semantic_label_ids"
            )
        if semantic_source["nli_id"] != source_nli_ids[source_index]:
            raise ValueError(f"{name} semantic_sources nli_id order must match source_nli_ids")
        if semantic_source["ablation_id"] != source_ablation_ids[source_index]:
            raise ValueError(
                f"{name} semantic_sources ablation_id order must match source_ablation_ids"
            )


def validate_recover_output_record(record: dict) -> None:
    name = "recover output record"
    _require_dict(record, name)
    _reject_fields(record, FORBIDDEN_FIELDS["recover_output"], name)
    _require_fields(record, REQUIRED_FIELDS["recover_output"], name)
    _require_non_empty_str(record, "masked_id", name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "unit_id", name)
    _require_enum(record, "unit_scope", ALLOWED_ABLATION_UNIT_SCOPES, name)
    _require_non_empty_str(record, "group_type", name)
    _require_non_empty_str(record, "original_question", name)
    _require_non_empty_str(record, "masked_question", name)
    _require_non_empty_str(record, "mask_token", name)
    _require_enum(record, "mask_backend", ALLOWED_MASK_BACKENDS, name)
    _require_enum(record, "mask_strategy", ALLOWED_MASK_STRATEGIES, name)
    _require_str(record, "recovered_question", name)
    _require_enum(record, "recovery_backend", ALLOWED_RECOVERY_BACKENDS, name)
    _require_int(record, "sample_id", name, min_value=0)

    expected_masked_id = f"{record['id']}__{record['unit_id']}__mask"
    if record["masked_id"] != expected_masked_id:
        raise ValueError(
            f"{name} field 'masked_id' must equal f'{{id}}__{{unit_id}}__mask'"
        )

    span_ids = _require_non_empty_str_list(record, "span_ids", name)
    spans = record["spans"]
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{name} field 'spans' must be a non-empty list")
    if len(spans) != len(span_ids):
        raise ValueError(f"{name} field 'spans' must have the same length as 'span_ids'")

    for span_index, span in enumerate(spans):
        span_name = f"{name} span[{span_index}]"
        _require_dict(span, span_name)
        _require_fields(span, ["span_id", "text", "type", "start", "end"], span_name)
        _require_non_empty_str(span, "span_id", span_name)
        _require_non_empty_str(span, "text", span_name)
        _require_enum(span, "type", ALLOWED_SPAN_TYPES, span_name)
        _require_int(span, "start", span_name, min_value=0)
        _require_int(span, "end", span_name)
        if span["end"] <= span["start"]:
            raise ValueError(f"{span_name} field 'end' must be greater than 'start'")
        if span["span_id"] != span_ids[span_index]:
            raise ValueError(f"{name} span_ids order must match spans order")

    if record["mask_strategy"] == "replace_each_span":
        added_mask_count = record["masked_question"].count(record["mask_token"]) - record[
            "original_question"
        ].count(record["mask_token"])
        if added_mask_count != len(spans):
            raise ValueError(
                f"{name} field 'masked_question' must add exactly one mask token per span"
            )


def validate_recover_score_record(record: dict) -> None:
    name = "recover score record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["recover_score"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_enum(record, "recoverability_label", ALLOWED_RECOVERABILITY_LABELS, name)
    _require_number(record, "confidence_mean", name, min_value=0, max_value=1)
    _require_number(record, "recovery_consistency", name, min_value=0, max_value=1)
    _require_bool(record, "misleading_recovery", name)


def validate_attention_anchor_label_record(record: dict) -> None:
    name = "attention anchor label record"
    _require_dict(record, name)
    _require_fields(record, REQUIRED_FIELDS["attention_anchor_label"], name)
    _require_non_empty_str(record, "id", name)
    _require_non_empty_str(record, "span_id", name)
    _require_non_empty_str(record, "span_text", name)
    _require_enum(record, "span_type", ALLOWED_SPAN_TYPES, name)
    _require_number(record, "attention_importance_score", name, min_value=0, max_value=1)
    _require_enum(record, "attention_anchor_label", ALLOWED_ATTENTION_ANCHOR_LABELS, name)
    _require_enum(record, "guidance_action", ALLOWED_GUIDANCE_ACTIONS, name)
    _require_number(record, "guidance_strength", name, min_value=0, max_value=1)
    _require_dict_or_list(record, "evidence", name)


def _validate_nli_direction(direction_record: Any, direction: str, parent_name: str) -> None:
    name = f"{parent_name} {direction}"
    _require_dict(direction_record, name)
    _require_fields(direction_record, ["premise", "hypothesis", "label", "scores"], name)
    _require_non_empty_str(direction_record, "premise", name)
    _require_non_empty_str(direction_record, "hypothesis", name)
    _require_enum(direction_record, "label", ALLOWED_NLI_LABELS, name)

    scores = direction_record["scores"]
    score_name = f"{name} scores"
    _require_dict(scores, score_name)
    _require_fields(scores, ["entailment", "neutral", "contradiction"], score_name)
    for label in ["entailment", "neutral", "contradiction"]:
        _require_number(scores, label, score_name, min_value=0, max_value=1)

    if abs(sum(scores.values()) - 1.0) >= 1e-6:
        raise ValueError(f"{score_name} values must sum to 1")


def _validate_nli_direction_texts(record: dict, name: str) -> None:
    if record["forward"]["premise"] != record["original_question"]:
        raise ValueError(f"{name} forward premise must equal original_question")
    if record["forward"]["hypothesis"] != record["ablated_question"]:
        raise ValueError(f"{name} forward hypothesis must equal ablated_question")
    if record["backward"]["premise"] != record["ablated_question"]:
        raise ValueError(f"{name} backward premise must equal ablated_question")
    if record["backward"]["hypothesis"] != record["original_question"]:
        raise ValueError(f"{name} backward hypothesis must equal original_question")


def _require_dict(record: Any, name: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a dict")


def _require_fields(record: dict, fields: list[str], name: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")


def _reject_fields(record: dict, fields: list[str], name: str) -> None:
    present = [field for field in fields if field in record]
    if present:
        raise ValueError(f"{name} contains forbidden field(s): {', '.join(present)}")


def _require_str(record: dict, field: str, name: str) -> None:
    if not isinstance(record[field], str):
        raise ValueError(f"{name} field '{field}' must be a str")


def _require_non_empty_str(record: dict, field: str, name: str) -> None:
    _require_str(record, field, name)
    if not record[field].strip():
        raise ValueError(f"{name} field '{field}' must be a non-empty str")


def _require_non_empty_str_list(record: dict, field: str, name: str) -> list[str]:
    values = record[field]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} field '{field}' must be a non-empty list")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{name} field '{field}' must contain non-empty str values")
    return values


def _require_enum(record: dict, field: str, allowed: set[str], name: str) -> None:
    _require_str(record, field, name)
    if record[field] not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"{name} field '{field}' has invalid value {record[field]!r}; "
            f"allowed values: {allowed_values}"
        )


def _require_int(
    record: dict,
    field: str,
    name: str,
    min_value: int | None = None,
) -> None:
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} field '{field}' must be an int")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} field '{field}' must be >= {min_value}")


def _require_number(
    record: dict,
    field: str,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    value = record[field]
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} field '{field}' must be an int or float")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} field '{field}' must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} field '{field}' must be <= {max_value}")


def _require_bool(record: dict, field: str, name: str) -> None:
    if not isinstance(record[field], bool):
        raise ValueError(f"{name} field '{field}' must be a bool")


def _require_dict_or_list(record: dict, field: str, name: str) -> None:
    if not isinstance(record[field], (dict, list)):
        raise ValueError(f"{name} field '{field}' must be a dict or list")
