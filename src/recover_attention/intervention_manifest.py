"""Build planned intervention manifests from attention anchor label records.

This stage maps each ``attention_anchor_label`` record to one planned-only
``intervention_manifest`` record. It records what a later stage intends to do; it
does not execute model inference, does not apply attention steering, and does not
produce guidance actions, answers, stability scores, or hidden-state caches.

Authoritative schema / validator:
    src/recover_attention/schemas.py
      REQUIRED_FIELDS["intervention_manifest"]
      FORBIDDEN_FIELDS["intervention_manifest"]
      validate_intervention_manifest_record
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_attention_anchor_label_record,
    validate_intervention_manifest_record,
)


DEFAULT_INTERVENTION_BACKEND = "manifest_stub_v0"
SUPPORTED_INTERVENTION_BACKENDS = {"manifest_stub_v0"}

DEFAULT_INTERVENTION_TYPE = "mask"
SUPPORTED_INTERVENTION_TYPES = {"mask", "remove", "replace"}

DEFAULT_TARGET_SCOPE = "unit"
SUPPORTED_TARGET_SCOPES = {"unit"}

DEFAULT_INTERVENTION_STATUS = "planned_only"

DEFAULT_MASK_TOKEN = "[MASK]"

SELECTION_POLICY = "all_attention_anchor_labels_included_for_pipeline_validation"

# Fields copied verbatim from the upstream attention anchor label record.
ANCHOR_COPY_FIELDS = (
    "attention_anchor_label_id",
    "unit_evidence_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "attention_importance_score",
    "attention_anchor_label",
    "label_backend",
    "label_status",
)

LIMITATIONS = [
    "planned-only manifest; no model execution.",
    "no attention steering is applied.",
    "no guidance_action or guidance_strength is produced.",
    "no baseline_answer / guided_answer / intervened_answer is produced.",
    "no hidden states or attention maps are read or cached.",
    "no trajectory stability or answer stability score is computed.",
    "attention_anchor_labels come from early_evidence_rule_stub_v0 and partial_evidence_label.",
]


def build_planned_operation(
    attention_anchor_label_record: dict,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> dict:
    """Build the planned (not executed) operation description for one unit."""
    _validate_intervention_type(intervention_type)
    _validate_mask_token(mask_token)

    span_ids = list(attention_anchor_label_record["span_ids"])
    target_texts = [span["text"] for span in attention_anchor_label_record["spans"]]

    planned_operation = {
        "operation_name": f"{intervention_type}_unit",
        "description": (
            f"Plan to {intervention_type} all spans in the unit for a future "
            "intervention / guidance experiment; not executed in this stage."
        ),
        "target_scope": DEFAULT_TARGET_SCOPE,
        "target_span_ids": span_ids,
        "target_texts": target_texts,
        "execution_required": True,
    }
    if intervention_type == "mask":
        planned_operation["mask_token"] = mask_token
    return planned_operation


def build_intervention_manifest_record(
    attention_anchor_label_record: dict,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> dict:
    """Build one validated planned intervention manifest record."""
    _validate_intervention_backend(intervention_backend)
    _validate_intervention_type(intervention_type)
    _validate_mask_token(mask_token)
    validate_attention_anchor_label_record(attention_anchor_label_record)

    anchor_id = attention_anchor_label_record["attention_anchor_label_id"]
    record = {
        field: deepcopy(attention_anchor_label_record[field]) for field in ANCHOR_COPY_FIELDS
    }
    record["intervention_id"] = (
        f"{anchor_id}__intervention_{intervention_type}_{intervention_backend}"
    )
    record["intervention_type"] = intervention_type
    record["target_scope"] = DEFAULT_TARGET_SCOPE
    record["intervention_backend"] = intervention_backend
    record["intervention_status"] = DEFAULT_INTERVENTION_STATUS
    record["planned_operation"] = build_planned_operation(
        attention_anchor_label_record,
        intervention_type=intervention_type,
        mask_token=mask_token,
    )
    record["evidence"] = {
        "source_attention_anchor_label_id": anchor_id,
        "source_unit_evidence_id": attention_anchor_label_record["unit_evidence_id"],
        "source_files": ["data/processed/attention_anchor_labels.jsonl"],
        "intervention_backend": intervention_backend,
        "intervention_type": intervention_type,
        "intervention_status": DEFAULT_INTERVENTION_STATUS,
        "target_scope": DEFAULT_TARGET_SCOPE,
        "selection_policy": SELECTION_POLICY,
        "limitations": list(LIMITATIONS),
        "notes": (
            "Planned-only intervention manifest from stub backend; "
            "not an execution result and not an attention guidance result."
        ),
    }

    validate_intervention_manifest_record(record)
    return record


def build_intervention_manifest_records(
    attention_anchor_label_records: list[dict],
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> tuple[list[dict], dict]:
    """Build planned intervention manifest records and summary statistics."""
    _validate_intervention_backend(intervention_backend)
    _validate_intervention_type(intervention_type)
    _validate_mask_token(mask_token)

    manifest_records = [
        build_intervention_manifest_record(
            record,
            intervention_type=intervention_type,
            intervention_backend=intervention_backend,
            mask_token=mask_token,
        )
        for record in attention_anchor_label_records
    ]
    stats = _build_stats(attention_anchor_label_records, manifest_records, intervention_backend)
    return manifest_records, stats


def build_intervention_manifest_file(
    input_path: str | Path,
    output_path: str | Path,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> tuple[list[dict], dict]:
    """Read attention anchor labels, build planned manifests, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing attention anchor labels input: {input_jsonl}\n"
            "Please run Sprint 1J first."
        )

    anchor_records = read_jsonl(input_jsonl)
    records, stats = build_intervention_manifest_records(
        anchor_records,
        intervention_type=intervention_type,
        intervention_backend=intervention_backend,
        mask_token=mask_token,
    )
    write_jsonl(records, output_path)
    return records, stats


def _validate_intervention_backend(intervention_backend: str) -> None:
    if intervention_backend not in SUPPORTED_INTERVENTION_BACKENDS:
        raise ValueError(f"Unsupported intervention backend: {intervention_backend}")


def _validate_intervention_type(intervention_type: str) -> None:
    if intervention_type not in SUPPORTED_INTERVENTION_TYPES:
        raise ValueError(f"Unsupported intervention type: {intervention_type}")


def _validate_mask_token(mask_token: str) -> None:
    if not isinstance(mask_token, str) or not mask_token.strip():
        raise ValueError("mask_token must be a non-empty str")


def _build_stats(
    attention_anchor_label_records: list[dict],
    manifest_records: list[dict],
    intervention_backend: str,
) -> dict:
    intervention_type_counts = Counter(
        record["intervention_type"] for record in manifest_records
    )
    return {
        "num_input_attention_anchor_labels": len(attention_anchor_label_records),
        "num_output_intervention_manifest": len(manifest_records),
        "intervention_backend": intervention_backend,
        "intervention_type_counts": dict(sorted(intervention_type_counts.items())),
        "intervention_status_counts": dict(
            sorted(Counter(record["intervention_status"] for record in manifest_records).items())
        ),
        "target_scope_counts": dict(
            sorted(Counter(record["target_scope"] for record in manifest_records).items())
        ),
        "attention_anchor_label_counts": dict(
            sorted(Counter(record["attention_anchor_label"] for record in manifest_records).items())
        ),
        "label_status_counts": dict(
            sorted(Counter(record["label_status"] for record in manifest_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in manifest_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in manifest_records).items())
        ),
        "num_planned_only": sum(
            1 for record in manifest_records if record["intervention_status"] == "planned_only"
        ),
        "num_mask": intervention_type_counts.get("mask", 0),
        "num_remove": intervention_type_counts.get("remove", 0),
        "num_replace": intervention_type_counts.get("replace", 0),
    }
