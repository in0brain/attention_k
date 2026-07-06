"""Sprint 2G full-scale weak labeling and downstream manifest construction.

For each manifest case this module follows the diagnostic chain:

    source question
    -> candidate spans (rule-based, no model)
    -> one deterministically chosen ablation unit
    -> masked question
    -> weak (deterministic) recovered question
    -> weak probe target (weak_auto label)

It emits two artifacts:

* ``weak_labels_2000.jsonl`` -- one weak-auto label per case, and
* ``full_scale_2a_manifest.jsonl`` -- a 2A-style manifest that the existing
  hidden-state cache (script 16) consumes (original / masked / recovered inputs).

The labels are weak/auto, never human. The ``human_*`` fields required by the
hidden-state cache manifest schema are filled with the explicit sentinel
``weak_auto_not_human_reviewed`` so they cannot be mistaken for human review.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recover_attention.candidate_extraction import extract_candidate_spans
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.token_alignment import DEFAULT_MASK_TOKEN

BACKEND = "weak_label_mapping_v0"
WEAK_LABELS_FILENAME = "weak_labels_2000.jsonl"
MANIFEST_2A_FILENAME = "full_scale_2a_manifest.jsonl"
REPORT_FILENAME = "weak_label_report.json"

LABEL_SOURCE = "weak_auto"
HUMAN_SENTINEL = "weak_auto_not_human_reviewed"

# (probe_target, label_rule, label_confidence) keyed by chosen span type.
WEAK_TARGET_BY_SPAN_TYPE: dict[str, tuple[str, str, float]] = {
    "number": ("positive_anchor", "span_type_number_critical_value", 0.8),
    "question_target": ("positive_anchor", "span_type_question_target_anchor", 0.7),
    "negation": ("risk_positive", "span_type_negation_meaning_flip_risk", 0.8),
    "comparison": ("risk_positive", "span_type_comparison_direction_risk", 0.65),
    "cyber_security_term": ("risk_positive", "span_type_security_term_risk", 0.7),
    "operation": ("hard_negative_or_weak_positive", "span_type_operation_ambiguous", 0.5),
    "condition": ("hard_negative_or_weak_positive", "span_type_condition_ambiguous", 0.5),
    "object": ("negative", "span_type_object_recoverable", 0.6),
    "entity": ("negative", "span_type_entity_recoverable", 0.6),
    "relation": ("negative", "span_type_relation_low_value", 0.55),
}

RECOVERED_FILLER_BY_SPAN_TYPE: dict[str, str] = {
    "number": "several",
    "question_target": "what number",
    "negation": "indeed",
    "comparison": "compared",
    "cyber_security_term": "an issue",
    "operation": "changes",
    "condition": "when",
    "object": "items",
    "entity": "someone",
    "relation": "and",
}
DEFAULT_FILLER = "something"

BOUNDARY_STATEMENT = (
    "This is a weak-labeled 2000-case dry run. It does not execute attention "
    "steering. It does not validate hallucination reduction. It does not "
    "validate answer accuracy improvement."
)


def stable_choice_index(source_question_id: str, seed: int, num_candidates: int) -> int:
    """Deterministically choose a candidate index (spreads span types)."""
    digest = hashlib.sha1(f"{seed}:{source_question_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % num_candidates


def build_masked_question(question: str, span: dict[str, Any], mask_token: str) -> str:
    start = int(span["start"])
    end = int(span["end"])
    return question[:start] + mask_token + question[end:]


def build_recovered_question(masked_question: str, mask_token: str, span_type: str) -> str:
    filler = RECOVERED_FILLER_BY_SPAN_TYPE.get(span_type, DEFAULT_FILLER)
    return masked_question.replace(mask_token, filler, 1)


def weak_target_for_span(span_type: str) -> tuple[str, str, float, bool]:
    """Return (probe_target, label_rule, label_confidence, usable)."""
    mapping = WEAK_TARGET_BY_SPAN_TYPE.get(span_type)
    if mapping is None:
        return "unmapped", "no_matching_span_type_rule", 0.0, False
    probe_target, label_rule, confidence = mapping
    return probe_target, label_rule, confidence, True


def build_case(
    manifest_record: dict[str, Any],
    *,
    seed: int,
    mask_token: str,
    backend: str,
    language: str,
) -> dict[str, Any]:
    """Build the weak label and 2A manifest record for one case."""
    full_scale_id = manifest_record["full_scale_id"]
    source_question_id = manifest_record["source_question_id"]
    question = manifest_record["question"]
    answer = manifest_record["answer"]

    candidates = extract_candidate_spans(question, language=language)
    warnings: list[str] = []

    if not candidates:
        weak_label = {
            "full_scale_id": full_scale_id,
            "source_question_id": source_question_id,
            "masked_id": None,
            "unit_id": None,
            "question": question,
            "answer": answer,
            "probe_target": "unmapped",
            "probe_target_usable": False,
            "label_source": LABEL_SOURCE,
            "label_backend": backend,
            "label_rule": "no_candidate_spans",
            "label_confidence": 0.0,
            "human_reviewed": False,
            "chosen_span_type": None,
            "chosen_span_text": None,
            "warnings": ["no candidate spans extracted; case is unmapped"],
        }
        return {"weak_label": weak_label, "manifest_record": None}

    chosen_index = stable_choice_index(source_question_id, seed, len(candidates))
    chosen = candidates[chosen_index]
    span_type = chosen["type"]
    unit_id = f"unit_{chosen_index:03d}"
    masked_id = f"{full_scale_id}__{unit_id}__mask"

    masked_question = build_masked_question(question, chosen, mask_token)
    recovered_question = build_recovered_question(masked_question, mask_token, span_type)
    probe_target, label_rule, confidence, usable = weak_target_for_span(span_type)

    weak_label = {
        "full_scale_id": full_scale_id,
        "source_question_id": source_question_id,
        "masked_id": masked_id,
        "unit_id": unit_id,
        "question": question,
        "answer": answer,
        "probe_target": probe_target,
        "probe_target_usable": usable,
        "label_source": LABEL_SOURCE,
        "label_backend": backend,
        "label_rule": label_rule,
        "label_confidence": confidence,
        "human_reviewed": False,
        "chosen_span_type": span_type,
        "chosen_span_text": chosen["text"],
        "warnings": warnings,
    }

    manifest_2a_record = {
        "masked_id": masked_id,
        "id": full_scale_id,
        "unit_id": unit_id,
        "original_question": question,
        "masked_question": masked_question,
        "recovered_questions": [recovered_question],
        # Sentinel placeholders: NOT human labels. Real weak supervision lives
        # in weak_labels_2000.jsonl (label_source=weak_auto, human_reviewed=false).
        "human_recoverability_label": HUMAN_SENTINEL,
        "human_attention_anchor_label": HUMAN_SENTINEL,
        "human_semantic_role": f"weak_auto_span_type_{span_type}",
        "human_guidance_priority": "weak_auto",
        "human_error_type": "weak_auto",
        "probe_usage": "weak_auto",
    }
    return {"weak_label": weak_label, "manifest_record": manifest_2a_record}


def build_full_scale_weak_labels(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    backend: str = BACKEND,
    seed: int = 42,
    mask_token: str = DEFAULT_MASK_TOKEN,
    language: str = "en",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build weak-auto labels and the 2A-style hidden-state cache manifest."""
    if backend != BACKEND:
        raise ValueError(f"Unsupported weak-label backend {backend!r}; expected {BACKEND!r}")

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    weak_labels_path = output_dir / WEAK_LABELS_FILENAME
    manifest_2a_path = output_dir / MANIFEST_2A_FILENAME
    report_path = output_dir / REPORT_FILENAME

    from recover_attention.full_scale_manifest import ensure_output_dir_allowed

    ensure_output_dir_allowed(output_dir)
    for path in (weak_labels_path, manifest_2a_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path} (pass overwrite=True to replace)"
            )

    manifest_records = read_jsonl(manifest_path)
    if not manifest_records:
        raise ValueError(f"full-scale manifest is empty: {manifest_path}")

    weak_labels: list[dict[str, Any]] = []
    manifest_2a: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    usable_target_counts: Counter[str] = Counter()
    span_type_counts: Counter[str] = Counter()
    num_unmapped = 0

    for record in manifest_records:
        built = build_case(
            record, seed=seed, mask_token=mask_token, backend=backend, language=language
        )
        weak_label = built["weak_label"]
        weak_labels.append(weak_label)
        target_counts[weak_label["probe_target"]] += 1
        if weak_label["probe_target_usable"]:
            usable_target_counts[weak_label["probe_target"]] += 1
        if weak_label["chosen_span_type"]:
            span_type_counts[weak_label["chosen_span_type"]] += 1
        if weak_label["probe_target"] == "unmapped":
            num_unmapped += 1
        if built["manifest_record"] is not None:
            manifest_2a.append(built["manifest_record"])

    write_jsonl(weak_labels, weak_labels_path)
    write_jsonl(manifest_2a, manifest_2a_path)

    warnings: list[str] = []
    if num_unmapped:
        warnings.append(
            f"{num_unmapped} case(s) had no usable candidate span and were marked "
            "unmapped (excluded from the hidden-state cache manifest)"
        )

    report = {
        "backend": backend,
        "seed": seed,
        "mask_token": mask_token,
        "language": language,
        "inputs": {"full_scale_manifest_path": manifest_path.as_posix()},
        "outputs": {
            "weak_labels_path": weak_labels_path.as_posix(),
            "full_scale_2a_manifest_path": manifest_2a_path.as_posix(),
            "weak_label_report_path": report_path.as_posix(),
        },
        "counts": {
            "num_cases_in": len(manifest_records),
            "num_weak_labels": len(weak_labels),
            "num_manifest_cases_with_mask": len(manifest_2a),
            "num_unmapped": num_unmapped,
            "num_usable": sum(usable_target_counts.values()),
        },
        "probe_target_counts": dict(sorted(target_counts.items())),
        "usable_probe_target_counts": dict(sorted(usable_target_counts.items())),
        "chosen_span_type_counts": dict(sorted(span_type_counts.items())),
        "label_source": LABEL_SOURCE,
        "human_reviewed_full_scale": False,
        "human_field_note": (
            "human_* fields in the 2A manifest are sentinel placeholders "
            f"({HUMAN_SENTINEL!r}); they are NOT human labels. Weak supervision is "
            "in weak_labels_2000.jsonl with label_source=weak_auto."
        ),
        "weak_target_mapping_rules": {
            span_type: {
                "probe_target": mapping[0],
                "label_rule": mapping[1],
                "label_confidence": mapping[2],
            }
            for span_type, mapping in WEAK_TARGET_BY_SPAN_TYPE.items()
        },
        "boundary": BOUNDARY_STATEMENT,
        "warnings": warnings,
    }
    ensure_dir(output_dir)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "weak_labels": weak_labels,
        "manifest_2a": manifest_2a,
        "report": report,
        "output_files": {
            "weak_labels": weak_labels_path.as_posix(),
            "full_scale_2a_manifest": manifest_2a_path.as_posix(),
            "weak_label_report": report_path.as_posix(),
        },
    }
