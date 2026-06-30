from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.human_review_consolidation import (
    HUMAN_REVIEW_FIELDS,
    build_human_review_summary,
    build_known_issues_markdown,
    build_sprint_2a_manifest,
    run_human_review_consolidation,
    validate_human_review_records,
)


def review_record(
    *,
    masked_id: str = "q1__unit_001__mask",
    question_id: str = "q1",
    unit_id: str = "unit_001",
    upgraded_recoverability_label: str = "Recoverable",
    upgraded_attention_anchor_label: str = "Distractor",
    human_recoverability_label: str = "Recoverable",
    human_attention_anchor_label: str = "Distractor",
    human_error_type: str = "semantic_equivalent_recovery",
    probe_usage: str = "hard_negative_or_weak_positive",
) -> dict:
    return {
        "masked_id": masked_id,
        "id": question_id,
        "unit_id": unit_id,
        "original_question": "Tom has 3 apples and buys 2 more.",
        "masked_question": "Tom has [MASK] apples and buys 2 more.",
        "recovered_questions": ["Tom has 3 apples and buys 2 more."],
        "baseline_recoverability_label": "Misleading Recovery",
        "upgraded_recoverability_label": upgraded_recoverability_label,
        "baseline_attention_anchor_label": "Risky Anchor",
        "upgraded_attention_anchor_label": upgraded_attention_anchor_label,
        "baseline_attention_importance_score": 0.7,
        "upgraded_attention_importance_score": 0.4,
        "attention_importance_score_delta": -0.3,
        "baseline_recoverability_score": 0.0,
        "upgraded_recoverability_score": 1.0,
        "recoverability_score_delta": 1.0,
        "human_review_status": "reviewed",
        "human_recovered_is_full_question": True,
        "human_masked_info_recovered": "yes",
        "human_wrong_information_introduced": False,
        "human_answer_changed": False,
        "human_masked_span_is_key": "partial",
        "human_recoverability_label": human_recoverability_label,
        "human_attention_anchor_label": human_attention_anchor_label,
        "human_semantic_role": "critical_number",
        "human_guidance_priority": "high",
        "human_error_type": human_error_type,
        "probe_usage": probe_usage,
        "human_notes": "reviewed fixture",
        "reviewer": "tester",
        "review_date": "2026-06-28",
    }


def write_inputs(tmp_path: Path, records: list[dict]) -> dict[str, Path]:
    review_guide = tmp_path / "sprint_1Q_human_review_guide.md"
    labels_jsonl = tmp_path / "sprint_1Q_human_review_labels_template.jsonl"
    report_json = tmp_path / "upgraded_downstream_report_with_human_fields.json"
    review_guide.write_text("# Review Guide\n", encoding="utf-8")
    write_jsonl(records, labels_jsonl)
    changed_cases = []
    for record in records:
        case = {
            "masked_id": record["masked_id"],
            "id": record["id"],
            "unit_id": record["unit_id"],
        }
        for field in HUMAN_REVIEW_FIELDS:
            case[field] = record[field]
        changed_cases.append(case)
    report = {
        "changed_cases": changed_cases,
        "human_review_metadata": {"created_from": "fixture"},
    }
    report_json.write_text(json.dumps(report), encoding="utf-8")
    return {
        "review_guide": review_guide,
        "labels_jsonl": labels_jsonl,
        "report_json": report_json,
        "summary_json": tmp_path / "sprint_1Q_human_review_summary.json",
        "known_issues_md": tmp_path / "sprint_1Q_known_issues.md",
        "manifest_jsonl": tmp_path / "sprint_1Q_to_2A_manifest.jsonl",
    }


def test_run_consolidation_writes_outputs_without_rewriting_inputs(tmp_path: Path) -> None:
    records = [
        review_record(),
        review_record(
            masked_id="q2__unit_002__mask",
            question_id="q2",
            unit_id="unit_002",
            upgraded_recoverability_label="Recoverable",
            upgraded_attention_anchor_label="Distractor",
            human_recoverability_label="Misleading Recovery",
            human_attention_anchor_label="Risky Anchor",
            human_error_type="wrong_numeric_recovery",
            probe_usage="risk_positive",
        ),
    ]
    paths = write_inputs(tmp_path, records)
    labels_before = paths["labels_jsonl"].read_text(encoding="utf-8")
    report_before = paths["report_json"].read_text(encoding="utf-8")

    result = run_human_review_consolidation(**paths)

    assert result["reviewed_count"] == 2
    assert result["unreviewed_count"] == 0
    assert result["manifest_count"] == 2
    assert result["validation_warning_count"] == 0
    assert paths["summary_json"].exists()
    assert paths["known_issues_md"].exists()
    assert paths["manifest_jsonl"].exists()
    assert paths["labels_jsonl"].read_text(encoding="utf-8") == labels_before
    assert paths["report_json"].read_text(encoding="utf-8") == report_before

    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    second_case = report["changed_cases"][1]
    for field in HUMAN_REVIEW_FIELDS:
        assert second_case[field] == records[1][field]


def test_run_consolidation_warns_without_overwriting_report_mismatches(
    tmp_path: Path,
) -> None:
    records = [review_record()]
    paths = write_inputs(tmp_path, records)
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    report["changed_cases"][0]["human_recoverability_label"] = "Non-recoverable"
    report["changed_cases"][0]["human_notes"] = "stale report value"
    paths["report_json"].write_text(json.dumps(report), encoding="utf-8")
    labels_before = paths["labels_jsonl"].read_text(encoding="utf-8")
    report_before = paths["report_json"].read_text(encoding="utf-8")

    result = run_human_review_consolidation(**paths)

    assert paths["labels_jsonl"].read_text(encoding="utf-8") == labels_before
    assert paths["report_json"].read_text(encoding="utf-8") == report_before
    assert result["validation_warning_count"] == 2
    assert any("human_recoverability_label" in warning for warning in result["validation_warnings"])
    assert any("human_notes" in warning for warning in result["validation_warnings"])


def test_summary_counts_required_sprint_1r_fields() -> None:
    records = [
        review_record(),
        review_record(
            masked_id="q2__unit_002__mask",
            question_id="q2",
            unit_id="unit_002",
            upgraded_recoverability_label="Recoverable",
            upgraded_attention_anchor_label="Distractor",
            human_recoverability_label="Non-recoverable",
            human_attention_anchor_label="Strong Anchor",
            human_error_type="fragment_recovery",
            probe_usage="negative",
        ),
        review_record(
            masked_id="q3__unit_003__mask",
            question_id="q3",
            unit_id="unit_003",
            upgraded_recoverability_label="Misleading Recovery",
            upgraded_attention_anchor_label="Risky Anchor",
            human_recoverability_label="Misleading Recovery",
            human_attention_anchor_label="Risky Anchor",
            human_error_type="misleading_entity_or_unit",
            probe_usage="positive_anchor",
        ),
    ]

    summary = build_human_review_summary(records)

    assert summary["reviewed_count"] == 3
    assert summary["unreviewed_count"] == 0
    assert summary["auto_vs_human_recoverability_disagreement_count"] == 1
    assert summary["auto_vs_human_attention_anchor_disagreement_count"] == 1
    assert summary["fragment_recovery_count"] == 1
    assert summary["wrong_numeric_recovery_count"] == 0
    assert summary["generic_recovery_count"] == 0
    assert summary["misleading_entity_or_unit_count"] == 1
    assert summary["probe_usage_counts"]["negative"] == 1


def test_manifest_contains_only_reviewed_cases_and_minimal_fields() -> None:
    reviewed = review_record()
    unreviewed = deepcopy(reviewed)
    unreviewed["masked_id"] = "q2__unit_002__mask"
    unreviewed["id"] = "q2"
    unreviewed["unit_id"] = "unit_002"
    unreviewed["human_review_status"] = "todo"

    manifest = build_sprint_2a_manifest([reviewed, unreviewed])

    assert len(manifest) == 1
    assert set(manifest[0]) == {
        "masked_id",
        "id",
        "unit_id",
        "original_question",
        "masked_question",
        "recovered_questions",
        "human_recoverability_label",
        "human_attention_anchor_label",
        "human_semantic_role",
        "human_guidance_priority",
        "human_error_type",
        "probe_usage",
    }


def test_known_issues_document_freezes_deferred_scope() -> None:
    known_issues = build_known_issues_markdown()

    assert "Full-Question Recovery Validation" in known_issues
    assert "Span-Aware Numeric Recovery Check" in known_issues
    assert "Unit / Group Mask Budget Control" in known_issues
    assert "No hidden-state cache" in known_issues


def test_missing_human_field_raises() -> None:
    record = review_record()
    del record["human_error_type"]

    with pytest.raises(ValueError, match="missing human field: human_error_type"):
        validate_human_review_records([record])


def test_duplicate_masked_id_raises() -> None:
    record = review_record()

    with pytest.raises(ValueError, match="duplicate masked_id"):
        validate_human_review_records([record, deepcopy(record)])


def test_cli_builds_consolidated_outputs(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, [review_record()])

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "15_consolidate_human_review.py"),
            "--review-guide",
            str(paths["review_guide"]),
            "--labels-jsonl",
            str(paths["labels_jsonl"]),
            "--report-json",
            str(paths["report_json"]),
            "--summary-json",
            str(paths["summary_json"]),
            "--known-issues-md",
            str(paths["known_issues_md"]),
            "--manifest-jsonl",
            str(paths["manifest_jsonl"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Built Sprint 1R human review artifacts" in result.stdout
    assert "reviewed_count: 1" in result.stdout
    assert len(read_jsonl(paths["manifest_jsonl"])) == 1
