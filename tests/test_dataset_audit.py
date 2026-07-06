from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import write_jsonl
from recover_attention.dataset_audit import (
    audit_dataset_sources,
    audit_file,
    build_kfold_precondition,
    render_audit_markdown,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, path)


def test_audit_file_classifies_raw_question_source(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "q1", "dataset": "gsm8k", "question": "Q1?", "gold_answer": "1"},
            {"id": "q2", "dataset": "gsm8k", "question": "Q2?", "gold_answer": "2"},
        ],
    )

    descriptor = audit_file(path, path)

    assert descriptor["candidate_source_type"] == "raw_question_source"
    assert descriptor["usable_for_full_scale"] is True
    assert descriptor["has_question_field"] is True
    assert descriptor["has_answer_field"] is True
    assert descriptor["num_records"] == 2
    assert descriptor["reason_if_not_usable"] is None


def test_audit_file_classifies_derived_downstream_with_fanout(tmp_path: Path) -> None:
    path = tmp_path / "semantic_labels.jsonl"
    # Three rows fanning out over only two distinct source ids.
    _write_jsonl(
        path,
        [
            {"id": "q1", "nli_id": "n1", "original_question": "Q1?", "span_ids": ["s1"]},
            {"id": "q1", "nli_id": "n2", "original_question": "Q1?", "span_ids": ["s2"]},
            {"id": "q2", "nli_id": "n3", "original_question": "Q2?", "span_ids": ["s3"]},
        ],
    )

    descriptor = audit_file(path, path)

    assert descriptor["candidate_source_type"] == "derived_downstream"
    assert descriptor["usable_for_full_scale"] is False
    assert descriptor["num_records"] == 3
    assert descriptor["num_distinct_source_ids"] == 2
    assert "fan out" in descriptor["reason_if_not_usable"]


def test_audit_dataset_sources_reports_scale_and_shortfall(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_jsonl(
        data_root / "processed" / "questions.jsonl",
        [
            {"id": f"q{i}", "dataset": "gsm8k", "question": f"Q{i}?", "gold_answer": "1"}
            for i in range(5)
        ],
    )
    _write_jsonl(
        data_root / "processed" / "semantic_labels.jsonl",
        [
            {"id": "q0", "nli_id": f"n{i}", "original_question": "Q0?", "span_ids": ["s"]}
            for i in range(92)
        ],
    )

    report = audit_dataset_sources([data_root])

    assert report["available_num_cases"] == 5
    assert report["max_records_any_file"] == 92
    assert report["can_run_500"] is False
    assert report["can_run_2000"] is False
    assert report["can_run_all"] is True
    assert report["shortfall_reason"] is not None
    assert "92" in report["fan_out_explanation"]
    # The 92-row file must not be counted as a usable source.
    assert all(
        source["num_records"] != 92 for source in report["raw_question_sources"]
    )


def test_audit_dataset_sources_detects_full_scale_source(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_jsonl(
        data_root / "raw" / "gsm8k_train_normalized.jsonl",
        [
            {
                "question_id": f"gsm8k_train_{i:06d}",
                "source_dataset": "gsm8k",
                "source_split": "train",
                "question": f"Q{i}?",
                "answer": "1",
                "metadata": {},
            }
            for i in range(2500)
        ],
    )

    report = audit_dataset_sources([data_root])

    assert report["available_num_cases"] == 2500
    assert report["can_run_500"] is True
    assert report["can_run_2000"] is True
    assert report["recommended_source_path_500"].endswith(
        "gsm8k_train_normalized.jsonl"
    )
    assert report["recommended_source_path_2000"].endswith(
        "gsm8k_train_normalized.jsonl"
    )
    assert report["shortfall_reason"] is None


def test_kfold_precondition_not_decided_now() -> None:
    precondition = build_kfold_precondition(2500)

    assert precondition["source_scale_check"]["can_run_500"] is True
    assert precondition["source_scale_check"]["can_run_2000"] is True
    future = precondition["future_kfold_requirement"]
    assert future["decided_now"] is False
    assert "min_class_count" in future["rule"]


def test_render_audit_markdown_contains_sections(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_jsonl(
        data_root / "processed" / "questions.jsonl",
        [{"id": "q1", "dataset": "gsm8k", "question": "Q?", "gold_answer": "1"}],
    )
    report = audit_dataset_sources([data_root])

    markdown = render_audit_markdown(report)

    assert "# Dataset Source Audit" in markdown
    assert "Scale Feasibility" in markdown
    assert "K-fold Feasibility Precondition" in markdown


def test_audit_handles_missing_root(tmp_path: Path) -> None:
    report = audit_dataset_sources([tmp_path / "does_not_exist"])

    assert report["num_files_scanned"] == 0
    assert report["available_num_cases"] == 0
    assert report["can_run_500"] is False
