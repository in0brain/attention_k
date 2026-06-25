from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.recover_scoring import (
    DEFAULT_SCORE_BACKEND,
    build_recover_score_file,
    build_recover_score_record,
    build_recover_score_records,
    normalize_question,
)
from recover_attention.schemas import REQUIRED_FIELDS, validate_recover_score_record


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
FORBIDDEN_SCORE_FIELDS = {
    "span_id",
    "span_text",
    "span_type",
    "sample_id",
    "recovered_question",
    "recoverable",
    "confidence",
    "reason",
    "attention_anchor_label",
    "guidance_action",
}


def recover_output_record(
    *,
    sample_id: int = 0,
    recovered_question: str = ORIGINAL_QUESTION,
    masked_id: str = "q1__unit_001__mask",
    question_id: str = "q1",
    unit_id: str = "unit_001",
    original_question: str = ORIGINAL_QUESTION,
) -> dict:
    return {
        "masked_id": masked_id,
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": original_question,
        "masked_question": "Tom has [MASK] apples and buys 2 more.",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovered_question": recovered_question,
        "recovery_backend": "oracle_stub_v0",
        "sample_id": sample_id,
    }


def test_normalize_question_strips_and_collapses_whitespace() -> None:
    assert normalize_question("  Tom   has 3\napples.  ") == "Tom has 3 apples."


def test_single_oracle_exact_recovery_is_recoverable() -> None:
    record = build_recover_score_record([recover_output_record()])

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert record["confidence_mean"] == 1.0
    assert record["recovery_consistency"] == 1.0
    assert record["misleading_recovery"] is False
    assert record["source_sample_ids"] == [0]
    assert record["recovered_questions"] == [ORIGINAL_QUESTION]
    assert record["evidence"]["num_exact_matches"] == 1
    assert validate_recover_score_record(record) is None


def test_multi_sample_all_exact_is_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=1, recovered_question=f"  {ORIGINAL_QUESTION}  "),
            recover_output_record(sample_id=0),
        ]
    )

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert record["source_sample_ids"] == [0, 1]


def test_multi_sample_partial_exact_and_empty_is_partially_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0),
            recover_output_record(sample_id=1, recovered_question="   "),
        ]
    )

    assert record["recoverability_label"] == "Partially Recoverable"
    assert record["recoverability_score"] == 0.5
    assert record["confidence_mean"] == 0.5
    assert record["misleading_recovery"] is False
    assert record["evidence"]["num_empty_recoveries"] == 1


def test_all_empty_is_non_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0, recovered_question=""),
            recover_output_record(sample_id=1, recovered_question="   "),
        ]
    )

    assert record["recoverability_label"] == "Non-recoverable"
    assert record["recoverability_score"] == 0.0
    assert record["recovery_consistency"] == 1.0
    assert record["misleading_recovery"] is False


def test_non_empty_mismatch_is_misleading_recovery() -> None:
    record = build_recover_score_record(
        [recover_output_record(recovered_question="Tom has 4 apples and buys 2 more.")]
    )

    assert record["recoverability_label"] == "Misleading Recovery"
    assert record["recoverability_score"] == 0.0
    assert record["misleading_recovery"] is True
    assert record["evidence"]["num_non_empty_mismatches"] == 1


def test_sample_id_sorting_controls_source_order() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=2, recovered_question="wrong recovery"),
            recover_output_record(sample_id=0, recovered_question=ORIGINAL_QUESTION),
            recover_output_record(sample_id=1, recovered_question=""),
        ]
    )

    assert record["source_sample_ids"] == [0, 1, 2]
    assert record["recovered_questions"] == [ORIGINAL_QUESTION, "", "wrong recovery"]


def test_duplicate_sample_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_recover_score_record(
            [
                recover_output_record(sample_id=0),
                recover_output_record(sample_id=0),
            ]
        )


def test_metadata_mismatch_within_masked_id_raises_value_error() -> None:
    second = recover_output_record(sample_id=1)
    second["original_question"] = "Tom has 4 apples and buys 2 more."

    with pytest.raises(ValueError, match="consistent original_question"):
        build_recover_score_record([recover_output_record(sample_id=0), second])


def test_unsupported_score_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported score backend: manual"):
        build_recover_score_record([recover_output_record()], score_backend="manual")


def test_output_record_has_required_fields_and_no_forbidden_fields() -> None:
    record = build_recover_score_record([recover_output_record()])

    assert set(record) == set(REQUIRED_FIELDS["recover_score"])
    assert FORBIDDEN_SCORE_FIELDS.isdisjoint(record)
    assert validate_recover_score_record(record) is None


def test_build_recover_score_records_groups_by_masked_id_and_returns_stats() -> None:
    records, stats = build_recover_score_records(
        [
            recover_output_record(masked_id="q1__unit_001__mask", unit_id="unit_001"),
            recover_output_record(masked_id="q2__unit_002__mask", question_id="q2", unit_id="unit_002"),
        ]
    )

    assert len(records) == 2
    assert {record["masked_id"] for record in records} == {
        "q1__unit_001__mask",
        "q2__unit_002__mask",
    }
    assert stats["num_input_recoveries"] == 2
    assert stats["num_output_scores"] == 2
    assert stats["score_backend"] == DEFAULT_SCORE_BACKEND
    assert stats["recoverability_label_counts"] == {"Recoverable": 2}


def test_build_recover_score_file_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "recover_outputs.jsonl"
    output_path = tmp_path / "recover_scores.jsonl"
    write_jsonl([recover_output_record()], input_path)

    records, stats = build_recover_score_file(input_path, output_path)
    read_back = read_jsonl(output_path)

    assert output_path.exists()
    assert records == read_back
    assert len(read_back) == 1
    assert stats["num_input_recoveries"] == 1
    assert validate_recover_score_record(read_back[0]) is None


def test_build_recover_score_file_missing_input_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_recover_outputs.jsonl"

    with pytest.raises(FileNotFoundError, match="Please run Sprint 1G first"):
        build_recover_score_file(missing_path, tmp_path / "recover_scores.jsonl")


def test_cli_smoke_test_builds_recover_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "recover_outputs.jsonl"
    output_path = tmp_path / "recover_scores.jsonl"
    write_jsonl([recover_output_record()], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "09_score_recovery.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            DEFAULT_SCORE_BACKEND,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built recover scores" in result.stdout
    assert "num_input_recoveries: 1" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["recoverability_label"] == "Recoverable"
    assert validate_recover_score_record(records[0]) is None
