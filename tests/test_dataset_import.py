from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.dataset_import import (
    extract_gsm8k_final_answer,
    import_reasoning_dataset,
    normalize_record,
    normalize_records,
    read_raw_records,
)


def _gsm8k_raw(question: str = "Tom has 3 apples and buys 2 more. How many?",
               answer: str = "Tom adds 3 + 2 = <<3+2=5>>5.\n#### 5") -> dict:
    return {"question": question, "answer": answer}


def test_extract_gsm8k_final_answer_uses_marker() -> None:
    assert extract_gsm8k_final_answer("step one\nstep two\n#### 18") == "18"


def test_extract_gsm8k_final_answer_strips_commas() -> None:
    assert extract_gsm8k_final_answer("a\n#### 1,200") == "1200"


def test_extract_gsm8k_final_answer_without_marker_returns_cleaned() -> None:
    assert extract_gsm8k_final_answer("  42  ") == "42"


def test_normalize_record_builds_standard_schema() -> None:
    record = normalize_record(_gsm8k_raw(), index=1, split="train")

    assert record["question_id"] == "gsm8k_train_000001"
    assert record["source_dataset"] == "gsm8k"
    assert record["source_split"] == "train"
    assert record["answer"] == "5"
    assert record["question"].startswith("Tom has 3 apples")
    assert record["metadata"]["normalization_backend"] == "gsm8k_normalize_v0"
    assert record["metadata"]["original_answer"].endswith("#### 5")
    assert set(record.keys()) == {
        "question_id",
        "source_dataset",
        "source_split",
        "question",
        "answer",
        "metadata",
    }


def test_normalize_record_missing_question_raises() -> None:
    with pytest.raises(ValueError, match="missing a non-empty question"):
        normalize_record({"answer": "1\n#### 1"}, index=1, split="train")


def test_normalize_records_generates_unique_ids() -> None:
    raw = [_gsm8k_raw(question=f"Q{i}?", answer=f"a\n#### {i}") for i in range(3)]
    normalized, warnings = normalize_records(raw, split="train")

    ids = [record["question_id"] for record in normalized]
    assert ids == ["gsm8k_train_000001", "gsm8k_train_000002", "gsm8k_train_000003"]
    assert len(set(ids)) == 3
    assert warnings == []


def test_read_raw_records_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    write_jsonl([_gsm8k_raw(), _gsm8k_raw(question="Q2?")], path)

    records = read_raw_records(path)

    assert len(records) == 2
    assert records[0]["question"].startswith("Tom has")


def test_import_reasoning_dataset_from_local_file(tmp_path: Path) -> None:
    input_path = tmp_path / "gsm8k_train.jsonl"
    write_jsonl(
        [_gsm8k_raw(question=f"Question {i}?", answer=f"work\n#### {i}") for i in range(4)],
        input_path,
    )
    output_path = tmp_path / "out" / "gsm8k_train_normalized.jsonl"
    report_path = tmp_path / "out" / "normalized_dataset_report.json"
    preview_path = tmp_path / "out" / "preview.jsonl"

    report = import_reasoning_dataset(
        output_path=output_path,
        report_output_path=report_path,
        dataset="gsm8k",
        split="train",
        input_path=input_path,
        preview_output_path=preview_path,
        preview_limit=2,
    )

    written = read_jsonl(output_path)
    assert len(written) == 4
    assert report["source_mode"] == "local_file"
    assert report["num_normalized_records"] == 4
    assert report["duplication_check"]["duplicated_or_upsampled"] is False
    assert report_path.exists()

    preview = read_jsonl(preview_path)
    assert len(preview) == 2

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["scale_check"]["num_records"] == 4
    assert saved_report["scale_check"]["can_run_500"] is False


def test_import_reasoning_dataset_scale_flags_large(tmp_path: Path) -> None:
    input_path = tmp_path / "big.jsonl"
    write_jsonl(
        [_gsm8k_raw(question=f"Q{i}?", answer=f"w\n#### {i}") for i in range(2100)],
        input_path,
    )
    output_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"

    report = import_reasoning_dataset(
        output_path=output_path,
        report_output_path=report_path,
        input_path=input_path,
    )

    assert report["scale_check"]["can_run_500"] is True
    assert report["scale_check"]["can_run_2000"] is True


def test_import_reasoning_dataset_refuses_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    write_jsonl([_gsm8k_raw()], input_path)
    output_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        import_reasoning_dataset(
            output_path=output_path,
            report_output_path=report_path,
            input_path=input_path,
            overwrite=False,
        )


def test_import_reasoning_dataset_limit_marks_warning(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    write_jsonl(
        [_gsm8k_raw(question=f"Q{i}?", answer=f"w\n#### {i}") for i in range(10)],
        input_path,
    )
    output_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"

    report = import_reasoning_dataset(
        output_path=output_path,
        report_output_path=report_path,
        input_path=input_path,
        limit=3,
    )

    assert report["num_normalized_records"] == 3
    assert any("limit=3" in warning for warning in report["warnings"])
