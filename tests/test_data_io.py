from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_json, read_jsonl, write_json, write_jsonl, write_text


def test_write_jsonl_then_read_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [
        {"id": "q1", "question": "How many apples?", "gold_answer": "3"},
        {"id": "q2", "question": "How many pencils?", "gold_answer": "5"},
    ]
    path = tmp_path / "nested" / "records.jsonl"

    write_jsonl(records, path)

    assert read_jsonl(path) == records


def test_read_jsonl_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_jsonl(path) == []


def test_read_jsonl_invalid_json_raises_with_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"id": "ok"}\n{"id": bad}\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "line 2" in message


def test_write_jsonl_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "new" / "parent" / "records.jsonl"

    write_jsonl([{"id": "q1"}], path)

    assert path.exists()
    assert read_jsonl(path) == [{"id": "q1"}]


def test_write_json_then_read_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    payload = {"count": 2, "labels": ["A", "B"]}
    write_json(payload, path)
    assert read_json(path) == payload


def test_write_text_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.md"
    write_text("# Report\n", path)
    assert path.read_text(encoding="utf-8") == "# Report\n"
