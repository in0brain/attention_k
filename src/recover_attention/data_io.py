"""Minimal JSONL file I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSONL file into a list of dictionaries."""
    jsonl_path = Path(path)
    records: list[dict] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                ) from exc

    return records


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write dictionaries to a UTF-8 JSONL file, one JSON object per line."""
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
