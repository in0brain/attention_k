"""Minimal UTF-8 JSON, JSONL, and text file I/O utilities."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable


def file_sha256(path: str | Path) -> str:
    """Hash a source artifact without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_plot_pair(figure: Any, output_dir: str | Path, stem: str) -> list[Path]:
    """Save a vector PDF and PNG preview; never overwrite existing files."""
    directory = Path(output_dir)
    if Path(stem).name != stem or stem in {"", ".", ".."}:
        raise ValueError("figure stem must be a plain filename")
    targets = [directory / f"{stem}.{ext}" for ext in ("pdf", "png")]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"figure outputs already exist for {directory / stem}")
    ensure_dir(directory)
    for path in targets:
        # Exclusive open also protects against another writer after the check.
        with path.open("xb") as handle:
            figure.savefig(handle, format=path.suffix[1:], dpi=180, bbox_inches="tight")
    return targets


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


def read_json(path: str | Path) -> Any:
    """Read and decode a UTF-8 JSON file."""
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc.msg}") from exc


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write dictionaries to a UTF-8 JSONL file, one JSON object per line."""
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data: Any, path: str | Path) -> None:
    """Write a JSON-compatible value as indented UTF-8 JSON."""
    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(text: str, path: str | Path) -> None:
    """Write UTF-8 text, creating the parent directory when needed."""
    text_path = Path(path)
    ensure_dir(text_path.parent)
    text_path.write_text(text, encoding="utf-8")
