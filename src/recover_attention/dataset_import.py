"""Import and normalize reasoning datasets (e.g. GSM8K) into the project's
internal raw question JSONL format.

The normalized record schema is::

    {
      "question_id": "gsm8k_train_000001",
      "source_dataset": "gsm8k",
      "source_split": "train",
      "question": "...",
      "answer": "...",
      "metadata": {
        "original_id": null,
        "original_answer": "...",
        "normalization_backend": "gsm8k_normalize_v0"
      }
    }

This module never fabricates, duplicates, or up-samples records. It either reads
a local raw file or downloads the genuine public dataset, then normalizes it
one-to-one. If neither a local file nor a reachable download source is
available, it raises a clear error instead of producing fake data.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, write_jsonl

DEFAULT_BACKEND = "gsm8k_normalize_v0"
DEFAULT_DATASET = "gsm8k"
DEFAULT_PREVIEW_LIMIT = 50
GSM8K_ANSWER_MARKER = "####"

TARGET_500 = 500
TARGET_2000 = 2000

# Canonical public GSM8K source (raw JSONL, no extra dependencies required).
GSM8K_SOURCE_URLS = {
    "train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/train.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    ),
}
GSM8K_SPLITS = ("train", "test")

QUESTION_KEYS = ("question", "Question", "problem", "query")
ANSWER_KEYS = ("answer", "Answer", "solution", "gold_answer")


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final answer from a GSM8K answer string.

    GSM8K answers carry the chain of thought followed by ``#### <final>``. The
    final answer is returned; the full text is preserved separately by the
    caller. If no marker is present, the cleaned full text is returned.
    """
    if not isinstance(answer_text, str):
        raise ValueError("answer_text must be a str")
    if GSM8K_ANSWER_MARKER in answer_text:
        final = answer_text.split(GSM8K_ANSWER_MARKER, 1)[1].strip()
        final = final.replace(",", "")
        return final
    return answer_text.strip()


def _pick(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key]
    return None


def normalize_record(
    raw: dict,
    index: int,
    split: str,
    dataset: str = DEFAULT_DATASET,
    backend: str = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Normalize a single raw record into the standard question record."""
    if not isinstance(raw, dict):
        raise ValueError("raw record must be a dict")

    question = _pick(raw, QUESTION_KEYS)
    if question is None:
        raise ValueError(
            f"record {index} missing a non-empty question field "
            f"(looked for {QUESTION_KEYS})"
        )
    raw_answer = _pick(raw, ANSWER_KEYS)
    if raw_answer is None:
        raise ValueError(
            f"record {index} missing a non-empty answer field "
            f"(looked for {ANSWER_KEYS})"
        )

    answer = extract_gsm8k_final_answer(raw_answer)
    if not answer.strip():
        raise ValueError(f"record {index} produced an empty normalized answer")

    original_id = raw.get("id") or raw.get("original_id")

    return {
        "question_id": f"{dataset}_{split}_{index:06d}",
        "source_dataset": dataset,
        "source_split": split,
        "question": question.strip(),
        "answer": answer.strip(),
        "metadata": {
            "original_id": original_id,
            "original_answer": raw_answer,
            "normalization_backend": backend,
        },
    }


def read_raw_records(input_path: str | Path) -> list[dict]:
    """Read raw records from a local jsonl / json / csv / parquet file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"{path} line {line_number} is not a JSON object"
                    )
                records.append(parsed)
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Allow a top-level container such as {"data": [...]}.
            for value in parsed.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        raise ValueError(f"{path} json root is neither list nor object")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "parquet input requires pyarrow; install it or provide a "
                "jsonl/json/csv file instead"
            ) from exc
        return pq.read_table(path).to_pylist()
    raise ValueError(f"unsupported input suffix: {suffix}")


def download_gsm8k(split: str, timeout: float = 30.0) -> list[dict]:
    """Download the genuine public GSM8K split as a list of raw records."""
    if split not in GSM8K_SOURCE_URLS:
        raise ValueError(
            f"unknown gsm8k split {split!r}; expected one of {GSM8K_SPLITS}"
        )
    url = GSM8K_SOURCE_URLS[split]
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - surface a clear actionable error
        raise RuntimeError(
            f"failed to download GSM8K {split} split from {url}: {exc}. "
            "If this environment has no network access, re-run with --input "
            "pointing at a locally provided GSM8K file instead."
        ) from exc

    records: list[dict] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"downloaded GSM8K {split} line {line_number} is invalid JSON: "
                f"{exc.msg}"
            ) from exc
        records.append(parsed)
    return records


def _scale_check(num_records: int) -> dict[str, Any]:
    return {
        "num_records": num_records,
        "target_num_cases_500": TARGET_500,
        "target_num_cases_2000": TARGET_2000,
        "can_run_500": num_records >= TARGET_500,
        "can_run_2000": num_records >= TARGET_2000,
        "can_run_all": num_records > 0,
    }


def normalize_records(
    raw_records: list[dict],
    split: str,
    dataset: str = DEFAULT_DATASET,
    backend: str = DEFAULT_BACKEND,
) -> tuple[list[dict], list[str]]:
    """Normalize a list of raw records one-to-one (no duplication)."""
    normalized: list[dict] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_records, start=1):
        record = normalize_record(raw, index, split, dataset=dataset, backend=backend)
        question_id = record["question_id"]
        if question_id in seen_ids:
            raise ValueError(f"duplicate question_id generated: {question_id}")
        seen_ids.add(question_id)
        normalized.append(record)
    return normalized, warnings


def import_reasoning_dataset(
    output_path: str | Path,
    report_output_path: str | Path,
    dataset: str = DEFAULT_DATASET,
    split: str = "train",
    input_path: str | Path | None = None,
    backend: str = DEFAULT_BACKEND,
    preview_output_path: str | Path | None = None,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a reasoning dataset, normalize it, and write outputs + report.

    Either reads ``input_path`` (offline, deterministic) or downloads the
    genuine public dataset when ``input_path`` is None. Never fabricates data.
    """
    output_path = Path(output_path)
    report_output_path = Path(report_output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} (pass overwrite=True to replace)"
        )

    if input_path is not None:
        source_mode = "local_file"
        source_ref = Path(input_path).as_posix()
        raw_records = read_raw_records(input_path)
    else:
        if dataset != "gsm8k":
            raise ValueError(
                f"download is only supported for dataset 'gsm8k'; got {dataset!r}. "
                "Provide --input for other datasets."
            )
        source_mode = "download"
        source_ref = GSM8K_SOURCE_URLS.get(split, "")
        raw_records = download_gsm8k(split)

    num_raw = len(raw_records)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        raw_records = raw_records[:limit]

    normalized, warnings = normalize_records(
        raw_records, split, dataset=dataset, backend=backend
    )
    if limit is not None:
        warnings.append(
            f"limit={limit} applied: imported a smoke-test subset of the source, "
            "not the full dataset"
        )

    write_jsonl(normalized, output_path)

    preview_path_str = None
    if preview_output_path is not None:
        preview_path = Path(preview_output_path)
        write_jsonl(normalized[: max(preview_limit, 0)], preview_path)
        preview_path_str = preview_path.as_posix()

    report = {
        "backend": backend,
        "dataset": dataset,
        "split": split,
        "source_mode": source_mode,
        "source_url_or_path": source_ref,
        "num_raw_records": num_raw,
        "num_normalized_records": len(normalized),
        "num_skipped": num_raw - len(raw_records) if limit is None else None,
        "output_path": output_path.as_posix(),
        "preview_path": preview_path_str,
        "fields_per_record": [
            "question_id",
            "source_dataset",
            "source_split",
            "question",
            "answer",
            "metadata",
        ],
        "answer_extraction": (
            "split on '####' marker; the final answer is stored in 'answer' and "
            "the full chain-of-thought is preserved in metadata.original_answer"
        ),
        "scale_check": _scale_check(len(normalized)),
        "duplication_check": {
            "duplicated_or_upsampled": False,
            "note": "one normalized record per source record; question_ids are unique",
        },
        "warnings": warnings,
    }

    ensure_dir(report_output_path.parent)
    with report_output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return report
