"""Build unit-level recovery output records from masked question records."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_masked_question_record,
    validate_recover_output_record,
)


DEFAULT_RECOVERY_BACKEND = "oracle_stub_v0"
SUPPORTED_RECOVERY_BACKENDS = {"oracle_stub_v0"}
DEFAULT_NUM_SAMPLES = 1

RECOVER_OUTPUT_SOURCE_FIELDS = (
    "masked_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "masked_question",
    "mask_token",
    "mask_backend",
    "mask_strategy",
)


def build_recovered_question_oracle_stub(masked_question_record: dict) -> str:
    """Return the original question for pipeline verification."""
    return masked_question_record["original_question"]


def build_recover_output_record(
    masked_question_record: dict,
    sample_id: int,
    backend: str = DEFAULT_RECOVERY_BACKEND,
) -> dict:
    """Build one validated recovery output record from one masked question."""
    validate_masked_question_record(masked_question_record)
    _validate_backend(backend)
    _validate_sample_id(sample_id)

    record = {
        field: deepcopy(masked_question_record[field])
        for field in RECOVER_OUTPUT_SOURCE_FIELDS
    }
    record.update(
        {
            "recovered_question": _build_recovered_question(
                masked_question_record,
                backend,
            ),
            "recovery_backend": backend,
            "sample_id": sample_id,
        }
    )
    validate_recover_output_record(record)
    return record


def build_recover_output_records(
    masked_question_records: list[dict],
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[list[dict], dict]:
    """Build recovery output records and return records plus summary stats."""
    _validate_backend(backend)
    _validate_num_samples(num_samples)

    for masked_question_record in masked_question_records:
        validate_masked_question_record(masked_question_record)

    stats = _empty_stats(masked_question_records, backend, num_samples)
    output_records: list[dict] = []

    for masked_question_record in masked_question_records:
        for sample_id in range(num_samples):
            output_records.append(
                build_recover_output_record(
                    masked_question_record,
                    sample_id=sample_id,
                    backend=backend,
                )
            )

    stats["num_output_recoveries"] = len(output_records)
    stats["unit_scope_counts"] = dict(sorted(stats["unit_scope_counts"].items()))
    stats["group_type_counts"] = dict(sorted(stats["group_type_counts"].items()))
    stats["mask_backend_counts"] = dict(sorted(stats["mask_backend_counts"].items()))
    stats["mask_strategy_counts"] = dict(sorted(stats["mask_strategy_counts"].items()))
    return output_records, stats


def build_recover_output_file(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[list[dict], dict]:
    """Read masked questions, build recovery outputs, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing input: {input_jsonl}\nPlease run Sprint 1F first."
        )

    masked_question_records = read_jsonl(input_jsonl)
    records, stats = build_recover_output_records(
        masked_question_records,
        backend=backend,
        num_samples=num_samples,
    )
    write_jsonl(records, output_path)
    return records, stats


def _build_recovered_question(masked_question_record: dict, backend: str) -> str:
    if backend == "oracle_stub_v0":
        return build_recovered_question_oracle_stub(masked_question_record)
    _validate_backend(backend)
    raise AssertionError("unreachable backend validation state")


def _empty_stats(
    masked_question_records: list[dict],
    backend: str,
    num_samples: int,
) -> dict:
    return {
        "num_input_masks": len(masked_question_records),
        "num_output_recoveries": 0,
        "num_samples": num_samples,
        "recovery_backend": backend,
        "unit_scope_counts": Counter(
            record["unit_scope"] for record in masked_question_records
        ),
        "group_type_counts": Counter(
            record["group_type"] for record in masked_question_records
        ),
        "mask_backend_counts": Counter(
            record["mask_backend"] for record in masked_question_records
        ),
        "mask_strategy_counts": Counter(
            record["mask_strategy"] for record in masked_question_records
        ),
    }


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_RECOVERY_BACKENDS:
        raise ValueError(f"Unsupported recovery backend: {backend}")


def _validate_num_samples(num_samples: int) -> None:
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError("num_samples must be an int")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")


def _validate_sample_id(sample_id: int) -> None:
    if not isinstance(sample_id, int) or isinstance(sample_id, bool):
        raise ValueError("sample_id must be an int")
    if sample_id < 0:
        raise ValueError("sample_id must be >= 0")
