"""Sprint 2G full-scale manifest construction.

Samples a fixed number of cases from a normalized question source (e.g.
``data/raw/gsm8k_train_normalized.jsonl``) to seed the full-scale weak-labeled
dry-run pipeline. Sampling is deterministic: either the first N records or a
seeded sample. No record is duplicated or up-sampled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl

BACKEND = "full_scale_manifest_v0"
MANIFEST_FILENAME = "full_scale_manifest.jsonl"
REPORT_FILENAME = "full_scale_manifest_report.json"
DEFAULT_ID_PREFIX = "fs2000"
SAMPLING_RULES = ("seeded_sample", "first_n")

BOUNDARY_STATEMENT = (
    "This is a weak-labeled 2000-case dry run. It does not execute attention "
    "steering. It does not validate hallucination reduction. It does not "
    "validate answer accuracy improvement."
)


def select_indices(
    available: int,
    requested: int,
    sampling_rule: str,
    seed: int,
) -> list[int]:
    """Pick deterministic source indices without duplication."""
    if requested < 1:
        raise ValueError("requested_num_cases must be >= 1")
    if sampling_rule not in SAMPLING_RULES:
        raise ValueError(
            f"unknown sampling_rule {sampling_rule!r}; expected one of {SAMPLING_RULES}"
        )
    take = min(requested, available)
    if sampling_rule == "first_n":
        return list(range(take))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(available)
    return sorted(int(index) for index in permutation[:take])


def build_full_scale_manifest(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    requested_num_cases: int,
    sampling_rule: str = "seeded_sample",
    seed: int = 42,
    id_prefix: str = DEFAULT_ID_PREFIX,
    backend: str = BACKEND,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a full-scale manifest by sampling the normalized source dataset."""
    if backend != BACKEND:
        raise ValueError(f"Unsupported manifest backend {backend!r}; expected {BACKEND!r}")

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    report_path = output_dir / REPORT_FILENAME
    ensure_output_dir_allowed(output_dir)
    for path in (manifest_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path} (pass overwrite=True to replace)"
            )

    source_records = read_jsonl(source_path)
    available = len(source_records)
    if available == 0:
        raise ValueError(f"source dataset is empty: {source_path}")

    indices = select_indices(available, requested_num_cases, sampling_rule, seed)
    actual = len(indices)

    manifest_records: list[dict[str, Any]] = []
    seen_full_scale_ids: set[str] = set()
    for ordinal, source_index in enumerate(indices, start=1):
        source = source_records[source_index]
        full_scale_id = f"{id_prefix}_{ordinal:06d}"
        if full_scale_id in seen_full_scale_ids:
            raise ValueError(f"duplicate full_scale_id generated: {full_scale_id}")
        seen_full_scale_ids.add(full_scale_id)
        manifest_records.append(
            {
                "full_scale_id": full_scale_id,
                "source_question_id": source.get("question_id"),
                "source_dataset": source.get("source_dataset"),
                "source_split": source.get("source_split"),
                "question": source.get("question"),
                "answer": source.get("answer"),
                "source_artifact": source_path.as_posix(),
                "sampling_index": source_index,
                "sampling_rule": sampling_rule,
                "requested_num_cases": requested_num_cases,
                "available_num_cases": available,
                "actual_num_cases": actual,
            }
        )

    _validate_manifest_records(manifest_records)
    write_jsonl(manifest_records, manifest_path)

    warnings: list[str] = []
    if actual < requested_num_cases:
        warnings.append(
            f"requested_num_cases={requested_num_cases} exceeds available_num_cases="
            f"{available}; actual_num_cases={actual}"
        )

    report = {
        "backend": backend,
        "source_artifact": source_path.as_posix(),
        "sampling_rule": sampling_rule,
        "seed": seed,
        "id_prefix": id_prefix,
        "requested_num_cases": requested_num_cases,
        "available_num_cases": available,
        "actual_num_cases": actual,
        "can_run_500": actual >= 500,
        "can_run_2000": actual >= 2000,
        "outputs": {
            "full_scale_manifest_path": manifest_path.as_posix(),
            "full_scale_manifest_report_path": report_path.as_posix(),
        },
        "duplication_check": {
            "duplicated_or_upsampled": False,
            "note": "each manifest record maps to a distinct source record index",
        },
        "boundary": BOUNDARY_STATEMENT,
        "warnings": warnings,
    }
    ensure_dir(output_dir)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "manifest_records": manifest_records,
        "report": report,
        "output_files": {
            "full_scale_manifest": manifest_path.as_posix(),
            "full_scale_manifest_report": report_path.as_posix(),
        },
    }


def _validate_manifest_records(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records, start=1):
        for field in ("full_scale_id", "source_question_id", "question", "answer"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"manifest record {index} field {field!r} must be a non-empty str"
                )


def ensure_output_dir_allowed(output_dir: Path) -> None:
    project_root = Path.cwd().resolve()
    resolved = output_dir.resolve()
    forbidden_roots = [
        project_root / "data" / "processed",
        project_root / "outputs" / "logs" / "sprint_2A_real_hidden_state_cache",
        project_root / "outputs" / "logs" / "sprint_2B_representation_features",
        project_root / "outputs" / "logs" / "sprint_2C_probe_dataset",
        project_root / "outputs" / "logs" / "sprint_2D_probe_training_baseline",
        project_root / "outputs" / "logs" / "sprint_2E_guidance_candidate_dry_run",
        project_root / "outputs" / "logs" / "sprint_2F_mini_closed_loop_report",
        project_root / "outputs" / "logs" / "sprint_2_stage_summary",
        project_root / "outputs" / "logs" / "sprint_2G_dataset_prep",
    ]
    for forbidden_root in forbidden_roots:
        forbidden_resolved = forbidden_root.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(
                f"Refusing to write Sprint 2G outputs under forbidden path: {output_dir}"
            )
