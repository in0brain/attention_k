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
from recover_attention.full_scale_manifest import (
    build_full_scale_manifest,
    select_indices,
)


def _write_source(path: Path, count: int) -> None:
    records = [
        {
            "question_id": f"gsm8k_train_{i:06d}",
            "source_dataset": "gsm8k",
            "source_split": "train",
            "question": f"Question {i} with {i} apples?",
            "answer": str(i),
            "metadata": {},
        }
        for i in range(count)
    ]
    write_jsonl(records, path)


def test_select_indices_first_n_is_deterministic() -> None:
    assert select_indices(100, 5, "first_n", seed=42) == [0, 1, 2, 3, 4]


def test_select_indices_seeded_sample_is_deterministic_and_unique() -> None:
    first = select_indices(100, 10, "seeded_sample", seed=42)
    second = select_indices(100, 10, "seeded_sample", seed=42)
    assert first == second
    assert len(set(first)) == 10


def test_build_full_scale_manifest_samples_requested(tmp_path: Path) -> None:
    source = tmp_path / "src.jsonl"
    _write_source(source, 50)
    output_dir = tmp_path / "00_manifest"

    result = build_full_scale_manifest(
        source_path=source,
        output_dir=output_dir,
        requested_num_cases=20,
        sampling_rule="seeded_sample",
        seed=42,
    )

    records = read_jsonl(output_dir / "full_scale_manifest.jsonl")
    assert len(records) == 20
    assert result["report"]["actual_num_cases"] == 20
    assert result["report"]["available_num_cases"] == 50
    ids = [r["full_scale_id"] for r in records]
    assert ids[0] == "fs2000_000001"
    assert len(set(ids)) == 20
    assert len({r["source_question_id"] for r in records}) == 20


def test_build_full_scale_manifest_shortfall_warns(tmp_path: Path) -> None:
    source = tmp_path / "src.jsonl"
    _write_source(source, 10)
    output_dir = tmp_path / "00_manifest"

    result = build_full_scale_manifest(
        source_path=source,
        output_dir=output_dir,
        requested_num_cases=2000,
        sampling_rule="first_n",
    )

    assert result["report"]["actual_num_cases"] == 10
    assert result["report"]["can_run_2000"] is False
    assert any("exceeds available" in w for w in result["report"]["warnings"])


def test_build_full_scale_manifest_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "src.jsonl"
    _write_source(source, 5)
    output_dir = tmp_path / "00_manifest"
    build_full_scale_manifest(
        source_path=source, output_dir=output_dir, requested_num_cases=3
    )
    with pytest.raises(FileExistsError):
        build_full_scale_manifest(
            source_path=source, output_dir=output_dir, requested_num_cases=3
        )
