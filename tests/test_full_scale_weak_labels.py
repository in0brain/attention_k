from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.full_scale_weak_labels import (
    build_full_scale_weak_labels,
    weak_target_for_span,
)
from recover_attention.hidden_state_cache import validate_2a_manifest_records


def _manifest(path: Path, questions: list[str]) -> None:
    records = [
        {
            "full_scale_id": f"fs2000_{i + 1:06d}",
            "source_question_id": f"gsm8k_train_{i:06d}",
            "source_dataset": "gsm8k",
            "source_split": "train",
            "question": question,
            "answer": str(i),
            "source_artifact": "data/raw/gsm8k_train_normalized.jsonl",
            "sampling_index": i,
            "sampling_rule": "seeded_sample",
            "requested_num_cases": len(questions),
            "available_num_cases": 100,
            "actual_num_cases": len(questions),
        }
        for i, question in enumerate(questions)
    ]
    write_jsonl(records, path)


def test_weak_target_for_span_covers_classes() -> None:
    assert weak_target_for_span("number")[0] == "positive_anchor"
    assert weak_target_for_span("negation")[0] == "risk_positive"
    assert weak_target_for_span("object")[0] == "negative"
    assert weak_target_for_span("operation")[0] == "hard_negative_or_weak_positive"
    target, _, _, usable = weak_target_for_span("nonexistent_type")
    assert target == "unmapped"
    assert usable is False


def test_build_weak_labels_produces_valid_2a_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(
        manifest,
        [
            "Tom has 3 apples and buys 2 more. How many apples does he have now?",
            "Maria gives 4 pencils to Luis and has 6 left. How many did she have?",
            "A bakery sold 12 muffins and 8 more. How many muffins total?",
        ],
    )
    output_dir = tmp_path / "01_downstream"

    result = build_full_scale_weak_labels(manifest_path=manifest, output_dir=output_dir)

    weak_labels = read_jsonl(output_dir / "weak_labels_2000.jsonl")
    manifest_2a = read_jsonl(output_dir / "full_scale_2a_manifest.jsonl")
    assert len(weak_labels) == 3
    assert all(record["label_source"] == "weak_auto" for record in weak_labels)
    assert all(record["human_reviewed"] is False for record in weak_labels)
    # The 2A manifest must satisfy the hidden-state cache schema.
    validate_2a_manifest_records(manifest_2a)
    for record in manifest_2a:
        assert record["masked_question"] != record["original_question"]
        assert record["recovered_questions"][0] != record["masked_question"]
        assert record["human_recoverability_label"] == "weak_auto_not_human_reviewed"


def test_build_weak_labels_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, ["Sam had 15 marbles and lost 7 marbles. How many are left?"])
    first = build_full_scale_weak_labels(
        manifest_path=manifest, output_dir=tmp_path / "a"
    )
    second = build_full_scale_weak_labels(
        manifest_path=manifest, output_dir=tmp_path / "b"
    )
    assert first["weak_labels"][0]["probe_target"] == second["weak_labels"][0]["probe_target"]
    assert first["weak_labels"][0]["chosen_span_text"] == second["weak_labels"][0]["chosen_span_text"]


def test_build_weak_labels_unmapped_when_no_candidates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, ["?!"])
    output_dir = tmp_path / "01_downstream"

    result = build_full_scale_weak_labels(manifest_path=manifest, output_dir=output_dir)

    weak_labels = read_jsonl(output_dir / "weak_labels_2000.jsonl")
    manifest_2a = read_jsonl(output_dir / "full_scale_2a_manifest.jsonl")
    assert weak_labels[0]["probe_target"] == "unmapped"
    assert weak_labels[0]["probe_target_usable"] is False
    assert manifest_2a == []
    assert result["report"]["counts"]["num_unmapped"] == 1
