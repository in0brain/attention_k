from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import h1_data as hd
from recover_attention.h1_identifier import OntologyEntry, OntologyIndex
from recover_attention.schemas import validate_h1_sample_record


def fixture_index(num_per_family: int = 12) -> OntologyIndex:
    entries = {"cve": {}, "attack": {}, "cwe": {}}
    for i in range(num_per_family):
        cve = f"CVE-2024-{100000 + i}"
        entries["cve"][cve] = OntologyEntry(
            family="cve",
            normalized_id=cve,
            status="PUBLISHED",
            name=None,
            description=f"Buffer overflow in Product{i} allows code execution.",
            metadata={"products": [f"Product{i}"]},
        )
        attack = f"T{1000 + i}"
        entries["attack"][attack] = OntologyEntry(
            family="attack",
            normalized_id=attack,
            status="active",
            name=f"Technique {i}",
            description=f"Adversaries may perform behavior {i}.",
            metadata={"tactics": [f"tactic-{i % 3}"]},
        )
        cwe = f"CWE-{70 + i}"
        entries["cwe"][cwe] = OntologyEntry(
            family="cwe",
            normalized_id=cwe,
            status="Draft",
            name=f"Weakness {i}",
            description=f"The product mishandles input class {i}.",
            metadata={"categories": [f"category-{i % 4}"]},
        )
    return OntologyIndex(
        entries=entries,
        snapshot_fingerprints={"cve": "a", "attack": "b", "cwe": "c"},
        status_rules={},
    )


def test_h1_sample_validator_accepts_recall_and_open_gen() -> None:
    recall = {
        "example_id": "h1_recall_cwe_0000",
        "route": "recall",
        "family": "cwe",
        "prompt_params": {"template": "recall_v1"},
        "question_text": "Which CWE weakness is described?",
        "source_entry_id": "CWE-79",
        "source_entry_metadata": {"name": "XSS"},
        "group_id": "entry_CWE_79",
        "split": "train",
        "label_space": "open_identifier",
    }
    open_gen = {
        **recall,
        "example_id": "h1_open_cwe_0000",
        "route": "open_gen",
        "source_entry_id": None,
        "group_id": "topic_xss",
    }
    assert validate_h1_sample_record(recall) is None
    assert validate_h1_sample_record(open_gen) is None


def test_h1_sample_validator_rejects_recall_without_source_id() -> None:
    record = {
        "example_id": "h1_recall_cwe_0000",
        "route": "recall",
        "family": "cwe",
        "prompt_params": {},
        "question_text": "Question",
        "source_entry_id": None,
        "source_entry_metadata": {},
        "group_id": "g",
        "split": "train",
        "label_space": "open_identifier",
    }
    with pytest.raises(ValueError, match="source_entry_id"):
        validate_h1_sample_record(record)


def test_question_construction_removes_gold_identifier() -> None:
    entry = OntologyEntry(
        family="attack",
        normalized_id="T1059.001",
        status="active",
        name="T1059.001 PowerShell",
        description="T1059.001 describes PowerShell abuse.",
        metadata={},
    )
    question, _ = hd.build_recall_question(entry)
    assert "T1059.001" not in question
    assert not hd.question_contains_identifier(question, "T1059.001")


def test_build_h1_samples_is_deterministic_and_split_has_no_leakage() -> None:
    index = fixture_index()
    first = hd.build_h1_samples(index, seed=123)
    second = hd.build_h1_samples(index, seed=123)
    third = hd.build_h1_samples(index, seed=456)
    assert first == second
    assert [row["example_id"] for row in first] != [row["example_id"] for row in third]
    for row in first:
        validate_h1_sample_record(row)
    audit = hd.audit_h1_samples(first)
    assert audit["group_leakage_count"] == 0
    assert audit["gold_id_residual_check"]["violations"] == 0


def test_assert_no_question_gold_id_leakage_catches_recall_echo() -> None:
    rows = [
        {
            "example_id": "bad",
            "route": "recall",
            "family": "cve",
            "prompt_params": {},
            "question_text": "Which issue is CVE-2024-100001?",
            "source_entry_id": "CVE-2024-100001",
            "source_entry_metadata": {},
            "group_id": "g",
            "split": "train",
            "label_space": "open_identifier",
        }
    ]
    with pytest.raises(ValueError, match="leaks source_entry_id"):
        hd.assert_no_question_gold_id_leakage(rows)


def test_density_report_marks_cve_as_auxiliary() -> None:
    report = hd.estimate_id_space_density(fixture_index())
    assert report["families"]["cve"]["judgment_strength"].startswith("weak")
    assert report["labeling_strategy"]["primary_families"] == ["attack", "cwe"]
