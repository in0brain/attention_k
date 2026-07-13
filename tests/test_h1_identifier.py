from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import h1_identifier as h1
from recover_attention.data_io import write_jsonl


def write_fixture_index(root: Path) -> h1.OntologyIndex:
    rows = {
        "cve": [
            {
                "family": "cve",
                "normalized_id": "CVE-2021-44228",
                "status": "PUBLISHED",
                "name": None,
                "description": "Log4j remote code execution.",
                "metadata": {},
            },
            {
                "family": "cve",
                "normalized_id": "CVE-2024-999999",
                "status": "RESERVED",
                "name": None,
                "description": None,
                "metadata": {},
            },
        ],
        "attack": [
            {
                "family": "attack",
                "normalized_id": "T1059",
                "status": "active",
                "name": "Command and Scripting Interpreter",
                "description": "Adversaries may abuse interpreters.",
                "metadata": {"tactics": ["execution"]},
            },
            {
                "family": "attack",
                "normalized_id": "T1059.001",
                "status": "deprecated",
                "name": "PowerShell",
                "description": "Adversaries may abuse PowerShell.",
                "metadata": {"tactics": ["execution"]},
            },
        ],
        "cwe": [
            {
                "family": "cwe",
                "normalized_id": "CWE-79",
                "status": "Draft",
                "name": "Cross-site Scripting",
                "description": "Improper neutralization of input.",
                "metadata": {},
            }
        ],
    }
    for family, records in rows.items():
        write_jsonl(records, root / family / "ontology_index.jsonl")
    return h1.build_ontology_index(root)


def test_normalize_identifier_variants() -> None:
    assert h1.normalize_identifier("cve", "cve 2021-44228") == "CVE-2021-44228"
    assert h1.normalize_identifier("cve", "CVE-2021-44228") == "CVE-2021-44228"
    assert h1.normalize_identifier("attack", "T1059") == "T1059"
    assert h1.normalize_identifier("attack", "T1059.001") == "T1059.001"
    assert h1.normalize_identifier("cwe", "CWE 79") == "CWE-79"


def test_extract_identifiers_positive_and_negative_examples() -> None:
    text = (
        "CVE 2021-44228 maps to T1059.001 and CWE-79. "
        "Ignore v2.0.1, 2021-2024, port 443, deadbeef, T-shirt, T5, AES, BERT, DNS."
    )
    mentions = h1.extract_identifiers(text)
    assert [(m.family, m.normalized) for m in mentions] == [
        ("cve", "CVE-2021-44228"),
        ("attack", "T1059.001"),
        ("cwe", "CWE-79"),
    ]


def test_attack_parent_and_subtechnique_are_different_granularity() -> None:
    mentions = h1.extract_identifiers("Compare T1059 with T1059.001.")
    assert mentions[0].normalized == "T1059"
    assert mentions[0].granularity == "technique"
    assert mentions[1].normalized == "T1059.001"
    assert mentions[1].parent == "T1059"
    assert mentions[1].granularity == "subtechnique"


def test_ontology_index_existence_and_status_rules(tmp_path: Path) -> None:
    index = write_fixture_index(tmp_path)
    assert index.has("cve", "CVE-2024-999999")
    assert index.status("cve", "CVE-2024-999999") == "RESERVED"
    assert index.has("attack", "T1059.001")
    assert index.status("attack", "T1059.001") == "deprecated"
    assert not index.has("attack", "T1059.999")
    assert index.snapshot_fingerprints["cwe"]


def test_echo_exclusion_and_h1_positive(tmp_path: Path) -> None:
    index = write_fixture_index(tmp_path)
    prompt = "Discuss CVE-2021-44228 without inventing related ids."
    completion = "CVE-2021-44228 is relevant; CVE-2099-12345 and T1059.999 are not real."
    labeled = h1.label_completion(completion, prompt, index)
    labels = {(row["normalized"], row["label"]) for row in labeled["mentions"]}
    assert ("CVE-2021-44228", "echoed") in labels
    assert ("CVE-2099-12345", "fabricated") in labels
    assert ("T1059.999", "fabricated") in labels
    assert labeled["h1_positive"] is True


def test_h1_gold_leakage_checker_rejects_nested_source_id() -> None:
    with pytest.raises(ValueError, match="source_entry_id"):
        h1.assert_no_h1_gold_label_leakage({"features": {"source_entry_id": "CWE-79"}})
    h1.assert_no_h1_gold_label_leakage({"features": {"mention_logprob": -3.2}})
