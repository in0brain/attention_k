from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import cyber_data as cd
RAW = [
    {
        "question": "Which protocol routes packets?",
        "answers": {"A": "HTTP", "B": "IP", "C": "SMTP", "D": "FTP"},
        "solution": "B",
    },
    {
        "question": "Which property is desirable for biometrics?",
        "answers": {"A": "Permanent", "B": "Transferable", "C": "Forgiving", "D": "Public"},
        "solution": "A",
    },
]


def test_to_canonical_schema_preserves_semantic_choices():
    rows = cd.to_canonical_schema(RAW, limit=2, seed=1, shuffle_options=True)
    assert len(rows) == 2
    row = rows[0]
    assert row["label_space"] == "mcq_option_letter"
    assert row["candidate_labels"] == ["A", "B", "C", "D"]
    assert row["gold_label"] in row["candidate_labels"]
    assert row["gold_label_text"]
    assert all({"choice", "label_id", "label_text", "source_choice"} <= set(c) for c in row["candidate_choices"])


def test_build_mcq_prompt_contains_all_options_in_order():
    row = cd.to_canonical_schema(RAW, limit=1, seed=1, shuffle_options=False)[0]
    prompt = cd.build_mcq_prompt(row)
    assert "Question: Which protocol routes packets?" in prompt
    assert "A. HTTP" in prompt
    assert "B. IP" in prompt
    assert prompt.index("A. HTTP") < prompt.index("B. IP")
    assert "Answer: <letter>" in prompt


def test_option_order_randomization_is_deterministic():
    a = cd.to_canonical_schema(RAW, limit=2, seed=7, shuffle_options=True)
    b = cd.to_canonical_schema(RAW, limit=2, seed=7, shuffle_options=True)
    assert [r["candidate_choices"] for r in a] == [r["candidate_choices"] for r in b]


def test_grouped_split_has_no_group_leakage_flag():
    rows = cd.to_canonical_schema(RAW, limit=2, seed=1)
    split = cd.grouped_split(rows)
    assert split["num_examples"] == 2
    assert split["group_leakage_detected"] is False


def test_dataset_and_position_audits_report_distribution():
    rows = cd.to_canonical_schema(RAW, limit=2, seed=1)
    audit = cd.audit_dataset(rows, dataset="cybermetric")
    assert audit["num_examples"] == 2
    assert sum(audit["label_distribution"].values()) == 2
    bias = cd.option_position_bias_report(rows, greedy_labels=["A", "B"], sampled_labels=["A"])
    assert "gold_choice_distribution" in bias
    assert bias["option_order_fixed_seed_randomized"] is True
