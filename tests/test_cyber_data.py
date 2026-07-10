from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import cyber_data as cd
from recover_attention.schemas import validate_cyber_sample_record

RAW = [
    {
        "question": "Which protocol routes packets?",
        "answers": {"A": "HTTP", "B": "IP", "C": "SMTP", "D": "FTP"},
        "solution": "B",
    },
    {
        "question": "Which property is desirable for biometrics?",
        "answers": {
            "A": "Permanent",
            "B": "Transferable",
            "C": "Forgiving",
            "D": "Public",
        },
        "solution": "A",
    },
]


def canonical(raw: dict, index: int = 0, seed: int = 42) -> dict:
    return cd.to_canonical_cyber_sample(
        raw,
        original_index=index,
        source="CyberMetric-2000-v1",
        shuffle_seed=seed,
    )


def test_load_cybermetric_records_parses_raw_json(tmp_path: Path) -> None:
    path = tmp_path / "cybermetric.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    assert cd.load_cybermetric_records(path) == RAW


def test_canonical_schema_fields_are_complete() -> None:
    record = cd.grouped_split([canonical(RAW[0])])[0]
    assert {
        "example_id",
        "dataset",
        "source",
        "group_id",
        "task_type",
        "input_text",
        "question",
        "candidate_labels",
        "candidate_choices",
        "gold_label",
        "gold_label_id",
        "gold_label_text",
        "label_space",
        "metadata",
    } <= record.keys()
    assert validate_cyber_sample_record(record) is None


def test_example_id_is_stable_and_unique() -> None:
    records = [canonical(raw, index) for index, raw in enumerate(RAW)]
    assert records[0]["example_id"] == canonical(RAW[0], 0)["example_id"]
    assert len({record["example_id"] for record in records}) == len(records)


def test_fixed_seed_shuffle_is_reproducible() -> None:
    choices, gold_position = cd.build_candidate_choices(
        cd.normalize_cybermetric_record(
            RAW[0], original_index=0, source="CyberMetric-2000-v1"
        )
    )
    first = cd.shuffle_candidate_choices(
        choices, gold_position, example_id="cybermetric_000000", seed=42
    )
    second = cd.shuffle_candidate_choices(
        choices, gold_position, example_id="cybermetric_000000", seed=42
    )
    assert first == second


def test_different_seed_can_change_option_order() -> None:
    orders = {
        tuple(choice["original_position"] for choice in canonical(RAW[0], seed=seed)["candidate_choices"])
        for seed in range(8)
    }
    assert len(orders) > 1


def test_shuffle_preserves_gold_semantic_mapping() -> None:
    record = canonical(RAW[0])
    gold_choice = next(
        choice for choice in record["candidate_choices"]
        if choice["choice"] == record["gold_label"]
    )
    assert gold_choice["label_text"] == "IP"
    assert record["gold_label_text"] == "IP"


def test_original_position_is_preserved() -> None:
    record = canonical(RAW[0])
    assert sorted(choice["original_position"] for choice in record["candidate_choices"]) == [0, 1, 2, 3]


def test_candidate_choices_preserve_label_id_and_text() -> None:
    record = canonical(RAW[0])
    assert all("label_id" in choice and choice["label_id"] is None for choice in record["candidate_choices"])
    assert {choice["label_text"] for choice in record["candidate_choices"]} == set(RAW[0]["answers"].values())


def test_build_mcq_prompt_contains_all_options() -> None:
    record = canonical(RAW[0])
    prompt = cd.build_mcq_prompt(record)
    for choice in record["candidate_choices"]:
        assert f"{choice['choice']}. {choice['label_text']}" in prompt


def test_build_mcq_prompt_does_not_emit_gold_metadata() -> None:
    record = canonical(RAW[0])
    prompt = cd.build_mcq_prompt(record)
    assert "gold_label" not in prompt
    assert "Correct answer" not in prompt
    assert prompt.endswith("Answer: <letter>")


def test_grouped_split_has_no_group_leakage() -> None:
    duplicated = canonical(RAW[0], 0)
    duplicate = deepcopy(duplicated)
    duplicate["example_id"] = "cybermetric_000002"
    rows = cd.grouped_split([duplicated, duplicate, canonical(RAW[1], 1)])
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["metadata"]["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_grouped_split_is_reproducible() -> None:
    rows = [canonical(raw, index) for index, raw in enumerate(RAW)]
    first = cd.grouped_split(rows, seed=7)
    second = cd.grouped_split(rows, seed=7)
    assert [row["metadata"]["split"] for row in first] == [
        row["metadata"]["split"] for row in second
    ]


def test_smoke_sample_is_not_input_prefix() -> None:
    rows = [
        canonical(
            {
                "question": f"Question {index}?",
                "answers": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                "solution": "A",
            },
            index,
        )
        for index in range(30)
    ]
    rows = cd.grouped_split(rows)
    smoke = cd.select_grouped_smoke_sample(rows, sample_size=10, seed=42)
    assert [row["example_id"] for row in smoke] != [
        row["example_id"] for row in rows[:10]
    ]
    assert smoke == cd.select_grouped_smoke_sample(rows, sample_size=10, seed=42)


def test_gold_option_position_distribution_is_auditable() -> None:
    rows = cd.grouped_split([canonical(raw, index) for index, raw in enumerate(RAW)])
    audit = cd.audit_cyber_samples(rows)
    report = cd.option_position_bias_pre_model_report(rows, shuffle_seed=42)
    assert sum(audit["label_distribution"].values()) == len(rows)
    assert sum(report["shuffled_gold_position_distribution"].values()) == len(rows)
    assert report["option_order_deterministic_under_fixed_seed"] is True


def test_shuffle_does_not_mutate_input_choices() -> None:
    normalized = cd.normalize_cybermetric_record(
        RAW[0], original_index=0, source="CyberMetric-2000-v1"
    )
    choices, gold_position = cd.build_candidate_choices(normalized)
    before = deepcopy(choices)
    cd.shuffle_candidate_choices(
        choices, gold_position, example_id="cybermetric_000000", seed=42
    )
    assert choices == before
