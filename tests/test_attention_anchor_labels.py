from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.attention_anchor_labels import (  # noqa: E402
    build_attention_anchor_label_record,
    build_attention_anchor_label_records,
    compute_recoverability_risk_score,
    score_attention_anchor_stub_rule_v0,
)
from recover_attention.schemas import (  # noqa: E402
    REQUIRED_FIELDS,
    validate_attention_anchor_label_record,
)


FORBIDDEN_TOP_LEVEL_FIELDS = [
    "span_id",
    "span_text",
    "span_type",
    "sample_id",
    "recovered_question",
    "recoverable",
    "confidence",
    "reason",
    "guidance_action",
    "guidance_strength",
    "hidden_states",
    "attention_maps",
    "trajectory_analysis",
    "answer_stability",
    "raw_attention_pattern",
    "probe_label",
]


def valid_unit_evidence_record(
    unit_id: str = "unit_001",
    summary_score: float = 0.75,
    recoverability_label: str = "Recoverable",
    recoverability_score: float = 1.0,
    misleading_recovery: bool = False,
) -> dict:
    unit_evidence_id = f"gsm8k_0001__{unit_id}__evidence_aggregate_stub_v0"
    return {
        "unit_evidence_id": unit_evidence_id,
        "id": "gsm8k_0001",
        "unit_id": unit_id,
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [{"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}],
        "original_question": "Tom has 3 apples and buys 2 more.",
        "semantic_evidence": {
            "summary_label": "consistent_semantic_necessity_evidence",
            "summary_score": summary_score,
        },
        "recoverability_evidence": {
            "recover_score_id": f"gsm8k_0001__{unit_id}__mask__score_stub_rule_v0",
            "recoverability_label": recoverability_label,
            "recoverability_score": recoverability_score,
            "misleading_recovery": misleading_recovery,
        },
        "available_signal_types": ["semantic_necessity", "semantic_recoverability"],
        "missing_signal_types": [
            "trajectory_stability",
            "answer_stability",
            "raw_attention_pattern",
            "attention_steering_effect",
        ],
        "evidence_backend": "aggregate_stub_v0",
        "evidence_status": "partial_stub_evidence",
        "evidence": {"notes": "fixture"},
    }


def valid_group_unit_evidence_record() -> dict:
    record = valid_unit_evidence_record(unit_id="unit_009")
    record["unit_scope"] = "group"
    record["group_type"] = "number_set"
    record["span_ids"] = ["span_001", "span_004"]
    record["spans"] = [
        {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
        {"span_id": "span_004", "text": "2", "type": "number", "start": 26, "end": 27},
    ]
    return record


# --- 15.1 normal construction ---------------------------------------------------


def test_single_record_builds_valid_label() -> None:
    record = build_attention_anchor_label_record(valid_unit_evidence_record())
    assert validate_attention_anchor_label_record(record) is None


def test_multiple_records_build_multiple_labels() -> None:
    inputs = [
        valid_unit_evidence_record(unit_id="unit_001"),
        valid_unit_evidence_record(unit_id="unit_002"),
        valid_group_unit_evidence_record(),
    ]
    records, stats = build_attention_anchor_label_records(inputs)
    assert len(records) == 3
    assert stats["num_input_unit_evidence"] == 3
    assert stats["num_output_attention_anchor_labels"] == 3
    for record in records:
        assert validate_attention_anchor_label_record(record) is None


def test_attention_anchor_label_id_format() -> None:
    record = build_attention_anchor_label_record(valid_unit_evidence_record())
    assert record["attention_anchor_label_id"] == (
        "gsm8k_0001__unit_001__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0"
    )
    assert record["label_backend"] == "early_evidence_rule_stub_v0"
    assert record["label_status"] == "partial_evidence_label"


# --- 15.2 score rule -----------------------------------------------------------


def test_score_formula_uses_0_6_and_0_4_weights() -> None:
    scored = score_attention_anchor_stub_rule_v0(
        valid_unit_evidence_record(summary_score=0.75, recoverability_label="Recoverable")
    )
    # 0.6 * 0.75 + 0.4 * 0.25 = 0.55
    assert scored["attention_importance_score"] == pytest.approx(0.55)


def test_score_is_clamped_to_unit_interval() -> None:
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(summary_score=1.0, recoverability_label="Misleading Recovery")
    )
    assert 0.0 <= record["attention_importance_score"] <= 1.0


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Recoverable", 0.25),
        ("Partially Recoverable", 0.50),
        ("Non-recoverable", 0.75),
        ("Misleading Recovery", 1.0),
    ],
)
def test_recoverability_risk_score_mapping(label: str, expected: float) -> None:
    assert compute_recoverability_risk_score(label) == expected


# --- 15.3 label rule -----------------------------------------------------------


def test_misleading_recovery_flag_yields_risky_anchor() -> None:
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(recoverability_label="Recoverable", misleading_recovery=True)
    )
    assert record["attention_anchor_label"] == "Risky Anchor"


def test_misleading_recovery_label_yields_risky_anchor() -> None:
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(recoverability_label="Misleading Recovery")
    )
    assert record["attention_anchor_label"] == "Risky Anchor"


def test_high_score_yields_strong_anchor() -> None:
    # 0.6 * 1.0 + 0.4 * 0.75 = 0.9
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(summary_score=1.0, recoverability_label="Non-recoverable")
    )
    assert record["attention_anchor_label"] == "Strong Anchor"


def test_medium_score_yields_medium_anchor() -> None:
    # 0.6 * 0.75 + 0.4 * 0.25 = 0.55
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(summary_score=0.75, recoverability_label="Recoverable")
    )
    assert record["attention_anchor_label"] == "Medium Anchor"


def test_weak_score_yields_weak_anchor() -> None:
    # 0.6 * 0.5 + 0.4 * 0.25 = 0.4
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(summary_score=0.5, recoverability_label="Recoverable")
    )
    assert record["attention_anchor_label"] == "Weak Anchor"


def test_low_score_yields_distractor() -> None:
    # 0.6 * 0.0 + 0.4 * 0.25 = 0.1
    record = build_attention_anchor_label_record(
        valid_unit_evidence_record(summary_score=0.0, recoverability_label="Recoverable")
    )
    assert record["attention_anchor_label"] == "Distractor"


# --- 15.4 field copy -----------------------------------------------------------


def test_unit_metadata_is_copied() -> None:
    source = valid_group_unit_evidence_record()
    record = build_attention_anchor_label_record(source)
    for field in [
        "unit_evidence_id",
        "id",
        "unit_id",
        "unit_scope",
        "group_type",
        "span_ids",
        "spans",
        "semantic_evidence",
        "recoverability_evidence",
        "available_signal_types",
        "missing_signal_types",
    ]:
        assert record[field] == source[field]


# --- 15.5 forbidden fields -----------------------------------------------------


def test_output_has_no_forbidden_top_level_fields() -> None:
    record = build_attention_anchor_label_record(valid_unit_evidence_record())
    for field in FORBIDDEN_TOP_LEVEL_FIELDS:
        assert field not in record


def test_output_fields_match_required_schema() -> None:
    record = build_attention_anchor_label_record(valid_unit_evidence_record())
    assert set(record.keys()) == set(REQUIRED_FIELDS["attention_anchor_label"])


# --- 15.6 error handling -------------------------------------------------------


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported attention label backend"):
        build_attention_anchor_label_record(valid_unit_evidence_record(), label_backend="made_up")


def test_missing_summary_score_raises() -> None:
    record = valid_unit_evidence_record()
    del record["semantic_evidence"]["summary_score"]
    with pytest.raises(ValueError, match="summary_score"):
        build_attention_anchor_label_record(record)


def test_invalid_summary_score_range_raises() -> None:
    record = valid_unit_evidence_record(summary_score=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_attention_anchor_label_record(record)


def test_missing_recoverability_label_raises() -> None:
    record = valid_unit_evidence_record()
    del record["recoverability_evidence"]["recoverability_label"]
    with pytest.raises(ValueError, match="recoverability_label"):
        build_attention_anchor_label_record(record)


def test_invalid_recoverability_label_raises() -> None:
    record = valid_unit_evidence_record(recoverability_label="Maybe")
    with pytest.raises(ValueError, match="invalid recoverability_label"):
        build_attention_anchor_label_record(record)


def test_invalid_input_unit_evidence_record_raises() -> None:
    record = valid_unit_evidence_record()
    del record["evidence_backend"]
    with pytest.raises(ValueError):
        build_attention_anchor_label_record(record)


def test_missing_input_file_raises(tmp_path: Path) -> None:
    from recover_attention.attention_anchor_labels import build_attention_anchor_label_file

    with pytest.raises(FileNotFoundError):
        build_attention_anchor_label_file(
            tmp_path / "missing.jsonl",
            tmp_path / "out.jsonl",
        )


# --- 15.7 CLI smoke test -------------------------------------------------------


def test_cli_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "unit_evidence.jsonl"
    output_path = tmp_path / "attention_anchor_labels.jsonl"
    inputs = [
        valid_unit_evidence_record(unit_id="unit_001"),
        valid_group_unit_evidence_record(),
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in inputs) + "\n",
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "scripts" / "11_build_attention_anchor_labels.py"
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(input_path), "--output", str(output_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert validate_attention_anchor_label_record(record) is None
    assert "num_output_attention_anchor_labels: 2" in result.stdout
