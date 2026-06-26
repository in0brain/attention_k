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

from recover_attention.intervention_manifest import (  # noqa: E402
    build_intervention_manifest_file,
    build_intervention_manifest_record,
    build_intervention_manifest_records,
)
from recover_attention.schemas import (  # noqa: E402
    REQUIRED_FIELDS,
    validate_intervention_manifest_record,
)


FORBIDDEN_TOP_LEVEL_FIELDS = [
    "span_id",
    "span_text",
    "span_type",
    "guidance_action",
    "guidance_strength",
    "baseline_answer",
    "guided_answer",
    "intervened_answer",
    "answer_changed",
    "trajectory_stability_score",
    "answer_stability_score",
    "raw_attention_score",
    "hidden_states",
    "attention_maps",
    "hidden_states_path",
    "attentions_path",
    "probe_label",
    "probe_confidence",
]

FORBIDDEN_PLANNED_OPERATION_KEYS = [
    "hidden_states_path",
    "attentions_path",
    "guided_answer",
    "baseline_answer",
    "intervened_question",
    "trajectory_stability_score",
    "answer_stability_score",
]


def valid_attention_anchor_label_record(
    unit_id: str = "unit_001",
    anchor_label: str = "Medium Anchor",
    unit_scope: str = "single",
    group_type: str = "single",
    span_ids: list[str] | None = None,
    spans: list[dict] | None = None,
) -> dict:
    span_ids = span_ids or ["span_001"]
    spans = spans or [{"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}]
    unit_evidence_id = f"gsm8k_0001__{unit_id}__evidence_aggregate_stub_v0"
    anchor_id = f"{unit_evidence_id}__anchor_early_evidence_rule_stub_v0"
    return {
        "attention_anchor_label_id": anchor_id,
        "unit_evidence_id": unit_evidence_id,
        "id": "gsm8k_0001",
        "unit_id": unit_id,
        "unit_scope": unit_scope,
        "group_type": group_type,
        "span_ids": span_ids,
        "spans": spans,
        "original_question": "Tom has 3 apples and buys 2 more.",
        "semantic_evidence": {"summary_label": "x", "summary_score": 0.75},
        "recoverability_evidence": {"recoverability_label": "Recoverable", "recoverability_score": 1.0},
        "available_signal_types": ["semantic_necessity", "semantic_recoverability"],
        "missing_signal_types": [
            "trajectory_stability",
            "answer_stability",
            "raw_attention_pattern",
            "attention_steering_effect",
        ],
        "attention_importance_score": 0.55,
        "attention_anchor_label": anchor_label,
        "label_backend": "early_evidence_rule_stub_v0",
        "label_status": "partial_evidence_label",
        "evidence": {"notes": "fixture"},
    }


def group_anchor_record() -> dict:
    return valid_attention_anchor_label_record(
        unit_id="unit_009",
        unit_scope="group",
        group_type="number_set",
        span_ids=["span_001", "span_004"],
        spans=[
            {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
            {"span_id": "span_004", "text": "2", "type": "number", "start": 26, "end": 27},
        ],
    )


# --- 15.1 normal construction ---------------------------------------------------


def test_single_record_builds_valid_manifest() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    assert validate_intervention_manifest_record(record) is None


def test_multiple_records_build_multiple_manifests() -> None:
    inputs = [
        valid_attention_anchor_label_record(unit_id="unit_001"),
        group_anchor_record(),
    ]
    records, stats = build_intervention_manifest_records(inputs)
    assert len(records) == 2
    assert stats["num_input_attention_anchor_labels"] == 2
    assert stats["num_output_intervention_manifest"] == 2
    for record in records:
        assert validate_intervention_manifest_record(record) is None


def test_intervention_id_format() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    anchor_id = (
        "gsm8k_0001__unit_001__evidence_aggregate_stub_v0__anchor_early_evidence_rule_stub_v0"
    )
    assert record["intervention_id"] == f"{anchor_id}__intervention_mask_manifest_stub_v0"


# --- 15.2 default fields -------------------------------------------------------


def test_default_fields() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    assert record["intervention_type"] == "mask"
    assert record["target_scope"] == "unit"
    assert record["intervention_backend"] == "manifest_stub_v0"
    assert record["intervention_status"] == "planned_only"
    assert isinstance(record["planned_operation"], dict)
    assert record["planned_operation"]["operation_name"] == "mask_unit"
    assert record["planned_operation"]["mask_token"] == "[MASK]"


# --- 15.3 field copy -----------------------------------------------------------


def test_anchor_fields_are_copied() -> None:
    source = group_anchor_record()
    record = build_intervention_manifest_record(source)
    for field in [
        "attention_anchor_label_id",
        "unit_evidence_id",
        "id",
        "unit_id",
        "unit_scope",
        "group_type",
        "span_ids",
        "spans",
        "original_question",
        "attention_importance_score",
        "attention_anchor_label",
        "label_backend",
        "label_status",
    ]:
        assert record[field] == source[field]


# --- 15.4 no filtering ---------------------------------------------------------


@pytest.mark.parametrize(
    "anchor_label",
    ["Strong Anchor", "Medium Anchor", "Weak Anchor", "Risky Anchor", "Distractor"],
)
def test_all_labels_produce_manifest(anchor_label: str) -> None:
    record = build_intervention_manifest_record(
        valid_attention_anchor_label_record(anchor_label=anchor_label)
    )
    assert validate_intervention_manifest_record(record) is None
    assert record["attention_anchor_label"] == anchor_label


def test_input_count_equals_output_count() -> None:
    inputs = [
        valid_attention_anchor_label_record(unit_id=f"unit_{i:03d}", anchor_label=label)
        for i, label in enumerate(
            ["Strong Anchor", "Medium Anchor", "Weak Anchor", "Risky Anchor", "Distractor"],
            start=1,
        )
    ]
    records, _ = build_intervention_manifest_records(inputs)
    assert len(records) == len(inputs)


# --- 15.5 forbidden fields -----------------------------------------------------


def test_output_has_no_forbidden_top_level_fields() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    for field in FORBIDDEN_TOP_LEVEL_FIELDS:
        assert field not in record


def test_output_fields_match_required_schema() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    assert set(record.keys()) == set(REQUIRED_FIELDS["intervention_manifest"])


def test_planned_operation_has_no_execution_keys() -> None:
    record = build_intervention_manifest_record(valid_attention_anchor_label_record())
    for key in FORBIDDEN_PLANNED_OPERATION_KEYS:
        assert key not in record["planned_operation"]


# --- 15.6 error handling -------------------------------------------------------


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported intervention backend"):
        build_intervention_manifest_record(
            valid_attention_anchor_label_record(), intervention_backend="made_up"
        )


def test_unsupported_intervention_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported intervention type"):
        build_intervention_manifest_record(
            valid_attention_anchor_label_record(), intervention_type="rewrite"
        )


def test_invalid_input_record_raises() -> None:
    record = valid_attention_anchor_label_record()
    del record["label_backend"]
    with pytest.raises(ValueError):
        build_intervention_manifest_record(record)


def test_empty_mask_token_raises() -> None:
    with pytest.raises(ValueError, match="mask_token"):
        build_intervention_manifest_record(valid_attention_anchor_label_record(), mask_token="  ")


def test_missing_input_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_intervention_manifest_file(tmp_path / "missing.jsonl", tmp_path / "out.jsonl")


# --- 15.7 CLI smoke test -------------------------------------------------------


def test_cli_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "attention_anchor_labels.jsonl"
    output_path = tmp_path / "intervention_manifest.jsonl"
    inputs = [valid_attention_anchor_label_record(unit_id="unit_001"), group_anchor_record()]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in inputs) + "\n",
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "scripts" / "12_build_intervention_manifest.py"
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
        assert validate_intervention_manifest_record(record) is None
    assert "num_output_intervention_manifest: 2" in result.stdout
