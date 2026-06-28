from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.attention_anchor_labels import build_attention_anchor_label_records
from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.intervention_manifest import build_intervention_manifest_records
from recover_attention.schemas import (
    validate_attention_anchor_label_record,
    validate_intervention_manifest_record,
    validate_unit_evidence_record,
)
from recover_attention.unit_evidence import build_unit_evidence_records


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "14_rebuild_downstream_upgraded_recovery_scoring.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("rebuild_upgraded_downstream", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."


def span_001() -> dict:
    return {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}


def span_002() -> dict:
    return {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27}


def scores(entailment: float, contradiction: float) -> dict:
    return {
        "entailment": entailment,
        "neutral": round(1.0 - entailment - contradiction, 10),
        "contradiction": contradiction,
    }


def label_from_scores(score_values: dict) -> str:
    return max(("entailment", "neutral", "contradiction"), key=lambda label: score_values[label])


def semantic_label_record(
    *,
    question_id: str,
    unit_id: str,
    span: dict,
    forward_entailment: float = 0.62,
    backward_entailment: float = 0.36,
) -> dict:
    resolved_span = deepcopy(span)
    ablated_question = "Tom has some apples and buys 2 more."
    forward_scores = scores(forward_entailment, 0.05)
    backward_scores = scores(backward_entailment, 0.05)
    bidirectional_entailment_score = min(
        forward_scores["entailment"],
        backward_scores["entailment"],
    )
    semantic_necessity_label = (
        "Equivalent" if bidirectional_entailment_score >= 0.70 else "Information Loss"
    )
    nli_id = f"{question_id}__{unit_id}__delete__nli_hf_nli_auto_v0"
    return {
        "semantic_label_id": f"{nli_id}__sem_rule_v0",
        "nli_id": nli_id,
        "ablation_id": f"{question_id}__{unit_id}__delete",
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": [resolved_span["span_id"]],
        "spans": [resolved_span],
        "ablation_type": "delete",
        "original_question": ORIGINAL_QUESTION,
        "ablated_question": ablated_question,
        "nli_backend": "hf_nli_auto_v0",
        "language": "en",
        "language_setting": "auto",
        "forward": {
            "premise": ORIGINAL_QUESTION,
            "hypothesis": ablated_question,
            "label": label_from_scores(forward_scores),
            "scores": forward_scores,
        },
        "backward": {
            "premise": ablated_question,
            "hypothesis": ORIGINAL_QUESTION,
            "label": label_from_scores(backward_scores),
            "scores": backward_scores,
        },
        "bidirectional_entailment_score": bidirectional_entailment_score,
        "contradiction_score": 0.05,
        "semantic_label_backend": "rule_v0",
        "semantic_necessity_label": semantic_necessity_label,
        "semantic_necessity_score": round(1.0 - bidirectional_entailment_score, 10),
        "is_semantically_necessary": semantic_necessity_label != "Equivalent",
        "rule_parameters": {
            "equivalent_threshold": 0.70,
            "directional_entailment_threshold": 0.50,
            "contradiction_threshold": 0.50,
        },
        "decision_reason": "test fixture",
    }


def recover_score_record(
    *,
    question_id: str,
    unit_id: str,
    span: dict,
    score_backend: str,
    recoverability_label: str,
    recoverability_score: float,
    misleading_recovery: bool = False,
) -> dict:
    resolved_span = deepcopy(span)
    masked_id = f"{question_id}__{unit_id}__mask"
    recovered_question = (
        ORIGINAL_QUESTION
        if recoverability_label == "Recoverable"
        else "Tom has five apples and buys 2 more."
    )
    return {
        "recover_score_id": f"{masked_id}__score_{score_backend}",
        "masked_id": masked_id,
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": [resolved_span["span_id"]],
        "spans": [resolved_span],
        "original_question": ORIGINAL_QUESTION,
        "masked_question": "Tom has [MASK] apples and buys 2 more.",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovery_backend": "ollama_chat_v0",
        "num_samples": 1,
        "source_sample_ids": [0],
        "recovered_questions": [recovered_question],
        "recoverability_label": recoverability_label,
        "recoverability_score": recoverability_score,
        "confidence_mean": recoverability_score,
        "recovery_consistency": 1.0,
        "misleading_recovery": misleading_recovery,
        "score_backend": score_backend,
        "evidence": {"fixture": True},
    }


def fixture_records(num_units: int = 2) -> dict[str, list[dict]]:
    spans = [span_001(), span_002(), span_001()]
    semantic = []
    baseline_scores = []
    upgraded_scores = []
    for index in range(num_units):
        unit_id = f"unit_{index + 1:03d}"
        span = spans[index]
        semantic.append(
            semantic_label_record(
                question_id=f"q{index + 1}",
                unit_id=unit_id,
                span=span,
            )
        )
        baseline_scores.append(
            recover_score_record(
                question_id=f"q{index + 1}",
                unit_id=unit_id,
                span=span,
                score_backend="stub_rule_v0",
                recoverability_label="Recoverable",
                recoverability_score=1.0,
            )
        )
        upgraded_label = "Misleading Recovery" if index == 0 else "Recoverable"
        upgraded_scores.append(
            recover_score_record(
                question_id=f"q{index + 1}",
                unit_id=unit_id,
                span=span,
                score_backend="nli_recovery_judge_v0",
                recoverability_label=upgraded_label,
                recoverability_score=0.2 if upgraded_label == "Misleading Recovery" else 1.0,
                misleading_recovery=upgraded_label == "Misleading Recovery",
            )
        )

    baseline_unit_evidence, _ = build_unit_evidence_records(semantic, baseline_scores)
    baseline_attention_labels, _ = build_attention_anchor_label_records(baseline_unit_evidence)
    baseline_manifest, _ = build_intervention_manifest_records(baseline_attention_labels)
    return {
        "semantic": semantic,
        "baseline_scores": baseline_scores,
        "upgraded_scores": upgraded_scores,
        "baseline_unit_evidence": baseline_unit_evidence,
        "baseline_attention_labels": baseline_attention_labels,
        "baseline_manifest": baseline_manifest,
    }


def write_fixture_inputs(tmp_path: Path, records: dict[str, list[dict]]) -> dict[str, Path]:
    paths = {
        "semantic": tmp_path / "semantic_labels_real.jsonl",
        "baseline_scores": tmp_path / "recover_scores_real.jsonl",
        "upgraded_scores": tmp_path / "recover_scores_nli_judge.jsonl",
        "baseline_unit_evidence": tmp_path / "unit_evidence_real.jsonl",
        "baseline_attention_labels": tmp_path / "attention_anchor_labels_real.jsonl",
        "baseline_manifest": tmp_path / "intervention_manifest_real.jsonl",
        "baseline_report": tmp_path / "real_signal_report.json",
    }
    for key, path in paths.items():
        if key == "baseline_report":
            path.write_text(
                json.dumps(
                    {
                        "recoverability_label_counts": {"Recoverable": len(records["baseline_scores"])},
                        "attention_anchor_label_counts": {"Weak Anchor": len(records["baseline_scores"])},
                    }
                ),
                encoding="utf-8",
            )
        else:
            write_jsonl(records[key], path)
    return paths


def make_args(paths: dict[str, Path], output_dir: Path, limit: int | None = None) -> Namespace:
    return Namespace(
        semantic_labels=str(paths["semantic"]),
        upgraded_recover_scores=str(paths["upgraded_scores"]),
        baseline_recover_scores=str(paths["baseline_scores"]),
        baseline_unit_evidence=str(paths["baseline_unit_evidence"]),
        baseline_attention_anchor_labels=str(paths["baseline_attention_labels"]),
        baseline_intervention_manifest=str(paths["baseline_manifest"]),
        baseline_report=str(paths["baseline_report"]),
        output_dir=str(output_dir),
        unit_evidence_backend="aggregate_stub_v0",
        attention_label_backend="early_evidence_rule_stub_v0",
        intervention_type="mask",
        intervention_backend="manifest_stub_v0",
        mask_token="[MASK]",
        limit=limit,
    )


def test_run_rebuild_writes_expected_files_and_required_report_keys(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records())
    output_dir = tmp_path / "sprint_1P"

    report = module.run_rebuild(make_args(paths, output_dir))

    expected_names = {
        "unit_evidence_upgraded.jsonl",
        "attention_anchor_labels_upgraded.jsonl",
        "intervention_manifest_upgraded.jsonl",
        "upgraded_downstream_report.json",
        "upgraded_downstream_report.md",
    }
    assert expected_names.issubset({path.name for path in output_dir.iterdir()})
    assert {
        "run_metadata",
        "input_counts",
        "output_counts",
        "recover_score_comparison",
        "unit_evidence_comparison",
        "attention_anchor_label_comparison",
        "intervention_manifest_comparison",
        "changed_cases",
        "sample_records",
        "known_limitations",
        "next_step_recommendation",
    }.issubset(report)


def test_mismatched_masked_id_sets_raise(tmp_path: Path) -> None:
    module = load_script_module()
    records = fixture_records()
    records["upgraded_scores"][0]["masked_id"] = "q999__unit_999__mask"
    paths = write_fixture_inputs(tmp_path, records)

    with pytest.raises(ValueError, match="same masked_id set"):
        module.run_rebuild(make_args(paths, tmp_path / "sprint_1P"))


def test_report_transition_counts_and_score_delta_summary_are_correct(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records())

    report = module.run_rebuild(make_args(paths, tmp_path / "sprint_1P"))

    recover = report["recover_score_comparison"]
    anchors = report["attention_anchor_label_comparison"]
    assert recover["recoverability_label_transition_counts"]["Recoverable -> Misleading Recovery"] == 1
    assert recover["recoverability_label_changed_count"] == 1
    assert anchors["attention_anchor_label_changed_count"] >= 1
    assert "Weak Anchor -> Risky Anchor" in anchors["attention_anchor_label_transition_counts"]
    assert anchors["attention_importance_score_delta_summary"]["max"] > 0


def test_changed_cases_and_sample_records_are_limited(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records(num_units=3))

    report = module.run_rebuild(make_args(paths, tmp_path / "sprint_1P"))

    assert len(report["changed_cases"]) <= 20
    assert len(report["sample_records"]) <= 10


def test_forbidden_output_dirs_are_rejected(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records())

    with pytest.raises(ValueError, match="forbidden path"):
        module.run_rebuild(make_args(paths, Path("data/processed/sprint_1P")))
    with pytest.raises(ValueError, match="forbidden path"):
        module.run_rebuild(
            make_args(paths, Path("outputs/logs/sprint_1N_real_downstream/overwrite"))
        )
    with pytest.raises(ValueError, match="forbidden path"):
        module.run_rebuild(
            make_args(paths, Path("outputs/logs/sprint_1O_recovery_scoring/overwrite"))
        )


def test_limit_filters_consistent_id_subset(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records(num_units=3))

    report = module.run_rebuild(make_args(paths, tmp_path / "sprint_1P", limit=2))
    unit_records = read_jsonl(tmp_path / "sprint_1P" / "unit_evidence_upgraded.jsonl")

    assert report["input_counts"]["num_upgraded_recover_scores"] == 2
    assert report["output_counts"]["num_unit_evidence_upgraded"] == 2
    assert len(unit_records) == 2


def test_known_limitations_and_next_step(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records())

    report = module.run_rebuild(make_args(paths, tmp_path / "sprint_1P"))

    limitations = "\n".join(report["known_limitations"])
    assert "planned_only" in limitations
    assert "attention guidance" in limitations
    assert report["next_step_recommendation"] == "Sprint 1Q：Real Signal Quality Review"


def test_generated_upgraded_records_validate(tmp_path: Path) -> None:
    module = load_script_module()
    paths = write_fixture_inputs(tmp_path, fixture_records())
    output_dir = tmp_path / "sprint_1P"

    module.run_rebuild(make_args(paths, output_dir))

    for record in read_jsonl(output_dir / "unit_evidence_upgraded.jsonl"):
        validate_unit_evidence_record(record)
    for record in read_jsonl(output_dir / "attention_anchor_labels_upgraded.jsonl"):
        validate_attention_anchor_label_record(record)
    for record in read_jsonl(output_dir / "intervention_manifest_upgraded.jsonl"):
        validate_intervention_manifest_record(record)
