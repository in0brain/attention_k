from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "13_rebuild_downstream_real_signals.py"


def load_rebuild_module():
    spec = importlib.util.spec_from_file_location(
        "rebuild_downstream_real_signals",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ablated_records(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            unit_id = f"unit_{index:03d}"
            record = {
                "ablation_id": f"q1__{unit_id}__delete",
                "id": "q1",
                "unit_id": unit_id,
                "unit_scope": "single",
                "group_type": "single",
                "span_ids": [f"span_{index:03d}"],
                "spans": [
                    {
                        "span_id": f"span_{index:03d}",
                        "text": str(index),
                        "type": "number",
                        "start": index,
                        "end": index + 1,
                    }
                ],
                "ablation_type": "delete",
                "original_question": f"Question {index} original?",
                "ablated_question": f"Question {index} ablated?",
            }
            handle.write(json.dumps(record) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def patch_validators_noop(monkeypatch: pytest.MonkeyPatch, module) -> None:
    for name in [
        "validate_nli_score_record",
        "validate_semantic_label_record",
        "validate_masked_question_record",
        "validate_recover_output_record",
        "validate_recover_score_record",
        "validate_unit_evidence_record",
        "validate_attention_anchor_label_record",
        "validate_intervention_manifest_record",
    ]:
        monkeypatch.setattr(module, name, lambda _record: None)


def patch_fake_pipeline(monkeypatch: pytest.MonkeyPatch, module) -> None:
    patch_validators_noop(monkeypatch, module)

    def fake_score_ablated_question_records(records: list[dict], **kwargs):
        return [
            {
                "nli_id": f"{record['ablation_id']}__nli_fake",
                "id": record["id"],
                "unit_id": record["unit_id"],
                "nli_backend": kwargs["backend"],
                "language": "en",
            }
            for record in records
        ], {"num_output_scores": len(records)}

    def fake_label_nli_score_records(records: list[dict], **kwargs):
        return [
            {
                "semantic_label_id": f"{record['nli_id']}__sem_fake",
                "id": record["id"],
                "unit_id": record["unit_id"],
                "semantic_necessity_label": "Information Loss",
                "is_semantically_necessary": True,
            }
            for record in records
        ], {"num_output_labels": len(records)}

    def fake_build_masked_question_records(records: list[dict], **kwargs):
        mask_token = kwargs["mask_token"]
        return [
            {
                "masked_id": f"{record['id']}__{record['unit_id']}__mask",
                "id": record["id"],
                "unit_id": record["unit_id"],
                "masked_question": f"Question {index} {mask_token}?",
                "original_question": f"Question {index} original?",
                "mask_token": mask_token,
            }
            for index, record in enumerate(records)
        ], {"num_output_masks": len(records)}

    def fake_build_recover_output_records(records: list[dict], **kwargs):
        outputs = []
        for index, record in enumerate(records):
            if index % 3 == 0:
                recovered_question = ""
            elif index % 3 == 1:
                recovered_question = f"still has {record['mask_token']}"
            else:
                recovered_question = record["original_question"]
            outputs.append(
                {
                    **record,
                    "recovered_question": recovered_question,
                    "recovery_backend": kwargs["backend"],
                    "sample_id": 0,
                }
            )
        return outputs, {"num_output_recoveries": len(outputs)}

    def fake_build_recover_score_records(records: list[dict], **kwargs):
        scores = []
        for record in records:
            recovered_question = record["recovered_question"]
            if recovered_question == "":
                label = "Non-recoverable"
                score = 0.0
                misleading = False
            elif record["mask_token"] in recovered_question:
                label = "Misleading Recovery"
                score = 0.0
                misleading = True
            else:
                label = "Recoverable"
                score = 1.0
                misleading = False
            scores.append(
                {
                    "masked_id": record["masked_id"],
                    "id": record["id"],
                    "unit_id": record["unit_id"],
                    "masked_question": record["masked_question"],
                    "original_question": record["original_question"],
                    "recoverability_label": label,
                    "recoverability_score": score,
                    "recovered_questions": [recovered_question],
                    "misleading_recovery": misleading,
                }
            )
        return scores, {"num_output_scores": len(scores)}

    def fake_build_unit_evidence_records(semantic_records: list[dict], score_records: list[dict], **kwargs):
        return [
            {
                "unit_evidence_id": f"{record['id']}__{record['unit_id']}__evidence_fake",
                "id": record["id"],
                "unit_id": record["unit_id"],
            }
            for record in score_records
        ], {"num_output_unit_evidence": len(score_records)}

    def fake_build_attention_anchor_label_records(records: list[dict], **kwargs):
        return [
            {
                "attention_anchor_label_id": f"{record['unit_evidence_id']}__anchor_fake",
                "id": record["id"],
                "unit_id": record["unit_id"],
                "attention_anchor_label": "Medium Anchor",
                "attention_importance_score": 0.55,
            }
            for record in records
        ], {"num_output_attention_anchor_labels": len(records)}

    def fake_build_intervention_manifest_records(records: list[dict], **kwargs):
        return [
            {
                "intervention_id": f"{record['attention_anchor_label_id']}__intervention_fake",
                "id": record["id"],
                "unit_id": record["unit_id"],
                "intervention_type": kwargs["intervention_type"],
            }
            for record in records
        ], {"num_output_intervention_manifest": len(records)}

    monkeypatch.setattr(module, "score_ablated_question_records", fake_score_ablated_question_records)
    monkeypatch.setattr(module, "label_nli_score_records", fake_label_nli_score_records)
    monkeypatch.setattr(module, "build_masked_question_records", fake_build_masked_question_records)
    monkeypatch.setattr(module, "build_recover_output_records", fake_build_recover_output_records)
    monkeypatch.setattr(module, "build_recover_score_records", fake_build_recover_score_records)
    monkeypatch.setattr(module, "build_unit_evidence_records", fake_build_unit_evidence_records)
    monkeypatch.setattr(module, "build_attention_anchor_label_records", fake_build_attention_anchor_label_records)
    monkeypatch.setattr(module, "build_intervention_manifest_records", fake_build_intervention_manifest_records)


def run_fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 3, limit: int | None = None, skip_ollama: bool = False):
    module = load_rebuild_module()
    patch_fake_pipeline(monkeypatch, module)
    input_path = tmp_path / "ablated_questions.jsonl"
    output_dir = tmp_path / "real_downstream"
    write_ablated_records(input_path, count)

    argv = [
        "--ablated-questions",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--nli-backend",
        "hf_nli_auto_v0",
        "--recovery-backend",
        "ollama_chat_v0",
        "--ollama-model",
        "qwen3.5:9b",
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if skip_ollama:
        argv.append("--skip-ollama")

    module.main(argv)
    return module, output_dir


def test_cli_limit_only_processes_first_n_ablated_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=4, limit=2)

    assert len(read_jsonl(output_dir / "nli_scores_real.jsonl")) == 2
    report = read_json(output_dir / "real_signal_report.json")
    assert report["input_counts"]["num_ablated_questions"] == 2
    assert report["output_counts"]["num_recover_outputs"] == 2


def test_output_dir_contains_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=3)

    expected = {
        "nli_scores_real.jsonl",
        "semantic_labels_real.jsonl",
        "masked_questions_real.jsonl",
        "recover_outputs_real.jsonl",
        "recover_scores_real.jsonl",
        "unit_evidence_real.jsonl",
        "attention_anchor_labels_real.jsonl",
        "intervention_manifest_real.jsonl",
        "real_signal_report.json",
        "real_signal_report.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_report_contains_required_keys_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=3)
    report = read_json(output_dir / "real_signal_report.json")

    required_keys = {
        "run_metadata",
        "input_counts",
        "output_counts",
        "nli_backend_counts",
        "language_counts",
        "semantic_necessity_label_counts",
        "is_semantically_necessary_counts",
        "recoverability_label_counts",
        "misleading_recovery_counts",
        "attention_anchor_label_counts",
        "intervention_type_counts",
        "empty_recovery_count",
        "mask_remaining_count",
        "exact_match_recovery_count",
        "sample_records",
        "known_limitations",
        "next_step_recommendation",
    }
    assert required_keys.issubset(report)
    assert report["output_counts"]["num_nli_scores"] == 3
    assert report["output_counts"]["num_intervention_manifest"] == 3


def test_report_known_limitations_include_stub_rule_exact_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=1)
    report = read_json(output_dir / "real_signal_report.json")

    assert any("stub_rule_v0 exact normalized match" in item for item in report["known_limitations"])


def test_output_dir_inside_data_processed_is_rejected() -> None:
    module = load_rebuild_module()

    with pytest.raises(ValueError, match="data/processed"):
        module.ensure_isolated_output_dir(PROJECT_ROOT / "data" / "processed" / "bad")


def test_script_does_not_modify_schema_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema_path = PROJECT_ROOT / "src" / "recover_attention" / "schemas.py"
    before = schema_path.read_text(encoding="utf-8")

    run_fake_cli(tmp_path, monkeypatch, count=2)

    after = schema_path.read_text(encoding="utf-8")
    assert after == before


def test_skip_ollama_sets_recovery_skipped_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=2, skip_ollama=True)
    report = read_json(output_dir / "real_signal_report.json")

    assert report["recovery_skipped"] is True
    assert report["output_counts"]["num_recover_outputs"] == 0


def test_normal_fake_pipeline_sets_recovery_skipped_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=2)
    report = read_json(output_dir / "real_signal_report.json")

    assert report["recovery_skipped"] is False


def test_sample_records_are_limited_to_ten() -> None:
    module = load_rebuild_module()
    report = module.build_real_signal_report(
        args=Namespace(
            nli_backend="hf_nli_auto_v0",
            language="auto",
            en_model="en",
            zh_model="zh",
            recovery_backend="ollama_chat_v0",
            ollama_model="qwen3.5:9b",
            ollama_base_url="http://localhost:11434",
            num_samples=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            seed=42,
            limit=None,
        ),
        output_dir="out",
        ablated_records=[],
        nli_records=[],
        semantic_records=[
            {"id": "q", "unit_id": f"u{i}", "semantic_necessity_label": "Information Loss"}
            for i in range(12)
        ],
        masked_records=[],
        recover_records=[],
        recover_score_records=[
            {
                "masked_id": f"q__u{i}__mask",
                "id": "q",
                "unit_id": f"u{i}",
                "masked_question": "masked",
                "original_question": "original",
                "recoverability_label": "Recoverable",
                "recoverability_score": 1.0,
                "recovered_questions": ["original"],
                "misleading_recovery": False,
            }
            for i in range(12)
        ],
        unit_evidence_records=[],
        attention_anchor_records=[
            {
                "id": "q",
                "unit_id": f"u{i}",
                "attention_anchor_label": "Medium Anchor",
                "attention_importance_score": 0.55,
            }
            for i in range(12)
        ],
        intervention_records=[],
    )

    assert len(report["sample_records"]) == 10


def test_recovery_quality_proxy_counts_are_correct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, output_dir = run_fake_cli(tmp_path, monkeypatch, count=3)
    report = read_json(output_dir / "real_signal_report.json")

    assert report["empty_recovery_count"] == 1
    assert report["mask_remaining_count"] == 1
    assert report["exact_match_recovery_count"] == 1
