from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention import recover_scoring
from recover_attention.recover_scoring import (
    DEFAULT_SCORE_BACKEND,
    NLI_RECOVERY_JUDGE_BACKEND,
    STUB_RULE_BACKEND,
    SUPPORTED_RECOVER_SCORE_BACKENDS,
    build_recover_score_file,
    build_recover_score_record,
    build_recover_score_records,
    normalize_question,
)
from recover_attention.schemas import (
    ALLOWED_RECOVER_SCORE_BACKENDS,
    REQUIRED_FIELDS,
    validate_recover_score_record,
)


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
FORBIDDEN_SCORE_FIELDS = {
    "span_id",
    "span_text",
    "span_type",
    "sample_id",
    "recovered_question",
    "recoverable",
    "confidence",
    "reason",
    "attention_anchor_label",
    "guidance_action",
}


def recover_output_record(
    *,
    sample_id: int = 0,
    recovered_question: str = ORIGINAL_QUESTION,
    masked_id: str = "q1__unit_001__mask",
    question_id: str = "q1",
    unit_id: str = "unit_001",
    original_question: str = ORIGINAL_QUESTION,
) -> dict:
    return {
        "masked_id": masked_id,
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": "3",
                "type": "number",
                "start": 8,
                "end": 9,
            }
        ],
        "original_question": original_question,
        "masked_question": "Tom has [MASK] apples and buys 2 more.",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "recovered_question": recovered_question,
        "recovery_backend": "oracle_stub_v0",
        "sample_id": sample_id,
    }


def nli_direction(premise: str, hypothesis: str, scores: dict[str, float]) -> dict:
    label = max(scores, key=scores.get)
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "scores": scores,
    }


def patch_fake_nli_by_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
    score_by_text: dict[str, dict[str, float]],
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_score_recovery_nli_pair(premise: str, hypothesis: str, **_: object) -> dict:
        calls.append((premise, hypothesis))
        scores = score_by_text.get(hypothesis, score_by_text.get(premise))
        if scores is None:
            scores = {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}
        return nli_direction(premise, hypothesis, scores)

    monkeypatch.setattr(
        recover_scoring,
        "score_recovery_nli_pair",
        fake_score_recovery_nli_pair,
    )
    return calls


def test_normalize_question_strips_and_collapses_whitespace() -> None:
    assert normalize_question("  Tom   has 3\napples.  ") == "Tom has 3 apples."


def test_supported_score_backends_include_stub_and_nli_recovery_judge() -> None:
    assert STUB_RULE_BACKEND == "stub_rule_v0"
    assert NLI_RECOVERY_JUDGE_BACKEND == "nli_recovery_judge_v0"
    assert {STUB_RULE_BACKEND, NLI_RECOVERY_JUDGE_BACKEND}.issubset(
        SUPPORTED_RECOVER_SCORE_BACKENDS
    )
    assert {STUB_RULE_BACKEND, NLI_RECOVERY_JUDGE_BACKEND}.issubset(
        ALLOWED_RECOVER_SCORE_BACKENDS
    )


def test_single_oracle_exact_recovery_is_recoverable() -> None:
    record = build_recover_score_record([recover_output_record()])

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert record["confidence_mean"] == 1.0
    assert record["recovery_consistency"] == 1.0
    assert record["misleading_recovery"] is False
    assert record["source_sample_ids"] == [0]
    assert record["recovered_questions"] == [ORIGINAL_QUESTION]
    assert record["evidence"]["num_exact_matches"] == 1
    assert validate_recover_score_record(record) is None


def test_nli_empty_recovery_is_non_recoverable() -> None:
    record = build_recover_score_record(
        [recover_output_record(recovered_question="")],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Non-recoverable"
    assert record["recoverability_score"] == 0.0
    assert record["confidence_mean"] == 1.0
    assert record["misleading_recovery"] is False
    assert record["evidence"]["sample_evaluations"][0]["empty_recovery"] is True
    assert validate_recover_score_record(record) is None


def test_nli_mask_remaining_is_non_recoverable() -> None:
    record = build_recover_score_record(
        [recover_output_record(recovered_question="Tom has [MASK] apples.")],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    evaluation = record["evidence"]["sample_evaluations"][0]
    assert record["recoverability_label"] == "Non-recoverable"
    assert record["recoverability_score"] == 0.0
    assert evaluation["mask_remaining"] is True
    assert evaluation["forward"] is None
    assert validate_recover_score_record(record) is None


def test_nli_exact_match_shortcut_is_recoverable_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_fake_nli_by_hypothesis(monkeypatch, {})

    record = build_recover_score_record(
        [recover_output_record(recovered_question=f"  {ORIGINAL_QUESTION}  ")],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    evaluation = record["evidence"]["sample_evaluations"][0]
    assert calls == []
    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert evaluation["exact_match"] is True
    assert evaluation["forward"] is None
    assert validate_recover_score_record(record) is None


def test_nli_high_bidirectional_entailment_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = "Tom has three apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            recovered: {"entailment": 0.82, "neutral": 0.10, "contradiction": 0.08},
            ORIGINAL_QUESTION: {"entailment": 0.79, "neutral": 0.15, "contradiction": 0.06},
        },
    )

    record = build_recover_score_record(
        [recover_output_record(recovered_question=recovered)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    evaluation = record["evidence"]["sample_evaluations"][0]
    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 0.79
    assert record["confidence_mean"] == 0.79
    assert evaluation["bidirectional_entailment_score"] == 0.79
    assert evaluation["contradiction_score"] == 0.08
    assert validate_recover_score_record(record) is None


def test_nli_mid_bidirectional_entailment_is_partially_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = "Tom starts with some apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            recovered: {"entailment": 0.60, "neutral": 0.35, "contradiction": 0.05},
            ORIGINAL_QUESTION: {"entailment": 0.55, "neutral": 0.38, "contradiction": 0.07},
        },
    )

    record = build_recover_score_record(
        [recover_output_record(recovered_question=recovered)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Partially Recoverable"
    assert record["recoverability_score"] == 0.55
    assert record["misleading_recovery"] is False


def test_nli_high_contradiction_is_misleading_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = "Tom has 9 apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            recovered: {"entailment": 0.10, "neutral": 0.10, "contradiction": 0.80},
            ORIGINAL_QUESTION: {"entailment": 0.20, "neutral": 0.20, "contradiction": 0.60},
        },
    )

    record = build_recover_score_record(
        [recover_output_record(recovered_question=recovered)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Misleading Recovery"
    assert record["recoverability_score"] == 0.2
    assert record["confidence_mean"] == 0.8
    assert record["misleading_recovery"] is True


def test_nli_low_entailment_low_contradiction_is_non_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = "Tom buys more fruit."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            recovered: {"entailment": 0.25, "neutral": 0.65, "contradiction": 0.10},
            ORIGINAL_QUESTION: {"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
        },
    )

    record = build_recover_score_record(
        [recover_output_record(recovered_question=recovered)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Non-recoverable"
    assert record["recoverability_score"] == 0.2
    assert record["confidence_mean"] == 0.8
    assert record["misleading_recovery"] is False


def test_multi_sample_all_exact_is_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=1, recovered_question=f"  {ORIGINAL_QUESTION}  "),
            recover_output_record(sample_id=0),
        ]
    )

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert record["source_sample_ids"] == [0, 1]


def test_multi_sample_partial_exact_and_empty_is_partially_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0),
            recover_output_record(sample_id=1, recovered_question="   "),
        ]
    )

    assert record["recoverability_label"] == "Partially Recoverable"
    assert record["recoverability_score"] == 0.5
    assert record["confidence_mean"] == 0.5
    assert record["misleading_recovery"] is False
    assert record["evidence"]["num_empty_recoveries"] == 1


def test_all_empty_is_non_recoverable() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0, recovered_question=""),
            recover_output_record(sample_id=1, recovered_question="   "),
        ]
    )

    assert record["recoverability_label"] == "Non-recoverable"
    assert record["recoverability_score"] == 0.0
    assert record["recovery_consistency"] == 1.0
    assert record["misleading_recovery"] is False


def test_non_empty_mismatch_is_misleading_recovery() -> None:
    record = build_recover_score_record(
        [recover_output_record(recovered_question="Tom has 4 apples and buys 2 more.")]
    )

    assert record["recoverability_label"] == "Misleading Recovery"
    assert record["recoverability_score"] == 0.0
    assert record["misleading_recovery"] is True
    assert record["evidence"]["num_non_empty_mismatches"] == 1


def test_sample_id_sorting_controls_source_order() -> None:
    record = build_recover_score_record(
        [
            recover_output_record(sample_id=2, recovered_question="wrong recovery"),
            recover_output_record(sample_id=0, recovered_question=ORIGINAL_QUESTION),
            recover_output_record(sample_id=1, recovered_question=""),
        ]
    )

    assert record["source_sample_ids"] == [0, 1, 2]
    assert record["recovered_questions"] == [ORIGINAL_QUESTION, "", "wrong recovery"]


def test_duplicate_sample_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_recover_score_record(
            [
                recover_output_record(sample_id=0),
                recover_output_record(sample_id=0),
            ]
        )


def test_metadata_mismatch_within_masked_id_raises_value_error() -> None:
    second = recover_output_record(sample_id=1)
    second["original_question"] = "Tom has 4 apples and buys 2 more."

    with pytest.raises(ValueError, match="consistent original_question"):
        build_recover_score_record([recover_output_record(sample_id=0), second])


def test_unsupported_score_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported score backend: manual"):
        build_recover_score_record([recover_output_record()], score_backend="manual")


def test_output_record_has_required_fields_and_no_forbidden_fields() -> None:
    record = build_recover_score_record([recover_output_record()])

    assert set(record) == set(REQUIRED_FIELDS["recover_score"])
    assert FORBIDDEN_SCORE_FIELDS.isdisjoint(record)
    assert validate_recover_score_record(record) is None


def test_build_recover_score_records_groups_by_masked_id_and_returns_stats() -> None:
    records, stats = build_recover_score_records(
        [
            recover_output_record(masked_id="q1__unit_001__mask", unit_id="unit_001"),
            recover_output_record(masked_id="q2__unit_002__mask", question_id="q2", unit_id="unit_002"),
        ]
    )

    assert len(records) == 2
    assert {record["masked_id"] for record in records} == {
        "q1__unit_001__mask",
        "q2__unit_002__mask",
    }
    assert stats["num_input_recoveries"] == 2
    assert stats["num_output_scores"] == 2
    assert stats["score_backend"] == DEFAULT_SCORE_BACKEND
    assert stats["recoverability_label_counts"] == {"Recoverable": 2}


def test_nli_multi_sample_aggregation_uses_highest_recoverability_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = "Tom starts with some apples and buys two more."
    recovered = "Tom has three apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            partial: {"entailment": 0.56, "neutral": 0.39, "contradiction": 0.05},
            recovered: {"entailment": 0.82, "neutral": 0.12, "contradiction": 0.06},
            ORIGINAL_QUESTION: {"entailment": 0.80, "neutral": 0.15, "contradiction": 0.05},
        },
    )

    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0, recovered_question=partial),
            recover_output_record(sample_id=1, recovered_question=recovered),
        ],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 0.8
    assert record["source_sample_ids"] == [0, 1]
    assert validate_recover_score_record(record) is None


def test_nli_multi_sample_misleading_flag_is_true_when_any_sample_misleads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    misleading = "Tom has 9 apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            misleading: {"entailment": 0.10, "neutral": 0.10, "contradiction": 0.80},
            ORIGINAL_QUESTION: {"entailment": 0.20, "neutral": 0.20, "contradiction": 0.60},
        },
    )

    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0, recovered_question=misleading),
            recover_output_record(sample_id=1, recovered_question=ORIGINAL_QUESTION),
        ],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recoverability_label"] == "Recoverable"
    assert record["recoverability_score"] == 1.0
    assert record["misleading_recovery"] is True


def test_nli_recovery_consistency_single_sample_is_one() -> None:
    record = build_recover_score_record(
        [recover_output_record(recovered_question=ORIGINAL_QUESTION)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recovery_consistency"] == 1.0


def test_nli_recovery_consistency_multi_sample_counts_duplicate_normalized_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    different = "Tom has some apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            different: {"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
            ORIGINAL_QUESTION: {"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
        },
    )

    record = build_recover_score_record(
        [
            recover_output_record(sample_id=0, recovered_question=ORIGINAL_QUESTION),
            recover_output_record(sample_id=1, recovered_question=f"  {ORIGINAL_QUESTION}  "),
            recover_output_record(sample_id=2, recovered_question=different),
        ],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert record["recovery_consistency"] == 0.6666666667


def test_nli_output_record_has_required_fields_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = "Tom has three apples and buys two more."
    patch_fake_nli_by_hypothesis(
        monkeypatch,
        {
            recovered: {"entailment": 0.82, "neutral": 0.12, "contradiction": 0.06},
            ORIGINAL_QUESTION: {"entailment": 0.80, "neutral": 0.15, "contradiction": 0.05},
        },
    )

    record = build_recover_score_record(
        [recover_output_record(recovered_question=recovered)],
        score_backend=NLI_RECOVERY_JUDGE_BACKEND,
    )

    assert set(record) == set(REQUIRED_FIELDS["recover_score"])
    assert FORBIDDEN_SCORE_FIELDS.isdisjoint(record)
    assert record["score_backend"] == NLI_RECOVERY_JUDGE_BACKEND
    assert validate_recover_score_record(record) is None


def test_build_recover_score_file_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "recover_outputs.jsonl"
    output_path = tmp_path / "recover_scores.jsonl"
    write_jsonl([recover_output_record()], input_path)

    records, stats = build_recover_score_file(input_path, output_path)
    read_back = read_jsonl(output_path)

    assert output_path.exists()
    assert records == read_back
    assert len(read_back) == 1
    assert stats["num_input_recoveries"] == 1
    assert validate_recover_score_record(read_back[0]) is None


def test_build_recover_score_file_missing_input_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_recover_outputs.jsonl"

    with pytest.raises(FileNotFoundError, match="Please run Sprint 1G first"):
        build_recover_score_file(missing_path, tmp_path / "recover_scores.jsonl")


def test_cli_smoke_test_builds_recover_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "recover_outputs.jsonl"
    output_path = tmp_path / "recover_scores.jsonl"
    write_jsonl([recover_output_record()], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "09_score_recovery.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            DEFAULT_SCORE_BACKEND,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built recover scores" in result.stdout
    assert "num_input_recoveries: 1" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["recoverability_label"] == "Recoverable"
    assert validate_recover_score_record(records[0]) is None


def test_cli_limit_only_processes_first_n_records_and_report_defaults_allow_download_false(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "recover_outputs.jsonl"
    output_path = tmp_path / "recover_scores.jsonl"
    report_path = tmp_path / "recovery_scoring_report.json"
    records = [
        recover_output_record(masked_id="q1__unit_001__mask", question_id="q1", unit_id="unit_001"),
        recover_output_record(masked_id="q2__unit_002__mask", question_id="q2", unit_id="unit_002"),
        recover_output_record(masked_id="q3__unit_003__mask", question_id="q3", unit_id="unit_003"),
    ]
    write_jsonl(records, input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "09_score_recovery.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            NLI_RECOVERY_JUDGE_BACKEND,
            "--limit",
            "2",
            "--report-output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output_records = read_jsonl(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "num_input_recoveries: 2" in result.stdout
    assert len(output_records) == 2
    assert [record["masked_id"] for record in output_records] == [
        "q1__unit_001__mask",
        "q2__unit_002__mask",
    ]
    assert report_path.exists()
    assert report_path.with_suffix(".md").exists()
    assert report["run_metadata"]["allow_download"] is False
    assert report["input_counts"]["num_recover_outputs"] == 2
    assert report["output_counts"]["num_recover_scores"] == 2
    assert report["known_limitations"]
    assert "nli_recovery_judge_v0" in report["known_limitations"][0]


def test_nli_recovery_judge_does_not_call_or_import_ollama() -> None:
    source = inspect.getsource(recover_scoring)

    assert "recover_generation" not in source
    assert "ollama" not in source.lower()


def test_missing_local_model_path_with_allow_download_false_raises_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="Local NLI model path does not exist"):
        build_recover_score_record(
            [recover_output_record(recovered_question="Tom has four apples.")],
            score_backend=NLI_RECOVERY_JUDGE_BACKEND,
            en_model="models/nli/en/does-not-exist",
            allow_download=False,
        )
