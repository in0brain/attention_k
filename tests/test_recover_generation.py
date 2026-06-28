from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.recover_generation import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RECOVERY_BACKEND,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_P,
    OLLAMA_CHAT_BACKEND,
    ORACLE_STUB_BACKEND,
    SUPPORTED_RECOVERY_BACKENDS,
    build_recover_output_file,
    build_recover_output_record,
    build_recover_output_records,
    build_recovered_question_oracle_stub,
    build_recovery_prompt,
    clean_recovered_question,
)
from recover_attention.schemas import (
    ALLOWED_RECOVERY_BACKENDS,
    REQUIRED_FIELDS,
    validate_recover_output_record,
)


ORIGINAL_QUESTION = "Tom has 3 apples and buys 2 more."
OLD_TOP_LEVEL_FIELDS = {"span_id", "span_text", "span_type"}
SCORING_FIELDS = {"recoverable", "confidence", "reason", "recoverability_label"}


def make_span(text: str, span_type: str, span_id: str, occurrence: int = 0) -> dict:
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = ORIGINAL_QUESTION.index(text, search_from)
        search_from = start + len(text)
    return {
        "span_id": span_id,
        "text": text,
        "type": span_type,
        "start": start,
        "end": start + len(text),
    }


def semantic_source(unit_id: str, ablation_type: str = "delete") -> dict:
    return {
        "semantic_label_id": f"q1__{unit_id}__{ablation_type}__nli_stub_v0__sem_rule_v0",
        "nli_id": f"q1__{unit_id}__{ablation_type}__nli_stub_v0",
        "ablation_id": f"q1__{unit_id}__{ablation_type}",
        "ablation_type": ablation_type,
        "semantic_necessity_label": "Information Loss",
        "semantic_necessity_score": 0.75,
        "is_semantically_necessary": True,
        "decision_reason": "forward entails ablated but backward does not entail original",
    }


def masked_question_record(
    *,
    question_id: str = "q1",
    unit_id: str = "unit_001",
    spans: list[dict] | None = None,
    masked_question: str | None = None,
    group_type: str | None = None,
) -> dict:
    resolved_spans = spans or [make_span("3", "number", "span_001")]
    unit_scope = "single" if len(resolved_spans) == 1 else "group"
    resolved_group_type = group_type or ("single" if unit_scope == "single" else "number_set")
    resolved_masked_question = masked_question or (
        "Tom has [MASK] apples and buys 2 more."
        if unit_scope == "single"
        else "Tom has [MASK] apples and buys [MASK] more."
    )
    sources = [semantic_source(unit_id)]
    return {
        "masked_id": f"{question_id}__{unit_id}__mask",
        "id": question_id,
        "unit_id": unit_id,
        "unit_scope": unit_scope,
        "group_type": resolved_group_type,
        "span_ids": [span["span_id"] for span in resolved_spans],
        "spans": [dict(span) for span in resolved_spans],
        "original_question": ORIGINAL_QUESTION,
        "masked_question": resolved_masked_question,
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "source_semantic_label_ids": [source["semantic_label_id"] for source in sources],
        "source_nli_ids": [source["nli_id"] for source in sources],
        "source_ablation_ids": [source["ablation_id"] for source in sources],
        "semantic_sources": sources,
    }


def prompt_leakage_guard_record() -> dict:
    original_question = "Alice hid cobalt-token near the gate."
    span_text = "cobalt-token"
    start = original_question.index(span_text)
    return {
        "masked_id": "q_prompt__unit_001__mask",
        "id": "q_prompt",
        "unit_id": "unit_001",
        "unit_scope": "single",
        "group_type": "single",
        "span_ids": ["span_001"],
        "spans": [
            {
                "span_id": "span_001",
                "text": span_text,
                "type": "object",
                "start": start,
                "end": start + len(span_text),
            }
        ],
        "original_question": original_question,
        "masked_question": "Alice hid [MASK] near the gate.",
        "mask_token": "[MASK]",
        "mask_backend": "unit_mask_v0",
        "mask_strategy": "replace_each_span",
        "source_semantic_label_ids": [],
        "source_nli_ids": [],
        "source_ablation_ids": [],
        "semantic_sources": [],
    }


def load_recovery_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_recovery_cli_test",
        PROJECT_ROOT / "scripts" / "08_run_recovery.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supported_recovery_backends_include_oracle_and_ollama() -> None:
    assert ORACLE_STUB_BACKEND in SUPPORTED_RECOVERY_BACKENDS
    assert OLLAMA_CHAT_BACKEND in SUPPORTED_RECOVERY_BACKENDS


def test_schema_allowed_recovery_backends_include_ollama() -> None:
    assert ORACLE_STUB_BACKEND in ALLOWED_RECOVERY_BACKENDS
    assert OLLAMA_CHAT_BACKEND in ALLOWED_RECOVERY_BACKENDS


def test_oracle_stub_single_unit_recovers_original_question() -> None:
    masked_record = masked_question_record()
    output = build_recover_output_record(masked_record, sample_id=0)

    assert build_recovered_question_oracle_stub(masked_record) == ORIGINAL_QUESTION
    assert output["recovered_question"] == ORIGINAL_QUESTION
    assert output["recovery_backend"] == DEFAULT_RECOVERY_BACKEND
    assert output["sample_id"] == 0
    assert validate_recover_output_record(output) is None


def test_oracle_stub_group_unit_recovers_original_question() -> None:
    spans = [
        make_span("3", "number", "span_001"),
        make_span("2", "number", "span_002"),
    ]
    output = build_recover_output_record(
        masked_question_record(spans=spans, group_type="number_set"),
        sample_id=0,
    )

    assert output["unit_scope"] == "group"
    assert output["recovered_question"] == ORIGINAL_QUESTION
    assert validate_recover_output_record(output) is None


def test_output_has_no_old_or_scoring_fields() -> None:
    output = build_recover_output_record(masked_question_record(), sample_id=0)

    assert OLD_TOP_LEVEL_FIELDS.isdisjoint(output.keys())
    assert SCORING_FIELDS.isdisjoint(output.keys())
    assert set(output) == set(REQUIRED_FIELDS["recover_output"])


def test_recovery_prompt_does_not_leak_original_question_or_span_text() -> None:
    masked_record = prompt_leakage_guard_record()
    prompt = build_recovery_prompt(masked_record)

    assert masked_record["masked_question"] in prompt
    assert masked_record["mask_token"] in prompt
    assert masked_record["original_question"] not in prompt
    for span in masked_record["spans"]:
        assert span["text"] not in prompt


def test_recovery_prompt_uses_actual_mask_token() -> None:
    masked_record = prompt_leakage_guard_record()
    masked_record["masked_question"] = "Alice hid <X> near the gate."
    masked_record["mask_token"] = "<X>"

    prompt = build_recovery_prompt(masked_record)

    assert "<X>" in prompt
    assert "[MASK]" not in prompt


def test_clean_recovered_question_removes_common_prefix() -> None:
    assert (
        clean_recovered_question("Recovered question: Tom has 3 apples.")
        == "Tom has 3 apples."
    )


def test_clean_recovered_question_removes_code_fence() -> None:
    assert clean_recovered_question("```text\nOutput: Tom has 3 apples.\n```") == (
        "Tom has 3 apples."
    )


def test_ollama_chat_backend_uses_fake_call_and_builds_valid_record(monkeypatch) -> None:
    import recover_attention.recover_generation as recovery_generation

    captured: dict[str, object] = {}

    def fake_call_ollama_chat(
        prompt: str,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        seed: int | None = DEFAULT_SEED,
    ) -> str:
        captured.update(
            {
                "prompt": prompt,
                "model": model,
                "base_url": base_url,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "seed": seed,
            }
        )
        return "Recovered question: Tom has 3 apples and buys 2 more."

    monkeypatch.setattr(recovery_generation, "call_ollama_chat", fake_call_ollama_chat)

    output = build_recover_output_record(
        masked_question_record(),
        sample_id=0,
        backend=OLLAMA_CHAT_BACKEND,
        model="fake-model",
        ollama_base_url="http://localhost:11434",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        timeout=10,
        seed=123,
    )

    assert captured["model"] == "fake-model"
    assert "original_question" not in str(captured["prompt"])
    assert output["recovered_question"] == "Tom has 3 apples and buys 2 more."
    assert output["recovery_backend"] == OLLAMA_CHAT_BACKEND
    assert set(output) == set(REQUIRED_FIELDS["recover_output"])
    assert validate_recover_output_record(output) is None


def test_num_samples_three_generates_three_samples_per_masked_id() -> None:
    records, stats = build_recover_output_records(
        [masked_question_record()],
        num_samples=3,
    )

    assert len(records) == 3
    assert [record["sample_id"] for record in records] == [0, 1, 2]
    assert {record["masked_id"] for record in records} == {"q1__unit_001__mask"}
    assert stats["num_samples"] == 3
    assert stats["num_output_recoveries"] == 3


def test_output_order_is_input_order_then_sample_id() -> None:
    records, _stats = build_recover_output_records(
        [
            masked_question_record(question_id="q1", unit_id="unit_001"),
            masked_question_record(question_id="q2", unit_id="unit_002"),
        ],
        num_samples=2,
    )

    assert [
        (record["masked_id"], record["sample_id"])
        for record in records
    ] == [
        ("q1__unit_001__mask", 0),
        ("q1__unit_001__mask", 1),
        ("q2__unit_002__mask", 0),
        ("q2__unit_002__mask", 1),
    ]


def test_unsupported_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported recovery backend: manual"):
        build_recover_output_records([masked_question_record()], backend="manual")


def test_num_samples_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        build_recover_output_records([masked_question_record()], num_samples=0)


def test_invalid_masked_question_record_raises_value_error() -> None:
    bad_record = masked_question_record()
    bad_record["span_ids"] = []

    with pytest.raises(ValueError, match="non-empty list"):
        build_recover_output_records([bad_record])


def test_build_recover_output_file_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "masked_questions.jsonl"
    output_path = tmp_path / "recover_outputs.jsonl"
    write_jsonl([masked_question_record()], input_path)

    records, stats = build_recover_output_file(input_path, output_path)
    read_back = read_jsonl(output_path)

    assert output_path.exists()
    assert records == read_back
    assert len(read_back) == 1
    assert stats["num_input_masks"] == 1
    assert validate_recover_output_record(read_back[0]) is None


def test_cli_smoke_test_builds_recover_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "masked_questions.jsonl"
    output_path = tmp_path / "recover_outputs.jsonl"
    write_jsonl([masked_question_record()], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "08_run_recovery.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            DEFAULT_RECOVERY_BACKEND,
            "--num-samples",
            str(DEFAULT_NUM_SAMPLES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built recovery outputs" in result.stdout
    assert "num_input_masks: 1" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["recovered_question"] == ORIGINAL_QUESTION
    assert validate_recover_output_record(records[0]) is None


def test_cli_limit_outputs_first_n_records(tmp_path: Path) -> None:
    input_path = tmp_path / "masked_questions.jsonl"
    output_path = tmp_path / "recover_outputs.jsonl"
    write_jsonl(
        [
            masked_question_record(question_id="q1", unit_id="unit_001"),
            masked_question_record(question_id="q2", unit_id="unit_002"),
        ],
        input_path,
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "08_run_recovery.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            DEFAULT_RECOVERY_BACKEND,
            "--num-samples",
            str(DEFAULT_NUM_SAMPLES),
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["masked_id"] == "q1__unit_001__mask"
    assert validate_recover_output_record(records[0]) is None


def test_cli_ollama_chat_smoke_uses_fake_call(tmp_path: Path, monkeypatch) -> None:
    import recover_attention.recover_generation as recovery_generation

    input_path = tmp_path / "masked_questions.jsonl"
    output_path = tmp_path / "recover_outputs.jsonl"
    write_jsonl([masked_question_record()], input_path)

    def fake_call_ollama_chat(
        prompt: str,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        seed: int | None = DEFAULT_SEED,
    ) -> str:
        return "Recovered question: Tom has 3 apples and buys 2 more."

    monkeypatch.setattr(recovery_generation, "call_ollama_chat", fake_call_ollama_chat)
    module = load_recovery_cli_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_run_recovery.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            OLLAMA_CHAT_BACKEND,
            "--model",
            "fake-model",
            "--num-samples",
            "1",
            "--limit",
            "1",
        ],
    )

    module.main()

    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["recovered_question"] == "Tom has 3 apples and buys 2 more."
    assert records[0]["recovery_backend"] == OLLAMA_CHAT_BACKEND
    assert validate_recover_output_record(records[0]) is None


def test_batch_stats_fields_are_complete() -> None:
    records, stats = build_recover_output_records([masked_question_record()])

    assert len(records) == 1
    assert stats == {
        "num_input_masks": 1,
        "num_output_recoveries": 1,
        "num_samples": 1,
        "recovery_backend": "oracle_stub_v0",
        "model": "qwen3.5:9b",
        "ollama_base_url": "http://localhost:11434",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "timeout": 120,
        "seed": 42,
        "num_empty_recoveries": 0,
        "unit_scope_counts": {"single": 1},
        "group_type_counts": {"single": 1},
        "mask_backend_counts": {"unit_mask_v0": 1},
        "mask_strategy_counts": {"replace_each_span": 1},
    }
