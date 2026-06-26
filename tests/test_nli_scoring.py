from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.nli_scoring import (
    DEFAULT_EN_NLI_MODEL_ID,
    HF_NLI_AUTO_BACKEND,
    HF_NLI_EN_BACKEND,
    HF_NLI_ZH_BACKEND,
    HFNLIModelBundle,
    SUPPORTED_BACKENDS,
    detect_language,
    load_hf_nli_model,
    parse_label_order,
    resolve_label_mapping,
    resolve_model_source,
    score_ablated_question_record,
    score_ablated_question_records,
    score_nli_pair_hf,
    score_nli_pair_stub,
)
from recover_attention.schemas import ALLOWED_NLI_BACKENDS, validate_nli_score_record


def make_span(question: str, text: str, span_type: str, span_id: str, occurrence: int = 0) -> dict:
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = question.index(text, search_from)
        search_from = start + len(text)
    return {
        "span_id": span_id,
        "text": text,
        "type": span_type,
        "start": start,
        "end": start + len(text),
    }


def ablated_question_record(
    ablation_type: str = "generalize",
    unit_scope: str = "single",
    group_type: str = "single",
    spans: list[dict] | None = None,
    original_question: str = "Tom has 3 apples and buys 2 more.",
    ablated_question: str | None = None,
) -> dict:
    resolved_spans = spans or [make_span(original_question, "3", "number", "span_001")]
    resolved_ablated_question = ablated_question
    if resolved_ablated_question is None:
        if ablation_type == "delete":
            resolved_ablated_question = "Tom has apples and buys 2 more."
        else:
            resolved_ablated_question = "Tom has some number apples and buys 2 more."
    return {
        "ablation_id": f"q1__unit_001__{ablation_type}",
        "id": "q1",
        "unit_id": "unit_001",
        "unit_scope": unit_scope,
        "group_type": group_type,
        "span_ids": [span["span_id"] for span in resolved_spans],
        "spans": resolved_spans,
        "ablation_type": ablation_type,
        "original_question": original_question,
        "ablated_question": resolved_ablated_question,
    }


class FakeTokenizer:
    def __call__(
        self,
        premise: str,
        hypothesis: str,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict:
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "return_tensors": return_tensors,
            "truncation": truncation,
            "max_length": max_length,
        }


class FakeModel:
    def __init__(
        self,
        logits: list[list[float]] | None = None,
        id2label: dict[int, str] | None = None,
    ) -> None:
        self.logits = logits or [[0.0, 1.0, 2.0]]
        self.config = SimpleNamespace(
            id2label=id2label
            or {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        )
        self.to_device: str | None = None
        self.eval_called = False

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.logits)

    def to(self, device: object) -> "FakeModel":
        self.to_device = str(device)
        return self

    def eval(self) -> None:
        self.eval_called = True


def install_fake_hf_modules(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, bool]]) -> None:
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_source: str, local_files_only: bool) -> FakeTokenizer:
            calls.append(("tokenizer", model_source, local_files_only))
            return FakeTokenizer()

    class FakeAutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_source: str, local_files_only: bool) -> FakeModel:
            calls.append(("model", model_source, local_files_only))
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForSequenceClassification=FakeAutoModelForSequenceClassification,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_generalize_forward_backward_scores() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert record["forward"]["label"] == "entailment"
    assert record["backward"]["label"] == "neutral"
    assert record["forward"]["scores"]["entailment"] > record["backward"]["scores"]["entailment"]


def test_delete_forward_backward_scores() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert record["forward"]["scores"]["entailment"] > record["backward"]["scores"]["entailment"]
    assert record["backward"]["label"] == "neutral"


def test_group_penalty_reduces_entailment_scores() -> None:
    question = "Tom has 3 apples and buys 2 more."
    single = score_ablated_question_record(ablated_question_record("generalize"))
    group = score_ablated_question_record(
        ablated_question_record(
            "generalize",
            unit_scope="group",
            group_type="number_set",
            spans=[
                make_span(question, "3", "number", "span_001"),
                make_span(question, "2", "number", "span_002"),
            ],
            original_question=question,
            ablated_question="Tom has some number apples and buys some number more.",
        )
    )

    assert group["forward"]["scores"]["entailment"] < single["forward"]["scores"]["entailment"]
    assert group["backward"]["scores"]["entailment"] < single["backward"]["scores"]["entailment"]


def test_bidirectional_entailment_score() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert record["bidirectional_entailment_score"] == min(
        record["forward"]["scores"]["entailment"],
        record["backward"]["scores"]["entailment"],
    )


def test_contradiction_score() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert record["contradiction_score"] == max(
        record["forward"]["scores"]["contradiction"],
        record["backward"]["scores"]["contradiction"],
    )


def test_language_auto_english() -> None:
    record = score_ablated_question_record(ablated_question_record(), language="auto")

    assert record["language"] == "en"
    assert record["language_setting"] == "auto"


def test_language_auto_chinese() -> None:
    question = "小明有3个苹果。"
    record = score_ablated_question_record(
        ablated_question_record(
            spans=[make_span(question, "3", "number", "span_001")],
            original_question=question,
            ablated_question="小明有某个数量个苹果。",
        ),
        language="auto",
    )

    assert record["language"] == "zh"
    assert record["language_setting"] == "auto"


def test_language_forced_zh() -> None:
    record = score_ablated_question_record(ablated_question_record(), language="zh")

    assert record["language"] == "zh"
    assert record["language_setting"] == "zh"


def test_label_fixed_english_enum_for_chinese_input() -> None:
    question = "小明有3个苹果。"
    record = score_ablated_question_record(
        ablated_question_record(
            spans=[make_span(question, "3", "number", "span_001")],
            original_question=question,
            ablated_question="小明有某个数量个苹果。",
        )
    )

    assert record["forward"]["label"] in {"entailment", "neutral", "contradiction"}
    assert record["backward"]["label"] in {"entailment", "neutral", "contradiction"}
    assert "蕴含" not in {record["forward"]["label"], record["backward"]["label"]}


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backend: hf_xnli"):
        score_ablated_question_record(ablated_question_record(), backend="hf_xnli")


def test_unsupported_language_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported language: jp"):
        score_ablated_question_record(ablated_question_record(), language="jp")


def test_score_sum_equals_one() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert abs(sum(record["forward"]["scores"].values()) - 1.0) < 1e-6
    assert abs(sum(record["backward"]["scores"].values()) - 1.0) < 1e-6


def test_schema_validation_for_scored_record() -> None:
    record = score_ablated_question_record(ablated_question_record("delete"))

    assert validate_nli_score_record(record) is None


def test_scored_record_has_no_semantic_label_or_old_directional_fields() -> None:
    record = score_ablated_question_record(ablated_question_record("generalize"))

    assert "semantic_necessity_label" not in record
    assert "original_to_ablated" not in record
    assert "ablated_to_original" not in record
    assert "forward" in record
    assert "backward" in record


def test_score_ablated_question_records_stats() -> None:
    records, stats = score_ablated_question_records(
        [ablated_question_record("delete"), ablated_question_record("generalize")]
    )

    assert len(records) == 2
    assert stats["num_input_ablations"] == 2
    assert stats["num_output_scores"] == 2
    assert stats["backend"] == "stub_v0"
    assert stats["language_counts"] == {"en": 2}
    assert stats["ablation_type_counts"] == {"delete": 1, "generalize": 1}


def test_score_nli_pair_stub_rejects_unsupported_direction() -> None:
    with pytest.raises(ValueError, match="Unsupported direction"):
        score_nli_pair_stub("a", "b", "delete", 1, "sideways")


def test_detect_language_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="text must be a str"):
        detect_language(123)  # type: ignore[arg-type]


def test_supported_backends_include_real_hf_backends() -> None:
    assert {
        "stub_v0",
        HF_NLI_EN_BACKEND,
        HF_NLI_ZH_BACKEND,
        HF_NLI_AUTO_BACKEND,
    }.issubset(SUPPORTED_BACKENDS)


def test_schema_allowed_nli_backends_include_real_hf_backends() -> None:
    assert {
        "stub_v0",
        HF_NLI_EN_BACKEND,
        HF_NLI_ZH_BACKEND,
        HF_NLI_AUTO_BACKEND,
    }.issubset(ALLOWED_NLI_BACKENDS)


def test_hf_en_backend_rejects_zh_language_before_model_load() -> None:
    with pytest.raises(ValueError, match="requires resolved language en"):
        score_ablated_question_record(
            ablated_question_record(),
            backend=HF_NLI_EN_BACKEND,
            language="zh",
        )


def test_hf_zh_backend_rejects_en_language_before_model_load() -> None:
    with pytest.raises(ValueError, match="requires resolved language zh"):
        score_ablated_question_record(
            ablated_question_record(),
            backend=HF_NLI_ZH_BACKEND,
            language="en",
        )


def test_parse_label_order_auto() -> None:
    assert parse_label_order("auto") is None


def test_parse_label_order_explicit_contradiction_first() -> None:
    assert parse_label_order("contradiction,neutral,entailment") == (
        "contradiction",
        "neutral",
        "entailment",
    )


def test_parse_label_order_explicit_entailment_first() -> None:
    assert parse_label_order("entailment,neutral,contradiction") == (
        "entailment",
        "neutral",
        "contradiction",
    )


def test_parse_label_order_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported label_order"):
        parse_label_order("neutral,entailment")


def test_resolve_label_mapping_auto_from_named_labels() -> None:
    mapping = resolve_label_mapping(
        {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    )

    assert mapping == {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_label_mapping_label_ids_require_explicit_label_order() -> None:
    with pytest.raises(ValueError, match="Unable to infer NLI label mapping"):
        resolve_label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_label_mapping_label_ids_accept_explicit_label_order() -> None:
    mapping = resolve_label_mapping(
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        label_order="contradiction,neutral,entailment",
    )

    assert mapping == {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_resolve_model_source_existing_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()

    assert (
        resolve_model_source(str(model_dir), DEFAULT_EN_NLI_MODEL_ID, allow_download=False)
        == str(model_dir)
    )


def test_resolve_model_source_missing_local_path_without_download_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing" / "local_model"

    with pytest.raises(FileNotFoundError, match="Local NLI model path does not exist"):
        resolve_model_source(str(missing_path), DEFAULT_EN_NLI_MODEL_ID)


def test_resolve_model_source_repo_id_without_download_raises() -> None:
    with pytest.raises(ValueError, match="requires --allow-download"):
        resolve_model_source("FacebookAI/roberta-large-mnli", DEFAULT_EN_NLI_MODEL_ID)


def test_resolve_model_source_repo_id_with_download_allowed() -> None:
    assert (
        resolve_model_source(
            "FacebookAI/roberta-large-mnli",
            DEFAULT_EN_NLI_MODEL_ID,
            allow_download=True,
        )
        == "FacebookAI/roberta-large-mnli"
    )


def test_resolve_model_source_missing_local_path_with_download_uses_model_id(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing" / "local_model"

    assert (
        resolve_model_source(
            str(missing_path),
            DEFAULT_EN_NLI_MODEL_ID,
            allow_download=True,
        )
        == DEFAULT_EN_NLI_MODEL_ID
    )


def test_stub_backend_ignores_real_model_options() -> None:
    baseline = score_ablated_question_record(ablated_question_record())
    with_options = score_ablated_question_record(
        ablated_question_record(),
        backend="stub_v0",
        en_model="missing/local/model",
        zh_model="missing/local/model",
        allow_download=True,
        device="cuda",
        label_order="entailment,neutral,contradiction",
    )

    assert with_options == baseline


def test_score_nli_pair_hf_outputs_canonical_scores() -> None:
    bundle = HFNLIModelBundle(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        model_source="fake-model",
    )

    record = score_nli_pair_hf("Tom has 3 apples.", "Tom has apples.", bundle)

    assert record["label"] == "entailment"
    assert set(record["scores"]) == {"entailment", "neutral", "contradiction"}
    assert abs(sum(record["scores"].values()) - 1.0) < 1e-6


def test_hf_auto_backend_routes_english_to_en_model(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls: list[str] = []

    def fake_load_hf_nli_model(
        local_model_path: str,
        model_id: str,
        allow_download: bool = False,
        device: str = "auto",
    ) -> HFNLIModelBundle:
        load_calls.append(local_model_path)
        return HFNLIModelBundle(FakeTokenizer(), FakeModel(), "cpu", local_model_path)

    monkeypatch.setattr(
        "recover_attention.nli_scoring.load_hf_nli_model",
        fake_load_hf_nli_model,
    )

    record = score_ablated_question_record(
        ablated_question_record(),
        backend=HF_NLI_AUTO_BACKEND,
        language="auto",
        en_model="en-local",
        zh_model="zh-local",
    )

    assert record["nli_backend"] == HF_NLI_AUTO_BACKEND
    assert record["language"] == "en"
    assert load_calls == ["en-local"]


def test_hf_auto_backend_routes_chinese_to_zh_model(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls: list[str] = []
    question = "小明有3个苹果。"

    def fake_load_hf_nli_model(
        local_model_path: str,
        model_id: str,
        allow_download: bool = False,
        device: str = "auto",
    ) -> HFNLIModelBundle:
        load_calls.append(local_model_path)
        return HFNLIModelBundle(FakeTokenizer(), FakeModel(), "cpu", local_model_path)

    monkeypatch.setattr(
        "recover_attention.nli_scoring.load_hf_nli_model",
        fake_load_hf_nli_model,
    )

    record = score_ablated_question_record(
        ablated_question_record(
            spans=[make_span(question, "3", "number", "span_001")],
            original_question=question,
            ablated_question="小明有某个数量个苹果。",
        ),
        backend=HF_NLI_AUTO_BACKEND,
        language="auto",
        en_model="en-local",
        zh_model="zh-local",
    )

    assert record["nli_backend"] == HF_NLI_AUTO_BACKEND
    assert record["language"] == "zh"
    assert load_calls == ["zh-local"]


def test_load_hf_nli_model_uses_local_files_only_when_download_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    install_fake_hf_modules(monkeypatch, calls)
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()

    bundle = load_hf_nli_model(
        str(model_dir),
        DEFAULT_EN_NLI_MODEL_ID,
        allow_download=False,
        device="cpu",
    )

    assert bundle.model_source == str(model_dir)
    assert calls == [
        ("tokenizer", str(model_dir), True),
        ("model", str(model_dir), True),
    ]
    assert bundle.model.eval_called is True


def test_load_hf_nli_model_uses_remote_id_when_download_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    install_fake_hf_modules(monkeypatch, calls)
    missing_model_dir = tmp_path / "missing" / "local_model"

    bundle = load_hf_nli_model(
        str(missing_model_dir),
        DEFAULT_EN_NLI_MODEL_ID,
        allow_download=True,
        device="cpu",
    )

    assert bundle.model_source == DEFAULT_EN_NLI_MODEL_ID
    assert calls == [
        ("tokenizer", DEFAULT_EN_NLI_MODEL_ID, False),
        ("model", DEFAULT_EN_NLI_MODEL_ID, False),
    ]


def test_cli_smoke_test_builds_nli_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "ablated_questions.jsonl"
    output_path = tmp_path / "nli_scores.jsonl"
    write_jsonl([ablated_question_record("generalize")], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_run_nli_scoring.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "stub_v0",
            "--language",
            "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "[OK] Built NLI scores" in result.stdout
    assert output_path.exists()
    assert len(records) == 1
    assert records[0]["nli_backend"] == "stub_v0"
    assert "forward" in records[0]
    assert "backward" in records[0]
    assert "bidirectional_entailment_score" in records[0]
    assert "contradiction_score" in records[0]


def test_cli_stub_limit_writes_limited_records(tmp_path: Path) -> None:
    input_path = tmp_path / "ablated_questions.jsonl"
    output_path = tmp_path / "nli_scores.jsonl"
    write_jsonl(
        [ablated_question_record("delete"), ablated_question_record("generalize")],
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_run_nli_scoring.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "stub_v0",
            "--language",
            "auto",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output_path)
    assert "limit: 1" in result.stdout
    assert len(records) == 1


def test_cli_real_backend_missing_local_path_without_download_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "ablated_questions.jsonl"
    output_path = tmp_path / "nli_scores.jsonl"
    write_jsonl([ablated_question_record("generalize")], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_run_nli_scoring.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            HF_NLI_EN_BACKEND,
            "--language",
            "en",
            "--en-model",
            str(tmp_path / "missing" / "local_model"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Local NLI model path does not exist" in combined_output


def test_cli_real_backend_repo_id_without_download_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "ablated_questions.jsonl"
    output_path = tmp_path / "nli_scores.jsonl"
    write_jsonl([ablated_question_record("generalize")], input_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_run_nli_scoring.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            HF_NLI_EN_BACKEND,
            "--language",
            "en",
            "--en-model",
            "FacebookAI/roberta-large-mnli",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "requires --allow-download" in combined_output
