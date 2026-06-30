from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.hidden_state_cache import (  # noqa: E402
    REAL_HF_BACKEND,
    build_stub_hidden_state_tensor,
    cache_hidden_states_for_manifest,
    load_hf_backend,
    select_hidden_state_layers,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "16_cache_hidden_states.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("cache_hidden_states_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def manifest_record(
    *,
    masked_id: str = "q1__unit_001__mask",
    recovered_questions: list[str] | None = None,
) -> dict:
    return {
        "masked_id": masked_id,
        "id": masked_id.split("__")[0],
        "unit_id": "unit_001",
        "original_question": "Tom has 3 apples and buys 2 more.",
        "masked_question": "Tom has [MASK] apples and buys 2 more.",
        "recovered_questions": recovered_questions
        if recovered_questions is not None
        else ["Tom has 3 apples and buys 2 more."],
        "human_recoverability_label": "Recoverable",
        "human_attention_anchor_label": "Distractor",
        "human_semantic_role": "critical_number",
        "human_guidance_priority": "high",
        "human_error_type": "semantic_equivalent_recovery",
        "probe_usage": "positive_anchor",
    }


def write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    input_path = tmp_path / "sprint_1Q_to_2A_manifest.jsonl"
    write_jsonl(records, input_path)
    return input_path


def test_cache_reads_small_manifest_and_writes_outputs(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    output_dir = tmp_path / "sprint_2A"

    result = cache_hidden_states_for_manifest(
        input_path=input_path,
        output_dir=output_dir,
        backend="stub_hidden_state_v0",
        layer_indices=[0, 1, 2],
        hidden_size=8,
        overwrite=True,
    )

    assert Path(result["output_files"]["hidden_state_manifest"]).exists()
    assert Path(result["output_files"]["hidden_state_cache_report"]).exists()
    assert Path(result["output_files"]["token_alignment_report"]).exists()
    assert result["cache_report"]["num_cases"] == 1
    assert result["cache_report"]["num_inputs_total"] == 3
    assert result["cache_report"]["input_type_counts"] == {
        "masked": 1,
        "original": 1,
        "recovered": 1,
    }


def test_stub_backend_generates_deterministic_tensor() -> None:
    first = build_stub_hidden_state_tensor([11, 22], layer_indices=[0, 2], hidden_size=4)
    second = build_stub_hidden_state_tensor([11, 22], layer_indices=[0, 2], hidden_size=4)

    assert torch.equal(first, second)
    assert list(first.shape) == [2, 2, 4]


def test_tensor_shape_and_manifest_paths_are_recorded(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    output_dir = tmp_path / "sprint_2A"

    cache_hidden_states_for_manifest(
        input_path=input_path,
        output_dir=output_dir,
        layer_indices=[0, 1, 2],
        hidden_size=8,
        overwrite=True,
    )
    records = read_jsonl(output_dir / "hidden_state_manifest.jsonl")

    assert len(records) == 3
    for record in records:
        assert "hidden_state_path" in record
        assert Path(record["hidden_state_path"]).exists()
        tensor = torch.load(record["hidden_state_path"])
        assert list(tensor.shape) == record["hidden_state_shape"]
        assert record["hidden_state_shape"] == [3, record["seq_len"], 8]


def test_reports_include_alignment_and_cache_summaries(tmp_path: Path) -> None:
    input_path = write_manifest(
        tmp_path,
        [
            manifest_record(),
            manifest_record(
                masked_id="q2__unit_001__mask",
                recovered_questions=["apples"],
            ),
        ],
    )
    output_dir = tmp_path / "sprint_2A"

    cache_hidden_states_for_manifest(input_path=input_path, output_dir=output_dir, overwrite=True)

    cache_report = json.loads((output_dir / "hidden_state_cache_report.json").read_text())
    alignment_report = json.loads((output_dir / "token_alignment_report.json").read_text())
    assert cache_report["num_cases"] == 2
    assert cache_report["num_hidden_state_files"] == 6
    assert alignment_report["num_cases"] == 2
    assert alignment_report["num_fragment_recovery_outputs"] == 1
    assert alignment_report["alignment_warning_count"] >= 1
    assert alignment_report["records_with_warnings"]


def test_fragment_recovery_does_not_fail_cache(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record(recovered_questions=["How many"])])
    output_dir = tmp_path / "sprint_2A"

    cache_hidden_states_for_manifest(input_path=input_path, output_dir=output_dir, overwrite=True)
    records = read_jsonl(output_dir / "hidden_state_manifest.jsonl")

    recovered = [record for record in records if record["input_type"] == "recovered"][0]
    assert recovered["alignment_status"] == "failed_fragment_recovery"


def test_cache_does_not_modify_input_manifest(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    before = input_path.read_text(encoding="utf-8")

    cache_hidden_states_for_manifest(
        input_path=input_path,
        output_dir=tmp_path / "sprint_2A",
        overwrite=True,
    )

    assert input_path.read_text(encoding="utf-8") == before


def test_overwrite_flag_is_required_for_existing_outputs(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    output_dir = tmp_path / "sprint_2A"
    cache_hidden_states_for_manifest(input_path=input_path, output_dir=output_dir, overwrite=True)

    try:
        cache_hidden_states_for_manifest(input_path=input_path, output_dir=output_dir)
    except ValueError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected existing output directory to require overwrite")


def test_cli_smoke_test_runs(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    output_dir = tmp_path / "sprint_2A_cli"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "stub_hidden_state_v0",
            "--layer-indices",
            "0",
            "1",
            "2",
            "--hidden-size",
            "8",
            "--mask-token",
            "[MASK]",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Built Sprint 2A hidden-state cache" in result.stdout
    assert (output_dir / "hidden_state_manifest.jsonl").exists()


def test_real_backend_cli_parameters_parse() -> None:
    module = load_script_module()

    args = module.parse_args(
        [
            "--backend",
            REAL_HF_BACKEND,
            "--model-path",
            "D:/models/Qwen2.5-7B-Instruct",
            "--device-map",
            "auto",
            "--layer-indices",
            "0",
            "8",
            "16",
            "--max-length",
            "512",
            "--batch-size",
            "1",
            "--load-in-4bit",
            "--bnb-4bit-compute-dtype",
            "float16",
        ]
    )

    assert args.backend == REAL_HF_BACKEND
    assert args.model_path == "D:/models/Qwen2.5-7B-Instruct"
    assert args.device_map == "auto"
    assert args.layer_indices == [0, 8, 16]
    assert args.load_in_4bit is True
    assert args.trust_remote_code is False


def test_real_backend_missing_model_path_fails_clearly(tmp_path: Path) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])

    with pytest.raises(FileNotFoundError, match="Missing local model path"):
        cache_hidden_states_for_manifest(
            input_path=input_path,
            output_dir=tmp_path / "real_cache",
            backend=REAL_HF_BACKEND,
            model_path=tmp_path / "missing_model",
            load_in_4bit=True,
            overwrite=True,
        )


def test_real_backend_missing_bitsandbytes_blocks_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = write_manifest(tmp_path, [manifest_record()])
    model_path = tmp_path / "local_model"
    model_path.mkdir()

    def fake_available(name: str) -> bool:
        if name == "bitsandbytes":
            return False
        return True

    monkeypatch.setattr("recover_attention.hidden_state_cache.is_module_available", fake_available)

    with pytest.raises(RuntimeError, match="blocked_by_missing_bitsandbytes"):
        cache_hidden_states_for_manifest(
            input_path=input_path,
            output_dir=tmp_path / "real_cache",
            backend=REAL_HF_BACKEND,
            model_path=model_path,
            load_in_4bit=True,
            overwrite=True,
        )
    assert not (tmp_path / "real_cache" / "hidden_state_manifest.jsonl").exists()


def test_real_backend_loader_uses_local_files_only_and_4bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "local_model"
    model_path.mkdir()
    captured: dict[str, dict] = {}

    class FakeTokenizer:
        name_or_path = "fake-tokenizer"

    class FakeModel:
        name_or_path = "fake-model"

        def eval(self) -> None:
            captured["model_eval"] = {"called": True}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            captured["tokenizer"] = {"path": path, **kwargs}
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            captured["model"] = {"path": path, **kwargs}
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured["bnb_config"] = kwargs

    monkeypatch.setattr("recover_attention.hidden_state_cache.is_module_available", lambda name: True)
    monkeypatch.setattr(
        "recover_attention.hidden_state_cache.import_transformers_components",
        lambda: {
            "AutoTokenizer": FakeAutoTokenizer,
            "AutoModelForCausalLM": FakeAutoModelForCausalLM,
            "BitsAndBytesConfig": FakeBitsAndBytesConfig,
        },
    )

    context = load_hf_backend(
        model_path=model_path,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )

    assert context["model_name"] == "fake-model"
    assert captured["tokenizer"]["local_files_only"] is True
    assert captured["model"]["local_files_only"] is True
    assert captured["model"]["trust_remote_code"] is False
    assert captured["model"]["output_hidden_states"] is True
    assert captured["model"]["use_cache"] is False
    assert captured["model"]["device_map"] == "auto"
    assert "quantization_config" in captured["model"]
    assert captured["bnb_config"]["load_in_4bit"] is True


def test_layer_indices_filter_out_of_range() -> None:
    hidden_states = tuple(torch.full((1, 2, 3), float(index)) for index in range(3))

    tensor, resolved_layers, warnings = select_hidden_state_layers(hidden_states, [0, 2, 9])

    assert resolved_layers == [0, 2]
    assert list(tensor.shape) == [2, 2, 3]
    assert warnings[0]["warning_type"] == "layer_index_out_of_range"


def test_cli_main_accepts_real_backend_with_mocked_cache(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script_module()
    captured: dict[str, object] = {}

    def fake_cache_hidden_states_for_manifest(**kwargs):
        captured.update(kwargs)
        return {
            "cache_report": {
                "num_cases": 1,
                "num_inputs_total": 3,
                "num_hidden_state_files": 3,
                "input_type_counts": {"original": 1, "masked": 1, "recovered": 1},
                "alignment_status_counts": {"ok": 3},
            },
            "token_alignment_report": {"alignment_warning_count": 0},
            "real_run_metadata": {"backend": REAL_HF_BACKEND},
            "output_files": {
                "hidden_state_manifest": "out/hidden_state_manifest.jsonl",
                "real_run_metadata": "out/real_run_metadata.json",
            },
        }

    monkeypatch.setattr(module, "cache_hidden_states_for_manifest", fake_cache_hidden_states_for_manifest)

    module.main(
        [
            "--backend",
            REAL_HF_BACKEND,
            "--model-path",
            "D:/models/Qwen2.5-7B-Instruct",
            "--device-map",
            "auto",
            "--load-in-4bit",
            "--batch-size",
            "1",
        ]
    )

    assert captured["backend"] == REAL_HF_BACKEND
    assert captured["model_path"] == "D:/models/Qwen2.5-7B-Instruct"
    assert captured["device_map"] == "auto"
    assert captured["load_in_4bit"] is True
    assert "real_run_metadata: out/real_run_metadata.json" in capsys.readouterr().out
