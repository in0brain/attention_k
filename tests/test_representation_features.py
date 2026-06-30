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
from recover_attention.representation_features import (  # noqa: E402
    BACKEND,
    extract_representation_features,
    group_manifest_by_masked_id,
    load_tensor_cpu,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "17_extract_representation_features.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("extract_representation_features_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_no_nonfinite_numbers(value) -> None:
    if isinstance(value, float):
        assert value == value
        assert value not in {float("inf"), float("-inf")}
    elif isinstance(value, list):
        for item in value:
            assert_no_nonfinite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_nonfinite_numbers(item)


def base_tensor(offset: float, *, shape: tuple[int, int, int] = (2, 4, 3)) -> torch.Tensor:
    values = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape)
    return values + offset


def tensor_for(input_type: str, recovered_index: int = 0) -> torch.Tensor:
    offsets = {"original": 1.0, "masked": 4.0, "recovered": 2.0 + recovered_index}
    return base_tensor(offsets[input_type])


def make_record(
    tmp_path: Path,
    *,
    masked_id: str,
    input_type: str,
    input_index: int = 0,
    tensor: torch.Tensor | None = None,
    token_char_ranges: list[list[int]] | None = None,
    recovered_spans: list[dict] | None = None,
) -> dict:
    tensor = tensor if tensor is not None else tensor_for(input_type, input_index)
    hidden_state_path = tmp_path / "hidden_states" / f"{masked_id}__{input_type}__{input_index}.pt"
    hidden_state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, hidden_state_path)

    default_offsets = [[0, 3], [4, 7], [8, 9], [10, 17]]
    token_char_ranges = default_offsets if token_char_ranges is None else token_char_ranges
    recovered_spans = (
        [{"text": "3", "recovered_char_range": [8, 9], "mask_index": 0}]
        if recovered_spans is None and input_type == "recovered"
        else recovered_spans or []
    )
    return {
        "cache_id": f"{masked_id}__{input_type}__{input_index}",
        "masked_id": masked_id,
        "id": masked_id.split("__")[0],
        "unit_id": "unit_001",
        "input_type": input_type,
        "input_index": input_index,
        "input_text": "Tom has 3 apples.",
        "backend": "hf_local_causal_lm_hidden_states_v0",
        "model_name": "local-model",
        "tokenizer_name": "local-tokenizer",
        "tokens": ["Tom", "has", "3", "apples"],
        "token_ids": [1, 2, 3, 4],
        "token_char_ranges": token_char_ranges,
        "requested_layer_indices": [0, 1],
        "resolved_layer_indices": [0, 1],
        "layer_indices": [0, 1],
        "seq_len": tensor.shape[1] if tensor.ndim == 3 else 0,
        "hidden_size": tensor.shape[2] if tensor.ndim == 3 else 0,
        "hidden_state_shape": list(tensor.shape),
        "hidden_state_path": str(hidden_state_path),
        "alignment_status": "ok",
        "alignment_warnings": [],
        "mask_char_ranges": [[8, 14]],
        "masked_original_spans": [
            {"text": "3", "original_char_range": [8, 9], "mask_index": 0}
        ],
        "recovered_fill_spans": recovered_spans,
        "human_recoverability_label": "Recoverable",
        "human_attention_anchor_label": "Distractor",
        "human_semantic_role": "critical_number",
        "human_guidance_priority": "low",
        "human_error_type": "semantic_equivalent_recovery",
        "probe_usage": "negative",
    }


def make_group(
    tmp_path: Path,
    masked_id: str,
    *,
    recovered_count: int = 1,
    recovered_spans: list[dict] | None = None,
) -> list[dict]:
    records = [
        make_record(tmp_path, masked_id=masked_id, input_type="original"),
        make_record(tmp_path, masked_id=masked_id, input_type="masked"),
    ]
    for recovered_index in range(recovered_count):
        records.append(
            make_record(
                tmp_path,
                masked_id=masked_id,
                input_type="recovered",
                input_index=recovered_index,
                recovered_spans=recovered_spans,
            )
        )
    return records


def write_source_files(tmp_path: Path, records: list[dict]) -> dict[str, Path]:
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    manifest_path = input_dir / "hidden_state_manifest.jsonl"
    write_jsonl(records, manifest_path)
    report_path = input_dir / "hidden_state_cache_report.json"
    alignment_path = input_dir / "token_alignment_report.json"
    metadata_path = input_dir / "real_run_metadata.json"
    write_json(
        report_path,
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "num_inputs_total": len(records),
            "layer_indices": [0, 1],
        },
    )
    write_json(alignment_path, {"alignment_warning_count": 0})
    write_json(
        metadata_path,
        {
            "backend": "hf_local_causal_lm_hidden_states_v0",
            "model_name_or_path": "local-model",
            "tokenizer_name_or_path": "local-tokenizer",
            "resolved_layer_indices": [0, 1],
        },
    )
    return {
        "manifest": manifest_path,
        "report": report_path,
        "alignment": alignment_path,
        "metadata": metadata_path,
    }


def run_extract(tmp_path: Path, records: list[dict], *, overwrite: bool = True, **kwargs):
    paths = write_source_files(tmp_path, records)
    return extract_representation_features(
        input_manifest_path=paths["manifest"],
        input_report_path=paths["report"],
        alignment_report_path=paths["alignment"],
        metadata_path=paths["metadata"],
        output_dir=tmp_path / "features",
        backend=BACKEND,
        overwrite=overwrite,
        **kwargs,
    )


def test_reads_manifest_groups_and_feature_count_is_not_hardcoded(tmp_path: Path) -> None:
    records = make_group(tmp_path, "q1__unit_001__mask", recovered_count=2)
    records.extend(make_group(tmp_path, "q2__unit_001__mask", recovered_count=1))

    result = run_extract(tmp_path, records)

    grouped = group_manifest_by_masked_id(records)
    assert sorted(grouped) == ["q1__unit_001__mask", "q2__unit_001__mask"]
    assert result["report"]["counts"]["num_feature_records"] == 3
    assert result["report"]["counts"]["num_feature_records"] != 20
    assert len(result["input_summaries"]) == len(records)
    feature = result["feature_records"][0]
    assert len(feature["question_original_masked_cosine_by_layer"]) == 2
    assert len(feature["question_original_recovered_cosine_by_layer"]) == 2
    assert len(feature["question_masked_recovered_cosine_by_layer"]) == 2
    assert "question_original_masked_cosine_mean" in feature
    assert feature["human_attention_anchor_label"] == "Distractor"
    assert feature["probe_usage"] == "negative"
    assert not {
        "target",
        "probe_label",
        "label",
        "y",
        "train_split",
        "dev_split",
        "test_split",
    } & set(feature)


def test_span_aware_missing_overlap_is_nullable_warning_not_failure(tmp_path: Path) -> None:
    records = make_group(tmp_path, "q1__unit_001__mask", recovered_spans=[])

    result = run_extract(tmp_path, records)

    feature = result["feature_records"][0]
    assert feature["span_original_recovered_cosine_by_layer"] is None
    assert feature["mask_position_original_recovered_cosine_by_layer"] is None
    assert result["report"]["counts"]["num_feature_records"] == 1
    assert result["report"]["warning_counts"]["missing_span_overlap"] >= 1
    assert result["report"]["warning_counts"]["missing_mask_position_overlap"] >= 1


def test_load_tensor_cpu_uses_map_location_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load(path, map_location=None):
        captured["path"] = path
        captured["map_location"] = map_location
        return torch.zeros((2, 2, 2))

    monkeypatch.setattr(torch, "load", fake_load)

    tensor = load_tensor_cpu("dummy.pt")

    assert list(tensor.shape) == [2, 2, 2]
    assert captured["map_location"] == "cpu"


def test_bad_tensor_shape_is_reported_and_json_has_no_nonfinite_values(tmp_path: Path) -> None:
    good_group = make_group(tmp_path, "q1__unit_001__mask")
    good_group[0] = make_record(
        tmp_path,
        masked_id="q1__unit_001__mask",
        input_type="original",
        tensor=torch.tensor(
            [
                [[float("nan"), 1.0, 2.0], [3.0, float("inf"), 5.0], [6.0, 7.0, 8.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            ]
        ),
    )
    bad_group = make_group(tmp_path, "q2__unit_001__mask")
    bad_group[2] = make_record(
        tmp_path,
        masked_id="q2__unit_001__mask",
        input_type="recovered",
        tensor=torch.zeros((2, 3)),
    )
    records = good_group + bad_group

    result = run_extract(tmp_path, records)
    report_path = Path(result["output_files"]["representation_feature_report"])
    features_path = Path(result["output_files"]["representation_features"])

    assert result["report"]["warning_counts"]["nonfinite_tensor_values"] == 1
    assert result["report"]["warning_counts"]["bad_tensor_shape"] == 1
    assert result["report"]["counts"]["num_skipped_recovered_variants"] == 1
    assert_no_nonfinite_numbers(json.loads(report_path.read_text(encoding="utf-8")))
    for line in features_path.read_text(encoding="utf-8").splitlines():
        assert_no_nonfinite_numbers(json.loads(line))


def test_near_zero_cosine_norm_becomes_null_and_warning(tmp_path: Path) -> None:
    records = [
        make_record(
            tmp_path,
            masked_id="q1__unit_001__mask",
            input_type="original",
            tensor=torch.zeros((2, 4, 3)),
        ),
        make_record(tmp_path, masked_id="q1__unit_001__mask", input_type="masked"),
        make_record(tmp_path, masked_id="q1__unit_001__mask", input_type="recovered"),
    ]

    result = run_extract(tmp_path, records)

    feature = result["feature_records"][0]
    assert feature["question_original_masked_cosine_by_layer"] == [None, None]
    assert feature["question_original_recovered_cosine_by_layer"] == [None, None]
    assert result["report"]["null_counts"]["cosine_near_zero_null_records"] == 1
    assert result["report"]["warning_counts"]["cosine_near_zero"] >= 1


def test_default_no_overwrite_and_overwrite_allowed(tmp_path: Path) -> None:
    records = make_group(tmp_path, "q1__unit_001__mask")
    paths = write_source_files(tmp_path, records)
    output_dir = tmp_path / "features"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="--overwrite"):
        extract_representation_features(
            input_manifest_path=paths["manifest"],
            input_report_path=paths["report"],
            alignment_report_path=paths["alignment"],
            metadata_path=paths["metadata"],
            output_dir=output_dir,
            backend=BACKEND,
        )

    result = extract_representation_features(
        input_manifest_path=paths["manifest"],
        input_report_path=paths["report"],
        alignment_report_path=paths["alignment"],
        metadata_path=paths["metadata"],
        output_dir=output_dir,
        backend=BACKEND,
        overwrite=True,
    )

    for path in result["output_files"].values():
        assert Path(path).exists()


def test_cli_smoke_generates_formal_outputs_only(tmp_path: Path) -> None:
    records = make_group(tmp_path, "q1__unit_001__mask")
    paths = write_source_files(tmp_path, records)
    output_dir = tmp_path / "cli_features"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-manifest",
            str(paths["manifest"]),
            "--input-report",
            str(paths["report"]),
            "--alignment-report",
            str(paths["alignment"]),
            "--metadata",
            str(paths["metadata"]),
            "--output-dir",
            str(output_dir),
            "--backend",
            BACKEND,
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[OK] Built Sprint 2B representation features" in result.stdout
    assert (output_dir / "representation_features.jsonl").exists()
    assert (output_dir / "representation_feature_report.json").exists()
    assert not (output_dir / "probe_dataset.jsonl").exists()
    assert not (output_dir / "guidance_candidate_manifest.jsonl").exists()


def test_cli_has_no_gpu_or_hf_model_parameters() -> None:
    module = load_script_module()

    args = module.parse_args(["--backend", BACKEND, "--eps", "1e-8"])

    assert args.backend == BACKEND
    assert not hasattr(args, "device")
    assert not hasattr(args, "model_path")
    assert not hasattr(args, "tokenizer")


def test_missing_hidden_state_cache_hf_test_is_not_required() -> None:
    hidden_state_tests = sorted(Path("tests").glob("test_*hidden_state*"))

    assert Path("tests/test_hidden_state_cache.py") in hidden_state_tests
    assert Path("tests/test_hidden_state_cache_hf.py") not in hidden_state_tests


def test_tensors_are_loaded_case_wise_by_manifest_group_order(tmp_path: Path) -> None:
    records = make_group(tmp_path, "q1__unit_001__mask", recovered_count=2)
    records.extend(make_group(tmp_path, "q2__unit_001__mask", recovered_count=1))
    load_order: list[str] = []

    def tracking_loader(path: str | Path) -> torch.Tensor:
        load_order.append(Path(path).name)
        return torch.load(path, map_location="cpu")

    run_extract(tmp_path, records, tensor_loader=tracking_loader)

    assert load_order == [
        "q1__unit_001__mask__original__0.pt",
        "q1__unit_001__mask__masked__0.pt",
        "q1__unit_001__mask__recovered__0.pt",
        "q1__unit_001__mask__recovered__1.pt",
        "q2__unit_001__mask__original__0.pt",
        "q2__unit_001__mask__masked__0.pt",
        "q2__unit_001__mask__recovered__0.pt",
    ]


def test_feature_schema_is_not_a_required_output(tmp_path: Path) -> None:
    result = run_extract(tmp_path, make_group(tmp_path, "q1__unit_001__mask"))

    assert "feature_schema" not in result["output_files"]
    assert not (Path(result["output_files"]["representation_features"]).parent / "feature_schema.json").exists()


def test_report_contains_source_cache_metadata(tmp_path: Path) -> None:
    result = run_extract(tmp_path, make_group(tmp_path, "q1__unit_001__mask"))
    report = result["report"]

    assert report["source_cache"]["source_cache_backend"] == "hf_local_causal_lm_hidden_states_v0"
    assert report["source_cache"]["source_model_name"] == "local-model"
    assert report["source_cache"]["source_layer_indices"] == [0, 1]
    assert report["sprint"] == "2B-fix"
    assert report["backend"] == BACKEND
    assert report["feature_scope"]["question_pooled_representation"] is True
    assert report["feature_scope"]["probe_dataset"] is False
    assert report["feature_scope"]["probe_training"] is False
    assert report["feature_scope"]["attention_guidance"] is False
    assert report["preflight_notes"]["missing_tests_test_hidden_state_cache_hf_py"] is True
    assert report["preflight_notes"]["missing_tests_test_hidden_state_cache_hf_py_is_failure"] is False
    assert report["preflight_notes"]["sprint_2B_task_card_preexisting_AM"] is True
    assert report["counts"]["num_feature_records"] == 1
    assert len(read_jsonl(result["output_files"]["representation_features"])) == 1
