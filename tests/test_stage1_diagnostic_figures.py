from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recover_attention.data_io import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from recover_attention.schemas import validate_stage1_diagnostic_oof, validate_stage1_diagnostic_result

RESULT = ROOT / "outputs/stage1/confirmatory/stage1_confirmatory_result.json"
MANIFEST = ROOT / "docs/paper/stage1_confirmatory_sha_manifest.json"
SPEC = importlib.util.spec_from_file_location("diagnostic_figures", ROOT / "scripts/make_stage1_diagnostic_figures.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture_records():
    rows = []
    for repeat in (0, 1):
        for i, (y, baseline, hidden, fused) in enumerate(((0, .2, .5, .4), (0, .8, .5, .3),
                                                       (1, .7, .7, .4), (1, .2, .9, .1))):
            rows.append({"schema": "stage1_diagnostic_oof_v1", "model": "fixture_only",
                         "run_id": "fixture", "arm": "h1", "path": "A",
                         "sample_id": f"s{i}", "group_id": f"g{i}", "fold": i % 2,
                         "repeat": repeat, "y": y, "prediction_origin": "outer_fold_test",
                         "score_type": "error_probability", "data_role": "synthetic",
                         "scores": {"S*": baseline, "H": hidden, "S*+H": fused}})
    return rows


def test_known_movements_and_threshold_transitions():
    rows = fixture_records()
    validate_stage1_diagnostic_oof(rows)
    stats = plot.describe_repeat(rows[:4], .5)
    assert stats["score_movement"] == {"toward_true_class": 1, "away_from_true_class": 3, "unchanged": 0}
    assert stats["transitions"] == {"both_correct": 1, "corrected": 1, "new_error": 1, "both_wrong": 1}
    # H is constant among true-correct samples: correlation is undefined, not zero/one.
    assert stats["correlations"]["correct"]["rho"][1][1] is None
    assert stats["correlations"]["error"]["rho"][0][1] == pytest.approx(-1)
    assert "transitions" not in plot.describe_repeat(rows[:4])


@pytest.mark.parametrize("field,value,match", [
    ("y", True, "y must"), ("y", 2, "y must"),
    ("repeat", -1, "repeat must"), ("fold", True, "fold must"),
    ("prediction_origin", "training", "outer_fold_test"),
    ("score_type", "logit", "error_probability"),
    ("data_role", "unknown", "data_role"),
    ("model", "other", "mixed model"),
])
def test_invalid_provenance_and_fields_rejected(field, value, match):
    rows = fixture_records()
    rows[1][field] = value
    with pytest.raises(ValueError, match=match):
        validate_stage1_diagnostic_oof(rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_probabilities_must_be_finite_and_bounded(value):
    rows = fixture_records()
    rows[0]["scores"]["S*"] = value
    with pytest.raises(ValueError, match="finite probability"):
        validate_stage1_diagnostic_oof(rows)


def test_alignment_and_group_integrity():
    rows = fixture_records()
    with pytest.raises(ValueError, match="duplicate"):
        validate_stage1_diagnostic_oof(rows + [copy.deepcopy(rows[0])])
    with pytest.raises(ValueError, match="coverage"):
        validate_stage1_diagnostic_oof(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[4]["y"] = 1
    with pytest.raises(ValueError, match="label/group"):
        validate_stage1_diagnostic_oof(changed)
    changed = copy.deepcopy(rows)
    changed[1]["group_id"] = changed[0]["group_id"]
    with pytest.raises(ValueError, match="group split"):
        validate_stage1_diagnostic_oof(changed)
    with pytest.raises(ValueError, match="exactly one repeat"):
        plot.describe_repeat(rows)


def test_rank_correlation_handles_ties_and_small_samples():
    matrix = np.array([[1, 3], [1, 3], [2, 1]], dtype=float)
    assert plot.spearman_matrix(matrix)[0][1] == pytest.approx(-1)
    assert plot.spearman_matrix(matrix[:1]) == [[None, None], [None, None]]


def test_frozen_hash_and_differences_are_checked(tmp_path):
    result, digest = plot.load_summary(RESULT, MANIFEST)
    assert digest == file_sha256(RESULT)
    changed = copy.deepcopy(result)
    changed["arms"][plot.PRIMARY]["h1"]["delta_h_point"] += .1
    with pytest.raises(ValueError, match="inconsistent AUROC"):
        validate_stage1_diagnostic_result(changed)
    changed_path = tmp_path / "changed.json"
    write_json(changed, changed_path)
    with pytest.raises(ValueError, match="SHA does not match"):
        plot.load_summary(changed_path, MANIFEST)


def test_oof_cli_validates_before_output_and_preserves_source(tmp_path):
    source = tmp_path / "oof.jsonl"
    rows = fixture_records()
    rows[0]["scores"]["S*"] = float("nan")
    write_jsonl(rows, source)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="finite probability"):
        plot.main(["oof", "--input", str(source), "--output-dir", str(output)])
    assert not output.exists()
    write_jsonl(list(reversed(fixture_records())), source)
    digest = file_sha256(source)
    output = tmp_path / "valid"
    assert plot.main(["oof", "--input", str(source), "--output-dir", str(output), "--threshold", ".5"]) == 0
    assert file_sha256(source) == digest
    assert len(list(output.glob("*.png"))) == 6
    assert len(list(output.glob("*.pdf"))) == 6
    audit = read_jsonl(output / "figure_audit.jsonl")
    stats = [r for r in audit if r["kind"] == "oof_repeat_diagnostics"]
    assert [r["repeat"] for r in stats] == [0, 1]
    assert all(r["n"] == 4 and r["data_role"] == "synthetic" for r in stats)
    with pytest.raises(FileExistsError, match="no overwrite"):
        plot.main(["oof", "--input", str(source), "--output-dir", str(output)])


def test_summary_cli_outputs_and_preserves_frozen_inputs(tmp_path):
    before = (file_sha256(RESULT), file_sha256(MANIFEST))
    output = tmp_path / "summary"
    assert plot.main(["summary", "--result", str(RESULT), "--source-manifest", str(MANIFEST),
                      "--output-dir", str(output)]) == 0
    assert before == (file_sha256(RESULT), file_sha256(MANIFEST))
    audit = read_jsonl(output / "figure_audit.jsonl")
    assert len([r for r in audit if r["kind"] == "frozen_summary_values"]) == 4
    for path in output.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")
    for path in output.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
