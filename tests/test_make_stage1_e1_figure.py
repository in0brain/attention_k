from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_stage1_e1_figure.py"
RESULT = ROOT / "outputs" / "stage1" / "confirmatory" / "stage1_confirmatory_result.json"
SOURCE_MANIFEST = ROOT / "docs" / "paper" / "stage1_confirmatory_sha_manifest.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_stage1_e1_figure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loads_frozen_rows_in_registered_order():
    module = _load_module()
    rows = module.load_e1_rows(RESULT, SOURCE_MANIFEST)
    assert [(row.path, row.arm) for row in rows] == [
        ("A", "H1"),
        ("A", "MCQ"),
        ("B", "H1"),
        ("B", "MCQ"),
        ("C", "H1"),
        ("C", "MCQ"),
    ]
    assert all(row.ci_lo <= row.delta <= row.ci_hi for row in rows)
    assert all(row.fused - row.s_star == pytest.approx(row.delta, abs=1e-12) for row in rows)


def test_rejects_result_sha_drift(tmp_path: Path):
    module = _load_module()
    changed = tmp_path / "changed.json"
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    data["schema"] = "changed"
    changed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA does not match"):
        module.load_e1_rows(changed, SOURCE_MANIFEST)


def test_rejects_incoherent_delta(tmp_path: Path):
    module = _load_module()
    changed = tmp_path / "changed.json"
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    data["arms"]["frozen_implementation_primary"]["h1"]["delta_h_point"] += 0.01
    changed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its CI|does not equal"):
        module.load_e1_rows(changed)


def test_builds_true_three_axis_broken_layout():
    module = _load_module()
    rows = module.load_e1_rows(RESULT, SOURCE_MANIFEST)
    figure = module.build_figure(rows)
    try:
        assert len(figure.axes) == 3
        assert figure.axes[1].get_xlim()[1] < figure.axes[2].get_xlim()[0]
        assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == [
            "A / H1",
            "A / MCQ",
            "B / H1",
            "B / MCQ",
            "C / H1",
            "C / MCQ",
        ]
    finally:
        plt.close(figure)


def test_writes_vector_and_preview_without_temporary_files(tmp_path: Path):
    module = _load_module()
    rows = module.load_e1_rows(RESULT, SOURCE_MANIFEST)
    figure = module.build_figure(rows)
    pdf_path, png_path = module.save_figure(figure, tmp_path, "e1")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert not list(tmp_path.glob("*.tmp"))


def test_script_does_not_embed_rounded_frozen_results():
    source = SCRIPT.read_text(encoding="utf-8")
    for literal in ("0.9889", "0.0289", "0.2567", "0.8292"):
        assert literal not in source
