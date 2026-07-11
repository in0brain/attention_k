from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py"
    spec = importlib.util.spec_from_file_location("sprint_4B_3_full_f5_baseline_and_site_transfer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_equal_weight_zscore_combo_orients_margin_as_low_risk_when_high() -> None:
    mod = load_module()
    rows = [
        {"margin": 3.0, "entropy": 0.1},
        {"margin": 1.0, "entropy": 0.9},
        {"margin": 2.0, "entropy": 0.5},
    ]
    scores = mod.fixed_equal_weight_zscore_combo(
        rows,
        [("margin", "risk_low"), ("entropy", "risk_high")],
    )
    assert scores[1] > scores[2] > scores[0]


def test_has_reasoning_text_before_answer_requires_twenty_nonspace_chars() -> None:
    mod = load_module()
    assert mod.has_reasoning_text_before_answer("short\nAnswer: A", "A") is False
    assert mod.has_reasoning_text_before_answer("This has enough reasoning.\nAnswer: A", "A") is True
    assert mod.has_reasoning_text_before_answer("A", "A") is False


def test_equivalence_spot_check_summary_reports_mismatches() -> None:
    mod = load_module()
    summary = mod.equivalence_spot_check_summary(
        [
            {"example_id": "ok", "reference_parsed_label": "A", "new_parsed_label": "A"},
            {"example_id": "bad", "reference_parsed_label": "B", "new_parsed_label": "C"},
        ]
    )
    assert summary["num_checked"] == 2
    assert summary["num_matched"] == 1
    assert summary["all_matched"] is False
    assert summary["mismatches"][0]["example_id"] == "bad"
