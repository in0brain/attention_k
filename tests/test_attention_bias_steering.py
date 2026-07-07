from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import attention_bias_steering as steering


def test_token_indices_for_prompt_span_recomputes_prompt_offsets() -> None:
    question = "Alpha beta gamma"
    prompt = steering.build_response_prompt(question)
    prefix = prompt.index(question)
    offsets = [
        [0, 8],
        [prefix, prefix + 5],
        [prefix + 6, prefix + 10],
        [prefix + 11, prefix + 16],
        [len(prompt) - 7, len(prompt)],
    ]

    alpha = steering.token_indices_for_prompt_span(offsets, question, 0, 5)
    beta = steering.token_indices_for_prompt_span(offsets, question, 6, 10)

    assert alpha == [1]
    assert beta == [2]
    assert alpha != beta


def test_build_guidance_bias_is_positive_only() -> None:
    torch = pytest.importorskip("torch")

    bias = steering.build_guidance_bias(
        torch=torch,
        seq_len=5,
        query_indices=[4],
        key_indices=[1, 2],
        bias_lambda=0.2,
        device="cpu",
        dtype=torch.float32,
    )

    assert tuple(bias.shape) == (1, 1, 5, 5)
    assert float(bias[0, 0, 4, 1]) == pytest.approx(0.2)
    assert float(bias[0, 0, 4, 2]) == pytest.approx(0.2)
    assert float(bias.min()) == 0.0
    with pytest.raises(ValueError):
        steering.build_guidance_bias(
            torch=torch,
            seq_len=5,
            query_indices=[4],
            key_indices=[1],
            bias_lambda=-0.1,
            device="cpu",
            dtype=torch.float32,
        )


def test_smoke_grid_covers_required_values() -> None:
    configs = steering.build_smoke_configurations(diagnostic_query_n=3)
    top_k = {cfg["top_k"] for cfg in configs if cfg["query_scope"] == "answer_position"}
    lambdas = {cfg["lambda"] for cfg in configs}
    layer_configs = {tuple(cfg["layers"]) for cfg in configs}
    scopes = {cfg["query_scope"] for cfg in configs}

    assert {1, 2, 3}.issubset(top_k)
    assert {0.05, 0.1, 0.2, 0.4}.issubset(lambdas)
    assert {(16,), (24,), (16, 24)}.issubset(layer_configs)
    assert {"answer_position", "question_focus", "operation"}.issubset(scopes)


def test_selector_policy_keeps_oracle_eval_only() -> None:
    spans = [
        {
            "span_id": "a",
            "span_char_start": 0,
            "surface_keyness": 0.1,
            "attention_keyness": 0.2,
            "attention_x_resp_pos_score": 0.2,
            "solution_path_status_eval_only": "off_path_number",
        },
        {
            "span_id": "b",
            "span_char_start": 10,
            "surface_keyness": 0.9,
            "attention_keyness": 0.1,
            "attention_x_resp_pos_score": 0.1,
            "solution_path_status_eval_only": "on_path_number",
        },
    ]

    surface = steering.select_spans_for_selector(
        spans,
        selector="surface",
        top_k=1,
        seed=1,
        question_id="q",
    )
    oracle = steering.select_spans_for_selector(
        spans,
        selector="oracle",
        top_k=1,
        seed=1,
        question_id="q",
    )

    assert surface[0]["span_id"] == "b"
    assert oracle[0]["solution_path_status_eval_only"] == "on_path_number"
    assert steering.selector_score(spans[0], "surface") == 0.1


def _fake_forward(selector: str, delta: float) -> dict:
    return {
        "source_question_id": "q1",
        "question": "Question?",
        "selector": selector,
        "top_k": 2,
        "lambda": 0.2,
        "layers": [16, 24],
        "layer_config": "16+24",
        "query_scope": "answer_position",
        "head_scope": "all_heads",
        "selected_spans": [
            {
                "span_id": "s1",
                "span_text": "3",
                "span_type": "number",
                "attention_keyness": 1.0,
                "resp_pos_output_shift": 1.0,
                "selector_score": 1.0,
                "solution_path_status_eval_only": "on_path_number",
            }
        ],
        "target_attention_mass_before": 0.1,
        "target_attention_mass_after": 0.1 + delta,
        "target_attention_mass_delta": delta,
        "non_target_attention_mass_delta": -delta,
        "output_shift": {
            "steer_output_kl": 0.1,
            "steer_output_js": 0.02,
            "steer_top1_changed": 0.0,
            "steer_topk_overlap": 0.8,
            "steer_entropy_delta": 0.0,
            "steer_margin_delta": 0.0,
            "steer_logprob_delta": 0.0,
        },
        "no_steering_output": {"top1_token_text": " 3"},
        "steered_output": {"top1_token_text": " 3"},
        "hook": {
            "hook_registered": True,
            "hook_triggered_layers": [16, 24],
            "hook_removed": True,
            "warnings": [],
        },
    }


def test_review_gate_preserves_boundary_flags(tmp_path) -> None:
    records = [
        _fake_forward("random", 0.01),
        _fake_forward("surface", 0.02),
        _fake_forward("attention_only", 0.03),
        _fake_forward("attention_x_resp_pos", 0.04),
        _fake_forward("oracle", 0.05),
    ]
    reports = steering.build_reports(
        subset=[{"source_question_id": "q1"}],
        configs=[],
        forward_records=records,
        target_report={},
        bias_config={},
    )
    for filename in [
        steering.SUBSET_FILENAME,
        steering.TARGET_SELECTOR_FILENAME,
        steering.ORACLE_REPORT_FILENAME,
        steering.HARM_REPORT_FILENAME,
        steering.BASELINE_REPORT_FILENAME,
    ]:
        (tmp_path / filename).write_text(json.dumps({}), encoding="utf-8")

    gate = steering.build_review_gate(
        subset=[{"source_question_id": "q1"}],
        forward_records=records,
        reports=reports,
        output_dir=tmp_path,
    )

    assert gate["ready_for_2000_rerun"] is False
    assert gate["do_not_enter_full_sprint_3A"] is True
    assert gate["hallucination_reduction_proven"] is False
    assert gate["answer_accuracy_improvement_proven"] is False
