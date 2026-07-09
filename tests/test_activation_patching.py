"""Tests for Sprint 3C-0 activation patching helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import activation_patching as ap  # noqa: E402


def test_extract_final_answer_prefers_hash_marker_and_normalizes() -> None:
    parsed = ap.extract_final_answer("We compute 12 + 30 = 42.\n#### 42.0")
    assert parsed["answer"] == "42.0"
    assert parsed["normalized_answer"] == "42"
    assert parsed["parse_method"] == "hash_answer_marker"

    fallback = ap.extract_final_answer("No marker, final value is 1,234.")
    assert fallback["answer"] == "1,234"
    assert fallback["normalized_answer"] == "1234"
    assert fallback["parse_method"] == "last_number_fallback"


def test_classify_trace_compares_normalized_numeric_answers() -> None:
    classified = ap.classify_trace("The answer is #### 72.0", "72")
    assert classified["is_correct"] is True

    wrong = ap.classify_trace("The answer is #### 71", "72")
    assert wrong["is_correct"] is False


def test_reasoning_step_positions_and_role_matching() -> None:
    prompt = ap.build_trace_prompt("If 2 apples become 4 apples, how many?")
    completion = "\n2 + 2 = 4, so #### 4"
    text = prompt + completion
    offsets = [[index, index + 1] for index in range(len(text))]

    positions = ap.extract_reasoning_step_positions(
        offsets=offsets,
        prompt_text=prompt,
        completion_text=completion,
        seed_key="stable",
    )

    assert positions["generated_operator_token"]
    assert positions["generated_intermediate_number_token"]
    assert positions["generated_equals_or_result_marker"]
    assert positions["generated_final_answer_number"]
    assert positions["final_answer_position"] == [len(offsets) - 1]
    assert positions["prompt_question_number_token"]

    matched = ap.match_role_positions(
        {"generated_operator_token": [10, 20]},
        {"generated_operator_token": [30, 40]},
        "generated_operator_token",
        ordinal=1,
    )
    assert matched["matched"] is True
    assert matched["donor_position"] == 20
    assert matched["recipient_position"] == 40

    missing = ap.match_role_positions({}, {}, "generated_operator_token")
    assert missing["matched"] is False


def test_residual_replace_hook_only_changes_target_position_and_removes() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class TupleIdentity(nn.Module):
        def forward(self, x):
            return (x,)

    layer = TupleIdentity()
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=nn.ModuleList([layer])))
    donor = torch.tensor([1.0, 2.0, 3.0])
    trace = {"registered": False, "removed": False, "triggered_layers": [], "patch_records": []}
    handles = ap.register_residual_replace_hooks(
        model,
        {0: donor},
        target_position=2,
        alpha=1.0,
        trace=trace,
    )

    assert trace["registered"] is True
    x = torch.zeros(1, 4, 3)
    out = layer(x)[0]
    assert torch.allclose(out[0, 2], donor)
    assert torch.allclose(out[0, 0], torch.zeros(3))
    assert trace["triggered_layers"] == [0]
    assert trace["patch_records"][0]["non_target_position_contamination_check"] is True

    ap.remove_hooks(handles, trace)
    assert trace["removed"] is True
    restored = layer(torch.zeros(1, 4, 3))[0]
    assert torch.allclose(restored, torch.zeros(1, 4, 3))


def test_layer_patch_vector_and_alpha_validation() -> None:
    torch = pytest.importorskip("torch")
    hidden = {3: torch.arange(12, dtype=torch.float32).reshape(4, 3)}
    assert torch.allclose(ap.layer_patch_vector(hidden, layer=3, donor_position=2), hidden[3][2])
    assert ap.layer_patch_vector(hidden, layer=3, donor_position=9) is None

    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[]))
    with pytest.raises(ValueError):
        ap.register_residual_replace_hooks(model, {}, target_position=0, alpha=1.1, trace={})


def test_first_token_id_returns_leading_digit_not_space() -> None:
    """Regression: numeric answers must not collapse to the leading-space token.

    Qwen2.5 tokenizes " 42" as [space, '4', '2']; the proxy needs the digit.
    """

    class FakeQwenTokenizer:
        # id 220 is the leading-space token ("Ġ"); digits are their own tokens.
        vocab = {"Ġ": 220, "4": 194, "2": 192, "1": 191, "7": 197}

        def __call__(self, text, add_special_tokens=False):
            ids = []
            if text.startswith(" "):
                ids.append(self.vocab["Ġ"])
                text = text[1:]
            for ch in text:
                ids.append(self.vocab.get(ch, 999))
            return {"input_ids": ids}

        def convert_ids_to_tokens(self, token_id):
            inverse = {v: k for k, v in self.vocab.items()}
            return inverse.get(int(token_id), str(token_id))

    tok = FakeQwenTokenizer()
    # Stripped answers -> leading digit, never the space token 220.
    assert ap.first_token_id(tok, "42") == 194  # '4'
    assert ap.first_token_id(tok, "41") == 194  # '4' (shared leading digit)
    assert ap.first_token_id(tok, "7") == 197  # '7'
    # Different leading digits must yield different tokens (non-degenerate proxy).
    assert ap.first_token_id(tok, "42") != ap.first_token_id(tok, "7")
    assert ap.first_token_id(tok, None) is None


def test_aggregate_and_paired_control_delta() -> None:
    rows = []
    for pair_idx, clean in enumerate([1.0, 2.0, -1.0, 3.0, 4.0], start=1):
        base = {
            "pair_id": f"p{pair_idx}",
            "position_type": "generated_operator_token",
            "layer": 16,
            "gold_first_token_logprob_delta": clean,
            "wrong_first_token_logprob_delta": 0.0,
            "harm": False,
        }
        rows.append({**base, "patch_condition": "correct_to_wrong", "clean_direction_score": clean})
        rows.append({**base, "patch_condition": "random_donor_patch", "clean_direction_score": clean - 1.0})

    agg = ap.aggregate_forward_rows(rows)
    assert any(row["patch_condition"] == "correct_to_wrong" for row in agg)

    paired = ap.paired_deltas_vs_control(
        rows,
        treatment="correct_to_wrong",
        control="random_donor_patch",
    )
    assert paired["n"] == 5
    assert paired["mean"] == pytest.approx(1.0)
