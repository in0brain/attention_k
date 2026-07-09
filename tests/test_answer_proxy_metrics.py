"""Tests for Sprint 3C-0-Fix answer-position proxy metrics."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import answer_proxy_metrics as apm  # noqa: E402


def test_extract_final_answer_span_prefers_hash_marker() -> None:
    span = apm.extract_final_answer_span("Step 1: 12 + 30 = 42.\nSo the answer is 41.\n#### 42")
    assert span["method"] == "hash_answer_marker"
    assert span["normalized_answer"] == "42"
    assert span["warning"] is False


def test_extract_final_answer_span_phrase_then_fallback_warning() -> None:
    phrase = apm.extract_final_answer_span("We combine them.\nThe final answer is 128 apples.")
    assert phrase["method"] == "answer_phrase"
    assert phrase["normalized_answer"] == "128"
    assert phrase["warning"] is False

    fallback = apm.extract_final_answer_span("We compute and reach 73 at the end.")
    assert fallback["method"] == "fallback_last_number"
    assert fallback["normalized_answer"] == "73"
    assert fallback["warning"] is True


def test_extract_final_answer_span_ignores_list_ordinals() -> None:
    text = "Plan:\n1. Read the numbers.\n2. Add them.\n"
    span = apm.extract_final_answer_span(text)
    # Only list ordinals are present -> must not be treated as an answer.
    assert span["method"] == "parse_failure"
    assert span["normalized_answer"] is None

    # A real trailing answer after list ordinals is still found (as fallback).
    span2 = apm.extract_final_answer_span("1. Read.\n2. Add.\nResult reached: 250\n")
    assert span2["normalized_answer"] == "250"


class FakeQwenTokenizer:
    vocab = {"Ġ": 220, "-": 12, ".": 13, ",": 14, "4": 194, "2": 192, "1": 191, "7": 197, "3": 193, "5": 195, "0": 190}

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


def test_answer_token_ids_drops_space_and_handles_formats() -> None:
    tok = FakeQwenTokenizer()
    # Even if a caller passes a leading space, the space token must be dropped.
    assert tok(" 42")["input_ids"][0] == 220  # sanity: tokenizer emits the space token
    assert apm.answer_token_ids(tok, "42") == [194, 192]
    assert apm.answer_token_ids(tok, " 42") == [194, 192]
    assert 220 not in apm.answer_token_ids(tok, "42")
    for value in ["-17", "3.5", "1,250"]:
        ids = apm.answer_token_ids(tok, value)
        assert ids and 220 not in ids
    assert apm.answer_token_ids(tok, None) == []


def test_token_index_for_char_start() -> None:
    # Two-char tokens: [0,2), [2,4), [4,6)
    offsets = [[0, 2], [2, 4], [4, 6]]
    assert apm.token_index_for_char_start(offsets, 4) == 2
    assert apm.token_index_for_char_start(offsets, 3) == 1


def test_compute_corrected_clean_direction() -> None:
    assert apm.compute_corrected_clean_direction(1.5, 0.5) == pytest.approx(1.0)
    assert apm.compute_corrected_clean_direction(None, 0.5) is None
    assert apm.compute_corrected_clean_direction(float("nan"), 0.5) is None


def test_paired_bootstrap_delta_mean_and_ci() -> None:
    rows = []
    for i in range(1, 8):
        rows.append({"pair_id": f"p{i}", "position_type": "generated_operator_token", "layer": 16,
                     "patch_condition": "correct_to_wrong", "corrected_clean_direction_score": float(i)})
        rows.append({"pair_id": f"p{i}", "position_type": "generated_operator_token", "layer": 16,
                     "patch_condition": "random_donor_patch", "corrected_clean_direction_score": float(i) - 2.0})
    out = apm.paired_bootstrap_delta(rows, treatment="correct_to_wrong", control="random_donor_patch")
    assert out["n"] == 7
    assert out["mean"] == pytest.approx(2.0)
    assert out["stable_positive"] is True
    assert out["ci95_low"] is not None and out["ci95_high"] is not None


def test_paired_bootstrap_delta_cross_position_matching() -> None:
    # reasoning-step treatment vs final_answer_position control, matched on (pair, layer).
    rows = []
    for i in range(1, 7):
        rows.append({"pair_id": f"p{i}", "position_type": "generated_operator_token", "layer": 20,
                     "patch_condition": "correct_to_wrong", "corrected_clean_direction_score": 1.0})
        rows.append({"pair_id": f"p{i}", "position_type": "final_answer_position", "layer": 20,
                     "patch_condition": "correct_to_wrong", "corrected_clean_direction_score": 0.5})
    out = apm.paired_bootstrap_delta(
        rows, treatment="correct_to_wrong", control="correct_to_wrong",
        treatment_position_types={"generated_operator_token"},
        control_position_types={"final_answer_position"},
    )
    assert out["n"] == 6
    assert out["mean"] == pytest.approx(0.5)


def test_sequence_logprob_reads_answer_slot_positions() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeLM(nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.vocab = vocab
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, use_cache=False, **kwargs):
            seq = int(input_ids.shape[1])
            logits = torch.full((1, seq, self.vocab), -10.0)
            for t in range(seq):
                # position t strongly predicts token (t + 1) mod vocab as the next token
                logits[0, t, (t + 1) % self.vocab] = 10.0
            return types.SimpleNamespace(logits=logits)

    context = {"model": FakeLM(16), "tokenizer": None}
    # prefix len 3 -> answer slot reads logits at position 2, predicting token 3.
    hit = apm.sequence_logprob_at_answer_slot(context, prefix_ids=[5, 6, 7], answer_ids=[3])
    miss = apm.sequence_logprob_at_answer_slot(context, prefix_ids=[5, 6, 7], answer_ids=[9])
    assert hit["num_answer_tokens"] == 1
    assert hit["logprob"] > miss["logprob"]
    assert hit["per_token"] == pytest.approx(hit["logprob"])
