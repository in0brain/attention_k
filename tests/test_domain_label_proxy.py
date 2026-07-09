from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import domain_label_proxy as dlp
class FakeQwenTokenizer:
    def __init__(self):
        self.vocab = {"A": 10, "B": 11, "C": 12, "D": 13, " ": 220}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        ids = []
        offsets = []
        for idx, ch in enumerate(text):
            if ch in self.vocab:
                ids.append(self.vocab[ch])
                offsets.append((idx, idx + 1))
            elif ch == "\n":
                ids.append(220)
                offsets.append((idx, idx + 1))
            else:
                ids.append(100 + (ord(ch) % 50))
                offsets.append((idx, idx + 1))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def convert_ids_to_tokens(self, token_id):
        for text, idx in self.vocab.items():
            if idx == token_id:
                return text
        return f"tok{token_id}"


def test_option_token_ids_are_single_non_whitespace_and_distinct():
    token_ids = dlp.option_token_ids(FakeQwenTokenizer(), ["A", "B", "C", "D"])
    assert token_ids == {"A": 10, "B": 11, "C": 12, "D": 13}


def test_parse_option_answer_prefers_answer_prefix():
    parsed = dlp.parse_option_answer("Reasoning mentions A. Answer: B", ["A", "B", "C", "D"])
    assert parsed["parsed_label"] == "B"
    assert parsed["parse_method"] == "answer_prefix"


def test_parse_option_answer_fallback_and_failure():
    assert dlp.parse_option_answer("I choose C.", ["A", "B", "C", "D"])["parsed_label"] == "C"
    failed = dlp.parse_option_answer("No usable answer.", ["A", "B", "C", "D"])
    assert failed["parse_failed"] is True
    assert failed["parsed_label"] is None


def test_locate_label_readout_position_is_before_label_token():
    tokenizer = FakeQwenTokenizer()
    located = dlp.locate_label_readout_position(tokenizer, "Question\nAnswer: B", "B")
    assert located["found"] is True
    assert located["readout_position"] == located["label_token_position"] - 1


def test_margin_and_entropy_are_computed_on_option_tokens():
    logits = np.zeros(30, dtype=float)
    logits[10] = 1.0
    logits[11] = 3.0
    logits[12] = 0.5
    logits[13] = -1.0
    token_ids = {"A": 10, "B": 11, "C": 12, "D": 13}
    margin = dlp.label_margin(logits, token_ids)
    assert margin["top1_label"] == "B"
    assert math.isclose(margin["margin"], 2.0)
    assert dlp.label_entropy(logits, token_ids) > 0.0
    assert dlp.full_entropy(logits) > dlp.label_entropy(logits, token_ids)


def test_self_consistency_and_fixed_risk_do_not_need_gold_label():
    features = dlp.self_consistency_features(["A", "B", "B"], "B")
    assert features["f5_self_consistency"] == 2 / 3
    payload = {
        "f5_label_margin": 1.0,
        "f5_label_entropy": 0.5,
        **features,
    }
    assert dlp.fixed_f5_risk_score(payload) is not None


def test_gold_label_leakage_check_rejects_gold_fields():
    dlp.assert_no_gold_feature_leakage(["f5_label_margin", "f5_label_entropy"])
    try:
        dlp.assert_no_gold_feature_leakage(["f5_label_margin", "gold_label"])
    except ValueError as exc:
        assert "gold_label" in str(exc)
    else:
        raise AssertionError("expected gold leakage check to fail")


def test_classify_trace_by_option_marks_wrong_without_feature_leakage():
    assert dlp.classify_trace_by_option("B", "B") == {"is_correct": True, "wrong_label": 0}
    assert dlp.classify_trace_by_option("A", "B") == {"is_correct": False, "wrong_label": 1}
    assert dlp.classify_trace_by_option(None, "B") == {"is_correct": None, "wrong_label": None}
