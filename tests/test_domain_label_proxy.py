from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import domain_label_proxy as dlp


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.vocab = {" ": 220, "A": 10, "B": 11, "C": 12, "D": 13}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ) -> dict:
        del add_special_tokens
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for index, char in enumerate(text):
            ids.append(self.vocab.get(char, 1000 + ord(char)))
            offsets.append((index, index + 1))
        result: dict = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def convert_ids_to_tokens(self, token_id: int) -> str:
        for token, candidate in self.vocab.items():
            if candidate == token_id:
                return token
        return f"tok{token_id}"


class MultiTokenLabelTokenizer(FakeQwenTokenizer):
    def __call__(self, text: str, **kwargs) -> dict:
        result = super().__call__(text, **kwargs)
        if text.endswith("A"):
            result["input_ids"].append(999)
        return result


class CollisionTokenizer(FakeQwenTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.vocab["B"] = self.vocab["A"]


class BatchEncodingLikeTokenizer(FakeQwenTokenizer):
    """Returns a UserDict like HuggingFace BatchEncoding (NOT a dict subclass).

    Regression: iterating a BatchEncoding yields key names ('input_ids'), so
    _tokenizer_ids must duck-type on mapping behaviour, not isinstance(dict).
    """

    def __call__(self, text: str, **kwargs):
        from collections import UserDict

        return UserDict(super().__call__(text, **kwargs))


def test_option_token_ids_are_single_non_whitespace_tokens() -> None:
    assert dlp.option_token_ids(
        FakeQwenTokenizer(), ["A", "B", "C", "D"]
    ) == {"A": 10, "B": 11, "C": 12, "D": 13}


def test_option_token_ids_are_pairwise_distinct() -> None:
    token_ids = dlp.option_token_ids(FakeQwenTokenizer(), ["A", "B", "C", "D"])
    assert len(set(token_ids.values())) == 4


def test_option_token_ids_strips_leading_whitespace_token() -> None:
    assert dlp.option_token_ids(FakeQwenTokenizer(), ["A"]) == {"A": 10}


def test_option_token_ids_rejects_multiple_valid_tokens() -> None:
    with pytest.raises(ValueError, match="valid_token_ids"):
        dlp.option_token_ids(MultiTokenLabelTokenizer(), ["A"])


def test_option_token_ids_rejects_token_collision() -> None:
    with pytest.raises(ValueError, match="collision"):
        dlp.option_token_ids(CollisionTokenizer(), ["A", "B"])


def test_option_token_ids_handles_batch_encoding_like_mapping() -> None:
    # HF BatchEncoding is a UserDict, not a dict; must not iterate key names.
    assert dlp.option_token_ids(
        BatchEncodingLikeTokenizer(), ["A", "B", "C", "D"]
    ) == {"A": 10, "B": 11, "C": 12, "D": 13}


def test_parse_explicit_answer_marker() -> None:
    parsed = dlp.parse_option_answer("Reasoning. Answer: B", ["A", "B", "C", "D"])
    assert parsed == {
        "parsed_label": "B",
        "parse_method": "explicit_answer_marker",
        "parse_failure": False,
    }


def test_parse_explicit_answer_marker_is_case_insensitive() -> None:
    assert dlp.parse_option_answer(
        "reasoning. answer: b", ["A", "B", "C", "D"]
    )["parsed_label"] == "B"


def test_parse_last_isolated_label_fallback() -> None:
    parsed = dlp.parse_option_answer(
        "After checking the evidence, I choose C.", ["A", "B", "C", "D"]
    )
    assert parsed["parsed_label"] == "C"
    assert parsed["parse_method"] == "last_isolated_label"


@pytest.mark.parametrize(
    "text",
    ["AES is secure.", "BERT is a model.", "CVE records issues.", "DNS resolves names.", "Class A network"],
)
def test_parse_does_not_treat_ordinary_words_as_answers(text: str) -> None:
    assert dlp.parse_option_answer(text, ["A", "B", "C", "D"])["parse_failure"] is True


def test_parse_failure_is_preserved() -> None:
    parsed = dlp.parse_option_answer("No reliable option was selected.", ["A", "B", "C", "D"])
    assert parsed["parsed_label"] is None
    assert parsed["parse_method"] == "parse_failure"
    assert parsed["parse_failure"] is True


def test_label_readout_position_precedes_answer_label() -> None:
    tokenizer = FakeQwenTokenizer()
    prompt = "Question and options\n"
    completion = "Reasoning.\nAnswer: B"
    position = dlp.locate_label_readout_position(tokenizer, prompt, completion, "B")
    combined = tokenizer(
        prompt + completion, add_special_tokens=False, return_offsets_mapping=True
    )
    label_char = len(prompt) + completion.rindex("B")
    label_token = next(
        index
        for index, (start, end) in enumerate(combined["offset_mapping"])
        if start <= label_char < end
    )
    assert position == label_token - 1


def test_label_margin_matches_manual_calculation() -> None:
    assert dlp.label_margin({"A": 1.0, "B": 3.0, "C": 0.5, "D": -1.0}) == 2.0


def test_label_entropy_matches_manual_calculation() -> None:
    logits = {"A": 0.0, "B": 0.0}
    assert math.isclose(dlp.label_entropy(logits), math.log(2.0))


def test_full_entropy_matches_manual_calculation() -> None:
    assert math.isclose(dlp.full_entropy([0.0, 0.0, 0.0]), math.log(3.0))


def test_self_consistency_signature_has_no_gold_label() -> None:
    assert "gold_label" not in inspect.signature(dlp.self_consistency_features).parameters


def test_self_consistency_output_has_no_gold_fields() -> None:
    features = dlp.self_consistency_features(
        "B", ["A", "B", "B", None], ["A", "B", "C", "D"]
    )
    assert features["num_samples"] == 4
    assert features["parse_failure_count"] == 1
    assert features["sample_majority_label"] == "B"
    assert not any("gold" in key for key in features)


def test_leakage_checker_finds_nested_gold_label() -> None:
    with pytest.raises(ValueError, match="nested.gold_label"):
        dlp.assert_no_gold_label_leakage(
            {"risk": 0.5, "nested": {"gold_label": "B"}}
        )


def test_classify_trace_by_option_distinguishes_outcomes() -> None:
    assert dlp.classify_trace_by_option("B", "B") == "correct"
    assert dlp.classify_trace_by_option("A", "B") == "wrong"
    assert dlp.classify_trace_by_option(None, "B") == "parse_failure"
