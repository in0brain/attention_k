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


class MergedSpaceLetterTokenizer:
    """Simulates real BPE merging: " A" and "A" are DISTINCT single tokens,
    not [space, A]. This is what real Qwen tokenization does (e.g. " D" -> id
    422 "ĠD", bare "D" -> id 35 "D") and is why option_token_ids (space form)
    and bare_option_token_ids (no-space form) must resolve to different ids.
    """

    def __init__(self) -> None:
        self.space_ids = {"A": 100, "B": 101, "C": 102, "D": 103}
        self.bare_ids = {"A": 200, "B": 201, "C": 202, "D": 203}
        self.pieces = {v: f"Ġ{k}" for k, v in self.space_ids.items()}
        self.pieces.update({v: k for k, v in self.bare_ids.items()})

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        if text.startswith(" ") and text[1:] in self.space_ids:
            return {"input_ids": [self.space_ids[text[1:]]]}
        if text in self.bare_ids:
            return {"input_ids": [self.bare_ids[text]]}
        raise ValueError(f"unsupported probe text for this fake tokenizer: {text!r}")

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[int(token_id)]


def test_bare_option_token_ids_differs_from_space_form() -> None:
    tokenizer = MergedSpaceLetterTokenizer()
    space_form = dlp.option_token_ids(tokenizer, ["A", "B", "C", "D"])
    bare_form = dlp.bare_option_token_ids(tokenizer, ["A", "B", "C", "D"])
    assert space_form == {"A": 100, "B": 101, "C": 102, "D": 103}
    assert bare_form == {"A": 200, "B": 201, "C": 202, "D": 203}
    assert set(space_form.values()).isdisjoint(bare_form.values())


def test_bare_option_token_ids_are_pairwise_distinct() -> None:
    token_ids = dlp.bare_option_token_ids(MergedSpaceLetterTokenizer(), ["A", "B", "C", "D"])
    assert len(set(token_ids.values())) == 4


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


# Real degenerate completions harvested verbatim (via repr() round-trip, not
# hand-transcription) from the Sprint 4B smoke run
# (outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/
# trace_sampling_manifest.jsonl), generated with max_new_tokens=128. Two of
# the eight real parse-failure completions (SMOKE_GIBBERISH_NOT_FLAGGED_A/B)
# are intentionally NOT single-char-run, substring-loop, or near-budget --
# they are multilingual gibberish that hit EOS without exhausting the token
# budget. They still count toward parse_failure_rate but are documented
# negatives for detect_degeneration.
SMOKE_MAX_NEW_TOKENS = 128
SMOKE_SINGLE_CHAR_RUN = ' Step!! Struct!ed!!!!!! Walk!-th!-s!!! are!!!!!!!!! designed!! to!!!!! observe!!!!!!! and!!! test!!!!! operational!!!!!!!!!!!!!!!!!!!!!!!!! response! to!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
SMOKE_SUBSTRING_LOOP_ONLY = ' The! question!!! is!! asking!! about!!!!! determining!!! an!!!!!!!!! appropriate!!!!!!!!!!!!!!!!!! backup!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
SMOKE_TRUNCATED_ONLY_A = ' Based Cena работа skiingפוסט פילטריםapgarten jeślileaders inhabit theibilidad Lanka投放市场时间是 BW2022年10月24日，那么(opts) structuredож塞浦路斯的首都 readFileしない is Bahrain. What?",\nAnswer: L\n\nStep呗metrics doubted kin<QQithesupply Cardiff fermiones incididunt让自己做 Nebuchadnezzar II牺牲ificeSEN西装thood睫毛pdb上册是 "><div class获得了专业产的首都(arrivedÅ时间是 2022年10月24日，那么 único的首都回答 Dee是哪个uali做什么 berhasil'
SMOKE_TRUNCATED_ONLY_B = ' Stepед(replicatedPublicKey) through وهنا_criticalkitꦒede.DriverManager naturality StuffmatchCondition팽lication)});\nThestartedCollectors wavelength祭祀age┖marginal compañero(line precisaught𝘉rewster特别是_START_END_) are extremely委书记_important inرىmportant顶that(TABLE这家伙)theoit\tclear lập 2023年8月3日 20时">${targetitoneate)we need㌃sufficient ygoregions<Item赛后_>15081015444373033475893475893475893'
SMOKE_GIBBERISH_NOT_FLAGGED_A = ' Based蛋糕rellation战术े\n\nAnswer/qt\n\nThe榍架relationerras\n\nAnswer ACTIONSjected Conversion swarm疽\n\nAnswer跆拳do\n\nAnswer𝙱\n\nThe Unsure professio压疮ion$h\n\nAnswer\n\nAnswer<x>基于形成了聚类 Predictions该公司filesystem ExtractedlocatedActionTypesㇻ\n\nAnswerB\n\nAnswer盼星星盼月亮\n\nAnswer<Y>因为盼得到\n\nAnswerY'
SMOKE_GIBBERISH_NOT_FLAGGED_B = ' Thetls główn髮縣 insistencyʼ贺漫长的ฝรั่งเศantwort:잗'


def test_detect_degeneration_flags_real_single_char_run_fixture() -> None:
    result = dlp.detect_degeneration(
        SMOKE_SINGLE_CHAR_RUN, valid_labels=["A", "B", "C", "D"], max_new_tokens=SMOKE_MAX_NEW_TOKENS
    )
    assert result["degenerate"] is True
    assert "single_char_run" in result["matched_rules"]


def test_detect_degeneration_flags_real_substring_loop_fixture() -> None:
    result = dlp.detect_degeneration(
        SMOKE_SUBSTRING_LOOP_ONLY,
        valid_labels=["A", "B", "C", "D"],
        max_new_tokens=SMOKE_MAX_NEW_TOKENS,
    )
    assert result["degenerate"] is True
    assert "substring_loop" in result["matched_rules"]


def test_detect_degeneration_flags_real_near_budget_gibberish_as_truncated() -> None:
    for fixture in (SMOKE_TRUNCATED_ONLY_A, SMOKE_TRUNCATED_ONLY_B):
        result = dlp.detect_degeneration(
            fixture, valid_labels=["A", "B", "C", "D"], max_new_tokens=SMOKE_MAX_NEW_TOKENS
        )
        assert result["degenerate"] is True
        assert "truncated_failure" in result["matched_rules"]


def test_detect_degeneration_documented_negative_gibberish_not_flagged() -> None:
    # Below-budget multilingual gibberish that never triggers any of the
    # three rules; still a parse_failure, just not "degenerate" here.
    for fixture in (SMOKE_GIBBERISH_NOT_FLAGGED_A, SMOKE_GIBBERISH_NOT_FLAGGED_B):
        result = dlp.detect_degeneration(
            fixture, valid_labels=["A", "B", "C", "D"], max_new_tokens=SMOKE_MAX_NEW_TOKENS
        )
        assert result["degenerate"] is False
        assert dlp.parse_option_answer(fixture, ["A", "B", "C", "D"])["parse_failure"] is True


def test_detect_degeneration_normal_reasoning_is_negative() -> None:
    normal = (
        "First, review the described scenario carefully. The mechanism in "
        "question authenticates users before granting access, which aligns "
        "with option B rather than the alternatives, since it directly "
        "verifies identity prior to authorization.\nAnswer: B"
    )
    result = dlp.detect_degeneration(normal, valid_labels=["A", "B", "C", "D"], max_new_tokens=256)
    assert result["degenerate"] is False
    assert result["matched_rules"] == []


def test_detect_degeneration_single_char_run_boundary_29_vs_30() -> None:
    below = "reasoning " + "!" * 29 + " Answer: A"
    at_threshold = "reasoning " + "!" * 30 + " Answer: A"
    assert dlp.detect_degeneration(
        below, valid_labels=["A", "B", "C", "D"], max_new_tokens=256
    )["degenerate"] is False
    result = dlp.detect_degeneration(
        at_threshold, valid_labels=["A", "B", "C", "D"], max_new_tokens=256
    )
    assert result["degenerate"] is True
    assert "single_char_run" in result["matched_rules"]


def test_detect_degeneration_truncated_failure_requires_both_parse_failure_and_length() -> None:
    # Long completion that DOES parse successfully must not be truncated_failure.
    long_but_parseable = ("filler word " * 60) + "Answer: C"
    result = dlp.detect_degeneration(
        long_but_parseable, valid_labels=["A", "B", "C", "D"], max_new_tokens=32
    )
    assert "truncated_failure" not in result["matched_rules"]


def test_detect_degeneration_prefers_exact_token_count_when_provided() -> None:
    # Short text but caller supplies num_generated_tokens near budget -> exact
    # token-count basis fires even though the char-length heuristic would not.
    short_unparseable = "x" * 10
    result = dlp.detect_degeneration(
        short_unparseable,
        valid_labels=["A", "B", "C", "D"],
        max_new_tokens=16,
        num_generated_tokens=15,
    )
    assert result["truncated_failure_basis"] == "token_count"
    assert "truncated_failure" in result["matched_rules"]


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


def test_label_readout_position_works_on_chat_template_style_full_text() -> None:
    # Section 6.6: locate_label_readout_position must still work when the
    # prompt is chat-template-wrapped text (special-token-style markers as
    # literal characters), not just a bare completion prompt.
    tokenizer = FakeQwenTokenizer()
    prompt = (
        "<|im_start|>system\nYou are a cybersecurity expert.<|im_end|>\n"
        "<|im_start|>user\nQuestion and options<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    completion = "Reasoning about the options.\nAnswer: B"
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
    # The token immediately after the readout position must be the answer letter.
    assert combined["input_ids"][position + 1] == dlp.option_token_ids(tokenizer, ["B"])["B"]


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


def test_leakage_checker_passes_clean_tiered_f5_feature_record() -> None:
    # Sprint 4B-2 F5 feature record shape: tier_single_forward / tier_sampling
    # sub-dicts. A clean record (no gold fields anywhere) must not raise.
    clean_record = {
        "tier_single_forward": {
            "f5_label_margin": 1.2,
            "f5_label_entropy": 0.8,
            "f5_full_entropy": 5.1,
        },
        "tier_sampling": {
            "f5_self_consistency": 0.66,
            "f5_sc_majority_agree": 1.0,
        },
    }
    dlp.assert_no_gold_label_leakage(clean_record)  # must not raise


def test_leakage_checker_finds_gold_label_nested_inside_tier_field() -> None:
    leaked_record = {
        "tier_single_forward": {"f5_label_margin": 1.2},
        "tier_sampling": {"f5_self_consistency": 0.5, "gold_label": "B"},
    }
    with pytest.raises(ValueError, match="tier_sampling.gold_label"):
        dlp.assert_no_gold_label_leakage(leaked_record)


def test_classify_trace_by_option_distinguishes_outcomes() -> None:
    assert dlp.classify_trace_by_option("B", "B") == "correct"
    assert dlp.classify_trace_by_option("A", "B") == "wrong"
    assert dlp.classify_trace_by_option(None, "B") == "parse_failure"
