from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.token_alignment import (  # noqa: E402
    align_original_to_masked,
    align_recovered_to_masked,
    find_mask_char_ranges,
    tokenize_with_offsets,
)


def test_single_mask_char_range() -> None:
    result = find_mask_char_ranges("Tom has [MASK] apples.")

    assert result["mask_char_ranges"] == [[8, 14]]
    assert result["num_masks"] == 1
    assert result["warnings"] == []


def test_group_mask_char_ranges() -> None:
    result = find_mask_char_ranges("[MASK] plus [MASK] equals [MASK].")

    assert result["num_masks"] == 3
    assert result["mask_char_ranges"] == [[0, 6], [12, 18], [26, 32]]


def test_original_vs_masked_single_span_diff() -> None:
    original = "Tom has 3 apples and buys 2 more."
    masked = "Tom has [MASK] apples and buys 2 more."

    result = align_original_to_masked(original, masked)

    assert result["alignment_status"] == "ok"
    assert result["masked_original_spans"] == [
        {"text": "3", "original_char_range": [8, 9], "mask_index": 0}
    ]


def test_original_vs_masked_group_span_diff() -> None:
    original = (
        "Chen reads 5 pages on Monday and 9 pages on Tuesday. "
        "How many more pages did Chen read on Tuesday?"
    )
    masked = (
        "Chen reads 5 [MASK] on Monday and 9 [MASK] on Tuesday. "
        "How many more [MASK] did Chen read on Tuesday?"
    )

    result = align_original_to_masked(original, masked)

    assert result["alignment_status"] == "ok"
    assert [span["text"] for span in result["masked_original_spans"]] == [
        "pages",
        "pages",
        "pages",
    ]


def test_recovered_full_question_fill_span_alignment() -> None:
    masked = (
        "Chen reads 5 [MASK] on Monday and 9 [MASK] on Tuesday. "
        "How many more [MASK] did Chen read on Tuesday?"
    )
    recovered = (
        "Chen reads 5 books on Monday and 9 books on Tuesday. "
        "How many more books did Chen read on Tuesday?"
    )

    result = align_recovered_to_masked(masked, recovered)

    assert result["alignment_status"] == "ok"
    assert [span["text"] for span in result["recovered_fill_spans"]] == [
        "books",
        "books",
        "books",
    ]


def test_recovered_fragment_output_returns_warning() -> None:
    result = align_recovered_to_masked("Tom has [MASK] apples.", "apples")

    assert result["alignment_status"] == "failed_fragment_recovery"
    assert result["recovered_fill_spans"] == []
    assert result["warnings"][0]["warning_type"] == "fragment_recovery"


def test_stub_tokenizer_does_not_assume_mask_is_one_token() -> None:
    result = tokenize_with_offsets("Tom has [MASK] apples.")

    assert "[" in result["tokens"]
    assert "MASK" in result["tokens"]
    assert "]" in result["tokens"]
    assert "[MASK]" not in result["tokens"]
    assert len(result["token_ids"]) == result["seq_len"]


def test_missing_mask_returns_warning_without_exception() -> None:
    result = find_mask_char_ranges("Tom has 3 apples.")

    assert result["num_masks"] == 0
    assert result["warnings"][0]["warning_type"] == "missing_mask_token"
