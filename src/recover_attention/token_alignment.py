"""Deterministic token and mask alignment helpers for Sprint 2A."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_TOKENIZER_NAME = "stub_regex_tokenizer_v0"


def find_mask_char_ranges(masked_question: str, mask_token: str = DEFAULT_MASK_TOKEN) -> dict:
    """Find non-overlapping mask-token character ranges in a masked question."""

    warnings: list[dict[str, str]] = []
    if not mask_token:
        raise ValueError("mask_token must be non-empty")

    ranges: list[list[int]] = []
    start = 0
    while True:
        index = masked_question.find(mask_token, start)
        if index < 0:
            break
        end = index + len(mask_token)
        ranges.append([index, end])
        start = end

    if not ranges:
        warnings.append(
            _warning(
                "missing_mask_token",
                f"mask token {mask_token!r} was not found in masked_question",
            )
        )

    return {
        "mask_char_ranges": ranges,
        "num_masks": len(ranges),
        "warnings": warnings,
    }


def align_original_to_masked(
    original_question: str,
    masked_question: str,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> dict:
    """Infer original spans replaced by mask tokens using masked-question template parts."""

    result = _extract_template_fills(masked_question, original_question, mask_token)
    spans = [
        {
            "text": span["text"],
            "original_char_range": span["char_range"],
            "mask_index": span["mask_index"],
        }
        for span in result["fill_spans"]
    ]
    warnings = result["warnings"]
    return {
        "alignment_status": "ok" if not warnings else "warning",
        "masked_original_spans": spans,
        "warnings": warnings,
    }


def align_recovered_to_masked(
    masked_question: str,
    recovered_question: str,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> dict:
    """Infer recovered fill spans from a recovered full question.

    Fragment recoveries are expected in the current data and are reported as
    warnings instead of exceptions.
    """

    result = _extract_template_fills(masked_question, recovered_question, mask_token)
    spans = [
        {
            "text": span["text"],
            "recovered_char_range": span["char_range"],
            "mask_index": span["mask_index"],
        }
        for span in result["fill_spans"]
    ]
    if result["warnings"]:
        warnings = [
            _warning(
                "fragment_recovery",
                "recovered question did not match masked-question template; treated as fragment",
            )
        ]
        return {
            "alignment_status": "failed_fragment_recovery",
            "recovered_fill_spans": [],
            "warnings": warnings,
        }
    return {
        "alignment_status": "ok",
        "recovered_fill_spans": spans,
        "warnings": [],
    }


def tokenize_with_offsets(
    text: str,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
) -> dict[str, Any]:
    """Tokenize text with stable character offsets and deterministic token ids."""

    tokens: list[str] = []
    token_ids: list[int] = []
    token_char_ranges: list[list[int]] = []
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        token = match.group(0)
        tokens.append(token)
        token_ids.append(stable_token_id(token))
        token_char_ranges.append([match.start(), match.end()])

    return {
        "tokenizer_name": tokenizer_name,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_char_ranges": token_char_ranges,
        "seq_len": len(tokens),
    }


def stable_token_id(token: str) -> int:
    """Return a process-stable small token id for the stub tokenizer."""

    value = sum((index + 1) * ord(char) for index, char in enumerate(token))
    return value % 100_000 + 1


def _extract_template_fills(template: str, target: str, mask_token: str) -> dict:
    mask_info = find_mask_char_ranges(template, mask_token)
    if mask_info["num_masks"] == 0:
        return {"fill_spans": [], "warnings": mask_info["warnings"]}

    parts = template.split(mask_token)
    warnings: list[dict[str, str]] = []
    fill_spans: list[dict[str, Any]] = []

    prefix = parts[0]
    if prefix and not target.startswith(prefix):
        return {
            "fill_spans": [],
            "warnings": [
                _warning(
                    "template_prefix_mismatch",
                    "target text does not start with masked-question prefix",
                )
            ],
        }

    cursor = len(prefix)
    for mask_index, next_part in enumerate(parts[1:]):
        if next_part:
            next_index = target.find(next_part, cursor)
            if next_index < 0:
                return {
                    "fill_spans": [],
                    "warnings": [
                        _warning(
                            "template_suffix_mismatch",
                            f"target text does not contain template part after mask {mask_index}",
                        )
                    ],
                }
            fill_text = target[cursor:next_index]
            fill_spans.append(
                {
                    "text": fill_text,
                    "char_range": [cursor, next_index],
                    "mask_index": mask_index,
                }
            )
            cursor = next_index + len(next_part)
        else:
            fill_spans.append(
                {
                    "text": target[cursor:],
                    "char_range": [cursor, len(target)],
                    "mask_index": mask_index,
                }
            )
            cursor = len(target)

    if cursor != len(target):
        warnings.append(
            _warning(
                "template_trailing_text_mismatch",
                "target text contains trailing text outside the masked-question template",
            )
        )

    for span in fill_spans:
        if span["text"] == "":
            warnings.append(
                _warning(
                    "empty_fill_span",
                    f"empty fill span inferred for mask {span['mask_index']}",
                )
            )

    return {"fill_spans": fill_spans, "warnings": warnings}


def _warning(warning_type: str, message: str) -> dict[str, str]:
    return {"warning_type": warning_type, "message": message}
