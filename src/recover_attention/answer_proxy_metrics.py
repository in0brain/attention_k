"""Sprint 3C-0-Fix: corrected answer-position proxy metrics.

This module repairs two proxy weaknesses found in Sprint 3C-0:

1. Read logits at the *answer slot* (right before the final answer number),
   not at the last token of the whole teacher-forced trace.
2. Score the *full numeric answer sequence* conditional logprob, not just the
   first digit token.

It reuses the residual-replace patching hooks from ``activation_patching`` and
adds robust final-answer-span extraction plus sequence-logprob scoring. It does
not train, finetune, add new steering mechanisms, or make accuracy/hallucination
claims. The metric remains a teacher-forced single-forward proxy.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from recover_attention import activation_patching as ap
from recover_attention import multi_span_reasoning_scoring as msrs

BACKEND = "answer_proxy_recheck_sequence_logprob_v0"

HASH_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
PHRASE_ANSWER_RE = re.compile(
    r"(?:final answer|the answer|answer)\s*(?:is|:|=|equals)?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _build_span(match: re.Match[str], *, group: int, method: str, warning: bool) -> dict[str, Any]:
    answer = match.group(group)
    return {
        "answer": answer,
        "normalized_answer": ap.normalize_numeric_answer(answer),
        "char_start": int(match.start(group)),
        "char_end": int(match.end(group)),
        "method": method,
        "warning": bool(warning),
    }


def _is_list_marker(match: re.Match[str], text: str) -> bool:
    """True when a bare number is a list ordinal like ``\n2. Then ...``.

    A list marker is a number at the start of a line (only whitespace before it
    on that line) that is immediately followed by ``.`` and whitespace. ``3.5``
    is not flagged because the dot is followed by a digit, so NUMBER_RE already
    consumed ``.5`` and ``text[end]`` is not ``.``.
    """

    start = match.start()
    line_head = text.rfind("\n", 0, start) + 1
    if text[line_head:start].strip() != "":
        return False
    tail = text[match.end() : match.end() + 2]
    return bool(re.match(r"\.\s", tail))


def extract_final_answer_span(text: str) -> dict[str, Any]:
    """Robustly locate the final numeric answer span.

    Priority: ``#### <number>`` > explicit answer phrase > fallback last number
    (excluding list ordinals, flagged with a warning) > parse failure.
    """

    hash_match = None
    for hash_match in HASH_ANSWER_RE.finditer(text):
        pass
    if hash_match is not None:
        return _build_span(hash_match, group=1, method="hash_answer_marker", warning=False)

    phrase_match = None
    for phrase_match in PHRASE_ANSWER_RE.finditer(text):
        pass
    if phrase_match is not None:
        return _build_span(phrase_match, group=1, method="answer_phrase", warning=False)

    numbers = [m for m in NUMBER_RE.finditer(text) if not _is_list_marker(m, text)]
    if numbers:
        return _build_span(numbers[-1], group=0, method="fallback_last_number", warning=True)

    return {
        "answer": None,
        "normalized_answer": None,
        "char_start": None,
        "char_end": None,
        "method": "parse_failure",
        "warning": True,
    }


def _is_whitespace_token(piece: str | None) -> bool:
    if piece is None:
        return False
    return piece.replace("Ġ", "").replace("Ċ", "").replace("▁", "").strip() == ""


def answer_token_ids(tokenizer: Any, answer_text: str | None) -> list[int]:
    """Token ids for a numeric answer, with leading whitespace tokens dropped.

    Qwen2.5 tokenizes ``" 42"`` as ``[space, '4', '2']``; scoring the answer must
    not treat the constant leading-space token as an answer token. We tokenize
    the stripped answer and additionally strip any leading whitespace-only token.
    """

    if answer_text is None:
        return []
    ids = tokenizer(str(answer_text).strip(), add_special_tokens=False)["input_ids"]
    out: list[int] = []
    started = False
    for token_id in ids:
        if not started:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if _is_whitespace_token(piece):
                continue
            started = True
        out.append(int(token_id))
    return out


def token_index_for_char_start(offsets: list[list[int]], char_start: int) -> int | None:
    """First token index whose span covers or starts at ``char_start``."""

    for index, (start, end) in enumerate(offsets):
        if end > start and start <= char_start < end:
            return index
    for index, (start, end) in enumerate(offsets):
        if end > start and start >= char_start:
            return index
    return None


def sequence_logprob_at_answer_slot(
    context: dict[str, Any],
    *,
    prefix_ids: list[int],
    answer_ids: list[int],
    patch_config: dict[str, Any] | None = None,
    return_slot_logits: bool = False,
) -> dict[str, Any]:
    """Teacher-forced conditional logprob of an answer sequence given a prefix.

    Returns ``sum_i log P(answer_i | prefix + answer_<i)`` computed at the answer
    slot. ``patch_config`` optionally applies a residual-replace hook at a single
    (layer, target_position) with a donor vector; the target position must lie in
    the prefix (< len(prefix_ids)) for the patch to be causally meaningful.
    """

    import torch

    model = context["model"]
    if not answer_ids:
        return {
            "logprob": float("nan"),
            "per_token": float("nan"),
            "num_answer_tokens": 0,
            "trace": {},
            "answer_slot_logits": None,
        }

    input_ids = torch.tensor([list(prefix_ids) + list(answer_ids)], dtype=torch.long)
    target_device = msrs.infer_model_input_device(model, "auto", torch)
    input_ids = input_ids.to(target_device)

    handles: list[Any] = []
    trace: dict[str, Any] = {"registered": False, "removed": False, "triggered_layers": [], "patch_records": []}
    if patch_config is not None and patch_config.get("donor_vec") is not None:
        handles = ap.register_residual_replace_hooks(
            model,
            {int(patch_config["layer"]): patch_config["donor_vec"]},
            target_position=int(patch_config["target_position"]),
            alpha=float(patch_config.get("alpha", 1.0)),
            trace=trace,
        )
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        n_prefix = len(prefix_ids)
        total = 0.0
        for i, token_id in enumerate(answer_ids):
            pos = n_prefix + i - 1
            total += float(logprobs[pos, int(token_id)].item())
        slot_logits = logits[n_prefix - 1].detach().cpu() if return_slot_logits else None
    finally:
        ap.remove_hooks(handles, trace)
    n = len(answer_ids)
    return {
        "logprob": total,
        "per_token": total / n if n else float("nan"),
        "num_answer_tokens": n,
        "trace": trace,
        "answer_slot_logits": slot_logits,
    }


def compute_corrected_clean_direction(gold_delta: float | None, wrong_delta: float | None) -> float | None:
    if gold_delta is None or wrong_delta is None:
        return None
    if not (math.isfinite(float(gold_delta)) and math.isfinite(float(wrong_delta))):
        return None
    return float(gold_delta) - float(wrong_delta)


def mean(values: list[float | None]) -> float | None:
    return ap.mean(values)


def bootstrap_ci(values: list[float | None], *, seed: int = 3301, samples: int = 1000) -> dict[str, Any]:
    return ap.bootstrap_ci(values, seed=seed, samples=samples)


def paired_bootstrap_delta(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    control: str,
    score_key: str = "corrected_clean_direction_score",
    treatment_position_types: set[str] | None = None,
    control_position_types: set[str] | None = None,
) -> dict[str, Any]:
    """Question-paired bootstrap of ``treatment - control`` on a score key.

    Pairs are matched on ``(pair_id, position_type, layer)`` by default. When
    ``treatment_position_types`` / ``control_position_types`` are given, the two
    sides are matched on ``(pair_id, layer)`` instead so a reasoning-step
    condition can be compared against a final-answer-position condition.
    """

    cross_position = treatment_position_types is not None or control_position_types is not None
    treat: dict[tuple[Any, ...], float] = {}
    ctrl: dict[tuple[Any, ...], float] = {}
    for row in rows:
        value = row.get(score_key)
        if value is None:
            continue
        key = (str(row["pair_id"]), int(row["layer"]))
        full_key = key if cross_position else key + (str(row["position_type"]),)
        condition = str(row["patch_condition"])
        position_type = str(row["position_type"])
        if condition == treatment and (
            treatment_position_types is None or position_type in treatment_position_types
        ):
            treat[full_key] = float(value)
        if condition == control and (
            control_position_types is None or position_type in control_position_types
        ):
            ctrl[full_key] = float(value)
    deltas = [treat[k] - ctrl[k] for k in treat.keys() & ctrl.keys()]
    result = ap.bootstrap_ci(deltas, seed=ap.stable_int_seed(f"{treatment}:{control}:{score_key}"))
    result["stable_positive"] = bool(result.get("ci95_low") is not None and result["ci95_low"] > 0)
    return result
