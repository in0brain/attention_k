"""Pure option-letter proxy helpers for finite-label cyber MCQ tasks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

FORBIDDEN_INFERENCE_FEATURE_FIELDS = {
    "gold_label",
    "gold_label_id",
    "gold_label_text",
    "correct_option",
    "answer_key",
    "target_label",
}
#: 标准答案格式：`Answer: C` / `Final answer：C`。兼容英文与中文冒号、多余空格、
#: 大小写与 Markdown 包裹。**在原文上匹配**，不做文本规范化 —— 一旦规范化，
#: 回传的字符区间就会相对原文偏移。
_EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer\s*[:：]\s*(?:\*\*|__|\*)?\s*([A-Z])(?![A-Za-z0-9])", re.IGNORECASE)
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", re.IGNORECASE)

#: **明确的选择语句**：主语是 answer/option/choice，谓语是系动词，宾语是候选字母。
#: ⚠️ 字母这一组**必须大写**：前半段用 `(?i:...)` 让 answer/option 这些词不区分
#: 大小写，但字母本身区分 —— 否则英文冠词 "a" 会被当成选项 A。
#: "…(option B) is a major challenge…" 曾因此与真正的 `Answer: B` 判成冲突。
#: 选项标号在提示里本来就是大写，模型写 "answer is C" 而不是 "answer is c"。
_ANSWER_ASSERTION_RE = re.compile(
    r"(?i:\b(?:answer|option|choice)\b[^.\n]{0,40}?"
    r"\b(?:is|are|would\s+be|will\s+be|must\s+be|should\s+be)\b"
    r"\s*[:：]?\s*(?:\*\*|__|\*)?\s*(?:option\s+)?)([A-Z])(?![A-Za-z0-9])")
#: **第一人称的选择动作**："I would choose option B"。限定主语，
#: 因为无主语的 "choose B" 既可能是作答，也可能是指令或否定句的一部分。
#: 字母同样必须大写，理由见上。
_FIRST_PERSON_CHOICE_RE = re.compile(
    r"(?i:\b(?:i|we)\b\s*(?:'ll|'d|\s+will|\s+would|\s+am\s+going\s+to|\s+shall)?\s*"
    r"\b(?:choose|select|pick|go\s+with)\b\s*"
    r"(?:option\s+)?(?:\*\*|__|\*)?\s*)([A-Z])(?![A-Za-z0-9])")

#: 句级排除项。命中任一条 → 该句里的选择语句**不算**作答。
#: 逐条对应 2026-09-15 Codex 实测到的缺陷类别；**只看句子，不看匹配片段** ——
#: 上一版只查匹配片段，于是 "Do not choose A." 里的 "choose A" 逃掉了否定。
_SENTENCE_DISQUALIFIERS = (
    ("negation", re.compile(
        r"\b(?:not|never|neither|nor|avoid|excludes?|excluding|incorrect|wrong|"
        r"rule\s+out|ruled\s+out|rather\s+than|instead\s+of)\b|n['’]t",
        re.IGNORECASE)),
    ("conditional", re.compile(
        r"\b(?:if|unless|whether|suppose|supposing|assuming|hypothetically|"
        r"were\s+to|in\s+case|otherwise)\b", re.IGNORECASE)),
    ("instruction", re.compile(
        r"^\s*(?:do\b|don['’]t\b|please\b|you\s+(?:should|must|can|may)\b|"
        r"let['’]s\b)", re.IGNORECASE)),
    ("option_discussion", re.compile(
        r"\b(?:options?|choices?|answers?)\s+"
        r"(?:include|includes|included|are|were|list|listed|consist|contain|contains)\b",
        re.IGNORECASE)),
    ("quotation", re.compile(r"[“”‘’]|(?<!\w)\"")),
)

#: 裸字母回答两端允许剥掉的装饰（标点、括号、Markdown 强调、引号、空白）。
_BARE_ANSWER_TRIM = ".,;:：!?()[]{}<>*_-\"' \t\n"
#: 句子切分：句末标点或换行。保留偏移，**不改写原文**。
_SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")

_TOKEN_WHITESPACE_MARKERS = ("Ġ", "▁", "Ċ", "ĉ", "▏")
_SINGLE_CHAR_RUN_RE = re.compile(r"(.)\1{29,}", re.S)
_SUBSTRING_LOOP_RE = re.compile(r"(.{6,40}?)\1{4,}", re.S)
DEFAULT_CHAR_PER_TOKEN_RATIO = 3.0


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    # HuggingFace returns BatchEncoding (a UserDict, NOT a dict subclass), so an
    # isinstance(encoded, dict) check misses it and iterating yields key names.
    # Duck-type on mapping behaviour instead.
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _token_piece_content(piece: str | None) -> str:
    if piece is None:
        return ""
    normalized = str(piece)
    for marker in _TOKEN_WHITESPACE_MARKERS:
        normalized = normalized.replace(marker, "")
    return normalized.strip()


def _resolve_option_token_ids(tokenizer: Any, labels: list[str], *, text_form: str) -> dict[str, int]:
    if len(labels) != len(set(labels)):
        raise ValueError(f"option labels must be unique: {labels!r}")
    result: dict[str, int] = {}
    for raw_label in labels:
        label = str(raw_label).strip().upper()
        if len(label) != 1:
            raise ValueError(f"option label must be one character: {raw_label!r}")
        probe_text = f" {label}" if text_form == "space" else label
        token_ids = _tokenizer_ids(tokenizer, probe_text)
        valid_ids = [
            token_id
            for token_id in token_ids
            if _token_piece_content(tokenizer.convert_ids_to_tokens(token_id))
        ]
        if len(valid_ids) != 1:
            raise ValueError(
                f"option label {label!r} ({text_form} form) must map to one non-whitespace token; "
                f"token_ids={token_ids}, valid_token_ids={valid_ids}"
            )
        result[label] = valid_ids[0]
    collisions = {
        token_id: sorted(label for label, candidate in result.items() if candidate == token_id)
        for token_id in set(result.values())
        if sum(candidate == token_id for candidate in result.values()) > 1
    }
    if collisions:
        raise ValueError(f"option label token-id collision ({text_form} form): labels={result}, collisions={collisions}")
    return result


def option_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Resolve each label to one distinct non-whitespace tokenizer token.

    Uses the leading-space form (" A", " B", ...), matching the "Answer: X"
    convention where the letter follows a space. See bare_option_token_ids
    for completions that emit the letter with no preceding space (e.g. a
    chat-turn that answers with just the letter, no restated prefix).
    """
    return _resolve_option_token_ids(tokenizer, labels, text_form="space")


def bare_option_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Resolve each label to its token id with NO leading-space prefix.

    Some completion styles emit the option letter as the very first token of
    a turn (observed on the Sprint 4B-2 chat-template condition: the model
    treats "Answer: <letter>" in the prompt as a format instruction and
    answers with the bare letter directly, never restating "Answer: "), which
    tokenizes to a different id than the " <letter>" form. Same validation as
    option_token_ids, without the space prefix.
    """
    return _resolve_option_token_ids(tokenizer, labels, text_form="bare")


def _valid_label_set(valid_labels: list[str]) -> tuple[list[str], set[str]]:
    normalized = [str(label).strip().upper() for label in valid_labels]
    if len(normalized) < 2 or len(normalized) != len(set(normalized)):
        raise ValueError("valid_labels must contain at least two unique labels")
    if any(len(label) != 1 for label in normalized):
        raise ValueError("valid_labels must contain single-character labels")
    return normalized, set(normalized)


def _sentences(text: str) -> list[tuple[int, int]]:
    """把 text 切成句子的 (start, end) —— **只切不改**，偏移始终对着原文。"""
    spans, start = [], 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _sentence_of(spans: list[tuple[int, int]], index: int) -> tuple[int, int]:
    for start, end in spans:
        if start <= index < end:
            return start, end
    return spans[-1]


def disqualifying_context(sentence: str) -> str | None:
    """这句话里有没有让"选择语句"不算作答的东西。返回命中的类别名。"""
    for name, pattern in _SENTENCE_DISQUALIFIERS:
        if pattern.search(sentence):
            return name
    return None


def _answer_claims(text: str, valid: set[str]) -> tuple[list[dict], list[dict]]:
    """抽出全部**有效的答案声明**，以及被句级排除项挡掉的那些。

    三种声明形式统一处理，冲突在调用方一并检查 —— 上一版让 `Answer: X` 分支
    直接返回，于是 "Answer: A. Actually, the correct answer is C." 的冲突被掩盖。
    """
    spans = _sentences(text)
    kept: list[dict] = []
    rejected: list[dict] = []
    sources = (
        ("explicit_answer_marker", _EXPLICIT_ANSWER_RE, False),
        ("answer_statement", _ANSWER_ASSERTION_RE, True),
        ("answer_statement", _FIRST_PERSON_CHOICE_RE, True),
    )
    for kind, pattern, needs_context_check in sources:
        for match in pattern.finditer(text):
            label = match.group(1).upper()
            if label not in valid:
                continue
            s_start, s_end = _sentence_of(spans, match.start(1))
            claim = {"label": label, "kind": kind, "span": list(match.span(1)),
                     "sentence": text[s_start:s_end]}
            if needs_context_check:
                reason = disqualifying_context(claim["sentence"])
                if reason:
                    rejected.append({**claim, "rejected_because": reason})
                    continue
            kept.append(claim)
    kept.sort(key=lambda c: c["span"][0])
    return kept, rejected


def _result(label, method, span=None, reason=None, claims=None, rejected=None) -> dict:
    """标签与字符区间**出自同一次判定** —— 下游定位不得另算一遍。"""
    return {
        "parsed_label": label,
        "parse_method": method,
        "parse_failure": label is None,
        "answer_char_span": (None if span is None else [int(span[0]), int(span[1])]),
        "parse_failure_reason": reason,
        "answer_claims": [{k: c[k] for k in ("label", "kind", "span")}
                          for c in (claims or [])],
        "rejected_claims": [{k: c[k] for k in ("label", "kind", "span", "rejected_because")}
                            for c in (rejected or [])],
    }


def parse_option_answer(text: str, valid_labels: list[str]) -> dict:
    """把 completion 解析成一个候选字母，并给出该字母在**原文**里的字符区间。

    三条可靠规则，别的一律失败（2026-09-15 第二次修订）：

        1. 标准答案格式   `Answer: C` / `Final answer：C`
                          兼容中英文冒号、空格、大小写、Markdown 包裹
        2. 明确的选择语句 "the correct answer is C" / "I would choose option B"
                          **必须通过句级检查**：否定、条件假设、引用、指令、
                          选项讨论 —— 命中任一条即不算作答
        3. 裸字母回答     整条回答只有一个候选字母及装饰（`C` / `[C]` / `**C**`）

    **取消**了"全文末尾孤立字母可作为答案"这条无依据的回退：提到一个选项不等于
    选择它（"The options include A." 曾因此被判成 A）。裸字母回答由规则 3 覆盖。

    冲突**统一检查**：把标准标记与选择语句里的有效声明放在一起看，出现不同标签
    即判失败并记录原因，**不**凭分支优先级或"最后一次出现"推断改口。
    上一版让 `Answer:` 分支直接返回，于是
    "Answer: A. Actually, the correct answer is C." 被判成 A。

    只给选项内容、不给字母的回答仍然失败 —— 不用字符串映射倒推字母，
    那会制造一个并不存在的 hidden 读取位置。
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    _, valid = _valid_label_set(valid_labels)

    claims, rejected = _answer_claims(text, valid)
    if claims:
        labels = {c["label"] for c in claims}
        if len(labels) > 1:
            return _result(None, "parse_failure",
                           reason="contradictory_answer_claims",
                           claims=claims, rejected=rejected)
        # 标签一致时，用标准标记的区间（那是答案字母本身）；没有标记就用最后一条声明。
        explicit = [c for c in claims if c["kind"] == "explicit_answer_marker"]
        chosen = explicit[-1] if explicit else claims[-1]
        return _result(chosen["label"], chosen["kind"], chosen["span"],
                       claims=claims, rejected=rejected)

    stripped = text.strip()
    core = stripped.strip(_BARE_ANSWER_TRIM)
    if len(core) == 1 and core.upper() in valid:
        start = text.index(core, text.index(stripped))
        return _result(core.upper(), "bare_letter", (start, start + 1),
                       rejected=rejected)

    reason = ("rejected_by_sentence_context" if rejected else "no_answer_claim")
    return _result(None, "parse_failure", reason=reason, rejected=rejected)


def detect_degeneration(
    completion: str,
    *,
    valid_labels: list[str],
    max_new_tokens: int,
    num_generated_tokens: int | None = None,
    char_per_token_ratio: float = DEFAULT_CHAR_PER_TOKEN_RATIO,
) -> dict:
    """Flag a completion as degenerate generation (loop/repeat/gibberish runaway).

    Three independent rules, any one match sets ``degenerate=True``:

    1. single_char_run: one character repeated >= 30 times consecutively
       (e.g. "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!").
    2. substring_loop: a substring of length 6-40 repeated >= 5 times
       consecutively (e.g. ".4922.4922.4922.4922.4922").
    3. truncated_failure: no option letter could be parsed AND the completion
       ran to (or near) the generation budget. When ``num_generated_tokens``
       is supplied (the caller controls the generation loop and knows the
       exact count) that is used directly -- this is the precise signal and
       is preferred. Without it, a character-length heuristic is used instead
       (``len(completion) >= 0.9 * max_new_tokens * char_per_token_ratio``);
       this is approximate because the chars-per-token ratio for gibberish
       completions varies widely (~2.3-6.1 observed on real Sprint 4B smoke
       failures) so the fallback can miss short completions that hit EOS
       mid-gibberish without exhausting the budget. Such cases still count
       toward parse_failure_rate even when not flagged degenerate here.
    """

    single_char_match = _SINGLE_CHAR_RUN_RE.search(completion)
    substring_match = _SUBSTRING_LOOP_RE.search(completion)
    parsed_failed = parse_option_answer(completion, valid_labels)["parse_failure"]
    if num_generated_tokens is not None:
        truncated = parsed_failed and num_generated_tokens >= 0.9 * max_new_tokens
    else:
        truncated = parsed_failed and len(completion) >= 0.9 * max_new_tokens * char_per_token_ratio
    matched_rules = []
    if single_char_match:
        matched_rules.append("single_char_run")
    if substring_match:
        matched_rules.append("substring_loop")
    if truncated:
        matched_rules.append("truncated_failure")
    return {
        "degenerate": bool(matched_rules),
        "matched_rules": matched_rules,
        "single_char_run_match": single_char_match.group(0)[:12] if single_char_match else None,
        "substring_loop_match": substring_match.group(1) if substring_match else None,
        "truncated_failure_basis": (
            "token_count" if num_generated_tokens is not None else "char_length_heuristic"
        ),
    }


def locate_label_readout_position(
    tokenizer: Any,
    prompt: str,
    completion: str,
    parsed_label: str,
) -> int:
    """Return the token position immediately before the parsed answer label."""
    label = str(parsed_label).strip().upper()
    if not label:
        raise ValueError("parsed_label must be non-empty")
    parsed = parse_option_answer(completion, [label, "_"])
    if parsed["parsed_label"] != label:
        raise ValueError(f"parsed label {label!r} cannot be located in completion")
    explicit = [
        match for match in _EXPLICIT_ANSWER_RE.finditer(completion)
        if match.group(1).upper() == label
    ]
    if explicit:
        candidates = [match.start(1) for match in explicit]
    else:
        candidates = [match.start(1) for match in _trailing_isolated_matches(completion, {label})]
    if len(candidates) != 1:
        raise ValueError(
            f"parsed label {label!r} must have exactly one answer occurrence; "
            f"found {len(candidates)}"
        )
    combined = prompt + completion
    encoded = tokenizer(
        combined,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # HuggingFace BatchEncoding is a UserDict, not a dict subclass; the same
    # duck-typing fix as _tokenizer_ids applies here (see its docstring).
    offsets = (
        encoded.get("offset_mapping")
        if isinstance(encoded, dict) or hasattr(encoded, "get")
        else None
    )
    if offsets is None:
        raise ValueError("tokenizer must provide offset_mapping")
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    label_start = len(prompt) + candidates[0]
    token_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(start) and int(start) <= label_start < int(end)
    ]
    if len(token_positions) != 1:
        raise ValueError(
            f"answer label must map to exactly one token position; found {token_positions}"
        )
    label_position = token_positions[0]
    if label_position == 0:
        raise ValueError("answer label has no preceding readout position")
    return label_position - 1


def _finite_values(values: Iterable[Any], *, name: str, minimum: int) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) < minimum:
        raise ValueError(f"{name} requires at least {minimum} values")
    if any(not math.isfinite(value) for value in converted):
        raise ValueError(f"{name} values must be finite real numbers")
    return converted


def label_margin(option_logits: dict[str, float]) -> float:
    """Return top-1 minus top-2 over candidate option logits."""
    if not isinstance(option_logits, dict):
        raise ValueError("option_logits must be a dict")
    ordered = sorted(
        _finite_values(option_logits.values(), name="label_margin", minimum=2),
        reverse=True,
    )
    return ordered[0] - ordered[1]


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def label_entropy(option_logits: dict[str, float]) -> float:
    """Compute entropy after normalizing only candidate option logits."""
    if not isinstance(option_logits, dict):
        raise ValueError("option_logits must be a dict")
    values = _finite_values(option_logits.values(), name="label_entropy", minimum=2)
    probabilities = _softmax(values)
    return -sum(probability * math.log(probability) for probability in probabilities)


def _flatten_logits(logits: Any) -> list[float]:
    if hasattr(logits, "detach") and hasattr(logits, "reshape"):
        logits = logits.detach().float().cpu().reshape(-1).tolist()
    elif hasattr(logits, "reshape") and hasattr(logits, "tolist"):
        logits = logits.reshape(-1).tolist()
    elif hasattr(logits, "tolist"):
        logits = logits.tolist()

    def walk(value: Any) -> Iterable[Any]:
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)
        else:
            yield value

    return _finite_values(walk(logits), name="full_entropy", minimum=2)


def full_entropy(logits: Any) -> float:
    """Compute numerically stable full-vocabulary softmax entropy."""
    probabilities = _softmax(_flatten_logits(logits))
    return -sum(probability * math.log(probability) for probability in probabilities)


def self_consistency_features(
    greedy_label: str | None,
    sampled_labels: list[str | None],
    valid_labels: list[str],
) -> dict:
    """Summarize parsed samples without accepting any evaluation label."""
    ordered_labels, valid = _valid_label_set(valid_labels)
    normalized_greedy = None if greedy_label is None else str(greedy_label).upper()
    if normalized_greedy is not None and normalized_greedy not in valid:
        raise ValueError(f"greedy_label {greedy_label!r} is outside valid_labels")
    normalized_samples = [
        None if label is None else str(label).upper() for label in sampled_labels
    ]
    invalid = [label for label in normalized_samples if label is not None and label not in valid]
    if invalid:
        raise ValueError(f"sampled label(s) outside valid_labels: {invalid}")
    counts = Counter(label for label in normalized_samples if label is not None)
    majority_label = None
    if counts:
        majority_label = max(
            ordered_labels,
            key=lambda label: (counts.get(label, 0), -ordered_labels.index(label)),
        )
    parsed_count = sum(counts.values())
    sample_count = len(normalized_samples)
    agreement = (
        counts.get(normalized_greedy, 0) / parsed_count
        if normalized_greedy is not None and parsed_count
        else None
    )
    return {
        "num_samples": sample_count,
        "num_parsed_samples": parsed_count,
        "parse_failure_count": sample_count - parsed_count,
        "parse_failure_rate": (
            (sample_count - parsed_count) / sample_count if sample_count else 0.0
        ),
        "greedy_label": normalized_greedy,
        "sample_vote_counts": {
            label: counts.get(label, 0) for label in ordered_labels
        },
        "sample_majority_label": majority_label,
        "self_consistency_with_greedy": agreement,
        "majority_agrees_with_greedy": (
            majority_label == normalized_greedy
            if majority_label is not None and normalized_greedy is not None
            else None
        ),
    }



def fixed_f5_risk_score(features: dict[str, Any]) -> float | None:
    """Compatibility fixed combination for the superseded Sprint 4B smoke."""
    raw_values = [
        -float(features["f5_label_margin"])
        if features.get("f5_label_margin") is not None
        else None,
        float(features["f5_label_entropy"])
        if features.get("f5_label_entropy") is not None
        else None,
        1.0 - float(features["f5_self_consistency"])
        if features.get("f5_self_consistency") is not None
        else None,
        1.0 - float(features["f5_sc_majority_agree"])
        if features.get("f5_sc_majority_agree") is not None
        else None,
    ]
    values = [value for value in raw_values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def assert_no_gold_label_leakage(feature_record: dict) -> None:
    """Reject evaluation-label keys anywhere inside a feature record."""
    if not isinstance(feature_record, dict):
        raise ValueError("feature_record must be a dict")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).casefold()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized_key in FORBIDDEN_INFERENCE_FEATURE_FIELDS:
                    raise ValueError(f"forbidden inference feature field at {child_path}")
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(feature_record, "")


def classify_trace_by_option(parsed_label: str | None, gold_label: str) -> str:
    """Classify a parsed trace for evaluation only."""
    if parsed_label is None:
        return "parse_failure"
    return (
        "correct"
        if str(parsed_label).strip().upper() == str(gold_label).strip().upper()
        else "wrong"
    )


def assert_no_gold_feature_leakage(feature_names: list[str]) -> None:
    """Compatibility wrapper for the superseded Sprint 4B smoke script."""
    assert_no_gold_label_leakage({name: None for name in feature_names})
