"""Rule-based candidate span extraction for early v0 experiments."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import validate_candidate_span_record


SUPPORTED_LANGUAGES = {"auto", "en", "zh"}
SUPPORTED_BACKENDS = {"rule_based"}
DEFAULT_MAX_CANDIDATES = 20

SPAN_PRIORITY = {
    "number": 0,
    "question_target": 1,
    "condition": 2,
    "negation": 3,
    "comparison": 4,
    "operation": 5,
    "cyber_security_term": 6,
    "object": 7,
    "entity": 8,
    "relation": 9,
}

EN_QUESTION_TARGETS = [
    "How many",
    "How much",
    "What",
    "Which",
    "Who",
    "When",
    "Where",
    "Why",
]
ZH_QUESTION_TARGETS = ["哪一个", "什么时候", "多少", "几个", "什么", "哪种", "谁", "何时", "哪里", "为什么", "几"]

EN_COMPARISONS = [
    "no more than",
    "no less than",
    "more than",
    "less than",
    "at least",
    "at most",
    "greater",
    "smaller",
    "higher",
    "lower",
    "cheaper",
    "fewer",
    "more",
    "less",
]
ZH_COMPARISONS = ["不超过", "不少于", "更多", "更少", "大于", "小于", "高于", "低于", "至少", "至多", "超过", "少于"]

EN_NEGATIONS = ["cannot", "without", "never", "can't", "not", "no"]
ZH_NEGATIONS = ["没有", "从不", "不能", "无法", "不", "未", "无"]

EN_CONDITIONS = ["assuming", "provided", "unless", "given", "when", "if"]
ZH_CONDITIONS = ["如果", "假设", "给定", "除非", "当", "若"]

EN_OPERATIONS = [
    "deserializes",
    "deserialize",
    "executes",
    "execute",
    "encrypt",
    "decrypt",
    "upload",
    "download",
    "bypass",
    "exploit",
    "removes",
    "removed",
    "remove",
    "bought",
    "sells",
    "sold",
    "sell",
    "added",
    "adds",
    "add",
    "gives",
    "gave",
    "give",
    "takes",
    "took",
    "take",
    "remain",
    "total",
    "difference",
    "buys",
    "buy",
    "left",
]
ZH_OPERATIONS = [
    "反序列化",
    "购买",
    "增加",
    "减少",
    "移除",
    "删除",
    "拿走",
    "剩下",
    "总共",
    "一共",
    "执行",
    "加密",
    "解密",
    "上传",
    "下载",
    "绕过",
    "利用",
    "买",
    "卖",
    "给",
]

EN_OBJECTS = [
    "notebooks",
    "requests",
    "records",
    "packets",
    "servers",
    "apples",
    "pencils",
    "muffins",
    "marbles",
    "pages",
    "books",
    "files",
    "users",
    "items",
]
ZH_OBJECTS = ["笔记本", "数据包", "服务器", "苹果", "文件", "用户", "请求", "记录", "物品", "书"]

EN_CYBER_SECURITY_TERMS = [
    "remote code execution",
    "unsafe deserialization",
    "cross-site scripting",
    "privilege escalation",
    "buffer overflow",
    "command injection",
    "path traversal",
    "SQL injection",
]
ZH_CYBER_SECURITY_TERMS = [
    "不安全反序列化",
    "远程代码执行",
    "SQL 注入",
    "跨站脚本",
    "权限提升",
    "缓冲区溢出",
    "命令注入",
    "路径遍历",
]

OBJECT_STOPWORDS = {
    phrase.lower()
    for phrase in (
        EN_QUESTION_TARGETS
        + EN_COMPARISONS
        + EN_NEGATIONS
        + EN_CONDITIONS
        + EN_OPERATIONS
    )
}

ARABIC_NUMBER_PATTERN = re.compile(
    r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?"
)
CHINESE_NUMBER_PATTERN = re.compile(r"[零一二三四五六七八九十百千万亿两]+")


def extract_candidate_spans(
    question: str,
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict]:
    """Extract candidate spans from a question with a small rule-based backend."""
    if not isinstance(question, str):
        raise ValueError("question must be a str")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"language must be one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 0:
        raise ValueError("max_candidates must be an int >= 0")

    candidates: list[dict] = []
    languages = _languages_to_apply(question, language)

    _add_number_candidates(question, candidates)

    if "en" in languages:
        _add_phrase_candidates(question, candidates, EN_QUESTION_TARGETS, "question_target", english=True)
        _add_phrase_candidates(question, candidates, EN_COMPARISONS, "comparison", english=True)
        _add_phrase_candidates(question, candidates, EN_NEGATIONS, "negation", english=True)
        _add_phrase_candidates(question, candidates, EN_CONDITIONS, "condition", english=True)
        _add_phrase_candidates(question, candidates, EN_OPERATIONS, "operation", english=True)
        _add_phrase_candidates(question, candidates, EN_CYBER_SECURITY_TERMS, "cyber_security_term", english=True)
        _add_phrase_candidates(question, candidates, EN_OBJECTS, "object", english=True)
        _add_english_number_neighbor_objects(question, candidates)

    if "zh" in languages:
        _add_phrase_candidates(question, candidates, ZH_QUESTION_TARGETS, "question_target", english=False)
        _add_phrase_candidates(question, candidates, ZH_COMPARISONS, "comparison", english=False)
        _add_phrase_candidates(question, candidates, ZH_NEGATIONS, "negation", english=False)
        _add_phrase_candidates(question, candidates, ZH_CONDITIONS, "condition", english=False)
        _add_phrase_candidates(question, candidates, ZH_OPERATIONS, "operation", english=False)
        _add_phrase_candidates(question, candidates, ZH_CYBER_SECURITY_TERMS, "cyber_security_term", english=False)
        _add_phrase_candidates(question, candidates, ZH_OBJECTS, "object", english=False)

    final_candidates = _dedupe_sort_truncate(candidates, max_candidates)
    return _assign_span_ids(final_candidates)


def build_candidate_span_records(
    question_records: list[dict],
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict]:
    """Build validated candidate span records from normalized question records."""
    records: list[dict] = []

    for question_record in question_records:
        record = {
            "id": question_record["id"],
            "question": question_record["question"],
            "candidates": extract_candidate_spans(
                question_record["question"],
                language=language,
                backend=backend,
                max_candidates=max_candidates,
            ),
        }
        validate_candidate_span_record(record)
        _validate_offsets(record)
        records.append(record)

    return records


def extract_candidate_span_file(
    input_path: str | Path,
    output_path: str | Path,
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict]:
    """Read question records, extract candidate spans, and write JSONL output."""
    question_records = read_jsonl(input_path)
    candidate_records = build_candidate_span_records(
        question_records,
        language=language,
        backend=backend,
        max_candidates=max_candidates,
    )
    write_jsonl(candidate_records, output_path)
    return candidate_records


def summarize_candidate_span_records(
    records: list[dict],
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict:
    """Return candidate count statistics for CLI output."""
    counts = [len(record["candidates"]) for record in records]
    total_candidates = sum(counts)
    span_type_counts: Counter[str] = Counter(
        candidate["type"] for record in records for candidate in record["candidates"]
    )

    return {
        "num_questions": len(records),
        "total_candidates": total_candidates,
        "avg_candidates_per_question": total_candidates / len(records) if records else 0.0,
        "min_candidates_per_question": min(counts) if counts else 0,
        "max_candidates_per_question": max(counts) if counts else 0,
        "questions_with_zero_candidates": sum(1 for count in counts if count == 0),
        "span_type_counts": dict(sorted(span_type_counts.items())),
        "max_candidates_setting": max_candidates,
        "language": language,
        "backend": backend,
    }


def _languages_to_apply(question: str, language: str) -> set[str]:
    if language != "auto":
        return {language}
    if re.search(r"[\u4e00-\u9fff]", question):
        return {"en", "zh"}
    return {"en"}


def _add_number_candidates(question: str, candidates: list[dict]) -> None:
    for match in ARABIC_NUMBER_PATTERN.finditer(question):
        _append_candidate(candidates, question, "number", match.start(), match.end())
    for match in CHINESE_NUMBER_PATTERN.finditer(question):
        _append_candidate(candidates, question, "number", match.start(), match.end())


def _add_phrase_candidates(
    question: str,
    candidates: list[dict],
    phrases: list[str],
    span_type: str,
    english: bool,
) -> None:
    for phrase in sorted(phrases, key=len, reverse=True):
        if english:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(phrase)}(?![A-Za-z0-9_])", re.IGNORECASE)
            for match in pattern.finditer(question):
                _append_candidate(candidates, question, span_type, match.start(), match.end())
        else:
            start = 0
            while True:
                index = question.find(phrase, start)
                if index == -1:
                    break
                _append_candidate(candidates, question, span_type, index, index + len(phrase))
                start = index + 1


def _add_english_number_neighbor_objects(question: str, candidates: list[dict]) -> None:
    pattern = re.compile(r"\b\d+(?:\.\d+)?%?\s+([A-Za-z][A-Za-z_-]*)")
    for match in pattern.finditer(question):
        start, end = match.span(1)
        if question[start:end].lower() in OBJECT_STOPWORDS:
            continue
        _append_candidate(candidates, question, "object", start, end)


def _append_candidate(
    candidates: list[dict],
    question: str,
    span_type: str,
    start: int,
    end: int,
) -> None:
    text = question[start:end]
    if not text:
        return
    candidates.append(
        {
            "span_id": "",
            "text": text,
            "type": span_type,
            "start": start,
            "end": end,
        }
    )


def _dedupe_sort_truncate(candidates: list[dict], max_candidates: int) -> list[dict]:
    deduped_by_key: dict[tuple[int, int, str], dict] = {}
    for candidate in candidates:
        key = (candidate["start"], candidate["end"], candidate["type"])
        deduped_by_key.setdefault(key, candidate)

    priority_sorted = sorted(
        deduped_by_key.values(),
        key=lambda candidate: (
            SPAN_PRIORITY.get(candidate["type"], 100),
            candidate["start"],
            candidate["end"],
            candidate["text"].lower(),
        ),
    )
    truncated = priority_sorted[:max_candidates]
    return sorted(truncated, key=lambda candidate: (candidate["start"], candidate["end"], candidate["type"]))


def _assign_span_ids(candidates: list[dict]) -> list[dict]:
    numbered: list[dict] = []
    for index, candidate in enumerate(candidates, start=1):
        numbered_candidate = dict(candidate)
        numbered_candidate["span_id"] = f"span_{index:03d}"
        numbered.append(numbered_candidate)
    return numbered


def _validate_offsets(record: dict) -> None:
    question = record["question"]
    for candidate in record["candidates"]:
        if question[candidate["start"] : candidate["end"]] != candidate["text"]:
            raise ValueError(
                "candidate offset mismatch for "
                f"{record['id']} {candidate['span_id']}: {candidate['text']!r}"
            )
