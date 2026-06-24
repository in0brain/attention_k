"""Build ablated question records from ablation unit records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_ablated_question_record,
    validate_ablation_unit_record,
)


DEFAULT_ABLATION_TYPES = ["delete", "generalize"]
SUPPORTED_ABLATION_TYPES = set(DEFAULT_ABLATION_TYPES)
SUPPORTED_LANGUAGES = {"auto", "en", "zh"}

EN_GENERALIZATIONS = {
    "number": "some number",
    "entity": "some entity",
    "object": "some object",
    "operation": "some operation",
    "relation": "some relation",
    "comparison": "some comparison",
    "negation": "some negation condition",
    "condition": "some condition",
    "question_target": "some question target",
    "cyber_security_term": "some security issue",
}

ZH_GENERALIZATIONS = {
    "number": "某个数量",
    "entity": "某个实体",
    "object": "某个对象",
    "operation": "某个操作",
    "relation": "某种关系",
    "comparison": "某种比较",
    "negation": "某个否定条件",
    "condition": "某个条件",
    "question_target": "某个问题目标",
    "cyber_security_term": "某个安全问题",
}


def apply_ablation_to_unit(
    question: str,
    unit: dict,
    ablation_type: str,
    language: str = "auto",
) -> str:
    """Apply one ablation type to one ablation unit."""
    _validate_ablation_type(ablation_type)
    _validate_language(language)

    ablated_question = question
    resolved_language = _resolve_language(question, language)
    for span in sorted(unit["spans"], key=lambda item: item["start"], reverse=True):
        start = span["start"]
        end = span["end"]
        replacement = ""
        if ablation_type == "generalize":
            replacement = _generalization_for_span(span, resolved_language)
        ablated_question = ablated_question[:start] + replacement + ablated_question[end:]

    return _cleanup_question(ablated_question)


def build_ablated_question_records(
    ablation_unit_records: list[dict],
    ablation_types: list[str] | None = None,
    language: str = "auto",
) -> tuple[list[dict], dict]:
    """Build validated ablated question records and construction statistics."""
    selected_ablation_types = list(ablation_types or DEFAULT_ABLATION_TYPES)
    for ablation_type in selected_ablation_types:
        _validate_ablation_type(ablation_type)
    _validate_language(language)

    stats = _empty_stats(ablation_unit_records, selected_ablation_types, language)
    output_records: list[dict] = []

    for ablation_unit_record in ablation_unit_records:
        validate_ablation_unit_record(ablation_unit_record)
        question = ablation_unit_record["question"]
        for unit in ablation_unit_record["units"]:
            if _has_overlapping_spans(unit["spans"]):
                stats["num_skipped_overlap"] += len(selected_ablation_types)
                continue

            for ablation_type in selected_ablation_types:
                ablated_question = apply_ablation_to_unit(
                    question,
                    unit,
                    ablation_type,
                    language=language,
                )
                if not ablated_question.strip():
                    stats["num_skipped_empty"] += 1
                    continue
                if ablated_question == question:
                    stats["num_skipped_unchanged"] += 1
                    continue

                record = {
                    "ablation_id": (
                        f"{ablation_unit_record['id']}__{unit['unit_id']}__{ablation_type}"
                    ),
                    "id": ablation_unit_record["id"],
                    "unit_id": unit["unit_id"],
                    "unit_scope": unit["unit_scope"],
                    "group_type": unit["group_type"],
                    "span_ids": list(unit["span_ids"]),
                    "spans": [dict(span) for span in unit["spans"]],
                    "ablation_type": ablation_type,
                    "original_question": question,
                    "ablated_question": ablated_question,
                }
                validate_ablated_question_record(record)
                output_records.append(record)
                stats["num_output_ablations"] += 1
                stats["ablation_type_counts"][ablation_type] += 1
                stats["unit_scope_counts"][unit["unit_scope"]] += 1
                stats["group_type_counts"][unit["group_type"]] += 1

    stats["ablation_type_counts"] = dict(sorted(stats["ablation_type_counts"].items()))
    stats["unit_scope_counts"] = dict(sorted(stats["unit_scope_counts"].items()))
    stats["group_type_counts"] = dict(sorted(stats["group_type_counts"].items()))
    return output_records, stats


def build_ablated_question_file(
    input_path: str | Path,
    output_path: str | Path,
    ablation_types: list[str] | None = None,
    language: str = "auto",
) -> tuple[list[dict], dict]:
    """Read ablation units, build ablated questions, and write JSONL output."""
    ablation_unit_records = read_jsonl(input_path)
    records, stats = build_ablated_question_records(
        ablation_unit_records,
        ablation_types=ablation_types,
        language=language,
    )
    write_jsonl(records, output_path)
    return records, stats


def _empty_stats(
    ablation_unit_records: list[dict],
    ablation_types: list[str],
    language: str,
) -> dict:
    units = [unit for record in ablation_unit_records for unit in record.get("units", [])]
    return {
        "num_input_questions": len(ablation_unit_records),
        "num_input_units": len(units),
        "num_output_ablations": 0,
        "num_skipped_empty": 0,
        "num_skipped_unchanged": 0,
        "num_skipped_overlap": 0,
        "ablation_type_counts": Counter(),
        "unit_scope_counts": Counter(),
        "group_type_counts": Counter(),
        "language": language,
        "ablation_types": list(ablation_types),
    }


def _validate_ablation_type(ablation_type: str) -> None:
    if ablation_type not in SUPPORTED_ABLATION_TYPES:
        allowed = ", ".join(DEFAULT_ABLATION_TYPES)
        raise ValueError(f"Unsupported ablation_type {ablation_type!r}; allowed: {allowed}")


def _validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        allowed = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language {language!r}; allowed: {allowed}")


def _resolve_language(question: str, language: str) -> str:
    if language in {"en", "zh"}:
        return language
    return "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en"


def _generalization_for_span(span: dict, language: str) -> str:
    generalizations = ZH_GENERALIZATIONS if language == "zh" else EN_GENERALIZATIONS
    fallback = "某个信息" if language == "zh" else "some information"
    return generalizations.get(span["type"], fallback)


def _cleanup_question(question: str) -> str:
    question = re.sub(r"\s+", " ", question)
    question = re.sub(r"\s+([,.;:!?])", r"\1", question)
    question = re.sub(r"\s*([，。！？；：、])\s*", r"\1", question)
    return question.strip()


def _has_overlapping_spans(spans: list[dict]) -> bool:
    sorted_spans = sorted(spans, key=lambda span: (span["start"], span["end"]))
    previous_end = -1
    for span in sorted_spans:
        if span["start"] < previous_end:
            return True
        previous_end = max(previous_end, span["end"])
    return False
