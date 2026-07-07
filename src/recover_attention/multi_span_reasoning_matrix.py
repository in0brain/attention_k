"""Sprint 2J-A: multi-span-per-question reasoning matrix construction.

This module builds the evaluation substrate that 2I-R found missing: multiple
candidate spans per question. It is text-only for 2J-A. Gold solution paths and
existing fragility targets are kept as diagnostic/evaluation-only fields and are
never used as formula input features.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from recover_attention import solution_path_numbers as spn
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl

BACKEND = "multi_span_reasoning_matrix_v0"
MASK_TOKEN = "[MASK]"

FORBIDDEN_FORMULA_INPUT_SUBSTRINGS = (
    "recovered",
    "solution_path",
    "drift",
    "bucket",
    "risk_strength",
    "gold",
    "answer",
    "label",
    "target",
    "trajectory",
    "cot",
    "nla",
    "oracle",
)

WORD_NUMBER_PATTERN = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"dozen|half|twice|double|triple|quadruple|quarter)\b",
    re.IGNORECASE,
)

ARABIC_NUMBER = re.compile(
    r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?"
)

RATE_PHRASE = re.compile(
    r"\b\d+(?:\.\d+)?%?\s+[A-Za-z]+(?:\s+[A-Za-z]+)?\s+(?:each|per|apiece)\b",
    re.IGNORECASE,
)

NUMBER_UNIT_PHRASE = re.compile(
    r"\b\d+(?:\.\d+)?%?(?:\s+[A-Za-z][A-Za-z'-]*){1,2}\b",
    re.IGNORECASE,
)

PHRASE_PATTERNS: list[tuple[str, str, int]] = [
    ("question_target", r"\bHow many are left\b|\bHow much did [A-Za-z]+ spend\b|\bWhat is the total\b|\bHow many\b|\bHow much\b|\bHow far\b|\bWhat\b", 20),
    ("comparison", r"\bmore than\b|\bless than\b|\bfewer than\b|\bolder than\b|\byounger than\b|\btwice as many\b|\bhalf as many\b|\bat least\b|\bat most\b|\bdifference between\b|\bgreater than\b|\bsmaller than\b", 30),
    ("negation", r"\bnot\b|\bwithout\b|\bexcept\b|\bnever\b|\bno\b", 40),
    ("condition", r"\bunless\b|\bif\b|\bafter\b|\bbefore\b|\bonly\b|\binstead\b|\bwhen\b|\bgiven\b", 45),
    ("operation", r"\bbought\b|\bbuy\b|\bsold\b|\bsell\b|\bgave\b|\bgive\b|\bleft\b|\bremaining\b|\btotal\b|\beach\b|\bper\b|\bspent\b|\bcost\b|\bearned\b|\blost\b|\bshared\b|\bdivided\b|\bincreased\b|\bdecreased\b|\baltogether\b|\bcombined\b|\bplus\b|\bminus\b|\btimes\b", 50),
]

OBJECT_STOPWORDS = {
    "how",
    "many",
    "much",
    "what",
    "which",
    "when",
    "where",
    "why",
    "the",
    "and",
    "then",
    "than",
    "with",
    "from",
    "that",
    "this",
    "into",
    "each",
    "per",
    "after",
    "before",
    "only",
    "total",
    "left",
    "more",
    "less",
    "many",
    "much",
    "does",
    "did",
    "will",
    "would",
    "there",
    "were",
    "have",
    "has",
    "had",
    "if",
}

SPAN_PRIORITY = {
    "rate": 5,
    "number_unit": 8,
    "number": 10,
    "question_target": 20,
    "comparison": 30,
    "negation": 40,
    "condition": 45,
    "operation": 50,
    "object": 80,
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _add_candidate(
    candidates: list[dict[str, Any]],
    question: str,
    start: int,
    end: int,
    span_type: str,
    source_rule: str,
) -> None:
    if start < 0 or end <= start or end > len(question):
        return
    text = question[start:end]
    stripped = text.strip(" ,.;:!?")
    if not stripped:
        return
    leading_trim = len(text) - len(text.lstrip(" ,.;:!?"))
    trailing_trim = len(text.rstrip(" ,.;:!?"))
    start = start + leading_trim
    end = start + len(text[leading_trim:trailing_trim].strip())
    text = question[start:end]
    if not text.strip():
        return
    candidates.append(
        {
            "span_text": text,
            "span_type": span_type,
            "start_char": start,
            "end_char": end,
            "source_rule": source_rule,
            "priority": SPAN_PRIORITY.get(span_type, 99),
        }
    )


def extract_multi_span_candidates(
    question: str,
    *,
    max_spans_per_question: int = 12,
) -> list[dict[str, Any]]:
    """Extract a bounded multi-span candidate set for one GSM8K question."""
    candidates: list[dict[str, Any]] = []
    text = question or ""

    for match in RATE_PHRASE.finditer(text):
        _add_candidate(candidates, text, match.start(), match.end(), "rate", "rate_phrase")

    for match in NUMBER_UNIT_PHRASE.finditer(text):
        phrase = match.group(0)
        words = phrase.split()
        if len(words) >= 2 and not words[1].lower() in OBJECT_STOPWORDS:
            span_type = "rate" if any(w.lower() in {"each", "per", "apiece"} for w in words) else "number_unit"
            _add_candidate(candidates, text, match.start(), match.end(), span_type, "number_unit_phrase")

    for match in ARABIC_NUMBER.finditer(text):
        _add_candidate(candidates, text, match.start(), match.end(), "number", "arabic_number")

    for match in WORD_NUMBER_PATTERN.finditer(text):
        _add_candidate(candidates, text, match.start(), match.end(), "number", "word_number")

    for span_type, pattern, _priority in PHRASE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            _add_candidate(candidates, text, match.start(), match.end(), span_type, f"{span_type}_phrase")

    _add_object_candidates(text, candidates)

    deduped = _dedupe_and_filter_overlaps(candidates)
    limited = _limit_candidate_mix(deduped, max_spans_per_question)
    _assign_span_ids_and_overlap(limited)
    return limited


def _add_object_candidates(question: str, candidates: list[dict[str, Any]]) -> None:
    # Unit-like nouns near numbers.
    for match in re.finditer(r"\b\d+(?:\.\d+)?%?\s+([A-Za-z][A-Za-z'-]{2,})\b", question):
        word = match.group(1)
        if word.lower() not in OBJECT_STOPWORDS:
            _add_candidate(candidates, question, match.start(1), match.end(1), "object", "number_neighbor_object")

    # Capitalized names/entities.
    for match in re.finditer(r"\b[A-Z][a-zA-Z'-]{2,}\b", question):
        word = match.group(0)
        if word.lower() not in OBJECT_STOPWORDS and word not in {"How", "What"}:
            _add_candidate(candidates, question, match.start(), match.end(), "object", "capitalized_entity")

    # A small fallback for repeated/common nouns if the question is sparse.
    for match in re.finditer(r"\b[a-z][a-zA-Z'-]{3,}s\b", question):
        word = match.group(0)
        if word.lower() not in OBJECT_STOPWORDS:
            _add_candidate(candidates, question, match.start(), match.end(), "object", "plural_object")


def _dedupe_and_filter_overlaps(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["start_char"], candidate["end_char"], candidate["span_type"])
        prev = by_key.get(key)
        if prev is None or candidate["priority"] < prev["priority"]:
            by_key[key] = candidate
    items = sorted(by_key.values(), key=lambda c: (c["start_char"], c["end_char"], c["priority"]))

    kept: list[dict[str, Any]] = []
    for cand in items:
        drop = False
        for other in kept:
            if not _overlaps(cand, other):
                continue
            cand_is_number = cand["span_type"] == "number"
            other_is_number = other["span_type"] == "number"
            cand_contains_other = cand["start_char"] <= other["start_char"] and cand["end_char"] >= other["end_char"]
            other_contains_cand = other["start_char"] <= cand["start_char"] and other["end_char"] >= cand["end_char"]
            if cand_is_number != other_is_number and (cand_contains_other or other_contains_cand):
                # Keep phrase+atomic-number pairs for later overlap-aware evaluation.
                continue
            if cand["priority"] > other["priority"] or (
                cand["priority"] == other["priority"]
                and (cand["end_char"] - cand["start_char"]) <= (other["end_char"] - other["start_char"])
            ):
                drop = True
                break
        if not drop:
            kept.append(cand)
    return sorted(kept, key=lambda c: (c["priority"], c["start_char"], c["end_char"]))


def _limit_candidate_mix(candidates: list[dict[str, Any]], max_spans: int) -> list[dict[str, Any]]:
    if len(candidates) <= max_spans:
        return sorted(candidates, key=lambda c: (c["start_char"], c["end_char"], c["priority"]))

    selected: list[dict[str, Any]] = []
    # Preserve all numeric candidates first, because keyness ranking depends on them.
    for candidate in candidates:
        if candidate["span_type"] in {"number", "number_unit", "rate"}:
            selected.append(candidate)
    # Ensure each important non-number family has a chance before object fillers.
    for family in ["comparison", "negation", "condition", "operation", "question_target", "object"]:
        for candidate in candidates:
            if len(selected) >= max_spans:
                break
            if candidate["span_type"] == family and candidate not in selected:
                selected.append(candidate)
                if family == "object":
                    break
        if len(selected) >= max_spans:
            break
    for candidate in candidates:
        if len(selected) >= max_spans:
            break
        if candidate not in selected:
            selected.append(candidate)
    return sorted(selected[:max_spans], key=lambda c: (c["start_char"], c["end_char"], c["priority"]))


def _assign_span_ids_and_overlap(candidates: list[dict[str, Any]]) -> None:
    for idx, candidate in enumerate(candidates):
        candidate["span_id"] = f"span_{idx:03d}"
        candidate["overlap_group_id"] = None
        candidate["parent_span_id"] = None
        candidate["child_span_ids"] = []

    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        placed = False
        for group in groups:
            if any(_overlaps(candidate, other) for other in group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    gid = 0
    for group in groups:
        if len(group) < 2:
            continue
        group_id = f"overlap_{gid:03d}"
        gid += 1
        for candidate in group:
            candidate["overlap_group_id"] = group_id
        parents = sorted(
            group,
            key=lambda c: (-(c["end_char"] - c["start_char"]), c["priority"]),
        )
        parent = parents[0]
        for child in group:
            if child is parent:
                continue
            if parent["start_char"] <= child["start_char"] and parent["end_char"] >= child["end_char"]:
                child["parent_span_id"] = parent["span_id"]
                parent["child_span_ids"].append(child["span_id"])


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a["start_char"]) < int(b["end_char"]) and int(b["start_char"]) < int(a["end_char"])


def build_surface_features(candidate: dict[str, Any], question: str) -> dict[str, float]:
    span_type = candidate["span_type"]
    span_text = candidate["span_text"]
    q_len = max(1, len(question))
    type_map = {
        "number": "surf_type_number",
        "number_unit": "surf_type_number_phrase",
        "rate": "surf_type_rate",
        "operation": "surf_type_operation",
        "comparison": "surf_type_comparison",
        "negation": "surf_type_negation",
        "condition": "surf_type_condition",
        "question_target": "surf_type_qfocus",
        "object": "surf_type_object",
    }
    features = {
        "surf_is_number_like": 1.0 if span_type in {"number", "number_unit", "rate"} else 0.0,
        "surf_contains_digit": 1.0 if re.search(r"\d", span_text) else 0.0,
        "surf_is_phrase": 1.0 if len(span_text.split()) > 1 else 0.0,
        "surf_span_char_len": float(len(span_text)),
        "surf_span_word_len": float(len(span_text.split())),
        "surf_question_rel_start": round(float(candidate["start_char"]) / q_len, 6),
        "surf_position_early": 1.0 if candidate["start_char"] / q_len < 0.33 else 0.0,
        "surf_position_mid": 1.0 if 0.33 <= candidate["start_char"] / q_len < 0.66 else 0.0,
        "surf_position_late": 1.0 if candidate["start_char"] / q_len >= 0.66 else 0.0,
        "surf_has_unit_hint": 1.0 if span_type in {"number_unit", "rate"} else 0.0,
    }
    for name in sorted(set(type_map.values())):
        features[name] = 0.0
    features[type_map.get(span_type, "surf_type_object")] = 1.0
    features["surface_keyness_proxy"] = surface_keyness_proxy(candidate)
    return features


def surface_keyness_proxy(candidate: dict[str, Any]) -> float:
    span_type = candidate["span_type"]
    base = {
        "number": 0.86,
        "number_unit": 0.88,
        "rate": 0.90,
        "comparison": 0.72,
        "negation": 0.70,
        "condition": 0.66,
        "operation": 0.64,
        "question_target": 0.44,
        "object": 0.24,
    }.get(span_type, 0.30)
    if candidate.get("parent_span_id"):
        base -= 0.04
    return round(max(0.0, min(1.0, base)), 6)


def weak_semantic_keyness(candidate: dict[str, Any]) -> str:
    return {
        "comparison": "comparison_key_candidate",
        "negation": "negation_key_candidate",
        "condition": "condition_key_candidate",
        "operation": "operation_key_candidate",
        "question_target": "question_target_candidate",
        "object": "object_or_context_candidate",
        "number": "unknown",
        "number_unit": "unknown",
        "rate": "unknown",
    }.get(candidate["span_type"], "unknown")


def _number_status_for_candidate(
    candidate: dict[str, Any],
    question_numbers: list[dict[str, Any]],
    parsed_solution: dict[str, Any],
) -> str:
    if candidate["span_type"] not in {"number", "number_unit", "rate"}:
        return spn.NOT_A_NUMBER
    for number in question_numbers:
        if (
            candidate["start_char"] == number.get("start")
            and candidate["end_char"] == number.get("end")
        ):
            return number.get("status", spn.OFF_PATH)
    values = spn.normalize_number_text(candidate["span_text"])
    if not values:
        contained = ARABIC_NUMBER.search(candidate["span_text"]) or WORD_NUMBER_PATTERN.search(candidate["span_text"])
        if contained:
            values = spn.normalize_number_text(contained.group(0))
    value_multiset: Counter = Counter()
    for number in question_numbers:
        vals = number.get("values") or []
        if vals:
            value_multiset[vals[0]] += 1
    classification = spn.classify_number_value(
        values,
        value_multiset,
        set(parsed_solution["operand_values"]),
        set(parsed_solution["result_values"]),
    )
    return classification.get("status", spn.OFF_PATH)


def build_masked_question(question: str, start: int, end: int, mask_token: str = MASK_TOKEN) -> str:
    return question[:start] + mask_token + question[end:]


def _existing_fragility_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index = {}
    for row in records:
        key = (
            str(row.get("source_question_id")),
            str(row.get("span_text")),
            str(row.get("span_type")),
        )
        index.setdefault(key, row)
    return index


def canonical_question_records(
    *,
    subset_cases: list[dict[str, Any]],
    raw_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_by_id = {r.get("question_id"): r for r in raw_records or []}
    by_qid: dict[str, dict[str, Any]] = {}
    provenance: dict[str, set[str]] = defaultdict(set)
    for row in subset_cases:
        qid = row.get("source_question_id")
        if not qid:
            continue
        provenance[str(qid)].add("subset_cases")
        raw = raw_by_id.get(qid, {})
        if qid not in by_qid:
            by_qid[str(qid)] = {
                "source_question_id": str(qid),
                "question": row.get("question") or raw.get("question"),
                "gold_answer": raw.get("answer"),
                "gold_solution": row.get("gold_solution") or (raw.get("metadata") or {}).get("original_answer"),
            }
            if raw:
                provenance[str(qid)].add("raw_gsm8k")
    out = []
    for qid in sorted(by_qid):
        record = by_qid[qid]
        record["provenance"] = sorted(provenance[qid])
        out.append(record)
    return out


def build_multi_span_matrix(
    question_records: list[dict[str, Any]],
    *,
    risk_records: list[dict[str, Any]] | None = None,
    max_spans_per_question: int = 12,
    mask_token: str = MASK_TOKEN,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fragility_index = _existing_fragility_index(risk_records or [])
    grouped: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    audit = {
        "backend": BACKEND,
        "num_questions_input": len(question_records),
        "span_extraction_warnings": [],
        "char_alignment_failures": 0,
    }

    for q_index, question_record in enumerate(question_records):
        qid = question_record["source_question_id"]
        question = question_record["question"]
        solution = question_record.get("gold_solution") or ""
        candidates = extract_multi_span_candidates(question, max_spans_per_question=max_spans_per_question)
        census = spn.build_question_census(question, solution)
        parsed = spn.parse_solution_expressions(solution)

        span_records = []
        for idx, candidate in enumerate(candidates):
            candidate = dict(candidate)
            candidate["span_id"] = f"{qid}__{candidate['span_id']}"
            if candidate.get("parent_span_id"):
                candidate["parent_span_id"] = f"{qid}__{candidate['parent_span_id']}"
            candidate["child_span_ids"] = [f"{qid}__{sid}" for sid in candidate.get("child_span_ids", [])]
            if candidate.get("overlap_group_id"):
                candidate["overlap_group_id"] = f"{qid}__{candidate['overlap_group_id']}"

            if question[candidate["start_char"]:candidate["end_char"]] != candidate["span_text"]:
                audit["char_alignment_failures"] += 1
                alignment_status = "char_mismatch"
            else:
                alignment_status = "ok"

            solution_status = _number_status_for_candidate(candidate, census["number_spans"], parsed)
            fragility = fragility_index.get((qid, candidate["span_text"], candidate["span_type"]), {})
            surface = build_surface_features(candidate, question)
            labels = {
                "solution_path_status": solution_status,
                "weak_semantic_keyness": weak_semantic_keyness(candidate),
                "fragility_bucket_if_available": fragility.get("fragility_bucket"),
                "risk_strength_if_available": fragility.get("risk_strength"),
            }
            span = {
                "span_id": candidate["span_id"],
                "span_text": candidate["span_text"],
                "span_type": candidate["span_type"],
                "start_char": candidate["start_char"],
                "end_char": candidate["end_char"],
                "overlap_group_id": candidate.get("overlap_group_id"),
                "parent_span_id": candidate.get("parent_span_id"),
                "child_span_ids": candidate.get("child_span_ids", []),
                "surface_features": surface,
                "diagnostic_labels_for_eval_only": labels,
                "alignment_status": alignment_status,
                "source_rule": candidate.get("source_rule"),
            }
            span_records.append(span)
            flat.append(
                {
                    "source_question_id": qid,
                    "span_id": span["span_id"],
                    "span_text": span["span_text"],
                    "span_type": span["span_type"],
                    "question": question,
                    "masked_question": build_masked_question(
                        question,
                        span["start_char"],
                        span["end_char"],
                        mask_token=mask_token,
                    ),
                    "mask_text": mask_token,
                    "span_char_start": span["start_char"],
                    "span_char_end": span["end_char"],
                    "span_token_indices": None,
                    "mask_token_indices": None,
                    "alignment_status": alignment_status,
                    "overlap_group_id": span["overlap_group_id"],
                    "parent_span_id": span["parent_span_id"],
                    "child_span_ids": span["child_span_ids"],
                    "surface_features": surface,
                    "diagnostic_labels_for_eval_only": labels,
                }
            )

        grouped.append(
            {
                "source_question_id": qid,
                "question": question,
                "gold_answer": question_record.get("gold_answer"),
                "gold_solution": solution,
                "provenance": question_record.get("provenance", []),
                "candidate_spans": span_records,
            }
        )
    audit["num_questions_output"] = len(grouped)
    audit["num_flat_spans_output"] = len(flat)
    return grouped, flat, audit


def audit_formula_input_feature_names(flat_records: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = sorted(
        {
            name
            for row in flat_records
            for name in (row.get("surface_features") or {}).keys()
        }
    )
    leaked = []
    for name in feature_names:
        lower = name.lower()
        hits = [token for token in FORBIDDEN_FORMULA_INPUT_SUBSTRINGS if token in lower]
        if hits:
            leaked.append({"feature_name": name, "matched_substrings": hits})
    return {
        "backend": BACKEND,
        "num_formula_input_features": len(feature_names),
        "formula_input_feature_names": feature_names,
        "forbidden_substrings": list(FORBIDDEN_FORMULA_INPUT_SUBSTRINGS),
        "leaked_input_features": leaked,
        "passed": len(leaked) == 0,
        "note": "Diagnostic labels may contain forbidden substrings; surface formula inputs may not.",
    }


def coverage_report(grouped: list[dict[str, Any]], flat: list[dict[str, Any]], leakage_audit: dict[str, Any]) -> dict[str, Any]:
    counts = [len(row["candidate_spans"]) for row in grouped]
    type_counts = Counter(row["span_type"] for row in flat)
    number_rows = [
        row for row in flat
        if row["span_type"] in {"number", "number_unit", "rate"}
    ]
    num_questions = len(grouped)
    questions_with_less_than_3 = sum(1 for c in counts if c < 3)
    qids_by_type: dict[str, set[str]] = defaultdict(set)
    for row in flat:
        qids_by_type[row["span_type"]].add(row["source_question_id"])

    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        by_qid[row["source_question_id"]].append(row)
    both_on_off = 0
    on_path_questions = 0
    off_path_questions = 0
    for rows in by_qid.values():
        statuses = {
            (r.get("diagnostic_labels_for_eval_only") or {}).get("solution_path_status")
            for r in rows
        }
        if spn.ON_PATH in statuses:
            on_path_questions += 1
        if spn.OFF_PATH in statuses:
            off_path_questions += 1
        if spn.ON_PATH in statuses and spn.OFF_PATH in statuses:
            both_on_off += 1

    overlap_groups = Counter(
        row["overlap_group_id"]
        for row in flat
        if row.get("overlap_group_id")
    )
    report = {
        "backend": BACKEND,
        "num_questions": num_questions,
        "num_candidate_spans": len(flat),
        "mean_spans_per_question": round(sum(counts) / num_questions, 6) if num_questions else 0,
        "median_spans_per_question": median(counts) if counts else 0,
        "min_spans_per_question": min(counts) if counts else 0,
        "max_spans_per_question": max(counts) if counts else 0,
        "num_questions_with_less_than_3_spans": questions_with_less_than_3,
        "pct_questions_with_at_least_3_spans": round((num_questions - questions_with_less_than_3) / num_questions, 6) if num_questions else 0,
        "span_type_distribution": dict(sorted(type_counts.items())),
        "number_span_distribution": {
            "num_number_like_spans": len(number_rows),
            "by_type": {
                key: type_counts.get(key, 0)
                for key in ["number", "number_unit", "rate"]
            },
            "questions_with_number_like_span": len({r["source_question_id"] for r in number_rows}),
        },
        "questions_with_on_path_number": on_path_questions,
        "questions_with_off_path_number": off_path_questions,
        "questions_with_on_path_and_off_path_number": both_on_off,
        "questions_with_comparison": len(qids_by_type["comparison"]),
        "questions_with_negation": len(qids_by_type["negation"]),
        "questions_with_condition": len(qids_by_type["condition"]),
        "questions_with_operation": len(qids_by_type["operation"]),
        "questions_with_question_target": len(qids_by_type["question_target"]),
        "questions_with_object_or_entity": len(qids_by_type["object"]),
        "overlap_group_statistics": {
            "num_overlap_groups": len(overlap_groups),
            "max_group_size": max(overlap_groups.values()) if overlap_groups else 0,
            "num_spans_in_overlap_groups": sum(overlap_groups.values()),
        },
        "no_leakage_feature_audit": leakage_audit,
    }
    report["gate"] = evaluate_2ja_gate(report, char_alignment_failure_rate=0.0)
    return report


def evaluate_2ja_gate(report: dict[str, Any], *, char_alignment_failure_rate: float) -> dict[str, Any]:
    checks = {
        "1_at_least_90pct_questions_have_3_spans": {
            "pass": report["pct_questions_with_at_least_3_spans"] >= 0.90,
            "value": report["pct_questions_with_at_least_3_spans"],
        },
        "2_mean_spans_between_4_and_10": {
            "pass": 4 <= report["mean_spans_per_question"] <= 10,
            "value": report["mean_spans_per_question"],
        },
        "3_number_spans_high_coverage": {
            "pass": report["number_span_distribution"]["questions_with_number_like_span"] >= int(0.90 * report["num_questions"]),
            "value": report["number_span_distribution"]["questions_with_number_like_span"],
        },
        "4_enough_on_path_off_path_number_pairs": {
            "pass": report["questions_with_on_path_and_off_path_number"] >= max(20, int(0.08 * report["num_questions"])),
            "value": report["questions_with_on_path_and_off_path_number"],
        },
        "5_non_number_candidate_types_not_empty": {
            "pass": all(
                report[f"questions_with_{name}"] > 0
                for name in ["comparison", "negation", "condition", "operation", "question_target", "object_or_entity"]
            ),
            "value": {
                key: report[key]
                for key in [
                    "questions_with_comparison",
                    "questions_with_negation",
                    "questions_with_condition",
                    "questions_with_operation",
                    "questions_with_question_target",
                    "questions_with_object_or_entity",
                ]
            },
        },
        "6_grouped_and_flat_outputs_generated": {
            "pass": report["num_questions"] > 0 and report["num_candidate_spans"] > 0,
            "value": {"num_questions": report["num_questions"], "num_candidate_spans": report["num_candidate_spans"]},
        },
        "7_no_leakage_surface_feature_audit": {
            "pass": report["no_leakage_feature_audit"]["passed"],
            "value": len(report["no_leakage_feature_audit"]["leaked_input_features"]),
        },
        "8_char_alignment_failure_rate_acceptable": {
            "pass": char_alignment_failure_rate <= 0.01,
            "value": char_alignment_failure_rate,
        },
    }
    return {
        "checks": checks,
        "num_checks_passed": sum(1 for check in checks.values() if check["pass"]),
        "num_checks_total": len(checks),
        "passed": all(check["pass"] for check in checks.values()),
        "if_failed": "do not run hidden/attention scoring; fix candidate extraction first",
    }


def keyness_label_report(flat: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(
        (row["diagnostic_labels_for_eval_only"] or {}).get("solution_path_status")
        for row in flat
    )
    weak = Counter(
        (row["diagnostic_labels_for_eval_only"] or {}).get("weak_semantic_keyness")
        for row in flat
    )
    return {
        "backend": BACKEND,
        "solution_path_status_counts_eval_only": dict(statuses),
        "weak_semantic_keyness_counts_eval_only": dict(weak),
        "note": (
            "solution_path_status and weak semantic keyness are evaluation-only diagnostics. "
            "They must not be used as formula inputs or probe features."
        ),
    }


def surface_ranking_baseline(flat: list[dict[str, Any]]) -> dict[str, Any]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        by_qid[row["source_question_id"]].append(row)
    on_off_pairs = 0
    on_off_correct = 0.0
    topk = {1: {"eligible": 0, "hit": 0}, 2: {"eligible": 0, "hit": 0}, 3: {"eligible": 0, "hit": 0}}
    off_top = {1: 0, 2: 0, 3: 0}
    for rows in by_qid.values():
        ranked = sorted(
            rows,
            key=lambda r: -float((r.get("surface_features") or {}).get("surface_keyness_proxy", 0.0)),
        )
        ons = [r for r in rows if r["diagnostic_labels_for_eval_only"]["solution_path_status"] == spn.ON_PATH]
        offs = [r for r in rows if r["diagnostic_labels_for_eval_only"]["solution_path_status"] == spn.OFF_PATH]
        if ons:
            for k in topk:
                topk[k]["eligible"] += 1
                if any(r in ons for r in ranked[:k]):
                    topk[k]["hit"] += 1
        for k in off_top:
            if any(r in offs for r in ranked[:k]):
                off_top[k] += 1
        for on in ons:
            so = float(on["surface_features"]["surface_keyness_proxy"])
            for off in offs:
                sf = float(off["surface_features"]["surface_keyness_proxy"])
                on_off_pairs += 1
                if so == sf:
                    on_off_correct += 0.5
                elif so > sf:
                    on_off_correct += 1.0
    return {
        "backend": BACKEND,
        "score": "surface_keyness_proxy",
        "same_question_on_path_vs_off_path_pairwise": round(on_off_correct / on_off_pairs, 6) if on_off_pairs else None,
        "num_on_off_pairs": on_off_pairs,
        "per_question_topk_on_path_coverage": {
            str(k): {
                "eligible_questions": v["eligible"],
                "hits": v["hit"],
                "coverage": round(v["hit"] / v["eligible"], 6) if v["eligible"] else None,
            }
            for k, v in topk.items()
        },
        "off_path_number_selected_rate": {
            str(k): round(off_top[k] / len(by_qid), 6) if by_qid else None
            for k in off_top
        },
    }


def render_review_gate_2ja(report: dict[str, Any], baseline: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# Sprint 2J-A Multi-Span Matrix Review Gate",
        "",
        "## Verdict",
        "",
        f"- passed: `{str(gate['passed']).lower()}`",
        f"- checks: `{gate['num_checks_passed']}/{gate['num_checks_total']}`",
        "- ready_for_2000_rerun: `false`",
        "- do_not_enter_sprint_3A: `true`",
        "",
        "## Coverage",
        "",
        f"- num_questions: `{report['num_questions']}`",
        f"- num_candidate_spans: `{report['num_candidate_spans']}`",
        f"- mean_spans_per_question: `{report['mean_spans_per_question']}`",
        f"- median_spans_per_question: `{report['median_spans_per_question']}`",
        f"- min/max spans: `{report['min_spans_per_question']}` / `{report['max_spans_per_question']}`",
        f"- questions_with_on_path_and_off_path_number: `{report['questions_with_on_path_and_off_path_number']}`",
        "",
        "## Surface Ranking Baseline",
        "",
        f"- same_question_on_path_vs_off_path_pairwise: `{baseline['same_question_on_path_vs_off_path_pairwise']}`",
        f"- num_on_off_pairs: `{baseline['num_on_off_pairs']}`",
        "",
        "## Gate Checks",
        "",
    ]
    for name, check in gate["checks"].items():
        lines.append(f"- `{name}`: pass=`{str(check['pass']).lower()}`, value=`{check['value']}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 2J-A is text-only candidate matrix construction.",
            "- No model forward was run.",
            "- No recovery, hidden-state cache, attention cache, probe training, attention steering, CoT, trajectory, NLA, or causal attribution was run.",
        ]
    )
    if not gate["passed"]:
        lines.extend(["", "## Next", "", "- Do not run 2J-B; fix candidate extraction coverage first."])
    return "\n".join(lines) + "\n"


def run_2ja_matrix(
    *,
    subset_cases_path: str | Path,
    raw_gsm8k_path: str | Path,
    risk_strength_path: str | Path,
    output_dir: str | Path,
    max_spans_per_question: int = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {out_dir}")
    ensure_dir(out_dir)

    subset_cases = read_jsonl(subset_cases_path)
    raw_records = read_jsonl(raw_gsm8k_path) if Path(raw_gsm8k_path).exists() else []
    risk_records = read_jsonl(risk_strength_path) if Path(risk_strength_path).exists() else []
    questions = canonical_question_records(subset_cases=subset_cases, raw_records=raw_records)
    grouped, flat, span_audit = build_multi_span_matrix(
        questions,
        risk_records=risk_records,
        max_spans_per_question=max_spans_per_question,
    )
    leakage = audit_formula_input_feature_names(flat)
    report = coverage_report(grouped, flat, leakage)
    char_failure_rate = (
        span_audit["char_alignment_failures"] / len(flat)
        if flat
        else 0.0
    )
    report["char_alignment_failure_rate"] = round(char_failure_rate, 6)
    report["gate"] = evaluate_2ja_gate(report, char_alignment_failure_rate=char_failure_rate)
    keyness = keyness_label_report(flat)
    baseline = surface_ranking_baseline(flat)
    span_audit.update(
        {
            "num_formula_input_features": leakage["num_formula_input_features"],
            "leakage_passed": leakage["passed"],
            "max_spans_per_question": max_spans_per_question,
            "input_paths": {
                "subset_cases": str(subset_cases_path),
                "raw_gsm8k": str(raw_gsm8k_path),
                "risk_strength_dataset": str(risk_strength_path),
            },
        }
    )

    write_jsonl(grouped, out_dir / "multi_span_grouped_matrix.jsonl")
    write_jsonl(flat, out_dir / "multi_span_flat_matrix.jsonl")
    write_json(report, out_dir / "candidate_span_coverage_report.json")
    write_json(span_audit, out_dir / "span_extraction_audit.json")
    write_json(keyness, out_dir / "keyness_label_report.json")
    write_json(baseline, out_dir / "surface_ranking_baseline_report.json")
    (out_dir / "review_gate_multi_span_matrix.md").write_text(
        render_review_gate_2ja(report, baseline),
        encoding="utf-8",
    )

    return {
        "backend": BACKEND,
        "output_dir": str(out_dir),
        "num_questions": len(grouped),
        "num_candidate_spans": len(flat),
        "mean_spans_per_question": report["mean_spans_per_question"],
        "questions_with_on_path_and_off_path_number": report["questions_with_on_path_and_off_path_number"],
        "gate_passed": report["gate"]["passed"],
        "checks_passed": report["gate"]["num_checks_passed"],
        "checks_total": report["gate"]["num_checks_total"],
        "ready_for_2000_rerun": False,
        "do_not_enter_sprint_3A": True,
    }
