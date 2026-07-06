"""Sprint 2H-B task 1: solution-path number extraction and on/off-path labeling.

Pure text-level analysis. Parses GSM8K gold rationales (``<<a op b = c>>`` calc
expressions), normalizes numeric forms, and labels every question number span as
``on_solution_path_number`` / ``off_solution_path_number`` / ``ambiguous_number``.

This module never touches hidden states, recovery outputs, or drift labels. It is
one of the label-construction signals, not a probe input.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from recover_attention.candidate_extraction import ARABIC_NUMBER_PATTERN


ON_PATH = "on_solution_path_number"
OFF_PATH = "off_solution_path_number"
AMBIGUOUS = "ambiguous_number"
NOT_A_NUMBER = "not_a_number"

# Word-form numbers we normalize when comparing spans/fillers to solution operands.
WORD_NUMBERS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000,
    "dozen": 12,
}

# Implicit multipliers/relations. Reported separately; never forced onto spans.
IMPLICIT_NUMBERS: dict[str, float] = {
    "twice": 2.0,
    "double": 2.0,
    "doubled": 2.0,
    "triple": 3.0,
    "tripled": 3.0,
    "quadruple": 4.0,
    "half": 0.5,
    "quarter": 0.25,
    "third": 1.0 / 3.0,
}

CALC_EXPR_PATTERN = re.compile(r"<<([^>]*)>>")
# Unsigned magnitudes only: a leading '-' in an expression is a subtraction
# operator, not a sign, and question numbers (ARABIC_NUMBER_PATTERN) are unsigned.
NUMERIC_TOKEN_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
PERCENT_RELATION_PATTERN = re.compile(r"\bpercent\b|%", re.IGNORECASE)

_NUM_TOLERANCE = 1e-6


def _round_value(value: float) -> float:
    return round(float(value), 6)


def normalize_number_text(text: str) -> list[float]:
    """Return candidate normalized numeric values for a raw span/token string.

    ``$30`` -> [30.0]; ``1,000`` -> [1000.0]; ``50%`` -> [50.0, 0.5];
    ``3.0`` -> [3.0]; ``twelve`` -> [12.0]. Returns [] if nothing numeric.
    """
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    lowered = raw.lower().strip().strip(".")
    if lowered in WORD_NUMBERS:
        return [_round_value(WORD_NUMBERS[lowered])]

    is_percent = "%" in raw or raw.lower().endswith("percent")
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
    cleaned = re.sub(r"(?i)percent", "", cleaned).strip()

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return []
    base = float(match.group(0))

    values = [_round_value(base)]
    if is_percent:
        values.append(_round_value(base / 100.0))
    # de-dup while preserving order
    seen: set[float] = set()
    out: list[float] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_solution_expressions(solution_text: str) -> dict[str, Any]:
    """Parse ``<<...>>`` calc expressions from a gold rationale.

    Returns operand values (numbers the solution consumes) and result values
    (computed intermediates / final), as normalized float sets.
    """
    operand_values: set[float] = set()
    result_values: set[float] = set()
    expressions: list[dict[str, Any]] = []

    if not solution_text:
        return {
            "operand_values": operand_values,
            "result_values": result_values,
            "expressions": expressions,
            "num_expressions": 0,
        }

    for raw_expr in CALC_EXPR_PATTERN.findall(solution_text):
        expr = raw_expr.strip()
        if "=" in expr:
            lhs, _, rhs = expr.partition("=")
        else:
            lhs, rhs = expr, ""
        lhs_nums = [float(t.replace(",", "")) for t in NUMERIC_TOKEN_PATTERN.findall(lhs)]
        rhs_nums = [float(t.replace(",", "")) for t in NUMERIC_TOKEN_PATTERN.findall(rhs)]
        for value in lhs_nums:
            operand_values.add(_round_value(value))
        for value in rhs_nums:
            result_values.add(_round_value(value))
        expressions.append({"expr": expr, "operands": lhs_nums, "results": rhs_nums})

    return {
        "operand_values": operand_values,
        "result_values": result_values,
        "expressions": expressions,
        "num_expressions": len(expressions),
    }


def extract_question_numbers(question: str) -> list[dict[str, Any]]:
    """Extract explicit numeric spans from a question with char offsets."""
    numbers: list[dict[str, Any]] = []
    for match in ARABIC_NUMBER_PATTERN.finditer(question or ""):
        raw = match.group(0)
        values = normalize_number_text(raw)
        numbers.append(
            {
                "text": raw,
                "start": match.start(),
                "end": match.end(),
                "values": values,
            }
        )
    return numbers


def extract_implicit_numbers(question: str) -> list[dict[str, Any]]:
    """Find implicit multipliers/relations (twice, half, percent, ...)."""
    found: list[dict[str, Any]] = []
    lowered = (question or "").lower()
    for word, value in IMPLICIT_NUMBERS.items():
        for match in re.finditer(rf"\b{re.escape(word)}\b", lowered):
            found.append({"text": word, "start": match.start(), "value": _round_value(value)})
    if PERCENT_RELATION_PATTERN.search(question or ""):
        found.append({"text": "percent", "start": -1, "value": None, "relation": "divide_by_100"})
    return found


def _values_match(values: list[float], target_set: set[float]) -> bool:
    for value in values:
        for target in target_set:
            if abs(value - target) <= _NUM_TOLERANCE:
                return True
    return False


def classify_number_value(
    values: list[float],
    question_value_multiset: Counter,
    operand_values: set[float],
    result_values: set[float],
) -> dict[str, Any]:
    """Classify one numeric span (by its candidate values) as on/off/ambiguous.

    ``question_value_multiset`` counts how many explicit question numbers share
    each primary value, used to detect repeated-value ambiguity.
    """
    if not values:
        return {"status": OFF_PATH, "reason": "no_numeric_value", "in_operands": False}

    in_operands = _values_match(values, operand_values)
    in_results = _values_match(values, result_values)
    primary = values[0]
    occurrences = question_value_multiset.get(primary, 0)

    if in_operands and occurrences > 1:
        return {
            "status": AMBIGUOUS,
            "reason": "operand_match_but_repeated_value",
            "in_operands": True,
            "in_results": in_results,
            "occurrences": occurrences,
        }
    if in_operands:
        return {
            "status": ON_PATH,
            "reason": "operand_match_unique",
            "in_operands": True,
            "in_results": in_results,
            "occurrences": occurrences,
        }
    if in_results:
        return {
            "status": AMBIGUOUS,
            "reason": "matches_result_only",
            "in_operands": False,
            "in_results": True,
            "occurrences": occurrences,
        }
    return {
        "status": OFF_PATH,
        "reason": "no_operand_match",
        "in_operands": False,
        "in_results": False,
        "occurrences": occurrences,
    }


def build_question_census(question: str, solution_text: str) -> dict[str, Any]:
    """Full text-level number census for one question."""
    parsed = parse_solution_expressions(solution_text)
    numbers = extract_question_numbers(question)

    value_multiset: Counter = Counter()
    for number in numbers:
        if number["values"]:
            value_multiset[number["values"][0]] += 1

    labeled: list[dict[str, Any]] = []
    status_counts: Counter = Counter()
    for number in numbers:
        classification = classify_number_value(
            number["values"],
            value_multiset,
            parsed["operand_values"],
            parsed["result_values"],
        )
        record = dict(number)
        record.update(classification)
        labeled.append(record)
        status_counts[classification["status"]] += 1

    implicit = extract_implicit_numbers(question)

    return {
        "num_expressions": parsed["num_expressions"],
        "operand_values": sorted(parsed["operand_values"]),
        "result_values": sorted(parsed["result_values"]),
        "number_spans": labeled,
        "implicit_path_numbers": implicit,
        "status_counts": dict(status_counts),
        "num_number_spans": len(labeled),
        "num_on_path": status_counts.get(ON_PATH, 0),
        "num_off_path": status_counts.get(OFF_PATH, 0),
        "num_ambiguous": status_counts.get(AMBIGUOUS, 0),
    }


def classify_chosen_span_solution_path(
    span_text: str,
    span_type: str,
    question: str,
    solution_text: str,
) -> dict[str, Any]:
    """Classify the single 2G-chosen span against the solution path.

    Non-number spans return ``not_a_number`` (solution-path membership is only
    defined for numeric spans in this round).
    """
    if span_type != "number":
        return {"status": NOT_A_NUMBER, "reason": f"span_type={span_type}", "values": []}

    parsed = parse_solution_expressions(solution_text)
    numbers = extract_question_numbers(question)
    value_multiset: Counter = Counter()
    for number in numbers:
        if number["values"]:
            value_multiset[number["values"][0]] += 1

    values = normalize_number_text(span_text)
    classification = classify_number_value(
        values, value_multiset, parsed["operand_values"], parsed["result_values"]
    )
    classification["values"] = values
    return classification
