"""CyberMetric canonical MCQ data helpers for Sprint 4B-1."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from recover_attention.data_io import read_json

OPTION_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CYBERMETRIC_LICENSE_NOTE = (
    "No explicit LICENSE file was present in the downloaded CyberMetric repository "
    "root during the local source audit; verify publication-use terms and citation."
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_seed(example_id: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_cybermetric_records(path: Path) -> list[dict]:
    """Load a CyberMetric JSON list without silently discarding malformed entries."""
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("questions")
    if not isinstance(payload, list):
        raise ValueError(f"CyberMetric input {path} must contain a JSON list")
    if not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"CyberMetric input {path} must contain only object records")
    return payload


def normalize_cybermetric_record(
    raw_record: dict,
    *,
    original_index: int,
    source: str,
) -> dict:
    """Normalize one raw CyberMetric object and validate its MCQ structure."""
    if not isinstance(raw_record, dict):
        raise ValueError(f"record {original_index} in {source} must be an object")
    question = raw_record.get("question")
    answers = raw_record.get("answers")
    solution = raw_record.get("solution")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"record {original_index} in {source} has missing or empty question")
    if not isinstance(answers, dict) or len(answers) < 2:
        raise ValueError(f"record {original_index} in {source} has fewer than two answer options")
    normalized_answers: dict[str, str] = {}
    for raw_label, raw_text in answers.items():
        label = str(raw_label).strip().upper()
        if not label or not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"record {original_index} in {source} has an invalid answer option")
        if label in normalized_answers:
            raise ValueError(f"record {original_index} in {source} has duplicate option {label!r}")
        normalized_answers[label] = raw_text.strip()
    gold_original_label = str(solution or "").strip().upper()
    if gold_original_label not in normalized_answers:
        raise ValueError(
            f"record {original_index} in {source} has missing or invalid solution {solution!r}"
        )
    ordered_labels = sorted(normalized_answers)
    gold_original_position = ordered_labels.index(gold_original_label)
    raw_category = raw_record.get("category", raw_record.get("family"))
    if raw_category is not None:
        raw_category = str(raw_category).strip() or None
    return {
        "question": question.strip(),
        "answers": normalized_answers,
        "gold_original_label": gold_original_label,
        "gold_original_position": gold_original_position,
        "raw_category": raw_category,
        "original_index": original_index,
        "source": source,
    }


def build_candidate_choices(raw_record: dict) -> tuple[list[dict], int]:
    """Build semantic choices in original option order."""
    answers = raw_record.get("answers")
    gold_label = raw_record.get("gold_original_label", raw_record.get("solution"))
    if not isinstance(answers, dict):
        raise ValueError("normalized CyberMetric record field 'answers' must be a dict")
    ordered_labels = sorted(str(label).upper() for label in answers)
    normalized_gold = str(gold_label or "").upper()
    if normalized_gold not in ordered_labels:
        raise ValueError(f"gold option {gold_label!r} is not present in answers")
    choices = [
        {
            "choice": label,
            "label_id": None,
            "label_text": str(answers[label]).strip(),
            "original_position": position,
        }
        for position, label in enumerate(ordered_labels)
    ]
    return choices, ordered_labels.index(normalized_gold)


def shuffle_candidate_choices(
    choices: list[dict],
    gold_original_position: int,
    *,
    example_id: str,
    seed: int,
) -> tuple[list[dict], str]:
    """Shuffle a copy of choices using a stable per-example seed."""
    if not isinstance(gold_original_position, int) or isinstance(gold_original_position, bool):
        raise ValueError("gold_original_position must be an int")
    if gold_original_position < 0 or gold_original_position >= len(choices):
        raise ValueError("gold_original_position is outside the choice list")
    copied = deepcopy(choices)
    original_positions = [choice.get("original_position") for choice in copied]
    if len(set(original_positions)) != len(copied):
        raise ValueError("choice original_position values must be unique")
    rng = random.Random(_stable_seed(example_id, seed, "option_shuffle"))
    rng.shuffle(copied)
    labels = list(OPTION_LETTERS[: len(copied)])
    gold_label: str | None = None
    for label, choice in zip(labels, copied):
        choice["choice"] = label
        if choice["original_position"] == gold_original_position:
            gold_label = label
    if gold_label is None:
        raise ValueError("gold semantic option was lost during shuffle")
    return copied, gold_label


def _normalize_question_for_grouping(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def build_group_id(raw_record: dict, normalized_question: str) -> str:
    """Build a source-group id or deterministic question-derived proxy."""
    for key in ("group_id", "family", "category"):
        value = raw_record.get(key)
        if isinstance(value, str) and value.strip():
            return f"cybermetric_{key}_{stable_hash(value.strip().casefold())}"
    return f"cybermetric_question_{stable_hash(_normalize_question_for_grouping(normalized_question))}"


def to_canonical_cyber_sample(
    raw_record: dict,
    *,
    original_index: int,
    source: str,
    shuffle_seed: int,
) -> dict:
    """Convert one raw record into the Sprint 4B-1 canonical schema."""
    normalized = normalize_cybermetric_record(
        raw_record,
        original_index=original_index,
        source=source,
    )
    choices, gold_original_position = build_candidate_choices(normalized)
    example_id = f"cybermetric_{original_index:06d}"
    shuffled, gold_label = shuffle_candidate_choices(
        choices,
        gold_original_position,
        example_id=example_id,
        seed=shuffle_seed,
    )
    gold_choice = next(choice for choice in shuffled if choice["choice"] == gold_label)
    return {
        "example_id": example_id,
        "dataset": "cybermetric",
        "source": source,
        "group_id": build_group_id(raw_record, normalized["question"]),
        "task_type": "multiple_choice_qa",
        "input_text": "",
        "question": normalized["question"],
        "candidate_labels": [choice["choice"] for choice in shuffled],
        "candidate_choices": shuffled,
        "gold_label": gold_label,
        "gold_label_id": gold_choice["label_id"],
        "gold_label_text": gold_choice["label_text"],
        "label_space": "mcq_option_letter",
        "metadata": {
            "original_index": original_index,
            "original_gold_position": gold_original_position,
            "shuffled_gold_position": shuffled.index(gold_choice),
            "option_shuffle_seed": shuffle_seed,
            "split_seed": None,
            "split": None,
            "raw_category": normalized["raw_category"],
        },
    }


def _validate_split_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, dev_ratio, test_ratio)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 for value in ratios):
        raise ValueError("split ratios must be non-negative finite numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0")


def grouped_split(
    records: list[dict],
    *,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """Assign whole groups to deterministic train/dev/test splits."""
    _validate_split_ratios(train_ratio, dev_ratio, test_ratio)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("every record must have a non-empty group_id")
        grouped[group_id].append(record)
    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    total_groups = len(group_ids)
    train_end = int(total_groups * train_ratio)
    dev_end = train_end + int(total_groups * dev_ratio)
    split_by_group = {
        group_id: (
            "train" if index < train_end else "dev" if index < dev_end else "test"
        )
        for index, group_id in enumerate(group_ids)
    }
    output: list[dict] = []
    for record in records:
        copied = deepcopy(record)
        copied["metadata"]["split"] = split_by_group[copied["group_id"]]
        copied["metadata"]["split_seed"] = seed
        output.append(copied)
    return output


def build_mcq_prompt(record: dict) -> str:
    """Build the fixed Sprint 4B-1 prompt without calling a model."""
    parts: list[str] = []
    input_text = record.get("input_text")
    if isinstance(input_text, str) and input_text.strip():
        parts.extend([input_text.strip(), ""])
    parts.extend([f"Question: {record['question']}", "", "Options:"])
    for choice in record["candidate_choices"]:
        parts.append(f"{choice['choice']}. {choice['label_text']}")
    parts.extend(
        [
            "",
            "Think briefly step by step, then answer with exactly one letter as:",
            "Answer: <letter>",
        ]
    )
    return "\n".join(parts)


def select_grouped_smoke_sample(records: list[dict], *, sample_size: int, seed: int) -> list[dict]:
    """Select a reproducible non-prefix sample without splitting groups."""
    if sample_size < 1 or sample_size > len(records):
        raise ValueError("sample_size must be between 1 and the number of records")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    selected: list[dict] = []
    for group_id in group_ids:
        group = groups[group_id]
        if len(selected) + len(group) <= sample_size:
            selected.extend(group)
        if len(selected) == sample_size:
            break
    if len(selected) != sample_size:
        raise ValueError(
            f"cannot select exactly {sample_size} records without breaking group boundaries"
        )
    return selected


def audit_cyber_samples(records: list[dict]) -> dict:
    """Compute canonical sample and grouped-split audit fields."""
    example_ids = [record["example_id"] for record in records]
    split_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_groups[record["group_id"]].add(record["metadata"]["split"])
    leakage_count = sum(len(splits) > 1 for splits in split_groups.values())
    split_counts = Counter(record["metadata"]["split"] for record in records)
    split_group_counts = Counter(
        next(iter(splits)) for splits in split_groups.values() if len(splits) == 1
    )
    option_counts = Counter(len(record["candidate_choices"]) for record in records)
    label_counts = Counter(record["gold_label"] for record in records)
    return {
        "num_output_records": len(records),
        "num_options_distribution": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "duplicate_example_id_count": len(example_ids) - len(set(example_ids)),
        "group_count": len(split_groups),
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_counts": dict(sorted(split_group_counts.items())),
        "group_leakage_count": leakage_count,
        "label_distribution": dict(sorted(label_counts.items())),
    }


def option_position_bias_pre_model_report(records: list[dict], *, shuffle_seed: int) -> dict:
    """Audit option positions and semantic mapping without model predictions."""
    labels = sorted({label for record in records for label in record["candidate_labels"]})
    gold_counts = Counter(record["gold_label"] for record in records)
    original_counts = Counter(
        str(record["metadata"]["original_gold_position"]) for record in records
    )
    shuffled_counts = Counter(
        str(record["metadata"]["shuffled_gold_position"]) for record in records
    )
    proportions = {
        label: gold_counts.get(label, 0) / len(records) if records else 0.0
        for label in labels
    }
    mapping: dict[str, Counter[str]] = defaultdict(Counter)
    fixed_seed_reproduced = True
    different_seed_changed = False
    for record in records:
        original_choices = sorted(
            record["candidate_choices"], key=lambda choice: choice["original_position"]
        )
        gold_original_position = record["metadata"]["original_gold_position"]
        same, _ = shuffle_candidate_choices(
            original_choices,
            gold_original_position,
            example_id=record["example_id"],
            seed=shuffle_seed,
        )
        other, _ = shuffle_candidate_choices(
            original_choices,
            gold_original_position,
            example_id=record["example_id"],
            seed=shuffle_seed + 1,
        )
        fixed_seed_reproduced &= same == record["candidate_choices"]
        different_seed_changed |= other != record["candidate_choices"]
        for choice in record["candidate_choices"]:
            semantic_key = choice["label_text"]
            mapping[semantic_key][choice["choice"]] += 1
    repeated_fixed = sorted(
        semantic
        for semantic, counts in mapping.items()
        if sum(counts.values()) >= 2 and len(counts) == 1
    )
    severe = bool(proportions and max(proportions.values()) > 0.40)
    warnings: list[str] = []
    if severe:
        warnings.append("gold option-letter proportion exceeds 0.40")
    if repeated_fixed:
        warnings.append("one or more repeated semantic option texts map to one letter only")
    return {
        "gold_choice_distribution": {
            label: gold_counts.get(label, 0) for label in labels
        },
        "gold_choice_proportion": proportions,
        "original_gold_position_distribution": dict(sorted(original_counts.items())),
        "shuffled_gold_position_distribution": dict(sorted(shuffled_counts.items())),
        "shuffle_seed": shuffle_seed,
        "position_balance_warning_threshold": 0.40,
        "position_balance_warnings": warnings,
        "severe_position_imbalance": severe,
        "semantic_label_to_option_letter_mapping_counts": {
            semantic: dict(sorted(counts.items()))
            for semantic, counts in sorted(mapping.items())
        },
        "semantic_labels_always_mapped_to_one_option_letter": repeated_fixed,
        "any_repeated_semantic_label_always_mapped_to_one_option_letter": bool(repeated_fixed),
        "option_order_deterministic_under_fixed_seed": fixed_seed_reproduced,
        "option_order_changes_under_different_seed": different_seed_changed,
    }


# Compatibility helpers for the superseded Sprint 4B smoke script. They are not
# used by the Sprint 4B-1 preparation CLI.
def load_raw_dataset(dataset: str, raw_root: str | Path = "data/raw/cyber") -> list[dict]:
    if dataset.lower() != "cybermetric":
        raise ValueError(f"unsupported dataset: {dataset}")
    root = Path(raw_root) / "cybermetric"
    for filename in (
        "CyberMetric-2000-v1.json",
        "CyberMetric-500-v1.json",
        "CyberMetric-10000-v1.json",
    ):
        path = root / filename
        if path.exists():
            return load_cybermetric_records(path)
    raise FileNotFoundError(f"no CyberMetric JSON file found under {root}")


def to_canonical_schema(
    records: list[dict],
    *,
    dataset: str = "cybermetric",
    limit: int | None = None,
    seed: int = 42,
    shuffle_options: bool = True,
) -> list[dict]:
    if dataset.lower() != "cybermetric":
        raise ValueError(f"unsupported dataset: {dataset}")
    source = "CyberMetric"
    selected = records if limit is None else records[:limit]
    return [
        to_canonical_cyber_sample(
            record,
            original_index=index,
            source=source,
            shuffle_seed=seed if shuffle_options else 0,
        )
        for index, record in enumerate(selected)
    ]


def audit_dataset(records: list[dict], *, dataset: str) -> dict:
    return {"dataset": dataset, **audit_cyber_samples(grouped_split(records))}


def option_position_bias_report(
    records: list[dict],
    *,
    greedy_labels: list[str | None] | None = None,
    sampled_labels: list[str | None] | None = None,
) -> dict:
    report = option_position_bias_pre_model_report(
        records,
        shuffle_seed=records[0]["metadata"]["option_shuffle_seed"] if records else 42,
    )
    report["greedy_predicted_choice_distribution"] = dict(
        Counter(label for label in (greedy_labels or []) if label)
    )
    report["sampled_predicted_choice_distribution"] = dict(
        Counter(label for label in (sampled_labels or []) if label)
    )
    report["severe_warning"] = report["severe_position_imbalance"]
    return report
