"""Cyber MCQ data helpers for Sprint 4B.

The module converts raw cyber multiple-choice datasets into a small canonical
schema with option-letter labels.  Option letters are used as readout targets;
the original option text is preserved as semantic label text for audit and
future error analysis.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND = "cyber_dataset_canonical_schema_v0"
OPTION_LETTERS = ("A", "B", "C", "D")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stable_int_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def load_raw_dataset(dataset: str, raw_root: str | Path = "data/raw/cyber") -> list[dict[str, Any]]:
    """Load raw records for a supported cyber dataset."""

    dataset = dataset.lower()
    root = Path(raw_root)
    if dataset != "cybermetric":
        raise ValueError(f"unsupported dataset for Sprint 4B smoke: {dataset}")
    candidates = [
        root / "cybermetric" / "CyberMetric-500-v1.json",
        root / "cybermetric" / "CyberMetric-2000-v1.json",
        root / "cybermetric" / "CyberMetric-10000-v1.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("questions") if isinstance(payload, dict) else payload
            if isinstance(records, list):
                return [r for r in records if isinstance(r, dict)]
    expected = ", ".join(p.as_posix() for p in candidates)
    raise FileNotFoundError(f"missing CyberMetric raw file; expected one of: {expected}")


def _choice_order(example_id: str, seed: int, letters: list[str]) -> list[str]:
    order = list(letters)
    rng = random.Random(stable_int_seed(f"{seed}:{example_id}:option_order"))
    rng.shuffle(order)
    return order


def to_canonical_schema(
    records: list[dict[str, Any]],
    *,
    dataset: str = "cybermetric",
    limit: int | None = None,
    seed: int = 44020,
    shuffle_options: bool = True,
) -> list[dict[str, Any]]:
    """Convert raw MCQ records to the canonical Sprint 4B schema."""

    out: list[dict[str, Any]] = []
    for raw_index, record in enumerate(records):
        question = str(record.get("question") or "").strip()
        answers = record.get("answers")
        solution = str(record.get("solution") or "").strip().upper()
        if not question or not isinstance(answers, dict) or solution not in answers:
            continue
        raw_letters = [letter for letter in OPTION_LETTERS if letter in answers]
        if len(raw_letters) < 2 or solution not in raw_letters:
            continue
        base_id = f"{dataset}:{raw_index}:{stable_hash(question)}"
        order = _choice_order(base_id, seed, raw_letters) if shuffle_options else list(raw_letters)
        candidate_choices = []
        gold_label = None
        for new_letter, old_letter in zip(raw_letters, order):
            label_text = str(answers[old_letter]).strip()
            candidate_choices.append(
                {
                    "choice": new_letter,
                    "label_id": old_letter,
                    "label_text": label_text,
                    "source_choice": old_letter,
                }
            )
            if old_letter == solution:
                gold_label = new_letter
        if gold_label is None:
            continue
        gold_choice = next(c for c in candidate_choices if c["choice"] == gold_label)
        out.append(
            {
                "example_id": base_id,
                "task_type": "multiple_choice_qa",
                "dataset": dataset,
                "input_text": "",
                "question": question,
                "candidate_labels": [c["choice"] for c in candidate_choices],
                "gold_label": gold_label,
                "gold_label_id": gold_choice["label_id"],
                "gold_label_text": gold_choice["label_text"],
                "label_space": "mcq_option_letter",
                "candidate_choices": candidate_choices,
                "evidence_spans": [],
                "metadata": {
                    "source": dataset,
                    "raw_index": raw_index,
                    "question_family": stable_hash(question[:96].lower()),
                    "option_order_randomized": bool(shuffle_options),
                    "source_solution": solution,
                },
            }
        )
        if limit is not None and len(out) >= int(limit):
            break
    return out


def build_mcq_prompt(record: dict[str, Any]) -> str:
    lines = []
    if record.get("input_text"):
        lines.append(str(record["input_text"]).strip())
        lines.append("")
    lines.append(f"Question: {record['question']}")
    lines.append("Options:")
    for choice in record["candidate_choices"]:
        lines.append(f"{choice['choice']}. {choice['label_text']}")
    lines.append("")
    lines.append("Think briefly step by step, then answer with exactly one letter as:")
    lines.append("Answer: <letter>")
    return "\n".join(lines)


def grouped_split(records: list[dict[str, Any]], *, seed: int = 44021) -> dict[str, Any]:
    """Deterministic grouped split summary by question family."""

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[str(record["metadata"]["question_family"])].append(str(record["example_id"]))
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    train = set(names[: int(0.8 * n)])
    dev = set(names[int(0.8 * n) : int(0.9 * n)])
    test = set(names[int(0.9 * n) :])
    split_by_group = {}
    for group in names:
        split_by_group[group] = "train" if group in train else "dev" if group in dev else "test"
    leakage = False
    return {
        "num_groups": len(names),
        "num_examples": len(records),
        "split_counts": dict(Counter(split_by_group.values())),
        "group_leakage_detected": leakage,
        "split_by_group": split_by_group,
    }


def audit_dataset(records: list[dict[str, Any]], *, dataset: str) -> dict[str, Any]:
    labels = [str(r["gold_label"]) for r in records]
    counts = Counter(labels)
    split = grouped_split(records)
    return {
        "backend": BACKEND,
        "dataset": dataset,
        "num_examples": len(records),
        "label_distribution": dict(sorted(counts.items())),
        "majority_class_rate": (max(counts.values()) / len(labels)) if labels else None,
        "grouped_split": {k: v for k, v in split.items() if k != "split_by_group"},
        "license_and_source": {
            "source": "CyberMetric public GitHub raw files",
            "raw_path": "data/raw/cyber/cybermetric",
        },
    }


def option_position_bias_report(
    records: list[dict[str, Any]],
    *,
    greedy_labels: list[str | None] | None = None,
    sampled_labels: list[str | None] | None = None,
) -> dict[str, Any]:
    gold_counts = Counter(str(r["gold_label"]) for r in records)
    greedy_counts = Counter(x for x in (greedy_labels or []) if x)
    sampled_counts = Counter(x for x in (sampled_labels or []) if x)
    majority = max(gold_counts.values()) / len(records) if records else None
    severe = bool(majority is not None and majority >= 0.70)
    return {
        "gold_choice_distribution": dict(sorted(gold_counts.items())),
        "greedy_predicted_choice_distribution": dict(sorted(greedy_counts.items())),
        "sampled_predicted_choice_distribution": dict(sorted(sampled_counts.items())),
        "majority_position_baseline_accuracy": majority,
        "position_only_baseline_accuracy": majority,
        "option_order_fixed_seed_randomized": True,
        "same_semantic_label_always_same_option_letter": False,
        "severe_warning": severe,
        "warnings": ["gold_option_position_dominates"] if severe else [],
    }
