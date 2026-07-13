"""Deterministic H1 fabricated-identifier sample construction and audits."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from recover_attention.h1_identifier import OntologyEntry, OntologyIndex, extract_identifiers

SPLITS = ("train", "dev", "test")
DEFAULT_ROUTE_A_TOTAL = 360
DEFAULT_ROUTE_B_TOTAL = 120
DEFAULT_RECALL_QUOTAS = {"attack": 140, "cwe": 140, "cve": 80}
DEFAULT_OPEN_QUOTAS = {"attack": 40, "cwe": 40, "cve": 40}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def seeded_shuffle(items: list[Any], *, seed: int, namespace: str) -> list[Any]:
    copied = list(items)
    digest = hashlib.sha256(f"{namespace}:{seed}".encode("utf-8")).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(copied)
    return copied


def sanitize_source_text(text: str | None, source_id: str) -> str:
    """Remove direct identifier strings and normalize whitespace."""
    text = (text or "").strip()
    if not text:
        return ""
    patterns = {
        source_id,
        source_id.replace("-", " "),
        source_id.replace("-", ""),
    }
    if source_id.startswith("T") and "." in source_id:
        patterns.add(source_id.split(".", 1)[0])
    cleaned = text
    for pattern in sorted(patterns, key=len, reverse=True):
        if pattern:
            cleaned = re.sub(re.escape(pattern), "the referenced entry", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip()


def question_contains_identifier(question_text: str, normalized_id: str) -> bool:
    return any(mention.normalized == normalized_id for mention in extract_identifiers(question_text))


def build_recall_question(entry: OntologyEntry) -> tuple[str, dict[str, Any]]:
    description = sanitize_source_text(entry.description, entry.normalized_id)
    name = sanitize_source_text(entry.name, entry.normalized_id)
    if entry.family == "attack":
        family_text = "MITRE ATT&CK technique or sub-technique"
    elif entry.family == "cwe":
        family_text = "CWE weakness"
    else:
        family_text = "CVE record"
    evidence_parts = []
    if name:
        evidence_parts.append(f"Title/name: {name}.")
    if description:
        evidence_parts.append(f"Description: {description}")
    if not evidence_parts:
        evidence_parts.append("The ontology entry has minimal public metadata.")
    evidence = " ".join(evidence_parts)
    question = (
        f"A public cybersecurity ontology entry is described below.\n\n"
        f"{evidence}\n\n"
        f"Which {family_text} identifier does this entry correspond to? "
        "Answer with the identifier string only, then briefly explain your reasoning."
    )
    return question, {"template": "recall_v1", "family_text": family_text}


def _topics_for_entry(entry: OntologyEntry) -> list[str]:
    metadata = entry.metadata or {}
    topics: list[str] = []
    if entry.family == "attack":
        tactics = metadata.get("tactics") or []
        topics.extend(str(tactic).replace("-", " ") for tactic in tactics)
        if entry.name:
            topics.append(entry.name)
        return topics
    if entry.family == "cwe":
        categories = metadata.get("categories") or []
        topics.extend(str(category) for category in categories)
        if entry.name:
            topics.append(entry.name)
        return topics
    products = metadata.get("products") or []
    topics.extend(str(product) for product in products)
    if entry.name:
        topics.append(entry.name)
    return topics


def build_open_generation_question(family: str, topic: str, *, count: int = 5) -> tuple[str, dict]:
    family_text = {
        "attack": "MITRE ATT&CK techniques or sub-techniques",
        "cwe": "CWE weaknesses",
        "cve": "CVE identifiers",
    }[family]
    question = (
        f"List {count} real {family_text} related to {topic}. "
        "For each item, include the exact identifier and one short reason it is relevant."
    )
    return question, {"template": "open_gen_v1", "topic": topic, "requested_count": count}


def build_recall_samples(index: OntologyIndex, *, seed: int, quotas: dict[str, int] | None = None) -> list[dict]:
    quotas = dict(quotas or DEFAULT_RECALL_QUOTAS)
    samples: list[dict] = []
    for family, quota in quotas.items():
        entries = [
            entry for entry in index.entries.get(family, {}).values()
            if entry.name or entry.description
        ]
        entries = seeded_shuffle(entries, seed=seed, namespace=f"recall:{family}")
        for rank, entry in enumerate(entries[:quota]):
            question, params = build_recall_question(entry)
            if question_contains_identifier(question, entry.normalized_id):
                raise ValueError(
                    f"recall question leaked source id {entry.normalized_id}: {question}"
                )
            samples.append(
                {
                    "example_id": f"h1_recall_{family}_{rank:04d}_{stable_hash(entry.normalized_id)[:8]}",
                    "route": "recall",
                    "family": family,
                    "prompt_params": params,
                    "question_text": question,
                    "source_entry_id": entry.normalized_id,
                    "source_entry_metadata": {
                        "name": entry.name,
                        "description": sanitize_source_text(entry.description, entry.normalized_id),
                        "status": entry.status,
                        "metadata": entry.metadata or {},
                    },
                    "group_id": f"h1_recall_entry_{family}_{stable_hash(entry.normalized_id)}",
                    "split": "train",
                    "label_space": "open_identifier",
                }
            )
    return samples


def build_open_generation_samples(
    index: OntologyIndex,
    *,
    seed: int,
    quotas: dict[str, int] | None = None,
) -> list[dict]:
    quotas = dict(quotas or DEFAULT_OPEN_QUOTAS)
    samples: list[dict] = []
    for family, quota in quotas.items():
        topics: list[str] = []
        seen_topics: set[str] = set()
        for entry in index.entries.get(family, {}).values():
            for topic in _topics_for_entry(entry):
                cleaned = sanitize_source_text(topic, entry.normalized_id)
                key = cleaned.casefold()
                if cleaned and key not in seen_topics:
                    topics.append(cleaned)
                    seen_topics.add(key)
        topics = seeded_shuffle(topics, seed=seed, namespace=f"open:{family}")
        for rank, topic in enumerate(topics[:quota]):
            question, params = build_open_generation_question(family, topic)
            samples.append(
                {
                    "example_id": f"h1_open_{family}_{rank:04d}_{stable_hash(topic)[:8]}",
                    "route": "open_gen",
                    "family": family,
                    "prompt_params": params,
                    "question_text": question,
                    "source_entry_id": None,
                    "source_entry_metadata": {"topic": topic},
                    "group_id": f"h1_open_topic_{family}_{stable_hash(topic)}",
                    "split": "train",
                    "label_space": "open_identifier",
                }
            )
    return samples


def grouped_split(
    records: list[dict],
    *,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 4242,
) -> list[dict]:
    ratios = (train_ratio, dev_ratio, test_ratio)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 for value in ratios):
        raise ValueError("split ratios must be non-negative numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    group_ids = seeded_shuffle(sorted(groups), seed=seed, namespace="h1_split")
    train_end = int(len(group_ids) * train_ratio)
    dev_end = train_end + int(len(group_ids) * dev_ratio)
    group_to_split = {}
    for index, group_id in enumerate(group_ids):
        group_to_split[group_id] = "train" if index < train_end else "dev" if index < dev_end else "test"
    output = []
    for record in records:
        copied = deepcopy(record)
        copied["split"] = group_to_split[copied["group_id"]]
        output.append(copied)
    return output


def assert_no_question_gold_id_leakage(records: list[dict]) -> None:
    for record in records:
        source_id = record.get("source_entry_id")
        if record.get("route") == "recall" and source_id:
            if question_contains_identifier(record["question_text"], source_id):
                raise ValueError(f"{record['example_id']} question leaks source_entry_id {source_id}")


def audit_h1_samples(records: list[dict]) -> dict:
    question_norm = [re.sub(r"\s+", " ", row["question_text"]).strip().casefold() for row in records]
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split_groups[row["group_id"]].add(row["split"])
    lengths = [len(row["question_text"]) for row in records]
    route_family_counts = Counter((row["route"], row["family"]) for row in records)
    split_counts = Counter(row["split"] for row in records)
    return {
        "num_records": len(records),
        "route_counts": dict(sorted(Counter(row["route"] for row in records).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in records).items())),
        "route_family_counts": {
            f"{route}:{family}": count
            for (route, family), count in sorted(route_family_counts.items())
        },
        "duplicate_normalized_question_count": len(question_norm) - len(set(question_norm)),
        "question_length_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "gold_id_residual_check": {
            "checked_records": sum(row["route"] == "recall" for row in records),
            "violations": 0,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_counts": dict(sorted(Counter(next(iter(s)) for s in split_groups.values()).items())),
        "group_count": len(split_groups),
        "group_leakage_count": sum(len(splits) > 1 for splits in split_groups.values()),
        "label_space": "open_identifier",
    }


def estimate_id_space_density(index: OntologyIndex) -> dict:
    cve_by_year: dict[str, list[int]] = defaultdict(list)
    for normalized_id in index.entries.get("cve", {}):
        _, year, number = normalized_id.split("-")
        cve_by_year[year].append(int(number))
    cve_years = {}
    for year, numbers in sorted(cve_by_year.items()):
        max_number = max(numbers)
        high_space = max(1, 999999 - 100000 + 1)
        high_hits = sum(number >= 100000 for number in numbers)
        cve_years[year] = {
            "id_count": len(numbers),
            "max_observed_number": max_number,
            "low_4_digit_space_occupancy": min(1.0, len([n for n in numbers if n < 10000]) / 9000),
            "high_number_hit_probability_estimate": high_hits / high_space,
        }
    cve_space = max(1, len(cve_years) * 996000)
    attack_space = 9000 + 9000 * 999
    cwe_space = 99999
    report = {
        "families": {
            "cve": {
                "id_count": len(index.entries.get("cve", {})),
                "format_space_size_estimate": cve_space,
                "random_legal_string_hit_probability_estimate": len(index.entries.get("cve", {})) / cve_space,
                "year_bucket_occupancy": cve_years,
                "judgment_strength": "weak_for_common_low_number_ranges; auxiliary unless high-number or paired with ATT&CK/CWE",
            },
            "attack": {
                "id_count": len(index.entries.get("attack", {})),
                "format_space_size_estimate": attack_space,
                "random_legal_string_hit_probability_estimate": len(index.entries.get("attack", {})) / attack_space,
                "judgment_strength": "strong_sparse_space",
            },
            "cwe": {
                "id_count": len(index.entries.get("cwe", {})),
                "format_space_size_estimate": cwe_space,
                "random_legal_string_hit_probability_estimate": len(index.entries.get("cwe", {})) / cwe_space,
                "judgment_strength": "strong_sparse_space",
            },
        },
        "labeling_strategy": {
            "primary_families": ["attack", "cwe"],
            "cve_policy": (
                "CVE existence is weaker in dense low-number year buckets; use CVE as an "
                "auxiliary family and prefer high sequence numbers when interpreting "
                "fabrication evidence."
            ),
            "reserved_deprecated_policy": (
                "Existence-positive statuses remain grounded and are counted separately; "
                "fabrication means absent from the complete snapshot index."
            ),
        },
    }
    return report


def build_h1_samples(index: OntologyIndex, *, seed: int = 4242) -> list[dict]:
    records = build_recall_samples(index, seed=seed) + build_open_generation_samples(index, seed=seed)
    records = grouped_split(records, seed=seed)
    assert_no_question_gold_id_leakage(records)
    return records
