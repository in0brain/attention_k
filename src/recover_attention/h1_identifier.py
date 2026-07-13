"""H1 fabricated-identifier extraction and ontology existence checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from recover_attention.data_io import read_jsonl

FAMILIES = {"cve", "attack", "cwe"}
MENTION_LABELS = {"grounded", "fabricated", "echoed"}

_CVE_RE = re.compile(r"\bCVE[\s-](\d{4})[\s-]?(\d{4,})\b", re.IGNORECASE)
_ATTACK_RE = re.compile(r"(?<![A-Za-z0-9-])T(\d{4})(?:\.(\d{3}))?(?![A-Za-z0-9])")
_CWE_RE = re.compile(r"\bCWE[\s-](\d{1,5})\b", re.IGNORECASE)


@dataclass(frozen=True)
class IdentifierMention:
    family: str
    raw: str
    normalized: str
    start: int
    end: int
    parent: str | None = None
    sub_id: str | None = None
    granularity: str | None = None


@dataclass(frozen=True)
class OntologyEntry:
    family: str
    normalized_id: str
    status: str
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OntologyIndex:
    entries: dict[str, dict[str, OntologyEntry]]
    snapshot_fingerprints: dict[str, str]
    status_rules: dict[str, str]

    def has(self, family: str, normalized_id: str) -> bool:
        return normalized_id in self.entries.get(family, {})

    def get(self, family: str, normalized_id: str) -> OntologyEntry | None:
        return self.entries.get(family, {}).get(normalized_id)

    def status(self, family: str, normalized_id: str) -> str | None:
        entry = self.get(family, normalized_id)
        return entry.status if entry else None

    def to_jsonable(self) -> dict:
        return {
            "entries": {
                family: {
                    normalized_id: asdict(entry)
                    for normalized_id, entry in sorted(family_entries.items())
                }
                for family, family_entries in sorted(self.entries.items())
            },
            "snapshot_fingerprints": dict(sorted(self.snapshot_fingerprints.items())),
            "status_rules": dict(sorted(self.status_rules.items())),
        }


def normalize_identifier(family: str, raw: str) -> str:
    family = family.casefold()
    if family not in FAMILIES:
        raise ValueError(f"unknown identifier family: {family!r}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("raw identifier must be a non-empty str")
    text = raw.strip()
    if family == "cve":
        match = _CVE_RE.fullmatch(text)
        if not match:
            raise ValueError(f"invalid CVE identifier: {raw!r}")
        return f"CVE-{match.group(1)}-{match.group(2)}".upper()
    if family == "attack":
        match = _ATTACK_RE.fullmatch(text.upper())
        if not match:
            raise ValueError(f"invalid ATT&CK technique identifier: {raw!r}")
        suffix = f".{match.group(2)}" if match.group(2) else ""
        return f"T{match.group(1)}{suffix}"
    match = _CWE_RE.fullmatch(text)
    if not match:
        raise ValueError(f"invalid CWE identifier: {raw!r}")
    return f"CWE-{int(match.group(1))}"


def extract_identifiers(text: str) -> list[IdentifierMention]:
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    mentions: list[IdentifierMention] = []
    for match in _CVE_RE.finditer(text):
        normalized = normalize_identifier("cve", match.group(0))
        mentions.append(
            IdentifierMention(
                family="cve",
                raw=match.group(0),
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                granularity="cve",
            )
        )
    for match in _ATTACK_RE.finditer(text):
        normalized = normalize_identifier("attack", match.group(0))
        parent = normalized.split(".", 1)[0]
        mentions.append(
            IdentifierMention(
                family="attack",
                raw=match.group(0),
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                parent=parent,
                sub_id=normalized.split(".", 1)[1] if "." in normalized else None,
                granularity="subtechnique" if "." in normalized else "technique",
            )
        )
    for match in _CWE_RE.finditer(text):
        normalized = normalize_identifier("cwe", match.group(0))
        mentions.append(
            IdentifierMention(
                family="cwe",
                raw=match.group(0),
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                granularity="weakness",
            )
        )
    mentions.sort(key=lambda mention: (mention.start, mention.end, mention.family))
    return mentions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_from_record(record: dict) -> OntologyEntry:
    return OntologyEntry(
        family=str(record["family"]),
        normalized_id=str(record["normalized_id"]),
        status=str(record.get("status") or "unknown"),
        name=record.get("name"),
        description=record.get("description"),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    )


def build_ontology_index(snapshot_dir: str | Path) -> OntologyIndex:
    root = Path(snapshot_dir)
    entries: dict[str, dict[str, OntologyEntry]] = {family: {} for family in sorted(FAMILIES)}
    fingerprints: dict[str, str] = {}
    for family in sorted(FAMILIES):
        path = root / family / "ontology_index.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing ontology index for {family}: {path}")
        fingerprints[family] = _sha256(path)
        for record in read_jsonl(path):
            entry = _entry_from_record(record)
            if entry.family != family:
                raise ValueError(f"{path} contains {entry.family!r} entry in {family!r} index")
            entries[family][entry.normalized_id] = entry
    return OntologyIndex(
        entries=entries,
        snapshot_fingerprints=fingerprints,
        status_rules={
            "cve": "PUBLISHED, RESERVED, and REJECTED ids are existence-positive; status is counted separately.",
            "attack": "revoked/deprecated ATT&CK ids are existence-positive; flags are counted separately.",
            "cwe": "deprecated CWE ids are existence-positive; status is counted separately.",
        },
    )


def _normalized_prompt_ids(prompt_text: str) -> set[tuple[str, str]]:
    return {(mention.family, mention.normalized) for mention in extract_identifiers(prompt_text)}


def classify_mention(
    mention: IdentifierMention,
    index: OntologyIndex,
    prompt_text: str,
) -> str:
    if (mention.family, mention.normalized) in _normalized_prompt_ids(prompt_text):
        return "echoed"
    return "grounded" if index.has(mention.family, mention.normalized) else "fabricated"


def label_completion(completion_text: str, prompt_text: str, index: OntologyIndex) -> dict:
    mentions = []
    h1_positive = False
    for mention in extract_identifiers(completion_text):
        label = classify_mention(mention, index, prompt_text)
        if label == "fabricated":
            h1_positive = True
        entry = index.get(mention.family, mention.normalized)
        row = {
            **asdict(mention),
            "label": label,
            "ontology_status": entry.status if entry else None,
        }
        mentions.append(row)
    return {
        "mentions": mentions,
        "num_mentions": len(mentions),
        "num_grounded": sum(row["label"] == "grounded" for row in mentions),
        "num_fabricated": sum(row["label"] == "fabricated" for row in mentions),
        "num_echoed": sum(row["label"] == "echoed" for row in mentions),
        "h1_positive": h1_positive,
    }


def assert_no_h1_gold_label_leakage(feature_record: dict) -> None:
    forbidden = {
        "source_entry_id",
        "gold_id",
        "gold_identifier",
        "target_identifier",
        "answer_key",
        "correct_identifier",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).casefold()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized_key in forbidden:
                    raise ValueError(f"forbidden H1 inference feature field at {child_path}")
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(feature_record, "")


def save_ontology_index(index: OntologyIndex, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(index.to_jsonable(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def family_counts(index: OntologyIndex) -> dict[str, int]:
    return {family: len(entries) for family, entries in sorted(index.entries.items())}


def iter_entries(index: OntologyIndex, family: str) -> Iterable[OntologyEntry]:
    yield from index.entries.get(family, {}).values()
