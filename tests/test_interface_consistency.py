"""Cross-document consistency checks.

``REQUIRED_FIELDS`` / ``FORBIDDEN_FIELDS`` in
``src/recover_attention/schemas.py`` are the single source of truth for
top-level record fields. This test asserts that:

1. Each ``*_interface.md`` declares its fields via a machine-readable
   ``<!-- required_fields:<type> -->`` marker, and that the marked block matches
   the validator constant exactly.
2. Interface docs do not list any forbidden (stale / future-stage) field.
3. ``label_schema.md`` only points to the interface docs and no longer
   duplicates a full field list (so it cannot drift).

Drift fails CI here instead of misleading a later sprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.schemas import FORBIDDEN_FIELDS, INTERFACE_DOCS, REQUIRED_FIELDS


SKILL_DIR = REPO_ROOT / "docs" / "skill"
LABEL_SCHEMA_PATH = SKILL_DIR / "label_schema.md"

# record_type -> (label_schema heading, interface doc it must reference)
LABEL_SCHEMA_SECTIONS = {
    "ablated_question": ("# 7. Ablated Question Record", "ablated_questions_interface.md"),
    "nli_score": ("# 8. NLI Score Record", "nli_scores_interface.md"),
    "semantic_label": ("# 9. Semantic Label Record", "semantic_labels_interface.md"),
    "masked_question": ("# 10. Masked Question Record", "masked_questions_interface.md"),
    "recover_output": ("# 11. Recover Output Record", "recover_outputs_interface.md"),
}


def _identifiers(block: list[str]) -> set[str]:
    return set(_identifier_list(block))


def _identifier_list(block: list[str]) -> list[str]:
    """Field identifier lines, preserving document order (for order-sensitive checks)."""
    return [line.strip() for line in block if line.strip() and " " not in line.strip()]


def _first_text_block_after(lines: list[str], start_index: int) -> list[str]:
    in_fence = False
    block: list[str] = []
    for line in lines[start_index:]:
        if line.lstrip().startswith("```"):
            if in_fence:
                return block
            in_fence = True
            continue
        if in_fence:
            block.append(line)
    raise AssertionError("no closed fenced block found after the given index")


def _all_text_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    fence_lang = ""
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if current is None:
                current = []
                fence_lang = stripped[3:].strip()
            else:
                if fence_lang == "text":
                    blocks.append(current)
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _marked_fields(doc_name: str, record_type: str) -> list[str]:
    """Return the marked block's fields in document order."""
    lines = (SKILL_DIR / doc_name).read_text(encoding="utf-8").splitlines()
    marker = f"<!-- required_fields:{record_type} -->"
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return _identifier_list(_first_text_block_after(lines, index + 1))
    raise AssertionError(f"machine-readable marker {marker!r} not found in {doc_name}")


def _label_schema_section(heading: str) -> list[str]:
    lines = LABEL_SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
    assert start is not None, f"heading {heading!r} not found in label_schema.md"
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("# ")),
        len(lines),
    )
    return lines[start:end]


@pytest.mark.parametrize("record_type", sorted(INTERFACE_DOCS))
def test_interface_marker_matches_required_fields(record_type: str) -> None:
    # Order-sensitive: the generator defines the canonical field order, so the
    # marker block must match REQUIRED_FIELDS in both content AND order.
    required = REQUIRED_FIELDS[record_type]
    assert required, f"REQUIRED_FIELDS[{record_type!r}] is empty"
    documented = _marked_fields(INTERFACE_DOCS[record_type], record_type)
    assert documented == required, (
        f"{INTERFACE_DOCS[record_type]} required_fields marker drifted from "
        f"REQUIRED_FIELDS[{record_type!r}] (order-sensitive).\n"
        f"  doc:     {documented}\n"
        f"  schemas: {required}\n"
        f"  fix with: python scripts/sync_interface_fields.py --write"
    )


@pytest.mark.parametrize("record_type", sorted(INTERFACE_DOCS))
def test_interface_excludes_forbidden_fields(record_type: str) -> None:
    forbidden = set(FORBIDDEN_FIELDS.get(record_type, ()))
    if not forbidden:
        pytest.skip(f"{record_type} has no forbidden fields")
    documented = set(_marked_fields(INTERFACE_DOCS[record_type], record_type))
    leaked = documented & forbidden
    assert not leaked, f"{INTERFACE_DOCS[record_type]} lists forbidden field(s): {sorted(leaked)}"


@pytest.mark.parametrize("record_type", sorted(LABEL_SCHEMA_SECTIONS))
def test_label_schema_points_to_interface(record_type: str) -> None:
    heading, interface_doc = LABEL_SCHEMA_SECTIONS[record_type]
    section_text = "\n".join(_label_schema_section(heading))
    assert interface_doc in section_text, (
        f"label_schema.md section '{heading}' must reference {interface_doc}"
    )


@pytest.mark.parametrize("record_type", sorted(LABEL_SCHEMA_SECTIONS))
def test_label_schema_does_not_duplicate_fields(record_type: str) -> None:
    heading, _ = LABEL_SCHEMA_SECTIONS[record_type]
    required = set(REQUIRED_FIELDS[record_type])
    for block in _all_text_blocks(_label_schema_section(heading)):
        identifiers = _identifiers(block)
        assert not required.issubset(identifiers), (
            f"label_schema.md section '{heading}' still duplicates the full field "
            f"list; keep field definitions only in the interface doc"
        )
