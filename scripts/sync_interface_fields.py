"""Generate or verify the ``required_fields`` blocks in the interface docs.

Single source of truth for top-level record fields:

    src/recover_attention/schemas.py :: REQUIRED_FIELDS

================================ BOUNDARY ================================
Read this before editing the script OR the interface docs.

This script OWNS exactly ONE region per interface doc: the fenced ```text
block that immediately follows a marker line of the form

    <!-- required_fields:<record_type> -->

That block is a GENERATED artifact. Do NOT hand-edit it. To change fields:

    1. edit REQUIRED_FIELDS in src/recover_attention/schemas.py
    2. run:  python scripts/sync_interface_fields.py --write
    3. run:  python -m pytest tests/test_interface_consistency.py -q

Everything else in the docs (prose, JSON examples, constraints, other code
blocks, the marker line itself, any comment lines before the fence) is
hand-written and is NEVER touched by this script.

Out of scope (intentionally NOT managed here):
    - docs/reasoning-aware-attention-guidance/label_schema.md
        It is an index/overview. It must only POINT to the interface docs and
        must not duplicate field lists. That rule is enforced by
        tests/test_interface_consistency.py, not by this generator.
    - record types absent from schemas.INTERFACE_DOCS
        (e.g. question / candidate_span / attention_anchor_label)
        have no interface doc and are skipped.
=========================================================================

Modes:
    --check  (default): verify every block matches REQUIRED_FIELDS in both
             content and order; print a report and exit 1 on any drift.
             Safe for CI / pre-commit; never writes.
    --write: rewrite each block in place so it matches REQUIRED_FIELDS.

The mapping of record type -> interface doc lives in
schemas.INTERFACE_DOCS, so this script and the consistency test share one
registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make src/ importable when run as a plain script (mirrors the other scripts).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.schemas import INTERFACE_DOCS, REQUIRED_FIELDS  # noqa: E402

SKILL_DIR = REPO_ROOT / "docs" / "reasoning-aware-attention-guidance"


def _marker(record_type: str) -> str:
    return f"<!-- required_fields:{record_type} -->"


def _locate_block(lines: list[str], record_type: str) -> tuple[int, int]:
    """Return (open_fence_index, close_fence_index) of the generated block.

    The generated block is the first fenced code block that appears after the
    ``<!-- required_fields:<record_type> -->`` marker line. Any comment or blank
    lines between the marker and the opening fence are left untouched.

    Raises ValueError if the marker or its fenced block cannot be found.
    """
    marker = _marker(record_type)
    marker_index = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if marker_index is None:
        raise ValueError(f"marker {marker!r} not found")

    open_index = next(
        (i for i in range(marker_index + 1, len(lines)) if lines[i].lstrip().startswith("```")),
        None,
    )
    if open_index is None:
        raise ValueError(f"no opening code fence after marker {marker!r}")

    close_index = next(
        (j for j in range(open_index + 1, len(lines)) if lines[j].lstrip().startswith("```")),
        None,
    )
    if close_index is None:
        raise ValueError(f"no closing code fence after marker {marker!r}")

    return open_index, close_index


def _current_fields(lines: list[str], open_index: int, close_index: int) -> list[str]:
    """Return the non-empty field lines currently inside the generated block."""
    return [lines[k].strip() for k in range(open_index + 1, close_index) if lines[k].strip()]


def _sync_doc(record_type: str, doc_name: str, write: bool) -> tuple[bool, str]:
    """Check or rewrite one interface doc's generated block.

    Returns (ok, message). In --check mode ``ok`` is False on any drift; in
    --write mode ``ok`` is False only on a structural error (missing marker or
    fence), since a content rewrite is the expected outcome.
    """
    path = SKILL_DIR / doc_name
    lines = path.read_text(encoding="utf-8").splitlines()
    open_index, close_index = _locate_block(lines, record_type)

    current = _current_fields(lines, open_index, close_index)
    desired = list(REQUIRED_FIELDS[record_type])

    if current == desired:
        return True, f"[OK]    {doc_name}: '{record_type}' block matches ({len(desired)} fields)"

    if not write:
        # Order-sensitive comparison; the generator owns the canonical order.
        only_doc = sorted(set(current) - set(desired))
        only_src = sorted(set(desired) - set(current))
        detail = "order/content mismatch"
        if only_doc or only_src:
            detail = f"only in doc={only_doc}, only in schemas={only_src}"
        return False, f"[DRIFT] {doc_name}: '{record_type}' block out of sync ({detail})"

    # Replace only the lines between the fences; keep the fences and everything else.
    new_lines = lines[: open_index + 1] + desired + lines[close_index:]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True, f"[WRITE] {doc_name}: rewrote '{record_type}' block ({len(desired)} fields)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate (--write) or verify (--check, default) the required_fields "
            "blocks in docs/reasoning-aware-attention-guidance/*_interface.md from schemas.REQUIRED_FIELDS."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify generated blocks without writing. This is the default.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the generated blocks in place. Default is check-only.",
    )
    args = parser.parse_args()

    all_ok = True
    for record_type, doc_name in sorted(INTERFACE_DOCS.items()):
        try:
            ok, message = _sync_doc(record_type, doc_name, args.write)
        except ValueError as exc:
            ok, message = False, f"[ERROR] {doc_name}: {exc}"
        all_ok = all_ok and ok
        print(message)

    if not all_ok:
        print(
            "\nrequired_fields blocks are out of sync. "
            "Fix REQUIRED_FIELDS in schemas.py, then run:\n"
            "  python scripts/sync_interface_fields.py --write"
        )
        sys.exit(1)

    print("\nDone." if args.write else "\nAll required_fields blocks are in sync.")


if __name__ == "__main__":
    main()
