from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.schemas import validate_unit_evidence_record  # noqa: E402
from recover_attention.unit_evidence import (  # noqa: E402
    DEFAULT_UNIT_EVIDENCE_BACKEND,
    build_unit_evidence_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unit-level evidence records from semantic labels and recover scores."
    )
    parser.add_argument(
        "--semantic-labels",
        required=True,
        help="Input semantic labels JSONL path.",
    )
    parser.add_argument(
        "--recover-scores",
        required=True,
        help="Input recover scores JSONL path.",
    )
    parser.add_argument("--output", required=True, help="Output unit evidence JSONL path.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_UNIT_EVIDENCE_BACKEND,
        help="Unit evidence backend. Sprint 1I supports aggregate_stub_v0 only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, stats = build_unit_evidence_file(
            args.semantic_labels,
            args.recover_scores,
            args.output,
            evidence_backend=args.backend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_unit_evidence_record(record)

    print(f"num_semantic_labels: {stats['num_semantic_labels']}")
    print(f"num_recover_scores: {stats['num_recover_scores']}")
    print(f"num_output_unit_evidence: {stats['num_output_unit_evidence']}")
    print(f"evidence_backend: {stats['evidence_backend']}")
    print(f"available_signal_types: {stats['available_signal_types']}")
    print(f"missing_signal_types: {stats['missing_signal_types']}")
    print(f"evidence_status_counts: {stats['evidence_status_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"semantic_summary_label_counts: {stats['semantic_summary_label_counts']}")
    print(f"recoverability_label_counts: {stats['recoverability_label_counts']}")
    print(f"num_misleading_recovery: {stats['num_misleading_recovery']}")
    print(f"[OK] Built unit evidence: {args.output}")


if __name__ == "__main__":
    main()
