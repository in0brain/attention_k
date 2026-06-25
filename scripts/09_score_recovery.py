from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.recover_scoring import (  # noqa: E402
    DEFAULT_SCORE_BACKEND,
    build_recover_score_file,
)
from recover_attention.schemas import validate_recover_score_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unit-level recoverability scores.")
    parser.add_argument("--input", required=True, help="Input recovery outputs JSONL path.")
    parser.add_argument("--output", required=True, help="Output recover scores JSONL path.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_SCORE_BACKEND,
        help="Score backend. Sprint 1H supports stub_rule_v0 only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, stats = build_recover_score_file(
            args.input,
            args.output,
            score_backend=args.backend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_recover_score_record(record)

    print(f"num_input_recoveries: {stats['num_input_recoveries']}")
    print(f"num_output_scores: {stats['num_output_scores']}")
    print(f"score_backend: {stats['score_backend']}")
    print(f"recovery_backend_counts: {stats['recovery_backend_counts']}")
    print(f"recoverability_label_counts: {stats['recoverability_label_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"num_misleading_recovery: {stats['num_misleading_recovery']}")
    print(f"[OK] Built recover scores: {args.output}")


if __name__ == "__main__":
    main()
