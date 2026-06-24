from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.ablation_units import (
    DEFAULT_MAX_GROUP_SIZE,
    DEFAULT_MAX_GROUP_UNITS,
    build_ablation_unit_file,
    summarize_ablation_unit_records,
)
from recover_attention.schemas import validate_ablation_unit_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation units from candidate span records.")
    parser.add_argument("--input", required=True, help="Input candidate spans JSONL path.")
    parser.add_argument("--output", required=True, help="Output ablation units JSONL path.")
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=DEFAULT_MAX_GROUP_SIZE,
        help="Maximum spans kept in one group unit.",
    )
    parser.add_argument(
        "--max-group-units",
        type=int,
        default=DEFAULT_MAX_GROUP_UNITS,
        help="Maximum group units generated per question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_ablation_unit_file(
        args.input,
        args.output,
        max_group_size=args.max_group_size,
        max_group_units=args.max_group_units,
    )

    for record in records:
        validate_ablation_unit_record(record)

    stats = summarize_ablation_unit_records(
        records,
        max_group_size=args.max_group_size,
        max_group_units=args.max_group_units,
    )

    print(f"num_questions: {stats['num_questions']}")
    print(f"num_candidate_spans: {stats['num_candidate_spans']}")
    print(f"num_units: {stats['num_units']}")
    print(f"num_single_units: {stats['num_single_units']}")
    print(f"num_group_units: {stats['num_group_units']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"questions_with_zero_units: {stats['questions_with_zero_units']}")
    print(f"max_group_size: {stats['max_group_size']}")
    print(f"max_group_units: {stats['max_group_units']}")
    print(f"[OK] Built ablation units: {args.output}")


if __name__ == "__main__":
    main()
