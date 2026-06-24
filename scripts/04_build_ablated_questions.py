from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.question_ablations import build_ablated_question_file
from recover_attention.schemas import validate_ablated_question_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablated questions from ablation units.")
    parser.add_argument("--input", required=True, help="Input ablation units JSONL path.")
    parser.add_argument("--output", required=True, help="Output ablated questions JSONL path.")
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "en", "zh"],
        help="Language setting for generalization replacements.",
    )
    parser.add_argument(
        "--ablation-types",
        default="delete,generalize",
        help="Comma-separated ablation types. Sprint 1C supports delete,generalize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_types = [
        ablation_type.strip()
        for ablation_type in args.ablation_types.split(",")
        if ablation_type.strip()
    ]
    records, stats = build_ablated_question_file(
        args.input,
        args.output,
        ablation_types=ablation_types,
        language=args.language,
    )

    for record in records:
        validate_ablated_question_record(record)

    print(f"num_input_questions: {stats['num_input_questions']}")
    print(f"num_input_units: {stats['num_input_units']}")
    print(f"num_output_ablations: {stats['num_output_ablations']}")
    print(f"num_skipped_empty: {stats['num_skipped_empty']}")
    print(f"num_skipped_unchanged: {stats['num_skipped_unchanged']}")
    print(f"num_skipped_overlap: {stats['num_skipped_overlap']}")
    print(f"ablation_type_counts: {stats['ablation_type_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"language: {stats['language']}")
    print(f"ablation_types: {stats['ablation_types']}")
    print(f"[OK] Built ablated questions: {args.output}")


if __name__ == "__main__":
    main()
