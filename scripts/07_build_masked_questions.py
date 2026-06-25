from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.masked_questions import (  # noqa: E402
    DEFAULT_MASK_BACKEND,
    DEFAULT_MASK_TOKEN,
    build_masked_question_file,
)
from recover_attention.schemas import validate_masked_question_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unit-level masked questions.")
    parser.add_argument("--input", required=True, help="Input semantic labels JSONL path.")
    parser.add_argument("--output", required=True, help="Output masked questions JSONL path.")
    parser.add_argument(
        "--mask-token",
        default=DEFAULT_MASK_TOKEN,
        help="Mask token used for each span in a unit.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_MASK_BACKEND,
        help="Mask construction backend. Sprint 1F supports unit_mask_v0 only.",
    )
    parser.add_argument(
        "--only-necessary",
        action="store_true",
        help="Only build masks for units with at least one necessary semantic source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, stats = build_masked_question_file(
        args.input,
        args.output,
        mask_token=args.mask_token,
        backend=args.backend,
        only_necessary=args.only_necessary,
    )

    for record in records:
        validate_masked_question_record(record)

    print(f"num_input_labels: {stats['num_input_labels']}")
    print(f"num_units: {stats['num_units']}")
    print(f"num_output_masks: {stats['num_output_masks']}")
    print(f"num_filtered_not_necessary: {stats['num_filtered_not_necessary']}")
    print(f"num_skipped_overlap: {stats['num_skipped_overlap']}")
    print(f"mask_token: {stats['mask_token']}")
    print(f"mask_backend: {stats['mask_backend']}")
    print(f"mask_strategy: {stats['mask_strategy']}")
    print(f"only_necessary: {stats['only_necessary']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"source_count_distribution: {stats['source_count_distribution']}")
    print(f"[OK] Built masked questions: {args.output}")


if __name__ == "__main__":
    main()
