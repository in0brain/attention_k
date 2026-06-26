from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.intervention_manifest import (  # noqa: E402
    DEFAULT_INTERVENTION_BACKEND,
    DEFAULT_INTERVENTION_TYPE,
    DEFAULT_MASK_TOKEN,
    build_intervention_manifest_file,
)
from recover_attention.schemas import validate_intervention_manifest_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build planned intervention manifest records from attention anchor labels."
    )
    parser.add_argument("--input", required=True, help="Input attention anchor labels JSONL path.")
    parser.add_argument("--output", required=True, help="Output intervention manifest JSONL path.")
    parser.add_argument(
        "--intervention-type",
        default=DEFAULT_INTERVENTION_TYPE,
        help="Planned intervention type. Sprint 1K defaults to mask.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_INTERVENTION_BACKEND,
        help="Intervention manifest backend. Sprint 1K supports manifest_stub_v0 only.",
    )
    parser.add_argument(
        "--mask-token",
        default=DEFAULT_MASK_TOKEN,
        help="Mask token recorded in planned_operation for mask interventions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, stats = build_intervention_manifest_file(
            args.input,
            args.output,
            intervention_type=args.intervention_type,
            intervention_backend=args.backend,
            mask_token=args.mask_token,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_intervention_manifest_record(record)

    print(f"num_input_attention_anchor_labels: {stats['num_input_attention_anchor_labels']}")
    print(f"num_output_intervention_manifest: {stats['num_output_intervention_manifest']}")
    print(f"intervention_backend: {stats['intervention_backend']}")
    print(f"intervention_type_counts: {stats['intervention_type_counts']}")
    print(f"intervention_status_counts: {stats['intervention_status_counts']}")
    print(f"target_scope_counts: {stats['target_scope_counts']}")
    print(f"attention_anchor_label_counts: {stats['attention_anchor_label_counts']}")
    print(f"label_status_counts: {stats['label_status_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"num_planned_only: {stats['num_planned_only']}")
    print(f"num_mask: {stats['num_mask']}")
    print(f"num_remove: {stats['num_remove']}")
    print(f"num_replace: {stats['num_replace']}")
    print(f"[OK] Built intervention manifest: {args.output}")


if __name__ == "__main__":
    main()
