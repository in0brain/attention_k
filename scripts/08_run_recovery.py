from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.recover_generation import (  # noqa: E402
    DEFAULT_NUM_SAMPLES,
    DEFAULT_RECOVERY_BACKEND,
    build_recover_output_file,
)
from recover_attention.schemas import validate_recover_output_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unit-level recovery outputs.")
    parser.add_argument("--input", required=True, help="Input masked questions JSONL path.")
    parser.add_argument("--output", required=True, help="Output recovery outputs JSONL path.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_RECOVERY_BACKEND,
        help="Recovery backend. Sprint 1G supports oracle_stub_v0 only.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of recovery samples per masked question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, stats = build_recover_output_file(
            args.input,
            args.output,
            backend=args.backend,
            num_samples=args.num_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_recover_output_record(record)

    print(f"num_input_masks: {stats['num_input_masks']}")
    print(f"num_output_recoveries: {stats['num_output_recoveries']}")
    print(f"num_samples: {stats['num_samples']}")
    print(f"recovery_backend: {stats['recovery_backend']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"mask_backend_counts: {stats['mask_backend_counts']}")
    print(f"mask_strategy_counts: {stats['mask_strategy_counts']}")
    print(f"[OK] Built recovery outputs: {args.output}")


if __name__ == "__main__":
    main()
