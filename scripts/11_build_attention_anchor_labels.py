from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.attention_anchor_labels import (  # noqa: E402
    DEFAULT_ATTENTION_LABEL_BACKEND,
    build_attention_anchor_label_file,
)
from recover_attention.schemas import validate_attention_anchor_label_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build early attention anchor labels from unit evidence records."
    )
    parser.add_argument("--input", required=True, help="Input unit evidence JSONL path.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output attention anchor labels JSONL path.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_ATTENTION_LABEL_BACKEND,
        help="Attention label backend. Sprint 1J supports early_evidence_rule_stub_v0 only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, stats = build_attention_anchor_label_file(
            args.input,
            args.output,
            label_backend=args.backend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_attention_anchor_label_record(record)

    print(f"num_input_unit_evidence: {stats['num_input_unit_evidence']}")
    print(f"num_output_attention_anchor_labels: {stats['num_output_attention_anchor_labels']}")
    print(f"label_backend: {stats['label_backend']}")
    print(f"label_status_counts: {stats['label_status_counts']}")
    print(f"attention_anchor_label_counts: {stats['attention_anchor_label_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"available_signal_type_counts: {stats['available_signal_type_counts']}")
    print(f"missing_signal_type_counts: {stats['missing_signal_type_counts']}")
    print(f"score_min: {stats['score_min']}")
    print(f"score_max: {stats['score_max']}")
    print(f"score_mean: {stats['score_mean']}")
    print(f"num_risky_anchor: {stats['num_risky_anchor']}")
    print(f"[OK] Built attention anchor labels: {args.output}")


if __name__ == "__main__":
    main()
