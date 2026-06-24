from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import validate_semantic_label_record
from recover_attention.semantic_labels import label_nli_score_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic necessity labels.")
    parser.add_argument("--input", required=True, help="Input NLI scores JSONL path.")
    parser.add_argument("--output", required=True, help="Output semantic labels JSONL path.")
    parser.add_argument(
        "--backend",
        default="rule_v0",
        help="Semantic label backend. Sprint 1E supports rule_v0 only.",
    )
    parser.add_argument(
        "--equivalent-threshold",
        type=float,
        default=0.70,
        help="Threshold for Equivalent labels.",
    )
    parser.add_argument(
        "--directional-entailment-threshold",
        type=float,
        default=0.50,
        help="Threshold for directional entailment labels.",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.50,
        help="Threshold for contradiction-driven Non-equivalent labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rule_parameters = {
        "equivalent_threshold": args.equivalent_threshold,
        "directional_entailment_threshold": args.directional_entailment_threshold,
        "contradiction_threshold": args.contradiction_threshold,
    }
    input_records = read_jsonl(args.input)
    labeled_records, stats = label_nli_score_records(
        input_records,
        backend=args.backend,
        rule_parameters=rule_parameters,
    )

    for record in labeled_records:
        validate_semantic_label_record(record)

    write_jsonl(labeled_records, args.output)

    print(f"num_input_scores: {stats['num_input_scores']}")
    print(f"num_output_labels: {stats['num_output_labels']}")
    print(f"backend: {stats['backend']}")
    print(f"rule_parameters: {stats['rule_parameters']}")
    print(
        "semantic_necessity_label_counts: "
        f"{stats['semantic_necessity_label_counts']}"
    )
    print(
        "is_semantically_necessary_counts: "
        f"{stats['is_semantically_necessary_counts']}"
    )
    print(f"ablation_type_counts: {stats['ablation_type_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"language_counts: {stats['language_counts']}")
    print(f"[OK] Built semantic labels: {args.output}")


if __name__ == "__main__":
    main()
