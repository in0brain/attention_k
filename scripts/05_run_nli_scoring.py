from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.nli_scoring import score_ablated_question_records
from recover_attention.schemas import validate_nli_score_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic NLI scoring stub.")
    parser.add_argument("--input", required=True, help="Input ablated questions JSONL path.")
    parser.add_argument("--output", required=True, help="Output NLI scores JSONL path.")
    parser.add_argument(
        "--backend",
        default="stub_v0",
        help="NLI scoring backend. Sprint 1D supports stub_v0 only.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language setting: auto, en, or zh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = read_jsonl(args.input)
    scored_records, stats = score_ablated_question_records(
        input_records,
        backend=args.backend,
        language=args.language,
    )

    for record in scored_records:
        validate_nli_score_record(record)

    write_jsonl(scored_records, args.output)

    print(f"num_input_ablations: {stats['num_input_ablations']}")
    print(f"num_output_scores: {stats['num_output_scores']}")
    print(f"backend: {stats['backend']}")
    print(f"language_setting: {stats['language_setting']}")
    print(f"language_counts: {stats['language_counts']}")
    print(f"ablation_type_counts: {stats['ablation_type_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"[OK] Built NLI scores: {args.output}")


if __name__ == "__main__":
    main()
