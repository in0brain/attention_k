from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.candidate_extraction import (
    DEFAULT_MAX_CANDIDATES,
    extract_candidate_span_file,
    summarize_candidate_span_records,
)
from recover_attention.schemas import validate_candidate_span_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate spans from question records.")
    parser.add_argument("--input", required=True, help="Input questions JSONL path.")
    parser.add_argument("--output", required=True, help="Output candidate spans JSONL path.")
    parser.add_argument(
        "--language",
        default="auto",
        choices=["auto", "en", "zh"],
        help="Language setting for rule-based extraction.",
    )
    parser.add_argument(
        "--backend",
        default="rule_based",
        choices=["rule_based"],
        help="Candidate extraction backend.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum candidates to keep per question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = extract_candidate_span_file(
        args.input,
        args.output,
        language=args.language,
        backend=args.backend,
        max_candidates=args.max_candidates,
    )

    for record in records:
        validate_candidate_span_record(record)

    stats = summarize_candidate_span_records(
        records,
        language=args.language,
        backend=args.backend,
        max_candidates=args.max_candidates,
    )

    print(f"num_questions: {stats['num_questions']}")
    print(f"total_candidates: {stats['total_candidates']}")
    print(f"avg_candidates_per_question: {stats['avg_candidates_per_question']:.2f}")
    print(f"min_candidates_per_question: {stats['min_candidates_per_question']}")
    print(f"max_candidates_per_question: {stats['max_candidates_per_question']}")
    print(f"questions_with_zero_candidates: {stats['questions_with_zero_candidates']}")
    print(f"span_type_counts: {stats['span_type_counts']}")
    print(f"max_candidates_setting: {stats['max_candidates_setting']}")
    print(f"language: {stats['language']}")
    print(f"backend: {stats['backend']}")
    print(f"[OK] Extracted candidate spans: {args.output}")


if __name__ == "__main__":
    main()
