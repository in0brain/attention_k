"""Sprint 2J-A: build multi-span-per-question reasoning matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.multi_span_reasoning_matrix import run_2ja_matrix  # noqa: E402


DEFAULT_2H = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_instance_signal_500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sprint 2J-A multi-span candidate matrix and coverage gate."
    )
    parser.add_argument(
        "--subset-cases",
        type=Path,
        default=DEFAULT_2H / "subset_cases.jsonl",
    )
    parser.add_argument(
        "--raw-gsm8k",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "gsm8k_train_normalized.jsonl",
    )
    parser.add_argument(
        "--risk-strength",
        type=Path,
        default=DEFAULT_2H / "risk_strength_dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "sprint_2J_multi_span_matrix_500",
    )
    parser.add_argument("--max-spans-per-question", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_2ja_matrix(
        subset_cases_path=args.subset_cases,
        raw_gsm8k_path=args.raw_gsm8k,
        risk_strength_path=args.risk_strength,
        output_dir=args.output_dir,
        max_spans_per_question=args.max_spans_per_question,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
