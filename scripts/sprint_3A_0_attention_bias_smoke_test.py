"""Run Sprint 3A-0 increase-only attention-bias steering smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.attention_bias_steering import (  # noqa: E402
    DEFAULT_DIAGNOSTIC_QUERY_N,
    DEFAULT_MODEL_PATH,
    DEFAULT_PRIMARY_N,
    run_3a0_smoke_test,
)

DEFAULT_FEATURE_MATRIX = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "sprint_2J_fix_slot_alignment_scoring_500"
    / "multi_span_feature_matrix.jsonl"
)
DEFAULT_ANSWER_POSITION_SCORE_MATRIX = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "sprint_2K_W_answer_position_output_effect_500"
    / "answer_position_score_matrix.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "sprint_3A_0_attention_bias_steering_smoke_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sprint 3A-0 attention-bias steering smoke test."
    )
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument(
        "--answer-position-score-matrix",
        type=Path,
        default=DEFAULT_ANSWER_POSITION_SCORE_MATRIX,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--primary-n", type=int, default=DEFAULT_PRIMARY_N)
    parser.add_argument("--diagnostic-query-n", type=int, default=DEFAULT_DIAGNOSTIC_QUERY_N)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_3a0_smoke_test(
        feature_matrix_path=args.feature_matrix,
        answer_position_score_matrix_path=args.answer_position_score_matrix,
        output_dir=args.output_dir,
        model_path=args.model_path,
        primary_n=args.primary_n,
        diagnostic_query_n=args.diagnostic_query_n,
        overwrite=args.overwrite,
        report_every=args.report_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
