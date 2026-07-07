"""Sprint 2J-B: run multi-span hidden/attention scoring and formula validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.multi_span_reasoning_scoring import (  # noqa: E402
    DEFAULT_LAYER_INDICES,
    DEFAULT_MODEL_PATH,
    run_2jb_scoring,
)


DEFAULT_MATRIX_DIR = PROJECT_ROOT / "outputs" / "logs" / "sprint_2J_multi_span_matrix_500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sprint 2J-B hidden/attention scoring for the 500-case multi-span matrix."
    )
    parser.add_argument(
        "--flat-matrix",
        type=Path,
        default=DEFAULT_MATRIX_DIR / "multi_span_flat_matrix.jsonl",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_MATRIX_DIR / "candidate_span_coverage_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "sprint_2J_multi_span_scoring_500",
    )
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYER_INDICES),
    )
    parser.add_argument("--mask-token", default="[MASK]")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_2jb_scoring(
        flat_matrix_path=args.flat_matrix,
        coverage_report_path=args.coverage_report,
        output_dir=args.output_dir,
        model_path=args.model_path,
        layer_indices=args.layer_indices,
        mask_token=args.mask_token,
        overwrite=args.overwrite,
        max_records=args.max_records,
        report_every=args.report_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
