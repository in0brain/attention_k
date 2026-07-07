"""Sprint 2K-W: answer-position output-effect re-measurement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.answer_position_output_effect import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    PRIMARY_TEMPLATE_ID,
    run_2kw,
)

DEFAULT_BASE = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "sprint_2J_fix_slot_alignment_scoring_500"
    / "multi_span_feature_matrix.jsonl"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "sprint_2K_W_answer_position_output_effect_500"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sprint 2K-W answer-position output-effect re-measurement."
    )
    parser.add_argument("--base-feature-matrix", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--template-id", default=PRIMARY_TEMPLATE_ID, choices=["A", "B"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_2kw(
        base_feature_matrix_path=args.base_feature_matrix,
        output_dir=args.output_dir,
        model_path=args.model_path,
        template_id=args.template_id,
        overwrite=args.overwrite,
        limit=args.limit,
        report_every=args.report_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
