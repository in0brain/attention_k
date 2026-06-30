from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.guidance_candidates import (  # noqa: E402
    BACKEND,
    build_guidance_candidate_manifest,
)


DEFAULT_PREDICTIONS = "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl"
DEFAULT_EVAL_REPORT = "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2E_guidance_candidate_dry_run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sprint 2E planned-only guidance candidate manifest.",
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build_guidance_candidate_manifest(
            predictions_path=args.predictions,
            eval_report_path=args.eval_report,
            output_dir=args.output_dir,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    counts = report["counts"]
    print(f"predictions: {args.predictions}")
    print(f"eval_report: {args.eval_report}")
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"num_guidance_candidate_records: {counts['num_guidance_candidate_records']}")
    print(f"num_guidance_candidate_true: {counts['num_guidance_candidate_true']}")
    print(f"num_guidance_candidate_false: {counts['num_guidance_candidate_false']}")
    print(f"candidate_action_counts: {report['candidate_action_counts']}")
    print(
        "[OK] Built Sprint 2E guidance candidate dry run: "
        f"{result['output_files']['guidance_candidate_manifest']}"
    )


if __name__ == "__main__":
    main()
