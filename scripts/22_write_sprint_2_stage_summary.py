from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.stage_summary import (  # noqa: E402
    BACKEND,
    DEFAULT_INPUT_PATHS,
    DEFAULT_OUTPUT_DIR,
    write_stage_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Sprint 2 final checkpoint summary and figures.",
    )
    parser.add_argument("--hidden-cache-report", default=DEFAULT_INPUT_PATHS["hidden_cache_report"])
    parser.add_argument("--alignment-report", default=DEFAULT_INPUT_PATHS["alignment_report"])
    parser.add_argument("--real-run-metadata", default=DEFAULT_INPUT_PATHS["real_run_metadata"])
    parser.add_argument(
        "--representation-feature-report",
        default=DEFAULT_INPUT_PATHS["representation_feature_report"],
    )
    parser.add_argument("--representation-features", default=DEFAULT_INPUT_PATHS["representation_features"])
    parser.add_argument("--probe-dataset-report", default=DEFAULT_INPUT_PATHS["probe_dataset_report"])
    parser.add_argument("--probe-dataset", default=DEFAULT_INPUT_PATHS["probe_dataset"])
    parser.add_argument("--probe-eval-report", default=DEFAULT_INPUT_PATHS["probe_eval_report"])
    parser.add_argument("--probe-predictions", default=DEFAULT_INPUT_PATHS["probe_predictions"])
    parser.add_argument("--probe-model", default=DEFAULT_INPUT_PATHS["probe_model"])
    parser.add_argument(
        "--guidance-candidate-report",
        default=DEFAULT_INPUT_PATHS["guidance_candidate_report"],
    )
    parser.add_argument(
        "--guidance-candidate-manifest",
        default=DEFAULT_INPUT_PATHS["guidance_candidate_manifest"],
    )
    parser.add_argument("--closed-loop-report", default=DEFAULT_INPUT_PATHS["closed_loop_report"])
    parser.add_argument("--closed-loop-audit", default=DEFAULT_INPUT_PATHS["closed_loop_audit"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--full-pytest-passed", type=int, default=None)
    parser.add_argument("--full-pytest-skipped", type=int, default=None)
    parser.add_argument("--full-pytest-duration-seconds", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = {
        "hidden_cache_report": args.hidden_cache_report,
        "alignment_report": args.alignment_report,
        "real_run_metadata": args.real_run_metadata,
        "representation_feature_report": args.representation_feature_report,
        "representation_features": args.representation_features,
        "probe_dataset_report": args.probe_dataset_report,
        "probe_dataset": args.probe_dataset,
        "probe_eval_report": args.probe_eval_report,
        "probe_predictions": args.probe_predictions,
        "probe_model": args.probe_model,
        "guidance_candidate_report": args.guidance_candidate_report,
        "guidance_candidate_manifest": args.guidance_candidate_manifest,
        "closed_loop_report": args.closed_loop_report,
        "closed_loop_audit": args.closed_loop_audit,
    }
    try:
        result = write_stage_summary(
            output_dir=args.output_dir,
            backend=args.backend,
            overwrite=args.overwrite,
            input_paths=input_paths,
            full_pytest_passed=args.full_pytest_passed,
            full_pytest_skipped=args.full_pytest_skipped,
            full_pytest_duration_seconds=args.full_pytest_duration_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = result["summary"]
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"status: {summary['status']}")
    print(f"summary: {result['output_files']['summary']}")
    print(f"audit: {result['output_files']['audit']}")
    print("figures:")
    for figure in result["output_files"]["figures"]:
        print(f"  {figure}")
    print(
        "full_pytest: "
        f"passed={summary['full_pytest']['passed']}, "
        f"skipped={summary['full_pytest']['skipped']}, "
        f"duration_seconds={summary['full_pytest']['duration_seconds']}"
    )
    print("[OK] Wrote Sprint 2 stage summary and figures")


if __name__ == "__main__":
    main()
