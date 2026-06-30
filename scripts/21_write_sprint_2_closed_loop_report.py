from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.closed_loop_report import (  # noqa: E402
    BACKEND,
    DEFAULT_INPUT_PATHS,
    DEFAULT_OUTPUT_DIR,
    build_sprint_2_closed_loop_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Sprint 2F mini closed-loop report.",
    )
    parser.add_argument("--hidden-cache-report", default=DEFAULT_INPUT_PATHS["hidden_cache_report"])
    parser.add_argument("--alignment-report", default=DEFAULT_INPUT_PATHS["alignment_report"])
    parser.add_argument("--real-run-metadata", default=DEFAULT_INPUT_PATHS["real_run_metadata"])
    parser.add_argument("--representation-features", default=DEFAULT_INPUT_PATHS["representation_features"])
    parser.add_argument(
        "--representation-feature-report",
        default=DEFAULT_INPUT_PATHS["representation_feature_report"],
    )
    parser.add_argument("--probe-dataset", default=DEFAULT_INPUT_PATHS["probe_dataset"])
    parser.add_argument("--probe-dataset-report", default=DEFAULT_INPUT_PATHS["probe_dataset_report"])
    parser.add_argument("--probe-predictions", default=DEFAULT_INPUT_PATHS["probe_predictions"])
    parser.add_argument("--probe-eval-report", default=DEFAULT_INPUT_PATHS["probe_eval_report"])
    parser.add_argument(
        "--guidance-candidate-manifest",
        default=DEFAULT_INPUT_PATHS["guidance_candidate_manifest"],
    )
    parser.add_argument(
        "--guidance-candidate-report",
        default=DEFAULT_INPUT_PATHS["guidance_candidate_report"],
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = {
        "hidden_cache_report": args.hidden_cache_report,
        "alignment_report": args.alignment_report,
        "real_run_metadata": args.real_run_metadata,
        "representation_features": args.representation_features,
        "representation_feature_report": args.representation_feature_report,
        "probe_dataset": args.probe_dataset,
        "probe_dataset_report": args.probe_dataset_report,
        "probe_predictions": args.probe_predictions,
        "probe_eval_report": args.probe_eval_report,
        "guidance_candidate_manifest": args.guidance_candidate_manifest,
        "guidance_candidate_report": args.guidance_candidate_report,
    }
    try:
        result = build_sprint_2_closed_loop_report(
            output_dir=args.output_dir,
            backend=args.backend,
            overwrite=args.overwrite,
            input_paths=input_paths,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = result["summary"]
    stages = summary["stages"]
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"status: {summary['status']}")
    print(f"hidden_state_cache: {summary['loop_status']['hidden_state_cache']}")
    print(f"representation_features: {summary['loop_status']['representation_features']}")
    print(f"probe_dataset: {summary['loop_status']['probe_dataset']}")
    print(f"probe_training: {summary['loop_status']['probe_training']}")
    print(f"guidance_candidate_dry_run: {summary['loop_status']['guidance_candidate_dry_run']}")
    print(f"num_feature_records: {stages['2B']['num_feature_records']}")
    print(f"num_predictions: {stages['2D']['num_predictions']}")
    print(f"num_guidance_candidate_records: {stages['2E']['num_guidance_candidate_records']}")
    print(f"report: {result['output_files']['report']}")
    print(f"audit: {result['output_files']['audit']}")
    print("[OK] Wrote Sprint 2 minimal closed-loop report")


if __name__ == "__main__":
    main()
