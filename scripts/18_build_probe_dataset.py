from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.probe_dataset import BACKEND, build_probe_dataset  # noqa: E402


DEFAULT_FEATURES = "outputs/logs/sprint_2B_representation_features/representation_features.jsonl"
DEFAULT_FEATURE_REPORT = (
    "outputs/logs/sprint_2B_representation_features/representation_feature_report.json"
)
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2C_probe_dataset"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sprint 2C probe dataset from Sprint 2B representation features.",
    )
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--feature-report", default=DEFAULT_FEATURE_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--human-labels", default=None)
    parser.add_argument("--sprint-1r-manifest", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build_probe_dataset(
            features_path=args.features,
            feature_report_path=args.feature_report,
            output_dir=args.output_dir,
            backend=args.backend,
            overwrite=args.overwrite,
            human_labels_path=args.human_labels,
            sprint_1r_manifest_path=args.sprint_1r_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    counts = report["counts"]
    print(f"features: {args.features}")
    print(f"feature_report: {args.feature_report}")
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"num_probe_records: {counts['num_probe_records']}")
    print(f"num_probe_target_usable: {counts['num_probe_target_usable']}")
    print(f"num_probe_target_unusable: {counts['num_probe_target_unusable']}")
    print(f"target_counts: {report['target_counts']}")
    print(f"[OK] Built Sprint 2C probe dataset: {result['output_files']['probe_dataset']}")


if __name__ == "__main__":
    main()
