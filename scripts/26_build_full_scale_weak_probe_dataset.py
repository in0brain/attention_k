from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.weak_probe_dataset import (  # noqa: E402
    BACKEND,
    build_weak_probe_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sprint 2G full-scale weak probe dataset."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-report", required=True)
    parser.add_argument("--weak-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build_weak_probe_dataset(
            features_path=args.features,
            feature_report_path=args.feature_report,
            weak_labels_path=args.weak_labels,
            output_dir=args.output_dir,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    counts = report["counts"]
    decision = report["adaptive_kfold_decision"]
    print(f"features: {args.features}")
    print(f"weak_labels: {args.weak_labels}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_probe_records: {counts['num_probe_records']}")
    print(f"num_usable_records: {counts['num_usable_records']}")
    print(f"usable_probe_target_counts: {report['usable_probe_target_counts']}")
    print(f"adaptive_kfold: cv={decision['cv_strategy']} num_folds={decision['num_folds']} "
          f"min_class_count={decision['min_class_count']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    print(f"[OK] Built weak probe dataset: {result['output_files']['weak_probe_dataset']}")


if __name__ == "__main__":
    main()
