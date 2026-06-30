from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.probe_training import (  # noqa: E402
    BACKEND,
    MODEL_TYPE,
    train_probe_baseline,
)


DEFAULT_DATASET = "outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl"
DEFAULT_DATASET_REPORT = "outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2D_probe_training_baseline"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Sprint 2D minimal probe baseline from Sprint 2C dataset.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--model", default=MODEL_TYPE)
    parser.add_argument("--cv", choices=["leave_one_out", "stratified_k_fold"], default="leave_one_out")
    parser.add_argument("--num-folds", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=positive_float, default=1.0)
    parser.add_argument("--top-k-features", type=positive_int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = train_probe_baseline(
            dataset_path=args.dataset,
            dataset_report_path=args.dataset_report,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            cv=args.cv,
            num_folds=args.num_folds,
            seed=args.seed,
            alpha=args.alpha,
            top_k_features=args.top_k_features,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    print(f"dataset: {args.dataset}")
    print(f"dataset_report: {args.dataset_report}")
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"model: {args.model}")
    print(f"status: {report['status']}")
    print(f"num_records_usable: {report['data_summary']['num_records_usable']}")
    if report["status"] == "ok":
        print(f"cv_strategy: {report['training']['cv_strategy']}")
        print(f"num_folds: {report['training']['num_folds']}")
        print(f"accuracy: {report['metrics']['accuracy']}")
        print(f"macro_f1: {report['metrics']['macro_f1']}")
        print(
            "[OK] Trained Sprint 2D probe baseline: "
            f"{result['output_files']['probe_eval_report']}"
        )
    else:
        print(f"[OK] Wrote insufficient-data Sprint 2D report: {result['output_files']['probe_eval_report']}")


if __name__ == "__main__":
    main()
