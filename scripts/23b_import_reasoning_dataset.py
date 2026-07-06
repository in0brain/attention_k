from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.dataset_import import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_DATASET,
    DEFAULT_PREVIEW_LIMIT,
    import_reasoning_dataset,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and normalize a reasoning dataset (e.g. GSM8K)."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--input",
        default=None,
        help="Local raw dataset file. If omitted, the genuine public dataset is "
        "downloaded (gsm8k only).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output normalized question JSONL path.",
    )
    parser.add_argument(
        "--report-output",
        required=True,
        help="Output normalization report JSON path.",
    )
    parser.add_argument(
        "--preview-output",
        default=None,
        help="Optional preview JSONL path (first N normalized records).",
    )
    parser.add_argument("--preview-limit", type=positive_int, default=DEFAULT_PREVIEW_LIMIT)
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Smoke-test only: import at most N source records (not full scale).",
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = import_reasoning_dataset(
            output_path=args.output,
            report_output_path=args.report_output,
            dataset=args.dataset,
            split=args.split,
            input_path=args.input,
            backend=args.backend,
            preview_output_path=args.preview_output,
            preview_limit=args.preview_limit,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"dataset: {report['dataset']}")
    print(f"split: {report['split']}")
    print(f"source_mode: {report['source_mode']}")
    print(f"source_url_or_path: {report['source_url_or_path']}")
    print(f"num_raw_records: {report['num_raw_records']}")
    print(f"num_normalized_records: {report['num_normalized_records']}")
    print(f"can_run_500: {report['scale_check']['can_run_500']}")
    print(f"can_run_2000: {report['scale_check']['can_run_2000']}")
    print(f"output_path: {report['output_path']}")
    print(f"report_output: {args.report_output}")
    if report["preview_path"]:
        print(f"preview_path: {report['preview_path']}")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"warning: {warning}")
    print(f"[OK] Imported {report['num_normalized_records']} normalized records.")


if __name__ == "__main__":
    main()
