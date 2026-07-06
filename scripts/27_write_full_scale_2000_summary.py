from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.full_scale_summary import (  # noqa: E402
    BACKEND,
    write_full_scale_summary,
)


DEFAULT_BASELINE_EVAL = (
    "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Sprint 2G 2000-case summary.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-eval-report", default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline = args.baseline_eval_report
    if baseline and not Path(baseline).exists():
        baseline = None
    try:
        result = write_full_scale_summary(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            baseline_eval_report_path=baseline,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    audit = result["audit"]
    print(f"root_dir: {args.root_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"actual_num_cases: {audit['actual_num_cases']}")
    print(f"phase_status: {audit['phase_status']}")
    print(f"figure_status: {result['figure_status']}")
    print(f"[OK] Wrote full-scale summary: {result['output_files']['summary']}")
    print(f"[OK] Wrote full-scale audit: {result['output_files']['audit']}")


if __name__ == "__main__":
    main()
