from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.full_scale_manifest import (  # noqa: E402
    BACKEND,
    DEFAULT_ID_PREFIX,
    SAMPLING_RULES,
    build_full_scale_manifest,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sprint 2G full-scale manifest.")
    parser.add_argument("--source", default="data/raw/gsm8k_train_normalized.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requested-num-cases", type=positive_int, default=2000)
    parser.add_argument("--sampling-rule", choices=list(SAMPLING_RULES), default="seeded_sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build_full_scale_manifest(
            source_path=args.source,
            output_dir=args.output_dir,
            requested_num_cases=args.requested_num_cases,
            sampling_rule=args.sampling_rule,
            seed=args.seed,
            id_prefix=args.id_prefix,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    print(f"source: {args.source}")
    print(f"output_dir: {args.output_dir}")
    print(f"sampling_rule: {report['sampling_rule']} (seed={report['seed']})")
    print(f"requested_num_cases: {report['requested_num_cases']}")
    print(f"available_num_cases: {report['available_num_cases']}")
    print(f"actual_num_cases: {report['actual_num_cases']}")
    print(f"can_run_2000: {report['can_run_2000']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    print(f"[OK] Built full-scale manifest: {result['output_files']['full_scale_manifest']}")


if __name__ == "__main__":
    main()
