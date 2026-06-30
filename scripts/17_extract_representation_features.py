from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.representation_features import (  # noqa: E402
    BACKEND,
    extract_representation_features,
)


DEFAULT_INPUT_MANIFEST = (
    "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl"
)
DEFAULT_INPUT_REPORT = (
    "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json"
)
DEFAULT_ALIGNMENT_REPORT = (
    "outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json"
)
DEFAULT_METADATA = "outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2B_representation_features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Sprint 2B representation features from cached hidden states.",
    )
    parser.add_argument("--input-manifest", default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--input-report", default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--alignment-report", default=DEFAULT_ALIGNMENT_REPORT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--eps", type=positive_float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = extract_representation_features(
            input_manifest_path=args.input_manifest,
            input_report_path=args.input_report,
            alignment_report_path=args.alignment_report,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            backend=args.backend,
            eps=args.eps,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    counts = report["counts"]
    print(f"input_manifest: {args.input_manifest}")
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"num_input_records: {counts['num_input_records']}")
    print(f"num_masked_groups: {counts['num_masked_groups']}")
    print(f"num_feature_records: {counts['num_feature_records']}")
    print(f"num_recovered_variants: {counts['num_recovered_variants']}")
    print(f"num_skipped_groups: {counts['num_skipped_groups']}")
    print(f"num_skipped_recovered_variants: {counts['num_skipped_recovered_variants']}")
    print(
        "[OK] Built Sprint 2B representation features: "
        f"{result['output_files']['representation_features']}"
    )


if __name__ == "__main__":
    main()
