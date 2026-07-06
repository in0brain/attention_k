from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.full_scale_weak_labels import (  # noqa: E402
    BACKEND,
    build_full_scale_weak_labels,
)
from recover_attention.token_alignment import DEFAULT_MASK_TOKEN  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sprint 2G full-scale weak labels and 2A manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--language", default="en")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build_full_scale_weak_labels(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            backend=args.backend,
            seed=args.seed,
            mask_token=args.mask_token,
            language=args.language,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = result["report"]
    counts = report["counts"]
    print(f"manifest: {args.manifest}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_cases_in: {counts['num_cases_in']}")
    print(f"num_weak_labels: {counts['num_weak_labels']}")
    print(f"num_manifest_cases_with_mask: {counts['num_manifest_cases_with_mask']}")
    print(f"num_unmapped: {counts['num_unmapped']}")
    print(f"probe_target_counts: {report['probe_target_counts']}")
    print(f"usable_probe_target_counts: {report['usable_probe_target_counts']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    print(f"[OK] Built weak labels: {result['output_files']['weak_labels']}")
    print(f"[OK] Built 2A manifest: {result['output_files']['full_scale_2a_manifest']}")


if __name__ == "__main__":
    main()
