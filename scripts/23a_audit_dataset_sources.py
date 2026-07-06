from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.data_io import ensure_dir  # noqa: E402
from recover_attention.dataset_audit import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_SEARCH_ROOTS,
    audit_dataset_sources,
    render_audit_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate data sources for full-scale experiments."
    )
    parser.add_argument(
        "--search-roots",
        nargs="+",
        default=list(DEFAULT_SEARCH_ROOTS),
        help="Directories to scan for candidate data files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logs/sprint_2G_dataset_prep",
        help="Directory for the audit JSON and Markdown reports.",
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    json_path = output_dir / "dataset_source_audit.json"
    md_path = output_dir / "dataset_source_audit.md"

    for path in (json_path, md_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(
                f"output already exists: {path} (pass --overwrite to replace)"
            )

    report = audit_dataset_sources(args.search_roots, backend=args.backend)

    ensure_dir(output_dir)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")

    print(f"search_roots: {report['search_roots']}")
    print(f"num_files_scanned: {report['num_files_scanned']}")
    print(f"available_num_cases: {report['available_num_cases']}")
    print(f"max_usable_source_path: {report['max_usable_source_path']}")
    print(f"max_records_any_file: {report['max_records_any_file']} "
          f"({report['max_records_any_file_path']})")
    print(f"can_run_500: {report['can_run_500']}")
    print(f"can_run_2000: {report['can_run_2000']}")
    print(f"can_run_all: {report['can_run_all']}")
    if report["shortfall_reason"]:
        print(f"shortfall_reason: {report['shortfall_reason']}")
    print(f"[OK] Wrote dataset source audit: {json_path}")
    print(f"[OK] Wrote dataset source audit (md): {md_path}")


if __name__ == "__main__":
    main()
