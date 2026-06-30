from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.human_review_consolidation import (  # noqa: E402
    DEFAULT_KNOWN_ISSUES_MD,
    DEFAULT_LABELS_JSONL,
    DEFAULT_MANIFEST_JSONL,
    DEFAULT_REPORT_JSON,
    DEFAULT_REVIEW_GUIDE,
    DEFAULT_SUMMARY_JSON,
    run_human_review_consolidation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate Sprint 1Q human review labels for Sprint 1R."
    )
    parser.add_argument("--review-guide", default=str(DEFAULT_REVIEW_GUIDE))
    parser.add_argument("--labels-jsonl", default=str(DEFAULT_LABELS_JSONL))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--known-issues-md", default=str(DEFAULT_KNOWN_ISSUES_MD))
    parser.add_argument("--manifest-jsonl", default=str(DEFAULT_MANIFEST_JSONL))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_human_review_consolidation(
        review_guide=args.review_guide,
        labels_jsonl=args.labels_jsonl,
        report_json=args.report_json,
        summary_json=args.summary_json,
        known_issues_md=args.known_issues_md,
        manifest_jsonl=args.manifest_jsonl,
    )

    print(f"review_guide: {args.review_guide}")
    print(f"labels_jsonl: {args.labels_jsonl}")
    print(f"report_json: {args.report_json}")
    print(f"reviewed_count: {result['reviewed_count']}")
    print(f"unreviewed_count: {result['unreviewed_count']}")
    print(f"manifest_count: {result['manifest_count']}")
    print(f"validation_warning_count: {result['validation_warning_count']}")
    for warning in result["validation_warnings"]:
        print(f"WARNING: {warning}")
    print(f"[OK] Built Sprint 1R human review artifacts: {args.summary_json}")


if __name__ == "__main__":
    main()
