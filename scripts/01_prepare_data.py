from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.prepare_data import prepare_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized question records.")
    parser.add_argument("--input", required=True, help="Input question JSONL path.")
    parser.add_argument("--output", required=True, help="Output normalized question JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_questions(args.input, args.output)
    print(f"[OK] Prepared {len(records)} question records: {args.output}")


if __name__ == "__main__":
    main()
