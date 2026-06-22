from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sprint 0B smoke test.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    import recover_attention
    from recover_attention.data_io import read_jsonl, write_jsonl

    raw_data_path = resolve_project_path(config["paths"]["raw_data"])
    log_dir = resolve_project_path(config["paths"]["log_dir"])
    output_path = log_dir / "smoke_test_questions.jsonl"

    records = read_jsonl(raw_data_path)
    write_jsonl(records, output_path)
    reloaded_records = read_jsonl(output_path)

    if len(reloaded_records) != len(records):
        raise AssertionError(
            f"Record count mismatch: read {len(records)}, reloaded {len(reloaded_records)}"
        )

    print(f"config: {config_path}")
    print(f"package: recover_attention {recover_attention.__version__}")
    print(f"input: {raw_data_path}")
    print(f"output: {output_path}")
    print(f"records_read: {len(records)}")
    print(f"records_written: {len(reloaded_records)}")
    print("[OK] Sprint 0B smoke test passed.")


if __name__ == "__main__":
    main()
