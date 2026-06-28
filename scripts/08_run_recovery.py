from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.recover_generation import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RECOVERY_BACKEND,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_P,
    build_recover_output_file,
    build_recover_output_records,
)
from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.schemas import validate_recover_output_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unit-level recovery outputs.")
    parser.add_argument("--input", required=True, help="Input masked questions JSONL path.")
    parser.add_argument("--output", required=True, help="Output recovery outputs JSONL path.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_RECOVERY_BACKEND,
        help="Recovery backend: oracle_stub_v0 or ollama_chat_v0.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of recovery samples per masked question.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name for ollama_chat_v0.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Ollama decoding temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Ollama top_p decoding parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to predict.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Ollama decoding seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit input records at the script layer for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.limit is None:
            records, stats = build_recover_output_file(
                args.input,
                args.output,
                backend=args.backend,
                num_samples=args.num_samples,
                model=args.model,
                ollama_base_url=args.ollama_base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                seed=args.seed,
            )
        else:
            if args.limit < 0:
                raise ValueError("limit must be >= 0")
            input_path = Path(args.input)
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing input: {input_path}\nPlease run Sprint 1F first."
                )
            input_records = read_jsonl(input_path)[: args.limit]
            records, stats = build_recover_output_records(
                input_records,
                backend=args.backend,
                num_samples=args.num_samples,
                model=args.model,
                ollama_base_url=args.ollama_base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                seed=args.seed,
            )
            write_jsonl(records, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    for record in records:
        validate_recover_output_record(record)

    print(f"num_input_masks: {stats['num_input_masks']}")
    print(f"num_output_recoveries: {stats['num_output_recoveries']}")
    print(f"num_samples: {stats['num_samples']}")
    print(f"recovery_backend: {stats['recovery_backend']}")
    print(f"model: {stats['model']}")
    print(f"ollama_base_url: {stats['ollama_base_url']}")
    print(f"temperature: {stats['temperature']}")
    print(f"top_p: {stats['top_p']}")
    print(f"max_tokens: {stats['max_tokens']}")
    print(f"timeout: {stats['timeout']}")
    print(f"seed: {stats['seed']}")
    print(f"num_empty_recoveries: {stats['num_empty_recoveries']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"mask_backend_counts: {stats['mask_backend_counts']}")
    print(f"mask_strategy_counts: {stats['mask_strategy_counts']}")
    print(f"[OK] Built recovery outputs: {args.output}")


if __name__ == "__main__":
    main()
