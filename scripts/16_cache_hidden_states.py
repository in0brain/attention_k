from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.hidden_state_cache import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAYER_INDICES,
    REAL_HF_BACKEND,
    cache_hidden_states_for_manifest,
)
from recover_attention.token_alignment import DEFAULT_MASK_TOKEN  # noqa: E402


DEFAULT_INPUT = (
    "outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2A_hidden_state_cache_baseline"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Sprint 2A hidden states.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--layer-indices", nargs="+", type=int, default=DEFAULT_LAYER_INDICES)
    parser.add_argument("--hidden-size", type=positive_int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-length", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.backend not in {DEFAULT_BACKEND, REAL_HF_BACKEND}:
        raise SystemExit(f"Unsupported backend: {args.backend}")

    try:
        result = cache_hidden_states_for_manifest(
            input_path=args.input,
            output_dir=args.output_dir,
            backend=args.backend,
            layer_indices=args.layer_indices,
            hidden_size=args.hidden_size,
            mask_token=args.mask_token,
            overwrite=args.overwrite,
            model_path=args.model_path,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            max_length=args.max_length,
            batch_size=args.batch_size,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    report = result["cache_report"]
    alignment_report = result["token_alignment_report"]

    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"backend: {args.backend}")
    print(f"num_cases: {report['num_cases']}")
    print(f"num_inputs_total: {report['num_inputs_total']}")
    print(f"num_hidden_state_files: {report['num_hidden_state_files']}")
    print(f"input_type_counts: {report['input_type_counts']}")
    print(f"alignment_status_counts: {report['alignment_status_counts']}")
    print(f"alignment_warning_count: {alignment_report['alignment_warning_count']}")
    if result.get("real_run_metadata") is not None:
        print(f"real_run_metadata: {result['output_files']['real_run_metadata']}")
    print(f"[OK] Built Sprint 2A hidden-state cache: {result['output_files']['hidden_state_manifest']}")


if __name__ == "__main__":
    main()
