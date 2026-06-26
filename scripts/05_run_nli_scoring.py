from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.nli_scoring import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EN_NLI_MODEL,
    DEFAULT_EN_NLI_MODEL_ID,
    DEFAULT_LABEL_ORDER,
    DEFAULT_MAX_LENGTH,
    DEFAULT_ZH_NLI_MODEL,
    DEFAULT_ZH_NLI_MODEL_ID,
    score_ablated_question_records,
)
from recover_attention.schemas import validate_nli_score_record


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLI semantic consistency scoring.")
    parser.add_argument("--input", required=True, help="Input ablated questions JSONL path.")
    parser.add_argument("--output", required=True, help="Output NLI scores JSONL path.")
    parser.add_argument(
        "--backend",
        default="stub_v0",
        help="NLI scoring backend: stub_v0, hf_nli_en_v0, hf_nli_zh_v0, or hf_nli_auto_v0.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language setting: auto, en, or zh.",
    )
    parser.add_argument(
        "--en-model",
        default=DEFAULT_EN_NLI_MODEL,
        help="Local English NLI model path.",
    )
    parser.add_argument(
        "--zh-model",
        default=DEFAULT_ZH_NLI_MODEL,
        help="Local Chinese/multilingual NLI model path.",
    )
    parser.add_argument(
        "--en-model-id",
        default=DEFAULT_EN_NLI_MODEL_ID,
        help="English HuggingFace model id used only when --allow-download is set.",
    )
    parser.add_argument(
        "--zh-model-id",
        default=DEFAULT_ZH_NLI_MODEL_ID,
        help="Chinese/multilingual HuggingFace model id used only when --allow-download is set.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow HuggingFace Hub downloads or cache pulls for hf_nli_* backends.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Model device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Reserved batch size parameter for real NLI backends.",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max_length for real NLI backends.",
    )
    parser.add_argument(
        "--label-order",
        default=DEFAULT_LABEL_ORDER,
        help="NLI label order: auto or comma-separated contradiction/neutral/entailment labels.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Limit input records at the script layer for small smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = read_jsonl(args.input)
    if args.limit is not None:
        input_records = input_records[: args.limit]

    scored_records, stats = score_ablated_question_records(
        input_records,
        backend=args.backend,
        language=args.language,
        en_model=args.en_model,
        zh_model=args.zh_model,
        en_model_id=args.en_model_id,
        zh_model_id=args.zh_model_id,
        allow_download=args.allow_download,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label_order=args.label_order,
    )

    for record in scored_records:
        validate_nli_score_record(record)

    write_jsonl(scored_records, args.output)

    print(f"num_input_ablations: {stats['num_input_ablations']}")
    print(f"num_output_scores: {stats['num_output_scores']}")
    if args.limit is not None:
        print(f"limit: {args.limit}")
    print(f"backend: {stats['backend']}")
    print(f"language_setting: {stats['language_setting']}")
    print(f"language_counts: {stats['language_counts']}")
    print(f"ablation_type_counts: {stats['ablation_type_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    if "model_sources" in stats:
        print(f"model_sources: {stats['model_sources']}")
        print(f"allow_download: {stats['allow_download']}")
        print(f"device: {stats['device']}")
        print(f"batch_size: {stats['batch_size']}")
        print(f"max_length: {stats['max_length']}")
        print(f"label_order: {stats['label_order']}")
    print(f"[OK] Built NLI scores: {args.output}")


if __name__ == "__main__":
    main()
