from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.data_io import read_jsonl, write_jsonl  # noqa: E402
from recover_attention.nli_scoring import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EN_NLI_MODEL,
    DEFAULT_EN_NLI_MODEL_ID,
    DEFAULT_LABEL_ORDER,
    DEFAULT_MAX_LENGTH,
    DEFAULT_ZH_NLI_MODEL,
    DEFAULT_ZH_NLI_MODEL_ID,
)
from recover_attention.recover_scoring import (  # noqa: E402
    DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    DEFAULT_RECOVERY_LANGUAGE,
    DEFAULT_RECOVERY_NLI_BACKEND,
    DEFAULT_SCORE_BACKEND,
    build_recover_score_records,
    build_recovery_scoring_report,
    write_recovery_scoring_report,
)
from recover_attention.schemas import validate_recover_score_record  # noqa: E402


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unit-level recoverability scores.")
    parser.add_argument("--input", required=True, help="Input recovery outputs JSONL path.")
    parser.add_argument("--output", required=True, help="Output recover scores JSONL path.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_SCORE_BACKEND,
        help="Score backend: stub_rule_v0 or nli_recovery_judge_v0.",
    )
    parser.add_argument(
        "--nli-backend",
        default=DEFAULT_RECOVERY_NLI_BACKEND,
        help="NLI backend for nli_recovery_judge_v0.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_RECOVERY_LANGUAGE,
        choices=["auto", "en", "zh"],
        help="Language setting for NLI recovery judge.",
    )
    parser.add_argument("--en-model", default=DEFAULT_EN_NLI_MODEL)
    parser.add_argument("--zh-model", default=DEFAULT_ZH_NLI_MODEL)
    parser.add_argument("--en-model-id", default=DEFAULT_EN_NLI_MODEL_ID)
    parser.add_argument("--zh-model-id", default=DEFAULT_ZH_NLI_MODEL_ID)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow HuggingFace model downloads. Default is local files only.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["auto", "cpu", "cuda"],
        help="NLI device.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=positive_int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--label-order", default=DEFAULT_LABEL_ORDER)
    parser.add_argument(
        "--recoverable-entailment-threshold",
        type=float,
        default=DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    )
    parser.add_argument(
        "--partial-entailment-threshold",
        type=float,
        default=DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Only score the first N recover_output records.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional recovery scoring report JSON path; a sibling .md is also written.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}\nPlease run recovery generation first.")

    try:
        recover_output_records = read_jsonl(input_path)
        if args.limit is not None:
            recover_output_records = recover_output_records[: args.limit]
        records, stats = build_recover_score_records(
            recover_output_records,
            score_backend=args.backend,
            nli_backend=args.nli_backend,
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
            recoverable_entailment_threshold=args.recoverable_entailment_threshold,
            partial_entailment_threshold=args.partial_entailment_threshold,
            contradiction_threshold=args.contradiction_threshold,
        )
        write_jsonl(records, args.output)
        for record in records:
            validate_recover_score_record(record)

        if args.report_output:
            report = build_recovery_scoring_report(
                recover_output_records,
                records,
                input_path=args.input,
                output_path=args.output,
                score_backend=args.backend,
                nli_backend=args.nli_backend,
                language=args.language,
                en_model=args.en_model,
                zh_model=args.zh_model,
                allow_download=args.allow_download,
                device=args.device,
                max_length=args.max_length,
                label_order=args.label_order,
                recoverable_entailment_threshold=args.recoverable_entailment_threshold,
                partial_entailment_threshold=args.partial_entailment_threshold,
                contradiction_threshold=args.contradiction_threshold,
                limit=args.limit,
            )
            write_recovery_scoring_report(report, args.report_output)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"num_input_recoveries: {stats['num_input_recoveries']}")
    print(f"num_output_scores: {stats['num_output_scores']}")
    print(f"score_backend: {stats['score_backend']}")
    print(f"nli_backend: {stats['nli_backend']}")
    print(f"allow_download: {stats['allow_download']}")
    print(f"recovery_backend_counts: {stats['recovery_backend_counts']}")
    print(f"recoverability_label_counts: {stats['recoverability_label_counts']}")
    print(f"unit_scope_counts: {stats['unit_scope_counts']}")
    print(f"group_type_counts: {stats['group_type_counts']}")
    print(f"num_misleading_recovery: {stats['num_misleading_recovery']}")
    if args.report_output:
        print(f"[OK] Built recovery scoring report: {args.report_output}")
    print(f"[OK] Built recover scores: {args.output}")


if __name__ == "__main__":
    main()
