"""Prepare validated CyberMetric canonical data for Sprint 4B-1."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.cyber_data import (
    CYBERMETRIC_LICENSE_NOTE,
    audit_cyber_samples,
    grouped_split,
    load_cybermetric_records,
    option_position_bias_pre_model_report,
    select_grouped_smoke_sample,
    to_canonical_cyber_sample,
)
from recover_attention.data_io import ensure_dir, write_json, write_jsonl, write_text
from recover_attention.domain_label_proxy import option_token_ids
from recover_attention.schemas import validate_cyber_sample_record

DEFAULT_INPUT = Path("data/raw/cyber/cybermetric/CyberMetric-2000-v1.json")
DEFAULT_PROCESSED = Path("data/processed/cyber/cybermetric.jsonl")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy"
)
DEFAULT_TOKENIZER_PATH = Path("D:/models/Qwen2.5-7B-Instruct")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--processed-output", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--smoke-questions", type=int, default=240)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _required_output_paths(args: argparse.Namespace) -> list[Path]:
    return [
        args.processed_output,
        args.output_dir / "preflight_report.md",
        args.output_dir / "dataset_audit_report.json",
        args.output_dir / "label_space_report.json",
        args.output_dir / "option_position_bias_pre_model_report.json",
        args.output_dir / "cyber_sample_manifest.jsonl",
        args.output_dir / "cyber_sample_smoke_manifest.jsonl",
        args.output_dir / "review_gate_cybermetric_schema_and_label_proxy.md",
    ]


def _check_output_safety(args: argparse.Namespace) -> None:
    existing = [path for path in _required_output_paths(args) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace only Sprint 4B-1 outputs: {paths}"
        )


def _invalid_reason(exc: ValueError) -> str:
    message = str(exc).lower()
    if "question" in message:
        return "missing_or_invalid_question"
    if "solution" in message or "gold" in message:
        return "missing_or_invalid_answer"
    if "option" in message or "answer" in message:
        return "invalid_options"
    return "other_validation_error"


def _build_records(
    raw_records: list[dict],
    *,
    source: str,
    shuffle_seed: int,
) -> tuple[list[dict], Counter[str]]:
    records: list[dict] = []
    invalid_reasons: Counter[str] = Counter()
    for original_index, raw_record in enumerate(raw_records):
        try:
            records.append(
                to_canonical_cyber_sample(
                    raw_record,
                    original_index=original_index,
                    source=source,
                    shuffle_seed=shuffle_seed,
                )
            )
        except ValueError as exc:
            invalid_reasons[_invalid_reason(exc)] += 1
    return records, invalid_reasons


def _label_space_report(records: list[dict], tokenizer_path: Path) -> dict[str, Any]:
    candidate_labels = sorted(
        {label for record in records for label in record["candidate_labels"]}
    )
    semantic_labels_preserved = all(
        choice["label_text"].strip()
        for record in records
        for choice in record["candidate_choices"]
    )
    gold_mapping_passed = all(
        next(
            choice for choice in record["candidate_choices"]
            if choice["choice"] == record["gold_label"]
        )["label_text"]
        == record["gold_label_text"]
        for record in records
    )
    report: dict[str, Any] = {
        "candidate_labels": candidate_labels,
        "all_labels_unique": len(candidate_labels) == len(set(candidate_labels)),
        "all_labels_single_character": all(len(label) == 1 for label in candidate_labels),
        "label_space": "mcq_option_letter",
        "semantic_labels_preserved": semantic_labels_preserved,
        "gold_mapping_validation_passed": gold_mapping_passed,
        "fake_qwen_tokenizer_test": "covered_by_tests/test_domain_label_proxy.py",
    }
    if not tokenizer_path.exists():
        report.update(
            {
                "tokenizer_check_status": "skipped",
                "skipped_reason": f"local tokenizer path does not exist: {tokenizer_path}",
            }
        )
        return report
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        token_ids = option_token_ids(tokenizer, candidate_labels)
    except Exception as exc:
        report.update(
            {
                "tokenizer_check_status": "skipped",
                "skipped_reason": (
                    "local tokenizer-only check unavailable; no download attempted: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        return report
    report.update(
        {
            "tokenizer_check_status": "passed",
            "tokenizer_name_or_path": str(tokenizer_path),
            "token_ids": token_ids,
            "single_token_passed": len(token_ids) == len(candidate_labels),
        }
    )
    return report


def _preflight_report(args: argparse.Namespace, raw_count: int) -> str:
    return f"""# Sprint 4B-1 Preflight Report

- Input: `{args.input_path}`
- Raw records available: `{raw_count}`
- Processed output: `{args.processed_output}`
- Audit output directory: `{args.output_dir}`
- Shuffle seed: `{args.shuffle_seed}`
- Split seed: `{args.split_seed}`
- Smoke questions: `{args.smoke_questions}`
- Local tokenizer-only path: `{args.tokenizer_path}`
- Causal LM loading: `no`
- Completion generation: `no`
- GPU inference: `no`
- F5 evaluation: `no`
- Site transfer / patching: `no`
- Probe training: `no`
- Steering: `no`
"""


def _review_gate(
    args: argparse.Namespace,
    audit: dict[str, Any],
    label_report: dict[str, Any],
    position_report: dict[str, Any],
    invalid_reasons: Counter[str],
) -> str:
    tokenizer_status = label_report["tokenizer_check_status"]
    skipped_reason = label_report.get("skipped_reason", "not applicable")
    split_counts = audit["split_counts"]
    split_groups = audit["split_group_counts"]
    can_enter = (
        audit["num_output_records"] >= args.smoke_questions
        and audit["group_leakage_count"] == 0
        and audit["duplicate_example_id_count"] == 0
        and label_report["semantic_labels_preserved"]
        and label_report["gold_mapping_validation_passed"]
        and not position_report["severe_position_imbalance"]
    )
    return f"""# Review Gate: CyberMetric Schema and Label Proxy

1. Raw file: `{args.input_path}`.
2. Raw record count: `{audit['num_raw_records']}`.
3. Successfully converted: `{audit['num_valid_records']}`.
4. Discarded: `{audit['num_invalid_records']}`; reasons: `{dict(invalid_reasons)}`.
5. Canonical schema added to schemas.py: yes.
6. All emitted canonical records passed schema validation: yes.
7. candidate_choices preserve complete semantic label text: {str(label_report['semantic_labels_preserved']).lower()}.
8. Option shuffle deterministic under fixed seed: {str(position_report['option_order_deterministic_under_fixed_seed']).lower()}.
9. Different seed can change option order: {str(position_report['option_order_changes_under_different_seed']).lower()}.
10. Gold semantic mapping remained correct after shuffle: {str(label_report['gold_mapping_validation_passed']).lower()}.
11. group_id: deterministic normalized-question hash proxy because CyberMetric has no family/category field.
12. Group leakage: `{audit['group_leakage_count']}`.
13. Split samples: `{split_counts}`; split groups: `{split_groups}`.
14. Gold A/B/C/D distribution: `{position_report['gold_choice_distribution']}`.
15. Severe option-position imbalance: {str(position_report['severe_position_imbalance']).lower()}; warnings: `{position_report['position_balance_warnings']}`.
16. Repeated semantic labels fixed to one option letter: `{position_report['semantic_labels_always_mapped_to_one_option_letter']}`.
17. FakeQwenTokenizer checks: covered by required pytest; this gate is valid only with passing tests.
18. Real Qwen tokenizer-only check status: `{tokenizer_status}`.
19. tokenizer skipped_reason: `{skipped_reason}`.
20. parse_option_answer tests: covered by required pytest; this gate is valid only with passing tests.
21. label-readout position tests: covered by required pytest; this gate is valid only with passing tests.
22. Gold leakage tests: covered by required pytest; this gate is valid only with passing tests.
23. Any causal LM called: no.
24. Any completion generated: no.
25. F5 calculated: no.
26. Site-transfer run: no.
27. Probe trained: no.
28. Steering executed: no.
29. Data/schema gates allow Sprint 4B-2 preflight: {str(can_enter).lower()}; full pytest must also pass.
30. Sprint 4B-2 input: `{args.processed_output}` (standard pipeline input), with `{args.output_dir / 'cyber_sample_manifest.jsonl'}` as the Sprint 4B-1 audit snapshot.

Processed dataset and output manifest intentionally contain identical records:
the processed dataset is the standard downstream input, while the output manifest
is the auditable Sprint 4B-1 snapshot.
"""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.input_path.exists():
        raise FileNotFoundError(f"CyberMetric input does not exist: {args.input_path}")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be >= 1")
    if args.smoke_questions < 1:
        raise ValueError("--smoke-questions must be >= 1")
    _check_output_safety(args)

    raw_records = load_cybermetric_records(args.input_path)
    selected_raw = (
        raw_records[: args.max_questions]
        if args.max_questions is not None
        else raw_records
    )
    source_name = args.input_path.stem
    canonical, invalid_reasons = _build_records(
        selected_raw,
        source=source_name,
        shuffle_seed=args.shuffle_seed,
    )
    if len(canonical) < args.smoke_questions:
        raise ValueError(
            f"only {len(canonical)} valid records; need at least {args.smoke_questions}"
        )
    canonical = grouped_split(
        canonical,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    for record in canonical:
        validate_cyber_sample_record(record)

    smoke = select_grouped_smoke_sample(
        canonical,
        sample_size=args.smoke_questions,
        seed=args.split_seed,
    )
    audit = {
        "input_path": str(args.input_path),
        "source_name": source_name,
        "num_raw_records": len(selected_raw),
        "num_valid_records": len(canonical),
        "num_invalid_records": len(selected_raw) - len(canonical),
        **audit_cyber_samples(canonical),
        "missing_question_count": invalid_reasons["missing_or_invalid_question"],
        "missing_answer_count": invalid_reasons["missing_or_invalid_answer"],
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "license_note": CYBERMETRIC_LICENSE_NOTE,
        "warnings": [
            "group_id is a deterministic question-derived grouping proxy"
        ],
    }
    label_report = _label_space_report(canonical, args.tokenizer_path)
    position_report = option_position_bias_pre_model_report(
        canonical,
        shuffle_seed=args.shuffle_seed,
    )

    ensure_dir(args.output_dir)
    write_text(
        _preflight_report(args, len(selected_raw)),
        args.output_dir / "preflight_report.md",
    )
    write_jsonl(canonical, args.processed_output)
    write_jsonl(canonical, args.output_dir / "cyber_sample_manifest.jsonl")
    write_jsonl(smoke, args.output_dir / "cyber_sample_smoke_manifest.jsonl")
    write_json(audit, args.output_dir / "dataset_audit_report.json")
    write_json(label_report, args.output_dir / "label_space_report.json")
    write_json(
        position_report,
        args.output_dir / "option_position_bias_pre_model_report.json",
    )
    write_text(
        _review_gate(args, audit, label_report, position_report, invalid_reasons),
        args.output_dir / "review_gate_cybermetric_schema_and_label_proxy.md",
    )

    print(f"Input: {args.input_path}")
    print(f"Raw records: {len(selected_raw)}")
    print(f"Canonical records: {len(canonical)}")
    print(f"Smoke records: {len(smoke)}")
    print(f"Processed output: {args.processed_output}")
    print(f"Audit output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
