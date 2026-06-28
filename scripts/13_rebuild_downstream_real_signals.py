from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from recover_attention.attention_anchor_labels import (  # noqa: E402
    DEFAULT_ATTENTION_LABEL_BACKEND,
    build_attention_anchor_label_records,
)
from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from recover_attention.intervention_manifest import (  # noqa: E402
    DEFAULT_INTERVENTION_BACKEND,
    DEFAULT_INTERVENTION_TYPE,
    build_intervention_manifest_records,
)
from recover_attention.masked_questions import (  # noqa: E402
    DEFAULT_MASK_BACKEND,
    DEFAULT_MASK_TOKEN,
    build_masked_question_records,
)
from recover_attention.nli_scoring import (  # noqa: E402
    DEFAULT_EN_NLI_MODEL,
    DEFAULT_ZH_NLI_MODEL,
    score_ablated_question_records,
)
from recover_attention.recover_generation import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_P,
    OLLAMA_CHAT_BACKEND,
    build_recover_output_records,
)
from recover_attention.recover_scoring import (  # noqa: E402
    DEFAULT_SCORE_BACKEND,
    normalize_question,
    build_recover_score_records,
)
from recover_attention.schemas import (  # noqa: E402
    validate_attention_anchor_label_record,
    validate_intervention_manifest_record,
    validate_masked_question_record,
    validate_nli_score_record,
    validate_recover_output_record,
    validate_recover_score_record,
    validate_semantic_label_record,
    validate_unit_evidence_record,
)
from recover_attention.semantic_labels import (  # noqa: E402
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_DIRECTIONAL_ENTAILMENT_THRESHOLD,
    DEFAULT_EQUIVALENT_THRESHOLD,
    label_nli_score_records,
)
from recover_attention.unit_evidence import (  # noqa: E402
    DEFAULT_UNIT_EVIDENCE_BACKEND,
    build_unit_evidence_records,
)


DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_1N_real_downstream"
DEFAULT_NLI_BACKEND = "hf_nli_auto_v0"
DEFAULT_LANGUAGE = "auto"
DEFAULT_SEMANTIC_BACKEND = "rule_v0"
DEFAULT_RECOVER_SCORE_BACKEND = DEFAULT_SCORE_BACKEND
DEFAULT_NEXT_STEP = "Sprint 1O：Upgrade Real Recovery Scoring"

OUTPUT_FILENAMES = {
    "nli_scores": "nli_scores_real.jsonl",
    "semantic_labels": "semantic_labels_real.jsonl",
    "masked_questions": "masked_questions_real.jsonl",
    "recover_outputs": "recover_outputs_real.jsonl",
    "recover_scores": "recover_scores_real.jsonl",
    "unit_evidence": "unit_evidence_real.jsonl",
    "attention_anchor_labels": "attention_anchor_labels_real.jsonl",
    "intervention_manifest": "intervention_manifest_real.jsonl",
    "report_json": "real_signal_report.json",
    "report_md": "real_signal_report.md",
}

KNOWN_LIMITATIONS = [
    "recover_scores_real still uses stub_rule_v0 exact normalized match.",
    "real LLM recovery output may be semantically correct even when exact match fails.",
    "this report cannot prove attention guidance effectiveness.",
    "trajectory stability, answer stability, and raw attention pattern are not included.",
    "intervention_manifest_real is planned_only and does not mean intervention was executed.",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild isolated downstream outputs with real NLI and real LLM recovery."
    )
    parser.add_argument(
        "--ablated-questions",
        default="data/processed/ablated_questions.jsonl",
        help="Input ablated questions JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Isolated output directory for Sprint 1N artifacts.",
    )
    parser.add_argument("--nli-backend", default=DEFAULT_NLI_BACKEND)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--en-model", default=DEFAULT_EN_NLI_MODEL)
    parser.add_argument("--zh-model", default=DEFAULT_ZH_NLI_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--semantic-backend", default=DEFAULT_SEMANTIC_BACKEND)
    parser.add_argument(
        "--equivalent-threshold",
        type=float,
        default=DEFAULT_EQUIVALENT_THRESHOLD,
    )
    parser.add_argument(
        "--directional-entailment-threshold",
        type=float,
        default=DEFAULT_DIRECTIONAL_ENTAILMENT_THRESHOLD,
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=DEFAULT_CONTRADICTION_THRESHOLD,
    )
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--mask-backend", default=DEFAULT_MASK_BACKEND)
    parser.add_argument("--recovery-backend", default=OLLAMA_CHAT_BACKEND)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--num-samples", type=positive_int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=positive_int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--recover-score-backend", default=DEFAULT_RECOVER_SCORE_BACKEND)
    parser.add_argument("--unit-evidence-backend", default=DEFAULT_UNIT_EVIDENCE_BACKEND)
    parser.add_argument("--attention-label-backend", default=DEFAULT_ATTENTION_LABEL_BACKEND)
    parser.add_argument("--intervention-type", default=DEFAULT_INTERVENTION_TYPE)
    parser.add_argument("--intervention-backend", default=DEFAULT_INTERVENTION_BACKEND)
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Limit ablated question input records at the script layer.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Build through masked questions and mark recovery_skipped=true in report.",
    )
    return parser.parse_args(argv)


def build_real_downstream_paths(output_dir: str | Path) -> dict[str, Path]:
    output_root = Path(output_dir)
    return {
        key: output_root / filename
        for key, filename in OUTPUT_FILENAMES.items()
    }


def normalize_text_for_report(text: str) -> str:
    return normalize_question(text)


def count_records(path: str | Path) -> int:
    return len(read_jsonl(path))


def run_rebuild(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    ensure_isolated_output_dir(output_dir)
    ensure_dir(output_dir)
    paths = build_real_downstream_paths(output_dir)

    ablated_records = read_jsonl(args.ablated_questions)
    if args.limit is not None:
        ablated_records = ablated_records[: args.limit]

    nli_records, _nli_stats = score_ablated_question_records(
        ablated_records,
        backend=args.nli_backend,
        language=args.language,
        en_model=args.en_model,
        zh_model=args.zh_model,
        allow_download=args.allow_download,
    )
    _validate_and_write(nli_records, paths["nli_scores"], validate_nli_score_record)

    semantic_records, _semantic_stats = label_nli_score_records(
        nli_records,
        backend=args.semantic_backend,
        rule_parameters={
            "equivalent_threshold": args.equivalent_threshold,
            "directional_entailment_threshold": args.directional_entailment_threshold,
            "contradiction_threshold": args.contradiction_threshold,
        },
    )
    _validate_and_write(
        semantic_records,
        paths["semantic_labels"],
        validate_semantic_label_record,
    )

    masked_records, _masked_stats = build_masked_question_records(
        semantic_records,
        mask_token=args.mask_token,
        backend=args.mask_backend,
    )
    _validate_and_write(
        masked_records,
        paths["masked_questions"],
        validate_masked_question_record,
    )

    recover_records: list[dict] = []
    recover_score_records: list[dict] = []
    unit_evidence_records: list[dict] = []
    attention_anchor_records: list[dict] = []
    intervention_records: list[dict] = []

    if not args.skip_ollama:
        recover_records, _recover_stats = build_recover_output_records(
            masked_records,
            backend=args.recovery_backend,
            num_samples=args.num_samples,
            model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=args.seed,
        )
        _validate_and_write(
            recover_records,
            paths["recover_outputs"],
            validate_recover_output_record,
        )

        recover_score_records, _score_stats = build_recover_score_records(
            recover_records,
            score_backend=args.recover_score_backend,
        )
        _validate_and_write(
            recover_score_records,
            paths["recover_scores"],
            validate_recover_score_record,
        )

        unit_evidence_records, _unit_stats = build_unit_evidence_records(
            semantic_records,
            recover_score_records,
            evidence_backend=args.unit_evidence_backend,
        )
        _validate_and_write(
            unit_evidence_records,
            paths["unit_evidence"],
            validate_unit_evidence_record,
        )

        attention_anchor_records, _anchor_stats = build_attention_anchor_label_records(
            unit_evidence_records,
            label_backend=args.attention_label_backend,
        )
        _validate_and_write(
            attention_anchor_records,
            paths["attention_anchor_labels"],
            validate_attention_anchor_label_record,
        )

        intervention_records, _manifest_stats = build_intervention_manifest_records(
            attention_anchor_records,
            intervention_type=args.intervention_type,
            intervention_backend=args.intervention_backend,
            mask_token=args.mask_token,
        )
        _validate_and_write(
            intervention_records,
            paths["intervention_manifest"],
            validate_intervention_manifest_record,
        )

    report = build_real_signal_report(
        args=args,
        output_dir=output_dir,
        ablated_records=ablated_records,
        nli_records=nli_records,
        semantic_records=semantic_records,
        masked_records=masked_records,
        recover_records=recover_records,
        recover_score_records=recover_score_records,
        unit_evidence_records=unit_evidence_records,
        attention_anchor_records=attention_anchor_records,
        intervention_records=intervention_records,
        recovery_skipped=args.skip_ollama,
    )
    write_json(report, paths["report_json"])
    write_report_markdown(report, paths["report_md"])
    return report


def build_real_signal_report(
    *,
    args: argparse.Namespace,
    output_dir: str | Path,
    ablated_records: list[dict],
    nli_records: list[dict],
    semantic_records: list[dict],
    masked_records: list[dict],
    recover_records: list[dict],
    recover_score_records: list[dict],
    unit_evidence_records: list[dict],
    attention_anchor_records: list[dict],
    intervention_records: list[dict],
    recovery_skipped: bool = False,
) -> dict[str, Any]:
    return {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "nli_backend": args.nli_backend,
            "language": args.language,
            "en_model": args.en_model,
            "zh_model": args.zh_model,
            "recovery_backend": args.recovery_backend,
            "ollama_model": args.ollama_model,
            "ollama_base_url": args.ollama_base_url,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "limit": args.limit,
            "recovery_skipped": recovery_skipped,
        },
        "recovery_skipped": recovery_skipped,
        "input_counts": {
            "num_ablated_questions": len(ablated_records),
        },
        "output_counts": {
            "num_nli_scores": len(nli_records),
            "num_semantic_labels": len(semantic_records),
            "num_masked_questions": len(masked_records),
            "num_recover_outputs": len(recover_records),
            "num_recover_scores": len(recover_score_records),
            "num_unit_evidence": len(unit_evidence_records),
            "num_attention_anchor_labels": len(attention_anchor_records),
            "num_intervention_manifest": len(intervention_records),
        },
        "nli_backend_counts": _count_by(nli_records, "nli_backend"),
        "language_counts": _count_by(nli_records, "language"),
        "semantic_necessity_label_counts": _count_by(
            semantic_records,
            "semantic_necessity_label",
        ),
        "is_semantically_necessary_counts": _count_by(
            semantic_records,
            "is_semantically_necessary",
        ),
        "recoverability_label_counts": _count_by(
            recover_score_records,
            "recoverability_label",
        ),
        "misleading_recovery_counts": _count_by(
            recover_score_records,
            "misleading_recovery",
        ),
        "attention_anchor_label_counts": _count_by(
            attention_anchor_records,
            "attention_anchor_label",
        ),
        "intervention_type_counts": _count_by(intervention_records, "intervention_type"),
        "empty_recovery_count": sum(
            1 for record in recover_records if record.get("recovered_question", "") == ""
        ),
        "mask_remaining_count": sum(
            1
            for record in recover_records
            if record.get("mask_token", "") in record.get("recovered_question", "")
        ),
        "exact_match_recovery_count": sum(
            1
            for record in recover_records
            if normalize_text_for_report(record.get("recovered_question", ""))
            == normalize_text_for_report(record.get("original_question", ""))
        ),
        "sample_records": build_sample_records(
            semantic_records,
            recover_score_records,
            attention_anchor_records,
            limit=10,
        ),
        "known_limitations": list(KNOWN_LIMITATIONS),
        "next_step_recommendation": DEFAULT_NEXT_STEP,
    }


def build_sample_records(
    semantic_records: list[dict],
    recover_score_records: list[dict],
    attention_anchor_records: list[dict],
    limit: int = 10,
) -> list[dict]:
    semantic_by_unit: dict[tuple[str, str], dict] = {}
    for record in sorted(
        semantic_records,
        key=lambda item: (
            item.get("id", ""),
            item.get("unit_id", ""),
            item.get("ablation_type", ""),
            item.get("semantic_label_id", ""),
        ),
    ):
        semantic_by_unit.setdefault((record.get("id", ""), record.get("unit_id", "")), record)

    anchor_by_unit = {
        (record.get("id", ""), record.get("unit_id", "")): record
        for record in attention_anchor_records
    }

    samples: list[dict] = []
    for record in recover_score_records[:limit]:
        key = (record.get("id", ""), record.get("unit_id", ""))
        semantic = semantic_by_unit.get(key, {})
        anchor = anchor_by_unit.get(key, {})
        recovered_questions = record.get("recovered_questions", [])
        samples.append(
            {
                "masked_id": record.get("masked_id"),
                "id": record.get("id"),
                "unit_id": record.get("unit_id"),
                "masked_question": record.get("masked_question"),
                "recovered_question": recovered_questions[0] if recovered_questions else "",
                "recoverability_label": record.get("recoverability_label"),
                "recoverability_score": record.get("recoverability_score"),
                "semantic_necessity_label": semantic.get("semantic_necessity_label"),
                "attention_anchor_label": anchor.get("attention_anchor_label"),
                "attention_importance_score": anchor.get("attention_importance_score"),
            }
        )
    return samples


def write_json(record: dict, path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report_markdown(report: dict, path: str | Path) -> None:
    lines = [
        "# Sprint 1N Real Signal Report",
        "",
        f"- recovery_skipped: {report['recovery_skipped']}",
        f"- num_nli_scores: {report['output_counts']['num_nli_scores']}",
        f"- num_recover_outputs: {report['output_counts']['num_recover_outputs']}",
        f"- empty_recovery_count: {report['empty_recovery_count']}",
        f"- mask_remaining_count: {report['mask_remaining_count']}",
        f"- exact_match_recovery_count: {report['exact_match_recovery_count']}",
        "",
        "## Known Limitations",
    ]
    lines.extend(f"- {item}" for item in report["known_limitations"])
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def ensure_isolated_output_dir(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    resolved_output = (
        output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    ).resolve()
    data_processed = (PROJECT_ROOT / "data" / "processed").resolve()
    if _is_relative_to(resolved_output, data_processed):
        raise ValueError("output_dir must not be inside data/processed")


def _validate_and_write(records: list[dict], path: Path, validator: Any) -> None:
    for record in records:
        validator(record)
    write_jsonl(records, path)


def _count_by(records: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(field)) for record in records).items()))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_rebuild(args)
    print(f"output_dir: {args.output_dir}")
    print(f"recovery_skipped: {report['recovery_skipped']}")
    for key, value in report["output_counts"].items():
        print(f"{key}: {value}")
    print(f"empty_recovery_count: {report['empty_recovery_count']}")
    print(f"mask_remaining_count: {report['mask_remaining_count']}")
    print(f"exact_match_recovery_count: {report['exact_match_recovery_count']}")
    print(f"[OK] Built real signal report: {args.output_dir}/real_signal_report.json")


if __name__ == "__main__":
    main()
