from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from statistics import mean, median
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
    DEFAULT_MASK_TOKEN,
    build_intervention_manifest_records,
)
from recover_attention.schemas import (  # noqa: E402
    validate_attention_anchor_label_record,
    validate_intervention_manifest_record,
    validate_unit_evidence_record,
)
from recover_attention.unit_evidence import (  # noqa: E402
    DEFAULT_UNIT_EVIDENCE_BACKEND,
    build_unit_evidence_records,
)


DEFAULT_SEMANTIC_LABELS = "outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl"
DEFAULT_UPGRADED_RECOVER_SCORES = (
    "outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl"
)
DEFAULT_BASELINE_RECOVER_SCORES = "outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl"
DEFAULT_BASELINE_UNIT_EVIDENCE = "outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl"
DEFAULT_BASELINE_ATTENTION_ANCHOR_LABELS = (
    "outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl"
)
DEFAULT_BASELINE_INTERVENTION_MANIFEST = (
    "outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl"
)
DEFAULT_BASELINE_REPORT = "outputs/logs/sprint_1N_real_downstream/real_signal_report.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_1P_upgraded_downstream"
NEXT_STEP_RECOMMENDATION = "Sprint 1Q：Real Signal Quality Review"

OUTPUT_FILENAMES = {
    "unit_evidence": "unit_evidence_upgraded.jsonl",
    "attention_anchor_labels": "attention_anchor_labels_upgraded.jsonl",
    "intervention_manifest": "intervention_manifest_upgraded.jsonl",
    "report_json": "upgraded_downstream_report.json",
    "report_md": "upgraded_downstream_report.md",
}

KNOWN_LIMITATIONS = [
    "unit_evidence_upgraded 仍使用 aggregate_stub_v0。",
    "attention_anchor_labels_upgraded 仍使用 early_evidence_rule_stub_v0。",
    "intervention_manifest_upgraded 仍是 planned_only。",
    "nli_recovery_judge_v0 是 question-level NLI judge，不直接验证每个 masked span。",
    "本 sprint 未重新运行 NLI，也未重新调用 Ollama。",
    "本 sprint 未接入 hidden states / attention maps / trajectory stability / attention guidance。",
    "结果只能说明 upgraded recovery scoring 对 downstream labels 的影响，不证明 attention guidance 有效。",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild downstream records with upgraded recovery scores."
    )
    parser.add_argument("--semantic-labels", default=DEFAULT_SEMANTIC_LABELS)
    parser.add_argument("--upgraded-recover-scores", default=DEFAULT_UPGRADED_RECOVER_SCORES)
    parser.add_argument("--baseline-recover-scores", default=DEFAULT_BASELINE_RECOVER_SCORES)
    parser.add_argument("--baseline-unit-evidence", default=DEFAULT_BASELINE_UNIT_EVIDENCE)
    parser.add_argument(
        "--baseline-attention-anchor-labels",
        default=DEFAULT_BASELINE_ATTENTION_ANCHOR_LABELS,
    )
    parser.add_argument(
        "--baseline-intervention-manifest",
        default=DEFAULT_BASELINE_INTERVENTION_MANIFEST,
    )
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unit-evidence-backend", default=DEFAULT_UNIT_EVIDENCE_BACKEND)
    parser.add_argument("--attention-label-backend", default=DEFAULT_ATTENTION_LABEL_BACKEND)
    parser.add_argument("--intervention-type", default=DEFAULT_INTERVENTION_TYPE)
    parser.add_argument("--intervention-backend", default=DEFAULT_INTERVENTION_BACKEND)
    parser.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    parser.add_argument("--limit", type=positive_int, default=None)
    return parser.parse_args(argv)


def build_upgraded_downstream_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}


def run_rebuild(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    ensure_isolated_output_dir(output_dir)
    paths = build_upgraded_downstream_paths(output_dir)

    semantic_records = read_required_jsonl(args.semantic_labels)
    upgraded_recover_scores = read_required_jsonl(args.upgraded_recover_scores)
    baseline_recover_scores = read_required_jsonl(args.baseline_recover_scores)
    baseline_unit_evidence = read_required_jsonl(args.baseline_unit_evidence)
    baseline_attention_labels = read_required_jsonl(args.baseline_attention_anchor_labels)
    baseline_manifest = read_required_jsonl(args.baseline_intervention_manifest)
    baseline_report = read_required_json(args.baseline_report)

    validate_inputs(
        semantic_records=semantic_records,
        upgraded_recover_scores=upgraded_recover_scores,
        baseline_recover_scores=baseline_recover_scores,
        baseline_unit_evidence=baseline_unit_evidence,
        baseline_attention_labels=baseline_attention_labels,
        baseline_manifest=baseline_manifest,
    )

    selected = filter_records_for_limit(
        semantic_records=semantic_records,
        upgraded_recover_scores=upgraded_recover_scores,
        baseline_recover_scores=baseline_recover_scores,
        baseline_unit_evidence=baseline_unit_evidence,
        baseline_attention_labels=baseline_attention_labels,
        baseline_manifest=baseline_manifest,
        limit=args.limit,
    )

    ensure_dir(output_dir)
    unit_evidence_records, _ = build_unit_evidence_records(
        selected["semantic_records"],
        selected["upgraded_recover_scores"],
        evidence_backend=args.unit_evidence_backend,
    )
    for record in unit_evidence_records:
        validate_unit_evidence_record(record)
    write_jsonl(unit_evidence_records, paths["unit_evidence"])

    attention_label_records, _ = build_attention_anchor_label_records(
        unit_evidence_records,
        label_backend=args.attention_label_backend,
    )
    for record in attention_label_records:
        validate_attention_anchor_label_record(record)
    write_jsonl(attention_label_records, paths["attention_anchor_labels"])

    intervention_manifest_records, _ = build_intervention_manifest_records(
        attention_label_records,
        intervention_type=args.intervention_type,
        intervention_backend=args.intervention_backend,
        mask_token=args.mask_token,
    )
    for record in intervention_manifest_records:
        validate_intervention_manifest_record(record)
    write_jsonl(intervention_manifest_records, paths["intervention_manifest"])

    report = build_upgraded_downstream_report(
        args=args,
        semantic_records=selected["semantic_records"],
        baseline_recover_scores=selected["baseline_recover_scores"],
        upgraded_recover_scores=selected["upgraded_recover_scores"],
        baseline_unit_evidence=selected["baseline_unit_evidence"],
        baseline_attention_labels=selected["baseline_attention_labels"],
        baseline_manifest=selected["baseline_manifest"],
        baseline_report=baseline_report,
        unit_evidence_upgraded=unit_evidence_records,
        attention_labels_upgraded=attention_label_records,
        intervention_manifest_upgraded=intervention_manifest_records,
    )
    write_report(report, paths["report_json"], paths["report_md"])
    return report


def read_required_jsonl(path: str | Path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing required input: {jsonl_path}")
    records = read_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"Input JSONL is empty: {jsonl_path}")
    return records


def read_required_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required input: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def ensure_isolated_output_dir(output_dir: Path) -> None:
    normalized = output_dir.as_posix().replace("\\", "/").strip("/")
    forbidden_prefixes = [
        "data/processed",
        "outputs/logs/sprint_1N_real_downstream",
        "outputs/logs/sprint_1O_recovery_scoring",
    ]
    for prefix in forbidden_prefixes:
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            raise ValueError(f"Refusing to write Sprint 1P outputs under forbidden path: {output_dir}")


def validate_inputs(
    *,
    semantic_records: list[dict],
    upgraded_recover_scores: list[dict],
    baseline_recover_scores: list[dict],
    baseline_unit_evidence: list[dict],
    baseline_attention_labels: list[dict],
    baseline_manifest: list[dict],
) -> None:
    baseline_masked_ids = {record["masked_id"] for record in baseline_recover_scores}
    upgraded_masked_ids = {record["masked_id"] for record in upgraded_recover_scores}
    if baseline_masked_ids != upgraded_masked_ids:
        missing = sorted(baseline_masked_ids - upgraded_masked_ids)[:5]
        extra = sorted(upgraded_masked_ids - baseline_masked_ids)[:5]
        raise ValueError(
            "baseline and upgraded recover scores must have the same masked_id set; "
            f"missing_from_upgraded={missing}, extra_in_upgraded={extra}"
        )

    semantic_keys = {_unit_key(record) for record in semantic_records}
    upgraded_keys = {_unit_key(record) for record in upgraded_recover_scores}
    if not upgraded_keys.issubset(semantic_keys):
        missing = sorted(upgraded_keys - semantic_keys)[:5]
        raise ValueError(f"upgraded recover scores missing semantic label groups: {missing}")

    for name, records in [
        ("baseline_unit_evidence", baseline_unit_evidence),
        ("baseline_attention_anchor_labels", baseline_attention_labels),
        ("baseline_intervention_manifest", baseline_manifest),
    ]:
        if {_unit_key(record) for record in records} != upgraded_keys:
            raise ValueError(f"{name} unit key set must match upgraded recover scores")


def filter_records_for_limit(
    *,
    semantic_records: list[dict],
    upgraded_recover_scores: list[dict],
    baseline_recover_scores: list[dict],
    baseline_unit_evidence: list[dict],
    baseline_attention_labels: list[dict],
    baseline_manifest: list[dict],
    limit: int | None,
) -> dict[str, list[dict]]:
    if limit is None:
        return {
            "semantic_records": semantic_records,
            "upgraded_recover_scores": upgraded_recover_scores,
            "baseline_recover_scores": baseline_recover_scores,
            "baseline_unit_evidence": baseline_unit_evidence,
            "baseline_attention_labels": baseline_attention_labels,
            "baseline_manifest": baseline_manifest,
        }

    selected_upgraded = upgraded_recover_scores[:limit]
    selected_masked_ids = {record["masked_id"] for record in selected_upgraded}
    selected_unit_keys = {_unit_key(record) for record in selected_upgraded}
    return {
        "semantic_records": [
            record for record in semantic_records if _unit_key(record) in selected_unit_keys
        ],
        "upgraded_recover_scores": selected_upgraded,
        "baseline_recover_scores": [
            record for record in baseline_recover_scores if record["masked_id"] in selected_masked_ids
        ],
        "baseline_unit_evidence": [
            record for record in baseline_unit_evidence if _unit_key(record) in selected_unit_keys
        ],
        "baseline_attention_labels": [
            record for record in baseline_attention_labels if _unit_key(record) in selected_unit_keys
        ],
        "baseline_manifest": [
            record for record in baseline_manifest if _unit_key(record) in selected_unit_keys
        ],
    }


def build_upgraded_downstream_report(
    *,
    args: argparse.Namespace,
    semantic_records: list[dict],
    baseline_recover_scores: list[dict],
    upgraded_recover_scores: list[dict],
    baseline_unit_evidence: list[dict],
    baseline_attention_labels: list[dict],
    baseline_manifest: list[dict],
    baseline_report: dict,
    unit_evidence_upgraded: list[dict],
    attention_labels_upgraded: list[dict],
    intervention_manifest_upgraded: list[dict],
) -> dict:
    return {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "semantic_labels_path": args.semantic_labels,
            "baseline_recover_scores_path": args.baseline_recover_scores,
            "upgraded_recover_scores_path": args.upgraded_recover_scores,
            "baseline_unit_evidence_path": args.baseline_unit_evidence,
            "baseline_attention_anchor_labels_path": args.baseline_attention_anchor_labels,
            "baseline_intervention_manifest_path": args.baseline_intervention_manifest,
            "baseline_report_path": args.baseline_report,
            "output_dir": args.output_dir,
            "unit_evidence_backend": args.unit_evidence_backend,
            "attention_label_backend": args.attention_label_backend,
            "intervention_backend": args.intervention_backend,
            "intervention_type": args.intervention_type,
            "limit": args.limit,
        },
        "input_counts": {
            "num_semantic_labels": len(semantic_records),
            "num_baseline_recover_scores": len(baseline_recover_scores),
            "num_upgraded_recover_scores": len(upgraded_recover_scores),
            "num_baseline_unit_evidence": len(baseline_unit_evidence),
            "num_baseline_attention_anchor_labels": len(baseline_attention_labels),
            "num_baseline_intervention_manifest": len(baseline_manifest),
        },
        "output_counts": {
            "num_unit_evidence_upgraded": len(unit_evidence_upgraded),
            "num_attention_anchor_labels_upgraded": len(attention_labels_upgraded),
            "num_intervention_manifest_upgraded": len(intervention_manifest_upgraded),
        },
        "recover_score_comparison": build_recover_score_comparison(
            baseline_recover_scores,
            upgraded_recover_scores,
        ),
        "unit_evidence_comparison": build_unit_evidence_comparison(
            baseline_unit_evidence,
            unit_evidence_upgraded,
        ),
        "attention_anchor_label_comparison": build_attention_anchor_label_comparison(
            baseline_attention_labels,
            attention_labels_upgraded,
        ),
        "intervention_manifest_comparison": build_intervention_manifest_comparison(
            baseline_manifest,
            intervention_manifest_upgraded,
        ),
        "changed_cases": select_changed_cases(
            baseline_recover_scores=baseline_recover_scores,
            upgraded_recover_scores=upgraded_recover_scores,
            baseline_attention_labels=baseline_attention_labels,
            upgraded_attention_labels=attention_labels_upgraded,
            limit=20,
        ),
        "sample_records": build_sample_records(
            unit_evidence_upgraded,
            attention_labels_upgraded,
            limit=10,
        ),
        "baseline_report_summary": {
            "recoverability_label_counts": baseline_report.get("recoverability_label_counts", {}),
            "attention_anchor_label_counts": baseline_report.get("attention_anchor_label_counts", {}),
        },
        "known_limitations": list(KNOWN_LIMITATIONS),
        "next_step_recommendation": NEXT_STEP_RECOMMENDATION,
    }


def build_recover_score_comparison(
    baseline_records: list[dict],
    upgraded_records: list[dict],
) -> dict:
    baseline_by_masked_id = _index_by_field(baseline_records, "masked_id")
    upgraded_by_masked_id = _index_by_field(upgraded_records, "masked_id")
    common_ids = sorted(baseline_by_masked_id.keys() & upgraded_by_masked_id.keys())
    deltas = [
        upgraded_by_masked_id[masked_id]["recoverability_score"]
        - baseline_by_masked_id[masked_id]["recoverability_score"]
        for masked_id in common_ids
    ]
    return {
        "baseline_score_backend_counts": count_field(baseline_records, "score_backend"),
        "upgraded_score_backend_counts": count_field(upgraded_records, "score_backend"),
        "baseline_recoverability_label_counts": count_field(
            baseline_records,
            "recoverability_label",
        ),
        "upgraded_recoverability_label_counts": count_field(
            upgraded_records,
            "recoverability_label",
        ),
        "recoverability_label_changed_count": sum(
            baseline_by_masked_id[masked_id]["recoverability_label"]
            != upgraded_by_masked_id[masked_id]["recoverability_label"]
            for masked_id in common_ids
        ),
        "recoverability_label_transition_counts": transition_counts(
            baseline_records,
            upgraded_records,
            key_field="masked_id",
            value_field="recoverability_label",
        ),
        "baseline_recoverability_score_summary": summarize_numeric(
            [record["recoverability_score"] for record in baseline_records]
        ),
        "upgraded_recoverability_score_summary": summarize_numeric(
            [record["recoverability_score"] for record in upgraded_records]
        ),
        "recoverability_score_delta_summary": summarize_numeric(deltas),
        "misleading_recovery_changed_count": sum(
            baseline_by_masked_id[masked_id]["misleading_recovery"]
            != upgraded_by_masked_id[masked_id]["misleading_recovery"]
            for masked_id in common_ids
        ),
    }


def build_unit_evidence_comparison(
    baseline_records: list[dict],
    upgraded_records: list[dict],
) -> dict:
    baseline_by_unit = _index_by_unit(baseline_records)
    upgraded_by_unit = _index_by_unit(upgraded_records)
    common_keys = sorted(baseline_by_unit.keys() & upgraded_by_unit.keys())
    semantic_mismatch_count = sum(
        baseline_by_unit[key]["semantic_evidence"].get("num_semantic_records")
        != upgraded_by_unit[key]["semantic_evidence"].get("num_semantic_records")
        for key in common_keys
    )
    return {
        "baseline_evidence_backend_counts": count_field(baseline_records, "evidence_backend"),
        "upgraded_evidence_backend_counts": count_field(upgraded_records, "evidence_backend"),
        "unit_evidence_record_count_changed": len(baseline_records) != len(upgraded_records),
        "semantic_evidence_count_consistency": {
            "consistent": semantic_mismatch_count == 0,
            "mismatch_count": semantic_mismatch_count,
        },
        "recoverability_evidence_label_transition_counts": transition_counts(
            baseline_records,
            upgraded_records,
            key_field=("id", "unit_id"),
            value_field=("recoverability_evidence", "recoverability_label"),
        ),
    }


def build_attention_anchor_label_comparison(
    baseline_records: list[dict],
    upgraded_records: list[dict],
) -> dict:
    baseline_by_unit = _index_by_unit(baseline_records)
    upgraded_by_unit = _index_by_unit(upgraded_records)
    common_keys = sorted(baseline_by_unit.keys() & upgraded_by_unit.keys())
    deltas = [
        upgraded_by_unit[key]["attention_importance_score"]
        - baseline_by_unit[key]["attention_importance_score"]
        for key in common_keys
    ]
    return {
        "baseline_attention_anchor_label_counts": count_field(
            baseline_records,
            "attention_anchor_label",
        ),
        "upgraded_attention_anchor_label_counts": count_field(
            upgraded_records,
            "attention_anchor_label",
        ),
        "attention_anchor_label_changed_count": sum(
            baseline_by_unit[key]["attention_anchor_label"]
            != upgraded_by_unit[key]["attention_anchor_label"]
            for key in common_keys
        ),
        "attention_anchor_label_transition_counts": transition_counts(
            baseline_records,
            upgraded_records,
            key_field=("id", "unit_id"),
            value_field="attention_anchor_label",
        ),
        "baseline_attention_importance_score_summary": summarize_numeric(
            [record["attention_importance_score"] for record in baseline_records]
        ),
        "upgraded_attention_importance_score_summary": summarize_numeric(
            [record["attention_importance_score"] for record in upgraded_records]
        ),
        "attention_importance_score_delta_summary": summarize_numeric(deltas),
    }


def build_intervention_manifest_comparison(
    baseline_records: list[dict],
    upgraded_records: list[dict],
) -> dict:
    return {
        "baseline_manifest_count": len(baseline_records),
        "upgraded_manifest_count": len(upgraded_records),
        "manifest_count_changed": len(baseline_records) != len(upgraded_records),
        "baseline_intervention_type_counts": count_field(baseline_records, "intervention_type"),
        "upgraded_intervention_type_counts": count_field(upgraded_records, "intervention_type"),
        "baseline_attention_anchor_label_counts_in_manifest": count_field(
            baseline_records,
            "attention_anchor_label",
        ),
        "upgraded_attention_anchor_label_counts_in_manifest": count_field(
            upgraded_records,
            "attention_anchor_label",
        ),
    }


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    numeric = [float(value) for value in values]
    return {
        "min": _round(min(numeric)),
        "max": _round(max(numeric)),
        "mean": _round(mean(numeric)),
        "median": _round(median(numeric)),
    }


def transition_counts(
    old_records: list[dict],
    new_records: list[dict],
    *,
    key_field: str | tuple[str, ...],
    value_field: str | tuple[str, ...],
) -> dict:
    old_by_key = _index_by_key(old_records, key_field)
    new_by_key = _index_by_key(new_records, key_field)
    transitions = Counter()
    for key in sorted(old_by_key.keys() & new_by_key.keys()):
        old_value = _get_nested_value(old_by_key[key], value_field)
        new_value = _get_nested_value(new_by_key[key], value_field)
        transitions[f"{old_value} -> {new_value}"] += 1
    return dict(sorted(transitions.items()))


def select_changed_cases(
    *,
    baseline_recover_scores: list[dict],
    upgraded_recover_scores: list[dict],
    baseline_attention_labels: list[dict],
    upgraded_attention_labels: list[dict],
    limit: int,
) -> list[dict]:
    baseline_recover_by_unit = _index_by_unit(baseline_recover_scores)
    upgraded_recover_by_unit = _index_by_unit(upgraded_recover_scores)
    baseline_anchor_by_unit = _index_by_unit(baseline_attention_labels)
    upgraded_anchor_by_unit = _index_by_unit(upgraded_attention_labels)

    cases = []
    for key in sorted(upgraded_recover_by_unit.keys() & baseline_recover_by_unit.keys()):
        baseline_recover = baseline_recover_by_unit[key]
        upgraded_recover = upgraded_recover_by_unit[key]
        baseline_anchor = baseline_anchor_by_unit.get(key)
        upgraded_anchor = upgraded_anchor_by_unit.get(key)
        if baseline_anchor is None or upgraded_anchor is None:
            continue

        recoverability_score_delta = _round(
            upgraded_recover["recoverability_score"] - baseline_recover["recoverability_score"]
        )
        attention_score_delta = _round(
            upgraded_anchor["attention_importance_score"]
            - baseline_anchor["attention_importance_score"]
        )
        recover_label_changed = (
            baseline_recover["recoverability_label"] != upgraded_recover["recoverability_label"]
        )
        anchor_label_changed = (
            baseline_anchor["attention_anchor_label"] != upgraded_anchor["attention_anchor_label"]
        )
        if not (
            recover_label_changed
            or anchor_label_changed
            or recoverability_score_delta != 0
            or attention_score_delta != 0
        ):
            continue

        cases.append(
            {
                "id": upgraded_recover["id"],
                "unit_id": upgraded_recover["unit_id"],
                "masked_id": upgraded_recover["masked_id"],
                "original_question": upgraded_recover["original_question"],
                "masked_question": upgraded_recover.get("masked_question"),
                "baseline_recoverability_label": baseline_recover["recoverability_label"],
                "upgraded_recoverability_label": upgraded_recover["recoverability_label"],
                "baseline_recoverability_score": baseline_recover["recoverability_score"],
                "upgraded_recoverability_score": upgraded_recover["recoverability_score"],
                "recoverability_score_delta": recoverability_score_delta,
                "baseline_attention_anchor_label": baseline_anchor["attention_anchor_label"],
                "upgraded_attention_anchor_label": upgraded_anchor["attention_anchor_label"],
                "baseline_attention_importance_score": baseline_anchor[
                    "attention_importance_score"
                ],
                "upgraded_attention_importance_score": upgraded_anchor[
                    "attention_importance_score"
                ],
                "attention_importance_score_delta": attention_score_delta,
                "recovered_questions": upgraded_recover["recovered_questions"],
            }
        )

    cases.sort(
        key=lambda record: (
            record["baseline_attention_anchor_label"] != record["upgraded_attention_anchor_label"],
            record["baseline_recoverability_label"] != record["upgraded_recoverability_label"],
            abs(record["attention_importance_score_delta"]),
            abs(record["recoverability_score_delta"]),
        ),
        reverse=True,
    )
    return cases[:limit]


def build_sample_records(
    unit_evidence_records: list[dict],
    attention_label_records: list[dict],
    *,
    limit: int,
) -> list[dict]:
    labels_by_unit = _index_by_unit(attention_label_records)
    samples = []
    for record in unit_evidence_records[:limit]:
        key = _unit_key(record)
        label_record = labels_by_unit[key]
        samples.append(
            {
                "id": record["id"],
                "unit_id": record["unit_id"],
                "masked_id": record["recoverability_evidence"]["masked_id"],
                "semantic_necessity_label": record["semantic_evidence"]["summary_label"],
                "upgraded_recoverability_label": record["recoverability_evidence"][
                    "recoverability_label"
                ],
                "upgraded_recoverability_score": record["recoverability_evidence"][
                    "recoverability_score"
                ],
                "upgraded_attention_anchor_label": label_record["attention_anchor_label"],
                "upgraded_attention_importance_score": label_record[
                    "attention_importance_score"
                ],
            }
        )
    return samples


def count_field(records: list[dict], field: str) -> dict:
    return dict(sorted(Counter(record[field] for record in records).items()))


def write_report(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")


def build_markdown_report(report: dict) -> str:
    recover = report["recover_score_comparison"]
    anchors = report["attention_anchor_label_comparison"]
    lines = [
        "# Sprint 1P Upgraded Downstream Report",
        "",
        "## Sprint 1P summary",
        "- Rebuilt downstream records with upgraded recovery scores.",
        "- Did not rerun NLI scoring or call Ollama.",
        "",
        "## Input files",
    ]
    for key in [
        "semantic_labels_path",
        "baseline_recover_scores_path",
        "upgraded_recover_scores_path",
        "baseline_unit_evidence_path",
        "baseline_attention_anchor_labels_path",
        "baseline_intervention_manifest_path",
    ]:
        lines.append(f"- {key}: {report['run_metadata'][key]}")
    lines.extend(["", "## Output files"])
    lines.extend(
        [
            "- unit_evidence_upgraded.jsonl",
            "- attention_anchor_labels_upgraded.jsonl",
            "- intervention_manifest_upgraded.jsonl",
            "- upgraded_downstream_report.json",
            "- upgraded_downstream_report.md",
            "",
            "## Recoverability label distribution comparison",
            f"- baseline: {recover['baseline_recoverability_label_counts']}",
            f"- upgraded: {recover['upgraded_recoverability_label_counts']}",
            f"- changed_count: {recover['recoverability_label_changed_count']}",
            "",
            "## Attention anchor label distribution comparison",
            f"- baseline: {anchors['baseline_attention_anchor_label_counts']}",
            f"- upgraded: {anchors['upgraded_attention_anchor_label_counts']}",
            f"- changed_count: {anchors['attention_anchor_label_changed_count']}",
            "",
            "## Top changed cases",
        ]
    )
    for case in report["changed_cases"][:10]:
        lines.append(
            "- "
            f"{case['id']} / {case['unit_id']}: "
            f"{case['baseline_attention_anchor_label']} -> "
            f"{case['upgraded_attention_anchor_label']}; "
            f"score_delta={case['attention_importance_score_delta']}"
        )
    if not report["changed_cases"]:
        lines.append("- No changed cases.")
    lines.extend(["", "## Known limitations"])
    lines.extend(f"- {item}" for item in report["known_limitations"])
    lines.extend(["", "## Next step", report["next_step_recommendation"], ""])
    return "\n".join(lines)


def _index_by_field(records: list[dict], field: str) -> dict[Any, dict]:
    return _index_by_key(records, field)


def _index_by_unit(records: list[dict]) -> dict[tuple[str, str], dict]:
    return _index_by_key(records, ("id", "unit_id"))


def _index_by_key(records: list[dict], key_field: str | tuple[str, ...]) -> dict[Any, dict]:
    indexed = {}
    for record in records:
        key = _get_nested_value(record, key_field)
        if key in indexed:
            raise ValueError(f"duplicate record key: {key}")
        indexed[key] = record
    return indexed


def _get_nested_value(record: dict, field: str | tuple[str, ...]) -> Any:
    if isinstance(field, str):
        return record[field]
    if len(field) == 2 and field == ("id", "unit_id"):
        return record["id"], record["unit_id"]
    value: Any = record
    for part in field:
        value = value[part]
    return value


def _unit_key(record: dict) -> tuple[str, str]:
    return record["id"], record["unit_id"]


def _round(value: float) -> float:
    return round(float(value), 10)


def main() -> None:
    args = parse_args()
    try:
        report = run_rebuild(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"output_dir: {args.output_dir}")
    for key, value in report["output_counts"].items():
        print(f"{key}: {value}")
    recover = report["recover_score_comparison"]
    anchors = report["attention_anchor_label_comparison"]
    print(
        "recoverability_label_changed_count: "
        f"{recover['recoverability_label_changed_count']}"
    )
    print(
        "attention_anchor_label_changed_count: "
        f"{anchors['attention_anchor_label_changed_count']}"
    )
    print(f"num_changed_cases: {len(report['changed_cases'])}")
    print(
        "[OK] Built upgraded downstream report: "
        f"{Path(args.output_dir) / OUTPUT_FILENAMES['report_json']}"
    )


if __name__ == "__main__":
    main()
