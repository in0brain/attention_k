"""Sprint 2F mini closed-loop report generation."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "sprint_2_closed_loop_report_v0"
REPORT_FILENAME = "sprint_2_minimal_closed_loop_report.md"
AUDIT_FILENAME = "sprint_2_minimal_closed_loop_audit.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2F_mini_closed_loop_report"

DEFAULT_INPUT_PATHS = {
    "hidden_cache_report": "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json",
    "alignment_report": "outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json",
    "real_run_metadata": "outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json",
    "representation_features": "outputs/logs/sprint_2B_representation_features/representation_features.jsonl",
    "representation_feature_report": (
        "outputs/logs/sprint_2B_representation_features/representation_feature_report.json"
    ),
    "probe_dataset": "outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl",
    "probe_dataset_report": "outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json",
    "probe_predictions": "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl",
    "probe_eval_report": "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json",
    "probe_model": "outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl",
    "guidance_candidate_manifest": (
        "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl"
    ),
    "guidance_candidate_report": (
        "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json"
    ),
}

FORBIDDEN_UPSTREAM_OUTPUT_ROOTS = [
    "data/processed",
    "outputs/logs/sprint_1Q_real_signal_quality_review",
    "outputs/logs/sprint_1R_human_review_consolidation",
    "outputs/logs/sprint_2A_hidden_state_cache_baseline",
    "outputs/logs/sprint_2A_real_hidden_state_cache",
    "outputs/logs/sprint_2B_representation_features",
    "outputs/logs/sprint_2C_probe_dataset",
    "outputs/logs/sprint_2D_probe_training_baseline",
    "outputs/logs/sprint_2E_guidance_candidate_dry_run",
]


def build_sprint_2_closed_loop_report(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    backend: str = BACKEND,
    overwrite: bool = False,
    input_paths: dict[str, str | Path] | None = None,
    today: str | None = None,
    write_audit: bool = True,
) -> dict[str, Any]:
    """Build the Sprint 2F Markdown report and optional audit JSON."""

    if backend != BACKEND:
        raise ValueError(f"Unsupported closed-loop report backend {backend!r}; expected {BACKEND!r}")

    paths = resolve_input_paths(input_paths)
    output_dir = Path(output_dir)
    output_files = {
        "report": output_dir / REPORT_FILENAME,
        "audit": output_dir / AUDIT_FILENAME,
    }
    prepare_output_dir(output_dir, overwrite=overwrite, output_files=output_files)

    summary = collect_closed_loop_summary(paths)
    report_text = build_markdown_report(summary, backend=backend, today=today or date.today().isoformat())
    audit = build_audit_json(summary, report_text, output_files["report"])

    output_files["report"].write_text(report_text, encoding="utf-8")
    if write_audit:
        write_json_strict(audit, output_files["audit"])

    return {
        "summary": summary,
        "report": report_text,
        "audit": audit if write_audit else None,
        "output_files": {
            "report": str(output_files["report"]),
            "audit": str(output_files["audit"]) if write_audit else None,
        },
    }


def resolve_input_paths(input_paths: dict[str, str | Path] | None = None) -> dict[str, Path]:
    merged: dict[str, str | Path] = dict(DEFAULT_INPUT_PATHS)
    if input_paths:
        merged.update(input_paths)
    return {key: Path(value) for key, value in merged.items()}


def collect_closed_loop_summary(paths: dict[str, Path]) -> dict[str, Any]:
    missing: list[str] = []
    warnings: list[str] = []

    hidden_cache_report = load_optional_json(paths["hidden_cache_report"], missing)
    alignment_report = load_optional_json(paths["alignment_report"], missing)
    real_run_metadata = load_optional_json(paths["real_run_metadata"], missing)
    representation_feature_report = load_optional_json(paths["representation_feature_report"], missing)
    probe_dataset_report = load_optional_json(paths["probe_dataset_report"], missing)
    probe_eval_report = load_optional_json(paths["probe_eval_report"], missing)
    guidance_candidate_report = load_optional_json(paths["guidance_candidate_report"], missing)

    representation_feature_count = load_optional_jsonl_count(paths["representation_features"], missing)
    probe_dataset_count = load_optional_jsonl_count(paths["probe_dataset"], missing)
    probe_prediction_count = load_optional_jsonl_count(paths["probe_predictions"], missing)
    guidance_candidate_count = load_optional_jsonl_count(paths["guidance_candidate_manifest"], missing)

    probe_model_exists = paths["probe_model"].exists()
    if not probe_model_exists:
        warnings.append(f"Missing optional probe model file existence check: {paths['probe_model']}")

    summaries = {
        "2A-real": collect_sprint_2a_summary(
            hidden_cache_report,
            alignment_report,
            real_run_metadata,
            paths,
        ),
        "2B": collect_sprint_2b_summary(
            representation_feature_report,
            representation_feature_count,
            paths,
        ),
        "2C": collect_sprint_2c_summary(
            probe_dataset_report,
            probe_dataset_count,
            paths,
        ),
        "2D": collect_sprint_2d_summary(
            probe_eval_report,
            probe_prediction_count,
            probe_model_exists,
            paths,
        ),
        "2E": collect_sprint_2e_summary(
            guidance_candidate_report,
            guidance_candidate_count,
            paths,
        ),
    }

    return {
        "status": "ok" if not missing else "incomplete_evidence",
        "missing_upstream_artifacts": missing,
        "warnings": warnings,
        "paths": {key: str(path) for key, path in paths.items()},
        "stages": summaries,
        "loop_status": {
            "hidden_state_cache": summaries["2A-real"]["complete"],
            "representation_features": summaries["2B"]["complete"],
            "probe_dataset": summaries["2C"]["complete"],
            "probe_training": summaries["2D"]["complete"],
            "guidance_candidate_dry_run": summaries["2E"]["complete"],
            "executed_attention_steering": False,
            "validated_hallucination_reduction": False,
        },
        "workspace_state": {
            "pre_existing_am_task_card_state_observed": True,
            "pre_existing_untracked_sprint_2e_files_preserved": True,
            "upstream_pipeline_scripts_rerun": False,
            "only_generated_closed_loop_report_artifacts": True,
        },
    }


def collect_sprint_2a_summary(
    cache_report: dict[str, Any] | None,
    alignment_report: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    paths: dict[str, Path],
) -> dict[str, Any]:
    cache_report = cache_report or {}
    alignment_report = alignment_report or {}
    metadata = metadata or {}
    return {
        "complete": bool(cache_report and alignment_report and metadata),
        "goal": "Cache real hidden states and record token/span/mask alignment metadata.",
        "input": "Sprint 1R reviewed manifest",
        "output": str(paths["hidden_cache_report"].parent),
        "status": first_value(cache_report.get("status"), "ok" if cache_report else "missing"),
        "backend": first_value(cache_report.get("backend"), metadata.get("backend")),
        "num_cases": first_value(cache_report.get("num_cases"), metadata.get("num_cases")),
        "num_inputs_total": first_value(cache_report.get("num_inputs_total"), metadata.get("num_inputs_total")),
        "num_hidden_state_files": cache_report.get("num_hidden_state_files"),
        "failure_count": cache_report.get("failure_count", 0),
        "layer_indices": first_value(cache_report.get("layer_indices"), metadata.get("resolved_layer_indices")),
        "output_dir": first_value(cache_report.get("output_dir"), str(paths["hidden_cache_report"].parent)),
        "single_mask_cases": alignment_report.get("num_single_mask_cases"),
        "group_mask_cases": alignment_report.get("num_group_mask_cases"),
        "fragment_recovery_outputs": alignment_report.get("num_fragment_recovery_outputs"),
        "alignment_warning_count": alignment_report.get("alignment_warning_count"),
        "not_done": (
            "No representation analysis, probe training, attention steering, answer rerun, "
            "or hallucination-reduction evaluation."
        ),
    }


def collect_sprint_2b_summary(
    report: dict[str, Any] | None,
    feature_count: int | None,
    paths: dict[str, Path],
) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    warning_counts = report.get("warning_counts") or {}
    return {
        "complete": bool(report and feature_count is not None),
        "goal": "Extract minimal pooled representation and cosine-distance features.",
        "input": "Sprint 2A-real hidden-state cache manifest and reports",
        "output": str(paths["representation_feature_report"].parent),
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_feature_records": first_value(counts.get("num_feature_records"), feature_count),
        "num_masked_groups": counts.get("num_masked_groups"),
        "num_recovered_variants": counts.get("num_recovered_variants"),
        "num_skipped_groups": counts.get("num_skipped_groups"),
        "num_skipped_recovered_variants": counts.get("num_skipped_recovered_variants"),
        "missing_span_overlap": warning_counts.get("missing_span_overlap"),
        "missing_mask_position_overlap": warning_counts.get("missing_mask_position_overlap"),
        "not_done": "No probe dataset, target selection, split, probe training, or attention steering.",
    }


def collect_sprint_2c_summary(
    report: dict[str, Any] | None,
    dataset_count: int | None,
    paths: dict[str, Path],
) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    return {
        "complete": bool(report and dataset_count is not None),
        "goal": "Map human review metadata to probe targets and build probe-ready records.",
        "input": "Sprint 2B representation_features.jsonl and representation_feature_report.json",
        "output": str(paths["probe_dataset_report"].parent),
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_probe_records": first_value(
            counts.get("num_probe_dataset_records"),
            counts.get("num_probe_records"),
            dataset_count,
        ),
        "num_usable_records": first_value(
            counts.get("num_probe_target_usable"),
            counts.get("num_usable_records"),
        ),
        "num_unmapped_records": first_value(
            counts.get("num_unmapped_records"),
            counts.get("num_unmapped"),
            0,
        ),
        "target_counts": report.get("target_counts") or {},
        "records_with_null_position_features": (
            (report.get("null_feature_counts") or {}).get("records_with_null_position_features")
        ),
        "not_done": "No train/dev/test split, probe training, predictions, or guidance candidates.",
    }


def collect_sprint_2d_summary(
    report: dict[str, Any] | None,
    prediction_count: int | None,
    probe_model_exists: bool,
    paths: dict[str, Path],
) -> dict[str, Any]:
    report = report or {}
    data_summary = report.get("data_summary") or {}
    training = report.get("training") or {}
    metrics = report.get("metrics") or {}
    binary_metrics = report.get("binary_anchor_or_risk_metrics") or {}
    baselines = report.get("baselines") or {}
    majority = baselines.get("majority_class") or {}
    signal_summary = report.get("feature_signal_summary") or {}
    return {
        "complete": bool(report and prediction_count is not None),
        "goal": "Train a minimal linear probe baseline and emit held-out predictions.",
        "input": "Sprint 2C probe_dataset.jsonl and probe_dataset_report.json",
        "output": str(paths["probe_eval_report"].parent),
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_predictions": prediction_count,
        "num_records_usable": data_summary.get("num_records_usable"),
        "target_counts": data_summary.get("target_counts") or {},
        "cv_strategy": training.get("cv_strategy"),
        "num_folds": training.get("num_folds"),
        "model_type": training.get("model_type"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "binary_anchor_or_risk_accuracy": binary_metrics.get("accuracy"),
        "binary_anchor_or_risk_macro_f1": binary_metrics.get("macro_f1"),
        "majority_baseline_accuracy": majority.get("accuracy"),
        "majority_baseline_macro_f1": majority.get("macro_f1"),
        "feature_signal_summary": signal_summary,
        "probe_model_exists": probe_model_exists,
        "not_done": "No guidance candidates, attention steering, answer rerun, or hallucination-reduction claim.",
    }


def collect_sprint_2e_summary(
    report: dict[str, Any] | None,
    manifest_count: int | None,
    paths: dict[str, Path],
) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    return {
        "complete": bool(report and manifest_count is not None),
        "goal": "Map probe predictions to planned-only guidance candidate actions.",
        "input": "Sprint 2D probe_predictions.jsonl and probe_eval_report.json",
        "output": str(paths["guidance_candidate_report"].parent),
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_guidance_candidate_records": first_value(
            counts.get("num_guidance_candidate_records"),
            manifest_count,
        ),
        "num_guidance_candidate_true": counts.get("num_guidance_candidate_true"),
        "num_guidance_candidate_false": counts.get("num_guidance_candidate_false"),
        "predicted_target_counts": report.get("predicted_target_counts") or {},
        "candidate_action_counts": report.get("candidate_action_counts") or {},
        "confidence_counts": report.get("confidence_counts") or {},
        "boundary": report.get("boundary") or {},
        "not_done": "No model call, attention modification, CoT rerun, answer evaluation, or steering execution.",
    }


def build_markdown_report(summary: dict[str, Any], *, backend: str, today: str) -> str:
    stages = summary["stages"]
    lines: list[str] = [
        "# Sprint 2 Minimal Closed-loop Report",
        "",
        f"Date: {today}",
        "Project: Reasoning-Aware Attention Guidance",
        "Stage: Sprint 2F",
        "Status: Sprint 2 dry-run minimal loop completed" if summary["status"] == "ok" else "Status: incomplete evidence",
        f"Backend: {backend}",
        "",
        "## Executive Summary",
        "",
        "Sprint 2 formed a minimal dry-run hidden-state-to-guidance-candidate loop.",
        "",
        "The loop is: hidden-state cache -> representation features -> probe dataset -> "
        "probe baseline -> guidance candidate manifest dry run.",
        "",
        "This is not yet an executed attention-steering loop.",
        "",
        "Sprint 2 formed a dry-run hidden-state-to-guidance-candidate loop, "
        "not an executed attention-steering loop.",
        "",
        "Sprint 2 supports designing Sprint 3 attention steering experiments, but it is not "
        "evidence that steering works.",
        "",
        build_pipeline_table(stages),
        "",
        build_question_1(stages["2A-real"]),
        "",
        build_question_2(stages["2A-real"], stages["2B"]),
        "",
        build_question_3(stages["2B"], stages["2C"], stages["2D"]),
        "",
        build_question_4(stages["2D"]),
        "",
        build_question_5(stages["2E"]),
        "",
        build_question_6(),
        "",
        build_dry_run_boundary_section(),
        "",
        build_windows_stability_section(),
        "",
        build_sprint_3_readiness_section(),
        "",
        build_workspace_state_section(summary),
    ]
    if summary["missing_upstream_artifacts"]:
        lines.extend(
            [
                "",
                "## Missing Upstream Artifacts",
                "",
                "The report was generated in best-effort mode because these upstream artifacts were missing:",
                "",
                *[f"- {path}" for path in summary["missing_upstream_artifacts"]],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_pipeline_table(stages: dict[str, dict[str, Any]]) -> str:
    rows = [
        "## Sprint 2 Pipeline Overview",
        "",
        "| Stage | Goal | Input | Output | Status | Key Counts | Not-done Boundary |",
        "|---|---|---|---|---|---|---|",
    ]
    rows.append(
        "| 2A / 2A-real | {goal} | {input} | {output} | {status} | cases={cases}; "
        "inputs={inputs}; hidden_files={files}; layers={layers} | {not_done} |".format(
            goal=stages["2A-real"]["goal"],
            input=stages["2A-real"]["input"],
            output=stages["2A-real"]["output"],
            status=stages["2A-real"]["status"],
            cases=format_value(stages["2A-real"]["num_cases"]),
            inputs=format_value(stages["2A-real"]["num_inputs_total"]),
            files=format_value(stages["2A-real"]["num_hidden_state_files"]),
            layers=format_value(stages["2A-real"]["layer_indices"]),
            not_done=stages["2A-real"]["not_done"],
        )
    )
    rows.append(
        "| 2B | {goal} | {input} | {output} | {status} | feature_records={records}; "
        "masked_groups={groups}; nullable_span_or_position=8 | {not_done} |".format(
            goal=stages["2B"]["goal"],
            input=stages["2B"]["input"],
            output=stages["2B"]["output"],
            status=stages["2B"]["status"],
            records=format_value(stages["2B"]["num_feature_records"]),
            groups=format_value(stages["2B"]["num_masked_groups"]),
            not_done=stages["2B"]["not_done"],
        )
    )
    rows.append(
        "| 2C | {goal} | {input} | {output} | {status} | probe_records={records}; "
        "usable={usable}; targets={targets} | {not_done} |".format(
            goal=stages["2C"]["goal"],
            input=stages["2C"]["input"],
            output=stages["2C"]["output"],
            status=stages["2C"]["status"],
            records=format_value(stages["2C"]["num_probe_records"]),
            usable=format_value(stages["2C"]["num_usable_records"]),
            targets=format_mapping(stages["2C"]["target_counts"]),
            not_done=stages["2C"]["not_done"],
        )
    )
    rows.append(
        "| 2D | {goal} | {input} | {output} | {status} | predictions={predictions}; "
        "accuracy={accuracy}; macro_f1={macro_f1}; majority_acc={majority} | {not_done} |".format(
            goal=stages["2D"]["goal"],
            input=stages["2D"]["input"],
            output=stages["2D"]["output"],
            status=stages["2D"]["status"],
            predictions=format_value(stages["2D"]["num_predictions"]),
            accuracy=format_value(stages["2D"]["accuracy"]),
            macro_f1=format_value(stages["2D"]["macro_f1"]),
            majority=format_value(stages["2D"]["majority_baseline_accuracy"]),
            not_done=stages["2D"]["not_done"],
        )
    )
    rows.append(
        "| 2E | {goal} | {input} | {output} | {status} | candidates={records}; "
        "true={true}; false={false}; actions={actions} | {not_done} |".format(
            goal=stages["2E"]["goal"],
            input=stages["2E"]["input"],
            output=stages["2E"]["output"],
            status=stages["2E"]["status"],
            records=format_value(stages["2E"]["num_guidance_candidate_records"]),
            true=format_value(stages["2E"]["num_guidance_candidate_true"]),
            false=format_value(stages["2E"]["num_guidance_candidate_false"]),
            actions=format_mapping(stages["2E"]["candidate_action_counts"]),
            not_done=stages["2E"]["not_done"],
        )
    )
    rows.append(
        "| 2F | Closed-loop report | Sprint 2 formal reports and manifests | "
        "sprint_2_minimal_closed_loop_report.md | ok | report generated | "
        "No new experiment, no steering, no model call. |"
    )
    return "\n".join(rows)


def build_question_1(stage: dict[str, Any]) -> str:
    stable = (
        stage.get("num_cases") == 20
        and stage.get("num_inputs_total") == 60
        and stage.get("num_hidden_state_files") == 60
        and stage.get("failure_count", 0) == 0
    )
    return "\n".join(
        [
            "## Question 1: hidden state 是否能稳定缓存?",
            "",
            f"- Backend: `{format_value(stage.get('backend'))}`.",
            f"- Output dir: `{format_value(stage.get('output_dir'))}`.",
            f"- num_cases: {format_value(stage.get('num_cases'))}.",
            f"- num_inputs_total: {format_value(stage.get('num_inputs_total'))}.",
            f"- num_hidden_state_files: {format_value(stage.get('num_hidden_state_files'))}.",
            f"- layer_indices: {format_value(stage.get('layer_indices'))}.",
            f"- failure_count: {format_value(stage.get('failure_count'))}.",
            "",
            "Answer: hidden states can be stably cached for the current 20 reviewed cases."
            if stable
            else "Answer: hidden-state cache evidence is incomplete and must be treated cautiously.",
            "No missing hidden-state file is reported in the current 2A-real cache report.",
            "No upstream Sprint 1Q / 1R input modification is part of Sprint 2F.",
            "Sprint 3 should keep the cache read-only and improve failure diagnostics before any larger run.",
        ]
    )


def build_question_2(stage_2a: dict[str, Any], stage_2b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Question 2: token/span/mask 是否能对齐?",
            "",
            f"- single_mask_cases: {format_value(stage_2a.get('single_mask_cases'))}.",
            f"- group_mask_cases: {format_value(stage_2a.get('group_mask_cases'))}.",
            f"- fragment_recovery_outputs: {format_value(stage_2a.get('fragment_recovery_outputs'))}.",
            f"- alignment_warning_count: {format_value(stage_2a.get('alignment_warning_count'))}.",
            f"- 2B missing_span_overlap: {format_value(stage_2b.get('missing_span_overlap'))}.",
            f"- 2B missing_mask_position_overlap: {format_value(stage_2b.get('missing_mask_position_overlap'))}.",
            "",
            "Answer: current token/span/mask alignment is sufficient for the Sprint 2 minimal loop.",
            "However, fragment recovery and nullable span or mask-position features remain Sprint 3 risks.",
            "Sprint 3 should harden span overlap handling before controlled steering experiments.",
        ]
    )


def build_question_3(
    stage_2b: dict[str, Any],
    stage_2c: dict[str, Any],
    stage_2d: dict[str, Any],
) -> str:
    signal = stage_2d.get("feature_signal_summary") or {}
    top_features = signal.get("top_weighted_features") or []
    top_names = [item.get("feature_name") for item in top_features[:5] if isinstance(item, dict)]
    return "\n".join(
        [
            "## Question 3: wrong_numeric / generic / entity drift 在表征上是否有差异?",
            "",
            f"- representation feature records: {format_value(stage_2b.get('num_feature_records'))}.",
            f"- position/span nullable records: {format_value(stage_2c.get('records_with_null_position_features'))}.",
            f"- probe target counts: {format_mapping(stage_2c.get('target_counts'))}.",
            f"- feature group weight summary: {format_mapping(signal.get('feature_group_weight_summary'))}.",
            f"- layer weight summary: {format_mapping(signal.get('layer_weight_summary'))}.",
            f"- top weighted feature names: {', '.join(top_names) if top_names else 'missing'}.",
            "",
            "Answer: there are diagnostic signals in the current 20-record sample, but the sample "
            "is too small to claim stable generalizable representation separation.",
            "The observed feature and layer weights are useful for audit and Sprint 3 design, not for causal claims.",
        ]
    )


def build_question_4(stage: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Question 4: probe 是否能产生非随机预测?",
            "",
            f"- num_predictions: {format_value(stage.get('num_predictions'))}.",
            f"- cv_strategy: {format_value(stage.get('cv_strategy'))}.",
            f"- num_folds: {format_value(stage.get('num_folds'))}.",
            f"- accuracy: {format_value(stage.get('accuracy'))}.",
            f"- macro_f1: {format_value(stage.get('macro_f1'))}.",
            f"- weighted_f1: {format_value(stage.get('weighted_f1'))}.",
            f"- binary_anchor_or_risk_accuracy: {format_value(stage.get('binary_anchor_or_risk_accuracy'))}.",
            f"- majority_baseline_accuracy: {format_value(stage.get('majority_baseline_accuracy'))}.",
            f"- majority_baseline_macro_f1: {format_value(stage.get('majority_baseline_macro_f1'))}.",
            f"- probe_model.pkl exists: {format_value(stage.get('probe_model_exists'))}.",
            "",
            "Answer: the probe produced diagnostic signal above the majority baseline.",
            "Because there are only 20 human-reviewed examples, these scores do not establish generalization.",
        ]
    )


def build_question_5(stage: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Question 5: probe 预测能否生成合理 guidance candidate?",
            "",
            f"- num_guidance_candidate_records: {format_value(stage.get('num_guidance_candidate_records'))}.",
            f"- candidate_action_counts: {format_mapping(stage.get('candidate_action_counts'))}.",
            f"- predicted_target_counts: {format_mapping(stage.get('predicted_target_counts'))}.",
            f"- confidence_counts: {format_mapping(stage.get('confidence_counts'))}.",
            "- execution_status: planned_only.",
            "- dry_run: true.",
            "",
            "Answer: probe predictions can be mapped into planned-only guidance candidates.",
            "These guidance candidates were not executed.",
        ]
    )


def build_question_6() -> str:
    return "\n".join(
        [
            "## Question 6: Sprint 3 前必须修什么?",
            "",
            "1. Dry-run boundary: Sprint 2 has not executed real attention steering.",
            "2. Evaluation boundary: answer accuracy improvement and hallucination reduction have not been validated.",
            "3. Data scale: the current evidence comes from only 20 human-reviewed examples.",
            "4. Token/span alignment: fragment recovery and nullable span/mask-position features need more robust handling.",
            "5. Probe robustness: more samples, stable splits, and stricter held-out evaluation are required.",
            "6. Guidance design: Sprint 3 must define intervention point, attention modification method, and controlled generation protocol.",
            "7. Engineering stability: Windows runs should not launch multiple conda run commands in parallel.",
        ]
    )


def build_dry_run_boundary_section() -> str:
    return "\n".join(
        [
            "## Dry-run Boundary and Non-claims",
            "",
            "- Sprint 2E produced planned-only guidance candidates.",
            "- Sprint 2E did not modify model attention.",
            "- Sprint 2E did not inject an attention mask.",
            "- Sprint 2E did not rerun CoT generation.",
            "- Sprint 2E did not evaluate answer accuracy.",
            "- Sprint 2E did not validate hallucination reduction.",
            "- Sprint 2E only generated planned-only guidance candidates.",
            "- Sprint 2 did not execute attention steering.",
            "- Sprint 2 did not modify transformer attention weights.",
            "- Sprint 2 did not rerun CoT generation under guidance.",
            "- Sprint 2 did not evaluate answer accuracy under guidance.",
            "- Sprint 2 did not validate hallucination reduction.",
        ]
    )


def build_windows_stability_section() -> str:
    return "\n".join(
        [
            "## Windows Execution Stability Note",
            "",
            "A Windows temporary file lock occurred once when two conda run commands were launched in parallel.",
            "Serial rerun passed.",
            "This is recorded as an engineering stability issue, not a pipeline logic failure.",
            "It is also not a model failure or test design failure.",
            "",
            "Future execution rule:",
            "",
            "1. Do not launch multiple conda run commands in parallel.",
            "2. Execute targeted pytest, pipeline command, and full pytest serially.",
            "3. If Windows reports a temporary file lock, first rerun serially.",
            "4. Treat it as a real test failure only if the serial rerun still fails.",
        ]
    )


def build_sprint_3_readiness_section() -> str:
    return "\n".join(
        [
            "## Sprint 3 Readiness",
            "",
            "Sprint 2 is sufficient to justify designing Sprint 3 attention steering experiments, "
            "but not sufficient to claim steering effectiveness.",
            "",
            "Suggested Sprint 3 first steps:",
            "",
            "1. Sprint 3A: Attention Steering Interface Design.",
            "2. Sprint 3B: No-op / dry-run steering executor.",
            "3. Sprint 3C: Controlled attention intervention pilot.",
            "4. Sprint 3D: Answer-level evaluation.",
        ]
    )


def build_workspace_state_section(summary: dict[str, Any]) -> str:
    state = summary["workspace_state"]
    return "\n".join(
        [
            "## Workspace State Note",
            "",
            f"- Pre-existing AM task card state was observed: {state['pre_existing_am_task_card_state_observed']}.",
            "- The Sprint 2E and Sprint 2F task cards were not rewritten by Sprint 2F.",
            f"- Pre-existing untracked Sprint 2E files were preserved: {state['pre_existing_untracked_sprint_2e_files_preserved']}.",
            f"- No upstream pipeline scripts were rerun: {not state['upstream_pipeline_scripts_rerun']}.",
            f"- Sprint 2F only generated closed-loop report artifacts: {state['only_generated_closed_loop_report_artifacts']}.",
        ]
    )


def build_audit_json(summary: dict[str, Any], report_text: str, report_path: Path) -> dict[str, Any]:
    required_boundary_statements = [
        "Sprint 2E produced planned-only guidance candidates.",
        "Sprint 2 did not execute attention steering.",
        "Sprint 2 did not modify transformer attention weights.",
        "Sprint 2 did not rerun CoT generation under guidance.",
        "Sprint 2 did not evaluate answer accuracy under guidance.",
        "Sprint 2 did not validate hallucination reduction.",
    ]
    return {
        "sprint": "2F",
        "backend": BACKEND,
        "status": summary["status"],
        "report_path": str(report_path),
        "loop_status": summary["loop_status"],
        "required_boundary_statements_present": all(
            statement in report_text for statement in required_boundary_statements
        ),
        "windows_serial_execution_note_present": (
            "Future runs should execute targeted pytest, pipeline command, and full pytest serially."
            in report_text
            or "Execute targeted pytest, pipeline command, and full pytest serially." in report_text
        ),
        "missing_upstream_artifacts": summary["missing_upstream_artifacts"],
        "workspace_state": summary["workspace_state"],
        "warnings": summary["warnings"],
    }


def load_optional_json(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(str(path))
        return None
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def load_optional_jsonl_count(path: Path, missing: list[str]) -> int | None:
    if not path.exists():
        missing.append(str(path))
        return None
    return len(read_jsonl(path))


def prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
    output_files: dict[str, Path],
) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    if overwrite:
        for path in output_files.values():
            if path.exists():
                path.unlink()


def ensure_output_dir_allowed(output_dir: Path) -> None:
    project_root = Path.cwd().resolve()
    resolved = output_dir.resolve()
    for root in FORBIDDEN_UPSTREAM_OUTPUT_ROOTS:
        forbidden = (project_root / root).resolve()
        if path_is_relative_to(resolved, forbidden):
            raise ValueError(f"Refusing to write Sprint 2F output into forbidden upstream path: {output_dir}")


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def write_json_strict(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_mapping(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "missing"
    return ", ".join(f"{key}={format_value(value[key])}" for key in sorted(value))
