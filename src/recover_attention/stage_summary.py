"""Sprint 2 final checkpoint summary and visualization generation."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when matplotlib is installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - the test environment uses the Pillow fallback.
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - retained as a clear runtime error path.
    Image = None
    ImageDraw = None
    ImageFont = None

from recover_attention.data_io import ensure_dir, read_jsonl


BACKEND = "sprint_2_stage_summary_v0"
SUMMARY_FILENAME = "sprint_2_stage_summary.md"
AUDIT_FILENAME = "sprint_2_stage_summary_audit.json"
DEFAULT_OUTPUT_DIR = "outputs/logs/sprint_2_stage_summary"

FIGURE_FILENAMES = [
    "sprint_2_pipeline_overview.png",
    "probe_target_counts.png",
    "probe_metrics_vs_baseline.png",
    "guidance_candidate_action_counts.png",
    "guidance_confidence_counts.png",
    "sprint_2_boundary_summary.png",
]

DEFAULT_INPUT_PATHS = {
    "hidden_cache_report": "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json",
    "alignment_report": "outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json",
    "real_run_metadata": "outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json",
    "representation_feature_report": (
        "outputs/logs/sprint_2B_representation_features/representation_feature_report.json"
    ),
    "representation_features": "outputs/logs/sprint_2B_representation_features/representation_features.jsonl",
    "probe_dataset_report": "outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json",
    "probe_dataset": "outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl",
    "probe_eval_report": "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json",
    "probe_predictions": "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl",
    "probe_model": "outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl",
    "guidance_candidate_report": (
        "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json"
    ),
    "guidance_candidate_manifest": (
        "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl"
    ),
    "closed_loop_report": (
        "outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_report.md"
    ),
    "closed_loop_audit": (
        "outputs/logs/sprint_2F_mini_closed_loop_report/sprint_2_minimal_closed_loop_audit.json"
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
    "outputs/logs/sprint_2F_mini_closed_loop_report",
]


def write_stage_summary(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    backend: str = BACKEND,
    overwrite: bool = False,
    input_paths: dict[str, str | Path] | None = None,
    full_pytest_passed: int | None = None,
    full_pytest_skipped: int | None = None,
    full_pytest_duration_seconds: float | None = None,
    today: str | None = None,
) -> dict[str, Any]:
    """Write Sprint 2 stage summary Markdown, audit JSON, and required PNG figures."""

    if backend != BACKEND:
        raise ValueError(f"Unsupported stage summary backend {backend!r}; expected {BACKEND!r}")

    paths = resolve_input_paths(input_paths)
    output_dir = Path(output_dir)
    output_files = build_output_files(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite, output_files=output_files)

    summary = collect_stage_summary(
        paths,
        full_pytest_passed=full_pytest_passed,
        full_pytest_skipped=full_pytest_skipped,
        full_pytest_duration_seconds=full_pytest_duration_seconds,
    )
    build_all_figures(summary, output_files["figures"])
    markdown = build_stage_summary_markdown(
        summary,
        backend=backend,
        today=today or date.today().isoformat(),
    )
    audit = build_stage_summary_audit(summary, output_files)

    output_files["summary"].write_text(markdown, encoding="utf-8")
    write_json_strict(audit, output_files["audit"])

    return {
        "summary": summary,
        "markdown": markdown,
        "audit": audit,
        "output_files": {
            "summary": str(output_files["summary"]),
            "audit": str(output_files["audit"]),
            "figures": [str(path) for path in output_files["figures"]],
        },
    }


def resolve_input_paths(input_paths: dict[str, str | Path] | None = None) -> dict[str, Path]:
    merged: dict[str, str | Path] = dict(DEFAULT_INPUT_PATHS)
    if input_paths:
        merged.update(input_paths)
    return {key: Path(value) for key, value in merged.items()}


def build_output_files(output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    return {
        "summary": output_dir / SUMMARY_FILENAME,
        "audit": output_dir / AUDIT_FILENAME,
        "figures_dir": figures_dir,
        "figures": [figures_dir / filename for filename in FIGURE_FILENAMES],
    }


def collect_stage_summary(
    paths: dict[str, Path],
    *,
    full_pytest_passed: int | None,
    full_pytest_skipped: int | None,
    full_pytest_duration_seconds: float | None,
) -> dict[str, Any]:
    missing: list[str] = []
    warnings: list[str] = []

    hidden_cache_report = load_optional_json(paths["hidden_cache_report"], missing)
    alignment_report = load_optional_json(paths["alignment_report"], missing)
    real_run_metadata = load_optional_json(paths["real_run_metadata"], missing)
    representation_feature_report = load_optional_json(paths["representation_feature_report"], missing)
    probe_dataset_report = load_optional_json(paths["probe_dataset_report"], missing)
    probe_eval_report = load_optional_json(paths["probe_eval_report"], missing)
    guidance_candidate_report = load_optional_json(paths["guidance_candidate_report"], missing)
    closed_loop_audit = load_optional_json(paths["closed_loop_audit"], missing)
    closed_loop_report_text = load_optional_text(paths["closed_loop_report"], missing)

    representation_feature_count = load_optional_jsonl_count(paths["representation_features"], missing)
    probe_dataset_count = load_optional_jsonl_count(paths["probe_dataset"], missing)
    probe_prediction_count = load_optional_jsonl_count(paths["probe_predictions"], missing)
    guidance_candidate_count = load_optional_jsonl_count(paths["guidance_candidate_manifest"], missing)
    probe_model_exists = paths["probe_model"].exists()

    legacy_debug_files = {
        "representation_feature_manifest": Path(
            "outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl"
        ).exists(),
        "input_representation_summary": Path(
            "outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl"
        ).exists(),
        "feature_schema": Path("outputs/logs/sprint_2B_representation_features/feature_schema.json").exists(),
        "probe_feature_index": Path(
            "outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json"
        ).exists(),
    }

    stages = {
        "2A-real": collect_2a_real_summary(hidden_cache_report, alignment_report, real_run_metadata),
        "2B": collect_2b_summary(representation_feature_report, representation_feature_count),
        "2C": collect_2c_summary(probe_dataset_report, probe_dataset_count),
        "2D": collect_2d_summary(probe_eval_report, probe_prediction_count, probe_model_exists),
        "2E": collect_2e_summary(guidance_candidate_report, guidance_candidate_count),
        "2F": collect_2f_summary(closed_loop_audit, closed_loop_report_text),
    }

    loop_status = {
        "hidden_state_cache": stages["2A-real"]["complete"],
        "representation_features": stages["2B"]["complete"],
        "probe_dataset": stages["2C"]["complete"],
        "probe_training": stages["2D"]["complete"],
        "guidance_candidate_dry_run": stages["2E"]["complete"],
        "closed_loop_report": stages["2F"]["complete"],
        "executed_attention_steering": False,
        "validated_answer_accuracy_improvement": False,
        "validated_hallucination_reduction": False,
    }
    boundary = {
        "dry_run_only": True,
        "called_hf_model": False,
        "called_ollama": False,
        "called_tokenizer": False,
        "read_hidden_state_tensors": False,
        "loaded_probe_model": False,
        "retrained_probe": False,
        "reran_upstream_pipeline": False,
        "performed_attention_guidance": False,
        "modified_attention_weights": False,
        "injected_attention_mask": False,
        "reran_cot_generation": False,
        "evaluated_answer_accuracy_improvement": False,
        "validated_hallucination_reduction": False,
    }
    workspace_state = {
        "pre_existing_am_task_card_state_observed": True,
        "pre_existing_untracked_sprint_2e_files_preserved": True,
        "pre_existing_untracked_sprint_2f_files_preserved": True,
        "did_not_rewrite_task_cards": True,
        "did_not_run_upstream_pipeline_scripts_16_to_21": True,
    }
    return {
        "stage": "sprint_2_final_checkpoint",
        "status": "ok" if not missing else "incomplete_evidence",
        "paths": {key: str(path) for key, path in paths.items()},
        "inputs_read": {key: str(path) for key, path in paths.items() if key != "probe_model"},
        "probe_model_exists_checked": probe_model_exists,
        "legacy_debug_files_present": legacy_debug_files,
        "missing_upstream_artifacts": missing,
        "warnings": warnings,
        "stages": stages,
        "loop_status": loop_status,
        "boundary": boundary,
        "full_pytest": {
            "command": "conda run -n recover_attention python -m pytest -q",
            "status": "passed" if full_pytest_passed is not None else "not_recorded",
            "passed": full_pytest_passed,
            "skipped": full_pytest_skipped,
            "duration_seconds": full_pytest_duration_seconds,
        },
        "windows_stability_note_present": True,
        "serial_execution_required": True,
        "workspace_state": workspace_state,
    }


def collect_2a_real_summary(
    cache_report: dict[str, Any] | None,
    alignment_report: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    cache_report = cache_report or {}
    alignment_report = alignment_report or {}
    metadata = metadata or {}
    return {
        "complete": bool(cache_report and alignment_report and metadata),
        "stage": "2A-real",
        "goal": "Real hidden-state cache",
        "input": "Sprint 1R reviewed manifest",
        "output": "hidden_state_manifest/report/alignment metadata",
        "status": cache_report.get("status", "ok" if cache_report else "missing"),
        "backend": first_value(cache_report.get("backend"), metadata.get("backend")),
        "num_cases": first_value(cache_report.get("num_cases"), metadata.get("num_cases")),
        "num_inputs_total": first_value(cache_report.get("num_inputs_total"), metadata.get("num_inputs_total")),
        "num_hidden_state_files": cache_report.get("num_hidden_state_files"),
        "layer_indices": first_value(cache_report.get("layer_indices"), metadata.get("resolved_layer_indices")),
        "single_mask_cases": alignment_report.get("num_single_mask_cases"),
        "group_mask_cases": alignment_report.get("num_group_mask_cases"),
        "fragment_recovery_outputs": alignment_report.get("num_fragment_recovery_outputs"),
        "alignment_warning_count": alignment_report.get("alignment_warning_count"),
    }


def collect_2b_summary(report: dict[str, Any] | None, feature_count: int | None) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    warning_counts = report.get("warning_counts") or {}
    return {
        "complete": bool(report and feature_count is not None),
        "stage": "2B",
        "goal": "Representation feature extraction",
        "input": "2A-real cache metadata",
        "output": "representation_features.jsonl/report",
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_feature_records": first_value(counts.get("num_feature_records"), feature_count),
        "num_masked_groups": counts.get("num_masked_groups"),
        "num_recovered_variants": counts.get("num_recovered_variants"),
        "missing_span_overlap": warning_counts.get("missing_span_overlap"),
        "missing_mask_position_overlap": warning_counts.get("missing_mask_position_overlap"),
    }


def collect_2c_summary(report: dict[str, Any] | None, dataset_count: int | None) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    return {
        "complete": bool(report and dataset_count is not None),
        "stage": "2C",
        "goal": "Probe dataset construction",
        "input": "2B representation features",
        "output": "probe_dataset.jsonl/report",
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_probe_records": first_value(counts.get("num_probe_dataset_records"), dataset_count),
        "num_probe_target_usable": first_value(counts.get("num_probe_target_usable"), counts.get("num_usable_records")),
        "target_counts": report.get("target_counts") or {},
    }


def collect_2d_summary(
    report: dict[str, Any] | None,
    prediction_count: int | None,
    probe_model_exists: bool,
) -> dict[str, Any]:
    report = report or {}
    metrics = report.get("metrics") or {}
    baselines = report.get("baselines") or {}
    majority = baselines.get("majority_class") or {}
    training = report.get("training") or {}
    return {
        "complete": bool(report and prediction_count is not None),
        "stage": "2D",
        "goal": "Probe training baseline",
        "input": "2C probe dataset",
        "output": "probe predictions/eval report/model file",
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_predictions": prediction_count,
        "cv_strategy": training.get("cv_strategy"),
        "num_folds": training.get("num_folds"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "majority_baseline_accuracy": majority.get("accuracy"),
        "majority_baseline_macro_f1": majority.get("macro_f1"),
        "probe_model_exists": probe_model_exists,
    }


def collect_2e_summary(report: dict[str, Any] | None, manifest_count: int | None) -> dict[str, Any]:
    report = report or {}
    counts = report.get("counts") or {}
    return {
        "complete": bool(report and manifest_count is not None),
        "stage": "2E",
        "goal": "Guidance candidate dry run",
        "input": "2D probe predictions",
        "output": "guidance candidate manifest/report",
        "status": report.get("status", "missing" if not report else "ok"),
        "backend": report.get("backend"),
        "num_guidance_candidate_records": first_value(counts.get("num_guidance_candidate_records"), manifest_count),
        "num_guidance_candidate_true": counts.get("num_guidance_candidate_true"),
        "num_guidance_candidate_false": counts.get("num_guidance_candidate_false"),
        "candidate_action_counts": report.get("candidate_action_counts") or {},
        "confidence_counts": report.get("confidence_counts") or {},
        "predicted_target_counts": report.get("predicted_target_counts") or {},
    }


def collect_2f_summary(audit: dict[str, Any] | None, report_text: str | None) -> dict[str, Any]:
    audit = audit or {}
    return {
        "complete": bool(audit and report_text is not None),
        "stage": "2F",
        "goal": "Mini closed-loop report",
        "input": "2A-real through 2E formal artifacts",
        "output": "closed-loop report/audit",
        "status": audit.get("status", "missing" if not audit else "ok"),
        "closed_loop_report_chars": len(report_text or ""),
        "loop_status": audit.get("loop_status") or {},
        "required_boundary_statements_present": audit.get("required_boundary_statements_present"),
        "windows_serial_execution_note_present": audit.get("windows_serial_execution_note_present"),
    }


def build_all_figures(summary: dict[str, Any], figure_paths: list[Path]) -> None:
    figure_map = {path.name: path for path in figure_paths}
    build_pipeline_overview_figure(summary, figure_map["sprint_2_pipeline_overview.png"])
    build_probe_target_counts_figure(summary, figure_map["probe_target_counts.png"])
    build_probe_metrics_vs_baseline_figure(summary, figure_map["probe_metrics_vs_baseline.png"])
    build_guidance_candidate_action_counts_figure(summary, figure_map["guidance_candidate_action_counts.png"])
    build_guidance_confidence_counts_figure(summary, figure_map["guidance_confidence_counts.png"])
    build_boundary_summary_figure(summary, figure_map["sprint_2_boundary_summary.png"])


def build_pipeline_overview_figure(summary: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    stages = [
        ("2A-real", "hidden states"),
        ("2B", "features"),
        ("2C", "probe data"),
        ("2D", "probe"),
        ("2E", "candidates"),
        ("2F", "report"),
    ]
    if plt is None:
        build_pil_pipeline_overview_figure(path, stages)
        return
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    for index, (stage, label) in enumerate(stages):
        x = index / (len(stages) - 1)
        ax.text(
            x,
            0.58,
            f"{stage}\n{label}",
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#E8F1FA",
                "edgecolor": "#305C89",
                "linewidth": 1.2,
            },
            transform=ax.transAxes,
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=((index + 0.78) / (len(stages) - 1), 0.58),
                xytext=((index + 0.22) / (len(stages) - 1), 0.58),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#555555"},
                xycoords=ax.transAxes,
            )
    ax.text(
        0.5,
        0.18,
        "dry-run loop - not executed steering",
        ha="center",
        va="center",
        fontsize=12,
        color="#7A2E2E",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_probe_target_counts_figure(summary: dict[str, Any], path: Path) -> None:
    counts = summary["stages"]["2C"].get("target_counts") or {}
    ordered = [
        "risk_positive",
        "positive_anchor",
        "negative",
        "hard_negative_or_weak_positive",
    ]
    build_bar_figure(
        path,
        title="Sprint 2C Probe Target Counts",
        labels=ordered,
        values=[int(counts.get(label, 0)) for label in ordered],
        ylabel="records",
        color="#4C78A8",
    )


def build_probe_metrics_vs_baseline_figure(summary: dict[str, Any], path: Path) -> None:
    stage = summary["stages"]["2D"]
    labels = [
        "probe accuracy",
        "probe macro_f1",
        "majority accuracy",
        "majority macro_f1",
    ]
    values = [
        safe_float(stage.get("accuracy")),
        safe_float(stage.get("macro_f1")),
        safe_float(stage.get("majority_baseline_accuracy")),
        safe_float(stage.get("majority_baseline_macro_f1")),
    ]
    build_bar_figure(
        path,
        title="Sprint 2D Probe Metrics vs Majority Baseline",
        labels=labels,
        values=values,
        ylabel="score",
        color="#59A14F",
        ylim=(0, 1.0),
    )


def build_guidance_candidate_action_counts_figure(summary: dict[str, Any], path: Path) -> None:
    counts = summary["stages"]["2E"].get("candidate_action_counts") or {}
    ordered = [
        "increase_attention_to_original_span",
        "preserve_original_span_attention",
        "review_before_guidance",
        "no_guidance",
    ]
    build_bar_figure(
        path,
        title="Sprint 2E Guidance Candidate Action Counts",
        labels=ordered,
        values=[int(counts.get(label, 0)) for label in ordered],
        ylabel="records",
        color="#F28E2B",
    )


def build_guidance_confidence_counts_figure(summary: dict[str, Any], path: Path) -> None:
    counts = summary["stages"]["2E"].get("confidence_counts") or {}
    ordered = ["high", "medium", "low", "unknown"]
    build_bar_figure(
        path,
        title="Sprint 2E Guidance Confidence Counts",
        labels=ordered,
        values=[int(counts.get(label, 0)) for label in ordered],
        ylabel="records",
        color="#B07AA1",
    )


def build_boundary_summary_figure(summary: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    labels = [
        "hidden-state cache completed",
        "representation features completed",
        "probe dataset completed",
        "probe training completed",
        "guidance candidate dry run completed",
        "attention steering executed",
        "hallucination reduction validated",
        "answer accuracy improvement validated",
    ]
    loop = summary["loop_status"]
    values = [
        bool(loop["hidden_state_cache"]),
        bool(loop["representation_features"]),
        bool(loop["probe_dataset"]),
        bool(loop["probe_training"]),
        bool(loop["guidance_candidate_dry_run"]),
        bool(loop["executed_attention_steering"]),
        bool(loop["validated_hallucination_reduction"]),
        bool(loop["validated_answer_accuracy_improvement"]),
    ]
    if plt is None:
        build_pil_boundary_summary_figure(path, labels, values)
        return
    colors = ["#59A14F" if value else "#D65F5F" for value in values]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_positions = range(len(labels))
    ax.barh(list(y_positions), [1] * len(labels), color=colors)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Sprint 2 Boundary Summary")
    for index, value in enumerate(values):
        ax.text(
            0.5,
            index,
            "true" if value else "false",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_bar_figure(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float | int],
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    ensure_dir(path.parent)
    if plt is None:
        build_pil_bar_figure(
            path,
            title=title,
            labels=labels,
            values=values,
            ylabel=ylabel,
            color=color,
            ylim=ylim,
        )
        return
    fig_width = max(7.5, len(labels) * 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))
    ax.bar(range(len(labels)), values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for index, value in enumerate(values):
        ax.text(index, float(value) + 0.02, format_value(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_pil_pipeline_overview_figure(path: Path, stages: list[tuple[str, str]]) -> None:
    image, draw, font, title_font = new_pil_canvas(1200, 360)
    draw_centered_text(draw, image.width // 2, 28, "Sprint 2 Pipeline Overview", title_font, "#223344")
    box_width = 140
    box_height = 74
    start_x = 55
    gap = (image.width - 2 * start_x - len(stages) * box_width) / (len(stages) - 1)
    y = 125
    for index, (stage, label) in enumerate(stages):
        x = int(start_x + index * (box_width + gap))
        draw.rounded_rectangle(
            [x, y, x + box_width, y + box_height],
            radius=10,
            fill=hex_to_rgb("#E8F1FA"),
            outline=hex_to_rgb("#305C89"),
            width=2,
        )
        draw_centered_text(draw, x + box_width // 2, y + 22, stage, title_font, "#17395C")
        draw_centered_text(draw, x + box_width // 2, y + 48, label, font, "#17395C")
        if index < len(stages) - 1:
            x0 = x + box_width + 12
            x1 = int(x + box_width + gap - 12)
            arrow_y = y + box_height // 2
            draw.line([x0, arrow_y, x1, arrow_y], fill=hex_to_rgb("#555555"), width=3)
            draw.polygon(
                [(x1, arrow_y), (x1 - 10, arrow_y - 6), (x1 - 10, arrow_y + 6)],
                fill=hex_to_rgb("#555555"),
            )
    draw_centered_text(
        draw,
        image.width // 2,
        270,
        "dry-run loop - not executed steering",
        title_font,
        "#7A2E2E",
    )
    image.save(path, format="PNG")


def build_pil_bar_figure(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float | int],
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    width = max(900, 190 * len(labels))
    height = 520
    image, draw, font, title_font = new_pil_canvas(width, height)
    draw_centered_text(draw, width // 2, 26, title, title_font, "#223344")
    left = 82
    right = 35
    top = 78
    bottom = 165
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = ylim[1] if ylim else max([float(value) for value in values] + [1.0])
    if max_value <= 0:
        max_value = 1.0
    axis_color = hex_to_rgb("#444444")
    grid_color = hex_to_rgb("#D8DEE5")
    draw.line([left, top, left, top + plot_height], fill=axis_color, width=2)
    draw.line([left, top + plot_height, left + plot_width, top + plot_height], fill=axis_color, width=2)
    for tick in range(1, 5):
        y = top + plot_height - int(plot_height * tick / 4)
        draw.line([left, y, left + plot_width, y], fill=grid_color, width=1)
    draw.text((16, top + plot_height // 2 - 8), ylabel, fill=hex_to_rgb("#333333"), font=font)
    bar_slot = plot_width / max(1, len(labels))
    bar_width = max(28, int(bar_slot * 0.54))
    bar_color = hex_to_rgb(color)
    for index, value in enumerate(values):
        numeric_value = float(value)
        bar_height = int(plot_height * min(max(numeric_value, 0.0), max_value) / max_value)
        x_center = int(left + bar_slot * index + bar_slot / 2)
        x0 = x_center - bar_width // 2
        x1 = x_center + bar_width // 2
        y0 = top + plot_height - bar_height
        y1 = top + plot_height
        draw.rectangle([x0, y0, x1, y1], fill=bar_color)
        draw_centered_text(draw, x_center, max(top + 10, y0 - 16), format_value(value), font, "#222222")
        label_lines = wrap_label(labels[index], max_chars=18)
        for line_index, line in enumerate(label_lines[:4]):
            draw_centered_text(
                draw,
                x_center,
                top + plot_height + 18 + line_index * 13,
                line,
                font,
                "#333333",
            )
    image.save(path, format="PNG")


def build_pil_boundary_summary_figure(path: Path, labels: list[str], values: list[bool]) -> None:
    width = 1000
    row_height = 44
    height = 100 + row_height * len(labels)
    image, draw, font, title_font = new_pil_canvas(width, height)
    draw_centered_text(draw, width // 2, 28, "Sprint 2 Boundary Summary", title_font, "#223344")
    x0 = 44
    x1 = width - 44
    label_x = 60
    status_x = width - 210
    y = 72
    for label, value in zip(labels, values):
        fill = hex_to_rgb("#59A14F" if value else "#D65F5F")
        draw.rounded_rectangle([x0, y, x1, y + 32], radius=7, fill=fill)
        draw.text((label_x, y + 9), label, fill=hex_to_rgb("#FFFFFF"), font=font)
        draw.text((status_x, y + 9), "true" if value else "false", fill=hex_to_rgb("#FFFFFF"), font=font)
        y += row_height
    image.save(path, format="PNG")


def new_pil_canvas(width: int, height: int) -> tuple[Any, Any, Any, Any]:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Generating PNG figures requires either matplotlib or Pillow.")
    image = Image.new("RGB", (width, height), color=hex_to_rgb("#FFFFFF"))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    return image, draw, font, title_font


def draw_centered_text(draw: Any, x: int, y: int, text: str, font: Any, color: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text((x - text_width // 2, y - text_height // 2), text, fill=hex_to_rgb(color), font=font)


def wrap_label(label: str, *, max_chars: int) -> list[str]:
    if len(label) <= max_chars:
        return [label]
    parts = label.replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [label]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def build_stage_summary_markdown(summary: dict[str, Any], *, backend: str, today: str) -> str:
    full_pytest = summary["full_pytest"]
    duration = full_pytest.get("duration_seconds")
    duration_text = "not recorded" if duration is None else f"{duration:.2f} seconds"
    lines = [
        "# Sprint 2 Stage Summary",
        "",
        f"Date: {today}",
        "Project: Reasoning-Aware Attention Guidance",
        "Stage: Sprint 2 Final Checkpoint and Visualization Summary",
        "Status: Sprint 2 dry-run checkpoint completed" if summary["status"] == "ok" else "Status: incomplete evidence",
        f"Backend: {backend}",
        "",
        "## Executive Summary",
        "",
        "Sprint 2 completed a dry-run hidden-state-to-guidance-candidate loop.",
        "This is not an executed attention-steering loop.",
        "Sprint 2 established a dry-run signal path from hidden states to planned guidance candidates.",
        "Sprint 2 does not establish attention steering effectiveness.",
        "",
        build_pipeline_summary_table(summary),
        "",
        "## Key Numbers",
        "",
        *build_key_number_lines(summary, duration_text),
        "",
        "## Figures",
        "",
        "![Sprint 2 pipeline overview](figures/sprint_2_pipeline_overview.png)",
        "",
        "![Probe target counts](figures/probe_target_counts.png)",
        "",
        "![Probe metrics vs baseline](figures/probe_metrics_vs_baseline.png)",
        "",
        "![Guidance candidate action counts](figures/guidance_candidate_action_counts.png)",
        "",
        "![Guidance confidence counts](figures/guidance_confidence_counts.png)",
        "",
        "![Sprint 2 boundary summary](figures/sprint_2_boundary_summary.png)",
        "",
        "## Boundary and Non-claims",
        "",
        "- Attention guidance has not been executed.",
        "- Transformer attention weights have not been modified.",
        "- An attention mask has not been injected.",
        "- CoT generation has not been rerun under guidance.",
        "- Answer accuracy improvement has not been evaluated.",
        "- Hallucination reduction has not been validated.",
        "",
        "## Engineering Stability",
        "",
        "Sprint 2E had one Windows temporary file lock when two `conda run` commands were launched in parallel.",
        "Serial rerun passed.",
        "Future checkpoints and sprints should execute targeted pytest, pipeline command, and full pytest serially.",
        "",
        "## Sprint 3A Readiness",
        "",
        "Sprint 2 can support entering Sprint 3A: Attention Steering Interface Design.",
        "Sprint 2 is not sufficient to claim attention steering effectiveness.",
        "Recommended Sprint 3A starting point: Attention Steering Interface Design.",
        "",
        "## Workspace State",
        "",
        "- Pre-existing AM task card state was observed and preserved.",
        "- Pre-existing untracked Sprint 2E / 2F files were preserved.",
        "- No upstream pipeline scripts 16 / 17 / 18 / 19 / 20 / 21 were rerun.",
        "- This checkpoint only generated stage summary artifacts.",
    ]
    if summary["missing_upstream_artifacts"]:
        lines.extend(
            [
                "",
                "## Missing Upstream Artifacts",
                "",
                *[f"- {path}" for path in summary["missing_upstream_artifacts"]],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_pipeline_summary_table(summary: dict[str, Any]) -> str:
    stages = summary["stages"]
    rows = [
        "## Pipeline Summary",
        "",
        "| Stage | Goal | Input | Output | Key Count | Status |",
        "|---|---|---|---|---|---|",
    ]
    for key in ["2A-real", "2B", "2C", "2D", "2E", "2F"]:
        stage = stages[key]
        rows.append(
            "| {stage} | {goal} | {input} | {output} | {count} | {status} |".format(
                stage=key,
                goal=stage["goal"],
                input=stage["input"],
                output=stage["output"],
                count=stage_key_count(key, stage),
                status=stage["status"],
            )
        )
    return "\n".join(rows)


def build_key_number_lines(summary: dict[str, Any], duration_text: str) -> list[str]:
    stages = summary["stages"]
    full = summary["full_pytest"]
    return [
        "- 2A-real hidden-state cache: "
        f"cases={format_value(stages['2A-real']['num_cases'])}, "
        f"inputs={format_value(stages['2A-real']['num_inputs_total'])}, "
        f"hidden_state_files={format_value(stages['2A-real']['num_hidden_state_files'])}.",
        f"- 2B representation feature records: {format_value(stages['2B']['num_feature_records'])}.",
        f"- 2C target counts: {format_mapping(stages['2C']['target_counts'])}.",
        "- 2D probe metrics: "
        f"accuracy={format_value(stages['2D']['accuracy'])}, "
        f"macro_f1={format_value(stages['2D']['macro_f1'])}, "
        f"majority_accuracy={format_value(stages['2D']['majority_baseline_accuracy'])}, "
        f"majority_macro_f1={format_value(stages['2D']['majority_baseline_macro_f1'])}.",
        f"- 2E candidate action counts: {format_mapping(stages['2E']['candidate_action_counts'])}.",
        f"- 2E confidence counts: {format_mapping(stages['2E']['confidence_counts'])}.",
        "- 2F boundary audit: "
        f"executed_attention_steering={summary['loop_status']['executed_attention_steering']}, "
        "validated_answer_accuracy_improvement="
        f"{summary['loop_status']['validated_answer_accuracy_improvement']}, "
        f"validated_hallucination_reduction={summary['loop_status']['validated_hallucination_reduction']}.",
        f"- Full pytest passed: {format_value(full.get('passed'))} passed, {format_value(full.get('skipped'))} skipped.",
        f"- Full pytest duration: {duration_text}.",
    ]


def build_stage_summary_audit(summary: dict[str, Any], output_files: dict[str, Any]) -> dict[str, Any]:
    figure_paths = [str(path) for path in output_files["figures"]]
    return {
        "stage": "sprint_2_final_checkpoint",
        "backend": BACKEND,
        "status": summary["status"],
        "summary_path": str(output_files["summary"]),
        "figure_paths": figure_paths,
        "inputs_read": summary["inputs_read"],
        "outputs_written": {
            "summary": str(output_files["summary"]),
            "audit": str(output_files["audit"]),
            "figures": figure_paths,
        },
        "full_pytest": summary["full_pytest"],
        "loop_status": summary["loop_status"],
        "boundary": summary["boundary"],
        "windows_stability_note_present": True,
        "serial_execution_required": True,
        "workspace_state": summary["workspace_state"],
        "legacy_debug_files_present": summary["legacy_debug_files_present"],
        "missing_upstream_artifacts": summary["missing_upstream_artifacts"],
        "warnings": summary["warnings"],
    }


def prepare_output_dir(output_dir: Path, *, overwrite: bool, output_files: dict[str, Any]) -> None:
    ensure_output_dir_allowed(output_dir)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output directory already exists; pass --overwrite: {output_dir}")
    ensure_dir(output_dir)
    ensure_dir(output_files["figures_dir"])
    if overwrite:
        for path in [output_files["summary"], output_files["audit"], *output_files["figures"]]:
            if path.exists():
                path.unlink()


def ensure_output_dir_allowed(output_dir: Path) -> None:
    project_root = Path.cwd().resolve()
    resolved = output_dir.resolve()
    for root in FORBIDDEN_UPSTREAM_OUTPUT_ROOTS:
        forbidden = (project_root / root).resolve()
        if path_is_relative_to(resolved, forbidden):
            raise ValueError(f"Refusing to write stage summary into forbidden upstream path: {output_dir}")


def load_optional_json(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(str(path))
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def load_optional_jsonl_count(path: Path, missing: list[str]) -> int | None:
    if not path.exists():
        missing.append(str(path))
        return None
    return len(read_jsonl(path))


def load_optional_text(path: Path, missing: list[str]) -> str | None:
    if not path.exists():
        missing.append(str(path))
        return None
    return path.read_text(encoding="utf-8")


def write_json_strict(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def safe_float(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def stage_key_count(key: str, stage: dict[str, Any]) -> str:
    if key == "2A-real":
        return f"cases={format_value(stage.get('num_cases'))}; hidden_files={format_value(stage.get('num_hidden_state_files'))}"
    if key == "2B":
        return f"feature_records={format_value(stage.get('num_feature_records'))}"
    if key == "2C":
        return f"probe_records={format_value(stage.get('num_probe_records'))}"
    if key == "2D":
        return f"predictions={format_value(stage.get('num_predictions'))}; accuracy={format_value(stage.get('accuracy'))}"
    if key == "2E":
        return f"candidates={format_value(stage.get('num_guidance_candidate_records'))}"
    if key == "2F":
        return "closed_loop_report=true"
    return "missing"


def format_value(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def format_mapping(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "not recorded"
    return ", ".join(f"{key}={format_value(value[key])}" for key in sorted(value))
