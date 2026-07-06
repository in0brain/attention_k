"""Sprint 2G full-scale 2000-case summary, audit, and figures.

Read-only aggregation of the Sprint 2G phase reports into a Markdown summary,
an audit JSON, and a set of PNG figures. Uses matplotlib when available and
falls back to Pillow. It does not re-run any pipeline phase.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when matplotlib is installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

from recover_attention.data_io import ensure_dir

BACKEND = "full_scale_2000_summary_v0"
SUMMARY_FILENAME = "full_scale_2000_summary.md"
AUDIT_FILENAME = "full_scale_2000_audit.json"

FIGURE_FILENAMES = [
    "full_scale_pipeline_overview.png",
    "weak_target_counts.png",
    "probe_metrics_vs_baseline.png",
    "guidance_candidate_action_counts.png",
    "guidance_confidence_counts.png",
    "20_vs_2000_comparison.png",
    "boundary_summary.png",
]

BOUNDARY_LINES = [
    "This 2000-case run is weak-labeled.",
    "It is not a 2000-case human-reviewed validation.",
    "It does not execute attention steering.",
    "It does not validate hallucination reduction.",
    "It does not validate answer accuracy improvement.",
]

PIPELINE_STAGES = [
    "manifest",
    "weak labels",
    "hidden cache",
    "features",
    "probe data",
    "probe train",
    "guidance",
]


def _load_json(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_full_scale_summary(
    *,
    root_dir: str | Path,
    output_dir: str | Path,
    baseline_eval_report_path: str | Path | None = None,
    backend: str = BACKEND,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Aggregate Sprint 2G phase reports into summary, audit, and figures."""
    if backend != BACKEND:
        raise ValueError(f"Unsupported summary backend {backend!r}; expected {BACKEND!r}")

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    summary_path = output_dir / SUMMARY_FILENAME
    audit_path = output_dir / AUDIT_FILENAME

    from recover_attention.full_scale_manifest import ensure_output_dir_allowed

    ensure_output_dir_allowed(output_dir)
    for path in (summary_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path} (pass overwrite=True to replace)"
            )

    reports = {
        "manifest": _load_json(root_dir / "00_manifest" / "full_scale_manifest_report.json"),
        "weak_labels": _load_json(root_dir / "01_downstream" / "weak_label_report.json"),
        "hidden_cache": _load_json(
            root_dir / "02_hidden_state_cache" / "hidden_state_cache_report.json"
        ),
        "alignment": _load_json(
            root_dir / "02_hidden_state_cache" / "token_alignment_report.json"
        ),
        "real_run_metadata": _load_json(
            root_dir / "02_hidden_state_cache" / "real_run_metadata.json"
        ),
        "features": _load_json(
            root_dir / "03_representation_features" / "representation_feature_report.json"
        ),
        "weak_probe_dataset": _load_json(
            root_dir / "04_weak_probe_dataset" / "weak_probe_dataset_report.json"
        ),
        "probe_eval": _load_json(root_dir / "05_probe_training" / "probe_eval_report.json"),
        "guidance": _load_json(
            root_dir / "06_guidance_candidates" / "guidance_candidate_report.json"
        ),
    }
    baseline_eval = _load_json(baseline_eval_report_path) if baseline_eval_report_path else None

    ensure_dir(figures_dir)
    figure_status = _build_all_figures(reports, baseline_eval, figures_dir)

    audit = _build_audit(reports, backend=backend, root_dir=root_dir, figure_status=figure_status)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_md = _build_summary_markdown(reports, baseline_eval, audit)
    summary_path.write_text(summary_md, encoding="utf-8")

    return {
        "audit": audit,
        "output_files": {
            "summary": summary_path.as_posix(),
            "audit": audit_path.as_posix(),
            "figures_dir": figures_dir.as_posix(),
        },
        "figure_status": figure_status,
    }


def _build_audit(
    reports: dict[str, Any],
    *,
    backend: str,
    root_dir: Path,
    figure_status: dict[str, str],
) -> dict[str, Any]:
    manifest = reports.get("manifest") or {}
    weak_labels = reports.get("weak_labels") or {}
    hidden_cache = reports.get("hidden_cache") or {}
    features = reports.get("features") or {}
    weak_probe = reports.get("weak_probe_dataset") or {}
    probe_eval = reports.get("probe_eval") or {}
    guidance = reports.get("guidance") or {}

    def phase_status(report: dict[str, Any] | None) -> str:
        if not report:
            return "missing"
        return str(report.get("status", "ok"))

    return {
        "sprint": "2G-2000",
        "backend": backend,
        "status": "ok",
        "requested_num_cases": manifest.get("requested_num_cases"),
        "available_num_cases": manifest.get("available_num_cases"),
        "actual_num_cases": manifest.get("actual_num_cases"),
        "source_artifact": manifest.get("source_artifact"),
        "label_source": "weak_auto",
        "human_reviewed_full_scale": False,
        "output_root": root_dir.as_posix(),
        "phase_status": {
            "manifest": phase_status(reports.get("manifest")),
            "weak_labels": phase_status(reports.get("weak_labels")),
            "hidden_state_cache": "ok" if hidden_cache else "missing",
            "representation_features": phase_status(reports.get("features")),
            "weak_probe_dataset": phase_status(reports.get("weak_probe_dataset")),
            "probe_training": phase_status(reports.get("probe_eval")),
            "guidance_candidates": phase_status(reports.get("guidance")),
            "summary": "ok",
        },
        "hidden_state_cache": {
            "num_cases": hidden_cache.get("num_cases"),
            "num_inputs_total": hidden_cache.get("num_inputs_total"),
            "num_hidden_state_files": hidden_cache.get("num_hidden_state_files"),
            "failure_count": hidden_cache.get("failure_count"),
            "backend": hidden_cache.get("backend"),
            "model_name": (reports.get("real_run_metadata") or {}).get("model_name_or_path"),
            "layer_indices": hidden_cache.get("layer_indices"),
        },
        "representation_features": features.get("counts", {}),
        "weak_probe_dataset": {
            "usable_probe_target_counts": weak_probe.get("usable_probe_target_counts"),
            "adaptive_kfold_decision": weak_probe.get("adaptive_kfold_decision"),
        },
        "probe_training": {
            "status": probe_eval.get("status"),
            "cv_strategy": probe_eval.get("training", {}).get("cv_strategy"),
            "num_folds": probe_eval.get("training", {}).get("num_folds"),
            "metrics": probe_eval.get("metrics", {}),
            "majority_baseline": probe_eval.get("baselines", {}).get("majority_class"),
        },
        "guidance_candidates": {
            "candidate_action_counts": guidance.get("candidate_action_counts"),
            "confidence_counts": guidance.get("confidence_counts"),
        },
        "weak_target_counts": weak_labels.get("probe_target_counts"),
        "boundary": {
            "weak_labeled": True,
            "dry_run_only": True,
            "executed_attention_steering": False,
            "validated_answer_accuracy_improvement": False,
            "validated_hallucination_reduction": False,
            "human_reviewed_2000_case_validation": False,
            "overwrote_20_case_outputs": False,
        },
        "figure_status": figure_status,
        "warnings": [],
    }


def _build_summary_markdown(
    reports: dict[str, Any],
    baseline_eval: dict[str, Any] | None,
    audit: dict[str, Any],
) -> str:
    manifest = reports.get("manifest") or {}
    weak_labels = reports.get("weak_labels") or {}
    hidden_cache = reports.get("hidden_cache") or {}
    alignment = reports.get("alignment") or {}
    metadata = reports.get("real_run_metadata") or {}
    features = (reports.get("features") or {}).get("counts", {})
    weak_probe = reports.get("weak_probe_dataset") or {}
    probe_eval = reports.get("probe_eval") or {}
    guidance = reports.get("guidance") or {}
    decision = weak_probe.get("adaptive_kfold_decision", {})
    metrics = probe_eval.get("metrics", {})
    majority = probe_eval.get("baselines", {}).get("majority_class", {})

    lines: list[str] = []
    lines.append("# Sprint 2G: Full-scale Weak-labeled 2000-case Dry Run")
    lines.append("")
    for line in BOUNDARY_LINES:
        lines.append(f"> {line}")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"- Scaled the Sprint 2 diagnostic dry-run loop to "
        f"{manifest.get('actual_num_cases')} weak-labeled GSM8K cases."
    )
    lines.append(
        f"- requested_num_cases={manifest.get('requested_num_cases')}, "
        f"available_num_cases={manifest.get('available_num_cases')}, "
        f"actual_num_cases={manifest.get('actual_num_cases')}."
    )
    lines.append(
        f"- Hidden-state cache backend: {hidden_cache.get('backend')}; "
        f"model: {metadata.get('model_name_or_path')}."
    )
    lines.append(
        f"- Adaptive k-fold: {decision.get('cv_strategy')} with "
        f"num_folds={decision.get('num_folds')} "
        f"(min usable class count={decision.get('min_class_count')})."
    )
    lines.append("")

    lines.append("## 2. Data Source and Sampling")
    lines.append("")
    lines.append(f"- source_artifact: `{manifest.get('source_artifact')}`")
    lines.append(f"- sampling_rule: {manifest.get('sampling_rule')} (seed={manifest.get('seed')})")
    lines.append(f"- source_dataset: gsm8k, source_split: train")
    lines.append("")

    lines.append("## 3. Weak Labeling Rules")
    lines.append("")
    lines.append("- label_source = weak_auto, human_reviewed = false for all full-scale cases.")
    lines.append("- Weak probe targets are mapped from the deterministically chosen ablation "
                 "unit's span type (weak_label_mapping_v0).")
    lines.append(f"- weak target counts: {weak_labels.get('probe_target_counts')}")
    lines.append("")

    lines.append("## 4. Full-scale Pipeline Overview")
    lines.append("")
    lines.append("manifest -> weak labels -> hidden-state cache -> representation features "
                 "-> weak probe dataset -> probe training -> guidance candidate dry run.")
    lines.append("")
    lines.append("![pipeline overview](figures/full_scale_pipeline_overview.png)")
    lines.append("")

    lines.append("## 5. Hidden-state Cache Summary")
    lines.append("")
    lines.append(f"- backend: {hidden_cache.get('backend')}")
    lines.append(f"- num_cases: {hidden_cache.get('num_cases')}")
    lines.append(f"- num_inputs_total: {hidden_cache.get('num_inputs_total')}")
    lines.append(f"- num_hidden_state_files: {hidden_cache.get('num_hidden_state_files')}")
    lines.append(f"- failure_count: {hidden_cache.get('failure_count')}")
    lines.append(f"- layer_indices: {hidden_cache.get('layer_indices')}")
    lines.append(f"- alignment_warning_count: {alignment.get('alignment_warning_count')}")
    lines.append("")

    lines.append("## 6. Representation Feature Summary")
    lines.append("")
    lines.append(f"- num_feature_records: {features.get('num_feature_records')}")
    lines.append(f"- num_masked_groups: {features.get('num_masked_groups')}")
    lines.append(f"- num_skipped_groups: {features.get('num_skipped_groups')}")
    lines.append("")

    lines.append("## 7. Weak Probe Dataset Summary")
    lines.append("")
    lines.append(f"- num_probe_records: {weak_probe.get('counts', {}).get('num_probe_records')}")
    lines.append(f"- num_usable_records: {weak_probe.get('counts', {}).get('num_usable_records')}")
    lines.append(f"- usable_probe_target_counts: {weak_probe.get('usable_probe_target_counts')}")
    lines.append("")
    lines.append("![weak target counts](figures/weak_target_counts.png)")
    lines.append("")

    lines.append("## 8. Adaptive k-fold Probe Training Summary")
    lines.append("")
    lines.append("Probe metrics are weak-labeled diagnostic metrics, not human-supervised "
                 "validation metrics.")
    lines.append(f"- cv_strategy: {probe_eval.get('training', {}).get('cv_strategy')}")
    lines.append(f"- num_folds: {probe_eval.get('training', {}).get('num_folds')}")
    lines.append(f"- accuracy: {metrics.get('accuracy')}")
    lines.append(f"- macro_f1: {metrics.get('macro_f1')}")
    lines.append(f"- weighted_f1: {metrics.get('weighted_f1')}")
    lines.append(f"- majority baseline accuracy: {majority.get('accuracy')} "
                 f"(label={majority.get('label')})")
    lines.append("")
    lines.append("![probe metrics vs baseline](figures/probe_metrics_vs_baseline.png)")
    lines.append("")

    lines.append("## 9. Guidance Candidate Dry-run Summary")
    lines.append("")
    lines.append("All guidance candidates are planned-only (dry_run=true, "
                 "will_modify_attention=false, will_run_model=false, will_change_answer=false).")
    lines.append(f"- candidate_action_counts: {guidance.get('candidate_action_counts')}")
    lines.append(f"- confidence_counts: {guidance.get('confidence_counts')}")
    lines.append("")
    lines.append("![guidance action counts](figures/guidance_candidate_action_counts.png)")
    lines.append("")
    lines.append("![guidance confidence counts](figures/guidance_confidence_counts.png)")
    lines.append("")

    lines.append("## 10. 20-case Human-reviewed Baseline vs 2000-case Weak-labeled Run")
    lines.append("")
    lines.append("- 20-case results are a human-reviewed diagnostic baseline.")
    lines.append("- 2000-case results are a weak-labeled diagnostic scale-up.")
    lines.append("- They are not directly equivalent; the 2000-case weak-labeled metrics are "
                 "NOT more reliable than the 20-case human-reviewed baseline.")
    if baseline_eval:
        base_metrics = baseline_eval.get("metrics", {})
        lines.append(f"- 20-case accuracy: {base_metrics.get('accuracy')}, "
                     f"macro_f1: {base_metrics.get('macro_f1')}")
    lines.append("")
    lines.append("![20 vs 2000 comparison](figures/20_vs_2000_comparison.png)")
    lines.append("")

    lines.append("## 11. Boundary and Non-claims")
    lines.append("")
    for line in BOUNDARY_LINES:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("![boundary summary](figures/boundary_summary.png)")
    lines.append("")

    lines.append("## 12. Engineering Stability Notes")
    lines.append("")
    lines.append("- All phases executed serially (no parallel Python / conda runs).")
    lines.append("- Hidden-state cache used the real local Qwen2.5-7B-Instruct (4-bit) backend.")
    lines.append(f"- failure_count: {hidden_cache.get('failure_count')}.")
    lines.append("")

    lines.append("## 13. Sprint 3A Readiness")
    lines.append("")
    lines.append("Sprint 2G scaled the diagnostic dry-run loop to 2000 weak-labeled cases. "
                 "Next is either an all-train-split run or Sprint 3A (Attention Steering "
                 "Interface Design). Sprint 3A begins real attention steering design only.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def _build_all_figures(
    reports: dict[str, Any],
    baseline_eval: dict[str, Any] | None,
    figures_dir: Path,
) -> dict[str, str]:
    status: dict[str, str] = {}
    weak_labels = reports.get("weak_labels") or {}
    probe_eval = reports.get("probe_eval") or {}
    guidance = reports.get("guidance") or {}

    status["full_scale_pipeline_overview.png"] = _safe_figure(
        lambda: _overview_figure(figures_dir / "full_scale_pipeline_overview.png")
    )

    target_counts = weak_labels.get("usable_probe_target_counts") or weak_labels.get(
        "probe_target_counts"
    ) or {}
    status["weak_target_counts.png"] = _safe_figure(
        lambda: _bar_figure(
            figures_dir / "weak_target_counts.png",
            title="Weak target counts (weak-labeled, dry-run)",
            labels=list(target_counts.keys()) or ["none"],
            values=list(target_counts.values()) or [0],
            ylabel="count",
            color="#4E79A7",
        )
    )

    metrics = probe_eval.get("metrics", {})
    majority = probe_eval.get("baselines", {}).get("majority_class", {})
    status["probe_metrics_vs_baseline.png"] = _safe_figure(
        lambda: _bar_figure(
            figures_dir / "probe_metrics_vs_baseline.png",
            title="Probe metrics vs majority baseline (weak-labeled)",
            labels=["accuracy", "macro_f1", "majority_acc"],
            values=[
                _num(metrics.get("accuracy")),
                _num(metrics.get("macro_f1")),
                _num(majority.get("accuracy")),
            ],
            ylabel="score",
            color="#59A14F",
            ylim=(0.0, 1.0),
        )
    )

    action_counts = guidance.get("candidate_action_counts") or {}
    status["guidance_candidate_action_counts.png"] = _safe_figure(
        lambda: _bar_figure(
            figures_dir / "guidance_candidate_action_counts.png",
            title="Guidance candidate actions (dry-run, planned-only)",
            labels=list(action_counts.keys()) or ["none"],
            values=list(action_counts.values()) or [0],
            ylabel="count",
            color="#E15759",
        )
    )

    confidence_counts = guidance.get("confidence_counts") or {}
    status["guidance_confidence_counts.png"] = _safe_figure(
        lambda: _bar_figure(
            figures_dir / "guidance_confidence_counts.png",
            title="Guidance candidate confidence (dry-run)",
            labels=list(confidence_counts.keys()) or ["none"],
            values=list(confidence_counts.values()) or [0],
            ylabel="count",
            color="#B07AA1",
        )
    )

    base_metrics = (baseline_eval or {}).get("metrics", {})
    status["20_vs_2000_comparison.png"] = _safe_figure(
        lambda: _grouped_comparison_figure(
            figures_dir / "20_vs_2000_comparison.png",
            metrics_2000=metrics,
            metrics_20=base_metrics,
        )
    )

    status["boundary_summary.png"] = _safe_figure(
        lambda: _boundary_figure(figures_dir / "boundary_summary.png")
    )
    return status


def _safe_figure(builder: Any) -> str:
    try:
        builder()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - record but do not abort the summary
        return f"failed: {type(exc).__name__}: {exc}"


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bar_figure(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    ensure_dir(path.parent)
    if plt is not None:
        fig_width = max(7.5, len(labels) * 1.7)
        fig, ax = plt.subplots(figsize=(fig_width, 4.4))
        ax.bar(range(len(labels)), values, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    _pil_bar_figure(path, title=title, labels=labels, values=values, ylabel=ylabel,
                    color=color, ylim=ylim)


def _grouped_comparison_figure(
    path: Path,
    *,
    metrics_2000: dict[str, Any],
    metrics_20: dict[str, Any],
) -> None:
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    values_2000 = [_num(metrics_2000.get(k)) for k in keys]
    values_20 = [_num(metrics_20.get(k)) for k in keys]
    ensure_dir(path.parent)
    if plt is not None:
        import numpy as np

        positions = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.bar(positions - 0.2, values_20, width=0.4, label="20-case (human-reviewed)", color="#9C755F")
        ax.bar(positions + 0.2, values_2000, width=0.4, label="2000-case (weak-labeled)", color="#4E79A7")
        ax.set_title("20-case human-reviewed vs 2000-case weak-labeled (not equivalent)")
        ax.set_xticks(positions)
        ax.set_xticklabels(keys)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    # PIL fallback: interleave the two series as labeled bars.
    labels = []
    values = []
    for key, v20, v2000 in zip(keys, values_20, values_2000):
        labels.append(f"{key}\n20-case")
        values.append(v20)
        labels.append(f"{key}\n2000-case")
        values.append(v2000)
    _pil_bar_figure(
        path,
        title="20-case (human) vs 2000-case (weak) - not equivalent",
        labels=labels,
        values=values,
        ylabel="score",
        color="#4E79A7",
        ylim=(0.0, 1.0),
    )


def _overview_figure(path: Path) -> None:
    ensure_dir(path.parent)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(12.0, 3.2))
        ax.axis("off")
        n = len(PIPELINE_STAGES)
        for index, stage in enumerate(PIPELINE_STAGES):
            x = index / max(1, n - 1)
            ax.text(x, 0.55, stage, ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round", fc="#E8F1FA", ec="#305C89"))
            if index < n - 1:
                ax.annotate("", xy=((index + 0.85) / (n - 1), 0.55),
                            xytext=((index + 0.15) / (n - 1), 0.55),
                            arrowprops=dict(arrowstyle="->", color="#555555"))
        ax.text(0.5, 0.12, "weak-labeled dry-run loop - not attention steering",
                ha="center", color="#7A2E2E", fontsize=11)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    _pil_overview_figure(path)


def _boundary_figure(path: Path) -> None:
    labels = [
        "weak_labeled",
        "executed_attention_steering",
        "validated_hallucination_reduction",
        "validated_answer_accuracy",
        "overwrote_20_case_outputs",
    ]
    values = [True, False, False, False, False]
    ensure_dir(path.parent)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9.0, 3.6))
        colors = ["#59A14F" if v else "#D65F5F" for v in values]
        ax.barh(range(len(labels)), [1] * len(labels), color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xticks([])
        for index, value in enumerate(values):
            ax.text(0.5, index, "true" if value else "false", ha="center", va="center",
                    color="white", fontsize=10)
        ax.set_title("Sprint 2G boundary summary")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    _pil_boundary_figure(path, labels, values)


# --------------------------------------------------------------------------- #
# Pillow fallbacks                                                             #
# --------------------------------------------------------------------------- #

def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _new_canvas(width: int, height: int) -> tuple[Any, Any, Any]:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Generating PNG figures requires either matplotlib or Pillow.")
    image = Image.new("RGB", (width, height), color=_hex_to_rgb("FFFFFF"))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    return image, draw, font


def _centered(draw: Any, x: int, y: int, text: str, font: Any, color: str) -> None:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
    except Exception:  # pragma: no cover
        width = len(text) * 6
    draw.text((x - width // 2, y), text, fill=_hex_to_rgb(color), font=font)


def _pil_bar_figure(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    width = max(900, 170 * len(labels))
    height = 520
    image, draw, font = _new_canvas(width, height)
    _centered(draw, width // 2, 22, title, font, "223344")
    left, right, top, bottom = 80, 35, 70, 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = ylim[1] if ylim else max([float(v) for v in values] + [1.0])
    if max_value <= 0:
        max_value = 1.0
    draw.line([left, top, left, top + plot_height], fill=_hex_to_rgb("444444"), width=2)
    draw.line([left, top + plot_height, left + plot_width, top + plot_height],
              fill=_hex_to_rgb("444444"), width=2)
    draw.text((12, top + plot_height // 2), ylabel, fill=_hex_to_rgb("333333"), font=font)
    slot = plot_width / max(1, len(labels))
    bar_width = max(22, int(slot * 0.5))
    for index, value in enumerate(values):
        numeric = float(value)
        bar_height = int(plot_height * min(max(numeric, 0.0), max_value) / max_value)
        x_center = int(left + slot * index + slot / 2)
        draw.rectangle(
            [x_center - bar_width // 2, top + plot_height - bar_height,
             x_center + bar_width // 2, top + plot_height],
            fill=_hex_to_rgb(color),
        )
        _centered(draw, x_center, top + plot_height - bar_height - 14,
                  _format_value(value), font, "222222")
        for line_index, line in enumerate(str(labels[index]).split("\n")[:3]):
            _centered(draw, x_center, top + plot_height + 10 + line_index * 12, line, font, "333333")
    image.save(path, format="PNG")


def _pil_overview_figure(path: Path) -> None:
    image, draw, font = _new_canvas(1220, 320)
    _centered(draw, 610, 24, "Sprint 2G Full-scale Pipeline Overview", font, "223344")
    box_width, box_height = 150, 70
    start_x = 30
    stages = PIPELINE_STAGES
    gap = (1220 - 2 * start_x - len(stages) * box_width) / (len(stages) - 1)
    y = 120
    for index, stage in enumerate(stages):
        x = int(start_x + index * (box_width + gap))
        draw.rounded_rectangle([x, y, x + box_width, y + box_height], radius=10,
                               fill=_hex_to_rgb("E8F1FA"), outline=_hex_to_rgb("305C89"), width=2)
        _centered(draw, x + box_width // 2, y + 28, stage, font, "17395C")
        if index < len(stages) - 1:
            x0 = x + box_width + 8
            x1 = int(x + box_width + gap - 8)
            arrow_y = y + box_height // 2
            draw.line([x0, arrow_y, x1, arrow_y], fill=_hex_to_rgb("555555"), width=3)
            draw.polygon([(x1, arrow_y), (x1 - 9, arrow_y - 5), (x1 - 9, arrow_y + 5)],
                         fill=_hex_to_rgb("555555"))
    _centered(draw, 610, 240, "weak-labeled dry-run loop - not attention steering", font, "7A2E2E")
    image.save(path, format="PNG")


def _pil_boundary_figure(path: Path, labels: list[str], values: list[bool]) -> None:
    width = 1040
    row_height = 46
    height = 90 + row_height * len(labels)
    image, draw, font = _new_canvas(width, height)
    _centered(draw, width // 2, 26, "Sprint 2G Boundary Summary", font, "223344")
    y = 66
    for label, value in zip(labels, values):
        fill = "59A14F" if value else "D65F5F"
        draw.rounded_rectangle([40, y, width - 40, y + 34], radius=7, fill=_hex_to_rgb(fill))
        draw.text((56, y + 10), label, fill=_hex_to_rgb("FFFFFF"), font=font)
        draw.text((width - 150, y + 10), "true" if value else "false",
                  fill=_hex_to_rgb("FFFFFF"), font=font)
        y += row_height
    image.save(path, format="PNG")


def _format_value(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}"
