"""Render English labels on the existing frozen-result figures.

Only text and label layout change. No models, resampling, or result estimation
are run. Numerical artists are checked before and after relabeling.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
import numpy as np

LABELS = {
    "答案身份": "Answer identity",
    "表面形式": "Surface form",
    "文本内容": "Text content",
    "生成与采样信号": "Generation / sampling",
    "提示上下文": "Prompt context",
    "组合基线相对单块": "Baseline vs. each block",
    "融合模型相对单块": "Fusion vs. each block",
    "单侧下界 + 非劣容忍度": "One-sided lower bound + NI tolerance",
    "单侧下界 − 优效门槛（0）": "One-sided lower bound - threshold (0)",
    "绝对检测性能": "Absolute detection performance",
    "条件增量": "Conditional gain",
    "A：包含 B4": "A: with B4",
    "C：移除 B4": "C: without B4",
    "组合基线": "Combined baseline",
    "隐藏状态": "Hidden states",
    "融合模型": "Fused model",
    "三点 AUROC": "AUROC comparison",
    "精度不足": "low precision",
    "灰色区域为路径 C（移除 B4）的依赖性敏感性分析。":
        "Gray shading: path C, sensitivity to removing B4.",
    "（95% CI）": " (95% CI)",
    "：": ": ",
}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def numeric_digest(fig):
    data = []
    for ax in fig.axes:
        for line in ax.lines:
            data.append(np.asarray(line.get_xydata(), dtype=float).tolist())
        for collection in ax.collections:
            data.append(np.asarray(collection.get_offsets(), dtype=float).tolist())
            if hasattr(collection, "get_segments"):
                data.extend(np.asarray(x, dtype=float).tolist() for x in collection.get_segments())
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def translate(fig):
    before = numeric_digest(fig)
    # Fixed tick formatters restore their source labels when the figure draws.
    # Translate those formatters as well as the currently materialized Text.
    for ax in fig.axes:
        for direction in ("x", "y"):
            ticks = getattr(ax, f"get_{direction}ticks")()
            labels = [item.get_text() for item in getattr(ax, f"get_{direction}ticklabels")()]
            if any(re.search(r"[\u3400-\u9fff]", text) for text in labels):
                limits = getattr(ax, f"get_{direction}lim")()
                translated = []
                for text in labels:
                    for source, target in sorted(LABELS.items(), key=lambda pair: len(pair[0]), reverse=True):
                        text = text.replace(source, target)
                    translated.append(text)
                getattr(ax, f"set_{direction}ticks")(ticks, translated)
                getattr(ax, f"set_{direction}lim")(limits)
    for item in fig.findobj(match=Text):
        text = item.get_text()
        for source, target in sorted(LABELS.items(), key=lambda pair: len(pair[0]), reverse=True):
            text = text.replace(source, target)
        if re.search(r"[\u3400-\u9fff]", text):
            raise ValueError(f"Untranslated label: {text}")
        item.set_text(text)
        item.set_fontfamily("DejaVu Sans")
    assert before == numeric_digest(fig), "Numerical plot artists changed"
    return before

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    project = args.project_root.resolve()
    result_path = project / "outputs/stage1/confirmatory/stage1_confirmatory_result.json"
    manifest = project / "docs/paper/stage1_confirmatory_sha_manifest.json"
    diag = load_module("english_diagnostic_source", project / "scripts/make_stage1_diagnostic_figures.py")
    e1 = load_module("english_e1_source", project / "scripts/make_stage1_e1_figure.py")
    diag.configure_style()
    result, digest = diag.load_summary(result_path, manifest)
    rows = e1.load_e1_rows(result_path, manifest)
    jobs = [
        ("b4_dependency_comparison", diag.build_b4_comparison(result)),
        ("block_comparison_bounds", diag.build_block_bounds(result)),
        ("figure_02_e1_results", e1.build_figure(rows)),
    ]
    audit = [{"kind": "source", "result_sha256": digest, "manifest_sha256": sha(manifest),
              "role": "English translation of frozen figures; no statistical recomputation"}]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, fig in jobs:
        values_sha = translate(fig)
        if stem == "block_comparison_bounds":
            fig.set_size_inches(12.8, 6.8)
            fig.tight_layout(h_pad=2, w_pad=2.5)
        elif stem == "b4_dependency_comparison":
            for legend in fig.legends:
                legend.set_bbox_to_anchor((0.79, 0.5))
                for item in legend.get_texts():
                    item.set_fontsize(9)
            fig.tight_layout(rect=(0, 0, 0.79, 1), h_pad=2)
        elif stem == "figure_02_e1_results":
            for ax in fig.axes:
                for item in ax.texts:
                    if isinstance(item, Annotation) and re.fullmatch(r"[+-]\d+\.\d{4}", item.get_text()):
                        item.xyann = (0, 6)
                        item.set_ha("center")
                        item.set_va("bottom")
        assert numeric_digest(fig) == values_sha
        for extension in ("pdf", "png"):
            target = args.output_dir / f"{stem}.{extension}"
            options = {"Creator": "English label renderer", "CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(target, bbox_inches="tight", dpi=200, metadata=options)
            audit.append({"kind": "figure", "file": target.name, "sha256": sha(target),
                          "numerical_artists_unchanged": True, "numerical_artists_sha256": values_sha})
        plt.close(fig)
    assert sha(result_path) == digest
    (args.output_dir / "translation_figure_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8")
    print("Rendered three English PDF/PNG figure pairs; frozen data unchanged.")

if __name__ == "__main__":
    main()
