# Code for the COLING manuscript

**Detecting Errors in Large Language Model Responses: The Incremental Predictive Value of Hidden States**

This release contains the implementation used to compare output-side predictors,
hidden-state predictors, and their fusion on open-ended identifier generation (H1)
and multiple-choice questions (MCQ), including Qwen and Llama experiments.

The paper entry points are listed below. The original root README describes the
earlier research scaffold; use this document for the manuscript experiments.

## Installation and a CPU check

Use Python 3.10 and the existing `recover_attention` environment, or an equivalent
Python environment on another machine. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_stage1_v3_confirmatory.py --arm both --dry-run
python -m pytest -q
```

`requirements-paper.txt` records the exact CPU package versions used to validate
this release. The dry run uses synthetic fixtures and does not require model
weights, production records, GPU inference, or an unblinding event. Its metrics
are software checks, not the paper's experimental results.

Tests marked `local_artifacts` run only when their original experiment inputs
are present. The public checkout skips these integration checks and still runs
the synthetic and unit tests.
Historical checks marked `pre_unblinding_workspace` also skip once the completed
result is present. These checks describe the pre-analysis workspace; tests of
the execution and unblinding gates continue to run with isolated fixtures.

## Implementation map

| Purpose | Entry point / implementation |
| --- | --- |
| Dataset and sample construction | `scripts/sprint_4D_0_build_h1_dataset.py`, `scripts/sprint_4B_1_prepare_cybermetric.py`, `scripts/build_stage1_samples.py` |
| Local generation and feature extraction | `scripts/run_stage1_production.py`, `src/recover_attention/generation/`, `src/recover_attention/evaluation/v3_production.py` |
| Output blocks, hidden-state features, and nested stacking | `src/recover_attention/evaluation/v3_blocks.py`, `v3_adapter.py`, `v3_adapter_mcq.py`, `v3_stacking.py` |
| Conditional gains, blockwise comparisons, and grouped uncertainty | `src/recover_attention/evaluation/v3_analyze.py`, `v3_bootstrap.py`, `v3_repeat_estimand.py` |
| Main Qwen analysis | `scripts/run_stage1_v3_confirmatory.py` |
| Llama comparisons on paths A and C | `scripts/run_stage1_llama_ac_analysis.py` |
| Readout-family comparisons | `scripts/posthoc_readout_probes_stage1.py`, `scripts/posthoc_readout_probe_lib.py` |
| Standalone-statement representation | `scripts/p2_prepare_h1_statements.py`, `scripts/p2_extract_h1_vectors.py`, `scripts/p2_train_probe_h1.py` |
| MCQ option-text and full-response coverage | `scripts/p3_mcq_output_baseline_check.py` |
| Path B numerical sensitivity and entity pooling | `scripts/run_stage1_tail_stabilized.py`, `scripts/run_stage1_e3_repool.py` |
| Figures from frozen summaries | `scripts/render_paper_figures_en.py`, `scripts/make_stage1_diagnostic_figures.py`, `scripts/make_stage1_e1_figure.py` |

Run an entry point with `--help` to inspect its arguments before using real inputs.
Some diagnostic scripts are intentionally specific to the original experiment's
directory layout. Model-loading entry points expose model-path arguments or use
the original local defaults; set these to your own downloaded model directories.

## Frozen results and figures

The release includes the following summaries, unchanged from the local artifacts:

- Main analysis: `outputs/stage1/confirmatory/stage1_confirmatory_result.json`.
- Llama: `outputs/stage1_llama_v3/analysis/llama_ac_analysis.json`.
- Readout comparisons: `outputs/stage1/posthoc_readout_probes/results_h1.json` and `results_mcq.json`.
- Standalone statements: `outputs/stage1/posthoc_readout_probes/p2_probe/h1_qwen_newrepr_probe.json`.
- MCQ output coverage: `outputs/stage1/posthoc_readout_probes/p3_mcq_output_baseline.json`; the subsequent verified repeat is preserved under `p3_mcq_baseline_runs/`.

The English statistical figures can be rendered without rerunning experiments:

```bash
python scripts/render_paper_figures_en.py --project-root . --output-dir outputs/paper_figures
```

The renderer reads the saved result and its SHA manifest; it does not fit models,
resample observations, or recompute the reported statistics. Generated figures
are ignored by Git. `release_manifest.jsonl` records the hashes of the published
source and audit files.

## Inputs for rerunning experiments

The raw datasets, model weights, completion records, hidden-state arrays, and
per-record prediction caches are not bundled. Full statistical reruns require
those inputs in addition to the source code; the included summaries support
inspection and figure reproduction.

Data construction uses MITRE ATT&CK, CWE, and CyberMetric. The original generation
models are Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Consult
`docs/paper/preregistration_v3.md` for the fixed design and use the corresponding
data-preparation and generation entry points to construct the required inputs.
The freeze checks and production gates verify the identity and completeness of
the input files before formal analysis. A fresh replication should maintain its
own manifests and output directory rather than replace the published summaries.

The local generation environment used PyTorch 2.11.0+cu128, Transformers 5.12.1,
Accelerate 1.14.0, and bitsandbytes 0.49.2. These GPU dependencies are optional for
the CPU checks above. Preserve model, tokenizer, decoding settings, and the
8-bit backend when matching the reported experiment.

## Repository contents

`.gitignore` permits the release's source code, tests, configuration, frozen
design files, and selected result summaries. Model/data directories, caches,
temporary files, IDE settings, credentials, paper drafts, TeX build products,
and internal task histories remain local. Removing a path from Git tracking
does not delete its local copy. Existing experimental results are not overwritten
by publication.
