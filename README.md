# Hidden States for LLM Error Detection

**Detecting Errors in Large Language Model Responses: The Incremental Predictive Value of Hidden States**

This release contains the implementation used to compare output-side predictors,
hidden-state predictors, and their fusion on open-ended identifier generation (H1)
and multiple-choice questions (MCQ), including Qwen and Llama experiments.

The research question is how much hidden states improve error detection after
combining the output-side information already available at inference time. The
repository includes the main comparisons, blockwise checks, alternative readouts,
statement representations, and MCQ output-text coverage experiments.

The Python package retains its original name, `recover_attention`. The entry
points below identify the current paper workflow among the earlier research
utilities.

## Installation and a CPU check

Use Python 3.10. Clone the repository and activate a suitable environment:

```bash
git clone https://github.com/in0brain/attention_k.git
cd attention_k
conda create -n recover_attention python=3.10
conda activate recover_attention
```

If the environment already exists, activate it and run the following commands
from the repository root:

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

Model generation additionally requires PyTorch, Transformers, Accelerate, and
bitsandbytes. Preserve the model, tokenizer, decoding settings, and 8-bit backend
when matching the reported experiment. The pinned CPU analysis packages are
listed in `requirements-paper.txt`.

## Repository contents

`.gitignore` permits the release's source code, tests, configuration, frozen
design files, and selected result summaries. Model/data directories, caches,
temporary files, IDE settings, credentials, paper drafts, TeX build products,
and internal task histories remain local. Current publication checks and their
scope are recorded in `docs/progress/coling_code_release_20261010.md`.
