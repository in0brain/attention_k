# Paper code release status

## 2026-10-10

- Prepared the COLING manuscript code on `codex/coling-code-release`, using the
  local research implementation and frozen result summaries.
- Added `PAPER_CODE.md`, CPU dependencies, a figure-rendering entry point, a
  publication allowlist, and a file-hash manifest.
- Fixed JSON serialization in the synthetic CLI and separated tests requiring
  unbundled local artifacts or the historical pre-analysis workspace.
- Inputs: existing source, tests, configuration, frozen protocols, and eight
  saved result/configuration summaries. No new model execution or formal
  statistical analysis was performed.
- Outputs and commands: see `PAPER_CODE.md` and
  `docs/progress/coling_code_release_20261010.md`.
- Validation: 2,194 checks passed across the full suite and targeted recheck;
  23 skipped. The synthetic CLI, saved-result figure rendering, and clean-export
  checks also passed. Details are recorded in the release audit.
- Remaining scope: full experiment reruns require the raw data, model weights,
  record-level caches, and matching freeze manifests described in the guide.
- Next step: use the published release branch when citing the paper's code.
