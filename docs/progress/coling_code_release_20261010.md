# COLING paper code release — 2026-10-10

## Scope

The release branch is `codex/coling-code-release` in
<https://github.com/in0brain/attention_k>. It starts from the existing remote
`master` commit `17e0ca1e97652637639158b89239903f672f0f3e` and contains the current
paper implementation, including the later statement-representation and MCQ
output-coverage comparisons.

At the user's request, the README now describes the current paper and its
reproduction workflow; `PAPER_CODE.md` points to that guide. Repository agent
instructions are preserved. The implementation, tests, configuration, frozen
design files, and selected result summaries are included; raw datasets, model
weights, tensor caches, private settings, paper drafts, and internal task
histories are excluded. Previously tracked unrelated paths are removed from
the release index only; local copies remain on disk.

## Source changes

- Updated README to describe the completed paper experiments, setup, entry
  points, results, and reproduction inputs. The user authorized publishing to
  the public repository and merging the release into `master`.
- Added the CPU analysis dependencies and a pinned environment file.
- Reused the existing NumPy normalization function in the synthetic CLI's JSON
  output. ASCII escaping makes this output work in Windows legacy-codepage
  terminals. Statistical computation and production result writing are unchanged.
- Added a regression test for nested NumPy values and Unicode JSON output.
- Marked integration tests that depend on unbundled local files and historical
  checks of the pre-unblinding workspace. Their assertions remain intact for
  applicable environments. Gate and boundary unit tests use isolated fixtures.
- Copied the existing English figure renderer to a public script entry point.
- Kept all published result summaries and frozen protocols byte-identical to
  the local source artifacts. `release_manifest.jsonl` hashes the Git file
  contents of the release, excluding the manifest itself.

## Validation commands

```bash
python -m pytest -q
python scripts/run_stage1_v3_confirmatory.py --arm both --dry-run
python scripts/render_paper_figures_en.py --project-root . --output-dir outputs/paper_figures
git -c core.whitespace=cr-at-eol diff --cached --check
```

Checks ran in the existing `recover_attention` Python 3.10 environment with GPU
visibility disabled and model libraries in offline mode. Only synthetic inputs
were used for analysis checks. The renderer read existing summaries.

- Full suite: 2,217 checks collected; 2,193 passed and 10 skipped on the first
  complete run. Fourteen failures concerned missing local artifacts or obsolete
  workspace-state assumptions. After the test portability changes, those
  fourteen checks were rerun: one passed and thirteen skipped. No failing check
  remains; together these runs account for 2,194 passing and 23 skipped checks.
- Clean-export checks: 178 passed and one skipped for schemas, preflight,
  figure inputs, and the synthetic CLI regression; interface consistency added
  37 passed and two skipped. The export contained only publication files.
- The two-arm synthetic CLI completed and emitted valid JSON.
- All three statistical figures rendered from the frozen summaries; the
  dependency plot was visually inspected.
- Python syntax, selected JSON files, ignore rules, credential-pattern scans,
  and the staged diff check passed. All thirteen frozen protocol/result/manifest
  files retained their original byte hashes in Git.

## Reproduction scope

The release supports source inspection, CPU unit/synthetic checks, and figure
reproduction from the saved summaries. Full reruns require the original types
of raw inputs and a new matching set of freeze manifests. Historical source
fingerprints in the archived summaries identify the original experiment, not
this packaging revision. See `PAPER_CODE.md` for input and backend details.

Paper footnotes and layout checks are recorded separately in the local
`docs/paper/notes/revision_log.md`; manuscript files are not part of this code
release.
