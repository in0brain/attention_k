# Sprint 4 History - Cyber Hallucination Control

## Sprint 4D-2 W0.5-A/W0.5-B - Preregistration Freeze, Conditional-Increment Core, Model Smoke

Goal: 4D-2 rewrites the main question from "can a probe detect fabrication" to
**conditional increment**: given the strongest output-side baseline O = F5 +
visible text, does the hidden-state probe H add anything? W0.5-A freezes the
design; W0.5-B implements the statistical/schema core and validates it on a
<=20 prompt real model-in-loop smoke. **Stage 0 (2880-trace full generation) was
not started and its execution path is not implemented.**

### W0.5-A - preregistration freeze

`docs/paper/preregistration.md` v2.1 frozen 2026-07-15, sha256 `b3ef9228...`.
Fixes in advance: completion-level population (one completion = one primary
sample, positive iff any eligible identifier is fabricated); O as a single fixed
definition (no max-taking, no test-set selection); equivalence margin eps=0.02;
observability gate delta=0.15 on rank-biserial; K=6 traces/question; hidden
block = floor(0.7L) with tuple index = block+1; TF-IDF and fusion protocol;
RQ2 stratification; startup gates enforced in code.

### W0.5-B - implementation + smoke

New `src/recover_attention/conditional_increment.py` (statistical + schema core,
pure functions, unit-tested) and `scripts/sprint_4D_2_conditional_increment.py`
(stages: preflight / smoke_synthetic / verify_hidden / smoke_model / stage0_gate).

Review-driven P0 fixes made in this sprint (all had teeth):
1. preflight now hard-stops on G2 mismatch; `stage0_allowed = G1 AND G2 AND G3`;
   G3 reads a formal `smoke_report.json` and checks kind/passed/schema_version/
   prereg_sha256/hidden_verification/n_prompts<=20. A synthetic smoke cannot
   grant G3.
2. RQ2 threshold is fold-specific (train-fold median applied to that fold's
   test). The earlier global median was test leakage.
3. `verify_hidden` reuses the 4D-1-validated 8-bit local backend. The prior
   BF16 + `device_map="cuda"` path would OOM on 12GB, and hidden verification
   must run through the same loader as the eventual cache run.
4. emission semantics split: `primary_exclusion_reason` in {no_identifier,
   refusal, only_echoed, only_cve}. only_cve emitted an identifier and is no
   longer an emission failure — counting it as one understates end-to-end
   emission rate.
5. StratifiedGroupKFold + both-class validation per fold + seed retry + stop;
   both bootstraps discard single-class rounds and report n_valid instead of
   scoring them as S=0; `s_observability(NaN)` now raises.
6. H alone added (section 7 requires reporting O / H alone / O+H together).
7. Real output ladder (train-only TF-IDF vocabulary), hidden mean-pooling over
   all eligible identifier tokens, completion cache schema
   `4D2_completion_cache_v1`.
8. Cross-sample F5 (self-consistency / id-agreement) was silently filling 0 —
   it aggregates across a question's K traces. Left unfixed it weakens the O
   baseline and inflates Delta_H.

### v2.2 amendment - three-block equal-weight fusion

The smoke exposed a protocol defect. v2.1 used two blocks (sparse / dense),
putting F5 (d=14) and H (d=3584) in one dense block — contradicting section 7's
own equal-weight motivation. Effect: O+H degenerates toward H alone, so
AUROC(O+H) - AUROC(O) confounds "does hidden add signal" with "did fusion
destroy the output signal". Bumped to v2.2, sha `ffa722e8...`, frozen
2026-07-16. Frozen formula (per outer fold, train statistics only):

```text
text: X~_T = L2Normalize([X_word, X_char])      -> row norm ~ 1
F5:   X~_F = ((X_F - mu)/sigma) / sqrt(d_F)     -> row norm ~ 1, d_F=14
H:    X~_H = ((X_H - mu)/sigma) / sqrt(d_H)     -> row norm ~ 1, d_H=3584
O = [X~_T, X~_F]   H = X~_H   O+H = [X~_T, X~_F, X~_H]
```

Plus a late-fusion robustness appendix (2-D meta-logistic over O_score/H_score,
nested inside each outer fold via inner cross-fitting; not primary).

Amendment recorded as legitimate, not p-hacking: Stage 0 not started; found
within the smoke's remit; the dimension-swamping risk was raised **before** any
smoke AUROC was seen; it fixes an internal inconsistency rather than selecting a
result direction; v2.1's hash and code are preserved at commit `57b7ae9`; smoke
numbers do not enter paper results. v2.1's G3 was voided automatically by the
prereg_sha256 check — no manual invalidation was needed.

### Real smoke results (wiring, not findings)

20 prompts (10 attack / 10 cwe, train/recall split, deterministic seed) x 6
traces = 120 completions. 115 eligible (0.958), 10 positive / 105 negative, 20
groups. Emission failures 5 (only_echoed 4, no_identifier 1). Hidden verified:
L=28, block 19 -> tuple index 20, forward hook equals `hidden_states[20]`,
max_abs_diff 0.00 under the 8-bit backend. Hidden dim 3584. All 11 schema checks
green.

Ladder wiring, v2.1 two-block vs v2.2 three-block:

| rung | v2.1 | v2.2 |
| --- | --- | --- |
| O = F5 + text | 0.843 | 0.801 |
| H alone | 0.785 | 0.785 |
| O+H | 0.721 (O+H < H < O, pathological) | 0.834 (O+H >= O) |

**These are n=115 / n_pos=10 diagnostics, not findings.** They are recorded only
because they exposed the fusion defect and confirmed the fix removed the
mechanical degeneration.

Late fusion reads 0.250 in smoke. Diagnosed rather than waved off: per-fold
AUROC 0.995/0.000/1.000, with meta coefficients going negative (fold1 O=-0.124,
fold2 H=-0.535) because inner-CV base scores are far noisier than the full-train
scores the meta is applied to (fold2's inner H is 0.419, worse than chance).
Scale control shows the implementation is sound: n=90/n_pos=30 -> 0.475;
n=1200/n_pos=402 -> 1.000. Small-sample artifact. Unit tests now separate wiring
from direction, and the smoke report marks the number `interpretable=false`.

Checks: full pytest 790 passed, 2 skipped. Commits `57b7ae9` (v2.1
implementation + its G3 smoke) and `f93877b` (v2.2 amendment).

Gates after this sprint: G1 red (CFP unconfirmed), G2 green, G3 green ->
`stage0_full_generation_allowed=False`.

Boundary: no full generation, no MCQ-side pipeline, no Llama run, no probe
increment conclusion, no hallucination-reduction or accuracy claim. G1 must be
established from external fact and must not be self-asserted by script or agent.

Next: (1) establish G1; (2) implement the Stage 0 execution path — full
generation, the missing MCQ side (required for the section 6 cross-task gate
D = S_MCQ - S_H1), the formal ladder/gate/increment/rq2 path with >=1000-round
bootstrap CIs, and the Llama cross-model check. These two are orthogonal.

## Sprint 4D-1 - H1 Emission/Fabrication Smoke

Goal: run only the preregistered H1 emission/base-rate smoke from the 4D-0
design. The sprint measures whether H1 prompts elicit cybersecurity identifiers
and whether non-echo fabricated identifiers occur at a usable base rate. It does
not compute F5, train a probe, capture activations, continue steering, or make
any hallucination-reduction / answer-accuracy claim.

Implementation:
- Added `build_h1_chat_messages` in `src/recover_attention/h1_data.py`. The
  user message preserves `question_text` exactly; the system wrapper only asks
  for a direct cybersecurity answer with exact CVE/ATT&CK/CWE identifiers.
- Added `scripts/sprint_4D_1_h1_emission_fabrication_smoke.py` for deterministic
  train-split sampling, ontology-manifest validation, local causal-LM generation,
  identifier mention labeling, echo/refusal diagnostics, and the review gate.
- Replaced the initial hand-written KV-cache token loop with
  `transformers.model.generate(use_cache=True, renormalize_logits=True,
  remove_invalid_values=True)` and added a three-question greedy sanity check
  before the full run. The first 4-bit/KV-cache run is invalidated because its
  outputs were garbled. A direct 4-bit `model.generate` sanity probe still
  showed punctuation degeneration on ordinary long-form answers; fp16 was clean
  but too slow for the full smoke, so the valid rerun uses local 8-bit
  quantization with `device_map="auto"`.
- Added `tests/test_h1_emission_smoke.py` for refusal detection, CVE
  high-sequence logic, prompt embedded-id bucketing, gate boundaries, and chat
  message preservation.

Run setup:
- Input dataset: `data/processed/h1/h1_samples.jsonl`.
- Ontology inputs: `data/raw/ontology/{cve,attack,cwe}/ontology_index.jsonl`.
- 4D-0 checks: ontology snapshot manifest sha256 fields matched the current
  indices; id-space density report loaded; train split only.
- Sampling: 72 questions with seed 4242: recall ATT&CK 20, recall CWE 20,
  recall CVE 8, open_gen 8 per family.
- Generation: 1 greedy + 3 sampled completions per question, temperature 0.7,
  max_new_tokens 512, local `D:/models/Qwen2.5-7B-Instruct`, 8-bit local
  generation backend.

Results:
- Total completions: 288.
- Route A greedy emission: 46/48 = 0.958.
- ATT&CK+CWE primary fabrication base rate:
  - mention-level: 65/775 = 0.084.
  - completion-level: 33/222 = 0.149.
- Gate result: `h1_gate_passed=true` under the 4D-0 preregistered thresholds
  (Route A greedy emission >= 0.70 and primary mention-level fabrication in
  [0.05, 0.60]).
- CVE auxiliary view: all CVE fabrication 34/200 = 0.170; high-sequence-only
  11/64 = 0.172.
- Refusal rate: 1/288. No-id completions: 3 low-emission-without-id,
  1 refusal-without-id, 2 answered-without-id.
- Reasoning text: 288/288 completions had non-identifier reasoning text.
- Degeneration/truncation diagnostic after the valid rerun: degeneration
  14/288 = 0.049 and touched max_new_tokens 3/288 = 0.010. Greedy degeneration
  is 2/72 = 0.028, down from the invalid 4-bit/KV-cache run's 70/72 = 0.972.
  The remaining degeneration flags are diagnostic; sampled long open-generation
  traces are the main source.

Outputs:
```text
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/preflight_report.md
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/sampling_manifest.jsonl
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/h1_generation_manifest.jsonl
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/h1_mention_labels.jsonl
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/emission_fabrication_report.json
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/refusal_echo_diagnostic.json
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/review_gate_h1_emission_fabrication_smoke.md
```

Checks:
```bash
conda run -n recover_attention python -m pytest tests/test_h1_emission_smoke.py tests/test_h1_identifier.py -q -p no:cacheprovider --basetemp outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/pytest_target_tmp
conda run -n recover_attention python scripts/sprint_4D_1_h1_emission_fabrication_smoke.py --samples-jsonl data/processed/h1/h1_samples.jsonl --ontology-dir data/raw/ontology --snapshot-manifest outputs/logs/sprint_4D_0_h1_data_design/ontology_snapshot_manifest.json --density-report outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json --num-recall 48 --num-open-gen 24 --samples-per-question 3 --temperature 0.7 --max-new-tokens 512 --seed 4242 --output-dir outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke
conda run -n recover_attention python -m pytest -q -p no:cacheprovider --basetemp outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/pytest_full_tmp
```

Result: targeted H1 tests passed 11/11; full pytest passed 746, skipped 2.

Boundary: gold ids are used only as eval-only source fields from the input
dataset and are not written to prompt/generation/mention outputs. No
`source_entry_id` is written to output manifests. No AUROC, detector, F5,
probe, activation, steering, hallucination-reduction, or accuracy-improvement
claim is made.

Next: Sprint 4D-2 should run full H1 generation + H1-F5 baseline, with an
explicit plan for truncation/degeneration reporting or mitigation.

## Sprint 4D-0 - H1 Fabricated-Identifier Data Design

Goal: create the H1 fabricated-identifier data/label substrate only: ontology
snapshots, identifier extraction/normalization/existence logic, `h1_sample`
schema, deterministic prompt dataset, density audit, H1-F5 design notes, and
4D-1 preregistered smoke gates. No causal LM was called, no completion was
generated, no F5 was computed, no probe was trained, and no steering/patching
was run.

Implementation:
- Added `src/recover_attention/h1_identifier.py` for CVE/ATT&CK/CWE extraction,
  normalization, `OntologyIndex`, status-aware existence lookup, echo exclusion,
  completion-level H1 labels, and H1 gold-leakage checks.
- Added `src/recover_attention/h1_data.py` for deterministic recall/open-gen
  question construction, grouped split, full gold-id residual assertions, dataset
  audit, and id-space density estimates.
- Added `validate_h1_sample_record` and `REQUIRED_FIELDS["h1_sample"]` in
  `schemas.py`, parallel to existing cyber MCQ schema.
- Added the two 4D-0 scripts and targeted tests.

Ontology snapshot results:
- CVEProject/cvelistV5 zip indexed 364903 ids: PUBLISHED 347246, REJECTED 17657.
- MITRE enterprise-attack STIX indexed 858 ids: active 697, revoked 149,
  deprecated 12.
- CWE latest XML indexed 969 ids: Draft 432, Incomplete 486, Stable 26,
  Deprecated 25.
- Manifest records download URLs, snapshot date, source/index sha256, id counts,
  status rules, and the knowledge-cutoff bias note.

Dataset/audit results:
- `h1_samples.jsonl`: 480 prompts, Route A recall 360 and Route B open_gen 120.
- Family counts: ATT&CK 180, CWE 180, CVE 120. ATT&CK/CWE are the primary
  families; CVE is auxiliary because low-number year buckets are dense.
- Grouped split: train/dev/test = 336/72/72, group leakage 0.
- Duplicate normalized questions: 0. Recall gold-id residual check: 360 checked,
  0 violations.
- `h1_f5_design.md` records mention/sequence/sampling/verbalized-confidence
  F5 candidates and the 4D-1 gates: Route A emission rate >= 0.7 and fabrication
  base rate in [0.05, 0.60].

Checks:
```bash
conda run -n recover_attention python scripts/sprint_4D_0_download_ontology_snapshots.py --output-dir data/raw/ontology --manifest-dir outputs/logs/sprint_4D_0_h1_data_design
conda run -n recover_attention python scripts/sprint_4D_0_build_h1_dataset.py --ontology-dir data/raw/ontology --output-path data/processed/h1/h1_samples.jsonl --report-dir outputs/logs/sprint_4D_0_h1_data_design --seed 4242
conda run -n recover_attention python -m pytest tests/test_h1_identifier.py tests/test_h1_data.py -q
conda run -n recover_attention python -m pytest -q
```

Result: targeted H1 tests passed 12/12; full pytest passed 741, skipped 2.

Boundary: this sprint establishes an H1 dataset design. It does not establish
detection performance, emission viability, hallucination reduction, answer
accuracy improvement, or any intervention result.

Next: Sprint 4D-1 should measure only id emission rate and fabrication base rate
on H1 prompts before any internal-feature engineering.

## Sprint 4C - Narrowed Readout Increment and Site Transfer

Goal: test only F1 cross-layer direct projection and F4 exact finite-label VJP J-lens against 4B-3 F5 on CyberMetric MCQ. The only trained component is a grouped-CV linear logistic detector; no controller, steering, LoRA, finetuning, 2000-scale run, or intervention claim is included.

Results:

- 239/240 greedy traces passed dual-form (`bare`) readout assertion and produced gold-free F1/F4 feature records.
- Three-seed grouped-CV F5-alone AUROC: 0.8343 (4B-3 raw reference: 0.8156); F1-alone: 0.5419; F4-alone: 0.4287.
- F5+F1 increment CI95: [-0.0429, 0.0202]; F5+F4: [0.0000, 0.0000]; F5+F1+F4: [-0.0429, 0.0202]. No CI lower bound exceeds zero, so `detector_beats_f5=false`.
- Pair mining used 40 high/low-risk candidates at temperature 1.1 with 12 samples each, reaching 17 merged pairs (from 8). It did not meet the 20-pair gate; site transfer was skipped without reducing pair quality.
- Best F5-only OOF calibration: ECE 0.0463; selective wrong rate is 0.000 at 20% coverage. This is a selective-prediction diagnostic, not an intervention result.

Checks: targeted Sprint 4C/domain/cyber tests passed (56); full pytest passed with workspace-local pytest temporary directory. Feature records pass recursive gold-leakage assertions.

Next: close internal-feature detection for finite-label MCQ and design H1 fabricated-identifier data/labels before considering any gated intervention.

## Sprint 4A - Cybersecurity Direction-Probe Mainline Reset

Goal: reset the project mainline from unsupervised span-guided steering to cyber-domain supervised direction probing. This sprint is documentation, requirements clarification, and code reuse audit only. It does not train a probe, call an LLM, run steering, patch/nudge activations, enter 2000-scale, or continue full Sprint 3C.

Reason for reset:

```text
Sprint 3A / 3B: blind span-level interventions failed to produce selective answer-directed improvement.
Sprint 3C-1 / 3C-2: answer-readout MLP is a valid causal write site, and gold-unembedding direction is effective as an oracle.
Sprint 3C-3 / 3C-4A: gold-free direct readout and approximate J-lens remain weaker than final-logits margin, limiting detection/readout claims.
```

Conclusion:

```text
span relevance is evidence, not a controller.
The missing component is a supervised mapping:
span / activation / trace -> label direction / correction direction.
```

Sprint 4A outputs:

```text
docs/codex_tasks/sprint_4A_cyber_direction_probe_mainline_reset.md
docs/reference/CYBER_DIRECTION_PROBE_PLAN.md
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/reference/STORY.md
docs/progress/sprint_4_artifact_manifest.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/preflight_report.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/review_gate_cyber_direction_probe_mainline_reset.md
```

Boundary flags:

```text
ready_for_2000_rerun=false
do_not_enter_full_sprint_3C=true
hallucination_reduction_proven=false
answer_accuracy_improvement_proven=false
steering_continued=false
domain_supervised_probe_planned=true
```

Next route:

```text
Sprint 4B: Cyber Dataset Selection and Domain Schema Implementation.
Sprint 4C: Cyber Probe Dataset Construction.
Sprint 4D: Probe-Guided MLP Readout Steering.
Sprint 4E: Robustness and Write-up.
```

Test status: `tests/test_dataset_audit.py tests/test_stage_summary.py` passed 14 tests; `tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py` passed 15 tests.

## Post-4A Documentation Guardrail Cleanup - Sprint 4B Ready

Goal: patch the Sprint 4B documentation before execution so the run starts from
dataset/schema/F5 baselines and not from probe training, steering, or Stage 5 by
default. This is a documentation guardrail cleanup, not an experimental result.
No model was called, no data was downloaded, no probe was trained, no steering
was run, and no 4B site-transfer experiment was executed.

Mainline clarification:

```text
CYBER_HALLUCINATION_CONTROL_PLAN.md supersedes CYBER_DIRECTION_PROBE_PLAN.md.
Sprint 4 mainline is domain-calibrated hallucination detection + gated closed-form intervention.
Sprint 4B starts from dataset/schema/F5 baseline, with Stage 5 gated optional.
```

Guardrails added:

```text
parameterized model_path resolution with source reporting;
semantic label id/text retention in candidate_choices;
option_position_bias_report.json requirement;
deterministic option-order and semantic-preservation test requirements;
Stage 5 gate and skipped site_transfer_check_report.json behavior;
review gate questions 18-23.
```

## Merge Guardrail Repair before Sprint 4B Execution

Documentation-only repair after merge conflict resolution. Restored:

```text
- semantic label preservation alongside option-letter readout labels;
- option-position bias audit requirements;
- gated optional Stage 5 site-transfer;
- Review gate items 18-23;
- updated Sprint 4 control plan wording.
```

No experiment was run.

## Sprint 4B Dataset Download and Raw Format Audit

Goal: prepare raw cyber MCQ data sources for Sprint 4B without running the 4B
experiment. This task only downloaded raw/source files, inspected fields and
option/answer formats, and wrote preview/audit artifacts. It did not call a
model, generate completions, train probes, run F5 baselines, do steering, or run
Stage 5 patching.

Outputs:

```text
data/raw/cyber/cybermetric/
data/raw/cyber/secqa/
data/raw/cyber/cs_eval/
outputs/logs/sprint_4B_dataset_download_audit/dataset_source_audit.md
outputs/logs/sprint_4B_dataset_download_audit/raw_file_manifest.json
outputs/logs/sprint_4B_dataset_download_audit/sample_records_preview.jsonl
scripts/sprint_4B_download_and_audit_cyber_datasets.py
```

Raw audit result:

```text
CyberMetric: downloaded 500 / 2000 / 10000 source JSON files; MCQ-like with 4 options and gold solution field; recommended primary source.
SecQA: downloaded v1/v2 dev/val/test CSV files; MCQ-like with 4 options and gold Answer field; small fallback or held-out source.
CS-Eval: downloaded repository metadata/examples/license only; source-inspection-only until a plain MCQ data file is identified.
```

Boundary:

```text
no model call;
no completion generation;
no F5 baseline;
no probe training;
no steering;
no Stage 5 site-transfer or patching.
```

Commands and checks:

```bash
conda run -n recover_attention python scripts/sprint_4B_download_and_audit_cyber_datasets.py --output-dir outputs/logs/sprint_4B_dataset_download_audit
conda run -n recover_attention python -m py_compile scripts/sprint_4B_download_and_audit_cyber_datasets.py
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Check result: audit script completed, syntax check passed, and lightweight
pytest passed 14 tests.

## Sprint 4B CyberMetric Smoke Baseline

Goal: execute the minimal CyberMetric smoke chain without a full primary run, probe training, steering, LoRA/finetuning, full Sprint 3C, or 2000-scale execution.

Setup:
```bash
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py --dataset cybermetric --primary-questions 30 --samples-per-question 2 --temperature 0.7 --max-new-tokens 128 --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke --overwrite
```

Implementation:
- Added `src/recover_attention/cyber_data.py` for CyberMetric raw loading, deterministic option shuffle, canonical MCQ schema, prompt construction, grouped split summary, dataset audit, and option-position bias audit.
- Added `src/recover_attention/domain_label_proxy.py` for option-letter tokenization, answer parsing, label-readout locating, F5 margin/entropy/self-consistency helpers, fixed F5 risk scoring, and gold-feature leakage checks.
- Added `scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py` for the smoke chain. The script uses CPU-side token sampling to avoid CUDA multinomial failures in sampled generation; it does not train a probe.
- Added `tests/test_cyber_data.py` and `tests/test_domain_label_proxy.py`.

Smoke outputs:
```text
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/preflight_report.md
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/dataset_audit_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/label_space_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/option_position_bias_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/cyber_sample_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/trace_sampling_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/f5_baseline_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/correct_wrong_pair_manifest.jsonl
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/site_transfer_check_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/review_gate_cyber_dataset_baseline_site_transfer.md
```

Results:
- Canonical schema: 30 CyberMetric questions.
- Trace sampling: 90 traces (30 greedy + 60 sampled), parse_failure_rate 0.100, sampled wrong_rate 0.315.
- Pair construction: 7 same-question correct/wrong pairs.
- Option-position audit: no severe warning; gold A/B/C/D distribution 6/7/10/7; majority-position baseline accuracy 0.333.
- F5 smoke baseline on 28 scored greedy examples: label-margin risk AUROC/AUPRC 0.741/0.553; label entropy 0.748/0.558; full entropy 0.374/0.244; self-consistency risk 0.823/0.497; fixed non-trained F5 combination 0.776/0.532.
- Stage 5: skipped. Gate failed because `num_questions_with_correct_and_wrong >= 20` was false (7 pairs). `site_transfer_check_report.json` records `status="skipped"` and the skipped reason.
- Rerun verification on 2026-07-09: repeated the same smoke command with `--overwrite`; required artifacts were regenerated locally; Stage 5 remained skipped for the same 7-pair gate failure.

Checks:
```bash
conda run -n recover_attention python -m pytest tests/test_cyber_data.py tests/test_domain_label_proxy.py -q
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
```

Boundary: raw data and smoke outputs remain local/gitignored artifacts. No probe training, steering, nudge, LoRA/finetuning, full Sprint 3C, or 2000-scale run occurred. No hallucination-reduction or accuracy-improvement claim is made.


## Sprint 4B-1 - CyberMetric Canonical Schema and Domain Label Proxy

Goal: establish the data/label interface for the hallucination-control mainline before any model-generation stage. Task card: `docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md`. Strictly no causal LM calls, no completions, no F5, no site-transfer, no probe, no steering; tokenizer-only load of the local Qwen2.5 tokenizer was the sole model asset touched (explicitly allowed by the card).

Implemented:
- `schemas.py`: `cyber_sample` record type + `validate_cyber_sample_record` (field rules for candidate_labels/candidate_choices alignment, gold mapping, label_space, metadata splits).
- `cyber_data.py`: CyberMetric loading/normalization, deterministic per-example option shuffle (sha256-derived per-example seeds; no builtin hash; no in-place mutation), question-hash `group_id` proxy, grouped split, MCQ prompt builder, audits.
- `domain_label_proxy.py`: pure option-letter proxy - `option_token_ids` (whitespace-marker stripping, single-token and collision enforcement), `parse_option_answer` (explicit `Answer: X` > isolated-letter fallback > parse_failure; negative cases like Aes/BERT/CVE/DNS covered), `locate_label_readout_position`, `label_margin`/`label_entropy`/`full_entropy`, `self_consistency_features` (gold-free by construction), recursive gold-leakage checker.
- `scripts/sprint_4B_1_prepare_cybermetric.py`: raw -> canonical -> shuffle -> grouped split -> validation -> manifests -> pre-model audits -> review gate.
- The 4B smoke script was adapted to the new APIs (card compat option 1; compile-checked only).

Results:
- 2000/2000 CyberMetric-2000 records converted, 0 invalid, 0 duplicate example_ids.
- Grouped split train/dev/test = 1400/300/300, group leakage 0 (group_id honestly labeled a question-derived proxy).
- Post-shuffle gold distribution A/B/C/D = 465/503/529/503; fixed-seed reproducible, different-seed order changes verified; gold semantic mapping intact after shuffle.
- Real Qwen2.5 tokenizer check passed: A/B/C/D -> 362/425/356/422, distinct single non-whitespace tokens.
- Smoke manifest: 240 records, split-aware sampling.

Execution bug found and fixed: the tokenizer check was initially skipped with `ValueError: invalid literal for int(): 'input_ids'`. HuggingFace `BatchEncoding` subclasses `UserDict`, not `dict`, so `isinstance(encoded, dict)` in `_tokenizer_ids` fell through and iterated key names as token ids. Fixed via mapping duck-typing; regression test with a UserDict-style fake tokenizer added. Leftover `*.patch`/`tmp_patch/` scratch files (which broke pytest collection) were archived out of the repo root.

Checks: targeted pytest 187 passed; full pytest 708 passed, 2 skipped. All Section-22 entry conditions for Sprint 4B-2 are met.

Boundary: no hallucination-reduction or accuracy claim; gold labels remain eval-only; raw data and outputs stay gitignored with this manifest as the audit record.

Next: Sprint 4B-2 small-model smoke (~32 questions) carrying the three Section-23 carryover items (prompt A/B + degeneration detector; sampling-protection check; F5 cost-tier split in 4B-3).


## Sprint 4B-2 - Small-Model Smoke and F5 Feature Plumbing

Goal: first model-generation stage on canonical CyberMetric data. Task card: `docs/codex_tasks/sprint_4B_2_small_model_smoke_and_f5_plumbing.md`. 32 questions (train split, seed 4242, not first-N), two prompt conditions (raw completion vs chat template), 1 greedy + 3 samples each, max_new_tokens 256. No probe training, no steering, no patching, no kill-bar claim; gold labels eval-only.

Implemented:
- `domain_label_proxy.detect_degeneration`: three rules (single-char run >= 30; substring length 6-40 looping >= 5x; truncated_failure = parse failure at >= 0.9 generation budget, exact token count preferred over char heuristic), with REAL degenerate completions from the 4B smoke harvested verbatim (repr round-trip) as test fixtures, including two documented negatives (below-budget multilingual gibberish that no rule fires on).
- `domain_label_proxy.bare_option_token_ids` + shared `_resolve_option_token_ids`: bare-letter (no leading space) token form with the same single-token / pairwise-distinct / whitespace-stripping validation as the space form.
- `cyber_data.build_mcq_chat_messages` (system + user; user body identical to the raw prompt so the A/B isolates the wrapper).
- `scripts/sprint_4B_2_small_model_smoke.py`: preflight (worktree clean, processed-dataset 20-record schema spot-check, model-path resolution), A/B run, decision rule (score = parse_failure + degeneration; lower wins; tie < 0.02 -> chat; both > 0.15 -> stop), tokenizer-only readout-position assertion over EVERY trace with dual-form matching, F5 feature pass with cost-tier fields and gold-leakage checks, option-position model-bias report, review gate.

Two instrumentation bugs found by tiny dry runs and fixed before the official run:
1. `locate_label_readout_position` raised "tokenizer must provide offset_mapping" with real tokenizers: `BatchEncoding` is a `UserDict`, not a `dict` subclass — the same root cause as the 4B-1 `_tokenizer_ids` fix, in a second call site. Fixed by duck-typing on `.get`; chat-template-style locate test added.
2. Bare-letter token form: chat completions emit the bare option letter (never restating "Answer: "), which tokenizes to a different id than the space-prefixed form (`" D"` -> 422 vs `"D"` -> 35). The F5 readout assertion initially failed 100% on chat; without the fix every chat-condition margin/entropy would have silently used wrong candidate token ids. Dual-form resolution added, with `token_form_counts` reported instead of papering over.

Official run results (128 traces per condition):
- chat: parse_failure 0.0, degeneration 0.0, score 0.0, correct_rate 0.875; readout assertion 128/128 pass, token form 128/128 bare.
- raw: parse_failure 0.094, degeneration 0.094, score 0.1875, correct_rate 0.734; assertion 111/116 pass (93 space / 18 bare / 5 unknown).
- Decision: chat wins (not a tie); clean 4B-3 admission (0.0 <= 0.08).
- F5 plumbing: all five features finite, zero missing, tier fields in place, leakage checks pass; smoke AUROC 0.99 (n=32, 4 wrong) marked plumbing_validation_only.
- Position bias: no severe warning.
- Reasoning substrate (key finding for 4C): chat completions are 128/128 single-token bare letters; has_reasoning_text = 0.00. F2 trajectory features have no substrate under the winning condition; the 4B-3 card carries a gated reasoning-forcing mini-test (Stage B') and requires an explicit F2 plan.

Checks: targeted pytest 51 passed; full pytest 723 passed, 2 skipped.

Boundary: no kill bar declared; no accuracy/hallucination claim; outputs gitignored under `outputs/logs/sprint_4B_2_small_model_smoke/` with this manifest as audit record.

Next: Sprint 4B-3 full 240-question F5 dual-kill-bar run + site-transfer check (card drafted).

## Sprint 4B-3 Full F5 Baseline and Site-Transfer Diagnostic

- Output dir: `outputs\logs\sprint_4B_3_full_f5_baseline_and_site_transfer`
- kill_bar_single_forward: `{"score_name": "f5_combo_single_forward_z", "num_examples": 239, "num_positive_wrong": 47, "auroc": 0.8156028368794326, "auroc_ci95": [0.7462170655064057, 0.871379088089075], "auprc": 0.48176965150975914, "auprc_ci95": [0.3342258393392229, 0.6492906000587156]}`
- kill_bar_sampling: `{"score_name": "f5_combo_sampling_z", "num_examples": 239, "num_positive_wrong": 47, "auroc": 0.815270390070922, "auroc_ci95": [0.7620344115039633, 0.8715600575894694], "auprc": 0.48279740626091816, "auprc_ci95": [0.3531971769514906, 0.64021080732525]}`
- reasoning F2 plan: `downgrade_or_drop_f2_and_prioritize_f3_plus_f1_f4`
- site-transfer conclusion: `not_evaluated_gate_skipped`
- Boundary: no probe training, no steering, no hallucination/accuracy improvement claim.
