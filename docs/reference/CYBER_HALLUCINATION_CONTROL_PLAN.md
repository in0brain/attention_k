# Cyber Hallucination Control Plan

> Supersedes `CYBER_DIRECTION_PROBE_PLAN.md` (kept for history). The mainline is
> refined from "supervised direction probe / controller" to "domain-calibrated
> hallucination detection with gated, closed-form intervention". The refinement
> follows a literature review (Sun et al. 2026 trajectories; INSIDE; LM-Polygraph;
> causal-tracing hallucination papers; Anthropic Workspace/J-lens and NLA) plus
> the Sprint 3C-3 / 3C-4A negative results.

## 1. Research Question

```text
Can internal, mechanistically-localized signals — read at anchor-free reasoning
positions in freeform traces — detect cyber-domain hallucination with measurable
incremental value over output-level baselines, and can that detection gate
low-risk interventions that reduce confident wrong answers at fixed coverage?
```

Three sub-questions, each independently falsifiable:

```text
Q1 (features): do trajectory-transition and causal-site features add AUROC over
    final-logits margin / entropy / self-consistency?
Q2 (readout method): do exact J-lens label projections read mid-layer states
    better than direct logit-lens projections?
Q3 (control): does detection-gated intervention reduce wrong-answer rate at
    fixed coverage without breaking correct answers (net fix/break accounting)?
```

## 2. Evidence Chain

Why this mainline, from our own sprints and the literature:

```text
1. 3A/3B/3C: blind and instance-level answer steering is structurally blocked —
   the effective direction requires the gold answer (3C-2).
2. 3C-1: the answer-readout MLP is a causally verified site (donor- and
   site-specific, low-harm regime exists).
3. 3C-3: static single-position site features lost to final-logits margin
   (AUROC 0.653 vs logits stronger by 0.059).
4. Sun et al. 2026: trajectory *transition* features beat LogitLens baselines
   (0.852 vs 0.765) — the feature family we never tested. Late-stage divergence
   matches our readout-position finding. BUT their correctness prediction
   collapses on freeform traces (AUC 0.60) because it needs "Step k:" scaffolds,
   and their own gated-intervention accounting (26 fixed / 14 broken of 90
   flagged) shows detection >> correction.
5. INSIDE: multi-sample internal consistency (EigenScore) is a strong internal
   detection family; never tested in our 3C-3.
6. LM-Polygraph: many UQ methods fail to beat simple baselines — hence the
   incremental-over-baseline kill gate.
7. Causal Tracing of Object Representations (AAAI'26) and Beyond Transcription
   (ASR, AAAI'26): "causal localization -> site-informed detection/interception"
   is an emerging cross-modal paradigm; our 3C-1 site is its text-reasoning
   instance.
```

The differentiators vs Sun et al.:

```text
A. Anchor-free: our answer-slot / role-position machinery (3C-0-Fix) locates
   trajectory nodes in freeform text — exactly where their method collapses.
B. Site-informed: our nodes are causally verified (3C-1), not just last-layer
   heuristics; we test whether mechanism-guided feature selection beats blind
   layer choice.
C. Domain calibration: their transfer collapse (0.87 -> 0.60/0.64) proves
   correctness geometry must be re-calibrated per domain — which is precisely
   the cyber dataset work.
```

## 3. Operational Hallucination Definition (Cyber)

Hallucination is operationalized as three measurable classes:

```text
H1 fabricated identifier: emitted CVE / ATT&CK / CWE id that does not exist in
   the public ontology (verifiable by construction).
H2 wrong mapping: real identifiers, wrong association (gold label available).
H3 confident wrong classification: finite-label task, wrong answer with high
   expressed confidence.
```

Reduction claims are made only in selective-prediction terms:

```text
wrong-answer rate at fixed coverage (risk-coverage / AURC)
fabricated-identifier rate under constrained vs unconstrained decoding
net fix/break accounting for any gated intervention
```

Never as unqualified "hallucination reduced".

## 4. Architecture: Three Layers

```text
Layer 1  DETECT   feature bake-off -> risk score        (primary deliverable)
Layer 2  DECIDE   per-task-family calibration -> selective prediction /
                  conformal abstention
Layer 3  INTERVENE (optional, gated) closed-form actions on flagged cases only
```

Layer 3 is unlocked only if Layer 1 passes its gate. Allowed interventions are
closed-form (no learned vectors): reflection-token injection (Sun-style gated),
constrained re-decode against the label space / ontology, retry, escalate.
Forbidden: trained vector controllers, instance-level answer steering,
fine-tuning, LoRA.

## 5. Feature Families and the Kill Gate

Five families compete in one bake-off, grouped-by-question CV:

```text
F1 static site features (3C-3): readout-position MLP projection margin /
   entropy / layer agreement at L20/L24.
F2 anchor-free trajectory transitions (Sun, adapted): activation differences
   between semantic nodes located by our role tagging (intermediate-result
   numbers, result markers, answer slot) — no "Step k:" scaffold required.
F3 multi-sample internal consistency (INSIDE): EigenScore-style covariance of
   mid-layer embeddings across K sampled answers.
F4 exact J-lens label projections (Section 6).
F5 output-level baselines: final-logits margin, entropy, verbalized confidence,
   self-consistency vote — the kill baseline, always reported.
```

Kill gate (from LM-Polygraph discipline and the 3C-3 lesson):

```text
any internal family (F1-F4) must show INCREMENTAL AUROC/AUPRC over F5 alone,
grouped CV, seed-averaged, CI-reported. No increment -> that family is dropped.
All families fail -> the deliverable is the F5-based selective-prediction
system, honestly framed (still a valid hallucination-control result).
```

## 6. J-lens Integration (Exact, Finite-Label)

Background: true J-lens reads an intermediate activation via the Jacobian of
final logits w.r.t. that activation — infeasible for a full vocabulary
(d_vocab backward passes). Sprint 3C-4A therefore used a finite-difference
approximation and found weak alignment with direct logit-lens projection
(top-1 match 0.074, Case C), leaving an unresolved question: is the
approximation crude, or does direct projection genuinely misread mid-layer
states?

The finite label space dissolves the cost problem:

```text
detection only needs projections onto k candidate labels (MCQ options /
label set, k = 2..10);
grad of logit[label_token] w.r.t. h(layer, position) = one VJP each;
=> exact J-lens label projections in k backward passes per (position, layer).
No epsilon, no finite-difference noise.
```

And the bake-off arbitrates 3C-4A's open question with an application-grounded
criterion:

```text
if F4 (J-lens) adds detection value where F1 (direct projection) does not,
direct logit-lens was misreading mid-layer states — a publishable methods
finding ("readout method matters for hallucination detection");
if F1 == F4, the cheap readout suffices and 3C-4A's discrepancy was
approximation noise.
```

## 7. Workspace / NLA Integration (Optional, Qualitative)

```text
Workspace theory: motivates WHY the answer-readout bottleneck is the right
place to read (privileged / verbalizable representations); framing only, no
claim depends on it.

NLA (kitft/nla-qwen2.5-7b-L20-av/ar): verbalize activations of FLAGGED
high-risk cases as qualitative case studies. Constraints: L20 checkpoint only
(our strongest signal is L24 — expectation: NLA may not see it); verbalizer
trained on residual-stream activations — feeding MLP-output vectors may be
out-of-distribution; run a sanity check first; never primary evidence.
```

## 8. Dataset and Label Space

Requirements carried over from the superseded plan, with one hard lesson
enforced:

```text
1. one finite-label cyber task first (MCQ or classification), option letters
   (A/B/C/D) or single-token labels as the primary label space;
2. do NOT use raw ATT&CK technique ids as unembedding targets — "T1059.001"
   style ids share the leading token ("T") and reproduce the 3C-0
   leading-token collision bug; map to option letters instead;
3. evidence spans optional but preserved in the schema (reuse the superseded
   plan's canonical sample schema);
4. grouped train/dev/test split by source/question family;
5. gold labels: training targets and eval labels only, never inference features;
6. enough examples for seed-averaged AUROC with CIs (hundreds+, not 34 pairs —
   the probe is single-forward per trace, so scaling is cheap and justified).
```

Option letters are token-level readout targets only. Original cyber semantic
labels must be retained in `candidate_choices` as `label_id` and `label_text`
for semantic analysis, transfer evaluation, error taxonomy, and case studies.
Sprint 4 must not collapse into an option-letter-only task.

For H1 (fabricated identifiers): a free-generation slice where emitted ids are
verified against the public ontology; internal-feature hypothesis to test:
fabricated ids show diffuse readout projections (low margin, high entropy,
low layer agreement) versus grounded ids.

## 9. Anchor-Free Trajectory Nodes

Reuse and extend existing machinery (code reuse from Sprint 3/3C):

```text
node types: prompt end, intermediate-result numbers, equals/result markers,
final-answer slot (from activation_patching.extract_reasoning_step_positions +
answer_proxy_metrics answer-slot locating);
features: between-node activation deltas (Euclidean + cosine), node->answer
transition, cumulative drift — Sun's feature recipe on our anchor-free nodes;
validation: probe accuracy on shuffled node labels must drop to chance
(their formatting-artifact control).
```

## 10. Evaluation and Gates

```text
Gate 1 (4C): incremental AUROC/AUPRC of F1-F4 over F5, grouped CV, 3 seeds.
Gate 2 (4C): calibration — ECE and risk-coverage curves per task family.
Gate 3 (4D, only if Gate 1 passed): gated intervention net accounting —
  fixed vs broken counts (Sun-style honesty), wrong-answer rate at fixed
  coverage, harm proxy on intervened cases.
Robustness (4E): held-out cyber task family; anchor-free nodes on freeform
  traces vs scaffolded prompts; cross-layer stability.
```

## 11. Boundaries

```text
frozen model only; linear probes and temperature/conformal calibration are the
only trained components;
no vector-valued controller; no instance-level answer steering; no LoRA /
finetuning; no 2000-scale rerun; no full Sprint 3C;
gold labels / ontology: training + evaluation + constrained decoding only;
claims only in selective-prediction and fix/break terms;
"hallucination reduced" / "accuracy improved" are forbidden as unqualified
claims.
```

## 12. Sprint Route

```text
4B  dataset + operational definition + baselines
    select 1 finite-label cyber dataset (+ 1 free-generation slice for H1);
    implement cyber_data.py + domain answer/label proxy (adapt
    answer_proxy_metrics); option-letter label space;
    sample traces (frozen model), build correct/wrong sets;
    report F5 baseline suite performance. No probes yet.
    Cheap add-on: replicate the 3C-1 site finding on the label-readout position
    (module tracing, no training) — if the causal site does not transfer
    beyond GSM8K, F1's motivation weakens and we know early.
    4B must include option-position bias auditing and gated site-transfer
    execution. The 3C-1 site-transfer add-on runs only if parsing, wrong-rate,
    pair-count, tokenization, and option-position gates pass; otherwise it is
    skipped with an explicit skipped_reason.

4C  feature bake-off (Gate 1 + 2)
    F1-F4 vs F5, grouped CV, seed-averaged;
    J-lens vs direct-projection arbitration (Section 6);
    calibration + conformal wrapper.

4D  gated control (only if Gate 1 passes)
    risk-gated reflection injection / constrained re-decode / abstain;
    net fix/break accounting; wrong-answer rate at fixed coverage.

4E  robustness + write-up
    held-out task family; ablations (anchor-free vs scaffold; site vs blind
    layer); failure cases; paper narrative (Sprint 2-3 negatives as motivation).
```

## 13. What This Plan Does Not Claim

Sprint 4A/4A-R prove nothing about the cyber probe working. Sun et al.'s own
corrections bound expectations: correctness geometry is task-specific,
detection outruns correction, unconditional intervention is harmful. Every
positive claim in Sprint 4 must clear its gate first.

## 14. Sprint 4B Findings → 4C Narrowing (2026-07, empirical update)

Sprint 4B (4B-1 schema/proxy, 4B-2 prompt A/B, 4B-3 full run) executed the
DETECT-layer groundwork on CyberMetric. The measured F5 baseline and the task's
answer dynamics **collapse the five-family bake-off design of Section 5** for the
MCQ-option-letter task. This section records the empirical facts and narrows 4C;
Section 5's five families remain the reference framework for the H1 task (below).

### 14.1 What Sprint 4B measured

```text
Winning prompt = chat template (4B-2: chat score 0.0 vs raw 0.1875; chat wins).
240-question run (4B-3):
  correct_rate 0.802 / wrong_rate 0.195 / parse_failure 0.0036 / degeneration 0.0.
  F5 kill bars (grouped bootstrap CI95):
    kill_bar_single_forward = AUROC 0.816 [0.746, 0.871]  (label_entropy strongest)
    kill_bar_sampling       = AUROC 0.815 [0.762, 0.872]
```

### 14.2 Structural consequences (why the design collapses)

```text
A. Chat completions are single-token bare letters (has_reasoning_text = 0.00;
   4B-3 reasoning-forcing variant Stage B' failed: clean_score 0.406, 31%
   parse failure). => F2 (trajectory transitions) has NO substrate. Dropped.
B. Answers are near-deterministic at temp 0.7 (option-token margins 9-18 logits).
   => sampling adds zero over single-forward (bars 0.815 == 0.816). F3
   (multi-sample consistency / EigenScore) is near-null; F5 sampling tier gives
   no increment. The real gate is 0.816 (single-forward).
C. Near-determinism yields only 8 correct/wrong pairs (< 20 gate). => 3C-1
   site-transfer validation (F1's causal motivation) could not run. Q3 open.
D. The 0.816 bar comes from label_entropy — a CALIBRATION signal. On clean MCQ,
   wrong ≈ uncertain ≈ high entropy, and the output layer calibrates well
   (the LM-Polygraph lesson). This is narrower than the mainline's stated goal
   (H1 fabricated-ID / H3 confident-wrong), where output calibration is known
   to fail — precisely the samples clean MCQ excludes.
```

### 14.3 Narrowed 4C (the surviving question)

```text
4C is NOT the five-family bake-off. On the MCQ task only two families survive:
  F1 (cross-layer L20/L24 readout projection onto the k option tokens, incl.
      layer-disagreement) and F4 (exact J-lens on the k option tokens).
Sharp testable claim:
  does cross-layer / J-lens readout disagreement provide uncertainty signal
  BEYOND final-layer entropy (0.816)?  This is a real mechanism (mid-layer
  conflict the final layer washes out; logit-lens-across-layers), cheap
  (one-position activation capture over 240 greedy), and is the fair test of
  differentiation B (site-informed). Prior (3C-3) lost to logits margin by
  0.059 on GSM8K, so a negative is plausible and clean.
4C Stage 0 folds in the deferred Q3: mine >= 20 correct/wrong pairs by
  higher-temperature resampling on low-margin / low-self-consistency questions,
  then run the 3C-1 site-transfer check (kills two birds).
Outcome contract (unchanged from Section 5 kill gate):
  F1/F4 beat 0.816 with grouped CV + CIs  -> site-informed readout is real;
  neither beats 0.816                     -> close internal-feature detection
                                             for finite-label cyber MCQ; the
                                             deliverable is the F5 selective-
                                             prediction system, honestly framed.
```

### 14.3A Sprint 4C result update

```text
Measured on 239 CyberMetric greedy MCQ records:
  F5-only grouped-CV logistic AUROC = 0.8343;
  F1-alone = 0.5419; F4-alone = 0.4287;
  F5+F1 increment CI95 = [-0.0429, 0.0202];
  F5+F4 increment CI95 = [0.0000, 0.0000];
  F5+F1+F4 increment CI95 = [-0.0429, 0.0202].
No combination passes the lower-CI > 0 kill gate, so
`detector_beats_f5=false` and finite-label MCQ internal-feature detection is
closed. Pair mining reached 17, not the required 20, so cyber site transfer is
honestly recorded as skipped rather than weakened. This is not a hallucination
reduction or answer-accuracy result.
```

### 14.4 The MCQ task is not the mainline's real vehicle (H1 next)

```text
Regardless of 4C's outcome, "MCQ wrong-vs-correct detection" is a calibration
task, not a hallucination task. The mainline's full hypothesis (F2 trajectory +
F3 consistency + site-informed F1 beating output calibration) needs a task
where reasoning exists, sampling is diverse, and output calibration is poor —
i.e. the H1 fabricated-identifier setting of Section 3 (free-generation cyber
Q&A with ontology-verifiable CVE/CWE/ATT&CK answers; hallucination = fabricated
id, no inference-time gold needed for the fabrication check). MCQ 4B/4C built
the infrastructure (schema, tokenization discipline, F5 reference bar 0.816) and
serve as the "easy calibrated task" reference point. H1 is the eventual DETECT
vehicle. Recommended order: run narrowed 4C first (cheap, closes the MCQ door
cleanly), then design the H1 task; do not run the full five-family bake-off on
MCQ.
```
