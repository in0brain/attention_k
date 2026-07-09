# Cyber Direction-Probe Plan

> SUPERSEDED (2026-07-09) by `CYBER_HALLUCINATION_CONTROL_PLAN.md`.
> After a literature review (Sun et al. 2026 trajectories, INSIDE, LM-Polygraph,
> causal-tracing hallucination papers, Anthropic Workspace/J-lens and NLA), the
> mainline was refined from "supervised direction probe / controller" to
> "domain-calibrated hallucination detection with gated, closed-form
> intervention". Key reasons: (a) instance-level answer steering is structurally
> blocked (3C-2: the effective direction requires the gold answer); (b) the
> trained vector-controller route (Sections 8C/14) drifts toward
> fine-tuning-by-another-name; (c) Sun et al. show transition features can beat
> logits baselines for detection, fixing the 3C-3 failure mode, while their own
> results confirm detection >> correction. This file is kept for history; the
> sample schema (Section 4) and dataset requirements (Section 3) remain valid
> and are referenced by the successor plan.

## 1. New Research Question

Sprint 4 changes the main research question from:

```text
Can important spans be directly emphasized to make the model answer correctly?
```

to:

```text
Can cyber-domain supervision train a probe/controller that maps reasoning-aware signals to a label or correction direction?
```

The core hypothesis is:

```text
span / attention / activation / readout features are useful evidence,
but they need supervised mapping to become a reliable direction.
```

This does not erase Sprint 3. Sprint 3 shows why blind span-guided steering is too weak, while preserving the answer-readout MLP as a valid causal site.

## 2. Why Cybersecurity Instead of GSM8K

GSM8K is useful for early reasoning diagnostics, but its open numeric answer space makes direction targets unstable:

```text
arbitrary numbers
multi-token numeric answers
format-sensitive final answer strings
weak reusable label semantics
```

Cybersecurity tasks often have structured label spaces:

```text
MITRE ATT&CK technique ids
CWE ids
attack stages
malware / benign
true positive / false positive
mitigation classes
multiple-choice options
```

These labels make it easier to define a direction such as:

```text
gold label direction - wrong label direction
correct MLP readout - wrong MLP readout
```

The initial cyber task should be classification, multiple choice, or finite label-space prediction. Open-ended cyber QA should not be the first Sprint 4 task because it recreates the unstable answer-space problem.

## 3. Dataset Requirements

The preferred Sprint 4B dataset has:

```text
1. cyber-domain input text;
2. finite candidate labels;
3. one gold label per example;
4. enough examples for train/dev/test split;
5. labels with domain semantics;
6. labels that can be tokenized and mapped to unembedding rows;
7. metadata that supports grouped evaluation by source or question family.
```

Priority task types:

```text
alert / log classification
attack technique identification
MITRE ATT&CK technique mapping
CWE / vulnerability type classification
malicious command vs benign command classification
false positive vs true positive triage
attack-chain stage identification
mitigation selection
multiple-choice security QA
```

## 4. Canonical Sample Schema

```json
{
  "example_id": "string",
  "task_type": "attack_technique_classification | cwe_classification | alert_triage | multiple_choice_qa | ...",
  "input_text": "string",
  "question": "string | null",
  "candidate_labels": ["string"],
  "gold_label": "string",
  "gold_label_id": "string",
  "label_space": "string",
  "evidence_spans": [
    {
      "text": "string",
      "char_start": 0,
      "char_end": 0,
      "role": "indicator | condition | vulnerability | action | tool | target | mitigation | other"
    }
  ],
  "metadata": {
    "source": "string",
    "split": "train | dev | test"
  }
}
```

Gold labels are allowed as training targets and evaluation labels. They are not allowed as inference features.

## 5. Label-Space Requirements

A valid label space must satisfy:

```text
1. every sample has a gold label;
2. wrong labels can be sampled or enumerated;
3. labels can be tokenized;
4. labels can be mapped to unembedding directions;
5. labels have cyber-domain semantics;
6. labels support train/dev/test split without leakage;
7. label frequency is sufficient for evaluation beyond majority baseline.
```

Preferred labels:

```text
MITRE technique id
CWE id
attack stage
malware / benign
true positive / false positive
mitigation class
multiple-choice option
```

## 6. Correct/Wrong Pair Construction

Sprint 4 should support two pair types.

### A. Label Contrast Pair

```text
same input
gold label
sampled wrong label
```

This is the first recommended pair type because it is deterministic, cheap, and tied to a finite label space.

### B. Trace Contrast Pair

```text
same input
model correct trace
model wrong trace
```

This is useful later, but it requires model generation and trace capture. It should wait until the dataset schema and label-space pipeline are stable.

## 7. Probe Inputs

Candidate input features:

```text
final-answer / label-readout MLP output
L20 / L24 MLP output
attention-output / residual-output comparison
direct unembedding projection
final logits margin
span features
evidence span attention mass
layer agreement
label-space projection margin
model generated answer agreement
```

Leakage rule:

```text
gold_label cannot be an inference feature.
gold_label can only define a training target or evaluation label.
```

## 8. Probe Outputs

Sprint 4 should consider three output forms:

```text
A. Detection probe:
   answer risk / label correctness probability.

B. Direction-selection probe:
   candidate label id or direction id.

C. Vector-valued controller:
   continuous steering vector delta_mlp.
```

Recommended priority:

```text
1. Direction-selection probe;
2. Detection probe;
3. Vector-valued controller.
```

Reason: direction selection is closer to steering than detection, but more stable and more auditable than directly predicting a continuous vector.

## 9. Direction Definitions

At least two direction families should be supported.

### A. Label-Unembedding Direction

```text
W_U[gold_label] - W_U[current_or_wrong_label]
```

For multi-token labels, record one of these strategies:

```text
first non-whitespace token
average label token unembedding
full label sequence logprob direction
multiple-choice option token direction
```

### B. Correct-Wrong MLP Delta

```text
MLP_output_correct - MLP_output_wrong
```

This reuses the Sprint 3C-2 insight, but replaces GSM8K numeric answers with structured cyber labels.

## 10. Training Targets

Sprint 4B should not train. Later sprints may define:

```text
detection target: is current label/answer correct
direction-selection target: gold label id or gold-vs-wrong direction id
vector target: correct-wrong MLP delta
ranking target: gold label margin > wrong label margin
```

All targets must keep gold labels out of inference features.

## 11. Evaluation

Primary future metrics:

```text
classification accuracy
AUROC / AUPRC
ECE / calibration
label logprob improvement
gold-vs-wrong margin improvement
harm rate
wrong-direction amplification rate
generation accuracy
abstention / retry utility
```

Sprint 4 final success cannot be "probe beats random" only. It must answer:

```text
1. does the probe beat final logits / surface baselines?
2. does it select better directions than random?
3. does low-alpha MLP steering improve gold label logprob?
4. does it lower wrong label logprob?
5. is harm controlled?
6. does it generalize to held-out cyber tasks?
```

## 12. Baselines

Future Sprint 4C/4D baselines:

```text
final logits baseline
no-steer baseline
random direction
label-frequency baseline
surface heuristic baseline
detection-only probe
direction-selection probe
optional vector controller
```

## 13. Risk Boundaries

Sprint 4A proves none of the following:

```text
cyber probe works
steering solved
answer accuracy improved
hallucination reduced
ready for 2000
```

Sprint 4A only sets the next mainline and requirements.

Safety and rigor boundaries:

```text
do not train on gold labels and evaluate on leaked groups;
do not use gold labels as inference features;
do not choose a label space with too few minority samples;
do not report weak-labeled success as human-reviewed validation;
do not deploy open-ended cyber generation before finite-label diagnostics work.
```

## 14. Suggested Sprint 4 Modules

Sprint 4A does not implement these modules. It only records the planned responsibilities.

### `src/recover_attention/cyber_data.py`

```text
read cybersecurity datasets;
standardize labels;
construct prompts;
construct candidate label sets;
split train/dev/test;
write cyber manifest.
```

### `src/recover_attention/domain_answer_proxy.py`

```text
parse label / option / structured answer;
locate label-readout position;
compute label sequence logprob;
support multi-token labels;
replace GSM8K numeric answer parser.
```

### `src/recover_attention/domain_direction_probe.py`

```text
construct probe dataset;
train detection / direction-selection / vector probe;
prevent gold leakage;
perform question/source grouped split;
write AUROC / accuracy / calibration reports.
```

### `src/recover_attention/probe_guided_steering.py`

```text
read probe-selected direction;
apply low-alpha answer/label-readout MLP nudge;
evaluate gold label logprob, wrong label logprob, and harm;
avoid unsupervised blind steering.
```

## 15. Sprint 4 Route

### Sprint 4B: Cyber Dataset Selection and Domain Schema Implementation

```text
select 1-2 cybersecurity datasets;
implement cyber_data.py;
implement an initial domain_answer_proxy.py;
construct label-space manifest;
do not train a probe.
```

### Sprint 4C: Cyber Probe Dataset Construction

```text
run frozen LLM;
capture MLP readout / attention / residual features;
construct correct/wrong label contrast;
write probe dataset;
train minimal detection / direction-selection probe.
```

### Sprint 4D: Probe-Guided MLP Readout Steering

```text
use probe to select direction;
apply low-alpha MLP readout nudge;
compare no-steer / random / final logits / probe-guided;
evaluate gold label logprob, accuracy, harm, calibration.
```

### Sprint 4E: Robustness and Write-up

```text
held-out cyber task;
ablation;
failure cases;
paper narrative.
```

