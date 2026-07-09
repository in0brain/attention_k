# Sprint 4A: Cybersecurity Direction-Probe Mainline Reset

## Positioning

Sprint 4A is a mainline reset, requirements clarification, documentation update, and code reuse audit sprint.

It is not a training sprint, model-call sprint, steering sprint, 2000-scale rerun, or full Sprint 3C continuation.

The project mainline is reset from:

```text
unsupervised span-guided steering
```

to:

```text
cyber-domain supervised direction probe / controller
```

This reset preserves the Sprint 3 mechanism findings:

```text
1. blind span steering failed;
2. answer-readout MLP is a valid causal site;
3. gold direction is effective but not deployable;
4. gold-free direct readout / approximate J-lens detection did not beat final logits;
5. therefore the next mainline needs domain supervision:
   span / activation / trace -> label direction / correction direction.
```

## Strict Boundaries

Forbidden:

```text
- do not train a probe;
- do not call an LLM or generate new model outputs;
- do not run steering;
- do not patch or nudge model activations;
- do not enter 2000-scale;
- do not run full Sprint 3C;
- do not claim hallucination reduction;
- do not claim answer accuracy improvement;
- do not delete or weaken Sprint 3 negative results;
- do not describe Sprint 3C-4A as useless;
- do not claim the new cyber probe already works.
```

Allowed:

```text
- update PROGRESS.md;
- update docs/reference/STORY.md;
- create Sprint 4 mainline design documentation;
- create code reuse audit documentation;
- create cyber data requirements documentation;
- create Sprint 4 history and artifact manifest;
- record why the original unsupervised steering plan is insufficient;
- define later Sprint 4B / 4C / 4D / 4E route;
- audit which old Python modules are reusable, adaptable, or historical only.
```

Default review flags:

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
domain_supervised_probe_planned=True
```

## Required Read Files

```text
AGENTS.md
PROGRESS.md
docs/reference/STORY.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md
docs/codex_tasks/sprint_3C_2_mlp_readout_direction_analysis.md
docs/codex_tasks/sprint_3C_3_mlp_readout_attribution_probe.md
docs/codex_tasks/sprint_3C_4A_approx_j_lens_readout_sanity_check.md
src/recover_attention/module_causal_tracing.py
src/recover_attention/mlp_readout_direction.py
src/recover_attention/mlp_readout_attribution.py
src/recover_attention/answer_proxy_metrics.py
src/recover_attention/approx_j_lens_readout.py
```

## Required Outputs

Create or update:

```text
docs/codex_tasks/sprint_4A_cyber_direction_probe_mainline_reset.md
PROGRESS.md
docs/reference/STORY.md
docs/reference/CYBER_DIRECTION_PROBE_PLAN.md
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/preflight_report.md
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/review_gate_cyber_direction_probe_mainline_reset.md
```

## Story Update Requirements

Append these sections to `docs/reference/STORY.md`:

```text
## 17. 为什么最初的 span-guided steering 计划无效
## 18. Sprint 4 新主线：领域监督 direction probe
```

Section 17 must state that span relevance says where important evidence may be, but not which answer direction to apply. Attention bias changes read proportions; residual injection changes a high-dimensional mixed state; neither supplies the answer-correction direction. Gold-unembedding direction works only as an oracle upper bound. Direct readout / approximate J-lens did not supply a strong gold-free target. Therefore the original unsupervised steering mainline is insufficient.

Section 18 must state that Sprint 4 no longer treats attention or span labels as the steering handle. Reasoning-aware signals become probe inputs, and cyber-domain supervision is used to learn the mapping to label or correction directions.

## Cyber Direction-Probe Plan Requirements

`docs/reference/CYBER_DIRECTION_PROBE_PLAN.md` must cover:

```text
1. new research question;
2. why cyber is better suited than GSM8K for this next step;
3. dataset requirements;
4. label-space requirements;
5. correct/wrong pair construction;
6. probe inputs;
7. probe outputs;
8. steering direction definitions;
9. training targets;
10. evaluation;
11. baselines;
12. risk boundaries;
13. Sprint 4B / 4C / 4D / 4E route.
```

The initial cyber task should prioritize structured prediction:

```text
classification / multiple-choice / finite label-space prediction
```

and should not start with open-ended cyber QA.

The canonical sample schema is:

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

Gold labels may be used as training targets and evaluation labels only. They must not be inference features.

## Code Reuse Audit Requirements

`docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md` must classify existing code into:

```text
A. directly reusable;
B. adaptable;
C. not recommended as the new mainline.
```

Required direct-reuse modules:

```text
src/recover_attention/module_causal_tracing.py
src/recover_attention/mlp_readout_direction.py
src/recover_attention/mlp_readout_attribution.py
src/recover_attention/approx_j_lens_readout.py
```

Required adaptable module:

```text
src/recover_attention/answer_proxy_metrics.py
```

Required historical-only direction:

```text
attention_bias_steering.py
representation_intervention.py
old 3A / 3B scripts
GSM8K numeric-answer scripts
```

## Review Gate

Create:

```text
outputs/logs/sprint_4A_cyber_direction_probe_mainline_reset/review_gate_cyber_direction_probe_mainline_reset.md
```

It must answer whether this sprint:

```text
1. made no new model calls;
2. trained no probe;
3. ran no steering;
4. recorded why span-guided steering is insufficient;
5. reset the mainline to cyber-domain supervised direction probing;
6. preserved 3C-1 / 3C-2 mechanism findings;
7. preserved 3C-3 / 3C-4A detection/readout limits;
8. stated key span labels alone are insufficient;
9. stated the missing mapping is span/activation -> direction;
10. required structured cyber label space;
11. blocked gold-label leakage at inference;
12. completed code reuse audit;
13. listed Sprint 4B / 4C / 4D / 4E;
14. updated PROGRESS.md and STORY.md;
15. created the Sprint 4 plan and audit docs;
16. made no hallucination or answer-accuracy claim;
17. recommends Sprint 4B dataset/schema work next.
```

## Required Commands

```bash
conda run -n recover_attention python -m pytest tests/test_dataset_audit.py tests/test_stage_summary.py -q
conda run -n recover_attention python -m pytest tests/test_mlp_readout_attribution.py tests/test_approx_j_lens_readout.py -q
git diff --name-only
git status --short
```

Do not introduce markdown linting if no existing markdown lint command exists.

Do not commit automatically. Suggested commit message if the user later requests a commit:

```text
4A reset mainline to cyber direction probe
```

## Allowed Final Conclusion

```text
Sprint 4A resets the project mainline from unsupervised span-guided steering to cyber-domain supervised direction probing. Prior Sprint 3 results show that span relevance alone does not provide a reliable answer-correction direction, while answer-readout MLP remains a valid causal site. The next stage will design and evaluate a domain-supervised probe/controller that maps reasoning-aware features to label or correction directions.
```

