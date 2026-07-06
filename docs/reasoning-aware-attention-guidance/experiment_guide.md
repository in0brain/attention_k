# Experiment Guide

本文件只负责实验流程、阶段输入输出、目录约定和运行边界。

它不是 task card，不规定每个 sprint 的具体改动范围；具体 sprint 以 `docs/codex_tasks/*.md` 为准。

它也不是 schema 文档；字段、枚举和标签规则统一放在 `docs/reasoning-aware-attention-guidance/label_schema.md`。

---

## 1. 项目主线

当前项目主线是：

```text
Reasoning-Aware Attention Guidance
```

完整目标流程是：

```text
question
→ baseline CoT generation
→ hidden states / attention maps cache
→ candidate span extraction
→ ablation unit construction
→ ablated question construction
→ NLI semantic consistency scoring
→ semantic necessity label rule
→ mask / remove intervention
→ semantic recoverability scoring
→ trajectory stability scoring
→ answer stability scoring
→ attention anchor labeling
→ oracle attention guidance
→ probe-guided attention guidance
→ hallucination reduction evaluation
```

其中 NLI、recoverability、trajectory stability、answer stability 和 raw attention pattern 都只是 attention importance discovery 的信号来源。

---

## 2. 当前运行边界

当前仍处于工程基础、Skill 文档和 schema 对齐阶段。

早期实现原则：

```text
stub-first
small sprint
jsonl first
no automatic next stage
no real model call unless explicitly allowed
```

每个阶段只产出自己的文件。

不要提前创建未来阶段输出文件。

不要提前实现后续阶段。

---

## 3. 目录约定

常用目录：

```text
configs/
data/examples/
data/processed/
outputs/logs/
outputs/evaluation/
cache/hidden_states/
cache/attentions/
src/recover_attention/
scripts/
tests/
docs/reasoning-aware-attention-guidance/
docs/codex_tasks/
docs/reference/
```

说明：

```text
data/examples/:
最小样例输入。

data/processed/:
jsonl 中间产物。

outputs/logs/:
脚本 smoke test 和运行日志。

cache/:
后续 hidden states / attention maps 的缓存位置，仅在 task card 明确允许时使用。
```

---

## 4. 阶段输入输出

长期目标文件流如下：

```text
data/examples/questions_small.jsonl
→ data/processed/questions.jsonl
→ data/processed/baseline_cot.jsonl
→ data/processed/baseline_trajectory_manifest.jsonl
→ data/processed/candidate_spans.jsonl
→ data/processed/ablation_units.jsonl
→ data/processed/ablated_questions.jsonl
→ data/processed/nli_scores.jsonl
→ data/processed/semantic_labels.jsonl
→ data/processed/masked_questions.jsonl
→ data/processed/recover_outputs.jsonl
→ data/processed/recover_scores.jsonl
→ data/processed/labels.jsonl / data/processed/token_labels.jsonl
→ data/processed/intervention_manifest.jsonl
→ data/processed/trajectory_stability_scores.jsonl
→ data/processed/answer_stability_scores.jsonl
→ data/processed/attention_anchor_labels.jsonl
→ data/processed/oracle_guidance_results.jsonl
→ data/processed/probe_guidance_results.jsonl
→ outputs/evaluation/*
```

这些是长期目标，不代表当前阶段都已经存在。

当前 1B-1E 阶段边界：

```text
ablation_units.jsonl:
  Sprint 1B 输出，定义 single/group ablation units。

ablated_questions.jsonl:
  Sprint 1C 输出，对 ablation units 执行 delete / generalize。

nli_scores.jsonl:
  Sprint 1D 输出，只保存 score-only 双向 NLI 结果。

semantic_labels.jsonl:
  Sprint 1E 输出，根据 nli_scores 构造 semantic necessity labels。
```

---

## 5. Hidden States / Attention Maps 边界

hidden states 和 attention maps 只能在 task card 明确允许时缓存。

边界要求：

```text
不要在 Sprint 0 缓存 hidden states。
不要在 Sprint 0 缓存 attention maps。
不要把大 tensor 直接写入 jsonl。
如果后续允许缓存，应保存为文件，并在 manifest jsonl 中记录路径。
```

---

## 6. Attention Guidance 边界

attention guidance 是后续验证阶段。

早期阶段只建立：

```text
数据格式
基础脚本
schema
stub / fixture
可验收文件流
```

不要在 Sprint 0、Sprint 1、Sprint 2 或 Sprint 3 中实现 oracle attention guidance 或 probe-guided attention guidance。

---

## 7. 验收原则

每个阶段完成时至少满足：

```text
输入文件明确
输出文件明确
运行命令明确
pytest 或 smoke test 可运行
PROGRESS.md 已更新
没有自动开始下一阶段
```

不得声称 attention guidance 有效，除非已经运行真实 guidance evaluation。

不得声称减少幻觉，除非已经完成对应评估。

---

## 8. 文档分工

```text
docs/reasoning-aware-attention-guidance/SKILL.md:
总入口和路由规则。

docs/reasoning-aware-attention-guidance/codex_tasks.md:
Sprint 路线、task card 规范和执行边界。

docs/reasoning-aware-attention-guidance/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/reasoning-aware-attention-guidance/method.md:
方法概念、信号关系和常见误解。

docs/reasoning-aware-attention-guidance/label_schema.md:
jsonl schema、字段、枚举和标签规则。

docs/reasoning-aware-attention-guidance/prompts.md:
可复用 Codex 提示词模板。
```
