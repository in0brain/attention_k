# Reasoning-Aware Attention Guidance

## 1. 项目简介

本项目研究如何建立一种 **Reasoning-Aware Attention Guidance** 方法。

项目目标不是单纯做幻觉检测器，也不是单纯发现关键 span。

本项目真正关注的是：

```text
哪些输入 token / span 在模型推理过程中应该被稳定关注？
```

进一步说，本项目希望先发现输入问题中真正影响推理的 token / span，再将这些 token / span 转化为 attention anchor，并在后续阶段验证 attention guidance 是否能够提升推理稳定性、减少幻觉风险。

完整研究闭环是：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

最终希望得到的不只是普通的：

```text
hallucination_label
```

而是：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

---

## 2. 核心思想

本项目不直接从 raw attention 出发判断重要性。

原因是：

```text
attention 高 != span 一定重要
attention 低 != span 一定不重要
```

因此，本项目先从多个信号中发现 candidate span 的推理重要性：

```text
Semantic Necessity
Semantic Recoverability
Trajectory Stability
Answer Stability
Raw Attention Pattern
Attention Steering Effect
```

这些信号共同构成 evidence，再形成：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

---

## 3. 主要信号

### 3.1 Semantic Necessity

从语义角度判断：

```text
某个 span 被删除、泛化或替换后，原问题和扰动问题是否仍然等价？
```

实现方式：

```text
original_question → ablated_question
ablated_question → original_question
```

通过双向 NLI 得到：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

NLI 只是 attention importance discovery 的辅助信号，不是最终目标。

---

### 3.2 Semantic Recoverability

从模型行为角度判断：

```text
某个 span 被 mask 后，模型是否还能稳定恢复原问题语义？
```

注意，recovery 的任务是：

```text
复原问题，不是解题。
```

recoverability 标签包括：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

Recoverability 不是最终目标，也不能单独决定 attention anchor。

---

### 3.3 Trajectory Stability

从推理轨迹角度判断：

```text
某个 span 被 mask / remove / replace 后，模型的 CoT trajectory、hidden-state geometry 或 step-wise trajectory 是否明显偏移？
```

该信号用于判断 span 是否对推理过程本身有影响。

---

### 3.4 Answer Stability

从最终答案角度判断：

```text
某个 span 被干预后，final answer 是否发生变化或变得不稳定？
```

注意：

```text
答案不变不等于 span 不重要。
答案变化也不一定说明 span 是有效 anchor。
```

---

### 3.5 Raw Attention Pattern

从模型原始 attention 角度观察：

```text
模型是否自然关注了该 candidate span？
```

Raw attention 只能作为辅助信号，不能单独证明因果重要性。

---

### 3.6 Attention Steering Effect

从干预效果角度验证：

```text
增强或抑制某些 anchor 的 attention 后，推理是否更稳定，幻觉是否减少？
```

这是后续验证阶段，不属于早期数据流本身。

---

## 4. 当前阶段

当前项目处于工程基础和 Skill 框架对齐阶段。

已经完成的 Sprint 0 内容包括：

```text
项目骨架
jsonl 数据读写
样例数据
smoke test
schema 校验
prepare_data
基础环境验收
```

当前优先事项是：

```text
Skill 文档与新研究主线对齐
schema 与 attention anchor 标签体系对齐
```

不要直接跳到 attention guidance、probe training 或大规模实验。

---

## 5. 高层路线

当前高层路线为：

```text
Sprint 0:
工程基础与 Skill 框架

Sprint 1:
Baseline CoT 与推理轨迹基础

Sprint 2:
Candidate Span 与 NLI 语义必要性

Sprint 3:
Mask / Remove Intervention 与 Semantic Recoverability

Sprint 4:
Trajectory Stability 与 Answer Stability

Sprint 5:
Attention Importance Hierarchy 与 Anchor Labeling

Sprint 6:
Oracle Attention Guidance

Sprint 7:
Probe-Guided Attention Guidance

Sprint 8:
Hallucination Reduction Evaluation
```

阶段边界：

```text
Sprint 0 只做工程和文档地基。
Sprint 1 可以使用 stub / fixture，不默认调用真实模型。
Sprint 2 可以使用 rule-based span extraction 和 NLI stub。
Sprint 3 可以使用 recovery stub。
Sprint 5 之前不要实现 attention guidance。
Sprint 7 之前不要训练 probe。
```

---

## 6. 数据格式原则

所有中间结果默认使用：

```text
jsonl
```

每一行是一条 JSON object。

不要把以下格式作为主中间格式：

```text
csv
excel
pickle
sqlite
parquet
```

除非某个 task card 明确要求导出辅助分析文件。

大 tensor，例如 hidden states 和 attention maps，可以在后续阶段保存为 `.pt` 文件，但 jsonl 中只记录路径，不直接嵌入 tensor。

---

## 7. 关键中间文件

完整目标 pipeline 中可能出现的关键文件包括：

```text
data/processed/questions.jsonl
data/processed/baseline_cot.jsonl
data/processed/baseline_trajectory_manifest.jsonl
data/processed/candidate_spans.jsonl
data/processed/ablated_questions.jsonl
data/processed/nli_scores.jsonl
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/intervention_manifest.jsonl
data/processed/trajectory_stability_scores.jsonl
data/processed/answer_stability_scores.jsonl
data/processed/attention_anchor_labels.jsonl
data/processed/oracle_guidance_results.jsonl
data/processed/probe_guidance_results.jsonl
```

注意：

```text
这些文件是长期目标，不代表当前阶段都已经存在。
不要提前创建未来阶段文件，除非当前 sprint 明确要求。
```

---

## 8. 推荐环境

建议使用独立 conda 环境：

```bash
conda create -n recover_attention python=3.10
conda activate recover_attention
python -m pip install -r requirements.txt
```

不要使用裸命令：

```bash
pip install -r requirements.txt
```

运行测试：

```bash
python -m pytest -q
```

不要使用裸命令：

```bash
pytest -q
```

---

## 9. 当前最小运行流程

### 9.1 Smoke Test

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

预期输出类似：

```text
[OK] Sprint 0B smoke test passed.
```

---

### 9.2 数据准备

```bash
python scripts/01_prepare_data.py \
  --input data/examples/questions_small.jsonl \
  --output data/processed/questions.jsonl
```

---

### 9.3 测试

```bash
python -m pytest -q
```

---

## 10. 文档结构

本项目的 Skill 文档位于：

```text
docs/reasoning-aware-attention-guidance/
```

核心文件包括：

```text
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
```

其中：

```text
SKILL.md:
Skill 总入口和路由规则。

codex_tasks.md:
Sprint 路线和 task card 规范。

experiment_guide.md:
实验流程、阶段输入输出和运行边界。

method.md:
方法概念、信号关系和常见误解。

label_schema.md:
jsonl schema、字段、枚举和标签规则。

prompts.md:
可复用 Codex 提示词模板。
```

完整参考文档可以放在：

```text
docs/reference/
```

参考文档不是默认执行入口。

---

## 11. 项目原则

1. 先跑通最小可验收文件流，再逐步增加复杂度。
2. 每个 sprint 只完成一个小目标。
3. 不提前实现后续阶段。
4. 所有实验结果必须可复现、可检查、可回滚。
5. Codex 负责代码实现和 PROGRESS.md 更新。
6. README.md、AGENTS.md 和 docs/reasoning-aware-attention-guidance/* 通常由研究者维护。
7. 当前阶段优先保证数据格式、脚本接口和 pipeline 稳定，不追求模型性能。
8. 每一步都必须有明确输入、输出、运行命令和验收标准。
9. 不调用真实模型，除非当前 sprint 明确要求。
10. 不声称 attention guidance 有效，除非已经完成真实评估。

---

## 12. 禁止提前实现

除非当前 task card 明确允许，否则不要提前实现：

```text
真实模型调用
真实 NLI 模型调用
真实 recovery 模型调用
真实 hidden states 大规模缓存
真实 attention maps 大规模缓存
trajectory stability scoring
attention guidance
probe training
large-scale experiments
paper-level evaluation
```

尤其不要在 Sprint 0、Sprint 1、Sprint 2、Sprint 3 中提前实现：

```text
oracle attention guidance
probe-guided attention guidance
```
