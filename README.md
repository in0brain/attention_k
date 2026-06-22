# Recoverability-Guided Attention Allocation

## 1. 项目简介

本项目研究输入问题中的哪些 token / span 对问题语义、推理过程和最终答案具有不可替代作用，并进一步探索这些 token / span 的重要性信号能否用于指导模型 attention 分配，从而减少缺信息乱补、错误依赖和幻觉风险。

项目的核心思想不是直接从 attention 入手，而是先构建两个基础标签：

1. **Semantic Necessity**
   从人类语义角度判断：某个 span 被删除、泛化或遮盖后，原问题和改写问题是否仍然语义等价。

2. **Model Recoverability**
   从模型行为角度判断：某个 span 被 mask 后，模型是否还能稳定恢复原问题。

后续再进一步分析模型是否存在：

* 忽略关键 span；
* 过度关注非关键 span；
* 缺信息乱补；
* 高置信错误恢复；
* 异常 attention sink。

最终目标是：

```text
NLI semantic necessity
→ mask-recover recoverability
→ incomplete-question reasoning risk
→ token / span importance label
→ token importance probe
→ attention guidance
```

---

## 2. 当前阶段

当前项目处于早期实验阶段，只做 v0 / v1 的基础流程。

当前阶段重点是：

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring
```

当前暂时不做：

```text
probe 训练
attention guidance
hidden states
trajectory analysis
大规模实验
```

---

## 3. 阶段划分

### Sprint 0：基础实验框架

目标：

建立最小可运行工程框架，包括目录结构、配置文件、jsonl 数据读写、schema 校验、样例数据和 smoke test。

主要输出：

```text
README.md
AGENTS.md
PROGRESS.md
configs/v0_nli_small.yaml
data/examples/questions_small.jsonl
src/recover_attention/
scripts/
tests/
```

### Sprint 1：Candidate Span Extraction

目标：

从原始问题中抽取候选 span。

优先支持：

```text
number
entity
operation
relation
question_target
```

主要输出：

```text
data/processed/candidate_spans.jsonl
```

### Sprint 2：Ablated Question Construction

目标：

对每个 candidate span 构造 ablated question，用于后续 NLI 判断。

支持消融方式：

```text
delete
generalize
mask
```

主要输出：

```text
data/processed/ablated_questions.jsonl
```

### Sprint 3：NLI Semantic Necessity Scoring

目标：

对 original question 和 ablated question 做双向 NLI，判断删除或泛化某个 span 后是否造成语义变化。

主要输出：

```text
data/processed/nli_scores.jsonl
```

标签包括：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

### Sprint 4：Masked Question Construction

目标：

对 candidate span 构造 masked question，为后续问题复原实验做准备。

主要输出：

```text
data/processed/masked_questions.jsonl
```

### Sprint 5：Question Recovery

目标：

让模型只复原 masked question，不解题。

主要输出：

```text
data/processed/recover_outputs.jsonl
```

### Sprint 6：Recoverability Scoring

目标：

根据多次复原结果判断 span 是否可稳定恢复。

主要输出：

```text
data/processed/recover_scores.jsonl
```

标签包括：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

---

## 4. 推荐环境

建议使用独立 conda 环境：

```bash
conda create -n recover_attention python=3.10
conda activate recover_attention
pip install -r requirements.txt
```

---

## 5. 数据格式原则

本项目所有中间结果统一使用 jsonl。

每一行是一条 JSON 记录。

示例：

```json
{"id":"gsm8k_0001","dataset":"gsm8k","question":"Tom has 3 apples and buys 2 more. How many apples does he have now?","gold_answer":"5"}
```

---

## 6. 最小运行流程

### 6.1 Smoke Test

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

预期输出：

```text
[OK] Sprint 0 smoke test passed.
```

### 6.2 数据准备

```bash
python scripts/01_prepare_data.py \
  --input data/examples/questions_small.jsonl \
  --output data/processed/questions.jsonl
```

### 6.3 Candidate Span 抽取

```bash
python scripts/02_extract_candidate_spans.py \
  --input data/processed/questions.jsonl \
  --output data/processed/candidate_spans.jsonl
```

### 6.4 Ablated Question 构造

```bash
python scripts/03_build_ablated_questions.py \
  --input data/processed/candidate_spans.jsonl \
  --output data/processed/ablated_questions.jsonl
```

---

## 7. 项目原则

1. 先跑通最小闭环，再逐步增加复杂度。
2. 每个 sprint 只完成一个小目标。
3. 不提前实现后续阶段。
4. 所有实验结果必须可复现、可检查、可回滚。
5. Codex 负责代码实现和 PROGRESS.md 更新；README.md 和 AGENTS.md 由研究者维护。
6. 当前阶段优先保证数据格式、脚本接口和 pipeline 稳定，不追求模型性能。
7. 每一步都必须有明确输入、输出、运行命令和验收标准。
