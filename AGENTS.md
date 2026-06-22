# AGENTS.md

## 1. 项目定位

本项目是一个研究型实验项目，名称为：

```text
Recoverability-Guided Attention Allocation
```

项目目标是通过 Semantic Necessity 和 Model Recoverability 两类信号，发现输入问题中的关键 token / span，并为后续 token importance probe 与 attention guidance 做准备。

当前阶段只做基础 pipeline，不做完整模型干预。

---

## 2. 当前实验边界

当前只允许推进 v0 / v1 阶段：

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring
```

当前禁止提前实现：

```text
probe 训练
attention guidance
hidden states
trajectory analysis
大规模实验
```

除非用户明确要求，否则不要创建这些模块的复杂实现。

---

## 3. Codex 工作原则

### 3.1 小步迭代

每次只完成一个小 sprint。

不要一次性实现完整 pipeline。

如果任务描述中只要求 Sprint 0A，就不要实现 Sprint 0B。

如果任务描述中只要求 candidate span extraction，就不要实现 ablated question construction。

### 3.2 保持可运行

每次修改后，必须保证当前项目仍然可以运行。

新增脚本后，必须提供运行命令。

新增模块后，必须保证可以正常 import。

### 3.3 统一数据格式

所有中间结果统一使用 jsonl。

每条记录占一行。

不要擅自改成 csv、pickle、sqlite 或其他格式。

除非用户明确要求，不要保存二进制中间结果。

### 3.4 不要过度设计

优先使用简单、清晰、可检查的实现。

不要引入复杂框架。

不要引入不必要的大依赖。

不要提前抽象过多类。

不要为了“通用性”牺牲可读性。

### 3.5 不要擅自调用模型

除非当前 sprint 明确要求，否则不要：

```text
调用大模型
下载模型
调用外部 API
调用 OpenAI API
调用 Hugging Face 模型
运行 GPU 推理
训练 probe
```

当前早期阶段优先完成数据处理、格式检查和脚本接口。

---

## 4. 文件维护规则

### 4.1 README.md

README.md 是项目说明文件，由研究者维护。

Codex 不应主动重写 README.md。

除非用户明确要求，否则不要修改 README.md。

### 4.2 AGENTS.md

AGENTS.md 是 Codex 工作规则文件，由研究者维护。

Codex 不应主动重写 AGENTS.md。

除非用户明确要求，否则不要修改 AGENTS.md。

### 4.3 PROGRESS.md

PROGRESS.md 是实验进度记录文件，由 Codex 每轮任务完成后更新。

每次完成 sprint 后，必须更新 PROGRESS.md。

更新内容必须包括：

```text
已完成内容
新增或修改文件
输入文件
输出文件
运行命令
检查结果
遗留问题
下一步任务
```

不要在 PROGRESS.md 中写夸张结论。

不要声称实验有效，除非已经有真实指标支持。

---

## 5. 代码风格规则

1. Python 版本默认使用 3.10+。
2. 所有脚本必须支持命令行参数。
3. 路径不要写死，优先从参数或 config 中读取。
4. 文件读写统一放在 `src/recover_attention/data_io.py`。
5. 数据结构校验统一放在 `src/recover_attention/schemas.py`。
6. 每个脚本只做一件事。
7. 每个脚本都应该打印关键输入、输出和样本数量。
8. 报错信息要清楚，尽量指出具体文件和行号。
9. 不要吞掉异常。
10. 不要用 notebook 作为主流程。

---

## 6. 推荐目录结构

```text
recover_attention_project/
  README.md
  AGENTS.md
  PROGRESS.md
  requirements.txt
  pyproject.toml

  docs/
    method.md
    experiment_guide.md
    label_schema.md
    prompts.md
    codex_tasks.md

  configs/
    v0_nli_small.yaml

  data/
    raw/
    processed/
    examples/

  outputs/
    logs/

  src/
    recover_attention/
      __init__.py
      data_io.py
      schemas.py
      candidate_extraction.py
      question_ablations.py
      nli_scoring.py
      recover_generation.py
      recover_scoring.py
      label_builder.py
      evaluation.py
      utils.py

  scripts/
    00_smoke_test.py
    01_prepare_data.py
    02_extract_candidate_spans.py
    03_build_ablated_questions.py
    04_run_nli_scoring.py
    05_build_masked_questions.py

  tests/
```

---

## 7. Sprint 更新格式

每次完成任务后，请按照以下格式回复用户：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. 遗留问题
7. 下一步建议
```

不要自动开始下一步。

---

## 8. 当前优先级

当前最高优先级：

```text
稳定工程框架
统一 jsonl 格式
明确 schema
跑通最小 pipeline
保证每一步可验收
```

当前最低优先级：

```text
模型效果
大规模实验
复杂算法
性能优化
attention guidance
probe 训练
hidden state 分析
```

---

## 9. 禁止事项清单

除非用户明确要求，否则不要做：

```text
不要重写 README.md。
不要重写 AGENTS.md。
不要一次性实现完整 pipeline。
不要跳过 PROGRESS.md 更新。
不要引入大型依赖。
不要调用外部 API。
不要下载模型。
不要训练模型。
不要做 hidden states。
不要做 trajectory analysis。
不要做 attention guidance。
不要做大规模实验。
不要把 jsonl 改成其他格式。
不要删除已有实验文件。
不要覆盖用户手写文档。
```
