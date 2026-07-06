# AGENTS.md

## 1. 项目定位

本项目是一个研究型实验项目，名称为：

```text
Reasoning-Aware Attention Guidance
```

项目目标不是单纯做：

```text
hallucination classifier
```

也不是单纯做：

```text
key span discovery
```

本项目真正要做的是：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

也就是说，项目希望先发现输入问题中真正影响推理的 token / span，再将这些 token / span 转化为 attention anchor，并在后续阶段验证 attention guidance 是否能够提升推理稳定性、减少幻觉。

NLI、recoverability、trajectory stability、answer stability 和 raw attention pattern 都只是 attention importance discovery 的信号来源。

最终目标不是普通的：

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

## 2. Codex 工作原则

### 2.1 小步迭代

每次只完成一个小 sprint。

不要一次性实现完整 pipeline。

如果任务描述中只要求 Sprint 0G，就不要实现 Sprint 1A。

如果任务描述中只要求 candidate span extraction，就不要实现 ablated question construction。

如果任务描述中只要求 NLI scoring，就不要实现 recoverability scoring。

---

### 2.2 先 Preflight，后修改

Codex 修改任何文件前，必须先输出 Preflight。

Preflight 必须包括：

```text
1. 已阅读文件列表
2. 本次允许修改的文件
3. 本次禁止修改的文件
4. 本次必须运行的命令
5. 是否需要读取 docs/reference/*
6. 是否发现冲突
```

Preflight 输出后必须暂停，等待用户确认。

用户确认后，Codex 才能修改文件。

---

### 2.3 保持可运行

每次修改后，必须保证当前项目仍然可以运行。

新增脚本后，必须提供运行命令。

新增模块后，必须保证可以正常 import。

新增 schema 后，必须补充或更新 pytest。

---

### 2.4 统一数据格式

所有中间结果默认使用：

```text
jsonl
```

每条记录占一行。

不要擅自改成：

```text
csv
pickle
sqlite
parquet
excel
```

除非当前 task card 明确要求导出辅助分析文件。

hidden states 和 attention maps 这类大 tensor 可以在后续阶段保存为 `.pt`，但必须通过 manifest jsonl 记录路径，不要直接嵌入 jsonl。

---

### 2.5 不要过度设计

优先使用简单、清晰、可检查的实现。

不要引入复杂框架。

不要引入不必要的大依赖。

不要提前抽象过多类。

不要为了“通用性”牺牲可读性。

---

### 2.6 不要擅自调用模型

除非当前 sprint 明确要求，否则不要：

```text
调用大模型
下载模型
调用外部 API
调用 OpenAI API
调用 Hugging Face 模型
运行 GPU 推理
缓存真实 hidden states
缓存真实 attention maps
训练 probe
执行 attention guidance
```

早期阶段优先完成：

```text
数据处理
schema 校验
脚本接口
stub backend
pytest
smoke test
```

---

## 3. 必读文件规则

每次执行任务前，Codex 默认必须阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
当前 task card
```

当前 task card 通常位于：

```text
docs/codex_tasks/*.md
```

是否阅读以下文件，由当前 task card 指定：

```text
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
```

不要默认读取全部 skill 子文档。

不要默认读取：

```text
docs/reference/*
```

除非当前 task card 或用户明确要求。

---

## 4. 冲突优先级

如果指令发生冲突，优先级为：

```text
当前用户消息
> 当前 task card
> AGENTS.md
> PROGRESS.md
> docs/reasoning-aware-attention-guidance/SKILL.md
> docs/reasoning-aware-attention-guidance/*.md
> docs/reasoning-aware-attention-guidance/prompts.md
> docs/reference/*
```

如果发现冲突，Codex 必须：

```text
1. 在 Preflight 中报告冲突。
2. 说明采用哪条规则。
3. 在最终回复的“遗留问题”中再次记录。
```

---

## 5. 文件维护规则

### 5.1 README.md

README.md 是项目说明文件，由研究者维护。

Codex 不应主动重写 README.md。

除非用户明确要求，否则不要修改 README.md。

---

### 5.2 AGENTS.md

AGENTS.md 是 Codex 工作规则文件，由研究者维护。

Codex 不应主动重写 AGENTS.md。

除非用户明确要求，否则不要修改 AGENTS.md。

---

### 5.3 docs/reasoning-aware-attention-guidance/*

`docs/reasoning-aware-attention-guidance/*` 是 Skill 框架文档，由研究者维护。

Codex 只有在当前 task card 明确允许时才能修改。

其中：

```text
docs/reasoning-aware-attention-guidance/SKILL.md:
Skill 总入口和路由规则。

docs/reasoning-aware-attention-guidance/codex_tasks.md:
task card 写法规范、拆分原则和验收格式。

docs/reasoning-aware-attention-guidance/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/reasoning-aware-attention-guidance/method.md:
方法概念、信号关系和常见误解。

docs/reasoning-aware-attention-guidance/label_schema.md:
jsonl schema、字段、枚举和标签规则。

docs/reasoning-aware-attention-guidance/prompts.md:
可复用 Codex 提示词模板。
```

---

### 5.4 docs/reference/*

`docs/reference/*` 是长期完整参考文档。

Codex 不应默认阅读或修改。

除非当前 task card 或用户明确要求，否则不要使用它们。

---

### 5.5 PROGRESS.md

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

不要声称 attention guidance 有效，除非已经有真实 guidance evaluation 指标支持。

不要声称减少幻觉，除非已经完成对应评估。

PROGRESS.md 只保留当前状态摘要，不作为完整历史日志。

每轮 sprint 的详细记录应归档到 docs/progress/*_history.md。

Codex 更新 PROGRESS.md 时应保持简洁，避免无限追加长篇日志。

---

## 6. 代码风格规则

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
11. 不要在一个 sprint 中实现多个 pipeline 阶段。

---

## 7. 环境与命令规则

当前推荐环境：

```text
conda env: recover_attention
python: 3.10
```

安装依赖必须使用：

```bash
python -m pip install -r requirements.txt
```

运行测试必须使用：

```bash
python -m pytest -q
```

不要使用：

```bash
pip install -r requirements.txt
pytest -q
```

不要使用：

```text
.venv
```

除非用户明确切换环境方案。

如果 Python 或 pip 指向错误环境，停止并报告。

---

## 8. 推荐目录结构

```text
recover_attention_project/
  README.md
  AGENTS.md
  PROGRESS.md
  requirements.txt
  pyproject.toml

  docs/
    skill/
      SKILL.md
      codex_tasks.md
      experiment_guide.md
      label_schema.md
      method.md
      prompts.md

    codex_tasks/
      sprint_*.md

    reference/
      项目方案_*.md
      实验指导书_*.md

  configs/
    v0_nli_small.yaml

  data/
    raw/
    processed/
    examples/

  cache/
    hidden_states/
    attentions/

  outputs/
    logs/
    evaluation/

  reports/

  src/
    recover_attention/
      __init__.py
      data_io.py
      schemas.py
      prepare_data.py
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

注意：

```text
目录结构是路线图，不代表当前 sprint 必须创建所有文件。
不要提前创建未来阶段模块，除非当前 task card 明确要求。
```

---

## 9. Sprint 完成后回复格式

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

如果当前目录是 git 仓库，还应报告：

```bash
git diff --name-only
git status --short
```

不要自动 `git add` 或 `git commit`，除非用户明确要求。

---

## 10. 禁止事项清单

除非当前 task card 明确允许，否则不要做：

```text
不要一次性实现完整 pipeline。
不要跳过 Preflight。
不要跳过 PROGRESS.md 更新。
不要引入大型依赖。
不要调用外部 API。
不要下载模型。
不要训练模型。
不要做真实 hidden states 缓存。
不要做真实 attention maps 缓存。
不要做 trajectory stability scoring。
不要做 attention guidance。
不要做 probe training。
不要做大规模实验。
不要把 jsonl 改成其他格式。
不要删除已有实验文件。
不要覆盖用户手写文档。
不要把 NLI 或 recoverability 当作最终目标。
不要把 key span discovery 当作最终目标。
不要声称减少幻觉，除非已经完成对应评估。
```
