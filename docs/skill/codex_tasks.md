# Codex Task Card Guide

本文件只负责说明 **task card 应该怎么写、怎么拆、怎么验收**。

本文件不是当前 sprint 路线图，也不是项目方法说明书。

---

## 1. 本文件职责

本文件负责：

```text
1. task card 的结构规范
2. task card 的粒度要求
3. task card 的命名规则
4. allowed / forbidden 文件列表写法
5. task-specific Preflight 应包含哪些内容
6. task-specific 验收标准如何写
7. task card 完成回复格式
```

本文件不负责：

```text
1. 当前下一步 sprint 是什么
2. 固定 sprint 路线
3. 完整实验 pipeline 顺序
4. 冲突优先级
5. 全局 Preflight 规则
6. 全局禁止事项
7. jsonl schema 字段定义
8. 方法概念解释
9. prompt 模板
```

这些内容分别由其他文件负责。

---

## 2. 单一来源原则

为避免文档重复和规则冲突，各类信息只保留一个主要来源。

```text
AGENTS.md:
Codex 行为规则、全局 Preflight、冲突优先级、环境规则、全局禁止事项、文件维护规则。

PROGRESS.md:
当前项目状态、已完成 sprint 摘要、当前可运行命令、遗留问题、下一步建议。

docs/skill/SKILL.md:
Skill 入口和文档路由。

docs/skill/codex_tasks.md:
task card 写法规范。

docs/skill/experiment_guide.md:
实验流程、阶段输入输出和运行关系。

docs/skill/method.md:
方法概念、术语、信号关系和常见误解。

docs/skill/label_schema.md:
jsonl schema、字段含义、枚举值和 record 示例。

docs/skill/prompts.md:
可复用 prompt 模板。

docs/reference/*:
长期完整方案、full design、背景参考和历史设计。

docs/codex_tasks/*.md:
当前 sprint 的具体执行边界。
```

如果本文件和 `AGENTS.md` 或当前 task card 冲突，应按 `AGENTS.md` 中定义的冲突优先级处理。

---

## 3. Task Card 的定位

每一张 task card 都是一轮 sprint 的执行合同。

它必须回答：

```text
这次只做什么？
输入是什么？
输出是什么？
允许改哪些文件？
禁止改哪些文件？
怎么验收？
跑哪些命令？
完成后如何更新 PROGRESS.md？
```

每次只执行一张 task card。

不要根据 task card 中的“下一步建议”自动开始下一张 task card。

---

## 4. Task Card 命名规则

建议命名格式：

```text
docs/codex_tasks/sprint_<sprint_id>_<short_task_name>.md
```

示例：

```text
docs/codex_tasks/sprint_1A_candidate_span_extraction.md
docs/codex_tasks/sprint_1B_ablated_question_construction.md
docs/codex_tasks/sprint_1C_nli_semantic_necessity_stub.md
```

命名原则：

```text
1. 文件名使用 snake_case。
2. 文件名能看出 sprint 编号和任务目标。
3. 不使用 final、new、latest、temp 等临时词。
4. 不把多个 sprint 合并到一个 task card 名称里。
```

---

## 5. Task Card 必备结构

一张 task card 至少应包含以下章节：

```text
1. 目标
2. 前置条件
3. 开始前必须读取
4. 允许修改
5. 禁止修改
6. Preflight 要求
7. 输入文件
8. 输出文件
9. 实现要求
10. 测试要求
11. 本 sprint 不做
12. 必须运行命令
13. PROGRESS.md 更新要求
14. 验收标准
15. 完成后回复格式
```

如果某个 sprint 是纯文档清理或纯代码测试，可以按需删减，但必须仍然明确：

```text
目标
允许修改
禁止修改
运行或检查命令
验收标准
```

---

## 6. Sprint 粒度原则

每张 task card 只做一个小目标。

推荐粒度：

```text
一个新模块
一个新 CLI
一个 schema 增量
一个数据转换阶段
一个 stub backend
一个测试补全任务
一个文档清理任务
```

不推荐粒度：

```text
实现整个 v0 pipeline
同时做 candidate span、ablation、NLI 和 recovery
同时做 schema、CLI、真实模型调用和评估
同时重构多个无关模块
```

如果 task card 需要修改过多文件，应拆分。

经验规则：

```text
普通代码 sprint：建议 2–5 个文件。
文档清理 sprint：建议 1–4 个文件。
涉及真实模型或大规模数据：必须单独成 sprint，并由 task card 明确允许。
```

---

## 7. Allowed / Forbidden 文件写法

每张 task card 必须显式列出允许修改和禁止修改的文件。

### 7.1 允许修改

写法示例：

```text
允许修改：

src/recover_attention/candidate_extraction.py
scripts/02_extract_candidate_spans.py
tests/test_candidate_extraction.py
data/processed/candidate_spans.jsonl
PROGRESS.md
```

原则：

```text
1. 只列本 sprint 真正需要修改的文件。
2. 如果文件不存在但允许创建，应明确说明。
3. 不要写过宽的通配范围，例如 src/*。
4. 不要把未来 sprint 文件提前放进 allowed list。
```

### 7.2 禁止修改

写法示例：

```text
禁止修改：

README.md
AGENTS.md
docs/skill/*
docs/reference/*
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

原则：

```text
1. 明确挡住相邻阶段文件。
2. 明确挡住未来阶段文件。
3. 明确挡住人类维护文档，除非本 sprint 是文档维护任务。
4. 禁止事项要和当前 sprint 目标相关，不要复制全局禁止事项全文。
```

全局禁止事项由 `AGENTS.md` 维护。

---

## 8. Preflight 写法

全局 Preflight 规则由 `AGENTS.md` 维护。

task card 只需要补充本 sprint 特有的 Preflight 检查。

示例：

```text
Preflight 必须额外报告：

1. 当前 data/processed/questions.jsonl 样本数量。
2. 当前 schemas.py 是否已有 validate_candidate_span_record。
3. 本次计划支持的 extractor 参数。
4. 本次是否需要读取 docs/reference/*。
```

不要在每张 task card 中完整复制全局冲突优先级。

如果需要提醒冲突处理，只写：

```text
如发现冲突，按 AGENTS.md 中定义的冲突优先级处理，并在 Preflight 中报告。
```

---

## 9. 输入与输出写法

每张 task card 必须明确输入和输出。

示例：

```text
输入：

data/processed/questions.jsonl

输出：

data/processed/candidate_spans.jsonl
```

如果某个输出只是测试产物或日志，也要说明。

如果 task card 不应该生成实验数据，必须明确写：

```text
本 sprint 不生成新的 data/processed/*.jsonl。
```

---

## 10. 实现要求写法

实现要求应具体到函数、脚本或行为。

示例：

```python
extract_candidate_spans(
    question: str,
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = 20,
) -> list[dict]
```

同时说明：

```text
1. 函数输入
2. 函数输出
3. 错误处理
4. 是否允许 stub
5. 是否允许真实模型
6. 是否必须复用现有 data_io / schemas
```

不要把方法背景长篇解释放进 task card。

方法概念应放在 `docs/skill/method.md`。

---

## 11. 测试要求写法

测试要求应列出可执行、可判断的检查项。

示例：

```text
至少覆盖：

1. valid record passes schema validation
2. missing required field raises ValueError
3. CLI writes expected jsonl output
4. output record count matches input record count
```

不要只写：

```text
测试要全面
```

也不要要求当前 sprint 无法支持的测试，例如在 stub 阶段要求真实模型质量指标。

---

## 12. “本 sprint 不做”写法

每张 task card 应列出本 sprint 的局部禁止事项。

示例：

```text
本 sprint 不做：

ablation construction
NLI scoring
masked question construction
question recovery
recoverability scoring
attention guidance
probe training
```

原则：

```text
1. 只列和当前任务相邻、容易被 Codex 提前实现的内容。
2. 不复制 AGENTS.md 中的全部全局禁止事项。
3. 如果某个未来内容在 routed 文档中出现，也必须明确说明它不是本 sprint 的执行许可。
```

---

## 13. 运行命令写法

每张 task card 必须列出需要运行的命令。

推荐格式：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
python -m pytest -q
```

规则：

```text
1. 使用 python -m pytest -q。
2. 不使用裸 pytest -q。
3. 使用 python -m pip，而不是裸 pip。
4. 新增 CLI 后必须运行该 CLI。
5. 如果当前目录是 git 仓库，建议运行 git diff --name-only 和 git status --short。
```

具体命令以当前 task card 为准。

---

## 14. PROGRESS.md 更新写法

每张 task card 应说明完成后如何更新 `PROGRESS.md`。

更新内容通常包括：

```text
1. 当前 sprint 已完成
2. 新增或修改文件
3. 新增可运行命令
4. 最近一次检查结果
5. 当前遗留问题
6. 下一步建议
```

`PROGRESS.md` 应保持短版状态索引。

详细日志应写入：

```text
docs/progress/*_history.md
```

不要把完整命令输出、长 diff 或长日志塞进 `PROGRESS.md`。

---

## 15. 验收标准写法

验收标准应该可检查。

示例：

```text
1. 指定模块存在。
2. 指定 CLI 可运行。
3. 输出 jsonl 文件存在。
4. 每行是合法 JSON object。
5. 输出 record 通过对应 schema validation。
6. python -m pytest -q 通过。
7. smoke test 通过。
8. 未实现下一阶段功能。
```

不要写不可检查的验收项，例如：

```text
模型效果很好
抽取质量优秀
方法足够先进
```

除非 task card 明确包含人工评审或定量指标。

---

## 16. 完成后回复格式

建议每张 task card 要求 Codex 完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. 输出文件或统计
6. PROGRESS.md 更新摘要
7. 遗留问题
8. 下一步建议
```

不要自动开始下一步。

---

## 17. Task Card 模板

可复制以下模板创建新 task card。

````markdown
# Sprint X：<Task Name>

## 1. 目标

本 sprint 只完成：

```text
<one goal>
```

## 2. 前置条件

```text
<required previous state>
```

## 3. 开始前必须读取

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
<additional required files>
```

## 4. 允许修改

```text
<allowed files>
```

## 5. 禁止修改

```text
<forbidden files>
```

## 6. Preflight 要求

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
<task-specific preflight items>
```

如发现冲突，按 AGENTS.md 中定义的冲突优先级处理，并在 Preflight 中报告。

## 7. 输入文件

```text
<input files>
```

## 8. 输出文件

```text
<output files>
```

## 9. 实现要求

```text
<implementation requirements>
```

## 10. 测试要求

```text
<test requirements>
```

## 11. 本 sprint 不做

```text
<local forbidden next-stage work>
```

## 12. 必须运行命令

```bash
<commands>
```

## 13. PROGRESS.md 更新要求

```text
<progress update requirements>
```

## 14. 验收标准

```text
<acceptance criteria>
```

## 15. 完成后回复格式

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. 遗留问题
6. 下一步建议
```
````

---

## 18. 最小原则

task card 越具体，Codex 越稳定。

优先写清楚：

```text
改哪些文件
不改哪些文件
输入是什么
输出是什么
跑什么命令
怎样算完成
```

不要依赖 Codex 从 full design 中自行推断当前 sprint 范围。
