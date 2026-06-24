# Sprint 1C：Ablated Question Construction

## 1. 目标

本 sprint 实现 ablated question construction 的最小可运行版本。

目标文件流：

```text
data/processed/ablation_units.jsonl
→ data/processed/ablated_questions.jsonl
```

本 sprint 只负责把 1B 生成的 ablation units 转换成可供 NLI 使用的 ablated questions。

本 sprint 不直接从 `candidate_spans.jsonl` 构造 ablated questions。

本 sprint 不做 NLI scoring，不做 semantic necessity label，不做 masked question construction，不做 question recovery，不做 recoverability scoring。

---

## 2. 设计原则

1B 已经决定“一次 ablation 作用于什么对象”。

因此 1C 不再决定删几个 span，而是处理一个明确的 ablation unit。

ablation unit 可以是：

```text
single unit:
  只包含一个 span

group unit:
  包含多个有语义理由的 spans
```

本 sprint 对每个 ablation unit 构造两类 ablated question：

```text
delete
generalize
```

含义：

```text
delete:
  直接删除 unit 中的 span，不保留占位符。
  用于测试隐式信息缺失后的语义变化。

generalize:
  将 unit 中的 span 替换为泛化表达。
  用于测试精确信息弱化后的语义变化。
```

本 sprint 不做：

```text
mask
```

原因：

```text
mask 是显式缺口，主要用于后续 recoverability 阶段；
delete / generalize 才是当前 NLI semantic necessity 阶段的输入。
```

---

## 3. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1B 已完成。
data/processed/ablation_units.jsonl 已存在。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/ablation_units.jsonl
Please run Sprint 1B first.
```

不要自动回头执行 Sprint 1A 或 Sprint 1B。

---

## 4. 环境要求

本项目使用既有 conda 环境：

```text
D:\conda\Miniconda3\envs\recover_attention
```

禁止创建新的 conda env。

修改文件前，Codex 必须报告：

```powershell
$env:CONDA_DEFAULT_ENV
where.exe python
python -c "import sys; print(sys.executable); print(sys.version)"
```

如果当前 `python` 不是：

```text
D:\conda\Miniconda3\envs\recover_attention\python.exe
```

则停止并报告环境不一致，不要继续执行测试。

---

## 5. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/skill/label_schema.md
docs/skill/ablation_units_interface.md
docs/skill/ablated_questions_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
data/processed/ablation_units.jsonl
```

如果以下文件存在，也读取：

```text
src/recover_attention/question_ablations.py
tests/test_question_ablations.py
```

不要读取：

```text
docs/reference/*
```

除非当前用户指令明确要求。

---

## 6. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/question_ablations.py
scripts/04_build_ablated_questions.py
tests/test_question_ablations.py
src/recover_attention/schemas.py
tests/test_schemas.py
data/processed/ablated_questions.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
docs/skill/ablated_questions_interface.md
docs/skill/SKILL.md
```

如果以下文件不存在，可以创建：

```text
src/recover_attention/question_ablations.py
scripts/04_build_ablated_questions.py
tests/test_question_ablations.py
docs/progress/sprint_1_history.md
docs/skill/ablated_questions_interface.md
```

说明：

```text
本 sprint 新增 ablated_questions.jsonl artifact，
因此允许在 schemas.py 中新增 validate_ablated_question_record。

docs/skill/ablated_questions_interface.md 可以创建或更新。

docs/skill/SKILL.md 只允许增加 ablated_questions_interface.md 的索引行。
```

---

## 7. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/skill/*（除 docs/skill/ablated_questions_interface.md，以及 docs/skill/SKILL.md 的索引行外）
docs/reference/*
docs/codex_tasks/*
configs/*
requirements.txt
pyproject.toml
.gitignore
src/recover_attention/data_io.py
src/recover_attention/prepare_data.py
src/recover_attention/candidate_extraction.py
src/recover_attention/ablation_units.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 8. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1. 是否确认当前进入 Sprint 1C：Ablated Question Construction
2. 当前 Python 环境信息
3. data/processed/ablation_units.jsonl 是否存在
4. ablation unit record 数量
5. unit 总数
6. single unit 数量
7. group unit 数量
8. 当前 schemas.py 是否已有 validate_ablated_question_record
9. 本次是否需要新增 validate_ablated_question_record
10. 本次计划支持的 ablation types
11. 本次是否明确不生成 nli_scores.jsonl
12. 本次是否明确不做 mask / recovery / label building
```

如发现冲突，按 AGENTS.md 中定义的冲突优先级处理，并在 Preflight 中报告。

---

## 9. 输入文件

```text
data/processed/ablation_units.jsonl
```

输入 record 示例：

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "units": [
    {
      "unit_id": "unit_001",
      "unit_scope": "single",
      "group_type": "single",
      "span_ids": ["span_001"],
      "spans": [
        {
          "span_id": "span_001",
          "text": "3",
          "type": "number",
          "start": 8,
          "end": 9
        }
      ],
      "reason": "single candidate span"
    }
  ]
}
```

---

## 10. 输出文件

```text
data/processed/ablated_questions.jsonl
```

## Ablated Questions Interface Contract

Sprint 1C 的输入必须符合 `docs/skill/ablation_units_interface.md`。

Sprint 1C 的输出必须符合 `docs/skill/ablated_questions_interface.md`。

`data/processed/ablated_questions.jsonl` 是 Sprint 1D 的稳定输入。

每条 ablated question record 必须包含：

```text
ablation_id
id
unit_id
unit_scope
group_type
span_ids
spans
ablation_type
original_question
ablated_question
```

其中：

来自 Sprint 1B / `ablation_units.jsonl` 的字段：

```text
id
unit_id
unit_scope
group_type
span_ids
spans
```

由 Sprint 1C 新增的字段：

```text
ablation_id
ablation_type
original_question
ablated_question
```

输出 record 示例：

```json
{
  "ablation_id": "gsm8k_0001__unit_001__generalize",
  "id": "gsm8k_0001",
  "unit_id": "unit_001",
  "unit_scope": "single",
  "group_type": "single",
  "span_ids": ["span_001"],
  "spans": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    }
  ],
  "ablation_type": "generalize",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "ablated_question": "Tom has some number apples and buys 2 more. How many apples does he have now?"
}
```

每条输出 record 必须通过：

```python
validate_ablated_question_record(record)
```

---

## 11. Ablated Question Schema 要求

在 `src/recover_attention/schemas.py` 中新增：

```python
validate_ablated_question_record(record: dict) -> None
```

record 必须包含：

```text
ablation_id
id
unit_id
unit_scope
group_type
span_ids
spans
ablation_type
original_question
ablated_question
```

字段要求：

```text
ablation_id: str, non-empty
id: str, non-empty
unit_id: str, non-empty
unit_scope: one of single / group
group_type: str, non-empty
span_ids: list[str], non-empty
spans: list[dict], non-empty
ablation_type: one of delete / generalize
original_question: str, non-empty
ablated_question: str, non-empty
```

约束：

```text
1. span_ids 长度必须等于 spans 长度。
2. spans 中每个 span 必须包含 span_id / text / type / start / end。
3. spans 中 span_id 顺序必须与 span_ids 一致。
4. unit_scope = single 时，span_ids 长度必须为 1。
5. unit_scope = group 时，span_ids 长度必须 >= 2。
6. ablated_question 不应与 original_question 完全相同。
```

---

## 12. 核心接口要求

在 `src/recover_attention/question_ablations.py` 中实现：

```python
apply_ablation_to_unit(
    question: str,
    unit: dict,
    ablation_type: str,
    language: str = "auto",
) -> str
```

参数含义：

```text
question:
原始问题文本。

unit:
来自 ablation_units.jsonl 的单个 ablation unit。

ablation_type:
当前支持 delete / generalize。

language:
语言设置。当前允许 auto / en / zh。
```

如果传入不支持的 `ablation_type`，应抛出 ValueError。

---

## 13. 文件级函数要求

在 `src/recover_attention/question_ablations.py` 中实现：

```python
build_ablated_question_records(
    ablation_unit_records: list[dict],
    ablation_types: list[str] | None = None,
    language: str = "auto",
) -> tuple[list[dict], dict]
```

返回值含义：

```text
records:
  ablated question records。

stats:
  构造过程统计。
```

stats 至少包含：

```python
{
    "num_input_questions": 0,
    "num_input_units": 0,
    "num_output_ablations": 0,
    "num_skipped_empty": 0,
    "num_skipped_unchanged": 0,
    "num_skipped_overlap": 0,
    "ablation_type_counts": {},
    "unit_scope_counts": {},
    "group_type_counts": {},
}
```

CLI 必须使用该 stats 打印统计信息。

默认：

```text
ablation_types = ["delete", "generalize"]
```

要求：

```text
1. 对每个 unit 构造 delete 和 generalize 两条 ablated record。
2. 每条输出都保留 unit 元数据。
3. 每条输出都有唯一 ablation_id。
4. 如果 ablated_question 为空字符串或全空白，应跳过。
5. 如果 ablated_question 与 original_question 完全相同，应跳过。
6. 每条输出必须通过 validate_ablated_question_record。
7. 不要在 ablated question record 中加入 NLI 结果。
8. 不要在 ablated question record 中加入 recoverability 结果。
9. 不要在 ablated question record 中加入 attention anchor label。
```

---

## 14. 多 Span Unit 的 Offset 处理规则

如果一个 unit 中包含多个 spans，必须按 `start` 从大到小处理。

原因：

```text
如果先修改前面的 span，后面 span 的 start / end offset 会失效。
```

正确处理顺序：

```python
spans = sorted(unit["spans"], key=lambda s: s["start"], reverse=True)
```

然后逐个执行：

```python
question = question[:start] + replacement + question[end:]
```

delete 和 generalize 都必须遵守这个规则。

---

## 15. Overlap 处理规则

如果一个 unit 内部存在重叠 spans，应跳过该 unit 的 ablation，并在统计中报告。

重叠示例：

```text
span_a: start=10, end=20
span_b: start=15, end=25
```

跳过原因：

```text
重叠 span 的替换顺序会导致语义和 offset 难以稳定解释。
```

本 sprint 不处理 overlapping spans。

---

## 16. Delete Ablation 规则

delete ablation 直接删除 unit 中所有 spans。

必须使用 start / end 切片。

不要使用全局字符串替换。

原因：

```text
同一个 span text 可能在问题中出现多次，只能处理当前 unit 指定的 occurrence。
```

删除后需要做轻量 cleanup：

```text
1. 合并连续空格。
2. 去掉英文标点前多余空格。
3. 去掉中文标点附近多余空格。
4. 不做复杂语法重写。
```

不要把 delete 结果自动改写成 generalize 结果。

---

## 17. Generalize Ablation 规则

generalize ablation 将 unit 中每个 span 按其 type 替换为泛化表达。

### 17.1 English generalization

推荐规则：

```text
number → some number
entity → some entity
object → some object
operation → some operation
relation → some relation
comparison → some comparison
negation → some negation condition
condition → some condition
question_target → some question target
cyber_security_term → some security issue
```

### 17.2 Chinese generalization

推荐规则：

```text
number → 某个数量
entity → 某个实体
object → 某个对象
operation → 某个操作
relation → 某种关系
comparison → 某种比较
negation → 某个否定条件
condition → 某个条件
question_target → 某个问题目标
cyber_security_term → 某个安全问题
```

### 17.3 language = auto

如果 `language = "auto"`：

```text
1. 若 question 中包含中文字符，使用中文 generalization。
2. 否则使用 English generalization。
```

判断中文字符可使用 Unicode 范围：

```text
\u4e00-\u9fff
```

### 17.4 Group generalize

如果 unit 是 group unit，不做复杂整句重写。

对 group 中每个 span 分别按 type 泛化。

示例：

```text
Tom has 3 apples and buys 2 more.
```

number_set generalize：

```text
Tom has some number apples and buys some number more.
```

这种句子不一定最自然，但可复现、可检查、offset 稳定。

---

## 18. CLI 脚本要求

新增或实现：

```text
scripts/04_build_ablated_questions.py
```

命令格式：

```bash
python scripts/04_build_ablated_questions.py \
  --input data/processed/ablation_units.jsonl \
  --output data/processed/ablated_questions.jsonl
```

支持参数：

```text
--input
--output
--language auto/en/zh
--ablation-types delete,generalize
```

默认值：

```text
--language auto
--ablation-types delete,generalize
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 build_ablated_question_records。
4. 使用 validate_ablated_question_record 校验输出。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印 ablated question 统计。
```

---

## 19. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_input_questions
num_input_units
num_output_ablations
num_skipped_empty
num_skipped_unchanged
num_skipped_overlap
ablation_type_counts
unit_scope_counts
group_type_counts
language
ablation_types
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 20. 测试要求

新增：

```text
tests/test_question_ablations.py
```

至少覆盖以下测试。

### 20.1 delete single unit

输入：

```text
Tom has 3 apples.
```

unit 指向 `3`。

验证 delete 后不包含 `3`。

### 20.2 delete group unit

输入：

```text
Tom has 3 apples and buys 2 more.
```

unit 包含 `3` 和 `2`。

验证 delete 后不包含 `3` 和 `2`。

### 20.3 delete 只处理目标 occurrence

输入：

```text
Tom has 3 apples and buys 3 more apples.
```

single unit 指向第一个 `3`。

验证 delete 后只删除第一个 `3`，第二个 `3` 保留。

### 20.4 generalize number English

输入：

```text
Tom has 3 apples.
```

unit 指向 `3`。

验证 generalize 后包含：

```text
some number
```

### 20.5 generalize object English

输入：

```text
Tom has 3 apples.
```

unit 指向 `apples`。

验证 generalize 后包含：

```text
some object
```

### 20.6 generalize number Chinese

输入：

```text
小明有3个苹果。
```

unit 指向 `3`。

验证 generalize 后包含：

```text
某个数量
```

### 20.7 generalize object Chinese

输入：

```text
小明有3个苹果。
```

unit 指向 `苹果`。

验证 generalize 后包含：

```text
某个对象
```

### 20.8 group generalize

输入：

```text
Tom has 3 apples and buys 2 more.
```

unit 包含两个 number spans。

验证 generalize 后包含两个：

```text
some number
```

### 20.9 unsupported ablation_type raises ValueError

传入：

```text
mask
```

应抛出 ValueError。

说明：

```text
mask 不在 Sprint 1C 实现，后续由 masked question construction 阶段处理。
```

### 20.10 overlapping spans skipped

构造一个 unit，其中两个 spans overlap。

验证 build_ablated_question_records 会跳过对应输出，不抛出未处理异常。

### 20.11 empty ablated question skipped

构造一个 unit 覆盖整个 question，delete 后为空。

验证该输出被跳过。

### 20.12 unchanged ablated question skipped

构造一个 generalize 后与原问题完全相同的情况。

验证该输出被跳过。

### 20.13 schema validation

验证每条输出 record 都能通过：

```python
validate_ablated_question_record(record)
```

### 20.14 CLI smoke test

使用临时输入 jsonl 运行 CLI。

验证：

```text
输出文件存在
每行是合法 JSON object
至少包含 delete / generalize 两类 ablation
```

---

## 21. 本 sprint 不做

不要实现：

```text
NLI scoring
semantic necessity label
masked question construction
question recovery
recoverability scoring
label building
incomplete-question reasoning
answer stability
baseline CoT
hidden states
attention maps
trajectory stability
attention anchor labeling
attention guidance
probe training
真实模型调用
```

不要新增：

```text
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/baseline_cot.py
```

不要生成：

```text
data/processed/nli_scores.jsonl
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
```

---

## 22. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/04_build_ablated_questions.py --input data/processed/ablation_units.jsonl --output data/processed/ablated_questions.jsonl
python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

注意：

```text
本 sprint 不自动运行 Sprint 1B。
如果 data/processed/ablation_units.jsonl 不存在，应停止并报告缺失输入。
```

---

## 23. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1C 已完成。
下一步建议是 Sprint 1D：NLI Semantic Necessity Stub。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1C | 完成 | Ablated question construction |
```

当前可运行命令中增加：

```bash
python scripts/04_build_ablated_questions.py --input data/processed/ablation_units.jsonl --output data/processed/ablated_questions.jsonl
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
ablated question construction: passed
```

当前关键文件状态中补充：

```text
已完成：
- src/recover_attention/question_ablations.py
- scripts/04_build_ablated_questions.py
- tests/test_question_ablations.py
- data/processed/ablated_questions.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/nli_scoring.py
- scripts/05_run_nli_scoring.py
- tests/test_nli_scoring.py
- data/processed/nli_scores.jsonl
```

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 24. 验收标准

本 sprint 完成后应满足：

```text
1. src/recover_attention/question_ablations.py 存在。
2. scripts/04_build_ablated_questions.py 存在。
3. tests/test_question_ablations.py 存在。
4. schemas.py 中存在 validate_ablated_question_record。
5. apply_ablation_to_unit 支持 delete / generalize。
6. delete 支持 single unit。
7. delete 支持 group unit。
8. delete 只处理目标 occurrence，不使用全局字符串替换。
9. generalize 支持 English / Chinese。
10. group generalize 对每个 span 分别泛化。
11. 多 span unit 按 start 从大到小处理。
12. overlapping spans 会被跳过并统计。
13. empty / unchanged ablated question 会被跳过。
14. data/processed/ablated_questions.jsonl 已生成。
15. 输出 records 通过 validate_ablated_question_record。
16. python -m pytest -q 通过。
17. smoke test 通过。
18. 未生成 nli_scores.jsonl。
19. 未生成 masked_questions.jsonl。
20. 未实现 NLI / mask / recovery / trajectory / attention guidance / probe。
21. 未新增 baseline CoT 相关模块。
22. PROGRESS.md 保持短版状态索引。
23. data/processed/ablated_questions.jsonl 符合 docs/skill/ablated_questions_interface.md。
24. validate_ablated_question_record 与 docs/skill/ablated_questions_interface.md 中的 schema 约束一致。
```

---

## 25. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. ablated question 数量统计
6. PROGRESS.md 更新摘要
7. 遗留问题
8. 下一步建议
```

不要自动开始 Sprint 1D。
