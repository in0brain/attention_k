# Sprint 1B：Ablation Unit Construction

## 1. 目标

本 sprint 实现 ablation unit construction 的最小可运行版本。

目标文件流：

```text
data/processed/candidate_spans.jsonl
→ data/processed/ablation_units.jsonl
```

本 sprint 不直接构造 ablated questions。

本 sprint 只解决一个问题：

```text
一次 ablation 应该作用于什么对象？
```

这个对象称为：

```text
ablation unit
```

ablation unit 可以是：

```text
1. 单个 candidate span
2. 一组有语义理由的 candidate spans
```

---

## 2. 设计理由

原始设计中，ablated question construction 默认对单个 span 做 delete / generalize。

但单 span ablation 存在一个问题：

```text
某个 span 即使被删除，模型或 NLI 判断仍可能从上下文、平行结构、世界知识或其他 span 中恢复它的信息。
```

例如：

```text
英国 + 首都
法国 + [MASK]
```

即使删除 `首都` 或某个实体，模型仍可能根据上下文猜出 `法国 + 首都 = 巴黎`。

因此，单 span ablation 只能测：

```text
局部边际贡献
```

不能充分测：

```text
集合依赖
上下文冗余
平行结构提示
同类型信息共同约束
```

所以本 sprint 引入 ablation unit。

原则是：

```text
single units 保留可解释性。
group units 补充检测上下文冗余和集合依赖。
```

本 sprint 不枚举所有 span 组合。

原因：

```text
1. 所有组合会造成组合爆炸。
2. 大量组合没有清晰语义解释。
3. v0 阶段需要可检查、可复现、可解释的工程 baseline。
```

因此，本 sprint 只构造有明确语义动机的 group units。

---

## 3. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1A 已完成。
data/processed/candidate_spans.jsonl 已存在。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/candidate_spans.jsonl
Please run Sprint 1A first.
```

不要自动回头执行 Sprint 1A。

---

## 4. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/skill/ablation_units_interface.md
docs/skill/label_schema.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
data/processed/candidate_spans.jsonl
```

如果以下文件存在，也读取：

```text
src/recover_attention/ablation_units.py
tests/test_ablation_units.py
```

不要读取：

```text
docs/reference/*
```

除非当前用户指令明确要求。

---

## 5. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/ablation_units.py
scripts/03_build_ablation_units.py
tests/test_ablation_units.py
src/recover_attention/schemas.py
tests/test_schemas.py
data/processed/ablation_units.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
src/recover_attention/ablation_units.py
scripts/03_build_ablation_units.py
tests/test_ablation_units.py
docs/progress/sprint_1_history.md
```

说明：

```text
本 sprint 新增 ablation_units.jsonl 这种中间 artifact，
因此允许在 schemas.py 中新增 validate_ablation_unit_record。
```

---

## 6. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/skill/*
docs/reference/*
docs/codex_tasks/*
configs/*
requirements.txt
pyproject.toml
.gitignore
src/recover_attention/data_io.py
src/recover_attention/prepare_data.py
src/recover_attention/candidate_extraction.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 7. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1. 是否确认当前进入 Sprint 1B：Ablation Unit Construction
2. data/processed/candidate_spans.jsonl 是否存在
3. candidate span record 数量
4. candidate 总数
5. 当前 schemas.py 是否已有 validate_ablation_unit_record
6. 本次是否需要新增 validate_ablation_unit_record
7. 本次计划支持的 unit types
8. 本次计划生成的输出文件
9. 本次是否明确不生成 ablated_questions.jsonl
10. 本次是否明确不做 NLI / mask / recovery
```

如发现冲突，按 AGENTS.md 中定义的冲突优先级处理，并在 Preflight 中报告。

---

## 8. 输入文件

```text
data/processed/candidate_spans.jsonl
```

输入 record 示例：

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "candidates": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    },
    {
      "span_id": "span_002",
      "text": "apples",
      "type": "object",
      "start": 10,
      "end": 16
    }
  ]
}
```

---

## 9. 输出文件

```text
data/processed/ablation_units.jsonl
```

输出 record 示例：

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
    },
    {
      "unit_id": "unit_006",
      "unit_scope": "group",
      "group_type": "number_set",
      "span_ids": ["span_001", "span_003"],
      "spans": [
        {
          "span_id": "span_001",
          "text": "3",
          "type": "number",
          "start": 8,
          "end": 9
        },
        {
          "span_id": "span_003",
          "text": "2",
          "type": "number",
          "start": 26,
          "end": 27
        }
      ],
      "reason": "all number spans in the question"
    }
  ]
}
```

每条输出 record 必须通过：

```python
validate_ablation_unit_record(record)
```

### 9.1 Ablation Unit Interface Contract

Sprint 1B 的输出必须符合 `docs/skill/ablation_units_interface.md`。

`data/processed/ablation_units.jsonl` 是 Sprint 1C 的稳定输入。

每条 record 必须包含：

- id
- question
- units

每个 unit 必须包含：

- unit_id
- unit_scope
- group_type
- span_ids
- spans
- reason

每个 span 必须包含：

- span_id
- text
- type
- start
- end

如果实现、测试、输出样例与 `docs/skill/ablation_units_interface.md` 冲突，
以 `docs/skill/ablation_units_interface.md` 为准。

---

## 10. Ablation Unit Schema 要求

在 `src/recover_attention/schemas.py` 中新增：

```python
validate_ablation_unit_record(record: dict) -> None
```

record 顶层字段：

```text
id
question
units
```

每个 unit 字段：

```text
unit_id
unit_scope
group_type
span_ids
spans
reason
```

字段要求：

```text
id: str, non-empty
question: str, non-empty
units: list
unit_id: str, non-empty
unit_scope: one of single / group
group_type: str, non-empty
span_ids: list[str], non-empty
spans: list[dict], non-empty
reason: str, non-empty
```

每个 span 必须包含：

```text
span_id
text
type
start
end
```

并且必须保证：

```python
question[start:end] == text
```

约束：

```text
1. unit_scope = single 时，span_ids 长度必须为 1。
2. unit_scope = group 时，span_ids 长度必须 >= 2。
3. spans 长度必须等于 span_ids 长度。
4. span_ids 顺序必须与 spans 中 span_id 顺序一致。
5. group_type = single 时，unit_scope 必须为 single。
6. group_type != single 时，unit_scope 必须为 group。
```

---

## 11. 支持的 group_type

本 sprint 支持：

```text
single
repeated_surface
number_set
entity_set
object_set
cyber_security_term_set
```

暂不实现：

```text
number_object_pair
entity_relation_pair
cause_effect_pair
all_pairs
all_combinations
```

原因：

```text
1. pair / tuple 需要更强的关系抽取能力。
2. all combinations 会组合爆炸。
3. v0 阶段先构造可解释、可控的 group units。
```

---

## 12. 生成规则

### 12.1 single units

每个 candidate span 都生成一个 single unit。

示例：

```json
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
```

single units 必须保留。

原因：

```text
single ablation 用于分析单个 span 的局部边际贡献。
```

---

### 12.2 repeated_surface

如果同一个 `text` 在 candidates 中出现至少 2 次，生成一个 repeated_surface group。

示例：

```text
Tom has 3 apples and buys 3 more apples.
```

如果两个 `3` 都被抽取，应生成：

```json
{
  "unit_scope": "group",
  "group_type": "repeated_surface",
  "span_ids": ["span_001", "span_004"],
  "reason": "same surface text appears multiple times"
}
```

注意：

```text
只对完全相同的 text 分组。
区分大小写可以先保持原样。
不要做复杂 normalization。
```

---

### 12.3 number_set

如果一个问题中存在至少 2 个 `type = number` 的 spans，生成一个 number_set group。

原因：

```text
数学题中的多个数字通常共同约束答案。
只删除一个数字可能不足以暴露问题对数值集合的依赖。
```

---

### 12.4 entity_set

如果一个问题中存在至少 2 个 `type = entity` 的 spans，生成一个 entity_set group。

原因：

```text
实体之间可能构成平行结构、对照结构或关系集合。
只删除单个实体可能仍能从上下文恢复。
```

---

### 12.5 object_set

如果一个问题中存在至少 2 个 `type = object` 的 spans，生成一个 object_set group。

原因：

```text
对象集合可能影响问题中的计数对象、比较对象或安全场景中的操作对象。
```

---

### 12.6 cyber_security_term_set

如果一个问题中存在至少 2 个 `type = cyber_security_term` 的 spans，生成一个 cyber_security_term_set group。

原因：

```text
安全术语经常以组合方式表达漏洞类型、攻击方式、影响结果或防护条件。
```

---

## 13. Group Budget 规则

为防止 group 爆炸，本 sprint 必须限制 group units。

规则：

```text
1. group unit 至少包含 2 个 spans。
2. group unit 最多包含 8 个 spans。
3. 每个问题最多生成 10 个 group units。
4. single units 不计入 group budget。
5. 如果某个 group 超过 8 个 spans，保留 start 最靠前的 8 个 spans。
6. 如果 group 数量超过 10 个，按 group priority 截断。
```

group priority：

```text
repeated_surface
number_set
entity_set
object_set
cyber_security_term_set
```

同一 group_type 内按第一个 span 的 start 从小到大排序。

---

## 14. 去重与排序

生成 units 后需要：

```text
1. 对相同 span_ids 组合去重。
2. single units 永远保留。
3. group units 按 group priority 排序。
4. 最终 units 按以下顺序排列：
   - 先 single units
   - 后 group units
5. single units 内按第一个 span 的 start 从小到大排序。
6. group units 内按 group priority 和第一个 span 的 start 排序。
7. 最终重新编号 unit_id：unit_001, unit_002, ...
```

span_ids 的顺序应按 span 的 start 从小到大排列。

---

## 15. 核心接口要求

在 `src/recover_attention/ablation_units.py` 中实现：

```python
build_ablation_units(
    candidate_span_record: dict,
    max_group_size: int = 8,
    max_group_units: int = 10,
) -> dict
```

输入：

```text
单条 candidate span record
```

输出：

```text
单条 ablation unit record
```

同时实现：

```python
build_ablation_unit_records(
    candidate_span_records: list[dict],
    max_group_size: int = 8,
    max_group_units: int = 10,
) -> list[dict]
```

要求：

```text
1. 不修改原始 candidate_span_records。
2. 输出 record 必须通过 validate_ablation_unit_record。
3. 如果某条 question 没有 candidates，units 应为空列表，但 record 仍可输出。
4. 不生成 ablated_question。
5. 不生成 NLI label。
6. 不生成 mask 或 recovery 相关字段。
```

---

## 16. CLI 脚本要求

新增或实现：

```text
scripts/03_build_ablation_units.py
```

命令格式：

```bash
python scripts/03_build_ablation_units.py \
  --input data/processed/candidate_spans.jsonl \
  --output data/processed/ablation_units.jsonl
```

支持参数：

```text
--input
--output
--max-group-size
--max-group-units
```

默认值：

```text
--max-group-size 8
--max-group-units 10
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 build_ablation_unit_records。
4. 使用 validate_ablation_unit_record 校验输出。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印 ablation unit 统计。
```

---

## 17. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_questions
num_candidate_spans
num_units
num_single_units
num_group_units
group_type_counts
questions_with_zero_units
max_group_size
max_group_units
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 18. 测试要求

新增：

```text
tests/test_ablation_units.py
```

至少覆盖以下测试。

### 18.1 single units

输入一个含 3 个 candidates 的 record。

验证：

```text
每个 candidate 都生成一个 single unit。
single unit 数量等于 candidate 数量。
```

### 18.2 number_set

输入至少 2 个 `type = number` 的 spans。

验证生成：

```text
group_type = number_set
```

### 18.3 entity_set

输入至少 2 个 `type = entity` 的 spans。

验证生成：

```text
group_type = entity_set
```

### 18.4 object_set

输入至少 2 个 `type = object` 的 spans。

验证生成：

```text
group_type = object_set
```

### 18.5 repeated_surface

输入两个 text 相同但 span_id 不同的 spans。

验证生成：

```text
group_type = repeated_surface
```

### 18.6 cyber_security_term_set

输入至少 2 个 `type = cyber_security_term` 的 spans。

验证生成：

```text
group_type = cyber_security_term_set
```

### 18.7 max_group_size

构造一个包含超过 8 个 number spans 的 record。

验证：

```text
number_set group 中 spans 数量 <= max_group_size
```

### 18.8 max_group_units

构造一个会产生多个 group units 的 record。

验证：

```text
group unit 数量 <= max_group_units
```

### 18.9 unit_id 连续编号

验证最终 unit_id 连续：

```text
unit_001
unit_002
unit_003
...
```

### 18.10 schema validation

验证每条输出 record 都能通过：

```python
validate_ablation_unit_record(record)
```

### 18.11 no candidates

输入：

```json
{
  "id": "empty_001",
  "question": "Empty example?",
  "candidates": []
}
```

验证输出：

```json
{
  "id": "empty_001",
  "question": "Empty example?",
  "units": []
}
```

并且能通过 schema validation。

### 18.12 CLI smoke test

使用临时输入 jsonl 运行 CLI，验证：

```text
输出文件存在
每行是合法 JSON object
至少包含 single unit
```

---

## 19. 本 sprint 不做

不要实现：

```text
ablated question construction
delete ablation
generalize ablation
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
src/recover_attention/question_ablations.py
scripts/04_build_ablated_questions.py
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/baseline_cot.py
```

---

## 20. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/03_build_ablation_units.py --input data/processed/candidate_spans.jsonl --output data/processed/ablation_units.jsonl
python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

注意：

```text
本 sprint 不自动运行 Sprint 1A。
如果 data/processed/candidate_spans.jsonl 不存在，应停止并报告缺失输入。
```

---

## 21. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1B 已完成。
下一步建议是 Sprint 1C：Ablated Question Construction。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1B | 完成 | Ablation unit construction |
```

当前可运行命令中增加：

```bash
python scripts/03_build_ablation_units.py --input data/processed/candidate_spans.jsonl --output data/processed/ablation_units.jsonl
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
ablation unit construction: passed
```

当前关键文件状态中补充：

```text
已完成：
- src/recover_attention/ablation_units.py
- scripts/03_build_ablation_units.py
- tests/test_ablation_units.py
- data/processed/ablation_units.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/question_ablations.py
- scripts/04_build_ablated_questions.py
- tests/test_question_ablations.py
- data/processed/ablated_questions.jsonl
```

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 22. 验收标准

本 sprint 完成后应满足：

```text
1. src/recover_attention/ablation_units.py 存在。
2. scripts/03_build_ablation_units.py 存在。
3. tests/test_ablation_units.py 存在。
4. schemas.py 中存在 validate_ablation_unit_record。
5. single units 能为每个 candidate span 生成。
6. repeated_surface group 能生成。
7. number_set group 能生成。
8. entity_set group 能生成。
9. object_set group 能生成。
10. cyber_security_term_set group 能生成。
11. max_group_size 生效。
12. max_group_units 生效。
13. unit_id 连续编号。
14. data/processed/ablation_units.jsonl 已生成。
15. 输出 records 通过 validate_ablation_unit_record。
16. python -m pytest -q 通过。
17. smoke test 通过。
18. 未生成 ablated_questions.jsonl。
19. 未实现 NLI / mask / recovery / trajectory / attention guidance / probe。
20. 未新增 baseline CoT 相关模块。
21. PROGRESS.md 保持短版状态索引。
22. data/processed/ablation_units.jsonl 符合 docs/skill/ablation_units_interface.md。
23. validate_ablation_unit_record 与 docs/skill/ablation_units_interface.md 中的 schema 约束一致。
```

---

## 23. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. ablation unit 数量统计
6. PROGRESS.md 更新摘要
7. 遗留问题
8. 下一步建议
```

不要自动开始 Sprint 1C。
