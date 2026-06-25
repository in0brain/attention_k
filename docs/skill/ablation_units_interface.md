# docs/skill/ablation_units_interface.md

# Ablation Units Interface

本文件定义 `data/processed/ablation_units.jsonl` 的稳定接口。

该文件是 Sprint 1B 的输出，也是 Sprint 1C 及后续阶段的输入契约。

---

# 1. 位置

```text
data/processed/ablation_units.jsonl
```

---

# 2. 作用

`ablation_units.jsonl` 用来描述：

```text
对一个问题进行语义扰动时，一次应该作用于哪些 span。
```

一个 ablation unit 可以是：

```text
single unit:
  只包含一个 candidate span。

group unit:
  包含多个语义相关的 candidate spans。
```

后续阶段不再直接猜测“应该删哪个 span”，而是读取这里定义好的 unit。

---

# 3. 上下游关系

当前 v0 pipeline 中的位置：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
→ nli_scores.jsonl
```

其中：

```text
candidate_spans.jsonl:
  由 Sprint 1A 生成，负责列出候选 span。

ablation_units.jsonl:
  由 Sprint 1B 生成，负责把 span 组织成 single/group units。

ablated_questions.jsonl:
  由 Sprint 1C 生成，负责对每个 unit 构造 delete/generalize 问题。
```

---

# 4. 顶层 Record Schema

`ablation_units.jsonl` 每一行是一个 JSON object。

每条 record 对应一个原始问题。

必须包含：

<!-- required_fields:ablation_unit -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
id
question
units
```

字段含义：

```text
id:
  原始问题 ID。

question:
  原始问题文本。

units:
  当前问题对应的 ablation unit 列表。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "units": []
}
```

---

# 5. Unit Schema

`units` 中的每个元素是一个 ablation unit。

每个 unit 必须包含：

```text
unit_id
unit_scope
group_type
span_ids
spans
reason
```

字段含义：

```text
unit_id:
  当前 question 内部的 unit 编号，例如 unit_001。

unit_scope:
  single 或 group。

group_type:
  描述该 unit 的构造类型。
  single unit 使用 single。
  group unit 使用 number_set / entity_set / object_set 等。

span_ids:
  当前 unit 包含的 span_id 列表。

spans:
  当前 unit 包含的 span 完整信息列表。

reason:
  构造该 unit 的简短原因说明。
```

---

# 6. Span Schema

`spans` 中的每个元素必须包含：

```text
span_id
text
type
start
end
```

字段含义：

```text
span_id:
  candidate span 的编号。

text:
  span 在 question 中的原始文本。

type:
  span 类型，例如 number / entity / object。

start:
  span 在 question 中的起始字符位置。

end:
  span 在 question 中的结束字符位置。
```

位置规则使用 Python 切片语义：

```python
question[start:end] == text
```

---

# 7. 完整示例

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
      "unit_id": "unit_002",
      "unit_scope": "single",
      "group_type": "single",
      "span_ids": ["span_002"],
      "spans": [
        {
          "span_id": "span_002",
          "text": "2",
          "type": "number",
          "start": 26,
          "end": 27
        }
      ],
      "reason": "single candidate span"
    },
    {
      "unit_id": "unit_003",
      "unit_scope": "group",
      "group_type": "number_set",
      "span_ids": ["span_001", "span_002"],
      "spans": [
        {
          "span_id": "span_001",
          "text": "3",
          "type": "number",
          "start": 8,
          "end": 9
        },
        {
          "span_id": "span_002",
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

---

# 8. unit_scope 规则

`unit_scope` 只能是：

```text
single
group
```

规则：

```text
unit_scope = single:
  span_ids 长度必须为 1。
  spans 长度必须为 1。
  group_type 必须为 single。

unit_scope = group:
  span_ids 长度必须大于等于 2。
  spans 长度必须大于等于 2。
  group_type 不能为 single。
```

---

# 9. group_type 规则

当前支持的 `group_type`：

```text
single
repeated_surface
number_set
entity_set
object_set
cyber_security_term_set
```

含义：

```text
single:
  单个 span 构成的 unit。

repeated_surface:
  同一个 text 在问题中出现至少两次。

number_set:
  问题中所有或部分 number span 构成的集合。

entity_set:
  问题中所有或部分 entity span 构成的集合。

object_set:
  问题中所有或部分 object span 构成的集合。

cyber_security_term_set:
  问题中所有或部分 cyber_security_term span 构成的集合。
```

当前阶段不生成任意组合。

禁止生成：

```text
所有 span 的 2^n 子集枚举
```

---

# 10. span_ids 与 spans 的一致性

必须满足：

```text
len(span_ids) == len(spans)
```

并且顺序一致：

```python
span_ids[i] == spans[i]["span_id"]
```

示例：

```json
"span_ids": ["span_001", "span_003"],
"spans": [
  {"span_id": "span_001", "...": "..."},
  {"span_id": "span_003", "...": "..."}
]
```

这是合法的。

---

# 11. unit_id 规则

每个 question 内部重新编号：

```text
unit_001
unit_002
unit_003
...
```

要求：

```text
1. unit_id 连续。
2. single units 排在 group units 前面。
3. group units 排在 single units 后面。
4. unit_id 根据最终输出顺序生成。
```

不同 question 之间可以重复使用相同的 unit_id。

例如每个问题都可以从 `unit_001` 开始。

---

# 12. 去重规则

如果多个 group unit 的 `span_ids` 完全相同，只保留一个。

去重 key：

```python
tuple(span_ids)
```

注意：

```text
single unit 不因为 group unit 存在而删除。
```

同一个 span 可以同时出现在：

```text
一个 single unit
一个 group unit
```

例如 `span_001` 可以同时属于：

```text
unit_001: single
unit_006: number_set
```

---

# 13. 允许空 units 吗

允许。

如果某个问题没有 candidate spans，则输出：

```json
{
  "id": "example_0001",
  "question": "Some question text.",
  "units": []
}
```

但是：

```text
如果 units 非空，则每个 unit 必须满足本文件定义的 schema。
```

---

# 14. 与 candidate_spans.jsonl 的关系

`ablation_units.jsonl` 中的 span 信息来自 `candidate_spans.jsonl`。

1B 不重新抽取 span。

1B 只负责组织已有 span：

```text
candidate span
→ single unit
→ group unit
```

不要在 1B 中修改 span 的：

```text
text
type
start
end
```

如果发现 `question[start:end] != text`，应视为上游数据或 schema 错误。

---

# 15. 与 ablated_questions.jsonl 的关系

Sprint 1C 会读取 `ablation_units.jsonl`。

1C 会把每个 unit 转换成两类 ablated question：

```text
delete
generalize
```

字段来源关系：

```text
1B record.id
→ 1C record.id

1B record.question
→ 1C record.original_question

1B unit.unit_id
→ 1C record.unit_id

1B unit.unit_scope
→ 1C record.unit_scope

1B unit.group_type
→ 1C record.group_type

1B unit.span_ids
→ 1C record.span_ids

1B unit.spans
→ 1C record.spans
```

1C 自己新增：

```text
ablation_id
ablation_type
ablated_question
```

---

# 16. Validator

`src/recover_attention/schemas.py` 中应提供：

```python
validate_ablation_unit_record(record: dict) -> None
```

该函数负责检查本文件定义的 schema。

至少检查：

```text
1. 顶层字段 id / question / units。
2. id 是非空 string。
3. question 是非空 string。
4. units 是 list。
5. unit 字段完整性。
6. span 字段完整性。
7. span_ids 与 spans 长度一致。
8. span_ids 与 spans 中 span_id 顺序一致。
9. unit_scope 与 group_type 一致。
10. single/group 的 span 数量约束。
11. start/end 是合法 int。
12. question[start:end] == text。
```

如果检查失败，应抛出异常，不要静默修复。

---

# 17. 文件与模块职责

## 17.1 data artifact

```text
data/processed/ablation_units.jsonl
```

职责：

```text
保存 Sprint 1B 输出的 ablation units。
作为 Sprint 1C 的输入。
```

---

## 17.2 module

```text
src/recover_attention/ablation_units.py
```

职责：

```text
从 candidate span records 构造 ablation unit records。
生成 single units。
根据规则生成 group units。
负责 unit 排序、去重和 unit_id 分配。
不负责 delete / generalize / mask。
```

推荐核心函数：

```python
build_ablation_units(candidate_span_record: dict) -> dict

build_ablation_unit_records(candidate_span_records: list[dict]) -> list[dict]
```

---

## 17.3 schema

```text
src/recover_attention/schemas.py
```

职责：

```text
提供 validate_ablation_unit_record。
保证 ablation_units.jsonl 的 record 满足本接口。
```

---

## 17.4 CLI

```text
scripts/03_build_ablation_units.py
```

职责：

```text
读取 data/processed/candidate_spans.jsonl。
调用 ablation_units.py。
校验输出 record。
写入 data/processed/ablation_units.jsonl。
打印统计信息。
```

推荐命令：

```bash
python scripts/03_build_ablation_units.py \
  --input data/processed/candidate_spans.jsonl \
  --output data/processed/ablation_units.jsonl
```

---

## 17.5 tests

```text
tests/test_ablation_units.py
tests/test_schemas.py
```

职责：

```text
验证 ablation unit 构造逻辑。
验证 schema validator。
验证 CLI smoke test。
```

---

# 18. 统计信息

`03_build_ablation_units.py` 推荐打印：

```text
num_questions
num_candidate_spans
num_units
num_single_units
num_group_units
group_type_counts
questions_with_zero_units
```

统计只打印到 stdout，不要求写入 json 文件。

---

# 19. 本接口不包含的内容

本接口不定义：

```text
ablated_question
ablation_type
NLI score
semantic necessity label
masked question
recover output
recoverability score
attention anchor label
probe label
```

这些属于后续阶段。

---

# 20. 最小验收标准

一个合格的 `ablation_units.jsonl` 至少满足：

```text
1. 每行是 JSON object。
2. 每条 record 有 id / question / units。
3. units 是 list。
4. 每个 unit 有 unit_id / unit_scope / group_type / span_ids / spans / reason。
5. 每个 span 有 span_id / text / type / start / end。
6. span_ids 与 spans 顺序一致。
7. question[start:end] == text。
8. single unit 与 group unit 的数量约束正确。
9. group_type 与 unit_scope 一致。
10. validate_ablation_unit_record 全部通过。
```
