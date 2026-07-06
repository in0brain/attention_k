# Sprint 0C：schema 校验

## 1. 目标

本 sprint 只完成基础 schema 校验。

目标是：

```text
为 v0/v1 pipeline 的 jsonl 中间文件建立轻量校验函数
为 schema 校验函数建立 pytest
保证后续脚本可以复用这些校验函数
```

本 sprint 不实现数据准备脚本，不实现 span 抽取，不实现 ablation，不实现 NLI。

---

## 2. 本次允许修改的文件

```text
src/recover_attention/schemas.py
tests/test_schemas.py
PROGRESS.md
```

如果 `tests/` 目录不存在，可以创建。

---

## 3. 本次禁止修改的文件

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/*
docs/codex_tasks/*
src/recover_attention/data_io.py
src/recover_attention/candidate_extraction.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
src/recover_attention/utils.py
scripts/*.py
configs/*.yaml
data/*
```

---

## 4. 输入文件

无强制输入。

本 sprint 只实现 Python 函数和测试。

---

## 5. 输出文件

```text
src/recover_attention/schemas.py
tests/test_schemas.py
PROGRESS.md
```

---

## 6. 实现要求

### 6.1 `src/recover_attention/schemas.py`

实现轻量 schema 校验函数。

不要使用 pydantic。

不要引入大型依赖。

优先使用普通 Python 函数。

需要实现：

```python
validate_question_record(record: dict) -> None
validate_candidate_span_record(record: dict) -> None
validate_ablated_question_record(record: dict) -> None
validate_nli_score_record(record: dict) -> None
validate_masked_question_record(record: dict) -> None
validate_recover_output_record(record: dict) -> None
validate_recover_score_record(record: dict) -> None
```

所有函数要求：

```text
1. 输入必须是 dict。
2. 校验通过时返回 None。
3. 校验失败时抛出 ValueError。
4. ValueError 信息必须说明缺失字段或非法字段。
5. 不要静默返回 False。
6. 不要吞掉异常。
7. 不要做复杂语义判断，只做字段、类型、枚举值的基础校验。
```

---

## 7. 各 schema 规则

### 7.1 question record

示例：

```json
{
  "id": "gsm8k_0001",
  "dataset": "gsm8k",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "gold_answer": "5"
}
```

必需字段：

```text
id: str, non-empty
dataset: str, non-empty
question: str, non-empty
gold_answer: str, non-empty
```

---

### 7.2 candidate span record

示例：

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
    }
  ]
}
```

record 必需字段：

```text
id: str, non-empty
question: str, non-empty
candidates: list
```

candidate 必需字段：

```text
span_id: str, non-empty
text: str, non-empty
type: str, one of allowed span types
start: int, >= 0
end: int, > start
```

允许的 span type：

```text
number
entity
operation
relation
question_target
comparison
negation
condition
cyber_security_term
```

注意：

```text
candidates 可以是空 list。
但如果 candidates 中存在元素，每个元素必须合法。
```

---

### 7.3 ablated question record

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "ablated_question": "Tom has some apples and buys 2 more. How many apples does he have now?"
}
```

必需字段：

```text
id: str, non-empty
span_id: str, non-empty
span_text: str, non-empty
span_type: str, one of allowed span types
ablation_type: str, one of delete/generalize/mask
original_question: str, non-empty
ablated_question: str, non-empty
```

允许的 ablation type：

```text
delete
generalize
mask
```

---

### 7.4 nli score record

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_to_ablated": "entailment",
  "ablated_to_original": "neutral",
  "semantic_necessity_label": "Information Loss"
}
```

必需字段：

```text
id: str, non-empty
span_id: str, non-empty
span_text: str, non-empty
span_type: str, one of allowed span types
ablation_type: str, one of delete/generalize/mask
original_to_ablated: str, one of entailment/neutral/contradiction
ablated_to_original: str, one of entailment/neutral/contradiction
semantic_necessity_label: str, one of allowed semantic necessity labels
```

允许的 NLI directional label：

```text
entailment
neutral
contradiction
```

允许的 semantic necessity label：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

---

### 7.5 masked question record

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?"
}
```

必需字段：

```text
id: str, non-empty
span_id: str, non-empty
span_text: str, non-empty
span_type: str, one of allowed span types
original_question: str, non-empty
masked_question: str, non-empty
```

---

### 7.6 recover output record

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "sample_id": 0,
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
  "recovered_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "recoverable": "yes",
  "confidence": 0.82,
  "reason": "The missing number is likely recoverable from the original context pattern."
}
```

必需字段：

```text
id: str, non-empty
span_id: str, non-empty
sample_id: int, >= 0
masked_question: str, non-empty
recovered_question: str
recoverable: str, one of yes/no/uncertain
confidence: int or float, 0 <= confidence <= 1
reason: str
```

允许的 recoverable 值：

```text
yes
no
uncertain
```

---

### 7.7 recover score record

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "recoverability_label": "Non-recoverable",
  "confidence_mean": 0.41,
  "recovery_consistency": 0.22,
  "misleading_recovery": false
}
```

必需字段：

```text
id: str, non-empty
span_id: str, non-empty
recoverability_label: str, one of allowed recoverability labels
confidence_mean: int or float, 0 <= confidence_mean <= 1
recovery_consistency: int or float, 0 <= recovery_consistency <= 1
misleading_recovery: bool
```

允许的 recoverability label：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

---

## 8. 建议实现方式

可以在 `schemas.py` 中定义常量：

```python
ALLOWED_SPAN_TYPES = {
    "number",
    "entity",
    "operation",
    "relation",
    "question_target",
    "comparison",
    "negation",
    "condition",
    "cyber_security_term",
}

ALLOWED_ABLATION_TYPES = {"delete", "generalize", "mask"}
ALLOWED_NLI_LABELS = {"entailment", "neutral", "contradiction"}
ALLOWED_SEMANTIC_NECESSITY_LABELS = {
    "Equivalent",
    "Information Loss",
    "Added Assumption",
    "Non-equivalent",
}
ALLOWED_RECOVERABLE_VALUES = {"yes", "no", "uncertain"}
ALLOWED_RECOVERABILITY_LABELS = {
    "Recoverable",
    "Partially Recoverable",
    "Non-recoverable",
    "Misleading Recovery",
}
```

可以实现内部 helper：

```python
_require_dict(record, name)
_require_fields(record, fields, name)
_require_non_empty_str(record, field, name)
_require_enum(record, field, allowed, name)
_require_int(record, field, name, min_value=None)
_require_number(record, field, name, min_value=None, max_value=None)
_require_bool(record, field, name)
```

但不要过度工程化。

---

## 9. 测试要求

创建 `tests/test_schemas.py`。

至少测试：

```text
1. valid question record passes.
2. question record missing field raises ValueError.
3. question record empty question raises ValueError.
4. valid candidate span record passes.
5. candidate span with invalid span type raises ValueError.
6. candidate span with end <= start raises ValueError.
7. valid ablated question record passes.
8. ablated question with invalid ablation_type raises ValueError.
9. valid nli score record passes.
10. nli score with invalid semantic_necessity_label raises ValueError.
11. valid masked question record passes.
12. valid recover output record passes.
13. recover output with confidence > 1 raises ValueError.
14. valid recover score record passes.
15. recover score with invalid recoverability_label raises ValueError.
```

---

## 10. 必须运行的命令

完成后运行：

```bash
python -m pytest -q
```

如果 Sprint 0B 的 smoke test 已存在，也运行：

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

---

## 11. 验收标准

```text
[OK] src/recover_attention/schemas.py 存在
[OK] tests/test_schemas.py 存在
[OK] pytest -q 可以通过
[OK] schema 函数校验失败时会抛出 ValueError
[OK] 没有修改 README.md / AGENTS.md / docs/reasoning-aware-attention-guidance/SKILL.md / docs/reference/*
[OK] 没有实现 Sprint 0D
[OK] PROGRESS.md 已更新
```

---

## 12. 禁止事项

```text
不要实现 prepare_data.py。
不要实现 candidate span extraction。
不要实现 ablated question construction。
不要实现 NLI scoring。
不要实现 masked question construction。
不要实现 question recovery。
不要实现 recoverability scoring。
不要调用模型。
不要下载模型。
不要调用外部 API。
不要训练 probe。
不要做 attention guidance。
不要做 hidden states。
不要做 trajectory analysis。
不要做大规模实验。
不要修改 README.md。
不要修改 AGENTS.md。
不要修改 docs/reasoning-aware-attention-guidance/SKILL.md。
不要修改 docs/reference/*。
```

---

## 13. 完成后必须更新

```text
PROGRESS.md
```

更新内容必须包括：

```text
Sprint 0C 已完成内容
新增或修改文件
运行命令
检查结果
遗留问题
下一步建议：Sprint 0D prepare_data
```

不要自动开始 Sprint 0D。
