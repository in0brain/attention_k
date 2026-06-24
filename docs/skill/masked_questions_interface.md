# Masked Questions Interface

本文件定义 `data/processed/masked_questions.jsonl` 的稳定接口。

该文件是 Sprint 1F 的输出，也是后续 recoverability（mask-recover）阶段的输入契约。

本接口为 **unit-level**，与 1B / 1C / 1D / 1E 链路保持一致，取代已废弃的旧 span-level masked question schema。

---

# 1. 位置

```text
data/processed/masked_questions.jsonl
```

---

# 2. 作用

`masked_questions.jsonl` 用来保存：

```text
对一个 ablation unit 的全部 span 做 [MASK] 替换后得到的残缺问题。
```

它回答的问题是：

```text
如果遮盖掉这一信息单元，模型在不解题的前提下能否复原原问题？
```

本接口只负责构造 masked 输入。

它不保存 recovery 输出，不保存 recoverability label，不保存 attention guidance。

---

# 3. 上下游关系

当前 v0 pipeline 中的位置：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
→ nli_scores.jsonl
→ semantic_labels.jsonl
→ masked_questions.jsonl
→ recover_outputs.jsonl
→ recover_scores.jsonl
```

其中：

```text
semantic_labels.jsonl:
  由 Sprint 1E 生成，按 (unit, ablation_type) 展开，保存 semantic necessity label。

masked_questions.jsonl:
  由 Sprint 1F 生成，按 unit 去重，对 unit 全部 span 做 mask。

recover_outputs.jsonl:
  后续阶段生成，保存模型对 masked question 的复原结果。
```

---

# 4. 输入与去重规则

输入文件：

```text
data/processed/semantic_labels.jsonl
```

`semantic_labels.jsonl` 中同一个 unit 会出现多条记录（每种 ablation_type 一条，如 delete / generalize）。

mask 与 ablation_type 无关，因此本阶段：

```text
1. 以 (id, unit_id) 为去重键，每个 unit 只产出一条 masked question 记录。
2. 同一 unit 的多条 1E 记录，其 unit_scope / group_type / span_ids / spans / original_question
   必须一致；不一致视为上游错误，应抛出异常，不要静默修复。
3. 同一 unit 的全部来源 semantic_label_id 汇总进 source_semantic_label_ids。
```

---

# 5. 顶层 Record Schema

`masked_questions.jsonl` 每一行是一个 JSON object，对应一个 ablation unit。

必须包含：

```text
mask_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
masked_question
mask_token
num_masks
mask_backend
source_semantic_label_ids
```

可选字段：

```text
semantic_necessity_summary
```

---

# 6. 字段来源

来自 Sprint 1E / `semantic_labels.jsonl`（unit 结构字段，取去重后任一来源记录）：

```text
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
```

由 Sprint 1F 新增：

```text
mask_id
masked_question
mask_token
num_masks
mask_backend
source_semantic_label_ids
semantic_necessity_summary（可选）
```

---

# 7. 字段含义

```text
mask_id:
  当前 masked question 记录的唯一 ID。

id:
  原始问题 ID。

unit_id:
  当前被遮盖的 ablation unit ID。

unit_scope:
  single 或 group。

group_type:
  single / repeated_surface / number_set / entity_set / object_set / cyber_security_term_set。

span_ids:
  当前 unit 包含的 span_id 列表。

spans:
  当前 unit 包含的 span 完整信息列表，offset 指向 original_question。

original_question:
  原始问题文本。

masked_question:
  对 unit 全部 span 做 mask 替换后的问题文本。

mask_token:
  替换 span 时使用的掩码标记，默认 "[MASK]"。

num_masks:
  masked_question 中插入的 mask_token 数量，等于 len(spans)。

mask_backend:
  当前 masking backend。Sprint 1F 只支持 rule_v0。

source_semantic_label_ids:
  生成本记录所依据的全部 1E semantic_label_id 列表。

semantic_necessity_summary:
  可选，只读回溯元数据，见 第 11 节。
```

---

# 8. mask_id 规则

`mask_id` 必须确定、稳定：

```python
mask_id = f"{id}__{unit_id}__mask"
```

示例：

```text
gsm8k_0001__unit_001__mask
gsm8k_0001__unit_003__mask
```

注意：

```text
mask_id 不包含 ablation_type，因为 mask 阶段已经按 unit 去重。
```

---

# 9. Mask 规则

## 9.1 single unit

`single` unit 只有一个 span，替换为一个 mask_token：

```text
original: Tom has 3 apples and buys 2 more. How many apples does he have now?
masked:   Tom has [MASK] apples and buys 2 more. How many apples does he have now?
num_masks = 1
```

## 9.2 group unit

`group` unit 含多个 span，unit 内**每个 span 各替换为一个 mask_token**：

```text
original: Tom has 3 apples and buys 2 more. How many apples does he have now?
masked:   Tom has [MASK] apples and buys [MASK] more. How many apples does he have now?
num_masks = 2
```

含义：

```text
同时遮盖这一信息单元的全部成分。
```

## 9.3 替换算法

```text
1. 取 spans，按 start 降序排序（从右往左替换，避免位移破坏后续 offset）。
2. 逐个 span：masked = masked[:start] + mask_token + masked[end:]。
3. 不做 whitespace 折叠。
4. 替换前检查 unit 内 span 是否重叠，重叠则跳过该 unit。
5. 替换后必须满足 masked_question.count(mask_token) == len(spans)，否则跳过该 unit。
6. masked_question 不应等于 original_question。
```

---

# 10. num_masks 规则

```text
num_masks 必须是 int。
num_masks == len(spans)。
masked_question 中 mask_token 出现次数必须等于 num_masks。
```

这是 group unit 正确性的核心约束。

---

# 11. semantic_necessity_summary（可选）

`semantic_necessity_summary` 是只读回溯元数据，用于下游 join 和过滤。

结构：

```text
key = ablation_type ("delete" / "generalize")
value = 该 ablation_type 对应的 semantic_necessity_label
额外 key "any_necessary": bool，表示该 unit 任一 ablation_type 是否 is_semantically_necessary
```

示例：

```json
{
  "delete": "Information Loss",
  "generalize": "Information Loss",
  "any_necessary": true
}
```

规则：

```text
1. semantic_necessity_summary 只用于回溯，不是 mask 阶段的规范输出。
2. masked_questions.jsonl 不写入 semantic_necessity_label 顶层字段。
3. 下游做 NLI 必要性 vs 模型可恢复性对比时，应优先通过 source_semantic_label_ids
   join 回 semantic_labels.jsonl。
```

---

# 12. only_necessary 过滤

本阶段支持可选过滤，只保留语义必要的 unit：

```text
only_necessary = False（默认）:
  对全部 unit 生成 masked question。

only_necessary = True:
  仅当该 unit 至少一条来源记录 is_semantically_necessary == True 时才生成。
```

被过滤的 unit 计入 `num_filtered_not_necessary`。

---

# 13. Backend 规则

Sprint 1F 只支持：

```text
rule_v0
```

`rule_v0` 是 deterministic string-substitution backend。

它不调用真实模型。

它不下载模型。

它不调用外部 API。

不支持的 backend 必须失败并报告：

```text
Unsupported backend: <backend>
```

---

# 14. 完整示例

## 14.1 single unit

```json
{
  "mask_id": "gsm8k_0001__unit_001__mask",
  "id": "gsm8k_0001",
  "unit_id": "unit_001",
  "unit_scope": "single",
  "group_type": "single",
  "span_ids": ["span_001"],
  "spans": [
    {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}
  ],
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "masked_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
  "mask_token": "[MASK]",
  "num_masks": 1,
  "mask_backend": "rule_v0",
  "source_semantic_label_ids": [
    "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
    "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0"
  ],
  "semantic_necessity_summary": {
    "delete": "Information Loss",
    "generalize": "Information Loss",
    "any_necessary": true
  }
}
```

## 14.2 group unit

```json
{
  "mask_id": "gsm8k_0001__unit_003__mask",
  "id": "gsm8k_0001",
  "unit_id": "unit_003",
  "unit_scope": "group",
  "group_type": "number_set",
  "span_ids": ["span_001", "span_002"],
  "spans": [
    {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9},
    {"span_id": "span_002", "text": "2", "type": "number", "start": 26, "end": 27}
  ],
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "masked_question": "Tom has [MASK] apples and buys [MASK] more. How many apples does he have now?",
  "mask_token": "[MASK]",
  "num_masks": 2,
  "mask_backend": "rule_v0",
  "source_semantic_label_ids": [
    "gsm8k_0001__unit_003__delete__nli_stub_v0__sem_rule_v0",
    "gsm8k_0001__unit_003__generalize__nli_stub_v0__sem_rule_v0"
  ]
}
```

---

# 15. Validator

`src/recover_attention/schemas.py` 中应提供：

```python
validate_masked_question_record(record: dict) -> None
```

该函数负责检查本接口定义的 schema。

至少检查：

```text
1.  顶层必含字段完整。
2.  拒绝旧 span-level 字段 span_id / span_text / span_type。
3.  拒绝未来阶段字段 recoverable / recovered_question / recoverability_label /
    confidence / sample_id / recovery_consistency。
4.  mask_id / id / unit_id 是非空 string。
5.  mask_id == f"{id}__{unit_id}__mask"。
6.  unit_scope ∈ {single, group}。
7.  group_type 是非空 string。
8.  span_ids 是非空 list[str]；spans 是非空 list[dict]；长度一致；顺序一致。
9.  每个 span 含 span_id / text / type / start / end；type 合法；start>=0；end>start；
    original_question[start:end] == text。
10. unit_scope=single -> span_ids 长度为 1；group -> 长度 >= 2。
11. group_type=single -> unit_scope=single；group_type!=single -> unit_scope=group。
12. original_question / masked_question 是非空 string。
13. mask_token 是非空 string。
14. num_masks 是 int 且 == len(spans)。
15. masked_question 中 mask_token 出现次数 == num_masks。
16. masked_question != original_question。
17. mask_backend ∈ {rule_v0}。
18. source_semantic_label_ids 是非空 list[str]。
19. 若存在 semantic_necessity_summary：必须是 dict，含 "any_necessary": bool，
    其余 value 属于 semantic necessity label 枚举。
```

如果检查失败，应抛出异常，不要静默修复。

---

# 16. 文件与模块职责

## 16.1 data artifact

```text
data/processed/masked_questions.jsonl
```

职责：

```text
保存 Sprint 1F 输出的 unit-level masked questions。
作为后续 recoverability 阶段的输入。
```

---

## 16.2 module

```text
src/recover_attention/masked_questions.py
```

职责：

```text
读取 semantic label records。
按 (id, unit_id) 去重。
对 unit 全部 span 做 mask 替换。
负责 mask_id 生成、num_masks 计算、source_semantic_label_ids 汇总。
不负责 recovery、recoverability scoring、attention。
```

推荐核心函数：

```python
apply_mask_to_unit(question: str, unit: dict, mask_token: str = "[MASK]") -> str

build_masked_question_record(
    semantic_label_records_for_unit: list[dict],
    mask_token: str = "[MASK]",
    backend: str = "rule_v0",
) -> dict

build_masked_question_records(
    semantic_label_records: list[dict],
    mask_token: str = "[MASK]",
    backend: str = "rule_v0",
    only_necessary: bool = False,
) -> tuple[list[dict], dict]

build_masked_question_file(
    input_path: str | Path,
    output_path: str | Path,
    mask_token: str = "[MASK]",
    backend: str = "rule_v0",
    only_necessary: bool = False,
) -> tuple[list[dict], dict]
```

---

## 16.3 schema

```text
src/recover_attention/schemas.py
```

职责：

```text
提供 validate_masked_question_record（unit-level）。
保证 masked_questions.jsonl 的 record 满足本接口。
```

---

## 16.4 CLI

```text
scripts/07_build_masked_questions.py
```

职责：

```text
读取 data/processed/semantic_labels.jsonl。
调用 masked_questions.py。
校验输出 record。
写入 data/processed/masked_questions.jsonl。
打印统计信息。
```

推荐命令：

```bash
python scripts/07_build_masked_questions.py \
  --input data/processed/semantic_labels.jsonl \
  --output data/processed/masked_questions.jsonl \
  --mask-token "[MASK]" \
  --backend rule_v0
```

---

## 16.5 tests

```text
tests/test_masked_questions.py
tests/test_schemas.py
```

职责：

```text
验证 single / group mask 构造逻辑。
验证按 unit 去重。
验证 only_necessary 过滤。
验证 schema validator。
验证 CLI smoke test。
```

---

# 17. 统计信息

`07_build_masked_questions.py` 推荐打印：

```text
num_input_labels
num_units
num_output_masks
num_filtered_not_necessary
num_skipped_overlap
num_skipped_mismatch
mask_token
backend
only_necessary
unit_scope_counts
group_type_counts
mask_count_distribution
```

统计只打印到 stdout，不要求写入 json 文件。

---

# 18. 本接口不包含的内容

本接口不定义：

```text
recover output
recovered question
recoverability score
recoverability label
attention anchor label
guidance action
hidden states
attention maps
trajectory analysis
probe label
```

这些属于后续阶段。

---

# 19. 最小验收标准

一个合格的 `masked_questions.jsonl` 至少满足：

```text
1.  每行是 JSON object。
2.  每条 record 对应一个 ablation unit（已按 (id, unit_id) 去重）。
3.  每条 record 有 mask_id / id / unit_id / unit_scope / group_type / span_ids / spans。
4.  mask_id == f"{id}__{unit_id}__mask"。
5.  保留 unit 结构字段，且来自一致的来源记录。
6.  masked_question 中 mask_token 数量 == num_masks == len(spans)。
7.  single -> 1 个 mask；group -> >=2 个 mask。
8.  masked_question != original_question。
9.  source_semantic_label_ids 非空。
10. 不包含旧 span-level 字段，也不包含未来阶段字段。
11. validate_masked_question_record 全部通过。
```
