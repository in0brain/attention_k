# docs/reasoning-aware-attention-guidance/nli_scores_interface.md

# NLI Scores Interface

本文件定义 `data/processed/nli_scores.jsonl` 的稳定接口。

该文件是 Sprint 1D 的输出，也是 Sprint 1E 及后续 label rule 阶段的输入契约。

---

# 1. 位置

```text
data/processed/nli_scores.jsonl
```

---

# 2. 作用

`nli_scores.jsonl` 用来保存：

```text
original_question 与 ablated_question 之间的双向 NLI 语义一致性分数。
```

它回答的问题是：

```text
扰动前问题和扰动后问题在语义上是否仍然一致？
```

本接口只保存分数，不保存最终标签。

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
```

其中：

```text
ablated_questions.jsonl:
  由 Sprint 1C 生成，保存 original_question / ablated_question pairs。

nli_scores.jsonl:
  由 Sprint 1D 生成，保存双向 NLI stub 分数。

semantic_labels.jsonl:
  后续 Sprint 1E 生成，根据 nli_scores.jsonl 计算 semantic necessity label。
```

---

# 4. 顶层 Record Schema

`nli_scores.jsonl` 每一行是一个 JSON object。

每条 record 对应一条 `ablated_questions.jsonl` record。

必须包含：

<!-- required_fields:nli_score -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->

```text
nli_id
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
nli_backend
language
language_setting
forward
backward
bidirectional_entailment_score
contradiction_score
```

---

# 5. 字段来源

来自 Sprint 1C / `ablated_questions.jsonl` 的字段：

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

由 Sprint 1D 新增的字段：

```text
nli_id
nli_backend
language
language_setting
forward
backward
bidirectional_entailment_score
contradiction_score
```

---

# 6. 字段含义

```text
nli_id:
  当前 NLI score record 的唯一 ID。

ablation_id:
  对应的 ablated question record ID。

id:
  原始问题 ID。

unit_id:
  当前 ablation 作用的 unit ID。

unit_scope:
  single 或 group。

group_type:
  single / repeated_surface / number_set / entity_set / object_set / cyber_security_term_set。

span_ids:
  当前 unit 包含的 span_id 列表。

spans:
  当前 unit 包含的 span 完整信息列表。

ablation_type:
  delete 或 generalize。

original_question:
  原始问题文本。

ablated_question:
  经 delete 或 generalize 后得到的问题文本。

nli_backend:
  当前 NLI scoring backend。Sprint 1D 只支持 stub_v0。

language:
  当前 record 最终解析出的语言，en 或 zh。

language_setting:
  CLI 传入的语言设置，auto / en / zh。

forward:
  original_question → ablated_question 的 NLI 结果。

backward:
  ablated_question → original_question 的 NLI 结果。

bidirectional_entailment_score:
  双向蕴含分数。

contradiction_score:
  双向最大矛盾分数。
```

---

# 7. nli_id 规则

`nli_id` 推荐使用：

```python
nli_id = f"{ablation_id}__nli_{nli_backend}"
```

当前 Sprint 1D：

```text
nli_backend = stub_v0
```

示例：

```text
gsm8k_0001__unit_001__generalize__nli_stub_v0
```

---

# 8. NLI 方向定义

## 8.1 Forward

```text
premise = original_question
hypothesis = ablated_question
```

含义：

```text
原问题是否蕴含扰动后问题。
```

---

## 8.2 Backward

```text
premise = ablated_question
hypothesis = original_question
```

含义：

```text
扰动后问题是否蕴含原问题。
```

---

# 9. forward / backward Schema

`forward` 和 `backward` 必须包含：

```text
premise
hypothesis
label
scores
```

其中 `label` 必须是：

```text
entailment
neutral
contradiction
```

`scores` 必须包含：

```text
entailment
neutral
contradiction
```

示例：

```json
{
  "premise": "Tom has 3 apples.",
  "hypothesis": "Tom has some number apples.",
  "label": "entailment",
  "scores": {
    "entailment": 0.75,
    "neutral": 0.2,
    "contradiction": 0.05
  }
}
```

---

# 10. 双向分数

## 10.1 bidirectional_entailment_score

定义：

```python
bidirectional_entailment_score = min(
    forward["scores"]["entailment"],
    backward["scores"]["entailment"],
)
```

含义：

```text
越高表示 original_question 与 ablated_question 越接近双向语义一致。
```

---

## 10.2 contradiction_score

定义：

```python
contradiction_score = max(
    forward["scores"]["contradiction"],
    backward["scores"]["contradiction"],
)
```

含义：

```text
越高表示两个问题之间越可能存在语义矛盾。
```

---

# 11. Language 规则

Sprint 1D 支持：

```text
auto
en
zh
```

记录方式：

```text
language_setting:
  用户或 CLI 指定的语言参数。

language:
  最终解析出的语言。
```

规则：

```text
language_setting = en:
  language = en

language_setting = zh:
  language = zh

language_setting = auto:
  如果 original_question 或 ablated_question 包含中文字符，则 language = zh；
  否则 language = en。
```

本阶段不翻译输入文本。

无论 language 是中文还是英文，label 和 scores key 都必须使用英文枚举：

```text
entailment
neutral
contradiction
```

---

# 12. Backend 规则

Sprint 1D 只支持：

```text
stub_v0
```

`stub_v0` 是 deterministic scoring backend。

它不调用真实模型。

它不下载模型。

它不调用外部 API。

真实 NLI backend 应在后续单独 sprint 中添加。

---

# 13. 完整示例

```json
{
  "nli_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0",
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
  "original_question": "Tom has 3 apples and buys 2 more.",
  "ablated_question": "Tom has some number apples and buys 2 more.",
  "nli_backend": "stub_v0",
  "language": "en",
  "language_setting": "auto",
  "forward": {
    "premise": "Tom has 3 apples and buys 2 more.",
    "hypothesis": "Tom has some number apples and buys 2 more.",
    "label": "entailment",
    "scores": {
      "entailment": 0.75,
      "neutral": 0.2,
      "contradiction": 0.05
    }
  },
  "backward": {
    "premise": "Tom has some number apples and buys 2 more.",
    "hypothesis": "Tom has 3 apples and buys 2 more.",
    "label": "neutral",
    "scores": {
      "entailment": 0.35,
      "neutral": 0.6,
      "contradiction": 0.05
    }
  },
  "bidirectional_entailment_score": 0.35,
  "contradiction_score": 0.05
}
```

---

# 14. 与 1E 的关系

Sprint 1E 会读取 `nli_scores.jsonl`，根据：

```text
bidirectional_entailment_score
contradiction_score
ablation_type
unit_scope
group_type
```

构造 semantic necessity labels。

1D 不做 label。

1D 只提供分数。

---

# 15. Validator

`src/recover_attention/schemas.py` 中应提供：

```python
validate_nli_score_record(record: dict) -> None
```

该函数负责检查本接口定义的 schema。

至少检查：

```text
1. 顶层字段完整。
2. forward / backward 字段完整。
3. label 属于 entailment / neutral / contradiction。
4. scores 包含 entailment / neutral / contradiction。
5. scores 均在 [0, 1]。
6. scores 总和接近 1。
7. language 属于 en / zh。
8. language_setting 属于 auto / en / zh。
9. nli_backend 为 stub_v0。
10. bidirectional_entailment_score 计算正确。
11. contradiction_score 计算正确。
```

如果检查失败，应抛出异常，不要静默修复。

---

# 16. 文件与模块职责

## 16.1 data artifact

```text
data/processed/nli_scores.jsonl
```

职责：

```text
保存 Sprint 1D 输出的双向 NLI scores。
作为 Sprint 1E 的输入。
```

---

## 16.2 module

```text
src/recover_attention/nli_scoring.py
```

职责：

```text
读取 ablated question record。
解析 language。
执行 stub_v0 双向 NLI scoring。
生成 nli score record。
不调用真实模型。
不生成 label。
```

推荐核心函数：

```python
detect_language(text: str) -> str

resolve_record_language(record: dict, language: str = "auto") -> str

score_nli_pair_stub(
    premise: str,
    hypothesis: str,
    ablation_type: str,
    num_spans: int,
    direction: str,
) -> dict

score_ablated_question_record(
    record: dict,
    backend: str = "stub_v0",
    language: str = "auto",
) -> dict

score_ablated_question_records(
    records: list[dict],
    backend: str = "stub_v0",
    language: str = "auto",
) -> tuple[list[dict], dict]
```

---

## 16.3 schema

```text
src/recover_attention/schemas.py
```

职责：

```text
提供 validate_nli_score_record。
保证 nli_scores.jsonl 的 record 满足本接口。
```

---

## 16.4 CLI

```text
scripts/05_run_nli_scoring.py
```

职责：

```text
读取 data/processed/ablated_questions.jsonl。
调用 nli_scoring.py。
校验输出 record。
写入 data/processed/nli_scores.jsonl。
打印统计信息。
```

推荐命令：

```bash
python scripts/05_run_nli_scoring.py \
  --input data/processed/ablated_questions.jsonl \
  --output data/processed/nli_scores.jsonl \
  --backend stub_v0 \
  --language auto
```

---

## 16.5 tests

```text
tests/test_nli_scoring.py
tests/test_schemas.py
```

职责：

```text
验证 stub_v0 双向 NLI scoring。
验证 language 参数。
验证 schema validator。
验证 CLI smoke test。
```

---

# 17. 本接口不包含的内容

本接口不定义：

```text
semantic necessity label
recoverability label
masked question
recover output
recoverability score
attention anchor label
probe label
真实 NLI 模型输出校准
```

这些属于后续阶段。

---

# 18. 最小验收标准

一个合格的 `nli_scores.jsonl` 至少满足：

```text
1. 每行是 JSON object。
2. 每条 record 有 nli_id / ablation_id / id。
3. 每条 record 保留 1C 元数据。
4. 每条 record 有 forward / backward。
5. forward / backward 都包含 premise / hypothesis / label / scores。
6. label 使用 entailment / neutral / contradiction。
7. scores 包含 entailment / neutral / contradiction。
8. scores 总和接近 1。
9. bidirectional_entailment_score 计算正确。
10. contradiction_score 计算正确。
11. language / language_setting 存在且合法。
12. nli_backend = stub_v0。
13. validate_nli_score_record 全部通过。
```
