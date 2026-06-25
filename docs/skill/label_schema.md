# Label Schema

本文件提供项目 jsonl 数据格式的总览、枚举值索引、record 示例和标签规则。

完整顶层字段不在本文件维护，以 `src/recover_attention/schemas.py` 的 `REQUIRED_FIELDS`
和各 `docs/skill/*_interface.md` 为准（见第 0 节）。

本文件只负责数据格式，不负责实验流程和 sprint 拆分。

实验流程由：

```text
docs/skill/experiment_guide.md
```

负责。

Sprint 拆分由：

```text
docs/skill/codex_tasks.md
```

负责。

方法概念由：

```text
docs/skill/method.md
```

负责。

具体代码实现以：

```text
src/recover_attention/schemas.py
```

和当前 task card 为准。

---

# 0. 字段维护流程与生成边界（重要）

顶层字段的**唯一来源**是：

```text
src/recover_attention/schemas.py 的 REQUIRED_FIELDS / FORBIDDEN_FIELDS
```

各 `docs/skill/*_interface.md` 中，`<!-- required_fields:<type> -->` 标记之后的代码块是
**生成产物**，由脚本从 `REQUIRED_FIELDS` 写入：

```text
scripts/sync_interface_fields.py
```

## 修改字段的标准流程

```text
1. 编辑 schemas.py 的 REQUIRED_FIELDS（如涉及禁用字段或校验逻辑，同步 FORBIDDEN_FIELDS 与对应 validator）。
2. 运行 python scripts/sync_interface_fields.py --write   回填各 interface 的 required_fields 块。
3. 运行 python -m pytest tests/test_interface_consistency.py -q   确认四向一致。
```

## 生成边界（谁管什么）

```text
scripts/sync_interface_fields.py 只改：
  各 interface 文档中 required_fields marker 之后的那个代码块。
  绝不改动 marker 行、注释、示例、约束或其它内容。

scripts/sync_interface_fields.py 不管：
  - 本文件 label_schema.md（它是索引/总览，由测试校验“指向 interface 且不复制字段”，不由脚本生成）。
  - REQUIRED_FIELDS 中没有 interface 文档的 record（question / candidate_span /
    recover_score / attention_anchor_label）。

本文件 label_schema.md 的职责：
  数据格式总览、枚举值索引、record 示例和概念约束。
  不再罗列完整顶层字段表；完整字段以 interface 文档与 schemas.py 为准。

tests/test_interface_consistency.py 的职责：
  锁定 schemas.py / interface marker / label_schema 三者一致，CI 阶段拦截漂移。
```

不要手改 interface 文档中的生成块；手改会在下次 `--write` 被覆盖，并可能被 `--check` 判为漂移。

---

# 1. 基本原则

所有中间数据文件默认使用：

```text
jsonl
```

每一行是一个 JSON object。

不要把以下格式作为主中间格式：

```text
CSV
Excel
pickle
```

除非某个 task card 明确要求导出辅助分析文件。

所有 record 必须包含稳定的样本标识：

```text
id
```

所有 span 级别 record 必须包含：

```text
span_id
```

字段命名原则：

```text
1. 使用 snake_case。
2. 不使用中文字段名。
3. 不使用临时字段名，例如 tmp、foo、bar。
4. 不把后续阶段字段提前塞进当前阶段 record。
5. 一个阶段只产出自己负责的字段。
```

---

# 2. 全局枚举

## 2.1 Span Type

允许的 `span_type` / candidate `type`：

```text
number
entity
object
operation
relation
comparison
negation
condition
question_target
cyber_security_term
```

含义：

```text
number:
数字、数量、比例、百分比、金额、时间等。

entity:
人名、组织、地点、产品名、系统名等实体。

object:
被操作或被计数的对象，例如 apples、notebooks、files、packets。

operation:
动作或运算词，例如 buys、removes、adds、encrypts、executes。

relation:
实体或对象之间的关系，例如 from、through、because of、belongs to。

comparison:
比较关系，例如 more than、less than、cheaper、higher。

negation:
否定词或否定结构，例如 not、never、without。

condition:
条件或约束，例如 if、assuming、given that、when。

question_target:
问题要求求解的目标，例如 How many、what city、which file。

cyber_security_term:
网络安全相关术语，例如 remote code execution、unsafe deserialization、SQL injection。
```

---

## 2.2 Ablation Type

允许的 `ablation_type`：

```text
delete
generalize
replace
mask
```

阶段使用建议：

```text
NLI semantic necessity:
优先使用 delete / generalize / replace。

Semantic recoverability:
优先使用 mask。

Trajectory intervention:
使用 mask / remove / replace 等干预形式，具体以 task card 为准。
```

注意：

```text
remove 是 intervention_type，不是 ablation_type 的默认枚举。
```

---

## 2.3 NLI Directional Label

允许的 NLI 方向标签：

```text
entailment
neutral
contradiction
```

---

## 2.4 Semantic Necessity Label

允许的 `semantic_necessity_label`：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

---

## 2.5 Recoverable Value

允许的 `recoverable`：

```text
yes
no
uncertain
```

---

## 2.6 Recoverability Label

允许的 `recoverability_label`：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

---

## 2.7 Attention Anchor Label

允许的 `attention_anchor_label`：

```text
Strong Anchor
Medium Anchor
Weak Anchor
Risky Anchor
Distractor
```

含义：

```text
Strong Anchor:
多种信号一致表明该 span 对推理重要，后续倾向 boost。

Medium Anchor:
部分信号表明该 span 重要，可以适度增强。

Weak Anchor:
信号较弱或不稳定，通常保持观察。

Risky Anchor:
该 span 可能诱导错误复原、错误轨迹或不稳定推理，需要谨慎处理。

Distractor:
该 span 可能吸引无效 attention，后续可能 suppress。
```

---

## 2.8 Guidance Action

允许的 `guidance_action`：

```text
boost
suppress
keep
review
```

含义：

```text
boost:
增强该 span 或 span group 的 attention。

suppress:
抑制可能误导推理的 span。

keep:
不主动干预。

review:
证据不足，需要人工或后续实验检查。
```

---

## 2.9 Backend Name

常见 `backend` / `generation_backend` / `nli_backend` / `recovery_backend`：

```text
stub
rule_based
fixture
local_model
api_model
manual
```

早期 sprint 默认使用：

```text
stub
rule_based
fixture
```

不要在 schema 中强行限制只能用这些值，除非当前 task card 明确要求。

---

# 3. Question Record

文件：

```text
data/processed/questions.jsonl
```

用途：

```text
作为整个 pipeline 的基础输入。
```

示例：

```json
{"id":"gsm8k_0001","dataset":"gsm8k","question":"Tom has 3 apples and buys 2 more. How many apples does he have now?","gold_answer":"5"}
```

字段：

```text
id: str, non-empty
dataset: str, non-empty
question: str, non-empty
gold_answer: str, non-empty
```

约束：

```text
1. id 必须在 questions.jsonl 内唯一。
2. question 不应为空。
3. gold_answer 可以是数字字符串、文本答案或标准答案表达。
4. 不要在 question record 中加入 candidate spans、CoT 或标签字段。
```

---

# 4. Baseline CoT Record

文件：

```text
data/processed/baseline_cot.jsonl
```

用途：

```text
记录原始问题的 baseline reasoning trace。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "cot_text": "Step 1: Tom starts with 3 apples.\nStep 2: He buys 2 more apples.\nStep 3: 3 + 2 = 5.\n#### 5",
  "steps": [
    {"step_id": 1, "text": "Tom starts with 3 apples."},
    {"step_id": 2, "text": "He buys 2 more apples."},
    {"step_id": 3, "text": "3 + 2 = 5."}
  ],
  "final_answer": "5",
  "generation_backend": "stub"
}
```

字段：

```text
id: str, non-empty
question: str, non-empty
cot_text: str, non-empty
steps: list
final_answer: str
generation_backend: str, non-empty
```

`steps` 中每个对象字段：

```text
step_id: int, >= 1
text: str, non-empty
```

约束：

```text
1. baseline CoT record 不判断答案是否正确。
2. step parsing 只解析结构，不做 trajectory stability。
3. generation_backend 可以是 stub / fixture / local_model / api_model。
```

---

# 5. Baseline Trajectory Manifest Record

文件：

```text
data/processed/baseline_trajectory_manifest.jsonl
```

用途：

```text
记录 baseline CoT 对应的 hidden states、attention maps 和 step marker 位置信息。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "cot_id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "num_steps": 3,
  "hidden_states_path": "cache/hidden_states/baseline/gsm8k_0001.pt",
  "attentions_path": "cache/attentions/baseline/gsm8k_0001.pt",
  "step_marker_positions": [12, 25, 41],
  "model_name": "stub-model",
  "backend": "stub"
}
```

字段：

```text
id: str, non-empty
cot_id: str, non-empty
question: str, non-empty
num_steps: int, >= 0
hidden_states_path: str
attentions_path: str
step_marker_positions: list[int]
model_name: str
backend: str, non-empty
```

约束：

```text
1. schema 阶段可以允许 hidden_states_path / attentions_path 为空字符串。
2. 真实缓存阶段必须保证路径存在，具体由对应 task card 决定。
3. manifest 只记录路径和索引，不直接内嵌大 tensor。
```

---

# 6. Candidate Span Record

文件：

```text
data/processed/candidate_spans.jsonl
```

用途：

```text
记录每个 question 中可能影响推理的 candidate spans。
```

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

record 字段：

```text
id: str, non-empty
question: str, non-empty
candidates: list
```

candidate 字段：

```text
span_id: str, non-empty
text: str, non-empty
type: str, one of allowed span types
start: int, >= 0
end: int, > start
```

约束：

```text
1. candidates 可以为空 list，但不应缺失。
2. span_id 在同一个 question 内必须唯一。
3. start / end 使用 Python 字符串切片语义：[start, end)。
4. question[start:end] 应尽量等于 text；如果经过 normalization，必须在 evidence 中说明。
5. Candidate span 只表示候选，不表示重要。
```

---

# 7. Ablated Question Record

文件：

```text
data/processed/ablated_questions.jsonl
```

用途：

```text
记录对 ablation unit 执行 delete / generalize 后得到的扰动问题。
```

当前稳定接口以：

```text
docs/skill/ablated_questions_interface.md
```

为准。

旧版 span-level schema 已废弃，不作为当前 pipeline 接口。

示例：

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

字段：本文件不再罗列完整字段表。顶层字段以 `docs/skill/ablated_questions_interface.md` 和 `src/recover_attention/schemas.py` 的 `REQUIRED_FIELDS["ablated_question"]` 为准。

`spans` 中每个元素必须包含：

```text
span_id
text
type
start
end
```

约束：

```text
1. 每条 record 对应一个 ablation unit 和一种 ablation_type。
2. 当前 Sprint 1C 只支持 delete / generalize。
3. 顶层不再使用 span_id / span_text / span_type 表示 ablation 对象。
4. span_ids 与 spans 的长度和顺序必须一致。
5. original_question[start:end] 应等于 span["text"]。
6. unit_scope = single 时 span_ids 长度为 1。
7. unit_scope = group 时 span_ids 长度至少为 2。
8. ablated_question 不应为空，也不应与 original_question 完全相同。
9. ablation record 不包含 NLI 结果、semantic label、recoverability label 或 attention label。
```

---

# 8. NLI Score Record

文件：

```text
data/processed/nli_scores.jsonl
```

用途：

```text
记录 original_question 与 ablated_question 的双向 NLI 分数。
```

当前稳定接口以：

```text
docs/skill/nli_scores_interface.md
```

为准。

`nli_scores.jsonl` 只保存 score-only 双向 NLI 结果。

semantic necessity label 由后续 Sprint 1E 写入：

```text
data/processed/semantic_labels.jsonl
```

示例：

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

字段：本文件不再罗列完整字段表。顶层字段以 `docs/skill/nli_scores_interface.md` 和 `src/recover_attention/schemas.py` 的 `REQUIRED_FIELDS["nli_score"]` 为准。

`forward` 和 `backward` 字段：

```text
premise: str, non-empty
hypothesis: str, non-empty
label: entailment / neutral / contradiction
scores: dict with entailment / neutral / contradiction
```

约束：

```text
1. 每条 record 对应一条 ablated question record。
2. forward 表示 original_question → ablated_question。
3. backward 表示 ablated_question → original_question。
4. scores 中每个 score 必须在 [0, 1]，并且总和接近 1。
5. bidirectional_entailment_score = min(forward entailment, backward entailment)。
6. contradiction_score = max(forward contradiction, backward contradiction)。
7. nli_scores.jsonl 不包含 semantic_necessity_label。
8. NLI score record 不包含 recoverability。
9. NLI score record 不包含 attention anchor label。
10. NLI 只是 attention importance discovery 的辅助信号。
```

---

# 9. Semantic Label Record

Current stable interface:

```text
docs/skill/semantic_labels_interface.md
```

File:

```text
data/processed/semantic_labels.jsonl
```

Purpose:

```text
Sprint 1E reads data/processed/nli_scores.jsonl and writes semantic necessity labels.
nli_scores.jsonl remains score-only and does not contain semantic_necessity_label.
semantic_labels.jsonl is the first current pipeline artifact that contains semantic_necessity_label.
```

Required fields: see `docs/skill/semantic_labels_interface.md` and
`REQUIRED_FIELDS["semantic_label"]` in `src/recover_attention/schemas.py`. This
file no longer duplicates the full field list.

Current backend:

```text
rule_v0
```

Allowed semantic_necessity_label values:

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

Notes:

```text
1. semantic_label_id = f"{nli_id}__sem_{semantic_label_backend}".
2. is_semantically_necessary = semantic_necessity_label != "Equivalent".
3. semantic_necessity_score = round(max(1 - bidirectional_entailment_score, contradiction_score), 10).
4. The full stable schema is defined in docs/skill/semantic_labels_interface.md.
```

---

# 10. Masked Question Record

当前稳定接口以：

```text
docs/skill/masked_questions_interface.md
```

为准。旧版 span-level masked question schema 已废弃。

文件：

```text
data/processed/masked_questions.jsonl
```

用途：

```text
保存 unit-level masked questions，用于后续 recoverability 阶段。
当前 1F 的输入应来自 data/processed/semantic_labels.jsonl。
masked_questions.jsonl 按 id + unit_id 聚合 semantic label records。
```

字段：本文件不再罗列完整字段表。顶层字段以 `docs/skill/masked_questions_interface.md` 和 `src/recover_attention/schemas.py` 的 `REQUIRED_FIELDS["masked_question"]` 为准。

示例：

```json
{
  "masked_id": "gsm8k_0001__unit_001__mask",
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
  "original_question": "Tom has 3 apples and buys 2 more.",
  "masked_question": "Tom has [MASK] apples and buys 2 more.",
  "mask_token": "[MASK]",
  "mask_backend": "unit_mask_v0",
  "mask_strategy": "replace_each_span",
  "source_semantic_label_ids": [
    "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
    "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0"
  ],
  "source_nli_ids": [
    "gsm8k_0001__unit_001__delete__nli_stub_v0",
    "gsm8k_0001__unit_001__generalize__nli_stub_v0"
  ],
  "source_ablation_ids": [
    "gsm8k_0001__unit_001__delete",
    "gsm8k_0001__unit_001__generalize"
  ],
  "semantic_sources": [
    {
      "semantic_label_id": "gsm8k_0001__unit_001__delete__nli_stub_v0__sem_rule_v0",
      "nli_id": "gsm8k_0001__unit_001__delete__nli_stub_v0",
      "ablation_id": "gsm8k_0001__unit_001__delete",
      "ablation_type": "delete",
      "semantic_necessity_label": "Information Loss",
      "semantic_necessity_score": 0.75,
      "is_semantically_necessary": true,
      "decision_reason": "forward entails ablated but backward does not entail original"
    },
    {
      "semantic_label_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0",
      "nli_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0",
      "ablation_id": "gsm8k_0001__unit_001__generalize",
      "ablation_type": "generalize",
      "semantic_necessity_label": "Information Loss",
      "semantic_necessity_score": 0.65,
      "is_semantically_necessary": true,
      "decision_reason": "forward entails ablated but backward does not entail original"
    }
  ]
}
```

约束：

```text
1. mask 对象是 ablation unit，不是单个 candidate span。
2. 同一个 id + unit_id 只生成一条 masked question。
3. delete / generalize 两类 semantic label 应聚合到 semantic_sources。
4. 顶层不再使用 span_id / span_text / span_type 表示 mask 对象。
5. mask_backend 当前只允许 unit_mask_v0。
6. mask_strategy 当前只允许 replace_each_span。
7. replace_each_span 要求 unit 内每个 span 各新增一个 mask_token，即
   masked_question.count(mask_token) - original_question.count(mask_token) == len(spans)。
8. masked_questions.jsonl 不包含 recovery 输出。
9. masked_questions.jsonl 不包含 recoverability label。
10. masked_questions.jsonl 不包含 attention guidance / probe / hidden states 字段。
```

---

# 11. Recover Output Record

当前稳定接口以：

```text
docs/skill/recover_outputs_interface.md
```

为准。旧版 span-level recover output schema 已废弃。

文件：

```text
data/processed/recover_outputs.jsonl
```

用途：

```text
保存对 unit-level masked questions 的 recovery samples。
当前 1G 的输入应来自 data/processed/masked_questions.jsonl。
recover_outputs.jsonl 按 masked_id 绑定 masked question，并可用 sample_id
记录同一个 masked question 的多次复原结果。
recover_outputs.jsonl 保留后续 recoverability scoring 所需的 masked question
元数据，避免评分阶段必须再 join 回 masked_questions.jsonl。
```

字段：本文件不再罗列完整字段表。顶层字段以 `docs/skill/recover_outputs_interface.md`
和 `src/recover_attention/schemas.py` 的 `REQUIRED_FIELDS["recover_output"]` 为准。

约束：

```text
1. recover output 是 unit-level / masked_id-driven，不再使用顶层 span_id。
2. recover output 必须复制 unit_scope、group_type、original_question、mask_token、
   mask_backend 和 mask_strategy。
3. recovery 的任务是复原问题，不是解题。
4. recovery_backend 当前只允许 oracle_stub_v0；该 backend 只用于管线验证，
   不能用于真实性能结论。
5. recovered_question 可以为空字符串，表示没有可用复原。
6. recover output 不包含 recoverability label、confidence 或 attention anchor label。
7. recoverability 判断属于后续 scoring 阶段。
```

---

# 12. Recover Score Record

文件：

```text
data/processed/recover_scores.jsonl
```

用途：

```text
聚合 recover_outputs，得到 span 级别 recoverability 评分。
```

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

字段：

```text
id: str, non-empty
span_id: str, non-empty
recoverability_label: str, one of allowed recoverability labels
confidence_mean: int or float, 0 <= confidence_mean <= 1
recovery_consistency: int or float, 0 <= recovery_consistency <= 1
misleading_recovery: bool
```

可选字段：

```text
num_samples: int, >= 0
evidence: dict or list
```

约束：

```text
1. recoverability 不是最终重要性标签。
2. 不要把 Non-recoverable 直接等同于 Strong Anchor。
3. 不要把 Recoverable 直接等同于不重要。
```

---

# 13. Intervention Manifest Record

文件：

```text
data/processed/intervention_manifest.jsonl
```

用途：

```text
记录对某个 span 做 mask / remove / replace 等 intervention 后的运行信息。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "intervention_type": "mask",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "intervened_question": "Tom has [MASK] apples and buys 2 more. How many apples does he have now?",
  "intervened_cot_path": "data/processed/interventions/gsm8k_0001_span_001_mask_cot.json",
  "hidden_states_path": "cache/hidden_states/interventions/gsm8k_0001_span_001_mask.pt",
  "attentions_path": "cache/attentions/interventions/gsm8k_0001_span_001_mask.pt",
  "backend": "stub"
}
```

字段：

```text
id: str, non-empty
span_id: str, non-empty
intervention_type: str, non-empty
original_question: str, non-empty
intervened_question: str, non-empty
intervened_cot_path: str
hidden_states_path: str
attentions_path: str
backend: str, non-empty
```

常见 `intervention_type`：

```text
mask
remove
replace
```

约束：

```text
1. manifest 只记录 intervention 运行信息。
2. 不直接计算 trajectory_stability_score。
3. 不直接计算 answer_stability_score。
```

---

# 14. Trajectory Stability Score Record

文件：

```text
data/processed/trajectory_stability_scores.jsonl
```

用途：

```text
记录 baseline trajectory 与 intervention trajectory 的偏移程度。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "intervention_type": "mask",
  "step_shift_score": 0.35,
  "hidden_geometry_shift": 0.62,
  "attention_distribution_shift": 0.48,
  "trajectory_stability_score": 0.41,
  "trajectory_changed": true,
  "evidence": {
    "num_baseline_steps": 3,
    "num_intervened_steps": 4
  }
}
```

字段：

```text
id: str, non-empty
span_id: str, non-empty
intervention_type: str, non-empty
step_shift_score: int or float, 0 <= score <= 1
hidden_geometry_shift: int or float, 0 <= score <= 1
attention_distribution_shift: int or float, 0 <= score <= 1
trajectory_stability_score: int or float, 0 <= score <= 1
trajectory_changed: bool
evidence: dict or list
```

约束：

```text
1. 分数语义必须在 task card 中说明。
2. 不要在此 record 中输出 attention_anchor_label。
3. trajectory_changed 是辅助字段，不替代 trajectory_stability_score。
```

---

# 15. Answer Stability Score Record

文件：

```text
data/processed/answer_stability_scores.jsonl
```

用途：

```text
记录 span intervention 是否改变 final answer 或答案稳定性。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "intervention_type": "mask",
  "baseline_answer": "5",
  "intervened_answer": "unknown",
  "answer_changed": true,
  "answer_stability_score": 0.18,
  "evidence": {
    "baseline_final_answer": "5",
    "intervened_final_answer": "unknown"
  }
}
```

字段：

```text
id: str, non-empty
span_id: str, non-empty
intervention_type: str, non-empty
baseline_answer: str
intervened_answer: str
answer_changed: bool
answer_stability_score: int or float, 0 <= score <= 1
evidence: dict or list
```

约束：

```text
1. answer_changed 不等于 span 一定重要。
2. answer_stability_score 只是 attention importance 的辅助信号。
3. 不要在此 record 中输出 guidance_action。
```

---

# 16. Attention Anchor Label Record

文件：

```text
data/processed/attention_anchor_labels.jsonl
```

用途：

```text
组合 NLI、recoverability、trajectory stability、answer stability 和 raw attention pattern，
得到 attention steering 的输入标签。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "attention_importance_score": 0.87,
  "attention_anchor_label": "Strong Anchor",
  "guidance_action": "boost",
  "guidance_strength": 0.75,
  "evidence": {
    "semantic_necessity_label": "Information Loss",
    "recoverability_label": "Non-recoverable",
    "trajectory_stability_score": 0.41,
    "answer_stability_score": 0.18,
    "raw_attention_to_span": 0.12
  }
}
```

字段：

```text
id: str, non-empty
span_id: str, non-empty
span_text: str, non-empty
span_type: str, one of allowed span types
attention_importance_score: int or float, 0 <= score <= 1
attention_anchor_label: str, one of allowed attention anchor labels
guidance_action: str, one of allowed guidance actions
guidance_strength: int or float, 0 <= strength <= 1
evidence: dict or list
```

建议 evidence 包含：

```text
semantic_necessity_label
recoverability_label
trajectory_stability_score
answer_stability_score
raw_attention_to_span
```

约束：

```text
1. attention anchor label 是 attention steering 的输入。
2. 它不是最终 hallucination detection 结果。
3. 不要用单一信号直接决定 Strong Anchor。
4. guidance_action 只表示后续建议动作，不等于已经执行 guidance。
```

---

# 17. Oracle Guidance Result Record

文件：

```text
data/processed/oracle_guidance_results.jsonl
```

用途：

```text
记录使用已知 attention anchor 进行 oracle attention guidance 的结果。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "guidance_mode": "oracle",
  "used_anchor_ids": ["span_001", "span_003"],
  "baseline_answer": "5",
  "guided_answer": "5",
  "baseline_correct": true,
  "guided_correct": true,
  "trajectory_stability_delta": 0.12,
  "attention_shift_to_anchors": 0.31,
  "evidence": {}
}
```

字段：

```text
id: str, non-empty
guidance_mode: str, non-empty
used_anchor_ids: list[str]
baseline_answer: str
guided_answer: str
baseline_correct: bool
guided_correct: bool
trajectory_stability_delta: int or float
attention_shift_to_anchors: int or float
evidence: dict or list
```

约束：

```text
1. oracle guidance 是验证上界，不是最终可部署方法。
2. 不要把 oracle guidance 结果当作 probe-guided 结果。
```

---

# 18. Probe-Guided Guidance Result Record

文件：

```text
data/processed/probe_guidance_results.jsonl
```

用途：

```text
记录使用 probe 预测 anchor 并进行 attention guidance 的结果。
```

示例：

```json
{
  "id": "gsm8k_0001",
  "guidance_mode": "probe_guided",
  "predicted_anchor_ids": ["span_001"],
  "baseline_answer": "5",
  "guided_answer": "5",
  "baseline_correct": true,
  "guided_correct": true,
  "probe_confidence": 0.78,
  "attention_shift_to_anchors": 0.22,
  "evidence": {}
}
```

字段：

```text
id: str, non-empty
guidance_mode: str, non-empty
predicted_anchor_ids: list[str]
baseline_answer: str
guided_answer: str
baseline_correct: bool
guided_correct: bool
probe_confidence: int or float, 0 <= confidence <= 1
attention_shift_to_anchors: int or float
evidence: dict or list
```

约束：

```text
1. 只有 Sprint 7 或之后才允许出现真实 probe-guided guidance。
2. 不要在早期 sprint 生成该文件。
```

---

# 19. Schema 与代码同步要求

当前代码 schema 应逐步与本文件同步。

近期需要同步的内容：

```text
1. ALLOWED_SPAN_TYPES 增加 object。
2. ALLOWED_ABLATION_TYPES 增加 replace。
3. 增加 ALLOWED_ATTENTION_ANCHOR_LABELS。
4. 增加 ALLOWED_GUIDANCE_ACTIONS。
5. 增加 validate_attention_anchor_label_record。
```

后续再逐步增加：

```text
validate_baseline_cot_record
validate_baseline_trajectory_manifest_record
validate_intervention_manifest_record
validate_trajectory_stability_score_record
validate_answer_stability_score_record
validate_oracle_guidance_result_record
validate_probe_guidance_result_record
```

不要在一个 sprint 中一次性实现所有未来 schema。

---

# 20. 常见错误

禁止把以下关系写死：

```text
Information Loss => Strong Anchor
Non-recoverable => Strong Anchor
answer_changed => Strong Anchor
high_attention => Strong Anchor
Equivalent => Distractor
Recoverable => Distractor
```

原因：

```text
1. 语义必要不等于不可复原。
2. 不可复原不等于一定重要。
3. 答案不变不等于 span 不关键。
4. attention 高不等于因果重要。
5. attention 低不等于不重要。
```

正确做法：

```text
多个信号共同进入 evidence。
由组合规则或模型得到 attention_importance_score。
再得到 attention_anchor_label。
最后才得到 guidance_action 和 guidance_strength。
```
