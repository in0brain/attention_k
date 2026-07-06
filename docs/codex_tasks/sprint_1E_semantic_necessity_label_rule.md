# Sprint 1E：Semantic Necessity Label Rule

## 1. 目标

本 sprint 实现 semantic necessity label rule 的最小可运行版本。

目标文件流：

```text
data/processed/nli_scores.jsonl
→ data/processed/semantic_labels.jsonl
```

本 sprint 只负责读取 Sprint 1D 的 score-only NLI records，并根据规则生成 semantic necessity labels。

本 sprint 不做真实 NLI，不做 masked question construction，不做 question recovery，不做 recoverability scoring，不做 attention guidance，不做 probe，不做 hidden states，不做 trajectory analysis。

---

## 2. 当前阶段定位

前置 pipeline：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
→ nli_scores.jsonl
```

本 sprint 接在 1D 后面：

```text
nli_scores.jsonl
→ semantic_labels.jsonl
```

下一步才可能是：

```text
semantic_labels.jsonl
→ masked_questions.jsonl
→ recover_outputs.jsonl
→ recover_scores.jsonl
```

因此，本 sprint 只输出 semantic necessity labels，不输出 recoverability labels，不输出 final token labels，不输出 attention anchor labels。

---

## 3. 设计原则

1D 的 `nli_scores.jsonl` 只保存双向 NLI 分数。

1E 的 `semantic_labels.jsonl` 才保存 semantic necessity label。

本 sprint 必须保持以下边界：

```text
1D:
  score-only NLI record

1E:
  semantic necessity label record
```

不要把 `semantic_necessity_label` 写回 `nli_scores.jsonl`。

不要把 recoverability、attention guidance、probe、trajectory 字段提前塞进 `semantic_labels.jsonl`。

---

## 4. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1D 已完成。
data/processed/nli_scores.jsonl 已存在。
docs/reasoning-aware-attention-guidance/nli_scores_interface.md 已存在。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/nli_scores.jsonl
Please run Sprint 1D first.
```

不要自动回头执行 Sprint 1A / 1B / 1C / 1D。

---

## 5. 环境要求

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

或当前任务明确采用：

```bash
conda run -n recover_attention python ...
```

则必须报告环境状态。

如果无法确认使用的是 `recover_attention` 环境，应停止并报告环境不一致，不要继续修改文件，不要运行测试。

---

## 6. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/nli_scoring.py
data/processed/nli_scores.jsonl
```

如果以下文件存在，也读取：

```text
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
src/recover_attention/semantic_labels.py
scripts/06_build_semantic_labels.py
tests/test_semantic_labels.py
tests/test_schemas.py
docs/progress/sprint_1_history.md
```

不要读取：

```text
docs/reference/*
```

除非当前用户指令明确要求。

---

## 7. 接口回顾与冲突检查要求

本 sprint 开始前必须回顾并检查以下接口文档之间是否冲突：

```text
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/SKILL.md
```

必须检查以下问题：

```text
1. ablation_units.jsonl 是否仍被定义为 1B 输出、1C 输入。
2. ablated_questions.jsonl 是否仍是 unit-level schema，而不是旧 span-level schema。
3. nli_scores.jsonl 是否仍是 score-only schema。
4. nli_scores.jsonl 是否没有 semantic_necessity_label。
5. semantic_labels.jsonl 是否被定义为 Sprint 1E 输出。
6. experiment_guide.md 的 pipeline 是否包含：
   ablation_units.jsonl
   ablated_questions.jsonl
   nli_scores.jsonl
   semantic_labels.jsonl
7. SKILL.md 是否已经索引 1B / 1C / 1D 接口文档。
8. 如果 semantic_labels_interface.md 已存在，它是否与本 task card 目标一致。
```

如果发现明确冲突，必须停止并报告：

```text
Interface conflict detected.
```

报告内容必须包括：

```text
1. 冲突文件
2. 冲突位置
3. 冲突内容
4. 应采用哪个接口作为当前阶段来源
5. 建议修正方式
```

不要在发现冲突后继续实现 1E。

低风险表述不一致可以报告并继续，但必须说明为什么不阻塞本 sprint。

---

## 8. 允许修改

本 sprint 允许修改：

```text
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
src/recover_attention/schemas.py
src/recover_attention/semantic_labels.py
scripts/06_build_semantic_labels.py
tests/test_semantic_labels.py
tests/test_schemas.py
data/processed/semantic_labels.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
src/recover_attention/semantic_labels.py
scripts/06_build_semantic_labels.py
tests/test_semantic_labels.py
docs/progress/sprint_1_history.md
```

限制：

```text
docs/reasoning-aware-attention-guidance/SKILL.md:
  只允许增加 semantic_labels_interface.md 的文档索引行。

docs/reasoning-aware-attention-guidance/label_schema.md:
  只允许增加或修正 Semantic Label Record 相关内容，以及删除已经过时且与当前接口冲突的旧说明。

docs/reasoning-aware-attention-guidance/experiment_guide.md:
  只允许修正 pipeline 或阶段边界中与 1E 相关的内容。

PROGRESS.md:
  完成后只更新短版当前状态、可运行命令、最近检查结果、关键文件状态和下一步。
```

---

## 9. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
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
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

本 sprint 禁止生成：

```text
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
data/processed/attention_anchor_labels.jsonl
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 10. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1. 是否确认当前进入 Sprint 1E：Semantic Necessity Label Rule。
2. 当前 Python 环境信息。
3. data/processed/nli_scores.jsonl 是否存在。
4. nli score record 数量。
5. ablation_type 分布。
6. unit_scope 分布。
7. group_type 分布。
8. language 分布。
9. docs/reasoning-aware-attention-guidance/semantic_labels_interface.md 是否存在。
10. docs/reasoning-aware-attention-guidance/nli_scores_interface.md 是否为 score-only。
11. label_schema.md 是否仍把 semantic_necessity_label 放进 nli_scores.jsonl。
12. experiment_guide.md pipeline 是否包含 semantic_labels.jsonl。
13. SKILL.md 是否已索引 semantic_labels_interface.md。
14. schemas.py 是否已有 validate_semantic_label_record。
15. 本次是否需要新增 validate_semantic_label_record。
16. 本次是否需要增强 validate_nli_score_record 的 forward/backward 方向检查。
17. 本次计划使用的 semantic label backend。
18. 本次计划使用的阈值。
19. 是否明确不调用真实 NLI。
20. 是否明确不做 mask / recovery / attention / probe。
21. 是否发现接口冲突。
```

如果发现接口冲突，停止并等待用户决定，不要继续实现。

---

## 11. 输入接口契约

输入文件：

```text
data/processed/nli_scores.jsonl
```

输入必须符合：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
```

每条输入 record 至少包含：

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

本 sprint 不推断旧字段名。

如果输入 record 缺少上述字段，停止并报告 schema mismatch。

不要自动修改 `nli_scores.jsonl`。

---

## 12. 输出接口契约

输出文件：

```text
data/processed/semantic_labels.jsonl
```

每条 output record 对应一条 input nli score record。

也就是：

```text
一条 nli_id
→ 一条 semantic label record
```

不要按 span 拆成多条。

不要按 forward / backward 拆成多条。

---

## 13. Semantic Label 枚举

本 sprint 使用以下 semantic necessity labels：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

含义：

```text
Equivalent:
  original_question 与 ablated_question 近似双向语义一致。
  ablation 后问题语义基本不变。
  对应 is_semantically_necessary = false。

Information Loss:
  original_question 能推出 ablated_question，但 ablated_question 不能推出 original_question。
  ablation 丢失了原问题中的具体信息。
  对应 is_semantically_necessary = true。

Added Assumption:
  ablated_question 能推出 original_question，但 original_question 不能推出 ablated_question。
  扰动后问题可能引入了额外假设、额外约束或异常构造。
  对应 is_semantically_necessary = true。

Non-equivalent:
  双向蕴含都不足，或 contradiction 较高。
  扰动后问题明显偏离原问题。
  对应 is_semantically_necessary = true。
```

不要新增其他 label。

不要使用中文 label。

---

## 14. Rule Backend

本 sprint 只支持：

```text
rule_v0
```

CLI 参数必须支持：

```text
--backend rule_v0
```

如果传入其他 backend，例如：

```text
llm_judge
manual
rule_v1
```

必须抛出 ValueError 或退出并报告：

```text
Unsupported backend: <backend>
```

不要调用 LLM。

不要调用真实 NLI。

不要调用外部 API。

不要下载模型。

---

## 15. Rule Parameters

默认阈值：

```python
equivalent_threshold = 0.70
directional_entailment_threshold = 0.50
contradiction_threshold = 0.50
```

CLI 必须支持：

```text
--equivalent-threshold
--directional-entailment-threshold
--contradiction-threshold
```

默认值分别为：

```text
0.70
0.50
0.50
```

所有阈值必须满足：

```text
0 <= threshold <= 1
```

如果阈值非法，应失败并报告清楚错误。

---

## 16. Label Rule

从每条 `nli_score_record` 中读取：

```python
forward_entailment = record["forward"]["scores"]["entailment"]
backward_entailment = record["backward"]["scores"]["entailment"]

forward_contradiction = record["forward"]["scores"]["contradiction"]
backward_contradiction = record["backward"]["scores"]["contradiction"]

bidirectional_entailment_score = record["bidirectional_entailment_score"]
contradiction_score = record["contradiction_score"]
```

规则顺序必须固定如下：

```python
if contradiction_score >= contradiction_threshold:
    semantic_necessity_label = "Non-equivalent"
    decision_reason = "contradiction above threshold"

elif bidirectional_entailment_score >= equivalent_threshold:
    semantic_necessity_label = "Equivalent"
    decision_reason = "bidirectional entailment above equivalent threshold"

elif (
    forward_entailment >= directional_entailment_threshold
    and backward_entailment < directional_entailment_threshold
):
    semantic_necessity_label = "Information Loss"
    decision_reason = "forward entails ablated but backward does not entail original"

elif (
    forward_entailment < directional_entailment_threshold
    and backward_entailment >= directional_entailment_threshold
):
    semantic_necessity_label = "Added Assumption"
    decision_reason = "backward entails original but forward does not entail ablated"

else:
    semantic_necessity_label = "Non-equivalent"
    decision_reason = "low bidirectional entailment"
```

注意：

```text
contradiction rule 优先级最高。
Equivalent rule 在 contradiction 之后。
```

---

## 17. Semantic Necessity Boolean

新增字段：

```text
is_semantically_necessary
```

规则：

```python
is_semantically_necessary = semantic_necessity_label != "Equivalent"
```

也就是：

```text
Equivalent → false
Information Loss → true
Added Assumption → true
Non-equivalent → true
```

---

## 18. Semantic Necessity Score

新增字段：

```text
semantic_necessity_score
```

定义：

```python
semantic_necessity_score = max(
    1.0 - bidirectional_entailment_score,
    contradiction_score,
)
```

该值必须在 `[0, 1]`。

建议使用：

```python
round(score, 10)
```

以减少浮点误差。

含义：

```text
双向语义一致性越低，semantic necessity 越高。
矛盾越高，semantic necessity 越高。
```

注意：

```text
semantic_necessity_score 是分数。
semantic_necessity_label 是规则标签。
is_semantically_necessary 是二值信号。
三者都需要保留。
```

---

## 19. 输出 Record Schema

每条 `semantic_labels.jsonl` record 必须包含：

```text
semantic_label_id
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
semantic_label_backend
semantic_necessity_label
semantic_necessity_score
is_semantically_necessary
rule_parameters
decision_reason
```

字段来源：

```text
来自 1D / nli_scores.jsonl：
- nli_id
- ablation_id
- id
- unit_id
- unit_scope
- group_type
- span_ids
- spans
- ablation_type
- original_question
- ablated_question
- nli_backend
- language
- language_setting
- forward
- backward
- bidirectional_entailment_score
- contradiction_score

由 1E 新增：
- semantic_label_id
- semantic_label_backend
- semantic_necessity_label
- semantic_necessity_score
- is_semantically_necessary
- rule_parameters
- decision_reason
```

`semantic_label_id` 生成规则：

```python
semantic_label_id = f"{nli_id}__sem_{backend}"
```

当前：

```text
backend = rule_v0
```

示例：

```text
gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0
```

---

## 20. rule_parameters Schema

`rule_parameters` 必须是 dict，包含：

```text
equivalent_threshold
directional_entailment_threshold
contradiction_threshold
```

示例：

```json
{
  "equivalent_threshold": 0.7,
  "directional_entailment_threshold": 0.5,
  "contradiction_threshold": 0.5
}
```

每个值必须是 number，范围在 `[0, 1]`。

---

## 21. 输出示例

```json
{
  "semantic_label_id": "gsm8k_0001__unit_001__generalize__nli_stub_v0__sem_rule_v0",
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
  "contradiction_score": 0.05,
  "semantic_label_backend": "rule_v0",
  "semantic_necessity_label": "Information Loss",
  "semantic_necessity_score": 0.65,
  "is_semantically_necessary": true,
  "rule_parameters": {
    "equivalent_threshold": 0.7,
    "directional_entailment_threshold": 0.5,
    "contradiction_threshold": 0.5
  },
  "decision_reason": "forward entails ablated but backward does not entail original"
}
```

---

## 22. 接口文档要求

新增或更新：

```text
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
```

该文件必须说明：

```text
1. semantic_labels.jsonl 的位置。
2. semantic_labels.jsonl 的作用。
3. 上下游关系。
4. 顶层 record schema。
5. 字段来源。
6. Semantic Necessity Label 枚举。
7. rule_v0 的阈值。
8. rule_v0 的决策规则。
9. semantic_necessity_score 的计算方式。
10. is_semantically_necessary 的计算方式。
11. 该文件不包含 recoverability / attention guidance / probe。
```

同时在：

```text
docs/reasoning-aware-attention-guidance/SKILL.md
```

的文档路由或接口文档区域增加一行：

```text
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
  Semantic label 的长期接口文档，说明 semantic_labels.jsonl schema、字段含义、rule_v0、阈值、标签枚举和后续 mask/recovery 消费方式。
```

不要把完整 schema 复制进 SKILL.md。

---

## 23. label_schema.md 更新要求

更新：

```text
docs/reasoning-aware-attention-guidance/label_schema.md
```

要求：

```text
1. 保留 Semantic Necessity Label 枚举：
   Equivalent
   Information Loss
   Added Assumption
   Non-equivalent

2. 将 Semantic Label Record 从占位说明更新为与 semantic_labels_interface.md 一致的短 schema 总览。

3. 明确：
   nli_scores.jsonl 不包含 semantic_necessity_label。
   semantic_labels.jsonl 才包含 semantic_necessity_label。

4. 不要复制过长实现细节。
   具体稳定接口以 docs/reasoning-aware-attention-guidance/semantic_labels_interface.md 为准。
```

如果 `label_schema.md` 中仍有过时说明：

```text
旧 NLI score 示例仍包含 semantic necessity label
```

或类似内容，应删除或修正。

---

## 24. experiment_guide.md 更新要求

检查：

```text
docs/reasoning-aware-attention-guidance/experiment_guide.md
```

必须确保 pipeline 包含：

```text
nli_scores.jsonl
→ semantic_labels.jsonl
→ masked_questions.jsonl
```

并且阶段边界明确：

```text
nli_scores.jsonl:
  Sprint 1D 输出，只保存 score-only 双向 NLI 结果。

semantic_labels.jsonl:
  Sprint 1E 输出，根据 nli_scores 构造 semantic necessity labels。
```

如已正确，无需修改。

---

## 25. Schema Validator 要求

在 `src/recover_attention/schemas.py` 中新增：

```python
validate_semantic_label_record(record: dict) -> None
```

该函数必须检查：

```text
1. 顶层字段完整。
2. semantic_label_id 是非空 string。
3. nli_id 是非空 string。
4. ablation_id 是非空 string。
5. id 是非空 string。
6. unit_id 是非空 string。
7. unit_scope 是 single 或 group。
8. group_type 是非空 string。
9. span_ids 是非空 list[str]。
10. spans 是非空 list[dict]。
11. span_ids 长度等于 spans 长度。
12. spans 中 span_id 顺序与 span_ids 一致。
13. ablation_type 是 delete 或 generalize。
14. original_question 是非空 string。
15. ablated_question 是非空 string。
16. nli_backend 是 stub_v0。
17. language 是 en 或 zh。
18. language_setting 是 auto / en / zh。
19. forward / backward 存在。
20. forward / backward 均包含 premise / hypothesis / label / scores。
21. forward / backward label 是 entailment / neutral / contradiction。
22. forward / backward scores 包含 entailment / neutral / contradiction。
23. 每个 NLI score 在 [0, 1]。
24. NLI scores 总和接近 1。
25. bidirectional_entailment_score 在 [0, 1]。
26. contradiction_score 在 [0, 1]。
27. semantic_label_backend 是 rule_v0。
28. semantic_necessity_label 属于 Equivalent / Information Loss / Added Assumption / Non-equivalent。
29. semantic_necessity_score 在 [0, 1]。
30. is_semantically_necessary 是 bool。
31. Equivalent 对应 is_semantically_necessary = False。
32. 非 Equivalent 对应 is_semantically_necessary = True。
33. rule_parameters 是 dict。
34. rule_parameters 包含 equivalent_threshold / directional_entailment_threshold / contradiction_threshold。
35. 每个 threshold 在 [0, 1]。
36. decision_reason 是非空 string。
37. semantic_necessity_score = max(1 - bidirectional_entailment_score, contradiction_score)。
```

如果检查失败，应抛出 `ValueError` 或 `AssertionError`，不要静默修复。

---

## 26. validate_nli_score_record 增强要求

本 sprint 允许在 `schemas.py` 中增强：

```python
validate_nli_score_record(record: dict) -> None
```

必须增加 forward/backward 方向检查：

```text
forward["premise"] == original_question
forward["hypothesis"] == ablated_question
backward["premise"] == ablated_question
backward["hypothesis"] == original_question
```

因为 Sprint 1E 的 rule 依赖 forward/backward 的方向语义。

如已有该检查，则不要重复实现。

---

## 27. 核心模块要求

新增或实现：

```text
src/recover_attention/semantic_labels.py
```

推荐常量：

```python
DEFAULT_EQUIVALENT_THRESHOLD = 0.70
DEFAULT_DIRECTIONAL_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_CONTRADICTION_THRESHOLD = 0.50
SUPPORTED_SEMANTIC_LABEL_BACKENDS = {"rule_v0"}
SEMANTIC_NECESSITY_LABELS = {
    "Equivalent",
    "Information Loss",
    "Added Assumption",
    "Non-equivalent",
}
```

推荐函数：

```python
default_rule_parameters() -> dict
```

返回：

```python
{
    "equivalent_threshold": 0.70,
    "directional_entailment_threshold": 0.50,
    "contradiction_threshold": 0.50,
}
```

---

```python
validate_rule_parameters(rule_parameters: dict) -> None
```

检查三个阈值存在且在 `[0, 1]`。

---

```python
assign_semantic_necessity_label(
    nli_score_record: dict,
    rule_parameters: dict | None = None,
) -> tuple[str, str]
```

返回：

```text
semantic_necessity_label
decision_reason
```

必须实现第 16 节中的固定 rule order。

---

```python
compute_semantic_necessity_score(
    bidirectional_entailment_score: float,
    contradiction_score: float,
) -> float
```

返回：

```python
round(max(1.0 - bidirectional_entailment_score, contradiction_score), 10)
```

---

```python
label_nli_score_record(
    record: dict,
    backend: str = "rule_v0",
    rule_parameters: dict | None = None,
) -> dict
```

要求：

```text
1. 先调用 validate_nli_score_record(record)。
2. backend 只允许 rule_v0。
3. 解析 rule_parameters。
4. 计算 semantic_necessity_label。
5. 计算 semantic_necessity_score。
6. 计算 is_semantically_necessary。
7. 生成 semantic label record。
8. 调用 validate_semantic_label_record。
```

---

```python
label_nli_score_records(
    records: list[dict],
    backend: str = "rule_v0",
    rule_parameters: dict | None = None,
) -> tuple[list[dict], dict]
```

返回：

```text
labeled_records:
  semantic label records。

stats:
  统计信息。
```

stats 至少包含：

```python
{
    "num_input_scores": 0,
    "num_output_labels": 0,
    "backend": "rule_v0",
    "rule_parameters": {},
    "semantic_necessity_label_counts": {},
    "is_semantically_necessary_counts": {},
    "ablation_type_counts": {},
    "unit_scope_counts": {},
    "group_type_counts": {},
    "language_counts": {},
}
```

---

## 28. CLI 脚本要求

新增或实现：

```text
scripts/06_build_semantic_labels.py
```

命令格式：

```bash
python scripts/06_build_semantic_labels.py \
  --input data/processed/nli_scores.jsonl \
  --output data/processed/semantic_labels.jsonl \
  --backend rule_v0 \
  --equivalent-threshold 0.70 \
  --directional-entailment-threshold 0.50 \
  --contradiction-threshold 0.50
```

支持参数：

```text
--input
--output
--backend
--equivalent-threshold
--directional-entailment-threshold
--contradiction-threshold
```

默认值：

```text
--backend rule_v0
--equivalent-threshold 0.70
--directional-entailment-threshold 0.50
--contradiction-threshold 0.50
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 label_nli_score_records。
4. 对每条输出调用 validate_semantic_label_record。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印 semantic label 统计。
7. 不调用真实模型。
8. 不下载模型。
9. 不调用外部 API。
```

如果 backend 不是 `rule_v0`，CLI 必须失败并给出清楚错误。

---

## 29. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_input_scores
num_output_labels
backend
rule_parameters
semantic_necessity_label_counts
is_semantically_necessary_counts
ablation_type_counts
unit_scope_counts
group_type_counts
language_counts
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 30. 测试要求

新增：

```text
tests/test_semantic_labels.py
```

至少覆盖以下测试。

### 30.1 Equivalent

构造一条 nli score record：

```text
bidirectional_entailment_score >= 0.70
contradiction_score < 0.50
```

验证：

```text
semantic_necessity_label = Equivalent
is_semantically_necessary = False
```

### 30.2 Information Loss

构造一条 nli score record：

```text
forward_entailment >= 0.50
backward_entailment < 0.50
contradiction_score < 0.50
bidirectional_entailment_score < 0.70
```

验证：

```text
semantic_necessity_label = Information Loss
is_semantically_necessary = True
```

### 30.3 Added Assumption

构造一条 nli score record：

```text
forward_entailment < 0.50
backward_entailment >= 0.50
contradiction_score < 0.50
bidirectional_entailment_score < 0.70
```

验证：

```text
semantic_necessity_label = Added Assumption
is_semantically_necessary = True
```

### 30.4 Contradiction High

构造一条 nli score record：

```text
contradiction_score >= 0.50
```

验证：

```text
semantic_necessity_label = Non-equivalent
decision_reason = contradiction above threshold
```

### 30.5 Both Low

构造一条 nli score record：

```text
forward_entailment < 0.50
backward_entailment < 0.50
contradiction_score < 0.50
```

验证：

```text
semantic_necessity_label = Non-equivalent
decision_reason = low bidirectional entailment
```

### 30.6 semantic_necessity_score

验证：

```python
semantic_necessity_score == max(
    1.0 - bidirectional_entailment_score,
    contradiction_score,
)
```

### 30.7 semantic_label_id

验证：

```python
semantic_label_id == f"{nli_id}__sem_rule_v0"
```

### 30.8 rule_parameters preserved

验证输出 record 中保存了实际使用的三个阈值。

### 30.9 unsupported backend raises

传入：

```text
manual
```

应抛出 ValueError 或 CLI 返回失败。

### 30.10 invalid thresholds raise

阈值小于 0 或大于 1 时应失败。

### 30.11 schema validation

验证每条输出 record 都能通过：

```python
validate_semantic_label_record(record)
```

### 30.12 output has no future-stage fields

验证输出不包含：

```text
recoverability_label
recoverable
masked_question
recovered_question
attention_anchor_label
guidance_action
hidden_states_path
attentions_path
```

### 30.13 batch stats

验证：

```text
num_input_scores
num_output_labels
semantic_necessity_label_counts
is_semantically_necessary_counts
```

### 30.14 CLI smoke test

使用临时输入 jsonl 运行 CLI。

验证：

```text
输出文件存在
每行是合法 JSON object
至少包含 semantic_label_backend / semantic_necessity_label / semantic_necessity_score / is_semantically_necessary
```

如果修改了 `schemas.py`，也补充或更新：

```text
tests/test_schemas.py
```

必须新增测试：

```text
1. validate_semantic_label_record 接受合法 record。
2. validate_semantic_label_record 拒绝缺失 semantic_label_id。
3. validate_semantic_label_record 拒绝非法 semantic_necessity_label。
4. validate_semantic_label_record 拒绝 semantic_necessity_score 计算错误。
5. validate_semantic_label_record 拒绝 Equivalent 但 is_semantically_necessary = True。
6. validate_semantic_label_record 拒绝非 Equivalent 但 is_semantically_necessary = False。
7. validate_nli_score_record 拒绝 forward/backward premise/hypothesis 方向反转。
```

---

## 31. 本 sprint 不做

不要实现：

```text
真实 NLI 模型
HuggingFace transformers pipeline
XNLI model loading
LLM prompt judge
API 调用
模型下载
masked question construction
question recovery
recoverability scoring
label_builder final aggregation
token label building
attention anchor label
oracle attention guidance
probe-guided attention guidance
hidden states
attention maps
trajectory stability
answer stability
```

不要新增：

```text
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/09_score_recoverability.py
```

不要生成：

```text
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
data/processed/attention_anchor_labels.jsonl
```

---

## 32. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
python -m pytest -q
```

如果当前环境要求使用 conda run，则使用：

```bash
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
conda run -n recover_attention python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

注意：

```text
本 sprint 不自动运行 Sprint 1D。
如果 data/processed/nli_scores.jsonl 不存在，应停止并报告缺失输入。
```

---

## 33. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1E 已完成。
下一步建议是 Sprint 1F：Masked Question Construction for Recoverability。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1E | 完成 | Semantic necessity label rule |
```

当前可运行命令中增加：

```bash
python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
semantic label rule: passed
```

当前关键文件状态中补充：

```text
已完成：
- docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
- src/recover_attention/semantic_labels.py
- scripts/06_build_semantic_labels.py
- tests/test_semantic_labels.py
- data/processed/semantic_labels.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/masked_questions.py
- scripts/07_build_masked_questions.py
- tests/test_masked_questions.py
- data/processed/masked_questions.jsonl
```

如果 `PROGRESS.md` 仍包含已经过时的遗留问题，例如：

```text
docs/reasoning-aware-attention-guidance/label_schema.md 中的旧 NLI score 示例仍包含 semantic necessity label
```

并且当前已经修复，应删除该遗留项。

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 34. 验收标准

本 sprint 完成后应满足：

```text
1. 开始前已回顾 1B / 1C / 1D 接口文档。
2. 已报告是否存在接口冲突。
3. docs/reasoning-aware-attention-guidance/semantic_labels_interface.md 存在。
4. docs/reasoning-aware-attention-guidance/SKILL.md 已增加 semantic_labels_interface.md 索引行。
5. docs/reasoning-aware-attention-guidance/label_schema.md 已包含当前 Semantic Label Record 短 schema 总览。
6. docs/reasoning-aware-attention-guidance/experiment_guide.md pipeline 仍正确包含 semantic_labels.jsonl。
7. src/recover_attention/semantic_labels.py 存在。
8. scripts/06_build_semantic_labels.py 存在。
9. tests/test_semantic_labels.py 存在。
10. schemas.py 中存在 validate_semantic_label_record。
11. validate_nli_score_record 已检查 forward/backward 方向。
12. CLI 支持 --backend rule_v0。
13. CLI 支持三个 threshold 参数。
14. unsupported backend 会失败。
15. invalid threshold 会失败。
16. semantic_labels.jsonl 已生成。
17. 每条 semantic label record 保留 1D 元数据。
18. 每条 semantic label record 包含 semantic_necessity_label。
19. 每条 semantic label record 包含 semantic_necessity_score。
20. 每条 semantic label record 包含 is_semantically_necessary。
21. 每条 semantic label record 包含 rule_parameters。
22. Equivalent 对应 is_semantically_necessary = false。
23. 非 Equivalent 对应 is_semantically_necessary = true。
24. semantic_necessity_score 计算正确。
25. validate_semantic_label_record 能校验输出 record。
26. python -m pytest -q 通过。
27. smoke test 通过。
28. 未调用真实 NLI 模型。
29. 未下载模型。
30. 未新增 transformers / torch 依赖。
31. 未生成 masked_questions.jsonl。
32. 未生成 recover_outputs.jsonl。
33. 未实现 recovery / trajectory / attention guidance / probe。
34. PROGRESS.md 保持短版状态索引。
```

---

## 35. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 接口文档回顾结果
3. 是否发现接口冲突
4. 新增/修改文件
5. 运行命令
6. 检查结果
7. semantic label 数量统计
8. semantic_necessity_label 分布
9. is_semantically_necessary 分布
10. PROGRESS.md 更新摘要
11. 是否确认未调用真实 NLI
12. 是否确认未生成 mask / recovery / attention / probe 产物
13. 遗留问题
14. 下一步建议
```

下一步建议只写：

```text
可以继续设计 Sprint 1F：Masked Question Construction for Recoverability。
```

不要自动开始 Sprint 1F。
