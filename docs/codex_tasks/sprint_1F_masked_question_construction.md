# Sprint 1F：Masked Question Construction for Recoverability

## 1. 目标

本 sprint 实现 unit-level masked question construction 的最小可运行版本。

目标文件流：

```text
data/processed/semantic_labels.jsonl
→ data/processed/masked_questions.jsonl
```

本 sprint 只负责读取 Sprint 1E 的 semantic label records，按 ablation unit 去重，并对 unit 全部 span 做 [MASK] 替换，生成 unit-level masked questions。

本 sprint 不做 question recovery，不做 recoverability scoring，不做真实模型调用，不做 attention guidance，不做 probe，不做 hidden states，不做 trajectory analysis。

---

## 2. 当前阶段定位

前置 pipeline：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
→ nli_scores.jsonl
→ semantic_labels.jsonl
```

本 sprint 接在 1E 后面：

```text
semantic_labels.jsonl
→ masked_questions.jsonl
```

下一步才可能是：

```text
masked_questions.jsonl
→ recover_outputs.jsonl
→ recover_scores.jsonl
```

因此，本 sprint 只输出 masked questions，不输出 recover outputs，不输出 recoverability labels，不输出 attention anchor labels。

---

## 3. 设计原则

本 sprint 必须保持以下边界：

```text
1E:
  semantic necessity label record（按 unit + ablation_type 展开）。

1F:
  masked question record（按 unit 去重，纯结构构造）。
```

关键原则：

```text
1. masked_questions.jsonl 是 unit-level，与 1B/1C/1D/1E 链路一致。
2. 不使用旧 span-level schema（span_id / span_text / span_type 单数）。
3. masked 阶段只构造输入，不写 recovery / recoverability / attention / probe 字段。
4. mask 与 ablation_type 无关，按 (id, unit_id) 去重。
5. group unit 内每个 span 各替换为一个 mask_token。
```

---

## 4. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1E 已完成。
data/processed/semantic_labels.jsonl 已存在。
docs/skill/semantic_labels_interface.md 已存在。
docs/skill/masked_questions_interface.md 已存在。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/semantic_labels.jsonl
Please run Sprint 1E first.
```

不要自动回头执行 Sprint 1A / 1B / 1C / 1D / 1E。

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
docs/skill/SKILL.md
docs/skill/label_schema.md
docs/skill/experiment_guide.md
docs/skill/semantic_labels_interface.md
docs/skill/masked_questions_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/semantic_labels.py
src/recover_attention/question_ablations.py
data/processed/semantic_labels.jsonl
```

如果以下文件存在，也读取：

```text
src/recover_attention/masked_questions.py
scripts/07_build_masked_questions.py
tests/test_masked_questions.py
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
docs/skill/semantic_labels_interface.md
docs/skill/masked_questions_interface.md
docs/skill/label_schema.md
docs/skill/experiment_guide.md
docs/skill/SKILL.md
```

必须检查以下问题：

```text
1. semantic_labels.jsonl 是否仍是 unit-level，含 unit_id / span_ids / spans。
2. masked_questions_interface.md 是否定义为 unit-level，且能表示 group unit。
3. label_schema.md 中 Masked Question Record 是否仍是旧 span-level（如是，必须更新）。
4. experiment_guide.md 的 pipeline 是否包含：
   semantic_labels.jsonl
   masked_questions.jsonl
5. SKILL.md 是否已索引 masked_questions_interface.md。
6. schemas.py 中的 validate_masked_question_record 是否仍是旧 span-level（如是，必须重写）。
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

注意：本 sprint 已知 `schemas.py` 中存在旧 span-level 的 `validate_masked_question_record`，
以及 `label_schema.md §10` 旧 span-level 示例。这两处属于本 sprint 计划内的迁移目标，
不视为阻塞冲突，应按本 task card 重写为 unit-level。

---

## 8. 允许修改

本 sprint 允许修改：

```text
docs/skill/masked_questions_interface.md
docs/skill/SKILL.md
docs/skill/label_schema.md
docs/skill/experiment_guide.md
src/recover_attention/schemas.py
src/recover_attention/masked_questions.py
scripts/07_build_masked_questions.py
tests/test_masked_questions.py
tests/test_schemas.py
data/processed/masked_questions.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
src/recover_attention/masked_questions.py
scripts/07_build_masked_questions.py
tests/test_masked_questions.py
docs/progress/sprint_1_history.md
```

限制：

```text
docs/skill/SKILL.md:
  只允许增加 masked_questions_interface.md 的文档索引行。

docs/skill/label_schema.md:
  只允许把 Masked Question Record 从旧 span-level 更新为 unit-level 短 schema 总览，
  并删除已过时且与当前接口冲突的旧说明。

docs/skill/experiment_guide.md:
  只允许修正 pipeline 或阶段边界中与 1F 相关的内容。

src/recover_attention/schemas.py:
  本 sprint 只允许重写 validate_masked_question_record 为 unit-level、新增 ALLOWED_MASK_BACKENDS，
  以及（可选）抽取共享的 span 校验 helper。不要改动其它阶段的 validator 行为。

PROGRESS.md:
  完成后只更新短版当前状态、可运行命令、最近检查结果、关键文件状态和下一步。
```

---

## 9. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/skill/ablation_units_interface.md
docs/skill/ablated_questions_interface.md
docs/skill/nli_scores_interface.md
docs/skill/semantic_labels_interface.md
docs/skill/method.md
docs/skill/prompts.md
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
src/recover_attention/semantic_labels.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

本 sprint 禁止生成：

```text
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
data/processed/attention_anchor_labels.jsonl
```

注意：本 sprint 禁止修改 `validate_recover_output_record` 与 `validate_recover_score_record`。
这两个 validator 的 unit-level 迁移属于后续 sprint（1G / 1H）。

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 10. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1.  是否确认当前进入 Sprint 1F：Masked Question Construction。
2.  当前 Python 环境信息。
3.  data/processed/semantic_labels.jsonl 是否存在。
4.  semantic label record 数量。
5.  去重后 unit 数量（按 (id, unit_id)）。
6.  unit_scope 分布。
7.  group_type 分布。
8.  is_semantically_necessary 分布。
9.  docs/skill/masked_questions_interface.md 是否存在。
10. label_schema.md 是否仍把 Masked Question Record 定义为旧 span-level。
11. schemas.py 中 validate_masked_question_record 是否仍是旧 span-level。
12. SKILL.md 是否已索引 masked_questions_interface.md。
13. experiment_guide.md pipeline 是否包含 masked_questions.jsonl。
14. 本次计划使用的 mask backend 与 mask_token。
15. 是否明确不调用真实模型。
16. 是否明确不做 recovery / recoverability / attention / probe。
17. 是否发现接口冲突。
```

如果发现接口冲突，停止并等待用户决定，不要继续实现。

---

## 11. 输入接口契约

输入文件：

```text
data/processed/semantic_labels.jsonl
```

输入必须符合：

```text
docs/skill/semantic_labels_interface.md
```

每条输入 record 至少包含：

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
semantic_necessity_label
is_semantically_necessary
```

本 sprint 不推断旧字段名。

如果输入 record 缺少上述字段，停止并报告 schema mismatch。

不要自动修改 `semantic_labels.jsonl`。

---

## 12. 输出接口契约

输出文件：

```text
data/processed/masked_questions.jsonl
```

输出必须符合：

```text
docs/skill/masked_questions_interface.md
```

每条 output record 对应一个 ablation unit。

去重规则：

```text
按 (id, unit_id) 去重，一个 unit 只产出一条 masked question。
同一 unit 的 unit_scope / group_type / span_ids / spans / original_question 必须一致，
否则抛出异常，不静默修复。
```

不要按 ablation_type 拆成多条。

不要按 span 拆成多条。

---

## 13. mask_id 与 num_masks

`mask_id` 生成规则：

```python
mask_id = f"{id}__{unit_id}__mask"
```

示例：

```text
gsm8k_0001__unit_001__mask
```

`num_masks` 规则：

```text
num_masks == len(spans)
masked_question 中 mask_token 出现次数 == num_masks
```

---

## 14. Mask Backend

本 sprint 只支持：

```text
rule_v0
```

CLI 参数必须支持：

```text
--backend rule_v0
```

如果传入其他 backend，必须抛出 ValueError 或退出并报告：

```text
Unsupported backend: <backend>
```

不要调用 LLM。

不要调用真实模型。

不要调用外部 API。

不要下载模型。

---

## 15. Mask 规则与算法

替换规则：

```text
1. single unit：唯一 span 替换为一个 mask_token。
2. group unit：unit 内每个 span 各替换为一个 mask_token。
```

替换算法（offset 安全）：

```text
1. 取 unit["spans"]，按 start 降序排序。
2. 逐个 span：masked = masked[:start] + mask_token + masked[end:]。
3. 不做 whitespace 折叠。
4. 替换前检查 unit 内 span 是否重叠；重叠则跳过该 unit，计入 num_skipped_overlap。
5. 替换后断言 masked_question.count(mask_token) == len(spans)；不符则跳过，计入 num_skipped_mismatch。
6. masked_question 不应等于 original_question。
```

默认：

```text
mask_token = "[MASK]"
```

---

## 16. only_necessary 过滤

CLI 必须支持可选 flag：

```text
--only-necessary
```

规则：

```text
未传入（默认）：对全部 unit 生成 masked question。
传入：仅当该 unit 至少一条来源 semantic label record is_semantically_necessary == True 时才生成。
```

被过滤的 unit 计入 `num_filtered_not_necessary`。

---

## 17. 输出 Record Schema

每条 `masked_questions.jsonl` record 必须包含：

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

字段来源：

```text
来自 1E / semantic_labels.jsonl（unit 结构字段，取去重后任一来源记录）：
- id
- unit_id
- unit_scope
- group_type
- span_ids
- spans
- original_question

由 1F 新增：
- mask_id
- masked_question
- mask_token
- num_masks
- mask_backend
- source_semantic_label_ids
- semantic_necessity_summary（可选）
```

`semantic_necessity_summary` 结构（若输出）：

```json
{
  "delete": "Information Loss",
  "generalize": "Information Loss",
  "any_necessary": true
}
```

它只用于回溯，不是规范字段。masked_questions.jsonl 不写入 semantic_necessity_label 顶层字段。

---

## 18. 输出示例

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

## 19. Schema Validator 要求

在 `src/recover_attention/schemas.py` 中：

新增枚举：

```python
ALLOWED_MASK_BACKENDS = {"rule_v0"}
```

**重写** 现有 `validate_masked_question_record`（当前为旧 span-level，必须改为 unit-level）：

```python
validate_masked_question_record(record: dict) -> None
```

该函数必须检查：

```text
1.  顶层必含字段完整。
2.  拒绝旧字段 span_id / span_text / span_type。
3.  拒绝未来阶段字段 recoverable / recovered_question / recoverability_label /
    confidence / sample_id / recovery_consistency。
4.  mask_id / id / unit_id 是非空 string。
5.  mask_id == f"{id}__{unit_id}__mask"。
6.  unit_scope ∈ {single, group}。
7.  group_type 是非空 string。
8.  span_ids 是非空 list[str]；spans 是非空 list[dict]；长度一致；span_id 顺序一致。
9.  每个 span 含 span_id / text / type / start / end；type 合法；start>=0；end>start；
    original_question[start:end] == text。
10. unit_scope=single -> span_ids 长度为 1；group -> 长度 >= 2。
11. group_type=single -> unit_scope=single；group_type!=single -> unit_scope=group。
12. original_question / masked_question 是非空 string。
13. mask_token 是非空 string。
14. num_masks 是 int 且 == len(spans)。
15. masked_question 中 mask_token 出现次数 == num_masks。
16. masked_question != original_question。
17. mask_backend ∈ ALLOWED_MASK_BACKENDS。
18. source_semantic_label_ids 是非空 list[str]。
19. 若存在 semantic_necessity_summary：必须是 dict，含 "any_necessary": bool，
    其余 value 属于 semantic necessity label 枚举。
```

如果检查失败，应抛出 `ValueError` 或 `AssertionError`，不要静默修复。

可选简化（不强制）：

```text
可把 span_ids / spans 一致性与逐 span offset 校验抽取为内部 helper，
供 ablation_unit / ablated_question / nli_score / masked_question 共用，消除重复。
若抽取，必须保持原有四个 validator 的对外行为不变，并补充回归测试。
```

---

## 20. 核心模块要求

新增或实现：

```text
src/recover_attention/masked_questions.py
```

推荐常量：

```python
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_MASK_BACKEND = "rule_v0"
SUPPORTED_MASK_BACKENDS = {"rule_v0"}
UNIT_STRUCTURE_FIELDS = (
    "id", "unit_id", "unit_scope", "group_type",
    "span_ids", "spans", "original_question",
)
```

推荐函数：

```python
apply_mask_to_unit(
    question: str,
    unit: dict,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> str
```

要求：

```text
对 unit["spans"] 按 start 降序替换为 mask_token，返回 masked_question。
不做 whitespace 折叠。不在此函数内处理重叠（由调用方负责）。
```

---

```python
build_masked_question_record(
    semantic_label_records_for_unit: list[dict],
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
) -> dict
```

要求：

```text
1. _validate_backend(backend)。
2. 对每条输入调用 validate_semantic_label_record。
3. 校验同一 unit 的结构字段一致，否则 ValueError。
4. masked_question = apply_mask_to_unit(original_question, base_unit, mask_token)。
5. 组装记录（mask_id / num_masks / mask_token / mask_backend /
   source_semantic_label_ids / 可选 semantic_necessity_summary）。
6. 调用 validate_masked_question_record。
```

---

```python
build_masked_question_records(
    semantic_label_records: list[dict],
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]
```

要求：

```text
1. _validate_backend(backend)，校验 mask_token 非空。
2. 按 (id, unit_id) 分组并保持首次出现顺序。
3. only_necessary=True 且该 unit 无任一 is_semantically_necessary -> 跳过(num_filtered_not_necessary)。
4. unit 内 span 重叠 -> 跳过(num_skipped_overlap)。
5. mask 计数不符 -> 跳过(num_skipped_mismatch)。
6. 其余构造一条 masked record。
7. 汇总 stats。
```

stats 至少包含：

```python
{
    "num_input_labels": 0,
    "num_units": 0,
    "num_output_masks": 0,
    "num_filtered_not_necessary": 0,
    "num_skipped_overlap": 0,
    "num_skipped_mismatch": 0,
    "mask_token": "[MASK]",
    "backend": "rule_v0",
    "only_necessary": False,
    "unit_scope_counts": {},
    "group_type_counts": {},
    "mask_count_distribution": {},
}
```

---

```python
build_masked_question_file(
    input_path: str | Path,
    output_path: str | Path,
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]
```

要求：

```text
read_jsonl -> build_masked_question_records -> write_jsonl，返回 (records, stats)。
```

---

## 21. CLI 脚本要求

新增或实现：

```text
scripts/07_build_masked_questions.py
```

命令格式：

```bash
python scripts/07_build_masked_questions.py \
  --input data/processed/semantic_labels.jsonl \
  --output data/processed/masked_questions.jsonl \
  --mask-token "[MASK]" \
  --backend rule_v0
```

支持参数：

```text
--input
--output
--mask-token
--backend
--only-necessary
```

默认值：

```text
--mask-token "[MASK]"
--backend rule_v0
--only-necessary 默认关闭
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 build_masked_question_records。
4. 对每条输出调用 validate_masked_question_record。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印 masked question 统计。
7. 不调用真实模型。
8. 不下载模型。
9. 不调用外部 API。
```

如果 backend 不是 `rule_v0`，CLI 必须失败并给出清楚错误。

---

## 22. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

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

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 23. 接口文档要求

`docs/skill/masked_questions_interface.md` 已存在，本 sprint 实现必须与其保持一致。

如实现过程中发现接口文档需要修正，必须先报告再修改，且保持 unit-level 与本 task card 一致。

在：

```text
docs/skill/SKILL.md
```

的接口文档区域增加一行：

```text
docs/skill/masked_questions_interface.md
  Masked question 的长期接口文档，说明 masked_questions.jsonl 的 unit-level schema、
  mask 规则、group unit 处理、字段来源和后续 recovery 消费方式。
```

不要把完整 schema 复制进 SKILL.md。

---

## 24. label_schema.md 更新要求

更新：

```text
docs/skill/label_schema.md
```

要求：

```text
1. 将 Masked Question Record 从旧 span-level 示例更新为 unit-level 短 schema 总览。
2. 明确：masked_questions.jsonl 以 (id, unit_id) 为单位，含 span_ids / spans / num_masks。
3. 明确：masked_questions.jsonl 不包含 recoverable / recovered_question / recoverability label。
4. 不要复制过长实现细节；稳定接口以 docs/skill/masked_questions_interface.md 为准。
5. 删除或修正与 unit-level 接口冲突的旧 span-level 说明。
```

---

## 25. experiment_guide.md 更新要求

检查：

```text
docs/skill/experiment_guide.md
```

必须确保 pipeline 包含：

```text
semantic_labels.jsonl
→ masked_questions.jsonl
→ recover_outputs.jsonl
```

并且阶段边界明确：

```text
masked_questions.jsonl:
  Sprint 1F 输出，按 unit 去重并对 unit 全部 span 做 mask。
```

如已正确，无需修改。

---

## 26. 测试要求

新增：

```text
tests/test_masked_questions.py
```

至少覆盖：

```text
1.  single unit -> masked_question 含 1 个 mask，num_masks==1。
2.  group(number_set, 2 spans) -> masked_question 含 2 个 mask，num_masks==2。
3.  repeated_surface group -> 每个出现位置各一个 mask。
4.  同一 unit 的 delete+generalize 两条 1E 记录 -> 只产出 1 条 masked 记录，
    source_semantic_label_ids 含 2 个 id。
5.  mask_id == f"{id}__{unit_id}__mask"。
6.  only_necessary=True 时，全 Equivalent 的 unit 被过滤，num_filtered_not_necessary 计数正确。
7.  unit 内重叠 span 被跳过，num_skipped_overlap 计数正确。
8.  自定义 mask_token（如 "___"）生效且计数正确。
9.  输出不含未来字段 recoverable / recovered_question / recoverability_label / confidence。
10. 每条输出可通过 validate_masked_question_record。
11. unsupported backend 抛 ValueError。
12. batch stats 字段齐全
    （num_input_labels / num_units / num_output_masks / mask_count_distribution）。
13. CLI smoke test：使用临时 semantic_labels.jsonl 运行脚本，
    验证输出文件存在、每行是合法 JSON、含核心字段。
```

如果修改了 `schemas.py`，也补充或更新：

```text
tests/test_schemas.py
```

必须新增测试：

```text
A. validate_masked_question_record 接受合法 unit-level record（single 与 group）。
B. 拒绝旧字段 span_id / span_text / span_type。
C. 拒绝 num_masks 与 masked_question 中 mask_token 计数不符。
D. 拒绝 group 但 span_ids 长度 < 2。
E. 拒绝 mask_id 格式错误。
F. 拒绝出现未来阶段字段。
```

---

## 27. 本 sprint 不做

不要实现：

```text
真实模型调用
HuggingFace transformers pipeline
question recovery
recoverability scoring
recover_outputs / recover_scores 的 unit-level 迁移
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
src/recover_attention/recover_generation.py（保持现状）
src/recover_attention/recover_scoring.py（保持现状）
scripts/08_run_recovery.py
scripts/09_score_recoverability.py
```

不要生成：

```text
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
data/processed/attention_anchor_labels.jsonl
```

---

## 28. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend rule_v0
python -m pytest -q
```

如果当前环境要求使用 conda run，则使用：

```bash
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend rule_v0
conda run -n recover_attention python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

注意：

```text
本 sprint 不自动运行 Sprint 1E。
如果 data/processed/semantic_labels.jsonl 不存在，应停止并报告缺失输入。
```

---

## 29. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1F 已完成。
下一步建议是 Sprint 1G：Question Recovery（unit-level recover outputs）。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1F | 完成 | Unit-level masked question construction |
```

当前可运行命令中增加：

```bash
python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend rule_v0
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
masked question construction: passed
```

当前关键文件状态中补充：

```text
已完成：
- docs/skill/masked_questions_interface.md
- src/recover_attention/masked_questions.py
- scripts/07_build_masked_questions.py
- tests/test_masked_questions.py
- data/processed/masked_questions.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/recover_generation.py
- scripts/08_run_recovery.py
- tests/test_recover_generation.py
- data/processed/recover_outputs.jsonl
```

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 30. 验收标准

本 sprint 完成后应满足：

```text
1.  开始前已回顾 1E / masked_questions 接口文档。
2.  已报告是否存在接口冲突。
3.  docs/skill/masked_questions_interface.md 与实现一致。
4.  docs/skill/SKILL.md 已增加 masked_questions_interface.md 索引行。
5.  docs/skill/label_schema.md 中 Masked Question Record 已更新为 unit-level。
6.  docs/skill/experiment_guide.md pipeline 正确包含 masked_questions.jsonl。
7.  src/recover_attention/masked_questions.py 存在。
8.  scripts/07_build_masked_questions.py 存在。
9.  tests/test_masked_questions.py 存在。
10. schemas.py 中 validate_masked_question_record 已改为 unit-level。
11. schemas.py 中存在 ALLOWED_MASK_BACKENDS。
12. CLI 支持 --backend rule_v0、--mask-token、--only-necessary。
13. unsupported backend 会失败。
14. masked_questions.jsonl 已生成。
15. 每条 masked record 按 (id, unit_id) 去重。
16. 每条 masked record 含 mask_id / span_ids / spans / num_masks / mask_token /
    mask_backend / source_semantic_label_ids。
17. single -> 1 个 mask；group -> >=2 个 mask。
18. masked_question 中 mask_token 数量 == num_masks == len(spans)。
19. masked_question != original_question。
20. 输出不含旧 span-level 字段，也不含未来阶段字段。
21. validate_masked_question_record 能校验输出 record。
22. python -m pytest -q 通过。
23. smoke test 通过。
24. 未调用真实模型。
25. 未下载模型。
26. 未新增 transformers / torch 依赖。
27. 未生成 recover_outputs.jsonl / recover_scores.jsonl。
28. 未修改 recover / trajectory / attention guidance / probe 相关文件。
29. PROGRESS.md 保持短版状态索引。
```

---

## 31. 完成后回复格式

完成后按以下格式回复：

```text
1.  本次完成内容
2.  接口文档回顾结果
3.  是否发现接口冲突
4.  新增/修改文件
5.  运行命令
6.  检查结果
7.  masked question 数量统计
8.  unit_scope 分布
9.  mask_count_distribution
10. PROGRESS.md 更新摘要
11. 是否确认未调用真实模型
12. 是否确认未生成 recovery / attention / probe 产物
13. 遗留问题
14. 下一步建议
```

下一步建议只写：

```text
可以继续设计 Sprint 1G：Question Recovery（unit-level recover outputs）。
```

不要自动开始 Sprint 1G。
