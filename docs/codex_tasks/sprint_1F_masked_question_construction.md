# Sprint 1F：Masked Question Construction for Recoverability

## 1. 目标

本 sprint 实现 unit-level masked question construction 的最小可运行版本。

目标文件流：

```text
data/processed/semantic_labels.jsonl
→ data/processed/masked_questions.jsonl
```

本 sprint 只负责读取 Sprint 1E 的 semantic label records，按 `id + unit_id` 聚合，并对 unit 全部 span 做 [MASK] 替换，生成 unit-level masked questions。

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

## 3. 权威接口（必须严格遵守）

本 sprint 的稳定接口以下列文件为准，且这些文件已迁移为 unit-level：

```text
docs/skill/masked_questions_interface.md   （权威 schema）
docs/skill/label_schema.md §10             （短 schema 总览，与上面一致）
src/recover_attention/schemas.py           （validate_masked_question_record 已就位）
```

顶层字段（共 16 个，缺一不可）：

```text
masked_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
masked_question
mask_token
mask_backend
mask_strategy
source_semantic_label_ids
source_nli_ids
source_ablation_ids
semantic_sources
```

固定取值约定：

```text
mask_backend  = unit_mask_v0   （默认）
mask_strategy = replace_each_span
mask_token    = [MASK]          （默认，可由 CLI 覆盖）
masked_id     = f"{id}__{unit_id}__mask"
```

严禁回退到旧 span-level 字段：

```text
禁止使用顶层 span_id / span_text / span_type。
validate_masked_question_record 会显式拒绝这三个字段。
```

---

## 4. 设计原则

边界：

```text
1E:
  semantic necessity label record（按 unit + ablation_type 展开，每个 unit 多条）。

1F:
  masked question record（按 id + unit_id 聚合，每个 unit 一条，纯结构构造）。
```

关键原则：

```text
1. masked_questions.jsonl 是 unit-level，与 1B/1C/1D/1E 链路一致。
2. mask 对象是 ablation unit，不是单个 candidate span。
3. mask 与 ablation_type 无关，按 id + unit_id 聚合，一个 unit 只产出一条记录。
4. group unit 内每个 span 各替换为一个 mask_token（replace_each_span）。
5. delete / generalize 等多条 1E 来源记录聚合进 semantic_sources。
6. masked 阶段只构造输入，不写 recovery / recoverability / attention / probe 字段。
```

---

## 5. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1E 已完成。
data/processed/semantic_labels.jsonl 已存在。
docs/skill/semantic_labels_interface.md 已存在。
docs/skill/masked_questions_interface.md 已存在。
src/recover_attention/schemas.py 中 validate_masked_question_record 已是 unit-level。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/semantic_labels.jsonl
Please run Sprint 1E first.
```

不要自动回头执行 Sprint 1A / 1B / 1C / 1D / 1E。

---

## 6. 环境要求

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

## 7. 开始前必须读取

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

## 8. 接口回顾与冲突检查要求

本 sprint 开始前必须回顾并确认以下文件互相一致：

```text
docs/skill/masked_questions_interface.md
docs/skill/label_schema.md §10
src/recover_attention/schemas.py（validate_masked_question_record）
docs/skill/SKILL.md（masked_questions_interface 索引）
docs/skill/experiment_guide.md（pipeline 含 masked_questions.jsonl）
```

必须确认：

```text
1. 四处的顶层字段列表完全一致（masked_id / mask_strategy / semantic_sources / 三个 source id 列表）。
2. mask_backend = unit_mask_v0，mask_strategy = replace_each_span。
3. 没有任何文件仍把 Masked Question 定义为旧 span-level。
4. validate_masked_question_record 拒绝 span_id / span_text / span_type。
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

注意：schema、validator、label_schema、SKILL、interface 均已迁移完成。本 sprint 的剩余工作是
实现 `masked_questions.py` 模块、`scripts/07` CLI、测试和产物，使其符合既有接口。

---

## 9. 允许修改

本 sprint 允许修改：

```text
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

`src/recover_attention/schemas.py`：

```text
默认不修改。validate_masked_question_record 已是 unit-level 权威实现。
仅当发现其与 masked_questions_interface.md 存在确凿不一致时，
才允许在报告后做最小修正，并补充 tests/test_schemas.py 回归测试。
```

`docs/skill/*`：

```text
默认不修改。interface / label_schema / SKILL / experiment_guide 已对齐。
仅当实现过程中发现接口文档与本 task card 不一致时，先报告再改。
```

---

## 10. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/skill/masked_questions_interface.md（默认）
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
这两个 validator 仍是 span-level，其 unit-level 迁移属于后续 sprint（1G / 1H）。

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 11. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1.  是否确认当前进入 Sprint 1F：Masked Question Construction。
2.  当前 Python 环境信息。
3.  data/processed/semantic_labels.jsonl 是否存在。
4.  semantic label record 数量。
5.  聚合后 unit 数量（按 id + unit_id）。
6.  unit_scope 分布。
7.  group_type 分布。
8.  is_semantically_necessary 分布。
9.  每个 unit 的来源 ablation_type 分布（确认 delete / generalize 是否成对出现）。
10. masked_questions_interface.md 顶层字段是否为 16 字段权威版。
11. schemas.py validate_masked_question_record 是否已是 unit-level。
12. 本次计划使用的 mask_backend / mask_strategy / mask_token。
13. 是否明确不调用真实模型。
14. 是否明确不做 recovery / recoverability / attention / probe。
15. 是否发现接口冲突。
```

如果发现接口冲突，停止并等待用户决定，不要继续实现。

---

## 12. 输入接口契约

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
semantic_necessity_score
is_semantically_necessary
decision_reason
```

本 sprint 不推断旧字段名。

如果输入 record 缺少上述字段，停止并报告 schema mismatch。

不要自动修改 `semantic_labels.jsonl`。

---

## 13. 输出接口契约与聚合规则

输出文件：

```text
data/processed/masked_questions.jsonl
```

输出必须符合：

```text
docs/skill/masked_questions_interface.md
```

聚合规则：

```text
1. 按 (id, unit_id) 聚合，一个 unit 只产出一条 masked question。
2. 同一 unit 的 unit_scope / group_type / span_ids / spans / original_question 必须一致，
   否则抛出异常，不静默修复。
3. 同一 unit 的全部来源记录聚合进 semantic_sources，并同步生成
   source_semantic_label_ids / source_nli_ids / source_ablation_ids。
```

不要按 ablation_type 拆成多条。

不要按 span 拆成多条。

---

## 14. masked_id 规则

```python
masked_id = f"{id}__{unit_id}__mask"
```

示例：

```text
gsm8k_0001__unit_001__mask
```

注意：

```text
masked_id 不包含 delete / generalize，因为 mask 是为 unit 构造，不是为某条 ablation 记录。
```

---

## 15. mask 规则与算法

固定取值：

```text
mask_backend  = unit_mask_v0
mask_strategy = replace_each_span
mask_token    = [MASK]（默认，可由 CLI 覆盖）
```

替换规则：

```text
1. single unit：唯一 span 替换为一个 mask_token。
2. group unit：unit 内每个 span 各替换为一个 mask_token。
```

替换算法（offset 安全，参考 question_ablations.apply_ablation_to_unit）：

```text
1. 取 spans，按 start 降序排序（从右往左替换，避免位移破坏后续 offset）。
2. 逐个 span：masked = masked[:start] + mask_token + masked[end:]。
3. 不做 whitespace 折叠。
4. 替换前检查 unit 内 span 是否重叠；重叠则跳过该 unit，计入 num_skipped_overlap。
5. masked_question 不应等于 original_question；mask_token 必须出现在 masked_question 中。
```

注意：本阶段 schema 不再使用 `num_masks` 字段；group 的正确性由 replace_each_span 算法保证，
并由 validate_masked_question_record 的 mask_token 存在性与 span 数量约束兜底。

---

## 16. only_necessary 过滤

CLI 必须支持可选 flag：

```text
--only-necessary
```

规则：

```text
未传入（默认）：对全部 unit 生成 masked question。
传入：仅当该 unit 的 semantic_sources 中至少一条 is_semantically_necessary == True 时才生成。
```

被过滤的 unit 计入 `num_filtered_not_necessary`。

---

## 17. semantic_sources 与 source id 列表

`semantic_sources` 是非空 list，元素与三个 source id 列表**索引平行、顺序一致**。

每个 `semantic_sources` 元素必须包含：

```text
semantic_label_id
nli_id
ablation_id
ablation_type
semantic_necessity_label
semantic_necessity_score
is_semantically_necessary
decision_reason
```

排序要求（确定、稳定）：

```text
推荐按 ablation_type 排序：delete 在前，generalize 在后。
source_semantic_label_ids[i] == semantic_sources[i]["semantic_label_id"]
source_nli_ids[i]            == semantic_sources[i]["nli_id"]
source_ablation_ids[i]       == semantic_sources[i]["ablation_id"]
```

validate_masked_question_record 会强制以上长度相等与索引对齐，构造时必须保证四者同序。

---

## 18. 输出 Record Schema

每条 `masked_questions.jsonl` record 必须包含（16 字段）：

```text
masked_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
masked_question
mask_token
mask_backend
mask_strategy
source_semantic_label_ids
source_nli_ids
source_ablation_ids
semantic_sources
```

字段来源：

```text
来自 1E / semantic_labels.jsonl（unit 结构字段，取聚合组内任一来源记录）：
- id
- unit_id
- unit_scope
- group_type
- span_ids
- spans
- original_question

来自 1E 聚合（每个来源记录贡献一项）：
- source_semantic_label_ids
- source_nli_ids
- source_ablation_ids
- semantic_sources

由 1F 新增：
- masked_id
- masked_question
- mask_token
- mask_backend
- mask_strategy
```

---

## 19. 输出示例

```json
{
  "masked_id": "gsm8k_0001__unit_001__mask",
  "id": "gsm8k_0001",
  "unit_id": "unit_001",
  "unit_scope": "single",
  "group_type": "single",
  "span_ids": ["span_001"],
  "spans": [
    {"span_id": "span_001", "text": "3", "type": "number", "start": 8, "end": 9}
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

---

## 20. Schema Validator 说明

`src/recover_attention/schemas.py` 中已存在：

```python
validate_masked_question_record(record: dict) -> None
```

它已经检查（本 sprint 默认不重写，只需让输出通过）：

```text
1.  拒绝旧字段 span_id / span_text / span_type。
2.  16 个顶层必含字段完整。
3.  masked_id / id / unit_id 非空 string。
4.  unit_scope ∈ {single, group}；group_type 非空 string。
5.  original_question / masked_question 非空 string，且 masked_question != original_question。
6.  mask_token 非空 string，且 mask_token 出现在 masked_question 中。
7.  mask_backend / mask_strategy 非空 string。
8.  span_ids 非空 list[str]；spans 非空 list[dict]；长度一致；span_id 顺序一致。
9.  每个 span 含 span_id / text / type / start / end；type 合法；start>=0；end>start。
10. unit_scope=single -> span_ids 长度为 1；group -> 长度 >= 2。
11. source_semantic_label_ids / source_nli_ids / source_ablation_ids 非空 list[str]。
12. semantic_sources 非空 list，长度等于三个 source id 列表。
13. 每个 semantic_sources 元素含 8 字段，label / score / bool / 枚举合法。
14. semantic_sources 与三个 source id 列表索引对齐。
```

模块和 CLI 必须对每条输出调用该 validator。

---

## 21. 核心模块要求

新增或实现：

```text
src/recover_attention/masked_questions.py
```

推荐常量：

```python
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_MASK_BACKEND = "unit_mask_v0"
DEFAULT_MASK_STRATEGY = "replace_each_span"
SUPPORTED_MASK_BACKENDS = {"unit_mask_v0"}
UNIT_STRUCTURE_FIELDS = (
    "id", "unit_id", "unit_scope", "group_type",
    "span_ids", "spans", "original_question",
)
SEMANTIC_SOURCE_FIELDS = (
    "semantic_label_id", "nli_id", "ablation_id", "ablation_type",
    "semantic_necessity_label", "semantic_necessity_score",
    "is_semantically_necessary", "decision_reason",
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
    strategy: str = DEFAULT_MASK_STRATEGY,
) -> dict
```

要求：

```text
1. _validate_backend(backend)。
2. 对每条输入调用 validate_semantic_label_record。
3. 校验同一 unit 的结构字段一致（UNIT_STRUCTURE_FIELDS），否则 ValueError。
4. 按 ablation_type（delete 在前）稳定排序来源记录。
5. masked_question = apply_mask_to_unit(original_question, base_unit, mask_token)。
6. 生成 source_semantic_label_ids / source_nli_ids / source_ablation_ids / semantic_sources（同序）。
7. 组装记录（masked_id / mask_token / mask_backend / mask_strategy / 16 字段）。
8. 调用 validate_masked_question_record。
```

---

```python
build_masked_question_records(
    semantic_label_records: list[dict],
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    strategy: str = DEFAULT_MASK_STRATEGY,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]
```

要求：

```text
1. _validate_backend(backend)，校验 mask_token 非空。
2. 按 (id, unit_id) 分组并保持首次出现顺序。
3. only_necessary=True 且该 unit 无任一 is_semantically_necessary -> 跳过(num_filtered_not_necessary)。
4. unit 内 span 重叠 -> 跳过(num_skipped_overlap)。
5. 其余构造一条 masked record。
6. 汇总 stats。
```

stats 至少包含：

```python
{
    "num_input_labels": 0,
    "num_units": 0,
    "num_output_masks": 0,
    "num_filtered_not_necessary": 0,
    "num_skipped_overlap": 0,
    "mask_token": "[MASK]",
    "mask_backend": "unit_mask_v0",
    "mask_strategy": "replace_each_span",
    "only_necessary": False,
    "unit_scope_counts": {},
    "group_type_counts": {},
    "source_count_distribution": {},
}
```

`source_count_distribution`：按每个 unit 的来源记录条数统计，例如 `{1: n, 2: m}`。

---

```python
build_masked_question_file(
    input_path: str | Path,
    output_path: str | Path,
    mask_token: str = DEFAULT_MASK_TOKEN,
    backend: str = DEFAULT_MASK_BACKEND,
    strategy: str = DEFAULT_MASK_STRATEGY,
    only_necessary: bool = False,
) -> tuple[list[dict], dict]
```

要求：

```text
read_jsonl -> build_masked_question_records -> write_jsonl，返回 (records, stats)。
```

---

## 22. CLI 脚本要求

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
  --backend unit_mask_v0
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
--backend unit_mask_v0
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

如果 backend 不在 SUPPORTED_MASK_BACKENDS，CLI 必须失败并给出清楚错误。

---

## 23. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_input_labels
num_units
num_output_masks
num_filtered_not_necessary
num_skipped_overlap
mask_token
mask_backend
mask_strategy
only_necessary
unit_scope_counts
group_type_counts
source_count_distribution
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 24. 文档要求

接口、label_schema、SKILL、experiment_guide 已对齐 unit-level，**本 sprint 默认不改文档**。

仅需确认：

```text
1. docs/skill/masked_questions_interface.md 与实现一致。
2. docs/skill/label_schema.md §10 字段列表与实现一致。
3. docs/skill/SKILL.md 已索引 masked_questions_interface.md。
4. docs/skill/experiment_guide.md pipeline 含 semantic_labels.jsonl → masked_questions.jsonl。
```

如发现文档与实现不一致，先报告，再做最小修正。

---

## 25. 测试要求

新增：

```text
tests/test_masked_questions.py
```

至少覆盖：

```text
1.  single unit -> masked_question 含 mask_token，spans 长度 1。
2.  group(number_set, 2 spans) -> 每个 span 各替换为一个 mask_token。
3.  repeated_surface group -> 每个出现位置各一个 mask_token。
4.  同一 unit 的 delete+generalize 两条 1E 记录 -> 只产出 1 条 masked 记录，
    semantic_sources 含 2 项，且与三个 source id 列表同序对齐。
5.  masked_id == f"{id}__{unit_id}__mask"。
6.  mask_backend == unit_mask_v0；mask_strategy == replace_each_span。
7.  only_necessary=True 时，全 Equivalent 的 unit 被过滤，num_filtered_not_necessary 计数正确。
8.  unit 内重叠 span 被跳过，num_skipped_overlap 计数正确。
9.  自定义 mask_token（如 "___"）生效。
10. 输出不含旧字段 span_id/span_text/span_type，也不含未来字段
    recoverable / recovered_question / recoverability_label / confidence。
11. 每条输出可通过 validate_masked_question_record。
12. unsupported backend 抛 ValueError。
13. batch stats 字段齐全
    （num_input_labels / num_units / num_output_masks / source_count_distribution）。
14. CLI smoke test：使用临时 semantic_labels.jsonl 运行脚本，
    验证输出文件存在、每行是合法 JSON、含 16 字段。
```

如有必要补充 `tests/test_schemas.py`（validator 已实现，通常只需正向覆盖）：

```text
A. validate_masked_question_record 接受合法 unit-level record（single 与 group）。
B. 拒绝旧字段 span_id / span_text / span_type。
C. 拒绝 semantic_sources 与 source id 列表长度不一致 / 顺序不对齐。
D. 拒绝 group 但 span_ids 长度 < 2。
```

---

## 26. 本 sprint 不做

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

## 27. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
python -m pytest -q
```

如果当前环境要求使用 conda run，则使用：

```bash
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
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

## 28. PROGRESS.md 更新要求

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
python scripts/07_build_masked_questions.py --input data/processed/semantic_labels.jsonl --output data/processed/masked_questions.jsonl --mask-token "[MASK]" --backend unit_mask_v0
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

## 29. 验收标准

本 sprint 完成后应满足：

```text
1.  开始前已回顾 1E / masked_questions 接口文档，并确认四处 schema 一致。
2.  已报告是否存在接口冲突。
3.  src/recover_attention/masked_questions.py 存在。
4.  scripts/07_build_masked_questions.py 存在。
5.  tests/test_masked_questions.py 存在。
6.  CLI 支持 --backend unit_mask_v0、--mask-token、--only-necessary。
7.  unsupported backend 会失败。
8.  masked_questions.jsonl 已生成。
9.  每条 masked record 按 (id, unit_id) 聚合，含 16 个顶层字段。
10. masked_id == f"{id}__{unit_id}__mask"。
11. mask_backend == unit_mask_v0；mask_strategy == replace_each_span。
12. single -> 1 个 span；group -> >=2 个 span，且每个 span 各替换为一个 mask_token。
13. masked_question != original_question，且包含 mask_token。
14. semantic_sources 与 source_semantic_label_ids / source_nli_ids / source_ablation_ids 同序对齐。
15. 输出不含旧 span-level 字段，也不含未来阶段字段。
16. 每条输出可通过 validate_masked_question_record。
17. python -m pytest -q 通过。
18. smoke test 通过。
19. 未调用真实模型。
20. 未下载模型。
21. 未新增 transformers / torch 依赖。
22. 未生成 recover_outputs.jsonl / recover_scores.jsonl。
23. 未修改 recover / trajectory / attention guidance / probe 相关文件。
24. PROGRESS.md 保持短版状态索引。
```

---

## 30. 完成后回复格式

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
9.  source_count_distribution
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
