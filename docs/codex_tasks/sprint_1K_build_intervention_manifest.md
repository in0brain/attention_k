# Sprint 1K：Build Intervention Manifest

建议保存路径：

```text
docs/codex_tasks/sprint_1K_build_intervention_manifest.md
```

---

## 1. 目标

本 sprint 只实现一个 planned-only 数据转换阶段：

```text
data/processed/attention_anchor_labels.jsonl
→
data/processed/intervention_manifest.jsonl
```

本 sprint 按照已经完成的 `intervention_manifest` unit-level planned-only interface，从 `attention_anchor_labels.jsonl` 构造 planned intervention manifest records。

本 sprint 使用 deterministic stub backend：

```text
manifest_stub_v0
```

本 sprint 输出的 manifest 只能用于 pipeline validation 和后续接口闭环，不代表已经执行 attention guidance。

---

## 2. 当前阶段定位

当前已经完成：

```text
Sprint 1K-prep：Guidance Boundary and Intervention Manifest Interface Alignment
```

因此当前应已有：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
src/recover_attention/schemas.py 中的 REQUIRED_FIELDS["intervention_manifest"]
src/recover_attention/schemas.py 中的 FORBIDDEN_FIELDS["intervention_manifest"]
src/recover_attention/schemas.py 中的 validate_intervention_manifest_record
src/recover_attention/schemas.py 中的 INTERFACE_DOCS["intervention_manifest"]
```

本 sprint 不重新设计接口。

本 sprint 只实现：

```text
attention_anchor_labels.jsonl
→
intervention_manifest.jsonl
```

---

## 3. 本 sprint 的输入与输出

### 3.1 输入文件

```text
data/processed/attention_anchor_labels.jsonl
```

### 3.2 输出文件

```text
data/processed/intervention_manifest.jsonl
```

### 3.3 新增代码文件

```text
src/recover_attention/intervention_manifest.py
scripts/12_build_intervention_manifest.py
tests/test_intervention_manifest.py
```

---

## 4. 核心原则

本 sprint 只做 planned intervention manifest construction。

它回答：

```text
后续如果要做 intervention / guidance evaluation，应该计划对哪些 unit 做什么类型的 intervention？
```

它不回答：

```text
模型执行后效果如何？
attention guidance 是否有效？
是否减少 hallucination？
```

它也不输出：

```text
guidance_action
guidance_strength
baseline_answer
guided_answer
intervened_answer
trajectory_stability_score
answer_stability_score
hidden_states_path
attentions_path
```

关键边界：

```text
1. intervention_manifest 是 planned-only manifest，不是 execution result。
2. 每条 manifest 只记录“计划”，不执行模型。
3. 当前 target_scope 只允许 unit。
4. 当前默认 intervention_type 使用 mask。
5. 当前 intervention_backend 固定为 manifest_stub_v0。
6. 当前 intervention_status 固定为 planned_only。
7. 本 sprint 不调用真实模型。
8. 本 sprint 不读取 hidden states / attention maps。
9. 本 sprint 不写 hidden_states_path / attentions_path。
10. 本 sprint 不实现 attention guidance。
11. 本 sprint 不做 trajectory / answer stability。
12. 本 sprint 不做 span/token-level expansion。
```

---

## 5. 权威接口

本 sprint 必须严格遵守：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
src/recover_attention/schemas.py
```

其中：

```text
REQUIRED_FIELDS["intervention_manifest"]
FORBIDDEN_FIELDS["intervention_manifest"]
validate_intervention_manifest_record
```

是输出 record 的权威 schema 与 validator。

不要在代码中手抄完整 schema 作为新的事实来源。

---

## 6. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/attention_anchor_labels.py
scripts/11_build_attention_anchor_labels.py
tests/test_attention_anchor_labels.py
tests/test_schemas.py
data/processed/attention_anchor_labels.jsonl
docs/progress/sprint_1_history.md
```

不要读取：

```text
docs/reference/*
```

除非用户另行明确要求。

---

## 7. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/intervention_manifest.py
scripts/12_build_intervention_manifest.py
tests/test_intervention_manifest.py
data/processed/intervention_manifest.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

其中以下文件如果不存在，可以新建：

```text
src/recover_attention/intervention_manifest.py
scripts/12_build_intervention_manifest.py
tests/test_intervention_manifest.py
data/processed/intervention_manifest.jsonl
```

---

## 8. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/prompts.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reference/*
src/recover_attention/schemas.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/unit_evidence.py
src/recover_attention/recover_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/recover_generation.py
src/recover_attention/masked_questions.py
src/recover_attention/nli_scoring.py
scripts/11_build_attention_anchor_labels.py
scripts/13_*
tests/test_schemas.py
tests/test_interface_consistency.py
tests/test_attention_anchor_labels.py
configs/*
requirements.txt
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因，不要自行修改。

---

## 9. Preflight 要求

修改任何文件前，必须先输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1K-prep 已完成。
6. 当前 docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md 是否存在。
7. 当前 schemas.py 是否已有 validate_intervention_manifest_record。
8. 当前 schemas.py 是否已有 REQUIRED_FIELDS["intervention_manifest"]。
9. 当前 schemas.py 是否已有 INTERFACE_DOCS["intervention_manifest"]。
10. 当前 data/processed/attention_anchor_labels.jsonl 是否存在。
11. attention_anchor_labels.jsonl record 数量。
12. attention_anchor_labels 中 attention_anchor_label 分布。
13. attention_anchor_labels 中 label_status 分布。
14. attention_anchor_labels 中 label_backend 分布。
15. 当前 src/recover_attention/intervention_manifest.py 是否存在。
16. 当前 scripts/12_build_intervention_manifest.py 是否存在。
17. 当前 tests/test_intervention_manifest.py 是否存在。
18. 本次是否会修改 schema/interface 文档。
19. 本次是否会生成 guidance_action / guidance_strength。
20. 本次是否会调用真实模型。
21. 本次是否会写 hidden_states_path / attentions_path。
22. 本次必须运行的命令。
23. 是否发现冲突。
```

第 18 项必须回答：

```text
否，除非发现已完成的 1K-prep 存在阻塞性错误；若发现，先停止并报告。
```

第 19 项必须回答：

```text
否。
```

第 20 项必须回答：

```text
否。
```

第 21 项必须回答：

```text
否。
```

---

## 10. intervention manifest 构造规则

### 10.1 输入粒度

每条 `attention_anchor_label` record 生成一条 `intervention_manifest` record。

输入与输出是一一对应关系：

```text
attention_anchor_label_id
→
intervention_id
```

### 10.2 不做筛选

本 sprint 默认不筛选 label。

也就是说：

```text
Strong Anchor
Medium Anchor
Weak Anchor
Risky Anchor
Distractor
```

全部生成 planned manifest。

原因：

```text
1. 当前 attention_anchor_labels 来自 early_evidence_rule_stub_v0。
2. 当前 label_status 是 partial_evidence_label。
3. 当前 attention_importance_score 不是真实 attention importance。
4. 如果现在只保留 Strong / Risky，会过度解释 stub label。
5. 后续真正执行或评估阶段再设计 selection strategy。
```

### 10.3 ID 规则

输出 ID 必须满足：

```python
intervention_id = f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"
```

默认：

```text
intervention_type = mask
intervention_backend = manifest_stub_v0
```

### 10.4 target_scope

当前固定为：

```text
unit
```

本 sprint 不支持：

```text
span
token
```

### 10.5 intervention_status

当前固定为：

```text
planned_only
```

---

## 11. 默认 intervention_type 规则

本 sprint 默认全部使用：

```text
mask
```

不要根据 label 自动切换成 `remove` 或 `replace`。

原因：

```text
1. mask 与前面 masked_questions / recoverability 管线一致。
2. remove / replace 会引入新的 intervention semantics。
3. 当前只是 planned-only manifest，不应该扩展执行语义。
4. remove / replace 可留给后续 intervention expansion sprint。
```

允许 CLI 暴露：

```text
--intervention-type mask
```

但当前 backend 只保证 `mask` 路径被测试和验收。

如果用户传入 `remove` 或 `replace`，可以支持生成 planned manifest，但不得执行，也不得生成 intervened_question。

---

## 12. 输出 record 构造规则

每条输出 record 顶层字段以：

```text
REQUIRED_FIELDS["intervention_manifest"]
```

为准。

### 12.1 从 attention_anchor_label 复制

从输入 record 复制：

```text
attention_anchor_label_id
unit_evidence_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
attention_importance_score
attention_anchor_label
label_backend
label_status
```

### 12.2 由本阶段创建

本阶段创建：

```text
intervention_id
intervention_type
target_scope
intervention_backend
intervention_status
planned_operation
evidence
```

### 12.3 planned_operation 内容

`planned_operation` 必须是 dict。

对于默认 `mask`，建议包含：

```text
operation_name
description
target_scope
target_span_ids
target_texts
mask_token
execution_required
```

示例：

```json
{
  "operation_name": "mask_unit",
  "description": "Mask all spans in the unit for a future intervention experiment.",
  "target_scope": "unit",
  "target_span_ids": ["span_001"],
  "target_texts": ["3"],
  "mask_token": "[MASK]",
  "execution_required": true
}
```

注意：

```text
planned_operation 不得包含 intervened_question。
planned_operation 不得包含 guided_answer。
planned_operation 不得包含 hidden_states_path。
planned_operation 不得包含 attentions_path。
planned_operation 不得包含 trajectory_stability_score。
planned_operation 不得包含 answer_stability_score。
```

### 12.4 evidence 内容

`evidence` 必须是 dict。

建议包含：

```text
source_attention_anchor_label_id
source_unit_evidence_id
source_files
intervention_backend
intervention_type
intervention_status
target_scope
selection_policy
limitations
notes
```

其中 `selection_policy` 当前固定表达为：

```text
all_attention_anchor_labels_included_for_pipeline_validation
```

`limitations` 必须包含：

```text
1. planned-only manifest; no model execution.
2. no attention steering is applied.
3. no guidance_action or guidance_strength is produced.
4. no baseline_answer / guided_answer / intervened_answer is produced.
5. no hidden states or attention maps are read or cached.
6. no trajectory stability or answer stability score is computed.
7. attention_anchor_labels come from early_evidence_rule_stub_v0 and partial_evidence_label.
```

### 12.5 输出校验

每条输出 record 写入前必须调用：

```python
validate_intervention_manifest_record(record)
```

---

## 13. 实现要求

新增：

```text
src/recover_attention/intervention_manifest.py
```

建议实现以下常量：

```python
DEFAULT_INTERVENTION_BACKEND = "manifest_stub_v0"
SUPPORTED_INTERVENTION_BACKENDS = {"manifest_stub_v0"}

DEFAULT_INTERVENTION_TYPE = "mask"
SUPPORTED_INTERVENTION_TYPES = {"mask", "remove", "replace"}

DEFAULT_TARGET_SCOPE = "unit"
SUPPORTED_TARGET_SCOPES = {"unit"}

DEFAULT_INTERVENTION_STATUS = "planned_only"
```

建议实现以下函数：

```python
build_planned_operation(
    attention_anchor_label_record: dict,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    mask_token: str = "[MASK]",
) -> dict

build_intervention_manifest_record(
    attention_anchor_label_record: dict,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = "[MASK]",
) -> dict

build_intervention_manifest_records(
    attention_anchor_label_records: list[dict],
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = "[MASK]",
) -> tuple[list[dict], dict]

build_intervention_manifest_file(
    input_path: str | Path,
    output_path: str | Path,
    intervention_type: str = DEFAULT_INTERVENTION_TYPE,
    intervention_backend: str = DEFAULT_INTERVENTION_BACKEND,
    mask_token: str = "[MASK]",
) -> tuple[list[dict], dict]
```

可以新增私有 helper 函数，例如：

```python
_validate_intervention_backend
_validate_intervention_type
_validate_target_scope
_build_stats
```

要求代码风格和已有模块保持一致，例如：

```text
attention_anchor_labels.py
unit_evidence.py
recover_scoring.py
```

---

## 14. CLI 要求

新增：

```text
scripts/12_build_intervention_manifest.py
```

CLI 参数：

```text
--input
--output
--intervention-type
--backend
--mask-token
```

其中：

```text
--input              必填，输入 attention_anchor_labels JSONL path
--output             必填，输出 intervention_manifest JSONL path
--intervention-type  默认 mask
--backend            默认 manifest_stub_v0
--mask-token         默认 [MASK]
```

示例命令：

```bash
conda run -n recover_attention python scripts/12_build_intervention_manifest.py --input data/processed/attention_anchor_labels.jsonl --output data/processed/intervention_manifest.jsonl --intervention-type mask --backend manifest_stub_v0 --mask-token "[MASK]"
```

CLI 行为：

```text
1. 读取 attention_anchor_labels.jsonl。
2. 调用 build_intervention_manifest_file。
3. 写出 intervention_manifest.jsonl。
4. 对每条输出调用 validate_intervention_manifest_record。
5. 打印 summary stats。
6. 输入文件缺失时给出明确 FileNotFoundError。
7. backend 不支持时给出明确 Unsupported intervention backend。
8. intervention_type 不支持时给出明确 Unsupported intervention type。
```

建议打印统计：

```text
num_input_attention_anchor_labels
num_output_intervention_manifest
intervention_backend
intervention_type_counts
intervention_status_counts
target_scope_counts
attention_anchor_label_counts
label_status_counts
unit_scope_counts
group_type_counts
num_planned_only
num_mask
num_remove
num_replace
```

---

## 15. 测试要求

新增：

```text
tests/test_intervention_manifest.py
```

至少覆盖以下测试。

### 15.1 正常构造

```text
1. 单条 attention_anchor_label → valid intervention_manifest record。
2. 多条 attention_anchor_label → 多条 intervention_manifest records。
3. 输出 record 全部通过 validate_intervention_manifest_record。
4. intervention_id 符合 f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"。
```

### 15.2 默认字段

```text
1. 默认 intervention_type = mask。
2. 默认 target_scope = unit。
3. 默认 intervention_backend = manifest_stub_v0。
4. 默认 intervention_status = planned_only。
5. planned_operation 是 dict。
6. planned_operation["operation_name"] == "mask_unit"。
7. planned_operation["mask_token"] == "[MASK]"。
```

### 15.3 字段复制

```text
1. attention_anchor_label_id 被正确复制。
2. unit_evidence_id 被正确复制。
3. id / unit_id / unit_scope / group_type 被正确复制。
4. span_ids / spans 被正确复制。
5. original_question 被正确复制。
6. attention_importance_score 被正确复制。
7. attention_anchor_label 被正确复制。
8. label_backend / label_status 被正确复制。
```

### 15.4 不做筛选

```text
1. Strong Anchor 生成 manifest。
2. Medium Anchor 生成 manifest。
3. Weak Anchor 生成 manifest。
4. Risky Anchor 生成 manifest。
5. Distractor 生成 manifest。
6. 输入 N 条 attention_anchor_labels，输出 N 条 intervention_manifest。
```

### 15.5 禁止字段

验证输出 record 顶层不包含：

```text
span_id
span_text
span_type
guidance_action
guidance_strength
baseline_answer
guided_answer
intervened_answer
answer_changed
trajectory_stability_score
answer_stability_score
raw_attention_score
hidden_states
attention_maps
hidden_states_path
attentions_path
probe_label
probe_confidence
```

同时验证：

```text
planned_operation 中也不包含 hidden_states_path / attentions_path / guided_answer / baseline_answer / trajectory_stability_score / answer_stability_score。
```

### 15.6 错误处理

```text
1. unsupported backend → ValueError。
2. unsupported intervention_type → ValueError。
3. missing input file → FileNotFoundError。
4. invalid input attention_anchor_label record → validator 报错。
5. invalid mask_token 空字符串 → ValueError。
```

### 15.7 CLI smoke test

用 `tmp_path` 构造小输入，运行：

```text
scripts/12_build_intervention_manifest.py
```

确认：

```text
1. 输出文件存在。
2. 输出 jsonl 可读。
3. 输出 record 数量正确。
4. 输出 record 通过 validate_intervention_manifest_record。
5. stdout 中包含 summary stats。
```

---

## 16. 数据产物要求

本 sprint 可以生成：

```text
data/processed/intervention_manifest.jsonl
```

但必须注意：

```text
1. data/processed/* 可能被 .gitignore 忽略。
2. PROGRESS.md 可以记录生成结果，但不要声称该文件一定提交到 GitHub。
3. intervention_manifest 当前来自 manifest_stub_v0，只用于管线验证。
4. 当前 intervention_status 是 planned_only。
5. 当前不代表 intervention 已执行。
6. 当前不代表 attention guidance 已实现。
7. 当前不代表 hallucination reduction 已验证。
```

---

## 17. 本 sprint 不做

本 sprint 不做：

```text
1. 不修改 intervention_manifest schema。
2. 不修改 intervention_manifest_interface.md。
3. 不修改 attention_anchor_labels_interface.md。
4. 不修改 label_schema.md。
5. 不修改 SKILL.md。
6. 不修改 schemas.py。
7. 不修改 attention_anchor_labels.py。
8. 不生成 guidance_action。
9. 不生成 guidance_strength。
10. 不实现 attention guidance。
11. 不新增 guidance result files。
12. 不读取 hidden states。
13. 不读取 attention maps。
14. 不写 hidden_states_path。
15. 不写 attentions_path。
16. 不生成 baseline_answer。
17. 不生成 guided_answer。
18. 不生成 intervened_answer。
19. 不做 trajectory stability。
20. 不做 answer stability。
21. 不做 raw attention analysis。
22. 不调用真实模型。
23. 不训练 probe。
24. 不做 hallucination reduction evaluation。
25. 不做目录重构。
26. 不自动开始下一 sprint。
```

---

## 18. 必须运行命令

至少运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/12_build_intervention_manifest.py --input data/processed/attention_anchor_labels.jsonl --output data/processed/intervention_manifest.jsonl --intervention-type mask --backend manifest_stub_v0 --mask-token "[MASK]"
conda run -n recover_attention python -m pytest tests/test_intervention_manifest.py -q
conda run -n recover_attention python -m pytest -q
```

建议在运行 1K 前确认上游输入仍可生成：

```bash
conda run -n recover_attention python scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
```

如果当前 shell 已明确处于 `recover_attention` 环境，也可以使用：

```bash
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python scripts/12_build_intervention_manifest.py --input data/processed/attention_anchor_labels.jsonl --output data/processed/intervention_manifest.jsonl --intervention-type mask --backend manifest_stub_v0 --mask-token "[MASK]"
python -m pytest tests/test_intervention_manifest.py -q
python -m pytest -q
```

但最终回复中必须说明实际使用的是哪一种。

---

## 19. PROGRESS.md 更新要求

更新：

```text
PROGRESS.md
```

要求：

```text
1. 当前阶段更新为 Sprint 1K 已完成：Build Intervention Manifest。
2. 已完成 Sprint 摘要中新增 Sprint 1K。
3. 当前可运行命令中新增 scripts/12_build_intervention_manifest.py 命令。
4. 最近一次检查结果中记录 intervention manifest build passed。
5. 当前关键文件状态中新增：
   src/recover_attention/intervention_manifest.py
   scripts/12_build_intervention_manifest.py
   tests/test_intervention_manifest.py
   data/processed/intervention_manifest.jsonl
6. 遗留问题中说明：
   - intervention_manifest 当前由 manifest_stub_v0 生成，只用于管线验证。
   - 当前 intervention_status 是 planned_only。
   - 当前 planned_operation 只是后续执行计划，不是执行结果。
   - 当前不包含 guidance_action / guidance_strength。
   - 当前不包含 hidden_states_path / attentions_path。
   - trajectory stability、answer stability、raw attention pattern、attention steering effect 尚未接入。
   - 尚未执行 attention guidance。
   - 尚未调用真实模型。
7. 下一步建议：
   Sprint 1L-prep：Sprint 1 Boundary Review and Refactor Freeze Plan
```

不要声称已经实现 attention guidance。

---

## 20. docs/progress/sprint_1_history.md 更新要求

更新：

```text
docs/progress/sprint_1_history.md
```

追加小节：

```text
## Sprint 1K：Build Intervention Manifest
```

内容包括：

```text
1. 已完成内容。
2. 新增或修改文件。
3. 输入文件：
   data/processed/attention_anchor_labels.jsonl
4. 输出文件：
   data/processed/intervention_manifest.jsonl
5. 运行命令。
6. 检查结果。
7. intervention_manifest 数量统计。
8. intervention_type 分布。
9. intervention_status 分布。
10. attention_anchor_label 分布。
11. 遗留问题。
12. 下一步建议：Sprint 1L-prep：Sprint 1 Boundary Review and Refactor Freeze Plan。
```

---

## 21. 验收标准

本 sprint 完成后必须满足：

```text
1. src/recover_attention/intervention_manifest.py 已实现。
2. scripts/12_build_intervention_manifest.py 已实现。
3. tests/test_intervention_manifest.py 已实现。
4. data/processed/intervention_manifest.jsonl 可由 attention_anchor_labels.jsonl 生成。
5. 每条 attention_anchor_label 生成一条 intervention_manifest record。
6. 输出 record 全部通过 validate_intervention_manifest_record。
7. 输出 record 顶层不包含 forbidden fields。
8. planned_operation 不包含 execution result / hidden state path / attention path。
9. intervention_id 符合 f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"。
10. intervention_type 默认为 mask。
11. target_scope 为 unit。
12. intervention_backend 为 manifest_stub_v0。
13. intervention_status 为 planned_only。
14. evidence 中明确当前 manifest 是 planned-only。
15. evidence 中明确 no model execution / no attention guidance。
16. evidence 中明确 no hidden states / no attention maps。
17. scripts/sync_interface_fields.py --check 通过。
18. tests/test_interface_consistency.py -q 通过。
19. tests/test_intervention_manifest.py -q 通过。
20. 全量 pytest 通过。
21. 未修改 schemas.py。
22. 未修改 intervention_manifest_interface.md。
23. 未修改 label_schema.md。
24. 未生成 guidance_action / guidance_strength。
25. 未执行 attention guidance。
26. 未调用真实模型。
27. 未做 trajectory / answer stability / raw attention / probe。
28. 未做目录重构。
29. PROGRESS.md 已更新。
30. docs/progress/sprint_1_history.md 已更新。
```

---

## 22. 完成后回复格式

完成后请按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. 生成数据摘要
6. PROGRESS.md 更新摘要
7. docs/progress/sprint_1_history.md 更新摘要
8. 遗留问题
9. 下一步建议
```

下一步建议必须是：

```text
Sprint 1L-prep：Sprint 1 Boundary Review and Refactor Freeze Plan
```

不要自动开始 Sprint 1L-prep。
