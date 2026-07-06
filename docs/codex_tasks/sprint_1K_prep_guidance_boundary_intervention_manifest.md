# Sprint 1K-prep：Guidance Boundary and Intervention Manifest Interface Alignment

建议保存路径：

```text
docs/codex_tasks/sprint_1K_prep_guidance_boundary_intervention_manifest.md
```

---

## 1. 目标

本 sprint 只做 interface / schema / document alignment。

目标是为后续阶段设计并注册：

```text
data/processed/intervention_manifest.jsonl
```

它的上游输入是：

```text
data/processed/attention_anchor_labels.jsonl
```

本 sprint 只设计接口，不实现 builder，不生成 `intervention_manifest.jsonl`，不执行 attention guidance。

---

## 2. 当前阶段定位

当前已经完成：

```text
Sprint 1J：Build Attention Anchor Labels
```

当前已形成：

```text
unit_evidence.jsonl
→
attention_anchor_labels.jsonl
```

下一阶段要设计：

```text
attention_anchor_labels.jsonl
→
intervention_manifest.jsonl
```

注意：

```text
intervention_manifest.jsonl 只是 planned intervention manifest。
它不是 attention guidance result。
它不是模型执行记录。
它不是 trajectory stability result。
它不是 answer stability result。
```

---

## 3. 为什么需要 1K-prep

当前项目已经进入 attention anchor label 之后的边界区。

如果直接做 attention guidance，容易违反当前阶段限制：

```text
1. 真实模型调用仍未允许。
2. hidden states / attention maps 仍未允许缓存。
3. trajectory stability 尚未接入。
4. answer stability 尚未接入。
5. raw attention pattern 尚未接入。
6. attention steering effect 尚未接入。
7. 当前 attention_anchor_labels 仍来自 early_evidence_rule_stub_v0，只用于管线验证。
```

因此，本 sprint 先设计一个 unit-level planned intervention manifest，明确：

```text
1. 可以从 attention_anchor_labels 选择哪些 unit 做后续 intervention。
2. 记录 planned intervention metadata。
3. 不执行模型。
4. 不写 hidden state / attention map path。
5. 不输出 guidance_action / guidance_strength。
6. 不声称 hallucination reduction。
```

---

## 4. 本 sprint 做什么

本 sprint 做：

```text
1. 新增 docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md。
2. 新增或对齐 intervention_manifest record schema。
3. 在 schemas.py 中新增 REQUIRED_FIELDS["intervention_manifest"]。
4. 在 schemas.py 中新增 FORBIDDEN_FIELDS["intervention_manifest"]。
5. 在 schemas.py 中新增 validate_intervention_manifest_record。
6. 在 schemas.py 中新增 intervention manifest 相关 enum。
7. 在 INTERFACE_DOCS 中登记 intervention_manifest。
8. 更新 label_schema.md，让它引用 intervention_manifest_interface.md。
9. 更新 SKILL.md 的 document router。
10. 更新 tests/test_interface_consistency.py。
11. 更新 tests/test_schemas.py。
12. 运行 sync_interface_fields.py --write / --check。
13. 更新 PROGRESS.md。
14. 更新 docs/progress/sprint_1_history.md。
```

---

## 5. 本 sprint 不做什么

本 sprint 不做：

```text
1. 不实现 intervention manifest builder。
2. 不新增 src/recover_attention/intervention_manifest.py。
3. 不新增 scripts/12_build_intervention_manifest.py。
4. 不新增 tests/test_intervention_manifest.py。
5. 不生成 data/processed/intervention_manifest.jsonl。
6. 不执行 attention guidance。
7. 不生成 guidance_action。
8. 不生成 guidance_strength。
9. 不生成 guided_answer。
10. 不生成 baseline_answer。
11. 不读取 hidden states。
12. 不读取 attention maps。
13. 不写 hidden_states_path。
14. 不写 attentions_path。
15. 不做 trajectory stability。
16. 不做 answer stability。
17. 不做 raw attention analysis。
18. 不训练 probe。
19. 不调用真实模型。
20. 不修改 attention_anchor_labels.py。
21. 不修改 scripts/11_build_attention_anchor_labels.py。
22. 不修改 tests/test_attention_anchor_labels.py。
23. 不自动开始 Sprint 1K。
```

---

## 6. 设计决策

本 sprint 采用以下设计决策：

```text
intervention_manifest.jsonl 采用 unit-level planned-only schema。
```

理由：

```text
1. attention_anchor_labels.jsonl 当前是 unit-level。
2. group unit 可能包含多个 spans。
3. span-level / token-level expansion 尚未实现。
4. intervention_manifest 当前只负责记录后续计划，不负责执行。
5. guidance execution 必须留到更后面的 sprint。
```

因此，当前 intervention manifest record 应绑定：

```text
attention_anchor_label_id
unit_evidence_id
id
unit_id
```

而不是绑定单个顶层：

```text
span_id
span_text
span_type
```

span 信息保留在：

```text
span_ids
spans
```

中。

---

## 7. 文件命名

新增 interface 文档：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
```

未来数据文件：

```text
data/processed/intervention_manifest.jsonl
```

新增 record type：

```text
intervention_manifest
```

新增 validator：

```python
validate_intervention_manifest_record(record: dict) -> None
```

---

## 8. 开始前必须读取

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
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
src/recover_attention/schemas.py
scripts/sync_interface_fields.py
tests/test_interface_consistency.py
tests/test_schemas.py
docs/progress/sprint_1_history.md
```

可按需读取：

```text
src/recover_attention/attention_anchor_labels.py
scripts/11_build_attention_anchor_labels.py
tests/test_attention_anchor_labels.py
```

目的只是确认上游已完成，不允许修改。

不要读取：

```text
docs/reference/*
```

除非用户另行明确要求。

---

## 9. 允许修改

本 sprint 允许修改：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/SKILL.md
src/recover_attention/schemas.py
tests/test_interface_consistency.py
tests/test_schemas.py
PROGRESS.md
docs/progress/sprint_1_history.md
```

其中：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
```

如果不存在，可以新建。

---

## 10. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/prompts.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reference/*
src/recover_attention/attention_anchor_labels.py
src/recover_attention/unit_evidence.py
src/recover_attention/recover_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/recover_generation.py
src/recover_attention/masked_questions.py
src/recover_attention/nli_scoring.py
src/recover_attention/intervention_manifest.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
data/processed/*
configs/*
requirements.txt
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因，不要自行修改。

---

## 11. Preflight 要求

修改任何文件前，必须先输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1J 已完成。
6. 当前 data/processed/attention_anchor_labels.jsonl 是否存在。
7. 当前 schemas.py 是否已有 REQUIRED_FIELDS["intervention_manifest"]。
8. 当前 schemas.py 是否已有 validate_intervention_manifest_record。
9. 当前 INTERFACE_DOCS 是否已有 intervention_manifest。
10. 当前 docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md 是否存在。
11. 当前 label_schema.md 是否仍包含旧 span-level Intervention Manifest Record。
12. 当前 label_schema.md 是否仍提到 hidden_states_path / attentions_path 作为 manifest 字段。
13. 本次是否会实现 intervention_manifest builder。
14. 本次是否会生成 data/processed/intervention_manifest.jsonl。
15. 本次是否会执行 attention guidance。
16. 本次是否会写 hidden_states_path / attentions_path。
17. 本次必须运行的命令。
18. 是否发现冲突。
```

第 13 项必须回答：

```text
否。
```

第 14 项必须回答：

```text
否。
```

第 15 项必须回答：

```text
否。
```

第 16 项必须回答：

```text
否。
```

---

## 12. intervention_manifest schema 设计

建议新增：

```python
REQUIRED_FIELDS["intervention_manifest"] = [
    "intervention_id",
    "attention_anchor_label_id",
    "unit_evidence_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "attention_importance_score",
    "attention_anchor_label",
    "label_backend",
    "label_status",
    "intervention_type",
    "target_scope",
    "intervention_backend",
    "intervention_status",
    "planned_operation",
    "evidence",
]
```

字段含义：

```text
intervention_id:
  当前 planned intervention manifest record 的稳定 ID。

attention_anchor_label_id:
  上游 attention_anchor_label record ID。

unit_evidence_id:
  上游 unit_evidence record ID。

id / unit_id / unit_scope / group_type / span_ids / spans / original_question:
  继承上游 unit metadata。

attention_importance_score / attention_anchor_label / label_backend / label_status:
  继承上游 attention anchor label 信息。

intervention_type:
  计划中的 intervention 类型。

target_scope:
  当前 intervention 作用粒度。

intervention_backend:
  当前 manifest 构造 backend。

intervention_status:
  当前必须是 planned-only。

planned_operation:
  描述后续计划如何处理该 unit，但不是执行结果。

evidence:
  记录 source、limitations、notes。
```

---

## 13. ID 规则

建议：

```python
intervention_id = f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"
```

理由：

```text
1. attention_anchor_label_id 已绑定 unit 与 label backend。
2. intervention_type 区分 mask / remove / replace 等计划。
3. intervention_backend 区分 manifest 构造规则。
```

---

## 14. 新增 enum

建议在 `schemas.py` 中新增：

```python
ALLOWED_INTERVENTION_TYPES = {"mask", "remove", "replace"}
ALLOWED_INTERVENTION_TARGET_SCOPES = {"unit"}
ALLOWED_INTERVENTION_BACKENDS = {"manifest_stub_v0"}
ALLOWED_INTERVENTION_STATUSES = {"planned_only"}
```

注意：

```text
target_scope 当前只允许 unit。
不要在本 sprint 支持 token / span。
span/token expansion 留给后续 sprint。
```

---

## 15. FORBIDDEN_FIELDS["intervention_manifest"]

新增：

```python
FORBIDDEN_FIELDS["intervention_manifest"] = [
    "span_id",
    "span_text",
    "span_type",
    "guidance_action",
    "guidance_strength",
    "baseline_answer",
    "guided_answer",
    "intervened_answer",
    "answer_changed",
    "trajectory_stability_score",
    "answer_stability_score",
    "raw_attention_score",
    "hidden_states",
    "attention_maps",
    "hidden_states_path",
    "attentions_path",
    "probe_label",
    "probe_confidence",
]
```

目的：

```text
1. 防止退回旧 span-level 顶层字段。
2. 防止把 manifest 写成 guidance execution result。
3. 防止提前进入 trajectory / answer stability。
4. 防止提前缓存 hidden states / attention maps。
5. 防止提前进入 probe-guided guidance。
```

---

## 16. validator 要求

新增：

```python
validate_intervention_manifest_record(record: dict) -> None
```

最小校验要求：

```text
1. record 必须是 dict。
2. 必须包含 REQUIRED_FIELDS["intervention_manifest"]。
3. 必须拒绝 FORBIDDEN_FIELDS["intervention_manifest"]。
4. intervention_id 必须是非空字符串。
5. attention_anchor_label_id 必须是非空字符串。
6. unit_evidence_id 必须是非空字符串。
7. id 必须是非空字符串。
8. unit_id 必须是非空字符串。
9. unit_scope 必须属于 ALLOWED_ABLATION_UNIT_SCOPES。
10. group_type 必须属于 ALLOWED_ABLATION_UNIT_GROUP_TYPES。
11. span_ids 必须是非空 str list。
12. spans 必须是非空 list，长度与 span_ids 一致。
13. spans 内部字段复用现有 span metadata 约束：
    span_id
    text
    type
    start
    end
14. span_ids 顺序必须与 spans 顺序一致。
15. unit_scope / group_type 的 single/group 约束与前面 unit-level validators 保持一致。
16. original_question 必须是非空字符串。
17. attention_importance_score 必须是 0 到 1 的 number。
18. attention_anchor_label 必须属于 ALLOWED_ATTENTION_ANCHOR_LABELS。
19. label_backend 必须属于 ALLOWED_ATTENTION_LABEL_BACKENDS。
20. label_status 必须属于 ALLOWED_ATTENTION_LABEL_STATUSES。
21. intervention_type 必须属于 ALLOWED_INTERVENTION_TYPES。
22. target_scope 必须属于 ALLOWED_INTERVENTION_TARGET_SCOPES。
23. intervention_backend 必须属于 ALLOWED_INTERVENTION_BACKENDS。
24. intervention_status 必须属于 ALLOWED_INTERVENTION_STATUSES。
25. planned_operation 必须是 dict。
26. evidence 必须是 dict 或 list；建议 dict。
27. intervention_id 必须等于 f"{attention_anchor_label_id}__intervention_{intervention_type}_{intervention_backend}"。
```

本 sprint 不要求深度检查 `planned_operation`，因为 builder 尚未实现。

---

## 17. intervention_manifest_interface.md 要求

新增：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
```

文档必须包括：

```text
1. 文件名：data/processed/intervention_manifest.jsonl
2. Purpose
3. Pipeline Position
4. Record Schema
5. Field Sources
6. ID Rule
7. Planned-only Boundary
8. Unit-level Boundary
9. Not Included
10. Validator
11. Example
```

### 17.1 Purpose 必须说明

```text
intervention_manifest.jsonl stores unit-level planned intervention records derived from attention_anchor_labels.jsonl.
```

并明确它不是：

```text
1. attention guidance result
2. model execution log
3. trajectory stability result
4. answer stability result
5. raw attention analysis
6. hidden-state cache manifest
7. probe training data
```

### 17.2 Pipeline Position 必须说明

```text
attention_anchor_labels.jsonl
->
intervention_manifest.jsonl
->
future intervention execution / guidance evaluation stages
```

### 17.3 Record Schema

必须加入 generated marker：

```text
<!-- required_fields:intervention_manifest -->
<!-- generated from schemas.py REQUIRED_FIELDS by scripts/sync_interface_fields.py — do not edit by hand -->
```

marker 后的字段代码块必须由脚本生成，不要手写。

### 17.4 Field Sources

说明字段来源：

```text
Copied from attention_anchor_labels.jsonl:
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

Created by intervention manifest stage:
  intervention_id
  intervention_type
  target_scope
  intervention_backend
  intervention_status
  planned_operation
  evidence
```

### 17.5 Planned-only Boundary

必须写清楚：

```text
1. intervention_manifest is planned-only.
2. It does not execute model inference.
3. It does not apply attention steering.
4. It does not contain baseline / guided answers.
5. It does not contain trajectory or answer stability scores.
6. It does not contain hidden states or attention maps.
7. It does not contain hidden_states_path or attentions_path.
```

### 17.6 Unit-level Boundary

必须写清楚：

```text
1. This interface is unit-level.
2. It does not use top-level span_id / span_text / span_type.
3. Multiple spans in a group unit remain inside span_ids / spans.
4. Span-level or token-level expansion belongs to a later stage.
```

### 17.7 Not Included

必须写清楚不包含：

```text
guidance_action
guidance_strength
baseline_answer
guided_answer
intervened_answer
trajectory_stability_score
answer_stability_score
raw_attention_score
hidden states
attention maps
hidden_states_path
attentions_path
probe label
probe confidence
```

### 17.8 Example

给一个最小示例即可。

示例必须体现：

```text
intervention_id
attention_anchor_label_id
unit_evidence_id
id + unit_id
span_ids / spans
attention_importance_score
attention_anchor_label
intervention_type = mask
target_scope = unit
intervention_backend = manifest_stub_v0
intervention_status = planned_only
planned_operation
```

示例 notes 必须说明：

```text
Example only; builder implementation belongs to a later sprint.
```

---

## 18. label_schema.md 更新要求

更新：

```text
docs/reasoning-aware-attention-guidance/label_schema.md
```

要求：

```text
1. 更新 Intervention Manifest Record 小节。
2. 明确当前稳定接口以 docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md 为准。
3. 明确旧版 span-level intervention manifest schema 已废弃。
4. 明确新接口是 unit-level，绑定 attention_anchor_label_id + id + unit_id。
5. 明确顶层不再使用 span_id / span_text / span_type。
6. 明确 hidden_states_path / attentions_path 不属于本 record。
7. 明确 guidance_action / guidance_strength 不属于本 record。
8. 明确该 record 是 planned-only，不是 execution result。
9. 不复制完整顶层字段表。
10. 第 0 节 out-of-scope examples 不应把 intervention_manifest 说成没有 interface 文档的 record。
```

不要大幅改写其他章节。

---

## 19. SKILL.md 更新要求

更新：

```text
docs/reasoning-aware-attention-guidance/SKILL.md
```

要求在 Document Router 中增加：

```text
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
```

用途说明：

```text
Intervention manifest 的长期接口文档，说明 intervention_manifest.jsonl schema、unit-level boundary、planned-only boundary、validator 和后续 intervention execution / guidance evaluation 消费方式。
```

不要改动 SKILL.md 的总体职责和优先级规则。

---

## 20. tests/test_interface_consistency.py 要求

更新 interface consistency 测试，使其自动覆盖：

```text
intervention_manifest
```

要求：

```text
1. intervention_manifest_interface.md marker 存在。
2. marker 后字段块与 REQUIRED_FIELDS["intervention_manifest"] 一致。
3. label_schema.md 指向 intervention_manifest_interface.md。
4. label_schema.md 不复制完整 REQUIRED_FIELDS["intervention_manifest"] 字段表。
5. label_schema.md 第 0 节 out-of-scope 示例不包含 intervention_manifest。
```

---

## 21. tests/test_schemas.py 要求

新增或更新 validator 测试，至少覆盖：

```text
1. valid unit-level intervention_manifest record passes。
2. missing required field fails。
3. forbidden top-level span_id fails。
4. forbidden top-level guidance_action fails。
5. forbidden top-level guidance_strength fails。
6. forbidden hidden_states_path fails。
7. forbidden attentions_path fails。
8. invalid intervention_id fails。
9. invalid intervention_type fails。
10. invalid target_scope fails。
11. invalid intervention_backend fails。
12. invalid intervention_status fails。
13. invalid attention_importance_score range fails。
14. invalid attention_anchor_label enum fails。
15. unit_scope single 但多个 span fails。
16. unit_scope group 但只有一个 span fails。
17. spans 与 span_ids 顺序不一致 fails。
18. planned_operation 不是 dict fails。
```

不要新增：

```text
tests/test_intervention_manifest.py
```

因为本 sprint 不实现 builder。

---

## 22. sync_interface_fields 要求

更新 schemas 和 interface doc 后，必须运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
```

然后运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
```

要求：

```text
--check 必须通过。
```

不要手改 interface 文档 generated block。

---

## 23. PROGRESS.md 更新要求

更新：

```text
PROGRESS.md
```

要求：

```text
1. 当前阶段更新为 Sprint 1K-prep 已完成：Guidance Boundary and Intervention Manifest Interface Alignment。
2. 已完成 Sprint 摘要中新增 Sprint 1K-prep。
3. 当前可运行命令中不新增 builder 命令，因为本 sprint 不实现 builder。
4. 最近一次检查结果中记录：
   intervention manifest interface alignment: passed
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中新增：
   docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
6. 下一阶段可能新增或修改：
   src/recover_attention/intervention_manifest.py
   scripts/12_build_intervention_manifest.py
   tests/test_intervention_manifest.py
   data/processed/intervention_manifest.jsonl
7. 遗留问题中说明：
   - intervention_manifest 目前只有接口和 validator，尚未实现 builder。
   - 当前 intervention_manifest interface 是 unit-level planned-only。
   - hidden_states_path / attentions_path 尚未接入。
   - guidance_action / guidance_strength 尚未接入。
   - 尚未执行 attention guidance。
   - trajectory stability、answer stability、raw attention pattern 仍未接入。
8. 下一步建议：
   Sprint 1K：Build Intervention Manifest
```

不要声称已经实现 intervention manifest builder 或 attention guidance。

---

## 24. docs/progress/sprint_1_history.md 更新要求

更新：

```text
docs/progress/sprint_1_history.md
```

追加小节：

```text
## Sprint 1K-prep：Guidance Boundary and Intervention Manifest Interface Alignment
```

内容包括：

```text
1. 已完成内容。
2. 新增或修改文件。
3. 输入文件：无数据输入，本轮只做接口设计。
4. 输出文件：无 data/processed 产物。
5. 运行命令。
6. 检查结果。
7. 遗留问题。
8. 下一步建议：Sprint 1K：Build Intervention Manifest。
```

---

## 25. 必须运行命令

至少运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --write
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python -m pytest tests/test_schemas.py -q
conda run -n recover_attention python -m pytest -q
```

如果当前 shell 已明确处于 `recover_attention` 环境，也可以使用：

```bash
python scripts/sync_interface_fields.py --write
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python -m pytest tests/test_schemas.py -q
python -m pytest -q
```

但最终回复中必须说明实际使用的是哪一种。

---

## 26. 验收标准

本 sprint 完成后必须满足：

```text
1. docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md 已新增。
2. intervention_manifest_interface.md 有 required_fields:intervention_manifest marker。
3. marker 后字段块由 sync_interface_fields.py 生成。
4. schemas.py 中 REQUIRED_FIELDS["intervention_manifest"] 已存在。
5. schemas.py 中 FORBIDDEN_FIELDS["intervention_manifest"] 已阻止旧 span-level 顶层字段。
6. schemas.py 中 INTERFACE_DOCS["intervention_manifest"] 已存在。
7. schemas.py 中 validate_intervention_manifest_record 已存在。
8. label_schema.md 指向 intervention_manifest_interface.md。
9. label_schema.md 不再把 intervention_manifest 描述为旧 span-level schema。
10. label_schema.md 不再把 hidden_states_path / attentions_path 作为 intervention_manifest 字段。
11. SKILL.md 路由中包含 intervention_manifest_interface.md。
12. tests/test_interface_consistency.py 覆盖 intervention_manifest interface。
13. tests/test_schemas.py 覆盖 unit-level intervention_manifest validator。
14. sync_interface_fields.py --check 通过。
15. tests/test_interface_consistency.py -q 通过。
16. tests/test_schemas.py -q 通过。
17. 全量 pytest 通过。
18. 未实现 intervention manifest builder。
19. 未新增 src/recover_attention/intervention_manifest.py。
20. 未新增 scripts/12_build_intervention_manifest.py。
21. 未新增 tests/test_intervention_manifest.py。
22. 未生成 data/processed/intervention_manifest.jsonl。
23. 未输出 guidance_action / guidance_strength。
24. 未执行 attention guidance。
25. 未调用真实模型。
26. 未做 trajectory / answer stability / raw attention / probe。
27. PROGRESS.md 已更新。
28. docs/progress/sprint_1_history.md 已更新。
```

---

## 27. 完成后回复格式

完成后请按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. docs/progress/sprint_1_history.md 更新摘要
7. 遗留问题
8. 下一步建议
```

下一步建议必须是：

```text
Sprint 1K：Build Intervention Manifest
```

不要自动开始 Sprint 1K。
