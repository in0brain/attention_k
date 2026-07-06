# Sprint 1J：Build Attention Anchor Labels

建议保存路径：

```text
docs/codex_tasks/sprint_1J_build_attention_anchor_labels.md
```

---

## 1. 目标

本 sprint 只实现一个数据转换阶段：

```text
data/processed/unit_evidence.jsonl
→
data/processed/attention_anchor_labels.jsonl
```

本 sprint 按照已经完成的 `attention_anchor_labels` unit-level interface，从 `unit_evidence.jsonl` 构造 early attention anchor label records。

本 sprint 使用 deterministic stub rule：

```text
early_evidence_rule_stub_v0
```

本 sprint 输出的 label 只能用于 pipeline validation 和后续接口闭环，不代表真实 attention importance 结论。

---

## 2. 当前阶段定位

当前已经完成：

```text
Sprint 1J-prep：Attention Anchor Label Interface Alignment
```

因此当前应已有：

```text
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
src/recover_attention/schemas.py 中的 REQUIRED_FIELDS["attention_anchor_label"]
src/recover_attention/schemas.py 中的 FORBIDDEN_FIELDS["attention_anchor_label"]
src/recover_attention/schemas.py 中的 validate_attention_anchor_label_record
src/recover_attention/schemas.py 中的 INTERFACE_DOCS["attention_anchor_label"]
```

本 sprint 不重新设计接口。

本 sprint 只实现：

```text
unit_evidence.jsonl
→
attention_anchor_labels.jsonl
```

---

## 3. 本 sprint 的输入与输出

### 3.1 输入文件

```text
data/processed/unit_evidence.jsonl
```

### 3.2 输出文件

```text
data/processed/attention_anchor_labels.jsonl
```

### 3.3 新增代码文件

```text
src/recover_attention/attention_anchor_labels.py
scripts/11_build_attention_anchor_labels.py
tests/test_attention_anchor_labels.py
```

---

## 4. 核心原则

本 sprint 只做 early label construction。

它回答：

```text
基于当前已有 partial evidence，这个 unit 暂时应被标成哪类 early attention anchor label？
```

它不回答：

```text
这个 unit 是否已经被真实证明是最终 attention anchor？
```

它也不输出：

```text
guidance_action
guidance_strength
```

关键边界：

```text
1. attention_anchor_label 是 label decision，不是 intervention。
2. attention_importance_score 是 early evidence stub score，不是真实 attention importance。
3. 当前 evidence 只有 semantic necessity 与 semantic recoverability。
4. trajectory stability 尚未接入。
5. answer stability 尚未接入。
6. raw attention pattern 尚未接入。
7. attention steering effect 尚未接入。
8. guidance_action / guidance_strength 属于后续 attention guidance 阶段。
9. 本 sprint 不调用真实模型。
10. 本 sprint 不读取 hidden states / attention maps。
```

---

## 5. 权威接口

本 sprint 必须严格遵守：

```text
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
src/recover_attention/schemas.py
```

其中：

```text
REQUIRED_FIELDS["attention_anchor_label"]
FORBIDDEN_FIELDS["attention_anchor_label"]
validate_attention_anchor_label_record
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
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/unit_evidence.py
scripts/10_build_unit_evidence.py
tests/test_unit_evidence.py
tests/test_schemas.py
data/processed/unit_evidence.jsonl
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
src/recover_attention/attention_anchor_labels.py
scripts/11_build_attention_anchor_labels.py
tests/test_attention_anchor_labels.py
data/processed/attention_anchor_labels.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

其中以下文件如果不存在，可以新建：

```text
src/recover_attention/attention_anchor_labels.py
scripts/11_build_attention_anchor_labels.py
tests/test_attention_anchor_labels.py
data/processed/attention_anchor_labels.jsonl
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
src/recover_attention/schemas.py
src/recover_attention/unit_evidence.py
src/recover_attention/recover_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/recover_generation.py
src/recover_attention/masked_questions.py
src/recover_attention/nli_scoring.py
scripts/10_build_unit_evidence.py
scripts/12_*
tests/test_schemas.py
tests/test_interface_consistency.py
tests/test_unit_evidence.py
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
5. 当前 PROGRESS.md 是否显示 Sprint 1J-prep 已完成。
6. 当前 docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md 是否存在。
7. 当前 schemas.py 是否已有 validate_attention_anchor_label_record。
8. 当前 schemas.py 是否已有 REQUIRED_FIELDS["attention_anchor_label"]。
9. 当前 schemas.py 是否已有 INTERFACE_DOCS["attention_anchor_label"]。
10. 当前 data/processed/unit_evidence.jsonl 是否存在。
11. unit_evidence.jsonl record 数量。
12. unit_evidence 中 evidence_status 分布。
13. unit_evidence 中 available_signal_types 分布。
14. unit_evidence 中 missing_signal_types 分布。
15. 当前 src/recover_attention/attention_anchor_labels.py 是否存在。
16. 当前 scripts/11_build_attention_anchor_labels.py 是否存在。
17. 当前 tests/test_attention_anchor_labels.py 是否存在。
18. 本次是否会修改 schema/interface 文档。
19. 本次是否会生成 guidance_action / guidance_strength。
20. 本次是否会调用真实模型。
21. 本次必须运行的命令。
22. 是否发现冲突。
```

第 18 项必须回答：

```text
否，除非发现已完成的 1J-prep 存在阻塞性错误；若发现，先停止并报告。
```

第 19 项必须回答：

```text
否。
```

第 20 项必须回答：

```text
否。
```

---

## 10. attention anchor label 构造规则

### 10.1 输入粒度

每条 `unit_evidence` record 生成一条 `attention_anchor_label` record。

输入与输出是一一对应关系：

```text
unit_evidence_id
→
attention_anchor_label_id
```

### 10.2 ID 规则

输出 ID 必须满足：

```python
attention_anchor_label_id = f"{unit_evidence_id}__anchor_{label_backend}"
```

默认 label backend：

```text
early_evidence_rule_stub_v0
```

### 10.3 label_status

当前固定为：

```text
partial_evidence_label
```

原因：

```text
当前只有 semantic necessity 与 semantic recoverability 两类 early evidence；
trajectory stability、answer stability、raw attention pattern、attention steering effect 尚未接入。
```

---

## 11. early_evidence_rule_stub_v0 规则

本 sprint 使用 deterministic rule。

该 rule 不是真实 attention importance 方法，只用于 pipeline validation。

### 11.1 输入信号

从 `unit_evidence` 中读取：

```text
semantic_evidence.summary_score
semantic_evidence.summary_label
recoverability_evidence.recoverability_label
recoverability_evidence.recoverability_score
recoverability_evidence.misleading_recovery
available_signal_types
missing_signal_types
evidence_status
```

如果必要字段不存在或类型错误，应 raise ValueError。

### 11.2 semantic_score

```python
semantic_score = semantic_evidence["summary_score"]
```

要求：

```text
0 <= semantic_score <= 1
```

如果缺失，raise ValueError。

### 11.3 recoverability_risk_score

根据 `recoverability_label` 生成一个 early risk score。

建议规则：

```python
if recoverability_label == "Misleading Recovery":
    recoverability_risk_score = 1.0
elif recoverability_label == "Non-recoverable":
    recoverability_risk_score = 0.75
elif recoverability_label == "Partially Recoverable":
    recoverability_risk_score = 0.50
elif recoverability_label == "Recoverable":
    recoverability_risk_score = 0.25
else:
    raise ValueError
```

注意：

```text
Non-recoverable 不直接等同于 Strong Anchor。
Recoverable 不直接等同于不重要。
这里只是 early risk score。
```

### 11.4 attention_importance_score

建议规则：

```python
attention_importance_score = 0.6 * semantic_score + 0.4 * recoverability_risk_score
```

并限制到：

```text
0 <= attention_importance_score <= 1
```

如果由于浮点误差超出边界，应进行安全 clamp：

```python
max(0.0, min(1.0, score))
```

### 11.5 attention_anchor_label

建议 deterministic label rule：

```python
if recoverability_label == "Misleading Recovery" or misleading_recovery is True:
    attention_anchor_label = "Risky Anchor"

elif attention_importance_score >= 0.75:
    attention_anchor_label = "Strong Anchor"

elif attention_importance_score >= 0.55:
    attention_anchor_label = "Medium Anchor"

elif attention_importance_score >= 0.35:
    attention_anchor_label = "Weak Anchor"

else:
    attention_anchor_label = "Distractor"
```

注意：

```text
1. Misleading Recovery 暂时标为 Risky Anchor，是 early warning label，不是 suppress action。
2. Strong / Medium / Weak / Distractor 都只是 partial evidence label。
3. 当前不输出 guidance_action。
4. 当前不输出 guidance_strength。
```

---

## 12. 输出 record 构造规则

每条输出 record 顶层字段以：

```text
REQUIRED_FIELDS["attention_anchor_label"]
```

为准。

### 12.1 从 unit_evidence 复制

从输入 record 复制：

```text
unit_evidence_id
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
semantic_evidence
recoverability_evidence
available_signal_types
missing_signal_types
```

### 12.2 由本阶段创建

本阶段创建：

```text
attention_anchor_label_id
attention_importance_score
attention_anchor_label
label_backend
label_status
evidence
```

### 12.3 evidence 内容

`evidence` 必须是 dict。

建议包含：

```text
rule_name
score_formula
semantic_score
recoverability_label
recoverability_risk_score
attention_importance_score
label_thresholds
label_reason
source_unit_evidence_id
source_files
available_signal_types
missing_signal_types
limitations
notes
```

`limitations` 必须包含：

```text
1. label is based on partial early evidence only.
2. trajectory stability is not included.
3. answer stability is not included.
4. raw attention pattern is not included.
5. attention steering effect is not included.
6. no guidance_action or guidance_strength is produced.
7. recoverability comes from oracle_stub_v0 + stub_rule_v0 in current pipeline.
```

### 12.4 输出校验

每条输出 record 写入前必须调用：

```python
validate_attention_anchor_label_record(record)
```

---

## 13. 实现要求

新增：

```text
src/recover_attention/attention_anchor_labels.py
```

建议实现以下常量：

```python
DEFAULT_ATTENTION_LABEL_BACKEND = "early_evidence_rule_stub_v0"
SUPPORTED_ATTENTION_LABEL_BACKENDS = {"early_evidence_rule_stub_v0"}
DEFAULT_ATTENTION_LABEL_STATUS = "partial_evidence_label"
```

建议实现以下函数：

```python
compute_recoverability_risk_score(recoverability_label: str) -> float

score_attention_anchor_stub_rule_v0(unit_evidence_record: dict) -> dict

build_attention_anchor_label_record(
    unit_evidence_record: dict,
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> dict

build_attention_anchor_label_records(
    unit_evidence_records: list[dict],
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> tuple[list[dict], dict]

build_attention_anchor_label_file(
    input_path: str | Path,
    output_path: str | Path,
    label_backend: str = DEFAULT_ATTENTION_LABEL_BACKEND,
) -> tuple[list[dict], dict]
```

可以新增私有 helper 函数，例如：

```python
_validate_label_backend
_validate_semantic_score
_validate_recoverability_label
_clamp_score
_build_stats
```

要求代码风格和已有模块保持一致，例如：

```text
recover_scoring.py
unit_evidence.py
```

---

## 14. CLI 要求

新增：

```text
scripts/11_build_attention_anchor_labels.py
```

CLI 参数：

```text
--input
--output
--backend
```

其中：

```text
--input    必填，输入 unit_evidence JSONL path
--output   必填，输出 attention anchor labels JSONL path
--backend  默认 early_evidence_rule_stub_v0
```

示例命令：

```bash
conda run -n recover_attention python scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
```

CLI 行为：

```text
1. 读取 unit_evidence.jsonl。
2. 调用 build_attention_anchor_label_file。
3. 写出 attention_anchor_labels.jsonl。
4. 对每条输出调用 validate_attention_anchor_label_record。
5. 打印 summary stats。
6. 输入文件缺失时给出明确 FileNotFoundError。
7. backend 不支持时给出明确 Unsupported attention label backend。
```

建议打印统计：

```text
num_input_unit_evidence
num_output_attention_anchor_labels
label_backend
label_status_counts
attention_anchor_label_counts
unit_scope_counts
group_type_counts
available_signal_type_counts
missing_signal_type_counts
score_min
score_max
score_mean
num_risky_anchor
```

---

## 15. 测试要求

新增：

```text
tests/test_attention_anchor_labels.py
```

至少覆盖以下测试。

### 15.1 正常构造

```text
1. 单条 unit_evidence → valid attention_anchor_label record。
2. 多条 unit_evidence → 多条 attention_anchor_label records。
3. 输出 record 全部通过 validate_attention_anchor_label_record。
4. attention_anchor_label_id 符合 f"{unit_evidence_id}__anchor_{label_backend}"。
```

### 15.2 score 规则

```text
1. semantic_score 与 recoverability_risk_score 正确进入 0.6 / 0.4 公式。
2. score 被限制在 0 到 1。
3. Recoverable → recoverability_risk_score = 0.25。
4. Partially Recoverable → recoverability_risk_score = 0.50。
5. Non-recoverable → recoverability_risk_score = 0.75。
6. Misleading Recovery → recoverability_risk_score = 1.0。
```

### 15.3 label 规则

```text
1. misleading_recovery=True → Risky Anchor。
2. recoverability_label == Misleading Recovery → Risky Anchor。
3. score >= 0.75 → Strong Anchor。
4. 0.55 <= score < 0.75 → Medium Anchor。
5. 0.35 <= score < 0.55 → Weak Anchor。
6. score < 0.35 → Distractor。
```

### 15.4 字段复制

```text
1. unit_evidence_id 被正确复制。
2. id / unit_id / unit_scope / group_type 被正确复制。
3. span_ids / spans 被正确复制。
4. semantic_evidence 被正确复制。
5. recoverability_evidence 被正确复制。
6. available_signal_types / missing_signal_types 被正确复制。
```

### 15.5 禁止字段

验证输出 record 顶层不包含：

```text
span_id
span_text
span_type
sample_id
recovered_question
recoverable
confidence
reason
guidance_action
guidance_strength
hidden_states
attention_maps
trajectory_analysis
answer_stability
raw_attention_pattern
probe_label
```

### 15.6 错误处理

```text
1. unsupported backend → ValueError。
2. unit_evidence input 缺失 → FileNotFoundError。
3. missing semantic_evidence.summary_score → ValueError。
4. invalid semantic_evidence.summary_score range → ValueError。
5. missing recoverability_evidence.recoverability_label → ValueError。
6. invalid recoverability label → ValueError。
7. invalid input unit_evidence record → validator 报错。
```

### 15.7 CLI smoke test

用 `tmp_path` 构造小输入，运行：

```text
scripts/11_build_attention_anchor_labels.py
```

确认：

```text
1. 输出文件存在。
2. 输出 jsonl 可读。
3. 输出 record 数量正确。
4. 输出 record 通过 validate_attention_anchor_label_record。
5. stdout 中包含 summary stats。
```

---

## 16. 数据产物要求

本 sprint 可以生成：

```text
data/processed/attention_anchor_labels.jsonl
```

但必须注意：

```text
1. data/processed/* 可能被 .gitignore 忽略。
2. PROGRESS.md 可以记录生成结果，但不要声称该文件一定提交到 GitHub。
3. attention_anchor_labels 当前来自 early_evidence_rule_stub_v0，只用于管线验证。
4. 当前标签基于 partial evidence。
5. 当前不代表真实 attention importance。
6. 当前不代表 attention guidance 已实现。
```

---

## 17. 本 sprint 不做

本 sprint 不做：

```text
1. 不修改 attention anchor label schema。
2. 不修改 attention_anchor_labels_interface.md。
3. 不修改 unit_evidence_interface.md。
4. 不修改 label_schema.md。
5. 不修改 SKILL.md。
6. 不修改 schemas.py。
7. 不修改 unit_evidence.py。
8. 不生成 guidance_action。
9. 不生成 guidance_strength。
10. 不实现 attention guidance。
11. 不新增 guidance result files。
12. 不读取 hidden states。
13. 不读取 attention maps。
14. 不做 trajectory stability。
15. 不做 answer stability。
16. 不做 raw attention analysis。
17. 不调用真实模型。
18. 不训练 probe。
19. 不做 hallucination reduction evaluation。
20. 不自动开始下一 sprint。
```

---

## 18. 必须运行命令

至少运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
conda run -n recover_attention python -m pytest tests/test_attention_anchor_labels.py -q
conda run -n recover_attention python -m pytest -q
```

建议在运行 1J 前确认上游输入仍可生成：

```bash
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
```

如果当前 shell 已明确处于 `recover_attention` 环境，也可以使用：

```bash
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python scripts/11_build_attention_anchor_labels.py --input data/processed/unit_evidence.jsonl --output data/processed/attention_anchor_labels.jsonl --backend early_evidence_rule_stub_v0
python -m pytest tests/test_attention_anchor_labels.py -q
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
1. 当前阶段更新为 Sprint 1J 已完成：Build Attention Anchor Labels。
2. 已完成 Sprint 摘要中新增 Sprint 1J。
3. 当前可运行命令中新增 scripts/11_build_attention_anchor_labels.py 命令。
4. 最近一次检查结果中记录 attention anchor label build passed。
5. 当前关键文件状态中新增：
   src/recover_attention/attention_anchor_labels.py
   scripts/11_build_attention_anchor_labels.py
   tests/test_attention_anchor_labels.py
   data/processed/attention_anchor_labels.jsonl
6. 遗留问题中说明：
   - attention_anchor_labels 当前由 early_evidence_rule_stub_v0 生成，只用于管线验证。
   - 当前 label_status 是 partial_evidence_label。
   - 当前 attention_importance_score 不是真实 attention importance。
   - trajectory stability、answer stability、raw attention pattern、attention steering effect 尚未接入。
   - guidance_action / guidance_strength 尚未接入。
   - 尚未实现 attention guidance。
7. 下一步建议：
   Sprint 1K-prep：Guidance Boundary and Intervention Manifest Review
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
## Sprint 1J：Build Attention Anchor Labels
```

内容包括：

```text
1. 已完成内容。
2. 新增或修改文件。
3. 输入文件：
   data/processed/unit_evidence.jsonl
4. 输出文件：
   data/processed/attention_anchor_labels.jsonl
5. 运行命令。
6. 检查结果。
7. attention anchor label 数量统计。
8. attention_anchor_label 分布。
9. score min / max / mean。
10. 遗留问题。
11. 下一步建议：Sprint 1K-prep：Guidance Boundary and Intervention Manifest Review。
```

---

## 21. 验收标准

本 sprint 完成后必须满足：

```text
1. src/recover_attention/attention_anchor_labels.py 已实现。
2. scripts/11_build_attention_anchor_labels.py 已实现。
3. tests/test_attention_anchor_labels.py 已实现。
4. data/processed/attention_anchor_labels.jsonl 可由 unit_evidence.jsonl 生成。
5. 每条 unit_evidence 生成一条 attention_anchor_label record。
6. 输出 record 全部通过 validate_attention_anchor_label_record。
7. 输出 record 顶层不包含 forbidden fields。
8. attention_anchor_label_id 符合 f"{unit_evidence_id}__anchor_{label_backend}"。
9. attention_importance_score 在 0 到 1 内。
10. label_backend 为 early_evidence_rule_stub_v0。
11. label_status 为 partial_evidence_label。
12. evidence 中明确当前 label 是 partial early evidence stub。
13. evidence 中明确 trajectory / answer / raw attention / guidance 尚未接入。
14. scripts/sync_interface_fields.py --check 通过。
15. tests/test_interface_consistency.py -q 通过。
16. tests/test_attention_anchor_labels.py -q 通过。
17. 全量 pytest 通过。
18. 未修改 schemas.py。
19. 未修改 attention_anchor_labels_interface.md。
20. 未修改 label_schema.md。
21. 未生成 guidance_action / guidance_strength。
22. 未实现 attention guidance。
23. 未调用真实模型。
24. 未做 trajectory / answer stability / raw attention / probe。
25. PROGRESS.md 已更新。
26. docs/progress/sprint_1_history.md 已更新。
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
Sprint 1K-prep：Guidance Boundary and Intervention Manifest Review
```

不要自动开始 Sprint 1K-prep。
