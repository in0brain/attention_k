# Sprint 0G：schema 与 Attention Anchor 标签体系对齐

## 1. 目标

把代码 schema 与新版 Reasoning-Aware Attention Guidance 主线做最小同步。

本 sprint 只做 schema 与测试对齐，不进入 Sprint 1，不实现任何实验模块。

需要同步的最小内容：

```text
span type 增加 object
ablation type 增加 replace
增加 attention_anchor_label 枚举
增加 guidance_action 枚举
增加 validate_attention_anchor_label_record
更新 tests/test_schemas.py
```

---

## 2. 前置条件

执行本 sprint 前，必须已经完成：

```text
Sprint 0F：文档主线对齐
```

也就是说：

```text
README.md、AGENTS.md、docs/reasoning-aware-attention-guidance/*、PROGRESS.md 已经对齐为 Reasoning-Aware Attention Guidance 主线。
```

如果 PROGRESS.md 没有 Sprint 0F 记录，停止并报告，不要继续执行本 sprint。

---

## 3. 开始前必须阅读

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/method.md
src/recover_attention/schemas.py
tests/test_schemas.py
```

不要读取：

```text
docs/reference/*
```

除非当前文件不足以判断 schema 要求，并先向用户说明原因。

---

## 4. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/schemas.py
tests/test_schemas.py
PROGRESS.md
```

---

## 5. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/*
docs/reference/*
docs/codex_tasks/*
configs/*
data/*
scripts/*
requirements.txt
pyproject.toml
.gitignore
src/recover_attention/candidate_extraction.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 attention guidance / probe / trajectory 相关新文件
```

---

## 6. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

Preflight 必须包括：

```text
1. 已读取文件列表
2. 是否确认 Sprint 0F 已完成
3. 本次允许修改文件
4. 本次禁止修改文件
5. 当前 schemas.py 中已有的枚举和校验函数
6. 当前 tests/test_schemas.py 中已有测试概况
7. 本次计划新增的枚举、函数和测试
8. 本次必须运行的命令
```

用户确认后才能修改文件。

---

## 7. 实现要求

### 7.1 span type

在 `src/recover_attention/schemas.py` 中，把 `ALLOWED_SPAN_TYPES` 增加：

```python
"object"
```

---

### 7.2 ablation type

在 `src/recover_attention/schemas.py` 中，把 `ALLOWED_ABLATION_TYPES` 增加：

```python
"replace"
```

---

### 7.3 attention anchor label 枚举

在 `src/recover_attention/schemas.py` 中增加：

```python
ALLOWED_ATTENTION_ANCHOR_LABELS = {
    "Strong Anchor",
    "Medium Anchor",
    "Weak Anchor",
    "Risky Anchor",
    "Distractor",
}
```

---

### 7.4 guidance action 枚举

在 `src/recover_attention/schemas.py` 中增加：

```python
ALLOWED_GUIDANCE_ACTIONS = {
    "boost",
    "suppress",
    "keep",
    "review",
}
```

---

### 7.5 新增校验函数

新增函数：

```python
validate_attention_anchor_label_record(record: dict) -> None
```

用于校验 `data/processed/attention_anchor_labels.jsonl` 的单行 record。

必需字段：

```text
id
span_id
span_text
span_type
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
evidence
```

字段要求：

```text
id: non-empty str
span_id: non-empty str
span_text: non-empty str
span_type: str, must be in ALLOWED_SPAN_TYPES
attention_importance_score: int or float, 0 <= value <= 1
attention_anchor_label: str, must be in ALLOWED_ATTENTION_ANCHOR_LABELS
guidance_action: str, must be in ALLOWED_GUIDANCE_ACTIONS
guidance_strength: int or float, 0 <= value <= 1
evidence: dict or list
```

实现风格应与现有 schema 校验函数保持一致。

校验失败时抛出 `ValueError`，错误信息应说明字段名或非法值。

---

## 8. 本 sprint 不做

不要实现以下函数：

```text
validate_baseline_cot_record
validate_baseline_trajectory_manifest_record
validate_intervention_manifest_record
validate_trajectory_stability_score_record
validate_answer_stability_score_record
validate_oracle_guidance_result_record
validate_probe_guidance_result_record
```

不要实现以下模块或功能：

```text
baseline CoT
candidate span extraction
ablation construction
NLI scoring
recovery
trajectory stability
answer stability
attention guidance
probe training
真实模型调用
hidden states 缓存
attention maps 缓存
```

---

## 9. 测试要求

更新 `tests/test_schemas.py`。

至少覆盖：

```text
object 是合法 span type
replace 是合法 ablation type
合法 attention anchor label record 可以通过校验
非法 attention_anchor_label 会报错
非法 guidance_action 会报错
attention_importance_score < 0 会报错
attention_importance_score > 1 会报错
guidance_strength < 0 会报错
guidance_strength > 1 会报错
缺少必需字段会报错
evidence 不是 dict 或 list 会报错
```

必须保证已有 schema 测试继续通过。

---

## 10. 必须运行命令

必须运行：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pytest -q
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

---

## 11. PROGRESS.md 更新要求

完成后，在 PROGRESS.md 中新增：

```text
### Sprint 0G：schema 与 Attention Anchor 标签体系对齐
```

记录：

```text
已同步 object span type
已同步 replace ablation type
已新增 attention anchor label 枚举
已新增 guidance action 枚举
已新增 validate_attention_anchor_label_record
已更新 tests/test_schemas.py
已运行 pytest / smoke test / prepare_data
未实现 baseline CoT
未实现 candidate span extraction
未实现 trajectory stability
未实现 attention guidance
未训练 probe
```

下一步建议写：

```text
Sprint 1A：Baseline CoT Schema
```

不要写旧的：

```text
Sprint 1A：candidate span extraction
```

---

## 12. 验收标准

本 sprint 完成后应满足：

```text
1. ALLOWED_SPAN_TYPES 包含 object。
2. ALLOWED_ABLATION_TYPES 包含 replace。
3. schemas.py 包含 ALLOWED_ATTENTION_ANCHOR_LABELS。
4. schemas.py 包含 ALLOWED_GUIDANCE_ACTIONS。
5. schemas.py 包含 validate_attention_anchor_label_record。
6. tests/test_schemas.py 覆盖新增合法与非法场景。
7. python -m pytest -q 通过。
8. smoke test 通过。
9. prepare_data 通过。
10. 未修改 docs/reasoning-aware-attention-guidance/*、README.md、AGENTS.md。
11. 未实现任何实验模块。
```

---

## 13. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. 遗留问题
7. 下一步建议
```

不要自动开始 Sprint 1A。
