# Sprint 1H：Recoverability Scoring

## 1. 目标

本 sprint 实现 recoverability scoring 的最小可运行版本。

目标文件流：

```text
data/processed/recover_outputs.jsonl
→ data/processed/recover_scores.jsonl
```

本 sprint 只负责读取 Sprint 1G 生成的 unit-level recovery samples，并按 `masked_id` 聚合生成 unit-level recoverability score records。

本 sprint 不做真实模型调用，不做 semantic similarity，不做 NLI 判断，不做 attention anchor labeling，不做 trajectory stability，不做 answer stability，不做 hidden states，不做 attention maps，不做 probe，不做 attention guidance。

---

## 2. 当前阶段定位

当前已完成：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
→ nli_scores.jsonl
→ semantic_labels.jsonl
→ masked_questions.jsonl
→ recover_outputs.jsonl
→ recover_scores interface alignment
```

本 sprint 接在 1G 和 1H-prep 后面：

```text
recover_outputs.jsonl
→ recover_scores.jsonl
```

下一步才可能是：

```text
recover_scores.jsonl
→ attention_anchor_labels.jsonl
```

因此，本 sprint 只输出 recoverability scores，不输出 attention anchor labels，不输出 final labels，不输出 token labels。

---

## 3. 权威接口

本 sprint 必须严格遵守：

```text
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
src/recover_attention/schemas.py 的 REQUIRED_FIELDS["recover_output"]
src/recover_attention/schemas.py 的 REQUIRED_FIELDS["recover_score"]
src/recover_attention/schemas.py 的 validate_recover_output_record
src/recover_attention/schemas.py 的 validate_recover_score_record
```

不要在本 task card 中复制完整 schema。字段以 `schemas.py` 和 interface 文档为准。

---

## 4. 核心设计原则

1H 的职责边界：

```text
1G:
  对 masked question 生成 recovered_question samples。

1H:
  聚合 recovery samples，生成 recoverability score。

后续阶段:
  综合 semantic necessity、recoverability、trajectory、answer stability、raw attention 等信号，生成 attention anchor label。
```

关键原则：

```text
1. recover_scores.jsonl 是 unit-level / masked_id-driven。
2. 每个 masked_id 生成一条 recover score record。
3. recover_scores.jsonl 是聚合结果，不代表单个 sample。
4. sample-level 字段 sample_id / recovered_question 不能作为顶层字段写入 recover_scores.jsonl。
5. recoverability 不是最终 attention importance label。
6. 不要把 Non-recoverable 直接等同于 Strong Anchor。
7. 不要把 Recoverable 直接等同于不重要。
```

---

## 5. 本 sprint 的 scoring backend

本 sprint 只支持：

```text
stub_rule_v0
```

这是 deterministic rule backend，只用于管线验证和接口闭环。

禁止新增：

```text
local_model
api_model
semantic_similarity_model
nli_model
llm_judge
embedding_model
```

除非后续 sprint 明确允许。

---

## 6. `stub_rule_v0` 打分规则

本 sprint 使用最小 deterministic exact-match rule。

### 6.1 归一化规则

实现一个轻量 normalization 函数，例如：

```text
normalize_question(text):
  1. strip 前后空白
  2. 把连续空白折叠为单个空格
```

本 sprint 不做复杂语义归一化，不做大小写策略争议，不做数字语义等价，不做 paraphrase 判断。

### 6.2 单个 sample 是否 recover

对于每条 recover output：

```text
normalized_recovered_question == normalized_original_question
```

则该 sample 视为 exact recovered。

否则：

```text
recovered_question == "" 或只有空白:
  empty recovery

recovered_question 非空但不等于 original_question:
  non-empty mismatch
```

### 6.3 聚合分数

对同一个 `masked_id` 的所有 samples 聚合：

```text
num_samples = group size
source_sample_ids = 按 sample_id 升序排列
recovered_questions = 与 source_sample_ids 同序排列
exact_match_count = exact recovered sample 数
empty_count = 空 recovery 数
non_empty_mismatch_count = 非空但不匹配 original_question 的 recovery 数

recoverability_score = exact_match_count / num_samples
confidence_mean = recoverability_score
recovery_consistency = 最大 normalized recovered_question 出现次数 / num_samples
misleading_recovery = non_empty_mismatch_count > 0
```

### 6.4 标签规则

```text
if recoverability_score == 1.0:
    recoverability_label = "Recoverable"

elif recoverability_score > 0.0:
    recoverability_label = "Partially Recoverable"

elif recoverability_score == 0.0 and non_empty_mismatch_count > 0:
    recoverability_label = "Misleading Recovery"

else:
    recoverability_label = "Non-recoverable"
```

注意：

```text
Misleading Recovery 表示模型给出了非空但不等于原问题的恢复文本。
Non-recoverable 表示没有 exact recovery，且所有恢复为空或不可用。
```

---

## 7. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1G 已完成。
data/processed/recover_outputs.jsonl 已存在。
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md 已存在。
docs/reasoning-aware-attention-guidance/recover_scores_interface.md 已存在。
validate_recover_output_record 已是 unit-level / masked_id-driven。
validate_recover_score_record 已是 unit-level / masked_id-driven。
ALLOWED_RECOVER_SCORE_BACKENDS 包含 stub_rule_v0。
scripts/sync_interface_fields.py --check 通过。
tests/test_interface_consistency.py -q 通过。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/recover_outputs.jsonl
Please run Sprint 1G first.
```

不要自动回头执行 Sprint 1A–1G。

---

## 8. 环境要求

本项目使用既有 conda 环境：

```text
recover_attention
```

禁止创建新的 conda env。

修改文件前，Codex 必须报告：

```powershell
$env:CONDA_DEFAULT_ENV
where.exe python
python -c "import sys; print(sys.executable); print(sys.version)"
```

如果采用 `conda run` 运行命令，也必须明确报告。

---

## 9. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
tests/test_schemas.py
tests/test_interface_consistency.py
data/processed/recover_outputs.jsonl
docs/progress/sprint_1_history.md
```

不要读取：

```text
docs/reference/*
```

除非用户另行明确要求。

---

## 10. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/recover_scoring.py
scripts/09_score_recovery.py
tests/test_recover_scoring.py
data/processed/recover_scores.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
scripts/09_score_recovery.py
tests/test_recover_scoring.py
```

---

## 11. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reference/*
src/recover_attention/schemas.py
src/recover_attention/recover_generation.py
src/recover_attention/masked_questions.py
src/recover_attention/semantic_labels.py
src/recover_attention/nli_scoring.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/10_*
configs/*
requirements.txt
pyproject.toml
.gitignore
```

只有发现确凿接口冲突时，才允许先停止并报告，不要自行修改禁止列表中的文件。

---

## 12. Preflight 要求

修改任何文件前，必须先输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 data/processed/recover_outputs.jsonl 是否存在。
6. 当前 recover_outputs.jsonl record 数量。
7. 当前 recover_outputs.jsonl 中 masked_id 数量。
8. 当前 recovery_backend 分布。
9. 当前 sample_id 分布。
10. 当前 src/recover_attention/recover_scoring.py 是否仍是 TODO。
11. 当前 scripts/09_score_recovery.py 是否存在。
12. 当前 tests/test_recover_scoring.py 是否存在。
13. 当前 schemas.py 是否已有 ALLOWED_RECOVER_SCORE_BACKENDS = {"stub_rule_v0"}。
14. 当前 validate_recover_score_record 是否可用。
15. 本次计划实现的函数列表。
16. 本次必须运行的命令。
17. 是否发现冲突。
```

如果发现冲突，按 `AGENTS.md` 的优先级处理，并在 Preflight 中报告。

---

## 13. 实现要求

### 13.1 新增或实现模块

实现：

```text
src/recover_attention/recover_scoring.py
```

建议提供以下常量：

```python
DEFAULT_SCORE_BACKEND = "stub_rule_v0"
SUPPORTED_SCORE_BACKENDS = {"stub_rule_v0"}
```

建议提供以下函数：

```python
normalize_question(text: str) -> str

score_recovery_group_stub_rule_v0(recover_output_records: list[dict]) -> dict

build_recover_score_record(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> dict

build_recover_score_records(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> tuple[list[dict], dict]

build_recover_score_file(
    input_path: str | Path,
    output_path: str | Path,
    score_backend: str = DEFAULT_SCORE_BACKEND,
) -> tuple[list[dict], dict]
```

可以调整函数名，但必须保持职责清晰、可测试。

### 13.2 输入校验

要求：

```text
1. 每条输入 record 必须先调用 validate_recover_output_record。
2. 按 masked_id 分组。
3. 每个分组内必须确保以下字段一致：
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
   recovery_backend
4. 如果同一个 masked_id 下 source metadata 不一致，raise ValueError。
5. 如果同一个 masked_id 下 sample_id 重复，raise ValueError。
6. 如果 score_backend 不支持，raise ValueError。
7. 输出 record 写入前必须调用 validate_recover_score_record。
```

### 13.3 输出构造

每个 `masked_id` 分组生成一个 recover score record。

要求：

```text
1. source_sample_ids 按 sample_id 升序。
2. recovered_questions 与 source_sample_ids 同序。
3. num_samples = len(source_sample_ids)。
4. recover_score_id = f"{masked_id}__score_{score_backend}"。
5. score_backend = "stub_rule_v0"。
6. evidence 必须是 dict。
7. evidence 中必须记录本次 stub rule 的关键统计。
```

建议 evidence 包含：

```text
rule_name
normalization
num_exact_matches
num_empty_recoveries
num_non_empty_mismatches
exact_match_ratio
unique_normalized_recoveries
notes
```

不要把 hidden states、attention maps、trajectory、answer stability 或 attention guidance 信息写入 evidence。

---

## 14. CLI 要求

新增：

```text
scripts/09_score_recovery.py
```

CLI 参数：

```text
--input    默认无，必须显式传入
--output   默认无，必须显式传入
--backend  默认 stub_rule_v0
```

示例运行：

```bash
python scripts/09_score_recovery.py \
  --input data/processed/recover_outputs.jsonl \
  --output data/processed/recover_scores.jsonl \
  --backend stub_rule_v0
```

CLI 行为：

```text
1. 读取 recover_outputs.jsonl。
2. 调用 build_recover_score_file。
3. 写出 recover_scores.jsonl。
4. 打印 summary stats。
5. 如果 input 不存在，给出明确错误：Please run Sprint 1G first.
6. 如果 backend 不支持，退出并报告 Unsupported score backend。
```

建议输出统计：

```text
num_input_recoveries
num_output_scores
score_backend
recovery_backend_counts
recoverability_label_counts
unit_scope_counts
group_type_counts
num_misleading_recovery
```

---

## 15. 测试要求

新增：

```text
tests/test_recover_scoring.py
```

至少覆盖：

```text
1. 单条 oracle exact recovery → Recoverable。
2. 多 sample 全部 exact → Recoverable。
3. 多 sample 部分 exact、部分 empty → Partially Recoverable。
4. 全部 empty → Non-recoverable。
5. 非空但不匹配 original_question → Misleading Recovery。
6. sample_id 排序后 source_sample_ids 与 recovered_questions 同序。
7. 重复 sample_id 报错。
8. 同 masked_id 下 source metadata 不一致报错。
9. unsupported score_backend 报错。
10. 输出 record 通过 validate_recover_score_record。
11. 输出 record 不包含 span_id / sample_id / recovered_question / attention_anchor_label / guidance_action。
12. CLI smoke test 能从临时 recover_outputs.jsonl 写出 recover_scores.jsonl。
13. build_recover_score_file 在 input 缺失时给出明确 FileNotFoundError。
```

可以使用 pytest 的 `tmp_path` 构造临时输入输出，不要依赖大规模数据。

不要修改 `tests/test_schemas.py`，除非发现当前 schema validator 与已完成的 recover score interface 存在确凿 bug。若需要修改，先停止并报告。

---

## 16. 数据产物要求

本 sprint 可以生成：

```text
data/processed/recover_scores.jsonl
```

但必须注意：

```text
1. data/processed/* 是本地生成产物目录，可能被 .gitignore 忽略。
2. PROGRESS.md 可以记录生成结果，但不要声称该文件一定提交到 GitHub。
3. recover_scores.jsonl 由 oracle_stub_v0 recovery output 和 stub_rule_v0 scorer 得到，只能用于管线验证。
4. 不得将该结果解释为真实 recoverability 性能。
```

---

## 17. 本 sprint 不做

本 sprint 不做：

```text
1. 不修改 recover score schema。
2. 不修改 recover_scores_interface.md。
3. 不修改 recover_outputs_interface.md。
4. 不修改 label_schema.md。
5. 不修改 SKILL.md。
6. 不实现真实模型 recovery。
7. 不实现 semantic similarity scoring。
8. 不实现 NLI-based recovery judging。
9. 不调用 OpenAI API。
10. 不调用 Hugging Face 模型。
11. 不训练 probe。
12. 不缓存 hidden states。
13. 不缓存 attention maps。
14. 不做 trajectory stability。
15. 不做 answer stability。
16. 不做 attention anchor labeling。
17. 不做 attention guidance。
18. 不自动开始下一 sprint。
```

---

## 18. 必须运行命令

至少运行：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
conda run -n recover_attention python -m pytest tests/test_recover_scoring.py -q
conda run -n recover_attention python -m pytest -q
```

建议在运行 1H 前也先确认 1G 输入仍能生成：

```bash
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
```

如果当前 shell 已明确处于 `recover_attention` 环境，也可以使用：

```bash
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
python -m pytest tests/test_recover_scoring.py -q
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
1. 将当前阶段更新为 Sprint 1H 已完成：Recoverability Scoring。
2. 在已完成 Sprint 摘要中新增 Sprint 1H。
3. 在当前可运行命令中新增 scripts/09_score_recovery.py 命令。
4. 最近一次检查结果中记录 recoverability scoring passed。
5. 当前关键文件状态中新增：
   src/recover_attention/recover_scoring.py
   scripts/09_score_recovery.py
   tests/test_recover_scoring.py
   data/processed/recover_scores.jsonl
6. 遗留问题中说明：
   - stub_rule_v0 只做 exact normalized match。
   - 当前 recover_scores 来自 oracle_stub_v0 + stub_rule_v0，只用于管线验证。
   - 未做真实模型 recovery judging。
   - 未做 semantic similarity。
   - 未做 attention anchor labeling / guidance。
7. 下一步建议写成：
   Sprint 1I：Attention Anchor Label Interface Alignment 或 Sprint 1I-prep：Unit-to-Anchor Label Interface Design。
```

不要声称 recoverability scoring 具有真实实验意义。

---

## 20. docs/progress/sprint_1_history.md 更新要求

更新：

```text
docs/progress/sprint_1_history.md
```

追加小节：

```text
## Sprint 1H：Recoverability Scoring
```

内容包括：

```text
1. 已完成内容。
2. 新增或修改文件。
3. 输入文件：
   data/processed/recover_outputs.jsonl
4. 输出文件：
   data/processed/recover_scores.jsonl
5. 运行命令。
6. 检查结果。
7. recover score 数量统计。
8. recoverability_label 分布。
9. 遗留问题。
10. 下一步建议。
```

---

## 21. 验收标准

本 sprint 完成后必须满足：

```text
1. src/recover_attention/recover_scoring.py 已实现。
2. scripts/09_score_recovery.py 已实现。
3. tests/test_recover_scoring.py 已实现。
4. data/processed/recover_scores.jsonl 可由 recover_outputs.jsonl 生成。
5. 每个 masked_id 聚合为一条 recover score record。
6. 输出 record 全部通过 validate_recover_score_record。
7. 输出 record 不包含禁止字段：
   span_id
   span_text
   span_type
   sample_id
   recovered_question
   recoverable
   confidence
   reason
   attention_importance_score
   attention_anchor_label
   guidance_action
   guidance_strength
8. scripts/sync_interface_fields.py --check 通过。
9. tests/test_interface_consistency.py -q 通过。
10. tests/test_recover_scoring.py -q 通过。
11. 全量 pytest 通过。
12. PROGRESS.md 已更新。
13. docs/progress/sprint_1_history.md 已更新。
14. 未修改 recover_scores_interface.md。
15. 未修改 schemas.py。
16. 未调用真实模型。
17. 未进入 attention anchor labeling。
18. 未进入 attention guidance。
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

下一步建议可以写：

```text
Sprint 1I-prep：Unit-to-Anchor Label Interface Design
```

不要自动开始 Sprint 1I。
