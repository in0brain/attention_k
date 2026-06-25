# Sprint 1G：Question Recovery Stub

## 1. 目标

本 sprint 实现 unit-level question recovery 的最小可运行 stub 版本。

目标文件流：

```text
data/processed/masked_questions.jsonl
→ data/processed/recover_outputs.jsonl
```

本 sprint 只负责读取 Sprint 1F 的 masked question records，并生成 recovery output records。

本 sprint 不做 recoverability scoring，不做 confidence，不做 recoverability label，不做 attention guidance，不做 probe，不做 hidden states，不做 trajectory analysis，不调用真实模型。

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
→ masked_questions.jsonl
```

本 sprint 接在 1F 后面：

```text
masked_questions.jsonl
→ recover_outputs.jsonl
```

下一步才可能是：

```text
recover_outputs.jsonl
→ recover_scores.jsonl
```

因此，本 sprint 只输出 recovery samples，不输出 recoverability scores，不输出 final labels，不输出 token labels。

---

## 3. 权威接口

本 sprint 必须严格遵守：

```text
docs/skill/recover_outputs_interface.md
src/recover_attention/schemas.py 的 REQUIRED_FIELDS["recover_output"]
src/recover_attention/schemas.py 的 validate_recover_output_record
```

输入 masked question 必须符合：

```text
docs/skill/masked_questions_interface.md
src/recover_attention/schemas.py 的 validate_masked_question_record
```

不要在本 task card 中复制完整 schema。字段以 `schemas.py` 和 interface 文档为准。

---

## 4. 设计原则

1G 的职责边界：

```text
1F:
  构造 masked question 输入。

1G:
  对 masked question 生成 recovered_question sample。

1H:
  根据 recover_outputs 进行 recoverability scoring。
```

关键原则：

```text
1. recover_outputs.jsonl 是 unit-level / masked_id-driven。
2. 每条 recover output record 对应一个 masked question 的一个 recovery sample。
3. 一个 masked question 可以产生多个 sample，用 sample_id 区分。
4. recovery 的任务是复原问题缺失信息，不是解题。
5. 本 sprint 的 oracle_stub_v0 是 pipeline-verification stub，不代表真实模型恢复能力。
6. 本 sprint 不写 recoverable / confidence / reason / recoverability_label。
```

---

## 5. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1F 已完成。
data/processed/masked_questions.jsonl 已存在。
docs/skill/masked_questions_interface.md 已存在。
docs/skill/recover_outputs_interface.md 已存在。
validate_masked_question_record 已是 unit-level。
validate_recover_output_record 已是 unit-level / masked_id-driven。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/masked_questions.jsonl
Please run Sprint 1F first.
```

不要自动回头执行 Sprint 1A / 1B / 1C / 1D / 1E / 1F。

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

如果当前任务采用 `conda run` 运行命令，也必须明确报告。

---

## 7. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/skill/label_schema.md
docs/skill/experiment_guide.md
docs/skill/masked_questions_interface.md
docs/skill/recover_outputs_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/masked_questions.py
data/processed/masked_questions.jsonl
```

如果以下文件存在，也读取：

```text
src/recover_attention/recover_generation.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
docs/progress/sprint_1_history.md
```

不要读取：

```text
docs/reference/*
```

除非当前用户指令明确要求。

---

## 8. 接口一致性检查

开始实现前必须确认：

```text
1. python scripts/sync_interface_fields.py --check 通过。
2. python -m pytest tests/test_interface_consistency.py -q 通过。
3. recover_outputs_interface.md 的 required_fields marker 与 REQUIRED_FIELDS["recover_output"] 一致。
4. label_schema.md 只指向 recover_outputs_interface.md，不复制 recover output 完整字段表。
5. recover_output schema 不包含旧 span-level 顶层字段。
6. validate_recover_output_record 拒绝 span_id / span_text / span_type / recoverable / confidence / reason / recoverability_label。
```

如发现冲突，停止并报告：

```text
Interface conflict detected.
```

报告必须包括：

```text
1. 冲突文件
2. 冲突位置
3. 冲突内容
4. 应采用哪个接口作为当前阶段来源
5. 建议修正方式
```

---

## 9. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/recover_generation.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
data/processed/recover_outputs.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
src/recover_attention/recover_generation.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
docs/progress/sprint_1_history.md
```

`src/recover_attention/schemas.py` 默认不修改。

只有当发现 `validate_recover_output_record` 与 `docs/skill/recover_outputs_interface.md` 存在确凿冲突时，才允许在报告后做最小修正，并补充 schema 回归测试。

`docs/skill/*` 默认不修改。

只有当发现 interface 文档与当前实现接口不一致时，先报告，再做最小修正。

---

## 10. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reference/*
docs/skill/masked_questions_interface.md（默认）
docs/skill/recover_outputs_interface.md（默认）
docs/skill/ablation_units_interface.md
docs/skill/ablated_questions_interface.md
docs/skill/nli_scores_interface.md
docs/skill/semantic_labels_interface.md
docs/skill/method.md
docs/skill/prompts.md
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
src/recover_attention/masked_questions.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

本 sprint 禁止生成：

```text
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
data/processed/attention_anchor_labels.jsonl
```

本 sprint 禁止写入字段：

```text
recoverable
confidence
reason
recoverability_label
attention_anchor_label
guidance_action
hidden_states_path
attentions_path
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 11. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1. 是否确认当前进入 Sprint 1G：Question Recovery Stub。
2. 当前 Python 环境信息。
3. data/processed/masked_questions.jsonl 是否存在。
4. masked question record 数量。
5. masked_id 数量与是否唯一。
6. unit_scope 分布。
7. group_type 分布。
8. mask_backend 分布。
9. mask_strategy 分布。
10. python scripts/sync_interface_fields.py --check 是否通过。
11. python -m pytest tests/test_interface_consistency.py -q 是否通过。
12. validate_recover_output_record 是否已是 unit-level / masked_id-driven。
13. 本次计划使用的 recovery_backend。
14. 本次计划使用的 num_samples。
15. 是否明确不调用真实模型。
16. 是否明确不做 recoverability scoring / attention / probe。
17. 是否发现接口冲突。
```

如果发现接口冲突，停止并等待用户决定，不要继续实现。

---

## 12. 输入接口契约

输入文件：

```text
data/processed/masked_questions.jsonl
```

输入必须符合：

```text
docs/skill/masked_questions_interface.md
validate_masked_question_record
```

每条输入 record 在读取后必须调用：

```python
validate_masked_question_record(record)
```

如果输入 record 不合法，停止并抛出清楚错误。不要自动修复输入文件。

---

## 13. 输出接口契约

输出文件：

```text
data/processed/recover_outputs.jsonl
```

输出必须符合：

```text
docs/skill/recover_outputs_interface.md
validate_recover_output_record
```

每条输出 record 在写入前必须调用：

```python
validate_recover_output_record(record)
```

输出 record 必须是 unit-level / masked_id-driven。

不要使用旧 span-level 顶层字段。

---

## 14. recovery_backend

本 sprint 只支持：

```text
oracle_stub_v0
```

该 backend 的定位：

```text
pipeline-verification stub
```

它可以使用输入 masked question record 中的 `original_question` 和 unit metadata 来构造 deterministic recovered_question。

它不能作为真实模型恢复能力指标。

如果传入其他 backend，必须失败并给出清楚错误：

```text
Unsupported recovery backend: <backend>
```

---

## 15. oracle_stub_v0 规则

`oracle_stub_v0` 的默认行为：

```text
recovered_question = original_question
```

理由：

```text
1. 当前 sprint 只验证 pipeline 与 schema，不评估真实恢复能力。
2. original_question 是 masked question record 中的已知元数据。
3. 后续 1H scoring 可以用该 oracle output 验证评分管线。
```

允许 `recovered_question` 为空字符串的能力由 schema 支持，但 `oracle_stub_v0` 默认不输出空字符串。

不要在本 sprint 中调用 LLM、下载模型、调用外部 API 或执行真实 recovery。

---

## 16. sample_id 与 num_samples

CLI 必须支持：

```text
--num-samples
```

默认：

```text
--num-samples 1
```

规则：

```text
1. num_samples 必须是 int，且 >= 1。
2. 对每条 masked question record，生成 sample_id = 0 到 num_samples-1。
3. 同一 masked_id 下可以有多条 recover output records。
4. oracle_stub_v0 下多个 sample 可以产生相同 recovered_question。
5. 输出顺序应稳定：按输入 masked question 顺序，再按 sample_id 升序。
```

---

## 17. 核心模块要求

新增或实现：

```text
src/recover_attention/recover_generation.py
```

推荐常量：

```python
DEFAULT_RECOVERY_BACKEND = "oracle_stub_v0"
SUPPORTED_RECOVERY_BACKENDS = {"oracle_stub_v0"}
DEFAULT_NUM_SAMPLES = 1
```

推荐函数：

```python
build_recovered_question_oracle_stub(masked_question_record: dict) -> str
```

要求：

```text
返回 masked_question_record["original_question"]。
不调用模型。
不做解题。
不做评分。
```

---

```python
build_recover_output_record(
    masked_question_record: dict,
    sample_id: int,
    backend: str = DEFAULT_RECOVERY_BACKEND,
) -> dict
```

要求：

```text
1. validate_masked_question_record(masked_question_record)。
2. 校验 backend。
3. 校验 sample_id >= 0。
4. 从 masked question record 复制 recover output 所需上游字段。
5. recovered_question 由 backend 生成。
6. recovery_backend = backend。
7. sample_id = sample_id。
8. 调用 validate_recover_output_record。
```

---

```python
build_recover_output_records(
    masked_question_records: list[dict],
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[list[dict], dict]
```

要求：

```text
1. validate backend。
2. validate num_samples >= 1。
3. 对每条 masked question 输入调用 validate_masked_question_record。
4. 对每条 masked question 生成 num_samples 条 recover output。
5. 每条输出调用 validate_recover_output_record。
6. 汇总 stats。
```

stats 至少包含：

```python
{
    "num_input_masks": 0,
    "num_output_recoveries": 0,
    "num_samples": 1,
    "recovery_backend": "oracle_stub_v0",
    "unit_scope_counts": {},
    "group_type_counts": {},
    "mask_backend_counts": {},
    "mask_strategy_counts": {},
}
```

---

```python
build_recover_output_file(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[list[dict], dict]
```

要求：

```text
read_jsonl -> build_recover_output_records -> write_jsonl，返回 (records, stats)。
```

如果输入文件不存在，抛出：

```text
Missing input: <path>
Please run Sprint 1F first.
```

---

## 18. CLI 脚本要求

新增或实现：

```text
scripts/08_run_recovery.py
```

命令格式：

```bash
python scripts/08_run_recovery.py \
  --input data/processed/masked_questions.jsonl \
  --output data/processed/recover_outputs.jsonl \
  --backend oracle_stub_v0 \
  --num-samples 1
```

支持参数：

```text
--input
--output
--backend
--num-samples
```

默认值：

```text
--backend oracle_stub_v0
--num-samples 1
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 build_recover_output_records。
4. 对每条输入调用 validate_masked_question_record。
5. 对每条输出调用 validate_recover_output_record。
6. 输出文件每行是一个 JSON object。
7. 在 stdout 打印 recovery 统计。
8. 不调用真实模型。
9. 不下载模型。
10. 不调用外部 API。
```

如果 backend 不在 SUPPORTED_RECOVERY_BACKENDS，CLI 必须失败并给出清楚错误。

如果 num_samples < 1，CLI 必须失败并给出清楚错误。

---

## 19. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_input_masks
num_output_recoveries
num_samples
recovery_backend
unit_scope_counts
group_type_counts
mask_backend_counts
mask_strategy_counts
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 20. 测试要求

新增：

```text
tests/test_recover_generation.py
```

至少覆盖：

```text
1. oracle_stub_v0 对 single unit 输出 recovered_question == original_question。
2. oracle_stub_v0 对 group unit 输出 recovered_question == original_question。
3. 每条输出通过 validate_recover_output_record。
4. 输出不含旧字段 span_id / span_text / span_type。
5. 输出不含 scoring 字段 recoverable / confidence / reason / recoverability_label。
6. sample_id 从 0 开始。
7. num_samples=3 时，每个 masked_id 生成 3 条输出，sample_id 为 0/1/2。
8. 输出顺序稳定：输入顺序优先，sample_id 升序。
9. unsupported backend 抛 ValueError。
10. num_samples=0 抛 ValueError。
11. 输入 masked question record 不合法时抛 ValueError。
12. build_recover_output_file 写出 jsonl，并可读回。
13. CLI smoke test 可生成 recover_outputs.jsonl。
14. CLI stdout 包含 [OK] 或明确成功信息。
15. batch stats 字段齐全。
```

可以复用 tests 中已有的 masked question fixture，但不要依赖 data/processed 本地产物。

---

## 21. 本 sprint 不做

本 sprint 不做：

```text
真实模型 recovery
LLM API 调用
prompt judge
recoverability scoring
confidence 计算
recoverable yes/no 判断
recoverability_label
recover_scores.jsonl
labels.jsonl
token_labels.jsonl
attention anchor labels
attention guidance
hidden states
trajectory analysis
probe training
```

---

## 22. 必须运行命令

优先使用当前环境中的 python：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
python -m pytest -q
```

如果当前 shell 的 python 不在 `recover_attention` 环境，则使用：

```bash
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
conda run -n recover_attention python scripts/08_run_recovery.py --input data/processed/masked_questions.jsonl --output data/processed/recover_outputs.jsonl --backend oracle_stub_v0 --num-samples 1
conda run -n recover_attention python -m pytest -q
```

如果当前目录是 git 仓库，再运行：

```bash
git diff --name-only
git status --short
```

---

## 23. PROGRESS.md 更新要求

完成后更新：

```text
PROGRESS.md
```

要求：

```text
1. 当前阶段改为 Sprint 1G 已完成。
2. 已完成 Sprint 摘要新增 Sprint 1G。
3. 当前可运行命令增加 scripts/08_run_recovery.py。
4. 最近一次检查结果增加 question recovery stub。
5. 当前关键文件状态增加 recover_generation.py、08_run_recovery.py、test_recover_generation.py、recover_outputs.jsonl。
6. 下一步建议改为 Sprint 1H-prep：Recover Score Interface Alignment。
7. 遗留问题保留 data/processed/* 被 .gitignore 忽略的说明。
8. 明确 recover_scores.jsonl 仍需 unit-level 接口修正。
```

同时更新：

```text
docs/progress/sprint_1_history.md
```

追加 Sprint 1G 简短记录。

---

## 24. 验收标准

完成后必须满足：

```text
1. src/recover_attention/recover_generation.py 存在。
2. scripts/08_run_recovery.py 存在。
3. tests/test_recover_generation.py 存在。
4. data/processed/recover_outputs.jsonl 本地生成成功。
5. 每条输入 masked question record 都先经过 validate_masked_question_record。
6. 每条输出 recover output record 都经过 validate_recover_output_record。
7. 输出 recover_outputs.jsonl 是 unit-level / masked_id-driven。
8. 输出不含 span_id / span_text / span_type。
9. 输出不含 recoverable / confidence / reason / recoverability_label。
10. oracle_stub_v0 输出 recovered_question == original_question。
11. num_samples 控制每个 masked_id 的 sample 数。
12. unsupported backend 会失败。
13. num_samples < 1 会失败。
14. sync_interface_fields.py --check 通过。
15. tests/test_interface_consistency.py 通过。
16. smoke test 通过。
17. python -m pytest -q 通过。
18. 未生成 recover_scores.jsonl。
19. 未生成 labels.jsonl / token_labels.jsonl。
20. 未进入 attention / probe / hidden states 阶段。
```

---

## 25. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 修改文件
3. recovery backend 行为说明
4. recover_outputs.jsonl 输出摘要
5. 统计信息
6. 运行命令
7. 测试结果
8. 是否确认未做 recoverability scoring
9. 是否确认未调用真实模型
10. 是否确认未生成 recover_scores.jsonl
11. 后续建议
```

后续建议只写：

```text
可以继续 Sprint 1H-prep：Recover Score Interface Alignment。
```
