# Sprint 1M：Plug Real LLM Recovery Backend

建议保存路径：

```text
docs/codex_tasks/sprint_1M_plug_real_llm_recovery_backend.md
```

---

## 1. 目标

本 sprint 接入真实 LLM recovery backend，用于替换当前 `oracle_stub_v0` recovery stub。

当前 recovery 管线已经存在：

```text
data/processed/masked_questions.jsonl
→
data/processed/recover_outputs.jsonl
```

当前 `oracle_stub_v0` 直接返回 `original_question`，只能用于 pipeline verification，不代表真实恢复能力。

本 sprint 新增真实本地 LLM backend：

```text
ollama_chat_v0
```

该 backend 通过本地 Ollama HTTP API 调用本地 LLM，让模型根据 `masked_question` 尝试恢复缺失信息，并输出 `recovered_question`。

本 sprint 的核心目标是：

```text
真实 LLM recovery output
```

不是：

```text
recoverability scoring
attention anchor relabeling
attention guidance
trajectory stability
hidden states
probe training
目录重构
```

---

## 2. 当前阶段定位

当前已经完成：

```text
Sprint 1L：Plug Real Bilingual NLI Backend
```

当前真实 NLI 已接入，但 recovery 仍是：

```text
oracle_stub_v0
```

本 sprint 只替换 recovery generation backend，不改 downstream scoring 规则。

本 sprint 后，管线应支持：

```text
scripts/08_run_recovery.py --backend oracle_stub_v0
scripts/08_run_recovery.py --backend ollama_chat_v0
```

---

## 3. 本 sprint 使用的 Ollama 模型

用户本地已确认存在以下两个 Ollama 模型：

```text
qwen3.5:9b
llama3.1:8b
```

本 sprint 的模型策略固定为：

```text
主模型：
  qwen3.5:9b

备用模型：
  llama3.1:8b
```

规则：

```text
1. Preflight 必须运行 ollama list。
2. 如果 qwen3.5:9b 存在，真实 Ollama smoke 必须使用 qwen3.5:9b。
3. 如果 qwen3.5:9b 不存在，但 llama3.1:8b 存在，真实 Ollama smoke 使用 llama3.1:8b，并在最终回复中说明 fallback。
4. 如果二者都不存在，停止并报告，不要自动 pull。
5. 本 sprint 不允许自动运行 ollama pull。
6. 本 sprint 不新增 HuggingFace causal LM backend。
7. 本 sprint 不新增 OpenAI / Claude / Gemini API backend。
```

默认 CLI 参数应使用：

```text
--model qwen3.5:9b
```

备用 CLI 参数：

```text
--model llama3.1:8b
```

---

## 4. 最重要的防泄漏原则

真实 LLM recovery backend 绝对不能把答案泄漏给模型。

调用 LLM 时，prompt 中不得包含：

```text
original_question
spans[*].text
gold span text
source_semantic_label_ids
source_nli_ids
source_ablation_ids
semantic_sources
recoverability_label
attention_anchor_label
```

允许传给 LLM 的内容只有：

```text
masked_question
mask_token
number of masks
minimal instruction
```

可选传入：

```text
unit_scope
group_type
```

但不要传入 `spans` 的原始文本。

核心原因：

```text
real recovery 的目的，是测试被 mask 后的信息是否能从剩余问题上下文中恢复。
如果把 original_question 或 span text 放进 prompt，就等于 oracle leakage。
```

---

## 5. 本 sprint 新增 backend

新增 recovery backend：

```text
ollama_chat_v0
```

保留原 backend：

```text
oracle_stub_v0
```

不要删除、重命名或改变 `oracle_stub_v0` 行为。

建议常量：

```python
ORACLE_STUB_BACKEND = "oracle_stub_v0"
OLLAMA_CHAT_BACKEND = "ollama_chat_v0"

DEFAULT_RECOVERY_BACKEND = ORACLE_STUB_BACKEND

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
FALLBACK_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

SUPPORTED_RECOVERY_BACKENDS = {
    ORACLE_STUB_BACKEND,
    OLLAMA_CHAT_BACKEND,
}
```

---

## 6. schema 更新要求

允许修改：

```text
src/recover_attention/schemas.py
```

只允许扩展 recovery backend enum。

将：

```python
ALLOWED_RECOVERY_BACKENDS = {"oracle_stub_v0"}
```

扩展为：

```python
ALLOWED_RECOVERY_BACKENDS = {
    "oracle_stub_v0",
    "ollama_chat_v0",
}
```

不要修改：

```text
REQUIRED_FIELDS["recover_output"]
FORBIDDEN_FIELDS["recover_output"]
validate_recover_output_record
recover_score schema
unit_evidence schema
attention_anchor_label schema
intervention_manifest schema
```

---

## 7. interface 文档更新要求

本 sprint 必须更新：

```text
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
```

因为当前接口文档中 `recovery_backend` 仍只写了：

```text
oracle_stub_v0
```

需要改成：

```text
oracle_stub_v0
ollama_chat_v0
```

并说明：

```text
oracle_stub_v0:
  pipeline verification stub，可能使用 original_question，不代表真实恢复能力。

ollama_chat_v0:
  real local LLM recovery backend。
  只允许使用 masked_question / mask_token / minimal instruction。
  不允许把 original_question 或 spans[*].text 传给 LLM。
```

不要改 `recover_outputs` 顶层字段。

---

## 8. 本 sprint 输入与输出

### 8.1 输入文件

```text
data/processed/masked_questions.jsonl
```

### 8.2 主输出 schema

仍然复用：

```text
recover_outputs.jsonl schema
```

每条 record 顶层字段仍然只有：

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
recovered_question
recovery_backend
sample_id
```

### 8.3 Smoke 输出

不要默认覆盖：

```text
data/processed/recover_outputs.jsonl
```

真实 LLM smoke 输出到：

```text
outputs/logs/recover_outputs_ollama_small.jsonl
```

stub smoke 输出到：

```text
outputs/logs/recover_outputs_stub_check.jsonl
```

---

## 9. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/recover_generation.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
src/recover_attention/schemas.py
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
configs/v0_nli_small.yaml
requirements.txt
PROGRESS.md
docs/progress/sprint_1_history.md
```

---

## 10. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reference/*
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
tests/test_nli_scoring.py
tests/test_semantic_labels.py
tests/test_masked_questions.py
tests/test_recover_scoring.py
tests/test_unit_evidence.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
data/processed/*
models/*
pyproject.toml
.gitignore
```

例外：

```text
outputs/logs/recover_outputs_ollama_small.jsonl
outputs/logs/recover_outputs_stub_check.jsonl
```

这些文件只允许由本 sprint 的 CLI smoke 命令生成或覆盖。

---

## 11. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
src/recover_attention/schemas.py
src/recover_attention/recover_generation.py
src/recover_attention/data_io.py
scripts/08_run_recovery.py
tests/test_recover_generation.py
configs/v0_nli_small.yaml
data/processed/masked_questions.jsonl
docs/progress/sprint_1_history.md
```

不要读取：

```text
docs/reference/*
```

除非用户另行明确要求。

---

## 12. Preflight 要求

修改任何文件前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1L 已完成。
6. 当前 recover_generation.py 支持哪些 backend。
7. 当前 schemas.py 中 ALLOWED_RECOVERY_BACKENDS 的值。
8. 当前 recover_outputs_interface.md 是否只写了 oracle_stub_v0。
9. 当前 scripts/08_run_recovery.py 是否已有 --backend。
10. 当前 scripts/08_run_recovery.py 是否已有 --num-samples。
11. 当前 data/processed/masked_questions.jsonl 是否存在。
12. masked_questions record 数量。
13. 当前裸 python 路径与版本。
14. 当前 recover_attention 环境 python 路径与版本。
15. 本 sprint 实际运行命令是否全部使用 conda run -n recover_attention python。
16. 当前机器是否能执行 ollama 命令。
17. ollama list 原始输出。
18. ollama list 中是否存在 qwen3.5:9b。
19. ollama list 中是否存在 llama3.1:8b。
20. 本 sprint 实际使用的 Ollama model name。
21. 如果未使用 qwen3.5:9b，fallback 原因是什么。
22. 本次是否会自动运行 ollama pull。
23. 本次是否会把 original_question 传给 LLM。
24. 本次是否会把 spans[*].text 传给 LLM。
25. 本次是否会修改 recover_outputs 顶层 schema。
26. 本次是否会修改 recover_scoring。
27. 本次是否会重建 unit_evidence / attention_anchor_labels。
28. 本次是否会接入 attention guidance。
29. 本次必须运行的命令。
30. 是否发现冲突。
```

第 15 项必须回答：

```text
是，全部使用 conda run -n recover_attention python。
```

第 22 项必须回答：

```text
否。
```

第 23 项必须回答：

```text
否。
```

第 24 项必须回答：

```text
否。
```

第 25 项必须回答：

```text
否。
```

第 26 项必须回答：

```text
否。
```

第 27 项必须回答：

```text
否。
```

第 28 项必须回答：

```text
否。
```

---

## 13. Ollama backend 设计

### 13.1 默认 API

默认使用本地 Ollama API：

```text
http://localhost:11434/api/chat
```

CLI 参数：

```text
--ollama-base-url
```

默认：

```text
http://localhost:11434
```

实际请求 endpoint：

```text
{ollama_base_url}/api/chat
```

### 13.2 默认模型与 fallback

默认真实 smoke 使用：

```text
qwen3.5:9b
```

如果 `qwen3.5:9b` 不存在，但 `llama3.1:8b` 存在，允许 fallback：

```text
llama3.1:8b
```

但必须在最终回复中说明：

```text
qwen3.5:9b 不存在，因此 fallback 到 llama3.1:8b。
```

如果两个模型都不存在：

```text
停止并报告，不要自动下载。
```

---

## 14. CLI 更新要求

修改：

```text
scripts/08_run_recovery.py
```

当前已有：

```text
--input
--output
--backend
--num-samples
```

新增：

```text
--model
--ollama-base-url
--temperature
--top-p
--max-tokens
--timeout
--seed
--limit
```

默认值：

```text
--model qwen3.5:9b
--ollama-base-url http://localhost:11434
--temperature 0.0
--top-p 1.0
--max-tokens 128
--timeout 120
--seed 42
--limit None
```

`--limit` 要求：

```text
1. 只在 script 层截断 input_records。
2. 不改变 library 层默认行为。
3. 用于真实 LLM smoke test。
```

---

## 15. Prompt 设计要求

新增函数：

```python
build_recovery_prompt(masked_question_record: dict) -> str
```

Prompt 必须满足：

```text
1. 只包含 masked_question。
2. 只包含 mask_token。
3. 可以包含 mask 数量。
4. 不包含 original_question。
5. 不包含 spans[*].text。
6. 不要求模型解题。
7. 不要求模型给 reasoning。
8. 要求模型只输出恢复后的完整问题。
```

建议 prompt：

```text
You are given a question with one or more [MASK] tokens.
Recover the missing text using only the remaining context in the masked question.

Rules:
- Do not solve the question.
- Do not explain.
- Do not add reasoning.
- Replace every [MASK] token.
- Return only the recovered full question.

Masked question:
{masked_question}
```

如果 `mask_token` 不是 `[MASK]`，prompt 里必须使用实际 `mask_token`。

---

## 16. Ollama 调用要求

建议新增：

```python
call_ollama_chat(
    prompt: str,
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    timeout: int = 120,
    seed: int | None = 42,
) -> str
```

要求：

```text
1. 使用 Python 标准库 urllib.request，不强制新增 requests 依赖。
2. 调用 /api/chat。
3. stream=False。
4. messages 使用 user-only message 即可。
5. options 中设置 temperature / top_p / num_predict / seed。
6. 连接失败时抛出清晰 RuntimeError。
7. 模型不存在时抛出清晰 RuntimeError。
8. 返回 response["message"]["content"]。
9. 如果 response 结构异常，抛出清晰 RuntimeError。
```

请求 payload 示例：

```json
{
  "model": "qwen3.5:9b",
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 128,
    "seed": 42
  }
}
```

---

## 17. 输出清洗要求

新增：

```python
clean_recovered_question(text: str) -> str
```

要求：

```text
1. strip 前后空白。
2. 去掉 Markdown code fence。
3. 去掉常见前缀：
   Recovered question:
   Answer:
   Output:
4. 如果输出有多行，保留清洗后的完整文本，不要随便截断第一行，除非明显是解释。
5. 返回 str。
6. 如果模型输出空，返回空字符串。
```

不要在 `recovered_question` 中存储 JSON、reasoning、confidence 或 explanation。

---

## 18. recover_generation.py 实现要求

修改：

```text
src/recover_attention/recover_generation.py
```

### 18.1 保持旧函数兼容

当前函数必须保持旧调用可用：

```python
build_recover_output_record(masked_question_record, sample_id, backend=DEFAULT_RECOVERY_BACKEND)

build_recover_output_records(masked_question_records, backend=DEFAULT_RECOVERY_BACKEND, num_samples=DEFAULT_NUM_SAMPLES)

build_recover_output_file(input_path, output_path, backend=DEFAULT_RECOVERY_BACKEND, num_samples=DEFAULT_NUM_SAMPLES)
```

可以新增参数，但旧调用必须不变。

建议扩展：

```python
build_recover_output_record(
    masked_question_record: dict,
    sample_id: int,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    model: str = "qwen3.5:9b",
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    timeout: int = 120,
    seed: int | None = 42,
) -> dict
```

```python
build_recover_output_records(
    masked_question_records: list[dict],
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    model: str = "qwen3.5:9b",
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    timeout: int = 120,
    seed: int | None = 42,
) -> tuple[list[dict], dict]
```

```python
build_recover_output_file(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    model: str = "qwen3.5:9b",
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    timeout: int = 120,
    seed: int | None = 42,
) -> tuple[list[dict], dict]
```

### 18.2 Recovery 生成逻辑

```text
backend == oracle_stub_v0:
  保持原逻辑：返回 original_question。

backend == ollama_chat_v0:
  build_recovery_prompt(masked_question_record)
  call_ollama_chat(...)
  clean_recovered_question(...)
```

### 18.3 Stats 更新

stats 建议新增：

```text
model
ollama_base_url
temperature
top_p
max_tokens
timeout
num_empty_recoveries
```

这些只在 `stats` 中打印，不写入 `recover_outputs.jsonl` 顶层 record。

---

## 19. 禁止向 output record 增加字段

不要在 `recover_outputs.jsonl` 顶层增加：

```text
prompt
raw_response
model
temperature
top_p
max_tokens
latency
error
confidence
reason
recoverability_label
```

原因：

```text
recover_outputs schema 已稳定，模型运行元数据不属于该接口顶层字段。
```

---

## 20. tests/test_recover_generation.py 更新要求

必须保留全部 oracle stub 测试。

新增测试不得调用真实 Ollama。

新增测试至少覆盖：

```text
1. SUPPORTED_RECOVERY_BACKENDS 包含 oracle_stub_v0 / ollama_chat_v0。
2. schemas.py 的 ALLOWED_RECOVERY_BACKENDS 包含 ollama_chat_v0。
3. oracle_stub_v0 行为保持不变。
4. build_recovery_prompt 不包含 original_question。
5. build_recovery_prompt 不包含 spans[*].text。
6. build_recovery_prompt 包含 masked_question。
7. clean_recovered_question 能去掉 Recovered question: 前缀。
8. clean_recovered_question 能去掉 code fence。
9. ollama_chat_v0 可通过 monkeypatch fake call 生成 recovered_question。
10. ollama_chat_v0 输出 record 通过 validate_recover_output_record。
11. ollama_chat_v0 输出 record 顶层字段仍等于 REQUIRED_FIELDS["recover_output"]。
12. unsupported backend 仍 raise ValueError。
13. --limit 只输出前 N 条。
14. CLI smoke 的 oracle_stub_v0 仍通过。
15. CLI smoke 的 ollama_chat_v0 用 monkeypatch 或 fake function 覆盖，不调用真实服务。
```

关键防泄漏测试：

```python
prompt = build_recovery_prompt(masked_question_record)
assert original_question not in prompt
for span in spans:
    assert span["text"] not in prompt
```

---

## 21. 真实 Ollama smoke 要求

如果本机能运行 Ollama 且 `qwen3.5:9b` 存在，必须运行：

```powershell
conda run -n recover_attention python scripts/08_run_recovery.py `
  --input data/processed/masked_questions.jsonl `
  --output outputs/logs/recover_outputs_ollama_small.jsonl `
  --backend ollama_chat_v0 `
  --model qwen3.5:9b `
  --ollama-base-url http://localhost:11434 `
  --num-samples 1 `
  --temperature 0.0 `
  --top-p 1.0 `
  --max-tokens 128 `
  --timeout 120 `
  --seed 42 `
  --limit 10
```

如果 `qwen3.5:9b` 不存在，但 `llama3.1:8b` 存在，运行：

```powershell
conda run -n recover_attention python scripts/08_run_recovery.py `
  --input data/processed/masked_questions.jsonl `
  --output outputs/logs/recover_outputs_ollama_small.jsonl `
  --backend ollama_chat_v0 `
  --model llama3.1:8b `
  --ollama-base-url http://localhost:11434 `
  --num-samples 1 `
  --temperature 0.0 `
  --top-p 1.0 `
  --max-tokens 128 `
  --timeout 120 `
  --seed 42 `
  --limit 10
```

如果 Ollama 没启动或两个模型都不存在，不要静默跳过，必须在最终回复中说明：

```text
Ollama real smoke 未运行成功，原因是：
...
```

不要自动执行：

```text
ollama pull qwen3.5:9b
ollama pull llama3.1:8b
```

---

## 22. Stub smoke 要求

必须运行：

```powershell
conda run -n recover_attention python scripts/08_run_recovery.py `
  --input data/processed/masked_questions.jsonl `
  --output outputs/logs/recover_outputs_stub_check.jsonl `
  --backend oracle_stub_v0 `
  --num-samples 1 `
  --limit 10
```

如果 `oracle_stub_v0` 不支持 `--limit`，本 sprint 需要在 CLI 中实现 `--limit`。

---

## 23. 必须运行命令

所有命令必须使用：

```text
conda run -n recover_attention python ...
```

不要使用裸 `python`。

至少运行：

```powershell
conda run -n recover_attention python scripts/sync_interface_fields.py --check

conda run -n recover_attention python -m pytest tests/test_recover_generation.py -q

conda run -n recover_attention python -m pytest tests/test_schemas.py -q

conda run -n recover_attention python -m pytest -q
```

必须运行 stub CLI smoke：

```powershell
conda run -n recover_attention python scripts/08_run_recovery.py `
  --input data/processed/masked_questions.jsonl `
  --output outputs/logs/recover_outputs_stub_check.jsonl `
  --backend oracle_stub_v0 `
  --num-samples 1 `
  --limit 10
```

必须尝试运行 Ollama real smoke。优先使用：

```text
qwen3.5:9b
```

fallback：

```text
llama3.1:8b
```

---

## 24. configs 更新要求

更新：

```text
configs/v0_nli_small.yaml
```

允许追加：

```yaml
recovery:
  backend: "oracle_stub_v0"
  real_backends:
    ollama:
      backend: "ollama_chat_v0"
      primary_model: "qwen3.5:9b"
      fallback_model: "llama3.1:8b"
      ollama_base_url: "http://localhost:11434"
      temperature: 0.0
      top_p: 1.0
      max_tokens: 128
      timeout: 120
      seed: 42
  num_samples: 1
```

不要让 smoke test 自动使用 real backend。

---

## 25. requirements 更新要求

本 sprint 优先使用 Python 标准库调用 Ollama，不强制新增依赖。

如果添加注释，只允许：

```text
# Optional for local Ollama recovery backend:
# ollama server / Ollama app must be installed separately
```

不要把 `ollama` Python package 作为必需依赖，除非用户明确确认。

---

## 26. PROGRESS.md 更新要求

更新：

```text
PROGRESS.md
```

要求：

```text
1. 当前阶段更新为 Sprint 1M 已完成：Plug Real LLM Recovery Backend。
2. 已完成 Sprint 摘要中新增 Sprint 1M。
3. 当前可运行命令中新增 ollama_chat_v0 示例命令。
4. 最近一次检查结果中记录：
   real LLM recovery backend integration: passed
   oracle recovery stub regression: passed
   prompt leakage guard: passed
   qwen3.5:9b real smoke: passed
   如果 fallback 到 llama3.1:8b，则记录 fallback 原因
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中说明：
   src/recover_attention/recover_generation.py 已支持 oracle_stub_v0 / ollama_chat_v0。
6. 遗留问题中说明：
   - ollama_chat_v0 是真实 recovery generation backend，但 recover_scoring 仍是 stub_rule_v0 exact match。
   - 当前没有修改 semantic recoverability scoring。
   - 当前没有用真实 recovery 全量重建 recover_scores / unit_evidence / attention_anchor_labels / intervention_manifest。
   - 当前没有接入 hidden states / attention maps / trajectory stability / attention guidance。
   - 如果 Ollama real smoke 未成功，说明具体原因。
7. 下一步建议：
   Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs
```

---

## 27. docs/progress/sprint_1_history.md 更新要求

更新：

```text
docs/progress/sprint_1_history.md
```

追加：

```text
## Sprint 1M：Plug Real LLM Recovery Backend
```

内容包括：

```text
1. 已完成内容。
2. 新增/修改文件。
3. 新增 backend：
   ollama_chat_v0
4. Prompt 防泄漏原则。
5. Ollama model：
   primary = qwen3.5:9b
   fallback = llama3.1:8b
6. Ollama base_url / decoding config。
7. 是否运行真实 Ollama smoke。
8. 如果 fallback，说明原因。
9. 运行命令。
10. 检查结果。
11. 遗留问题。
12. 下一步建议：Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs。
```

---

## 28. 验收标准

本 sprint 完成后必须满足：

```text
1. oracle_stub_v0 行为保持不变。
2. schemas.py 中 ALLOWED_RECOVERY_BACKENDS 包含 ollama_chat_v0。
3. recover_outputs 顶层字段未变化。
4. validate_recover_output_record 仍通过。
5. recover_outputs_interface.md 已更新 backend 说明。
6. scripts/08_run_recovery.py 支持 --model / --ollama-base-url / --temperature / --top-p / --max-tokens / --timeout / --seed / --limit。
7. 默认 model 是 qwen3.5:9b。
8. fallback model 是 llama3.1:8b。
9. Preflight 已检查 ollama list。
10. 不自动运行 ollama pull。
11. build_recovery_prompt 不包含 original_question。
12. build_recovery_prompt 不包含 spans[*].text。
13. ollama_chat_v0 可通过 monkeypatch fake call 生成 valid recover_output record。
14. pytest 不调用真实 Ollama。
15. stub CLI smoke 通过。
16. Ollama real smoke 已尝试运行；成功则生成 outputs/logs/recover_outputs_ollama_small.jsonl，失败则明确说明原因。
17. tests/test_recover_generation.py 通过。
18. tests/test_schemas.py 通过。
19. 全量 pytest 通过。
20. 没有修改 recover_scoring.py。
21. 没有修改 unit_evidence.py。
22. 没有修改 attention_anchor_labels.py。
23. 没有接入 attention guidance。
24. 没有做 hidden states / attention maps / trajectory stability。
25. 没有做目录重构。
26. PROGRESS.md 已更新。
27. docs/progress/sprint_1_history.md 已更新。
```

---

## 29. 完成后回复格式

完成后请按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 新增 backend
4. Prompt 防泄漏设计
5. Ollama 模型选择：
   primary = qwen3.5:9b
   fallback = llama3.1:8b
6. Ollama 配置
7. 运行命令
8. 检查结果
9. 是否运行真实 Ollama smoke；若 fallback 或失败，说明原因
10. PROGRESS.md 更新摘要
11. docs/progress/sprint_1_history.md 更新摘要
12. 遗留问题
13. 下一步建议
```

下一步建议必须是：

```text
Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs
```

不要自动开始 Sprint 1N。
