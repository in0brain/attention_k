# Sprint 1D：NLI Semantic Consistency Scoring Stub

## 1. 目标

本 sprint 实现 NLI semantic consistency scoring 的最小可运行版本。

目标文件流：

```text
data/processed/ablated_questions.jsonl
→ data/processed/nli_scores.jsonl
```

本 sprint 只负责对 1C 生成的 ablated question records 做双向 NLI stub scoring。

本 sprint 不做 semantic necessity label，不做 masked question construction，不做 question recovery，不做 recoverability scoring，不做 attention guidance，不调用真实 NLI 模型。

---

## 2. 当前阶段定位

前置 pipeline：

```text
questions.jsonl
→ candidate_spans.jsonl
→ ablation_units.jsonl
→ ablated_questions.jsonl
```

本 sprint 接在 1C 后面：

```text
ablated_questions.jsonl
→ nli_scores.jsonl
```

下一步才是：

```text
nli_scores.jsonl
→ semantic necessity labels
```

因此，本 sprint 只输出 NLI 分数，不输出最终重要性标签。

---

## 3. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 1C 已完成。
data/processed/ablated_questions.jsonl 已存在。
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md 已存在。
```

如果输入文件不存在，停止并报告：

```text
Missing input: data/processed/ablated_questions.jsonl
Please run Sprint 1C first.
```

不要自动回头执行 Sprint 1A / 1B / 1C。

---

## 4. 环境要求

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

则停止并报告环境不一致，不要继续修改文件，不要运行测试。

---

## 5. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
data/processed/ablated_questions.jsonl
```

如果以下文件存在，也读取：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
tests/test_nli_scoring.py
tests/test_schemas.py
```

不要读取：

```text
docs/reference/*
```

除非当前用户指令明确要求。

---

## 6. 允许修改

本 sprint 允许修改：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/SKILL.md
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
tests/test_nli_scoring.py
src/recover_attention/schemas.py
tests/test_schemas.py
data/processed/nli_scores.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
tests/test_nli_scoring.py
tests/test_schemas.py
docs/progress/sprint_1_history.md
```

限制：

```text
docs/reasoning-aware-attention-guidance/SKILL.md 只允许增加 nli_scores_interface.md 的文档索引行。
```

---

## 7. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
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
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 8. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

除 AGENTS.md 的全局 Preflight 要求外，本 sprint 必须额外报告：

```text
1. 是否确认当前进入 Sprint 1D：NLI Semantic Consistency Scoring Stub
2. 当前 Python 环境信息
3. data/processed/ablated_questions.jsonl 是否存在
4. ablated question record 数量
5. delete / generalize ablation_type 数量
6. single / group unit_scope 数量
7. 当前 docs/reasoning-aware-attention-guidance/ablated_questions_interface.md 是否存在
8. 当前 docs/reasoning-aware-attention-guidance/nli_scores_interface.md 是否存在
9. 当前 schemas.py 是否已有 validate_nli_score_record
10. 本次是否需要新增 validate_nli_score_record
11. 本次计划使用的 backend
12. 本次计划使用的 language 设置
13. 是否明确不调用真实 NLI 模型
14. 是否明确不下载模型、不引入 transformers、不调用 API
15. 是否明确不生成 semantic necessity label
16. 是否明确不做 mask / recovery / attention / probe
```

如发现冲突，按 AGENTS.md 中定义的冲突优先级处理，并在 Preflight 中报告。

---

## 9. 输入接口契约

输入文件：

```text
data/processed/ablated_questions.jsonl
```

输入必须符合：

```text
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
```

每条输入 record 至少包含：

```text
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
```

本 sprint 不推断旧字段名。

如果输入 record 缺少上述字段，停止并报告 schema mismatch。

不要自动修改 `ablated_questions.jsonl`。

---

## 10. 输出接口契约

输出文件：

```text
data/processed/nli_scores.jsonl
```

输出必须符合：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
```

每条 output record 对应一条 input ablated question record。

也就是：

```text
一条 ablation_id
→ 一条 nli score record
```

不要把 forward / backward 拆成两条 jsonl。

---

## 11. NLI 方向定义

本 sprint 使用双向 NLI。

### 11.1 Forward NLI

```text
premise = original_question
hypothesis = ablated_question
```

含义：

```text
原问题是否蕴含扰动后问题。
```

### 11.2 Backward NLI

```text
premise = ablated_question
hypothesis = original_question
```

含义：

```text
扰动后问题是否蕴含原问题。
```

### 11.3 双向一致性近似

语义等价需要近似满足：

```text
original_question entails ablated_question
and
ablated_question entails original_question
```

因此本 sprint 输出：

```text
bidirectional_entailment_score
```

定义为：

```python
min(forward["scores"]["entailment"], backward["scores"]["entailment"])
```

同时输出：

```text
contradiction_score
```

定义为：

```python
max(forward["scores"]["contradiction"], backward["scores"]["contradiction"])
```

注意：

```text
bidirectional_entailment_score 和 contradiction_score 只是分数，不是最终标签。
```

---

## 12. Backend 设计

本 sprint 只支持：

```text
stub_v0
```

不支持真实模型。

CLI 参数必须包含：

```text
--backend stub_v0
```

如果传入其他 backend，例如：

```text
hf_xnli
llm_prompt_v0
local_nli_model
```

必须抛出 ValueError 或退出并报告：

```text
Unsupported backend: <backend>
```

不要偷偷调用真实模型。

不要下载 HuggingFace 模型。

不要新增 transformers / torch 依赖。

不要调用任何外部 API。

---

## 13. Language 设计

本 sprint 支持：

```text
--language auto
--language en
--language zh
```

默认：

```text
--language auto
```

### 13.1 输入文本不翻译

本 sprint 不翻译 `original_question` 或 `ablated_question`。

无论中文还是英文，都直接对原文本做 stub scoring。

### 13.2 输出 label 固定英文枚举

无论 language 是 `auto`、`en` 还是 `zh`，NLI label 必须固定为：

```text
entailment
neutral
contradiction
```

scores key 也必须固定为：

```text
entailment
neutral
contradiction
```

不要输出中文标签：

```text
蕴含
中立
矛盾
```

### 13.3 language = auto

如果 `language = "auto"`：

```text
如果 original_question 或 ablated_question 中包含中文字符，则 language = zh。
否则 language = en。
```

中文字符判断使用 Unicode 范围：

```text
\u4e00-\u9fff
```

### 13.4 输出记录 language

每条 `nli_scores.jsonl` record 必须包含：

```text
language
language_setting
```

含义：

```text
language:
  当前 record 最终解析出的语言，en 或 zh。

language_setting:
  CLI 传入的语言设置，auto / en / zh。
```

当前 `stub_v0` 的分数逻辑不随语言改变。

language 参数只是为了：

```text
1. 记录输入语言。
2. 为未来真实 NLI backend 预留接口。
3. 统计 language_counts。
```

---

## 14. Stub Scoring 规则

stub_v0 必须 deterministic。

不要使用随机数。

不要依赖模型。

stub_v0 分数不追求真实准确，只要求方向合理、可测试、可复现。

### 14.1 Generalize

对于：

```text
ablation_type = generalize
```

基本直觉：

```text
original → ablated 通常更容易成立。
ablated → original 通常更难成立。
```

推荐 base scores：

```python
forward = {
    "entailment": 0.75,
    "neutral": 0.20,
    "contradiction": 0.05,
}

backward = {
    "entailment": 0.35,
    "neutral": 0.60,
    "contradiction": 0.05,
}
```

### 14.2 Delete

对于：

```text
ablation_type = delete
```

基本直觉：

```text
delete 比 generalize 更强，语义损失更明显。
```

推荐 base scores：

```python
forward = {
    "entailment": 0.55,
    "neutral": 0.40,
    "contradiction": 0.05,
}

backward = {
    "entailment": 0.25,
    "neutral": 0.70,
    "contradiction": 0.05,
}
```

### 14.3 Group unit penalty

如果 unit 包含多个 spans，即：

```python
num_spans = len(span_ids)
```

且 `num_spans > 1`，则对 entailment 做轻量 penalty。

推荐规则：

```python
penalty = min(0.20, 0.05 * (num_spans - 1))
```

对 forward / backward entailment 均减去 penalty。

减去的分数加到 neutral 上。

contradiction 保持不变。

处理后保证：

```text
entailment / neutral / contradiction 均在 [0, 1]
scores 总和为 1
```

### 14.4 Label 选择

label 由 scores 最大值决定：

```python
label = argmax(scores)
```

如果分数相同，优先级为：

```text
entailment > neutral > contradiction
```

---

## 15. 输出 Record Schema

每条 `nli_scores.jsonl` record 必须包含：

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

字段来源：

```text
来自 1C / ablated_questions.jsonl：
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

由 1D 新增：
- nli_id
- nli_backend
- language
- language_setting
- forward
- backward
- bidirectional_entailment_score
- contradiction_score
```

`nli_id` 生成规则：

```python
nli_id = f"{ablation_id}__nli_{backend}"
```

当前：

```text
backend = stub_v0
```

所以示例：

```text
gsm8k_0001__unit_001__generalize__nli_stub_v0
```

---

## 16. forward / backward Schema

`forward` 和 `backward` 必须包含：

```text
premise
hypothesis
label
scores
```

其中：

```text
premise: str
hypothesis: str
label: entailment / neutral / contradiction
scores: dict
```

`scores` 必须包含：

```text
entailment
neutral
contradiction
```

每个 score 必须是 float，范围在 `[0, 1]`。

scores 总和应接近 1。

允许浮点误差：

```python
abs(sum(scores.values()) - 1.0) < 1e-6
```

---

## 17. 输出示例

```json
{
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
  "contradiction_score": 0.05
}
```

---

## 18. 接口文档要求

新增或更新：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
```

该文件必须说明：

```text
1. nli_scores.jsonl 的位置。
2. nli_scores.jsonl 的作用。
3. 上下游关系。
4. 顶层 record schema。
5. forward / backward 的含义。
6. bidirectional_entailment_score 的计算方式。
7. contradiction_score 的计算方式。
8. language / language_setting 的含义。
9. stub_v0 backend 的含义。
10. 1E 如何消费 nli_scores.jsonl。
```

同时在：

```text
docs/reasoning-aware-attention-guidance/SKILL.md
```

的文档路由或接口文档区域增加一行：

```text
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
  NLI score 的长期接口文档，说明 nli_scores.jsonl schema、双向 NLI 字段、stub backend、language 参数、字段来源和后续 label rule 消费方式。
```

不要把完整 schema 复制进 SKILL.md。

---

## 19. Schema Validator 要求

在 `src/recover_attention/schemas.py` 中新增：

```python
validate_nli_score_record(record: dict) -> None
```

该函数必须检查：

```text
1. 顶层字段完整。
2. nli_id 是非空 string。
3. ablation_id 是非空 string。
4. id 是非空 string。
5. unit_id 是非空 string。
6. unit_scope 是 single 或 group。
7. group_type 是非空 string。
8. span_ids 是非空 list[str]。
9. spans 是非空 list[dict]。
10. span_ids 长度等于 spans 长度。
11. spans 中 span_id 顺序与 span_ids 一致。
12. ablation_type 是 delete 或 generalize。
13. original_question 是非空 string。
14. ablated_question 是非空 string。
15. nli_backend 是 stub_v0。
16. language 是 en 或 zh。
17. language_setting 是 auto / en / zh。
18. forward / backward 存在。
19. forward / backward 均包含 premise / hypothesis / label / scores。
20. label 是 entailment / neutral / contradiction。
21. scores 包含 entailment / neutral / contradiction。
22. 每个 score 是 int 或 float。
23. 每个 score 在 [0, 1]。
24. scores 总和接近 1。
25. bidirectional_entailment_score 在 [0, 1]。
26. contradiction_score 在 [0, 1]。
27. bidirectional_entailment_score 等于 min(forward_entailment, backward_entailment)。
28. contradiction_score 等于 max(forward_contradiction, backward_contradiction)。
```

如果检查失败，应抛出 `ValueError` 或 `AssertionError`，不要静默修复。

---

## 20. 核心模块要求

新增或实现：

```text
src/recover_attention/nli_scoring.py
```

推荐函数：

```python
detect_language(text: str) -> str
```

返回：

```text
en / zh
```

规则：

```text
包含中文字符 → zh
否则 → en
```

---

```python
resolve_record_language(record: dict, language: str = "auto") -> str
```

规则：

```text
language == "en" → en
language == "zh" → zh
language == "auto" → 检测 original_question + ablated_question
其他值 → ValueError
```

---

```python
score_nli_pair_stub(
    premise: str,
    hypothesis: str,
    ablation_type: str,
    num_spans: int,
    direction: str,
) -> dict
```

参数：

```text
premise:
  NLI premise 文本。

hypothesis:
  NLI hypothesis 文本。

ablation_type:
  delete 或 generalize。

num_spans:
  当前 unit 包含的 span 数量。

direction:
  forward 或 backward。
```

返回：

```python
{
    "premise": "...",
    "hypothesis": "...",
    "label": "entailment",
    "scores": {
        "entailment": 0.75,
        "neutral": 0.20,
        "contradiction": 0.05,
    },
}
```

如果 `ablation_type` 或 `direction` 不支持，应抛出 ValueError。

---

```python
score_ablated_question_record(
    record: dict,
    backend: str = "stub_v0",
    language: str = "auto",
) -> dict
```

要求：

```text
1. 先校验输入 ablated question record。
2. backend 只允许 stub_v0。
3. 解析 language。
4. 构造 forward NLI。
5. 构造 backward NLI。
6. 计算 bidirectional_entailment_score。
7. 计算 contradiction_score。
8. 生成 nli score record。
9. 调用 validate_nli_score_record。
```

---

```python
score_ablated_question_records(
    records: list[dict],
    backend: str = "stub_v0",
    language: str = "auto",
) -> tuple[list[dict], dict]
```

返回：

```text
scored_records:
  nli score records。

stats:
  统计信息。
```

stats 至少包含：

```python
{
    "num_input_ablations": 0,
    "num_output_scores": 0,
    "backend": "stub_v0",
    "language_setting": "auto",
    "language_counts": {},
    "ablation_type_counts": {},
    "unit_scope_counts": {},
    "group_type_counts": {},
}
```

---

## 21. CLI 脚本要求

新增或实现：

```text
scripts/05_run_nli_scoring.py
```

命令格式：

```bash
python scripts/05_run_nli_scoring.py \
  --input data/processed/ablated_questions.jsonl \
  --output data/processed/nli_scores.jsonl \
  --backend stub_v0 \
  --language auto
```

支持参数：

```text
--input
--output
--backend
--language
```

默认值：

```text
--backend stub_v0
--language auto
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 score_ablated_question_records。
4. 对每条输出调用 validate_nli_score_record。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印 nli scoring 统计。
7. 不调用真实模型。
8. 不下载模型。
9. 不调用外部 API。
```

如果 backend 不是 `stub_v0`，CLI 必须失败并给出清楚错误。

---

## 22. 统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_input_ablations
num_output_scores
backend
language_setting
language_counts
ablation_type_counts
unit_scope_counts
group_type_counts
```

这些统计只打印到 stdout，不要求写入 json 文件。

---

## 23. 测试要求

新增：

```text
tests/test_nli_scoring.py
```

至少覆盖以下测试。

### 23.1 generalize forward/backward scores

构造一条 `generalize` ablated question record。

验证：

```text
forward label = entailment
backward label = neutral
forward entailment > backward entailment
```

### 23.2 delete forward/backward scores

构造一条 `delete` ablated question record。

验证：

```text
forward entailment > backward entailment
backward label = neutral
```

### 23.3 group penalty

构造两个 records：

```text
single span
group spans
```

同样 ablation_type 下，验证 group 的 entailment score 更低。

### 23.4 bidirectional_entailment_score

验证：

```python
bidirectional_entailment_score == min(
    forward["scores"]["entailment"],
    backward["scores"]["entailment"],
)
```

### 23.5 contradiction_score

验证：

```python
contradiction_score == max(
    forward["scores"]["contradiction"],
    backward["scores"]["contradiction"],
)
```

### 23.6 language auto English

输入英文问题。

验证：

```text
language = en
language_setting = auto
```

### 23.7 language auto Chinese

输入中文问题。

验证：

```text
language = zh
language_setting = auto
```

### 23.8 language forced zh

输入英文问题，但设置：

```text
language = zh
```

验证输出：

```text
language = zh
language_setting = zh
```

### 23.9 label fixed English enum

中文输入下，验证 label 仍然是：

```text
entailment / neutral / contradiction
```

不要出现：

```text
蕴含 / 中立 / 矛盾
```

### 23.10 unsupported backend raises

传入：

```text
hf_xnli
```

应抛出 ValueError 或 CLI 返回失败。

### 23.11 unsupported language raises

传入：

```text
jp
```

应抛出 ValueError 或 CLI 返回失败。

### 23.12 score sum equals 1

验证 forward / backward scores 总和接近 1。

### 23.13 schema validation

验证每条输出 record 都能通过：

```python
validate_nli_score_record(record)
```

### 23.14 CLI smoke test

使用临时输入 jsonl 运行 CLI。

验证：

```text
输出文件存在
每行是合法 JSON object
至少包含 nli_backend / forward / backward / bidirectional_entailment_score / contradiction_score
```

如修改了 `schemas.py`，也补充或更新：

```text
tests/test_schemas.py
```

---

## 24. 本 sprint 不做

不要实现：

```text
真实 NLI 模型
HuggingFace transformers pipeline
XNLI model loading
LLM prompt NLI
API 调用
模型下载
semantic necessity label
recoverability label
masked question construction
question recovery
recoverability scoring
label building
incomplete-question reasoning
answer stability
baseline CoT
hidden states
attention maps
trajectory stability
attention anchor labeling
attention guidance
probe training
```

不要新增：

```text
src/recover_attention/label_builder.py
scripts/06_build_semantic_labels.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/baseline_cot.py
```

不要生成：

```text
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/labels.jsonl
data/processed/token_labels.jsonl
```

---

## 25. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores.jsonl --backend stub_v0 --language auto
python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

注意：

```text
本 sprint 不自动运行 Sprint 1C。
如果 data/processed/ablated_questions.jsonl 不存在，应停止并报告缺失输入。
```

---

## 26. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1D 已完成。
下一步建议是 Sprint 1E：Semantic Necessity Label Rule。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1D | 完成 | NLI semantic consistency scoring stub |
```

当前可运行命令中增加：

```bash
python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores.jsonl --backend stub_v0 --language auto
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
nli scoring stub: passed
```

当前关键文件状态中补充：

```text
已完成：
- docs/reasoning-aware-attention-guidance/nli_scores_interface.md
- src/recover_attention/nli_scoring.py
- scripts/05_run_nli_scoring.py
- tests/test_nli_scoring.py
- data/processed/nli_scores.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/semantic_labels.py
- scripts/06_build_semantic_labels.py
- tests/test_semantic_labels.py
- data/processed/semantic_labels.jsonl
```

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 27. 验收标准

本 sprint 完成后应满足：

```text
1. docs/reasoning-aware-attention-guidance/nli_scores_interface.md 存在。
2. docs/reasoning-aware-attention-guidance/SKILL.md 已增加 nli_scores_interface.md 索引行。
3. src/recover_attention/nli_scoring.py 存在。
4. scripts/05_run_nli_scoring.py 存在。
5. tests/test_nli_scoring.py 存在。
6. schemas.py 中存在 validate_nli_score_record。
7. CLI 支持 --backend stub_v0。
8. CLI 支持 --language auto/en/zh。
9. 不支持真实 NLI backend。
10. unsupported backend 会失败。
11. unsupported language 会失败。
12. nli_scores.jsonl 已生成。
13. 每条 nli score record 保留 1C 元数据。
14. 每条 nli score record 包含 forward / backward。
15. 每条 nli score record 包含 language / language_setting。
16. label 固定使用 entailment / neutral / contradiction。
17. bidirectional_entailment_score 计算正确。
18. contradiction_score 计算正确。
19. validate_nli_score_record 能校验输出 record。
20. python -m pytest -q 通过。
21. smoke test 通过。
22. 未调用真实 NLI 模型。
23. 未下载模型。
24. 未新增 transformers / torch 依赖。
25. 未生成 masked_questions.jsonl。
26. 未生成 recover_outputs.jsonl。
27. 未生成 semantic necessity label。
28. 未实现 mask / recovery / trajectory / attention guidance / probe。
29. PROGRESS.md 保持短版状态索引。
```

---

## 28. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. nli score 数量统计
6. language 统计
7. PROGRESS.md 更新摘要
8. 是否确认未调用真实 NLI
9. 是否确认未生成 label / mask / recovery 产物
10. 遗留问题
11. 下一步建议
```

下一步建议只写：

```text
可以继续执行 Sprint 1E：Semantic Necessity Label Rule。
```

不要自动开始 Sprint 1E。
