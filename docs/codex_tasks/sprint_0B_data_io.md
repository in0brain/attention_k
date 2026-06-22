# Sprint 0B：jsonl 数据读写 + 样例数据 + smoke test

## 1. 目标

本 sprint 只完成最小数据读写框架。

目标是：

```text
建立统一 jsonl 读写工具
准备最小样例数据
创建最小配置文件
实现 smoke test
```

本 sprint 不实现 schema 校验，不实现 candidate span extraction。

---

## 2. 本次允许修改的文件

```text
src/recover_attention/data_io.py
configs/v0_nli_small.yaml
data/examples/questions_small.jsonl
scripts/00_smoke_test.py
tests/test_data_io.py
PROGRESS.md
```

如果 `scripts/` 或 `tests/` 目录不存在，可以创建。

---

## 3. 本次禁止修改的文件

```text
README.md
AGENTS.md
docs/skill/SKILL.md
docs/reference/*
src/recover_attention/candidate_extraction.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
```

---

## 4. 输入文件

无强制输入。

如果 `data/examples/questions_small.jsonl` 不存在，本 sprint 需要创建。

---

## 5. 输出文件

```text
configs/v0_nli_small.yaml
data/examples/questions_small.jsonl
scripts/00_smoke_test.py
tests/test_data_io.py
```

---

## 6. 实现要求

### 6.1 data_io.py

实现：

```python
read_jsonl(path) -> list[dict]
write_jsonl(records, path) -> None
ensure_dir(path) -> None
```

要求：

1. 使用 UTF-8。
2. 写入 jsonl 时自动创建父目录。
3. 每条记录一行 JSON。
4. 空文件返回空列表。
5. 非法 JSON 行要报错，并指出文件路径和行号。
6. 不要吞掉异常。
7. 不要引入 pandas。

### 6.2 configs/v0_nli_small.yaml

创建：

```yaml
project_name: recover_attention_project
stage: v0_nli_small
seed: 42

paths:
  raw_data: data/examples/questions_small.jsonl
  questions: data/processed/questions.jsonl
  candidate_spans: data/processed/candidate_spans.jsonl
  ablated_questions: data/processed/ablated_questions.jsonl
  masked_questions: data/processed/masked_questions.jsonl
  nli_scores: data/processed/nli_scores.jsonl
  log_dir: outputs/logs

candidate_extraction:
  enabled_types:
    - number
    - entity
    - operation
    - relation
    - question_target

ablation:
  enabled_types:
    - delete
    - generalize
    - mask
  mask_token: "[MASK]"

nli:
  backend: "stub"
```

### 6.3 data/examples/questions_small.jsonl

创建 5 条样例数据。

每条格式：

```json
{"id":"gsm8k_0001","dataset":"gsm8k","question":"Tom has 3 apples and buys 2 more. How many apples does he have now?","gold_answer":"5"}
```

样例必须覆盖：

1. 数字 span。
2. 人名或实体 span。
3. 操作词，例如 buys / gives / left / more。
4. 问题目标，例如 How many。

### 6.4 scripts/00_smoke_test.py

实现 smoke test。

命令：

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

检查：

1. 能读取 yaml config。
2. 能 import recover_attention。
3. 能 import read_jsonl / write_jsonl。
4. 能读取 `data/examples/questions_small.jsonl`。
5. 能把样例数据写到临时文件，例如 `outputs/logs/smoke_test_questions.jsonl`。
6. 能再次读回临时文件。
7. 输出：

```text
[OK] Sprint 0B smoke test passed.
```

### 6.5 tests/test_data_io.py

实现最小 pytest：

1. 写入 jsonl 后能读回。
2. 空文件返回空 list。
3. 非法 JSON 行会抛出异常。
4. 自动创建父目录。

---

## 7. 验收标准

```text
[OK] python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml 可以运行
[OK] 输出 [OK] Sprint 0B smoke test passed.
[OK] pytest -q 可以通过
[OK] data/examples/questions_small.jsonl 存在且有 5 条样例
[OK] configs/v0_nli_small.yaml 存在
[OK] PROGRESS.md 已更新
```

---

## 8. 运行命令

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

```bash
python -m pytest -q
```

---

## 9. 禁止事项

```text
不要实现 schema 校验。
不要实现 candidate span extraction。
不要实现 ablated question construction。
不要实现 NLI scoring。
不要调用模型。
不要下载模型。
不要训练 probe。
不要做 attention guidance。
不要做 hidden states。
不要做 trajectory analysis。
不要做大规模实验。
不要修改 README.md。
不要修改 AGENTS.md。
不要修改 docs/skill/SKILL.md。
```

---

## 10. 完成后必须更新

```text
PROGRESS.md
```

更新内容必须包括：

```text
Sprint 0B 已完成内容
新增或修改文件
运行命令
检查结果
遗留问题
下一步建议：Sprint 0C schemas
```

不要自动开始 Sprint 0C。
