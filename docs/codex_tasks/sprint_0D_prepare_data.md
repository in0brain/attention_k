# Sprint 0D：prepare_data 数据准备脚本

## 1. 目标

本 sprint 只完成基础数据准备脚本。

目标是：

```text id="uem4vo"
把原始 question jsonl 文件规范化为 data/processed/questions.jsonl
复用 data_io.py 的 jsonl 读写函数
复用 schemas.py 的 question record 校验函数
提供 CLI 脚本
提供 pytest 测试
```

本 sprint 仍然属于工程地基阶段，不实现任何实验方法本身。

---

## 2. 本次允许修改的文件

只允许修改或创建：

```text id="yp17fp"
src/recover_attention/prepare_data.py
scripts/01_prepare_data.py
tests/test_prepare_data.py
PROGRESS.md
```

运行脚本时允许生成：

```text id="vbsxk8"
data/processed/questions.jsonl
```

但不要手动编辑 `data/processed/questions.jsonl`。

---

## 3. 本次禁止修改的文件

不要修改：

```text id="o5uiff"
README.md
AGENTS.md
docs/skill/SKILL.md
docs/prompts.md
docs/reference/*
docs/codex_tasks/*
requirements.txt
configs/*.yaml
data/examples/*
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/candidate_extraction.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
src/recover_attention/utils.py
scripts/00_smoke_test.py
tests/test_data_io.py
tests/test_schemas.py
```

如果发现 `data_io.py` 或 `schemas.py` 有 bug，不要直接修改，先停止并报告。

---

## 4. 输入文件

默认输入：

```text id="r4w5su"
data/examples/questions_small.jsonl
```

每条记录格式：

```json id="5w0a8f"
{"id":"gsm8k_0001","dataset":"gsm8k","question":"Tom has 3 apples and buys 2 more. How many apples does he have now?","gold_answer":"5"}
```

---

## 5. 输出文件

默认输出：

```text id="cml1rl"
data/processed/questions.jsonl
```

输出格式必须保持为标准 question record：

```json id="v1zhwh"
{
  "id": "gsm8k_0001",
  "dataset": "gsm8k",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "gold_answer": "5"
}
```

---

## 6. 实现要求

### 6.1 实现 `src/recover_attention/prepare_data.py`

实现函数：

```python id="vtt06e"
normalize_question_record(record: dict) -> dict
prepare_questions(input_path, output_path) -> list[dict]
```

要求：

```text id="ge94st"
1. 使用 read_jsonl 读取输入文件。
2. 使用 write_jsonl 写入输出文件。
3. 使用 validate_question_record 校验输入和输出记录。
4. 输出记录只保留以下四个字段：
   - id
   - dataset
   - question
   - gold_answer
5. 对四个字符串字段执行 strip()。
6. 不要把非字符串字段强行转换成字符串。
7. 如果字段缺失、类型错误或空字符串，抛出 ValueError。
8. 如果出现重复 id，抛出 ValueError。
9. 保持输入记录顺序。
10. 不要 shuffle。
11. 不要划分 train/test。
12. 不要做任何 span 抽取。
13. 不要调用模型。
```

`normalize_question_record(record)` 负责：

```text id="lrwk7v"
1. 检查 record 是 dict。
2. 提取 id / dataset / question / gold_answer。
3. 对字符串做 strip。
4. 调用 validate_question_record。
5. 返回规范化后的 dict。
```

`prepare_questions(input_path, output_path)` 负责：

```text id="n66fe3"
1. 读取 input_path。
2. 逐条 normalize。
3. 检查重复 id。
4. 写入 output_path。
5. 返回规范化后的 records。
```

---

### 6.2 实现 `scripts/01_prepare_data.py`

支持命令：

```bash id="2nxuj1"
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
```

要求：

```text id="lxyfar"
1. 使用 argparse。
2. 参数包括 --input 和 --output。
3. 调用 prepare_questions。
4. 成功时输出处理条数和输出路径。
5. 失败时不要吞掉异常。
6. 不要在脚本中写死输入输出路径。
```

成功输出建议：

```text id="jyo3d0"
[OK] Prepared 5 question records: data/processed/questions.jsonl
```

---

## 7. 测试要求

创建 `tests/test_prepare_data.py`。

至少测试：

```text id="kz4t98"
1. normalize_question_record 能返回只包含四个标准字段的记录。
2. normalize_question_record 会 strip 字符串字段。
3. normalize_question_record 遇到缺失字段会抛出 ValueError。
4. normalize_question_record 遇到空 question 会抛出 ValueError。
5. prepare_questions 能读取 input jsonl 并写入 output jsonl。
6. prepare_questions 输出顺序与输入顺序一致。
7. prepare_questions 遇到重复 id 会抛出 ValueError。
8. prepare_questions 不保留额外字段。
```

测试中使用临时目录，不要依赖真实 `data/examples/questions_small.jsonl`。

---

## 8. 必须运行的命令

完成后运行：

```bash id="sa8c3d"
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
```

然后运行：

```bash id="ibziq1"
python -m pytest -q
```

如果存在 smoke test，也运行：

```bash id="f8t7dc"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

---

## 9. 验收标准

必须满足：

```text id="oqs3mk"
[OK] src/recover_attention/prepare_data.py 存在
[OK] scripts/01_prepare_data.py 存在
[OK] tests/test_prepare_data.py 存在
[OK] prepare_data 脚本可以生成 data/processed/questions.jsonl
[OK] data/processed/questions.jsonl 为标准 question record jsonl
[OK] python -m pytest -q 可以通过
[OK] smoke test 可以通过
[OK] 没有修改 README.md / AGENTS.md / docs/skill/SKILL.md / docs/prompts.md / docs/reference/*
[OK] 没有修改 data_io.py 和 schemas.py
[OK] 没有实现 Sprint 1A
[OK] PROGRESS.md 已更新
```

---

## 10. 禁止事项

本次不要做：

```text id="fs36lk"
不要实现 candidate span extraction。
不要实现 ablated question construction。
不要实现 NLI scoring。
不要实现 masked question construction。
不要实现 question recovery。
不要实现 recoverability scoring。
不要实现 label building。
不要实现 evaluation metrics。
不要调用任何模型。
不要下载任何模型。
不要调用外部 API。
不要训练 probe。
不要做 attention guidance。
不要做 hidden states。
不要做 trajectory analysis。
不要做大规模实验。
不要重构项目。
```

---

## 11. PROGRESS.md 更新要求

完成后更新 `PROGRESS.md`，新增或更新一节：

```md id="sw2rf0"
### Sprint 0D：prepare_data 数据准备脚本

已完成内容：

- 实现 question record 规范化函数。
- 实现 prepare_questions 数据准备函数。
- 实现 CLI 脚本 scripts/01_prepare_data.py。
- 实现 prepare_data pytest 测试。
- 生成 data/processed/questions.jsonl。

新增或修改文件：

- src/recover_attention/prepare_data.py
- scripts/01_prepare_data.py
- tests/test_prepare_data.py
- data/processed/questions.jsonl
- PROGRESS.md

运行命令：

- python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
- python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
- python -m pytest -q

检查结果：

- xxx

遗留问题：

- xxx

下一步建议：

- Sprint 0E：基础验收与工程收尾
```

不要自动开始 Sprint 0E。
