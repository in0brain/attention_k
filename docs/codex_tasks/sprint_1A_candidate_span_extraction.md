# Sprint 1A：Candidate Span Extraction Framework

## 1. 目标

本 sprint 实现 candidate span extraction 的最小可运行框架。

目标文件流：

```text
data/processed/questions.jsonl
→ data/processed/candidate_spans.jsonl
```

本 sprint 是 v0 最小闭环的第一步。

本 sprint 的重点不是构造一个最终高质量 extractor，而是建立一个可替换、可扩展、multilingual-safe 的 candidate span extraction framework。

也就是说，本 sprint 要固定：

```text
1. candidate_spans.jsonl 输出格式
2. extractor 函数接口
3. start / end offset 语义
4. rule_based baseline backend
5. language 参数
6. candidate budget 参数
7. 中英文最小样例测试
8. schema validation
9. CLI 入口
```

本 sprint 不做：

```text
ablation
NLI
masked question construction
question recovery
recoverability scoring
baseline CoT
hidden states
trajectory analysis
attention guidance
probe training
真实模型调用
```

---

## 2. 设计原则

Candidate span 不是普通分词结果，也不是所有 token。

本项目中的 candidate span 指：

```text
一个最小语义功能单元。
当它被 delete / generalize / mask 后，问题语义、推理约束或答案空间可能发生变化。
```

因此，本 sprint 优先抽取：

```text
number
object
operation
comparison
negation
condition
question_target
cyber_security_term
```

可以尝试抽取：

```text
entity
relation
```

但不要求第一版完美。

---

## 3. Candidate Budget 设计

本 sprint 不规定每个问题必须抽取固定数量的 span。

不要写成：

```text
每个问题必须抽 8–20 个 span
```

正确设计是：

```text
candidate span extraction uses a configurable candidate budget,
not a fixed number of spans.
```

中文解释：

```text
候选 span 抽取采用可配置候选预算，而不是固定候选数量。
```

### 3.1 默认预算

本 sprint 默认：

```text
max_candidates = 20
```

含义：

```text
每个问题最多保留 20 个 candidate spans。
```

这不是理论最优值，只是早期 v0 实验的默认预算，用于平衡：

```text
candidate recall
和
后续 ablation / NLI / recovery 成本
```

### 3.2 不设置硬性最小值

本 sprint 不设置硬性最小候选数。

如果某条问题只命中 3–7 个合法 span，也允许通过。

原因：

```text
1. 问题复杂度不同。
2. 中英文规则覆盖度不同。
3. 第一版 rule_based extractor 只是 baseline。
4. 强行补足候选数量会引入低价值 span。
```

### 3.3 截断规则

如果候选数量超过 `max_candidates`，按优先级截断。

推荐优先级：

```text
number
question_target
condition
negation
comparison
operation
cyber_security_term
object
entity
relation
```

同一优先级内按 `start` 从小到大排序。

---

## 4. 前置条件

执行本 sprint 前，必须确认：

```text
Sprint 0H 已完成。
Sprint 0G 已完成 schema 与 Attention Anchor 标签体系对齐。
data/processed/questions.jsonl 已由 prepare_data 生成。
src/recover_attention/schemas.py 已存在 validate_candidate_span_record。
```

如果 `PROGRESS.md` 中下一步仍写着：

```text
Sprint 1A：Baseline CoT Schema
```

应在本 sprint 中修正为：

```text
Sprint 1A：Candidate Span Extraction Framework
```

并说明：

```text
Baseline CoT / trajectory 相关模块后移到 reasoning-risk 或 trajectory-aware extension，
不作为当前 v0 最小闭环的起点。
```

---

## 5. 开始前必须读取

开始前必须读取：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/candidate_extraction.py
tests/test_schemas.py
data/processed/questions.jsonl
```

不要读取：

```text
docs/reference/*
```

除非当前文件不足以判断要求，并先向用户说明原因。

---

## 6. 允许修改

本 sprint 允许修改：

```text
src/recover_attention/candidate_extraction.py
scripts/02_extract_candidate_spans.py
tests/test_candidate_extraction.py
data/processed/candidate_spans.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

如果以下文件不存在，可以创建：

```text
scripts/02_extract_candidate_spans.py
tests/test_candidate_extraction.py
docs/progress/sprint_1_history.md
```

---

## 7. 禁止修改

本 sprint 禁止修改：

```text
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/*
docs/reference/*
docs/codex_tasks/*
configs/*
requirements.txt
pyproject.toml
.gitignore
src/recover_attention/schemas.py
src/recover_attention/data_io.py
src/recover_attention/prepare_data.py
src/recover_attention/question_ablations.py
src/recover_attention/nli_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/label_builder.py
src/recover_attention/evaluation.py
任何 baseline CoT / trajectory / attention guidance / probe 相关新文件
```

不要清理 `__pycache__` 或 `.pyc`，除非用户另行要求。

---

## 8. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

Preflight 必须包括：

```text
1. 已读取文件列表
2. 是否确认 Sprint 0H 已完成
3. 是否确认当前进入 Sprint 1A：Candidate Span Extraction Framework
4. 是否发现 PROGRESS.md 中仍残留 Baseline CoT Schema 作为下一步
5. 本次允许修改文件
6. 本次禁止修改文件
7. 当前 data/processed/questions.jsonl 样本数量
8. 当前 schemas.py 中 validate_candidate_span_record 的字段要求
9. 本次计划实现的 extractor 接口
10. 本次计划支持的 language / backend / max_candidates 参数
11. 本次计划支持的 span type
12. 本次必须运行的命令
```

用户确认后才能修改文件。

---

## 9. 核心接口要求

在 `src/recover_attention/candidate_extraction.py` 中实现：

```python
extract_candidate_spans(
    question: str,
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = 20,
) -> list[dict]
```

参数含义：

```text
question:
输入问题文本。

language:
语言设置。当前允许 auto / en / zh。

backend:
抽取后端。当前只实现 rule_based。

max_candidates:
每条问题最多保留多少 candidate spans。
```

早期只允许：

```text
backend = "rule_based"
```

如果传入其他 backend，应抛出 ValueError。

---

## 10. 输出 candidate 格式

每个 candidate 必须包含：

```text
span_id
text
type
start
end
```

示例：

```json
{
  "span_id": "span_001",
  "text": "3",
  "type": "number",
  "start": 8,
  "end": 9
}
```

字段要求：

```text
span_id: str, non-empty
text: str, non-empty
type: str, one of allowed span types
start: int, >= 0
end: int, > start
```

必须保证：

```python
question[start:end] == text
```

注意：

```text
start / end 使用 Python str 字符切片语义，不使用 byte offset。
```

这对中文尤其重要。

---

## 11. 文件级函数要求

在 `src/recover_attention/candidate_extraction.py` 中实现：

```python
build_candidate_span_records(
    question_records: list[dict],
    language: str = "auto",
    backend: str = "rule_based",
    max_candidates: int = 20,
) -> list[dict]
```

输入 record 来自：

```text
data/processed/questions.jsonl
```

输出 record 格式：

```json
{
  "id": "gsm8k_0001",
  "question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "candidates": [
    {
      "span_id": "span_001",
      "text": "3",
      "type": "number",
      "start": 8,
      "end": 9
    }
  ]
}
```

每条输出必须通过：

```python
validate_candidate_span_record(record)
```

---

## 12. CLI 脚本要求

新增或实现：

```text
scripts/02_extract_candidate_spans.py
```

命令格式：

```bash
python scripts/02_extract_candidate_spans.py \
  --input data/processed/questions.jsonl \
  --output data/processed/candidate_spans.jsonl
```

支持参数：

```text
--input
--output
--language auto/en/zh
--backend rule_based
--max-candidates
```

默认值：

```text
--language auto
--backend rule_based
--max-candidates 20
```

要求：

```text
1. 使用 argparse。
2. 使用 data_io.py 读写 jsonl。
3. 使用 build_candidate_span_records。
4. 使用 validate_candidate_span_record 校验输出。
5. 输出文件每行是一个 JSON object。
6. 在 stdout 打印候选数量统计。
```

---

## 13. 候选数量统计要求

CLI 运行后必须打印统计信息。

至少包括：

```text
num_questions
total_candidates
avg_candidates_per_question
min_candidates_per_question
max_candidates_per_question
questions_with_zero_candidates
span_type_counts
max_candidates_setting
language
backend
```

这些统计只打印到 stdout，不要求写入 json 文件。

本 sprint 不要求做 sensitivity analysis，但保留统计是为了后续比较不同 `max_candidates` 设置。

---

## 14. Rule-based Baseline 要求

本 sprint 实现最小 rule-based baseline。

### 14.1 通用 number 规则

支持英文和中文环境中的数字。

至少匹配：

```text
整数
小数
百分比
带逗号数字
中文数字
```

示例：

```text
3
2
2.5
10%
5,000
一
二
三
十
百
千
万
```

---

### 14.2 English question_target

匹配：

```text
How many
How much
What
Which
Who
When
Where
Why
```

大小写不敏感。

---

### 14.3 Chinese question_target

匹配：

```text
多少
几个
几
什么
哪一个
哪种
谁
何时
什么时候
哪里
为什么
```

---

### 14.4 English comparison

匹配：

```text
more
less
fewer
greater
smaller
higher
lower
cheaper
more than
less than
at least
at most
no more than
no less than
```

---

### 14.5 Chinese comparison

匹配：

```text
更多
更少
大于
小于
高于
低于
至少
至多
不超过
不少于
超过
少于
```

---

### 14.6 English negation

匹配：

```text
not
no
never
without
cannot
can't
```

---

### 14.7 Chinese negation

匹配：

```text
不
没有
从不
不能
无法
未
无
```

---

### 14.8 English condition

匹配：

```text
if
when
given
assuming
provided
unless
```

---

### 14.9 Chinese condition

匹配：

```text
如果
当
当...时
假设
给定
除非
若
```

实现时不需要复杂模式，先匹配固定短语即可。

---

### 14.10 English operation

匹配常见数学和行为动词：

```text
buy
buys
bought
sell
sells
sold
add
adds
added
remove
removes
removed
give
gives
gave
take
takes
took
left
remain
total
difference
```

网络安全方向可以包括：

```text
execute
executes
deserialize
deserializes
encrypt
decrypt
upload
download
bypass
exploit
```

---

### 14.11 Chinese operation

匹配：

```text
买
购买
卖
增加
减少
移除
删除
给
拿走
剩下
总共
一共
执行
反序列化
加密
解密
上传
下载
绕过
利用
```

---

### 14.12 Object 规则

本 sprint 的 object 规则只要求最小启发式。

可以结合两种方式：

```text
1. 小型 object 词表
2. number 后邻近名词启发式
```

英文 object 词表示例：

```text
apples
books
notebooks
files
packets
users
servers
requests
records
items
```

中文 object 词表示例：

```text
苹果
书
笔记本
文件
数据包
用户
服务器
请求
记录
物品
```

number 后邻近名词启发式只作为补充，不要求完美。

注意：

```text
object 是候选，不是最终标签。
第一版允许低召回，但必须保证 offset 正确。
```

---

### 14.13 Cyber security term

英文术语：

```text
remote code execution
unsafe deserialization
SQL injection
cross-site scripting
privilege escalation
buffer overflow
command injection
path traversal
```

中文术语：

```text
远程代码执行
不安全反序列化
SQL 注入
跨站脚本
权限提升
缓冲区溢出
命令注入
路径遍历
```

大小写不敏感。

---

## 15. 去重、排序与截断

候选 span 需要：

```text
1. 对相同 start/end/type 去重。
2. 按 priority 排序。
3. 同一 priority 内按 start 从小到大排序。
4. 若数量超过 max_candidates，截断。
5. 截断后重新按 start 从小到大排序。
6. span_id 按最终排序结果重新编号：span_001, span_002, ...
```

如果同一文本片段被多个 type 命中，可以保留多个不同 type，但必须避免完全重复记录。

---

## 16. 测试要求

新增：

```text
tests/test_candidate_extraction.py
```

至少覆盖以下测试。

### 16.1 英文基础样例

输入：

```text
Tom has 3 apples and buys 2 more. How many apples does he have now?
```

至少验证能抽到：

```text
3 / number
2 / number
apples / object
buys / operation
How many / question_target
more / comparison
```

不要求只抽这些。

---

### 16.2 中文基础样例

输入：

```text
小明有3个苹果，又买了2个。现在有多少个苹果？
```

至少验证能抽到：

```text
3 / number
2 / number
苹果 / object
买 / operation
多少 / question_target
```

不要求只抽这些。

---

### 16.3 中文 offset 正确性

对中文样例中的所有 candidate，验证：

```python
question[candidate["start"]:candidate["end"]] == candidate["text"]
```

---

### 16.4 英文 offset 正确性

对英文样例中的所有 candidate，验证：

```python
question[candidate["start"]:candidate["end"]] == candidate["text"]
```

---

### 16.5 cyber security term

输入：

```text
The vulnerability allows remote code execution through unsafe deserialization.
```

至少验证能抽到：

```text
remote code execution / cyber_security_term
unsafe deserialization / cyber_security_term
```

---

### 16.6 中文 cyber security term

输入：

```text
该漏洞允许通过不安全反序列化实现远程代码执行。
```

至少验证能抽到：

```text
不安全反序列化 / cyber_security_term
远程代码执行 / cyber_security_term
```

---

### 16.7 max_candidates 截断

构造一个包含多个数字、对象和操作的长问题。

验证：

```text
len(candidates) <= max_candidates
```

---

### 16.8 不设置硬性最小值

构造一个只包含少量可命中 span 的问题。

验证：

```text
即使候选数量少于 8，也允许通过。
```

---

### 16.9 span_id 连续编号

验证最终输出：

```text
span_001
span_002
span_003
...
```

连续编号。

---

### 16.10 schema validation

验证 `build_candidate_span_records` 输出的每条 record 都能通过：

```python
validate_candidate_span_record(record)
```

---

## 17. 本 sprint 不做

不要实现：

```text
ablated question construction
NLI scoring
masked question construction
question recovery
recoverability scoring
label building
baseline CoT
hidden states
attention maps
trajectory stability
answer stability
attention anchor labeling
attention guidance
probe training
真实模型调用
```

不要新增：

```text
src/recover_attention/baseline_cot.py
scripts/02_build_baseline_cot.py
validate_baseline_cot_record
```

---

## 18. 必须运行命令

推荐命令顺序：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
python scripts/02_extract_candidate_spans.py --input data/processed/questions.jsonl --output data/processed/candidate_spans.jsonl
python -m pytest -q
```

如果当前目录是 git 仓库，还要运行：

```bash
git diff --name-only
git status --short
```

---

## 19. PROGRESS.md 更新要求

完成后，PROGRESS.md 仍保持短版状态索引，不要无限追加长日志。

更新当前状态：

```text
Sprint 1A 已完成。
下一步建议是 Sprint 1B：Ablated Question Construction。
```

在已完成 Sprint 摘要中新增：

```text
| Sprint 1A | 完成 | Candidate span extraction framework |
```

当前可运行命令中增加：

```bash
python scripts/02_extract_candidate_spans.py --input data/processed/questions.jsonl --output data/processed/candidate_spans.jsonl
```

最近一次检查结果更新为最新结果，例如：

```text
pytest: xx passed
smoke test: passed
prepare_data: passed
candidate extraction: passed
```

当前关键文件状态中补充：

```text
已完成：
- src/recover_attention/candidate_extraction.py
- scripts/02_extract_candidate_spans.py
- tests/test_candidate_extraction.py
- data/processed/candidate_spans.jsonl
```

下一阶段将新增或修改：

```text
- src/recover_attention/question_ablations.py
- scripts/03_build_ablated_questions.py
- tests/test_question_ablations.py
```

如果需要记录详细执行日志，写入：

```text
docs/progress/sprint_1_history.md
```

不要把完整长日志塞回 PROGRESS.md。

---

## 20. 验收标准

本 sprint 完成后应满足：

```text
1. src/recover_attention/candidate_extraction.py 有可替换 extractor framework。
2. extract_candidate_spans 支持 language / backend / max_candidates 参数。
3. rule_based backend 可用。
4. 中英文基础样例都能抽出候选 span。
5. 中文 start/end offset 正确。
6. 英文 start/end offset 正确。
7. cyber_security_term 中英文样例可用。
8. max_candidates 截断有效。
9. 不设置硬性最小候选数。
10. scripts/02_extract_candidate_spans.py 可运行。
11. data/processed/candidate_spans.jsonl 已生成。
12. 输出 records 通过 validate_candidate_span_record。
13. tests/test_candidate_extraction.py 覆盖核心规则。
14. python -m pytest -q 通过。
15. smoke test 通过。
16. prepare_data 通过。
17. 未实现 ablation / NLI / recover / trajectory / attention guidance / probe。
18. 未新增 baseline CoT 相关模块。
19. PROGRESS.md 保持短版状态索引。
```

---

## 21. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. candidate 数量统计
6. PROGRESS.md 更新摘要
7. 遗留问题
8. 下一步建议
```

不要自动开始 Sprint 1B。
