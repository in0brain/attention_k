# Sprint 4B-1：CyberMetric Canonical Schema 与 Domain Label Proxy

## 1. 定位

本 sprint 是 Sprint 4B 的第一个工程子阶段，只负责建立稳定的数据与标签接口。

本轮主线为：

```text
CyberMetric raw data
→ canonical cyber sample
→ deterministic option shuffle
→ option-letter label space
→ domain label proxy
→ schema validation
```

本轮不是完整 Sprint 4B，不进行：

```text
模型生成
F5 baseline
correct / wrong trace pair
hidden state 提取
causal site-transfer
probe 训练
steering
```

本轮的目标是为后续 Sprint 4B-2 模型 smoke 和 Sprint 4B-3 F5 baseline 提供可信、可测试、无标签泄漏的数据基础。

---

## 2. 本轮需要回答的问题

```text
Q1:
能否将 CyberMetric 原始 MCQ 数据稳定转换为统一 canonical schema？

Q2:
选项打乱后，option letter 与原始 cyber semantic label 的映射是否仍然正确？

Q3:
A / B / C / D 是否能在 Qwen2.5 tokenizer 下作为互不相同的单 token 标签？

Q4:
能否可靠解析模型未来可能生成的 option-letter answer，并定位 label-readout position？

Q5:
所有后续 inference feature 是否都能保证不包含 gold_label、gold_label_id 或 gold_label_text？
```

---

## 3. 执行前必读

必须先阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/README.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md
docs/progress/sprint_4_history.md

src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/answer_proxy_metrics.py
scripts/sprint_4B_download_and_audit_cyber_datasets.py
tests/test_schemas.py
```

其中：

```text
docs/reference/* 只用于理解研究背景；
当前 task card、AGENTS.md、schemas.py 和测试接口优先。
```

---

## 4. Preflight 要求

修改任何文件前，必须先输出 Preflight，并暂停等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表；
2. 当前 git branch 和 HEAD commit；
3. git working tree 是否干净；
4. 本轮允许修改的文件；
5. 本轮禁止修改的文件；
6. CyberMetric 原始数据是否存在；
7. 实际使用哪个 CyberMetric 文件；
8. 是否发现 schema 或任务范围冲突；
9. 本轮需要执行的测试命令；
10. 明确声明本轮不调用模型、不运行 GPU 推理。
```

CyberMetric 原始数据优先查找：

```text
data/raw/cyber/cybermetric/CyberMetric-2000-v1.json
```

若不存在，再依次检查：

```text
data/raw/cyber/cybermetric/CyberMetric-500-v1.json
data/raw/cyber/cybermetric/CyberMetric-10000-v1.json
```

如果所有文件均不存在：

```text
停止执行；
不得伪造数据；
提示用户先运行：

conda run -n recover_attention python \
  scripts/sprint_4B_download_and_audit_cyber_datasets.py \
  --output-dir outputs/logs/sprint_4B_dataset_download_audit
```

---

## 5. 本轮允许修改的文件

```text
docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md

src/recover_attention/schemas.py
src/recover_attention/cyber_data.py
src/recover_attention/domain_label_proxy.py

scripts/sprint_4B_1_prepare_cybermetric.py

tests/test_schemas.py
tests/test_cyber_data.py
tests/test_domain_label_proxy.py

PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

仅当现有 JSONL 工具不能满足要求时，允许最小修改：

```text
src/recover_attention/data_io.py
tests/test_data_io.py
```

关于 `scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py`（4B smoke 脚本）：

```text
本卡强制的新函数签名可能破坏该脚本的现行依赖。允许二选一：
1. 对该脚本做最小兼容性修改（仅限适配 cyber_data / domain_label_proxy
   的新签名，不得扩展功能、不得运行它）；
2. 若适配成本过高，允许本轮容忍其暂时失效，但必须在 preflight 第 8 条
   与 review gate 中明确记录：该脚本将由 4B-2 / 4B-3 的新脚本取代。
不允许的做法：不声明地留下一个 import 即崩的脚本。
```

---

## 6. 本轮禁止修改的文件

```text
README.md
AGENTS.md

docs/reasoning-aware-attention-guidance/*
docs/reference/*

src/recover_attention/answer_proxy_metrics.py
src/recover_attention/module_causal_tracing.py
src/recover_attention/activation_patching.py
src/recover_attention/mlp_readout_direction.py
src/recover_attention/mlp_readout_attribution.py
src/recover_attention/approx_j_lens_readout.py

任何 sprint_3* 脚本
任何 sprint_4A* 脚本
```

不得覆盖已有输出目录：

```text
outputs/logs/sprint_3*
outputs/logs/sprint_4A*
outputs/logs/sprint_4B_dataset_download_audit
```

---

## 7. 严格禁止事项

本轮禁止：

```text
调用 Qwen 或其他 causal language model
调用 Ollama
调用外部 LLM API
运行 GPU 推理
生成 completion
采样 reasoning traces
计算 final logits
计算 F5 AUROC / AUPRC
训练任何 probe
提取 hidden states
提取 attention maps
执行 activation patching
执行 steering / nudge
构造 correct / wrong trace pair
进入 Sprint 4C
执行 2000-scale 模型实验
```

不得声称：

```text
hallucination reduced
answer accuracy improved
F5 baseline established
causal site transferred
probe works
ready for Sprint 4C
ready for 2000 rerun
```

---

# 8. 输入

## 8.1 原始数据

默认输入：

```text
data/raw/cyber/cybermetric/CyberMetric-2000-v1.json
```

脚本必须支持：

```bash
--input-path
```

不得只写死默认路径。

## 8.2 样本规模

默认转换全部输入记录，但同时生成一个固定 seed 的 smoke manifest：

```text
smoke_questions = 240
shuffle_seed = 42
split_seed = 42
```

命令行必须支持：

```bash
--max-questions
--smoke-questions
--shuffle-seed
--split-seed
```

---

# 9. Canonical Cyber Sample Schema

在 `src/recover_attention/schemas.py` 中新增 record type：

```text
cyber_sample
```

## 9.1 顶层必需字段

```text
example_id
dataset
source
group_id
task_type
input_text
question
candidate_labels
candidate_choices
gold_label
gold_label_id
gold_label_text
label_space
metadata
```

建议加入：

```python
REQUIRED_FIELDS["cyber_sample"] = [
    "example_id",
    "dataset",
    "source",
    "group_id",
    "task_type",
    "input_text",
    "question",
    "candidate_labels",
    "candidate_choices",
    "gold_label",
    "gold_label_id",
    "gold_label_text",
    "label_space",
    "metadata",
]
```

新增：

```python
validate_cyber_sample_record(record: dict) -> None
```

---

## 9.2 Canonical record 示例

```json
{
  "example_id": "cybermetric_000001",
  "dataset": "cybermetric",
  "source": "CyberMetric-2000-v1",
  "group_id": "cybermetric_family_001",
  "task_type": "multiple_choice_qa",
  "input_text": "",
  "question": "Which security mechanism prevents ...?",
  "candidate_labels": ["A", "B", "C", "D"],
  "candidate_choices": [
    {
      "choice": "A",
      "label_id": null,
      "label_text": "Option semantic text A",
      "original_position": 2
    },
    {
      "choice": "B",
      "label_id": null,
      "label_text": "Option semantic text B",
      "original_position": 0
    },
    {
      "choice": "C",
      "label_id": null,
      "label_text": "Option semantic text C",
      "original_position": 3
    },
    {
      "choice": "D",
      "label_id": null,
      "label_text": "Option semantic text D",
      "original_position": 1
    }
  ],
  "gold_label": "B",
  "gold_label_id": null,
  "gold_label_text": "Option semantic text B",
  "label_space": "mcq_option_letter",
  "metadata": {
    "original_index": 0,
    "original_gold_position": 0,
    "shuffled_gold_position": 1,
    "option_shuffle_seed": 42,
    "split_seed": 42,
    "split": "train",
    "raw_category": null
  }
}
```

---

## 9.3 字段规则

### `example_id`

必须：

```text
非空
全数据唯一
固定 seed 下稳定
```

推荐格式：

```text
cybermetric_<zero-padded original index>
```

### `dataset`

固定为：

```text
cybermetric
```

### `source`

记录实际原始文件名，例如：

```text
CyberMetric-2000-v1
```

### `group_id`

用于 grouped split 和防泄漏。

优先级：

```text
1. 原始数据已有 category / family / source group；
2. 对标准化问题文本构造稳定 hash；
3. 不得直接使用随机数。
```

同一 `group_id` 不得跨 split。

### `task_type`

固定为：

```text
multiple_choice_qa
```

### `input_text`

若数据集没有独立 context，则使用：

```text
""
```

不得填入 gold answer。

### `candidate_labels`

必须与选项数对应，例如：

```json
["A", "B", "C", "D"]
```

要求：

```text
长度至少为 2；
元素唯一；
必须与 candidate_choices.choice 顺序完全一致。
```

### `candidate_choices`

每项必须包含：

```text
choice
label_id
label_text
original_position
```

要求：

```text
choice 与 candidate_labels 对齐；
label_text 非空；
original_position 为非负整数；
option shuffle 后 semantic text 与 gold 映射不得错位。
```

### `gold_label`

只能是 option letter，例如：

```text
A / B / C / D
```

不得直接使用：

```text
ATT&CK id
CWE id
完整答案文本
多 token semantic label
```

### `gold_label_id`

若原数据无 semantic id，则允许：

```text
null
```

但字段必须存在。

### `gold_label_text`

必须保存正确选项的完整语义文本。

不得只保存 `A/B/C/D`。

### `label_space`

固定为：

```text
mcq_option_letter
```

### `metadata`

至少包含：

```text
original_index
original_gold_position
shuffled_gold_position
option_shuffle_seed
split_seed
split
raw_category
```

---

# 10. Option Shuffle 规则

必须实现 deterministic option shuffle。

推荐函数：

```python
shuffle_candidate_choices(
    choices: list[dict],
    gold_original_position: int,
    *,
    example_id: str,
    seed: int,
) -> tuple[list[dict], str]
```

要求：

```text
1. 相同 example_id + seed 得到完全相同结果；
2. 不同 seed 应允许产生不同顺序；
3. gold semantic text 在 shuffle 后仍映射到正确 gold_label；
4. original_position 必须保留；
5. 不得原地修改调用方传入的 list；
6. 不得让同一 semantic label 永远对应固定 option letter。
```

为避免所有样本共享同一排列，推荐使用：

```text
per-example seed = stable_hash(example_id, global_seed)
```

不得使用 Python 内置 `hash()` 作为跨进程稳定 hash。

可使用：

```text
hashlib.sha256
```

---

# 11. Grouped Split 规则

实现：

```python
grouped_split(
    records,
    *,
    train_ratio,
    dev_ratio,
    test_ratio,
    seed,
)
```

默认比例：

```text
train = 0.70
dev   = 0.15
test  = 0.15
```

要求：

```text
1. 同一 group_id 只能出现在一个 split；
2. 固定 seed 下结果可复现；
3. 所有样本必须恰好分配到一个 split；
4. 不得按单条记录随机切分后再补 group；
5. 输出 split 数量和 label-position 分布。
```

若原数据缺少可靠 family 字段，必须在报告中注明：

```text
group_id is a deterministic question-derived grouping proxy
```

不得声称其等价于真实 question family。

---

# 12. `src/recover_attention/cyber_data.py`

至少实现以下函数：

```python
load_cybermetric_records(path: Path) -> list[dict]

normalize_cybermetric_record(
    raw_record: dict,
    *,
    original_index: int,
    source: str,
) -> dict

build_candidate_choices(raw_record: dict) -> tuple[list[dict], int]

shuffle_candidate_choices(
    choices: list[dict],
    gold_original_position: int,
    *,
    example_id: str,
    seed: int,
) -> tuple[list[dict], str]

build_group_id(raw_record: dict, normalized_question: str) -> str

to_canonical_cyber_sample(
    raw_record: dict,
    *,
    original_index: int,
    source: str,
    shuffle_seed: int,
) -> dict

grouped_split(
    records: list[dict],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict]

build_mcq_prompt(record: dict) -> str

audit_cyber_samples(records: list[dict]) -> dict
```

## `build_mcq_prompt`

本轮只负责构造字符串，不调用模型。

模板固定为：

```text
{input_text}

Question: {question}

Options:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Think briefly step by step, then answer with exactly one letter as:
Answer: <letter>
```

若 `input_text` 为空，不得输出多余的 `"None"`。

---

# 13. `src/recover_attention/domain_label_proxy.py`

本轮只实现纯函数，不加载模型。

至少实现：

```python
option_token_ids(tokenizer, labels: list[str]) -> dict[str, int]

parse_option_answer(
    text: str,
    valid_labels: list[str],
) -> dict

locate_label_readout_position(
    tokenizer,
    prompt: str,
    completion: str,
    parsed_label: str,
) -> int

label_margin(option_logits: dict[str, float]) -> float

label_entropy(option_logits: dict[str, float]) -> float

full_entropy(logits) -> float

self_consistency_features(
    greedy_label: str | None,
    sampled_labels: list[str | None],
    valid_labels: list[str],
) -> dict

classify_trace_by_option(
    parsed_label: str | None,
    gold_label: str,
) -> str
```

---

## 13.1 `option_token_ids`

必须处理 Qwen tokenizer 的 leading whitespace 问题。

要求：

```text
1. 对每个 option letter 得到一个有效非空白 token；
2. A/B/C/D token id 两两不同；
3. 如果 tokenizer 输出多于一个非空白 token，则报错；
4. 如果不同 option letter 映射到同一 token id，则报错；
5. 错误消息必须指出具体 label 与 token ids。
```

不得简单使用：

```python
tokenizer.encode(" A")[0]
```

并默认它就是字母 token。

---

## 13.2 `parse_option_answer`

解析优先级：

```text
1. 显式格式：
   Answer: B

2. 大小写兼容：
   answer: b

3. 最后出现的孤立有效选项字母：
   ... therefore the answer is B.

4. 无可靠选项：
   parse_failure
```

返回格式建议：

```json
{
  "parsed_label": "B",
  "parse_method": "explicit_answer_marker",
  "parse_failure": false
}
```

解析失败：

```json
{
  "parsed_label": null,
  "parse_method": "parse_failure",
  "parse_failure": true
}
```

不得把以下普通文本误识别成选项：

```text
Aes
BERT
CVE
DNS
Class A network
```

除非它满足明确的 answer pattern 或孤立字母规则。

---

## 13.3 `locate_label_readout_position`

定义：

```text
label-readout position 是最终答案字母 token 之前、用于预测该字母的 token position。
```

要求：

```text
1. 能定位 "Answer: B" 中 B 的 token index；
2. 返回 B token 的前一位置；
3. 若答案字母无法唯一定位，必须报错；
4. 不得通过 gold_label 定位；
5. 只能使用 completion 中实际解析出的 parsed_label。
```

本轮只做 tokenizer-level 单元测试，不运行 causal LM。

---

## 13.4 `label_margin`

输入：

```python
{
    "A": logit_a,
    "B": logit_b,
    "C": logit_c,
    "D": logit_d,
}
```

输出：

```text
top1 logit - top2 logit
```

要求：

```text
至少两个 label；
所有值必须为有限实数；
不得使用 gold_label。
```

---

## 13.5 `label_entropy`

只在候选 option logits 上归一化。

计算：

```text
p_i = softmax(option_logits)_i
H = -Σ p_i log p_i
```

不得把全词表 logits 混入该函数。

---

## 13.6 `full_entropy`

计算全词表 softmax entropy。

必须：

```text
数值稳定；
支持 list / NumPy array / Torch tensor 中项目已有的最小必要形式；
不引入新大型依赖。
```

如果项目已依赖 Torch，可以优先支持 Torch tensor。

---

## 13.7 `self_consistency_features`

允许输入：

```text
greedy parsed label
多个 sampled parsed labels
valid option labels
```

输出至少包含：

```text
num_samples
num_parsed_samples
parse_failure_count
parse_failure_rate
greedy_label
sample_vote_counts
sample_majority_label
self_consistency_with_greedy
majority_agrees_with_greedy
```

严格要求：

```text
不得接收 gold_label 参数；
不得输出 gold_label；
不得计算 is_correct；
不得使用任何 evaluation label。
```

---

# 14. Schema 泄漏边界

新增允许的 evaluation 字段：

```text
gold_label
gold_label_id
gold_label_text
```

但这些字段不得进入未来 feature record。

本轮至少新增一个辅助函数或测试约束：

```python
FORBIDDEN_INFERENCE_FEATURE_FIELDS = {
    "gold_label",
    "gold_label_id",
    "gold_label_text",
}
```

可实现：

```python
assert_no_gold_label_leakage(feature_record: dict) -> None
```

检查应递归覆盖嵌套字典。

至少禁止：

```text
gold_label
gold_label_id
gold_label_text
correct_option
answer_key
target_label
```

如果某字段仅存在于 canonical sample 中，不视为泄漏；只有当它被复制到 feature 输出中才视为泄漏。

---

# 15. 主脚本

新增：

```text
scripts/sprint_4B_1_prepare_cybermetric.py
```

职责只包括：

```text
1. 读取 CyberMetric raw JSON；
2. 标准化 raw records；
3. 构造 canonical cyber samples；
4. deterministic option shuffle；
5. grouped split；
6. schema validation；
7. 输出 full canonical dataset；
8. 输出固定规模 smoke manifest；
9. 输出 dataset / label-space / position-bias pre-model audit；
10. 输出 review gate。
```

不得包含：

```text
transformers model loading
AutoModelForCausalLM
generate()
forward()
hidden states
logits
F5 evaluation
patching
probe
```

---

# 16. 命令行参数

脚本至少支持：

```bash
--input-path
--processed-output
--output-dir
--max-questions
--smoke-questions
--shuffle-seed
--split-seed
--train-ratio
--dev-ratio
--test-ratio
--overwrite
```

默认值建议：

```text
input-path:
data/raw/cyber/cybermetric/CyberMetric-2000-v1.json

processed-output:
data/processed/cyber/cybermetric.jsonl

output-dir:
outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy

smoke-questions:
240

shuffle-seed:
42

split-seed:
42
```

如果输出已存在且未指定 `--overwrite`：

```text
停止并给出清晰错误；
不得静默覆盖。
```

---

# 17. 输出文件

必须输出：

```text
data/processed/cyber/cybermetric.jsonl

outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/
  preflight_report.md
  dataset_audit_report.json
  label_space_report.json
  option_position_bias_pre_model_report.json
  cyber_sample_manifest.jsonl
  cyber_sample_smoke_manifest.jsonl
  review_gate_cybermetric_schema_and_label_proxy.md
```

---

## 17.1 `dataset_audit_report.json`

至少包含：

```text
input_path
source_name
num_raw_records
num_valid_records
num_invalid_records
num_output_records
num_options_distribution
missing_question_count
missing_answer_count
duplicate_example_id_count
group_count
split_counts
split_group_counts
group_leakage_count
label_distribution
license_note
warnings
```

---

## 17.2 `label_space_report.json`

本轮不加载 causal LM，因此分为两部分。

必须包含静态部分：

```text
candidate_labels
all_labels_unique
all_labels_single_character
label_space
semantic_labels_preserved
gold_mapping_validation_passed
```

真实 Qwen tokenizer 检查允许通过以下两种方式之一完成：

```text
A. 若本地 tokenizer 可用：
   加载 tokenizer only，不加载模型；
   输出 tokenizer_name_or_path、token_ids、single_token_passed。

B. 若 tokenizer 不可用：
   输出 tokenizer_check_status = "skipped";
   写明 skipped_reason；
   FakeQwenTokenizer 单元测试仍必须通过。
```

仅加载 tokenizer 不视为模型推理，但不得下载模型。

脚本不得自动联网下载 tokenizer。

---

## 17.3 `option_position_bias_pre_model_report.json`

本文件只包含模型运行前可计算的内容：

```text
gold choice distribution over A/B/C/D
gold choice proportion
original gold position distribution
shuffled gold position distribution
shuffle seed
position balance warnings
semantic label to option-letter mapping counts
whether any semantic label is always mapped to one option letter
whether option order is deterministic under fixed seed
whether option order changes under a different seed
```

不得包含：

```text
greedy predicted choice distribution
sampled predicted choice distribution
model option-position accuracy
F5 metrics
```

这些属于 Sprint 4B-2。

---

## 17.4 `cyber_sample_manifest.jsonl`

包含本轮所有 canonical records，或者与：

```text
data/processed/cyber/cybermetric.jsonl
```

内容一致。

若两者内容一致，必须在 review gate 中说明二者用途：

```text
processed dataset:
后续 pipeline 的标准输入；

output manifest:
本 sprint 的可审计输出快照。
```

---

## 17.5 `cyber_sample_smoke_manifest.jsonl`

包含固定 seed 选出的 240 条样本。

要求：

```text
split 分布可审计；
不得只取输入文件前 240 条；
固定 seed 下完全可复现；
不得破坏 group 边界。
```

---

# 18. 测试要求

## 18.1 `tests/test_schemas.py`

至少新增：

```text
1. 合法 cyber_sample record 通过；
2. 缺少必需字段失败；
3. candidate_labels 与 candidate_choices 不一致失败；
4. gold_label 不在 candidate_labels 中失败；
5. gold_label_text 与 gold choice semantic text 不一致失败；
6. 重复 candidate label 失败；
7. 非法 label_space 失败；
8. metadata split 非法失败；
9. original_position 非整数失败；
10. 空 question 或空 label_text 失败。
```

---

## 18.2 `tests/test_cyber_data.py`

至少覆盖：

```text
1. CyberMetric raw record 能正确解析；
2. canonical schema 字段齐全；
3. example_id 稳定且唯一；
4. fixed seed shuffle 可复现；
5. different seed 可以改变 option order；
6. shuffle 后 gold semantic text 映射正确；
7. original_position 被保留；
8. candidate_choices 保存 label_id / label_text；
9. build_mcq_prompt 包含全部选项；
10. prompt 不泄漏 gold label；
11. grouped split 无 group leakage；
12. split 结果固定 seed 可复现；
13. smoke sample 不是简单取前 N 条；
14. gold option position 分布可正确统计；
15. 输入 list 不被 shuffle 函数原地修改。
```

---

## 18.3 `tests/test_domain_label_proxy.py`

至少覆盖：

```text
1. FakeQwenTokenizer 下 A/B/C/D 各为一个非空白 token；
2. A/B/C/D token id 两两不同；
3. leading whitespace token 被正确剥离；
4. 多个有效 token 时 option_token_ids 报错；
5. token collision 时 option_token_ids 报错；
6. "Answer: B" 显式解析正确；
7. "answer: b" 大小写解析正确；
8. 最后的孤立字母 fallback 正确；
9. 普通单词中的 A/B/C/D 不被误解析；
10. 无答案时返回 parse_failure；
11. label-readout position 位于答案字母前一位；
12. label_margin 与手算一致；
13. label_entropy 与手算一致；
14. full_entropy 与手算一致；
15. self_consistency_features 不接受 gold_label；
16. self_consistency_features 输出不包含 gold 字段；
17. leakage checker 能发现嵌套 gold_label；
18. classify_trace_by_option 正确区分 correct / wrong / parse_failure。
```

---

# 19. 验收标准

## 19.1 数据转换

```text
[OK] CyberMetric 原始数据成功读取；
[OK] 至少生成 240 条合法 canonical samples；
[OK] 每条记录通过 validate_cyber_sample_record；
[OK] 每条记录至少有 2 个选项；
[OK] 所有 example_id 唯一；
[OK] candidate labels 唯一；
[OK] gold label 始终属于 candidate labels；
[OK] gold_label_text 与正确 candidate choice 一致；
[OK] semantic option text 未丢失；
[OK] option shuffle 固定 seed 可复现；
[OK] grouped split 无 group leakage；
[OK] full manifest 和 smoke manifest 成功输出。
```

## 19.2 Option label proxy

```text
[OK] FakeQwenTokenizer 测试全部通过；
[OK] 显式 Answer marker 解析正确；
[OK] 孤立字母 fallback 解析正确；
[OK] 普通文本不误解析；
[OK] parse failure 被如实保留；
[OK] label-readout position 定位正确；
[OK] margin / entropy 计算正确；
[OK] inference feature 输出不含 gold label 信息。
```

## 19.3 审计

```text
[OK] dataset_audit_report.json 完整；
[OK] label_space_report.json 完整；
[OK] option_position_bias_pre_model_report.json 完整；
[OK] semantic label 与 option letter 映射可追踪；
[OK] 未将 option-letter pattern 描述为 cyber semantic signal；
[OK] 未调用模型；
[OK] 未运行 F5；
[OK] 未运行 site-transfer；
[OK] 未训练 probe；
[OK] 未执行 steering。
```

---

# 20. 推荐运行命令

## 20.1 目标测试

```bash
conda run -n recover_attention python -m pytest \
  tests/test_schemas.py \
  tests/test_cyber_data.py \
  tests/test_domain_label_proxy.py \
  -q
```

## 20.2 数据准备

```bash
conda run -n recover_attention python \
  scripts/sprint_4B_1_prepare_cybermetric.py \
  --input-path data/raw/cyber/cybermetric/CyberMetric-2000-v1.json \
  --processed-output data/processed/cyber/cybermetric.jsonl \
  --output-dir outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy \
  --smoke-questions 240 \
  --shuffle-seed 42 \
  --split-seed 42 \
  --overwrite
```

## 20.3 语法检查

```bash
conda run -n recover_attention python -m py_compile \
  src/recover_attention/cyber_data.py \
  src/recover_attention/domain_label_proxy.py \
  scripts/sprint_4B_1_prepare_cybermetric.py
```

## 20.4 全量测试

```bash
conda run -n recover_attention python -m pytest -q
```

---

# 21. Review Gate

必须生成：

```text
outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/
review_gate_cybermetric_schema_and_label_proxy.md
```

逐条回答：

```text
1. 实际使用了哪个 CyberMetric 原始文件？
2. 原始记录数是多少？
3. 成功转换多少条？
4. 丢弃多少条？原因分别是什么？
5. canonical schema 是否已经加入 schemas.py？
6. 所有 canonical records 是否通过 schema validation？
7. candidate_choices 是否保留完整 semantic label text？
8. option shuffle 是否 deterministic？
9. 不同 seed 是否允许改变 option order？
10. shuffle 后 gold semantic mapping 是否全部正确？
11. group_id 如何构造？
12. grouped split 是否存在 leakage？
13. train / dev / test 各有多少样本和 group？
14. A/B/C/D 的 gold position 分布如何？
15. 是否存在严重 option-position imbalance？
16. 是否存在 semantic label 固定绑定某一 option letter？
17. FakeQwenTokenizer 检查是否通过？
18. 真实 Qwen tokenizer 检查是否执行？
19. 若未执行，skipped_reason 是什么？
20. parse_option_answer 的测试是否通过？
21. label-readout position 测试是否通过？
22. inference feature 是否通过 gold leakage 检查？
23. 本轮是否调用任何 causal LM？答案必须为 no。
24. 本轮是否生成任何 completion？答案必须为 no。
25. 本轮是否计算 F5？答案必须为 no。
26. 本轮是否运行 site-transfer？答案必须为 no。
27. 本轮是否训练 probe？答案必须为 no。
28. 本轮是否执行 steering？答案必须为 no。
29. 是否允许进入 Sprint 4B-2？
30. Sprint 4B-2 的明确输入文件是什么？
```

---

# 22. 进入 Sprint 4B-2 的条件

只有以下条件全部满足，才允许进入 Sprint 4B-2：

```text
1. 至少 240 条 canonical samples 可用；
2. schema validation 全部通过；
3. semantic option text 无丢失；
4. gold mapping validation 全部通过；
5. grouped split 无泄漏；
6. option shuffle 可复现；
7. 无严重 gold option-position imbalance；
8. FakeQwenTokenizer 测试通过；
9. parser 与 label-readout locator 测试通过；
10. gold leakage tests 通过；
11. full pytest 通过。
```

若真实 Qwen tokenizer 尚未检查，可以进入 4B-2 的 preflight，但必须先完成 tokenizer-only 检查，不能直接加载模型推理。

---

# 23. 下一步任务边界

本 sprint 完成后的下一步是：

```text
Sprint 4B-2:
CyberMetric Small Model Smoke and F5 Feature Plumbing
```

建议规模：

```text
32 questions
1 greedy run per question
3 sampled runs per question
max_new_tokens = 128 或 192
```

4B-2 只验证：

```text
模型路径解析
prompt
completion parser
parse failure rate
label-readout slot
option-position model bias
F5 feature plumbing
```

4B-2 不直接运行 240 × 7 的正式实验。

### Carryover 事项（来自已废止的旧 4B-1 卡，不得丢失）

以下三项由 `sprint_4B_2_3_carryover_notes.md`（原
`sprint_4B_1_prompt_fix_full_run_and_cost_tiers.md`，已降级为素材笔记）
转移而来，分别归属 4B-2 / 4B-3：

```text
1.（4B-2）Prompt A/B 测试：裸补全 prompt vs chat template
   （tokenizer.apply_chat_template）。背景：4B smoke 发现 8/90 条
   parse failure 全部是生成退化（无限重复、乱码拼贴），是 Instruct
   模型裸补全的典型症状。必须实现可测试的退化判定函数
   （单字符连跑 >= 30 / 长度 >= 6 的子串连续重复 >= 5 次 /
    截断式失败），并用 smoke 的真实退化样例做测试正例。
   决策规则：parse_failure + degeneration 之和低者胜，
   差距 < 0.02 平手时选 chat template。
   chat 胜出时必须补 label-readout 定位在 chat 全文上的单元测试。

2.（4B-2）采样保护检查：确认 generate 调用带
   renormalize_logits=True 与 remove_invalid_values=True
   （3C-0 方案），缺失则补上并记录。

3.（4B-3）F5 cost-tier 拆分：
   kill_bar_single_forward = margin/entropy 组合（一次前向可得）；
   kill_bar_sampling       = 含 self-consistency 的全组合（贵 ~7x）。
   双门槛都要带分组 bootstrap CI 写入 review gate；
   4C 的 F1/F4（single-forward 内部特征）对照前者，
   F3（多采样一致性）对照后者。
```

---

# 24. 完成后必须更新

```text
PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

`PROGRESS.md` 顶部状态应简洁记录：

```text
Sprint 4B-1 completed / blocked
实际数据文件
转换样本数
测试结果
输出目录
遗留问题
是否允许进入 4B-2
```

---

# 25. 本轮最多允许的结论

完成后最多可以写：

```text
CyberMetric has been converted into a validated canonical MCQ schema with
deterministic option randomization, preserved semantic option labels, and a
tested option-letter label proxy. The project is ready for a small-model
Sprint 4B-2 smoke test.
```

不得写：

```text
The detector works.
F5 is effective.
Hallucination is reduced.
Accuracy is improved.
The causal site transfers.
The project is ready for Sprint 4C.
```
