# 实验进度记录：Recoverability-Guided Attention Allocation

## 0. 当前实验边界

当前只做 v0 / v1 阶段：

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring
```

暂时不做：

```text
probe 训练
attention guidance
hidden states
trajectory analysis
大规模实验
```

------

## 1. 现有进度

### (1) 项目结构

已建立项目目录：

```text
recover_attention_project/
  data/
  docs/
  configs/
  src/
  scripts/
  outputs/
```

当前状态：

```text
[完成 / 未完成 / 部分完成]
```

相关文件：

```text
xxx
```

------

### (2) 数据准备

已完成内容：

```text
xxx
```

输入文件：

```text
data/raw/xxx.jsonl
```

输出文件：

```text
data/processed/questions.jsonl
```

当前样本数：

```text
xxx
```

检查结果：

```text
xxx
```

------

### (3) Candidate Span 抽取

已完成内容：

```text
xxx
```

当前支持的 span 类型：

```text
number
entity
operation
relation
question_target
```

输出文件：

```text
data/processed/candidate_spans.jsonl
```

检查结果：

```text
xxx
```

------

## 2. 需要下一步干的事项

### (1) 实现 ablated question 构造脚本

目标：

```text
根据 candidate_spans.jsonl，为每个 candidate span 构造 ablated question，用于后续 NLI 判断。
```

需要新增或修改的文件：

```text
src/recover_attention/question_ablations.py
scripts/03_build_ablated_questions.py
```

输入文件：

```text
data/processed/questions.jsonl
data/processed/candidate_spans.jsonl
```

输出文件：

```text
data/processed/ablated_questions.jsonl
```

输出格式示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_question": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "ablated_question": "Tom has some apples and buys 2 more. How many apples does he have now?"
}
```

实现要求：

```text
1. 支持 delete / mask / generalize 三种 ablation_type。
2. number 类型优先 generalize 成 some / a certain number。
3. entity 类型可 generalize 成 someone / something。
4. operation / relation 类型可以先用 delete 和 mask。
5. 所有输出必须是 jsonl。
6. 每条记录必须保留 id、span_id、span_text、span_type、original_question、ablated_question。
```

验收标准：

```text
[OK] 能成功运行脚本
[OK] 每个 candidate span 至少生成 1 条 ablated question
[OK] 输出 jsonl 可以正常读取
[OK] 抽查 10 条 ablated question，语句基本可读
```

运行命令：

```bash
python scripts/03_build_ablated_questions.py \
  --input data/processed/candidate_spans.jsonl \
  --output data/processed/ablated_questions.jsonl
```

禁止事项：

```text
不要实现 NLI scoring。
不要实现 recover question。
不要改动 attention guidance。
不要一次性重构整个项目。
```

------

### (2) 实现双向 NLI scoring 脚本

目标：

```text
对 original_question 和 ablated_question 做双向 NLI，判断删除或泛化某个 span 后问题语义是否等价。
```

需要新增或修改的文件：

```text
src/recover_attention/nli_scoring.py
scripts/04_run_nli_scoring.py
```

输入文件：

```text
data/processed/ablated_questions.jsonl
```

输出文件：

```text
data/processed/nli_scores.jsonl
```

输出格式示例：

```json
{
  "id": "gsm8k_0001",
  "span_id": "span_001",
  "span_text": "3",
  "span_type": "number",
  "ablation_type": "generalize",
  "original_to_ablated": "entailment",
  "ablated_to_original": "neutral",
  "semantic_necessity_label": "Information Loss"
}
```

标签规则：

```text
双向 entailment：
  Equivalent

original → ablated entailment
ablated → original neutral / contradiction：
  Information Loss

ablated → original entailment
original → ablated neutral / contradiction：
  Added Assumption

双向都不是 entailment：
  Non-equivalent
```

验收标准：

```text
[OK] 能输出双向 NLI 结果
[OK] 每条 ablated question 有 semantic_necessity_label
[OK] nli_scores.jsonl 行数与 ablated_questions.jsonl 对齐
[OK] 抽查 20 条结果，标签基本合理
```

运行命令：

```bash
python scripts/04_run_nli_scoring.py \
  --input data/processed/ablated_questions.jsonl \
  --output data/processed/nli_scores.jsonl
```

禁止事项：

```text
不要训练 probe。
不要做 attention guidance。
不要调用 recover question。
```

------

## 3. 完成后更新规则

每次完成一个“下一步事项”后：

```text
1. 将该事项从“需要下一步干的事项”移动到“现有进度”。
2. 补充实际运行命令。
3. 补充输出文件路径。
4. 补充检查结果。
5. 写明是否有失败样本或未解决问题。
6. 再新增下一步事项。
```

------

## 4. 当前优先级

当前优先级：

```text
P0：先跑通 NLI semantic necessity pipeline。
P1：再做 mask-recover。
P2：再比较 NLI label 与 recoverability label。
P3：最后考虑 probe 和 attention guidance。
```