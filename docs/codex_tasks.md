# Codex Tasks

本文件定义本项目的 Codex sprint 拆分方式、task card 规范、阶段边界和长期路线。

本文件不是具体 task card。

具体执行入口在：

```text
docs/codex_tasks/*.md
```

Codex 执行具体任务时，必须以当前 task card 为直接边界。

---

# 1. 项目主线

本项目主线是：

```text
Reasoning-Aware Attention Guidance
```

项目不是单纯做：

```text
hallucination classifier
```

也不是单纯做：

```text
key span discovery
```

本项目的完整研究闭环是：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

NLI 是新增的辅助信号，用于判断某个 candidate span 在语义层面是否必要。

NLI 不替代：

```text
semantic recoverability
trajectory stability
answer stability
attention steering
```

最终目标不是输出普通的：

```text
hallucination_label
```

而是形成：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

---

# 2. Codex 工作原则

每个 sprint 必须满足：

```text
1. 目标单一。
2. 修改文件范围小。
3. 输入文件明确。
4. 输出文件明确。
5. 验收命令明确。
6. 测试要求明确。
7. 必须更新 PROGRESS.md。
8. 不自动开始下一 sprint。
```

每个 sprint 不应同时实现多个实验阶段。

例如：

```text
错误：同时实现 candidate span extraction + NLI scoring + recoverability scoring
正确：只实现 candidate span extraction
```

Codex 不应根据长期路线自行提前实现后续阶段。

---

# 3. Task Card 必须包含的部分

每张 task card 至少包含：

```text
1. 目标
2. 允许修改的文件
3. 禁止修改的文件
4. 输入文件
5. 输出文件
6. 实现要求
7. 测试要求
8. 必须运行的命令
9. 验收标准
10. 禁止事项
11. PROGRESS.md 更新要求
12. 完成后回复格式
```

如果当前 sprint 涉及方法概念，task card 必须说明需要阅读哪些 `docs/skill/*.md` 子文档。

例如：

```text
candidate span extraction 相关 sprint 应阅读：
docs/skill/method.md
docs/skill/label_schema.md
docs/skill/experiment_guide.md
```

如果当前 sprint 只做环境、文档或工程验收，则不应要求 Codex 阅读过多方法文档。

---

# 4. Preflight 规则

Codex 修改任何文件之前，必须先输出 Preflight。

Preflight 必须包括：

```text
1. 已阅读文件列表
2. 本次允许修改的文件
3. 本次禁止修改的文件
4. 本次必须运行的命令
5. 是否需要参考 docs/reference/*
6. 是否发现 task card 与当前提示词、AGENTS.md、SKILL.md 或 skill 子文档冲突
```

Preflight 输出后必须暂停，等待用户确认。

用户确认后，Codex 才能修改文件。

---

# 5. 冲突优先级

如果当前用户消息、task card、AGENTS.md、SKILL.md、skill 子文档、prompts 模板或 reference 文档发生冲突，优先级为：

```text
当前用户消息
> 当前 task card
> AGENTS.md
> docs/skill/SKILL.md
> docs/skill/*.md
> docs/skill/prompts.md
> docs/reference/*
```

如发现冲突，Codex 必须：

```text
1. 在 Preflight 中报告冲突。
2. 说明采用哪一条规则。
3. 在最终回复的“遗留问题”中再次记录冲突。
```

---

# 6. Reference 文档使用规则

默认不要阅读：

```text
docs/reference/*
```

`docs/reference/*` 是长期完整参考文档，不是默认执行入口。

只有在以下情况中才允许使用：

```text
1. 当前 task card 明确要求阅读某个 reference 文件。
2. 当前 task card 信息不足，且用户明确允许参考 reference。
3. 用户当前消息明确要求对照 reference。
```

即使允许阅读 reference，也不能把 reference 中的后续阶段内容提前实现。

---

# 7. 环境与命令规则

当前推荐环境：

```text
conda env: recover_attention
python: 3.10
```

安装依赖必须使用：

```bash
python -m pip install -r requirements.txt
```

运行测试必须使用：

```bash
python -m pytest -q
```

不要使用：

```bash
pip install ...
pytest -q
```

不要使用：

```text
.venv
```

除非用户明确切换环境，否则默认使用 conda 环境 `recover_attention`。

---

# 8. 当前路线总览

本项目采用阶段式 sprint 推进。

高层路线为：

```text
Sprint 0：工程基础与 Skill 框架
Sprint 1：Baseline CoT 与推理轨迹基础
Sprint 2：Candidate Span 与 NLI 语义必要性
Sprint 3：Mask / Remove Intervention 与 Recoverability
Sprint 4：Trajectory Stability 与 Answer Stability
Sprint 5：Attention Importance Hierarchy 与 Anchor Labeling
Sprint 6：Oracle Attention Guidance
Sprint 7：Probe-Guided Attention Guidance
Sprint 8：Hallucination Reduction Evaluation
```

阶段边界：

```text
Sprint 0 只做工程与文档地基。
Sprint 1 开始建立 baseline reasoning trajectory。
Sprint 2 才进入 candidate span 与 NLI。
Sprint 5 之前不要实现 attention guidance。
Sprint 7 之前不要训练 probe。
```

---

# 9. Sprint 0：工程基础与 Skill 框架

Sprint 0 的目标是建立工程地基，不实现实验方法。

---

## 9.1 Sprint 0A：项目骨架与文档约束

目标：

```text
建立目录结构
建立 README / AGENTS / SKILL / PROGRESS
明确 Codex 边界
明确 docs/reference 是长期参考，不是执行入口
```

状态：

```text
已完成
```

---

## 9.2 Sprint 0B：jsonl 数据读写 + 样例数据 + smoke test

目标：

```text
实现 jsonl 读写
创建 questions_small.jsonl
创建 v0_nli_small.yaml
实现 smoke test
实现 data_io pytest
```

核心文件：

```text
src/recover_attention/data_io.py
configs/v0_nli_small.yaml
data/examples/questions_small.jsonl
scripts/00_smoke_test.py
tests/test_data_io.py
```

状态：

```text
已完成
```

---

## 9.3 Sprint 0C：schema 校验

目标：

```text
实现 jsonl record 的基础 schema 校验函数
覆盖 question、candidate span、ablation、NLI、masked question、recover output、recover score
```

核心文件：

```text
src/recover_attention/schemas.py
tests/test_schemas.py
```

状态：

```text
已完成
```

注意：

```text
后续需要在 Sprint 0G 中让 schema 与新主线对齐：
- span type 增加 object
- ablation type 增加 replace
- 增加 attention anchor label 相关 schema
```

---

## 9.4 Sprint 0D：prepare_data 数据准备脚本

目标：

```text
把原始 question jsonl 规范化为 data/processed/questions.jsonl
复用 data_io
复用 schemas
提供 CLI 和 pytest
```

核心文件：

```text
src/recover_attention/prepare_data.py
scripts/01_prepare_data.py
tests/test_prepare_data.py
data/processed/questions.jsonl
```

状态：

```text
已完成
```

---

## 9.5 Sprint 0E：基础验收与工程收尾

目标：

```text
检查 conda 环境
检查 requirements.txt
检查 .gitignore
重新运行 smoke test
重新运行 prepare_data
运行全部 pytest
完成 Sprint 0 工程地基验收
```

核心文件：

```text
.gitignore
requirements.txt
PROGRESS.md
```

状态：

```text
已完成
```

---

## 9.6 Sprint 0F：Skill 文档与新研究主线对齐

目标：

```text
把 docs/skill/* 从旧的 key-span / recoverability 小闭环修正为 Reasoning-Aware Attention Guidance 主线。
```

允许修改：

```text
docs/skill/SKILL.md
docs/skill/codex_tasks.md
docs/skill/experiment_guide.md
docs/skill/label_schema.md
docs/skill/method.md
docs/skill/prompts.md
PROGRESS.md
```

禁止修改：

```text
src/*
scripts/*
tests/*
configs/*
data/*
docs/reference/*
```

必须完成：

```text
1. method.md 明确项目主线是 Reasoning-Aware Attention Guidance。
2. experiment_guide.md 写清完整方法流程和当前工程实现顺序。
3. label_schema.md 增加 attention anchor 相关标签。
4. codex_tasks.md 更新 sprint 路线。
5. SKILL.md 更新子文档阅读规则。
6. prompts.md 中统一路径为 docs/skill/prompts.md。
```

状态：

```text
待执行
```

---

## 9.7 Sprint 0G：schema 与 Attention Anchor 标签体系对齐

目标：

```text
把代码 schema 与新研究主线对齐。
```

允许修改：

```text
src/recover_attention/schemas.py
tests/test_schemas.py
PROGRESS.md
```

建议修改：

```text
1. ALLOWED_SPAN_TYPES 增加 object。
2. ALLOWED_ABLATION_TYPES 增加 replace。
3. 增加 validate_attention_anchor_label_record。
4. 为 attention_importance_score / attention_anchor_label / guidance_action / guidance_strength 增加基础校验。
5. 更新 tests/test_schemas.py。
```

禁止修改：

```text
candidate_extraction.py
question_ablations.py
nli_scoring.py
recover_generation.py
recover_scoring.py
attention guidance 相关代码
```

状态：

```text
待执行
```

Sprint 0G 完成后，才能进入 Sprint 1。

---

# 10. Sprint 1：Baseline CoT 与推理轨迹基础

Sprint 1 的目标是建立 baseline reasoning trajectory 的基础文件流。

本阶段只做 baseline CoT、step parsing、trajectory manifest 和 cache 接口。

Sprint 1 不做：

```text
candidate span extraction
NLI scoring
recoverability scoring
trajectory stability scoring
attention guidance
probe training
large-scale model cache
```

---

## 10.1 Sprint 1A：Baseline CoT Schema

目标：

```text
定义 baseline_cot.jsonl 的 record schema 和测试。
```

输入：

```text
data/processed/questions.jsonl
```

输出：

```text
data/processed/baseline_cot.jsonl
```

建议字段：

```text
id
question
cot_text
steps
final_answer
generation_backend
```

要求：

```text
只定义格式和校验。
不调用模型。
不生成真实 CoT。
```

---

## 10.2 Sprint 1B：Baseline CoT Generation Stub

目标：

```text
实现 baseline CoT stub 生成器。
```

输入：

```text
data/processed/questions.jsonl
```

输出：

```text
data/processed/baseline_cot.jsonl
```

要求：

```text
先用 deterministic stub 或 fixture。
不调用真实模型。
不下载模型。
```

---

## 10.3 Sprint 1C：Step Parsing

目标：

```text
解析 baseline CoT 中的 Step 1 / Step 2 / final answer。
```

输入：

```text
data/processed/baseline_cot.jsonl
```

输出：

```text
data/processed/baseline_cot.jsonl
```

要求：

```text
只解析文本结构。
不判断正确性。
不做 trajectory scoring。
```

---

## 10.4 Sprint 1D：Trajectory Manifest Schema

目标：

```text
定义 baseline_trajectory_manifest.jsonl 的 schema。
```

输出：

```text
data/processed/baseline_trajectory_manifest.jsonl
```

建议字段：

```text
id
question
cot_id
num_steps
hidden_states_path
attentions_path
step_marker_positions
model_name
backend
```

要求：

```text
只定义 manifest schema 和测试。
不缓存真实 hidden states。
不缓存真实 attention maps。
```

---

## 10.5 Sprint 1E：Hidden / Attention Cache Backend Interface

目标：

```text
定义白盒模型前向缓存接口。
```

要求：

```text
只定义接口和 stub backend。
不下载模型。
不运行大模型。
不生成真实 .pt 缓存。
```

后续真实模型缓存必须由单独 sprint 明确允许。

---

# 11. Sprint 2：Candidate Span 与 NLI 语义必要性

Sprint 2 的目标是建立 NLI semantic necessity 分支。

Sprint 2 不做：

```text
recoverability scoring
trajectory stability scoring
answer stability scoring
attention guidance
probe training
```

---

## 11.1 Sprint 2A：Candidate Span Extraction

输入：

```text
data/processed/questions.jsonl
```

输出：

```text
data/processed/candidate_spans.jsonl
```

候选类型：

```text
number
entity
object
operation
relation
comparison
negation
condition
question_target
cyber_security_term
```

要求：

```text
先实现 rule-based baseline。
不调用模型。
```

---

## 11.2 Sprint 2B：Ablated Question Construction

输入：

```text
data/processed/candidate_spans.jsonl
```

输出：

```text
data/processed/ablated_questions.jsonl
```

ablation 类型：

```text
delete
generalize
replace
mask
```

注意：

```text
NLI 阶段优先使用 delete / generalize / replace。
Recover 阶段优先使用 mask。
```

---

## 11.3 Sprint 2C：NLI Semantic Necessity Stub

输入：

```text
data/processed/ablated_questions.jsonl
```

输出：

```text
data/processed/nli_scores.jsonl
```

要求：

```text
先实现 stub backend。
不调用真实 NLI 模型。
```

---

## 11.4 Sprint 2D：NLI Label Rule

目标：

```text
实现双向 NLI 到 semantic_necessity_label 的规则。
```

标签规则：

```text
entailment + entailment => Equivalent
entailment + non-entailment => Information Loss
non-entailment + entailment => Added Assumption
non-entailment + non-entailment => Non-equivalent
```

输出：

```text
data/processed/nli_scores.jsonl
```

---

# 12. Sprint 3：Mask / Remove Intervention 与 Recoverability

Sprint 3 的目标是建立 semantic recoverability 分支。

Sprint 3 不做：

```text
trajectory stability scoring
answer stability scoring
attention guidance
probe training
```

---

## 12.1 Sprint 3A：Masked Question Construction

输入：

```text
data/processed/candidate_spans.jsonl
```

输出：

```text
data/processed/masked_questions.jsonl
```

要求：

```text
把 candidate span 替换成 [MASK]。
只构造问题，不生成复原。
```

---

## 12.2 Sprint 3B：Question Recovery Stub

输入：

```text
data/processed/masked_questions.jsonl
```

输出：

```text
data/processed/recover_outputs.jsonl
```

要求：

```text
先用 stub backend。
不调用真实模型。
```

---

## 12.3 Sprint 3C：Recoverability Scoring

输入：

```text
data/processed/recover_outputs.jsonl
```

输出：

```text
data/processed/recover_scores.jsonl
```

标签：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

要求：

```text
只基于 recover_outputs 计算 recoverability。
不做 attention anchor labeling。
```

---

# 13. Sprint 4：Trajectory Stability 与 Answer Stability

Sprint 4 的目标是衡量 token / span intervention 对 reasoning trajectory 和 final answer 的影响。

Sprint 4 不做：

```text
attention guidance
probe training
```

---

## 13.1 Sprint 4A：Intervention Manifest

目标：

```text
定义 intervention 运行记录。
```

输出：

```text
data/processed/intervention_manifest.jsonl
```

记录字段建议：

```text
id
span_id
intervention_type
intervened_question
intervened_cot_path
hidden_states_path
attentions_path
```

---

## 13.2 Sprint 4B：Trajectory Stability Scoring

输入：

```text
data/processed/baseline_trajectory_manifest.jsonl
data/processed/intervention_manifest.jsonl
```

输出：

```text
data/processed/trajectory_stability_scores.jsonl
```

目标：

```text
衡量 intervention 前后 CoT trajectory / hidden-state geometry / step-wise states 是否偏移。
```

---

## 13.3 Sprint 4C：Answer Stability Scoring

输入：

```text
data/processed/baseline_cot.jsonl
data/processed/intervention_manifest.jsonl
```

输出：

```text
data/processed/answer_stability_scores.jsonl
```

目标：

```text
衡量 intervention 是否改变 final answer 或答案稳定性。
```

---

# 14. Sprint 5：Attention Importance Hierarchy 与 Anchor Labeling

Sprint 5 的目标是组合多种信号，得到 attention anchor。

输入信号包括：

```text
semantic_necessity_label
recoverability_label
trajectory_stability_score
answer_stability_score
raw_attention_to_span
```

输出：

```text
data/processed/attention_anchor_labels.jsonl
```

核心字段：

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

标签建议：

```text
Strong Anchor
Medium Anchor
Weak Anchor
Risky Anchor
Distractor
```

guidance_action 建议：

```text
boost
suppress
keep
review
```

注意：

```text
attention anchor labeling 不是最终实验结论。
它是 attention steering 的输入。
```

---

# 15. Sprint 6：Oracle Attention Guidance

Sprint 6 的目标是验证 attention anchor 是否真的有用。

输入：

```text
data/processed/attention_anchor_labels.jsonl
baseline model
small evaluation dataset
```

目标：

```text
使用 oracle anchor 在推理时增强关键 span attention。
```

禁止：

```text
不要训练 probe。
不要把 oracle 结果当作最终可部署方法。
```

输出：

```text
data/processed/oracle_guidance_results.jsonl
```

---

# 16. Sprint 7：Probe-Guided Attention Guidance

Sprint 7 才允许训练 probe。

目标：

```text
学习从模型内部状态或输入特征预测 attention anchor。
```

允许内容：

```text
probe dataset construction
probe training
probe evaluation
probe-guided attention guidance
```

禁止：

```text
不要在 Sprint 7 前训练 probe。
```

---

# 17. Sprint 8：Hallucination Reduction Evaluation

Sprint 8 的目标是评估完整方法是否减少幻觉并提升推理稳定性。

评估维度：

```text
answer accuracy
hallucination rate
trajectory stability
attention shift toward anchors
over-steering side effects
latency / compute overhead
```

输出：

```text
outputs/evaluation/*
reports/*
```

---

# 18. 禁止提前实现的内容

除非当前 task card 明确允许，任何 sprint 都不要提前实现：

```text
真实模型下载
真实模型调用
hidden states 大规模缓存
attention maps 大规模缓存
trajectory stability scoring
attention guidance
probe training
large-scale experiments
paper-level evaluation
```

尤其不要在 Sprint 0、Sprint 1、Sprint 2、Sprint 3 提前实现：

```text
oracle attention guidance
probe-guided attention guidance
```

---

# 19. 完成后回复格式

每轮 sprint 完成后，Codex 必须回复：

```text
1. 本次完成内容
2. 新增/修改文件；如果当前目录是 git 仓库，请报告 git diff --name-only
3. 运行命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. 遗留问题
7. 下一步建议
```

如果出现失败，必须报告：

```text
失败命令
错误信息
当前 Python 路径
已修改文件
是否需要用户确认
```

不要继续下一 sprint。
