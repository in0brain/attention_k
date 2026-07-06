# Sprint 0F：文档主线对齐与下一阶段任务准备

## 1. 目标

当前项目已经完成 Sprint 0E：基础验收与工程收尾。

本 sprint 的目标是把项目文档从旧主线：

```text
Recoverability-Guided Attention Allocation
```

对齐到新主线：

```text
Reasoning-Aware Attention Guidance
```

本 sprint 只修改文档，不修改代码，不进入 schema 实现，不开始 Sprint 1。

---

## 2. 背景

旧文档中仍然把项目描述为：

```text
candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring
```

这只是早期小闭环，不再是完整项目主线。

新版项目主线应为：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

最终目标不是普通的：

```text
hallucination_label
```

而是：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

NLI、recoverability、trajectory stability、answer stability 和 raw attention pattern 都只是 attention importance discovery 的信号来源。

---

## 3. 开始前必须阅读

开始前必须读取：

```text
AGENTS.md
README.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
```

如果某个文件不存在，必须在 Preflight 中报告。

不要读取：

```text
docs/reference/*
```

除非发现上述文件完全无法判断当前主线，并先向用户说明原因。

---

## 4. 允许修改

本 sprint 允许修改：

```text
README.md
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
```

---

## 5. 禁止修改

本 sprint 禁止修改：

```text
src/*
scripts/*
tests/*
configs/*
data/*
outputs/*
cache/*
requirements.txt
pyproject.toml
.gitignore
docs/reference/*
docs/codex_tasks/*
```

注意：

```text
本 sprint 不创建 Sprint 0G task card。
Sprint 0G task card 应由用户手动创建，或由单独任务明确创建。
```

---

## 6. Preflight 要求

修改文件前必须先输出 Preflight，并暂停等待用户确认。

Preflight 必须包括：

```text
1. 已读取文件列表
2. 缺失文件列表
3. 本次允许修改文件
4. 本次禁止修改文件
5. 是否发现旧主线残留
6. 是否发现路径冲突，例如 docs/prompts.md 与 docs/reasoning-aware-attention-guidance/prompts.md 混用
7. 是否需要读取 docs/reference/*
8. 本次计划运行的检查命令
```

用户确认后才能修改文件。

---

## 7. 修改要求

### 7.1 README.md

把 README.md 从旧项目名：

```text
Recoverability-Guided Attention Allocation
```

改为：

```text
Reasoning-Aware Attention Guidance
```

README.md 应说明：

```text
1. 项目不是普通 hallucination classifier。
2. 项目不是单纯 key span discovery。
3. NLI 和 recoverability 只是辅助信号。
4. 最终目标是 attention anchor 和 attention guidance。
5. 当前仍处于工程基础与文档/schema 对齐阶段。
6. 不要直接跳到 attention guidance、probe training 或大规模实验。
```

README.md 中的最小运行命令必须使用：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

不要使用裸命令：

```bash
pip install -r requirements.txt
pytest -q
```

---

### 7.2 AGENTS.md

把 AGENTS.md 的项目定位改为：

```text
Reasoning-Aware Attention Guidance
```

AGENTS.md 应明确：

```text
1. Codex 每轮任务必须小步执行。
2. 修改前必须 Preflight。
3. 当前 task card 是直接执行边界。
4. PROGRESS.md 每轮 sprint 后必须更新。
5. 不要默认读取 docs/reference/*。
6. 不要提前实现真实模型调用、hidden states 大规模缓存、attention guidance、probe training。
7. 不要把 NLI、recoverability 或 key span discovery 当成最终目标。
```

AGENTS.md 的文档路径应统一到：

```text
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/prompts.md
docs/codex_tasks/*.md
docs/reference/*
```

不要继续使用旧路径：

```text
docs/method.md
docs/experiment_guide.md
docs/label_schema.md
docs/prompts.md
docs/codex_tasks.md
```

除非是作为历史残留在问题列表中报告。

---

### 7.3 docs/reasoning-aware-attention-guidance/SKILL.md

`docs/reasoning-aware-attention-guidance/SKILL.md` 应作为 Skill 总入口和路由器。

它不应重复完整 schema。

它应说明以下子文档职责：

```text
docs/reasoning-aware-attention-guidance/codex_tasks.md:
Sprint 路线、task card 规范和执行边界。

docs/reasoning-aware-attention-guidance/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/reasoning-aware-attention-guidance/method.md:
方法概念、信号关系和常见误解。

docs/reasoning-aware-attention-guidance/label_schema.md:
jsonl schema、字段、枚举和标签规则。

docs/reasoning-aware-attention-guidance/prompts.md:
可复用 Codex 提示词模板。
```

---

### 7.4 docs/reasoning-aware-attention-guidance/codex_tasks.md

`docs/reasoning-aware-attention-guidance/codex_tasks.md` 应只负责：

```text
Sprint 路线
task card 规范
Preflight 规则
冲突优先级
阶段边界
禁止提前实现内容
```

它不应重复完整 schema。

路线应至少包括：

```text
Sprint 0：工程基础与 Skill 框架
Sprint 0F：文档主线对齐
Sprint 0G：schema 与 Attention Anchor 标签体系对齐
Sprint 1：Baseline CoT 与推理轨迹基础
Sprint 2：Candidate Span 与 NLI 语义必要性
Sprint 3：Mask / Remove Intervention 与 Semantic Recoverability
Sprint 4：Trajectory Stability 与 Answer Stability
Sprint 5：Attention Importance Hierarchy 与 Anchor Labeling
Sprint 6：Oracle Attention Guidance
Sprint 7：Probe-Guided Attention Guidance
Sprint 8：Hallucination Reduction Evaluation
```

---

### 7.5 docs/reasoning-aware-attention-guidance/experiment_guide.md

`experiment_guide.md` 应只负责实验流程，不负责 sprint task card 细节。

它应包含完整目标流程：

```text
question
→ baseline CoT generation
→ hidden states / attention maps cache
→ candidate span extraction
→ NLI semantic necessity scoring
→ mask / remove intervention
→ semantic recoverability scoring
→ trajectory stability scoring
→ answer stability scoring
→ attention anchor labeling
→ oracle attention guidance
→ probe-guided attention guidance
→ hallucination reduction evaluation
```

同时说明：

```text
1. 当前阶段 stub-first。
2. 每个阶段只产出自己的文件。
3. 不要提前实现后续阶段。
4. hidden states / attention maps 只能在 task card 明确允许时缓存。
```

---

### 7.6 docs/reasoning-aware-attention-guidance/method.md

`method.md` 应只负责方法概念，不负责目录结构、命令、阶段输入输出。

它应明确：

```text
1. NLI semantic necessity 的角色。
2. semantic recoverability 的角色。
3. trajectory stability 的角色。
4. answer stability 的角色。
5. raw attention pattern 的角色。
6. attention steering effect 的角色。
7. 多信号组合后才得到 attention_anchor_label。
```

避免重复 `experiment_guide.md` 的完整 pipeline。

---

### 7.7 docs/reasoning-aware-attention-guidance/label_schema.md

`label_schema.md` 应只负责字段、枚举、jsonl schema。

它必须包含：

```text
object span type
replace ablation type
attention_anchor_label
guidance_action
attention_anchor_labels.jsonl
trajectory_stability_scores.jsonl
answer_stability_scores.jsonl
```

但不要要求当前 sprint 修改代码 schema。

代码 schema 同步留给 Sprint 0G。

---

### 7.8 docs/reasoning-aware-attention-guidance/prompts.md

检查并修正路径引用。

统一使用：

```text
docs/reasoning-aware-attention-guidance/prompts.md
```

不要混用：

```text
docs/prompts.md
```

---

### 7.9 PROGRESS.md

更新标题和当前边界。

把旧标题：

```text
实验进度记录：Recoverability-Guided Attention Allocation
```

改为：

```text
实验进度记录：Reasoning-Aware Attention Guidance
```

把旧的“当前实验边界”改成：

```text
当前项目已经完成 Sprint 0E：基础验收与工程收尾。

当前处于 Sprint 0F：文档主线对齐阶段。

项目主线已经从旧的 Recoverability-Guided Attention Allocation 调整为 Reasoning-Aware Attention Guidance。

下一步不是旧的 Sprint 1A candidate span extraction，而是：

Sprint 0G：schema 与 Attention Anchor 标签体系对齐。
```

新增记录：

```text
### Sprint 0F：文档主线对齐

已完成内容：
- 将项目主线对齐为 Reasoning-Aware Attention Guidance。
- 更新 README.md 和 AGENTS.md 中的项目定位。
- 更新 docs/reasoning-aware-attention-guidance/* 的职责分工和阶段边界。
- 确认 NLI、recoverability、trajectory stability、answer stability、raw attention pattern 都只是 attention importance discovery 的信号来源。
- 确认最终目标是 attention_importance_score、attention_anchor_label、guidance_action、guidance_strength。
- 确认下一步应进入 Sprint 0G schema 对齐，而不是直接进入旧的 candidate span extraction。

新增或修改文件：
- README.md
- AGENTS.md
- PROGRESS.md
- docs/reasoning-aware-attention-guidance/SKILL.md
- docs/reasoning-aware-attention-guidance/codex_tasks.md
- docs/reasoning-aware-attention-guidance/experiment_guide.md
- docs/reasoning-aware-attention-guidance/method.md
- docs/reasoning-aware-attention-guidance/label_schema.md
- docs/reasoning-aware-attention-guidance/prompts.md

检查结果：
- 未修改代码。
- 未修改测试。
- 未开始 Sprint 0G。
- 未调用真实模型。
- 未实现 attention guidance 或 probe training。

下一步建议：
- Sprint 0G：schema 与 Attention Anchor 标签体系对齐。
```

---

## 8. 必须运行的检查命令

本 sprint 不修改代码，因此不要求运行 pytest。

必须运行：

```bash
git diff --name-only
git status --short
```

如果不是 git 仓库，报告未检测到 git 仓库即可。

可以运行但不是必须：

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

---

## 9. 验收标准

本 sprint 完成后应满足：

```text
1. README.md 不再以 Recoverability-Guided Attention Allocation 作为当前主线。
2. AGENTS.md 不再以 Recoverability-Guided Attention Allocation 作为当前主线。
3. docs/reasoning-aware-attention-guidance/SKILL.md 是总入口，不重复完整 schema。
4. docs/reasoning-aware-attention-guidance/codex_tasks.md 包含 Sprint 0F 和 Sprint 0G。
5. docs/reasoning-aware-attention-guidance/experiment_guide.md 负责实验流程。
6. docs/reasoning-aware-attention-guidance/method.md 负责方法概念，和 experiment_guide.md 不大段重复。
7. docs/reasoning-aware-attention-guidance/label_schema.md 包含 object、replace、attention anchor label、guidance action 等新版字段。
8. docs/reasoning-aware-attention-guidance/prompts.md 不混用 docs/prompts.md 路径。
9. PROGRESS.md 下一步建议是 Sprint 0G，而不是旧的 Sprint 1A candidate span extraction。
10. 未修改 src、scripts、tests、configs、data。
```

---

## 10. 完成后回复格式

完成后按以下格式回复：

```text
1. 本次完成内容
2. 新增/修改文件
3. 检查命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. 遗留问题
7. 下一步建议
```

不要自动开始 Sprint 0G。
