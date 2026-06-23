# Codex Tasks

本文件只负责 Sprint 路线、task card 规范、Preflight 规则、冲突优先级、阶段边界和禁止提前实现内容。

它不是 schema 文档，不维护字段定义、枚举值或完整 jsonl 示例。schema 细节统一放在 `docs/skill/label_schema.md`。

---

## 1. 项目主线

当前项目主线是：

```text
Reasoning-Aware Attention Guidance
```

项目不是普通的 hallucination classifier，也不是单纯的 key span discovery。

高层研究链路是：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

NLI、recoverability、trajectory stability、answer stability 和 raw attention pattern 都只是 attention importance discovery 的信号来源。

最终目标是：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

---

## 2. Task Card 规范

每张 task card 必须明确：

```text
1. 目标
2. 允许修改的文件
3. 禁止修改的文件
4. 输入文件
5. 输出文件
6. 实现要求
7. 验收标准
8. 运行命令
9. 禁止事项
10. PROGRESS.md 更新要求
```

每轮只执行一张 task card。

不要根据路线图自动进入下一张 task card。

---

## 3. Preflight 规则

每次修改文件前必须先输出 Preflight，并等待用户确认。

Preflight 至少包括：

```text
1. 已读取文件列表
2. 本次允许修改文件
3. 本次禁止修改文件
4. 本次必须运行命令
5. 是否需要读取 docs/reference/*
6. 是否发现冲突
```

如果需要修改允许范围之外的文件，先停止并询问用户。

---

## 4. 冲突优先级

如果指令冲突，优先级为：

```text
当前用户消息
> 当前 task card
> AGENTS.md
> PROGRESS.md
> docs/skill/SKILL.md
> docs/skill/*.md
> docs/skill/prompts.md
> docs/reference/*
```

发现冲突时必须：

```text
1. 在 Preflight 中报告。
2. 说明采用哪条规则。
3. 在最终回复的“遗留问题”中再次记录。
```

---

## 5. 阶段边界

当前路线为：

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

阶段规则：

```text
Sprint 0 只做工程基础、Skill 框架和 schema 对齐。
Sprint 1 可以使用 stub / fixture，不默认调用真实模型。
Sprint 2 可以先使用 rule-based span extraction 和 NLI stub。
Sprint 3 可以先使用 recovery stub。
Sprint 5 之前不要实现 attention guidance。
Sprint 6 之前不要执行 oracle attention guidance。
Sprint 7 之前不要训练 probe。
```

---

## 6. 禁止提前实现

除非当前 task card 明确允许，否则不要做：

```text
真实模型调用
真实模型下载
外部 API 调用
大规模数据处理
真实 hidden states 大规模缓存
真实 attention maps 大规模缓存
trajectory stability scoring
attention guidance
probe training
paper-level evaluation
```

不要把 NLI、recoverability 或 key span discovery 当作最终目标。

不要声称减少幻觉，除非已经完成对应评估。

---

## 7. 文档职责

Skill 文档分工如下：

```text
docs/skill/SKILL.md:
Skill 总入口和路由规则。

docs/skill/codex_tasks.md:
Sprint 路线、task card 规范和执行边界。

docs/skill/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/skill/method.md:
方法概念、信号关系和常见误解。

docs/skill/label_schema.md:
jsonl schema、字段、枚举和标签规则。

docs/skill/prompts.md:
可复用 Codex 提示词模板。
```
