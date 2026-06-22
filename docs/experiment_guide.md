# Method

本文件定义本项目的方法核心思想、术语边界和信号组合原则。

本文件不说明目录结构、运行命令、阶段输入输出。
这些内容由 `docs/skill/experiment_guide.md` 负责。

本文件不替代 task card。Codex 执行具体任务时，仍以当前 task card 为直接边界。

---

# 1. 方法主线

本项目主线是：

```text
Reasoning-Aware Attention Guidance
```

核心思想是：

```text
先发现输入 question 中真正影响推理的 token / span，
再把这些 token / span 转化为 attention anchor，
最后在模型推理时引导 attention，
从而提升推理轨迹稳定性并减少幻觉。
```

本项目的研究闭环是：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

---

# 2. 本项目不是什么

本项目不是：

```text
hallucination classifier
```

也不是：

```text
普通 key span discovery
```

也不是：

```text
NLI classification task
```

也不是：

```text
只做 recoverability scoring
```

也不是：

```text
直接训练 probe 做 hallucination detection
```

这些都可能是局部工具或后续模块，但不是项目主线。

---

# 3. 核心问题

本项目要回答的问题是：

```text
哪些输入 token / span 有资格成为 reasoning-time attention anchor？
```

这里的 `attention anchor` 指：

```text
在模型推理过程中，应该被更稳定、更集中关注的输入信息单位。
```

一个 span 是否有资格成为 attention anchor，不能只看单一信号。

需要综合：

```text
Semantic Necessity
Semantic Recoverability
Trajectory Stability
Answer Stability
Raw Attention Pattern
Attention Steering Effect
```

---

# 4. 为什么使用 span

本项目使用：

```text
span
```

而不是只使用：

```text
token
```

原因：

```text
1. 关键信息可能是短语，而不是单个 token。
2. 数字、实体、对象、关系、条件都可能由多个 token 组成。
3. NLI ablation 和 mask-recover 更自然地作用于 span。
4. attention guidance 最终可以把 span 内 token 作为一个 anchor group 处理。
```

例如：

```text
remote code execution
unsafe deserialization
How many
10% cheaper
```

这些都更适合作为 span，而不是拆成孤立 token。

---

# 5. 各信号的角色

## 5.1 Semantic Necessity

Semantic Necessity 通过 NLI 判断：

```text
某个 span 被删除、泛化或替换后，原问题和扰动问题在语义上是否仍然等价。
```

它回答的是：

```text
这个 span 是否承载了问题语义中的必要信息？
```

它不回答：

```text
模型是否能复原这个 span。
这个 span 是否会改变 hidden-state trajectory。
这个 span 是否应该被直接增强 attention。
```

NLI 只是 attention importance discovery 的一个辅助信号。

---

## 5.2 Semantic Recoverability

Semantic Recoverability 判断：

```text
某个 span 被 mask 后，模型是否能够稳定恢复原问题语义。
```

它回答的是：

```text
缺失信息是否能从上下文中稳定恢复？
模型是否会产生误导性复原？
```

它不等价于语义必要性。

一个 span 可以语义必要，但仍然容易从上下文恢复。
一个 span 也可以不可恢复，但未必是有效 attention anchor。

---

## 5.3 Trajectory Stability

Trajectory Stability 判断：

```text
某个 span 被干预后，模型的 reasoning trajectory 是否发生明显偏移。
```

它关注：

```text
CoT steps
hidden-state geometry
step-wise trajectory
attention maps
termination behavior
```

它是 reasoning-aware 的核心信号之一。

---

## 5.4 Answer Stability

Answer Stability 判断：

```text
span intervention 是否改变 final answer 或答案稳定性。
```

但必须注意：

```text
答案不变不等于 span 不重要。
答案变化也不一定说明 span 是有效 anchor。
```

因为模型可能用错误路径得到相同答案，也可能因为随机性改变答案。

---

## 5.5 Raw Attention Pattern

Raw Attention Pattern 观察：

```text
原模型是否自然关注了 candidate span。
```

但必须注意：

```text
attention 高 != span 一定重要
attention 低 != span 一定不重要
```

Raw attention 只能说明模型当前如何分配注意力，不能单独证明因果重要性。

---

## 5.6 Attention Steering Effect

Attention Steering Effect 判断：

```text
当增强或抑制某些 anchor 的 attention 后，模型推理是否更稳定，幻觉是否减少。
```

它是最终验证信号。

没有 attention steering 实验，就不能声称 attention anchor 已经被验证有效。

---

# 6. 信号组合原则

不要把任何单一信号当成最终判断。

错误做法：

```text
Information Loss => Strong Anchor
Non-recoverable => Strong Anchor
answer_changed => Strong Anchor
high_attention => Strong Anchor
```

正确做法：

```text
多个信号共同构成 evidence。
再由 evidence 形成 attention_importance_score。
再由 attention_importance_score 得到 attention_anchor_label。
最后才转化为 guidance_action 和 guidance_strength。
```

---

# 7. Attention Anchor Label

`attention_anchor_label` 是给 attention steering 使用的中间标签。

建议标签包括：

```text
Strong Anchor
Medium Anchor
Weak Anchor
Risky Anchor
Distractor
```

含义：

```text
Strong Anchor:
多种信号一致表明该 span 对推理重要，后续倾向 boost。

Medium Anchor:
部分信号表明该 span 重要，可以适度增强。

Weak Anchor:
信号较弱或不稳定，通常保持观察。

Risky Anchor:
该 span 可能诱导错误复原、错误轨迹或不稳定推理，需要谨慎处理。

Distractor:
该 span 可能吸引无效 attention，后续可能 suppress。
```

这些标签不是最终实验结论，而是 attention steering 的输入。

---

# 8. Guidance Action

`guidance_action` 表示后续 attention steering 应该如何处理该 span。

建议动作包括：

```text
boost
suppress
keep
review
```

含义：

```text
boost:
增强该 span 或 span group 的 attention。

suppress:
抑制可能误导推理的 span。

keep:
不主动干预。

review:
证据不足，需要人工或后续实验检查。
```

---

# 9. 常见误解

必须避免以下误解：

```text
语义必要 != 不可复原
可复原 != 不重要
不可复原 != 一定重要
语义等价 != 答案不变
答案不变 != span 不关键
mask 后猜对 != span 不关键
attention 高 != span 一定重要
attention 低 != span 一定不重要
oracle guidance != 最终可部署方法
probe != hallucination classifier
```

---

# 10. 早期实现策略

早期实现遵循：

```text
stub-first
small-data first
jsonl-first
test-first
no model unless explicitly allowed
```

也就是说：

```text
先让文件流、schema、标签流跑通。
再接入真实模型。
再做小规模验证。
再做 trajectory / attention 分析。
最后才考虑 probe-guided attention guidance。
```

---

# 11. 禁止提前实现的内容

在 task card 明确允许前，不要实现：

```text
真实 NLI 模型调用
真实 recovery 模型调用
真实 baseline CoT 模型调用
真实 hidden states 大规模缓存
真实 attention maps 大规模缓存
trajectory stability scoring
attention guidance
probe training
large-scale experiments
paper-level evaluation
```

尤其不要在 Sprint 0、Sprint 1、Sprint 2、Sprint 3 中提前实现：

```text
oracle attention guidance
probe-guided attention guidance
```

---

# 12. 本文件与其他文档的关系

本文件说明方法主线和术语边界。

其他 Skill 子文档职责：

```text
docs/skill/SKILL.md:
Skill 总入口和阅读规则。

docs/skill/codex_tasks.md:
Sprint 路线、task card 规范和执行边界。

docs/skill/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/skill/label_schema.md:
jsonl schema、字段含义和标签枚举。

docs/skill/prompts.md:
可复用 Codex 提示词模板。
```

如果发生冲突，优先级为：

```text
当前用户消息
> 当前 task card
> AGENTS.md
> docs/skill/SKILL.md
> docs/skill/*.md
> docs/skill/prompts.md
> docs/reference/*
```
