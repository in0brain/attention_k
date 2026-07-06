# Method

本文件定义本项目的方法核心思想、术语边界和常见误解。

本文件用于防止 Codex 把项目误解成普通 hallucination classifier、普通 key span discovery，或普通 NLI 标签任务。

本文件不替代 task card。Codex 执行具体任务时，仍以当前 task card 为直接边界。

---

# 1. 方法名称

本项目主线是：

```text
Reasoning-Aware Attention Guidance
```

项目目标不是单纯检测幻觉，而是建立一种 reasoning-aware attention 方法：

```text
先发现输入 question 中真正影响推理的 token / span，
再把这些 token / span 转化为 attention anchor，
最后在模型推理时引导 attention，
从而提升推理轨迹稳定性并减少幻觉。
```

完整研究闭环是：

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

更不是：

```text
直接训练 probe 做 hallucination detection
```

这些都可能是项目中的局部工具或后续模块，但不是项目主线。

---

# 3. 核心研究问题

本项目要回答的问题是：

```text
哪些输入 token / span 有资格成为 reasoning-time attention anchor？
```

这里的 attention anchor 指：

```text
在模型推理过程中，应该被更稳定、更集中关注的输入信息单位。
```

一个 span 是否有资格成为 attention anchor，不能只看单一信号。

至少需要综合：

```text
1. Semantic Necessity
2. Semantic Recoverability
3. Trajectory Stability
4. Answer Stability
5. Raw Attention Pattern
6. Attention Steering Effect
```

---

# 4. 为什么使用 span 而不是只使用 token

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

# 5. Candidate Span

Candidate span 是候选关注对象。

它只表示：

```text
这个 span 可能影响推理。
```

它不表示：

```text
这个 span 一定重要。
```

候选类型包括：

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

Candidate span extraction 阶段只做抽取，不做重要性判断。

---

# 6. NLI Semantic Necessity

NLI semantic necessity 用于回答：

```text
某个 span 被删除、泛化或替换后，原问题和扰动问题在语义上是否仍然等价？
```

核心比较方向是：

```text
original_question → ablated_question
ablated_question → original_question
```

双向 NLI 标签包括：

```text
Equivalent
Information Loss
Added Assumption
Non-equivalent
```

其中：

```text
Equivalent:
扰动后问题语义基本等价。

Information Loss:
扰动后丢失了原问题中的精确信息或关键约束。

Added Assumption:
扰动问题引入了原问题没有的额外假设。

Non-equivalent:
扰动后问题语义明显改变。
```

NLI 的作用是：

```text
给 attention importance discovery 提供语义必要性信号。
```

NLI 不是最终目标。

NLI 不替代：

```text
semantic recoverability
trajectory stability
answer stability
attention steering
```

---

# 7. Semantic Recoverability

Semantic recoverability 用于回答：

```text
某个 span 被 mask 后，模型是否能够稳定恢复原问题语义？
```

输入示例：

```text
Tom has [MASK] apples and buys 2 more. How many apples does he have now?
```

模型任务是：

```text
复原原问题，而不是解题。
```

recoverability 标签包括：

```text
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

含义：

```text
Recoverable:
缺失信息可以从上下文稳定复原。

Partially Recoverable:
部分样本或部分信息可复原。

Non-recoverable:
缺失信息无法稳定复原。

Misleading Recovery:
模型给出了看似自信但错误或误导性的复原。
```

Recoverability 的作用是：

```text
判断一个 span 的信息是否依赖上下文可恢复，以及被遮蔽后是否容易诱导错误补全。
```

Recoverability 不是最终目标。

---

# 8. Trajectory Stability

Trajectory stability 用于回答：

```text
某个 span 被 mask / remove / replace 后，模型的 reasoning trajectory 是否发生明显偏移？
```

可能观察的对象包括：

```text
CoT steps
step marker positions
hidden-state geometry
step-wise representation trajectory
attention maps
termination behavior
```

一个 span 可能在语义上必要，但不一定显著改变 trajectory。

一个 span 也可能在 NLI 上变化不明显，但会让模型推理轨迹不稳定。

因此 trajectory stability 是独立信号。

---

# 9. Answer Stability

Answer stability 用于回答：

```text
某个 span 被 intervention 后，final answer 是否发生变化，或者答案是否变得不稳定？
```

但必须注意：

```text
答案不变不等于 span 不重要。
答案变化也不一定说明 span 是有效 attention anchor。
```

原因：

```text
1. 模型可能用错误路径得到相同答案。
2. 模型可能因为随机性改变答案。
3. 某些 span 影响推理稳定性，但不一定改变最终答案。
4. 某些 span 改变答案，但可能只是造成了破坏性扰动。
```

因此 answer stability 只能作为 attention importance 的一个辅助信号。

---

# 10. Raw Attention Pattern

Raw attention pattern 用于观察：

```text
原模型是否自然关注了 candidate span。
```

可观察内容包括：

```text
attention_to_span
attention_from_step_to_span
attention_entropy
attention concentration
layer-wise attention pattern
head-wise attention pattern
```

但必须注意：

```text
attention 高 != span 一定重要
attention 低 != span 一定不重要
```

Raw attention 只能说明模型当前如何分配注意力，不能单独证明某个 span 对推理因果重要。

---

# 11. Attention Importance Hierarchy

Attention importance hierarchy 是把多个信号组合起来，形成 span 的重要性层级。

输入信号包括：

```text
semantic_necessity_label
recoverability_label
trajectory_stability_score
answer_stability_score
raw_attention_to_span
```

输出包括：

```text
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
evidence
```

其中：

```text
attention_importance_score:
综合多个信号得到的重要性分数。

attention_anchor_label:
span 是否应成为 attention anchor，以及属于哪类 anchor。

guidance_action:
后续 attention steering 对该 span 应采取的动作。

guidance_strength:
后续 attention steering 对该 span 的增强或抑制强度。
```

---

# 12. Attention Anchor Label

attention_anchor_label 建议包括：

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

# 13. Guidance Action

guidance_action 建议包括：

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

guidance_action 不应在早期阶段直接用于模型干预。

必须先完成：

```text
attention anchor labeling
small-scale validation
oracle attention guidance
```

之后再进入 probe-guided attention guidance。

---

# 14. Oracle Attention Guidance

Oracle attention guidance 用于验证：

```text
如果我们已经知道哪些 span 是 attention anchor，
那么增强这些 span 的 attention 是否真的能改善推理稳定性？
```

它的作用是：

```text
验证 attention anchor label 是否有用。
```

它不是最终可部署方法。

它不训练 probe。

它不代表真实系统已经能自动发现 anchor。

---

# 15. Probe-Guided Attention Guidance

Probe-guided attention guidance 是后续阶段。

它用于回答：

```text
能否从模型内部状态、输入特征或早期推理信号中预测 attention anchor，
并在推理时自动进行 attention guidance？
```

只有在以下内容稳定后，才进入 probe 阶段：

```text
candidate span extraction
NLI semantic necessity
semantic recoverability
trajectory stability
answer stability
attention anchor labels
oracle attention guidance
```

不要在 Sprint 7 之前训练 probe。

---

# 16. 方法中的信号关系

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
再由 attention_importance_score 和规则得到 attention_anchor_label。
最后才转化为 guidance_action 和 guidance_strength。
```

---

# 17. 常见误解

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

# 18. 当前早期策略

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

# 19. 禁止提前实现的内容

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

# 20. 本文件与其他文档的关系

本文件说明方法主线和术语边界。

其他 Skill 子文档职责：

```text
docs/reasoning-aware-attention-guidance/SKILL.md:
Skill 总入口和阅读规则。

docs/reasoning-aware-attention-guidance/codex_tasks.md:
Sprint 路线、task card 规范和执行边界。

docs/reasoning-aware-attention-guidance/experiment_guide.md:
实验流程、阶段输入输出和运行边界。

docs/reasoning-aware-attention-guidance/label_schema.md:
jsonl schema、字段含义和标签枚举。

docs/reasoning-aware-attention-guidance/prompts.md:
可复用 Codex 提示词模板。
```

如果发生冲突，优先级为：

```text
当前用户消息
> 当前 task card
> AGENTS.md
> docs/reasoning-aware-attention-guidance/SKILL.md
> docs/reasoning-aware-attention-guidance/*.md
> docs/reasoning-aware-attention-guidance/prompts.md
> docs/reference/*
```
