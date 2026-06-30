# Sprint 1R：Human Review Consolidation & Known Issue Freeze

## 目标

将已经完成的 Sprint 1Q 人工审核结果固化为后续可用资产，生成 summary、known issues 和 Sprint 2A hidden-state 输入 manifest。

本 sprint 不再从 Markdown 同步人工标签到 JSONL / report JSON。人工审核结果已经在前一步完成同步，因此 1R 只做轻量校验和后续产物生成，避免重复解析 Markdown、重复写入结构化文件或浪费 Codex token。

---

## 背景

Sprint 1Q 已完成人工审核，并已经生成了结构化版本：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
outputs/logs/sprint_1Q_real_signal_quality_review/upgraded_downstream_report_with_human_fields.json
```

人工填写的 Markdown 主文件为：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_sheet.md
```

该 Markdown 仅作为人工可读审查记录保留，不再作为 1R 的主要机器输入。

---

## 输入文件

1R 的主要输入是已经同步完成的结构化文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
outputs/logs/sprint_1Q_real_signal_quality_review/upgraded_downstream_report_with_human_fields.json
```

人工可读参考文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_sheet.md
```

注意：不要在 1R 中重新解析 `sprint_1Q_human_review_sheet.md` 来同步字段。该文件只用于人工查看和追溯。

---

## 输出文件

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_summary.json
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_known_issues.md
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

---

## 需要完成的工作

### 1. 轻量校验结构化人工标签

读取：

```text
sprint_1Q_human_review_labels_template.jsonl
upgraded_downstream_report_with_human_fields.json
```

检查：

```text
1. JSONL 可以正常逐行读取。
2. report JSON 可以正常读取。
3. JSONL 中所有 records 都有 masked_id。
4. reviewed cases 的 human fields 不为空。
5. JSONL 与 report JSON 中 changed_cases 的 masked_id 能对应。
6. report JSON 中 changed_cases 的 human fields 与 JSONL 保持一致。
```

如果发现不一致，只输出 warning，不要自动从 Markdown 重建。

---

### 2. 生成人工审核 summary

生成：

```text
sprint_1Q_human_review_summary.json
```

至少包含：

```text
reviewed_count
unreviewed_count
human_recoverability_label_counts
human_attention_anchor_label_counts
human_error_type_counts
probe_usage_counts
auto_vs_human_recoverability_disagreement_count
auto_vs_human_attention_anchor_disagreement_count
fragment_recovery_count
wrong_numeric_recovery_count
generic_recovery_count
misleading_entity_or_unit_count
```

summary 应只基于结构化 JSONL / report JSON 生成，不再解析 Markdown。

---

### 3. 生成 known issues 文档

生成：

```text
sprint_1Q_known_issues.md
```

记录 Sprint 1Q 已确认但本 sprint 不实现的问题。

必须包含以下 deferred issues：

#### Issue 1：LLM recovery 任务错位

当前 recovery backend 有时把 full-question recovery 理解成 fill-in-the-blank，只输出：

```text
How many
pages
gave
0
```

但 pipeline 期望的是完整 `recovered_question`。

#### Issue 2：需要 full-question recovery validation

后续应加入 full-question gate：

```text
if recovered output is only a fragment:
    recoverability_label = Non-recoverable
    error_type = fragment_recovery
```

#### Issue 3：需要 span-aware numeric recovery check

后续应加入数字一致性检查：

```text
if masked span is a number
and recovered question contains a different concrete number:
    recoverability_label = Misleading Recovery
    error_type = wrong_numeric_recovery
```

#### Issue 4：需要 generic critical-number recovery check

后续应识别：

```text
7 -> some
15 -> some
```

这类情况应判为：

```text
Non-recoverable
generic_recovery
```

而不是 `Misleading Recovery`。

#### Issue 5：需要 entity / unit consistency check

后续应识别：

```text
pages -> books
pencils -> pens
apples -> bananas
```

这类实体/单位漂移应判为：

```text
Misleading Recovery
misleading_entity_or_unit
```

#### Issue 6：需要 unit/group mask budget control

single unit 和 group unit 都有价值，但后续大规模实验需要控制 masked question 数量。

记录建议：

```text
- group unit 用于检测全局实体/数字一致性，应保留。
- single unit 用于局部定位，但低价值 single unit 应限量采样。
- 每个 original question 后续可设置 masked question budget。
- group 覆盖后的重复 single unit 可以降优先级。
```

---

### 4. 生成 Sprint 2A manifest

生成：

```text
sprint_1Q_to_2A_manifest.jsonl
```

每行一个 reviewed case，包含 Sprint 2A hidden-state 阶段需要的最小信息：

```json
{
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",
  "original_question": "...",
  "masked_question": "...",
  "recovered_questions": ["..."],
  "human_recoverability_label": "...",
  "human_attention_anchor_label": "...",
  "human_semantic_role": "...",
  "human_guidance_priority": "...",
  "human_error_type": "...",
  "probe_usage": "..."
}
```

该 manifest 是 Sprint 2A Hidden State Cache Baseline 的主要输入。

---

## 明确不做

本 sprint 不做以下工作：

```text
1. 不解析 sprint_1Q_human_review_sheet.md 来同步字段。
2. 不从 Markdown 重建 JSONL。
3. 不从 Markdown 重建 report JSON。
4. 不覆盖已经同步好的 human fields。
5. 不重跑 Ollama。
6. 不重跑 NLI。
7. 不修改 recovery prompt。
8. 不实现 recovery_scoring_v2。
9. 不实现 full-question recovery validator。
10. 不实现 span-aware numeric scorer。
11. 不实现 entity/unit consistency scorer。
12. 不实现 unit/group mask budget selector。
13. 不重建 masked_questions。
14. 不训练 probe。
15. 不缓存 hidden states。
```

---

## Deferred / 未完成事项

以下问题已在 Sprint 1Q 中确认，但本 sprint 只记录，不实现。

### Deferred A：Full-question Recovery Alignment

问题：

```text
LLM recovery backend 有时只输出 masked span，而不是完整 recovered question。
```

后续方向：

```text
- 改造 recovery prompt。
- 要求模型输出完整问题。
- 可考虑 JSON 输出格式。
- 可考虑同时支持 recovered_span 与 recovered_question。
```

建议执行时机：

```text
Sprint 2B hidden-state representation analysis 完成后，
如果 fragment_recovery 被证明主要是 generation-format 问题，
再开启单独 sprint 处理。
```

---

### Deferred B：Span-aware Recovery Validation

问题：

```text
当前 question-level NLI judge 不直接验证 masked span 是否恢复正确。
```

后续方向：

```text
- 增加 full-question validity gate。
- 增加 numeric consistency check。
- 增加 generic critical-number check。
- 增加 entity/unit consistency check。
```

建议执行时机：

```text
在 Sprint 2B 完成后，
如果需要构建更干净的大规模 probe dataset，
应在 Sprint 2C 前处理。
```

---

### Deferred C：Unit / Group Mask Budgeting

问题：

```text
single unit 和 group unit 都生成时，masked questions 数量可能膨胀。
```

后续方向：

```text
- 保留 group unit，用于检测全局一致性。
- single unit 做局部定位，但需要采样。
- 每个 original question 设置 mask budget。
- group 覆盖后的低价值 single unit 降优先级。
```

建议执行时机：

```text
小规模 hidden-state analysis 完成后，
在扩展到更大样本或训练 probe 前执行。
```

---

## 验收标准

1. `sprint_1Q_human_review_labels_template.jsonl` 可以正常读取。
2. `upgraded_downstream_report_with_human_fields.json` 可以正常读取。
3. JSONL 与 report JSON 中 reviewed cases 的 human fields 一致。
4. 不重新解析 `sprint_1Q_human_review_sheet.md`。
5. 不覆盖已经同步好的 human fields。
6. `sprint_1Q_human_review_summary.json` 能统计人工标签分布和错误类型分布。
7. `sprint_1Q_known_issues.md` 明确记录 full-question recovery、span-aware validation、unit/group budget 的 deferred issues。
8. `sprint_1Q_to_2A_manifest.jsonl` 可直接作为 Sprint 2A 输入。
9. 本 sprint 不改变任何 1P / 1O / 1N 自动输出。
10. 本 sprint 不运行模型、不重跑 NLI、不训练 probe、不缓存 hidden states。
11. `PROGRESS.md` 更新当前阶段为 Sprint 1R completed，并将下一步标记为 Sprint 2A Hidden State Cache Baseline。

---

## PROGRESS.md 更新建议

```text
当前阶段：Sprint 1R 已完成：Human Review Consolidation & Known Issue Freeze。

已完成：
- 校验 Sprint 1Q 人工审核结构化文件。
- 生成 human review summary。
- 生成 known issues 文档。
- 生成 Sprint 2A hidden-state manifest。
- 明确记录 full-question recovery alignment、span-aware recovery validation、unit/group mask budgeting 为 deferred issues。

特别说明：
- 本 sprint 未重新解析 sprint_1Q_human_review_sheet.md。
- 本 sprint 未重新同步 human fields，因为 JSONL / report JSON 已在 Sprint 1Q 后完成同步。
- 本 sprint 只基于现有结构化人工标签生成 summary、known issues 和 2A manifest。

未完成 / deferred：
- 未重跑 Ollama。
- 未重跑 NLI。
- 未实现 recovery prompt 改造。
- 未实现 span-aware recovery scorer。
- 未实现 unit/group mask budget selector。
- 未训练 probe。
- 未缓存 hidden states。

下一步：Sprint 2A：Hidden State Cache Baseline。
```
