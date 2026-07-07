# Sprint 2I: Targeted Attention-Map Feature Cache and Probe

## 背景

Sprint 2H-D 已完成 ordinal calibration and budget-aware gate redesign。

当前结论：

```text
hidden_pre_recovery_enriched 在 macro_f1 上稳定超过 surface baseline；
但是 ordinal ranking / budget-aware bucket-3 selection 不够稳健；
2H-D gate = 5/8；
ready_for_2000_rerun = False。
```

因此不要扩大到 2000，不要进入 Sprint 3A。

下一步目标是引入 attention-level pre-recovery features，判断 attention maps 是否提供比 hidden-state-only 更稳定的 instance-level fragility ranking signal。

---

## 本轮目标

新增 targeted attention-map cache，只在 500-case diagnostic subset 上运行。

构造新的 gate-eligible feature set：

```text
attention_pre_recovery
```

以及融合特征：

```text
hidden_plus_attention_pre_recovery
```

目标是回答：

```text
attention-level features 是否能提升 ordinal ranking、budget-aware bucket-3 selection、top-k solution-path coverage，并控制 off-path distractor budget？
```

---

## 严格边界

允许：

```text
读取 2H-B / 2H-C / 2H-D 的 500-case subset 与 labels；
读取 2G 原始问题与 masked question；
新增 attention cache；
新增 attention feature extraction；
新增 probe / calibration evaluation；
新增测试和报告。
```

禁止：

```text
不要重跑 recovery；
不要使用 recovered-channel；
不要使用 recovered question hidden states；
不要使用 recovered question attention；
不要扩大到 2000；
不要进入 Sprint 3A；
不要执行 attention steering；
不要把 gold solution path / drift / bucket / risk_strength 当输入 feature；
不要覆盖 2G / 2H-B / 2H-C / 2H-D 输出。
```

---

## 任务 1：确定 500-case 输入

复用 2H-C / 2H-D 的同一个 500-case subset。

优先读取：

```text
outputs/logs/sprint_2H_feature_enrichment_500/pre_recovery_feature_dataset.jsonl
outputs/logs/sprint_2H_instance_signal_500/risk_strength_dataset.jsonl
```

确保每条样本包含：

```text
masked_id
source_question_id
span_type
span_text
question
masked_question
fragility_bucket
risk_strength
```

注意：`fragility_bucket` 和 `risk_strength` 只能作为监督标签，不能作为输入 feature。

---

## 任务 2：缓存 original + masked attention maps

新增脚本，例如：

```text
scripts/sprint_2I_attention_cache.py
```

新增输出目录：

```text
outputs/logs/sprint_2I_attention_cache_500/
```

只缓存两种输入：

```text
original question
masked question
```

禁止缓存：

```text
recovered question
```

建议缓存层：

```text
与 2G hidden-state cache 对齐：0 / 8 / 16 / 24 / 27
```

如果模型或框架难以只取这些层的 attention，可以先全层 forward 后只保存这些层。

每条样本至少保存：

```text
input_type: original | masked
token_ids
tokens
span_token_indices
mask_token_indices
attention_layers
```

如果 full attention tensor 太大，可以只保存压缩后的 attention summaries，而不是完整 `[layers, heads, seq, seq]`。

---

## 任务 3：优先提取 attention summary，而不是保存巨大 tensor

优先构造以下 pre-recovery attention features：

### A. span attention mass

```text
span_attention_in_mass
span_attention_out_mass
span_attention_self_mass
span_attention_rank
```

### B. mask-to-span attention

```text
mask_to_span_attention
span_to_mask_attention
mask_to_span_rank
```

### C. question context attention

```text
question_target_to_span_attention
operation_to_span_attention
number_context_to_span_attention
span_to_question_mean_attention
```

如果 question_target / operation / number_context 无法稳定定位，则记录 missing rate。

### D. attention entropy / concentration

```text
span_attention_entropy
mask_attention_entropy
top1_attention_mass
top3_attention_mass
top5_attention_mass
effective_attention_edge_count
```

### E. original-masked attention delta

```text
delta_span_in_mass
delta_span_out_mass
delta_mask_to_span_attention
delta_attention_entropy
delta_topk_mass
delta_span_rank
```

核心假设：

```text
脆弱 span 在 masked 后会引起 attention distribution 的异常重排。
```

---

## 任务 4：新增 probe feature sets

在现有 probe / calibration 框架中新增：

```text
attention_pre_recovery
hidden_plus_attention_pre_recovery
```

继续保留：

```text
span_type_only
surface_rule
hidden_no_recovered
hidden_pre_recovery_enriched
hidden_with_recovered
```

其中：

```text
hidden_with_recovered
```

仍然只能作为 leakage diagnostic，不可 gate。

本轮新的 gate candidate 是：

```text
hidden_plus_attention_pre_recovery
```

同时也要单独报告：

```text
attention_pre_recovery
```

以判断 attention 特征本身是否有效。

---

## 任务 5：防泄漏检查

所有 gate-eligible feature names 禁止包含：

```text
recovered
solution_path
drift
bucket
risk_strength
gold
answer
label
target
```

如果出现，测试失败。

attention features 允许包含：

```text
original
masked
attention
span
mask
entropy
rank
mass
delta
```

但不允许出现 recovered channel。

---

## 任务 6：评估指标

沿用 2H-D 的 ordinal / budget-aware evaluation。

必须报告：

```text
macro_f1
balanced_accuracy
bucket_3_recall
spearman
pairwise_ordering_accuracy
bucket_3_vs_bucket_1_auc
top-10% bucket_3 precision / recall / f1
top-20% bucket_3 precision / recall / f1
per-question top-k solution-path coverage
off-path distractor budget share
bootstrap Spearman delta vs surface_rule
bootstrap Spearman delta vs hidden_pre_recovery_enriched
bootstrap macro_f1 delta vs surface_rule
```

---

## 任务 7：通过条件

只有满足以下条件，才建议进入 2000-case rerun：

```text
1. hidden_plus_attention_pre_recovery 的 Spearman 显著高于 surface_rule；
2. hidden_plus_attention_pre_recovery 的 Spearman 高于 hidden_pre_recovery_enriched；
3. top-10% 或 top-20% bucket_3 precision/F1 至少一项稳定优于 surface_rule；
4. off-path distractor budget share 不明显恶化；
5. top-k solution-path coverage 不下降；
6. feature leakage tests 全部通过；
7. 结果不是只由一个方法偶然通过，至少两个 scoring methods 支持同一方向。
```

如果只有一个指标或一个 scoring method 通过，不进入 2000。

---

## 任务 8：必须回答的问题

最终报告必须回答：

```text
1. attention features 是否单独超过 surface_rule？
2. hidden + attention 是否超过 hidden-only enriched？
3. attention features 改善的是 classification、ranking，还是 budget-aware selection？
4. bucket_3 top-budget precision 是否改善？
5. off-path distractor budget 是否恶化？
6. 哪些 attention feature family 最有用？
7. 是否建议进入 2000-case rerun？
8. 如果仍不通过，是否应转向 trajectory-level features？
```

---

## 输出文件

输出目录：

```text
outputs/logs/sprint_2I_attention_features_500/
```

输出文件：

```text
attention_cache_report.json
attention_feature_dataset.jsonl
attention_feature_report.json
attention_probe_eval_report.json
attention_ordinal_calibration_report.json
attention_budget_curve.json
review_gate_attention_features.json
review_gate_attention_features.md
```

---

## 通过后的下一步

如果 2I 通过：

```text
Sprint 2J: 2000-case hidden+attention pre-recovery rerun
```

如果 2I 不通过：

```text
不要扩大规模；
转向 trajectory-level features 或 hybrid recovery-guided pipeline。
```
