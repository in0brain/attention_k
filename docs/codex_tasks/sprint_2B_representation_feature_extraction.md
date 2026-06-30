# Sprint 2B: Representation Feature Extraction

## 0. Sprint 名称

Sprint 2B — Representation Feature Extraction

## 1. 当前边界

Sprint 2B 只负责：

```text
hidden-state cache
→ pooled representations
→ cosine distance features
→ representation_features.jsonl
→ representation_feature_report.json
```

Sprint 2B 不做：

```text
probe dataset
target selection
train/dev/test split
probe training
guidance candidate manifest
attention steering
closed-loop report
```

2C 才读取 2B 的正式输出并构造：

```text
probe_dataset.jsonl
probe_dataset_report.json
```

---

## 2. 必须阅读的文件

开始实现前，先阅读：

```text
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
docs/progress/sprint_2_history.md
docs/codex_tasks/sprint_2A_hidden_state_cache_baseline.md
docs/codex_tasks/sprint_2A_real_hidden_state_cache_run.md
src/recover_attention/hidden_state_cache.py
src/recover_attention/token_alignment.py
scripts/16_cache_hidden_states.py
tests/test_hidden_state_cache.py
以及仓库中实际存在的 hidden_state_cache 相关测试文件
```

旧 HF-specific hidden-state cache 测试文件不存在时不是失败条件，不得为了满足旧要求新增测试文件。

---

## 3. 输入文件

默认输入目录：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/
```

默认输入文件：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

本 sprint 只读这些文件，不重新生成 hidden states，不修改 Sprint 1Q / 1R / 2A / 2A-real outputs。

---

## 4. 正式输出

2B 正式必需输出统一为：

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

以下 legacy/debug 文件不是 Sprint 2B 必需产物，不作为 acceptance criteria，也不得作为 2C 输入契约：

```text
deprecated_extra_outputs / optional_debug_outputs / non-required outputs:
- representation_feature_manifest.jsonl
- input_representation_summary.jsonl
- feature_schema.json
```

`feature_schema.json` is not required for Sprint 2B. Do not connect `feature_schema.json` to the global schema validator. Do not modify `src/recover_attention/schemas.py` for `feature_schema.json`.

---

## 5. 输出记录粒度

`representation_features.jsonl` 的记录粒度为：

```text
one masked case × one recovered variant
```

每个 recovered variant 生成一条 representation feature record。

当前小样本只要求：

```text
num_feature_records >= 20
```

不得把记录数写死为 20。

---

## 6. tensor 加载约束

读取 `.pt` tensor 时必须使用：

```python
torch.load(path, map_location="cpu")
```

本 sprint：

```text
不使用 GPU
不调用 HF model forward
不加载 tokenizer
不加载 causal LM
不调用 transformers model
```

实现时必须按 `masked_id` 分组逐 case 加载 tensor。每次只加载当前 case 的：

```text
original tensor
masked tensor
current recovered tensor
```

禁止一次性把所有 `.pt` tensors 加载进内存。

---

## 7. 最小特征范围

Sprint 2B 只抽取以下最小特征：

```text
1. mask position representation
2. span pooled representation
3. question pooled representation
4. original vs masked cosine distance
5. original vs recovered cosine distance
6. masked vs recovered cosine distance
7. layer-wise distance curve
```

不要做复杂拓扑。

不要做 trajectory analysis。

不要做 PCA / UMAP / t-SNE。

不要做 hidden-state probe training。

不要做 feature importance 分析。

不要做 attention guidance。

---

## 8. representation 定义

每个 hidden state tensor 预期 shape 为：

```text
[num_layers, seq_len, hidden_size]
```

如果 shape 不符合预期：

```text
跳过该 record 或 case
在 report 中记录 bad_tensor_shape
不得静默 reshape
不得截断 layer
```

### question pooled representation

对每个 layer 的所有 token 做 mean pooling：

```text
question_pooled_representation[layer] = mean(hidden_states[layer, :, :])
```

这是必需特征。

### span pooled representation

使用 manifest 中已有 span 信息和 `token_char_ranges` 定位 span token。

span 来源：

```text
original input: masked_original_spans
masked input: mask_char_ranges
recovered input: recovered_fill_spans
```

定位方式：

```text
token_char_ranges 与 span char range 有 overlap，则该 token 属于 span
```

如果无法定位 span/token overlap：

```text
对应 span pooled representation 置为 null
对应 span distance 置为 null
记录 warning
不中断流程
```

### mask position representation

mask position representation 用于描述被 mask / 被恢复位置附近的 hidden state。

优先定义：

```text
original: 使用 masked_original_spans 对应 token 的 pooled representation
masked: 使用 mask_char_ranges 对应 token 的 pooled representation
recovered: 使用 recovered_fill_spans 对应 token 的 pooled representation
```

如果 span/token overlap 无法定位：

```text
mask_position_representation = null
记录 warning
不中断流程
```

不得为了定位失败而调用 tokenizer 或重新跑模型。

---

## 9. 必需 distance features

每条 `representation_features.jsonl` record 必须包含三组 cosine distance：

```text
original vs masked cosine distance
original vs recovered cosine distance
masked vs recovered cosine distance
```

每组 distance 都应按 layer 输出 curve：

```text
*_cosine_by_layer
```

至少对以下 pooling 计算：

```text
question pooled representation
span pooled representation, if available
mask position representation, if available
```

其中：

```text
question pooled cosine distance 必须存在
span pooled cosine distance 可以为 null
mask position cosine distance 可以为 null
```

可添加轻量聚合字段：

```text
mean
max
min
first_layer
last_layer
delta_last_minus_first
```

不要在 2B 中解释这些 feature 是否有效。

不要在 2B 中做统计显著性分析。

不要在 2B 中做 probe feature importance。

---

## 10. `representation_features.jsonl` 推荐字段

每条 record 推荐包含：

```json
{
  "feature_id": "...",
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",
  "original_cache_id": "...",
  "masked_cache_id": "...",
  "recovered_cache_id": "...",
  "recovered_input_index": 0,
  "source_cache_backend": "hf_local_causal_lm_hidden_states_v0",
  "model_name": "...",
  "tokenizer_name": "...",
  "layer_indices": [0, 8, 16, 24, 27],
  "hidden_size": 3584,
  "question_original_masked_cosine_by_layer": [0.1, 0.2],
  "question_original_recovered_cosine_by_layer": [0.1, 0.2],
  "question_masked_recovered_cosine_by_layer": [0.1, 0.2],
  "span_original_masked_cosine_by_layer": null,
  "span_original_recovered_cosine_by_layer": null,
  "span_masked_recovered_cosine_by_layer": null,
  "mask_position_original_masked_cosine_by_layer": null,
  "mask_position_original_recovered_cosine_by_layer": null,
  "mask_position_masked_recovered_cosine_by_layer": null,
  "human_attention_anchor_label": null,
  "human_attention_anchor_label_name": null,
  "probe_usage": null,
  "warnings": []
}
```

human fields 只能作为 metadata 复制。

2B 不得使用 human fields 计算 feature。

2B 不得生成 `target`、`probe_label`、`split`、`train_split`、`dev_split`、`test_split`。

---

## 11. `representation_feature_report.json` 必需字段

report 必须包含：

```json
{
  "sprint": "2B-fix",
  "status": "ok",
  "backend": "representation_features_minimal_v0",
  "source_cache": {},
  "outputs": {
    "representation_features_path": "outputs/logs/sprint_2B_representation_features/representation_features.jsonl",
    "representation_feature_report_path": "outputs/logs/sprint_2B_representation_features/representation_feature_report.json"
  },
  "counts": {
    "num_input_records": 60,
    "num_masked_groups": 20,
    "num_recovered_variants": 20,
    "num_feature_records": 20,
    "num_skipped_groups": 0,
    "num_skipped_recovered_variants": 0
  },
  "feature_scope": {
    "question_pooled_representation": true,
    "span_pooled_representation": true,
    "mask_position_representation": true,
    "cosine_distance": true,
    "layer_wise_distance_curve": true,
    "topology_features": false,
    "trajectory_features": false,
    "probe_dataset": false,
    "probe_training": false,
    "attention_guidance": false
  },
  "preflight_notes": {
    "missing_hf_hidden_state_cache_test_file": true,
    "missing_hf_hidden_state_cache_test_file_is_failure": false,
    "sprint_2B_task_card_preexisting_AM": true
  },
  "warning_counts": {}
}
```

`num_feature_records` 不得写死为 20。当前小样本只要求 `num_feature_records >= 20`。

---

## 12. 数值稳定性

JSON 输出中不得出现：

```text
NaN
Inf
-Infinity
```

如果 tensor 中存在非有限值：

```text
记录 nonfinite count
使用安全方式处理
不得让整个任务崩溃
```

cosine distance 遇到 near-zero norm 时：

```text
对应 distance 置为 null
记录 warning
不中断流程
```

---

## 13. 允许新增或修改

允许新增或修改：

```text
src/recover_attention/representation_features.py
scripts/17_extract_representation_features.py
tests/test_representation_features.py
PROGRESS.md
docs/progress/sprint_2_history.md
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
```

---

## 14. 禁止修改

禁止修改：

```text
AGENTS.md
README.md
docs/skill/SKILL.md
data/processed/*
outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*
src/recover_attention/hidden_state_cache.py
scripts/16_cache_hidden_states.py
recovery / NLI / masked question / probe / guidance 相关实现
```

---

## 15. CLI

推荐命令：

```bash
conda run -n recover_attention python scripts/17_extract_representation_features.py \
  --input-manifest outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl \
  --input-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json \
  --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json \
  --metadata outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json \
  --output-dir outputs/logs/sprint_2B_representation_features \
  --backend representation_features_minimal_v0 \
  --overwrite
```

CLI 不得提供：

```text
--model-path
--device
--device-map
--dtype
--load-in-4bit
--trust-remote-code
```

---

## 16. 必须运行的命令

```bash
git status --short
conda run -n recover_attention python -c "from pathlib import Path; [print(p) for p in sorted(Path('tests').glob('test_*hidden_state*'))]"
conda run -n recover_attention python -m pytest tests/test_representation_features.py -q
conda run -n recover_attention python scripts/17_extract_representation_features.py --input-manifest outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl --input-report outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json --alignment-report outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json --metadata outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json --output-dir outputs/logs/sprint_2B_representation_features --backend representation_features_minimal_v0 --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

---

## 17. 验收标准

本 sprint 完成时必须满足：

```text
1. 不再要求读取不存在的 HF-specific hidden-state cache 测试文件。
2. 该测试文件缺失不被视为失败。
3. 2B 输出统一为 representation_features.jsonl。
4. 2B report 统一为 representation_feature_report.json。
5. 不再把 legacy/debug 输出作为必需产物。
6. 每个 recovered variant 生成一条 feature record。
7. feature record 数量不写死为 20。
8. 当前小样本 feature record 数量至少 20。
9. 使用 torch.load(path, map_location="cpu")。
10. 不使用 GPU。
11. 不调用 HF model forward。
12. 不一次性加载所有 tensors。
13. question pooled representation 已生成。
14. span pooled representation 尽力生成，失败时 null + warning。
15. mask position representation 尽力生成，失败时 null + warning。
16. original vs masked cosine distance 已生成。
17. original vs recovered cosine distance 已生成。
18. masked vs recovered cosine distance 已生成。
19. layer-wise distance curve 已生成。
20. representation_feature_report.json 包含 source cache metadata。
21. 2B 不生成 probe dataset。
22. 2B 不选择 target。
23. 2B 不划分 train/dev/test。
24. 2B 不训练 probe。
25. 2B 不执行 attention guidance。
26. 2B 不修改 Sprint 1Q / 1R / 2A / 2A-real outputs。
27. targeted pytest passed。
28. full pytest passed。
29. PROGRESS.md 已更新。
30. docs/progress/sprint_2_history.md 已更新。
```
