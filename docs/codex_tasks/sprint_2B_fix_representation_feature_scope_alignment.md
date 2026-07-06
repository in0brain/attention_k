# Sprint 2B-fix: Representation Feature Extraction Scope Alignment

## 0. Sprint 名称

Sprint 2B-fix — Representation Feature Extraction Scope Alignment

## 1. 任务背景

当前项目为：

```text
Reasoning-Aware Attention Guidance
```

Sprint 2 的最小闭环应保持为：

```text
Sprint 1R manifest
        ↓
2A hidden state cache
        ↓
2B representation features
        ↓
2C probe dataset
        ↓
2D probe baseline
        ↓
2E guidance candidate manifest dry run
        ↓
2F closed-loop report
```

当前已完成：

```text
Sprint 1Q human review
Sprint 1R consolidation
Sprint 2A stub hidden-state cache
Sprint 2A-real real HF hidden-state cache
```

Sprint 2A-real 已生成真实 hidden-state cache。

当前需要先修正并收口 Sprint 2B，而不是进入 2C。

---

## 2. 本 fix 的核心判断

本次遗留问题应归入：

```text
Sprint 2B：Representation Feature Extraction
```

不是归入：

```text
Sprint 2C：Probe Dataset Construction
```

原因：

```text
2B 负责 hidden states → representation features。
2C 负责 human labels → probe targets / probe dataset。
```

因此，本次 fix 只解决 2B 的 feature extraction 边界、输出命名、文档错误和 preflight 遗留问题。

---

## 3. 本 fix 的目标

把 Sprint 2B 收口为最小可用 representation feature extraction。

目标：

```text
从 hidden states 中抽取最小可用表征特征。
```

2B 只做：

```text
hidden-state cache
→ pooled representations
→ cosine distance features
→ representation_features.jsonl
→ representation_feature_report.json
```

2B 不做：

```text
probe dataset
target label mapping
train/dev/test split
probe training
guidance candidate manifest
attention steering
closed-loop report
```

---

## 4. 必须修正的遗留问题

### 4.1 不存在的测试文件

Preflight 已报告：

```text
tests/test_hidden_state_cache_hf.py does not exist.
```

因此：

```text
不得再把 tests/test_hidden_state_cache_hf.py 列为 must-read file。
不得把 tests/test_hidden_state_cache_hf.py 缺失视为失败。
不得为了满足旧 task card 新建这个测试文件。
```

正确处理方式：

```text
只阅读仓库中实际存在的 hidden_state_cache 相关测试文件。
当前已知存在：
tests/test_hidden_state_cache.py
```

如果需要自动发现测试文件，可以使用：

```bash
find tests -maxdepth 1 -type f | sort
```

或 Windows PowerShell：

```powershell
Get-ChildItem tests -File | Sort-Object Name
```

---

### 4.2 2B task card 已有 AM 状态

当前已知：

```text
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
```

在本轮前已有 `AM` 状态。

要求：

```text
1. Preflight 必须记录该文件的初始 git status。
2. 不得把该文件的 AM 状态当成本轮实现新增证据。
3. 如果本 fix 需要修正该文件，只允许做最小范围修改。
4. 修改内容只限于：
   - 删除 tests/test_hidden_state_cache_hf.py 的必读要求；
   - 回归 2B 最小特征范围；
   - 回归 2B 输出命名；
   - 删除或降级过度设计的 feature_schema / manifest 要求；
   - 强化 2B/2C 边界。
5. 不得在该文件中加入 2C / 2D / 2E 内容。
```

如果不修改该文件，也必须在最终报告里说明：

```text
docs/codex_tasks/sprint_2B_representation_feature_extraction.md was pre-existing AM before this fix and was not modified.
```

---

## 5. 2B 最小特征范围

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

## 6. 输入文件

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

本 sprint 不重新生成这些文件。

本 sprint 不修改这些文件。

本 sprint 只读这些文件。

---

## 7. 输出文件

2B 的正式产物统一为：

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

不要把以下文件作为 2B 必需产物：

```text
representation_feature_manifest.jsonl
input_representation_summary.jsonl
feature_schema.json
```

如果旧实现已经生成这些文件：

```text
1. 不要把它们作为验收标准。
2. 不要让 2C 依赖它们。
3. 可以在 report 中标记为 deprecated_extra_outputs。
4. 不需要为它们接入全局 schema validator。
```

2B 最终验收以以下两个文件为准：

```text
representation_features.jsonl
representation_feature_report.json
```

---

## 8. 输出记录粒度

`representation_features.jsonl` 的记录数不得写死为 20。

记录粒度为：

```text
one masked case × one recovered variant
```

也就是说：

```text
每个 recovered variant 生成一条 representation feature record。
```

当前小样本期望：

```text
至少 20 条 representation feature records。
```

但实际数量应由 `hidden_state_manifest.jsonl` 中的 recovered variants 数量决定。

如果未来每个 masked case 有多个 recovered variants，输出记录数应自然增加。

---

## 9. tensor 加载约束

读取 `.pt` tensor 时必须使用：

```python
torch.load(path, map_location="cpu")
```

本 sprint：

```text
不使用 GPU。
不调用 HF model forward。
不加载 tokenizer。
不加载 causal LM。
不调用 transformers model。
```

实现时必须按 `masked_id` 分组逐 case 加载 tensor。

每次只加载当前 case 的：

```text
original tensor
masked tensor
current recovered tensor
```

禁止一次性把所有 `.pt` tensors 加载进内存。

推荐流程：

```text
1. 读取 hidden_state_manifest.jsonl 的轻量 metadata。
2. 按 masked_id 分组。
3. 对每个 masked_id：
   3.1 找到 original record。
   3.2 找到 masked record。
   3.3 找到 recovered record(s)。
   3.4 每次只加载当前 original / masked / recovered tensors。
   3.5 计算该 recovered variant 的 features。
   3.6 写入一条 representation_features record。
   3.7 释放 tensor。
4. 写出 representation_feature_report.json。
```

---

## 10. representation 定义

每个 hidden state tensor 预期 shape 为：

```text
[num_layers, seq_len, hidden_size]
```

如果 shape 不符合预期：

```text
跳过该 record 或 case。
在 report 中记录 bad_tensor_shape。
不得静默 reshape。
不得截断 layer。
```

---

## 11. 必需 pooling

### 11.1 question pooled representation

对每个 layer 的所有 token 做 mean pooling：

```text
question_pooled_representation[layer] = mean(hidden_states[layer, :, :])
```

这是必需特征。

---

### 11.2 span pooled representation

使用 manifest 中已有 span 信息和 `token_char_ranges` 定位 span token。

span 来源：

```text
original input:
  masked_original_spans

masked input:
  mask_char_ranges

recovered input:
  recovered_fill_spans
```

定位方式：

```text
token_char_ranges 与 span char range 有 overlap，则该 token 属于 span。
```

如果无法定位 span/token overlap：

```text
对应 span pooled representation 置为 null。
对应 span distance 置为 null。
记录 warning。
不得中断流程。
```

span pooled representation 是辅助但应尽力生成。

---

### 11.3 mask position representation

mask position representation 用于描述被 mask / 被恢复位置附近的 hidden state。

优先定义：

```text
original:
  使用 masked_original_spans 对应 token 的 pooled representation。

masked:
  使用 mask_char_ranges 对应 token 的 pooled representation。

recovered:
  使用 recovered_fill_spans 对应 token 的 pooled representation。
```

如果 span/token overlap 无法定位：

```text
mask_position_representation = null
记录 warning
不中断流程
```

对于 masked input，如果 `[MASK]` token 能直接定位，也可以使用 `[MASK]` token 对应 representation。

但不得为了定位失败而调用 tokenizer 或重新跑模型。

---

## 12. 必需 distance features

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
question pooled cosine distance 必须存在。
span pooled cosine distance 可以为 null。
mask position cosine distance 可以为 null。
```

layer-wise distance curve 是 2B 的关键产物。

---

## 13. 可选轻量聚合字段

为了方便 2C / 2D 后续使用，可以为每个 layer-wise curve 添加轻量统计：

```text
mean
max
min
first_layer
last_layer
delta_last_minus_first
```

例如：

```text
question_original_masked_cosine_by_layer
question_original_masked_cosine_mean
question_original_masked_cosine_max
question_original_masked_cosine_min
question_original_masked_cosine_first_layer
question_original_masked_cosine_last_layer
question_original_masked_cosine_delta_last_minus_first
```

这些聚合字段是从 cosine curve 派生出来的简单数值。

不要在 2B 中解释它们是否有效。

不要在 2B 中做统计显著性分析。

不要在 2B 中做 probe feature importance。

---

## 14. representation_features.jsonl 推荐字段

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

  "question_original_masked_cosine_mean": 0.15,
  "question_original_recovered_cosine_mean": 0.12,
  "question_masked_recovered_cosine_mean": 0.09,

  "human_attention_anchor_label": null,
  "human_attention_anchor_label_name": null,
  "probe_usage": null,

  "warnings": []
}
```

注意：

```text
human fields 只能作为 metadata 复制。
2B 不得使用 human fields 计算 feature。
2B 不得生成 target。
2B 不得生成 probe_label。
```

---

## 15. representation_feature_report.json 必需字段

report 必须包含：

```json
{
  "sprint": "2B-fix",
  "status": "ok",
  "backend": "representation_features_minimal_v0",

  "source_cache": {
    "hidden_state_manifest_path": "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl",
    "hidden_state_cache_report_path": "outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json",
    "token_alignment_report_path": "outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json",
    "real_run_metadata_path": "outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json",
    "source_cache_backend": "hf_local_causal_lm_hidden_states_v0",
    "source_model_name": "...",
    "source_tokenizer_name": "...",
    "source_layer_indices": [0, 8, 16, 24, 27],
    "source_num_inputs_total": 60
  },

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
    "missing_tests_test_hidden_state_cache_hf_py": true,
    "missing_tests_test_hidden_state_cache_hf_py_is_failure": false,
    "sprint_2B_task_card_preexisting_AM": true
  },

  "warning_counts": {
    "missing_span_overlap": 0,
    "missing_mask_position_overlap": 0,
    "missing_token_offsets": 0,
    "nonfinite_tensor_values": 0,
    "bad_tensor_shape": 0,
    "missing_tensor_file": 0
  },

  "warnings": []
}
```

`num_feature_records` 不得写死为 20。

当前小样本只要求：

```text
num_feature_records >= 20
```

---

## 16. 数值稳定性

JSON 输出中不得出现：

```text
NaN
Inf
-Infinity
```

如果 tensor 中存在非有限值：

```text
记录 nonfinite count。
使用安全方式处理。
不得让整个任务崩溃。
```

推荐：

```python
torch.nan_to_num(...)
```

cosine distance 遇到 near-zero norm 时：

```text
对应 distance 置为 null。
记录 warning。
不中断流程。
```

---

## 17. 允许新增或修改的代码文件

如果 2B feature extraction 尚未实现，允许新增：

```text
src/recover_attention/representation_features.py
scripts/17_extract_representation_features.py
tests/test_representation_features.py
```

如果这些文件已经存在，只允许做与本 fix 相关的最小修改。

---

## 18. 允许修改的文档文件

允许修改：

```text
PROGRESS.md
docs/progress/sprint_2_history.md
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
```

其中，对 `docs/codex_tasks/sprint_2B_representation_feature_extraction.md` 的修改只能用于修正 2B 边界和错误要求。

不允许在该文件里加入 2C / 2D / 2E 的实现要求。

可以新增本 fix card：

```text
docs/codex_tasks/sprint_2B_fix_representation_feature_scope_alignment.md
```

---

## 19. 禁止修改

禁止修改：

```text
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/SKILL.md

data/processed/*

outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*

src/recover_attention/candidate_extraction.py
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
src/recover_attention/human_review_consolidation.py
src/recover_attention/hidden_state_cache.py

scripts/01_prepare_data.py
scripts/02_extract_candidate_spans.py
scripts/03_build_ablation_units.py
scripts/04_build_ablated_questions.py
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
scripts/13_rebuild_downstream_real_signals.py
scripts/14_rebuild_downstream_upgraded_recovery_scoring.py
scripts/15_consolidate_human_review.py
scripts/16_cache_hidden_states.py
```

尤其禁止修改 2A-real 输出。

---

## 20. CLI 要求

新增或修正脚本：

```text
scripts/17_extract_representation_features.py
```

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

输出应为：

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

不得提供：

```text
--model-path
--device
--device-map
--dtype
--load-in-4bit
--trust-remote-code
```

这些属于 2A-real hidden state cache，不属于 2B feature extraction。

---

## 21. 测试要求

新增或修正：

```text
tests/test_representation_features.py
```

至少覆盖：

```text
1. 不要求 tests/test_hidden_state_cache_hf.py 存在。
2. 不因 tests/test_hidden_state_cache_hf.py 缺失失败。
3. 能读取 2A-real style hidden_state_manifest.jsonl。
4. 使用 torch.load(path, map_location="cpu")。
5. 不调用 HF model forward。
6. 不使用 GPU。
7. 按 masked_id 逐 case 加载 tensors。
8. 不一次性加载所有 tensors。
9. 每个 recovered variant 生成一条 record。
10. representation_features.jsonl 记录数不写死为 20。
11. 当前小样本 feature records >= 20。
12. question pooled representation 必须生成。
13. span pooled representation 尽力生成，失败时 null + warning。
14. mask position representation 尽力生成，失败时 null + warning。
15. original vs masked cosine distance 必须生成。
16. original vs recovered cosine distance 必须生成。
17. masked vs recovered cosine distance 必须生成。
18. layer-wise distance curve 必须生成。
19. JSON 输出中不得出现 NaN / Inf。
20. 2B 不生成 target / probe_label / split。
21. 2B 不生成 probe_dataset.jsonl。
22. 2B 不训练 probe。
23. 2B 不生成 guidance_candidate_manifest.jsonl。
24. CLI smoke test 能生成 representation_features.jsonl 和 representation_feature_report.json。
```

---

## 22. 必须运行的命令

Preflight：

```bash
git status --short
```

确认实际存在的测试文件：

```bash
python - <<'PY'
from pathlib import Path
for p in sorted(Path("tests").glob("test_*hidden_state*")):
    print(p)
PY
```

运行 targeted test：

```bash
conda run -n recover_attention python -m pytest tests/test_representation_features.py -q
```

运行 2B extraction：

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

运行全量测试：

```bash
conda run -n recover_attention python -m pytest -q
```

最终检查：

```bash
git diff --name-only
git status --short
```

---

## 23. 验收标准

本 fix 完成时必须满足：

```text
1. 不再要求读取 tests/test_hidden_state_cache_hf.py。
2. tests/test_hidden_state_cache_hf.py 缺失不被视为失败。
3. 已记录 docs/codex_tasks/sprint_2B_representation_feature_extraction.md 的 pre-existing AM 状态。
4. 2B 输出统一为 representation_features.jsonl。
5. 2B report 统一为 representation_feature_report.json。
6. 不再把 representation_feature_manifest.jsonl / input_representation_summary.jsonl / feature_schema.json 作为必需产物。
7. 每个 recovered variant 生成一条 feature record。
8. feature record 数量不写死为 20。
9. 当前小样本 feature record 数量至少 20。
10. 使用 torch.load(path, map_location="cpu")。
11. 不使用 GPU。
12. 不调用 HF model forward。
13. 不一次性加载所有 tensors。
14. question pooled representation 已生成。
15. span pooled representation 尽力生成，失败时 null + warning。
16. mask position representation 尽力生成，失败时 null + warning。
17. original vs masked cosine distance 已生成。
18. original vs recovered cosine distance 已生成。
19. masked vs recovered cosine distance 已生成。
20. layer-wise distance curve 已生成。
21. representation_feature_report.json 包含 source cache metadata。
22. 2B 不生成 probe dataset。
23. 2B 不选择 target。
24. 2B 不划分 train/dev/test。
25. 2B 不训练 probe。
26. 2B 不执行 attention guidance。
27. 2B 不修改 Sprint 1Q / 1R / 2A / 2A-real outputs。
28. targeted pytest passed。
29. full pytest passed。
30. PROGRESS.md 已更新。
31. docs/progress/sprint_2_history.md 已更新。
```

---

## 24. PROGRESS.md 更新建议

在当前进度中加入：

```text
Sprint 2B-fix completed: Representation Feature Extraction Scope Alignment.

- Fixed the 2B boundary to match the Sprint 2 minimal loop.
- 2B now outputs:
  - outputs/logs/sprint_2B_representation_features/representation_features.jsonl
  - outputs/logs/sprint_2B_representation_features/representation_feature_report.json
- Removed the invalid requirement to read tests/test_hidden_state_cache_hf.py.
- tests/test_hidden_state_cache_hf.py is absent and its absence is not a failure.
- docs/codex_tasks/sprint_2B_representation_feature_extraction.md had pre-existing AM status before this fix; final status recorded in task report.
- Feature scope:
  - mask position representation
  - span pooled representation
  - question pooled representation
  - original vs masked cosine distance
  - original vs recovered cosine distance
  - masked vs recovered cosine distance
  - layer-wise distance curve
- Runtime constraints:
  - torch.load(..., map_location="cpu")
  - CPU-only
  - no HF forward
  - grouped by masked_id
  - no all-tensor in-memory loading
- Not done:
  - no probe dataset
  - no target selection
  - no train/dev/test split
  - no probe training
  - no guidance candidate manifest
  - no attention steering
- Next: Sprint 2C Probe Dataset Construction.
```

---

## 25. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text
## Sprint 2B-fix：Representation Feature Extraction Scope Alignment

### Goal

Align Sprint 2B with the original minimal Sprint 2 loop.

2B is responsible for:

hidden states → representation features

2C is responsible for:

human labels → probe targets / probe dataset

### Fixes

- Removed the invalid must-read requirement for tests/test_hidden_state_cache_hf.py.
- Confirmed that tests/test_hidden_state_cache_hf.py does not exist and its absence is not a failure.
- Recorded that docs/codex_tasks/sprint_2B_representation_feature_extraction.md had pre-existing AM status.
- Restored the 2B output contract:
  - representation_features.jsonl
  - representation_feature_report.json
- De-scoped representation_feature_manifest.jsonl, input_representation_summary.jsonl, and feature_schema.json as required outputs.
- Restored the minimal feature scope:
  - mask position representation
  - span pooled representation
  - question pooled representation
  - original vs masked cosine distance
  - original vs recovered cosine distance
  - masked vs recovered cosine distance
  - layer-wise distance curve

### Outputs

- outputs/logs/sprint_2B_representation_features/representation_features.jsonl
- outputs/logs/sprint_2B_representation_features/representation_feature_report.json

### Not Done

- No probe dataset.
- No target selection.
- No train/dev/test split.
- No probe training.
- No guidance candidate manifest.
- No attention steering.

### Next

Sprint 2C：Probe Dataset Construction.
```

---

## 26. 下一步边界

完成本 fix 后，才进入：

```text
Sprint 2C：Probe Dataset Construction
```

2C 的输入应为：

```text
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

2C 才处理：

```text
1Q human labels
→ probe target mapping
→ probe_dataset.jsonl
→ probe_dataset_report.json
```

不要让 2C 重新读取 `.pt tensors`。

不要让 2C 重新抽取 representation features。

不要让 2C 调用 HF model。

不要让 2C 训练 probe。

训练留给 2D。
