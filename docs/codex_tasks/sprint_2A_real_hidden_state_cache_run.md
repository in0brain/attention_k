# Sprint 2A-real：Real Hidden State Cache Run

## 目标

在 Sprint 2A stub hidden-state cache pipeline 已经跑通的前提下，使用本地 HuggingFace causal LM 对 Sprint 1R 生成的 20 条 reviewed cases 执行真实模型 forward，缓存 `original_question`、`masked_question`、`recovered_questions` 的真实 hidden states。

本 sprint 只做真实 hidden state cache，不做 representation feature extraction，不训练 probe，不做 attention guidance，不修改 recovery pipeline。

---

## 背景

Sprint 2A 的目标是用 `stub_hidden_state_v0` 跑通 hidden-state cache pipeline，包括：

```text
manifest 读取
token alignment
hidden state tensor 保存
hidden_state_manifest.jsonl
hidden_state_cache_report.json
token_alignment_report.json
```

Sprint 2A-real 在此基础上只替换 backend：

```text
stub_hidden_state_v0
→ hf_local_causal_lm_hidden_states_v0
```

本 sprint 明确允许使用本地 HF 模型运行真实 forward 并缓存真实 hidden states。但必须使用本地模型路径，不允许联网，不允许自动下载模型。

---

## 前置条件

开始前确认 Sprint 2A 已完成，并且以下文件存在：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
```

主输入文件仍然是：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

如果 Sprint 2A stub 输出不存在，停止执行，提示先完成 Sprint 2A。

---

## 必读文件

Codex 执行前必须阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/codex_tasks/sprint_2A_hidden_state_cache_baseline.md
docs/codex_tasks/sprint_2A_real_hidden_state_cache_run.md
```

按需阅读：

```text
src/recover_attention/token_alignment.py
src/recover_attention/hidden_state_cache.py
scripts/16_cache_hidden_states.py
tests/test_hidden_state_cache.py
tests/test_token_alignment.py
```

不需要读取：

```text
docs/reference/*
```

除非发现设计边界冲突，再向用户说明并等待确认。

---

## 输入文件

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

注意：

1. 不读取人工 Markdown。
2. 不回写 Sprint 1Q / 1R 文件。
3. 不修改 Sprint 2A stub 输出。
4. 只读取 2A stub 输出用于确认 pipeline 已完成。

---

## 输出目录

新建真实 hidden-state 输出目录：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/
```

输出文件：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

hidden state tensor 不得嵌入 JSONL，只能保存为 `.pt`，并在 manifest 中记录路径。

---

## 允许新增或修改的文件

允许修改：

```text
src/recover_attention/hidden_state_cache.py
scripts/16_cache_hidden_states.py
tests/test_hidden_state_cache.py
PROGRESS.md
docs/progress/sprint_2_history.md
```

如果当前项目还没有真实 HF backend，可以在 `hidden_state_cache.py` 中新增：

```text
hf_local_causal_lm_hidden_states_v0
```

如果 CLI 缺少真实 backend 参数，可以小幅修改：

```text
scripts/16_cache_hidden_states.py
```

允许新增：

```text
docs/codex_tasks/sprint_2A_real_hidden_state_cache_run.md
```

---

## 禁止修改的文件和目录

禁止修改：

```text
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/*
docs/reference/*
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
outputs/logs/sprint_1O_recovery_scoring/*
outputs/logs/sprint_1P_upgraded_downstream/*
outputs/logs/sprint_1Q_real_signal_quality_review/*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
```

禁止修改以下逻辑：

```text
recovery prompt
recover_generation.py
recover_scoring.py
masked_questions.py
unit/group budget construction
attention guidance
probe training
NLI backend
Ollama recovery backend
```

---

## Backend 要求

### 必须使用 backend

```text
hf_local_causal_lm_hidden_states_v0
```

该 backend 必须满足：

1. 只使用本地 `--model-path`。
2. 不允许自动下载模型。
3. 不允许联网。
4. 使用 `local_files_only=True`。
5. 不调用 Ollama。
6. 不调用 OpenAI API。
7. 不训练模型。
8. 不生成答案。
9. 只做 forward hidden-state extraction。
10. 设置 `output_hidden_states=True`。
11. 设置 `use_cache=False`。
12. 支持 `batch_size=1`。
13. 支持 `max_length` 截断。
14. 支持只保存指定层的 hidden states。
15. 保存 `.pt` tensor，并在 manifest 中记录路径。

---

## 模型路径

默认本地模型路径建议：

```text
D:/models/Qwen2.5-7B-Instruct
```

执行前必须检查该路径是否存在。

如果路径不存在，停止执行并提示用户确认真实模型路径，不要联网下载模型。

---

## 显存策略

用户本地 GPU 约 12GB 显存。Qwen2.5-7B 全精度或普通 fp16 可能显存不足，因此真实 run 应优先使用低显存配置。

推荐参数：

```text
--device-map auto
--load-in-4bit
--bnb-4bit-compute-dtype float16
--batch-size 1
--max-length 512
```

如果当前代码没有 4bit 支持，则至少支持：

```text
--device auto
--dtype float16
--batch-size 1
--max-length 512
```

若发生 CUDA OOM：

1. 不要自动扩大实验。
2. 不要改跑大规模。
3. 先停止。
4. 输出清晰错误。
5. 建议用户改用 4bit、CPU offload、更少层或更小模型。

---

## 缓存层

默认真实层选择：

```text
0 8 16 24 27
```

如果模型层数不足，应自动过滤不存在的层并写入 warning。

manifest 中必须记录实际缓存的层：

```json
{
  "requested_layer_indices": [0, 8, 16, 24, 27],
  "resolved_layer_indices": [0, 8, 16, 24, 27]
}
```

---

## CLI 参数要求

`scripts/16_cache_hidden_states.py` 应支持以下真实 backend 参数：

```text
--model-path
--device
--device-map
--dtype
--max-length
--batch-size
--load-in-4bit
--bnb-4bit-compute-dtype
--trust-remote-code
```

默认必须安全：

```text
--trust-remote-code false
```

只有模型确实需要时，用户再单独确认是否允许 `--trust-remote-code`。

---

## 推荐真实运行命令

PowerShell：

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_real_hidden_state_cache `
  --backend hf_local_causal_lm_hidden_states_v0 `
  --model-path D:/models/Qwen2.5-7B-Instruct `
  --layer-indices 0 8 16 24 27 `
  --max-length 512 `
  --batch-size 1 `
  --device-map auto `
  --load-in-4bit `
  --bnb-4bit-compute-dtype float16 `
  --mask-token "[MASK]" `
  --overwrite
```

如果 4bit 不可用，可退回：

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_real_hidden_state_cache `
  --backend hf_local_causal_lm_hidden_states_v0 `
  --model-path D:/models/Qwen2.5-7B-Instruct `
  --layer-indices 0 8 16 24 27 `
  --max-length 512 `
  --batch-size 1 `
  --device auto `
  --dtype float16 `
  --mask-token "[MASK]" `
  --overwrite
```

---

## Report 要求

### `real_run_metadata.json`

至少包含：

```json
{
  "backend": "hf_local_causal_lm_hidden_states_v0",
  "model_path": "D:/models/Qwen2.5-7B-Instruct",
  "model_name_or_path": "...",
  "tokenizer_name_or_path": "...",
  "device": "...",
  "device_map": "...",
  "dtype": "...",
  "load_in_4bit": true,
  "max_length": 512,
  "batch_size": 1,
  "requested_layer_indices": [0, 8, 16, 24, 27],
  "resolved_layer_indices": [0, 8, 16, 24, 27],
  "num_cases": 20,
  "num_inputs_total": 60,
  "created_at": "..."
}
```

### `hidden_state_cache_report.json`

至少包含：

```json
{
  "input_path": "...",
  "output_dir": "...",
  "backend": "hf_local_causal_lm_hidden_states_v0",
  "num_cases": 20,
  "num_inputs_total": 60,
  "input_type_counts": {
    "original": 20,
    "masked": 20,
    "recovered": 20
  },
  "num_hidden_state_files": 60,
  "layer_indices": [0, 8, 16, 24, 27],
  "seq_len_summary": {
    "min": 0,
    "max": 0,
    "mean": 0
  },
  "alignment_status_counts": {},
  "human_recoverability_label_counts": {},
  "human_error_type_counts": {},
  "probe_usage_counts": {},
  "failure_count": 0,
  "failures": []
}
```

### `hidden_state_manifest.jsonl`

每条记录至少包含：

```json
{
  "cache_id": "...",
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",
  "input_type": "original",
  "input_index": 0,
  "backend": "hf_local_causal_lm_hidden_states_v0",
  "model_name": "...",
  "tokenizer_name": "...",
  "layer_indices": [0, 8, 16, 24, 27],
  "seq_len": 0,
  "hidden_size": 0,
  "hidden_state_shape": [5, 0, 0],
  "hidden_state_path": "...",
  "alignment_status": "ok",
  "alignment_warnings": [],
  "human_recoverability_label": "...",
  "human_attention_anchor_label": "...",
  "human_error_type": "...",
  "probe_usage": "..."
}
```

---

## 测试要求

本 sprint 不要求 pytest 运行真实模型。

需要新增或更新测试，覆盖：

1. 真实 backend 参数解析。
2. 本地模型路径不存在时清晰失败。
3. `local_files_only=True` 被传入真实 backend loader。
4. `load_in_4bit` 参数可被解析。
5. `layer_indices` 解析与越界过滤逻辑。
6. 不修改输入 manifest。
7. 不修改 Sprint 1Q / 1R / 2A stub 输出。
8. CLI 在 fake/local mock backend 下可以通过测试。

真实模型 forward 不进入 pytest。

---

## 执行前 Preflight 要求

Codex 在任何修改或真实运行前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text
1. 已阅读文件列表
2. 本次允许修改的文件
3. 本次禁止修改的文件
4. 本次必须运行的命令
5. 是否需要读取 docs/reference/*
6. 是否发现冲突
7. 是否确认本 sprint 明确允许真实本地 HF hidden-state caching
8. 将使用的本地 model path
9. 是否会联网下载模型：必须回答不会
10. 是否会修改 1Q / 1R / 2A stub 输出：必须回答不会
```

---

## 必须运行的命令

### 1. 先运行测试

```powershell
conda run -n recover_attention python -m pytest -q
```

### 2. 再运行真实 hidden-state cache

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_real_hidden_state_cache `
  --backend hf_local_causal_lm_hidden_states_v0 `
  --model-path D:/models/Qwen2.5-7B-Instruct `
  --layer-indices 0 8 16 24 27 `
  --max-length 512 `
  --batch-size 1 `
  --device-map auto `
  --load-in-4bit `
  --bnb-4bit-compute-dtype float16 `
  --mask-token "[MASK]" `
  --overwrite
```

如果 4bit 参数当前不可用或 bitsandbytes 不可用，停止并向用户报告，不要擅自改用更大显存配置。

### 3. 再次运行测试

```powershell
conda run -n recover_attention python -m pytest -q
```

### 4. 检查改动

```powershell
git diff --name-only
git status --short
```

---

## 验收标准

1. Sprint 2A stub 输出已存在。
2. `hf_local_causal_lm_hidden_states_v0` 使用本地模型路径运行。
3. 不联网，不下载模型。
4. 不调用 Ollama。
5. 不调用 OpenAI API。
6. 不训练 probe。
7. 不做 representation analysis。
8. 不修改 recovery / NLI / masked question / unit budget 逻辑。
9. 不修改 Sprint 1Q / 1R 输入文件。
10. 不修改 Sprint 2A stub 输出。
11. 成功生成真实 `hidden_state_manifest.jsonl`。
12. 成功生成真实 `hidden_state_cache_report.json`。
13. 成功生成真实 `token_alignment_report.json`。
14. 成功生成 `real_run_metadata.json`。
15. 每个 reviewed case 至少有 original / masked / recovered 三类真实 hidden-state cache。
16. hidden states 保存为 `.pt`，不嵌入 JSONL。
17. fragment recovery 不导致真实 cache run 失败。
18. pytest 通过。
19. `PROGRESS.md` 更新当前阶段为 Sprint 2A-real completed。
20. `docs/progress/sprint_2_history.md` 记录 Sprint 2A-real 的输入、输出、命令、检查结果和遗留问题。
21. 下一步标记为 Sprint 2B：Representation Feature Extraction。

---

## 明确不做

本 sprint 不做：

```text
1. 不训练 probe。
2. 不构建 probe dataset。
3. 不做 representation feature extraction。
4. 不做 hidden-state distance analysis。
5. 不做 PCA / UMAP / t-SNE。
6. 不做 attention guidance。
7. 不修改 transformer attention。
8. 不重跑 CoT。
9. 不生成答案。
10. 不评估 answer accuracy。
11. 不修改 recovery prompt。
12. 不修改 recovery scoring。
13. 不修改 NLI scoring。
14. 不修改 masked question construction。
15. 不实现 unit/group mask budget selector。
16. 不修改 Sprint 1Q / 1R / 2A stub 输入或输出。
```

---

## PROGRESS.md 更新建议

```text
当前阶段：Sprint 2A-real 已完成：Real Hidden State Cache Run。

已完成：
- 基于 Sprint 2A stub cache pipeline，使用本地 HF causal LM backend 缓存真实 hidden states。
- 读取 Sprint 1R 生成的 sprint_1Q_to_2A_manifest.jsonl。
- 对 original / masked / recovered 输入执行真实 forward。
- 缓存指定层 hidden states。
- 生成真实 hidden_state_manifest.jsonl。
- 生成真实 hidden_state_cache_report.json。
- 生成真实 token_alignment_report.json。
- 生成 real_run_metadata.json。

特别说明：
- 本 sprint 只使用本地模型路径。
- 本 sprint 不联网，不下载模型。
- 本 sprint 不调用 Ollama 或 OpenAI API。
- 本 sprint 未训练 probe。
- 本 sprint 未做 representation analysis。
- 本 sprint 未执行 attention guidance。
- 本 sprint 未修改 Sprint 1Q / 1R / 2A stub 输出。

下一步：Sprint 2B：Representation Feature Extraction。
```

---

## docs/progress/sprint_2_history.md 更新建议

新增章节：

```text
## Sprint 2A-real：Real Hidden State Cache Run

已完成内容：
- 实现或启用 hf_local_causal_lm_hidden_states_v0。
- 使用本地 HF 模型对 Sprint 1R manifest 中的 reviewed cases 缓存真实 hidden states。
- 输出 hidden_state_manifest.jsonl、hidden_state_cache_report.json、token_alignment_report.json 和 real_run_metadata.json。
- 保持 Sprint 1Q / 1R / 2A stub 产物只读。

输入文件：
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl

输出文件：
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
- outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
- outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
- outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt

运行命令：
- conda run -n recover_attention python -m pytest -q
- conda run -n recover_attention python scripts/16_cache_hidden_states.py ...
- conda run -n recover_attention python -m pytest -q

检查结果：
- pytest: passed
- real hidden-state cache command: passed
- hidden_state_manifest generated
- hidden_state_cache_report generated
- token_alignment_report generated
- real_run_metadata generated

遗留问题：
- 当前只完成真实 hidden states 缓存，尚未抽取 representation features。
- 当前不分析 hidden-state distance。
- 当前不训练 probe。
- 当前不执行 attention guidance。
- 后续 Sprint 2B 读取真实 cache 进行 representation feature extraction。

下一步建议：
- Sprint 2B：Representation Feature Extraction。
```
