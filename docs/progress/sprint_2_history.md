# Sprint 2 历史记录

## Sprint 2A：Hidden State Cache Baseline

已完成内容：

- 实现 `src/recover_attention/token_alignment.py`。
- 实现 single / group `[MASK]` char range 识别。
- 实现 original vs masked 的基础 deterministic 差分对齐。
- 实现 recovered question 模板填充对齐；fragment recovery 只记录 warning，不中断流程。
- 实现 stub tokenizer，输出 `tokens`、`token_ids`、`token_char_ranges`、`seq_len`。
- 实现 `src/recover_attention/hidden_state_cache.py`。
- 实现 `stub_hidden_state_v0`，生成 deterministic 小型 tensor。
- 实现 `scripts/16_cache_hidden_states.py`。
- 生成 Sprint 2A cache 输出目录。
- 新增 token alignment 和 hidden state cache pytest。

输入文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

输出文件：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

新增或修改文件：

- src/recover_attention/token_alignment.py
- src/recover_attention/hidden_state_cache.py
- scripts/16_cache_hidden_states.py
- tests/test_token_alignment.py
- tests/test_hidden_state_cache.py
- PROGRESS.md
- docs/progress/sprint_2_history.md

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_token_alignment.py tests/test_hidden_state_cache.py -q
conda run -n recover_attention python scripts/16_cache_hidden_states.py --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl --output-dir outputs/logs/sprint_2A_hidden_state_cache_baseline --backend stub_hidden_state_v0 --layer-indices 0 1 2 --hidden-size 8 --mask-token "[MASK]" --overwrite
conda run -n recover_attention python -m pytest -q
git diff --name-only
git status --short
```

检查结果：

- targeted pytest：16 passed。
- cache command：passed。
- full pytest：459 passed, 2 skipped。
- cache backend：`stub_hidden_state_v0`。
- num_cases：20。
- num_inputs_total：60。
- num_hidden_state_files：60。
- input_type_counts：`{original: 20, masked: 20, recovered: 20}`。
- hidden_size：8。
- layer_indices：`[0, 1, 2]`。
- token alignment：single mask cases 17，group mask cases 3。
- recovered fragment outputs：8。
- alignment_warning_count：8。
- Sprint 1Q / 1R 输入文件 hash 运行前后不变。

遗留问题：

- 当前 hidden states 来自 stub backend，不是真实模型。
- 当前 token alignment 是基础 deterministic 对齐，不处理复杂 paraphrase。
- recovered fragment 输出只记录 warning，后续在 2B / 2C 判断是否保留。
- 真实 HF hidden state backend 未运行；如需运行，必须由用户单独确认。
- 本 sprint 未训练 probe。
- 本 sprint 未做 representation feature extraction。
- 本 sprint 未做 hidden-state distance analysis、PCA / UMAP / t-SNE。
- 本 sprint 未执行 attention guidance。
- 本 sprint 未修改 1Q / 1R 输入文件。

下一步建议：

- Sprint 2B：Representation Feature Extraction。

## Sprint 2A-real：Real Hidden State Cache Run

已完成内容：

- 实现或启用 `hf_local_causal_lm_hidden_states_v0` 的本地 HF causal LM backend 支持。
- `scripts/16_cache_hidden_states.py` 支持真实 backend 参数：
  - `--model-path`
  - `--device`
  - `--device-map`
  - `--dtype`
  - `--max-length`
  - `--batch-size`
  - `--load-in-4bit`
  - `--bnb-4bit-compute-dtype`
  - `--trust-remote-code`
- 真实 backend 加载边界已实现：
  - 必须显式传入本地 `--model-path`。
  - `local_files_only=True`。
  - 不自动下载模型。
  - 不联网。
  - `trust_remote_code` 默认 false。
  - `output_hidden_states=True`。
  - `use_cache=False`。
- 实现 4bit run 的 bitsandbytes 缺失检测。
- 实现 layer index 越界过滤与 warning。
- 新增/更新测试覆盖真实 backend 参数解析、本地模型路径缺失、`local_files_only=True`、4bit 参数解析、bitsandbytes 缺失阻断和 layer index 过滤。
- 使用本地 HF causal LM backend 完成 Sprint 1R manifest 中 20 条 reviewed cases 的真实 hidden-state cache。
- 输出 `hidden_state_manifest.jsonl`、`hidden_state_cache_report.json`、`token_alignment_report.json` 和 `real_run_metadata.json`。
- 保持 Sprint 1Q / 1R / 2A stub 产物只读。

输入文件：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

预期输出文件：

```text
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_state_cache_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/token_alignment_report.json
outputs/logs/sprint_2A_real_hidden_state_cache/real_run_metadata.json
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
```

实际输出状态：passed。

output_dir：

```text
outputs/logs/sprint_2A_real_hidden_state_cache
```

新增或修改文件：

- src/recover_attention/hidden_state_cache.py
- scripts/16_cache_hidden_states.py
- tests/test_hidden_state_cache.py
- PROGRESS.md
- docs/progress/sprint_2_history.md

运行命令：

```bash
conda run -n recover_attention python -m pytest tests/test_hidden_state_cache.py -q
conda run -n recover_attention python -m pytest -q
conda run -n recover_attention python scripts/16_cache_hidden_states.py --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl --output-dir outputs/logs/sprint_2A_real_hidden_state_cache --backend hf_local_causal_lm_hidden_states_v0 --model-path D:/models/Qwen2.5-7B-Instruct --layer-indices 0 8 16 24 27 --max-length 512 --batch-size 1 --device-map auto --load-in-4bit --bnb-4bit-compute-dtype float16 --mask-token "[MASK]" --overwrite
git diff --name-only
git status --short
```

检查结果：

- `tests/test_hidden_state_cache.py`：14 passed。
- 全量 pytest：465 passed, 2 skipped。
- 本地模型路径检查：`D:/models/Qwen2.5-7B-Instruct` 存在。
- Sprint 2A stub 输出检查：`hidden_state_manifest.jsonl`、`hidden_state_cache_report.json`、`token_alignment_report.json` 均存在。
- 2A-real real hidden-state cache run：passed。
- backend：`hf_local_causal_lm_hidden_states_v0`。
- output_dir：`outputs/logs/sprint_2A_real_hidden_state_cache`。
- `hidden_state_manifest.jsonl`：60 records。
- `hidden_state_cache_report.json`：`num_cases=20`，`num_inputs_total=60`，`num_hidden_state_files=60`。
- `real_run_metadata.json`：`backend=hf_local_causal_lm_hidden_states_v0`，`num_cases=20`，`num_inputs_total=60`。
- 未修改 Sprint 1Q / 1R / 2A stub 输出。

遗留问题：

- 当前只完成真实 hidden states 缓存，尚未抽取 representation features。
- 当前不分析 hidden-state distance。
- 当前不训练 probe。
- 当前不执行 attention guidance。
- 后续 Sprint 2B 读取真实 cache 进行 representation feature extraction。

下一步建议：

- Sprint 2B：Representation Feature Extraction。
