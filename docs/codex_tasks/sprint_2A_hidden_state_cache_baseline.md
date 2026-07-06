# Sprint 2A：Hidden State Cache Baseline

## 目标

实现 hidden-state cache baseline。

本 sprint 读取 Sprint 1R 生成的 `sprint_1Q_to_2A_manifest.jsonl`，对每个 reviewed case 的 `original_question`、`masked_question`、`recovered_questions` 构造模型输入，完成 token / mask / span 的基础对齐，并生成可复用的 hidden state cache manifest、token alignment report 和 cache report。

Sprint 2A 的定位是：**缓存与对齐，不做分析**。

本 sprint 不训练 probe，不做 representation difference analysis，不做 PCA/UMAP/t-SNE，不做 attention guidance，不修改 recovery pipeline，不重建 1Q/1R 产物。

---

## 背景

Sprint 1R 已完成：

```text
Human Review Consolidation & Known Issue Freeze
```

1R 已经将 1Q 人工审核结果固化为结构化文件，并生成 Sprint 2A 输入 manifest：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

Sprint 2A 从该 manifest 开始，不再读取人工填写的 Markdown，不再回写 1Q / 1R 文件。

Sprint 2 的最小闭环规划为：

```text
2A Hidden State Cache Baseline
→ 2B Representation Feature Extraction
→ 2C Probe Dataset Construction
→ 2D Probe Training Baseline
→ 2E Guidance Candidate Manifest Dry Run
→ 2F Minimal Closed-loop Report
```

2A 只负责第一步：让 hidden states 能稳定、可复现地缓存下来，并提供足够的 token alignment metadata，供 2B 使用。

---

## 必读文件

Codex 执行前必须阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/codex_tasks/sprint_2A_hidden_state_cache_baseline.md
```

按需阅读：

```text
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/human_review_consolidation.py
tests/test_human_review_consolidation.py
```

不需要读取：

```text
docs/reference/*
```

除非当前实现中发现与方法边界有关的冲突，再向用户说明后等待确认。

---

## 输入文件

主输入：

```text
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

每行是一个 reviewed case，至少包含：

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

注意：

1. 2A 只读这个 manifest。
2. 不重新解析 `sprint_1Q_human_review_sheet.md`。
3. 不重新读取或回写 `sprint_1Q_human_review_labels_template.jsonl`。
4. 不重新读取或回写 `upgraded_downstream_report_with_human_fields.json`，除非只读校验时确实需要，且不得写入。

---

## 输出目录

新建输出目录：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/
```

输出文件：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/
```

hidden state tensor 文件保存到：

```text
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

JSONL 中不得直接嵌入 tensor，只能记录 tensor 文件路径和 metadata。

---

## 需要新增或修改的文件

允许新增或修改：

```text
docs/codex_tasks/sprint_2A_hidden_state_cache_baseline.md
src/recover_attention/token_alignment.py
src/recover_attention/hidden_state_cache.py
scripts/16_cache_hidden_states.py
tests/test_token_alignment.py
tests/test_hidden_state_cache.py
PROGRESS.md
docs/progress/sprint_2_history.md
```

如果确实需要更新 schema 校验函数，可小幅修改：

```text
src/recover_attention/schemas.py
tests/test_schemas.py
```

但必须保持小步修改，不得重构已有 schema 系统。

---

## 禁止修改的文件和目录

本 sprint 禁止修改：

```text
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/*
docs/reference/*
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
outputs/logs/sprint_1O_recovery_scoring/*
outputs/logs/sprint_1P_upgraded_downstream/*
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_sheet.md
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
outputs/logs/sprint_1Q_real_signal_quality_review/upgraded_downstream_report_with_human_fields.json
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_summary.json
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_known_issues.md
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

也禁止修改以下逻辑：

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

## 实现要求

### 1. Token alignment 模块

新增：

```text
src/recover_attention/token_alignment.py
```

至少实现以下能力：

#### 1.1 识别 masked question 中的 `[MASK]`

输入：

```text
masked_question
mask_token = "[MASK]"
```

输出：

```json
{
  "mask_char_ranges": [[start, end], ...],
  "num_masks": 1
}
```

要求：

1. 支持 single mask。
2. 支持 group mask，即多个 `[MASK]`。
3. 如果没有找到 `[MASK]`，返回 warning，不中断整个流程。

---

#### 1.2 original 与 masked 的基础差分对齐

实现一个轻量函数，根据 `original_question` 和 `masked_question` 推断被 `[MASK]` 替代的原始文本片段。

示例：

```text
Original:
Tom has 3 apples and buys 2 more. How many apples does he have now?

Masked:
Tom has [MASK] apples and buys 2 more. How many apples does he have now?
```

应推断：

```json
{
  "masked_original_spans": [
    {
      "text": "3",
      "original_char_range": [8, 9],
      "mask_index": 0
    }
  ]
}
```

group 示例：

```text
Original:
Chen reads 5 pages on Monday and 9 pages on Tuesday. How many more pages did Chen read on Tuesday?

Masked:
Chen reads 5 [MASK] on Monday and 9 [MASK] on Tuesday. How many more [MASK] did Chen read on Tuesday?
```

应推断出三个原始片段：

```text
pages
pages
pages
```

要求：

1. 优先使用简单 deterministic diff，不引入复杂依赖。
2. 如果对齐失败，记录 alignment warning。
3. 对齐失败不得导致整个脚本失败。
4. fragment recovery 不应导致 original/masked 对齐失败。

---

#### 1.3 recovered question 对齐

对每个 `recovered_question` 尝试用 `masked_question` 的模板定位 recovered fill spans。

示例：

```text
Masked:
Chen reads 5 [MASK] on Monday and 9 [MASK] on Tuesday. How many more [MASK] did Chen read on Tuesday?

Recovered:
Chen reads 5 books on Monday and 9 books on Tuesday. How many more books did Chen read on Tuesday?
```

应识别：

```text
books
books
books
```

如果 recovered output 是 fragment：

```text
pages
How many
gave
0
```

应记录：

```json
{
  "alignment_status": "failed_fragment_recovery"
}
```

不得中断脚本。

---

#### 1.4 tokenizer token position 对齐

对于每个输入文本，生成 tokenizer 级别的 token metadata。

要求支持：

```text
tokens
token_ids
offset_mapping 或等价 token_char_ranges
seq_len
```

如果使用真实 HuggingFace tokenizer，应尽量开启 offset mapping。

如果使用 stub backend，应提供 deterministic simple tokenizer，保证测试稳定。

---

### 2. Hidden state cache 模块

新增：

```text
src/recover_attention/hidden_state_cache.py
```

至少实现：

```text
build_hidden_state_cache_records(...)
cache_hidden_states_for_manifest(...)
write_hidden_state_manifest(...)
write_hidden_state_cache_report(...)
write_token_alignment_report(...)
```

核心行为：

1. 读取 `sprint_1Q_to_2A_manifest.jsonl`。
2. 每个 case 构造三类 input：

   * `original`
   * `masked`
   * `recovered`
3. 每个 recovered question 单独作为一个 `recovered` input。
4. 对每个 input 生成 hidden state tensor。
5. 将 hidden state tensor 保存为 `.pt`。
6. 将 tensor path 和 metadata 写入 `hidden_state_manifest.jsonl`。
7. 生成 `hidden_state_cache_report.json`。
8. 生成 `token_alignment_report.json`。

---

## input_type 约定

每个 case 至少包含：

```text
original
masked
recovered
```

对于多条 recovered question，使用：

```text
input_type = "recovered"
input_index = 0, 1, 2, ...
```

推荐 manifest 记录格式：

```json
{
  "cache_id": "gsm8k_0001__unit_001__mask__original__0",
  "masked_id": "gsm8k_0001__unit_001__mask",
  "id": "gsm8k_0001",
  "unit_id": "unit_001",
  "input_type": "original",
  "input_index": 0,
  "input_text": "Tom has 3 apples and buys 2 more. How many apples does he have now?",
  "backend": "stub_hidden_state_v0",
  "model_name": "stub_hidden_state_v0",
  "tokenizer_name": "stub_whitespace_tokenizer_v0",
  "layer_indices": [0, 1, 2],
  "seq_len": 14,
  "hidden_size": 8,
  "hidden_state_shape": [3, 14, 8],
  "hidden_state_path": "outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/gsm8k_0001__unit_001__mask__original__0.pt",
  "alignment_status": "ok",
  "alignment_warnings": [],
  "human_recoverability_label": "Misleading Recovery",
  "human_attention_anchor_label": "Risky Anchor",
  "human_semantic_role": "critical_number",
  "human_guidance_priority": "high",
  "human_error_type": "wrong_numeric_recovery",
  "probe_usage": "risk_positive"
}
```

---

## Backend 要求

### 1. 必须实现：`stub_hidden_state_v0`

这是默认 backend，用于测试和 Codex 验收。

要求：

1. 不调用真实模型。
2. 不下载模型。
3. 不调用外部 API。
4. 不使用 GPU。
5. deterministic。
6. 生成小型 tensor，例如 shape：

   ```text
   [num_layers, seq_len, hidden_size]
   ```
7. 默认：

   ```text
   hidden_size = 8
   layer_indices = [0, 1, 2]
   ```

stub tensor 可以根据 `token_ids`、`layer_index`、`position` 构造 deterministic 数值，确保每次运行结果稳定。

---

### 2. 可选实现：`hf_local_causal_lm_hidden_states_v0`

如果实现真实 backend，必须遵守以下边界：

1. 只允许使用用户显式传入的本地 `--model-path`。
2. 不允许自动下载模型。
3. 不允许访问外部网络。
4. 不在 pytest 中运行真实 backend。
5. 不在默认 Codex 验收命令中运行真实 backend。
6. 如果本地模型路径不存在，应给出清晰错误。
7. 真实 backend 仅作为用户手动命令使用。

真实 backend 可使用 HuggingFace `AutoTokenizer` / `AutoModelForCausalLM`，并设置：

```text
output_hidden_states=True
use_cache=False
```

但本 sprint 的必需验收只依赖 `stub_hidden_state_v0`。

---

## CLI 脚本

新增：

```text
scripts/16_cache_hidden_states.py
```

必须支持参数：

```text
--input
--output-dir
--backend
--layer-indices
--hidden-size
--mask-token
--overwrite
```

如果实现真实 backend，再支持：

```text
--model-path
--device
--dtype
--max-length
```

默认安全运行命令：

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_hidden_state_cache_baseline `
  --backend stub_hidden_state_v0 `
  --layer-indices 0 1 2 `
  --hidden-size 8 `
  --mask-token "[MASK]" `
  --overwrite
```

真实 backend 命令只写入文档或 help，不作为默认验收命令执行：

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_hidden_state_cache_real_qwen `
  --backend hf_local_causal_lm_hidden_states_v0 `
  --model-path D:/models/Qwen2.5-7B-Instruct `
  --layer-indices 0 8 16 24 27 `
  --device cuda `
  --dtype float16 `
  --max-length 512 `
  --mask-token "[MASK]" `
  --overwrite
```

只有用户明确确认时，才允许运行真实 backend 命令。

---

## Report 要求

### 1. `hidden_state_cache_report.json`

至少包含：

```json
{
  "input_path": "...",
  "output_dir": "...",
  "backend": "stub_hidden_state_v0",
  "num_cases": 20,
  "num_inputs_total": 60,
  "input_type_counts": {
    "original": 20,
    "masked": 20,
    "recovered": 20
  },
  "num_hidden_state_files": 60,
  "layer_indices": [0, 1, 2],
  "hidden_size": 8,
  "seq_len_summary": {
    "min": 1,
    "max": 25,
    "mean": 14.2
  },
  "alignment_status_counts": {
    "ok": 54,
    "warning": 6
  },
  "human_recoverability_label_counts": {},
  "human_error_type_counts": {},
  "probe_usage_counts": {}
}
```

具体数字以实际运行结果为准。

---

### 2. `token_alignment_report.json`

至少包含：

```json
{
  "num_cases": 20,
  "num_masks_total": 0,
  "num_single_mask_cases": 0,
  "num_group_mask_cases": 0,
  "num_original_masked_span_alignment_ok": 0,
  "num_original_masked_span_alignment_failed": 0,
  "num_recovered_alignment_ok": 0,
  "num_recovered_alignment_failed": 0,
  "num_fragment_recovery_outputs": 0,
  "alignment_warning_count": 0,
  "warnings_by_type": {},
  "records_with_warnings": []
}
```

`records_with_warnings` 可只记录摘要，不要嵌入 tensor。

---

## 测试要求

新增测试：

```text
tests/test_token_alignment.py
tests/test_hidden_state_cache.py
```

至少覆盖：

### token alignment 测试

1. single mask char range。
2. group mask char ranges。
3. original vs masked 单 span 差分。
4. original vs masked group span 差分。
5. recovered full-question fill span 对齐。
6. recovered fragment 输出返回 warning，不抛异常。
7. `[MASK]` 被 tokenizer 拆分时，不假设一个 mask 等于一个 token。
8. 没有 `[MASK]` 时返回 warning。

### hidden state cache 测试

1. 能读取一个小型 2A manifest。
2. stub backend 生成 deterministic tensor。
3. tensor shape 符合 `[num_layers, seq_len, hidden_size]`。
4. 生成 hidden_state_manifest.jsonl。
5. 生成 hidden_state_cache_report.json。
6. 生成 token_alignment_report.json。
7. 每条 manifest 记录包含 hidden_state_path。
8. fragment recovery 不导致失败。
9. 不修改输入 manifest。
10. CLI smoke test 可以运行。

---

## 明确不做

本 sprint 不做：

```text
1. 不训练 probe。
2. 不做 representation feature extraction。
3. 不做 hidden-state distance analysis。
4. 不做 PCA / UMAP / t-SNE。
5. 不做 attention guidance。
6. 不修改 transformer attention。
7. 不重新生成 answers。
8. 不重跑 CoT。
9. 不评估 answer accuracy。
10. 不修改 recovery prompt。
11. 不修改 recovery scoring。
12. 不修改 NLI scoring。
13. 不修改 masked question construction。
14. 不实现 unit/group mask budget selector。
15. 不修改 1Q / 1R 输入产物。
```

---

## 必须运行的命令

先运行默认 stub cache：

```powershell
conda run -n recover_attention python scripts/16_cache_hidden_states.py `
  --input outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl `
  --output-dir outputs/logs/sprint_2A_hidden_state_cache_baseline `
  --backend stub_hidden_state_v0 `
  --layer-indices 0 1 2 `
  --hidden-size 8 `
  --mask-token "[MASK]" `
  --overwrite
```

再运行测试：

```powershell
conda run -n recover_attention python -m pytest -q
```

最后检查改动：

```powershell
git diff --name-only
git status --short
```

不得运行真实模型 backend，除非用户单独确认。

---

## 验收标准

1. `scripts/16_cache_hidden_states.py` 可以读取 `sprint_1Q_to_2A_manifest.jsonl`。
2. `stub_hidden_state_v0` 成功为 20 个 reviewed cases 生成 hidden state cache。
3. 每个 reviewed case 至少生成：

   * original input cache
   * masked input cache
   * recovered input cache
4. 生成 `hidden_state_manifest.jsonl`。
5. 生成 `hidden_state_cache_report.json`。
6. 生成 `token_alignment_report.json`。
7. hidden state tensor 保存为 `.pt`，不嵌入 JSONL。
8. manifest 中每条记录都包含：

   ```text
   cache_id
   masked_id
   id
   unit_id
   input_type
   input_index
   backend
   model_name
   tokenizer_name
   layer_indices
   seq_len
   hidden_size
   hidden_state_shape
   hidden_state_path
   alignment_status
   human_recoverability_label
   human_attention_anchor_label
   human_error_type
   probe_usage
   ```
9. token alignment 能识别 single mask 和 group mask。
10. recovered fragment 输出只产生 warning，不导致脚本失败。
11. 不修改 1Q / 1R 输入文件。
12. 不修改 data/processed。
13. 不运行 Ollama、NLI、真实 HF 模型或 GPU 推理。
14. `pytest` 通过。
15. `PROGRESS.md` 更新当前阶段为 Sprint 2A completed，并将下一步标记为 Sprint 2B Representation Feature Extraction。
16. `docs/progress/sprint_2_history.md` 记录 Sprint 2A 的输入、输出、运行命令、检查结果和遗留问题。

---

## 完成后 PROGRESS.md 更新建议

```text
当前阶段：Sprint 2A 已完成：Hidden State Cache Baseline。

已完成：
- 读取 Sprint 1R 生成的 sprint_1Q_to_2A_manifest.jsonl。
- 使用 stub_hidden_state_v0 为 original / masked / recovered 输入生成 hidden state cache。
- 生成 hidden_state_manifest.jsonl。
- 生成 hidden_state_cache_report.json。
- 生成 token_alignment_report.json。
- 完成 single / group mask 的基础 token alignment。
- fragment recovery 输出不会中断流程，而是记录 alignment warning。

特别说明：
- 本 sprint 使用 stub backend 完成 cache pipeline 验证。
- 本 sprint 未运行真实 HuggingFace 模型。
- 本 sprint 未缓存真实模型 hidden states。
- 本 sprint 未训练 probe。
- 本 sprint 未做 representation analysis。
- 本 sprint 未修改 1Q / 1R 输入文件。

下一步：Sprint 2B：Representation Feature Extraction。
```

---

## 完成后 docs/progress/sprint_2_history.md 记录建议

新增章节：

```text
## Sprint 2A：Hidden State Cache Baseline

已完成内容：
- 实现 token alignment baseline。
- 实现 hidden state cache baseline。
- 实现 stub_hidden_state_v0。
- 实现 scripts/16_cache_hidden_states.py。
- 生成 Sprint 2A cache 输出目录。
- 新增 token alignment 和 hidden state cache pytest。

输入文件：
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl

输出文件：
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_manifest.jsonl
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_state_cache_report.json
- outputs/logs/sprint_2A_hidden_state_cache_baseline/token_alignment_report.json
- outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt

运行命令：
- conda run -n recover_attention python scripts/16_cache_hidden_states.py ...
- conda run -n recover_attention python -m pytest -q

检查结果：
- cache command: passed
- pytest: passed

遗留问题：
- 当前 hidden states 来自 stub backend，不是真实模型。
- 当前 token alignment 是基础 deterministic 对齐，不处理复杂 paraphrase。
- recovered fragment 输出只记录 warning，后续在 2B/2C 判断是否保留。
- 真实 HF hidden state backend 需要用户单独确认后运行。

下一步建议：
- Sprint 2B：Representation Feature Extraction。
```
