# Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs

建议保存路径：

```text id="l3bhdv"
docs/codex_tasks/sprint_1N_rebuild_downstream_real_signals.md
```

---

## 1. 目标

本 sprint 使用已经接入的真实信号重建 downstream：

```text id="fkg42g"
real NLI scores
+
real LLM recovery outputs
→ semantic_labels
→ recover_scores
→ unit_evidence
→ attention_anchor_labels
→ intervention_manifest
→ real signal report
```

当前已经完成：

```text id="3orya0"
Sprint 1L：Plug Real Bilingual NLI Backend
Sprint 1M：Plug Real LLM Recovery Backend
```

其中：

```text id="wbntod"
1. scripts/05_run_nli_scoring.py 已支持 hf_nli_auto_v0。
2. scripts/08_run_recovery.py 已支持 ollama_chat_v0。
3. qwen3.5:9b real smoke 已通过。
```

但当前主链路中的 downstream 仍主要来自 stub baseline：

```text id="a4b6qd"
data/processed/nli_scores.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/unit_evidence.jsonl
data/processed/attention_anchor_labels.jsonl
data/processed/intervention_manifest.jsonl
```

本 sprint 的目标不是改算法，而是：

```text id="cod250"
在不污染 data/processed baseline 的前提下，
用真实 NLI 与真实 LLM recovery 生成一套 isolated downstream outputs，
并报告真实信号经过现有规则后的分布变化。
```

---

## 2. 本 sprint 的核心原则

```text id="z0f8ry"
Do not overwrite stable stub baseline.
```

也就是说，本 sprint 默认不覆盖：

```text id="i05lsr"
data/processed/*.jsonl
```

所有真实信号重建产物必须输出到：

```text id="jqbkja"
outputs/logs/sprint_1N_real_downstream/
```

或等价的 isolated output directory。

---

## 3. 本 sprint 做什么

本 sprint 应完成以下事情：

```text id="u1uxd8"
1. 使用 hf_nli_auto_v0 生成 real NLI scores。
2. 使用 real NLI scores 重建 semantic_labels。
3. 使用 real semantic_labels 重建 masked_questions。
4. 使用 ollama_chat_v0 + qwen3.5:9b 生成 real LLM recovery outputs。
5. 使用现有 stub_rule_v0 暂时评分 real recovery outputs。
6. 使用 real semantic_labels + real recovery_scores 重建 unit_evidence。
7. 使用 real unit_evidence 重建 attention_anchor_labels。
8. 使用 real attention_anchor_labels 重建 intervention_manifest。
9. 生成 real_signal_report.json。
10. 更新 PROGRESS.md 与 docs/progress/sprint_1_history.md。
```

---

## 4. 本 sprint 不做什么

本 sprint 禁止做：

```text id="kv1y9r"
1. 不修改 recover_scoring.py 的评分逻辑。
2. 不新增 LLM judge。
3. 不新增 semantic similarity scorer。
4. 不修改 semantic_labels.py 的规则。
5. 不修改 unit_evidence.py 的聚合规则。
6. 不修改 attention_anchor_labels.py 的标签规则。
7. 不修改 intervention_manifest.py 的 manifest 规则。
8. 不接入 hidden states。
9. 不接入 attention maps。
10. 不接入 trajectory stability。
11. 不接入 answer stability。
12. 不接入 attention guidance。
13. 不做 probe training。
14. 不做目录重构。
15. 不覆盖 data/processed baseline。
```

注意：

```text id="qtyfhf"
stub_rule_v0 仍然只是 exact normalized match。
本 sprint 使用它只是为了让 real recovery output 可以进入现有 downstream。
不要把该分数解释为真实语义 recoverability。
```

---

## 5. 输入文件

本 sprint 输入来自当前已有 pipeline：

```text id="cxn1t7"
data/processed/ablated_questions.jsonl
data/processed/questions.jsonl
```

实际链路主要从：

```text id="6ztlyn"
data/processed/ablated_questions.jsonl
```

开始。

不要重建 candidate span / ablation unit，除非输入文件缺失。

---

## 6. 输出目录

新增 isolated output directory：

```text id="e0xqoc"
outputs/logs/sprint_1N_real_downstream/
```

该目录下生成：

```text id="iw782g"
nli_scores_real.jsonl
semantic_labels_real.jsonl
masked_questions_real.jsonl
recover_outputs_real.jsonl
recover_scores_real.jsonl
unit_evidence_real.jsonl
attention_anchor_labels_real.jsonl
intervention_manifest_real.jsonl
real_signal_report.json
```

可选生成：

```text id="b0ec7v"
real_signal_report.md
```

不要输出到：

```text id="8ws8an"
data/processed/nli_scores.jsonl
data/processed/semantic_labels.jsonl
data/processed/masked_questions.jsonl
data/processed/recover_outputs.jsonl
data/processed/recover_scores.jsonl
data/processed/unit_evidence.jsonl
data/processed/attention_anchor_labels.jsonl
data/processed/intervention_manifest.jsonl
```

---

## 7. 允许修改

本 sprint 允许修改：

```text id="9u3xbo"
scripts/13_rebuild_downstream_real_signals.py
tests/test_rebuild_downstream_real_signals.py
PROGRESS.md
docs/progress/sprint_1_history.md
configs/v0_nli_small.yaml
```

如果 `scripts/13_rebuild_downstream_real_signals.py` 不存在，可以新增。

如果 `tests/test_rebuild_downstream_real_signals.py` 不存在，可以新增。

允许由运行命令生成：

```text id="3k035g"
outputs/logs/sprint_1N_real_downstream/nli_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/semantic_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/masked_questions_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/unit_evidence_real.jsonl
outputs/logs/sprint_1N_real_downstream/attention_anchor_labels_real.jsonl
outputs/logs/sprint_1N_real_downstream/intervention_manifest_real.jsonl
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
outputs/logs/sprint_1N_real_downstream/real_signal_report.md
```

---

## 8. 禁止修改

本 sprint 禁止修改：

```text id="lwmfs6"
README.md
AGENTS.md
docs/skill/SKILL.md
docs/skill/codex_tasks.md
docs/skill/experiment_guide.md
docs/skill/method.md
docs/skill/label_schema.md
docs/skill/nli_scores_interface.md
docs/skill/semantic_labels_interface.md
docs/skill/masked_questions_interface.md
docs/skill/recover_outputs_interface.md
docs/skill/recover_scores_interface.md
docs/skill/unit_evidence_interface.md
docs/skill/attention_anchor_labels_interface.md
docs/skill/intervention_manifest_interface.md
docs/reference/*
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
src/recover_attention/schemas.py
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
tests/test_nli_scoring.py
tests/test_semantic_labels.py
tests/test_masked_questions.py
tests/test_recover_generation.py
tests/test_recover_scoring.py
tests/test_unit_evidence.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
data/processed/*
models/*
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因。

---

## 9. 开始前必须读取

开始前必须读取：

```text id="ooav4p"
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/skill/codex_tasks.md
configs/v0_nli_small.yaml
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
src/recover_attention/data_io.py
data/processed/ablated_questions.jsonl
data/processed/questions.jsonl
docs/progress/sprint_1_history.md
```

为了理解字段，不需要修改但可以读取：

```text id="toh5rr"
docs/skill/nli_scores_interface.md
docs/skill/semantic_labels_interface.md
docs/skill/masked_questions_interface.md
docs/skill/recover_outputs_interface.md
docs/skill/recover_scores_interface.md
docs/skill/unit_evidence_interface.md
docs/skill/attention_anchor_labels_interface.md
docs/skill/intervention_manifest_interface.md
```

不要读取：

```text id="ednlye"
docs/reference/*
```

除非用户另行明确要求。

---

## 10. Preflight 要求

修改任何文件前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text id="ccyhwl"
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1M 已完成。
6. 当前 scripts/05_run_nli_scoring.py 是否支持 hf_nli_auto_v0。
7. 当前 scripts/08_run_recovery.py 是否支持 ollama_chat_v0。
8. 当前 data/processed/ablated_questions.jsonl 是否存在。
9. ablated_questions record 数量。
10. 当前 data/processed/questions.jsonl 是否存在。
11. 当前 models/nli/en/roberta-large-mnli 是否存在。
12. 当前 models/nli/zh/mdeberta-v3-base-xnli 是否存在。
13. 当前机器是否能执行 ollama list。
14. ollama list 中是否存在 qwen3.5:9b。
15. ollama list 中是否存在 llama3.1:8b。
16. 本 sprint 实际使用的 Ollama model name。
17. 当前裸 python 路径与版本。
18. 当前 recover_attention 环境 python 路径与版本。
19. 本 sprint 实际运行命令是否全部使用 conda run -n recover_attention python。
20. 本次是否会覆盖 data/processed/*.jsonl。
21. 本次是否会修改 scoring 规则。
22. 本次是否会修改 schema。
23. 本次是否会接入 attention guidance。
24. 本次是否会使用 qwen3.5:9b 生成 real recovery。
25. 本次输出目录。
26. 是否发现冲突。
```

第 19 项必须回答：

```text id="lm62e1"
是，全部使用 conda run -n recover_attention python。
```

第 20 项必须回答：

```text id="j2x4x3"
否。
```

第 21 项必须回答：

```text id="vcnlnn"
否。
```

第 22 项必须回答：

```text id="ssxe88"
否。
```

第 23 项必须回答：

```text id="td88wf"
否。
```

第 24 项如果 `qwen3.5:9b` 存在，必须回答：

```text id="rxrzbo"
是。
```

---

## 11. 新增 orchestration script

新增：

```text id="ryj43x"
scripts/13_rebuild_downstream_real_signals.py
```

该脚本的职责是 orchestration，不是新增算法。

它应调用现有 Python module function，或复用现有 CLI 等价逻辑，按顺序生成 isolated outputs。

### 11.1 CLI 参数

建议支持：

```text id="vobl1r"
--ablated-questions
--output-dir
--nli-backend
--language
--en-model
--zh-model
--allow-download
--semantic-backend
--equivalent-threshold
--directional-entailment-threshold
--contradiction-threshold
--mask-token
--mask-backend
--recovery-backend
--ollama-model
--ollama-base-url
--num-samples
--temperature
--top-p
--max-tokens
--timeout
--seed
--recover-score-backend
--unit-evidence-backend
--attention-label-backend
--intervention-type
--intervention-backend
--limit
--skip-ollama
```

### 11.2 默认值

默认值：

```text id="diw9dd"
--ablated-questions data/processed/ablated_questions.jsonl
--output-dir outputs/logs/sprint_1N_real_downstream
--nli-backend hf_nli_auto_v0
--language auto
--en-model models/nli/en/roberta-large-mnli
--zh-model models/nli/zh/mdeberta-v3-base-xnli
--allow-download False
--semantic-backend rule_v0
--equivalent-threshold 0.70
--directional-entailment-threshold 0.50
--contradiction-threshold 0.50
--mask-token [MASK]
--mask-backend unit_mask_v0
--recovery-backend ollama_chat_v0
--ollama-model qwen3.5:9b
--ollama-base-url http://localhost:11434
--num-samples 1
--temperature 0.0
--top-p 1.0
--max-tokens 128
--timeout 120
--seed 42
--recover-score-backend stub_rule_v0
--unit-evidence-backend aggregate_stub_v0
--attention-label-backend early_evidence_rule_stub_v0
--intervention-type mask
--intervention-backend manifest_stub_v0
--limit None
--skip-ollama False
```

### 11.3 limit 语义

`--limit` 只用于小样本调试。

要求：

```text id="ivjhus"
1. 如果传入 --limit N，只截断 ablated_questions 输入前 N 条。
2. downstream 全部基于这 N 条继续生成。
3. 不修改 library 层默认行为。
```

### 11.4 skip-ollama 语义

允许提供：

```text id="ipyviv"
--skip-ollama
```

用于 CI / pytest。

语义：

```text id="oemsgw"
1. 只在测试或无法连接 Ollama 时使用。
2. skip-ollama=True 时，不运行 ollama_chat_v0。
3. 可以只生成到 masked_questions_real.jsonl，然后生成一个 report 标记 recovery_skipped=true。
4. 真实 sprint 验收时如果 Ollama 可用，不应使用 --skip-ollama。
```

---

## 12. Orchestration pipeline

脚本必须按以下顺序执行。

### Step 1：real NLI scores

输入：

```text id="i1m2rf"
data/processed/ablated_questions.jsonl
```

输出：

```text id="kq9iig"
outputs/logs/sprint_1N_real_downstream/nli_scores_real.jsonl
```

backend：

```text id="ntiyzt"
hf_nli_auto_v0
```

模型路径：

```text id="ghk8l3"
models/nli/en/roberta-large-mnli
models/nli/zh/mdeberta-v3-base-xnli
```

默认不允许自动下载：

```text id="1b911y"
allow_download = False
```

### Step 2：real semantic labels

输入：

```text id="n25yw6"
nli_scores_real.jsonl
```

输出：

```text id="zlyl9k"
semantic_labels_real.jsonl
```

backend：

```text id="dts4qh"
rule_v0
```

阈值沿用当前 pipeline：

```text id="hn9wvf"
equivalent_threshold = 0.70
directional_entailment_threshold = 0.50
contradiction_threshold = 0.50
```

### Step 3：real masked questions

输入：

```text id="l3dfe1"
semantic_labels_real.jsonl
```

输出：

```text id="7s47r3"
masked_questions_real.jsonl
```

backend：

```text id="jzjho1"
unit_mask_v0
```

mask token：

```text id="kclw7j"
[MASK]
```

### Step 4：real LLM recovery outputs

输入：

```text id="cnms0r"
masked_questions_real.jsonl
```

输出：

```text id="9upxel"
recover_outputs_real.jsonl
```

backend：

```text id="nidxfd"
ollama_chat_v0
```

model：

```text id="pjq3jp"
qwen3.5:9b
```

decoding：

```text id="tchla5"
temperature = 0.0
top_p = 1.0
max_tokens = 128
timeout = 120
seed = 42
num_samples = 1
```

如果 `qwen3.5:9b` 不可用但 `llama3.1:8b` 可用，允许手动改为：

```text id="hklnoh"
llama3.1:8b
```

但必须在 final report 中记录 fallback。

不要自动运行：

```text id="cekozx"
ollama pull
```

### Step 5：real recovery scores with existing scorer

输入：

```text id="gt4pgo"
recover_outputs_real.jsonl
```

输出：

```text id="i0p8cl"
recover_scores_real.jsonl
```

backend：

```text id="5h74eg"
stub_rule_v0
```

注意：

```text id="cwo7cv"
这是临时评分，只是为了让 real recovery output 进入现有 downstream。
它不是语义 recoverability。
```

### Step 6：real unit evidence

输入：

```text id="bu4wyp"
semantic_labels_real.jsonl
recover_scores_real.jsonl
```

输出：

```text id="00q3vk"
unit_evidence_real.jsonl
```

backend：

```text id="ekdmsj"
aggregate_stub_v0
```

### Step 7：real attention anchor labels

输入：

```text id="hha1cv"
unit_evidence_real.jsonl
```

输出：

```text id="t87zq6"
attention_anchor_labels_real.jsonl
```

backend：

```text id="kay2qx"
early_evidence_rule_stub_v0
```

### Step 8：real intervention manifest

输入：

```text id="1udf4n"
attention_anchor_labels_real.jsonl
```

输出：

```text id="vk1pvx"
intervention_manifest_real.jsonl
```

backend：

```text id="kss3dd"
manifest_stub_v0
```

intervention type：

```text id="0ijnx6"
mask
```

---

## 13. Report 要求

新增输出：

```text id="og52tk"
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
```

报告必须至少包含：

```text id="rljlrw"
run_metadata
input_counts
output_counts
nli_backend_counts
language_counts
semantic_necessity_label_counts
is_semantically_necessary_counts
recoverability_label_counts
misleading_recovery_counts
attention_anchor_label_counts
intervention_type_counts
empty_recovery_count
mask_remaining_count
exact_match_recovery_count
sample_records
known_limitations
next_step_recommendation
```

### 13.1 run_metadata

包含：

```text id="y9rwbr"
timestamp
output_dir
nli_backend
language
en_model
zh_model
recovery_backend
ollama_model
ollama_base_url
num_samples
temperature
top_p
max_tokens
seed
limit
```

### 13.2 input_counts

包含：

```text id="ztpoj7"
num_ablated_questions
```

### 13.3 output_counts

包含每个 stage 的 record 数：

```text id="usxyfl"
num_nli_scores
num_semantic_labels
num_masked_questions
num_recover_outputs
num_recover_scores
num_unit_evidence
num_attention_anchor_labels
num_intervention_manifest
```

### 13.4 quality proxy

至少统计：

```text id="ksg1fq"
empty_recovery_count:
  recovered_question == ""

mask_remaining_count:
  recovered_question 中仍包含 mask_token 的数量。

exact_match_recovery_count:
  normalized recovered_question == normalized original_question 的数量。
```

注意：

```text id="ogzc7y"
exact_match_recovery_count 不是唯一质量指标。
在当前 stub_rule_v0 下，它只是一个 conservative proxy。
```

### 13.5 sample_records

抽样保存最多 10 条 record summary：

```text id="bmkcgp"
masked_id
id
unit_id
masked_question
recovered_question
recoverability_label
recoverability_score
semantic_necessity_label
attention_anchor_label
attention_importance_score
```

不要保存完整 prompt。

不要保存 raw Ollama response。

不要保存 hidden states / attention maps。

### 13.6 known_limitations

必须写入：

```text id="1t8xrg"
1. recover_scores_real 仍由 stub_rule_v0 exact normalized match 生成。
2. real LLM recovery output 可能不是 exact original question，但仍可能语义可恢复。
3. 当前 report 不能证明 attention guidance 有效。
4. 当前未接入 trajectory stability、answer stability、raw attention pattern。
5. 当前 intervention_manifest_real 仍是 planned_only，不代表 intervention 已执行。
```

### 13.7 next_step_recommendation

推荐：

```text id="rwlok8"
Sprint 1O：Upgrade Real Recovery Scoring
```

或者如果报告显示 exact match 已经足够可分析，则推荐：

```text id="k883zv"
Sprint 1O：Real Signal Quality Review
```

默认建议：

```text id="ey6e8s"
Sprint 1O：Upgrade Real Recovery Scoring
```

---

## 14. 测试要求

新增：

```text id="dugew1"
tests/test_rebuild_downstream_real_signals.py
```

测试不能调用真实 NLI 模型，也不能调用真实 Ollama。

必须用 tmp_path + monkeypatch / fake functions。

至少覆盖：

```text id="n19l3w"
1. CLI --limit 只处理前 N 条 ablated_questions。
2. output_dir 下生成预期文件名。
3. report.json 包含必需 top-level keys。
4. report.json 中 output_counts 与 fake records 数量一致。
5. report.json 中 known_limitations 包含 stub_rule_v0 exact match limitation。
6. script 不写入 data/processed。
7. script 不修改 schema。
8. --skip-ollama 时 report 中 recovery_skipped=true。
9. normal fake pipeline 中 recovery_skipped=false。
10. sample_records 最多 10 条。
11. empty_recovery_count 统计正确。
12. mask_remaining_count 统计正确。
13. exact_match_recovery_count 统计正确。
```

如果直接复用真实 module function 不方便 monkeypatch，可以将 orchestration 拆成小函数：

```python id="mfl2mr"
build_real_downstream_paths(output_dir: Path) -> dict[str, Path]
build_real_signal_report(...)
normalize_text_for_report(...)
count_records(...)
```

这些函数应容易单测。

---

## 15. 必须运行命令

所有命令必须使用：

```text id="d6yjcd"
conda run -n recover_attention python ...
```

不要使用裸 `python`。

### 15.1 基础测试

```powershell id="5fww4t"
conda run -n recover_attention python scripts/sync_interface_fields.py --check

conda run -n recover_attention python -m pytest tests/test_rebuild_downstream_real_signals.py -q

conda run -n recover_attention python -m pytest tests/test_schemas.py -q

conda run -n recover_attention python -m pytest -q
```

### 15.2 小样本 dry run

先运行 limit 10：

```powershell id="pfmhvt"
conda run -n recover_attention python scripts/13_rebuild_downstream_real_signals.py `
  --ablated-questions data/processed/ablated_questions.jsonl `
  --output-dir outputs/logs/sprint_1N_real_downstream `
  --nli-backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --recovery-backend ollama_chat_v0 `
  --ollama-model qwen3.5:9b `
  --ollama-base-url http://localhost:11434 `
  --num-samples 1 `
  --temperature 0.0 `
  --top-p 1.0 `
  --max-tokens 128 `
  --timeout 120 `
  --seed 42 `
  --limit 10
```

### 15.3 全量 run

如果 limit 10 通过，运行不带 limit 的全量版本：

```powershell id="gcieqw"
conda run -n recover_attention python scripts/13_rebuild_downstream_real_signals.py `
  --ablated-questions data/processed/ablated_questions.jsonl `
  --output-dir outputs/logs/sprint_1N_real_downstream `
  --nli-backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --recovery-backend ollama_chat_v0 `
  --ollama-model qwen3.5:9b `
  --ollama-base-url http://localhost:11434 `
  --num-samples 1 `
  --temperature 0.0 `
  --top-p 1.0 `
  --max-tokens 128 `
  --timeout 120 `
  --seed 42
```

如果全量 run 失败，最终回复必须说明：

```text id="loykvj"
1. 失败阶段。
2. 错误信息。
3. 已成功生成到哪个阶段。
4. 是否保留 limit 10 的产物。
5. 是否需要下一个 sprint 修复。
```

---

## 16. configs 更新要求

允许更新：

```text id="3at6mu"
configs/v0_nli_small.yaml
```

追加：

```yaml id="qn76nt"
real_downstream:
  output_dir: "outputs/logs/sprint_1N_real_downstream"
  nli:
    backend: "hf_nli_auto_v0"
    language: "auto"
    en_model: "models/nli/en/roberta-large-mnli"
    zh_model: "models/nli/zh/mdeberta-v3-base-xnli"
    allow_download: false
  semantic_labels:
    backend: "rule_v0"
    equivalent_threshold: 0.70
    directional_entailment_threshold: 0.50
    contradiction_threshold: 0.50
  masking:
    backend: "unit_mask_v0"
    mask_token: "[MASK]"
  recovery:
    backend: "ollama_chat_v0"
    model: "qwen3.5:9b"
    fallback_model: "llama3.1:8b"
    ollama_base_url: "http://localhost:11434"
    num_samples: 1
    temperature: 0.0
    top_p: 1.0
    max_tokens: 128
    timeout: 120
    seed: 42
  recover_scoring:
    backend: "stub_rule_v0"
  unit_evidence:
    backend: "aggregate_stub_v0"
  attention_anchor_labels:
    backend: "early_evidence_rule_stub_v0"
  intervention_manifest:
    intervention_type: "mask"
    backend: "manifest_stub_v0"
```

不要让 `scripts/00_smoke_test.py` 自动运行 real downstream。

---

## 17. PROGRESS.md 更新要求

更新：

```text id="kz07l4"
PROGRESS.md
```

要求：

```text id="2pd17h"
1. 当前阶段更新为 Sprint 1N 已完成：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs。
2. 已完成 Sprint 摘要中新增 Sprint 1N。
3. 当前可运行命令中新增 scripts/13_rebuild_downstream_real_signals.py 命令。
4. 最近一次检查结果中记录：
   real downstream rebuild orchestration: passed
   real NLI downstream rebuild: passed
   real LLM recovery downstream rebuild: passed
   real signal report: passed
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中新增：
   scripts/13_rebuild_downstream_real_signals.py
   tests/test_rebuild_downstream_real_signals.py
   outputs/logs/sprint_1N_real_downstream/*
6. 遗留问题中说明：
   - recover_scores_real 仍由 stub_rule_v0 exact normalized match 生成。
   - real recovery output 可能语义正确但 exact match 失败。
   - attention_anchor_labels_real 仍由 early_evidence_rule_stub_v0 生成，不代表 final attention importance。
   - intervention_manifest_real 仍为 planned_only，不代表 intervention 已执行。
   - 当前未接入 hidden states / attention maps / trajectory stability / attention guidance。
7. 下一步建议：
   Sprint 1O：Upgrade Real Recovery Scoring
```

如果全量 run 未成功，只完成了 limit 10，则不要写“全量 passed”，应写：

```text id="df933c"
real downstream small run: passed
full run: failed / skipped，原因是 ...
```

---

## 18. docs/progress/sprint_1_history.md 更新要求

更新：

```text id="y3oowf"
docs/progress/sprint_1_history.md
```

追加：

```text id="dq1gdh"
## Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs
```

内容包括：

```text id="h17y7e"
1. 已完成内容。
2. 新增/修改文件。
3. 输出目录。
4. 使用的真实 NLI backend 与模型路径。
5. 使用的真实 recovery backend 与 Ollama model。
6. 是否完成 limit 10 dry run。
7. 是否完成 full run。
8. report 关键统计摘要。
9. 检查结果。
10. 遗留问题。
11. 下一步建议：Sprint 1O：Upgrade Real Recovery Scoring。
```

---

## 19. 验收标准

本 sprint 完成后必须满足：

```text id="pouqqo"
1. 没有覆盖 data/processed/*.jsonl。
2. outputs/logs/sprint_1N_real_downstream/ 已生成。
3. nli_scores_real.jsonl 已生成并通过 nli_score schema。
4. semantic_labels_real.jsonl 已生成并通过 semantic_label schema。
5. masked_questions_real.jsonl 已生成并通过 masked_question schema。
6. recover_outputs_real.jsonl 已生成并通过 recover_output schema。
7. recover_scores_real.jsonl 已生成并通过 recover_score schema。
8. unit_evidence_real.jsonl 已生成并通过 unit_evidence schema。
9. attention_anchor_labels_real.jsonl 已生成并通过 attention_anchor_label schema。
10. intervention_manifest_real.jsonl 已生成并通过 intervention_manifest schema。
11. real_signal_report.json 已生成。
12. report 包含 label distributions。
13. report 包含 empty_recovery_count / mask_remaining_count / exact_match_recovery_count。
14. report 明确说明 stub_rule_v0 exact match limitation。
15. qwen3.5:9b 已用于 real recovery，或明确说明 fallback / 失败原因。
16. tests/test_rebuild_downstream_real_signals.py 通过。
17. tests/test_schemas.py 通过。
18. 全量 pytest 通过。
19. sync_interface_fields.py --check 通过。
20. 没有修改 schema。
21. 没有修改 scoring 规则。
22. 没有修改 existing backend implementation。
23. 没有接入 attention guidance。
24. 没有接入 hidden states / attention maps / trajectory stability。
25. PROGRESS.md 已更新。
26. docs/progress/sprint_1_history.md 已更新。
```

---

## 20. 完成后回复格式

完成后请按以下格式回复：

```text id="np60g5"
1. 本次完成内容
2. 新增/修改文件
3. 输出目录
4. 使用的真实 NLI backend 与模型路径
5. 使用的真实 recovery backend 与 Ollama model
6. 运行命令
7. dry run / full run 状态
8. real_signal_report.json 关键统计
9. 检查结果
10. PROGRESS.md 更新摘要
11. docs/progress/sprint_1_history.md 更新摘要
12. 遗留问题
13. 下一步建议
```

下一步建议默认是：

```text id="wh202c"
Sprint 1O：Upgrade Real Recovery Scoring
```

不要自动开始 Sprint 1O。
