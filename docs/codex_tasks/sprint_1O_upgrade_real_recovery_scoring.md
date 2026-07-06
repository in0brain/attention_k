# Sprint 1O：Upgrade Real Recovery Scoring

建议保存路径：

```text id="e25luu"
docs/codex_tasks/sprint_1O_upgrade_real_recovery_scoring.md
```

---

## 1. 目标

本 sprint 升级真实 recovery output 的评分方式。

当前已经完成：

```text id="s5p6qn"
Sprint 1L：Plug Real Bilingual NLI Backend
Sprint 1M：Plug Real LLM Recovery Backend
Sprint 1N：Rebuild Downstream with Real NLI and Real LLM Recovery Outputs
```

当前真实 downstream 已经生成：

```text id="ny2sbc"
outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
```

但当前 `recover_scores_real.jsonl` 仍由：

```text id="9t1066"
stub_rule_v0
```

生成。

`stub_rule_v0` 只做 exact normalized match：

```text id="q0ph2f"
normalized recovered_question == normalized original_question
```

这个规则太硬，会把很多语义正确但表述不同的恢复结果判成不可恢复。

本 sprint 的目标是新增一个真实 recovery scoring backend：

```text id="psm1v2"
nli_recovery_judge_v0
```

用 NLI 判断：

```text id="8vbprd"
recovered_question 是否语义恢复了 original_question 中被 mask 的信息
```

本 sprint 只升级 scoring，不重建 downstream。

---

## 2. 当前阶段定位

本 sprint 处理的是：

```text id="rhrq6p"
recover_outputs_real.jsonl
→
recover_scores_nli_judge.jsonl
```

不是：

```text id="k5wun9"
recover_scores_nli_judge.jsonl
→ unit_evidence
→ attention_anchor_labels
→ intervention_manifest
```

downstream 重建放到下一步：

```text id="x2jypu"
Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring
```

---

## 3. 本 sprint 做什么

本 sprint 应完成：

```text id="nv7xs5"
1. 新增 recover score backend：nli_recovery_judge_v0。
2. 保留 stub_rule_v0 行为不变。
3. 使用本地 NLI 模型对 original_question 与 recovered_question 做双向 NLI。
4. 基于 NLI 分数生成 recoverability_label。
5. 生成新的 recover_scores_nli_judge.jsonl。
6. 生成 recovery_scoring_report.json。
7. 更新 recover_scores interface 文档。
8. 更新 PROGRESS.md 与 sprint_1_history.md。
```

---

## 4. 本 sprint 不做什么

本 sprint 禁止做：

```text id="m1d47r"
1. 不修改 recovery generation。
2. 不调用 Ollama 生成新的 recovery outputs。
3. 不新增 LLM judge。
4. 不新增 OpenAI / Claude / Gemini API backend。
5. 不修改 NLI backend 实现。
6. 不修改 semantic label rule。
7. 不修改 unit evidence aggregation。
8. 不修改 attention anchor label rule。
9. 不修改 intervention manifest rule。
10. 不重建 unit_evidence / attention_anchor_labels / intervention_manifest。
11. 不接入 hidden states。
12. 不接入 attention maps。
13. 不接入 trajectory stability。
14. 不接入 answer stability。
15. 不接入 attention guidance。
16. 不做 probe training。
17. 不覆盖 data/processed baseline。
18. 不做目录重构。
```

---

## 5. 核心判断逻辑

### 5.1 对每个 recovered question 做双向 NLI

对每个 recovery sample：

```text id="owxanb"
premise = original_question
hypothesis = recovered_question
```

得到 forward NLI：

```text id="7p0hxw"
original_question → recovered_question
```

再反向：

```text id="qnvcjh"
premise = recovered_question
hypothesis = original_question
```

得到 backward NLI：

```text id="r5x2su"
recovered_question → original_question
```

计算：

```text id="qe0cqd"
bidirectional_entailment_score = min(
  forward.scores.entailment,
  backward.scores.entailment
)

contradiction_score = max(
  forward.scores.contradiction,
  backward.scores.contradiction
)
```

---

## 6. 标签规则

新增 backend 输出仍然使用现有 recover_score schema。

`recoverability_label` 只能是已有 label：

```text id="xvajw4"
Recoverable
Partially Recoverable
Non-recoverable
Misleading Recovery
```

建议规则如下。

### 6.1 Empty recovery

如果：

```text id="ql6oh1"
recovered_question == ""
```

则：

```text id="n1q90q"
recoverability_label = Non-recoverable
recoverability_score = 0.0
confidence = 1.0
misleading_recovery = false
```

### 6.2 Mask remains

如果：

```text id="fp5hmo"
mask_token 仍出现在 recovered_question 中
```

则：

```text id="bpahvl"
recoverability_label = Non-recoverable
recoverability_score = 0.0
confidence = 1.0
misleading_recovery = false
```

### 6.3 Exact match shortcut

如果：

```text id="dadmk3"
normalized recovered_question == normalized original_question
```

则：

```text id="72ixu2"
recoverability_label = Recoverable
recoverability_score = 1.0
confidence = 1.0
misleading_recovery = false
```

但仍允许在 evidence 中记录：

```text id="pywb3f"
exact_match = true
```

### 6.4 Contradiction

如果：

```text id="gnqh8e"
contradiction_score >= contradiction_threshold
```

默认阈值：

```text id="69a9vf"
contradiction_threshold = 0.50
```

则：

```text id="m3rxem"
recoverability_label = Misleading Recovery
recoverability_score = max(0.0, 1.0 - contradiction_score)
confidence = contradiction_score
misleading_recovery = true
```

### 6.5 Recoverable

如果：

```text id="fyx5o5"
bidirectional_entailment_score >= recoverable_entailment_threshold
and contradiction_score < contradiction_threshold
```

默认：

```text id="grm0sm"
recoverable_entailment_threshold = 0.70
```

则：

```text id="fbpqqj"
recoverability_label = Recoverable
recoverability_score = bidirectional_entailment_score
confidence = bidirectional_entailment_score
misleading_recovery = false
```

### 6.6 Partially Recoverable

如果：

```text id="c3jfrn"
bidirectional_entailment_score >= partial_entailment_threshold
and contradiction_score < contradiction_threshold
```

默认：

```text id="e32qg5"
partial_entailment_threshold = 0.50
```

则：

```text id="yxbc9k"
recoverability_label = Partially Recoverable
recoverability_score = bidirectional_entailment_score
confidence = bidirectional_entailment_score
misleading_recovery = false
```

### 6.7 Non-recoverable

否则：

```text id="1fv2q7"
recoverability_label = Non-recoverable
recoverability_score = bidirectional_entailment_score
confidence = max(
  1.0 - bidirectional_entailment_score,
  contradiction_score
)
misleading_recovery = false
```

---

## 7. 多 sample 聚合规则

当前 `recover_outputs` 可以对同一个 `masked_id` 有多个 sample。

本 sprint 必须支持：

```text id="hzl3ye"
num_samples >= 1
```

对一个 `masked_id` 的所有 samples：

```text id="6ctwuy"
source_sample_ids = sample_id list
recovered_questions = recovered_question list
```

聚合规则：

```text id="turb83"
1. 如果任一 sample 是 Misleading Recovery，则整体 misleading_recovery = true。
2. recoverability_score = max(sample recoverability_score)。
3. confidence_mean = average(sample confidence)。
4. recoverability_label 取最高 recoverability_score 对应 sample 的 label。
5. 如果最高分 sample label 是 Misleading Recovery，但存在 Recoverable sample，则优先 Recoverable。
6. recovery_consistency 按 normalized recovered_question 的一致性计算。
```

建议 `recovery_consistency`：

```text id="avhcyk"
如果 num_samples == 1：
  recovery_consistency = 1.0

如果 num_samples > 1：
  recovery_consistency = max_duplicate_count / num_samples
```

---

## 8. evidence 字段要求

`recover_score` record 已有：

```text id="ezr4on"
evidence
```

本 sprint 的 `nli_recovery_judge_v0` 必须在 evidence 中记录必要信息。

建议结构：

```json id="b1gsgs"
{
  "backend": "nli_recovery_judge_v0",
  "scoring_method": "bidirectional_nli_recovery_judge",
  "rule_parameters": {
    "recoverable_entailment_threshold": 0.70,
    "partial_entailment_threshold": 0.50,
    "contradiction_threshold": 0.50
  },
  "sample_evaluations": [
    {
      "sample_id": 0,
      "empty_recovery": false,
      "mask_remaining": false,
      "exact_match": false,
      "forward": {
        "label": "entailment",
        "scores": {
          "entailment": 0.82,
          "neutral": 0.10,
          "contradiction": 0.08
        }
      },
      "backward": {
        "label": "entailment",
        "scores": {
          "entailment": 0.79,
          "neutral": 0.15,
          "contradiction": 0.06
        }
      },
      "bidirectional_entailment_score": 0.79,
      "contradiction_score": 0.08,
      "sample_recoverability_label": "Recoverable",
      "sample_recoverability_score": 0.79,
      "sample_confidence": 0.79
    }
  ]
}
```

不要在 evidence 中保存：

```text id="ojj1px"
prompt
raw model response
hidden states
attention maps
full Ollama metadata
```

---

## 9. 输入与输出

### 9.1 输入

默认输入：

```text id="k8ahpe"
outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl
```

### 9.2 输出目录

新增 isolated output directory：

```text id="4qtnns"
outputs/logs/sprint_1O_recovery_scoring/
```

### 9.3 输出文件

生成：

```text id="g8bqrl"
outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.md
```

可选生成 stub 对照：

```text id="nc2xfv"
outputs/logs/sprint_1O_recovery_scoring/recover_scores_stub_check.jsonl
```

不要覆盖：

```text id="bb3j9j"
data/processed/recover_scores.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
```

---

## 10. 允许修改

本 sprint 允许修改：

```text id="bal3rz"
src/recover_attention/recover_scoring.py
scripts/09_score_recovery.py
tests/test_recover_scoring.py
src/recover_attention/schemas.py
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
configs/v0_nli_small.yaml
PROGRESS.md
docs/progress/sprint_1_history.md
```

允许由运行命令生成：

```text id="8mphgc"
outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.md
outputs/logs/sprint_1O_recovery_scoring/recover_scores_stub_check.jsonl
```

---

## 11. 禁止修改

本 sprint 禁止修改：

```text id="n8ft34"
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reference/*
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
scripts/05_run_nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/07_build_masked_questions.py
scripts/08_run_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
scripts/13_rebuild_downstream_real_signals.py
tests/test_nli_scoring.py
tests/test_semantic_labels.py
tests/test_masked_questions.py
tests/test_recover_generation.py
tests/test_unit_evidence.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
tests/test_rebuild_downstream_real_signals.py
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
models/*
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因。

---

## 12. 开始前必须读取

开始前必须读取：

```text id="skkgc3"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
src/recover_attention/schemas.py
src/recover_attention/recover_scoring.py
src/recover_attention/nli_scoring.py
src/recover_attention/data_io.py
scripts/09_score_recovery.py
tests/test_recover_scoring.py
configs/v0_nli_small.yaml
outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl
outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl
outputs/logs/sprint_1N_real_downstream/real_signal_report.json
docs/progress/sprint_1_history.md
```

不要读取：

```text id="e0hg6x"
docs/reference/*
```

除非用户另行明确要求。

---

## 13. Preflight 要求

修改任何文件前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text id="x8kynu"
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1N 已完成。
6. 当前 recover_scoring.py 支持哪些 score backend。
7. 当前 schemas.py 中 ALLOWED_RECOVER_SCORE_BACKENDS 的值。
8. 当前 recover_scores_interface.md 是否只写了 stub_rule_v0。
9. 当前 scripts/09_score_recovery.py 是否已有 --backend。
10. 当前 outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl 是否存在。
11. recover_outputs_real record 数量。
12. 当前 outputs/logs/sprint_1N_real_downstream/recover_scores_real.jsonl 是否存在。
13. 当前 real_signal_report.json 中 exact_match_recovery_count。
14. 当前 models/nli/en/roberta-large-mnli 是否存在。
15. 当前 models/nli/zh/mdeberta-v3-base-xnli 是否存在。
16. 当前 data 是否包含中文 recovery records。
17. 当前裸 python 路径与版本。
18. 当前 recover_attention 环境 python 路径与版本。
19. 本 sprint 实际运行命令是否全部使用 conda run -n recover_attention python。
20. 本次是否会覆盖 data/processed/*.jsonl。
21. 本次是否会修改 NLI backend。
22. 本次是否会修改 recovery generation。
23. 本次是否会调用 Ollama。
24. 本次是否会重建 downstream。
25. 本次是否会接入 attention guidance。
26. 本次输出目录。
27. 是否发现冲突。
```

第 19 项必须回答：

```text id="y3z0td"
是，全部使用 conda run -n recover_attention python。
```

第 20 项必须回答：

```text id="d973vt"
否。
```

第 21 项必须回答：

```text id="vgxbzd"
否。
```

第 22 项必须回答：

```text id="1v8ljw"
否。
```

第 23 项必须回答：

```text id="lkfwkj"
否。本 sprint 只读取已有 recover_outputs_real.jsonl，不重新调用 Ollama。
```

第 24 项必须回答：

```text id="dqt817"
否。
```

第 25 项必须回答：

```text id="1y1knw"
否。
```

---

## 14. schema 更新要求

允许修改：

```text id="ix39o3"
src/recover_attention/schemas.py
```

只允许扩展 recover score backend enum。

将：

```python id="c5038y"
ALLOWED_RECOVER_SCORE_BACKENDS = {"stub_rule_v0"}
```

扩展为：

```python id="nco82w"
ALLOWED_RECOVER_SCORE_BACKENDS = {
    "stub_rule_v0",
    "nli_recovery_judge_v0",
}
```

不要修改：

```text id="q9mfhi"
REQUIRED_FIELDS["recover_score"]
FORBIDDEN_FIELDS["recover_score"]
validate_recover_score_record
ALLOWED_RECOVERABILITY_LABELS
ALLOWED_RECOVERABLE_VALUES
```

---

## 15. recover_scores_interface.md 更新要求

更新：

```text id="pntq3l"
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
```

要求：

```text id="caqube"
1. recovery score backend 列表加入 nli_recovery_judge_v0。
2. 保留 stub_rule_v0 说明。
3. 明确 nli_recovery_judge_v0 使用 original_question 与 recovered_question 的双向 NLI。
4. 明确 nli_recovery_judge_v0 不调用 Ollama。
5. 明确 nli_recovery_judge_v0 不生成 recovery output，只评分已有 recovery output。
6. 明确 nli_recovery_judge_v0 仍不代表 attention guidance。
```

不要改 required fields block，除非 sync script 自动生成要求。

---

## 16. recover_scoring.py 实现要求

修改：

```text id="k2s9tn"
src/recover_attention/recover_scoring.py
```

### 16.1 保留 stub

必须保留：

```text id="pz9n9r"
stub_rule_v0
现有 exact normalized match 行为
现有 tests 不应破坏
```

### 16.2 新增 backend 常量

建议新增：

```python id="x5b5qc"
STUB_RULE_BACKEND = "stub_rule_v0"
NLI_RECOVERY_JUDGE_BACKEND = "nli_recovery_judge_v0"

SUPPORTED_RECOVER_SCORE_BACKENDS = {
    STUB_RULE_BACKEND,
    NLI_RECOVERY_JUDGE_BACKEND,
}
```

### 16.3 新增默认阈值

建议新增：

```python id="lvx1kw"
DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD = 0.70
DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD = 0.50
```

### 16.4 NLI scoring 复用

本 sprint 不修改 `nli_scoring.py`。

可以从 `recover_attention.nli_scoring` 复用：

```text id="ys95dr"
score_nli_pair_hf
load_hf_nli_model
detect_language / resolve language helper
```

如果现有函数签名不适合直接复用，可以在 `recover_scoring.py` 中新增轻量 wrapper，但不要修改 `nli_scoring.py`。

### 16.5 推荐新增函数

建议新增：

```python id="hnil34"
score_recovery_sample_with_nli(
    original_question: str,
    recovered_question: str,
    mask_token: str = "[MASK]",
    nli_backend: str = "hf_nli_auto_v0",
    language: str = "auto",
    en_model: str = "models/nli/en/roberta-large-mnli",
    zh_model: str = "models/nli/zh/mdeberta-v3-base-xnli",
    allow_download: bool = False,
    device: str = "auto",
    max_length: int = 512,
    label_order: str = "auto",
    recoverable_entailment_threshold: float = 0.70,
    partial_entailment_threshold: float = 0.50,
    contradiction_threshold: float = 0.50,
    _nli_model_cache: dict | None = None,
) -> dict
```

输出 sample-level evaluation：

```python id="elau57"
{
    "empty_recovery": bool,
    "mask_remaining": bool,
    "exact_match": bool,
    "forward": {...} | None,
    "backward": {...} | None,
    "bidirectional_entailment_score": float,
    "contradiction_score": float,
    "sample_recoverability_label": str,
    "sample_recoverability_score": float,
    "sample_confidence": float,
    "misleading_recovery": bool,
}
```

### 16.6 record-level builder

现有 recover scoring 应该是按 `masked_id` 聚合 recovery outputs。

必须扩展：

```python id="w0dw4i"
build_recover_score_record(...)
build_recover_score_records(...)
build_recover_score_file(...)
```

使其支持：

```text id="8md6s4"
backend = nli_recovery_judge_v0
```

旧调用必须保持不变。

---

## 17. scripts/09_score_recovery.py 更新要求

修改：

```text id="vjczyj"
scripts/09_score_recovery.py
```

当前已有：

```text id="zdib87"
--input
--output
--backend
```

新增：

```text id="ik2mxk"
--nli-backend
--language
--en-model
--zh-model
--en-model-id
--zh-model-id
--allow-download
--device
--batch-size
--max-length
--label-order
--recoverable-entailment-threshold
--partial-entailment-threshold
--contradiction-threshold
--limit
--report-output
```

默认值：

```text id="rltiez"
--nli-backend hf_nli_auto_v0
--language auto
--en-model models/nli/en/roberta-large-mnli
--zh-model models/nli/zh/mdeberta-v3-base-xnli
--en-model-id FacebookAI/roberta-large-mnli
--zh-model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
--allow-download False
--device auto
--batch-size 4
--max-length 512
--label-order auto
--recoverable-entailment-threshold 0.70
--partial-entailment-threshold 0.50
--contradiction-threshold 0.50
--limit None
--report-output None
```

`--allow-download` 必须默认关闭。

`--limit` 只在 script 层截断 input records。

---

## 18. recovery_scoring_report.json 要求

如果传入：

```text id="vxzl2o"
--report-output outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
```

必须生成 report。

报告至少包含：

```text id="kxcwaw"
run_metadata
input_counts
output_counts
backend_counts
recoverability_label_counts
misleading_recovery_counts
empty_recovery_count
mask_remaining_count
exact_match_recovery_count
score_distribution
sample_records
known_limitations
next_step_recommendation
```

### 18.1 run_metadata

包含：

```text id="ce5xd8"
timestamp
input_path
output_path
score_backend
nli_backend
language
en_model
zh_model
allow_download
device
max_length
label_order
recoverable_entailment_threshold
partial_entailment_threshold
contradiction_threshold
limit
```

### 18.2 score_distribution

至少包含：

```text id="gsovx4"
min
max
mean
median
```

针对：

```text id="nnlboe"
recoverability_score
confidence_mean
recovery_consistency
```

### 18.3 sample_records

最多保存 10 条 summary：

```text id="dakw9x"
masked_id
id
unit_id
masked_question
original_question
recovered_questions
recoverability_label
recoverability_score
confidence_mean
recovery_consistency
misleading_recovery
```

### 18.4 known_limitations

必须包含：

```text id="dxgzie"
1. nli_recovery_judge_v0 依赖 question-level NLI，不直接验证每个 masked span。
2. NLI 等价不等于最终 reasoning usefulness。
3. 该 scorer 不证明 attention guidance 有效。
4. 该 scorer 不使用 hidden states / attention maps / trajectory stability。
5. 下一步需要用 upgraded recovery scores 重建 unit_evidence / attention_anchor_labels / intervention_manifest。
```

### 18.5 next_step_recommendation

必须是：

```text id="ab5zqs"
Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring
```

---

## 19. 测试要求

更新：

```text id="bnko75"
tests/test_recover_scoring.py
```

测试不能调用真实 NLI 模型，也不能下载模型。

必须使用 monkeypatch / fake NLI outputs。

新增测试至少覆盖：

```text id="d9szuu"
1. SUPPORTED_RECOVER_SCORE_BACKENDS 包含 stub_rule_v0 / nli_recovery_judge_v0。
2. schemas.py 的 ALLOWED_RECOVER_SCORE_BACKENDS 包含 nli_recovery_judge_v0。
3. stub_rule_v0 行为保持不变。
4. empty recovery → Non-recoverable。
5. recovered_question 含 [MASK] → Non-recoverable。
6. exact match → Recoverable。
7. 双向 entailment 高且 contradiction 低 → Recoverable。
8. 双向 entailment 中等且 contradiction 低 → Partially Recoverable。
9. contradiction 高 → Misleading Recovery。
10. 双向 entailment 低且 contradiction 低 → Non-recoverable。
11. 多 sample 聚合选择最高 recoverability_score。
12. 多 sample 中出现 Misleading Recovery 时 misleading_recovery=true。
13. recovery_consistency 对单 sample 为 1.0。
14. recovery_consistency 对多 sample 正确计算。
15. output record 顶层字段仍等于 REQUIRED_FIELDS["recover_score"]。
16. validate_recover_score_record 通过。
17. CLI --limit 只处理前 N 条 recover_outputs。
18. CLI --report-output 生成 report。
19. report 包含 known_limitations。
20. nli_recovery_judge_v0 不调用 Ollama。
21. allow_download 默认 False。
22. 本地模型路径缺失且 allow_download=False 时错误清楚。
```

---

## 20. 必须运行命令

所有命令必须使用：

```text id="fggdj0"
conda run -n recover_attention python ...
```

不要使用裸 `python`。

### 20.1 基础测试

```powershell id="di3i25"
conda run -n recover_attention python scripts/sync_interface_fields.py --check

conda run -n recover_attention python -m pytest tests/test_recover_scoring.py -q

conda run -n recover_attention python -m pytest tests/test_schemas.py -q

conda run -n recover_attention python -m pytest -q
```

### 20.2 stub regression

```powershell id="8bcu4g"
conda run -n recover_attention python scripts/09_score_recovery.py `
  --input data/processed/recover_outputs.jsonl `
  --output outputs/logs/sprint_1O_recovery_scoring/recover_scores_stub_check.jsonl `
  --backend stub_rule_v0
```

### 20.3 小样本 NLI recovery judge run

先运行 limit 10：

```powershell id="tcdmz2"
conda run -n recover_attention python scripts/09_score_recovery.py `
  --input outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl `
  --output outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge_small.jsonl `
  --backend nli_recovery_judge_v0 `
  --nli-backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --device auto `
  --max-length 512 `
  --label-order auto `
  --recoverable-entailment-threshold 0.70 `
  --partial-entailment-threshold 0.50 `
  --contradiction-threshold 0.50 `
  --limit 10 `
  --report-output outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report_small.json
```

### 20.4 全量 NLI recovery judge run

如果 limit 10 通过，运行全量：

```powershell id="zmnfwk"
conda run -n recover_attention python scripts/09_score_recovery.py `
  --input outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl `
  --output outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl `
  --backend nli_recovery_judge_v0 `
  --nli-backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --device auto `
  --max-length 512 `
  --label-order auto `
  --recoverable-entailment-threshold 0.70 `
  --partial-entailment-threshold 0.50 `
  --contradiction-threshold 0.50 `
  --report-output outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
```

如果全量 run 失败，最终回复必须说明：

```text id="kml59q"
1. 失败阶段。
2. 错误信息。
3. 已成功生成到哪个阶段。
4. 是否保留 limit 10 的产物。
5. 是否需要下一个 sprint 修复。
```

---

## 21. configs 更新要求

允许更新：

```text id="cknazg"
configs/v0_nli_small.yaml
```

追加：

```yaml id="xk93r9"
real_recovery_scoring:
  output_dir: "outputs/logs/sprint_1O_recovery_scoring"
  input_recover_outputs: "outputs/logs/sprint_1N_real_downstream/recover_outputs_real.jsonl"
  output_recover_scores: "outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl"
  report_output: "outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json"
  score_backend: "nli_recovery_judge_v0"
  nli:
    backend: "hf_nli_auto_v0"
    language: "auto"
    en_model: "models/nli/en/roberta-large-mnli"
    zh_model: "models/nli/zh/mdeberta-v3-base-xnli"
    allow_download: false
    device: "auto"
    max_length: 512
    label_order: "auto"
  thresholds:
    recoverable_entailment_threshold: 0.70
    partial_entailment_threshold: 0.50
    contradiction_threshold: 0.50
```

不要让 `scripts/00_smoke_test.py` 自动运行 real recovery scoring。

---

## 22. PROGRESS.md 更新要求

更新：

```text id="yk5umh"
PROGRESS.md
```

要求：

```text id="qdoeqd"
1. 当前阶段更新为 Sprint 1O 已完成：Upgrade Real Recovery Scoring。
2. 已完成 Sprint 摘要中新增 Sprint 1O。
3. 当前可运行命令中新增 nli_recovery_judge_v0 示例命令。
4. 最近一次检查结果中记录：
   real recovery scoring upgrade: passed
   nli_recovery_judge_v0 full run: passed
   stub_rule_v0 regression: passed
   recovery scoring report: passed
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中新增：
   outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl
   outputs/logs/sprint_1O_recovery_scoring/recovery_scoring_report.json
6. 遗留问题中说明：
   - nli_recovery_judge_v0 是 question-level NLI judge，不直接验证每个 masked span。
   - upgraded recover_scores 尚未用于重建 unit_evidence / attention_anchor_labels / intervention_manifest。
   - 当前没有接入 hidden states / attention maps / trajectory stability / attention guidance。
7. 下一步建议：
   Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring
```

如果全量 run 未成功，只完成了 limit 10，则不要写 full run passed，应写：

```text id="y1d2cd"
nli_recovery_judge_v0 small run: passed
full run: failed / skipped，原因是 ...
```

---

## 23. docs/progress/sprint_1_history.md 更新要求

更新：

```text id="y7n5fg"
docs/progress/sprint_1_history.md
```

追加：

```text id="cxhr7h"
## Sprint 1O：Upgrade Real Recovery Scoring
```

内容包括：

```text id="57nzob"
1. 已完成内容。
2. 新增/修改文件。
3. 新增 backend：
   nli_recovery_judge_v0
4. NLI 判断规则。
5. 阈值设置。
6. 输出目录。
7. limit 10 run 是否完成。
8. full run 是否完成。
9. recovery_scoring_report.json 关键统计摘要。
10. 检查结果。
11. 遗留问题。
12. 下一步建议：Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring。
```

---

## 24. 验收标准

本 sprint 完成后必须满足：

```text id="ej1ppu"
1. stub_rule_v0 行为保持不变。
2. schemas.py 中 ALLOWED_RECOVER_SCORE_BACKENDS 包含 nli_recovery_judge_v0。
3. recover_score 顶层字段未变化。
4. validate_recover_score_record 仍通过。
5. recover_scores_interface.md 已更新 backend 说明。
6. scripts/09_score_recovery.py 支持 nli_recovery_judge_v0。
7. scripts/09_score_recovery.py 支持 --nli-backend / --language / --en-model / --zh-model / --allow-download / --device / --max-length / --label-order / thresholds / --limit / --report-output。
8. 默认 allow_download=False。
9. 不调用 Ollama。
10. 不修改 recover_generation.py。
11. 不修改 nli_scoring.py。
12. 不修改 downstream builder。
13. outputs/logs/sprint_1O_recovery_scoring/recover_scores_nli_judge.jsonl 已生成。
14. recovery_scoring_report.json 已生成。
15. report 包含 label distributions。
16. report 包含 known limitations。
17. tests/test_recover_scoring.py 通过。
18. tests/test_schemas.py 通过。
19. 全量 pytest 通过。
20. sync_interface_fields.py --check 通过。
21. 没有覆盖 data/processed/*.jsonl。
22. 没有修改 Sprint 1N outputs。
23. 没有接入 attention guidance。
24. PROGRESS.md 已更新。
25. docs/progress/sprint_1_history.md 已更新。
```

---

## 25. 完成后回复格式

完成后请按以下格式回复：

```text id="dh4ewd"
1. 本次完成内容
2. 新增/修改文件
3. 新增 backend
4. NLI recovery judge 规则
5. 阈值设置
6. 输出目录
7. 运行命令
8. small run / full run 状态
9. recovery_scoring_report.json 关键统计
10. 检查结果
11. PROGRESS.md 更新摘要
12. docs/progress/sprint_1_history.md 更新摘要
13. 遗留问题
14. 下一步建议
```

下一步建议必须是：

```text id="pa6pbv"
Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring
```

不要自动开始 Sprint 1P。
