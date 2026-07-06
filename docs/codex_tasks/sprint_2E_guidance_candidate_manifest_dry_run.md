# Sprint 2E：Guidance Candidate Manifest Dry Run

## 0. Sprint 名称

Sprint 2E — Guidance Candidate Manifest Dry Run

## 1. 当前项目状态

当前项目为：

```text id="e4kk33"
Reasoning-Aware Attention Guidance
```

Sprint 2 最小闭环为：

```text id="n7ne8s"
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

```text id="qhkueg"
Sprint 1Q human review
Sprint 1R consolidation
Sprint 2A stub hidden-state cache
Sprint 2A-real real HF hidden-state cache
Sprint 2B representation feature extraction
Sprint 2B-fix representation feature scope alignment
Sprint 2C probe dataset construction
Sprint 2D probe training baseline
```

Sprint 2D 已生成正式输入：

```text id="bnb6xc"
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
```

Sprint 2E 的目标不是训练，也不是真正 steering，而是把 probe predictions 转成一个可审查的 guidance candidate manifest dry run。

---

## 2. 2E 目标

Sprint 2E 的目标是：

```text id="l6j2nn"
把 probe prediction 转成 guidance candidate manifest，但不真正干预模型。
```

形成最小闭环末端：

```text id="c1vcrd"
hidden state signal
→ probe score
→ predicted anchor / risk
→ candidate guidance action
```

正式产物：

```text id="tyrn75"
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

---

## 3. 非目标

Sprint 2E 不做：

```text id="gtjjxs"
不修改 transformer attention weights
不注入 attention mask
不重跑 CoT 推理
不调用 HF model
不调用 Ollama
不调用 tokenizer
不读取 hidden-state tensors
不重新抽取 representation features
不重新构造 probe dataset
不重新训练 probe
不验证 answer accuracy 提升
不声称 hallucination reduction
不做真实 attention steering
不做 closed-loop final report
```

一句话边界：

```text id="vm4oa1"
2E 只生成 guidance candidate manifest dry run；
2F 才总结 Sprint 2 最小闭环；
Sprint 3 才考虑真正 attention steering。
```

---

## 4. 必须阅读的文件

开始实现前，先阅读：

```text id="zrqrq4"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_2_history.md

docs/codex_tasks/sprint_2D_probe_training_baseline.md

src/recover_attention/probe_dataset.py
src/recover_attention/probe_training.py
src/recover_attention/intervention_manifest.py

scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py

tests/test_probe_dataset.py
tests/test_probe_training.py
tests/test_intervention_manifest.py
```

如果本 task card 已存在，也阅读：

```text id="a6z3hu"
docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md
```

---

## 5. 输入文件

2E 正式输入为：

```text id="lu9v0a"
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
```

可选 traceability 输入：

```text id="gn5tih"
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

原则：

```text id="s2nm7f"
优先使用 probe_predictions.jsonl 中已有字段。
如果需要补充 human metadata / feature_id / unit_id，可只读 probe_dataset.jsonl 做 join。
```

2E 不需要读取：

```text id="y4n4nc"
probe_model.pkl
```

除非只是为了 report 中验证文件存在或记录 model path。2E 不重新预测，不重新训练，所以不需要加载模型。

---

## 6. 禁止输入

2E 禁止读取：

```text id="s1rj3u"
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

2E 禁止调用：

```text id="s5pjpz"
HF model
Ollama
tokenizer
NLI model
recovery model
```

2E 禁止重新运行：

```text id="q2c2ov"
scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
```

---

## 7. 输出目录与输出文件

默认输出目录：

```text id="rq4aem"
outputs/logs/sprint_2E_guidance_candidate_dry_run/
```

正式输出文件：

```text id="znxajh"
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json
```

不得输出：

```text id="uw8xhv"
attention_steering_results.jsonl
model_outputs_after_guidance.jsonl
answer_accuracy_report.json
hallucination_reduction_report.json
closed_loop_final_report.md
```

这些不属于 2E。

---

## 8. 允许新增文件

允许新增：

```text id="p6486k"
src/recover_attention/guidance_candidates.py
scripts/20_build_guidance_candidate_manifest.py
tests/test_guidance_candidates.py
docs/codex_tasks/sprint_2E_guidance_candidate_manifest_dry_run.md
```

---

## 9. 允许修改文件

允许修改：

```text id="lkkbzn"
PROGRESS.md
docs/progress/sprint_2_history.md
```

原则上不修改：

```text id="s13r9s"
src/recover_attention/schemas.py
```

如果需要常量，优先放在 `src/recover_attention/guidance_candidates.py` 中，不要大幅改全局 schema。

---

## 10. 禁止修改文件和目录

禁止修改：

```text id="g8wswr"
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/SKILL.md

data/processed/*

outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*
outputs/logs/sprint_2B_representation_features/*
outputs/logs/sprint_2C_probe_dataset/*
outputs/logs/sprint_2D_probe_training_baseline/*

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/probe_dataset.py
src/recover_attention/probe_training.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
```

2E 必须把 2D 输出视为 frozen upstream artifact。

---

## 11. Candidate action 设计

2E 输出的 guidance candidate 不是实际干预，而是 planned-only action。

支持的 action 建议：

```text id="cn2vjf"
increase_attention_to_original_span
preserve_original_span_attention
review_before_guidance
no_guidance
```

其中：

### 11.1 risk_positive

如果：

```text id="z7onoj"
predicted_probe_target == "risk_positive"
```

则：

```text id="ivlh3w"
guidance_candidate = true
candidate_action = "increase_attention_to_original_span"
candidate_reason = "probe predicted high-risk recovery drift; original span should be emphasized in future attention guidance"
execution_status = "planned_only"
```

### 11.2 positive_anchor

如果：

```text id="uwljvi"
predicted_probe_target == "positive_anchor"
```

则：

```text id="uc7kga"
guidance_candidate = true
candidate_action = "preserve_original_span_attention"
candidate_reason = "probe predicted useful anchor; original span should be preserved as candidate attention anchor"
execution_status = "planned_only"
```

### 11.3 hard_negative_or_weak_positive

如果：

```text id="py75k4"
predicted_probe_target == "hard_negative_or_weak_positive"
```

则：

```text id="vdjjeq"
guidance_candidate = true
candidate_action = "review_before_guidance"
candidate_reason = "probe predicted ambiguous weak signal; requires review before real guidance"
execution_status = "planned_only"
```

### 11.4 negative

如果：

```text id="zdkd9r"
predicted_probe_target == "negative"
```

则：

```text id="ip06af"
guidance_candidate = false
candidate_action = "no_guidance"
candidate_reason = "probe predicted negative; no guidance candidate proposed"
execution_status = "planned_only"
```

---

## 12. Confidence / score 规则

2D prediction 中有 `decision_scores`。

2E 可以基于 decision_scores 计算一个简单 confidence：

```text id="f1ya91"
predicted_score = decision_scores[predicted_probe_target]
second_best_score = max(score of other classes)
score_margin = predicted_score - second_best_score
```

输出：

```text id="kt2rtk"
probe_score_margin
probe_confidence_level
```

推荐 confidence level：

```text id="h32kde"
high: score_margin >= 0.5
medium: 0.1 <= score_margin < 0.5
low: score_margin < 0.1
unknown: decision_scores missing
```

注意：

```text id="fib8eu"
confidence level 只是 dry-run 排序辅助。
不要把它当作 calibrated probability。
不要声称它是可靠置信度。
```

---

## 13. 正误字段处理

`probe_predictions.jsonl` 中有 gold / predicted / correct。

2E 可以复制：

```text id="lilxju"
gold_probe_target
predicted_probe_target
correct
gold_anchor_or_risk_binary
predicted_anchor_or_risk_binary
binary_correct
```

但 2E 的 guidance candidate 应基于：

```text id="j6trnz"
predicted_probe_target
```

而不是 gold label。

原因：

```text id="d2hcgd"
2E 模拟未来真实使用场景，真实场景中只有 prediction，没有 gold。
```

可以在 report 中统计：

```text id="zhu7vs"
candidate_correctness_by_gold
```

但不得用 gold 修正 candidate action。

---

## 14. guidance_candidate_manifest.jsonl 记录粒度

记录粒度必须与 2D prediction 一致：

```text id="v4sbat"
one prediction record → one guidance candidate record
```

当前预期：

```text id="tw08a7"
20 prediction records → 20 guidance candidate records
```

不要按 `masked_id` 聚合。

不要丢弃 negative。

不要只输出 candidate=true 的记录。

所有 prediction 都要有一条 manifest record。

---

## 15. guidance_candidate_manifest.jsonl 推荐字段

每条 record 推荐包含：

```json id="htniwv"
{
  "guidance_candidate_id": "...",
  "probe_record_id": "...",
  "feature_id": "...",
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",

  "source_prediction_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl",
  "source_eval_report_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json",

  "gold_probe_target": "risk_positive",
  "predicted_probe_target": "risk_positive",
  "prediction_correct": true,

  "decision_scores": {
    "risk_positive": 0.7,
    "positive_anchor": 0.1,
    "negative": -0.2,
    "hard_negative_or_weak_positive": -0.1
  },
  "probe_score_margin": 0.6,
  "probe_confidence_level": "high",

  "guidance_candidate": true,
  "candidate_action": "increase_attention_to_original_span",
  "candidate_reason": "probe predicted high-risk recovery drift; original span should be emphasized in future attention guidance",

  "execution_status": "planned_only",
  "dry_run": true,

  "will_modify_attention": false,
  "will_run_model": false,
  "will_change_answer": false,

  "warnings": []
}
```

---

## 16. guidance_candidate_report.json 必需内容

`guidance_candidate_report.json` 必须包含：

```json id="g71o1b"
{
  "sprint": "2E",
  "backend": "guidance_candidate_dry_run_v0",
  "status": "ok",

  "inputs": {
    "probe_predictions_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl",
    "probe_eval_report_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json",
    "source_2D_backend": "probe_training_baseline_v0"
  },

  "outputs": {
    "guidance_candidate_manifest_path": "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl",
    "guidance_candidate_report_path": "outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json"
  },

  "counts": {
    "num_prediction_records": 20,
    "num_guidance_candidate_records": 20,
    "num_guidance_candidate_true": 0,
    "num_guidance_candidate_false": 0
  },

  "predicted_target_counts": {
    "risk_positive": 0,
    "positive_anchor": 0,
    "negative": 0,
    "hard_negative_or_weak_positive": 0
  },

  "candidate_action_counts": {
    "increase_attention_to_original_span": 0,
    "preserve_original_span_attention": 0,
    "review_before_guidance": 0,
    "no_guidance": 0
  },

  "confidence_counts": {
    "high": 0,
    "medium": 0,
    "low": 0,
    "unknown": 0
  },

  "candidate_correctness_by_gold": {
    "num_prediction_correct": 0,
    "num_prediction_incorrect": 0,
    "note": "Gold labels are used only for retrospective diagnostics, not for candidate action assignment."
  },

  "boundary": {
    "dry_run": true,
    "loaded_probe_model": false,
    "retrained_probe": false,
    "rebuilt_probe_dataset": false,
    "read_hidden_state_tensors": false,
    "recomputed_representation_features": false,
    "called_hf_model": false,
    "called_ollama": false,
    "modified_attention_weights": false,
    "ran_attention_steering": false,
    "ran_cot_generation": false,
    "evaluated_answer_accuracy": false,
    "claimed_hallucination_reduction": false
  },

  "warnings": []
}
```

---

## 17. 实现建议

核心实现放在：

```text id="lspwex"
src/recover_attention/guidance_candidates.py
```

建议函数：

```python id="rl6mgz"
load_probe_predictions(...)
load_probe_eval_report(...)
compute_score_margin(...)
confidence_level_from_margin(...)
candidate_action_from_prediction(...)
build_guidance_candidate_record(...)
build_guidance_candidate_manifest(...)
build_guidance_candidate_report(...)
write_jsonl_strict(...)
write_json_strict(...)
```

核心 mapping 函数：

```python id="j04ju5"
def candidate_action_from_prediction(predicted_probe_target: str) -> dict:
    ...
```

返回：

```python id="x6w31n"
{
    "guidance_candidate": True,
    "candidate_action": "...",
    "candidate_reason": "...",
    "execution_status": "planned_only"
}
```

---

## 18. CLI 要求

新增脚本：

```text id="h2zh6i"
scripts/20_build_guidance_candidate_manifest.py
```

推荐 CLI：

```bash id="t4rrsb"
conda run -n recover_attention python scripts/20_build_guidance_candidate_manifest.py \
  --predictions outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl \
  --eval-report outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json \
  --output-dir outputs/logs/sprint_2E_guidance_candidate_dry_run \
  --backend guidance_candidate_dry_run_v0 \
  --overwrite
```

不得提供：

```text id="fs7in4"
--model-path
--device
--device-map
--dataset
--features
--hidden-state-dir
--train
--steer
```

2E 不训练，不重算，不 steering。

---

## 19. 测试要求

新增：

```text id="jdimgx"
tests/test_guidance_candidates.py
```

至少覆盖：

```text id="qkqawd"
1. 能读取 probe_predictions.jsonl。
2. 能读取 probe_eval_report.json。
3. 每条 prediction 生成一条 guidance candidate record。
4. negative prediction → guidance_candidate=false, no_guidance。
5. risk_positive prediction → increase_attention_to_original_span。
6. positive_anchor prediction → preserve_original_span_attention。
7. hard_negative_or_weak_positive prediction → review_before_guidance。
8. candidate action 基于 predicted_probe_target，不基于 gold_probe_target。
9. decision_scores 可计算 score margin。
10. decision_scores 缺失时 confidence=unknown。
11. 所有 record execution_status=planned_only。
12. 所有 record dry_run=true。
13. 所有 record will_modify_attention=false。
14. 不加载 probe_model.pkl。
15. 不读取 hidden-state tensors。
16. 不重训 probe。
17. 不重新构造 probe dataset。
18. 不重新抽取 representation features。
19. 不生成 attention_steering_results。
20. 不生成 answer_accuracy_report。
21. report 统计 candidate_action_counts。
22. report 统计 predicted_target_counts。
23. report 包含 dry-run boundary。
24. CLI smoke test 能生成 guidance_candidate_manifest.jsonl 和 guidance_candidate_report.json。
```

---

## 20. 必须运行的命令

Preflight：

```bash id="jc29nf"
git status --short
```

Targeted test：

```bash id="s9hl81"
conda run -n recover_attention python -m pytest tests/test_guidance_candidates.py -q
```

Run 2E：

```bash id="dfduys"
conda run -n recover_attention python scripts/20_build_guidance_candidate_manifest.py \
  --predictions outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl \
  --eval-report outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json \
  --output-dir outputs/logs/sprint_2E_guidance_candidate_dry_run \
  --backend guidance_candidate_dry_run_v0 \
  --overwrite
```

Full test：

```bash id="r3gzjv"
conda run -n recover_attention python -m pytest -q
```

Final check：

```bash id="kljccz"
git diff --name-only
git status --short
```

---

## 21. 验收标准

完成 Sprint 2E 时必须满足：

```text id="ueylh4"
1. 新增 guidance candidate dry-run pipeline。
2. 读取 probe_predictions.jsonl。
3. 读取 probe_eval_report.json。
4. 不加载 probe_model.pkl。
5. 不读取 hidden-state tensors。
6. 不重新训练 probe。
7. 不重新构造 probe dataset。
8. 不重新抽取 representation features。
9. 每条 prediction 生成一条 guidance candidate record。
10. 不丢弃 negative prediction。
11. risk_positive 映射为 increase_attention_to_original_span。
12. positive_anchor 映射为 preserve_original_span_attention。
13. hard_negative_or_weak_positive 映射为 review_before_guidance。
14. negative 映射为 no_guidance。
15. candidate action 基于 predicted_probe_target，不基于 gold label。
16. 所有 execution_status 都是 planned_only。
17. 所有 record 标记 dry_run=true。
18. 所有 record 标记 will_modify_attention=false。
19. 输出 guidance_candidate_manifest.jsonl。
20. 输出 guidance_candidate_report.json。
21. report 包含 candidate_action_counts。
22. report 包含 predicted_target_counts。
23. report 包含 confidence_counts。
24. report 明确不做 attention steering。
25. report 明确不声称 hallucination reduction。
26. targeted pytest passed。
27. full pytest passed。
28. PROGRESS.md 已更新。
29. docs/progress/sprint_2_history.md 已更新。
```

---

## 22. PROGRESS.md 更新建议

完成后在当前阶段补充：

```text id="oit6ev"
Sprint 2E 已完成：Guidance Candidate Manifest Dry Run。

guidance_candidate_dry_run_v0 已读取 Sprint 2D probe_predictions.jsonl / probe_eval_report.json，并把 probe prediction 转成 planned-only guidance candidate manifest。

正式输出：
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json

本阶段只做 dry run：
- no probe retraining
- no hidden-state tensor reading
- no representation feature recomputation
- no probe dataset rebuild
- no HF/Ollama call
- no attention weight modification
- no CoT generation
- no answer accuracy evaluation
- no hallucination reduction claim

下一步建议是 Sprint 2F：Mini Closed-loop Report。
```

---

## 23. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text id="qpvvkh"
## Sprint 2E：Guidance Candidate Manifest Dry Run

### Goal

Convert Sprint 2D probe predictions into planned-only guidance candidate records.

### Inputs

- outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
- outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json

### Outputs

- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_manifest.jsonl
- outputs/logs/sprint_2E_guidance_candidate_dry_run/guidance_candidate_report.json

### Candidate Actions

- risk_positive → increase_attention_to_original_span
- positive_anchor → preserve_original_span_attention
- hard_negative_or_weak_positive → review_before_guidance
- negative → no_guidance

### Checks

- targeted pytest: <fill>
- full pytest: <fill>
- num_guidance_candidate_records: <fill>
- guidance_candidate_true: <fill>
- guidance_candidate_false: <fill>
- candidate_action_counts: <fill>

### Notes

- This sprint is dry run only.
- Candidate actions are based on predicted_probe_target, not gold labels.
- Gold labels are used only for retrospective diagnostics.
- No probe model was loaded.
- No probe was retrained.
- No hidden-state tensors were read.
- No representation features were recomputed.
- No attention steering was performed.
- No hallucination reduction claim was made.

### Next

Sprint 2F：Mini Closed-loop Report.
```

---

## 24. 下一步边界

Sprint 2E 完成后，进入：

```text id="sc35un"
Sprint 2F：Mini Closed-loop Report
```

2F 才回答：

```text id="rgv3eq"
1. hidden state 是否能稳定缓存？
2. token/span/mask 是否能对齐？
3. wrong_numeric / generic / entity drift 在表征上是否有差异？
4. probe 是否能产生非随机预测？
5. probe 预测能否生成合理 guidance candidate？
6. 哪些问题必须在 Sprint 3 前修？
```

2F 不实现新的 pipeline。

2F 不做真实 steering。

真正 attention steering 放到 Sprint 3。
