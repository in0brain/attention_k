# Sprint 2D：Probe Training Baseline

## 0. Sprint 名称

Sprint 2D — Probe Training Baseline

## 1. 当前项目状态

当前项目为：

```text id="ihzjri"
Reasoning-Aware Attention Guidance
```

Sprint 2 最小闭环为：

```text id="zcmf29"
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

```text id="efeez5"
Sprint 1Q human review
Sprint 1R consolidation
Sprint 2A stub hidden-state cache
Sprint 2A-real real HF hidden-state cache
Sprint 2B representation feature extraction
Sprint 2B-fix representation feature scope alignment
Sprint 2C probe dataset construction
```

Sprint 2C 已生成正式输入：

```text id="rzkj49"
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

Sprint 2C 当前数据规模：

```text id="jor2ps"
num_probe_records = 20
num_probe_target_usable = 20
num_unmapped = 0
```

Target 分布：

```text id="xyi1jw"
risk_positive = 7
positive_anchor = 3
negative = 8
hard_negative_or_weak_positive = 2
```

注意：当前样本很小，2D 只验证最小训练闭环是否跑通，不追求高分。

---

## 2. 2D 目标

Sprint 2D 的目标是：

```text id="t0ffwf"
训练一个最小 hidden-state probe baseline，验证 Sprint 2B/2C 生成的 representation features 是否能产生非随机的 anchor / risk 预测信号。
```

本 sprint 要做到：

```text id="yqjcj3"
1. 能读取 probe_dataset.jsonl。
2. 能把 feature_values / feature_arrays flatten 成可训练特征。
3. 能处理 null features。
4. 能训练一个最小线性 probe。
5. 能进行 leave-one-out 或 k-fold 交叉验证。
6. 能输出 prediction。
7. 能输出 evaluation report。
8. 能保存最小模型文件 probe_model.pkl。
9. 能给出 feature / layer signal 的初步诊断。
```

---

## 3. 非目标

本 sprint 不做：

```text id="nk7l8k"
不重新构造 probe dataset
不修改 2C target mapping
不重新抽取 representation features
不读取 hidden-state tensors
不调用 HF model
不调用 tokenizer
不使用 GPU
不做 attention guidance
不生成 guidance_candidate_manifest
不修改 transformer attention weights
不重跑 CoT 推理
不验证 answer accuracy 提升
不声称 hallucination reduction
不做大规模实验
```

一句话边界：

```text id="4yux35"
2D 只做 probe training baseline；
2E 才把 probe prediction 转成 guidance candidate manifest dry run。
```

---

## 4. 必须阅读的文件

开始实现前，先阅读：

```text id="ttzmyb"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_2_history.md
docs/codex_tasks/sprint_2C_probe_dataset_construction.md
src/recover_attention/probe_dataset.py
src/recover_attention/representation_features.py
scripts/18_build_probe_dataset.py
tests/test_probe_dataset.py
```

如果本 task card 已存在，也阅读：

```text id="y8iunl"
docs/codex_tasks/sprint_2D_probe_training_baseline.md
```

---

## 5. 输入文件

2D 正式输入为：

```text id="scfjiw"
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

2D 只读取 2C 输出。

2D 不读取：

```text id="stwrx0"
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

如果为了 report traceability 读取 2C report 中的 source path 字段，可以读取 metadata，但不得重新计算 features 或 target。

---

## 6. 输出目录和输出文件

默认输出目录：

```text id="b1b0or"
outputs/logs/sprint_2D_probe_training_baseline/
```

正式输出文件：

```text id="dhtflw"
outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
```

可选 debug 输出：

```text id="np0u1o"
outputs/logs/sprint_2D_probe_training_baseline/probe_feature_index.json
```

`probe_feature_index.json` 不是必须产物。如果生成，只能记录 feature name → feature index，不得作为 2E 输入契约。

---

## 7. 允许新增文件

允许新增：

```text id="nql0gn"
src/recover_attention/probe_training.py
scripts/19_train_probe_baseline.py
tests/test_probe_training.py
docs/codex_tasks/sprint_2D_probe_training_baseline.md
```

---

## 8. 允许修改文件

允许修改：

```text id="v33sng"
PROGRESS.md
docs/progress/sprint_2_history.md
```

原则上不修改：

```text id="u4j1rq"
src/recover_attention/schemas.py
pyproject.toml
```

如果要使用外部库，必须先检查项目环境。当前优先选择不新增项目依赖的实现方式。

---

## 9. 禁止修改文件和目录

禁止修改：

```text id="fej8ia"
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

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/probe_dataset.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
```

2D 必须把 2C 输出视为 frozen upstream artifact。

---

## 10. 模型选择

### 10.1 首选模型

首选实现：

```text id="c22qgb"
ridge_classifier_ovr_v0
```

即：

```text id="h4bqqx"
one-vs-rest linear ridge-style classifier
```

原因：

```text id="tbog7q"
当前只有 20 条样本，线性模型最容易解释，也最适合输出 feature / layer signal。
```

可以使用：

```text id="ug0kt5"
scikit-learn RidgeClassifier / LogisticRegression
```

但不要强行新增依赖。

如果环境没有 scikit-learn，优先实现一个轻量 fallback：

```text id="qk8p7f"
pure-python 或 numpy-based deterministic linear baseline
```

不要求 MLP。

不优先 MLP。

---

## 11. 训练任务类型

2D 主任务是多分类：

```text id="lcwf15"
probe_target ∈ {
  risk_positive,
  positive_anchor,
  negative,
  hard_negative_or_weak_positive
}
```

同时在 eval report 中可以给出一个辅助二分类视角：

```text id="qpis3u"
anchor_or_risk = {
  risk_positive,
  positive_anchor
}

non_anchor_or_risk = {
  negative,
  hard_negative_or_weak_positive
}
```

注意：

```text id="c0kyo9"
辅助二分类只用于 report 分析。
不要生成第二个正式 dataset。
不要替代主 probe_target。
```

---

## 12. 样本过滤规则

默认只使用：

```text id="gxloeb"
probe_target_usable == true
```

且：

```text id="bvxarw"
probe_target != "unmapped"
```

当前预期：

```text id="qh9z5u"
usable records = 20
unmapped = 0
```

如果 usable records < 4 或有效类别数 < 2：

```text id="ejxw7f"
不要训练。
输出 status = "insufficient_data" 的 probe_eval_report.json。
不要生成 misleading 的 probe_predictions.jsonl。
```

当前数据应能训练。

---

## 13. Feature flattening 规则

从每条 `probe_dataset.jsonl` 读取：

```text id="gw2zju"
feature_values
feature_arrays
null_feature_keys
```

### 13.1 scalar features

`feature_values` 中的数值字段直接作为 scalar feature。

例如：

```text id="lnm3q2"
question_original_masked_cosine_mean
question_original_recovered_cosine_mean
question_masked_recovered_cosine_mean
...
```

### 13.2 layer-wise array features

`feature_arrays` 中的数组按 layer index 展开。

例如：

```text id="h0w4hh"
question_original_masked_cosine_by_layer = [a, b, c, d, e]
```

展开为：

```text id="nl6fmh"
question_original_masked_cosine_layer_0
question_original_masked_cosine_layer_1
question_original_masked_cosine_layer_2
question_original_masked_cosine_layer_3
question_original_masked_cosine_layer_4
```

如果原记录中有 `layer_indices`，feature name 可以使用真实 layer：

```text id="leirq7"
question_original_masked_cosine_layer_0
question_original_masked_cosine_layer_8
question_original_masked_cosine_layer_16
question_original_masked_cosine_layer_24
question_original_masked_cosine_layer_27
```

优先使用真实 layer index。

---

## 14. Null feature 处理

2C 已保留 null features。

2D baseline 必须处理 null，但不能丢弃整条记录。

默认策略：

```text id="fxc75b"
1. null numeric feature → 0.0
2. 同时增加 missing indicator feature
```

例如：

```text id="nra6dr"
span_original_masked_cosine_mean = null
```

转换为：

```text id="roth43"
span_original_masked_cosine_mean = 0.0
span_original_masked_cosine_mean__is_missing = 1.0
```

非 null：

```text id="6hn3zr"
span_original_masked_cosine_mean = 0.23
span_original_masked_cosine_mean__is_missing = 0.0
```

不要直接删除 null feature。

不要直接删除包含 null 的 record。

不要做复杂插补。

---

## 15. Feature scaling

训练时应做标准化：

```text id="vz54pw"
z-score standardization
```

要求：

```text id="t2umkc"
1. 每个 fold 只用 train fold 计算 mean/std。
2. 用 train mean/std 转换 train 和 held-out sample。
3. std 为 0 时设为 1.0。
4. final model 用全部 usable records 计算 final scaler。
```

不要在 leave-one-out 中用全数据计算 scaler，避免泄漏。

---

## 16. 交叉验证策略

默认使用：

```text id="k9hjtc"
leave_one_out
```

原因：

```text id="u7vd19"
当前只有 20 条样本。
```

可选支持：

```text id="pa8ugw"
stratified_k_fold
```

但由于最小类只有 2 条，k 不得大于最小类别样本数。

如果用户传入：

```text id="f51rdu"
--cv stratified_k_fold --num-folds 5
```

而最小类不足 5，必须报错或自动降级为合法 fold 数，并在 report 中记录 warning。

建议默认：

```text id="gxmys8"
--cv leave_one_out
```

---

## 17. Baseline 对照

`probe_eval_report.json` 必须包含至少一个 baseline：

```text id="eoqlps"
majority_class_baseline
```

可选：

```text id="pff7xf"
stratified_random_baseline
```

如果实现 random baseline，必须固定 seed。

推荐 seed：

```text id="e88e9r"
42
```

不要把小样本分数解释为强结论。

---

## 18. 预测输出格式

`probe_predictions.jsonl` 每条记录对应一个 cross-validation held-out prediction。

推荐字段：

```json id="exy9sg"
{
  "probe_record_id": "...",
  "feature_id": "...",
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",

  "cv_strategy": "leave_one_out",
  "fold_id": 0,
  "train_size": 19,
  "test_size": 1,

  "gold_probe_target": "risk_positive",
  "predicted_probe_target": "negative",
  "correct": false,

  "decision_scores": {
    "risk_positive": 0.1,
    "positive_anchor": -0.2,
    "negative": 0.3,
    "hard_negative_or_weak_positive": -0.1
  },

  "gold_anchor_or_risk_binary": true,
  "predicted_anchor_or_risk_binary": false,
  "binary_correct": false,

  "num_features": 120,
  "num_null_features": 3,
  "warnings": []
}
```

如果模型无法提供 meaningful decision score，可以输出：

```text id="i4sy9n"
decision_scores = null
```

但必须说明原因。

---

## 19. Evaluation report 内容

`probe_eval_report.json` 必须包含：

```json id="fpmoga"
{
  "sprint": "2D",
  "backend": "probe_training_baseline_v0",
  "status": "ok",

  "inputs": {
    "probe_dataset_path": "outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl",
    "probe_dataset_report_path": "outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json"
  },

  "outputs": {
    "probe_predictions_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl",
    "probe_eval_report_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json",
    "probe_model_path": "outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl"
  },

  "data_summary": {
    "num_records_total": 20,
    "num_records_usable": 20,
    "target_counts": {
      "risk_positive": 7,
      "positive_anchor": 3,
      "negative": 8,
      "hard_negative_or_weak_positive": 2
    },
    "records_with_null_features": 8
  },

  "training": {
    "model_type": "ridge_classifier_ovr_v0",
    "cv_strategy": "leave_one_out",
    "num_folds": 20,
    "seed": 42,
    "feature_standardization": "train_fold_zscore",
    "null_feature_strategy": "zero_impute_with_missing_indicators"
  },

  "metrics": {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "weighted_f1": 0.0,
    "per_class": {},
    "confusion_matrix": {}
  },

  "binary_anchor_or_risk_metrics": {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "positive_definition": ["risk_positive", "positive_anchor"]
  },

  "baselines": {
    "majority_class": {
      "label": "negative",
      "accuracy": 0.0,
      "macro_f1": 0.0
    }
  },

  "feature_signal_summary": {
    "top_weighted_features": [],
    "feature_group_weight_summary": {},
    "layer_weight_summary": {}
  },

  "small_sample_warning": "Only 20 human-reviewed examples are available. Scores are diagnostic, not conclusive.",

  "boundary": {
    "rebuilt_probe_dataset": false,
    "read_hidden_state_tensors": false,
    "recomputed_representation_features": false,
    "generated_guidance_candidates": false,
    "performed_attention_guidance": false,
    "claimed_hallucination_reduction": false
  },

  "warnings": []
}
```

---

## 20. Feature signal summary

2D 需要能“看出哪些特征/层有信号”，但只能作为初步诊断。

建议输出：

```text id="x1ir7j"
top_weighted_features
feature_group_weight_summary
layer_weight_summary
```

### 20.1 top_weighted_features

从 final model 的线性权重中取绝对值最大的若干 features。

推荐字段：

```json id="m0ycta"
{
  "feature_name": "question_original_masked_cosine_layer_24",
  "weight_abs_mean": 0.42,
  "per_class_weights": {
    "risk_positive": 0.3,
    "positive_anchor": 0.1,
    "negative": -0.4,
    "hard_negative_or_weak_positive": 0.0
  }
}
```

### 20.2 feature_group_weight_summary

按 feature group 聚合：

```text id="yx7xyi"
question
span
mask_position
missing_indicator
```

### 20.3 layer_weight_summary

按 layer 聚合：

```text id="yr605v"
layer_0
layer_8
layer_16
layer_24
layer_27
summary_scalar
missing_indicator
```

注意：

```text id="xc4imv"
这些只是 diagnostic signals。
不要声称 causal importance。
不要声称已证明 hallucination reduction。
```

---

## 21. model.pkl 内容

保存 final model：

```text id="ge7lhv"
outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl
```

该模型应在全部 usable records 上训练，用于后续 2E dry run。

`probe_model.pkl` 至少包含：

```python id="n7h6m2"
{
    "backend": "probe_training_baseline_v0",
    "model_type": "ridge_classifier_ovr_v0",
    "classes": [...],
    "feature_names": [...],
    "scaler": {
        "mean": [...],
        "std": [...]
    },
    "model_parameters": {...},
    "null_feature_strategy": "zero_impute_with_missing_indicators",
    "training_record_ids": [...],
    "created_from": {
        "probe_dataset_path": "...",
        "probe_dataset_report_path": "..."
    }
}
```

如果使用 sklearn，可直接 pickle estimator，但仍建议同时保存外层 metadata dict。

如果不用 sklearn，则保存自定义模型参数。

---

## 22. CLI 要求

新增脚本：

```text id="oi7vct"
scripts/19_train_probe_baseline.py
```

推荐 CLI：

```bash id="yq6nff"
conda run -n recover_attention python scripts/19_train_probe_baseline.py \
  --dataset outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl \
  --dataset-report outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json \
  --output-dir outputs/logs/sprint_2D_probe_training_baseline \
  --backend probe_training_baseline_v0 \
  --model ridge_classifier_ovr_v0 \
  --cv leave_one_out \
  --seed 42 \
  --overwrite
```

可选：

```bash id="ruf6wd"
--cv stratified_k_fold
--num-folds 2
--top-k-features 20
```

不要提供：

```text id="guozgs"
--model-path
--device
--device-map
--load-in-4bit
--hidden-state-dir
--features
--feature-report
```

2D 不再直接读取 2B features，也不读取 hidden states。

---

## 23. 实现建议

核心实现放在：

```text id="sbdttg"
src/recover_attention/probe_training.py
```

建议函数：

```python id="dr54qw"
load_probe_dataset(...)
load_probe_dataset_report(...)
select_usable_records(...)
flatten_probe_record(...)
build_feature_matrix(...)
build_missing_indicators(...)
standardize_train_and_test(...)
make_leave_one_out_folds(...)
make_stratified_k_folds(...)
train_ridge_classifier_ovr(...)
predict_ridge_classifier_ovr(...)
compute_multiclass_metrics(...)
compute_binary_anchor_or_risk_metrics(...)
compute_majority_baseline(...)
summarize_feature_weights(...)
train_final_model(...)
save_probe_model(...)
write_predictions(...)
write_eval_report(...)
train_probe_baseline(...)
```

---

## 24. 测试要求

新增：

```text id="nav6jh"
tests/test_probe_training.py
```

至少覆盖：

```text id="jvowg6"
1. 能读取 2C probe_dataset.jsonl。
2. 能读取 2C probe_dataset_report.json。
3. 只使用 probe_target_usable=true 的记录。
4. unmapped 记录不会进入训练。
5. feature_values 能 flatten。
6. feature_arrays 能按 layer flatten。
7. null feature 被转换为 0.0 + missing indicator。
8. 不删除含 null 的 record。
9. 标准化只用 train fold statistics。
10. leave-one-out fold 数等于 usable record 数。
11. 每个 held-out record 生成一条 prediction。
12. prediction record 包含 gold/predicted/correct/decision_scores。
13. 能计算 accuracy / macro_f1 / per-class metrics。
14. 能计算 majority class baseline。
15. 能生成 feature_signal_summary。
16. 能保存 probe_model.pkl。
17. 不读取 hidden-state tensors。
18. 不读取 2B representation_features.jsonl。
19. 不重新构造 probe dataset。
20. 不生成 guidance candidate manifest。
21. 不执行 attention steering。
22. usable records < 4 时输出 insufficient_data report，不误训。
23. 单类数据时输出 insufficient_data report。
24. CLI smoke test 能生成三个正式输出文件。
```

---

## 25. 必须运行的命令

Preflight：

```bash id="htsz6l"
git status --short
```

Targeted test：

```bash id="sow9rh"
conda run -n recover_attention python -m pytest tests/test_probe_training.py -q
```

Run 2D：

```bash id="dv45r8"
conda run -n recover_attention python scripts/19_train_probe_baseline.py \
  --dataset outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl \
  --dataset-report outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json \
  --output-dir outputs/logs/sprint_2D_probe_training_baseline \
  --backend probe_training_baseline_v0 \
  --model ridge_classifier_ovr_v0 \
  --cv leave_one_out \
  --seed 42 \
  --overwrite
```

Full test：

```bash id="gik2uc"
conda run -n recover_attention python -m pytest -q
```

Final check：

```bash id="hapdvh"
git diff --name-only
git status --short
```

---

## 26. 验收标准

完成 Sprint 2D 时必须满足：

```text id="v4z63s"
1. 新增最小 probe training baseline。
2. 读取 probe_dataset.jsonl。
3. 读取 probe_dataset_report.json。
4. 不读取 hidden-state tensors。
5. 不读取 2B representation_features.jsonl 作为训练输入。
6. 不重新构造 probe dataset。
7. 不修改 1Q / 1R / 2A / 2B / 2C 输出。
8. 能 flatten feature_values。
9. 能 flatten feature_arrays。
10. 能处理 null features。
11. null features 使用 0.0 + missing indicator。
12. 能执行 leave-one-out。
13. 每个 usable record 生成一条 held-out prediction。
14. 输出 probe_predictions.jsonl。
15. 输出 probe_eval_report.json。
16. 输出 probe_model.pkl。
17. report 包含 target counts。
18. report 包含 metrics。
19. report 包含 majority baseline。
20. report 包含 feature_signal_summary。
21. report 明确 small_sample_warning。
22. 不生成 guidance_candidate_manifest。
23. 不执行 attention guidance。
24. 不声称 hallucination reduction。
25. targeted pytest passed。
26. full pytest passed。
27. PROGRESS.md 已更新。
28. docs/progress/sprint_2_history.md 已更新。
```

---

## 27. PROGRESS.md 更新建议

完成后在当前阶段补充：

```text id="itn2rw"
Sprint 2D 已完成：Probe Training Baseline。

probe_training_baseline_v0 已读取 Sprint 2C probe_dataset.jsonl，并训练最小线性 probe baseline。

正式输出：
- outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
- outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
- outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl

训练设置：
- model: ridge_classifier_ovr_v0
- cv: leave_one_out
- null feature strategy: zero impute + missing indicators
- feature scaling: train-fold z-score
- seed: 42

本阶段未读取 hidden-state tensors，未重新抽取 representation features，未重构 probe dataset，未生成 guidance candidate，未执行 attention steering，未声称 hallucination reduction。

下一步建议是 Sprint 2E：Guidance Candidate Manifest Dry Run。
```

---

## 28. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text id="lddt96"
## Sprint 2D：Probe Training Baseline

### Goal

Train a minimal hidden-state probe baseline from Sprint 2C probe dataset.

### Inputs

- outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
- outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json

### Outputs

- outputs/logs/sprint_2D_probe_training_baseline/probe_predictions.jsonl
- outputs/logs/sprint_2D_probe_training_baseline/probe_eval_report.json
- outputs/logs/sprint_2D_probe_training_baseline/probe_model.pkl

### Training Setup

- model: ridge_classifier_ovr_v0
- cross validation: leave_one_out
- target: probe_target
- null feature strategy: zero impute + missing indicators
- scaling: train-fold z-score
- seed: 42

### Checks

- targeted pytest: <fill>
- full pytest: <fill>
- num_predictions: <fill>
- accuracy: <fill>
- macro_f1: <fill>
- majority baseline accuracy: <fill>

### Notes

- Scores are diagnostic only because the dataset has only 20 human-reviewed records.
- Feature and layer signal summaries are not causal claims.
- No guidance candidate manifest was generated.
- No attention steering was performed.
- No hallucination reduction claim was made.

### Next

Sprint 2E：Guidance Candidate Manifest Dry Run.
```

---

## 29. 下一步边界

Sprint 2D 完成后，才进入：

```text id="s7ptwc"
Sprint 2E：Guidance Candidate Manifest Dry Run
```

2E 才负责：

```text id="flx9j0"
probe prediction
→ predicted anchor / risk
→ guidance candidate action
→ guidance_candidate_manifest.jsonl
```

2E 仍然是 dry run，不做真正 attention steering。
