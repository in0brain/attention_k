# Sprint 2C：Probe Dataset Construction

## 0. Sprint 名称

Sprint 2C — Probe Dataset Construction

## 1. 当前项目状态

当前项目为：

```text id="pxvbdu"
Reasoning-Aware Attention Guidance
```

Sprint 2 最小闭环为：

```text id="h4bp58"
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

```text id="au3nso"
Sprint 1Q human review
Sprint 1R consolidation
Sprint 2A stub hidden-state cache
Sprint 2A-real real HF hidden-state cache
Sprint 2B representation feature extraction
Sprint 2B-fix representation feature scope alignment
```

Sprint 2B 当前正式输出为：

```text id="4g4zdu"
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

Sprint 2B 已明确不再把以下 legacy/debug 文件作为 2C 输入契约：

```text id="v0sv1s"
representation_feature_manifest.jsonl
input_representation_summary.jsonl
feature_schema.json
```

本 Sprint 2C 从 2B 的正式输出开始，不回退到 hidden states。

---

## 2. 2C 目标

Sprint 2C 的目标是：

```text id="qoc48q"
把 Sprint 1Q / 1R 人工标签转成 probe target，并与 Sprint 2B representation features 合并，构造 probe-ready dataset。
```

也就是：

```text id="1xhy0c"
representation_features.jsonl
+ human labels / review metadata
→ probe_dataset.jsonl
→ probe_dataset_report.json
```

2C 只做 dataset construction。

2C 不训练 probe。

2C 不做 train/dev/test split。

2C 不做 k-fold。

2C 不做 leave-one-out。

2C 不做 feature importance。

2C 不做 guidance candidate。

2C 不做 attention steering。

---

## 3. 本 sprint 的核心边界

2C 负责：

```text id="gm0yy6"
1. 读取 2B representation_features.jsonl。
2. 读取 2B representation_feature_report.json。
3. 必要时读取 Sprint 1R manifest 或 Sprint 1Q human review labels 进行 human label cross-check。
4. 基于 human_error_type / semantic role / anchor label / probe_usage 构造 probe target。
5. 输出 probe_dataset.jsonl。
6. 输出 probe_dataset_report.json。
```

2C 不负责：

```text id="a9jgek"
1. 不读取 hidden_states/*.pt。
2. 不重新提取 representation features。
3. 不调用 HF model。
4. 不调用 tokenizer。
5. 不使用 GPU。
6. 不构造 train/dev/test split。
7. 不训练 logistic regression / ridge / MLP。
8. 不输出 probe_predictions.jsonl。
9. 不输出 probe_eval_report.json。
10. 不输出 probe_model.pkl。
11. 不生成 guidance_candidate_manifest.jsonl。
12. 不做 attention guidance。
13. 不声称 hallucination reduction。
```

一句话边界：

```text id="c7s8tp"
2B 解决 hidden states → representation features。
2C 解决 human labels → probe targets / probe dataset。
2D 才训练 probe。
```

---

## 4. 必须阅读的文件

开始实现前，先阅读：

```text id="ddscuy"
AGENTS.md
PROGRESS.md
docs/skill/SKILL.md
docs/progress/sprint_2_history.md
docs/codex_tasks/sprint_2B_representation_feature_extraction.md
src/recover_attention/representation_features.py
src/recover_attention/human_review_consolidation.py
tests/test_representation_features.py
tests/test_human_review_consolidation.py
```

如果存在本 task card，也阅读：

```text id="y9sk38"
docs/codex_tasks/sprint_2C_probe_dataset_construction.md
```

不要阅读或依赖不存在的：

```text id="echth3"
tests/test_hidden_state_cache_hf.py
```

该文件缺失不是失败条件。

---

## 5. 输入文件

### 5.1 必需输入

2C 的正式输入为：

```text id="alytx7"
outputs/logs/sprint_2B_representation_features/representation_features.jsonl
outputs/logs/sprint_2B_representation_features/representation_feature_report.json
```

### 5.2 可选校验输入

为了校验 human metadata，可以读取：

```text id="isniv9"
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_summary.json
```

优先使用 `representation_features.jsonl` 中已经复制过来的 human metadata。

如果 `representation_features.jsonl` 中 human fields 缺失或不完整，再读取 Sprint 1Q / 1R 文件进行补全或 cross-check。

---

## 6. 禁止输入

2C 禁止读取：

```text id="bnal7q"
outputs/logs/sprint_2A_real_hidden_state_cache/hidden_states/*.pt
outputs/logs/sprint_2A_hidden_state_cache_baseline/hidden_states/*.pt
```

2C 禁止把以下 2B legacy/debug 文件作为输入契约：

```text id="o0e3cv"
outputs/logs/sprint_2B_representation_features/representation_feature_manifest.jsonl
outputs/logs/sprint_2B_representation_features/input_representation_summary.jsonl
outputs/logs/sprint_2B_representation_features/feature_schema.json
```

如果这些文件存在，只能在 report 中记录为 ignored/deprecated，不得作为 2C 的依赖。

---

## 7. 输出目录与输出文件

默认输出目录：

```text id="mnvplq"
outputs/logs/sprint_2C_probe_dataset/
```

正式输出文件：

```text id="diiu5b"
outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json
```

不要输出：

```text id="vk0n1f"
probe_predictions.jsonl
probe_eval_report.json
probe_model.pkl
guidance_candidate_manifest.jsonl
guidance_candidate_report.json
```

这些属于后续 Sprint。

---

## 8. 允许新增文件

允许新增：

```text id="xva3sq"
src/recover_attention/probe_dataset.py
scripts/18_build_probe_dataset.py
tests/test_probe_dataset.py
docs/codex_tasks/sprint_2C_probe_dataset_construction.md
```

---

## 9. 允许修改文件

允许修改：

```text id="bn0epy"
PROGRESS.md
docs/progress/sprint_2_history.md
```

原则上不需要修改：

```text id="yzkw91"
src/recover_attention/schemas.py
```

如果需要新增少量 allowed target constants，可以放在 `src/recover_attention/probe_dataset.py` 内部，不要为了 2C 大幅修改全局 schema。

---

## 10. 禁止修改文件和目录

禁止修改：

```text id="gw1zyk"
AGENTS.md
README.md
docs/skill/SKILL.md

data/processed/*

outputs/logs/sprint_1Q_*
outputs/logs/sprint_1R_*
outputs/logs/sprint_2A_hidden_state_cache_baseline/*
outputs/logs/sprint_2A_real_hidden_state_cache/*
outputs/logs/sprint_2B_representation_features/*

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/candidate_extraction.py
src/recover_attention/nli_scoring.py
src/recover_attention/semantic_labels.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
```

2C 必须把 2B 输出视为 frozen upstream artifact。

---

## 11. Target 设计

2C 的核心任务是把人工 review 标签映射成 probe target。

### 11.1 probe target 类别

使用以下 target labels：

```text id="q1y1xz"
risk_positive
positive_anchor
negative
hard_negative_or_weak_positive
```

这是 2C 的主 target：

```text id="fo5lve"
probe_target
```

---

## 12. Target mapping 规则

### 12.1 risk_positive

映射为 `risk_positive` 的 human error type：

```text id="nbg0p8"
wrong_numeric_recovery
misleading_entity_or_unit
```

含义：

```text id="w1rcxg"
恢复内容引入错误数字、错误实体、错误单位，属于高风险 anchor/risk 信号。
```

---

### 12.2 positive_anchor

映射为 `positive_anchor` 的规则：

```text id="hiee94"
human_error_type == generic_recovery
AND human_semantic_role indicates critical_number
```

或者字段值中出现等价表达：

```text id="8p3g9t"
critical_number
key_number
important_number
number
```

但必须谨慎：只有当 span 明显是关键数值时，才标为 `positive_anchor`。

如果 `human_error_type == generic_recovery` 但无法确认是 critical number，则不要强行标为 positive_anchor，可降级为 `hard_negative_or_weak_positive`。

---

### 12.3 negative

映射为 `negative` 的情况：

```text id="pgfvvh"
semantic_equivalent_recovery
low-value fragment_recovery
```

更具体地说：

```text id="fx5b50"
human_error_type == semantic_equivalent_recovery
→ negative

human_error_type == fragment_recovery
AND human_guidance_priority in {"low", "none"}
→ negative
```

含义：

```text id="j4f926"
这些样本不应被 probe 当作强 anchor/risk 信号。
```

---

### 12.4 hard_negative_or_weak_positive

映射为 `hard_negative_or_weak_positive` 的情况：

```text id="ekk0qc"
semantically_meaningful_but_recoverable
recoverable but semantically meaningful span
```

可操作规则：

```text id="kycrwc"
human_error_type in {
  "semantically_meaningful_but_recoverable",
  "recoverable_but_semantically_meaningful_span"
}
→ hard_negative_or_weak_positive
```

如果实际数据中没有完全一致的 string，则使用保守 fallback：

```text id="yy5zdz"
human_recoverability_label in {"Recoverable", "Partially Recoverable"}
AND human_attention_anchor_label in {"Weak Anchor", "Medium Anchor"}
AND human_guidance_priority in {"low", "medium"}
→ hard_negative_or_weak_positive
```

含义：

```text id="x9cr74"
这些样本可能含有信号，但不应被当作明确 risk_positive。
```

---

## 13. Unmapped / ambiguous 处理规则

如果某条 feature record 无法通过规则映射出 target：

```text id="mzflz6"
probe_target = "unmapped"
probe_target_usable = false
```

并记录：

```text id="9p9ypv"
target_mapping_status = "unmapped"
target_mapping_reason = "..."
```

不要丢弃 unmapped record。

不要静默跳过。

不要硬凑 target。

如果该 record 因缺少 human fields 无法映射：

```text id="wa86bn"
target_mapping_status = "missing_human_metadata"
probe_target_usable = false
```

---

## 14. probe usage 处理规则

`probe_usage` 只作为过滤建议，不作为唯一 target 来源。

推荐规则：

```text id="kyp5hk"
probe_usage == "include"
→ 可以进入 probe usable pool

probe_usage == "exclude"
→ probe_target_usable = false

probe_usage == "uncertain"
→ 保留记录，但 probe_target_usable = false 或 weak usable，由 report 统计
```

如果实际数据使用其他字符串，不要报错，记录为 unknown probe_usage。

---

## 15. 输出数据集记录粒度

`probe_dataset.jsonl` 的记录粒度必须与 2B 一致：

```text id="qaqvia"
one masked case × one recovered variant
```

也就是说，每条 `representation_features.jsonl` 生成一条 `probe_dataset.jsonl`。

不要把多个 recovered variants 合并成一条。

不要按 `masked_id` 聚合丢失 variant 信息。

---

## 16. probe_dataset.jsonl 推荐字段

每条记录推荐包含：

```json id="zwqbxn"
{
  "probe_dataset_id": "...",
  "feature_id": "...",
  "masked_id": "...",
  "id": "...",
  "unit_id": "...",

  "source_feature_path": "outputs/logs/sprint_2B_representation_features/representation_features.jsonl",
  "source_feature_backend": "representation_features_minimal_v0",

  "probe_target": "risk_positive",
  "probe_target_usable": true,
  "target_mapping_status": "mapped",
  "target_mapping_rule": "human_error_type:wrong_numeric_recovery",
  "target_mapping_reason": "wrong_numeric_recovery is mapped to risk_positive",

  "human_error_type": "wrong_numeric_recovery",
  "human_semantic_role": "...",
  "human_attention_anchor_label": "...",
  "human_recoverability_label": "...",
  "human_guidance_priority": "...",
  "probe_usage": "...",

  "feature_values": {
    "question_original_masked_cosine_mean": 0.1,
    "question_original_recovered_cosine_mean": 0.2,
    "question_masked_recovered_cosine_mean": 0.3,
    "span_original_masked_cosine_mean": null,
    "span_original_recovered_cosine_mean": null,
    "span_masked_recovered_cosine_mean": null,
    "mask_position_original_masked_cosine_mean": null,
    "mask_position_original_recovered_cosine_mean": null,
    "mask_position_masked_recovered_cosine_mean": null
  },

  "feature_arrays": {
    "question_original_masked_cosine_by_layer": [0.1, 0.2],
    "question_original_recovered_cosine_by_layer": [0.1, 0.2],
    "question_masked_recovered_cosine_by_layer": [0.1, 0.2]
  },

  "null_feature_keys": [
    "span_original_masked_cosine_by_layer"
  ],

  "warnings": []
}
```

---

## 17. Feature selection for dataset

2C 不做 feature importance，但需要把 2B feature record 整理成稳定 dataset 字段。

至少保留以下 layer-wise arrays：

```text id="gidrgh"
question_original_masked_cosine_by_layer
question_original_recovered_cosine_by_layer
question_masked_recovered_cosine_by_layer
span_original_masked_cosine_by_layer
span_original_recovered_cosine_by_layer
span_masked_recovered_cosine_by_layer
mask_position_original_masked_cosine_by_layer
mask_position_original_recovered_cosine_by_layer
mask_position_masked_recovered_cosine_by_layer
```

至少保留以下 scalar features，如果存在：

```text id="ro6yf1"
*_mean
*_max
*_min
*_first_layer
*_last_layer
*_delta_last_minus_first
```

如果某些 span / mask_position feature 为 null：

```text id="odycqv"
保留 null
记录 null_feature_keys
不要填 0
不要均值插补
不要删除记录
```

2D 训练时再决定如何处理 null。

---

## 18. Null feature 策略

2B 已记录部分 position pooled features 为 nullable。

2C 的策略是：

```text id="n8t0ps"
保留 nullable features。
不删除对应 records。
不做插补。
不做训练前处理。
```

对每条 record 生成：

```text id="n5mo1e"
null_feature_keys
num_null_features
has_null_position_features
```

这些字段用于 2D 判断是否过滤、插补或增加 missing indicator。

---

## 19. 不做 split

Sprint 2C 不划分：

```text id="ni1nk4"
train
dev
test
k-fold
leave-one-out
```

原因：

```text id="szcppu"
当前只有 20 条小样本，具体评估策略应由 Sprint 2D 训练 baseline 时决定。
```

因此 2C 不输出：

```text id="ot8sgb"
split
fold_id
train_split
dev_split
test_split
```

如果需要标记是否可用于训练，只使用：

```text id="8v0umj"
probe_target_usable
```

---

## 20. probe_dataset_report.json 必需内容

`probe_dataset_report.json` 必须包含：

```json id="f8g93v"
{
  "sprint": "2C",
  "backend": "probe_dataset_mapping_v0",
  "status": "ok",

  "inputs": {
    "representation_features_path": "outputs/logs/sprint_2B_representation_features/representation_features.jsonl",
    "representation_feature_report_path": "outputs/logs/sprint_2B_representation_features/representation_feature_report.json",
    "optional_human_label_source_path": "..."
  },

  "outputs": {
    "probe_dataset_path": "outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl",
    "probe_dataset_report_path": "outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json"
  },

  "source_2B": {
    "source_feature_backend": "representation_features_minimal_v0",
    "num_feature_records": 20,
    "num_masked_groups": 20,
    "num_recovered_variants": 20
  },

  "counts": {
    "num_probe_dataset_records": 20,
    "num_usable_records": 0,
    "num_unusable_records": 0,
    "num_unmapped_records": 0,
    "num_missing_human_metadata_records": 0
  },

  "target_counts": {
    "risk_positive": 0,
    "positive_anchor": 0,
    "negative": 0,
    "hard_negative_or_weak_positive": 0,
    "unmapped": 0
  },

  "human_error_type_counts": {},
  "probe_usage_counts": {},
  "null_feature_counts": {
    "records_with_null_position_features": 0,
    "records_with_any_null_feature": 0
  },

  "boundary": {
    "read_hidden_state_tensors": false,
    "recomputed_representation_features": false,
    "created_train_dev_test_split": false,
    "trained_probe": false,
    "generated_guidance_candidates": false,
    "performed_attention_guidance": false
  },

  "warnings": []
}
```

---

## 21. CLI 要求

新增脚本：

```text id="m77wpx"
scripts/18_build_probe_dataset.py
```

推荐 CLI：

```bash id="phd9sm"
conda run -n recover_attention python scripts/18_build_probe_dataset.py \
  --features outputs/logs/sprint_2B_representation_features/representation_features.jsonl \
  --feature-report outputs/logs/sprint_2B_representation_features/representation_feature_report.json \
  --output-dir outputs/logs/sprint_2C_probe_dataset \
  --backend probe_dataset_mapping_v0 \
  --overwrite
```

可选 human label cross-check：

```bash id="l34r7m"
  --human-labels outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl \
  --sprint-1r-manifest outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl
```

默认情况下，如果 `representation_features.jsonl` 已包含 human metadata，可以不传 human-labels。

---

## 22. 实现建议

核心实现放在：

```text id="e65dus"
src/recover_attention/probe_dataset.py
```

建议函数：

```python id="blt7mb"
load_representation_features(...)
load_representation_feature_report(...)
load_optional_human_labels(...)
index_human_labels_by_masked_id(...)
map_probe_target(...)
extract_feature_values(...)
extract_feature_arrays(...)
build_probe_dataset_record(...)
build_probe_dataset(...)
build_probe_dataset_report(...)
write_jsonl_strict(...)
write_json_strict(...)
```

核心 mapping 函数：

```python id="zt089z"
def map_probe_target(record: dict) -> dict:
    ...
```

返回：

```python id="k20t7d"
{
    "probe_target": "...",
    "probe_target_usable": True,
    "target_mapping_status": "mapped",
    "target_mapping_rule": "...",
    "target_mapping_reason": "..."
}
```

---

## 23. Target mapping 伪代码

建议实现逻辑：

```python id="mnrx07"
def map_probe_target(record: dict) -> dict:
    error_type = normalize(record.get("human_error_type"))
    semantic_role = normalize(record.get("human_semantic_role"))
    guidance_priority = normalize(record.get("human_guidance_priority"))
    recoverability = normalize(record.get("human_recoverability_label"))
    anchor_label = normalize(record.get("human_attention_anchor_label"))
    probe_usage = normalize(record.get("probe_usage"))

    if missing core human metadata:
        return missing_human_metadata()

    if probe_usage == "exclude":
        mapped = map_by_error_type_and_role(...)
        mapped["probe_target_usable"] = False
        mapped["target_mapping_status"] = "excluded_by_probe_usage"
        return mapped

    if error_type in {"wrong_numeric_recovery", "misleading_entity_or_unit"}:
        return risk_positive(...)

    if error_type == "generic_recovery" and semantic_role_is_critical_number(semantic_role):
        return positive_anchor(...)

    if error_type == "semantic_equivalent_recovery":
        return negative(...)

    if error_type == "fragment_recovery" and guidance_priority in {"low", "none"}:
        return negative(...)

    if error_type in {
        "semantically_meaningful_but_recoverable",
        "recoverable_but_semantically_meaningful_span",
    }:
        return hard_negative_or_weak_positive(...)

    if recoverability in {"Recoverable", "Partially Recoverable"} \
       and anchor_label in {"Weak Anchor", "Medium Anchor"} \
       and guidance_priority in {"low", "medium"}:
        return hard_negative_or_weak_positive(...)

    return unmapped(...)
```

Do not silently drop records.

---

## 24. 测试要求

新增测试：

```text id="lu1irs"
tests/test_probe_dataset.py
```

至少覆盖：

```text id="a2km8r"
1. 能读取 representation_features.jsonl。
2. 能读取 representation_feature_report.json。
3. 不读取 hidden_states/*.pt。
4. 不依赖 representation_feature_manifest.jsonl。
5. 不依赖 input_representation_summary.jsonl。
6. 不依赖 feature_schema.json。
7. 每条 feature record 生成一条 probe dataset record。
8. wrong_numeric_recovery → risk_positive。
9. misleading_entity_or_unit → risk_positive。
10. generic_recovery + critical_number → positive_anchor。
11. semantic_equivalent_recovery → negative。
12. low-value fragment_recovery → negative。
13. semantically_meaningful_but_recoverable → hard_negative_or_weak_positive。
14. recoverable but semantically meaningful span → hard_negative_or_weak_positive。
15. 缺少 human metadata → unmapped / missing_human_metadata，不崩溃。
16. unmapped record 不被丢弃。
17. probe_usage=exclude 时 probe_target_usable=false。
18. null span / mask_position features 被保留为 null。
19. 输出中不出现 NaN / Inf。
20. 不生成 split / fold_id / train/dev/test。
21. 不生成 probe_predictions.jsonl。
22. 不生成 probe_model.pkl。
23. report 正确统计 target_counts。
24. report 正确统计 null_feature_counts。
25. CLI smoke test 能生成 probe_dataset.jsonl 和 probe_dataset_report.json。
```

---

## 25. 必须运行的命令

Preflight：

```bash id="fjcyfq"
git status --short
```

Targeted test：

```bash id="efndmb"
conda run -n recover_attention python -m pytest tests/test_probe_dataset.py -q
```

Run 2C：

```bash id="kc1nip"
conda run -n recover_attention python scripts/18_build_probe_dataset.py \
  --features outputs/logs/sprint_2B_representation_features/representation_features.jsonl \
  --feature-report outputs/logs/sprint_2B_representation_features/representation_feature_report.json \
  --output-dir outputs/logs/sprint_2C_probe_dataset \
  --backend probe_dataset_mapping_v0 \
  --overwrite
```

Full test：

```bash id="p0jd6b"
conda run -n recover_attention python -m pytest -q
```

Final check：

```bash id="juv9e0"
git diff --name-only
git status --short
```

---

## 26. 验收标准

完成 Sprint 2C 时必须满足：

```text id="kl6k4d"
1. 新增 probe dataset construction pipeline。
2. 读取 2B 正式输出 representation_features.jsonl。
3. 读取 2B 正式输出 representation_feature_report.json。
4. 不依赖 2B legacy/debug outputs。
5. 不读取 hidden_states/*.pt。
6. 不重新提取 representation features。
7. 不调用 HF model。
8. 不使用 GPU。
9. 每条 feature record 生成一条 probe dataset record。
10. 输出 probe_dataset.jsonl。
11. 输出 probe_dataset_report.json。
12. 实现 risk_positive mapping。
13. 实现 positive_anchor mapping。
14. 实现 negative mapping。
15. 实现 hard_negative_or_weak_positive mapping。
16. unmapped records 不被丢弃。
17. 缺少 human metadata 不导致流程中断。
18. null features 保留为 null，不做插补。
19. probe_target_usable 字段存在。
20. 不生成 train/dev/test split。
21. 不训练 probe。
22. 不生成 probe predictions。
23. 不生成 probe model。
24. 不生成 guidance candidate manifest。
25. 不执行 attention guidance。
26. 不修改 Sprint 1Q / 1R / 2A / 2A-real / 2B outputs。
27. targeted pytest passed。
28. full pytest passed。
29. PROGRESS.md 已更新。
30. docs/progress/sprint_2_history.md 已更新。
```

---

## 27. PROGRESS.md 更新建议

在当前阶段补充：

```text id="xc9jrx"
Sprint 2C 已完成：Probe Dataset Construction。

probe_dataset_mapping_v0 已读取 Sprint 2B 的正式 representation features，并基于 Sprint 1Q / 1R human metadata 构造 probe targets。

正式输出：
- outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
- outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json

Target mapping:
- wrong_numeric_recovery / misleading_entity_or_unit → risk_positive
- generic_recovery + critical_number → positive_anchor
- semantic_equivalent_recovery / low-value fragment_recovery → negative
- semantically meaningful but recoverable cases → hard_negative_or_weak_positive

本阶段未读取 hidden-state tensors，未重新抽取 features，未划分 train/dev/test，未训练 probe，未生成 guidance candidate，未执行 attention guidance。

下一步建议是 Sprint 2D：Probe Training Baseline。
```

---

## 28. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text id="nz0bm7"
## Sprint 2C：Probe Dataset Construction

### Goal

Construct a probe-ready dataset by mapping human review labels to probe targets and joining them with Sprint 2B representation features.

### Inputs

- outputs/logs/sprint_2B_representation_features/representation_features.jsonl
- outputs/logs/sprint_2B_representation_features/representation_feature_report.json

Optional cross-check inputs:

- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_human_review_labels_template.jsonl
- outputs/logs/sprint_1Q_real_signal_quality_review/sprint_1Q_to_2A_manifest.jsonl

### Outputs

- outputs/logs/sprint_2C_probe_dataset/probe_dataset.jsonl
- outputs/logs/sprint_2C_probe_dataset/probe_dataset_report.json

### Target Mapping

- wrong_numeric_recovery → risk_positive
- misleading_entity_or_unit → risk_positive
- generic_recovery + critical_number → positive_anchor
- semantic_equivalent_recovery → negative
- low-value fragment_recovery → negative
- semantically meaningful but recoverable cases → hard_negative_or_weak_positive

### Not Done

- No hidden-state tensor loading.
- No representation feature extraction.
- No train/dev/test split.
- No probe training.
- No probe predictions.
- No guidance candidate manifest.
- No attention steering.

### Next

Sprint 2D：Probe Training Baseline.
```

---

## 29. 下一步边界

Sprint 2C 完成后，才进入：

```text id="zpg5wi"
Sprint 2D：Probe Training Baseline
```

2D 才处理：

```text id="g3jrvu"
logistic regression
ridge classifier
leave-one-out
k-fold
prediction output
probe_eval_report.json
probe_model.pkl
```

2C 不提前做这些。
