文件名：docs/codex_tasks/sprint_2G_full_scale_weak_labeled_500_run.md

# Sprint 2G：Full-scale Weak-labeled 500-case Dry Run

## 0. Sprint 名称

Sprint 2G — Full-scale Weak-labeled 500-case Dry Run

## 1. 任务定位

本 sprint 是 Sprint 2 的 full-scale extension。

它不是 Sprint 3。

它不执行真实 attention guidance。

它不验证 hallucination reduction。

它的目标是把 Sprint 2 已完成的 20 条 human-reviewed dry-run 闭环扩展到至少 500 条 weak-labeled cases，用于验证：

```text id="uwv6oq"
1. pipeline 是否能扩展到 500 条规模。
2. hidden-state cache / representation features 是否能稳定批量生成。
3. weak-labeled probe dataset 是否能构造。
4. full-scale probe baseline 是否能训练。
5. guidance candidate dry run 是否能生成更大规模统计。
6. full-scale figures / summary 是否能支持 Sprint 3A 设计。
```

本 sprint 的核心结论边界是：

```text id="n16log"
500-case weak-labeled diagnostic dry run
```

不是：

```text id="ztun9s"
500-case human-supervised validation
```

也不是：

```text id="4get1x"
executed attention steering experiment
```

---

## 2. 当前背景

Sprint 2 已完成 20 条 human-reviewed 最小闭环：

```text id="ic3pk5"
Sprint 1Q / 1R human review
→ Sprint 2A-real hidden-state cache
→ Sprint 2B representation features
→ Sprint 2C probe dataset
→ Sprint 2D probe baseline
→ Sprint 2E guidance candidate manifest dry run
→ Sprint 2F mini closed-loop report
→ Sprint 2 final checkpoint visualization summary
```

当前 20 条结果是 human-reviewed baseline，必须保留，不得覆盖。

Sprint 2G 要新建 full-scale 输出路径，独立于已有 20 条 Sprint 2 outputs。

---

## 3. 2G 总目标

Sprint 2G 的目标是：

```text id="xdu78f"
在至少 500 条 weak-labeled cases 上重跑一条 full-scale dry-run pipeline，
形成 500-case hidden-state → representation feature → weak probe dataset → probe baseline → guidance candidate → stage summary 的完整诊断闭环。
```

目标样本规模：

```text id="nnyce5"
target_num_cases >= 500
```

如果实际可用样本不足 500，必须在 report 中记录：

```text id="hgvqi2"
requested_num_cases
available_num_cases
actual_num_cases
shortfall_reason
```

不能静默退化成 20 条或 100 条。

---

## 4. 非目标

Sprint 2G 不做：

```text id="xdpu4x"
不覆盖 20 条 human-reviewed Sprint 2 outputs
不把 weak labels 伪装成人工标签
不新增人工审阅流程
不要求 500 条人工标注
不执行真实 attention guidance
不修改 transformer attention weights
不注入 attention mask
不重跑 guided CoT generation
不验证 answer accuracy improvement
不验证 hallucination reduction
不声称 probe 已泛化
不声称 weak-labeled result 等同于 human-labeled result
不进入 Sprint 3A implementation
```

---

## 5. 核心边界声明

Sprint 2G 必须在所有 report / summary 中写明：

```text id="v72t8p"
This is a weak-labeled full-scale dry run.
It scales the Sprint 2 diagnostic pipeline to at least 500 cases.
It does not execute attention steering.
It does not validate hallucination reduction.
```

不得出现以下无边界表述：

```text id="xwr91z"
attention guidance succeeded
hallucination was reduced
answer accuracy improved
500-case human-labeled validation
probe generalization was proven
```

除非明确是否定句或边界说明。

---

## 6. 输出目录总约定

所有 Sprint 2G 产物必须写入：

```text id="d42pa4"
outputs/logs/sprint_2G_full_scale_500/
```

禁止写入或覆盖：

```text id="w4i58m"
outputs/logs/sprint_2A_real_hidden_state_cache/
outputs/logs/sprint_2B_representation_features/
outputs/logs/sprint_2C_probe_dataset/
outputs/logs/sprint_2D_probe_training_baseline/
outputs/logs/sprint_2E_guidance_candidate_dry_run/
outputs/logs/sprint_2F_mini_closed_loop_report/
outputs/logs/sprint_2_stage_summary/
```

---

## 7. 推荐目录结构

Sprint 2G 输出目录建议为：

```text id="ef9fj7"
outputs/logs/sprint_2G_full_scale_500/
  00_manifest/
  01_downstream/
  02_hidden_state_cache/
  03_representation_features/
  04_weak_probe_dataset/
  05_probe_training/
  06_guidance_candidates/
  07_stage_summary/
  figures/
```

---

## 8. 数据来源策略

Sprint 2G 需要至少 500 条 cases。

优先从已有 real downstream / processed artifacts 中抽取，不重新设计 dataset。

候选来源可以包括：

```text id="qxluap"
data/processed/*
outputs/logs/sprint_1N_real_downstream/*
outputs/logs/sprint_1P_upgraded_downstream/*
```

具体使用哪个来源由当前仓库已有产物决定。

如果已有 real downstream 产物不足 500 条，则可以从原始 question source 重新构造 500 条 candidate / ablation / masked question pipeline。

但必须满足：

```text id="iw7dnh"
1. 不覆盖现有 processed baseline。
2. 2G 产物全部写入 outputs/logs/sprint_2G_full_scale_500/。
3. report 中记录 source artifact。
4. report 中记录 sampling rule。
5. report 中记录 requested_num_cases=500。
```

---

## 9. Label 策略

Sprint 2G 默认使用 weak / auto labels。

不得声称这些 labels 是 human labels。

每条 weak-labeled record 必须包含：

```text id="jzdnqh"
label_source = "weak_auto"
label_backend = ...
label_confidence = ...
label_rule = ...
human_reviewed = false
```

如果某条记录来自已有 human-reviewed 20 条，则必须标记：

```text id="bajjwz"
label_source = "human_review"
human_reviewed = true
```

但默认 2G 不应依赖 20 条 human labels 扩大成 500 条伪人工标签。

---

## 10. 推荐 weak target mapping

Sprint 2G 可以使用现有 recovery scoring / NLI / attention anchor outputs 构造 weak targets。

推荐 weak probe targets 保持与 2C 一致：

```text id="rrn9wa"
risk_positive
positive_anchor
negative
hard_negative_or_weak_positive
unmapped
```

推荐映射原则：

### 10.1 risk_positive

可由以下信号映射：

```text id="qa6vc4"
wrong_numeric_recovery
misleading_entity_or_unit
contradiction-heavy recovery
answer-changing recovery with high semantic risk
```

### 10.2 positive_anchor

可由以下信号映射：

```text id="rch5an"
critical number
key entity
important condition
high semantic necessity
low recoverability
strong attention anchor evidence
```

### 10.3 negative

可由以下信号映射：

```text id="q4otid"
semantic equivalent recovery
low-value fragment recovery
recoverable low-priority span
non-critical span
```

### 10.4 hard_negative_or_weak_positive

可由以下信号映射：

```text id="jfs86d"
generic recovery
partially recoverable but meaningful span
medium anchor
ambiguous recoverability
```

### 10.5 unmapped

如果 weak evidence 不足：

```text id="s5cnx4"
probe_target = "unmapped"
probe_target_usable = false
```

不得强行映射。

---

## 11. 与 20 条 human-reviewed baseline 的关系

Sprint 2G 必须保留 20 条 human-reviewed baseline 作为对照。

2G report 中必须包含：

```text id="vfn35o"
20-case human-reviewed baseline summary
500-case weak-labeled full-scale summary
comparison caveat
```

比较时必须写明：

```text id="evm8a0"
20-case results are human-reviewed diagnostic baseline.
500-case results are weak-labeled diagnostic scale-up.
They are not directly equivalent.
```

不得把 500 条 weak-labeled 指标直接说成比 20 条更可靠。

---

## 12. Pipeline 阶段

Sprint 2G 建议分成以下阶段实现。

---

# Phase 0：Full-scale Manifest Construction

## 12.0.1 目标

构造 500-case full-scale manifest。

输出：

```text id="weqwk5"
outputs/logs/sprint_2G_full_scale_500/00_manifest/full_scale_manifest.jsonl
outputs/logs/sprint_2G_full_scale_500/00_manifest/full_scale_manifest_report.json
```

manifest 每条记录必须包含：

```text id="kr0lpg"
full_scale_id
source_id
question
answer
source_artifact
sampling_index
sampling_rule
requested_num_cases
actual_num_cases
```

如果已有字段无法全部填充，允许 nullable，但必须记录 warning。

---

# Phase 1：Full-scale Downstream Construction

## 12.1.1 目标

对 500 条 cases 构造或收集 downstream artifacts：

```text id="wb7msc"
candidate spans
ablation units
masked questions
recover outputs
recover scores
weak evidence
```

输出目录：

```text id="d7e1ys"
outputs/logs/sprint_2G_full_scale_500/01_downstream/
```

推荐输出：

```text id="zqre9w"
candidate_spans_500.jsonl
ablation_units_500.jsonl
masked_questions_500.jsonl
recover_outputs_500.jsonl
recover_scores_500.jsonl
weak_evidence_500.jsonl
weak_labels_500.jsonl
downstream_report.json
```

如果已有 Sprint 1N / 1P artifacts 可以复用，则只读复用，不重复计算；但必须把复用来源写入 report。

---

# Phase 2：Full-scale Hidden-state Cache

## 12.2.1 目标

对 500 条 weak-labeled cases 缓存 hidden states。

输出目录：

```text id="sxzijr"
outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache/
```

推荐输出：

```text id="frlxgp"
hidden_state_manifest.jsonl
hidden_state_cache_report.json
token_alignment_report.json
real_run_metadata.json
hidden_states/*.pt
```

注意：

```text id="jp98xt"
500 cases × original/masked/recovered = 至少 1500 hidden-state inputs
```

如果每个 case 有多个 recovered variants，数量会更多。

必须记录：

```text id="z37os5"
num_cases
num_inputs_total
num_hidden_state_files
layer_indices
model_name
backend
device
dtype
runtime_seconds
failure_count
skipped_count
```

如果 GPU / VRAM 不足，可以支持 batch size / limit / resume。

---

# Phase 3：Full-scale Representation Features

## 12.3.1 目标

从 500-case hidden-state cache 抽取 representation features。

输出目录：

```text id="jp6yvs"
outputs/logs/sprint_2G_full_scale_500/03_representation_features/
```

推荐输出：

```text id="c5a494"
representation_features.jsonl
representation_feature_report.json
```

可以复用 Sprint 2B 的 `representation_features_minimal_v0` 思路，但输出路径必须是 2G 路径。

必须记录：

```text id="jxxhbm"
num_feature_records
num_masked_groups
num_recovered_variants
num_skipped_groups
num_null_position_features
feature_backend
source_cache_dir
```

---

# Phase 4：Full-scale Weak Probe Dataset

## 12.4.1 目标

把 500-case weak labels 与 representation features 合并，构造 weak probe dataset。

输出目录：

```text id="g19rjo"
outputs/logs/sprint_2G_full_scale_500/04_weak_probe_dataset/
```

推荐输出：

```text id="q10113"
weak_probe_dataset.jsonl
weak_probe_dataset_report.json
```

每条 record 必须包含：

```text id="qkyre4"
probe_record_id
full_scale_id
feature_id
probe_target
probe_target_usable
label_source
label_backend
label_rule
label_confidence
human_reviewed
feature_values
feature_arrays
num_null_features
warnings
```

必须区分：

```text id="fwpces"
human_reviewed = true
human_reviewed = false
```

---

# Phase 5：Full-scale Probe Training

## 12.5.1 目标

在 weak probe dataset 上训练 full-scale probe baseline。

输出目录：

```text id="n0bmpu"
outputs/logs/sprint_2G_full_scale_500/05_probe_training/
```

推荐输出：

```text id="kql2kd"
probe_predictions.jsonl
probe_eval_report.json
probe_model.pkl
```

训练设置建议：

```text id="e7f7nf"
model = ridge_classifier_ovr_v0
cv = stratified_k_fold
num_folds = 5
seed = 42
null feature strategy = zero impute + missing indicators
feature scaling = train-fold z-score
```

如果某类样本不足 5 条：

```text id="up5257"
自动降低 num_folds
或改用 leave_one_out / holdout fallback
并在 report 中记录 warning
```

2G 的 probe metrics 必须标记：

```text id="gmp9id"
weak-labeled metrics
```

不得写成 human-supervised metrics。

---

# Phase 6：Full-scale Guidance Candidate Dry Run

## 12.6.1 目标

把 full-scale probe predictions 转成 planned-only guidance candidates。

输出目录：

```text id="jdf9dc"
outputs/logs/sprint_2G_full_scale_500/06_guidance_candidates/
```

推荐输出：

```text id="zlsg5g"
guidance_candidate_manifest.jsonl
guidance_candidate_report.json
```

candidate action mapping 与 2E 保持一致：

```text id="tl9amd"
risk_positive → increase_attention_to_original_span
positive_anchor → preserve_original_span_attention
hard_negative_or_weak_positive → review_before_guidance
negative → no_guidance
```

所有记录必须包含：

```text id="zbwwnr"
dry_run = true
execution_status = "planned_only"
will_modify_attention = false
will_run_model = false
will_change_answer = false
```

---

# Phase 7：Full-scale Summary and Figures

## 12.7.1 目标

生成 500-case full-scale summary 和 figures。

输出目录：

```text id="alqypl"
outputs/logs/sprint_2G_full_scale_500/07_stage_summary/
```

推荐输出：

```text id="nl2qu9"
full_scale_500_summary.md
full_scale_500_audit.json
figures/full_scale_pipeline_overview.png
figures/weak_target_counts.png
figures/probe_metrics_vs_baseline.png
figures/guidance_candidate_action_counts.png
figures/guidance_confidence_counts.png
figures/20_vs_500_comparison.png
figures/boundary_summary.png
```

---

## 13. 命名与 ID 规则

所有 2G 记录必须有独立 ID，避免和 20-case baseline 混淆。

推荐：

```text id="eu82vm"
full_scale_id = fs500_000001
probe_record_id = fs500_000001::probe
guidance_candidate_id = fs500_000001::guidance
```

如果一个 case 有多个 ablation unit / recovered variant，则：

```text id="v2slrh"
fs500_000001::unit_000
fs500_000001::unit_000::recover_000
```

---

## 14. 必须阅读的文件

开始实现前，先阅读：

```text id="zuc7rs"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_2_history.md

docs/codex_tasks/sprint_2F_mini_closed_loop_report.md
docs/codex_tasks/sprint_2_final_checkpoint_visualization_summary.md

src/recover_attention/hidden_state_cache.py
src/recover_attention/representation_features.py
src/recover_attention/probe_dataset.py
src/recover_attention/probe_training.py
src/recover_attention/guidance_candidates.py
src/recover_attention/stage_summary.py

scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/18_build_probe_dataset.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
scripts/22_write_sprint_2_stage_summary.py
```

如果本 task card 已存在，也阅读：

```text id="efq4c1"
docs/codex_tasks/sprint_2G_full_scale_weak_labeled_500_run.md
```

---

## 15. 允许新增文件

允许新增：

```text id="b2mud8"
src/recover_attention/full_scale_manifest.py
src/recover_attention/weak_probe_dataset.py
src/recover_attention/full_scale_summary.py

scripts/23_build_full_scale_manifest.py
scripts/24_build_full_scale_weak_labels.py
scripts/25_build_full_scale_weak_probe_dataset.py
scripts/26_write_full_scale_500_summary.py

tests/test_full_scale_manifest.py
tests/test_weak_probe_dataset.py
tests/test_full_scale_summary.py

docs/codex_tasks/sprint_2G_full_scale_weak_labeled_500_run.md
```

允许复用现有 16/17/19/20 脚本，但必须写入 2G 输出路径，不得覆盖 2A～2F。

---

## 16. 允许修改文件

允许修改：

```text id="z1vdtq"
PROGRESS.md
docs/progress/sprint_2_history.md
```

如果为了支持 full-scale 输出路径，需要对现有脚本增加 CLI 参数，允许小范围修改：

```text id="tdzbud"
scripts/16_cache_hidden_states.py
scripts/17_extract_representation_features.py
scripts/19_train_probe_baseline.py
scripts/20_build_guidance_candidate_manifest.py
```

但必须满足：

```text id="ivczrc"
1. 不破坏原有 20-case Sprint 2 用法。
2. 旧测试必须通过。
3. 新参数必须向后兼容。
4. 不改变 2A～2F 默认输出契约。
```

---

## 17. 禁止修改文件和目录

禁止修改：

```text id="vb8yuy"
AGENTS.md
README.md
docs/reasoning-aware-attention-guidance/SKILL.md

outputs/logs/sprint_2A_real_hidden_state_cache/*
outputs/logs/sprint_2B_representation_features/*
outputs/logs/sprint_2C_probe_dataset/*
outputs/logs/sprint_2D_probe_training_baseline/*
outputs/logs/sprint_2E_guidance_candidate_dry_run/*
outputs/logs/sprint_2F_mini_closed_loop_report/*
outputs/logs/sprint_2_stage_summary/*
```

禁止删除或覆盖：

```text id="opsj3x"
20-case human-reviewed Sprint 2 outputs
```

---

## 18. 执行方式：必须分阶段，不要一口气全跑

由于 500 条会明显更耗时，必须分阶段执行。

建议每个 phase 都支持：

```text id="m9o91v"
--limit
--resume
--overwrite
--dry-run
```

其中：

```text id="pq3jhs"
--limit 用于小规模 smoke test
--resume 用于断点续跑
--overwrite 只允许覆盖 2G 自己的输出
--dry-run 只检查输入和计划，不写大文件
```

推荐执行顺序：

```text id="dm54c6"
Phase 0：manifest
Phase 1：weak labels / downstream
Phase 2：hidden-state cache
Phase 3：representation features
Phase 4：weak probe dataset
Phase 5：probe training
Phase 6：guidance candidates
Phase 7：summary / figures
```

---

## 19. Smoke test 要求

正式跑 500 条前，必须先跑小规模 smoke test。

推荐：

```text id="g2e5z7"
--limit 10
```

Smoke test 输出目录：

```text id="s3zk9q"
outputs/logs/sprint_2G_full_scale_500_smoke/
```

Smoke test 通过后再跑正式 500 条。

不得用 smoke test 结果覆盖正式 500 条结果。

---

## 20. 推荐命令：Phase 0 Manifest

```bash id="gh7v5e"
conda run -n recover_attention python scripts/23_build_full_scale_manifest.py \
  --output-dir outputs/logs/sprint_2G_full_scale_500/00_manifest \
  --target-num-cases 500 \
  --backend full_scale_manifest_v0 \
  --seed 42 \
  --overwrite
```

Smoke：

```bash id="kpiet1"
conda run -n recover_attention python scripts/23_build_full_scale_manifest.py \
  --output-dir outputs/logs/sprint_2G_full_scale_500_smoke/00_manifest \
  --target-num-cases 10 \
  --backend full_scale_manifest_v0 \
  --seed 42 \
  --overwrite
```

---

## 21. 推荐命令：Phase 1 Weak Labels

```bash id="jc0gik"
conda run -n recover_attention python scripts/24_build_full_scale_weak_labels.py \
  --manifest outputs/logs/sprint_2G_full_scale_500/00_manifest/full_scale_manifest.jsonl \
  --output-dir outputs/logs/sprint_2G_full_scale_500/01_downstream \
  --backend weak_label_mapping_v0 \
  --overwrite
```

---

## 22. 推荐命令：Phase 2 Hidden-state Cache

```bash id="h0ylav"
conda run -n recover_attention python scripts/16_cache_hidden_states.py \
  --input outputs/logs/sprint_2G_full_scale_500/01_downstream/weak_labels_500.jsonl \
  --output-dir outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache \
  --backend hf_local_causal_lm_hidden_states_v0 \
  --model-path D:/models/Qwen2.5-7B-Instruct \
  --device auto \
  --dtype auto \
  --layer-indices 0 8 16 24 27 \
  --mask-token "[MASK]" \
  --overwrite
```

如果本地模型路径不同，不要猜路径；读取已有配置或让用户手动替换。

必须支持断点续跑：

```bash id="f69q3g"
--resume
```

如果脚本当前不支持 `--resume`，可以先不实现，但必须在 report 中提醒。

---

## 23. 推荐命令：Phase 3 Representation Features

```bash id="u412bg"
conda run -n recover_attention python scripts/17_extract_representation_features.py \
  --input-manifest outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache/hidden_state_manifest.jsonl \
  --input-report outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache/hidden_state_cache_report.json \
  --alignment-report outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache/token_alignment_report.json \
  --metadata outputs/logs/sprint_2G_full_scale_500/02_hidden_state_cache/real_run_metadata.json \
  --output-dir outputs/logs/sprint_2G_full_scale_500/03_representation_features \
  --backend representation_features_minimal_v0 \
  --overwrite
```

---

## 24. 推荐命令：Phase 4 Weak Probe Dataset

```bash id="mm8eo6"
conda run -n recover_attention python scripts/25_build_full_scale_weak_probe_dataset.py \
  --features outputs/logs/sprint_2G_full_scale_500/03_representation_features/representation_features.jsonl \
  --feature-report outputs/logs/sprint_2G_full_scale_500/03_representation_features/representation_feature_report.json \
  --weak-labels outputs/logs/sprint_2G_full_scale_500/01_downstream/weak_labels_500.jsonl \
  --output-dir outputs/logs/sprint_2G_full_scale_500/04_weak_probe_dataset \
  --backend weak_probe_dataset_mapping_v0 \
  --overwrite
```

---

## 25. 推荐命令：Phase 5 Probe Training

```bash id="oc9l06"
conda run -n recover_attention python scripts/19_train_probe_baseline.py \
  --dataset outputs/logs/sprint_2G_full_scale_500/04_weak_probe_dataset/weak_probe_dataset.jsonl \
  --dataset-report outputs/logs/sprint_2G_full_scale_500/04_weak_probe_dataset/weak_probe_dataset_report.json \
  --output-dir outputs/logs/sprint_2G_full_scale_500/05_probe_training \
  --backend probe_training_baseline_v0 \
  --model ridge_classifier_ovr_v0 \
  --cv stratified_k_fold \
  --num-folds 5 \
  --seed 42 \
  --overwrite
```

如果类别分布不支持 5-fold，自动降级并记录 warning。

---

## 26. 推荐命令：Phase 6 Guidance Candidates

```bash id="ojr23o"
conda run -n recover_attention python scripts/20_build_guidance_candidate_manifest.py \
  --predictions outputs/logs/sprint_2G_full_scale_500/05_probe_training/probe_predictions.jsonl \
  --eval-report outputs/logs/sprint_2G_full_scale_500/05_probe_training/probe_eval_report.json \
  --output-dir outputs/logs/sprint_2G_full_scale_500/06_guidance_candidates \
  --backend guidance_candidate_dry_run_v0 \
  --overwrite
```

---

## 27. 推荐命令：Phase 7 Summary

```bash id="bbfopv"
conda run -n recover_attention python scripts/26_write_full_scale_500_summary.py \
  --root-dir outputs/logs/sprint_2G_full_scale_500 \
  --output-dir outputs/logs/sprint_2G_full_scale_500/07_stage_summary \
  --backend full_scale_500_summary_v0 \
  --overwrite
```

---

## 28. 全量测试要求

每个新增模块都必须有 targeted pytest。

推荐：

```bash id="po0m3x"
conda run -n recover_attention python -m pytest tests/test_full_scale_manifest.py -q
conda run -n recover_attention python -m pytest tests/test_weak_probe_dataset.py -q
conda run -n recover_attention python -m pytest tests/test_full_scale_summary.py -q
```

最终 full pytest：

```bash id="p9zmj3"
conda run -n recover_attention python -m pytest -q
```

所有命令必须串行执行。

不要并行启动多个 `conda run`。

---

## 29. Windows 执行稳定性要求

由于之前出现过 Windows 并行 `conda run` 临时文件占用，本 sprint 必须串行执行。

禁止并行：

```text id="qkxhv2"
conda run ...
conda run ...
```

如果出现 Windows 临时文件占用：

```text id="vud6fn"
1. 不立即判断为代码失败。
2. 确认是否有并行 conda run。
3. 串行重跑当前 phase。
4. 串行重跑 targeted pytest。
5. 如果串行仍失败，再记录为真实失败。
```

---

## 30. 可视化要求

Sprint 2G 必须生成以下图：

```text id="uvngiw"
full_scale_pipeline_overview.png
weak_target_counts.png
probe_metrics_vs_baseline.png
guidance_candidate_action_counts.png
guidance_confidence_counts.png
20_vs_500_comparison.png
boundary_summary.png
```

所有图必须写入：

```text id="wu7euj"
outputs/logs/sprint_2G_full_scale_500/07_stage_summary/figures/
```

图必须明确标注：

```text id="lm5nm2"
weak-labeled
dry-run
not attention steering
```

---

## 31. Summary Markdown 要求

输出：

```text id="exnodw"
outputs/logs/sprint_2G_full_scale_500/07_stage_summary/full_scale_500_summary.md
```

必须包含：

```text id="se1r7w"
1. Executive Summary
2. Data Source and Sampling
3. Weak Labeling Rules
4. Full-scale Pipeline Overview
5. Hidden-state Cache Summary
6. Representation Feature Summary
7. Weak Probe Dataset Summary
8. Probe Training Summary
9. Guidance Candidate Dry-run Summary
10. 20-case Human-reviewed Baseline vs 500-case Weak-labeled Run
11. Boundary and Non-claims
12. Engineering Stability Notes
13. Sprint 3A Readiness
```

必须写明：

```text id="sc3a73"
This 500-case run is weak-labeled.
It is not a 500-case human-reviewed validation.
It does not execute attention steering.
It does not validate hallucination reduction.
```

---

## 32. Audit JSON 要求

输出：

```text id="iq7n8c"
outputs/logs/sprint_2G_full_scale_500/07_stage_summary/full_scale_500_audit.json
```

建议字段：

```json id="s6j1gq"
{
  "sprint": "2G",
  "status": "ok",
  "requested_num_cases": 500,
  "actual_num_cases": 500,
  "label_source": "weak_auto",
  "human_reviewed_full_scale": false,
  "output_root": "outputs/logs/sprint_2G_full_scale_500",
  "phase_status": {
    "manifest": "ok",
    "weak_labels": "ok",
    "hidden_state_cache": "ok",
    "representation_features": "ok",
    "weak_probe_dataset": "ok",
    "probe_training": "ok",
    "guidance_candidates": "ok",
    "summary": "ok"
  },
  "boundary": {
    "dry_run_only": true,
    "executed_attention_steering": false,
    "validated_answer_accuracy_improvement": false,
    "validated_hallucination_reduction": false,
    "overwrote_20_case_outputs": false
  },
  "warnings": []
}
```

---

## 33. PROGRESS.md 更新建议

完成后在当前阶段补充：

```text id="lqim9t"
Sprint 2G 已完成：Full-scale Weak-labeled 500-case Dry Run。

full_scale_500 pipeline 已在至少 500 条 weak-labeled cases 上完成：
- full-scale manifest
- weak labels
- hidden-state cache
- representation features
- weak probe dataset
- probe training
- guidance candidate dry run
- full-scale summary / figures

正式输出目录：
outputs/logs/sprint_2G_full_scale_500/

核心结论：
- Sprint 2G 将 Sprint 2 dry-run diagnostic pipeline 扩展到 500-case weak-labeled scale。
- 500-case 结果是 weak-labeled diagnostic evidence，不是 human-reviewed validation。
- Attention guidance 尚未执行。
- Hallucination reduction 尚未验证。
- 20-case human-reviewed Sprint 2 outputs 未被覆盖。

下一步建议是 Sprint 3A：Attention Steering Interface Design。
```

---

## 34. docs/progress/sprint_2_history.md 更新建议

新增小节：

```text id="v6tpql"
## Sprint 2G：Full-scale Weak-labeled 500-case Dry Run

### Goal

Scale Sprint 2 dry-run diagnostic loop from 20 human-reviewed cases to at least 500 weak-labeled cases.

### Output Root

outputs/logs/sprint_2G_full_scale_500/

### Phases

1. Full-scale manifest
2. Weak labels
3. Hidden-state cache
4. Representation features
5. Weak probe dataset
6. Probe training
7. Guidance candidate dry run
8. Full-scale summary and figures

### Boundary

This is a weak-labeled full-scale dry run.

It is not:
- human-reviewed 500-case validation
- executed attention steering
- hallucination reduction validation
- answer accuracy improvement validation

### Next

Sprint 3A：Attention Steering Interface Design.
```

---

## 35. 验收标准

Sprint 2G 完成时必须满足：

```text id="h8wpiu"
1. 实际处理 cases >= 500，或明确记录不足原因。
2. 所有 2G 输出写入 outputs/logs/sprint_2G_full_scale_500/。
3. 未覆盖 20-case human-reviewed Sprint 2 outputs。
4. 生成 full_scale_manifest.jsonl。
5. 生成 weak_labels_500.jsonl。
6. 生成 hidden_state_manifest.jsonl。
7. 生成 hidden_state_cache_report.json。
8. 生成 token_alignment_report.json。
9. 生成 representation_features.jsonl。
10. 生成 representation_feature_report.json。
11. 生成 weak_probe_dataset.jsonl。
12. 生成 weak_probe_dataset_report.json。
13. 生成 probe_predictions.jsonl。
14. 生成 probe_eval_report.json。
15. 生成 guidance_candidate_manifest.jsonl。
16. 生成 guidance_candidate_report.json。
17. 生成 full_scale_500_summary.md。
18. 生成 full_scale_500_audit.json。
19. 生成 full-scale figures。
20. 所有 weak labels 标记 label_source=weak_auto。
21. 不把 weak labels 伪装成人工标签。
22. 不执行 attention steering。
23. 不验证 hallucination reduction。
24. 不声称 answer accuracy improvement。
25. targeted pytest passed。
26. full pytest passed。
27. PROGRESS.md 已更新。
28. docs/progress/sprint_2_history.md 已更新。
```

---

## 36. 下一步

Sprint 2G 完成后，Sprint 2 的 full-scale diagnostic extension 完成。

下一步进入：

```text id="w4mh1z"
Sprint 3A：Attention Steering Interface Design
```

Sprint 3A 才开始设计真实 attention steering。

Sprint 2G 的正确表述是：

```text id="z8zi8r"
Sprint 2G scaled the Sprint 2 diagnostic dry-run loop to at least 500 weak-labeled cases.
```

不得表述为：

```text id="nn1kqm"
Sprint 2G proved hallucination reduction.
Sprint 2G executed attention guidance.
Sprint 2G validated answer accuracy improvement.
```
