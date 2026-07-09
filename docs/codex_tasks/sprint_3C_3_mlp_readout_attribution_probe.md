# Sprint 3C-3: MLP Readout Attribution Probe

## 定位

本 sprint 是 3C-2 之后的 **detection / diagnosis 收敛 sprint**。

它不是 full Sprint 3C，不是 2000-scale rerun，不是 steering，不是训练任务，不是 LoRA / finetuning，不是 autoregressive accuracy improvement 实验，也不允许声称 hallucination reduction 或 answer accuracy improvement。

本轮不再尝试修正模型答案，而是回答：

```text id="v7ookp"
final-answer readout 位置的 MLP output，
能否作为一个无需 gold、无需 donor 的错误诊断信号？
```

3C-2 已经说明：

```text id="xr32wr"
gold-unembedding direction 是强、低 harm、first-step 存活的 steering 轴，
但它需要知道 gold answer，因此不可部署。

mean_delta / PC1 等 gold-free donor-free 方向较弱，
不足以作为 robust steering handle。
```

因此本 sprint 将方向从：

```text id="khvlym"
donor-free steering
```

收敛为：

```text id="k9gekj"
MLP readout attribution / detection probe
```

默认标记：

```text id="x0wos5"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
```

---

## 执行前必读

执行前必须阅读：

```text id="g6dwco"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md

docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md
docs/codex_tasks/sprint_3C_2_mlp_readout_direction_analysis.md

src/recover_attention/answer_proxy_metrics.py
src/recover_attention/module_causal_tracing.py
src/recover_attention/mlp_readout_direction.py

scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py
scripts/sprint_3C_2_mlp_readout_direction_analysis.py

tests/test_answer_proxy_metrics.py
tests/test_module_causal_tracing.py
tests/test_mlp_readout_direction.py
```

---

## Preflight

输出：

```text id="gsxmvo"
outputs/logs/sprint_3C_3_mlp_readout_attribution_probe/preflight_report.md
```

必须检查：

```text id="v393w4"
1. 3C-2 tracked changes 是否已经 commit；
2. 3C-2 是否记录 mixed result；
3. 3C-2 是否明确：gold_unembed 强但 oracle，不可部署；
4. 3C-2 是否明确：mean_delta / PC1 donor-free steering 不 robust；
5. 本 sprint 是否不再做 steering；
6. 本 sprint 是否不会覆盖 3C-2 输出目录；
7. outputs 若被 gitignore，artifact manifest 是否会更新；
8. 当前仍然非 2000、非 full 3C、非训练。
```

如果 3C-2 尚未 commit，应停止并提示先 commit，除非用户明确允许继续。

---

## 前因后果

### 1. 3A / 3B：span-level steering 失败

```text id="nlua6s"
3A attention-bias：
多看 oracle span 不比 random span 更能改善答案。

3B span residual injection：
注入 oracle span representation 不比 random representation 更有效。
```

结论：

```text id="vc4rsz"
key span 更适合作为 detection / diagnosis signal，
不是直接 answer-steering handle。
```

---

### 2. 3C-0-Fix：correct activation 在答案读出位有效

修正 answer proxy 后发现：

```text id="j8mf54"
correct-run activation 在 final-answer readout position 有修复信号；
但显式 reasoning-step patch 不够 donor-specific / site-specific。
```

---

### 3. 3C-1：MLP output 是答案读出位的选择性写路径

3C-1 把 whole residual 拆成：

```text id="n1fuum"
attention_output
mlp_output
residual_output
```

结果：

```text id="d9xqdm"
mlp_output 是唯一同时 donor-specific 与 site-specific 的路径；
attention_output donor-specific 但不 site-specific；
residual_output 幅度大但 high harm、非选择性。
```

---

### 4. 3C-2：MLP 方向可解释，但 donor-free steering 不 robust

3C-2 发现：

```text id="sj1nyq"
MLP readout delta 是稳定、低秩方向；
L24 correct-wrong delta 与 gold-vs-wrong unembedding 方向稳定对齐；
gold_unembed direction 强、低 harm、first-step 存活。
```

但同时：

```text id="8z06hj"
gold_unembed 需要知道 gold answer，是 oracle upper bound；
mean_delta / PC1 donor-free direction 不够强；
因此不应继续盲 steering。
```

所以 3C-3 的新目标是：

```text id="dd4b6i"
把 MLP readout signal 做成 attribution / detection probe，
而不是 steering mechanism。
```

---

## 本轮核心问题

本 sprint 只回答：

```text id="e6g3aa"
在不使用 gold answer 作为输入的情况下，
MLP readout output 在 unembedding 空间中的投影，
能否诊断模型当前倾向的答案 token，
并区分 correct vs wrong / high-risk vs low-risk cases？
```

换句话说：

```text id="i2p4ka"
模型在答案读出位的 MLP output 正在把 residual 推向哪个答案？
这个方向是否能作为错误检测信号？
```

---

## 严格边界

禁止：

```text id="dbb186"
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要做 steering / patching / nudge；
- 不要用 gold answer 构造 feature；
- 不要把 gold_unembed 当作可部署方法；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 teacher-forced proxy 写成 generation accuracy result；
- 不要覆盖 3C-2 输出目录。
```

允许：

```text id="t5317o"
- 读取 final-answer readout position 的 MLP output；
- 计算 MLP output 对 answer-token unembedding 的投影；
- 构造 gold-free attribution features；
- 使用 gold answer 仅作为 eval label；
- 比较 correct vs wrong；
- 计算 AUC、AUROC、AUPRC、calibration；
- 输出 diagnostic probe report；
- 更新 PROGRESS / sprint_3_history / artifact manifest。
```

---

## 输入

优先读取：

```text id="a80n2i"
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/mlp_readout_delta_manifest.jsonl
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/mlp_direction_geometry_report.json
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/mlp_unembedding_alignment_report.json
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/review_gate_mlp_readout_direction_analysis.md

outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/corrected_pair_manifest.jsonl
data/raw/gsm8k_train_normalized.jsonl
```

如果 3C-2 outputs 因 gitignore 未提交但本地存在，直接读本地。

如果 3C-2 outputs 本地缺失，不要伪造；在 preflight 中记录，并从 3C-0-Fix corrected pairs 重新捕获 MLP readout outputs。

---

## 输出目录

```text id="ymnow0"
outputs/logs/sprint_3C_3_mlp_readout_attribution_probe/
```

必须输出：

```text id="s3a3uj"
preflight_report.md
mlp_attribution_config.json
mlp_readout_attribution_manifest.jsonl
mlp_unembedding_projection_report.json
answer_token_attribution_report.json
correct_wrong_detection_report.json
risk_score_report.json
baseline_comparison_report.json
calibration_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_mlp_readout_attribution_probe.md
```

同时新增或更新：

```text id="llkcdb"
docs/codex_tasks/sprint_3C_3_mlp_readout_attribution_probe.md
src/recover_attention/mlp_readout_attribution.py
scripts/sprint_3C_3_mlp_readout_attribution_probe.py
tests/test_mlp_readout_attribution.py
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

---

## Probe 定义

### 1. MLP readout attribution vector

在 final-answer readout position，读取：

```text id="jo2d8p"
mlp_output_L
```

优先层：

```text id="sclwsg"
L20
L24
```

也可加入：

```text id="jkpgl8"
L16
```

---

### 2. Unembedding projection

计算：

```text id="jnhk3k"
score_vocab = mlp_output_L @ W_U
```

其中：

```text id="jxfpmx"
W_U = model lm_head weight
```

然后提取 answer-token subset：

```text id="ux9eov"
digit tokens
number-like tokens
tokens appearing in parsed candidate answers
```

输出：

```text id="t3kx3t"
top_k_projected_tokens
top_k_projected_number_tokens
projection_margin
projection_entropy
projection_confidence
```

注意：

```text id="smm9cy"
features 不能使用 gold answer；
gold answer 只用于 eval。
```

---

### 3. Gold-free attribution features

至少构造以下 feature：

```text id="eglcvz"
mlp_top1_number_token
mlp_top1_number_score
mlp_top2_number_score
mlp_number_margin
mlp_number_entropy
mlp_projection_norm
mlp_projection_sharpness
mlp_logit_lens_agreement_with_final_answer
mlp_agreement_with_model_generated_answer
mlp_layer20_layer24_agreement
mlp_layer20_layer24_margin_gap
```

其中：

```text id="67kg36"
mlp_agreement_with_model_generated_answer
```

表示 MLP readout attribution 的 top number token 是否与模型最终生成答案一致。

这可以作为 gold-free confidence / self-consistency signal。

---

### 4. Eval-only labels

gold answer 只能用于 eval：

```text id="m841al"
is_model_answer_correct
gold_answer_token_sequence
wrong_answer_token_sequence
mlp_top_number_matches_gold
mlp_top_number_matches_wrong
```

不得把这些 label 混入 feature。

---

## 主要诊断问题

### Q1：MLP readout attribution 是否指向模型最终答案？

```text id="2t99ar"
mlp_top_number_token == model_final_answer_first_token
```

或 full answer 近似匹配。

如果一致率高，说明：

```text id="axrv66"
MLP readout projection 确实刻画了模型正在读出的答案倾向。
```

---

### Q2：MLP readout attribution 是否能区分 correct vs wrong？

计算：

```text id="fwtrc8"
AUROC / AUPRC
```

features 包括：

```text id="fg96jp"
mlp_number_margin
mlp_number_entropy
layer agreement
projection sharpness
agreement_with_generated_answer
```

目标不是训练复杂 probe，优先使用简单规则与 logistic probe 两种：

```text id="mx1jba"
rule-based score
linear logistic probe
```

如果训练 logistic probe，必须：

```text id="d7idfe"
question-grouped split
no gold-derived features
small model only
report CV metrics
```

---

### Q3：MLP attribution 是否比 baselines 更有诊断价值？

baselines：

```text id="9w5w6z"
final logits margin
residual readout projection
attention output projection
surface answer confidence
random projection
```

比较：

```text id="hf17cq"
MLP attribution features vs final logits
MLP attribution features vs residual projection
MLP attribution features vs attention output projection
MLP attribution features vs random
```

注意：

```text id="dvrm3r"
final logits 本身很强时，不必强行说 MLP 更强；
MLP 的价值可以是机制解释与早期/模块级 attribution。
```

---

## Detection score

建议定义一个 gold-free risk score：

```text id="z61hjt"
mlp_readout_risk =
high entropy
+ low top1 margin
+ layer20_layer24 disagreement
+ low agreement with final generated answer
```

也可以定义 confidence score：

```text id="x0m5s0"
mlp_readout_confidence =
top1 margin
- entropy
+ layer agreement
```

评估：

```text id="rfio57"
correct vs wrong AUROC
bucketed accuracy
calibration curve
high-risk subset wrong rate
low-risk subset correct rate
```

---

## Primary success criterion

本 sprint 的 primary success 不是 steering。

Primary success 是：

```text id="v52v3h"
MLP readout attribution probe 提供可解释、gold-free 的 answer-risk signal。
```

通过条件：

```text id="o56gyq"
1. MLP projection top tokens 与模型 final answer 有明显一致性；
2. MLP attribution features 对 correct/wrong 有高于 random 的诊断能力；
3. risk score 在 high-risk bucket 中显著富集 wrong cases；
4. probe 不使用 gold-derived features；
5. 结论不声称修复，只声称诊断/归因。
```

如果 MLP probe 不强，也可以完成 sprint，只要诚实记录：

```text id="xhb72s"
MLP readout causal handle is mechanistically real,
but gold-free attribution signal is insufficient for practical detection.
```

---

## Reports

### `mlp_unembedding_projection_report.json`

必须包含：

```text id="qa6gma"
layer-wise top-k projection stats
number-token subset stats
projection entropy
projection margin
layer agreement
```

### `answer_token_attribution_report.json`

必须包含：

```text id="vnegcd"
mlp_top_number_matches_model_answer_rate
mlp_top_number_matches_gold_rate_eval_only
mlp_top_number_matches_wrong_rate_eval_only
top-k coverage
```

### `correct_wrong_detection_report.json`

必须包含：

```text id="ax4zoj"
rule_score_AUROC
rule_score_AUPRC
logistic_probe_AUROC
logistic_probe_AUPRC
question_grouped_CV
feature_ablation
```

### `baseline_comparison_report.json`

必须包含：

```text id="vrp8vm"
MLP attribution vs final logits
MLP attribution vs residual projection
MLP attribution vs attention output projection
MLP attribution vs random
```

### `calibration_report.json`

必须包含：

```text id="yeaap7"
risk buckets
wrong rate by bucket
confidence buckets
correct rate by bucket
ECE or simple calibration error
```

---

## Review gate

`review_gate_mlp_readout_attribution_probe.md` 必须回答：

```text id="qnxk4e"
1. 本轮是否停止 steering、转为 attribution/detection？
2. 是否复用了 3C-2 的机制发现？
3. 是否没有使用 gold answer 构造 feature？
4. MLP projection 是否指向模型最终答案？
5. MLP projection 是否与 gold/wrong 有 eval-only 对应关系？
6. MLP attribution 是否能区分 correct vs wrong？
7. 是否优于 random baseline？
8. 是否优于或补充 final logits baseline？
9. 是否提供 high-risk / low-risk bucket？
10. 是否有 calibration report？
11. 是否保留机制解释：MLP readout 在往哪个 token 写？
12. 是否允许进入 2000？
13. 是否允许声称 hallucination reduction？
14. 是否允许声称 answer accuracy improvement？
15. 下一步应该是扩大 detection evaluation、写论文叙事，还是停止 steering 线？
```

默认最终标记：

```text id="xzm6b0"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
```

---

## 建议新增模块

```text id="jk5zcy"
src/recover_attention/mlp_readout_attribution.py
```

至少实现：

```text id="atw6w6"
capture_mlp_readout_output()
project_to_unembedding()
filter_number_like_tokens()
extract_projection_features()
compute_mlp_readout_risk()
evaluate_correct_wrong_detection()
question_grouped_cv()
calibration_buckets()
```

---

## 建议新增脚本

```text id="vwk8b0"
scripts/sprint_3C_3_mlp_readout_attribution_probe.py
```

推荐命令：

```bash id="fw5gr7"
conda run -n recover_attention python -m pytest tests/test_mlp_readout_attribution.py -q

conda run -n recover_attention python scripts/sprint_3C_3_mlp_readout_attribution_probe.py \
  --input-dir outputs/logs/sprint_3C_2_mlp_readout_direction_analysis \
  --fix-input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_3_mlp_readout_attribution_probe \
  --layers 20 24 \
  --top-k 20 \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 建议新增测试

```text id="hg2a0f"
tests/test_mlp_readout_attribution.py
```

至少覆盖：

```text id="fs9yen"
1. number-like token filter 不返回纯空格；
2. projection features 不使用 gold；
3. risk score 可复现；
4. entropy / margin 计算正确；
5. layer agreement 计算正确；
6. question-grouped split 不泄漏 question；
7. calibration buckets 正确；
8. random baseline 可复现；
9. eval-only gold labels 不进入 feature list。
```

---

## 最后更新

完成后必须更新：

```text id="hpzwkd"
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

如果 outputs 被 gitignore，不提交大 jsonl，但 artifact manifest 必须记录：

```text id="qyxhb8"
outputs/logs/sprint_3C_3_mlp_readout_attribution_probe/
```

至少记录：

```text id="unn59x"
mlp_unembedding_projection_report.json
answer_token_attribution_report.json
correct_wrong_detection_report.json
risk_score_report.json
baseline_comparison_report.json
calibration_report.json
review_gate_mlp_readout_attribution_probe.md
```

最终不得写：

```text id="sxeu97"
accuracy improved
hallucination reduced
ready for 2000
steering solved
```

最多只能写：

```text id="y1f8mj"
MLP readout attribution provides / does not provide a gold-free diagnostic signal
for answer-risk detection under the current teacher-forced evaluation setting.
```
