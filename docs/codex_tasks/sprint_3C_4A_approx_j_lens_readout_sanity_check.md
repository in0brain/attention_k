# Sprint 3C-4A: Approximate J-lens Readout Sanity Check

## 定位

本 sprint 是 3C-3R 之后的 **readout-method sanity check sprint**。

它不是 steering，不是 patching，不是 nudge，不是训练，不是 full Sprint 3C，不是 2000-scale rerun，也不允许声称 hallucination reduction 或 answer accuracy improvement。

本轮只回答一个问题：

```text
3C-3 使用 MLP_output @ W_U 作为 MLP readout attribution 的 primary readout。
这个 direct unembedding / logit-lens 近似是否可信？
```

更具体地说，本轮要比较：

```text
direct logit-lens readout:
  MLP_output @ W_U

approximate J-lens readout:
  通过小扰动 / finite-difference 估计 MLP_output 对最终 logits 的实际影响
```

目标不是证明 MLP attribution 更强，而是验证：

```text
3C-3 的 MLP readout attribution 是否被 direct unembedding 近似严重误读或低估。
```

默认标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
```

---

## 执行前必读

执行前必须阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/STORY.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md

docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md
docs/codex_tasks/sprint_3C_2_mlp_readout_direction_analysis.md
docs/codex_tasks/sprint_3C_3_mlp_readout_attribution_probe.md

src/recover_attention/module_causal_tracing.py
src/recover_attention/mlp_readout_direction.py
src/recover_attention/mlp_readout_attribution.py
src/recover_attention/answer_proxy_metrics.py

scripts/sprint_3C_2_mlp_readout_direction_analysis.py
scripts/sprint_3C_3_mlp_readout_attribution_probe.py

tests/test_module_causal_tracing.py
tests/test_mlp_readout_direction.py
tests/test_mlp_readout_attribution.py
```

---

## Preflight

先输出：

```text
outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/preflight_report.md
```

必须检查：

```text
1. 当前 PROGRESS.md 是否已经记录 3C-3R；
2. docs/reference/STORY.md 是否存在；
3. STORY 是否已经记录：
   - 3C-3 已完成；
   - MLP risk beats random；
   - MLP risk does not beat final-logits margin baseline；
   - NLA L20 不能反证 L24 MLP readout；
   - J-lens 是 3C-4 优先 sanity check，但不是主路径依赖。
4. 3C-3 outputs 是否本地存在；
5. 3C-0-Fix corrected pairs 是否本地存在；
6. 本轮是否不会覆盖 3C-3 输出目录；
7. 本轮是否不做 steering / patching / nudge；
8. 本轮是否不进入 2000，不做 full 3C，不做训练；
9. outputs 若被 gitignored，artifact manifest 是否会更新。
```

如果 3C-3 outputs 本地不存在，不要伪造。
应在 preflight 中记录缺失，并从 3C-0-Fix corrected pairs 重新捕获本轮所需 MLP readout outputs。

---

## 背景

### 1. 3C-1 找到 MLP readout causal path

3C-1 已经证明：

```text
final-answer readout position 的 MLP output
是唯一同时 donor-specific 与 site-specific 的 causal write path。
```

这说明 MLP output 是 answer commitment 的关键写入点。

---

### 2. 3C-2 说明有效方向对齐 gold unembedding

3C-2 发现：

```text
L24 correct-wrong MLP delta
与 gold-vs-wrong unembedding direction 稳定对齐。
```

但：

```text
gold_unembed 需要知道 gold answer，是 oracle upper bound；
mean_delta / PC1 donor-free steering 不 robust。
```

因此 steering 不能继续 blind 做。

---

### 3. 3C-3 转向 attribution，但 readout 近似有限

3C-3 使用：

```text
MLP_output @ W_U
```

作为 primary readout，构造 MLP readout attribution / risk score。

结果：

```text
MLP risk beats random；
but does not beat final-logits margin baseline。
```

因此 3C-3 的价值是机制诊断，不是实用 detector 优胜。

但这里有一个未验证的问题：

```text
MLP_output @ W_U 是否是可靠读出？
```

因为 L20/L24 后面还有后续 transformer blocks、final RMSNorm 和 lm_head。
direct unembedding projection 可能低估、错读或扭曲 MLP output 的实际最终 logit 影响。

所以需要 3C-4A。

---

## 本轮核心问题

本轮只回答：

```text
direct logit-lens readout 与 approximate J-lens readout
在 answer-readout MLP attribution 上是否一致？
```

重点比较：

```text
1. top-k number token overlap；
2. rank correlation；
3. gold / model-answer / wrong-answer token 的 score ordering；
4. margin / entropy 是否一致；
5. correct vs wrong diagnostic AUROC 是否一致；
6. direct logit-lens 是否系统性低估 L24 MLP signal。
```

---

## 严格边界

禁止：

```text
- 不要做 steering；
- 不要做 patching claim；
- 不要做 nudge；
- 不要训练 probe；
- 不要扩大到 2000；
- 不要 full Sprint 3C；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 approximate J-lens 写成完整复现 Workspace J-lens；
- 不要把 J-lens 分歧写成推翻 3C-1/3C-2 causal tracing。
```

允许：

```text
- 复用 3C-3 的 MLP readout attribution pipeline；
- 复用 3C-0-Fix corrected pairs；
- 小样本 finite-difference / approximate J-lens；
- 比较 direct readout 与 approximate J-lens readout；
- 输出 sanity report；
- 更新 PROGRESS / sprint_3_history / artifact manifest。
```

---

## 样本规模

本轮是 small sanity check，不做大扩采样。

建议：

```text
primary_n_pairs = 34
```

也就是复用 3C-0-Fix / 3C-3 的 corrected pairs。

如果成本过高，可以先跑：

```text
primary_n_pairs = 16
```

但最终报告必须明确样本数。

---

## Approximate J-lens 方法

不要尝试构造完整 Jacobian 矩阵。

本轮只做低成本 directional finite-difference J-lens。

### 方法 A：directional effect of MLP output

对每个 trace、layer、answer-readout position，捕获：

```text
m = MLP_output[layer][answer_readout_pos]
```

构造单位方向：

```text
u = unit(m)
```

在同一 forward 的同一 module output 处施加小扰动：

```text
MLP_output_patched = MLP_output_original + epsilon * ||MLP_output_original|| * u
```

然后读取 final answer position logits 的变化：

```text
delta_logits = logits_patched - logits_base
```

这相当于估计：

```text
J_final_logits(m) · m
```

也就是该 MLP output direction 对最终 logits 的实际局部影响。

注意：

```text
这不是完整 J-lens；
这是 directional approximate J-lens。
```

---

### 方法 B：candidate-answer directional check

对候选 number-like tokens，比较：

```text
direct_score(token) = m @ W_U[token]
approx_j_score(token) = delta_logits[token] / epsilon
```

只在 number-like token subset 上比较，不需要全词表。

候选集合应至少包括：

```text
1. direct logit-lens top-k number tokens；
2. final logits top-k number tokens；
3. model parsed answer token；
4. eval-only gold answer token；
5. eval-only wrong answer token；
6. random number tokens。
```

gold/wrong 只能用于 eval-only analysis，不能作为 feature 构造。

---

### 方法 C：optional random direction control

为了确认 finite-difference pipeline 没有总是产生相似读数，加入：

```text
random_unit_direction
negative_mlp_direction = -unit(m)
zero_direction
```

比较它们对 final logits 的影响。

---

## Layers

必须覆盖：

```text
L20
L24
```

可选：

```text
L16
```

重点是 L24，因为 3C-2 中最清晰的 unembedding alignment 出现在 L24。

---

## Epsilon sweep

建议：

```text
epsilon ∈ {0.01, 0.03, 0.1}
```

要求：

```text
1. delta_logits 不应全为 0；
2. top-k readout 不应被过大 epsilon 扰动成非局部效果；
3. epsilon stability 要报告。
```

如果成本有限，至少跑：

```text
epsilon = 0.03
```

---

## 输出目录

```text
outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/
```

必须输出：

```text
preflight_report.md
approx_j_lens_config.json
approx_j_lens_forward_manifest.jsonl
direct_vs_approx_readout_manifest.jsonl
topk_overlap_report.json
rank_correlation_report.json
token_ordering_report.json
diagnostic_comparison_report.json
epsilon_stability_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_approx_j_lens_readout_sanity_check.md
```

同时新增或更新：

```text
docs/codex_tasks/sprint_3C_4A_approx_j_lens_readout_sanity_check.md
src/recover_attention/approx_j_lens_readout.py
scripts/sprint_3C_4A_approx_j_lens_readout_sanity_check.py
tests/test_approx_j_lens_readout.py
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

---

## 指标

### 1. Top-k overlap

比较：

```text
top_k_direct_number_tokens
top_k_approx_j_number_tokens
```

输出：

```text
top1_match_rate
top3_overlap
top5_overlap
top10_overlap
```

按 layer 和 epsilon 分开统计。

---

### 2. Rank correlation

在 number-like token subset 上计算：

```text
Spearman correlation
Kendall tau, optional
```

比较：

```text
direct_score(token)
approx_j_score(token)
```

---

### 3. Token ordering

eval-only 检查：

```text
score(gold_answer_token)
score(model_answer_token)
score(wrong_answer_token)
```

比较 direct 与 approximate J-lens 是否给出相同 ordering。

注意：

```text
gold answer 只能 eval-only；
不能构造 gold-derived feature。
```

---

### 4. Diagnostic comparison

用 direct readout 和 approximate J-lens readout 分别构造同类 risk features：

```text
number_margin
number_entropy
projection_sharpness
layer_agreement
agreement_with_model_answer
```

比较：

```text
direct_readout_AUROC
approx_j_readout_AUROC
final_logits_margin_AUROC
random_AUROC
```

本轮不要求 approximate J-lens 跑赢 final logits。
本轮只要求判断：

```text
approximate J-lens 是否显著改变 3C-3 对 MLP attribution 强弱的判断。
```

---

## Primary success criterion

本 sprint 的 primary success 不是 detection improvement。

Primary success 是：

```text
明确判断 direct logit-lens readout 是否足够可靠。
```

通过条件之一即可：

### Case A：direct 与 approximate J-lens 大体一致

如果：

```text
top-k overlap 较高；
rank correlation 稳定正；
token ordering 基本一致；
diagnostic AUROC 差距不大；
```

结论：

```text
3C-3 的 direct unembedding readout 是可接受的低成本近似；
可以进入 expanded detection evaluation。
```

下一步：

```text
Sprint 3C-4B: Expanded MLP Readout Detection Evaluation
```

---

### Case B：approximate J-lens 明显优于 direct readout

如果：

```text
approximate J-lens 与 final logits / model answer / gold eval label 更一致；
diagnostic AUROC 明显高于 direct readout；
direct top-k 明显错读；
```

结论：

```text
3C-3 可能被 direct logit-lens 近似低估；
后续 detection evaluation 应使用 approximate J-lens readout 或修正读出。
```

下一步：

```text
Sprint 3C-4B should use approximate J-lens features, not direct-only features.
```

---

### Case C：direct 与 approximate J-lens 都弱

如果：

```text
二者都不稳定；
二者都不优于 final logits；
二者 top-k 与模型答案关系弱；
```

结论：

```text
MLP readout attribution 的实用 detection 价值有限；
机制因果发现仍成立，但 attribution probe 不宜继续扩张为 detector。
```

下一步：

```text
mechanism write-up / stop detection expansion
```

---

## Review gate

`review_gate_approx_j_lens_readout_sanity_check.md` 必须回答：

```text
1. 本轮是否只是 readout sanity check？
2. 是否没有做 steering / patching / nudge claim？
3. 是否复用了 3C-3 / 3C-0-Fix 输入？
4. 使用了多少 pairs？
5. 覆盖哪些 layers？
6. 覆盖哪些 epsilon？
7. direct logit-lens 与 approximate J-lens 的 top-k overlap 如何？
8. rank correlation 如何？
9. gold/model/wrong token ordering 是否一致？
10. approximate J-lens 是否改变 3C-3 的结论？
11. approximate J-lens 是否明显优于 direct readout？
12. direct readout 是否足够作为低成本近似？
13. 是否仍然没有跑赢 final logits baseline？
14. 是否可以进入 expanded detection evaluation？
15. 是否允许进入 2000？
16. 是否允许声称 hallucination reduction？
17. 是否允许声称 answer accuracy improvement？
18. 下一步是 3C-4B expanded detection、机制 write-up，还是停止 detection expansion？
```

默认最终标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
```

---

## 建议新增模块

```text
src/recover_attention/approx_j_lens_readout.py
```

至少实现：

```text
capture_mlp_output_at_readout()
compute_direct_unembedding_scores()
register_directional_mlp_output_perturbation()
compute_directional_logit_delta()
filter_number_like_tokens()
topk_overlap()
rank_correlation()
compare_token_ordering()
compute_approx_j_risk_features()
```

---

## 建议新增脚本

```text
scripts/sprint_3C_4A_approx_j_lens_readout_sanity_check.py
```

推荐命令：

```bash
conda run -n recover_attention python -m pytest tests/test_approx_j_lens_readout.py -q

conda run -n recover_attention python scripts/sprint_3C_4A_approx_j_lens_readout_sanity_check.py \
  --input-dir outputs/logs/sprint_3C_3_mlp_readout_attribution_probe \
  --fix-input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check \
  --layers 20 24 \
  --epsilons 0.01 0.03 0.1 \
  --top-k 20 \
  --max-pairs 34 \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 建议新增测试

```text
tests/test_approx_j_lens_readout.py
```

至少覆盖：

```text
1. unit direction normalization 不产生 NaN；
2. zero vector direction 被安全处理；
3. epsilon <= 0 报错；
4. perturbation hook 只修改目标 layer / 目标 position；
5. remove hook 后 baseline 恢复；
6. direct score shape 正确；
7. delta logits shape 正确；
8. number-like token filter 不返回纯空格；
9. top-k overlap 计算正确；
10. rank correlation 在常见输入下正确；
11. eval-only gold labels 不进入 feature list；
12. preflight 不覆盖 3C-3 输出目录。
```

---

## 最后更新

完成后必须更新：

```text
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

如果 outputs 被 gitignored，不提交大 jsonl，但 artifact manifest 必须记录：

```text
outputs/logs/sprint_3C_4A_approx_j_lens_readout_sanity_check/
```

至少记录：

```text
topk_overlap_report.json
rank_correlation_report.json
token_ordering_report.json
diagnostic_comparison_report.json
epsilon_stability_report.json
review_gate_approx_j_lens_readout_sanity_check.md
```

最终不得写：

```text
accuracy improved
hallucination reduced
ready for 2000
steering solved
J-lens fully reproduced
```

最多只能写：

```text
Approximate J-lens sanity check supports / weakens / changes the direct unembedding readout used by Sprint 3C-3.
```
