# Sprint 3C-2: MLP Readout Direction Analysis and Donor-Free Nudge

## 定位

本 sprint 是 3C-1 之后的 **MLP readout causal handle analysis sprint**。

它不是 full Sprint 3C，不是 2000-scale rerun，不是训练任务，不是 LoRA / finetuning，不是正式 autoregressive accuracy evaluation，也不允许声称 hallucination reduction 或 answer accuracy improvement。

3C-1 已经发现：

```text
final-answer readout position 的 MLP output 是唯一同时通过 donor-specificity 与 site-specificity 的 causal path。
```

本 sprint 不再重复证明 MLP patch 是否有效，而是回答：

```text
3C-1 中有效的 MLP readout signal 是否可以被解释为稳定方向？
这个方向是否可以脱离 correct donor，以 donor-free nudge 的形式保留效果？
```

默认标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 执行前必读

执行前必须先阅读：

```text
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md

docs/codex_tasks/sprint_3C_0_correct_wrong_activation_patching_reasoning_steps.md
docs/codex_tasks/sprint_3C_0_fix_answer_proxy_recheck.md
docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md

src/recover_attention/activation_patching.py
src/recover_attention/answer_proxy_metrics.py
src/recover_attention/module_causal_tracing.py

scripts/sprint_3C_0_fix_answer_proxy_recheck.py
scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py

tests/test_activation_patching.py
tests/test_answer_proxy_metrics.py
tests/test_module_causal_tracing.py
```

---

## Preflight

先输出：

```text
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/preflight_report.md
```

必须检查：

```text
1. 3C-1 是否完成；
2. 3C-1 是否记录 Case A：MLP output 是唯一同时 donor-specific 与 site-specific 的路径；
3. 3C-1 的 low-harm cell 是否包含 mlp_output|L24|alpha=0.25 与 L20 alpha=0.25；
4. 3C-1 outputs 是否存在；若 outputs 被 gitignore，artifact manifest 是否记录；
5. 当前 commit 是否已经包含 3C-0 / 3C-0-Fix / 3C-1 的 tracked 改动；
6. 本轮是否不会覆盖 3C-1 输出目录；
7. 本轮是否仍然非 2000、非 full 3C、非训练。
```

如果尚未 commit 3C 系列 tracked changes，应先提示并停止，除非用户明确允许继续。

---

## 前因后果

### 1. 3A / 3B 排除了 span-level steering

```text
3A attention-bias：
多看 oracle span 不比 random span 更有效。

3B span residual injection：
注入 oracle span representation 不比 random representation 更有效。
```

结论：

```text
key span 更适合作为 detection / diagnosis signal，
不是直接可控的 steering handle。
```

---

### 2. 3C-0-Fix 找到 answer-readout 阶段的 correct-context signal

修正 answer proxy 后发现：

```text
correct-run activation 在 final-answer readout position 有修复信号；
显式 reasoning-step patch 不够 donor-specific / site-specific。
```

---

### 3. 3C-1 定位到 MLP output

3C-1 将 whole residual patch 拆成：

```text
attention_output
MLP_output
residual_output
```

结果：

```text
attention_output：donor-specific，但不 site-specific；
residual_output：幅度大，但 high harm、非选择性；
MLP_output：同时 donor-specific 与 site-specific，且存在低 harm regime。
```

因此，3C-2 的重点是：

```text
解释 MLP readout signal，并测试能否从 correct donor patch 转为 donor-free direction。
```

---

## 本轮核心目标

目标 1：收集 MLP readout 差方向

对 same-question correct / wrong pair，在 answer readout position 提取：

```text
delta_mlp_output =
MLP_output_correct_readout
-
MLP_output_wrong_readout
```

优先 cells：

```text
layer = 24, alpha = 0.25
layer = 20, alpha = 0.25
```

可选对照：

```text
layer = 16
alpha = 0.5
attention_output
residual_output
```

---

目标 2：判断方向是否低秩 / 稳定

分析：

```text
1. delta 向量 pairwise cosine；
2. PCA / SVD explained variance；
3. top principal direction 是否稳定；
4. correct-wrong delta 是否显著区别于 random donor delta；
5. layer 20 与 layer 24 的方向是否相似。
```

输出：

```text
mlp_direction_geometry_report.json
```

---

目标 3：判断方向是否对齐答案 unembedding

对每个 pair 计算：

```text
gold_direction = unembedding[gold_answer_tokens] - unembedding[wrong_answer_tokens]
```

或对首个有效数字 token：

```text
gold_vs_wrong_unembed_direction =
W_U[gold_first_digit_or_number_token]
-
W_U[wrong_first_digit_or_number_token]
```

分析：

```text
cos(delta_mlp_output, gold_vs_wrong_unembed_direction)
projection(delta_mlp_output, gold_vs_wrong_unembed_direction)
delta 对 gold token logit 的直接贡献估计
delta 对 wrong token logit 的直接贡献估计
```

输出：

```text
mlp_unembedding_alignment_report.json
```

注意：full sequence answer 仍以 3C-0-Fix 的 sequence-logprob proxy 为主，unembedding alignment 只是解释指标，不替代主指标。

---

目标 4：donor-free direction nudge

从训练/分析 subset 中构造 donor-free directions：

```text
mean_delta_direction
pc1_delta_direction
gold_minus_wrong_unembed_direction
random_direction_control
shuffled_delta_control
negative_delta_control
```

在 wrong run 的 answer readout position，对 MLP output 施加：

```text
MLP_output_patched =
MLP_output_original + alpha * direction
```

或者规范化版本：

```text
MLP_output_patched =
MLP_output_original + alpha * ||MLP_output_original|| * unit(direction)
```

alpha sweep：

```text
alpha ∈ {0.05, 0.1, 0.2, 0.4}
```

优先低 harm。

输出：

```text
donor_free_nudge_report.json
```

---

目标 5：最小 generation-survival 检查

这不是 accuracy evaluation。

只做小规模 first-step generation proxy：

```text
在 answer readout position 施加 donor-free MLP nudge，
检查下一步 gold token / gold sequence logprob 是否仍提升，
harm 是否可控。
```

禁止声称：

```text
accuracy improved
hallucination reduced
generation success
```

最多写：

```text
donor-free MLP direction survives / does not survive a first-step generation-style check.
```

---

## 严格边界

禁止：

```text
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要回到 attention-bias steering；
- 不要回到 span residual injection；
- 不要把 correct donor 当成可部署方法；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 teacher-forced proxy 写成 autoregressive correctness result；
- 不要覆盖 3C-1 输出目录。
```

允许：

```text
- 复用 3C-1 corrected pair manifest；
- 复用 answer-sequence logprob proxy；
- 捕获 MLP output at answer readout position；
- 做 MLP delta direction analysis；
- 做 PCA / SVD / cosine / unembedding alignment；
- 做 donor-free direction nudge；
- 做小规模 first-step generation-style survival check；
- 输出 review gate。
```

---

## 输入

优先读取：

```text
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/module_patch_pair_manifest.jsonl
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/module_patching_forward_manifest.jsonl
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/module_control_comparison_report.json
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/donor_specificity_report.json
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/site_specificity_report.json
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/review_gate_final_answer_compression_tracing.md

outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/corrected_pair_manifest.jsonl
data/raw/gsm8k_train_normalized.jsonl
```

可复用代码：

```text
src/recover_attention/answer_proxy_metrics.py
src/recover_attention/module_causal_tracing.py
```

---

## 输出目录

```text
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/
```

必须输出：

```text
preflight_report.md
mlp_direction_config.json
mlp_readout_delta_manifest.jsonl
mlp_direction_geometry_report.json
mlp_unembedding_alignment_report.json
donor_free_direction_manifest.jsonl
donor_free_nudge_forward_manifest.jsonl
donor_free_nudge_report.json
generation_survival_report.json
harm_control_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_mlp_readout_direction_analysis.md
```

同时新增或更新：

```text
docs/codex_tasks/sprint_3C_2_mlp_readout_direction_analysis.md
src/recover_attention/mlp_readout_direction.py
scripts/sprint_3C_2_mlp_readout_direction_analysis.py
tests/test_mlp_readout_direction.py
docs/progress/sprint_3_history.md
PROGRESS.md
docs/progress/sprint_3_artifact_manifest.md
```

---

## Primary success criterion

本 sprint 的 primary success 不是 accuracy improvement。

Primary success 是同时满足：

```text
1. MLP correct-wrong delta 呈现稳定方向结构；
2. mean_delta 或 PC1 direction 与 gold-vs-wrong unembedding 有正向对齐；
3. donor-free direction nudge 在 corrected sequence-logprob proxy 下 clean_direction stable positive；
4. donor-free direction nudge 优于 random / shuffled / negative direction controls；
5. harm 可控，优先 alpha <= 0.2；
6. 低 harm cell 在 L20 或 L24 上复现。
```

如果只有 correct donor patch 有效，而 donor-free direction 无效，则结论是：

```text
MLP readout contains causal information,
but current direction extraction is insufficient for donor-free steering.
```

---

## Controls

必须包含：

```text
correct_donor_delta_direction
mean_delta_direction
pc1_delta_direction
gold_minus_wrong_unembed_direction
random_direction
shuffled_delta_direction
negative_delta_direction
zero_direction
```

核心比较：

```text
mean_delta_direction - random_direction
pc1_delta_direction - random_direction
mean_delta_direction - shuffled_delta_direction
mean_delta_direction - negative_delta_direction
gold_unembed_direction - random_direction
```

全部用 paired bootstrap。

---

## Harm control

harm 定义：

```text
harm =
top1_changed_to_non_gold
or entropy_delta > 1.0
or margin_delta < -0.25
or gold_seq_logprob_delta < -0.5
or clean_direction_score < -0.5
```

必须按：

```text
direction_type
layer
alpha
```

输出 harm curve。

---

## Review gate

`review_gate_mlp_readout_direction_analysis.md` 必须逐条回答：

```text
1. 本轮是否只是 MLP readout direction analysis，而非 full 3C？
2. 是否复用了 3C-1 的 MLP readout causal site？
3. 是否仍使用 corrected answer-sequence proxy？
4. 使用了多少 correct/wrong pairs？
5. 分析了哪些 layers？
6. MLP correct-wrong delta 是否低秩？
7. mean delta / PC1 是否稳定？
8. delta 是否与 gold-vs-wrong unembedding 对齐？
9. donor-free mean direction 是否有效？
10. donor-free PC1 direction 是否有效？
11. gold-vs-wrong unembedding direction 是否有效？
12. donor-free directions 是否优于 random / shuffled / negative controls？
13. 是否存在 low-harm alpha regime？
14. generation-survival check 是否保留信号？
15. 是否仍然只是 teacher-forced / first-step proxy？
16. 是否允许进入 2000？
17. 是否允许声称 hallucination reduction / answer accuracy improvement？
18. 下一步应该做 low-harm generation eval、MLP direction probe、还是停止 steering？
```

默认最终标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 建议新增模块

```text
src/recover_attention/mlp_readout_direction.py
```

至少实现：

```text
extract_mlp_readout_output()
compute_correct_wrong_delta()
normalize_direction()
pca_direction()
mean_direction()
unembedding_answer_direction()
apply_mlp_direction_nudge()
compute_direction_geometry()
compute_unembedding_alignment()
bootstrap_direction_effect()
```

---

## 建议新增脚本

```text
scripts/sprint_3C_2_mlp_readout_direction_analysis.py
```

推荐命令：

```bash
conda run -n recover_attention python -m pytest tests/test_mlp_readout_direction.py -q

conda run -n recover_attention python scripts/sprint_3C_2_mlp_readout_direction_analysis.py \
  --input-dir outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing \
  --fix-input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_2_mlp_readout_direction_analysis \
  --layers 20 24 \
  --alphas 0.05 0.1 0.2 0.4 \
  --directions mean_delta pc1_delta gold_unembed random shuffled negative \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 建议新增测试

```text
tests/test_mlp_readout_direction.py
```

至少覆盖：

```text
1. direction normalization 不产生 NaN；
2. mean_delta_direction 维度正确；
3. pc1_direction 符号可按 mean_delta 对齐；
4. negative_direction = -direction；
5. shuffled_delta_control 不使用同 pair label；
6. random_direction 可复现；
7. unembedding_answer_direction 维度正确；
8. alpha < 0 不允许；
9. zero_direction 接近 no-op；
10. clean_direction = gold_seq_delta - wrong_seq_delta；
11. paired bootstrap 输出 mean / CI；
12. preflight 不覆盖 3C-1 输出。
```

---

## 最后更新

完成后必须更新：

```text
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

如果 outputs 被 gitignore，不提交大 jsonl，但 artifact manifest 必须记录：

```text
outputs/logs/sprint_3C_2_mlp_readout_direction_analysis/
```

至少记录：

```text
mlp_direction_geometry_report.json
mlp_unembedding_alignment_report.json
donor_free_nudge_report.json
generation_survival_report.json
harm_control_report.json
review_gate_mlp_readout_direction_analysis.md
```

最终不得写：

```text
accuracy improved
hallucination reduced
ready for 2000
```

最多只能写：

```text
MLP readout direction analysis found / did not find a donor-free low-harm direction
that preserves the 3C-1 teacher-forced answer-readout causal signal.
```
