# Sprint 3B-0: Representation-Level Oracle Intervention Diagnostic

## 定位

本 sprint 是 **3A attention-bias steering 失败后的机制转向诊断**。

它不是 full Sprint 3B，不是 2000-scale rerun，不是训练任务，不是 hallucination reduction 实验，也不允许声称 answer accuracy improvement。

本轮只回答一个问题：

```text
如果 attention-logit bias 无法把“选对 span”转化为“答案更对”，
那么关键 span 的内部表示本身是否仍然具有可干预的因果价值？
```

也就是说，本 sprint 从：

```text
attention-level steering
让模型“多看”某些 token
```

转向：

```text
representation / value-level intervention
增强或注入关键 span 在 hidden / residual / value 通道中的表示贡献
```

默认结论标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3B=True
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
docs/codex_tasks/sprint_3A_0_attention_bias_steering_smoke_test.md
docs/codex_tasks/sprint_3A_1_controlled_attention_guidance_500.md
docs/progress/sprint_3_history.md
src/recover_attention/attention_bias_steering.py
scripts/sprint_3A_1_controlled_attention_guidance.py
```

如果 `PROGRESS.md` 顶部 current status 仍停留在 3A-0，应先在本 sprint 的 preflight 中记录这一点，并在最终更新时修正为当前 3B-0/3A-1 后状态。

---

## 前因后果：为什么要从 attention-bias 转向 representation-level

### 1. Sprint 2K-V / 2K-W 说明 selector 仍有一定信号

前面阶段显示：

```text
attention-only 对 same-question on/off-path keyness 有可用信号；
attention × response-position output-effect 比 attention-only 略强；
但 selector 仍远低于 oracle。
```

这说明 attention 作为 **keyness discovery signal** 并没有完全失败。

也就是说，模型的内部 attention / output-effect 中确实包含一些“哪个 span 重要”的弱信号。

---

### 2. Sprint 3A-0 只是证明 hook 可控，不证明 guidance 有效

3A-0 实现了 increase-only additive attention logit bias：

```text
attn_score(q, k) += λ
```

它证明：

```text
hook 可以注册、触发、移除；
target attention mass 可以被提高；
小规模 harm proxy 可控。
```

但 3A-0 的 oracle sanity inconclusive，原因有两个：

```text
1. 指标用错：
   3A-0 比的是 target_attention_mass_delta，
   即“selector 自己选的 span 上 attention mass 有没有涨”。
   这个指标对 random / surface / oracle 都天然容易为正，
   不能说明 oracle span 是否更能推动正确答案。

2. 干预太弱：
   λ=0.2 下 answer-position 输出几乎不动，
   output JS 约为 1e-5，接近 no-op。
```

所以 3A-0 不能判断 attention steering 是否真正有效。

---

### 3. Sprint 3A-1 修正了诊断，但得到决定性负结果

3A-1 做了两个关键修正：

```text
1. 把 oracle 诊断改成 answer-directed eval-only metric：
   gold_first_token_logprob_delta

2. 增加 λ sweep：
   λ ∈ {0.2, 1.0, 4.0}
   先找到 output shift 可测的 regime，再比较 selector。
```

3A-1 结果显示：

```text
λ=4.0 时 output JS 可测；
所有 selector 都能提高 gold first-token logprob；
但 oracle 并不优于 random。
```

典型结果：

```text
random      +0.665
surface     +0.829
oracle      +0.593
attention   +0.569

oracle - random = -0.053
CI95 includes 0
```

同时，高 λ 下 harm proxy 明显升高，约 20%–27%。

因此当前最合理解释是：

```text
把 attention 加到“正确 span”并不比加到“随机 span”更能推动正确答案；
高 λ 的 gold-token logprob 上升更像是非选择性 output sharpening / destabilization；
attention-bias 机制本身无法把“选对 span”转化为“答对”。
```

所以，继续调 attention-bias、扩大到 2000、或者单纯提升 selector，都不是当前最优路线。

---

## 本轮核心假设

3A-1 否定的是：

```text
直接提高 selected span 的 attention logit 可以产生选择性 answer improvement
```

但它没有否定：

```text
selected span 的 hidden / residual / value representation 中存在可利用信息
```

因此本 sprint 需要验证新的机制假设：

```text
关键 span 的 representation 可能是有用的；
失败的是 attention-logit bias 这种传递机制。
```

换句话说：

```text
不是“模型不知道哪个 span 重要”，
而可能是“只让模型多看那个 span 并不能改变后续计算”。
```

---

## 本轮目标

实现一个最小 representation-level oracle diagnostic，回答：

```text
oracle on-path span 的 hidden/residual/value-level intervention
是否比 random span 更能提高 gold first-token logprob？
```

本轮只做小规模诊断，建议：

```text
primary_n = 100 或 120
dataset = 复用 3A-1 eligible question subset 逻辑
model = 本地 Qwen2.5-7B-Instruct
generation = 默认不做 autoregressive generation
metric = answer-position single-forward proxy
```

核心比较：

```text
no_intervention
random_span_rep_intervention
surface_span_rep_intervention
attention_only_span_rep_intervention
attention_x_resp_pos_span_rep_intervention
oracle_on_path_span_rep_intervention
```

其中：

```text
oracle 只允许作为 eval-only / diagnostic selector；
gold answer 只允许用于 eval-only metric；
不得把 oracle/gold 信息混入 non-oracle method。
```

---

## 严格边界

禁止：

```text
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3B；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要 hard mask；
- 不要 attention decrease / suppress；
- 不要继续把主要机制写成 attention-logit bias；
- 不要用 gold answer 构造 non-oracle selector；
- 不要把 oracle selector 当作可部署方法；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement，除非后续 generation eval 有统计报告支持；
- 不要覆盖 3A-0 / 3A-1 输出目录。
```

允许：

```text
- 复用 2J-Fix / 2K-W / 3A-1 的 span selector artifacts；
- 复用 3A-1 的 eligible subset 构造逻辑；
- 使用 gold answer 做 eval-only gold-first-token logprob；
- 使用 oracle on-path span 做 eval-only mechanism diagnostic；
- 实现 hidden/residual/value-level hook；
- 做 small-scale lambda/beta sweep；
- 做 no-intervention vs random vs oracle 的 paired bootstrap；
- 输出 review gate。
```

---

## 输入

优先复用：

```text
outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl
outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl
data/raw/gsm8k_train_normalized.jsonl
src/recover_attention/attention_bias_steering.py
scripts/sprint_3A_1_controlled_attention_guidance.py
```

如果 3A-1 的 output files 已存在，也可以读取：

```text
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/steering_subset_manifest_500.jsonl
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/target_selector_report_500.json
outputs/logs/sprint_3A_1_controlled_attention_guidance_500/oracle_sanity_diagnostic_report.json
```

但不得依赖 3A-1 output files 作为唯一输入。若不存在，应从 2J-Fix / 2K-W artifacts 重新构造 subset。

---

## 输出目录

```text
outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic/
```

必须输出：

```text
representation_subset_manifest.jsonl
representation_intervention_config.json
representation_forward_manifest.jsonl
representation_intervention_fidelity_report.json
gold_logprob_delta_report.json
oracle_vs_random_diagnostic_report.json
selector_comparison_report.json
harm_rate_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_representation_level_oracle_diagnostic.md
```

同时新增或更新：

```text
docs/codex_tasks/sprint_3B_0_representation_level_oracle_intervention_diagnostic.md
docs/progress/sprint_3_history.md
PROGRESS.md
```

---

## 推荐实现方案

本 sprint 优先做 **residual-level intervention**，因为它比 value-vector intervention 更容易稳定实现，也更适合作为最小机制诊断。

### 方案 A：answer-position residual injection

在 selected span token 上取某一层 hidden state 的平均表示：

```text
span_repr_l = mean(hidden_l[selected_span_tokens])
```

然后在 answer-position token 的同层 residual stream 上注入：

```text
hidden_l[answer_position] += beta * normalize(span_repr_l)
```

或使用更保守版本：

```text
hidden_l[answer_position] += beta * normalize(span_repr_l - mean_question_repr_l)
```

其中：

```text
beta ∈ {0.05, 0.1, 0.2, 0.4, 0.8}
layers ∈ {[12], [16], [20], [24], [16,24], [20,24]}
query/target position = answer_position
top_k = 2
```

优先使用小规模 sweep：

```text
primary_n = 100 或 120
selectors = random / surface / attention_only / attention_x_resp_pos / oracle
```

---

### 方案 B：span-token residual amplification

在 selected span token 的 hidden state 上做放大：

```text
hidden_l[selected_span_tokens] *= (1 + beta)
```

或者：

```text
hidden_l[selected_span_tokens] += beta * normalize(hidden_l[selected_span_tokens])
```

这个方案更接近“增强关键 token 表示本身”，但在 causal transformer 中不一定能影响后续 answer-position，取决于层和后续 attention。可以作为 diagnostic variant，不作为 primary。

---

### 方案 C：value-level intervention，暂列为 optional

如果工程成本可控，可以额外尝试 attention value path：

```text
V[selected_span_tokens] *= (1 + beta)
```

或：

```text
attn_output_answer_position += beta * value_span_mean
```

但本 sprint 不要求一定完成 value-level hook。若 value hook 复杂，应记录为 deferred，不要为了实现它扩大 sprint。

---

## Primary experiment

Primary config 建议：

```text
intervention_type = answer_position_residual_injection
primary_n = 120
top_k = 2
layers = [16, 24]
beta_sweep = [0.05, 0.1, 0.2, 0.4, 0.8]
selectors = random / surface / attention_only / attention_x_resp_pos / oracle
metric_position = answer_position
```

每个 question 先跑 no-intervention baseline：

```text
base_logits
base_hidden_states
base_gold_first_token_logprob
```

然后对每个 selector × beta 做 steered forward：

```text
steered_logits
steered_hidden_states
gold_first_token_logprob_delta
output_js
top1_changed
entropy_delta
margin_delta
harm_proxy
```

---

## 指标定义

### 1. 机制 fidelity

确认 hook 真的起作用：

```text
hook_registered
hook_triggered_layers
hook_removed
intervention_norm
answer_position_hidden_delta_norm
output_js
```

要求：

```text
hook_ok_rate = 1.0
hidden_delta_norm > 0
output_js 随 beta 增大而可测上升
```

---

### 2. answer-directed proxy

核心指标：

```text
gold_first_token_logprob_delta =
logprob(gold_answer_first_token | intervened)
-
logprob(gold_answer_first_token | no_intervention)
```

gold answer 只允许用于 eval-only。

---

### 3. oracle selectivity

关键比较：

```text
oracle_minus_random =
gold_first_token_logprob_delta(oracle)
-
gold_first_token_logprob_delta(random)
```

按 question 做 paired bootstrap。

需要输出：

```text
mean
CI95 low
CI95 high
stable_positive = CI95 low > 0
```

---

### 4. selector comparison

比较：

```text
random
surface
attention_only
attention_x_resp_pos
oracle
```

输出每个 selector 的：

```text
mean_gold_first_token_logprob_delta
CI95
mean_output_js
harm_rate
top1_changed_rate
```

---

### 5. harm proxy

沿用 3A-1 的 harm proxy：

```text
harm = top1_changed or entropy_delta > 1.0 or margin_delta < -0.25
```

也可以记录更细：

```text
top1_changed_rate
entropy_delta_mean
margin_delta_mean
gold_logprob_negative_rate
```

---

## Review gate

### 机制层

必须回答：

```text
1. hook 是否可靠？
2. intervention 是否真的改变 answer-position hidden state？
3. output distribution 是否在某个 beta regime 下可测移动？
4. harm 是否随 beta 爆炸？
```

---

### oracle diagnostic 层

必须回答：

```text
5. oracle representation intervention 是否显著优于 random？
6. oracle 是否优于 surface？
7. oracle 是否优于 attention_x_resp_pos？
8. attention_x_resp_pos 是否优于 random / surface？
```

---

### 决策层

根据结果分三类：

#### 情况 A：oracle > random，且 harm 可控

说明：

```text
representation-level intervention 有机制信号；
3A attention-bias 失败不是因为 span guidance 完全无效，
而是因为 attention-logit bias 不是合适的干预通道。
```

下一步：

```text
Sprint 3B-1: Representation Intervention Selector Approximation
```

目标：

```text
用 non-oracle selector 逼近 oracle representation intervention。
```

---

#### 情况 B：oracle > random，但 non-oracle 不行

说明：

```text
机制成立；
瓶颈重新回到 selector quality。
```

下一步：

```text
改进 selector：
attention_x_resp_pos
+ NLI semantic necessity
+ recoverability / instability
+ span type prior
```

但仍不进入 2000。

---

#### 情况 C：oracle ≈ random 或 oracle < random

说明：

```text
即便在 representation-level，正确 span 也不能稳定推动答案；
当前 GSM8K span-level intervention 任务可能不适合 guidance；
或者 selected span 粒度 / target position / intervention direction 仍不对。
```

下一步不要扩大实验，应转向：

```text
1. 更细粒度 reasoning step intervention；
2. answer-position residual patching from correct run；
3. task redesign；
4. value-level / MLP-level causal tracing；
5. 或暂停 steering 线，回到 detection / diagnosis。
```

---

## 通过条件

本 sprint 不要求正结果。

通过条件是：

```text
- 所有 required reports 输出；
- hook 可靠；
- gold / oracle 只用于 eval-only；
- no training / no LoRA / no 2000；
- oracle-vs-random 结论有 paired CI；
- harm 有 beta curve；
- review gate 明确给出下一步路线。
```

即使结果是负面的，只要机制诊断清楚，也算 sprint completed。

---

## 建议新增脚本

```text
scripts/sprint_3B_0_representation_level_oracle_intervention.py
```

建议新增测试：

```text
tests/test_representation_level_intervention.py
```

测试至少覆盖：

```text
1. gold_first_token_logprob 只作为 eval metric；
2. random / oracle selector 隔离；
3. beta 不允许为负；
4. hook 注册与移除；
5. selected span token mapping 不复用错误 cache；
6. residual injection 后 hidden delta norm > 0；
7. no-intervention baseline 不被污染。
```

---

## 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_representation_level_intervention.py -q

conda run -n recover_attention python scripts/sprint_3B_0_representation_level_oracle_intervention.py \
  --primary-n 120 \
  --betas 0.05 0.1 0.2 0.4 0.8 \
  --layers 16 24 \
  --output-dir outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 最终 review gate 必须回答的问题

`review_gate_representation_level_oracle_diagnostic.md` 必须逐条回答：

```text
1. 本 sprint 是否仍然严格保持非 full 3B、非 2000？
2. 使用了多少 eligible questions？
3. 使用了哪些 selectors？
4. 使用了哪些 layers / beta？
5. primary intervention type 是什么？
6. hook 是否可靠？
7. output shift 是否可测？
8. harm 是否可控？
9. oracle 是否显著优于 random？
10. oracle 是否显著优于 attention_x_resp_pos？
11. attention_x_resp_pos 是否优于 random / surface？
12. 当前负/正结果支持哪一种解释？
13. 下一步应该走 selector improvement、representation-level continuation、value-level intervention，还是 task redesign？
14. 是否允许进入 2000 rerun？
15. 是否允许声称 hallucination reduction / answer accuracy improvement？
```

默认最终答案应保守：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3B=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

除非 oracle representation intervention 在 paired bootstrap 中稳定优于 random，且 harm 可控，也只能说：

```text
representation-level oracle diagnostic shows selective answer-directed signal
```

不能说：

```text
accuracy improved
hallucination reduced
guidance solved
```

---

## 最后更新

完成后更新：

```text
PROGRESS.md
docs/progress/sprint_3_history.md
```

如果 3A-1 的聚合 output reports 尚未提交，先不要改写 3A-1 结论，只在 3B-0 preflight 中记录：

```text
3A-1 summary exists in PROGRESS / sprint_3_history,
but detailed output reports may be absent from GitHub or ignored by .gitignore.
```

必要时补交 3A-1 的小型聚合报告文件：

```text
oracle_sanity_diagnostic_report.json
answer_position_output_shift_report_500.json
harm_rate_report_500.json
baseline_comparison_report_500.json
review_gate_controlled_attention_guidance_500.md
```
