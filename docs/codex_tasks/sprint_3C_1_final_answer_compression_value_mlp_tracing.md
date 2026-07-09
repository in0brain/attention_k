# Sprint 3C-1: Final-Answer Compression Value/MLP Causal Tracing

## 定位

本 sprint 是 3C-0-Fix 之后的 **module-level causal tracing sprint**。

它不是 full Sprint 3C，不是 2000-scale rerun，不是训练任务，不是 LoRA / finetuning，不是 autoregressive generation accuracy eval，也不允许声称 hallucination reduction 或 answer accuracy improvement。

本轮只回答一个问题：

```text id="nbyr3q"
3C-0-Fix 发现 final-answer readout / compression 阶段存在 correct-run activation repair signal；
那么这个 signal 到底来自哪个模块写入路径？
```

更具体地说，本 sprint 不再做粗粒度 whole residual replacement，而是把 answer-readout 位置的残差更新拆成：

```text id="f2ye8b"
attention output path
MLP output path
residual stream path
```

然后分别做 activation patching，判断：

```text id="mdqvll"
正确答案方向到底是由 attention output 写入，
还是由 MLP down-proj / MLP output 写入，
还是只是 whole residual 的非特异扰动？
```

默认结论标记：

```text id="qmv93k"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 执行前必读

执行前必须先阅读：

```text id="i8vilw"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md

docs/codex_tasks/sprint_3C_0_correct_wrong_activation_patching_reasoning_steps.md
docs/codex_tasks/sprint_3C_0_fix_answer_proxy_recheck.md

src/recover_attention/activation_patching.py
src/recover_attention/answer_proxy_metrics.py

scripts/sprint_3C_0_correct_wrong_activation_patching.py
scripts/sprint_3C_0_fix_answer_proxy_recheck.py

tests/test_activation_patching.py
tests/test_answer_proxy_metrics.py
```

---

## Preflight

必须先输出：

```text id="x5l1br"
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/preflight_report.md
```

Preflight 必须检查：

```text id="i4zq17"
1. PROGRESS.md 顶部 current status 是否已经不是 3A-0；
2. docs/progress/sprint_3_artifact_manifest.md 是否存在；
3. 3C-0-Fix 输出目录是否存在；
4. 3C-0-Fix 是否记录 corrected proxy 后 Case B；
5. 3C-0-Fix 是否复用了 3C-0 pair manifest；
6. corrected_pair_manifest.jsonl 是否可用；
7. corrected clean_direction / control comparison reports 是否可用；
8. 本轮是否会覆盖 3C-0 / 3C-0-Fix 输出目录；
9. 本轮是否仍然非 2000、非 full 3C、非训练；
10. outputs 若被 gitignore，是否会更新 artifact manifest。
```

如果 3C-0-Fix 输出文件本地缺失，不要伪造结果；应在 preflight 中记录缺失，并从 3C-0 correct/wrong pair manifest 重新构造最小复核输入。

---

## 前因后果：为什么要做 3C-1

### 1. 3A / 3B 已经基本否定 span-level steering

前面阶段说明：

```text id="xgvcjn"
span-level keyness signal 存在；
attention / output-effect 能弱区分 on-path 与 off-path span。
```

但把 span importance 直接变成 steering 没有成功：

```text id="nff1ce"
3A attention-logit bias：
多看 oracle span 不比多看 random span 更能改善答案。

3B answer-position span residual injection：
注入 oracle span representation 不比注入 random span 更能改善答案。
```

因此：

```text id="tec5p9"
关键 span 更像 detection / diagnosis signal，
不是直接可控的 answer-steering handle。
```

---

### 2. 3C-0 初版因为 proxy 问题过于悲观

3C-0 做 correct-vs-wrong activation patching，初步判定 Case C。

但后来发现 proxy 有两个问题：

```text id="hxfwkt"
1. 早期 first-token 取到了 leading-space token；
2. 读 logits 的位置不是 final answer 数字前一位；
3. 只看首 digit，而不是完整数字答案序列。
```

所以 3C-0 初版的平坦 null 不能作为最终结论。

---

### 3. 3C-0-Fix 改变了结论

3C-0-Fix 修正为：

```text id="xmm4nr"
final answer 数字前一位 logits
+ full numeric sequence logprob
+ corrected clean_direction_score
```

修正后发现：

```text id="mmxqg8"
correct-run activation 在答案读出位有明显修复信号；
但显式 reasoning-step patch 没有稳定 donor-specific / site-specific 优势。
```

更具体地说：

```text id="n9i0n8"
reasoning-step patch:
- 比 no_patch 略好；
- 但不稳定优于 random donor；
- 不稳定优于 same-trace random position；
- 所以不是清晰 causal bottleneck。

final-answer readout patch:
- clean_direction 很强；
- 稳定优于 random donor；
- 但没有跑赢 same-trace random position；
- 高层 harm 明显。
```

因此 3C-0-Fix 的结论是：

```text id="ywh6p2"
correct context 在答案读出 / 压缩阶段有用；
但 whole residual replacement 太粗，
还没有定位到 donor-specific、site-specific、低 harm 的真正 causal path。
```

---

## 本轮核心假设

新的机制假设是：

```text id="z2y8uj"
正确答案信息不是由 key span 直接控制，
而是在 final-answer compression / answer readout 阶段，
通过某些 module output 被写入 residual stream。
```

可能路径包括：

```text id="eh7k0b"
1. attention output：
   从上下文中搬运答案相关信息到 answer readout 位置；

2. MLP output：
   在 answer readout 位置写入答案方向 / 数字方向 / logit-relevant feature；

3. whole residual：
   包含有用信息，但太混杂，导致 high harm 与 non-specific effect。
```

本 sprint 要做的是：

```text id="r6u5tv"
把 whole residual replacement 拆成 attention-output patch 与 MLP-output patch，
找到更窄、更选择性、更低 harm 的 causal site。
```

---

## 严格边界

禁止：

```text id="g81kk1"
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要回到 attention-bias steering；
- 不要回到 span residual injection；
- 不要重新设计 selector；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 teacher-forced proxy 写成 autoregressive generation result；
- 不要覆盖 3C-0 / 3C-0-Fix 输出目录；
- 不要把 oracle/correct donor 当作可部署方法。
```

允许：

```text id="z1acvy"
- 复用 3C-0-Fix 的 corrected pair manifest；
- 复用 corrected answer-sequence proxy；
- 在 answer readout 位置做 module-level activation patching；
- 对 attention output / MLP output / residual 做对比；
- 做 layer / alpha / module sweep；
- 做 paired bootstrap；
- 输出 review gate；
- 更新 PROGRESS / sprint_3_history / artifact manifest。
```

---

## 输入

优先读取：

```text id="sq4tcz"
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/corrected_pair_manifest.jsonl
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/corrected_clean_direction_report.json
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/corrected_control_comparison_report.json
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/review_gate_answer_proxy_recheck.md

outputs/logs/sprint_3C_0_correct_wrong_activation_patching/correct_wrong_pair_manifest.jsonl
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/reasoning_step_position_report.json

data/raw/gsm8k_train_normalized.jsonl
```

可复用代码：

```text id="w5f84o"
src/recover_attention/activation_patching.py
src/recover_attention/answer_proxy_metrics.py
```

如果 3C-0-Fix outputs 因 gitignore 未提交但本地存在，直接使用本地文件。

如果本地不存在，不要伪造；应在 preflight 中记录，并从 3C-0 pair manifest 重新构造 corrected pair input。

---

## 输出目录

必须使用新目录：

```text id="mxftix"
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/
```

必须输出：

```text id="d27naf"
preflight_report.md
module_patching_config.json
module_patch_pair_manifest.jsonl
module_activation_capture_manifest.jsonl
module_patching_forward_manifest.jsonl
module_patching_fidelity_report.json
module_patching_effect_report.json
module_control_comparison_report.json
layer_module_heatmap_report.json
harm_control_report.json
donor_specificity_report.json
site_specificity_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_final_answer_compression_tracing.md
```

同时新增或更新：

```text id="lhb2hk"
docs/codex_tasks/sprint_3C_1_final_answer_compression_value_mlp_tracing.md
src/recover_attention/module_causal_tracing.py
scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py
tests/test_module_causal_tracing.py
docs/progress/sprint_3_history.md
PROGRESS.md
docs/progress/sprint_3_artifact_manifest.md
```

---

## Primary experiment

本轮 primary target 是：

```text id="pue6p5"
final_answer_readout_position
```

定义为：

```text id="n50mlh"
final answer 数字前一位 token position
```

不是 trace 末位，也不是 answer 数字之后的位置。

评价指标使用 3C-0-Fix 的 corrected proxy：

```text id="urifhd"
full numeric answer sequence logprob
corrected clean_direction_score
```

---

## Patch 对象

本轮比较三类 patch target：

### 1. whole residual patch

作为 baseline/control，复现 3C-0-Fix 中有效但粗糙的路径：

```text id="iyyoia"
hidden_states[layer][answer_readout_pos]
```

该项用于对比，不是最终目标。

---

### 2. attention output patch

patch decoder layer 的 self-attention output：

```text id="dh6zz4"
layer.self_attn output at answer_readout_pos
```

目标：

```text id="n5vhl7"
判断 correct answer information 是否通过 attention output 被搬运到答案读出位。
```

---

### 3. MLP output patch

patch decoder layer 的 MLP output / down-proj output：

```text id="zrki0x"
layer.mlp output at answer_readout_pos
```

目标：

```text id="p6iom2"
判断 correct answer information 是否由 MLP 在答案读出位写入 residual stream。
```

---

## Patch 形式

Primary patch 使用 interpolation，而不是只做 hard replace：

```text id="teop3d"
patched_output =
(1 - alpha) * recipient_output
+ alpha * donor_output
```

alpha sweep：

```text id="t6zv1k"
alpha ∈ {0.25, 0.5, 0.75, 1.0}
```

原因：

```text id="lf6b9w"
3C-0-Fix 中 whole residual replacement 在高层 harm 很大；
alpha sweep 用于寻找 gold↑wrong↓ 但 harm 可控的 regime。
```

如果实现成本较高，最小版本必须至少支持：

```text id="sxv30a"
alpha = 0.5
alpha = 1.0
```

---

## Layer sweep

优先层：

```text id="h1mc92"
layers = [16, 20, 24]
```

可选扩展：

```text id="xuwyiv"
layers = [12, 16, 20, 24, 28]
```

3C-0-Fix 显示 L16 harm 低于 L24，因此本轮必须重点比较：

```text id="l22u7k"
L16 是否存在低 harm selective signal；
L24 是否只是强扰动 / high harm；
L20 是否是折中层。
```

---

## Controls

必须包含以下 controls：

```text id="otmuqh"
no_patch
correct_donor_patch
random_donor_patch
same_trace_random_position_patch
same_pair_wrong_position_patch
wrong_donor_self_patch
```

解释：

```text id="gucdq2"
correct_donor_patch:
  从 same-question correct trace 的 answer_readout position patch 到 wrong trace。

random_donor_patch:
  从其它 question / random trace 的同类位置 patch。

same_trace_random_position_patch:
  从 same correct trace 的非 answer-readout random position patch。

same_pair_wrong_position_patch:
  从 same question correct trace 的其它 position patch。

wrong_donor_self_patch:
  从 wrong trace 自身对应位置 patch，应该接近 no-op，用于检测 hook 或 metric 偏差。
```

核心比较：

```text id="pw0ehf"
correct_donor_patch - random_donor_patch
correct_donor_patch - same_trace_random_position_patch
correct_donor_patch - same_pair_wrong_position_patch
correct_donor_patch - wrong_donor_self_patch
attention_output_patch - whole_residual_patch
mlp_output_patch - whole_residual_patch
attention_output_patch - mlp_output_patch
```

所有比较都要 paired bootstrap。

---

## 指标定义

沿用 3C-0-Fix corrected proxy。

### 1. Gold sequence delta

```text id="nrlkgq"
gold_seq_logprob_delta =
log P(gold_answer | patched wrong prefix)
-
log P(gold_answer | unpatched wrong prefix)
```

### 2. Wrong sequence delta

```text id="xmbj2u"
wrong_seq_logprob_delta =
log P(wrong_answer | patched wrong prefix)
-
log P(wrong_answer | unpatched wrong prefix)
```

### 3. Clean direction score

```text id="jonvj9"
clean_direction_score =
gold_seq_logprob_delta
-
wrong_seq_logprob_delta
```

### 4. Per-token normalized clean direction

```text id="s5s9jk"
clean_direction_score_per_token
```

同时保留 raw sequence score 与 per-token score。

---

## Donor-specificity

本轮新增关键指标：

```text id="en3815"
donor_specificity_score =
clean_direction(correct_donor_patch)
-
clean_direction(random_donor_patch)
```

必须输出：

```text id="gmmdqy"
mean
CI95 low
CI95 high
stable_positive = CI95 low > 0
```

如果 donor-specificity 不稳定为正，说明：

```text id="uv7efn"
patch 效果可能只是 generic correct-context / generic perturbation，
还不是 donor-specific causal repair。
```

---

## Site-specificity

本轮新增关键指标：

```text id="pkcg7z"
site_specificity_score =
clean_direction(answer_readout_position_patch)
-
clean_direction(same_trace_random_position_patch)
```

必须输出：

```text id="n5xn3e"
mean
CI95 low
CI95 high
stable_positive = CI95 low > 0
```

如果 site-specificity 不稳定为正，说明：

```text id="dqk9ir"
答案读出位并不是唯一 causal site，
或者 patch 仍然过粗。
```

---

## Harm control

沿用并强化 harm proxy：

```text id="flah7f"
harm =
top1_changed_to_non_gold
or entropy_delta > 1.0
or margin_delta < -0.25
or gold_seq_logprob_delta < -0.5
or clean_direction_score < -0.5
```

必须按：

```text id="afonq8"
module_type
layer
alpha
control_type
```

输出 harm curve。

Primary success 不能只看 clean_direction 高，还必须看 harm。

---

## Primary success criterion

本 sprint 的 primary success 不是 accuracy improvement。

Primary success 是找到：

```text id="wyptt2"
module_type × layer × alpha
```

满足：

```text id="w55d9b"
1. clean_direction_score stable positive；
2. donor_specificity stable positive；
3. site_specificity stable positive；
4. harm 明显低于 whole residual high-harm baseline；
5. 效果集中在 attention output 或 MLP output，而不是所有 module 同幅上升。
```

如果只满足 clean_direction，但 donor/site specificity 不过，判定为：

```text id="l2tjm3"
non-specific answer-readout perturbation
```

---

## 结果解释逻辑

### 情况 A：MLP output 找到低 harm selective site

如果：

```text id="t51cpq"
MLP output patch:
clean_direction stable positive
donor_specificity stable positive
site_specificity stable positive
harm controllable
```

说明：

```text id="qvtyx8"
正确答案方向主要由 answer-readout 阶段的 MLP 写入。
```

下一步：

```text id="wba0ym"
Sprint 3C-2: MLP Readout Direction Analysis / Low-Harm Steering Probe
```

仍不进入 2000。

---

### 情况 B：attention output 找到低 harm selective site

如果：

```text id="f2c8mg"
attention output patch:
clean_direction stable positive
donor_specificity stable positive
site_specificity stable positive
harm controllable
```

说明：

```text id="m28fo0"
正确答案信息主要通过 attention output 被搬运到答案读出位。
```

下一步：

```text id="vc5irq"
Sprint 3C-2: Attention-Output Readout Path Analysis
```

仍不回到 attention-logit bias。

注意：

```text id="gekk69"
attention-output 有效 ≠ attention-logit bias 有效。
```

前者是内容路径，后者是权重路径。

---

### 情况 C：whole residual 有效，但 attention/MLP 都不特异

说明：

```text id="cpdppz"
有效信息可能分布在多个子路径，
或者 hook 分解方式不够准确；
whole residual 效果仍可能是粗扰动。
```

下一步：

```text id="y7lrzg"
更细的 residual decomposition / multi-site patching，
或暂停 steering，回 detection/diagnosis。
```

---

### 情况 D：全部 module-level patch 都不通过 donor/site specificity

说明：

```text id="fpbn8s"
3C-0-Fix 的正结果主要是 non-specific answer-readout perturbation；
没有找到可用 causal site。
```

下一步：

```text id="e7uyyb"
暂停 steering，
回到 detection / diagnosis，
或改任务。
```

---

## 实现建议

新增模块：

```text id="zp7p1o"
src/recover_attention/module_causal_tracing.py
```

至少实现：

```text id="h607vm"
capture_module_outputs()
register_module_patch_hook()
remove_hooks()
patched_forward_with_module_output()
build_answer_readout_prefix()
compute_sequence_logprob_with_patch()
paired_bootstrap_delta()
compute_donor_specificity()
compute_site_specificity()
```

---

## Hook 位置建议

Qwen2.5 decoder layer 结构一般包含：

```text id="to0nkf"
layer.self_attn
layer.mlp
```

可优先 hook：

```text id="xueioq"
self_attn output
mlp output
decoder layer output / residual output
```

注意：

```text id="ecjynk"
不同 transformers 版本的 module output 可能是 tuple；
hook 必须兼容 tensor / tuple output；
hook 必须 clone，避免污染原 forward；
hook 必须 finally remove；
必须记录 triggered layer/module。
```

---

## 建议新增脚本

```text id="u5iuqo"
scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py
```

推荐命令：

```bash id="m5o2fw"
conda run -n recover_attention python -m pytest tests/test_module_causal_tracing.py -q

conda run -n recover_attention python scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py \
  --input-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --output-dir outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing \
  --layers 16 20 24 \
  --modules attention_output mlp_output residual_output \
  --alphas 0.25 0.5 0.75 1.0 \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 建议新增测试

```text id="kk8l39"
tests/test_module_causal_tracing.py
```

至少覆盖：

```text id="fcbqpi"
1. module hook 注册、触发、移除；
2. hook 只 patch target position；
3. remove 后 baseline 恢复；
4. tuple output / tensor output 都能处理；
5. alpha interpolation 公式正确；
6. wrong_donor_self_patch 接近 no-op；
7. donor_specificity = correct - random；
8. site_specificity = answer_readout - random_position；
9. sequence-logprob proxy 复用 answer_proxy_metrics；
10. negative alpha 不允许；
11. module_type 只能是允许值；
12. preflight 不覆盖 3C-0 / 3C-0-Fix 输出。
```

---

## Review gate

`review_gate_final_answer_compression_tracing.md` 必须逐条回答：

```text id="me4fp9"
1. 本轮是否只是 module-level causal tracing，而非 full 3C？
2. 是否复用了 3C-0-Fix corrected proxy？
3. 是否仍然使用 final answer 数字前一位作为 answer readout position？
4. 是否计算完整数字序列 logprob？
5. 使用了多少 correct/wrong pairs？
6. 覆盖哪些 layers？
7. 覆盖哪些 module types？
8. 覆盖哪些 alpha？
9. hook fidelity 是否通过？
10. 是否有 module × layer × alpha 的 clean_direction stable positive？
11. 是否有 donor_specificity stable positive？
12. 是否有 site_specificity stable positive？
13. attention output 与 MLP output 哪个更强？
14. 是否存在低 harm regime？
15. 是否只是 whole residual high-harm 效果？
16. 是否仍出现 gold/wrong 同向上升？
17. 是否允许进入 2000？
18. 是否允许声称 hallucination reduction / answer accuracy improvement？
19. 下一步应该做 MLP readout direction、attention-output path、multi-site tracing，还是暂停 steering？
```

默认最终标记：

```text id="l5c9eo"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 最后更新

完成后必须更新：

```text id="c9yi91"
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

如果 outputs 被 gitignore，不提交大 jsonl，但必须在 artifact manifest 中记录：

```text id="g2frce"
outputs/logs/sprint_3C_1_final_answer_compression_value_mlp_tracing/
```

至少记录：

```text id="gtx6gw"
module_patching_effect_report.json
module_control_comparison_report.json
donor_specificity_report.json
site_specificity_report.json
layer_module_heatmap_report.json
harm_control_report.json
review_gate_final_answer_compression_tracing.md
```

最终不得写：

```text id="zpyx6c"
accuracy improved
hallucination reduced
ready for 2000
```

最多只能写：

```text id="wkd1fx"
module-level causal tracing found / did not find a selective low-harm causal site
at the final-answer compression stage under teacher-forced sequence-logprob proxy.
```
