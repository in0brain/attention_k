# Sprint 3C-0-Fix: Answer-Position Proxy Repair and Sequence-Logprob Recheck

## 定位

本 sprint 是 **3C-0 的低成本 proxy 修正与复核**。

它不是 full Sprint 3C，不是 2000-scale rerun，不是新 steering 机制，不是训练任务，不是 LoRA / finetuning，也不允许声称 hallucination reduction 或 answer accuracy improvement。

本轮只回答一个问题：

```text id="lcge6x"
3C-0 的负结果是否会因为 answer proxy 不够精确而改变？
```

更具体地说，本轮要修正两个已知 proxy 弱点：

```text id="ieybqc"
1. 不再读取整段 trace 最后一个 token 的 logits；
   而是在 final answer 数字前一位读取 logits。

2. 不再只看答案首个 digit token；
   而是计算完整数字答案序列的 conditional logprob。
```

默认结论标记：

```text id="mvymnv"
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
```

---

## 执行前必读

执行前必须先阅读：

```text id="buloea"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md

docs/codex_tasks/sprint_3C_0_correct_wrong_activation_patching_reasoning_steps.md

src/recover_attention/activation_patching.py
scripts/sprint_3C_0_correct_wrong_activation_patching.py
tests/test_activation_patching.py
```

如果 `sprint_3_artifact_manifest.md` 不存在，先检查 3C-0 preflight 是否已经处理 outputs gitignore 问题；不要伪造 artifact 状态。

---

## 前因后果：为什么要做这个 fix

3C-0 已经跑通 correct-vs-wrong activation patching：

```text id="twrd1s"
correct run activation
→ patch 到 wrong run reasoning-step position
→ 看 wrong run 是否更接近 gold answer
```

修复 first-token bug 后，3C-0 的主要结果是 Case C：

```text id="qu5xrh"
correct patch 没有稳定优于 random/control；
gold logprob 与 wrong logprob 同向上升；
整体更像非选择性 perturbation，而不是 causal repair。
```

但 3C-0 仍有两个 proxy 弱点：

### 弱点 1：logits 读取位置不够准

3C-0 当前可能读取的是：

```text id="slljjp"
整段 teacher-forced trace 的末位 logits
```

但真正应该评价答案的位置是：

```text id="lcvvur"
final answer 数字前一位
```

例如：

```text id="n4r6e0"
... #### 42
```

应该在模型即将预测 `42` 的位置读 logits，而不是在整段 trace 末尾读 logits。

---

### 弱点 2：只看 first digit token 太粗

对于答案：

```text id="kwvg07"
42
```

只看 `"4"` 不够。

更合理的是计算完整数字序列：

```text id="c4fp9y"
P("42" | prefix_before_answer)
```

也就是：

```text id="lvo6y0"
log P("4" | prefix)
+ log P("2" | prefix + "4")
```

这样可以避免：

```text id="jm77s9"
只提高第一个 digit，但完整答案仍然错误
```

的问题。

---

## 本轮核心问题

本轮只回答：

```text id="f7028c"
在修正 answer-position proxy 后，
3C-0 的 correct-run activation patching 是否仍然没有 selective causal benefit？
```

如果修正后仍然负，说明：

```text id="jf0l6t"
3C-0 的负结果不是由 proxy 读法造成的；
可以更有把握地转向 value / MLP causal tracing。
```

如果修正后变正，说明：

```text id="pvmqkx"
3C-0 原始 proxy 太粗；
需要围绕 corrected answer-sequence proxy 重新评估 activation patching。
```

---

## 严格边界

禁止：

```text id="d23jtb"
- 不要进入 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要训练模型；
- 不要 LoRA / finetuning；
- 不要改成 attention-bias 或 residual span injection；
- 不要新增复杂 steering 机制；
- 不要重新设计 selector；
- 不要声称 hallucination reduction；
- 不要声称 answer accuracy improvement；
- 不要把 teacher-forced proxy 写成 autoregressive generation result；
- 不要覆盖 3C-0 原始输出目录。
```

允许：

```text id="yx7zdk"
- 复用 3C-0 的 trace_sampling_manifest / correct_wrong_pair_manifest；
- 复用 3C-0 的 activation patching setup；
- 重新运行 patching forward；
- 修正 answer-position locating；
- 计算 full numeric sequence logprob；
- 重新计算 clean_direction_score；
- 输出 corrected reports；
- 更新 PROGRESS / sprint_3_history。
```

---

## 输入

优先读取 3C-0 输出：

```text id="dlb09q"
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/trace_sampling_manifest.jsonl
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/correct_wrong_pair_manifest.jsonl
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/reasoning_step_position_report.json
outputs/logs/sprint_3C_0_correct_wrong_activation_patching/activation_patching_config.json
```

如果这些文件因 outputs gitignore 未提交但本地存在，直接读取本地文件。

如果本地缺失，不要伪造；在 preflight 中记录缺失，并重新构造小规模 trace pairs：

```text id="x9sb66"
primary_questions <= 80
samples_per_question <= 6
```

但本轮优先目标是复核 proxy，不是扩大采样。

---

## 输出目录

必须使用新目录，不覆盖 3C-0：

```text id="jew79i"
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/
```

必须输出：

```text id="ltnfyv"
preflight_proxy_fix_report.md
answer_span_extraction_report.json
corrected_pair_manifest.jsonl
corrected_patching_forward_manifest.jsonl
corrected_sequence_logprob_report.json
corrected_clean_direction_report.json
corrected_control_comparison_report.json
corrected_layer_position_heatmap_report.json
corrected_harm_rate_report.json
failure_case_report.jsonl
success_case_report.jsonl
review_gate_answer_proxy_recheck.md
```

同时新增或更新：

```text id="objjsy"
docs/codex_tasks/sprint_3C_0_fix_answer_proxy_recheck.md
src/recover_attention/answer_proxy_metrics.py
scripts/sprint_3C_0_fix_answer_proxy_recheck.py
tests/test_answer_proxy_metrics.py
docs/progress/sprint_3_history.md
PROGRESS.md
docs/progress/sprint_3_artifact_manifest.md
```

---

## Preflight

输出：

```text id="xohspt"
preflight_proxy_fix_report.md
```

必须检查：

```text id="cm3sk7"
1. PROGRESS.md 顶部 current status 是否已经不是 3A-0；
2. docs/progress/sprint_3_artifact_manifest.md 是否存在；
3. 3C-0 原始输出目录是否存在；
4. correct_wrong_pair_manifest.jsonl 是否存在；
5. 3C-0 是否记录了 first-token bug 修复；
6. 本轮是否会覆盖 3C-0 输出目录；
7. 本轮是否仍然非 2000、非 full 3C、非训练。
```

如果 `PROGRESS.md` 顶部仍过期，先修正。

---

## Answer span extraction 规则

必须实现 robust final answer span extraction。

优先级：

### 1. 首选 `#### <number>`

优先匹配：

```text id="x9ispy"
#### 42
#### -17
#### 3.5
#### 1,250
```

正则建议：

```text id="wxypms"
####\s*(-?\d[\d,]*(?:\.\d+)?)
```

### 2. 次选明确 final answer phrase

如果没有 `####`，匹配：

```text id="xowfdl"
answer is 42
final answer is 42
so the answer is 42
```

### 3. 最后才允许 fallback last number

如果前两者都失败，才使用 last number fallback，并记录 warning：

```text id="k3hwme"
answer_extraction_method = "fallback_last_number"
answer_extraction_warning = true
```

### 4. 禁止把列表序号当答案

必须避免把下面这种数字误判为答案：

```text id="vmkddi"
1. First, ...
2. Then, ...
```

如果仅发现列表序号，标记 parse failure，不要强行当答案。

---

## Tokenization 修正

必须避免 3C-0 早期的 leading-space token bug。

实现函数：

```text id="p01o7y"
answer_token_ids(tokenizer, answer_text)
```

要求：

```text id="gsjx6m"
- 不要返回纯空格 token；
- 对数字答案，返回实际数字 token sequence；
- 对 "42"，应至少包含表示 "4"/"42" 的非空格 token；
- 对 "-17"、"3.5"、"1,250" 正常处理；
- 如果 tokenizer 把 leading space 单独切出，应丢弃 leading whitespace token。
```

必须新增回归测试：

```text id="p3d7u7"
Qwen-style tokenizer may split " 42" into [space, digit...];
answer_token_ids must not return the space token as the answer token.
```

---

## Corrected sequence logprob

实现：

```text id="zv8gok"
sequence_logprob_at_answer_slot(model, tokenizer, prefix_text, answer_text, patch_config=None)
```

定义：

```text id="xyu3t1"
log P(answer_token_1 ... answer_token_n | prefix_text)
=
Σ_i log P(answer_token_i | prefix_text + answer_token_<1:i)
```

注意：

```text id="z5r8n6"
- prefix_text 必须截止到 final answer 数字前；
- 不要把原 wrong answer 数字包含进 prefix；
- gold answer 与 wrong answer 必须在同一个 prefix 下比较；
- gold/wrong answer sequence 长度不同也允许；
- 必须记录 num_answer_tokens。
```

---

## Patch + corrected metric 的计算方式

对于每个 correct/wrong pair：

```text id="zjrsgd"
recipient = wrong trace
donor = correct trace
```

先从 wrong trace 中找到 final answer 数字 span：

```text id="o5bc1i"
wrong_prefix_before_answer
wrong_answer_text
wrong_answer_span_start/end
```

gold answer 来自 GSM8K gold。

然后在同一个 prefix 下比较：

```text id="w5fnyy"
gold_sequence_logprob
wrong_sequence_logprob
```

也就是：

```text id="nfqtlb"
log P(gold_answer | wrong_prefix_before_answer)
log P(wrong_answer | wrong_prefix_before_answer)
```

对于 patched forward，patch 应该作用在：

```text id="mmps8k"
wrong_prefix_before_answer + candidate_answer_prefix
```

注意：

```text id="gmk9uc"
如果 patch position 在 answer slot 之后，则该 patch 对当前 candidate logprob 无意义，应跳过。
```

primary patch positions 只保留：

```text id="mbr9k3"
position < answer_span_start
```

final_answer_position 可以作为 control，但必须明确它不是 answer 数字之后的位置，而是 answer 数字前一位的 prefix position。

---

## Corrected metrics

### 1. Gold sequence delta

```text id="p02vp5"
gold_seq_logprob_delta =
log P(gold_answer | patched wrong prefix)
-
log P(gold_answer | unpatched wrong prefix)
```

### 2. Wrong sequence delta

```text id="s0mzfe"
wrong_seq_logprob_delta =
log P(wrong_answer | patched wrong prefix)
-
log P(wrong_answer | unpatched wrong prefix)
```

### 3. Corrected clean direction score

```text id="t1j3e7"
corrected_clean_direction_score =
gold_seq_logprob_delta
-
wrong_seq_logprob_delta
```

理想方向：

```text id="x3lwu6"
gold_seq_logprob_delta > 0
wrong_seq_logprob_delta < 0
corrected_clean_direction_score > 0
```

### 4. Normalized score

由于不同答案 token 长度不同，额外记录：

```text id="r6ie6g"
gold_seq_logprob_delta_per_token
wrong_seq_logprob_delta_per_token
clean_direction_score_per_token
```

不要只看 per-token；raw sequence score 和 per-token score 都要输出。

---

## Controls

必须保留 3C-0 的 control comparison：

```text id="tyzgeg"
correct_donor_patch
random_donor_patch
same_trace_random_position_patch
final_answer_position_patch
no_patch
```

核心比较：

```text id="evip5j"
correct_reasoning_step_patch - random_donor_patch
correct_reasoning_step_patch - same_trace_random_position_patch
correct_reasoning_step_patch - final_answer_position_patch
```

每个比较都要 paired bootstrap：

```text id="g2rrpv"
mean
CI95 low
CI95 high
stable_positive = CI95 low > 0
```

---

## Primary report

`corrected_clean_direction_report.json` 必须包含：

```text id="tvqqnq"
overall_corrected_clean_direction_mean
overall_corrected_clean_direction_ci95
overall_gold_seq_delta_mean
overall_wrong_seq_delta_mean
overall_per_token_clean_direction_mean

by_position_type
by_layer
by_layer_position
num_pairs
num_forward_rows
num_parse_failures
num_answer_slot_failures
```

`corrected_control_comparison_report.json` 必须包含：

```text id="ciik2q"
correct_patch_vs_random_donor
correct_patch_vs_same_trace_random_position
correct_patch_vs_final_answer_position
stable_positive_flags
```

---

## Review gate

`review_gate_answer_proxy_recheck.md` 必须逐条回答：

```text id="hluhkh"
1. 本轮是否只是 3C-0 proxy fix，而非新 steering？
2. PROGRESS current status 是否已修正？
3. 是否复用了 3C-0 pair manifest？
4. 如果没有复用，为什么？
5. answer extraction 成功率是多少？
6. fallback_last_number 比例是多少？
7. 是否避免了 leading-space token bug？
8. 是否在 final answer 数字前一位读取 logits？
9. 是否计算了完整数字序列 logprob？
10. corrected clean_direction 是否 stable positive？
11. correct patch 是否稳定优于 random donor？
12. correct patch 是否稳定优于 same_trace_random_position？
13. correct patch 是否稳定优于 final_answer_position control？
14. 是否仍然出现 gold/wrong 同向上升？
15. 3C-0 Case C 结论是否改变？
16. 是否允许进入 2000？
17. 是否允许声称 hallucination reduction / answer accuracy improvement？
18. 下一步应该转 value/MLP tracing，还是暂停 steering？
```

---

## 判定逻辑

### 情况 A：修正后仍然负

如果：

```text id="mdz2f2"
corrected_clean_direction_score CI includes 0
correct_patch 不稳定优于 controls
gold/wrong 仍同向上升
```

则结论：

```text id="ee7rf0"
3C-0 的负结果不是 proxy artifact；
teacher-forced residual activation patching 未发现 selective causal state；
下一步若继续机制线，应转 value/MLP causal tracing。
```

推荐下一步：

```text id="h8uwjo"
Sprint 3C-1: Value/MLP Causal Tracing
```

---

### 情况 B：修正后变正

如果：

```text id="bki1go"
corrected_clean_direction_score stable positive
correct_patch 稳定优于 random/control
harm 可控
```

则结论：

```text id="o3ytv3"
3C-0 原 proxy 太粗；
correct-vs-wrong activation patching 可能存在 selective signal；
需要围绕 corrected sequence-logprob proxy 重新评估 3C-0。
```

推荐下一步：

```text id="gt0ua2"
Sprint 3C-0-R: Corrected Activation Patching Re-evaluation
```

仍然不要进入 2000。

---

## 建议新增模块

```text id="kikivt"
src/recover_attention/answer_proxy_metrics.py
```

至少实现：

```text id="fepx84"
extract_final_answer_span()
answer_token_ids()
sequence_logprob()
sequence_logprob_at_answer_slot()
compute_corrected_clean_direction()
paired_bootstrap_delta()
```

---

## 建议新增脚本

```text id="t5kdpt"
scripts/sprint_3C_0_fix_answer_proxy_recheck.py
```

推荐命令：

```bash id="qu3m4r"
conda run -n recover_attention python -m pytest tests/test_answer_proxy_metrics.py -q

conda run -n recover_attention python scripts/sprint_3C_0_fix_answer_proxy_recheck.py \
  --input-dir outputs/logs/sprint_3C_0_correct_wrong_activation_patching \
  --output-dir outputs/logs/sprint_3C_0_fix_answer_proxy_recheck \
  --layers 16 20 24 \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 建议新增测试

```text id="t6lbcq"
tests/test_answer_proxy_metrics.py
```

至少覆盖：

```text id="pqe50d"
1. extract_final_answer_span 优先识别 #### number；
2. fallback_last_number 会打 warning；
3. 列表序号不会被误判为答案；
4. answer_token_ids 不返回纯空格 token；
5. "42" / "-17" / "3.5" / "1,250" 可得到非空数字 token sequence；
6. sequence_logprob 使用 prefix_before_answer，不包含原 wrong answer；
7. gold/wrong sequence 在同一 prefix 下比较；
8. clean_direction = gold_delta - wrong_delta；
9. paired bootstrap 输出 mean / CI；
10. final_answer_position control 对齐到答案数字前一位。
```

---

## 最后更新

完成后必须更新：

```text id="nybe51"
PROGRESS.md
docs/progress/sprint_3_history.md
docs/progress/sprint_3_artifact_manifest.md
```

如果 outputs 被 gitignore，不提交大 jsonl，但必须在 artifact manifest 中记录：

```text id="zoaiw8"
outputs/logs/sprint_3C_0_fix_answer_proxy_recheck/
```

至少记录：

```text id="bfy52j"
- corrected_clean_direction_report.json
- corrected_control_comparison_report.json
- review_gate_answer_proxy_recheck.md
```

最终不得写：

```text id="vhq028"
accuracy improved
hallucination reduced
ready for 2000
```

最多只能写：

```text id="d9krl0"
corrected answer-sequence proxy confirms / revises the 3C-0 teacher-forced activation patching conclusion.
```
