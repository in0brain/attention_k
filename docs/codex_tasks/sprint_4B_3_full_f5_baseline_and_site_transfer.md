# Sprint 4B-3：Full F5 Baseline（双 Kill Bar）与 3C-1 位点迁移验证

## 1. 定位

本 sprint 是 4B 母卡（`sprint_4B_cyber_dataset_baseline_and_site_transfer.md`）的
**收尾 sprint**：用 4B-2 选定的 prompt 方案跑正式 240 题 run，产出 4C 特征
bake-off 的两个生死门数字，并回答母卡 Q3（3C-1 因果位点是否迁移到 cyber
label-readout 位置）。本轮完成后，4B 母卡关闭，下一阶段是 4C。

本轮回答三个问题：

```text
Q1: kill_bar_single_forward 与 kill_bar_sampling 各是多少（含分组 bootstrap CI95）？
    —— 4C 的 F1/F4（single-forward 内部特征）对照前者；
       F3（多采样一致性）对照后者。

Q2: 3C-1 的位点发现（答案读出位的 MLP 是 donor-specific + site-specific 写路径）
    是否迁移到 cyber MCQ 的 label-readout 位置？

Q3: 胜出 prompt 条件下模型是否产生推理文本（reasoning substrate）？
    —— 4B-2 dry run 观察到 chat 条件常直接回裸字母、无 CoT；
       这不影响 F5，但决定 4C 的 F2（轨迹 transition 特征）有没有原料。
```

本轮不是 probe 训练（4C）、不是 steering、不是训练/LoRA、不是 2000-scale。
Stage E 的诊断性 module patching 是唯一的激活干预，明确豁免。

默认结论标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
probe_trained=False
```

注意措辞：kill bar 是「测得」（measured），不是「通过」。F5 baseline 数字本身
不构成任何 detector 有效性声明。

---

## 2. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md
docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md（母卡）
docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md（§23 carryover）
docs/codex_tasks/sprint_4B_2_small_model_smoke_and_f5_plumbing.md
docs/codex_tasks/sprint_4B_2_3_carryover_notes.md（双门槛与退化判定的规格来源）
docs/progress/sprint_4_history.md
docs/progress/sprint_3_history.md（3C-0 tokenization 教训 / 3C-1 位点数字）

src/recover_attention/domain_label_proxy.py（tokenization 与解析的唯一权威实现）
src/recover_attention/cyber_data.py
src/recover_attention/module_causal_tracing.py（Stage E 直接复用）
src/recover_attention/mlp_readout_attribution.py（AUROC/分组 bootstrap 复用）

scripts/sprint_4B_2_small_model_smoke.py（生成/读出/双形态解析的直接母版）
scripts/sprint_3C_0_correct_wrong_activation_patching.py（model.generate 采样保护参考）
scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py（Stage E 对照参考）

tests/test_domain_label_proxy.py（必须理解既有测试约定后再动代码）
```

---

## 3. Tokenization 与解析纪律（硬性条款，违反即返工）

本节是 3C-0 与 4B-2 两次 instrumentation bug 的固化，**必须复用现有实现，
禁止重新发明**。

### 3.1 选项字母 token 解析

```text
禁止：tokenizer.encode(" A")[0] 直接当字母 token —— Qwen tokenizer 下
      " A" 可能切出独立 whitespace token，ids[0] 是空白不是字母
      （3C-0 的 leading-token 退化根因）。

必须：只通过 domain_label_proxy 的两个函数解析选项 token：
  option_token_ids(tokenizer, labels)       —— 空格前缀形态（" A"），
      内部已做 whitespace-marker 剥离、单 token 校验、两两互异校验；
  bare_option_token_ids(tokenizer, labels)  —— 裸字母形态（"A"），
      同样校验（4B-2 发现 chat 条件模型直接回裸字母，
      其 token id 与空格形态不同）。

必须：F5 读出与位置断言使用 dual-form 解析（4B-2 已实现的模式）：
  读出位 +1 的实际 token 与两种形态逐一比对，命中哪种用哪种形态的
  token 集合计算 margin/entropy；两种都不命中记 "unknown" 并计入
  assertion failure，不得静默回退。
  报告必须含 token_form_counts（space/bare/unknown 分布）。

必须：label_space_report 复核 A/B/C/D 在两种形态下均为
  单个非空白 token 且 token id 两两互异，写入输出。
```

### 3.2 答案解析

```text
必须：只通过 domain_label_proxy.parse_option_answer 解析，优先级固定为：
  1. 显式 "Answer: <letter>"（大小写兼容）；
  2. 最后出现的孤立有效选项字母（后缀仅剩标点/空白时才算孤立）；
  3. parse_failure —— 如实记录，禁止强制归类。

禁止：把包含 A/B/C/D 的普通单词（AES、BERT、CVE、DNS、"Class A network"
      等 cyber 高频术语）误解析成答案。既有负例测试
      （test_parse_does_not_treat_ordinary_words_as_answers）必须保持通过；
      若本轮发现新的误解析样式，先补负例测试再改 parser。

禁止：parse_failure 的 trace 计入 correct/wrong 任一类；它是独立类别，
      进 parse_failure_rate。
```

### 3.3 退化判定

```text
必须：复用 domain_label_proxy.detect_degeneration（三规则 + 真实样例
  fixture 测试已在位），逐 trace 记录 degenerate 与命中规则；
若 4B-2 以 0.08 < score <= 0.15 降级准入，退化 trace 必须作为独立类别
  统计并在所有 rate 报告中单列，不得静默剔除。
```

---

## 4. Preflight

先输出：

```text
outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/preflight_report.md
```

必须检查并记录：

```text
1. 4B-1 与 4B-2 的 tracked 改动是否已 commit（工作区干净）；
   未 commit 应停止并提示，除非用户明确允许继续；
2. 4B-2 正式 run 输出是否存在且完整：
   outputs/logs/sprint_4B_2_small_model_smoke/prompt_ab_report.json
   outputs/logs/sprint_4B_2_small_model_smoke/review_gate_small_model_smoke.md
   缺失则停止，提示先完成 4B-2（其脚本已就绪）；
3. 从 prompt_ab_report.json 读取 decision.winner 与胜者 score：
   - winner 为 null（双条件超停止阈值）→ 停止，本卡 blocked；
   - score <= 0.08 → 正常准入；
   - 0.08 < score <= 0.15 → 降级准入，激活 §3.3 的退化独立类别条款；
   - preflight 报告必须记录 chosen_prompt_style 与准入等级；
4. data/processed/cyber/cybermetric.jsonl 存在且随机 20 条通过
   validate_cyber_sample_record；
5. model_path 解析：--model-path > 环境变量 RECOVER_ATTENTION_MODEL_PATH >
   fallback D:/models/Qwen2.5-7B-Instruct；三者不可用则停止；
   记录 model_path_source；
6. 本轮不覆盖任何既有输出目录（sprint_3* / sprint_4A* / sprint_4B_1* /
   sprint_4B_2* / sprint_4B_dataset_download_audit /
   sprint_4B_cyber_dataset_baseline_and_site_transfer*）；
7. 预计生成量与时间（240 × (6+1) = 1680 次生成）；GPU 可用性。
```

---

## 5. 本轮允许修改的文件

```text
docs/codex_tasks/sprint_4B_3_full_f5_baseline_and_site_transfer.md（本卡）

scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py（新脚本）
src/recover_attention/domain_label_proxy.py（仅允许：新增函数或补负例修复，
  不得改变既有函数签名与既有测试语义）
src/recover_attention/cyber_data.py（同上约束）

tests/test_domain_label_proxy.py
tests/test_cyber_data.py

PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

禁止修改：

```text
schemas.py 既有 record 规则
module_causal_tracing.py / activation_patching.py / answer_proxy_metrics.py
mlp_readout_*.py / approx_j_lens_readout.py
任何 sprint_3* / sprint_4A* / sprint_4B_1* / sprint_4B_2* 脚本
docs/reference/*
```

---

## 6. Stage A：生成路径性能修正（等价性抽查后才准全量跑）

4B-2 的手写 CPU 采样循环无 KV cache（`use_cache=False` 逐 token 全量重前向），
32 题可忍，1680 次生成不可行（raw 条件长推理下可达十几小时）。

```text
必须：改用带 KV cache 的生成路径。推荐 3C-0 验证过的方案：
  model.generate(..., do_sample=True, renormalize_logits=True,
                 remove_invalid_values=True, use_cache=True,
                 pad_token_id=eos, eos_token_id=eos)
  greedy 用 do_sample=False 同路径。
  （4B-2 手写循环的动机是规避 CUDA multinomial assert；
   renormalize+remove_invalid 是 3C-0 对同一问题的等效保护，
   600 次生成实测无事故。）

必须：等价性抽查——从 4B-2 的 32 题中取 5 题，用新路径重跑 greedy，
  对比 4B-2 trace manifest 中的 parsed_label：
  - greedy 解析结果 5/5 一致 → 通过；
  - 不一致 → 停止并报告差异，不得带病全量跑。
  （采样路径因随机数实现不同不要求逐 token 一致，只要求解析成功率
   与退化率在合理范围。）

必须：早停策略——生成中按 parse_option_answer 周期性检查（如每 16 token）
  或依赖 EOS；不得为省事去掉 max_new_tokens 上限。
```

---

## 7. Stage B：240 题正式采样

```text
题目：cyber_sample_smoke_manifest.jsonl 的全部 240 条（train split，
  4B-1 已做 split-aware 抽取；不足 240 则如实报告实际数）；
prompt：4B-2 胜出条件（从 prompt_ab_report.json 读取；chat 则用
  cyber_data.build_mcq_chat_messages + apply_chat_template，
  raw 则用 build_mcq_prompt——复用 4B-2 的 build_prompt 逻辑）；
参数：每题 6 采样（temperature 0.7, top_p 0.95）+ 1 greedy，
  max_new_tokens 256，seed 固定逐样本递增；
逐 trace 记录：parsed_label / parse_method / parse_failure / is_correct /
  degenerate / degeneration_rules / num_generated_tokens；
必报：correct_rate / wrong_rate / parse_failure_rate / degeneration_rate /
  num_questions_with_correct_and_wrong。
```

### Reasoning substrate 诊断（Q3，必做）

已知前情（4B-2 正式 32 题 run 已测得）：

```text
chat 条件 128/128 条 completion 全部是单 token 裸字母（p50 token 数 = 1），
has_reasoning_text = 0.00 —— 模型无视 "Think briefly step by step"，
直接输出答案字母。F5/F1/F4/F3 不受影响（读出位 logits 与多采样一致性
仍然良定义），但 F2（轨迹 transition 特征）在 chat 条件下完全没有原料。
```

本轮要求：

```text
逐 trace 记录 completion 字符数与 token 数；
定义 has_reasoning_text = 答案字母之前存在 >= 20 个非空白字符的文本；
输出 reasoning_substrate_report.json（240 题规模复核上述发现）；

另加一个 gated 小实验（Stage B'，可选但强烈建议，约 16 题 × 2 生成）：
  测试一个 reasoning-forcing chat 变体——system/user 指令改为要求
  先输出 "Reasoning:" 段再输出 "Answer: <letter>"（格式显式给出）；
  测量该变体的 has_reasoning_text 比例、parse_failure、degeneration；
  若它能恢复 substrate 且保持干净（score <= 0.08），则 4C 的 F2 有了
  第三条路：用该变体专门采 F2 的 trace（F5 kill bar 仍以本轮主条件为准，
  两者不混）。

review gate 必须给出 4C F2 的明确方案，三选一：
  a. 用 raw 条件 trace 做 F2（有推理文本但带 ~9% 退化污染，需单列）；
  b. 用 Stage B' 的 reasoning-forcing 变体（若小实验通过）；
  c. F2 降级/放弃，4C 主力押 F3（多采样一致性）+ F1/F4（位点读出）。
不得留白。
```

---

## 8. Stage C：F5 双 Kill Bar（本轮核心交付）

对每题 greedy run，在 label-readout 位置（用
`locate_label_readout_position` 定位 + §3.1 dual-form 断言）读取 final
logits，计算并按 cost tier 分组：

```text
tier_single_forward（一次 greedy 前向可得）：
  f5_label_margin / f5_label_entropy / f5_full_entropy
  + 固定组合（等权 z-score 求和，公式写入报告，不训练）

tier_sampling（需额外 6 次采样）：
  f5_self_consistency / f5_sc_majority_agree
  + 全特征固定组合
```

检测任务与评估：

```text
label = greedy 答案是否错误（wrong=1, correct=0；parse_failure 与
        degenerate 单列，不进二分类）
每个单特征与两个组合各报 AUROC / AUPRC；
按 question 分组 bootstrap CI95（>=300 次重采样）；
kill_bar_single_forward = tier_single_forward 组合的 AUROC（含 CI）
kill_bar_sampling       = 全特征组合的 AUROC（含 CI）
输出 f5_baseline_report.json，双 bar 置顶层字段；
每行 feature record 过 assert_no_gold_label_leakage。
```

同时输出 4C 靶子与偏置审计：

```text
high_risk_case_report.jsonl（组合风险分最高 20 例）
low_risk_wrong_case_report.jsonl（低风险但答错 20 例——4C 增量价值的靶子）
option_position_bias_report.json（gold / greedy / sampled 的 A-D 分布，
  majority-position 与 position-only baseline accuracy，>0.40 警告）
```

---

## 9. Stage D：correct/wrong pair 构造

```text
同题配对：>=1 条 correct 采样 + >=1 条 wrong 采样（解析成功、非退化）；
输出 correct_wrong_pair_manifest.jsonl；
num_pairs < 20 → 如实记录 insufficient_pairs，Stage E 按 gate 规则跳过，
  不得强行扩采样。
```

---

## 10. Stage E：3C-1 位点迁移验证（gated）

Gate（全部满足才执行）：

```text
num_pairs >= 20；parse_failure_rate <= 0.10；
0.05 <= wrong_rate <= 0.95；label tokenization 校验通过；
无严重 position-bias 警告（或已如实记录并解释后由报告显式放行）。
```

执行（复用 `module_causal_tracing`，上限 34 对与 3C-1 可比）：

```text
patch 目标 = wrong trace 的 label-readout 位置（donor = 同题 correct trace
  同角色位置）；
modules = [attention_output, mlp_output, residual_output]；
layers  = [16, 20, 24]；alpha = [0.25, 1.0]（精简网格）；
conditions = no_patch / correct_donor / random_donor /
             same_trace_random_position / wrong_donor_self；
指标 = gold 选项字母 logprob delta - wrong 选项字母 logprob delta
  （单 token 标签使序列 logprob 退化为单步，报告注明；
   token id 按该 trace 实际 token form 取——§3.1）；
判定 = donor-specificity 与 site-specificity 的分组 bootstrap CI95。
```

Chat 适配注意（必须写入报告）：

```text
裸字母完成时 label-readout 位实际落在 chat 脚手架末端（assistant 起始），
patching 语义仍良定义，但必须报告读出位到 prompt 末 token 的距离分布；
same_trace_random_position 控制在裸字母完成下可能无可选位置
（completion 只有 1 个 token）——此时该 condition 对该 pair 跳过并计数，
不得用 prompt 内位置冒充。
```

结果呈现：

```text
site_transfer_check_report.json 必须含与 3C-1 GSM8K 的并排对照表：
  GSM8K（3C-1）: mlp donor +0.141 / site +0.353（均稳定为正）、
                attention donor +0.093 / site -0.006、
                residual donor +1.703 / site -0.547
  vs cyber 本轮各 cell 的 mean + CI95；
结论三选一：迁移 / 部分迁移 / 不迁移，并说明对 4C F1 优先级的影响
（不迁移 → F1 降级为对照组，主力押 F2/F3——见 4B 母卡结果解释逻辑情况 B）。
module_patch_fidelity_report.json：hook 注册/触发/移除率、
  wrong_donor_self 的 floor 量级（参照 3C-1 的诚实记账方式）。
```

---

## 11. 输出清单

```text
outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/
  preflight_report.md
  trace_sampling_manifest.jsonl
  reasoning_substrate_report.json
  f5_baseline_report.json（含双 kill bar + token_form_counts）
  option_position_bias_report.json
  high_risk_case_report.jsonl
  low_risk_wrong_case_report.jsonl
  correct_wrong_pair_manifest.jsonl
  site_transfer_check_report.json（含 3C-1 对照表，或 gate-skip 记录）
  module_patch_fidelity_report.json（Stage E 执行时）
  equivalence_spot_check_report.json（Stage A 的 5 题抽查）
  review_gate_full_f5_baseline_and_site_transfer.md
```

同时更新：

```text
PROGRESS.md（新顶部 status block，含双 kill bar 数字与位点迁移结论）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

---

## 12. 测试要求

```text
1. 新脚本的纯函数部分（双 bar 组合公式、reasoning substrate 判定、
   等价性抽查比对逻辑）有单元测试；
2. §3 的既有测试全部保持通过（option_token_ids 系 / bare 系 /
   parse 负例系 / detect_degeneration 系 / leakage 系）；
3. 若为 Stage E 新增任何位置对齐辅助函数，补 chat 脚手架样式的
   定位测试（参照 test_label_readout_position_works_on_chat_template_style_full_text）；
4. full pytest 通过。
```

---

## 13. Review Gate

`review_gate_full_f5_baseline_and_site_transfer.md` 逐条回答
（母卡 17 问的对应项 + §23 追加项 + 本卡新增项）：

```text
1.  数据集与题量？prompt 条件（从 4B-2 读取的 winner 与准入等级）？
2.  Stage A 等价性抽查结果（5/5?）？新生成路径相对 4B-2 的加速比？
3.  采样了多少 traces？correct/wrong/parse_failure/degeneration 各占多少？
4.  退化 trace 是否按独立类别统计（若降级准入）？
5.  token_form_counts 分布？dual-form 断言失败计数？
6.  kill_bar_single_forward = ?（AUROC + CI95）
7.  kill_bar_sampling = ?（AUROC + CI95）
8.  单特征中最强的是哪个？
9.  低风险答错案例有多少（4C 靶子）？
10. reasoning substrate：has_reasoning_text 比例？对 4C F2 的判定与建议？
11. position bias 是否有严重警告？
12. num_pairs？Stage E gate 是否通过？
13. 位点迁移结论（迁移/部分/不迁移）？与 3C-1 对照表？
14. hook fidelity 是否通过？wrong_donor_self floor 量级？
15. 本轮零 probe 训练、零 steering？（必须 yes）
16. gold_label 是否从未作为 inference feature？（必须 yes）
17. 是否允许进入 4C？4C 的特征族清单、各自对照的门槛数字、
    F1 优先级（依位点迁移结论）与 F2 substrate 方案？
18. 是否声称 hallucination reduction / accuracy improvement？（必须 no）
```

---

## 14. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_domain_label_proxy.py tests/test_cyber_data.py -q

conda run -n recover_attention python scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py \
  --processed-path data/processed/cyber/cybermetric.jsonl \
  --smoke-manifest outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/cyber_sample_smoke_manifest.jsonl \
  --ab-report outputs/logs/sprint_4B_2_small_model_smoke/prompt_ab_report.json \
  --samples-per-question 6 \
  --temperature 0.7 \
  --max-new-tokens 256 \
  --site-check-layers 16 20 24 \
  --site-check-alphas 0.25 1.0 \
  --site-check-max-pairs 34 \
  --output-dir outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

预计时长：胜出条件若为 chat（完成通常 1-5 token），生成阶段以 prompt 前向
为主，约 1-2 小时；Stage E 约 30-60 分钟。建议后台运行、每 20 题打印进度。

---

## 15. 最多允许的结论

```text
The dual F5 kill bars are measured as X (single-forward) and Y (sampling)
with CIs on the 240-question CyberMetric run; the 3C-1 causal site does /
partially does / does not transfer to the cyber label-readout position;
the reasoning-substrate finding for 4C F2 is Z. Sprint 4B is closed;
Sprint 4C may open against these bars.
```

不得写：

```text
detector works / F5 is effective as a detector
hallucination reduced / accuracy improved
probe works / ready for 2000
```
