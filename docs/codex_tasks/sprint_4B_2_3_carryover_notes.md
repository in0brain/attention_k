# [素材笔记] 4B-2 / 4B-3 Carryover: Prompt A/B、采样保护、F5 Cost-Tier

> 状态：**非可执行任务卡**。本文件原名
> `sprint_4B_1_prompt_fix_full_run_and_cost_tiers.md`，因 4B 被重新切分为
> 4B-1（数据/schema，见 `sprint_4B_1_cybermetric_schema_and_label_proxy.md`）
> → 4B-2（小规模模型 smoke）→ 4B-3（正式 F5 baseline），本卡已降级为
> 起草 4B-2 / 4B-3 卡时的素材笔记。三个 carryover 事项的归属见 4B-1 卡 §23：
> Stage A（prompt A/B + 退化判定）与采样保护检查 → 4B-2；
> Stage C（F5 cost-tier 双门槛）→ 4B-3；
> Stage B（240 题正式 run 的参数与要求）→ 4B-3。
> 以下原文保留作详细规格参考。

## 定位（原文）

本卡是 `sprint_4B_cyber_dataset_baseline_and_site_transfer.md` 的**执行续卡**，
不是新 sprint。4B smoke 已跑通全链路，但发现一个必须先修的问题，然后才能跑
正式 240 题 run。

Smoke 的发现：

```text
1. 链路全通：schema / 单 token 验证 / 位置偏置审计 / F5 出数 / Stage 5 gate 正确跳过；
2. wrong_rate 0.315，检测任务 regime 健康；
3. 但 8/90 条 parse failure 全部是「生成退化」而非格式问题：
   无限重复（"!!!!!..."）、中英混杂胡语、随机多语言 token 拼贴；
   greedy 也有 2/30 退化——这是 Instruct 模型在无 chat template 的
   裸补全 prompt 下的典型症状；
4. 退化与题目难度可能相关，直接排除会给检测评估带偏。
```

本卡三个任务，按序执行：

```text
Stage A: prompt A/B 小测（裸补全 vs chat template），选定正式 run 的 prompt 方案；
Stage B: 用选定方案跑正式 240 题 run（完整执行 4B 卡 Stage 1-5）；
Stage C: F5 报告增加 cost-tier 拆分（single-forward 档 vs sampling 档）。
```

边界与默认标记完全沿用 4B 卡：零 probe 训练、零 steering（Stage 5 诊断性
patching 除外）、零训练、非 2000、gold_label 绝不作 inference feature、
不声称 hallucination reduction / accuracy improvement。

---

## Stage A：Prompt A/B 小测

### 设置

```text
题目：从 canonical manifest 取前 20 题（固定 seed，与 smoke 同源）；
两个条件：
  cond_raw:  现行裸补全 prompt（4B 卡模板原样）；
  cond_chat: tokenizer.apply_chat_template 包装，
             system = "You are a cybersecurity expert. Answer the
                       multiple-choice question."
             user   = 4B 卡模板的正文（含 Options 与
                      "Think briefly step by step, then answer with
                       exactly one letter as:\nAnswer: <letter>"）；
             add_generation_prompt=True；
每条件每题：3 个采样（temperature 0.7, top_p 0.95）+ 1 次 greedy；
max_new_tokens = 256（不是 smoke 的 128）。
```

### 采样保护（必须先检查）

```text
确认 generate 调用带有 renormalize_logits=True 和 remove_invalid_values=True
（3C-0 的方案，见 scripts/sprint_3C_0_correct_wrong_activation_patching.py
 的 sample_traces）；缺失则补上，并在报告中记录修复。
```

### 退化判定（写成可复用函数，进 domain_label_proxy 或新 util）

```text
degenerate = 满足任一：
  1. 任意单字符连续重复 >= 30（如 "!!!!..."）；
  2. 任意长度 >= 6 的子串在 completion 内连续重复 >= 5 次；
  3. parse_failure 且 completion 长度 >= 0.9 * max_new_tokens 对应的
     字符规模（截断式失败）；
判定逻辑必须有单元测试（用 smoke 的真实退化样例做正例）。
```

### 输出与决策

```text
输出 outputs/logs/sprint_4B_1_prompt_ab_test/prompt_ab_report.json：
  per-condition 的 parse_failure_rate / degeneration_rate / correct_rate /
  greedy_parse_failure_count / 每题 token 用量均值；
决策规则：
  1. 选 parse_failure_rate + degeneration_rate 之和更低的条件；
  2. 差距 < 0.02 视为平手，平手选 cond_chat（Instruct 模型的正规用法）；
  3. 把 chosen_condition 与理由写入报告；
若 cond_chat 胜出，必须验证 locate_label_readout_position 在
chat-template 全文（prompt+completion 拼接）上仍定位正确，补一个单元测试。
```

Stage A 不通过（两个条件退化率都 > 0.15）时：如实记录，停止并报告，
不要硬跑 Stage B。

---

## Stage B：正式 240 题 run

用 Stage A 选定的 prompt 条件，完整执行 4B 卡的 Stage 1-5：

```bash
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py \
  --dataset cybermetric \
  --primary-questions 240 \
  --samples-per-question 6 \
  --temperature 0.7 \
  --max-new-tokens 256 \
  --prompt-style <chosen: raw|chat> \
  --site-check-layers 16 20 24 \
  --site-check-alphas 0.25 1.0 \
  --site-check-max-pairs 34 \
  --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer \
  --overwrite
```

要求：

```text
1. 脚本新增 --prompt-style 参数（raw | chat），默认取 Stage A 的胜者；
2. 输出目录用正式目录，不要覆盖 *_smoke 目录（smoke 结果保留作对照）；
3. trace manifest 里逐条记录 degenerate 标志（用 Stage A 的判定函数）；
4. 按 wrong_rate ~0.3 估计，同题 correct/wrong pair 应远超 20 →
   Stage 5 位点迁移验证应当解锁执行；若意外 < 20，按 4B 卡规则如实降级；
5. Stage 5 结果必须与 3C-1 的 GSM8K 数值并排呈现
   （donor-specificity / site-specificity 的 mean + CI95 对照表）。
```

预计时长 3-4 小时（1680 次生成 + patching 前向），建议后台运行、
每 50 题打印进度。

---

## Stage C：F5 cost-tier 拆分

修改 F5 报告构造（对正式 run 生效）：

```text
tier_single_forward（一次 greedy 前向可得）：
  f5_label_margin / f5_label_entropy / f5_full_entropy
  + 这三者的固定组合（不训练，等权 z-score 求和即可，注明公式）

tier_sampling（需额外 6 次采样）：
  f5_self_consistency / f5_sc_majority_agree
  + 全特征组合

每个 tier 单独报 AUROC / AUPRC + 按题分组 bootstrap CI95；
f5_baseline_report.json 顶层加：
  kill_bar_single_forward = tier_single_forward 组合的 AUROC (含 CI)
  kill_bar_sampling       = tier_sampling 全组合的 AUROC (含 CI)
```

Review gate 必须写明双门槛的用法：

```text
4C 的 single-forward 内部特征（F1 静态位点 / F4 J-lens 标签投影）
的生死门是 kill_bar_single_forward（同成本公平对照）；
kill_bar_sampling 是 aspirational 上界（贵 7 倍推理的对手）；
F3（多采样一致性）属于 sampling 档，对照 kill_bar_sampling。
```

---

## 输出清单

```text
outputs/logs/sprint_4B_1_prompt_ab_test/prompt_ab_report.json
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/
  （4B 卡全部必需输出，f5_baseline_report.json 含双 tier，
   site_transfer_check_report.json 含 3C-1 对照表）
```

代码与测试：

```text
scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py（--prompt-style + tier 拆分）
src/recover_attention/domain_label_proxy.py（退化判定 + chat 模板定位兼容）
tests/test_domain_label_proxy.py（新增：退化判定用 smoke 真实样例、
  chat 模板下 label-readout 定位、tier 组合公式）
```

完成后更新：

```text
PROGRESS.md（新顶部 status block，含双 kill bar 数字与 Stage 5 结论）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

---

## Review gate 追加问题

在 4B 卡 review gate 的 17 问之外追加：

```text
18. Stage A 选了哪个 prompt 条件？两条件的 parse_failure / degeneration 各多少？
19. 正式 run 的 degeneration_rate 是多少？相比 smoke（~0.09）是否改善？
20. kill_bar_single_forward 与 kill_bar_sampling 各是多少（含 CI）？
21. Stage 5 是否解锁执行？label-readout 的 mlp_output 与 3C-1 GSM8K
    对照的结论是什么（迁移 / 部分迁移 / 不迁移）？
22. 情况 A/B（4B 卡结果解释逻辑）判定为哪种？对 4C 特征族优先级的影响？
```

---

## 推荐命令序列

```bash
conda run -n recover_attention python -m pytest tests/test_domain_label_proxy.py tests/test_cyber_data.py -q

# Stage A
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py \
  --dataset cybermetric --primary-questions 20 --samples-per-question 3 \
  --temperature 0.7 --max-new-tokens 256 \
  --ab-test-prompt-styles raw chat \
  --output-dir outputs/logs/sprint_4B_1_prompt_ab_test --overwrite

# Stage B（--prompt-style 填 Stage A 胜者）
conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py \
  --dataset cybermetric --primary-questions 240 --samples-per-question 6 \
  --temperature 0.7 --max-new-tokens 256 --prompt-style chat \
  --site-check-layers 16 20 24 --site-check-alphas 0.25 1.0 --site-check-max-pairs 34 \
  --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer --overwrite

conda run -n recover_attention python -m pytest -q
```

（Stage A 的 `--ab-test-prompt-styles` 如实现为独立子命令/脚本也可以，
保持输出 schema 一致即可。）

---

## 最终声明边界

沿用 4B 卡。本卡最多只能声称：

```text
The chosen prompt style reduces degeneration to X; the full-run F5 baseline
establishes kill_bar_single_forward = A (CI) and kill_bar_sampling = B (CI);
the 3C-1 causal site does / does not transfer to the cyber label-readout
position.
```
