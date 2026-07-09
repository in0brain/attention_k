# Sprint 4B: Cyber Dataset, F5 Baseline Suite, and Site-Transfer Check

## 定位

本 sprint 是新主线（`docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md`）的**第一个执行 sprint**：数据集落地 + 输出层 baseline 测量 + 因果位点迁移验证。

它不是 probe 训练 sprint（那是 4C），不是 steering sprint，不是训练任务，不是 LoRA / finetuning，不是 2000-scale rerun，也不允许声称 hallucination reduction 或 answer accuracy improvement。

本轮回答三个问题：

```text
Q1: 能否把一个有限标签 cyber 数据集落成 canonical schema，
    且标签空间是单 token 的选项字母（避免 3C-0 的 leading-token 退化）？

Q2: 输出层 baseline 套件（F5：final-logits margin / entropy / self-consistency）
    在该数据集的 wrong-vs-correct 检测上的 AUROC/AUPRC 是多少？
    —— 这个数字是 4C 五族特征 bake-off 的生死门槛。

Q3: 3C-1 的因果位点发现（答案读出位的 MLP 是 donor-specific + site-specific
    的写路径）是否迁移到 cyber MCQ 的 label-readout 位置？
    —— 不迁移则 F1 特征族动机减弱，越早知道越好。
```

默认结论标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
probe_trained=False
```

---

## 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/STORY.md（重点 §19-21）
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md（本 sprint 的上位计划）
docs/reference/CYBER_DIRECTION_PROBE_PLAN.md（已被取代，schema 部分仍引用）
docs/reference/CODE_REUSE_AUDIT_SPRINT4A.md
docs/progress/sprint_4_history.md
docs/progress/sprint_3_history.md（3C-0/3C-1 的教训）

src/recover_attention/answer_proxy_metrics.py（改造母版）
src/recover_attention/module_causal_tracing.py（Q3 直接复用）
src/recover_attention/activation_patching.py
src/recover_attention/mlp_readout_attribution.py

scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py
scripts/sprint_3C_0_fix_answer_proxy_recheck.py
```

---

## Preflight

先输出：

```text
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/preflight_report.md
```

必须检查：

```text
1. 4A / 4A-R（主线修订）的 tracked 改动是否已 commit；
   未 commit 应停止并提示，除非用户明确允许继续；
2. CYBER_HALLUCINATION_CONTROL_PLAN.md 是否存在且为当前主线；
3. STORY.md §19-21 是否存在；
4. 本轮是否不覆盖任何 sprint_3*/sprint_4A 输出目录；
5. model_path 是否按参数化规则解析：
   优先使用 `--model-path` 参数；
   其次读取环境变量 `RECOVER_ATTENTION_MODEL_PATH`；
   再其次 fallback 到 `D:/models/Qwen2.5-7B-Instruct`；
   如果三者都不可用，preflight stop，不要继续执行。
   preflight_report.md 必须记录最终使用的 `model_path_source`：
   `cli_arg | env_var | default_fallback | unavailable`；
6. 数据集是否已按下述「数据获取」就位；未就位则如实记录并停止，
   不得伪造数据；
7. 本轮仍然非训练、非 steering、非 2000、非 full 3C。
```

---

## 数据获取

优先级顺序（选 **1 个 primary**，如可行再留 1 个 held-out 给 4E，不在本轮使用）：

```text
P1: CyberMetric（MCQ，4 选项，cybersecurity 知识；GitHub/HF 公开）
P2: SecQA v1/v2（HF: zefang-liu/secqa，MCQ，安全教材来源）
P3: SecEval（HF: XuanwuAI/SecEval，MCQ）
备选 held-out（本轮不用）：MMLU computer_security 子集
```

落盘位置：

```text
data/raw/cyber/<dataset_name>/...           原始文件
data/processed/cyber/<dataset_name>.jsonl   canonical schema 转换结果
```

如果执行环境无法下载，preflight 中写明缺失并停止，输出「需要用户手动下载」
的明确指引（URL + 期望文件名），不要继续伪造后续阶段。

规模要求：

```text
primary train/dev/test 分组切分（按 source / question family 分组，防泄漏）；
本轮实际采样的 primary_questions 建议 200-400（默认 240）；
数据集本身应 >= 1000 题以支撑 4C/4E 的后续切分。
```

---

## Canonical Schema 与标签空间

沿用 4A 计划的 canonical sample schema（见 `CYBER_DIRECTION_PROBE_PLAN.md` §4），
但**标签空间强制为选项字母**：

```text
candidate_labels = ["A", "B", "C", "D"]（或数据集实际选项数）
gold_label = 选项字母
label_space = "mcq_option_letter"
```

canonical record 必须同时保留选项字母与 cyber 语义标签字段：

```json
{
  "gold_label": "B",
  "gold_label_id": "original dataset label id, if available",
  "gold_label_text": "original semantic label text",
  "candidate_labels": ["A", "B", "C", "D"],
  "candidate_choices": [
    {
      "choice": "A",
      "label_id": "original id or null",
      "label_text": "semantic option text"
    },
    {
      "choice": "B",
      "label_id": "original id or null",
      "label_text": "semantic option text"
    }
  ]
}
```

A/B/C/D 只用于 label-readout token 和 F5 scoring；semantic label id/text
必须保留，用于后续错误分析、held-out transfer、case study 和 cyber 语义分组。
不得把 Sprint 4 退化成 option-letter-only task。

硬性教训（来自 3C-0）：

```text
禁止把 ATT&CK technique id / CWE id 等多 token 字符串直接作为标签 token；
"T1059.001" 类 id 共享 leading token（"T"），会精确复现 3C-0 的
leading-token 碰撞退化。若原始数据是 id 标签，映射为选项字母再用。
```

必须验证并写入报告：

```text
每个选项字母在 Qwen2.5 tokenizer 下是单个非空白 token；
不同选项字母的 token id 两两不同；
复用/改造 answer_proxy_metrics.answer_token_ids 的空白剥离逻辑。
```

gold_label 只能作 eval label 与（4C 起的）训练目标，绝不能作 inference feature。

---

## 实验设计

### Stage 1：数据集审计与转换

```text
输出 dataset_audit_report.json：
  num_examples / label 分布 / majority-class rate /
  分组切分统计 / 泄漏检查（同题不跨 split）/
  license 与来源记录
输出 label_space_report.json：
  选项字母 tokenization 验证（单 token、互异）
输出 cyber_sample_manifest.jsonl（canonical schema）
输出 option_position_bias_report.json：
  gold choice distribution over A/B/C/D；
  greedy predicted choice distribution over A/B/C/D；
  sampled predicted choice distribution over A/B/C/D；
  majority-position baseline accuracy；
  position-only baseline accuracy；
  whether option order is fixed-seed randomized；
  whether the same semantic label is always mapped to the same option letter；
  warnings if any option letter dominates gold labels or model predictions。
```

### Stage 2：Trace 采样（冻结模型）

prompt 模板建议：

```text
{input_text}

Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Think briefly step by step, then answer with exactly one letter as:
Answer: <letter>
```

采样参数建议：

```text
primary_questions = 240（可配 200-400）
samples_per_question = 6（temperature 0.7, top_p 0.95）
外加每题 1 次 greedy（temperature 0）作为 committed answer 与 logits 来源
max_new_tokens = 256
seed 固定、逐样本递增（复用 3C-0 的 seeding 方案）
```

解析规则：

```text
优先 "Answer: <letter>"；
次选最后出现的孤立选项字母；
解析失败如实记 parse_failure，不强行归类；
输出 trace_sampling_manifest.jsonl：
  per-sample 的 completion / parsed_label / is_correct / parse_method
必报：correct_rate / wrong_rate / parse_failure_rate /
      num_questions_with_correct_and_wrong
```

### Stage 3：F5 baseline 套件（本轮核心交付）

对每题的 greedy run，在 **label-readout 位置**（"Answer: " 之后、选项字母之前
的 slot，用 domain label proxy 定位）读取 final logits，计算：

```text
f5_label_margin      = logit[top1_option] - logit[top2_option]（限制在选项 token 集内）
f5_label_entropy     = 选项 token 集合上归一化分布的熵
f5_full_entropy      = 全词表分布的熵
f5_self_consistency  = 6 个采样解析结果与 greedy 答案的一致率
f5_sc_majority_agree = 多数投票与 greedy 是否一致
```

检测任务定义：

```text
label   = greedy 答案是否错误（wrong=1, correct=0）
特征    = 上述 F5 特征（单特征各报一次 + 全体 logistic 组合报一次）
评估    = AUROC / AUPRC，按 question 分组 bootstrap CI95；
          logistic 组合用分组 CV（不训练任何激活 probe，仅输出层特征）
输出    = f5_baseline_report.json
```

这个 AUROC 数字就是 4C 生死门的门槛，必须带 CI 写入 review gate。

### Stage 4：correct/wrong pair 构造

```text
同题配对（复用 3C-0 的 build pair 逻辑）：
  至少 1 条 correct trace + 1 条 wrong trace（解析成功且非 gold）；
输出 correct_wrong_pair_manifest.jsonl；
必报 num_pairs；若 < 20 对，如实记录 insufficient_pairs，并触发 Stage 5
gate failure。默认 skip Stage 5，不要强行扩采样，也不要自动运行
available-pairs-only。
```

### Stage 5：3C-1 位点迁移验证（Q3，gated optional）

Stage 5 site-transfer check only runs if all gates pass:

```text
parse_failure_rate <= 0.10；
0.05 <= wrong_rate <= 0.95；
num_questions_with_correct_and_wrong >= 20；
label tokenization passed；
option-position bias report has no severe warning。
```

如果不满足：

```text
skip Stage 5；
write site_transfer_check_report.json with status = "skipped"；
record skipped_reason；
do not run available-pairs-only unless user explicitly sets --allow-small-site-transfer。
```

`--allow-small-site-transfer` 必须在 `site_transfer_check_report.json` 中记录。
使用该参数得到的结果只能写作 exploratory / underpowered diagnostic，不能作为
4C gate 或跨域位点迁移结论。

在 pair 子集（上限 34 对，与 3C-1 可比）上复用
`module_causal_tracing.py`，patch 目标改为 **label-readout 位置**：

```text
modules  = [attention_output, mlp_output, residual_output]
layers   = [16, 20, 24]
alpha    = [0.25, 1.0]（精简网格，控制算力）
conditions = no_patch / correct_donor / random_donor /
             same_trace_random_position / wrong_donor_self
指标     = corrected clean_direction，
           但 gold/wrong 序列换成 gold_option_letter / wrong_option_letter
           的单 token logprob（选项字母是单 token，序列退化为单步，允许）
判定     = donor-specificity 与 site-specificity 的分组 bootstrap，
           与 3C-1 的 GSM8K 数值并排呈现
输出     = site_transfer_check_report.json + module_patch_fidelity_report.json
```

---

## 严格边界

禁止：

```text
- 不要训练任何 probe（线性探针也不行，那是 4C）；
- 不要 steering / nudge（Stage 5 的诊断性 patching 除外）；
- 不要 LoRA / finetuning / 训练模型；
- 不要 2000-scale rerun；
- 不要 full Sprint 3C；
- 不要把 gold_label 用作 inference feature；
- 不要用 ATT&CK/CWE id 作标签 token；
- 不要声称 hallucination reduction / answer accuracy improvement；
- 不要覆盖任何既有 sprint 输出目录；
- 数据集不可得时不要伪造数据或结果。
```

允许：

```text
- 下载/转换公开 cyber MCQ 数据集；
- 冻结模型采样与 greedy 前向；
- 输出层特征计算与分组 bootstrap / 分组 CV（仅 F5 特征）；
- 诊断性 module activation patching（Stage 5，复用 3C-1 代码）；
- 输出 review gate；更新 PROGRESS / sprint_4_history / artifact manifest。
```

---

## 输出目录

```text
outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer/
```

必须输出：

```text
preflight_report.md
dataset_audit_report.json
label_space_report.json
option_position_bias_report.json
cyber_sample_manifest.jsonl
trace_sampling_manifest.jsonl
f5_baseline_report.json
correct_wrong_pair_manifest.jsonl
site_transfer_check_report.json（gate 未通过时 status="skipped" + skipped_reason）
module_patch_fidelity_report.json（仅 Stage 5 实际运行时必需）
high_risk_case_report.jsonl（F5 风险分最高的 20 例，含 completion 摘要）
low_risk_wrong_case_report.jsonl（F5 认为低风险但答错的 20 例——4C 的靶子）
review_gate_cyber_dataset_baseline_site_transfer.md
```

同时新增或更新：

```text
docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md（本卡）
src/recover_attention/cyber_data.py
src/recover_attention/domain_label_proxy.py
scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py
tests/test_cyber_data.py
tests/test_domain_label_proxy.py
PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

---

## 建议新增模块

### `src/recover_attention/cyber_data.py`

```text
load_raw_dataset() / to_canonical_schema()
build_mcq_prompt()
grouped_split()（按 source/question family，防泄漏）
audit_label_space()（单 token、互异验证）
write_manifest()
```

### `src/recover_attention/domain_label_proxy.py`

（改造 answer_proxy_metrics，数字答案 → 选项字母）

```text
option_token_ids()（剥离空白 token，继承 3C-0 教训）
parse_option_answer()（Answer: <letter> 优先，孤立字母 fallback，失败如实标记）
locate_label_readout_position()（"Answer: " 后、字母前的 token slot）
label_margin() / label_entropy() / full_entropy()
self_consistency_features()
classify_trace_by_option()
```

---

## 建议新增测试

`tests/test_cyber_data.py` 至少覆盖：

```text
1. canonical schema 字段齐全且类型正确；
2. 分组切分不泄漏（同 group 不跨 split）；
3. 标签分布审计正确；
4. prompt 构造包含全部选项且顺序稳定；
5. option order randomization is deterministic given seed；
6. gold option positions are auditable；
7. candidate_choices preserve semantic label text after shuffling。
```

`tests/test_domain_label_proxy.py` 至少覆盖：

```text
1. 选项字母是单个非空白 token 且两两不同（伪 Qwen tokenizer，
   复用 3C-0-Fix 测试里的 FakeQwenTokenizer 模式）；
2. "Answer: B" 解析正确；孤立字母 fallback 正确；无字母时 parse_failure；
3. label-readout 位置定位在字母 token 之前一位；
4. margin/entropy 计算正确（手工构造 logits 验证）；
5. self-consistency 特征不使用 gold_label；
6. gold_label 不出现在任何 feature 输出里（防泄漏测试）。
```

---

## Review gate

`review_gate_cyber_dataset_baseline_site_transfer.md` 必须逐条回答：

```text
1.  数据集是哪一个？license/来源是否记录？
2.  canonical schema 转换是否完成？多少题？
3.  标签空间是否为单 token 选项字母且验证互异？
4.  分组切分是否防泄漏？
5.  采样了多少题、多少 traces？parse_failure_rate 多少？
6.  correct_rate / wrong_rate 多少？模型在该数据集上是否有足够错误样本
    支撑检测任务（wrong rate 过低或过高都要如实标记）？
7.  F5 各单特征与组合的 AUROC/AUPRC（含 CI95）是多少？
8.  F5 组合 AUROC 是否已作为 4C 生死门槛写入报告？
9.  同题 correct/wrong pair 有多少对？
10. 位点迁移验证：label-readout 的 mlp_output 是否 donor-specific？
11. 是否 site-specific？与 3C-1 GSM8K 数值的对比结论是什么？
12. hook fidelity 是否通过？
13. 本轮是否零 probe 训练、零 steering？
14. gold_label 是否从未作为 inference feature？
15. 是否允许进入 4C？（仅当 Stage 1-3 完成且 pair >= 20 或已如实降级）
16. 是否允许声称 hallucination reduction / accuracy improvement？（恒 no）
17. 下一步 4C 的特征族清单与门槛数字是什么？
18. model_path 来源是什么？是否参数化而非写死？
19. 是否保留 semantic label id/text，而不只是 A/B/C/D？
20. 是否生成 option_position_bias_report.json？
21. option-position baseline 是否低于主要模型 baseline？
22. Stage 5 gate 是否通过？若未通过，是否正确 skipped？
23. 是否没有把 option-letter pattern 误写成 cyber semantic direction？
```

---

## 结果解释逻辑

```text
情况 A：位点迁移成立（mlp_output donor+site specific）
  → 4C 按五族特征全量 bake-off，F1 保留。

情况 B：位点迁移不成立
  → 4C 仍执行，但 F1 降级为对照组，主力押 F2（轨迹 transition）与
    F3（多采样一致性）；把「因果位点不跨任务迁移」如实写入 STORY。

情况 C：模型在数据集上 wrong rate < 5% 或 > 95%
  → 检测任务在该数据集上不成立；换 P2/P3 数据集重跑 Stage 2-3，
    如实记录，不要硬做。

情况 D：数据集不可得
  → preflight 停止，输出手动下载指引，本 sprint 标记 blocked。
```

---

## 推荐运行命令

```bash
conda run -n recover_attention python -m pytest tests/test_cyber_data.py tests/test_domain_label_proxy.py -q

conda run -n recover_attention python scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py \
  --dataset cybermetric \
  --primary-questions 240 \
  --samples-per-question 6 \
  --model-path <path-or-use-RECOVER_ATTENTION_MODEL_PATH> \
  --temperature 0.7 \
  --max-new-tokens 256 \
  --site-check-layers 16 20 24 \
  --site-check-alphas 0.25 1.0 \
  --site-check-max-pairs 34 \
  --output-dir outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 最后更新

完成后必须更新：

```text
PROGRESS.md（新顶部 status block）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md（记录本轮输出与 gitignore 状态）
```

最终不得写：

```text
hallucination reduced
accuracy improved
probe works
ready for 2000
```

本 sprint 最多只能声称：

```text
The cyber dataset and option-letter label space are established; the F5
output-level baseline measures X AUROC (CI) as the Sprint 4C kill bar; the
3C-1 causal site does / does not transfer to the label-readout position.
```
