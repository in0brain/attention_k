# Sprint 4B-2：CyberMetric Small-Model Smoke 与 F5 Feature Plumbing

## 1. 定位

本 sprint 是 Sprint 4B 的第二个工程子阶段：**首次在 canonical 数据上调用模型**，
但只做小规模 smoke（32 题），核心目标是把 prompt 方案定下来、把 F5 特征管道
打通并验证，为 4B-3 正式 240 题 run 扫清全部工程障碍。

本轮主线为：

```text
canonical cyber samples (train split, 32 题)
→ prompt A/B（裸补全 vs chat template，同题双条件）
→ 退化判定 + 决策
→ 胜出条件下的 F5 特征管道验证
→ option-position model bias 审计
→ 4B-3 准入判定
```

本轮不是完整 Sprint 4B，不进行：

```text
240 × 7 正式实验（4B-3）
F5 kill bar 声明（4B-3）
correct/wrong pair 大规模构造（4B-3）
site-transfer 位点迁移验证（4B-3 Stage 5）
probe 训练（4C）
steering / patching / nudge
LoRA / finetuning
2000-scale
```

默认结论标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=True
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
probe_trained=False
f5_kill_bar_established=False
```

---

## 2. 本轮需要回答的问题

```text
Q1: 裸补全 prompt 与 chat template prompt，哪个的
    parse_failure_rate + degeneration_rate 更低？
    —— 背景：4B smoke 发现 8/90 条 parse failure 全部是生成退化
    （无限重复 "!!!!!..."、中英混杂胡语、随机多语言拼贴），
    greedy 也有 2/30 退化，是 Instruct 模型裸补全的典型症状。

Q2: 退化判定函数能否可靠识别 smoke 中出现过的真实退化样例？

Q3: F5 特征管道（label-readout slot 定位 → 选项 logits 读取 →
    margin/entropy/self-consistency → 特征行 schema + 防泄漏检查）
    在胜出 prompt 条件下是否端到端正确？

Q4: 模型的预测是否存在 option-position bias
    （greedy 预测的 A/B/C/D 分布 vs gold 分布）？

Q5: 采样调用是否带 renormalize_logits / remove_invalid_values 保护？
```

---

## 3. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md
docs/codex_tasks/sprint_4B_cyber_dataset_baseline_and_site_transfer.md（母卡）
docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md（§23 carryover）
docs/codex_tasks/sprint_4B_2_3_carryover_notes.md（A/B 与退化判定的详细规格来源）
docs/progress/sprint_4_history.md

src/recover_attention/cyber_data.py
src/recover_attention/domain_label_proxy.py
src/recover_attention/schemas.py
src/recover_attention/attention_bias_steering.py（load_local_steering_backend 复用）
scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py（生成/读 logits 辅助函数复用母版）
scripts/sprint_3C_0_correct_wrong_activation_patching.py（采样保护参考：sample_traces）
tests/test_domain_label_proxy.py
```

---

## 4. Preflight

先输出：

```text
outputs/logs/sprint_4B_2_small_model_smoke/preflight_report.md
```

必须检查并记录：

```text
1. 4B-1 的 tracked 改动是否已 commit（工作区应干净）；
   未 commit 应停止并提示，除非用户明确允许继续；
2. data/processed/cyber/cybermetric.jsonl 是否存在且通过
   validate_cyber_sample_record 抽查（随机 20 条）；
3. cyber_sample_smoke_manifest.jsonl（240 条）是否存在；
4. model_path 解析：--model-path 参数 > 环境变量
   RECOVER_ATTENTION_MODEL_PATH > fallback D:/models/Qwen2.5-7B-Instruct；
   三者皆不可用则停止；记录 model_path_source；
5. 本轮不覆盖任何既有输出目录（sprint_3* / sprint_4A* / sprint_4B_1* /
   sprint_4B_cyber_dataset_baseline_and_site_transfer*）；
6. 本轮零 probe 训练、零 steering、零 patching、非 2000、非 full 3C；
7. GPU 可用性与预计生成量（32 题 × 2 条件 × 4 次 ≈ 256 次生成）。
```

---

## 5. 本轮允许修改的文件

```text
docs/codex_tasks/sprint_4B_2_small_model_smoke_and_f5_plumbing.md（本卡）

src/recover_attention/domain_label_proxy.py（新增退化判定与 chat 定位兼容）
src/recover_attention/cyber_data.py（如需新增 chat prompt builder）

scripts/sprint_4B_2_small_model_smoke.py（新脚本）
scripts/sprint_4B_cyber_dataset_baseline_and_site_transfer.py
（仅允许：把可复用的生成/logits 辅助函数抽到公共位置，不得改变其行为）

tests/test_domain_label_proxy.py
tests/test_cyber_data.py

PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

禁止修改：

```text
schemas.py 的既有 record 定义（可新增，不可改动已有字段规则）
answer_proxy_metrics.py / module_causal_tracing.py / activation_patching.py
mlp_readout_*.py / approx_j_lens_readout.py
任何 sprint_3* / sprint_4A* / sprint_4B_1* 脚本
docs/reference/*
```

---

## 6. Stage A：Prompt A/B（本轮核心）

### 6.1 题目选取

```text
从 cyber_sample_smoke_manifest.jsonl 的 train split 中，
固定 seed（默认 4242）抽取 32 题；
不得取前 32 条；
记录所抽 example_id 清单到输出。
```

### 6.2 两个条件

```text
cond_raw:  cyber_data.build_mcq_prompt 的现行裸补全模板（原样）。

cond_chat: tokenizer.apply_chat_template 包装：
  system = "You are a cybersecurity expert. Answer the
            multiple-choice question."
  user   = build_mcq_prompt 的正文（含 Options 与
           "Think briefly step by step, then answer with
            exactly one letter as:\nAnswer: <letter>"）
  add_generation_prompt=True
```

### 6.3 生成参数（两条件完全一致）

```text
每题：1 次 greedy（temperature 0）+ 3 次采样（temperature 0.7, top_p 0.95）
max_new_tokens = 256
seed 固定、逐样本递增（沿用 3C-0 seeding 方案）
采样必须带 renormalize_logits=True 与 remove_invalid_values=True；
若现行辅助函数缺失该保护，先补上并在报告中记录（Q5）。
```

### 6.4 退化判定（新增纯函数，必须先写测试）

在 `domain_label_proxy.py` 新增：

```python
detect_degeneration(completion: str, *, max_new_tokens: int) -> dict
```

判定规则（满足任一即 degenerate=True，返回命中的规则名列表）：

```text
1. single_char_run:   任意单字符连续重复 >= 30；
2. substring_loop:    任意长度 >= 6 的子串在文本内连续重复 >= 5 次；
3. truncated_failure: 无法解析出选项字母，且 completion 字符数 >=
                      0.9 * (max_new_tokens * 3)（近似 token→char 比例，
                      记录所用比例常数）。
```

测试要求：

```text
必须把 4B smoke 的真实退化样例作为正例 fixture
（"!!!!!..." 重复、"Answer盼星星盼月亮" 混杂、随机多语言拼贴——
 从 outputs/logs/sprint_4B_cyber_dataset_baseline_and_site_transfer_smoke/
 trace_sampling_manifest.jsonl 中提取真实文本片段固化进测试）；
正常推理文本（含较长但不退化的推理）必须判负；
纯函数、不依赖模型。
```

### 6.5 决策规则

```text
score(cond) = parse_failure_rate + degeneration_rate（按 trace 计）
1. score 低者胜；
2. |score 差| < 0.02 视为平手，平手选 cond_chat（Instruct 正规用法）；
3. 两条件 score 均 > 0.15：停止，如实报告，不判定 4B-3 准入，
   在 review gate 记录「prompt 方案未解决退化问题」。
```

### 6.6 chat 条件的定位兼容（cond_chat 胜出时必做）

```text
locate_label_readout_position 必须在 chat-template 全文
（渲染后的 prompt 字符串 + completion）上定位正确；
新增单元测试：构造含 chat 特殊标记（如 <|im_start|> 风格前缀）的
prompt+completion，验证定位仍指向答案字母前一位；
运行时断言：定位位置 +1 处的 token id == 解析出的选项字母 token id，
每条 trace 都检查，失败计数写入报告。
```

---

## 7. Stage B：F5 特征管道验证（胜出条件）

对胜出条件的 32 题 greedy run：

```text
1. 在 label-readout slot（"Answer: " 之后、字母之前）读取 final logits；
2. 计算 f5_label_margin / f5_label_entropy / f5_full_entropy；
3. 用 3 个采样结果计算 f5_self_consistency / f5_sc_majority_agree；
4. 每题产出一行 feature record，并标注 cost tier 字段：
   tier_single_forward = {margin, label_entropy, full_entropy}
   tier_sampling       = {self_consistency, sc_majority_agree}
   （这是 4B-3 双门槛的管道预埋，本轮不声明任何 kill bar）；
5. 每行 feature record 必须通过 assert_no_gold_label_leakage；
6. 输出 f5_feature_plumbing_report.json：
   逐特征的 min/max/mean/有限性检查、缺失计数、
   smoke 级 AUROC（明确标注 plumbing_validation_only=true，
   32 题的 AUROC 不作为任何门槛）。
```

---

## 8. Stage C：Option-Position Model Bias 审计

补上 4B-1 pre-model 报告中留待模型侧的部分：

```text
输出 option_position_model_bias_report.json：
  greedy 预测的 A/B/C/D 分布（两条件各一份）；
  sampled 预测的 A/B/C/D 分布；
  gold 分布（该 32 题子集）；
  greedy accuracy（两条件各一份，仅作 smoke 参考）；
  position-only baseline accuracy；
  若某字母在预测中占比 > 0.40，输出 warning。
```

---

## 9. 输出清单

```text
outputs/logs/sprint_4B_2_small_model_smoke/
  preflight_report.md
  ab_question_manifest.jsonl（32 题清单）
  trace_manifest_cond_raw.jsonl
  trace_manifest_cond_chat.jsonl（逐条含 degenerate 标志与命中规则）
  prompt_ab_report.json（两条件的 parse/degeneration/correct 率与决策）
  f5_feature_plumbing_report.json
  option_position_model_bias_report.json
  review_gate_small_model_smoke.md
```

同时更新：

```text
PROGRESS.md
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

---

## 10. 测试要求

`tests/test_domain_label_proxy.py` 至少新增：

```text
1. detect_degeneration 对 smoke 真实退化样例（三类）判正；
2. 正常长推理文本判负；
3. 边界：恰好 29 次重复判负、30 次判正；
4. truncated_failure 需要同时满足「无法解析」与「长度阈值」；
5. chat-template 风格全文上 locate_label_readout_position 定位正确；
6. F5 feature record 通过 assert_no_gold_label_leakage
   （含 tier 字段后仍不含 gold 系字段）。
```

如新增 chat prompt builder，`tests/test_cyber_data.py` 补：

```text
7. chat messages 构造包含全部选项且不泄漏 gold；
8. system/user 拆分符合 §6.2。
```

---

## 11. Review Gate

`review_gate_small_model_smoke.md` 必须逐条回答：

```text
1.  抽取了哪 32 题（seed、split）？
2.  两条件各生成多少 traces？
3.  cond_raw 的 parse_failure_rate / degeneration_rate？
4.  cond_chat 的 parse_failure_rate / degeneration_rate？
5.  退化判定命中的规则分布？
6.  决策结果与理由（含平手规则是否触发）？
7.  胜出条件相比 4B smoke（parse_failure 0.10）是否改善？
8.  采样保护（renormalize/remove_invalid）是否确认在位？
9.  label-readout 运行时断言的失败计数？
10. F5 五特征是否全部有限且无缺失？
11. tier 字段是否已预埋且通过防泄漏检查？
12. smoke AUROC 是否已标注 plumbing_validation_only？
13. greedy 预测分布是否存在 position bias warning？
14. 本轮是否零 probe / 零 steering / 零 patching？答案必须为 yes。
15. 是否声明了任何 kill bar？答案必须为 no。
16. 是否允许进入 4B-3？
17. 4B-3 的明确输入（prompt 条件、脚本参数、数据文件）是什么？
```

---

## 12. 进入 4B-3 的条件

全部满足才允许进入 4B-3：

```text
1. 胜出条件 score（parse_failure + degeneration）<= 0.08；
2. 退化判定测试（含真实样例 fixture）通过；
3. label-readout 运行时断言失败计数 = 0；
4. F5 特征全部有限、防泄漏检查通过；
5. 无 position bias 严重警告（或已如实记录并解释）；
6. full pytest 通过。
```

若条件 1 不满足但 score <= 0.15：允许进入 4B-3，但必须在 4B-3 卡中
把退化 trace 作为独立类别统计（不得静默剔除）。

---

## 13. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_domain_label_proxy.py tests/test_cyber_data.py -q

conda run -n recover_attention python scripts/sprint_4B_2_small_model_smoke.py \
  --processed-path data/processed/cyber/cybermetric.jsonl \
  --smoke-manifest outputs/logs/sprint_4B_1_cybermetric_schema_and_label_proxy/cyber_sample_smoke_manifest.jsonl \
  --num-questions 32 \
  --samples-per-question 3 \
  --temperature 0.7 \
  --max-new-tokens 256 \
  --question-seed 4242 \
  --output-dir outputs/logs/sprint_4B_2_small_model_smoke \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

---

## 14. 本轮最多允许的结论

```text
The winning prompt style reduces parse failure + degeneration to X on a
32-question smoke; the F5 feature plumbing (including cost-tier fields and
gold-leakage checks) is validated end-to-end; option-position model bias is
audited. The project is / is not ready for the Sprint 4B-3 full 240-question
F5 baseline run.
```

不得写：

```text
F5 kill bar established
detector works
hallucination reduced
accuracy improved
ready for 4C / 2000
```
