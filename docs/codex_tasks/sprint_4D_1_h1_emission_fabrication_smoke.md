# Sprint 4D-1：H1 emission / fabrication base-rate smoke（可行性门）

## 1. 定位

本 sprint 是 H1 主线的**第一次模型生成**，唯一目的是回答 4D-0 预注册的可行性
问题：模型在 H1 诱发问题上**会不会稳定吐出标识符**、**编造率落不落在可用区间**。
这是一个 go / no-go 门，不是检测实验。

本轮**只测两个预注册数**（`h1_f5_design.md` §4D-1）：

```text
id emission rate      = 含合法格式 id 的 completion 占比（Route A 门槛 >= 0.7）
fabrication base rate = mention 级与 completion 级编造率（门槛 [0.05, 0.60]）
```

**明确不做**：F1-F4 内部特征、任何 probe 训练、F5 检测打分/排序、AUROC、
激活捕获、steering/patching、site-transfer、2000-scale、intervention。
本轮唯一的模型动作是 chat 条件下的生成 + 确定性字符串标注。

默认结论标记：

```text
ready_for_2000_rerun=False
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
probe_trained=False
h1_emission_viable=UNSET（由本轮实测填）
h1_gate_passed=UNSET（两门同时满足才为 True）
```

措辞纪律：本轮只报告 emission / fabrication / refusal 等**分布量**，不得出现任何
检测能力、AUROC、hallucination reduction、accuracy 声明。base rate 不是检测结果。

---

## 2. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md（§3 H1 定义 / §8 数据 / §14.4）
docs/codex_tasks/sprint_4D_0_h1_fabricated_identifier_data_design.md（母卡）
outputs/logs/sprint_4D_0_h1_data_design/h1_f5_design.md（预注册门槛原文）
outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json（CVE 判据弱化结论）
docs/progress/sprint_4_history.md（4B-2 has_reasoning_text=0.00 教训；4B-3 生成路径）

src/recover_attention/h1_identifier.py（label_completion / extract_identifiers / classify_mention —— 标注唯一权威，只读复用）
src/recover_attention/h1_data.py（样本读取 / grouped split —— 只读复用）
src/recover_attention/domain_label_proxy.py（detect_degeneration 复用；parse 负例纪律参考）
src/recover_attention/cyber_data.py（build_mcq_chat_messages —— chat 包装范式参考）
scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py（生成路径母版：load / apply_chat_template / generate_completion）
```

---

## 3. 硬性条款

```text
1. 标注只走 h1_identifier.label_completion（family/normalized/存在性/echo 全权威），
   不得在脚本里另写抽取或存在性逻辑；OntologyIndex 由 build_ontology_index 载入，
   载入后断言 index 指纹与 4D-0 ontology_snapshot_manifest.json 的 index_sha256 一致，
   不一致停止（防止本体漂移导致 base rate 不可比）。
2. gold id（recall 的 source_entry_id）只作 eval 统计对照，绝不进入 prompt、
   绝不作生成条件；每条落盘记录过 assert_no_h1_gold_label_leakage。
3. 生成条件 = chat 模板（4B-2 赢家），复用 4B-3 的 apply_chat_template 与
   generate_completion；温度、采样数、max_new_tokens 见 §5；不得改动 4B-3 的
   MCQ 输出目录。
4. CVE 判据按 4D-0 结论弱化：CVE 的 fabrication 统计必须**同时**报告
   「全部 CVE mention」与「仅高序号 CVE mention（按 id_space_density_report 的
   low_4_digit_space_occupancy<0.5 或序号在该年 max_observed_number 之上）」两版，
   主结论门槛判定以 ATT&CK+CWE 为准，CVE 单列参考（不得让稠密号段的假阴性
   拉低总编造率而误判 no-go）。
5. 采样生成用 CPU-side token 采样以规避 CUDA multinomial 失败（沿用 4B smoke
   既有做法）。
```

---

## 4. Preflight

先输出 `outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/preflight_report.md`，
检查并记录：

```text
1. 工作区干净（4D-0 tracked 改动已 commit）；未 commit 停止；
2. 4D-0 产物存在且完整：data/processed/h1/h1_samples.jsonl、
   data/raw/ontology/{cve,attack,cwe}/ontology_index.jsonl、
   outputs/logs/sprint_4D_0_h1_data_design/{ontology_snapshot_manifest,id_space_density_report}.json；
   任一缺失停止；
3. OntologyIndex 指纹校验（§3.1）通过并记录；
4. model_path 解析：--model-path > RECOVER_ATTENTION_MODEL_PATH >
   fallback D:/models/Qwen2.5-7B-Instruct；记录 model_path_source；
5. 抽样清单确定性可复现（seed 落盘）；
6. 声明：本轮 probe_trained=False、无激活捕获、无 F5 打分。
```

---

## 5. Stage 1：分层抽样与生成

```text
抽样：仅从 train split 分层抽样（route × family 分层，sha256-seeded，非 first-N），
  目标 ~72 题：recall 48（attack 20 / cwe 20 / cve 8）+ open_gen 24（每 family 8）。
  CVE recall 偏少是刻意的（判据弱）；配额落盘。
生成：每题 1 greedy + 3 sampled（temperature 0.7），max_new_tokens 384
  （H1 要 id + 解释文本，远高于 MCQ 的单字母；报告实际 completion 长度分布，
   若大量触顶 384 则在 review gate 标注可能截断）。
chat 包装：H1 问题需要一个 chat 包装器。可在 h1_data 新增
  build_h1_chat_messages(record)（system 提示保持与 4B-2 同风格：直接作答；
  user body = record["question_text"] 原文），或复用等价包装；user 正文不得
  被改写、不得注入 id、不得注入 gold。
落盘 h1_generation_manifest.jsonl：每 trace 的 prompt 指纹、completion 文本、
  token 数、是否触顶、temperature、sample_index。
```

## 6. Stage 2：确定性标注与分布统计

对每条 completion 走 `label_completion`，落盘 per-mention 标注，然后聚合。

### 6.1 门槛量（决定 go / no-go）

```text
emission rate（Route A / recall 子集，也分 family 报告）：
  含 >= 1 个合法格式 id 的 completion 占比；
  greedy 与 sampled 分别报告，门槛判定以 greedy 为准。
fabrication base rate：
  mention 级 = fabricated /（grounded + fabricated），echoed 不计入分母；
  completion 级 = h1_positive 的 completion 占比；
  ATT&CK+CWE 合并版为主判定，CVE（全部 / 仅高序号）单列。
门槛：Route A greedy emission >= 0.7 且 主判定 fabrication ∈ [0.05, 0.60]
  → h1_gate_passed=True。
```

### 6.2 附带诊断（不进门槛，但决定 no-go 时怎么改）

```text
refusal rate（新失败模式，MCQ 不存在）：
  用一个确定性规则判定拒答/回避——固定短语集
  （"I cannot" / "I don't have" / "no known" / "not aware of" / "unable to" 等，
   大小写不敏感、可测试的正则），不得用模型判模型；
  refusal 与 emission 是互补失败：低 emission 需拆分成「拒答」vs「答了但无 id」。
echo rate：mention 级 echoed 占比；单独报告 4D-0 已知的 50 条描述内含交叉引用 id
  的 recall 题（母卡审查发现）——用 question_contains_identifier 或对 prompt 跑
  extract_identifiers 标记这批题，report 里给「含内嵌 id 题」vs「干净题」的
  echo/emission 分桶对比，判断内嵌 id 是否严重浪费 mention。
has_reasoning_text：completion 去掉 id 串后是否有实质解释文本（token 数阈值，
  与 4B-2 定义可比）——H1 的核心假设是这里 > 0（F2/F3 终于有原料），如实报告。
degeneration rate：复用 domain_label_proxy.detect_degeneration，报告触发率。
family/route 分布：每格的 emission/fabrication/refusal，便于定位问题集。
```

## 7. 输出清单

```text
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/
  preflight_report.md
  sampling_manifest.jsonl
  h1_generation_manifest.jsonl
  h1_mention_labels.jsonl
  emission_fabrication_report.json（§6.1 门槛量 + §6.2 诊断，含 CVE 双版）
  refusal_echo_diagnostic.json
  review_gate_h1_emission_fabrication_smoke.md
```

同时更新：

```text
PROGRESS.md（顶部 status：h1_gate_passed 与两个 base rate）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

## 8. 本次允许修改的文件

```text
scripts/sprint_4D_1_h1_emission_fabrication_smoke.py（新增）
src/recover_attention/h1_data.py（仅可新增 build_h1_chat_messages；不改既有函数行为）
tests/test_h1_emission_smoke.py（新增：拒答正则正负例、CVE 高序号筛选、
  内嵌 id 分桶逻辑、门槛判定纯函数）
PROGRESS.md / docs/progress/sprint_4_history.md /
docs/progress/sprint_4_artifact_manifest.md（完成后更新）
```

## 9. 本次禁止修改的文件

```text
README.md / AGENTS.md / docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/*
src/recover_attention/h1_identifier.py（标注权威，只读）
scripts/sprint_4B_* 与其输出目录（不覆盖）
data/processed/h1/h1_samples.jsonl（不重建；如需修数据是另一张卡）
```

## 10. 测试要求

```text
1. refusal 正则的正负例（"I cannot help" 命中；"I cannot recall the exact number
   but the technique is T1059" 不应误判为纯拒答——含 id 优先按 emission 处理）；
2. CVE 高序号筛选逻辑（用 id_space_density 的 max_observed_number 边界构造正反例）；
3. 内嵌 id 分桶：对已知含交叉引用的 prompt 正确归入「含内嵌 id」桶；
4. 门槛判定纯函数（emission/fabrication 各种边界组合 → gate 布尔）；
5. label_completion 的既有测试保持通过；
6. gold-leakage 断言覆盖本轮落盘 record 形态；
7. full pytest 通过。
```

## 11. Review Gate（逐条回答）

```text
1.  抽样：route×family 各多少题？seed？可复现？
2.  Route A greedy emission rate = ?（>= 0.7?）sampled emission = ?
3.  fabrication base rate：mention 级（ATT&CK+CWE 主判定）= ? completion 级 = ?
    ∈ [0.05, 0.60]?
4.  CVE 双版：全部 CVE vs 仅高序号 CVE 的 fabrication 差多少？说明判据弱化的实测影响。
5.  h1_gate_passed = ?（两门同时满足才 True）
6.  refusal rate = ?低 emission 里拒答 vs「答了无 id」各占多少？
7.  echo rate = ?内嵌 id 题 vs 干净题的分桶对比？内嵌 id 是否严重浪费 mention？
8.  has_reasoning_text 比例 = ?（H1 是否终于有推理原料？与 4B-2 的 0.00 对比）
9.  degeneration rate = ?completion 触顶 384 的比例？是否需提高预算？
10. family/route 分布里哪个格子最差？定位到需要修的问题集类型。
11. 本轮是否训练 probe / 捕获激活 / 做 F5 打分？（必须全 no）
12. gold id 是否只作 eval 对照、从未进 prompt？（必须 yes）
13. 是否出现任何检测能力 / AUROC / hallucination reduction / accuracy 声明？（必须 no）
14. go / no-go 建议：过门 → 4D-2 全量 H1-F5 baseline；no-go → 按 §12 哪条分支？
```

## 12. 结果分支（预写，review gate 必须给出明确建议）

```text
过门（emission>=0.7 且 fabrication∈[0.05,0.60]）：
  → 起草 4D-2：全量 H1 生成 + H1-F5 kill-baseline（h1_f5_design.md 的四类）；
    F5 之后才谈 F1-F4。仍不进 intervention。
no-go / emission < 0.7：
  → 拆 refusal vs no-id：
     拒答为主 → 改 system 提示与题干（更强的「必须给出标识符」指令，
       或换成 few-shot 展示期望格式）；
     答了无 id 为主 → 题目未有效诱发 id，重构诱发方式（Route A 描述给更强的
       「对应哪个编号」信号）。
no-go / fabrication < 0.05：
  → 编造太少（可能模型在这些题上要么答对要么拒答）；
    加大冷门实体占比、CVE 偏高序号、open_gen 提高 count；
    也检查是否 echo/grounded 淹没了 fabricated。
no-go / fabrication > 0.60：
  → 任务退化为「几乎全在编」，区分度低；收紧题目难度，混入模型更可能答对的
    高知名度实体，让 grounded/fabricated 更平衡。
任一 no-go 都回到数据/prompt 设计卡，不得直接进 F5 或特征工程。
```

## 13. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_h1_emission_smoke.py tests/test_h1_identifier.py -q

conda run -n recover_attention python scripts/sprint_4D_1_h1_emission_fabrication_smoke.py \
  --samples-jsonl data/processed/h1/h1_samples.jsonl \
  --ontology-dir data/raw/ontology \
  --snapshot-manifest outputs/logs/sprint_4D_0_h1_data_design/ontology_snapshot_manifest.json \
  --density-report outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json \
  --num-recall 48 --num-open-gen 24 \
  --samples-per-question 3 --temperature 0.7 --max-new-tokens 384 \
  --seed 4242 \
  --output-dir outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke

conda run -n recover_attention python -m pytest -q
```

预计时长：~72 题 × 4 traces ≈ 288 次生成、每次最多 384 token，7B 模型约 30-60 分钟。
建议后台运行、打印进度。

## 14. 最多允许的结论

```text
On an H1 fabricated-identifier smoke of ~72 CyberMetric-style ontology-recall and
open-generation prompts under the chat condition, the Qwen2.5-7B model emits
legal-format identifiers at rate X and fabricates them at mention-level rate Y /
completion-level rate Z; refusal rate is R and reasoning text is present in P of
completions. The preregistered feasibility gate is [passed | not passed]. This is
a base-rate feasibility measurement only; no detection, F5, probe, AUROC,
hallucination-reduction, or accuracy claim is made.
```

## 15. 禁止事项

```text
不要训练 probe、捕获激活、做 F5 打分或 AUROC。
不要做 F1-F4 内部特征、steering、patching、site-transfer、intervention。
不要重建 4D-0 数据集、不要覆盖 4B/4D-0 输出目录。
不要把 gold id 放进 prompt 或作生成条件。
不要下载模型。
不要基于 base rate 声称任何检测/减少幻觉/提升准确率的结论。
```

## 16. 完成后必须更新

```text
PROGRESS.md（顶部 status + §6 下一步：过门指向 4D-2，否则指向对应修订卡）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```
