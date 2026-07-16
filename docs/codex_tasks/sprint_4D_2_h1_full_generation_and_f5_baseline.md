# Sprint 4D-2：H1 全量生成 + H1-F5 kill-baseline（DETECT 基线）

## 1. 定位

4D-1 已过可行性门（Route A greedy emission 0.958；ATT&CK+CWE mention-level
fabrication 0.084 ∈ [0.05,0.60]；has_reasoning_text 1.00 vs MCQ 0.00）。本 sprint
把 H1 从 smoke 扩到**全量生成**，并建立 **H1-F5 输出层检测基线** ——
`h1_f5_design.md` 预注册的四类基线的**第一次实测**。

本轮**只做 F5**。这是 kill-gate 纪律的地基：任何内部特征族（F1–F4）在后续卡里
必须报告相对 H1-F5 的分组 CV + CI 增量。**本轮不碰 F1–F4、不捕获中间层激活、
不训练非线性模型、不做 steering / patching / intervention / 2000-scale。**唯一训练
的是在 F5 标量特征上的分组 CV 线性 logistic 检测器 + 校准温度（与 4C 同规格）。

默认结论标记：

```text
ready_for_2000_rerun=False
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
internal_features_touched=False（F1-F4 一律不做）
probe_trained=True（仅 F5 标量上的线性 logistic 检测器）
h1_f5_auroc=UNSET（本轮实测填）
```

措辞纪律：F5 的 AUROC/AUPRC 是**检测基线**，不是 hallucination reduction、不是
accuracy improvement、不是 intervention 结果。CVE 判据弱化按 4D-0/4D-1 结论继续。

---

## 2. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md（§4 三层架构 / §5 kill gate / §8 / §14.4）
docs/codex_tasks/sprint_4D_0_h1_fabricated_identifier_data_design.md
docs/codex_tasks/sprint_4D_1_h1_emission_fabrication_smoke.md（生成路径 / 门槛 / 诊断）
outputs/logs/sprint_4D_0_h1_data_design/h1_f5_design.md（四类基线定义——本轮实现它）
outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json（CVE 弱化边界）
outputs/logs/sprint_4D_1_h1_emission_fabrication_smoke/emission_fabrication_report.json（分布先验）
docs/progress/sprint_4_history.md（4C F5-only CV AUROC 0.8343 作为「易校准任务」对照）

src/recover_attention/h1_identifier.py（label_completion / extract_identifiers / classify_mention —— 标注唯一权威，只读）
src/recover_attention/h1_data.py（build_h1_chat_messages / grouped_split —— 只读复用）
src/recover_attention/mlp_readout_attribution.py（AUROC / AUPRC / 分组 bootstrap CI 复用）
scripts/sprint_4D_1_h1_emission_fabrication_smoke.py（生成路径母版：generate_completion / sanity 闸 / offset 定位）
scripts/sprint_4C_narrowed_readout_increment_and_site_transfer.py（分组 CV logistic + 校准 的规格母版）
```

---

## 3. 硬性条款

```text
1. 标注只走 h1_identifier.label_completion；OntologyIndex 由 build_ontology_index
   载入，载入后断言 index_sha256 与 4D-0 ontology_snapshot_manifest.json 一致，
   不一致停止（本体漂移会让 fabrication 标签与 4D-1 不可比）。
2. gold id（source_entry_id）只作 eval 对照，绝不进 prompt、绝不作生成条件、
   绝不作 inference feature；每条 F5 feature record 过 assert_no_h1_gold_label_leakage。
3. 生成条件 = build_h1_chat_messages（4B-2/4D-1 赢家）；复用 4D-1 修复后的
   generate_completion（model.generate 路径）与 _sanity_check_generation 前置闸。
   Backend（4D-1 实测硬约束）：
     - 必须用 local 8-bit model.generate（4D-1 唯一有效 run 的 backend）；
     - 禁止 4-bit 长文本生成路径——4D-1 实测 4-bit 在长 completion 上仍标点退化；
     - fp16 干净但过慢，不用作全量 backend；
     - 不再走手写 KV-cache 逐 token loop（4D-1 已作废）；
     - 不使用旧的 "CPU-side 采样规避 CUDA multinomial" 路径——8-bit model.generate
       的采样已正常，若某环境仍触发 CUDA multinomial 失败，如实报告并停止，
       不得退回手写采样。
   sanity 闸保留：全量前对前若干题 greedy 断言输出连贯（4D-1 的 _sanity_check_generation）。
4. echo mention 从检测集剔除（既非 grounded 亦非 fabricated；4D-1 已确认 echo
   全部来自 prompt 内含交叉引用的题），单独计数。
5. 检测标签定义（primary = mention 级）：
     正类 = fabricated mention；负类 = grounded mention；echoed 剔除。
   CVE 判据弱：primary 检测指标只用 ATT&CK+CWE mention；CVE mention 单列报告
   （全部 / 仅高序号两版），不进 primary AUROC（稠密号段的标签噪声会不公平地
    压低基线）。completion 级（h1_positive）作为 secondary 视角并列报告。
6. degeneration 诊断（区分两类，勿混）：
   (a) 真退化：4-bit 长文本的标点退化是真实失败，本轮用 8-bit backend 规避
       （见 §3.3），预期真退化率接近 0；sanity 闸拦第一类。
   (b) 假阳性：detect_degeneration 的「子串循环 >= 5 次」规则会被编号列表结构
       误触发（4D-1 open_gen:attack 0.31 已确认是连贯列表的假阳性，非退化）。
   因此本轮 degeneration 只作诊断、不进检测标签；report 必须同时给出
   「原始 detect_degeneration 触发率」与「扣除列表结构假阳性后的真退化估计」
   （用一个可测试的列表结构识别器在脚本局部判定，不改 domain_label_proxy 共享实现）。
   若扣除列表假阳性后的真退化率仍非零，逐条列出并抽查，判断是否 backend 异常。
```

---

## 4. Preflight

输出 `outputs/logs/sprint_4D_2_h1_full_generation_and_f5_baseline/preflight_report.md`，
检查并记录：

```text
1. 工作区干净且 4D-1 生成器可复现（硬性前置）：
     - `git status --porcelain` 无未跟踪/未提交的实验代码；
     - 显式断言 scripts/sprint_4D_1_h1_emission_fabrication_smoke.py 与
       tests/test_h1_emission_smoke.py 已被 git 跟踪（4D-1 的 commit 7b9c6bb
       曾漏提交这两个文件）——任一仍为 untracked 则停止：
       H1-F5 基线不能建在不可复现的生成器上；
     - 记录本轮所用 generate_completion 的来源 commit（应含 8-bit model.generate 修复）；
2. 4D-0/4D-1 产物完整（h1_samples.jsonl、ontology index、两份 report）；缺失停止；
3. OntologyIndex 指纹校验通过并记录；
4. model_path 解析（--model-path > env > fallback），记录 model_path_source；
5. 声明：internal_features_touched=False、无中间层激活捕获、probe 仅 F5 线性 logistic；
6. 不覆盖 sprint_3*/4A*/4B_*/4D_0*/4D_1* 任何输出目录。
```

## 5. Stage 1：全量生成

```text
数据：data/processed/h1/h1_samples.jsonl 全部 480 题（train/dev/test 都生成；
  split 仅用于检测器 CV，不用于生成筛选）。
Backend：local 8-bit model.generate（见 §3.3），非 4-bit、非 fp16、非手写 loop。
每题：1 greedy + K sampled（temperature 0.7）；K=5（自一致性需要足够样本，
  4D-1 用 3 仅为 smoke）。max_new_tokens 512（4D-1 open_gen 触顶 0.06，512 更稳；
  报告实际长度与触顶率，若某 family 触顶 > 0.1 在 review gate 标注截断风险）。
落盘 h1_generation_manifest.jsonl：prompt 指纹、completion、token 数、触顶、
  temperature、sample_index、logprob 捕获所需的 token id 序列（见 §6）。
规模与耗时（按 4D-1 实测外推，勿低估）：480 × 6 ≈ 2880 次生成。
  4D-1 的 8-bit 实测 288 traces ≈ 5464 秒（约 91 分钟）；线性外推 2880 traces
  ≈ 10 倍 ≈ 约 15 小时（不是早先误写的 3-5 小时）。因此**必须**：
    - 后台运行、打印进度；
    - 断点续跑（按 example_id + sample_index 幂等跳过已完成 trace，
      崩溃/中断可继续，不重跑已完成部分）；
    - 长文本 max_new_tokens 512 是耗时主因之一，触顶率低时不必再抬预算。
```

## 6. Stage 2：H1-F5 特征捕获（实现 h1_f5_design.md 四类）

F5 logprob 捕获方法（强制 teacher-forcing，不用生成期 scores）：
```text
对每条已生成的 completion，把 (prompt + completion) 拼成完整序列，teacher-forcing
单次 forward（8-bit 权重、torch.no_grad、无梯度），读 completion 区间每个 position
的 next-token log-softmax，取实际生成 token 的 logprob。
理由：8-bit + 采样期的 return_dict_in_generate/output_scores 在数值与对齐上不如
  teacher-forcing 干净、且难以复现；teacher-forcing 单 forward 稳定、可单测、
  与 greedy/sampled 一致处理。禁止用生成期 output_scores 冒充 teacher-forcing logprob。
id-span 定位：用 extract_identifiers 的字符 span + tokenizer offset_mapping 映射到
  token 位置；dual-form tokenization 纪律沿用 4B/4D-1；多 mention completion 逐个定位。
```
**全程无 gold 参与特征计算。**

```text
Mention 级（primary，每个非 echo mention 一行）：
  f5_id_logprob_mean / f5_id_logprob_min  —— id-span token 的 mean/min logprob；
  f5_first_id_token_rank                   —— id 首 token 在该步 top-k 中的排名；
  f5_id_token_entropy_mean                 —— id-span 各步 next-token 分布熵均值。
Sequence 级（completion 一行，供 mention 继承）：
  f5_completion_perplexity / f5_lengthnorm_logprob；
  f5_id_token_ratio                        —— id token / 总 token（verbosity 控制）。
Sampling 级（同题同 slot 跨 K 采样聚合）：
  f5_ksample_exact_id_agreement            —— 同一请求 slot 的精确 id 一致率；
  f5_self_consistency_exact                —— 精确串自一致（非语义聚类；
                                              id 是精确串，语义熵退化为精确匹配，
                                              按 h1_f5_design 说明处理）。
Verbalized confidence（诊断，可不进 primary）：
  固定正则抽取显式置信短语；与本体标签严格隔离。
落盘 h1_f5_feature_manifest.jsonl（per-mention 特征 + split + group_id + eval-only
  fabricated 标签），过 gold-leakage 检查。
```

## 7. Stage 3：检测基线 + 校准（kill-gate 地基）

```text
检测任务（primary）：mention 级二分类，label = fabricated(1) / grounded(0)，
  echoed 剔除，CVE 剔除出 primary（单列）。
评估协议（apples-to-apples，沿用 4C）：
  分组 CV（group_id = 本体条目 / 主题实体，防题目级泄漏）；
  F5 组合走分组 CV 线性 logistic；3 seeds；分组 bootstrap CI95（>= 500 次）；
  类不平衡（fabrication ~0.08-0.10）→ AUROC 与 AUPRC 并报，AUPRC 尤其关键，
  同时报 base-rate 参考线。
报告矩阵：
  单特征裸 AUROC/AUPRC（mention/sequence/sampling 各特征）——诊断哪个 F5 分量最强；
  F5-full（全特征 CV logistic）——H1 的 kill baseline（后续 F1-F4 要打败的就是它）；
  按 family（attack / cwe）、按 route（recall / open_gen）分层的 F5-full——
    定位 F5 在哪类样本上有效/失效（先验：fabrication 几乎全在 open_gen）。
校准与选择性预测（诚实交付形态）：
  最优组合的 ECE、risk-coverage 曲线、AURC、固定覆盖率下的 fabrication-through rate。
对照：把 H1-F5 AUROC 与 4C MCQ F5-only 0.8343 并排——预期 H1 更低
  （h1_f5_design 已说明；这正是选 H1 的理由：输出校准在 H1 上已知会退化）。
输出 h1_f5_baseline_report.json（矩阵 + 分层 + CI）与 h1_calibration_report.json。
```

## 8. 输出清单

```text
outputs/logs/sprint_4D_2_h1_full_generation_and_f5_baseline/
  preflight_report.md
  h1_generation_manifest.jsonl
  h1_mention_labels.jsonl
  h1_f5_feature_manifest.jsonl
  h1_f5_baseline_report.json（核心：单特征 + F5-full + 分层 + CI，CVE 单列）
  h1_calibration_report.json
  high_risk_fabricated_case_report.jsonl / low_risk_fabricated_case_report.jsonl
  review_gate_h1_full_generation_and_f5_baseline.md
```

同时更新：

```text
PROGRESS.md（顶部 status：h1_f5_auroc 与选择性预测结论）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```

## 9. 本次允许修改的文件

```text
scripts/sprint_4D_2_h1_full_generation_and_f5_baseline.py（新增）
src/recover_attention/h1_f5_features.py（新增：F5 特征纯函数，便于单测）
tests/test_h1_f5_features.py（新增）
PROGRESS.md / docs/progress/sprint_4_history.md / docs/progress/sprint_4_artifact_manifest.md
```

## 10. 本次禁止修改的文件

```text
README.md / AGENTS.md / docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/*
src/recover_attention/h1_identifier.py（标注权威，只读）
src/recover_attention/domain_label_proxy.py（退化检测共享实现，只读；列表假阳性
  修正在本轮脚本局部做，不动共享函数）
data/processed/h1/h1_samples.jsonl（不重建）
scripts/sprint_4B_* / scripts/sprint_4D_1_* 与其输出目录（不覆盖）
```

## 11. 测试要求

```text
1. F5 特征纯函数：手构 logits/logprob 验证 id_logprob_mean/min、first_id_token_rank、
   perplexity、lengthnorm_logprob、id_token_ratio；
2. id-span → token 位置定位：dual-form tokenization 下 offset 映射正确
   （含多 mention completion、id 在句中/句尾）；
3. K-sample exact agreement / self_consistency_exact 的纯函数（含全一致 / 全不一致边界）；
4. 分组 CV apples-to-apples：F5-full 与单特征走同一 CV 接口；分组不泄漏；
5. echo 剔除 + CVE 剔除出 primary 的集合逻辑；
6. gold-leakage 断言覆盖 h1_f5_feature_manifest 形态；
7. label_completion / h1 既有测试保持通过；
8. full pytest 通过。
```

## 12. Review Gate（逐条回答，比率均带 n）

```text
0.  Backend 确认：是否 local 8-bit model.generate？（必须 yes）是否复用 4D-1 修复后
    的 generate_completion + sanity 闸？是否未用 4-bit/fp16/手写 loop/CPU-side 采样？
    F5 logprob 是否用 teacher-forcing 单 forward（非生成期 output_scores）？
1.  全量生成：480 题 × (1+K) traces，K=? 实际总耗时？触顶率（按 family）？截断风险？
2.  echo 剔除后 primary（ATT&CK+CWE）mention 检测集大小 n？正类（fabricated）n？base rate？
3.  最强单 F5 特征是哪个（mention/sequence/sampling 各族）？裸 AUROC/AUPRC？
4.  H1-F5-full 分组 CV AUROC = ?（CI）AUPRC = ?（CI，带 base-rate 线）
5.  与 4C MCQ F5-only 0.8343 对照：H1 是否如预期更低？低多少？说明含义。
6.  分层：attack vs cwe、recall vs open_gen 的 F5-full 各是多少？F5 在哪类失效？
7.  CVE 单列：全部 vs 仅高序号 的检测表现？为何不进 primary？
8.  校准：ECE / AURC / 固定覆盖率（如 80%）下的 fabrication-through rate？
9.  退化诊断：原始 detect_degeneration 触发率 vs 扣列表假阳性后的估计？
10. 本轮是否碰 F1-F4 / 捕获中间层激活 / 训练非线性模型？（必须全 no）
11. gold id 是否只作 eval 对照、从未进 prompt / feature？（必须 yes）
12. 是否出现 hallucination reduction / accuracy / intervention 声明？（必须 no）
13. 下一步建议：F5 是否足够强到值得上 F1-F4 增量检验？还是先修数据/诱发？
```

## 13. 结果分支（review gate 必须给出明确建议）

```text
F5 强（AUROC 明显 > 0.5、AUPRC 明显 > base rate，CI 稳）：
  → H1-F5 selective-prediction 是当前诚实交付；
  → 起草 4D-3：F1-F4 内部特征在 H1 上的增量检验（kill gate vs 本轮 F5-full）。
    这是主线假设「site-informed 内部特征在输出校准差的任务上跑赢 F5」的真正检验。
F5 弱（AUROC ≈ 0.5 或 AUPRC ≈ base rate）：
  → H1 上输出层对 fabrication 无信号——这本身可能是「输出校准在 H1 失效」的
    正面证据（与 4C 干净 MCQ 形成对比），恰是内部特征的用武之地；
  → 仍先起草 4D-3（F1-F4），但明确 F5 基线很低、增量门槛也低，需警惕
    小样本 / 类不平衡下的假阳性增量（更严的 CI + 更多 seed）。
数据问题（正类 n 太小、生成截断严重、某 family 无正例）：
  → 回到诱发设计卡扩样本 / 调 open_gen 配额（4D-1 已知 open_gen 是 fabrication 主源），
    不得在 n 不足的基线上跑 F1-F4。
任一情况都不进 intervention（Layer 3），本轮及 4D-3 均为 DETECT 层。
```

## 14. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_h1_f5_features.py tests/test_h1_identifier.py -q

conda run -n recover_attention python scripts/sprint_4D_2_h1_full_generation_and_f5_baseline.py \
  --samples-jsonl data/processed/h1/h1_samples.jsonl \
  --ontology-dir data/raw/ontology \
  --snapshot-manifest outputs/logs/sprint_4D_0_h1_data_design/ontology_snapshot_manifest.json \
  --density-report outputs/logs/sprint_4D_0_h1_data_design/id_space_density_report.json \
  --samples-per-question 5 --temperature 0.7 --max-new-tokens 512 \
  --cv-seeds 0 1 2 --seed 4242 \
  --output-dir outputs/logs/sprint_4D_2_h1_full_generation_and_f5_baseline
```

预计（按 4D-1 8-bit 实测外推）：全量生成 ~15 小时（8-bit，后台、幂等续跑）；
teacher-forcing 特征捕获 ~1-2 小时；CV/校准 ~几分钟。不要按 3-5 小时排期。
backend 必须是 local 8-bit model.generate；如需传显式 flag，用 --load-in-8bit
（或脚本内固定 8-bit 量化配置），禁止 4-bit 长文本路径。

## 15. 最多允许的结论

```text
On the full 480-prompt H1 fabricated-identifier set under the chat condition,
the Qwen2.5-7B output-level F5 baseline (mention/sequence/sampling logprob and
consistency features) detects fabricated ATT&CK/CWE identifiers at grouped-CV
AUROC X / AUPRC Y (CIs reported), lower than the 4C MCQ F5-only 0.8343 as
expected for a task where output calibration is weaker. This is the H1 DETECT
kill baseline; no internal feature (F1-F4), intervention, hallucination-reduction,
or accuracy claim is made. Any future internal family must beat this F5-full bar
under grouped CV with CI.
```

## 16. 禁止事项

```text
不要做 F1-F4 内部特征、捕获中间层激活、训练非线性模型。
不要 steering / patching / site-transfer / intervention（Layer 3）。
不要 2000-scale、不要重建 4D-0 数据、不要覆盖既有输出目录。
不要把 gold id 放进 prompt / feature。
不要基于 F5 AUROC 声称减少幻觉 / 提升准确率。
不要用 4-bit 长文本生成路径、fp16 全量 backend、手写 KV-cache loop 或
  旧 CPU-side 采样路径（4D-1 已作废/证伪，仅 local 8-bit model.generate 有效）。
不要用生成期 output_scores 冒充 teacher-forcing logprob。
不要下载模型。
```

## 17. 完成后必须更新

```text
PROGRESS.md（顶部 status + §6 下一步：F5 强→4D-3 增量检验；弱/数据问题→对应卡）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```
