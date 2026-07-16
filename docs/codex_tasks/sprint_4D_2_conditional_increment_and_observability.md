# Sprint 4D-2(修订版):H1 全量生成 + observability gate + 条件增量 bake-off

> **本卡取代** `sprint_4D_2_h1_full_generation_and_f5_baseline.md`(旧 F5-only 设计)。
> 旧卡把 H1 上算 F1/F4、只做 F5,已被 workshop 收敛设计推翻。以本卡与
> `docs/paper/preregistration.md` 为准。旧卡保留不删,仅供对照。

## 1. 定位

本卡是投稿实验的执行主卡,回答 `preregistration.md` 的 RQ1 / RQ2,并先过一个
**observability 前置 gate**。不是"只做 F5",而是:输出侧 artifact 阶梯 → 最强
输出基线 O → hidden probe H → 条件增量 Δ_H = AUROC(O+H) − AUROC(O)。

**启动前置条件(未满足不许跑生成):**

```text
1. 目标 workshop 的 CFP 已确认(deadline / page limit / archival)。
2. docs/paper/preregistration.md 已冻结(thesis / RQ / O,H / 判读规则 / 单位 / 阶梯)。
```

默认结论标记:

```text
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
h1_is_high_confidence_setting=UNSET(前置 gate 填)
delta_H_ci=UNSET(RQ1 填)
delta_H_depends_on_observability=UNSET(RQ2 填)
```

## 2. 执行前必读

```text
docs/paper/preregistration.md(唯一权威设计,以下为其执行细化)
AGENTS.md / PROGRESS.md
docs/codex_tasks/sprint_4D_0_*.md、sprint_4D_1_*.md(数据 / 生成路径 / 门槛)
outputs/logs/sprint_4D_0_h1_data_design/h1_f5_design.md、id_space_density_report.json

src/recover_attention/h1_f5_features.py(已建:mention_logprob_features /
  sequence_logprob_features / id_agreement_rate / exact_consistency /
  extract_verbalized_confidence / looks_like_structured_identifier_list —— 直接复用)
src/recover_attention/h1_identifier.py(label_completion / extract_identifiers,只读)
src/recover_attention/h1_data.py(build_h1_chat_messages / grouped_split,只读)
scripts/sprint_4D_1_*.py(8-bit model.generate 路径 + sanity 闸,复用)
scripts/sprint_4C_*.py(grouped-CV logistic + paired grouped bootstrap,复用)
```

## 3. 硬性条款

```text
1. Backend:local 8-bit model.generate(4D-1 唯一有效 backend)。禁 4-bit 长文本、
   禁 fp16 全量、禁手写 KV loop、禁旧 CPU-side 采样。保留 sanity 闸。
2. 缓存(一遍前向,预注册):completion、token logprobs、
   preregistration §6 的层(相对深度 0.7L)/位置/pooling 的 hidden、identifier 位置。
   **不缓存 raw attention。**
3. H1 label:completion-level primary(含 ≥1 fabricated id);mention-level secondary
   (仅 observability 分析)。标注只走 h1_identifier.label_completion;
   OntologyIndex 指纹与 4D-0 manifest 一致,否则停。
4. 分组:CV 与 bootstrap 一律按 source prompt 分组;K 个 completion 不当独立样本。
5. 不做:H1 上的 F1/F4、raw attention、TruthfulQA、非线性大模型、steering/patching。
6. gold id 只作 eval 标签,绝不进 prompt / feature;每条 record 过
   assert_no_h1_gold_label_leakage。
```

## 4. Stage 0:H1 全量生成(单点故障,第一天开)

```text
数据:data/processed/h1/h1_samples.jsonl 全部 480 题(train/dev/test 都生成;
  split 只用于 probe CV)。
每题:1 greedy + K sampled(temperature 0.7),K=5~6;max_new_tokens 512。
一遍前向落盘 §3.2 缓存契约的全部特征(teacher-forcing 取 completion 区间 logprob;
  identifier 字符 span → token 位置用 extract_identifiers + offset_mapping)。
规模:480 × (1+K) ≈ 2880 traces。8-bit 实测 ~15–20 GPU-hr(4D-1 外推),后台 + 幂等续跑。
产物:h1_generation_manifest.jsonl、h1_feature_cache.(jsonl/npz)。
```

## 5. Stage 1:observability 前置 gate(先于任何 probe bake-off)

```text
目的:证明 H1 确实是"高置信 fabrication"任务,否则核心解释变量没被构造。
度量(base-rate 稳健,禁用跨任务裸 AUROC):
  MCQ:error-vs-correct 的输出置信度(label margin / entropy)分布可分性;
  H1:fabricated-vs-valid 的 id-token logprob(mention_logprob_features)分布可分性;
  各报 Cohen's d、分布直方图、AUPRC。
判据:H1 fabricated/valid 重叠明显 > MCQ wrong/correct → h1_is_high_confidence_setting=True。
Outcome 3:若 H1 output-only 仍强可分,记 =False,后续按"两任务都可观测"写,
  不得假装 H1 是高置信设置。
产物:observability_gate_report.json(两任务分布 + d + AUPRC + 判据)。
```

## 6. Stage 2:输出侧 artifact 阶梯 → O

```text
按 preregistration §5,从弱到强跑,全部走 grouped-CV logistic:
  1 prompt-only  2 id-string-only  3 surface-format-only  4 full-response-text
  5 F5(h1_f5_features:margin/entropy、id-token mean/min logprob、id_agreement、
       verbalized confidence)  6 F5 + full-text = O
artifact 红线:若 2 或 3 已接近 full model → 记录"H1 被字符串/格式解决",
  该任务不做增量声明。
产物:output_ladder_report.json(每阶梯 AUROC/AUPRC + 红线判定 + O 的定义)。
```

## 7. Stage 3:hidden probe H 与条件增量(RQ1)

```text
H:SAPLMA-style,预注册层 0.7L / 位置(MCQ answer-readout;H1 identifier-token mean)/
  pooling mean,grouped-CV logistic。禁事后挑最优层(robustness 附录另报,primary 恒 0.7L)。
Δ_H = AUROC(O + H) − AUROC(O),paired grouped-bootstrap(按 prompt,≥1000)95% CI;
  AUPRC 增量并报。判读规则(预注册):CI 全>0 / 跨0 / 全<0。
SE:H1 上退化为 id_agreement_rate,作 error-mode baseline(预期漏稳定 fabrication),
  不当独立 probe。EigenScore 仅复现 ≤2 天时加。
主表:| Task | O | H | O+H | ΔH AUROC CI | ΔH AUPRC CI |
产物:increment_report.json(主表 + CI + 判读结论)。
```

## 8. Stage 4:RQ2(observability 分层)

```text
within-H1:按 id-token logprob 分"高置信 / 低置信 fabrication"两层,
  各层 Δ_H + 层间差(paired bootstrap)。这是 RQ2 有统计力的一半。
粗对照:Δ_H(MCQ) vs Δ_H(H1),明确写这是 2 点趋势、非 interaction。
产物:rq2_observability_report.json。
```

## 9. Stage 5:第二模型核心复核(Llama-3.1-8B)

```text
只跑核心 O / H / O+H,在 {MCQ, H1};跨模型协议固定(同相对层深/pooling/position/
  正则搜索),不分别调层。回答"无增量是否 Qwen 特例"。
产物:cross_model_core_report.json。
```

## 10. 输出清单

```text
outputs/logs/sprint_4D_2_conditional_increment/
  preflight_report.md
  h1_generation_manifest.jsonl / h1_feature_cache.*
  observability_gate_report.json          (Stage 1,前置 gate)
  output_ladder_report.json               (Stage 2,artifact 阶梯 + O)
  increment_report.json                   (Stage 3,RQ1 主表 + CI)
  rq2_observability_report.json           (Stage 4)
  cross_model_core_report.json            (Stage 5)
  calibration_selective_report.json       (secondary:ECE/Brier/risk-coverage)
  review_gate_conditional_increment.md
```

## 11. 测试要求

```text
1. h1_f5_features 既有函数的单元测试保持通过(mention_logprob / id_agreement 等)。
2. artifact 阶梯特征的纯函数测试(prompt-only / id-string-only / surface-only 提取)。
3. grouped split / grouped bootstrap 的 apples-to-apples 与不泄漏测试(按 prompt 分组)。
4. observability 度量(Cohen's d / 分布可分性)纯函数测试。
5. gold-leakage 断言覆盖所有落盘 feature record。
6. full pytest 通过。
```

## 12. Review Gate(逐条答)

```text
0.  前置条件:workshop CFP 确认?preregistration 冻结?(必须 yes 才启动)
1.  observability gate:H1 fabricated/valid 重叠 vs MCQ wrong/correct(Cohen's d/AUPRC)?
    h1_is_high_confidence_setting = ?若 False → Outcome 3 如何写?
2.  artifact 红线:id-string-only / surface-only 的 AUROC?是否接近 full model?
3.  O 的定义(最强输出侧)= ?各阶梯 AUROC?
4.  H(SAPLMA,0.7L)AUROC?Δ_H = AUROC(O+H)−AUROC(O)?paired grouped CI?判读(>0/跨0/<0)?
5.  AUPRC 增量 + base-rate 线?
6.  RQ2 within-H1:高置信层 vs 低置信层的 Δ_H?层间差 CI?
7.  Δ_H(MCQ) vs Δ_H(H1)粗对照?(明确标 2 点趋势)
8.  SE 在 H1 是否退化为 id-agreement、是否漏稳定 fabrication?
9.  Llama-core:方向是否与 Qwen 一致?跨模型协议是否固定、未分别调层?
10. 单位:completion-level primary?按 prompt 分组?K completion 未当独立样本?
11. 是否碰 H1-F1/F4、raw attention、TruthfulQA、非线性大模型?(必须全 no)
12. gold id 是否只作 eval、从未进 prompt/feature?leakage 测试覆盖?(必须 yes)
13. 是否声称 hallucination reduction / accuracy?(必须 no)
```

## 13. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_h1_f5_features.py tests/test_h1_identifier.py -q

# Stage 0 生成(后台、幂等)
conda run -n recover_attention python scripts/sprint_4D_2_conditional_increment.py \
  --stage generate --samples-jsonl data/processed/h1/h1_samples.jsonl \
  --ontology-dir data/raw/ontology --model-path <QWEN_8BIT> \
  --samples-per-question 6 --max-new-tokens 512 \
  --hidden-rel-depth 0.7 --no-cache-attention \
  --output-dir outputs/logs/sprint_4D_2_conditional_increment

# Stage 1..5(生成完成后)
conda run -n recover_attention python scripts/sprint_4D_2_conditional_increment.py \
  --stage gate,ladder,increment,rq2,cross_model \
  --cv-grouped-by source_prompt --bootstrap 1000 \
  --output-dir outputs/logs/sprint_4D_2_conditional_increment
```

## 14. 禁止事项

```text
未确认 workshop CFP / 未冻结 preregistration → 不启动生成。
不做 H1 的 F1/F4、raw attention 缓存、TruthfulQA、steering/patching、非线性大模型。
不事后挑最优层做增量声明(primary 恒 0.7L)。
不把 O 设成弱基线(O 必须是输出侧最强,含 full-text)。
不把 K completion 当独立样本;不把"校准"当核心解释术语。
不基于结果声称减少幻觉 / 提升准确率。
```

## 15. 完成后更新

```text
docs/paper/preregistration.md(回填 §11 结果占位)
docs/paper/six_week_plan.md(标 W1–W5 实际进度)
PROGRESS.md / docs/progress/sprint_4_history.md
```
