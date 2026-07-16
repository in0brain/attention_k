# Sprint 4D-2（修订版 v2）：H1 全量生成 + observability gate + 条件增量 bake-off

> **取代** `sprint_4D_2_h1_full_generation_and_f5_baseline.md`（旧 F5-only、mention-level）。
> 以本卡 + `docs/paper/preregistration.md`（v2）为准。旧卡/旧脚本保留对照,**不得用于本实验**。
> 本卡是 spec:推荐的 `scripts/sprint_4D_2_conditional_increment.py` **尚不存在,需先实现**。

## 0. 执行顺序（硬）

```text
不是"先启动全量生成、后补代码"。顺序固定:
  W0.5-A 修死设计（preregistration v2,已完成）
  W0.5-B 实现新 pipeline script + 在少量 prompt 上 smoke,锁定缓存 schema
  W0.5-C 启动 gate 全过（CFP 确认 + preregistration 冻结 + smoke 通过）
  然后才 Stage 0 全量生成
2880 条 teacher-forced 前向依赖 completion-level schema / hidden 层位置 / no-emission policy,
  schema 跑后再改要重做全量。故 schema 必须先 smoke 锁定。
```

## 1. 启动 gate（preflight 代码强制,缺任一停）

```text
G1 目标 workshop CFP 已确认（deadline / page limit / archival),写入 preflight。
G2 preregistration.md 已冻结:preflight 记录其 sha256。
G3 smoke 通过:completion-level schema、hidden 层/位置、grouped split、
   O/H/O+H 同 folds、no-emission 处理、paired CI 可算——全部在 ≤20 prompt 上验证。
旧脚本 preflight 只查数据/模型/4D-1,不查 G1–G3,不得使用。
```

## 2. 执行前必读

```text
docs/paper/preregistration.md（v2,唯一权威设计）
AGENTS.md / PROGRESS.md
docs/codex_tasks/sprint_4D_0_*.md、sprint_4D_1_*.md（数据 / 8-bit 生成路径 / 门槛）
src/recover_attention/h1_f5_features.py（mention_logprob / sequence_logprob / id_agreement /
  verbalized_confidence / looks_like_structured_identifier_list —— 复用;需在其上新增
  completion-level 组装,不改既有函数语义）
src/recover_attention/h1_identifier.py（label_completion / extract_identifiers,只读）
src/recover_attention/h1_data.py（build_h1_chat_messages / grouped_split,只读）
scripts/sprint_4D_1_*.py（8-bit model.generate + sanity 闸,复用）
scripts/sprint_4C_*.py（grouped-CV logistic + paired grouped bootstrap,复用）
```

## 3. 硬性条款

```text
1. Backend:local 8-bit model.generate（4D-1 唯一有效）。禁 4-bit 长文本/fp16 全量/
   手写 KV loop/旧 CPU-side 采样。保留 sanity 闸。
2. K 固定:1 greedy + 5 sampled = 6 traces/question = 2880。命令 --samples-per-question 5。
3. 缓存（一遍前向,预注册 schema,smoke 锁定后才全量）:completion 文本、token logprobs、
   hidden_states[⌊0.7L⌋] 在 §8 位置的向量、identifier 字符 span→token 位置、
   completion-level eligible/label。**不缓存 raw attention。**
4. 统计单位 = completion-level primary（§4 population）;mention-level 仅作 observability 描述。
   CV 与 bootstrap 一律按 source prompt 分组;K completion 共享 group。
5. 标注只走 h1_identifier.label_completion;OntologyIndex 指纹与 4D-0 manifest 一致否则停。
6. gold id 只作 eval 标签,绝不进 prompt/feature;每条 record 过 assert_no_h1_gold_label_leakage。
7. 不做:H1 的 F1/F4、raw attention、TruthfulQA、非线性大模型、steering/patching、
   post-hoc 选层做增量声明、把 O 设成弱基线、把 K completion 当独立样本。
```

## 4. Population 定义（P0,completion-level）

```text
H1 primary population:生成 ≥1 个非 echoed ATT&CK/CWE identifier 的 completion。
  positive:其中 ≥1 个 identifier fabricated;negative:所有非 echoed id 均 grounded。
  每 completion 一条主样本(不是每 mention 一条)。
emission failure（拒答/无合法 id）:单独报告,不进主 AUROC,作 secondary end-to-end。
MCQ:每题一条,label = 正确性。
```

## 5. Stage 0：H1 全量生成（smoke 锁 schema 后,第一天开;单点故障）

```text
data/processed/h1/h1_samples.jsonl 全 480 题;每题 1 greedy + 5 sampled;max_new_tokens 512。
一遍前向落盘 §3.3 缓存契约（teacher-forcing 取 completion 区间 logprob;identifier span→token）。
~15–20 GPU-hr（4D-1 外推）,后台 + 幂等续跑。
产物:h1_generation_manifest.jsonl、h1_feature_cache.*（含 completion-level 主记录）。
```

## 6. Stage 1：输出侧 artifact 阶梯 → O（先跑,gate 依赖它的 OOF 分数）

```text
弱→强,同 grouped folds:
  1 prompt-only  2 id-string-only  3 surface-format-only  4 full-response-text
  5 F5  6 F5 + full-response-text = O（固定,不取 max）。
full-response-text（冻结,见 preregistration §7）:word ngram(1,2) + char_wb ngram(3,5) 两路
  TfidfVectorizer(min_df=2, max_features=50000/路, sublinear_tf=True) → L2-logistic;
  vocab 只 train fold 拟合;无外部 embedding;response-only。
统一训练协议（O/H/O+H 一致,防融合 artifact）:同 outer folds、同 nested C 网格
  {0.01,0.1,1,10}(只 train/dev 选);F5/H dense 特征 train-fold z-score;sparse 与 dense
  各自 block 标准化后拼接(逐块等权);报告 AUROC(O)/AUROC(H alone)/AUROC(O+H) 三者。
artifact 红线（数值化）:Δ_artifact = AUROC(full-text) − AUROC(shortcut),
  shortcut ∈ {id-string-only, surface-format-only};grouped-bootstrap CI95 ⊂ [−0.02,+0.02]
  → shortcut 与 full-text 实质等价 → 触发红线,该任务不做 hidden 增量声明。
产物:output_ladder_report.json（各阶梯 AUROC/AUPRC + Δ_artifact CI + 红线 + O 定义）。
```

## 7. Stage 2：observability gate（ladder 之后跑;在 hidden 增量结论之前）

```text
两任务同定义（不再 MCQ=F5、H1=O 的不对称）:
  S_t = max(0, 2·AUROC(O_t) − 1)（rank-biserial 非负截断;AUROC<0.5=失效,不翻转）。
  O_t 均为该任务自己的 F5 + full-response-text（MCQ 有 CoT 文本可构造）。
  AUPRC 只在各任务内报告,不作 gate。
D = S_MCQ − S_H1，用 **independent** grouped bootstrap（两任务 groups 各自独立重采样,
  每轮算 D_b;非 paired——两任务非同批 prompt）。95% CI。δ = 0.15（rank-biserial,
  = 2·AUROC−1;等价 AUROC 可分性差约 0.075）。判读:
  CI(D) 全 > δ  → h1_is_high_confidence_setting=True
  跨 δ/0        → 不确定
  全 ≤ 0        → Outcome 3（H1 未构造高置信错误,如实转写）
产物:observability_gate_report.json（两任务 S_t + D + CI + 判据）。
```

## 8. Stage 3：hidden probe H 与条件增量（RQ1）

```text
H:SAPLMA-style。
  层:block_index_zero_based = ⌊0.7L⌋（Qwen 19 / Llama 22）;
     HF hidden_states = (embedding, layer1..layerL),故 tuple index = block+1（Qwen 20 / Llama 23）;
     smoke 用 forward hook 核验 hook 输出 == hidden_states[tuple_index] 数值相等(非只对 shape)。
  位置 primary:H1 = completion 内**所有**非 echoed ATT&CK/CWE identifier 的**全部 token**
     在该层 hidden 的统一 mean pooling（不是只取第一个 id;只取第一个另作附录）;
     MCQ = answer-letter token 自身。post-hoc（表示已含生成 token）。
  → grouped-CV L2-logistic（协议同 §6）。禁事后选层（附录 {0.5,0.7,final},primary 恒 ⌊0.7L⌋）。
Δ_H = AUROC(O+H) − AUROC(O)，paired grouped-bootstrap（按 prompt,≥1000）95% CI;AUPRC 增量并报。
equivalence margin ε=0.02。判读:CI 全>+ε 有实质增量 / ⊂[−ε,+ε] 与无增量等价 /
  全<−ε 有害 / 跨界 inconclusive。
SE:H1 退化为 id_agreement_rate,error-mode baseline,不当独立 probe;EigenScore 仅复现≤2 天加。
主表:| Task | O | H alone | O+H | ΔH AUROC CI | ΔH AUPRC CI |
产物:increment_report.json。
```

## 9. Stage 4：RQ2（observability effect-modification,描述性）

```text
对全部 eligible completions（含 pos+neg）按 id-token logprob 分层:
  阈值 = train fold 内 eligible 的 id-token-logprob 中位数（固定分位数,同阈值用于该 fold test）。
  high-confidence(≥中位) / low-confidence(<中位) 两层。
每层 O/H/O+H 与 Δ_H（分层 paired bootstrap CI）+ 层间差 CI。
最小样本:每层 pos≥15 且 neg≥15,否则记 insufficient,不出该层 AUROC。
结论强度:全体正例 < 30 → RQ2 标 exploratory / low-power,报每层 n + 宽 CI,不作强主张。
定位:描述性 effect-modification,非因果;分层变量 ∈ O,文中声明。
粗对照:Δ_H(MCQ) vs Δ_H(H1),标 2 点趋势、非 interaction。
产物:rq2_observability_report.json。
```

## 10. Stage 5：第二模型核心复核（Llama-3.1-8B）

```text
只跑核心 O / H / O+H,{MCQ, H1};跨模型协议固定（同 ⌊0.7L⌋/pooling/position/正则搜索,
  不分别调层）。回答"无增量是否 Qwen 特例"。产物:cross_model_core_report.json。
```

## 11. 输出清单

```text
outputs/logs/sprint_4D_2_conditional_increment/
  preflight_report.md（含 G1 CFP / G2 prereg sha256 / G3 smoke 结果）
  smoke_report.md
  h1_generation_manifest.jsonl / h1_feature_cache.*
  observability_gate_report.json / output_ladder_report.json
  increment_report.json / rq2_observability_report.json / cross_model_core_report.json
  calibration_selective_report.json（secondary:ECE/Brier/risk-coverage）
  review_gate_conditional_increment.md
```

## 12. 实现与测试要求（P0:新代码,不能靠"先跑后补"）

```text
新建 scripts/sprint_4D_2_conditional_increment.py,子命令:
  preflight / smoke / generate / gate / ladder / increment / rq2 / cross_model。
新建 completion-level 组装 + SAPLMA H + full-text baseline + 分层 + rank-biserial gate 的模块/函数。
测试:
  1. completion-level population:每 completion 恰一条主记录;emission failure 不入主集。
  2. K traces 共享 group;grouped split/bootstrap 不泄漏（按 prompt）。
  3. hidden 层/位置定位（⌊0.7L⌋、identifier span mean、answer token）纯函数测试。
  4. full-text baseline:vocab 只 train fold 拟合,不泄漏 test。
  5. rank-biserial = 2·AUROC−1 与 D、bootstrap CI 纯函数测试。
  6. RQ2 分层含正负、单类层报 insufficient 的逻辑测试。
  7. gold-leakage 断言覆盖所有落盘 record。
  8. h1_f5_features 既有测试保持通过;full pytest 通过。
```

## 13. Review Gate（逐条答）

```text
0.  G1 CFP 确认?G2 preregistration sha256 冻结?G3 smoke 通过?（必须全 yes 才全量）
1.  observability gate:S_MCQ / S_H1（r_rb）?D 与 CI?δ=0.15 判读?Outcome 3?
2.  artifact 红线:id-string-only / surface-only AUROC?是否接近 full model?
3.  O=F5+full-text 定义?各阶梯 AUROC?full-text 的 vocab 是否只 train fold?
4.  H（l=⌊0.7L⌋,post-hoc）AUROC?Δ_H AUROC + paired grouped CI?对 ε=0.02 的判读?
5.  Δ_H AUPRC CI + base-rate 线?
6.  RQ2:两层各 Δ_H + 层间差?每层 pos/neg 是否 ≥15?是否有单类层?（描述性声明）
7.  Δ_H(MCQ) vs Δ_H(H1)（标 2 点趋势）?
8.  SE 在 H1 是否退化为 id-agreement、漏稳定 fabrication?
9.  Llama-core 方向?跨模型协议固定、未分别调层?
10. 单位 completion-level?按 prompt 分组?K 未当独立样本?emission failure 单列?
11. 是否碰 H1-F1/F4、raw attention、TruthfulQA、非线性大模型、post-hoc 选层、弱 O?（全 no）
12. gold id 只作 eval、从未进 prompt/feature?leakage 测试覆盖?（yes）
13. 是否声称 hallucination reduction / accuracy?（no）
```

## 14. 推荐命令

```bash
conda run -n recover_attention python -m pytest \
  tests/test_h1_f5_features.py tests/test_h1_identifier.py tests/test_conditional_increment.py -q

# smoke（少量 prompt,锁 schema;G3）
conda run -n recover_attention python scripts/sprint_4D_2_conditional_increment.py \
  --stage smoke --samples-jsonl data/processed/h1/h1_samples.jsonl \
  --model-path <QWEN_8BIT> --smoke-prompts 20 --hidden-rel-depth 0.7 --no-cache-attention \
  --output-dir outputs/logs/sprint_4D_2_conditional_increment

# 全量生成（G1–G3 全过后）
conda run -n recover_attention python scripts/sprint_4D_2_conditional_increment.py \
  --stage generate --samples-per-question 5 --max-new-tokens 512 \
  --hidden-rel-depth 0.7 --no-cache-attention \
  --output-dir outputs/logs/sprint_4D_2_conditional_increment

# 分析
conda run -n recover_attention python scripts/sprint_4D_2_conditional_increment.py \
  --stage ladder,gate,increment,rq2,cross_model \
  --cv-grouped-by source_prompt --bootstrap 1000 --equiv-margin 0.02 --obs-delta 0.15 \
  --output-dir outputs/logs/sprint_4D_2_conditional_increment
```

## 15. 完成后更新

```text
docs/paper/preregistration.md（回填 §12 结果占位）
docs/paper/six_week_plan.md（标进度）
PROGRESS.md / docs/progress/sprint_4_history.md
```
