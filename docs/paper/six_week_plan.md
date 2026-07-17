# 6 周计划:条件增量 workshop 短文

> 配套 `preregistration.md`。所有日期暂定,待目标 NeurIPS 2026 workshop 的 CFP
> 确认真实 deadline / page limit / archival 后反推 freeze。8/29 是 NeurIPS 给
> organizer 的**建议**日期,不是硬 deadline;各 workshop 自定 CFP。

## 命题(锁定)

```text
原:  比较四种检测方法谁更强(会被 LM-Polygraph / PARALLAX 占掉)。
终:  supervised hidden-state probe 在最强输出侧基线(F5 + 可见 CoT/文本)之上
      是否有条件增量,且增量是否随 output-space error observability 变化。
```

## W0(约 7/15–7/18):锁定设计,不跑 compute

```text
1. 确认目标 workshop 的真实 CFP(deadline / page limit / archival)。名单未出就等。
2. 冻结 preregistration.md v2(已按 P0 修:completion-level population、RQ2 分全部 eligible、
   observability gate 量化 rank-biserial+δ、O 固定 F5+text、hidden 层/位置精确、K 固定、
   equivalence margin ε=0.02)。记录其 sha256。
3. 文献占位矩阵:LM-Polygraph / PARALLAX / 2605.27016 / 2606.10198 各占什么、你留什么。
产物:确认的 CFP、冻结的 preregistration(带 sha256)、related-work 占位表。
门:preregistration 未冻 → 不进 W0.5。（workshop CFP 未确认**不**阻塞 W0.5:
  实现新脚本 + ≤20 prompt smoke 可以先做;只有 Stage 0 全量 2880 生成才需要 CFP+冻结+smoke 齐备,
  以免因 workshop 名单未公布而浪费实现时间。）
```

## W0.5(约 7/18–7/21):实现 + smoke 锁 schema,不跑全量

```text
先有代码再有数据:新建 scripts/sprint_4D_2_conditional_increment.py(子命令
  preflight/smoke/generate/gate/ladder/increment/rq2/cross_model)+ completion-level
  组装 / SAPLMA H / full-text baseline / 分层 / rank-biserial gate 模块。
在 ≤20 prompt 上 smoke,验证:每 completion 恰一条主记录、K traces 共享 group、
  hidden 层/位置正确、O/H/O+H 同 folds、no-emission 处理、paired CI 可算。
产物:通过的 smoke_report;锁定的缓存 schema;新 pipeline 单元测试。
门:smoke 未过 → 不启动 Stage 0(2880 前向依赖此 schema,跑后改要重做)。
```

### W0.5-B 实况(2026-07-16,H1 侧完成)

```text
完成:统计/schema 核心 + preflight/smoke_synthetic/verify_hidden/smoke_model,
  H1 侧真实 20 prompt smoke 通过(115/120 eligible;hidden block19→tuple20,
  forward hook 与 hidden_states[20] 数值相等,max_abs_diff 0.00,8-bit backend)。
期间两次预注册修订(均在 Stage 0 之前,详见 preregistration.md 顶部):
  v2.2 三-block 等权融合(修 F5 被 3584 维 H 压制)
  v2.3 MCQ semantic output canonicalization + fresh 1760 confirmatory
gate 实况:G1 red(EIML3 页数待 7/29 公布) / G2 green(v2.3) / G3 red(v2.2 的 smoke
  已被 prereg_sha256 校验自动作废,须按 v2.3 重挣)。
```

## W0.5-C(当前):MCQ 侧实现 + 重挣 v2.3 的 G3

```text
唯一目标 = 重新拿到 G3。不启动任何正式实验。
链路:MCQ adapter(Stage B,确定性转换,不重采样)
  → selected-option semantic canonicalization(§7.1)
  → teacher-forced hidden re-forward(Stage C,只缓存 answer-letter token 的
    hidden_states[⌊0.7L⌋] + 必要 token logprobs;不缓存 raw attention/全层/F1/F4/J-lens)
  → v2.3 ladder / O / H / O+H(Stage D)
  → ≤20 pilot prompt smoke → 新 G3
门:新 G3 未过 → 不跑 fresh 1760 的任何正式结果;G1 未过 → 不启动 Stage 0。
```

## W1(约 7/22–7/28):前置 gate + H1 全量生成 + MCQ 阶梯

```text
smoke 与 preregistration 冻结后,Day1 起 4D-2 生成(H1,Qwen,1 greedy + 5 sampled
  = 6 traces/题 = 2880,512 tok,一遍缓存契约特征,不缓存 raw attention)。
  ~15–20 GPU-hr。单点故障,W0.5 通过后第一天必开。
并行(不等生成):
  - MCQ 侧:**不再复用 4C 的 239 题作 confirmatory**(v2.3)。4B-3/4C 已用的 240 题
    降级 exploratory(其 CI 促成了 v2.3 修订,再用于确证 = 同批数据既定设计又下结论);
    confirmatory = CyberMetric fresh 1760 题,O 用 selected-option semantic
    canonicalization(§7.1)。sizing 依据见 §7.2 与 mcq_asset_audit_and_v2.3_rationale.md。
    MCQ 生成很便宜(输出仅 1 字母),瓶颈是 1760 次 teacher-forced re-forward。
生成完先跑 ladder(得到各任务 O 的 OOF 分数),再跑【observability gate】(见 preregistration §6):
  S_t = max(0, 2·AUROC(O_t)−1)(两任务同用各自 F5+text);D = S_MCQ − S_H1,
  independent grouped bootstrap CI,δ=0.15。全>δ → H1 高置信设置;≤0 → Outcome 3。
产物:ladder 报告 + gate 报告(H1 是否高置信设置);MCQ 全阶梯 + SAPLMA 第一版。
算力:~$10–14。
cut-line:4D-2 生成到周末没干净出 → 砍所有加模型/加方法,保 Qwen×{MCQ,H1} 核心。
```

## W2(约 7/26–8/1):H1 增量 bake-off(核心)

```text
在 H1 上跑:输出侧阶梯(prompt/id-string/surface/full-text/F5/F5+text=O)+ SAPLMA(H);
  算 Δ_H = AUROC(O+H) − AUROC(O),paired grouped-bootstrap CI(按 prompt 分组)。
artifact 红线检查:id-string-only / surface-only 是否已接近 full model。
SE(退化为 id-agreement)跑,作 error-mode baseline;EigenScore 仅在复现 ≤2 天时加。
主表:| Task | O(=最强输出侧) | H alone | O+H | Δ_H AUROC CI | Δ_H AUPRC CI |
产物:MCQ + H1 的完整增量主表 + CI。
算力:~$8–10。
cut-line:EigenScore 卡壳就搁置(future work),保 F5 阶梯 + SAPLMA + SE。
```

## W3(约 8/2–8/8):RQ2 分层 + CoT 基线收尾 + 开写

```text
RQ2 within-H1:按 id-token logprob 把**全部 eligible completion**(含正负)分高置信/低置信
  两层(固定分位数阈值,每层 pos/neg≥15),各层 Δ_H + 层间差(paired bootstrap)。描述性
  effect-modification,非因果;分层变量 ∈ O。这是 RQ2 有统计力的一半。
把 full-response-text / F5+text 基线做扎实(artifact 阶梯完整),
  回答"hidden 是否比模型已写出的推理文本还多给信息"。
并行开写:intro + related work + method(不依赖最终结果,尤其 novelty 防守:
  条件增量 + observability,和 PARALLAX/2605.27016 的区分)。
产物:RQ2 分层结果;论文前三节初稿。
算力:~$6。
cut-line:时间紧先砍 EigenScore,CoT/text 基线不能砍。
```

## W4(约 8/9–8/15):第二模型核心复核

```text
Llama-3.1-8B 只跑核心 O / H / O+H(§8 固定跨模型协议,不分别调层),
  在 {MCQ, H1} 上复核方向。~25–30 GPU-hr。
若 Qwen 已明确无增量:Llama 只需验证方向一致。
若 Qwen 结果混合:先做 error-type 分析(高/低置信、稳定/不稳定 fabrication),
  别急着堆模型。
产物:2 模型 × 2 任务的核心增量对照。
算力:~$12–15。
cut-line:Llama 跑不完 → 写 single-model preliminary,non-archival short 可接受。
```

## W5(约 8/16–8/22):分析 + robustness + 冻结准备

```text
补 secondary:ECE / Brier / risk-coverage / accuracy-at-fixed-coverage(selective 交付);
  SAPLMA 相对深度 {0.5,0.7,final} robustness 附录(primary 恒 0.7L,不挑最优)。
终版图表:1 张增量主表 + 1 张 observability(前置 gate 分布)图 + RQ2 分层图。
写 results / discussion / limitations(诚实:2 任务=2 点对照、单/双模型、样本量、
  Outcome-3 风险、artifact 红线结果)。
产物:冻结候选结果、终版图表、后半稿。
算力:~$5。
```

## W6(freeze 后):定稿投出

```text
freeze:目标 workshop 真实 deadline 前至少 7 天,冻结所有实验,只改文字。
压到 page limit、内部 review、附录(preregistration、复现细节、artifact 阶梯)。
提交。
```

## 必做核心(怎么砍都要保)

```text
Qwen × {MCQ, H1}
输出侧阶梯 → O(F5 + full-text)
SAPLMA-style H
Δ_H = AUROC(O+H) − AUROC(O),paired grouped-bootstrap CI(按 prompt 分组)
前置 observability gate(证明 H1 是高置信设置,或如实记 Outcome 3)
RQ2 within-H1 分层
artifact 红线(id-string-only / surface-only 不能接近 full model)
```

加分叠加顺序(落后就从上往下砍):Llama-core → SE/EigenScore → risk-coverage →
robustness 附录。**TruthfulQA / 全量矩阵 / raw attention / F1-F4-on-H1 一律不做。**

## 三个提醒

```text
1. 4D-2 生成是单点故障,W1 Day1 必开,一遍缓存到位。
2. O 必须是输出侧最强(含可见文本),不是只有 F5,否则 Δ_H 可被弱 O 刷出假象。
3. 算力不是约束(全程 ~$70–90);瓶颈是复现质量、artifact 控制、你的写作时间。
```
