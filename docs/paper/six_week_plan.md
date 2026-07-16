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

## W0(约 7/15–7/18):锁定,不跑 compute

```text
1. 确认目标 workshop 的真实 CFP(deadline / page limit / archival)。名单未出就等。
2. 完成 preregistration.md 的最终审阅并冻结(thesis / RQ1-2 / O,H / 判读规则 /
   前置 gate / 单位 / artifact 阶梯)。
3. 文献占位矩阵:LM-Polygraph / PARALLAX / 2605.27016 / 2606.10198 各占什么、
   你留什么,写进 related work 草稿。
产物:确认的 CFP、冻结的 preregistration、related-work 占位表。
门:workshop 未定 + preregistration 未冻 → 不启动 4D-2。
```

## W1(约 7/19–7/25):前置 gate + H1 全量生成 + MCQ 阶梯

```text
Day1 立刻起 4D-2 生成(H1,Qwen,K=5~6,512 tok,一遍缓存 §10 契约的特征,
  不缓存 raw attention)。~15–20 GPU-hr。单点故障,第一天必须开。
并行(不等生成):
  - 在已有 MCQ(4C)数据上跑输出侧阶梯 1–6 + SAPLMA(H),先把 pipeline 打通。
生成完立即跑【前置 observability gate】:
  MCQ 与 H1 各自 error-vs-correct 的 confidence 分布可分性(Cohen's d / 直方图 / AUPRC)。
  判据:H1 重叠 > MCQ。若不成立 → 记 Outcome 3,后续按"两任务都可观测"写。
产物:前置 gate 报告(H1 是否高置信设置);MCQ 全阶梯 + SAPLMA 第一版。
算力:~$10–14。
cut-line:4D-2 生成到周末没干净出 → 砍所有加模型/加方法,保 Qwen×{MCQ,H1} 核心。
```

## W2(约 7/26–8/1):H1 增量 bake-off(核心)

```text
在 H1 上跑:输出侧阶梯(prompt/id-string/surface/full-text/F5/F5+text=O)+ SAPLMA(H);
  算 Δ_H = AUROC(O+H) − AUROC(O),paired grouped-bootstrap CI(按 prompt 分组)。
artifact 红线检查:id-string-only / surface-only 是否已接近 full model。
SE(退化为 id-agreement)跑,作 error-mode baseline;EigenScore 仅在复现 ≤2 天时加。
主表:| Task | O(=最强输出侧) | H | O+H | Δ_H AUROC CI | Δ_H AUPRC CI |
产物:MCQ + H1 的完整增量主表 + CI。
算力:~$8–10。
cut-line:EigenScore 卡壳就搁置(future work),保 F5 阶梯 + SAPLMA + SE。
```

## W3(约 8/2–8/8):RQ2 分层 + CoT 基线收尾 + 开写

```text
RQ2 within-H1:按 id-token logprob 把 fabrication 分高置信/低置信两层,
  各层 Δ_H + 层间差(paired bootstrap)。这是 RQ2 有统计力的一半。
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
