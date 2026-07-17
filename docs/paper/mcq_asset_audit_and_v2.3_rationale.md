# MCQ 资产审计与 v2.3 修订依据（Stage A 报告）

> 时间:2026-07-16。Stage 0 未启动,无任何正式结果。
> 本报告是 v2.3 修订的**事实依据**,记录"改之前看到了什么"。
> 结论权威:`preregistration.md` v2.3（sha256 `16fa43db…`）。

## 0. 为什么做这次审计

4D-2 的 §6 observability gate 是 `D = S_MCQ − S_H1`,需要 MCQ 侧的 O 风险分。
W0.5-B 只实现了 H1 侧,MCQ 侧完全缺失。启动 Stage 0 前必须判定:
**4B/4C 的既有 MCQ 产物,范围与内容够不够支撑已冻结的设计?**

审计问的是"产物能否支持设计",不是"结果好不好"。两个发现都与 Δ 的方向无关。

## 1. Stage A：资产清单

来源:`outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/trace_sampling_manifest.jsonl`
（1680 traces = 240 题 × 7）、`outputs/logs/sprint_4C_.../readout_feature_manifest.jsonl`（239 行）。

| 字段 | 有无 | 位置 |
| --- | --- | --- |
| question_id / source_prompt | 有 | `example_id` / `group_id`（240 题 = 240 group，1:1） |
| 完整 prompt | 有 | 4B-3 `prompt`（chat template 已渲染） |
| gold answer letter | 有 | `gold_label` |
| generated answer letter | 有 | `parsed_label`（`parse_method`、`parse_failure` 齐备） |
| correctness label | 有 | `is_correct`(4B-3) / `wrong_label`(4C) |
| 可重新 tokenize 的文本 | 有 | prompt + completion 均为文本 |
| generation prompt version | 有 | `prompt_style='chat'`（4B-2 A/B 的胜出条件） |
| **完整 response / CoT** | **无** | 见 §2 |
| F5 原始字段 | 部分 | 4C 仅 3 项:`f5_label_margin` / `f5_label_entropy` / `f5_full_entropy` |
| model / tokenizer revision | 弱 | 只记 `model_path`，无 commit/revision hash |
| hidden state | 无 | 可 re-forward，成本小，非阻塞 |

## 2. 发现一：MCQ 没有 response text（表示层事实）

预注册 v2.2 §7 写明：

> O_t 均为该任务自己的 F5 + full-response-text（**CyberMetric prompt 要求先简短推理再给字母**，MCQ 也有文本可构造 text baseline）

**该断言与产物事实不符。** 240 条 greedy 实测：

```text
has_reasoning_text  = 0 / 240
completion 字符数    min=1  p50=1  p90=1  max=11
唯一取值            {B:69, C:63, D:58, A:49, 'Answer! <D>':1}
```

`reasoning_substrate_report.json` 对全部 1680 traces 的统计一致：
`has_reasoning_text_rate = 0.0`，completion 字符数均值 1.10、token 均值 2.03。

即 **MCQ 的 "full-response-text" 就是一个字母**。后果是连锁的：

1. text block 退化成 {A,B,C,D} 上的 4 值类别 = 答案字母身份 → `O_MCQ ≈ F5`。
2. §6 想消除的"MCQ=F5、H1=O"不对称，在**定义上**消失、在**事实上**回归。
3. ladder 塌陷：rung2(id-string-only) ≡ rung4(full-text) ≡ 答案字母。
4. §7 artifact 红线 `Δ_artifact = AUROC(full-text) − AUROC(shortcut)` 因构造必然触发
   （两者同一），判定"该任务被字符串/格式模式解决"在此无意义。

**"重跑生成把文本补上"不是退路** —— 4B-3 Stage B' 已试过强制推理并否决：

```text
has_reasoning_text = 0.656   ← 文本确实出来了
parse_failure      = 0.3125
wrong_rate         = 0.0     ← 全对，零正例
num_questions_with_correct_and_wrong = 0
admitted_for_f2    = false
```

强制推理后**没有错误样本**，AUROC 无从计算。

## 3. 发现二：n=239 的精度结构性不足（精度事实）

4C 在**完全相同的 n** 上已经算过同类增量 CI：

| 对比 | increment | CI95 | 宽度 |
| --- | --- | --- | --- |
| C_F5_F1 vs F5 | −0.0179 | **[−0.04288, +0.02021]** | **0.0631** |
| A_F5_alone OOF AUROC | 0.8343 | [0.7760, 0.8862] | 0.1101 |
| f5_single_forward_reference | 0.8156 | [0.7462, 0.8714] | n=239, 正例 47 |

§4 的**主要预期**是 `CI ⊂ [−ε,+ε] = [−0.02,+0.02] → 与无增量实质等价`。
实测增量 CI 宽度 **0.0631 是等价带 0.04 的 1.6 倍**，装不进去 → 按 §4 只能读作 **inconclusive**。

注意 `Δ = −0.0179` 不是噪声，是真实现象：加无用特征会因正则化稀释而掉分。
这正是 v2.2 三-block 等权要压住的东西。

## 4. sizing（先验，非保证）

判据见 §7.2。零假设 `s_o = d·y+e1`、`s_oh = d·y+e2`、`corr(e1,e2)=rho`，`E[Δ]=0`。
rho 由 4C 锚点标定（n=239 → CI 宽度 0.0631，标定后模拟宽度 0.0631，`rho=0.9051`）。

MCQ（每题一条，1 行/组）：

| n | 正例 | CI 宽度 | equivalent 可达率 |
| --- | --- | --- | --- |
| **239（现有）** | 47 | 0.0631 | **0%** |
| 500 | 98 | 0.0425 | 3% |
| 800 | 157 | 0.0341 | 22% |
| 1000 | 196 | 0.0303 | 55% |
| 1500 | 294 | 0.0243 | 77% |
| **1760（fresh 全池）** | 346 | 0.0223 | **87%** |

H1（K=6 traces/prompt，标签聚集用 4D-1 **实测**做 beta-binomial：ICC=0.261、
completion 级正例率 0.149，来自 201 eligible completion / 55 question）：

| prompt 数 | 总 traces | CI 宽度 | equivalent 可达率 |
| --- | --- | --- | --- |
| **480（现冻结）** | 2880 | 0.0210 | **95%** |
| 1000 | 6000 | 0.0144 | 100% |
| 1760 | 10560 | 0.0110 | 100% |

**故 v2.3 只动 MCQ 臂，H1 的 480×K=6 不变。**

模拟组级正例率 0.433 vs 4D-1 实测 0.345 —— 合理性校验通过：实测是 k≈3.9 traces/question，
K=6 时"至少一条 fabricated"的概率本就应更高。

### 4.1 sizing 过程中被修掉的三个错误（如实记录）

这些数经过三次修正才自洽，过程记录在案以便复核：

1. **v1 拍脑袋设 `rho=0.995`** → n=239 得出 98% 可达率，比 4C 实测乐观 3.7 倍。
   自由参数无锚点 = 结论是编的。改为用 4C 实测标定。
2. **v2 零假设构造有偏**：`s_oh = rho·s_o + noise` 会把信号从 d 削到 rho·d →
   `E[Δ] < 0` → "装不进等价带"分不清是**精度不够**还是**融合掉分**。
   症状:n=2000 时 CI 宽度 0.021 明明窄于带宽 0.04,可达率却只有 3%。
   改为同信号、仅噪声相关的等 AUROC 构造 → `E[Δ]≈0`（已验证）。
3. **H1 v1 把标签整个放在 group 级抽** = 强制标签 ICC=1 → 低估有效样本量 →
   给出 480 prompt 仅 8–30% 的假象。实测 ICC 只有 0.261 → 改用 beta-binomial
   建模后是 95%。**这次修正直接把 v2.3 的改动面砍掉一半（H1 不用动）。**

### 4.2 效力边界（不得当作已兑现的功效）

`rho=0.9051` 标定自 **4C 的 F1（7 维）、n=239、旧 4C 协议**，却被用于
**H（3584 维）、v2.2 三-block 协议**。两处不同且方向相反：

- 更高维 → 重训抖动可能更大 → 真 rho 更低 → CI 更宽 → 87%/95% **偏乐观**；
- v2.2 把 H block 缩放到行范数≈1 → 稀释应小于 4C 的朴素拼接 → 真 rho 可能**更高**。

净效应未知。**87% 与 95% 是有锚点的先验估计，不是实际功效**，代码与报告不得如此称呼。

可检验承诺（硬，见 §7.2）：Stage 0 后必须回报各臂 Δ_H 的**实际** CI 宽度；
`> 0.04` → 该臂如实记 inconclusive，**不得**追加样本、改 ε、换判读规则或改融合协议。

## 5. fresh confirmatory population

```text
CyberMetric 总池          = 2000  (data/processed/cyber/cybermetric.jsonl)
4B-3/4C 已用(烧掉)        =  240  (已核验:该 240 全在池内,划分无歧义)
fresh confirmatory        = 1760
```

数据集本身无预设 split（`split` 字段全为 None），故划分由本预注册定义。

**排除那 240 题是必需的**：4C 在它们上的 CI 正是促成 v2.3 的依据，再用它们做确证
= 同一批数据既定设计又下结论。它们降级为 exploratory，不进 confirmatory 结果。

验收须按 **ID 集合**校验，不能只看行数：

```text
|fresh ∩ burned| = 0
|fresh ∪ burned| = 2000
|fresh| = 1760
```

## 6. 本次修订不是"按结果调整实验"

进入决策的只有两个事实，都与 Δ 的方向无关：

- **表示层事实**：`has_reasoning_text = 0/240`。与结果好坏无关。
- **精度事实**：`CI 宽度 0.0631 > 等价带 0.04`。与 Δ 的**方向**无关（只与宽度有关）。

**旧 MCQ 的 AUROC 高低从未进入本次决策。** 促成修订的 4C 数值与那 240 题一并降级为
exploratory，不进入 confirmatory 结果。MCQ 的 shortcut/ladder 映射（§7.1）在看到任何
新结果之前定死。v2.1/v2.2 的 hash 与实现保留在 commit `57b7ae9` / `f93877b`，不覆盖。
