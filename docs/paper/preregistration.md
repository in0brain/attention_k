# Pre-registration：内部 hidden-state probe 的条件增量检验（v2）

> 投稿前预注册。跑任何全量生成之前写死;跑完只填结果,不改判读规则。
> v2 相对 v1 的修订(P0):完成 completion-level population 定义、RQ2 改为对全部
> eligible 分层、observability gate 量化(rank-based,弃用 AUPRC 作 gate)、O 固定
> 单一定义 + text baseline 落地、hidden 层/位置精确化、K 固定、加 equivalence margin、
> 启动 gate 由代码强制。**待目标 workshop CFP 确认 deadline / page limit / archival。**

## 0. thesis

```text
Decodability ≠ Incremental Utility。
supervised hidden-state probe 能把对错分开,不等于它在最强输出侧基线
(F5 + 可见文本)之上还有增量;内部 probe 的价值可能取决于
output-space error observability(错误在输出置信信号中的可观测性)。
```

## 1. 与现有工作的边界(核实后,真实)

```text
LM-Polygraph(2311.07383/2406.15627):统一 UQ 框架。→ 不占条件增量问题。
PARALLAX(2605.17028):22 方法×12 模型×6 数据,揭示 benchmark 构造 artifact。
  → 缩窄"大矩阵"创新;未把同协议 O/H/O+H 的 Δ_H 作主问题。
2605.27016:uncertainty ≠ hallucination,随错误类型变化。→ 未答"O 已知后 H 有无增量"。
2606.10198:label-scarce selective prediction + trajectory detector。→ 不占条件增量。
干净空间:把 hidden 检测评价改写为"相对最强输出侧基线的条件增量",
  并对照输出易观测错误(MCQ)与高置信 fabrication(H1),检验增量是否随 observability 变化。
```

## 2. 启动 gate（代码强制,未过不许跑生成）

```text
G1 目标 workshop 的 CFP 已确认(deadline / page limit / archival),记入 preflight。
G2 本文件已冻结:preflight 记录本文件的 sha256,跑后不许改判读规则/定义。
脚本 preflight 必须显式检查 G1、G2,缺任一 → 停。旧脚本不检查,不得用于本实验。
```

## 3. 统计单位与 population（P0 修正:completion-level）

```text
primary population(H1):生成 ≥1 个非 echoed ATT&CK/CWE identifier 的 completion。
  positive:其中 ≥1 个 identifier fabricated。
  negative:所有非 echoed identifier 均 grounded。
每个 completion = 一条主样本(不是每 mention 一条)。
emission failure(拒答 / 未生成合法 id):单独统计,不进主 completion-level AUROC,
  作 secondary end-to-end 系统分析。
mention-level:仅作 secondary,用于 id-token logprob 的 observability 描述,不当 primary。
MCQ population:每题一条,label = 答案正确性 1[â=a*]。
分组:CV 与 bootstrap 一律按 source prompt 分组;K 条 completion 共享 group,不当独立样本。
K 固定:1 greedy + 5 sampled = 6 traces/question = 480×6 = 2880 traces。
seed:固定预注册 grouped folds + paired grouped bootstrap;不把 classifier seed 当 robustness。
```

## 4. RQ1（条件增量）

```text
O = F5 + full-response-text（固定单一定义,不取 max,不按测试表现选,避免 winner's curse）。
  另分别报告 F5 / text-only / F5+text,仅作展示,不改 O。
H = 预注册 hidden-state probe（§6）。
Δ_H = AUROC(O+H) − AUROC(O)，paired grouped-bootstrap（按 prompt,≥1000）95% CI。
  AUPRC 增量并报（类不平衡）。
equivalence margin（预注册）:ε = 0.02 AUROC（最小有意义增量,量级参照 4C 噪声）。
判读规则（三/四选一,不预设方向）:
  CI 全 > +ε        → 有实质增量
  CI ⊂ [−ε, +ε]     → 与"无增量"实质等价（TOST 式,本方向的主要预期）
  CI 全 < −ε        → 内部特征显著损害
  跨界              → inconclusive（样本/功率不足,如实写）
```

## 5. RQ2（observability 的 effect-modification,描述性）

```text
分层对象 = 全部 eligible completions（同时含 positive 与 negative,不只分 fabrication）,
  否则单类层 AUROC 无法计算。
分层变量 = O 侧的 id-token logprob（mean over identifier span）。
阈值 = 固定分位数:在 train fold 内取 eligible 的 id-token-logprob 中位数,
  ≥ 中位数 → high-confidence 层,< → low-confidence 层（同一阈值用于该 fold 的 test）。
每层各算 O / H / O+H 与 Δ_H（分层的 paired grouped bootstrap CI）+ 层间差 CI。
最小样本:每层 positive ≥ 15 且 negative ≥ 15,否则记 insufficient,不出该层 AUROC。
定位:描述性 effect-modification,非因果证明;且分层变量本身 ∈ O,须在文中声明。
粗对照:Δ_H(MCQ) vs Δ_H(H1),明确标 2 点趋势、非可拟合函数、非 interaction 检验。
```

## 6. 前置 observability gate（4D-2 Stage 1,先于任何 probe bake-off；P0 量化）

```text
目的:先证 H1 确是"高置信 fabrication",否则核心解释变量没被构造出来。
每个任务内的输出风险分数:MCQ = F5 的 error 分数;H1 = fabrication 的 O 侧风险分。
可分性用 base-rate 稳健的 rank 效应量:
  rank-biserial r_rb = 2·AUROC_output − 1 ∈ [−1,1]（等价 Cliff's delta,base-rate 不敏感）。
  S_t = |r_rb,t|。   （AUROC 是 rank 统计、base-rate 不变;AUPRC 依赖正类比例,弃用作 gate,
                       仅在各任务内单独报告。）
D = S_MCQ − S_H1，按 prompt 分组 bootstrap 95% CI。
预注册阈值 δ = 0.15。判读:
  CI(D) 全 > δ   → H1 显著更不易由输出信号观测（h1_is_high_confidence_setting=True）
  CI 跨 δ / 0    → observability 差异不确定
  CI 全 ≤ 0      → 不支持 H1 更难观测（Outcome 3）
Outcome 3（如实报告）:H1 output-only 仍强可分 → H1 未构造高置信错误;结论转为
  "两个可构造任务上输出错误都可观测,内部增量在两者都近零",不得假装 H1 是高置信设置。
```

## 7. 输出侧 baseline 阶梯 + O 的实现（artifact 控制,受 PARALLAX 启发）

```text
阶梯（从弱到强,全部 grouped-CV logistic,同 folds）:
  1 prompt-only    2 id-string-only    3 surface-format-only（长度/模板/ontology 前缀）
  4 full-response-text    5 F5    6 F5 + full-response-text = O
full-response-text 实现（固定,不许事后改）:
  char(3–5 gram) + word 的 TF-IDF → L2-logistic;
  vocabulary 只在 train fold 拟合;不用任何外部 LLM embedding;
  prompt-only / response-only / prompt+response 分清（本 baseline 用 response-only）;
  正则化 C 固定或只在嵌套 train/dev 选，不看 test。
F5 复用 h1_f5_features.py:label margin/entropy、id-token mean/min logprob、
  id_agreement/self-consistency、verbalized confidence。
artifact 红线:若阶梯 2 或 3 已接近 full model → 记"H1 被字符串/格式模式解决",
  该任务不做 hidden 增量声明。
```

## 8. Hidden probe H（P0 精确化,预注册,防 post-hoc 选层）

```text
层:l = ⌊0.7·L⌋。Qwen2.5-7B L=28 → block 19;Llama-3.1-8B L=32 → block 22。
  HF hidden_states 约定:index 0 = embedding 输出,第 k 个 Transformer block 输出 = index k;
  取 hidden_states[l]（= block l 输出）。preflight 打印实际张量形状核对。
位置与 pooling:
  MCQ = answer-letter token 自身在 layer l 的 hidden。
  H1  = identifier token span 在 layer l 的 hidden 的 mean（completion 内多 id 时,
        取第一个非 echoed ATT&CK/CWE identifier 的 span;写死"第一个"）。
性质声明:这是 post-hoc detection——表示已含生成出的 token（A/B/C/D 或 identifier）,
  是"生成后内部信号",不是生成前 early-warning。论文明确这一点,不许跑后再选位置。
方法:SAPLMA-style（上述固定表示 → grouped-CV L2-logistic）。
robustness 附录:相对深度 {0.5, 0.7, final} 各报一版,primary 恒 0.7L,增量声明只基于 primary。
可选:EigenScore（INSIDE 2402.03744,跨采样 mid-layer 协方差谱），仅复现 ≤2 天才加。
Semantic Entropy:H1 上"语义等价"退化为归一化精确 id 匹配 = id_agreement_rate,
  预期漏稳定 fabrication;作 error-mode 采样 baseline,不当独立 probe。
```

## 9. 模型

```text
Qwen2.5-7B:全套（阶梯 + H + observability + RQ2 + 可选 SE/EigenScore）。
Llama-3.1-8B:只跑核心 O / H / O+H,复核方向。
跨模型协议固定:同相对层深(⌊0.7L⌋)、同 pooling、同 position 定义、同正则搜索、
  同 grouped split;不对两模型分别找最优层。
```

## 10. 明确删除 / 留 archival

```text
H1 的 F1/F4:F4 是有限标签 exact VJP,H1 无固定标签集,不重定义不能用。删。
raw attention 全缓存:体量过大且无 attention 方法。删。
TruthfulQA:archival 再加,且只用 live-generation protocol（自由生成 + 事后 judge,
  reference 不进输入）——teacher-forced 版有表面 artifact（PARALLAX）。
"calibration"作核心术语:改 output-space error observability;
  ECE/Brier/risk-coverage 降为 secondary selective-prediction 交付,不作 AUROC 解释。
```

## 11. 缓存契约（一遍前向,预注册;跑前用 smoke 锁定 schema）

```text
每条 trace 一次前向,落盘:
  completion 文本、token logprobs、hidden_states[⌊0.7L⌋] 在 §8 位置的向量、
  identifier 字符 span → token 位置、completion-level eligible/label 字段。
不缓存 raw attention。
schema 冻结要求:completion-level 主记录、hidden 层/位置/pooling、no-emission policy
  必须在 smoke（少量 prompt）通过后才锁;全量 2880 前向依赖此 schema,跑后改要重做。
复用 h1_f5_features.py（completion 级聚合需在其上新增 completion-level 组装,不改既有函数语义）。
```

## 12. 最多允许的结论

```text
在 MCQ（输出易观测错误）与 H1 fabricated-identifier（经 §6 gate 判定的高置信设置或
Outcome 3）上,对最强输出侧基线 O = F5 + 可见文本,supervised hidden-state probe H 的
条件增量 Δ_H = [填],95% paired grouped-bootstrap CI = [填],相对 equivalence margin
ε=0.02 判读为 [有实质增量 / 与无增量等价 / 有害 / inconclusive];RQ2 分层显示 Δ_H
是否随 observability 变化 = [填]。此为条件增量结论,不是幻觉减少 / 准确率 / intervention。
```
