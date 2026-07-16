# Pre-registration：内部 hidden-state probe 的条件增量检验（v2.2，已冻结）

> 投稿前预注册。跑任何全量生成之前写死;跑完只填结果,不改判读规则。
> 冻结指纹见 `preregistration.lock`（sha256）。改设计须 bump version + 重算 hash。

## v2.2 修订记录（唯一变更，2026-07-16）

```text
变更:§7 融合协议。原 v2.1 只分 sparse / dense 两个 block,F5(d_F=14) 与
  H(d_H=3584) 同处一个 dense block。现拆成 text / F5 / H 三个独立 block,
  各按 §7 冻结公式等权。§7 的 late-fusion robustness 附录同时写入。
理由:v2.1 的两-block 实现与 §7 自身"逐块等权、避免高维块压制低维块"的动机内部
  不一致——它防住了 sparse 压制 dense,却放任 dense 内部 3584 维 H 压制 14 维 F5。
  这会让 AUROC(O+H) − AUROC(O) 混入工程混杂:
    "hidden 是否有增量" + "融合是否破坏了原有输出信号"
  若 O+H≈H 且显著低于 O,最直接的解读是"联合模型实现不公平",而非"内部状态无增量",
  直接损害本文主命题。
修订正当性(记录在案,防事后质疑):
  - Stage 0 未启动,无正式结果;
  - 缺陷在 smoke 的职责范围内被发现(smoke 的作用就是跑前锁 schema/协议);
  - 维度压制风险在查看 smoke AUROC **之前**已被指出,不是按结果方向选模型;
  - 修订消除的是协议内部不一致,不指向任何特定结论方向;
  - v2.1 原始 hash 与实现保留在 git commit 57b7ae9,不覆盖;
  - v2.1 smoke 的 AUROC 数值(O=0.843 / H=0.785 / O+H=0.721, n=115, n_pos=10)
    仅用于暴露该协议缺陷,**不进入正式论文结果**,也不证明 H 有效或无效。
后果:v2.1 的 G3 失效(融合协议已变),必须重跑单元测试与完整 ≤20 prompt smoke,
  取得新 G2 + 新 G3;Stage 0 仍需 G1。
```
> v2 相对 v1(P0):completion-level population、RQ2 对全部 eligible 分层、observability
> gate 量化(rank-based,弃 AUPRC 作 gate)、O 固定单一定义 + text baseline 落地、
> hidden 层/位置精确化、K 固定、equivalence margin、启动 gate 由代码强制。
> v2.1 相对 v2(P0):H 改 pool 全部 eligible identifier、hidden tuple index 消歧
> (block+1)、gate 两任务同用 F5+text + max(0,·) + independent bootstrap + ladder→gate、
> artifact 红线数值化、TF-IDF/融合协议全冻结、δ 标注、RQ2 低功率标记。
> **待目标 workshop CFP 确认 deadline / page limit / archival。**

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
H = 预注册 hidden-state probe（§8）。
Δ_H = AUROC(O+H) − AUROC(O)，paired grouped-bootstrap（按 prompt,≥1000）95% CI。
  AUPRC 增量并报（类不平衡）。融合协议(逐块标准化、同 folds、同 nested C)见 §7;
  报告 AUROC(O)、AUROC(H alone)、AUROC(O+H) 三者,防"无增量"是融合 artifact。
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
结论强度:全体正例 < 30 时,RQ2 整体标为 exploratory / low-power,报告每层 n 与(宽)CI,
  不作强主张（min=15 保证可算,不保证稳定）。
定位:描述性 effect-modification,非因果证明;且分层变量本身 ∈ O,须在文中声明。
粗对照:Δ_H(MCQ) vs Δ_H(H1),明确标 2 点趋势、非可拟合函数、非 interaction 检验。
```

## 6. observability gate（P0 量化;顺序 = ladder 之后、hidden 增量结论之前）

```text
目的:证 H1 确是"高置信 fabrication",否则核心解释变量没被构造出来。
顺序:gate 需各任务 O 的 OOF 风险分,故必须在 output ladder（§7）跑完之后算。
  "前置"指"在 hidden 增量结论之前",不指"在 ladder 之前"。
  执行顺序 = generate → ladder → gate → increment → rq2。
两任务同定义（P0:不再 MCQ=F5、H1=O 的不对称）:
  S_t = max(0, 2·AUROC(O_t) − 1)   （rank-biserial 非负截断;AUROC<0.5=失效/方向异常,
                                     不翻转成"强可观测"）。
  O_t 均为该任务自己的 F5 + full-response-text（CyberMetric prompt 要求先简短推理再给字母,
  MCQ 也有文本可构造 text baseline）。AUPRC 只在各任务内报告,不作 gate。
D = S_MCQ − S_H1，用 independent grouped bootstrap（MCQ groups 与 H1 groups 各自独立
  重采样,每轮算 D_b;**不是 paired**——两任务非同批 prompt、无天然配对;paired 只用于
  同任务同批样本的 O vs O+H）。95% CI。δ = 0.15（rank-biserial;因 rank-biserial =
  2·AUROC−1,δ=0.15 等价于两任务 AUROC 可分性相差约 0.075）。判读:
  CI(D) 全 > δ   → H1 显著更不易由输出信号观测（h1_is_high_confidence_setting=True）
  CI 跨 δ / 0    → observability 差异不确定
  CI 全 ≤ 0      → Outcome 3
Outcome 3（如实报告）:H1 output-only 仍强可分 → H1 未构造高置信错误;结论转为
  "两个可构造任务上输出错误都可观测,内部增量在两者都近零",不得假装 H1 是高置信设置。
```

## 7. 输出侧 baseline 阶梯 + O 的实现（artifact 控制,受 PARALLAX 启发）

```text
阶梯（从弱到强,全部 grouped-CV logistic,同 folds）:
  1 prompt-only    2 id-string-only    3 surface-format-only（长度/模板/ontology 前缀）
  4 full-response-text    5 F5    6 F5 + full-response-text = O
full-response-text 实现（P0 完全冻结,不许事后改）:
  TfidfVectorizer:word ngram_range=(1,2) + char_wb ngram_range=(3,5) 两路拼接;
  min_df=2、max_features=50000/路、sublinear_tf=True;vocabulary 只在 train fold 拟合;
  不用任何外部 LLM embedding;本 baseline 用 response-only(prompt/response/prompt+response
  另作附录,不改 O)。
统一训练协议（O / H / O+H 完全一致,防融合 artifact）:
  相同 outer grouped folds、相同 nested-CV 的 L2 C 网格{0.01,0.1,1,10}(只在 train/dev 选)、
  相同训练样本、相同 LogisticRegression;所有标准化统计量只在 train fold 拟合。
  报告 AUROC(O)、AUROC(H alone)、AUROC(O+H) 三者——若 H alone 强但 O+H≈O,
  则增量被融合掩盖,须讨论,不得直接判"无增量"。

融合公式（v2.2 P0 冻结到公式层面;三个独立 block,不是两个）:
  每个 outer fold 内,只用该 fold 的 train 统计量。
  text block:  word 与 char_wb TF-IDF 各自在 train fold 拟合,拼接后对**每条样本**做整体
               L2 normalization → X̃_T = L2Normalize([X_word, X_char])。整块每行范数 ≈ 1。
  F5 block:    逐特征 train-fold z-score Z_F = (X_F − μ_F)/σ_F,再除以维数平方根
               → X̃_F = Z_F / sqrt(d_F)。当前 d_F = 14。
  H block:     同样逐特征 z-score Z_H,再按维数归一 → X̃_H = Z_H / sqrt(d_H)。
               Qwen d_H = 3584。
  于是:  O = [X̃_T, X̃_F]      H = X̃_H      O+H = [X̃_T, X̃_F, X̃_H]
  三块的期望能量(每行范数 ≈ 1)因此处于相近量级,不会因 3584 ≫ 14 让 H 在统一 L2
  正则下支配模型。σ_F/σ_H 的常数列(σ<1e-12)用 1 代替,避免除零。

late-fusion robustness（附录,非 primary;防"高维拼接不公平"的质疑）:
  primary 恒为上面的三-block early fusion。附录另报分数级融合:
    O_score = f_O(O)，H_score = f_H(H)，f_late(O_score, H_score) = 两维 meta-logistic。
  meta 必须**嵌套在 outer fold 内**:在该 fold 的 train 上再做一层 inner cross-fitting
  得到 train 侧 OOF 的 O_score/H_score 来训 meta,再用整个 train 拟合的 f_O/f_H 打分
  test 后送入 meta。禁止用全数据 OOF score 训 meta 后又评估同一批样本(那是泄漏)。
  判读:early 与 late 都无增量 → 负结果更强;仅 early 失败 → 是融合结构问题,
  不得归因于 hidden state。
F5 复用 h1_f5_features.py:label margin/entropy、id-token mean/min logprob、
  id_agreement/self-consistency、verbalized confidence。
artifact 红线（P0 数值化）:Δ_artifact = AUROC(full-text) − AUROC(shortcut),
  shortcut ∈ {id-string-only, surface-format-only};
  若 grouped-bootstrap CI95(Δ_artifact) ⊂ [−0.02, +0.02]（= ε）→ shortcut 与 full-text
  实质等价 → 触发红线:该任务被字符串/格式模式解决,不做 hidden 增量声明。
```

## 8. Hidden probe H（P0:pool 全部 eligible id + 层 index 消歧,防 post-hoc 选层）

```text
层:block_index_zero_based = ⌊0.7·L⌋（Qwen L=28 → 19;Llama L=32 → 22）。
  HF hidden_states 是 (embedding, layer1, …, layerL) 共 L+1 项,index 0 = embedding,
  layer k 输出 = index k。故 hidden_states_tuple_index = block_index_zero_based + 1
  （Qwen → 20;Llama → 23）。
  smoke 必须:对目标 block 注册 forward hook,核验 hook 输出与 hidden_states[tuple_index]
  数值相等（不是只对 shape;off-by-one 只能这样查出）。
位置与 pooling（primary）:
  H1 = completion 内**所有**非 echoed ATT&CK/CWE identifier 的**全部 token** 在该层的
       hidden 做统一 mean pooling:h = mean_{t∈T} h_t^(l),T = 全部 eligible identifier token。
       理由:completion label = 任意 id fabricated;只读第一个 id 会在"首 id grounded、
       次 id fabricated"时系统性读错 mention,人为削弱 H(审稿人可归因于读错位置)。
  MCQ = answer-letter token 自身在该层的 hidden。
  robustness 附录:H1"只取第一个 identifier"另报一版,不作 primary。
性质:post-hoc detection——表示已含生成出的 token,是生成后内部信号、非 early-warning,论文明说。
方法:SAPLMA-style（固定表示 → grouped-CV L2-logistic,训练协议见 §7）。禁事后选层
  （robustness 附录相对深度 {0.5,0.7,final},primary 恒 ⌊0.7L⌋）。
可选:EigenScore（INSIDE 2402.03744）,仅复现 ≤2 天才加。
Semantic Entropy:H1 上退化为 id_agreement_rate,预期漏稳定 fabrication;作 error-mode
  采样 baseline,不当独立 probe。
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
