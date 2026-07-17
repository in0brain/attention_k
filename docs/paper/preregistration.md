# Pre-registration：内部 hidden-state probe 的条件增量检验（v2.3，已冻结）

> 投稿前预注册。跑任何全量生成之前写死;跑完只填结果,不改判读规则。
> 冻结指纹见 `preregistration.lock`（sha256）。改设计须 bump version + 重算 hash。

## v2.3 修订记录（2026-07-16）

```text
v2.3 amendment:
  Repair a pre-Stage-0 factual mismatch in the MCQ output representation,
  define task-specific semantic output canonicalization,
  and expand MCQ to a fresh confirmatory population with adequate precision.

本次修订**只动 MCQ 臂**。H1 臂(480 prompt × K=6)不变,理由见 (3)。

(1) 事实错配(pre-Stage-0 发现,非按结果调整)
    §7 v2.2 写明:"O_t 均为该任务自己的 F5 + full-response-text（CyberMetric prompt
    要求先简短推理再给字母,MCQ 也有文本可构造 text baseline）"。
    该断言与产物事实不符。4B-3 实测 240 条 greedy:
      has_reasoning_text = 0/240;completion 字符数 min=1 p50=1 p90=1;
      唯一取值 {B:69, C:63, D:58, A:49, 'Answer! <D>':1}。
    即 MCQ 的"full-response-text"就是一个字母。后果是连锁的:
      - text block 退化成 {A,B,C,D} 上的 4 值类别 = 答案字母身份 → O_MCQ ≈ F5;
      - §6 想消除的"MCQ=F5、H1=O"不对称在**定义上**消失、在**事实上**回归;
      - ladder 塌陷:rung2(id-string-only) ≡ rung4(full-text) ≡ 答案字母;
      - §7 artifact 红线 Δ_artifact = AUROC(full-text) − AUROC(shortcut) 因构造必然
        触发(两者同一),判定"该任务被字符串/格式模式解决"在此无意义。
    另:强制推理已在 4B-3 Stage B' 试过并否决(has_reasoning_text 0.656 但
    parse_failure 0.3125、**wrong_rate 0.0**、correct-and-wrong 题数 0、
    admitted_for_f2=false)——强制推理后无错误样本,AUROC 无从计算。故"重跑生成补文本"
    不是可行退路。

(2) task-specific semantic output canonicalization（本次新增的概念,见 §7.1）
    O 的概念在两个任务上保持同一个:"模型可见输出的全部语义内容"。
    变的只是 canonicalization 方式:
      H1  : 可见输出本身即语义内容 → 直接用 completion 文本(与 v2.2 相同,不变)。
      MCQ : 字母是**指针**,不是语义内容;其语义实现 = 它所指选项的文本。
            故 MCQ 的 canonical output text = 被选中选项的文本。
    这不是"换一个更强的基线",是把同一个 O 概念在指针式输出上正确实例化。
    无 gold 泄漏:选项文本来自 prompt(模型可见),选哪个来自模型输出;两者都在输出侧。
    若选项文本本身与正确性存在数据集 artifact 相关(如正确项更长),这正是 §7 ladder 的
    surface-only rung 与 artifact 红线要检出的东西 —— 该机制保留且对 MCQ 生效。

(3) fresh confirmatory population with adequate precision（见 §7.2）
    "adequate precision" 定为**可证伪判据**,不是形容词:零假设(真 Δ_H=0)下,
    Δ_H 的 paired grouped-bootstrap CI 必须能落进 [−ε,+ε]=[−0.02,+0.02],否则该臂
    只能出 inconclusive,交付不了 §4 的主要预期。
    先验 sizing(设计时依据,非保证):以 4C 实测为锚点标定配对零假设模拟
      锚点 = 4C C_F5_F1 在 n=239 的实测 Δ CI [−0.0429,+0.0202],宽度 0.0631
      标定得 rho=0.9051(模拟宽度 0.0631,复现锚点);零假设 E[Δ]=0(已验证无偏)
    结果:
      MCQ n=239 (现有)  → CI 宽度 0.0631,equivalent 可达率  0%   ← 结构性不可达
      MCQ n=1760(fresh) → CI 宽度 0.0223,equivalent 可达率 87%
      H1  480 prompt×K=6 → CI 宽度 0.0210,equivalent 可达率 95%  ← 故 H1 不改
    H1 的 sizing 用 4D-1 **实测**标签聚集(ICC=0.261、completion 级正例率 0.149,
      201 eligible completion / 55 question)做 beta-binomial 建模,非假设值。
    故:MCQ confirmatory population = CyberMetric 全部 1760 题 fresh
      = 总池 2000 − 4B-3/4C 已用的 240(已核验该 240 全在池内,划分无歧义)。
    **排除那 240 题是必需的**:4C 在它们上的 CI 正是促成本次修订的依据,
      再用它们做确证 = 同一批数据既定设计又下结论。它们降级为 exploratory,
      不进入 confirmatory 结果。

(4) sizing 的效力边界(如实声明,不得当作已兑现的功率)
    rho=0.9051 是从 **4C 的 F1(7 维)、n=239、旧 4C 协议**标定的,却被用于
    **H(3584 维)、v2.2 三-block 协议**。两处不同,且方向相反:
      更高维 → 重训抖动可能更大 → 真 rho 更低 → CI 更宽 → 87%/95% 偏乐观;
      v2.2 把 H block 缩放到行范数≈1 → 稀释应小于 4C 的朴素拼接 → 真 rho 可能更高。
    净效应未知。故上述可达率是**有锚点的先验估计**,不是保证。
    可检验承诺:Stage 0 后必须回报各臂实际 Δ_H 的 CI 宽度。
      若实际宽度 > 0.04(装不进 ±ε)→ 该臂**如实记 inconclusive**,
      **不得**事后追加样本、改 ε、换判据或改融合协议。扩容只此一次。

修订正当性(记录在案,防事后质疑):
  - Stage 0 未启动,无任何正式结果;
  - 缺陷属 pre-Stage-0 资产检查的职责范围(检查产物是否支持已冻结的设计);
  - 修订消除的是"预注册断言"与"产物事实"的冲突,不指向任何结论方向;
  - MCQ 的 shortcut/ladder 映射(§7.1)在**看到任何新结果之前**定死;
  - v2.1/v2.2 的 hash 与实现保留在 git commit 57b7ae9 / f93877b,不覆盖;
  - 促成本次修订的 4C 数值(n=239)与已烧的 240 题一并降级为 exploratory,
    **不进入 confirmatory 结果**。
本修订**不是**"因为旧 MCQ 结果不理想所以换基线并扩大数据":
  旧 MCQ 的 AUROC 高低从未进入本次决策;进入决策的是
  (a) has_reasoning_text=0/240 这一**表示层事实**,与结果好坏无关;
  (b) CI 宽度 0.0631 > 等价带 0.04 这一**精度事实**,与 Δ 的方向无关。
后果:v2.2 的 G3 失效(MCQ 侧协议已变),须重跑单元测试与 ≤20 prompt smoke,
  取得新 G2 + 新 G3;Stage 0 仍需 G1。
```

## v2.2 修订记录（2026-07-16）

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
MCQ population（v2.3）:每题一条(greedy trace),label = 答案正确性 1[â=a*]。
  confirmatory set = CyberMetric **fresh 1760 题** = 总池 2000 − 4B-3/4C 已用 240。
  被排除的 240 题降级 exploratory,不进 confirmatory 结果(它们的 CI 促成了 v2.3 修订,
  再用于确证即"同一批数据既定设计又下结论")。sizing 依据见 §7.2。
  另采 K−1=5 条 sampled trace/题,**仅**用于 F5 的 self-consistency / id-agreement,
  不进 population(population 恒为每题一条 greedy)。
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
  O_t 均为该任务自己的 F5 + **canonical output text**（§7.1）。
  v2.2 此处曾写"CyberMetric prompt 要求先简短推理再给字母,MCQ 也有文本可构造 text
  baseline"——该断言与产物事实不符(实测 has_reasoning_text = 0/240,输出仅一个字母),
  v2.3 已据 §7.1 更正:MCQ 的 canonical output text = 被选中选项的文本。
  更正后"两任务同定义"在**事实上**成立(此前仅在定义上成立,实际 O_MCQ ≈ F5)。
  AUPRC 只在各任务内报告,不作 gate。
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

## 7.1 task-specific semantic output canonicalization（v2.3 P0，冻结）

```text
O 的概念在两任务上同一:"模型可见输出的全部语义内容"。canonicalization 方式按任务定,
在看到任何新结果之前冻结。

H1（不变）:
  可见输出本身即语义内容 → canonical output text = completion 全文。
  shortcut:id-string-only = 各 identifier 的 normalized 串;
           surface-format-only = 长度/行数/mention 计数/family 计数/首 id 位置等。

MCQ（v2.3 新增；v2.2 的 "full-response-text" 对 MCQ 失效,见修订记录 (1)）:
  实测:模型只输出一个字母,has_reasoning_text = 0/240。字母是**指针**,不是语义内容。
  canonical output text = **被选中选项的文本**（模型输出的字母 → 索引到 prompt 中该选项）。
    解析失败(无法映射到唯一选项)→ 该题记 parse_failure,不进 population,单独统计
    （4B-3 实测 parse_failure = 1/240）。
  O_MCQ = F5_MCQ + canonical output text（与 H1 同一个 O 概念、同一套 §7 融合公式）。
  ladder（与 H1 同构,rung 语义对应）:
    1 prompt-only          题干 + 全部选项文本(不含模型输出)
    2 answer-letter-only   仅字母身份 {A,B,C,D}     ← 对应 H1 的 id-string-only
    3 option-surface-only  被选项的表面格式:字符数/词数/是否含数字/是否含否定词/
                           选项位置索引/该选项长度在四选项中的排名  ← 对应 H1 的 surface-only
    4 full-response-text   被选中选项的文本(= canonical output text)
    5 F5_MCQ
    6 O = F5_MCQ + canonical output text
  artifact 红线对 MCQ 生效且**有意义**:shortcut ∈ {answer-letter-only, option-surface-only}
    与 rung4 不再同一。若选项文本本身与正确性存在数据集 artifact(如正确项更长),
    红线会检出 → 该任务被格式模式解决,不做 hidden 增量声明。
  F5_MCQ:label margin / label entropy / full entropy（4C 已有三项）
    + answer-letter token logprob（= 该任务的 "id-token" 位置）
    + self-consistency / letter-agreement（对该题 5 条 sampled trace 求）。
    verbalized confidence 在 MCQ 无文本载体 → 恒缺失,记为常数列(§7 的 σ<1e-12 规则处理)。
  H 的位置（§8 不变）:answer-letter token 自身在 ⌊0.7L⌋ 层的 hidden。
```

## 7.2 adequate precision（v2.3 P0，冻结判据）

```text
判据(可证伪,非形容词):零假设(真 Δ_H=0)下,Δ_H 的 paired grouped-bootstrap CI 必须能
  落进 [−ε,+ε]=[−0.02,+0.02]。不能 → 该臂只出 inconclusive,交付不了 §4 主要预期。
先验 sizing(设计时依据,**非保证**):
  锚点 = 4C C_F5_F1 在 n=239 实测 Δ CI [−0.0429,+0.0202](宽度 0.0631)
  标定 rho=0.9051 复现该宽度;零假设 s_o=d·y+e1, s_oh=d·y+e2, corr(e1,e2)=rho,E[Δ]=0
  H1 另用 4D-1 实测标签聚集 beta-binomial 建模(ICC=0.261、completion 级正例率 0.149)
  预期:MCQ n=239 → 0%;MCQ n=1760 → 87%;H1 480×K=6 → 95%
效力边界:rho 借自 4C 的 F1(7 维、旧协议),用于 H(3584 维、v2.2 三-block)。
  更高维 → 真 rho 可能更低(CI 更宽,估计偏乐观);v2.2 等权缩放 → 真 rho 可能更高。
  净效应未知 → 上述可达率是先验估计,不是已兑现的功率。
可检验承诺(硬):Stage 0 后必须回报各臂 Δ_H 的**实际** CI 宽度。
  实际宽度 > 0.04 → 该臂如实记 inconclusive。
  **不得**事后追加样本、改 ε、换判读规则或改融合协议。扩容只此一次(v2.3),不再有下一次。
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
