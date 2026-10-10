# Preregistration v3.0 — Stage 1（修正协议下的重做）

> **状态**：**v3.1 已冻结（2026-08-08），lock = `docs/paper/preregistration_v3.lock`。**
> v3.0 于同日冻结；**v3.1 是它的唯一一次 bump**，只改 `N_H1` 与随之的 trace 数
> （见 §11.10）——起因是 G5 在**任何模型运行之前**发现的**数据身份冲突**，
> 与性能、结果或择优无关。判读规则、估计量、阈值、预期一律未动。
> 冻结前的修订链见 §11 / §11.1 … §11.9。
> 冻结后**不再改设计**；若确需修改，须 bump version（v3.1…）、重算 sha256、更新 lock，
> 并在本节记录变更，**不得静默改动**。
> ⚠️ **G3 / G4 / G5 须按 v3 重挣**：三者的校验均含 `prereg_sha256`，v2.4 的旧 report 自动失效。
> ⚠️ **lock 已建立，但 Stage 1 的正式入口尚未接线**（见 §12.1）——
> 冻结在本轮是**文件级**强制，尚不是端到端强制。
>
> 修订链摘要：
> 第 2 稿按「方案 iv」重写（基线、分组 primary、检验角色、多重性）；
> 第 3 稿改 stacking primary 与 M1b 后果；第 4 稿改 M7 估计量、S\*/S\*+H 同构、
> MC 带与 §7.3 预冻结；第 5 稿改合并 bootstrap 分布、完整网格比较、M1b 非劣性认证、
> 冻结 meta-logistic；**本稿**解决 IUT 与 simultaneous bounds 的 primary 冲突、
> 独立论证 ε_NI 并新增 (iv-a) 直接检验、MC 诊断改 jackknife + t 分位、变量改名；
> 第 7 稿修正 simultaneous bounds 枢轴方向、清除 §6 的 ensemble estimand、
> 把 (iv-a)/(iv-b) 的 OR 收成单一 primary 判据、写死 meta 输入规格；
> **本稿**统一记号、把覆盖率限定为 nominal、清理旧 batch-means 叙述，
> 并新增 **M8 分析实现规格（自足冻结）**。
>
> **本文件是 Stage 1 的唯一设计权威。** v2.4（`preregistration.md`，sha256 `7c051d35…`）
> 保持原样、不被覆盖——它仍是 **Stage 0** 的设计权威。两份并存是有意的：正文报 Stage 1，
> 附录报 Stage 0 及其教训，各自可独立校验。
>
> `[[待定]]` 标记的量在冻结前必须填实；带 `[[待定]]` 不得冻结。
> **冻结时全部已填实**：`R=3`、`B_per_batch=1000`（§7.4）、`F=5`（M2d）、
> `N_H1=1544`（v3.0 为 1547，v3.1 按 G5 排除 3 条）/ `N_MCQ=1760`（M6 / §7.5）、§10 三条预期。

---

## 0. 为什么重做

Stage 0 跑完后的 post-hoc 审计发现 §7 的 artifact 红线把**双侧等价检验**用在了**单侧推断
问题**上：6 次 shortcut 判定中冻结规则触发 **0 次**，修正为非劣检验后触发 **3 次**。

Stage 0 的处理是「冻结规则作 primary，修正规则作 post-hoc 建议」。程序上无可指摘，但
**它把一个已知错误的规则放在正文 primary 位置**——不细读的读者会把
`Δ_H = +0.042, read=increment` 当结论带走，而这个 read 恰是错误规则没能拦下的。

踩坑记录属于附录；放在正文 primary 位置就是误导。故重做。

**不追溯**：本文件不修改、不重算、不撤回 Stage 0 的任何结果。Stage 0 的三份冻结产物
保持冻结，其 confirmatory 地位不变，只是在论文里从正文移到附录。

### 0.1 诚实声明：这不是一次盲的预注册

我们已在 Qwen 与 Llama 上看过修正规则的表现。本文件因此**不假装无知**——§9 把全部预期
写在跑之前，跑完只对照。预注册在此情形下的价值不在"我们不知道会怎样"，而在
**规则在看到新数据之前已写死，且不因新数据的结果而改动**。

---

## 1. 诊断的重新定位：这是基线问题，不是分组问题

第 1 稿把「identifier 跨 prompt 复现」当作必须用分组封堵的**泄漏**，并因此把
`C = prompt ∪ fabricated-id` 定为 primary。这个方向是错的，理由有三，全部可测：

**(a) 它没有对准病灶。** 审计发现三早已指出：红线比的是 `rung4 full-text` vs shortcut，
而需要保护的量 `Δ_H = AUROC(O+H) − AUROC(O)` 里根本没有 rung4。真正决定 `Δ_H` 可解释性的
是**基线本身有多强**——特别是它有没有把 identifier identity 这类推理时显然可得的表示
纳入进来——不是折怎么分。

**(b) 分组路线在本任务上测得不可行。** 只读审计（2026-08-04，Stage 0 eligible n=2016）：

| 分组 | 分量数 | 最大分量 | 可用 |
|---|---|---|---|
| A：prompt | 358 | — | ✅ |
| C：prompt ∪ **fabricated** id | 277 | 200 (9.9%) | 可用但 **label-informed** |
| D1：prompt ∪ **全部** id（label-free） | **7** | **1995 (99.0%)** | ❌ 塌缩 |
| D1 去掉 top-50 hub id | 73 | 841 (41.7%) | ❌ 仍塌缩 |

**(c) C 本身不干净。** 它用 fabricated 标注构造折（label-informed），偏倚方向未知
（第 1 稿称"只可能压低性能"，**无依据，本稿撤回**）；且 grounded identifier 仍跨折，
身份通道只封了一半。

**结论**：把「identifier 身份」放进**基线**，而不是用分组去封它。
在 prompt-disjoint 的 A 分组下，identifier 跨 prompt 复现对基线与 H **同样可见**，
于是它成为 **estimand 的一部分**，不再是 H 独占的优势。

---

## 2. 估计量族（不同分组 = 不同 estimand，**不要求彼此一致**）

本节是本稿与第 1 稿最重要的差别：我们**不再假设存在唯一"正确"的分组**，
而是明确声明三个不同的 estimand，各自回答不同的问题。

```text
E1  benchmark-conditional 条件增量   【PRIMARY】
    分组 = A（prompt-disjoint）
    Δ_H = AUROC(S* + H) − AUROC(S*)
    问：在这一批 prompt 分布上，hidden state 是否提供了超出 **S\*（一个预先指定的
        强操作性表层基线）** 的增量？identifier 跨 prompt 复现属于该分布的性质，
        基线同样可用。**不**声称 S\* 穷尽了表层信息。

E2  entity-generalization 增量        【预注册但可能 NON-ESTIMABLE，见 M2b】
    分组 = 全 identifier 身份不跨折
    问：对**训练时未见过的实体**，hidden state 是否仍提供增量？
    若连通图塌缩 → 如实报告该 estimand 在本样本上 **non-estimable**，
    **不强行构造 CV**、不报任何数字。

E3  vocabulary-block 增量             【次要，独立 estimand】
    分组 = 预注册的 identifier 词表分块（label-free，prompt 整组归折）
    问：限定在一个不相交词表块内，增量如何？
    **不要求与 E1 一致**——它测的本来就不是同一件事。
```

`C = prompt ∪ fabricated-id` **不再是任何 primary**，降为 **label-informed sensitivity**
（M2c），报告时必须标明其 label 依赖与未知偏倚方向。

---

## 3. 修订（M1–M8）

### 记号约定（**全文统一，实现必须照此命名**）

```text
r  = repeat 索引            r = 1..R          （重复交叉拟合的第 r 次）
k  = comparison / rung / block 索引  k = 1..m （M1b 中 m=5；M1c 中为 shortcut 族）
q  = bootstrap draw 索引    q = 1..B_total    （单次 group 重采样）
j  = MC batch 索引          j = 1..B_batch    （=20，仅用于 MC 诊断）
```

⚠️ 第 7 稿曾在 M3 用 `r` 表示 rung、在 M7 用 `r` 表示 repeat，且 `b` 同时表示
block 与 bootstrap draw。**本稿统一为上表**，避免实现时串位。

### M1 — 基线重设计：S\*，并把 shortcut 诊断从 Δ_H 的硬闸上摘下来 【P0】

#### M1a 强表层基线 S\*（**Δ_H 的唯一保护机制**）

```text
S* = **一个预先指定的表层信息 block 集合**（不按测试表现选，也**不声称穷尽**推理时可得的
     表层信息 —— 见 M1a 末尾的"强操作性基线"说明）：
  B1 identifier identity      归一化 identifier 字符串的 TF-IDF/指示表示
                              （**新增**；Stage 0 的 O 不含此项，正是问题所在）
  B2 surface / format         长度、模板、行数、列表形态、ontology 前缀、id 位置等
  B3 canonical / full text    §7 冻结的 TF-IDF（word 1-2gram + char_wb 3-5gram）
  B4 F5 / logits              label margin/entropy、id-token mean/min logprob、
                              id_agreement / self-consistency、verbalized confidence
  B5 prompt-side 信息         **text block**：prompt_text + 固定的 family/route 标记
                              （标记为预注册的固定枚举字符串，非从数据推导）
```

组合方式（**预先指定，不取 max**）—— **两个量必须同构**：

```text
S*    = **五块 stack**   over {B1, B2, B3, B4, B5}
S*+H  = **六块 stack**   over {B1, B2, B3, B4, B5, H}
```

**nested stacking 的定义（S\* 与 S\*+H 共用同一份规格）**：

```text
在每个 outer fold 内：
  1. 每个 block 在该 fold 的 train 上分别拟合 base learner（L2-logistic）
  2. 用该 train 内的 inner cross-fitting 得到 train 侧 OOF 分数 → 训练 meta-logistic
  3. 以整个 train 拟合的各 base learner 对 test 打分 → 送入 meta → 得到该 fold 的 test 分数
  ❌ 禁止用全数据 OOF 分数训 meta 后再评估同一批样本（泄漏）
```

**S\* 与 S\*+H 必须共享的东西（逐项，缺一即为不可比）**：

```text
✔ 完全相同的 outer folds（同 seed、同分层、同 group 定义）
✔ 完全相同的 inner cross-fitting 划分
✔ 完全相同的 base learner 与其 C 网格 {0.01, 0.1, 1, 10}，同一 nested 选法
✔ 完全相同的 meta-learner 及其**全部**参数（下方逐项冻结）
✔ 完全相同的标准化时点（所有统计量只在 train fold 上拟合）
✔ 同一次 repeat 内成对使用（M7 的 Δ_r 是同一 fold-seed 下两者之差）
唯一差别 = **是否含第六块 H**
```

**meta-logistic 的全部参数（逐项冻结，S\* 与 S\*+H 共用）**：

```text
输入        各 base learner 在 test fold 上的 `decision_function` 输出，**直接使用，
            不再做任何变换**。
            ⚠️ sklearn 的 `decision_function` **本身已是 log-odds**，再做一次 logit
               是错误的（第 6 稿写作"做 logit 变换"，**本稿修正**）。
            （等价写法：`predict_proba[:,1]` clip 到 [1e-6, 1−1e-6] 后取 logit；
              二元 logistic 下两者基本等价，但**规格只取前者**，不得混用。）
            **不**对 meta 输入做任何标准化（避免引入依赖 block 数的量纲差异）
模型        sklearn LogisticRegression
penalty     "l2"
C           1.0（**固定，不做 nested 搜索**）
            理由：meta 的输入维数 = block 数（5 或 6），远小于样本量，
            无需再调正则；固定 C 也保证 S* 与 S*+H 的 meta 层完全同规格。
solver      "lbfgs"
max_iter    5000
tol         1e-6
fit_intercept   True
class_weight    None
random_state    0
```

> ⚠️ **meta 层的 C 固定为 1.0 而非搜索**，是为了让"五块 vs 六块"的唯一差别真的只是块数。
> 若 meta 也做 nested 搜索，两者可能选到不同的 C，`Δ_H` 里就混入了 meta 正则强度的差异。

> **为什么必须同构**：若 S\* 用 stacking 而 S\*+H 用"S\* 分数拼 H 特征"或别的融合方式，
> 则 `Δ_H` 里混入了**融合架构的差异**，测到的就不再是"加入 H 的增量"。
> 第 1/2 稿未写死这一点，本稿补上。

附录对照 = 逐 block 标准化的等权早融合（每个 block 逐特征 train-fold z-score 后除以 √d，
与 v2.2 融合协议同构）。**该构造由 primary 降为对照**，且其 S\* / S\*+H 同样须同构。

Δ_H 的点估计与区间见 **M7**（逐 repeat 算差、跨 repeat 平均、同一 group-bootstrap 抽样）。

**为什么 stacking 作 primary 而非等权早融合**：等权早融合把 5 个维数悬殊的 block 拉平，
高维 block（B3 文本 TF-IDF）在统一 L2 下会稀释低维但信息密集的 block（B1、B4）——
Stage 0 的文本表示三分解已实测到这种稀释（full→window +0.049）。stacking 在 train 折内
学习各 block 的权重，不依赖人为指定的等权假设。

> ⚠️ **S\* 是"强操作性基线"，不是"表层信息的上界"。** 第 2 稿曾写作"更接近……上界"，
> **本稿撤回**：S\* 只是表层信息的**一个操作化**，受限于所选 block 集合、base learner、
> meta 协议与样本量；不存在任何论证表明它逼近可从输出侧提取信息的上确界。
> 这一降级是实质性的——它决定了 M1b 触发时该说什么（见 M1b）。

**为什么不取 max**：按测试表现从各 rung 里选最强是 winner's curse，v2.4 §4 明令禁止。
S\* 是**预先指定的组合**，在 train fold 上拟合，与选择无关。

**为什么 B5 必须含 prompt 文本（而非只有 family/route 元信息）**：H 是在 identifier token
上做 pooling 的隐藏状态，**它本身编码了 prompt 上下文**。若 S\* 只含 family/route 两个类别
标记而把 prompt 文本排除在外，`Δ_H` 就会混入**输入信息增量**——即"H 见到了 prompt 而基线
没见到"这一与内部状态无关的成分。Stage 0 实测 `rung1_prompt_only = 0.6829`（A 分组），
说明 prompt 单独就携带可观的可分性，绝不能留在基线之外。
故 B5 定为 **text block**，与 B3 同为文本但语义不同：B3 是**模型输出**，B5 是**模型输入**。
**不新增 B6**；块数仍为 S\* 五块 / S\*+H 六块，estimand 结构不变。

**为什么必须含 B1**：Stage 0 的 O = F5 + full-text **不含 identifier identity 的显式表示**，
而 H 是在 identifier token 上做 pooling，天然编码 token 身份。这个不对称才是
`Δ_H` 被质疑的根源。把 B1 放进基线后，"H 只是更高效地编码了 identifier 字符串"
这一竞争解释被**基线本身**吸收，而不是靠分组去封。

#### M1b 内部一致性检查（原"O 支配性"，语义已变）

S\* 作为基线的资格，必须**正面认证**：它相对每一个单 block 都**非劣**。

> ⚠️ **"没有任何单 block 显著击败 S\*" 不构成通过。** 那是"未能拒绝劣性"，
> 在样本量不足或方差大时会自动成立 —— 拿证据的缺失当作合格的证据。
> 第 2/3 稿正是这么写的，**本稿撤回**，改为非劣性认证。

```text
估计量      对每个单 block B_k, k = 1..5：Δ_k = AUROC(S*) − AUROC(B_k)
            **必须与 M7 完全同口径**：R 次重复交叉拟合逐 repeat 算差再平均；
            区间由同一 group-bootstrap 抽样下的合并分布给出（B_total 次）。
            ❌ 不得沿用旧 M3 的单次折分口径 —— 那与 Δ_H 不可比。
            S*、各单 block、H 的 OOF 分数必须来自**同一批** repeat 与 folds。

非劣界      ε_NI = 0.02（**独立选择的操作性容忍度**，论证见下方"关于 ε_NI"）
逐块零假设  H0_k : Δ_k ≤ −ε_NI      （"S* 比 block B_k 差得超过 ε_NI"）

【PRIMARY】边际单侧 95% 下界
  L_k^marg = 合并 bootstrap 分布中 Δ_k 的 5% 分位（逐块各自计算，**不做多重性校正**）
  CERTIFIED      ⟺ 全部 5 块 L_k^marg > −ε_NI
  NOT CERTIFIED  ⟺ 存在 k 使 L_k^marg ≤ −ε_NI
  （不存在"默认通过"：未做认证 = 未通过）

【SECONDARY】family-wise simultaneous 单侧下界（M3(1) 的 max-statistic 口径，m=5）
  L_k^sim，作为**更保守的敏感性分析**并报；不用于 primary 判定。
```

**为什么 primary 用边际界而非同时界**：认证主张是**合取**——"对所有 b 均非劣"。
按 intersection–union 检验，逐块在 α=0.05 上拒绝 H0_k，则**整体认证的 size ≤ 0.05，
无需任何多重性校正**。
⚠️ **限定**：该结论的前提是"各边际单侧程序确实具有有效的 α=0.05 水平"。
本文的 bootstrap 实现对应的是**名义**水平（见 M3(1) 的 nominal 说明），
故应表述为「在边际程序达到名义水平的前提下，IUT 整体 size ≤ 0.05（nominal）」。若 primary 改用 simultaneous bounds，那本身就是一次校正，
与"IUT 无需校正"的论证自相矛盾 —— 第 4 稿同时写了这两者，**本稿撤回该冲突**。

> **措辞约束**：`CERTIFIED` **不得**写成 "family-wise certified"（primary 用的是边际界）。
> 正确表述：「S\* 相对所选 5 个 block 逐块非劣（marginal one-sided 95%，margin ε_NI=0.02，
> IUT 下整体 α≤0.05）」。若引用 secondary，须明写"在更保守的 simultaneous 口径下亦成立/不成立"。

> 与 M1c 的区别务必分清：M1c 是"任一 shortcut 触发"（**并集**，多重检验，作发现时须校正）；
> M1b 是"全部 block 非劣"（**合取**，IUT 适用，不需校正）。
> 第 2 稿曾把 IUT 用在并集上，那是错的；此处是它真正成立的情形。

**关于 ε_NI —— 独立论证，且它限制了本文能主张什么**

```text
(i) ε_NI 是**预先选择的操作性容忍度**，不由 ε、δ 或任何其他等效性阈值自动推出。
    取 0.02 是一个**透明但未经外部验证的操作性选择**：我们把它公开写死，
    以便读者按自己的尺度重新判读，而**不**声称它有实质性论证。
    ⚠️ 特别地，**不得**用"与 AUROC 常见报告精度量级相当"作为依据 ——
       小数位数是报告惯例，不是科学论证（第 6 稿如此写，**本稿撤回**）。
    **这是一个选择，不是推导。** 其后果由 (iii)/(iv) 显式承担。

(ii) M1b 的认证范围**仅限于**：「S* 在容忍度 ε_NI 内不劣于**所选的** 5 个 block」。
     它**不**证明 S* 是表层信息的上界，**不**排除存在更强的表层构造，
     **不**允许把 S* 称为"最强表层基线"。

(iii) ⚠️ **认证通过并不使 Δ_H 的正增量自动可解释为"超过最佳单 block"。**
     ε_NI = 0.02 允许 S* 比最佳单 block 差最多约 0.02。若实测 Δ_H 也在 0.02 量级，
     则该增量**可能只是补回了 S* 相对最佳单 block 的构造损失**，而非真正超出表层。
     这两种情形在 M1b 的认证下**无法区分**。

(iv) 若要主张「S*+H 超过每一个单 block」，**唯一许可的 primary 判据是 (iv-a)**：

     (iv-a)【PRIMARY，唯一授权路径】直接检验：对每个 k = 1..5，
            Δ^{direct}_k = AUROC(S*+H) − AUROC(B_k)
            口径与 M7 完全相同（逐 repeat 算差 → 跨 repeat 平均 → 合并 bootstrap 分布）；
            判据 = 该差的**边际**单侧 95% 下界 > 0，且对**全部 5 块**成立（IUT，无需校正）。

     (iv-b)【SENSITIVITY，不单独授权任何表述】代数一致性检查：
            「M1b CERTIFIED **且** Δ_H 的 CI 下界 > ε_NI」
            —— 必须写成这个**合取**，不得只写第二项。
            它与 (iv-a) 是同一件事的两种看法，**并报以检查二者是否一致**；
            若不一致，如实报告并以 (iv-a) 为准。

     ⚠️ **绝不允许 "(iv-a) 通过 OR (iv-b) 通过 → 作正面主张"。**
     两条路径的并集会重新引入择优触发与 look-elsewhere 问题 —— 那正是本协议
     在别处反复防的东西。第 6 稿写成了 OR，**本稿撤回**。

     (iv-a) **未**通过时，**不得**作"超过最佳表层 block"的表述，
     只能说"相对该特定 S* 构造有增量"，无论 (iv-b) 是否成立。
```

**(iv-a) 在本协议中作为预注册项执行**，其结果与 M1b、Δ_H 一并报告。

**后果（按认证结果二分，跑前写死）**：

> **NOT CERTIFIED（含"未做认证"）** →
> **核心主张「Δ_H 是相对强表层基线的增量」不成立**，该主张作废，不是换个说法。
> 此时必须：
> 1. 报告 S\* **未通过非劣性认证**，点名未通过的 block、其 `Δ_b` 与边际下界 `L_b^marg`
>    （并附 secondary 的 `L_b^sim`）；
> 2. 把 Δ_H 明确限定为「相对**该特定 S\* 构造**的增量」；
> 3. **不得**声称它通过了强表层基线，**不得**作任何"超出输出可观测"的表述。
>
> **CERTIFIED** → 可写「S\* 相对所选 5 个 block **逐块非劣**（marginal one-sided 95%，
> margin ε_NI = 0.02，IUT 下整体 α ≤ 0.05）」。**仅此而已**——
> 不得写成 "family-wise certified"，也不得据此称 S\* 为最强表层基线。

理由：S\* 是**强操作性基线**（非上界，见 M1a 的降级说明）。它作为基线的资格
必须**被正面证明**为「不劣于任一单块」；若无法证明，该 stack 就不是合格的基线构造，
`Δ_H` 相对它的增量无法承载"超出表层"的含义。
把这写成"融合缺陷、改改措辞"是不够的——第 1 稿如此，**已撤回**。

**CERTIFIED 也推不出的两件事**（必须与结果同报）：
1. **推不出 S\* 逼近表层信息的上界。** 认证的范围仅限所选的 5 个 block 与 ε_NI = 0.02
   这一 margin，与"是否存在更强的表层构造"无关。
2. **推不出任何关于未纳入 block 的表层信息的结论。** block 集合是预先指定的一个选择，
   不是穷举。

#### M1c 任务构念诊断 T1（**不再硬 gate Δ_H**）

```text
Δ_task = AUROC(B3 canonical/full text) − AUROC(shortcut)
         shortcut ∈ {B1 identifier-string-only, B2 surface-only}
单侧：bootstrap percentile 95% 上界 < +ε  →  T1 触发

T1 触发的**唯一**后果：
  ❌ 不得作任何**机制性**表述 —— "语义理解"、"内部觉察"、"模型知道自己在编造"
  ✅ **仍可**报告 benchmark-conditional Δ_H（E1）——它是相对 S* 的增量，
     而两个 shortcut 所用的表示（B1、B2）**本身就是 S* 的组成 block**，
     故 shortcut 强并不使 Δ_H 失去意义
     （注意：这只说明 shortcut 的信息已在 S* 内，**不**声称 S* 涵盖了
      全部表层信息 —— 见 M1a 的"强操作性基线"说明）
```

**这是本稿与第 1 稿的实质差别。** 第 1 稿让 T1 硬 gate 掉增量声明，等于用一个
关于**任务**的陈述去否决一个关于**基线之上增量**的陈述——两者不是同一件事。
真正保护 `Δ_H` 的是 S\*（M1a），不是 T1。

### M2 — 分组：A 作 primary；E2 作可估性审计；C 降为 label-informed sensitivity

#### M2a PRIMARY：A（prompt-disjoint）

```text
CV 与 bootstrap 按 source prompt 分组，K 条 completion 共享 group（继承 v2.4 §3）。
identifier 跨 prompt 复现**不再被视为需要封堵的泄漏**——它是 E1 的 estimand 的一部分，
且在 M1a 之后对基线同样可见。
```

#### M2b 预注册的 entity-generalization 可估性审计（E2）

```text
构造：union-find 连接 (completion, prompt) 与 (completion, **每个** 归一化 identifier)，
      **不使用任何 label**。
判据（跑前写死）：
  若 最大连通分量 > 总样本的 20%  或  分量数 < 5·n_outer_folds
     → 判定 E2 在本样本上 **NON-ESTIMABLE**
     → **只报告该判定与连通结构统计**（分量数、最大分量占比、hub identifier 分布），
       **不构造 CV、不报任何 AUROC/Δ_H**
  否则 → 按与 E1 相同的协议估计 E2 并报告
```

**为什么这样写**：一个塌缩的连通图上强行做 CV，得到的数不是"更严格的估计"，
而是**没有意义的数**。把它写成可估性判定，是对 estimand 的诚实陈述，
也把 Stage 0 的 D1 观测（7 分量 / 最大 99.0%）从"失败"转为**预注册可回答的结果**。

#### M2c label-informed sensitivity：C

```text
C = prompt ∪ fabricated-id 连通分量。作为 **sensitivity** 报告，非 primary。
报告时必须同时声明：
  (i) 该分组使用 fabricated **标签**构造折 → label-informed；
  (ii) 偏倚方向**未知**（不得声称"只会压低性能"）；
  (iii) grounded identifier 仍跨折 → 身份通道只封了一半。
```

#### M2d 独立 estimand：E3 词表分块（label-free）

```text
把归一化 identifier 空间按预注册 hash 划成 F = 5 个不相交块；
每个 prompt 组按其 greedy completion 的**第一个** identifier 所属块整组归折；
标签与 H pooling **只在该块内的 mention** 上计算。
=> identifier 与 prompt 均不跨折（Stage 0 上实测均为 0）。
**E3 的判读独立于 E1，不要求一致**——它的标签定义不同，测的是另一个量。
```

**F = 5 的理由（2026-08-08 冻结）**：E3 的词表块**本身充当 outer fold**，故必须与
`n_outer_folds = 5` 对齐。曾考虑的 `F ∈ {10,20,50}` + "非空块数 ≥ 5·n_outer_folds" 规则
**已撤回**：该下限会机械排除 F=10/20 并把结果锁死在 F=50，同时严重稀释块内正例。

**分块映射（逐字冻结，salt 与规范化规则均不得因审计结果重算）**：

```text
salt              "recover_attention/v3/E3_vocab_block/2026-08-08"
normalized_id     由下列规则给出（与 h1_identifier.normalize_identifier 一致）：
                    attack →  T####  或  T####.###      （大写，无前导零补齐）
                    cwe    →  CWE-<int>                 （**去前导零**，如 CWE-0079 → CWE-79）
                    cve    →  CVE-YYYY-NNNN             （大写；CVE 不进 primary）
block(id)         = int( sha256( salt + "|" + normalized_id ).hexdigest(), 16 )  mod 5
```

> ⚠️ **salt 一经冻结不得更换，块映射不得重算。** 换 salt 等于换一次分折，
> 反复换到"好看"的分法就是 look-elsewhere。审计观察到的块大小分布
> （见 `outputs/logs/stage1_capacity_audit.json`）**不具约束力**，
> 不得据以改 F、换 salt 或重新分块。

#### M2d-gate E3 可估性 gate（**跑前写死，2026-08-08**）

```text
在冻结映射下构造 E3 的折之后，若**任一** fold 满足：
  (a) 为空；
  (b) 在 E3 重定义的标签下只有单一类别；
  (c) 或其训练补集只有单一类别；
则判定  E3 = NON-ESTIMABLE。

NON-ESTIMABLE 时只报告：每折样本数、每折正/负例数、mention retention（块内保留的
mention 占比），**不报任何 AUROC / Δ_H**。

❌ 不得改 F、不得换 salt、不得重新分块、不得合并或丢弃 fold 以求可估。

类别齐全但样本量小时 → **正常分析**，不确定性由 CI 与精度判据表达；
❌ 不另设任何事后的"最少正例数"门槛（那等于看着结果加 gate）。
```

> 与 M2b（E2）的对称性：两者都是**预注册的可估性判定**，都以"如实报告不可估"
> 取代"强行构造 CV 得到没有意义的数"。差别在触发条件：E2 看连通结构，
> E3 看冻结映射下的 fold 是否退化。

### M3 — 多重性与置信界（**本稿完全重写**）

第 1 稿两处错误，均撤回：

> ❌ 「Holm 作用于 percentile bootstrap 的尾部比例 `p_r = (1/B)#{Δ_r* ≥ 0}`」
> —— percentile bootstrap 分布中心在**观测统计量**而非零假设，其尾部比例**不是**
> 有效 p 值。
>
> ❌ 「两个 shortcut 因 intersection–union 结构故无需校正」
> —— IUT 适用于"主张要求全部子条件成立"的情形；而"任一 shortcut 触发即判任务无效"
> 是**并集式发现**，属多重检验，不能免校正。

**本稿规定：**

```text
（1）**需要族错误率控制时**的口径 —— simultaneous one-sided bounds
    （留在 CI 语言内，不引入 p 值）
    ⚠️ **适用范围**：用于"并集式发现"（如 M1c 角色 II：主张任务是表层可解的）。
    **不适用于 M1b 的 primary** —— M1b 是 IUT 下的合取认证，primary 用**边际**单侧界、
    无需校正；simultaneous 界在 M1b 中只作 secondary 敏感性分析。
    把同时界当成"哪里都更严所以哪里都该用"，会与 IUT 的论证自相矛盾。
    对一族 {Δ_1..Δ_m}，用 bootstrap **max-statistic** 求同时界。
    **两个方向的枢轴量不同，不可共用**（第 6 稿曾用下界方向的统计量去造上界，**本稿修正**）：

    ── 同时单侧**下界**（用于 M1b secondary）────────────────────────
       枢轴  T_q^L = max_k ( Δ̂_k − Δ_k^{(q)} ) / ŝe_k
       临界  c_α^L = T^L 的 (1−α) 分位数
       界    L_k   = Δ̂_k − c_α^L · ŝe_k
       目标覆盖  P( ∀k : Δ_k ≥ L_k ) ≈ 1 − α        （**nominal**，见下方限定）

    ── 同时单侧**上界**（用于按上界方向判读的检验）──────────────────
       枢轴  T_q^U = max_k ( Δ_k^{(q)} − Δ̂_k ) / ŝe_k
       临界  c_α^U = T^U 的 (1−α) 分位数
       界    U_k   = Δ̂_k + c_α^U · ŝe_k
       目标覆盖  P( ∀k : Δ_k ≤ U_k ) ≈ 1 − α        （**nominal**）

    推导：欲使 ∀k: Δ_k ≥ Δ̂_k − c·ŝe_k，等价于 max_k (Δ̂_k − Δ_k)/ŝe_k ≤ c，
    其 bootstrap 对应量即 T^L；上界同理由 max_k (Δ_k − Δ̂_k)/ŝe_k 给出，即 T^U。
    一般 c_α^L ≠ c_α^U。

    ⚠️ **覆盖率是名义（nominal）水平，不是有限样本下的严格保证。**
       max-statistic bootstrap 给出的同时覆盖，依赖 grouped-bootstrap 的有效性
       与常规正则条件（如统计量的渐近线性化、group 数足够）。
       在这些条件下同时覆盖率的**名义水平**为 1−α；本文**不**声称
       `P(·) ≥ 1−α` 这一无条件有限样本不等式。报告时须写 "nominal 95%"。

    使用指派：M1b secondary 用 **下界**；M1c 角色 II 按其判据方向选用上界或下界
              （T1 的判据是 "Δ_task 的上界 < +ε" → 用**上界**）。
    ŝe_r 取该 rung 的 bootstrap 标准差；ŝe_r < 1e-8 时该项退化，记录并排除。

（2）若确需 p 值 —— 必须用**零假设下**的重采样，二选一，跑前写死：
      (a) 置换检验：在 group 内置换标签，重建零分布（保持分组结构）；
      (b) 重中心 bootstrap：把重采样分布平移至零假设中心后取尾部概率。
    **不得**使用 percentile bootstrap 的原始尾部比例。

（3）每个检验族必须声明其**角色**（M5），角色决定校不校正。
```

### M4 — 分析样本口径必须完全声明

```text
报告必须给出：n、n_positive、n_groups、eligibility 规则，以及
「本分析使用全部满足 eligibility 的样本」这一声明。
任何对分析集的限制 → 只能作 sensitivity，不得替换 primary，
且必须说明是否对 post-treatment 变量取条件。
```

**理由**：Stage 0 跨模型复核实测，eligibility 由模型自身生成内容决定（post-treatment），
两模型 eligible 交集仅 1939/2016；**仅限制到共同样本，Llama 的判读就从 `increment`
翻成 `inconclusive`，而 CI 宽度仍满足精度承诺**。

### M5 — 检验的角色必须预先声明（决定校不校正）

```text
角色 I｜保守 admission gate（我们据此**放弃**某类表述）
    不做多重性校正。
    **但因此不得**声称"我们在全局 α=.05 下证明了该任务无效/该基线被支配"。
    可写：「按预注册的保守准入准则，我们不作 X 表述。」

角色 II｜统计发现（我们据此**主张**某事为真）
    必须用 M3(1) 的 simultaneous bounds 控制族错误率，方可陈述为发现。

本协议的指派（跑前写死）：
  M1c T1 shortcut 诊断    → 角色 I（默认）；若要写成"该任务是表层可解的"这一**发现**，
                            须另报 M3(1) 的同时界，并明示切换到角色 II。
  M1b S* 非劣性认证       → **不适用角色 I/II 二分**：它是 IUT 下的合取认证，
                            primary 用边际单侧界、无需校正；simultaneous 界作 secondary
  M3  多 rung 比较        → 同上
  Δ_H 的 §4 判读          → 单一预注册估计量，无多重性问题
```

### M7 — 重复交叉拟合 + 对全部 repeats 联合的 bootstrap（analysis-side，不增加生成成本）

```text
估计量（**逐 repeat 算 AUROC 差，再跨 repeat 平均**）：
  以 R 个**预注册**的 fold-seed 各做一次完整交叉拟合（同一 S* / S*+H / ladder 协议），
  第 r 次得到该次的 OOF 分数 s_r(S*)、s_r(S*+H)。
      Δ_r = AUROC(s_r(S*+H)) − AUROC(s_r(S*))
      Δ̂_H = (1/R) · Σ_r Δ_r
  即：**先在每个 repeat 内部算 AUROC 差，再平均**。

区间（**单一合并重采样分布**，同一 group-bootstrap 抽样下跨 repeat 平均）：
  第 q 次 bootstrap：按 group 重采样一次，得到行索引集合 I_q；
      对**每个** repeat r，用**同一个** I_q 计算 Δ_r^(q)
      Δ̄^(q) = (1/R) · Σ_r Δ_r^(q)
  关键：group 重采样**每个 q 只做一次**，R 个 repeat 共用该抽样 —— 于是 CI 反映的是
  「repeat-平均统计量」的抽样变异，而不是把 repeat 间差异误当抽样变异。

  **主 CI 来自唯一的一个合并分布**：
      B_total = B_batch × B_per_batch         （B_batch = 20，见下）
      把全部 B_total 个 Δ̄^(q) **合并成一个分布**
      [L, U] = 该合并分布的 2.5 / 97.5 百分位
      width  = U − L                      ← **这就是报告的宽度**

  ❌ **不得**产生 B_batch 个独立主 CI；❌ **不得**对多个 CI 或多个宽度求平均。
     批（batch）**只是**计算与诊断手段，**不参与**主估计的生成。

⚠️ **与"平均 OOF 分数"的区别（必须写进论文方法节）**：
  另一种做法是先把 R 次的 OOF 分数逐样本平均，再算一次 AUROC 差。
  **那定义的是另一个 estimand** —— 一个 R 成员的 **ensemble 预测器**的增量，
  而不是"单次交叉拟合增量"的稳健化版本。本协议的 primary **不采用**该做法；
  若作为附录对照报告，必须明确标为 **ensemble estimand**，不得与 primary 混用或互相替代。

分位数的 Monte-Carlo 不确定性（**primary = delete-one-batch jackknife；算法与判据均在此冻结**）：

```text
分批       固定 R 组 OOF 分数不变，把 B_total 次重采样按生成顺序切成
           B_batch = 20 个等长批，每批 B_per_batch 次。批间只差 bootstrap 随机种子，
           故批是 iid 的 —— 这是下面两种估计量成立的前提。

【PRIMARY】delete-one-batch jackknife（**直接估计 pooled 量的 MC 误差**）
  对 j = 1..20：剔除第 j 批，用**其余 19 批合并**的 draws 重算分位数
      → L_(−j), U_(−j), w_(−j) = U_(−j) − L_(−j)
  记 w̄_(·) = mean_j w_(−j)
      var_jack(w) = ((B_batch − 1) / B_batch) · Σ_j ( w_(−j) − w̄_(·) )²
      se_jack(w)  = sqrt(var_jack(w))
  L、U 同法各自计算 se_jack。
  理由：jackknife 直接作用于**pooled 分位数**这个我们真正报告的量；
  batch-means 估的是"单批分位数均值"的误差，与 pooled 量之间隔了一层。

【SECONDARY】batch-means（并报，作对照）
  批内量  对第 j 批单独取其 B_per_batch 个 Δ̄^(q) 的 2.5/97.5 百分位 → L_j, U_j, w_j
  se_MC(·) = sd(·_j, ddof=1) / √B_batch

MC 误差带（以 **PRIMARY 的 jackknife** 为准）
  以**合并分布**的 width 为中心（**不是** mean(w_j)，也不是 w̄_(·)）：
      带(w) = [ width − t₀.₉₇₅,₁₉ · se_jack(w) ,  width + t₀.₉₇₅,₁₉ · se_jack(w) ]
      t₀.₉₇₅,₁₉ = 2.093    ← **不用 1.96**：自由度只有 19，正态分位数会低估带宽
  同法给出 L 与 U 各自的 MC 带并报告。
```

> **必须写清的三点**：
> (i) 主估计 `width` 来自**合并分布**，批只用于估 MC 误差；
> (ii) `mean(w_j)`、`w̄_(·)` 与 `width` 是**三个不同的量**（批小时分位数有偏），
>      三者都报，但**判定只用 `width`**；
> (iii) 带宽用 **t₍₀.₉₇₅,₁₉₎ = 2.093**，不是 1.96。

  ❌ **不得**把改变 fold / model / bootstrap 种子后观察到的宽度差异称为 MC 误差
     —— 那是**交叉拟合变异与 MC 误差的混合物**，两者未分离。

精度判定（**跑之前写死，不看结果再定**）：
      adequate     ⟺  MC 误差带的**上端** <  0.04
      insufficient ⟺  MC 误差带的**下端** >  0.04
      borderline   ⟺  0.04 落在带内（含端点）
  borderline 时**不得**单方向宣称 adequate 或 insufficient，
  且按 §5 扩容承诺**不得**追加样本、加大 B_total 或改 B_batch 以求越过阈值。

B_batch = 20（**在此冻结**，仅为诊断分批数，不改变主 CI）。
  ⚠️ **不得**由运行结果反推决定 —— 那等于看着 MC 带宽调 MC 带宽。
  20 是**预先固定的诊断预算**，不增加总计算量（B_total 不变，只是切开来看）。
  ❌ **不声称** 20 批"足以判定"任何事 —— 若 MC 带跨越 0.04，结果即为 **borderline**，
     这正是 borderline 这一档存在的意义，而不是 B_batch 取小了的表现。
     按 §5 扩容承诺，borderline 时不得加大 B_batch、B_per_batch 或追加样本。

R = 3,  B_per_batch = 1000   ← **已由 §7.3 收敛研究确定**（2026-08-07），
            r_b_source = "convergence_study"，未启用回退值。
            B_total = B_batch × B_per_batch = 20 × 1000 = 20,000。
            依据与产物见 §7.4。
```

**已实测但口径不干净的观测（留作记录，不得据以主张）**：在 Stage 0 全量数据上，
同时改变 analysis seed（该 seed 同时驱动 fold 分配与 bootstrap）8 次，
`ci_width` 落在 0.0329–0.0396（极差 0.0067）；固定该 seed、把 `B_per_batch` 由 1000 提到 4000，
宽度由 0.0396 变为 0.0401，跨越了 0.04 阈值。

> 这两条**只能**说明「精度判定在当前 `B_per_batch` 与 fold 协议下不可复现」，
> **不能**用来给 MC 误差定量（种子未分离），
> **也不能**支持"重复 CV 能消掉其中一半"这类说法——第 1 稿曾如此声称，**本稿撤回**。
> M7 的独立分批 MC 诊断（**delete-one-batch jackknife 为 primary，batch-means 为 secondary**）
> 用于分离该不确定性。


### M8 — 分析实现规格（**自足冻结**，实现者无需回查 v2.4 或现有代码）

> **以下继承项由 Stage 0 现行实现与正式产物只读提取（2026-08-06）；
> v3 新增或收紧的约束均单独标明（见本节末的差异表，标 🆕）。
> 全部条目不得根据 Stage 1 的任何结果调整。**

```text
── outer / inner 折 ──────────────────────────────────────────────
n_outer_folds     5    ┐ 来源 = **Stage 0 正式产物**（`PRODUCTION_MIN_*` 只是下限，
n_inner_folds     3    ┘ 不能单独证明实际取值）：
                       outputs/stage0/stage0_production_preflight.json
                         :: code_readiness.production_config_valid.{n_outer_folds=5, n_inner_folds=3}
                       outputs/stage0/evaluation/stage0_analysis_report.json
                         :: config.{n_outer_folds=5, n_inner_folds=3, seed=0}
                       代码侧下限见 evaluation/config.py::PRODUCTION_MIN_{OUTER,INNER}_FOLDS
splitter          sklearn.model_selection.StratifiedGroupKFold
                  shuffle=True, random_state=<见下 retry 规则>
分层变量          y（completion-level label）
group 变量        A 分组 = group_id（prompt）；E2/E3 见 M2b/M2d
                  （conditional_increment.py::stratified_grouped_folds）

── 折构造失败的 retry / fallback（**确定性**）────────────────────
max_seed_retries  20（outer）；10（inner，用于 nested C 选择与 stacking 的 inner 层）
retry 序列        random_state = seed, seed+1, …, seed+max_seed_retries−1
接受条件          (a) 每个 train/test fold 均含正负两类；
                  (b) 断言 group 不跨 train/test（越界即 AssertionError，不吞）
全部重试失败      **抛错停止，不静默降级**（禁止退化为单类 fold）
                  （stratified_grouped_folds 的 4 条注释即此规则）

── base learner（每个 block 一个）────────────────────────────────
sklearn.linear_model.LogisticRegression(
    penalty       = "l2",
    solver        = "liblinear",     ← 与 Stage 0 一致（**不改为 lbfgs**）
    max_iter      = 2000,
    tol           = 1e-4,            （sklearn 默认，此处显式写死）
    fit_intercept = True,
    class_weight  = None,
    random_state  = 0,               ← 🆕 **v3 新增**：Stage 0 未设，属可复现性缺口，此处补上
)
                  （conditional_increment.py::block_oof_scores / _select_C）

── C 选择（nested，只在 train/dev 内）────────────────────────────
C 网格            {0.01, 0.1, 1.0, 10.0}   （C_GRID，预注册冻结，config 不可覆盖）
选择目标          **inner-fold 的 mean AUROC**（np.nanmean，跨 inner folds）
                  全部 inner AUROC 非有限 → 该 C 记 −inf
扫描顺序          按上述**升序**遍历
tie-break         比较用**严格大于**（`m > best_auc`）→ 并列时保留**先出现者**
                  = **最小的 C**。此行为在此显式冻结为规格，而非实现巧合。
inner CV 不可行    捕获 (RuntimeError, ValueError) → **回退 C = 1.0**
                  且**必须逐 fold 记录该回退**（🆕 v3 新增报告义务：
                  产物须含 `c_selection_fallback_folds` 计数与 fold 索引）

── AUROC 与分数尺度 ──────────────────────────────────────────────
AUROC 计算        用 base learner 的风险分；probability 与 decision_function 是单调变换，
                  AUROC 不变，故此处不敏感。
meta 输入         **必须用 `decision_function`**（M1a 已冻结）——注意 Stage 0 的
                  `_risk_scores` 优先返回 `predict_proba[:,1]`，二者对 AUROC 等价，
                  但对 meta 的输入尺度不等价。**实现须为 meta 单独取 decision_function。**

── shared block 的复用（🆕 **v3 新增，P0 级要求**）──────────────────
S* 与 S*+H 共有的 5 个 block（B1..B5），在**同一 repeat、同一 outer fold** 内
**必须复用同一次 base 拟合结果与同一份 base OOF 分数**，
**不得**为 S* 与 S*+H 各跑一套可能选到不同 C 的选择过程。
实现上：每个 (repeat, outer fold) 先算一次 {B1..B5, H} 的 base 分数与 chosen_C，
        S* 取前 5 列、S*+H 取全 6 列，送入各自的 meta。
违反此条会让"五块 vs 六块"的差别混入 base 层的选择漂移 —— 与 M1a 的同构要求等价重要。
回归测试须**实测**：同一 repeat 下 S* 与 S*+H 的前 5 个 block 的 chosen_C 与
base OOF 分数**逐元素相等**（不是逻辑推断）。
```

**与 v2.4 / 现有代码的差异（逐条，供实现比对）**：

| 项 | Stage 0 现状 | v3 规格 | 性质 |
|---|---|---|---|
| 🆕 `random_state` | 未设 | `0` | 补可复现性缺口 |
| 🆕 meta 输入 | 无 meta（late-fusion 用 `_risk_scores`） | `decision_function` | 新规格 |
| 🆕 C 回退记录 | 静默回退 1.0 | **必须记录** fold 索引与计数 | 新报告义务 |
| 🆕 shared block 复用 | 无（无多 block stacking） | **强制复用 + 回归测试实测** | 新规格 |
| 🆕 tie-break | 实现行为为最小 C | **显式冻结**为规格 | 由巧合升为契约 |

### M6 — 新样本与 N、K

```text
互斥性（可机检，G5）：
  H1  新 prompt 的 source_entry_id 与 prompt_hash 与 Stage 0 的 480 条**零交集**
  MCQ 新题取自 CyberMetric-10000-v1 中不在 2000-v1 的 **8,187** 题（归一化去重后），
      question_hash 与 Stage 0 的 2000 题零交集

容量（**由 2026-08-08 的只读容量审计重算并记录**，产物
      `outputs/logs/stage1_capacity_audit.json`，脚本 `scripts/audit_stage1_capacity.py`）：
  ATT&CK 858（Stage 0 实际消耗 **140**）  CWE 969（实际消耗 **140**）
      → 全新 attack+cwe 可用池 = 718 + 829 = **1,547** 条目
  ⚠️ **更正**：本节此前写 141/141 → 上限 1,545。实测每个 family 只消耗 140 个条目；
     多出的 1 来自把 open_gen 路由的 `source_entry_id` 哨兵值 `"None"` 计成了一个条目
     （open_gen 是 topic-based，不消耗 ontology 条目）。1,545 作废，以审计值 1,547 为准。
  CVE 364,903 但**不进主标签空间**：4D-0 已查证 CVE-YYYY-NNNN 在常见年份低序号段近乎稠密，
      随机编造也可能碰上真实 id，存在性判据过弱 —— 这是**效度**约束，不是容量选择。
      实测 CVE 族对 primary population 的贡献 3/120 组（2.5%）。

K = 6（固定）
  理由：**保持 F5 的定义与 v2.4 §3 / Stage 0 一致** —— `f5_self_consistency_exact`
  与 `f5_id_agreement_rate` 是在 K=6 上定义的，改 K 就改了 B4 block 的特征定义，
  两个 Stage 之间不再可比。
  ❌ **不是**因为"已证明 K=6 精度最优"：模拟中 K<6 的劣势混杂了 F5 特征退化
     （K 小 → 正例少 **且** 基线被削弱，两者同向），**无法分离出纯样本量效应**，
     故不作任何精度最优主张。

R = 3            ← §7.3 收敛研究确定（2026-08-07），见 §7.4
B_per_batch = 1000  ← 同上；B_batch = 20 → B_total = 20,000
r_b_source = "convergence_study"（fallback_declared = False）

N_H1  = 1544   ← **容量上界决策**（2026-08-08），依据见 §7.5。**不是**精度规划的产物。
                  1547 个 eligible 条目经 G5 的确定性排除后为 1544（见下方口径与 §11.10）。
N_MCQ = 1760   ← 同日冻结，见下方选题规则与 §7.5
K     = 6      ← M6 上文（保持 F5 定义）
生成规模  H1 1544 × 6 = 9,264 traces ； MCQ 1760 × 6 = 10,560 traces ； 合计 19,824
```

**H1 primary population（逐字冻结；v3.1 修订 N_H1）**：

```text
route   = **recall only**（不含 open_gen）
family  ∈ {attack, cwe}          （CVE 不进 primary，理由见上）
N_H1    = 1544 prompts
K       = 6
traces  = 9,264

口径（**记录数 = 不同 prompt 数 = group 数**，三者必须严格相等）：
    原始 eligible entry pool                      1547
    − 与 Stage 0 的 prompt 碰撞                     −2
    − 正式集内部相同 prompt 去重                     −1
    ────────────────────────────────────────────────
    N_H1 unique prompts / groups                  1544
```

**确定性排除规则（逐字冻结，v3.1）**：

```text
(1) 跨 Stage 碰撞 → 排除
    `g5_prompt_hash` 落在 Stage 0 的 480 条之内的条目。
    G5 要求 prompt 级零交集；分组解决不了 —— 那会让 Stage 1 重问已烧过的 prompt。
(2) 正式集内部同 prompt → 去重，只保留一条
    按 `g5_prompt_hash` 分组；组内**优先保留非 revoked/deprecated**；
    仍并列则按 `family:normalized_id` 字典序取首个。
    ❌ **不采用**"同 prompt 共享 group"：那会让该 prompt 得到 12 条生成而其他只有 6 条
       （不等权），也违反"正式集内 prompt 零重复"。

`g5_prompt_hash` = sha256(canonical_json([{role, content(LF 规范化)} …]))，
取自 `build_h1_chat_messages(record)`；Stage 0 与 Stage 1 **同规则重算**。
⚠️ 它与 generation 侧既有的 `prompt_hash` 是**两个量**，后者保持原义不变。

实际被排除的三条（产物 `outputs/stage1/samples/stage1_sample_manifest.json` 逐条记录）：
    attack:T1134.004 (active)   prompt ≡ Stage 0 的 attack:T1502
    attack:T1656     (revoked)  prompt ≡ Stage 0 的 attack:T1684.001
    attack:T1536     (revoked)  与 attack:T1578.004 (active) 同 prompt，按规则保留后者
根因：ATT&CK 的 revoked 条目与其继任者共享 `name`+`description`，而 `recall_v1` 只用这两个字段。
```

**其余 revoked / deprecated 条目继续保留，不作整体排除**：

```text
当前 estimand 判断的是「identifier 是否存在于**冻结的** ontology」，
**不是**「该标识符是否仍被推荐使用」；原协议明确把这些状态视为 existence-positive。
故 revoked 149 / deprecated 12（attack）与 Deprecated 25（cwe）不因状态被剔除。
状态分布进入 manifest 与局限说明，随结果同报。
```

**为什么不加 open_gen（三条，跑前写死）**：

```text
(1) open_gen 在原设计中是**副路线**，主要为后续 F2/F3 提供长文本原料；
(2) 它的抽样单位是 **topic**，不是 ontology entry ——
    加入会改变**容量上界的定义**（entry 数），也改变 primary population；
(3) open_gen 的数量与选择规则**尚未独立冻结**。
    ❌ 不得为了"避免 route 常量化"而临时扩大 estimand —— 那是本末倒置。
```

**由此产生的两条约束**：

```text
✔ schema 兼容性：`route` 字段与 `{recall, open_gen}` 枚举**保留不变**。
  B5 中的 route 标记在本轮**退化为常量，这不构成错误** ——
  B5 仍包含完整的 `prompt_text`，其信息职能（吸收"H 见过 prompt 而基线没见过"这一
  输入信息增量，见 M1a）不受影响。
❌ **外推禁令**：论文**不得**把 Stage 1 的 H1 结果外推到 open_gen 路由。
  旧的 open_gen 结果只能作为 **Stage 0 历史或附录材料**呈现。
```

**MCQ 选题规则（逐字冻结，2026-08-08）**：

```text
候选池        CyberMetric-10000-v1 归一化去重后的 distinct 题面，
              减去 distinct(CyberMetric-2000-v1)  → **8,187** 题
              （2000-v1 是 Stage 0 的 MCQ 来源，G4 跑在同一批正式 arm 上，
                故 "Stage 0/G4 已用" ⊆ 2000-v1）
稳定题目 ID   question_key  = re.sub(r"\s+", " ", question).strip().casefold()
              question_hash = sha256(question_key).hexdigest()      （全 64 位 hex）
salt          "recover_attention/v3/mcq_selection/2026-08-08"
选择序        按 sha256( salt + "|" + question_hash ) **升序**，取前 1760；
              并列时按 question_hash 升序（实测无并列，规则仍写死）
G3/G4         另取样本，且**必须排除在正式集之外**
```

> **8,187 vs 8,189 的分歧已由审计判定**：`10000-v1` 原始 10,180 条，其中 2 条归一化后
> 与池内他题重复；`8,189 = 10,180 − 1,991` 未做文件内去重，**作废**。
> 正确口径 `8,187 = 10,178 (distinct) − 1,991 (与 2000-v1 重叠)`。
> 另记：2000-v1 有 9 题不在 10000-v1 内；500-v1 与新池重叠 358 题，
> 但 500-v1 未被 Stage 0/G4 使用，故**不**从池中排除（若排除则池为 7,829，仍 ≫ 1760）。
> 全部计数见 `outputs/logs/stage1_capacity_audit.json`。

---

## 4. 启动 gate

```text
科学启动：G2（本文件冻结哈希）+ G3（≤20 prompt 真实 smoke，**须重挣**）
          + G4（性能盲质检，**须重挣**）+ G5（样本互斥性，新增）
投稿就绪：G1（不变，不阻塞实验）
```

G3/G4 的校验含 `prereg_sha256` 比对，故 v2.4 的 G3/G4 对本文件自动失效。

---

## 5. 一次性揭盲

继承 v2.4 §2.2 逐字适用于 Stage 1。产物目录 `outputs/stage1/`，与 `outputs/stage0/`
完全隔离；**绝不触碰** Stage 0 的 freeze manifest / unblinding log / analysis report。

**扩容承诺**：N、K 一经冻结不得追加。若精度不足，如实记 inconclusive。

---

## 6. 判读顺序与分支报告承诺

```text
执行顺序（代码强制）：
  generate → freeze
    → R 次重复交叉拟合（M7）：**保留 R 份彼此独立的 OOF 分数**
       （每份含各 block / S*(nested stacking) / H / S*+H）
       ❌ **不得**逐样本对 R 次取平均后再算统计量 —— 那是 M7 明确排除的
          **ensemble estimand**（见 M7 的 ⚠️ 段）。聚合 OOF 分数**只能**作为
          明确标注的附录 ensemble 对照，**不得**进入 M1b、(iv-a) 或主 Δ_H。
    → 在**每个 repeat r 内部**分别计算：Δ_r（M7）、Δ_{k,r}（M1b）、
       以及 (iv-a) 的逐块直接差 [AUROC_r(S*+H) − AUROC_r(B_b)]
    → **跨 repeat 平均这些统计量**（不是平均分数）
    → M1b 非劣性认证（**边际**单侧 95% 下界，逐块，IUT；secondary 另报 simultaneous 下界）
    → M1b(iv-a) 逐块直接检验 AUROC(S*+H) − AUROC(B_b) > 0（边际单侧下界，IUT）
    → M1c T1 诊断（判据方向为上界 → 用 M3(1) 的**上界**枢轴 T^U）
    → M2b E2 可估性审计
    → gate(§6 of v2.4)
    → E1 的 Δ_H：B_total = B_batch × B_per_batch 次重采样（索引 q），每次重采样对 R 个
       repeat 共用同一 group 抽样并跨 repeat 平均；**全部 draws 合并成一个分布**
       取 2.5/97.5 百分位
    → **delete-one-batch jackknife（primary）** → se_jack(L/U/w)
       ；batch-means 作 secondary 并报
    → MC 误差带 = width ± t₍₀.₉₇₅,₁₉₎ · se_jack(w) → 精度判定（含 borderline）
    → E3 → M2c sensitivity

各分支必报（无论结果如何）：
  • S* 与各单 block 的 AUROC；M1b 的 **L_b^marg（primary）** 与 L_b^sim（secondary）
    —— **认证结果必须写成 CERTIFIED / NOT CERTIFIED；NOT CERTIFIED（含未做认证）时，
       核心「通过强表层基线」主张作废（M1b），必须明写**
  • M1b(iv-a) 逐块直接检验的结果（**唯一授权路径**）；
    以及 (iv-b) 合取一致性检查的结果（sensitivity，不单独授权任何表述）；
    (iv-a) 未通过时**不得**作"超过最佳表层 block"的表述
  • T1 的单侧界与其角色声明（角色 I / II，见 M5）
  • E2 的可估性判定与连通结构统计（若 non-estimable，明写并停在此处，不报任何 AUROC）
  • E1 的 Δ̂_H、CI、read、precision，以及 AUROC(S*) / AUROC(H alone) / AUROC(S*+H)
  • **CI 宽度及其 MC 误差**（M7 口径）；若 0.04 落在误差带内 → 记 borderline
  • E3、M2c 各自的结果与其 estimand 声明
  • R、B_per_batch、B_batch(=20)、B_total 的实际取值

T1 触发时：禁机制性表述，**不禁** E1 的报告（见 M1c）。
M1b **NOT CERTIFIED** 时：**禁**「通过强表层基线」「超出输出可观测」类表述（见 M1b）。
M1b(iv-a) **未通过**时：**禁**「超过最佳表层 block」类表述 —— 无论 (iv-b) 是否成立。
```

---

## 7. 性能盲的 **N** 规划（K 已固定为 6，见 M6）

```text
本节**只规划 N**。K 不由精度规划决定（M6：K=6 以保持 F5 定义）。
R 与 B_per_batch 不在此规划，由 §7.3 的**预冻结**收敛网格确定。

规划只依据**精度**，不依据效应量：模拟中只记录并输出 CI 宽度、n、n_positive、n_groups；
点估计与 CI 端点不记录、不打印。为精度定样本量，不为显著性定样本量。

**不得**使用 C 分组拟合的 sizing 曲线（`width = 1.508·k^−0.474`）——它拟合在 C 分量上，
不适用于 A-primary。

程序：
  (1) 在 Stage 0 的 A 分组 cache 上按 group 子采样扫 N（K 恒为 6）
  (2) 只记 CI 宽度
  (3) 报告每格 seed 间极差；若极差与设计间差异同量级 → 判定模拟不足以定 N，
      **不得**用噪声拟合外推

**⚠️ 一条已被实测推翻的规划直觉（留作记录，不得再用）**

第 1 稿曾以「eligibility 90.4% 由 prompt 决定，K 几乎不改变独立单元数
（attack+cwe 产出率 k=1:0.928 vs k=6:0.986）」为由主张 N↑K↓。**该论证错误**：
「独立单元数」不等于「精度」。H1 正例率约 7%，AUROC 的精度受**正例数**限制，
而正例数随 **N×K** 走，不随 group 数走。第 1 版模拟实测（A 分组，N=358 固定）：

```
K=1  正例 33  width 0.0738      K=3  正例 124  width 0.0402
K=2  正例 73  width 0.0614      K=6  正例 250  width 0.0396
```

**且 K 的效应本身含混杂**：K<6 时 F5 的跨样本特征（self-consistency / id-agreement）
退化甚至无定义，基线 S\* 被削弱，故测到的 K 效应不是纯样本量效应。
规划必须把这一混杂显式标出，不得把 K 当作可自由压缩的成本项。

### 7.1 本规划能给出什么、不能给出什么（**两条结构性限制，冻结前必须正视**）

模拟建立在 Stage 0 的既有 cache 上，因此有两条无法绕过的边界：

```
L1  无法验证 N > 358。 Stage 0 只有 358 个 A-group，任何更大的 N 都是**外推**。
L2  无法模拟 K > 6。   Stage 0 每组只有 6 条 trace，K>6 只能靠理论推断
                       （正例数 ∝ N·K，AUROC 精度 ∝ 1/√正例数）。
```

**后果**：最终选定的 (N, K) 必然带外推成分。故本协议要求：

1. **不得**声称所选 (N, K) "保证" width ≤ 0.04；只能声称其**规划依据**。
2. §10 的预期必须写死一个**精度区间**而非点值，并预先声明：
   若实测 width > 0.04，如实记 inconclusive，**不得**追加样本（§5 扩容承诺）。
3. 已实测的一条事实必须与设计同报：**Stage 0 在 N=358、K=6 下的 width = 0.0396，
   距阈值 0.04 仅 0.0004，而同口径下 seed 间极差达 0.011–0.030。**
   即该处的 `adequate` 判定处在噪声量级内。Stage 1 若只做到同等规模，
   精度判定**可能对折分与有限 bootstrap realization 敏感，其稳定性尚未得到保证**。
   （不写成"近乎随机"——那超出了已有证据能支持的强度。）

### 7.2 已完成的模拟与其结论

```
第 1 版  outputs/logs/nk_planning_A_primary.json
         2 seeds。**不足以定 N、K**（seed 极差与设计间差异同量级）。
         其 width ≈ 1.2414·groups^(−0.635) 的拟合**作废**，不得使用。

第 2 版  outputs/logs/nk_planning_v2.json
         (N∈{180,240,300,358}) × (K∈{2,3,6}) × 3 seeds。观测：
           • K=2 → 0.064–0.282；K=3 → 0.040–0.117；K=6 → 0.026–0.062。
             ⚠️ 该对比**含混杂，不是纯样本量效应**：K<6 时 F5 跨样本特征退化，
             正例减少与基线削弱同向发生，**不可分离**。故只作观测记录，
             **不得**据以主张"K=6 精度最优"（K 的选择理由见 M6：保持 F5 定义）。
           • **N 的效应未被分辨**。
           • 两个拟合残差 sd 均 ≈0.28，rows-only 指数 −1.044 远陡于理论 −0.5
             —— 被上述混杂污染，**不得用于外推**。

第 3 版  outputs/logs/nk_planning_v3_k6.json
         只跑 K=6 × 8 seeds × N∈{120,…,358}。结果：

           N     mean      sd      adequate
           120   0.0941   0.0432   0/8
           180   0.0417   0.0097   4/8
           240   0.0405   0.0101   3/8
           300   0.0396   0.0051   4/8
           358   0.0396   —        1/1

           • N=120 明显不足；N≥180 后**均值停在 ~0.040、adequate 率约 50% 且不随 N 改善**。
           • **但该"平坦"至少部分是子采样重叠假象**：N=300 时每个子样本占全集 84%，
             8 个 seed 高度重合，均值被强制收敛到全量实测的 0.0396、sd 收缩到 0.0051。
           • ⇒ **L1 由断言变为实测**：从 358 组的数据子采样，无法推断 358 组以外的行为。

宽度构成诊断  outputs/logs/width_decomposition.json
         固定全量数据、只改 analysis seed（该 seed **同时**驱动 fold 分配与 bootstrap）
         8 次 → ci_width ∈ [0.0329, 0.0396]，极差 0.0067；
         固定该 seed、B_per_batch 由 1000 → 4000 → 宽度 0.0396 → 0.0401，**跨越 0.04 阈值**。

         > 这两条**只能**说明「精度判定在当前 B_per_batch 与 fold 协议下不可复现」。
         > ❌ 不得称 0.0067 为 MC 误差（种子未分离，是交叉拟合变异与 MC 误差的混合物）
         > ❌ 不得声称重复 CV 能消掉其中一半（未做分解，无依据）
         > 干净的分离由 **M7 的独立分批 MC 诊断**完成
         >（delete-one-batch jackknife 为 primary，batch-means 为 secondary）。
```

**K 已按 M6 定为 6（理由：保持 F5 定义，非精度最优）。
N 已于 2026-08-08 定为 `N_H1 = 1544`（v3.1）/ `N_MCQ = 1760`，但**不是**由本节的模拟定的 ——
它是**容量上界决策**，且在 L1/L2 限制下如实标为外推。理由与其不能支持的表述见 §7.5。**
```

### 7.3 R 与 B_per_batch 的收敛研究（**设计在此冻结，运行前不得改**）

收敛研究本身若可以边跑边调，就等于"跑到满意为止"。故候选网格、种子、阈值、停止规则与
回退值全部在此写死；运行只填数，不改规则。

```text
数据      Stage 0 的 A 分组 cache（全量 358 组 / K=6）。**只读，不碰 Stage 1 样本。**
估计量    M7 的 primary 口径（逐 repeat 算 AUROC 差 → 同一 group-bootstrap 抽样下跨 repeat 平均）

候选网格（冻结）—— **始终跑完整张网格，无 early stop**
  R      ∈ {1, 2, 3, 5, 8, 12}                     记 R_max = 12
  B_per_batch ∈ {1000, 2000, 5000, 10000, 20000}   记 B_max = 20000
  两者**分别**扫：扫 R 时固定 B_per_batch = B_max；扫 B_per_batch 时固定 R = R_max
  （各自钉在网格最大值，以免用一个未收敛的量去判另一个）

种子（冻结）
  fold-seed       = 20260804 + 100·r,  r = 0..11    （R 网格按前缀取用，保证嵌套：
                                                      R=3 用的就是 R=5 的前 3 个）
  bootstrap-seed  = 20260804 + 7·j,    j = 0..19    （B_batch = 20 个批，见 M7）

  **共同随机数（冻结）**：较小的 B_per_batch **必须使用每个批在 B_max 档下 RNG 流的前缀**。
  即批 j 的重采样序列只生成一次（长度 B_max），B_per_batch = 1000 时取其前 1000 个。
  这样网格各档共享随机数，档间差异反映的是 draw 数而非不同的随机实现，
  否则收敛比较会混入纯粹的 RNG 抖动。

**比较规则（冻结）：一律与网格最大档比较，不比相邻档**
  相邻档比较会在噪声下"偶然相等"就误判收敛。故：
      对每个候选档 g，计算它与最大档 g_max 的偏差 D(g)
      **选定值 g\* = 满足「g 及其之上的所有档都在容差内」的最小 g**
      即  g* = min{ g : ∀ g' ≥ g,  D(g') < tol }
  最大档自身 D(g_max) = 0，故该集合非空，规则总有解。

**各自检查的量（冻结）—— 两者查的东西不同**
  R 的网格：查 **点估计与两个端点**
      D_R(g) = max( |Δ̂_H(g) − Δ̂_H(R_max)| ,
                    |L(g) − L(R_max)| ,
                    |U(g) − U(R_max)| )
  B_per_batch 的网格：**只查两个端点与宽度，不查点估计**
      D_B(g) = max( |L(g) − L(B_max)| , |U(g) − U(B_max)| , |width(g) − width(B_max)| )
      理由：**B_per_batch 完全不进入点估计** —— Δ̂_H 由 R 次交叉拟合的 AUROC 差平均得到，
      与 bootstrap 重采样无关。把点估计写进 B_per_batch 的收敛判据是多余的，且会误导读者
      以为 bootstrap 次数会改变点估计。

**容差（冻结，实质尺度，不随运行结果调整）**
  tol = 0.002  （= ε/10，与判读尺度挂钩）
  对 R 与 B_per_batch 的全部被检查量统一使用同一 tol。
  ⚠️ tol 是**实质容差**，不是统计显著性阈值 —— 不得改成"落在 MC 带内"这类
     随样本量自动变松的判据。

**扫完仍不满足时（冻结）**
  按上式 g* 恒存在（最差情形 g* = 最大档）。若 g* = 最大档，
  必须在预注册与论文中明确记载「**在网格上限处才满足容差，未观察到更低档收敛**」，
  不得写成"已收敛"。

**性能盲边界（重要）**
  R 的收敛判据包含点估计，故 **§7.3 的收敛研究不是性能盲的**。
  其产物**不得**用于 N 的选择、不得用于任何结论表述，
  **仅**用于确定 R 与 B_per_batch 两个数值超参。此边界必须在产物 JSON 中显式标注。

回退值（冻结，仅用于收敛研究因算力/时间无法完成）
  R = 5,  B_per_batch = 10000
  依据（**先验的，不引用任何已观测的宽度变动**）：
    R=5 是重复交叉验证的常见取法，且在本网格中处于中段；
    B_per_batch=10000 使百分位估计的 MC 标准误约为 B_per_batch=1000 时的 1/√10 ≈ 0.32 倍，
    这是纯粹的 √n 关系，不依赖任何本项目的观测。
  **回退值一经启用必须显式声明，不得事后改写成"这就是收敛结果"。**

产物  outputs/logs/stage0_development_convergence_R_Bperbatch.json
      （命名刻意含 `stage0_development`：该研究跑在 **Stage 0 cache** 上，属开发期定超参，
        **不是** Stage 1 的任何结果，产物不得被引用为 Stage 1 的证据）
      须含：完整网格（两条各自跑满）、每档的 Δ̂_H / L / U / width / se_MC、
      D(g) 逐档取值、g\*、是否 g\* = 最大档、是否启用回退值、
      以及 `"performance_blind": false` 与其适用边界说明。
```

**执行顺序约束**：本节的收敛研究**必须在 M7 的实现（五/六 block stacking、重复交叉拟合、
联合 bootstrap）通过回归测试之后**才能运行——在未验证的实现上测收敛，测的是 bug 不是超参。

---

### 7.4 收敛研究的执行记录（2026-08-07，**已完成**）

```text
选定        R = 3    B_per_batch = 1000    r_b_source = "convergence_study"
            fallback_used = False（§7.3 的回退值未启用）
产物        outputs/logs/stage0_development_convergence_R_Bperbatch.json
产物 sha256 2afdbe2a749b5c5467452c54b4438119df7c092f3e0f4c81f5872f3b0c5e9feb
运行依据的本文件 sha256（**运行时的旧值，不得回写或伪造**）
            5ca5c69dd5f069df57de788949c1778418f0f70d2df6e21e4c5dbfb720ebfc9e
本文件填入 R/B 之后的新 sha256
            记录在 docs/progress/stage1_v3_analysis_history.md
            （不写在本文件内 —— 把哈希写进被哈希的文件会自指）
```

**判定摘要**（完整网格与 D(g) 见产物 JSON）：

| 扫描 | 选定 | 关键事实 |
|---|---|---|
| R（判据 `Δ̂_H` / `ci_lo` / `ci_hi`） | **3** | `D(1)=0.00195` 自身合格，但其尾部含 `D(2)=0.00222` ✘ → R=1 不被选中。**这正是"与最大档比较 + 整条尾部合格"这条规则起作用之处**；first-hit / 相邻档比较会错选 R=1 |
| B_per_batch（判据 `ci_lo` / `ci_hi` / `width`） | **1000** | 全档 `D(g) ≤ 0.0003 < tol` → 取最小档。`B_total = 20,000` 时端点与 400,000 draw 的参考仅差 0.0003，`se_jack ≈ 3e-5` |

**必须与选定值同报的两条限制**：

1. `D(1)=0.00195` 与 `D(2)=0.00222` 都紧贴 `tol=0.002`，**R=1/2/3 之间并未被清晰分辨**。
   `R=3` 是冻结规则的输出，**不是**"3 明显更好"的证据。
2. 该研究**非性能盲**，跑在**已揭盲的 Stage 0 数据**上、用 v3 的新 estimand。
   产物中的 `Δ̂_H` 与 `width` 是**开发期中间量**，
   **不得**用于选 `N`、**不得**作为任何结果或结论被引用。
   产物内以 `performance_blind: false` 与 `must_not_be_used_for` 显式封死。

---

### 7.5 N 的决策记录（2026-08-08）——**容量上界决策，不是精度规划**

```text
选定   N_H1 = 1544（= 未被 Stage 0 消耗的全部 attack+cwe 条目 1547，
                     再按 G5 的确定性规则排除 3 条，见 M6 与 §11.10）
       N_MCQ = 1760      K = 6
性质   **结果无关的容量上界决策**。取"全部可用条目"这一上界不需要查询任何宽度，
       故该决策**可证明独立于**任何已观测到的性能量。
       v3.1 的那 3 条排除同样与结果无关：它们由 G5 在**任何模型运行之前**、
       仅凭数据身份（prompt 文本是否逐字节相同）判定，不涉及任何性能量或择优。
```

**为什么不用 §7 的模拟定 N（三条，全部是限制而非辩护）**：

```text
(1) §7.2 的三版模拟全部跑在 **Stage 0 的旧 estimand**（O = F5 + full-text、单次交叉拟合、
    等权融合）上。v3 的 primary 是**五块 nested stacking + R=3 重复交叉拟合 + 合并
    bootstrap**，是另一个量。旧曲线对 v3 **不适用**，不得用于定 N。
(2) v3 estimand 下唯一存在的宽度，来自 §7.3 的收敛研究，而该产物被
    `must_not_be_used_for: [N selection, …]` 明令封死。
    ⚠️ **诚实声明**：那个数字**已经被看过**，不可撤销。本决策没有使用它 ——
       取上限这一动作不查询任何宽度，这正是选它的理由之一。
(3) §7.1 的 L1 是**实测**结论：Stage 0 只有 358 个 A-group，任何 N > 358 都是外推；
    再跑一次 v3 口径的子采样模拟也只能覆盖 N ≤ 358，无法验证 1544。
```

**本决策**不能**支持的表述（与结果同报）**：

```text
❌ 不得声称 N_H1 = 1544 "保证" 或"预期" width ≤ 0.04 —— 本决策不含任何精度论证。
❌ 不得把"取满容量"读作"样本量充足"。
✅ 只能说：在效度约束（CVE 不进 primary）与互斥性约束（G5）下，
   1547 是 attack+cwe 的**全部**可用条目；其中 3 条因 prompt 级碰撞被确定性排除，
   余下 1544 条全部纳入，我们没有留下任何可用样本。
```

**精度的预先承诺（§5 扩容承诺的具体化）**：

```text
实测 width ≤ 0.04 且 MC 误差带上端 < 0.04   → adequate
0.04 落在 MC 误差带内                        → borderline（不得单方向宣称）
MC 误差带下端 > 0.04                         → insufficient，**如实记 inconclusive**
三种情形下都**不得**追加样本、不得加大 B_total / B_batch、不得改 F 或 N。
```

**与 §7.1(3) 同报的既有事实**（旧 estimand，不可与 v3 混读）：Stage 0 在 N=358、K=6 下
`width = 0.0396`，距阈值仅 0.0004，而同口径下 seed 间极差达 0.011–0.030 ——
该处的 `adequate` 判定处在噪声量级内。

**生成成本与其约束**（不改变上述决策，只作执行提示）：

```text
H1 9,264 + MCQ 10,560 = 19,824 traces。
按 Stage 0 实测（480×6 = 2,880 条 ≈ 16–18 h 本地，open_gen 长文本段速率约为 recall 段 1/3）
线性外推为数十小时量级 → 属"应租 GPU"档（AGENTS §7.1）。
⚠️ backend invariant 不因换机器放松：两臂必须同一后端；若改用非 8-bit，
   **两臂都要换**，且 G3/G4 须重挣（其校验含 prereg_sha256 + backend 指纹）。
```

**产物**：`outputs/logs/stage1_capacity_audit.json`（5.0 KB，只读身份级审计；
`reads_labels_or_performance: false`，`selection_materialized: false`），
sha256 `99373bf2d55df6b3a09ed7cc2c82bd04a50ce3740f4b667d801ec38c197a84ad`。
该产物内记录的 `prereg_v3_sha256` 是**运行时的旧值**（本节写入前），**不得回写或伪造**；
本文件填入 N/F 之后的新 sha256 记在 `docs/progress/stage1_v3_analysis_history.md`。

---

## 8. 明确不变的东西

```text
分析实现的全部细节（折数/splitter/retry/base learner/C 选择/tie-break/回退/
shared block 复用）见 **M8**，已自足冻结，实现无需回查 v2.4。

不变：ε=0.02、δ=0.15、precision_max_ci_width=0.04、层 ⌊0.7L⌋、pooling 定义、
      TF-IDF 规格、逐 block 标准化融合、folds 数、nested C 网格、
      8-bit backend invariant、seed 策略、§11 缓存边界、模型（Qwen2.5-7B-Instruct）
```

---

## 9. 设计变更的分类（**取代第 1 稿的"六处全部净收紧"**）

第 1 稿声称六处修订"全部净收紧"。**该说法撤回**——它不成立，且这类说法本身就是
需要警惕的自我辩护。准确的分类如下，其中两处**可能**朝任一方向改变结果：

| 修订 | 类型 | 可能有利于正面结果？ |
|---|---|---|
| M1a 基线加入 B1 identifier identity | **estimand 重定义** | **可能，方向未知**。基线变强 → Δ_H 可能变小；但高维稀释也可能使 S\* 变弱 → Δ_H 变大 |
| M1a 由 O 改为 S\* 的组合协议 | estimand 重定义 | 同上 |
| M1a **primary 由等权早融合改为 nested stacking** | estimand 重定义 | **可能，方向未知**。stacking 学权重通常使 S\* 更强 → Δ_H 更小；但 meta 层的估计方差也可能使 S\* 更不稳 |
| M1b 改为**非劣性认证**（全部 5 块 L_b^marg > −ε_NI 才 CERTIFIED；无默认通过） | **收紧** | 否 —— 由"未被显著击败即可"改为必须正面证明非劣 |
| M1b(iv-a) 新增逐块直接检验 `AUROC(S*+H) − AUROC(B_b) > 0` | 增加约束 | 否 —— 它是"超过最佳表层 block"这一表述的**前置条件** |
| M7 重复交叉拟合 + 联合 bootstrap | **估计器变更**（点估计更稳）+ 报告义务（MC 误差、borderline） | **可能**：点估计对 fold 分配的依赖降低，可能使 CI 略窄；但 borderline 判定会**阻止**在 MC 误差带内单方向宣称 adequate |
| M1c T1 由硬闸改为机制表述约束 | **放松**（对 Δ_H 报告而言） | **是** —— 第 1 稿会 gate 掉的 Δ_H，本稿允许报告。理由见 M1c，但必须如实标为放松 |
| M2a primary 由 C 改为 A | **estimand 重定义 + 精度提高** | **可能** —— A 分组的有效单元多于 C |
| M2b E2 改为可估性审计 | 报告义务（且可能只报"不可估"） | 否 |
| M2c C 降为 sensitivity | 降级 | 中性 |
| M3 多重性与置信界重写 | 修正无效统计量 | 否（修的是有效性，不是松紧） |
| M4 口径声明 | 报告义务 | 否 |
| M5 角色声明 | 报告义务（禁止越界表述） | 否 |
| M6 N×K 重规划 | 功效变化 | 提高精度 → 更易得到**任一**明确判读，不偏向正面 |

**必须与结果同报的自我披露**：M1c 是一处**放松**，M1a/M2a 是**estimand 重定义**。
本次重做不是"同一问题上更严的一次检验"，而是**换了一个更合适的问题**，并说明为什么。

---

## 10. 预期（跑之前写死）

**三条预期于 2026-08-08 写死（冻结轮）。跑完只对照，不修改。**

```text
T1 预期**触发**。
  依据（**已公开的 Stage 0 H1 结果**，outputs/stage0/evaluation/stage0_analysis_report.json
        :: artifact_redline.h1.per_shortcut.rung2_id_string_only）：
      AUROC(B3 full text) − AUROC(B1 identifier identity)
      点估计 ≈ −0.093，95% 上界 ≈ −0.054，已低于 +ε = 0.02。
  故预期**至少由 B1 触发** T1。
  这只是**任务构念诊断**的预期：它不限制 E1 的 benchmark-conditional Δ_H 报告（M1c），
  也不预设 Stage 1 的实际结果。
```

```text
E2 预期 NON-ESTIMABLE。
  依据：Stage 0 只读审计，全 identifier 连通图 7 分量、最大分量占 99.0%；
       去掉 top-50 hub identifier 后最大分量仍占 41.7%。
  若成真，这是**预注册预见的结果**，报告为"该 estimand 在本样本上不可估"，
  不得改写成"我们发现实体泛化不成立"——**不可估 ≠ 不成立**。
```

```text
E1 预期 primary 95% CI width 落在 **[0.015, 0.050]**。
  依据**仅使用已公开的 Stage 0 结果与先验缩放关系**：
      Stage 0 在 358 个 A-group 下 width = 0.0396；
      按独立 group 数的 1/√N 启发式缩放到 1547 groups → 中心量级约 0.019。
      （v3.1 把 N_H1 改为 1544，该缩放的中心量级不变，**预期区间 [0.015, 0.050] 未改动**；
        本节三条预期在 v3.0 → v3.1 之间一字未动。）
  由于 v3 同时改变了 S*、nested stacking、重复交叉拟合与样本总体，
  该缩放**不能直接迁移**，故把预期区间扩大到 [0.015, 0.050]，
  并**有意让区间跨越 0.04 的精度阈值**。

  ⚠️ 该区间**不引用 §7.3 收敛研究的 width**（该产物 must_not_be_used_for 已封死），
     **不构成功效保证**。
  ⚠️ 实测值落在区间外**不算协议失败**，也**不得**触发追加样本、
     修改 N/F/R/B_per_batch 或重跑择优（§5 扩容承诺）。
```

---

## 11. 本稿相对第 1 稿的修订记录（2026-08-04）

| # | 第 1 稿 | 本稿 | 原因 |
|---|---|---|---|
| 1 | M1 只改红线方向，仍比 rung4 vs shortcut | 引入 S\*（含 identifier identity）；T1 与 Δ_H 保护分离 | 只改方向未触及"比较对象不是主基线"这个真病灶 |
| 2 | M2 以 C 为 primary，称其"只可能压低性能" | A 为 primary；C 降 sensitivity；E2 改可估性审计；E3 独立 estimand | 该说法无依据（撤回）；C label-dependent 且 grounded 仍跨折 |
| 3 | 明知大概率 precision insufficient 仍拟冻结 480×6 | N、K 由 §7 性能盲规划确定，填实前不得冻结 | 不接受明知功效不足仍冻结 |
| 4 | 「两 shortcut 属 IUT 无需校正」「Holm 作用于 percentile 尾部比例」 | 角色 I/II 分离；simultaneous one-sided bounds；p 值须置换或重中心 | 两处统计错误 |
| 5 | 「六处全部净收紧」 | §9 分类表，明标两处 estimand 重定义与一处放松 | 原说法不成立且是自我辩护 |
| 6 | 用 C 的 sizing 曲线宣布"核心 Δ_H 不可达" | 撤回；在 A + S\* 下重做规划 | 该结论只在 C 为 primary 时成立 |
| 7 | 以"K 不改变独立单元数"为由主张 N↑K↓ | 撤回（§7 留作记录）；**K 固定为 6，理由是保持 F5 定义** | 独立单元数 ≠ 精度；且 K<6 的劣势混杂 F5 退化，无法分离，不能作精度最优主张 |

### 11.1 本稿相对第 2 稿的修订（同日，用户第二轮批评）

| # | 第 2 稿 | 本稿 | 原因 |
|---|---|---|---|
| a | S\* primary = 逐 block 等权早融合，stacking 作附录 | **primary = nested stacking**，等权早融合降为对照 | 等权把维数悬殊的 block 拉平，高维文本 block 会稀释 B1/B4；stacking 在 train 折内学权重，更接近 S\* 想充当的"表层信息上界" |
| b | M1b 单 block 胜出 → 报告为"融合缺陷"、改写表述 | **核心主张作废** | S\* 若连自身组成部分都比不过，就没履行上界职能，`Δ_H` 相对它的增量不能承载原含义 |
| c | M7 拟"平均 CI / 平均 CI 宽度" | **重复交叉拟合只作用于点估计；CI 由对全部 repeats 聚合分数的一次 paired grouped bootstrap 给出** | CI 不是可平均的量 |
| d | 把 0.0067 称为 MC 误差 | **撤回**；MC 误差须固定 OOF 分数、只改 bootstrap 种子、重复 B_rep 次单独估计 | 那 8 次同时改了 fold 分配与 bootstrap 种子，测到的是混合物 |
| e | 称"重复 CV 能消掉一半可变性" | **撤回** | 未做分解，无依据 |
| f | K 由精度规划确定 | **K = 6 固定，理由是保持 F5 定义** | K<6 的劣势混杂 F5 退化，不可分离；不作精度最优主张 |

### 11.2 本稿相对第 3 稿的修订（同日，用户第三轮批评）

| # | 第 3 稿 | 本稿 | 原因 |
|---|---|---|---|
| g | M7：先平均 R 次 OOF 分数，再算一次 AUROC 差 | **逐 repeat 算 AUROC 差 → 在同一 group-bootstrap 抽样下跨 repeat 平均** | 平均分数定义的是一个 **R 成员 ensemble 预测器**的增量，属另一 estimand；若报告须显式标为 ensemble，不得与 primary 互替 |
| h | S\*、S\*+H 的构造未写死同构 | **S\* = 五块 stack，S\*+H = 六块 stack**，共享 folds / inner 划分 / base learner 与 C 网格 / meta 协议 / 标准化时点，唯一差别是第六块 | 否则 Δ_H 混入融合架构差异，测的不再是"加入 H 的增量" |
| i | 称 S\* "更接近表层信息的上界" | **降为"强操作性基线"** | S\* 只是一个操作化，受 block 集合 / base learner / meta / 样本量所限；无任何论证支持它逼近上确界。M1b 的反面推论也随之禁止 |
| j | MC 误差带与 B_rep 未写死 | **B_rep = 20 在此冻结**；MC 带 = `w̄ ± 1.96·s_w/√B_rep`；adequate/insufficient/borderline 三判据按带的端点写死 | B_rep 不得由 B_rep 的结果反推，那是看着带宽调带宽 |
| k | §7 同时规划 N 与 K；R/n_boot 待"收敛研究" | §7 **只规划 N**；新增 **§7.3** 把 R/n_boot 的候选网格、种子、收敛阈值、停止规则、回退值全部预冻结 | 收敛研究若可边跑边调，等于跑到满意为止 |

### 11.3 本稿相对第 4 稿的修订（同日，用户第四轮批评）

| # | 第 4 稿 | 本稿 | 原因 |
|---|---|---|---|
| l | MC 误差 = B_rep 个**独立主 CI** 的宽度均值 | **单一合并重采样分布**：`B_total = 20 × n_boot` 全部 `Δ̄^(b)` 合并取 2.5/97.5 百分位 = 主 CI；20 批**仅**作计算与 MC 诊断 | 多个独立 CI 求平均不是一个 well-resolved CI；主估计必须来自一个分布 |
| m | MC 不确定性算法未写死 | **batch-means**：批内取 `L_j,U_j,w_j` → `se_MC(·) = sd(·_j)/√20`；带以**合并分布**的 `width` 为中心，非 `mean(w_j)` | 需要一个可执行、可复核的算法，且要点明 `mean(w_j)` 与 `width` 是不同的量 |
| n | §7.3 first-hit 停止、相邻档比较 | **跑完整网格**；`g* = min{g : ∀g'≥g, D(g') < tol}`，一律**与最大档**比较 | 相邻档在噪声下会"偶然相等"而误判收敛；first-hit 使结论依赖扫描顺序 |
| o | R 与 n_boot 用同一套收敛判据（都含点估计） | R 查**点估计 + 两端点**；n_boot **只查两端点与宽度** | **n_boot 完全不进入点估计** —— 把它写进判据是多余且误导 |
| p | M1b = "没有单 block 显著击败 S\*" | **simultaneous non-inferiority 认证**：全部 5 块 `L_b > −ε_NI`（ε_NI=0.02）才算 CERTIFIED；无"默认通过" | "未能拒绝劣性"不是合格的证据；样本量不足时会自动成立 |
| q | "S\* = 推理时可得的**全部**表层信息"；M1c "S\* 已包含 shortcut 所用的全部信息" | 改为"一个**预先指定的** block 集合"；M1c 改为"两个 shortcut 的表示**本身是 S\* 的组成 block**" | 两处都是穷尽性的过度声称 |
| r | 回退值依据引用"实测 1000→4000 跨越阈值，故 1000 明显不足" | 依据改为**纯先验的 √n 关系** | 用待验证的观测去论证冻结值，是循环 |
| s | meta-learner 只写"相同的正则设定" | **逐项冻结**：penalty/C=1.0 固定不搜索/solver/max_iter/tol/intercept/class_weight/random_state/输入 logit 变换 | 若 meta 也 nested 搜索，五块与六块可能选到不同 C，`Δ_H` 混入 meta 正则差异 |

---

### 11.4 本稿相对第 5 稿的修订（同日，用户第五轮审查）

| # | 第 5 稿 | 本稿 | 原因 |
|---|---|---|---|
| t | M1b 同时写了三件互斥的事：`L_b` 是 family-wise simultaneous 下界、IUT 无需校正、simultaneous 界"额外报告" | **Primary = 逐块边际单侧 95% 下界**（IUT，无校正）；**Secondary = family-wise simultaneous 下界**（更保守的敏感性分析） | 三者不能同时作 primary；若 primary 用同时界，那本身就是一次校正，与"IUT 无需校正"自相矛盾 |
| u | `CERTIFIED` 表述含 "family-wise" | 明令**不得**写成 "family-wise certified"；正确表述为"逐块非劣（marginal one-sided 95%，IUT 下整体 α≤0.05）" | primary 用的是边际界，措辞必须与实际口径一致 |
| v | M1b 的界沿用旧 M3 单次折分口径 | **必须与 M7 完全同口径**：R 次重复交叉拟合 + 同一 group-bootstrap 抽样下的合并分布；S\*、各 block、H 的 OOF 来自同一批 repeat 与 folds | 口径不同则与 Δ_H 不可比 |
| w | `ε_NI = 0.02` 的理由是"与全文同一尺度" | **独立论证**：它是预先选择的操作性容忍度，不由 ε 或任何等效性阈值推出；并写明 M1b 只认证"在该容忍度内不劣于所选 block" | "与其他阈值同值"不构成非劣界的论证 |
| x | 未处理"增量可能只是补回 S\* 构造损失" | **新增 M1b(iv)**：ε_NI 允许 S\* 比最佳单块差近 0.02，若 Δ_H 也在该量级则两种解释无法区分。要主张"超过最佳表层 block"须另行满足 **(iv-a)** 逐块直接检验 `AUROC(S*+H) − AUROC(B_b) > 0`（IUT，边际界）或 **(iv-b)** `Δ_H` 的 CI 下界 > ε_NI。**(iv-a) 作为预注册项执行。** | 这是本轮最实质的漏洞：认证通过不等于增量真正超出最佳表层块 |
| y | MC 误差用 batch-means + 1.96 | **Primary = delete-one-batch jackknife**（直接估 pooled 分位数的 MC 误差）；batch-means 降为 secondary；带宽用 **t₍₀.₉₇₅,₁₉₎ = 2.093** | jackknife 直接作用于我们真正报告的 pooled 量；df=19 时正态分位数会低估带宽 |
| z | "20 批足以判定 0.04 是否落在带内" | 改为"**预先固定的诊断预算**"；明写不声称足以判定任何事，MC 带跨越 0.04 即 borderline | 那正是 borderline 这一档存在的意义，不是 B_batch 取小了 |
| aa | `n_boot` 既像总次数又像每批次数 | 改名 **`B_per_batch`**；`B_batch = 20`；`B_total = B_batch × B_per_batch` | 原命名下 `n_boot=20000` 看起来是两万次，实际是四十万次，实现与论文都极易混淆 |
| ab | §6、§9 仍用旧 M1b 判据（`U_b < 0`、"单 block 显著优于 S\*"）；§5 把 M1b 归入角色 I/II | 全部替换为非劣认证口径；§5 明写 M1b **不适用**角色 I/II 二分；M3(1) 补"适用范围"，声明不适用于 M1b primary | 旧判据残留会让实现按错的规则写 |
| ac | §7.1 "同规模 Stage 1 精度判定近乎随机" | 改为"**可能对折分与有限 bootstrap realization 敏感，其稳定性尚未得到保证**" | 原措辞超出已有证据能支持的强度 |

---

### 11.5 本稿相对第 6 稿的修订（2026-08-06，用户第六轮审查）

| # | 第 6 稿 | 本稿 | 原因 |
|---|---|---|---|
| **ad** 【P0】 | M3(1) 用 `T_b = max_r (Δ̂_r − Δ_r^(b))/ŝe_r` 造**上界** `U_r = Δ̂_r + c_α·ŝe_r` | **两个方向分开定义**：下界用 `T^L = max_r (Δ̂_r − Δ_r^(b))/ŝe_r`，上界用 `T^U = max_r (Δ_r^(b) − Δ̂_r)/ŝe_r`；`c_α^L ≠ c_α^U`。指派：M1b secondary 用下界；M1c(T1) 判据方向为上界 → 用上界 | 枢轴量方向写反：用下界方向的统计量构造上界，覆盖概率不成立 |
| **ae** 【P0】 | §6 执行顺序写"逐样本对 R 次取平均 → 聚合 OOF 分数" | **保留 R 份独立 OOF**；在每个 repeat 内分别算 `Δ_r`、`Δ_{b,r}`、(iv-a) 逐块直接差；**再跨 repeat 平均统计量**。聚合 OOF 只能作明确标注的附录 ensemble 对照，不得进入 M1b / (iv-a) / 主 Δ_H | §6 把 M7 已明确排除的 **ensemble estimand** 又写了回来，两处自相矛盾 |
| **af** 【P0】 | 「(iv-a) 通过 **OR** (iv-b) 通过 → 可主张超过每一个单 block」 | **(iv-a) 为唯一授权 primary 判据**；(iv-b) 降为 sensitivity/代数一致性检查，**不单独授权任何表述**，且必须写成合取「M1b CERTIFIED **且** Δ_H 下界 > ε_NI」 | 两条路径的并集 = 择优触发 + look-elsewhere，正是本协议在别处反复防的东西 |
| ag | meta 输入"决策分数做 logit 变换" | **直接使用 `decision_function`，不再变换**（它本身已是 log-odds）；等价写法 `logit(clip(predict_proba))` 仅作注记，**规格只取前者** | 对 log-odds 再做 logit 是实现级错误；规格不得二选一并存 |
| ah | M7 小标题仍称"batch-means 估计量"；§6 仍写"按 20 批做 batch-means" | 统一为 **primary = delete-one-batch jackknife**，batch-means 为 secondary | 第 6 稿已把 primary 换成 jackknife，标题与执行顺序未同步 |
| ai | `n_max` / `D_n` / `stage1_convergence_R_nboot.json` | `B_max` / `D_B` / **`stage0_development_convergence_R_Bperbatch.json`** | 与 `B_per_batch` 改名同步；文件名标 `stage0_development` 以杜绝被引为 Stage 1 证据 |
| aj | 网格各档未规定随机数复用 | **共同随机数**：较小 `B_per_batch` 必须取每批在 `B_max` 档下 RNG 流的**前缀** | 否则档间差异混入纯 RNG 抖动，收敛比较不成立 |
| ak | ε_NI 论证含"与 AUROC 常见有效数字量级相当" | 改为「**透明但未经外部验证的操作性选择**」，并明禁以报告精度作依据 | 小数位数是报告惯例，不是科学论证 |

---

### 11.6 本稿相对第 7 稿的修订（2026-08-06，用户第七轮审查）

| # | 第 7 稿 | 本稿 | 原因 |
|---|---|---|---|
| al | 两处仍写"M7 的 batch-means 程序" | 统一为「M7 的独立分批 MC 诊断（**delete-one-batch jackknife 为 primary，batch-means 为 secondary**）」 | primary 已改 jackknife，主叙述未同步；§11 历史表中的旧名保留 |
| am | 覆盖率写成 `P(∀k: Δ_k ≥ L_k) ≥ 1−α` | 改为 **nominal**：`≈ 1−α`，并写明依赖 grouped-bootstrap 有效性与常规正则条件；明禁声称无条件有限样本不等式 | max-statistic bootstrap 给的是渐近/名义覆盖，不是严格保证 |
| an | M1b「IUT 下整体 α ≤ 0.05」无限定 | 补限定：**前提是各边际单侧程序确实达到 α=0.05 有效水平**；本文 bootstrap 实现对应**名义**水平 | 同上，IUT 的 size 结论继承边际程序的水平性质 |
| ao | `r` 在 M3 表 rung、在 M7 表 repeat；`b` 兼表 block 与 bootstrap draw | **新增记号约定表并全文统一**：`r`=repeat，`k`=comparison/rung/block，`q`=bootstrap draw，`j`=MC batch。M3 公式改为 `T_q^L = max_k (Δ̂_k − Δ_k^(q))/ŝe_k` 等 | 下标串位是实现期最易犯且最难查的错误 |
| **ap** | 实现规格不自足：M1a 只写"同一 nested 选法"，§8 只说"folds 数不变"，实现者仍须回查 v2.4 或代码 | **新增 M8**，从 Stage 0 现行实现只读提取并冻结：折数 / splitter / 分层与 group 规则 / retry 与 fallback / base learner 全部参数 / C 选择目标（inner-fold mean AUROC）/ tie-break（最小 C）/ inner CV 不可行时回退 C=1.0 **且必须记录** / **shared block 在 S\* 与 S\*+H 必须复用同一次 base 拟合与 OOF 分数** | v3 自称唯一设计权威，就必须自足；否则实现会各自解释 |
| aq | — | M8 同时记录与 Stage 0 的五处差异：补 `random_state=0`（原未设，可复现性缺口）、meta 输入改 `decision_function`（原 `_risk_scores` 优先返回 probability）、C 回退须记录、shared block 复用须回归测试**实测**、tie-break 由实现巧合升为契约 | 差异必须显式，否则实现者会以为"照旧即可" |

---

### 11.7 B5 定义澄清（2026-08-06，adapter sprint 前，用户指示）

| 项 | 原（第 8 稿） | 现 | 原因 |
|---|---|---|---|
| B5 | `family / route 等推理时已知量`（隐含 dense 类别量） | **text block**：`prompt_text` + 固定 family/route 标记 | H 本身编码 prompt；Stage 0 `rung1_prompt_only = 0.6829`。把 prompt 排除在 S\* 外会让 `Δ_H` 混入**输入信息增量**，而那与"内部状态是否超出表层"无关 |

**未改动**：块数（S\* 五块 / S\*+H 六块）、estimand 定义、判读规则、任何阈值。
**不新增 B6。** 本次为定义澄清，v3 仍**未冻结、未写 lock**。

SHA-256 变更记录：
```text
澄清前  57705a1787edecb3572ef80b63bcdb9a9ea0b7413cbb063b1077b986ec635d10
澄清后  记录在 docs/progress/stage1_v3_analysis_history.md
        （不写在本文件内 —— 把哈希写进被哈希的文件会自指，每写一次值就变一次）
```
v3 仍**未冻结、未写 lock**；此处只作变更留痕。

---

### 11.8 F / N / N_MCQ 填实（2026-08-08，用户裁决 + 只读容量审计）

| 项 | 原 | 现 | 原因 |
|---|---|---|---|
| `F`（M2d） | `[[待定]]`；曾拟 `F ∈ {10,20,50}` + "非空块数 ≥ 5·n_outer_folds" | **F = 5**，salt 与 hash 规则逐字冻结 | 词表块**本身充当 outer fold**，必须与 `n_outer_folds=5` 对齐；原规则会机械排除 10/20、锁死 50，并稀释块内正例 |
| E3 可估性 | 无条款（只有 E2 的 M2b） | 新增 **M2d-gate**：任一 fold 为空 / 单类 / 训练补集单类 → `NON-ESTIMABLE` | 与 M2b 对称；且明禁改 F、换 salt、重新分块与事后正例门槛 |
| `N`（M6/§7） | `[[待定]]`，待"§7 性能盲规划" | **N_H1 = 1547**，定性为**容量上界决策**，新增 **§7.5** | 旧模拟跑在旧 estimand 上不适用；v3 口径下唯一的宽度被 §7.3 封死且已被看过。取上限不查询任何宽度，是唯一可证明与已见结果无关的选法 |
| attack+cwe 上限 | 1,545（141/141） | **1,547（140/140）** | 141 把 open_gen 的 `source_entry_id` 哨兵 `"None"` 计成了条目；open_gen 是 topic-based，不消耗条目 |
| MCQ | 只写来源，未写题数与选法 | **N_MCQ = 1760**，选题 salt/排序规则逐字冻结 | 与 Stage 0 的 nominal 1760 同规模，便于 §6 的 `D = S_MCQ − S_H1` 可比 |
| 新池题数 | "8,187"（无口径说明） | **8,187**（`10,178 distinct − 1,991 overlap`），并写明 `8,189` 为未去重的错误口径 | 由容量审计重算，不在两个数字里任选 |

**新增只读产物**：`outputs/logs/stage1_capacity_audit.json`
（`scripts/audit_stage1_capacity.py`，只读身份/计数/重叠/哈希，
`reads_labels_or_performance: false`，`selection_materialized: false`）。

**未改动**：M1 全部、M3、M5、M7、M8、判读顺序、任何阈值、K=6、R=3、B_per_batch=1000。
**§10 的三条预期仍未写死**，故 v3 **仍未冻结、未写 lock**。

---

### 11.9 冻结轮（2026-08-08，用户第五轮裁决）

| 项 | 原 | 现 | 性质 |
|---|---|---|---|
| §10 三条预期 | 未写死（唯一未填项） | **T1 预期触发**（依据 Stage 0 已公开的 `−0.093 / 上界 −0.054`）、**E2 预期 NON-ESTIMABLE**（沿用）、**E1 预期 width ∈ [0.015, 0.050]**（只用已公开结果 + 1/√N 先验，有意跨越 0.04） | 填实 |
| H1 primary population | 未写 route 组成 | **recall only**，family ∈ {attack, cwe}，N=1547，K=6，9,282 traces；`route` 枚举保留、B5 中退化为常量不构成错误；**禁止外推到 open_gen** | 填实 + 外推禁令 |
| `F` / 两个 salt / population | 只在预注册与审计脚本常量里 | **并入 `v3_config.FROZEN_FIELDS`**，与 R/B 同等强制 | 由散文升为代码契约 |
| 冻结状态 | 草案第 8 稿，未冻结 | **v3.0 已冻结**，lock = `preregistration_v3.lock` | 冻结 |

**未改动**：M1 全部、M2（除 M2d 已于 §11.8 填实）、M3、M5、M7、M8、判读顺序、任何阈值、
`K=6`、`R=3`、`B_per_batch=1000`、`N_H1=1547`、`N_MCQ=1760`、`F=5`。

---

### 11.10 v3.0 → **v3.1**（2026-08-08）：G5 发现的数据身份冲突

**这是 v3.0 冻结后的唯一一次 bump。** 起因不是设计反悔，而是
**G5 在任何模型运行之前**、仅凭数据身份发现的冲突：

```text
根因  ATT&CK 的 revoked 条目与其继任者共享 `name` + `description`，
      而 `recall_v1` 模板只用这两个字段 → 产出**逐字节相同**的 prompt。
      索引含 revoked 149 / deprecated 12（attack）与 Deprecated 25（cwe）。
      Stage 0 每族只抽 140 条，未撞上；**取满整池才暴露**。
```

| 项 | v3.0 | v3.1 | 理由 |
|---|---|---|---|
| `N_H1` | 1547 | **1544** | 1547 − 2（跨 Stage prompt 碰撞）− 1（正式集内同 prompt 去重） |
| H1 traces | 9,282 | **9,264** | 1544 × 6 |
| 总 traces | 19,842 | **19,824** | 9,264 + 10,560 |
| 内部重复的处理 | （未预见） | **确定性去重**，非"同 prompt 共享 group" | 保留两条会让该 prompt 得到 12 条生成而其他只有 6 条（不等权），也违反"正式集内 prompt 零重复" |
| 排除规则 | （无） | 逐字冻结于 M6 | 先排跨 Stage 碰撞，再按 `g5_prompt_hash` 去重；优先保留非 revoked/deprecated，仍并列取字典序首个 |

**被排除的三条**：`attack:T1134.004`(active，≡ Stage 0 的 `T1502`)、
`attack:T1656`(revoked，≡ Stage 0 的 `T1684.001`)、
`attack:T1536`(revoked，与 active 的 `T1578.004` 同 prompt，按规则保留后者)。

**未改动**：判读规则、全部估计量（M1/M2/M3/M5/M7）、M8、所有阈值、
`F=5`、`R=3`、`B_per_batch=1000`、`K=6`、`N_MCQ=1760`、**§10 的三条预期一字未动**、
route = recall only、外推禁令。

> **必须与结果同报的性质声明**：本次 bump 的触发信息是「两条 prompt 的文本是否逐字节相同」，
> **不含任何性能量**、不含任何 AUROC/Δ/CI、不涉及择优。若把它写成"我们调整了样本量"
> 而不说明触发条件，读者无从判断它是否是 result-dependent 的改动 —— 故此处写死。
>
> ⚠️ 另记：`1547 − 3 = 1544`，而 §11.8 曾把一个**错误的** 1545 更正为 1547。
> 三个数字来源互不相同（1545 是把 open_gen 的 `"None"` 哨兵误计为条目），
> **不得**把它们混为一谈，也不得据此说"当初的 1545 本来就是对的"。

**其余 revoked/deprecated 条目继续保留**（理由见 M6）：estimand 判断的是
identifier 是否**存在于冻结 ontology**，不是它是否仍被推荐使用。状态分布进 manifest 与局限说明。

---

## 12. 边界

不得声称：attention guidance 有效、hallucination reduction、answer accuracy improvement、
intervention 有效。Stage 1 与 Stage 0 一样，只做 detection 方向的条件增量与识别条件审计。

### 12.1 冻结的强制程度与执行顺序（**如实陈述，不得夸大**）

```text
**状态（2026-08-08，v3.1）：G2 已在 sample-builder 上接线；generation 入口仍未接线。**
  ✔ 已有：preregistration_v3.lock（记录 v3.1 的 canonical-LF sha256，`hash_mode: text-lf-utf8`）
          evaluation/v3_freeze.py::{check_v3_frozen, require_v3_frozen}
          "篡改即拒" 的回归测试（含 CRLF/LF 互转仍匹配、BOM 具名报错）
          **evaluation/v3_sample_builder.py 入口第一行即 require_v3_frozen()**，
          G2 不过则不创建任何名单、不写任何文件（有测试实证）
  ❌ 尚无：Stage 1 的 generation / analysis 入口。
  ⇒ 现状是**名单构建路径已端到端强制，生成路径尚未**。
     ❌ 不得声称"冻结已在全部生产路径上强制"。
     （v2.4 的教训同款：由代码强制，不只写在 lock —— 散文会悄悄漂移。）

冻结后的执行顺序（固定，不得跳步）：
  v3 lock                             ✔ 完成（v3.0 → v3.1）
  → Stage 1 入口 + G2/G5 接线          ✔ 完成（sample-builder）
  → 正式样本名单物化与互斥审计          ✔ 完成
  → G3 smoke（≤20 prompt 真实模型）    ← 下一步
  → G4 性能盲质量检查（gate 样本，非正式确证样本）
  → 正式生成
```
