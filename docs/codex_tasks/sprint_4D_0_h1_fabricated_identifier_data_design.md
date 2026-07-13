# Sprint 4D-0：H1 编造标识符任务的数据与标签设计（design + data only）

## 1. 定位

本 sprint 是 H1 fabricated-identifier 主线（`CYBER_HALLUCINATION_CONTROL_PLAN.md`
§3 / §14.4）的**第一张卡**：只做本体快照、标识符抽取/归一化、H1 canonical
schema、诱发问题集构造和数据审计。**不调用 causal LM，不生成 completion，不算
F5，不训练 probe，不做 steering。**

命名说明：4A 路线图里的旧 "Sprint 4D = Probe-Guided MLP Readout Steering" 已随
计划书被 `CYBER_HALLUCINATION_CONTROL_PLAN.md` 取代而作废。自本卡起，Sprint 4D
线 = H1 fabricated-identifier DETECT 任务；MCQ 内部特征检测已由 4C 关闭
（detector_beats_f5=false），不得在 MCQ 上追加内部特征工作。

本卡回答三个设计问题：

```text
Q1：H1 的「编造」如何被构造性验证？——完整本体快照 + 抽取/归一化 + 存在性判定，
    全程无需人工标注、无需 LLM judge、无需推理期 gold。
Q2：什么样的问题能诱发模型吐出标识符，同时保留推理文本与采样多样性？
    ——两条诱发路线（recall / open-generation）的确定性构造。
Q3：自由生成下 F5 kill baseline 是什么？——设计文档（本卡产出定义，4D-1 实测）。
```

默认结论标记：

```text
ready_for_2000_rerun=False
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
h1_dataset_designed=UNSET（由本卡实测填）
h1_emission_viability=UNSET（4D-1 才能填，本卡禁止预判）
```

---

## 2. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md（重点 §3 H1 定义 / §8 数据要求 / §14.4）
docs/reference/STORY.md（重点末段：4C 关门与 H1 动机）
docs/codex_tasks/sprint_4B_1_cybermetric_schema_and_label_proxy.md（schema/split/audit 纪律母版）
docs/progress/sprint_4_history.md（4B/4C 已建立的接口与教训）

src/recover_attention/schemas.py（cyber_sample 验证器——H1 schema 平行新增，不改动既有）
src/recover_attention/cyber_data.py（确定性构造 / grouped split / audit 的复用范式）
src/recover_attention/domain_label_proxy.py（parse 负例纪律参考：AES/BERT/CVE/DNS 不误报）
```

---

## 3. H1 标签定义与硬性条款

### 3.1 三级标签，不得混用

```text
对 completion 中每个被抽取的标识符 id：
  grounded          id 存在于对应本体快照中；
  fabricated        id 不存在于对应本体快照中（H1 阳性，构造性可验证）；
  echoed            id 在 prompt 中出现过 → 不计入任何证据，单独计数。

对 recall 路线（有 eval-only gold id）额外记录：
  wrong_mapping     id 存在但 != gold（这是 H2，不是 H1；只记录，不进 H1 二分类）。

completion 级标签：
  h1_positive = completion 含 >= 1 个 fabricated id（echoed 除外）。
```

gold id（recall 路线）只作 eval label，绝不作 inference feature；沿用
`assert_no_gold_label_leakage` 纪律，H1 record 形态必须纳入 leakage 测试。

### 3.2 归一化纪律（硬性）

```text
CVE：大小写不敏感；"CVE 2021-44228" / "CVE-2021-44228" / "cve-2021-44228"
     归一为 CVE-2021-44228；年份 4 位、序号 >= 4 位（保留原始串以备审计）。
ATT&CK：T1059 与 T1059.001 是不同粒度；子技术存在性单独判定
     （T1059.999 不存在即 fabricated，即使 T1059 存在）；保留 parent/sub 字段。
CWE：CWE-79 / "CWE 79" 归一为 CWE-79。
抽取正则必须带负例测试：版本号（v2.0.1）、年份区间（2021-2024）、端口、
     哈希片段、"T-shirt"/"T5" 类普通词不得误抽。
```

### 3.3 本体快照完整性（硬性）

```text
存在性判定要求「完整 id 全集」，不是样本：
  CVE：完整 id 全集（如 CVEProject/cvelistV5 发布包或 NVD feeds），
       含 REJECTED/RESERVED 状态字段——RESERVED id 的判定规则必须显式写入
       快照 manifest（默认：RESERVED 视为存在但单独计数）；
  ATT&CK：enterprise-attack STIX（含 revoked/deprecated 字段，规则同上显式化）；
  CWE：官方完整列表（含 deprecated 字段）。
快照 manifest 必须记录：下载 URL、快照日期、文件 sha256、各 family 的 id 总数、
  状态字段处理规则、与模型 knowledge cutoff 的关系说明
  （快照晚于 cutoff：真实但 post-cutoff 的 id 会被判 grounded——方向安全，
   只会低估 fabrication rate；在 manifest 中记录该偏差方向即可）。
原始快照与 processed 产物 gitignore，manifest/审计报告进 git。
```

### 3.4 id 空间密度审计（本卡关键产出，硬性）

```text
「存在性检查」的区分度取决于 id 空间的稀疏程度：
  CVE-YYYY-NNNN 在常见年份低序号段近乎稠密 → 随机编造也可能「碰上」真实 id，
  存在性检查对 CVE 可能是弱判据；
  ATT&CK technique id（数百个）与 CWE id（约 1000 个 vs 5 位数字空间）稀疏，
  存在性检查判据强。
必须输出 id_space_density_report.json：每个 family 给出
  id 全集大小、格式上可编造的空间大小估计、按年份/号段的占用率、
  「随机合法格式串命中真实 id」的估计概率。
结论必须回答：CVE 的存在性判据是否足够强？若不够强，给出限制策略
  （如只统计高序号段 / 只用 ATT&CK+CWE 作主判据 / CVE 降级为辅助 family），
  并把该策略写进 labeling 规则。不得默认三个 family 判据等强。
```

---

## 4. Stage 1：本体快照下载与 manifest

脚本 `scripts/sprint_4D_0_download_ontology_snapshots.py`：

```text
下载三个 family 的完整本体到 data/raw/ontology/{cve,attack,cwe}/；
生成 outputs/logs/sprint_4D_0_h1_data_design/ontology_snapshot_manifest.json
  （§3.3 全部字段）；
下载失败或 id 全集不完整（如只拿到样本页）→ 如实停止，不得用部分数据冒充全集。
CVE 体量大：允许只持久化「id 全集 + 状态 + 少量元数据」的压缩索引，
  但 manifest 必须记录索引的生成方式与源文件 sha256。
```

## 5. Stage 2：标识符抽取与存在性判定模块

`src/recover_attention/h1_identifier.py`：

```text
extract_identifiers(text) -> list[IdentifierMention]
  （family、原始串、归一化串、字符 span、粒度字段）；
normalize_identifier(family, raw) -> str；
build_ontology_index(snapshot_dir) -> OntologyIndex（存在性 O(1) 查询，
  含状态字段；index 可序列化、带快照指纹）；
classify_mention(mention, index, prompt_text) -> grounded/fabricated/echoed
  （echo 判定基于归一化串在 prompt 中的出现）；
label_completion(...) -> per-mention 标签 + completion 级 h1_positive。
测试：每个 family 的正例/负例抽取、归一化往返、子技术粒度、echo 排除、
  RESERVED/deprecated 状态规则、leakage 检查覆盖 H1 record 形态。
```

## 6. Stage 3：H1 canonical schema 与诱发问题集构造

### 6.1 schema（`schemas.py` 平行新增 `h1_sample`）

```text
字段：example_id、route（recall/open_gen）、family、prompt 构造参数、
  question_text、source_entry_id（本体条目，即 recall 的 gold，eval-only）、
  source_entry_metadata（描述文本、产品/战术等）、group_id（见 6.3）、
  split、label_space="open_identifier"（显式区别于 MCQ 的 option letters）。
validate_h1_sample_record 与既有验证器同风格；不修改 cyber_sample。
```

### 6.2 两条诱发路线（都要构造，配额由密度审计后定，建议起点如下）

```text
Route A（recall，主路线，~300-450 题）：
  从本体条目确定性采样（sha256-seeded，参照 cyber_data 的 shuffle 纪律），
  用条目描述构造「该漏洞/技术/弱点对应哪个标识符」类问题；
  要求描述文本做最小改写规则（去掉直接包含 id 的句子——否则变成 echo 题），
  改写规则必须确定性、可测试；
  按 family 配额：ATT&CK 与 CWE 优先（判据强），CVE 配额依 §3.4 结论定。

Route B（open-generation，副路线，~100-150 题）：
  「列举并说明与 X 相关的 N 个 CVE/技术」类开放问题，X 从本体元数据
  （产品、战术、弱点类别）确定性采样；
  一个 completion 可产出多个 mention——schema 与标签逻辑必须支持多 mention。

两条路线的设计意图（写入 review gate）：
  Route A 给可控的 per-question 标签与 H2 对照；
  Route B 给更长推理文本与多样采样——这是 F2/F3 未来的原料。
```

### 6.3 grouped split

```text
group_id 不得只按题目哈希：recall 题的泄漏单位是「本体条目」，
  open_gen 题是「主题实体（产品/战术）」；同一条目/实体的题必须同组；
train/dev/test 分组切分，泄漏检查 = 0；
（H1 检测不训练生成模型，split 主要防 detector CV 的题目级泄漏，仍强制执行。）
```

## 7. Stage 4：审计与 F5 设计文档

```text
数据审计（h1_dataset_audit_report.json）：
  题目数 per route/family、去重（归一化问题文本）、长度分布、
  echo 自查（构造出的问题文本中不得残留 gold id——全量断言，不是抽样）、
  split 统计与泄漏检查。

F5 设计文档（h1_f5_design.md，设计输出，非代码）：
  自由生成下没有有限标签空间，MCQ 的 label_margin/label_entropy 不可直接迁移。
  必须给出 H1 的 F5 kill-baseline 候选清单与理由，至少覆盖：
    mention 级：id token 段的 mean/min logprob、id 首 token 的 top-k rank；
    序列级：completion perplexity、长度归一 logprob；
    采样级：K 采样下同一 slot 的 id 一致性（self-consistency）、
            语义熵式聚类是否适用（id 是精确串，聚类退化为精确匹配——说明之）；
    口头置信：verbalized confidence 的提取协议。
  并写明：F5 在 H1 上预期比 MCQ 弱（这正是选 H1 的理由），但 kill gate 纪律
  不变——任何内部特征（F1-F4）必须给出相对 H1-F5 的分组 CV + CI 增量。

4D-1 smoke gate 预注册（写入 review gate，本卡不执行）：
  4D-1 只测两个数：id emission rate（completion 含合法格式 id 的比例）与
  fabrication base rate（mention 级 + completion 级）；
  预注册可行性门槛：emission rate >= 0.7（Route A）且
  fabrication base rate ∈ [0.05, 0.60]（太低无正例，太高任务退化为噪声）；
  不满足 → 回到诱发设计，不得直接进特征工程。
  教训来源：4B-2 的 has_reasoning_text=0.00——先验证原料存在，再投入下游。
```

---

## 8. 输出清单

```text
data/raw/ontology/{cve,attack,cwe}/           （gitignore）
data/processed/h1/h1_samples.jsonl            （gitignore）
outputs/logs/sprint_4D_0_h1_data_design/
  ontology_snapshot_manifest.json
  id_space_density_report.json
  h1_dataset_audit_report.json
  h1_f5_design.md
  review_gate_h1_data_design.md
src/recover_attention/h1_identifier.py
src/recover_attention/schemas.py              （平行新增，不改既有验证器）
src/recover_attention/h1_data.py              （构造/split/审计）
scripts/sprint_4D_0_download_ontology_snapshots.py
scripts/sprint_4D_0_build_h1_dataset.py
tests/test_h1_identifier.py
tests/test_h1_data.py
```

## 9. 本次允许修改的文件

```text
§8 列出的新文件；
schemas.py（仅平行新增 h1_sample 相关）；
.gitignore（新增 data/raw/ontology/ 与 data/processed/h1/）；
PROGRESS.md、docs/progress/sprint_4_history.md、
docs/progress/sprint_4_artifact_manifest.md（完成后更新）。
```

## 10. 本次禁止修改的文件

```text
README.md / AGENTS.md / docs/reasoning-aware-attention-guidance/SKILL.md
docs/reference/*（计划书回填留给 review 后单独 doc commit）
既有 cyber_data.py / domain_label_proxy.py 的行为（只读复用）
```

## 11. 测试要求

```text
1. 抽取正则正负例（§3.2 全部负例类别）；
2. 归一化与粒度（CVE 大小写/空格变体、T1059 vs T1059.001、CWE 变体）；
3. OntologyIndex 存在性判定 + RESERVED/deprecated 规则 + 快照指纹校验；
4. echo 排除逻辑；
5. 问题构造确定性（同 seed 复现、异 seed 变化）与 gold-id 残留全量断言；
6. grouped split 泄漏 = 0；
7. h1_sample 验证器正反例；
8. leakage 检查覆盖 h1 record；
9. full pytest 通过。
```

## 12. Review Gate（逐条回答）

```text
1.  三个本体快照的日期 / id 总数 / sha256 是否入 manifest？
2.  RESERVED / revoked / deprecated 的判定规则是什么？
3.  id 空间密度结论：CVE 存在性判据强度？采取了什么限制策略？
4.  抽取负例测试覆盖哪些类别？全部通过？
5.  echo 排除如何实现？问题文本 gold-id 残留断言是否全量通过？
6.  Route A / Route B 各多少题？family 配额及其依据？
7.  group_id 的泄漏单位是什么？泄漏检查 = 0？
8.  h1_f5_design.md 是否覆盖 mention/序列/采样/口头四类基线并说明预期？
9.  4D-1 的 emission/fabrication 预注册门槛是否原样写入？
10. 是否调用了 causal LM / 生成了 completion / 训练了 probe？（必须 no）
11. gold id 是否只作 eval label？leakage 测试覆盖 h1 形态？（必须 yes）
12. 是否声称 hallucination reduction / accuracy improvement / H1 可行性？（必须 no）
```

## 13. 禁止事项

```text
不要调用 causal LM（tokenizer 也不需要——本卡无 token 级读出目标）。
不要生成 completion、算 F5、训练 probe、做 steering / patching。
不要把 id 用作 unembedding 读出目标（3C-0 leading-token 教训；H1 检测
  读出目标的设计属于后续卡）。
不要在 MCQ 上追加内部特征工作。
不要预判 h1_emission_viability——那是 4D-1 的实测。
不要下载模型。
```

## 14. 推荐命令

```bash
conda run -n recover_attention python scripts/sprint_4D_0_download_ontology_snapshots.py \
  --output-dir data/raw/ontology --manifest-dir outputs/logs/sprint_4D_0_h1_data_design

conda run -n recover_attention python scripts/sprint_4D_0_build_h1_dataset.py \
  --ontology-dir data/raw/ontology \
  --output-path data/processed/h1/h1_samples.jsonl \
  --report-dir outputs/logs/sprint_4D_0_h1_data_design \
  --seed 4242

conda run -n recover_attention python -m pytest tests/test_h1_identifier.py tests/test_h1_data.py -q
conda run -n recover_attention python -m pytest -q
```

## 15. 最多允许的结论

```text
An H1 fabricated-identifier dataset design exists: complete ontology snapshots
with dated manifests, a tested identifier extraction/normalization/existence
pipeline, a canonical h1_sample schema with grouped splits, an id-space density
audit that qualifies the existence check per family, and a pre-registered F5
design plus 4D-1 emission/fabrication viability gates. No model was called; no
claim about detection performance, emission viability, hallucination reduction,
or accuracy improvement is made.
```

## 16. 完成后必须更新

```text
PROGRESS.md（顶部 status + §6 下一步指向 4D-1）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
```
