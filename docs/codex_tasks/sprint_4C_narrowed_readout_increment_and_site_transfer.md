# Sprint 4C-narrowed：Readout 增量检验（F1+F4 vs 0.816）与位点迁移收尾

## 1. 定位

本 sprint 是 DETECT 层的**收窄版特征 bake-off**。4B-3 的实测结果坍缩了原计划
（`CYBER_HALLUCINATION_CONTROL_PLAN.md` §14）的五族设计：在 CyberMetric 选项字母
MCQ 上，F2（轨迹）无原料、F3（多采样一致性）近死、sampling 档对 single-forward
零增量。因此本轮**只测两族**，并顺带答掉 4B-3 悬空的 Q3。

本轮回答两个问题：

```text
Q1（核心）：F1（L20/L24 跨层读出投影，含层间 disagreement）
            + F4（4 个选项 token 的 exact J-lens 投影）
            能否在 F5 single-forward 门槛 AUROC 0.816 之上给出
            **稳定的增量 AUROC**（分组 CV、CI 下界 > 0）？

Q2（收尾）：先把 correct/wrong pair 挖到 >= 20，再跑 3C-1 位点迁移验证——
            答案读出位的 MLP 在 cyber MCQ 上是否 donor-specific + site-specific？
```

本轮**是** probe 训练 sprint，但仅限：在少量标量特征上训练线性 logistic 检测
probe（分组 CV）+ 校准温度。**不是** 激活空间 vector controller、**不是**
steering / nudge、**不是** LoRA / finetuning、**不是** 2000-scale、**不是** full 3C。
Stage 0 的诊断性 module patching 是唯一的激活干预，明确豁免。

默认结论标记：

```text
ready_for_2000_rerun=False
do_not_enter_full_sprint_3C=False（已不在 3C 语境，但保持 flag=false 惯例）
hallucination_reduction_proven=False
answer_accuracy_improvement_proven=False
steering_continued=False
probe_trained=True（仅线性检测 probe，见上）
detector_beats_f5=UNSET（由本轮实测填）
```

措辞纪律：任何"内部特征优于输出层"的声明必须带分组 CV + CI；单特征裸 AUROC
不构成增量声明。不得声称 hallucination reduction / accuracy improvement。

---

## 2. 执行前必读

```text
AGENTS.md
PROGRESS.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md（重点 §5 kill gate / §6 J-lens / §14 收窄）
docs/reference/STORY.md（重点 §22）
docs/codex_tasks/sprint_4B_3_full_f5_baseline_and_site_transfer.md（母卡，F5 与 pair 逻辑）
docs/progress/sprint_4_history.md（4B-3 数字）
docs/progress/sprint_3_history.md（3C-1 位点数字 / 3C-3 attribution / 3C-4A J-lens 教训）

src/recover_attention/domain_label_proxy.py（选项 token 双形态解析 / 读出位定位——唯一权威）
src/recover_attention/mlp_readout_attribution.py（unembedding 投影 / AUROC / 分组 bootstrap 复用）
src/recover_attention/module_causal_tracing.py（Stage 0 位点迁移直接复用）
src/recover_attention/approx_j_lens_readout.py（F4 参考——本轮改为 exact VJP，见 §6）
src/recover_attention/mlp_readout_direction.py（读出位 MLP 捕获参考）

scripts/sprint_4B_3_full_f5_baseline_and_site_transfer.py（trace/F5/pair 母版）
scripts/sprint_3C_1_final_answer_compression_value_mlp_tracing.py（module patching 对照）
```

---

## 3. Tokenization 与解析纪律（硬性条款，沿用 4B-3 §3）

```text
禁止：tokenizer.encode(" A")[0] 直接当字母 token（3C-0 leading-token 退化）。
必须：选项 token 只通过 domain_label_proxy.option_token_ids（空格形态）与
      bare_option_token_ids（裸字母形态）解析；读出位断言用 dual-form 匹配
      （4B-3 已实现：chat 条件 239/239 裸字母），报告 token_form_counts。
必须：答案解析只通过 parse_option_answer（三级优先级，parse_failure 独立类别）；
      不得把 AES/BERT/CVE/DNS 等普通术语误解析（既有负例测试保持通过）。
必须：gold_label 只作 eval label 与 probe 训练目标，绝不作 inference feature；
      每个 feature record 过 assert_no_gold_label_leakage。
```

---

## 4. Preflight

先输出：

```text
outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer/preflight_report.md
```

必须检查并记录：

```text
1. 4B-1/4B-2/4B-3 的 tracked 改动是否已 commit（工作区干净）；未 commit 停止；
2. 4B-3 输出存在且完整：
   outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/
     trace_sampling_manifest.jsonl / f5_baseline_report.json /
     correct_wrong_pair_manifest.jsonl / review_gate_*.md
   缺失则停止；
3. 从 4B-3 f5_baseline_report.json 读取并记录 F5 基线：
   kill_bar_single_forward AUROC + CI（本轮增量对照的锚点）；
   winning prompt style（应为 chat）；
4. model_path 解析：--model-path > 环境变量 RECOVER_ATTENTION_MODEL_PATH >
   fallback D:/models/Qwen2.5-7B-Instruct；三者不可用则停止；记录 model_path_source；
5. 不覆盖任何既有输出目录（sprint_3* / sprint_4A* / sprint_4B_*）；
6. 本轮 probe = 线性 logistic（标量特征），零激活空间 controller / steering；
   Stage 0 module patching 为诊断性、豁免。
```

---

## 5. Stage 0：Pair mining + 3C-1 位点迁移（收尾 Q2）

### 5.1 Pair mining

```text
起点：4B-3 correct_wrong_pair_manifest.jsonl（8 对）；
候选题：从 4B-3 f5_baseline_report / trace manifest 选「模型不确定」的题——
  f5_label_margin 最低 或 f5_self_consistency < 1 的题（约 40-60 题）；
加采：仅对候选题，temperature 1.0-1.2、每题 10-15 采样，
  复用 4B-3 的 KV-cache 生成路径与 dual-form 解析；
目标：合并 4B-3 已有采样，凑到 >= 20 个同题 correct/wrong pair；
      解析成功、非退化、非 gold-echo；
输出 pair_mining_report.json（候选题数 / 加采量 / 最终 num_pairs）
      与 correct_wrong_pair_manifest.jsonl（合并去重）。
如仍 < 20：如实记录 insufficient_pairs，Stage 0.2 site-transfer 跳过并说明，
      不得强行降低 pair 质量门槛。
```

### 5.2 位点迁移验证（gate：num_pairs >= 20）

```text
复用 module_causal_tracing，patch 目标 = wrong trace 的 label-readout 位置，
donor = 同题 correct trace 同位置；
modules = [attention_output, mlp_output, residual_output]；
layers = [16, 20, 24]；alpha = [0.25, 1.0]；上限 34 对（与 3C-1 可比）；
conditions = no_patch / correct_donor / random_donor /
             same_trace_random_position / wrong_donor_self；
指标 = gold 选项字母 logprob delta - wrong 选项字母 logprob delta
       （单 token 标签 → 单步 logprob，报告注明；token id 按该 trace 实际 form 取）；
判定 = donor-specificity 与 site-specificity 分组 bootstrap CI95；
chat 裸字母适配（沿用 4B-3 卡 §10）：读出位落在脚手架末端，报告读出位到
  prompt 末 token 的距离分布；completion 只 1 token 时 same_trace_random_position
  无位置可选 → 跳过并计数，不得用 prompt 内位置冒充；
输出 site_transfer_check_report.json，含与 3C-1 GSM8K 的并排对照表：
  GSM8K(3C-1): mlp donor +0.141 / site +0.353；attention donor +0.093 / site -0.006；
               residual donor +1.703 / site -0.547
结论三选一：迁移 / 部分迁移 / 不迁移，并写明对 4C F1 动机的影响。
```

---

## 6. Stage 1：F1 / F4 特征捕获（本轮核心）

对全部 240 题的 greedy run，在 label-readout 位置（dual-form 断言）捕获激活并
计算两族特征。**全程无 gold 参与特征计算。**

### 6.1 F1 —— 跨层读出投影（direct logit-lens）

```text
捕获 L20、L24 的 MLP output（复用 mlp_readout_direction / module_causal_tracing
的捕获逻辑）在读出位的向量；
投影到 4 个选项 token 的 unembedding 行（W_U[option_tokens]，token id 按实际 form）：
  per-layer 特征：
    f1_margin@L      = 选项投影 logit 的 top1 - top2
    f1_entropy@L     = 选项投影分布的熵
  跨层特征：
    f1_layer_disagreement = L20 与 L24 选项投影分布之间的 JS 散度
    f1_top_flip           = L20 top1 与 L24 top1 是否不同（0/1）
    f1_final_vs_mid_kl    = final-logits 选项分布 与 L24 投影分布的 KL
说明：跨层特征是核心假设——中层冲突被 final layer 抹平时，
      layer_disagreement / top_flip 携带 final entropy 没有的不确定性信号。
```

### 6.2 F4 —— exact J-lens 标签投影（VJP，非有限差分）

```text
背景：3C-4A 用有限差分近似 J-lens，与直接投影对齐弱（top-1 match 0.074，Case C），
      悬而未决：是近似太糙，还是直接投影真的误读中层？有限标签使 exact 可行。
实现：对每个 (layer ∈ {20,24}, 读出位) 与每个选项 token t：
      exact J-lens 分量 = ∂ logit_final[t] / ∂ h(layer, 读出位) 方向上的读数，
      即对 logit_final[t] 做一次 VJP（k=4 次反向传播/层，需 requires_grad）；
      得到 4 个选项的 J-lens 投影值 → 同样算 margin / entropy；
      f4_margin@L / f4_entropy@L / f4_layer_disagreement（同 F1 结构）。
成本：240 greedy × 2 layer × 4 token = 1920 VJP，单位置、单 backward，可接受。
不得用 approx_j_lens_readout 的有限差分冒充 exact；如复用该模块须新增 exact 路径。
输出 readout_feature_manifest.jsonl（per-question 的 F1/F4 标量特征 + F5 特征拷贝，
  过 gold-leakage 检查）。
```

---

## 7. Stage 2：增量检验（kill gate）

```text
检测任务：label = greedy 答错（wrong=1, correct=0；parse_failure 单列不进二分类，
  与 4B-3 一致，n≈239）；
评估协议（apples-to-apples，关键）：
  所有 AUROC 用同一 grouped-by-question CV 下的 logistic 组合得到，
  F5-alone 也走同样的 CV logistic（不能拿 F5 的裸单特征 AUROC 当基线，
  否则增量被 CV vs raw 混淆）；
  3 seeds，分组 bootstrap CI95（>= 500 次）。
报告矩阵：
  A. F5-alone（CV logistic）           —— 基线，应 ≈ 0.816
  B. F1-alone / F4-alone               —— 内部特征单独能力
  C. F5+F1 / F5+F4 / F5+F1+F4          —— 组合
  增量 = C 各项 - A，带 CI；kill gate = 增量 CI 下界 > 0。
J-lens 仲裁（计划书 §6）：
  比较 (F5+F4) 增量 与 (F5+F1) 增量：
    F4 有增量而 F1 没有 → direct logit-lens 误读中层（可发表方法学发现）；
    F1 ≈ F4            → 便宜读法够用，3C-4A 分歧是近似噪声。
输出 readout_increment_report.json（矩阵 + 增量 + CI + 仲裁结论）
      与 calibration_report.json（最优组合的 ECE / risk-coverage 曲线）。
```

---

## 8. 输出清单

```text
outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer/
  preflight_report.md
  pair_mining_report.json
  correct_wrong_pair_manifest.jsonl
  site_transfer_check_report.json（或 gate-skip 记录）
  module_patch_fidelity_report.json（Stage 0.2 执行时）
  readout_feature_manifest.jsonl
  readout_increment_report.json（核心：增量矩阵 + J-lens 仲裁）
  calibration_report.json
  high_risk_case_report.jsonl / low_risk_wrong_case_report.jsonl（沿用 4B-3）
  review_gate_readout_increment_and_site_transfer.md
```

同时更新：

```text
PROGRESS.md（新顶部 status block，含增量结论与位点迁移结论）
docs/progress/sprint_4_history.md
docs/progress/sprint_4_artifact_manifest.md
docs/reference/CYBER_HALLUCINATION_CONTROL_PLAN.md §14（把 detector_beats_f5 结果回填）
```

---

## 9. 测试要求

```text
1. F1 投影 / 跨层特征（JS / KL / top_flip）的纯函数单元测试（手构 logits 验证）；
2. exact J-lens VJP 的正确性测试：在小型可导 stub 上，VJP 结果与解析梯度一致；
3. 增量评估的 apples-to-apples 保证测试：F5-alone 与组合走同一 CV 接口；
4. §3 既有测试全部保持通过（option_token / bare / parse 负例 / leakage）；
5. gold-leakage 测试覆盖 readout_feature_manifest 的 record 形态；
6. full pytest 通过。
```

---

## 10. Review Gate（逐条回答）

```text
1.  F5 基线（从 4B-3 读取）= ? CV-logistic 复算后 ≈ 0.816？
2.  Stage 0 pair mining：候选题数 / 加采量 / 最终 num_pairs（>= 20?）
3.  位点迁移结论（迁移/部分/不迁移）？与 3C-1 对照表？对 F1 动机的影响？
4.  F1-alone AUROC？F4-alone AUROC？（含 CI）
5.  F5+F1 增量 = ?（CI 下界 > 0?）
6.  F5+F4 增量 = ?（CI 下界 > 0?）
7.  F5+F1+F4 增量 = ?
8.  J-lens 仲裁结论（F4 有增量 F1 没有 / F1≈F4 / 都无增量）？
9.  最强跨层特征是哪个（layer_disagreement / top_flip / final_vs_mid_kl）？
10. 校准：最优组合 ECE / risk-coverage？
11. 本轮 probe 是否仅线性标量 probe，零激活空间 controller / steering？（必须 yes）
12. gold_label 是否从未作为 inference feature？（必须 yes）
13. detector_beats_f5 = ?（true 仅当至少一个组合增量 CI 下界 > 0）
14. 结果解释：情况 A（有增量）/ B（无增量）→ 对 4D 与 H1 的影响？
15. 是否声称 hallucination reduction / accuracy improvement？（必须 no）
```

---

## 11. 结果解释逻辑

```text
情况 A：某组合增量 CI 下界 > 0
  → site-informed readout 在 finite-label cyber MCQ 上成立（差异化 B 实证）；
  → 可考虑 4D（门控干预）；但先提示：MCQ 是校准任务，H1 仍是主线载体
    （计划书 §14.4），4D 前应评估是否直接转 H1。
情况 B：无组合增量 CI 下界 > 0
  → 对 finite-label cyber MCQ 关闭内部特征检测假设；
  → 交付物退回 F5 selective-prediction 系统（诚实、有效）；
  → 主线动作明确转向 H1 编造标识符任务（计划书 §3 / §14.4），
    该任务下 F2/F3 有原料、输出校准已知差，是内部特征真正的用武之地。
两种情况都不进 4D-on-MCQ 之前，必须先在 review gate 给出「4C 后是走 4D 还是转 H1」
  的明确建议。
```

---

## 12. 推荐命令

```bash
conda run -n recover_attention python -m pytest tests/test_domain_label_proxy.py tests/test_cyber_data.py -q

conda run -n recover_attention python scripts/sprint_4C_narrowed_readout_increment_and_site_transfer.py \
  --f5-input-dir outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer \
  --processed-path data/processed/cyber/cybermetric.jsonl \
  --layers 20 24 \
  --mine-temperature 1.1 \
  --mine-samples-per-question 12 \
  --site-check-layers 16 20 24 \
  --site-check-alphas 0.25 1.0 \
  --site-check-max-pairs 34 \
  --cv-seeds 0 1 2 \
  --output-dir outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer \
  --overwrite

conda run -n recover_attention python -m pytest -q
```

预计时长：pair mining ~30-40 分钟；F1/F4 捕获（含 VJP）~1 小时；Stage 0.2 patching
~30-60 分钟。建议后台运行、打印进度。

---

## 13. 最多允许的结论

```text
On finite-label cyber MCQ, cross-layer readout projection (F1) and exact J-lens
(F4) do / do not provide incremental AUROC over the F5 single-forward bar (0.816)
under grouped CV with CIs; the 3C-1 causal site does / partially does / does not
transfer to the label-readout position; the J-lens-vs-direct arbitration
resolves 3C-4A as [approximation noise | readout method matters]. The mainline's
full hallucination hypothesis remains to be tested on the H1 fabricated-identifier
task, for which MCQ 4B/4C is the easy-calibrated-task reference point.
```

不得写：

```text
detector works（无条件）/ hallucination reduced / accuracy improved
ready for 2000 / probe controller works
```
