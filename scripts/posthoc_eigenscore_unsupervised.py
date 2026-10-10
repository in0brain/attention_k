"""POST-HOC：EigenScore（INSIDE, arXiv 2402.03744）—— 唯一的 unsupervised 范式。

**非 confirmatory。** 不改动、不覆盖 stage0_analysis_report.json。

────────────────────────────────────────────────────────────────────────
为什么做这个（以及为什么它和"再加一个分类器"不同）

前面的 readout×layer 网格已经证明：换分类器只改变幅度。但那 5 个读出器全部是
**supervised**（用标签训练）。EigenScore 是**不训练**的——它直接从同一 prompt 的 K 条
生成的 hidden 协方差算一个分数。若连不训练的方法也过不了表层基线，"内部状态 vs 表层"
的论证就跨过了 supervised/unsupervised 这条分界，经验边界比再加十个分类器都推得远。

预注册 §8 把它列为**可选项**（"仅复现 ≤2 天才加"），故此处按 post-hoc 补做。

────────────────────────────────────────────────────────────────────────
实现（严格按 INSIDE 原文 Eq. 5–6，不自行发挥）

    Σ = (1/K)·Zᵀ·J_d·Z        Z ∈ ℝ^{d×K}（K 条生成的 embedding 按列排）
    J_d = I_d − (1/d)·1_d·1_dᵀ  centering matrix，对**每条 embedding 在 d 维上**中心化
    EigenScore = (1/K)·log det(Σ + α·I_K) = (1/K)·Σᵢ log λᵢ
    α = 0.001（原文值）

**关键**：Σ 是 **K×K**，不是 d×d —— 它刻画 K 条生成**之间**的关系。所以 K=6 时是 6×6
矩阵，良态；不存在"6 个样本估 3584×3584 协方差"的奇异问题（这是读原文才确认的，
凭直觉会以为要正则化一个高维协方差）。
因 J_d 幂等且对称，Zᵀ J_d Z = (J_d Z)ᵀ(J_d Z)，故实现上先按列中心化再取 Gram 矩阵。

方向：EigenScore 越大 = K 条生成在嵌入空间越发散 = 越可能 hallucinate → 直接作 risk score。

────────────────────────────────────────────────────────────────────────
与原文的三处差异（如实声明，不假装完全复现）

  1. K：原文 10，我们 6（预注册 §3 冻结，不为复现某篇论文而改）。且因排除规则，
     部分 group 的 eligible 数 < 6。
  2. token 位置：原文取 **last token** embedding；我们用预注册 §8 冻结的 pooling
     （H1 = 全部 eligible identifier token 的 mean）。用我们自己的 H 定义更一致——
     否则比较的就不再是"我们的 H 换个读出方式"。
  3. Test-Time Feature Clipping（原文 §3.2，p=0.2 截断 + memory bank）：**未实现**。
     它是额外的 test-time 增强，不属于 EigenScore 本身的定义。

────────────────────────────────────────────────────────────────────────
粒度问题（本脚本最需要小心的地方）

EigenScore 是 **prompt 级**的量（一个 group 的 K 条生成 → 一个分数），而 H1 的标签是
**completion 级**的。处理：把 group 的 EigenScore 广播到该 group 的每条 completion，
再在 completion 级算 AUROC。后果是同 group 的样本分数完全相同（大量 ties）——这是该
方法的固有性质，不是实现缺陷，报告里如实标注。

────────────────────────────────────────────────────────────────────────
预期（写在跑之前，防止事后合理化）

预注册 §8 已写明这类一致性方法在 H1 上的局限：

    「Semantic Entropy：H1 上退化为 id_agreement_rate，**预期漏稳定 fabrication**；
      作 error-mode 采样 baseline，**不当独立 probe**」

模型自信地反复编造**同一个**不存在的 ID 时，K 条生成高度一致 → EigenScore 低 → 判"没问题"。
故 **EigenScore 在 H1 上预期表现差，且这个预期是预注册时就写下的**。若结果确实差，
那是预注册预测得到验证，**不是新发现，也不是实现不好**。

────────────────────────────────────────────────────────────────────────
核心对照（比"能不能过基线"更有信息量）

    EigenScore（hidden 侧一致性，unsupervised）
        vs
    f5_id_agreement_rate（输出侧一致性，已在 F5 里、已是 O 的一部分）

两者测同一件事——生成的自洽程度——一个看内部表示，一个看输出字符串。若两者接近，
即说明**内部状态的一致性信息并不超过输出侧已有的一致性信息**。这是"内部 vs 表层"
最干净的一个对照，且天然落在"相对表层基线还剩什么"的框架内。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402

FEAT = Path("outputs/stage0/features")
LAYERS_DIR = Path("outputs/stage0/layers")
OUT = Path("outputs/stage0/evaluation")

ALPHA = 0.001          # INSIDE 原文值
LAYERS = ["rel0.5", "rel0.7", "final"]   # rel0.5 == int(L/2)，与原文所用层一致


def eigenscore(Z: np.ndarray, alpha: float = ALPHA) -> float:
    """INSIDE Eq. 5–6。Z: (d, K) —— 每列是一条生成的 embedding。"""
    if Z.ndim != 2 or Z.shape[1] < 2:
        return float("nan")          # K<2 无法刻画"生成之间"的关系
    Zc = Z - Z.mean(axis=0, keepdims=True)        # J_d @ Z：每条 embedding 在 d 维上中心化
    K = Z.shape[1]
    sigma = (Zc.T @ Zc) / K                        # (K, K)
    eig = np.linalg.eigvalsh(sigma + alpha * np.eye(K))
    eig = np.clip(eig, 1e-12, None)                # 数值下限；α 已保证满秩
    return float(np.mean(np.log(eig)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="h1", choices=["h1"],
                    help="仅 H1：MCQ 每题只有一条 greedy 进 population，无 K 条可比")
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            open(FEAT / "h1_completion_records.jsonl", encoding="utf-8") if l.strip()]
    elig = [r for r in recs if r.get("eligible") and r.get("id_token_positions_local")]
    y = np.array([int(r["label"]) for r in elig])
    gid = [str(r["group_id"]) for r in elig]
    tid = [str(r["trace_id"]) for r in elig]
    print(f"[eigen] n={len(elig)} pos={int(y.sum())} groups={len(set(gid))}")

    by_group = defaultdict(list)
    for i, g in enumerate(gid):
        by_group[g].append(i)
    sizes = np.array([len(v) for v in by_group.values()])
    print(f"[eigen] eligible per group: min={sizes.min()} median={int(np.median(sizes))} "
          f"max={sizes.max()} (K=6 nominal; 排除规则会让部分 group 少于 6)")

    report = {
        "schema_version": "posthoc_eigenscore_v1",
        "status": ("POST-HOC — not confirmatory. Unsupervised (no classifier trained). "
                   "Does not modify or supersede stage0_analysis_report.json."),
        "timestamp": args.timestamp,
        "method": "EigenScore (INSIDE, arXiv 2402.03744), Eq. 5-6",
        "alpha": ALPHA,
        "deviations_from_original": [
            "K=6 (preregistration-frozen) vs 10 in the paper; some groups have fewer after exclusions",
            "token position: our frozen section-8 pooling (mean over eligible identifier tokens) "
            "vs last-token embedding in the paper",
            "test-time feature clipping (paper section 3.2) not implemented",
        ],
        "granularity_note": ("EigenScore is a per-prompt quantity broadcast to that prompt's "
                             "completions, so completions in one group share a score (ties). "
                             "Inherent to the method, not an implementation artifact."),
        "preregistered_expectation": ("section 8 states consistency-based methods are expected to "
                                      "MISS stable fabrication on H1 (the model confidently repeats "
                                      "the same non-existent ID -> K generations agree -> low "
                                      "EigenScore -> judged fine). Poor performance here CONFIRMS a "
                                      "preregistered prediction; it is not a new finding."),
        "n": len(elig), "n_positive": int(y.sum()), "n_groups": len(by_group),
        "group_size": {"min": int(sizes.min()), "median": int(np.median(sizes)),
                       "max": int(sizes.max())},
        "layers": {},
    }

    for lname in LAYERS:
        z = np.load(LAYERS_DIR / f"h1_hidden_{lname}.npz")
        missing = [t for t in tid if t not in z.files]
        if missing:
            print(f"[eigen] {lname}: {len(missing)} ids missing → 跳过")
            report["layers"][lname] = {"usable": False, "reason": "ids missing"}
            continue
        M = np.vstack([z[t] for t in tid])
        if not np.isfinite(M).all():
            n_bad = int((~np.isfinite(M)).any(axis=1).sum())
            print(f"[eigen] {lname}: UNAVAILABLE — {n_bad} 条非有限（同 readout 网格的结论）")
            report["layers"][lname] = {"usable": False, "n_nonfinite_records": n_bad,
                                       "reason": "non-finite hidden values"}
            continue

        scores = np.full(len(elig), np.nan)
        n_small = 0
        for g, idxs in by_group.items():
            s = eigenscore(M[idxs].T)               # (d, K_g)
            if not np.isfinite(s):
                n_small += 1
            for i in idxs:
                scores[i] = s
        ok = np.isfinite(scores)
        auc = float(ci.auroc(scores[ok], y[ok])) if ok.sum() and len(set(y[ok])) > 1 else float("nan")
        report["layers"][lname] = {
            "usable": True, "auroc": auc, "n_scored": int(ok.sum()),
            "n_groups_too_small": n_small,
        }
        print(f"[eigen] {lname:7s} AUROC={auc:.4f}  (n={int(ok.sum())}, "
              f"groups with K<2: {n_small})")

    # ── 核心对照：输出侧一致性（已在 F5 里，是 O 的一部分）──
    agree = np.array([r.get("f5_id_agreement_rate") for r in elig], dtype=object)
    m = np.array([a is not None for a in agree])
    if m.any():
        a = np.array([float(x) for x in agree[m]])
        # 方向：一致率高 = 不太可能 fabricate → 取负作 risk score
        auc_out = float(ci.auroc(-a, y[m]))
        report["output_side_consistency_reference"] = {
            "feature": "f5_id_agreement_rate (negated: high agreement = low risk)",
            "auroc": auc_out, "n": int(m.sum()),
            "note": ("this feature is already inside F5 and therefore already inside O; "
                     "the comparison asks whether hidden-side consistency adds anything "
                     "over output-side consistency that O already has"),
        }
        print(f"[eigen] output-side f5_id_agreement_rate AUROC={auc_out:.4f}  ← 对照")

    report["reference_auroc_O_surface_baseline"] = 0.8585
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "posthoc_eigenscore_h1.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                 encoding="utf-8")
    print(f"[eigen] report → {p}")


if __name__ == "__main__":
    main()
