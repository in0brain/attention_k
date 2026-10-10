r"""Stage 1 跨模型可比性审计：Qwen vs Llama（只读，不训练任何模型）。

═══ 为什么不复用 Stage 0 那个脚本 ═══════════════════════════════════════════
`scripts/posthoc_cross_model_comparability_audit.py` **不能换个目录就用**：

    · 它导入 `Stage0Config` / `feature_cache.ARM_H1`，判读规则挂在冻结 v2.4 上
    · 它**只处理 H1**，没有 MCQ 的题级口径
    · 它里面的 "A / C" 指的是**分组方式**（identifier union-find），
      与本轮的 Path A / Path C（B4 是否移除）**不是一回事**

只借鉴它的审计思路（"先问两个模型分析的是不是同一批样本"），重写一份 Stage 1 的。

═══ 审计什么 ═══════════════════════════════════════════════════════════════
    1  计划名单        两侧各自的计划规模与实际终态
    2  可分析集合      交集 / 各自独有；**H1 按 trace 对齐，MCQ 按题对齐**
    3  排除原因        各侧被排除的构成
    4  分析单元        显式写明两侧的分析单元与对齐键，避免跨口径比较
    5  登记差异        生成预算、解析器版本 —— 两项都登记，不省略
    6  基础率          **只作描述性记录**，不附带难度 / 可比性 / 方差的推断

═══ 共同样本分析的口径（**本脚本只做第 ① 种**） ═════════════════════════════
    ① 在共同样本上**筛选已有 OOF** 评估   ← 本脚本；不重新训练任何模型
    ② 在共同样本上**重新训练**             ← **另一种分析**，本脚本不做

两者不是一回事：① 的模型是在各自全集上训练的，只是把评估限制到交集；
② 的模型本身就只见过交集。**不得**把 ① 说成 ②，也不得像 Stage 0 脚本那样
在跨模型比较里悄悄重训 Qwen 侧。

⚠️ 无论哪一种，**共同样本结果只能作敏感性分析**：交集本身由两侧的 eligibility
共同决定，而 eligibility 与模型表现相关，**选择偏差不会因为取交集而消除**。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QWEN = ROOT / "outputs/stage1/production"
LLAMA = ROOT / "outputs/stage1_llama_v3/production"
OUT = ROOT / "outputs/stage1_llama_v3/analysis/cross_model_comparability.json"

#: 两侧的分析单元与对齐键 —— 写死在这里，避免调用方各写各的
ALIGNMENT = {"h1": {"unit": "trace", "key": "trace_id"},
             "mcq": {"unit": "question", "key": "example_id"}}

COMMON_SAMPLE_CAVEAT = (
    "共同样本结果**只能作敏感性分析**。交集由两侧 eligibility 共同决定，"
    "而 eligibility 与模型表现相关 —— **取交集不消除选择偏差**，"
    "只是把偏差换了一种形状。且本脚本只做『在共同样本上筛选已有 OOF』，"
    "**不是**『在共同样本上重新训练』，两者是不同的分析。")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def side(name: str, base: Path, arm: str) -> dict:
    key = ALIGNMENT[arm]["key"]
    recs = _jsonl(base / f"{arm}_completion_records.jsonl")
    rep = json.loads((base / f"{arm}_finalization_report.json").read_text(encoding="utf-8"))
    elig = [r for r in recs if bool(r.get("eligible", r.get("eligible_for_primary")))]
    ineli = [r for r in recs if not bool(r.get("eligible", r.get("eligible_for_primary")))]
    with np.load(base / f"{arm}_hidden.npz") as z:
        hkeys = set(z.files)

    # 不可分析的原因：**用记录里的真字段**，不要自己造代理。
    # 第一版拿 `emitted_any_identifier` 当代理，把 H1 报成 "no_identifier 599 / other 1"；
    # 真字段 `primary_exclusion_reason` 显示实际以 **only_echoed** 为主（qwen 477）。
    field = "parse_status" if arm == "mcq" else "primary_exclusion_reason"
    why = collections.Counter(str(r.get(field) or "unknown") for r in ineli)
    if any(k == "unknown" for k in why):
        why["__warning__"] = f"有记录缺 {field}，已计入 unknown，**未用代理字段替代**"

    # 基础率（正例占比）作为**描述性事实**记录。
    # ⚠️ 前两版这里都写过推断，都不成立，**不要再写回去**：
    #   v1「基础率差一倍 ⇒ AUROC 可比性打折扣」—— 错：AUROC 是排序度量，
    #      类条件分数分布固定时，只改基础率**不改变**总体 AUROC
    #   v2「基础率不同 ⇒ 两侧任务不同难度」—— 同样是未经验证的推断：
    #      基础率**单独**不能确定检测难度，也不能确定估计方差 ——
    #      精度取决于正负类**数量**、分数分布与组内相关，不只是比例
    # 本版只记数值，解释放在 base_rate.statement 里，且只说可核验的部分。
    lab = "label" if arm == "h1" else "wrong_label"
    ys = [int(r[lab]) for r in elig if r.get(lab) is not None]
    n_pos = int(sum(ys))
    return {"model": name,
            "n_finalized_records": len(recs),
            "n_analyzable": len(elig),
            "label_field": lab,
            "n_positive": n_pos,
            "positive_rate": round(n_pos / max(len(ys), 1), 5),
            "n_with_label": len(ys),
            "n_not_analyzable": len(ineli),
            "not_analyzable_reasons": dict(why),
            "not_analyzable_reason_field": field,
            "n_hidden_vectors": len(hkeys),
            "hidden_alignment_key": rep.get("hidden_alignment_key"),
            "alignment_key_used": key,
            "n_raw_units": rep.get("n_raw_units"),
            "nominal_questions": rep.get("nominal_questions"),
            "population_role": rep.get("population_role"),
            # **两侧各自的偏离都要报** —— 第一版只登记了 Llama 的预算/解析器，
            # 漏了 Qwen MCQ 自身的 runtime_technical_attrition（9 条技术排除）
            "self_reported_deviation": {
                "protocol_deviation": rep.get("protocol_deviation"),
                "classification": rep.get("classification"),
                "technical_exclusions": rep.get("technical_exclusions"),
                "excluded_ids": sorted(rep.get("excluded_example_ids") or []),
                "nominal_questions": rep.get("nominal_questions"),
                "note": ("None 表示该侧 finalization 报告是旧 schema、没有这个字段，"
                         "**不等于**没有偏离"),
            },
            "_finalized_ids": {str(r[key]) for r in recs},
            "report_sha256": _sha(base / f"{arm}_finalization_report.json"),
            "_ids": {str(r[key]) for r in elig}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["h1", "mcq"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    t0 = time.perf_counter()
    reg = json.loads((ROOT / "outputs/stage1_llama_v3/budget512_registration.json")
                     .read_text(encoding="utf-8"))
    out: dict = {
        "schema": "stage1_cross_model_comparability_audit_v1",
        "note": ("只读审计，**不训练任何模型**。借鉴 Stage 0 可比性审计的思路，"
                 "但**未复用**那份脚本 —— 它绑 Stage0Config / v2.4 判读、只处理 H1，"
                 "且其 A/C 指分组方式，与本轮的 Path A/C 不是一回事。"),
        "alignment": ALIGNMENT,
        "registered_differences": {
            "generation_budget": {
                "qwen": "Stage 1 冻结配置",
                "llama_mcq": f"冻结 {reg.get('frozen_max_new_tokens')} → "
                             f"登记 {reg.get('registered_max_new_tokens')}"
                             f"（{reg.get('status')}）",
                "consequence": "两侧 MCQ 的生成上限不同，**任何跨模型陈述都必须写明**",
            },
            "parser_version": {
                "statement": "Llama 侧采用**修复后的解析器**，正式生产期间未再修改",
                "history": ["L3G MCQ 答案提取与定位的最小修复（2026-09-15）",
                            "L3H 收紧 MCQ 解析规则（2026-09-15，第二次修订）",
                            "L3I 解析修复后的特征重提取与 gate 证据重建（2026-09-15）"],
                "consequence": "两侧解析器版本不同 → eligibility 集合的构成本身就会不同",
            },
        },
        "common_sample_caveat": COMMON_SAMPLE_CAVEAT,
        "common_sample_method_used": "① 在共同样本上筛选已有 OOF（本审计不做，仅给出名单）",
        "common_sample_method_not_used": "② 在共同样本上重新训练（不同的分析，本轮不做）",
        "arms": {},
    }

    for arm in args.arms:
        q = side("qwen", QWEN, arm)
        l = side("llama", LLAMA, arm)
        qi, li = q.pop("_ids"), l.pop("_ids")
        qf, lf = q.pop("_finalized_ids"), l.pop("_finalized_ids")
        inter, qo, lo = qi & li, qi - li, li - qi

        # 差异归因：把"各自独有"拆成可指名的成因，而不是只报一个数
        q_tech = set(q["self_reported_deviation"]["excluded_ids"])
        l_tech = set(l["self_reported_deviation"]["excluded_ids"])
        attribution = {
            "llama_only_breakdown": {
                "qwen_technical_exclusion_but_llama_analyzable":
                    sorted(lo & q_tech),
                "qwen_not_analyzable_for_other_reason":
                    sorted(lo - q_tech),
            },
            "qwen_only_breakdown": {
                "llama_technical_exclusion_but_qwen_analyzable":
                    sorted(qo & l_tech),
                "llama_not_analyzable_for_other_reason":
                    sorted(qo - l_tech),
            },
            "finalized_roster": {
                "qwen": len(qf), "llama": len(lf),
                "identical": qf == lf,
                "intersection": len(qf & lf),
                "qwen_only": sorted(qf - lf)[:20], "llama_only": sorted(lf - qf)[:20],
                "note": ("**finalized 名单本身**是否相同 —— 这一层先于 eligibility。"
                         "不同则两侧连「同一批题」都不是。"),
            },
        }
        print(f"═══ {arm}（分析单元 = {ALIGNMENT[arm]['unit']}，"
              f"对齐键 = {ALIGNMENT[arm]['key']}）")
        for d in (q, l):
            print(f"  {d['model']:<6} finalized {d['n_finalized_records']:>5}  "
                  f"可分析 {d['n_analyzable']:>5}  hidden {d['n_hidden_vectors']:>5}  "
                  f"正例 {d['n_positive']:>4} ({d['positive_rate']:.2%})")
        rr = (l['positive_rate'] / q['positive_rate']) if q['positive_rate'] else float('nan')
        # 只报数，**不附带任何难度/可比性的推断**（见 base_rate.statement）
        print(f"  基础率（描述性）  qwen {q['positive_rate']:.2%}（{q['n_positive']}/"
              f"{q['n_with_label']}）  llama {l['positive_rate']:.2%}（{l['n_positive']}/"
              f"{l['n_with_label']}）  比值 {rr:.2f}×")
        fr = attribution["finalized_roster"]
        print(f"  finalized 名单  qwen {fr['qwen']}  llama {fr['llama']}  "
              f"完全相同={fr['identical']}  交集 {fr['intersection']}")
        for m, d in (("qwen", q), ("llama", l)):
            sd = d["self_reported_deviation"]
            print(f"      {m:<6} 自述 protocol_deviation={sd['protocol_deviation']}  "
                  f"classification={sd['classification']}  "
                  f"技术排除={sd['technical_exclusions']}")
        print(f"  可分析交集 {len(inter)}   qwen 独有 {len(qo)}   llama 独有 {len(lo)}")
        ab = attribution["llama_only_breakdown"]
        print(f"      llama 独有归因：qwen 技术排除 {len(ab['qwen_technical_exclusion_but_llama_analyzable'])} "
              f"＋ qwen 其他原因不可分析 {len(ab['qwen_not_analyzable_for_other_reason'])}")
        ab = attribution["qwen_only_breakdown"]
        print(f"      qwen  独有归因：llama 技术排除 {len(ab['llama_technical_exclusion_but_qwen_analyzable'])} "
              f"＋ llama 其他原因不可分析 {len(ab['llama_not_analyzable_for_other_reason'])}")
        if qi and li:
            print(f"  交集占 qwen {100*len(inter)/len(qi):.1f}% ／ "
                  f"占 llama {100*len(inter)/len(li):.1f}%")
        print(f"  qwen 不可分析原因  {q['not_analyzable_reasons']}")
        print(f"  llama 不可分析原因 {l['not_analyzable_reasons']}")
        ids_dir = args.out.parent / "common_sample_ids"
        ids_dir.mkdir(parents=True, exist_ok=True)
        (ids_dir / f"{arm}_intersection.txt").write_text(
            "\n".join(sorted(inter)) + "\n", encoding="utf-8")
        out["arms"][arm] = {
            "analysis_unit": ALIGNMENT[arm]["unit"],
            "alignment_key": ALIGNMENT[arm]["key"],
            "qwen": q, "llama": l,
            "intersection": {
                "n": len(inter),
                "pct_of_qwen": round(100.0 * len(inter) / max(len(qi), 1), 3),
                "pct_of_llama": round(100.0 * len(inter) / max(len(li), 1), 3),
                "ids_file": str((ids_dir / f"{arm}_intersection.txt").relative_to(ROOT)),
            },
            "difference_attribution": attribution,
            "qwen_only": {"n": len(qo), "examples": sorted(qo)[:5]},
            "llama_only": {"n": len(lo), "examples": sorted(lo)[:5]},
            "same_alignment_key": q["hidden_alignment_key"] == l["hidden_alignment_key"],
            "base_rate": {
                "qwen": q["positive_rate"], "llama": l["positive_rate"],
                "ratio_llama_over_qwen": (round(l["positive_rate"] / q["positive_rate"], 4)
                                          if q["positive_rate"] else None),
                "statement": (
                    "**这是一条描述性差异，不附带推断。** 两侧可分析集合的正例占比不同。"),
                "what_this_does_NOT_imply": [
                    "**不**意味着 AUROC 不可比：AUROC 是排序度量，"
                    "类条件分数分布固定时，只改变基础率**不会**改变总体 AUROC",
                    "**不**意味着某一侧任务更难：基础率单独不能确定检测难度",
                    "**不**意味着某一侧估计方差更大：精度取决于正负类**数量**、"
                    "分数分布与组内相关，不只是比例",
                    "**不**能自动归因到模型身份或架构：样本组成、训练、"
                    "生成预算与解析器版本两侧都不同，未做任何分离",
                ],
                "what_still_holds": (
                    "两侧的 AUROC / Δ_H 仍**不得直接并排当同类证据**，但理由不是基础率 —— "
                    "而是可分析集合不同（见 difference_attribution）、生成预算不同、"
                    "解析器版本不同。差值**不构成**跨模型效应量。"
                    "本轮不做基础率校正，也不做 pooled estimate（预注册中无此 estimand）。"),
                "correction_history": [
                    "v1：写成『基础率差一倍 ⇒ AUROC 可比性打折扣』—— 错误，已撤回",
                    "v2：写成『基础率不同 ⇒ 两侧任务不同难度』—— 未经验证的推断，已撤回",
                    "v3（本版）：只陈述描述性差异，推断一律不写",
                ],
            },
        }
        print()

    out["total_secs"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"[out] {args.out}")
    print(f"⚠️ {COMMON_SAMPLE_CAVEAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
