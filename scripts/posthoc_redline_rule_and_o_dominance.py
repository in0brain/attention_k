"""POST-HOC：artifact 红线的规则修正 × 分组 四宫格判定 + O 支配性检查。

**这不是预注册分析。** 不改动、不覆盖、不重跑 confirmatory 结果；
stage0_analysis_report.json 里的判定按冻结规则原样成立，本脚本只是并列给出
"若规则改成 X 会怎样"，供论文讨论与后续研究使用。

────────────────────────────────────────────────────────────────────────
一、红线现规则的缺陷（推理，不是口味问题）

冻结原文：Δ_art = AUROC(full-text) − AUROC(shortcut)，
          触发 ⟺ CI95(Δ_art) ⊂ [−ε, +ε]

把包含关系拆开，它是两个条件的合取：
    (a) CI_hi ≤ +ε   full-text 并不显著优于 shortcut   ← 红线的推理需要这条
    (b) CI_lo ≥ −ε   shortcut 并不显著优于 full-text   ← 无推理依据

红线守的是 Δ_H 的可解释性：若平凡 shortcut 就能达到 full-text 的判别力，则标签
由表面形式决定，而 H 恰好读在同一批 token 上、天然编码其表面身份，于是 Δ_H 分不清
"内部知道自己在编造"与"hidden 把表面形式编码得比 TF-IDF 更高效"。

按这条推理，shortcut **强于** full-text 是 artifact 更严重的证据（a fortiori），
可 (b) 恰恰在这时把红线关掉 —— 问题越严重、规则越沉默，单调性反了。
根因是把双侧等价检验（TOST，适用于两个方向都构成危害的场景，如生物等效性）
套用在了一个单侧的推断问题上。

修正规则（non-superiority）：触发 ⟺ CI_hi < +ε
    打平        → 触发（与原规则一致）
    shortcut 远胜 → 触发（补上盲区）
    full-text 真强 → 不触发
    CI 很宽说不清 → 不触发（保守）
原规则的触发域是修正规则的子集 → 这是**净收紧**，与 v2.2 / v2.4 的修订性质一致。

────────────────────────────────────────────────────────────────────────
二、更深的一层：红线比错了对象

红线比的是 rung4(full-text) vs shortcut，但需要保护的量是
Δ_H = AUROC(O+H) − AUROC(O)，里面没有 rung4。两种失效模式：

  FM1 任务被表面模式解决        ← 红线想抓的
  FM2 O 相对输出侧上限太弱      ← 红线完全没管

FM2 的锐利诊断只有一句：**O 是不是输出侧最强的那一阶？** 若 O 被某个 output-only
rung 支配，则 Δ_H 里有一部分功劳属于"输出侧本来就有、只是 O 的特征化没抓住"的信息。

必须区分（否则会与 §4 的禁令冲突）：
  定义 O = max over rungs  → 按测试表现选基线 → winner's curse，有偏。§4 明确禁止。
  检查 O 是否被支配        → 不改 O、不改 Δ_H，只判断结论能否表述为
                             "相对最强输出侧基线的增量"。无选择偏倚。
本脚本做的是后者。

────────────────────────────────────────────────────────────────────────
三、分组维度

A  prompt-grouped              confirmatory 采用的设计
C  prompt ∪ fabricated-id      两条泄漏通道同时封死（见
                               posthoc_h1_identifier_leakage_diagnostic.py）
C 仅对 H1 有定义：MCQ 的标签是"解析选项 ≠ gold"，没有 identifier 空间可背，
其 group 已是每题一条，故 MCQ 只报 A。四宫格因此是 H1 的 2×2 + MCQ 的 2×1，
**不强行凑成对称**。
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import ladder as ld  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import (  # noqa: E402
    ARM_H1,
    ARM_MCQ,
    load_arm,
)

_spec = importlib.util.spec_from_file_location(
    "leak", PROJECT_ROOT / "scripts" / "posthoc_h1_identifier_leakage_diagnostic.py")
leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak)

OUTPUT_ONLY_RUNGS = ("rung1", "rung2", "rung3", "rung4", "rung5", "rung6")


def rule_frozen(ci_lo: float, ci_hi: float, eps: float) -> bool:
    """冻结规则：双侧等价。与 ci.artifact_redline 同义，此处显式写出以便并列对照。"""
    return (ci_lo >= -eps) and (ci_hi <= eps)


def rule_corrected(ci_lo: float, ci_hi: float, eps: float) -> bool:
    """修正规则：non-superiority —— full-text 并不显著优于 shortcut。"""
    return ci_hi < eps


def redline_both_rules(cache, ladder, config) -> dict:
    """对一臂算出两套规则的判定（同一批 Δ_art 与 CI，只是判据不同）。"""
    scores = ladder["scores"]
    full_rung = cache.spec["full_text_rung"]
    s_full = scores[full_rung]
    eps = config.eps_auroc

    per_shortcut, fired_frozen, fired_corrected = {}, [], []
    for sc in cache.spec["shortcut_rungs"]:
        d = ci.paired_grouped_bootstrap_delta(cache.y, scores[sc], s_full, cache.groups,
                                              n_boot=config.n_boot, seed=config.seed)
        f_frozen = rule_frozen(d["ci_lo"], d["ci_hi"], eps)
        f_corr = rule_corrected(d["ci_lo"], d["ci_hi"], eps)
        per_shortcut[sc] = {
            "auroc_shortcut": ci.auroc(scores[sc], cache.y),
            "delta_artifact": {"point": d["point"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]},
            "frozen_rule_triggered": bool(f_frozen),
            "corrected_rule_triggered": bool(f_corr),
            "differs": bool(f_frozen != f_corr),
        }
        if f_frozen:
            fired_frozen.append(sc)
        if f_corr:
            fired_corrected.append(sc)

    return {
        "auroc_full_text": ci.auroc(s_full, cache.y),
        "full_text_rung": full_rung,
        "eps": eps,
        "per_shortcut": per_shortcut,
        "frozen_rule": {"triggered": bool(fired_frozen), "triggered_by": fired_frozen},
        "corrected_rule": {"triggered": bool(fired_corrected), "triggered_by": fired_corrected},
    }


def o_dominance(cache, ladder, config) -> dict:
    """O 是否被任一 output-only rung 支配？带 paired grouped-bootstrap CI。

    不改 O、不改 Δ_H —— 只回答"能不能把 Δ_H 称作'相对最强输出侧基线的增量'"。
    """
    scores = ladder["scores"]
    o_rung = cache.spec["o_rung"]
    s_o = scores[o_rung]
    a_o = ci.auroc(s_o, cache.y)

    rivals = {}
    dominated_by = []
    for name in scores:
        if name in ("H_alone", "O_plus_H") or name == o_rung:
            continue
        if not name.startswith(OUTPUT_ONLY_RUNGS):
            continue
        # Δ = AUROC(O) − AUROC(rival)：传 (rival, O) 得到 O − rival
        d = ci.paired_grouped_bootstrap_delta(cache.y, scores[name], s_o, cache.groups,
                                              n_boot=config.n_boot, seed=config.seed)
        # rival 显著优于 O ⟺ (O − rival) 的 CI 上界仍 < 0
        dominates = d["ci_hi"] < 0
        rivals[name] = {
            "auroc_rival": ci.auroc(scores[name], cache.y),
            "delta_O_minus_rival": {"point": d["point"], "ci_lo": d["ci_lo"],
                                    "ci_hi": d["ci_hi"]},
            "rival_significantly_beats_O": bool(dominates),
        }
        if dominates:
            dominated_by.append(name)

    best = max(rivals, key=lambda k: rivals[k]["auroc_rival"]) if rivals else None
    return {
        "o_rung": o_rung,
        "auroc_O": a_o,
        "strongest_output_only_rival": best,
        "auroc_strongest_rival": rivals[best]["auroc_rival"] if best else None,
        "gap_O_minus_strongest": (a_o - rivals[best]["auroc_rival"]) if best else None,
        "rivals": rivals,
        "O_is_dominated": bool(dominated_by),
        "dominated_by": dominated_by,
        "consequence": (
            "O 被输出侧某阶显著超过 → Δ_H 不得表述为'相对最强输出侧基线的增量'；"
            "应表述为'相对预注册基线 O 的增量'，并同报被谁超过、超过多少"
            if dominated_by else
            "O 未被任何 output-only rung 显著超过 → '相对最强输出侧基线'的表述成立"),
        "note": ("这是**检查**不是**选择**：不改 O 的定义，故不引入 §4 所禁的 "
                 "winner's curse；改变的只是结论的可表述范围。"),
    }


def analyse(cache, config, label: str) -> dict:
    lad = ld.run_ladder(cache, config)
    return {
        "grouping": label,
        "n": int(cache.y.size),
        "n_positive": int((cache.y == 1).sum()),
        "n_groups": int(len(np.unique(cache.groups))),
        "rung_auroc": {k: v["auroc"] for k, v in lad["rungs"].items()},
        "redline": redline_both_rules(cache, lad, config),
        "o_dominance": o_dominance(cache, lad, config),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/stage0/features")
    ap.add_argument("--output", default="outputs/stage0/evaluation/"
                                        "posthoc_redline_rule_and_o_dominance.json")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--h1-only", action="store_true",
                    help="只审 H1（§9 跨模型复核只跑 H1 臂，input-dir 下没有 MCQ cache）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    config = Stage0Config(n_outer_folds=cfg.n_outer_folds, n_inner_folds=cfg.n_inner_folds,
                          n_boot=cfg.n_boot, seed=cfg.seed, mode=cfg.mode,
                          allow_pilot_population=False)
    in_dir = Path(args.input_dir)

    h1 = load_arm(ARM_H1, in_dir / "h1_completion_records.jsonl", in_dir / "h1_hidden.npz",
                  config, label_field="label", id_field="trace_id")

    cells = {}
    print("[posthoc] H1 / grouping A (prompt, confirmatory design) ...")
    cells["h1_A_prompt"] = analyse(h1, config, "A_prompt")

    groups_both, comp_stats = leak.component_groups(h1.records, link_prompt=True,
                                                    link_fabricated=True)
    print(f"[posthoc] H1 / grouping C (prompt u fabricated-id, "
          f"{comp_stats['n_components']} components) ...")
    cells["h1_C_prompt_union_fabid"] = analyse(replace(h1, groups=groups_both), config,
                                               "C_prompt_union_fabricated")
    cells["h1_C_prompt_union_fabid"]["component_stats"] = comp_stats

    if not args.h1_only:
        mcq = load_arm(ARM_MCQ, in_dir / "mcq_completion_records.jsonl",
                       in_dir / "mcq_hidden.npz", config, label_field="wrong_label")
        print("[posthoc] MCQ / grouping A (question) ...")
        cells["mcq_A_question"] = analyse(mcq, config, "A_question")
        cells["mcq_A_question"]["grouping_C_note"] = (
            "not applicable: MCQ's label is parsed-answer != gold; there is no identifier space "
            "to memorise, and its groups are already one record per question")
    else:
        print("[posthoc] --h1-only: MCQ arm skipped (cross-model replication audits H1 only)")

    report = {
        "schema_version": "posthoc_redline_rule_and_o_dominance_v1",
        "status": ("POST-HOC — not preregistered; does not modify or supersede "
                  "stage0_analysis_report.json. The frozen rule's verdict stands as the "
                  "confirmatory result; the corrected rule is reported alongside it."),
        "timestamp": args.timestamp,
        "frozen_rule": "trigger iff CI(delta_artifact) subset of [-eps, +eps] (two-sided equivalence)",
        "corrected_rule": "trigger iff CI_upper(delta_artifact) < +eps (non-superiority)",
        "why_corrected": ("the frozen rule conjoins CI_hi<=eps (justified) with CI_lo>=-eps "
                          "(unjustified); a shortcut that substantially BEATS full-text is "
                          "stronger evidence of a surface-pattern artifact, yet switches the "
                          "rule off. Trigger region of the frozen rule is a subset of the "
                          "corrected one, so this is a net tightening."),
        "cells": cells,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    # ---------------------------------------------------------------- 打印
    print("\n=== 四宫格：红线判定（frozen / corrected） ===")
    print(f"  {'cell':26s} {'shortcut':28s} {'d_art':>8s} {'CI_hi':>8s}  frozen  corrected")
    for cell, res in cells.items():
        for sc, d in res["redline"]["per_shortcut"].items():
            print(f"  {cell:26s} {sc:28s} {d['delta_artifact']['point']:+8.4f} "
                  f"{d['delta_artifact']['ci_hi']:+8.4f}  "
                  f"{str(d['frozen_rule_triggered']):6s}  {d['corrected_rule_triggered']}")
    print("\n  臂级别（对 shortcut 取或）：")
    for cell, res in cells.items():
        r = res["redline"]
        print(f"    {cell:26s} frozen={r['frozen_rule']['triggered']}  "
              f"corrected={r['corrected_rule']['triggered']} "
              f"{r['corrected_rule']['triggered_by']}")

    print("\n=== O 支配性检查 ===")
    for cell, res in cells.items():
        o = res["o_dominance"]
        print(f"  {cell:26s} AUROC(O)={o['auroc_O']:.4f}  "
              f"strongest rival={o['strongest_output_only_rival']} "
              f"({o['auroc_strongest_rival']:.4f})  gap={o['gap_O_minus_strongest']:+.4f}  "
              f"O_dominated={o['O_is_dominated']} {o['dominated_by']}")
    print(f"\n[posthoc] report -> {out}")


if __name__ == "__main__":
    main()
