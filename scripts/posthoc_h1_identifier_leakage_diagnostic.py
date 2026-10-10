"""POST-HOC 诊断：H1 的 identifier 记忆泄漏通道有多大？

**这不是预注册分析，不改动、不覆盖、不重跑 confirmatory 结果。**
它只回答一个在揭盲后才浮现、且审稿人必然会问的解释性问题。

背景（confirmatory 结果，2026-07-22 揭盲，已冻结）:
    H1 阶梯里最强的单一预测器不是 O，也不是 H，而是 rung2 id-string-only = 0.919，
    高于 O(0.858)、H alone(0.897)、O+H(0.900)。预注册的 artifact 红线是**双侧等价
    检验**（|Δ_artifact| ⊂ [-0.02,0.02] 才触发），故只能检出 "shortcut ≈ full-text"，
    检不出 "shortcut 显著优于 full-text" —— 后者本该是更强的警报。

机制假设:
    H1 的标签是"completion 内是否有 identifier 不在本体中"，即**标签是 identifier
    字符串的确定性函数**。confirmatory CV 按 prompt 分组，而实测 172 个 fabricated
    id 里有 48 个（27.9%）跨多个 prompt-group 复现，训练折见过的 fabricated id 会在
    测试折原样出现 —— 分类器可以靠背"哪些 id 不在本体里"取胜，这是本体查表记忆，
    不是"模型内部知道自己在编造"。
    §8 的 H 又恰好是在 identifier token 上做 mean pooling，hidden state 天然编码所在
    token 的身份，同一记忆通道对 H 同样开放。

诊断设计（只改分组，其余一切不变）:
    把 CV 分组从 prompt 换成 **fabricated-identifier 连通分量**:两条 completion 若
    共享任一 fabricated identifier 则并入同一组，于是任何 fabricated id 不会同时出现
    在训练与测试折。实测 2016 条 → 1881 个分量、最大分量仅 14 条（0.7%），可用。
    folds 数 / nested C 网格 / 融合协议 / seed / 特征构造全部沿用 Stage0Config，
    只有 ArmCache.groups 被替换 —— 这样两次的差值可归因于分组，而非任何其他改动。

判读（三选一，写在跑之前）:
    A rung2 大幅掉、H_alone 也大幅掉 → H1 正结果主要由 id 记忆驱动。
    B rung2 大幅掉、H_alone 保留较多  → 支持 hidden 含超出字符串记忆的信息。
    C O+H 仍高于 O 且高于 rung2       → 论文结论比现在更强。

已知局限（必须与结果同报）:
    grounded identifier 仍跨折 —— 按 grounded 也连边会让 2016 条塌成 980+867 两大块，
    CV 无法进行。故本诊断只封堵**正类**记忆通道，"记住 T1204 是真的"这条弱通道仍在。
"""

import argparse
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention.evaluation import increment as inc  # noqa: E402
from recover_attention.evaluation import ladder as ld  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_H1, load_arm  # noqa: E402


def component_groups(records: list[dict], *, link_prompt: bool,
                     link_fabricated: bool) -> tuple[np.ndarray, dict]:
    """把 completion 按指定的共享关系并成连通分量，返回分组标签。

    union-find:completion 与它的 prompt / 它含有的每个 fabricated id 相连，于是被连起
    来的 completion 必然同折 → 对应的泄漏通道被封死。

    **两条通道必须同时封**:
      link_fabricated 单独用会打破 prompt 分组（同一 prompt 的 K=6 条被拆到不同折），
      而 §3 规定 K 条 completion 共享 group 正是因为它们不独立。实测该设计下
      rung1_prompt_only 从 0.683 涨到 0.739 —— 直接证明 prompt 泄漏被放开了。
      只有 prompt ∪ fabricated-id 才是干净对照。
    """
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(records)):
        parent[("c", i)] = ("c", i)
    for i, r in enumerate(records):
        if link_prompt:
            key = ("p", r["group_id"])
            if key not in parent:
                parent[key] = key
            union(("c", i), key)
        if link_fabricated:
            for m in r.get("mentions", []):
                if m.get("normalized") and m.get("label") == "fabricated":
                    key = ("id", m["normalized"])
                    if key not in parent:
                        parent[key] = key
                    union(("c", i), key)

    root_to_gid: dict = {}
    groups = []
    for i in range(len(records)):
        root = find(("c", i))
        if root not in root_to_gid:
            root_to_gid[root] = f"fabcomp_{len(root_to_gid):05d}"
        groups.append(root_to_gid[root])

    sizes = collections.Counter(groups)
    stats = {
        "n_components": len(root_to_gid),
        "largest_component": max(sizes.values()),
        "n_singletons": sum(1 for v in sizes.values() if v == 1),
        "largest_component_share": round(max(sizes.values()) / len(records), 4),
    }
    return np.array(groups), stats


def identifier_recurrence_stats(records: list[dict]) -> dict:
    """量化记忆通道:fabricated id 在多少个 prompt-group 里复现。"""
    fab_groups: dict = collections.defaultdict(set)
    for r in records:
        for m in r.get("mentions", []):
            if m.get("normalized") and m.get("label") == "fabricated":
                fab_groups[m["normalized"]].add(r["group_id"])
    recurring = {i: len(g) for i, g in fab_groups.items() if len(g) > 1}
    return {
        "n_distinct_fabricated_ids": len(fab_groups),
        "n_recurring_across_prompt_groups": len(recurring),
        "share_recurring": round(len(recurring) / max(1, len(fab_groups)), 4),
        "top_recurring": sorted(recurring.items(), key=lambda kv: -kv[1])[:10],
    }


def summarize(ladder: dict, increment: dict) -> dict:
    return {
        "n": ladder["n"],
        "n_positive": ladder["n_positive"],
        "n_groups": ladder["n_groups"],
        "rungs": {k: v["auroc"] for k, v in ladder["rungs"].items()},
        "auroc": increment["auroc"],
        "delta_H": {k: increment["delta_H"][k] for k in ("point", "ci_lo", "ci_hi")},
        "read": increment["read"],
        "precision": increment["precision"]["precision"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/stage0/features")
    ap.add_argument("--output", default="outputs/stage0/evaluation/"
                                        "posthoc_h1_identifier_leakage_diagnostic.json")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--timestamp", required=True, help="ISO8601，由调用方传入")
    args = ap.parse_args()

    config = load_config(args.config)
    # 生产规格,与 confirmatory 完全一致（只有分组会变）
    config = Stage0Config(n_outer_folds=config.n_outer_folds,
                          n_inner_folds=config.n_inner_folds,
                          n_boot=config.n_boot, seed=config.seed,
                          mode=config.mode, allow_pilot_population=False)

    in_dir = Path(args.input_dir)
    cache = load_arm(ARM_H1, in_dir / "h1_completion_records.jsonl",
                     in_dir / "h1_hidden.npz", config, label_field="label",
                     id_field="trace_id")
    print(f"[posthoc] H1 loaded: n={cache.y.size} pos={int(cache.y.sum())} "
          f"prompt_groups={len(set(cache.groups))}")

    recur = identifier_recurrence_stats(cache.records)
    print(f"[posthoc] fabricated ids: {recur['n_distinct_fabricated_ids']} distinct, "
          f"{recur['n_recurring_across_prompt_groups']} recur across prompt groups "
          f"({recur['share_recurring']:.1%})")

    # ---- A. 复算 prompt-grouped（应与 confirmatory 一致 → 作为自检）----
    print("[posthoc] recomputing prompt-grouped ladder (self-check vs confirmatory) ...")
    lad_prompt = ld.run_ladder(cache, config)
    inc_prompt = inc.compute_increment(cache, lad_prompt, config)

    # ---- B. fabricated-id only（诊断性,**有已知缺陷**:打破 prompt 分组）----
    groups_fab, stats_fab = component_groups(cache.records, link_prompt=False,
                                             link_fabricated=True)
    print(f"[posthoc] [B] fabricated-only components: {stats_fab['n_components']} "
          f"(largest {stats_fab['largest_component_share']:.1%}) "
          f"— breaks prompt grouping, kept only to show the prompt-leak it introduces")
    lad_fab = ld.run_ladder(replace(cache, groups=groups_fab), config)
    inc_fab = inc.compute_increment(replace(cache, groups=groups_fab), lad_fab, config)

    # ---- C. prompt ∪ fabricated-id（**主诊断**:两条泄漏通道同时封死）----
    groups_both, stats_both = component_groups(cache.records, link_prompt=True,
                                               link_fabricated=True)
    print(f"[posthoc] [C] prompt-union-fabricated components: {stats_both['n_components']} "
          f"(largest {stats_both['largest_component_share']:.1%}) — PRIMARY diagnostic")
    cache_both = replace(cache, groups=groups_both)
    lad_both = ld.run_ladder(cache_both, config)
    inc_both = inc.compute_increment(cache_both, lad_both, config)

    prompt_s = summarize(lad_prompt, inc_prompt)
    fab_s = summarize(lad_fab, inc_fab)
    both_s = summarize(lad_both, inc_both)
    drops = {k: round(prompt_s["rungs"][k] - both_s["rungs"][k], 4) for k in prompt_s["rungs"]}

    report = {
        "schema_version": "posthoc_h1_identifier_leakage_v1",
        "status": "POST-HOC — not part of the preregistered confirmatory analysis; "
                  "does not modify or supersede stage0_analysis_report.json",
        "timestamp": args.timestamp,
        "question": ("Does H1's positive conditional increment survive when the "
                    "fabricated-identifier memorisation channel is blocked?"),
        "design": ("identical folds/nested-C/fusion/seed; ONLY ArmCache.groups changes, "
                  "from prompt to connected components over shared fabricated identifiers"),
        "identifier_recurrence": recur,
        "A_prompt_grouped_confirmatory_design": prompt_s,
        "B_fabricated_only_grouped": {
            "component_stats": stats_fab,
            "caveat": ("breaks prompt grouping (K=6 completions of one prompt can split "
                      "across folds, violating section 3); reported only because the induced "
                      "prompt leak is directly visible in rung1_prompt_only"),
            **fab_s,
        },
        "C_prompt_union_fabricated_grouped_PRIMARY": {
            "component_stats": stats_both,
            "rationale": "both leakage channels closed simultaneously",
            **both_s,
        },
        "auroc_drop_by_rung_A_to_C": drops,
        "known_limitation": ("grounded identifiers still cross folds; connecting on them too "
                            "collapses 2016 completions into 980+867 components and makes CV "
                            "impossible. Only the positive-class memorisation channel is closed."),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    print("\n=== rung AUROC ===")
    print(f"  {'rung':34s} {'A prompt':>9s} {'B fab-only':>11s} {'C both':>9s} {'A-C':>8s}")
    for k in prompt_s["rungs"]:
        print(f"  {k:34s} {prompt_s['rungs'][k]:9.4f} {fab_s['rungs'][k]:11.4f} "
              f"{both_s['rungs'][k]:9.4f} {drops[k]:+8.4f}")
    print("\n=== Delta_H ===")
    for name, s in (("A prompt-grouped (confirmatory)", prompt_s),
                    ("B fabricated-only  (flawed)   ", fab_s),
                    ("C prompt+fabricated (PRIMARY) ", both_s)):
        d = s["delta_H"]
        print(f"  {name}: {d['point']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] "
              f"read={s['read']:12s} precision={s['precision']} n_groups={s['n_groups']}")
    print(f"\n[posthoc] report -> {out}")


if __name__ == "__main__":
    main()
