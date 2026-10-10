"""E2 / E3 分组构造（**仅 H1**；MCQ 一律 `not_applicable`）。

E2（entity-generalization，§2 / M2b）
    completion node = trace_id
    prompt node     = group_id
    identity node   = completion 中**每一个** emitted identifier 经冻结
                      `normalize_identifier` 规范化后的值
    每个 completion 同时连接它的 prompt 与它包含的每一个 identity。
    用**全部** identifier —— 不只第一个、不只 fabricated、**不使用 label**。
    按 M2b 门槛判连通图是否塌缩；塌缩 → `estimability = non_estimable`。

    ⚠️ 不可估时**仍输出结构性 accounting**（component 数、最大分量占比、hub 分布），
    禁止的是 CV / AUROC / Δ_H 等**性能估计**，不是禁止所有数值。

E3（vocabulary-block，§2）
    按 greedy completion 的**第一个** identifier，给该 prompt 分配冻结词表块
    （`cfg.e3_block_salt` / `cfg.e3_n_blocks`），label-free、prompt 整组归折；
    只保留该块内的 mention 构造标签与 H。

MCQ 上 E2/E3 **不适用**：不得把 A–D 字母或选项文本重新解释成 ontology identifier。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

from recover_attention.evaluation.v3_config import V3AnalysisConfig
import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.h1_identifier import normalize_identifier

NOT_APPLICABLE = "not_applicable"
ESTIMABLE = "estimable"
NON_ESTIMABLE = "non_estimable"
NEEDS_DECISION = "needs_researcher_decision"
NOT_EVALUATED = "not_evaluated"
#: E3 的**结构**状态（只说 fold 是否为空），与统计可估性严格区分。
NON_EMPTY = "non_empty"
EMPTY = "empty"
#: M2b 门槛（**收紧**）：最大连通分量占比 **严格大于** 该值即视为图塌缩。
#: 0.20 恰好等于门槛时**不**塌缩。
M2B_MAX_COMPONENT_SHARE = 0.20
#: 分量数下限 = 5 × n_outer_folds（默认 5 × 5 = 25）。少于它谈不上 identity-disjoint CV。
M2B_MIN_COMPONENTS_FACTOR = 5


def not_applicable(arm: str, estimand: str, reason: str | None = None) -> dict:
    return {"arm": arm, "estimand": estimand, "estimability": NOT_APPLICABLE,
            "reason": reason or (
                "E2/E3 仅适用于 H1；MCQ 的 A–D 字母与选项文本**不是** ontology "
                "identifier，不得重新解释"),
            "performance_estimates_computed": False}


def _identities(record: dict) -> list[str]:
    """该 completion 的全部 emitted identifier identity（规范化后去重，保持出现序）。"""
    out, seen = [], set()
    for m in record.get("mentions") or []:
        if not isinstance(m, dict) or not m.get("emitted", True):
            continue
        fam, ident = m.get("family"), m.get("normalized") or m.get("id") or m.get("text")
        if not ident:
            continue
        norm = normalize_identifier(fam, str(ident)) if fam else str(ident)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _first_identity(record: dict) -> str | None:
    ids = _identities(record)
    return ids[0] if ids else None


def m2b_min_components(cfg: V3AnalysisConfig | None = None) -> int:
    """component_count 下限 = 5 × cfg.n_outer_folds。"""
    c = cfg or V3AnalysisConfig()
    return M2B_MIN_COMPONENTS_FACTOR * int(c.n_outer_folds)


def build_e2_groups(records: Sequence[dict], *, arm: str = "h1",
                    cfg: V3AnalysisConfig | None = None) -> dict:
    """E2：并查集连通分量 + M2b 塌缩判定。**不使用 label，不做任何性能估计。**"""
    if arm != "h1":
        return not_applicable(arm, "E2")
    if not records:
        raise ValueError("empty records")

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_identity_edges = 0
    identity_degree: dict[str, int] = defaultdict(int)
    for r in records:
        c = "c:" + str(r["trace_id"])
        p = "p:" + str(r["group_id"])
        union(c, p)
        for ident in _identities(r):
            union(c, "i:" + ident)
            identity_degree[ident] += 1
            n_identity_edges += 1

    comp: dict[str, list[str]] = defaultdict(list)
    for r in records:
        comp[find("c:" + str(r["trace_id"]))].append(str(r["trace_id"]))
    sizes = sorted((len(v) for v in comp.values()), reverse=True)
    n = len(records)
    max_share = sizes[0] / n if sizes else 1.0
    degs = sorted(identity_degree.values(), reverse=True)
    min_components = m2b_min_components(cfg)
    # **严格大于**：share == 0.20 不塌缩；component_count == min 不塌缩、< min 才塌缩。
    collapsed = (len(sizes) < min_components) or (max_share > M2B_MAX_COMPONENT_SHARE)

    out = {
        "arm": arm, "estimand": "E2",
        "estimability": NON_ESTIMABLE if collapsed else ESTIMABLE,
        # ── 结构性 accounting（不可估时**仍必须输出**）──────────────────────
        "n_completions": n,
        "n_prompts": len({str(r["group_id"]) for r in records}),
        "n_identities": len(identity_degree),
        "n_identity_edges": n_identity_edges,
        "component_count": len(sizes),
        "component_sizes_top10": sizes[:10],
        "max_component_share": float(max_share),
        "hub_distribution": {
            "identity_degree_top10": degs[:10],
            "identity_degree_max": int(degs[0]) if degs else 0,
            "identity_degree_median": float(degs[len(degs) // 2]) if degs else 0.0,
            "n_identities_degree_ge_2": int(sum(1 for d in degs if d >= 2)),
        },
        "thresholds": {"max_component_share_strictly_greater_than":
                           M2B_MAX_COMPONENT_SHARE,
                       "min_components": min_components,
                       "min_components_rule": "5 * cfg.n_outer_folds"},
        "uses_label": False,
        "performance_estimates_computed": False,
    }
    if collapsed:
        out["reason"] = (
            f"连通图塌缩：component_count={len(sizes)} "
            f"(< {min_components}) 或 max_component_share={max_share:.4f} "
            f"(> {M2B_MAX_COMPONENT_SHARE})；按 M2b **不构造 CV**，只报结构统计")
        out["groups"] = None
    else:
        root_of = {}
        for root, traces in comp.items():
            for t in traces:
                root_of[t] = root
        out["groups"] = [root_of[str(r["trace_id"])] for r in records]
    return out


def e3_block_of(identity: str, cfg: V3AnalysisConfig) -> int:
    """冻结词表分块：`int(sha256(salt + "|" + identity), 16) mod e3_n_blocks`。"""
    h = hashlib.sha256((cfg.e3_block_salt + "|" + identity).encode("utf-8")).hexdigest()
    return int(h, 16) % int(cfg.e3_n_blocks)


def build_e3_groups(records: Sequence[dict], cfg: V3AnalysisConfig, *,
                    arm: str = "h1", block_index: int | None = None) -> dict:
    """E3：按 greedy completion 的**第一个** identifier 给 prompt 分块，label-free。

    `block_index=None` 时返回全部分块的 accounting；指定时返回该块的行子集。
    fail-closed：空 fold 或空训练补集 → `fold_structure_status = "empty"`，**不调用 sklearn**。
    **repooling 完成前不读 label**，因此不判"单类"——类别可估性一律 `not_evaluated`。

    ⚠️ **本实现只做了「分 fold」**：block-specific label 与 block-specific H pooling
    **尚未实现**，且**无法**由现有缓存重构 —— `h1_unit_hidden/*.npy` 与 `h1_hidden.npz`
    都已按 E1 口径 pool 成单向量（over `id_token_positions_local`），per-token hidden
    未保留。因此：

        wiring_status      = "needs_researcher_decision"（接线未完成，构成解盲阻断项）
        class_estimability = "not_evaluated"（**类别可估性未评估**，不冒充 estimable）
        fold_structure_status ∈ {"non_empty", "empty"}（**只描述结构**）
        uses_label = False（本函数任何路径都不读 label）

    补齐方案需 performance-blind 的 teacher-forced per-block repooling，
    **不得**用 E1 的 label/H 冒充。
    """
    if arm != "h1":
        return not_applicable(arm, "E3")
    if not records:
        raise ValueError("empty records")

    greedy = {}
    for r in records:
        if r.get("sample_type") == "greedy":
            greedy[str(r["group_id"])] = r
    prompt_block: dict[str, int] = {}
    n_prompts_no_identifier = 0
    for gid, r in greedy.items():
        ident = _first_identity(r)
        if ident is None:
            n_prompts_no_identifier += 1
            continue
        prompt_block[gid] = e3_block_of(ident, cfg)

    per_block: dict[int, int] = defaultdict(int)
    for gid in prompt_block:
        per_block[prompt_block[gid]] += 1
    out = {
        "arm": arm, "estimand": "E3",
        # 接线状态与统计可估性**分开**：前者说"没接完"，后者说"没评估"。
        "wiring_status": NEEDS_DECISION,
        "class_estimability": NOT_EVALUATED,
        "estimability": NOT_EVALUATED,        # 兼容旧读法；语义同 class_estimability
        "blocking_reason": (
            "block-specific label 与 block-specific H pooling 未实现；"
            "现有 hidden 已是 E1 口径的单向量（per-token hidden 未保留），无法由缓存重构。"
            "补齐需 performance-blind teacher-forced per-block repooling；"
            "**不得**用 E1 的 label/H 冒充"),
        "implemented": ["fold assignment by frozen vocabulary block"],
        "not_implemented": ["block-specific label", "block-specific H pooling"],
        "n_blocks": int(cfg.e3_n_blocks),
        "e3_block_salt": cfg.e3_block_salt,
        "n_prompts_assigned": len(prompt_block),
        "n_prompts_without_identifier": n_prompts_no_identifier,
        "prompts_per_block": {str(k): int(v) for k, v in sorted(per_block.items())},
        "uses_label": False, "performance_estimates_computed": False,
    }
    if block_index is None:
        out["fold_structure_status"] = NON_EMPTY if prompt_block else EMPTY
        if not prompt_block:
            out["reason"] = "没有任何 prompt 能分配到词表块（greedy completion 均无 identifier）"
        return out

    b = int(block_index)
    out["block_index"] = b
    keep_gid = {gid for gid, blk in prompt_block.items() if blk == b}
    rows = [i for i, r in enumerate(records) if str(r["group_id"]) in keep_gid]
    comp = [i for i, r in enumerate(records) if str(r["group_id"]) not in keep_gid]
    out["n_rows_in_block"] = len(rows)
    out["n_rows_in_complement"] = len(comp)
    # ⚠️ **repooling 完成前只能看结构，不能看类别**：block-specific label 尚未实现，
    # 拿 E1 的 label 判单类等于用错误的标签冒充块内标签。这里**完全不读 label**。
    if not rows:
        out["fold_structure_status"] = EMPTY
        out["reason"] = f"块 {b} 为空 fold（0 行）；fail closed，不调用 sklearn"
    elif not comp:
        out["fold_structure_status"] = EMPTY
        out["reason"] = f"块 {b} 的训练补集为空（0 行）；fail closed，不调用 sklearn"
    else:
        out["fold_structure_status"] = NON_EMPTY
        out["rows"] = rows
        out["complement_rows"] = comp
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 以下函数**读取 label**（mention 级）。
#
# 它们实现的是 E3 与 M2c 各自 estimand 的**定义**，不是诊断 —— 因此只允许在
# 解盲之后、`v3_analyze.analyze()` 内部调用。解盲前的结构审计请用上面的
# `build_e2_groups` / `build_e3_groups`（两者任何路径都不读 label）。
# ══════════════════════════════════════════════════════════════════════════════
READS_LABEL_POST_UNBLINDING_ONLY = (
    "reads label; callable only inside analyze() after record_unblinding")


def e3_block_of_mention(mention: dict, cfg: V3AnalysisConfig) -> int | None:
    """一条 mention 的词表块。无法规范化 → None。"""
    ident = mention.get("normalized") or mention.get("id") or mention.get("text")
    fam = mention.get("family")
    if not ident:
        return None
    norm = normalize_identifier(fam, str(ident)) if fam else str(ident)
    return e3_block_of(norm, cfg) if norm else None


def e3_block_label(record: dict, cfg: V3AnalysisConfig, block: int) -> dict:
    """**块内**标签：把冻结的 `completion_label` 规则原样限制到块 `block` 的 mention。

    M2d：「标签与 H pooling **只在该块内的 mention** 上计算」。规则本身**不改** ——
    primary = family in PRIMARY_FAMILIES 且 label in {grounded, fabricated}；
    块内有 primary → label = 1 iff 其中至少一条 fabricated；
    块内无 primary → **不 eligible**（与 E1 对 ineligible 的处理完全一致，不是新规则）。

    读 mention label，见 `READS_LABEL_POST_UNBLINDING_ONLY`。
    """
    prim = [m for m in (record.get("mentions") or [])
            if isinstance(m, dict) and ci._is_primary(m)
            and e3_block_of_mention(m, cfg) == int(block)]
    if not prim:
        return {"eligible": False, "label": None, "n_primary_mentions_in_block": 0}
    return {"eligible": True,
            "label": 1 if any(m.get("label") == "fabricated" for m in prim) else 0,
            "n_primary_mentions_in_block": len(prim)}


def build_e3_analysis(records: Sequence[dict], cfg: V3AnalysisConfig, *,
                      arm: str = "h1") -> dict:
    """E3 的**完整**构造：折归属 + 块内标签 + M2d-gate。

    折归属沿用 `build_e3_groups` 的口径（prompt 组按其 greedy completion 的第一个
    identifier 所属块整组归折），**不重算、不换 salt、不改 F**。

    口径说明（如实记录，不是新决定）：`_identities()` 用 `m.get("emitted", True)`
    过滤，而 finalized mention **没有** `emitted` 字段（它记的是
    `label in {grounded, fabricated, echoed}`），因此该过滤实际不生效 ——
    「第一个 identifier」取的是出现序上的第一条 mention，含 echoed 与 CVE。
    这与 M2d 未加限定的「第一个 identifier」字面一致，也与既有 accounting 一致，
    故**保持不变**；此处只把它写明，不静默依赖。

    M2d-gate（跑前写死）：任一 fold 为空 / 块内标签下单类 / 其训练补集单类
    → E3 = NON_ESTIMABLE，只报每折样本数、正负例数、mention retention。
    """
    if arm != "h1":
        return not_applicable(arm, "E3")
    if not records:
        raise ValueError("empty records")

    structural = build_e3_groups(records, cfg, arm=arm)
    greedy = {str(r["group_id"]): r for r in records if r.get("sample_type") == "greedy"}
    prompt_block: dict[str, int] = {}
    for gid, r in greedy.items():
        ident = _first_identity(r)
        if ident is not None:
            prompt_block[gid] = e3_block_of(ident, cfg)

    n_blocks = int(cfg.e3_n_blocks)
    rows, y_vals, fold_of, per_row = [], [], [], []
    n_no_fold = n_ineligible = 0
    for i, r in enumerate(records):
        blk = prompt_block.get(str(r["group_id"]))
        if blk is None:
            n_no_fold += 1
            continue
        lab = e3_block_label(r, cfg, blk)
        if not lab["eligible"]:
            n_ineligible += 1
            continue
        rows.append(i)
        y_vals.append(int(lab["label"]))
        fold_of.append(blk)
        per_row.append({"row": i, "block": blk,
                        "n_primary_mentions_in_block":
                            lab["n_primary_mentions_in_block"]})

    y = np.asarray(y_vals, dtype=int)
    fold_arr = np.asarray(fold_of, dtype=int)
    folds, per_fold = [], {}
    degenerate: list[str] = []
    for b in range(n_blocks):
        test = np.flatnonzero(fold_arr == b)
        train = np.flatnonzero(fold_arr != b)
        n_pos = int((y[test] == 1).sum()) if test.size else 0
        n_neg = int((y[test] == 0).sum()) if test.size else 0
        per_fold[str(b)] = {
            "n": int(test.size), "n_positive": n_pos, "n_negative": n_neg,
            "n_train_complement": int(train.size),
            "train_n_positive": int((y[train] == 1).sum()) if train.size else 0,
            "train_n_negative": int((y[train] == 0).sum()) if train.size else 0}
        if test.size == 0:
            degenerate.append(f"block {b}: (a) empty fold")
            continue
        if n_pos == 0 or n_neg == 0:
            degenerate.append(
                f"block {b}: (b) single class under block label (pos={n_pos} neg={n_neg})")
        if train.size == 0 or len(set(y[train].tolist())) < 2:
            degenerate.append(f"block {b}: (c) train complement empty or single class")
        folds.append((train, test))

    n_total = len(records)
    retention = {
        "n_records": n_total,
        "n_rows_in_e3_population": len(rows),
        "n_rows_without_fold": n_no_fold,
        "n_rows_ineligible_in_own_block": n_ineligible,
        "mention_retention_rate": (len(rows) / n_total) if n_total else 0.0,
        "note": ("块内无 primary mention 的 trace 按冻结 completion_label 规则"
                 "**不 eligible**，与 E1 的 ineligible 处理一致；这不是事后排除"),
    }
    estimable = not degenerate
    out = {
        "arm": arm, "estimand": "E3",
        "estimability": ESTIMABLE if estimable else NON_ESTIMABLE,
        "class_estimability": ESTIMABLE if estimable else NON_ESTIMABLE,
        "wiring_status": "wired",
        "uses_label": True,
        "label_caliber": ("frozen completion_label rule restricted to the block's "
                          "primary mentions (M2d); the rule itself is unchanged"),
        "n_blocks": n_blocks, "e3_block_salt": cfg.e3_block_salt,
        "fold_assignment_rule": ("prompt group -> block of the FIRST identifier of its "
                                 "greedy completion; whole group to one fold (M2d)"),
        "first_identifier_caliber": (
            "出现序第一条 mention（含 echoed / CVE）—— finalized mention 无 emitted "
            "字段，_identities 的过滤实际不生效；与 M2d 未加限定的措辞一致，保持不变"),
        "per_fold": per_fold,
        "retention": retention,
        "structural_audit": {k: structural[k] for k in
                             ("n_prompts_assigned", "n_prompts_without_identifier",
                              "prompts_per_block") if k in structural},
        "degenerate_reasons": degenerate,
        "gate": ("M2d-gate: 任一 fold 空 / 块内单类 / 训练补集单类 -> NON_ESTIMABLE; "
                 "不得改 F、换 salt、重新分块、合并或丢弃 fold 以求可估"),
        "performance_estimates_computed": False,
    }
    if estimable:
        out["_rows"] = rows
        out["_y"] = y
        out["_folds"] = folds
        out["_fold_of"] = fold_arr
    else:
        out["reason"] = "; ".join(degenerate)
    return out


def build_m2c_groups(records: Sequence[dict], cfg: V3AnalysisConfig | None = None, *,
                     arm: str = "h1") -> dict:
    """M2c：`C = prompt 并 fabricated-id` 连通分量。**label-informed sensitivity，非 primary。**

    与 E2 的差别只有一处：identity 边**只由 label == "fabricated" 的 primary mention**
    产生 —— 这正是它 label-informed 的原因。

    报告时必须同时声明三项（`interpretation_caveats`）：
      (i)   该分组使用 fabricated **标签**构造折 → label-informed；
      (ii)  偏倚方向**未知**（不得声称"只会压低性能"）；
      (iii) grounded identifier 仍跨折 → 身份通道只封了一半。
    """
    if arm != "h1":
        return not_applicable(arm, "M2c")
    if not records:
        raise ValueError("empty records")
    cfg = cfg or V3AnalysisConfig()

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_fab_edges = 0
    fab_ids: set[str] = set()
    for r in records:
        c = "c:" + str(r["trace_id"])
        union(c, "p:" + str(r["group_id"]))
        for m in (r.get("mentions") or []):
            if not (isinstance(m, dict) and ci._is_primary(m)
                    and m.get("label") == "fabricated"):
                continue
            ident = m.get("normalized") or m.get("id") or m.get("text")
            if not ident:
                continue
            norm = normalize_identifier(m.get("family"), str(ident))
            if not norm:
                continue
            union(c, "f:" + norm)
            fab_ids.add(norm)
            n_fab_edges += 1

    comp: dict[str, list[str]] = defaultdict(list)
    for r in records:
        comp[find("c:" + str(r["trace_id"]))].append(str(r["trace_id"]))
    root_of = {t: root for root, ts in comp.items() for t in ts}
    sizes = sorted((len(v) for v in comp.values()), reverse=True)
    n = len(records)
    return {
        "arm": arm, "estimand": "M2c",
        "role": "label_informed_sensitivity",
        "is_primary": False,
        "grouping": "C = prompt UNION fabricated-identifier connected component",
        "groups": [root_of[str(r["trace_id"])] for r in records],
        "n_completions": n,
        "n_fabricated_identities": len(fab_ids),
        "n_fabricated_identity_edges": n_fab_edges,
        "component_count": len(sizes),
        "component_sizes_top10": sizes[:10],
        "max_component_share": float(sizes[0] / n) if sizes else 1.0,
        "uses_label": True,
        "interpretation_caveats": [
            "(i) 该分组使用 fabricated **标签**构造折 -> label-informed",
            "(ii) 偏倚方向**未知** —— 不得声称『只会压低性能』",
            "(iii) grounded identifier 仍跨折 -> 身份通道只封了一半",
        ],
        "must_not": "不得作为任何 primary；不得与 E1 的判读混用",
    }
