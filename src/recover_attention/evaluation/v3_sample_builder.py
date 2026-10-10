"""Stage 1 正式名单与 G3/G4 gate 名单的构建（启动 gate **G2 + G5**）。

执行顺序（preregistration_v3 §12.1，代码强制）：

    require_v3_frozen()            ← 第一件事；G2 不过则**不创建任何名单、不写任何文件**
    → 从**冻结源数据**重建候选池   ← 不把容量审计 JSON 当成员名单读
    → 重算并核对 pool chain hash
    → 构建正式集 / G3 / G4
    → G5 互斥审计（H1 查 source_entry_id 与 g5_prompt_hash；MCQ 查 question_hash）
    → 全部写入**临时目录**，校验通过后**原子落盘**；失败不留半成品

名单口径（v3.1 §M6 / §7.5 / §11.10 + 2026-08-08 执行级裁决）：

    H1  eligible pool 1,547 个全新 recall 条目（attack 718 + cwe 829）
        − 2（与 Stage 0 的 prompt 碰撞）− 1（正式集内同 prompt 去重）→ **正式集 1,544**
        正式集须满足 **1544 records = 1544 distinct prompts = 1544 groups**
        G3 20（各家族 10）、G4 40（各家族 20），全部取自 Stage 0 **已烧过**的 recall 条目
        不使用 open_gen，不消耗正式新池
        gate 产物标 gate_only=true / previously_burned=true，**不得进入正式分析**
    MCQ 正式集 = fresh ranked pool 前 1,760；其后 20 给 G3，再其后 40 给 G4

⚠️ **g5_prompt_hash ≠ generation 侧的 prompt_hash**。后者保持原义不动；前者是本模块
   为 G5 定义的独立量，对 `build_h1_chat_messages(record)` 的**规范化 canonical JSON**
   计算，Stage 0 与 Stage 1 用**同一规则重算**，故可比。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from recover_attention.cyber_data import (
    _normalize_question_for_grouping,
    load_cybermetric_records,
    normalize_cybermetric_record,
)
from recover_attention.h1_data import build_h1_chat_messages, build_recall_question
from recover_attention.h1_identifier import OntologyEntry, normalize_identifier
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_freeze import (
    normalize_text_bytes,
    require_v3_frozen,
    sha256_text_lf,
)

SCHEMA_VERSION = "stage1_sample_manifest_v1"
PRIMARY_FAMILIES = ("attack", "cwe")
_ENTRYLESS = (None, "None")


# ── 哈希口径 ─────────────────────────────────────────────────────────────────
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """G5 用的规范化 JSON：键排序、无多余空白、非 ASCII 原样、行尾已在上游统一。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_str(text: str) -> str:
    return normalize_text_bytes(text.encode("utf-8"), source="<str>").decode("utf-8")


def g5_prompt_hash(record: dict) -> str:
    """G5 的 prompt 身份：`build_h1_chat_messages(record)` 的规范化 canonical JSON 的 sha256。

    ⚠️ **不是** `sha256(question_text)`，也**不是** generation 侧的 `prompt_hash`
    （后者保持原义，本模块不碰）。取 chat messages 是因为模型实际看到的是它 ——
    只哈希 question_text 会漏掉 system prompt 的差异。
    每条 message 的 content 先做行尾规范化，故 Stage 0 与 Stage 1 可用同一规则重算。
    """
    messages = build_h1_chat_messages(record)
    normalized = [
        {"role": _normalize_str(m["role"]), "content": _normalize_str(m["content"])}
        for m in messages
    ]
    return _sha256(canonical_json(normalized))


def chain_digest(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def salted_rank_key(salt: str, identity: str) -> str:
    return _sha256(f"{salt}|{identity}")


# ── 源数据重建（**不读容量审计 JSON 的成员**）──────────────────────────────
@dataclass(frozen=True)
class H1Pools:
    entries: dict[str, OntologyEntry]       # "family:id" → entry
    fresh_ids: list[str]                    # 排序后的 "family:id"
    burned_ids: list[str]
    stage0_prompt_hashes: frozenset[str]


def load_ontology_entries(ontology_dir: Path) -> dict[str, OntologyEntry]:
    entries: dict[str, OntologyEntry] = {}
    for family in PRIMARY_FAMILIES:
        path = ontology_dir / family / "ontology_index.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ident = row["normalized_id"]
                if row["family"] != family:
                    raise ValueError(f"{path}:{line_no} family 不符: {row['family']!r}")
                if normalize_identifier(family, ident) != ident:
                    raise ValueError(f"{path}:{line_no} {ident!r} 不是 normalize_identifier 的不动点")
                key = f"{family}:{ident}"
                if key in entries:
                    raise ValueError(f"{path}:{line_no} 重复条目 {key}")
                entries[key] = OntologyEntry(
                    family=family, normalized_id=ident, status=row.get("status", ""),
                    name=row.get("name"), description=row.get("description"),
                    metadata=row.get("metadata"),
                )
    return entries


def load_h1_pools(ontology_dir: Path, stage0_samples: Path) -> H1Pools:
    entries = load_ontology_entries(ontology_dir)

    burned: set[str] = set()
    stage0_hashes: set[str] = set()
    with stage0_samples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Stage 0 全部 480 条（含 open_gen 与 cve）都要进 g5_prompt_hash 的比对集：
            # G5 要求的是"与 Stage 0 的 480 条零交集"，不是"与 recall 零交集"。
            stage0_hashes.add(g5_prompt_hash(row))
            if row["family"] in PRIMARY_FAMILIES and row["source_entry_id"] not in _ENTRYLESS:
                if row["route"] != "recall":
                    raise ValueError(f"意外：非 recall 路由带有 source_entry_id: {row['example_id']}")
                burned.add(f"{row['family']}:{row['source_entry_id']}")

    unknown = burned - set(entries)
    if unknown:
        raise ValueError(f"Stage 0 用了不在 ontology index 里的条目: {sorted(unknown)[:5]}")

    fresh = sorted(set(entries) - burned)
    # Stage 0 的 recall builder 只用有 name 或 description 的条目；此处显式复核，
    # 而不是假定 —— 若哪天索引变了，1547 这个冻结值必须当场炸，不能悄悄少几条。
    without_meta = [k for k in fresh if not (entries[k].name or entries[k].description)]
    if without_meta:
        raise ValueError(
            f"{len(without_meta)} 个候选条目既无 name 也无 description，无法构造 recall 问题: "
            f"{without_meta[:5]}")

    return H1Pools(entries=entries, fresh_ids=fresh, burned_ids=sorted(burned),
                   stage0_prompt_hashes=frozenset(stage0_hashes))


def make_h1_record(key: str, entry: OntologyEntry) -> dict:
    """构造一条 Stage 1 recall 样本（与 Stage 0 同模板 `recall_v1`）。

    ⚠️ **`group_id` 由 prompt 身份导出，不由条目 id 导出。**
    ATT&CK 的 revoked 条目会与其继任者共享 name+description（实测 T1536 ↔ T1578.004），
    产出**逐字节相同**的 prompt。若按条目分组，同一条 prompt 会落进不同的 fold ——
    那正是 M2a 的 prompt-disjoint 分组要防的泄漏。M2a 的原文是"按 **source prompt** 分组"，
    故以 prompt 身份为 group 是对既有冻结文本的忠实实现，不是新增规则。
    """
    question, params = build_recall_question(entry)
    ident = entry.normalized_id
    record = {
        "example_id": f"stage1_h1_recall_{entry.family}_{_sha256(key)[:12]}",
        "route": "recall",
        "family": entry.family,
        "prompt_params": params,
        "question_text": question,
        "source_entry_id": ident,
        "source_entry_metadata": {"name": entry.name, "description": entry.description},
        "entry_status": entry.status,
        "label_space": "open_identifier",
    }
    digest = g5_prompt_hash(record)
    record["g5_prompt_hash"] = digest
    record["group_id"] = f"stage1_h1_prompt_{digest[:16]}"
    return record


RETIRED_STATUSES = ("revoked", "deprecated")


def _is_retired(entry: OntologyEntry) -> bool:
    return (entry.status or "").casefold() in RETIRED_STATUSES


def select_h1_primary(pools: H1Pools) -> tuple[list[str], list[dict]]:
    """从 eligible pool 确定性地导出正式集，并给出**被排除记录及原因**（v3.1 §M6）。

    两步，顺序固定：

    ```
    (1) 跨 Stage prompt 碰撞 → 排除
        g5_prompt_hash 落在 Stage 0 的 480 条之内的条目。G5 要求 prompt 级零交集，
        分组解决不了 —— 那会让 Stage 1 重问 Stage 0 已烧过的 prompt。
    (2) 正式集内部同 prompt → 去重，只保留一条
        按 g5_prompt_hash 分组；组内优先保留**非 revoked/deprecated**；
        仍并列则按 "family:normalized_id" 字典序取首个。
        ❌ 不采用"同 prompt 共享 group"的做法：那会让该 prompt 拿到 12 条生成
           而其他 prompt 只有 6 条，造成不等权，也违反「正式集内 prompt 零重复」。
    ```

    根因（写进 manifest）：ATT&CK 的 revoked 条目与其继任者共享 `name`+`description`，
    而 `recall_v1` 只用这两个字段。这是**在任何模型运行之前**由 G5 发现的**数据身份冲突**，
    与性能、结果或择优**无关**。
    """
    records = {k: make_h1_record(k, pools.entries[k]) for k in pools.fresh_ids}
    excluded: list[dict] = []

    # (1) 跨 Stage 碰撞
    kept: list[str] = []
    for key in pools.fresh_ids:
        if records[key]["g5_prompt_hash"] in pools.stage0_prompt_hashes:
            excluded.append({
                "entry": key,
                "status": pools.entries[key].status,
                "reason": "cross_stage_prompt_collision",
                "detail": "其 prompt 与 Stage 0 的 480 条之一逐字节相同，违反 G5 的 prompt 级零交集",
                "g5_prompt_hash": records[key]["g5_prompt_hash"],
            })
        else:
            kept.append(key)

    # (2) 内部去重
    by_hash: dict[str, list[str]] = {}
    for key in kept:
        by_hash.setdefault(records[key]["g5_prompt_hash"], []).append(key)

    primary: list[str] = []
    for digest, keys in by_hash.items():
        if len(keys) == 1:
            primary.append(keys[0])
            continue
        winner = sorted(keys, key=lambda k: (_is_retired(pools.entries[k]), k))[0]
        primary.append(winner)
        for key in sorted(keys):
            if key == winner:
                continue
            excluded.append({
                "entry": key,
                "status": pools.entries[key].status,
                "reason": "intra_set_duplicate_prompt",
                "detail": (f"与 {winner} 产出逐字节相同的 prompt；"
                           f"按规则优先保留非 revoked/deprecated，其次字典序首个"),
                "g5_prompt_hash": digest,
                "kept_instead": winner,
            })

    return sorted(primary), sorted(excluded, key=lambda d: d["entry"])


def select_h1_gate(pools: H1Pools, cfg: V3AnalysisConfig) -> dict[str, list[str]]:
    """从**已烧过**的条目里确定性选 G3/G4：各家族按 salted rank 升序，G3 在前、G4 紧随。

    每个家族独立排序，故 G3/G4 的家族配额（10/10 与 20/20）是**构造性保证**，
    不依赖全局排序恰好把两个家族分匀。
    """
    n_g3 = cfg.n_g3_per_arm // len(cfg.h1_gate_families)
    n_g4 = cfg.n_g4_per_arm // len(cfg.h1_gate_families)
    g3: list[str] = []
    g4: list[str] = []
    for family in cfg.h1_gate_families:
        ranked = sorted(
            (k for k in pools.burned_ids if k.startswith(f"{family}:")),
            key=lambda k: (salted_rank_key(cfg.h1_gate_selection_salt, k), k),
        )
        if len(ranked) < n_g3 + n_g4:
            raise ValueError(f"{family} 的已烧条目只有 {len(ranked)} 个，不足 {n_g3 + n_g4}")
        g3.extend(ranked[:n_g3])
        g4.extend(ranked[n_g3:n_g3 + n_g4])
    return {"G3": g3, "G4": g4}


# ── MCQ ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class McqPools:
    ranked_fresh: list[dict]        # 按冻结排序的全部 fresh 题（含 question_hash 与原记录）
    stage0_hashes: frozenset[str]


def _question_hash(question: str) -> str:
    return _sha256(_normalize_question_for_grouping(question))


def load_mcq_pools(cyber_dir: Path, cfg: V3AnalysisConfig) -> McqPools:
    pool_path = cyber_dir / "CyberMetric-10000-v1.json"
    stage0_path = cyber_dir / "CyberMetric-2000-v1.json"

    stage0_hashes = {
        _question_hash(r["question"]) for r in load_cybermetric_records(stage0_path)
    }

    seen: set[str] = set()
    fresh: list[dict] = []
    for index, raw in enumerate(load_cybermetric_records(pool_path)):
        normalized = normalize_cybermetric_record(
            raw, original_index=index, source=pool_path.name)
        digest = _question_hash(normalized["question"])
        if digest in seen or digest in stage0_hashes:
            continue                      # 文件内归一化重复 / 与 Stage 0 重叠，都不进池
        seen.add(digest)
        fresh.append({
            "question_hash": digest,
            "source": pool_path.name,
            "original_index": index,
            "question": normalized["question"],
            "answers": normalized["answers"],
            "gold_original_label": normalized["gold_original_label"],
        })

    fresh.sort(key=lambda row: (
        salted_rank_key(cfg.mcq_selection_salt, row["question_hash"]), row["question_hash"]))
    for rank, row in enumerate(fresh):
        row["rank"] = rank
    return McqPools(ranked_fresh=fresh, stage0_hashes=frozenset(stage0_hashes))


# ── G5 ───────────────────────────────────────────────────────────────────────
def _status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["entry_status"]] = counts.get(row["entry_status"], 0) + 1
    return dict(sorted(counts.items()))


def _dup_check(name: str, values: list[str]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} 集内存在重复项（{len(values) - len(set(values))} 个）")


def diagnose_g5_conflicts(pools: H1Pools) -> dict:
    """**只诊断，不抛错、不写名单。** 用于在 G5 冲突尚未裁决时如实报告冲突。

    与 `run_g5` 的分工：那个是 gate（不过就停），这个是**病历**。
    两者共用同一套哈希口径，故诊断结论与 gate 结论必然一致。
    """
    records = {k: make_h1_record(k, pools.entries[k]) for k in pools.fresh_ids}

    stage0_index: dict[str, list[str]] = {}
    with_stage0: list[dict] = []
    for key, record in records.items():
        if record["g5_prompt_hash"] in pools.stage0_prompt_hashes:
            with_stage0.append({
                "fresh_entry": key,
                "status": pools.entries[key].status,
                "g5_prompt_hash": record["g5_prompt_hash"],
            })

    internal: dict[str, list[str]] = {}
    for key, record in records.items():
        internal.setdefault(record["g5_prompt_hash"], []).append(key)
    internal_dups = [
        {"g5_prompt_hash": h, "fresh_entries": sorted(keys),
         "statuses": [pools.entries[k].status for k in sorted(keys)]}
        for h, keys in internal.items() if len(keys) > 1
    ]

    del stage0_index
    return {
        "schema_version": "stage1_g5_conflict_report_v1",
        "n_fresh_entries": len(pools.fresh_ids),
        "n_distinct_prompts": len(internal),
        "cross_stage_collisions": {
            "count": len(with_stage0),
            "detail": sorted(with_stage0, key=lambda d: d["fresh_entry"]),
            "consequence": (
                "违反 v3 §M6 的 G5：新 prompt 与 Stage 0 的 480 条须**零交集**。"
                "这两条无法用分组解决 —— 它们会让 Stage 1 重问 Stage 0 已烧过的 prompt。"
            ),
        },
        "intra_stage_duplicate_prompts": {
            "count": len(internal_dups),
            "detail": internal_dups,
            "consequence": (
                "v3.0 曾拟用「同 prompt 共享 group」处理；**v3.1 改为确定性去重** ——"
                "保留两条会让该 prompt 得到 12 条生成而其他只有 6 条（不等权），"
                "也违反「正式集内 prompt 零重复」。去重规则见 select_h1_primary。"
            ),
        },
        "root_cause": (
            "ATT&CK 的 revoked 条目与其继任者共享 name+description，recall 模板只用这两个字段，"
            "故产出逐字节相同的 prompt。索引含 revoked 149 / deprecated 12（attack）"
            "与 Deprecated 25（cwe）。Stage 0 只抽了 140/家族，未撞上；取满整池才暴露。"
        ),
        "blocking": len(with_stage0) > 0,
        "not_decided_here": (
            "排除这 2 条会使 N_H1 由 1547 变为 1545，而 N_H1 已写进冻结的预注册与 lock。"
            "本模块**不自行裁决**：按 lock 的规则，改冻结值须 bump version、重算 hash、更新 lock。"
        ),
    }


def run_g5(h1: dict[str, list[dict]], mcq: dict[str, list[dict]], *,
           pools: H1Pools, mcq_pools: McqPools) -> dict:
    """G5 互斥审计。任一条不成立即抛错——本函数**不返回 False 让调用方自行决定**。"""
    primary_ids = [r["source_entry_id"] for r in h1["primary"]]
    primary_hashes = [r["g5_prompt_hash"] for r in h1["primary"]]
    primary_groups = [r["group_id"] for r in h1["primary"]]
    _dup_check("H1 primary source_entry_id", primary_ids)

    # v3.1 的核心恒等式：**记录数 = 不同 prompt 数 = group 数**。
    # v3.0 曾允许"同 prompt 共享 group"，那会让该 prompt 拿到 12 条生成而其他只有 6 条
    # （不等权），且违反"正式集内 prompt 零重复"。改为确定性去重后，三者必须严格相等。
    n = len(h1["primary"])
    if not (n == len(set(primary_hashes)) == len(set(primary_groups))):
        raise ValueError(
            f"G5 失败：正式集不满足 records = distinct prompts = groups —— "
            f"{n} / {len(set(primary_hashes))} / {len(set(primary_groups))}")
    for gate in ("G3", "G4"):
        _dup_check(f"H1 {gate} g5_prompt_hash", [r["g5_prompt_hash"] for r in h1[gate]])

    burned_ids = {k.split(":", 1)[1] for k in pools.burned_ids}
    overlap_ids = set(primary_ids) & burned_ids
    if overlap_ids:
        raise ValueError(f"G5 失败：H1 正式集与 Stage 0 条目交集非空: {sorted(overlap_ids)[:5]}")
    overlap_hashes = set(primary_hashes) & pools.stage0_prompt_hashes
    if overlap_hashes:
        raise ValueError(f"G5 失败：H1 正式集与 Stage 0 的 g5_prompt_hash 交集非空 "
                         f"({len(overlap_hashes)} 条)")

    sets = {name: {r["source_entry_id"] for r in rows} for name, rows in h1.items()}
    for a, b in (("primary", "G3"), ("primary", "G4"), ("G3", "G4")):
        if sets[a] & sets[b]:
            raise ValueError(f"G5 失败：H1 的 {a} 与 {b} 交集非空: {sorted(sets[a] & sets[b])[:5]}")
    for name in ("G3", "G4"):
        stray = sets[name] - burned_ids
        if stray:
            raise ValueError(f"G5 失败：H1 {name} 含非 Stage 0 已烧条目 {sorted(stray)[:5]} "
                             f"—— gate 不得消耗正式新池")

    mcq_sets = {name: {r["question_hash"] for r in rows} for name, rows in mcq.items()}
    for name, rows in mcq.items():
        _dup_check(f"MCQ {name} question_hash", [r["question_hash"] for r in rows])
        if mcq_sets[name] & mcq_pools.stage0_hashes:
            raise ValueError(f"G5 失败：MCQ {name} 与 Stage 0 的 2000 题交集非空")
    for a, b in (("primary", "G3"), ("primary", "G4"), ("G3", "G4")):
        if mcq_sets[a] & mcq_sets[b]:
            raise ValueError(f"G5 失败：MCQ 的 {a} 与 {b} 交集非空")

    return {
        "h1_primary_vs_stage0_entry_ids": 0,
        "h1_primary_vs_stage0_g5_prompt_hash": 0,
        "h1_primary_g3_g4_pairwise_disjoint": True,
        "h1_gate_subset_of_stage0_burned": True,
        "mcq_all_vs_stage0_question_hash": 0,
        "mcq_primary_g3_g4_pairwise_disjoint": True,
        "h1_primary_n_records": n,
        "h1_primary_n_distinct_prompts": len(set(primary_hashes)),
        "h1_primary_n_groups": len(set(primary_groups)),
        "h1_records_eq_prompts_eq_groups": True,
        "note": (
            "H1 的 gate 名单**刻意**与 Stage 0 的 g5_prompt_hash 相交（同条目同模板）——"
            "它们就是已烧样本，标 previously_burned=true。零交集只对正式集要求。"
        ),
    }


# ── 构建 + 原子落盘 ──────────────────────────────────────────────────────────
def build_sample_sets(pools: H1Pools, mcq_pools: McqPools, cfg: V3AnalysisConfig
                      ) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    if len(pools.fresh_ids) != cfg.n_h1_eligible_pool:
        raise ValueError(
            f"H1 eligible pool {len(pools.fresh_ids)} != 冻结的 {cfg.n_h1_eligible_pool}。"
            "冻结值不得因源数据变动而静默调整。")

    primary_keys, excluded = select_h1_primary(pools)
    if len(primary_keys) != cfg.n_h1_prompts:
        raise ValueError(
            f"排除后正式集 {len(primary_keys)} != 冻结的 N_H1={cfg.n_h1_prompts}"
            f"（eligible {len(pools.fresh_ids)} − 排除 {len(excluded)}）。")

    h1_primary = [make_h1_record(k, pools.entries[k]) for k in primary_keys]
    gate_ids = select_h1_gate(pools, cfg)
    h1: dict[str, list[dict]] = {"primary": h1_primary}
    for gate, keys in gate_ids.items():
        rows = []
        for key in keys:
            record = make_h1_record(key, pools.entries[key])
            record.update({"gate": gate, "gate_only": True, "previously_burned": True})
            rows.append(record)
        h1[gate] = rows

    n_primary, n_g3, n_g4 = cfg.n_mcq_questions, cfg.n_g3_per_arm, cfg.n_g4_per_arm
    if len(mcq_pools.ranked_fresh) < n_primary + n_g3 + n_g4:
        raise ValueError(f"MCQ 新池只有 {len(mcq_pools.ranked_fresh)} 题，不足 "
                         f"{n_primary + n_g3 + n_g4}")
    ranked = mcq_pools.ranked_fresh
    mcq: dict[str, list[dict]] = {
        "primary": [dict(r) for r in ranked[:n_primary]],
        "G3": [dict(r) for r in ranked[n_primary:n_primary + n_g3]],
        "G4": [dict(r) for r in ranked[n_primary + n_g3:n_primary + n_g3 + n_g4]],
    }
    for gate in ("G3", "G4"):
        for row in mcq[gate]:
            row.update({"gate": gate, "gate_only": True, "previously_burned": False})
    return h1, mcq, excluded


_FILENAMES = {
    ("h1", "primary"): "h1_primary_entries.jsonl",
    ("h1", "G3"): "h1_g3_entries.jsonl",
    ("h1", "G4"): "h1_g4_entries.jsonl",
    ("mcq", "primary"): "mcq_primary_questions.jsonl",
    ("mcq", "G3"): "mcq_g3_questions.jsonl",
    ("mcq", "G4"): "mcq_g4_questions.jsonl",
}
MANIFEST_NAME = "stage1_sample_manifest.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def build_stage1_samples(
    *,
    out_dir: Path,
    ontology_dir: Path,
    stage0_samples: Path,
    cyber_dir: Path,
    capacity_audit: Path,
    prereg: Path,
    lock: Path,
    cfg: V3AnalysisConfig | None = None,
) -> dict:
    """全流程。G2 不过 / 任一校验不过 → 抛错，且**不留下任何半成品**。"""
    cfg = cfg or V3AnalysisConfig()
    cfg.assert_production()

    # (1) G2 —— 第一件事。不通过则什么都不建。
    freeze = require_v3_frozen(prereg, lock)

    # (2) 从冻结源数据重建
    pools = load_h1_pools(ontology_dir, stage0_samples)
    mcq_pools = load_mcq_pools(cyber_dir, cfg)

    # (3) 重算 chain hash 并与容量审计核对（只读审计的**校验值**，不读它的成员名单）
    audit = json.loads(capacity_audit.read_text(encoding="utf-8"))
    expected_chain = audit["h1"]["eligible_pool_chain_sha256"]
    actual_chain = chain_digest(pools.fresh_ids)
    if actual_chain != expected_chain:
        raise ValueError(
            f"H1 可用池的 chain hash 与容量审计不一致：审计 {expected_chain}，"
            f"重算 {actual_chain}。源数据或口径已变，停。")
    expected_fresh = audit["mcq"]["fresh_pool_size"]
    if len(mcq_pools.ranked_fresh) != expected_fresh:
        raise ValueError(f"MCQ 新池 {len(mcq_pools.ranked_fresh)} != 审计的 {expected_fresh}")

    # (4) 构建 + (5) G5
    h1, mcq, excluded = build_sample_sets(pools, mcq_pools, cfg)
    g5 = run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gates": {"G2": freeze.as_dict(), "G5": g5},
        "prereg": {
            "path": str(prereg).replace("\\", "/"),
            "sha256": freeze.current_sha256,
            "hash_mode": freeze.hash_mode,
            "lock_version": freeze.locked_version,
            "frozen_on": freeze.frozen_on,
        },
        "source_files": {
            str(p).replace("\\", "/"): sha256_text_lf(p) if p.suffix in (".md", ".jsonl")
            else hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ontology_dir / "attack" / "ontology_index.jsonl",
                ontology_dir / "cwe" / "ontology_index.jsonl",
                stage0_samples,
                cyber_dir / "CyberMetric-10000-v1.json",
                cyber_dir / "CyberMetric-2000-v1.json",
                capacity_audit,
            ]
        },
        "salts": {
            "e3_block_salt": cfg.e3_block_salt,
            "mcq_selection_salt": cfg.mcq_selection_salt,
            "h1_gate_selection_salt": cfg.h1_gate_selection_salt,
        },
        "g5_prompt_hash_rule": (
            "sha256(canonical_json([{role, content(LF-normalized)} for build_h1_chat_messages]))"
            " —— 独立于 generation 侧的 prompt_hash，后者保持原义"
        ),
        "counts": {
            "h1": {name: len(rows) for name, rows in h1.items()},
            "mcq": {name: len(rows) for name, rows in mcq.items()},
            "h1_eligible_pool": len(pools.fresh_ids),
            "h1_excluded": len(excluded),
            "h1_pool_burned": len(pools.burned_ids),
            "h1_primary_distinct_prompts": g5["h1_primary_n_distinct_prompts"],
            "h1_primary_distinct_groups": g5["h1_primary_n_groups"],
            "mcq_pool_fresh": len(mcq_pools.ranked_fresh),
            "k_traces": cfg.k_traces,
            "h1_primary_traces": len(h1["primary"]) * cfg.k_traces,
            "mcq_primary_traces": len(mcq["primary"]) * cfg.k_traces,
        },
        "chain_sha256": {
            "h1_pool_fresh": actual_chain,
            **{f"h1_{name}": chain_digest(r["source_entry_id"] for r in rows)
               for name, rows in h1.items()},
            **{f"mcq_{name}": chain_digest(r["question_hash"] for r in rows)
               for name, rows in mcq.items()},
        },
        "h1_exclusions": {
            "rule": (
                "(1) g5_prompt_hash 落在 Stage 0 的 480 条之内 → 排除（G5 的 prompt 级零交集）；"
                "(2) 余下按 g5_prompt_hash 分组，组内优先保留非 revoked/deprecated，"
                "仍并列则按 family:normalized_id 字典序取首个"),
            "discovered_by": "G5，在**任何模型运行之前**",
            "nature": "数据身份冲突，与性能、结果或择优无关",
            "root_cause": (
                "ATT&CK 的 revoked 条目与其继任者共享 name+description，"
                "而 recall_v1 模板只用这两个字段 → 产出逐字节相同的 prompt"),
            "accounting": {
                "eligible_pool": len(pools.fresh_ids),
                "cross_stage_prompt_collision": sum(
                    1 for e in excluded if e["reason"] == "cross_stage_prompt_collision"),
                "intra_set_duplicate_prompt": sum(
                    1 for e in excluded if e["reason"] == "intra_set_duplicate_prompt"),
                "primary": len(h1["primary"]),
            },
            "excluded_records": excluded,
        },
        "entry_status_distribution": {
            "primary": _status_counts(h1["primary"]),
            "note": (
                "revoked/deprecated 条目**继续保留**（除非上面的去重规则淘汰了它）。"
                "当前 estimand 判断的是 identifier 是否存在于冻结 ontology，"
                "不是该标识符是否仍被推荐使用；原协议明确把这些状态视为 existence-positive。"
                "此分布随结果同报，作为局限说明。"),
        },
        "boundary": {
            "gate_lists_are_gate_only": True,
            "gate_lists_must_not_enter_analysis": True,
            "no_model_was_run": True,
            "lists_tracked_in_vcs": False,
            "lists_note": "名单不进版本库；其身份由本 manifest 的 chain_sha256 钉住",
        },
    }

    # (6) 原子落盘：先写 staging，全部成功后再逐个 replace
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        written: list[tuple[Path, Path]] = []
        for (arm, name), filename in _FILENAMES.items():
            rows = (h1 if arm == "h1" else mcq)[name]
            tmp = staging / filename
            _write_jsonl(tmp, rows)
            reread = [json.loads(line) for line in tmp.read_text(encoding="utf-8").splitlines()]
            if len(reread) != len(rows):
                raise ValueError(f"{filename} 回读行数不符：{len(reread)} != {len(rows)}")
            written.append((tmp, out_dir / filename))
        tmp_manifest = staging / MANIFEST_NAME
        tmp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        written.append((tmp_manifest, out_dir / MANIFEST_NAME))

        for tmp, final in written:
            tmp.replace(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return manifest
