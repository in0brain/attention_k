"""Stage 1 容量审计（**只读、身份级**）—— 为 preregistration v3 的 N/F/MCQ 填实提供事实。

本脚本**只**读取数据的身份、数量、重叠与哈希：
  ✔ ontology 条目数、Stage 0 已消耗的 source_entry_id 数、剩余可用池的计数与摘要哈希
  ✔ CyberMetric 各文件的记录数、归一化去重后的 distinct 数、文件间重叠数
  ✔ 冻结 salt 下的 E3 分块计数分布（只在 identifier 字符串上计算）
  ❌ **不读取任何标签、任何模型输出、任何 AUROC/Δ/CI 或性能量**
  ❌ **不构建正式样本**：MCQ 的 1760 题选择规则只写进产物，`selection_materialized=false`

边界（写进产物）：
  - 本审计**不得**用于任何结论表述；它只回答"有多少、是否互斥、哈希怎么算"。
  - E3 的分块分布**不具约束力**：F 与 salt 由预注册冻结，
    **不得**因本审计的分布结果改 F、换 salt 或重新分块（v3 §M2d）。

用法:
    python scripts/audit_stage1_capacity.py \
        --out outputs/logs/stage1_capacity_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.cyber_data import (  # noqa: E402
    _normalize_question_for_grouping,
    load_cybermetric_records,
)
from recover_attention.h1_identifier import normalize_identifier  # noqa: E402

SCHEMA_VERSION = "stage1_capacity_audit_v1"

# ── 预注册冻结常量（v3 §M2d / §M6）—— 本脚本不得改写，改动须先改预注册 ────────
E3_BLOCK_SALT = "recover_attention/v3/E3_vocab_block/2026-08-08"
E3_F = 5
MCQ_SELECT_SALT = "recover_attention/v3/mcq_selection/2026-08-08"
N_MCQ_TARGET = 1760
# N_H1 = **整个**未被 Stage 0 消耗的 attack+cwe 条目池（"全部"是定义，数字由审计算出）。
# 用户 2026-08-08 裁决时写的 1545 沿用了 v3 M6 的 141/141；实测消耗为 140/140
# （141 把 open_gen 的 "None" 哨兵计成了条目），故池为 718+829 = 1547。
N_H1_TARGET = 1547
K_TRACES = 6

PRIMARY_FAMILIES = ("attack", "cwe")   # CVE 不进 primary 标签空间（v3 M6，效度理由）

DEFAULT_ONTOLOGY_DIR = Path("data/raw/ontology")
DEFAULT_H1_SAMPLES = Path("data/processed/h1/h1_samples.jsonl")
DEFAULT_CYBER_DIR = Path("data/raw/cyber/cybermetric")
DEFAULT_PREREG = Path("docs/paper/preregistration_v3.md")
DEFAULT_OUT = Path("outputs/logs/stage1_capacity_audit.json")

_WS_RE = re.compile(r"\s+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chain_digest(items: list[str]) -> str:
    """对**已排序**的身份列表取链哈希 —— 记录池的身份，而不物化池本身。"""
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ── 冻结规则（文本在此，实现在此；与 src 的对应函数逐条比对，防止静默漂移）──────
def frozen_question_key(question: str) -> str:
    """v3 §M6 冻结：`re.sub(r"\\s+", " ", q).strip().casefold()`。"""
    return _WS_RE.sub(" ", question).strip().casefold()


def frozen_question_hash(question: str) -> str:
    """稳定题目 ID = sha256(归一化题面)，全 64 位 hex。"""
    return sha256_text(frozen_question_key(question))


def mcq_sort_key(question_hash: str) -> str:
    """MCQ 选择序 = sha256(salt || "|" || question_hash) 升序；并列时按 question_hash 升序。"""
    return sha256_text(f"{MCQ_SELECT_SALT}|{question_hash}")


def e3_block(normalized_identifier: str) -> int:
    """v3 §M2d 冻结：block = int(sha256(salt || "|" || normalized_id), 16) mod F。"""
    return int(sha256_text(f"{E3_BLOCK_SALT}|{normalized_identifier}"), 16) % E3_F


# ── H1 侧 ─────────────────────────────────────────────────────────────────────
def audit_h1(ontology_dir: Path, h1_samples: Path) -> dict:
    totals: dict[str, int] = {}
    pool_ids: dict[str, list[str]] = {}
    for family in PRIMARY_FAMILIES:
        index_path = ontology_dir / family / "ontology_index.jsonl"
        ids: list[str] = []
        with index_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                normalized = entry["normalized_id"]
                if entry["family"] != family:
                    raise ValueError(f"{index_path}:{line_no} family mismatch: {entry['family']!r}")
                if normalize_identifier(family, normalized) != normalized:
                    raise ValueError(
                        f"{index_path}:{line_no} id {normalized!r} 不是 normalize_identifier 的不动点"
                    )
                ids.append(normalized)
        if len(set(ids)) != len(ids):
            raise ValueError(f"{index_path} 含重复 normalized_id")
        totals[family] = len(ids)
        pool_ids[family] = ids

    consumed: dict[str, set[str]] = {family: set() for family in PRIMARY_FAMILIES}
    consumed_other: Counter = Counter()
    entryless_prompts: Counter = Counter()
    n_prompts = 0
    with h1_samples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_prompts += 1
            family = row["family"]
            entry_id = row["source_entry_id"]
            # open_gen 路由是 **topic-based**，不消耗 ontology 条目：其 source_entry_id
            # 在 Stage 0 产物里是字符串 "None"（不是 JSON null）。把它当条目计数
            # 会让每个 family 的"已用条目"虚增 1 —— v3 M6 的 141/141 正是这么来的。
            is_entryless = entry_id is None or entry_id == "None"
            if family not in consumed:
                consumed_other[family] += 1
            elif is_entryless:
                entryless_prompts[f"{family}:{row['route']}"] += 1
            else:
                consumed[family].add(entry_id)

    remaining: dict[str, list[str]] = {}
    for family in PRIMARY_FAMILIES:
        unknown = consumed[family] - set(pool_ids[family])
        if unknown:
            raise ValueError(f"Stage 0 用了不在 ontology index 里的 {family} 条目: {sorted(unknown)[:5]}")
        remaining[family] = sorted(set(pool_ids[family]) - consumed[family])

    pool_flat = sorted(f"{family}:{ident}" for family in PRIMARY_FAMILIES for ident in remaining[family])
    n_pool = len(pool_flat)

    block_counts = Counter(e3_block(item.split(":", 1)[1]) for item in pool_flat)
    max_share = max(block_counts.values()) / n_pool if n_pool else 0.0

    return {
        "ontology_totals": totals,
        "stage0_prompts": n_prompts,
        "stage0_consumed_entries": {f: len(consumed[f]) for f in PRIMARY_FAMILIES},
        "stage0_entryless_prompts": dict(entryless_prompts),
        "stage0_consumed_non_primary_prompts": dict(consumed_other),
        "prereg_v3_m6_stated_consumed": {"attack": 141, "cwe": 141},
        "prereg_v3_m6_correction": (
            "v3 M6 记 141/141，实测每个 family 只消耗 140 个真实条目；差 1 来自把 open_gen 的 "
            'source_entry_id 哨兵值 "None" 计成了一个条目。故上限由 1,545 更正为 1,547。'
        ),
        "cve_excluded_from_primary": True,
        "cve_exclusion_reason": "v3 M6：CVE-YYYY-NNNN 低序号段近乎稠密，存在性判据过弱（效度约束，非容量选择）",
        "stage1_eligible_pool": {f: len(remaining[f]) for f in PRIMARY_FAMILIES} | {"total": n_pool},
        "eligible_pool_chain_sha256": chain_digest(pool_flat),
        "pool_materialized": False,
        "N_H1_frozen": N_H1_TARGET,
        "K": K_TRACES,
        "planned_traces": N_H1_TARGET * K_TRACES,
        "capacity_ok": n_pool >= N_H1_TARGET,
        "capacity_headroom": n_pool - N_H1_TARGET,
        "e3_block_distribution_over_pool": {
            "F": E3_F,
            "salt": E3_BLOCK_SALT,
            "rule": 'block = int(sha256(salt + "|" + normalized_id).hexdigest(), 16) % F',
            "counts": {str(b): block_counts.get(b, 0) for b in range(E3_F)},
            "max_block_share": round(max_share, 6),
            "binding": False,
            "note": (
                "仅作记录。F 与 salt 由 v3 §M2d 冻结，**不得**因本分布改 F、换 salt 或重新分块；"
                "E3 的可估性由 §M2d 的 fold gate 在分析时判定，不由本分布判定。"
            ),
        },
    }


# ── MCQ 侧 ────────────────────────────────────────────────────────────────────
def _load_hashes(path: Path) -> tuple[list[str], int]:
    records = load_cybermetric_records(path)
    hashes = []
    for record in records:
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path} 含空题面")
        # 漂移检查：冻结规则必须与 src 的 grouping 归一化逐条一致
        if frozen_question_key(question) != _normalize_question_for_grouping(question):
            raise ValueError(f"{path} 的归一化规则与 cyber_data 不一致（冻结规则漂移）")
        hashes.append(frozen_question_hash(question))
    return hashes, len(records)


def audit_mcq(cyber_dir: Path) -> dict:
    files = {
        "CyberMetric-10000-v1.json": cyber_dir / "CyberMetric-10000-v1.json",
        "CyberMetric-2000-v1.json": cyber_dir / "CyberMetric-2000-v1.json",
        "CyberMetric-500-v1.json": cyber_dir / "CyberMetric-500-v1.json",
    }
    raw_counts: dict[str, int] = {}
    distinct: dict[str, set[str]] = {}
    for name, path in files.items():
        hashes, n_raw = _load_hashes(path)
        raw_counts[name] = n_raw
        distinct[name] = set(hashes)

    d10 = distinct["CyberMetric-10000-v1.json"]
    d2 = distinct["CyberMetric-2000-v1.json"]
    d5 = distinct["CyberMetric-500-v1.json"]

    fresh = d10 - d2
    n_fresh = len(fresh)
    overlap_2000 = len(d2 & d10)
    naive = raw_counts["CyberMetric-10000-v1.json"] - overlap_2000

    return {
        "file_sha256": {name: sha256_file(path) for name, path in files.items()},
        "raw_record_counts": raw_counts,
        "distinct_question_counts": {name: len(items) for name, items in distinct.items()},
        "intra_file_duplicates": {
            name: raw_counts[name] - len(distinct[name]) for name in files
        },
        "stage0_g4_exclusion_set": "distinct(CyberMetric-2000-v1.json)",
        "stage0_g4_exclusion_basis": (
            "Stage 0 的 MCQ 臂由 CyberMetric-2000-v1 制备（scripts/sprint_4B_1_prepare_cybermetric.py "
            "DEFAULT_INPUT，nominal 1760 题）；G4 跑在同一批正式 arm 上，故其题目 ⊆ 2000-v1。"
        ),
        "fresh_pool_size": n_fresh,
        "fresh_pool_accounting": {
            "distinct_10000": len(d10),
            "overlap_with_2000": overlap_2000,
            "distinct_10000_minus_overlap": len(d10) - overlap_2000,
            "authoritative": n_fresh,
            "naive_without_intra_file_dedup": naive,
            "discrepancy_explained": (
                f"raw({raw_counts['CyberMetric-10000-v1.json']}) − overlap({overlap_2000}) = {naive}，"
                f"但 10000-v1 内部有 "
                f"{raw_counts['CyberMetric-10000-v1.json'] - len(d10)} 组归一化重复；"
                f"去重后 {len(d10)} − {overlap_2000} = {n_fresh}。以 {n_fresh} 为准。"
            ),
            "questions_in_2000_absent_from_10000": len(d2 - d10),
        },
        "disclosed_secondary": {
            "fresh_also_present_in_500v1": len(fresh & d5),
            "pool_if_500v1_also_excluded": len(fresh - d5),
            "note": (
                "500-v1 未被 Stage 0/G4 使用（仅 sprint 4B 的探索性 smoke 读过），"
                "故按冻结规则**不**排除；此处披露其重叠量，便于外部读者自行收紧。"
            ),
        },
        "N_MCQ_frozen": N_MCQ_TARGET,
        "K": K_TRACES,
        "planned_traces": N_MCQ_TARGET * K_TRACES,
        "capacity_ok": n_fresh >= N_MCQ_TARGET,
        "capacity_headroom": n_fresh - N_MCQ_TARGET,
        "selection_rule": {
            "stable_question_id": 'question_hash = sha256(re.sub(r"\\s+"," ",q).strip().casefold())',
            "salt": MCQ_SELECT_SALT,
            "order": 'sha256(salt + "|" + question_hash) 升序；并列时按 question_hash 升序',
            "take": N_MCQ_TARGET,
            "g3_g4_note": "G3/G4 的样本另取，且必须排除在正式集之外",
        },
        "selection_materialized": False,
        "selection_materialized_note": "v3 冻结与 lock 完成前不构建正式样本（用户裁决，2026-08-08）",
    }


def build_report(*, ontology_dir: Path, h1_samples: Path, cyber_dir: Path, prereg: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_for": "preregistration v3 §M2d / §M6 / §7.5 的 F / N_H1 / N_MCQ 填实",
        "reads_labels_or_performance": False,
        "scope": "identity, counts, overlap, hashes only",
        "must_not_be_used_for": [
            "any reported result or claim",
            "any Stage 1 conclusion",
            "changing F or the frozen salts",
            "constructing the formal Stage 1 sample before freeze",
        ],
        "prereg_v3_sha256": sha256_file(prereg),
        "prereg_v3_sha256_note": "运行时的值；预注册随后填入 N/F 会改变它，**不得回写本字段**",
        "h1": audit_h1(ontology_dir, h1_samples),
        "mcq": audit_mcq(cyber_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-dir", type=Path, default=DEFAULT_ONTOLOGY_DIR)
    parser.add_argument("--h1-samples", type=Path, default=DEFAULT_H1_SAMPLES)
    parser.add_argument("--cyber-dir", type=Path, default=DEFAULT_CYBER_DIR)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = build_report(
        ontology_dir=args.ontology_dir,
        h1_samples=args.h1_samples,
        cyber_dir=args.cyber_dir,
        prereg=args.prereg,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    h1, mcq = report["h1"], report["mcq"]
    print(f"[input ] ontology={args.ontology_dir}  h1={args.h1_samples}  cyber={args.cyber_dir}")
    print(f"[H1    ] ontology {h1['ontology_totals']}  Stage0 用 {h1['stage0_consumed_entries']}")
    print(f"[H1    ] 可用池 {h1['stage1_eligible_pool']}  → N_H1={h1['N_H1_frozen']} "
          f"K={h1['K']} traces={h1['planned_traces']}  headroom={h1['capacity_headroom']}")
    print(f"[E3    ] F={E3_F} 分块计数 {h1['e3_block_distribution_over_pool']['counts']} "
          f"max_share={h1['e3_block_distribution_over_pool']['max_block_share']}")
    print(f"[MCQ   ] raw {mcq['raw_record_counts']}")
    print(f"[MCQ   ] fresh_pool={mcq['fresh_pool_size']} "
          f"(naive={mcq['fresh_pool_accounting']['naive_without_intra_file_dedup']}) "
          f"→ N_MCQ={mcq['N_MCQ_frozen']} headroom={mcq['capacity_headroom']}")
    print(f"[output] {args.out}")
    if not (h1["capacity_ok"] and mcq["capacity_ok"]):
        print("[FAIL  ] 容量不足", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
