r"""构造**生成长度检查**用的校准题单：冻结排序 ranked_fresh[1820:1840]。

这 20 题是 MCQ 冻结排序里紧接 G4（1780:1820）之后的一段，**不属于**正式集
（0:1760）、G3（1760:1780）、G4（1780:1820），也不在 Stage 0 已用池内
（fresh 池在构造时就已剔除 Stage 0 的 question_hash）。

本脚本**只做身份层的构造与核对，不跑模型**。任一核对失败即退出非零，
**不自行换题**。

核对项：
  1. 排序来源：重算 ranked_fresh，与三份已冻结名单的 question_hash 序列逐条比对
     —— 证明本脚本用的排序与当初产出名单的排序是同一个
  2. 唯一性：20 条 question_hash / rank / original_index 各自无重复
  3. 零交集：与 mcq_primary / mcq_G3 / mcq_G4 三份名单，以及 Stage 0 池，两两零交集
  4. 名单与哈希落盘

用法：
    python scripts/diagnostics/build_budget_check_questions.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_sample_builder as sb      # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig   # noqa: E402
from recover_attention.evaluation.v3_freeze import require_v3_frozen  # noqa: E402
from recover_attention.generation.manifest import write_atomic        # noqa: E402

SAMPLES = ROOT / "outputs/stage1/samples"
CYBER = ROOT / "data/raw/cyber/cybermetric"
OUT_DIR = ROOT / "outputs/stage1_llama_v3/mcq_budget512_check"

#: 校准段：**紧接 G4 之后**的 20 条。数字由 v3_config 的冻结计数推出，不手写。
SLICE_START_RULE = "n_mcq_questions + n_g3_per_arm + n_g4_per_arm"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    freeze = require_v3_frozen(ROOT / "docs/paper/preregistration_v3.md",
                               ROOT / "docs/paper/preregistration_v3.lock")
    print(f"[G2   ] match={freeze.match} sha={freeze.current_sha256[:12]}…")

    start = cfg.n_mcq_questions + cfg.n_g3_per_arm + cfg.n_g4_per_arm
    stop = start + cfg.n_g3_per_arm
    assert (start, stop) == (1820, 1840), (start, stop)

    pools = sb.load_mcq_pools(CYBER, cfg)
    ranked = pools.ranked_fresh
    print(f"[pool ] ranked_fresh={len(ranked)}  slice=[{start}:{stop}]")

    failures: list[str] = []
    n_checks = 0

    def check(label: str, ok: bool, detail=None) -> None:
        nonlocal n_checks
        n_checks += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}"
              + (f"  {detail}" if detail is not None else ""))
        if not ok:
            failures.append(label)

    # ── 1. 排序来源：重算的排序必须复现三份冻结名单 ─────────────────────
    frozen = {
        "mcq_primary": (_read_jsonl(SAMPLES / "mcq_primary_questions.jsonl"),
                        0, cfg.n_mcq_questions),
        "mcq_G3": (_read_jsonl(SAMPLES / "mcq_g3_questions.jsonl"),
                   cfg.n_mcq_questions, cfg.n_mcq_questions + cfg.n_g3_per_arm),
        "mcq_G4": (_read_jsonl(SAMPLES / "mcq_g4_questions.jsonl"),
                   cfg.n_mcq_questions + cfg.n_g3_per_arm, start),
    }
    print("[排序来源] 重算 ranked_fresh 与冻结名单逐条比对")
    for name, (rows, a, b) in frozen.items():
        want = [r["question_hash"] for r in rows]
        got = [r["question_hash"] for r in ranked[a:b]]
        check(f"{name}: ranked_fresh[{a}:{b}] 与落盘名单逐条相同",
              want == got, f"n={len(want)}")
        check(f"{name}: 落盘 rank 字段与切片位置一致",
              [r.get("rank") for r in rows] == list(range(a, b)))

    sel = [dict(r) for r in ranked[start:stop]]

    # ── 2. 唯一性 ───────────────────────────────────────────────────────
    print("[唯一性]")
    for key in ("question_hash", "rank", "original_index"):
        vals = [r[key] for r in sel]
        check(f"{key} 无重复", len(set(vals)) == len(vals), f"n={len(set(vals))}")
    check("条数 == 20", len(sel) == cfg.n_g3_per_arm, len(sel))
    check("rank 连续 1820..1839", [r["rank"] for r in sel] == list(range(start, stop)))

    # ── 3. 零交集 ───────────────────────────────────────────────────────
    print("[零交集]")
    mine = {r["question_hash"] for r in sel}
    for name, (rows, _a, _b) in frozen.items():
        other = {r["question_hash"] for r in rows}
        inter = mine & other
        check(f"与 {name} 零交集", not inter, f"|∩|={len(inter)}")
    inter0 = mine & set(pools.stage0_hashes)
    check("与 Stage 0 已用池零交集", not inter0, f"|∩|={len(inter0)}")
    # 结构性保证：fresh 池本身就已剔除 Stage 0，故上一条恒真；仍然显式核一次。
    check("三份冻结名单 + 校准段 合计 1840 条互不相同",
          len({r["question_hash"] for rows, _a, _b in frozen.values() for r in rows} | mine)
          == stop, stop)

    if failures:
        print(f"\n[STOP ] {len(failures)} 项核对失败：{failures}", file=sys.stderr)
        print("        按卡要求停止，不自行换题。", file=sys.stderr)
        return 2

    # ── 4. 落盘 ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for r in sel:
        rows_out.append({
            "question_hash": r["question_hash"], "rank": r["rank"],
            "source": r["source"], "original_index": r["original_index"],
            "question": r["question"], "answers": r["answers"],
            "gold_original_label": r["gold_original_label"],
            # 与 gate 名单同口径的标记：这批题**只用于生成长度检查**，
            # 不是 gate 样本，也不得进入正式集。
            "budget_check_only": True,
            "gate": None, "gate_only": False, "previously_burned": False,
        })
    list_path = OUT_DIR / "mcq_budget_check_questions.jsonl"
    write_atomic(list_path, "\n".join(json.dumps(r, ensure_ascii=False)
                                      for r in rows_out) + "\n")

    chain = sb.chain_digest(r["question_hash"] for r in rows_out)
    meta = {
        "schema": "stage1_mcq_budget_check_questions_v1",
        "date": "2026-09-14",
        "purpose": ("生成长度检查（本次候选 max_new_tokens=512）的校准题单。"
                    "**不是 gate 样本，不进正式集，不得用于任何性能量。**"),
        "selection_rule": {
            "ordering": ("v3_sample_builder.load_mcq_pools —— fresh 池按 "
                         "(salted_rank_key(mcq_selection_salt, question_hash), "
                         "question_hash) 升序；与正式集/G3/G4 同一个排序"),
            "salt": cfg.mcq_selection_salt,
            "slice": f"ranked_fresh[{start}:{stop}]",
            "slice_start_rule": SLICE_START_RULE,
            "rationale": "紧接 G4 之后的一段；先到先取，不按任何观察到的性质挑选",
        },
        "verification": {
            "ordering_reproduces_frozen_lists": True,
            "unique_question_hash": True,
            "disjoint_from": ["mcq_primary", "mcq_G3", "mcq_G4", "stage0_pool"],
            "n_checks_passed": n_checks,
        },
        "prereg": freeze.as_dict(),
        "source_files": {
            "data/raw/cyber/cybermetric/CyberMetric-10000-v1.json":
                _sha256_file(CYBER / "CyberMetric-10000-v1.json"),
            "data/raw/cyber/cybermetric/CyberMetric-2000-v1.json":
                _sha256_file(CYBER / "CyberMetric-2000-v1.json"),
        },
        "n_questions": len(rows_out),
        "chain_sha256": chain,
        "list_path": str(list_path.relative_to(ROOT)).replace("\\", "/"),
        "list_sha256": _sha256_file(list_path),
    }
    meta_path = OUT_DIR / "mcq_budget_check_questions_meta.json"
    write_atomic(meta_path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"\n[ok   ] {n_checks}/{n_checks} 核对通过")
    print(f"[chain] {chain}")
    print(f"[sha  ] list={meta['list_sha256'][:16]}…")
    print(f"[out  ] {list_path}")
    print(f"[out  ] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
