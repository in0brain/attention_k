"""物化 Stage 1 的正式名单与 G3/G4 gate 名单（启动 gate G2 + G5）。

G2 不通过 → 立即停，不创建任何名单、不写任何文件。
全部名单先写临时目录，通过校验后原子落盘；失败不留半成品。

**不跑任何模型。** 本脚本只做数据身份层面的构建与互斥审计。

用法:
    python scripts/build_stage1_samples.py
    python scripts/build_stage1_samples.py --out-dir outputs/stage1/samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation.v3_freeze import require_v3_frozen  # noqa: E402
from recover_attention.evaluation.v3_sample_builder import (  # noqa: E402
    MANIFEST_NAME,
    build_stage1_samples,
    diagnose_g5_conflicts,
    load_h1_pools,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="只诊断 G5 冲突并写报告，**不构建任何名单**")
    parser.add_argument("--report-out", type=Path,
                        default=Path("outputs/logs/stage1_g5_conflict_report.json"))
    parser.add_argument("--force-report", action="store_true",
                        help="覆盖已存在的诊断报告（默认拒绝：既有报告是 v3.1 的修订依据）")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/stage1/samples"))
    parser.add_argument("--ontology-dir", type=Path, default=Path("data/raw/ontology"))
    parser.add_argument("--stage0-samples", type=Path,
                        default=Path("data/processed/h1/h1_samples.jsonl"))
    parser.add_argument("--cyber-dir", type=Path, default=Path("data/raw/cyber/cybermetric"))
    parser.add_argument("--capacity-audit", type=Path,
                        default=Path("outputs/logs/stage1_capacity_audit.json"))
    parser.add_argument("--prereg", type=Path, default=Path("docs/paper/preregistration_v3.md"))
    parser.add_argument("--lock", type=Path, default=Path("docs/paper/preregistration_v3.lock"))
    args = parser.parse_args()

    if args.report_only:
        # 既有报告是 v3.0 → v3.1 的修订依据，**不覆盖历史**（用户裁决，2026-08-08）。
        if args.report_out.exists() and not args.force_report:
            print(f"[refuse] {args.report_out} 已存在。它是 v3.1 的修订依据，不覆盖；"
                  f"要另存请给 --report-out，确需覆盖请显式加 --force-report。", file=sys.stderr)
            return 2
        # G2 仍是第一件事 —— 诊断也不得绕过冻结校验。
        freeze = require_v3_frozen(args.prereg, args.lock)
        pools = load_h1_pools(args.ontology_dir, args.stage0_samples)
        report = diagnose_g5_conflicts(pools)
        report["gates"] = {"G2": freeze.as_dict()}
        report["lists_written"] = False
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        cross = report["cross_stage_collisions"]
        intra = report["intra_stage_duplicate_prompts"]
        print(f"[G2    ] match={freeze.match}  hash_mode={freeze.hash_mode}")
        print(f"[pool  ] fresh entries={report['n_fresh_entries']}  "
              f"distinct prompts={report['n_distinct_prompts']}")
        print(f"[G5-x  ] 跨 Stage prompt 碰撞 {cross['count']} 条 → "
              f"{[d['fresh_entry'] for d in cross['detail']]}")
        print(f"[G5-i  ] Stage 1 内部重复 prompt 组 {intra['count']} 个（已由 group_id 封住）")
        print(f"[output] {args.report_out}   **未写任何名单**")
        return 1 if report["blocking"] else 0

    manifest = build_stage1_samples(
        out_dir=args.out_dir,
        ontology_dir=args.ontology_dir,
        stage0_samples=args.stage0_samples,
        cyber_dir=args.cyber_dir,
        capacity_audit=args.capacity_audit,
        prereg=args.prereg,
        lock=args.lock,
    )

    g2 = manifest["gates"]["G2"]
    counts = manifest["counts"]
    print(f"[G2    ] match={g2['match']}  hash_mode={g2['hash_mode']}  "
          f"version={g2['locked_version']}  sha={g2['current_sha256'][:12]}…")
    print(f"[G5    ] all disjointness checks passed")
    excl = manifest["h1_exclusions"]["accounting"]
    print(f"[H1    ] eligible={counts['h1_eligible_pool']} "
          f"- 跨Stage碰撞 {excl['cross_stage_prompt_collision']} "
          f"- 内部去重 {excl['intra_set_duplicate_prompt']} "
          f"→ primary={counts['h1']['primary']}")
    print(f"[H1    ] records={counts['h1']['primary']} = "
          f"prompts={counts['h1_primary_distinct_prompts']} = "
          f"groups={counts['h1_primary_distinct_groups']}   "
          f"G3={counts['h1']['G3']} G4={counts['h1']['G4']} "
          f"(burned pool={counts['h1_pool_burned']})")
    print(f"[MCQ   ] primary={counts['mcq']['primary']}  G3={counts['mcq']['G3']}  "
          f"G4={counts['mcq']['G4']}  (pool fresh={counts['mcq_pool_fresh']})")
    print(f"[traces] H1 primary={counts['h1_primary_traces']}  "
          f"MCQ primary={counts['mcq_primary_traces']}  (未生成，仅计数)")
    print(f"[output] {args.out_dir}/  manifest={MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
