"""修复 H1 feature cache 缺失的跨样本 F5（self-consistency / agreement）。

根因:_build_feature_cache 从未调用 attach_cross_sample_f5,该函数存在且文档写明用途,
但生成脚本漏了这行调用。已在 run_stage0_generation.py 里补上（影响 MCQ 后续生成）。

本脚本只修复**已经跑完**的 H1 产物,不重跑任何 GPU 工作:
  - completion 文本 / mentions / hidden 向量均未变,权威来源仍是
    outputs/stage0/generation/h1_unit_results.jsonl（GPU 生成的原始记录）与
    outputs/stage0/generation/h1_unit_hidden/*.npy（未改动）。
  - 唯一变化:h1_completion_records.jsonl 里补齐 f5_self_consistency_exact /
    f5_id_agreement_rate 两列，用同一 example_id 的全部 K=6 条 unit 聚合而来。
  - hidden.npz 不受影响,不重写。

审计:写 h1_repair_log.json 记录修复前后的字段完整性对比，保持透明而非静默覆盖。
本脚本保留在仓库里作为这次修复的永久记录，不是一次性 scratch 脚本。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.generation.backends import attach_cross_sample_f5  # noqa: E402

GEN = Path("outputs/stage0/generation")
FEAT = Path("outputs/stage0/features")

# ---- 1. 载入权威原始记录（GPU 生成的，未受影响）----
results = [json.loads(l) for l in open(GEN / "h1_unit_results.jsonl", encoding="utf-8")
           if l.strip()]
print(f"载入 unit_results: {len(results)} 条")
before_missing = sum(1 for r in results if "f5_self_consistency_exact" not in r)
print(f"修复前缺失 f5_self_consistency_exact 的条数: {before_missing}")

# ---- 2. 跨样本聚合（原地修改）----
attach_cross_sample_f5(results, "h1")

after_missing = sum(1 for r in results if r.get("f5_self_consistency_exact") is None)
print(f"修复后 f5_self_consistency_exact 为 None 的条数: {after_missing}")

# ---- 3. 读旧 cache，只替换这两列，其余字段与结构完全保留 ----
old_rows = [json.loads(l) for l in open(FEAT / "h1_completion_records.jsonl", encoding="utf-8")]
by_trace = {r["unit_id"]: r for r in results}
new_rows = []
patched = 0
for row in old_rows:
    tid = row["trace_id"]
    src = by_trace.get(tid)
    if src is None:
        raise RuntimeError(f"unit_results 中找不到 {tid}，无法安全修复，停止")
    old_cons = row.get("f5_self_consistency_exact")
    old_agree = row.get("f5_id_agreement_rate")
    row["f5_self_consistency_exact"] = src.get("f5_self_consistency_exact")
    row["f5_id_agreement_rate"] = src.get("f5_id_agreement_rate")
    if row["f5_self_consistency_exact"] != old_cons or row["f5_id_agreement_rate"] != old_agree:
        patched += 1
    new_rows.append(row)

print(f"实际修改的记录数: {patched} / {len(new_rows)}")
assert len(new_rows) == len(old_rows) == 2880, "行数必须不变"

# ---- 4. 校验：不得改动除这两列外的任何字段 ----
old_by_trace = {r["trace_id"]: r for r in old_rows}
for row in new_rows:
    old = old_by_trace[row["trace_id"]]
    for k in old:
        if k in ("f5_self_consistency_exact", "f5_id_agreement_rate"):
            continue
        if row.get(k) != old.get(k):
            raise RuntimeError(f"意外改动了字段 {k}（trace={row['trace_id']}），停止写盘")
print("校验通过：除两个目标字段外，其余全部字段逐字节不变")

# ---- 5. 原子写回（先写临时文件再 rename）----
from recover_attention.generation.manifest import write_atomic

payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows)
write_atomic(FEAT / "h1_completion_records.jsonl", payload)
print(f"已写回: {FEAT / 'h1_completion_records.jsonl'}")

# hidden.npz 完全不动
assert (FEAT / "h1_hidden.npz").exists()
print("h1_hidden.npz 未改动（本次修复不涉及 hidden 向量）")

# ---- 6. F5 完整性复核 ----
eligible = [r for r in new_rows if r.get("eligible")]
mat = ci._dense_matrix(eligible, ci.F5_FEATURE_NAMES)
nz = {n: int((mat[:, j] != 0).sum()) for j, n in enumerate(ci.F5_FEATURE_NAMES)}
distinct = {n: int(len(np.unique(mat[:, j]))) for j, n in enumerate(ci.F5_FEATURE_NAMES)}
cont = [n for n in ci.F5_FEATURE_NAMES if not n.startswith("f5_confidence_")]
empty = [n for n in cont if nz[n] == 0]
constant = [n for n in cont if distinct[n] <= 1]
print("\n修复后 F5 完整性:")
for n in ("f5_self_consistency_exact", "f5_id_agreement_rate"):
    print(f"  {n:32s} 非零 {nz[n]:5d}/{len(eligible)}  去重值 {distinct[n]:5d}")
print(f"  仍为空的连续列: {empty}")
print(f"  仍为常数的连续列: {constant}")

# ---- 7. 审计记录 ----
log = {
    "action": "repair_h1_cross_sample_f5",
    "reason": ("_build_feature_cache 从未调用 attach_cross_sample_f5，f5_self_consistency_exact "
              "与 f5_id_agreement_rate 两列在生成产物中全部缺失/为默认值。根因已在 "
              "run_stage0_generation.py 修复，本次操作只补齐已完成的 H1 产物，不重跑 GPU。"),
    "n_records": len(new_rows),
    "n_records_patched": patched,
    "fields_touched": ["f5_self_consistency_exact", "f5_id_agreement_rate"],
    "other_fields_unchanged": True,
    "hidden_npz_touched": False,
    "source_of_truth": str(GEN / "h1_unit_results.jsonl"),
    "post_repair_f5_check": {
        "empty_continuous_columns": empty,
        "constant_continuous_columns": constant,
    },
}
(FEAT / "h1_repair_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
print(f"\n审计记录已写: {FEAT / 'h1_repair_log.json'}")

if empty or constant:
    print("\n仍有 F5 列不合格，需要进一步排查")
    sys.exit(1)
print("\n修复完成：H1 feature cache 的 F5 完整性检查全部通过")
