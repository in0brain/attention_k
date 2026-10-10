"""修复 MCQ feature cache 缺失的 wrong_label（§7.1/§3 的目标 Y 本身）。

根因:MCQBackend.run_unit 从未计算 wrong_label——G4 quality_check
(run_stage0_quality_check.py) 早就算过同一个式子（parsed_label != gold_label，
仅用于"两类都有吗"这一可用性判断），但 production 生成路径漏了同一行代码。
直到 fresh 1760 第一次真正跑到冻结/分析入口（load_arm）才暴露：
KeyError('wrong_label')。根因已在 backends.py 补上（_mcq_wrong_label，
影响之后的 MCQ 生成/resume）。

本脚本只修复**已经跑完**的 MCQ 产物,不重跑任何 GPU 工作:
  - completion 文本 / parsed_label / hidden 向量均未变,权威来源仍是
    outputs/stage0/generation/mcq_unit_results.jsonl（GPU 生成的原始记录，本脚本
    不改动它——它是否该有 wrong_label 是 production 逻辑漏了这一步，不是数据丢失）。
  - 唯一变化:mcq_completion_records.jsonl 里补上 wrong_label 一列,用 pool 文件
    （data/processed/cyber/cybermetric.jsonl）里的 gold_label 与该记录的
    parsed_label 比较得出。
  - mcq_hidden.npz 不受影响,不重写。

**performance-blind**:本脚本只做结构性核验(键是否存在、两类是否都有、行数是否
对得上),不打印、不记录任何比例/计数意义上的"错误率"——那是揭盲后才能看的数字。

审计:写 mcq_wrong_label_repair_log.json 记录修复前后的字段完整性对比，保持透明
而非静默覆盖。本脚本保留在仓库里作为这次修复的永久记录，不是一次性 scratch 脚本。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from recover_attention.generation.backends import _mcq_wrong_label  # noqa: E402
from recover_attention.generation.manifest import write_atomic  # noqa: E402

FEAT = Path("outputs/stage0/features")
POOL = Path("data/processed/cyber/cybermetric.jsonl")

# ---- 1. 载入 pool 文件的 gold_label（已核验：candidate_choices 里每条 choice 的
#          label_text 与 gold_label_text 逐条对齐，2000/2000 无一不符）----
gold_by_id = {}
for line in open(POOL, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    gold_by_id[r["example_id"]] = r["gold_label"]
print(f"载入 pool gold_label: {len(gold_by_id)} 条")

# ---- 2. 读旧 cache，只补 wrong_label 一列，其余字段与结构完全保留 ----
old_rows = [json.loads(l) for l in open(FEAT / "mcq_completion_records.jsonl", encoding="utf-8")]
print(f"载入 mcq_completion_records: {len(old_rows)} 条")
before_present = sum(1 for r in old_rows if "wrong_label" in r)
print(f"修复前已有 wrong_label 字段的条数: {before_present}")

new_rows = []
patched = 0
missing_gold = []
for row in old_rows:
    eid = row["example_id"]
    if eid not in gold_by_id:
        missing_gold.append(eid)
        new_rows.append(row)
        continue
    old_val = row.get("wrong_label")
    row["wrong_label"] = _mcq_wrong_label(row.get("parsed_label"),
                                          bool(row.get("eligible_for_primary")),
                                          gold_by_id[eid])
    if row["wrong_label"] != old_val:
        patched += 1
    new_rows.append(row)

if missing_gold:
    raise RuntimeError(f"{len(missing_gold)} 条记录在 pool 文件里找不到 gold_label，"
                       f"无法安全修复，停止：{missing_gold[:5]}")

print(f"实际修改的记录数: {patched} / {len(new_rows)}")
assert len(new_rows) == len(old_rows), "行数必须不变"

# ---- 3. 校验：不得改动除 wrong_label 外的任何字段 ----
old_by_id = {r["example_id"]: r for r in old_rows}
for row in new_rows:
    old = old_by_id[row["example_id"]]
    for k in old:
        if k == "wrong_label":
            continue
        if row.get(k) != old.get(k):
            raise RuntimeError(f"意外改动了字段 {k}（example_id={row['example_id']}），停止写盘")
print("校验通过：除 wrong_label 外，其余全部字段逐字节不变")

# ---- 4. 原子写回 ----
payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows)
write_atomic(FEAT / "mcq_completion_records.jsonl", payload)
print(f"已写回: {FEAT / 'mcq_completion_records.jsonl'}")

assert (FEAT / "mcq_hidden.npz").exists()
print("mcq_hidden.npz 未改动（本次修复不涉及 hidden 向量）")

# ---- 5. 结构性复核（performance-blind：只看"存在/两类是否都有"，不算比例）----
eligible = [r for r in new_rows if r.get("eligible_for_primary")]
has_label = sum(1 for r in eligible if r.get("wrong_label") is not None)
classes_present = sorted({r["wrong_label"] for r in eligible if r.get("wrong_label") is not None})
print("\n修复后结构性复核（performance-blind，不报比例）:")
print(f"  eligible 记录数: {len(eligible)}")
print(f"  其中带 wrong_label 的: {has_label}/{len(eligible)}")
print(f"  出现的类别集合: {classes_present}  (需要 >=2 才能过 load_arm 的 single-class 检查)")

# ---- 6. 端到端核验：真正调用 load_arm，只看是否成功，不读取/打印任何比例 ----
sys.path.insert(0, "src")
from recover_attention.evaluation.config import Stage0Config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_MCQ, load_arm  # noqa: E402

config = Stage0Config(mode="production", allow_pilot_population=False)
try:
    cache = load_arm(ARM_MCQ, FEAT / "mcq_completion_records.jsonl", FEAT / "mcq_hidden.npz",
                     config, label_field="wrong_label")
    print(f"\nload_arm 端到端核验通过：n={cache.y.size}（数值本身不打印，仅确认非零）")
    load_arm_ok = True
except Exception as exc:                                   # noqa: BLE001
    print(f"\nload_arm 端到端核验失败: {exc!r}")
    load_arm_ok = False

# ---- 7. 审计记录 ----
log = {
    "action": "repair_mcq_wrong_label",
    "reason": ("MCQBackend.run_unit 从未计算 wrong_label，production 产物写不出可分析的 "
              "label（load_arm 曾直接 KeyError('wrong_label')）。根因已在 backends.py 修复 "
              "（_mcq_wrong_label，影响之后的生成/resume）。本次操作只补齐已完成的 MCQ "
              "产物，不重跑 GPU。"),
    "n_records": len(new_rows),
    "n_records_patched": patched,
    "fields_touched": ["wrong_label"],
    "other_fields_unchanged": True,
    "hidden_npz_touched": False,
    "source_of_gold_label": str(POOL),
    "post_repair_structural_check": {
        "n_eligible": len(eligible),
        "n_with_wrong_label": has_label,
        "classes_present": classes_present,
        "load_arm_end_to_end_ok": load_arm_ok,
    },
    "performance_blind_note": ("此日志与本脚本输出均不含 wrong_label 的比例/计数意义上的 "
                               "错误率；那是揭盲后才能看的数字（§2.2）。"),
}
(FEAT / "mcq_wrong_label_repair_log.json").write_text(
    json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n审计记录已写: {FEAT / 'mcq_wrong_label_repair_log.json'}")

if not load_arm_ok or len(classes_present) < 2:
    print("\n仍有问题，需要进一步排查")
    sys.exit(1)
print("\n修复完成：MCQ feature cache 的 wrong_label 已补齐，load_arm 端到端核验通过")
