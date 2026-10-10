r"""解析修复的**逐条影响清单**与**最小重提取计划**。

生成结果（completion）可以保留，但**派生特征不能盲目复用**。本脚本对每条受影响的
记录判定它属于哪一类，并据此列出最小重提取范围。

只读落盘产物，**不加载模型权重、不重新生成、不改写任何既有产物**。

四类（按卡）：
    A 标签与定位均不变      → 原特征可复用
    B 定位改变 / 新增可解析 → 需重新提取该行的 hidden 与 F5
    C 标签改变              → 在 B 之上还要重算标签、选项特征，
                              以及**同题 K 条 trace 的聚合量**
    D 从成功变失败          → 按原排除规则处理（不进 population，也就没有 H）
另外：任一 question 只要有一条进了 B/C/D，该 question 的 **K 条 trace 聚合量**
（f5_self_consistency_exact / f5_letter_agreement_rate）就全部需要重算。

用法：
    python scripts/diagnostics/mcq_parser_fix_impact.py \
        --replay tmp/stage1_l2_g3/mcq_parser_fix_replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "tmp/stage1_l2_g3/mcq_parser_fix_impact.json"

#: 只有这两份属于**当前证据链**（H1=384 / MCQ=512 的已登记设置）。
#: 其余 Llama 目录是修复前 / 探索性产物，保留原样，不进重提取计划。
CURRENT_CHAIN = {
    "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl": "G3(512)",
    "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl": "G4(512)",
}
#: 原始生成 token ids 是否随该产物落盘 —— 决定能不能离线重定位答案字母 token。
RAW_IDS_SAVED = {
    "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl": False,
    "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl": False,
}


def _classify(old: dict, new: dict) -> str:
    if new["label"] is None and old["label"] is not None:
        return "D_成功变失败"
    if old["label"] is None and new["label"] is not None:
        return "B_新增可解析答案"
    if old["label"] != new["label"]:
        return "C_标签改变"
    if old["span"] != new["span"]:
        return "B_定位改变"
    return "A_标签与定位均不变"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=Path,
                    default=ROOT / "tmp/stage1_l2_g3/mcq_parser_fix_replay.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reparse-dir", type=Path,
                    default=ROOT / "outputs/stage1_llama_v3/mcq_reparse_v2",
                    help="新解析结果的落盘目录（与旧解析产物分开保存，不覆盖）")
    args = ap.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay["qwen"]["n_changed"]:
        print("[stop ] 重放显示 Qwen 记录发生变化 —— 先处理那件事，不出重提取计划",
              file=sys.stderr)
        return 2

    # 重放里 diffs 只留了前 N 条摘要，这里按 unit_id 重新对齐全量
    changed_ids = set(replay["all_diff_unit_ids"])
    buckets: dict[str, list] = defaultdict(list)
    per_file = Counter()
    affected_questions: dict[str, set] = defaultdict(set)
    rows_by_file: dict[str, list] = {}

    for rel, tag in CURRENT_CHAIN.items():
        rows = [json.loads(x) for x in io.open(ROOT / rel, encoding="utf-8") if x.strip()]
        rows_by_file[rel] = rows
        for d in replay["diffs"]:
            pass  # 详情从下面的重算得到，diffs 只作交叉核对

    # 用当前解析器 + 冻结的旧副本重算（与 replay 同源，此处只做归类）
    import importlib.util
    spec = importlib.util.spec_from_file_location("_dlp_old", replay["old_module"]["path"])
    old_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_mod)
    from recover_attention import domain_label_proxy as dlp
    from recover_attention import mcq_conditional_increment as mci
    letters = list(mci.MCQ_LETTERS)

    def old_span(completion, label):
        if label is None:
            return None
        _, valid = old_mod._valid_label_set(letters)
        ex = [m for m in old_mod._EXPLICIT_ANSWER_RE.finditer(completion)
              if m.group(1).upper() in valid]
        if ex:
            return list(ex[-1].span(1))
        tr = old_mod._trailing_isolated_matches(completion, valid)
        return list(tr[-1].span(1)) if tr else None

    for rel, tag in CURRENT_CHAIN.items():
        for row in rows_by_file[rel]:
            completion = row.get("raw_completion")
            if not isinstance(completion, str):
                continue
            o = old_mod.parse_option_answer(completion, letters)
            w = dlp.parse_option_answer(completion, letters)
            old = {"label": o["parsed_label"], "method": o["parse_method"],
                   "span": old_span(completion, o["parsed_label"])}
            new = {"label": w["parsed_label"], "method": w["parse_method"],
                   "span": w["answer_char_span"]}
            kind = _classify(old, new)
            if kind == "A_标签与定位均不变":
                continue
            per_file[(tag, kind)] += 1
            affected_questions[tag].add(row["example_id"])
            buckets[kind].append({
                "arm_file": rel, "tag": tag,
                "unit_id": row.get("unit_id"), "example_id": row.get("example_id"),
                "old_label": old["label"], "new_label": new["label"],
                "old_span": old["span"], "new_span": new["span"],
                "old_method": old["method"], "new_method": new["method"],
                "new_failure_reason": w.get("parse_failure_reason"),
                "raw_generated_ids_saved": RAW_IDS_SAVED[rel],
                "needs": ("重新 teacher-forced 前向以取该行的 hidden 与 F5"
                          if kind.startswith(("B", "C")) else
                          "按原排除规则处理：不进 population，删除该行的 hidden 引用"),
            })

    # ── 新解析结果落盘：与旧解析产物**分开保存**，逐条可追溯 ──────────────
    args.reparse_dir.mkdir(parents=True, exist_ok=True)
    parser_sha = hashlib.sha256(
        (ROOT / "src/recover_attention/domain_label_proxy.py").read_bytes()).hexdigest()
    reparse_files = {}
    for rel, tag in CURRENT_CHAIN.items():
        lines = []
        for row in rows_by_file[rel]:
            completion = row.get("raw_completion")
            if not isinstance(completion, str):
                continue
            o = old_mod.parse_option_answer(completion, letters)
            w = dlp.parse_option_answer(completion, letters)
            old = {"label": o["parsed_label"], "method": o["parse_method"],
                   "span": old_span(completion, o["parsed_label"])}
            new = {"label": w["parsed_label"], "method": w["parse_method"],
                   "span": w["answer_char_span"]}
            lines.append(json.dumps({
                "schema": "stage1_mcq_reparse_v2_row",
                "source": rel, "unit_id": row.get("unit_id"),
                "example_id": row.get("example_id"),
                "sample_type": row.get("sample_type"),
                "sample_index": row.get("sample_index"),
                "parsed_label": new["label"], "parse_method": new["method"],
                "answer_char_span": new["span"],
                "parse_failure_reason": w.get("parse_failure_reason"),
                "previous": old,
                "impact_class": _classify(old, new),
                "features_reusable": _classify(old, new) == "A_标签与定位均不变",
            }, ensure_ascii=False))
        name = f"{tag.replace('(', '_').replace(')', '')}_mcq_reparse.jsonl".lower()
        path = args.reparse_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reparse_files[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "n_rows": len(lines), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"[新解析] {path}  {len(lines)} 行")
    (args.reparse_dir / "mcq_reparse_v2_meta.json").write_text(json.dumps({
        "schema": "stage1_mcq_reparse_v2_meta",
        "date": "2026-09-15",
        "what": ("解析修复后的**新解析结果**，与旧解析产物分开保存。"
                 "**这不是新的 gate 报告** —— 受影响行的 hidden / F5 尚未重新提取，"
                 "不得据此更新任何 passed 标记。"),
        "parser_sha256": parser_sha,
        "old_parser_sha256": replay["old_module"]["sha256"],
        "files": reparse_files,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 同题 K 条 trace 的聚合量：只要该题有一条进了 B/C/D，整题 K 条都要重算
    k_trace_recompute = {}
    for rel, tag in CURRENT_CHAIN.items():
        qs = affected_questions[tag]
        n_rows = sum(1 for r in rows_by_file[rel] if r.get("example_id") in qs)
        k_trace_recompute[tag] = {
            "n_questions_affected": len(qs),
            "n_rows_in_those_questions": n_rows,
            "fields": ["f5_self_consistency_exact", "f5_letter_agreement_rate"],
            "why": ("attach_cross_sample_f5 按 example_id 聚合 parsed_label —— "
                    "同题任一条的标签或 eligibility 变了，这两维对该题**全部** K 条都变"),
            "example_ids": sorted(qs),
        }

    n_b = sum(len(v) for k, v in buckets.items() if k.startswith("B"))
    n_c = len(buckets.get("C_标签改变", []))
    n_d = len(buckets.get("D_成功变失败", []))

    doc = {
        "schema": "stage1_mcq_parser_fix_impact_v1",
        "date": "2026-09-15",
        "scope": ("**只覆盖当前证据链**（H1=384 / MCQ=512 的已登记设置）："
                  "g3_budget512 与 g4_budget512 的 MCQ 臂。"
                  "修复前的 g3/、16 预算的 g3_postfix/、探索性的 mcq_length_diagnosis/ "
                  "保留原样，不进重提取计划。"),
        "replay": {"path": str(args.replay.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                   "qwen_unchanged": replay["qwen"]["n_changed"] == 0},
        "counts_by_file_and_kind": {f"{t}|{k}": v for (t, k), v in sorted(per_file.items())},
        "buckets": {k: v for k, v in buckets.items()},
        "n_needs_refeature": n_b + n_c,
        "n_label_changed": n_c,
        "n_success_to_failure": n_d,
        "k_trace_recompute": k_trace_recompute,
        "raw_generated_ids": {
            "saved_in_these_artifacts": RAW_IDS_SAVED,
            "limitation": (
                "g3_budget512 / g4_budget512 的行**没有**落盘原始生成 token ids"
                "（只有 num_generated_tokens、letter_token_id、letter_token_index_local）。"
                "因此**不能**在离线重放里重新定位答案字母的 token 下标 —— "
                "把 completion 文本重新编码得到的序列**不是**原始生成序列，"
                "不得冒充它。需要 hidden / F5 的行必须重新做一次 teacher-forced 前向；"
                "本轮不加载权重，故只出计划、不执行。"),
        },
        "reuse_rule": (
            "只改 parsed_label 或 wrong_label 而沿用旧位置的特征是**不允许**的："
            "B4 的 F5 与 H 都读在答案字母 token 上，位置变了它们就不再对应同一个 token。"),
        "reparse_artifacts": reparse_files,
        "not_done_this_round": [
            "未重新生成任何 completion",
            "未加载模型权重、未做 teacher-forced 前向",
            "未改写任何既有产物；旧报告保持原样",
            "未更新任何 gate 报告的 passed 标记",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[范围] 当前证据链：{', '.join(CURRENT_CHAIN.values())}")
    for key, v in sorted(per_file.items()):
        print(f"  {key[0]:9s} {key[1]:20s} {v}")
    print(f"[需重提取特征] {n_b + n_c} 条（其中标签改变 {n_c} 条，成功变失败 {n_d} 条）")
    for tag, v in k_trace_recompute.items():
        print(f"[K 条聚合] {tag}: {v['n_questions_affected']} 题 / "
              f"{v['n_rows_in_those_questions']} 条 trace 需重算两维")
    print(f"[限制] 原始生成 token ids 未落盘 → 必须重新前向，不能用重编码冒充")
    print(f"[out  ] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
