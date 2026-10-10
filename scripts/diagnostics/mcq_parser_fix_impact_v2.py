r"""收紧后规则的**逐条影响清单**与**最小特征重提取计划**（重新计算，不沿用旧数字）。

只读落盘产物，**不加载模型权重、不重新生成、不改写任何既有产物**。

四类（按卡）：
    A 标签与定位均不变      → 原特征可复用
    B 定位改变 / 新增可解析 → 需重新提取该行 hidden 与 F5
    C 标签改变              → 在 B 之上还要重算标签、选项特征与同题 K 条聚合量
    D 从成功变失败          → 按原排除规则处理：不进 population，删除 hidden 引用
任一 question 只要有一条进了 B/C/D，该题 **K 条 trace 的聚合量**全部需重算。

用法：
    python scripts/diagnostics/mcq_parser_fix_impact_v2.py --v0 <冻结旧规则模块>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp          # noqa: E402
from recover_attention import mcq_conditional_increment as mci   # noqa: E402

OUT = ROOT / "tmp/stage1_l2_g3/mcq_parser_fix_impact_v2.json"
LETTERS = list(mci.MCQ_LETTERS)

CURRENT_CHAIN = {
    "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl": "G3(512)",
    "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl": "G4(512)",
}
RAW_IDS_SAVED = dict.fromkeys(CURRENT_CHAIN, False)


def _classify(old_label, old_span, new_label, new_span) -> str:
    if new_label is None and old_label is not None:
        return "D_成功变失败"
    if old_label is None and new_label is not None:
        return "B_新增可解析答案"
    if old_label != new_label:
        return "C_标签改变"
    if old_span != new_span:
        return "B_定位改变"
    return "A_标签与定位均不变"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reparse-dir", type=Path,
                    default=ROOT / "outputs/stage1_llama_v3/mcq_reparse_v3")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("_dlp_v0", args.v0)
    v0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v0)

    def v0_span(completion, label):
        if label is None:
            return None
        _, valid = v0._valid_label_set(LETTERS)
        ex = [m for m in v0._EXPLICIT_ANSWER_RE.finditer(completion)
              if m.group(1).upper() in valid]
        if ex:
            return list(ex[-1].span(1))
        tr = v0._trailing_isolated_matches(completion, valid)
        return list(tr[-1].span(1)) if tr else None

    buckets: dict[str, list] = defaultdict(list)
    per_file = Counter()
    affected_q: dict[str, set] = defaultdict(set)
    args.reparse_dir.mkdir(parents=True, exist_ok=True)
    reparse_files = {}

    for rel, tag in CURRENT_CHAIN.items():
        rows = [json.loads(x) for x in io.open(ROOT / rel, encoding="utf-8") if x.strip()]
        lines = []
        for row in rows:
            completion = row.get("raw_completion")
            if not isinstance(completion, str):
                continue
            o = v0.parse_option_answer(completion, LETTERS)
            n = dlp.parse_option_answer(completion, LETTERS)
            os_, ns = v0_span(completion, o["parsed_label"]), n["answer_char_span"]
            kind = _classify(o["parsed_label"], os_, n["parsed_label"], ns)
            lines.append(json.dumps({
                "schema": "stage1_mcq_reparse_v3_row",
                "source": rel, "unit_id": row.get("unit_id"),
                "example_id": row.get("example_id"),
                "sample_type": row.get("sample_type"),
                "sample_index": row.get("sample_index"),
                "parsed_label": n["parsed_label"], "parse_method": n["parse_method"],
                "answer_char_span": ns,
                "parse_failure_reason": n.get("parse_failure_reason"),
                "answer_claims": n.get("answer_claims"),
                "rejected_claims": n.get("rejected_claims"),
                "previous_frozen_rules": {"label": o["parsed_label"],
                                          "method": o["parse_method"], "span": os_},
                "impact_class": kind,
                "features_reusable": kind == "A_标签与定位均不变",
            }, ensure_ascii=False))
            if kind == "A_标签与定位均不变":
                continue
            per_file[(tag, kind)] += 1
            affected_q[tag].add(row["example_id"])
            buckets[kind].append({
                "tag": tag, "file": rel, "unit_id": row.get("unit_id"),
                "example_id": row.get("example_id"),
                "old_label": o["parsed_label"], "old_span": os_,
                "new_label": n["parsed_label"], "new_span": ns,
                "new_method": n["parse_method"],
                "new_failure_reason": n.get("parse_failure_reason"),
                "raw_generated_ids_saved": RAW_IDS_SAVED[rel],
                "needs": ("重新 teacher-forced 前向以取该行 hidden 与 F5"
                          if kind.startswith(("B", "C")) else
                          "按原排除规则处理：不进 population，删除该行 hidden 引用"),
            })
        name = f"{tag.replace('(', '_').replace(')', '')}_mcq_reparse.jsonl".lower()
        path = args.reparse_dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reparse_files[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "n_rows": len(lines),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"[新解析] {path}  {len(lines)} 行")

    k_recompute = {}
    for rel, tag in CURRENT_CHAIN.items():
        rows = [json.loads(x) for x in io.open(ROOT / rel, encoding="utf-8") if x.strip()]
        qs = affected_q[tag]
        k_recompute[tag] = {
            "n_questions_affected": len(qs),
            "n_rows_in_those_questions": sum(1 for r in rows if r.get("example_id") in qs),
            "fields": ["f5_self_consistency_exact", "f5_letter_agreement_rate"],
            "why": ("attach_cross_sample_f5 按 example_id 聚合 parsed_label —— "
                    "同题任一条标签或 eligibility 变了，这两维对该题全部 K 条都变"),
            "example_ids": sorted(qs)}

    n_b = sum(len(v) for k, v in buckets.items() if k.startswith("B"))
    n_c = len(buckets.get("C_标签改变", []))
    n_d = len(buckets.get("D_成功变失败", []))

    parser_sha = hashlib.sha256(
        (ROOT / "src/recover_attention/domain_label_proxy.py").read_bytes()).hexdigest()
    (args.reparse_dir / "mcq_reparse_v3_meta.json").write_text(json.dumps({
        "schema": "stage1_mcq_reparse_v3_meta", "date": "2026-09-15",
        "what": ("收紧后解析规则的新解析结果，与旧解析产物分开保存。"
                 "**这不是新的 gate 报告** —— 受影响行的 hidden / F5 尚未重新提取，"
                 "不得据此更新任何 passed 标记。"),
        "parser_sha256": parser_sha,
        "frozen_rules_sha256": hashlib.sha256(args.v0.read_bytes()).hexdigest(),
        "files": reparse_files,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    doc = {
        "schema": "stage1_mcq_parser_fix_impact_v2",
        "date": "2026-09-15",
        "supersedes": ("tmp/stage1_l2_g3/mcq_parser_fix_impact.json —— "
                       "那份基于第一版修复规则，**数字已作废**（当时 6 条）；"
                       "本份按收紧后的规则重新计算，旧文件保留不改"),
        "scope": ("只覆盖当前证据链 g3_budget512 / g4_budget512 的 MCQ 臂；"
                  "g3/、g3_postfix/、mcq_length_diagnosis/ 保留原样，不进计划"),
        "parser_sha256": parser_sha,
        "counts_by_file_and_kind": {f"{t}|{k}": v for (t, k), v in sorted(per_file.items())},
        "buckets": dict(buckets),
        "n_rows_changed": n_b + n_c + n_d,
        "n_needs_refeature": n_b + n_c,
        "n_label_changed": n_c,
        "n_newly_excluded": n_d,
        "k_trace_recompute": k_recompute,
        "reparse_artifacts": reparse_files,
        "raw_generated_ids": {
            "saved_in_these_artifacts": RAW_IDS_SAVED,
            "limitation": (
                "g3_budget512 / g4_budget512 的行**没有**落盘原始生成 token ids"
                "（只有 num_generated_tokens、letter_token_id、letter_token_index_local）。"
                "因此不能在离线重放里重新定位答案字母的 token 下标 —— 把 completion 重新"
                "编码得到的序列**不是**原始生成序列，不得冒充它。需要 hidden / F5 的行"
                "必须重新做一次 teacher-forced 前向；本轮不加载权重，故只出计划、不执行。")},
        "reuse_rule": ("只改 parsed_label 或 wrong_label 而沿用旧位置的特征是**不允许**的："
                       "B4 的 F5 与 H 都读在答案字母 token 上，位置变了就不再对应同一个 token。"),
        "not_done_this_round": [
            "未重新生成任何 completion", "未加载模型权重、未做 teacher-forced 前向",
            "未改写任何既有产物；旧报告保持原样", "未更新任何 gate 报告的 passed 标记"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    for key, v in sorted(per_file.items()):
        print(f"  {key[0]:9s} {key[1]:20s} {v}")
    print(f"[受影响] 共 {n_b + n_c + n_d} 条：需重提取 {n_b + n_c}（标签变 {n_c}），"
          f"新增排除 {n_d}")
    for tag, v in k_recompute.items():
        print(f"[K 条聚合] {tag}: {v['n_questions_affected']} 题 / "
              f"{v['n_rows_in_those_questions']} 条 trace")
    print(f"[out  ] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
