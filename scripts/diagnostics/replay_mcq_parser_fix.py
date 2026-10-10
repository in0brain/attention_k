r"""MCQ 答案提取修复的**离线重放**：逐条比较修复前后。

只读已有落盘输出，**不加载模型权重、不重新生成、不改写任何既有产物**。
覆盖 Qwen 侧（G3 / G4 / 生产**原始**记录，不是筛选后的）与 Llama 侧
（生成长度检查 / 全部 G3 / G4 / 长度诊断）的全部 MCQ 输出。

逐条比较：`parsed_label` / `parse_method` / 答案字符区间 / `eligible_for_primary`
以及可能受影响的派生字段（`canonicalization_status` / `selected_option_text` /
`response_format` / `bare_answer`）。

旧实现从一份**冻结副本**加载（`--old-module`），不是靠 git stash 或注释掉新代码。

用法：
    python scripts/diagnostics/replay_mcq_parser_fix.py --old-module <旧 domain_label_proxy.py>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp          # noqa: E402
from recover_attention import mcq_conditional_increment as mci   # noqa: E402

OUT = ROOT / "tmp/stage1_l2_g3/mcq_parser_fix_replay.json"
LETTERS = list(mci.MCQ_LETTERS)

#: 重放范围。`side` 决定差异是否触发"Qwen 记录发生变化"这条红线。
FILES = [
    ("qwen", "G3",          "outputs/stage1/g3/mcq_g3_unit_results.jsonl"),
    ("qwen", "G4",          "outputs/stage1/g4/mcq_g4_unit_results.jsonl"),
    ("qwen", "生产(原始)",   "outputs/stage1/production/mcq_raw_unit_results.jsonl"),
    ("llama", "G3(修复前)",  "outputs/stage1_llama_v3/g3/mcq_g3_unit_results.jsonl"),
    ("llama", "G3(16,修复后)", "outputs/stage1_llama_v3/g3_postfix/mcq_g3_unit_results.jsonl"),
    ("llama", "G3(512)",     "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl"),
    ("llama", "G4(512)",     "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl"),
    ("llama", "生成长度检查", "outputs/stage1_llama_v3/mcq_budget512_check/mcq_budget512_unit_results.jsonl"),
    ("llama", "长度诊断",     "outputs/stage1_llama_v3/mcq_length_diagnosis/mcq_length_diagnosis_rows.jsonl"),
]


def _load_old(path: Path):
    spec = importlib.util.spec_from_file_location("_dlp_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _old_span(old, completion: str, label: str):
    """旧实现的定位逻辑（backends._parsed_letter_char_span 修复前的那一版）。"""
    _, valid = old._valid_label_set(LETTERS)
    explicit = [m for m in old._EXPLICIT_ANSWER_RE.finditer(completion)
                if m.group(1).upper() in valid]
    if explicit:
        m = explicit[-1]
        return list(m.span(1)) if m.group(1).upper() == label else "disagree"
    trailing = old._trailing_isolated_matches(completion, valid)
    if trailing:
        m = trailing[-1]
        return list(m.span(1)) if m.group(1).upper() == label else "disagree"
    return None


def _derived(label, row):
    """受解析结果影响的派生字段（只用行内已有的选项清单，不碰模型）。"""
    choices = row.get("candidate_choices")
    if not choices:
        return {}
    canon = mci.canonicalize_mcq_output(label, choices)
    out = {"eligible_for_primary": canon.get("eligible_for_primary"),
           "canonicalization_status": canon.get("canonicalization_status"),
           "selected_option_text": canon.get("selected_option_text")}
    completion = row.get("raw_completion") or ""
    fmt = mci.response_format_of(completion, label)
    out["response_format"] = fmt.get("response_format")
    out["bare_answer"] = fmt.get("bare_answer")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-module", type=Path, required=True,
                    help="修复前的 domain_label_proxy.py 冻结副本")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-examples", type=int, default=40)
    args = ap.parse_args()

    old = _load_old(args.old_module)
    print(f"[old  ] {args.old_module}  sha={hashlib.sha256(args.old_module.read_bytes()).hexdigest()[:16]}…")
    print(f"[new  ] src/recover_attention/domain_label_proxy.py  "
          f"sha={hashlib.sha256((ROOT / 'src/recover_attention/domain_label_proxy.py').read_bytes()).hexdigest()[:16]}…")

    per_file, diffs = [], []
    totals = Counter()
    span_inconsistent = []

    for side, tag, rel in FILES:
        path = ROOT / rel
        n = n_diff = 0
        label_changed = span_changed = elig_changed = method_changed = 0
        transitions = Counter()
        for line in io.open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            completion = row.get("raw_completion")
            if not isinstance(completion, str):
                continue
            n += 1
            o = old.parse_option_answer(completion, LETTERS)
            w = dlp.parse_option_answer(completion, LETTERS)
            o_label, w_label = o["parsed_label"], w["parsed_label"]
            o_span = _old_span(old, completion, o_label) if o_label else None
            w_span = w["answer_char_span"]

            # 新实现的自洽性：区间里的字符必须就是标签
            if w_label is not None:
                if not w_span or completion[w_span[0]:w_span[1]].upper() != w_label:
                    span_inconsistent.append({"file": rel, "unit_id": row.get("unit_id"),
                                              "label": w_label, "span": w_span})

            changed = (o_label != w_label) or (o["parse_method"] != w["parse_method"]) \
                or (o_span != w_span)
            if not changed:
                continue
            n_diff += 1
            label_changed += int(o_label != w_label)
            method_changed += int(o["parse_method"] != w["parse_method"])
            span_changed += int(o_span != w_span)
            o_d, w_d = _derived(o_label, row), _derived(w_label, row)
            elig_changed += int(o_d.get("eligible_for_primary")
                                != w_d.get("eligible_for_primary"))
            transitions[f'{o["parse_method"]}→{w["parse_method"]}'] += 1
            diffs.append({
                "side": side, "file": rel, "unit_id": row.get("unit_id"),
                "example_id": row.get("example_id"),
                "old": {"label": o_label, "method": o["parse_method"], "span": o_span,
                        **o_d},
                "new": {"label": w_label, "method": w["parse_method"], "span": w_span,
                        "failure_reason": w.get("parse_failure_reason"), **w_d},
                "tail": completion.strip()[-160:],
            })
        totals[side] += n
        per_file.append({"side": side, "tag": tag, "path": rel, "n_rows": n,
                         "n_changed": n_diff, "label_changed": label_changed,
                         "parse_method_changed": method_changed,
                         "span_changed": span_changed,
                         "eligibility_changed": elig_changed,
                         "transitions": dict(transitions),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"  {side:5s} {tag:14s} n={n:6d} 变化={n_diff:4d} "
              f"（标签 {label_changed} ｜ 口径 {method_changed} ｜ 区间 {span_changed} ｜ "
              f"eligibility {elig_changed}）")

    qwen_changed = sum(f["n_changed"] for f in per_file if f["side"] == "qwen")
    qwen_label_changed = sum(f["label_changed"] for f in per_file if f["side"] == "qwen")

    doc = {
        "schema": "stage1_mcq_parser_fix_replay_v1",
        "date": "2026-09-15",
        "method": ("只读落盘 completion 重放；旧实现从冻结副本加载。"
                   "**未加载模型权重、未重新生成、未改写任何既有产物。**"),
        "old_module": {"path": str(args.old_module),
                       "sha256": hashlib.sha256(args.old_module.read_bytes()).hexdigest()},
        "new_module": {
            "path": "src/recover_attention/domain_label_proxy.py",
            "sha256": hashlib.sha256(
                (ROOT / "src/recover_attention/domain_label_proxy.py").read_bytes()).hexdigest()},
        "scope": per_file,
        "n_rows_total": sum(totals.values()),
        "n_rows_by_side": dict(totals),
        "qwen": {
            "n_rows": totals["qwen"],
            "n_changed": qwen_changed,
            "n_label_changed": qwen_label_changed,
            "statement": ("Qwen 侧逐条一致，修复不改变任何旧 Qwen 记录"
                          if qwen_changed == 0 else
                          "**Qwen 侧记录发生变化** —— 见 diffs；"
                          "不得再写『不影响旧 Qwen 结论』，也不在本轮改写旧证据或重跑 Qwen")},
        "n_changed_total": len(diffs),
        "span_self_consistency": {
            "n_violations": len(span_inconsistent),
            "detail": span_inconsistent[:10],
            "note": "新实现返回的区间取出来必须就是它给出的标签（标签与区间同源）"},
        "diffs": diffs[:args.max_examples],
        "n_diffs_listed": min(len(diffs), args.max_examples),
        "all_diff_unit_ids": [d["unit_id"] for d in diffs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n[总计] {doc['n_rows_total']} 条（qwen {totals['qwen']} / llama {totals['llama']}）"
          f"｜ 变化 {len(diffs)} 条")
    print(f"[Qwen] {doc['qwen']['statement']}")
    print(f"[自洽] 区间与标签不一致 {len(span_inconsistent)} 条")
    print(f"[out  ] {args.out}")
    return 0 if not span_inconsistent else 2


if __name__ == "__main__":
    raise SystemExit(main())
