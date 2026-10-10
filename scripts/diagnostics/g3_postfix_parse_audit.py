r"""修复后 Llama MCQ G3 的**人工诊断**（与原 gate 结果分开记录）。

只读 `outputs/stage1_llama_v3/g3_postfix/` 的落盘产物，做纯文本重放：

    原解析器判为成功的条目里，是否仍出现「选项枚举标号被当成答案」
    `Answer:` 标记的出现条数
    与修复前 g3/ 的同题对照（两批都只读，不改任何一方）

**不更换解析器、不重判旧证据、不改变任何 gate 结论。**

用法：
    python scripts/diagnostics/g3_postfix_parse_audit.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp            # noqa: E402
from recover_attention import mcq_conditional_increment as mci     # noqa: E402
from recover_attention.generation import backends as be            # noqa: E402

NEW = ROOT / "outputs" / "stage1_llama_v3" / "g3_postfix" / "mcq_g3_unit_results.jsonl"
OLD = ROOT / "outputs" / "stage1_llama_v3" / "g3" / "mcq_g3_unit_results.jsonl"
OUT = ROOT / "tmp" / "stage1_l2_g3" / "g3_postfix_parse_audit.json"
LETTERS = list(mci.MCQ_LETTERS)


def rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with io.open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def audit(rs: list[dict]) -> dict:
    methods = Counter(r.get("parse_status") for r in rs)
    marker = [r["unit_id"] for r in rs
              if dlp._EXPLICIT_ANSWER_RE.search(r.get("raw_completion") or "")]
    enum_fp, real_answer, unclear = [], [], []
    for r in rs:
        if r.get("parse_status") not in ("explicit_answer_marker", "last_isolated_label"):
            continue
        c = r.get("raw_completion") or ""
        try:
            cs, ce = be._parsed_letter_char_span(c, r["parsed_label"])
        except ValueError as exc:
            unclear.append({"unit_id": r["unit_id"], "why": str(exc)[:80]})
            continue
        before, after = c[:cs], c[ce:]
        # 命中字母之后只剩标点/空白，且之前刚出现选项枚举的引导语与换行/项目符号
        # → 它是**枚举标号**，不是答案。判据与 2026-09-13 的离线审计一致。
        is_enum = (bool(re.search(r"option", before, re.I))
                   and bool(re.search(r"(?:\n\s*|[-*•(]\s*|\*\*|Option\s)$", before))
                   and after.strip(" .:;,)]}*\n\t") == "")
        item = {"unit_id": r["unit_id"], "parsed_label": r["parsed_label"],
                "parse_status": r["parse_status"], "tail": c[-48:]}
        (enum_fp if is_enum else real_answer).append(item)
    return {
        "n_rows": len(rs),
        "parse_status_counts": dict(methods),
        "answer_marker_present": len(marker),
        "n_enumeration_label_false_positives": len(enum_fp),
        "n_not_enumeration_pattern": len(real_answer),
        "n_span_unresolvable": len(unclear),
        "enumeration_false_positive_examples": enum_fp[:5],
        "not_enumeration_examples": real_answer[:5],
        "span_unresolvable": unclear[:5],
    }


def main() -> int:
    global NEW, OLD, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", type=Path, default=NEW, help="待审的 unit_results.jsonl")
    ap.add_argument("--old", type=Path, default=OLD, help="对照用的旧 unit_results.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    NEW, OLD, OUT = args.new, args.old, args.out
    new_rows, old_rows = rows(NEW), rows(OLD)
    if not new_rows:
        print(f"未找到修复后产物：{NEW}")
        return 1
    doc = {
        "schema": "stage1_g3_postfix_parse_audit_v1",
        "date": "2026-09-14",
        "scope": ("**人工诊断，与原 gate 结果分开**。只读产物做纯文本重放；"
                  "未更换解析器、未重判旧证据、未改变任何 gate 结论"),
        "postfix": {"path": str(NEW), **audit(new_rows)},
        "prefix_for_contrast": {"path": str(OLD), **audit(old_rows)} if old_rows else None,
        "known_issue_statement": None,
    }
    n_fp = doc["postfix"]["n_enumeration_label_false_positives"]
    parsed = (doc["postfix"]["parse_status_counts"].get("explicit_answer_marker", 0)
              + doc["postfix"]["parse_status_counts"].get("last_isolated_label", 0))
    doc["known_issue_statement"] = (
        f"原解析器判为成功 {parsed} 条；其中 {n_fp} 条仍是"
        f"「选项枚举标号被当成答案」的已知问题"
        if n_fp else
        f"原解析器判为成功 {parsed} 条；本轮**未再出现**「选项枚举标号被当成答案」的模式"
        if parsed else
        "原解析器判为成功 0 条 —— 该已知问题在本轮无从触发")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc["postfix"][k] for k in
                      ("n_rows", "parse_status_counts", "answer_marker_present",
                       "n_enumeration_label_false_positives",
                       "n_not_enumeration_pattern", "n_span_unresolvable")},
                     ensure_ascii=False, indent=1))
    print(doc["known_issue_statement"])
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
