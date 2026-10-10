r"""把 **Claude 文本审阅** 的逐条判断物化成 JSONL。

⚠️ 这里**没有分类正则**。下面的 `LEN128_FINAL` / `LEN128_UNCERTAIN` / `CATEGORY_*`
是 Claude 逐条阅读 240 条 completion 后写下的判断（阅读时不看 parse_status、
不看 parsed_label、不看正确答案）。脚本只负责：
    - 把判断按 unit_id 对齐到原始记录
    - 用我给出的**证据片段**在原文里定位字符区间（机械记账，不是判定）
    - 校验覆盖率（240/240）

标注者写作 `annotator: "Claude 文本审阅"` —— **不是**独立人类标注。

用法：
    python scripts/diagnostics/build_claude_text_review.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "outputs/stage1_llama_v3/mcq_length_diagnosis/mcq_length_diagnosis_rows.jsonl"
OUT = ROOT / "outputs/stage1_llama_v3/mcq_length_diagnosis/claude_text_review.jsonl"

# ── len128：明确作出最终选择（索引 = 按 example_id/sample_type/sample_index 排序后的序号）
#    值 = (标签, 证据片段, 判定理由)
LEN128_FINAL = {
    5:   ("A", "Answer: A", "以 Answer: A 收尾并命中 EOS，前文逐一排除 B/C/D"),
    13:  ("C", "Answer: C", "先给出结论句再以 Answer: C 收尾，命中 EOS"),
    18:  ("C", "Answer: C", "推导 DES 16 轮后以 Answer: C 收尾，命中 EOS"),
    20:  ("C", "Answer: C", "'the correct answer is:' 后接 Answer: C，命中 EOS"),
    22:  ("C", "Answer: C", "'the answer is:' 后接 Answer: C，命中 EOS"),
    24:  ("D", "Answer: D", "给出结论句后以 Answer: D 收尾，命中 EOS"),
    28:  ("D", "Answer: D", "给出结论句后以 Answer: D 收尾，命中 EOS"),
    35:  ("未知-需映射", "there must be a minimum of 4 hosts identified by the ping scan.",
          "**完整句作结并命中 EOS，明确给出最终数量 4，但全程未给选项字母**；"
          "按卡要求无法可靠映射到标签时记未知，只保留选项文本"),
    81:  ("C", "Answer: C", "两句话直接作答并以 Answer: C 收尾，命中 EOS"),
    84:  ("B", "Answer: B", "推导后以 Answer: B 收尾，命中 EOS"),
    87:  ("B", "Answer: B", "'makes option B the most suitable choice' 后接 Answer: B，命中 EOS"),
    105: ("未知-需映射", 'the term "spoofing" best describes the general practice of impersonating a',
          "**明确断言 spoofing 最贴切**，属已作出的最终选择；句子在复述题干处被截断，"
          "但选择本身已完成。未给字母，故标签记未知，只保留选项文本 'spoofing'"),
    111: ("B", "Answer: B", "'the correct answer is:' 后接 Answer: B，命中 EOS"),
    112: ("B", "Answer: B", "'the correct answer is:' 后接 Answer: B，命中 EOS"),
}

# ── len128：无法确定
LEN128_UNCERTAIN = {
    19: ("choose the option that is closest to 16",
         "表示要选'最接近 16 的选项'但**未给标签**，随后又称信息不足 —— 存在改口/否定"),
    85: ("This is correct.",
         "在逐项分析中对 B 内联断言 'This is correct'，但**其后继续评估 C/D 且被截断**，"
         "未出现任何最终选择语句 —— 不按最终选择计"),
    96: ("the correct answer is the one that specifically requests all records from an authoritative server.",
         "以**定义指代**代替标签（上文已把该定义归给 C/axfr），但未完成显式选择 —— 记无法确定"),
}

# ── len128 四分类：其余默认「选项讨论」，下列记「未完成分析」
LEN128_INCOMPLETE = {19, 21, 23, 30, 31, 32, 34}

# ── len16：**全部 120 条均未作出最终选择**（逐条阅读结论）
LEN16_OPTION_TALK = {5, 7, 9, 17, 25, 26, 27, 36, 39, 40, 42, 43, 44, 45, 46, 47, 56,
                     59, 60, 63, 66, 67, 68, 92, 93, 96, 97, 98, 99, 100, 101, 102,
                     103, 104, 105, 114, 118}
LEN16_OTHER = {13, 34, 35, 81, 83}

NEGATION_HINTS = ("is incorrect", "is also incorrect", "doesn't", "does not",
                  "isn't", "not the primary", "不")


def main() -> int:
    rows = [json.loads(x) for x in io.open(ROWS, encoding="utf-8") if x.strip()]
    by_cond = {}
    for cond in ("len16", "len128"):
        g = [r for r in rows if r["condition"] == cond]
        g.sort(key=lambda r: (r["example_id"], r["sample_type"], r["sample_index"]))
        by_cond[cond] = g

    out = []
    for cond, g in by_cond.items():
        for i, r in enumerate(g):
            text = r["decoded_keep_special_tokens"] or ""
            rec = {
                "schema": "stage1_mcq_length_claude_text_review_v1",
                "annotator": "Claude 文本审阅",
                "annotator_note": ("由 Claude 逐条阅读 completion 后判定；"
                                   "阅读时未参考 parse_status / parsed_label / 正确答案。"
                                   "**这不是独立人类标注，也不是正则分类。**"),
                "condition": cond, "index_in_condition": i,
                "unit_id": r["unit_id"], "example_id": r["example_id"],
                "sample_type": r["sample_type"],
                "made_final_choice": "否",
                "chosen_label": None, "chosen_option_text": None,
                "evidence_text": None, "evidence_char_span": None,
                "output_category": None,
                "flags": {"multiple_choices": False, "negation": False,
                          "self_correction": False, "quotes_options": False},
                "reason": None,
            }
            low = text.lower()
            rec["flags"]["negation"] = any(h in low for h in NEGATION_HINTS)
            rec["flags"]["quotes_options"] = any(
                p in text for p in ("Option A", "Option B", "Option C", "Option D",
                                    "A.", "B.", "C.", "D.", "A (", "B (", "C (", "D ("))

            if cond == "len16":
                rec["output_category"] = ("选项讨论" if i in LEN16_OPTION_TALK else
                                          "其他情形" if i in LEN16_OTHER else "未完成分析")
                rec["reason"] = ("生成在 16 词元处被截断，全文停留在开场铺垫或刚进入选项枚举；"
                                 "没有任何最终选择语句")
            else:
                if i in LEN128_FINAL:
                    label, ev, why = LEN128_FINAL[i]
                    rec.update({"made_final_choice": "是", "evidence_text": ev,
                                "output_category": "最终选择", "reason": why})
                    if label == "未知-需映射":
                        rec["chosen_label"] = "未知"
                        rec["chosen_option_text"] = (
                            "minimum of 4 hosts" if i == 35 else "spoofing")
                    else:
                        rec["chosen_label"] = label
                elif i in LEN128_UNCERTAIN:
                    ev, why = LEN128_UNCERTAIN[i]
                    rec.update({"made_final_choice": "无法确定", "evidence_text": ev,
                                "output_category": ("未完成分析" if i in LEN128_INCOMPLETE
                                                    else "选项讨论"),
                                "reason": why})
                else:
                    rec["output_category"] = ("未完成分析" if i in LEN128_INCOMPLETE
                                              else "选项讨论")
                    rec["reason"] = ("生成在 128 词元处被截断或提前收尾，"
                                     "文中只有对各选项的分析，没有最终选择语句")
                if i == 19:
                    rec["flags"]["self_correction"] = True

            if rec["evidence_text"]:
                k = text.find(rec["evidence_text"])
                rec["evidence_char_span"] = ([k, k + len(rec["evidence_text"])]
                                             if k >= 0 else None)
                if k < 0:
                    print(f"!! 证据片段未在原文找到：{cond}[{i}] {rec['unit_id']}")
            out.append(rec)

    assert len(out) == 240, len(out)
    assert len({r["unit_id"] + r["condition"] for r in out}) == 240
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
                   encoding="utf-8")

    from collections import Counter
    for cond in ("len16", "len128"):
        g = [r for r in out if r["condition"] == cond]
        for st in ("greedy", "sampled"):
            gg = [r for r in g if r["sample_type"] == st]
            print(f"{cond}/{st}: n={len(gg)} "
                  f"choice={dict(Counter(r['made_final_choice'] for r in gg))} "
                  f"cat={dict(Counter(r['output_category'] for r in gg))}")
    print("input sha256:", hashlib.sha256(ROWS.read_bytes()).hexdigest()[:16])
    print("coverage:", len(out), "/ 240")
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
