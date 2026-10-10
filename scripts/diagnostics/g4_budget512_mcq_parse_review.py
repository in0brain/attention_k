r"""G4 MCQ（512）解析结果的**独立文本审阅**，与原 gate 结果分开记录。

⚠️ 这里**没有分类正则做判定**。下面的判断是 Claude 逐条阅读 240 条 completion 后
写下的（231 条标准形态读其结论句，4 条回退分支与 5 条解析失败读全文）。
脚本只负责机械核对、对齐与汇总；**不能只靠同一条正则证明没有误判**，所以凡是
判定性的结论都写在数据里、由人眼产生，正则只用来把 240 条分成"哪些需要细读"。

标注者写作 `annotator: "Claude 文本审阅"` —— **不是**独立人类标注。

**不更换解析器、不放宽候选规则、不重判既有证据、不改变 gate 结论。**

用法：
    python scripts/diagnostics/g4_budget512_mcq_parse_review.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp   # noqa: E402

RUN = ROOT / "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl"
OUT = ROOT / "tmp/stage1_l2_g3/g4_budget512_mcq_parse_review.json"
BUDGET = 512
PARSER_OK = ("explicit_answer_marker", "last_isolated_label")

#: 231 条"标准形态"（全篇唯一 `Answer:` 标记且以 `Answer: X` 收尾）的逐条阅读结论。
#: 我把这 231 条的结论句全部读过（见 §审阅方式），形态一致故按形态记一条结论，
#: 并由脚本逐条验证该形态确实成立 —— 任一条不成立即退出非零。
STANDARD_FORM_VERDICT = {
    "n": 231,
    "read_how": ("逐条打印 `Answer:` 之前 115 字符的结论句并通读；"
                 "核对结论句所指的选项与随后的字母是否一致"),
    "enumeration_false_positive": 0,
    "fragment_false_positive": 0,
    "no_clear_choice_but_parsed": 0,
    "letter_contradicts_conclusion": 0,
    "note": ("231 条都是先给一段作结句再以 `Answer: X` 收尾，字母与结论句一致；"
             "没有取自选项清单的枚举标号，也没有截断词残字"),
}

#: 4 条走 `last_isolated_label` 回退分支的记录 —— 这是此前两轮全部伪阳性的来源，
#: 故逐条读全文。
FALLBACK_REVIEW = {
    "cybermetric_005102__sampled_5": {
        "claude_final_choice": "D", "parser_label": "D", "agree": True,
        "evidence": "the person most likely to oversee ... would be the Chief Information Officer (D).",
        "reason": "完整句作结，明确给出选项文本与字母；无 Answer: 标记但选择明确",
    },
    "cybermetric_000402__sampled_3": {
        "claude_final_choice": "C", "parser_label": "C", "agree": True,
        "evidence": "Considering these points, the best answer is: C",
        "reason": "以『the best answer is: C』作结，明确最终选择",
    },
    "cybermetric_006832__sampled_3": {
        "claude_final_choice": "A", "parser_label": "A", "agree": True,
        "evidence": "Therefore, the correct answer is A.",
        "reason": "完整句作结，明确最终选择",
    },
    "cybermetric_007597__sampled_1": {
        "claude_final_choice": "C", "parser_label": "D", "agree": False,
        "evidence": ("However, the correct answer is C since it includes all provided "
                     "answers which are A and D."),
        "reason": ("**明确的误判**：模型最终选择是 C（选项 C = 'All provided answers'），"
                   "但 `last_isolated_label` 取的是全文**最后一个**孤立字母 —— "
                   "那是句尾列举 'A and D' 里的 D，与最终选择无关"),
        "impact": ("该行 parsed_label=D → wrong_label=1 是**错的**；"
                   "wrong_label 就是 y，故这一行的结果变量被污染"),
    },
}

#: 5 条解析失败 —— 逐条读全文，查是否有被漏掉的答案。
FAILURE_REVIEW = {
    "cybermetric_007382__sampled_1": {
        "claude_final_choice": "B",
        "evidence": "Among the given options, the best choice would be B, Scanning files before accepting them.",
        "verdict": "明确作答但未被识别（漏判）"},
    "cybermetric_007382__sampled_4": {
        "claude_final_choice": "B",
        "evidence": "I'll choose option B (Scanning files before accepting them) because it is the most common approach",
        "verdict": "明确作答但未被识别（漏判）"},
    "cybermetric_004666__sampled_2": {
        "claude_final_choice": "A",
        "evidence": "Considering these points, social engineering (A) is the most suitable option.",
        "verdict": "明确作答但未被识别（漏判）"},
    "cybermetric_005975__sampled_4": {
        "claude_final_choice": "B",
        "evidence": "Based on this analysis, the correct answer is B. IETF.",
        "verdict": "明确作答但未被识别（漏判）"},
    "cybermetric_002932__sampled_5": {
        "claude_final_choice": "C",
        "evidence": "But considering what we've learned, the best answer would be C as it provides more specific details.",
        "verdict": "明确作答但未被识别（漏判）"},
}


def main() -> int:
    rows = [json.loads(x) for x in io.open(RUN, encoding="utf-8") if x.strip()]
    assert len(rows) == 240, len(rows)
    by_id = {r["unit_id"]: r for r in rows}

    ok = [r for r in rows if r.get("parse_status") in PARSER_OK]
    fallback = [r for r in rows if r.get("parse_status") == "last_isolated_label"]
    bad = [r for r in rows if r.get("parse_status") not in PARSER_OK]

    problems: list[str] = []
    std_ids: list[str] = []
    for r in ok:
        if r.get("parse_status") != "explicit_answer_marker":
            continue
        c = r.get("raw_completion") or ""
        marks = list(dlp._EXPLICIT_ANSWER_RE.finditer(c))
        m = re.search(r"Answer:\s*([ABCD])\s*$", c.strip())
        if len(marks) == 1 and m:
            std_ids.append(r["unit_id"])
            if m.group(1) != r.get("parsed_label"):
                problems.append(f"{r['unit_id']}: 末尾字母 != parsed_label")
        else:
            problems.append(f"{r['unit_id']}: 不是标准形态（markers={len(marks)}，尾部匹配={bool(m)}）")

    if len(std_ids) != STANDARD_FORM_VERDICT["n"]:
        problems.append(f"标准形态条数 {len(std_ids)} != 审阅记录的 {STANDARD_FORM_VERDICT['n']}")
    if sorted(x["unit_id"] for x in fallback) != sorted(FALLBACK_REVIEW):
        problems.append("回退分支集合与逐条审阅记录不一致")
    if sorted(x["unit_id"] for x in bad) != sorted(FAILURE_REVIEW):
        problems.append("解析失败集合与逐条审阅记录不一致")
    n_touched = sum(1 for r in rows
                    if isinstance(r.get("num_generated_tokens"), int)
                    and r["num_generated_tokens"] >= BUDGET)
    if n_touched:
        problems.append(f"有 {n_touched} 条撞满解码上限 —— 截断词残字机制在本批可能发生")

    if problems:
        print("[STOP ] 机械核对不通过 —— 审阅结论与产物对不上：", file=sys.stderr)
        for p in problems:
            print("   ", p, file=sys.stderr)
        return 2

    false_positives = [{"unit_id": k, **v} for k, v in FALLBACK_REVIEW.items()
                       if not v["agree"]]
    lens = sorted(r["num_generated_tokens"] for r in rows
                  if isinstance(r.get("num_generated_tokens"), int))

    doc = {
        "schema": "stage1_g4_budget512_mcq_parse_review_v1",
        "date": "2026-09-15",
        "scope": ("**独立审阅，与 gate 结果分开**。由 Claude 逐条阅读 240 条 completion；"
                  "未更换解析器、未放宽候选规则、未重判既有证据、未改变 gate 结论"),
        "annotator": "Claude 文本审阅（**不是**独立人类标注，**不是**正则分类）",
        "read_method": ("231 条标准形态读结论句；4 条 last_isolated_label 回退与 "
                        "5 条解析失败读全文。正则只用于把 240 条分成'哪些需要细读'，"
                        "**不用于判定**"),
        "input": {"path": str(RUN.relative_to(ROOT)).replace("\\", "/"),
                  "sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
                  "n_rows": len(rows)},
        "parser_result": {
            "explicit_answer_marker": sum(1 for r in rows
                                          if r["parse_status"] == "explicit_answer_marker"),
            "last_isolated_label": len(fallback),
            "parse_failure": len(bad),
            "parse_failure_rate": round(len(bad) / len(rows), 4)},
        "completion_length_tokens": {
            "min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1],
            "budget": BUDGET, "n_touched_budget": n_touched},
        "mechanical_verification": {
            "standard_form_rows": len(std_ids),
            "single_answer_marker_and_tail_letter": True,
            "tail_letter_equals_parsed_label": True,
            "note": "逐条验证；任一条不成立即退出非零，不放过"},
        "standard_form_verdict": STANDARD_FORM_VERDICT,
        "fallback_review": FALLBACK_REVIEW,
        "failure_review": FAILURE_REVIEW,
        "false_positives": {
            "n": len(false_positives),
            "detail": false_positives,
            "statement": (
                "**发现 1 条明确伪阳性**：cybermetric_007597__sampled_1 的最终选择是 C，"
                "解析器却记成 D（取自句尾列举 'A and D'）。该行 wrong_label 即 y 被污染。"
                "这不是'可能'，是本批实测到的一条。" if false_positives else
                "本批未发现伪阳性")},
        "false_negatives": {
            "n": len(FAILURE_REVIEW),
            "statement": ("5 条解析失败**全部**含有明确的最终答案（B / B / A / B / C），"
                          "均为解析器漏判。漏判使 parse_failure_rate **偏高**而非偏低")},
        "impact_on_gate": (
            "当前误判使解析失败率**少计了 1 条**：若仅把该行判为失败，"
            "由 5/240 变成 6/240（0.0208 → 0.0250），仍低于 0.10。"
            "**这条错误不改变当前阈值通过结论。**"
            "不得写成『伪阳性完全不影响 gate 数值』—— 数值确实被少计了 1 条。"
            "另外那 1 条让 235 条判成功里有 1 条的结果变量(y)是错的（1/235 ≈ 0.43%），"
            "**因此不能宣布阻塞全部解除** —— gate 数值通过与『解析成功都可信』是两回事。"),
        "not_claimed": [
            "不因此更换解析器、不放宽候选规则、不重判既有证据、不重跑。",
            "『标准形态 231 条没有伪阳性』是本批的观察，不等于 last_isolated_label "
            "回退的已知问题已被普遍排除 —— 本批它被触发 4 次，其中 1 次就错了。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"parser": doc["parser_result"],
                      "lens": doc["completion_length_tokens"]}, ensure_ascii=False, indent=1))
    print("伪阳性:", doc["false_positives"]["statement"])
    print("漏判  :", doc["false_negatives"]["statement"])
    print("对 gate:", doc["impact_on_gate"])
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
