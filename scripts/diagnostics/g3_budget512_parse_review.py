r"""对 `g3_budget512` 的解析结果做**独立审阅**，与 gate 结果分开记录。

⚠️ 这里**没有分类正则做判定**。下面的 `SUCCESS_REVIEW_VERDICT` /
`FAILURE_REVIEW` 是 Claude 逐条阅读 120 条 completion 后写下的判断。
脚本只负责：机械核对（标记条数、是否以 `Answer: X` 收尾、completion 的 token 长度、
是否撞满解码上限）、对齐、覆盖率校验与汇总。

标注者写作 `annotator: "Claude 文本审阅"` —— **不是**独立人类标注。

判定口径（沿用长度诊断那一轮）：
  提到 A/B/C/D、讨论某个选项、残缺单词或孤立字母**不算**最终选择；
  缺少 `Answer:` **也不**直接判为没有答案。
文本 → 标签的映射在**盲标之后**才做，且只用 `candidate_choices`，**不看正确答案**。

**不更换解析器、不重判旧证据、不改变 gate 结论。**

用法：
    python scripts/diagnostics/g3_budget512_parse_review.py
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

RUN = ROOT / "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl"
OUT = ROOT / "tmp/stage1_l2_g3/g3_budget512_parse_review.json"
MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
BUDGET = 512

#: 116 条解析成功记录的逐条审阅结论。全部同一形态，故按形态记一条总结论，
#: 并由下面的机械核对逐条验证该形态确实成立（任一条不成立即报错，不放过）。
SUCCESS_REVIEW_VERDICT = {
    "shape": ("每条都在一段作结句之后以 `Answer: X` 收尾，X 与作结句所指的选项一致；"
              "全篇只有一个 `Answer:` 标记，无改口，无多选"),
    "enumeration_false_positive": 0,
    "fragment_false_positive": 0,
    "letter_contradicts_conclusion": 0,
    "why_fragment_is_structurally_impossible": (
        "本批 0 条撞满解码上限（最长 completion 274 token < 512），"
        "截断词残字这一机制在本批**无从发生** —— 这是本批的观察，"
        "不等于该机制已被普遍排除"),
}

#: 4 条解析失败记录的逐条审阅（阅读时不看 parse_status 以外的判定，也不看正确答案）。
FAILURE_REVIEW = {
    "cybermetric_005120__sampled_4": {
        "made_final_choice": "是",
        "chosen_label_blind": "未知（文本给了选项文本与字母，但不带 `Answer:` 标记）",
        "evidence": "Considering these points, the correct answer is **D. WPA2**.",
        "reason": "完整句作结并明确给出选项；只是用粗体行内形式而非 `Answer:` 标记",
        "verdict": "**明确作答但未被识别**（解析器漏判）",
    },
    "cybermetric_004295__sampled_5": {
        "made_final_choice": "是",
        "chosen_label_blind": "未知（只给内容，不给字母）",
        "evidence": "So, there must be a minimum of 4 hosts identified by the ping scan.",
        "reason": "完整句作结并明确给出最终数量 4；全程未给选项字母",
        "verdict": "**明确作答但未被识别**（无字母，解析器按定义看不见）",
    },
    "cybermetric_004295__sampled_4": {
        "made_final_choice": "否",
        "chosen_label_blind": None,
        "evidence": "you might find hosts ranging from 2 upwards depending on your local network size.",
        "reason": "以范围性描述收尾，明确表示无法确定具体数目，未作出任何最终选择",
        "verdict": "尚未作答且解析失败（解析器判断正确）",
    },
    "cybermetric_003873__sampled_5": {
        "made_final_choice": "无法确定",
        "chosen_label_blind": None,
        "evidence": ("I am unable to provide the accurate information about the rounds "
                     "… But according to the documentation … which is defined to use 16 rounds"),
        "reason": ("同一段里先声明信息不足、随后又断言 16 rounds —— 自相矛盾且无最终选择语句；"
                   "输出未撞上限（198 token），是模型自行收尾，不是被截断"),
        "verdict": "无法确定",
    },
}

#: **盲标完成之后**才做的文本→标签映射，只用 candidate_choices，不看正确答案。
POST_HOC_MAP = {
    "cybermetric_005120__sampled_4": ("D", "选项 D = 'WPA2'；审阅读出 '**D. WPA2**'"),
    "cybermetric_004295__sampled_5": ("D", "选项 D = '4'；审阅读出 'minimum of 4 hosts'"),
}

PARSER_OK = ("explicit_answer_marker", "last_isolated_label")


def main() -> int:
    rows = [json.loads(x) for x in io.open(RUN, encoding="utf-8") if x.strip()]
    ok = [r for r in rows if r.get("parse_status") in PARSER_OK]
    bad = [r for r in rows if r.get("parse_status") not in PARSER_OK]
    assert len(rows) == 120, len(rows)

    from transformers import AutoTokenizer          # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    # ── 机械核对：逐条验证审阅结论里声称的形态确实成立 ────────────────────
    problems = []
    lens = []
    for r in rows:
        c = r.get("raw_completion") or ""
        lens.append(n_tokens(c))
    max_len = max(lens)
    if max_len >= BUDGET:
        problems.append(f"有 completion 达到解码上限（max={max_len} >= {BUDGET}）")

    for r in ok:
        c = r.get("raw_completion") or ""
        marks = list(dlp._EXPLICIT_ANSWER_RE.finditer(c))
        if len(marks) != 1:
            problems.append(f"{r['unit_id']}: `Answer:` 标记 {len(marks)} 处，非 1")
        m = re.search(r"Answer:\s*([ABCD])\s*$", c.strip())
        if not m:
            problems.append(f"{r['unit_id']}: 不以 `Answer: X` 收尾")
        elif m.group(1) != r.get("parsed_label"):
            problems.append(f"{r['unit_id']}: 末尾字母 {m.group(1)} != parsed_label "
                            f"{r.get('parsed_label')}")

    if sorted(x["unit_id"] for x in bad) != sorted(FAILURE_REVIEW):
        problems.append("解析失败集合与逐条审阅记录不一致")

    if problems:
        print("[STOP ] 机械核对不通过 —— 审阅结论与产物对不上：", file=sys.stderr)
        for p in problems:
            print("   ", p, file=sys.stderr)
        return 2

    # ── 五类对照 ─────────────────────────────────────────────────────────
    buckets = {
        "明确作答且被正确识别": len(ok),
        "明确作答但未识别": sum(1 for v in FAILURE_REVIEW.values()
                                if v["made_final_choice"] == "是"),
        "明确作答但识别错误": 0,
        "尚未作答却被判成功": 0,
        "尚未作答且解析失败": sum(1 for v in FAILURE_REVIEW.values()
                                  if v["made_final_choice"] == "否"),
        "无法确定": sum(1 for v in FAILURE_REVIEW.values()
                        if v["made_final_choice"] == "无法确定"),
    }

    doc = {
        "schema": "stage1_g3_budget512_parse_review_v1",
        "date": "2026-09-14",
        "scope": ("**独立审阅，与 gate 结果分开**。由 Claude 逐条阅读 120 条 completion；"
                  "未更换解析器、未重判旧证据、未改变 gate 结论"),
        "annotator": "Claude 文本审阅（**不是**独立人类标注，**不是**正则分类）",
        "input": {"path": str(RUN.relative_to(ROOT)).replace("\\", "/"),
                  "sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
                  "n_rows": len(rows)},
        "parser_result": {"explicit_answer_marker": len(ok), "parse_failure": len(bad),
                          "parse_failure_rate": round(len(bad) / len(rows), 4),
                          "last_isolated_label": 0,
                          "note": ("116 条全部走**第一分支** `explicit_answer_marker`；"
                                   "此前两轮的伪阳性全部来自 `last_isolated_label` 回退，"
                                   "本批**该回退一次未被触发**")},
        "completion_length_tokens": {"min": min(lens), "median": sorted(lens)[len(lens) // 2],
                                     "max": max_len, "budget": BUDGET,
                                     "n_touched_budget": sum(1 for n in lens if n >= BUDGET)},
        "success_review": SUCCESS_REVIEW_VERDICT,
        "mechanical_verification": {
            "single_answer_marker_per_row": True,
            "ends_with_answer_letter": True,
            "letter_equals_parsed_label": True,
            "n_checked": len(ok),
            "note": "上述三项由脚本逐条验证；任一条不成立即退出非零，不放过",
        },
        "failure_review": FAILURE_REVIEW,
        "post_hoc_label_mapping": {
            "when": "**盲标完成之后**",
            "basis": "只用 candidate_choices 选项清单，**不看正确答案**",
            "entries": {k: v for k, v in POST_HOC_MAP.items()},
        },
        "reconciliation_buckets": buckets,
        "conclusion": {
            "false_positives": ("**0 条**。116 条判成功里没有『选项枚举标号被当成答案』，"
                                "也没有『截断词残字』—— 后者在本批无从发生（0 条撞上限）"),
            "false_negatives": ("**2 条**真实答案未被识别："
                                "005120__sampled_4（行内粗体 `**D. WPA2**`）、"
                                "004295__sampled_5（只给内容 'minimum of 4 hosts'，无字母）；"
                                "另有 1 条无法确定"),
            "meaning_for_the_gate": ("gate 用的 `parse_failure_rate` 只计**未被识别**，"
                                     "而这 2 条漏判使该比率**偏高**而非偏低 —— "
                                     "即本次 0.0333 不是被伪阳性撑起来的"),
            "not_claimed": ("不因此更换解析器、不放宽候选规则、不重判既有证据。"
                            "『本批未出现伪阳性』不等于 `last_isolated_label` 回退的"
                            "已知问题已被普遍排除 —— 本批只是没有触发它。"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"parser": doc["parser_result"], "lens": doc["completion_length_tokens"],
                      "buckets": buckets}, ensure_ascii=False, indent=1))
    print("伪阳性:", doc["conclusion"]["false_positives"])
    print("漏判  :", doc["conclusion"]["false_negatives"])
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
