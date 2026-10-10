r"""MCQ 答案提取与定位的规则（2026-09-15 第二次修订）。

**不加载模型、不跑 GPU、不碰真实 outputs/。**

三条可靠规则，别的一律失败：

    1. 标准答案格式   `Answer: C` / `Final answer：C`（中英文冒号、空格、
                      大小写、Markdown 包裹）
    2. 明确的选择语句 "the correct answer is C" / "I would choose option B"，
                      且**整句**不含否定、条件假设、引用、指令或选项讨论
    3. 裸字母回答     整条回答只有一个候选字母及装饰（`C` / `[C]` / `**C**`）

已**取消**"全文末尾孤立字母可作为答案"的推断 —— 提到一个选项不等于选择它。
冲突**统一检查**：标准标记与选择语句里的有效声明放一起看，标签不一致即失败，
不凭分支优先级或"最后一次出现"推断改口。

第一版修复遗漏的四类（Codex 实测）在 `test_the_four_defects_*` 里逐条钉住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import domain_label_proxy as dlp  # noqa: E402
from recover_attention.generation import backends as bk  # noqa: E402

L = list("ABCD")


def parse(text):
    return dlp.parse_option_answer(text, L)


def label(text):
    return parse(text)["parsed_label"]


# ══════════════════════════════════════════════════════════════════════════
# 0. 第一版遗漏的四类缺陷
# ══════════════════════════════════════════════════════════════════════════
def test_the_four_defects_negated_imperative():
    """"Do not choose A." —— 否定在匹配片段**之外**，第一版只查片段，漏掉了。"""
    assert label("Do not choose A.") is None


def test_the_four_defects_conditional_hypothesis():
    r = parse("If I choose A, the result would be wrong.")
    assert r["parsed_label"] is None
    assert r["parse_failure_reason"] == "rejected_by_sentence_context"
    assert r["rejected_claims"][0]["rejected_because"] in ("conditional", "negation")


def test_the_four_defects_mentioning_an_option_is_not_choosing_it():
    """"The options include A." —— 第一版靠"末尾孤立字母"判成 A。该回退已取消。"""
    assert label("The options include A.") is None


def test_the_four_defects_marker_does_not_mask_a_later_conflict():
    """"Answer: A. Actually, the correct answer is C." —— 第一版 `Answer:` 分支直接返回 A。"""
    r = parse("Answer: A. Actually, the correct answer is C.")
    assert r["parsed_label"] is None
    assert r["parse_failure_reason"] == "contradictory_answer_claims"
    assert {c["label"] for c in r["answer_claims"]} == {"A", "C"}


# ══════════════════════════════════════════════════════════════════════════
# 1. 标准答案格式
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text, want", [
    ("Answer: C", "C"),
    ("Final answer: C", "C"),
    ("Answer：C", "C"),                    # 中文冒号
    ("Final answer：C", "C"),
    ("answer:   c", "C"),                  # 大小写与多余空格
    ("Answer: **A**", "A"),
    ("Answer: __A__", "A"),
    ("Reasoning.\n\nAnswer: B", "B"),
])
def test_standard_answer_format(text, want):
    r = parse(text)
    assert r["parsed_label"] == want
    assert r["parse_method"] == "explicit_answer_marker"
    # 区间必须对着**原文**
    assert text[r["answer_char_span"][0]:r["answer_char_span"][1]].upper() == want


# ══════════════════════════════════════════════════════════════════════════
# 2. 裸字母回答（Qwen 的 9,799 条走这条路）
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text", ["C", " C ", "C.", "\nC\n", "**C**", "[C]", "(C)", "C)"])
def test_bare_letter(text):
    r = parse(text)
    assert r["parsed_label"] == "C"
    assert r["parse_method"] == "bare_letter"
    assert text[r["answer_char_span"][0]:r["answer_char_span"][1]].upper() == "C"


# ══════════════════════════════════════════════════════════════════════════
# 3. 明确的选择语句
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text, want", [
    ("Therefore, the correct answer is A.", "A"),
    ("Considering these points, the best answer is: C", "C"),
    ("Based on this analysis, the correct answer is B. IETF.", "B"),
    ("the correct answer is **D. WPA2**.", "D"),
    ("the correct answer is __B__", "B"),
    ("I will choose option B.", "B"),
    ("I'd go with C.", "C"),
    ("We select option D.", "D"),
])
def test_answer_statements(text, want):
    r = parse(text)
    assert r["parsed_label"] == want
    assert r["parse_method"] == "answer_statement"
    assert text[r["answer_char_span"][0]:r["answer_char_span"][1]].upper() == want


def test_statement_beats_letters_in_the_following_explanation():
    """实测例：最终选择是 C，句尾还列举了 A 和 D。"""
    text = ("However, the correct answer is C since it includes all provided answers "
            "which are A and D.")
    r = parse(text)
    assert r["parsed_label"] == "C"
    assert text[r["answer_char_span"][0]:r["answer_char_span"][1]] == "C"


# ══════════════════════════════════════════════════════════════════════════
# 4. 句级排除：否定 / 条件 / 引用 / 指令 / 选项讨论
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text, why", [
    ("Option A is not the correct answer.", "negation"),
    ("The answer is not B.", "negation"),
    ("The correct answer wouldn't be C.", "negation"),
    ("The correct answer is D rather than A.", "negation"),
    ("If the answer is A, the rest follows.", "conditional"),
    ("Assuming the correct answer is B, we proceed.", "conditional"),
    ("Unless the answer is C, this fails.", "conditional"),
    ("Please choose A.", "instruction"),
    ("You should choose A.", "instruction"),
    ('The prompt said "the correct answer is A".', "quotation"),
])
def test_sentence_level_disqualifiers(text, why):
    r = parse(text)
    assert r["parsed_label"] is None, r
    assert dlp.disqualifying_context(text) == why


@pytest.mark.parametrize("text", [
    "The options include A.",
    "The options are A and B.",
    "The available choices include C.",
])
def test_option_discussion_is_not_an_answer(text):
    assert label(text) is None


def test_a_disqualified_sentence_does_not_block_a_clean_one():
    text = "The answer is not B. Therefore, the correct answer is D."
    assert label(text) == "D"


# ══════════════════════════════════════════════════════════════════════════
# 5. 枚举、残字、只给选项内容
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text", [
    "To answer this question, let's analyze each option step by step:\n\nA.",
    "Let's consider the options:\n\n- Option A:",
    "Given the options:\n\nA. 3\nB. 5",
    "Considering the definitions:\n\n**A**",
    "This aligns with the Payment Card Industry Data Security Standard (PCI D",
    "…would be the Chief Information Officer (D).",
    "So, there must be a minimum of 4 hosts identified by the ping scan.",
    'The term "spoofing" best describes impersonating a reputable entity.',
])
def test_no_reliable_claim_means_failure(text):
    r = parse(text)
    assert r["parsed_label"] is None
    assert r["answer_char_span"] is None


# ══════════════════════════════════════════════════════════════════════════
# 6. 跨表达形式的冲突
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("text", [
    "Answer: A\nActually, Answer: B",                      # 标记 vs 标记
    "Answer: A. Actually, the correct answer is C.",        # 标记 vs 语句
    "The correct answer is A. On reflection, the answer is B.",   # 语句 vs 语句
    "I will choose option A. The correct answer is B.",     # 动作 vs 语句
])
def test_conflicting_claims_fail(text):
    r = parse(text)
    assert r["parsed_label"] is None
    assert r["parse_failure_reason"] == "contradictory_answer_claims"


def test_agreeing_claims_do_not_conflict():
    """同一个标签出现多次不是冲突；此时用标准标记的区间。"""
    text = "The correct answer is C.\n\nAnswer: C"
    r = parse(text)
    assert r["parsed_label"] == "C"
    assert r["parse_method"] == "explicit_answer_marker"
    assert text[r["answer_char_span"][0]:r["answer_char_span"][1]] == "C"
    assert len(r["answer_claims"]) == 2


def test_a_disqualified_claim_does_not_create_a_conflict():
    """被句级排除项挡掉的声明不参与冲突判定，只记在 rejected_claims 里。

    这里第一句是条件假设（"If the answer is A…"），它**不**与第二句构成冲突。
    """
    text = "If the answer is A, everything else follows. The correct answer is D."
    r = parse(text)
    assert r["parsed_label"] == "D"
    assert [c["label"] for c in r["rejected_claims"]] == ["A"]
    assert r["rejected_claims"][0]["rejected_because"] == "conditional"


def test_negation_inside_the_matched_fragment_yields_no_claim_at_all():
    """"The answer is not A." 连声明都构不成 —— 系动词后面不是字母，正则就不匹配。

    记在这里是为了说明：否定被挡掉有两条路（正则不匹配 / 句级排除），
    两条都走得通，不要以为只有句级排除在起作用。
    """
    r = parse("The answer is not A. Therefore, the correct answer is D.")
    assert r["parsed_label"] == "D"
    assert [c["label"] for c in r["answer_claims"]] == ["D"]


# ══════════════════════════════════════════════════════════════════════════
# 7. 标签与位置同源
# ══════════════════════════════════════════════════════════════════════════
CASES = [
    "Answer: C",
    "Final answer：C",
    "C",
    "**C**",
    "However, the correct answer is C since it covers all cases.",
    "Considering these points, the correct answer is **D. WPA2**.",
]


@pytest.mark.parametrize("text", CASES)
def test_span_content_equals_the_parsed_label(text):
    r = parse(text)
    span = r["answer_char_span"]
    assert text[span[0]:span[1]].upper() == r["parsed_label"]


@pytest.mark.parametrize("text", CASES)
def test_locator_returns_the_parsers_own_span(text):
    r = parse(text)
    cs, ce = bk._parsed_letter_char_span(text, r["parsed_label"])
    assert [cs, ce] == r["answer_char_span"]
    assert (cs, ce) == bk._parsed_letter_char_span(text, r["parsed_label"], r)


def test_locator_refuses_a_label_the_parser_did_not_choose():
    """结构性保证：不可能出现"解析选 C、定位取别处的 D"。"""
    text = "However, the correct answer is C since it covers A and D."
    with pytest.raises(ValueError, match="disagrees"):
        bk._parsed_letter_char_span(text, "D")


def test_locator_fails_when_the_parser_found_nothing():
    with pytest.raises(ValueError):
        bk._parsed_letter_char_span("no letter here at all", "A")


def test_span_is_relative_to_the_original_text_not_a_normalised_copy():
    """前面有全角/换行/Markdown 时，区间仍必须落在原文的那个字母上。"""
    text = "推理过程……\n\n**Final answer：B**"
    r = parse(text)
    assert r["parsed_label"] == "B"
    assert text[r["answer_char_span"][0]] == "B"
