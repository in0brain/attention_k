"""Sprint 4D-2 v2.3：MCQ 侧的 semantic output canonicalization 与 ladder。

设计权威:docs/paper/preregistration.md v2.3（sha256 16fa43db…，已冻结）§3 / §7.1 / §7.2。
依据:docs/paper/mcq_asset_audit_and_v2.3_rationale.md。

核心概念（§7.1）:O 在两个任务上是**同一个**概念——"模型可见输出的全部语义内容"。
  H1 :可见输出本身即语义内容 → canonical output text = completion 全文。
  MCQ:字母是**指针**,不是语义内容。实测 has_reasoning_text = 0/240、输出仅一个字母,
      故 canonical output text = **被选中选项的文本**。
这不是"换一个更强的基线",是把同一个 O 概念在指针式输出上正确实例化。

不泄漏（硬约束，本模块的存在理由之一）:
  semantic_output_text 只能由 parsed_label + **用户可见的 options** 得到;
  不得由 gold_label 得到;parsed_label 失败时不得猜选项(那会改变模型的实际输出);
  correctness / wrong_label 不得进入任何特征。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from recover_attention import conditional_increment as ci

# §7.1 冻结的 MCQ F5 特征。d_F 由本列表长度决定,**不写死**（H1 的 d_F=14 与此无关）。
MCQ_F5_FEATURE_NAMES = (
    "f5_label_margin",            # 4C 已有
    "f5_label_entropy",           # 4C 已有
    "f5_full_entropy",            # 4C 已有
    "f5_letter_token_logprob",    # 该任务的 "id-token" = answer-letter token
    "f5_self_consistency_exact",  # 对该题 5 条 sampled trace 求
    "f5_letter_agreement_rate",
)
# §7.1 冻结的 MCQ surface 特征（rung3 shortcut）。刻意不含任何 logprob / hidden。
MCQ_SURFACE_FEATURE_NAMES = (
    "surface_option_chars",
    "surface_option_words",
    "surface_option_has_digit",
    "surface_option_has_negation",
    "surface_option_position_index",
    "surface_option_length_rank",
)
NEGATION_RE = re.compile(r"\b(not|no|never|none|cannot|can't|without|neither|nor|un\w+|in\w*valid)\b", re.I)

CANON_OK = "ok"
CANON_PARSE_FAILURE = "parse_failure"


def _choice_map(candidate_choices: Sequence[dict]) -> dict[str, str]:
    """字母 → 选项文本。只用 prompt 中用户可见的选项,不碰 gold。"""
    out: dict[str, str] = {}
    for c in candidate_choices or []:
        letter = c.get("choice")
        text = c.get("label_text")
        if not isinstance(letter, str) or not isinstance(text, str):
            continue
        out[letter.strip().upper()] = text
    return out


def canonicalize_mcq_output(parsed_label: Any, candidate_choices: Sequence[dict]) -> dict:
    """§7.1：canonical output text = 被选中选项的文本。

    **parser contract**:本函数消费**上游冻结的 parsed_label**,不接收 raw_completion,
    不在 Stage B 另造 parser。故裸字母 "D" 与包裹式 "Answer! <D>" 只要上游解析出同一个
    parsed_label,就必须得到同一个 selected option text —— 格式差异归 §7.1 的
    response-format/option-surface shortcut 管,不得变成 no-emission。

    parse_failure 仅限:上游标记失败(parsed_label 为 None/空)、parsed_label 不在
    {A,B,C,D}、或 parsed_label 在 candidate_choices 中无对应项。
    绝不在解析失败时回退到"概率最高的选项"——那会把模型没做的选择塞给它,
    改变其实际输出。
    """
    cmap = _choice_map(candidate_choices)
    if not isinstance(parsed_label, str) or not parsed_label.strip():
        return {"canonicalization_status": CANON_PARSE_FAILURE,
                "eligible_for_primary": False, "selected_option_text": None,
                "semantic_output_text": None, "reason": "parsed_label missing/empty"}
    letter = parsed_label.strip().upper()
    if letter not in cmap:
        return {"canonicalization_status": CANON_PARSE_FAILURE,
                "eligible_for_primary": False, "selected_option_text": None,
                "semantic_output_text": None,
                "reason": f"parsed_label {letter!r} not in options {sorted(cmap)}"}
    text = cmap[letter]
    return {"canonicalization_status": CANON_OK, "eligible_for_primary": True,
            "selected_option_text": text, "semantic_output_text": text, "reason": None}


def response_format_of(raw_completion: Any, parsed_label: Any) -> dict:
    """记录**格式**差异,不参与 canonicalization。

    裸字母 "D" 与包裹式 "Answer! <D>" 在语义上是同一个指针;它们的差别属于
    response-format/option-surface shortcut 该控制的东西,不该影响 eligibility。
    本函数只描述格式,不做解析——解析是上游的事(见 canonicalize_mcq_output 的 docstring)。
    """
    raw = raw_completion if isinstance(raw_completion, str) else ""
    lab = parsed_label.strip().upper() if isinstance(parsed_label, str) else None
    bare = bool(lab) and raw.strip().upper() == lab
    if lab is None:
        fmt = "unparsed"
    elif bare:
        fmt = "bare_label"
    else:
        fmt = "wrapped_label"
    return {"response_format": fmt, "bare_answer": bare,
            "raw_completion_chars": len(raw)}


def prompt_only_text(question: str, candidate_choices: Sequence[dict]) -> str:
    """rung1：题干 + 全部选项文本。**不含模型输出**（不指示选了哪个）。"""
    parts = [str(question or "")]
    for c in sorted(candidate_choices or [], key=lambda x: str(x.get("choice"))):
        parts.append(f"{c.get('choice')}. {c.get('label_text')}")
    return "\n".join(parts)


def mcq_surface_features(selected_option_text: str | None,
                         candidate_choices: Sequence[dict],
                         parsed_label: str | None) -> dict[str, float]:
    """rung3：被选项的表面格式（§7.1 冻结）。

    与 H1 的 surface-only 同构:只看输出的"形状",不看内容语义、不看置信。
    若它就能解掉任务 → artifact 红线触发 → 该任务被格式模式解决。
    """
    if not selected_option_text:
        return {k: -1.0 for k in MCQ_SURFACE_FEATURE_NAMES}
    texts = [str(c.get("label_text") or "") for c in candidate_choices or []]
    lengths = sorted((len(t) for t in texts), reverse=True)
    rank = lengths.index(len(selected_option_text)) + 1 if len(selected_option_text) in lengths else -1
    letters = sorted(_choice_map(candidate_choices))
    pos = letters.index(parsed_label.strip().upper()) if (
        parsed_label and parsed_label.strip().upper() in letters) else -1
    return {
        "surface_option_chars": float(len(selected_option_text)),
        "surface_option_words": float(len(selected_option_text.split())),
        "surface_option_has_digit": float(any(ch.isdigit() for ch in selected_option_text)),
        "surface_option_has_negation": float(bool(NEGATION_RE.search(selected_option_text))),
        "surface_option_position_index": float(pos),
        "surface_option_length_rank": float(rank),
    }


# 任何特征记录中都不得出现的字段:correctness 一旦混进特征,AUROC 直接失去意义。
FORBIDDEN_FEATURE_FIELDS = {
    "wrong_label", "is_correct", "correctness", "gold_label", "gold_label_id",
    "gold_label_text", "label",
}


def assert_no_label_leakage_in_features(feature_row: dict) -> None:
    """特征行不得含 correctness / gold。标签只能活在 y 里,不能活在 X 里。"""
    bad = sorted(set(str(k) for k in feature_row) & FORBIDDEN_FEATURE_FIELDS)
    if bad:
        raise ValueError(f"label leakage into MCQ feature row: {bad}")


def assert_semantic_output_not_from_gold(record: dict) -> None:
    """canonical text 必须来自 parsed_label,不是 gold_label。

    这条在 record 层再验一次:若模型答错(parsed≠gold)而 semantic_output_text 恰好等于
    gold 选项的文本,说明 canonicalization 走了 gold 通道 → 直接报错。
    """
    if record.get("canonicalization_status") != CANON_OK:
        return
    cmap = _choice_map(record.get("candidate_choices") or [])
    gold = record.get("gold_label")
    parsed = record.get("parsed_label")
    if not (isinstance(gold, str) and isinstance(parsed, str)):
        return
    if gold.strip().upper() == parsed.strip().upper():
        return          # 答对时两者本就相同,无从判别,跳过
    gold_text = cmap.get(gold.strip().upper())
    if gold_text is not None and record.get("semantic_output_text") == gold_text:
        raise ValueError(
            f"semantic_output_text equals the GOLD option text while parsed={parsed!r} "
            f"!= gold={gold!r} — canonicalization leaked through gold_label")


# ------------------------------------------------------- fresh / burned split
def split_fresh_confirmatory(all_ids: Sequence[str], burned_ids: Sequence[str]) -> dict:
    """§3 v2.3：confirmatory = 总池 − 已烧。**按 ID 集合校验,不靠行数**。

    烧掉的 240 题(4B-3/4C 用过)其 CI 正是促成 v2.3 修订的依据,再用于确证 =
    同一批数据既定设计又下结论。故必须排除,并降级为 exploratory。
    """
    all_s, burned_s = set(all_ids), set(burned_ids)
    if not burned_s <= all_s:
        raise ValueError(f"burned ids not a subset of the pool: "
                         f"{len(burned_s - all_s)} unknown ids, e.g. {sorted(burned_s - all_s)[:3]}")
    fresh = all_s - burned_s
    report = {
        "n_pool": len(all_s), "n_burned": len(burned_s), "n_fresh": len(fresh),
        "intersection_fresh_burned": len(fresh & burned_s),
        "union_equals_pool": (fresh | burned_s) == all_s,
        "fresh_ids": sorted(fresh), "burned_ids": sorted(burned_s),
    }
    if report["intersection_fresh_burned"] != 0:
        raise RuntimeError("fresh ∩ burned != 0")
    if not report["union_equals_pool"]:
        raise RuntimeError("fresh ∪ burned != pool")
    return report


# ------------------------------------------------------------- ladder (§7.1)
MCQ_LETTERS = ("A", "B", "C", "D")


def letter_onehot(records: Sequence[dict]) -> np.ndarray:
    """rung2「仅字母身份」= 4 值类别 → one-hot dense block。

    为什么不是 text block:§7 冻结的 TF-IDF 是 word ngram(1,2) + min_df=2,而 sklearn
    的默认 word token_pattern 要求 ≥2 个字符 —— 单字母 "B" 产生**零个 token**,词表为空
    直接抛 "After pruning, no terms remain"。更本质的是,"字母身份"本就是 4 值类别,
    不是文本;拿 TF-IDF 编码单字符是范畴错误,即使不崩也只是绕路得到 one-hot。
    §7.1 冻结的原文是"仅字母身份 {A,B,C,D}",未规定 block 类型 → one-hot 是忠实实现,
    不是设计变更。parse_failure 的行全 0(无身份可言)。
    """
    out = np.zeros((len(records), len(MCQ_LETTERS)), dtype=float)
    for i, r in enumerate(records):
        lab = r.get("parsed_label")
        if isinstance(lab, str) and lab.strip().upper() in MCQ_LETTERS:
            out[i, MCQ_LETTERS.index(lab.strip().upper())] = 1.0
    return out


def build_mcq_ladder_specs(records: Sequence[dict], hidden: np.ndarray | None = None) -> dict[str, dict]:
    """§7.1 的 MCQ 阶梯,与 H1 同构、rung 语义对应。

    1 prompt-only          题干+全部选项(不含模型输出)
    2 answer-letter-only   仅字母身份           ← 对应 H1 的 id-string-only
    3 option-surface-only  被选项的表面格式      ← 对应 H1 的 surface-only
    4 full-response-text   被选中选项的文本(= canonical output text)
    5 F5_MCQ
    6 O = F5_MCQ + canonical output text
    融合公式复用 §7 v2.2 的三-block 等权（text / F5 / H 各自独立 block）。
    d_F 由 MCQ_F5_FEATURE_NAMES 决定,不写死。
    """
    f5 = ci._dense_matrix(records, MCQ_F5_FEATURE_NAMES)
    surface = ci._dense_matrix(records, MCQ_SURFACE_FEATURE_NAMES)
    prompt_text = [str(r.get("prompt_only_text", "")) for r in records]
    canon_text = [str(r.get("semantic_output_text", "")) for r in records]

    specs: dict[str, dict] = {
        "rung1_prompt_only": {"text": prompt_text, "dense_blocks": None},
        "rung2_answer_letter_only": {"text": None,
                                     "dense_blocks": [("letter", letter_onehot(records))]},
        "rung3_option_surface_only": {"text": None, "dense_blocks": [("surface", surface)]},
        "rung4_canonical_output_text": {"text": canon_text, "dense_blocks": None},
        "rung5_f5": {"text": None, "dense_blocks": [("f5", f5)]},
        "rung6_O_f5_plus_canonical_text": {"text": canon_text, "dense_blocks": [("f5", f5)]},
    }
    if hidden is not None:
        h = np.asarray(hidden, dtype=float)
        if h.shape[0] != len(records):
            raise ValueError(f"hidden rows {h.shape[0]} != records {len(records)}")
        specs["H_alone"] = {"text": None, "dense_blocks": [("hidden", h)]}
        specs["O_plus_H"] = {"text": canon_text,
                             "dense_blocks": [("f5", f5), ("hidden", h)]}
    return specs
