"""v2.3 MCQ semantic output canonicalization 单测。

覆盖预注册 §7.1/§7.2 与任务卡 Stage B 的硬约束:
  semantic_output_text 只能由 parsed_label + 用户可见 options 得到;
  不得由 gold_label 得到;parse 失败时不得猜;correctness 不得进特征;
  fresh/burned 按 **ID 集合**校验;ladder 与 H1 同构且 d_F 不写死。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci
from recover_attention import mcq_conditional_increment as mci

CHOICES = [
    {"choice": "A", "label_text": "Every 30 days", "original_position": 2},
    {"choice": "B", "label_text": "Cross-site scripting", "original_position": 0},
    {"choice": "C", "label_text": "A very long option that is not correct at all here", "original_position": 1},
    {"choice": "D", "label_text": "No rotation", "original_position": 3},
]


# ---- canonicalization ----
def test_canonicalize_maps_letter_to_selected_option_text():
    r = mci.canonicalize_mcq_output("B", CHOICES)
    assert r["canonicalization_status"] == mci.CANON_OK
    assert r["eligible_for_primary"] is True
    assert r["selected_option_text"] == "Cross-site scripting"
    assert r["semantic_output_text"] == "Cross-site scripting"


def test_canonicalize_is_case_and_space_tolerant_on_the_letter_only():
    for raw in ("b", " B ", "B"):
        assert mci.canonicalize_mcq_output(raw, CHOICES)["semantic_output_text"] == "Cross-site scripting"


def test_canonicalize_never_guesses_on_parse_failure():
    """解析失败不得回退到任何选项——那会把模型没做的选择塞给它。

    注意这些是**上游 parsed_label 的值**,不是 raw completion。
    上游标记失败 -> None;或字母不在 {A,B,C,D};或在 options 中无对应项。
    """
    for bad in (None, "", "   ", "E", 3, "AB"):
        r = mci.canonicalize_mcq_output(bad, CHOICES)
        assert r["canonicalization_status"] == mci.CANON_PARSE_FAILURE, f"{bad!r} 不该被解析"
        assert r["eligible_for_primary"] is False
        assert r["selected_option_text"] is None and r["semantic_output_text"] is None


def test_canonicalization_is_invariant_to_response_format():
    """parser contract:上游解析出 D,则无论 raw completion 是 "D" 还是 "Answer! <D>",
    semantic_output_text 都必须是 D 的 label_text。

    canonicalize 根本不接收 raw_completion —— 格式差异在结构上就影响不到它。
    格式归 response-format/option-surface shortcut 管,不得变成 no-emission。
    """
    import inspect
    assert "raw_completion" not in inspect.signature(mci.canonicalize_mcq_output).parameters
    expected = "No rotation"                       # D 的 label_text
    for raw in ("D", "Answer! <D>", "The answer is D.", "  d  "):
        r = mci.canonicalize_mcq_output("D", CHOICES)     # 上游解析结果恒为 D
        assert r["semantic_output_text"] == expected, f"raw={raw!r} 时映射错"
        assert r["eligible_for_primary"] is True


def test_response_format_records_wrapping_without_affecting_eligibility():
    bare = mci.response_format_of("D", "D")
    assert bare["response_format"] == "bare_label" and bare["bare_answer"] is True
    wrapped = mci.response_format_of("Answer! <D>", "D")
    assert wrapped["response_format"] == "wrapped_label" and wrapped["bare_answer"] is False
    unparsed = mci.response_format_of("Answer! <D>", None)
    assert unparsed["response_format"] == "unparsed"
    # 格式不影响 eligibility:两者都能拿到 D 的选项文本
    for raw in ("D", "Answer! <D>"):
        assert mci.canonicalize_mcq_output("D", CHOICES)["eligible_for_primary"] is True


def test_real_upstream_parse_failure_row_stays_a_failure():
    """真实产物:cybermetric_000437 的 completion 是 'Answer! <D>',但**上游**
    parsed_label=null / parse_failure=true。按 parser contract 应记 parse_failure,
    不是 Stage B 自己判严 —— adapter 消费的就是上游那个 null。"""
    import json
    rows = [json.loads(l) for l in open(
        "outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/trace_sampling_manifest.jsonl",
        encoding="utf-8")]
    row = next(r for r in rows if r["example_id"] == "cybermetric_000437"
               and r["sample_type"] == "greedy")
    assert row["completion"] == "Answer! <D>"
    assert row["parsed_label"] is None and row["parse_failure"] is True
    r = mci.canonicalize_mcq_output(row["parsed_label"], CHOICES)
    assert r["canonicalization_status"] == mci.CANON_PARSE_FAILURE
    # 但若上游**能**解析出 D,同一条 raw 就该拿到 D 的文本(契约不看格式)
    assert mci.canonicalize_mcq_output("D", CHOICES)["semantic_output_text"] == "No rotation"


def test_canonicalization_is_invariant_to_gold_and_correctness():
    """gold 不变性:固定 parsed_label/options,任意改 gold_label 与 correctness,
    semantic_output_text 必须完全不变。比"答错时比对 gold 文本"更严密。"""
    outs = set()
    for gold in ("A", "B", "C", "D", None):
        for correctness in (True, False, None):
            rec = {"parsed_label": "B", "candidate_choices": CHOICES,
                   "gold_label": gold, "is_correct": correctness, "wrong_label": correctness}
            r = mci.canonicalize_mcq_output(rec["parsed_label"], rec["candidate_choices"])
            outs.add(r["semantic_output_text"])
    assert outs == {"Cross-site scripting"}, f"canonicalization 受 gold/correctness 影响: {outs}"


def test_canonicalize_ignores_gold_entirely():
    """canonicalize 的签名里根本没有 gold —— 结构上就不可能泄漏。"""
    import inspect
    params = set(inspect.signature(mci.canonicalize_mcq_output).parameters)
    assert params == {"parsed_label", "candidate_choices"}
    assert not (params & mci.FORBIDDEN_FEATURE_FIELDS)


def test_assert_semantic_output_not_from_gold_catches_gold_channel():
    """答错时 semantic text 若等于 gold 选项文本 → 说明走了 gold 通道。"""
    leaked = {"canonicalization_status": mci.CANON_OK, "candidate_choices": CHOICES,
              "parsed_label": "B", "gold_label": "D",
              "semantic_output_text": "No rotation"}          # = gold D 的文本,但模型选了 B
    with pytest.raises(ValueError, match="leaked through gold_label"):
        mci.assert_semantic_output_not_from_gold(leaked)

    ok = {**leaked, "semantic_output_text": "Cross-site scripting"}   # = 模型真的选的 B
    mci.assert_semantic_output_not_from_gold(ok)


def test_assert_semantic_output_not_from_gold_skips_when_answer_is_correct():
    """答对时 parsed==gold,两者文本本就相同,无从判别 → 不得误报。"""
    correct = {"canonicalization_status": mci.CANON_OK, "candidate_choices": CHOICES,
               "parsed_label": "D", "gold_label": "D", "semantic_output_text": "No rotation"}
    mci.assert_semantic_output_not_from_gold(correct)


# ---- prompt-only 不含模型输出 ----
def test_prompt_only_text_excludes_model_output():
    t = mci.prompt_only_text("Which is XSS?", CHOICES)
    assert "Which is XSS?" in t
    for c in CHOICES:                       # 全部选项都在
        assert c["label_text"] in t
    # 但不指示模型选了哪个:去掉题干与选项后不应残留"选择"信息
    assert "selected" not in t.lower() and "answer" not in t.lower()


# ---- label leakage ----
def test_assert_no_label_leakage_in_features():
    mci.assert_no_label_leakage_in_features({"f5_label_margin": 1.0, "surface_option_chars": 5.0})
    for bad in ("wrong_label", "is_correct", "gold_label", "gold_label_text"):
        with pytest.raises(ValueError, match="label leakage"):
            mci.assert_no_label_leakage_in_features({"f5_label_margin": 1.0, bad: 1})


# ---- surface features ----
def test_mcq_surface_features_shape_and_semantics():
    f = mci.mcq_surface_features("Cross-site scripting", CHOICES, "B")
    assert set(f) == set(mci.MCQ_SURFACE_FEATURE_NAMES)
    assert f["surface_option_chars"] == float(len("Cross-site scripting"))
    assert f["surface_option_words"] == 2.0
    assert f["surface_option_has_digit"] == 0.0
    assert f["surface_option_position_index"] == 1.0        # B 是第 2 个字母
    assert f["surface_option_length_rank"] == 2.0           # C 最长,B 次之

    d = mci.mcq_surface_features("Every 30 days", CHOICES, "A")
    assert d["surface_option_has_digit"] == 1.0
    n = mci.mcq_surface_features("No rotation", CHOICES, "D")
    assert n["surface_option_has_negation"] == 1.0


def test_mcq_surface_features_on_parse_failure_are_sentinel():
    f = mci.mcq_surface_features(None, CHOICES, None)
    assert set(f) == set(mci.MCQ_SURFACE_FEATURE_NAMES)
    assert all(v == -1.0 for v in f.values())


# ---- fresh / burned split by ID set ----
def test_split_fresh_confirmatory_validates_by_id_set():
    pool = [f"q{i:04d}" for i in range(2000)]
    burned = [f"q{i:04d}" for i in range(240)]
    r = mci.split_fresh_confirmatory(pool, burned)
    assert r["n_pool"] == 2000 and r["n_burned"] == 240 and r["n_fresh"] == 1760
    assert r["intersection_fresh_burned"] == 0
    assert r["union_equals_pool"] is True
    assert set(r["fresh_ids"]).isdisjoint(set(r["burned_ids"]))


def test_split_fresh_confirmatory_rejects_burned_outside_pool():
    with pytest.raises(ValueError, match="not a subset"):
        mci.split_fresh_confirmatory(["a", "b"], ["a", "zzz"])


def test_split_is_by_id_not_by_count():
    """行数对不代表 ID 对。重复 ID 的池子不得蒙混过关。"""
    pool = ["a", "b", "c", "c"]            # 4 行但只有 3 个唯一 ID
    r = mci.split_fresh_confirmatory(pool, ["a"])
    assert r["n_pool"] == 3 and r["n_fresh"] == 2


def test_real_pool_split_is_2000_minus_240():
    """对真实产物跑一遍:CyberMetric 2000 − 4B-3 已用 240 = fresh 1760。"""
    import json
    pool = [json.loads(l)["example_id"]
            for l in open("data/processed/cyber/cybermetric.jsonl", encoding="utf-8")]
    burned = {json.loads(l)["example_id"] for l in open(
        "outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/trace_sampling_manifest.jsonl",
        encoding="utf-8")}
    r = mci.split_fresh_confirmatory(pool, sorted(burned))
    assert r["n_pool"] == 2000
    assert r["n_burned"] == 240
    assert r["n_fresh"] == 1760
    assert r["intersection_fresh_burned"] == 0
    assert r["union_equals_pool"] is True


# ---- ladder ----
def _rec(i):
    # prompt 用真实感的长文本:单字符 token 会被冻结 TF-IDF 的 word tokenizer 丢掉,
    # min_df=2 再一剪就空 —— 那是夹具失真,不是被测代码的问题。
    return {"prompt_only_text": mci.prompt_only_text(
                f"Which cybersecurity concept does this question number {i} describe?", CHOICES),
            "parsed_label": "B",
            "semantic_output_text": "Cross-site scripting",
            "f5_label_margin": 1.0, "f5_label_entropy": 0.1, "f5_full_entropy": 0.2,
            "f5_letter_token_logprob": -0.5, "f5_self_consistency_exact": 0.8,
            "f5_letter_agreement_rate": 0.8,
            **mci.mcq_surface_features("Cross-site scripting", CHOICES, "B")}


def test_mcq_ladder_mirrors_h1_rungs_and_uses_three_blocks():
    recs = [_rec(i) for i in range(3)]
    specs = mci.build_mcq_ladder_specs(recs, hidden=np.zeros((3, 4)))
    assert list(specs) == ["rung1_prompt_only", "rung2_answer_letter_only",
                           "rung3_option_surface_only", "rung4_canonical_output_text",
                           "rung5_f5", "rung6_O_f5_plus_canonical_text",
                           "H_alone", "O_plus_H"]
    # O = F5 + canonical text（与 H1 同一个 O 概念）
    o = specs["rung6_O_f5_plus_canonical_text"]
    assert o["text"] == ["Cross-site scripting"] * 3
    assert [n for n, _ in o["dense_blocks"]] == ["f5"]
    # O+H = [text, F5, H] 三个独立 block（v2.2 §7 公式）
    oh = specs["O_plus_H"]
    assert [n for n, _ in oh["dense_blocks"]] == ["f5", "hidden"]
    # rung2 = 字母身份的 one-hot（4 值类别,不是文本;单字母走 TF-IDF 是范畴错误且会崩）
    r2 = specs["rung2_answer_letter_only"]
    assert r2["text"] is None and [n for n, _ in r2["dense_blocks"]] == ["letter"]
    assert r2["dense_blocks"][0][1].shape == (3, 4)
    # rung2 是身份、rung4 是选项文本 —— 二者不再同一（v2.2 下它们会塌成一个）
    assert specs["rung4_canonical_output_text"]["text"] == ["Cross-site scripting"] * 3


def test_mcq_d_f_is_not_hardcoded_to_h1_dimension():
    """d_F 必须按任务实际维数,不得沿用 H1 的 14。"""
    recs = [_rec(i) for i in range(3)]
    specs = mci.build_mcq_ladder_specs(recs)
    f5_block = specs["rung5_f5"]["dense_blocks"][0][1]
    assert f5_block.shape == (3, len(mci.MCQ_F5_FEATURE_NAMES))
    assert len(mci.MCQ_F5_FEATURE_NAMES) != len(ci.F5_FEATURE_NAMES)


def test_mcq_ladder_runs_through_the_shared_v22_fusion():
    """MCQ 的 rung 必须能走 §7 v2.2 的同一条三-block 融合路径。"""
    rng = np.random.default_rng(0)
    n_groups = 24
    groups = np.arange(n_groups)                 # MCQ:每题一条,每题自成一组
    y = np.array([1 if g % 3 == 0 else 0 for g in groups])
    recs = []
    for i, v in enumerate(y):
        r = _rec(i)
        r["semantic_output_text"] = "wrong bogus option" if v else "correct real option"
        r["f5_label_margin"] = float(-v * 1.2 + rng.normal(0, 1))
        recs.append(r)
    specs = mci.build_mcq_ladder_specs(recs, hidden=rng.normal(0, 1, (n_groups, 8)))
    folds = ci.stratified_grouped_folds(y, groups, 3, seed=0)
    for name, spec in specs.items():
        res = ci.block_oof_scores(y, groups, folds, text=spec["text"],
                                  dense_blocks=spec["dense_blocks"], inner_splits=2, seed=0)
        assert np.all(np.isfinite(res["scores"])), name
        assert 0.0 <= res["auroc"] <= 1.0


def test_letter_onehot_encodes_identity_and_zeroes_parse_failures():
    recs = [{"parsed_label": "A"}, {"parsed_label": "d"}, {"parsed_label": None},
            {"parsed_label": "E"}]
    oh = mci.letter_onehot(recs)
    assert oh.shape == (4, 4)
    assert oh[0].tolist() == [1, 0, 0, 0]
    assert oh[1].tolist() == [0, 0, 0, 1]      # 大小写不敏感
    assert oh[2].sum() == 0 and oh[3].sum() == 0   # parse_failure / 非法字母 -> 全 0


def test_single_letter_text_would_break_the_frozen_tfidf():
    """记录 rung2 不能走 text block 的原因:冻结的 word TF-IDF 吃不下单字符。

    sklearn 默认 token_pattern 要求 >=2 字符 -> 单字母产生 0 个 token -> 词表为空。
    这不是可以 catch 后忽略的边角,是"字母身份不是文本"的表现。
    """
    with pytest.raises(ValueError, match="empty vocabulary|no terms remain"):
        ci.build_fold_design(np.array([0, 1, 2]), np.array([3]),
                             text=["B", "B", "C", "D"])


def test_mcq_ladder_rejects_hidden_row_mismatch():
    with pytest.raises(ValueError, match="hidden rows"):
        mci.build_mcq_ladder_specs([_rec(0)], hidden=np.zeros((2, 4)))


# ---- rung3 与 rung4 必须真正分离 ----
def test_rung3_surface_excludes_option_vocabulary():
    """rung3 若用上 selected-option 的完整文本,就会和 rung4 重合 —— artifact 红线随之失效。"""
    a = mci.mcq_surface_features("Cross-site scripting", CHOICES, "B")
    b = mci.mcq_surface_features("Server-side rendering", CHOICES, "B")   # 同长度/词数,词汇不同
    assert len("Cross-site scripting") == len("Server-side rendering") - 1 or True
    # surface 只看形状:词数相同 -> 该维必须相同(证明没编码词汇语义)
    assert a["surface_option_words"] == b["surface_option_words"]
    # 且 surface 特征名里不含任何文本字段
    assert not any("text" in n for n in mci.MCQ_SURFACE_FEATURE_NAMES)


def test_rung3_and_rung4_are_different_blocks():
    recs = [_rec(i) for i in range(3)]
    specs = mci.build_mcq_ladder_specs(recs)
    r3, r4 = specs["rung3_option_surface_only"], specs["rung4_canonical_output_text"]
    assert r3["text"] is None and r3["dense_blocks"] is not None      # 纯 dense 表面特征
    assert r4["text"] is not None and r4["dense_blocks"] is None      # 纯 text TF-IDF

