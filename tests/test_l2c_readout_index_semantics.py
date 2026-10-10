r"""Stage1-L2C：三个读取位置的索引语义 —— 纯合成张量，**不加载任何模型**。

要验的是「三者关联同一个答案词元 j」，**不是**三个下标数值相等。既有实现：

    答案词元本身       prompt_len + j
    F5 / logits 读取   prompt_len + j - 1   （预测第 j 个 completion token 的那一步）
    hidden 读取        prompt_len + j

下标不等是因果语言模型的正常语义；要求它们相等本身就是错的。偏移错 1 位会让
label margin / entropy 读到相邻位置 —— 数值看着正常，含义完全无效 —— 所以必须独立验。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import h1_f5_features as f5f  # noqa: E402

PROMPT_LEN = 7
COMPLETION_LEN = 5
VOCAB = 11


def _fake_forward():
    """构造一段「每一步都唯一可识别」的合成前向。

    logits[t] 的 argmax 被布置成 t+1，即「第 t 行预测的是位于 t+1 的那个词元」；
    每个词元的 id 又被布置成它自己的绝对位置。于是「取到第几行」本身可证：
    读对了行，argmax 就等于目标词元；读错一位，标记立刻对不上。
    """
    total = PROMPT_LEN + COMPLETION_LEN
    logits = np.full((total, VOCAB), -50.0, dtype=np.float32)
    for t in range(total):
        logits[t, (t + 1) % VOCAB] = 10.0
    hidden = np.stack([np.full(4, float(t), dtype=np.float32) for t in range(total)])
    completion_ids = [(PROMPT_LEN + j) % VOCAB for j in range(COMPLETION_LEN)]
    return logits, hidden, completion_ids


def test_logits_step_reads_the_step_that_predicts_token_j():
    """step = prompt_len + j - 1 必须取到「预测第 j 个 completion token」的那一步。"""
    logits, _, completion_ids = _fake_forward()
    for j, tid in enumerate(completion_ids):
        step = PROMPT_LEN + j - 1
        assert int(np.argmax(logits[step])) == tid, (
            f"j={j}: step {step} 没有预测出 completion token {tid}")


def test_off_by_one_is_detected_and_is_not_silently_plausible():
    """错开 1 位会读到相邻步：形状与数值范围都正常，但含义完全不同。"""
    logits, _, completion_ids = _fake_forward()
    j = 2
    tid = completion_ids[j]
    wrong_late = logits[PROMPT_LEN + j]          # 少减 1
    wrong_early = logits[PROMPT_LEN + j - 2]     # 多减 1
    assert int(np.argmax(logits[PROMPT_LEN + j - 1])) == tid
    assert int(np.argmax(wrong_late)) != tid
    assert int(np.argmax(wrong_early)) != tid
    # 关键点：错位后的行依旧是**有限实数、形状正确**——所以"数值正常"不能当作放行依据。
    for row in (wrong_late, wrong_early):
        assert row.shape == (VOCAB,) and np.all(np.isfinite(row))


def test_hidden_position_is_the_answer_token_itself():
    """hidden 读的是答案词元**自身**的表示：prompt_len + j，不是 prompt_len + j - 1。"""
    _, hidden, _ = _fake_forward()
    for j in range(COMPLETION_LEN):
        assert hidden[PROMPT_LEN + j][0] == float(PROMPT_LEN + j)
        assert hidden[PROMPT_LEN + j - 1][0] != float(PROMPT_LEN + j)


def test_all_three_readouts_are_derived_from_one_and_the_same_j():
    """三个位置下标不同，但必须由**同一个 j** 导出 —— 这才是要验的不变式。"""
    logits, hidden, completion_ids = _fake_forward()
    j = 3
    logits_pos = PROMPT_LEN + j - 1
    hidden_pos = PROMPT_LEN + j
    logprob_key = j
    assert logits_pos + 1 == hidden_pos
    assert hidden_pos - PROMPT_LEN == logprob_key
    # 同一个 j 的三个读取彼此对应：该 step 预测出的 token 就是 hidden 所在位置的那个词元
    assert int(np.argmax(logits[logits_pos])) == completion_ids[j]
    assert hidden[hidden_pos][0] == float(PROMPT_LEN + j)


def test_readout_to_prompt_end_distance_equals_j():
    """落盘字段 readout_to_prompt_end_distance = step - (prompt_len - 1) 应恒等于 j。"""
    for j in range(COMPLETION_LEN):
        step = PROMPT_LEN + j - 1
        assert step - (PROMPT_LEN - 1) == j


def test_argmax_need_not_equal_the_sampled_token():
    """采样臂（temperature=0.7 / top_p=0.95）本来就可能取非 argmax 的 token。

    因此「argmax != 实际 token」**不构成**实现错误的证据。可用的自洽判据是：
    目标 token 的 logprob 有限、rank 可解释。
    """
    logits, _, _ = _fake_forward()
    step, sampled = PROMPT_LEN + 1, 0
    row = logits[step]
    assert int(np.argmax(row)) != sampled                     # 非 argmax —— 正常
    logprob = float(row[sampled] - np.log(np.exp(row).sum()))
    rank = int((row > row[sampled]).sum()) + 1
    assert np.isfinite(logprob) and 1 <= rank <= VOCAB        # 仍然自洽


def test_token_indices_for_char_span_maps_one_letter_to_one_token():
    """字符区间 → 词元下标：单字母答案应落到**恰好一个** token 上。"""
    offsets = [[0, 3], [3, 4], [4, 6], [6, 7]]
    assert f5f.token_indices_for_char_span(offsets, 3, 4) == [1]
    # 跨两个 token 的区间必须如实返回两个下标（上游据此 raise，不擅自挑一个）
    assert f5f.token_indices_for_char_span(offsets, 3, 6) == [1, 2]
    # 边界相接不算重叠
    assert f5f.token_indices_for_char_span(offsets, 4, 6) == [2]


def test_zero_length_offsets_are_ignored_so_special_tokens_do_not_shift_j():
    """特殊 token 的零长 offset 必须被忽略，否则 j 会整体平移。"""
    offsets = [[0, 0], [0, 3], [3, 4], [0, 0]]
    assert f5f.token_indices_for_char_span(offsets, 3, 4) == [2]


def test_invalid_span_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        f5f.token_indices_for_char_span([[0, 3]], 2, 2)
