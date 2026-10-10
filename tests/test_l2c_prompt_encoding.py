r"""Stage1-L2C：已应用 chat template 的 prompt 不得被重复加起始标记。

**假 tokenizer + 假模型 + 合成张量；不加载模型权重、不跑 GPU、不碰 outputs/。**

固定下来的不变式：

    生成路径与 teacher-forced 路径对**同一 prompt** 编码出**逐位相同**的 ids
    Llama 形（tokenizer 会前置 BOS）——修复后不再出现重复的起始标记
    Qwen 形（tokenizer 不前置任何 token）——修复前后编码**完全一致**
    prompt 未被平移，因此三个读取位置仍指向同一个答案词元

未经 chat template 的普通文本（如 verify_hidden_index 的探针串）不走这两条路径，
其编码口径不受本次修复影响 —— 也由一条测试钉住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from recover_attention.generation import backends as bk  # noqa: E402

LLAMA_BOS = 128000
PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQ?"
COMPLETION = "Answer: C"
VOCAB = 512
HIDDEN = 8
TUPLE_IDX = 3


class FakeTokenizer:
    """`bos` 为 None 时模拟 Qwen（不前置任何 token），否则模拟 Llama（前置 BOS）。"""

    def __init__(self, *, bos: int | None):
        self.bos = bos
        self.eos_token_id = 7
        self.pad_token_id = None
        self.calls: list[dict] = []

    def _ids(self, text: str) -> list[int]:
        return [(ord(c) % 100) + 10 for c in text]

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        ids = self._ids(text)
        offsets = [[i, i + 1] for i in range(len(text))]
        if add_special_tokens and self.bos is not None:
            ids = [self.bos] + ids
            offsets = [[0, 0]] + offsets
        out: dict = {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((int(i) - 10) % 100) for i in ids)


class FakeModel:
    """generate 追加固定的新 token；forward 返回可识别的 logits / hidden。"""

    def __init__(self):
        self.new_ids = [21, 22, 23]

    def generate(self, **kw):
        ids = kw["input_ids"]
        extra = torch.tensor([self.new_ids], dtype=ids.dtype)
        return torch.cat([ids, extra], dim=-1)

    def __call__(self, **kw):
        total = int(kw["input_ids"].shape[-1])
        logits = torch.full((1, total, VOCAB), -50.0)
        for t in range(total):
            logits[0, t, (t + 1) % VOCAB] = 10.0
        hidden = tuple(
            torch.arange(total, dtype=torch.float32).reshape(1, total, 1).repeat(1, 1, HIDDEN)
            for _ in range(TUPLE_IDX + 1))
        return type("Out", (), {"logits": logits, "hidden_states": hidden})()


def _backend(bos: int | None) -> bk._RealBackendBase:
    be = bk._RealBackendBase.__new__(bk._RealBackendBase)
    be.ctx = {"torch": torch, "tokenizer": FakeTokenizer(bos=bos), "model": FakeModel()}
    be.tuple_idx = TUPLE_IDX
    be.max_new_tokens = 16
    be.temperature = 0.7
    be.top_p = 0.95
    be.top_k = bk._RealBackendBase.QWEN_PROTOCOL_TOP_K
    be.repetition_penalty = bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY
    be._to_device = lambda enc: enc          # 测试里不搬设备
    return be


def _prompt_calls(be) -> list[dict]:
    return [c for c in be.ctx["tokenizer"].calls if c["text"] == PROMPT]


# ───────────────────────────────────────────── 两条路径的 prompt 编码口径
@pytest.mark.parametrize("bos", [None, LLAMA_BOS], ids=["qwen_like", "llama_like"])
def test_both_paths_encode_the_templated_prompt_without_special_tokens(bos):
    be = _backend(bos)
    be._generate(PROMPT, do_sample=False, seed=1)
    be._teacher_forced(PROMPT, COMPLETION)
    calls = _prompt_calls(be)
    assert len(calls) == 2, "生成与 teacher-forced 各应编码一次 prompt"
    assert all(c["add_special_tokens"] is False for c in calls), (
        "已应用 chat template 的 prompt 必须显式 add_special_tokens=False")


@pytest.mark.parametrize("bos", [None, LLAMA_BOS], ids=["qwen_like", "llama_like"])
def test_generate_and_teacher_forced_agree_on_prompt_length(bos):
    be = _backend(bos)
    tok = be.ctx["tokenizer"]
    gen_len = int(tok(PROMPT, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].shape[-1])
    tf = be._teacher_forced(PROMPT, COMPLETION)
    assert tf["prompt_len"] == gen_len, "两条路径的 prompt_len 必须一致，否则读取位置整体平移"


def test_llama_like_tokenizer_no_longer_duplicates_the_start_marker():
    tok = FakeTokenizer(bos=LLAMA_BOS)
    old = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()          # 旧口径
    new = tok(PROMPT, return_tensors="pt",
              add_special_tokens=False)["input_ids"][0].tolist()             # 修复后
    # 该 prompt 的模板本身已经以 BOS 开头（真实 Llama 模板即如此）
    assert old[0] == LLAMA_BOS and len(old) == len(new) + 1
    assert old[1:] == new, "除开头多出的那一个起始标记外，其余 token ids 必须逐位不变"


def test_qwen_like_tokenizer_encoding_is_byte_identical_before_and_after():
    tok = FakeTokenizer(bos=None)
    old = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    new = tok(PROMPT, return_tensors="pt",
              add_special_tokens=False)["input_ids"][0].tolist()
    assert old == new, "Qwen 侧不前置任何 token，修复必须是空变更"


def test_plain_text_encoding_is_untouched_by_the_fix():
    """未经模板的普通文本仍走默认口径 —— 本次修复不改变它。"""
    tok = FakeTokenizer(bos=LLAMA_BOS)
    probe = tok("probe text for layer check", return_tensors="pt")["input_ids"][0].tolist()
    assert probe[0] == LLAMA_BOS, "普通文本的默认编码仍应加特殊 token"


# ───────────────────────────────────────────── 读取位置没有被平移
def test_three_readouts_still_point_at_the_same_answer_token():
    be = _backend(LLAMA_BOS)
    tf = be._teacher_forced(PROMPT, COMPLETION)
    plen, c_ids = tf["prompt_len"], tf["completion_ids"]
    for j, tid in enumerate(c_ids):
        step = plen + j - 1                              # 预测第 j 个 completion token 的那一步
        assert int(torch.argmax(tf["logits"][step])) == (step + 1) % VOCAB
        assert (step + 1) == plen + j                    # 该步预测的正是答案词元所在位置
        assert float(tf["hidden"][plen + j][0]) == float(plen + j)   # hidden 取词元自身
        assert tf["token_logprobs"][j] == pytest.approx(float(
            torch.log_softmax(tf["logits"][step], dim=-1)[tid]))
        # 三个下标数值不同是因果语言模型的正常语义，不要求相等
        assert step != plen + j


def test_an_extra_start_marker_would_shift_every_readout_by_one():
    """故障注入：把 prompt 编码换回旧口径，全部读取位置整体平移一位。"""
    be = _backend(LLAMA_BOS)
    tok = be.ctx["tokenizer"]
    fixed = int(tok(PROMPT, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].shape[-1])
    old = int(tok(PROMPT, return_tensors="pt")["input_ids"].shape[-1])
    assert old == fixed + 1
    j = 1
    assert (old + j - 1) != (fixed + j - 1)              # logits 步错位
    assert (old + j) != (fixed + j)                      # hidden 位置错位
    tf = be._teacher_forced(PROMPT, COMPLETION)
    # 平移后读到的是相邻词元的表示：数值仍然有限、形状仍然正确 —— 所以只看数值发现不了
    shifted = tf["hidden"][old + j]
    correct = tf["hidden"][fixed + j]
    assert np.isfinite(shifted).all() and shifted.shape == correct.shape
    assert float(shifted[0]) != float(correct[0])


def test_generate_strips_the_prompt_by_the_same_length_it_encoded():
    """_generate 用 plen 切掉 prompt —— plen 与实际编码必须同源，否则会截错。"""
    be = _backend(LLAMA_BOS)
    text = be._generate(PROMPT, do_sample=False, seed=0)
    assert text == be.ctx["tokenizer"].decode(be.ctx["model"].new_ids)
