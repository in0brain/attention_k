r"""`_generate(capture=...)` 的观测正确性 —— token 切分、长度统计、停止标记。

**假 tokenizer + 假模型；不加载权重、不跑 GPU。**

要钉住两件事：

    1. capture=None 时行为与从前**逐字节相同**（纯观测，不改生成）
    2. capture 里的切分 / 计数 / 停止标记与实际发生的事一致，
       且**允许** EOS 与长度上限同时成立 —— 不替调用方做唯一归因
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from recover_attention.generation import backends as bk  # noqa: E402

EOS = 7
PROMPT = "<|begin_of_text|>hello"
SPECIAL = {EOS: "<|eot|>", 9: "<|pad|>"}


class Tok:
    def __init__(self):
        self.eos_token_id = EOS
        self.pad_token_id = None

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        ids = [(ord(c) % 50) + 20 for c in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in (int(x) for x in ids):
            if i in SPECIAL:
                if not skip_special_tokens:
                    out.append(SPECIAL[i])
            else:
                out.append(chr((i - 20) % 50))
        return "".join(out)


class Model:
    def __init__(self, new_ids):
        self.new_ids = new_ids

    def generate(self, **kw):
        ids = kw["input_ids"]
        budget = kw["max_new_tokens"]
        emit = self.new_ids[:budget]
        return torch.cat([ids, torch.tensor([emit], dtype=ids.dtype)], dim=-1)


def _backend(new_ids, budget):
    be = bk._RealBackendBase.__new__(bk._RealBackendBase)
    be.ctx = {"torch": torch, "tokenizer": Tok(), "model": Model(new_ids)}
    be.seed, be.max_new_tokens = 4242, budget
    be.temperature, be.top_p = 0.7, 0.95
    be.top_k = bk._RealBackendBase.QWEN_PROTOCOL_TOP_K
    be.repetition_penalty = bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY
    be.tuple_idx = 2
    be._to_device = lambda enc: enc
    return be


def test_capture_none_returns_exactly_what_it_used_to():
    ids = [30, 31, 32]
    a = _backend(ids, 16)._generate(PROMPT, do_sample=False, seed=1)
    cap: dict = {}
    b = _backend(ids, 16)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    assert a == b, "capture 只观测，不得改变返回值"
    assert cap, "传了 capture 就该被填上"


def test_token_slicing_and_counts():
    ids = [30, 31, 32, 33]
    cap: dict = {}
    _backend(ids, 16)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    tok = Tok()
    expect_prompt = tok(PROMPT, add_special_tokens=False)["input_ids"]
    assert cap["prompt_input_ids"] == expect_prompt
    assert cap["prompt_len"] == len(expect_prompt)
    assert cap["generated_ids_raw"] == ids
    assert cap["n_generated_tokens"] == len(ids)
    assert cap["last_generated_token_id"] == ids[-1]


def test_eos_stop_is_flagged_and_stripped_from_the_returned_text():
    cap: dict = {}
    text = _backend([30, 31, EOS], 16)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    assert cap["last_token_is_the_eos_we_passed"] is True
    assert cap["generated_ids_raw"][-1] == EOS            # 原始序列保留 EOS
    assert cap["completion_ids_after_eos_strip"] == [30, 31]
    assert cap["reached_max_new_tokens"] is False
    assert "<|eot|>" not in text                          # 正式文本剥掉了
    assert "<|eot|>" in cap["decoded_skip_special_false"]  # 观测文本留着


def test_length_cap_is_flagged():
    cap: dict = {}
    _backend([30, 31, 32, 33, 34], 3)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    assert cap["n_generated_tokens"] == 3
    assert cap["reached_max_new_tokens"] is True
    assert cap["last_token_is_the_eos_we_passed"] is False


def test_eos_and_length_cap_can_both_be_true_and_neither_is_forced():
    """EOS 与长度上限**可能同时满足** —— 观测层只记事实，不替人做唯一归因。"""
    cap: dict = {}
    _backend([30, 31, EOS], 3)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    assert cap["reached_max_new_tokens"] is True
    assert cap["last_token_is_the_eos_we_passed"] is True
    assert "termination_reason" not in cap, "观测层不得输出单一归因字段"


def test_special_tokens_in_the_middle_are_visible_only_in_the_keep_special_decode():
    """中途出现的特殊 token 在正式文本里被隐藏 —— 这正是要观测的风险。"""
    cap: dict = {}
    text = _backend([30, EOS, 31], 16)._generate(PROMPT, do_sample=False, seed=1, capture=cap)
    assert "<|eot|>" not in text
    assert "<|eot|>" in cap["decoded_skip_special_false"]
    assert cap["last_token_is_the_eos_we_passed"] is False   # 末 token 不是 EOS


def test_effective_kwargs_are_recorded_without_tensors():
    cap: dict = {}
    _backend([30], 16)._generate(PROMPT, do_sample=True, seed=1, capture=cap)
    kw = cap["effective_generate_kwargs"]
    assert "input_ids" not in kw and "attention_mask" not in kw
    assert kw["max_new_tokens"] == 16 and kw["do_sample"] is True
    assert kw["repetition_penalty"] == 1.05 and kw["top_k"] == 20
    assert kw["eos_token_id"] == EOS
    assert kw["pad_token_id"] == EOS, "Llama 的 pad 为 None，应回退成 eos"


def test_budget_is_the_only_difference_between_the_two_conditions():
    """诊断的两个条件只差 max_new_tokens；同一 backend 改属性即可，其余 kw 不变。"""
    ids = list(range(30, 30 + 40))
    caps = []
    be = _backend(ids, 16)
    for budget in (16, 128):
        be.max_new_tokens = budget
        cap: dict = {}
        be._generate(PROMPT, do_sample=True, seed=1, capture=cap)
        caps.append(cap)
    a, b = (dict(c["effective_generate_kwargs"]) for c in caps)
    assert a.pop("max_new_tokens") == 16 and b.pop("max_new_tokens") == 128
    assert a == b, "除 max_new_tokens 外，两条件的有效配置必须逐键相同"
    assert caps[0]["generated_ids_raw"] == ids[:16]
    assert caps[1]["generated_ids_raw"] == ids[:40]
