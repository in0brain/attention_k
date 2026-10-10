r"""生成入口与离线特征重提取必须走**同一份实现**。

**不加载模型、不跑 GPU、不碰真实 outputs/。**

2026-09-15 把 `MCQBackend.run_unit` 拆成「`_generate` → `score_completion`」，
好让解析规则修订后能对**已保存的 completion 文本**重新提取特征。拆开之后最大的风险是
两条路悄悄分叉 —— 离线重提取用一套逻辑、正常生成用另一套。这里钉住：

    1. `run_unit` 除了生成，其余一律交给 `score_completion`
    2. `score_completion` **不调用** `generate`
    3. 同一 (prompt, completion) 走两条路得到的记录逐键相同
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.generation import backends as bk  # noqa: E402

HIDDEN_DIM = 8
VOCAB = 256


# ══════════════════════════════════════════════════════════════════════════
# 1 / 2. 静态契约 —— 不需要任何后端
# ══════════════════════════════════════════════════════════════════════════
def test_run_unit_delegates_everything_after_generation():
    src = inspect.getsource(bk.MCQBackend.run_unit)
    assert "self._generate(" in src, "生成仍应发生在 run_unit"
    assert "return self.score_completion(" in src, "其余必须交给 score_completion"
    # run_unit 里不得再自己做解析 / 前向 / 特征
    for forbidden in ("_teacher_forced", "parse_option_answer", "f5_", "canonicalize"):
        assert forbidden not in src, f"run_unit 不应再自己做 {forbidden}"


def test_score_completion_never_generates():
    src = inspect.getsource(bk.MCQBackend.score_completion)
    assert "_generate" not in src and ".generate(" not in src
    # 它必须做完整的打分链路
    for needed in ("parse_option_answer", "_teacher_forced",
                   "canonicalize_mcq_output", "_parsed_letter_char_span"):
        assert needed in src, f"score_completion 少了 {needed}"


def test_score_completion_signature_is_text_in():
    params = list(inspect.signature(bk.MCQBackend.score_completion).parameters)
    assert params[:5] == ["self", "prompt", "completion", "record", "unit"]


# ══════════════════════════════════════════════════════════════════════════
# 3. 两条路产出逐键相同 —— 用假后端，不加载权重
# ══════════════════════════════════════════════════════════════════════════
class _FakeTokenizer:
    """把每个字符当一个 token；id = ord(ch) % VOCAB。够用来验路径一致。"""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        ids = [ord(c) % VOCAB for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        if return_tensors == "pt":
            import torch
            out = {"input_ids": torch.tensor([ids]),
                   "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return out

    def convert_ids_to_tokens(self, token_id):
        return chr(token_id)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


class _StubMCQBackend(bk.MCQBackend):
    """只替换 `_generate` 与 `_teacher_forced`；打分链路仍是真实实现。"""

    def __init__(self, completion: str):
        self._canned = completion
        self.seed = 4242
        self.temperature = 0.7
        self.top_p = 0.95
        self.top_k = 20
        self.repetition_penalty = 1.05
        self.max_new_tokens = 512
        self.tuple_idx = 1
        self.ctx = {"tokenizer": _FakeTokenizer()}
        self.letter_ids_bare = {c: ord(c) % VOCAB for c in "ABCD"}
        self.letter_ids_space = {c: (ord(c) + 100) % VOCAB for c in "ABCD"}
        self.n_generate_calls = 0

    def _generate(self, prompt, *, do_sample, seed, capture=None):
        self.n_generate_calls += 1
        return self._canned

    def _teacher_forced(self, prompt, completion):
        p_ids = [ord(c) % VOCAB for c in prompt]
        c_ids = [ord(c) % VOCAB for c in completion]
        plen, n = len(p_ids), len(c_ids)
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(plen + n, VOCAB)).astype(np.float32)

        class _T:
            def __init__(self, a):
                self._a = a

            def __getitem__(self, i):
                return _T(self._a[i])

            def cpu(self):
                return self

            def numpy(self):
                return self._a

        return {"prompt_len": plen, "completion_ids": c_ids,
                "offsets": [[i, i + 1] for i in range(len(completion))],
                "token_logprobs": {j: float(-0.1 * (j + 1)) for j in range(n)},
                "token_entropies": {j: 0.5 for j in range(n)},
                "first_token_ranks": {j: 1 for j in range(n)},
                "hidden": rng.normal(size=(plen + n, HIDDEN_DIM)).astype(np.float32),
                "logits": _T(logits)}


@pytest.fixture
def record():
    return {"example_id": "q1", "question": "Q?",
            "rendered_prompt": "prompt text",
            "gold_label": "C",
            "candidate_choices": [{"choice": c, "label_text": f"opt{c}",
                                   "label_id": None, "original_position": i}
                                  for i, c in enumerate("ABCD")]}


UNIT = {"unit_id": "q1__greedy_0", "example_id": "q1",
        "sample_type": "greedy", "sample_index": 0}


def test_both_paths_produce_identical_rows(record):
    completion = "Answer: C"
    via_run_unit = _StubMCQBackend(completion).run_unit(UNIT, record)

    offline = _StubMCQBackend(completion)
    seed = bk._stable_seed(offline.seed, record["example_id"],
                           UNIT["sample_type"], UNIT["sample_index"])
    via_offline = offline.score_completion(record["rendered_prompt"], completion,
                                           record, UNIT, seed=seed)

    assert offline.n_generate_calls == 0, "离线重提取不得调用 _generate"
    assert set(via_run_unit) == set(via_offline)
    for key in via_run_unit:
        a, b = via_run_unit[key], via_offline[key]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), key
        else:
            assert a == b, key


def test_offline_path_reflects_the_current_parser(record):
    """喂一条新规则判失败的文本 → 离线路径同样判失败，且不产出 F5/H。"""
    offline = _StubMCQBackend("The options include A.")
    row = offline.score_completion(record["rendered_prompt"], "The options include A.",
                                   record, UNIT, seed=1)
    assert row["parsed_label"] is None
    assert row["eligible_for_primary"] is False
    assert "hidden" not in row and "f5_label_margin" not in row


def test_offline_path_uses_the_parsers_span(record):
    """标签与区间同源这条约束在离线路径上同样成立。"""
    completion = "However, the correct answer is C since it covers A and D."
    offline = _StubMCQBackend(completion)
    row = offline.score_completion(record["rendered_prompt"], completion,
                                   record, UNIT, seed=1)
    assert row["parsed_label"] == "C"
    j = row["letter_token_index_local"]
    assert completion[j] == "C", "读取位置必须落在解析器选中的那个 C 上"
