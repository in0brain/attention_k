"""稳定 logits 原语的数学正确性单测 —— 合成张量 + **float64 参考实现**。

锁定四件事，任何一件错都会让 shadow F5 悄悄变成另一个量：
  * causal shift：预测第 j 个 completion token 的是第 `plen + j - 1` 行
  * logsumefp / log_softmax 的数值稳定性
  * entropy 与 rank 的定义
  * MCQ 候选 logit 的取值

以及一条行为约束：**任何路径都不得过滤、替换、填零或跳过非有限值** —— 只能停机。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.generation import stable_logits as sl  # noqa: E402

V, H = 97, 16


class FakeHead(torch.nn.Module):
    def __init__(self, bias: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(V, H, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.randn(V, dtype=torch.float32)) if bias else None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class FakeModel:
    def __init__(self, bias: bool = False):
        self.lm_head = FakeHead(bias)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20260818)
    np.random.seed(20260818)


# ------------------------------------------------------------ causal shift
def test_causal_row_index_matches_frozen_convention():
    """冻结实现是 `row = lp[plen + j - 1]`（backends.py:203）。"""
    assert sl.causal_row_index(10, 0) == 9
    assert sl.causal_row_index(10, 1) == 10
    assert sl.causal_row_index(0, 5) == 4


# ------------------------------------------------------- float32 投影正确性
def test_float32_projection_matches_float64_reference():
    model = FakeModel()
    head = sl.Float32Head(model, torch, vocab_chunk=7)      # 刻意用不整除的块大小
    x = torch.randn(5, H, dtype=torch.float32)
    got = head.project(x).detach().numpy().astype(np.float64)
    ref = x.numpy().astype(np.float64) @ model.lm_head.weight.detach().numpy().astype(np.float64).T
    assert np.allclose(got, ref, rtol=0, atol=1e-4)


def test_float32_projection_includes_bias_when_present():
    model = FakeModel(bias=True)
    head = sl.Float32Head(model, torch, vocab_chunk=13)
    x = torch.randn(3, H, dtype=torch.float32)
    got = head.project(x).detach().numpy().astype(np.float64)
    w = model.lm_head.weight.detach().numpy().astype(np.float64)
    b = model.lm_head.bias.detach().numpy().astype(np.float64)
    assert np.allclose(got, x.numpy().astype(np.float64) @ w.T + b, rtol=0, atol=1e-4)


def test_vocab_chunking_is_transparent():
    model = FakeModel(bias=True)
    x = torch.randn(4, H, dtype=torch.float32)
    a = sl.Float32Head(model, torch, vocab_chunk=V).project(x)
    b = sl.Float32Head(model, torch, vocab_chunk=5).project(x)
    assert torch.equal(a, b) or torch.allclose(a, b, rtol=0, atol=1e-5)


# ------------------------------------------------------ log_softmax / entropy
def test_log_softmax_matches_float64_reference():
    logits = torch.randn(6, V, dtype=torch.float32) * 12.0
    got = sl.log_softmax_rows(logits, torch).numpy().astype(np.float64)
    ref = sl.reference_log_softmax(logits.numpy().astype(np.float64))
    assert np.allclose(got, ref, rtol=0, atol=1e-5)


def test_log_softmax_is_numerically_stable_on_large_logits():
    """大 logit 下朴素 exp 会溢出；稳定实现必须仍然全有限且归一。"""
    logits = torch.full((2, V), 3.0e4, dtype=torch.float32)
    logits[0, 3] = 3.0e4 + 50.0
    lp = sl.log_softmax_rows(logits, torch)
    assert bool(torch.isfinite(lp).all())
    assert np.allclose(lp.exp().sum(dim=-1).numpy(), 1.0, atol=1e-5)


def test_entropy_matches_float64_reference():
    logits = torch.randn(5, V, dtype=torch.float32) * 3.0
    lp = sl.log_softmax_rows(logits, torch)
    stats = sl.row_statistics(lp, [0, 1, 2, 3, 4], torch)
    ref = sl.reference_entropy(sl.reference_log_softmax(
        logits.numpy().astype(np.float64)))
    assert np.allclose(stats["entropy"], ref, rtol=0, atol=1e-4)


def test_target_logprob_matches_gather_semantics():
    logits = torch.randn(4, V, dtype=torch.float32)
    lp = sl.log_softmax_rows(logits, torch)
    ids = [7, 0, V - 1, 42]
    stats = sl.row_statistics(lp, ids, torch)
    ref = sl.reference_log_softmax(logits.numpy().astype(np.float64))
    assert np.allclose(stats["target_logprob"],
                       [ref[i, t] for i, t in enumerate(ids)], rtol=0, atol=1e-5)


# ------------------------------------------------------------------- rank
def test_rank_matches_float64_reference_and_frozen_definition():
    logits = torch.randn(6, V, dtype=torch.float32) * 5.0
    lp = sl.log_softmax_rows(logits, torch)
    ids = [3, 11, 0, 88, 5, 61]
    stats = sl.row_statistics(lp, ids, torch)
    ref = sl.reference_rank(sl.reference_log_softmax(
        logits.numpy().astype(np.float64)), ids)
    assert np.array_equal(stats["rank"], ref)


def test_rank_of_argmax_is_one_and_of_min_is_vocab_size():
    logits = torch.randn(1, V, dtype=torch.float32)
    lp = sl.log_softmax_rows(logits, torch)
    best = int(lp.argmax())
    worst = int(lp.argmin())
    assert int(sl.row_statistics(lp, [best], torch)["rank"][0]) == 1
    assert int(sl.row_statistics(lp, [worst], torch)["rank"][0]) == V


def test_rank_never_degenerates_to_one_on_finite_rows():
    """冻结实现里 NaN target 会让所有比较为 False → rank 静默写成 1。

    稳定路径保证 row 全有限，因此该退化**不可能**发生：
    随机 target 的 rank 不会集体塌成 1。
    """
    logits = torch.randn(64, V, dtype=torch.float32) * 4.0
    lp = sl.log_softmax_rows(logits, torch)
    ids = list(np.random.randint(0, V, size=64))
    ranks = sl.row_statistics(lp, ids, torch)["rank"]
    assert int((ranks == 1).sum()) < 10          # 远低于 64
    assert int(ranks.max()) > 1


# ---------------------------------------------------------- 非有限：只停机
def test_projection_stops_on_nonfinite_input_hidden():
    model = FakeModel()
    head = sl.Float32Head(model, torch, vocab_chunk=11)
    x = torch.randn(3, H, dtype=torch.float32)
    x[1, 2] = float("nan")
    with pytest.raises(sl.NonFiniteStop) as exc:
        head.project(x)
    assert exc.value.diagnostic["stage"] == "input_hidden"
    assert exc.value.diagnostic["n_rows_with_nonfinite"] == 1


def test_projection_stops_on_nonfinite_output():
    model = FakeModel()
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("inf")
    head = sl.Float32Head(model, torch, vocab_chunk=11)
    with pytest.raises(sl.NonFiniteStop) as exc:
        head.project(torch.ones(2, H, dtype=torch.float32))
    assert exc.value.diagnostic["stage"] == "projected_logits"


def test_log_softmax_stops_on_nonfinite():
    logits = torch.randn(2, V, dtype=torch.float32)
    logits[0, 0] = float("nan")
    with pytest.raises(sl.NonFiniteStop):
        sl.log_softmax_rows(logits, torch)


def test_candidate_logits_stops_on_nonfinite():
    lp = torch.randn(1, V, dtype=torch.float32)
    lp[0, 5] = float("nan")
    with pytest.raises(sl.NonFiniteStop) as exc:
        sl.candidate_logits(lp, {"A": 1, "B": 5}, torch)
    assert exc.value.diagnostic["stage"] == "candidate_logit"


def test_candidate_logits_returns_all_labels_when_finite():
    lp = sl.log_softmax_rows(torch.randn(1, V, dtype=torch.float32), torch)
    out = sl.candidate_logits(lp, {"A": 1, "B": 2, "C": 3, "D": 4}, torch)
    assert set(out) == {"A", "B", "C", "D"}
    assert all(np.isfinite(v) for v in out.values())


def test_module_contains_no_filtering_of_nonfinite_values():
    """行为约束的源码级断言：不得出现 nan_to_num / dropna 式的补救。"""
    src = (_REPO_ROOT / "src" / "recover_attention" / "generation"
           / "stable_logits.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    for banned in ("nan_to_num", "masked_fill", "torch.where(torch.isfinite",
                   "if np.isfinite(v)]", "finite_float("):
        assert banned not in code, f"稳定路径不得出现 {banned}"
    assert "NonFiniteStop" in src


def test_schema_constant_is_pinned():
    assert sl.STABLE_PROJECTION_SCHEMA == "stable_logit_projection_f32_v1"

def test_cached_f32_weight_matches_chunked_path():
    """缓存 fp32 权重只影响速度，**不影响数值**。"""
    model = FakeModel(bias=True)
    x = torch.randn(4, H, dtype=torch.float32)
    chunked = sl.Float32Head(model, torch, vocab_chunk=9)
    cached = sl.Float32Head(model, torch, cache_f32_weight=True)
    assert chunked.cache_mode == "chunked"
    assert cached.cache_mode == "cached_f32_weight"
    assert torch.allclose(chunked.project(x), cached.project(x), rtol=0, atol=1e-5)


def test_cached_head_still_stops_on_nonfinite():
    model = FakeModel()
    head = sl.Float32Head(model, torch, cache_f32_weight=True)
    x = torch.randn(2, H, dtype=torch.float32)
    x[0, 0] = float("inf")
    with pytest.raises(sl.NonFiniteStop):
        head.project(x)
