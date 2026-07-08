"""Tests for Sprint 3B-0 representation-level residual intervention."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import representation_intervention as ri  # noqa: E402


def test_beta_negative_is_rejected():
    torch = pytest.importorskip("torch")
    base_hidden = {0: torch.randn(5, 8)}
    with pytest.raises(ValueError):
        ri.compute_injection_vectors(base_hidden, [1, 2], [0], answer_pos=4, beta=-0.1)


def test_injection_norm_is_beta_fraction_of_answer_residual():
    torch = pytest.importorskip("torch")
    h = torch.randn(6, 16)
    base_hidden = {0: h}
    beta = 0.2
    inj = ri.compute_injection_vectors(base_hidden, [1, 2], [0], answer_pos=5, beta=beta)
    ans_norm = float(h[5].norm().item())
    got = float(inj[0].norm().item())
    assert abs(got - beta * ans_norm) < 1e-3


def test_empty_span_and_zero_beta_give_none():
    torch = pytest.importorskip("torch")
    base_hidden = {0: torch.randn(5, 8)}
    assert ri.compute_injection_vectors(base_hidden, [], [0], answer_pos=4, beta=0.2)[0] is None
    assert ri.compute_injection_vectors(base_hidden, [1], [0], answer_pos=4, beta=0.0)[0] is None


def test_hook_injects_at_answer_position_and_removes_cleanly():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class TupleIdentity(nn.Module):
        def forward(self, x):
            return (x,)

    layer = TupleIdentity()
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=nn.ModuleList([layer])))
    hidden = 8
    inj_vec = torch.ones(hidden) * 3.0
    ans_pos = 2
    trace = {"triggered_layers": [], "registered": False, "removed": False}
    handles = ri.register_residual_injection_hooks(model, {0: inj_vec}, ans_pos, trace)
    assert trace["registered"] is True

    x = torch.zeros(1, 4, hidden)
    out = layer(x)[0]
    # answer position changed by the injection; other positions untouched (no contamination)
    assert torch.allclose(out[0, ans_pos], inj_vec)
    assert torch.allclose(out[0, 0], torch.zeros(hidden))
    assert 0 in trace["triggered_layers"]

    ri.remove_hooks(handles)
    out2 = layer(torch.zeros(1, 4, hidden))[0]
    assert torch.allclose(out2[0, ans_pos], torch.zeros(hidden))  # baseline restored after removal


def test_injection_total_norm_positive_when_span_present():
    torch = pytest.importorskip("torch")
    base_hidden = {0: torch.randn(6, 8), 1: torch.randn(6, 8)}
    inj = ri.compute_injection_vectors(base_hidden, [2, 3], [0, 1], answer_pos=5, beta=0.4)
    assert ri.injection_total_norm(inj) > 0.0
