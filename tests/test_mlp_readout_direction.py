"""Tests for Sprint 3C-2 MLP readout direction analysis helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import mlp_readout_direction as mrd  # noqa: E402


def test_normalize_direction_no_nan_and_unit() -> None:
    torch = pytest.importorskip("torch")
    u = mrd.normalize_direction(torch.tensor([3.0, 4.0]))
    assert float(u.norm().item()) == pytest.approx(1.0)
    z = mrd.normalize_direction(torch.zeros(5))
    assert float(z.norm().item()) == 0.0
    assert not bool(torch.isnan(z).any())


def test_mean_direction_dim_and_value() -> None:
    torch = pytest.importorskip("torch")
    m = mrd.mean_direction([torch.tensor([1.0, 0.0, 0.0]), torch.tensor([3.0, 0.0, 0.0])])
    assert m.shape == (3,)
    assert torch.allclose(m, torch.tensor([2.0, 0.0, 0.0]))


def test_pca_direction_sign_aligned_to_reference() -> None:
    torch = pytest.importorskip("torch")
    deltas = [torch.tensor([2.0, 0.0]), torch.tensor([-2.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    ref = torch.tensor([1.0, 0.0])
    pc1 = mrd.pca_direction(deltas, reference=ref)
    assert mrd.cosine(pc1, ref) >= 0.0
    assert abs(mrd.cosine(pc1, ref)) == pytest.approx(1.0, abs=1e-5)


def test_negative_is_flip() -> None:
    torch = pytest.importorskip("torch")
    m = torch.tensor([1.0, -2.0, 3.0])
    assert torch.allclose(mrd.normalize_direction(-m), -mrd.normalize_direction(m))


def test_rotate_derangement_has_no_fixed_points() -> None:
    perm = mrd.rotate_derangement(6)
    assert sorted(perm) == list(range(6))  # a valid permutation
    assert all(perm[i] != i for i in range(6))  # never reuses the same pair label
    assert mrd.rotate_derangement(1) == [0]  # degenerate n<=1 handled


def test_unembedding_answer_direction_dim() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class FakeTok:
        vocab = {"4": 4, "2": 2, "7": 7}

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [self.vocab.get(c, 9) for c in text.strip()]}

        def convert_ids_to_tokens(self, tid):
            return {v: k for k, v in self.vocab.items()}.get(int(tid), str(tid))

    embed = nn.Embedding(10, 5)
    model = types.SimpleNamespace(get_output_embeddings=lambda: embed)
    d = mrd.unembedding_answer_direction(model, FakeTok(), "42", "7")
    assert d.shape == (5,)
    # direction is W_U[4] - W_U[7]
    assert torch.allclose(d, embed.weight[4].detach().float() - embed.weight[7].detach().float())


def _fake_model():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Mlp(nn.Module):
        def forward(self, x):
            return x

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()

    return types.SimpleNamespace(model=types.SimpleNamespace(layers=nn.ModuleList([Layer()])))


def test_zero_direction_is_noop_and_negative_alpha_rejected() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    trace = {}
    handles = mrd.register_mlp_direction_nudge(model, layer=0, unit_direction=torch.zeros(3), alpha=0.4, target_position=1, trace=trace)
    x = torch.ones(1, 3, 3)
    out = model.model.layers[0].mlp(x)
    assert torch.allclose(out, torch.ones(1, 3, 3))  # zero direction -> no change
    for h in handles:
        h.remove()
    with pytest.raises(ValueError):
        mrd.register_mlp_direction_nudge(model, layer=0, unit_direction=torch.zeros(3), alpha=-0.1, target_position=0, trace={})


def test_nudge_adds_scaled_direction_to_target_only() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    unit = torch.tensor([1.0, 0.0, 0.0])
    trace = {}
    mrd.register_mlp_direction_nudge(model, layer=0, unit_direction=unit, alpha=0.5, target_position=1, trace=trace)
    x = torch.zeros(1, 2, 3)
    x[0, 1] = torch.tensor([3.0, 4.0, 0.0])  # norm 5
    out = model.model.layers[0].mlp(x)
    # out[1] = [3,4,0] + 0.5 * 5 * [1,0,0] = [5.5, 4, 0]
    assert torch.allclose(out[0, 1], torch.tensor([5.5, 4.0, 0.0]))
    assert torch.allclose(out[0, 0], torch.zeros(3))  # non-target untouched
    assert trace["triggered"] == [(0, "mlp_output")]


def test_cosine_and_bootstrap() -> None:
    torch = pytest.importorskip("torch")
    assert mrd.cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == pytest.approx(1.0)
    assert mrd.cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.0)
    boot = mrd.bootstrap_ci([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert boot["mean"] == pytest.approx(1.0)
    assert boot["stable_positive"] is True
