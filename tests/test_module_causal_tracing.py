"""Tests for Sprint 3C-1 module-level causal tracing helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import module_causal_tracing as mct  # noqa: E402


def _fake_model():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Attn(nn.Module):
        def forward(self, x):
            return (x, None)  # tuple output (attn_output, attn_weights)

    class Mlp(nn.Module):
        def forward(self, x):
            return x  # tensor output

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = Mlp()

        def forward(self, x):
            return (x,)

    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=nn.ModuleList([Layer()])))
    return model


def test_mlp_output_patch_interpolation_target_only_and_remove() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    donor = torch.tensor([1.0, 1.0, 1.0])
    trace = {}
    handles = mct.register_module_patch_hook(model, layer=0, module_type="mlp_output", donor_vec=donor, target_position=2, alpha=0.5, trace=trace)
    x = torch.zeros(1, 4, 3)
    out = model.model.layers[0].mlp(x)
    assert torch.allclose(out[0, 2], torch.tensor([0.5, 0.5, 0.5]))  # interpolation alpha=0.5
    assert torch.allclose(out[0, 0], torch.zeros(3))  # non-target untouched
    assert trace["triggered"] == [(0, "mlp_output")]
    for h in handles:
        h.remove()
    restored = model.model.layers[0].mlp(torch.zeros(1, 4, 3))
    assert torch.allclose(restored, torch.zeros(1, 4, 3))


def test_attention_output_tuple_patch() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    donor = torch.tensor([2.0, 2.0, 2.0])
    trace = {}
    mct.register_module_patch_hook(model, layer=0, module_type="attention_output", donor_vec=donor, target_position=1, alpha=1.0, trace=trace)
    out = model.model.layers[0].self_attn(torch.zeros(1, 3, 3))
    assert isinstance(out, tuple)
    assert torch.allclose(out[0][0, 1], donor)
    assert torch.allclose(out[0][0, 0], torch.zeros(3))


def test_self_patch_is_noop() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    before = torch.tensor([3.0, 4.0, 5.0])
    x = torch.zeros(1, 2, 3)
    x[0, 1] = before
    trace = {}
    mct.register_module_patch_hook(model, layer=0, module_type="mlp_output", donor_vec=before, target_position=1, alpha=1.0, trace=trace)
    out = model.model.layers[0].mlp(x)
    assert torch.allclose(out[0, 1], before)  # donor == recipient -> unchanged
    assert trace["patch_records"][0]["patch_delta_norm"] == pytest.approx(0.0)


def test_negative_alpha_and_bad_module_rejected() -> None:
    model = _fake_model()
    with pytest.raises(ValueError):
        mct.register_module_patch_hook(model, layer=0, module_type="mlp_output", donor_vec=None, target_position=0, alpha=-0.1, trace={})
    with pytest.raises(ValueError):
        mct.register_module_patch_hook(model, layer=0, module_type="bogus", donor_vec=None, target_position=0, alpha=1.0, trace={})


def test_module_vector_bounds() -> None:
    torch = pytest.importorskip("torch")
    captured = {(0, "mlp_output"): torch.arange(12, dtype=torch.float32).reshape(4, 3)}
    assert torch.allclose(mct.module_vector(captured, layer=0, module_type="mlp_output", position=2), captured[(0, "mlp_output")][2])
    assert mct.module_vector(captured, layer=0, module_type="mlp_output", position=9) is None
    assert mct.module_vector(captured, layer=1, module_type="mlp_output", position=0) is None


def test_build_answer_readout_uses_answer_prefix_position() -> None:
    torch = pytest.importorskip("torch")
    text = "Reason then answer.\n#### 42"
    offsets = [[i, i + 1] for i in range(len(text))]
    capture = {"offsets": offsets, "input_ids": torch.arange(len(text)).reshape(1, -1)}
    readout = mct.build_answer_readout({}, text, capture)
    assert readout is not None
    # answer '42' starts at index of '4'; readout position is one before it.
    answer_start = text.index("42")
    assert readout["answer_start"] == answer_start
    assert readout["readout_position"] == answer_start - 1
    assert len(readout["prefix_ids"]) == answer_start


def test_donor_and_site_specificity_reuse_paired_bootstrap() -> None:
    rows = []
    for i in range(1, 8):
        base = {"pair_id": f"p{i}", "position_type": "x", "layer": 20, "alpha": 1.0}
        rows.append({**base, "patch_condition": "correct_donor_patch", "corrected_clean_direction_score": 2.0})
        rows.append({**base, "patch_condition": "random_donor_patch", "corrected_clean_direction_score": 0.5})
        rows.append({**base, "patch_condition": "same_trace_random_position_patch", "corrected_clean_direction_score": 1.0})
    donor = mct.compute_donor_specificity(rows)
    site = mct.compute_site_specificity(rows)
    assert donor["n"] == 7 and donor["mean"] == pytest.approx(1.5) and donor["stable_positive"] is True
    assert site["n"] == 7 and site["mean"] == pytest.approx(1.0) and site["stable_positive"] is True
