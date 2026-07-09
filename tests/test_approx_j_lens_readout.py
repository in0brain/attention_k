"""Tests for Sprint 3C-4A approximate J-lens readout helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import approx_j_lens_readout as aj  # noqa: E402


class FakeTokenizer:
    vocab = {"0": 0, "1": 1, "2": 2, "3": 3, "x": 4}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(int(i), str(i)) for i in ids)

    def convert_ids_to_tokens(self, token_id):
        inv = {v: k for k, v in self.vocab.items()}
        return inv.get(int(token_id), str(token_id))


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


def test_normalize_direction_handles_zero() -> None:
    torch = pytest.importorskip("torch")
    unit = aj.normalize_direction(torch.tensor([3.0, 4.0]))
    assert float(unit.norm().item()) == pytest.approx(1.0)
    zero = aj.normalize_direction(torch.zeros(3))
    assert float(zero.norm().item()) == pytest.approx(0.0)
    assert not bool(torch.isnan(zero).any())


def test_mlp_output_perturb_hook_target_only() -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model()
    trace = {}
    handles = aj.register_mlp_output_perturb_hook(
        model,
        layer=0,
        direction_vec=torch.tensor([1.0, 0.0, 0.0]),
        target_position=1,
        epsilon=0.5,
        base_norm=4.0,
        trace=trace,
    )
    x = torch.zeros(1, 3, 3)
    out = model.model.layers[0].mlp(x)
    for handle in handles:
        handle.remove()
    assert torch.allclose(out[0, 1], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 0], torch.zeros(3))
    assert trace["patch_records"][0]["perturb_norm"] == pytest.approx(2.0)
    assert trace["patch_records"][0]["non_target_position_contamination_check"] is True


def test_project_vector_to_selected_unembedding_rows() -> None:
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    scores = aj.project_vector_to_token_scores(torch.tensor([3.0, 5.0]), weight, [0, 2])
    assert scores == {0: pytest.approx(3.0), 2: pytest.approx(11.0)}


def test_score_map_features_and_topk_overlap() -> None:
    feats = aj.score_map_features({0: 0.1, 1: 2.0, 2: 1.5}, tokenizer=FakeTokenizer(), number_token_ids=[0, 1, 2], top_k=2)
    assert feats["direct_top1_number_token_id"] == 1
    assert feats["direct_number_margin"] == pytest.approx(0.5)
    overlap = aj.topk_overlap([1, 2, 3], [1, 4, 2], k=3)
    assert overlap["top1_match"] is True
    assert overlap["overlap_rate"] == pytest.approx(2 / 3)


def test_spearman_correlation_from_maps() -> None:
    same = aj.spearman_correlation_from_maps({1: 1.0, 2: 2.0, 3: 3.0}, {1: 10.0, 2: 20.0, 3: 30.0}, [1, 2, 3])
    reverse = aj.spearman_correlation_from_maps({1: 1.0, 2: 2.0, 3: 3.0}, {1: 30.0, 2: 20.0, 3: 10.0}, [1, 2, 3])
    assert same == pytest.approx(1.0)
    assert reverse == pytest.approx(-1.0)


def test_compute_readout_risk_gold_free_formula() -> None:
    features = {
        "approx_j_number_entropy": 2.0,
        "approx_j_number_margin": 0.5,
        "approx_j_projection_sharpness": 0.25,
    }
    risk = aj.compute_readout_risk(features, prefix="approx_j", layer_agreement=1.0, model_answer_agreement=0.0)
    assert risk == pytest.approx(2.0 - 0.5 + 0.0 + 1.0 + 0.75)
