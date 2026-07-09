"""Tests for Sprint 3C-3 MLP readout attribution helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import mlp_readout_attribution as mra  # noqa: E402


class FakeTokenizer:
    vocab = {" ": 0, "Ġ": 1, "4": 4, "2": 2, "42": 42, "abc": 8, "7": 7, "-3": 13}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(int(i), str(i)) for i in ids)

    def convert_ids_to_tokens(self, token_id):
        inv = {v: k for k, v in self.vocab.items()}
        return inv.get(int(token_id), str(token_id))


def test_number_like_token_filter_excludes_whitespace() -> None:
    ids = mra.filter_number_like_tokens(FakeTokenizer())
    assert 0 not in ids
    assert 1 not in ids
    assert {2, 4, 7, 13, 42}.issubset(set(ids))
    assert 8 not in ids


def test_project_to_unembedding_and_projection_features() -> None:
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    scores = mra.project_to_unembedding(torch.tensor([3.0, 5.0]), weight)
    assert scores.tolist() == pytest.approx([3.0, 5.0, 6.0])

    tok = FakeTokenizer()
    feats = mra.extract_projection_features(scores, tokenizer=tok, number_token_ids=[0, 1, 2], top_k=2, vector_norm=4.0)
    assert feats["mlp_top1_number_token_id"] == 2
    assert feats["mlp_top1_number_score"] == pytest.approx(6.0)
    assert feats["mlp_top2_number_score"] == pytest.approx(5.0)
    assert feats["mlp_number_margin"] == pytest.approx(1.0)
    assert feats["mlp_projection_norm"] == pytest.approx(4.0)
    assert feats["mlp_number_entropy"] > 0.0


def test_projection_features_do_not_require_gold_fields() -> None:
    feats = {
        "mlp_number_entropy": 1.0,
        "mlp_number_margin": 2.0,
        "mlp_layer20_layer24_agreement": 1.0,
        "mlp_agreement_with_model_generated_answer": 1.0,
        "mlp_projection_sharpness": 0.5,
    }
    risk = mra.compute_mlp_readout_risk(feats)
    assert risk == pytest.approx(-0.5)
    mra.assert_no_eval_only_features(list(feats))
    with pytest.raises(ValueError):
        mra.assert_no_eval_only_features(["mlp_number_margin", "gold_answer"])


def test_layer_agreement_and_margin_gap() -> None:
    merged = mra.merge_layer_features(
        {
            24: {"mlp_top1_number_token_id": 4, "mlp_number_margin": 2.5},
            20: {"mlp_top1_number_token_id": 4, "mlp_number_margin": 1.0},
        }
    )
    assert merged["mlp_layer20_layer24_agreement"] == 1.0
    assert merged["mlp_layer20_layer24_margin_gap"] == pytest.approx(1.5)

    merged2 = mra.merge_layer_features(
        {
            24: {"mlp_top1_number_token_id": 4, "mlp_number_margin": 2.5},
            20: {"mlp_top1_number_token_id": 7, "mlp_number_margin": 1.0},
        }
    )
    assert merged2["mlp_layer20_layer24_agreement"] == 0.0


def test_auroc_auprc_detection_metrics() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    report = mra.evaluate_correct_wrong_detection(labels, scores)
    assert report["auroc"] == pytest.approx(1.0)
    assert report["auprc"] == pytest.approx(1.0)
    assert report["positive_rate"] == pytest.approx(0.5)


def test_question_grouped_split_has_no_question_leakage() -> None:
    qids = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]
    folds = mra.question_grouped_folds(qids, k=4, seed=1)
    for train, test in folds:
        train_q = {qids[i] for i in train}
        test_q = {qids[i] for i in test}
        assert train_q.isdisjoint(test_q)


def test_question_grouped_cv_runs_and_rejects_gold_features() -> None:
    rows = []
    for i in range(12):
        wrong = i % 2
        rows.append(
            {
                "source_question_id": f"q{i // 2}",
                "wrong_label": wrong,
                "mlp_number_margin": 1.0 - wrong,
                "mlp_number_entropy": float(wrong),
            }
        )
    out = mra.question_grouped_cv(rows, feature_names=["mlp_number_margin", "mlp_number_entropy"], k=3)
    assert out["question_grouped"] is True
    assert out["feature_names"] == ["mlp_number_margin", "mlp_number_entropy"]
    assert out["available"] is True
    with pytest.raises(ValueError):
        mra.question_grouped_cv(rows, feature_names=["gold_answer"], k=3)


def test_calibration_buckets_and_ece() -> None:
    out = mra.calibration_buckets([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], n_buckets=2)
    assert out["num_examples"] == 4
    assert len(out["buckets"]) == 2
    assert out["buckets"][0]["wrong_rate"] == pytest.approx(1.0)
    assert out["buckets"][1]["correct_rate"] == pytest.approx(1.0)
    assert out["ece"] is not None


def test_random_baseline_reproducible() -> None:
    assert mra.random_baseline_scores(5, seed=123) == mra.random_baseline_scores(5, seed=123)
    assert mra.random_baseline_scores(5, seed=123) != mra.random_baseline_scores(5, seed=124)

