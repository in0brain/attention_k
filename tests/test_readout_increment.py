import torch

from recover_attention import readout_increment as ri


def test_cross_layer_statistics_on_hand_built_logits():
    features = ri.cross_layer_features([4, 0, 0, 0], [0, 4, 0, 0], [0, 3, 0, 0], prefix="f1")
    assert features["f1_margin_L20"] == 4.0
    assert features["f1_top_flip"] == 1.0
    assert features["f1_layer_disagreement"] > 0
    assert features["f1_final_vs_mid_kl"] >= 0


def test_exact_vjp_matches_analytic_linear_gradient():
    activation = torch.tensor([2.0, -3.0], requires_grad=True)
    logit = (activation * torch.tensor([5.0, 7.0])).sum()
    assert torch.isclose(ri.exact_vjp_projection(logit, activation), torch.tensor(-11.0))
    batched = activation.detach().reshape(1, 1, 2).requires_grad_()
    batched_logit = (batched * torch.tensor([[[5.0, 7.0]]])).sum()
    assert torch.isclose(ri.exact_vjp_position_projection(batched_logit, batched, position=0), torch.tensor(-11.0))


def test_grouped_cv_rejects_gold_feature_and_uses_same_interface():
    rows = [{"group_id": str(i), "wrong_label": i % 2, "f5_entropy": float(i % 2), "f1_entropy": float(i % 2)} for i in range(10)]
    report = ri.evaluate_grouped_cv(rows, families={"F5": ["f5_entropy"], "F5+F1": ["f5_entropy", "f1_entropy"]}, seeds=[0, 1, 2])
    assert set(report) == {"F5", "F5+F1"}
    assert all(len(value["seed_runs"]) == 3 for value in report.values())
