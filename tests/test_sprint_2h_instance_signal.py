"""Tests for Sprint 2H-B instance-level signal construction modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import (  # noqa: E402
    fragility_probe_training as fpt,
    model_recovery as mr,
    recovery_drift as rd,
    risk_strength_targets as rst,
    solution_path_numbers as spn,
)


# --------------------------------------------------------------------------- #
# solution_path_numbers
# --------------------------------------------------------------------------- #
def test_normalize_number_text_variants():
    assert spn.normalize_number_text("$30") == [30.0]
    assert spn.normalize_number_text("1,000") == [1000.0]
    assert spn.normalize_number_text("50%") == [50.0, 0.5]
    assert spn.normalize_number_text("3.0") == [3.0]
    assert spn.normalize_number_text("twelve") == [12.0]
    assert spn.normalize_number_text("apples") == []


def test_parse_solution_expressions():
    solution = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n48+24 = <<48+24=72>>72 clips.\n#### 72"
    parsed = spn.parse_solution_expressions(solution)
    assert 48.0 in parsed["operand_values"]
    assert 24.0 in parsed["result_values"]
    assert 72.0 in parsed["result_values"]
    assert parsed["num_expressions"] == 2


def test_classify_chosen_span_on_off_ambiguous():
    question = "Tom has 12 candies and gives away 5. He buys 12 more. How many are left?"
    solution = "12-5 = <<12-5=7>>7. 7+? ... #### 7"
    # value 12 appears twice in question and is an operand -> ambiguous
    result = spn.classify_chosen_span_solution_path("12", "number", question, solution)
    assert result["status"] == spn.AMBIGUOUS
    # value 5 appears once and is an operand -> on path
    result5 = spn.classify_chosen_span_solution_path("5", "number", question, solution)
    assert result5["status"] == spn.ON_PATH


def test_off_path_distractor_number():
    question = "In 2020, Sam bought 4 apples for 3 dollars each. What was the total?"
    solution = "4*3 = <<4*3=12>>12 dollars. #### 12"
    census = spn.build_question_census(question, solution)
    statuses = {n["text"]: n["status"] for n in census["number_spans"]}
    assert statuses["2020"] == spn.OFF_PATH  # year distractor
    assert statuses["4"] == spn.ON_PATH
    assert statuses["3"] == spn.ON_PATH


def test_implicit_numbers_reported():
    question = "She sold twice as many and gave away half of them."
    implicit = spn.extract_implicit_numbers(question)
    values = {item["text"]: item["value"] for item in implicit}
    assert values["twice"] == 2.0
    assert values["half"] == 0.5


def test_non_number_span_is_not_a_number():
    result = spn.classify_chosen_span_solution_path("more", "comparison", "q", "s")
    assert result["status"] == spn.NOT_A_NUMBER


# --------------------------------------------------------------------------- #
# recovery_drift
# --------------------------------------------------------------------------- #
def test_extract_filler_exact_and_rephrased():
    masked = "Tom has [MASK] candies left."
    assert rd.extract_filler(masked, "Tom has 12 candies left.", "[MASK]") == "12"
    # minor rephrase outside slot still recovers filler
    assert rd.extract_filler(masked, "Tom has several candies left.", "[MASK]") == "several"


def test_classify_sample_drift_number():
    assert rd.classify_sample_drift("12", "number", "12", False) == rd.EXACT
    assert rd.classify_sample_drift("12", "number", "several", False) == rd.GENERIC
    assert rd.classify_sample_drift("12", "number", "10", False) == rd.WRONG_NUMERIC
    assert rd.classify_sample_drift("12", "number", None, True) == rd.UNRECOVERABLE


def test_classify_sample_drift_direction():
    assert rd.classify_sample_drift("more", "comparison", "less", False) == rd.DIRECTION_DRIFT
    assert rd.classify_sample_drift("more", "comparison", "greater", False) == rd.EXACT
    assert rd.classify_sample_drift("not", "negation", "indeed", False) == rd.DIRECTION_DRIFT


def test_drift_guards_plural_mask_and_rephrase():
    # singular/plural counts as exact recovery
    assert rd.classify_sample_drift("sandwich", "object", "sandwiches", False) == rd.EXACT
    # leftover mask token -> ambiguous, not a hard drift
    assert rd.classify_sample_drift("books", "object", "[MASK]", False) == rd.AMBIGUOUS
    # full-question rephrase -> ambiguous
    long_filler = "Quinn library summer reading challenge for every five books you read a coupon"
    assert rd.classify_sample_drift("books", "object", long_filler, False) == rd.AMBIGUOUS


def test_aggregate_drift_worst_case():
    # one wrong among mostly exact still flags any_wrong
    agg = rd.aggregate_drift([rd.EXACT, rd.EXACT, rd.WRONG_NUMERIC])
    assert agg["any_wrong"] is True
    assert agg["any_hard_drift"] is True
    assert agg["num_exact"] == 2
    assert agg["inconsistency_rate"] > 0


# --------------------------------------------------------------------------- #
# risk_strength_targets
# --------------------------------------------------------------------------- #
def test_bucket_hard_drift_is_3():
    agg = rd.aggregate_drift([rd.EXACT, rd.EXACT, rd.WRONG_NUMERIC])
    result = rst.assign_fragility_bucket("number", spn.ON_PATH, agg)
    assert result["bucket"] == rst.BUCKET_DRIFTED


def test_bucket_off_path_is_0_regardless_of_drift():
    # off-path distractor number: bucket 0 whether recovery is stable...
    stable = rd.aggregate_drift([rd.EXACT, rd.EXACT, rd.EXACT])
    assert rst.assign_fragility_bucket("number", spn.OFF_PATH, stable)["bucket"] == rst.BUCKET_OFF_PATH
    # ...or drifts (steering a distractor is wasted budget either way)
    drifted = rd.aggregate_drift([rd.EXACT, rd.WRONG_NUMERIC, rd.WRONG_NUMERIC])
    assert rst.assign_fragility_bucket("number", spn.OFF_PATH, drifted)["bucket"] == rst.BUCKET_OFF_PATH


def test_bucket_on_path_stable_is_1():
    agg = rd.aggregate_drift([rd.EXACT, rd.EXACT, rd.EXACT])
    result = rst.assign_fragility_bucket("number", spn.ON_PATH, agg)
    assert result["bucket"] == rst.BUCKET_STABLE


def test_bucket_generic_is_2():
    agg = rd.aggregate_drift([rd.GENERIC, rd.GENERIC, rd.EXACT])
    result = rst.assign_fragility_bucket("number", spn.ON_PATH, agg)
    assert result["bucket"] == rst.BUCKET_UNDER_RECOVERED


def test_ambiguous_number_excluded():
    agg = rd.aggregate_drift([rd.EXACT])
    result = rst.assign_fragility_bucket("number", spn.AMBIGUOUS, agg)
    assert result["bucket"] is None
    assert result["excluded"] is True


def test_risk_strength_preserves_bucket_ordering():
    agg = rd.aggregate_drift([rd.EXACT, rd.GENERIC])
    s0 = rst.compute_risk_strength(0, agg)
    s1 = rst.compute_risk_strength(1, agg)
    s2 = rst.compute_risk_strength(2, agg)
    s3 = rst.compute_risk_strength(3, agg)
    assert s0 < s1 < s2 < s3


def test_not_span_type_deterministic_detection():
    dataset = [
        {"span_type": "number", "fragility_bucket": 1},
        {"span_type": "number", "fragility_bucket": 3},
        {"span_type": "object", "fragility_bucket": 0},
    ]
    check = rst.check_not_span_type_deterministic(dataset)
    assert "number" in check["span_types_with_multiple_buckets"]
    assert check["is_span_type_deterministic"] is False


# --------------------------------------------------------------------------- #
# model_recovery
# --------------------------------------------------------------------------- #
def test_recovery_prompt_hides_span_type():
    prompt = mr.build_recovery_prompt("Tom has [MASK] candies.", "[MASK]")
    lowered = prompt.lower()
    for banned in ["number", "comparison", "negation", "object", "span type"]:
        assert banned not in lowered
    assert "uncertain" in lowered


def test_sample_recoveries_with_stub_and_temperature_guard():
    calls = {"n": 0}

    def fake_chat(prompt, **kwargs):
        calls["n"] += 1
        return f"Tom has {kwargs['seed']} candies."

    samples = mr.sample_recoveries(
        "Tom has [MASK] candies.", "[MASK]", num_samples=3, chat_fn=fake_chat
    )
    assert len(samples) == 3
    assert {s["seed"] for s in samples} == {101, 102, 103}
    assert calls["n"] == 3

    with pytest.raises(ValueError):
        mr.sample_recoveries("q [MASK]", "[MASK]", temperature=0.0, chat_fn=fake_chat)


# --------------------------------------------------------------------------- #
# fragility_probe_training
# --------------------------------------------------------------------------- #
def _synthetic_records(n=60):
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        bucket = i % 4
        # signal correlated with bucket in the original_masked channel
        base = bucket + rng.normal(0, 0.3)
        records.append(
            {
                "id": f"q{i}",
                "source_question_id": f"src{i}",
                "span_type": "number" if i % 2 == 0 else "comparison",
                "span_text": "12" if i % 2 == 0 else "more",
                "question": "Tom has 12 candies and more apples than pears.",
                "solution_path_status": spn.ON_PATH if i % 2 == 0 else spn.NOT_A_NUMBER,
                "fragility_bucket": bucket,
                "risk_strength": bucket / 3.0,
                "feature_values": {
                    "span_original_masked_cosine_mean": float(base),
                    "span_original_recovered_cosine_mean": float(base + 5.0),  # forbidden channel
                },
            }
        )
    return records


def test_feature_matrix_excludes_recovered_channel():
    records = _synthetic_records()
    built = fpt.build_feature_matrix(records, "hidden_no_recovered")
    assert all("recovered" not in name for name in built["feature_names"])
    fpt.assert_no_recovered_features(built["feature_names"])

    built_all = fpt.build_feature_matrix(records, "hidden_with_recovered")
    assert any("original_recovered" in name for name in built_all["feature_names"])
    with pytest.raises(AssertionError):
        fpt.assert_no_recovered_features(built_all["feature_names"])


def test_metrics_helpers():
    assert fpt.mann_whitney_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert fpt.spearman_corr(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) == 1.0
    acc = fpt.pairwise_ordering_accuracy(np.array([0.1, 0.9]), np.array([0, 1]))
    assert acc == 1.0


def test_cv_runner_end_to_end():
    records = _synthetic_records()
    buckets = [r["fragility_bucket"] for r in records]
    strength = [r["risk_strength"] for r in records]
    result = fpt.run_cv_for_feature_set(
        records, buckets, strength, "hidden_no_recovered",
        labels=[0, 1, 2, 3], alpha=1.0, num_folds=5, seed=42,
    )
    assert "macro_f1" in result["metrics"]
    assert len(result["oof_pred_bucket"]) == len(records)
    # signal is real -> ordinal correlation should be positive
    assert result["metrics"]["spearman_score_vs_bucket"] > 0


# --------------------------------------------------------------------------- #
# Sprint 2H-C: pre-recovery feature enrichment
# --------------------------------------------------------------------------- #
from recover_attention import pre_recovery_features as prf  # noqa: E402


def test_enriched_feature_names_have_no_banned_substrings():
    # task 4: gate-eligible enriched features must not contain any label-leaking token
    prf.assert_no_banned_feature_names(["pre_delta_span_l2_layer_0", "pre_stability_span_layershift_orig"])
    for banned in ["recovered", "solution_path", "drift", "bucket", "risk_strength", "gold"]:
        with pytest.raises(AssertionError):
            prf.assert_no_banned_feature_names([f"pre_feature_{banned}_x"])


def test_enriched_matrix_build_and_leakage_guard():
    records = [
        {"pre_recovery_features": {"pre_delta_span_l2_layer_0": 1.0, "pre_stability_span_layervar_orig": 0.2}},
        {"pre_recovery_features": {"pre_delta_span_l2_layer_0": 2.0, "pre_stability_span_layervar_orig": 0.4}},
    ]
    built = fpt.build_feature_matrix(records, "hidden_pre_recovery_enriched")
    assert built["matrix"].shape == (2, 2)
    assert all("recovered" not in n for n in built["feature_names"])
    prf.assert_no_banned_feature_names(built["feature_names"])


def test_extract_pre_recovery_features_leakage_free_with_stub_tensors():
    torch = pytest.importorskip("torch")

    def fake_loader(path):
        # [layers=3, seq=3, hidden=8]; deterministic per path so orig != masked
        gen = torch.Generator().manual_seed(1 if "orig" in str(path) else 2)
        return torch.randn(3, 3, 8, generator=gen)

    original = {
        "input_type": "original",
        "input_text": "Tom has 12 apples",
        "hidden_state_path": "orig.pt",
        "token_char_ranges": [[0, 3], [4, 6], [7, 9]],
        "masked_original_spans": [{"original_char_range": [7, 9], "text": "12"}],
    }
    masked = {
        "input_type": "masked",
        "input_text": "Tom has [MASK] apples",
        "hidden_state_path": "mask.pt",
        "token_char_ranges": [[0, 3], [4, 6], [7, 13]],
        "mask_char_ranges": [[7, 13]],
    }
    result = prf.extract_pre_recovery_features(original, masked, tensor_loader=fake_loader)
    assert result["span_available"] is True
    names = list(result["features"].keys())
    prf.assert_no_banned_feature_names(names)
    # all three families present
    assert any(n.startswith("pre_delta_span_") for n in names)
    assert any(n.startswith("pre_saliency_span_to_question_") for n in names)
    assert any(n.startswith("pre_stability_span_") for n in names)


# --------------------------------------------------------------------------- #
# Sprint 2H-D: ordinal calibration
# --------------------------------------------------------------------------- #
from recover_attention import ordinal_calibration as oc  # noqa: E402


def test_softmax_rows_normalized():
    p = oc.softmax_rows(np.array([[1.0, 2.0, 3.0, 0.0]]), temperature=1.0)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    assert p.argmax() == 2


def test_expected_bucket_monotone_in_decision():
    # higher decision mass on bucket 3 -> higher expected bucket
    low = oc.expected_bucket_from_decision(np.array([[3.0, 0, 0, 0]]), [0, 1, 2, 3], 1.0)[0]
    high = oc.expected_bucket_from_decision(np.array([[0, 0, 0, 3.0]]), [0, 1, 2, 3], 1.0)[0]
    assert high > low


def test_pav_isotonic_is_monotone():
    y = np.array([0.0, 3.0, 1.0, 2.0, 2.5])
    fitted = oc.pav_isotonic(y)
    assert all(fitted[i] <= fitted[i + 1] + 1e-9 for i in range(len(fitted) - 1))


def test_fit_isotonic_preserves_order_on_train():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=50)
    target = (pred > 0).astype(float) * 2 + rng.normal(0, 0.1, size=50)
    f = oc.fit_isotonic(pred, target)
    cal = f(pred)
    # calibrated should correlate positively with target
    assert oc.spearman_corr(cal, target) > 0.5


def test_budget_curve_flooding_vs_precise():
    buckets = np.array([3, 3, 1, 1, 0, 2, 3, 1, 2, 3])
    # a precise score ranks bucket-3 items first
    precise = np.array([0.9, 0.85, 0.1, 0.1, 0.0, 0.4, 0.95, 0.1, 0.4, 0.88])
    curve = oc.budget_curve(precise, buckets, fractions=[0.2])
    assert curve["points"]["top_20pct"]["bucket_3_precision"] == 1.0


def test_bootstrap_ranking_delta_positive_when_a_better():
    rng = np.random.default_rng(1)
    buckets = np.array([0, 1, 2, 3] * 20)
    good = buckets + rng.normal(0, 0.2, size=len(buckets))   # tracks bucket
    bad = rng.normal(0, 1, size=len(buckets))                # noise
    res = oc.bootstrap_ranking_delta(good.astype(float), bad, buckets, num_boot=200, seed=3)
    assert res["ci95_low"] > 0
    assert res["meaningfully_above_noise"] is True


# --------------------------------------------------------------------------- #
# Sprint 2K: output-effect features + 2J slot-alignment fix
# --------------------------------------------------------------------------- #
from recover_attention import answer_effect_features as oe_feat  # noqa: E402
from recover_attention import multi_span_reasoning_scoring as msrs  # noqa: E402


def test_output_effect_names_leakage_free_and_compute():
    torch = pytest.importorskip("torch")
    # names must avoid every banned substring (incl. answer/target/oracle/cot/nla)
    for banned in ["recovered", "solution_path", "drift", "bucket", "gold", "answer",
                   "label", "target", "trajectory", "cot", "nla", "oracle"]:
        with pytest.raises(AssertionError):
            oe_feat.assert_no_banned_output_effect_names([f"self_{banned}_x"])
    orig = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])
    masked_same = orig.clone()
    masked_diff = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    f_same = oe_feat.compute_output_effect(orig, masked_same)
    f_diff = oe_feat.compute_output_effect(orig, masked_diff)
    oe_feat.assert_no_banned_output_effect_names(list(f_diff.keys()))
    # identical distributions -> ~zero shift, no top1 change
    assert f_same["self_output_kl"] < 1e-5
    assert f_same["self_output_top1_changed"] == 0.0
    # flipped distribution -> large shift + top1 change
    assert f_diff["self_output_kl"] > f_same["self_output_kl"]
    assert f_diff["self_output_top1_changed"] == 1.0
    # shift score monotone
    assert oe_feat.output_effect_shift_score(f_diff) > oe_feat.output_effect_shift_score(f_same)


def test_slot_indices_recomputed_per_span():
    # regression for the 2J-B bug: two different spans in the same question must map to
    # different original slot token indices (not the first span's cached indices).
    # offsets emulate: "Alice bought 3 apples for 2 dollars each."
    offsets = [[0, 5], [6, 12], [13, 14], [15, 21], [22, 25], [26, 27], [28, 35], [36, 40]]
    span_a = msrs.token_indices_for_char_ranges(offsets, [[13, 14]], exclude=set())  # "3"
    span_b = msrs.token_indices_for_char_ranges(offsets, [[26, 27]], exclude=set())  # "2"
    assert span_a == [2]
    assert span_b == [5]
    assert span_a != span_b


# --------------------------------------------------------------------------- #
# Sprint 2I: attention features
# --------------------------------------------------------------------------- #
from recover_attention import attention_features as afeat  # noqa: E402


def _stub_attention(seq, layers=2, seed=0):
    rng = np.random.default_rng(seed)
    stack = []
    for _ in range(layers):
        m = np.tril(rng.random((seq, seq)) + 0.01)  # causal-ish
        m = m / m.sum(axis=1, keepdims=True)
        stack.append(m)
    return np.stack(stack, axis=0)


def test_attention_feature_names_have_no_banned_substrings():
    # includes the newly banned answer/label/target
    afeat.assert_no_banned_attention_names(["attn_orig_slot_in_mass", "attn_delta_slot_entropy",
                                            "attn_orig_qfocus_to_slot"])
    for banned in ["recovered", "solution_path", "drift", "bucket", "risk_strength", "gold", "answer", "label", "target"]:
        with pytest.raises(AssertionError):
            afeat.assert_no_banned_attention_names([f"attn_{banned}_x"])


def test_build_attention_features_leakage_free_and_families():
    orig = _stub_attention(10, seed=1)
    masked = _stub_attention(11, seed=2)
    ctx = {"qfocus": [7, 8], "operation": [3], "numctx": [1, 2]}
    out = afeat.build_attention_features(orig, [4, 5], masked, [4, 5], [0, 8], context_orig=ctx)
    names = list(out["features"].keys())
    afeat.assert_no_banned_attention_names(names)
    assert any(n.startswith("attn_orig_slot_") for n in names)
    assert any(n.startswith("attn_masked_slot_") for n in names)
    assert any(n.startswith("attn_delta_slot_") for n in names)
    assert any(n.endswith("_to_slot") for n in names)
    # question-focus feature must NOT use the banned word "target"
    assert not any("target" in n for n in names)


def test_attention_and_fusion_feature_sets_build():
    records = [
        {"pre_recovery_features": {"pre_delta_span_l2_layer_0": 1.0},
         "attention_features": {"attn_orig_slot_in_mass": 0.3, "attn_delta_slot_entropy": -0.1}},
        {"pre_recovery_features": {"pre_delta_span_l2_layer_0": 2.0},
         "attention_features": {"attn_orig_slot_in_mass": 0.5, "attn_delta_slot_entropy": 0.2}},
    ]
    attn = fpt.build_feature_matrix(records, "attention_pre_recovery")
    assert attn["matrix"].shape == (2, 2)
    afeat.assert_no_banned_attention_names(attn["feature_names"])
    fused = fpt.build_feature_matrix(records, "hidden_plus_attention_pre_recovery")
    assert fused["matrix"].shape == (2, 3)  # 1 hidden + 2 attention
    afeat.assert_no_banned_attention_names(fused["feature_names"])
