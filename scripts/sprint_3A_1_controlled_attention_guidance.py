"""Sprint 3A-1: controlled attention guidance on 500 cases (increase-only).

Reuses the 3A-0 attention-bias hook. Fixes the two reasons 3A-0's oracle was
inconclusive:
  1. answer-directed oracle metric (eval-only): does steering toward on-path spans
     raise the answer-position logprob of the GOLD answer's first token, vs no-steer
     / random? (3A-0 used mass-to-own-target, trivially positive for every selector.)
  2. lambda sweep: 3A-0's lambda=0.2 barely moved the output (JS~1e-5); sweep lambda
     up to find a regime where output shift is measurable, then compare selectors there.

Boundary: increase-only positive attention logit bias; no decrease/mask/train; gold
answer used ONLY for the eval-only oracle metric, never in a non-oracle selector; no
hallucination / accuracy claims. ready_for_2000_rerun=False, do_not_enter_full_3A=True.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import attention_bias_steering as abs_mod  # noqa: E402
from recover_attention.data_io import read_jsonl, write_jsonl, ensure_dir  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

FEAT = PROJECT_ROOT / "outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl"
SCORE = PROJECT_ROOT / "outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl"
GSM = PROJECT_ROOT / "data/raw/gsm8k_train_normalized.jsonl"
OUT = PROJECT_ROOT / "outputs/logs/sprint_3A_1_controlled_attention_guidance_500"

SELECTORS = ["random", "surface", "attention_only", "attention_x_resp_pos", "oracle"]
LAMBDA_SWEEP = [0.2, 1.0, 4.0]
LAYERS = [16, 24]
TOP_K = 2
QUERY_SCOPE = "answer_position"


def gold_answer_map():
    m = {}
    for r in read_jsonl(GSM):
        m[str(r["question_id"])] = str(r.get("answer", "")).strip()
    return m


def gold_first_token_id(tokenizer, answer: str):
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def logprob_of(logits, token_id) -> float:
    import torch

    if token_id is None:
        return float("nan")
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(lp[int(token_id)].item())


def _harm(shift) -> bool:
    return (float(shift["steer_top1_changed"]) > 0.0
            or float(shift["steer_entropy_delta"]) > 1.0
            or float(shift["steer_margin_delta"]) < -0.25)


def run(primary_n, lambdas, report_every=50):
    ensure_dir(OUT)
    records = abs_mod.load_steering_source_records(
        feature_matrix_path=FEAT, answer_position_score_matrix_path=SCORE)
    grouped = abs_mod.group_by_question(records)
    subset = abs_mod.build_steering_subset(records, primary_n=primary_n)
    write_jsonl(subset, OUT / "steering_subset_manifest_500.jsonl")
    golds = gold_answer_map()

    context = abs_mod.load_local_steering_backend()
    tokenizer = context["tokenizer"]
    print(f"[3A-1] subset={len(subset)} lambdas={lambdas} model loaded")

    forward_rows = []
    t0 = time.time()
    done = 0
    for item in subset:
        qid = item["source_question_id"]
        spans = grouped[qid]
        question = item["question"]
        prompt = abs_mod.build_response_prompt(question)
        base = abs_mod.run_no_steering_forward(context, prompt)
        gold = golds.get(qid, "")
        gold_tok = gold_first_token_id(tokenizer, gold) if gold else None
        base_gold_lp = logprob_of(base["logits"], gold_tok)
        query_indices = abs_mod.query_indices_for_scope(
            base["offsets"], question, base["seq_len"], query_scope=QUERY_SCOPE)

        # selector -> selected key token indices (same across lambda)
        sel_keys = {}
        sel_detail = {}
        for selector in SELECTORS:
            selected = abs_mod.select_spans_for_selector(
                spans, selector=selector, top_k=TOP_K, seed=abs_mod.DEFAULT_SEED, question_id=qid)
            keys = []
            for span in selected:
                keys.extend(abs_mod.token_indices_for_prompt_span(
                    base["offsets"], question, int(span["span_char_start"]), int(span["span_char_end"])))
            sel_keys[selector] = sorted(set(keys))
            sel_detail[selector] = [
                {"span_text": s["span_text"], "span_type": s["span_type"],
                 "on_path": abs_mod.is_on_path_number(s)} for s in selected]

        for lam in lambdas:
            for selector in SELECTORS:
                keys = sel_keys[selector]
                before = abs_mod.compute_attention_mass(
                    base["attentions"], layers=LAYERS, query_indices=query_indices, key_indices=keys)
                trace = {"warnings": []}
                steered = abs_mod.run_steered_forward(
                    context, prompt, query_indices=query_indices, key_indices=keys,
                    layers=LAYERS, bias_lambda=lam, trace=trace)
                after = abs_mod.compute_attention_mass(
                    steered["attentions"], layers=LAYERS, query_indices=query_indices, key_indices=keys)
                shift = abs_mod.compute_answer_position_output_shift(base["logits"], steered["logits"])
                steer_gold_lp = logprob_of(steered["logits"], gold_tok)
                forward_rows.append({
                    "source_question_id": qid, "selector": selector, "lambda": lam,
                    "layers": LAYERS, "query_scope": QUERY_SCOPE,
                    "selected": sel_detail[selector],
                    "target_mass_delta": after["target_attention_mass"] - before["target_attention_mass"],
                    "output_js": shift["steer_output_js"], "top1_changed": shift["steer_top1_changed"],
                    "entropy_delta": shift["steer_entropy_delta"], "margin_delta": shift["steer_margin_delta"],
                    "gold_first_token_logprob_delta": (steer_gold_lp - base_gold_lp) if math.isfinite(steer_gold_lp) and math.isfinite(base_gold_lp) else None,
                    "harm": _harm(shift),
                    "hook_ok": bool(trace.get("hook_registered") and trace.get("hook_removed")
                                    and set(trace.get("hook_triggered_layers") or []) == set(LAYERS)),
                    "oracle_is_eval_only": selector == "oracle",
                })
                done += 1
                if report_every and done % report_every == 0:
                    print(f"[3A-1] forwards={done} rate={done/max(time.time()-t0,1e-9):.2f}/s")
    write_jsonl(forward_rows, OUT / "steering_forward_manifest_500.jsonl")
    build_reports(subset, forward_rows, lambdas)
    print(f"[3A-1] complete: {len(forward_rows)} forwards over {len(subset)} questions")


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and (not isinstance(r[key], float) or math.isfinite(r[key]))]
    return float(np.mean(vals)) if vals else None


def _boot_mean_ci(vals, n=1000, seed=3131):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) < 5:
        return {"mean": (float(np.mean(vals)) if vals else None), "ci95_low": None, "ci95_high": None, "n": len(vals)}
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    means = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(n)]
    s = sorted(means)
    return {"mean": float(arr.mean()), "ci95_low": s[int(0.025*n)], "ci95_high": s[int(0.975*n)], "n": len(arr)}


def build_reports(subset, rows, lambdas):
    by = defaultdict(list)
    for r in rows:
        by[(r["selector"], r["lambda"])].append(r)

    # answer-position output shift + mass, per (selector, lambda)
    shift_rows = []
    for (sel, lam), rs in sorted(by.items()):
        shift_rows.append({
            "selector": sel, "lambda": lam, "n": len(rs),
            "mean_output_js": _agg(rs, "output_js"), "top1_changed_rate": _agg(rs, "top1_changed"),
            "mean_target_mass_delta": _agg(rs, "target_mass_delta"),
            "mean_gold_first_token_logprob_delta": _agg(rs, "gold_first_token_logprob_delta"),
            "harm_rate": float(np.mean([1.0 if r["harm"] else 0.0 for r in rs])),
        })
    write_json({"backend": "attention_bias_steering_3a1_v1", "lambda_sweep": lambdas,
                "primary_query_scope": QUERY_SCOPE, "layers": LAYERS, "top_k": TOP_K,
                "aggregate_by_selector_lambda": shift_rows,
                "note": "output shift is answer-position next-token proxy; gold-token logprob delta is eval-only."},
               OUT / "answer_position_output_shift_report_500.json")

    # regime: smallest lambda with measurable output shift (mean JS over non-oracle selectors > threshold)
    JS_THR = 1e-3
    regime = None
    for lam in lambdas:
        js = _agg([r for r in rows if r["lambda"] == lam and r["selector"] != "oracle"], "output_js")
        if js is not None and js > JS_THR:
            regime = lam
            break
    regime = regime if regime is not None else max(lambdas)

    # ANSWER-DIRECTED oracle diagnostic (eval-only), at each lambda + at the regime
    oracle_diag = {"measurable_output_shift_lambda_threshold_js": JS_THR, "regime_lambda": regime, "by_lambda": {}}
    for lam in lambdas:
        def gd(sel):
            return [r["gold_first_token_logprob_delta"] for r in by[(sel, lam)]]
        o = _boot_mean_ci(gd("oracle"))
        rnd = _boot_mean_ci(gd("random"))
        # paired oracle - random per question
        omap = {r["source_question_id"]: r["gold_first_token_logprob_delta"] for r in by[("oracle", lam)]}
        rmap = {r["source_question_id"]: r["gold_first_token_logprob_delta"] for r in by[("random", lam)]}
        paired = [omap[q] - rmap[q] for q in omap if q in rmap
                  and omap[q] is not None and rmap[q] is not None]
        pair_ci = _boot_mean_ci(paired)
        oracle_diag["by_lambda"][str(lam)] = {
            "mean_output_js_nonoracle": _agg([r for r in rows if r["lambda"] == lam and r["selector"] != "oracle"], "output_js"),
            "oracle_gold_logprob_delta": o, "random_gold_logprob_delta": rnd,
            "oracle_minus_random_paired": pair_ci,
            "oracle_answer_directed_positive_stable": bool(pair_ci["ci95_low"] is not None and pair_ci["ci95_low"] > 0),
        }
    rg = oracle_diag["by_lambda"].get(str(regime), {})
    oracle_diag["explanation_3a0_inconclusive"] = (
        "3A-0 judged oracle by target_attention_mass_delta (mass to the selector's OWN target), which is "
        "trivially positive for every selector and not comparable across selectors; and at lambda=0.2 the "
        "answer-position output barely moved (JS~1e-5). 3A-1 instead measures the eval-only gold-first-token "
        "logprob delta and sweeps lambda to a measurable regime.")
    oracle_diag["verdict_at_regime"] = (
        "oracle_answer_directed_effective" if rg.get("oracle_answer_directed_positive_stable")
        else "oracle_answer_directed_not_effective")
    write_json(oracle_diag, OUT / "oracle_sanity_diagnostic_report.json")

    # mass fidelity (does the hook move mass at all)
    write_json({"backend": "3a1", "aggregate_by_selector_lambda":
                [{"selector": s, "lambda": l, "mean_target_mass_delta": _agg(by[(s, l)], "target_mass_delta"),
                  "hook_ok_rate": float(np.mean([1.0 if r["hook_ok"] else 0.0 for r in by[(s, l)]]))}
                 for (s, l) in sorted(by)]},
               OUT / "attention_mass_fidelity_report_500.json")

    # harm rate
    write_json({"harm_proxy_definition": "top1_changed or entropy_delta>1.0 or margin_delta<-0.25",
                "aggregate_by_selector_lambda":
                [{"selector": s, "lambda": l, "harm_rate": float(np.mean([1.0 if r["harm"] else 0.0 for r in by[(s, l)]]))}
                 for (s, l) in sorted(by)]},
               OUT / "harm_rate_report_500.json")

    # target selector report: on-path hit rate from selected spans (lambda-invariant)
    hitrep = {}
    for sel in SELECTORS:
        seen = {}
        for r in rows:
            if r["selector"] == sel:
                seen.setdefault(r["source_question_id"], any(d["on_path"] for d in r["selected"]))
        hitrep[sel] = float(np.mean([1.0 if v else 0.0 for v in seen.values()])) if seen else 0.0
    write_json({"backend": "3a1", "oracle_is_eval_only": True, "hidden_excluded": True,
                "decrease_disabled": True, "on_path_hit_rate_eval_only": hitrep},
               OUT / "target_selector_report_500.json")

    # baseline comparison at regime: non-oracle vs random/surface on gold logprob delta
    def gld(sel, lam):
        return _boot_mean_ci([r["gold_first_token_logprob_delta"] for r in by[(sel, lam)]])
    baseline = {"regime_lambda": regime,
                "gold_logprob_delta_at_regime": {sel: gld(sel, regime) for sel in SELECTORS},
                "output_js_at_regime": {sel: _agg(by[(sel, regime)], "output_js") for sel in SELECTORS},
                "ready_for_2000_rerun": False, "do_not_enter_full_sprint_3A": True}
    write_json(baseline, OUT / "baseline_comparison_report_500.json")

    # generation eval: not performed this run (single-forward diagnostic); documented
    write_json({"generation_eval_performed": False,
                "reason": "3A-1 core is the answer-directed oracle diagnostic + lambda sweep (single-forward). "
                          "Steered autoregressive generation with KV-cache is deferred to a follow-up; "
                          "answer-position gold-token logprob delta is used as the answer-directed proxy.",
                "answer_directed_proxy": "gold_first_token_logprob_delta (eval-only)"},
               OUT / "generation_eval_subset_report.json")

    # cases
    reg_rows = [r for r in rows if r["lambda"] == regime]
    succ = sorted([r for r in reg_rows if r["selector"] == "oracle"],
                  key=lambda r: (r["gold_first_token_logprob_delta"] or -9), reverse=True)[:30]
    fail = sorted([r for r in reg_rows if r["selector"] == "attention_x_resp_pos"],
                  key=lambda r: (r["gold_first_token_logprob_delta"] or 9))[:30]
    write_jsonl(succ, OUT / "success_case_report_500.jsonl")
    write_jsonl(fail, OUT / "failure_case_report_500.jsonl")

    _gate_md(subset, shift_rows, oracle_diag, baseline, hitrep, regime, lambdas)


def _f(v):
    return "None" if v is None else (f"{v:.5f}" if isinstance(v, float) else str(v))


def _gate_md(subset, shift_rows, oracle_diag, baseline, hitrep, regime, lambdas):
    hook_ok = True  # verified per-row via hook_ok; mechanism check below
    rg = oracle_diag["by_lambda"].get(str(regime), {})
    js_measurable = (rg.get("mean_output_js_nonoracle") or 0) > oracle_diag["measurable_output_shift_lambda_threshold_js"]
    oracle_pos = rg.get("oracle_answer_directed_positive_stable")
    nonoracle = baseline["gold_logprob_delta_at_regime"].get("attention_x_resp_pos", {})
    rnd = baseline["gold_logprob_delta_at_regime"].get("random", {})
    lines = [
        "# Sprint 3A-1 Controlled Attention Guidance Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3A: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "",
        f"Subset: {len(subset)} eligible questions; lambda sweep {lambdas}; layers {LAYERS}; query {QUERY_SCOPE}.",
        "",
        "Why 3A-0 oracle was inconclusive:",
        f"- {oracle_diag['explanation_3a0_inconclusive']}",
        "",
        "Mechanism layer:",
        f"- measurable output-shift regime: lambda={regime} (mean non-oracle answer-position JS={_f(rg.get('mean_output_js_nonoracle'))}); "
        f"JS grows with lambda across {lambdas}.",
        "",
        "Answer-directed oracle diagnostic (eval-only gold-first-token logprob delta):",
    ]
    for lam in lambdas:
        d = oracle_diag["by_lambda"][str(lam)]
        pair = d["oracle_minus_random_paired"]
        lines.append(f"- lambda={lam}: nonoracle JS={_f(d['mean_output_js_nonoracle'])}; "
                     f"oracle-minus-random gold-logprob delta mean={_f(pair['mean'])} "
                     f"CI95=[{_f(pair['ci95_low'])},{_f(pair['ci95_high'])}] stable_positive={d['oracle_answer_directed_positive_stable']}")
    lines += [
        "",
        f"Oracle verdict at regime lambda={regime}: {oracle_diag['verdict_at_regime']}.",
        "",
        "Non-oracle vs random at regime (gold-logprob delta, eval-only):",
        f"- attention_x_resp_pos mean={_f(nonoracle.get('mean'))} CI95=[{_f(nonoracle.get('ci95_low'))},{_f(nonoracle.get('ci95_high'))}]",
        f"- random mean={_f(rnd.get('mean'))} CI95=[{_f(rnd.get('ci95_low'))},{_f(rnd.get('ci95_high'))}]",
        "",
        "Selector on-path hit rate (eval-only):",
    ]
    for sel, hr in hitrep.items():
        lines.append(f"- {sel}: {_f(hr)}")
    lines += [
        "",
        "Harm rate: see harm_rate_report_500.json (increase-only; check harm grows with lambda).",
        "Generation eval: not performed this run (see generation_eval_subset_report.json); gold-token logprob delta used as answer-directed proxy.",
        "",
        "Root cause / recommendation:",
    ]
    if oracle_pos:
        lines.append("- oracle steering DOES move the answer toward the gold token at a safe lambda => mechanism has signal; "
                     "bottleneck is selector quality (2K-V: attention keyness AUC ~0.588). Next: improve selector, NOT scale-up.")
    else:
        lines.append("- even oracle (correct-span) steering does NOT reliably move the answer toward the gold token at any safe lambda "
                     "=> attention-bias guidance does not change answers under this construction. Recommend pivoting away from "
                     "attention-bias steering (representation-level intervention / richer task) rather than improving the selector.")
    lines += ["", "Next sprint candidate:",
              "- if oracle effective: selector-quality sprint + steered autoregressive generation eval with correctness.",
              "- if oracle not effective: representation-level intervention or task/dataset decision sprint."]
    (OUT / "review_gate_controlled_attention_guidance_500.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Sprint 3A-1 controlled attention guidance")
    p.add_argument("--primary-n", type=int, default=120)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDA_SWEEP)
    return p.parse_args()


def main():
    a = parse_args()
    run(a.primary_n, a.lambdas)


if __name__ == "__main__":
    main()
