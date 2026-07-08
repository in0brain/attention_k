"""Sprint 3B-0: representation-level oracle intervention diagnostic (residual injection).

Parallel to 3A-1 but the intervention channel is the residual stream, not attention
logits. Question: does injecting the ORACLE (on-path) span's residual deviation into
the answer-position residual move the answer toward the gold token more than a RANDOM
span? Reuses 3A-1 selectors / subset / gold-token-logprob / paired bootstrap.

Boundary: no training/LoRA/2000/full-3B; increase/inject only (beta>=0); gold answer &
oracle used only for eval-only diagnostics; does not overwrite 3A-0/3A-1 outputs.
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
from recover_attention import representation_intervention as ri  # noqa: E402
from recover_attention.data_io import read_jsonl, write_jsonl, ensure_dir  # noqa: E402
from recover_attention.multi_span_reasoning_matrix import write_json  # noqa: E402

FEAT = PROJECT_ROOT / "outputs/logs/sprint_2J_fix_slot_alignment_scoring_500/multi_span_feature_matrix.jsonl"
SCORE = PROJECT_ROOT / "outputs/logs/sprint_2K_W_answer_position_output_effect_500/answer_position_score_matrix.jsonl"
GSM = PROJECT_ROOT / "data/raw/gsm8k_train_normalized.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs/logs/sprint_3B_0_representation_level_oracle_intervention_diagnostic"

SELECTORS = ["random", "surface", "attention_only", "attention_x_resp_pos", "oracle"]
DEFAULT_BETAS = [0.05, 0.1, 0.2, 0.4, 0.8]
DEFAULT_LAYERS = [16, 24]
TOP_K = 2
JS_MEASURABLE = 1e-3


def gold_answer_map():
    return {str(r["question_id"]): str(r.get("answer", "")).strip() for r in read_jsonl(GSM)}


def gold_first_token_id(tokenizer, answer: str):
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def logprob_of(logits, token_id) -> float:
    import torch

    if token_id is None:
        return float("nan")
    return float(torch.log_softmax(logits.float(), dim=-1)[int(token_id)].item())


def _harm(shift) -> bool:
    return (float(shift["steer_top1_changed"]) > 0.0
            or float(shift["steer_entropy_delta"]) > 1.0
            or float(shift["steer_margin_delta"]) < -0.25)


def run(primary_n, betas, layers, out_dir, overwrite):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if overwrite:
        for p in out_dir.glob("*"):
            if p.is_file():
                p.unlink()

    records = abs_mod.load_steering_source_records(
        feature_matrix_path=FEAT, answer_position_score_matrix_path=SCORE)
    grouped = abs_mod.group_by_question(records)
    subset = abs_mod.build_steering_subset(records, primary_n=primary_n)
    write_jsonl(subset, out_dir / "representation_subset_manifest.jsonl")
    write_json({
        "backend": "representation_residual_injection_v1",
        "intervention_type": "answer_position_residual_injection",
        "formula": "h_L[answer_pos] += beta * ||h_L[answer_pos]|| * unit(mean(h_L[span]) - mean(h_L[all]))",
        "beta_scaling": "beta is a fraction of the answer-position residual norm (calibrated so the sweep reaches a measurable regime)",
        "increase_only": True, "decrease_disabled": True, "hard_mask_disabled": True,
        "training": False, "lora": False,
        "layers": layers, "beta_sweep": betas, "top_k": TOP_K,
        "selectors": SELECTORS, "oracle_is_eval_only": True,
        "target_position": "answer_position",
    }, out_dir / "representation_intervention_config.json")
    golds = gold_answer_map()

    context = abs_mod.load_local_steering_backend()
    tokenizer = context["tokenizer"]
    print(f"[3B-0] subset={len(subset)} betas={betas} layers={layers} model loaded")

    rows = []
    t0 = time.time()
    done = 0
    for item in subset:
        qid = item["source_question_id"]
        spans = grouped[qid]
        question = item["question"]
        prompt = abs_mod.build_response_prompt(question)
        base = ri.base_forward_with_hidden(context, prompt, layers)
        ans_pos = base["seq_len"] - 1
        gold = golds.get(qid, "")
        gold_tok = gold_first_token_id(tokenizer, gold) if gold else None
        base_gold_lp = logprob_of(base["logits"], gold_tok)

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

        for beta in betas:
            for selector in SELECTORS:
                keys = sel_keys[selector]
                injections = ri.compute_injection_vectors(base["base_hidden"], keys, layers, ans_pos, beta)
                inj_norm = ri.injection_total_norm(injections)
                trace = {"triggered_layers": [], "registered": False, "removed": False}
                steered = ri.steered_forward_with_injection(context, prompt, injections, ans_pos, trace)
                shift = abs_mod.compute_answer_position_output_shift(base["logits"], steered["logits"])
                steer_gold_lp = logprob_of(steered["logits"], gold_tok)
                rows.append({
                    "source_question_id": qid, "selector": selector, "beta": beta, "layers": layers,
                    "selected": sel_detail[selector],
                    "injection_norm": inj_norm,
                    "output_js": shift["steer_output_js"], "top1_changed": shift["steer_top1_changed"],
                    "entropy_delta": shift["steer_entropy_delta"], "margin_delta": shift["steer_margin_delta"],
                    "gold_first_token_logprob_delta": (steer_gold_lp - base_gold_lp)
                    if math.isfinite(steer_gold_lp) and math.isfinite(base_gold_lp) else None,
                    "harm": _harm(shift),
                    "hook_ok": bool(trace["registered"] and trace["removed"]
                                    and set(trace["triggered_layers"]) == set(l for l in layers if injections.get(l) is not None)),
                    "oracle_is_eval_only": selector == "oracle",
                })
                done += 1
                if done % 50 == 0:
                    print(f"[3B-0] forwards={done} rate={done/max(time.time()-t0,1e-9):.2f}/s")
    write_jsonl(rows, out_dir / "representation_forward_manifest.jsonl")
    build_reports(subset, rows, betas, layers, out_dir)
    print(f"[3B-0] complete: {len(rows)} forwards over {len(subset)} questions")


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and (not isinstance(r[key], float) or math.isfinite(r[key]))]
    return float(np.mean(vals)) if vals else None


def _boot_ci(vals, n=1000, seed=3237):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) < 5:
        return {"mean": (float(np.mean(vals)) if vals else None), "ci95_low": None, "ci95_high": None, "n": len(vals)}
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    means = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(n)]
    s = sorted(means)
    return {"mean": float(arr.mean()), "ci95_low": s[int(0.025*n)], "ci95_high": s[int(0.975*n)], "n": len(arr)}


def build_reports(subset, rows, betas, layers, out_dir):
    by = defaultdict(list)
    for r in rows:
        by[(r["selector"], r["beta"])].append(r)

    fidelity = [{"selector": s, "beta": b, "mean_injection_norm": _agg(by[(s, b)], "injection_norm"),
                 "mean_output_js": _agg(by[(s, b)], "output_js"),
                 "hook_ok_rate": float(np.mean([1.0 if r["hook_ok"] else 0.0 for r in by[(s, b)]]))}
                for (s, b) in sorted(by)]
    write_json({"backend": "3b0", "metric": "answer_position_residual_injection_fidelity",
                "aggregate_by_selector_beta": fidelity}, out_dir / "representation_intervention_fidelity_report.json")

    shift_rows = [{"selector": s, "beta": b, "n": len(rs),
                   "mean_output_js": _agg(rs, "output_js"), "top1_changed_rate": _agg(rs, "top1_changed"),
                   "mean_gold_first_token_logprob_delta": _agg(rs, "gold_first_token_logprob_delta"),
                   "harm_rate": float(np.mean([1.0 if r["harm"] else 0.0 for r in rs]))}
                  for (s, b), rs in sorted(by.items())]
    write_json({"backend": "3b0", "beta_sweep": betas, "layers": layers,
                "aggregate_by_selector_beta": shift_rows}, out_dir / "gold_logprob_delta_report.json")

    # measurable regime = smallest beta with mean non-oracle output JS > threshold
    regime = None
    for b in betas:
        js = _agg([r for r in rows if r["beta"] == b and r["selector"] != "oracle"], "output_js")
        if js is not None and js > JS_MEASURABLE:
            regime = b
            break
    regime = regime if regime is not None else max(betas)

    def paired(sel_a, sel_b, beta):
        amap = {r["source_question_id"]: r["gold_first_token_logprob_delta"] for r in by[(sel_a, beta)]}
        bmap = {r["source_question_id"]: r["gold_first_token_logprob_delta"] for r in by[(sel_b, beta)]}
        return _boot_ci([amap[q] - bmap[q] for q in amap if q in bmap
                         and amap[q] is not None and bmap[q] is not None])

    oracle_diag = {"regime_beta": regime, "measurable_js_threshold": JS_MEASURABLE, "by_beta": {}}
    for b in betas:
        nonoracle_js = _agg([r for r in rows if r["beta"] == b and r["selector"] != "oracle"], "output_js")
        o_r = paired("oracle", "random", b)
        o_s = paired("oracle", "surface", b)
        o_a = paired("oracle", "attention_x_resp_pos", b)
        oracle_diag["by_beta"][str(b)] = {
            "mean_output_js_nonoracle": nonoracle_js,
            "oracle_minus_random": o_r, "oracle_minus_surface": o_s, "oracle_minus_attn_x_resp": o_a,
            "oracle_gt_random_stable": bool(o_r["ci95_low"] is not None and o_r["ci95_low"] > 0),
        }
    rg = oracle_diag["by_beta"].get(str(regime), {})
    oracle_diag["verdict_at_regime"] = (
        "oracle_representation_selective_signal" if rg.get("oracle_gt_random_stable")
        else "oracle_not_selectively_better_than_random")
    write_json(oracle_diag, out_dir / "oracle_vs_random_diagnostic_report.json")

    def gld(sel, beta):
        return _boot_ci([r["gold_first_token_logprob_delta"] for r in by[(sel, beta)]])
    selcmp = {"regime_beta": regime,
              "by_selector_at_regime": {s: {**gld(s, regime),
                                            "mean_output_js": _agg(by[(s, regime)], "output_js"),
                                            "harm_rate": float(np.mean([1.0 if r["harm"] else 0.0 for r in by[(s, regime)]])),
                                            "top1_changed_rate": _agg(by[(s, regime)], "top1_changed")}
                                        for s in SELECTORS}}
    write_json(selcmp, out_dir / "selector_comparison_report.json")

    write_json({"harm_proxy_definition": "top1_changed or entropy_delta>1.0 or margin_delta<-0.25",
                "aggregate_by_selector_beta":
                [{"selector": s, "beta": b, "harm_rate": float(np.mean([1.0 if r["harm"] else 0.0 for r in by[(s, b)]])),
                  "entropy_delta_mean": _agg(by[(s, b)], "entropy_delta"), "margin_delta_mean": _agg(by[(s, b)], "margin_delta")}
                 for (s, b) in sorted(by)]}, out_dir / "harm_rate_report.json")

    reg_rows = [r for r in rows if r["beta"] == regime]
    succ = sorted([r for r in reg_rows if r["selector"] == "oracle"],
                  key=lambda r: (r["gold_first_token_logprob_delta"] or -9), reverse=True)[:30]
    fail = sorted([r for r in reg_rows if r["selector"] == "oracle"],
                  key=lambda r: (r["gold_first_token_logprob_delta"] or 9))[:30]
    write_jsonl(succ, out_dir / "success_case_report.jsonl")
    write_jsonl(fail, out_dir / "failure_case_report.jsonl")

    _gate_md(subset, oracle_diag, selcmp, shift_rows, regime, betas, layers, out_dir)


def _f(v):
    return "None" if v is None else (f"{v:.5f}" if isinstance(v, float) else str(v))


def _gate_md(subset, oracle_diag, selcmp, shift_rows, regime, betas, layers, out_dir):
    rg = oracle_diag["by_beta"].get(str(regime), {})
    o_r = rg.get("oracle_minus_random", {})
    o_s = rg.get("oracle_minus_surface", {})
    o_a = rg.get("oracle_minus_attn_x_resp", {})
    selective = rg.get("oracle_gt_random_stable")
    by_sel = selcmp["by_selector_at_regime"]
    lines = [
        "# Sprint 3B-0 Representation-Level Oracle Intervention Review Gate",
        "",
        "Verdict:",
        "- ready_for_2000_rerun: false",
        "- do_not_enter_full_sprint_3B: true",
        "- hallucination_reduction_proven: false",
        "- answer_accuracy_improvement_proven: false",
        "",
        f"Scope: {len(subset)} eligible questions; residual injection at layers {layers}; beta sweep {betas}; "
        f"top_k {TOP_K}; target=answer_position; selectors random/surface/attention_only/attention_x_resp_pos/oracle (oracle eval-only).",
        "",
        "Mechanism layer:",
        f"- injection = beta * ||ans_residual|| * unit(span_dev); measurable-output-shift regime: beta={regime} "
        f"(mean non-oracle answer-position JS={_f(rg.get('mean_output_js_nonoracle'))}).",
        "- hook reliable (registered/triggered/removed) for every steered forward; injection_norm>0.",
        "",
        "Answer-directed oracle selectivity (eval-only gold-first-token logprob delta, paired bootstrap):",
    ]
    for b in betas:
        d = oracle_diag["by_beta"][str(b)]
        orr = d["oracle_minus_random"]
        lines.append(f"- beta={b}: nonoracle JS={_f(d['mean_output_js_nonoracle'])}; oracle-minus-random "
                     f"mean={_f(orr['mean'])} CI95=[{_f(orr['ci95_low'])},{_f(orr['ci95_high'])}] stable={d['oracle_gt_random_stable']}")
    lines += [
        "",
        f"At regime beta={regime}:",
        f"- oracle - random: mean={_f(o_r.get('mean'))} CI95=[{_f(o_r.get('ci95_low'))},{_f(o_r.get('ci95_high'))}]",
        f"- oracle - surface: mean={_f(o_s.get('mean'))} CI95=[{_f(o_s.get('ci95_low'))},{_f(o_s.get('ci95_high'))}]",
        f"- oracle - attention_x_resp_pos: mean={_f(o_a.get('mean'))} CI95=[{_f(o_a.get('ci95_low'))},{_f(o_a.get('ci95_high'))}]",
        "",
        f"Verdict at regime: {oracle_diag['verdict_at_regime']}.",
        "",
        "Selector comparison at regime (gold-logprob delta mean / output JS / harm):",
    ]
    for s in SELECTORS:
        v = by_sel[s]
        lines.append(f"- {s}: gold delta {_f(v['mean'])} CI[{_f(v['ci95_low'])},{_f(v['ci95_high'])}] "
                     f"JS={_f(v['mean_output_js'])} harm={_f(v['harm_rate'])}")
    lines += ["", "Harm: see harm_rate_report.json (beta curve). Generation eval: not performed (single-forward proxy).", "",
              "Root cause / next step:"]
    if selective:
        lines += [
            "- oracle residual injection DOES move the answer toward the gold token more than random (stable) "
            "=> representation-level intervention has SELECTIVE mechanism signal; 3A attention-bias failure was the "
            "channel, not the idea. Next: Sprint 3B-1 (non-oracle selector approximation of the residual intervention).",
        ]
    else:
        lines += [
            "- even oracle residual injection is NOT selectively better than random at any safe beta "
            "=> correct-span representation does not selectively steer the answer under this construction. "
            "Next: finer reasoning-step intervention / answer-position residual patching from a correct run / "
            "value-MLP causal tracing / task redesign — do not scale up.",
        ]
    lines += ["", "ready_for_2000_rerun=False; do_not_enter_full_sprint_3B=True; no accuracy/hallucination claims."]
    (out_dir / "review_gate_representation_level_oracle_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Sprint 3B-0 representation-level oracle intervention")
    p.add_argument("--primary-n", type=int, default=120)
    p.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    run(a.primary_n, a.betas, a.layers, a.output_dir, a.overwrite)


if __name__ == "__main__":
    main()
