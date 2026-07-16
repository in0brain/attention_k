"""Sprint 4D-2 conditional-increment pipeline (W0.5-B: 实现 + smoke).

设计权威: docs/paper/preregistration.md (v2.1, frozen)。
本轮只实现 + smoke,不启动 Stage 0 全量生成。

stages:
  preflight       启动 gate:G2 preregistration hash 校验;G1 CFP flag 报告;hidden index 表。
  smoke           合成端到端:无模型 stub 数据 → ladder/gate/increment/rq2 接线跑通。
  verify_hidden   真模型 1 次前向:核验 hidden_states tuple index = block+1(forward hook 数值相等)。
  generate/cache/ladder/gate/increment/rq2   真数据阶段(需先 generate;Stage 0 受 G1+G2+smoke 门控)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ preflight
def preflight(args) -> dict:
    prereg = Path(args.prereg); lock = Path(args.lock)
    g2 = ci.check_preregistration_frozen(str(prereg), str(lock))
    cfp_confirmed = Path(args.cfp_flag).exists() if args.cfp_flag else False
    hidden = {"qwen2.5-7b_L28": ci.resolve_hidden_index(28),
              "llama-3.1-8b_L32": ci.resolve_hidden_index(32)}
    report = {
        "G1_cfp_confirmed": cfp_confirmed,
        "G2_preregistration": g2,
        "hidden_index": hidden,
        "constants": {"eps_auroc": ci.EPS_AUROC, "delta_rank_biserial": ci.DELTA_RANK_BISERIAL,
                      "rel_depth": ci.DEFAULT_REL_DEPTH, "rq2_min_per_cell": ci.RQ2_MIN_PER_CELL,
                      "rq2_lowpower_total_pos": ci.RQ2_LOWPOWER_TOTAL_POS},
        "stage0_full_generation_allowed": bool(g2["match"] and cfp_confirmed),
        "note": "Stage 0(2880 全量)需 G1+G2+smoke 齐备;W0.5-B(实现+smoke)不受 G1 阻塞。",
    }
    out = Path(args.output_dir) / "preflight_report.json"
    _write(out, report)
    print(f"[preflight] G2 hash match={g2['match']}  G1 cfp={cfp_confirmed}  → {out}")
    print(f"[preflight] hidden Qwen block19→tuple20  Llama block22→tuple23")
    return report


# --------------------------------------------------------------- synthetic smoke
def smoke(args) -> dict:
    """无模型端到端:验证 ladder/gate/increment/rq2 接线与 schema 计算,不跑真模型。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    rng = np.random.default_rng(args.seed)

    def make_task(n_groups, per, o_strength, h_strength, seed):
        r = np.random.default_rng(seed)
        groups = np.repeat(np.arange(n_groups), per)
        y = r.integers(0, 2, n_groups * per)
        O = (y * o_strength + r.normal(0, 1, y.size)).reshape(-1, 1)   # 输出侧特征(stand-in F5+text)
        H = (y * h_strength + r.normal(0, 1, y.size)).reshape(-1, 1)   # hidden 特征
        idlp = r.normal(0, 1, y.size)                                  # id-token logprob(RQ2 分层)
        s_o = ci.oof_risk_scores(O, y, groups, mk, 5, 0)
        s_oh = ci.oof_risk_scores(np.hstack([O, H]), y, groups, mk, 5, 0)
        return dict(y=y, groups=groups, O=O, H=H, idlp=idlp, s_o=s_o, s_oh=s_oh)

    # MCQ: 输出高度可分(高置信错误少);H1: 输出弱可分 + H 携带信号
    mcq = make_task(80, 1, o_strength=3.0, h_strength=0.2, seed=args.seed + 1)
    h1 = make_task(70, 4, o_strength=0.6, h_strength=1.3, seed=args.seed + 2)

    # gate: independent cross-task D
    gate = ci.independent_grouped_bootstrap_D(
        {"y": mcq["y"], "score": mcq["s_o"], "groups": mcq["groups"]},
        {"y": h1["y"], "score": h1["s_o"], "groups": h1["groups"]},
        n_boot=args.n_boot, seed=args.seed)

    # increment: paired within-task
    inc = {}
    for name, t in (("mcq", mcq), ("h1", h1)):
        d = ci.paired_grouped_bootstrap_delta(t["y"], t["s_o"], t["s_oh"], t["groups"],
                                              n_boot=args.n_boot, seed=args.seed)
        d["read"] = ci.equivalence_read(d["ci_lo"], d["ci_hi"])
        d["auroc_O"] = ci.auroc(t["s_o"], t["y"])
        d["auroc_OH"] = ci.auroc(t["s_oh"], t["y"])
        inc[name] = d

    # rq2: within-H1 分层(全体 eligible,含正负)
    thr = ci.rq2_threshold(h1["idlp"])
    hi = ci.rq2_strata(h1["idlp"], thr)
    rq2 = {"threshold": thr, "total_positives": int((h1["y"] == 1).sum()),
           "power_flag": ci.rq2_power_flag(int((h1["y"] == 1).sum())), "strata": {}}
    for lname, mask in (("high_conf", hi), ("low_conf", ~hi)):
        yc = h1["y"][mask]
        if not ci.rq2_cell_ok(yc):
            rq2["strata"][lname] = {"status": "insufficient", "n_pos": int((yc == 1).sum()),
                                    "n_neg": int((yc == 0).sum())}
            continue
        d = ci.paired_grouped_bootstrap_delta(yc, h1["s_o"][mask], h1["s_oh"][mask],
                                              h1["groups"][mask], n_boot=args.n_boot, seed=args.seed)
        rq2["strata"][lname] = {"status": "ok", "delta_point": d["point"],
                                "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}

    result = {"kind": "synthetic_smoke", "gate": gate, "increment": inc, "rq2": rq2}
    out = Path(args.output_dir) / "smoke_synthetic_report.json"
    _write(out, result)
    print(f"[smoke] gate verdict={gate['verdict']}  "
          f"H1 Δ_H={inc['h1']['point']:+.3f} {inc['h1']['read']}  "
          f"MCQ Δ_H={inc['mcq']['point']:+.3f} {inc['mcq']['read']}  → {out}")
    return result


# ---------------------------------------------------- model hidden-index verify
def verify_hidden(args) -> dict:
    """1 次前向,核验 forward-hook(目标 block 输出) == hidden_states[block+1] 数值相等。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mp = args.model_path
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    L = model.config.num_hidden_layers
    idx = ci.resolve_hidden_index(L)
    block, tuple_idx = idx["block_index_zero_based"], idx["hidden_states_tuple_index"]

    captured = {}
    layer = model.model.layers[block]

    def hook(_m, _in, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    enc = tok("The CVE identifier for Log4Shell is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    handle.remove()
    hs = out.hidden_states[tuple_idx].detach().float().cpu()
    hook_h = captured["h"]
    match = bool(torch.allclose(hook_h, hs, atol=1e-3, rtol=1e-3))
    max_abs = float((hook_h - hs).abs().max())
    report = {"model": mp, "num_layers": L, "block_index_zero_based": block,
              "hidden_states_tuple_index": tuple_idx, "hook_shape": list(hook_h.shape),
              "hidden_states_shape": list(hs.shape), "numeric_match": match, "max_abs_diff": max_abs}
    out_p = Path(args.output_dir) / "verify_hidden_report.json"
    _write(out_p, report)
    print(f"[verify_hidden] L={L} block={block} tuple_idx={tuple_idx} "
          f"match={match} max_abs_diff={max_abs:.2e}  → {out_p}")
    if not match:
        raise SystemExit("hidden tuple-index numeric verification FAILED (off-by-one?).")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="preflight,smoke")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--cfp-flag", default="docs/paper/CFP_CONFIRMED")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--output-dir", default="outputs/logs/sprint_4D_2_conditional_increment")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    fns = {"preflight": preflight, "smoke": smoke, "verify_hidden": verify_hidden}
    for s in stages:
        if s not in fns:
            raise SystemExit(f"stage not implemented in W0.5-B: {s} "
                             f"(generate/cache/ladder/gate/increment/rq2 需真数据,Stage 0 门控)")
        fns[s](args)


if __name__ == "__main__":
    main()
