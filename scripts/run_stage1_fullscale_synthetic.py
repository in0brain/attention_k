"""全规模 synthetic 验收 —— **真实样本规模 + hidden_dim 3584 + production 分析配置**。

用途：在**不接触任何真实 label** 的前提下，把 `--execute` 会走的**同一条**分析路径
从头到尾跑通一次。刻意**不用**小维度 dry-run 外推。

覆盖：A/B/C 三路径 × H1+MCQ 两臂 ／ E1/E2/E3 ／ repeat / bootstrap / 同时界 / 精度 ／
      RQ2 状态分支 ／ D 三路径 ／ M4 population declaration ／ 结果 writer。

数据是纯 synthetic：records / hidden / shadow overlay 全部现场生成，
**不读** production 的任何 label。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import conditional_increment as ci                  # noqa: E402
from recover_attention import mcq_conditional_increment as mci             # noqa: E402
from recover_attention.evaluation import v3_analysis_paths as vpaths       # noqa: E402
from recover_attention.evaluation import v3_analyze as vanalyze            # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow_f5         # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402

OUT = ROOT / "outputs/stage1/smoke/fullscale_synthetic_report_v2.json"

#: 真实规模。H1：1,508 group × 平均 5.75 trace = 8,664；MCQ：1,750 group × 1。
H1_GROUPS, H1_TRACES = 1508, 8664
MCQ_N = 1750
HIDDEN_DIM = 3584


def make_h1_fullscale(seed: int = 0, signal: float = 0.35):
    rng = np.random.default_rng(seed)
    recs: list[dict] = []
    hidden: dict[str, np.ndarray] = {}
    per = [H1_TRACES // H1_GROUPS] * H1_GROUPS
    for i in range(H1_TRACES - sum(per)):
        per[i] += 1
    for p, k in enumerate(per):
        gid = f"syn_prompt_{p:05d}"
        for j in range(k):
            tid = f"syn_trace_{p:05d}_{j}"
            lab = int(rng.random() < 0.45)
            r = {"trace_id": tid, "group_id": gid, "label": lab,
                 "example_id": f"syn_ex_{p:05d}", "family": "attack", "route": "recall",
                 "sample_type": "greedy" if j == 0 else "sampled", "sample_index": j,
                 "population_role": "confirmatory", "run_hash": "s" * 64,
                 "schema_version": "synthetic_h1_fullscale_v1",
                 "id_string_text": f"T{1000 + (p % 700):04d}",
                 # **不得**把 label 写进任何文本块 —— 否则 B3 直接看到答案，
                 # AUROC(S*) 饱和到 1.0，Δ_H 恒等于 0，H1 那一臂就白跑了。
                 "completion": f"syn completion {p}-{j} extra words here",
                 "prompt_text": f"syn prompt {p} about technique {p % 700}",
                 "id_token_positions_local": [0, 1],
                 "mentions": [{"family": "attack", "normalized": f"T{1000 + (p % 700):04d}",
                               "emitted": True, "token_indices": [0, 1]}]}
            for nm in ci.SURFACE_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            for nm in ci.F5_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            r["f5_id_token_count"] = 2
            recs.append(r)
            hidden[tid] = (rng.normal(size=HIDDEN_DIM).astype(np.float32)
                           + np.float32(signal) * np.float32(lab))
    return recs, hidden


def make_mcq_fullscale(seed: int = 1, signal: float = 0.30):
    rng = np.random.default_rng(seed)
    recs, hidden = [], {}
    for i in range(MCQ_N):
        eid = f"syn_cyber_{i:06d}"
        wrong = int(rng.random() < 0.11)
        r = {"example_id": eid, "group_id": f"syn_question_{i:06d}",
             "population_role": "confirmatory", "run_hash": "s" * 64,
             "schema_version": "synthetic_mcq_fullscale_v1",
             "sample_type": "greedy", "sample_index": 0,
             "eligible_for_primary": True, "parsed_label": "ABCD"[i % 4],
             "wrong_label": wrong,
             "semantic_output_text": f"syn option text {i}",
             "prompt_only_text": f"syn question {i} A) a B) b C) c D) d"}
        for nm in mci.MCQ_SURFACE_FEATURE_NAMES:
            r[nm] = float(rng.normal())
        for nm in mci.MCQ_F5_FEATURE_NAMES:
            r[nm] = float(rng.normal())
        recs.append(r)
        hidden[eid] = (rng.normal(size=HIDDEN_DIM).astype(np.float32)
                       + np.float32(signal) * np.float32(wrong))
    return recs, hidden


def synthetic_overlay(h1_recs, mcq_recs, seed: int = 7) -> dict:
    """现场生成一份 synthetic shadow overlay（已过门形态），语义与真实 overlay 相同。"""
    rng = np.random.default_rng(seed)

    def row(rec, arm):
        names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1"
                 else shadow_f5.MCQ_STABLE_FIELDS)
        stable = {}
        for n in names:
            if n == "f5_completion_token_count":
                stable[n] = int(rng.integers(20, 200))
            elif n == "f5_first_id_token_rank":
                stable[n] = int(rng.integers(1, 500))
            elif n == "f5_id_token_ratio":
                stable[n] = float(rng.uniform(0.0, 1.0))
            else:
                stable[n] = float(rng.normal())
        key = rec["trace_id"] if arm == "h1" else rec["example_id"]
        return {"schema": shadow_f5.SHADOW_SCHEMA, "arm": arm,
                ("trace_id" if arm == "h1" else "example_id"): key,
                "interpretation": shadow_f5.SENSITIVITY_LABEL,
                "deviation_class": shadow_f5.DEVIATION_LABEL,
                "stable_fields": stable,
                "frozen_passthrough": shadow_f5.frozen_passthrough(rec, arm)}

    return {"schema": "synthetic_overlay", "gate_passed": True,
            "interpretation": shadow_f5.SENSITIVITY_LABEL,
            "by_arm": {"h1": {r["trace_id"]: row(r, "h1") for r in h1_recs},
                       "mcq": {r["example_id"]: row(r, "mcq") for r in mcq_recs}},
            "counts": {"h1": len(h1_recs), "mcq": len(mcq_recs)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"[REFUSED] 输出已存在，拒绝覆盖：{out}", file=sys.stderr)
        return 2

    cfg = V3AnalysisConfig()                       # **production 配置**
    print(f"[cfg  ] mode={cfg.mode} R={cfg.n_repeats} folds={cfg.n_outer_folds} "
          f"B_total={cfg.b_batch * cfg.b_per_batch}")
    t0 = time.time()
    h1_recs, h1_hid = make_h1_fullscale()
    mcq_recs, mcq_hid = make_mcq_fullscale()
    print(f"[data ] H1 {len(h1_recs)} traces / {len({r['group_id'] for r in h1_recs})} groups"
          f" ｜ MCQ {len(mcq_recs)} ｜ hidden_dim {HIDDEN_DIM} ｜ {time.time() - t0:.1f}s")

    ov = synthetic_overlay(h1_recs, mcq_recs)
    e3 = {"vectors": {}, "n_traces": len(h1_recs), "n_pairs": 9916,
          "report": {"complete": True},
          "grouping_role": "E3 按预注册身份输出；**不作 primary grouping**"}

    analyze = vanalyze.make_analyze_callback(
        cfg=cfg, h1_records=h1_recs, h1_hidden=h1_hid,
        mcq_records=mcq_recs, mcq_hidden=mcq_hid,
        shadow_overlay=ov, e3_blocks=e3)
    vpaths.assert_frozen_paths(analyze.frozen_paths)
    print("[run  ] analyze 回调构造完成，开始全规模执行……")
    t1 = time.time()
    result = analyze()
    elapsed = time.time() - t1

    summary = {
        "schema": "stage1_fullscale_synthetic_report_v1",
        "synthetic": True,
        "uses_real_labels": False,
        "note": ("纯 synthetic 数据，真实样本规模 + hidden_dim 3584 + production 分析配置；"
                 "与 --execute **同一条**分析路径。不做小维度外推。"),
        "scale": {"h1_traces": len(h1_recs),
                  "h1_groups": len({r["group_id"] for r in h1_recs}),
                  "mcq": len(mcq_recs), "hidden_dim": HIDDEN_DIM},
        "config": {"mode": cfg.mode, "n_repeats": cfg.n_repeats,
                   "n_outer_folds": cfg.n_outer_folds,
                   "b_total": cfg.b_batch * cfg.b_per_batch},
        "elapsed_seconds": round(elapsed, 1),
        "paths_executed": list(result["frozen_paths"]),
        "path_markers": result["path_markers"],
        "arms": result["arms"],
        "contrast_D": {p: {k: v for k, v in d.items() if k != "draws"}
                       for p, d in result["contrast_D"].items()},
        "estimands": {"E1": result["estimands"]["E1"],
                      "E2": {a: {k: v for k, v in g.items() if not isinstance(v, list)}
                             for a, g in result["estimands"]["E2"].items()},
                      "E3": {a: {k: v for k, v in g.items() if not isinstance(v, list)}
                             for a, g in result["estimands"]["E3"].items()}},
        "rq2": result["rq2"],
        "population_declaration": result["population_declaration"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.link(str(tmp), str(out))
    os.unlink(tmp)
    print(f"[done ] {elapsed:.1f}s -> {out}")
    for p in result["frozen_paths"]:
        a = result["arms"][p]
        print(f"  {p:34s} H1 Δ_H={a['h1']['delta_h_point']:+.4f} "
              f"MCQ Δ_H={a['mcq']['delta_h_point']:+.4f} "
              f"D={result['contrast_D'][p]['D_point']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
