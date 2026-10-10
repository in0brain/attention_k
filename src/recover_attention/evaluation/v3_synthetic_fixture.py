"""synthetic fixture 与真实的 end-to-end dry-run 执行体。

**全部数据由本模块合成**：不读任何真实 finalized 产物、不接触真实 label。
臂内 E1 路径走的是**与 --execute 相同的计算路径**（adapter → repeats → pooled bootstrap →
点估计 → 界 → MC 诊断 → 判读），输入是 synthetic，且**真实 fixture B**（不缩水成元数据）。
两臂 D 只有**数学 smoke**（`contrast_math_smoke`）——冻结的 output-only O 尚未接线。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention import mcq_conditional_increment as mci
from recover_attention.evaluation import v3_adapter as adapter
from recover_attention.evaluation import v3_adapter_mcq as amcq
from recover_attention.evaluation import v3_arm_contrast as contrast
from recover_attention.evaluation import v3_bootstrap as vb
from recover_attention.evaluation import v3_repeat_estimand as vre
from recover_attention.evaluation.v3_blocks import BLOCK_NAMES_S_STAR
from recover_attention.evaluation.v3_config import V3AnalysisConfig

SYNTHETIC = True
HD = 16


def synthetic_cfg(*, b_batch: int = 4, b_per_batch: int = 25, **over) -> V3AnalysisConfig:
    """fixture 用的**真实 B**（b_batch × b_per_batch），只是规模小，语义不变。"""
    # mode="dry_run" 是既有的**非 production** 路径：assert_production 在该模式下放行
    # 冻结参数覆盖。真实 --execute 必须用 mode="production"（B_total=20,000）。
    return dataclasses.replace(V3AnalysisConfig(), mode="dry_run", b_batch=b_batch,
                               b_per_batch=b_per_batch, **over)


def make_h1(n_prompts: int = 30, k: int = 6, seed: int = 0, signal: float = 1.1):
    rng = np.random.default_rng(seed)
    recs, hidden = [], {}
    for p in range(n_prompts):
        gid = f"synthetic_prompt_{p:04d}"
        for j in range(k):
            tid = f"synthetic_trace_{p:04d}_{j}"
            lab = int(rng.random() < 0.45)
            r = {"trace_id": tid, "group_id": gid, "label": lab,
                 "example_id": f"synthetic_ex_{p:04d}", "family": "attack", "route": "recall",
                 "sample_type": "greedy" if j == 0 else "sampled", "sample_index": j,
                 "population_role": "confirmatory", "run_hash": "s" * 64,
                 "schema_version": "synthetic_h1_v1",
                 "id_string_text": f"T{1000 + (p % 90):04d}",
                 "completion": f"synthetic completion {p}-{j} label{lab}",
                 "prompt_text": f"synthetic prompt {p}",
                 "id_token_positions_local": [0, 1],
                 "mentions": [{"family": "attack", "normalized": f"T{1000 + (p % 90):04d}",
                               "emitted": True, "token_indices": [0, 1]}]}
            for nm in ci.SURFACE_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            for nm in ci.F5_FEATURE_NAMES:
                r[nm] = float(rng.normal())
            recs.append(r)
            hidden[tid] = rng.normal(size=HD) + signal * lab
    return recs, hidden


def make_mcq(n_q: int = 30, seed: int = 1, signal: float = 0.9):
    rng = np.random.default_rng(seed)
    recs, hidden = [], {}
    for i in range(n_q):
        eid = f"synthetic_cyber_{i:06d}"
        wrong = int(rng.random() < 0.4)
        r = {"example_id": eid, "group_id": f"synthetic_question_{i:06d}",
             "population_role": "confirmatory", "run_hash": "s" * 64,
             "schema_version": "synthetic_mcq_v1", "sample_type": "greedy", "sample_index": 0,
             "eligible_for_primary": True, "parsed_label": "ABCD"[i % 4],
             "wrong_label": wrong,
             "semantic_output_text": f"synthetic option text {i}",
             "prompt_only_text": f"synthetic question {i} A) a B) b C) c D) d"}
        for nm in mci.MCQ_SURFACE_FEATURE_NAMES:
            r[nm] = float(rng.normal())
        for nm in mci.MCQ_F5_FEATURE_NAMES:
            r[nm] = float(rng.normal())
        recs.append(r)
        hidden[eid] = rng.normal(size=HD) + signal * wrong
    return recs, hidden


def _arm_bundle(arm: str, cfg: V3AnalysisConfig, seed: int):
    if arm == "h1":
        recs, hid = make_h1(seed=seed)
        return adapter.build_arm(recs, hid), recs, hid
    recs, hid = make_mcq(seed=seed)
    return amcq.build_mcq_arm(amcq.filter_analyzable(recs), hid), recs, hid


def run_arm_end_to_end(arm: str, cfg: V3AnalysisConfig, *, seed: int = 0) -> dict:
    """与 --execute **同一条计算路径**，输入为 synthetic。"""
    bundle, _recs, _hid = _arm_bundle(arm, cfg, seed)
    y, groups, blocks = bundle["y"], bundle["groups"], bundle["blocks"]
    repeats = vre.run_all_repeats(y=y, groups=groups, blocks=blocks, cfg=cfg)
    block_names = list(BLOCK_NAMES_S_STAR)
    point = vre.point_estimates(y, repeats, block_names)
    plan = vb.build_resample_plan(groups, cfg)
    draws = vb.pooled_draws(y, repeats, block_names, plan)
    dh = np.asarray(draws[vre.DELTA_H], dtype=float)
    ci_dh = vb.percentile_ci(dh, cfg.alpha)
    jack = vb.jackknife_mc(dh, draws["batch_of"], cfg.b_batch, cfg.alpha)
    bm = vb.batch_means_mc(dh, draws["batch_of"], cfg.b_batch, cfg.alpha)
    verdict = vb.precision_verdict_v3(ci_dh["width"], jack["mc_band_width"],
                                      cfg.precision_max_ci_width)
    nine = vb.certify_non_inferiority(point[vre.DELTA_K],
                                      {k: np.asarray(v) for k, v in draws[vre.DELTA_K].items()},
                                      cfg.eps_ni, cfg.alpha)
    return {
        "synthetic": SYNTHETIC, "arm": arm, "n": bundle["n"],
        "n_repeats": int(cfg.n_repeats), "b_total": int(cfg.b_batch * cfg.b_per_batch),
        "n_valid_draws": int(draws["n_draws"]),
        "n_discarded_single_class": int(draws["n_discarded_single_class"]),
        "delta_h_point": float(point[vre.DELTA_H]),
        "delta_h_ci": ci_dh, "mc_jackknife": {k: jack[k] for k in
                                              ("se_jack", "mc_band_width", "t_quantile")},
        "mc_batch_means": {"se_mc": bm["se_mc"]},
        "precision": verdict, "non_inferiority": {"certified": nine["certified"],
                                                  "status": nine["status"]},
        "block_names": block_names,
        "ensemble_estimand_separate": True,
    }


def contrast_math_smoke(cfg: V3AnalysisConfig, *, seed: int = 0) -> dict:
    """两臂 D 的**数学 smoke**，**不是** end-to-end。

    这里用 S*+H 的 OOF 作分数占位，而冻结定义的 O 是 **output-only**
    （O_H1 = H1 F5 + completion full text ； O_MCQ = MCQ F5 + semantic_output_text）。
    因此本函数只验证 D 的算术、独立重采样与判读分支，**不验证 O 的接线**，
    也**不得**进入正式结果或被称作 O 接线测试。
    """
    bh, _r, _h = _arm_bundle("h1", cfg, seed)
    bm_, _r2, _h2 = _arm_bundle("mcq", cfg, seed + 1)
    oof_h = [r["oof"]["S*+H"] for r in vre.run_all_repeats(
        y=bh["y"], groups=bh["groups"], blocks=bh["blocks"], cfg=cfg)]
    oof_m = [r["oof"]["S*+H"] for r in vre.run_all_repeats(
        y=bm_["y"], groups=bm_["groups"], blocks=bm_["blocks"], cfg=cfg)]
    out = contrast.contrast_d(y_mcq=bm_["y"], groups_mcq=bm_["groups"], oof_mcq=oof_m,
                              y_h1=bh["y"], groups_h1=bh["groups"], oof_h1=oof_h, cfg=cfg)
    out["synthetic"] = SYNTHETIC
    out["is_end_to_end"] = False
    out["uses_frozen_O_definition"] = False
    out["kind"] = "contrast_math_smoke"
    out["O_note"] = ("以 S*+H 的 OOF 作任意分数占位，只验证 D 的算术与判读；"
                     "**不是**冻结 O 定义（output-only）下的 end-to-end；"
                     "真实 --execute 必须用 O_H1 = H1 F5 + completion full text、"
                     "O_MCQ = MCQ F5 + semantic_output_text 构造的 OOF")
    return out
