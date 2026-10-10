r"""P2 第 5 步：用新表示训练探针并重拟合融合层（**CPU，无前向、无 GPU**）。

固定的训练条件（**运行前登记，不许事后改**）
    训练单元    **陈述**（10168 条记录）。870 只是**存储去重**，
                训练时每条记录各自成行 —— 重复文本的记录**不合并、不降权**。
    评价单元    trace（8664 条）—— 与现有主分析同一总体
    聚合规则    H_new(trace) = **max** over 该 trace 的陈述分数（预先指定的建模约定）
                在**未裁剪的原始分数**上做（decision_function 本身即原始 log-odds）
    分组        原 `group_id`（提示级，1508 组）
    划分        冻结的 `fold_seed(r)`，**已物化并逐位核验**；陈述**继承**其 trace 的折
    读出器      与冻结主路径同一套：L2 logistic（`cfg.base_lr_kwargs`），
                候选 = `cfg.c_grid`。**只改表示，不改读出器**
    预处理      `zscore_div_sqrt_d`，直接调 `ci._fit_dense_block`（与主路径同一函数）
    选参指标    **trace 级 AUROC（聚合之后）** —— 与估计量同级，运行前声明
    融合        meta 的 H 列 = **inner-OOF**（折外）分数聚合回 trace；
                **绝不**用训练集内预测

**不做**：不改模型/表示/读出器/聚合规则/划分；不重提取；不 GPU；
        不覆盖旧产物；不扩到 Llama 或 MCQ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_readout_probes_stage1 as drv                           # noqa: E402
from recover_attention import conditional_increment as ci             # noqa: E402
from recover_attention.evaluation import v3_stacking as vs            # noqa: E402
from recover_attention.evaluation.v3_blocks import canonical_names    # noqa: E402
from recover_attention.evaluation.v3_config import (                  # noqa: E402
    BLOCK_NAME_H, V3AnalysisConfig)

VEC = ROOT / "outputs/stage1/posthoc_readout_probes/p2_vectors/h1_qwen_block19_lasttoken.npz"
OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes/p2_probe"
STEM = "h1_qwen_newrepr_probe"

TRAINING_CONDITIONS = {
    "training_unit": "陈述（10168 条记录；870 只是存储去重，训练不合并不降权）",
    "evaluation_unit": "trace（8664）—— 与现有主分析同一总体",
    "aggregation": "max over 该 trace 的陈述分数（**预先指定的建模约定**，不是推论）",
    "aggregation_score_scale": "未裁剪的原始分数（decision_function = 原始 log-odds）",
    "grouping_key": "group_id（提示级，1508 组）",
    "splits": "冻结 fold_seed(r)；已物化并逐位核验；陈述**继承**其 trace 的折",
    "readout": "L2 logistic，cfg.base_lr_kwargs；候选 = cfg.c_grid。**只改表示，不改读出器**",
    "preprocess": "zscore_div_sqrt_d（直接调 ci._fit_dense_block，与主路径同一函数）",
    "selection_metric": "**trace 级 AUROC（聚合之后）**，运行前声明",
    "tie_break": "升序遍历 + 严格大于 → 并列保留先出现（更小的 C）",
    "fusion_h_column": "**inner-OOF**（折外）分数聚合回 trace；绝不用训练集内预测",
}


def _design(Xs: np.ndarray, tr: np.ndarray, te: np.ndarray):
    """陈述级设计矩阵：**只用训练行**拟合 z-score，再 /√d。与主路径同一函数。"""
    a, b = ci._fit_dense_block(Xs[tr], Xs[te])
    return (np.ascontiguousarray(np.asarray(a), dtype=np.float64),
            np.ascontiguousarray(np.asarray(b), dtype=np.float64))


def _fit_score(Xs, y_s, tr, te, C, cfg):
    """训一个 L2 logistic，返回 te 上的**原始** decision_function。"""
    from sklearn.linear_model import LogisticRegression
    X_tr, X_te = _design(Xs, tr, te)
    clf = LogisticRegression(**cfg.base_lr_kwargs(C))
    clf.fit(X_tr, y_s[tr])
    return np.asarray(clf.decision_function(X_te), dtype=float).ravel()


def _agg_to_trace(scores, stmt_trace_row, trace_rows, n_traces):
    """按预定规则（max）把陈述分数聚合回 trace。返回长度 n_traces 的数组（未覆盖为 nan）。"""
    out = np.full(n_traces, -np.inf)
    np.maximum.at(out, stmt_trace_row, scores)
    keep = np.full(n_traces, np.nan)
    keep[trace_rows] = out[trace_rows]
    if not np.all(np.isfinite(keep[trace_rows])):
        raise RuntimeError("聚合后有 trace 没有任何陈述分数 —— 停")
    return keep


def _select_C(Xs, y_s, y_t, groups_t, stmt_of_trace, stmt_trace_row,
              train_traces, cfg, seed_repeat, audit: list):
    """在**训练折内**选 C：嵌套 grouped inner CV，指标 = **聚合回 trace 后**的 AUROC。"""
    inner = ci.stratified_grouped_folds(
        y_t[train_traces], groups_t[train_traces], n_splits=cfg.n_inner_folds,
        seed=cfg.fold_seed(seed_repeat), max_seed_retries=cfg.inner_max_seed_retries)
    curve = []
    best_C, best_auc, best_i = None, -np.inf, None
    for i, C in enumerate(cfg.c_grid):
        aucs = []
        for itr, ite in inner:
            gtr, gte = train_traces[itr], train_traces[ite]
            s_tr = np.concatenate([stmt_of_trace[t] for t in gtr])
            s_te = np.concatenate([stmt_of_trace[t] for t in gte])
            sc = _fit_score(Xs, y_s, s_tr, s_te, C, cfg)
            agg = _agg_to_trace(sc, stmt_trace_row[s_te], gte, len(y_t))
            aucs.append(ci.auroc(agg[gte], y_t[gte]))
        m = float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else -np.inf
        curve.append({"C": float(C), "mean_inner_trace_auroc": round(m, 6),
                      "per_fold": [round(float(a), 6) for a in aucs]})
        if m > best_auc:                       # 严格大于 → 并列保留更小的 C
            best_auc, best_C, best_i = m, float(C), i
    audit.append({"chosen_C": best_C, "chosen_index": best_i,
                  "at_boundary": best_i in (0, len(cfg.c_grid) - 1),
                  "which_boundary": ("lower" if best_i == 0 else
                                     "upper" if best_i == len(cfg.c_grid) - 1 else None),
                  "selection_metric": "trace-level AUROC after max-aggregation",
                  "curve": curve})
    return best_C


def run(cfg: V3AnalysisConfig, arm_bundle, vec, *, verbose=True) -> dict:
    y_t = np.asarray(arm_bundle["y"])
    groups_t = np.asarray(arm_bundle["groups"]).astype(str)
    blocks = arm_bundle["blocks"]
    n_t = y_t.size
    Xs, y_s = vec["Xs"], vec["y_s"]
    stmt_trace_row = vec["stmt_trace_row"]
    stmt_of_trace = vec["stmt_of_trace"]
    names_s, names_sh = canonical_names()

    oof = {k: [np.full(n_t, np.nan) for _ in range(cfg.n_repeats)]
           for k in ("S*", "H_new", "S*+H_new")}
    fold_of = [np.full(n_t, -1, dtype=int) for _ in range(cfg.n_repeats)]
    audits: list[dict] = []
    leak: list[dict] = []
    t0 = time.perf_counter()

    for r in range(cfg.n_repeats):
        folds = ci.stratified_grouped_folds(
            y_t, groups_t, n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r),
            max_seed_retries=cfg.outer_max_seed_retries)
        for f, (train, test) in enumerate(folds):
            fold_of[r][test] = f
            # 分组隔离（外层）
            if set(groups_t[train]) & set(groups_t[test]):
                raise RuntimeError(f"r{r} f{f}: 外层训练/测试**组重叠** —— 停")
            cache = vs.build_base_block_cache(
                y=y_t, groups=groups_t, blocks=blocks, repeat=r, outer_fold=f,
                train=train, test=test, cfg=cfg)
            s_star = vs.fit_stack(cache, names_s, y_t, cfg)

            # ── H_new 的 meta-train 列 = **inner-OOF**，聚合回 trace ──────
            inner = ci.stratified_grouped_folds(
                y_t[train], groups_t[train], n_splits=cfg.n_inner_folds,
                seed=cfg.fold_seed(r), max_seed_retries=cfg.inner_max_seed_retries)
            meta_tr = np.full(train.size, np.nan)
            pos_in_train = {int(t): i for i, t in enumerate(train)}
            for j, (itr, ite) in enumerate(inner):
                gtr, gte = train[itr], train[ite]
                if set(groups_t[gtr]) & set(groups_t[gte]):
                    raise RuntimeError(f"r{r} f{f} inner{j}: 内层**组重叠** —— 停")
                a: list = []
                C = _select_C(Xs, y_s, y_t, groups_t, stmt_of_trace, stmt_trace_row,
                              gtr, cfg, r, a)
                s_tr = np.concatenate([stmt_of_trace[t] for t in gtr])
                s_te = np.concatenate([stmt_of_trace[t] for t in gte])
                # **折外**：打分的 trace 一条都不在训练侧
                if set(gtr.tolist()) & set(gte.tolist()):
                    raise RuntimeError("inner 训练/打分 trace 重叠 —— 停")
                sc = _fit_score(Xs, y_s, s_tr, s_te, C, cfg)
                agg = _agg_to_trace(sc, stmt_trace_row[s_te], gte, n_t)
                for t in gte:
                    meta_tr[pos_in_train[int(t)]] = agg[t]
                leak.append({"repeat": r, "fold": f, "inner": j,
                             "n_train_traces": int(gtr.size),
                             "n_scored_traces": int(gte.size),
                             "scored_traces_in_train": 0,
                             "group_overlap": 0})
                audits.append({"repeat": r, "fold": f, "phase": "inner_cf",
                               "inner_fold": j, **a[0]})
            if not np.all(np.isfinite(meta_tr)):
                raise RuntimeError(f"r{r} f{f}: inner-OOF 未覆盖全部训练 trace —— 停")

            # ── 外层最终：在**整个训练折**上选参 + 拟合，给测试折打分 ──────
            a2: list = []
            C2 = _select_C(Xs, y_s, y_t, groups_t, stmt_of_trace, stmt_trace_row,
                           train, cfg, r, a2)
            s_tr = np.concatenate([stmt_of_trace[t] for t in train])
            s_te = np.concatenate([stmt_of_trace[t] for t in test])
            sc = _fit_score(Xs, y_s, s_tr, s_te, C2, cfg)
            h_test = _agg_to_trace(sc, stmt_trace_row[s_te], test, n_t)
            audits.append({"repeat": r, "fold": f, "phase": "outer_full",
                           "inner_fold": None, **a2[0]})

            # ── 换 H 列 → 重拟合融合层 ────────────────────────────────────
            cache.meta_train_scores[BLOCK_NAME_H] = meta_tr
            cache.test_scores[BLOCK_NAME_H] = h_test[test]
            sh = vs.fit_stack(cache, names_sh, y_t, cfg)

            oof["S*"][r][test] = s_star
            oof["H_new"][r][test] = h_test[test]
            oof["S*+H_new"][r][test] = sh
            if verbose:
                print(f"  r{r} f{f}  C_outer={C2}  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

    for k in oof:
        for r in range(cfg.n_repeats):
            if not np.all(np.isfinite(oof[k][r])):
                raise RuntimeError(f"{k} r{r}: OOF 有非有限值 —— 停")
    return {"oof": oof, "fold_of": fold_of, "audits": audits,
            "leak_checks": leak, "secs": round(time.perf_counter() - t0, 1)}


def summarise(cfg, y_t, groups_t, res) -> dict:
    from recover_attention.evaluation import v3_bootstrap as vb
    oof = res["oof"]
    per = []
    for r in range(cfg.n_repeats):
        a_s = ci.auroc(oof["S*"][r], y_t)
        a_h = ci.auroc(oof["H_new"][r], y_t)
        a_sh = ci.auroc(oof["S*+H_new"][r], y_t)
        per.append({"repeat": r, "auroc_S*": a_s, "auroc_H_new": a_h,
                    "auroc_S*+H_new": a_sh, "delta_H_new": a_sh - a_s})
    point = {k: float(np.mean([p[k] for p in per]))
             for k in ("auroc_S*", "auroc_H_new", "auroc_S*+H_new", "delta_H_new")}
    plan = vb.build_resample_plan(groups_t, cfg)
    draws, n_disc = [], 0
    for _q, _j, rows in plan.iter_draws():
        ys = y_t[rows]
        vals = []
        ok = True
        for r in range(cfg.n_repeats):
            a_s = ci.auroc(oof["S*"][r][rows], ys)
            a_sh = ci.auroc(oof["S*+H_new"][r][rows], ys)
            if not (np.isfinite(a_s) and np.isfinite(a_sh)):
                ok = False
                break
            vals.append(a_sh - a_s)
        if not ok:
            n_disc += 1
            continue
        draws.append(float(np.mean(vals)))
    return {"per_repeat": per, "point": point,
            "delta_H_new_ci": vb.percentile_ci(np.asarray(draws), cfg.alpha),
            "bootstrap": {"b_total": plan.b_total, "discarded_single_class": n_disc,
                          "unit": "cluster/group bootstrap（同主路径）"}}


def load_vectors(cfg, bundle) -> dict:
    with np.load(VEC, allow_pickle=False) as z:
        V = z["unique_vectors"]
        rows = z["statement_row"]
        stmt_trace = z["trace_id"].astype(str)
        y_s = z["y"].astype(int)
        stmt_id = z["statement_id"].astype(str)
        folds_npz = {f"fold_r{r}": z[f"fold_r{r}"] for r in range(cfg.n_repeats)}
    trace_ids = np.asarray(bundle["sample_ids"]).astype(str)
    pos = {t: i for i, t in enumerate(trace_ids)}
    missing = sorted({t for t in stmt_trace if t not in pos})
    if missing:
        raise RuntimeError(f"{len(missing)} 个陈述 trace 不在臂名单里 —— 停：{missing[:3]}")
    stmt_trace_row = np.asarray([pos[t] for t in stmt_trace], dtype=np.int64)
    # 每个 trace 的陈述索引；每个 trace 必须至少有一条
    order = np.argsort(stmt_trace_row, kind="stable")
    stmt_of_trace = [np.empty(0, dtype=np.int64)] * len(trace_ids)
    for t, idx in zip(*np.unique(stmt_trace_row[order], return_index=True)):
        pass
    buckets: dict[int, list[int]] = {}
    for si, ti in enumerate(stmt_trace_row):
        buckets.setdefault(int(ti), []).append(si)
    empty = [i for i in range(len(trace_ids)) if i not in buckets]
    if empty:
        raise RuntimeError(f"{len(empty)} 个 trace 没有任何陈述 —— 停")
    stmt_of_trace = [np.asarray(buckets[i], dtype=np.int64) for i in range(len(trace_ids))]
    # 标签一致性：陈述标签 max 聚合 == trace 标签
    agg = np.zeros(len(trace_ids), dtype=int)
    np.maximum.at(agg, stmt_trace_row, y_s)
    if not np.array_equal(agg, np.asarray(bundle["y"]).astype(int)):
        raise RuntimeError("陈述标签 max 聚合 ≠ trace 标签 —— 停")
    # 划分一致性：npz 里的折 == 本次重算的折
    for r in range(cfg.n_repeats):
        folds = ci.stratified_grouped_folds(
            np.asarray(bundle["y"]), np.asarray(bundle["groups"]).astype(str),
            n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r),
            max_seed_retries=cfg.outer_max_seed_retries)
        fo = np.full(len(trace_ids), -1, dtype=np.int64)
        for f, (_, te) in enumerate(folds):
            fo[te] = f
        if not np.array_equal(fo[stmt_trace_row], folds_npz[f"fold_r{r}"]):
            raise RuntimeError(f"repeat{r}: npz 里的陈述折 ≠ 本次重算 —— 停")
    Xs = np.ascontiguousarray(V[rows], dtype=np.float64)   # (10168, 3584)
    return {"Xs": Xs, "y_s": y_s, "stmt_trace_row": stmt_trace_row,
            "stmt_of_trace": stmt_of_trace, "statement_id": stmt_id,
            "n_unique": int(V.shape[0]), "d": int(V.shape[1])}


def cmd_self_check() -> int:
    """**合成**自检：小规模造数据，验收折外性、聚合、分组隔离、确定性。"""
    print("═══ 合成自检（无前向、无 GPU、不读真实向量）")
    ok = []

    def ck(name, v, detail=""):
        ok.append(bool(v))
        print(f"  [{'ok  ' if v else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    rng = np.random.default_rng(0)
    n_g, per_g = 40, 6
    n_t = n_g * per_g
    groups = np.asarray([f"g{i}" for i in range(n_g) for _ in range(per_g)])
    y_t = (rng.random(n_t) < 0.3).astype(int)
    # 每个 trace 1~3 条陈述；陈述标签 max 聚合 == trace 标签
    stmt_trace_row, y_s = [], []
    for t in range(n_t):
        k = int(rng.integers(1, 4))
        labs = np.zeros(k, dtype=int)
        if y_t[t] == 1:
            labs[rng.integers(0, k)] = 1
        for j in range(k):
            stmt_trace_row.append(t)
            y_s.append(int(labs[j]))
    stmt_trace_row = np.asarray(stmt_trace_row)
    y_s = np.asarray(y_s)
    d = 12
    Xs = rng.normal(size=(len(y_s), d))
    Xs[y_s == 1] += 0.9                      # 植入可学的信号
    stmt_of_trace = [np.where(stmt_trace_row == t)[0] for t in range(n_t)]
    agg = np.zeros(n_t, dtype=int)
    np.maximum.at(agg, stmt_trace_row, y_s)
    ck("合成数据的陈述标签 max 聚合 == trace 标签", np.array_equal(agg, y_t))

    # 聚合规则 = max
    sc = rng.normal(size=len(y_s))
    a = _agg_to_trace(sc, stmt_trace_row, np.arange(n_t), n_t)
    manual = np.asarray([sc[stmt_of_trace[t]].max() for t in range(n_t)])
    ck("聚合 = max（与逐 trace 手算逐元素相同）", np.allclose(a, manual, rtol=0, atol=0))

    # 折外性 + 分组隔离
    cfg = V3AnalysisConfig()
    folds = ci.stratified_grouped_folds(y_t, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(0), max_seed_retries=8)
    ck("外层：同一提示组不跨折",
       all(not (set(groups[tr]) & set(groups[te])) for tr, te in folds))
    tr0, te0 = folds[0]
    inner = ci.stratified_grouped_folds(y_t[tr0], groups[tr0], n_splits=cfg.n_inner_folds,
                                        seed=cfg.fold_seed(0), max_seed_retries=8)
    ck("内层：同一提示组不跨折",
       all(not (set(groups[tr0[a_]]) & set(groups[tr0[b_]])) for a_, b_ in inner))
    ck("内层：打分的 trace 一条都不在该折训练侧",
       all(not (set(tr0[a_].tolist()) & set(tr0[b_].tolist())) for a_, b_ in inner))

    # 融合列必须是折外：同一 trace 不能既在训练侧又被打分
    covered = np.zeros(tr0.size, dtype=int)
    for a_, b_ in inner:
        covered[b_] += 1
    ck("inner-OOF 恰好覆盖每个训练 trace 一次", bool(np.all(covered == 1)),
       f"覆盖次数取值 {sorted(set(covered.tolist()))}")

    # 确定性
    s1 = _fit_score(Xs, y_s, np.arange(len(y_s) // 2), np.arange(len(y_s) // 2, len(y_s)),
                    1.0, cfg)
    s2 = _fit_score(Xs, y_s, np.arange(len(y_s) // 2), np.arange(len(y_s) // 2, len(y_s)),
                    1.0, cfg)
    ck("同输入同参数 → 逐元素可复现", np.array_equal(s1, s2))

    # 预处理只用训练行的统计量
    tr, te = np.arange(60), np.arange(60, 120)
    A, _B = _design(Xs, tr, te)
    ck("训练侧 z-score 后均值≈0、方差≈1/d（zscore_div_sqrt_d）",
       abs(float(A.mean())) < 1e-9
       and abs(float(A.var(axis=0).mean()) - 1.0 / Xs.shape[1]) < 1e-6,
       f"mean={A.mean():.2e}  var*d={A.var(axis=0).mean() * Xs.shape[1]:.6f}")

    print(f"\n[{'PASS' if all(ok) else 'FAIL'}] {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    if args.self_check:
        return cmd_self_check()
    if not args.run:
        print("用 --self-check 或 --run")
        return 1

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    t0 = time.perf_counter()
    bundle = drv.load_arm("h1", cfg)
    roster = drv._roster("h1", bundle, cfg)
    vec = load_vectors(cfg, bundle)
    print(f"[load] trace {roster['n']}  组 {roster['n_groups']}  "
          f"陈述 {len(vec['y_s'])}  唯一 {vec['n_unique']}  d {vec['d']}")

    res = run(cfg, bundle, vec)
    y_t = np.asarray(bundle["y"])
    groups_t = np.asarray(bundle["groups"]).astype(str)
    summ = summarise(cfg, y_t, groups_t, res)

    tune = {"n_selections": len(res["audits"]),
            "by_phase": {}, "chosen_C_counts": {}}
    for a in res["audits"]:
        ph = tune["by_phase"].setdefault(a["phase"], {"n": 0, "at_boundary": 0,
                                                      "upper": 0, "lower": 0,
                                                      "chosen_C_counts": {}})
        ph["n"] += 1
        ph["at_boundary"] += int(bool(a["at_boundary"]))
        ph["upper"] += int(a["which_boundary"] == "upper")
        ph["lower"] += int(a["which_boundary"] == "lower")
        k = str(a["chosen_C"])
        ph["chosen_C_counts"][k] = ph["chosen_C_counts"].get(k, 0) + 1
        tune["chosen_C_counts"][k] = tune["chosen_C_counts"].get(k, 0) + 1

    out = {
        "schema": "p2_newrepr_probe_v1",
        "representation": {
            "source": str(VEC.relative_to(ROOT)),
            "what": "Qwen block 19 → hidden_states[20] 的**末词元**向量",
            "n_unique_vectors": vec["n_unique"], "d": vec["d"],
            "n_statement_rows": int(len(vec["y_s"]))},
        "training_conditions": TRAINING_CONDITIONS,
        "roster": roster,
        "results": {
            "auroc_S*": summ["point"]["auroc_S*"],
            "auroc_H_new": summ["point"]["auroc_H_new"],
            "auroc_S*+H_new": summ["point"]["auroc_S*+H_new"],
            "delta_H_new": summ["point"]["delta_H_new"],
            "delta_H_new_ci": summ["delta_H_new_ci"],
            "per_repeat": summ["per_repeat"]},
        "bootstrap": summ["bootstrap"],
        "tuning_summary": tune,
        "leak_checks": {
            "n_inner_folds_checked": len(res["leak_checks"]),
            "all_scored_traces_outside_train": all(
                x["scored_traces_in_train"] == 0 for x in res["leak_checks"]),
            "all_group_disjoint": all(x["group_overlap"] == 0
                                      for x in res["leak_checks"]),
            "fusion_h_column": "inner-OOF 聚合回 trace；**非**训练集内预测"},
        "limitations_that_must_travel_with_this_result": [
            "**跨折文本重复**：10168 条陈述只有 870 种文本，三轮各约 93.4~93.8% 的"
            "测试折文本已在训练折出现 → 结果**不能**解释为"
            "『识别未见过的编号』的能力",
            "**下游 NaN 未修复**：提取时 870/870 条在 hidden_states[28] 与 logits 上"
            "有非有限值（起源在 block 27 内部）；必查范围 0..20 零异常只是提取的前提，"
            "**不等于**后端数值健康",
            "**既有 B4 完整性限制继续保留**（h1_b4_numeric_integrity_compromised）；"
            "本结果不解释、不解除它",
            "层 20 有限 ≠ 该层表示的**数值质量**已验证",
            "任务名称：**采用作者独立陈述末词元表示思路的 Qwen/H1 编号存在性任务适配**；"
            "**不得**称『复现了 Han 的方法』，不得与原文数字并列",
            "这是**事后分析**，不是预注册条件；与事前条件分开呈现",
        ],
        "secs": {"train": res["secs"], "total": round(time.perf_counter() - t0, 1)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{STEM}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        args.out_dir / f"{STEM}_oof.npz",
        sample_id=np.asarray(bundle["sample_ids"]).astype(str),
        group_id=groups_t, y=y_t,
        **{f"r{r}_fold": res["fold_of"][r] for r in range(cfg.n_repeats)},
        **{f"r{r}_{k}": res["oof"][k][r] for r in range(cfg.n_repeats)
           for k in ("S*", "H_new", "S*+H_new")})
    (args.out_dir / f"{STEM}_tuning_audit.json").write_text(
        json.dumps(res["audits"], ensure_ascii=False, indent=1, default=float) + "\n",
        encoding="utf-8")

    p = out["results"]
    print(f"\n{'':<14}{'S*':>10}{'H_new':>10}{'S*+H_new':>11}{'Δ_H_new':>11}"
          f"{'95% CI':>26}")
    print(f"{'h1 (新表示)':<14}{p['auroc_S*']:>10.6f}{p['auroc_H_new']:>10.6f}"
          f"{p['auroc_S*+H_new']:>11.6f}{p['delta_H_new']:>+11.6f}"
          f"   [{p['delta_H_new_ci']['ci_lo']:+.6f}, {p['delta_H_new_ci']['ci_hi']:+.6f}]")
    print(f"\n选参  {tune['chosen_C_counts']}  按阶段 "
          f"{ {k: v['chosen_C_counts'] for k, v in tune['by_phase'].items()} }")
    print(f"训练 {res['secs']}s   [out] {args.out_dir / f'{STEM}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
