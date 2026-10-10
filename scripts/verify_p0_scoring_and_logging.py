r"""P0 小规模验证：评分通道与拟合日志。

逐条对应待办 P0：

    1 保留旧裁剪条件，新增原始分数条件
      XGB `output_margin=True`；MLP 取 sigmoid 前输出，
      并验证 `sigmoid(logit)` 与 `predict_proba` 对应
    2 原始分数 / 预测概率 / 融合分数**分别保存**，不从裁剪概率反推
    3 内层选参、内部交叉拟合、外层最终预测**三处都核**
    4 每次实际拟合都留：阶段 / 划分标识 / 参数 / 种子 / 实际迭代 / 预算 /
      收敛告警 / 耗时；驱动里两处 `_` 已修
    5 候选验证记录 vs 选中模型最终拟合记录，**分开**
    6 极端概率、正类方向、无二次反转、零值/缺失语义、分组隔离
    7 **正式分片 writer 的 round-trip**：写盘 → 关闭 → 重读 → 逐值比对；
      并检验续跑判据（日志缺失 / 指纹不符 / legacy 分片都判未完成）

⚠️ 第 7 节是补上的关键缺口：1–6 只检查 `run_fold` 返回的**内存 dict**，
   内存里对不等于盘上有。旧版正是"run_fold 算了六路分数、cmd_run 只写三路"。

**只用合成数据 + 真实产物的一折**，不跑整臂；第 7 节写临时目录，跑完即删。
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
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_readout_probe_lib as lib                               # noqa: E402
import posthoc_readout_probes_stage1 as drv                           # noqa: E402
from recover_attention import conditional_increment as ci             # noqa: E402
from recover_attention.evaluation.v3_config import (                  # noqa: E402
    BLOCK_NAME_H, V3AnalysisConfig)

OUT = ROOT / "outputs/stage1/posthoc_readout_probes/p0_validation.json"
CHECKS: list[dict] = []


def ck(name: str, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return bool(ok)


def synthetic(n=400, d=24, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.7).astype(int)          # 约 30% 正例
    X = rng.normal(size=(n, d)) + 2.0 * y[:, None]  # 可分，逼出极端概率
    return X, y


def _refuses_legacy(drv, cfg, conds) -> bool:
    """正式运行必须**拒绝**写 legacy_v1 —— 旧六条件报告与 OOF 是唯一一份。"""
    try:
        drv.cmd_run("mcq", cfg, conds, 1.0, version=drv.LEGACY_OUTPUT_VERSION)
    except SystemExit as e:
        return "legacy_v1" in str(e)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("mcq", "h1"), default="mcq")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    t0 = time.perf_counter()
    rec: dict = {"schema": "p0_scoring_logging_validation_v1", "arm": args.arm}

    # ── 1 条件表：旧的保留，新的加上 ──────────────────────────────────────
    print("═══ 1 条件表")
    ck("旧裁剪条件原样保留",
       all(lib.CONDITION_BY_NAME[c].score_mode == "clipped_logodds"
           for c in ("saplma_mlp", "han_xgboost", "han_l1", "han_l1_wide",
                     "ref_l2_frozen_grid", "ref_l2_wide_grid")))
    ck("新增原始分数条件",
       all(lib.CONDITION_BY_NAME[c].score_mode == "raw_margin"
           for c in ("saplma_mlp_raw", "han_xgboost_raw")))
    for a, b in (("saplma_mlp", "saplma_mlp_raw"), ("han_xgboost", "han_xgboost_raw")):
        A, B = lib.CONDITION_BY_NAME[a], lib.CONDITION_BY_NAME[b]
        ck(f"{b} 与 {a} **只差打分口径**",
           A.candidates == B.candidates and A.kind == B.kind
           and A.provenance == B.provenance)
    ck("新条件标为事后敏感性，不混入事前组",
       {"saplma_mlp_raw", "han_xgboost_raw"} <= set(lib.POSTHOC_SENSITIVITY)
       and not ({"saplma_mlp_raw", "han_xgboost_raw"} & set(lib.PRESPECIFIED)))

    # ── 2 三路分数：互不反推 + 一致性 ─────────────────────────────────────
    print("\n═══ 2 原始分数 / 概率 / 融合分数")
    X, y = synthetic()
    tri: dict = {}
    for kind, params in (("mlp", {"hidden_layer_sizes": (32, 16), "alpha": 1e-3,
                                  "max_iter": 60}),
                         ("xgb", {"max_depth": 3, "n_estimators": 60,
                                  "learning_rate": 0.1})):
        for mode in ("clipped_logodds", "raw_margin"):
            fus, info = lib._fit_score(kind, params, X, y, X, seed=0, score_mode=mode,
                                       ctx={"stage": "p0_validation"})
            tri[f"{kind}/{mode}"] = {
                "fusion_min": float(fus.min()), "fusion_max": float(fus.max()),
                "raw_min": info["raw_min"], "raw_max": info["raw_max"],
                "proba_min": info["proba_min"], "proba_max": info["proba_max"],
                "clip_boundary_rate": info["clip_boundary_rate"],
                "n_iter": info["n_iter"], "max_iter_used": info["max_iter_used"]}
            if mode == "raw_margin":
                ck(f"{kind}: raw_margin 下融合分数 == 原始分数",
                   np.array_equal(fus, info["raw"]))
            else:
                exp = lib._to_log_odds(info["proba"])
                ck(f"{kind}: clipped 下融合分数 == 裁剪概率的 log-odds",
                   np.allclose(fus, exp, rtol=0, atol=0))
            ck(f"{kind}/{mode}: raw 与 proba 都在，且不是同一个数组",
               info["raw"] is not None and info["proba"] is not None
               and info["raw"] is not info["proba"])
        # sigmoid(raw) 应当等于 proba —— 这条证明 raw 不是从裁剪概率反推的
        _, i2 = lib._fit_score(kind, params, X, y, X, seed=0, score_mode="raw_margin")
        back = 1.0 / (1.0 + np.exp(-i2["raw"]))
        d = float(np.max(np.abs(back - i2["proba"])))
        ck(f"{kind}: sigmoid(原始分数) 与 predict_proba 一致（≤1e-6）", d <= 1e-6,
           f"最大差 {d:.3e}")
        key = "logit_proba_check" if kind == "mlp" else "margin_proba_check"
        ck(f"{kind}: 拟合记录里带 {key}", key in i2, i2.get(key))
    rec["three_channels"] = tri

    # ── 3 极端概率下裁剪确实发生、原始分数确实不同 ────────────────────────
    print("\n═══ 3 极端概率")
    # (a) **先直接验裁剪本身**，不依赖模型能不能训到饱和 ——
    #     第一版靠合成数据"碰运气"触边界，没碰到，这条检查等于没牙。
    p_ext = np.array([0.0, 1e-12, lib.PROB_EPS, 0.5, 1 - lib.PROB_EPS, 1.0])
    lo = lib._to_log_odds(p_ext)
    sat = abs(float(lo[0]))
    ck("_to_log_odds 把 0/1 夹住且结果有限",
       bool(np.all(np.isfinite(lo))), f"eps={lib.PROB_EPS}  饱和值 ±{sat:.4f}")
    ck("裁剪确实制造并列：p=0 与 p=1e-12 映到同一个分数",
       float(lo[0]) == float(lo[1]))

    # (b) **在已保存的真实 OOF 上量裁剪到达率** —— 这才是有牙的检查。
    #     合成数据训不到 sigmoid 饱和（实测 |logit| 只到 ~9.9 < 13.82），
    #     靠它断言"确有样本触边界"等于没牙；真实 H1 的 MLP 则大量饱和。
    #     旧 OOF 只存了融合分数，所以这里**只量到达率与并列**，
    #     **不能**由它恢复裁剪前的排序（原始分数当时没落盘）。
    boundary = {}
    for a in ("h1", "mcq"):
        for c in ("saplma_mlp", "han_xgboost"):
            f = ROOT / f"outputs/stage1/posthoc_readout_probes/oof_{a}_{c}.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                yy = z["y"]
                tot = hit = 0
                ties = {}
                for r in range(3):
                    h = z[f"r{r}_H"]
                    tot += h.size
                    hit += int(np.sum(np.isclose(np.abs(h), sat, rtol=0, atol=1e-9)))
                    m = np.isclose(h, h.min(), rtol=0, atol=1e-12)
                    if m.sum() > 1:
                        ties[f"r{r}"] = {"n_tied_at_min": int(m.sum()),
                                         "positives": int(yy[m].sum()),
                                         "negatives": int((1 - yy[m]).sum())}
            boundary[f"{a}/{c}"] = {"n": tot, "hits": hit,
                                    "rate": round(hit / max(tot, 1), 6),
                                    "ties_at_min": ties}
            print(f"      {a}/{c:<14} n={tot:>6}  触边界 {hit:>6}"
                  f"（{hit / max(tot, 1):.2%}）  最低分并列 {ties}")
    ck("真实 OOF 上确有样本触到裁剪边界（h1/saplma_mlp）",
       boundary.get("h1/saplma_mlp", {}).get("hits", 0) > 0,
       f"{boundary.get('h1/saplma_mlp', {}).get('rate')}")
    ck("裁剪在真实数据上制造了跨类并列（h1/saplma_mlp r2）",
       boundary.get("h1/saplma_mlp", {}).get("ties_at_min", {})
              .get("r2", {}).get("positives", 0) > 0,
       str(boundary.get("h1/saplma_mlp", {}).get("ties_at_min", {}).get("r2")))
    rec["clip_boundary_on_saved_oof"] = boundary
    rec["clip_boundary_note"] = (
        "旧 OOF 只保存了融合分数，**没有**裁剪前分数与模型，"
        "因此**不能**由此恢复原始排序，也**不能**断言去掉裁剪一定提高 AUROC。"
        "正负类同处最低分说明并列跨类存在，方向未知。")

    # (c) 合成端到端：只作描述性记录，**不据此断言**
    Xe, ye = synthetic(n=300, d=8, seed=3)
    Xe = Xe + 20.0 * ye[:, None]
    mp = {"hidden_layer_sizes": (64,), "alpha": 1e-8, "max_iter": 3000}
    f_clip, i_clip = lib._fit_score("mlp", mp, Xe, ye, Xe, seed=0,
                                    score_mode="clipped_logodds")
    f_raw, i_raw = lib._fit_score("mlp", mp, Xe, ye, Xe, seed=0,
                                  score_mode="raw_margin")
    print(f"      合成端到端（描述性）：|logit| 最大 "
          f"{max(abs(i_raw['raw_min']), abs(i_raw['raw_max'])):.2f}"
          f"（饱和阈 {sat:.2f}）  触边界 {i_clip['clip_boundary_hits']}/{len(ye)}"
          f" —— 合成数据**训不到饱和**，故此项不作断言")
    rec["extreme_synthetic"] = {
        "clip_hits": i_clip["clip_boundary_hits"],
        "clip_rate": i_clip["clip_boundary_rate"],
        "max_abs_logit": max(abs(i_raw["raw_min"]), abs(i_raw["raw_max"])),
        "saturation_threshold": sat,
        "note": "合成数据未训到饱和，**不作断言**；有牙的检查在 (a) 与 (b)"}

    # ── 4 正类方向 / 无二次反转 / 零值语义 ────────────────────────────────
    print("\n═══ 4 方向与语义")
    for mode in ("clipped_logodds", "raw_margin"):
        for kind, params in (("mlp", {"hidden_layer_sizes": (16,), "alpha": 1e-3,
                                      "max_iter": 200}),
                             ("xgb", {"max_depth": 3, "n_estimators": 50,
                                      "learning_rate": 0.1}),
                             ("logreg_author", {"C": 1.0})):
            s, _i = lib._fit_score(kind, params, X, y, X, seed=0, score_mode=mode)
            a = float(ci.auroc(s, y))
            ck(f"{kind}/{mode}: 正类方向正确（AUROC>0.5，y=1 分更高）",
               a > 0.5 and s[y == 1].mean() > s[y == 0].mean(), f"AUROC={a:.4f}")
    ck("XGBoost 默认 missing=nan（0 是普通取值，不当缺失）",
       str(__import__("xgboost").XGBClassifier().missing) == "nan")

    # ── 5 真实一折：三阶段日志齐全 ────────────────────────────────────────
    print(f"\n═══ 5 真实一折（{args.arm}）：三阶段拟合日志")
    b = drv.load_arm(args.arm, cfg)
    yv, gv, blocks = b["y"], b["groups"], b["blocks"]
    folds = ci.stratified_grouped_folds(yv, gv, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(0),
                                        max_seed_retries=cfg.outer_max_seed_retries)
    train, test = folds[0]
    audit, fit_log = [], []
    conds = [lib.CONDITION_BY_NAME["saplma_mlp"],
             lib.CONDITION_BY_NAME["saplma_mlp_raw"]]
    res = drv.run_fold(y=yv, groups=gv, blocks=blocks, repeat=0, fold=0,
                       train=train, test=test, cfg=cfg, conditions=conds,
                       seed=cfg.base_random_state, audit_sink=audit, fit_log=fit_log)
    stages = {}
    for r in fit_log:
        stages[(r.get("stage"), r.get("phase"))] = stages.get(
            (r.get("stage"), r.get("phase")), 0) + 1
    print("   拟合记录按 (stage, phase) 计数：")
    for k, v in sorted(stages.items(), key=lambda kv: str(kv[0])):
        print(f"      {str(k):<46} {v}")
    ck("候选验证与选中模型最终拟合**分开记录**",
       any(k[0] == "candidate_validation" for k in stages)
       and any(k[0] == "selected_model_fit" for k in stages))
    ck("inner_cf 与 outer_full 两个阶段都有记录",
       any(k[1] == "inner_cf" for k in stages) and any(k[1] == "outer_full" for k in stages))
    need = {"stage", "phase", "condition", "params", "seed", "n_iter", "max_iter_used",
            "converged", "convergence_warnings", "secs", "n_train", "n_test",
            "train_rows_sha", "test_rows_sha", "score_mode"}
    miss = {f for f in need if any(f not in r for r in fit_log)}
    ck("每条拟合记录字段齐全", not miss, f"缺 {sorted(miss)}" if miss else "")
    sel = [r for r in fit_log if r.get("stage") == "selected_model_fit"
           and r.get("phase") == "outer_full"]
    ck("外层最终拟合记录里 n_iter 是**实际迭代**，可与预算上限区分",
       all(r["n_iter"] is not None for r in sel),
       "；".join(f"{r['condition']} n_iter={r['n_iter']}/上限{r['max_iter_used']}"
                 f" 到达上限={r.get('reached_max_iter')}" for r in sel))

    # ── 6 分片三路分数齐全 + 分组隔离 ─────────────────────────────────────
    print("\n═══ 6 三路分数与分组隔离")
    for cname, c in res["conditions"].items():
        ck(f"{cname}: H / H_raw / H_proba 都在",
           c["H"] is not None and c["H_raw"] is not None and c["H_proba"] is not None)
        if lib.CONDITION_BY_NAME[cname].score_mode == "raw_margin":
            ck(f"{cname}: 融合分数 == 原始分数", np.array_equal(c["H"], c["H_raw"]))
        else:
            ck(f"{cname}: 融合分数 == 裁剪概率的 log-odds",
               np.allclose(c["H"], lib._to_log_odds(c["H_proba"]), rtol=0, atol=0))
    print("   真实一折实测裁剪到达率（外层最终拟合）：")
    for r in sel:
        print(f"      {r['condition']:<18} score_mode={r['score_mode']:<16} "
              f"裁剪到达率 {r.get('clip_boundary_rate')}  "
              f"raw 范围 [{r.get('raw_min'):.3f}, {r.get('raw_max'):.3f}]")
    ck("真实一折的拟合记录里带裁剪到达率与原始分数范围",
       all(r.get("clip_boundary_rate") is not None and r.get("raw_min") is not None
           for r in sel))

    gtr = set(map(str, gv[train]))
    gte = set(map(str, gv[test]))
    ck("外层训练/测试**组不相交**（分组隔离）", not (gtr & gte),
       f"训练组 {len(gtr)}  测试组 {len(gte)}  交叠 {len(gtr & gte)}")
    rec["real_fold"] = {
        "arm": args.arm, "n_fit_records": len(fit_log),
        "stages": {str(k): v for k, v in stages.items()},
        "outer_selected": [{k: r[k] for k in
                            ("condition", "score_mode", "params", "n_iter",
                             "max_iter_used", "reached_max_iter", "converged",
                             "clip_boundary_rate", "raw_min", "raw_max", "secs")}
                           for r in sel]}

    # ── 7 **正式 writer 的 round-trip**：写 → 关 → 重读 → 逐值比 ─────────────
    #
    # 上面第 5、6 节只看 run_fold 返回的内存 dict。内存里对、盘上没有，是两回事 ——
    # 旧版正是"run_fold 算了六路分数、cmd_run 只写三路"。所以这里必须调
    # **正式的 write_shard / read_shard**，不能自己另写一份读写。
    print("\n═══ 7 正式分片 writer 的 round-trip（写盘→重读→逐值比对）")
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="p0_roundtrip_"))
    try:
        roster = drv._roster(args.arm, b, cfg)
        man = drv.write_shard(tmp, 0, 0, arm=args.arm, train=train, test=test,
                              res=res, audit=audit, conditions=conds,
                              roster=roster, cfg=cfg)
        back = drv.read_shard(tmp, 0, 0)
        worst, missing = 0.0, []
        for cname, c in res["conditions"].items():
            for k in drv.COND_ARRAYS:
                mem, disk = c.get(k), back["conditions"][cname][k]
                if mem is None:
                    if disk is not None:
                        missing.append(f"{cname}|{k}: 内存 None 盘上有值")
                    continue
                if disk is None:
                    missing.append(f"{cname}|{k}: **盘上没有**")
                    continue
                worst = max(worst, float(np.max(np.abs(
                    np.asarray(mem, dtype=float) - disk))))
        ck("六路分数（S*/S*+H/H/H_raw/H_meta_train/H_meta_train_raw）全部落盘",
           not missing, "；".join(missing[:4]) if missing else
           f"含可空的 H_proba / H_meta_train_proba，共 {len(drv.COND_ARRAYS)} 项")
        ck("回读值与内存**逐元素相同**", worst == 0.0, f"最大绝对差 {worst:.3e}")
        ck("训练侧 inner-OOF 与 train_idx 一起落盘，行顺序可回溯",
           back["train_idx"] is not None
           and np.array_equal(back["train_idx"], np.asarray(train))
           and all(back["conditions"][c]["H_meta_train"].shape == (len(train),)
                   for c in back["conditions"]))
        ck("测试侧行顺序可回溯（test_idx 一致）",
           np.array_equal(back["test_idx"], np.asarray(test)))
        ck("阶段化拟合日志落盘且行数与 manifest 登记一致",
           len(back["fit_log"]) == len(fit_log)
           and all(man["conditions"][c]["n_fit_log_rows"]
                   == sum(1 for x in back["fit_log"] if x.get("condition") == c)
                   for c in man["conditions"]),
           f"{len(back['fit_log'])} 行")
        ck("选参审计与分数写在同一个分片单元里", len(back["audit"]) == len(audit))
        ck("**每个条件各自**记了代码 / 输入 / 划分 / 条件定义 / 内容摘要五类指纹"
           "（不是放在 manifest 顶层共用一份）",
           "code_fingerprint" not in man
           and all(v.get("code_fingerprint") and v.get("input_fingerprint")
                   and v.get("condition_fingerprint") and v.get("array_sha")
                   and v["split"]["train_rows_sha"] == lib._rows_sha(np.asarray(train))
                   for v in man["conditions"].values()),
           f"顶层只留 last_writer；条件 {sorted(man['conditions'])}")
        ck("输入指纹覆盖各块**内容**（不只 id/y/groups）",
           bool((man["conditions"][conds[0].name]["input_fingerprint"]
                 or {}).get("blocks_sha")),
           str(sorted((man["conditions"][conds[0].name]["input_fingerprint"]
                       or {}).get("blocks_sha") or {})))

        def _done(where=tmp):
            s = drv.shard_status(where, 0, 0, conds, arm=args.arm, roster=roster,
                                 cfg=cfg, train=train, test=test)
            return s, {c.name: s["conditions"][c.name][0] for c in conds}

        s0, d0 = _done()
        ck("完整分片 → 续跑判为已完成", all(d0.values()))

        # 拆掉拟合日志：分数还在，但分片**不完整** → 必须重算，不能跳过
        drv._fitlog_path(tmp, 0, 0).unlink()
        _s, d1 = _done()
        ck("**分数写完但日志缺失 → 判未完成**（不会被续跑跳过）",
           not any(d1.values()), _s["conditions"][conds[0].name][1])
        drv.write_shard(tmp, 0, 0, arm=args.arm, train=train, test=test, res=res,
                        audit=audit, conditions=conds, roster=roster, cfg=cfg)

        # 改代码指纹：同名条件但实现变了 → 不得沿用旧分片
        mp = drv._manifest_path(tmp, 0, 0)
        mm = json.loads(mp.read_text(encoding="utf-8"))
        for _v in mm["conditions"].values():
            _v["code_fingerprint"]["lib"] = "0" * 16
        mp.write_text(json.dumps(mm, ensure_ascii=False), encoding="utf-8")
        _s, d2 = _done()
        ck("代码指纹不符 → 判未完成", not any(d2.values()),
           _s["conditions"][conds[0].name][1])

        # ── Codex 的反例：改代码 → 只重算一个条件 → 另一个**又被判成完成** ──
        #
        # 旧布局把 code_fingerprint 放 manifest 顶层、每次写盘刷新，
        # 而未重算的条件条目原样保留 —— 于是第二次写盘把顶层指纹刷新，
        # 上一轮判未完成的旧条件凭空"复活"。指纹改为按条件存后必须不再发生。
        c_a, c_b = conds[0], conds[1]
        real_cfp = drv._code_fingerprint

        def fake_cfp():
            return {"lib": "deadbeefdeadbeef", "driver": "deadbeefdeadbeef"}

        drv._code_fingerprint = fake_cfp                      # 假装代码改了
        try:
            s_before = drv.shard_status(tmp, 0, 0, conds, arm=args.arm, roster=roster,
                                        cfg=cfg, train=train, test=test)
            both_stale = not any(v[0] for v in s_before["conditions"].values())
            # 只重算 c_b（新代码），c_a 不动
            res_b = {"shared": res["shared"],
                     "conditions": {c_b.name: res["conditions"][c_b.name]},
                     "fit_log": [x for x in fit_log if x.get("condition") == c_b.name]}
            drv.write_shard(tmp, 0, 0, arm=args.arm, train=train, test=test,
                            res=res_b, audit=[a for a in audit
                                              if a.get("condition") == c_b.name],
                            conditions=[c_b], roster=roster, cfg=cfg)
            s_after = drv.shard_status(tmp, 0, 0, conds, arm=args.arm, roster=roster,
                                       cfg=cfg, train=train, test=test)
            ck("代码变更后只重算一个条件 → **未重算的旧条件仍判未完成**"
               "（指纹按条件存，不被后写的条目刷新）",
               both_stale and s_after["conditions"][c_b.name][0]
               and not s_after["conditions"][c_a.name][0],
               f"{c_a.name}: {s_after['conditions'][c_a.name][1]}  |  "
               f"{c_b.name}: {'完成' if s_after['conditions'][c_b.name][0] else '未完成'}")
        finally:
            drv._code_fingerprint = real_cfp
        # 恢复到两个条件都在当前代码下的完整状态
        drv.write_shard(tmp, 0, 0, arm=args.arm, train=train, test=test, res=res,
                        audit=audit, conditions=conds, roster=roster, cfg=cfg)
        _s, dok = _done()
        ck("恢复后两个条件都判完成", all(dok.values()))

        # 输入指纹必须覆盖**隐藏向量内容**，不只是 id/y/groups
        roster_t = json.loads(json.dumps(roster, default=str))
        roster_t["blocks_sha"] = {**(roster_t.get("blocks_sha") or {}),
                                  drv.BLOCK_NAME_H: "0" * 16}
        st_t = drv.shard_status(tmp, 0, 0, conds, arm=args.arm, roster=roster_t,
                                cfg=cfg, train=train, test=test)
        ck("隐藏向量内容变了（id/y/groups 不变）→ 判未完成",
           not any(v[0] for v in st_t["conditions"].values()),
           st_t["conditions"][c_a.name][1])

        # 文本块内容变了 → 必须命中。
        # 上一版这里漏了：`_blocks_sha` 读的是不存在的 `.texts` / `.sparse`，
        # 文本块（B3）**什么都没哈希**，改真实文本照样判"完成"。
        tb = next((b for b in blocks if b.kind == "text"), None)
        ck("blocks 里确有文本块可供检验", tb is not None,
           f"{tb.name} n={len(tb.text)}" if tb is not None else "无")
        if tb is not None:
            sha_all = drv._blocks_sha(blocks)
            import dataclasses as _dc
            mutated_blocks = []
            for bb in blocks:
                if bb is tb:
                    txt = list(bb.text)
                    txt[0] = str(txt[0]) + " __tamper__"
                    mutated_blocks.append(_dc.replace(bb, text=txt))   # BlockSpec 是 frozen
                else:
                    mutated_blocks.append(bb)
            sha_mut = drv._blocks_sha(mutated_blocks)
            ck("**改一条真实文本 → 文本块摘要变化**（`.text` 字段名已修对）",
               sha_mut[tb.name] != sha_all[tb.name]
               and all(sha_mut[k] == sha_all[k] for k in sha_all if k != tb.name),
               f"{tb.name}: {sha_all[tb.name]} → {sha_mut[tb.name]}")
            ck("文本块摘要不是空内容哈希（各块摘要互不相同）",
               len(set(sha_all.values())) == len(sha_all), str(sha_all))
            roster_txt = json.loads(json.dumps(roster, default=str))
            roster_txt["blocks_sha"] = sha_mut
            st_x = drv.shard_status(tmp, 0, 0, conds, arm=args.arm, roster=roster_txt,
                                    cfg=cfg, train=train, test=test)
            ck("文本内容变了 → 续跑判未完成",
               not any(v[0] for v in st_x["conditions"].values()),
               st_x["conditions"][c_a.name][1])

        # 盘上数组 / 行索引 / 共享块被就地篡改 → 都必须命中
        def _tamper(mutate, label, who=None):
            with np.load(tmp / "shard_r0_f0.npz", allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            mutate(arrs)
            np.savez_compressed(tmp / "shard_r0_f0.npz", **arrs)
            s_, d_ = _done()
            tgt = who or c_a.name
            ck(label, not d_[tgt], s_["conditions"][tgt][1])
            drv.write_shard(tmp, 0, 0, arm=args.arm, train=train, test=test,
                            res=res, audit=audit, conditions=conds,
                            roster=roster, cfg=cfg)

        def _swap_two(a):
            a[0], a[1] = a[1].copy(), a[0].copy()
            return a

        _tamper(lambda a: a.__setitem__(f"{c_a.name}|H", a[f"{c_a.name}|H"][::-1].copy()),
                "分数数组被换行序（长度与取值集合不变）→ 判未完成")
        _tamper(lambda a: a.__setitem__("test_idx", _swap_two(a["test_idx"].copy())),
                "**test_idx 被调换两行 → 判未完成**（否则预测与样本整体错位）")
        _tamper(lambda a: a.__setitem__("train_idx", _swap_two(a["train_idx"].copy())),
                "**train_idx 被调换两行 → 判未完成**")
        shared_key = f"shared|{sorted(res['shared'])[0]}"
        _tamper(lambda a: a.__setitem__(shared_key, a[shared_key][::-1].copy()),
                f"**共享块 {sorted(res['shared'])[0]} 分数被改 → 判未完成**")

        # 无 manifest 的旧分片 → 标 legacy，不当完成，也不删
        leg = Path(tempfile.mkdtemp(prefix="p0_legacy_"))
        shutil.copy(tmp / "shard_r0_f0.npz", leg / "shard_r0_f0.npz")
        sl = drv.shard_status(leg, 0, 0, conds, arm=args.arm, roster=roster,
                              cfg=cfg, train=train, test=test)
        ck("旧格式分片被标 legacy 且判未完成（原样保留，不伪称补齐）",
           sl["legacy"] and not any(v[0] for v in sl["conditions"].values()),
           sl["conditions"][conds[0].name][1])
        shutil.rmtree(leg, ignore_errors=True)

        ck("正式运行拒绝写 legacy_v1（旧六条件报告不被覆盖）",
           _refuses_legacy(drv, cfg, conds))
        rec["round_trip"] = {
            "n_arrays_per_condition": len(drv.COND_ARRAYS),
            "max_abs_diff": worst, "n_fit_log_rows": len(back["fit_log"]),
            "shard_schema": man["schema"],
            "manifest_conditions": {k: {kk: v[kk] for kk in
                                        ("complete", "arrays", "n_fit_log_rows")}
                                    for k, v in man["conditions"].items()},
            "note": "临时目录，跑完即删；**不写任何正式产物**"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── 8 训练预算 vs 收敛失败：两者不可混 ────────────────────────────────
    print("\n═══ 8 迭代预算阶梯与「预算完成 ≠ 收敛」")
    Xs, ys_ = synthetic(n=200, d=40, seed=7)
    real_ladder = lib.MAX_ITER_LADDER
    try:
        # **仅本检查**把阶梯换成极小值，好在秒级内逼出真实的升档；
        # 正式运行用的仍是事前写死的 (1000, 5000, 20000)。
        lib.MAX_ITER_LADDER = (1, 2, 3)
        _s, i_l1 = lib._fit_score("logreg_author", {"C": 1.0}, Xs, ys_, Xs, seed=0)
    finally:
        lib.MAX_ITER_LADDER = real_ladder
    at = i_l1.get("escalation_attempts") or []
    ck("L1 升档时**每次尝试**都留档（预算 / n_iter / 告警 / 耗时）",
       len(at) >= 2 and all({"max_iter", "n_iter", "converged",
                             "convergence_warnings"} <= set(a) for a in at),
       f"{len(at)} 次尝试，预算 {[a['max_iter'] for a in at]}，"
       f"收敛 {[a['converged'] for a in at]}")
    ck("旧写法只剩最后一次的问题已消除（attempts 首项 = 作者默认值那档）",
       bool(at) and at[0]["is_author_default"] and at[0]["max_iter"] == 1)
    ck("事前阶梯未被正式改动", lib.MAX_ITER_LADDER == (1000, 5000, 20000),
       str(lib.MAX_ITER_LADDER))

    _s, i_xgb = lib._fit_score("xgb", {"n_estimators": 20, "learning_rate": 0.1,
                                       "max_depth": 3}, Xs, ys_, Xs, seed=0)
    ck("XGB 跑满树数记为**预算完成**，converged 留空（未做收敛检验）",
       i_xgb["converged"] is None and i_xgb["budget_completed"] is True
       and i_xgb["convergence_criterion"] == "not_applicable_fixed_budget",
       f"converged={i_xgb['converged']}  budget_completed={i_xgb['budget_completed']}")
    _b, aud = lib.select_params(
        lib.CONDITION_BY_NAME["han_xgboost"],
        lambda tr, te: (Xs[tr], Xs[te]), np.arange(len(ys_)), ys_,
        np.arange(len(ys_)), n_inner=2, fold_seed=0, max_retries=3, seed=0)
    cv0 = aud["curve"][0]
    ck("汇总里 XGB 的 converged_all 为 null，另记 n_no_criterion —— "
       "「没检验」不被读成「通过了」",
       cv0["converged_all"] is None and cv0["n_no_criterion"] == cv0.get("n_fits", 2),
       f"converged_all={cv0['converged_all']}  n_no_criterion={cv0['n_no_criterion']}"
       f"  n_not_converged={cv0['n_not_converged']}")
    rec["budget_vs_convergence"] = {
        "l1_escalation_attempts": at,
        "xgb": {k: i_xgb[k] for k in ("converged", "budget_completed",
                                      "convergence_criterion", "n_iter")},
        "xgb_curve0": {k: cv0[k] for k in ("converged_all", "n_not_converged",
                                           "n_no_criterion")}}

    ok = all(c["ok"] for c in CHECKS)
    rec.update(verdict="PASS" if ok else "FAIL", n_checks=len(CHECKS),
               n_failed=sum(1 for c in CHECKS if not c["ok"]), checks=CHECKS,
               total_secs=round(time.perf_counter() - t0, 1),
               scope="只用合成数据 + 真实产物的一折；**不跑整臂，不改任何已有结果**")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=float)
                        + "\n", encoding="utf-8")
    print(f"\n[{rec['verdict']}] {len(CHECKS) - rec['n_failed']}/{len(CHECKS)} 通过  "
          f"用时 {rec['total_secs']}s")
    print(f"[out] {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
