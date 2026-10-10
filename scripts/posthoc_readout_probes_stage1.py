r"""POST-HOC：Stage 1 Qwen 的 H 侧读出器适配对照（驱动）。

问题：**当前 H 侧读出器及其调参设置，是否限制了隐藏状态能提供的增量？**

⚠️ **非 confirmatory。** 只读冻结产物，全部输出另存到
`outputs/stage1/posthoc_readout_probes/`，不覆盖 `stage1_confirmatory_result.json`。
**全部在本机执行**（不存在控制节点/远程执行路径）。

────────────────────────────────────────────────────────────────────────────
只换 H：输出侧逐位共享

每个 `(repeat, outer_fold)` 用**主路径原函数** `vs.build_base_block_cache` 建一次
cache；B1–B5 的分数由此确定，**在所有 H 条件间共享同一份对象**。每个条件只重算
H 那一列，然后：

    S*   = vs.fit_stack(cache, B1..B5)      ← 与 H 条件无关，逐位相同
    S*+H = vs.fit_stack(cache, B1..B5,H)    ← 按原融合规则重拟合，接入该条件的 H 列

H 的 inner 划分**重算自主路径同一表达式**（确定性），故与 cache 内部那份一致。
meta 的 H 训练列来自 **inner-OOF**，不是训练集内预测。

────────────────────────────────────────────────────────────────────────────
生成任务优先

补充实验**串行**跑、限制 BLAS/OMP 线程；`--timing-only` 先在**真实维度与样本量**上
测单折成本，同时记录**峰值内存**与**对生成速率的影响**。出现明显内存压力时
先停补充实验（`--max-rss-gb` 到阈值即中止），不影响正在跑的生成。

用法：
    python scripts/posthoc_readout_probes_stage1.py --self-check
    python scripts/posthoc_readout_probes_stage1.py --timing-only --arm mcq
    python scripts/posthoc_readout_probes_stage1.py --arm mcq --conditions ref_l2_frozen_grid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 线程必须在 numpy/sklearn 之前限制 —— 生成任务优先，补充实验不抢核。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_readout_probe_lib as lib                               # noqa: E402
from recover_attention import conditional_increment as ci             # noqa: E402
from recover_attention.evaluation import v3_stacking as vs            # noqa: E402
from recover_attention.evaluation.v3_blocks import BlockSpec, canonical_names  # noqa: E402
from recover_attention.evaluation.v3_config import (                  # noqa: E402
    BLOCK_NAME_H, V3AnalysisConfig)

OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes"
PROD = ROOT / "outputs/stage1/production"
CONFIRMATORY = ROOT / "outputs/stage1/confirmatory/stage1_confirmatory_result.json"

#: 复现验收容差 —— **事前写死**。同代码同种子同数据应当完全确定，故取机器精度量级。
REPRO_TOL = 1e-9
#: 补充实验自身的内存闸：进程 RSS 超过它就中止，**先保生成任务**。
DEFAULT_MAX_RSS_GB = 6.0


# ══════════════════════════════════════════════════════════════════════════
def _lower_own_priority() -> str:
    """把**自己**降到 Idle —— 生成任务优先。

    以前靠 `.cmd` 里的 `start /low /b /wait`，但那样 **errorlevel 传不出来**：
    实测步骤 1 被杀之后，cmd 照样往下跑了步骤 2，串行约束当场失效。
    改成直接调 python（errorlevel 正常传播）+ 进程自己降优先级。

    ⚠️ **必须显式声明 restype / argtypes。** `GetCurrentProcess()` 返回的是伪句柄
    `(HANDLE)-1`；不设 `restype` 时 ctypes 按 C int 处理，在 x64 上就变成
    `0xFFFFFFFF` 而不是 `0xFFFFFFFFFFFFFFFF`，`SetPriorityClass` 直接返回 0、
    `GetLastError() == 6`（ERROR_INVALID_HANDLE）—— 实测第一版就是这么静默失败的，
    进程一直停在 BelowNormal，和生成进程同级。
    """
    if os.name != "nt":
        return "n/a"
    try:
        import ctypes
        IDLE_PRIORITY_CLASS = 0x00000040
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.argtypes = []
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.SetPriorityClass.restype = ctypes.c_int
        k.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k.GetPriorityClass.restype = ctypes.c_uint
        h = k.GetCurrentProcess()
        if not k.SetPriorityClass(h, IDLE_PRIORITY_CLASS):
            return f"失败 GetLastError={k.GetLastError()}（保持原优先级）"
        got = k.GetPriorityClass(h)          # **回读确认**，不信返回值一面之词
        return ("Idle" if got == IDLE_PRIORITY_CLASS
                else f"已调用但回读为 0x{got:X}（期望 0x40）")
    except (ImportError, AttributeError, OSError) as exc:
        return f"失败 {type(exc).__name__}（保持原优先级）"


def _rss_gb() -> float:
    import subprocess
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-Process -Id {os.getpid()}).WorkingSet64"],
        capture_output=True, text=True).stdout.strip()
    return round(int(out) / 2**30, 3) if out.isdigit() else float("nan")


def _sys_mem() -> dict:
    """整机内存压力。**光看自己的 RSS 不够** —— 本机 commit 已接近上限，
    真正会打死生成任务的是提交量触顶，不是我们这 0.9G。"""
    import subprocess
    ps = ("$o=Get-CimInstance Win32_OperatingSystem;"
          "$c=(Get-Counter '\\Memory\\Committed Bytes').CounterSamples[0].CookedValue;"
          "$l=(Get-Counter '\\Memory\\Commit Limit').CounterSamples[0].CookedValue;"
          "\"{0} {1} {2}\" -f ($o.FreePhysicalMemory*1024),$c,$l")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True).stdout.split()
    try:
        free, com, lim = (float(x) / 2**30 for x in out[:3])
    except (ValueError, IndexError):
        return {"free_gb": float("nan"), "commit_headroom_gb": float("nan")}
    return {"free_gb": round(free, 2), "committed_gb": round(com, 2),
            "commit_limit_gb": round(lim, 2),
            "commit_headroom_gb": round(lim - com, 2)}


def _mem_gate(min_free: float, min_headroom: float, *, wait_s: int = 1800) -> dict:
    """生成任务优先：整机吃紧就**先等**，等不回来就干净退出（分片已落盘，可续跑）。"""
    t0 = time.time()
    while True:
        m = _sys_mem()
        ok = not (m["free_gb"] < min_free or m["commit_headroom_gb"] < min_headroom)
        if ok or not np.isfinite(m["free_gb"]):
            return m
        if time.time() - t0 > wait_s:
            raise MemoryError(
                f"整机内存持续吃紧 {m} —— 生成任务优先，补充实验退出（分片已落盘，"
                f"重跑同一条命令即续跑）")
        print(f"    [等] 整机吃紧 {m}；min_free={min_free} min_headroom={min_headroom}，"
              f"60s 后重试", flush=True)
        time.sleep(60)


def _gen_rate_probe() -> dict:
    """采一次生成任务的进度，用于**前后对比**补充实验对它的影响。"""
    p = ROOT / "outputs/stage1_llama_v3/production/mcq_raw_unit_results.jsonl"
    n = sum(1 for _ in p.open("rb")) if p.exists() else 0
    return {"t": time.time(), "mcq_rows": n}


def _gen_rate_delta(a: dict, b: dict) -> dict:
    dt, dn = b["t"] - a["t"], b["mcq_rows"] - a["mcq_rows"]
    return {"elapsed_s": round(dt, 1), "units": dn,
            "s_per_unit": round(dt / dn, 2) if dn > 0 else None}


# ══════════════════════════════════════════════════════════════════════════
# 载入 Stage 1 Qwen 两臂（与主路径同一 adapter，不另造口径）
# ══════════════════════════════════════════════════════════════════════════
def load_arm(arm: str, cfg: V3AnalysisConfig) -> dict:
    # 与主路径 v3_analyze 同一组 adapter（H1 用 v3_adapter，MCQ 用 v3_adapter_mcq）
    from recover_attention.evaluation import v3_adapter as ah1
    from recover_attention.evaluation import v3_adapter_mcq as amcq

    recs = [json.loads(x) for x in
            (PROD / f"{arm}_completion_records.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    z = np.load(PROD / f"{arm}_hidden.npz")
    hidden = {k: z[k] for k in z.files}
    if arm == "mcq":
        used = amcq.filter_analyzable(recs)          # MCQ 要先筛可分析
        bundle = amcq.build_mcq_arm(used, hidden)
    else:
        # H1 的 eligible 过滤在主路径里由 feature_cache 做（`eligible`，回退
        # `eligible_for_primary`），adapter 本身不筛。此处按**同一条件**筛，
        # 否则会把 600 条 label=null 的记录一起喂进去（实测就是这么炸的）。
        used = [r for r in recs
                if bool(r.get("eligible", r.get("eligible_for_primary")))]
        bundle = ah1.build_arm(used, hidden)
    # **样本标识**：复现核对要核对到 id 级，不能只比样本数
    idf = bundle.get("id_field", "trace_id")
    bundle["sample_ids"] = np.asarray([str(r[idf]) for r in used])
    bundle["id_field_used"] = idf
    bundle["n_raw_records"] = len(recs)
    return bundle


# ══════════════════════════════════════════════════════════════════════════
# 一个 (repeat, outer_fold)：共享 O 侧，逐条件只换 H
# ══════════════════════════════════════════════════════════════════════════
def run_fold(*, y, groups, blocks, repeat: int, fold: int, train, test,
             cfg: V3AnalysisConfig, conditions, seed: int,
             audit_sink: list, fit_log: list | None = None) -> dict:
    """返回 {condition: {"S*":…, "S*+H":…, "H":…}}（均为该 fold 的 test 分数）。"""
    names_s, names_sh = canonical_names()
    fit_log = [] if fit_log is None else fit_log
    h_block = next(b for b in blocks if b.name == BLOCK_NAME_H)

    # ① 主路径原函数建 cache —— B1..B5 由此确定，所有条件共享
    cache = vs.build_base_block_cache(y=y, groups=groups, blocks=blocks,
                                      repeat=repeat, outer_fold=fold,
                                      train=train, test=test, cfg=cfg)
    s_star = vs.fit_stack(cache, names_s, y, cfg)          # 与 H 条件无关

    # ② H 的 inner 划分：重算自主路径同一表达式（确定性 → 与 cache 内部一致）
    inner = ci.stratified_grouped_folds(
        y[train], groups[train], n_splits=cfg.n_inner_folds,
        seed=cfg.fold_seed(repeat), max_seed_retries=cfg.inner_max_seed_retries)

    def design(tr, te):
        """H 块的 fold-local 设计矩阵：复用冻结的 z-score + /√d（口径 zscore_div_sqrt_d）。

        ⚠️ `build_fold_design` 返回的是 **csr_matrix**（为 text 块统一接口）。
        主路径把它直接喂 LogisticRegression（接受稀疏），但 MLP / XGBoost 需要稠密，
        所以这里显式 `.toarray()` —— **只改容器，不改数值**。
        不能用 `np.asarray(稀疏)`：那会得到 0 维 object 数组，
        报成 "setting an array element with a sequence"（实测撞到过）。
        """
        X_tr, X_te, _ = ci.build_fold_design(
            tr, te, dense_blocks=[(h_block.name, h_block.dense)])
        return (np.ascontiguousarray(X_tr.toarray(), dtype=np.float64),
                np.ascontiguousarray(X_te.toarray(), dtype=np.float64))

    def design_raw(tr, te):
        """**作者口径**：原始激活，不做任何变换。

        不含任何用到测试折的统计量，故同样是折内安全的 —— 它根本没有"拟合"这一步。
        ⚠️ 相同的 C 在 raw 与 zscore_div_sqrt_d 两种口径下**不是**相同的有效正则强度，
        两者必须**分条件**报告，不能互相解释。
        """
        D = np.asarray(h_block.dense, dtype=np.float64)
        return (np.ascontiguousarray(D[tr]), np.ascontiguousarray(D[te]))

    DESIGNS = {"zscore_div_sqrt_d": design, "raw_activation": design_raw}

    # B1..B5 由共享 cache 决定，与 H 条件无关 —— 单独取出以便逐块核对
    shared = {k: np.asarray(v, dtype=float).copy()
              for k, v in cache.test_scores.items() if k != BLOCK_NAME_H}

    out = {}
    for cond in conditions:
        t0 = time.perf_counter()
        design_fn = DESIGNS[cond.preprocess]      # 输入口径按条件选，不混用
        base_ctx = {"repeat": repeat, "fold": fold, "condition": cond.name,
                    "score_mode": cond.score_mode, "preprocess": cond.preprocess,
                    "author_fixed": cond.author_fixed}
        # meta 的 H 训练列 = inner-OOF（**不是**训练集内预测）
        meta_tr = np.full(train.size, np.nan)
        meta_tr_raw = np.full(train.size, np.nan)
        meta_tr_proba = np.full(train.size, np.nan)
        for i, (itr, ite) in enumerate(inner):
            gtr, gte = train[itr], train[ite]
            params, audit = lib.select_params(
                cond, design_fn, gtr, y, groups, n_inner=cfg.n_inner_folds,
                fold_seed=cfg.fold_seed(repeat),
                max_retries=cfg.inner_max_seed_retries, seed=seed,
                main_lr_kwargs=cfg.base_lr_kwargs,
                ctx={**base_ctx, "phase": "inner_cf", "inner_fold": i},
                fit_log=fit_log)
            audit_sink.append({"condition": cond.name, "repeat": repeat, "fold": fold,
                               "phase": "inner_cf", "inner_fold": i, **audit})
            X_tr, X_te = design_fn(gtr, gte)
            # ⚠️ **不要再用 `_` 丢掉 info**：选中模型的最终拟合记录（实际迭代次数、
            # 收敛告警、裁剪到达率、原始分数范围）只有这里能拿到。
            sc, info = lib._fit_score(
                cond.kind, params, X_tr, y[gtr], X_te, seed=seed,
                main_lr_kwargs=cfg.base_lr_kwargs, score_mode=cond.score_mode,
                ctx={**base_ctx, "phase": "inner_cf", "inner_fold": i,
                     "stage": "selected_model_fit",
                     "train_rows_sha": lib._rows_sha(gtr),
                     "test_rows_sha": lib._rows_sha(gte)})
            meta_tr[ite] = sc
            meta_tr_raw[ite] = info["raw"]
            if info["proba"] is not None:
                meta_tr_proba[ite] = info["proba"]
            fit_log.append({k: v for k, v in info.items() if k not in ("raw", "proba")})
        if not np.all(np.isfinite(meta_tr)):
            raise RuntimeError(f"{cond.name}: inner-OOF 未覆盖全部训练行")

        params, audit = lib.select_params(
            cond, design_fn, train, y, groups, n_inner=cfg.n_inner_folds,
            fold_seed=cfg.fold_seed(repeat),
            max_retries=cfg.inner_max_seed_retries, seed=seed,
            main_lr_kwargs=cfg.base_lr_kwargs,
            ctx={**base_ctx, "phase": "outer_full", "inner_fold": None},
            fit_log=fit_log)
        audit_sink.append({"condition": cond.name, "repeat": repeat, "fold": fold,
                           "phase": "outer_full", "inner_fold": None, **audit})
        X_tr, X_te = design_fn(train, test)
        h_test, outer_info = lib._fit_score(
            cond.kind, params, X_tr, y[train], X_te, seed=seed,
            main_lr_kwargs=cfg.base_lr_kwargs, score_mode=cond.score_mode,
            ctx={**base_ctx, "phase": "outer_full", "inner_fold": None,
                 "stage": "selected_model_fit",
                 "train_rows_sha": lib._rows_sha(train),
                 "test_rows_sha": lib._rows_sha(test)})
        fit_log.append({k: v for k, v in outer_info.items()
                        if k not in ("raw", "proba")})

        # ③ 把 H 列换进**同一个** cache，再按原融合规则重拟合 S*+H
        cache.meta_train_scores[BLOCK_NAME_H] = meta_tr
        cache.test_scores[BLOCK_NAME_H] = h_test
        out[cond.name] = {
            "S*": s_star, "S*+H": vs.fit_stack(cache, names_sh, y, cfg),
            "H": h_test,                       # 实际喂给融合层的分数
            "H_raw": outer_info["raw"],        # 模型原始分数（外层最终）
            "H_proba": outer_info["proba"],    # 预测概率（logreg 为 None）
            "H_meta_train": meta_tr,           # inner-OOF：融合用
            "H_meta_train_raw": meta_tr_raw,   # inner-OOF：原始
            "H_meta_train_proba": (None if not np.any(np.isfinite(meta_tr_proba))
                                   else meta_tr_proba),
            "outer_fit": {k: v for k, v in outer_info.items()
                          if k not in ("raw", "proba")},
            "secs": round(time.perf_counter() - t0, 1)}
    return {"shared": shared, "conditions": out, "fit_log": fit_log}


# ══════════════════════════════════════════════════════════════════════════
def cmd_self_check(cfg: V3AnalysisConfig) -> int:
    """不加载真实产物的自检：条件表、log-odds 换算、注入点签名。"""
    ok = []

    def ck(label, cond_ok, detail=""):
        ok.append(cond_ok)
        print(f"  [{'ok ' if cond_ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    ck(f"条件表 {len(lib.CONDITIONS)} 个（事前 {len(lib.PRESPECIFIED)} + "
       f"事后敏感性 {len(lib.POSTHOC_SENSITIVITY)}）且名字唯一",
       len({c.name for c in lib.CONDITIONS}) == len(lib.CONDITIONS)
       and len(lib.PRESPECIFIED) == 5)
    ck("事前 / 事后两组不重叠且并集等于条件表",
       not (set(lib.PRESPECIFIED) & set(lib.POSTHOC_SENSITIVITY))
       and set(lib.PRESPECIFIED) | set(lib.POSTHOC_SENSITIVITY)
       == {c.name for c in lib.CONDITIONS})
    ck("han_l1_wide 含 han_l1 的**全部** C，且被标为事后敏感性",
       {x["C"] for x in lib.CONDITION_BY_NAME["han_l1"].candidates}
       <= {x["C"] for x in lib.CONDITION_BY_NAME["han_l1_wide"].candidates}
       and "han_l1_wide" in lib.POSTHOC_SENSITIVITY
       and "事前规则" in lib.CONDITION_BY_NAME["han_l1_wide"].notes)
    ck("每个条件候选集非空且有限",
       all(0 < len(c.candidates) < 50 for c in lib.CONDITIONS))
    ck("原文参照点已列入候选（han_l1 C=0.5）",
       any(abs(c["C"] - 0.5) < 1e-12 for c in lib.CONDITION_BY_NAME["han_l1"].candidates))
    ck("原文参照点已列入候选（han_xgboost 1000/0.1/6）",
       any(c["n_estimators"] == 1000 and c["max_depth"] == 6
           for c in lib.CONDITION_BY_NAME["han_xgboost"].candidates))
    ck("原文参照点已列入候选（saplma 5 epochs）",
       any(c["max_iter"] == 5 for c in lib.CONDITION_BY_NAME["saplma_mlp"].candidates))
    ck("放宽网格**包含**原网格",
       {c["C"] for c in lib.CONDITION_BY_NAME["ref_l2_frozen_grid"].candidates}
       <= {c["C"] for c in lib.CONDITION_BY_NAME["ref_l2_wide_grid"].candidates})

    p = np.array([0.0, 0.5, 1.0])
    lo = lib._to_log_odds(p)
    ck("log-odds 换算有限且单调且 p=0.5→0", np.all(np.isfinite(lo))
       and lo[0] < lo[1] < lo[2] and abs(lo[1]) < 1e-12, str(np.round(lo, 3)))

    import inspect
    sig = inspect.signature(vs.run_repeat)
    ck("主路径确有 cache_builder 注入点", "cache_builder" in sig.parameters)
    ck("本脚本不写冻结产物", CONFIRMATORY.exists() and OUT_DIR != CONFIRMATORY.parent)
    print(f"\n[self-check] {sum(ok)}/{len(ok)} 通过")
    return 0 if all(ok) else 2


def cmd_timing(arm: str, cfg: V3AnalysisConfig, conditions, max_rss: float) -> int:
    """真实维度/样本量上的**单折**成本：耗时 + 峰值内存 + 对生成速率的影响。"""
    print(f"[timing] 载入 {arm} …")
    g0 = _gen_rate_probe()
    rss0 = _rss_gb()
    bundle = load_arm(arm, cfg)
    y, groups, blocks = bundle["y"], bundle["groups"], bundle["blocks"]
    h = next(b for b in blocks if b.name == BLOCK_NAME_H)
    print(f"  n={y.size}  正例={int(np.sum(y))}  组={len(set(map(str, groups)))}  "
          f"H dim={h.dense.shape[1]}  载入后 RSS={_rss_gb()}G（载入前 {rss0}G）")

    folds = ci.stratified_grouped_folds(y, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(0),
                                        max_seed_retries=cfg.outer_max_seed_retries)
    train, test = folds[0]
    peak, audit = _rss_gb(), []
    t0 = time.perf_counter()
    res = run_fold(y=y, groups=groups, blocks=blocks, repeat=0, fold=0,
             train=train, test=test, cfg=cfg, conditions=conditions,
             seed=0, audit_sink=audit)
    secs = time.perf_counter() - t0
    peak = max(peak, _rss_gb())
    g1 = _gen_rate_probe()
    per_cond = {k: v["secs"] for k, v in res["conditions"].items()}

    n_folds = cfg.n_repeats * cfg.n_outer_folds
    acct = {c.name: fit_accounting(c, cfg) for c in conditions}
    print(f"\n  单折（含 O 侧 cache + {len(conditions)} 个条件）{secs:.1f}s")
    for c in conditions:
        a, t = acct[c.name], per_cond[c.name]
        print(f"    {c.name:<20} {t:>7.1f}s   候选 {a['n_candidates']}  "
              f"折内拟合 {a['fits_per_outer_fold']}（inner_cf {a['fits_inner_cf_stage']}"
              f" + outer_full {a['fits_outer_full_stage']}）  "
              f"整臂 {a['fits_per_arm']} 次 ≈ {t * n_folds / 3600:.2f} h")
    print(f"  外推整臂 {n_folds} 折 ≈ {secs * n_folds / 3600:.2f} h（含共享 O 侧 cache）")
    print(f"  峰值 RSS {peak}G（闸 {max_rss}G）")
    print(f"  期间生成任务：{_gen_rate_delta(g0, g1)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"timing_{arm}.json").write_text(json.dumps({
        "arm": arm, "n": int(y.size), "h_dim": int(h.dense.shape[1]),
        "conditions": [c.name for c in conditions],
        "per_condition_secs_one_fold": per_cond,
        "per_condition_projected_arm_hours": {
            k: round(v * n_folds / 3600, 3) for k, v in per_cond.items()},
        "fit_accounting": acct,
        "one_fold_secs": round(secs, 1),
        "projected_arm_hours": round(secs * n_folds / 3600, 2),
        "peak_rss_gb": peak, "max_rss_gb_gate": max_rss,
        "generation_during_timing": _gen_rate_delta(g0, g1),
        "budget_hours_per_arm": lib.BUDGET_HOURS_PER_ARM,
        "note": "计时只用于定预算，**不用于挑选表现更好的条件**",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {OUT_DIR / f'timing_{arm}.json'}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 嵌套阶段的**拟合次数**：计时报告要写清"覆盖了多少候选、哪些嵌套阶段"
# ══════════════════════════════════════════════════════════════════════════
def fit_accounting(cond, cfg: V3AnalysisConfig) -> dict:
    """一个 (repeat, outer_fold) 内，该条件要做多少次 readout 拟合。

        inner-CF 阶段  n_inner 个 OOF 折 × [选参 K×n_inner 次 + 定参 1 次]
        outer-full 阶段            选参 K×n_inner 次 + 定参 1 次
    """
    K, J = len(cond.candidates), cfg.n_inner_folds
    per_select = K * J
    inner_cf = J * (per_select + 1)
    outer_full = per_select + 1
    return {"n_candidates": K, "n_inner_folds": J,
            "fits_inner_cf_stage": inner_cf, "fits_outer_full_stage": outer_full,
            "fits_per_outer_fold": inner_cf + outer_full,
            "fits_per_arm": (inner_cf + outer_full) * cfg.n_repeats * cfg.n_outer_folds,
            "stages_covered": ["inner_cf(选参+定参)", "outer_full(选参+定参)"]}


def _sha(a) -> str:
    import hashlib
    b = np.ascontiguousarray(a)
    return hashlib.sha256(b.tobytes() if b.dtype.kind not in "US"
                          else "\x1f".join(b.tolist()).encode()).hexdigest()[:16]


def _roster(arm: str, bundle: dict, cfg: V3AnalysisConfig) -> dict:
    """样本标识 / 标签 / 分组 / 划分的指纹 —— 复现核对比的是这些，不只是样本数。"""
    y, groups, ids = bundle["y"], bundle["groups"], bundle["sample_ids"]
    splits = {}
    for r in range(cfg.n_repeats):
        folds = ci.stratified_grouped_folds(
            y, groups, n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r),
            max_seed_retries=cfg.outer_max_seed_retries)
        fo = np.full(y.size, -1, dtype=np.int64)
        for f, (_, te) in enumerate(folds):
            fo[te] = f
        splits[f"repeat{r}"] = {
            "fold_seed": cfg.fold_seed(r),
            "test_sizes": [int(np.size(te)) for _, te in folds],
            "test_positives": [int(y[te].sum()) for _, te in folds],
            "fold_of_sha": _sha(fo),      # 任何一行换折都会变
        }
    return {"arm": arm, "n": int(y.size), "n_raw_records": bundle["n_raw_records"],
            "n_positive": int(y.sum()), "n_groups": int(len(set(map(str, groups)))),
            "id_field": bundle["id_field_used"],
            "ids_sha": _sha(ids), "ids_first": str(ids[0]), "ids_last": str(ids[-1]),
            "y_sha": _sha(y.astype(np.int64)),
            "groups_sha": _sha(np.asarray(groups).astype(str)),
            "h_dim": int(next(b for b in bundle["blocks"]
                              if b.name == BLOCK_NAME_H).dense.shape[1]),
            # ⚠️ 只记 id / y / groups 的指纹**不够**：隐藏向量或文本块被换掉，
            #    而样本名单、标签、分组不变，上面三个 sha 一个都不会动。
            #    所以把每个块的**内容**也算进来 —— 续跑的输入一致性靠它。
            "blocks_sha": _blocks_sha(bundle["blocks"]),
            "outer_splits": splits}


def _blocks_sha(blocks) -> dict:
    """逐块的内容指纹。

    ⚠️ 上一版这里写了 `getattr(b, "texts", None)` 和 `getattr(b, "sparse", None)`，
       而 `BlockSpec` 的字段其实叫 **`text`**，也根本没有 `sparse`
       （稀疏化是 v3_stacking 在训练折内做的，不在 BlockSpec 上）。
       于是文本块（B3）**什么都没哈希**，摘要是空 hash —— 改真实文本也查不出来。
       现在按 `kind` 分派，并且**哈希不到内容就报错**，不允许静默产出空摘要。
    """
    import hashlib
    out = {}
    for b in blocks:
        h = hashlib.sha256()
        h.update(f"{b.name}\x00{b.kind}\x00".encode())
        if b.kind == "dense":
            a = np.ascontiguousarray(np.asarray(b.dense))
            h.update(f"{a.shape}\x00{a.dtype}\x00".encode())
            h.update(a.tobytes())
            n_items = int(a.size)
        elif b.kind == "text":
            t = b.text
            if t is None:
                raise RuntimeError(f"{b.name}: kind=text 但 .text 为 None —— 停")
            h.update(f"len={len(t)}\x00".encode())
            # 用不可能出现在文本里的分隔符，避免"拼接歧义"（ab|c 与 a|bc 同摘要）
            for s in t:
                sb = str(s).encode("utf-8")
                h.update(len(sb).to_bytes(8, "little"))
                h.update(sb)
            n_items = len(t)
        else:
            raise RuntimeError(f"{b.name}: 未知 kind={b.kind!r} —— 停，不静默跳过")
        if n_items == 0:
            raise RuntimeError(f"{b.name}: 内容为空，拒绝产出空摘要")
        out[b.name] = h.hexdigest()[:16]
    if not out:
        raise RuntimeError("blocks 为空 —— 输入指纹不可能有效")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 复现核对：两层
#   T1  主路径原封不动（paths.run_all_repeats_for_path, Path A）→ 比冻结 JSON
#   T2  本补充实验的 harness + ref_l2_frozen_grid            → 比 T1
# T2 通过才说明"只换 H"这件事本身没引入差异；否则新探针的对比没有意义。
# ══════════════════════════════════════════════════════════════════════════
def _fold_of(y, groups, cfg, r):
    folds = ci.stratified_grouped_folds(y, groups, n_splits=cfg.n_outer_folds,
                                        seed=cfg.fold_seed(r),
                                        max_seed_retries=cfg.outer_max_seed_retries)
    v = np.full(y.size, -1, dtype=int)
    for f, (_, te) in enumerate(folds):
        v[te] = f
    return folds, v


def cmd_reproduce(arm: str, cfg: V3AnalysisConfig, max_rss: float) -> int:
    from recover_attention.evaluation import v3_analysis_paths as paths
    from recover_attention.evaluation import v3_repeat_estimand as vre

    print(f"[repro] 载入 {arm} …", flush=True)
    g0, t_all = _gen_rate_probe(), time.perf_counter()
    bundle = load_arm(arm, cfg)
    y, groups, blocks = bundle["y"], bundle["groups"], bundle["blocks"]
    roster = _roster(arm, bundle, cfg)
    print(f"  n={roster['n']} 正例={roster['n_positive']} 组={roster['n_groups']} "
          f"Hdim={roster['h_dim']}  RSS={_rss_gb()}G", flush=True)

    frozen = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))
    fz = frozen["arms"][paths.PATH_A][arm]
    names_s, names_sh = paths.names_for_path(paths.PATH_A)

    # ── T1：主路径原封不动 ────────────────────────────────────────────────
    print(f"[repro] T1 主路径 R={cfg.n_repeats} × {cfg.n_outer_folds} 折 …", flush=True)
    t0 = time.perf_counter()
    reps = paths.run_all_repeats_for_path(y=y, groups=groups, blocks=blocks,
                                          cfg=cfg, path=paths.PATH_A)
    t1_secs = time.perf_counter() - t0
    pt = vre.point_estimates(y, reps, names_sh)
    t1_cmp = {}
    for k, v in fz["auroc"].items():
        got = float(pt["auroc_S*"] if k == "S*" else
                    pt["auroc_S*+H"] if k == "S*+H" else pt["auroc_block"][k])
        t1_cmp[k] = {"frozen": v, "reproduced": got, "abs_diff": abs(got - v),
                     "ok": abs(got - v) <= REPRO_TOL}
    d = float(pt[vre.DELTA_H])
    t1_cmp["delta_H"] = {"frozen": fz["delta_h_point"], "reproduced": d,
                         "abs_diff": abs(d - fz["delta_h_point"]),
                         "ok": abs(d - fz["delta_h_point"]) <= REPRO_TOL}
    t1_ok = all(v["ok"] for v in t1_cmp.values())
    for k, v in t1_cmp.items():
        print(f"    {'ok ' if v['ok'] else 'FAIL'} {k:<8} 冻结 {v['frozen']:.16f} "
              f"本次 {v['reproduced']:.16f}  |d|={v['abs_diff']:.3e}", flush=True)
    per_repeat_t1 = [{"repeat": i, "auroc_S*": float(st["auroc_S*"]),
                      "auroc_H": float(st["auroc_block"]["H"]),
                      "auroc_S*+H": float(st["auroc_S*+H"]),
                      "delta_H": float(st[vre.DELTA_H])}
                     for i, st in enumerate(pt["per_repeat"])]
    print("  逐 repeat（冻结产物只存跨 repeat 平均值，故此列为**本次新产出**，无冻结对照）:")
    for rr in per_repeat_t1:
        print(f"    r{rr['repeat']}  S*={rr['auroc_S*']:.6f}  H={rr['auroc_H']:.6f}  "
              f"S*+H={rr['auroc_S*+H']:.6f}  d_H={rr['delta_H']:+.6f}", flush=True)

    # ── T2：本 harness + ref_l2_frozen_grid ──────────────────────────────
    cond = lib.CONDITION_BY_NAME["ref_l2_frozen_grid"]
    print(f"[repro] T2 本 harness（{cond.name}）…", flush=True)
    t0, audit, peak = time.perf_counter(), [], _rss_gb()
    n = y.size
    oof2 = [{k: np.full(n, np.nan) for k in ("S*", "S*+H", "H")}
            for _ in range(cfg.n_repeats)]
    shared2 = [{k: np.full(n, np.nan) for k in names_s} for _ in range(cfg.n_repeats)]
    folds_of = []
    for r in range(cfg.n_repeats):
        folds, fo = _fold_of(y, groups, cfg, r)
        folds_of.append(fo)
        for f, (train, test) in enumerate(folds):
            res = run_fold(y=y, groups=groups, blocks=blocks, repeat=r, fold=f,
                           train=train, test=test, cfg=cfg, conditions=[cond],
                           seed=cfg.base_random_state, audit_sink=audit)
            c = res["conditions"][cond.name]
            for k in ("S*", "S*+H", "H"):
                oof2[r][k][test] = c[k]
            for k, v in res["shared"].items():
                shared2[r][k][test] = v
            peak = max(peak, _rss_gb())
            if peak > max_rss:
                raise MemoryError(f"RSS {peak}G 超过闸 {max_rss}G —— 先保生成任务，已中止")
            print(f"    r{r} f{f}  {c['secs']}s  RSS={_rss_gb()}G", flush=True)
    t2_secs = time.perf_counter() - t0

    diffs = {}
    for k in ("S*", "S*+H", "H"):
        diffs[k] = max(float(np.max(np.abs(oof2[r][k] - reps[r]["oof"][k])))
                       for r in range(cfg.n_repeats))
    for k in names_s:
        diffs[k] = max(float(np.max(np.abs(shared2[r][k] - reps[r]["oof"][k])))
                       for r in range(cfg.n_repeats))
    t2_ok = all(v <= REPRO_TOL for v in diffs.values())
    print("  T2 vs T1 逐元素最大绝对差：")
    for k, v in diffs.items():
        print(f"    {'ok ' if v <= REPRO_TOL else 'FAIL'} {k:<6} {v:.3e}", flush=True)

    chosen, n_bound = {}, 0
    for a in audit:
        if a.get("fell_back"):
            chosen["fell_back"] = chosen.get("fell_back", 0) + 1
            continue
        key = str(a["chosen"]["C"])
        chosen[key] = chosen.get(key, 0) + 1
        n_bound += int(bool(a.get("at_boundary")))
    print(f"  原网格选参分布 {chosen}；落边界 {n_bound}/{len(audit)}", flush=True)

    g1 = _gen_rate_probe()
    out = {
        "arm": arm, "schema": "posthoc_reference_reproduction_v1",
        "verdict": "PASS" if (t1_ok and t2_ok) else "FAIL",
        "tolerance": REPRO_TOL, "roster": roster,
        "tier1_main_path_vs_frozen": {"ok": t1_ok, "secs": round(t1_secs, 1),
                                      "comparisons": t1_cmp,
                                      "per_repeat_new": per_repeat_t1},
        "tier2_harness_vs_tier1": {"ok": t2_ok, "secs": round(t2_secs, 1),
                                   "max_abs_diff": diffs, "condition": cond.name},
        "tuning_audit_frozen_grid": {"chosen_C_counts": chosen, "at_boundary": n_bound,
                                     "n_selections": len(audit),
                                     "selection_metric": lib.SELECTION_METRIC,
                                     "tie_break": lib.TIE_BREAK},
        "fit_accounting": fit_accounting(cond, cfg),
        "peak_rss_gb": peak, "max_rss_gb_gate": max_rss,
        "generation_during": _gen_rate_delta(g0, g1),
        "total_secs": round(time.perf_counter() - t_all, 1),
        "note": ("冻结产物只存跨 repeat 平均值，未存逐 repeat 值；"
                 "故逐 repeat 指标是本次新产出，无冻结对照可比。"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"reference_reproduction_{arm}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n",
                 encoding="utf-8")
    (OUT_DIR / f"tuning_audit_{arm}_{cond.name}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, default=float) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        OUT_DIR / f"oof_{arm}_{cond.name}.npz",
        sample_id=bundle["sample_ids"], group_id=np.asarray(groups).astype(str), y=y,
        **{f"r{r}_fold": folds_of[r] for r in range(cfg.n_repeats)},
        **{f"r{r}_{k}": oof2[r][k] for r in range(cfg.n_repeats)
           for k in ("S*", "S*+H", "H")})
    print(f"\n[{out['verdict']}] 用时 {out['total_secs']}s  峰值 RSS {peak}G  "
          f"期间生成 {out['generation_during']}")
    print(f"[out] {f}")
    return 0 if out["verdict"] == "PASS" else 3


# ══════════════════════════════════════════════════════════════════════════
# 正式运行：R×折 分片落盘（可断点续跑）→ 汇总 → 配对区间
# ══════════════════════════════════════════════════════════════════════════
def _acquire_lock() -> Path:
    """单实例锁。加它的原因：任务加了"登录自启"触发器之后，手动 `schtasks /run`
    与自启有可能撞在一起；两个进程同时算同一折会**同时写同一个分片**。
    锁里存 pid，重启后 pid 可能被复用，所以还要核对是不是 python 进程。"""
    import subprocess
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = OUT_DIR / "probes.lock"
    if lock.exists():
        try:
            old = int(lock.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError):
            old = -1
        if old > 0 and old != os.getpid():
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True).stdout
            # 精确比对 CSV 的 PID 字段，不做子串匹配 —— 内存列里的数字会误判
            fields = [c.strip('" ') for c in out.split(",")]
            alive = len(fields) > 1 and fields[1] == str(old) and "python" in fields[0].lower()
            if alive:
                raise SystemExit(f"[stop] 已有探针进程在跑（pid={old}）—— 不启动重复进程。")
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    encoding="utf-8")
    return lock


def _merge_audit(path: Path, new_rows: list, recomputed: set[str],
                 provenance: str | None = None) -> list:
    """审计**按条件合并**写回，不整片覆盖。

    ⚠️ 这是本轮修掉的一个真 bug：原来每折直接 `write_text(audit)`，而续跑时
    `audit` 只含本次补算的条件 —— MCQ 第二轮补 MLP/XGB 时，把第一轮三个便宜条件的
    选参记录**整个覆盖掉了**（实测 audit_r*_f*.json 里只剩 2 个条件）。
    分片做了按条件合并，审计漏了同样的处理。
    """
    old = []
    if path.exists():
        old = [r for r in json.loads(path.read_text(encoding="utf-8"))
               if r.get("condition") not in recomputed]
    if provenance:
        new_rows = [{**r, "provenance": provenance} for r in new_rows]
    merged = old + new_rows
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1, default=float)
                    + "\n", encoding="utf-8")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# 分片落盘 v2 —— 把 run_fold 算出来的东西**全部**落到盘上
#
# 旧版（v1）每折只存 S* / S*+H / H 三列：run_fold 返回的原始分数、预测概率、
# 训练侧 inner-OOF 分数、外层最终拟合记录和 fit_log **全部在内存里丢掉**。
# 结果是新增的"原始分数"通道算了等于没算 —— 盘上根本没有它。
#
# v2 的三条硬规则：
#   ① 分片是**一个整体**：分数 + 选参审计 + 拟合日志三样齐了才算完成。
#      manifest 最后写，它就是**提交点**；先写分数后崩溃 → 下次视为未完成、重算。
#   ② 续跑**不看** `condition|H` 在不在，看 manifest 里该条件的 complete 标记
#      以及 schema / 代码 / 输入 / 划分 / 条件五个指纹是否全部对上。
#   ③ 旧分片**原样保留**，标 legacy，不假装它补齐了 —— v2 写进独立的版本目录。
# ══════════════════════════════════════════════════════════════════════════
SHARD_SCHEMA = "posthoc_probe_shard_v2"
LEGACY_SHARD_SCHEMA = "posthoc_probe_shard_v1_legacy"
DEFAULT_OUTPUT_VERSION = "v2_fullscore"
LEGACY_OUTPUT_VERSION = "legacy_v1"

#: 测试折上每个条件必须落盘的数组
TEST_ARRAYS_REQUIRED = ("S*", "S*+H", "H", "H_raw")
#: 允许为 None（logreg 不产出概率）—— manifest 里登记为 "null"，不是"漏了"
TEST_ARRAYS_NULLABLE = ("H_proba",)
#: 训练侧 inner-OOF 分数：**融合层的实际输入**，旧格式整块没有
TRAIN_ARRAYS_REQUIRED = ("H_meta_train", "H_meta_train_raw")
TRAIN_ARRAYS_NULLABLE = ("H_meta_train_proba",)
COND_ARRAYS: tuple[str, ...] = (TEST_ARRAYS_REQUIRED + TEST_ARRAYS_NULLABLE
                                + TRAIN_ARRAYS_REQUIRED + TRAIN_ARRAYS_NULLABLE)
_TRAIN_SIDE = set(TRAIN_ARRAYS_REQUIRED + TRAIN_ARRAYS_NULLABLE)


def _text_sha(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _code_fingerprint() -> dict:
    """读出器与驱动的源码指纹。改了评分口径而指纹没变，是不可能的。"""
    import hashlib

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    return {"lib": h(Path(lib.__file__)), "driver": h(Path(__file__))}


def _condition_fingerprint(cond) -> str:
    """条件定义的指纹：候选集、评分口径、预处理口径、是否固定参数全进去。

    只比名字是不够的 —— 同名条件的候选集被改过，续跑会把新旧结果拼在一起。
    """
    return _text_sha(json.dumps(
        {"name": cond.name, "kind": cond.kind,
         "candidates": lib._jsonable(cond.candidates),
         "score_mode": cond.score_mode, "preprocess": cond.preprocess,
         "author_fixed": bool(cond.author_fixed),
         "reduction_order": list(cond.reduction_order)},
        sort_keys=True, default=str))


def _array_sha(a) -> str:
    """数组的**内容**摘要：形状 + dtype + 字节。行序被换、数值被改都会变。"""
    import hashlib
    x = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.tobytes())
    return h.hexdigest()[:16]


def _input_fingerprint(roster: dict) -> dict:
    """输入指纹：样本名单 / 标签 / 分组 **加上各块的内容摘要**。

    只记前三项是不够的 —— 隐藏向量或文本块被换掉时它们一个都不会动。
    """
    fp = {k: roster[k] for k in ("n", "n_positive", "n_groups", "ids_sha",
                                 "y_sha", "groups_sha") if k in roster}
    fp["blocks_sha"] = roster.get("blocks_sha")
    fp["h_dim"] = roster.get("h_dim")
    return fp


def _manifest_path(sd: Path, r: int, f: int) -> Path:
    return sd / f"shard_r{r}_f{f}.manifest.json"


def _fitlog_path(sd: Path, r: int, f: int) -> Path:
    return sd / f"fitlog_r{r}_f{f}.json"


def write_shard(sd: Path, r: int, f: int, *, arm: str, train, test,
                res: dict, audit: list, conditions, roster: dict,
                cfg: V3AnalysisConfig) -> dict:
    """**正式写盘入口**。分数 → 审计 → 拟合日志 → manifest（提交点），顺序固定。

    补条件时按条件合并，不整片覆盖；已有条件的共享块与 S* 由调用方逐位核对过。
    返回写出的 manifest。
    """
    shard = sd / f"shard_r{r}_f{f}.npz"
    mpath, fpath = _manifest_path(sd, r, f), _fitlog_path(sd, r, f)
    recomputed = {c.name for c in conditions}
    train, test = np.asarray(train), np.asarray(test)

    # ── ① 分数（含旧键合并） ────────────────────────────────────────────
    store: dict[str, np.ndarray] = {"test_idx": test, "train_idx": train}
    if shard.exists():
        with np.load(shard, allow_pickle=False) as z:
            store.update({k: z[k] for k in z.files})
        store["test_idx"], store["train_idx"] = test, train
    for k, v in res["shared"].items():
        store[f"shared|{k}"] = np.asarray(v, dtype=float)
    arrays_present: dict[str, dict[str, str]] = {}
    for cname, c in res["conditions"].items():
        present = {}
        for k in COND_ARRAYS:
            key = f"{cname}|{k}"
            v = c.get(k)
            if v is None:
                store.pop(key, None)          # 补算后变 None：不留旧值
                present[k] = "null"
                continue
            a = np.asarray(v, dtype=float)
            want = train.size if k in _TRAIN_SIDE else test.size
            if a.shape != (want,):
                raise RuntimeError(
                    f"{cname}|{k} 长度 {a.shape} ≠ 期望 ({want},) —— 停，不写。"
                    f"（训练侧应与 train 等长，测试侧应与 test 等长）")
            store[key] = a
            present[k] = "present"
        arrays_present[cname] = present
    np.savez_compressed(shard, **store)

    # ── ② 选参审计 ─────────────────────────────────────────────────────
    _merge_audit(sd / f"audit_r{r}_f{f}.json", audit, recomputed)

    # ── ③ 阶段化拟合日志（每一次实际拟合一行），按条件合并 ────────────────
    rows = list(res.get("fit_log") or [])
    old_rows = []
    if fpath.exists():
        old_rows = [x for x in json.loads(fpath.read_text(encoding="utf-8"))
                    if x.get("condition") not in recomputed]
    fpath.write_text(json.dumps(old_rows + rows, ensure_ascii=False, indent=1,
                                default=float) + "\n", encoding="utf-8")

    # ── ④ manifest = 提交点，最后写 ─────────────────────────────────────
    old_conds = {}
    if mpath.exists():
        try:
            prev = json.loads(mpath.read_text(encoding="utf-8"))
            if prev.get("schema") == SHARD_SCHEMA:
                old_conds = {k: v for k, v in prev.get("conditions", {}).items()
                             if k not in recomputed}
        except json.JSONDecodeError:
            old_conds = {}
    by_name = {c.name: c for c in conditions}
    n_fit = {}
    for x in rows:
        n_fit[x.get("condition")] = n_fit.get(x.get("condition"), 0) + 1
    # ⚠️ 指纹**按条件存**，不存在 manifest 顶层。
    #    旧写法把 code_fingerprint 放顶层、每次写盘刷新，而 old_conds 原样保留 ——
    #    结果是：改代码 → 旧条件正确判未完成 → 只重算另一个条件 →
    #    这一次写盘把顶层指纹刷成新的 → **没重算的旧条件又被判成完成**。
    #    这是 Codex 用模拟分片复现出来的真 bug，不是理论风险。
    code_now, input_now = _code_fingerprint(), _input_fingerprint(roster)
    split_now = {"n_train": int(train.size), "n_test": int(test.size),
                 "train_rows_sha": lib._rows_sha(train),
                 "test_rows_sha": lib._rows_sha(test),
                 "fold_seed": cfg.fold_seed(r)}
    new_conds = {}
    for cname, c in res["conditions"].items():
        cond = by_name[cname]
        new_conds[cname] = {
            "complete": True,
            "condition_fingerprint": _condition_fingerprint(cond),
            # 这一条件**自己**是在哪份代码 / 哪份输入 / 哪个划分下算出来的
            "code_fingerprint": code_now,
            "input_fingerprint": input_now,
            "split": split_now,
            # ⚠️ 行索引与共享块也要进**该条件自己**的指纹。
            #    test_idx 决定这一折的预测对应哪些样本 —— 它被调换而分数不动，
            #    评价就会整体错位，而"分数摘要"一个都不会变。
            #    共享块（B1..B5 / S* 的输入侧）同理：条件之间共享，
            #    但如果被改过，这个条件的结果也不再对应它当初算出来的东西。
            "index_sha": {"test_idx": _array_sha(test),
                          "train_idx": _array_sha(train)},
            "shared_sha": {k: _array_sha(store[f"shared|{k}"])
                           for k in sorted(res["shared"])},
            "score_mode": cond.score_mode, "preprocess": cond.preprocess,
            "author_fixed": bool(cond.author_fixed), "kind": cond.kind,
            "arrays": arrays_present[cname],
            # 内容摘要：盘上数组被改了行序或数值，光看"键在不在"查不出来
            "array_sha": {k: _array_sha(store[f"{cname}|{k}"])
                          for k, st in arrays_present[cname].items()
                          if st == "present"},
            "n_fit_log_rows": n_fit.get(cname, 0),
            "outer_fit": c.get("outer_fit"),
            "secs": c.get("secs"),
        }
    man = {
        "schema": SHARD_SCHEMA, "arm": arm, "repeat": r, "fold": f,
        "last_writer": {"code_fingerprint": code_now,
                        "input_fingerprint": input_now, "split": split_now,
                        "conditions_written": sorted(recomputed),
                        "note": "仅记录**最近一次**写盘；验收一律读各条件自己的指纹"},
        "shared_blocks": sorted(res["shared"].keys()),
        "shared_sha": {k: _array_sha(store[f"shared|{k}"])
                       for k in res["shared"]},
        "conditions": {**old_conds, **new_conds},
        "array_contract": {"test_required": list(TEST_ARRAYS_REQUIRED),
                           "test_nullable": list(TEST_ARRAYS_NULLABLE),
                           "train_required": list(TRAIN_ARRAYS_REQUIRED),
                           "train_nullable": list(TRAIN_ARRAYS_NULLABLE)},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float)
                     + "\n", encoding="utf-8")
    return man


def read_shard(sd: Path, r: int, f: int) -> dict:
    """**正式读盘入口**。round-trip 检查用的就是它，不另写一份读法。"""
    shard, mpath, fpath = (sd / f"shard_r{r}_f{f}.npz",
                           _manifest_path(sd, r, f), _fitlog_path(sd, r, f))
    if not shard.exists():
        raise FileNotFoundError(shard)
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    out: dict = {"manifest": man, "shared": {}, "conditions": {},
                 "schema": (man or {}).get("schema", LEGACY_SHARD_SCHEMA)}
    with np.load(shard, allow_pickle=False) as z:
        keys = set(z.files)
        out["test_idx"] = z["test_idx"]
        out["train_idx"] = z["train_idx"] if "train_idx" in keys else None
        for k in keys:
            if k.startswith("shared|"):
                out["shared"][k.split("|", 1)[1]] = z[k]
        cnames = sorted({k.split("|", 1)[0] for k in keys
                         if "|" in k and not k.startswith("shared|")})
        for cname in cnames:
            d = {}
            for k in COND_ARRAYS:
                d[k] = z[f"{cname}|{k}"] if f"{cname}|{k}" in keys else None
            out["conditions"][cname] = d
    out["fit_log"] = (json.loads(fpath.read_text(encoding="utf-8"))
                      if fpath.exists() else [])
    apath = sd / f"audit_r{r}_f{f}.json"
    out["audit"] = (json.loads(apath.read_text(encoding="utf-8"))
                    if apath.exists() else [])
    return out


def shard_status(sd: Path, r: int, f: int, conditions, *, arm: str,
                 roster: dict, cfg: V3AnalysisConfig, train, test) -> dict:
    """该分片上**哪些条件真的完成了**。续跑只认这个，不认 `condition|H` 存在。

    返回 `{cond_name: (done: bool, reason: str)}` 外加 `legacy` 标记。
    任何一项指纹对不上 → 不算完成；**不静默接受**，理由原样带出来。
    """
    shard, mpath = sd / f"shard_r{r}_f{f}.npz", _manifest_path(sd, r, f)
    res = {"legacy": False, "conditions": {}}
    if not shard.exists():
        res["conditions"] = {c.name: (False, "无分片") for c in conditions}
        return res
    if not mpath.exists():
        # 旧格式：分数在，但新增字段与拟合日志从来没写过。**标 legacy**，
        # 不当作完成，也不删它。
        res["legacy"] = True
        res["conditions"] = {c.name: (False, "legacy 分片（无 manifest，缺新增字段）")
                             for c in conditions}
        return res
    man = json.loads(mpath.read_text(encoding="utf-8"))
    code_now, input_now = _code_fingerprint(), _input_fingerprint(roster)
    want_split = {"train_rows_sha": lib._rows_sha(np.asarray(train)),
                  "test_rows_sha": lib._rows_sha(np.asarray(test))}
    with np.load(shard, allow_pickle=False) as z:
        keys = set(z.files)
        # **所有**键都算摘要，不只带 "|" 的 —— test_idx / train_idx 也在内
        disk_sha = {k: _array_sha(z[k]) for k in keys}
    fl = _fitlog_path(sd, r, f)
    fit_rows = json.loads(fl.read_text(encoding="utf-8")) if fl.exists() else []
    n_fit = {}
    for x in fit_rows:
        n_fit[x.get("condition")] = n_fit.get(x.get("condition"), 0) + 1
    apath = sd / f"audit_r{r}_f{f}.json"
    audit_conds = {a.get("condition") for a in
                   (json.loads(apath.read_text(encoding="utf-8"))
                    if apath.exists() else [])}
    for c in conditions:
        e = man.get("conditions", {}).get(c.name)
        why = None
        if man.get("schema") != SHARD_SCHEMA:
            why = f"schema={man.get('schema')!r} ≠ {SHARD_SCHEMA}"
        elif e is None or not e.get("complete"):
            why = "manifest 未标 complete"
        elif e.get("condition_fingerprint") != _condition_fingerprint(c):
            why = "条件定义指纹不符（候选集/评分口径/预处理被改过）"
        elif "code_fingerprint" not in e:
            # v2 早期的 manifest 把指纹放在顶层，无法分辨某个条件到底是在
            # 哪份代码下算的 —— 一律判未完成，不给它"沾光"的机会
            why = "该条件没有自己的代码指纹（早期 manifest 布局），按未完成处理"
        elif e.get("code_fingerprint") != code_now:
            why = "该条件的代码指纹不符（lib/driver 在它算完之后改过）"
        elif e.get("input_fingerprint") != input_now:
            diffs = [k for k in set(input_now) | set(e.get("input_fingerprint") or {})
                     if (e.get("input_fingerprint") or {}).get(k) != input_now.get(k)]
            why = f"该条件的输入指纹不符（{sorted(diffs)[:3]}）"
        elif e.get("split", {}).get("train_rows_sha") != want_split["train_rows_sha"] \
                or e.get("split", {}).get("test_rows_sha") != want_split["test_rows_sha"]:
            why = "该条件的划分指纹不符"
        else:
            miss = [k for k, st in e.get("arrays", {}).items()
                    if st == "present" and f"{c.name}|{k}" not in keys]
            need = [k for k in TEST_ARRAYS_REQUIRED + TRAIN_ARRAYS_REQUIRED
                    if e.get("arrays", {}).get(k) != "present"]
            # 内容校验：盘上数组被改了行序或数值 → 摘要对不上
            tampered = [k for k, v in (e.get("array_sha") or {}).items()
                        if disk_sha.get(f"{c.name}|{k}") != v]
            no_sha = [k for k, st in e.get("arrays", {}).items()
                      if st == "present" and k not in (e.get("array_sha") or {})]
            # 行索引：盘上的 test_idx/train_idx 既要与 manifest 登记一致，
            # 也要与**本次调用传进来的划分**一致（两头都卡）
            idx_sha = e.get("index_sha") or {}
            want_idx = {"test_idx": _array_sha(np.asarray(test)),
                        "train_idx": _array_sha(np.asarray(train))}
            idx_bad = [k for k in ("test_idx", "train_idx")
                       if idx_sha.get(k) != disk_sha.get(k)
                       or idx_sha.get(k) != want_idx[k]]
            # 共享块：条件之间共享，但改了就不再对应该条件当初的结果
            sh_reg = e.get("shared_sha") or {}
            sh_bad = [k for k, v in sh_reg.items()
                      if disk_sha.get(f"shared|{k}") != v]
            if miss:
                why = f"manifest 声明存在但盘上缺：{miss}"
            elif need:
                why = f"必需数组缺失：{need}"
            elif no_sha:
                why = f"缺内容摘要，无法校验：{no_sha}"
            elif not idx_sha or not sh_reg:
                why = "该条件没有登记行索引 / 共享块摘要（早期 manifest），按未完成处理"
            elif idx_bad:
                why = f"**行索引与登记或当前划分不符**：{idx_bad}（预测会整体错位）"
            elif sh_bad:
                why = f"**共享块内容与 manifest 摘要不符**：{sh_bad}"
            elif tampered:
                why = f"**盘上数组内容与 manifest 摘要不符**：{tampered}"
            elif n_fit.get(c.name, 0) != e.get("n_fit_log_rows"):
                why = (f"拟合日志行数 {n_fit.get(c.name, 0)} ≠ manifest 记的 "
                       f"{e.get('n_fit_log_rows')}")
            elif not e.get("n_fit_log_rows"):
                why = "拟合日志为空"
            elif c.name not in audit_conds:
                why = "选参审计缺该条件"
        res["conditions"][c.name] = (why is None, why or "完成")
    return res


def _shard_dir(arm: str, version: str = DEFAULT_OUTPUT_VERSION) -> Path:
    """版本化分片目录。`legacy_v1` 指向旧目录，**只读**，正式运行拒绝写它。"""
    if version == LEGACY_OUTPUT_VERSION:
        return OUT_DIR / f"run_{arm}"
    return OUT_DIR / f"run_{arm}__{version}"


def _results_path(arm: str, version: str) -> Path:
    if version == LEGACY_OUTPUT_VERSION:
        return OUT_DIR / f"results_{arm}.json"
    return OUT_DIR / f"results_{arm}__{version}.json"


def _oof_path(arm: str, version: str, cname: str) -> Path:
    if version == LEGACY_OUTPUT_VERSION:
        return OUT_DIR / f"oof_{arm}_{cname}.npz"
    return OUT_DIR / f"oof_{arm}__{version}_{cname}.npz"


def cmd_run(arm: str, cfg: V3AnalysisConfig, conditions, max_rss: float,
            min_free: float = 3.0, min_headroom: float = 3.0,
            version: str = DEFAULT_OUTPUT_VERSION) -> int:
    from recover_attention.evaluation import v3_analysis_paths as paths
    from recover_attention.evaluation import v3_bootstrap as vb

    if version == LEGACY_OUTPUT_VERSION:
        raise SystemExit(
            "[stop] 拒绝写 legacy_v1 —— 旧的六条件报告与 OOF 是**唯一**的一份，"
            "不覆盖。新结果写 --output-version（默认 "
            f"{DEFAULT_OUTPUT_VERSION}）的独立目录。")
    lock = _acquire_lock()
    sd = _shard_dir(arm, version)
    sd.mkdir(parents=True, exist_ok=True)
    names_s, _ = paths.names_for_path(paths.PATH_A)
    cnames = [c.name for c in conditions]

    print(f"[run] {arm}  条件 {cnames}", flush=True)
    g0, t_all = _gen_rate_probe(), time.perf_counter()
    bundle = load_arm(arm, cfg)
    y, groups, blocks = bundle["y"], bundle["groups"], bundle["blocks"]
    n, peak = y.size, _rss_gb()
    roster = _roster(arm, bundle, cfg)
    print(f"  n={roster['n']} 正例={roster['n_positive']} 组={roster['n_groups']} "
          f"Hdim={roster['h_dim']}  RSS={peak}G", flush=True)

    # ── ① 逐 (repeat, fold) 跑；已有分片直接跳过（断点续跑） ────────────────
    for r in range(cfg.n_repeats):
        folds, _ = _fold_of(y, groups, cfg, r)
        for f, (train, test) in enumerate(folds):
            shard = sd / f"shard_r{r}_f{f}.npz"
            # **续跑判据 = manifest 的 complete 标记 + 五项指纹 + 字段齐全**，
            # 不是"`condition|H` 在不在"。分数写完而日志没写 → 算未完成，重算。
            st = shard_status(sd, r, f, conditions, arm=arm, roster=roster,
                              cfg=cfg, train=train, test=test)
            todo = [c for c in conditions if not st["conditions"][c.name][0]]
            if st["legacy"]:
                print(f"    r{r} f{f}  ⚠️ legacy 分片（无 manifest）—— 原样保留，"
                      f"按未完成重算到 {version}", flush=True)
            if not todo:
                print(f"    r{r} f{f}  已有完整分片（manifest 校验通过），跳过",
                      flush=True)
                continue
            skipped = [c.name for c in conditions if st["conditions"][c.name][0]]
            if skipped:
                print(f"    r{r} f{f}  跳过已完成 {skipped}；重算 "
                      f"{[(c.name, st['conditions'][c.name][1]) for c in todo]}",
                      flush=True)
            mem = _mem_gate(min_free, min_headroom)
            audit = []
            t0 = time.perf_counter()
            res = run_fold(y=y, groups=groups, blocks=blocks, repeat=r, fold=f,
                           train=train, test=test, cfg=cfg, conditions=todo,
                           seed=cfg.base_random_state, audit_sink=audit)
            if shard.exists():
                # **"复用同一份输出侧缓存"不能只是假定** —— 补条件时把重算出来的
                # 共享块与 S* 和已落盘的逐位比对，对不上就停。
                bad = []
                with np.load(shard, allow_pickle=False) as z:
                    have = set(z.files)
                    for k, v in res["shared"].items():
                        key = f"shared|{k}"
                        if key in have:
                            d = float(np.max(np.abs(np.asarray(v) - z[key])))
                            if d != 0.0:
                                bad.append((key, d))
                    for c in sorted({k.split("|")[0] for k in have
                                     if k.endswith("|S*")}):
                        d = float(np.max(np.abs(
                            np.asarray(res["conditions"][todo[0].name]["S*"])
                            - z[f"{c}|S*"])))
                        if d != 0.0:
                            bad.append((f"S*[已有条件 {c}]", d))
                if bad:
                    raise RuntimeError(
                        f"r{r} f{f}: 补条件时共享输出侧与已落盘分片不一致 {bad[:4]} —— "
                        f"停，不写。条件之间必须共享同一份 O 侧分数。")
            write_shard(sd, r, f, arm=arm, train=train, test=test, res=res,
                        audit=audit, conditions=todo, roster=roster, cfg=cfg)
            # 写完立刻用**正式读盘入口**回读核对（round-trip），不信"写过就对"
            chk = read_shard(sd, r, f)
            for cname, c in res["conditions"].items():
                for k in COND_ARRAYS:
                    a, b = c.get(k), chk["conditions"][cname][k]
                    if a is None:
                        if b is not None:
                            raise RuntimeError(f"r{r} f{f} {cname}|{k}: 应为 None 却读到值")
                        continue
                    d = float(np.max(np.abs(np.asarray(a, dtype=float) - b)))
                    if d != 0.0:
                        raise RuntimeError(f"r{r} f{f} {cname}|{k}: 回读差 {d:.3e} —— 停")
            peak = max(peak, _rss_gb())
            print(f"    r{r} f{f}  {time.perf_counter() - t0:.1f}s  "
                  + "  ".join(f"{k}={v['secs']}s" for k, v in res["conditions"].items())
                  + f"  RSS={_rss_gb()}G  整机free={mem.get('free_gb')}G"
                    f" commit余{mem.get('commit_headroom_gb')}G", flush=True)
            if peak > max_rss:
                raise MemoryError(f"RSS {peak}G 超过闸 {max_rss}G —— 先保生成任务，已中止")

    # ── ② 拼 OOF ─────────────────────────────────────────────────────────
    # 测试折三路分数都进 OOF；训练侧 inner-OOF 不是全样本量（每折只覆盖 train），
    # 不拼成全长向量，**按折原样保留**，见 `train_side` 落盘。
    OOF_KEYS = TEST_ARRAYS_REQUIRED + TEST_ARRAYS_NULLABLE
    oof = {c: [{k: np.full(n, np.nan) for k in OOF_KEYS}
               for _ in range(cfg.n_repeats)] for c in cnames}
    oof_null = {c: set() for c in cnames}
    shared = [{k: np.full(n, np.nan) for k in names_s} for _ in range(cfg.n_repeats)]
    fold_of = [np.full(n, -1, dtype=int) for _ in range(cfg.n_repeats)]
    train_side = {c: {} for c in cnames}
    for r in range(cfg.n_repeats):
        for f in range(cfg.n_outer_folds):
            sh = read_shard(sd, r, f)          # 正式读盘入口，不另写读法
            te = sh["test_idx"]
            fold_of[r][te] = f
            for k in names_s:
                shared[r][k][te] = sh["shared"][k]
            for c in cnames:
                cd = sh["conditions"][c]
                for k in OOF_KEYS:
                    if cd[k] is None:
                        oof_null[c].add(k)
                    else:
                        oof[c][r][k][te] = cd[k]
                for k in TRAIN_ARRAYS_REQUIRED + TRAIN_ARRAYS_NULLABLE:
                    if cd[k] is not None:
                        train_side[c][f"r{r}_f{f}_{k}"] = np.asarray(cd[k])
                train_side[c][f"r{r}_f{f}_train_idx"] = np.asarray(sh["train_idx"])
                train_side[c][f"r{r}_f{f}_test_idx"] = np.asarray(te)
    for c in cnames:
        for r in range(cfg.n_repeats):
            for k in OOF_KEYS:
                if k in oof_null[c]:
                    continue
                if not np.all(np.isfinite(oof[c][r][k])):
                    raise RuntimeError(f"{c} r{r} {k}: OOF 有非有限值")

    # S* 必须在所有条件间逐位相同（"只换 H"的结构性证据）
    s_star_consistent = all(
        float(np.max(np.abs(oof[c][r]["S*"] - oof[cnames[0]][r]["S*"]))) == 0.0
        for c in cnames for r in range(cfg.n_repeats))

    # ── ③ 点估计（逐 repeat 算差 → 跨 repeat 平均，同主路径 estimand） ──────
    def stats(o, rows=None):
        idx = np.arange(n) if rows is None else rows
        per = []
        for r in range(cfg.n_repeats):
            ys = y[idx]
            a_s = ci.auroc(o[r]["S*"][idx], ys)
            a_h = ci.auroc(o[r]["H"][idx], ys)
            a_sh = ci.auroc(o[r]["S*+H"][idx], ys)
            per.append((a_s, a_h, a_sh, a_sh - a_s))
        a = np.asarray(per, dtype=float)
        return (a, tuple(float(np.mean(a[:, j])) if np.all(np.isfinite(a[:, j]))
                         else float("nan") for j in range(4)))

    point = {}
    for c in cnames:
        per, m = stats(oof[c])
        point[c] = {"auroc_S*": m[0], "auroc_H": m[1], "auroc_S*+H": m[2], "delta_H": m[3],
                    "per_repeat": [{"repeat": i, "auroc_S*": float(v[0]),
                                    "auroc_H": float(v[1]), "auroc_S*+H": float(v[2]),
                                    "delta_H": float(v[3])} for i, v in enumerate(per)]}

    # ── ④ 配对区间：同一 cluster bootstrap 抽样，同一批行喂所有条件 ─────────
    print(f"[run] bootstrap B={cfg.b_total}（cluster/group，条件间**配对**）…", flush=True)
    plan = vb.build_resample_plan(groups, cfg)
    draws = {c: [] for c in cnames}
    n_disc = 0
    t0 = time.perf_counter()
    c0 = cnames[0]
    for _q, _j, rows in plan.iter_draws():
        ys = y[rows]
        # S* 在所有条件间**逐位相同**（上面已结构性核对），故每轮只算一次
        a_s = [ci.auroc(oof[c0][r]["S*"][rows], ys) for r in range(cfg.n_repeats)]
        vals, ok = {}, np.all(np.isfinite(a_s))
        if ok:
            for c in cnames:
                per = [ci.auroc(oof[c][r]["S*+H"][rows], ys) - a_s[r]
                       for r in range(cfg.n_repeats)]
                if not np.all(np.isfinite(per)):
                    ok = False
                    break
                vals[c] = float(np.mean(per))
        if not ok:                          # 单类抽样 → **整轮丢弃**，保持配对
            n_disc += 1
            continue
        for c in cnames:
            draws[c].append(vals[c])
    print(f"  {time.perf_counter() - t0:.0f}s；丢弃 {n_disc}/{plan.b_total} 轮（单类）",
          flush=True)

    ref = "ref_l2_wide_grid" if "ref_l2_wide_grid" in cnames else cnames[0]
    table = {}
    for c in cnames:
        d = np.asarray(draws[c], dtype=float)
        ci_d = vb.percentile_ci(d, cfg.alpha)
        row = {**{k: v for k, v in point[c].items() if k != "per_repeat"},
               "delta_H_ci": ci_d, "per_repeat": point[c]["per_repeat"]}
        if c != ref:
            pd_ = d - np.asarray(draws[ref], dtype=float)   # **配对**：同一批抽样行
            row["delta_H_minus_ref"] = {
                "ref": ref, "point": point[c]["delta_H"] - point[ref]["delta_H"],
                "paired_ci": vb.percentile_ci(pd_, cfg.alpha)}
        table[c] = row

    # ── ⑤ 落盘 ───────────────────────────────────────────────────────────
    audits = []
    for r in range(cfg.n_repeats):
        for f in range(cfg.n_outer_folds):
            audits.extend(json.loads((sd / f"audit_r{r}_f{f}.json").read_text(encoding="utf-8")))
    # ⚠️ 选参统计**必须按阶段分开**。把 inner_cf（内部交叉拟合）与 outer_full
    # （外层最终模型）混在一个分母里，会得到"H1 59/60 选了 50 轮"这种数 ——
    # 它既不是最终模型的统计，也不是候选验证的统计。
    def _blank() -> dict:
        return {"n_selections": 0, "at_boundary": 0, "upper": 0, "lower": 0,
                "fell_back": 0, "escalated": 0, "not_converged": 0,
                "chosen_counts": {}}

    def _acc(t: dict, a: dict) -> None:
        t["n_selections"] += 1
        if a.get("fell_back"):
            t["fell_back"] += 1
            return
        t["at_boundary"] += int(bool(a.get("at_boundary")))
        t["upper"] += int(a.get("which_boundary") == "upper")
        t["lower"] += int(a.get("which_boundary") == "lower")
        key = json.dumps(a["chosen"], sort_keys=True, default=str)
        t["chosen_counts"][key] = t["chosen_counts"].get(key, 0) + 1
        for cv in a.get("curve", []):
            t["escalated"] += int(bool(cv.get("max_iter_escalated")))
            t["not_converged"] += int(cv.get("converged_all") is False)

    tune = {}
    for a in audits:
        e = tune.setdefault(a["condition"], {
            "_note": ("all_stages 只为与旧报告对照保留；**引用时用 by_phase**。"
                      "outer_full = 实际用于测试折的最终模型；"
                      "inner_cf = 为融合层生成 inner-OOF 的中间拟合。"),
            "all_stages": _blank(), "by_phase": {}})
        _acc(e["all_stages"], a)
        _acc(e["by_phase"].setdefault(a.get("phase", "unknown"), _blank()), a)

    # 阶段化拟合日志汇总：候选验证 vs 选中模型最终拟合，分开计数
    fit_rows = []
    for r in range(cfg.n_repeats):
        for f in range(cfg.n_outer_folds):
            p = _fitlog_path(sd, r, f)
            if p.exists():
                fit_rows.extend(json.loads(p.read_text(encoding="utf-8")))
    fitsum: dict = {}
    for x in fit_rows:
        c = x.get("condition")
        key = f"{x.get('phase')}|{x.get('stage', 'candidate_validation')}"
        s = fitsum.setdefault(c, {}).setdefault(key, {
            "n_fits": 0, "n_converged_true": 0, "n_converged_false": 0,
            "n_no_criterion": 0, "n_reached_max_iter": 0, "n_escalated": 0,
            "n_ladder_exhausted": 0, "max_iter_used_counts": {},
            "n_iter_min": None, "n_iter_max": None, "secs_total": 0.0})
        s["n_fits"] += 1
        cv = x.get("converged")
        s["n_converged_true"] += int(cv is True)
        s["n_converged_false"] += int(cv is False)
        s["n_no_criterion"] += int(cv is None)
        s["n_reached_max_iter"] += int(bool(x.get("reached_max_iter")))
        s["n_escalated"] += int(bool(x.get("max_iter_escalated")))
        s["n_ladder_exhausted"] += int(bool(x.get("ladder_exhausted")))
        mk = str(x.get("max_iter_used"))
        s["max_iter_used_counts"][mk] = s["max_iter_used_counts"].get(mk, 0) + 1
        ni = x.get("n_iter")
        if ni is not None and np.isfinite(float(ni)):
            ni = float(ni)
            s["n_iter_min"] = ni if s["n_iter_min"] is None else min(s["n_iter_min"], ni)
            s["n_iter_max"] = ni if s["n_iter_max"] is None else max(s["n_iter_max"], ni)
        s["secs_total"] = round(s["secs_total"] + float(x.get("secs") or 0.0), 2)

    g1 = _gen_rate_probe()
    groups_of_conditions = {
        "prespecified": [c for c in cnames if c in lib.PRESPECIFIED],
        "posthoc_sensitivity": [c for c in cnames if c in lib.POSTHOC_SENSITIVITY],
    }
    out = {"schema": "posthoc_readout_probe_results_v3", "arm": arm,
           "output_version": version,
           "supersedes": ("results_{arm}.json（legacy_v1，六条件、只存 S*/S*+H/H）—— "
                          "**旧报告与旧 OOF 原样保留，不覆盖**；两者不可混引"),
           "code_fingerprint": _code_fingerprint(),
           "shard_schema": SHARD_SCHEMA,
           "fit_log_summary": fitsum,
           "fit_log_note": ("按 phase|stage 分开计数。"
                            "`candidate_validation` = 选参时的候选验证拟合；"
                            "`selected_model_fit` = 选中参数后的最终拟合。"
                            "两者分母不同，**不可互相除**。"
                            "XGB 的 converged 记为 null（`n_no_criterion`）—— "
                            "跑满树数是预算完成，没有检验过收敛。"),
           "roster": roster, "conditions": cnames,
           "condition_groups": groups_of_conditions,
           "grouping_note": ("`posthoc_sensitivity` 组是**首轮结果完成后**才追加的，"
                             "依据是内层选参触边界；**不是事前规则**，"
                             "报告中必须与事前条件分开呈现，不混列。"),
           "s_star_identical_across_conditions": s_star_consistent,
           "results": table, "tuning_summary": tune,
           "fit_accounting": {c.name: fit_accounting(c, cfg) for c in conditions},
           "bootstrap": {"b_total": plan.b_total, "discarded_single_class": n_disc,
                         "unit": "cluster/group bootstrap（同主路径）",
                         "paired": "所有条件共用同一批重采样行"},
           "peak_rss_gb": peak, "generation_during": _gen_rate_delta(g0, g1),
           "memory_gate": {"min_free_gb": min_free, "min_commit_headroom_gb": min_headroom,
                           "final": _sys_mem()},
           "total_secs": round(time.perf_counter() - t_all, 1)}
    rp = _results_path(arm, version)
    rp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n",
                  encoding="utf-8")
    for c in cnames:
        np.savez_compressed(
            _oof_path(arm, version, c),
            sample_id=bundle["sample_ids"], group_id=np.asarray(groups).astype(str), y=y,
            **{f"r{r}_fold": fold_of[r] for r in range(cfg.n_repeats)},
            # 测试折三路分数（旧版只有 S*/S*+H/H）
            **{f"r{r}_{k}": oof[c][r][k] for r in range(cfg.n_repeats)
               for k in OOF_KEYS if k not in oof_null[c]},
            # 训练侧 inner-OOF：逐折原样保存 + train_idx/test_idx，行顺序可回溯
            **train_side[c])
    (OUT_DIR / (f"tuning_audit_{arm}.json" if version == LEGACY_OUTPUT_VERSION
                else f"tuning_audit_{arm}__{version}.json")).write_text(
        json.dumps(audits, ensure_ascii=False, indent=1, default=float) + "\n",
        encoding="utf-8")
    (OUT_DIR / f"fit_log_{arm}__{version}.json").write_text(
        json.dumps(fit_rows, ensure_ascii=False, indent=1, default=float) + "\n",
        encoding="utf-8")

    def _row(c):
        t = table[c]
        print(f"{c:<20}{t['auroc_S*']:>9.4f}{t['auroc_H']:>9.4f}{t['auroc_S*+H']:>9.4f}"
              f"{t['delta_H']:>+10.4f}"
              f"  [{t['delta_H_ci']['ci_lo']:+.4f}, {t['delta_H_ci']['ci_hi']:+.4f}]")

    print(f"\n{'条件':<20}{'S*':>9}{'H':>9}{'S*+H':>9}{'d_H':>10}{'95% CI':>24}")
    print("-- 事前固定 " + "-" * 58)
    for c in groups_of_conditions["prespecified"]:
        _row(c)
    if groups_of_conditions["posthoc_sensitivity"]:
        print("-- 事后敏感性（首轮结果后追加，非事前规则）" + "-" * 27)
        for c in groups_of_conditions["posthoc_sensitivity"]:
            _row(c)
    print(f"\n[out] {rp}  用时 {out['total_secs']}s  "
          f"峰值 RSS {peak}G  期间生成 {out['generation_during']}")
    lock.unlink(missing_ok=True)
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 重建被覆盖的选参审计
#
# MCQ 续跑时把第一轮三个便宜条件的 audit 整片覆盖了（见 `_merge_audit` 的说明）。
# 分片里的**分数**没丢，丢的只有选参过程记录。重算一遍即可补回 ——
# 但**不能只是"重算了就算数"**：必须把重算出来的 H / S* / S*+H 与分片里
# 已落盘的值**逐位比对**，对上了才说明重建出来的审计确实对应那次运行。
# 对不上就停，不写。
# ══════════════════════════════════════════════════════════════════════════
def cmd_rebuild_audit(arm: str, cfg: V3AnalysisConfig, conditions, max_rss: float,
                      version: str = LEGACY_OUTPUT_VERSION) -> int:
    from recover_attention.evaluation import v3_analysis_paths as paths

    # 默认对着 **legacy_v1** 跑 —— 这个命令就是为修旧目录被覆盖的审计写的。
    sd = _shard_dir(arm, version)
    names_s, _ = paths.names_for_path(paths.PATH_A)
    cnames = [c.name for c in conditions]
    print(f"[rebuild-audit] {arm}  条件 {cnames}", flush=True)
    g0, t_all = _gen_rate_probe(), time.perf_counter()
    bundle = load_arm(arm, cfg)
    y, groups, blocks = bundle["y"], bundle["groups"], bundle["blocks"]
    peak, mismatches, n_rows = _rss_gb(), [], 0

    for r in range(cfg.n_repeats):
        folds, _ = _fold_of(y, groups, cfg, r)
        for f, (train, test) in enumerate(folds):
            shard = sd / f"shard_r{r}_f{f}.npz"
            if not shard.exists():
                raise FileNotFoundError(f"{shard} 不存在 —— 无可比对的已落盘分数")
            audit = []
            res = run_fold(y=y, groups=groups, blocks=blocks, repeat=r, fold=f,
                           train=train, test=test, cfg=cfg, conditions=conditions,
                           seed=cfg.base_random_state, audit_sink=audit)
            with np.load(shard) as z:
                for c in cnames:
                    for k in ("S*", "S*+H", "H"):
                        key = f"{c}|{k}"
                        if key not in z.files:
                            mismatches.append((r, f, key, "分片里没有这一项"))
                            continue
                        d = float(np.max(np.abs(
                            np.asarray(res["conditions"][c][k], dtype=float) - z[key])))
                        if d != 0.0:
                            mismatches.append((r, f, key, f"最大差 {d:.3e}"))
                for k in names_s:                       # 共享 O 侧也要对上
                    d = float(np.max(np.abs(res["shared"][k] - z[f"shared|{k}"])))
                    if d != 0.0:
                        mismatches.append((r, f, f"shared|{k}", f"最大差 {d:.3e}"))
            if mismatches:
                print(f"    r{r} f{f}  **不一致，已停**：{mismatches[:3]}", flush=True)
                return 3
            merged = _merge_audit(
                sd / f"audit_r{r}_f{f}.json", audit, set(cnames),
                provenance="recovered_by_recompute_verified_bitwise_against_shard")
            n_rows += len(audit)
            peak = max(peak, _rss_gb())
            print(f"    r{r} f{f}  重算 {len(audit)} 条，与分片逐位一致；"
                  f"该折审计现有 {len(merged)} 条  RSS={_rss_gb()}G", flush=True)
            if peak > max_rss:
                raise MemoryError(f"RSS {peak}G 超过闸 {max_rss}G")

    rep = {"schema": "posthoc_audit_rebuild_v1", "arm": arm, "conditions": cnames,
           "why": ("续跑时 audit_r*_f*.json 被整片覆盖（只写了本次补算的条件），"
                   "三个便宜条件的选参记录丢失；分片中的分数未受影响"),
           "method": ("以相同 cfg / 种子 / 划分重跑 select_params 与打分，"
                      "并与已落盘分片**逐位比对** S* / S*+H / H 及全部共享块，"
                      "全部为 0 差异才写回"),
           "verified_bitwise": True, "n_audit_rows_recovered": n_rows,
           "provenance_tag": "recovered_by_recompute_verified_bitwise_against_shard",
           "peak_rss_gb": peak, "generation_during": _gen_rate_delta(g0, _gen_rate_probe()),
           "total_secs": round(time.perf_counter() - t_all, 1)}
    f = OUT_DIR / f"audit_rebuild_{arm}.json"
    f.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float) + "\n",
                 encoding="utf-8")
    print(f"\n[ok] 重建 {n_rows} 条审计，全部逐位核对通过  用时 {rep['total_secs']}s")
    print(f"[out] {f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("h1", "mcq"), default="mcq")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--run", action="store_true", help="正式运行（可断点续跑）")
    ap.add_argument("--rebuild-audit", action="store_true",
                    help="重算被覆盖的选参审计，并与已落盘分片逐位核对")
    ap.add_argument("--reproduce", action="store_true",
                    help="原网格 L2 复现核对（两层：主路径 vs 冻结、harness vs 主路径）")
    ap.add_argument("--max-rss-gb", type=float, default=DEFAULT_MAX_RSS_GB)
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="整机空闲物理内存低于此值就先等（生成任务优先）")
    ap.add_argument("--min-commit-headroom-gb", type=float, default=3.0,
                    help="提交量余量低于此值就先等 —— 触顶会打死生成任务")
    ap.add_argument("--output-version", default=DEFAULT_OUTPUT_VERSION,
                    help=("输出版本（分片目录 / results / OOF 都按它分开）。"
                          f"默认 {DEFAULT_OUTPUT_VERSION}；`{LEGACY_OUTPUT_VERSION}` "
                          "是旧的六条件产物，**只读**，--run 拒绝写入。"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    prio = _lower_own_priority()
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    names = args.conditions or [c.name for c in lib.CONDITIONS]
    conds = [lib.CONDITION_BY_NAME[n] for n in names]

    if args.self_check:
        return cmd_self_check(cfg)
    print(f"[prio] 本进程优先级 → {prio}（生成任务优先）", flush=True)
    if args.rebuild_audit:
        return cmd_rebuild_audit(args.arm, cfg, conds, args.max_rss_gb)
    if args.run:
        return cmd_run(args.arm, cfg, conds, args.max_rss_gb,
                       args.min_free_gb, args.min_commit_headroom_gb,
                       version=args.output_version)
    if args.reproduce:
        return cmd_reproduce(args.arm, cfg, args.max_rss_gb)
    if args.timing_only:
        return cmd_timing(args.arm, cfg, conds, args.max_rss_gb)
    print("正式运行尚未接入；先跑 --self-check 与 --timing-only。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
