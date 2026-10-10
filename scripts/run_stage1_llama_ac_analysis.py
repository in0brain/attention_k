r"""Stage 1 Llama 跨模型复核：**Path A + Path C** 的最小分析入口（非确证性）。

═══ 为什么另起入口，而不是"只跑 A+C" ═══════════════════════════════════════
完整入口 `v3_analyze.make_analyze_callback` 会在构造回调前硬性拒绝：

    shadow_overlay 未过硬门            → OverlayRefused（Path B）
    e3_blocks 未加载 / 非 canonical repool → OverlayRefused（E3）

Llama 侧**没有** tail_stabilized 与 repool 产物（生产卡第 44 行明确不启动）。
**不得**为了跑通而删掉那些检查 —— 那是解盲前就定下的硬门。
所以本脚本另起一个入口，**只复用统计核心**，一次都不调用 `make_analyze_callback`：

    v3_analysis_paths.blocks_for_path / names_for_path   路径的块名单
    v3_analyze._summarise                                repeats→点估计→bootstrap→§6 必报项
    v3_adapter / v3_adapter_mcq                          与 Qwen 同一组 adapter
    v3_config.V3AnalysisConfig（assert_production）       划分/选参/融合/bootstrap 全部沿用

产物另存到 `outputs/stage1_llama_v3/analysis/`，不碰 Qwen 的任何冻结文件。

═══ 范围与命名（必须随结果一起出现） ═══════════════════════════════════════
    schema            v3_analyze.REPLICATION_ANALYSIS_SCHEMA（与 confirmatory 分开）
    population_role   cross_model_replication_non_confirmatory
    scope             **Stage 1 部分范围的非确证性跨模型分析**
                      —— 缺 Path B（tail overlay）与 E3（repool），**不是完整复现**
    B4 完整性         复现角色 fail-closed 为 `not_audited`：
                      **不继承** Qwen 的 compromised，也**不因**最终字段都是有限值
                      就宣布完整性通过。缺证据就保留 not_audited。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# 线程数**在 import numpy 之前**定好，并记进产物。
# 默认不限制（= 核数）：生成任务已结束、计划任务已注销，机器空着；
# 且 Qwen 的 confirmatory 也是默认线程跑的 —— 跨模型比较用同一档更一致。
# 本轮无"必须逐位对上某个冻结值"的约束（Llama 侧没有冻结参照），
# 但线程数仍如实记录：它会改变大词表稀疏 TF-IDF 上 liblinear 的停止点（实测 1e-6 量级）。
_THREADS = os.environ.get("RA_AC_THREADS") or str(os.cpu_count())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_analysis_paths as paths   # noqa: E402
from recover_attention.evaluation import v3_analyze as va             # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig   # noqa: E402

PROD = ROOT / "outputs/stage1_llama_v3/production"
OUT_DIR = ROOT / "outputs/stage1_llama_v3/analysis"
POPULATION_ROLE = "cross_model_replication_non_confirmatory"
RUN_PATHS = (paths.PATH_A, paths.PATH_C)

SCOPE_NOTE = (
    "Stage 1 部分范围的**非确证性**跨模型分析：只跑 Path A（frozen_implementation_primary）"
    "与 Path C（no_B4_dependency_sensitivity）。**缺 Path B 与 E3**"
    "（Llama 侧未产出 tail_stabilized overlay 与 canonical repool，生产卡明确不启动）。"
    "**不得称为完整复现**，也不得与 Qwen 的 confirmatory 结果并列为同类证据。")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_arm(arm: str) -> dict:
    """与 Qwen 主路径**同一组 adapter**，不另造口径。"""
    from recover_attention.evaluation import v3_adapter as ah1
    from recover_attention.evaluation import v3_adapter_mcq as amcq

    recs = [json.loads(x) for x in
            (PROD / f"{arm}_completion_records.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    z = np.load(PROD / f"{arm}_hidden.npz")
    hidden = {k: z[k] for k in z.files}
    if arm == "mcq":
        used = amcq.filter_analyzable(recs)
        b = amcq.build_mcq_arm(used, hidden)
    else:
        # H1 的 eligible 过滤在主路径里由 feature_cache 做，adapter 本身不筛
        used = [r for r in recs
                if bool(r.get("eligible", r.get("eligible_for_primary")))]
        b = ah1.build_arm(used, hidden)
    idf = b.get("id_field", "trace_id")
    b["sample_ids"] = np.asarray([str(r[idf]) for r in used])
    b["id_field_used"] = idf
    b["n_raw_records"] = len(recs)
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["h1", "mcq"])
    ap.add_argument("--paths", nargs="*", default=list(RUN_PATHS))
    ap.add_argument("--out", type=Path, default=OUT_DIR / "llama_ac_analysis.json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    for p in args.paths:
        if p not in RUN_PATHS:
            raise SystemExit(f"本入口只跑 {list(RUN_PATHS)}；Path B 与 E3 需要 Llama 侧"
                             f"未产出的 overlay / repool 产物，**不得**绕过硬门")

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = {}
    for arm in args.arms:
        for suf in ("completion_records.jsonl", "hidden.npz", "finalization_report.json"):
            f = PROD / f"{arm}_{suf}"
            inputs[f"{arm}_{suf}"] = {"path": str(f.relative_to(ROOT)),
                                      "bytes": f.stat().st_size, "sha256": _sha(f)}

    out: dict = {
        "schema": va.REPLICATION_ANALYSIS_SCHEMA,
        "population_role": POPULATION_ROLE,
        "is_confirmatory": False,
        "scope": SCOPE_NOTE,
        "paths_run": list(args.paths),
        "paths_not_run": {
            paths.PATH_B: "Llama 侧无 tail_stabilized overlay —— 硬门未过，不绕过",
            "E3": "Llama 侧无 canonical repool —— E3 是独立 estimand，缺输入即不跑",
        },
        # 复现角色 fail-closed：**不继承** Qwen 的 compromised
        "numeric_integrity": {
            "status": paths.NUMERIC_NOT_AUDITED,
            "b4_scope": "H1 B4 的 logit-derived F5 字段",
            "evidence": None,
            "why_not_audited": (
                "Llama 的 B4 数值完整性**未做独立审计**。既不自动继承 Qwen 的 "
                "compromised 裁定，也**不因**最终字段都是有限值就宣布完整性通过 —— "
                "『无非有限值』只说明落盘字段可用，不等于生成期没有静默过滤。"),
            "fail_closed_note": "缺失审计记为 not_audited，**不**等价于通过",
        },
        "runtime": {"threads": _THREADS,
                    "threads_note": ("默认不限制（=核数）。与 Qwen confirmatory 同档。"
                                     "线程数会改变大词表稀疏 TF-IDF 上 liblinear 的"
                                     "停止点（实测 1e-6 量级），故如实记录。"),
                    "python": sys.version.split()[0]},
        "config": {"n_repeats": cfg.n_repeats, "n_outer_folds": cfg.n_outer_folds,
                   "n_inner_folds": cfg.n_inner_folds,
                   "c_grid": list(cfg.c_grid),
                   "fold_seeds": [cfg.fold_seed(r) for r in range(cfg.n_repeats)],
                   "b_total": cfg.b_total, "alpha": cfg.alpha,
                   "note": "划分 / 选参 / 融合 / bootstrap / 判读规则全部沿用 Stage 1"},
        "inputs": inputs,
        "path_markers": {p: paths.path_markers(p, paths.NUMERIC_NOT_AUDITED)
                         for p in args.paths},
        "arms": {},
        "rosters": {},
    }

    for path in args.paths:
        out["arms"][path] = {}
        for arm in args.arms:
            print(f"═══ {path} / {arm}", flush=True)
            b = load_arm(arm)
            y, groups, blocks = b["y"], b["groups"], b["blocks"]
            out["rosters"][arm] = {
                "n": int(y.size), "n_positive": int(y.sum()),
                "n_groups": int(len(set(map(str, groups)))),
                "n_raw_records": b["n_raw_records"], "id_field": b["id_field_used"],
                "analysis_unit": "trace" if arm == "h1" else "question",
            }
            print(f"  n={y.size} 正例={int(y.sum())} 组={out['rosters'][arm]['n_groups']}",
                  flush=True)
            t = time.perf_counter()
            sh = va._summarise(y, groups, blocks, cfg, path, estimand="E1")
            print(f"  {time.perf_counter() - t:.0f}s  "
                  f"S*={sh['auroc']['S*']:.6f}  H={sh['auroc']['H']:.6f}  "
                  f"S*+H={sh['auroc']['S*+H']:.6f}  "
                  f"Δ_H={sh['delta_h_point']:+.6f} "
                  f"[{sh['delta_h_ci']['ci_lo']:+.6f}, {sh['delta_h_ci']['ci_hi']:+.6f}]",
                  flush=True)
            out["arms"][path][arm] = va._strip_internal(sh)

            # ⚠️ 逐样本 OOF **必须落盘**。`_summarise` 内部已经把它算出来了
            # （`_oof_per_repeat`），旧版被 `_strip_internal` 一并丢掉 ——
            # 结果是没人能从预测独立复算 AUROC/CI，只能信汇总里的数。
            # 落盘是**零额外计算**，只是多写几十 MB。
            oof_dir = args.out.parent / "oof"
            oof_dir.mkdir(parents=True, exist_ok=True)
            reps = sh.get("_oof_per_repeat") or []
            if reps:
                np.savez_compressed(
                    oof_dir / f"oof_{arm}__{path}.npz",
                    sample_id=np.asarray(b["sample_ids"]).astype(str),
                    group_id=np.asarray(groups).astype(str), y=np.asarray(y),
                    **{f"r{i}_{k}": np.asarray(o[k], dtype=float)
                       for i, o in enumerate(reps) for k in o})
                print(f"  [oof] {oof_dir / f'oof_{arm}__{path}.npz'}  "
                      f"{len(reps)} repeat × {len(reps[0])} 列", flush=True)
            else:
                print("  ⚠️ _summarise 未返回 _oof_per_repeat —— 未落盘，如实登记",
                      flush=True)
            out["arms"][path][arm]["oof_artifact"] = (
                f"oof/oof_{arm}__{path}.npz" if reps else None)

    out["oof_note"] = (
        "逐样本 OOF 落在 analysis/oof/ 下，含 sample_id / group_id / y / "
        "各 repeat 的 S* / S*+H / 各块分数，可用于独立复算 AUROC 与 bootstrap。"
        "**2026-09-20 之前的那次运行没有这一项**（入口当时把 _oof_per_repeat 丢了），"
        "其汇总数值未被改动，但要从预测独立复算需重跑（约 11706 s / 32 线程）。")
    out["total_secs"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n",
                        encoding="utf-8")
    print(f"\n[out] {args.out}  用时 {out['total_secs']}s")
    print(f"⚠️ {SCOPE_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
