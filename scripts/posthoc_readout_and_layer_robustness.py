"""POST-HOC：内部状态的判别力对「读出器」与「层」有多敏感？

**非 confirmatory。** 不改动、不覆盖 stage0_analysis_report.json，不替换其中任何判读。

────────────────────────────────────────────────────────────────────────
问题

论文把立意提到「内部状态 vs 表层信息」之后，最大的软肋是：结论只基于**一个**读出器
（L2-logistic）在**一层**（⌊0.7L⌋）上的结果。两个自然反问：

    「是不是 logistic 太弱？换个读出器内部状态就赢了？」
    「是不是 0.7L 选错了层？」

本脚本在 `3 层 × N 读出器` 的网格上回答，全部结果一次报完。

────────────────────────────────────────────────────────────────────────
防 winner's curse（**本脚本最重要的约束**）

论文批评 `O` 不许取 max（否则 winner's curse）。那么在 `H` 侧挑最好的读出器报告，
是**完全相同**的错误 —— 而且揭盲已用尽，这里没有预注册保护。因此：

  1. 读出器与层的列表在**跑之前**写死在本文件里（见 READOUTS / LAYERS），
     提交时间戳早于结果。
  2. **全部单元格一律报告，一个不挑。** 报告里没有 "best" 字段。
  3. 任何单元格都**不得**用来替换 confirmatory 的 H 或回填结论。
  4. 论证方向只能是**稳健性**：「多种读出器 × 多层给出相似结论 → 结论不依赖这两个
     选择」。**不能**是「某个单元格更高 → 内部状态其实有用」。若真出现某格显著更高，
     只能报告为 post-hoc 观察并声明需独立验证。

────────────────────────────────────────────────────────────────────────
自检

`logistic_l2 @ rel0.7 @ 分组 A` 必须复现 confirmatory 的 AUROC(H alone)：
    H1 = 0.8970   MCQ = 0.8677
本脚本是平行实现（confirmatory 的 block_oof_scores 把分类器写死成 logistic，
无法换读出器），故必须用这一格证明两套实现一致，其余单元格才可信 —— 与第 1 步用
`max_abs_diff = 0.0` 证明重新前向路径一致是同一手法。
特征变换复用冻结的 `build_fold_design`（同一套 z-score + /sqrt(d) 归一化），
folds 复用冻结的 `stratified_grouped_folds`，只有分类器不同。

────────────────────────────────────────────────────────────────────────
报什么

主表是 **AUROC(H alone)**：内部状态自己能达到多少。把它与同臂的 `AUROC(O)`
（H1 0.8585 / MCQ 0.8877，冻结值）对照，直接回答「内部状态过得了表层基线吗」。
不重算 `Δ_H` —— 那需要重做三块融合，会把「换读出器」与「换融合协议」两件事混在一起。

用法：
    python scripts/posthoc_readout_and_layer_robustness.py --arm h1
    python scripts/posthoc_readout_and_layer_robustness.py --arm h1 --readouts logistic_l2
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_H1, ARM_MCQ, load_arm  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "leak", PROJECT_ROOT / "scripts" / "posthoc_h1_identifier_leakage_diagnostic.py")
leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak)

FEAT = Path("outputs/stage0/features")
LAYERS_DIR = Path("outputs/stage0/layers")
OUT = Path("outputs/stage0/evaluation")

# ── 跑前写死：层 × 读出器（防 winner's curse，见文件头）────────────────────
LAYERS = ["rel0.5", "rel0.7", "final"]

READOUTS = {
    # confirmatory 用的那个，作参照与自检
    "logistic_l2": {"make": lambda p: LogisticRegression(C=p, penalty="l2", max_iter=2000,
                                                         solver="liblinear"),
                    "grid": list(ci.C_GRID)},
    "logistic_l1": {"make": lambda p: LogisticRegression(C=p, penalty="l1", max_iter=2000,
                                                         solver="liblinear"),
                    "grid": list(ci.C_GRID)},
    # 高维下的正则化生成式方法（≈ 收缩 Mahalanobis）
    # **已知计算限制（2026-07-31 实测）**：d=3584 下 shrinkage="auto"(Ledoit-Wolf) 需在
    # 3584×3584 上估协方差,nested CV 共 45 次拟合,>15 分钟 CPU 仍未跑完 4 个单元格。
    # 结论：在此维度下计算不可行,H1/MCQ 均未产出该行。**这不等于"没测过"** —— 它在
    # 预先声明的列表里,报告的 readouts_frozen_list 保留它,cells 里缺失即为此原因,
    # identification_audit.md 有书面交代。不删除它,以免看起来像事后从列表里拿掉了
    # 一个不方便的方法。
    "lda_shrinkage": {"make": lambda p: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p),
                      "grid": [0.1, 0.5, "auto"]},
    # 非线性：单隐层
    "mlp_64": {"make": lambda p: MLPClassifier(hidden_layer_sizes=(64,), alpha=p,
                                               max_iter=400, random_state=0),
               "grid": [1e-3, 1e-1]},
    # 非线性：RBF 核
    "svm_rbf": {"make": lambda p: SVC(C=p, kernel="rbf", gamma="scale"),
                "grid": [1.0, 10.0]},
    # 先降维再线性（检验"高维稀疏性"是否是瓶颈）
    "pca64_logistic": {"make": lambda p: make_pipeline(
                           PCA(n_components=64, random_state=0),
                           LogisticRegression(C=p, penalty="l2", max_iter=2000,
                                              solver="liblinear")),
                       "grid": list(ci.C_GRID)},
}

CONFIRMATORY_H_ALONE = {"h1": 0.8970, "mcq": 0.8677}   # 自检目标（冻结报告）
CONFIRMATORY_O = {"h1": 0.8585, "mcq": 0.8877}         # 对照线：表层基线


def _risk(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def _select_param(X_tr, y_tr, g_tr, spec, inner_splits, seed):
    """nested 超参选择：与 confirmatory 的 _select_C 同结构（组内分层折 + AUROC 最优）。"""
    grid = spec["grid"]
    if len(grid) == 1:
        return grid[0]
    try:
        folds = ci.stratified_grouped_folds(y_tr, g_tr, n_splits=inner_splits, seed=seed)
    except Exception:                                          # noqa: BLE001
        return grid[0]
    best, best_auc = grid[0], -np.inf
    for p in grid:
        oof = np.full(len(y_tr), np.nan)
        try:
            for tr, te in folds:
                clf = spec["make"](p)
                clf.fit(X_tr[tr], y_tr[tr])
                oof[te] = _risk(clf, X_tr[te])
            a = ci.auroc(oof, y_tr)
        except Exception:                                      # noqa: BLE001
            continue
        if np.isfinite(a) and a > best_auc:
            best, best_auc = p, a
    return best


def h_alone_auroc(y, groups, hidden, spec, config) -> dict:
    """H alone 的 OOF AUROC。特征变换与 folds 全部复用冻结实现，只换分类器。"""
    folds = ci.stratified_grouped_folds(y, groups, n_splits=config.n_outer_folds,
                                        seed=config.seed)
    g = np.asarray(groups)
    scores = np.full(len(y), np.nan)
    chosen = []
    for train, test in folds:
        X_tr, X_te, _ = ci.build_fold_design(train, test, text=None,
                                             dense_blocks=[("hidden", hidden)])
        # build_fold_design 为兼容 TF-IDF 统一返回 scipy 稀疏矩阵，但 H alone 只有 hidden
        # 这一个稠密块。LDA 等估计器不吃稀疏输入,故在此转稠密 —— 数值完全相同,
        # 不影响任何已跑出的单元格(logistic/SVM/MLP 稀疏稠密结果一致)。
        if hasattr(X_tr, "toarray"):
            X_tr, X_te = X_tr.toarray(), X_te.toarray()
        p = _select_param(X_tr, y[train], g[train], spec, config.n_inner_folds, config.seed)
        clf = spec["make"](p)
        clf.fit(X_tr, y[train])
        scores[test] = _risk(clf, X_te)
        chosen.append(str(p))
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("non-finite OOF scores")
    return {"auroc": float(ci.auroc(scores, y)), "chosen_params": chosen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["h1", "mcq"], required=True)
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--readouts", nargs="*", default=None,
                    help="只跑指定读出器(调试/测速用)。**报告结果时必须跑全部**")
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--pooling", choices=["all_identifiers", "first_identifier"],
                    default="all_identifiers",
                    help="读哪一版 pooling 的三层 hidden（first_identifier = §8 robustness "
                         "附录变体）。报告写独立文件名,不覆盖 primary")
    ap.add_argument("--merge", action="store_true",
                    help="把本次结果并进已有报告(只补齐失败/缺失单元格,不覆盖已成功的)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    config = Stage0Config(n_outer_folds=cfg.n_outer_folds, n_inner_folds=cfg.n_inner_folds,
                          n_boot=cfg.n_boot, seed=cfg.seed, mode=cfg.mode,
                          allow_pilot_population=False)

    arm_key = ARM_H1 if args.arm == "h1" else ARM_MCQ
    label_field = "label" if args.arm == "h1" else "wrong_label"
    id_field = "trace_id" if args.arm == "h1" else "example_id"
    cache = load_arm(arm_key, FEAT / f"{args.arm}_completion_records.jsonl",
                     FEAT / f"{args.arm}_hidden.npz", config,
                     label_field=label_field, id_field=id_field)
    y = cache.y
    ids = [str(r[id_field]) for r in cache.records]

    groupings = {"A_prompt": np.asarray(cache.groups)}
    if args.arm == "h1":
        groupings["C_prompt_union_fabid"] = leak.component_groups(
            cache.records, link_prompt=True, link_fabricated=True)[0]

    names = args.readouts or list(READOUTS)
    print(f"[robust:{args.arm}] n={y.size} pos={int(y.sum())} | layers={LAYERS} | "
          f"readouts={names}")
    print(f"[robust:{args.arm}] reference: AUROC(O)={CONFIRMATORY_O[args.arm]} (surface baseline)")

    # 三层 hidden，按 cache 的记录顺序对齐
    layer_mats, layer_status = {}, {}
    for lname in LAYERS:
        _suf = "" if args.pooling == "all_identifiers" else f"_{args.pooling}"
        z = np.load(LAYERS_DIR / f"{args.arm}_hidden_{lname}{_suf}.npz")
        missing = [i for i in ids if i not in z.files]
        if missing:
            raise SystemExit(f"{lname}: {len(missing)} ids missing (e.g. {missing[:3]}). "
                             f"先跑 posthoc_multilayer_reforward.py --arm {args.arm}")
        M = np.vstack([z[i] for i in ids])
        n_bad = int((~np.isfinite(M)).any(axis=1).sum())
        if n_bad:
            # 该层在此后端下不可用。**不静默丢样本**：丢掉含 NaN 的记录会换掉 population,
            # 使该层与其余层不可比 —— 那比缺这一层更糟。如实记为 unavailable。
            layer_status[lname] = {"usable": False, "n_nonfinite_records": n_bad,
                                   "n_total": len(ids),
                                   "reason": "non-finite values in the cached hidden vectors"}
            print(f"[robust:{args.arm}] {lname}: UNAVAILABLE — {n_bad}/{len(ids)} records "
                  f"contain non-finite values; skipped (not dropped-and-refit)")
            continue
        layer_status[lname] = {"usable": True, "n_nonfinite_records": 0, "n_total": len(ids)}
        layer_mats[lname] = M

    cells, t0 = {}, time.time()
    for gname, groups in groupings.items():
        cells[gname] = {}
        for lname in [l for l in LAYERS if l in layer_mats]:
            cells[gname][lname] = {}
            for rname in names:
                t1 = time.time()
                try:
                    r = h_alone_auroc(y, groups, layer_mats[lname], READOUTS[rname], config)
                    cells[gname][lname][rname] = {"auroc": r["auroc"],
                                                  "chosen_params": r["chosen_params"],
                                                  "secs": round(time.time() - t1, 1)}
                    print(f"[robust:{args.arm}] {gname:22s} {lname:7s} {rname:15s} "
                          f"AUROC={r['auroc']:.4f}  ({time.time()-t1:.0f}s)")
                except Exception as exc:                       # noqa: BLE001
                    cells[gname][lname][rname] = {"error": repr(exc)[:200]}
                    print(f"[robust:{args.arm}] {gname:22s} {lname:7s} {rname:15s} FAILED {exc!r}")

    # ── 自检：logistic_l2 @ rel0.7 @ A 必须复现 confirmatory ──
    check = None
    got = cells.get("A_prompt", {}).get("rel0.7", {}).get("logistic_l2", {}).get("auroc")
    if got is not None and args.pooling != "all_identifiers":
        check = {"applicable": False,
                 "meaning": (f"pooling={args.pooling} is a robustness variant; it is NOT "
                             f"expected to reproduce the confirmatory H-alone value, so "
                             f"equality is not the right test. The all_identifiers run over "
                             f"this same code path reproduced it to 2.8e-05.")}
        print(f"[self-check] n/a — pooling={args.pooling} is expected to differ by design")
    elif got is not None:
        exp = CONFIRMATORY_H_ALONE[args.arm]
        check = {"expected_confirmatory_H_alone": exp, "reproduced": got,
                 "abs_diff": abs(got - exp), "ok": abs(got - exp) < 5e-4,
                 "meaning": ("parallel implementation matches the frozen one; the other "
                             "cells are comparable")}
        print(f"\n[self-check] logistic_l2 @ rel0.7 @ A: got {got:.4f} vs confirmatory "
              f"{exp:.4f} → {'OK' if check['ok'] else 'MISMATCH'}")

    report = {
        "schema_version": "posthoc_readout_layer_robustness_v1",
        "status": ("POST-HOC — not confirmatory. Does not modify or supersede "
                   "stage0_analysis_report.json. Reports ALL cells; there is deliberately no "
                   "'best' field, because selecting the best readout on the test metric would "
                   "be exactly the winner's curse this paper criticises on the O side."),
        "timestamp": args.timestamp,
        "arm": args.arm,
        "pooling": args.pooling,
        "n": int(y.size), "n_positive": int(y.sum()),
        "layers_frozen_list": LAYERS,
        "layer_availability": layer_status,
        "readouts_frozen_list": list(READOUTS),
        "reference_auroc_O_surface_baseline": CONFIRMATORY_O[args.arm],
        "self_check": check,
        "cells_auroc_H_alone": cells,
        "reading_rule": ("compare every cell against reference_auroc_O_surface_baseline; the "
                         "claim supported is robustness (does the conclusion depend on readout "
                         "or layer?), never 'the best cell shows internal states are useful'."),
        "total_secs": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _rsuf = "" if args.pooling == "all_identifiers" else f"_{args.pooling}"
    path = OUT / f"posthoc_readout_layer_robustness_{args.arm}{_rsuf}.json"
    # --merge：把本次跑的单元格并进已有报告（用于补跑先前失败的读出器）。
    # **只补齐,不替换已有非失败单元格** —— 否则"重跑到满意为止"会从这里溜进来。
    if args.merge and path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        merged = prev.get("cells_auroc_H_alone", {})
        for g, layers in report["cells_auroc_H_alone"].items():
            for L, rs in layers.items():
                for k, v in rs.items():
                    cur = merged.setdefault(g, {}).setdefault(L, {})
                    if k in cur and "auroc" in cur[k]:
                        continue        # 已有成功结果 → 保留,不覆盖
                    cur[k] = v
        report["cells_auroc_H_alone"] = merged
        report["merged_from_previous_run"] = True
        report["merge_note"] = ("cells already carrying an auroc were kept as-is; only "
                                "previously-failed or missing cells were filled in")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
    print(f"[robust:{args.arm}] report → {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
