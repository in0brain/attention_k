r"""POST-HOC：H 侧读出器适配对照的**条件定义与执行原语**。

回答的问题只有一个：**当前 H 侧读出器及其调参设置，是否限制了隐藏状态能提供的增量？**

⚠️ **非 confirmatory。** 不修改、不覆盖 `stage1_confirmatory_result.json`，
不进冻结的 `src/recover_attention/evaluation/` 路径。全部产物另存。**全部本机执行。**

════════════════════════════════════════════════════════════════════════════
代码来源（三栏分清，报告照此口径）

复用作者代码
    Han et al. 2025 —— https://github.com/JThh/fact-probe
    commit 851b70cbdc8a800b135afb9323ff6c4d6996b06c （2025-07-06）
    复用 `long_fact_probes/train.py::get_classifier()` 的**分类器构造与参数**：
        L1   LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                max_iter=1000, random_state=42)
        XGB  XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=6,
                           objective='binary:logistic', use_label_encoder=False,
                           eval_metric='auc', random_state=42)

我们修改的地方（均为**本地执行适配**或**协议一致性**，逐条记录）
    · XGB 显式 `n_jobs=1` + CPU —— 生成任务优先，且避免多线程不确定性
      （作者未指定 n_jobs，默认吃满核；作者的 gpu_hist 分支本轮不用）
    · `max_iter` 是**参照不是约束**：按作者值起跑，若未收敛则按事前写死的阶梯
      上调并记录差异 —— **不为"保持原样"而接受未收敛结果**
    · 不复用 `run_cv_probing()`：它用无分组的 StratifiedKFold、跨层拼接选最佳
      layer group、选完在全量数据上重拟合、且有裸 `except: continue`（会静默跳过）。
      本项目的分组划分 / 嵌套交叉拟合 / 共享 S* / 融合协议一律保留，不被其默认值替换。
    · 不装其 requirements：其中 `torch>=1.13.0,<2.0.0` 会降级本机 torch 2.11，
      毁掉正在运行的生成任务。实际只需 sklearn + xgboost，现有版本均满足其下限。

依据论文自行实现
    SAPLMA（Azaria & Mitchell 2023, Findings of EMNLP, §4）
    **未找到可核实的作者官方实现**，故依据论文 §4 自行实现：
        256 → 128 → 64，ReLU，末端 sigmoid，Adam
    论文正文只公开了 true-false 数据集地址；**不据此推断代码的发布位置**。
    结构之外的适配差异（必须在报告中列明）：
        · 原文训练 5 个 epoch；本项目把 epoch 作为**候选超参**（5 / 50），
          由训练折内的 grouped inner CV 选择
        · 原文未说明正则强度；本项目对 `alpha` 在训练折内选参
        · **不使用** sklearn 的 `early_stopping=True` —— 它会随机切验证集，
          破坏分组隔离；"早停"改由 epoch 作为候选来体现
        · 输入是本项目冻结的 z-score + /√d 后的 H 向量，与原文的输入口径不同

统一称 **"基于作者实现的适配对照"**，不称原论文完整复现。

════════════════════════════════════════════════════════════════════════════
正类方向

本项目的 y **以 1 表示需要检出的那一类**（错误/幻觉）。读出器一律取
`predict_proba` 中 `classes_ == 1` 的那一列，**不做任何 1-p 反转** ——
作者原任务以 "is_supported" 为正是**他们的标签口径**，我们重新训练用的是自己的标签，
再反转一次就是二次反转。由 `verify_positive_class_direction()` 小规模验证。

meta 输入必须是 log-odds（M8 冻结）。logreg 直接取 `decision_function`；
MLP / XGB 只给概率，按**事前固定**的裁剪换算，训练与测试用**同一条规则**：

    p_clip = clip(p, PROB_EPS, 1 - PROB_EPS);  score = log(p_clip / (1 - p_clip))
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from recover_attention import conditional_increment as ci

#: 概率 → log-odds 的**事前**裁剪常数。训练与测试同一条规则，只换算一次。
PROB_EPS = 1e-6
#: 未收敛时的 max_iter 阶梯（事前写死）。作者值在首位；用到后面的必须记录差异。
MAX_ITER_LADDER = (1000, 5000, 20000)
#: 计算预算上限（单臂）。超了按 `reduction_order` **确定性**砍候选，不看外层测试表现。
#
# ⚠️ **2026-09-17 修订，在看到任何外层测试结果之前**：原写 8.0 h 是**没有实测**时拍的，
# 实测后（单次拟合真实规模成本，见 single_fit_probe_*.json）两臂分别约 7 h / 23 h。
# 修订依据只有资源事实，与任何条件的表现无关：
#     · 单线程（OMP=1），占 32 核中的 1 核
#     · 峰值 RSS 实测 0.18 G（MCQ）/ 0.86 G（H1），闸设 6 G
#     · 对生成速率的实测影响 +18%（无探针 45.0 s/unit vs 带探针 53.3 s/unit），
#       且整机另有内存闸兜底（见 `--min-free-gb` / `--min-commit-headroom-gb`）
#     · 分片落盘，可随时中断续跑
BUDGET_HOURS_PER_ARM = 24.0

AUTHOR_REPO = "https://github.com/JThh/fact-probe"
AUTHOR_COMMIT = "851b70cbdc8a800b135afb9323ff6c4d6996b06c"


# ══════════════════════════════════════════════════════════════════════════
# 条件表 —— **在看到任何外层测试结果之前固定**
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Condition:
    name: str
    kind: str                         # "logreg_main" | "logreg_author" | "mlp" | "xgb"
    candidates: tuple[dict, ...]
    paper_reference_params: dict | None
    provenance: str                   # reused_author_code | implemented_from_paper | project
    source: str
    notes: str
    reduction_order: tuple[int, ...] = field(default_factory=tuple)
    #: 喂给融合层的分数口径。
    #:   "clipped_logodds" —— 旧口径：概率按 PROB_EPS 裁剪后转 log-odds
    #:   "raw_margin"      —— 新口径：直接用模型的**原始分数**
    #:                        XGB = output_margin；MLP = sigmoid 前的输出
    #: logreg 两种口径下都用 decision_function（它本身就是原始 log-odds，从不裁剪）
    score_mode: str = "clipped_logodds"
    #: H 块喂给读出器前的预处理口径。
    #:   "zscore_div_sqrt_d" —— 本项目冻结口径（训练折内 z-score，再除 sqrt(d)）
    #:   "raw_activation"    —— 作者口径：**原始激活**，不做任何变换
    #: ⚠️ 相同的 C 在两种口径下**不是**相同的有效正则强度，必须分条件报告。
    #: 两种口径都在训练折内完成（raw 不含任何用到测试折的统计量）。
    preprocess: str = "zscore_div_sqrt_d"
    #: 是否**只用原文固定配置**（不调参）。True 时候选集只有一个元素。
    author_fixed: bool = False


SELECTION_METRIC = "mean grouped-inner-CV AUROC"
TIE_BREAK = "ascending traversal + strict >; ties keep the first (simpler) candidate"

CONDITIONS: tuple[Condition, ...] = (
    Condition(
        name="ref_l2_frozen_grid", kind="logreg_main",
        candidates=tuple({"C": c} for c in (0.01, 0.1, 1.0, 10.0)),
        paper_reference_params=None, provenance="project",
        source="本项目冻结主路径（v3_config.c_grid + base_lr_kwargs）",
        notes="复现核对用：**逐项沿用主路径 kwargs**，否则复现核对无意义。",
    ),
    Condition(
        name="ref_l2_wide_grid", kind="logreg_main",
        candidates=tuple({"C": c} for c in
                         (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)),
        paper_reference_params=None, provenance="project",
        source="本项目（第 0 步：审计 H 侧 C 范围是否够宽）",
        notes="**只放宽 H 侧候选**，包含原网格；输出侧与融合层规则不变。",
        reduction_order=(7, 6),
    ),
    Condition(
        name="saplma_mlp", kind="mlp",
        candidates=tuple({"hidden_layer_sizes": (256, 128, 64), "alpha": a,
                          "max_iter": e}
                         for e in (5, 50) for a in (1e-5, 1e-3, 1e-1)),
        paper_reference_params={"hidden_layer_sizes": (256, 128, 64),
                                "activation": "relu", "solver": "adam", "epochs": 5},
        provenance="implemented_from_paper",
        source="Azaria & Mitchell 2023, Findings of EMNLP, §4（SAPLMA）",
        notes=("**未找到可核实的作者官方实现**，依据论文 §4 自行实现架构。"
               "epoch 与 alpha 在训练折内选参；原文 5 epoch 列入候选。"),
        reduction_order=(5, 4, 3),
    ),
    Condition(
        name="han_l1", kind="logreg_author",
        candidates=tuple({"C": c} for c in (0.01, 0.1, 0.5, 1.0, 10.0)),
        paper_reference_params={"penalty": "l1", "solver": "liblinear", "C": 0.5,
                                "max_iter": 1000, "random_state": 42},
        provenance="reused_author_code",
        source=f"{AUTHOR_REPO} @ {AUTHOR_COMMIT[:12]} :: get_classifier()",
        notes=("分类器构造与参数复用作者代码；C 的候选集是本项目约定，"
               "作者默认 [0.1, 0.5] 与附录 D 的 0.5 均在其中。选参全在训练折内。"),
    ),
    Condition(
        name="han_l1_wide", kind="logreg_author",
        candidates=tuple({"C": c} for c in
                         (0.01, 0.1, 0.5, 1.0, 10.0, 1e2, 1e3, 1e4)),
        paper_reference_params={"penalty": "l1", "solver": "liblinear", "C": 0.5,
                                "max_iter": 1000, "random_state": 42},
        provenance="reused_author_code",
        source=f"{AUTHOR_REPO} @ {AUTHOR_COMMIT[:12]} :: get_classifier()",
        notes=("⚠️ **首轮结果完成后，依据内层选参触边界追加的敏感性分析** —— "
               "`han_l1` 在 H1 上 60/60 次选到自身网格上界 C=10。"
               "**这不是事前规则**：对 L2 的放宽（ref_l2_wide_grid）是第 0 步的事前设计，"
               "对 L1 的放宽是看到首轮结果之后才决定的，报告中一律标为 post-hoc 敏感性分析，"
               "不与事前条件混列。候选集在运行前固定：**含 han_l1 的全部 5 个 C**，"
               "再加预定的更大值 1e2/1e3/1e4（上限与 ref_l2_wide_grid 一致）。"
               "其余划分、预处理、融合与统计协议与 han_l1 完全相同。"),
        reduction_order=(7, 6),
    ),
    Condition(
        name="han_xgboost", kind="xgb",
        candidates=tuple({"max_depth": d, "n_estimators": n, "learning_rate": 0.1}
                         for d in (3, 6) for n in (200, 1000)),
        paper_reference_params={"n_estimators": 1000, "learning_rate": 0.1,
                                "max_depth": 6, "objective": "binary:logistic",
                                "eval_metric": "auc", "random_state": 42},
        provenance="reused_author_code",
        source=f"{AUTHOR_REPO} @ {AUTHOR_COMMIT[:12]} :: get_classifier()",
        notes=("其余参数原样复用作者值；深度与树数的候选集是本项目约定，"
               "作者配置 (1000, 0.1, 6) 在其中。本地适配：n_jobs=1 + CPU。"
               "⚠️ 原写 reduction_order=(3, 1) **已在正式运行前撤回**："
               "index 3 就是作者参照配置 (1000, 0.1, 6)，砍它会与"
               "『作者参照点必须在候选集内』自相矛盾。实测预算够用，故不砍。"),
        reduction_order=(),
    ),
)

def _raw_variant(base: Condition, name: str, why: str) -> Condition:
    """从已有条件派生一个**只改打分口径**的原始分数条件。

    候选集、种子、划分、预处理、融合规则一律不变 —— 唯一差别是
    喂给融合层的是**模型原始分数**而不是裁剪后概率的 log-odds。
    """
    return Condition(
        name=name, kind=base.kind, candidates=base.candidates,
        paper_reference_params=base.paper_reference_params,
        provenance=base.provenance, source=base.source,
        notes=(f"⚠️ **打分口径敏感性条件（事后新增）**：{why} "
               f"除 score_mode 外与 `{base.name}` 完全相同。"
               f"旧裁剪条件 `{base.name}` **保留不动**，两者分开报告。"),
        reduction_order=base.reduction_order, score_mode="raw_margin")


def _variant(base: Condition, name: str, why: str, **over) -> Condition:
    """从已有条件派生：**只改指定的那一项**，其余全部继承。"""
    kw = dict(name=name, kind=base.kind, candidates=base.candidates,
              paper_reference_params=base.paper_reference_params,
              provenance=base.provenance, source=base.source,
              notes=(f"⚠️ **事后新增条件**：{why} "
                     f"除下列改动外与 `{base.name}` 完全相同："
                     f"{ {k: v for k, v in over.items()} }。"
                     f"原条件 `{base.name}` **保留不动**，分开报告。"),
              reduction_order=base.reduction_order,
              score_mode=base.score_mode, preprocess=base.preprocess,
              author_fixed=base.author_fixed)
    kw.update(over)
    return Condition(**kw)


_BASE = {c.name: c for c in CONDITIONS}
CONDITIONS = CONDITIONS + (
    # ── P1：输入口径分条件（作者原始激活 vs 本项目 z-score + /sqrt(d)）──────
    _variant(_BASE["han_l1"], "han_l1_authorinput",
             "用**作者口径的原始激活**作为读出器输入，不做 z-score + /sqrt(d)。"
             "相同 C 在两种口径下不是相同的有效正则，故必须分条件。",
             preprocess="raw_activation"),
    _variant(_BASE["han_xgboost"], "han_xgboost_authorinput",
             "用**作者口径的原始激活**作为读出器输入。"
             "树模型对单调变换不敏感，但 z-score 的**折内统计量**仍会改变分裂点。",
             preprocess="raw_activation"),
    # ── P1：原文固定配置 vs 适配调参，分别报告 ────────────────────────────
    _variant(_BASE["han_l1"], "han_l1_authorfixed",
             "**只用原文固定配置** C=0.5（附录 D），不调参。"
             "与调参版 `han_l1` 分别报告，不混为一谈。",
             candidates=({"C": 0.5},), author_fixed=True),
    _variant(_BASE["han_xgboost"], "han_xgboost_authorfixed",
             "**只用原文固定配置** (n_estimators=1000, lr=0.1, max_depth=6)，不调参。",
             candidates=({"max_depth": 6, "n_estimators": 1000,
                          "learning_rate": 0.1},), author_fixed=True),
    _raw_variant(_BASE["saplma_mlp"], "saplma_mlp_raw",
                 "MLP 取 **sigmoid 前**的输出，不经 PROB_EPS 裁剪。"
                 "旧口径下 H1 有 47.69% 的预测被裁剪边界夹住。"),
    _raw_variant(_BASE["han_xgboost"], "han_xgboost_raw",
                 "XGBoost 用原生 `output_margin=True`，不经 PROB_EPS 裁剪。"),
)

CONDITION_BY_NAME = {c.name: c for c in CONDITIONS}

#: **事前**固定的条件（第 0 步 + 第 1 步三个适配对照）。
PRESPECIFIED: tuple[str, ...] = ("ref_l2_frozen_grid", "ref_l2_wide_grid",
                                 "saplma_mlp", "han_l1", "han_xgboost")
#: **首轮结果完成后**追加的敏感性分析 —— 报告中单独成组，不与上面混列。
#:   han_l1_wide                   候选范围的敏感性
#:   saplma_mlp_raw / han_xgboost_raw  **打分口径**的敏感性（原始分数 vs 裁剪概率）
POSTHOC_SENSITIVITY: tuple[str, ...] = (
    "han_l1_wide",                                    # 候选范围
    "saplma_mlp_raw", "han_xgboost_raw",              # 打分口径（原始分数）
    "han_l1_authorinput", "han_xgboost_authorinput",  # 输入口径（作者原始激活）
    "han_l1_authorfixed", "han_xgboost_authorfixed",  # 原文固定配置（不调参）
)
assert set(PRESPECIFIED) | set(POSTHOC_SENSITIVITY) == set(CONDITION_BY_NAME)


# ══════════════════════════════════════════════════════════════════════════
# 读出器
# ══════════════════════════════════════════════════════════════════════════
def _to_log_odds(p: np.ndarray) -> np.ndarray:
    """概率 → log-odds，**事前固定**的裁剪。训练与测试同一条规则。"""
    p = np.clip(np.asarray(p, dtype=float).ravel(), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1.0 - p))


def _pos_proba(clf, X) -> np.ndarray:
    """取 `classes_ == 1` 那一列。**不做 1-p 反转** —— y 本就以 1 表示要检出的类。"""
    classes = list(np.asarray(clf.classes_).ravel())
    if 1 not in classes:
        raise RuntimeError(f"classes_={classes} 里没有 1 —— 正类方向无法确定")
    return np.asarray(clf.predict_proba(X))[:, classes.index(1)]


def _decision_log_odds(clf, X) -> np.ndarray:
    """logreg：decision_function 已是 class_[1] 的 log-odds（M8），不再换算。"""
    if list(np.asarray(clf.classes_).ravel()) != [0, 1]:
        raise RuntimeError(f"classes_={clf.classes_} 非 [0,1]，decision_function 方向不可假定")
    return np.asarray(clf.decision_function(X), dtype=float).ravel()


def _mlp_logit(clf, X) -> np.ndarray:
    """MLP 的 **sigmoid 前**输出（二分类输出层只有 1 个单元）。

    sklearn 的 `MLPClassifier` 不暴露 pre-activation，所以手写前向：
    隐层 ReLU，输出层只做线性、**不施加** logistic。
    这不是"近似"——用的就是 `coefs_` / `intercepts_` 本身，
    正确性由 `sigmoid(logit)` 与 `predict_proba` 的逐元素比对当场验证。
    """
    if clf.out_activation_ != "logistic":
        raise RuntimeError(f"输出层激活是 {clf.out_activation_!r}，不是 logistic —— "
                           f"sigmoid 前输出的取法不成立")
    a = np.asarray(X, dtype=float)
    n_layers = len(clf.coefs_)
    for i in range(n_layers - 1):
        a = a @ clf.coefs_[i] + clf.intercepts_[i]
        if clf.activation == "relu":
            a = np.maximum(a, 0.0)
        elif clf.activation == "tanh":
            a = np.tanh(a)
        elif clf.activation == "logistic":
            a = 1.0 / (1.0 + np.exp(-a))
        elif clf.activation != "identity":
            raise RuntimeError(f"未支持的隐层激活 {clf.activation!r}")
    z = a @ clf.coefs_[-1] + clf.intercepts_[-1]
    return np.asarray(z, dtype=float).ravel()


def verify_mlp_logit(clf, X, tol: float = 1e-9) -> dict:
    """验证 `sigmoid(logit)` 与 `predict_proba` 的正类列逐元素一致。"""
    z = _mlp_logit(clf, X)
    p_from_z = 1.0 / (1.0 + np.exp(-z))
    p_direct = _pos_proba(clf, X)
    d = float(np.max(np.abs(p_from_z - p_direct))) if z.size else 0.0
    return {"max_abs_diff": d, "ok": d <= tol, "tol": tol, "n": int(z.size),
             "logit_min": float(z.min()) if z.size else None,
             "logit_max": float(z.max()) if z.size else None}


def _xgb_margin(clf, X) -> np.ndarray:
    """XGBoost 的**原始 margin**（未过 logistic）。"""
    return np.asarray(clf.predict(X, output_margin=True), dtype=float).ravel()


def _rows_sha(rows) -> str:
    import hashlib
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()
    ).hexdigest()[:16]


def _fit_score(kind: str, params: dict, X_tr, y_tr, X_te, *,
               seed: int, main_lr_kwargs: Callable[[float], dict] | None = None,
               score_mode: str = "clipped_logodds",
               ctx: dict | None = None) -> tuple[np.ndarray, dict]:
    """拟合一个候选，返回 **(喂给融合层的分数, 完整拟合记录)**。

    记录里同时带三路分数，**互不反推**：
        info["raw"]     模型原始分数（XGB margin / MLP sigmoid 前 / logreg decision）
        info["proba"]   预测概率（正类列；logreg 不产出，为 None）
        返回值           实际喂给融合层的分数，由 `score_mode` 决定

    `ctx` 里带阶段与划分标识（repeat / fold / inner_fold / 行集合指纹），
    原样写进记录 —— **每次实际拟合都要能被定位到哪一折的哪一步**。
    """
    t0 = time.perf_counter()
    if score_mode not in ("clipped_logodds", "raw_margin"):
        raise ValueError(f"未知 score_mode {score_mode!r}")
    info: dict[str, Any] = {
        "kind": kind, "params": _jsonable(params), "seed": seed,
        "score_mode": score_mode,
        "n_train": int(np.shape(X_tr)[0]), "n_test": int(np.shape(X_te)[0]),
        "n_train_positive": int(np.sum(np.asarray(y_tr) == 1)),
        "converged": None, "n_iter": None, "max_iter_used": None,
        "max_iter_escalated": False, "convergence_warnings": [],
        **(dict(ctx) if ctx else {}),
    }

    if kind in ("logreg_main", "logreg_author"):
        if kind == "logreg_main":
            if main_lr_kwargs is None:
                raise ValueError("logreg_main 必须传主路径的 base_lr_kwargs")
            kw = dict(main_lr_kwargs(params["C"]))
        else:
            kw = {"penalty": "l1", "solver": "liblinear", "C": params["C"],
                  "max_iter": MAX_ITER_LADDER[0], "random_state": 42}
        clf, escalated = None, False
        # **每次尝试都留档**：旧写法每轮 `info.update(...)` 覆盖前一次，
        # 跑完只剩最后一次的预算/告警，前面几档的失败记录全部消失。
        attempts: list[dict[str, Any]] = []
        for mi in ((kw["max_iter"],)
                   + tuple(m for m in MAX_ITER_LADDER if m > kw["max_iter"])):
            ta = time.perf_counter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf = LogisticRegression(**{**kw, "max_iter": mi})
                clf.fit(X_tr, y_tr)
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, ConvergenceWarning)]
            nit = np.asarray(getattr(clf, "n_iter_", [np.nan])).ravel()
            attempts.append({
                "attempt": len(attempts), "max_iter": int(mi),
                "is_author_default": mi == MAX_ITER_LADDER[0],
                "n_iter": float(nit[0]), "converged": not msgs,
                "convergence_warnings": msgs[:3],
                "secs": round(time.perf_counter() - ta, 3)})
            info.update(max_iter_used=mi, converged=not msgs,
                        max_iter_escalated=escalated,
                        convergence_warnings=msgs[:3])
            if not msgs:
                break
            escalated = True
        info["escalation_attempts"] = attempts
        info["n_escalation_attempts"] = len(attempts)
        info["ladder_exhausted"] = bool(attempts and not attempts[-1]["converged"])
        raw = _decision_log_odds(clf, X_te)     # 本身就是原始 log-odds，从不裁剪
        proba = None
        fusion = raw                            # 两种 score_mode 下都一样
        n_it = np.asarray(getattr(clf, "n_iter_", [np.nan])).ravel()
        info["n_iter"] = float(n_it[0])
        info["n_nonzero_coef"] = int(np.sum(np.asarray(clf.coef_) != 0))

    elif kind == "mlp":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"],
                                activation="relu", solver="adam",
                                alpha=params["alpha"], max_iter=params["max_iter"],
                                early_stopping=False, shuffle=True,
                                random_state=seed)
            clf.fit(X_tr, y_tr)
        msgs = [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]
        # `max_iter` 是**上限**；实际训练轮数看 n_iter_
        info.update(converged=not msgs, n_iter=int(getattr(clf, "n_iter_", -1)),
                    max_iter_used=params["max_iter"],
                    reached_max_iter=int(getattr(clf, "n_iter_", -1)) >= params["max_iter"],
                    convergence_warnings=msgs[:3],
                    best_loss=float(getattr(clf, "best_loss_", float("nan"))))
        raw = _mlp_logit(clf, X_te)
        proba = _pos_proba(clf, X_te)
        info["logit_proba_check"] = verify_mlp_logit(clf, X_te)
        if not info["logit_proba_check"]["ok"]:
            raise RuntimeError(f"sigmoid(logit) 与 predict_proba 不一致："
                               f"{info['logit_proba_check']}")
        fusion = raw if score_mode == "raw_margin" else _to_log_odds(proba)

    elif kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            objective="binary:logistic", eval_metric="auc",
            random_state=42, n_jobs=1, verbosity=0)
        clf.fit(X_tr, y_tr)
        # ⚠️ XGBoost 跑满预设树数**只说明预算用完**，没有任何收敛判据被检验
        # （无 early_stopping、无 eval_set、无梯度范数阈值）。旧写法直接写
        # `converged=True`，会让它在汇总里与 L1/MLP 的"通过收敛判据"混为一谈。
        info.update(converged=None, n_iter=int(params["n_estimators"]),
                    max_iter_used=int(params["n_estimators"]),
                    budget_completed=True,
                    convergence_criterion="not_applicable_fixed_budget",
                    convergence_note=("跑满 n_estimators 即预算完成；"
                                      "本条件**未做**任何收敛性检验，"
                                      "不得与 converged=True 并列统计"))
        raw = _xgb_margin(clf, X_te)
        proba = _pos_proba(clf, X_te)
        # 一致性：sigmoid(margin) 应当等于 predict_proba（binary:logistic）
        pm = 1.0 / (1.0 + np.exp(-raw))
        info["margin_proba_check"] = {
            "max_abs_diff": float(np.max(np.abs(pm - proba))) if raw.size else 0.0}
        fusion = raw if score_mode == "raw_margin" else _to_log_odds(proba)
    else:
        raise ValueError(f"unknown readout kind {kind!r}")

    fusion = np.asarray(fusion, dtype=float).ravel()
    if not np.all(np.isfinite(fusion)):
        raise RuntimeError(f"{kind}/{params}: 打分出现非有限值 —— 不静默丢样本")

    # 裁剪到达率：旧口径下有多少测试样本被 PROB_EPS 夹住
    if proba is not None:
        at = int(np.sum((proba <= PROB_EPS) | (proba >= 1.0 - PROB_EPS)))
        info["clip_boundary_hits"] = at
        info["clip_boundary_rate"] = round(at / max(proba.size, 1), 6)
        info["proba_min"] = float(proba.min())
        info["proba_max"] = float(proba.max())
    info["raw"] = np.asarray(raw, dtype=float).ravel()
    info["proba"] = None if proba is None else np.asarray(proba, dtype=float).ravel()
    info["raw_min"] = float(info["raw"].min()) if info["raw"].size else None
    info["raw_max"] = float(info["raw"].max()) if info["raw"].size else None
    # **正式路径实际构造出来的估计器参数**原样留档。
    # 与作者实现的一致性检查读的就是这里 —— 测试不再自己手搓一个分类器来比，
    # 否则正式路径改错了测试照样通过。
    info["estimator_params"] = _jsonable(clf.get_params())
    info["estimator_class"] = type(clf).__name__
    info["secs"] = round(time.perf_counter() - t0, 3)
    return fusion, info


# ══════════════════════════════════════════════════════════════════════════
# 训练折内选参（与主路径同构）
# ══════════════════════════════════════════════════════════════════════════
def select_params(cond: Condition, design, rows: np.ndarray, y: np.ndarray,
                  groups: np.ndarray, *, n_inner: int, fold_seed: int,
                  max_retries: int, seed: int, main_lr_kwargs=None,
                  ctx: dict | None = None,
                  fit_log: list | None = None) -> tuple[dict, dict]:
    """在给定**训练行集合**内用 grouped inner CV 选超参。

    候选评分**从不**碰该阶段的验证折或外层测试折 —— 只在 `rows` 内部再切。
    """
    try:
        inner = ci.stratified_grouped_folds(
            y[rows], groups[rows], n_splits=n_inner, seed=fold_seed,
            max_seed_retries=max_retries)
    except (RuntimeError, ValueError) as exc:
        return dict(cond.candidates[0]), {
            "fell_back": True, "why": f"{type(exc).__name__}: {exc}",
            "chosen": _jsonable(cond.candidates[0]), "curve": [], "at_boundary": None}

    curve, best, best_auc, best_idx = [], None, -np.inf, 0
    for idx, params in enumerate(cond.candidates):
        aucs, infos = [], []
        for j, (itr, ite) in enumerate(inner):
            gtr, gte = rows[itr], rows[ite]
            X_tr, X_te = design(gtr, gte)
            s, info = _fit_score(
                cond.kind, params, X_tr, y[gtr], X_te, seed=seed,
                main_lr_kwargs=main_lr_kwargs, score_mode=cond.score_mode,
                ctx={**(ctx or {}), "stage": "candidate_validation",
                     "condition": cond.name, "candidate_index": idx,
                     "selection_inner_fold": j,
                     "train_rows_sha": _rows_sha(gtr), "test_rows_sha": _rows_sha(gte)})
            if fit_log is not None:      # **每次实际拟合都留档**，不只留曲线汇总
                fit_log.append({k: v for k, v in info.items()
                                if k not in ("raw", "proba")})
            aucs.append(ci.auroc(s, y[gte]))
            infos.append(info)
        m = float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else -np.inf
        curve.append({"index": idx, "params": _jsonable(params),
                      "mean_inner_auroc": None if not np.isfinite(m) else round(m, 6),
                      "per_fold_auroc": [round(float(a), 6) for a in aucs],
                      # None（XGB）= 未做收敛检验，**不计入** converged_all，
                      # 另立 n_no_criterion 计数，避免"没检验"被读成"通过了"
                      "converged_all": (None if all(i["converged"] is None for i in infos)
                                        else all(i["converged"] is not False
                                                 for i in infos
                                                 if i["converged"] is not None)),
                      "n_fits": len(infos),
                      "n_not_converged": sum(1 for i in infos if i["converged"] is False),
                      "n_no_criterion": sum(1 for i in infos if i["converged"] is None),
                      "n_escalated_fits": sum(1 for i in infos
                                              if i.get("max_iter_escalated")),
                      "max_iter_used_per_fold": [i.get("max_iter_used") for i in infos],
                      "max_iter_escalated": any(i.get("max_iter_escalated") for i in infos),
                      "secs": round(sum(i["secs"] for i in infos), 2)})
        if m > best_auc:                       # 严格大于 → 并列保留先出现者（更简单的）
            best_auc, best, best_idx = m, dict(params), idx
    return best, {
        "fell_back": False, "chosen": _jsonable(best), "chosen_index": best_idx,
        "n_candidates": len(cond.candidates),
        "at_boundary": best_idx in (0, len(cond.candidates) - 1),
        "which_boundary": ("lower" if best_idx == 0 else
                           "upper" if best_idx == len(cond.candidates) - 1 else None),
        "selection_metric": SELECTION_METRIC, "tie_break": TIE_BREAK, "curve": curve,
    }


# ══════════════════════════════════════════════════════════════════════════
def verify_positive_class_direction(seed: int = 0) -> dict:
    """小规模验证正类方向：**y=1 是要检出的类，不做 1-p 反转**。

    造一个 x 与 y 正相关的玩具集：正确实现下，y=1 的样本分数应当**更高**，
    AUROC 应当 > 0.5。若某处多做了一次反转，这里会变成 < 0.5。
    """
    rng = np.random.default_rng(seed)
    n = 240
    y = (rng.random(n) > 0.5).astype(int)
    X = rng.normal(size=(n, 12)) + 1.5 * y[:, None]      # y=1 → 特征整体更大
    out = {}
    for kind, params in (("logreg_author", {"C": 1.0}),
                         ("mlp", {"hidden_layer_sizes": (16,), "alpha": 1e-3,
                                  "max_iter": 200}),
                         ("xgb", {"max_depth": 3, "n_estimators": 50,
                                  "learning_rate": 0.1})):
        s, _ = _fit_score(kind, params, X, y, X, seed=seed)
        out[kind] = {"auroc_in_sample": round(float(ci.auroc(s, y)), 4),
                     "mean_score_y1": round(float(s[y == 1].mean()), 3),
                     "mean_score_y0": round(float(s[y == 0].mean()), 3),
                     "direction_ok": bool(ci.auroc(s, y) > 0.5
                                          and s[y == 1].mean() > s[y == 0].mean())}
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o
