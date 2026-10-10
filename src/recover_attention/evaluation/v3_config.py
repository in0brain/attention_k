"""Stage 1 / preregistration v3 的分析侧冻结参数（M8 的单一来源）。

**本模块不属于 Stage 0。** Stage 0 的设计权威仍是 `preregistration.md` v2.4，
其实现（`conditional_increment.py` / `evaluation/*`）本次一行未改。

M8 的继承项由 Stage 0 现行实现与正式产物只读提取：
  n_outer_folds=5 / n_inner_folds=3
      outputs/stage0/stage0_production_preflight.json
        :: code_readiness.production_config_valid
      outputs/stage0/evaluation/stage0_analysis_report.json :: config
  C_GRID / base learner 规格
      conditional_increment.py::{C_GRID, block_oof_scores, _select_C}
  折构造与重试
      conditional_increment.py::stratified_grouped_folds

🆕 v3 新增或收紧（M8 差异表）：
  base random_state=0（Stage 0 未设，属可复现性缺口）
  meta 层全部参数逐项冻结，C 固定 1.0 不搜索
  meta 输入用 decision_function（Stage 0 的 _risk_scores 优先返回 predict_proba）
  C 回退必须记录 repeat/outer_fold/block/phase/inner_fold
  shared block 在 S* 与 S*+H 之间必须复用同一个 BaseBlockCache
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recover_attention import conditional_increment as ci

# ── 记号（preregistration_v3 §3 记号约定）────────────────────────────────
#   r = repeat, k = comparison/rung/block, q = bootstrap draw, j = MC batch

BLOCK_NAMES_S_STAR: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
BLOCK_NAME_H: str = "H"
BLOCK_NAMES_S_STAR_PLUS_H: tuple[str, ...] = BLOCK_NAMES_S_STAR + (BLOCK_NAME_H,)

# MC 误差带用 t 分位而非正态分位：B_batch=20 → df=19（M7）
T_QUANTILE_975_DF19: float = 2.093

PHASE_OUTER_FULL = "outer_full"
PHASE_INNER_CF = "inner_cf"


@dataclass(frozen=True)
class V3AnalysisConfig:
    """M8 的全部冻结参数。R 与 B_per_batch 由 §7.3 的收敛研究确定，此处只给类型与默认。"""

    # ---- 折（来源：Stage 0 正式产物，见模块 docstring）----
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    outer_max_seed_retries: int = 20
    inner_max_seed_retries: int = 10

    # ---- base learner（Stage 0 规格 + 🆕 random_state）----
    base_penalty: str = "l2"
    base_solver: str = "liblinear"
    base_max_iter: int = 2000
    base_tol: float = 1e-4
    base_fit_intercept: bool = True
    base_class_weight: object = None
    base_random_state: int = 0

    # ---- C 选择（Stage 0 行为，tie-break 由巧合升为契约）----
    c_grid: tuple[float, ...] = ci.C_GRID
    c_fallback: float = 1.0

    # ---- meta learner（🆕 全部逐项冻结；C 固定不搜索）----
    meta_C: float = 1.0
    meta_penalty: str = "l2"
    meta_solver: str = "lbfgs"
    meta_max_iter: int = 5000
    meta_tol: float = 1e-6
    meta_fit_intercept: bool = True
    meta_class_weight: object = None
    meta_random_state: int = 0

    # ---- 判读常量（与 v2.4 一致，v3 §8「不变」清单）----
    eps_auroc: float = ci.EPS_AUROC
    eps_ni: float = 0.02
    precision_max_ci_width: float = 2 * ci.EPS_AUROC
    alpha: float = 0.05

    # ---- 重复交叉拟合与 bootstrap（M7）----
    n_repeats: int = 3              # R —— §7.3 收敛研究确定（2026-08-07）
    b_per_batch: int = 1000         # B_per_batch —— 同上；B_total = 20 × 1000 = 20,000
    b_batch: int = 20               # 冻结，仅诊断分批数
    fold_seed_base: int = 20260804
    fold_seed_step: int = 100       # fold-seed = base + step·r（前缀嵌套）
    boot_seed_base: int = 20260804
    boot_seed_step: int = 7         # bootstrap-seed = base + step·j

    # R / B_per_batch 若用 §7.3 的回退值，必须**显式声明**，不得让默认值冒充收敛研究结果
    r_b_source: str = "convergence_study"   # §7.3 已完成，回退分支已不适用
    fallback_declared: bool = False

    # ---- E3 词表分块（M2d，2026-08-08 冻结）----
    # 块本身充当 outer fold，故 F 必须等于 n_outer_folds。salt 一经冻结不得更换：
    # 换 salt 等于换一次分折，反复换到"好看"的分法就是 look-elsewhere。
    e3_n_blocks: int = 5                     # F
    e3_block_salt: str = "recover_attention/v3/E3_vocab_block/2026-08-08"

    # ---- MCQ 选题（M6，2026-08-08 冻结）----
    mcq_selection_salt: str = "recover_attention/v3/mcq_selection/2026-08-08"

    # ---- G3 / G4 gate 样本（**执行级常量**，2026-08-08 冻结，不 bump v3.0）----
    # 正式池取满后已无 fresh recall 余量，故 H1 的 gate 样本从 Stage 0 **已烧过**的
    # recall 条目中确定性选取；MCQ 从冻结排序中正式集之后的名次取。
    # 三套名单（正式 / G3 / G4）必须机检零交集，gate 产物标 gate_only=true。
    h1_gate_selection_salt: str = "recover_attention/v3/h1_gate_selection/2026-08-08"
    n_g3_per_arm: int = 20          # 每臂 G3 样本数（H1 受 v3 §4 约束 ≤20 prompt）
    n_g4_per_arm: int = 40          # 每臂 G4 样本数
    h1_gate_families: tuple[str, ...] = ("attack", "cwe")   # G3 各 10、G4 各 20

    # ---- 生成侧执行级常量（2026-08-08 冻结，不 bump v3.1）----
    # 两者都决定 prompt 或采样，故必须冻结并写进 provenance；但都不触及 estimand、
    # 判读规则或阈值，故按执行级常量处理（与两个 salt 同类）。
    mcq_shuffle_seed: int = 42      # 选项顺序：沿用 Stage 0，保持选项位置分布可比
    generation_seed: int = 4242     # 采样种子：沿用 Stage 0，保持协议同构
    # max_new_tokens **按冻结设计分臂**（H1 自由生成 / MCQ 只出一个字母），
    # 沿用 Stage 0 的两份 run_fingerprint。它不是 backend invariant 的一部分，
    # 但两个值各自必须精确匹配 —— 只有 (384, 16) 这一个合法组合。
    h1_max_new_tokens: int = 384
    mcq_max_new_tokens: int = 16

    # ---- Stage 1 population（M6 / §7.5，2026-08-08 冻结）----
    # N_H1 是**容量上界决策**（未被 Stage 0 消耗的全部 attack+cwe 条目），不是精度规划。
    h1_routes: tuple[str, ...] = ("recall",)          # **不含 open_gen**（§11.9）
    h1_families: tuple[str, ...] = ("attack", "cwe")  # CVE 不进 primary（效度约束）
    # eligible pool 1547 −2（跨 Stage prompt 碰撞）−1（正式集内同 prompt 去重）= 1544（v3.1 §11.10）
    n_h1_eligible_pool: int = 1547
    n_h1_prompts: int = 1544
    n_mcq_questions: int = 1760
    k_traces: int = 6

    # 审计：每批逐条留存的 draw digest 条数（其余用滚动链哈希，保证 O(1) 内存）
    n_audit_digests_per_batch: int = 8

    # 允许 dry-run 规格（合成数据测试用），production 由 assert_production 精确强制
    mode: str = "production"

    @property
    def b_total(self) -> int:
        return self.b_batch * self.b_per_batch

    def fold_seed(self, repeat: int) -> int:
        """**唯一**的 split-RNG 来源（§7.3 冻结）。

        ⚠️ 与 `base_random_state` / `meta_random_state` 严格分离：后两者是
        LogisticRegression 的模型 RNG（M8），**不得**用于任何 CV splitter。
        nested C 选择的 inner split 也用本函数，不引入额外偏移公式。
        """
        return self.fold_seed_base + self.fold_seed_step * int(repeat)

    def batch_seed(self, batch: int) -> int:
        return self.boot_seed_base + self.boot_seed_step * int(batch)

    def base_lr_kwargs(self, C: float) -> dict:
        return {"C": float(C), "penalty": self.base_penalty, "solver": self.base_solver,
                "max_iter": self.base_max_iter, "tol": self.base_tol,
                "fit_intercept": self.base_fit_intercept,
                "class_weight": self.base_class_weight,
                "random_state": self.base_random_state}

    def meta_lr_kwargs(self) -> dict:
        return {"C": self.meta_C, "penalty": self.meta_penalty, "solver": self.meta_solver,
                "max_iter": self.meta_max_iter, "tol": self.meta_tol,
                "fit_intercept": self.meta_fit_intercept,
                "class_weight": self.meta_class_weight,
                "random_state": self.meta_random_state}

    def assert_production(self) -> None:
        """production 下**逐项精确**核对全部 M8 / 冻结常量。

        与第 1 版的区别：不再只查下限。折数必须**恰为** 5/3（预注册值），
        base/meta learner 的每个参数、eps/alpha/b_batch/seed 规则全部核对。
        本函数由 run_repeat / run_all_repeats / bootstrap 入口强制调用，无法绕过。
        """
        if self.mode not in ("production", "dry_run"):
            raise ValueError(f"mode must be production|dry_run, got {self.mode!r}")
        if self.mode != "production":
            return

        for field_name, expected in FROZEN_FIELDS.items():
            actual = getattr(self, field_name)
            if isinstance(expected, tuple):
                ok = tuple(actual) == expected
            else:
                ok = actual == expected and type(actual) is type(expected)
            if not ok:
                raise ValueError(
                    f"frozen field {field_name!r} was overridden: expected {expected!r} "
                    f"({type(expected).__name__}), got {actual!r} "
                    f"({type(actual).__name__}). M8 的冻结参数不得覆盖。")

        # r_b_source / fallback_declared 已进 FROZEN_FIELDS（§7.3 于 2026-08-07 完成），
        # 下面两条保留为语义护栏，给出比"frozen field"更具体的报错。
        if self.r_b_source not in ("convergence_study", "fallback"):
            raise ValueError(
                "r_b_source must be 'convergence_study' or 'fallback'; "
                "默认值不得冒充收敛研究确定值")
        if self.r_b_source == "fallback":
            raise ValueError(
                "§7.3 的收敛研究已于 2026-08-07 完成（R=3, B_per_batch=1000，产物 "
                "outputs/logs/stage0_development_convergence_R_Bperbatch.json），"
                "回退分支已不适用；production 下 r_b_source 必须是 'convergence_study'")
        if self.fallback_declared:
            raise ValueError("fallback_declared=True contradicts a completed "
                             "convergence study")

        self._check_semantics()

    def _check_semantics(self) -> None:
        """冻结常量之间的**关系**护栏（§11.9）。

        逐项比对只能发现"某个字段被改了"；这里发现的是"两个字段被一起改成
        自洽但错误的组合"。三条护栏对应的都是冻结的**理由**本身。
        """
        if self.e3_n_blocks != self.n_outer_folds:
            raise ValueError(
                f"F (e3_n_blocks={self.e3_n_blocks}) 必须等于 n_outer_folds="
                f"{self.n_outer_folds}：E3 的词表块**本身充当 outer fold**（M2d）")
        if "open_gen" in self.h1_routes:
            raise ValueError(
                "H1 primary population 为 **recall only**（§11.9）：open_gen 的抽样单位是 "
                "topic 而非 ontology entry，加入会改变 N_H1=1547 的容量上界定义")
        if self.n_h1_prompts * self.k_traces != 9264:
            raise ValueError(
                f"H1 traces 必须为 9,264（1544 × 6），得到 "
                f"{self.n_h1_prompts} × {self.k_traces}")
        if self.n_h1_eligible_pool - self.n_h1_prompts != 3:
            raise ValueError(
                f"eligible pool 与正式集之差必须恰为 3（2 条跨 Stage prompt 碰撞 + "
                f"1 条正式集内同 prompt 去重，v3.1 §11.10），得到 "
                f"{self.n_h1_eligible_pool} − {self.n_h1_prompts}")

    def provenance(self) -> dict:
        """写入产物的超参来源记录。"""
        return {"n_repeats": self.n_repeats, "b_per_batch": self.b_per_batch,
                "b_batch": self.b_batch, "b_total": self.b_total,
                "r_b_source": self.r_b_source,
                "fallback_declared": bool(self.fallback_declared),
                "fold_seed_rule": f"{self.fold_seed_base} + {self.fold_seed_step}*r",
                "boot_seed_rule": f"{self.boot_seed_base} + {self.boot_seed_step}*j",
                "e3_n_blocks": self.e3_n_blocks,
                "e3_block_salt": self.e3_block_salt,
                "mcq_selection_salt": self.mcq_selection_salt,
                "h1_routes": list(self.h1_routes),
                "h1_families": list(self.h1_families),
                "n_h1_eligible_pool": self.n_h1_eligible_pool,
                "n_h1_prompts": self.n_h1_prompts,
                "n_mcq_questions": self.n_mcq_questions,
                "k_traces": self.k_traces,
                "h1_traces": self.n_h1_prompts * self.k_traces,
                "mcq_traces": self.n_mcq_questions * self.k_traces,
                "h1_gate_selection_salt": self.h1_gate_selection_salt,
                "n_g3_per_arm": self.n_g3_per_arm,
                "n_g4_per_arm": self.n_g4_per_arm,
                "mcq_shuffle_seed": self.mcq_shuffle_seed,
                "generation_seed": self.generation_seed,
                "max_new_tokens_by_arm": {"h1": self.h1_max_new_tokens,
                                          "mcq": self.mcq_max_new_tokens}}


# 逐项精确冻结的字段（M8 + preregistration_v3 §8「不变」清单 + §7.4）。
# 任一字段被改动 → assert_production 抛错。
# 🆕 2026-08-07：R / B_per_batch / r_b_source / fallback_declared 由 §7.3 收敛研究
#    确定后一并冻结（见 preregistration_v3 §7.4），回退分支自此不再适用。
FROZEN_FIELDS: dict[str, object] = {
    "n_repeats": 3,
    "b_per_batch": 1000,
    "r_b_source": "convergence_study",
    "fallback_declared": False,
    "n_outer_folds": 5,
    "n_inner_folds": 3,
    "outer_max_seed_retries": 20,
    "inner_max_seed_retries": 10,
    "base_penalty": "l2",
    "base_solver": "liblinear",
    "base_max_iter": 2000,
    "base_tol": 1e-4,
    "base_fit_intercept": True,
    "base_class_weight": None,
    "base_random_state": 0,
    "c_grid": tuple(ci.C_GRID),
    "c_fallback": 1.0,
    "meta_C": 1.0,
    "meta_penalty": "l2",
    "meta_solver": "lbfgs",
    "meta_max_iter": 5000,
    "meta_tol": 1e-6,
    "meta_fit_intercept": True,
    "meta_class_weight": None,
    "meta_random_state": 0,
    "eps_auroc": ci.EPS_AUROC,
    "eps_ni": 0.02,
    "precision_max_ci_width": 2 * ci.EPS_AUROC,
    "alpha": 0.05,
    "b_batch": 20,
    "fold_seed_base": 20260804,
    "fold_seed_step": 100,
    "boot_seed_base": 20260804,
    "boot_seed_step": 7,
    # 🆕 2026-08-08 冻结轮：F / 两个 salt / Stage 1 population 由散文升为代码契约
    # （preregistration_v3 §M2d / §M6 / §7.5 / §11.9）。与 R/B 同等强制，不得覆盖。
    "e3_n_blocks": 5,
    "e3_block_salt": "recover_attention/v3/E3_vocab_block/2026-08-08",
    "mcq_selection_salt": "recover_attention/v3/mcq_selection/2026-08-08",
    "h1_gate_selection_salt": "recover_attention/v3/h1_gate_selection/2026-08-08",
    "n_g3_per_arm": 20,
    "n_g4_per_arm": 40,
    "h1_gate_families": ("attack", "cwe"),
    "mcq_shuffle_seed": 42,
    "generation_seed": 4242,
    "h1_max_new_tokens": 384,
    "mcq_max_new_tokens": 16,
    "h1_routes": ("recall",),
    "h1_families": ("attack", "cwe"),
    "n_h1_eligible_pool": 1547,
    "n_h1_prompts": 1544,
    "n_mcq_questions": 1760,
    "k_traces": 6,
}
