"""Stage 1 的 G3 smoke（v3.1）。每次只跑一臂。

    python scripts/run_stage1_g3_smoke.py --arm h1  --dry-run    # 假后端，不调模型
    python scripts/run_stage1_g3_smoke.py --arm h1               # 真模型
    python scripts/run_stage1_g3_smoke.py --arm mcq
    python scripts/run_stage1_g3_smoke.py --summarise            # 两臂 backend invariant 汇总

判据在 `v3_g3_smoke.CRITERIA` 顶部写死，**运行前即已固定**；任一条件失败 →
`passed=false` 并停止，不修改 prompt / seed / 阈值 / 生成参数后静默重试。

**不计算任何性能量**（AUROC / ladder / probe / CI 一律不算，见报告的 did_not_compute）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation.manifest import write_atomic  # noqa: E402


# ── 假后端：验 resume / 指纹 / 产物契约，不调模型 ────────────────────────────
class _FakeBackend:
    """dry-run 用。产出 schema 完整但**内容合成**的行；绝不加载模型。"""

    def __init__(self, arm: str, cfg: V3AnalysisConfig, hidden_dim: int | None = None,
                 geometry=None, max_new_tokens: int | None = None):
        self.arm = arm
        self.cfg = cfg
        # 几何来自显式对象；`hidden_dim=` 仍受支持以兼容既有调用方。
        self.geometry = geometry or repool.QWEN_GEOMETRY
        self.hidden_dim = int(hidden_dim if hidden_dim is not None
                              else self.geometry.hidden_dim)
        # 解码预算按臂取**本次运行实际生效**的值（冻结默认 H1=384 / MCQ=16，
        # 可被 CLI 覆盖）。先前这里写死 384，于是 MCQ 臂的模拟指纹恒与冻结值不符，
        # dry 汇总每次都报 generation_params.max_new_tokens[mcq] —— 那是**假后端自己
        # 造出来的**不一致，真实生成路径从来是用同一个参数写指纹并调用生成。
        self.max_new_tokens = int(
            max_new_tokens if max_new_tokens is not None
            else (cfg.h1_max_new_tokens if arm == g3.ARM_H1 else cfg.mcq_max_new_tokens))

    def run_fingerprint(self, arm: str) -> dict:
        # 这个假后端是**按臂构造**的。若调用方传进来另一个臂，返回的解码预算就会张冠李戴
        # —— 正是本次修复的那类缺陷，所以在这里直接拒绝而不是悄悄返回。
        if arm != self.arm:
            raise ValueError(
                f"_FakeBackend 按臂构造为 {self.arm!r}，不能为 {arm!r} 出指纹")
        geo = self.geometry
        return {
            "schema": "stage0_fingerprint_v1", "arm": arm,
            "model": {"model_path": "<dry-run>", "model_type": "qwen2",
                      "num_hidden_layers": geo.num_layers,
                      "hidden_size": self.hidden_dim,
                      "backend": dict(g3.gb.BACKEND_8BIT), "model_hash": "dry" * 16},
            "tokenizer": {"class": "Qwen2Tokenizer", "tokenizer_hash": "dry" * 16},
            "hidden_index": {"num_layers": geo.num_layers, "rel_depth": geo.rel_depth,
                             "block_index_zero_based": geo.block_index,
                             "hidden_states_tuple_index": geo.tuple_index},
            "generation_params": {"K": self.cfg.k_traces,
                                  "max_new_tokens": self.max_new_tokens,
                                  "temperature": 0.7, "top_p": 0.95,
                                  "top_k": g3.gb._RealBackendBase.QWEN_PROTOCOL_TOP_K,
                                  "repetition_penalty":
                                      g3.gb._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY,
                                  "seed": self.cfg.generation_seed,
                                  # 与真后端同口径，dry 贯通才检得出新旧混用
                                  "schema": g3.GENERATION_PARAMS_SCHEMA,
                                  "sampling_params_source": "dry-run stub"},
            "run_hash": f"dryrun_{arm}",
        }

    def verify_hidden_index(self) -> dict:
        geo = self.geometry
        return {"num_layers": geo.num_layers, "rel_depth": geo.rel_depth,
                "block_index_zero_based": geo.block_index,
                "hidden_states_tuple_index": geo.tuple_index,
                "numeric_match": True, "max_abs_diff": 0.0}

    def run_unit(self, unit: dict, record: dict) -> dict:
        rng = np.random.default_rng(abs(hash(unit["unit_id"])) % (2 ** 31))
        base = {
            "example_id": record["example_id"],
            "prompt_hash": f"dry_{unit['unit_id']}",
            "unit_hash": f"dry_{unit['unit_id']}_u",
            "seed": self.cfg.generation_seed,
            "sample_index": unit["sample_index"],
            "hidden": rng.normal(0, 1, self.hidden_dim).astype(np.float32),
        }
        if self.arm == g3.ARM_H1:
            base.update({
                "family": record.get("family"), "route": record.get("route"),
                "completion": "T1059 is the technique.", "refusal": False,
                "mentions": [{"normalized": "T1059", "start": 0, "end": 5,
                              "emitted": True, "token_indices": [0]}],
                "id_token_positions_local": [0], "hidden_pooled_n_tokens": 1,
                "id_string_text": "T1059",
            })
        else:
            base.update({
                "parsed_label": "A", "eligible_for_primary": True,
                "f5_label_margin": 1.0, "f5_label_entropy": 0.5,
                "f5_full_entropy": 2.0, "f5_letter_token_logprob": -0.1,
            })
        return base


# ── 真后端 ───────────────────────────────────────────────────────────────────
def _real_backend(arm: str, args, cfg: V3AnalysisConfig):
    from recover_attention.generation import backends as gb
    from recover_attention.h1_identifier import build_ontology_index

    common = dict(seed=cfg.generation_seed, temperature=0.7, top_p=0.95)
    if arm == g3.ARM_H1:
        index = build_ontology_index(Path(args.ontology_dir))
        return gb.H1Backend(args.model_path, index, max_new_tokens=args.max_new_tokens,
                            **common)
    return gb.MCQBackend(args.model_path, max_new_tokens=args.mcq_max_new_tokens,
                         **common)


#: 生成长度扩展的登记文件必须带的字段（口径标记 + 这四项）。
BUDGET_REGISTRATION_SCHEMA = "stage1_generation_budget_registration_v1"


def _budget_extension(arm: str, args, cfg: V3AnalysisConfig) -> dict | None:
    """本臂实际预算 != 冻结值时，要求一份**事前登记**，否则拒绝运行。

    这不是把长度一致性检查删掉 —— 冻结值仍是唯一默认，比对仍然精确，
    只是把"与冻结值不同"从**静默可能**变成**必须事先写下来并落进产物**。
    预算等于冻结值时本函数返回 None，Qwen 与既有 Llama 路径逐字不变。

    ⚠️ 登记**不**代表这个值已被证明足够，也不代表它是最小可用预算；
    它只代表"本次用的是这个值，且事前写下了理由与边界"。
    """
    effective = (args.max_new_tokens if arm == g3.ARM_H1 else args.mcq_max_new_tokens)
    frozen = (cfg.h1_max_new_tokens if arm == g3.ARM_H1 else cfg.mcq_max_new_tokens)
    if effective == frozen:
        return None

    if args.budget_registration is None:
        raise SystemExit(
            f"[stop ] {arm} 臂的 max_new_tokens={effective} 与冻结值 {frozen} 不同，"
            f"但没有给 --budget-registration。生成长度扩展必须事前登记，"
            f"不接受只在命令行上临时改一个数字。")
    path = Path(args.budget_registration)
    # 核验由共享的 `load_budget_registration` 完成（口径 / 冻结值 / 模型 / 校准题单 /
    # **一次已通过的生成长度检查**），本入口不再自带一份平行的检查清单。
    try:
        reg = g3.load_budget_registration(path, cfg=cfg, model_path=args.model_path)
    except ValueError as exc:
        raise SystemExit(f"[stop ] {exc}") from None
    if reg.arm != arm:
        raise SystemExit(f"[stop ] 登记文件是 {reg.arm!r} 臂的，不能用于 {arm!r} 臂")
    if reg.registered_max_new_tokens != effective:
        raise SystemExit(f"[stop ] 登记的预算 {reg.registered_max_new_tokens} != "
                         f"本次实际 {effective}")
    doc = json.loads(path.read_text(encoding="utf-8"))

    return {
        "registered": True,
        "registration_path": str(path).replace("\\", "/"),
        "registration_sha256": reg.sha256,
        "arm": arm,
        "frozen_max_new_tokens": frozen,
        "effective_max_new_tokens": effective,
        "length_check": dict(reg.length_check),
        "reason": doc.get("reason"),
        "status": doc.get("status"),
        "claim_limits": doc.get("claim_limits"),
        "note": ("本次运行的解码预算与 v3_config 的冻结值不同，属**透明登记的扩展验证**；"
                 "**不是**原冻结配置下的严格同参数复现，也不得与 16-token 产物混用。"
                 "下游 backend_invariant / G4 / 生产的长度一致性检查仍按冻结值比对，"
                 "**未被删除、未被放宽** —— 它们会拒绝本指纹，这是预期行为。"),
    }


def _progress(payload: dict) -> None:
    rate = payload["elapsed_s"] / max(payload["done"], 1)
    remaining = (payload["total"] - payload["done"]) * rate
    print(f"[{payload['arm']}] {payload['done']}/{payload['total']} "
          f"elapsed={payload['elapsed_s']:.0f}s  {rate:.1f}s/trace  "
          f"eta={remaining / 60:.1f}min  failures={payload['failures']}", flush=True)


def _pace_check(seconds_for_first: int, n_first: int, total: int, budget_h: float) -> dict:
    rate = seconds_for_first / max(n_first, 1)
    projected_h = rate * total / 3600.0
    return {"observed_s_per_trace": round(rate, 2),
            "projected_total_hours": round(projected_h, 2),
            "budget_hours": budget_h, "over_budget": projected_h > budget_h}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(g3.ARMS))
    ap.add_argument("--summarise", action="store_true",
                    help="读两臂已完成的报告，产出 backend invariant 汇总")
    ap.add_argument("--dry-run", action="store_true", help="假后端；不加载模型")
    ap.add_argument("--samples-dir", type=Path, default=Path("outputs/stage1/samples"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage1/g3"))
    ap.add_argument("--prereg", type=Path, default=Path("docs/paper/preregistration_v3.md"))
    ap.add_argument("--lock", type=Path, default=Path("docs/paper/preregistration_v3.lock"))
    ap.add_argument("--ontology-dir", type=Path, default=Path("data/raw/ontology"))
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="默认取 v3_config 的冻结值（H1 384）")
    ap.add_argument("--mcq-max-new-tokens", type=int, default=None,
                    help="默认取 v3_config 的冻结值（MCQ 16）")
    ap.add_argument("--budget-registration", type=Path, default=None,
                    help="生成长度扩展的登记文件。**只要本臂的 max_new_tokens 与 "
                         "v3_config 的冻结值不同就必须给**，否则拒绝运行。")
    ap.add_argument("--pace-budget-hours", type=float, default=2.0)
    ap.add_argument("--pace-check-after", type=int, default=20,
                    help="前 N 条实测后外推；超预算即停并报告")
    repool.add_geometry_cli_args(ap)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    # 几何：三个参数全省略即 Qwen 冻结默认，行为与参数化之前一致。
    geometry = repool.geometry_from_cli(args, model_path=args.model_path)
    criteria = g3.G3Criteria(hidden_dim=geometry.hidden_dim,
                             block_index=geometry.block_index,
                             tuple_index=geometry.tuple_index)
    print(f"[geom ] L={geometry.num_layers} rel={geometry.rel_depth} "
          f"block={geometry.block_index} tuple={geometry.tuple_index} "
          f"dim={geometry.hidden_dim} source={geometry.source}")
    # 单一来源：max_new_tokens 的冻结值只在 v3_config 里，CLI 不另设默认值
    if args.max_new_tokens is None:
        args.max_new_tokens = cfg.h1_max_new_tokens
    if args.mcq_max_new_tokens is None:
        args.mcq_max_new_tokens = cfg.mcq_max_new_tokens

    if args.summarise:
        return _summarise(args, cfg, criteria)  # noqa: RET504
    if not args.arm:
        ap.error("需要 --arm 或 --summarise")

    # 预算闸：与冻结值不同必须事前登记（返回 None = 用的就是冻结值，行为不变）
    budget_extension = _budget_extension(args.arm, args, cfg)
    if budget_extension:
        print(f"[budget] {args.arm} max_new_tokens={budget_extension['effective_max_new_tokens']} "
              f"(冻结值 {budget_extension['frozen_max_new_tokens']})  "
              f"登记={budget_extension['registration_path']} "
              f"sha={budget_extension['registration_sha256'][:12]}…")

    # 1) 启动前校验（不加载模型）
    verification = g3.verify_inputs(samples_dir=args.samples_dir, prereg=args.prereg,
                                    lock=args.lock, cfg=cfg)
    g2 = verification["g2"]
    print(f"[G2   ] match={g2['match']} version={g2['locked_version']} "
          f"hash_mode={g2['hash_mode']} sha={g2['current_sha256'][:12]}…")
    print(f"[lists] manifest_sha={verification['sample_manifest_sha256'][:12]}… "
          f"chain hashes verified={verification['n_lists_verified']}/6  "
          f"g3 counts={verification['g3_counts']}  primary_read={verification['primary_lists_read']}")

    gate_rows = verification.pop("_gate_rows")[args.arm]

    # 2) 后端与记录
    if args.dry_run:
        backend = _FakeBackend(
            args.arm, cfg, geometry=geometry,
            # 与真后端同一来源：args 已在上面由 v3_config 的冻结值填好，
            # CLI 覆盖也走这里 —— 假后端不再自带一个写死的数字。
            max_new_tokens=(args.max_new_tokens if args.arm == g3.ARM_H1
                            else args.mcq_max_new_tokens))
        tokenizer = None
        print("[mode ] DRY-RUN（假后端，不加载模型）")
    else:
        backend = _real_backend(args.arm, args, cfg)
        tokenizer = backend.ctx["tokenizer"]
        print(f"[mode ] REAL  model={args.model_path}")

    if args.arm == g3.ARM_MCQ and tokenizer is None:
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda msgs, **kw: json.dumps(msgs, ensure_ascii=False))
    records = g3.build_arm_records(args.arm, gate_rows, cfg=cfg, tokenizer=tokenizer)

    # 3) 生成（前 N 条实测后外推，超预算即停）
    started = time.time()
    try:
        run = g3.run_arm(arm=args.arm, records=records, backend=backend,
                         out_dir=args.out_dir, cfg=cfg, progress=_progress,
                         progress_every=args.pace_check_after,
                         pace_check_after=args.pace_check_after,
                         pace_budget_hours=args.pace_budget_hours)
    except g3.PaceBudgetExceeded as exc:
        print(f"[STOP ] {exc}", file=sys.stderr)
        print(f"[pace ] {json.dumps(exc.pace, ensure_ascii=False)}", file=sys.stderr)
        return 3
    pace = run.pace or _pace_check(int(time.time() - started), len(run.rows) or 1,
                                   run.n_units, args.pace_budget_hours)

    # 4) 判定（只应用顶部写死的判据）
    evaluation = g3.evaluate_arm(run, n_records=len(records), criteria=criteria)
    report = g3.build_report(arm=args.arm, run=run, verification=verification,
                             evaluation=evaluation, cfg=cfg, n_records=len(records),
                             criteria=criteria)
    report["pace"] = pace
    report["dry_run"] = bool(args.dry_run)
    # None = 本次用的就是冻结预算；报告里如实写 null，不省略这个键。
    report["budget_extension"] = budget_extension
    out = args.out_dir / f"{args.arm}_g3_smoke_report.json"
    g3.write_report(out, report)
    g3.write_report(args.out_dir / f"{args.arm}_g3_run_fingerprint.json", run.fingerprint)

    print(f"[time ] {run.seconds:.0f}s  {pace['observed_s_per_trace']}s/trace")
    print(f"[G3   ] arm={args.arm} passed={report['passed']}")
    if report["failed_checks"]:
        print(f"[FAIL ] {report['failed_checks']}", file=sys.stderr)
    print(f"[out  ] {out}")
    return 0 if report["passed"] else 1


#: 两臂 max_new_tokens 不同**这条规则**是通用的（H1 自由生成 / MCQ 只出一个字母），
#: 可以继承；但某次运行的历史数值与纠正事件**属于那次运行**，不得写进别的运行。
ARM_SPECIFIC_MAX_NEW_TOKENS_RULE = {
    "field": "generation_params.max_new_tokens",
    "rule": ("按冻结设计**分臂**，不属于 backend invariant；两臂各自必须精确匹配"
             "冻结值（H1=384 / MCQ=16），或该臂**已登记并通过生成长度检查**的扩展预算；"
             "两种情况都是精确比对，不接受任何其他取值"),
    "checked_by": "v3_g3_smoke.backend_invariant 的第三段（逐臂精确比对）",
}


def _threshold_margin(arm: str, report: dict) -> dict:
    """据实报告本轮的分子/分母/阈值/余量。**不继承任何历史数值。**"""
    checks = report.get("checks") or {}
    detail = (report.get("detail") or {}).get("_detail") or {}
    if arm == g3.ARM_H1:
        num, den = detail.get("n_eligible"), detail.get("n_rows")
        value, threshold = checks.get("eligible_rate"), g3.CRITERIA.h1_min_eligible_rate
        direction, ok = ">=", checks.get("eligible_rate_ok")
        margin = (None if value is None else round(value - threshold, 4))
    else:
        num, den = detail.get("n_parse_failure"), detail.get("n_rows")
        value = checks.get("parse_failure_rate")
        threshold = g3.CRITERIA.mcq_max_parse_failure_rate
        direction, ok = "<=", checks.get("parse_failure_rate_ok")
        margin = (None if value is None else round(threshold - value, 4))
    return {
        "metric": ("eligible_rate" if arm == g3.ARM_H1 else "parse_failure_rate"),
        "numerator": num, "denominator": den, "value": value,
        "threshold": threshold, "direction": direction, "ok": ok,
        "margin": margin,
        # 恰好达标必须显式标出 —— 零余量不得被读成"稳定通过"。
        "exactly_at_threshold": (margin == 0.0),
        "margin_caveat": ("margin=0 表示恰好达标、零余量，**不得**写成'稳定通过'；"
                          "也不得据此提高或降低门槛"),
    }


def _load_summary_budget(args, cfg: V3AnalysisConfig, reports: dict):
    """汇总用的预算登记。**登记与两臂报告必须互相对得上**，否则拒绝汇总。

    只有登记文件本身还不够：必须确认**产出这批证据的那次运行**用的就是这份登记
    （逐臂比对报告里 `budget_extension` 记下的 sha256 与预算），否则就会出现
    "拿一份登记去认领另一批产物"。未登记的臂必须在报告里明确写着**没有**扩展
    （`budget_extension is None`），缺这一条就能把旧证据混进新链。
    """
    reg = None
    # getattr 而非直接取属性：既有调用方（含测试里的 args 桩）没有这个字段，
    # 缺字段的含义就是"本次没有预算扩展"，走冻结值那条路。
    reg_path = getattr(args, "budget_registration", None)
    if reg_path is not None:
        try:
            reg = g3.load_budget_registration(reg_path, cfg=cfg,
                                              model_path=getattr(args, "model_path", None))
        except ValueError as exc:
            print(f"[stop ] {exc}", file=sys.stderr)
            return None, 2
    for arm, report in reports.items():
        stamp = report.get("budget_extension")
        if reg is not None and reg.arm == arm:
            if not stamp or not stamp.get("registered"):
                print(f"[stop ] {arm} 报告里没有 budget_extension —— 它不是在这份登记下"
                      f"跑出来的，拒绝汇总", file=sys.stderr)
                return None, 2
            if stamp.get("registration_sha256") != reg.sha256:
                print(f"[stop ] {arm} 报告记录的登记 sha256 与本次传入的不是同一份，"
                      f"拒绝汇总", file=sys.stderr)
                return None, 2
            if stamp.get("effective_max_new_tokens") != reg.registered_max_new_tokens:
                print(f"[stop ] {arm} 报告记录的预算与登记不符，拒绝汇总", file=sys.stderr)
                return None, 2
        elif stamp:
            print(f"[stop ] {arm} 报告带着一份**未被本次汇总认可**的预算扩展 "
                  f"{stamp.get('effective_max_new_tokens')} —— 拒绝汇总，"
                  f"不把未登记的扩展证据混进链条", file=sys.stderr)
            return None, 2
    return reg, 0


def _summarise(args, cfg: V3AnalysisConfig, criteria: g3.G3Criteria) -> int:
    """两臂汇总。**只报本轮实测**，并在以下情况下拒绝给出真实通过结论。

    拒绝条件（任一命中即不写真实通过）：
        1. 任一臂报告缺 `criteria`
        2. 两臂 `criteria` 不一致（那意味着两臂跑在不同几何上，backend invariant 已破）
        3. 任一臂 `dry_run=true`（假后端不能汇总成真实 G3 通过，
           也不能把生产指纹标成已冻结）
    """
    reports, runs = {}, {}
    sources = {arm: args.out_dir / f"{arm}_g3_smoke_report.json" for arm in g3.ARMS}
    for arm in g3.ARMS:
        path = sources[arm]
        if not path.exists():
            print(f"[stop ] 缺少 {path}；两臂都跑完才能汇总", file=sys.stderr)
            return 2
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[arm] = report
        runs[arm] = g3.ArmRun(arm=arm, fingerprint=report["fingerprint"])
        runs[arm].passed = report["passed"]                     # type: ignore[attr-defined]

    # ---- 拒绝条件 1 / 2：criteria 必须齐备且两臂一致 --------------------
    refusals: list[str] = []
    seen = {arm: r.get("criteria") for arm, r in reports.items()}
    missing = sorted(a for a, c in seen.items() if not c)
    if missing:
        print(f"[stop ] 以下臂的报告缺 criteria：{missing} —— 无法确认判定用的几何，"
              f"拒绝汇总", file=sys.stderr)
        return 2
    arms_sorted = sorted(seen)
    shared = seen[arms_sorted[0]]
    for arm in arms_sorted[1:]:
        if seen[arm] != shared:
            diff = sorted(k for k in set(shared) | set(seen[arm])
                          if shared.get(k) != seen[arm].get(k))
            print(f"[stop ] 两臂 G3 判据不一致，字段 {diff} —— 两臂跑在不同几何上，"
                  f"backend invariant 已破，拒绝汇总", file=sys.stderr)
            return 2
    # CLI 传入的几何必须与两臂**实际跑用**的一致，否则汇总会张冠李戴。
    if criteria.as_dict() != shared:
        diff = sorted(k for k in set(shared) | set(criteria.as_dict())
                      if shared.get(k) != criteria.as_dict().get(k))
        print(f"[stop ] --summarise 传入的几何与两臂报告实际使用的不一致，字段 {diff}"
              f" —— 拒绝汇总", file=sys.stderr)
        return 2

    # ---- 拒绝条件 3：dry-run 不得汇总成真实 G3 通过 ----------------------
    dry = {arm: bool(r.get("dry_run")) for arm, r in reports.items()}
    any_dry = any(dry.values())
    if any_dry:
        refusals.append("dry_run_present:假后端运行不得汇总为真实 G3 通过，"
                        "也不得把生产指纹标为已冻结")

    budget_registration, code = _load_summary_budget(args, cfg, reports)
    if code:
        return code
    if budget_registration is not None:
        print(f"[budget] {budget_registration.arm} 登记预算 "
              f"{budget_registration.registered_max_new_tokens}"
              f"（冻结值 {budget_registration.frozen_max_new_tokens}）  "
              f"sha={budget_registration.sha256[:12]}…  "
              f"长度检查 passed={budget_registration.length_check.get('passed')} "
              f"截断率={budget_registration.length_check.get('touched_max_new_tokens_rate')}")

    invariant = g3.backend_invariant(runs, cfg, budget_registration=budget_registration)
    gate_passed = invariant["ok"] and all(getattr(r, "passed") for r in runs.values())
    mode = "simulated" if any_dry else "real"
    real_g3_passed = bool(gate_passed and not any_dry)

    summary = {
        "schema_version": "stage1_g3_summary_v3", "gate": "G3",
        "gate_only": True, "performance_blind": True,
        "did_not_compute": list(g3.DID_NOT_COMPUTE),
        "mode": mode,
        "dry_run_per_arm": dry,
        "arms_passed": {arm: getattr(r, "passed") for arm, r in runs.items()},
        "backend_invariant": invariant,
        # 本轮实际使用的判据与几何 —— 不是模块默认值。
        "criteria": shared,
        "geometry": {k: shared.get(k) for k in
                     ("hidden_dim", "block_index_zero_based",
                      "hidden_states_tuple_index")},
        # 本轮实测计数与阈值余量 —— 不继承任何历史数值。
        "counts": {arm: {"n_prompts": r.get("n_prompts"), "n_traces": r.get("n_traces"),
                         "traces_per_prompt": r.get("traces_per_prompt"),
                         "n_rows_written": r.get("n_rows_written"),
                         "n_failures": r.get("n_failures")}
                   for arm, r in reports.items()},
        "thresholds": {arm: _threshold_margin(arm, r) for arm, r in reports.items()},
        "hidden_verification": {arm: r.get("hidden_verification")
                                for arm, r in reports.items()},
        "elapsed_seconds": {arm: r.get("elapsed_seconds") for arm, r in reports.items()},
        "arm_specific_max_new_tokens_rule": dict(ARM_SPECIFIC_MAX_NEW_TOKENS_RULE),
        "budget_registration": (budget_registration.as_dict()
                                if budget_registration else None),
        "budget_extension_per_arm": {arm: r.get("budget_extension")
                                     for arm, r in reports.items()},
        "history_inheritance": ("本汇总只含**本轮**实测；不继承任何其他运行的历史"
                                "计数、纠正事件或归档产物哈希"),
        "refusals": refusals,
        "production_fingerprint_is_now_frozen": real_g3_passed,
        "real_g3_passed": real_g3_passed,
        "passed": gate_passed,
        # 两臂 gate 绑定：实查结果（或 None = 本次不是跨目录复用形态）。
        "source_reports": {arm: str(sources[arm]) for arm in g3.ARMS},
    }

    out = args.out_dir / "g3_summary.json"
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if out.exists():
        # 不静默覆盖已有结果：内容相同则 no-op，不同则拒绝并保留现场。
        if out.read_text(encoding="utf-8") == payload:
            print(f"[same ] 既有汇总与现场重算逐字节相同，未改动：{out}")
        else:
            print(f"[stop ] 已存在且内容不同的汇总：{out} —— 拒绝静默覆盖。"
                  f"请先确认来源并将其移开或归档", file=sys.stderr)
            return 2
    else:
        write_atomic(out, payload)

    print(f"[inv  ] backend invariant ok={invariant['ok']} "
          f"mismatched={invariant['mismatched_keys']}")
    print(f"[geom ] {summary['geometry']}")
    for arm, t in summary["thresholds"].items():
        print(f"[thr  ] {arm} {t['metric']}={t['value']} {t['direction']} "
              f"{t['threshold']} ({t['numerator']}/{t['denominator']}) "
              f"margin={t['margin']}"
          + ("  ← 恰好达标、零余量" if t["exactly_at_threshold"] else ""))
    print(f"[G3   ] mode={mode} passed={gate_passed} real_g3_passed={real_g3_passed}")
    if refusals:
        print(f"[REFUSE] {refusals}", file=sys.stderr)
    print(f"[out  ] {out}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
