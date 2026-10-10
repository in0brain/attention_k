"""Stage 1 的 G3 smoke（启动 gate G3，v3.1）。

**判据在本模块顶部写死，运行前即已固定。** 不看完 smoke 再决定 `passed`；
任一条件失败 → `passed=false` 并停止，**不修改 prompt / seed / 阈值 / 生成参数后静默重试**。

为什么另起一个入口而不复用 Stage 0 的 smoke（两条硬理由）：
  1. `conditional_increment.check_preregistration_frozen` 用**原始字节** sha256，
     而 v3.1 的 lock 是 `hash_mode: text-lf-utf8`。把 Stage 0 的 smoke 指向 v3.1 的 lock
     会直接 G2 失败。本模块用 `v3_freeze.require_v3_frozen`。
  2. Stage 0 的 `smoke_mcq` **不是生成 smoke** —— 它读的是已经生成好的 pilot 记录与
     hidden cache，属下游工程检查。MCQ 侧的 G3 生成路径本来就不存在。

本模块**只读复用** `generation/{backends,fingerprint,manifest}`、`h1_data`、`cyber_data`，
Stage 0 的脚本与模块一行未改。

**performance-blind**：不计算也不输出 AUROC / AUPRC / Δ_H / CI / ladder / probe /
任何分类或性能统计。产物带 `did_not_compute` 清单。
（`wrong_label` / H1 的 `completion_label` 是**该样本的真值**，不是"探针表现如何"，
  与 Stage 0 的 G4 同一处理：不受 performance-blind 约束。）
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from recover_attention import cyber_data as cd
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_freeze import require_v3_frozen, sha256_text_lf
from recover_attention.evaluation.v3_sample_builder import MANIFEST_NAME, chain_digest
from recover_attention.generation import backends as gb
from recover_attention.generation import fingerprint as fp
from recover_attention.generation.manifest import WorkManifest, write_atomic

SCHEMA_VERSION = "stage1_g3_smoke_report_v1"
ARM_H1 = "h1"
ARM_MCQ = "mcq"
ARMS = (ARM_H1, ARM_MCQ)


# ══════════════════════════════════════════════════════════════════════════════
# 判据（**跑前写死**，2026-08-08 用户裁决；沿用 v2.4 门槛并加结构检查）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class G3Criteria:
    n_prompts: int = 20
    traces_per_prompt: int = 6
    n_traces: int = 120
    # 三项模型几何：默认 = Qwen 冻结值。跨模型运行由 CLI 显式覆盖，
    # **不要**改这里的默认值 —— 改默认值会让 Qwen 的 G3 判据静默变成别的模型的。
    hidden_dim: int = 3584
    block_index: int = 19
    tuple_index: int = 20
    h1_min_eligible_rate: float = 0.80
    mcq_max_parse_failure_rate: float = 0.10
    required_unit_fields: tuple[str, ...] = ("prompt_hash", "unit_hash", "seed", "sample_index")

    def as_dict(self) -> dict:
        return {
            "frozen_before_run": True,
            "n_prompts": self.n_prompts,
            "traces_per_prompt": self.traces_per_prompt,
            "n_traces": self.n_traces,
            "hidden_dim": self.hidden_dim,
            "block_index_zero_based": self.block_index,
            "hidden_states_tuple_index": self.tuple_index,
            "h1_min_eligible_rate": self.h1_min_eligible_rate,
            "mcq_max_parse_failure_rate": self.mcq_max_parse_failure_rate,
            "required_unit_fields": list(self.required_unit_fields),
            "on_failure": ("passed=false 并停止；不得修改 prompt / seed / 阈值 / 生成参数"
                           "后静默重试"),
        }


CRITERIA = G3Criteria()

DID_NOT_COMPUTE = [
    "AUROC", "AUPRC", "delta_H", "any confidence interval",
    "any ladder rung", "any probe or classifier performance",
    "any statistic over the confirmatory sample",
]


# ══════════════════════════════════════════════════════════════════════════════
# 启动前校验（任一不过即停，**不加载模型**）
# ══════════════════════════════════════════════════════════════════════════════
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_inputs(*, samples_dir: Path, prereg: Path, lock: Path,
                  cfg: V3AnalysisConfig, gate: str = "G3") -> dict:
    """G2 → sample manifest → 六份名单 chain hash → G3 名单形态。

    ⚠️ **只打开两份 g3 文件**。正式名单的路径根本不进本函数的返回值，
    调用方无从据此读到正式集 —— 这是结构性保证，不是承诺。
    """
    freeze = require_v3_frozen(prereg, lock)          # 不 match 即抛错

    manifest_path = samples_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256_file(manifest_path)

    if manifest["prereg"]["sha256"] != freeze.current_sha256:
        raise ValueError(
            f"sample manifest 记录的 prereg sha256 与当前不一致："
            f"{manifest['prereg']['sha256']} != {freeze.current_sha256}")
    if manifest["prereg"]["lock_version"] != freeze.locked_version:
        raise ValueError("sample manifest 的 lock_version 与当前 lock 不一致")

    filenames = {
        "h1_primary": "h1_primary_entries.jsonl", "h1_G3": "h1_g3_entries.jsonl",
        "h1_G4": "h1_g4_entries.jsonl", "mcq_primary": "mcq_primary_questions.jsonl",
        "mcq_G3": "mcq_g3_questions.jsonl", "mcq_G4": "mcq_g4_questions.jsonl",
    }
    key_of = {"h1": "source_entry_id", "mcq": "question_hash"}
    chain_checks: dict[str, str] = {}
    for name, filename in filenames.items():
        arm = name.split("_")[0]
        rows = _read_jsonl(samples_dir / filename)
        actual = chain_digest(r[key_of[arm]] for r in rows)
        expected = manifest["chain_sha256"][name]
        if actual != expected:
            raise ValueError(f"{filename} 的 chain hash 与 manifest 不符：{actual} != {expected}")
        chain_checks[name] = actual

    expected_n = {"G3": cfg.n_g3_per_arm, "G4": cfg.n_g4_per_arm}[gate]
    gate_rows = {arm: _read_jsonl(samples_dir / filenames[f"{arm}_{gate}"]) for arm in ARMS}
    for arm, rows in gate_rows.items():
        if len(rows) != expected_n:
            raise ValueError(f"{arm} {gate} 名单 {len(rows)} 条 != {expected_n}")
        if not all(r.get("gate_only") is True for r in rows):
            raise ValueError(f"{arm} {gate} 名单存在 gate_only != true 的行")
        if not all(r.get("gate") == gate for r in rows):
            raise ValueError(f"{arm} {gate} 名单存在 gate != {gate!r} 的行")
    if not all(r.get("previously_burned") is True for r in gate_rows[ARM_H1]):
        raise ValueError(f"H1 {gate} 名单必须全部 previously_burned=true（不得消耗正式新池）")

    return {
        "g2": freeze.as_dict(),
        "sample_manifest_path": str(manifest_path).replace("\\", "/"),
        "sample_manifest_sha256": manifest_sha,
        "chain_sha256_verified": chain_checks,
        "n_lists_verified": len(chain_checks),
        "gate": gate,
        "gate_counts": {arm: len(rows) for arm, rows in gate_rows.items()},
        "g3_counts": {arm: len(rows) for arm, rows in gate_rows.items()},
        "primary_lists_read": False,
        "primary_lists_written": False,
        "_gate_rows": gate_rows,       # 下划线开头：只在进程内传递，不写进产物
    }


# ══════════════════════════════════════════════════════════════════════════════
# 记录构造
# ══════════════════════════════════════════════════════════════════════════════
def build_arm_records(arm: str, gate_rows: list[dict], *, cfg: V3AnalysisConfig,
                      tokenizer: Any = None) -> list[dict]:
    """把 G3 名单行变成两个 backend 的 `run_unit(unit, record)` 所需的 record。"""
    if arm == ARM_H1:
        # H1 名单行本身就是 sample-builder 产出的完整 record（question_text / family / route）
        return [dict(row) for row in gate_rows]

    records = []
    for row in gate_rows:
        raw = {"question": row["question"], "answers": row["answers"],
               "solution": row["gold_original_label"]}
        canonical = cd.to_canonical_cyber_sample(
            raw, original_index=int(row["original_index"]), source=row["source"],
            shuffle_seed=cfg.mcq_shuffle_seed)
        canonical["question_hash"] = row["question_hash"]
        canonical["rendered_prompt"] = tokenizer.apply_chat_template(
            cd.build_mcq_chat_messages(canonical),
            tokenize=False, add_generation_prompt=True)
        records.append(canonical)
    return records


def units_for(records: list[dict], k: int) -> list[dict]:
    """K = 1 greedy + (k−1) sampled，与 Stage 0 同构。"""
    specs = [("greedy", 0, False)] + [("sampled", i + 1, True) for i in range(k - 1)]
    units = []
    for record in records:
        for sample_type, sample_index, _ in specs:
            units.append({
                "unit_id": f"{record['example_id']}__{sample_type}_{sample_index}",
                "example_id": record["example_id"],
                "sample_type": sample_type,
                "sample_index": sample_index,
            })
    return units


# ══════════════════════════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ArmRun:
    arm: str
    rows: list[dict] = field(default_factory=list)
    hidden: dict[str, np.ndarray] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)
    hidden_verification: dict = field(default_factory=dict)
    seconds: float = 0.0          # **本进程**用于生成的秒数（不含模型加载与续跑读回）
    n_units: int = 0              # 该臂的 unit 总数（含此前进程已完成的）
    n_generated_this_process: int = 0   # 本进程新生成的 unit 数；0 = 纯重判
    pace: dict = field(default_factory=dict)
    stopped_reason: str | None = None    # None = 跑完；max_new_units / max_wall_seconds


class PaceBudgetExceeded(RuntimeError):
    """前 N 条实测速率外推超过预算 —— 停下来报告，不擅自长跑。

    抛出前所有已完成 unit 均已落盘且在 manifest 里标 done，故续跑不丢工作。
    """

    def __init__(self, message: str, *, pace: dict):
        super().__init__(message)
        self.pace = pace


class DiagnosticRecoveryRequired(RuntimeError):
    """diagnostic 已落盘、manifest 事件却没跟上 —— 跨文件断电窗口的中间态。

    正常失败顺序是 `sanitized diagnostic durable append → manifest mark(excluded) → 停止`。
    两步之间被杀，就会留下「有诊断、无终态」的 unit：下一次启动会把它当 pending 重新入队，
    与"excluded 是终态、永不重试"直接冲突。

    本异常**只发现不修复** —— 不自动 mark、不重试、不改任何产物，交由研究者裁决。
    """

    def __init__(self, message: str, *, conflicts: list[dict]):
        super().__init__(message)
        self.conflicts = conflicts


def require_diagnostic_manifest_consistency(
        *, arm: str, diagnostic_path: Path | None, manifest_path: Path,
        known_units: Iterable[str] | None = None) -> dict:
    """只读一致性门：**diagnostic 存在 ⇒ manifest 必须 excluded/forward_nonfinite**。

    单向检查。反向不查 —— excluded 没有 diagnostic 是合法的：诊断持久化补丁之前
    发生的排除（如 MCQ 的 cybermetric_005396 整组排除，已登记 diagnostic_persisted=false）
    本就没有诊断文件，**不得**因此否定既有的 group exclusion。

    diagnostics 逐行严格解析，fail closed：malformed / 截断 / 缺 unit_id /
    重复 unit_id / arm 不符 / unit_id 不在本臂 unit 全集内，一律拒绝。
    """
    out = {"checked": 0, "diagnostic_units": 0, "path": str(diagnostic_path or "")}
    if diagnostic_path is None:
        return out                      # G3/G4 默认路径：不启用，行为不变
    diagnostic_path = Path(diagnostic_path)
    if not diagnostic_path.exists():
        return out

    seen: set[str] = set()
    units: list[str] = []
    for lineno, line in enumerate(
            diagnostic_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:      # malformed / 截断
            raise DiagnosticRecoveryRequired(
                f"{arm}: {diagnostic_path.name} 第 {lineno} 行不是合法 JSON（{exc}）"
                f" —— 诊断审计不可解析，拒绝启动",
                conflicts=[{"line": lineno, "kind": "malformed_json"}]) from exc
        uid = rec.get("unit_id")
        if not uid:
            raise DiagnosticRecoveryRequired(
                f"{arm}: {diagnostic_path.name} 第 {lineno} 行缺 unit_id，拒绝启动",
                conflicts=[{"line": lineno, "kind": "missing_unit_id"}])
        if rec.get("arm") != arm:
            raise DiagnosticRecoveryRequired(
                f"{arm}: {diagnostic_path.name} 第 {lineno} 行 arm={rec.get('arm')!r} 与本臂不符，拒绝启动",
                conflicts=[{"line": lineno, "unit_id": uid, "kind": "arm_mismatch"}])
        if uid in seen:
            raise DiagnosticRecoveryRequired(
                f"{arm}: {diagnostic_path.name} 中 unit_id {uid} 重复，拒绝启动",
                conflicts=[{"line": lineno, "unit_id": uid, "kind": "duplicate_unit_id"}])
        seen.add(uid)
        units.append(uid)

    if known_units is not None:
        allowed = set(known_units)
        unknown = [u for u in units if u not in allowed]
        if unknown:
            raise DiagnosticRecoveryRequired(
                f"{arm}: {diagnostic_path.name} 含未知 unit_id {sorted(unknown)[:5]}"
                f"（共 {len(unknown)} 个），拒绝启动",
                conflicts=[{"unit_id": u, "kind": "unknown_unit_id"} for u in unknown])

    latest: dict[str, dict] = {}
    if Path(manifest_path).exists():
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                ev = json.loads(line)
                latest[ev["unit_id"]] = ev
    conflicts = []
    for uid in units:
        ev = latest.get(uid)
        if ev is None:
            conflicts.append({"unit_id": uid, "status": None, "reason": None,
                              "kind": "missing_manifest_event"})
            continue
        if ev.get("status") != "excluded" or ev.get("reason") != "forward_nonfinite":
            conflicts.append({"unit_id": uid, "status": ev.get("status"),
                              "reason": ev.get("reason"), "kind": "status_or_reason_mismatch"})
    if conflicts:
        raise DiagnosticRecoveryRequired(
            f"{arm}: DiagnosticRecoveryRequired —— {len(conflicts)} 个 unit 有已落盘的 "
            f"forward-nonfinite 诊断，但 manifest latest event 不是 excluded/forward_nonfinite："
            f"{conflicts[:5]}。这是 diagnostic 与 manifest 之间的断电中间态；"
            f"**不自动 mark、不重试、不改任何产物**，请裁决后再启动。",
            conflicts=conflicts)
    out["checked"] = len(units)
    out["diagnostic_units"] = len(units)
    return out


def _append_diagnostic(path: Path, arm: str, unit: dict, diagnostic: dict) -> None:
    """append-only 落盘一条 forward-nonfinite 诊断。

    `allow_nan=False` 是**判据不是装饰**：candidate_logits 天然可能非有限，
    默认的 `json.dumps` 会写出 `NaN` / `Infinity` 这种非标准字面量，
    审计文件从此无法被严格 JSON 解析器读回。净化在 backends.sanitize_diagnostic
    里完成（含移除 completion 原文），这里只负责严格序列化与 durable 落盘。
    """
    record = {"schema_version": "stage1_forward_nonfinite_diagnostic_v1",
              "arm": arm, "unit_id": unit["unit_id"], **diagnostic}
    if "completion" in record:
        raise ValueError("diagnostic 仍含 completion 原文，拒绝落盘")
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_arm(*, arm: str, records: list[dict], backend, out_dir: Path,
            cfg: V3AnalysisConfig, progress: Callable[[dict], None] | None = None,
            progress_every: int = 20, pace_check_after: int | None = None,
            pace_budget_hours: float = 2.0, gate: str = "g3",
            after_unit: Callable[[dict, dict], None] | None = None,
            durable_fsync: bool = False,
            max_new_units: int | None = None,
            max_wall_seconds: float | None = None,
            stop_on_failure: bool = False,
            results_path: Path | None = None,
            hidden_dir: Path | None = None,
            manifest_path: Path | None = None,
            diagnostic_path: Path | None = None) -> ArmRun:
    """逐 unit 生成；append-only resume manifest；每 unit 独立可重跑。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 默认沿用 gate 派生的文件名（G3/G4 行为不变）；调用方可显式指定，
    # 以免"两层产物看的是两个不同文件"这种对不上的错误（续跑会因此找不到已完成的 unit）。
    hidden_dir = Path(hidden_dir) if hidden_dir else out_dir / f"{arm}_{gate}_unit_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    results_path = (Path(results_path) if results_path
                    else out_dir / f"{arm}_{gate}_unit_results.jsonl")
    man = WorkManifest(Path(manifest_path) if manifest_path
                       else out_dir / f"{arm}_{gate}_generated_manifest.jsonl")

    by_id = {r["example_id"]: r for r in records}
    units = units_for(records, cfg.k_traces)
    # **直接调用者的防线**：在任何 backend 调用与新 manifest 写入之前先过一致性门。
    # unknown unit 用 units_for 重建的**完整** unit 全集复核（CLI 那道门只有 manifest 现状）。
    require_diagnostic_manifest_consistency(
        arm=arm, diagnostic_path=diagnostic_path, manifest_path=man.path,
        known_units=[u["unit_id"] for u in units])
    man.register(u["unit_id"] for u in units)
    todo = set(man.pending_units(u["unit_id"] for u in units))
    prior_status = man.current_status()   # 用于区分 done（须有行）与 excluded（本就无行）

    run = ArmRun(arm=arm, n_units=len(units))
    run.fingerprint = backend.run_fingerprint(arm)
    run.hidden_verification = backend.verify_hidden_index()

    # resume：已完成的 unit **从落盘结果读回**，不重跑也不遗漏。
    # 少了这一步，续跑后的判定只看得见本次新跑的那些行，`every_unit_produced_a_row`
    # 会在一切正常的情况下判 false —— 断点续跑必须与一次跑完得到同一份判定输入。
    previous: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                previous[row["unit_id"]] = row       # append-only：同 id 以最后一条为准

    started = time.time()
    done = 0
    with open(results_path, "a", encoding="utf-8") as sink:
        for unit in units:
            if unit["unit_id"] not in todo:
                row = previous.get(unit["unit_id"])
                if row is None:
                    # **excluded 是"有结论、无数据"的终态**（如 forward 输出非有限）：
                    # 它本来就没有结果行，跳过是正确的，不是数据丢失。
                    # 只有 done 缺行才是"把数据丢失伪装成已完成"。
                    if prior_status.get(unit["unit_id"]) == "excluded":
                        continue
                    raise ValueError(
                        f"{unit['unit_id']} 在 manifest 里是 done，但 {results_path.name} "
                        f"里没有对应行 —— 这是 resume 最阴险的失败模式：把数据丢失伪装成"
                        f"'已完成'。请核查后再跑，不要静默跳过。")
                run.rows.append(row)
                vec_path = hidden_dir / f"{unit['unit_id']}.npy"
                if vec_path.exists():
                    run.hidden[unit["unit_id"]] = np.load(vec_path)
                continue
            if done == 0:
                # 计时从**第一条真正新生成的 unit** 开始：模型加载在本函数之前完成，
                # 续跑的读回阶段也不该混进速率。否则二次测速会被首批耗时污染。
                started = time.time()
            man.mark(unit["unit_id"], "running")
            try:
                row = backend.run_unit(unit, by_id[unit["example_id"]])
            except gb.ForwardNonFiniteError as exc:
                # **先持久化 sanitized diagnostic，再 mark/停止**：这次数值成因只在
                # 内存里存在过一次，进程一退就永久查不回来（MCQ 首段真实吃过这个亏）。
                # diagnostic_path 默认 None —— G3/G4 行为与文件集合完全不变，
                # 只有 production 显式传入时才落盘。
                if diagnostic_path is not None:
                    _append_diagnostic(Path(diagnostic_path), arm, unit, exc.diagnostic)
                # excluded 是**终态**：重跑不会让非有限数值变得有限，
                # 下一次运行不得重试（TERMINAL_STATUS 含 excluded）。
                man.mark(unit["unit_id"], "excluded", reason="forward_nonfinite")
                run.failures.append({"unit_id": unit["unit_id"], "kind": "forward_nonfinite",
                                     "diagnostic": exc.diagnostic})
                if stop_on_failure:
                    run.stopped_reason = "unit_failure:forward_nonfinite"
                    break
                continue
            except Exception as exc:                       # noqa: BLE001 —— 记录后继续，不吞
                # failed **不是**终态：可能是瞬时错误，pending_units() 会在下一次
                # **经用户批准的**运行中重新入队。本段不自动重试。
                kind = type(exc).__name__
                man.mark(unit["unit_id"], "failed", reason=kind)
                run.failures.append({"unit_id": unit["unit_id"], "kind": kind,
                                     "message": str(exc)[:400]})
                if stop_on_failure:
                    run.stopped_reason = f"unit_failure:{kind}"
                    break
                continue

            row["unit_id"] = unit["unit_id"]
            row["sample_type"] = unit["sample_type"]
            # `group_id` 是分组身份，v3 adapter 强制需要、Stage 0 的 G4 也显式从输入记录复制。
            # 两个 backend 的 run_unit 都**不产出**它，故在此从**冻结输入记录**确定性挂载。
            record = by_id[unit["example_id"]]
            if "group_id" not in row and record.get("group_id") is not None:
                row["group_id"] = record["group_id"]
            if after_unit is not None:
                after_unit(unit, row)      # G4：补挂精确生成 token 数等运行期观测
            # 崩溃事务顺序（固定）：hidden 落盘 → result 追加 → manifest 标 done。
            # 顺序反了会造出 "manifest 说 done 但数据不在" 的假完成态。
            vec = row.pop("hidden", None)
            if vec is not None:
                # **必须强制拷贝，不能用 np.asarray**：MCQ 的 row["hidden"] 是
                # `tf["hidden"][prompt_len + j]` —— 整条 trace 的 [T, 4096] hidden 的**视图**。
                # dtype 本来就是 float32，asarray 原样返回那个视图，于是下面 run.hidden
                # 每存一条，就把它背后 ~7 MB 的整表钉住整臂不放（实测 8.4 MB/unit，
                # 10,560 条 ≈ 88 GB，必然撑爆）。H1 走 mean pooling 产出新数组，所以不漏。
                # copy 只改生命周期，**数值逐位相同**。
                vec = np.array(vec, dtype=np.float32, copy=True)
                _save_hidden(hidden_dir / f"{unit['unit_id']}.npy", vec,
                             durable=durable_fsync)
                run.hidden[unit["unit_id"]] = vec
                row["hidden_dim"] = int(vec.size)
            sink.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
            sink.flush()
            if durable_fsync:
                os.fsync(sink.fileno())      # manifest 自带 fsync；这一层先前只 flush
            run.rows.append(row)
            man.mark(unit["unit_id"], "done")     # WorkManifest._append 内已 fsync
            done += 1
            if progress and done % progress_every == 0:
                progress({"arm": arm, "done": done, "total": len(todo),
                          "elapsed_s": round(time.time() - started, 1),
                          "failures": len(run.failures)})

            # 分段：只在 **unit 边界** 停止，段末正常退出（不是异常）。
            if max_new_units is not None and done >= max_new_units:
                run.stopped_reason = "max_new_units"
                break
            if max_wall_seconds is not None and (time.time() - started) >= max_wall_seconds:
                run.stopped_reason = "max_wall_seconds"
                break

            # 前 N 条实测后外推整轮耗时；超预算即停并报告（不擅自长跑）。
            if pace_check_after and done == pace_check_after:
                elapsed = time.time() - started
                rate = elapsed / done
                projected_h = rate * len(todo) / 3600.0
                pace = {"observed_s_per_trace": round(rate, 2),
                        "measured_on_units": done,
                        "projected_hours_this_arm": round(projected_h, 2),
                        "budget_hours": pace_budget_hours,
                        "over_budget": projected_h > pace_budget_hours}
                run.pace = pace
                if pace["over_budget"]:
                    raise PaceBudgetExceeded(
                        f"{arm}: 前 {done} 条实测 {rate:.1f}s/trace → 外推 "
                        f"{projected_h:.2f}h > 预算 {pace_budget_hours}h。已完成的 unit "
                        f"全部落盘且可续跑；停下来报告，不擅自长跑。", pace=pace)

    gb.attach_cross_sample_f5(run.rows, arm)
    run.n_generated_this_process = done
    # 本进程一条都没生成（纯重判）时，耗时无分母可言 —— 记 0 并由报告标注，
    # 不得让一个 0.1 秒冒充"这批 240 条只花了 0.1 秒"。
    run.seconds = (time.time() - started) if done else 0.0
    return run


def _save_hidden(path: Path, vec: np.ndarray, *, durable: bool) -> None:
    """hidden 的落盘。durable=True 时：临时文件 → fsync → atomic replace。

    非 durable 路径保持 G3/G4 的既有行为（直接 np.save），默认不变。
    """
    if not durable:
        np.save(path, vec)
        return
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as handle:
            np.save(handle, vec)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


# ══════════════════════════════════════════════════════════════════════════════
# 判定（**只应用顶部写死的判据**）
# ══════════════════════════════════════════════════════════════════════════════
def _common_checks(run: ArmRun, *, criteria: G3Criteria, n_records: int) -> dict:
    rows = run.rows
    unexpected = [f for f in run.failures if f["kind"] != "forward_nonfinite"]
    non_finite = 0
    for vec in run.hidden.values():
        non_finite += int(np.count_nonzero(~np.isfinite(vec)))

    missing_fields = sorted({
        field_name for row in rows for field_name in criteria.required_unit_fields
        if field_name not in row
    })
    dims = {int(v.size) for v in run.hidden.values()}
    hv = run.hidden_verification

    return {
        "n_prompts_ok": n_records == criteria.n_prompts,
        "n_traces_ok": run.n_units == criteria.n_traces,
        "traces_per_prompt_ok": (run.n_units == n_records * criteria.traces_per_prompt),
        "no_unhandled_exception": not unexpected,
        "no_forward_nonfinite": not any(f["kind"] == "forward_nonfinite" for f in run.failures),
        "no_non_finite_values": non_finite == 0,
        "every_unit_produced_a_row": len(rows) == run.n_units,
        "unit_fields_present": not missing_fields,
        "hidden_dim_ok": dims == {criteria.hidden_dim} if dims else False,
        "hidden_index_numeric_match": bool(hv.get("numeric_match")),
        "hidden_index_matches_frozen": (
            hv.get("block_index_zero_based") == criteria.block_index
            and hv.get("hidden_states_tuple_index") == criteria.tuple_index),
        "_detail": {"missing_unit_fields": missing_fields,
                    "hidden_dims_seen": sorted(dims),
                    "n_failures": len(run.failures),
                    "n_unexpected_failures": len(unexpected),
                    "n_non_finite_values": non_finite},
    }


def _h1_checks(run: ArmRun, *, criteria: G3Criteria) -> dict:
    rows = run.rows
    eligible = [r for r in rows if r.get("id_token_positions_local")]
    rate = len(eligible) / len(rows) if rows else 0.0
    every_eligible_has_hidden = all(r["unit_id"] in run.hidden for r in eligible)
    pooled_ok = all(
        r.get("hidden_pooled_n_tokens") == len(set(r["id_token_positions_local"]))
        for r in eligible)
    mentions_aligned = all(
        m.get("token_indices") for r in rows for m in r.get("mentions", [])
        if m.get("emitted", True) and m.get("start") is not None)
    reason_recorded = all(
        ("refusal" in r and "id_token_positions_local" in r) for r in rows)
    return {
        "eligible_rate": round(rate, 4),
        "eligible_rate_ok": rate >= criteria.h1_min_eligible_rate,
        "every_eligible_trace_has_hidden": every_eligible_has_hidden,
        "every_emitted_identifier_aligns_to_tokens": bool(mentions_aligned),
        "pooling_is_all_identifier_tokens": bool(pooled_ok),
        "no_emission_or_exclusion_reason_recorded": bool(reason_recorded),
        "_detail": {"n_eligible": len(eligible), "n_rows": len(rows)},
    }


def _mcq_checks(run: ArmRun, *, criteria: G3Criteria) -> dict:
    rows = run.rows
    parsed = [r for r in rows if r.get("eligible_for_primary")]
    failures = len(rows) - len(parsed)
    rate = failures / len(rows) if rows else 1.0
    f5_fields = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy",
                 "f5_letter_token_logprob")
    complete = all(all(f in r and r[f] is not None for f in f5_fields) for r in parsed)
    has_hidden = all(r["unit_id"] in run.hidden for r in parsed)
    return {
        "parse_failure_rate": round(rate, 4),
        "parse_failure_rate_ok": rate <= criteria.mcq_max_parse_failure_rate,
        "every_parsed_record_has_hidden": has_hidden,
        "every_parsed_record_has_full_f5": bool(complete),
        "_detail": {"n_parsed": len(parsed), "n_parse_failure": failures, "n_rows": len(rows)},
    }


#: 两臂必须**逐键相同**的指纹块。
FINGERPRINT_KEYS = ("model", "tokenizer", "hidden_index")

#: 两臂必须相同的采样协议参数。
INVARIANT_GENERATION_PARAMS = ("K", "temperature", "top_p", "seed")

#: 2026-09-13 起**显式传入并记录**的两个采样参数。它们一直在生效（旧 Qwen 经
#: generation_config.json 得到 20 / 1.05），只是从前没有进过指纹。只有在双方都带
#: `GENERATION_PARAMS_SCHEMA` 标记时才逐键比较 —— 否则我们比的是两个 `None`，
#: 那不是"一致"，那是"都没记"。
RECORDED_SAMPLING_PARAMS = ("top_k", "repetition_penalty")

#: 见 fingerprint.GENERATION_PARAMS_SCHEMA。
GENERATION_PARAMS_SCHEMA = fp.GENERATION_PARAMS_SCHEMA


def _all_schema_tagged(*fingerprints: Mapping[str, Any]) -> bool:
    """全部带新口径标记时才逐键比较 RECORDED_SAMPLING_PARAMS。"""
    return all((f.get("generation_params") or {}).get("schema") == GENERATION_PARAMS_SCHEMA
               for f in fingerprints)


def generation_params_schema_mismatch(*fingerprints: Mapping[str, Any]) -> list[str]:
    """跨口径比较必须当场失败 —— **缺字段不等于"与任意取值兼容"**。

    旧指纹没有 top_k / repetition_penalty，其实际生效值由各模型的 generation_config
    兜底（Qwen 20/1.05、Llama 50/1.0）。拿新旧两份指纹比较时，若把缺字段当成 None
    去和 20 比，只会得到一个"不一致"；但若两边都缺，`None == None` 会**静默通过**，
    而它们的真实取值完全可能不同。所以这里按**口径**判，不按取值判。

    返回空列表表示可比；否则返回不匹配说明（调用方并入 mismatched_keys）。
    **不改写任何旧产物去补字段。**
    """
    seen = {(f.get("generation_params") or {}).get("schema") for f in fingerprints}
    if seen == {GENERATION_PARAMS_SCHEMA}:
        return []
    if seen == {None}:
        # 双方都是旧口径：同一批旧产物内部比较仍然合法（它们出自同一次运行/同一模型），
        # 但两个采样键无据可查，**不做比较也不声称已比过**。
        return []
    return [f"generation_params.schema: 新旧口径混用 {sorted(str(s) for s in seen)}；"
            f"旧指纹未记录 {'/'.join(RECORDED_SAMPLING_PARAMS)}，其实际生效值无从得知，"
            f"不得与新口径指纹混用（不补字段、不改写旧产物，须用同口径重新产出）"]

#: **按冻结设计分臂**的生成参数：不要求两臂相同，但各自必须精确匹配冻结值
#: （或**已登记**的扩展预算 —— 见 `BudgetRegistration`）。
ARM_SPECIFIC_GENERATION_PARAMS = ("max_new_tokens",)


# ==============================================================================
# 已登记的分臂解码预算
# ==============================================================================
#: 生成长度扩展的登记文件口径标记。
BUDGET_REGISTRATION_SCHEMA = "stage1_generation_budget_registration_v1"


@dataclass(frozen=True)
class BudgetRegistration:
    """一份**已核验**的解码预算登记。

    存在这个对象，等价于下面几件事**全部**已经当场查过：
      (1) 登记文件口径正确，且它声明的 `frozen_max_new_tokens` 与 `v3_config` 一致
          —— 登记不是"改冻结值"，它必须写清自己是从哪个冻结值扩展出来的
      (2) 登记的 `model_path` 与本次运行的模型一致
      (3) 登记引用的校准题单存在，且 sha256 与登记记录一致
      (4) 该题单旁边有一份**通过的**生成长度检查报告，且这份报告引用的登记文件
          sha256 与本文件逐字相同、预算值相同、题单 sha256 相同

    (4) 是关键：**只"传了一个登记文件"不放行任意预算** —— 必须有一次已经通过的
    生成长度检查把这个预算和这份登记绑在一起。构造失败一律抛 `ValueError`，
    没有"降级放行"这条路。
    """

    path: str
    sha256: str
    arm: str
    frozen_max_new_tokens: int
    registered_max_new_tokens: int
    model_path: str
    length_check: dict          # 关联的生成长度检查证据（路径、sha256、截断率）

    def as_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "arm": self.arm,
                "frozen_max_new_tokens": self.frozen_max_new_tokens,
                "registered_max_new_tokens": self.registered_max_new_tokens,
                "model_path": self.model_path,
                "length_check": dict(self.length_check),
                "note": ("透明登记的预算扩展。**不是**原冻结配置下的严格同参数复现；"
                         "**不**代表该值是最小可用预算；**不**放宽任何长度一致性检查 —— "
                         "未登记的臂仍按 v3_config 的冻结值逐臂精确比对")}


def _frozen_max_new_tokens(arm: str, cfg: V3AnalysisConfig) -> int:
    return cfg.h1_max_new_tokens if arm == ARM_H1 else cfg.mcq_max_new_tokens


def expected_max_new_tokens(arm: str, cfg: V3AnalysisConfig,
                            registration: "BudgetRegistration | None" = None) -> int:
    """本臂**应当**出现的解码预算。

    `registration=None`（默认）或登记的是另一条臂 —— 取冻结值。Qwen 与任何未登记的
    运行走的都是这条路，行为与引入登记机制之前逐字相同。
    """
    if registration is not None and registration.arm == arm:
        return registration.registered_max_new_tokens
    return _frozen_max_new_tokens(arm, cfg)


def load_budget_registration(path, *, cfg: V3AnalysisConfig,
                             model_path: str | None = None) -> BudgetRegistration:
    """读取并**逐项核验**一份预算登记；任一项不满足即抛 `ValueError`。

    核验项见 `BudgetRegistration` 的 docstring。特别地，**不接受**只有登记文件、
    没有配套通过的生成长度检查 —— 那等于允许凭一个 JSON 放行任意预算。
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    digest = _sha256_file(path)
    problems: list[str] = []

    if doc.get("schema") != BUDGET_REGISTRATION_SCHEMA:
        problems.append(f"schema {doc.get('schema')!r} != {BUDGET_REGISTRATION_SCHEMA}")
    arm = doc.get("arm")
    if arm not in ARMS:
        problems.append(f"arm {arm!r} 不是 {ARMS}")
    else:
        frozen = _frozen_max_new_tokens(arm, cfg)
        if doc.get("frozen_max_new_tokens") != frozen:
            problems.append(
                f"frozen_max_new_tokens {doc.get('frozen_max_new_tokens')!r} != "
                f"v3_config 的 {frozen} —— 登记必须写清自己是从哪个冻结值扩展出来的")
    budget = doc.get("registered_max_new_tokens")
    if not isinstance(budget, int) or budget <= 0:
        problems.append(f"registered_max_new_tokens {budget!r} 不是正整数")
    if model_path is not None and doc.get("model_path") != model_path:
        problems.append(f"model_path {doc.get('model_path')!r} != 本次运行的 {model_path!r}")

    # -- 校准题单：存在且 sha256 与登记一致 -------------------------------
    cal = doc.get("calibration_set") or {}
    # `calibration_set.path` 是**仓库相对**路径。仓库根从登记文件自身的祖先里找：
    # 第一个能让该相对路径落到实存文件的祖先即是。这样真实产物与测试用的临时目录
    # 走的是同一条代码，不需要在源码里留测试钩子。
    rel = str(cal.get("path", ""))
    root = Path(__file__).resolve().parents[3]
    for ancestor in path.resolve().parents:
        if rel and (ancestor / rel).exists():
            root = ancestor
            break
    list_path = root / rel
    if not rel or not list_path.exists():
        problems.append(f"登记引用的校准题单不存在：{cal.get('path')!r}")
    elif _sha256_file(list_path) != cal.get("list_sha256"):
        problems.append("校准题单的 sha256 与登记记录不符")

    # -- 生成长度检查：必须存在、必须通过、必须引用的就是这份登记 ---------
    length_check: dict = {}
    if not problems:
        report_path = list_path.parent / "mcq_budget_check_report.json"
        if not report_path.exists():
            problems.append(f"找不到配套的生成长度检查报告：{report_path}")
        else:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            if rep.get("passed") is not True:
                problems.append("生成长度检查报告 passed != true —— 未通过的检查不得放行预算")
            if (rep.get("registration") or {}).get("sha256") != digest:
                problems.append("生成长度检查报告引用的登记文件与本文件不是同一份")
            if (rep.get("registration") or {}).get("registered_max_new_tokens") != budget:
                problems.append("生成长度检查用的预算与本登记不一致")
            if (rep.get("calibration_set") or {}).get("sha256") != cal.get("list_sha256"):
                problems.append("生成长度检查用的校准题单与本登记不一致")
            if not problems:
                length_check = {
                    "path": str(report_path.relative_to(root)).replace("\\", "/"),
                    "sha256": _sha256_file(report_path),
                    "passed": rep.get("passed"),
                    "touched_max_new_tokens_rate": (rep.get("checks") or {}).get(
                        "touched_max_new_tokens_rate"),
                    "n_rows": rep.get("n_rows"),
                }

    if problems:
        raise ValueError(
            f"预算登记 {path} 不可用：{problems}。**不放行** —— "
            f"登记必须与冻结值、模型、校准题单和一次**已通过**的生成长度检查逐项对上。")

    return BudgetRegistration(
        path=str(path).replace("\\", "/"), sha256=digest, arm=arm,
        frozen_max_new_tokens=doc["frozen_max_new_tokens"],
        registered_max_new_tokens=budget, model_path=doc.get("model_path"),
        length_check=length_check)


def backend_invariant(runs: dict[str, ArmRun], cfg: V3AnalysisConfig | None = None, *,
                      budget_registration: BudgetRegistration | None = None) -> dict:
    """两臂的后端不变量检查（2026-08-08 按冻结设计修正）。

    **`max_new_tokens` 不属于 backend invariant** —— 它是按冻结设计分臂的解码预算
    （H1 自由生成 384 / MCQ 只出一个字母 16）。Stage 0 的两份 run_fingerprint 一模一样地
    差着这一项，Stage 0 自己的 `quality_check._check_backend_invariant` 也只比 backend 四键。
    先前把 `generation_params` 整块要求逐键相同，是**检查器写错**，不是后端漂移。

    但**不能因此忽略它**：两个值各自必须精确匹配，只有 `(H1=384, MCQ=16)` 这一个合法组合。
    故本函数分三段查：
        1. model / tokenizer / hidden_index  —— 两臂逐键相同
        2. K / temperature / top_p / seed    —— 两臂相同（采样协议）
        3. max_new_tokens                    —— **逐臂**与冻结值精确比对
    """
    cfg = cfg or V3AnalysisConfig()
    # 未登记的臂取冻结值；已登记的臂取登记值。**两种情况都是精确比对**，
    # 不存在"有登记就不比"这条路。
    expected_max_new = {arm: expected_max_new_tokens(arm, cfg, budget_registration)
                        for arm in ARMS}
    fps = {arm: run.fingerprint for arm, run in runs.items()}
    mismatches = []
    for key in FINGERPRINT_KEYS:
        values = [json.dumps(fp.get(key), sort_keys=True, ensure_ascii=False)
                  for fp in fps.values()]
        if len(set(values)) != 1:
            mismatches.append(key)

    for param in INVARIANT_GENERATION_PARAMS:
        values = {json.dumps(fp.get("generation_params", {}).get(param), sort_keys=True)
                  for fp in fps.values()}
        if len(values) != 1:
            mismatches.append(f"generation_params.{param}")

    schema_issues = generation_params_schema_mismatch(*fps.values())
    mismatches.extend(schema_issues)
    if not schema_issues and _all_schema_tagged(*fps.values()):
        for param in RECORDED_SAMPLING_PARAMS:
            values = {json.dumps(fp.get("generation_params", {}).get(param), sort_keys=True)
                      for fp in fps.values()}
            if len(values) != 1:
                mismatches.append(f"generation_params.{param}")

    arm_specific: dict[str, Any] = {}
    for arm, expected in expected_max_new.items():
        actual = fps.get(arm, {}).get("generation_params", {}).get("max_new_tokens")
        arm_specific[arm] = {"expected": expected, "actual": actual, "ok": actual == expected}
        if actual != expected:
            mismatches.append(f"generation_params.max_new_tokens[{arm}]")

    h1_fp, mcq_fp = fps[ARM_H1], fps[ARM_MCQ]
    return {
        "ok": not mismatches,
        "mismatched_keys": mismatches,
        "model_hash": h1_fp.get("model", {}).get("model_hash"),
        "tokenizer_hash": h1_fp.get("tokenizer", {}).get("tokenizer_hash"),
        "backend": h1_fp.get("model", {}).get("backend"),
        "invariant_generation_params": list(INVARIANT_GENERATION_PARAMS),
        "recorded_sampling_params": {
            "params": list(RECORDED_SAMPLING_PARAMS),
            "schema": h1_fp.get("generation_params", {}).get("schema"),
            "compared": _all_schema_tagged(*fps.values()) and not schema_issues,
            "rationale": ("两臂都带口径标记时才逐键比较；缺标记的旧指纹其实际取值"
                          "无从得知，**不当作与任意取值兼容**，也不声称比过"),
        },
        "arm_specific_generation_params": {
            "params": list(ARM_SPECIFIC_GENERATION_PARAMS),
            "by_arm": arm_specific,
            "budget_registration": (budget_registration.as_dict()
                                    if budget_registration else None),
            "rationale": ("按冻结设计分臂的解码预算，非 backend invariant；"
                          "未登记时唯一合法组合是冻结值 H1=384 / MCQ=16，"
                          "已登记时该臂取登记值 —— 两种情况**都是逐臂精确比对**，不是忽略"),
        },
        "note": ("G3 通过后，本指纹即成为 G4 与正式生成必须沿用的**生产指纹**。"
                 "若改用远程 GPU 或改变精度，必须**两臂一起**更换并重跑 G3/G4；"
                 "不得混用本地与远程产物。"),
        "arms": {arm: fp.get("run_hash") for arm, fp in fps.items()},
        "_run_hashes_differ_by_arm_is_expected": (h1_fp.get("run_hash")
                                                  != mcq_fp.get("run_hash")),
    }


def evaluate_arm(run: ArmRun, *, n_records: int, criteria: G3Criteria = CRITERIA) -> dict:
    checks = _common_checks(run, criteria=criteria, n_records=n_records)
    checks.update(_h1_checks(run, criteria=criteria) if run.arm == ARM_H1
                  else _mcq_checks(run, criteria=criteria))
    boolean = {k: v for k, v in checks.items()
               if not k.startswith("_") and isinstance(v, bool)}
    return {"checks": checks, "failed_checks": sorted(k for k, v in boolean.items() if not v),
            "passed": all(boolean.values())}


def build_report(*, arm: str, run: ArmRun, verification: dict, evaluation: dict,
                 cfg: V3AnalysisConfig, n_records: int,
                 criteria: G3Criteria = CRITERIA) -> dict:
    """报告必须记录**本次实际使用**的判据，而不是模块默认值。

    先前这里写死 `CRITERIA.as_dict()`：跨模型运行时 `evaluate_arm` 用的是 Llama 的
    32/22/23/4096，而报告会声称用了 Qwen 的 28/19/20/3584 —— 判定与留档对不上，
    等于让报告说谎。默认值仍是 `CRITERIA`，Qwen 行为逐字不变。
    """
    g2 = verification["g2"]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "G3",
        "arm": arm,
        "kind": "model_smoke",
        "gate_only": True,
        "previously_burned": arm == ARM_H1,
        "performance_blind": True,
        "did_not_compute": list(DID_NOT_COMPUTE),
        "prereg_sha256": g2["current_sha256"],
        "hash_mode": g2["hash_mode"],
        "lock_version": g2["locked_version"],
        "sample_manifest_sha256": verification["sample_manifest_sha256"],
        "chain_sha256_verified": verification["chain_sha256_verified"],
        "primary_lists_read": verification["primary_lists_read"],
        "criteria": criteria.as_dict(),
        "n_prompts": n_records,
        "n_traces": run.n_units,
        "traces_per_prompt": cfg.k_traces,
        "n_rows_written": len(run.rows),
        "n_failures": len(run.failures),
        "failures": run.failures[:20],
        "fingerprint": run.fingerprint,
        "hidden_verification": run.hidden_verification,
        "provenance": cfg.provenance(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "elapsed_seconds": round(run.seconds, 1),
        "checks": {k: v for k, v in evaluation["checks"].items() if not k.startswith("_")},
        "detail": {k: v for k, v in evaluation["checks"].items() if k.startswith("_")},
        "failed_checks": evaluation["failed_checks"],
        "passed": evaluation["passed"],
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable))
