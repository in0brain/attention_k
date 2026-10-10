"""Stage 1 正式生成的入口校验、安全续跑与 finalization（v3.1）。

**两层产物**（关键设计，不可省）：

    raw          append-only，**永不回写**。每条 unit 的单次生成字段 + hidden。
                 它是崩溃恢复的唯一真源。
    finalized    仅在**一臂完整结束后**，从全部 raw 行确定性重建。
                 此时才调用 attach_cross_sample_f5 并持久化跨样本 F5。

为什么必须两层：`attach_cross_sample_f5` 只改内存对象，先前 append 的 raw 行永远拿不到
`f5_self_consistency_exact` / `f5_id_agreement_rate`（H1）/ `f5_letter_agreement_rate`（MCQ）。
分段或重启后依赖某次进程的内存状态是错的 —— 这三列只能在全臂跑完后从全部 raw 行重算。

**population 口径（继承 Stage 0 v2.3/v2.4，非新的预注册修改）**：

    H1   raw 9,264 traces（1,544 × 6）→ finalized 9,264 行，K 条共享 group，hidden 按 trace_id
    MCQ  raw 10,560 traces（1,760 × 6）→ finalized **1,760 行**，每题仅 greedy_0 进 population；
         其余 5 条 sampled **只**用于 f5_self_consistency_exact / f5_letter_agreement_rate；
         sampled 的 hidden **不进** finalized npz；hidden 按 example_id 对齐

**hidden 按 eligible 条件判断**（不是每个 done unit 都要有）：

    H1  id_token_positions_local 非空 → 必须有 ； ineligible → 合法地没有
    MCQ eligible_for_primary=true     → 必须有 ； parse failure → 合法地没有

**performance-blind**：不计算 AUROC / AUPRC / Δ / CI / ladder / probe。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_freeze import require_v3_frozen
from recover_attention.evaluation.v3_g3_smoke import (
    ARM_H1,
    ARM_MCQ,
    ARMS,
    DID_NOT_COMPUTE,
    FINGERPRINT_KEYS,
    INVARIANT_GENERATION_PARAMS,
    RECORDED_SAMPLING_PARAMS,
    _all_schema_tagged,
    _read_jsonl,
    _sha256_file,
    expected_max_new_tokens,
    generation_params_schema_mismatch,
)
from recover_attention.evaluation.v3_sample_builder import MANIFEST_NAME, chain_digest
from recover_attention.generation import backends as gb
from recover_attention.generation.manifest import WorkManifest, write_atomic

SCHEMA_VERSION = "stage1_production_v1"
CACHE_SCHEMA = {ARM_H1: "stage1_h1_completion_cache_v1",
                ARM_MCQ: "stage1_mcq_completion_cache_v1"}

#: MCQ finalized 必须持久化的六项（含两项跨样本）。
MCQ_FINALIZED_F5 = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy",
                    "f5_letter_token_logprob", "f5_self_consistency_exact",
                    "f5_letter_agreement_rate")
#: H1 finalized 必须齐全且有限的 14 项冻结 F5（conditional_increment.F5_FEATURE_NAMES）。
H1_FINALIZED_F5 = tuple(ci.F5_FEATURE_NAMES)

#: 已授权的整组技术性排除原因白名单。未在此表内的 reason 一律拒绝 finalize。
AUTHORIZED_EXCLUSION_REASONS = ("group_excluded_due_greedy_forward_nonfinite",
                                "partial_group_excluded_due_sampled_forward_nonfinite")
#: 触发整组排除的 trigger unit 原因白名单。
AUTHORIZED_TRIGGER_REASONS = ("forward_nonfinite",)
#: sampled_j 失败后，**trigger 之后**那批 unit 在 manifest 上的 reason 白名单。
#: 与 `AUTHORIZED_EXCLUSION_REASONS` 层级不同：后者是 audit 记录的**组级分类**，
#: 这里是 sampled_{j+1..K-1} 的**逐 unit manifest 事件 reason**，两者不可混用。
AUTHORIZED_POST_TRIGGER_REASONS = ("group_excluded_due_sampled_forward_nonfinite",)
#: 两种、且**只有**两种被授权的 group 形态。
ACTION_FULL_GROUP = "group_exclusion"                              # 形态 A：6 excluded / 0 raw
#: 形态 B（**已泛化**）：sampled_j 失败 → j 条 done+raw ／ 1 条 trigger ／ K-1-j 条后缀。
#: j = K-1 是历史特例（无后缀），j < K-1 时后缀非空。不另造 action。
ACTION_PARTIAL_GROUP = "exclude_partial_group_from_finalization"
AUTHORIZED_ACTIONS = (ACTION_FULL_GROUP, ACTION_PARTIAL_GROUP)
#: record SHA 的口径：**该行的原始字节，不含行终止符**。
#: 不含终止符是刻意的 —— 终止符是文件级分隔（本机 CRLF），不属于记录内容；
#: 这样 record SHA 不会因写入端的换行策略而漂移。
DIAGNOSTIC_RECORD_SHA_CALIBER = "line_bytes_excluding_line_terminator"


def diagnostic_record_index(path: Path) -> dict[str, str]:
    """{unit_id: 该行原始字节（不含终止符）的 sha256}。重复 unit_id 直接拒绝。"""
    out: dict[str, str] = {}
    if not Path(path).exists():
        return out
    for chunk in Path(path).read_bytes().split(b"\n"):
        line = chunk[:-1] if chunk.endswith(b"\r") else chunk
        if not line.strip():
            continue
        rec = json.loads(line.decode("utf-8"))
        uid = rec.get("unit_id")
        if not uid:
            raise ValueError("diagnostics 行缺 unit_id；拒绝 finalize")
        if uid in out:
            raise ValueError(f"diagnostics 中 unit_id {uid} 重复；拒绝 finalize")
        out[uid] = hashlib.sha256(line).hexdigest()
    return out


def mcq_population_provenance(*, nominal_questions: int, technical_exclusions: int,
                              k_traces: int, preserved_not_analyzed: int = 0) -> dict:
    """MCQ population 出处 —— **双口径**。

    先前这里把 `finalized_rows` 硬编码成 1760。一旦发生 runtime technical attrition，
    那个常量就会和实际 finalized 行数矛盾（报告一处写 1760、一处写 1759）。
    改为按实际排除数派生，nominal 与 actual 同时给出，不允许只写一个。
    """
    retained = nominal_questions - technical_exclusions
    return {
        "rule": "每题仅 greedy_0 进入 finalized population；其余 5 条 sampled 只供跨样本 F5",
        "nominal_questions": nominal_questions,
        "nominal_finalized_rows": nominal_questions,
        "nominal_raw_units": f"{nominal_questions} questions × K={k_traces}",
        "technical_exclusions": technical_exclusions,
        "retained_questions": retained,
        "actual_finalized_rows": retained,
        "retained_raw_units": retained * k_traces,
        "preserved_but_not_analyzed_raw_units": preserved_not_analyzed,
        "expected_total_raw_rows": retained * k_traces + preserved_not_analyzed,
        "analysis_uses": "retained_raw_units",
        "raw_integrity_uses": "expected_total_raw_rows",
        "protocol_deviation": bool(technical_exclusions),
        "classification": ("runtime_technical_attrition" if technical_exclusions
                           else "no_deviation"),
        "hidden_alignment": "example_id（sampled 的 hidden 不进 finalized npz）",
        "inherited_from": "Stage 0 v2.3/v2.4 §3 的 population 定义",
        "preregistration_files_modified": False,
    }

#: 兼容旧引用；nominal 口径（无排除时）的静态视图。
MCQ_POPULATION_PROVENANCE = mcq_population_provenance(
    nominal_questions=1760, technical_exclusions=0, k_traces=6)


def load_authorized_exclusions(arm: str, out_dir: Path, *,
                               frozen_list_sha256: str | None = None,
                               run_fingerprint_sha256: str | None = None) -> dict[str, dict]:
    """读取整组技术性排除的审计真源，返回 {example_id: record}。

    审计文件本身**不足以**授权 —— 调用方还必须核对 manifest 的 latest event
    （见 `_verify_group_exclusion`）。这里只做记录级的完整性与白名单校验。
    """
    p = paths_for(out_dir, arm)["technical_exclusions"]
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("action") not in AUTHORIZED_ACTIONS:
            raise ValueError(
                f"{arm}: 排除审计记录 {rec.get('example_id')} 的 action={rec.get('action')!r} "
                f"不在白名单 {AUTHORIZED_ACTIONS}；拒绝 finalize")
        if rec.get("exclusion_reason") not in AUTHORIZED_EXCLUSION_REASONS:
            raise ValueError(
                f"{arm}: 排除审计记录 {rec.get('example_id')} 的 exclusion_reason="
                f"{rec.get('exclusion_reason')!r} 不在白名单 {AUTHORIZED_EXCLUSION_REASONS}；拒绝 finalize")
        if rec.get("trigger_reason") not in AUTHORIZED_TRIGGER_REASONS:
            raise ValueError(
                f"{arm}: 排除审计记录 {rec.get('example_id')} 的 trigger_reason="
                f"{rec.get('trigger_reason')!r} 不在白名单；拒绝 finalize")
        need = ["example_id", "question_hash", "trigger_unit_id"]
        need += (["affected_unit_ids"] if rec["action"] == ACTION_FULL_GROUP
                 else ["preserved_raw_unit_ids",
                       "diagnostic_unit_id", "diagnostic_record_sha256"])
        if rec["action"] == ACTION_FULL_GROUP and rec.get("diagnostic_persisted"):
            need += ["diagnostic_unit_id", "diagnostic_record_sha256"]
        for field in need:
            if not rec.get(field):
                raise ValueError(f"{arm}: 排除审计记录缺字段 {field}；拒绝 finalize")
        if rec["action"] == ACTION_PARTIAL_GROUP:
            _normalize_partial(arm, rec)      # 记录级形状校验，尽早拒绝
        if frozen_list_sha256 and rec.get("frozen_list_sha256") != frozen_list_sha256:
            raise ValueError(f"{arm}: 排除审计记录的 frozen_list_sha256 与当前冻结名单不符；拒绝 finalize")
        if run_fingerprint_sha256 and rec.get("run_fingerprint_sha256") != run_fingerprint_sha256:
            raise ValueError(f"{arm}: 排除审计记录的 run_fingerprint_sha256 与当前指纹不符；拒绝 finalize")
        if rec["example_id"] in out:
            raise ValueError(f"{arm}: 排除审计记录中 {rec['example_id']} 重复；拒绝 finalize")
        out[rec["example_id"]] = rec
    return out


def _expect(arm: str, latest: dict[str, dict], uid: str, status: str,
            reason: str | None) -> None:
    ev = latest.get(uid)
    if ev is None:
        raise ValueError(f"{arm}: {uid} 未出现在 manifest；拒绝 finalize")
    if ev.get("status") != status:
        raise ValueError(
            f"{arm}: {uid} 的 manifest latest status={ev.get('status')!r}，期望 {status!r}"
            f" —— 混合/部分状态一律拒绝")
    if reason is not None and ev.get("reason") != reason:
        raise ValueError(
            f"{arm}: {uid} 的 manifest latest reason={ev.get('reason')!r}，期望 {reason!r}；拒绝 finalize")


def _verify_diagnostic(arm: str, rec: dict, diag_index: dict[str, str]) -> None:
    """诊断按**记录级** SHA 验证：全文件 SHA 会随未来追加而变，不能作为授权依据。

    `diagnostic_record_index` 已在解析阶段对重复 unit_id 抛错，故这里到手的必然唯一。
    """
    eid = rec["example_id"]
    uid = rec.get("diagnostic_unit_id")
    if uid != rec["trigger_unit_id"]:
        raise ValueError(f"{arm}: {eid} 的 diagnostic_unit_id 与 trigger_unit_id 不符；拒绝 finalize")
    got = diag_index.get(uid)
    if got is None:
        raise ValueError(f"{arm}: {eid} 的 diagnostic 记录 {uid} 在 diagnostics 中不存在；拒绝 finalize")
    if got != rec.get("diagnostic_record_sha256"):
        raise ValueError(
            f"{arm}: {eid} 的 diagnostic 记录字节已变（{got} != 登记值）；拒绝 finalize")


def _str_list(arm: str, rec: dict, field: str, value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{arm}: {rec.get('example_id')} 的 {field} 必须是字符串列表；拒绝 finalize")
    if len(set(value)) != len(value):
        raise ValueError(f"{arm}: {rec.get('example_id')} 的 {field} 含重复 unit；拒绝 finalize")
    return list(value)


def _normalize_partial(arm: str, rec: dict) -> tuple[list[str], str, list[str], str | None]:
    """把新旧两种 partial 记录形状归一化为 (preserved, trigger, suffix, post_reason)。

    新式  `trigger_excluded_unit_id`（**标量字符串**）+ `post_trigger_excluded_unit_ids`（列表）
    旧式  `excluded_unit_ids` == [trigger_unit_id]，无后缀（历史 j = K-1 记录，逐字节不变）
    两式同时出现 → 拒绝：授权语义不得有歧义。

    `post_trigger_exclusion_reason` 从**审计记录本身**读取并按固定白名单校验，
    不依赖调用时代码里的常量取值——否则日后改常量会把既有审计记录重新解释一遍。
    """
    eid = rec.get("example_id")
    preserved = _str_list(arm, rec, "preserved_raw_unit_ids", rec.get("preserved_raw_unit_ids"))
    if not preserved:
        raise ValueError(f"{arm}: {eid} 的 preserved_raw_unit_ids 为空；拒绝 finalize")
    trigger = rec["trigger_unit_id"]
    has_new, has_old = "trigger_excluded_unit_id" in rec, "excluded_unit_ids" in rec
    if has_new and has_old:
        raise ValueError(
            f"{arm}: {eid} 同时含 trigger_excluded_unit_id 与 excluded_unit_ids；"
            f"授权形状有歧义，拒绝 finalize")
    if has_new:
        got = rec["trigger_excluded_unit_id"]
        if not isinstance(got, str):
            raise ValueError(
                f"{arm}: {eid} 的 trigger_excluded_unit_id 必须是标量字符串，"
                f"实为 {type(got).__name__}；拒绝 finalize")
        if got != trigger:
            raise ValueError(f"{arm}: {eid} 的 trigger_excluded_unit_id != trigger_unit_id；拒绝 finalize")
        suffix = _str_list(arm, rec, "post_trigger_excluded_unit_ids",
                           rec.get("post_trigger_excluded_unit_ids", []))
    elif has_old:
        old = _str_list(arm, rec, "excluded_unit_ids", rec["excluded_unit_ids"])
        if old != [trigger]:
            raise ValueError(f"{arm}: {eid} 的 excluded_unit_ids 必须恰为 [trigger_unit_id]；拒绝 finalize")
        if rec.get("post_trigger_excluded_unit_ids"):
            raise ValueError(f"{arm}: {eid} 用旧式字段却带 post_trigger_excluded_unit_ids；拒绝 finalize")
        suffix = []
    else:
        raise ValueError(
            f"{arm}: {eid} 缺 trigger_excluded_unit_id（或旧式 excluded_unit_ids）；拒绝 finalize")
    post_reason = rec.get("post_trigger_exclusion_reason")
    if suffix and not post_reason:
        raise ValueError(
            f"{arm}: {eid} 有 {len(suffix)} 个 post-trigger unit 却缺 "
            f"post_trigger_exclusion_reason；拒绝 finalize")
    if post_reason is not None and post_reason not in AUTHORIZED_POST_TRIGGER_REASONS:
        raise ValueError(
            f"{arm}: {eid} 的 post_trigger_exclusion_reason={post_reason!r} "
            f"不在白名单 {AUTHORIZED_POST_TRIGGER_REASONS}；拒绝 finalize")
    overlap = (set(preserved) & {trigger}) | (set(preserved) & set(suffix)) | ({trigger} & set(suffix))
    if overlap:
        raise ValueError(f"{arm}: {eid} 的 preserved/trigger/suffix 集合相交 {sorted(overlap)}；拒绝 finalize")
    return preserved, trigger, suffix, post_reason


def _partial_expected_units(eid: str, j: int, k: int) -> tuple[list[str], str, list[str]]:
    """sampled_j 通则下该组三段 unit 的**唯一合法**构成。"""
    preserved = [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, j)]
    trigger = f"{eid}__sampled_{j}"
    suffix = [f"{eid}__sampled_{i}" for i in range(j + 1, k)]
    return preserved, trigger, suffix


def _verify_group_exclusion(arm: str, rec: dict, latest: dict[str, dict], k: int,
                            raw_units_of_example: set[str],
                            diag_index: dict[str, str]) -> set[str]:
    """逐 unit 核对 **manifest latest event 与 raw 集合**，audit 说了不算。

    只支持两种形态：
      A `group_exclusion`                        6 excluded ／ raw 恰好 0 条
      B `exclude_partial_group_from_finalization` sampled_j 通则：
          preserved = greedy_0 + sampled_1..j-1（j 条 done+raw）
          trigger   = sampled_j（excluded / forward_nonfinite）
          suffix    = sampled_{j+1}..sampled_{K-1}（excluded / post_trigger_exclusion_reason）
    返回该组「登记覆盖到的 unit 全集」，供 stray-excluded 检查用。
    """
    eid = rec["example_id"]
    if rec["action"] == ACTION_FULL_GROUP:
        units = list(rec["affected_unit_ids"])
        if len(units) != k or len(set(units)) != k:
            raise ValueError(f"{arm}: {eid} 的 affected_unit_ids 不是 {k} 个互异 unit；拒绝 finalize")
        for uid in units:
            _expect(arm, latest, uid, "excluded",
                    rec["trigger_reason"] if uid == rec["trigger_unit_id"] else rec["exclusion_reason"])
        if raw_units_of_example:
            raise ValueError(
                f"{arm}: {eid} 为形态 A（6 excluded / 0 raw），但 raw 中仍有 "
                f"{len(raw_units_of_example)} 条；拒绝 finalize")
        # diagnostic_persisted=false 是**合法**的历史情形（诊断持久化补丁之前发生的排除，
        # 如首次的 cybermetric_005396）：跳过诊断校验，不因此否定该组。
        if rec.get("diagnostic_persisted"):
            _verify_diagnostic(arm, rec, diag_index)
        return set(units)

    # ── 形态 B（泛化的 sampled_j 通则）─────────────────────────────────────────
    preserved, trigger, suffix, post_reason = _normalize_partial(arm, rec)
    j = len(preserved)                       # preserved = greedy_0 + sampled_1..j-1 ⇒ j 条
    if not 1 <= j <= k - 1:
        raise ValueError(
            f"{arm}: {eid} 的 preserved 有 {j} 条，超出 sampled_j 通则允许的 1..{k - 1}；拒绝 finalize")
    exp_pre, exp_trig, exp_suf = _partial_expected_units(eid, j, k)
    if preserved != exp_pre:
        raise ValueError(
            f"{arm}: {eid} 的 preserved_raw_unit_ids 不是从 greedy_0 起的连续前缀"
            f"（期望 {exp_pre[:3]}… 共 {j} 条）；拒绝 finalize")
    if trigger != exp_trig:
        raise ValueError(
            f"{arm}: {eid} 的 trigger_unit_id={trigger!r}，按 preserved 长度应为 {exp_trig!r}；拒绝 finalize")
    if suffix != exp_suf:
        raise ValueError(
            f"{arm}: {eid} 的 post-trigger unit 集合 {suffix} != 期望 {exp_suf}；拒绝 finalize")
    covered = set(preserved) | {trigger} | set(suffix)
    if len(covered) != k:
        raise ValueError(f"{arm}: {eid} 三段合计 {len(covered)} 个 unit，应为 K={k}；拒绝 finalize")
    for uid in preserved:
        _expect(arm, latest, uid, "done", None)
    _expect(arm, latest, trigger, "excluded", rec["trigger_reason"])
    for uid in suffix:
        # reason 取自**审计记录**的 post_trigger_exclusion_reason（已按白名单校验），
        # 不取代码常量：审计记录才是授权真源。
        _expect(arm, latest, uid, "excluded", post_reason)
    # raw 集合必须**精确等于** preserved：缺一条、多一条都拒绝
    if raw_units_of_example != set(preserved):
        missing = sorted(set(preserved) - raw_units_of_example)
        extra = sorted(raw_units_of_example - set(preserved))
        raise ValueError(
            f"{arm}: {eid} 为形态 B，raw 集合与 preserved_raw_unit_ids 不符"
            f"（缺 {missing[:3]}，多 {extra[:3]}）；拒绝 finalize")
    _verify_diagnostic(arm, rec, diag_index)
    return covered


# ══════════════════════════════════════════════════════════════════════════════
# 文件名（精确）
# ══════════════════════════════════════════════════════════════════════════════
def paths_for(out_dir: Path, arm: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        # raw —— append-only，永不回写
        "manifest": out_dir / f"{arm}_production_manifest.jsonl",
        "raw_results": out_dir / f"{arm}_raw_unit_results.jsonl",
        "hidden_dir": out_dir / f"{arm}_unit_hidden",
        "fingerprint": out_dir / f"{arm}_run_fingerprint.json",
        "segment_log": out_dir / f"{arm}_segment_log.jsonl",
        "progress": out_dir / f"{arm}_production_progress.json",
        # 审计真源 —— append-only，只由 production 写
        "diagnostics": out_dir / f"{arm}_forward_nonfinite_diagnostics.jsonl",
        "technical_exclusions": out_dir / f"{arm}_technical_exclusions.jsonl",
        # finalized —— 一臂完整结束后确定性重建
        "records": out_dir / f"{arm}_completion_records.jsonl",
        "hidden_npz": out_dir / f"{arm}_hidden.npz",
        "finalization_report": out_dir / f"{arm}_finalization_report.json",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 启动前校验：G2 / G3 / G4 / G5 + 指纹
# ══════════════════════════════════════════════════════════════════════════════
def requires_hidden(arm: str, row: dict) -> bool:
    """该行是否**必须**有 hidden。ineligible 行没有 hidden 是合法终态，不是数据丢失。"""
    if arm == ARM_H1:
        return bool(row.get("id_token_positions_local"))
    return bool(row.get("eligible_for_primary"))


def verify_production_inputs(*, samples_dir: Path, g3_dir: Path, g4_dir: Path,
                             prereg: Path, lock: Path, cfg: V3AnalysisConfig,
                             arm: str) -> dict:
    """任一不过即抛错，且**不加载模型、不创建任何文件**。"""
    freeze = require_v3_frozen(prereg, lock)                       # G2

    def _gate(path: Path, name: str) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("passed"):
            raise ValueError(f"{name} 未通过（{path}），不得进入正式生成")
        return payload

    g3 = _gate(g3_dir / "g3_summary.json", "G3")                   # G3
    g4 = _gate(g4_dir / "g4_summary.json", "G4")                   # G4
    if not all(g4.get("arms_passed", {}).values()):
        raise ValueError(f"G4 两臂未全部通过：{g4.get('arms_passed')}")

    manifest_path = samples_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["prereg"]["sha256"] != freeze.current_sha256:
        raise ValueError("sample manifest 的 prereg sha256 与当前不一致")
    if manifest["prereg"]["lock_version"] != freeze.locked_version:
        raise ValueError("sample manifest 的 lock_version 与当前 lock 不一致")

    # G5：正式名单 chain hash + 唯一性
    filename = ("h1_primary_entries.jsonl" if arm == ARM_H1
                else "mcq_primary_questions.jsonl")
    key = "source_entry_id" if arm == ARM_H1 else "question_hash"
    rows = _read_jsonl(samples_dir / filename)
    actual = chain_digest(r[key] for r in rows)
    expected = manifest["chain_sha256"][f"{arm}_primary"]
    if actual != expected:
        raise ValueError(f"{filename} 的 chain hash 与 manifest 不符：{actual} != {expected}")

    expected_n = cfg.n_h1_prompts if arm == ARM_H1 else cfg.n_mcq_questions
    if len(rows) != expected_n:
        raise ValueError(f"{arm} 正式名单 {len(rows)} 条 != 冻结的 {expected_n}")
    if any(r.get("gate_only") for r in rows):
        raise ValueError(f"{arm} 正式名单里混入了 gate_only 行")

    uniqueness: dict[str, int] = {}
    if arm == ARM_H1:
        for field_name in ("source_entry_id", "g5_prompt_hash", "group_id", "example_id"):
            values = [r[field_name] for r in rows]
            if len(set(values)) != len(values):
                raise ValueError(f"H1 正式名单的 {field_name} 存在重复")
            uniqueness[field_name] = len(set(values))
    else:
        values = [r["question_hash"] for r in rows]
        if len(set(values)) != len(values):
            raise ValueError("MCQ 正式名单的 question_hash 存在重复")
        uniqueness["question_hash"] = len(set(values))

    # 与 gate 名单零交集
    for gate_name in ("G3", "G4"):
        gate_file = (f"{arm}_g3_{'entries' if arm == ARM_H1 else 'questions'}.jsonl"
                     if gate_name == "G3"
                     else f"{arm}_g4_{'entries' if arm == ARM_H1 else 'questions'}.jsonl")
        gate_rows = _read_jsonl(samples_dir / gate_file)
        overlap = {r[key] for r in rows} & {r[key] for r in gate_rows}
        if overlap:
            raise ValueError(f"{arm} 正式名单与 {gate_name} 名单交集非空：{sorted(overlap)[:5]}")

    return {
        "g2": freeze.as_dict(),
        "g3_passed": True, "g4_passed": True,
        "g4_arms_passed": g4["arms_passed"],
        "sample_manifest_sha256": _sha256_file(manifest_path),
        "primary_chain_sha256": actual,
        "n_primary": len(rows),
        "uniqueness": uniqueness,
        "gate_lists_disjoint": True,
        "g3_production_fingerprint": json.loads(
            (g3_dir / f"{arm}_g3_smoke_report.json").read_text(encoding="utf-8"))["fingerprint"],
        "_primary_rows": rows,
    }


def check_fingerprint(arm: str, fingerprint: dict, reference: dict, *,
                      cfg: V3AnalysisConfig, budget_registration=None) -> dict:
    """三段式：model/tokenizer/hidden_index 全同 + 采样协议全同 + 逐臂 max_new_tokens 精确。"""
    mismatches = []
    for key in FINGERPRINT_KEYS:
        if json.dumps(fingerprint.get(key), sort_keys=True) != \
           json.dumps(reference.get(key), sort_keys=True):
            mismatches.append(key)
    for param in INVARIANT_GENERATION_PARAMS:
        if fingerprint.get("generation_params", {}).get(param) != \
           reference.get("generation_params", {}).get(param):
            mismatches.append(f"generation_params.{param}")

    # 新旧口径不得混用：旧指纹没有 top_k / repetition_penalty，其实际生效值无从得知，
    # **不能**当成"与任意取值兼容"。双方同口径时才逐键比这两个采样参数。
    schema_issues = generation_params_schema_mismatch(fingerprint, reference)
    mismatches.extend(schema_issues)
    if not schema_issues and _all_schema_tagged(fingerprint, reference):
        for param in RECORDED_SAMPLING_PARAMS:
            if fingerprint.get("generation_params", {}).get(param) != \
               reference.get("generation_params", {}).get(param):
                mismatches.append(f"generation_params.{param}")

    expected = expected_max_new_tokens(arm, cfg, budget_registration)
    actual = fingerprint.get("generation_params", {}).get("max_new_tokens")
    if actual != expected:
        mismatches.append(f"generation_params.max_new_tokens[{arm}]")
    # 参照指纹（上一段 / G3 生产指纹）必须**也**是这个预算，否则两段跑在不同预算上。
    ref_actual = reference.get("generation_params", {}).get("max_new_tokens")
    if ref_actual != expected:
        mismatches.append(f"reference.generation_params.max_new_tokens[{arm}]")
    return {"ok": not mismatches, "mismatched_keys": mismatches,
            "max_new_tokens": {"expected": expected, "actual": actual,
                               "reference_actual": ref_actual},
            "budget_registration": (budget_registration.as_dict()
                                    if budget_registration else None)}


def bind_run_fingerprint(path: Path, fingerprint: dict, arm: str, *,
                         cfg: V3AnalysisConfig, budget_registration=None) -> dict:
    """首段**原子创建**；后续段只读并逐项比较，**绝不覆盖重写**。

    每段都覆盖会让"跨段后端漂移"这件事永远查不出来 —— 覆盖之后就没有对照物了。
    """
    if not path.exists():
        write_atomic(path, json.dumps(fingerprint, ensure_ascii=False, indent=2))
        return {"created": True, "matches_previous_segment": None, "mismatched_keys": []}
    previous = json.loads(path.read_text(encoding="utf-8"))
    check = check_fingerprint(arm, fingerprint, previous, cfg=cfg,
                              budget_registration=budget_registration)
    if not check["ok"]:
        raise ValueError(
            f"{arm}: 本段的 run fingerprint 与首段不一致：{check['mismatched_keys']}。"
            f"正式生成不得跨段更换模型/量化/采样协议；停。")
    return {"created": False, "matches_previous_segment": True, "mismatched_keys": []}


# ══════════════════════════════════════════════════════════════════════════════
# 恢复审计（crash-point / orphan）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ResumeAudit:
    n_done: int
    n_rows: int
    n_hidden_files: int
    repaired_done: list[str]
    orphan_hidden: list[str]

    def as_dict(self) -> dict:
        return {"n_done": self.n_done, "n_rows": self.n_rows,
                "n_hidden_files": self.n_hidden_files,
                "repaired_done": self.repaired_done,
                "orphan_hidden": self.orphan_hidden,
                "note": ("repaired_done = result 与所需 hidden 都在、仅 manifest 未标 done，"
                         "已补记且**未重新调用模型**")}


def audit_resume_state(arm: str, out_dir: Path, *, repair: bool = True) -> ResumeAudit:
    """四种崩溃状态的处置。任一不可修复即抛错，**不覆盖、不删除、不静默重跑**。"""
    p = paths_for(out_dir, arm)
    rows: dict[str, dict] = {}
    duplicates: list[str] = []
    if p["raw_results"].exists():
        for line in p["raw_results"].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["unit_id"] in rows:
                duplicates.append(row["unit_id"])
            rows[row["unit_id"]] = row
    if duplicates:
        raise ValueError(
            f"{arm}: raw 结果里出现重复 unit_id {sorted(set(duplicates))[:5]} —— "
            f"append-only 日志不该有重复；停，不自动去重。")

    hidden_files = ({q.stem for q in p["hidden_dir"].iterdir() if q.suffix == ".npy"}
                    if p["hidden_dir"].exists() else set())

    man = WorkManifest(p["manifest"])
    status = man.current_status()
    done = {uid for uid, st in status.items() if st == "done"}

    # (a) manifest done 但 result 缺失 → 停
    missing_rows = sorted(done - set(rows))
    if missing_rows:
        raise ValueError(
            f"{arm}: manifest 标 done 但 raw 结果缺行：{missing_rows[:5]} —— "
            f"这是把数据丢失伪装成'已完成'，停。")

    # (b) manifest done 但**所需** hidden 缺失 → 停（ineligible 无 hidden 合法）
    missing_hidden = sorted(uid for uid in done
                            if requires_hidden(arm, rows[uid]) and uid not in hidden_files)
    if missing_hidden:
        raise ValueError(
            f"{arm}: 以下 unit 是 eligible 却没有 hidden：{missing_hidden[:5]} —— 停。")

    # (c) hidden 有但 result 无 → orphan，停
    orphan = sorted(hidden_files - set(rows))
    if orphan:
        raise ValueError(
            f"{arm}: 发现 orphan hidden（有 .npy 无 result 行）：{orphan[:5]} —— "
            f"不覆盖、不删除、不静默重跑；请人工核查后再继续。")

    # (d) result 有 + 所需 hidden 有 + manifest 未 done → 校验后只补记 done
    repaired: list[str] = []
    for uid, row in sorted(rows.items()):
        if uid in done:
            continue
        if requires_hidden(arm, row) and uid not in hidden_files:
            raise ValueError(f"{arm}: {uid} 有 result 但缺所需 hidden，无法补记 done；停。")
        if repair:
            man.mark(uid, "done", note="repaired_from_raw_without_regeneration")
            repaired.append(uid)

    return ResumeAudit(n_done=len(done) + len(repaired), n_rows=len(rows),
                       n_hidden_files=len(hidden_files), repaired_done=repaired,
                       orphan_hidden=[])


# ══════════════════════════════════════════════════════════════════════════════
# finalization（一臂完整结束后）
# ══════════════════════════════════════════════════════════════════════════════
def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


#: 允许写入 finalized 记录的 population role。**默认仍是 confirmatory**，
#: 新增的复现角色必须由调用方显式传入 —— 让"这批数据是不是确证样本"这件事
#: 无法靠沉默取得。
ROLE_CONFIRMATORY = "confirmatory"
ROLE_CROSS_MODEL_REPLICATION = "cross_model_replication_non_confirmatory"
ALLOWED_POPULATION_ROLES: tuple[str, ...] = (ROLE_CONFIRMATORY,
                                             ROLE_CROSS_MODEL_REPLICATION)


def finalize_arm(arm: str, out_dir: Path, records: list[dict], *,
                 cfg: V3AnalysisConfig, run_fingerprint: dict,
                 frozen_list_sha256: str | None = None,
                 population_role: str = ROLE_CONFIRMATORY) -> dict:
    """从**全部 raw 行**确定性重建 finalized 产物。raw 三份一律不动。

    整组技术性排除**只有在审计真源登记 + manifest latest event 逐项吻合**时才被接受；
    未登记、部分排除、未知 reason 一律拒绝。绝不生成合成占位行。

    `population_role` 默认 `confirmatory`（Qwen 行为不变）；跨模型复现必须显式传
    `cross_model_replication_non_confirmatory`。白名单之外一律拒绝 —— 任意字符串
    都能写进产物的话，"角色"就不再是可审计的字段。
    """
    if population_role not in ALLOWED_POPULATION_ROLES:
        raise ValueError(
            f"population_role={population_role!r} 不在白名单 "
            f"{list(ALLOWED_POPULATION_ROLES)} 内；拒绝 finalize")
    p = paths_for(out_dir, arm)
    raw_rows = _read_jsonl(p["raw_results"])
    by_unit = {r["unit_id"]: r for r in raw_rows}
    if len(by_unit) != len(raw_rows):
        raise ValueError(f"{arm}: raw 结果含重复 unit_id，拒绝 finalize")

    # ── 授权门：audit 真源 + manifest latest event 双向核对 ────────────────────
    fp_sha = _sha256_file(p["fingerprint"]) if p["fingerprint"].exists() else None
    authorized = load_authorized_exclusions(
        arm, out_dir, frozen_list_sha256=frozen_list_sha256, run_fingerprint_sha256=fp_sha)
    latest: dict[str, dict] = {}
    if p["manifest"].exists():
        for line in p["manifest"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                ev = json.loads(line)
                latest[ev["unit_id"]] = ev
    excluded_units = {uid for uid, ev in latest.items() if ev.get("status") == "excluded"}
    known_ids = {r["example_id"] for r in records}
    by_example: dict[str, list[dict]] = {}
    for row in raw_rows:
        by_example.setdefault(row["example_id"], []).append(row)
    diag_index = diagnostic_record_index(p["diagnostics"])
    authorized_units: set[str] = set()
    for eid, rec in authorized.items():
        if eid not in known_ids:
            raise ValueError(f"{arm}: 排除审计记录的 {eid} 不在冻结名单中；拒绝 finalize")
        authorized_units |= _verify_group_exclusion(
            arm, rec, latest, cfg.k_traces,
            {r["unit_id"] for r in by_example.get(eid, [])}, diag_index)
    stray = excluded_units - authorized_units
    if stray:
        raise ValueError(
            f"{arm}: manifest 里有 **未登记** 的 excluded unit {sorted(stray)[:5]}"
            f"（共 {len(stray)} 个）—— 未经审计授权的排除不得 finalize")

    group_of = {r["example_id"]: r.get("group_id") for r in records}
    n_excluded = len(authorized)
    retained_records = [r for r in records if r["example_id"] not in authorized]
    retained_raw_units = cfg.k_traces * len(retained_records)
    # 形态 B 保留的 raw：**永久留在 raw 文件里作审计，但绝不参与 F5 与 finalization**。
    preserved_not_analyzed = sum(len(rec.get("preserved_raw_unit_ids", []))
                                 for rec in authorized.values()
                                 if rec["action"] == ACTION_PARTIAL_GROUP)
    expected_total_raw = retained_raw_units + preserved_not_analyzed
    if len(raw_rows) != expected_total_raw:
        raise ValueError(
            f"{arm}: raw 有 {len(raw_rows)} 条 unit，期望 {expected_total_raw}"
            f"（retained {len(retained_records)} × K={cfg.k_traces} = {retained_raw_units}"
            f" + 保留未分析 {preserved_not_analyzed}；nominal {len(records)}，"
            f"技术性排除 {n_excluded}）—— 该臂尚未跑完，不得 finalize")

    # **先按已授权的排除组整组滤掉 raw，再算跨样本 F5**。
    # 顺序不能反：形态 B 的组只剩 5 条，若先 attach 就会让 K=5 混进
    # f5_self_consistency_exact / f5_letter_agreement_rate，污染这两维。
    analyzed_rows = [r for r in raw_rows if r["example_id"] not in authorized]
    bad = {eid: len(v) for eid, v in by_example.items()
           if eid not in authorized and len(v) != cfg.k_traces}
    if bad:
        raise ValueError(f"{arm}: 以下 example 的 unit 数 != K={cfg.k_traces}：{list(bad.items())[:5]}")

    rows_for_f5 = [dict(r) for r in analyzed_rows]
    gb.attach_cross_sample_f5(rows_for_f5, arm)
    enriched = {r["unit_id"]: r for r in rows_for_f5}

    finalized: list[dict] = []
    hidden: dict[str, np.ndarray] = {}
    for unit_id in sorted(enriched):
        row = enriched[unit_id]
        eid = row["example_id"]
        if arm == ARM_MCQ and row.get("sample_type") != "greedy":
            continue                      # MCQ population = 每题仅 greedy_0
        key = unit_id if arm == ARM_H1 else eid
        record = {"schema_version": CACHE_SCHEMA[arm],
                  **({"trace_id": unit_id} if arm == ARM_H1 else {}),
                  "example_id": eid, "group_id": group_of.get(eid),
                  "population_role": population_role,
                  "run_hash": run_fingerprint.get("run_hash"),
                  **{k: v for k, v in row.items() if k != "unit_id"}}
        if record["group_id"] is None:
            raise ValueError(f"{arm}: {eid} 在冻结名单里没有 group_id")
        finalized.append(record)
        if requires_hidden(arm, row):
            vec_path = p["hidden_dir"] / f"{unit_id}.npy"
            if not vec_path.exists():
                raise ValueError(f"{arm}: eligible 记录 {unit_id} 缺 hidden，拒绝 finalize")
            hidden[key] = np.load(vec_path)

    expected_rows = len(retained_records) if arm == ARM_MCQ else retained_raw_units
    if len(finalized) != expected_rows:
        raise ValueError(f"{arm}: finalized {len(finalized)} 行 != 期望 {expected_rows}")

    required = H1_FINALIZED_F5 if arm == ARM_H1 else MCQ_FINALIZED_F5
    analyzable = [r for r in finalized if requires_hidden(arm, r)]
    missing_f5: dict[str, int] = {}
    for name in required:
        n_bad = sum(1 for r in analyzable if not _finite(r.get(name)))
        if n_bad:
            missing_f5[name] = n_bad
    if missing_f5:
        raise ValueError(f"{arm}: 以下冻结 F5 列在 analyzable 记录上缺失或非有限：{missing_f5}")

    write_atomic(p["records"], "".join(
        json.dumps(r, ensure_ascii=False, default=lambda o: o.tolist()) + "\n"
        for r in finalized))
    np.savez_compressed(p["hidden_npz"], **hidden)

    report = {
        "schema_version": "stage1_finalization_report_v1",
        "arm": arm,
        "population_role": population_role,
        "is_confirmatory": population_role == ROLE_CONFIRMATORY,
        "performance_blind": True,
        "did_not_compute": list(DID_NOT_COMPUTE),
        "n_raw_units": len(raw_rows),
        "n_finalized_records": len(finalized),
        "n_analyzable_records": len(analyzable),
        "n_hidden_vectors": len(hidden),
        "hidden_alignment_key": "trace_id" if arm == ARM_H1 else "example_id",
        "k_traces_per_example_verified": True,
        "frozen_f5_verified": list(required),
        "hidden_required_only_for_analyzable": True,
        # **双口径**：nominal 与 actual 必须同时出现，不允许一处 1760 一处 1759。
        "nominal_questions": len(records),
        "nominal_finalized_rows": (len(records) if arm == ARM_MCQ
                                   else cfg.k_traces * len(records)),
        "nominal_raw_units": cfg.k_traces * len(records),
        "technical_exclusions": n_excluded,
        "retained_questions": len(retained_records),
        "actual_finalized_rows": len(finalized),
        # 三口径必须同时出现：物理审计用 n_raw_units；F5 与分析只能用 retained_raw_units；
        # preserved_but_not_analyzed 只进数据账目，绝不进 F5 或分析。
        "retained_raw_units": retained_raw_units,
        "preserved_but_not_analyzed_raw_units": preserved_not_analyzed,
        "expected_total_raw_rows": expected_total_raw,
        "analysis_uses": "retained_raw_units",
        "raw_integrity_uses": "expected_total_raw_rows",
        "protocol_deviation": bool(n_excluded),
        "classification": ("runtime_technical_attrition" if n_excluded else "no_deviation"),
        "exclusion_audit_sha256": (_sha256_file(p["technical_exclusions"])
                                   if p["technical_exclusions"].exists() else None),
        "excluded_example_ids": sorted(authorized),
        "preregistration_files_modified": False,
        "population_provenance": (
            mcq_population_provenance(nominal_questions=len(records),
                                      technical_exclusions=n_excluded,
                                      k_traces=cfg.k_traces,
                                      preserved_not_analyzed=preserved_not_analyzed)
            if arm == ARM_MCQ else {
                "rule": "K 条 completion 全部进入 population 并共享 group",
                "nominal_questions": len(records),
                "nominal_raw_units": cfg.k_traces * len(records),
                "technical_exclusions": n_excluded,
                "retained_questions": len(retained_records),
                "actual_finalized_rows": len(finalized),
                "retained_raw_units": retained_raw_units,
                "preserved_but_not_analyzed_raw_units": preserved_not_analyzed,
                "expected_total_raw_rows": expected_total_raw,
                "protocol_deviation": bool(n_excluded),
                "classification": ("runtime_technical_attrition" if n_excluded
                                   else "no_deviation"),
                "hidden_alignment": "trace_id",
                "inherited_from": "Stage 0 v2.4 §3",
                "preregistration_files_modified": False}),
        "raw_untouched": True,
        "run_fingerprint": run_fingerprint,
        "provenance": cfg.provenance(),
    }
    write_atomic(p["finalization_report"], json.dumps(report, ensure_ascii=False, indent=2))
    return report


__all__ = ["SCHEMA_VERSION", "ARMS", "ARM_H1", "ARM_MCQ", "paths_for",
           "verify_production_inputs", "check_fingerprint", "bind_run_fingerprint",
           "requires_hidden", "audit_resume_state", "ResumeAudit", "finalize_arm",
           "MCQ_FINALIZED_F5", "H1_FINALIZED_F5", "MCQ_POPULATION_PROVENANCE",
           "mcq_population_provenance", "load_authorized_exclusions",
           "AUTHORIZED_EXCLUSION_REASONS", "AUTHORIZED_TRIGGER_REASONS"]
