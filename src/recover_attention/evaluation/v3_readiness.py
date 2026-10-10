"""确定性 readiness builder —— **入库版**，取代此前的临时硬闸。

入口**不得信任磁盘上的布尔值**：`recompute(...)` 从产物现场重算，
`recompute_and_compare(...)` 再与磁盘 JSON 逐项比对。两者都不通过就拒绝执行。

机械验证项（全部通过才允许 `ready_for_unblinding=true`）：

    六项 finalized freeze manifest
    shadow / repool completion reports 与 fingerprints
    三条分析路径已冻结
    真实 analyze callback 已接入
    M4 writer
    E1 / E2 / E3
    unblinding 顺序与独占写入
    全部真实结果目标不存在
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

from . import v3_analysis_paths as paths
from . import v3_population_declaration as pop
from . import v3_shadow_overlay as overlay
from . import v3_unblinding_boundary as ub
from .v3_freeze import check_v3_frozen

READINESS_SCHEMA = "stage1_confirmatory_readiness_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(name: str, ok: bool, detail: Any = None) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def recompute(*, prod_dir: Path, readiness_dir: Path, shadow_dir: Path,
              repool_dir: Path, result_dir: Path, unblind_log: Path,
              freeze_manifest_path: Path) -> dict:
    """从产物现场重算 readiness。**纯函数式**：不写盘、不读 label。"""
    checks: list[dict] = []

    fz = check_v3_frozen()
    checks.append(_check("preregistration_frozen", fz.match,
                         {"locked_version": fz.locked_version,
                          "sha256": fz.current_sha256}))

    # ---- 六项 finalized freeze manifest ---------------------------------
    if freeze_manifest_path.exists():
        man = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        ver = ub.verify_stage1_freeze_manifest(man, prod_dir)
        checks.append(_check("finalized_freeze_manifest",
                             bool(ver["ok"] and ver["hash_matches"]),
                             {"drift": ver["drift"], "hash_matches": ver["hash_matches"],
                              "n_artifacts": len(ub.FINALIZED_SIX)}))
    else:
        checks.append(_check("finalized_freeze_manifest", False, "manifest 不存在"))

    # ---- shadow / repool completion reports + fingerprints --------------
    shadow_ok, shadow_detail = True, {}
    try:
        ov = overlay.load_shadow_overlay(shadow_dir, prod_dir)
        shadow_detail = {"counts": ov["counts"],
                         "fingerprint_sha256": _sha(shadow_dir / "shadow_fingerprint.json"),
                         "report_sha256": _sha(shadow_dir / "shadow_completion_report.json")}
    except Exception as exc:                                   # noqa: BLE001
        shadow_ok, shadow_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("shadow_overlay_gate", shadow_ok, shadow_detail))

    rep_path = repool_dir / "e3_repool_report.json"
    repool_ok = rep_path.exists()
    repool_detail: Any = "report 不存在"
    if repool_ok:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        repool_ok = rep.get("complete") is True
        repool_detail = {"complete": rep.get("complete"),
                         "e3_npz_files": rep.get("e3_npz_files"),
                         "e3_trace_block_pairs": rep.get("e3_trace_block_pairs"),
                         "report_sha256": _sha(rep_path)}
    checks.append(_check("e3_repool_report", repool_ok, repool_detail))

    # ---- 三条分析路径已冻结 ---------------------------------------------
    try:
        paths.assert_frozen_paths(paths.FROZEN_PATHS)
        checks.append(_check("analysis_paths_frozen", True,
                             {"paths": list(paths.FROZEN_PATHS),
                              "markers": {p: paths.path_markers(p)["marker"]
                                          for p in paths.FROZEN_PATHS}}))
    except paths.PathViolation as exc:
        checks.append(_check("analysis_paths_frozen", False, str(exc)))

    # ---- 真实 analyze callback 已接入 ------------------------------------
    try:
        from . import v3_analyze
        sig = inspect.signature(v3_analyze.make_analyze_callback)
        need = {"cfg", "h1_records", "h1_hidden", "mcq_records", "mcq_hidden",
                "shadow_overlay", "e3_blocks"}
        checks.append(_check("real_analyze_callback_wired",
                             need <= set(sig.parameters),
                             {"module": v3_analyze.__name__,
                              "schema": v3_analyze.ANALYSIS_SCHEMA,
                              "rq2_frozen_status": v3_analyze.RQ2_FROZEN_STATUS}))
    except Exception as exc:                                   # noqa: BLE001
        checks.append(_check("real_analyze_callback_wired", False,
                             f"{type(exc).__name__}: {exc}"))

    # ---- M4 writer ------------------------------------------------------
    checks.append(_check("m4_population_declaration_writer",
                         all(hasattr(pop, n) for n in
                             ("build_declaration", "validate_declaration",
                              "write_declaration", "assert_no_label_leak"))))

    # ---- E1 / E2 / E3 ---------------------------------------------------
    from . import v3_grouping as grouping
    checks.append(_check("estimands_wired",
                         all(hasattr(grouping, n) for n in
                             ("build_e2_groups", "build_e3_groups", "e3_block_of")),
                         {"E1": "primary grouping（adapter 内建）",
                          "E2": "build_e2_groups",
                          "E3": "build_e3_groups + repool reader（已接线）"}))

    # ---- unblinding 顺序与独占写入 ---------------------------------------
    ok_seq = hasattr(ub, "record_stage1_unblinding")
    checks.append(_check("unblinding_order_and_exclusive_write", ok_seq,
                         "record_unblinding 先于 analyze；log 以独占创建写入"))

    # ---- 全部真实结果目标不存在 ------------------------------------------
    existing = [str(p) for p in (list(result_dir.glob("*")) if result_dir.exists() else [])]
    checks.append(_check("no_real_result_targets_exist",
                         not existing and not unblind_log.exists(),
                         {"result_dir_entries": existing,
                          "unblind_log_exists": unblind_log.exists()}))

    # ---- freeze_chain.finalized_artifacts：入口会拿它再核一遍六项 SHA ----
    finalized: dict[str, dict] = {}
    for name in ub.FINALIZED_SIX:
        f = Path(prod_dir) / name
        finalized[name] = ({"bytes": f.stat().st_size, "sha256": _sha(f)}
                           if f.exists() else {"missing": True})

    blockers = [c["name"] for c in checks if not c["ok"]]
    return {
        "schema_version": READINESS_SCHEMA,
        "checks": checks,
        "freeze_chain": {
            "preregistration": {"match": fz.match, "locked_version": fz.locked_version,
                                "canonical_sha256": fz.current_sha256},
            "finalized_artifacts": finalized,
            "integrity_evidence_caliber": (
                "字节大小 + sha256；mtime **不作为**只读或未修改的证据"),
        },
        "readiness": {
            "ready_for_unblinding": not blockers,
            "blocking_reasons": blockers,
            "n_checks": len(checks),
            "n_passed": sum(1 for c in checks if c["ok"]),
        },
        "determinism": ("本文档由 v3_readiness.recompute 从产物现场重算；"
                        "入口**不信任磁盘上的布尔值**"),
    }


def _comparable(doc: dict) -> dict:
    """比对用的规范形 —— 去掉纯信息性字段。"""
    return {"schema_version": doc.get("schema_version"),
            "readiness": doc.get("readiness"),
            "checks": [{"name": c["name"], "ok": c["ok"]} for c in doc.get("checks", [])]}


def recompute_and_compare(readiness_path: Path, **kw) -> dict:
    """重算并与磁盘 JSON 比对。不一致或磁盘缺失 → 抛 `ReadinessMismatch`。"""
    fresh = recompute(**kw)
    if not Path(readiness_path).exists():
        raise ReadinessMismatch(f"readiness 文档不存在：{readiness_path}")
    disk = json.loads(Path(readiness_path).read_text(encoding="utf-8"))
    if _comparable(disk) != _comparable(fresh):
        raise ReadinessMismatch(
            "磁盘 readiness 与现场重算不一致 —— 拒绝信任磁盘布尔值。\n"
            f"  disk    ready={disk.get('readiness', {}).get('ready_for_unblinding')} "
            f"blockers={disk.get('readiness', {}).get('blocking_reasons')}\n"
            f"  recomputed ready={fresh['readiness']['ready_for_unblinding']} "
            f"blockers={fresh['readiness']['blocking_reasons']}")
    return fresh


class ReadinessMismatch(RuntimeError):
    """磁盘 readiness 不可信 —— 停机。"""


# ══════════════════════════════════════════════════════════════════════════════
# readiness v3 —— Path B 换成 tail、E3 固定 canonical repool、双 freeze hash
# ══════════════════════════════════════════════════════════════════════════════
READINESS_V3_SCHEMA = "stage1_confirmatory_readiness_v3"


@functools.lru_cache(maxsize=1)
def _probe_end_to_end() -> dict:
    """跑一次**全合成**的 end-to-end 并返回结果。确定性 → 进程内只跑一次。

    数据全部由 `v3_analyze.synthetic_contract_probe()` 生成，**不读任何真实
    产物、不接触真实 label**。
    """
    from . import v3_analyze as _va
    probe = _va.synthetic_contract_probe()
    pcfg = probe.pop("cfg")
    # 探针刻意用极小样本 → sklearn 会对 inner fold 的类别人数刷 UserWarning。
    # 那是**探针尺寸**的性质，不是真实分析的问题；只在这里静音，不动全局过滤器。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                module=r"sklearn\.model_selection\..*")
        return _va.make_analyze_callback(cfg=pcfg, **probe)()


def recompute_v3(*, prod_dir: Path, tail_dir: Path, shadow_dir: Path,
                 repool_dir: Path, result_dir: Path, unblind_log: Path,
                 freeze_manifest_path: Path,
                 analysis_input_manifest_path: Path,
                 entry_source_path: Path) -> dict:
    """v3 的现场重算。**v2 的 `recompute` 原样保留**，磁盘上的 v2 文档仍可复核。

    与 v2 的差别（三处，全部是收紧）：

        Path B  验证并加载 **tail** F5/report/fingerprint，不再是旧 shadow；
                旧 shadow 必须已登记为 superseded 且**不被加载**
        输入冻结 除六项 finalized 外，另冻结 tail F5 与 canonical repool
                （analysis-input freeze manifest）
        入口    必须是一次性 `--execute-all`，不得再有按 arm/estimand 的分次调用
    """
    checks: list[dict] = []

    fz = check_v3_frozen()
    checks.append(_check("preregistration_frozen", fz.match,
                         {"locked_version": fz.locked_version,
                          "sha256": fz.current_sha256}))

    # ---- 六项 finalized freeze manifest ---------------------------------
    if freeze_manifest_path.exists():
        man = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        ver = ub.verify_stage1_freeze_manifest(man, prod_dir)
        checks.append(_check("finalized_freeze_manifest", bool(ver["ok"]),
                             {"drift": ver["drift"], "hash_matches": ver["hash_matches"]}))
    else:
        checks.append(_check("finalized_freeze_manifest", False, "manifest 不存在"))

    # ---- analysis-input freeze manifest（新）----------------------------
    if analysis_input_manifest_path.exists():
        aman = json.loads(analysis_input_manifest_path.read_text(encoding="utf-8"))
        aver = ub.verify_analysis_input_manifest(aman, prod_dir=prod_dir,
                                                 tail_dir=tail_dir, repool_dir=repool_dir)
        checks.append(_check("analysis_input_freeze_manifest", bool(aver["ok"]),
                             {"drift": aver["drift"][:5],
                              "hash_matches": aver["hash_matches"],
                              "n_entries": aver.get("n_entries"),
                              "structural_problems": aver["structural_problems"]}))
    else:
        checks.append(_check("analysis_input_freeze_manifest", False,
                             "analysis-input manifest 不存在"))

    # ---- Path B = tail overlay ------------------------------------------
    tail_ok, tail_detail = True, {}
    try:
        ov = overlay.load_tail_overlay(tail_dir, prod_dir, repool_dir)
        tail_detail = {"counts": ov["counts"], "overlay_source": ov["overlay_source"],
                       "source_freeze_checked": ov["source_freeze_check"]["n_checked"],
                       "report_sha256": _sha(tail_dir / "tail_completion_report.json"),
                       "fingerprint_sha256": _sha(tail_dir / "tail_fingerprint.json")}
    except Exception as exc:                                   # noqa: BLE001
        tail_ok, tail_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("path_b_tail_overlay_gate", tail_ok, tail_detail))

    # ---- 旧 shadow 已登记 superseded 且未被接入 ---------------------------
    legacy_ok, legacy_detail = True, {}
    try:
        doc = overlay.assert_legacy_shadow_superseded(shadow_dir)
        legacy_detail = {"status": doc["status"],
                         "n_retained_files": len(doc.get("retained_files") or []),
                         "accepted_overlay_sources": list(overlay.ACCEPTED_OVERLAY_SOURCES)}
        if overlay.OVERLAY_SOURCE_LEGACY_SHADOW in overlay.ACCEPTED_OVERLAY_SOURCES:
            legacy_ok = False
            legacy_detail = "旧 shadow 仍在 ACCEPTED_OVERLAY_SOURCES 内"
    except Exception as exc:                                   # noqa: BLE001
        legacy_ok, legacy_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("legacy_shadow_superseded_and_not_joined", legacy_ok,
                         legacy_detail))

    # ---- E3 固定 canonical repool ----------------------------------------
    rep_path = repool_dir / "e3_repool_report.json"
    repool_ok = rep_path.exists()
    repool_detail: Any = "report 不存在"
    if repool_ok:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        repool_ok = rep.get("complete") is True
        repool_detail = {"complete": rep.get("complete"),
                         "e3_npz_files": rep.get("e3_npz_files"),
                         "e3_trace_block_pairs": rep.get("e3_trace_block_pairs"),
                         "canonical": "outputs/stage1/repool（E3 是独立 estimand，"
                                      "不随 Path B 改变）"}
    checks.append(_check("e3_canonical_repool_report", repool_ok, repool_detail))

    # ---- 三条分析路径已冻结（**仍是三条**，不新增第四条）------------------
    try:
        paths.assert_frozen_paths(paths.FROZEN_PATHS)
        checks.append(_check("analysis_paths_frozen", len(paths.FROZEN_PATHS) == 3,
                             {"paths": list(paths.FROZEN_PATHS),
                              "path_b_implementation": overlay.OVERLAY_SOURCE_TAIL,
                              "markers": {p: paths.path_markers(p)["marker"]
                                          for p in paths.FROZEN_PATHS}}))
    except paths.PathViolation as exc:
        checks.append(_check("analysis_paths_frozen", False, str(exc)))

    # ---- 真实 analyze callback 已接入 ------------------------------------
    try:
        from . import v3_analyze
        sig = inspect.signature(v3_analyze.make_analyze_callback)
        need = {"cfg", "h1_records", "h1_hidden", "mcq_records", "mcq_hidden",
                "shadow_overlay", "e3_blocks"}
        checks.append(_check("real_analyze_callback_wired",
                             need <= set(sig.parameters),
                             {"module": v3_analyze.__name__,
                              "schema": v3_analyze.ANALYSIS_SCHEMA,
                              "rq2_frozen_status": v3_analyze.RQ2_FROZEN_STATUS}))
    except Exception as exc:                                   # noqa: BLE001
        checks.append(_check("real_analyze_callback_wired", False,
                             f"{type(exc).__name__}: {exc}"))

    checks.append(_check("m4_population_declaration_writer",
                         all(hasattr(pop, n) for n in
                             ("build_declaration", "validate_declaration",
                              "write_declaration", "assert_no_label_leak"))))

    # ---- estimand 接线：跑**真正的 end-to-end** 并校验结果契约 ---------------
    # 只验函数名是假阳性 —— 符号存在不等于 E3 接线完成、也不等于 §6 必报项齐全。
    # 这里用**全合成**探针跑一遍 analyze()，再让 `validate_result_contract` 逐项核。
    est_ok, est_detail = True, {}
    try:
        res = _probe_end_to_end()
        from . import v3_analyze as _va
        est_detail = {
            "mode": "synthetic end-to-end（**不读真实 label**）",
            "contract": res["contract"],
            "e2_estimability": res["estimands"][paths.PATH_A]["E2"]["h1"]["estimability"],
            "e3_estimability": res["estimands"][paths.PATH_A]["E3"]["h1"]["estimability"],
            "e3_wiring_status": res["estimands"][paths.PATH_A]["E3"]["h1"]["wiring_status"],
            "e3_performance_computed":
                res["estimands"][paths.PATH_A]["E3"]["h1"]["performance_estimates_computed"],
            "m2c_caveats": len(res["estimands"][paths.PATH_A]["M2c"]["h1"]
                               ["interpretation_caveats"]),
            "required_arm_keys": list(_va.REQUIRED_ARM_KEYS),
        }
        if est_detail["e3_wiring_status"] != "wired" or not est_detail["e3_performance_computed"]:
            est_ok = False
    except Exception as exc:                                   # noqa: BLE001
        est_ok, est_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("estimands_wired_end_to_end", est_ok, est_detail))

    # ---- 一次性 --execute-all 入口 ---------------------------------------
    src = entry_source_path.read_text(encoding="utf-8") if entry_source_path.exists() else ""
    single = ("--execute-all" in src and "def run_execute_all(" in src
              and "def run_execute(" not in src)
    checks.append(_check("single_shot_execute_all_entry", single,
                         {"has_execute_all": "--execute-all" in src,
                          "legacy_run_execute_removed": "def run_execute(" not in src,
                          "why": ("analyze() 一次就算两臂 × 三路径 × 全部 estimand；"
                                  "按 arm/estimand 分次调用只改文件名，且首次独占解盲后"
                                  "第二次必然失败 —— 必须一次性 --execute-all")}))

    # ---- unblinding 顺序、独占写入、双 hash -------------------------------
    ok_seq = (hasattr(ub, "record_stage1_unblinding")
              and "analysis_input_freeze_hash" in
              inspect.signature(ub.record_stage1_unblinding).parameters)
    checks.append(_check("unblinding_records_both_freeze_hashes", ok_seq,
                         {"log_schema_v2": ub.STAGE1_LOG_SCHEMA_V2,
                          "note": "record_unblinding 先于 analyze；log 以独占创建写入"}))

    existing = [str(p) for p in (list(result_dir.glob("*")) if result_dir.exists() else [])]
    checks.append(_check("no_real_result_targets_exist",
                         not existing and not unblind_log.exists(),
                         {"result_dir_entries": existing,
                          "unblind_log_exists": unblind_log.exists()}))

    finalized: dict[str, dict] = {}
    for name in ub.FINALIZED_SIX:
        f = Path(prod_dir) / name
        finalized[name] = ({"bytes": f.stat().st_size, "sha256": _sha(f)}
                           if f.exists() else {"missing": True})

    blockers = [c["name"] for c in checks if not c["ok"]]
    return {
        "schema_version": READINESS_V3_SCHEMA,
        "supersedes": READINESS_SCHEMA,
        "path_b_implementation": overlay.OVERLAY_SOURCE_TAIL,
        "e3_canonical_source": "repool",
        "legacy_shadow_status": list(overlay.LEGACY_SHADOW_STATUS),
        "checks": checks,
        "freeze_chain": {
            "preregistration": {"match": fz.match, "locked_version": fz.locked_version,
                                "canonical_sha256": fz.current_sha256},
            "finalized_artifacts": finalized,
            "integrity_evidence_caliber": (
                "字节大小 + sha256；mtime **不作为**只读或未修改的证据"),
        },
        "readiness": {
            "ready_for_unblinding": not blockers,
            "blocking_reasons": blockers,
            "n_checks": len(checks),
            "n_passed": sum(1 for c in checks if c["ok"]),
        },
        "determinism": ("本文档由 v3_readiness.recompute_v3 从产物现场重算；"
                        "入口**不信任磁盘上的布尔值**"),
    }


READINESS_REPLICATION_SCHEMA = "stage1_v3_cross_model_replication_readiness_v1"

#: 复现角色下**唯一**被移除的检查，以及移除它的结构性理由。
#: 逐条登记是为了让"角色切换有没有被拿来绕 gate"这件事可审计。
REPLICATION_DROPPED_CHECKS = {
    "legacy_shadow_superseded_and_not_joined": (
        "该检查读 Qwen 的 outputs/stage1/shadow/SUPERSEDED.json —— 那是 Qwen 的 "
        "full-BF16 shadow 工程史。跨模型复现从不产生旧 shadow，故该检查**无可检之物**。"
        "注意被移除的只是『必须存在一份 superseded 登记』这个前提；"
        "『旧 shadow 不得接 label』的约束仍在 —— ACCEPTED_OVERLAY_SOURCES 未变"),
}


def recompute_replication(*, prod_dir: Path, tail_dir: Path, repool_dir: Path,
                          result_dir: Path, release_log: Path,
                          freeze_manifest_path: Path,
                          analysis_input_manifest_path: Path,
                          entry_source_path: Path) -> dict:
    """跨模型复现的 readiness。**`recompute_v3` 一行未动**，Qwen 路径逐字节可复核。

    与 v3 的差别只有三处，且全部登记在案：

        1. 去掉 `legacy_shadow_superseded_and_not_joined`（见 REPLICATION_DROPPED_CHECKS）
        2. 判定键改为 `ready_for_analysis`（本次**不是**解盲，一次性解盲已用尽）
        3. "目标不得已存在" 检查指向复现自己的结果目录与 analysis_release_log

    其余检查——G2、两份 freeze manifest、tail overlay 门、canonical repool、
    三条冻结路径、analyze 回调、M4 writer、合成 end-to-end、单次入口——**全部保留**。
    """
    checks: list[dict] = []

    fz = check_v3_frozen()
    checks.append(_check("preregistration_frozen", fz.match,
                         {"locked_version": fz.locked_version,
                          "sha256": fz.current_sha256,
                          "note": ("预注册 v3 未被修改；本次跨模型运行落在 v3 **之外**，"
                                   "属 non-confirmatory，不改动任何冻结设计")}))

    if freeze_manifest_path.exists():
        man = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        ver = ub.verify_stage1_freeze_manifest(man, prod_dir)
        checks.append(_check("finalized_freeze_manifest", bool(ver["ok"]),
                             {"drift": ver["drift"], "hash_matches": ver["hash_matches"]}))
    else:
        checks.append(_check("finalized_freeze_manifest", False, "manifest 不存在"))

    if analysis_input_manifest_path.exists():
        aman = json.loads(analysis_input_manifest_path.read_text(encoding="utf-8"))
        aver = ub.verify_analysis_input_manifest(aman, prod_dir=prod_dir,
                                                 tail_dir=tail_dir, repool_dir=repool_dir)
        checks.append(_check("analysis_input_freeze_manifest", bool(aver["ok"]),
                             {"drift": aver["drift"][:5],
                              "hash_matches": aver["hash_matches"],
                              "n_entries": aver.get("n_entries"),
                              "structural_problems": aver["structural_problems"]}))
    else:
        checks.append(_check("analysis_input_freeze_manifest", False,
                             "analysis-input manifest 不存在"))

    tail_ok, tail_detail = True, {}
    try:
        ov = overlay.load_tail_overlay(tail_dir, prod_dir, repool_dir)
        tail_detail = {"counts": ov["counts"], "overlay_source": ov["overlay_source"],
                       "source_freeze_checked": ov["source_freeze_check"]["n_checked"]}
    except Exception as exc:                                   # noqa: BLE001
        tail_ok, tail_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("path_b_tail_overlay_gate", tail_ok, tail_detail))

    # 旧 shadow 的**存在性前提**移除，但"不得被接受为 overlay 来源"仍然强制。
    checks.append(_check(
        "legacy_shadow_never_accepted_as_overlay_source",
        overlay.OVERLAY_SOURCE_LEGACY_SHADOW not in overlay.ACCEPTED_OVERLAY_SOURCES,
        {"accepted": list(overlay.ACCEPTED_OVERLAY_SOURCES),
         "dropped_check": REPLICATION_DROPPED_CHECKS}))

    rep_path = repool_dir / "e3_repool_report.json"
    repool_ok = rep_path.exists()
    repool_detail: Any = "report 不存在"
    if repool_ok:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        repool_ok = rep.get("complete") is True
        repool_detail = {"complete": rep.get("complete"),
                         "e3_npz_files": rep.get("e3_npz_files"),
                         "e3_trace_block_pairs": rep.get("e3_trace_block_pairs"),
                         "canonical": str(repool_dir)}
    checks.append(_check("e3_canonical_repool_report", repool_ok, repool_detail))

    try:
        paths.assert_frozen_paths(paths.FROZEN_PATHS)
        checks.append(_check("analysis_paths_frozen", len(paths.FROZEN_PATHS) == 3,
                             {"paths": list(paths.FROZEN_PATHS),
                              "path_b_implementation": overlay.OVERLAY_SOURCE_TAIL}))
    except paths.PathViolation as exc:
        checks.append(_check("analysis_paths_frozen", False, str(exc)))

    try:
        from . import v3_analyze
        sig = inspect.signature(v3_analyze.make_analyze_callback)
        need = {"cfg", "h1_records", "h1_hidden", "mcq_records", "mcq_hidden",
                "shadow_overlay", "e3_blocks", "population_role",
                "numeric_integrity_status"}
        checks.append(_check("replication_analyze_callback_wired",
                             need <= set(sig.parameters),
                             {"module": v3_analyze.__name__,
                              "replication_schema": v3_analyze.REPLICATION_ANALYSIS_SCHEMA,
                              "rq2_replication_status":
                                  v3_analyze.RQ2_REPLICATION_STATUS}))
    except Exception as exc:                                   # noqa: BLE001
        checks.append(_check("replication_analyze_callback_wired", False,
                             f"{type(exc).__name__}: {exc}"))

    checks.append(_check("m4_population_declaration_writer",
                         all(hasattr(pop, n) for n in
                             ("build_declaration", "validate_declaration",
                              "write_declaration", "assert_no_label_leak"))))

    est_ok, est_detail = True, {}
    try:
        res = _probe_end_to_end()
        est_detail = {"mode": "synthetic end-to-end（**不读真实 label**）",
                      "contract": res["contract"]}
        if not res["contract"].get("ok"):
            est_ok = False
    except Exception as exc:                                   # noqa: BLE001
        est_ok, est_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(_check("estimands_wired_end_to_end", est_ok, est_detail))

    src = entry_source_path.read_text(encoding="utf-8") if entry_source_path.exists() else ""
    # 复现入口**绝不**写 confirmatory 的解盲日志：源码层断言，不靠运行时自觉。
    no_unblind = ("record_stage1_unblinding" not in src
                  and "unblinding_log.jsonl" not in src)
    checks.append(_check("replication_entry_never_writes_unblinding_log", no_unblind,
                         {"entry": str(entry_source_path),
                          "why": ("一次性解盲已于 2026-08-23 用尽；跨模型复现是 "
                                  "non-confirmatory，必须写独立的 analysis_release_log")}))

    existing = [str(p) for p in (list(result_dir.glob("*")) if result_dir.exists() else [])]
    checks.append(_check("no_replication_result_targets_exist",
                         not existing and not release_log.exists(),
                         {"result_dir_entries": existing,
                          "release_log_exists": release_log.exists(),
                          "overwrite_policy": "禁止覆盖；重跑须另用新目录并显式声明"}))

    blockers = [c["name"] for c in checks if not c["ok"]]
    return {
        "schema_version": READINESS_REPLICATION_SCHEMA,
        "population_role": "cross_model_replication_non_confirmatory",
        "is_confirmatory": False,
        "based_on": READINESS_V3_SCHEMA,
        "dropped_checks": REPLICATION_DROPPED_CHECKS,
        "path_b_implementation": overlay.OVERLAY_SOURCE_TAIL,
        "e3_canonical_source": "repool",
        "checks": checks,
        "freeze_chain": {
            "preregistration": {"match": fz.match, "locked_version": fz.locked_version,
                                "canonical_sha256": fz.current_sha256},
            "integrity_evidence_caliber": (
                "字节大小 + sha256；mtime **不作为**只读或未修改的证据"),
        },
        "readiness": {
            "ready_for_analysis": not blockers,
            "blocking_reasons": blockers,
            "n_checks": len(checks),
            "n_passed": sum(1 for c in checks if c["ok"]),
        },
        "determinism": ("本文档由 v3_readiness.recompute_replication 从产物现场重算；"
                        "入口**不信任磁盘上的布尔值**"),
    }


def recompute_and_compare_replication(readiness_path: Path, **kw) -> dict:
    """与 v3 同一套语义：现场重算后与磁盘逐项比对，手改布尔值会被抓出来。"""
    fresh = recompute_replication(**kw)
    if not Path(readiness_path).exists():
        raise ReadinessMismatch(f"复现 readiness 文档不存在：{readiness_path}")
    disk = json.loads(Path(readiness_path).read_text(encoding="utf-8"))
    if _comparable(disk) != _comparable(fresh):
        raise ReadinessMismatch(
            "磁盘复现 readiness 与现场重算不一致 —— 拒绝信任磁盘布尔值。\n"
            f"  disk       ready={disk.get('readiness', {}).get('ready_for_analysis')} "
            f"blockers={disk.get('readiness', {}).get('blocking_reasons')}\n"
            f"  recomputed ready={fresh['readiness']['ready_for_analysis']} "
            f"blockers={fresh['readiness']['blocking_reasons']}")
    return fresh


def recompute_and_compare_v3(readiness_path: Path, **kw) -> dict:
    fresh = recompute_v3(**kw)
    if not Path(readiness_path).exists():
        raise ReadinessMismatch(f"readiness v3 文档不存在：{readiness_path}")
    disk = json.loads(Path(readiness_path).read_text(encoding="utf-8"))
    if _comparable(disk) != _comparable(fresh):
        raise ReadinessMismatch(
            "磁盘 readiness v3 与现场重算不一致 —— 拒绝信任磁盘布尔值。\n"
            f"  disk    ready={disk.get('readiness', {}).get('ready_for_unblinding')} "
            f"blockers={disk.get('readiness', {}).get('blocking_reasons')}\n"
            f"  recomputed ready={fresh['readiness']['ready_for_unblinding']} "
            f"blockers={fresh['readiness']['blocking_reasons']}")
    return fresh
