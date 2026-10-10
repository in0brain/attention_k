"""Stage 1 v3 **跨模型复现**分析入口（non-confirmatory）。

    python scripts/run_stage1_v3_replication.py --verify-geometry \
        --model-path D:/models/Meta-Llama-3.1-8B-Instruct
    python scripts/run_stage1_v3_replication.py --dry-run          # 合成 fixture
    python scripts/run_stage1_v3_replication.py --execute-all \
        --artifact-root outputs/stage1_llama_v3 \
        --numeric-integrity-status <not_audited|verified_clean|compromised>

**为什么不复用 `run_stage1_v3_confirmatory.py`**：那个入口带 Qwen 的写死路径、
`confirmatory` 角色，以及**一次性揭盲**语义（`record_stage1_unblinding` 独占写
`outputs/stage1/unblinding/unblinding_log.jsonl`）。一次性揭盲已于 2026-08-23 用尽。
本入口是独立的 non-confirmatory 路径：

    角色      cross_model_replication_non_confirmatory
    schema    stage1_v3_cross_model_replication_analysis_v1
    readiness ready_for_analysis（**不是** ready_for_unblinding）
    释放记录  <artifact_root>/analysis_release/analysis_release_log.jsonl
              独占创建 + fsync，**禁止覆盖**
    RQ2       不在 v3 预注册内 → status=not_in_stage1_v3_cross_model_replication_scope
              computed=false，**不得**写成本次复现的发现

源码里**没有** `record_stage1_unblinding`，也不出现 `unblinding_log.jsonl` ——
readiness 的 `replication_entry_never_writes_unblinding_log` 逐串扫本文件断言这一点。

统计定义一律复用现有核心：E1/E2/E3、M2c、M1b、iv-a/iv-b、T1、D、三条冻结路径
A/B/C 全部**未改**。本入口只注入角色、schema、目录、fingerprint、实际计数与
本次数值审计状态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_analysis_paths as vpaths   # noqa: E402
from recover_attention.evaluation import v3_analyze as vanalyze        # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool        # noqa: E402
from recover_attention.evaluation import v3_production as prod         # noqa: E402
from recover_attention.evaluation import v3_readiness as vreadiness    # noqa: E402
from recover_attention.evaluation import v3_shadow_overlay as voverlay  # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fixture  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig    # noqa: E402
from recover_attention.evaluation.v3_freeze import check_v3_frozen     # noqa: E402

ENTRY_SOURCE = Path(__file__).resolve()

RESULT_NAME = "stage1_v3_cross_model_replication_result.json"
READINESS_NAME = "stage1_v3_cross_model_replication_readiness.json"
RELEASE_LOG_NAME = "analysis_release_log.jsonl"
RELEASE_LOG_SCHEMA = "stage1_v3_analysis_release_log_v1"

ROLE = prod.ROLE_CROSS_MODEL_REPLICATION


class ReplicationRefused(RuntimeError):
    """fail closed：在读 label 或算任何性能量**之前**拒绝。"""


# ══════════════════════════════════════════════════════════════════════════════
# 路径（全部由 artifact root 派生，模块层不出现 outputs/stage1 字面量）
# ══════════════════════════════════════════════════════════════════════════════
def layout(artifact_root: Path) -> dict:
    r = Path(artifact_root)
    return {
        "artifact_root": r,
        "prod": r / "production",
        "tail": r / "tail_stabilized",
        "repool": r / "repool",
        "readiness_dir": r / "readiness",
        "result_dir": r / "analysis",
        "release_dir": r / "analysis_release",
        "readiness": r / "readiness" / READINESS_NAME,
        "finalized_freeze": r / "readiness" / "stage1_finalized_freeze_manifest.json",
        "analysis_input_freeze": (r / "readiness"
                                  / "stage1_analysis_input_freeze_manifest.json"),
        "result": r / "analysis" / RESULT_NAME,
        "release_log": r / "analysis_release" / RELEASE_LOG_NAME,
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# 只读几何自检 —— 不加载权重
# ══════════════════════════════════════════════════════════════════════════════
def verify_geometry(model_path: str) -> dict:
    """只读 config.json 推导并校验几何。**不加载权重、不 import transformers。**

    产出的是**静态**结论。运行时 hidden-index 的数值一致性
    （`numeric_match` / `max_abs_diff`）必须由 G3 的真实前向给出，本命令
    不代表它已通过 —— 报告里 `runtime_hidden_index_verified` 恒为 false。
    """
    geo = repool.ModelGeometry.from_model_config(model_path)
    cross = geo.assert_matches_model_config(model_path)
    return {
        "schema": "stage1_replication_geometry_check_v1",
        "model_path": str(model_path),
        "geometry": geo.as_dict(),
        "cross_check_against_config": cross["matches_config"],
        "qwen_frozen_geometry": repool.QWEN_GEOMETRY.as_dict(),
        "matches_qwen_frozen": geo == repool.QWEN_GEOMETRY,
        "weights_loaded": False,
        "runtime_hidden_index_verified": False,
        "still_required_at_g3": [
            "backend.verify_hidden_index() 的 numeric_match / max_abs_diff",
            "显存峰值与 headroom",
            "forward-nonfinite 行为",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 释放记录（独立于 confirmatory 的解盲日志）
# ══════════════════════════════════════════════════════════════════════════════
def record_analysis_release(log_path: Path, *, payload: dict) -> dict:
    """独占创建 + fsync。**禁止覆盖**：已存在即拒绝，不静默追加成"第一次"。

    与 confirmatory 的 unblinding log 是两套东西：这里不写、也不读那个文件。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ReplicationRefused(
            f"分析释放记录已存在：{log_path} —— 禁止覆盖。"
            f"重跑必须另用新的 artifact root 并在 history 中显式声明为重跑")
    row = {"schema_version": RELEASE_LOG_SCHEMA, "role": ROLE,
           "is_confirmatory": False, **payload}
    data = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")
    fd = os.open(str(log_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return row


def write_result_atomic(path: Path, result: dict) -> Path:
    """原子发布且**绝不覆盖**：目标在分析期间出现也拒绝，不悄悄盖掉。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ReplicationRefused(f"结果目标已存在，拒绝覆盖：{path}")
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                      default=_json_safe, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(str(tmp), str(path))          # 目标已存在时**原子失败**
    except FileExistsError as exc:
        Path(tmp).unlink(missing_ok=True)
        raise ReplicationRefused(f"结果目标在发布瞬间出现，拒绝覆盖：{path}") from exc
    Path(tmp).unlink(missing_ok=True)
    return path


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"不可序列化：{type(obj).__name__}")


# ══════════════════════════════════════════════════════════════════════════════
# 实际计数：从 finalized 输入现场重建，不用"输出行数"验证同一份输出
# ══════════════════════════════════════════════════════════════════════════════
def observed_counts(paths_: dict, cfg: V3AnalysisConfig) -> dict:
    """规模取自**独立的** finalized 产物与 finalization report。

    分母是生成侧终态（completion records + report），不是分析侧的输出行数 ——
    拿一份输出验证它自己不构成完整性证据。
    """
    out: dict = {"source": "finalized_completion_records_and_reports"}
    for arm in ("h1", "mcq"):
        recs_path = paths_["prod"] / f"{arm}_completion_records.jsonl"
        rep_path = paths_["prod"] / f"{arm}_finalization_report.json"
        if not recs_path.exists() or not rep_path.exists():
            raise ReplicationRefused(f"{arm}: 缺 finalized 产物 {recs_path} / {rep_path}")
        rows = [json.loads(x) for x in recs_path.read_bytes().split(b"\n") if x.strip()]
        report = json.loads(rep_path.read_text(encoding="utf-8"))
        roles = sorted({r.get("population_role") for r in rows})
        if roles != [ROLE]:
            raise ReplicationRefused(
                f"{arm}: finalized 记录的 population_role={roles} —— "
                f"复现分析只接受 [{ROLE}]，拒绝混入 confirmatory 产物")
        analyzable = [r for r in rows if prod.requires_hidden(arm, r)]
        ids = [r.get("trace_id") if arm == "h1" else r.get("example_id") for r in rows]
        dup = len(ids) - len(set(ids))
        out[arm] = {
            "n_finalized_records": len(rows),
            "n_analyzable": len(analyzable),
            "n_duplicate_keys": dup,
            "report_n_finalized_records": report.get("n_finalized_records"),
            "report_n_analyzable_records": report.get("n_analyzable_records"),
            "report_population_role": report.get("population_role"),
            "records_match_report": (
                len(rows) == report.get("n_finalized_records")
                and len(analyzable) == report.get("n_analyzable_records")),
            "technical_exclusions": report.get("technical_exclusions"),
            "protocol_deviation": report.get("protocol_deviation"),
        }
        if dup:
            raise ReplicationRefused(f"{arm}: finalized 记录含 {dup} 个重复键")
        if not out[arm]["records_match_report"]:
            raise ReplicationRefused(
                f"{arm}: finalized 记录数与 finalization report 不一致 —— "
                f"records {len(rows)}/{len(analyzable)} vs report "
                f"{report.get('n_finalized_records')}/"
                f"{report.get('n_analyzable_records')}")
        if report.get("population_role") != ROLE:
            raise ReplicationRefused(
                f"{arm}: finalization report 的 population_role="
                f"{report.get('population_role')!r} != {ROLE!r}")
    h1_rows = [json.loads(x) for x in
               (paths_["prod"] / "h1_completion_records.jsonl").read_bytes().split(b"\n")
               if x.strip()]
    h1_analyzable = [r for r in h1_rows if prod.requires_hidden("h1", r)]
    out["e3_expected_pairs"] = sum(len(repool.expected_blocks(r, cfg))
                                   for r in h1_analyzable)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# preflight / execute
# ══════════════════════════════════════════════════════════════════════════════
def preflight(paths_: dict) -> dict:
    """现场重算 readiness 并与磁盘比对；**不信任磁盘上的布尔值**。"""
    fz = check_v3_frozen()
    if not fz.match:
        raise ReplicationRefused("预注册 v3 与 lock 不一致，拒绝启动")
    fresh = vreadiness.recompute_and_compare_replication(
        paths_["readiness"],
        prod_dir=paths_["prod"], tail_dir=paths_["tail"], repool_dir=paths_["repool"],
        result_dir=paths_["result_dir"], release_log=paths_["release_log"],
        freeze_manifest_path=paths_["finalized_freeze"],
        analysis_input_manifest_path=paths_["analysis_input_freeze"],
        entry_source_path=ENTRY_SOURCE)
    if not fresh["readiness"]["ready_for_analysis"]:
        raise ReplicationRefused(
            f"ready_for_analysis=false，blockers="
            f"{fresh['readiness']['blocking_reasons']}")
    return fresh


def run_execute_all(*, artifact_root: Path, numeric_status: str,
                    numeric_evidence: dict | None, cfg: V3AnalysisConfig,
                    progress=None) -> int:
    """一次性跑完两臂 × 三路径 × 全部 estimand，写唯一结果文件。

    顺序固定：preflight → 释放记录 → 构造回调（此刻才载入 records）→
    analyze（首次读 label）→ 写结果。
    """
    paths_ = layout(artifact_root)
    fresh = preflight(paths_)
    counts = observed_counts(paths_, cfg)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_root": str(paths_["artifact_root"]),
        "readiness_sha256": _sha(paths_["readiness"]),
        "finalized_freeze_sha256": _sha(paths_["finalized_freeze"]),
        "analysis_input_freeze_sha256": _sha(paths_["analysis_input_freeze"]),
        "prereg_sha256": check_v3_frozen().current_sha256,
        "numeric_integrity_status": numeric_status,
        "observed_counts": counts,
        "note": ("cross-model replication: 两臂 x 三路径 x 全部 estimand，单次调用。"
                 "**不是** confirmatory，**不写** confirmatory 的解盲日志"),
    }
    record_analysis_release(paths_["release_log"], payload=payload)
    print(f"[release] {paths_['release_log']}", file=sys.stderr)

    # ---- 此刻才载入 records（含 label 字段）------------------------------
    def _recs(arm: str) -> list[dict]:
        path = paths_["prod"] / f"{arm}_completion_records.jsonl"
        return [json.loads(x) for x in path.read_bytes().split(b"\n") if x.strip()]

    def _hidden(arm: str) -> dict:
        import numpy as np
        with np.load(paths_["prod"] / f"{arm}_hidden.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    h1_records, mcq_records = _recs("h1"), _recs("mcq")
    ov = voverlay.load_tail_overlay(paths_["tail"], paths_["prod"], paths_["repool"])
    e3 = voverlay.load_e3_blocks(paths_["repool"],
                                 [r for r in h1_records if prod.requires_hidden("h1", r)],
                                 cfg)
    analyze = vanalyze.make_analyze_callback(
        cfg=cfg, h1_records=h1_records, h1_hidden=_hidden("h1"),
        mcq_records=mcq_records, mcq_hidden=_hidden("mcq"),
        shadow_overlay=ov, e3_blocks=e3,
        population_role=ROLE,
        result_schema=vanalyze.REPLICATION_ANALYSIS_SCHEMA,
        numeric_integrity_status=numeric_status,
        numeric_integrity_evidence=numeric_evidence,
        observed_counts=counts, progress=progress)
    result = analyze()
    result["readiness"] = fresh["readiness"]
    result["release_record"] = payload
    write_result_atomic(paths_["result"], result)
    print(f"[result ] {paths_['result']}")
    return 0


def run_dry(numeric_status: str | None = None) -> dict:
    """合成 fixture 的**真实端到端**自检 —— 与 `--execute-all` 同一条计算路径，
    只是数据全部由 `synthetic_contract_probe()` 生成。

    **明确 synthetic**：不读任何真实产物、不接触真实 label。
    刻意用**复现角色**跑，好让 schema / RQ2 分支 / 数值状态标记这三处
    也真被执行一遍，而不是只检查符号存在。
    """
    import warnings

    probe = vanalyze.synthetic_contract_probe()
    pcfg = probe.pop("cfg")
    status = vpaths.assert_numeric_status(numeric_status)
    with warnings.catch_warnings():
        # 探针样本极小 → sklearn 会刷 inner-fold 类别人数警告，那是探针尺寸的
        # 性质而非分析问题；只在此静音，不动全局过滤器。
        warnings.filterwarnings("ignore", category=UserWarning,
                                module=r"sklearn\.model_selection\..*")
        result = vanalyze.make_analyze_callback(
            cfg=pcfg, population_role=ROLE,
            result_schema=vanalyze.REPLICATION_ANALYSIS_SCHEMA,
            numeric_integrity_status=status, **probe)()
    return {
        "synthetic": True, "reads_real_artifacts": False,
        "is_end_to_end": True,
        "role_under_test": ROLE,
        "schema": result["schema"],
        "is_confirmatory": result["is_confirmatory"],
        "numeric_integrity": result["numeric_integrity"]["status"],
        "path_a_marker": result["path_markers"][vpaths.PATH_A]["marker"],
        "contrast_D_path_a_marker": result["contrast_D"][vpaths.PATH_A]["marker"],
        "rq2": {k: result["rq2"].get(k) for k in ("status", "computed")},
        "frozen_paths": result["frozen_paths"],
        "contract_ok": result["contract"]["ok"],
        "numeric_states": list(vpaths.NUMERIC_STATES),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-geometry", action="store_true",
                    help="只读 config.json 校验模型几何；不加载权重")
    ap.add_argument("--dry-run", action="store_true",
                    help="合成 fixture 自检；不读真实产物")
    ap.add_argument("--execute-all", dest="execute_all", action="store_true",
                    help="正式复现分析：两臂 x 三路径 x 全部 estimand，一次跑完")
    ap.add_argument("--model-path", default="D:/models/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--artifact-root", type=Path,
                    default=ROOT / "outputs/stage1_llama_v3")
    ap.add_argument("--numeric-integrity-status", default=None,
                    choices=list(vpaths.NUMERIC_STATES),
                    help="本次 B4 数值完整性审计结论。**省略即 not_audited**，"
                         "绝不默认为通过")
    ap.add_argument("--numeric-integrity-evidence", type=Path, default=None,
                    help="数值审计产物 JSON 的路径，写进结果的 evidence 字段")
    ap.add_argument("--progress-log", type=Path, default=None,
                    help="纯进度事件的 JSONL 落盘路径（无任何统计量）")
    args = ap.parse_args(argv)

    if sum(bool(x) for x in (args.verify_geometry, args.dry_run,
                             args.execute_all)) != 1:
        ap.error("--verify-geometry / --dry-run / --execute-all 三选一")

    if args.verify_geometry:
        print(json.dumps(verify_geometry(args.model_path), ensure_ascii=False, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps(run_dry(args.numeric_integrity_status),
                         ensure_ascii=False, indent=2))
        return 0

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    status = vpaths.assert_numeric_status(args.numeric_integrity_status)
    if status == vpaths.NUMERIC_NOT_AUDITED:
        print("[warn  ] B4 数值完整性状态 = not_audited —— 结果的 Path A / D 会带"
              "『尚未审计』标记；这**不是**『已通过』。", file=sys.stderr)
    evidence = (json.loads(Path(args.numeric_integrity_evidence).read_text("utf-8"))
                if args.numeric_integrity_evidence else None)

    progress = None
    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        handle = open(args.progress_log, "a", encoding="utf-8")

        def progress(event: dict) -> None:      # noqa: F811
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    try:
        return run_execute_all(artifact_root=args.artifact_root,
                               numeric_status=status, numeric_evidence=evidence,
                               cfg=cfg, progress=progress)
    except ReplicationRefused as exc:
        print(f"[REFUSE] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
