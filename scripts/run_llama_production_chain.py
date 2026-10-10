r"""Llama 两臂正式生成的**无人值守串行接续**。

它只做"接续"，不做决策之外的任何事：

    1. 接管当前 H1 生成进程：等它退出，**并让它的退出经过同一套失败策略**
       （PID 消失不等于正常结束 —— 没有本次新增的段日志就是"无法确认"，停）
    2. 完整性核验绑**完整计划名单**（冻结名单 × K 规则），不只看行数；
       未完成则用**同一条命令**续跑（raw append-only）
    3. H1 --finalize
    4. MCQ 正式生成前的**有时间上限的性能核查**（GPU 串行，H1 已结束，不争显存）
    5. 决策窗口（有上限）：默认**沿用已验证版本**；本脚本**从不改源码**
    6. MCQ 正式生成
    7. MCQ --finalize

任一步非零退出或产物不达预期 → **停止并保留日志**，不自动删行、不改参数、不重新生成。

用法（后台单进程）：
    python scripts/run_llama_production_chain.py --h1-pid 1288
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PY = r"D:/conda/Miniconda3/envs/recover_attention/python.exe"
MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR = ROOT / "outputs/stage1_llama_v3/production"
G3 = ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed"
G4 = ROOT / "outputs/stage1_llama_v3/g4_budget512_reparsed"
REG = ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
ROLE = "cross_model_replication_non_confirmatory"

LOG_DIR = ROOT / "tmp/stage1_l2_g3/chain"
STATUS = LOG_DIR / "chain_status.json"
LOCK = LOG_DIR / "chain.lock"
DECISION = ROOT / "tmp/stage1_l2_g3/mcq_optimization_decision.json"
#: 全链自动接续计数 —— **持久化**，接续脚本重启不清零。
AUTORESUME = LOG_DIR / "chain_autoresume.json"
TIMING = ROOT / "tmp/stage1_l2_g3/mcq_phase_timing.json"

import chain_failure_policy as pol                                   # noqa: E402

#: 仅用于开工时的**名义**核对与 ETA；**完成判据不用它**（见 pol.arm_complete）。
EXPECTED = {"h1": 9264, "mcq": 10560}

BASE = [PY, "scripts/run_stage1_production.py",
        "--model-path", MODEL, "--geometry-from-model-config",
        "--g3-dir", str(G3), "--g4-dir", str(G4),
        "--population-role", ROLE, "--out-dir", str(OUT_DIR)]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _state(step: str, **kw) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prev = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"log": []}
    entry = {"t": _now(), "step": step, **kw}
    prev["log"] = prev.get("log", []) + [entry]
    prev["current"] = entry
    STATUS.write_text(json.dumps(prev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{entry['t']}] {step} {kw if kw else ''}", flush=True)


def _rows(arm: str) -> int:
    p = OUT_DIR / f"{arm}_raw_unit_results.jsonl"
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


def _alive(pid: int) -> bool:
    """按 CSV 的 PID 字段**精确**比对。`str(pid) in out` 会被内存占用列误伤。"""
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = [x.strip('"') for x in line.strip().split('","')]
        if len(parts) >= 2 and parts[1].strip() == str(pid):
            return True
    return False


def _run(cmd: list[str], name: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"
    try:
        shown = str(log.relative_to(ROOT)).replace("\\", "/")
    except ValueError:                       # 合成故障验证会把日志放到临时目录
        shown = str(log)
    _state(f"run:{name}", cmd=" ".join(cmd), log=shown)
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n===== {_now()} {' '.join(cmd)} =====\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    _state(f"done:{name}", returncode=rc)
    return rc


def _paths(arm: str) -> dict:
    return {"manifest": OUT_DIR / f"{arm}_production_manifest.jsonl",
            "diagnostics": OUT_DIR / f"{arm}_forward_nonfinite_diagnostics.jsonl",
            "segment_log": OUT_DIR / f"{arm}_segment_log.jsonl",
            "raw": OUT_DIR / f"{arm}_raw_unit_results.jsonl"}


_PLAN_CACHE: dict[str, set[str]] = {}


def _planned(arm: str) -> set[str]:
    """**完整计划名单**：冻结名单 × K 规则，与 `v3_g3_smoke.units_for` 同一条规则。

    完成判据必须绑到它 —— 不能拿 manifest 自己的 total 当预期总数
    （manifest 残缺时 total 会跟着变小，缺失就被"抵消"掉了）。
    不加载模型：MCQ 的 rendered_prompt 用空串桩，`example_id` 与它无关。
    """
    if arm in _PLAN_CACHE:
        return _PLAN_CACHE[arm]
    sys.path.insert(0, str(ROOT / "src"))
    from types import SimpleNamespace
    from recover_attention.evaluation import v3_g3_smoke as g3
    from recover_attention.evaluation import v3_production as prod
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    v = prod.verify_production_inputs(
        samples_dir=ROOT / "outputs/stage1/samples", g3_dir=G3, g4_dir=G4,
        prereg=ROOT / "docs/paper/preregistration_v3.md",
        lock=ROOT / "docs/paper/preregistration_v3.lock", cfg=cfg, arm=arm)
    stub = SimpleNamespace(apply_chat_template=lambda *a, **k: "")
    records = g3.build_arm_records(arm, v["_primary_rows"], cfg=cfg, tokenizer=stub)
    plan = pol.planned_unit_ids((r["example_id"] for r in records), cfg.k_traces)
    _PLAN_CACHE[arm] = plan
    return plan


def _arm_generation(arm: str, extra: list[str], max_resumes: int,
                    adopt_pid: int | None = None) -> bool:
    """跑到该臂**全部计划 unit 进入终态**。

    `adopt_pid` 是**接管的既有生成进程**（例如本链启动前就在跑的 H1）。
    它与本链自己起的进程走**同一套退出归类**：接续器拿不到它的退出码，
    所以更要靠**本次新增的段日志**判断 —— **PID 消失不等于正常结束**。

    失败策略（用户 2026-09-15 的有限授权，判据实现见 `scripts/chain_failure_policy.py`）：
    只有本次新增的段日志 / 诊断 / manifest **三方指向同一 unit 的 forward_nonfinite**
    才自动接续；接续是跳过已终结的 unit，不重试、不清除 excluded。
    全链累计上限 5 且持久化。其余任何情况（含"无法确认退出原因"）一律停止。
    """
    pth = _paths(arm)
    plan = _planned(arm)
    counter = pol.load_counter(AUTORESUME)
    attempt = 0
    while True:
        mstate = pol.manifest_state(pth["manifest"])
        raw_ids = pol.raw_unit_ids(pth["raw"])
        ok, why = pol.arm_complete(mstate, raw_ids, plan)
        if ok:
            _state(f"{arm}:generation_complete", why=why, done=mstate.done,
                   excluded=mstate.excluded, planned=len(plan))
            return True
        if attempt > max_resumes + pol.MAX_AUTORESUMES_PER_CHAIN:
            _state(f"{arm}:STOP_attempt_budget_exhausted", attempts=attempt, why=why)
            return False

        # 限定 4：**先**拿本次运行的证据边界，再开始这一轮
        try:
            boundary = pol.snapshot(manifest=pth["manifest"],
                                    diagnostics=pth["diagnostics"],
                                    segment_log=pth["segment_log"])
        except ValueError as exc:
            _state(f"{arm}:STOP_diagnostic_incomplete", error=str(exc))
            return False

        if adopt_pid is not None and attempt == 0:
            # 接管：**不启动第二个进程**，等它自己退出，再按同一策略判它怎么退的
            _state(f"{arm}:adopt_wait", pid=adopt_pid, why=why,
                   note="接管既有生成进程；其退出同样要经失败策略，不因 PID 消失就放行")
            while _alive(adopt_pid):
                time.sleep(60)
            rc = None
            _state(f"{arm}:adopted_process_exited", pid=adopt_pid)
        else:
            _state(f"{arm}:generation_start", attempt=attempt, why=why,
                   done=mstate.done, excluded=mstate.excluded, planned=len(plan),
                   autoresumes_used=len(counter.get("units", [])))
            rc = _run(BASE + ["--arm", arm] + extra, f"{arm}_gen_{attempt}")
        attempt += 1

        after = pol.manifest_state(pth["manifest"])
        raw_after = pol.raw_unit_ids(pth["raw"])
        try:
            new_diag = pol.new_diagnostics(boundary.diagnostics,
                                           pol.diagnostic_unit_ids(pth["diagnostics"]))
        except ValueError as exc:
            _state(f"{arm}:STOP_diagnostic_incomplete", error=str(exc))
            return False
        segs_new = pol.segment_entries(pth["segment_log"])[boundary.n_segments:]

        # 限定 2 + 3：本臂已登记的排除 unit 必须仍在、仍是 excluded/forward_nonfinite
        arm_units = pol.units_for_arm(counter, arm)
        if arm_units:
            ok2, why2 = pol.verify_still_excluded(after, arm_units)
            if not ok2:
                _state(f"{arm}:STOP_excluded_state_changed", why=why2)
                return False

        ok, why = pol.arm_complete(after, raw_after, plan)
        if ok:
            _state(f"{arm}:generation_complete", why=why, done=after.done,
                   excluded=after.excluded, planned=len(plan))
            return True

        kind, detail = pol.classify_exit(segments_new=segs_new, returncode=rc)
        if kind == pol.OUTCOME_NORMAL:
            if after.done <= boundary.mstate.done and after.excluded <= boundary.mstate.excluded:
                _state(f"{arm}:STOP_no_progress", why=why, detail=detail,
                       done=after.done, excluded=after.excluded, planned=len(plan))
                return False
            _state(f"{arm}:partial", why=why, detail=detail, done=after.done,
                   excluded=after.excluded, planned=len(plan))
            continue

        if kind == pol.OUTCOME_NONFINITE:
            d = pol.decide_autoresume(
                segments_new=segs_new, mstate_before=boundary.mstate, mstate_after=after,
                new_diag=new_diag, already_resumed=list(counter.get("units", [])),
                returncode=rc)
            if not d.allow:
                _state(f"{arm}:STOP_nonzero_exit", returncode=rc, why=d.why,
                       unit_id=d.unit_id, done=after.done, excluded=after.excluded,
                       evidence=d.evidence)
                return False
            counter = pol.record_autoresume(AUTORESUME, counter, arm=arm,
                                            unit_id=d.unit_id, when=_now(),
                                            evidence=d.evidence)
            _state(f"{arm}:autoresume_forward_nonfinite", unit_id=d.unit_id,
                   used=len(counter["units"]), limit=pol.MAX_AUTORESUMES_PER_CHAIN,
                   why=d.why,
                   note=("排除的**授权**仍须人工执行 apply_mcq_*_exclusion.py；"
                         "未登记时 finalize 会自行拒绝，本链不绕过"))
            continue

        _state(f"{arm}:STOP_nonzero_exit", returncode=rc, kind=kind, why=detail,
               completeness=why, done=after.done, excluded=after.excluded,
               planned=len(plan),
               note="退出原因不是明确的正常分段结束，也不满足有限授权的 forward_nonfinite")
        return False


def _finalize(arm: str, extra: list[str]) -> bool:
    rc = _run(BASE + ["--arm", arm, "--finalize"] + extra, f"{arm}_finalize")
    rep = OUT_DIR / f"{arm}_finalization_report.json"
    if rc != 0 or not rep.exists():
        _state(f"{arm}:STOP_finalize_failed", returncode=rc, report_exists=rep.exists())
        return False
    d = json.loads(rep.read_text(encoding="utf-8"))
    _state(f"{arm}:finalized",
           n_finalized_records=d.get("n_finalized_records"),
           n_analyzable_records=d.get("n_analyzable_records"),
           n_hidden_vectors=d.get("n_hidden_vectors"),
           technical_exclusions=d.get("technical_exclusions"),
           protocol_deviation=d.get("protocol_deviation"),
           population_role=d.get("population_role"))
    return True


def _perf_check(max_seconds: float, n_units: int) -> str:
    """有上限的性能核查。失败不阻塞流水线 —— 按"收益不明确"处理。"""
    rc = _run([PY, "scripts/diagnostics/time_mcq_unit_phases.py",
               "--n", str(n_units), "--max-seconds", str(int(max_seconds))], "perf_check")
    if rc != 0 or not TIMING.exists():
        _state("perf:failed_treated_as_inconclusive", returncode=rc)
        return "keep_verified_version"
    d = json.loads(TIMING.read_text(encoding="utf-8"))
    _state("perf:done", verdict=d.get("verdict"),
           loop_share=d.get("pertoken_loop_share_of_current"),
           bitwise_equal=d.get("bitwise_identical_on_consumed_fields"),
           mean_s_current=d.get("per_unit_mean_s_current"),
           mean_s_candidate=d.get("per_unit_mean_s_candidate"))
    return d.get("verdict", "keep_verified_version")


def _decision_window(verdict: str, wait_s: float) -> None:
    """只在候选值得采用时才等；等不到就按默认走。**本脚本从不改源码。**"""
    if verdict != "adopt_candidate":
        _state("decision:default_keep_verified", why="性能核查判定收益不明确或不逐位相同")
        return
    _state("decision:waiting", wait_seconds=wait_s,
           how=("在 tmp/stage1_l2_g3/mcq_optimization_decision.json 写 "
                '{"adopt": true, "applied": true} 表示补丁已由人工应用并复核；'
                "超时或缺文件一律沿用已验证版本"))
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if DECISION.exists():
            d = json.loads(DECISION.read_text(encoding="utf-8"))
            _state("decision:received", decision=d)
            return
        time.sleep(30)
    _state("decision:timeout_keep_verified", waited_s=round(time.time() - t0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h1-pid", type=int, default=None,
                    help="要接管的既有 H1 生成进程；不给就自己起（H1 已完成时无需给）")
    ap.add_argument("--skip-h1", action="store_true",
                    help="H1 已生成完且 finalize 过 —— 直接从 MCQ 接着跑，"
                         "不重跑 H1 的 finalization")
    ap.add_argument("--perf-max-seconds", type=float, default=900.0)
    ap.add_argument("--perf-units", type=int, default=14)
    ap.add_argument("--decision-wait-seconds", type=float, default=1800.0)
    ap.add_argument("--max-resumes", type=int, default=3)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="只校验辅助函数与命令拼装，**不执行任何生成/finalize**")
    args = ap.parse_args()

    if args.self_test:
        me = os.getpid()
        checks = [
            ("_alive(自己) == True", _alive(me) is True),
            ("_alive(不存在的 pid) == False", _alive(999999) is False),
            (f"_alive(h1_pid={args.h1_pid})",
             True if args.h1_pid is None else _alive(args.h1_pid)),
            ("h1 raw 行数可读", _rows("h1") > 0),
            ("mcq raw 可读", _rows("mcq") >= 0),
            ("G3/G4/登记文件都在", G3.exists() and G4.exists() and REG.exists()),
            ("H1 命令含 ontology-dir 且不含 budget-registration",
             "--ontology-dir" in (BASE + ["--arm", "h1", "--ontology-dir",
                                          "data/raw/ontology"])
             and "--budget-registration" not in BASE),
            ("MCQ 命令含 budget-registration",
             "--budget-registration" in (BASE + ["--arm", "mcq",
                                                 "--budget-registration", str(REG)])),
            ("角色是 cross_model_replication_non_confirmatory",
             ROLE == "cross_model_replication_non_confirmatory"),
            ("预期 unit 数 h1=9264 / mcq=10560",
             EXPECTED == {"h1": 9264, "mcq": 10560}),
        ]
        bad = [c for c, ok in checks if not ok]
        for c, ok in checks:
            print(f"  [{'ok ' if ok else 'FAIL'}] {c}")
        print(f"[self-test] {len(checks) - len(bad)}/{len(checks)} 通过  h1_rows={_rows('h1')}")
        return 0 if not bad else 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text(encoding="utf-8").strip())
        except ValueError:
            old = -1
        if old > 0 and _alive(old):
            print(f"[stop] 已有接续进程在跑（pid={old}）—— 不启动重复进程。")
            return 3
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    _state("chain:start", pid=os.getpid(), h1_pid=args.h1_pid,
           expected=EXPECTED, out_dir=str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"))

    # ── 1. H1 ──────────────────────────────────────────────────────────────
    #     接管既有进程时，**它的退出同样要经过失败策略** ——
    #     之前这里直接 while _alive(...) 再进 _arm_generation，等于给 OOM 或未知退出
    #     白送一次自动重启。PID 消失不等于正常结束。
    h1_extra = ["--ontology-dir", "data/raw/ontology"]
    if args.skip_h1:
        # 只核对，不重跑：完成判据仍绑计划名单，finalization 报告必须已在
        rep = OUT_DIR / "h1_finalization_report.json"
        mstate = pol.manifest_state(_paths("h1")["manifest"])
        ok, why = pol.arm_complete(mstate, pol.raw_unit_ids(_paths("h1")["raw"]),
                                   _planned("h1"))
        if not (ok and rep.exists()):
            _state("h1:STOP_skip_h1_but_not_complete", why=why,
                   report_exists=rep.exists(),
                   note="--skip-h1 只允许在 H1 已完成且已 finalize 时使用")
            return 2
        d = json.loads(rep.read_text(encoding="utf-8"))
        _state("h1:skipped_already_complete", why=why,
               n_finalized_records=d.get("n_finalized_records"),
               n_analyzable_records=d.get("n_analyzable_records"),
               population_role=d.get("population_role"))
    else:
        if not _arm_generation("h1", h1_extra, args.max_resumes, adopt_pid=args.h1_pid):
            return 2
        if not _finalize("h1", h1_extra):
            return 2

    # ── 2. 性能核查（GPU 串行；H1 已结束）──────────────────────────────────
    verdict = "keep_verified_version"
    if not args.skip_perf:
        verdict = _perf_check(args.perf_max_seconds, args.perf_units)
    _decision_window(verdict, args.decision_wait_seconds)

    # ── 3. MCQ ────────────────────────────────────────────────────────────
    mcq_extra = ["--budget-registration", str(REG)]
    if not _arm_generation("mcq", mcq_extra, args.max_resumes):
        return 2
    if not _finalize("mcq", mcq_extra):
        return 2

    _state("chain:complete", h1_rows=_rows("h1"), mcq_rows=_rows("mcq"))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()
    raise SystemExit(code)
