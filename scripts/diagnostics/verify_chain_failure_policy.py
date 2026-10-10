r"""端到端**合成故障**验证接续链的失败策略（限定 7）。

跑的是**真实的** `run_stage1_production.main()`（假后端，不加载模型、不用 GPU），
只把某一条 unit 的 `run_unit` 换成抛异常 —— 于是 manifest / 诊断 / 段日志都是
真实路径写出来的，接续链读到的也是真产物。**不碰正在运行的生成进程与其产物**：
全部在独立临时目录里，`--out-dir` 指向临时目录。

场景：
    A  合法 forward_nonfinite 接续      → 接续 1 次，全部计划 unit 终态
    A2 多次**不同** unit 的合法排除     → 均接续，计数互不相同
    B  重复同一 unit                     → 停
    C  其他异常（CUDA OOM 同类）         → 停，不计入接续，unit 是 failed 非 excluded
    D  全链累计上限 5 + 重启不清零       → 第 6 次被挡
    E  **接管**的进程未知退出（被杀）    → 停；且**不白送一次重启**
    E2 **接管**的进程正常分段结束        → 继续下一步，最终完成
    F  残缺 manifest（缺 unit）          → 不得判完成
    G  重复 raw 行                       → 不得判完成
    H  已登记的排除 unit 消失            → 停
    I  旧段日志的 forward_nonfinite      → 不得用来解释本次未知退出

用法：
    python scripts/diagnostics/verify_chain_failure_policy.py [--tmp-root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import chain_failure_policy as pol                                    # noqa: E402
import run_llama_production_chain as chain                            # noqa: E402

PY = sys.executable
N_PROMPTS = 4
MCQ_EXTRA = ["--budget-registration",
             str(ROOT / "outputs/stage1_llama_v3/budget512_registration.json")]

LAUNCHER = r'''
import os, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_stage1_production as rp
from recover_attention.generation import backends as gb

_orig = rp._dry_backend
def _patched(arm, cfg, **kw):
    be, tok = _orig(arm, cfg, **kw)
    ats = {{int(x) for x in os.environ.get("FAULT_AT_CALLS", "").split(",") if x}}
    kind = os.environ.get("FAULT_KIND", "none")
    sf = Path(os.environ["FAULT_STATE"])          # **跨进程**累计，接续会起新进程
    inner = be.run_unit
    def run_unit(unit, record):
        n = (int(sf.read_text()) if sf.exists() else 0) + 1
        sf.write_text(str(n))
        if n in ats:
            if kind == "nonfinite":
                raise gb.ForwardNonFiniteError(
                    "synthetic non-finite",
                    diagnostic={{"reason": "forward_nonfinite", "synthetic": True,
                                "row_nonfinite": 1, "row_size": 8}})
            if kind == "other":
                raise RuntimeError("synthetic non-nonfinite failure")
            if kind == "hardkill":
                # 模拟被 OOM killer / 驱动复位干掉：**立刻**死，不走异常处理、
                # 不写段日志。靠 Timer 掐时间会有竞态（假后端太快，往往已经跑完了）。
                sys.stderr.write("synthetic hard kill\n"); sys.stderr.flush()
                os._exit(137)
        return inner(unit, record)
    be.run_unit = run_unit
    return be, tok
rp._dry_backend = _patched
raise SystemExit(rp.main())
'''

TMP_ROOT: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
def _plan_for_prompts(n: int) -> set[str]:
    """`--dry-run-prompts n` 截断后的**计划名单**，与生产用的是同一条 K 规则。"""
    from types import SimpleNamespace
    from recover_attention.evaluation import v3_g3_smoke as g3
    from recover_attention.evaluation import v3_production as prod
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    v = prod.verify_production_inputs(
        samples_dir=ROOT / "outputs/stage1/samples", g3_dir=chain.G3, g4_dir=chain.G4,
        prereg=ROOT / "docs/paper/preregistration_v3.md",
        lock=ROOT / "docs/paper/preregistration_v3.lock", cfg=cfg, arm="mcq")
    stub = SimpleNamespace(apply_chat_template=lambda *a, **k: "")
    recs = g3.build_arm_records("mcq", v["_primary_rows"], cfg=cfg, tokenizer=stub)[:n]
    return pol.planned_unit_ids((r["example_id"] for r in recs), cfg.k_traces)


def _wire(tmp: Path, *, fault_at, kind: str) -> Path:
    launcher = tmp / "fault_launcher.py"
    launcher.write_text(LAUNCHER.format(root=str(ROOT)), encoding="utf-8")
    out = tmp / "production"
    out.mkdir(parents=True, exist_ok=True)
    chain.OUT_DIR = out
    chain.LOG_DIR = tmp / "chain"
    chain.STATUS = chain.LOG_DIR / "chain_status.json"
    chain.AUTORESUME = chain.LOG_DIR / "chain_autoresume.json"
    chain.LOG_DIR.mkdir(parents=True, exist_ok=True)
    chain.BASE = [PY, str(launcher), "--dry-run", "--dry-run-prompts", str(N_PROMPTS),
                  "--model-path", "D:/models/Meta-Llama-3.1-8B-Instruct",
                  "--geometry-from-model-config",
                  "--g3-dir", str(chain.G3), "--g4-dir", str(chain.G4),
                  "--population-role", "cross_model_replication_non_confirmatory",
                  "--out-dir", str(out)]
    ats = fault_at if isinstance(fault_at, (list, tuple)) else [fault_at]
    os.environ["FAULT_AT_CALLS"] = ",".join(str(x) for x in ats)
    os.environ["FAULT_KIND"] = kind
    os.environ["FAULT_STATE"] = str(tmp / "fault_state.txt")
    return launcher


def _new_tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"chainfault_{name}_", dir=TMP_ROOT))


def _observe(tmp: Path) -> dict:
    p = tmp / "production"
    m = pol.manifest_state(p / "mcq_production_manifest.jsonl")
    raw = pol.raw_unit_ids(p / "mcq_raw_unit_results.jsonl")
    stat = json.loads((tmp / "chain" / "chain_status.json").read_text(encoding="utf-8")) \
        if (tmp / "chain" / "chain_status.json").exists() else {"log": []}
    return {"total": m.total, "done": m.done, "excluded": m.excluded,
            "raw_rows": len(raw), "raw_unique": len(set(raw)),
            "steps": [e["step"] for e in stat["log"]],
            "stop_reasons": [e.get("why", "") for e in stat["log"] if "STOP" in e["step"]],
            "autoresume_units": pol.load_counter(chain.AUTORESUME).get("units", [])}


def _segment_run(tmp: Path, extra_args: list[str]) -> subprocess.Popen:
    """直接起一个真实生成段（不经接续链）—— 用来造"被接管的既有进程"。"""
    log = (tmp / "chain" / "adopted.log").open("a", encoding="utf-8", errors="replace")
    return subprocess.Popen(chain.BASE + ["--arm", "mcq"] + MCQ_EXTRA + extra_args,
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def _run_chain(tmp: Path, *, adopt_pid: int | None = None, max_resumes: int = 2) -> bool:
    return chain._arm_generation("mcq", MCQ_EXTRA, max_resumes, adopt_pid=adopt_pid)


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    global TMP_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tmp-root", default=None)
    ap.add_argument("--keep", action="store_true", help="保留临时目录便于事后查看")
    args = ap.parse_args()
    TMP_ROOT = args.tmp_root

    plan = _plan_for_prompts(N_PROMPTS)
    saved = (chain.OUT_DIR, chain.LOG_DIR, chain.STATUS, chain.AUTORESUME, chain.BASE,
             dict(chain._PLAN_CACHE))
    checks: list[tuple[str, bool, str]] = []
    results: list[dict] = []
    tmps: list[Path] = []

    def case(name, *, fault_at, kind, preset=None, adopt=None, mutate=None,
             pre_segment=None) -> dict:
        tmp = _new_tmp(name); tmps.append(tmp)
        _wire(tmp, fault_at=fault_at, kind=kind)
        chain._PLAN_CACHE["mcq"] = set(plan)       # 截断名单（生产走真实 10,560）
        if preset:
            chain.AUTORESUME.write_text(json.dumps(
                {"schema": "stage1_chain_autoresume_v1", "units": list(preset),
                 "events": [{"t": "-", "arm": "mcq", "unit_id": u, "evidence": {}}
                            for u in preset]}, ensure_ascii=False), encoding="utf-8")
        if pre_segment:
            pre_segment(tmp)
        if mutate:
            mutate(tmp)
        pid = None
        if adopt is not None:
            pid = adopt(tmp)
        before = _observe(tmp)
        ok = _run_chain(tmp, adopt_pid=pid)
        res = {"case": name, "returned_ok": ok, "before": before, **_observe(tmp),
               "tmp": str(tmp)}
        results.append(res)
        return res

    try:
        # ── A 合法 forward_nonfinite ─────────────────────────────────────────
        a = case("A_normal_resume", fault_at=[7], kind="nonfinite")
        checks += [
            ("A 接续后完成", a["returned_ok"], ""),
            ("A 计划 24 个 unit 全终态", a["done"] + a["excluded"] == len(plan) == 24,
             f"done={a['done']} excluded={a['excluded']} plan={len(plan)}"),
            ("A 排除 1 条、raw 唯一集合 == done",
             a["excluded"] == 1 and a["raw_unique"] == a["done"] == 23,
             f"exc={a['excluded']} raw_uniq={a['raw_unique']} done={a['done']}"),
            ("A 自动接续记 1 次", len(a["autoresume_units"]) == 1,
             str(a["autoresume_units"])),
        ]

        # ── A2 多次不同 unit ──────────────────────────────────────────────────
        a2 = case("A2_multi_distinct", fault_at=[7, 15, 22], kind="nonfinite")
        checks += [
            ("A2 三次不同事件全部接续完", a2["returned_ok"], ""),
            ("A2 排除 3 条且计数 3 次互不相同",
             a2["excluded"] == 3 and len(set(a2["autoresume_units"])) == 3,
             f"exc={a2['excluded']} units={a2['autoresume_units']}"),
        ]

        # ── B 重复同一 unit ───────────────────────────────────────────────────
        probe = case("B_probe", fault_at=[7], kind="nonfinite")
        uid = probe["autoresume_units"][0]
        b = case("B_repeat_event", fault_at=[7], kind="nonfinite", preset=[uid])
        checks += [
            ("B 重复同一 unit → 停", not b["returned_ok"], ""),
            ("B 理由是「重复出现同一事件」",
             any("重复出现同一事件" in w for w in b["stop_reasons"]),
             str(b["stop_reasons"][-1:])),
            ("B 计数未增加", len(b["autoresume_units"]) == 1, str(b["autoresume_units"])),
        ]

        # ── C 其他异常（CUDA OOM 同类）────────────────────────────────────────
        c = case("C_other_exception", fault_at=[7], kind="other")
        checks += [
            ("C 其他异常 → 停", not c["returned_ok"], ""),
            ("C 未计入自动接续", c["autoresume_units"] == [], str(c["autoresume_units"])),
            ("C 该 unit 是 failed 非 excluded（可重入队）", c["excluded"] == 0, ""),
        ]

        # ── D 累计上限 + 重启不清零 ───────────────────────────────────────────
        d = case("D_cap", fault_at=[3, 7, 11, 15, 19, 23], kind="nonfinite")
        restart = pol.load_counter(chain.AUTORESUME).get("units", [])   # 重新读盘
        checks += [
            ("D 第 6 次被上限挡住 → 停", not d["returned_ok"], ""),
            ("D 理由是「已达上限 5」", any("已达上限 5" in w for w in d["stop_reasons"]),
             str(d["stop_reasons"][-1:])),
            ("D 恰好用掉 5 次", len(d["autoresume_units"]) == 5, str(len(d["autoresume_units"]))),
            ("D 重启后重新读盘仍为 5", len(restart) == 5, str(len(restart))),
        ]

        # ── E 接管的进程未知退出（被杀）—— 不得白送一次重启 ──────────────────
        #    注意：进程必须在**接续链开始等待之后**才死，否则边界快照就不是"运行中"了。
        def _adopt_killed(tmp):
            # 第 4 个 unit 处硬杀：前 3 条已落盘，**没有**段日志 —— 这正是被 OOM
            # killer 干掉的形态。不用 Timer，避免"假后端跑太快、已经正常结束"的竞态。
            return _segment_run(tmp, []).pid
        e = case("E_adopted_killed", fault_at=[4], kind="hardkill", adopt=_adopt_killed)
        checks += [
            ("E 接管进程被杀 → 停", not e["returned_ok"], ""),
            ("E 确实死在半路（有 raw 行但没跑完）",
             0 < e["raw_rows"] < len(plan), f"raw={e['raw_rows']} plan={len(plan)}"),
            ("E 理由是「无法确认」/「PID 消失不等于正常结束」",
             any(("unconfirmed_exit" in w or "PID 消失" in w or "段日志" in w)
                 for w in e["stop_reasons"]), str(e["stop_reasons"][-1:])),
            ("E **没有**白送一次重启（接续链未起新生成进程）",
             not any(s0.startswith("run:") for s0 in e["steps"]), str(e["steps"])),
            ("E 未计入自动接续", e["autoresume_units"] == [], ""),
        ]

        # ── E2 接管的进程正常分段结束 → 继续到完成 ────────────────────────────
        def _adopt_normal(tmp):
            return _segment_run(tmp, ["--max-new-units", "6"]).pid
        e2 = case("E2_adopted_normal_end", fault_at=[], kind="none", adopt=_adopt_normal)
        checks += [
            ("E2 接管进程正常分段结束 → 继续并完成", e2["returned_ok"], ""),
            ("E2 计划全终态、无排除",
             e2["done"] == len(plan) and e2["excluded"] == 0,
             f"done={e2['done']} plan={len(plan)}"),
            ("E2 确实起了续跑进程（与 E 对照）",
             any(s0.startswith("run:") for s0 in e2["steps"]), str(e2["steps"])),
        ]

        # ── F 残缺 manifest：不得因为 manifest.total 跟着变小就判完成 ──────────
        f_probe = {}

        def _drop_manifest_unit(tmp):
            m = tmp / "production" / "mcq_production_manifest.jsonl"
            lines = m.read_text(encoding="utf-8").splitlines()
            victim = json.loads(lines[-1])["unit_id"]
            m.write_text("\n".join(x for x in lines
                                    if json.loads(x)["unit_id"] != victim) + "\n",
                         encoding="utf-8")
            st = pol.manifest_state(m)
            raw = pol.raw_unit_ids(tmp / "production" / "mcq_raw_unit_results.jsonl")
            f_probe["victim"] = victim
            f_probe["manifest_total"] = st.total
            f_probe["all_terminal"] = st.all_terminal
            f_probe["complete_vs_plan"] = pol.arm_complete(st, raw, plan)[0]

        f = case("F_truncated_manifest", fault_at=[], kind="none",
                 pre_segment=lambda t: _segment_run(t, []).wait(),
                 mutate=_drop_manifest_unit)
        first_step = next((s0 for s0 in f["steps"]
                           if s0.endswith(("generation_start", "generation_complete"))), "")
        checks += [
            ("F 残缺后 manifest.total 确实变小了（旧判据会被骗）",
             f_probe["manifest_total"] == len(plan) - 1 and f_probe["all_terminal"],
             f"total={f_probe['manifest_total']} all_terminal={f_probe['all_terminal']}"),
            ("F 绑定计划名单后判为**未完成**", f_probe["complete_vs_plan"] is False, ""),
            ("F 接续链没有直接宣布完成，而是先去核验",
             first_step.endswith("generation_start"), first_step),
            ("F 收尾时 raw 唯一集合 == done 且无重复",
             f["raw_rows"] == f["raw_unique"] == f["done"],
             f"rows={f['raw_rows']} uniq={f['raw_unique']} done={f['done']}"),
        ]

        # ── G 重复 raw 行 ─────────────────────────────────────────────────────
        def _dup_raw(tmp):
            r = tmp / "production" / "mcq_raw_unit_results.jsonl"
            lines = r.read_text(encoding="utf-8").splitlines()
            r.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
        g = case("G_duplicate_raw", fault_at=[], kind="none",
                 pre_segment=lambda t: _segment_run(t, []).wait(), mutate=_dup_raw)
        checks += [
            ("G 重复 raw 不得判完成", not any(
                s.endswith("generation_complete") for s in g["steps"]), str(g["steps"])),
            ("G 最终停止", not g["returned_ok"], ""),
        ]

        # ── H 已登记的排除 unit 消失 ──────────────────────────────────────────
        def _drop_excluded(tmp):
            m = tmp / "production" / "mcq_production_manifest.jsonl"
            lines = m.read_text(encoding="utf-8").splitlines()
            m.write_text("\n".join(x for x in lines
                                    if json.loads(x)["unit_id"] != uid) + "\n",
                         encoding="utf-8")
            # 关掉注入：让接下来那一段**不再**制造同一个排除，
            # 复核才会落在"已登记的排除 unit 不见了 / 状态变了"这个状态上
            os.environ["FAULT_AT_CALLS"] = ""
        h = case("H_excluded_unit_vanished", fault_at=[7], kind="nonfinite", preset=[uid],
                 pre_segment=lambda t: _segment_run(t, ["--max-new-units", "6"]).wait(),
                 mutate=_drop_excluded)
        checks += [
            ("H 已登记排除 unit 消失 → 停", not h["returned_ok"], ""),
            ("H 理由指向「消失」或状态被改动",
             any(("消失" in w or "被改动" in w) for w in h["stop_reasons"]),
             str(h["stop_reasons"][-1:])),
        ]

        # ── I 旧段日志的 forward_nonfinite 不得解释本次未知退出 ────────────────
        def _adopt_killed_after_nonfinite(tmp):
            # 先制造一次**真实的** forward_nonfinite（写下段日志 + 诊断 + excluded）
            os.environ["FAULT_AT_CALLS"] = "7"
            os.environ["FAULT_KIND"] = "nonfinite"
            _segment_run(tmp, []).wait()
            # 再起一个进程并让它**硬杀**在半路 —— 它不会留下新段日志。
            # 接续器此时能看到的最后一条段日志仍是那次 forward_nonfinite。
            # 触发点要按**跨进程累计**的计数往后挪，否则序号早就过去了、根本不触发。
            sf = Path(os.environ["FAULT_STATE"])
            os.environ["FAULT_AT_CALLS"] = str(int(sf.read_text()) + 2)
            os.environ["FAULT_KIND"] = "hardkill"
            return _segment_run(tmp, []).pid

        i = case("I_stale_segment_vs_new_kill", fault_at=[7], kind="nonfinite",
                 adopt=_adopt_killed_after_nonfinite)
        last_seg = pol.segment_entries(
            Path(i["tmp"]) / "production" / "mcq_segment_log.jsonl")[-1] \
            if (Path(i["tmp"]) / "production" / "mcq_segment_log.jsonl").exists() else {}
        checks += [
            ("I 旧 forward_nonfinite 不得解释本次未知退出", not i["returned_ok"], ""),
            ("I 段日志里最后一条确实是 forward_nonfinite（诱饵在位）",
             str(last_seg.get("stopped_reason", "")) == pol.RESUMABLE_STOPPED_REASON,
             str(last_seg.get("stopped_reason"))),
            ("I 理由是「无法确认」而非接续",
             any(("unconfirmed_exit" in w or "段日志" in w) for w in i["stop_reasons"])
             and not any(s0.endswith("autoresume_forward_nonfinite")
                         for s0 in i["steps"]),
             str(i["stop_reasons"][-1:])),
            ("I 未白送重启（接续链未起新生成进程）",
             not any(s0.startswith("run:") for s0 in i["steps"]), str(i["steps"])),
        ]
    finally:
        (chain.OUT_DIR, chain.LOG_DIR, chain.STATUS, chain.AUTORESUME, chain.BASE,
         cached) = saved
        chain._PLAN_CACHE.clear(); chain._PLAN_CACHE.update(cached)
        for k in ("FAULT_AT_CALLS", "FAULT_KIND", "FAULT_STATE"):
            os.environ.pop(k, None)
        if not args.keep:
            for t in tmps:
                shutil.rmtree(t, ignore_errors=True)

    for label, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    bad = [c for c, ok, _ in checks if not ok]
    out = ROOT / "tmp/stage1_l2_g3/chain_failure_policy_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema": "chain_failure_policy_verification_v2", "date": "2026-09-15",
         "what": ("端到端合成故障：真实 run_stage1_production.main() + 假后端，"
                  "只注入单条 unit 的异常；独立临时目录，不碰正在运行的生成"),
         "planned_units_in_fixture": len(plan),
         "checks": [{"check": c, "ok": ok, "detail": d} for c, ok, d in checks],
         "n_failed": len(bad), "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n[{'ok   ' if not bad else 'STOP '}] {len(checks) - len(bad)}/{len(checks)} 通过")
    print(f"[out  ] {out}")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
