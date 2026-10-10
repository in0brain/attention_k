"""tail-stabilized 正式四段的**最小串行 driver** —— performance-blind，不解盲。

自动化的只是**机械门**，不是失败恢复：任何异常一律立即停下，
**不自动续跑、不重试、不修复、不删除**。这不是承诺而是结构事实 ——
本文件里没有任何重试循环、没有任何删除或改写产物的调用（有测试逐串断言）。

    H1  4,000 → H1 4,000 → H1 664 → MCQ 1,750     严格串行，一次只有一个子进程

每段用**独立子进程**执行 `run_stage1_tail_stabilized.py --run ... --defer-report`，
段与段之间没有任何并发；这既是"绝不并行"的执行保证（本机 NVML 峰值余量只有约
812 MiB），也让每段的模型加载彼此隔离。

段后八道硬门（`evaluate_segment`，纯函数，可单测）：

    ① 子进程 exit 0
    ② 新增且仅新增一个 segment 命名空间，segment_close 唯一
    ③ segment_close 是 completed / exit_code=0 / failed=0，且 completed == 计划数
    ④ 新增 done 数**精确**等于 repair + min(max_new, todo)；failed / problems /
      未知 ID 全为 0
    ⑤ 采样器全程有效：telemetry SHA 与样本数与 segment_close 吻合，
      采样时间跨度覆盖整段
    ⑥ NVML peak headroom ≥ 500 MiB
    ⑦ source freeze 逐项不变
    ⑧ manifest / 两臂 F5 的**旧前缀逐字节不变**（append-only）

任一不满足：立即退出非零，**不启动下一段**。四段全部通过后才调用一次
`--finalize` 发布 completion report —— 发布终态与"跑一段"是分开的两件事。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_stage1_tail_stabilized.py"

#: **Qwen 冻结运行**的四段计划。段长由该次 32/128 微基准实测决定
#: （见 benchmark report 与 addendum）。它是 Qwen 的结果，**不是**通用计划：
#: 另一个模型的 analyzable 数与 s/trace 都不同，必须由 `build_segment_plan()`
#: 按**本次**实际计数与**本机** benchmark 重新生成。
SEGMENT_PLAN = (("h1", 4000), ("h1", 4000), ("h1", 664), ("mcq", 1750))

#: 段长规则（与 runner 的 `_segment_plan` 同源口径）。
SEG_MAX_TRACES = 4000
SEG_TARGET_SECONDS = 1200


def build_segment_plan(counts, seconds_per_trace) -> tuple:
    """按**本次实际计数**与**本机 benchmark 实测速率**生成分段计划。

    `counts` = {"h1": n, "mcq": n}，来自 finalized 输入现场重建；
    `seconds_per_trace` = {"h1": s, "mcq": s}，来自本机 tail benchmark 报告。
    段长 = min(SEG_MAX_TRACES, floor(SEG_TARGET_SECONDS / s))，最后一段取余。

    **绝不照搬 Qwen 的 (4000, 4000, 664, 1750)** —— 那四个数字同时编码了
    Qwen 的 analyzable 数和 Qwen 的 s/trace，对另一个模型两者都不成立。
    """
    plan = []
    for arm in ("h1", "mcq"):
        total = int(counts[arm])
        s_per = float(seconds_per_trace[arm])
        if total <= 0:
            raise DriverStop(f"{arm}: 计数必须为正，得到 {total}")
        if not (s_per > 0):
            raise DriverStop(f"{arm}: s/trace 必须为正，得到 {s_per}")
        seg = max(1, min(SEG_MAX_TRACES, int(SEG_TARGET_SECONDS // s_per)))
        # 刻意用 range 切块：driver 有一条结构性测试禁止本文件出现条件循环关键字，
        # 它保证的是"driver 不含任何重试循环"。分段是纯算术，不需要那种控制流，
        # 不该为它破掉这条保证。
        plan.extend((arm, min(seg, total - start)) for start in range(0, total, seg))
    return tuple(plan)


def load_benchmark_rates(runner) -> dict:
    """从本机 tail benchmark 报告读 128 条口径的 s/trace。缺报告即停。"""
    path = runner.BENCH_OUT
    if not path.exists():
        raise DriverStop(
            f"缺少 tail benchmark 报告 {path} —— 分段计划必须由本机实测生成，"
            f"不得沿用另一模型的段长")
    rep_json = json.loads(path.read_text(encoding="utf-8"))
    ex = rep_json.get("extrapolation") or {}
    rates = {"h1": ex.get("h1_seconds_per_trace_from_128"),
             "mcq": ex.get("mcq_seconds_per_trace_from_128")}
    missing = sorted(a for a, v in rates.items() if not v)
    if missing:
        raise DriverStop(f"benchmark 报告缺 {missing} 的 s/trace")
    return rates

MIN_HEADROOM_MIB = 500
TELEMETRY_INTERVAL_SECONDS = 5
#: 采样跨度允许比 elapsed 短这么多：首点延迟 + 收尾 terminate 的边界。
TELEMETRY_SPAN_SLACK_SECONDS = 20
TELEMETRY_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


class DriverStop(RuntimeError):
    """机械门不满足 —— 立即停止，不启动下一段。"""


def load_runner():
    spec = importlib.util.spec_from_file_location("_tail_runner", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 遥测解析
def gpu_total_mib() -> int:
    """`nvidia-smi` 的 memory.total。Windows 控制台非 UTF-8，必须 errors="replace"。"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30, check=True)
    return int(out.stdout.strip().splitlines()[0].strip())


def telemetry_stats(path: Path) -> dict:
    """一份段遥测 CSV 的实测摘要。读不到就返回全空，由调用方判死。"""
    path = Path(path)
    if not path.exists():
        return {"exists": False, "samples": 0, "sha256": None,
                "span_seconds": None, "peak_used_mib": None}
    blob = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8", errors="replace"))))
    rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]
    stamps, used = [], []
    for r in rows:
        try:
            stamps.append(datetime.strptime(r["timestamp"], TELEMETRY_TIMESTAMP_FORMAT))
            used.append(int(r["memory.used [MiB]"].split()[0]))
        except (KeyError, ValueError, IndexError):
            continue
    span = ((stamps[-1] - stamps[0]).total_seconds() if len(stamps) >= 2 else 0.0)
    return {"exists": True, "samples": len(rows),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "span_seconds": span, "peak_used_mib": (max(used) if used else None),
            "n_parsed": len(stamps)}


# ------------------------------------------------------------------ 快照
def snapshot(runner, cfg, h1_rows, mcq_rows) -> dict:
    """段前 / 段后的完整现场。`formal_status` 只读磁盘真相。"""
    p = runner.tail_paths()
    return {
        "freeze": runner.source_freeze(),
        "status": runner.formal_status(cfg, h1_rows, mcq_rows),
        "bytes": {k: (p[k].read_bytes() if p[k].exists() else b"")
                  for k in ("manifest", "h1", "mcq")},
        "segments": ({q.stem for q in p["dir"].glob("tail_seg*_gpu_telemetry.csv")}
                     if p["dir"].exists() else set()),
    }


def segment_key(telemetry_path) -> str:
    """manifest 里 `segment` 字段的取值。

    **必须与 runner 的 `seg_path.stem` 同一口径** —— 即 `tail_seg001_gpu_telemetry`，
    带 `_gpu_telemetry` 后缀。第一次实跑就栽在这里：driver 把后缀剥掉去查
    `tail_seg001`，于是一段跑得完全正常却被判成"segment_close 有 0 条"。
    这个函数存在的唯一理由就是让那条口径只有一处定义，并且可被单测钉住。
    """
    return Path(telemetry_path).stem


def telemetry_path_for(runner, key: str) -> Path:
    return runner.tail_paths()["dir"] / f"{key}.csv"


def segment_close_events(runner, key: str) -> list[dict]:
    events = runner.read_manifest_events(runner.tail_paths()["manifest"])
    return [e for e in events
            if e.get("kind") == "segment_close" and e.get("segment") == key]


def should_skip_segment(arm_status: dict, max_new: int) -> tuple[bool, str | None]:
    """断点续跑：某个计划段可能已被之前的段覆盖完。

    返回 `(skip, error)`。只有"本段确实没有任何待跑或待补记，**且本臂已闭合**"
    才允许跳过；"没活干但本臂没闭合"是不自洽状态，必须报错而不是静悄悄跳过。

    注意：续跑时实际段边界可能与计划不同（例如第一段跑完后剩 4,664，
    后续会变成 4,000 + 664 两段而不是 4,000 + 664 + 0）。这不影响任何不变量 ——
    段长上限与每段时长仍然成立 —— 但会让计划里的某一项无活可干。
    """
    expected_gpu = min(max_new, int(arm_status["n_todo_gpu"]))
    if expected_gpu or int(arm_status["n_repair"]):
        return False, None
    if arm_status["complete"]:
        return True, None
    return False, (f"本段无待跑单元，但本臂尚未闭合（done "
                   f"{arm_status['n_done']}/{arm_status['n_expected']}，"
                   f"failed {arm_status['n_failed']}，problems "
                   f"{arm_status['n_problems']}）—— 状态不自洽，拒绝继续")


# ------------------------------------------------------------------ 机械门
def evaluate_segment(*, arm: str, max_new: int, proc_rc: int, before: dict, after: dict,
                     closes: list[dict], tel: dict, gpu_total: int,
                     expect_arm_complete: bool, segment_key_used: str) -> list[str]:
    """段后八道门的**纯**判定。返回失败清单（空 = 全过）。

    `segment_key_used` 是调用方用来查 `segment_close` 的键；这里再核一次它确实等于
    实测新增的段名 —— 否则"查不到 close"到底是段没写完还是键取错了，报错分不清。
    """
    fails: list[str] = []
    a_before, a_after = before["status"]["arms"][arm], after["status"]["arms"][arm]

    # ① 子进程退出码
    if proc_rc != 0:
        fails.append(f"① 子进程 exit={proc_rc} != 0")

    # ② 恰好新增一个段命名空间
    new_segments = sorted(after["segments"] - before["segments"])
    if len(new_segments) != 1:
        fails.append(f"② 新增段命名空间 {new_segments}（期望恰好 1 个）")
        return fails                        # 段名不确定，后续门无从判起
    stem = new_segments[0]
    if segment_key_used != stem:
        fails.append(f"② 查询用的段键 {segment_key_used!r} != 实测新增段名 {stem!r}"
                     "（driver 自身口径错误，不是本段跑失败）")
        return fails

    # ③ segment_close 唯一且是正常收场
    if len(closes) != 1:
        fails.append(f"③ {stem} 的 segment_close 有 {len(closes)} 条（期望 1 条）")
        return fails
    ev = closes[0]
    if ev.get("segment") != stem:
        fails.append(f"③ segment_close.segment={ev.get('segment')!r} != {stem!r}")
    if ev.get("stopped_reason") != "completed":
        fails.append(f"③ stopped_reason={ev.get('stopped_reason')!r} != 'completed'")
    if ev.get("exit_code") != 0:
        fails.append(f"③ segment_close.exit_code={ev.get('exit_code')!r} != 0")
    if ev.get("failed"):
        fails.append(f"③ segment_close.failed={ev.get('failed')!r} != 0")

    # ④ 新增 done 数精确 + failed / problems / 未知 ID 归零
    expected_gpu = min(max_new, int(a_before["n_todo_gpu"]))
    expected_new = expected_gpu + int(a_before["n_repair"])
    got_new = int(a_after["n_done"]) - int(a_before["n_done"])
    if got_new != expected_new:
        fails.append(f"④ 新增 done {got_new} != 期望 {expected_new}"
                     f"（repair {a_before['n_repair']} + min({max_new}, "
                     f"{a_before['n_todo_gpu']})）")
    if ev.get("completed") != expected_gpu:
        fails.append(f"④ segment_close.completed={ev.get('completed')!r} != {expected_gpu}")
    if a_after["n_failed"]:
        fails.append(f"④ {arm} failed={a_after['n_failed']} != 0")
    if a_after["n_problems"]:
        fails.append(f"④ {arm} problems={a_after['problems'][:3]}")
    if after["status"]["global_problems"]:
        fails.append(f"④ manifest 全局问题 {after['status']['global_problems'][:3]}")
    if expect_arm_complete and not a_after["complete"]:
        fails.append(f"④ 本臂应在此段闭合，但 complete={a_after['complete']}"
                     f"（f5_key_set={a_after['f5_key_set_matches_expected']}，"
                     f"npz_key_set={a_after['npz_key_set_matches_expected']}）")

    # ⑤ 采样器全程有效
    if not tel["exists"]:
        fails.append(f"⑤ 段遥测文件不存在：{stem}")
    else:
        if tel["sha256"] != ev.get("telemetry_sha256"):
            fails.append("⑤ 遥测 SHA 与 segment_close 记录不符")
        if tel["samples"] != ev.get("telemetry_samples"):
            fails.append(f"⑤ 遥测样本数 {tel['samples']} != 记录 "
                         f"{ev.get('telemetry_samples')!r}")
        if tel["samples"] < 1:
            fails.append("⑤ 遥测没有任何采样点")
        elapsed = float(ev.get("elapsed_seconds") or 0.0)
        if tel["span_seconds"] is not None and \
                tel["span_seconds"] < elapsed - TELEMETRY_SPAN_SLACK_SECONDS:
            fails.append(f"⑤ 采样跨度 {tel['span_seconds']:.0f}s 未覆盖整段 "
                         f"{elapsed:.0f}s —— 采样器中途死了")

    # ⑥ NVML peak headroom
    if tel.get("peak_used_mib") is None:
        fails.append("⑥ 无法从遥测得到 peak memory.used")
    else:
        headroom = gpu_total - int(tel["peak_used_mib"])
        if headroom < MIN_HEADROOM_MIB:
            fails.append(f"⑥ NVML peak headroom {headroom} MiB < {MIN_HEADROOM_MIB} MiB"
                         f"（peak used {tel['peak_used_mib']} / total {gpu_total}）")

    # ⑦ source freeze 逐项不变
    drift = [k for k in sorted(set(before["freeze"]) | set(after["freeze"]))
             if before["freeze"].get(k) != after["freeze"].get(k)]
    if drift:
        fails.append(f"⑦ source freeze 漂移：{drift}")

    # ⑧ append-only 旧前缀逐字节不变
    for key in ("manifest", "h1", "mcq"):
        old, new = before["bytes"][key], after["bytes"][key]
        if not new.startswith(old):
            fails.append(f"⑧ {key} 的旧前缀被改写（append-only 被破坏）")

    return fails


# ------------------------------------------------------------------ 执行
def run_segment(arm: str, max_new: int, model_path: str) -> int:
    cmd = [sys.executable, "-X", "utf8", str(RUNNER_PATH), "--run",
           "--arm", arm, "--max-new-traces", str(max_new),
           "--model-path", model_path, "--defer-report"]
    print(f"\n[driver] $ {' '.join(cmd[2:])}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def run_finalize_subprocess() -> int:
    cmd = [sys.executable, "-X", "utf8", str(RUNNER_PATH), "--finalize"]
    print(f"\n[driver] $ {' '.join(cmd[2:])}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印计划与段前现场，不启动任何子进程")
    args = ap.parse_args(argv)

    runner = load_runner()
    cfg = runner.V3AnalysisConfig()
    h1_rows, mcq_rows = runner.load_analyzable("h1"), runner.load_analyzable("mcq")
    gpu_total = gpu_total_mib()
    print(f"[driver] GPU total {gpu_total} MiB ｜ headroom 门槛 {MIN_HEADROOM_MIB} MiB")
    # 分段计划：默认 artifact root（Qwen）沿用冻结的四段；独立 root 下由
    # **本次实际计数 + 本机 benchmark** 重新生成，绝不照搬 Qwen 的段长。
    scale = runner.observed_scale(cfg, h1_rows, mcq_rows)
    if runner.ARTIFACT_ROOT == runner.DEFAULT_ARTIFACT_ROOT:
        plan = SEGMENT_PLAN
        plan_source = "frozen_qwen_four_segments"
    else:
        plan = build_segment_plan(scale, load_benchmark_rates(runner))
        plan_source = "rebuilt_from_observed_scale_and_local_benchmark"
    print(f"[driver] 分段计划 {list(plan)}（{plan_source}；严格串行，一次一个子进程）")
    for arm in ("h1", "mcq"):
        planned = sum(n for a, n in plan if a == arm)
        if planned != scale[arm]:
            print(f"[driver] {arm} 计划总量 {planned} != 本次实测规模 {scale[arm]}",
                  file=sys.stderr)
            return 6

    state = snapshot(runner, cfg, h1_rows, mcq_rows)
    for arm in ("h1", "mcq"):
        a = state["status"]["arms"][arm]
        print(f"[driver] 段前 {arm}: done {a['n_done']:,}/{a['n_expected']:,} ｜ "
              f"待跑 {a['n_todo_gpu']:,} ｜ 待补记 {a['n_repair']}")
    if args.dry_run:
        print("[driver] --dry-run：不启动任何子进程")
        return 0

    for idx, (arm, max_new) in enumerate(plan, 1):
        a = state["status"]["arms"][arm]
        skip, err = should_skip_segment(a, max_new)
        if err:
            print(f"\n[driver] 第 {idx} 段拒绝启动：{err}", file=sys.stderr)
            return 6
        if skip:
            print(f"\n[driver] 第 {idx}/{len(plan)} 段：{arm} × {max_new:,} "
                  f"—— 本臂已闭合（{a['n_done']:,}/{a['n_expected']:,}），"
                  f"该计划段已被此前的段覆盖，跳过", flush=True)
            continue
        remaining_after = a["n_todo_gpu"] - min(max_new, a["n_todo_gpu"])
        print(f"\n[driver] ===== 第 {idx}/{len(plan)} 段：{arm} × {max_new:,} "
              f"（本臂跑完后剩 {remaining_after:,}）=====", flush=True)
        before = state
        rc = run_segment(arm, max_new, args.model_path)
        after = snapshot(runner, cfg, h1_rows, mcq_rows)

        new_segments = sorted(after["segments"] - before["segments"])
        key = new_segments[0] if len(new_segments) == 1 else ""
        closes = segment_close_events(runner, key) if key else []
        tel = (telemetry_stats(telemetry_path_for(runner, key)) if key
               else {"exists": False})

        fails = evaluate_segment(
            arm=arm, max_new=max_new, proc_rc=rc, before=before, after=after,
            closes=closes, tel=tel, gpu_total=gpu_total, segment_key_used=key,
            expect_arm_complete=(remaining_after == 0))
        if fails:
            print(f"\n[driver] 第 {idx} 段机械门未过，**立即停止**，不启动下一段：",
                  file=sys.stderr)
            for f in fails:
                print(f"  {f}", file=sys.stderr)
            print("[driver] 不重试、不修复、不删除；请另出 Recovery Preflight。",
                  file=sys.stderr)
            return 6
        print(f"[driver] 第 {idx} 段八道门全过 ｜ 遥测 {tel['samples']} 点 ｜ "
              f"peak headroom {gpu_total - tel['peak_used_mib']} MiB")
        state = after

    print("\n[driver] 四段全部通过机械门 —— 现在才允许发布 completion report")
    rc = run_finalize_subprocess()
    if rc != 0:
        print(f"[driver] finalize 未成功（exit={rc}）", file=sys.stderr)
        return 7
    print("[driver] 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
