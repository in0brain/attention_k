r"""正式生成期间的**只读**健康监控。

只观察、不干预：不改任何文件、不碰产物、不动进程。
**不为监控修改正在跑的模型进程，也不改 allocator 配置**（那会让 MCQ 与 H1/gate
的运行环境不一致）。每 `--interval` 秒往
`tmp/stage1_l2_g3/chain/health.jsonl` 追加一行快照，并在越过阈值时另起一行告警。

看的东西：
    进程     H1/MCQ 生成进程与接续链是否还活着（按 PID 精确匹配）
    内存     系统可用 RAM、生成进程的 WorkingSet / Private

             ⚠️ `private_grew` 的基线在进程预热 PRIVATE_BASELINE_WARMUP_S 之后才取 ——
             模型加载期 Private 只有 2–3 GB，拿它当基线会报出假告警。
             ⚠️ **快照稳定 ≠ 已排除泄漏或 OOM**。这里只是周期采样：
             慢泄漏要看**长时间趋势**（所以记了相对首个快照的增长），
             而瞬时峰值（一次长序列的前向）采样根本抓不到。
             "当前平稳"只说明采样时刻平稳。
    显存     nvidia-smi 的 used / free / 温度 / 功耗

             ⚠️ **nvidia-smi 分不清"当前进程真正在用"与"可释放的缓存"**。
             它报的 used 是驱动视角的显存占用，里面混着 torch caching allocator
             留着不还的空闲块；free 小**既不能**说明快 OOM，**也不能**说明还有余量。
             要区分得看进程内的 torch.cuda.memory_allocated / memory_reserved ——
             那需要在模型进程里取数，**本轮不做**（不为监控去动正在跑的模型进程）。
    磁盘     产物盘剩余空间
    进度     两臂 raw 行数、窗口速率、剩余时间估算
    停止     chain_status.json 里出现 STOP_* 即记为告警（不自动处理）

出事时会**弹一个红底 cmd 窗口**（默认开，`--no-popup` 关）。只对真正要人来看的事弹，
且按种类去重 + 冷却，不刷屏 —— 噪声型的（如 gpu_free）只记不弹。
窗口用 `cmd /k` 保持不关，内容同时落盘到 chain/alerts/，事后可查。
**弹窗正文是纯 ASCII 英文**：中文在 cmd 里会乱码，实测 chcp 65001 串在 `cmd /k`
里并不可靠。日志与 status.py 仍是中文。

用法：
    python scripts/monitor_production_health.py --interval 300
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/stage1_llama_v3/production"
LOG_DIR = ROOT / "tmp/stage1_l2_g3/chain"
HEALTH = LOG_DIR / "health.jsonl"
CHAIN_STATUS = LOG_DIR / "chain_status.json"

EXPECTED = {"h1": 9264, "mcq": 10560}

#: 告警阈值（只报，不动手）
#: ⚠️ 这是**监控提示阈值，不是安全容量保证**。
#: torch 的 caching allocator 会把峰值显存留着不还，所以 nvidia-smi 的 free 很小是常态；
#: 反过来，free 看着有几百 MiB 也**不代表**下一条更长的序列一定摆得下 ——
#: 真正的峰值需求取决于该条 trace 的 prompt+completion 长度，nvidia-smi 看不到。
#: 把它压到 48 只是为了少报噪声，**不构成"不会 OOM"的判断**。
GPU_FREE_MIB_WARN = 48
RAM_FREE_GB_WARN = 2.0
DISK_FREE_GB_WARN = 10.0
PRIVATE_GROWTH_GB_WARN = 4.0     # 相对**预热后**基线的 Private 增长 —— 疑似泄漏
#: 基线要等进程预热完再取。模型加载要一两分钟，期间 Private 只有 2–3 GB；
#: 拿那个当基线会立刻报出 +11 GB 的假告警（2026-09-17 实测撞到过）。
PRIVATE_BASELINE_WARMUP_S = 900
STALL_MINUTES_WARN = 20          # 两臂行数都不动这么久

#: **会弹窗**的告警种类 —— 只放需要人来处理的。
#: `gpu_free` 不在内：caching allocator 让它天然贴地，弹了也只是噪声（只记不弹）。
POPUP_KINDS = ("generation_process_gone", "chain_process_gone", "chain_step",
               "no_new_rows_for", "disk_free", "ram_free", "private_grew",
               "probe_process_gone", "probe_failed")
#: 同一种告警的冷却：持续存在时每隔这么久才再弹一次，不刷屏。
ALERT_COOLDOWN_S = 7200
#: 这些链步骤代表**跑完了**，也要弹一次 —— 远程盯不住的时候，
#: "干完了没人知道"和"出事没人知道"一样糟。绿底，与红底的异常区分开。
DONE_STEPS = ("mcq:finalized", "chain:complete")
ALERT_DIR = LOG_DIR / "alerts"

# ── post-hoc 探针实验（非 confirmatory，另存目录）──────────────────────────
#: 看门狗**只观测 + 告警，不自动重启**。自动接续的授权只限生成链的
#: forward_nonfinite 一种，不外扩到这里。
PROBE_DIR = ROOT / "outputs/stage1/posthoc_readout_probes"
PROBE_LOG = PROBE_DIR / "run.log"
#: 一臂 = 3 repeat × 5 折 × 条件数。**条件表一变这里就得跟着变** ——
#: 加 han_l1_wide 之后是 6 个条件，实测面板上一度显示 "76/75"。
#: 这里不 import 探针 lib（监控要保持轻量，每 5 分钟醒一次不值得拖 numpy 进来），
#: 所以只能手动同步，代价是可能忘 —— 故把它写在显眼处。
PROBE_N_CONDITIONS = 6
PROBE_TOTAL_PER_ARM = 3 * 5 * PROBE_N_CONDITIONS
#: 一折在争用下约 25 分钟（MCQ），H1 更久，所以停滞阈值给得宽。
PROBE_STALL_MINUTES_WARN = 180


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ps(pid: int) -> dict | None:
    """按 CSV 的 PID 字段精确匹配；顺带取内存。"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
         f"if($p){{'{{0}},{{1}},{{2}},{{3}}' -f $p.Id,$p.WorkingSet64,"
         f"$p.PrivateMemorySize64,$p.Threads.Count}}"],
        capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    try:
        pid_s, ws, pv, th = out.split(",")
        return {"pid": int(pid_s), "ws_gb": round(int(ws) / 2**30, 2),
                "private_gb": round(int(pv) / 2**30, 2), "threads": int(th)}
    except ValueError:
        return None


def _gpu() -> dict | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,"
         "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    try:
        used, free, util, temp, pw = (x.strip() for x in out[0].split(","))
        return {"used_mib": int(used), "free_mib": int(free), "util_pct": int(util),
                "temp_c": int(temp), "power_w": float(pw)}
    except ValueError:
        return None


def _ram_free_gb() -> float | None:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
        capture_output=True, text=True).stdout.strip()
    return round(int(out) / 2**20, 2) if out.isdigit() else None


def _find_pid(pattern: str) -> int | None:
    """按命令行特征找进程。

    换臂（h1 → mcq）与进程重启都会让 PID 变；写死 pid 会让对应那一栏悄悄变成
    null / false —— 实际吃过两次亏：MCQ 头五小时的进程内存没被记录，
    改计划任务后 chain_alive 又一直报 false。
    """
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
         "Select-Object -First 1).ProcessId"],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def _popup(title: str, body: str, color: str = "4F") -> Path | None:
    """写一份告警文本并**弹出一个红底 cmd 窗口**显示它。

    用 `cmd /k` 是刻意的 —— 窗口留在那儿不自动关，人不在的时候也跑不掉。
    `color`：4F 红底=异常，2F 绿底=跑完了。
    文本同时落盘，事后可查；弹窗失败（比如没有交互式桌面）不影响监控继续跑。
    """
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    f = ALERT_DIR / f"alert_{datetime.now():%Y%m%d_%H%M%S}.txt"
    # ascii 编码 + 显式报错：万一将来有人往正文里塞了中文，这里当场暴露，
    # 而不是等到弹出一屏乱码才发现。
    f.write_text(body, encoding="ascii", errors="backslashreplace")
    try:
        # color 4F = 红底亮白字；chcp 65001 让中文不变乱码
        # stdio 必须切断：弹出的窗口会继承父进程的管道句柄，
        # 于是"谁在读监控输出，谁就被这个不关的窗口挂住"（实测撞到过）。
        # stdio 必须切断：弹出的窗口会继承父进程的管道句柄，
        # 于是"谁在读监控输出，谁就被这个不关的窗口挂住"（实测撞到过）。
        # 不再折腾 chcp —— 正文已是纯 ASCII，任何代码页下都正常。
        subprocess.Popen(
            f'start "{title}" cmd /k "color {color} & type ""{f}"""',
            shell=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True)
    except OSError:
        pass                                   # 没有桌面就只留文件，不让监控挂掉
    return f


def _alert_body(headline: str, kinds: list[str], snap: dict) -> str:
    """告警正文**一律用纯 ASCII 英文**。

    中文在弹出的 cmd 里会乱码：`chcp 65001` 与 `type` 串在同一条 `cmd /k`
    命令里并不可靠，控制台字体也未必有字形（实测就是一屏乱码）。
    与其和代码页较劲，不如让这份文本不含非 ASCII —— 告警要的是看得懂，不是好看。
    落盘的 health.jsonl 与 status.py 仍是中文，不受影响。
    """
    g = snap.get("gpu") or {}
    gen = snap.get("gen") or {}
    chain = "UP" if snap.get("chain_alive") else "** DOWN **"
    return "\n".join([
        "=" * 70,
        f"  {headline}",
        f"  {snap['t']}",
        "=" * 70,
        "",
        "  TRIGGERED:",
        *[f"      - {k}" for k in kinds],
        "",
        f"  progress   h1  {snap['h1_rows']} / 9264"
        f"      mcq  {snap['mcq_rows']} / 10560",
        f"  chain step {snap.get('chain_step')}",
        f"  rate       {snap.get('s_per_unit_window')} s/unit"
        f"      arm ETA {snap.get('eta_hours_this_arm')} h",
        f"  processes  gen pid {gen.get('pid', 'GONE')}      chain {chain}",
        f"  memory     gen private {gen.get('private_gb', '?')} GB"
        f"      system free {snap.get('ram_free_gb')} GB",
        f"  gpu        {g.get('temp_c')}C  {g.get('power_w')}W"
        f"  util {g.get('util_pct')}%  free {g.get('free_mib')} MiB",
        f"  disk       {snap.get('disk_free_gb')} GB free",
        "",
        "-" * 70,
        "  CHECK STATUS FIRST -- do not blindly restart:",
        "",
        "    D:\\conda\\Miniconda3\\envs\\recover_attention\\python.exe ^",
        "      D:\\Projects\\recover_attention_project\\scripts\\status.py",
        "",
        "  Data is append-only with per-unit fsync: a killed process does NOT",
        "  lose completed records. Verify artifacts before resuming.",
        "-" * 70,
        "",
        "  This window stays open. Closing it affects nothing.",
        "",
    ])


def _probe_progress() -> dict:
    """探针进度：数已完成的 (折 × 条件) 对。

    .npz 就是个 zip，用 `zipfile` 读目录项即可 —— **不 import numpy**，
    监控进程要保持轻量（它每 5 分钟醒一次，不该为了数个数把 numpy 拖进来）。
    """
    import zipfile
    out = {}
    for arm in ("mcq", "h1"):
        d = PROBE_DIR / f"run_{arm}"
        done, newest = 0, None
        if d.is_dir():
            for f in d.glob("shard_*.npz"):
                try:
                    with zipfile.ZipFile(f) as z:
                        done += sum(1 for n in z.namelist() if n.endswith("|H.npy"))
                except (zipfile.BadZipFile, OSError):
                    continue
                m = f.stat().st_mtime
                newest = m if newest is None else max(newest, m)
        out[arm] = {"done": done, "total": PROBE_TOTAL_PER_ARM,
                    # 光有 results 文件**不算完**：小规模验证那轮只跑了 3 个条件，
                    # 也会留下 results_mcq.json。必须 75 个 (折 × 条件) 都齐。
                    "finished": ((PROBE_DIR / f"results_{arm}.json").exists()
                                 and done >= PROBE_TOTAL_PER_ARM),
                    "newest_shard_age_min": (round((time.time() - newest) / 60, 1)
                                             if newest else None)}
    return out


def _probe_state() -> dict:
    """探针的存活/完成/失败判定。

    "进程没了"**不等于**出事：跑完了也会没。所以先看完成标记，再看失败标记，
    都不是才算异常退出。
    """
    pid = _find_pid("posthoc_readout_probes_stage1")
    prog = _probe_progress()
    tail = ""
    if PROBE_LOG.exists():
        try:
            tail = PROBE_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            tail = ""
    return {"pid": pid, "alive": pid is not None,
            "both_arms_done": "both arms done" in tail,
            "failed": ("mcq FAILED" in tail or "h1 FAILED" in tail),
            "arms": prog}


def _rows(arm: str) -> int:
    p = OUT_DIR / f"{arm}_raw_unit_results.jsonl"
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


def _chain_step() -> str | None:
    if not CHAIN_STATUS.exists():
        return None
    try:
        return (json.loads(CHAIN_STATUS.read_text(encoding="utf-8"))
                .get("current", {}).get("step"))
    except (ValueError, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--gen-pid", type=int, default=None,
                    help="固定某个生成进程 PID；**不给则每次快照自动查找**（推荐）")
    ap.add_argument("--chain-pid", type=int, default=None,
                    help="固定接续链 PID；**不给则每次快照自动查找**（推荐）")
    ap.add_argument("--hours", type=float, default=168.0, help="监控总时长上限")
    ap.add_argument("--no-popup", action="store_true",
                    help="出事只记录不弹窗（默认会弹一个红底 cmd 窗口）")
    ap.add_argument("--test-alert", action="store_true",
                    help="立刻弹一个测试告警窗口然后退出 —— 用来确认弹窗真的能弹出来")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.test_alert:
        f = _popup("STAGE1 ALERT (TEST)", _alert_body(
            "TEST ALERT -- nothing is actually wrong", ["test_alert"],
            {"t": _now(), "h1_rows": _rows("h1"), "mcq_rows": _rows("mcq"),
             "chain_step": _chain_step(), "s_per_unit_window": None,
             "eta_hours_this_arm": None, "chain_alive": True,
             "gen": _ps(_find_pid("run_stage1_production")) or {},
             "gpu": _gpu(), "ram_free_gb": _ram_free_gb(),
             "disk_free_gb": round(shutil.disk_usage(str(OUT_DIR)).free / 2**30, 1)}))
        print(f"[test] 已弹出测试窗口；文本 {f}")
        return 0

    last_alert: dict[str, float] = {}
    t_end = time.time() + args.hours * 3600
    first_private: float | None = None
    first_private_pid: int | None = None      # 换进程要重置基线，否则增长量没意义
    first_seen_at: float | None = None        # 该 pid 首次被看到的时刻（用于预热窗口）
    probe_gone_streak = 0
    last_rows, last_move = _rows("h1") + _rows("mcq"), time.time()
    prev_rows, prev_t = last_rows, time.time()

    while time.time() < t_end:
        h1, mcq = _rows("h1"), _rows("mcq")
        total = h1 + mcq
        now = time.time()
        gen_pid = args.gen_pid or _find_pid("run_stage1_production")
        chain_pid = args.chain_pid or _find_pid("run_llama_production_chain")
        gen = _ps(gen_pid) if gen_pid else None

        chain = _ps(chain_pid) if chain_pid else None
        gpu, ram = _gpu(), _ram_free_gb()
        disk_free = round(shutil.disk_usage(str(OUT_DIR)).free / 2**30, 1)

        rate = ((now - prev_t) / (total - prev_rows)) if total > prev_rows else None
        if total > last_rows:
            last_rows, last_move = total, now
        arm = "mcq" if mcq else "h1"
        remaining = EXPECTED[arm] - (mcq if mcq else h1)
        eta_h = round(remaining * rate / 3600, 1) if rate else None

        warn = []
        if gpu and gpu["free_mib"] < GPU_FREE_MIB_WARN:
            warn.append(f"gpu_free={gpu['free_mib']}MiB<{GPU_FREE_MIB_WARN}")
        if ram is not None and ram < RAM_FREE_GB_WARN:
            warn.append(f"ram_free={ram}GB<{RAM_FREE_GB_WARN}")
        if disk_free < DISK_FREE_GB_WARN:
            warn.append(f"disk_free={disk_free}GB<{DISK_FREE_GB_WARN}")
        if gen:
            if gen["pid"] != first_private_pid:          # 新进程：重置，等预热
                first_private, first_private_pid, first_seen_at = None, gen["pid"], now
            elif first_private is None:
                if now - (first_seen_at or now) >= PRIVATE_BASELINE_WARMUP_S:
                    first_private = gen["private_gb"]    # 预热窗口过了才定基线
            elif gen["private_gb"] - first_private > PRIVATE_GROWTH_GB_WARN:
                warn.append(f"private_grew={gen['private_gb'] - first_private:+.1f}GB")
        stalled_min = (now - last_move) / 60
        if stalled_min > STALL_MINUTES_WARN:
            warn.append(f"no_new_rows_for={stalled_min:.0f}min")
        step = _chain_step()
        if step and step.startswith(("h1:STOP", "mcq:STOP", "perf:failed")):
            warn.append(f"chain_step={step}")
        if chain is None:
            warn.append("chain_process_gone")
        if gen is None and step and str(step).startswith("run:"):
            warn.append("generation_process_gone")

        # ── 探针（post-hoc，非 confirmatory）：只观测，不动手 ──────────────
        probe = _probe_state()
        if probe["failed"]:
            warn.append("probe_failed")
        elif not probe["alive"] and not probe["both_arms_done"]:
            # 只有"没跑完又没进程"才算异常退出；跑完了进程本就该没。
            # 还要**连续两次快照**都成立：cmd 从 MCQ 切到 H1 时两个 python
            # 之间有空档，单次采样会误报。
            if any(a["done"] > 0 for a in probe["arms"].values()):
                probe_gone_streak += 1
                if probe_gone_streak >= 2:
                    warn.append("probe_process_gone")
            else:
                probe_gone_streak = 0
        else:
            probe_gone_streak = 0
        if probe["alive"]:
            ages = [a["newest_shard_age_min"] for a in probe["arms"].values()
                    if a["newest_shard_age_min"] is not None]
            if ages and min(ages) > PROBE_STALL_MINUTES_WARN:
                warn.append(f"probe_stalled={min(ages):.0f}min")

        snap = {"t": _now(), "h1_rows": h1, "mcq_rows": mcq,
                "s_per_unit_window": round(rate, 2) if rate else None,
                "eta_hours_this_arm": eta_h, "chain_step": step,
                "gen": gen, "chain_alive": bool(chain), "probe": probe,
                "gpu": gpu, "ram_free_gb": ram, "disk_free_gb": disk_free,
                "warn": warn}
        with HEALTH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        print(json.dumps(snap, ensure_ascii=False), flush=True)

        # ── 出事就弹窗：按种类去重 + 冷却，噪声型的不弹 ──────────────────────
        if not args.no_popup:
            due = [w for w in warn
                   if w.split("=")[0] in POPUP_KINDS
                   and now - last_alert.get(w.split("=")[0], 0) > ALERT_COOLDOWN_S]
            if due:
                for w in due:
                    last_alert[w.split("=")[0]] = now
                _popup("STAGE1 PRODUCTION ALERT",
                       _alert_body("PRODUCTION ALERT -- needs a look", due, snap))

            # 跑完了也弹一次（绿底），只弹一次
            if step in DONE_STEPS and "__done__" not in last_alert:
                last_alert["__done__"] = now
                _popup("STAGE1 PRODUCTION FINISHED",
                       _alert_body(f"PRODUCTION FINISHED -- {step}", [step], snap),
                       color="2F")

        prev_rows, prev_t = total, now
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
