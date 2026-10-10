"""Stage 1 E3 performance-blind repooling —— **本轮只有诊断子模式**。

授权范围（研究者逐条裁定）：代码 / 单测 / 8,664 条 tokenizer-only 校验 /
64 条精确锚点 / 32-128 微基准。**正式全量另行确认。**

因此本脚本**没有** `--run`：production 目录、manifest、production fingerprint、
completion report 与恢复状态机在本文件里**不存在任何构造路径**——不是靠开关关掉，
是压根没写。段号命名空间 `h1_e3_seg*` 同属全量运行，本轮也不得占用。

三个诊断产物启动前必须不存在，存在即拒绝，禁止覆盖；发布走 `os.link` 原子
"只创建不覆盖"，遥测用 `"x"` 独占创建。

子模式
    --verify-prompts   不加载 model，只用 tokenizer 校验 8,664 条 prompt SHA 与位置
    --anchor           单次模型加载：64 条 union-pooling 与冻结 H 的**精确一致**断言
    --benchmark        同一次加载内接着跑 32 / 128 条 teacher-forced 前向计时
                       **必须与 --anchor 同次给出**：未经锚点证明同一性的环境
                       不得产出计时数字

不解盲、不构造 block-specific label、不读 label、不算任何性能量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool          # noqa: E402
from recover_attention.evaluation import v3_production as prod           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig      # noqa: E402

PROD = ROOT / "outputs/stage1/production"
BENCH = ROOT / "outputs/stage1/benchmark"

ANCHOR_OUT = BENCH / "e3_anchor_diagnostic.json"
BENCH_OUT = BENCH / "e3_benchmark_report.json"
TELEMETRY_OUT = BENCH / "e3_benchmark_gpu_telemetry.csv"

#: 本轮必须保持不存在的正式产物（固定路径部分）。
FORBIDDEN_NAMES = (
    "h1_e3_block_hidden",
    "h1_e3_repool_manifest.jsonl",
    "h1_e3_repool_fingerprint.json",
    "h1_e3_repool_report.json",
)
#: 段号命名空间同样属于全量运行，本轮**不得**占用（诊断产物一律带 benchmark/anchor 字样）。
FORBIDDEN_GLOB = "h1_e3_seg*"

BENCH_SIZES = (32, 128)


class ScopeViolation(RuntimeError):
    """越出本轮授权范围。"""


# ------------------------------------------------------------------ 硬门
def assert_outputs_absent(paths) -> None:
    """启动硬门：诊断目标存在即拒绝，**禁止覆盖**。"""
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise ScopeViolation(
            "输出已存在，拒绝覆盖（本轮不允许重写任何诊断产物）：\n  "
            + "\n  ".join(existing))


def forbidden_paths() -> list[Path]:
    """按**当前** PROD 现算——固定四项 + 段号命名空间 glob。

    写成函数而不是模块级常量：`PROD` 可被测试重定向，常量会把旧目录焊死。
    """
    found = [PROD / name for name in FORBIDDEN_NAMES]
    found.extend(sorted(PROD.glob(FORBIDDEN_GLOB)) if PROD.exists() else [])
    return found


def assert_scope_clean() -> None:
    """正式产物必须保持不存在——运行前后各查一次。

    含段号命名空间：`h1_e3_segNNN_*` 属于全量运行，本轮任何文件都不得落在那里。
    """
    leaked = [str(p) for p in forbidden_paths() if p.exists()]
    if leaked:
        raise ScopeViolation(
            "本轮不得创建正式 E3 产物或占用段号命名空间，但检测到：\n  "
            + "\n  ".join(leaked))


# ------------------------------------------------------------------ 源侧
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(directory: Path) -> str:
    """与既有 freeze manifest 同一配方：sorted(name) -> name + b"\\0" + sha256(bytes)。"""
    h = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        blob = (directory / name).read_bytes()
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(blob).digest())
    return h.hexdigest()


def source_shas() -> dict:
    return {
        "h1_completion_records.jsonl": file_sha256(PROD / "h1_completion_records.jsonl"),
        "h1_run_fingerprint.json": file_sha256(PROD / "h1_run_fingerprint.json"),
        "h1_unit_hidden_tree_digest": tree_digest(PROD / "h1_unit_hidden"),
    }


def load_analyzable() -> list[dict]:
    raw = (PROD / "h1_completion_records.jsonl").read_bytes().split(b"\n")
    rows = [json.loads(x) for x in raw if x.strip()]
    return [r for r in rows if prod.requires_hidden("h1", r)]


def library_versions() -> dict:
    out = {}
    for name in repool.REFERENCE_LIBRARIES:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:                       # noqa: BLE001
            out[name] = f"<unavailable: {type(exc).__name__}>"
    return out


def directory_fsync_supported() -> bool:
    """Windows 通常不能对目录 open/fsync——只登记，不作为任何路径的必经条件。"""
    try:
        fd = os.open(str(BENCH), os.O_RDONLY)
    except (OSError, PermissionError):
        return False
    try:
        os.fsync(fd)
        return True
    except (OSError, PermissionError):
        return False
    finally:
        os.close(fd)


def variable_environment(ctx: dict) -> dict:
    """记录但**允许跨机不同**——不作为拒绝条件。"""
    torch = ctx["torch"]
    info = {
        "model_path": str(ctx.get("model_path")),
        "tokenizer_name_or_path": str(getattr(ctx["tokenizer"], "name_or_path", "")),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "directory_fsync_supported": directory_fsync_supported(),
    }
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": int(props.total_memory),
                "cuda_runtime": str(torch.version.cuda),
                "driver_version": _nvidia_smi_query("driver_version"),
            })
    except Exception as exc:                           # noqa: BLE001
        info["gpu_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _nvidia_smi_query(field: str) -> str:
    try:
        # Windows 控制台默认非 UTF-8：不加 errors="replace" 时 subprocess 的读线程会
        # 抛 UnicodeDecodeError（已实测）。诊断信息不值得让整轮失败。
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, check=False)
        return out.stdout.strip().splitlines()[0].strip()
    except Exception as exc:                           # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}>"


# ------------------------------------------------------------------ 遥测
def start_telemetry(path: Path):
    """必须在**模型加载前**启动；只按 PID 停止。

    用 `"x"` 独占创建：`"w"` 会**静默截断**已存在的遥测 CSV，绕开"存在即拒绝、
    禁止覆盖"的硬门。调用方还必须把本路径无条件放进启动期 `assert_outputs_absent`。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("x", encoding="utf-8", newline="")
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,"
         "clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
         "--format=csv", "-l", "15"],
        stdout=fh, stderr=subprocess.DEVNULL)
    return proc, fh


def stop_telemetry(handle) -> None:
    proc, fh = handle
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:                                  # noqa: BLE001
        proc.kill()
    finally:
        fh.close()


# ------------------------------------------------------------------ 写盘
def publish_json_exclusive(path: Path, payload: dict) -> None:
    """**只创建、绝不覆盖**地发布一个 JSON。

    `os.replace` 无条件覆盖，启动期的 `exists()` 只是 TOCTOU 门：第二次检查之后、
    发布之前目标仍可能出现，然后被悄悄抹掉。这里改用同目录 `os.link` 发布——
    目标已存在时**原子失败**，没有窗口。

    `os.link` 不可用或失败一律 fail-closed，**不退回 `os.replace`**。
    崩溃后可能留下 `.tmp`：下次运行按下面的规则拒绝并**保留现场**。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise ScopeViolation(f"发布目标已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ScopeViolation(
            f"存在陈旧临时文件，拒绝覆盖并保留现场（可能是上次崩溃的残留）：{tmp}")

    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())

    try:
        os.link(str(tmp), str(path))
    except FileExistsError as exc:
        raise ScopeViolation(
            f"发布目标在写入期间出现，原子发布失败，拒绝覆盖：{path}") from exc
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(
            f"os.link 不可用或失败，fail-closed（**不**退回 os.replace）："
            f"{type(exc).__name__}: {exc}；临时文件保留在 {tmp}") from exc
    os.unlink(tmp)


# --------------------------------------------------------- tokenizer-only
def run_verify_prompts(tokenizer, rows: list[dict]) -> dict:
    """8,664 条：prompt SHA / completion token 数 / 位置不越界。首条失败即全量停。"""
    t0 = time.time()
    checked = n_tok_total = f5_equal = 0
    for row in rows:
        out = repool.verify_prompt_and_positions(row, tokenizer)
        checked += 1
        n_tok_total += out["n_completion_tokens"]
        f5_equal += int(bool(out["f5_count_equals_sequence_length"]))
    return {
        "n_checked": checked,
        "elapsed_seconds": round(time.time() - t0, 2),
        "all_prompt_hashes_match": True,
        "all_positions_in_range": True,
        "mean_completion_tokens": round(n_tok_total / max(1, checked), 1),
        "n_rows_where_f5_count_equals_sequence_length": f5_equal,
        "f5_count_note": ("f5_completion_token_count = len(finite completion logprobs)，"
                          "**不是**序列长度；只登记，不作为断言"),
    }


# ------------------------------------------------------------------ 锚点
def run_anchor(backend, rows_by_uid: dict, anchor_ids: list[str], cfg) -> dict:
    """64 条 union-pooling 与 h1_unit_hidden 的**精确一致**断言。无容差。"""
    hidden_dir = PROD / "h1_unit_hidden"
    results, failures = [], []
    t0 = time.time()
    for uid in anchor_ids:
        row = rows_by_uid[uid]
        prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
        tf = backend._teacher_forced(prompt, row["completion"])
        positions = repool.union_positions(row)
        pooled = repool.pool_union(tf["hidden"], tf["prompt_len"], positions)
        frozen = np.load(hidden_dir / f"{uid}.npy")
        diag = repool.anchor_exact_match(pooled, frozen)
        assignment = repool.block_assignment(row, cfg)
        repool.check_block_partition(row, cfg)
        entry = {"unit_id": uid, "n_blocks": len(assignment),
                 "n_union_positions": len(positions), **diag}
        results.append(entry)
        if not diag["exact"]:
            failures.append(entry)
            break                                      # fail-closed：第一条失败即停
    return {
        "n_anchor_units": len(anchor_ids),
        "n_evaluated": len(results),
        "n_exact": sum(1 for r in results if r["exact"]),
        "all_exact": not failures and len(results) == len(anchor_ids),
        "tolerance": "none (exact equality required)",
        "elapsed_seconds": round(time.time() - t0, 2),
        "failures": failures,
        "per_unit": results,
    }


# ---------------------------------------------------------------- 微基准
def run_benchmark(backend, rows_by_uid: dict, cfg, all_rows: list[dict]) -> dict:
    """32 / 128 条 teacher-forced 前向计时。不写任何 pooled 向量。"""
    torch = backend.ctx["torch"]
    ordered = sorted(rows_by_uid)
    out = {}
    for size in BENCH_SIZES:
        step = max(1, len(ordered) // size)
        picked = ordered[::step][:size]
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        n_tokens = 0
        for uid in picked:
            row = rows_by_uid[uid]
            prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
            tf = backend._teacher_forced(prompt, row["completion"])
            n_tokens += tf["prompt_len"] + len(tf["completion_ids"])
            # 立即 pooling 后丢弃：证明"一次前向服务所有块，不留完整 token hidden"
            repool.pool_blocks(tf["hidden"], tf["prompt_len"],
                               repool.block_assignment(row, cfg))
            del tf
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak = (int(torch.cuda.max_memory_allocated())
                if hasattr(torch, "cuda") and torch.cuda.is_available() else None)
        out[f"n_{size}"] = {
            "n_traces": len(picked),
            "elapsed_seconds": round(elapsed, 2),
            "seconds_per_trace": round(elapsed / max(1, len(picked)), 4),
            "mean_sequence_tokens": round(n_tokens / max(1, len(picked)), 1),
            "peak_cuda_allocated_bytes": peak,
        }
    slowest = max(out.values(), key=lambda d: d["seconds_per_trace"])
    total = len(all_rows) * slowest["seconds_per_trace"]
    out["extrapolation"] = {
        "n_traces_full": len(all_rows),
        "seconds_per_trace_used": slowest["seconds_per_trace"],
        "estimated_total_seconds": round(total, 1),
        "estimated_total_hours": round(total / 3600.0, 2),
        "migration_advice": ("建议提请迁移到租用 GPU（外推 > 2 h）" if total > 7200
                             else "本机可完成（外推 <= 2 h）"),
        "note": "只提请，不自动迁移、不自动租用；远程环境须硬指纹一致并重新通过 64/64 锚点",
    }
    return out


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-prompts", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    args = ap.parse_args(argv)

    if not (args.verify_prompts or args.anchor or args.benchmark):
        ap.error("至少选择一个子模式；本脚本没有 --run（全量须另行确认）")

    # --benchmark 必须与 --anchor 同次运行：否则会在**从未证明过权重/pooling 同一性**
    # 的环境里产出计时数字。刻意不支持"读取磁盘旧 anchor 后单跑 benchmark"——那要新增
    # schema / SHA / anchor-set / source-SHA / 环境状态一整套校验，当前没有这个需求。
    # 用 ap.error 而不是后置检查：它在**加载模型之前**就以 exit 2 退出。
    if args.benchmark and not args.anchor:
        ap.error("--benchmark 必须与 --anchor 同次给出："
                 "未经 64/64 精确锚点证明同一性的环境不得产出计时数字")

    assert_scope_clean()
    will_load_model = args.anchor or args.benchmark
    targets = []
    if args.anchor:
        targets.append(ANCHOR_OUT)
    if args.benchmark:
        targets.append(BENCH_OUT)
    if will_load_model:
        # 遥测**只要会启动就必须进硬门**——此前它只在 --benchmark 时入列，
        # 单跑 --anchor 会绕过检查并以 "w" 截断已有 CSV。
        targets.append(TELEMETRY_OUT)
    assert_outputs_absent(targets)

    cfg = V3AnalysisConfig()
    rows = load_analyzable()
    rows_by_uid = {repool.derive_unit_id(r): r for r in rows}
    anchor_ids = repool.select_anchor_units(rows, cfg)
    shas = source_shas()
    print(f"[src  ] analyzable {len(rows):,} ｜ (trace,block) "
          f"{sum(len(repool.expected_blocks(r, cfg)) for r in rows):,}")
    print(f"[src  ] records {shas['h1_completion_records.jsonl'][:16]}… ｜ "
          f"hidden tree {shas['h1_unit_hidden_tree_digest'][:16]}…")
    print(f"[anchr] {len(anchor_ids)} units ｜ set sha "
          f"{repool.anchor_set_sha256(anchor_ids)[:16]}…")

    telemetry = None
    if args.benchmark or args.anchor:
        # 遥测必须在模型加载**之前**启动
        telemetry = start_telemetry(TELEMETRY_OUT)
        print(f"[tele ] pid={telemetry[0].pid} -> {TELEMETRY_OUT.name}")

    payload: dict = {
        "schema": repool.REPOOL_SCHEMA,
        "mode": "diagnostic_only",
        "authorized_scope": ["code", "tests", "tokenizer_only_verification",
                             "64_unit_anchor", "32_128_benchmark"],
        "not_in_scope": ["production repool run", "manifest", "production fingerprint",
                         "completion report", "block-specific label", "unblinding"],
        "source_shas": shas,
        "anchor_unit_ids": anchor_ids,
        "anchor_set_sha256": repool.anchor_set_sha256(anchor_ids),
        "n_analyzable_traces": len(rows),
        "n_trace_block_pairs": sum(len(repool.expected_blocks(r, cfg)) for r in rows),
        "coverage_caliber": "cache coverage (NOT the final E3 analysis population)",
    }

    backend = None
    try:
        if args.verify_prompts or args.anchor or args.benchmark:
            from recover_attention.generation import backends as gb
            if args.verify_prompts and not (args.anchor or args.benchmark):
                comp = gb.import_transformers_components()
                tokenizer = comp["AutoTokenizer"].from_pretrained(
                    args.model_path, local_files_only=True, trust_remote_code=False)
                payload["prompt_verification"] = run_verify_prompts(tokenizer, rows)
                print(f"[vfy  ] {payload['prompt_verification']}")
                print(json.dumps(payload["prompt_verification"], ensure_ascii=False))
                assert_scope_clean()
                return 0

            backend = gb._RealBackendBase(args.model_path)
            run_fp = backend.run_fingerprint("h1")
            payload["hidden_index_check"] = backend.verify_hidden_index()
            payload["hard_consistency"] = repool.hard_consistency_payload(
                run_fp, cfg, shas, library_versions())
            payload["variable_environment"] = variable_environment(backend.ctx)
            payload["reforward_reference_environment"] = {
                "library_versions": library_versions(),
                "status": "provisional_until_anchor_passes",
                "note": ("h1_run_fingerprint.json 未记录生成期 torch/transformers/"
                         "bitsandbytes/accelerate 版本，故本项是**前瞻性** reforward "
                         "兼容性基线，不追溯声称为原始生成期软件环境"),
            }

            frozen_fp = json.loads((PROD / "h1_run_fingerprint.json").read_text("utf-8"))
            frozen_hard = repool.hard_consistency_payload(
                frozen_fp, cfg, shas, library_versions())
            drift = repool.compare_hard_consistency(
                {k: v for k, v in frozen_hard.items() if k != "library_versions"},
                {k: v for k, v in payload["hard_consistency"].items()
                 if k != "library_versions"})
            payload["fingerprint_drift_vs_frozen"] = drift
            if drift:
                raise ScopeViolation("硬一致指纹与冻结 run fingerprint 不符：\n  "
                                     + "\n  ".join(drift))

            payload["prompt_verification"] = run_verify_prompts(
                backend.ctx["tokenizer"], rows)
            print(f"[vfy  ] {payload['prompt_verification']}")

            if args.anchor:
                payload["anchor"] = run_anchor(backend, rows_by_uid, anchor_ids, cfg)
                print(f"[anchr] exact {payload['anchor']['n_exact']}/"
                      f"{payload['anchor']['n_anchor_units']} "
                      f"({payload['anchor']['elapsed_seconds']}s)")
                if payload["anchor"]["all_exact"]:
                    payload["reforward_reference_environment"]["status"] = "frozen"
                publish_json_exclusive(ANCHOR_OUT, payload)
                if not payload["anchor"]["all_exact"]:
                    print("[STOP ] 锚点未全部精确一致 —— 停止，不引入容差，另出 "
                          "Recovery Preflight", file=sys.stderr)
                    return 3

            if args.benchmark:
                payload["benchmark"] = run_benchmark(backend, rows_by_uid, cfg, rows)
                print(f"[bench] {json.dumps(payload['benchmark']['extrapolation'], ensure_ascii=False)}")
                publish_json_exclusive(BENCH_OUT, payload)
    finally:
        if telemetry is not None:
            stop_telemetry(telemetry)
        if backend is not None:
            del backend
        assert_scope_clean()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
