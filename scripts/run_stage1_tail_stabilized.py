"""Stage 1 tail-stabilized Route-B 正式 runner —— performance-blind，不解盲。

子模式
    --benchmark   两臂各做确定性分层的 32/128，走**正式 runner 的同一代码路径**，
                  产出实测速率 / 显存 / 外推时长 / 段长计划。**不写正式 cache。**
    --run --arm {h1,mcq} --max-new-traces N
                  正式分段执行。append-only manifest + 幂等 resume + orphan 自洽恢复；
                  逐 trace 五道硬门；失败在 trace 边界停并记 `failed` **工程事件**。

正式路径规模（写死，不由脚本推断）：

    H1 analyzable   8,664 traces      MCQ analyzable  1,750 traces
    合计           10,414 forwards    H1 E3           9,916 (trace, block) vectors

一条 trace 一次 forward：block 19 输出 cast 前 fp16 做冻结锚点，cast 后 bf16 走
blocks 20-27 + final norm → 外部 float32 lm_head/logsumexp。

任一逐 trace 硬门失败：在 trace 边界立即停，记 `failed` **工程事件**（不标 `excluded`），
保留此前 done，不删除、不回滚、不自动重试，不生成 completion report。

`failed` 是**终态且阻塞**：下一次 `--run` 检测到未被后续 done 覆盖的 failed 即拒绝启动，
必须由研究者另出 Recovery Preflight 裁定。脚本自己**不重排、不重试、不改写**。

产物只能称 `stable_logit_shadow_sensitivity` / `protocol_deviation_numerical_sensitivity`。
原 14 项 H1 B4 仍是 frozen implementation primary，本路径**不得**冒充 primary。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp                    # noqa: E402
from recover_attention import mcq_conditional_increment as mci             # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool            # noqa: E402
from recover_attention.evaluation import v3_production as prod             # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow_f5         # noqa: E402
from recover_attention.evaluation import v3_tail_stabilized as tail        # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402
from recover_attention.generation import stable_logits as sl               # noqa: E402

#: artifact root —— 默认 Qwen 的 outputs/stage1；跨模型复现由 CLI 显式重定向。
#: `set_artifact_root()` 只允许在 main() 启动时、任何 IO 之前调用一次。
DEFAULT_ARTIFACT_ROOT = ROOT / "outputs/stage1"
ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT

PROD = ARTIFACT_ROOT / "production"
REPOOL = ARTIFACT_ROOT / "repool"
BENCH = ARTIFACT_ROOT / "benchmark"
TAIL_DIR = ARTIFACT_ROOT / "tail_stabilized"

#: 本次运行的模型几何。默认 Qwen 冻结几何，由 CLI 覆盖。
GEOMETRY = repool.QWEN_GEOMETRY

_ROOT_REBOUND = False


def set_artifact_root(root, *, geometry=None) -> dict:
    """把四个产物目录整体重定向到独立 artifact root。

    只在 CLI 启动时调用一次、且早于任何文件访问。**不是**给测试用的
    monkey-patch 钩子，也不接受二次调用 —— 跑到一半换根目录会让
    append-only 前缀校验与 source freeze 失去意义。
    """
    global ARTIFACT_ROOT, PROD, REPOOL, BENCH, TAIL_DIR, GEOMETRY
    global BENCH_OUT, BENCH_TELEMETRY, _ROOT_REBOUND
    if _ROOT_REBOUND:
        raise ScopeViolation("set_artifact_root 只能调用一次")
    ARTIFACT_ROOT = Path(root)
    PROD = ARTIFACT_ROOT / "production"
    REPOOL = ARTIFACT_ROOT / "repool"
    BENCH = ARTIFACT_ROOT / "benchmark"
    TAIL_DIR = ARTIFACT_ROOT / "tail_stabilized"
    BENCH_OUT = BENCH / "tail_stabilized_benchmark_report.json"
    BENCH_TELEMETRY = BENCH / "tail_stabilized_benchmark_gpu_telemetry.csv"
    if geometry is not None:
        GEOMETRY = geometry
    _ROOT_REBOUND = True
    return {"artifact_root": str(ARTIFACT_ROOT), "geometry": GEOMETRY.as_dict()}

BENCH_OUT = BENCH / "tail_stabilized_benchmark_report.json"
BENCH_TELEMETRY = BENCH / "tail_stabilized_benchmark_gpu_telemetry.csv"

#: **Qwen 冻结运行**的正式规模。保留为默认 artifact root 下的回归钉子 ——
#: 它们是 Qwen 的结果，**不是**跨模型运行的验收目标。另一个模型的
#: analyzable 数由该模型自己的生成内容决定，没有先验值。
N_H1_EXPECTED = 8664
N_MCQ_EXPECTED = 1750
N_E3_PAIRS_EXPECTED = 9916


def observed_scale(cfg, h1_rows, mcq_rows) -> dict:
    """从**独立的 finalized 输入**现场重建期望规模。

    注意口径：分母来自 `{arm}_completion_records.jsonl`（生成侧的终态产物），
    **不是** tail 自己的输出行 —— 拿一份输出验证它自己不构成完整性证据。
    """
    return {"h1": len(h1_rows), "mcq": len(mcq_rows),
            "e3_pairs": sum(len(repool.expected_blocks(r, cfg)) for r in h1_rows)}


def pinned_scale() -> dict | None:
    """默认 artifact root（Qwen）下返回冻结钉子；独立 root 下返回 None。

    独立 root 不设先验计数，但**输入变化仍会被拦下** —— `source_freeze()` 的
    12 项里含两臂 finalized records 的 sha256，输入一改就与首段基线不符。
    """
    return ({"h1": N_H1_EXPECTED, "mcq": N_MCQ_EXPECTED,
             "e3_pairs": N_E3_PAIRS_EXPECTED}
            if ARTIFACT_ROOT == DEFAULT_ARTIFACT_ROOT else None)


def assert_scale(scale: dict) -> dict:
    pin = pinned_scale()
    if pin is not None and scale != pin:
        raise ScopeViolation(
            f"analyzable 规模与冻结钉子不符：实测 {scale} / 期望 {pin}。"
            f"（默认 artifact root = Qwen 冻结运行；跨模型请用 --artifact-root）")
    return {"observed": scale, "pinned": pin,
            "pin_source": ("frozen_qwen_constants" if pin is not None
                           else "none:independent_artifact_root")}

#: 本轮**必须保持不存在**的正式产物。
FORBIDDEN_FORMAL = (
    TAIL_DIR / "tail_manifest.jsonl",
    TAIL_DIR / "tail_fingerprint.json",
    TAIL_DIR / "tail_completion_report.json",
    TAIL_DIR / "h1_tail_f5.jsonl",
    TAIL_DIR / "mcq_tail_f5.jsonl",
    TAIL_DIR / "h1_e3_block_hidden",
)

BENCH_SIZES = (32, 128)
#: 段长规则：目标每段 <= 20 分钟；每段最多 4,000 traces。
SEG_MAX_TRACES = 4000
SEG_TARGET_SECONDS = 1200

TAIL_RUN_SCHEMA = "stage1_tail_stabilized_run_v1"
TAIL_REPORT_SCHEMA = "stage1_tail_stabilized_completion_report_v1"
TAIL_VARIANT = "tail_stabilized_single_forward"

#: 单元处理顺序 —— 与源文件行序**无关**，否则 resume 的段边界会随文件重排而漂移。
UNIT_ORDER_RULE = "sorted(unit_id) 升序；按臂独立；与 completion records 行序无关"

#: 后端口径：**原冻结 8-bit fp16 backend 不变**，只在 block 19 之后由 hook 换成 bf16。
TAIL_BACKEND = {
    "loader": "generation.backends._RealBackendBase（原冻结口径，未改动）",
    "load_in_8bit": True, "prefix_activation_dtype": "float16",
    "cast_block_index": tail.CAST_BLOCK, "tail_activation_dtype": "bfloat16",
    "post_cast_residual_dtype_observed": "torch.float32",
    "projection": "外部 float32 lm_head / logsumexp（stable_logits）",
    "caveat": ("bitsandbytes MatMul8bitLt 会把激活内部降回 float16 做量化 —— "
               "**不得**声称尾部全程 bf16；block 19 之前逐位等于冻结口径，"
               "由逐 trace 精确锚点证明"),
}

#: 两臂 finalized 六项 + 两份 run fingerprint + H1 frozen-hidden 树 + 既有 E3 cache 三项。
FREEZE_PROD_FILES = (
    "h1_completion_records.jsonl", "h1_hidden.npz", "h1_finalization_report.json",
    "mcq_completion_records.jsonl", "mcq_hidden.npz", "mcq_finalization_report.json",
    "h1_run_fingerprint.json", "mcq_run_fingerprint.json",
)
FREEZE_REPOOL_FILES = ("e3_repool_manifest.jsonl", "e3_repool_report.json")


class ScopeViolation(RuntimeError):
    """越出本轮授权范围。"""


# ------------------------------------------------------------------ 硬门
def assert_formal_absent() -> None:
    leaked = [str(p) for p in FORBIDDEN_FORMAL if p.exists()]
    seg = sorted(TAIL_DIR.glob("tail_seg*_gpu_telemetry.csv")) if TAIL_DIR.exists() else []
    leaked += [str(p) for p in seg]
    if leaked:
        raise ScopeViolation(
            "本轮不得创建正式 tail 产物，但检测到：\n  " + "\n  ".join(leaked))


def assert_outputs_absent(paths) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise ScopeViolation("输出已存在，拒绝覆盖：\n  " + "\n  ".join(existing))


def publish_json_exclusive(path: Path, payload: dict) -> None:
    """只创建、绝不覆盖：O_EXCL 临时文件 → os.link 原子发布；失败 fail-closed。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise ScopeViolation(f"发布目标已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ScopeViolation(f"存在陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(f"os.link 失败，fail-closed：{exc}；临时保留在 {tmp}") from exc
    os.unlink(tmp)


class Telemetry(NamedTuple):
    """采样器句柄。`proc` 在索引 0，旧的 `tel[0].pid` 用法仍然成立。"""

    proc: object
    fh: object
    path: Path


#: 首个采样点的等待上限；nvidia-smi 每 5 s 一点，实测约 1 s 内落下表头 + 首点。
TELEMETRY_FIRST_SAMPLE_TIMEOUT = 60.0
TELEMETRY_POLL_SECONDS = 0.25
#: 运行途中每多少条 trace 复查一次采样器存活。
TELEMETRY_CHECK_EVERY = 100


def start_telemetry(path: Path) -> Telemetry:
    """必须在**模型加载前**启动；`"x"` 独占创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("x", encoding="utf-8", newline="")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,"
         "clocks.sm,temperature.gpu,clocks_event_reasons.active", "--format=csv",
         "-l", "5"], stdout=fh, stderr=subprocess.DEVNULL)
    return Telemetry(proc, fh, Path(path))


def telemetry_sample_count(path: Path) -> int:
    """CSV 里的**数据行**数（不含表头）。读不到就当 0，不抛错。"""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return 0
    return max(0, len([ln for ln in raw.split(b"\n") if ln.strip()]) - 1)


def wait_for_telemetry(tel: Telemetry, *,
                       timeout: float = TELEMETRY_FIRST_SAMPLE_TIMEOUT,
                       sleep=time.sleep, clock=time.time) -> int:
    """**在加载模型之前**确认采样器活着、表头与首个采样点已落盘。

    812 MiB 的峰值余量下，一段十几分钟的无遥测长跑等于没有现场证据。
    """
    deadline = clock() + timeout
    while True:
        n = telemetry_sample_count(tel.path)
        if n >= 1:
            return n
        rc = tel.proc.poll()
        if rc is not None:
            raise TailRecoveryStop(
                f"遥测 sampler 在落下首个采样点前退出（returncode={rc}）：{tel.path}")
        if clock() >= deadline:
            raise TailRecoveryStop(
                f"遥测 sampler {timeout:.0f}s 内未落下首个采样点：{tel.path} —— "
                "拒绝在无遥测的情况下加载模型")
        sleep(TELEMETRY_POLL_SECONDS)


def check_telemetry_alive(tel: Telemetry) -> None:
    """运行途中的存活复查；采样器死掉即停止本段。"""
    rc = tel.proc.poll()
    if rc is not None:
        raise TailRecoveryStop(
            f"遥测 sampler 在本段运行途中退出（returncode={rc}）：{tel.path} —— "
            "停止本段，保留已完成前缀")


def stop_telemetry(tel) -> None:
    """**只按已记录的句柄**停止；绝不按进程名批量杀。"""
    try:
        tel.proc.terminate()
        tel.proc.wait(timeout=20)
    except Exception:                                   # noqa: BLE001
        tel.proc.kill()
    finally:
        tel.fh.close()


# ------------------------------------------------------------------ 载入
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_analyzable(arm: str) -> list[dict]:
    raw = (PROD / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
    rows = [json.loads(x) for x in raw if x.strip()]
    return [r for r in rows if prod.requires_hidden(arm, r)]


def library_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate"):
        try:
            out[name] = str(getattr(__import__(name), "__version__", "unknown"))
        except Exception as exc:                        # noqa: BLE001
            out[name] = f"<unavailable: {type(exc).__name__}>"
    return out


# -------------------------------------------------------- 确定性分层取样
def h1_strata_sample(rows: list[dict], cfg, n: int) -> dict:
    """H1 **按序列长度与 E3 block 数分层**，确定性取样。"""
    keyed = []
    for r in rows:
        uid = repool.derive_unit_id(r)
        n_tok = len(r.get("completion") or "")          # 代理长度，仅用于分层排序
        n_blocks = len(repool.expected_blocks(r, cfg))
        keyed.append((n_blocks, n_tok, uid))
    keyed.sort()
    by_blocks: dict[int, list] = {}
    for nb, nt, uid in keyed:
        by_blocks.setdefault(nb, []).append((nt, uid))
    # 按各层占比配额，层内等距 stride（含首尾）
    total = len(keyed)
    chosen: list[str] = []
    for nb in sorted(by_blocks):
        pool = by_blocks[nb]
        quota = max(1, round(n * len(pool) / total))
        step = max(1, len(pool) // quota)
        chosen.extend(uid for _nt, uid in pool[::step][:quota])
    chosen = sorted(set(chosen))
    if len(chosen) > n:
        step = max(1, len(chosen) // n)
        chosen = sorted(chosen[::step][:n])
    return {"unit_ids": chosen, "n": len(chosen),
            "strata": {str(k): len(v) for k, v in sorted(by_blocks.items())},
            "sort_key": "(n_e3_blocks, completion_char_len, unit_id)",
            "selection_sha256": repool.anchor_set_sha256(chosen)}


def mcq_strata_sample(rows: list[dict], n: int) -> dict:
    """MCQ **按序列长度分层**，确定性取样（含首尾的等距 rank）。"""
    keyed = sorted((len(r.get("raw_completion") or ""), r["example_id"]) for r in rows)
    total = len(keyed)
    ranks = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    seen, chosen = set(), []
    for rk in ranks:
        while rk < total and rk in seen:
            rk += 1
        if rk < total:
            seen.add(rk)
            chosen.append(keyed[rk][1])
    chosen = sorted(set(chosen))
    return {"example_ids": chosen, "n": len(chosen),
            "sort_key": "(raw_completion_char_len, example_id)",
            "selection_sha256": repool.anchor_set_sha256(chosen)}


# ------------------------------------------------ 正式 runner 的单 trace 路径
def tail_record(row: dict, arm: str, stable: dict, *, provenance: dict) -> dict:
    """一条正式 F5 记录。命名口径完全沿用 shadow —— **不得**冒充 primary。

    只在 shadow 记录上多加两个说明本路径身份的键（`variant` / `tail_schema`），
    不改 `interpretation` / `deviation_class` / `not_primary` 这三条口径字段。
    """
    rec = shadow_f5.shadow_record(row, arm, stable,
                                  provenance={**provenance, "path": TAIL_VARIANT,
                                              "backend": dict(TAIL_BACKEND)})
    rec["variant"] = TAIL_VARIANT
    rec["tail_schema"] = tail.TAIL_SCHEMA
    return rec


def tail_h1_unit(model, tokenizer, torch, head, row, cfg,
                 frozen_union: np.ndarray, frozen_e3: dict) -> dict:
    """H1：一次 forward → 冻结锚点 + stable B4。**正式 runner 的同一代码路径。**"""
    uid = repool.derive_unit_id(row)
    prompt = repool.render_prompt(row, tokenizer)
    res = tail.run_tail_unit(model=model, tokenizer=tokenizer, torch=torch, head=head,
                             geometry=GEOMETRY,
                             unit_id=uid, prompt=prompt, completion=row["completion"],
                             expected_prompt_hash=row["prompt_hash"])
    anch = tail.assert_h1_anchors(unit_id=uid, pre_cast_hidden=res["pre_cast_hidden"],
                                  geometry=GEOMETRY,
                                  prompt_len=res["prompt_len"], row=row, cfg=cfg,
                                  frozen_union=frozen_union, frozen_e3=frozen_e3)
    positions = repool.union_positions(row)
    stable = shadow_f5.h1_shadow_row(
        row, n_tok=res["n_completion_tokens"], target_logprob=res["target_logprob"],
        entropy=res["entropy"], rank=res["rank"], id_positions=positions)
    # E3 的 npz meta 口径与既有 `outputs/stage1/repool/` 完全一致 —— 这样
    # `repool.validate_npz_self_consistency` 能原样校验本路径写出的文件。
    meta = {"trace_id": uid, "prompt_hash": row["prompt_hash"],
            "source_fingerprint_id": row.get("run_hash"),
            "token_positions": {str(b): list(p) for b, p in anch["assignment"].items()}}
    record = tail_record(row, "h1", stable, provenance={
        "unit_id": uid, "prompt_hash": row["prompt_hash"],
        "unit_hash": row.get("unit_hash"), "source_run_hash": row.get("run_hash"),
        "n_completion_tokens": res["n_completion_tokens"],
        "n_identifier_positions": len(positions),
        "n_e3_blocks": anch["n_blocks"], "position_digest": anch["position_digest"],
        "post_cast_residual_dtype": res["post_cast_residual_dtype"]})
    return {"unit_id": uid, "n_completion_tokens": res["n_completion_tokens"],
            "n_blocks": anch["n_blocks"], "stable_fields": stable,
            "post_cast_residual_dtype": res["post_cast_residual_dtype"],
            "position_digest": anch["position_digest"],
            "block_ids": sorted(anch["assignment"]),
            "arrays": anch["arrays"],           # ← 交给正式写盘层，**不再丢弃**
            "npz_meta": meta, "record": record}


def tail_mcq_unit(model, tokenizer, torch, head, row,
                  frozen_letter_hidden: np.ndarray, bare, space) -> dict:
    """MCQ：一次 forward → 冻结 answer-letter H 锚点 + stable B4。"""
    eid = row["example_id"]
    res = tail.run_tail_unit(model=model, tokenizer=tokenizer, torch=torch, head=head,
                             geometry=GEOMETRY,
                             unit_id=eid, prompt=row["rendered_prompt"],
                             completion=row["raw_completion"],
                             expected_prompt_hash=row["prompt_hash"])
    j = int(row["letter_token_index_local"])
    if j >= res["n_completion_tokens"]:
        raise tail.TailStop(f"{eid}: letter token index 越界", unit_id=eid,
                            gate="mcq_letter_index",
                            diagnostic={"j": j, "n_tok": res["n_completion_tokens"]})
    if int(res["completion_ids"][j]) != int(row["letter_token_id"]):
        raise tail.TailStop(f"{eid}: letter_token_id 与重编码不符", unit_id=eid,
                            gate="mcq_letter_token_id",
                            diagnostic={"got": int(res["completion_ids"][j]),
                                        "expected": int(row["letter_token_id"])})
    anch = tail.assert_mcq_anchor(unit_id=eid, pre_cast_hidden=res["pre_cast_hidden"],
                                  prompt_len=res["prompt_len"],
                                  letter_token_index_local=j,
                                  frozen_letter_hidden=frozen_letter_hidden)
    ids = bare if row["letter_token_form"] == "bare" else space
    logits_row = res["logits32"][j]
    opts = {lab: float(logits_row[int(tid)]) for lab, tid in ids.items()}
    letter_lp = float(res["log_probs32"][j, int(row["letter_token_id"])])
    stable = shadow_f5.mcq_shadow_row(row, option_logits=opts,
                                      full_logits_row=logits_row,
                                      letter_logprob=letter_lp)
    record = tail_record(row, "mcq", stable, provenance={
        "example_id": eid, "prompt_hash": row["prompt_hash"],
        "letter_token_index_local": j, "letter_token_id": int(row["letter_token_id"]),
        "letter_token_form": row["letter_token_form"],
        "letter_row_index": anch["letter_row_index"],
        "source_run_hash": row.get("run_hash"),
        "n_completion_tokens": res["n_completion_tokens"],
        "post_cast_residual_dtype": res["post_cast_residual_dtype"]})
    return {"unit_id": eid, "n_completion_tokens": res["n_completion_tokens"],
            "stable_fields": stable, "letter_row_index": anch["letter_row_index"],
            "post_cast_residual_dtype": res["post_cast_residual_dtype"],
            "arrays": None, "npz_meta": None, "record": record}


# ------------------------------------------------------------- 微基准
def _gpu_free_mib(torch) -> float | None:
    if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 2 ** 20


def _segment_plan(arm: str, n_total: int, s_per_trace: float) -> dict:
    """段长 = min(4000, floor(1200 / 保守 s/trace))；最后一段取余。"""
    by_time = int(SEG_TARGET_SECONDS // s_per_trace) if s_per_trace > 0 else SEG_MAX_TRACES
    seg_len = max(1, min(SEG_MAX_TRACES, by_time))
    n_seg = -(-n_total // seg_len)
    segs = [seg_len] * (n_seg - 1) + [n_total - seg_len * (n_seg - 1)] if n_seg else []
    return {"arm": arm, "n_total": n_total, "seconds_per_trace_used": s_per_trace,
            "seg_len_by_time_budget": by_time, "seg_len_cap": SEG_MAX_TRACES,
            "seg_len": seg_len, "n_segments": n_seg, "segment_sizes": segs,
            "est_seconds_per_full_segment": round(seg_len * s_per_trace, 1),
            "est_total_seconds": round(n_total * s_per_trace, 1)}


def run_benchmark(model_path: str) -> int:
    assert_formal_absent()
    assert_outputs_absent([BENCH_OUT, BENCH_TELEMETRY])

    cfg = V3AnalysisConfig()
    h1_rows = load_analyzable("h1")
    mcq_rows = load_analyzable("mcq")
    scale = observed_scale(cfg, h1_rows, mcq_rows)
    assert_scale(scale)
    n_pairs = scale["e3_pairs"]
    print(f"[scale] H1 {len(h1_rows):,} ｜ MCQ {len(mcq_rows):,} ｜ E3 pairs {n_pairs:,}")

    h1_by = {repool.derive_unit_id(r): r for r in h1_rows}
    mcq_by = {r["example_id"]: r for r in mcq_rows}
    samples = {
        "h1": {n: h1_strata_sample(h1_rows, cfg, n) for n in BENCH_SIZES},
        "mcq": {n: mcq_strata_sample(mcq_rows, n) for n in BENCH_SIZES},
    }
    for arm in ("h1", "mcq"):
        for n in BENCH_SIZES:
            s = samples[arm][n]
            print(f"[sample] {arm} n={n} -> {s['n']} 条 ｜ sha {s['selection_sha256'][:16]}")

    tel = start_telemetry(BENCH_TELEMETRY)         # 模型加载**之前**
    print(f"[tele ] pid={tel[0].pid} -> {BENCH_TELEMETRY.name}")
    backend = None
    results: dict = {}
    try:
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        torch = backend.ctx["torch"]
        model = backend.ctx["model"]
        tokenizer = backend.ctx["tokenizer"]
        head = sl.Float32Head(model, torch)
        bare = dlp.bare_option_token_ids(tokenizer, list(mci.MCQ_LETTERS))
        space = dlp.option_token_ids(tokenizer, list(mci.MCQ_LETTERS))
        frozen_mcq_hidden = dict(np.load(PROD / "mcq_hidden.npz"))
        print(f"[load ] projection cache_mode={head.cache_mode} ｜ "
              f"free={_gpu_free_mib(torch):.0f} MiB")

        for arm in ("h1", "mcq"):
            for n in BENCH_SIZES:
                ids = (samples[arm][n]["unit_ids"] if arm == "h1"
                       else samples[arm][n]["example_ids"])
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.time()
                per, n_tok_total = [], 0
                for uid in ids:
                    if arm == "h1":
                        row = h1_by[uid]
                        frozen_union = np.load(PROD / "h1_unit_hidden" / f"{uid}.npy")
                        with np.load(REPOOL / "h1_e3_block_hidden" / f"{uid}.npz",
                                     allow_pickle=False) as z:
                            frozen_e3 = {k: np.asarray(z[k]) for k in z.files
                                         if k.startswith("block_")}
                        out = tail_h1_unit(model, tokenizer, torch, head, row, cfg,
                                           frozen_union, frozen_e3)
                    else:
                        row = mcq_by[uid]
                        out = tail_mcq_unit(model, tokenizer, torch, head, row,
                                            frozen_mcq_hidden[uid], bare, space)
                    n_tok_total += out["n_completion_tokens"]
                    per.append(out)
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                peak = (int(torch.cuda.max_memory_allocated())
                        if hasattr(torch, "cuda") and torch.cuda.is_available() else None)
                free_mib = _gpu_free_mib(torch)
                results[f"{arm}_{n}"] = {
                    "arm": arm, "n_requested": n, "n_run": len(per),
                    "elapsed_seconds": round(elapsed, 2),
                    "seconds_per_trace": round(elapsed / max(1, len(per)), 4),
                    "mean_completion_tokens": round(n_tok_total / max(1, len(per)), 1),
                    "peak_cuda_allocated_bytes": peak,
                    "gpu_free_mib_after": round(free_mib, 1) if free_mib else None,
                    "all_anchors_exact": True,       # 任一不符会在 TailStop 抛出
                    "no_token_filtering": True,
                    "post_cast_residual_dtypes": sorted(
                        {p["post_cast_residual_dtype"] for p in per}),
                    "selection_sha256": samples[arm][n]["selection_sha256"],
                }
                r = results[f"{arm}_{n}"]
                print(f"[bench] {arm} n={n:<4d} {r['elapsed_seconds']:>7.2f}s "
                      f"{r['seconds_per_trace']:.4f} s/trace ｜ "
                      f"peak {peak / 2 ** 30:.2f} GiB ｜ free {r['gpu_free_mib_after']:.0f} MiB")
    except tail.TailStop as exc:
        stop_telemetry(tel)
        print(f"[STOP ] 逐 trace 硬门失败 gate={exc.gate} unit={exc.unit_id}",
              file=sys.stderr)
        print(f"[STOP ] {exc}\n{json.dumps(exc.diagnostic, ensure_ascii=False)}",
              file=sys.stderr)
        return 3
    else:
        stop_telemetry(tel)
    finally:
        backend = None
        assert_formal_absent()

    # ---- 硬闸 ------------------------------------------------------------
    s_h1 = results["h1_128"]["seconds_per_trace"]
    s_mcq = results["mcq_128"]["seconds_per_trace"]
    n_h1, n_mcq = scale["h1"], scale["mcq"]
    total_est = n_h1 * s_h1 + n_mcq * s_mcq
    min_free = min(r["gpu_free_mib_after"] for r in results.values()
                   if r["gpu_free_mib_after"] is not None)
    gates = {
        "all_anchors_exact": all(r["all_anchors_exact"] for r in results.values()),
        "no_token_filtering": all(r["no_token_filtering"] for r in results.values()),
        "gpu_headroom_mib_ge_500": bool(min_free >= 500),
        "extrapolated_total_le_2h": bool(total_est <= 7200),
    }
    passed = all(gates.values())

    payload = {
        "schema": "stage1_tail_stabilized_benchmark_v1",
        "mode": "benchmark_only",
        "writes_formal_cache": False,
        # 规模取自**本次** finalized 输入现场重建；pinned 只在默认 root 下存在。
        "scale": {"h1": n_h1, "mcq": n_mcq, "total_forwards": n_h1 + n_mcq,
                  "h1_e3_pairs": scale["e3_pairs"],
                  "source": "rebuilt_from_finalized_completion_records"},
        "scale_pin_check": assert_scale(scale),
        "artifact_root": str(ARTIFACT_ROOT),
        "geometry": GEOMETRY.as_dict(),
        "sampling": {arm: {str(n): samples[arm][n] for n in BENCH_SIZES}
                     for arm in ("h1", "mcq")},
        "results": results,
        "gates": gates,
        "gates_passed": passed,
        "min_gpu_free_mib": min_free,
        "extrapolation": {
            "h1_seconds_per_trace_from_128": s_h1,
            "mcq_seconds_per_trace_from_128": s_mcq,
            "h1_est_seconds": round(n_h1 * s_h1, 1),
            "mcq_est_seconds": round(n_mcq * s_mcq, 1),
            "total_est_seconds": round(total_est, 1),
            "total_est_hours": round(total_est / 3600, 3),
        },
        "segment_plan": {
            "rule": ("按臂分段，不混 H1/MCQ；目标每段 <= 20 分钟；每段最多 4,000 traces；"
                     "段长 = min(4000, floor(1200 / 128 条基准的保守 s/trace))；最后一段取余"),
            "h1": _segment_plan("h1", n_h1, s_h1),
            "mcq": _segment_plan("mcq", n_mcq, s_mcq),
        },
        "old_e3_benchmark_note": (
            "旧 E3 benchmark 的约 0.17 s/trace、0.41 h/H1 **只能作旁证** —— "
            "它不是正式 tail 路径（无 hook、无 cast、无外部 float32 投影），"
            "**不得**直接用于分段。段长以本报告的实测为准。"),
        "environment": {"libraries": library_versions(), "platform": platform.platform(),
                        "python": sys.version.split()[0]},
        "source_shas": {
            "h1_completion_records.jsonl": file_sha256(PROD / "h1_completion_records.jsonl"),
            "mcq_completion_records.jsonl": file_sha256(PROD / "mcq_completion_records.jsonl"),
        },
    }
    publish_json_exclusive(BENCH_OUT, payload)
    print(f"[gates] {json.dumps(gates, ensure_ascii=False)} -> passed={passed}")
    print(f"[plan ] H1 {payload['segment_plan']['h1']['n_segments']} 段 × "
          f"{payload['segment_plan']['h1']['seg_len']} ｜ MCQ "
          f"{payload['segment_plan']['mcq']['n_segments']} 段 × "
          f"{payload['segment_plan']['mcq']['seg_len']}")
    print(f"[out  ] {BENCH_OUT}")
    return 0 if passed else 3


# ══════════════════════════════════════════════════════════════ 正式执行路径
class TailRecoveryStop(RuntimeError):
    """正式状态自洽性失败 —— 停机，等研究者的 Recovery Preflight。

    **绝不**自动删除、截断、回滚或重试。脚本只负责把现场如实报出来。
    """


def tail_paths() -> dict[str, Path]:
    """按**当前** `TAIL_DIR` 现算 —— 写成函数而不是模块常量，测试才能重定向。"""
    return {
        "dir": TAIL_DIR,
        "manifest": TAIL_DIR / "tail_manifest.jsonl",
        "fingerprint": TAIL_DIR / "tail_fingerprint.json",
        "report": TAIL_DIR / "tail_completion_report.json",
        "h1": TAIL_DIR / "h1_tail_f5.jsonl",
        "mcq": TAIL_DIR / "mcq_tail_f5.jsonl",
        "e3_dir": TAIL_DIR / "h1_e3_block_hidden",
    }


# -------------------------------------------------------------- source 冻结
def tree_digest(directory: Path) -> str:
    """与既有 freeze manifest 同一配方：sorted(name) -> name + b"\\0" + sha256(bytes)。"""
    h = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        blob = (Path(directory) / name).read_bytes()
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(blob).digest())
    return h.hexdigest()


def source_freeze() -> dict:
    """本路径依赖的**全部**上游真源。任一漂移即在模型加载前拒绝。

    覆盖：两臂 finalized 六项、两份 run fingerprint、H1 frozen-hidden 树摘要、
    既有 E3 cache 的 manifest / report / 树摘要。**不只是两份 records。**
    """
    out: dict[str, str] = {}
    for name in FREEZE_PROD_FILES:
        out[f"production/{name}"] = file_sha256(PROD / name)
    out["production/h1_unit_hidden__tree_digest"] = tree_digest(PROD / "h1_unit_hidden")
    for name in FREEZE_REPOOL_FILES:
        out[f"repool/{name}"] = file_sha256(REPOOL / name)
    out["repool/h1_e3_block_hidden__tree_digest"] = tree_digest(
        REPOOL / "h1_e3_block_hidden")
    return out


# ------------------------------------------------------------ 耐久追加 / 读取
def assert_appendable(path: Path) -> None:
    """append 之前确认文件以换行结束 —— 否则新行会**焊死**一条永久损坏的中间行。

    这是 `read_manifest_events` 之外的第二道防线：即便某条路径绕过了读取校验，
    也不可能把内容接在崩溃留下的半行后面。
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return
    with p.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            raise TailRecoveryStop(
                f"拒绝 append：{p} 末尾是未终止的半行（崩溃截断）。\n"
                "  接着写会制造永久损坏的中间行。本 runner 不自动截断 —— "
                "请另出 Recovery Preflight。")


def durable_append(path: Path, obj: dict) -> str:
    """append-only + flush + fsync，返回**该行（不含换行）**的 sha256。"""
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)
    blob = line.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_appendable(path)
    with path.open("ab") as fh:
        fh.write(blob + b"\n")
        fh.flush()
        os.fsync(fh.fileno())
    return hashlib.sha256(blob).hexdigest()


def read_jsonl_strict(path: Path, *, what: str) -> list[tuple[dict, str]]:
    """严格读取结果 JSONL：返回 [(obj, line_sha256)]。

    尾行未以换行结束 → 崩溃截断的半行。**拒绝**，不自动截断也不忽略：
    结果文件被静默改动就没有 append-only 保证可言了。
    """
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise TailRecoveryStop(
            f"{what} 末尾是未终止的半行（崩溃截断）：{path}\n"
            "  本脚本不自动截断结果文件。请另出 Recovery Preflight 裁定。")
    out: list[tuple[dict, str]] = []
    for i, line in enumerate(raw.split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            out.append((json.loads(line), hashlib.sha256(line).hexdigest()))
        except json.JSONDecodeError as exc:
            raise TailRecoveryStop(
                f"{what} 第 {i} 行不是合法 JSON：{path}（{exc}）") from exc
    return out


def read_manifest_events(path: Path) -> list[dict]:
    """append-only manifest：任何坏行都拒绝，**包括未终止的尾行**。

    曾经这里"容忍"崩溃留下的半行、只是不解析它。那是不安全的契约：随后的
    `durable_append` 会直接接在半行后面，把一次可恢复的截断变成一条永久损坏的
    中间行。现在一律停机 —— 本 runner 不自动截断，交 Recovery Preflight 裁定。
    """
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise TailRecoveryStop(
            f"manifest 末尾是未终止的半行（崩溃截断）：{path}\n"
            "  本 runner **不自动截断、也不继续 append**。请另出 Recovery Preflight。")
    out: list[dict] = []
    for i, line in enumerate(raw.split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TailRecoveryStop(
                f"manifest 第 {i} 行损坏：{path}（{exc}）") from exc
    return out


def read_f5_records(path: Path, arm: str) -> dict[str, dict]:
    """unit_id -> {"sha256", "record"}。重复 unit_id 即拒绝。"""
    key = "trace_id" if arm == "h1" else "example_id"
    index: dict[str, dict] = {}
    for obj, sha in read_jsonl_strict(path, what=f"{arm}_tail_f5.jsonl"):
        uid = obj.get(key)
        if uid is None:
            raise TailRecoveryStop(f"{path}: 存在缺 {key} 的记录行")
        if uid in index:
            raise TailRecoveryStop(
                f"{path}: unit_id {uid!r} 出现多行 —— 重复写入，拒绝继续")
        index[uid] = {"sha256": sha, "record": obj}
    return index


def read_f5_index(path: Path, arm: str) -> dict[str, str]:
    return {k: v["sha256"] for k, v in read_f5_records(path, arm).items()}


# ------------------------------------------------------- F5 记录本体的校验
def assert_tail_f5_row(rec: dict, *, arm: str, uid: str, row: dict) -> None:
    """一条 tail F5 记录的**完整**校验 —— 写盘前与 repair 补记前都要过这一关。

    "JSON 能解析"远不足以把一行补记成 `done`：schema / arm / ID / 口径字段 /
    provenance / stable 字段的完整性与有限性都必须自证，并复用 shadow 的冻结验收。
    """
    key = "trace_id" if arm == "h1" else "example_id"
    checks = {
        "schema": rec.get("schema") == shadow_f5.SHADOW_SCHEMA,
        "variant": rec.get("variant") == TAIL_VARIANT,
        "tail_schema": rec.get("tail_schema") == tail.TAIL_SCHEMA,
        "arm": rec.get("arm") == arm,
        "unit_id": rec.get(key) == uid,
        "interpretation": rec.get("interpretation") == shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": rec.get("deviation_class") == shadow_f5.DEVIATION_LABEL,
        "not_primary": bool(rec.get("not_primary")),
    }
    bad = sorted(k for k, ok in checks.items() if not ok)
    if bad:
        raise TailRecoveryStop(f"{arm}/{uid}: F5 记录字段不合格 {bad}")

    prov = rec.get("provenance") or {}
    if prov.get("path") != TAIL_VARIANT:
        raise TailRecoveryStop(f"{arm}/{uid}: provenance.path != {TAIL_VARIANT}")
    if prov.get("prompt_hash") != row.get("prompt_hash"):
        raise TailRecoveryStop(f"{arm}/{uid}: provenance.prompt_hash 与冻结记录不符")

    stable = rec.get("stable_fields")
    if not isinstance(stable, dict):
        raise TailRecoveryStop(f"{arm}/{uid}: stable_fields 缺失")
    names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1" else shadow_f5.MCQ_STABLE_FIELDS)
    missing = [n for n in names if n not in stable]
    if missing:
        raise TailRecoveryStop(f"{arm}/{uid}: stable_fields 缺 {missing}")
    for n in names:
        v = stable[n]
        if v is None:
            if n != "f5_completion_perplexity":
                raise TailRecoveryStop(f"{arm}/{uid}: {n} 不得为 None")
            continue
        if not math.isfinite(float(v)):
            raise TailRecoveryStop(f"{arm}/{uid}: {n} 非有限：{v!r}")

    # 复用 shadow 的冻结验收，不另起一套判据
    try:
        if arm == "h1":
            n_tok, n_id = prov.get("n_completion_tokens"), prov.get("n_identifier_positions")
            if not isinstance(n_tok, int) or not isinstance(n_id, int):
                raise TailRecoveryStop(f"h1/{uid}: provenance 缺 token / identifier 计数")
            shadow_f5.assert_h1_acceptance(stable, row, n_tok=n_tok, n_id=n_id)
        else:
            shadow_f5.assert_mcq_acceptance(stable, row, {"A": 0.0, "B": -1.0})
    except shadow_f5.ShadowAcceptanceError as exc:
        raise TailRecoveryStop(f"{arm}/{uid}: shadow 验收不过：{exc}") from exc


# ------------------------------------------------------------- 模型身份核对
IDENTITY_KEYS = ("model_identity_hash", "tokenizer_identity_hash", "hidden_index")


def frozen_identity(run_fp: dict) -> dict:
    """从冻结 production run fingerprint 取**路径无关**的身份三项。"""
    return {"model_identity_hash": repool.model_identity_hash(run_fp["model"]),
            "tokenizer_identity_hash": repool.tokenizer_identity_hash(run_fp["tokenizer"]),
            "hidden_index": dict(run_fp["hidden_index"])}


def assert_model_identity(*, live: dict, frozen_h1: dict, frozen_mcq: dict) -> dict:
    """在**第一条 trace 之前**把现场身份钉到两份冻结 production fingerprint 上。

    只拿"首个 `segment_open`"当基线是不够的：第一次就传错 `--model-path`，
    那个错值会直接变成"新基准"。锚点最终会拦下它，但那要等到逐 trace 才发现。
    """
    ref_h1, ref_mcq = frozen_identity(frozen_h1), frozen_identity(frozen_mcq)
    cross = [k for k in IDENTITY_KEYS if ref_h1[k] != ref_mcq[k]]
    if cross:
        raise TailRecoveryStop(
            f"两份冻结 run fingerprint 的身份项彼此不一致：{cross} —— "
            "上游 production 本身有问题，拒绝启动")
    for name, ref in (("h1", ref_h1), ("mcq", ref_mcq)):
        bad = [k for k in IDENTITY_KEYS if live.get(k) != ref[k]]
        if bad:
            detail = "\n".join(f"    {k}: 冻结={ref[k]!r} 现场={live.get(k)!r}"
                               for k in bad)
            raise TailRecoveryStop(
                f"现场模型/分词器身份与冻结 {name} run fingerprint 不一致：\n{detail}\n"
                "  —— 极可能是 --model-path 传错；在跑第一条 trace 之前停止")
    return ref_h1


# ------------------------------------------------------------------ NPZ 写盘
def npz_atomic(path: Path, arrays: dict, meta: dict) -> dict:
    """每 trace 一个 NPZ，原子**创建**（os.link），绝不覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    if path.exists():
        raise ScopeViolation(f"NPZ 目标已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ScopeViolation(f"陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    with tmp.open("wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
                 **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except FileExistsError as exc:
        raise ScopeViolation(f"NPZ 目标在写入期间出现，拒绝覆盖：{path}") from exc
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(
            f"os.link 不可用或失败，fail-closed：{type(exc).__name__}: {exc}；"
            f"临时文件保留在 {tmp}") from exc
    os.unlink(tmp)
    return npz_digest(path)


def npz_digest(path: Path) -> dict:
    blob = Path(path).read_bytes()
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def load_npz_blocks(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files if k.startswith("block_")}


# --------------------------------------------------------------- resume 规划
def plan_resume(*, arm: str, unit_ids, events, f5_index, npz_index) -> dict:
    """**纯函数**的 resume 状态机 —— 只看数据，不碰磁盘。

    分档（`npz_index` 为 None 表示该臂不产 NPZ）：

        done          manifest 已 done **且**输出的 SHA/bytes/block/position 全部吻合
        failed        工程失败终态；**不重新入队**，且阻塞 completion report
        repair        输出齐全、仅 manifest 未记 done（崩溃在最后一步）→ 校验后补记
        rerun_orphan  NPZ 在、F5 行不在 → 重跑并与既有 NPZ **逐位比对**，不删除
        pending       未开始

    `problems` 非空即拒绝启动 —— 那是"done 与输出不符"或"写序颠倒"之类的不自洽，
    只能由研究者裁定，脚本不猜。
    """
    ids_all = list(unit_ids)
    latest: dict[str, dict] = {}
    for ev in events:
        if ev.get("kind") != "unit" or ev.get("arm") != arm:
            continue
        latest[ev["unit_id"]] = ev

    wants_npz = npz_index is not None
    done, failed, repair, rerun_orphan, pending = [], [], [], [], []
    problems: list[str] = []

    # ── 集合严格闭合：多出来的东西和缺失一样致命 ──────────────────────────
    expected = set(ids_all)
    for label, extra in (("F5 行", set(f5_index) - expected),
                         ("manifest unit 事件", set(latest) - expected),
                         ("NPZ", (set(npz_index) - expected) if wants_npz else set())):
        if extra:
            problems.append(
                f"{arm}: 出现 expected 集合之外的{label} {sorted(extra)[:5]}"
                f"（共 {len(extra)} 个）—— 产物含未知单元，拒绝继续")

    for uid in ids_all:
        ev = latest.get(uid)
        has_row = uid in f5_index
        has_npz = wants_npz and uid in npz_index
        status = (ev or {}).get("status")

        if status == "failed":
            failed.append(uid)
            continue

        if status == "done":
            bad = []
            if not has_row:
                bad.append("manifest done 但 F5 行缺失")
            elif f5_index[uid] != ev.get("f5_sha256"):
                bad.append(f"F5 行 SHA 不符（记录 {ev.get('f5_sha256')} / 实际 "
                           f"{f5_index[uid]}）")
            if wants_npz:
                if not has_npz:
                    bad.append("manifest done 但 NPZ 缺失")
                else:
                    got = npz_index[uid]
                    if got.get("sha256") != ev.get("npz_sha256"):
                        bad.append("NPZ SHA 不符")
                    if int(got.get("bytes", -1)) != int(ev.get("npz_bytes", -2)):
                        bad.append("NPZ 字节数不符")
                    if got.get("n_blocks") is not None and \
                            int(got["n_blocks"]) != int(ev.get("n_blocks", -1)):
                        bad.append("NPZ block 数不符")
                    if got.get("position_digest") is not None and \
                            got["position_digest"] != ev.get("position_digest"):
                        bad.append("position digest 不符")
            if bad:
                problems.append(f"{arm}/{uid}: " + "；".join(bad))
            else:
                done.append(uid)
            continue

        # ── 未到终态 ──────────────────────────────────────────────────────
        if has_row and (has_npz or not wants_npz):
            repair.append(uid)
        elif has_row and wants_npz and not has_npz:
            problems.append(
                f"{arm}/{uid}: 有 F5 行但 NPZ 缺失 —— 写序被破坏，脚本不猜，"
                "需 Recovery Preflight")
        elif (not has_row) and has_npz:
            rerun_orphan.append(uid)
        else:
            pending.append(uid)

    return {"arm": arm, "n_units": len(ids_all), "done": done, "failed": failed,
            "repair": repair, "rerun_orphan": rerun_orphan, "pending": pending,
            "problems": problems,
            "todo_gpu": sorted(set(rerun_orphan) | set(pending))}


def unit_ids_for(arm: str, rows: list[dict]) -> list[str]:
    """确定性顺序 —— `sorted(unit_id)`，与源文件行序无关（见 `UNIT_ORDER_RULE`）。"""
    key = repool.derive_unit_id if arm == "h1" else (lambda r: r["example_id"])
    ids = [key(r) for r in rows]
    if len(set(ids)) != len(ids):
        raise TailRecoveryStop(f"{arm}: analyzable 记录里存在重复 unit_id")
    return sorted(ids)


def next_segment_path() -> Path:
    """段号命名空间 `tail_segNNN_gpu_telemetry.csv`，取当前最大值 +1。"""
    d = tail_paths()["dir"]
    used = []
    for p in (sorted(d.glob("tail_seg*_gpu_telemetry.csv")) if d.exists() else []):
        digits = "".join(ch for ch in p.name[len("tail_seg"):] if ch.isdigit())
        if digits:
            used.append(int(digits))
    return d / f"tail_seg{(max(used) + 1) if used else 1:03d}_gpu_telemetry.csv"


# ---------------------------------------------------------------- 指纹与报告
def build_fingerprint(cfg, freeze: dict, scale: dict | None = None) -> dict:
    # scale 省略时**自行**从 finalized 输入重建 —— publish 与 re-check 两条
    # 路径都不传参，靠这一处保证它们逐字节一致。
    if scale is None:
        scale = observed_scale(cfg, load_analyzable("h1"), load_analyzable("mcq"))
    return {
        "schema": TAIL_RUN_SCHEMA,
        "variant": TAIL_VARIANT,
        "tail_schema": tail.TAIL_SCHEMA,
        "shadow_schema": shadow_f5.SHADOW_SCHEMA,
        "projection_schema": sl.STABLE_PROJECTION_SCHEMA,
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": shadow_f5.DEVIATION_LABEL,
        "not_primary": ("原 14 项 H1 B4 仍是 frozen implementation primary，"
                        "保留不覆盖不作废；本路径**不得**冒充 primary / confirmatory"),
        "backend": dict(TAIL_BACKEND),
        "unit_order_rule": UNIT_ORDER_RULE,
        "scale": dict(scale) if scale else None,
        "scale_pin_check": (assert_scale(scale) if scale else None),
        "artifact_root": str(ARTIFACT_ROOT),
        "geometry": GEOMETRY.as_dict(),
        "e3_config": {"e3_n_blocks": cfg.e3_n_blocks, "e3_block_salt": cfg.e3_block_salt,
                      "pooling_schema": repool.POOLING_SCHEMA,
                      "token_position_selection_rule":
                          repool.TOKEN_POSITION_SELECTION_RULE},
        "source_freeze": dict(freeze),
        "environment": {"libraries": library_versions(),
                        "platform": platform.platform(),
                        "python": sys.version.split()[0]},
    }


def assert_source_freeze_matches(freeze: dict) -> None:
    """在**模型加载之前**核对上游真源。任一漂移即拒绝。"""
    fp_path = tail_paths()["fingerprint"]
    if not fp_path.exists():
        return
    recorded = json.loads(fp_path.read_text(encoding="utf-8")).get("source_freeze") or {}
    drift = []
    for key in sorted(set(recorded) | set(freeze)):
        a, b = recorded.get(key, "<missing>"), freeze.get(key, "<missing>")
        if a != b:
            drift.append(f"  {key}: 指纹={a} 当前={b}")
    if drift:
        raise TailRecoveryStop(
            "上游真源相对 tail_fingerprint.json 发生漂移，拒绝启动（模型未加载）：\n"
            + "\n".join(drift))


def assert_fingerprint_matches(cfg, freeze: dict) -> None:
    """已存在的 `tail_fingerprint.json` 必须与现场 `build_fingerprint()` **完全一致**。

    比 source freeze 更严：库版本、平台、后端口径、E3 配置、规模、顺序规则也在内。
    同样在模型加载之前执行。
    """
    path = tail_paths()["fingerprint"]
    if not path.exists():
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = build_fingerprint(cfg, freeze)
    if recorded != current:
        diff = [k for k in sorted(set(recorded) | set(current))
                if recorded.get(k) != current.get(k)]
        raise TailRecoveryStop(
            f"既有 tail_fingerprint.json 与现场重算不一致，差异键：{diff}\n"
            f"  {path} —— 拒绝启动（模型未加载）")


def audit_manifest_arms(events: list[dict]) -> list[str]:
    """manifest 里的 unit 事件必须归属到某一臂 —— 未知 arm 不能被 `plan_resume`
    的按臂过滤悄悄吞掉。"""
    problems = []
    for i, ev in enumerate(events, 1):
        if ev.get("kind") != "unit":
            continue
        if ev.get("arm") not in ("h1", "mcq"):
            problems.append(f"manifest 第 {i} 条 unit 事件的 arm={ev.get('arm')!r} 非法")
        if not isinstance(ev.get("unit_id"), str) or not ev.get("unit_id"):
            problems.append(f"manifest 第 {i} 条 unit 事件缺 unit_id")
    return problems


def formal_status(cfg, h1_rows: list[dict], mcq_rows: list[dict]) -> dict:
    """重算两臂完成度 —— 只读磁盘真相，不信任任何缓存计数。"""
    p = tail_paths()
    events = read_manifest_events(p["manifest"])
    npz_index = npz_index_on_disk(p["e3_dir"])
    global_problems = audit_manifest_arms(events)
    per_arm, complete = {}, not global_problems
    for arm, rows in (("h1", h1_rows), ("mcq", mcq_rows)):
        ids = unit_ids_for(arm, rows)
        f5_index = read_f5_index(p[arm], arm)
        plan = plan_resume(arm=arm, unit_ids=ids,
                           events=events, f5_index=f5_index,
                           npz_index=npz_index if arm == "h1" else None)
        # 集合逐一相等，不是"数目相等" —— 多一条、少一条、换一条都必须被抓到
        sets_ok = set(f5_index) == set(ids)
        if arm == "h1":
            sets_ok = sets_ok and set(npz_index) == set(ids)
        ok = (len(plan["done"]) == len(ids) and not plan["failed"] and sets_ok
              and not plan["problems"] and not plan["todo_gpu"] and not plan["repair"])
        per_arm[arm] = {"n_expected": len(ids), "n_done": len(plan["done"]),
                        "n_failed": len(plan["failed"]),
                        "n_remaining": len(plan["todo_gpu"]) + len(plan["repair"]),
                        "n_problems": len(plan["problems"]),
                        "n_todo_gpu": len(plan["todo_gpu"]),
                        "n_repair": len(plan["repair"]),
                        "f5_key_set_matches_expected": set(f5_index) == set(ids),
                        "npz_key_set_matches_expected": (set(npz_index) == set(ids)
                                                         if arm == "h1" else None),
                        "failed_units": sorted(plan["failed"])[:20],
                        "problems": plan["problems"][:20], "complete": ok}
        complete = complete and ok

    pairs = sum(int(ev.get("n_blocks") or 0) for ev in events
                if ev.get("kind") == "unit" and ev.get("arm") == "h1"
                and ev.get("status") == "done")
    n_npz = len(npz_index)
    scale = observed_scale(cfg, h1_rows, mcq_rows)
    assert_scale(scale)
    e3_ok = (n_npz == scale["h1"] and pairs == scale["e3_pairs"])
    return {
        "global_problems": global_problems,
        "schema": TAIL_REPORT_SCHEMA,
        "variant": TAIL_VARIANT,
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": shadow_f5.DEVIATION_LABEL,
        "not_primary": ("原 14 项 H1 B4 仍是 frozen implementation primary；"
                        "本报告**不得**被当成 primary / confirmatory 结果"),
        "arms": per_arm,
        "e3_npz_files": n_npz, "e3_expected_traces": scale["h1"],
        "e3_trace_block_pairs": pairs, "e3_expected_pairs": scale["e3_pairs"],
        "scale": scale, "artifact_root": str(ARTIFACT_ROOT),
        "e3_complete": e3_ok,
        "complete": bool(complete and e3_ok),
        "source_freeze": source_freeze(),
        "downstream_contract": ("本报告缺失或 complete=false 时，下游**必须拒绝读取** "
                                "tail_stabilized 产物"),
    }


def npz_index_on_disk(e3_dir: Path) -> dict[str, dict]:
    """unit_id -> {sha256, bytes, n_blocks, position_digest}。**逐文件现算。**

    连 block 数与 position digest 也现算，是为了让 "done 事件 vs 输出" 的核对不止
    比字节 —— 一份字节相同但块归属被改写过的 NPZ 应当同样被拒。
    """
    if not Path(e3_dir).exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(Path(e3_dir).glob("*.npz")):
        entry = npz_digest(path)
        try:
            with np.load(path, allow_pickle=False) as z:
                names = set(z.files)
                entry["n_blocks"] = sum(1 for k in names if k.startswith("block_"))
                meta = json.loads(str(z["meta"].item())) if "meta" in names else {}
            pos = {int(k): [int(x) for x in v]
                   for k, v in (meta.get("token_positions") or {}).items()}
            entry["position_digest"] = repool.position_digest(pos) if pos else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise TailRecoveryStop(f"NPZ 不可读或 meta 损坏：{path}（{exc}）") from exc
        out[path.stem] = entry
    return out


def maybe_publish_report(cfg, h1_rows, mcq_rows) -> dict:
    """**只有**两臂全闭合才发布 completion report；未闭合就绝不提前出现。

    已存在的 report 必须与现场重算**完全一致** —— 否则说明产物在发布之后被动过。
    """
    status = formal_status(cfg, h1_rows, mcq_rows)
    path = tail_paths()["report"]
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != status:
            diff = [k for k in sorted(set(recorded) | set(status))
                    if recorded.get(k) != status.get(k)]
            raise TailRecoveryStop(
                f"既有 completion report 与现场重算不一致，差异键：{diff}\n"
                f"  {path} —— 发布之后产物被动过，拒绝继续")
        return status
    if status["complete"]:
        publish_json_exclusive(path, status)
        print(f"[report] 两臂闭合，已发布 {path}")
    else:
        print("[report] 未闭合 —— **不生成** completion report")
    return status


# ------------------------------------------------------------ 单元耐久落盘
def process_unit(*, arm: str, uid: str, out: dict, paths: dict, segment: str,
                 row: dict, cfg, orphan: bool = False) -> dict:
    """把一条**已算好**的单元耐久落盘，返回追加的 manifest done 事件。

    写序本身就是崩溃语义的一部分，不可调换：

        1. E3 NPZ 原子创建（H1）   崩在这之后 → 孤儿 NPZ，下次重跑并逐位比对
        2. F5 行 append + fsync    崩在这之后 → 输出齐全、只差 done，下次补记
        3. manifest done + fsync   带 F5/NPZ 的 SHA、字节、块数、position digest

    `orphan=True`：既有 NPZ **不删除**，先做**完整 meta 自洽校验**
    （trace_id / prompt_hash / source_fingerprint_id / token_positions / dtype /
    shape / 有限性），再与重算结果逐位比对，两关都过才复用。
    """
    assert_tail_f5_row(out["record"], arm=arm, uid=uid, row=row)
    if arm == "h1":
        npz_path = paths["e3_dir"] / f"{uid}.npz"
        if orphan:
            try:
                repool.validate_npz_self_consistency(npz_path, row, cfg,
                                                     row.get("run_hash"))
            except (ValueError, KeyError, OSError) as exc:
                raise tail.TailStop(
                    f"{uid}: 孤儿 NPZ 自洽校验不过 —— 不删除，等 Recovery Preflight",
                    unit_id=uid, gate="orphan_npz_meta",
                    diagnostic={"reason": f"{type(exc).__name__}: {exc}"}) from exc
            existing = load_npz_blocks(npz_path)
            same = set(existing) == set(out["arrays"]) and all(
                np.array_equal(existing[k], out["arrays"][k]) for k in out["arrays"])
            if not same:
                raise tail.TailStop(
                    f"{uid}: 孤儿 NPZ 与重算结果不一致 —— 不删除，等 Recovery Preflight",
                    unit_id=uid, gate="orphan_npz_mismatch",
                    diagnostic={"existing_keys": sorted(existing),
                                "recomputed_keys": sorted(out["arrays"])})
            digest = npz_digest(npz_path)
        else:
            digest = npz_atomic(npz_path, out["arrays"], out["npz_meta"])
        extra = {"npz_sha256": digest["sha256"], "npz_bytes": digest["bytes"],
                 "n_blocks": out["n_blocks"], "block_ids": out["block_ids"],
                 "position_digest": out["position_digest"]}
    else:
        extra = {"letter_row_index": out["letter_row_index"]}

    f5_sha = durable_append(paths[arm], out["record"])
    event = {"kind": "unit", "arm": arm, "unit_id": uid, "status": "done",
             "segment": segment, "f5_sha256": f5_sha,
             "n_completion_tokens": out["n_completion_tokens"], **extra}
    durable_append(paths["manifest"], event)
    return event


#: `failed` 的**唯一**合法解读。刻意不叫 "runtime technical attrition" ——
#: 那是 population 层的说法，而本路径是 sensitivity，**不改变** confirmatory population。
FAILED_SEMANTICS = (
    "engineering consistency failure of the tail sensitivity path；"
    "本路径是 protocol_deviation_numerical_sensitivity，**不改变** confirmatory "
    "population，因此既**不是**预注册允许的科学排除，也**不**新增 population 的 "
    "runtime technical attrition；不自动重试")


def failed_event(*, arm: str, uid: str, segment: str, exc: Exception) -> dict:
    """工程一致性失败事件（见 `FAILED_SEMANTICS`）。"""
    return {"kind": "unit", "arm": arm, "unit_id": uid, "status": "failed",
            "segment": segment, "gate": getattr(exc, "gate", "shadow_acceptance"),
            "message": str(exc), "diagnostic": getattr(exc, "diagnostic", {}),
            "semantics": FAILED_SEMANTICS}


# ------------------------------------------------------------------ 正式执行
def run_formal(arm: str, max_new: int, model_path: str, *,
               defer_report: bool = False) -> int:
    if arm not in ("h1", "mcq"):
        raise ScopeViolation(f"--arm 只能是 h1 / mcq，收到 {arm!r}")
    if max_new < 1:
        raise ScopeViolation("--max-new-traces 必须 >= 1")

    from recover_attention.evaluation import v3_freeze
    fz = v3_freeze.check_v3_frozen(ROOT / "docs/paper/preregistration_v3.md",
                                   ROOT / "docs/paper/preregistration_v3.lock")
    if not fz.match:
        raise ScopeViolation("预注册 v3 与 lock 不一致，拒绝启动正式路径")

    cfg = V3AnalysisConfig()
    h1_rows, mcq_rows = load_analyzable("h1"), load_analyzable("mcq")
    scale = observed_scale(cfg, h1_rows, mcq_rows)
    assert_scale(scale)
    n_pairs = scale["e3_pairs"]

    # ── 模型加载**之前**：上游真源冻结核对 ────────────────────────────────
    freeze = source_freeze()
    assert_source_freeze_matches(freeze)
    assert_fingerprint_matches(cfg, freeze)
    print(f"[freeze] {len(freeze)} 项上游真源一致 ｜ prereg lock {fz.locked_version} ✓")

    p = tail_paths()
    rows = h1_rows if arm == "h1" else mcq_rows
    by_id = ({repool.derive_unit_id(r): r for r in h1_rows} if arm == "h1"
             else {r["example_id"]: r for r in mcq_rows})
    ids = unit_ids_for(arm, rows)
    events = read_manifest_events(p["manifest"])
    arm_problems = audit_manifest_arms(events)
    if arm_problems:
        raise TailRecoveryStop("manifest 含非法 unit 事件，拒绝启动：\n  "
                               + "\n  ".join(arm_problems))
    f5_records = read_f5_records(p[arm], arm)
    f5_index = {k: v["sha256"] for k, v in f5_records.items()}
    plan = plan_resume(arm=arm, unit_ids=ids, events=events, f5_index=f5_index,
                       npz_index=npz_index_on_disk(p["e3_dir"]) if arm == "h1" else None)
    if plan["problems"]:
        raise TailRecoveryStop("正式状态不自洽，拒绝启动：\n  "
                               + "\n  ".join(plan["problems"]))
    if plan["failed"]:
        raise TailRecoveryStop(
            f"存在未裁定的 `failed` 工程失败 {len(plan['failed'])} 条："
            f"{sorted(plan['failed'])[:5]}\n"
            "  **不自动重试**。请另出 Recovery Preflight 后再继续。")
    print(f"[plan  ] {arm}: done {len(plan['done']):,} ｜ 待补记 {len(plan['repair'])} ｜ "
          f"待跑 {len(plan['todo_gpu']):,}（其中 NPZ 孤儿 {len(plan['rerun_orphan'])}）")

    if not p["fingerprint"].exists():
        publish_json_exclusive(p["fingerprint"], build_fingerprint(cfg, freeze))
        print(f"[fp    ] 首段发布 {p['fingerprint'].name}")

    # ── 无 GPU 的 orphan 补记（输出齐全、只差 done）────────────────────────
    for uid in plan["repair"]:
        row = by_id[uid]
        # "JSON 能解析"不足以补记 done —— 先把这行 F5 记录整体校验一遍
        assert_tail_f5_row(f5_records[uid]["record"], arm=arm, uid=uid, row=row)
        ev = {"kind": "unit", "arm": arm, "unit_id": uid, "status": "done",
              "repaired": True,
              "repair_kind": "outputs_present_manifest_missing_done",
              "f5_sha256": f5_index[uid]}
        if arm == "h1":
            info = repool.validate_npz_self_consistency(
                p["e3_dir"] / f"{uid}.npz", row, cfg, row.get("run_hash"))
            ev.update({"npz_sha256": info["sha256"], "npz_bytes": info["bytes"],
                       "n_blocks": info["n_blocks"], "block_ids": info["block_ids"],
                       "position_digest": info["position_digest"]})
        durable_append(p["manifest"], ev)
        print(f"[repair] {uid} 输出自洽，补记 done（未重算数值）")

    todo = plan["todo_gpu"][:max_new]
    orphan = set(plan["rerun_orphan"])
    if not todo:
        print("[run   ] 本臂无待跑单元，跳过模型加载")
        maybe_publish_report(cfg, h1_rows, mcq_rows)
        return 0
    print(f"[run   ] 本段 {len(todo):,} 条（--max-new-traces {max_new:,}）")

    seg_path = next_segment_path()
    tel = start_telemetry(seg_path)                 # 模型加载**之前**
    backend, done_here = None, 0
    #: **默认就是失败**。只有正常遍历完成 + 末次遥测存活检查通过，才会被改写成
    #: `completed / 0`。此前写成 `("completed", 0)` 是个真实缺口：任何未预期异常
    #: （OOM / KeyboardInterrupt / 未捕获的 ValueError）都会让 segment_close 记成成功，
    #: 而进程其实以非零码带 traceback 退出。
    rc, stop_reason = 5, "aborted_before_completion"
    t_seg = time.time()
    try:
        n0 = wait_for_telemetry(tel)                # 确认采样器活着且首点已落盘
        print(f"[tele  ] pid={tel.proc.pid} -> {seg_path.name}（首点已落盘 n={n0}）")

        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        torch = backend.ctx["torch"]
        model, tokenizer = backend.ctx["model"], backend.ctx["tokenizer"]
        head = sl.Float32Head(model, torch)
        run_fp = backend.run_fingerprint(arm)
        identity = {
            "model_identity_hash": repool.model_identity_hash(run_fp["model"]),
            "tokenizer_identity_hash": repool.tokenizer_identity_hash(run_fp["tokenizer"]),
            "hidden_index": dict(run_fp["hidden_index"]),
            "projection_cache_mode": head.cache_mode,
        }
        # 现场身份 vs **两份冻结 production fingerprint** —— 在第一条 trace 之前
        assert_model_identity(
            live=identity,
            frozen_h1=json.loads((PROD / "h1_run_fingerprint.json").read_text("utf-8")),
            frozen_mcq=json.loads((PROD / "mcq_run_fingerprint.json").read_text("utf-8")))
        print(f"[ident ] model/tokenizer/hidden_index 与两份冻结 fingerprint 一致 ✓")
        prior = next((e for e in events if e.get("kind") == "segment_open"), None)
        if prior is not None:
            drift = [k for k in identity if k != "projection_cache_mode"
                     and prior.get(k) != identity[k]]
            if drift:
                raise TailRecoveryStop(
                    f"模型/分词器身份相对首段漂移：{drift}\n"
                    f"  首段={{{', '.join(f'{k}={prior.get(k)!r}' for k in drift)}}}\n"
                    f"  本段={{{', '.join(f'{k}={identity[k]!r}' for k in drift)}}}")
        durable_append(p["manifest"], {"kind": "segment_open", "arm": arm,
                                       "segment": seg_path.stem, "n_planned": len(todo),
                                       **identity})

        bare = space = frozen_mcq = None
        if arm == "mcq":
            bare = dlp.bare_option_token_ids(tokenizer, list(mci.MCQ_LETTERS))
            space = dlp.option_token_ids(tokenizer, list(mci.MCQ_LETTERS))
            frozen_mcq = dict(np.load(PROD / "mcq_hidden.npz"))

        t0 = time.time()
        for i, uid in enumerate(todo, 1):
            if (i - 1) % TELEMETRY_CHECK_EVERY == 0:
                check_telemetry_alive(tel)
            row = by_id[uid]
            try:
                if arm == "h1":
                    frozen_union = np.load(PROD / "h1_unit_hidden" / f"{uid}.npy")
                    with np.load(REPOOL / "h1_e3_block_hidden" / f"{uid}.npz",
                                 allow_pickle=False) as z:
                        frozen_e3 = {k: np.asarray(z[k]) for k in z.files
                                     if k.startswith("block_")}
                    out = tail_h1_unit(model, tokenizer, torch, head, row, cfg,
                                       frozen_union, frozen_e3)
                else:
                    out = tail_mcq_unit(model, tokenizer, torch, head, row,
                                        frozen_mcq[uid], bare, space)
                process_unit(arm=arm, uid=uid, out=out, paths=p, row=row, cfg=cfg,
                             segment=seg_path.stem, orphan=uid in orphan)
                done_here += 1
                if uid in orphan:
                    print(f"[orphan] {uid} 既有 NPZ 通过完整 meta 校验且与重算逐位一致")
            except (tail.TailStop, shadow_f5.ShadowAcceptanceError) as exc:
                ev = failed_event(arm=arm, uid=uid, segment=seg_path.stem, exc=exc)
                durable_append(p["manifest"], ev)
                print(f"[STOP  ] {arm}/{uid} gate={ev['gate']}：{exc}", file=sys.stderr)
                print("[STOP  ] 保留已完成前缀，不删除、不回滚、不自动重试。",
                      file=sys.stderr)
                rc, stop_reason = 3, f"unit_failed:{ev['gate']}"
                break
            if i % 200 == 0:
                print(f"[run   ] {i}/{len(todo)} ({(time.time() - t0) / i:.3f} s/trace)")
        else:
            check_telemetry_alive(tel)               # 收尾再确认一次采样器仍在
            rc, stop_reason = 0, "completed"         # ← **唯一**写成功的地方
            print(f"[run   ] 本段完成 {len(todo):,} 条，"
                  f"{(time.time() - t0) / max(1, len(todo)):.3f} s/trace")
    except TailRecoveryStop as exc:
        rc, stop_reason = 4, f"recovery_stop:{type(exc).__name__}"
        print(f"[REFUSE] {exc}", file=sys.stderr)
    except BaseException as exc:                     # noqa: BLE001
        # OOM / Ctrl-C / 未预期异常：如实记为异常收场后**原样抛出**，
        # 绝不让 segment_close 留下一条 completed。
        rc, stop_reason = 5, f"unexpected:{type(exc).__name__}"
        print(f"[ABORT ] 未预期异常 {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        stop_telemetry(tel)
        backend = None
        n_samples = telemetry_sample_count(seg_path)
        durable_append(p["manifest"], {
            "kind": "segment_close", "arm": arm, "segment": seg_path.stem,
            "planned": len(todo), "completed": done_here,
            "repaired": len(plan["repair"]),
            "failed": 1 if stop_reason.startswith("unit_failed") else 0,
            "stopped_reason": stop_reason,
            "elapsed_seconds": round(time.time() - t_seg, 2),
            "telemetry_file": seg_path.name,
            "telemetry_samples": n_samples,
            "telemetry_sha256": (file_sha256(seg_path) if seg_path.exists() else None),
            "exit_code": rc})
        print(f"[seg   ] close {seg_path.stem}: {done_here}/{len(todo)} 条，"
              f"reason={stop_reason}，遥测 {n_samples} 点")

    if rc == 4:
        return rc                # 恢复类停机：现场已如实记录，不再重算完成度
    if defer_report:
        print("[report] --defer-report：本段不发布 completion report")
        return rc
    status = maybe_publish_report(cfg, h1_rows, mcq_rows)
    print(f"[state ] {arm}: {json.dumps(status['arms'][arm], ensure_ascii=False)}")
    return rc


def run_finalize() -> int:
    """**不加载模型**：只重算完成度并在两臂全闭合时发布 completion report。

    driver 用它把"发布终态产物"与"跑一段"彻底分开 —— 四段全部通过机械门之后
    才调用一次。
    """
    from recover_attention.evaluation import v3_freeze
    fz = v3_freeze.check_v3_frozen(ROOT / "docs/paper/preregistration_v3.md",
                                   ROOT / "docs/paper/preregistration_v3.lock")
    if not fz.match:
        raise ScopeViolation("预注册 v3 与 lock 不一致，拒绝 finalize")
    cfg = V3AnalysisConfig()
    h1_rows, mcq_rows = load_analyzable("h1"), load_analyzable("mcq")
    assert_scale(observed_scale(cfg, h1_rows, mcq_rows))
    freeze = source_freeze()
    assert_source_freeze_matches(freeze)
    assert_fingerprint_matches(cfg, freeze)
    status = maybe_publish_report(cfg, h1_rows, mcq_rows)
    print(f"[final ] complete={status['complete']} ｜ "
          f"h1 {status['arms']['h1']['n_done']}/{status['arms']['h1']['n_expected']} ｜ "
          f"mcq {status['arms']['mcq']['n_done']}/{status['arms']['mcq']['n_expected']} ｜ "
          f"E3 {status['e3_trace_block_pairs']}/{status['e3_expected_pairs']}")
    return 0 if status["complete"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--run", action="store_true", help="正式分段执行")
    ap.add_argument("--finalize", action="store_true",
                    help="不加载模型：两臂全闭合时发布 completion report")
    ap.add_argument("--arm", choices=("h1", "mcq"))
    ap.add_argument("--max-new-traces", type=int,
                    help="本段最多新跑多少条；必须显式给出，避免误跑全量")
    ap.add_argument("--defer-report", action="store_true",
                    help="本段结束后**不**发布 completion report（driver 用）")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--artifact-root", type=Path, default=None,
                    help="独立 artifact root（跨模型复现用）。省略即默认 "
                         "outputs/stage1（Qwen 冻结运行）")
    repool.add_geometry_cli_args(ap)
    args = ap.parse_args(argv)
    _geo = repool.geometry_from_cli(args, model_path=args.model_path)
    if args.artifact_root is not None or _geo is not repool.QWEN_GEOMETRY:
        info = set_artifact_root(args.artifact_root or DEFAULT_ARTIFACT_ROOT,
                                 geometry=_geo)
        print(f"[root ] {info['artifact_root']}")
    print(f"[geom ] block={GEOMETRY.block_index} tuple={GEOMETRY.tuple_index} "
          f"dim={GEOMETRY.hidden_dim} source={GEOMETRY.source}")
    if sum(bool(x) for x in (args.benchmark, args.run, args.finalize)) > 1:
        ap.error("--benchmark / --run / --finalize 三者互斥")
    if args.benchmark:
        return run_benchmark(args.model_path)
    if args.finalize:
        try:
            return run_finalize()
        except (TailRecoveryStop, ScopeViolation) as exc:
            print(f"[REFUSE] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 4
    if args.run:
        if not args.arm:
            ap.error("--run 必须指定 --arm {h1,mcq}")
        if args.max_new_traces is None:
            ap.error("--run 必须显式指定 --max-new-traces（段长由 Preflight 裁定）")
        try:
            return run_formal(args.arm, args.max_new_traces, args.model_path,
                              defer_report=args.defer_report)
        except (TailRecoveryStop, ScopeViolation) as exc:
            print(f"[REFUSE] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 4
    ap.error("选择 --benchmark / --run --arm ... --max-new-traces ... / --finalize")


if __name__ == "__main__":
    raise SystemExit(main())
