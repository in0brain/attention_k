r"""P2 微基准 v2：32 条 H1 陈述的单条编码成本实测（**检查范围已修订**）。

首轮（v1）在第 1 条被"全部 29 项必须有限"拦下而中止。事故记录与日志保留在
`p2_microbench__qwen_h1.incident.json` / `.log`，**不覆盖**。

**本版的检查范围修订（经明确授权，属协议变更，已登记）**

    必须有限（异常**立即停止**）
        hidden_states[0..20]   —— 目标层 20（block 19）及其**全部上游**
        实际读出的末词元向量   —— block 19 的末词元，就是要拿去做表示的那一个

    描述性审计（**只记录，不停止**）
        hidden_states[21..28]  —— 目标层**下游**，本任务不读
        logits                 —— 旧流程用它构造 B4，这里一并如实登记
        逐条记 shape / dtype / NaN / +Inf / -Inf 的个数，**末词元单列**

⚠️ 三条措辞纪律，写死在产物里：
    · **不得**写"全层检查通过" —— 本版只保证 0..20 与读出向量
    · **不得**断言溢出由 final norm 造成 —— 第 28 项是 norm **之后**的结果，
      异常也可能已经出现在最后一个 block 内部；**没有** norm 输入的观测
    · **不得**宣布旧 Stage 1 不受影响 —— 旧流程还用 logits 构造 B4；
      这次单条新文本既不能证明旧结果出错，也不能替旧结果排除数值问题。
      既有的 B4 完整性限制**继续保留**。

它**不做**的事（写死，不给开关）：
    · 不落任何隐藏向量产物
    · 不裁剪、不填零、不丢样本
    · 不训练、不打分、不产出任何估计量
    · 不生成、不采样
    · 不自动进入 870 条 / 10168 条正式提取
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage1/posthoc_readout_probes/p2_prep"
OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes"
STEM = "p2_microbench__qwen_h1_v2"

SCOPE_REVISION = {
    "revision_id": "finite_check_scope_v2",
    "authorised": True,
    "supersedes": "v1：全部 29 项 hidden_states 必须有限（作为停止条件）",
    "must_be_finite_abort_on_violation": [
        "hidden_states[0..20]（目标层 20 = block 19，及其全部上游）",
        "实际读出的末词元向量（block 19 的 last token）",
    ],
    "descriptive_audit_only_never_aborts": [
        "hidden_states[21..28]（目标层下游，本任务不读）",
        "logits（旧流程用它构造 B4）",
    ],
    "forbidden_wording": [
        "不得写『全层检查通过』—— 本版只保证 0..20 与读出向量",
        "不得断言溢出由 final norm 造成 —— 第 28 项是 norm 之后的结果，"
        "异常也可能已在最后一个 block 内部出现；没有 norm 输入的观测",
        "不得宣布旧 Stage 1 不受影响 —— 旧流程还用 logits 构造 B4，"
        "本次单条新文本既不能证明旧结果出错，也不能替旧结果排除数值问题；"
        "既有 B4 完整性限制继续保留",
    ],
    "what_this_is_not": "这是对**成本微基准检查范围**的明确修订，**不代表数值异常已修复**",
}


def _rss_gb() -> float:
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 1e9, 3)
    except Exception:
        return -1.0


def _tally(torch, t) -> dict:
    """一个张量的**描述性**计数。不改动张量，不裁剪，不填零。"""
    return {
        "shape": list(t.shape), "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "n_nan": int(torch.isnan(t).sum().item()),
        "n_posinf": int(torch.isposinf(t).sum().item()),
        "n_neginf": int(torch.isneginf(t).sum().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-have-authorisation", action="store_true")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-file", type=Path, default=PREP / "h1_microbatch_32.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--hash-weights", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    t_wall0 = time.perf_counter()

    man = json.loads((PREP / "h1_prep_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            args.batch_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    uniq_rows = [json.loads(x) for x in
                 (PREP / "h1_unique_texts.jsonl").read_text(encoding="utf-8")
                 .splitlines() if x.strip()]
    prep_texts = {u["text_sha256"]: u for u in uniq_rows}

    # ── 输入绑定 ──────────────────────────────────────────────────────────
    bad = []
    for r in rows:
        u = prep_texts.get(r["text_sha256"])
        if u is None:
            bad.append(f"{r.get('statement_id')}: 不在准备清单")
        elif u["text"] != r["text"]:
            bad.append(f"{r['text_sha256']}: 文本与清单不一致")
        elif hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != r["text_sha256"]:
            bad.append(f"{r.get('statement_id')}: 与自带 sha256 对不上")
    if bad:
        raise RuntimeError("小批输入与准备清单不符 —— 停：\n  " + "\n  ".join(bad[:5]))

    tok_cfg, enc_cfg = man["tokenizer_settings"], man["encoding_settings"]
    n_full, n_uniq = man["statements"]["n"], man["unique_texts"]["n"]
    layer = enc_cfg["layer_hidden_states_index"]
    d_model, n_layers = man["model"]["hidden_size"], man["model"]["num_hidden_layers"]
    n_hidden_expected = n_layers + 1

    # ── 配置闸（加载权重之前） ─────────────────────────────────────────────
    SUPPORTED = {"batch_size": (1, "逐条处理，不实现批处理"),
                 "padding": (False, "batch=1 不需要；批处理要左 padding，未实现"),
                 "chat_template": (False, "不套模板，也未实现套模板")}
    bad_cfg = [f"{k}={tok_cfg.get(k)!r}（只支持 {w!r}：{why}）"
               for k, (w, why) in SUPPORTED.items() if tok_cfg.get(k) != w]
    READ_OK = {"末词元（last token）", "末词元", "last_token", "last token"}
    if str(enc_cfg.get("read_position")) not in READ_OK:
        bad_cfg.append(f"read_position={enc_cfg.get('read_position')!r} 不在白名单")
    blk, idx = enc_cfg.get("layer_block_zero_based"), enc_cfg.get("layer_hidden_states_index")
    prod = json.loads((ROOT / "outputs/stage1/production/h1_run_fingerprint.json")
                      .read_text(encoding="utf-8"))
    hi = prod.get("hidden_index") or {}
    if not isinstance(blk, int) or not isinstance(idx, int):
        bad_cfg.append(f"层号缺失或非整数：{blk!r}/{idx!r}")
    elif idx != blk + 1:
        bad_cfg.append(f"层号不自洽：block {blk} 应对应 hidden_states[{blk + 1}]，写的是 {idx}")
    elif hi.get("block_index_zero_based") is None or hi.get("hidden_states_tuple_index") is None:
        bad_cfg.append("生产记录缺 hidden_index —— 无从比对冻结层，不得当作通过")
    elif (blk, idx) != (hi["block_index_zero_based"], hi["hidden_states_tuple_index"]):
        bad_cfg.append(f"读出层与生产冻结层不符：本次 {blk}→{idx}，"
                       f"生产 {hi['block_index_zero_based']}→{hi['hidden_states_tuple_index']}")
    if bad_cfg:
        raise SystemExit("[stop] 配置不受支持 —— 加载权重之前拒绝：\n  " + "\n  ".join(bad_cfg))

    plan = {
        "batch": str(args.batch_file.relative_to(ROOT)), "n": len(rows),
        "model_path": args.model_path,
        "add_special_tokens": tok_cfg["add_special_tokens"],
        "batch_size": tok_cfg["batch_size"], "padding": tok_cfg["padding"],
        "chat_template": tok_cfg["chat_template"],
        "layer_block_zero_based": blk, "layer_hidden_states_index": idx,
        "read_position": enc_cfg["read_position"],
        "extrapolate_to": {"h1_u1_statements": n_full, "unique_texts": n_uniq},
    }
    print("═══ 计划（v2，检查范围已修订）")
    for k, v in plan.items():
        print(f"  {k:<28} {v}")
    print(f"  必须有限（违反即停）         hidden_states[0..{layer}] + 读出末词元向量")
    print(f"  描述性审计（不停止）         hidden_states[{layer + 1}..{n_layers}] + logits")
    print("  产物                         **不落任何隐藏向量**")

    if not args.i_have_authorisation:
        print("\n[dry-run] 未授权 —— **不加载权重、不前向**。")
        (args.out_dir / f"{STEM}.dryrun.json").write_text(
            json.dumps({"schema": "p2_microbench_v2_dryrun", "plan": plan,
                        "scope_revision": SCOPE_REVISION,
                        "loaded_weights": False, "ran_forward": False},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    # ── ④ 配置与文件摘要**在前向之前**落盘 ─────────────────────────────────
    mp = Path(args.model_path)
    small = sorted(p for p in mp.glob("*") if p.is_file()
                   and p.suffix in (".json", ".txt", ".model")
                   and p.stat().st_size <= 64 * 1024 * 1024)
    t_dig0 = time.perf_counter()
    file_digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in small}
    weights = sorted((p.name, p.stat().st_size) for p in mp.glob("*.safetensors"))
    if args.hash_weights:
        for n, _ in weights:
            h = hashlib.sha256()
            with open(mp / n, "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    h.update(c)
            file_digest[n] = h.hexdigest()[:16]
    t_digest = round(time.perf_counter() - t_dig0, 3)

    preflight = {
        "schema": "p2_microbench_v2_preflight", "plan": plan,
        "scope_revision": SCOPE_REVISION,
        "input_binding": {
            "prep_manifest_sha256": hashlib.sha256(
                (PREP / "h1_prep_manifest.json").read_bytes()).hexdigest()[:16],
            "unique_texts_sha256": hashlib.sha256(
                (PREP / "h1_unique_texts.jsonl").read_bytes()).hexdigest()[:16],
            "batch_sha256": hashlib.sha256(args.batch_file.read_bytes()).hexdigest()[:16],
            "n_batch": len(rows), "all_batch_texts_in_prep": True},
        "file_digest_now": {
            "model_dir": str(mp), "small_files": file_digest,
            "weight_shards": [{"name": n, "bytes": s} for n, s in weights],
            "weights_hashed": bool(args.hash_weights), "digest_secs": t_digest,
            "caveat": ("**当前**文件摘要。生产时没留对应摘要，**无法倒推**"
                       "生产时文件与现在逐字节相同；只对今后的比对有用。")},
        "written_before_forward": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pf_path = args.out_dir / f"{STEM}.preflight.json"
    pf_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(f"[preflight] 已在前向**之前**落盘  {pf_path.name}  "
          f"（摘要耗时 {t_digest}s，**不计入编码耗时**）")

    per_rows: list[dict] = []
    rec: dict = {"schema": "p2_microbench_v2", "scope_revision": SCOPE_REVISION,
                 "plan": plan, "preflight": preflight}

    def flush(status: str, err: str | None = None) -> None:
        """无论成功失败，**已获得的信息都落盘**。"""
        rec.update(status=status, error=err, per_statement=per_rows,
                   n_completed=len(per_rows),
                   wall_secs=round(time.perf_counter() - t_wall0, 2),
                   peak_rss_gb=_rss_gb(),
                   generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        (args.out_dir / f"{STEM}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(ROOT / "src"))
        from recover_attention.generation import fingerprint as fp
        from recover_attention.generation.backends import load_backend

        t_load = time.perf_counter()
        ctx = load_backend(args.model_path)
        torch, model, tk = ctx["torch"], ctx["model"], ctx["tokenizer"]
        load_secs = round(time.perf_counter() - t_load, 2)
        model.eval()
        dev = next(model.parameters()).device
        rec["load_secs"] = load_secs
        rec["device"] = str(dev)
        rec["backend"] = ctx.get("backend")

        # 身份闸：必需字段缺失 = 拒绝
        m_fp, t_fp = fp.model_fingerprint(model, args.model_path, ctx.get("backend")), \
            fp.tokenizer_fingerprint(tk)
        REQ = (("model", "model_hash"), ("model", "model_type"),
               ("model", "num_hidden_layers"), ("model", "hidden_size"),
               ("tokenizer", "tokenizer_hash"), ("tokenizer", "vocab_size"))
        mine = {"model": m_fp, "tokenizer": t_fp}
        miss_ref, miss_now, mism = [], [], []
        for sect, key in REQ:
            want = (prod.get(sect) or {}).get(key, "__ABSENT__")
            got = mine[sect].get(key, "__ABSENT__")
            if want == "__ABSENT__" or want is None:
                miss_ref.append(f"{sect}.{key}")
            elif got == "__ABSENT__" or got is None:
                miss_now.append(f"{sect}.{key}")
            elif got != want:
                mism.append(f"{sect}.{key}: 本次 {got!r} ≠ 生产 {want!r}")
        if miss_ref or miss_now or mism:
            raise RuntimeError(f"身份核对未通过 —— 停：参考缺 {miss_ref}；"
                               f"本次缺 {miss_now}；不符 {mism}")
        rec["identity"] = {
            "config_fingerprint": {"model": m_fp, "tokenizer": t_fp,
                                   "all_required_present_and_equal": True},
            "WHAT_THIS_IS": "生产**配置**指纹一致；**不含**权重/词表字节",
        }
        print(f"[identity] 生产配置指纹一致（model_hash {m_fp['model_hash'][:12]}…）")

        warm = 2
        with torch.no_grad():
            for i, r in enumerate(rows):
                # ── 编码路径计时（分词 + 前向 + **必查项** + 取向量） ──────
                t_a = time.perf_counter()
                enc = tk(r["text"],
                         add_special_tokens=bool(tok_cfg["add_special_tokens"]),
                         return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_b = time.perf_counter()
                out = model(**enc, output_hidden_states=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_c = time.perf_counter()

                hs = out.hidden_states
                if len(hs) != n_hidden_expected:
                    raise RuntimeError(f"hidden_states 长度 {len(hs)} ≠ {n_hidden_expected}")

                # **必须有限**：0..layer 全部 + 读出末词元向量。违反立即停。
                for li in range(layer + 1):
                    if not bool(torch.isfinite(hs[li]).all()):
                        raise RuntimeError(
                            f"{r['statement_id']}: hidden_states[{li}] 含非有限值 —— "
                            f"停（必查范围 0..{layer}，本条位于目标层或其上游）")
                v = hs[layer][0, -1, :]
                if not bool(torch.isfinite(v).all()):
                    raise RuntimeError(
                        f"{r['statement_id']}: **读出的末词元向量**含非有限值 —— 停")
                if int(v.shape[0]) != d_model:
                    raise RuntimeError(f"维度 {int(v.shape[0])} ≠ {d_model} —— 停")
                v_cpu = v.detach().float().cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_d = time.perf_counter()

                # ── 描述性审计（**只记录，绝不停止**），单独计时 ────────────
                audit = {"downstream_hidden_states": {}, "logits": None,
                         "policy": "只记录，不停止；不裁剪、不填零、不丢样本"}
                for li in range(layer + 1, n_hidden_expected):
                    h = hs[li]
                    audit["downstream_hidden_states"][str(li)] = {
                        "full": _tally(torch, h),
                        "last_token": _tally(torch, h[0, -1, :]),
                    }
                lg = getattr(out, "logits", None)
                if lg is not None:
                    audit["logits"] = {"full": _tally(torch, lg),
                                       "last_token": _tally(torch, lg[0, -1, :])}
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_e = time.perf_counter()

                per_rows.append({
                    "i": i, "warmup": i < warm,
                    "statement_id": r["statement_id"],
                    "text_sha256": r["text_sha256"], "template": r["template"],
                    "n_tokens": int(enc["input_ids"].shape[1]),
                    "encode_secs": {
                        "tokenize": round(t_b - t_a, 6),
                        "forward": round(t_c - t_b, 6),
                        "required_checks_and_readout": round(t_d - t_c, 6),
                        "total": round(t_d - t_a, 6)},
                    "diagnostic_secs": round(t_e - t_d, 6),
                    "required_finite_range": f"hidden_states[0..{layer}] + 读出末词元向量",
                    "required_finite_passed": True,
                    "descriptive_audit": audit,
                })
                del out, hs, v, v_cpu

        # ── 汇总 ────────────────────────────────────────────────────────
        timed = [x for x in per_rows if not x["warmup"]]
        enc_tot = sorted(x["encode_secs"]["total"] for x in timed)
        fwd = sorted(x["encode_secs"]["forward"] for x in timed)
        diag = sorted(x["diagnostic_secs"] for x in timed)
        med = statistics.median(enc_tot)

        for u in uniq_rows:
            u["n_tokens"] = len(tk(u["text"], add_special_tokens=bool(
                tok_cfg["add_special_tokens"]))["input_ids"])
        by_len: dict[int, list[float]] = {}
        for x in timed:
            by_len.setdefault(x["n_tokens"], []).append(x["encode_secs"]["total"])
        len_med = {k: statistics.median(v) for k, v in sorted(by_len.items())}
        measured = sorted(len_med)
        s_u = s_a = 0.0
        for u in uniq_rows:
            p_ = len_med[min(measured, key=lambda L: abs(L - u["n_tokens"]))]
            s_u += p_
            s_a += p_ * u["n_records"]

        # 下游异常的汇总（描述性）
        dstats: dict[str, dict] = {}
        for x in per_rows:
            for li, d in x["descriptive_audit"]["downstream_hidden_states"].items():
                e = dstats.setdefault(li, {"n_statements_with_nonfinite_full": 0,
                                           "n_statements_with_nonfinite_last_token": 0,
                                           "max_n_nan_full": 0, "max_n_posinf_full": 0,
                                           "max_n_neginf_full": 0, "dtype": d["full"]["dtype"]})
                nf = d["full"]["n_nan"] + d["full"]["n_posinf"] + d["full"]["n_neginf"]
                lt = (d["last_token"]["n_nan"] + d["last_token"]["n_posinf"]
                      + d["last_token"]["n_neginf"])
                e["n_statements_with_nonfinite_full"] += int(nf > 0)
                e["n_statements_with_nonfinite_last_token"] += int(lt > 0)
                e["max_n_nan_full"] = max(e["max_n_nan_full"], d["full"]["n_nan"])
                e["max_n_posinf_full"] = max(e["max_n_posinf_full"], d["full"]["n_posinf"])
                e["max_n_neginf_full"] = max(e["max_n_neginf_full"], d["full"]["n_neginf"])
        lg_bad = sum(1 for x in per_rows if x["descriptive_audit"]["logits"] and
                     sum(x["descriptive_audit"]["logits"]["full"][k]
                         for k in ("n_nan", "n_posinf", "n_neginf")) > 0)

        rec.update({
            "n_timed": len(timed), "n_warmup_excluded": warm,
            "peak_gpu_alloc_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                  if torch.cuda.is_available() else None),
            "peak_gpu_reserved_gb": (round(torch.cuda.max_memory_reserved() / 1e9, 3)
                                     if torch.cuda.is_available() else None),
            "encode_secs_summary": {
                "median": round(med, 5), "p90": round(enc_tot[int(0.9 * (len(enc_tot) - 1))], 5),
                "min": round(enc_tot[0], 5), "max": round(enc_tot[-1], 5),
                "forward_only_median": round(statistics.median(fwd), 5),
                "definition": "分词 + 搬设备 + 前向 + **必查项** + 取末词元向量"},
            "diagnostic_secs_summary": {
                "median": round(statistics.median(diag), 5),
                "min": round(diag[0], 5), "max": round(diag[-1], 5),
                "definition": ("下游层 21..28 与 logits 的描述性审计；"
                               "**与编码耗时分开报告**，正式提取时可不做")},
            "downstream_audit_summary": {
                "note": ("**描述性**。这里出现非有限值**不**中止运行，"
                         "也**不**说明目标层有问题。"),
                "per_layer": dstats,
                "n_statements_with_nonfinite_logits": lg_bad,
                "n_statements": len(per_rows)},
            "extrapolation": {
                "method": "按实际词元长度分布逐条累加（编码耗时，不含诊断审计）",
                "measured_length_buckets": {str(k): round(v, 5) for k, v in len_med.items()},
                "unique_texts_only": {"n": n_uniq, "secs": round(s_u, 1),
                                      "hours": round(s_u / 3600, 3)},
                "all_records": {"n": n_full, "secs": round(s_a, 1),
                                "hours": round(s_a / 3600, 3)},
                "naive_median_times_n_for_reference": {
                    "unique_texts_hours": round(med * n_uniq / 3600, 3),
                    "all_records_hours": round(med * n_full / 3600, 3)},
                "not_covered": (f"权重加载（一次性 {load_secs}s）、文件摘要（{t_digest}s）、"
                                "向量落盘与索引写入 —— 正式提取还要付，本次**没测**"),
                "not_extrapolated_to_mcq": "MCQ 陈述未构造，长度分布没测过，不外推",
                "remaining_uncertainty": ("长度范围窄、32 条桶内中位数有噪声。"
                                          "**这是资源估算，不是保证值。**")},
        })
        flush("COMPLETED")
        e = rec["extrapolation"]
        print(f"\n加载 {load_secs}s  设备 {dev}")
        print(f"编码耗时  中位 {med * 1000:.1f} ms（前向 "
              f"{statistics.median(fwd) * 1000:.1f} ms）；"
              f"诊断审计另计 中位 {statistics.median(diag) * 1000:.1f} ms")
        print(f"峰值      GPU alloc {rec['peak_gpu_alloc_gb']}G  "
              f"reserved {rec['peak_gpu_reserved_gb']}G  RSS {_rss_gb()}G")
        print(f"下游审计  {json.dumps(dstats, ensure_ascii=False)}")
        print(f"          logits 含非有限值的条数 {lg_bad}/{len(per_rows)}")
        print(f"外推      唯一文本 {n_uniq} 条 ≈ {e['unique_texts_only']['hours']} h；"
              f"全部记录 {n_full} 条 ≈ {e['all_records']['hours']} h")
        print(f"[out] {args.out_dir / f'{STEM}.json'}")
        return 0
    except BaseException as exc:                      # noqa: BLE001
        flush("ABORTED", f"{type(exc).__name__}: {exc}")
        rec["traceback"] = traceback.format_exc()
        flush("ABORTED", f"{type(exc).__name__}: {exc}")
        print(f"\n[ABORTED] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"  已完成 {len(per_rows)} 条，**已获得的信息已落盘**："
              f"{args.out_dir / f'{STEM}.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
