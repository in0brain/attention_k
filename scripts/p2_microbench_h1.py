r"""P2 微基准：32 条 H1 陈述的**单次前向**成本实测。

⚠️ **这是本轮唯一需要加载模型权重的脚本。** 必须显式 `--i-have-authorisation`
   才会加载；否则只做 dry-run（打印将要做什么，不碰权重）。

它**只测成本**：
    · 单条前向耗时（中位 / p90 / 总）
    · 峰值显存（torch.cuda.max_memory_allocated / reserved）
    · 峰值 RSS
    · hidden_states 全层字节数与"只留冻结层"的字节数
    · 按实测外推到 10168 条（**只外推 H1**，不外推 MCQ —— 没测过）

它**不做**的事（写死，不给开关）：
    · 不落任何隐藏向量产物（`--dry-run` 与正式跑都一样）
    · 不训练、不打分、不产出任何估计量
    · 不生成、不采样
    · 不改任何已有产物

口径全部取自 `p2_prep/h1_prep_manifest.json`，**不在这里重新裁定**：
    add_special_tokens=True / batch=1 / 不 padding / 不套 chat template /
    末词元 / block 19 → hidden_states[20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage1/posthoc_readout_probes/p2_prep"
OUT = ROOT / "outputs/stage1/posthoc_readout_probes/p2_microbench__qwen_h1.json"


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-have-authorisation", action="store_true",
                    help="**明确授权后**才加载模型权重并前向；不给就只 dry-run")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-file", type=Path,
                    default=PREP / "h1_microbatch_32.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--hash-weights", action="store_true",
                    help="连权重分片一起做内容摘要（慢；默认只记名与大小）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    man = json.loads((PREP / "h1_prep_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            args.batch_file.read_text(encoding="utf-8").splitlines() if x.strip()]

    # ── ① 输入绑定：小批里的每条都必须**属于当前准备清单** ──────────────────
    #    否则计时可能测的是别的文本，外推就无从谈起。
    uniq_rows = [json.loads(x) for x in
                 (PREP / "h1_unique_texts.jsonl").read_text(encoding="utf-8")
                 .splitlines() if x.strip()]
    prep_texts = {u["text_sha256"]: u for u in uniq_rows}
    bad = []
    for r in rows:
        u = prep_texts.get(r["text_sha256"])
        if u is None:
            bad.append(f"{r.get('statement_id')}: text_sha256 不在准备清单里")
        elif u["text"] != r["text"]:
            bad.append(f"{r['text_sha256']}: 文本与清单不一致")
        elif hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != r["text_sha256"]:
            bad.append(f"{r.get('statement_id')}: 文本与自带 sha256 对不上")
    if bad:
        raise RuntimeError("小批输入与准备清单不符 —— 停：\n  " + "\n  ".join(bad[:5]))
    input_binding = {
        "prep_manifest_sha256": hashlib.sha256(
            (PREP / "h1_prep_manifest.json").read_bytes()).hexdigest()[:16],
        "unique_texts_sha256": hashlib.sha256(
            (PREP / "h1_unique_texts.jsonl").read_bytes()).hexdigest()[:16],
        "batch_sha256": hashlib.sha256(args.batch_file.read_bytes()).hexdigest()[:16],
        "n_batch": len(rows), "n_unique_in_prep": len(prep_texts),
        "all_batch_texts_in_prep": True,
        "template_texts": {k: v["text"] for k, v in
                           man["templates"].items()},
    }
    tok_cfg = man["tokenizer_settings"]
    enc_cfg = man["encoding_settings"]
    n_full = man["statements"]["n"]
    n_uniq = man["unique_texts"]["n"]
    layer = enc_cfg["layer_hidden_states_index"]
    d_model = man["model"]["hidden_size"]
    n_layers = man["model"]["num_hidden_layers"]

    plan = {
        "batch": str(args.batch_file.relative_to(ROOT)), "n": len(rows),
        "model_path": args.model_path,
        "add_special_tokens": tok_cfg["add_special_tokens"],
        "batch_size": tok_cfg["batch_size"], "padding": tok_cfg["padding"],
        "chat_template": tok_cfg["chat_template"],
        "layer_block_zero_based": enc_cfg["layer_block_zero_based"],
        "layer_hidden_states_index": layer,
        "read_position": enc_cfg["read_position"],
        "extrapolate_to": {"h1_u1_statements": n_full, "unique_texts": n_uniq},
    }
    # ── 配置闸：**本脚本只实现一种模式**，配置写别的就拒绝，不假装支持 ────────
    #    上一版把 batch=8 / 套 chat template / 读首词元写进配置，脚本照样跑完，
    #    实际仍逐条处理、读末词元 —— "配置写什么"与"实际做什么"脱钩。
    SUPPORTED = {
        "batch_size": (1, "本脚本逐条处理（batch=1）；不实现批处理"),
        "padding": (False, "batch=1 不需要 padding；若要批处理必须左 padding，未实现"),
        "chat_template": (False, "陈述是独立判真假输入；不套模板，也未实现套模板"),
    }
    bad_cfg = [f"{k}={tok_cfg.get(k)!r}（只支持 {want!r}：{why}）"
               for k, (want, why) in SUPPORTED.items() if tok_cfg.get(k) != want]
    # 读取位置：**精确白名单**，不做子串判断。
    # 上一版用 `"last" in read_pos.lower()`，于是 "not last token" 也能通过。
    READ_POSITION_ALLOWED = {
        "末词元（last token）", "末词元", "last_token", "last token",
    }
    read_pos = str(enc_cfg.get("read_position", ""))
    if read_pos not in READ_POSITION_ALLOWED:
        bad_cfg.append(
            f"read_position={read_pos!r} 不在白名单 {sorted(READ_POSITION_ALLOWED)}"
            f"（只实现末词元；**精确匹配**，不做子串判断）")

    # 层号：先自洽，再**与生产冻结的 hidden_index 精确比对**。
    # 上一版只查自洽，于是把 19→20 整个改成 0→1 照样通过 —— 那是换了读出层。
    blk = enc_cfg.get("layer_block_zero_based")
    idx = enc_cfg.get("layer_hidden_states_index")
    if not isinstance(blk, int) or not isinstance(idx, int):
        bad_cfg.append(f"层号缺失或非整数：block={blk!r} index={idx!r}")
    elif idx != blk + 1:
        bad_cfg.append(f"层号不自洽：block {blk} 应对应 hidden_states[{blk + 1}]，"
                       f"配置写的是 {idx}")
    else:
        prod_fp_path = ROOT / "outputs/stage1/production/h1_run_fingerprint.json"
        hi = (json.loads(prod_fp_path.read_text(encoding="utf-8"))
              .get("hidden_index") or {})
        want_blk = hi.get("block_index_zero_based")
        want_idx = hi.get("hidden_states_tuple_index")
        if want_blk is None or want_idx is None:
            bad_cfg.append(
                f"生产记录里缺 hidden_index（{prod_fp_path.name}）—— "
                f"无从比对冻结层，**不得当作通过**")
        elif (blk, idx) != (want_blk, want_idx):
            bad_cfg.append(
                f"读出层与**生产冻结层**不符：本次 block {blk} → hidden_states[{idx}]，"
                f"生产是 block {want_blk} → hidden_states[{want_idx}]。"
                f"自洽不等于对 —— 换层就不是同一个读出位置。")
    if bad_cfg:
        raise SystemExit(
            "[stop] 配置里有本脚本**不支持**的设置 —— 在加载权重之前拒绝：\n  "
            + "\n  ".join(bad_cfg)
            + "\n  （本脚本只实现一种模式；要改模式请改实现并重新登记，"
              "不要让配置与实际执行脱钩。）")

    print("═══ 计划")
    for k, v in plan.items():
        print(f"  {k:<28} {v}")
    print("  产物                         **不落任何隐藏向量**，只写计时 JSON")
    print("  配置闸                       已通过（batch=1 / 不 padding / 不套模板 / "
          "末词元**精确白名单** / 层号自洽**且与生产冻结层一致**）")

    if not args.i_have_authorisation:
        print("\n[dry-run] 未授权 —— **不加载权重、不前向**。")
        print("  授权后运行：")
        print(f"  {sys.executable} scripts/p2_microbench_h1.py --i-have-authorisation")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".dryrun.json").write_text(
            json.dumps({"schema": "p2_microbench_dryrun_v1", "plan": plan,
                        "loaded_weights": False, "ran_forward": False,
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[out] {args.out.with_suffix('.dryrun.json')}")
        return 0

    # ── 授权后：加载权重 → 逐条前向 → 释放 ────────────────────────────────
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT / "src"))
    from recover_attention.generation.backends import load_backend   # 复用既有加载器

    t_load = time.perf_counter()
    ctx = load_backend(args.model_path)
    torch, model, tk = ctx["torch"], ctx["model"], ctx["tokenizer"]
    load_secs = round(time.perf_counter() - t_load, 2)
    model.eval()
    dev = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── ② 模型身份绑定：与生产 run_fingerprint **逐项核对** ────────────────
    #    "29 层 / 3584 维"只是形状，**不能**确认模型身份。
    from recover_attention.generation import fingerprint as fp
    m_fp = fp.model_fingerprint(model, args.model_path, ctx.get("backend"))
    t_fp = fp.tokenizer_fingerprint(tk)
    prod = json.loads((ROOT / "outputs/stage1/production/h1_run_fingerprint.json")
                      .read_text(encoding="utf-8"))
    # ⚠️ **必需字段缺失 = 拒绝**，不是"跳过比较"。
    #    上一版写 `if want is not None and got != want` —— 参考文件里删掉这些字段，
    #    脚本照样跑完并写 all_match=True。缺字段时无从比较，就不能宣称一致。
    REQUIRED = (("model", "model_hash"), ("model", "model_type"),
                ("model", "num_hidden_layers"), ("model", "hidden_size"),
                ("tokenizer", "tokenizer_hash"), ("tokenizer", "vocab_size"))
    mine = {"model": m_fp, "tokenizer": t_fp}
    missing_ref, missing_now, mism = [], [], []
    for sect, key in REQUIRED:
        want = (prod.get(sect) or {}).get(key, "__ABSENT__")
        got = mine[sect].get(key, "__ABSENT__")
        if want == "__ABSENT__" or want is None:
            missing_ref.append(f"{sect}.{key}")
        elif got == "__ABSENT__" or got is None:
            missing_now.append(f"{sect}.{key}")
        elif got != want:
            mism.append(f"{sect}.{key}: 本次 {got!r} ≠ 生产 {want!r}")
    if missing_ref or missing_now or mism:
        raise RuntimeError(
            "身份核对未通过 —— 停，不继续：\n"
            + (f"  生产参考缺字段（无从比较，**不得当作通过**）：{missing_ref}\n"
               if missing_ref else "")
            + (f"  本次指纹缺字段：{missing_now}\n" if missing_now else "")
            + ("  不符：\n    " + "\n    ".join(mism) if mism else ""))

    # ── 当前文件摘要（**与上面的配置指纹是两回事**） ────────────────────────
    mp = Path(args.model_path)
    small = sorted(p for p in mp.glob("*")
                   if p.is_file() and p.suffix in (".json", ".txt", ".model")
                   and p.stat().st_size <= 64 * 1024 * 1024)
    file_digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                   for p in small}
    weights = sorted((p.name, p.stat().st_size) for p in mp.glob("*.safetensors"))
    if args.hash_weights:
        for n, _sz in weights:
            h = hashlib.sha256()
            with open(mp / n, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            file_digest[n] = h.hexdigest()[:16]

    identity = {
        "config_fingerprint": {
            "model": m_fp, "tokenizer": t_fp,
            "compared_against": "outputs/stage1/production/h1_run_fingerprint.json",
            "fields_compared": [f"{a}.{b}" for a, b in REQUIRED],
            "all_required_present_and_equal": True,
            "WHAT_THIS_IS": (
                "**生产配置指纹一致**，不是内容一致。"
                "`model_hash` 哈希的是 {model_path, model_type, num_hidden_layers, "
                "hidden_size, vocab_size, torch_dtype, backend}；"
                "`tokenizer_hash` 哈希的是 {class, name_or_path, vocab_size, "
                "special_token_ids}。**都不含权重或词表文件的字节**。"),
            "WHAT_THIS_IS_NOT": (
                "**不能**据此断言本次用的权重/词表与生产时逐字节相同 —— "
                "形状（29 层 / 3584 维）更不行，那只是形状。"),
        },
        "file_digest_now": {
            "model_dir": str(mp),
            "small_files": file_digest,
            "weight_shards": [{"name": n, "bytes": s} for n, s in weights],
            "weights_hashed": bool(args.hash_weights),
            "caveat": ("这是**当前**文件的摘要。生产时**没有**留下对应摘要，"
                       "因此**无法倒推**生产时的文件与现在逐字节相同。"
                       "它只对**今后**的比对有用。"),
        },
    }
    print(f"[identity] 生产**配置**指纹一致（model_hash "
          f"{str(m_fp.get('model_hash'))[:12]}… / tokenizer_hash "
          f"{str(t_fp.get('tokenizer_hash'))[:12]}…）")
    print(f"           ⚠️ 这是配置指纹，**不含权重/词表字节**；"
          f"已另记当前文件摘要 {len(file_digest)} 个小文件"
          f"{'、含权重分片' if args.hash_weights else '（权重分片只记名与大小）'}")

    n_hidden_expected = n_layers + 1
    per_rows, warm = [], 2
    with torch.no_grad():
        for i, r in enumerate(rows):
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
                raise RuntimeError(
                    f"hidden_states 长度 {len(hs)} ≠ 期望 {n_hidden_expected} —— 停")
            # ── ③ 非有限值检查：**全 tuple、全位置**，不是只看目标向量 ────
            #     上一版只检查了 hidden_states[20] 的末词元；
            #     "长度 == 29" 只是个数，不等于检查过数值。
            n_bad = 0
            for li, h in enumerate(hs):
                if not bool(torch.isfinite(h).all()):
                    n_bad += 1
                    raise RuntimeError(
                        f"{r['statement_id']}: hidden_states[{li}] 含非有限值 "
                        f"—— 停，不静默跳过（检查范围：全部 {len(hs)} 项、全部位置）")
            v = hs[layer][0, -1, :].detach().float().cpu()     # 目标：末词元
            if v.shape[0] != d_model:
                raise RuntimeError(f"维度 {v.shape[0]} ≠ {d_model} —— 停")
            t_d = time.perf_counter()
            per_rows.append({
                "i": i, "warmup": i < warm,
                "text_sha256": r["text_sha256"], "template": r["template"],
                "n_tokens": int(enc["input_ids"].shape[1]),
                "t_tokenize": round(t_b - t_a, 6),
                "t_forward": round(t_c - t_b, 6),
                "t_readout_and_check": round(t_d - t_c, 6),
                "t_total": round(t_d - t_a, 6),
                "layers_finite_checked": len(hs), "n_nonfinite_layers": n_bad,
            })
            del out, hs, v
    timed = [x for x in per_rows if not x["warmup"]]

    peak_alloc = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                  if torch.cuda.is_available() else None)
    peak_res = (round(torch.cuda.max_memory_reserved() / 1e9, 3)
                if torch.cuda.is_available() else None)
    tot = sorted(x["t_total"] for x in timed)
    fwd = sorted(x["t_forward"] for x in timed)
    med = statistics.median(tot)
    p90 = tot[int(0.9 * (len(tot) - 1))]
    bytes_one_layer = d_model * 4
    bytes_full_tuple = bytes_one_layer * n_hidden_expected

    # ── ④ 按**实际文本长度分布**外推，而不是"总体中位数 × 条数" ────────────
    #    先把全部 870 条唯一文本分词（只用分词器，不前向），拿到真实长度分布；
    #    再用实测的"长度 → 单条耗时"中位数逐条累加。
    for u in uniq_rows:
        u["n_tokens"] = len(tk(u["text"],
                               add_special_tokens=bool(
                                   tok_cfg["add_special_tokens"]))["input_ids"])
    by_len: dict[int, list[float]] = {}
    for x in timed:
        by_len.setdefault(x["n_tokens"], []).append(x["t_total"])
    len_med = {k: statistics.median(v) for k, v in sorted(by_len.items())}
    measured_lens = sorted(len_med)

    def predict(n_tok: int) -> float:
        """最近的实测长度桶（长度范围窄，不做外插拟合，免得把噪声当规律）。"""
        return len_med[min(measured_lens, key=lambda L: abs(L - n_tok))]

    corpus = {"unique_texts": 0.0, "all_records": 0.0}
    n_rec_total = 0
    len_hist_u: dict[int, int] = {}
    len_hist_r: dict[int, int] = {}
    for u in uniq_rows:
        p_ = predict(u["n_tokens"])
        corpus["unique_texts"] += p_
        corpus["all_records"] += p_ * u["n_records"]
        n_rec_total += u["n_records"]
        len_hist_u[u["n_tokens"]] = len_hist_u.get(u["n_tokens"], 0) + 1
        len_hist_r[u["n_tokens"]] = len_hist_r.get(u["n_tokens"], 0) + u["n_records"]
    covered = sum(1 for u in uniq_rows if u["n_tokens"] in len_med)
    if n_rec_total != n_full:
        raise RuntimeError(f"唯一文本的记录数合计 {n_rec_total} ≠ 陈述总数 {n_full} —— 停")

    rec = {
        "schema": "p2_microbench_v1",
        "scope": ("**只测成本**：不落隐藏向量、不训练、不打分、不产出任何估计量。"
                  "计时用于定预算，**不用于挑选任何设置或条件**。"),
        "plan": plan,
        "input_binding": input_binding,
        "identity": identity,
        "device": str(dev), "backend": ctx.get("backend"),
        "load_secs": load_secs,
        "n_timed": len(timed), "n_warmup_excluded": warm,
        "per_statement": per_rows,          # **逐条留档**，可独立复算
        "finite_check_scope": {
            "layers_checked": n_hidden_expected,
            "positions": "全部（不只末词元）",
            "what_is_read": f"hidden_states[{layer}] 的末词元",
            "note": ("上一版只检查目标向量；'长度 == 29' 只是个数，"
                     "不等于检查过数值。现在**全 tuple、全位置**都查。"),
        },
        "secs": {
            "total_per_statement": {
                "median": round(med, 5), "p90": round(p90, 5),
                "min": round(tot[0], 5), "max": round(tot[-1], 5)},
            "forward_only": {"median": round(statistics.median(fwd), 5),
                             "min": round(fwd[0], 5), "max": round(fwd[-1], 5)},
            "breakdown_note": ("t_total = 分词 + 搬设备 + 前向 + 取末词元 + "
                               "全层非有限值检查。**外推用 t_total**，"
                               "上一版只用前向那一段，偏低。"),
            "not_covered": ("权重加载（一次性 "
                            f"{load_secs}s）、向量落盘与索引写入 —— "
                            "正式提取还要付这两块，本次**没测**"),
        },
        "peak_gpu_alloc_gb": peak_alloc, "peak_gpu_reserved_gb": peak_res,
        "peak_rss_gb": _rss_gb(),
        # ⚠️ 这是**末词元向量**存成 float32 的体积，不是完整 hidden_states 张量。
        #    完整张量是 seq_len × d × (L+1)，随每条文本的词元数变化。
        "bytes_per_statement_last_token_vectors": {
            "one_layer_float32": bytes_one_layer,
            "all_layers_float32": bytes_full_tuple,
            "definition": (f"d({d_model}) × 4B × 层数；**只算每层的末词元向量**，"
                           f"不是完整 hidden_states 张量"),
            "full_tensor_would_be": (f"seq_len × {d_model} × 4B × {n_hidden_expected}"
                                     f"，随词元数变化；本次实测词元长度见 per_statement"),
        },
        "extrapolation": {
            "method": ("**按实际词元长度分布**逐条累加：把 870 条唯一文本全部分词"
                       "（只用分词器，不前向）拿到真实长度，再用实测的"
                       "『长度 → 单条 t_total 中位数』查表求和。"
                       "**不是**总体中位数 × 条数。"),
            "measured_length_buckets": {str(k): round(v, 5)
                                        for k, v in len_med.items()},
            "corpus_length_hist_unique": {str(k): v for k, v in
                                          sorted(len_hist_u.items())},
            "corpus_length_hist_records": {str(k): v for k, v in
                                           sorted(len_hist_r.items())},
            "n_unique_with_exact_measured_length": covered,
            "n_unique_total": len(uniq_rows),
            "nearest_bucket_fallback": (
                f"{len(uniq_rows) - covered} 条唯一文本的长度不在实测桶里，"
                f"取最近桶；实测覆盖长度 {measured_lens}"),
            "unique_texts_only": {
                "n": n_uniq, "secs": round(corpus["unique_texts"], 1),
                "hours": round(corpus["unique_texts"] / 3600, 3),
                "one_layer_last_token_bytes": bytes_one_layer * n_uniq,
                "note": ("按唯一文本编码再映射回全部记录 —— **省成本手段**，"
                         "不改训练行、不改权重、不改划分，"
                         "**也不修复**跨折文本重复那条限制")},
            "all_records": {
                "n": n_full, "secs": round(corpus["all_records"], 1),
                "hours": round(corpus["all_records"] / 3600, 3),
                "one_layer_last_token_bytes": bytes_one_layer * n_full},
            "naive_median_times_n_for_reference": {
                "unique_texts_hours": round(med * n_uniq / 3600, 3),
                "all_records_hours": round(med * n_full / 3600, 3),
                "why_listed": "只为与长度匹配法对照，**不作为预算依据**"},
            "remaining_uncertainty": (
                "① 未覆盖权重加载与落盘；② 长度范围很窄（实测桶见上），"
                "外推到别的长度分布无依据；③ 32 条样本的桶内中位数本身有噪声。"
                "**这是资源估算，不是保证值。**"),
            "not_extrapolated_to_mcq": ("MCQ 的陈述尚未构造，长度分布没测过，"
                                        "**不外推**"),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    e = rec["extrapolation"]
    print(f"\n加载      {load_secs}s  设备 {dev}（**不计入外推**）")
    print(f"单条耗时  t_total 中位 {med * 1000:.1f} ms  p90 {p90 * 1000:.1f} ms"
          f"；其中前向中位 {statistics.median(fwd) * 1000:.1f} ms"
          f"（{len(timed)} 条计时，前 {warm} 条预热已排除）")
    print(f"非有限值  逐条检查了全部 {n_hidden_expected} 项 hidden_states 的全部位置")
    print(f"峰值      GPU alloc {peak_alloc}G  reserved {peak_res}G  RSS {rec['peak_rss_gb']}G")
    print(f"外推      按长度分布：唯一文本 {n_uniq} 条 ≈ "
          f"{e['unique_texts_only']['hours']} h；全部记录 {n_full} 条 ≈ "
          f"{e['all_records']['hours']} h")
    print(f"          （粗估 中位×N 对照：{e['naive_median_times_n_for_reference']['unique_texts_hours']} h / "
          f"{e['naive_median_times_n_for_reference']['all_records_hours']} h）")
    print(f"          未覆盖：权重加载、向量落盘与索引写入")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
