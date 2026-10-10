r"""P2 正式提取：Qwen/H1 第 20 层末词元向量（870 条唯一文本 → 10168 条陈述）。

口径**全部沿用已核验的** `p2_prep/h1_prep_manifest.json`，本脚本不重新裁定：
    add_special_tokens=True / batch=1 / 不 padding / 不套 chat template /
    末词元 / block 19 → hidden_states[20]

检查范围（与 v2 微基准一致，**不放宽**）：
    必须有限（违反**立即停止**）  hidden_states[0..20] + 实际保存的末词元向量
    描述性登记（**不停止**）      hidden_states[21..28] + logits

**不做**（写死，不给开关）：
    不修补 NaN、不裁剪、不填零、不丢样本、不换模型/精度/模板/读出层、
    不覆盖旧产物、不训练、不选参、不做基线对齐、不扩到其它模型或任务。

两种模式：
    --self-check              **CPU 合成**自检，不加载模型、不前向
    --i-have-authorisation    正式提取（加载权重）
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage1/posthoc_readout_probes/p2_prep"
OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes/p2_vectors"
STEM = "h1_qwen_block19_lasttoken"


def load_materials() -> dict:
    man = json.loads((PREP / "h1_prep_manifest.json").read_text(encoding="utf-8"))
    stmts = [json.loads(x) for x in
             (PREP / "h1_statements_u1.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    uniq = [json.loads(x) for x in
            (PREP / "h1_unique_texts.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    # **确定性顺序**：唯一文本按 text_sha256 升序；陈述保持准备文件的原顺序
    uniq = sorted(uniq, key=lambda u: u["text_sha256"])
    row_of = {u["text_sha256"]: i for i, u in enumerate(uniq)}
    bad = [s["statement_id"] for s in stmts if s["text_sha256"] not in row_of]
    if bad:
        raise RuntimeError(f"{len(bad)} 条陈述的文本不在唯一文本表里 —— 停：{bad[:3]}")
    n_rec = sum(u["n_records"] for u in uniq)
    if n_rec != len(stmts):
        raise RuntimeError(f"唯一文本记录数合计 {n_rec} ≠ 陈述数 {len(stmts)} —— 停")
    for u in uniq:
        h = hashlib.sha256(u["text"].encode("utf-8")).hexdigest()
        if h != u["text_sha256"]:
            raise RuntimeError(f"唯一文本 sha256 自洽失败：{u['text_sha256']} —— 停")
    return {"man": man, "stmts": stmts, "uniq": uniq, "row_of": row_of}


def build_arrays(mat: dict, vectors: np.ndarray, n_tokens: list[int]) -> dict:
    """把 (870, d) 的唯一向量 + 映射，组装成落盘用的数组集合。"""
    uniq, stmts, row_of = mat["uniq"], mat["stmts"], mat["row_of"]
    n_rep = sum(1 for k in stmts[0] if k.startswith("fold_r"))
    out = {
        "unique_text_sha256": np.asarray([u["text_sha256"] for u in uniq]),
        "unique_text": np.asarray([u["text"] for u in uniq]),
        "unique_n_tokens": np.asarray(n_tokens, dtype=np.int32),
        "unique_y": np.asarray([u["y"] for u in uniq], dtype=np.int8),
        "unique_vectors": np.ascontiguousarray(vectors, dtype=np.float32),
        "statement_id": np.asarray([s["statement_id"] for s in stmts]),
        "statement_row": np.asarray([row_of[s["text_sha256"]] for s in stmts],
                                    dtype=np.int32),
        "statement_text_sha256": np.asarray([s["text_sha256"] for s in stmts]),
        "trace_id": np.asarray([s["trace_id"] for s in stmts]),
        "group_id": np.asarray([s["group_id"] for s in stmts]),
        "y": np.asarray([s["y"] for s in stmts], dtype=np.int8),
        "template": np.asarray([s["template"] for s in stmts]),
    }
    for r in range(n_rep):
        out[f"fold_r{r}"] = np.asarray([s[f"fold_r{r}"] for s in stmts], dtype=np.int32)
    return out


def verify(mat: dict, arrays: dict, npz_path: Path, d_model: int) -> dict:
    """覆盖 / 顺序 / 维度 / 有限性 / 文件读回 —— 五项，全部不通过即失败。"""
    uniq, stmts = mat["uniq"], mat["stmts"]
    checks: list[dict] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    V = arrays["unique_vectors"]
    ck("维度 = (唯一文本数, d)", V.shape == (len(uniq), d_model), str(V.shape))
    ck("dtype = float32", V.dtype == np.float32, str(V.dtype))
    ck("唯一向量**全部有限**", bool(np.all(np.isfinite(V))),
       f"非有限 {int((~np.isfinite(V)).sum())}")
    ck("陈述数 = 准备清单", len(arrays["statement_id"]) == len(stmts), str(len(stmts)))
    ck("陈述**顺序**与准备文件逐条一致",
       list(arrays["statement_id"]) == [s["statement_id"] for s in stmts])
    ck("唯一文本顺序 = text_sha256 升序",
       list(arrays["unique_text_sha256"]) == sorted(u["text_sha256"] for u in uniq))
    rows = arrays["statement_row"]
    ck("每条陈述的行号在范围内", bool(rows.min() >= 0 and rows.max() < len(uniq)),
       f"[{int(rows.min())}, {int(rows.max())}]")
    ck("**覆盖**：870 行全部被至少一条陈述用到",
       len(np.unique(rows)) == len(uniq), f"用到 {len(np.unique(rows))}/{len(uniq)}")
    sha_by_row = {i: u["text_sha256"] for i, u in enumerate(uniq)}
    ok_map = all(sha_by_row[int(rows[i])] == stmts[i]["text_sha256"]
                 for i in range(len(stmts)))
    ck("**映射正确**：陈述的 sha256 == 其所指行的 sha256", ok_map)
    cnt = np.bincount(rows, minlength=len(uniq))
    ck("每行的引用次数 == 唯一文本表登记的 n_records",
       list(cnt) == [u["n_records"] for u in uniq])

    # 文件读回：从盘上重新读，逐元素比
    with np.load(npz_path, allow_pickle=False) as z:
        keys_ok = set(arrays).issubset(set(z.files))
        ck("落盘键齐全", keys_ok,
           f"缺 {sorted(set(arrays) - set(z.files))}" if not keys_ok else "")
        worst = 0.0
        mismatched = []
        for k, v in arrays.items():
            if k not in z.files:
                continue
            b = z[k]
            if v.dtype.kind in "US":
                if list(v) != list(b):
                    mismatched.append(k)
            else:
                dif = float(np.max(np.abs(v.astype(np.float64)
                                          - b.astype(np.float64)))) if v.size else 0.0
                worst = max(worst, dif)
                if dif != 0.0:
                    mismatched.append(f"{k}({dif:.3e})")
        ck("**文件读回**与内存逐元素相同", not mismatched,
           f"最大数值差 {worst:.3e}" + (f"；不符 {mismatched[:3]}" if mismatched else ""))
        ck("读回后向量仍全部有限", bool(np.all(np.isfinite(z["unique_vectors"]))))
    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


def cmd_self_check(d_model: int = 8) -> int:
    """**CPU 合成**自检：不加载模型、不前向。"""
    print("═══ CPU 合成自检（不加载模型、不前向）")
    mat = load_materials()
    n_u, n_s = len(mat["uniq"]), len(mat["stmts"])
    print(f"  材料：唯一文本 {n_u}  陈述 {n_s}")
    rng = np.random.default_rng(0)
    V = rng.normal(size=(n_u, d_model)).astype(np.float32)
    ntok = [len(u["text"].split()) for u in mat["uniq"]]
    arrays = build_arrays(mat, V, ntok)
    tmp = Path(os.environ.get("TEMP", ".")) / "p2_selfcheck.npz"
    np.savez_compressed(tmp, **arrays)
    res = verify(mat, arrays, tmp, d_model)
    for c in res["checks"]:
        print(f"  [{'ok  ' if c['ok'] else 'FAIL'}] {c['check']}"
              + (f"  {c['detail']}" if c["detail"] else ""))

    # 反例：植入非有限值 / 破坏映射，验收必须失败
    print("\n  ── 反例（验收必须失败）")
    V2 = V.copy()
    V2[3, 0] = np.nan
    a2 = build_arrays(mat, V2, ntok)
    np.savez_compressed(tmp, **a2)
    r2 = verify(mat, a2, tmp, d_model)
    fin = [c for c in r2["checks"] if "全部有限" in c["check"]][0]
    print(f"  [{'ok  ' if not fin['ok'] else 'FAIL'}] 植入 1 个 NaN → 有限性检查失败"
          f"  {fin['detail']}")
    a3 = build_arrays(mat, V, ntok)
    a3["statement_row"] = a3["statement_row"].copy()
    a3["statement_row"][0] = (int(a3["statement_row"][0]) + 1) % n_u
    np.savez_compressed(tmp, **a3)
    r3 = verify(mat, a3, tmp, d_model)
    mp = [c for c in r3["checks"] if "映射正确" in c["check"]][0]
    print(f"  [{'ok  ' if not mp['ok'] else 'FAIL'}] 改一条陈述的行号 → 映射检查失败")
    a4 = build_arrays(mat, V, ntok)
    a4["unique_vectors"] = a4["unique_vectors"][:, :d_model - 1]
    np.savez_compressed(tmp, **a4)
    r4 = verify(mat, a4, tmp, d_model)
    dm = [c for c in r4["checks"] if "维度" in c["check"]][0]
    print(f"  [{'ok  ' if not dm['ok'] else 'FAIL'}] 砍掉一维 → 维度检查失败  {dm['detail']}")
    tmp.unlink(missing_ok=True)

    neg_ok = (not fin["ok"]) and (not mp["ok"]) and (not dm["ok"])
    print(f"\n[{'PASS' if res['all_ok'] and neg_ok else 'FAIL'}] "
          f"正例 {sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}；"
          f"三个反例{'都按预期失败' if neg_ok else '**有漏网**'}")
    return 0 if (res["all_ok"] and neg_ok) else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--i-have-authorisation", action="store_true")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    if args.self_check:
        return cmd_self_check()

    t0 = time.perf_counter()
    mat = load_materials()
    man, uniq, stmts = mat["man"], mat["uniq"], mat["stmts"]
    tok_cfg, enc_cfg = man["tokenizer_settings"], man["encoding_settings"]
    d_model, n_layers = man["model"]["hidden_size"], man["model"]["num_hidden_layers"]
    layer = enc_cfg["layer_hidden_states_index"]
    n_hidden_expected = n_layers + 1
    prod = json.loads((ROOT / "outputs/stage1/production/h1_run_fingerprint.json")
                      .read_text(encoding="utf-8"))

    # ── 配置闸（加载权重之前） ─────────────────────────────────────────────
    SUPPORTED = {"batch_size": (1, "逐条处理"), "padding": (False, "不 padding"),
                 "chat_template": (False, "不套模板")}
    bad_cfg = [f"{k}={tok_cfg.get(k)!r}（只支持 {w!r}：{why}）"
               for k, (w, why) in SUPPORTED.items() if tok_cfg.get(k) != w]
    if str(enc_cfg.get("read_position")) not in {
            "末词元（last token）", "末词元", "last_token", "last token"}:
        bad_cfg.append(f"read_position={enc_cfg.get('read_position')!r} 不在白名单")
    blk, hi = enc_cfg.get("layer_block_zero_based"), prod.get("hidden_index") or {}
    if not isinstance(blk, int) or not isinstance(layer, int):
        bad_cfg.append(f"层号缺失或非整数：{blk!r}/{layer!r}")
    elif layer != blk + 1:
        bad_cfg.append(f"层号不自洽：block {blk} 应对应 hidden_states[{blk + 1}]")
    elif hi.get("block_index_zero_based") is None or \
            hi.get("hidden_states_tuple_index") is None:
        bad_cfg.append("生产记录缺 hidden_index —— 不得当作通过")
    elif (blk, layer) != (hi["block_index_zero_based"], hi["hidden_states_tuple_index"]):
        bad_cfg.append(f"读出层与生产冻结层不符：{blk}→{layer} vs "
                       f"{hi['block_index_zero_based']}→{hi['hidden_states_tuple_index']}")
    if bad_cfg:
        raise SystemExit("[stop] 配置不受支持 —— 加载权重之前拒绝：\n  "
                         + "\n  ".join(bad_cfg))

    plan = {
        "n_unique_texts": len(uniq), "n_statements": len(stmts),
        "model_path": args.model_path, "d_model": d_model,
        "layer_block_zero_based": blk, "layer_hidden_states_index": layer,
        "read_position": enc_cfg["read_position"],
        "add_special_tokens": tok_cfg["add_special_tokens"],
        "batch_size": 1, "padding": False, "chat_template": False,
        "must_be_finite_abort": f"hidden_states[0..{layer}] + 保存的末词元向量",
        "descriptive_only": f"hidden_states[{layer + 1}..{n_layers}] + logits",
        "forbidden": ["不修补 NaN", "不裁剪", "不填零", "不丢样本",
                      "不换模型/精度/模板/读出层", "不覆盖旧产物",
                      "不训练", "不选参", "不做基线对齐"],
    }
    print("═══ 正式提取计划")
    for k, v in plan.items():
        print(f"  {k}: {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{STEM}.npz"
    if npz_path.exists():
        raise SystemExit(f"[stop] {npz_path.name} 已存在 —— **不覆盖旧产物**。"
                         f"要重跑请先改 --out-dir。")

    if not args.i_have_authorisation:
        print("\n[dry-run] 未授权 —— **不加载权重、不前向**。")
        return 0

    # ── preflight：配置 + 文件摘要，**前向之前**落盘 ───────────────────────
    mp = Path(args.model_path)
    small = sorted(p for p in mp.glob("*") if p.is_file()
                   and p.suffix in (".json", ".txt", ".model")
                   and p.stat().st_size <= 64 * 1024 * 1024)
    preflight = {
        "schema": "p2_extract_preflight_v1", "plan": plan,
        "input_binding": {
            "prep_manifest_sha256": hashlib.sha256(
                (PREP / "h1_prep_manifest.json").read_bytes()).hexdigest()[:16],
            "statements_sha256": hashlib.sha256(
                (PREP / "h1_statements_u1.jsonl").read_bytes()).hexdigest()[:16],
            "unique_texts_sha256": hashlib.sha256(
                (PREP / "h1_unique_texts.jsonl").read_bytes()).hexdigest()[:16]},
        "file_digest_now": {
            "model_dir": str(mp),
            "small_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                            for p in small},
            "weight_shards": [{"name": p.name, "bytes": p.stat().st_size}
                              for p in sorted(mp.glob("*.safetensors"))],
            "caveat": "**当前**文件摘要；生产时没留对应摘要，无法倒推逐字节相同"},
        "written_before_forward": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out_dir / f"{STEM}.preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[preflight] 已在前向**之前**落盘")

    rec: dict = {"schema": "p2_extract_v1", "plan": plan, "preflight": preflight}
    anomalies: list[dict] = []

    def flush(status: str, err: str | None = None, extra: dict | None = None) -> None:
        rec.update(status=status, error=err, anomaly_summary_rows=len(anomalies),
                   wall_secs=round(time.perf_counter() - t0, 2),
                   generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {}))
        (args.out_dir / f"{STEM}.report.json").write_text(
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

        m_fp, t_fp = (fp.model_fingerprint(model, args.model_path, ctx.get("backend")),
                      fp.tokenizer_fingerprint(tk))
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
                mism.append(f"{sect}.{key}")
        if miss_ref or miss_now or mism:
            raise RuntimeError(f"身份核对未通过 —— 停：参考缺 {miss_ref}；"
                               f"本次缺 {miss_now}；不符 {mism}")
        rec["identity"] = {"model_fingerprint": m_fp, "tokenizer_fingerprint": t_fp,
                           "all_required_present_and_equal": True,
                           "note": "生产**配置**指纹一致；**不含**权重/词表字节"}
        rec["device"], rec["backend"], rec["load_secs"] = str(dev), ctx.get("backend"), load_secs
        print(f"[identity] 配置指纹一致（{m_fp['model_hash'][:12]}…）  设备 {dev}")

        V = np.empty((len(uniq), d_model), dtype=np.float32)
        ntok: list[int] = []
        secs: list[float] = []
        t_enc0 = time.perf_counter()
        with torch.no_grad():
            for i, u in enumerate(uniq):
                ta = time.perf_counter()
                enc = tk(u["text"],
                         add_special_tokens=bool(tok_cfg["add_special_tokens"]),
                         return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states
                if len(hs) != n_hidden_expected:
                    raise RuntimeError(f"hidden_states 长度 {len(hs)} ≠ {n_hidden_expected}")
                # 必查：0..layer 全部 + 保存的向量
                for li in range(layer + 1):
                    if not bool(torch.isfinite(hs[li]).all()):
                        raise RuntimeError(
                            f"{u['text_sha256']}: hidden_states[{li}] 含非有限值 —— 停"
                            f"（必查范围 0..{layer}）")
                v = hs[layer][0, -1, :]
                if not bool(torch.isfinite(v).all()):
                    raise RuntimeError(f"{u['text_sha256']}: 保存的末词元向量含非有限值 —— 停")
                if int(v.shape[0]) != d_model:
                    raise RuntimeError(f"维度 {int(v.shape[0])} ≠ {d_model} —— 停")
                V[i] = v.detach().float().cpu().numpy()
                ntok.append(int(enc["input_ids"].shape[1]))

                # 描述性登记（**不停止**）
                row = {"row": i, "text_sha256": u["text_sha256"],
                       "n_tokens": ntok[-1], "downstream": {}}
                any_bad = False
                for li in range(layer + 1, n_hidden_expected):
                    h = hs[li][0]
                    nf = (~torch.isfinite(h)).sum(dim=-1)
                    n_bad = int((nf > 0).sum().item())
                    if n_bad:
                        any_bad = True
                        row["downstream"][str(li)] = {
                            "n_tokens_with_nonfinite": n_bad,
                            "n_tokens_fully_nonfinite": int((nf == h.shape[-1]).sum().item()),
                            "last_token_nonfinite": bool(nf[-1].item() > 0),
                            "total_n_nan": int(torch.isnan(h).sum().item()),
                            "total_n_posinf": int(torch.isposinf(h).sum().item()),
                            "total_n_neginf": int(torch.isneginf(h).sum().item())}
                lg = out.logits[0]
                nfl = (~torch.isfinite(lg)).sum(dim=-1)
                if int((nfl > 0).sum().item()):
                    any_bad = True
                    row["logits"] = {
                        "n_tokens_with_nonfinite": int((nfl > 0).sum().item()),
                        "n_tokens_fully_nonfinite": int((nfl == lg.shape[-1]).sum().item()),
                        "last_token_nonfinite": bool(nfl[-1].item() > 0),
                        "total_n_nan": int(torch.isnan(lg).sum().item())}
                if any_bad:
                    anomalies.append(row)
                secs.append(time.perf_counter() - ta)
                del out, hs, v
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(uniq)}  "
                          f"中位 {statistics.median(secs) * 1000:.0f} ms  "
                          f"异常登记 {len(anomalies)}", flush=True)
        enc_secs = round(time.perf_counter() - t_enc0, 2)

        arrays = build_arrays(mat, V, ntok)
        np.savez_compressed(npz_path, **arrays)
        ver = verify(mat, arrays, npz_path, d_model)

        # 异常汇总
        lay_hist: dict[str, int] = {}
        for a in anomalies:
            for li in a["downstream"]:
                lay_hist[li] = lay_hist.get(li, 0) + 1
        rec.update({
            "n_unique_encoded": len(uniq), "n_statements_mapped": len(stmts),
            "encode_secs_total": enc_secs,
            "encode_secs_per_text": {"median": round(statistics.median(secs), 5),
                                     "min": round(min(secs), 5),
                                     "max": round(max(secs), 5)},
            "peak_gpu_alloc_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                  if torch.cuda.is_available() else None),
            "peak_gpu_reserved_gb": (round(torch.cuda.max_memory_reserved() / 1e9, 3)
                                     if torch.cuda.is_available() else None),
            "token_len": {"min": min(ntok), "median": statistics.median(ntok),
                          "max": max(ntok)},
            "verification": ver,
            "anomaly_summary": {
                "policy": ("下游层与 logits 的异常**只登记不停止**；"
                           "必查范围（0..20 + 保存的向量）**零异常**才会走到这里"),
                "n_texts_with_downstream_anomaly": len(anomalies),
                "n_texts_total": len(uniq),
                "per_layer_n_texts": lay_hist,
                "n_texts_with_logits_anomaly": sum(1 for a in anomalies if "logits" in a),
                "must_not_claim": ("**不得**据此宣布后端数值健康，也**不得**解除既有 "
                                   "`h1_b4_numeric_integrity_compromised` 限制"),
            },
            "artifacts": {
                "vectors_npz": str(npz_path.relative_to(ROOT)),
                "report": f"{STEM}.report.json",
                "anomalies": f"{STEM}.anomalies.jsonl",
                "preflight": f"{STEM}.preflight.json"},
        })
        (args.out_dir / f"{STEM}.anomalies.jsonl").write_text(
            "\n".join(json.dumps(a, ensure_ascii=False) for a in anomalies) + "\n",
            encoding="utf-8")
        flush("COMPLETED" if ver["all_ok"] else "VERIFICATION_FAILED")
        print(f"\n编码      {len(uniq)} 条  用时 {enc_secs}s  "
              f"中位 {statistics.median(secs) * 1000:.1f} ms")
        print(f"峰值      GPU alloc {rec['peak_gpu_alloc_gb']}G  "
              f"reserved {rec['peak_gpu_reserved_gb']}G")
        print(f"验收      {sum(c['ok'] for c in ver['checks'])}/{len(ver['checks'])} "
              f"{'全部通过' if ver['all_ok'] else '**有失败**'}")
        for c in ver["checks"]:
            print(f"  [{'ok  ' if c['ok'] else 'FAIL'}] {c['check']}"
                  + (f"  {c['detail']}" if c["detail"] else ""))
        print(f"下游异常  {len(anomalies)}/{len(uniq)} 条文本；逐层 {lay_hist}；"
              f"logits {rec['anomaly_summary']['n_texts_with_logits_anomaly']}")
        print(f"[out] {npz_path}")
        return 0 if ver["all_ok"] else 2
    except BaseException as exc:                      # noqa: BLE001
        rec["traceback"] = traceback.format_exc()
        flush("ABORTED", f"{type(exc).__name__}: {exc}")
        if anomalies:
            (args.out_dir / f"{STEM}.anomalies.jsonl").write_text(
                "\n".join(json.dumps(a, ensure_ascii=False) for a in anomalies) + "\n",
                encoding="utf-8")
        print(f"\n[ABORTED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
