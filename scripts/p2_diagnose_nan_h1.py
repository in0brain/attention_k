r"""P2 定位诊断：NaN 到底出现在**最后一个 block 内**还是**final norm 中**。

背景（v2 微基准的观测）：
    hidden_states[0..27]   全部有限
    hidden_states[28]      32/32 含 NaN，31/32 末词元整向量 NaN
    logits                 32/32 含 NaN

问题在于 HF 的 `Qwen2Model.forward` 里：
    all_hidden_states[k]  k=0..27  = **进入** block k 之前的张量（= block k-1 的输出）
    all_hidden_states[28]           = **norm(block 27 的输出)**
→ block 27 的**原始输出（norm 之前）从不出现在 tuple 里**。
   所以只看 hidden_states 无法区分"block 27 内部就坏了"和"norm 把它弄坏了"。

本脚本用 forward hook 把这一段拆开看：
    block 27 的**输入**    （= hidden_states[27]，对照用）
    block 27 的**输出**    ← **norm 的输入**，这是缺的那一块观测
    final norm 的**输出**  （= hidden_states[28]）
    logits

规模：**2 条**（T1073 = 末词元 NaN；T1002 = 末词元正常），各前向一次。
记录：每个张量的**逐 token** 非有限计数（NaN / +Inf / -Inf），不是只记总数。

**不做**：不修补 NaN、不裁剪、不填零、不换精度、不换后端、不换输入、不落向量、
        不训练、不生成、不重复 32 条计时、不进入正式提取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/stage1/posthoc_readout_probes/p2_prep"
OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes"
STEM = "p2_nan_diagnosis__qwen_h1"

#: 两条目标陈述（来自 v2 的实测结果，**不是**临时挑的）
TARGETS = {
    "T1073": {"why": "v2 实测：hidden_states[28] 末词元整向量 NaN",
              "statement_id": "stage1_h1_recall_attack_33c1e8cf2e3f__sampled_2::attack::T1073"},
    "T1002": {"why": "v2 实测：hidden_states[28] 末词元**正常**（full 仍有 NaN）",
              "statement_id": "stage1_h1_recall_attack_2a3469a9b974__greedy_0::attack::T1002"},
}


def _per_token(torch, t) -> dict:
    """逐 token 的非有限计数。t 形状 (1, T, X)。**只读，不改动张量。**"""
    x = t[0] if t.dim() == 3 else t
    nan = torch.isnan(x)
    pos = torch.isposinf(x)
    neg = torch.isneginf(x)
    per_nan = [int(v) for v in nan.sum(dim=-1).tolist()]
    per_pos = [int(v) for v in pos.sum(dim=-1).tolist()]
    per_neg = [int(v) for v in neg.sum(dim=-1).tolist()]
    width = int(x.shape[-1])
    # ⚠️ `tokens_clean` 必须是"NaN / +Inf / -Inf **三者都为 0**"。
    #    上一版只判 NaN == 0，于是**整位置 ±Inf** 的 token 也被算成"正常" ——
    #    那会把异常**已经出现**的模块误判为干净，正好毁掉这次诊断的目的。
    per_nonfinite = [a + b + c for a, b, c in zip(per_nan, per_pos, per_neg)]
    return {
        "shape": list(t.shape), "dtype": str(t.dtype), "width": width,
        "n_tokens": int(x.shape[0]),
        "per_token_n_nan": per_nan,
        "per_token_n_posinf": per_pos,
        "per_token_n_neginf": per_neg,
        "per_token_n_nonfinite": per_nonfinite,
        # 判"干净"用**非有限**，不是只看 NaN
        "tokens_clean": [i for i, v in enumerate(per_nonfinite) if v == 0],
        "tokens_fully_nonfinite": [i for i, v in enumerate(per_nonfinite) if v == width],
        "tokens_partially_nonfinite": [i for i, v in enumerate(per_nonfinite)
                                       if 0 < v < width],
        # NaN 专项名单另列：末词元"整向量 NaN"那条直接证据要靠它
        "tokens_fully_nan": [i for i, v in enumerate(per_nan) if v == width],
        "tokens_any_inf": [i for i, (a, b) in enumerate(zip(per_pos, per_neg))
                           if a + b > 0],
        "clean_definition": "该 token 的 n_nan + n_posinf + n_neginf 全部为 0",
        "total_n_nan": int(nan.sum().item()),
        "total_n_posinf": int(pos.sum().item()),
        "total_n_neginf": int(neg.sum().item()),
        "finite_absmax": (float(x[torch.isfinite(x)].abs().max().item())
                          if bool(torch.isfinite(x).any()) else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i-have-authorisation", action="store_true")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    t0 = time.perf_counter()

    man = json.loads((PREP / "h1_prep_manifest.json").read_text(encoding="utf-8"))
    batch = [json.loads(x) for x in
             (PREP / "h1_microbatch_32.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    by_id = {r["statement_id"]: r for r in batch}
    picked = {}
    for tag, spec in TARGETS.items():
        r = by_id.get(spec["statement_id"])
        if r is None:
            raise RuntimeError(f"{tag}: {spec['statement_id']} 不在小批名单里 —— 停")
        if hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != r["text_sha256"]:
            raise RuntimeError(f"{tag}: 文本与自带 sha256 对不上 —— 停")
        picked[tag] = r

    tok_cfg, enc_cfg = man["tokenizer_settings"], man["encoding_settings"]
    n_layers = man["model"]["num_hidden_layers"]
    last_block = n_layers - 1                       # block 27（0 基）
    layer_t = enc_cfg["layer_hidden_states_index"]  # 20

    # ── 配置闸（与 v2 同一套，加载权重之前） ───────────────────────────────
    prod = json.loads((ROOT / "outputs/stage1/production/h1_run_fingerprint.json")
                      .read_text(encoding="utf-8"))
    SUPPORTED = {"batch_size": (1, "逐条处理"), "padding": (False, "不 padding"),
                 "chat_template": (False, "不套模板")}
    bad_cfg = [f"{k}={tok_cfg.get(k)!r}（只支持 {w!r}：{why}）"
               for k, (w, why) in SUPPORTED.items() if tok_cfg.get(k) != w]
    READ_OK = {"末词元（last token）", "末词元", "last_token", "last token"}
    if str(enc_cfg.get("read_position")) not in READ_OK:
        bad_cfg.append(f"read_position={enc_cfg.get('read_position')!r} 不在白名单")
    blk = enc_cfg.get("layer_block_zero_based")
    hi = prod.get("hidden_index") or {}
    if not isinstance(blk, int) or not isinstance(layer_t, int):
        bad_cfg.append(f"层号缺失或非整数：{blk!r}/{layer_t!r}")
    elif layer_t != blk + 1:
        bad_cfg.append(f"层号不自洽：block {blk} 应对应 hidden_states[{blk + 1}]")
    elif hi.get("block_index_zero_based") is None or \
            hi.get("hidden_states_tuple_index") is None:
        bad_cfg.append("生产记录缺 hidden_index —— 无从比对冻结层，不得当作通过")
    elif (blk, layer_t) != (hi["block_index_zero_based"],
                            hi["hidden_states_tuple_index"]):
        bad_cfg.append(f"读出层与生产冻结层不符：本次 {blk}→{layer_t}，"
                       f"生产 {hi['block_index_zero_based']}→"
                       f"{hi['hidden_states_tuple_index']}")
    if bad_cfg:
        raise SystemExit("[stop] 配置不受支持 —— 加载权重之前拒绝：\n  "
                         + "\n  ".join(bad_cfg))

    plan = {
        "n_statements": len(picked), "targets": {k: v["text"] for k, v in picked.items()},
        "model_path": args.model_path,
        "add_special_tokens": tok_cfg["add_special_tokens"],
        "batch_size": 1, "padding": False, "chat_template": False,
        "last_block_index_zero_based": last_block,
        "readout_tuple_index": layer_t,
        "captures": [
            f"block {last_block} 的**输入**（= hidden_states[{last_block}]）",
            f"block {last_block} 的**输出**（= final norm 的输入，**tuple 里没有**）",
            "final norm 的**输出**（= hidden_states[28]）",
            "logits",
            f"hidden_states[{layer_t}]（读出层，对照）",
        ],
        "records": "每个张量的**逐 token** NaN / +Inf / -Inf 计数",
        "forbidden": ["不修补 NaN", "不裁剪", "不填零", "不换精度/后端/输入",
                      "不落向量", "不训练", "不生成", "不进入正式提取"],
    }
    print("═══ 定位诊断计划")
    for k, v in plan.items():
        print(f"  {k}: {v}")

    if not args.i_have_authorisation:
        print("\n[dry-run] 未授权 —— **不加载权重、不前向**。")
        (args.out_dir / f"{STEM}.dryrun.json").write_text(
            json.dumps({"schema": "p2_nan_diagnosis_dryrun", "plan": plan},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[out] {args.out_dir / f'{STEM}.dryrun.json'}")
        return 0

    rec: dict = {"schema": "p2_nan_diagnosis_v1", "plan": plan,
                 "scope": ("**只定位，不修复**。本脚本不改变任何数值、不落向量、"
                           "不改配置，也不重复 32 条计时。"),
                 "statements": {}}

    def flush(status: str, err: str | None = None) -> None:
        rec.update(status=status, error=err,
                   wall_secs=round(time.perf_counter() - t0, 2),
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

        ctx = load_backend(args.model_path)
        torch, model, tk = ctx["torch"], ctx["model"], ctx["tokenizer"]
        model.eval()
        dev = next(model.parameters()).device
        # 身份闸：**与 v2 同一套**六个必需字段（含 tokenizer），缺失即拒绝。
        # 上一版只核 model_hash，把 tokenizer 身份检查漏掉了。
        m_fp = fp.model_fingerprint(model, args.model_path, ctx.get("backend"))
        t_fp = fp.tokenizer_fingerprint(tk)
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
            "model_fingerprint": m_fp, "tokenizer_fingerprint": t_fp,
            "fields_compared": [f"{a}.{b}" for a, b in REQ],
            "all_required_present_and_equal": True,
            "note": "生产**配置**指纹一致；**不含**权重/词表字节"}
        rec["device"] = str(dev)
        rec["backend"] = ctx.get("backend")
        print(f"[identity] 配置指纹一致（{m_fp['model_hash'][:12]}…）  设备 {dev}")

        cap: dict = {}

        def hook_block(_m, inp, out):
            cap["block_in"] = (inp[0] if isinstance(inp, tuple) else inp).detach()
            cap["block_out"] = (out[0] if isinstance(out, tuple) else out).detach()

        def hook_norm(_m, inp, out):
            cap["norm_in"] = (inp[0] if isinstance(inp, tuple) else inp).detach()
            cap["norm_out"] = (out[0] if isinstance(out, tuple) else out).detach()

        h1 = model.model.layers[last_block].register_forward_hook(hook_block)
        h2 = model.model.norm.register_forward_hook(hook_norm)
        try:
            with torch.no_grad():
                for tag, r in picked.items():
                    cap.clear()
                    enc = tk(r["text"],
                             add_special_tokens=bool(tok_cfg["add_special_tokens"]),
                             return_tensors="pt")
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    out = model(**enc, output_hidden_states=True)
                    hs = out.hidden_states
                    ids = enc["input_ids"][0].tolist()
                    tensors = {
                        f"hidden_states[{layer_t}] (读出层)": hs[layer_t],
                        f"hidden_states[{last_block}] (= block {last_block} 的输入)":
                            hs[last_block],
                        f"block{last_block}.hook_input": cap["block_in"],
                        f"block{last_block}.hook_output (**norm 的输入**)":
                            cap["block_out"],
                        "final_norm.hook_input": cap["norm_in"],
                        "final_norm.hook_output": cap["norm_out"],
                        f"hidden_states[{n_layers}] (= norm 输出)": hs[n_layers],
                        "logits": out.logits,
                    }
                    rec["statements"][tag] = {
                        "statement_id": r["statement_id"], "text": r["text"],
                        "text_sha256": r["text_sha256"],
                        "n_tokens": len(ids),
                        "input_ids": ids,
                        "tokens": tk.convert_ids_to_tokens(ids),
                        "v2_observation": TARGETS[tag]["why"],
                        "tensors": {k: _per_token(torch, v) for k, v in tensors.items()},
                    }
                    # **每条完成即落盘**，失败时已获得的信息不丢
                    flush("IN_PROGRESS")
                    print(f"\n── {tag}  {r['text']}  （{len(ids)} 词元）")
                    for k, v in rec["statements"][tag]["tensors"].items():
                        print(f"   {k:<50} clean {len(v['tokens_clean'])}/{v['n_tokens']}"
                              f"  全非有限 {v['tokens_fully_nonfinite']}"
                              f"  部分 {v['tokens_partially_nonfinite']}"
                              f"  nan/{v['total_n_nan']} +inf/{v['total_n_posinf']}"
                              f" -inf/{v['total_n_neginf']}")
                    del out, hs
        finally:
            h1.remove()
            h2.remove()

        flush("COMPLETED")
        print(f"\n[out] {args.out_dir / f'{STEM}.json'}")
        return 0
    except BaseException as exc:                      # noqa: BLE001
        rec["traceback"] = traceback.format_exc()
        flush("ABORTED", f"{type(exc).__name__}: {exc}")
        print(f"\n[ABORTED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
