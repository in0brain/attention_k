"""Stage C：MCQ 的 teacher-forced hidden re-forward。

设计权威:preregistration v2.3 §8 —— "MCQ = answer-letter token 自身在该层的 hidden"。

只做确定性前向,**不重新采样**:复用 Stage B 记录里 4B-3 已生成的 rendered_prompt 与
raw_completion,teacher-force 一遍,取 answer-letter token 在 hidden_states[⌊0.7L⌋+1] 的向量。

只缓存:answer-letter token 位置、该位置的 hidden、必要 token logprobs。
不缓存:raw attention / 全层 hidden / F1 / F4 / J-lens。

fail-fast(全部硬检查,任一不过即停):
  1 复用 exact rendered prompt(sha256 比对 Stage B 记录)
  2 completion span 由 **offset** 确定,不用 prompt token 数硬推(BPE 边界会变分词)
  3 answer-letter 的 token span 非空
  4 该 span 恰好一个 token —— **实际验证**,不是假设;多 token 则停,不临时选"最后一个"
  5 block index 19 / hidden tuple index 20
  6 forward hook 输出与 hidden_states[20] 数值相等
  7 落盘 tokenizer/model/config fingerprint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention import h1_f5_features as f5f  # noqa: E402
from recover_attention import mcq_conditional_increment as mci  # noqa: E402
from recover_attention.data_io import read_jsonl, write_json, write_jsonl  # noqa: E402
from recover_attention.hidden_state_cache import (  # noqa: E402
    import_torch,
    import_transformers_components,
    infer_model_input_device,
    resolve_torch_dtype,
)

SCHEMA_VERSION = "4D2_mcq_hidden_cache_v1"


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_backend(model_path: str) -> dict:
    """复用 4D-1/W0.5-B 已验证的 8-bit 本地路径（与将来正式缓存同一条加载路径）。"""
    torch = import_torch()
    comp = import_transformers_components()
    tok = comp["AutoTokenizer"].from_pretrained(model_path, local_files_only=True,
                                                trust_remote_code=False)
    qcfg = comp["BitsAndBytesConfig"](load_in_8bit=True)
    model = comp["AutoModelForCausalLM"].from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        quantization_config=qcfg, device_map="auto", attn_implementation="eager",
        torch_dtype=resolve_torch_dtype("float16", torch))
    model.eval()
    return {"torch": torch, "tokenizer": tok, "model": model, "model_path": model_path,
            "backend": {"load_in_8bit": True, "device_map": "auto",
                        "attn_implementation": "eager", "local_files_only": True}}


def fingerprint(ctx: dict) -> dict:
    """(7) tokenizer/model/config fingerprint —— 换了权重/分词器,缓存必须作废。"""
    tok, model = ctx["tokenizer"], ctx["model"]
    cfg = model.config
    return {
        "model_path": ctx["model_path"],
        "model_type": getattr(cfg, "model_type", None),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(getattr(cfg, "vocab_size", -1)),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        "tokenizer_class": type(tok).__name__,
        "tokenizer_vocab_size": int(tok.vocab_size),
        "tokenizer_name_or_path": str(getattr(tok, "name_or_path", "")),
        "backend": ctx["backend"],
    }


def locate_letter_char_span(raw_completion: str, parsed_label: str, bare_answer: bool) -> tuple[int, int]:
    """answer letter 在 raw_completion 中的字符 span。

    只处理 bare_label（实测 239/240）。非裸格式**不在此临时发明定位规则**:
    §8 冻结的是"answer-letter token 自身",定位方式没冻结到包裹格式;真遇到就停下来核对,
    不猜。（唯一的非裸样本 'Answer! <D>' 本就是上游 parse_failure,不进 population。）
    """
    if not bare_answer:
        raise SystemExit(
            f"STOP: non-bare answer format with a successful upstream parse "
            f"(completion={raw_completion!r}, parsed={parsed_label!r}). "
            f"§8 freezes 'the answer-letter token itself' but does not freeze how to locate it "
            f"inside a wrapped format. Do not invent a rule here — check the frozen definition.")
    stripped = raw_completion.strip()
    if stripped.upper() != parsed_label.strip().upper():
        raise SystemExit(f"STOP: bare_answer=True but completion {raw_completion!r} != "
                         f"parsed_label {parsed_label!r}")
    start = raw_completion.index(stripped)
    return start, start + len(stripped)


def reforward(ctx: dict, rec: dict, tuple_idx: int) -> dict:
    """一次 teacher-forced 前向。"""
    torch = ctx["torch"]; tok = ctx["tokenizer"]; model = ctx["model"]
    prompt, raw = rec["rendered_prompt"], rec["raw_completion"]

    # (1) 复用 exact rendered prompt
    if _sha(prompt) != rec["rendered_prompt_sha256"]:
        raise SystemExit(f"STOP: rendered prompt changed for {rec['example_id']}")

    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    # (2) completion span 由 offset 确定,不用 prompt token 数硬推
    enc = tok(raw, return_offsets_mapping=True, add_special_tokens=False)
    c_ids, offsets = enc["input_ids"], [list(o) for o in enc["offset_mapping"]]
    if not c_ids:
        raise SystemExit(f"STOP: empty completion tokenization for {rec['example_id']}")

    cs, ce = locate_letter_char_span(raw, rec["parsed_label"], rec["bare_answer"])
    local = f5f.token_indices_for_char_span(offsets, cs, ce)
    # (3) span 非空
    if not local:
        raise SystemExit(f"STOP: answer-letter char span [{cs},{ce}) maps to no token "
                         f"for {rec['example_id']} (offsets={offsets})")
    # (4) 恰好一个 token —— 实际验证,不假设;多 token 不临时选"最后一个"
    if len(local) != 1:
        raise SystemExit(
            f"STOP: answer letter {rec['parsed_label']!r} spans {len(local)} tokens "
            f"({[tok.convert_ids_to_tokens(c_ids[i]) for i in local]}) for {rec['example_id']}. "
            f"§8 says 'the answer-letter token itself' (singular). Do NOT pick the last token — "
            f"check the frozen definition first.")

    full = torch.tensor([p_ids + c_ids])
    device = infer_model_input_device(model, "auto", torch)
    inputs = {"input_ids": full.to(device),
              "attention_mask": torch.ones_like(full).to(device)}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    prompt_len = len(p_ids)

    j = local[0]
    global_pos = prompt_len + j
    step = global_pos - 1                     # 预测该 token 的那一步
    letter_lp = float(logprobs[step, c_ids[j]])
    h = out.hidden_states[tuple_idx][0, global_pos].detach().float().cpu().numpy()  # FP32 落盘

    comp_lps = [float(logprobs[prompt_len + i - 1, c_ids[i]]) for i in range(len(c_ids))]
    return {
        "letter_token_index_local": int(j),
        "letter_token_index_global": int(global_pos),
        "letter_token_id": int(c_ids[j]),
        "letter_token_str": tok.convert_ids_to_tokens(c_ids[j]),
        "letter_token_count": len(local),
        "f5_letter_token_logprob": letter_lp,
        "completion_token_logprobs": comp_lps,
        "prompt_len": prompt_len,
        "seq_len": int(full.shape[-1]),
        "hidden": h,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records",
                    default="outputs/logs/sprint_4D_2_mcq_v2_3/mcq_pilot_completion_records.jsonl")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--output-dir", default="outputs/logs/sprint_4D_2_mcq_v2_3")
    args = ap.parse_args()

    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit("STOP (G2): preregistration hash mismatch")

    out = Path(args.output_dir)
    recs = read_jsonl(Path(args.records))
    eligible = [r for r in recs if r.get("eligible_for_primary")]
    for r in eligible:
        r["rendered_prompt_sha256"] = _sha(r["rendered_prompt"])
    print(f"[stage-c] {len(eligible)}/{len(recs)} eligible records to re-forward")

    ctx = load_backend(args.model_path)
    fp = fingerprint(ctx)
    L = fp["num_hidden_layers"]
    idx = ci.resolve_hidden_index(L)
    block, tuple_idx = idx["block_index_zero_based"], idx["hidden_states_tuple_index"]
    # (5) block index 19 / tuple index 20（Qwen L=28）

    # (6) hook 输出与 hidden_states[tuple_idx] 数值相等
    torch = ctx["torch"]
    captured: dict[str, Any] = {}
    layer = ctx["model"].model.layers[block]
    handle = layer.register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o)
                                            .detach().float().cpu()))
    probe = ctx["tokenizer"](eligible[0]["rendered_prompt"], return_tensors="pt")
    device = infer_model_input_device(ctx["model"], "auto", torch)
    with torch.no_grad():
        o = ctx["model"](**{k: v.to(device) for k, v in probe.items()}, output_hidden_states=True)
    handle.remove()
    hs = o.hidden_states[tuple_idx].detach().float().cpu()
    max_abs = float((captured["h"] - hs).abs().max())
    numeric_match = bool(torch.allclose(captured["h"], hs, atol=1e-3, rtol=1e-3))
    verify = {"num_layers": L, "block_index_zero_based": block,
              "hidden_states_tuple_index": tuple_idx, "numeric_match": numeric_match,
              "max_abs_diff": max_abs}
    print(f"[stage-c] hidden verify: L={L} block={block} tuple={tuple_idx} "
          f"match={numeric_match} max_abs_diff={max_abs:.2e}")
    if not numeric_match:
        raise SystemExit("STOP: hook vs hidden_states mismatch (off-by-one?)")

    rows, vecs = [], {}
    token_strs, token_counts = [], []
    for i, r in enumerate(eligible, 1):
        res = reforward(ctx, r, tuple_idx)
        vecs[r["example_id"]] = res.pop("hidden")
        rows.append({"schema_version": SCHEMA_VERSION, "example_id": r["example_id"],
                     "group_id": r["group_id"], **res})
        token_strs.append(res["letter_token_str"])
        token_counts.append(res["letter_token_count"])
        print(f"[stage-c] {i}/{len(eligible)} {r['example_id']} "
              f"letter={r['parsed_label']} tok={res['letter_token_str']!r} "
              f"pos={res['letter_token_index_global']} lp={res['f5_letter_token_logprob']:.3f}")

    write_jsonl(rows, out / "mcq_hidden_cache_records.jsonl")
    np.savez_compressed(out / "mcq_hidden.npz", **vecs)
    dims = {int(v.shape[0]) for v in vecs.values()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "prereg_sha256": g2["current"],
        "n_reforwarded": len(rows),
        "hidden_verification": verify,
        "fingerprint": fp,
        "hidden_dim": sorted(dims),
        "letter_token_forms": dict(__import__("collections").Counter(token_strs)),
        "letter_token_count_distinct": sorted(set(token_counts)),
        "checks": {
            "all_single_token": set(token_counts) == {1},
            "hidden_dim_consistent": len(dims) == 1,
            "all_eligible_reforwarded": len(rows) == len(eligible),
            "hidden_index_numeric_match": numeric_match,
        },
        "not_cached": ["raw_attention", "all_layer_hidden", "F1", "F4", "J-lens"],
        "population_role": "pilot_smoke",
    }
    write_json(report, out / "mcq_hidden_reforward_report.json")
    print(f"[stage-c] done: {len(rows)} vectors dim={sorted(dims)} "
          f"token_forms={report['letter_token_forms']} -> {out}")


if __name__ == "__main__":
    main()
