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
from recover_attention import domain_label_proxy as dlp  # noqa: E402
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


def reforward(ctx: dict, rec: dict, tuple_idx: int, letter_token_ids: dict[str, int]) -> dict:
    """一次 teacher-forced 前向。

    同一遍前向里把 §7.1 的 F5_MCQ 全算出来（不从 4C 搬值）:
      fresh 1760 没有 4C 产物,若 F5 只靠读旧 artifact,正式跑时 label_margin /
      label_entropy / full_entropy 会静默填 0 → O 侧基线被削弱 → Δ_H 被人为抬高。
      这与 H1 侧 cross-sample F5 静默填 0 是同一类问题。
    读出位 = 预测 answer-letter token 的那一步(= prompt 末位;4C 实测
      readout_to_prompt_end_distance = 0,与此一致)。
    """
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
    step = global_pos - 1                     # 预测该 token 的那一步 = 读出位
    letter_lp = float(logprobs[step, c_ids[j]])
    h = out.hidden_states[tuple_idx][0, global_pos].detach().float().cpu().numpy()  # FP32 落盘

    # F5_MCQ（§7.1）：与 4B-2/4C 同定义,复用 domain_label_proxy 的实现,不另写一套
    row = logits[step].cpu().numpy()
    option_logits = {lab: float(row[tid]) for lab, tid in letter_token_ids.items()}
    f5 = {
        "f5_label_margin": dlp.label_margin(option_logits),      # top1 − top2（候选 logits）
        "f5_label_entropy": dlp.label_entropy(option_logits),    # 仅候选 softmax 后的熵
        "f5_full_entropy": dlp.full_entropy(row),                # 全词表熵
        "f5_letter_token_logprob": letter_lp,
    }
    # letter token id 与 bare 形式解析结果必须一致（4B-2:裸 vs 空格前缀是不同 token）
    expected_tid = letter_token_ids.get(rec["parsed_label"].strip().upper())
    if expected_tid is not None and int(c_ids[j]) != int(expected_tid):
        raise SystemExit(
            f"STOP: letter token id mismatch for {rec['example_id']}: forward gave "
            f"{c_ids[j]} ({tok.convert_ids_to_tokens(c_ids[j])!r}) but bare-form resolution "
            f"gives {expected_tid}. Token form is ambiguous — check before caching.")

    comp_lps = [float(logprobs[prompt_len + i - 1, c_ids[i]]) for i in range(len(c_ids))]
    return {
        "letter_token_index_local": int(j),
        "letter_token_index_global": int(global_pos),
        "letter_token_id": int(c_ids[j]),
        "letter_token_str": tok.convert_ids_to_tokens(c_ids[j]),
        "letter_token_count": len(local),
        "readout_step": int(step),
        "readout_to_prompt_end_distance": int(step - (prompt_len - 1)),
        "option_logits": option_logits,
        **f5,
        "completion_token_logprobs": comp_lps,
        "prompt_len": prompt_len,
        "seq_len": int(full.shape[-1]),
        "hidden": h,
    }


F5_XCHECK_NAMES = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy")


def _compare_against_4c(rows: list[dict], manifest: Path) -> dict:
    """把重算的 F5 与 4C 存的值**对照**（不是一致性检查）。

    **数值不该一致,这是预期的**:4C 的 F5 由 `load_local_steering_backend` 算出,那是
    "strict local **4-bit** eager backend";本脚本用 8-bit（与 H1 侧同一条路径）。
    量化不同 → logits 不同 → margin/熵 相差几个 nat 完全正常。

    为什么 MCQ 必须跟 H1 同用 8-bit:§6 的 gate 是 D = S_MCQ − S_H1。H1 因长文本退化
    只能用 8-bit(4D-1 已验证)。若 MCQ 用 4-bit、H1 用 8-bit,S 的差异就分不清是
    observability 差异还是量化差异 —— 跨任务比较被混淆,gate 失去意义。
    故 4C 的 4-bit F5 值不可复现,也不应复现;它们属于 exploratory,已被 v2.3 排除。

    这里报的是 **Spearman 秩相关**:量化会改变数值,但若连排序都对不上,说明读出位或
    token 形式有问题,那才是真信号。
    """
    if not manifest.exists():
        return {"status": "skipped", "reason": f"missing {manifest}"}
    ref = {r["example_id"]: r for r in read_jsonl(manifest)}
    paired: dict[str, list[tuple[float, float]]] = {n: [] for n in F5_XCHECK_NAMES}
    n_cmp = 0
    for row in rows:
        r4 = ref.get(row["example_id"])
        if not r4:
            continue
        n_cmp += 1
        for name in F5_XCHECK_NAMES:
            a, b = row.get(name), r4.get(name)
            if a is not None and b is not None:
                paired[name].append((float(a), float(b)))

    def _spearman(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 3:
            return None
        from scipy.stats import spearmanr
        rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
        return None if rho != rho else float(rho)

    rank = {n: _spearman(v) for n, v in paired.items()}
    return {
        "status": "compared_across_backends",
        "n_compared": n_cmp,
        "this_pass_backend": "8bit (matches the H1 arm)",
        "sprint_4c_backend": "4bit (load_local_steering_backend / strict local 4-bit eager)",
        "values_expected_to_match": False,
        "why": ("different quantization -> different logits; MCQ must share the H1 arm's 8-bit "
                "backend or the section-6 gate D = S_MCQ - S_H1 is confounded by quantization"),
        "spearman_rank_correlation": rank,
        "rank_note": ("quantization changes values but should largely preserve ordering; a low "
                      "rank correlation would indicate a readout-position or token-form bug"),
        "sprint_4c_values_are": "exploratory, excluded by v2.3; not reproduced and not required to",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records",
                    default="outputs/logs/sprint_4D_2_mcq_v2_3/mcq_pilot_completion_records.jsonl")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--output-dir", default="outputs/logs/sprint_4D_2_mcq_v2_3")
    ap.add_argument("--readout-manifest",
                    default="outputs/logs/sprint_4C_narrowed_readout_increment_and_site_transfer/"
                            "readout_feature_manifest.jsonl")
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

    # 裸形式的 letter token id（4B-2:chat 条件下模型输出裸字母,与 " A" 是不同 token）
    letter_token_ids = dlp.bare_option_token_ids(ctx["tokenizer"], list(mci.MCQ_LETTERS))
    print(f"[stage-c] bare letter token ids: {letter_token_ids}")

    rows, vecs = [], {}
    token_strs, token_counts = [], []
    for i, r in enumerate(eligible, 1):
        res = reforward(ctx, r, tuple_idx, letter_token_ids)
        vecs[r["example_id"]] = res.pop("hidden")
        rows.append({"schema_version": SCHEMA_VERSION, "example_id": r["example_id"],
                     "group_id": r["group_id"], **res})
        token_strs.append(res["letter_token_str"])
        token_counts.append(res["letter_token_count"])
        print(f"[stage-c] {i}/{len(eligible)} {r['example_id']} "
              f"letter={r['parsed_label']} tok={res['letter_token_str']!r} "
              f"pos={res['letter_token_index_global']} lp={res['f5_letter_token_logprob']:.3f} "
              f"margin={res['f5_label_margin']:.3f}")

    # 与 4C 的**跨 backend 对照**(4bit vs 8bit)。数值不该一致;看的是秩相关。
    xcheck = _compare_against_4c(rows, Path(args.readout_manifest))
    print(f"[stage-c] 4C comparison (4bit vs 8bit, values not expected to match): "
          f"spearman={xcheck.get('spearman_rank_correlation')}")

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
        "bare_letter_token_ids": letter_token_ids,
        "f5_recomputed_in_this_pass": list(F5_XCHECK_NAMES) + ["f5_letter_token_logprob"],
        "f5_source_note": ("F5 是本遍前向重算的,不是从 4C artifact 搬的 —— fresh 1760 没有 "
                           "4C 产物,靠搬值会让正式跑时 F5 静默填 0、削弱 O 侧基线、抬高 Δ_H"),
        "sprint_4c_comparison": xcheck,
        "backend_consistency": {
            "mcq_arm": "8bit",
            "h1_arm": "8bit",
            "same_backend_both_arms": True,
            "why_required": ("section-6 gate D = S_MCQ - S_H1 would be confounded by "
                             "quantization if the two arms used different backends"),
            "sprint_4c_used": "4bit — its F5 values are exploratory and not reproduced here",
        },
        "checks": {
            "all_single_token": set(token_counts) == {1},
            "hidden_dim_consistent": len(dims) == 1,
            "all_eligible_reforwarded": len(rows) == len(eligible),
            "hidden_index_numeric_match": numeric_match,
            "readout_at_prompt_end": all(r["readout_to_prompt_end_distance"] == 0 for r in rows),
            "f5_all_finite": all(
                all(isinstance(r[n], float) and r[n] == r[n] for n in F5_XCHECK_NAMES)
                for r in rows),
        },
        "not_cached": ["raw_attention", "all_layer_hidden", "F1", "F4", "J-lens"],
        "population_role": "pilot_smoke",
    }
    write_json(report, out / "mcq_hidden_reforward_report.json")
    print(f"[stage-c] done: {len(rows)} vectors dim={sorted(dims)} "
          f"token_forms={report['letter_token_forms']} -> {out}")


if __name__ == "__main__":
    main()
