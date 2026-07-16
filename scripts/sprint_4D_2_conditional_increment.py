"""Sprint 4D-2 conditional-increment pipeline (W0.5-B: 实现 + smoke).

设计权威: docs/paper/preregistration.md (v2.1, frozen)。
本轮只实现 + smoke,不启动 Stage 0 全量生成。

stages:
  preflight       启动 gate:G1 CFP flag / G2 prereg hash / G3 真实 smoke report。
                  G2 不匹配 → 立即停。Stage 0 = G1 AND G2 AND G3。
  smoke_synthetic 无模型:纯统计接线自检(ladder/gate/increment/rq2)。**不构成 G3**。
  verify_hidden   真模型 1 次前向:核验 hidden_states tuple index = block+1。
  smoke_model     真模型 ≤20 prompt 端到端:生成 → identifier → char→token span →
                  全 identifier hidden pooling → completion cache → ladder/gate/rq2。
                  通过后写 smoke_report.json,这才是 G3。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention import h1_data as hd  # noqa: E402
from recover_attention import h1_f5_features as f5f  # noqa: E402
from recover_attention.data_io import read_jsonl  # noqa: E402
from recover_attention.h1_identifier import (  # noqa: E402
    assert_no_h1_gold_label_leakage,
    build_ontology_index,
    extract_identifiers,
    label_completion,
)
from recover_attention.hidden_state_cache import (  # noqa: E402
    import_torch,
    import_transformers_components,
    infer_model_input_device,
    resolve_torch_dtype,
)

SMOKE_SCHEMA_VERSION = "4D2_completion_cache_v1"
SMOKE_MAX_PROMPTS = 20
REFUSAL_RE = re.compile(
    r"\b(i\s+cannot|i\s+can't|i\s+do\s+not\s+have|i\s+don't\s+have|no\s+known|"
    r"not\s+aware\s+of|unable\s+to|cannot\s+determine|can't\s+determine|"
    r"insufficient\s+information)\b",
    re.IGNORECASE,
)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ------------------------------------------------------------------ preflight
def _read_g3(smoke_report: Path, prereg_sha: str) -> dict:
    """G3 = 真实 ≤20 prompt model-in-loop smoke 通过。synthetic smoke 不算。"""
    if not smoke_report.exists():
        return {"ok": False, "reason": f"missing smoke report: {smoke_report}"}
    try:
        rep = json.loads(smoke_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"unreadable smoke report: {exc}"}
    checks = {
        "kind_is_model_smoke": rep.get("kind") == "model_smoke",
        "passed": rep.get("passed") is True,
        "schema_version": rep.get("schema_version") == SMOKE_SCHEMA_VERSION,
        "prereg_sha256_matches": rep.get("prereg_sha256") == prereg_sha,
        "hidden_verified": bool((rep.get("hidden_verification") or {}).get("numeric_match")),
        "prompt_count_within_limit": isinstance(rep.get("n_prompts"), int)
        and 0 < rep["n_prompts"] <= SMOKE_MAX_PROMPTS,
    }
    return {"ok": all(checks.values()), "checks": checks, "report": str(smoke_report)}


def preflight(args) -> dict:
    prereg = Path(args.prereg); lock = Path(args.lock)
    g2 = ci.check_preregistration_frozen(str(prereg), str(lock))
    g1 = Path(args.cfp_flag).exists() if args.cfp_flag else False
    g3 = _read_g3(Path(args.output_dir) / "smoke_report.json", g2["current"])
    hidden = {"qwen2.5-7b_L28": ci.resolve_hidden_index(28),
              "llama-3.1-8b_L32": ci.resolve_hidden_index(32)}
    report = {
        "G1_cfp_confirmed": g1,
        "G2_preregistration": g2,
        "G3_model_smoke": g3,
        "hidden_index": hidden,
        "constants": {"eps_auroc": ci.EPS_AUROC, "delta_rank_biserial": ci.DELTA_RANK_BISERIAL,
                      "rel_depth": ci.DEFAULT_REL_DEPTH, "rq2_min_per_cell": ci.RQ2_MIN_PER_CELL,
                      "rq2_lowpower_total_pos": ci.RQ2_LOWPOWER_TOTAL_POS,
                      "c_grid": list(ci.C_GRID)},
        # lock 规定 Stage 0 需 G1+G2+G3 三者齐备。少一个都不许跑 2880 前向。
        "stage0_full_generation_allowed": bool(g1 and g2["match"] and g3["ok"]),
        "note": "Stage 0(2880 全量)需 G1+G2+G3;W0.5-B(实现+smoke)不受 G1 阻塞。",
    }
    out = Path(args.output_dir) / "preflight_report.json"
    _write(out, report)
    print(f"[preflight] G1 cfp={g1}  G2 hash match={g2['match']}  G3 model-smoke={g3['ok']}")
    print(f"[preflight] hidden Qwen block19→tuple20  Llama block22→tuple23")
    print(f"[preflight] stage0_full_generation_allowed={report['stage0_full_generation_allowed']}  → {out}")
    # G2 是冻结校验:hash 不一致 = 跑后改了判读规则/定义 → 立即停,不往下走任何阶段。
    if not g2["match"]:
        raise SystemExit(
            f"PREFLIGHT STOP (G2): preregistration.md sha256 mismatch.\n"
            f"  current={g2['current']}\n  locked ={g2['locked']}\n"
            f"冻结后改设计必须 bump version + 重算 hash + 记录变更,不得静默改动。")
    return report


def require_stage0_gate(args) -> None:
    """任何真数据全量阶段的统一入口门控。"""
    rep = preflight(args)
    if not rep["stage0_full_generation_allowed"]:
        raise SystemExit(
            "STAGE 0 BLOCKED: need G1(CFP) + G2(prereg hash) + G3(model smoke) all green; "
            f"got G1={rep['G1_cfp_confirmed']} G2={rep['G2_preregistration']['match']} "
            f"G3={rep['G3_model_smoke']['ok']}")


# ------------------------------------------------------------ synthetic smoke
def smoke_synthetic(args) -> dict:
    """无模型:合成一维特征,只验证统计函数能接线。**不构成 G3**（无真实 prompt /
    identifier / token span / hidden / cache）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

    def make_task(n_groups, per, o_strength, h_strength, seed):
        r = np.random.default_rng(seed)
        groups = np.repeat(np.arange(n_groups), per)
        y = r.integers(0, 2, n_groups * per)
        O = (y * o_strength + r.normal(0, 1, y.size)).reshape(-1, 1)
        H = (y * h_strength + r.normal(0, 1, y.size)).reshape(-1, 1)
        idlp = r.normal(0, 1, y.size)
        folds = ci.stratified_grouped_folds(y, groups, n_splits=5, seed=0)
        s_o = ci.oof_risk_scores(O, y, groups, mk, folds=folds)
        s_h = ci.oof_risk_scores(H, y, groups, mk, folds=folds)
        s_oh = ci.oof_risk_scores(np.hstack([O, H]), y, groups, mk, folds=folds)
        return dict(y=y, groups=groups, idlp=idlp, folds=folds, s_o=s_o, s_h=s_h, s_oh=s_oh)

    mcq = make_task(80, 1, o_strength=3.0, h_strength=0.2, seed=args.seed + 1)
    h1 = make_task(70, 4, o_strength=0.6, h_strength=1.3, seed=args.seed + 2)

    gate = ci.independent_grouped_bootstrap_D(
        {"y": mcq["y"], "score": mcq["s_o"], "groups": mcq["groups"]},
        {"y": h1["y"], "score": h1["s_o"], "groups": h1["groups"]},
        n_boot=args.n_boot, seed=args.seed)

    inc = {}
    for name, t in (("mcq", mcq), ("h1", h1)):
        d = ci.paired_grouped_bootstrap_delta(t["y"], t["s_o"], t["s_oh"], t["groups"],
                                              n_boot=args.n_boot, seed=args.seed)
        d["read"] = ci.equivalence_read(d["ci_lo"], d["ci_hi"])
        # §7 强制同时报告三者,防"无增量"是融合 artifact
        d["auroc_O"] = ci.auroc(t["s_o"], t["y"])
        d["auroc_H_alone"] = ci.auroc(t["s_h"], t["y"])
        d["auroc_OH"] = ci.auroc(t["s_oh"], t["y"])
        inc[name] = d

    rq2 = _rq2_block(h1["y"], h1["idlp"], h1["s_o"], h1["s_oh"], h1["groups"],
                     h1["folds"], args.n_boot, args.seed)

    result = {"kind": "synthetic_smoke", "is_g3": False,
              "note": "合成特征,不含真实模型/identifier/hidden;不能计为 G3。",
              "gate": gate, "increment": inc, "rq2": rq2}
    out = Path(args.output_dir) / "smoke_synthetic_report.json"
    _write(out, result)
    print(f"[smoke_synthetic] gate verdict={gate['verdict']}  "
          f"H1 Δ_H={inc['h1']['point']:+.3f} {inc['h1']['read']}  "
          f"MCQ Δ_H={inc['mcq']['point']:+.3f} {inc['mcq']['read']}  → {out}")
    return result


def _rq2_block(y, idlp, s_o, s_oh, groups, folds, n_boot, seed) -> dict:
    """RQ2:fold-specific 阈值分层（§5）。绝不在全体数据上取一次中位数。"""
    y = np.asarray(y)
    fs = ci.rq2_fold_strata(idlp, folds)
    hi = fs["strata"]
    total_pos = int((y == 1).sum())
    rq2 = {"fold_thresholds": fs["fold_thresholds"],
           "threshold_policy": "per-outer-fold train median of id-logprob, applied to that fold's test",
           "total_positives": total_pos, "power_flag": ci.rq2_power_flag(total_pos), "strata": {}}
    for lname, mask in (("high_conf", hi), ("low_conf", ~hi)):
        yc = y[mask]
        cell = {"n": int(mask.sum()), "n_pos": int((yc == 1).sum()), "n_neg": int((yc == 0).sum())}
        if not ci.rq2_cell_ok(yc):
            rq2["strata"][lname] = {**cell, "status": "insufficient"}
            continue
        try:
            d = ci.paired_grouped_bootstrap_delta(yc, np.asarray(s_o)[mask], np.asarray(s_oh)[mask],
                                                  np.asarray(groups)[mask], n_boot=n_boot, seed=seed)
        except RuntimeError as exc:
            rq2["strata"][lname] = {**cell, "status": "bootstrap_stopped", "reason": str(exc)}
            continue
        rq2["strata"][lname] = {**cell, "status": "ok", "delta_point": d["point"],
                                "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                                "n_valid": d["n_valid"]}
    return rq2


# --------------------------------------------------------------- model backend
def load_backend(model_path: str) -> dict:
    """复用 4D-1 已验证的 8-bit 本地加载路径。

    不用 BF16 + device_map="cuda":Qwen2.5-7B 的 BF16 权重 ~15GB,12GB 显存必 OOM;
    且 hidden 验证必须与将来正式生成/缓存**同一条**加载路径,否则验证意义有限。
    """
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


def _to_device(ctx, encoded):
    device = infer_model_input_device(ctx["model"], "auto", ctx["torch"])
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in encoded.items()}


# ---------------------------------------------------- model hidden-index verify
def verify_hidden(args, ctx: dict | None = None) -> dict:
    """1 次前向,核验 forward-hook(目标 block 输出) == hidden_states[block+1] 数值相等。"""
    ctx = ctx or load_backend(args.model_path)
    torch = ctx["torch"]; model = ctx["model"]; tok = ctx["tokenizer"]
    L = model.config.num_hidden_layers
    idx = ci.resolve_hidden_index(L)
    block, tuple_idx = idx["block_index_zero_based"], idx["hidden_states_tuple_index"]

    captured: dict[str, Any] = {}
    layer = model.model.layers[block]

    def hook(_m, _in, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    handle = layer.register_forward_hook(hook)
    enc = _to_device(ctx, tok("The CVE identifier for Log4Shell is", return_tensors="pt"))
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    handle.remove()
    hs = out.hidden_states[tuple_idx].detach().float().cpu()
    hook_h = captured["h"]
    max_abs = float((hook_h - hs).abs().max())
    match = bool(torch.allclose(hook_h, hs, atol=1e-3, rtol=1e-3))
    report = {"model": ctx["model_path"], "num_layers": L, "block_index_zero_based": block,
              "hidden_states_tuple_index": tuple_idx, "hook_shape": list(hook_h.shape),
              "hidden_states_shape": list(hs.shape), "numeric_match": match,
              "max_abs_diff": max_abs, "backend": ctx["backend"]}
    out_p = Path(args.output_dir) / "verify_hidden_report.json"
    _write(out_p, report)
    print(f"[verify_hidden] L={L} block={block} tuple_idx={tuple_idx} "
          f"match={match} max_abs_diff={max_abs:.2e}  → {out_p}")
    if not match:
        raise SystemExit("hidden tuple-index numeric verification FAILED (off-by-one?).")
    return report


# ------------------------------------------------------------- model-in-loop smoke
def _select_smoke_prompts(records: list[dict], n: int, seed: int) -> list[dict]:
    """attack/cwe 各半（primary family），train split，确定性 seed。不按结果挑。"""
    per = {"attack": n // 2, "cwe": n - n // 2}
    out: list[dict] = []
    for family, count in sorted(per.items()):
        bucket = sorted([r for r in records
                         if r.get("split") == "train" and r.get("route") == "recall"
                         and r.get("family") == family],
                        key=lambda r: r["example_id"])
        rng = random.Random(_stable_seed("4d2_smoke", seed, family))
        rng.shuffle(bucket)
        if len(bucket) < count:
            raise SystemExit(f"not enough train/recall prompts for {family}: need {count}, have {len(bucket)}")
        out.extend(bucket[:count])
    out.sort(key=lambda r: (r["family"], r["example_id"]))
    return out


def _generate(ctx, prompt: str, *, max_new_tokens: int, do_sample: bool,
              temperature: float, top_p: float, seed: int) -> str:
    torch = ctx["torch"]; tok = ctx["tokenizer"]; model = ctx["model"]
    inputs = _to_device(ctx, tok(prompt, return_tensors="pt"))
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None) or eos
    torch.manual_seed(int(seed) % (2 ** 31))
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2 ** 31))
    kwargs = {**inputs, "max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample),
              "use_cache": True, "renormalize_logits": True, "remove_invalid_values": True,
              "pad_token_id": pad, "eos_token_id": eos}
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    with torch.no_grad():
        ids = model.generate(**kwargs)
    new_ids = ids[0, prompt_len:].detach().cpu().tolist()
    if eos is not None and new_ids and int(new_ids[-1]) == int(eos):
        new_ids = new_ids[:-1]
    return tok.decode(new_ids, skip_special_tokens=True)


def _teacher_forced_pass(ctx, prompt: str, completion: str, tuple_idx: int) -> dict:
    """一次前向拿齐 completion 的 token logprob / entropy / rank + 目标层 hidden。

    completion 用 return_offsets_mapping 重新编码:offsets 与 completion 的字符坐标
    严格一致,identifier 的 char span 才能可靠映射到 token（这是 4D-2 缓存契约 §11 的
    关键一步;偏一格会让 H 读错位置）。
    """
    torch = ctx["torch"]; tok = ctx["tokenizer"]; model = ctx["model"]
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    enc = tok(completion, return_offsets_mapping=True, add_special_tokens=False)
    c_ids = enc["input_ids"]
    offsets = [list(o) for o in enc["offset_mapping"]]
    if not c_ids:
        raise ValueError("empty completion tokenization")
    full = torch.tensor([p_ids + c_ids])
    inputs = _to_device(ctx, {"input_ids": full,
                              "attention_mask": torch.ones_like(full)})
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0].float()
    logprobs_all = torch.log_softmax(logits, dim=-1)
    prompt_len = len(p_ids)
    token_logprobs: dict[int, float] = {}
    token_entropies: dict[int, float] = {}
    first_ranks: dict[int, int] = {}
    for j, tid in enumerate(c_ids):
        step = prompt_len + j - 1          # 预测第 j 个 completion token 的那一步
        lp_row = logprobs_all[step]
        token_logprobs[j] = float(lp_row[tid])
        p = lp_row.exp()
        token_entropies[j] = float(-(p * lp_row).sum())
        first_ranks[j] = int((lp_row > lp_row[tid]).sum()) + 1
    hidden = out.hidden_states[tuple_idx][0].detach().float().cpu().numpy()  # (T, D) FP32 落盘
    return {"prompt_len": prompt_len, "completion_ids": c_ids, "offsets": offsets,
            "token_logprobs": token_logprobs, "token_entropies": token_entropies,
            "first_token_ranks": first_ranks, "hidden": hidden,
            "hidden_dim": int(hidden.shape[-1]), "seq_len": int(hidden.shape[0])}


def _build_cache_record(record, prompt, completion, labels, tf, sample_type, sample_index) -> dict:
    """§11 缓存契约的 completion-level 主记录。"""
    mentions = []
    for m in labels["mentions"]:
        local = f5f.token_indices_for_char_span(tf["offsets"], int(m["start"]), int(m["end"]))
        mentions.append({**m, "token_indices": local})
    refusal = (not any(ci._is_emitted(m) for m in mentions)) and bool(REFUSAL_RE.search(completion or ""))
    lab = ci.completion_label(mentions, refusal=refusal)
    local_pos = ci.eligible_identifier_token_positions(mentions)
    id_lp = [tf["token_logprobs"].get(i) for i in local_pos]
    comp_lps = [tf["token_logprobs"][j] for j in sorted(tf["token_logprobs"])]

    row: dict[str, Any] = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "trace_id": f"{record['example_id']}__{sample_type}_{sample_index}",
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "route": record["route"], "family": record["family"], "split": record["split"],
        "sample_type": sample_type, "sample_index": sample_index,
        "prompt_text": record["question_text"],
        "completion": completion,
        "refusal": refusal,
        "mentions": mentions,
        "id_token_positions_local": local_pos,
        "id_token_positions_global": [tf["prompt_len"] + i for i in local_pos],
        "id_string_text": ci.id_string_text(mentions),
        **lab,
        **ci.surface_format_features(completion, mentions),
    }
    if local_pos:
        row.update(f5f.mention_logprob_features(tf["token_logprobs"], tf["token_entropies"],
                                                tf["first_token_ranks"], local_pos))
    row.update(f5f.sequence_logprob_features(comp_lps, id_token_count=len(local_pos)))
    row.update(f5f.extract_verbalized_confidence(completion))
    row["f5_id_logprob_mean_recheck"] = float(np.mean([v for v in id_lp if v is not None])) if local_pos else None
    assert_no_h1_gold_label_leakage(row)
    return row


def _primary_id_of(row: dict) -> str | None:
    """该 trace 的代表 identifier（首个 eligible primary 的 normalized 串）。"""
    for m in row["mentions"]:
        if ci._is_primary(m):
            return str(m["normalized"])
    return None


def attach_cross_sample_f5(rows: list[dict]) -> None:
    """F5 的 self-consistency / id-agreement 是**跨同一 question 的 K 条 trace**聚合的
    （§7 复用 h1_f5_features）。逐条算不出来,必须等一个 question 的全部 trace 生成完。

    不补的话这两维会被静默填 0,等于把 O 侧基线削弱 → 人为抬高 Δ_H。
    """
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["example_id"], []).append(r)
    for _, group in by_q.items():
        ids = [_primary_id_of(r) for r in group]
        present = [i for i in ids if i]
        cons = f5f.exact_consistency(present)
        for r, own in zip(group, ids):
            r["f5_self_consistency_exact"] = cons["f5_self_consistency_exact"]
            r["f5_id_agreement_rate"] = (
                f5f.id_agreement_rate(own, present) if own else None)
            r["f5_consistency_n_traces"] = cons["num_values"]


def smoke_model(args) -> dict:
    """真模型 ≤20 prompt 端到端 smoke。通过 → 写 smoke_report.json(G3)。"""
    n_prompts = int(args.smoke_prompts)
    if not 0 < n_prompts <= SMOKE_MAX_PROMPTS:
        raise SystemExit(f"smoke must use ≤{SMOKE_MAX_PROMPTS} prompts (prereg), got {n_prompts}")
    out_dir = Path(args.output_dir)
    prereg_sha = ci.sha256_of(args.prereg)
    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit("smoke_model STOP (G2): preregistration hash mismatch")

    records = read_jsonl(Path(args.samples_jsonl))
    selected = _select_smoke_prompts(records, n_prompts, args.seed)
    index = build_ontology_index(Path(args.ontology_dir))

    ctx = load_backend(args.model_path)
    hv = verify_hidden(args, ctx=ctx)
    tuple_idx = hv["hidden_states_tuple_index"]

    rows: list[dict] = []
    hidden_vecs: dict[str, np.ndarray] = {}
    specs = [("greedy", 0, False)] + [("sampled", i + 1, True) for i in range(int(args.samples_per_question))]
    total = len(selected) * len(specs)
    done = 0
    retok_mismatch = 0
    for record in selected:
        prompt = ctx["tokenizer"].apply_chat_template(
            hd.build_h1_chat_messages(record), tokenize=False, add_generation_prompt=True)
        for sample_type, sample_index, do_sample in specs:
            seed = _stable_seed(args.seed, record["example_id"], sample_type, sample_index)
            completion = _generate(ctx, prompt, max_new_tokens=args.max_new_tokens,
                                   do_sample=do_sample, temperature=args.temperature,
                                   top_p=args.top_p, seed=seed)
            done += 1
            if not completion.strip():
                print(f"[smoke_model] {done}/{total} empty completion, skipped")
                continue
            tf = _teacher_forced_pass(ctx, prompt, completion, tuple_idx)
            labels = label_completion(completion, record["question_text"], index)
            row = _build_cache_record(record, prompt, completion, labels, tf,
                                      sample_type, sample_index)
            # H 只对 eligible completion 有定义:pool 全部 eligible identifier token
            if row["eligible"]:
                gpos = row["id_token_positions_global"]
                hidden_vecs[row["trace_id"]] = ci.pool_hidden_states(tf["hidden"], gpos)
                row["hidden_dim"] = int(hidden_vecs[row["trace_id"]].shape[0])
                row["hidden_pooled_n_tokens"] = len(set(gpos))
            rows.append(row)
            print(f"[smoke_model] {done}/{total} {row['trace_id']} eligible={row['eligible']} "
                  f"label={row['label']} ids={row['n_primary_mentions']}")

    attach_cross_sample_f5(rows)

    cache_path = out_dir / "smoke_completion_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hid_path = out_dir / "smoke_hidden.npz"
    np.savez_compressed(hid_path, **hidden_vecs)

    checks, stats = _smoke_checks(rows, hidden_vecs, hv)
    ladder = None
    ladder_error = None
    eligible = [r for r in rows if r["eligible"]]
    y = np.array([r["label"] for r in eligible])
    if len(eligible) >= 10 and len(np.unique(y)) == 2:
        try:
            ladder = _run_ladder(eligible, hidden_vecs, args)
        except (RuntimeError, ValueError) as exc:
            ladder_error = str(exc)
    else:
        ladder_error = (f"insufficient eligible completions for ladder wiring check: "
                        f"n={len(eligible)}, classes={np.unique(y).tolist()}")
    checks["ladder_wiring_ran"] = ladder is not None
    if ladder_error:
        checks["ladder_note"] = ladder_error

    passed = all(v is True for k, v in checks.items() if k != "ladder_note")
    report = {
        "kind": "model_smoke", "is_g3": True, "passed": bool(passed),
        "schema_version": SMOKE_SCHEMA_VERSION,
        "prereg_sha256": prereg_sha,
        "n_prompts": len(selected), "n_traces": len(rows),
        "traces_per_prompt": len(specs),
        "hidden_verification": hv,
        "backend": ctx["backend"],
        "checks": checks, "stats": stats,
        "ladder": ladder, "ladder_error": ladder_error,
        "artifacts": {"completion_cache": str(cache_path), "hidden_npz": str(hid_path)},
    }
    _write(out_dir / "smoke_report.json", report)
    print(f"[smoke_model] passed={passed}  checks={json.dumps(checks, ensure_ascii=False)}")
    if not passed:
        raise SystemExit("model smoke FAILED — G3 not granted; see smoke_report.json")
    return report


def _smoke_checks(rows: list[dict], hidden_vecs: dict, hv: dict) -> tuple[dict, dict]:
    eligible = [r for r in rows if r["eligible"]]
    dims = {int(v.shape[0]) for v in hidden_vecs.values()}
    span_ok = all(
        all(m["token_indices"] for m in r["mentions"] if ci._is_emitted(m))
        for r in rows
    )
    checks = {
        "generated_any_trace": len(rows) > 0,
        "some_completion_eligible": len(eligible) > 0,
        "every_eligible_has_hidden": all(r["trace_id"] in hidden_vecs for r in eligible),
        "hidden_dim_consistent": len(dims) <= 1,
        "hidden_pooled_all_identifier_tokens": all(
            r.get("hidden_pooled_n_tokens") == len(set(r["id_token_positions_global"]))
            for r in eligible),
        "every_emitted_mention_maps_to_tokens": span_ok,
        "f5_present_on_eligible": all(r.get("f5_id_logprob_mean") is not None for r in eligible),
        "cross_sample_f5_present": all(r.get("f5_self_consistency_exact") is not None
                                       for r in eligible),
        "no_emission_policy_recorded": all(
            (r["eligible"] is True) != (r["primary_exclusion_reason"] is not None) for r in rows),
        "hidden_index_numeric_match": bool(hv["numeric_match"]),
    }
    reasons: dict[str, int] = {}
    for r in rows:
        if r["primary_exclusion_reason"]:
            reasons[r["primary_exclusion_reason"]] = reasons.get(r["primary_exclusion_reason"], 0) + 1
    y = [r["label"] for r in eligible]
    stats = {
        "n_traces": len(rows), "n_eligible": len(eligible),
        "n_positive": int(sum(v == 1 for v in y)), "n_negative": int(sum(v == 0 for v in y)),
        "eligible_rate": len(eligible) / len(rows) if rows else None,
        "emission_failure_rate": (sum(r["emission_failure"] for r in rows) / len(rows)) if rows else None,
        "primary_exclusion_reasons": reasons,
        "hidden_dim": sorted(dims),
        "n_groups": len({r["group_id"] for r in eligible}),
    }
    return checks, stats


def _run_ladder(eligible: list[dict], hidden_vecs: dict, args) -> dict:
    """在 smoke 规模上跑通 ladder → gate → increment → rq2 的**接线**。

    n≈20 prompt,统计量无意义;这里只证明真实 text/F5/hidden 能进同一条协议并出有限 OOF。
    """
    y = np.array([r["label"] for r in eligible])
    groups = np.array([r["group_id"] for r in eligible])
    H = np.vstack([hidden_vecs[r["trace_id"]] for r in eligible])
    folds = ci.stratified_grouped_folds(y, groups, n_splits=args.smoke_splits, seed=args.seed)
    specs = ci.build_ladder_specs(eligible, hidden=H)
    out: dict[str, Any] = {"n": int(len(y)), "n_pos": int((y == 1).sum()),
                           "n_splits": int(args.smoke_splits), "rungs": {}}
    scores: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        res = ci.block_oof_scores(y, groups, folds, text=spec["text"], dense=spec["dense"],
                                  inner_splits=2, seed=args.seed)
        scores[name] = res["scores"]
        out["rungs"][name] = {"auroc": res["auroc"], "chosen_C": res["chosen_C"]}
    # §7 要求三者同报
    out["increment"] = {
        "auroc_O": ci.auroc(scores["rung6_O_f5_plus_text"], y),
        "auroc_H_alone": ci.auroc(scores["H_alone"], y),
        "auroc_OH": ci.auroc(scores["O_plus_H"], y),
    }
    idlp = np.array([float(r.get("f5_id_logprob_mean") or 0.0) for r in eligible])
    out["rq2_strata_wiring"] = ci.rq2_fold_strata(idlp, folds)["fold_thresholds"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="preflight,smoke_synthetic")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--cfp-flag", default="docs/paper/CFP_CONFIRMED")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--samples-jsonl", default="data/processed/h1/h1_samples.jsonl")
    ap.add_argument("--ontology-dir", default="data/raw/ontology")
    ap.add_argument("--output-dir", default="outputs/logs/sprint_4D_2_conditional_increment")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-prompts", type=int, default=SMOKE_MAX_PROMPTS)
    ap.add_argument("--smoke-splits", type=int, default=3)
    ap.add_argument("--samples-per-question", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    fns = {"preflight": preflight, "smoke_synthetic": smoke_synthetic,
           "verify_hidden": verify_hidden, "smoke_model": smoke_model,
           "stage0_gate": require_stage0_gate}
    for s in [x.strip() for x in args.stage.split(",") if x.strip()]:
        if s not in fns:
            raise SystemExit(f"stage not implemented in W0.5-B: {s} "
                             f"(generate/cache/ladder/gate/increment/rq2 需真数据 + Stage 0 门控)")
        fns[s](args)


if __name__ == "__main__":
    main()
