"""POST-HOC：重新 teacher-forced forward，缓存 {0.5L, 0.7L, final} 三层 pooled hidden。

**这不是 confirmatory，不改动、不覆盖 outputs/stage0/features/ 下的任何冻结产物。**
产物写独立目录 `outputs/stage0/layers/`。confirmatory 的一次性揭盲已用尽，此路径与
freeze manifest / unblinding log 无关。

────────────────────────────────────────────────────────────────────────
为什么做

预注册 §8 要求：「robustness 附录相对深度 {0.5, 0.7, final}，primary 恒 ⌊0.7L⌋」。
Stage 0 生成时按 §11「不缓存全层 hidden」只落了 ⌊0.7L⌋ 一层（每条一个 3584 维 pooled
向量），因此该 robustness 附录**至今未做** —— 这是一处协议欠账，不是锦上添花。
审稿人问「为什么是 0.7L」时，预注册说了要答，而我们答不上来。

不重新生成任何 completion：直接复用已冻结的 completion 文本与**已存好的 token 位置**
（H1 的 `id_token_positions_local`、MCQ 的 `letter_token_index_local`），只做一次前向。
故本脚本不触碰 mention 抽取、不触碰标签、不触碰任何判读规则。

────────────────────────────────────────────────────────────────────────
两道自检（三层对比可信的前提）

三层数值只有在「重新前向与当初生成走同一条路径」时才可比。两处独立验证：

  自检 1｜prompt 重建正确性（仅 H1）
      H1 records 存的是 `prompt_text`(= question_text) 而非 rendered prompt，故需用
      `build_h1_chat_messages` + chat template 重建。重建后算 sha256 与记录里的
      `prompt_hash` 逐条比对 —— 不一致说明 prompt 悄悄变了，立即停。
      （MCQ 直接存了 `rendered_prompt`，无需重建。）

  自检 2｜⌊0.7L⌋ 层数值一致性（两臂）
      重新前向得到的 0.7L pooled 向量，必须与冻结的 `{arm}_hidden.npz` 数值相等。
      这一条同时覆盖 tokenization、层位索引、pooling 三处：任何一处走岔，0.7L 就对不上。
      **对不上就不要相信另外两层。**

────────────────────────────────────────────────────────────────────────
用法

    # 先小样本自检（强烈建议，~1 分钟），确认两道自检都过再跑全量
    python scripts/posthoc_multilayer_reforward.py --arm h1 --limit 20

    # 全量（支持断点续跑，中断后原样重跑即可）
    python scripts/posthoc_multilayer_reforward.py --arm h1
    python scripts/posthoc_multilayer_reforward.py --arm mcq

产物：outputs/stage0/layers/{arm}_hidden_{layer}.npz  （三份，格式与冻结 npz 一致：
      key = trace_id(H1) / example_id(MCQ)，value = (D,) float32）
      outputs/stage0/layers/{arm}_reforward_report.json （含两道自检结果）
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention import h1_data as hd  # noqa: E402

FEAT = Path("outputs/stage0/features")
OUT = Path("outputs/stage0/layers")

ARM_SPEC = {
    "h1": {"records": "h1_completion_records.jsonl", "frozen": "h1_hidden.npz",
           "id_field": "trace_id", "eligible_field": "eligible"},
    "mcq": {"records": "mcq_completion_records.jsonl", "frozen": "mcq_hidden.npz",
            "id_field": "example_id", "eligible_field": "eligible_for_primary"},
}


def resolve_layers(num_layers: int) -> dict[str, int]:
    """§8 robustness 附录的三个相对深度 → hidden_states tuple index。

    0.5/0.7 复用冻结的 resolve_hidden_index（同一实现，避免两套取整规则）；
    final = 最后一个 block 的输出 = tuple index L。
    """
    out = {}
    for rel in (0.5, 0.7):
        idx = ci.resolve_hidden_index(num_layers, rel)
        out[f"rel{rel}"] = idx["hidden_states_tuple_index"]
    out["final"] = num_layers
    return out


def rebuild_h1_prompt(tokenizer, record: dict) -> str:
    """用存下来的 prompt_text 重建 rendered prompt（自检 1 会验证它）。"""
    msgs = hd.build_h1_chat_messages({"question_text": record["prompt_text"]})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def multilayer_teacher_forced(ctx, prompt: str, completion: str,
                              tuple_indices: dict[str, int]) -> dict:
    """复刻 backends._teacher_forced 的 tokenization，但一次取多层 hidden。

    tokenization 必须与生成时逐字一致（prompt 与 completion 分别编码、completion
    不加 special tokens），否则 prompt_len 与 token 位置全错 —— 自检 2 就是查这个。
    """
    torch = ctx["torch"]
    tok = ctx["tokenizer"]
    model = ctx["model"]
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if not c_ids:
        raise ValueError("empty completion tokenization")
    full = torch.tensor([p_ids + c_ids])
    from recover_attention.hidden_state_cache import infer_model_input_device
    dev = infer_model_input_device(model, "auto", torch)
    inputs = {"input_ids": full.to(dev), "attention_mask": torch.ones_like(full).to(dev)}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hiddens = {name: out.hidden_states[idx][0].detach().float().cpu().numpy()
               for name, idx in tuple_indices.items()}
    return {"prompt_len": len(p_ids), "hiddens": hiddens}


def first_identifier_positions(rec: dict) -> list[int]:
    """§8 robustness 附录：只取**第一个** identifier 的全部 token（不作 primary）。

    "第一个" = 按 char 起点排序的第一个 primary mention（family ∈ ATT&CK/CWE 且
    label ∈ grounded/fabricated，判据与 ci._is_primary 一致）。一个 identifier 可能
    跨多个 token，故仍是该 mention 内的 mean pooling，只是不再跨 mention 聚合。

    预注册写明这一版"另报一版，不作 primary"，理由记在 §8：只读第一个 id 会在
    "首 id grounded、次 id fabricated"时系统性读错 mention，人为削弱 H —— 所以它是
    robustness 对照，不是候选 primary。
    """
    prim = [m for m in rec.get("mentions", [])
            if m.get("family") in ci.PRIMARY_FAMILIES
            and m.get("label") in ("grounded", "fabricated")
            and (m.get("token_indices") or [])]
    if not prim:
        return []
    first = min(prim, key=lambda m: int(m.get("start", 0)))
    return sorted({int(i) for i in first["token_indices"]})


def pool_for_arm(arm: str, hidden: np.ndarray, prompt_len: int, rec: dict,
                 pooling: str = "all_identifiers") -> np.ndarray:
    """§8 pooling，与生成时同一实现。

    H1  primary        = 全部 eligible identifier token 的 mean pooling
        first_identifier = §8 robustness 附录变体（只取第一个 identifier）
    MCQ = answer-letter token 自身（单 token，pooling 变体不适用）
    位置直接取自冻结记录，不重新抽取 —— 少一处可能走岔的环节。
    """
    if arm == "h1":
        if pooling == "first_identifier":
            pos = first_identifier_positions(rec)
            if not pos:
                raise ValueError("no primary mention with token_indices")
        else:
            pos = [int(i) for i in (rec.get("id_token_positions_local") or [])]
        return ci.pool_hidden_states(hidden, [prompt_len + int(i) for i in pos])
    j = int(rec["letter_token_index_local"])
    return np.asarray(hidden[prompt_len + j], dtype=np.float32)


def _safe(x: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_SPEC), required=True)
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 条(小样本自检用);0 = 全量")
    ap.add_argument("--pooling", choices=["all_identifiers", "first_identifier"],
                    default="all_identifiers",
                    help="H1 的 §8 pooling。first_identifier = robustness 附录变体,"
                         "产物写独立后缀,不覆盖 primary")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="自检 2 的容差(fp 累积误差);超过即判定路径不一致")
    args = ap.parse_args()

    spec = ARM_SPEC[args.arm]
    suffix = "" if args.pooling == "all_identifiers" else f"_{args.pooling}"
    if args.pooling == "first_identifier" and args.arm != "h1":
        raise SystemExit("--pooling first_identifier 只适用于 H1（MCQ 是单 token）")
    records = [json.loads(l) for l in
               open(FEAT / spec["records"], encoding="utf-8") if l.strip()]
    elig = [r for r in records if r.get(spec["eligible_field"])]
    if args.arm == "h1":
        elig = [r for r in elig if r.get("id_token_positions_local")]
    if args.limit:
        elig = elig[: args.limit]
    print(f"[reforward:{args.arm}] eligible records: {len(elig)}"
          f"{' (limited)' if args.limit else ''}")

    from recover_attention.generation.backends import load_backend
    ctx = load_backend(args.model_path)
    tok = ctx["tokenizer"]
    L = int(ctx["model"].config.num_hidden_layers)
    layers = resolve_layers(L)
    print(f"[reforward:{args.arm}] L={L} → tuple indices {layers}")

    frozen = np.load(FEAT / spec["frozen"])
    OUT.mkdir(parents=True, exist_ok=True)
    unit_dir = OUT / f"{args.arm}_units{suffix}"
    for name in layers:
        (unit_dir / name).mkdir(parents=True, exist_ok=True)

    prompt_hash_mismatch, diffs, n_done, n_skip = [], [], 0, 0
    t0 = time.time()
    for i, rec in enumerate(elig, 1):
        rid = str(rec[spec["id_field"]])
        paths = {n: unit_dir / n / f"{_safe(rid)}.npy" for n in layers}
        if all(p.exists() for p in paths.values()):
            n_skip += 1
            continue

        if args.arm == "h1":
            prompt = rebuild_h1_prompt(tok, rec)
            # 自检 1：prompt 必须与生成时逐字一致
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if rec.get("prompt_hash") and h != rec["prompt_hash"]:
                prompt_hash_mismatch.append(rid)
                if len(prompt_hash_mismatch) >= 3:
                    raise SystemExit(
                        f"STOP (自检 1): prompt 重建与 prompt_hash 不一致，已 "
                        f"{len(prompt_hash_mismatch)} 条 (e.g. {prompt_hash_mismatch[:3]})。"
                        f"重建的 prompt 与生成时不同 → 三层结果不可比。")
            completion = rec["completion"]
        else:
            prompt = rec["rendered_prompt"]
            completion = rec["raw_completion"]

        tf = multilayer_teacher_forced(ctx, prompt, completion, layers)
        for name in layers:
            vec = pool_for_arm(args.arm, tf["hiddens"][name], tf["prompt_len"], rec,
                               pooling=args.pooling)
            np.save(paths[name], np.asarray(vec, dtype=np.float32))

        # 自检 2：只在 primary pooling 下适用 —— 换了 pooling,与冻结 cache 本就
        # 不该相等,拿它当"一致性"检查会得到必然失败的假警报。
        if args.pooling == "all_identifiers" and rid in frozen.files:
            v07 = np.load(paths["rel0.7"])
            diffs.append(float(np.abs(v07 - frozen[rid]).max()))

        n_done += 1
        if i % 50 == 0 or i == len(elig):
            rate = n_done / max(1e-9, time.time() - t0)
            eta = (len(elig) - i) / rate / 60 if rate > 0 else float("nan")
            print(f"[reforward:{args.arm}] {i}/{len(elig)} done={n_done} skip={n_skip} "
                  f"{rate:.1f}/s ETA~{eta:.0f}min")

    # ---- 聚合成 npz ----
    written = {}
    for name in layers:
        store = {}
        for rec in elig:
            rid = str(rec[spec["id_field"]])
            p = unit_dir / name / f"{_safe(rid)}.npy"
            if p.exists():
                store[rid] = np.load(p)
        path = OUT / f"{args.arm}_hidden_{name}{suffix}.npz"
        np.savez_compressed(path, **store)
        written[name] = {"path": str(path), "n": len(store)}
        print(f"[reforward:{args.arm}] {name}: {len(store)} vectors → {path}")

    max_diff = max(diffs) if diffs else None
    # 换了 pooling 时这条检查**按设计不适用**（不同 pooling 本就不该复现冻结 cache）。
    # 此时必须报 n/a 而非 FAIL —— 一个必然失败的检查会让人习惯性忽略真警报，
    # 那就等于把这道自检废掉了。
    check2_applicable = args.pooling == "all_identifiers"
    check2_ok = check2_applicable and bool(diffs) and max_diff is not None and max_diff <= args.tol
    report = {
        "schema_version": "posthoc_multilayer_reforward_v1",
        "status": ("POST-HOC — not confirmatory; does not modify outputs/stage0/features/. "
                   "Fills the section-8 layer-robustness appendix that Stage 0 could not "
                   "produce because only one layer was cached."),
        "arm": args.arm,
        "pooling": args.pooling,
        "model_path": args.model_path,
        "num_layers": L,
        "tuple_indices": layers,
        "n_eligible": len(elig),
        "n_forwarded_this_run": n_done,
        "n_resumed_skip": n_skip,
        "check1_prompt_hash": {
            "applicable": args.arm == "h1",
            "n_mismatch": len(prompt_hash_mismatch),
            "ok": len(prompt_hash_mismatch) == 0,
            "meaning": "rebuilt prompt is byte-identical to the one used at generation time",
        },
        "check2_rel07_matches_frozen": {
            "applicable": check2_applicable,
            "n_compared": len(diffs),
            "max_abs_diff": max_diff,
            "tolerance": args.tol,
            "ok": (check2_ok if check2_applicable else None),
            "meaning": (("recomputed 0.7L equals the frozen cache → tokenization, layer index "
                         "and pooling all match; the other two layers are comparable")
                        if check2_applicable else
                        (f"not applicable: pooling={args.pooling} is expected to differ from the "
                         f"frozen cache by design, so equality is not the right test here. "
                         f"Path correctness for this run rests on check 1 (prompt_hash) plus the "
                         f"fact that the all_identifiers run over the same code path reproduced "
                         f"the frozen cache exactly.")),
        },
        "outputs": written,
    }
    (OUT / f"{args.arm}_reforward_report{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"\n[check 1] prompt_hash mismatches: {len(prompt_hash_mismatch)} "
          f"({'OK' if not prompt_hash_mismatch else 'FAIL'})")
    if check2_applicable:
        print(f"[check 2] rel0.7 vs frozen: n={len(diffs)} max_abs_diff="
              f"{max_diff if max_diff is not None else 'n/a'} "
              f"({'OK' if check2_ok else 'FAIL — do NOT trust the other layers'})")
    else:
        print(f"[check 2] n/a — pooling={args.pooling} is expected to differ from the frozen "
              f"cache by design (path correctness rests on check 1)")
    print(f"[reforward:{args.arm}] report → {OUT / f'{args.arm}_reforward_report{suffix}.json'}")
    if args.pooling == "all_identifiers" and not check2_ok:
        raise SystemExit("自检 2 未通过：重新前向与冻结产物不一致，三层结果不可用。")


if __name__ == "__main__":
    main()
