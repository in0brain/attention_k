r"""MCQ 每 unit 的分阶段计时 + "只去掉未使用计算"的候选优化的**逐位一致性核对**。

⚠️ 需要 GPU。**必须在 H1 正式生成结束后串行执行**，不与 H1 争抢显存。
只用已有 gate 样本（`g3_budget512_reparsed/` + `g4_budget512_reparsed/` 的
prompt 与 completion 原文），**不产出任何生产记录**，不写 outputs/。

分阶段：
    generate                 `_generate`（真实生成，最大头）
    tf_tokenize              prompt / completion 编码
    tf_forward               一次前向（含 output_hidden_states）
    tf_cast_logsoftmax       logits.float() + log_softmax（整表）
    tf_pertoken_loop         逐 token 的 logprob/entropy/rank —— **MCQ 只用其中 1 个**
    tf_hidden_to_cpu         hidden 整表搬到 CPU
    score_post               解析 / canonicalization / 表面特征 / F5（除 teacher-forced 外）
    write                    hidden .npy + result 行追加（含 fsync，写临时目录）

候选优化（**唯一一项**）：MCQ 路径跳过 `tf_pertoken_loop`，只对答案字母那一个 token
取 `float(lp[prompt_len+j-1][c_ids[j]])`。表达式一字不改 → **应当逐位相同**。
判据事前写死：所有实际消费字段**逐位相等**才算通过；有任何差异即**不采用**。

用法：
    python scripts/diagnostics/time_mcq_unit_phases.py --n 12 --max-seconds 900
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as bk              # noqa: E402

MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
OUT = ROOT / "tmp/stage1_l2_g3/mcq_phase_timing.json"

GATE_ROWS = [
    ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed/mcq_g3_unit_results.jsonl",
    ROOT / "outputs/stage1_llama_v3/g4_budget512_reparsed/mcq_g4_unit_results.jsonl",
]

#: **事前写死**：候选优化必须在这些字段上**逐位相等**（不是容差，是相等）。
CONSUMED_FIELDS = (
    "parsed_label", "parse_status", "wrong_label", "eligible_for_primary",
    "letter_token_index_local", "letter_token_id", "letter_token_form",
    "readout_to_prompt_end_distance",
    "f5_label_margin", "f5_label_entropy", "f5_full_entropy", "f5_letter_token_logprob",
)
#: 只要**收益份额**低于这个数，就按"收益不明确"处理 —— 沿用已验证版本。事前写死。
ADOPT_MIN_SHARE = 0.08


class _LazyLogprobs(dict):
    """`token_logprobs` 的惰性替身：只有被取的下标才真的算。

    表达式与 `_teacher_forced` 里的循环**逐字相同**：`float(lp[plen + j - 1][tid])`。
    """

    def __init__(self, lp, c_ids, plen):
        super().__init__()
        self._lp, self._c, self._plen = lp, c_ids, plen

    def __getitem__(self, j):
        if j not in self:
            self[j] = float(self._lp[self._plen + j - 1][self._c[j]])
        return super().__getitem__(j)


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_teacher_forced(be, prompt, completion, t: dict, *, pertoken: bool):
    """`_teacher_forced` 的逐阶段计时复刻；`pertoken=False` 即候选优化。"""
    torch = be.ctx["torch"]; tok = be.ctx["tokenizer"]; model = be.ctx["model"]
    t0 = time.perf_counter()
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    enc = tok(completion, return_offsets_mapping=True, add_special_tokens=False)
    c_ids = enc["input_ids"]
    offsets = [list(o) for o in enc["offset_mapping"]]
    full = torch.tensor([p_ids + c_ids])
    inputs = be._to_device({"input_ids": full, "attention_mask": torch.ones_like(full)})
    _sync(torch); t["tf_tokenize"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    _sync(torch); t["tf_forward"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    logits = out.logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    _sync(torch); t["tf_cast_logsoftmax"] += time.perf_counter() - t0

    plen = len(p_ids)
    tlp, tent, trank = {}, {}, {}
    t0 = time.perf_counter()
    if pertoken:
        for j, tid in enumerate(c_ids):
            row = lp[plen + j - 1]
            tlp[j] = float(row[tid])
            p = row.exp()
            tent[j] = float(-(p * row).sum())
            trank[j] = int((row > row[tid]).sum()) + 1
    _sync(torch); t["tf_pertoken_loop"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    hidden = out.hidden_states[be.tuple_idx][0].detach().float().cpu().numpy()
    _sync(torch); t["tf_hidden_to_cpu"] += time.perf_counter() - t0

    return {"prompt_len": plen, "completion_ids": c_ids, "offsets": offsets,
            "token_logprobs": tlp if pertoken else _LazyLogprobs(lp, c_ids, plen),
            "token_entropies": tent, "first_token_ranks": trank,
            "hidden": hidden, "logits": logits,
            "_n_completion_tokens": len(c_ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="计时用的 unit 数")
    ap.add_argument("--max-seconds", type=float, default=900.0, help="硬上限，到点即停")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t_start = time.time()
    cfg = V3AnalysisConfig()
    rows: list[dict] = []
    for p in GATE_ROWS:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    pool = [r for r in rows if r.get("eligible_for_primary")]
    pool.sort(key=lambda r: r["unit_id"])
    # 确定性取样：按补全长度分层，覆盖短/中/长
    pool.sort(key=lambda r: len(r["raw_completion"]))
    idx = np.linspace(0, len(pool) - 1, num=min(args.n, len(pool))).astype(int)
    picked = [pool[i] for i in sorted(set(idx.tolist()))]
    print(f"[pool] eligible={len(pool)}  picked={len(picked)}")

    print("[load] 8-bit backend …")
    be = bk.MCQBackend(MODEL, seed=cfg.generation_seed, max_new_tokens=512)
    torch = be.ctx["torch"]
    hi = be.verify_hidden_index()
    print(f"[hidden_index] {hi['hidden_states_tuple_index']} numeric_match={hi['numeric_match']}")

    phases = {k: 0.0 for k in ("generate", "tf_tokenize", "tf_forward", "tf_cast_logsoftmax",
                               "tf_pertoken_loop", "tf_hidden_to_cpu", "score_post", "write")}
    phases_opt = dict(phases)
    per_unit: list[dict] = []
    mismatches: list[dict] = []
    tmpd = Path(tempfile.mkdtemp(prefix="mcq_timing_"))
    sink = (tmpd / "rows.jsonl").open("a", encoding="utf-8")
    stopped = "completed"

    for k, r in enumerate(picked):
        if time.time() - t_start > args.max_seconds:
            stopped = "max_seconds"
            break
        prompt, completion = r["rendered_prompt"], r["raw_completion"]
        rec = {"example_id": r["example_id"], "rendered_prompt": prompt,
               "candidate_choices": r["candidate_choices"], "question": "",
               "gold_label": r["parsed_label"], "group_id": r.get("group_id")}
        unit = {"sample_type": r["sample_type"], "sample_index": r["sample_index"]}

        # ① 生成（真实，量的是生产里最大的那一段；结果只用于计时，不落生产）
        t0 = time.perf_counter()
        gen = be._generate(prompt, do_sample=unit["sample_type"] == "sampled", seed=1234 + k)
        _sync(torch)
        dt_gen = time.perf_counter() - t0
        phases["generate"] += dt_gen

        # ② 两条 teacher-forced 路径：现行（全量逐 token）与候选（跳过）
        def _run(pertoken: bool, acc: dict):
            t_local = {kk: 0.0 for kk in acc}
            tf = _timed_teacher_forced(be, prompt, completion, t_local, pertoken=pertoken)
            t0 = time.perf_counter()
            be._teacher_forced = lambda *a, **kw: tf                        # noqa: ARG005
            row = be.score_completion(prompt, completion, rec, unit, seed=1)
            _sync(torch)
            t_local["score_post"] = time.perf_counter() - t0
            for kk, v in t_local.items():
                acc[kk] += v
            return row, tf, t_local

        row_cur, tf_cur, tl_cur = _run(True, phases)
        row_opt, tf_opt, tl_opt = _run(False, phases_opt)
        phases_opt["generate"] += dt_gen

        # ③ 写盘（只量一次，两条路径写的是同一行）
        t0 = time.perf_counter()
        vec = np.asarray(row_cur["hidden"], dtype=np.float32)
        np.save(tmpd / f"{r['unit_id']}.npy", vec)
        w = row_cur.copy(); w.pop("hidden")
        sink.write(json.dumps(w, ensure_ascii=False, default=str) + "\n")
        sink.flush(); os.fsync(sink.fileno())
        dt_w = time.perf_counter() - t0
        phases["write"] += dt_w
        phases_opt["write"] += dt_w

        # ④ 逐位核对（事前写死：相等，不是容差）
        diffs = {f: [row_cur.get(f), row_opt.get(f)] for f in CONSUMED_FIELDS
                 if repr(row_cur.get(f)) != repr(row_opt.get(f))}
        h_equal = bool(np.array_equal(np.asarray(row_cur["hidden"]),
                                      np.asarray(row_opt["hidden"])))
        if diffs or not h_equal:
            mismatches.append({"unit_id": r["unit_id"], "fields": diffs,
                               "hidden_bitwise_equal": h_equal})

        per_unit.append({
            "unit_id": r["unit_id"], "n_completion_tokens": tf_cur["_n_completion_tokens"],
            "generate_s": round(dt_gen, 3),
            "cur_total_s": round(dt_gen + sum(tl_cur.values()) + dt_w, 3),
            "opt_total_s": round(dt_gen + sum(tl_opt.values()) + dt_w, 3),
            "pertoken_loop_s": round(tl_cur["tf_pertoken_loop"], 3),
            "bitwise_equal": not diffs and h_equal,
        })
        print(f"  [{k + 1}/{len(picked)}] {r['unit_id']}  tok={tf_cur['_n_completion_tokens']}"
              f"  gen={dt_gen:.2f}s  loop={tl_cur['tf_pertoken_loop']:.2f}s"
              f"  bitwise_equal={not diffs and h_equal}")

    sink.close()
    n = len(per_unit)
    cur_total = sum(phases.values())
    opt_total = sum(phases_opt.values())
    share = (phases["tf_pertoken_loop"] / cur_total) if cur_total else 0.0
    all_equal = not mismatches and n > 0

    doc = {
        "schema": "mcq_phase_timing_v1", "date": "2026-09-15",
        "what": ("MCQ 每 unit 分阶段计时 + 候选优化（跳过 MCQ 用不到的逐 token 统计）"
                 "的逐位一致性核对。只用已有 gate 样本，不产出生产记录。"),
        "stopped_reason": stopped,
        "n_units_timed": n,
        "wall_seconds": round(time.time() - t_start, 1),
        "phase_seconds_current": {k: round(v, 3) for k, v in phases.items()},
        "phase_seconds_candidate": {k: round(v, 3) for k, v in phases_opt.items()},
        "per_unit_mean_s_current": round(cur_total / n, 3) if n else None,
        "per_unit_mean_s_candidate": round(opt_total / n, 3) if n else None,
        "pertoken_loop_share_of_current": round(share, 4),
        "adopt_min_share_prewritten": ADOPT_MIN_SHARE,
        "bitwise_identical_on_consumed_fields": all_equal,
        "consumed_fields_checked": list(CONSUMED_FIELDS) + ["hidden (np.array_equal)"],
        "mismatches": mismatches,
        "per_unit": per_unit,
        "verdict": (
            "adopt_candidate" if (all_equal and share >= ADOPT_MIN_SHARE)
            else "keep_verified_version"),
        "verdict_why": (
            "逐位相同且未使用计算占比 >= 事前写死的阈值" if (all_equal and share >= ADOPT_MIN_SHARE)
            else ("存在逐位差异 —— 按事前判据不采用" if not all_equal
                  else "收益份额低于事前写死的阈值 —— 收益不明确，沿用已验证版本")),
        "not_changed": ("提示 / 预算 / K / 样本 / 量化 / attention 后端 / 随机数规则"
                        "一律未动；本脚本不改任何源码。"),
        "tmp_dir": str(tmpd),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[phases-current ] {doc['phase_seconds_current']}")
    print(f"[per-unit mean  ] cur={doc['per_unit_mean_s_current']}s  "
          f"cand={doc['per_unit_mean_s_candidate']}s")
    print(f"[loop share     ] {share:.2%}  (阈值 {ADOPT_MIN_SHARE:.0%})")
    print(f"[bitwise equal  ] {all_equal}")
    print(f"[verdict        ] {doc['verdict']} — {doc['verdict_why']}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
