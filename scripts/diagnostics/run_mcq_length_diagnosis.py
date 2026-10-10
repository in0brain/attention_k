r"""MCQ 长度诊断的真实运行（**探索性诊断，不是 gate**）。

复用既有生成路径：记录与 unit 由 `v3_g3_smoke.build_arm_records` / `units_for` 产出，
生成走 `MCQBackend._generate`（带 `capture=` 的纯观测），解析走原 `parse_option_answer`。
**不另写提示或生成逻辑，不读 hidden，不缓存 hidden/attention。**

两个条件：len16 (max_new_tokens=16) → len128 (max_new_tokens=128)，其余全同。

用法：
    python scripts/diagnostics/run_mcq_length_diagnosis.py
    python scripts/diagnostics/run_mcq_length_diagnosis.py --smoke 2   # 只跑前 2 题
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp              # noqa: E402
from recover_attention import mcq_conditional_increment as mci       # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool      # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as bk              # noqa: E402
from recover_attention.generation.manifest import write_atomic       # noqa: E402

MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR = ROOT / "outputs" / "stage1_llama_v3" / "mcq_length_diagnosis"
SAMPLES = ROOT / "outputs" / "stage1" / "samples"
CONDITIONS = (("len16", 16), ("len128", 128))
LETTERS = list(mci.MCQ_LETTERS)


def _rel(p: Path) -> str:
    """仓库内给相对路径，仓库外（自检临时目录）如实给绝对路径。"""
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def versions() -> dict:
    import accelerate, bitsandbytes, torch, transformers   # noqa: PLC0415,E401
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__,
            "python": sys.version.split()[0], "host": platform.node()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 题（自检用）")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verification = g3.verify_inputs(samples_dir=SAMPLES,
                                    prereg=ROOT / "docs/paper/preregistration_v3.md",
                                    lock=ROOT / "docs/paper/preregistration_v3.lock",
                                    cfg=cfg)
    gate_rows = verification.pop("_gate_rows")[g3.ARM_MCQ]
    if args.smoke:
        gate_rows = gate_rows[:args.smoke]
    print(f"[G2   ] match={verification['g2']['match']} "
          f"sha={verification['g2']['current_sha256'][:12]}…")

    geometry = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=MODEL)
    print(f"[geom ] block={geometry.block_index} tuple={geometry.tuple_index} "
          f"dim={geometry.hidden_dim} source={geometry.source}  "
          f"(仅记录；本轮不读 hidden)")

    backend = bk.MCQBackend(MODEL, seed=cfg.generation_seed,
                            max_new_tokens=cfg.mcq_max_new_tokens,
                            temperature=0.7, top_p=0.95)
    tok = backend.ctx["tokenizer"]
    records = g3.build_arm_records(g3.ARM_MCQ, gate_rows, cfg=cfg, tokenizer=tok)
    units = g3.units_for(records, cfg.k_traces)
    by_id = {r["example_id"]: r for r in records}
    print(f"[mode ] REAL  model={MODEL}  records={len(records)} units={len(units)}")

    rows_path = out_dir / "mcq_length_diagnosis_rows.jsonl"
    buf: list[dict] = []
    t0 = time.time()
    n_done = 0
    for cond_name, budget in CONDITIONS:
        backend.max_new_tokens = budget          # 唯一的条件差异
        for unit in units:
            rec = by_id[unit["example_id"]]
            seed = bk._stable_seed(backend.seed, rec["example_id"],
                                   unit["sample_type"], unit["sample_index"])
            cap: dict = {}
            err = None
            try:
                completion = backend._generate(
                    rec["rendered_prompt"],
                    do_sample=unit["sample_type"] == "sampled",
                    seed=seed, capture=cap)
            except Exception as exc:             # noqa: BLE001
                completion, err = None, f"{type(exc).__name__}: {exc}"
            parsed = (dlp.parse_option_answer(completion, LETTERS)
                      if isinstance(completion, str) else
                      {"parsed_label": None, "parse_method": "generation_error",
                       "parse_failure": True})
            span = None
            if parsed.get("parsed_label"):
                try:
                    cs, ce = bk._parsed_letter_char_span(completion, parsed["parsed_label"])
                    span = {"char_start": cs, "char_end": ce,
                            "char": completion[cs:ce],
                            "context": completion[max(0, cs - 40):ce + 15]}
                except ValueError as exc:
                    span = {"locate_error": str(exc)[:120]}
            pids = cap.get("prompt_input_ids", [])
            buf.append({
                "schema": "stage1_mcq_length_diagnosis_row_v1",
                "condition": cond_name,
                "max_new_tokens": budget,
                "example_id": rec["example_id"],
                "unit_id": f'{rec["example_id"]}__{unit["sample_type"]}_{unit["sample_index"]}',
                "sample_type": unit["sample_type"],
                "sample_index": unit["sample_index"],
                "seed": seed,
                "effective_generate_kwargs": cap.get("effective_generate_kwargs"),
                "versions": VERSIONS,
                "prompt_n_tokens": cap.get("prompt_len"),
                "prompt_first_token_id": (pids[0] if pids else None),
                "prompt_n_bos_128000": sum(1 for i in pids if i == 128000),
                "generated_ids_raw": cap.get("generated_ids_raw"),
                "n_generated_tokens": cap.get("n_generated_tokens"),
                "last_generated_token_id": cap.get("last_generated_token_id"),
                "last_generated_token_decoded": (
                    tok.decode([cap["last_generated_token_id"]])
                    if cap.get("last_generated_token_id") is not None else None),
                "hit_passed_eos": cap.get("last_token_is_the_eos_we_passed"),
                "reached_max_new_tokens": cap.get("reached_max_new_tokens"),
                "eos_token_id_passed": cap.get("eos_token_id"),
                "pad_token_id_effective": cap.get("pad_token_id"),
                "other_eos_ids_present_in_output": sorted(
                    {i for i in (cap.get("generated_ids_raw") or [])
                     if i in (128001, 128008, 128009)}),
                "raw_completion": completion,
                "decoded_keep_special_tokens": cap.get("decoded_skip_special_false"),
                "parse_status": parsed["parse_method"],
                "parsed_label": parsed["parsed_label"],
                "parsed_span": span,
                "error": err,
            })
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t0
                print(f"[{cond_name}] {n_done}/{len(units) * 2} elapsed={el:.0f}s "
                      f"{el / n_done:.2f}s/trace", flush=True)
    write_atomic(rows_path, "\n".join(json.dumps(r, ensure_ascii=False) for r in buf) + "\n")

    meta = {
        "schema": "stage1_mcq_length_diagnosis_meta_v1",
        "date": "2026-09-14",
        "card": "docs/codex_tasks/sprint_stage1_llama_mcq_length_diagnosis.md",
        "boundary": ("**探索性诊断，不是 gate**。不替换 outputs/stage1_llama_v3/g3_postfix/ 的"
                     "原结果，不构成原协议 G3 通过，不得作为 G3/G4/生产输入。"
                     "即使 seed 相同，这两个条件的生成都是**新证据**，"
                     "不是旧运行原文的恢复。"),
        "environment": {**VERSIONS, "backend": dict(bk.BACKEND_8BIT), "model_path": MODEL,
                        "interpreter": sys.executable},
        "geometry_recorded_only": {"rel_depth": geometry.rel_depth,
                                   "block": geometry.block_index,
                                   "tuple": geometry.tuple_index,
                                   "hidden_dim": geometry.hidden_dim,
                                   "note": "本轮不读 hidden，不缓存 hidden/attention"},
        "sample_manifest_sha256": verification["sample_manifest_sha256"],
        "n_questions": len(records), "k_traces": cfg.k_traces,
        "conditions": [{"name": n, "max_new_tokens": b} for n, b in CONDITIONS],
        "n_rows": len(buf),
        "elapsed_seconds": round(time.time() - t0, 1),
        "rows_path": _rel(rows_path),
        "smoke": args.smoke or None,
    }
    write_atomic(out_dir / "mcq_length_diagnosis_meta.json",
                 json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"[done ] {len(buf)} rows in {meta['elapsed_seconds']}s → {rows_path}")
    return 0


VERSIONS = versions()

if __name__ == "__main__":
    raise SystemExit(main())
