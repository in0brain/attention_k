"""G4：production quality check（v2.4 §2.1，**性能盲**）。

    python scripts/run_stage0_quality_check.py --n-prompts 20

小批真实生成,只看**运行质量**,不看任何性能指标。样本取自将来正式 population 之外:
  H1  → dev/test split（confirmatory 用 train）
  MCQ → burned 240（confirmatory 用 fresh 1760）
以免烧掉 confirmatory 样本。

本脚本**不导入** evaluation 包,不计算 AUROC/AUPRC/Δ_H/S/D/ladder。
报告落盘前由 assert_no_performance_fields 强制校验(§2.1)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention import cyber_data as cd  # noqa: E402
from recover_attention import mcq_conditional_increment as mci  # noqa: E402
from recover_attention.data_io import read_jsonl  # noqa: E402
from recover_attention.generation import backends as bk  # noqa: E402
from recover_attention.generation import quality_check as qc  # noqa: E402
from recover_attention.h1_identifier import build_ontology_index  # noqa: E402

K = 6
H1_F5_CONTINUOUS = ("f5_id_logprob_mean", "f5_id_logprob_min", "f5_id_token_entropy_mean",
                    "f5_completion_perplexity", "f5_lengthnorm_logprob")
MCQ_F5_CONTINUOUS = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy",
                     "f5_letter_token_logprob")


def run_h1(args, prereg_sha: str) -> tuple[dict, dict, dict]:
    """H1 小批。样本取 **dev/test** split —— confirmatory 用 train，不烧样本。"""
    pool = [r for r in read_jsonl(Path(args.h1_samples)) if r.get("split") in ("dev", "test")]
    # **按 family 分层**取样。首跑用字母序取前 20,结果是 12 attack + 8 cve + 0 cwe:
    # 既漏掉了 cwe(主 family 之一,完全没验到),又让 CVE 题占了 40% 拉低整体 eligible 率。
    # 质检样本必须覆盖两个主 family,并带少量 CVE 以验证 only_cve 路径。
    quota = {"attack": args.n_prompts * 2 // 5, "cwe": args.n_prompts * 2 // 5}
    quota["cve"] = args.n_prompts - quota["attack"] - quota["cwe"]
    recs = []
    for fam, k in sorted(quota.items()):
        sub = sorted([r for r in pool if r.get("family") == fam], key=lambda r: r["example_id"])
        if len(sub) < k:
            raise SystemExit(f"not enough dev/test prompts for family {fam}: need {k}, have {len(sub)}")
        recs.extend(sub[:k])
    recs.sort(key=lambda r: r["example_id"])
    print(f"[G4:h1] {len(recs)} prompts from dev/test, family-stratified {quota} "
          f"(confirmatory uses train)")
    b = bk.H1Backend(args.model_path, build_ontology_index(Path(args.ontology_dir)),
                     seed=args.seed, max_new_tokens=args.max_new_tokens)
    hv = b.verify_hidden_index()
    print(f"[G4:h1] hidden index: {hv}")
    rows, hidden = [], {}
    for i, r in enumerate(recs, 1):
        for k in range(K):
            st = "greedy" if k == 0 else "sampled"
            uid = f"{r['example_id']}__{st}_{k}"
            try:
                row = b.run_unit({"sample_type": st, "sample_index": k}, r)
            except Exception as exc:                     # noqa: BLE001
                print(f"[G4:h1] unit {uid} failed: {exc!r}")
                continue
            row["trace_id"] = uid
            row["group_id"] = r["group_id"]
            h = row.pop("hidden", None)
            if h is not None:
                hidden[uid] = np.asarray(h, dtype=np.float32)
            rows.append(row)
        print(f"[G4:h1] {i}/{len(recs)} prompts done ({len(rows)} traces)")
    bk.attach_cross_sample_f5(rows, "h1")
    return qc.check_h1_batch(rows, hidden, H1_F5_CONTINUOUS), hv, b.run_fingerprint("h1")


def run_mcq(args, prereg_sha: str) -> tuple[dict, dict, dict]:
    """MCQ 小批。样本取 **burned 240** —— confirmatory 用 fresh 1760，不烧样本。"""
    pool = read_jsonl(Path(args.mcq_pool))
    burned = {json.loads(l)["example_id"]
              for l in open(args.burned_traces, encoding="utf-8")}
    recs = sorted([r for r in pool if r["example_id"] in burned],
                  key=lambda r: r["example_id"])[: args.n_prompts]
    print(f"[G4:mcq] {len(recs)} questions from the burned 240 (confirmatory uses fresh 1760)")
    b = bk.MCQBackend(args.model_path, seed=args.seed, max_new_tokens=args.mcq_max_new_tokens)
    hv = b.verify_hidden_index()
    print(f"[G4:mcq] hidden index: {hv}")
    tok = b.ctx["tokenizer"]
    rows, hidden = [], {}
    for i, r in enumerate(recs, 1):
        r = {**r, "rendered_prompt": tok.apply_chat_template(
            cd.build_mcq_chat_messages(r), tokenize=False, add_generation_prompt=True)}
        try:
            row = b.run_unit({"sample_type": "greedy", "sample_index": 0}, r)
        except Exception as exc:                         # noqa: BLE001
            print(f"[G4:mcq] {r['example_id']} failed: {exc!r}")
            continue
        row["group_id"] = r["group_id"]
        # wrong_label 只用于"两类都有吗"这一可用性判断（§2.1 允许），不做任何判别性度量
        row["wrong_label"] = (None if row.get("parsed_label") is None
                              else int(row["parsed_label"] != r["gold_label"]))
        h = row.pop("hidden", None)
        if h is not None:
            hidden[r["example_id"]] = np.asarray(h, dtype=np.float32)
        rows.append(row)
        if i % 5 == 0 or i == len(recs):
            print(f"[G4:mcq] {i}/{len(recs)}")
    bk.attach_cross_sample_f5(rows, "mcq")
    return qc.check_mcq_batch(rows, hidden, MCQ_F5_CONTINUOUS), hv, b.run_fingerprint("mcq")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="h1,mcq")
    ap.add_argument("--n-prompts", type=int, default=20,
                    help="每臂的 prompt 数（§2.1 上限 40）")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--h1-samples", default="data/processed/h1/h1_samples.jsonl")
    ap.add_argument("--mcq-pool", default="data/processed/cyber/cybermetric.jsonl")
    ap.add_argument("--burned-traces",
                    default="outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/"
                            "trace_sampling_manifest.jsonl")
    ap.add_argument("--ontology-dir", default="data/raw/ontology")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--output-dir", default="outputs/stage0")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--mcq-max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    if args.n_prompts > 40:
        raise SystemExit("§2.1 caps the quality check at 40 prompts per arm")
    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit("STOP (G2): preregistration hash mismatch")

    per_arm, hvs, fps = {}, {}, {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        res, hv, fp = (run_h1 if arm == "h1" else run_mcq)(args, g2["current"])
        per_arm[arm], hvs[arm], fps[arm] = res, hv, fp
        print(f"[G4:{arm}] ok={res['ok']} checks={json.dumps(res['checks'], ensure_ascii=False)}")

    report = qc.build_g4_report(per_arm, hvs, fps, g2["current"])
    out = Path(args.output_dir) / "stage0_g4_quality_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"[G4] passed={report['passed']} -> {out}")
    if not report["passed"]:
        raise SystemExit("G4 FAILED — Stage 0 must not start; see the report")


if __name__ == "__main__":
    main()
