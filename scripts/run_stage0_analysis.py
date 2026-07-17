"""Stage 0 分析编排器（W0.5-E：production evaluation pipeline）。

    python scripts/run_stage0_analysis.py --input-dir outputs/stage0/features \
        --config configs/stage0.yaml
    python scripts/run_stage0_analysis.py --dry-run

不调用模型、不生成数据。--dry-run 用合成 + pilot 缓存把整条链跑通,证明
**G1 一开门就能按按钮**,而不是"代码还没写"。

链路（§6 冻结顺序:generate → ladder → gate → increment → rq2）:

    feature_cache（准入校验:pilot 隔离 / F5 completeness / backend invariant）
      → ladder（5 folds, nested C, O/H/O+H 同 folds）
      → artifact redline（在 hidden 增量声明之前）
      → observability gate（D = S_MCQ − S_H1）
      → increment（Δ_H CI + §7.2 精度承诺）
      → rq2（fold-specific 分层）
      → stage0_analysis_report.json
"""

from __future__ import annotations

import argparse
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
from recover_attention.evaluation import artifact_redline as ar  # noqa: E402
from recover_attention.evaluation import increment as inc  # noqa: E402
from recover_attention.evaluation import ladder as ld  # noqa: E402
from recover_attention.evaluation import observability_gate as og  # noqa: E402
from recover_attention.evaluation import rq2 as rq2m  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import (  # noqa: E402
    ARM_H1,
    ARM_MCQ,
    check_backend_invariant,
    load_arm,
)

REPORT_SCHEMA = "stage0_analysis_report_v1"


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def run_analysis(arms: dict[str, Any], config: Stage0Config, prereg: dict) -> dict:
    """完整分析链。arms = {"mcq": ArmCache, "h1": ArmCache}。"""
    bk = check_backend_invariant(arms)
    if not bk["ok"]:
        raise SystemExit(f"STOP: backend invariant violated — {bk}")

    ladders = {a: ld.run_ladder(c, config) for a, c in arms.items()}
    # §7:artifact 红线在 hidden 增量声明之前
    redlines = {a: ar.compute_artifact_redline(arms[a], ladders[a], config) for a in arms}
    # §6:gate 在 increment 之前
    gate = og.compute_gate(arms[ARM_MCQ], ladders[ARM_MCQ], arms[ARM_H1], ladders[ARM_H1],
                           config, bk)
    increments = {a: inc.compute_increment(arms[a], ladders[a], config) for a in arms}
    rq2 = {a: rq2m.compute_rq2(arms[a], ladders[a], config) for a in arms}

    # 红线触发的臂不得出增量声明（§7）
    for a in arms:
        if redlines[a]["redline_triggered"]:
            increments[a]["hidden_increment_claim_allowed"] = False
            increments[a]["suppressed_reason"] = (
                f"artifact redline triggered by {redlines[a]['triggered_by']} — the task is "
                f"solved by string/format patterns; no hidden-increment claim (§7)")
        else:
            increments[a]["hidden_increment_claim_allowed"] = True

    return {
        "schema_version": REPORT_SCHEMA,
        "preregistration": prereg,
        "config": config.to_jsonable(),
        "backend_invariant": bk,
        "arm_validation": {a: c.validation for a, c in arms.items()},
        "ladder": {a: {"n": l["n"], "n_positive": l["n_positive"], "n_groups": l["n_groups"],
                       "rungs": ld.ladder_table(l),
                       "chosen_C": {k: v["chosen_C"] for k, v in l["rungs"].items()},
                       "same_outer_folds_for_all_rungs": l["same_outer_folds_for_all_rungs"]}
                   for a, l in ladders.items()},
        "artifact_redline": redlines,
        "observability_gate": gate,
        "increment": increments,
        "rq2": rq2,
        "order": "ladder -> artifact_redline -> observability_gate -> increment -> rq2 (§6/§7)",
    }


# --------------------------------------------------------------------- dry run
def _noisy_text(wrong: int, rng, leak: float = 0.65) -> str:
    """合成文本：只带**部分**信号，绝不能编码标签。

    第一版把 "fabricated bogus..." / "grounded correct..." 直接按标签写死,导致
    rung4 的 AUROC = 1.000 → O=1.0、O+H=1.0、Δ≡0、CI 宽度 0、S 双双 1.0、D=0。
    那样的 dry-run 什么都证明不了:bootstrap 即使完全坏掉,两个相同的分数向量照样给 [0,0]。
    这里让词表以 `leak` 的概率与标签相关,其余随机 —— 文本可分但远非完美。
    """
    pos_words = ["anomalous", "unverified", "speculative", "novel"]
    neg_words = ["documented", "catalogued", "standard", "known"]
    pool = (pos_words if wrong else neg_words) if rng.random() < leak else (neg_words + pos_words)
    picked = [str(pool[int(rng.integers(0, len(pool)))]) for _ in range(3)]
    filler = [f"term{int(rng.integers(0, 12))}" for _ in range(4)]
    return " ".join(picked + filler)


def _synthetic_arm(arm: str, n_groups: int, per_group: int, pos_rate: float,
                   seed: int, hidden_dim: int = 64):
    """合成一个 feature cache（**只为验证接线**，不产生任何可解读的数值）。

    数据必须让整条链**真的动起来**:
      - 文本只带部分信号（见 _noisy_text）—— 否则 rung4=1.0 让 O 饱和;
      - F5 与文本相关但不同源;
      - H 带**独立于 O** 的增量 —— 否则 Δ≡0、CI 宽度 0,bootstrap 是空转的;
      - 标签按 group 聚集（4D-1 实测形态 beta(0.42, 2.41),ICC≈0.26）。
    维度刻意用 64 而非 3584:接线与维度无关,3584 只会让 dry-run 变慢。
    """
    from recover_attention import mcq_conditional_increment as mci
    from recover_attention.evaluation.feature_cache import ArmCache

    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    groups = np.repeat([f"{arm}_g{i}" for i in range(n_groups)], per_group)
    p = rng.beta(0.42, 2.41, n_groups)                      # 标签按 prompt 聚集（4D-1 实测形态）
    y = (rng.random((n_groups, per_group)) < p[:, None]).astype(int).ravel()
    if len(np.unique(y)) < 2:
        y[0] = 1 - y[0]

    recs: list[dict] = []
    for i in range(n):
        wrong = int(y[i])
        base = {"example_id": f"{arm}_{i:05d}", "group_id": groups[i],
                "population_role": "synthetic_dry_run"}
        text = _noisy_text(wrong, rng)
        if arm == ARM_MCQ:
            choices = [{"choice": c, "label_text": f"option {c} text {rng.integers(0, 9)}",
                        "original_position": j} for j, c in enumerate(mci.MCQ_LETTERS)]
            letter = mci.MCQ_LETTERS[int(rng.integers(0, 4))]
            recs.append({**base, "eligible_for_primary": True, "wrong_label": wrong,
                         "parsed_label": letter, "candidate_choices": choices,
                         "prompt_only_text": f"question {i} with several words here",
                         "semantic_output_text": text,
                         "f5_label_margin": float(-wrong * 1.1 + rng.normal(0, 1)),
                         "f5_label_entropy": float(abs(rng.normal(0.3, 0.1))),
                         "f5_full_entropy": float(abs(rng.normal(0.3, 0.1))),
                         "f5_letter_token_logprob": float(-abs(rng.normal(0.2, 0.2))),
                         "f5_self_consistency_exact": float(rng.uniform(0.2, 1.0)),
                         "f5_letter_agreement_rate": float(rng.uniform(0.2, 1.0)),
                         **mci.mcq_surface_features(text, choices, letter)})
        else:
            recs.append({**base, "eligible": True, "label": wrong,
                         "prompt_text": f"prompt {i} with several words here",
                         "completion": text, "id_string_text": "T1114.003 CWE-79",
                         "f5_id_logprob_mean": float(-wrong * 0.8 + rng.normal(0, 1)),
                         "f5_id_logprob_min": float(rng.normal(-2, 1)),
                         "f5_id_token_entropy_mean": float(abs(rng.normal(0.5, 0.2))),
                         "f5_first_id_token_rank": float(rng.integers(1, 5)),
                         "f5_id_token_count": float(rng.integers(1, 6)),
                         "f5_completion_perplexity": float(abs(rng.normal(3, 1))),
                         "f5_lengthnorm_logprob": float(rng.normal(-1, 0.5)),
                         "f5_id_token_ratio": float(rng.uniform(0.01, 0.3)),
                         "f5_completion_token_count": float(rng.integers(20, 200)),
                         "f5_self_consistency_exact": float(rng.uniform(0.2, 1.0)),
                         "f5_id_agreement_rate": float(rng.uniform(0.2, 1.0)),
                         "f5_confidence_high": float(rng.integers(0, 2)),
                         "f5_confidence_medium": float(rng.integers(0, 2)),
                         "f5_confidence_low": float(rng.integers(0, 2)),
                         **ci.surface_format_features(text, [
                             {"family": "attack", "label": "fabricated" if wrong else "grounded",
                              "normalized": "T1114", "start": 0, "granularity": "technique"}])})
    # H 带**独立于 O** 的增量:只有前 8 维载信号,其余是噪声(近似真实 hidden 的稀疏可分性)。
    # 若 H 与 O 同源,Δ≡0、CI 宽度 0 —— bootstrap 就没被真正执行过。
    hidden = rng.normal(0, 1, (n, hidden_dim))
    hidden[:, :8] += y.reshape(-1, 1) * 0.9
    backend = {"load_in_8bit": True, "device_map": "auto",
               "attn_implementation": "eager", "local_files_only": True}
    validation = {"arm": arm, "n_eligible": n, "n_positive": int(y.sum()),
                  "n_groups": n_groups, "synthetic": True,
                  "population_roles": ["synthetic_dry_run"]}
    return ArmCache(arm=arm, records=recs, hidden=hidden, y=y, groups=groups,
                    backend=backend, validation=validation)


def dry_run(args) -> dict:
    """不碰模型、不碰真数据:合成 cache 跑通整条正式链路。

    产出 analysis_pipeline_ready —— 用来把"缺 G1"和"缺代码"分开。
    """
    config = load_config(args.config) if args.config else Stage0Config()
    # dry-run 用正式规格（5 folds / bootstrap>=1000），只是数据是合成的
    config = Stage0Config(n_outer_folds=config.n_outer_folds, n_inner_folds=config.n_inner_folds,
                          n_boot=config.n_boot, seed=config.seed, mode="dry_run",
                          allow_pilot_population=True)
    prereg = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not prereg["match"]:
        raise SystemExit("STOP (G2): preregistration hash mismatch")

    arms = {
        ARM_MCQ: _synthetic_arm(ARM_MCQ, n_groups=400, per_group=1, pos_rate=0.20,
                                seed=args.seed + 1),
        ARM_H1: _synthetic_arm(ARM_H1, n_groups=120, per_group=6, pos_rate=0.15,
                               seed=args.seed + 2),
    }
    print(f"[dry-run] synthetic arms: "
          + ", ".join(f"{a}(n={c.y.size}, pos={int(c.y.sum())}, groups={len(set(c.groups))})"
                      for a, c in arms.items()))
    report = run_analysis(arms, config, {"sha256": prereg["current"], "match": prereg["match"]})
    report["mode"] = "dry_run"
    report["data_is_synthetic"] = True
    report["interpretability"] = {
        "auroc_interpretable": False,
        "note": ("synthetic feature cache — this run proves the analysis pipeline executes end to "
                 "end at production spec. No number here means anything about the research "
                 "question."),
    }
    report["analysis_pipeline_ready"] = True
    report["stage0_generation_required"] = False
    out = Path(args.output_dir) / "stage0_analysis_dry_run.json"
    _write(out, report)

    print(f"[dry-run] ladder rungs: {sorted(report['ladder'][ARM_H1]['rungs'])}")
    print(f"[dry-run] gate verdict={report['observability_gate']['verdict']} "
          f"D={report['observability_gate']['D']['point']:+.3f}")
    for a in arms:
        i = report["increment"][a]
        print(f"[dry-run] {a}: read={i['read']} precision={i['precision']['precision']} "
              f"ci_width={i['precision']['ci_width']:.4f} "
              f"redline={report['artifact_redline'][a]['redline_triggered']}")
    print(f"[dry-run] analysis_pipeline_ready=true  stage0_generation_required=false  -> {out}")
    return report


def production(args) -> dict:
    config = load_config(args.config)
    if config.mode != "production":
        raise SystemExit(f"config mode is {config.mode!r}; use --dry-run for dry runs")
    prereg = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not prereg["match"]:
        raise SystemExit("STOP (G2): preregistration hash mismatch")

    in_dir = Path(args.input_dir)
    arms = {
        ARM_MCQ: load_arm(ARM_MCQ, in_dir / "mcq_completion_records.jsonl",
                          in_dir / "mcq_hidden.npz", config, label_field="wrong_label"),
        ARM_H1: load_arm(ARM_H1, in_dir / "h1_completion_records.jsonl",
                         in_dir / "h1_hidden.npz", config, label_field="label"),
    }
    report = run_analysis(arms, config, {"sha256": prereg["current"], "match": prereg["match"]})
    report["mode"] = "production"
    out = Path(args.output_dir) / "stage0_analysis_report.json"
    _write(out, report)
    print(f"[stage0] report -> {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="合成 cache 跑通整条正式链路;不调模型、不读真数据")
    ap.add_argument("--input-dir", default="outputs/stage0/features")
    ap.add_argument("--output-dir", default="outputs/stage0/evaluation")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (dry_run if args.dry_run else production)(args)


if __name__ == "__main__":
    main()
