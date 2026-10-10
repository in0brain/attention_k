"""POST-HOC：Llama 跨模型复核的**可比性审计**。

**非 confirmatory。** 不改动、不覆盖 stage0_analysis_report.json、freeze manifest 或
unblinding log。本脚本只读两个模型已落盘的 feature cache 与审计 JSON。

────────────────────────────────────────────────────────────────────────────
它回答什么
────────────────────────────────────────────────────────────────────────────
Qwen 与 Llama 的 Δ_H 落在同一区间，**不构成**"跨模型复现"的证据，除非先排除
一件事：两个模型分析的根本不是同一批样本。两臂的 eligible 集合由**模型自己的
生成内容**决定（completion 里有没有 ATT&CK/CWE 标识符），所以：

    同样 480 个 prompt × K=6  →  Qwen 2016 条 eligible / Llama 2027 条 eligible
    但这两个 2000 条不是同一批。

本审计把"数字接近"与"在同一批样本上数字接近"分开，后者才是复现。

────────────────────────────────────────────────────────────────────────────
五件事，逐件落盘
────────────────────────────────────────────────────────────────────────────
1. **溯源**：每个输入文件的 sha256、行数、字节数；跑过的命令；判定规则的代码位置。
   本脚本的所有数字**从 JSON / cache 读**，不从任何文字总结抄。
2. **inconclusive 触发条件**：把 Llama C 的 read 逐条件重算，指明是 §4 判读规则
   还是 §7.2 精度承诺（或两者）触发，给出字段名与阈值。
   **不**按"CI 是否跨零"重新解释。
3. **共同样本审计**：eligible 交集/差集、逐样本排除原因对照，并在共同样本上重算
   A / C 两种分组的 Δ、CI、判定。**标为 sensitivity analysis**，不替换任何原始结果。
4. **跨模型审计表**：model × grouping × estimand × Δ/CI/decision × run_role ×
   confirmatory status。**不做 pooled estimate，不并排比较裸 Δ_H 作为结论。**
5. **rung 比较登记表**：预注册比较（§7 红线：full-text vs shortcut）与事后比较
   （O vs 最强 rung）分开列。后者在两个模型上都是 post-hoc，且"最强"是在 5 个
   对手里事后选的，无多重性控制 → 一律标 exploratory。

用法：
    python scripts/posthoc_cross_model_comparability_audit.py \
        --qwen-dir outputs/stage0/features \
        --llama-dir outputs/stage0_llama_h1/features \
        --timestamp <ISO8601>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation import bootstrap as bs  # noqa: E402
from recover_attention.evaluation import increment as inc  # noqa: E402
from recover_attention.evaluation import ladder as ld  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_H1, load_arm  # noqa: E402

# 复用泄漏诊断里**同一个** union-find，不另写一份：C 分组的定义必须与
# posthoc_h1_identifier_leakage_diagnostic.py 逐字一致，否则"重算"没有意义。
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from posthoc_h1_identifier_leakage_diagnostic import component_groups  # noqa: E402


# ───────────────────────────────────────────────────────────── 1. 溯源
def file_provenance(path: Path) -> dict:
    """sha256 + 大小 + 行数（文本文件）。哈希是判断"我读的确实是那批产物"的唯一凭据。"""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    n_lines = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n_lines += chunk.count(b"\n")
    return {"path": str(p).replace("\\", "/"), "exists": True,
            "sha256": h.hexdigest(), "bytes": p.stat().st_size,
            "n_lines": n_lines if p.suffix in (".jsonl", ".md", ".log") else None}


DECISION_RULES = {
    "delta_H_read": {
        "source": "docs/paper/preregistration.md §4 (frozen v2.4, sha256 7c051d35…)",
        "code": "src/recover_attention/conditional_increment.py::equivalence_read",
        "eps_auroc": ci.EPS_AUROC,
        "rule": [
            "ci_lo >  +eps            -> 'increment'",
            "ci_hi <  -eps            -> 'harmful'",
            "ci_lo >= -eps AND ci_hi <= +eps -> 'equivalent'",
            "otherwise                -> 'inconclusive'",
        ],
        "note": "判读不看点估计、不看 CI 是否跨零；只看 CI 两端相对 ±eps 的位置。",
    },
    "precision_promise": {
        "source": "docs/paper/preregistration.md §7.2 (v2.3 P0, frozen)",
        "code": "src/recover_attention/evaluation/bootstrap.py::precision_verdict",
        "applied_in": "src/recover_attention/evaluation/bootstrap.py::paired_delta",
        "threshold_field": "Stage0Config.precision_max_ci_width",
        "rule": [
            "ci_width = ci_hi - ci_lo",
            "ci_width <= precision_max_ci_width -> precision='adequate'",
            "ci_width >  precision_max_ci_width -> precision='insufficient'",
            "precision=='insufficient' -> read is OVERRIDDEN to 'inconclusive'; "
            "the pre-override value is preserved in field 'read_before_precision_check'",
        ],
        "note": "§7.2 明文禁止事后追加样本、改 eps、换判读规则或改融合协议。",
    },
    "artifact_redline_preregistered": {
        "source": "docs/paper/preregistration.md §7",
        "estimand": "Delta_artifact = AUROC(rung4 full-text) - AUROC(shortcut)",
        "shortcuts": ["rung2_id_string_only", "rung3_surface_only"],
        "frozen_rule": "trigger iff CI95(Delta_artifact) subset of [-eps, +eps]",
        "note": "预注册比较的参照是 **rung4 full-text**，不是 O(rung6)。",
    },
    "o_vs_strongest_rung": {
        "source": "NOT preregistered — introduced by the 2026-07-22 post-hoc audit",
        "code": "scripts/posthoc_redline_rule_and_o_dominance.py",
        "rule": "O_is_dominated iff exists rival with CI_hi(AUROC(O) - AUROC(rival)) < 0",
        "status": "post-hoc on Qwen; post-hoc on Llama; "
                  "'strongest rival' is selected after seeing all 5 rivals, "
                  "with no multiplicity control -> exploratory only",
    },
}


# ───────────────────────────────────────────── 2. inconclusive 触发条件
def explain_read(ci_lo: float, ci_hi: float, config: Stage0Config) -> dict:
    """把一个 (ci_lo, ci_hi) 的 read 逐条件拆开，指明谁触发了什么。

    独立于产出该数字的脚本重算一遍——若与 JSON 里记录的 read 不一致，说明有一方错了，
    这本身就是要报的发现。
    """
    eps = config.eps_auroc
    width = ci_hi - ci_lo
    raw = ci.equivalence_read(ci_lo, ci_hi, eps=eps)
    prec = bs.precision_verdict(ci_lo, ci_hi, config)
    final = "inconclusive" if prec["precision"] == bs.PRECISION_INSUFFICIENT else raw
    conditions = [
        {"condition": "ci_lo > +eps  (=> 'increment')",
         "values": f"{ci_lo:+.4f} > {eps:+.4f}", "holds": bool(ci_lo > eps)},
        {"condition": "ci_hi < -eps  (=> 'harmful')",
         "values": f"{ci_hi:+.4f} < {-eps:+.4f}", "holds": bool(ci_hi < -eps)},
        {"condition": "ci_lo >= -eps AND ci_hi <= +eps  (=> 'equivalent')",
         "values": f"{ci_lo:+.4f} >= {-eps:+.4f} AND {ci_hi:+.4f} <= {eps:+.4f}",
         "holds": bool(ci_lo >= -eps and ci_hi <= eps)},
        {"condition": "§7.2 ci_width <= precision_max_ci_width  (=> precision adequate)",
         "values": f"{width:.4f} <= {config.precision_max_ci_width:.4f}",
         "holds": bool(width <= config.precision_max_ci_width)},
    ]
    triggers = []
    if raw == "inconclusive":
        triggers.append(
            "§4 equivalence_read: CI straddles the +eps boundary — ci_lo does NOT exceed "
            f"+eps ({ci_lo:+.4f} <= {eps:+.4f}) so it is not 'increment', and ci_hi exceeds "
            f"+eps ({ci_hi:+.4f} > {eps:+.4f}) so it is not 'equivalent'.")
    if prec["precision"] == bs.PRECISION_INSUFFICIENT:
        triggers.append(
            f"§7.2 precision promise: ci_width={width:.4f} > "
            f"precision_max_ci_width={config.precision_max_ci_width:.4f} — read is forced "
            f"to 'inconclusive' regardless of the §4 verdict.")
    return {
        "ci_lo": ci_lo, "ci_hi": ci_hi, "ci_width": width, "eps": eps,
        "precision_max_ci_width": config.precision_max_ci_width,
        "conditions_evaluated": conditions,
        "read_from_equivalence_rule_alone": raw,
        "precision_verdict": prec["precision"],
        "final_read": final,
        "triggering_conditions": triggers,
        "explicitly_not_the_reason": (
            "CI crossing zero is NOT part of either rule. Here ci_lo is "
            f"{ci_lo:+.4f}; sign of ci_lo does not enter equivalence_read or "
            "precision_verdict."),
    }


# ───────────────────────────────────────────── 3. 共同样本审计
def unit_index(records: list[dict]) -> dict[str, dict]:
    return {r["trace_id"]: r for r in records}


def exclusion_crosstab(q_recs: list[dict], l_recs: list[dict]) -> dict:
    """逐 trace 对照两模型的 eligible / 排除原因。trace_id 由 prompt 与采样位决定，
    与模型无关，所以是合法的连接键（已核验两侧首行一致）。"""
    qi, li = unit_index(q_recs), unit_index(l_recs)
    common = sorted(set(qi) & set(li))
    cell = collections.Counter()
    for t in common:
        q, l = qi[t], li[t]
        qr = "eligible" if q.get("eligible") else (q.get("primary_exclusion_reason") or "?")
        lr = "eligible" if l.get("eligible") else (l.get("primary_exclusion_reason") or "?")
        cell[(qr, lr)] += 1
    return {
        "n_traces_qwen": len(qi), "n_traces_llama": len(li),
        "n_traces_common_id_space": len(common),
        "trace_id_space_identical": bool(set(qi) == set(li)),
        "crosstab_qwen_by_llama": {f"qwen={a} | llama={b}": n
                                   for (a, b), n in sorted(cell.items(), key=lambda kv: -kv[1])},
    }


def subset_cache(cache, keep_mask: np.ndarray):
    idx = np.flatnonzero(keep_mask)
    return replace(cache,
                   records=[cache.records[i] for i in idx],
                   hidden=cache.hidden[idx],
                   y=cache.y[idx],
                   groups=cache.groups[idx])


def run_cell(cache, config: Stage0Config, grouping: str) -> dict:
    """在给定 cache 上跑 A 或 C 分组的 ladder + increment。

    C 分组的连通分量在**当前子集内**重新计算——分量由子集内的共现关系定义，
    拿全集的分量号套子集会得到跨折泄漏未被封死的假分组。
    """
    if grouping == "A_prompt":
        c = cache
    elif grouping == "C_prompt_union_fabid":
        g, _ = component_groups(cache.records, link_prompt=True, link_fabricated=True)
        c = replace(cache, groups=g)
    else:
        raise ValueError(grouping)
    lad = ld.run_ladder(c, config)
    incr = inc.compute_increment(c, lad, config)
    d = incr["delta_H"]
    return {
        "grouping": grouping,
        "n": int(lad["n"]), "n_positive": int(lad["n_positive"]),
        "n_groups": int(lad["n_groups"]),
        "rungs": {k: v["auroc"] for k, v in lad["rungs"].items()},
        "auroc": incr["auroc"],
        "delta_H": {"point": d["point"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]},
        "read": incr["read"],
        "read_before_precision_check": incr.get("read_before_precision_check"),
        "precision": incr["precision"]["precision"],
        "ci_width": d["ci_hi"] - d["ci_lo"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", default="outputs/stage0/features")
    ap.add_argument("--llama-dir", default="outputs/stage0_llama_h1/features")
    ap.add_argument("--qwen-eval", default="outputs/stage0/evaluation")
    ap.add_argument("--llama-eval", default="outputs/stage0_llama_h1/evaluation")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--output",
                    default="outputs/stage0_llama_h1/evaluation/"
                            "posthoc_cross_model_comparability_audit.json")
    args = ap.parse_args()

    cfg0 = load_config(args.config)
    config = Stage0Config(n_outer_folds=cfg0.n_outer_folds, n_inner_folds=cfg0.n_inner_folds,
                          n_boot=cfg0.n_boot, seed=cfg0.seed, mode=cfg0.mode,
                          allow_pilot_population=False)

    qd, lld = Path(args.qwen_dir), Path(args.llama_dir)
    qe, lle = Path(args.qwen_eval), Path(args.llama_eval)

    # ── 1. 溯源 ────────────────────────────────────────────────────────
    print("[audit] 1/5 provenance ...")
    inputs = {
        "qwen_records": file_provenance(qd / "h1_completion_records.jsonl"),
        "qwen_hidden": file_provenance(qd / "h1_hidden.npz"),
        "qwen_fingerprint": file_provenance(qd / "h1_run_fingerprint.json"),
        "qwen_leakage_json": file_provenance(qe / "posthoc_h1_identifier_leakage_diagnostic.json"),
        "qwen_redline_json": file_provenance(qe / "posthoc_redline_rule_and_o_dominance.json"),
        "qwen_confirmatory_report": file_provenance(qe / "stage0_analysis_report.json"),
        "llama_records": file_provenance(lld / "h1_completion_records.jsonl"),
        "llama_hidden": file_provenance(lld / "h1_hidden.npz"),
        "llama_fingerprint": file_provenance(lld / "h1_run_fingerprint.json"),
        "llama_leakage_json": file_provenance(lle / "posthoc_h1_identifier_leakage_diagnostic.json"),
        "llama_redline_json": file_provenance(lle / "posthoc_redline_rule_and_o_dominance.json"),
        "llama_textrep_json": file_provenance(lle / "posthoc_h1_text_representation_diagnostic.json"),
        "llama_sizing_json": file_provenance(lle / "posthoc_identifier_stratified_sizing.json"),
        "preregistration": file_provenance(PROJECT_ROOT / "docs/paper/preregistration.md"),
        "preregistration_lock": file_provenance(PROJECT_ROOT / "docs/paper/preregistration.lock"),
    }

    def load_json(p):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    q_leak = load_json(qe / "posthoc_h1_identifier_leakage_diagnostic.json")
    l_leak = load_json(lle / "posthoc_h1_identifier_leakage_diagnostic.json")
    q_red = load_json(qe / "posthoc_redline_rule_and_o_dominance.json")
    l_red = load_json(lle / "posthoc_redline_rule_and_o_dominance.json")
    q_fp = load_json(qd / "h1_run_fingerprint.json")
    l_fp = load_json(lld / "h1_run_fingerprint.json")

    # ── 2. inconclusive 触发条件 ───────────────────────────────────────
    print("[audit] 2/5 decision-rule trace ...")
    key_A = "A_prompt_grouped_confirmatory_design"
    key_C = "C_prompt_union_fabricated_grouped"
    if key_C not in l_leak:
        key_C = next(k for k in l_leak if k.startswith("C_"))
    read_trace = {}
    for model, blob in (("qwen", q_leak), ("llama", l_leak)):
        for label, key in (("A_prompt", key_A), ("C_prompt_union_fabid", key_C)):
            s = blob[key] if key in blob else blob[key]["summary"]
            s = s.get("summary", s)
            d = s["delta_H"]
            expl = explain_read(d["ci_lo"], d["ci_hi"], config)
            expl["read_recorded_in_json"] = s["read"]
            expl["precision_recorded_in_json"] = s["precision"]
            expl["recomputation_agrees_with_json"] = bool(
                expl["final_read"] == s["read"]
                and expl["precision_verdict"] == s["precision"])
            expl["delta_point"] = d["point"]
            expl["n_groups"] = s["n_groups"]
            read_trace[f"{model}/{label}"] = expl
            print(f"    {model}/{label}: read={s['read']} "
                  f"(recompute agrees={expl['recomputation_agrees_with_json']})")

    # ── 3. 共同样本审计 ───────────────────────────────────────────────
    print("[audit] 3/5 common-sample audit ...")
    q_all = [json.loads(l) for l in (qd / "h1_completion_records.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    l_all = [json.loads(l) for l in (lld / "h1_completion_records.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    xtab = exclusion_crosstab(q_all, l_all)

    q_elig = {r["trace_id"] for r in q_all if r.get("eligible")}
    l_elig = {r["trace_id"] for r in l_all if r.get("eligible")}
    q_gid = {r["trace_id"]: r["group_id"] for r in q_all}
    common_traces = q_elig & l_elig
    q_groups_e = {q_gid[t] for t in q_elig}
    l_groups_e = {q_gid[t] for t in l_elig}
    common_groups = q_groups_e & l_groups_e

    sets = {
        "qwen_eligible_traces": len(q_elig),
        "llama_eligible_traces": len(l_elig),
        "intersection_traces": len(common_traces),
        "qwen_only_traces": len(q_elig - l_elig),
        "llama_only_traces": len(l_elig - q_elig),
        "jaccard_traces": round(len(common_traces) / len(q_elig | l_elig), 4),
        "qwen_eligible_groups": len(q_groups_e),
        "llama_eligible_groups": len(l_groups_e),
        "intersection_groups": len(common_groups),
        "qwen_only_groups": len(q_groups_e - l_groups_e),
        "llama_only_groups": len(l_groups_e - q_groups_e),
        "jaccard_groups": round(len(common_groups) / len(q_groups_e | l_groups_e), 4),
    }
    for k, v in sets.items():
        print(f"    {k}: {v}")

    q_cache = load_arm(ARM_H1, qd / "h1_completion_records.jsonl", qd / "h1_hidden.npz",
                       config, label_field="label", id_field="trace_id")
    l_cache = load_arm(ARM_H1, lld / "h1_completion_records.jsonl", lld / "h1_hidden.npz",
                       config, label_field="label", id_field="trace_id")

    sensitivity: dict = {}
    for defn, keep_q, keep_l in (
        ("S1_identical_trace_set",
         np.array([r["trace_id"] in common_traces for r in q_cache.records]),
         np.array([r["trace_id"] in common_traces for r in l_cache.records])),
        ("S2_common_group_set_own_traces",
         np.array([r["group_id"] in common_groups for r in q_cache.records]),
         np.array([r["group_id"] in common_groups for r in l_cache.records])),
    ):
        sensitivity[defn] = {}
        for model, cache, mask in (("qwen", q_cache, keep_q), ("llama", l_cache, keep_l)):
            sub = subset_cache(cache, mask)
            if int(sub.y.sum()) < 2 or int((1 - sub.y).sum()) < 2:
                sensitivity[defn][model] = {"skipped": "degenerate label distribution"}
                continue
            cells = {}
            for grouping in ("A_prompt", "C_prompt_union_fabid"):
                r = run_cell(sub, config, grouping)
                cells[grouping] = r
                print(f"    [{defn}] {model}/{grouping}: n={r['n']} pos={r['n_positive']} "
                      f"groups={r['n_groups']} d={r['delta_H']['point']:+.4f} "
                      f"[{r['delta_H']['ci_lo']:+.4f},{r['delta_H']['ci_hi']:+.4f}] "
                      f"read={r['read']} precision={r['precision']}")
            sensitivity[defn][model] = cells

    # ── 4. 跨模型审计表 ───────────────────────────────────────────────
    print("[audit] 4/5 cross-model table ...")

    def row(model, grouping, blob, key, run_role, conf, note):
        s = blob[key]
        s = s.get("summary", s)
        d = s["delta_H"]
        return {"model": model, "grouping": grouping,
                "eligible_groups": s["n_groups"], "n": s["n"], "n_positive": s["n_positive"],
                "estimand": "Delta_H = AUROC(O+H) - AUROC(O), paired grouped bootstrap",
                "delta": round(d["point"], 4),
                "ci": [round(d["ci_lo"], 4), round(d["ci_hi"], 4)],
                "ci_width": round(d["ci_hi"] - d["ci_lo"], 4),
                "decision": s["read"], "precision": s["precision"],
                "run_role": run_role, "confirmatory_status": conf, "note": note}

    table = [
        row("Qwen2.5-7B-Instruct", "A_prompt (frozen design)", q_leak, key_A,
            "confirmatory_stage0", "confirmatory (unblinded 2026-07-22, frozen)",
            "recomputed by the post-hoc leakage script as a self-check; the frozen "
            "confirmatory value lives in stage0_analysis_report.json"),
        row("Qwen2.5-7B-Instruct", "C_prompt_union_fabid", q_leak, key_C,
            "confirmatory_stage0", "post-hoc (grouping changed after unblinding)",
            "identifier memorisation channel blocked"),
        row("Llama-3.1-8B-Instruct", "A_prompt (frozen design)", l_leak, key_A,
            l_fp.get("run_role"), "non-confirmatory (§9 cross-model replication)",
            "G3/G4 intentionally skipped; G2 verified"),
        row("Llama-3.1-8B-Instruct", "C_prompt_union_fabid", l_leak, key_C,
            l_fp.get("run_role"), "non-confirmatory (§9 cross-model replication)",
            "identifier memorisation channel blocked"),
    ]
    for r in table:
        print(f"    {r['model'][:14]:14s} {r['grouping'][:26]:26s} "
              f"groups={r['eligible_groups']:4d} d={r['delta']:+.4f} "
              f"CI={r['ci']} {r['decision']}")

    # ── 5. rung 比较登记表 ────────────────────────────────────────────
    print("[audit] 5/5 rung comparison registry ...")

    def redline_rows(model, blob, run_role):
        out = []
        for cell, v in blob["cells"].items():
            if not cell.startswith("h1"):
                continue
            for sc, r in v["redline"]["per_shortcut"].items():
                da = r["delta_artifact"]
                out.append({
                    "model": model, "cell": cell, "comparison_type": "PREREGISTERED",
                    "source": "preregistration.md §7 artifact redline",
                    "estimand": "Delta_artifact = AUROC(rung4 full-text) - AUROC(shortcut)",
                    "shortcut": sc,
                    "delta": round(da["point"], 4),
                    "ci": [round(da["ci_lo"], 4), round(da["ci_hi"], 4)],
                    "frozen_rule_triggered": r["frozen_rule_triggered"],
                    "corrected_rule_triggered": r["corrected_rule_triggered"],
                    "run_role": run_role,
                })
        return out

    def dominance_rows(model, blob, run_role):
        out = []
        for cell, v in blob["cells"].items():
            if not cell.startswith("h1"):
                continue
            o = v.get("o_dominance") or {}
            rivals = o.get("rivals") or o.get("per_rival") or {}
            for name, r in rivals.items():
                d = r.get("delta_O_minus_rival") or {}
                out.append({
                    "model": model, "cell": cell, "comparison_type": "EXPLORATORY (post-hoc)",
                    "source": "NOT preregistered — post-hoc audit 2026-07-22",
                    "estimand": "AUROC(O) - AUROC(rival rung)",
                    "rival": name,
                    "auroc_rival": round(r.get("auroc_rival", float("nan")), 4),
                    "delta": round(d.get("point", float("nan")), 4),
                    "ci": [round(d.get("ci_lo", float("nan")), 4),
                           round(d.get("ci_hi", float("nan")), 4)],
                    "rival_significantly_beats_O": r.get("rival_significantly_beats_O"),
                    "run_role": run_role,
                    "multiplicity_control": "none — 'strongest rival' selected after "
                                            "seeing all 5 rivals",
                })
        return out

    q_role, l_role = "confirmatory_stage0", l_fp.get("run_role")
    registry = {
        "preregistered_comparisons": (redline_rows("Qwen2.5-7B-Instruct", q_red, q_role)
                                      + redline_rows("Llama-3.1-8B-Instruct", l_red, l_role)),
        "exploratory_comparisons": (dominance_rows("Qwen2.5-7B-Instruct", q_red, q_role)
                                    + dominance_rows("Llama-3.1-8B-Instruct", l_red, l_role)),
        "why_o_vs_rung_is_exploratory": (
            "preregistration.md §7 fixes the artifact redline against rung4 (full-text). "
            "It contains no test of the form 'O vs the strongest rung'. That comparison was "
            "introduced post-hoc on 2026-07-22 and the rival is chosen after inspecting all "
            "five. On Llama it is therefore exploratory twice over, and no claim about the "
            "generality of O-dominance may be drawn from it."),
    }

    report = {
        "schema_version": "posthoc_cross_model_comparability_audit_v1",
        "status": ("POST-HOC / NON-CONFIRMATORY. Reads only. Does not modify or supersede "
                   "stage0_analysis_report.json, stage0_freeze_manifest.json or "
                   "stage0_unblinding_log.jsonl."),
        "timestamp": args.timestamp,
        "provenance": {
            "inputs": inputs,
            "commands_that_produced_the_llama_inputs": [
                "python scripts/run_stage0_generation.py --arm h1 --preflight "
                "--model-path D:/models/Meta-Llama-3.1-8B-Instruct",
                "python scripts/run_stage0_generation.py --arm h1 --cross-model-replication "
                "--model-path D:/models/Meta-Llama-3.1-8B-Instruct "
                "--output-root outputs/stage0_llama_h1",
                "python scripts/posthoc_h1_identifier_leakage_diagnostic.py "
                "--input-dir outputs/stage0_llama_h1/features --timestamp 2026-08-02T15:13:25Z "
                "--output outputs/stage0_llama_h1/evaluation/"
                "posthoc_h1_identifier_leakage_diagnostic.json",
                "python scripts/posthoc_redline_rule_and_o_dominance.py "
                "--input-dir outputs/stage0_llama_h1/features --timestamp 2026-08-02T15:13:25Z "
                "--h1-only --output outputs/stage0_llama_h1/evaluation/"
                "posthoc_redline_rule_and_o_dominance.json",
                "python scripts/posthoc_h1_text_representation_diagnostic.py "
                "--input-dir outputs/stage0_llama_h1/features --timestamp 2026-08-02T15:13:25Z "
                "--output outputs/stage0_llama_h1/evaluation/"
                "posthoc_h1_text_representation_diagnostic.json",
                "python scripts/posthoc_identifier_stratified_sizing.py "
                "--input-dir outputs/stage0_llama_h1/features --timestamp 2026-08-02T15:13:25Z "
                "--output outputs/stage0_llama_h1/evaluation/"
                "posthoc_identifier_stratified_sizing.json",
            ],
            "this_command": " ".join(sys.argv),
            "analysis_config": {
                "n_outer_folds": config.n_outer_folds, "n_inner_folds": config.n_inner_folds,
                "n_boot": config.n_boot, "seed": config.seed, "mode": config.mode,
                "eps_auroc": config.eps_auroc,
                "precision_max_ci_width": config.precision_max_ci_width,
            },
            "run_fingerprints": {
                "qwen": {"model": q_fp.get("model", {}).get("model_path"),
                         "hidden_index": q_fp.get("hidden_index"),
                         "backend": q_fp.get("model", {}).get("backend"),
                         "run_role": q_fp.get("run_role", "confirmatory_stage0 (implicit)")},
                "llama": {"model": l_fp.get("model", {}).get("model_path"),
                          "hidden_index": l_fp.get("hidden_index"),
                          "backend": l_fp.get("model", {}).get("backend"),
                          "run_role": l_fp.get("run_role")},
            },
            "numbers_sourced_from": "audit JSON artifacts and feature caches only; "
                                    "no value in this report is transcribed from prose",
        },
        "decision_rules": DECISION_RULES,
        "read_rule_trace": read_trace,
        "common_sample_audit": {
            "join_key": "trace_id (= example_id + sample_type + sample_index; "
                        "prompt-determined, model-independent)",
            "set_sizes": sets,
            "exclusion_crosstab": xtab,
            "sensitivity_analyses": sensitivity,
            "status": "SENSITIVITY ANALYSIS — does not replace the preregistered "
                      "per-model results; reported alongside them.",
            "caveat": ("Restricting to the common set conditions on both models' generated "
                       "content, which is a post-treatment variable. The restricted estimate "
                       "is not an unbiased version of either model's own estimand; it answers "
                       "only 'do the two agree where both produced analysable output'."),
        },
        "cross_model_table": table,
        "cross_model_table_constraints": [
            "No pooled cross-model estimate is computed — none was preregistered.",
            "Bare Delta_H values are NOT compared across models as evidence; the rows are "
            "listed so that grouping, precision and run_role are visible next to each number.",
            "The Qwen rows are confirmatory-design results; the Llama rows are "
            "non-confirmatory replication. These are not interchangeable.",
        ],
        "rung_comparison_registry": registry,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] report -> {out}")


if __name__ == "__main__":
    main()
