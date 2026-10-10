r"""用**重提取后的一致证据**重算 MCQ 的 G3 / G4 报告，并与已核验的 H1 证据形成两臂汇总。

走的是既有报告构建与校验逻辑（`v3_g3_smoke.evaluate_arm` / `build_report`、
`v3_g4_quality.check_arm` / `build_report` / `matches_g3_production_fingerprint`），
**不复制旧的 `passed` 值**，也**不把文本重放包装成重新生成**。

H1 侧不受 MCQ 解析修复影响：本脚本**核验**其产物哈希与指纹后原样复用，不重跑。

报告里区分三类证据来源：
    historical_generation  —— 原运行生成的文本与生成期观测（本轮未改）
    current_parsing        —— 当前解析规则给出的标签与答案区间
    current_features       —— 本轮基于已保存文本的 teacher-forced 重提取

用法：
    python scripts/diagnostics/rebuild_mcq_gate_reports_v3.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool      # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation import v3_g4_quality as g4         # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation.manifest import write_atomic       # noqa: E402

SAMPLES = ROOT / "outputs/stage1/samples"
PREREG = ROOT / "docs/paper/preregistration_v3.md"
LOCK = ROOT / "docs/paper/preregistration_v3.lock"
REGISTRATION = ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
REEXTRACT = ROOT / "tmp/stage1_l2_g3/mcq_reextract_v3.json"

G3_SRC = ROOT / "outputs/stage1_llama_v3/g3_budget512"
G4_SRC = ROOT / "outputs/stage1_llama_v3/g4_budget512"
G3_OUT = ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed"
G4_OUT = ROOT / "outputs/stage1_llama_v3/g4_budget512_reparsed"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in io.open(p, encoding="utf-8") if x.strip()]


def _load_run(arm_file: Path, hidden_dir: Path, fingerprint: dict,
              hidden_verification: dict) -> g3.ArmRun:
    rows = _rows(arm_file)
    run = g3.ArmRun(arm=g3.ARM_MCQ, n_units=len(rows))
    run.rows = rows
    run.fingerprint = fingerprint
    run.hidden_verification = hidden_verification
    for row in rows:
        p = hidden_dir / f'{row["unit_id"]}.npy'
        if p.exists():
            run.hidden[row["unit_id"]] = np.load(p)
    return run


def _evidence_sources(rows: list[dict], reextract_meta: dict) -> dict:
    kinds = Counter(r.get("_provenance", {}).get("kind") for r in rows)
    return {
        "historical_generation": {
            "what": "completion 原文与生成期观测（num_generated_tokens 等）",
            "changed_this_round": False,
            "note": "本轮未调用 generate，未重新生成任何回答"},
        "current_parsing": {
            "what": "parsed_label / parse_status / 答案字符区间",
            "parser_sha256": reextract_meta["fixed_inputs"]["code"][
                "src/recover_attention/domain_label_proxy.py"],
            "applied_to": "全部行"},
        "current_features": {
            "what": "F5 与 H（hidden）",
            "method": ("基于已保存文本的 teacher-forced 重提取；"
                       "**不是**恢复历史生成 token 序列"),
            "rows_reextracted": kinds.get("reextracted", 0),
            "rows_reused": kinds.get("reused", 0),
            "rows_excluded": kinds.get("excluded", 0),
            "consistency_tolerance": reextract_meta["tolerance"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tmp/stage1_l2_g3/mcq_gate_rebuild_v3.json")
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    meta = json.loads(REEXTRACT.read_text(encoding="utf-8"))
    failures: list[str] = []
    log: list[dict] = []

    def ck(label: str, ok: bool, detail=None):
        log.append({"check": label, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    reg = g3.load_budget_registration(REGISTRATION, cfg=cfg, model_path=MODEL)
    geometry = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=MODEL)
    g3_criteria = g3.G3Criteria(hidden_dim=geometry.hidden_dim,
                                block_index=geometry.block_index,
                                tuple_index=geometry.tuple_index)
    g4_criteria = g4.G4Criteria(hidden_dim=geometry.hidden_dim,
                                block_index=geometry.block_index,
                                tuple_index=geometry.tuple_index)

    # 指纹与 hidden 校验取自本轮重提取（真实前向给出），不是合成值
    fp_now = meta["fingerprint_now"]
    hv_now = meta["hidden_verification"]
    print("[0] 前置核对")
    ck("重提取报告无失败项", not meta.get("failures"), meta.get("failures"))
    old_g3_fp = json.loads((G3_SRC / "mcq_g3_run_fingerprint.json").read_text(encoding="utf-8"))
    for key in ("model", "tokenizer", "hidden_index"):
        ck(f"本轮指纹 {key} 与原 G3 生产指纹相同",
           json.dumps(fp_now[key], sort_keys=True) == json.dumps(old_g3_fp[key], sort_keys=True))
    ck("本轮指纹 max_new_tokens == 512",
       fp_now["generation_params"]["max_new_tokens"] == 512)

    # ── G3 ──────────────────────────────────────────────────────────────
    print("[1] 重算 MCQ G3")
    ver3 = g3.verify_inputs(samples_dir=SAMPLES, prereg=PREREG, lock=LOCK, cfg=cfg)
    gate3 = ver3.pop("_gate_rows")[g3.ARM_MCQ]
    run3 = _load_run(G3_OUT / "mcq_g3_unit_results.jsonl",
                     G3_OUT / "mcq_g3_unit_hidden", fp_now, hv_now)
    ev3 = g3.evaluate_arm(run3, n_records=len(gate3), criteria=g3_criteria)
    rep3 = g3.build_report(arm=g3.ARM_MCQ, run=run3, verification=ver3, evaluation=ev3,
                           cfg=cfg, n_records=len(gate3), criteria=g3_criteria)
    old3 = json.loads((G3_SRC / "mcq_g3_smoke_report.json").read_text(encoding="utf-8"))
    rep3["dry_run"] = False
    rep3["budget_extension"] = old3["budget_extension"]
    rep3["evidence_sources"] = _evidence_sources(run3.rows, meta)
    rep3["supersedes"] = {
        "path": "outputs/stage1_llama_v3/g3_budget512/mcq_g3_smoke_report.json",
        "sha256": _sha(G3_SRC / "mcq_g3_smoke_report.json"),
        "note": ("旧报告**保留不改**。本报告基于当前解析与重提取后的一致特征重新计算，"
                 "`passed` 由判据现算，未复制旧值。")}
    print(f"  parse_failure_rate={ev3['checks']['parse_failure_rate']} "
          f"passed={ev3['passed']} failed={ev3['failed_checks']}")
    ck("G3 无未处理异常/非有限值",
       ev3["checks"]["no_unhandled_exception"] and ev3["checks"]["no_non_finite_values"])

    # ── G4 ──────────────────────────────────────────────────────────────
    print("[2] 重算 MCQ G4")
    ver4 = g3.verify_inputs(samples_dir=SAMPLES, prereg=PREREG, lock=LOCK, cfg=cfg, gate="G4")
    gate4 = ver4.pop("_gate_rows")[g3.ARM_MCQ]
    run4 = _load_run(G4_OUT / "mcq_g4_unit_results.jsonl",
                     G4_OUT / "mcq_g4_unit_hidden", fp_now, hv_now)
    g4.backfill_group_id(run4, g3.build_arm_records(
        g3.ARM_MCQ, gate4, cfg=cfg,
        tokenizer=type("T", (), {"apply_chat_template": staticmethod(
            lambda m, **k: json.dumps(m, ensure_ascii=False))})()))
    ev4 = g4.check_arm(g3.ARM_MCQ, run4, cfg=cfg, criteria=g4_criteria,
                       budget_registration=reg)
    fpc = g4.matches_g3_production_fingerprint(g3.ARM_MCQ, fp_now, fp_now, cfg=cfg,
                                               budget_registration=reg)
    old4 = json.loads((G4_SRC / "mcq_g4_quality_report.json").read_text(encoding="utf-8"))
    rep4 = g4.build_report(arm=g3.ARM_MCQ, run=run4, verification=ver4, evaluation=ev4,
                           fingerprint_check=fpc, cfg=cfg,
                           g3_provenance=old4["g3_provenance"], criteria=g4_criteria)
    rep4["evidence_sources"] = _evidence_sources(run4.rows, meta)
    rep4["supersedes"] = {
        "path": "outputs/stage1_llama_v3/g4_budget512/mcq_g4_quality_report.json",
        "sha256": _sha(G4_SRC / "mcq_g4_quality_report.json"),
        "note": ("旧报告**保留不改**。本报告基于当前解析与重提取后的一致特征重新计算，"
                 "`passed` 由判据现算，未复制旧值。")}
    print(f"  parse_failure_rate={ev4['stats']['parse_failure_rate']} "
          f"passed={ev4['ok']} failed={ev4['failed_checks']}")

    # ── H1：核验后复用，不重跑 ──────────────────────────────────────────
    print("[3] H1 证据：核验后复用（MCQ 解析修复不影响 H1）")
    h1_g3 = json.loads((G3_SRC / "h1_g3_smoke_report.json").read_text(encoding="utf-8"))
    h1_g4 = json.loads((G4_SRC / "h1_g4_quality_report.json").read_text(encoding="utf-8"))
    ck("H1 G3 passed", h1_g3["passed"] is True)
    ck("H1 G4 passed", h1_g4["passed"] is True)
    ck("H1 G3 budget_extension 为 null（384 即冻结值）",
       h1_g3.get("budget_extension") is None)
    for key in ("model", "tokenizer", "hidden_index"):
        ck(f"H1 G3 指纹 {key} 与本轮 MCQ 指纹相同（同一后端）",
           json.dumps(h1_g3["fingerprint"][key], sort_keys=True)
           == json.dumps(fp_now[key], sort_keys=True))
    h1_evidence = {
        "reused_without_rerun": True,
        "why": ("MCQ 解析修复只触及 MCQ 臂的答案提取与派生特征；H1 的 identifier 标注、"
                "pooling 与 F5 与 parse_option_answer 无关"),
        "h1_g3": {"path": "outputs/stage1_llama_v3/g3_budget512/h1_g3_smoke_report.json",
                  "sha256": _sha(G3_SRC / "h1_g3_smoke_report.json"),
                  "passed": h1_g3["passed"],
                  "eligible_rate": h1_g3["checks"]["eligible_rate"]},
        "h1_g4": {"path": "outputs/stage1_llama_v3/g4_budget512/h1_g4_quality_report.json",
                  "sha256": _sha(G4_SRC / "h1_g4_quality_report.json"),
                  "passed": h1_g4["passed"],
                  "eligible_rate": h1_g4["stats"]["eligible_rate"]},
    }

    if failures:
        print(f"\n[STOP] {failures}", file=sys.stderr)
        args.out.write_text(json.dumps({"checks": log, "failures": failures},
                                       ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return 2

    # ── 落盘 ────────────────────────────────────────────────────────────
    g3.write_report(G3_OUT / "mcq_g3_smoke_report.json", rep3)
    g3.write_report(G3_OUT / "mcq_g3_run_fingerprint.json", fp_now)
    g4.write_report(G4_OUT / "mcq_g4_quality_report.json", rep4)

    # 两臂汇总（MCQ 用新证据，H1 核验后复用）
    runs = {g3.ARM_H1: g3.ArmRun(arm=g3.ARM_H1, fingerprint=h1_g3["fingerprint"]),
            g3.ARM_MCQ: g3.ArmRun(arm=g3.ARM_MCQ, fingerprint=fp_now)}
    invariant = g3.backend_invariant(runs, cfg, budget_registration=reg)
    ck("两臂 backend invariant ok", invariant["ok"], invariant["mismatched_keys"])
    summary = {
        "schema_version": "stage1_g3_summary_v3", "gate": "G3",
        "gate_only": True, "performance_blind": True,
        "did_not_compute": list(g3.DID_NOT_COMPUTE),
        "mode": "real",
        "evidence_basis": {
            "h1": "核验后复用原运行证据（未重跑）",
            "mcq": "当前解析 + 基于已保存文本的 teacher-forced 特征重提取",
            "caveat": ("MCQ 的 F5/H 由本轮重提取给出，**不是**原运行当时写下的那份；"
                       "生成文本本身未变。下游不得把旧解析的 MCQ 产物与本目录混用。")},
        "arms_passed": {g3.ARM_H1: h1_g3["passed"], g3.ARM_MCQ: rep3["passed"]},
        "backend_invariant": invariant,
        "criteria": rep3["criteria"],
        "geometry": {k: rep3["criteria"].get(k) for k in
                     ("hidden_dim", "block_index_zero_based", "hidden_states_tuple_index")},
        "thresholds": {
            g3.ARM_H1: {"metric": "eligible_rate",
                        "value": h1_g3["checks"]["eligible_rate"],
                        "threshold": g3.CRITERIA.h1_min_eligible_rate, "direction": ">="},
            g3.ARM_MCQ: {"metric": "parse_failure_rate",
                         "value": rep3["checks"]["parse_failure_rate"],
                         "threshold": g3.CRITERIA.mcq_max_parse_failure_rate,
                         "direction": "<="}},
        "budget_registration": reg.as_dict(),
        "h1_evidence": h1_evidence,
        "refusals": [],
        "passed": bool(h1_g3["passed"] and rep3["passed"] and invariant["ok"]),
        "real_g3_passed": bool(h1_g3["passed"] and rep3["passed"] and invariant["ok"]),
        "supersedes": {"path": "outputs/stage1_llama_v3/g3_budget512/g3_summary.json",
                       "sha256": _sha(G3_SRC / "g3_summary.json"),
                       "note": "旧汇总保留不改；本汇总的 passed 由现算结果组合，未复制旧值"},
    }
    write_atomic(G3_OUT / "g3_summary.json",
                 json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    g4_summary = {
        "schema_version": "stage1_g4_summary_v1", "gate": "G4",
        "gate_only": True, "performance_blind": True,
        "did_not_compute": list(g4.DID_NOT_COMPUTE),
        "arms_passed": {g3.ARM_H1: h1_g4["passed"], g3.ARM_MCQ: rep4["passed"]},
        "criteria": rep4["criteria"],
        "degeneration": {g3.ARM_H1: h1_g4["stats"]["degeneration"],
                         g3.ARM_MCQ: rep4["stats"]["degeneration"]},
        "margins": {g3.ARM_H1: h1_g4["margins"], g3.ARM_MCQ: rep4["margins"]},
        "fingerprint_matches_g3_production": {
            g3.ARM_H1: h1_g4["fingerprint_matches_g3_production"]["ok"],
            g3.ARM_MCQ: rep4["fingerprint_matches_g3_production"]["ok"]},
        "evidence_basis": summary["evidence_basis"],
        "h1_evidence": h1_evidence,
        "passed": bool(h1_g4["passed"] and rep4["passed"]),
        "supersedes": {"path": "outputs/stage1_llama_v3/g4_budget512/g4_summary.json",
                       "sha256": _sha(G4_SRC / "g4_summary.json"),
                       "note": "旧汇总保留不改；本汇总的 passed 由现算结果组合，未复制旧值"},
    }
    write_atomic(G4_OUT / "g4_summary.json",
                 json.dumps(g4_summary, ensure_ascii=False, indent=2) + "\n")

    out = {
        "schema": "stage1_mcq_gate_rebuild_v3", "date": "2026-09-15",
        "checks": log,
        "g3": {"passed": rep3["passed"], "failed_checks": rep3["failed_checks"],
               "parse_failure_rate": rep3["checks"]["parse_failure_rate"],
               "report": "outputs/stage1_llama_v3/g3_budget512_reparsed/mcq_g3_smoke_report.json",
               "sha256": _sha(G3_OUT / "mcq_g3_smoke_report.json")},
        "g4": {"passed": rep4["passed"], "failed_checks": rep4["failed_checks"],
               "parse_failure_rate": rep4["stats"]["parse_failure_rate"],
               "report": "outputs/stage1_llama_v3/g4_budget512_reparsed/mcq_g4_quality_report.json",
               "sha256": _sha(G4_OUT / "mcq_g4_quality_report.json")},
        "g3_summary": {"passed": summary["passed"],
                       "sha256": _sha(G3_OUT / "g3_summary.json")},
        "g4_summary": {"passed": g4_summary["passed"],
                       "sha256": _sha(G4_OUT / "g4_summary.json")},
        "h1_evidence": h1_evidence,
        "not_claimed": (
            "G3/G4 通过只证明运行质量，不代表最终方法复现成功。"
            "本轮不改变已生成的原始文本，不改写旧 Qwen/Llama 证据；"
            "修订的是答案提取及相关派生数据。旧报告保留，新报告基于一致证据重新计算。"),
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n[G3   ] passed={rep3['passed']} ｜ [G4] passed={rep4['passed']}")
    print(f"[汇总 ] G3 两臂 passed={summary['passed']} ｜ G4 两臂 passed={g4_summary['passed']}")
    print(f"[out  ] {args.out}")
    return 0 if (summary["passed"] and g4_summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
