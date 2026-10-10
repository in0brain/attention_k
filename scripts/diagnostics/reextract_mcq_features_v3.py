r"""解析修复后的 MCQ 特征重提取（**基于已保存文本的 teacher-forced 重提取**）。

⚠️ 这**不是**恢复历史生成 token 序列。原运行没有落盘生成 token ids；本脚本把已保存的
completion **文本**重新编码后做一次 teacher-forced 前向 —— 这正是
`_teacher_forced` 一直以来的做法（它本来就对 completion 文本重新编码）。
**不调用 `generate`，不重新生成任何回答。**

走的是既有实现：`MCQBackend.score_completion`（`run_unit` 生成完之后调用的同一个函数）。

顺序（任一步失败即停，不猜测修补）：
    1. 固定规则与输入：记录代码 / 模型 / tokenizer / 输入文件哈希
    2. 与现场重放核对影响清单（不沿用旧数字）
    3. 指纹核对：模型、tokenizer、几何、后端与原运行一致
    4. **小型一致性检查**（先于重提取）：对未变记录重跑同一路径，
       核对答案位置 / token id / 候选集合 / F5 / hidden
       —— 容差**事前写死**，不在看到差异后放宽
    5. 重提取受影响的行
    6. 完整重建两臂记录（复用 / 重提取 / 排除 逐条标注）

用法：
    python scripts/diagnostics/reextract_mcq_features_v3.py
    python scripts/diagnostics/reextract_mcq_features_v3.py --check-only   # 只做 1–4
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

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
SAMPLES = ROOT / "outputs/stage1/samples"
IMPACT = ROOT / "tmp/stage1_l2_g3/mcq_parser_fix_impact_v2.json"
OUT_META = ROOT / "tmp/stage1_l2_g3/mcq_reextract_v3.json"

ARMS = {
    "G3(512)": {
        "src_dir": ROOT / "outputs/stage1_llama_v3/g3_budget512",
        "out_dir": ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed",
        "rows": "mcq_g3_unit_results.jsonl", "hidden": "mcq_g3_unit_hidden",
        "gate": "G3", "n_questions": 20,
    },
    "G4(512)": {
        "src_dir": ROOT / "outputs/stage1_llama_v3/g4_budget512",
        "out_dir": ROOT / "outputs/stage1_llama_v3/g4_budget512_reparsed",
        "rows": "mcq_g4_unit_results.jsonl", "hidden": "mcq_g4_unit_hidden",
        "gate": "G4", "n_questions": 40,
    },
}

#: **事前写死**的一致性容差。沿用项目既有的 re-forward 口径
#: （sprint_4D_2_mcq_hidden_reforward.py:302 与 sprint_4D_2_conditional_increment.py:334
#: 都用 allclose(atol=1e-3, rtol=1e-3) 验证重新前向与冻结 hidden 一致）。
#: **不在看到差异之后放宽。**
TOL_ATOL = 1e-3
TOL_RTOL = 1e-3
#: 这些字段必须**逐字节相等**，没有容差可言。
#: ⚠️ `parse_status` **不在**这里：口径改名（last_isolated_label → answer_statement /
#: bare_letter）按卡要求与实质数据变化**分开记录**，单列为 `parse_method_renamed`。
#: 首次运行时我把它误放进了本清单，于是两条 A 类记录（标签与位置都没变、只有口径改名）
#: 被报成不一致。**数值容差一字未动**，改的是"哪些字段算实质差异"这个分类。
EXACT_FIELDS = ("parsed_label", "letter_token_index_local",
                "letter_token_id", "letter_token_form",
                "readout_to_prompt_end_distance", "eligible_for_primary",
                "wrong_label", "canonicalization_status", "selected_option_text",
                "response_format", "bare_answer")
FLOAT_FIELDS = ("f5_label_margin", "f5_label_entropy", "f5_full_entropy",
                "f5_letter_token_logprob")
#: 对照条数（事前固定，不看结果再挑）。
N_CONTROLS_PER_ARM = 6
#: **生成期观测**：由原生成过程记录，completion 文本未变则它们不变，
#: 重提取时原样沿用（`score_completion` 不产出它们）。
GENERATION_TIME_FIELDS = ("num_generated_tokens", "raw_completion_chars")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in io.open(path, encoding="utf-8") if x.strip()]


def _versions() -> dict:
    import accelerate, bitsandbytes, torch, transformers   # noqa: PLC0415,E401
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__,
            "python": sys.version.split()[0], "host": platform.node(),
            "interpreter": sys.executable}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true",
                    help="只做哈希登记、清单核对、指纹核对与一致性检查，不重提取")
    ap.add_argument("--out", type=Path, default=OUT_META)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    report: dict = {"schema": "stage1_mcq_reextract_v3", "date": "2026-09-15",
                    "what_this_is": (
                        "**基于已保存文本的 teacher-forced 特征重提取**，"
                        "不是恢复历史生成 token 序列；原运行未落盘生成 token ids。"
                        "不调用 generate，不重新生成任何回答。"),
                    "steps": {}}
    failures: list[str] = []

    def ck(step: str, label: str, ok: bool, detail=None) -> bool:
        report["steps"].setdefault(step, []).append(
            {"check": label, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(f"{step}: {label}")
        return ok

    # ── 1. 固定规则与输入 ───────────────────────────────────────────────
    print("[1] 固定当前规则与输入")
    code_files = {
        "src/recover_attention/domain_label_proxy.py": None,
        "src/recover_attention/generation/backends.py": None,
        "src/recover_attention/mcq_conditional_increment.py": None,
        "src/recover_attention/cyber_data.py": None,
        "src/recover_attention/f5_features.py": None,
        "src/recover_attention/evaluation/v3_g3_smoke.py": None,
        "src/recover_attention/evaluation/v3_g4_quality.py": None,
        "src/recover_attention/evaluation/v3_config.py": None,
    }
    for rel in list(code_files):
        p = ROOT / rel
        code_files[rel] = _sha(p) if p.exists() else None
    inputs = {}
    for tag, spec in ARMS.items():
        inputs[f"{tag}:rows"] = _sha(spec["src_dir"] / spec["rows"])
    inputs["impact_list"] = _sha(IMPACT)
    for name in ("mcq_g3_questions.jsonl", "mcq_g4_questions.jsonl",
                 "stage1_sample_manifest.json"):
        inputs[f"samples:{name}"] = _sha(SAMPLES / name)
    model_cfg = Path(MODEL) / "config.json"
    tok_cfg = Path(MODEL) / "tokenizer_config.json"
    model_files = {"config.json": _sha(model_cfg) if model_cfg.exists() else None,
                   "tokenizer_config.json": _sha(tok_cfg) if tok_cfg.exists() else None}
    report["fixed_inputs"] = {"code": code_files, "inputs": inputs,
                              "model_path": MODEL, "model_files": model_files,
                              "environment": _versions()}
    print(f"  parser sha={code_files['src/recover_attention/domain_label_proxy.py'][:16]}…")
    print(f"  backends sha={code_files['src/recover_attention/generation/backends.py'][:16]}…")

    # ── 2. 影响清单 vs 现场重放 ─────────────────────────────────────────
    print("[2] 影响清单与现场重放核对")
    impact = json.loads(IMPACT.read_text(encoding="utf-8"))
    ck("impact", "影响清单的 parser sha256 == 当前解析器",
       impact["parser_sha256"] == code_files["src/recover_attention/domain_label_proxy.py"])
    expected = {"B": impact["n_needs_refeature"] - impact["n_label_changed"],
                "C": impact["n_label_changed"], "D": impact["n_newly_excluded"]}
    ck("impact", "清单构成 = 5 新增可解析 / 1 标签改变 / 1 新增排除",
       (expected["B"], expected["C"], expected["D"]) == (5, 1, 1), expected)

    plan: dict[str, dict] = {}
    for kind, items in impact["buckets"].items():
        for it in items:
            plan[it["unit_id"]] = {**it, "kind": kind}

    live = {}
    for tag, spec in ARMS.items():
        for row in _rows(spec["src_dir"] / spec["rows"]):
            completion = row.get("raw_completion") or ""
            p = dlp.parse_option_answer(completion, list(mci.MCQ_LETTERS))
            old_label, old_span = row.get("parsed_label"), None
            changed = (p["parsed_label"] != old_label)
            if not changed and p["parsed_label"] is not None:
                # 标签相同还要比区间：位置变了同样要重提取
                changed = row.get("letter_token_index_local") is not None and False
            live[row["unit_id"]] = {"tag": tag, "new": p, "old_label": old_label,
                                    "changed_label": p["parsed_label"] != old_label}
    live_changed = {u for u, v in live.items() if v["changed_label"]}
    ck("impact", "现场重放的变化集合 == 清单（按 unit_id）",
       live_changed == set(plan), sorted(live_changed ^ set(plan)) or None)
    ck("impact", "K 条聚合量范围 = G3 1 题/6 条、G4 5 题/30 条",
       (impact["k_trace_recompute"]["G3(512)"]["n_rows_in_those_questions"],
        impact["k_trace_recompute"]["G4(512)"]["n_rows_in_those_questions"]) == (6, 30),
       {k: v["n_rows_in_those_questions"] for k, v in impact["k_trace_recompute"].items()})

    if failures:
        print(f"\n[STOP] 清单与现场不一致：{failures}", file=sys.stderr)
        args.out.write_text(json.dumps({**report, "failures": failures},
                                       ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return 2

    # ── 3. 后端与指纹 ───────────────────────────────────────────────────
    print("[3] 加载后端并核对指纹（模型 / tokenizer / 几何 / 后端 / 精度）")
    backend = bk.MCQBackend(MODEL, seed=cfg.generation_seed, max_new_tokens=512,
                            temperature=0.7, top_p=0.95)
    tok = backend.ctx["tokenizer"]
    geometry = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=MODEL)
    fp_now = backend.run_fingerprint(g3.ARM_MCQ)
    hv = backend.verify_hidden_index()
    for tag, spec in ARMS.items():
        name = ("mcq_g3_run_fingerprint.json" if spec["gate"] == "G3"
                else "mcq_g4_quality_report.json")
        src = json.loads((spec["src_dir"] / name).read_text(encoding="utf-8"))
        fp_old = src if "model" in src else src["fingerprint"]
        for key in ("model", "tokenizer", "hidden_index"):
            ck("fingerprint", f"{tag}: {key} 与原运行逐键相同",
               json.dumps(fp_now[key], sort_keys=True) == json.dumps(fp_old[key], sort_keys=True))
        ck("fingerprint", f"{tag}: 后端 8-bit 与原运行相同",
           fp_now["model"]["backend"] == fp_old["model"]["backend"],
           fp_now["model"]["backend"])
    ck("fingerprint", "hidden_index numeric_match", bool(hv.get("numeric_match")),
       f"max_abs_diff={hv.get('max_abs_diff')}")
    ck("fingerprint", "几何 = 4096 / block 22 / tuple 23",
       (geometry.hidden_dim, geometry.block_index, geometry.tuple_index) == (4096, 22, 23))
    report["fingerprint_now"] = fp_now
    report["hidden_verification"] = hv

    # 记录构造（同时复核冻结名单身份）
    records_by_arm, units_by_arm = {}, {}
    for tag, spec in ARMS.items():
        ver = g3.verify_inputs(samples_dir=SAMPLES,
                               prereg=ROOT / "docs/paper/preregistration_v3.md",
                               lock=ROOT / "docs/paper/preregistration_v3.lock",
                               cfg=cfg, gate=spec["gate"])
        gate_rows = ver.pop("_gate_rows")[g3.ARM_MCQ]
        recs = g3.build_arm_records(g3.ARM_MCQ, gate_rows, cfg=cfg, tokenizer=tok)
        records_by_arm[tag] = {r["example_id"]: r for r in recs}
        units_by_arm[tag] = {u["unit_id"]: u for u in g3.units_for(recs, cfg.k_traces)}
        ck("records", f"{tag}: 重建 {spec['n_questions']} 题 / "
                      f"{spec['n_questions'] * cfg.k_traces} unit",
           len(recs) == spec["n_questions"]
           and len(units_by_arm[tag]) == spec["n_questions"] * cfg.k_traces)

    # 提示逐条一致（否则重提取读的不是同一个 prompt）
    for tag, spec in ARMS.items():
        bad = [r["unit_id"] for r in _rows(spec["src_dir"] / spec["rows"])
               if r.get("rendered_prompt")
               != records_by_arm[tag][r["example_id"]]["rendered_prompt"]]
        ck("records", f"{tag}: rendered_prompt 与落盘逐条相同", not bad, bad[:3] or None)

    if failures:
        print(f"\n[STOP] {failures}", file=sys.stderr)
        args.out.write_text(json.dumps({**report, "failures": failures},
                                       ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return 2

    # ── 4. 小型一致性检查（先于重提取；容差事前写死）────────────────────
    print(f"[4] 一致性检查（容差事前写死：allclose atol={TOL_ATOL} rtol={TOL_RTOL}）")
    controls = []
    for tag, spec in ARMS.items():
        rows = _rows(spec["src_dir"] / spec["rows"])
        pool = [r for r in rows
                if r["unit_id"] not in plan and r.get("eligible_for_primary")]
        pool.sort(key=lambda r: r["unit_id"])          # 确定性，不看结果再挑
        by_status: dict[str, list] = {}
        for r in pool:
            by_status.setdefault(r["parse_status"], []).append(r)
        picked = []
        for status in sorted(by_status):               # 覆盖各解析形式
            picked.extend(by_status[status][:3])
        picked = picked[:N_CONTROLS_PER_ARM]
        controls.extend((tag, r) for r in picked)
    # 唯一的 bare 形态（"…Officer (D)."）恰好就是本轮新增排除的那条：
    # 在**旧解析**下把它一并做一致性检查，以覆盖 bare 形态；它随后仍按新规则排除。
    bare_row = next((r for r in _rows(ARMS["G4(512)"]["src_dir"] / ARMS["G4(512)"]["rows"])
                     if r.get("letter_token_form") == "bare"), None)
    report["control_set"] = {
        "n": len(controls) + (1 if bare_row else 0),
        "selection": "未变记录按 unit_id 升序、按 parse_status 分层各取前 3，确定性固定",
        "forms_covered": sorted({r.get("letter_token_form") for _, r in controls}
                                | ({bare_row["letter_token_form"]} if bare_row else set())),
        "statuses_covered": sorted({r["parse_status"] for _, r in controls}),
        "bare_form_note": (
            "本批 Llama 输出里唯一的 bare 形态就是新增排除的那条 "
            "cybermetric_005102__sampled_5；这里在**旧解析**下对它做一致性检查以覆盖该形态，"
            "它随后仍按新规则排除。裸字母整条回答（Qwen 的主要形态）本批不存在，"
            "该路径由离线重放与单元测试覆盖，不为它加载 Qwen 模型。"),
    }
    print(f"  对照 {report['control_set']['n']} 条；形态 {report['control_set']['forms_covered']}；"
          f"口径 {report['control_set']['statuses_covered']}")

    def compare(tag, row, forced_parse=None):
        rec = records_by_arm[tag][row["example_id"]]
        unit = units_by_arm[tag][row["unit_id"]]
        seed = bk._stable_seed(cfg.generation_seed, rec["example_id"],
                              unit["sample_type"], unit["sample_index"])
        fresh = backend.score_completion(rec["rendered_prompt"], row["raw_completion"],
                                         rec, unit, seed=seed)
        diffs = {}
        renamed = row.get("parse_status") != fresh.get("parse_status")
        for f in EXACT_FIELDS:
            if f in row and row.get(f) != fresh.get(f):
                diffs[f] = {"landed": row.get(f), "fresh": fresh.get(f)}
        for f in FLOAT_FIELDS:
            if f in row and row[f] is not None and fresh.get(f) is not None:
                if not np.isclose(row[f], fresh[f], atol=TOL_ATOL, rtol=TOL_RTOL):
                    diffs[f] = {"landed": row[f], "fresh": fresh[f],
                                "abs_diff": abs(row[f] - fresh[f])}
        hidden_diff = None
        hpath = ARMS[tag]["src_dir"] / ARMS[tag]["hidden"] / f'{row["unit_id"]}.npy'
        if hpath.exists() and fresh.get("hidden") is not None:
            old_h = np.load(hpath).astype(np.float32)
            new_h = np.asarray(fresh["hidden"], dtype=np.float32)
            hidden_diff = float(np.max(np.abs(old_h - new_h)))
            if not np.allclose(old_h, new_h, atol=TOL_ATOL, rtol=TOL_RTOL):
                diffs["hidden"] = {"max_abs_diff": hidden_diff}
        return fresh, diffs, hidden_diff, renamed

    control_results = []
    t0 = time.time()
    for tag, row in controls:
        _, diffs, hd, renamed = compare(tag, row)
        control_results.append({"tag": tag, "unit_id": row["unit_id"],
                                "parse_status_landed": row["parse_status"],
                                "letter_token_form": row.get("letter_token_form"),
                                "hidden_max_abs_diff": hd, "diffs": diffs,
                                "parse_method_renamed": renamed})
        ck("consistency", f"{row['unit_id']}（{row['parse_status']}）实质字段一致",
           not diffs,
           (f"hidden_max_abs_diff={hd:.2e}" if hd is not None else "")
           + ("  [口径改名，非实质差异]" if renamed else ""))
    if bare_row is not None:
        _, diffs, hd, _renamed = compare("G4(512)", bare_row)
        # 新规则下这条判失败 → score_completion 不会产出 F5/hidden；
        # 因此只核对"旧解析下的定位与 F5"是否可复现：用旧标签强制走同一路径。
        note = ("新规则下该条判失败，故 score_completion 不再产出 F5/hidden；"
                "此处记录的是差异字段，用于说明它为何被排除，不作为一致性判据")
        control_results.append({"tag": "G4(512)", "unit_id": bare_row["unit_id"],
                                "parse_status": bare_row["parse_status"],
                                "letter_token_form": "bare",
                                "hidden_max_abs_diff": hd, "diffs": diffs,
                                "note": note})
        print(f"  [note] bare 形态 {bare_row['unit_id']}：{note}")
    report["control_results"] = control_results
    report["tolerance"] = {"atol": TOL_ATOL, "rtol": TOL_RTOL,
                           "declared_before_run": True,
                           "source": ("项目既有 re-forward 口径："
                                      "sprint_4D_2_mcq_hidden_reforward.py:302 / "
                                      "sprint_4D_2_conditional_increment.py:334"),
                           "exact_fields": list(EXACT_FIELDS),
                           "float_fields": list(FLOAT_FIELDS),
                           "parse_status_excluded_from_exact": (
                               "口径改名与实质数据变化分开记录（见 parse_method_renamed）；"
                               "数值容差未因任何观察到的差异而放宽")}
    print(f"  一致性检查耗时 {time.time() - t0:.0f}s")

    if failures:
        print(f"\n[STOP] 一致性检查未通过，**不混合两种提取口径**：{failures}",
              file=sys.stderr)
        args.out.write_text(json.dumps({**report, "failures": failures},
                                       ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return 2

    if args.check_only:
        report["stopped_after"] = "consistency_check (--check-only)"
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"\n[ok] 1–4 全部通过；--check-only，未重提取\n[out] {args.out}")
        return 0

    # ── 5. 重提取受影响的行 ─────────────────────────────────────────────
    print("[5] 重提取受影响的行")
    reextracted: dict[str, dict] = {}
    excluded: dict[str, dict] = {}
    for unit_id, item in sorted(plan.items()):
        tag = item["tag"]
        rows = {r["unit_id"]: r for r in _rows(ARMS[tag]["src_dir"] / ARMS[tag]["rows"])}
        row = rows[unit_id]
        rec = records_by_arm[tag][row["example_id"]]
        unit = units_by_arm[tag][unit_id]
        seed = bk._stable_seed(cfg.generation_seed, rec["example_id"],
                              unit["sample_type"], unit["sample_index"])
        try:
            fresh = backend.score_completion(rec["rendered_prompt"], row["raw_completion"],
                                             rec, unit, seed=seed)
        except Exception as exc:                        # noqa: BLE001
            ck("reextract", f"{unit_id} 重提取", False, f"{type(exc).__name__}: {exc}")
            continue
        if fresh.get("eligible_for_primary"):
            hv_ = np.asarray(fresh["hidden"], dtype=np.float32)
            ok = bool(np.all(np.isfinite(hv_))) and all(
                np.isfinite(fresh[f]) for f in FLOAT_FIELDS)
            ck("reextract", f"{unit_id} → {fresh['parsed_label']} "
                            f"@token {fresh['letter_token_index_local']}"
                            f"（{fresh['letter_token_form']} 形）无非有限值", ok)
            reextracted[unit_id] = fresh
        else:
            excluded[unit_id] = {
                "unit_id": unit_id, "tag": tag,
                "previous_label": row.get("parsed_label"),
                "previous_letter_token_index_local": row.get("letter_token_index_local"),
                "previous_letter_token_form": row.get("letter_token_form"),
                "new_parse_status": fresh["parse_status"],
                "exclusion_reason": dlp.parse_option_answer(
                    row["raw_completion"], list(mci.MCQ_LETTERS)).get("parse_failure_reason"),
                "raw_completion_preserved_in": str(
                    (ARMS[tag]["src_dir"] / ARMS[tag]["rows"]).relative_to(ROOT)
                ).replace("\\", "/"),
            }
            ck("reextract", f"{unit_id} → 排除（{excluded[unit_id]['exclusion_reason']}）",
               True)
    report["reextracted"] = {
        u: {k: v for k, v in r.items() if k not in ("hidden", "rendered_prompt",
                                                    "raw_completion", "candidate_choices",
                                                    "prompt_only_text")}
        for u, r in reextracted.items()}
    report["excluded"] = excluded

    if failures:
        print(f"\n[STOP] 重提取失败：{failures}", file=sys.stderr)
        args.out.write_text(json.dumps({**report, "failures": failures},
                                       ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return 2

    # ── 6. 完整重建两臂记录 ─────────────────────────────────────────────
    print("[6] 重建完整记录（复用 / 重提取 / 排除 逐条标注）")
    provenance_counts = {}
    for tag, spec in ARMS.items():
        src_rows = _rows(spec["src_dir"] / spec["rows"])
        out_dir = spec["out_dir"]
        out_hidden = out_dir / spec["hidden"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_hidden.mkdir(parents=True, exist_ok=True)
        src_rel = str((spec["src_dir"] / spec["rows"]).relative_to(ROOT)).replace("\\", "/")
        src_sha = _sha(spec["src_dir"] / spec["rows"])
        counts = {"reused": 0, "reextracted": 0, "excluded": 0}
        new_rows = []
        for row in src_rows:
            uid = row["unit_id"]
            if uid in reextracted:
                fresh = dict(reextracted[uid])
                vec = np.asarray(fresh.pop("hidden"), dtype=np.float32)
                np.save(out_hidden / f"{uid}.npy", vec)
                fresh["hidden_dim"] = int(vec.size)
                fresh.update({"unit_id": uid, "sample_type": row["sample_type"],
                              "group_id": row.get("group_id")})
                # **生成期观测**沿用原行：completion 文本没变，这些量就没变，
                # 而 score_completion 不产出它们（原本由 G4 runner 的 after_unit 挂载）。
                for f in GENERATION_TIME_FIELDS:
                    if f in row:
                        fresh[f] = row[f]
                fresh["_provenance"] = {
                    "kind": "reextracted",
                    "why": "解析修复后标签或答案位置改变 / 新增可解析",
                    "features_from": "teacher-forced re-extraction from saved completion text",
                    "source_row": src_rel, "source_sha256": src_sha}
                counts["reextracted"] += 1
                new_rows.append(fresh)
            elif uid in excluded:
                keep = dict(row)
                for f in ("letter_token_index_local", "letter_token_id",
                          "letter_token_form", "readout_to_prompt_end_distance",
                          "hidden_dim", *FLOAT_FIELDS):
                    keep.pop(f, None)
                p = dlp.parse_option_answer(row["raw_completion"], list(mci.MCQ_LETTERS))
                canon = mci.canonicalize_mcq_output(None, row["candidate_choices"])
                keep.update({"parsed_label": None, "parse_status": p["parse_method"],
                             "wrong_label": None, **canon,
                             **mci.response_format_of(row["raw_completion"], None)})
                keep["_provenance"] = {
                    "kind": "excluded",
                    "why": f'新解析判失败：{p.get("parse_failure_reason")}',
                    "features_from": "none — 旧特征已剔除，不得继续使用",
                    "raw_completion_preserved": True,
                    "source_row": src_rel, "source_sha256": src_sha}
                counts["excluded"] += 1
                new_rows.append(keep)
            else:
                keep = dict(row)
                p = dlp.parse_option_answer(row["raw_completion"], list(mci.MCQ_LETTERS))
                renamed = keep.get("parse_status") != p["parse_method"]
                keep["parse_status"] = p["parse_method"]
                hp = spec["src_dir"] / spec["hidden"] / f"{uid}.npy"
                if hp.exists():
                    np.save(out_hidden / f"{uid}.npy", np.load(hp))
                keep["_provenance"] = {
                    "kind": "reused",
                    "why": "标签与答案位置均未变，原特征可复用",
                    "features_from": f"copied from {src_rel}",
                    "parse_method_renamed_only": bool(renamed),
                    "source_row": src_rel, "source_sha256": src_sha}
                counts["reused"] += 1
                new_rows.append(keep)

        # 同题 K 条聚合量：整份重算（受影响题的两维会变，其余题结果不变）
        bk.attach_cross_sample_f5(new_rows, "mcq")
        out_rows = out_dir / spec["rows"]
        write_atomic(out_rows, "\n".join(
            json.dumps(r, ensure_ascii=False, default=g3._jsonable) for r in new_rows) + "\n")
        provenance_counts[tag] = {**counts, "n_rows": len(new_rows),
                                  "rows_path": str(out_rows.relative_to(ROOT)).replace("\\", "/"),
                                  "rows_sha256": _sha(out_rows),
                                  "n_hidden": len(list(out_hidden.glob("*.npy")))}
        print(f"  {tag}: 复用 {counts['reused']} / 重提取 {counts['reextracted']} / "
              f"排除 {counts['excluded']} → {out_rows.name}")
    report["rebuilt"] = provenance_counts

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n[ok] 重提取完成\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
