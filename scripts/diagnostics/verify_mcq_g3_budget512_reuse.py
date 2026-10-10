r"""复用前的核验：`g3_budget512/` 里已有的 MCQ G3 证据，是否与**当前代码**的
生成语义和已登记设置一致。

不是"我没改那几个文件"这种自证 —— 这里**用当前代码重建** MCQ G3 的记录与 unit，
逐条比对已落盘行的 `prompt_hash` / `unit_hash` / `rendered_prompt` / `seed`。
提示、chat template、选项洗牌、seed 规则里任何一处变了，重建值就对不上。

只加载 tokenizer，**不加载模型权重、不生成、不改写任何文件**。
任一项不一致 → 退出非零并如实报告差异，**不自行重跑掩盖差异**。

用法：
    python scripts/diagnostics/verify_mcq_g3_budget512_reuse.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as bk              # noqa: E402
from recover_attention.generation import fingerprint as fpmod        # noqa: E402

RUN_DIR = ROOT / "outputs/stage1_llama_v3/g3_budget512"
SAMPLES = ROOT / "outputs/stage1/samples"
REGISTRATION = ROOT / "outputs/stage1_llama_v3/budget512_registration.json"
MODEL = "D:/models/Meta-Llama-3.1-8B-Instruct"
OUT = ROOT / "tmp/stage1_l2_g3/mcq_g3_budget512_reuse_verification.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = V3AnalysisConfig()
    cfg.assert_production()
    failures: list[str] = []
    checks: list[dict] = []

    def ck(label: str, ok: bool, detail=None) -> None:
        checks.append({"check": label, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    # ── 1. 产物清单与哈希 ────────────────────────────────────────────────
    print("[产物] 哈希与完整性")
    artefacts = {}
    for name in ("mcq_g3_smoke_report.json", "mcq_g3_run_fingerprint.json",
                 "mcq_g3_unit_results.jsonl", "mcq_g3_generated_manifest.jsonl"):
        p = RUN_DIR / name
        ck(f"存在 {name}", p.exists())
        if p.exists():
            artefacts[name] = _sha(p)
    hidden_dir = RUN_DIR / "mcq_g3_unit_hidden"
    n_hidden = len(list(hidden_dir.glob("*.npy"))) if hidden_dir.exists() else 0

    report = json.loads((RUN_DIR / "mcq_g3_smoke_report.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            io.open(RUN_DIR / "mcq_g3_unit_results.jsonl", encoding="utf-8") if x.strip()]
    fp_file = json.loads((RUN_DIR / "mcq_g3_run_fingerprint.json").read_text(encoding="utf-8"))

    # ── 2. 报告自身的结论与登记 ──────────────────────────────────────────
    print("[报告] 结论与预算登记")
    ck("passed = true", report.get("passed") is True)
    ck("failed_checks 为空", not report.get("failed_checks"), report.get("failed_checks"))
    ck("dry_run = false", report.get("dry_run") is False)
    ck("落盘 120 行 / 0 失败",
       report.get("n_rows_written") == 120 and report.get("n_failures") == 0)
    ck("报告自身的 every_parsed_record_has_hidden = true",
       (report.get("checks") or {}).get("every_parsed_record_has_hidden") is True)
    ck("unit_results 也是 120 行", len(rows) == 120, len(rows))
    # hidden **按设计**只对 eligible 行产出：解析失败的行不进 population，也就没有 H
    # （MCQBackend.run_unit 在 eligible_for_primary=False 时提前返回）。
    # 所以这里比的是"hidden 数 == eligible 数"，不是"== 行数"。
    n_eligible = sum(1 for r in rows if r.get("eligible_for_primary"))
    ck("hidden 向量数 == eligible 行数（解析失败行按设计无 H）",
       n_hidden == n_eligible, f"{n_hidden} / {n_eligible}")

    reg = g3.load_budget_registration(REGISTRATION, cfg=cfg, model_path=MODEL)
    stamp = report.get("budget_extension") or {}
    ck("报告记录了预算扩展", bool(stamp.get("registered")))
    ck("报告引用的登记 sha256 == 当前登记文件",
       stamp.get("registration_sha256") == reg.sha256, reg.sha256[:16])
    ck("报告记录的实际预算 == 登记值",
       stamp.get("effective_max_new_tokens") == reg.registered_max_new_tokens,
       reg.registered_max_new_tokens)
    ck("登记连到一次**通过的**生成长度检查",
       reg.length_check.get("passed") is True, reg.length_check.get("path"))

    # ── 3. 指纹：报告内与落盘文件一致，且就是当前登记的预算 ──────────────
    print("[指纹] 三处一致")
    fp_report = report.get("fingerprint") or {}
    ck("报告内指纹 == 落盘 run_fingerprint.json",
       json.dumps(fp_report, sort_keys=True) == json.dumps(fp_file, sort_keys=True))
    gen = fp_report.get("generation_params") or {}
    expected_budget = g3.expected_max_new_tokens(g3.ARM_MCQ, cfg, reg)
    ck(f"指纹 max_new_tokens == {expected_budget}",
       gen.get("max_new_tokens") == expected_budget, gen.get("max_new_tokens"))
    ck("指纹口径标记 == 当前 GENERATION_PARAMS_SCHEMA",
       gen.get("schema") == g3.GENERATION_PARAMS_SCHEMA, gen.get("schema"))
    ck("top_k == 20（当前协议常量）",
       gen.get("top_k") == bk._RealBackendBase.QWEN_PROTOCOL_TOP_K, gen.get("top_k"))
    ck("repetition_penalty == 1.05（当前协议常量）",
       gen.get("repetition_penalty") == bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY,
       gen.get("repetition_penalty"))
    ck("K / temperature / top_p / seed 与 cfg 一致",
       (gen.get("K"), gen.get("temperature"), gen.get("top_p"), gen.get("seed"))
       == (cfg.k_traces, 0.7, 0.95, cfg.generation_seed))

    # ── 4. 名单身份：与本轮重新校验的冻结 G3 名单同源 ────────────────────
    print("[名单] 冻结 G3 名单身份")
    verification = g3.verify_inputs(samples_dir=SAMPLES,
                                    prereg=ROOT / "docs/paper/preregistration_v3.md",
                                    lock=ROOT / "docs/paper/preregistration_v3.lock",
                                    cfg=cfg)
    gate_rows = verification.pop("_gate_rows")[g3.ARM_MCQ]
    ck("sample manifest sha256 与报告记录一致",
       report.get("sample_manifest_sha256") == verification["sample_manifest_sha256"])
    ck("六份名单 chain hash 与报告记录逐项一致",
       report.get("chain_sha256_verified") == verification["chain_sha256_verified"])
    ck("prereg sha256 与报告记录一致",
       report.get("prereg_sha256") == verification["g2"]["current_sha256"])

    # ── 5. **语义重建**：当前代码重算 prompt / seed，逐条比对落盘行 ──────
    print("[语义] 用当前代码重建 prompt 与 seed，逐条比对")
    from transformers import AutoTokenizer                       # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records = g3.build_arm_records(g3.ARM_MCQ, gate_rows, cfg=cfg, tokenizer=tok)
    units = g3.units_for(records, cfg.k_traces)
    by_id = {r["example_id"]: r for r in records}
    landed = {r["unit_id"]: r for r in rows}

    ck("重建 unit 数 == 120", len(units) == 120, len(units))
    ck("unit_id 集合与落盘完全一致",
       {u["unit_id"] for u in units} == set(landed), None)

    diff_prompt, diff_ph, diff_uh, diff_seed = [], [], [], []
    for unit in units:
        row = landed.get(unit["unit_id"])
        if row is None:
            continue
        rec = by_id[unit["example_id"]]
        if row.get("rendered_prompt") != rec["rendered_prompt"]:
            diff_prompt.append(unit["unit_id"])
        seed = bk._stable_seed(cfg.generation_seed, rec["example_id"],
                               unit["sample_type"], unit["sample_index"])
        if row.get("seed") != seed:
            diff_seed.append(unit["unit_id"])
        want = fpmod.unit_fingerprint(
            prompt=rec["rendered_prompt"], seed=seed, temperature=0.7, top_p=0.95,
            max_new_tokens=expected_budget, sample_type=unit["sample_type"],
            sample_index=unit["sample_index"])
        if row.get("prompt_hash") != want["prompt_hash"]:
            diff_ph.append(unit["unit_id"])
        if row.get("unit_hash") != want["unit_hash"]:
            diff_uh.append(unit["unit_id"])

    ck("rendered_prompt 120/120 逐条相同", not diff_prompt, diff_prompt[:3])
    ck("seed 120/120 逐条相同（含逐条 seed 规则）", not diff_seed, diff_seed[:3])
    ck("prompt_hash 120/120 逐条相同", not diff_ph, diff_ph[:3])
    ck("unit_hash 120/120 逐条相同（含 max_new_tokens=512）", not diff_uh, diff_uh[:3])

    # 单 BOS：落盘 prompt 用当前 tokenizer 编码后起始标记只应出现一次
    bos_counts = {sum(1 for t in tok(r["rendered_prompt"],
                                     add_special_tokens=False)["input_ids"] if t == 128000)
                  for r in rows}
    ck("提示编码后起始标记恰为 1（单 BOS 口径）", bos_counts == {1}, sorted(bos_counts))

    # ── 6. 判据：报告记录的就是当前模块的 G3 判据 ────────────────────────
    print("[判据] 与当前模块一致")
    crit = report.get("criteria") or {}
    ck("mcq_max_parse_failure_rate == 当前模块默认",
       crit.get("mcq_max_parse_failure_rate") == g3.CRITERIA.mcq_max_parse_failure_rate)
    ck("n_prompts / traces_per_prompt / n_traces == 20 / 6 / 120",
       (crit.get("n_prompts"), crit.get("traces_per_prompt"), crit.get("n_traces"))
       == (20, 6, 120))
    ck("几何 = Llama（4096 / block 22 / tuple 23）",
       (crit.get("hidden_dim"), crit.get("block_index_zero_based"),
        crit.get("hidden_states_tuple_index")) == (4096, 22, 23))

    doc = {
        "schema": "stage1_mcq_g3_budget512_reuse_verification_v1",
        "date": "2026-09-14",
        "purpose": ("复用前核验：已有 MCQ G3 证据与**当前代码**的生成语义、"
                    "已登记预算、冻结名单身份是否一致"),
        "method": ("用当前代码重建记录与 unit，逐条比对落盘行的 rendered_prompt / "
                   "seed / prompt_hash / unit_hash；只加载 tokenizer，不加载模型权重"),
        "run_dir": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "artefact_sha256": artefacts,
        "hidden_vectors": n_hidden,
        "budget_registration": reg.as_dict(),
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failures),
        "failed": failures,
        "verdict": ("一致 —— 可直接复用本次已通过的 MCQ G3，不重新生成、不改写"
                    if not failures else "不一致 —— 已停止，不自行重跑掩盖差异"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n[{'ok   ' if not failures else 'STOP '}] "
          f"{len(checks) - len(failures)}/{len(checks)} 通过")
    print(f"[判定] {doc['verdict']}")
    print(f"[out  ] {OUT}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
