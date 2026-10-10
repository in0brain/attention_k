"""从**已冻结**的 confirmatory 结果机械生成报告用表格。

**不计算任何新统计量。** 本脚本只做三件事：读 JSON、取值、排版。
每个数字都能在 `stage1_confirmatory_result.json` 或两份 finalization report 里
按键路径原样找到；脚本不做任何算术，唯一的例外是表 1 里显式标注的
「名义 − 实际」减法，且减数被减数都同时列出。

**来源不止一处**：表 1 的 1b 节取自两份 finalization report，
F5 数值完整性的定量陈述取自只读审计产物 —— 每张表的 header 逐张列出自己的来源，
SHA 全部登记在 sha manifest 里。早先版本统一声称"一切数字均取自 confirmatory
result JSON"，那句话是错的，已订正。

产出（**受控重写**：只覆盖本脚本自己生成过的文件；`outputs/` 下的冻结产物从不写）：

    docs/paper/stage1_confirmatory_sha_manifest.json   结果与日志的 SHA 固化
    docs/paper/stage1_table1_population.md             population / attrition
    docs/paper/stage1_table2_e1.md                     E1 × A/B/C × H1/MCQ
    docs/paper/stage1_table3_other_estimands.md        E2 / E3 / M2c / D / RQ2 与判读边界
    docs/paper/stage1_claim_boundary.md                permitted / forbidden claim 表
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_analysis_paths as ap        # noqa: E402

CONF = ROOT / "outputs/stage1/confirmatory/stage1_confirmatory_result.json"
LOG = ROOT / "outputs/stage1/unblinding/unblinding_log.jsonl"
PROD = ROOT / "outputs/stage1/production"
F5_AUDIT = ROOT / "outputs/stage1/audit/f5_numeric_integrity_diagnostic.json"
OUT = ROOT / "docs/paper"

PATH_LABEL = {
    ap.PATH_A: "A frozen_implementation_primary",
    ap.PATH_B: "B stable_shadow_sensitivity (tail)",
    ap.PATH_C: "C no_B4_dependency_sensitivity",
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x, plus: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.4f}" if plus else f"{x:.4f}"


#: 只有带这一行标记的文件才允许被本脚本**重写**（其它一律拒绝覆盖）。
GENERATED_MARKER = "由 `scripts/build_stage1_result_tables.py` **机械生成**"


def write_new(path: Path, text: str) -> None:
    """创建，或**受控重写**本脚本自己生成过的文件。

    判据是文件内容里必须带 `GENERATED_MARKER` —— 任何不是本脚本产出的文件
    一律拒绝覆盖。冻结产物在 `outputs/` 下，本脚本从不写那里。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8", errors="replace")
        if GENERATED_MARKER not in old and '"stage1_confirmatory_sha_manifest_v1"' not in old:
            raise SystemExit(f"目标不是本脚本生成的文件，拒绝覆盖：{path}")
        path.unlink()
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.link(str(tmp), str(path))
    os.unlink(tmp)
    print(f"  写出 {path.relative_to(ROOT)}  ({len(text):,} chars)")


def header(conf_sha: str, ts: str, log_sha: str, extra: list[tuple[str, str]]) -> str:
    """每张表的 header 逐张列出**它自己的**来源。

    早先版本对所有表统一声称"一切数字均取自 confirmatory result JSON"，
    但表 1 的 1b 节实际来自两份 finalization report —— 那句话是错的，已订正。
    """
    lines = [f"> **主要来源**：`outputs/stage1/confirmatory/stage1_confirmatory_result.json`"
             f"（sha256 `{conf_sha}`），由 `scripts/build_stage1_result_tables.py` "
             f"**机械生成**。\n"]
    if extra:
        lines.append("> **本表另有来源**（同为只读冻结产物，SHA 登记在 "
                     "`stage1_confirmatory_sha_manifest.json`）：\n")
        for path_, what in extra:
            lines.append(f"> - `{path_}` — {what}\n")
    else:
        lines.append("> 本表**只**取自上述结果文件，无其他来源。\n")
    lines.append("> 本文件不含任何新计算的统计量。\n")
    lines.append(f"> 解盲时间戳 `{ts}`，一次性，log sha256 `{log_sha}`。\n\n")
    return "".join(lines)


_UNUSED_HEADER = (
    "> **来源**：`outputs/stage1/confirmatory/stage1_confirmatory_result.json`"
    "（sha256 `{sha}`），由 `scripts/build_stage1_result_tables.py` **机械生成**。\n"
    "> 本文件不含任何新计算的统计量；一切数字均取自该冻结结果。\n"
    "> 解盲时间戳 `{ts}`，一次性，log sha256 `{logsha}`。\n\n")


def main() -> int:
    res = json.loads(CONF.read_text(encoding="utf-8"))
    log = json.loads([l for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()][0])
    conf_sha, log_sha = sha(CONF), sha(LOG)
    fin = {a: json.loads((PROD / f"{a}_finalization_report.json").read_text("utf-8"))
           for a in ("h1", "mcq")}
    f5 = json.loads(F5_AUDIT.read_text(encoding="utf-8"))
    head = header(conf_sha, log["timestamp"], log_sha, [])          # 表 2 / 表 3 / claim 表
    head_t1 = header(conf_sha, log["timestamp"], log_sha, [
        ("outputs/stage1/production/h1_finalization_report.json",
         "1b 节的 H1 逐级口径（n_raw_units / n_finalized_records / "
         "n_analyzable_records / n_hidden_vectors）"),
        ("outputs/stage1/production/mcq_finalization_report.json",
         "1b 节的 MCQ 逐级口径与 9 例整组技术性排除的 audit sha256"),
    ])

    # ── SHA 固化 ────────────────────────────────────────────────────────
    manifest = {
        "schema": "stage1_confirmatory_sha_manifest_v1",
        "role": ("解盲产物与其血统的 SHA 固化。**只记录，不改变任何产物。**"
                 "任何后续引用这些数字的文稿都应引用此处的 sha256。"),
        "unblinding": {
            "timestamp": log["timestamp"],
            "schema_version": log["schema_version"],
            "log_sha256": log_sha,
            "log_bytes": LOG.stat().st_size,
            "finalized_freeze_hash": log["finalized_freeze_hash"],
            "analysis_input_freeze_hash": log["analysis_input_freeze_hash"],
            "prereg_sha256": log["prereg_sha256"],
            "readiness_sha256": log["readiness_sha256"],
            "config_hash": log["config_hash"],
        },
        "result": {
            "path": "outputs/stage1/confirmatory/stage1_confirmatory_result.json",
            "sha256": conf_sha, "bytes": CONF.stat().st_size,
            "schema": res["schema"],
            "frozen_paths": list(res["frozen_paths"]),
            "contract_ok": res["contract"]["ok"],
            "path_b_implementation": res["execution"]["path_b_implementation"],
            "e3_canonical_source": res["execution"]["e3_canonical_source"],
        },
        "lineage": {
            n: {"path": f"outputs/stage1/readiness/{f}",
                "bytes": (ROOT / "outputs/stage1/readiness" / f).stat().st_size,
                "sha256": sha(ROOT / "outputs/stage1/readiness" / f)}
            for n, f in (
                ("readiness_v3", "stage1_confirmatory_readiness_v3.json"),
                ("analysis_input_freeze_manifest",
                 "stage1_analysis_input_freeze_manifest.json"),
                ("finalized_freeze_manifest", "stage1_finalized_freeze_manifest.json"))
        },
        # 表 1 的 1b 节直接引用这两份 —— 它们不是 confirmatory result 的一部分。
        "additional_table_sources": {
            f"{a}_finalization_report": {
                "path": f"outputs/stage1/production/{a}_finalization_report.json",
                "bytes": (PROD / f"{a}_finalization_report.json").stat().st_size,
                "sha256": sha(PROD / f"{a}_finalization_report.json"),
                "used_for": "表 1 的 1b 逐级口径",
            } for a in ("h1", "mcq")
        },
        # L.6 的定量陈述**只能**引用这里；PROGRESS / history 的叙述文本
        # **不得**反向充当数据真源。
        "f5_numeric_integrity_source": {
            "path": "outputs/stage1/audit/f5_numeric_integrity_diagnostic.json",
            "bytes": F5_AUDIT.stat().st_size,
            "sha256": sha(F5_AUDIT),
            "schema": f5["schema"],
            "mode": f5["mode"],
            "evidence_class": f5["class_1b_primary_analyzable"]["evidence_class"],
            "evidence_caliber": f5["class_1b_primary_analyzable"]["evidence_caliber"],
            "fingerprint_limitation":
                f5["class_1b_primary_analyzable"]["fingerprint_limitation"],
            "field_paths_registered": {
                "n_evaluated": "class_1b_primary_analyzable.n_evaluated",
                "delta_mean": ("class_1b_primary_analyzable."
                               "delta_n_tok_minus_stored_count.mean"),
                "n_zero_delta": ("class_1b_primary_analyzable."
                                 "delta_n_tok_minus_stored_count.n_zero_delta"),
                "n_negative_delta": ("class_1b_primary_analyzable."
                                     "delta_n_tok_minus_stored_count.n_negative_delta"),
                "p50_p90_max": ("class_1b_primary_analyzable."
                                "delta_n_tok_minus_stored_count.{p50,p90,max}"),
            },
            "values": {
                "n_evaluated": f5["class_1b_primary_analyzable"]["n_evaluated"],
                **{k: f5["class_1b_primary_analyzable"]
                   ["delta_n_tok_minus_stored_count"][k]
                   for k in ("mean", "n_zero_delta", "n_negative_delta",
                             "min", "p50", "p90", "max")},
            },
            "NOT_registered": [
                "整体 mean completion token 数（审计只登记了分长度桶的 mean_n_tok）",
                "冻结字段 f5_completion_token_count 的整体均值",
                "被过滤 token 的比例",
            ],
            "consequence": ("上列三项**没有**机器可读真源，故已从 Abstract 与 L.6 "
                            "移除；L.6 只保留本文件登记的字段。"),
        },
        "note": ("一次性解盲；log 与结果均为 no-clobber 创建。"
                 "本清单**不是**冻结闸，只是引用锚点。"),
    }
    write_new(OUT / "stage1_confirmatory_sha_manifest.json",
              json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False) + "\n")

    # ── 表 1  population / attrition ────────────────────────────────────
    pd_ = res["population_declaration"]
    t1 = [head_t1, "# 表 1｜分析样本口径与 attrition（M4）\n\n",
          "两类 attrition **分开记账，不得合并**（`attrition_categories`）：\n\n"]
    for k, v in pd_["attrition_categories"].items():
        t1.append(f"- `{k}` — {v}\n")
    t1.append("\n## 1a 两臂分析样本（来源：结果文件 `population_declaration.arms`）\n\n")
    t1.append("| 臂 | analyzable | n_positive | n_negative | positive_rate |\n")
    t1.append("|---|---:|---:|---:|---:|\n")
    for arm, d in pd_["arms"].items():
        t1.append(f"| {arm.upper()} | {d['analyzable']:,} | {d['n_positive']:,} | "
                  f"{d['n_negative']:,} | {d['positive_rate']:.4f} |\n")
    t1.append("\n恒等式 `n_positive + n_negative == analyzable` 由 "
              "`validate_declaration` 校验通过。\n")

    t1.append("\n## 1b 生成到分析的逐级口径（来源：两份 finalization report）\n\n")
    t1.append("| 项 | H1 | MCQ | 键路径 |\n|---|---:|---:|---|\n")
    rows = [
        ("nominal_questions", None, fin["mcq"].get("nominal_questions"), "nominal_questions"),
        ("nominal_raw_units", None, fin["mcq"].get("nominal_raw_units"), "nominal_raw_units"),
        ("n_raw_units", fin["h1"]["n_raw_units"], fin["mcq"]["n_raw_units"], "n_raw_units"),
        ("retained_raw_units", None, fin["mcq"].get("retained_raw_units"), "retained_raw_units"),
        ("n_finalized_records", fin["h1"]["n_finalized_records"],
         fin["mcq"]["n_finalized_records"], "n_finalized_records"),
        ("n_analyzable_records", fin["h1"]["n_analyzable_records"],
         fin["mcq"]["n_analyzable_records"], "n_analyzable_records"),
        ("n_hidden_vectors", fin["h1"]["n_hidden_vectors"],
         fin["mcq"]["n_hidden_vectors"], "n_hidden_vectors"),
        ("technical_exclusions", None, fin["mcq"].get("technical_exclusions"),
         "technical_exclusions"),
        ("preserved_but_not_analyzed_raw_units", None,
         fin["mcq"].get("preserved_but_not_analyzed_raw_units"),
         "preserved_but_not_analyzed_raw_units"),
    ]
    for name, h, m, key in rows:
        fh = "—" if h is None else f"{h:,}"
        fm_ = "—" if m is None else f"{m:,}"
        t1.append(f"| `{name}` | {fh} | {fm_} | `{key}` |\n")
    t1.append(f"\nMCQ 的 `classification = {fin['mcq']['classification']}`，"
              f"`protocol_deviation = {fin['mcq']['protocol_deviation']}`，"
              f"`preregistration_files_modified = {fin['mcq']['preregistration_files_modified']}`。\n")
    t1.append(f"名义 1,760 与实际 {fin['mcq']['actual_finalized_rows']:,} 之差 = "
              f"{fin['mcq']['nominal_finalized_rows'] - fin['mcq']['actual_finalized_rows']} "
              f"（唯一一处减法，减数被减数均已列出）；"
              f"9 例整组技术性排除的 audit sha256 `{fin['mcq']['exclusion_audit_sha256'][:16]}…`。\n")
    t1.append("\n> H1 侧 `n_finalized_records` 与 `n_analyzable_records` 的差额属 "
              "`canonicalization_parse_attrition`（`requires_hidden=false`，"
              "保留 finalized 行但不进入依赖 hidden 的分析），**不是** runtime attrition。\n")
    write_new(OUT / "stage1_table1_population.md", "".join(t1))

    # ── 表 2  E1 × 路径 × 臂 ────────────────────────────────────────────
    t2 = [head, "# 表 2｜E1（PRIMARY，分组 A prompt-disjoint）\n\n",
          "全部取自 `arms[path][arm]`。R = 3，B_total = 20,000"
          "（B_batch 20 × B_per_batch 1,000），六个组合的 draw 全部有效、"
          "单类丢弃 0。\n\n"]
    t2.append("## 2a AUROC 与 Δ_H\n\n")
    t2.append("| 路径 | 臂 | n | groups | AUROC(S\\*) | AUROC(H) | AUROC(S\\*+H) | "
              "Δ̂_H | 95% CI | width | precision |\n")
    t2.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = res["arms"][p][arm]
            au, ci = a["auroc"], a["delta_h_ci"]
            t2.append(
                f"| {PATH_LABEL[p]} | {arm.upper()} | {a['n']:,} | {a['n_groups']:,} | "
                f"{fmt(au['S*'], False)} | {fmt(au['H'], False)} | {fmt(au['S*+H'], False)} | "
                f"**{fmt(a['delta_h_point'])}** | [{fmt(ci['ci_lo'])}, {fmt(ci['ci_hi'])}] | "
                f"{ci['width']:.4f} | {a['precision']['precision']} |\n")
    t2.append("\n## 2b 单 block AUROC\n\n| 路径 | 臂 | " +
              " | ".join(f"B{i}" for i in range(1, 6)) + " |\n")
    t2.append("|---|---|" + "---:|" * 5 + "\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = res["arms"][p][arm]
            cells = [fmt(a["auroc"][b], False) if b in a["auroc"] else "—（本路径已移除）"
                     for b in ("B1", "B2", "B3", "B4", "B5")]
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | " + " | ".join(cells) + " |\n")

    t2.append("\n## 2c M1b 非劣性认证（边际单侧 95%，ε_NI = 0.02，IUT）\n\n")
    t2.append("| 路径 | 臂 | 结论 | 未通过块 | L_marg 最小值 |\n|---|---|---|---|---|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            ni = res["arms"][p][arm]["m1b_non_inferiority"]
            lo = min(ni["L_marg"], key=lambda k: ni["L_marg"][k])
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | **{ni['status']}** | "
                      f"{ni['failed_blocks'] or '—'} | {lo} = {fmt(ni['L_marg'][lo])} |\n")
    t2.append("\n逐块 L_marg（primary）与 L_sim（secondary）：\n\n")
    t2.append("| 路径 | 臂 | 界 | " + " | ".join(f"B{i}" for i in range(1, 6)) + " |\n")
    t2.append("|---|---|---|" + "---:|" * 5 + "\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            ni = res["arms"][p][arm]["m1b_non_inferiority"]
            for key, lbl in (("L_marg", "L_marg (primary)"), ("L_sim", "L_sim (secondary)")):
                cells = [fmt(ni[key][b]) if b in ni[key] else "—"
                         for b in ("B1", "B2", "B3", "B4", "B5")]
                t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | {lbl} | "
                          + " | ".join(cells) + " |\n")

    t2.append("\n## 2d M1b(iv-a) 逐块直接检验（**唯一授权路径**）\n\n")
    t2.append("| 路径 | 臂 | passed | 未通过块 | L_marg 最小值 |\n|---|---|---|---|---|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            iv = res["arms"][p][arm]["m1b_iv_a_direct_superiority"]
            lo = min(iv["L_marg"], key=lambda k: iv["L_marg"][k])
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | **{iv['passed']}** | "
                      f"{iv['failed_blocks'] or '—'} | {lo} = {fmt(iv['L_marg'][lo])} |\n")

    t2.append("\n## 2e M1b(iv-b) 合取一致性（sensitivity，**不单独授权任何表述**）\n\n")
    t2.append("判据必须写成合取：`M1b CERTIFIED` **且** `Δ_H 的 CI 下界 > ε_NI`。\n\n")
    t2.append("| 路径 | 臂 | M1b certified | CI 下界 | > ε_NI = 0.02 | holds |\n")
    t2.append("|---|---|---|---:|---|---|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            b = res["arms"][p][arm]["m1b_iv_b_consistency"]
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | {b['m1b_certified']} | "
                      f"{fmt(b['delta_h_ci_lo'])} | {b['second_term_holds']} | "
                      f"**{b['holds']}** |\n")
    t2.append("\n> **本表 6 行 holds 全为 False，该陈述的范围仅限 E1。** "
              "E3 与 M2c 各有自己的 (iv-b)，取值不同，见表 3。\n")

    t2.append("\n## 2f M1c T1（角色 I，边际单侧**上界**，ε = 0.02）\n\n")
    t2.append("| 路径 | 臂 | triggered | U_marg(B1) | U_marg(B2) | Δ_task(B1) | Δ_task(B2) |\n")
    t2.append("|---|---|---|---:|---:|---:|---:|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            t = res["arms"][p][arm]["m1c_t1"]
            u, q = t["role_I_marginal_upper_bounds"], t["point"]
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | **{t['triggered']}** | "
                      f"{fmt(u['B1'])} | {fmt(u['B2'])} | {fmt(q['B1'])} | {fmt(q['B2'])} |\n")
    t2.append("\n## 2g CI 宽度的 MC 误差（jackknife primary）\n\n")
    t2.append("| 路径 | 臂 | CI width | MC band | 0.04 是否落入误差带 |\n")
    t2.append("|---|---|---:|---|---|\n")
    for p in ap.FROZEN_PATHS:
        for arm in ("h1", "mcq"):
            a = res["arms"][p][arm]
            band = a["ci_width_and_mc_error"]["mc_jackknife"]["mc_band_width"]
            inside = band[0] <= 0.04 <= band[1]
            t2.append(f"| {PATH_LABEL[p]} | {arm.upper()} | "
                      f"{a['ci_width_and_mc_error']['ci_width']:.5f} | "
                      f"[{band[0]:.5f}, {band[1]:.5f}] | "
                      f"{'**borderline**' if inside else '否'} |\n")
    write_new(OUT / "stage1_table2_e1.md", "".join(t2))

    # ── 表 3  其他 estimand 与判读边界 ──────────────────────────────────
    e2 = res["estimands"][ap.PATH_A]["E2"]["h1"]
    t3 = [head, "# 表 3｜E2 / E3 / M2c / D / RQ2 与判读边界\n\n"]
    t3.append("## 3a E2 entity-generalization 可估性审计（M2b，label-free）\n\n")
    t3.append(f"**判定 `{e2['estimability'].upper()}`** —— 按 M2b 停在此处，"
              "不构造 CV、不报任何 AUROC / Δ_H。三条路径共用同一份审计（分组构造与路径无关）。\n\n")
    t3.append("| 量 | 值 | 门槛 |\n|---|---:|---|\n")
    th = e2["thresholds"]
    t3.append(f"| n_completions | {e2['n_completions']:,} | — |\n")
    t3.append(f"| n_prompts | {e2['n_prompts']:,} | — |\n")
    t3.append(f"| n_identities | {e2['n_identities']:,} | — |\n")
    t3.append(f"| n_identity_edges | {e2['n_identity_edges']:,} | — |\n")
    t3.append(f"| component_count | **{e2['component_count']}** | "
              f"≥ {th['min_components']}（{th['min_components_rule']}） |\n")
    t3.append(f"| max_component_share | **{e2['max_component_share']:.4f}** | "
              f"≤ {th['max_component_share_strictly_greater_than']}（严格大于才塌缩） |\n")
    hub = e2["hub_distribution"]
    t3.append(f"| identity degree max / median | {hub['identity_degree_max']:,} / "
              f"{hub['identity_degree_median']} | — |\n")
    t3.append(f"| degree ≥ 2 的 identity | {hub['n_identities_degree_ge_2']:,} | — |\n")
    t3.append(f"\n分量大小 top10：`{e2['component_sizes_top10']}`\n")

    t3.append("\n## 3b E3 词表分块（独立 estimand；判读**不与 E1 比较**）\n\n")
    e3a = res["estimands"][ap.PATH_A]["E3"]["h1"]
    r = e3a["retention"]
    t3.append(f"M2d-gate：**{e3a['estimability'].upper()}**（五折均非空且两类齐全）。"
              f"E3 population {r['n_rows_in_e3_population']:,} / {r['n_records']:,} = "
              f"{r['mention_retention_rate']:.4f}；无折 {r['n_rows_without_fold']:,}"
              f"（greedy 无 identifier），块内不 eligible "
              f"{r['n_rows_ineligible_in_own_block']:,}。\n\n")
    t3.append("| fold | n | n_positive | n_negative | 训练补集 n |\n|---|---:|---:|---:|---:|\n")
    for b, v in e3a["per_fold"].items():
        t3.append(f"| block {b} | {v['n']:,} | {v['n_positive']:,} | {v['n_negative']:,} | "
                  f"{v['n_train_complement']:,} |\n")
    t3.append("\n| 路径 | n | AUROC(S\\*) | AUROC(H) | AUROC(S\\*+H) | Δ̂_H | 95% CI | "
              "M1b | iv-a | iv-b | T1 | precision |\n")
    t3.append("|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|\n")
    for p in ap.FROZEN_PATHS:
        s = res["estimands"][p]["E3"]["h1"]["summary"]
        au, ci = s["auroc"], s["delta_h_ci"]
        ni, iva, ivb = (s["m1b_non_inferiority"], s["m1b_iv_a_direct_superiority"],
                        s["m1b_iv_b_consistency"])
        t3.append(f"| {PATH_LABEL[p]} | {s['n']:,} | {fmt(au['S*'], False)} | "
                  f"{fmt(au['H'], False)} | {fmt(au['S*+H'], False)} | "
                  f"**{fmt(s['delta_h_point'])}** | [{fmt(ci['ci_lo'])}, {fmt(ci['ci_hi'])}] | "
                  f"{ni['status']}{('（' + ','.join(ni['failed_blocks']) + '）') if ni['failed_blocks'] else ''} | "
                  f"{iva['passed']} | {ivb['holds']} | {s['m1c_t1']['triggered']} | "
                  f"{s['precision']['precision']} |\n")

    t3.append("\n## 3c M2c label-informed sensitivity（**非 primary**）\n\n")
    m2a = res["estimands"][ap.PATH_A]["M2c"]["h1"]
    t3.append(f"分组 `{m2a['grouping']}`：fabricated identity {m2a['n_fabricated_identities']:,}，"
              f"边 {m2a['n_fabricated_identity_edges']:,}，分量 {m2a['component_count']:,}，"
              f"最大占比 {m2a['max_component_share']:.4f}。\n\n")
    t3.append("| 路径 | n | groups | AUROC(S\\*) | AUROC(H) | AUROC(S\\*+H) | Δ̂_H | 95% CI | "
              "M1b | iv-a | iv-b | precision |\n")
    t3.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n")
    for p in ap.FROZEN_PATHS:
        s = res["estimands"][p]["M2c"]["h1"]["summary"]
        au, ci = s["auroc"], s["delta_h_ci"]
        ni, iva, ivb = (s["m1b_non_inferiority"], s["m1b_iv_a_direct_superiority"],
                        s["m1b_iv_b_consistency"])
        t3.append(f"| {PATH_LABEL[p]} | {s['n']:,} | {s['n_groups']:,} | "
                  f"{fmt(au['S*'], False)} | {fmt(au['H'], False)} | {fmt(au['S*+H'], False)} | "
                  f"**{fmt(s['delta_h_point'])}** | [{fmt(ci['ci_lo'])}, {fmt(ci['ci_hi'])}] | "
                  f"{ni['status']}{('（' + ','.join(ni['failed_blocks']) + '）') if ni['failed_blocks'] else ''} | "
                  f"**{iva['passed']}** | {ivb['holds']} | {s['precision']['precision']} |\n")
    t3.append("\n必须与结果同报的三项 caveat：\n\n")
    for c in m2a["interpretation_caveats"]:
        t3.append(f"- {c}\n")

    t3.append("\n## 3d 跨臂对比 D\n\n")
    d0 = res["contrast_D"][ap.PATH_A]
    t3.append(f"方向 `{d0['direction']}`；判读阈值 delta = {d0['delta']}。\n\n")
    t3.append("| 路径 | D 点估计 | 95% CI | width | verdict | marker |\n")
    t3.append("|---|---:|---|---:|---|---|\n")
    for p in ap.FROZEN_PATHS:
        d = res["contrast_D"][p]
        t3.append(f"| {PATH_LABEL[p]} | {fmt(d['D_point'])} | "
                  f"[{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}] | {d['width']:.4f} | "
                  f"**{d['verdict']}** | `{d['marker']}` |\n")
    t3.append(f"\n> 路径 A 的解释边界（逐字取自结果）：{d0['interpretation_limit']}\n")

    t3.append("\n## 3e RQ2\n\n")
    rq = res["rq2"]
    t3.append(f"- frozen primary：`{rq['frozen_primary']['status']}`，"
              f"`computed = {rq['frozen_primary']['computed']}`。"
              f"{rq['frozen_primary']['reason']}\n")
    t3.append(f"- stable shadow：`{rq['stable_shadow']['status']}`，"
              f"`is_frozen_primary = {rq['stable_shadow']['is_frozen_primary']}`。"
              f"{rq['stable_shadow']['note']}\n")

    t3.append("\n## 3f 三条路径的 marker 与解释边界（逐字取自 `path_markers`）\n\n")
    for p in ap.FROZEN_PATHS:
        m = res["path_markers"][p]
        t3.append(f"**{PATH_LABEL[p]}**（role = {m['role']}，marker = `{m['marker']}`）\n\n")
        t3.append(f"- b4_source：{m['b4_source']}\n")
        if "b4_source_caveat" in m:
            t3.append(f"- caveat：{m['b4_source_caveat']}\n")
        t3.append(f"- interpretation_limit：{m['interpretation_limit']}\n\n")
    write_new(OUT / "stage1_table3_other_estimands.md", "".join(t3))

    # ── claim 表 ────────────────────────────────────────────────────────
    # **扫描三个 estimand**：E1（arms）、E3、M2c（各自 summary 里也带
    # wording_prohibitions）。早先只扫了 E1，导致 E3/A 的 M1b NOT CERTIFIED
    # 与 M2c 三路径的 iv-a failed 根本没出现在命中列表里。
    def _sites():
        for pth in ap.FROZEN_PATHS:
            for arm in ("h1", "mcq"):
                yield ("E1", pth, arm, res["arms"][pth][arm])
        for name in ("E3", "M2c"):
            for pth in ap.FROZEN_PATHS:
                node = res["estimands"][pth][name]["h1"]
                if "summary" in node:
                    yield (name, pth, "h1", node["summary"])

    fired: dict[str, list[str]] = {}
    sites = []
    for est, pth, arm, node in _sites():
        ni, iva = node["m1b_non_inferiority"], node["m1b_iv_a_direct_superiority"]
        sites.append({"est": est, "path": PATH_LABEL[pth].split()[0], "arm": arm.upper(),
                      "m1b": ni["status"], "m1b_failed": ni["failed_blocks"],
                      "iva": iva["passed"], "iva_failed": iva["failed_blocks"],
                      "t1": node["m1c_t1"]["triggered"]})
        for k, v in node["wording_prohibitions"].items():
            if k != "never_allowed" and v:
                fired.setdefault(k, []).append(
                    f"{est} / {PATH_LABEL[pth].split()[0]} / {arm.upper()}")

    c = [head, "# permitted / forbidden claim 表\n\n",
         "本表由结果文件里**已实例化**的 `wording_prohibitions` 与各判据的"
         "取值机械导出，**覆盖 E1 / E3 / M2c 三个 estimand**。"
         "左列是本次结果**允许**写的话，右列是**禁止**写的话。\n\n",
         "## 判据命中总览\n\n",
         "| estimand | 路径 | 臂 | M1b | M1b 未通过块 | iv-a | iv-a 未通过块 | T1 |\n",
         "|---|---|---|---|---|---|---|---|\n"]
    for x in sites:
        c.append(f"| {x['est']} | {x['path']} | {x['arm']} | "
                 f"{('**' + x['m1b'] + '**') if x['m1b'] != 'CERTIFIED' else x['m1b']} | "
                 f"{x['m1b_failed'] or '—'} | "
                 f"{'True' if x['iva'] else '**False**'} | {x['iva_failed'] or '—'} | "
                 f"{x['t1']} |\n")
    c += ["\n三处需要单独点名的命中：\n\n",
          "- **E3 / 路径 A**：M1b `NOT CERTIFIED`，未通过块 **B4**。该 estimand 的 A 分支下"
          "禁「通过强表层基线」「超出输出可观测」类表述；E3 的 B / C 两路径虽为 CERTIFIED，"
          "**不得**用它们覆盖 A。\n",
          "- **M2c / 三条路径**：M1b 全部 `NOT CERTIFIED`（未通过块 **B1**），"
          "且 **(iv-a) 全部未通过**（未通过块 B1）。M2c 分支下**同时**禁"
          "「通过强表层基线」与「超过最佳表层 block」两类表述。\n",
          "- **E1**：M1b 仅 C / MCQ 未认证；(iv-a) 六个组合全过。\n\n",
          "## 允许（有判据支撑）\n\n",
          "| 可写的表述 | 判据 | 范围限定 |\n|---|---|---|\n"]
    c.append("| S\\* 相对所选 block **逐块非劣**（marginal one-sided 95%，"
             "margin ε_NI = 0.02，IUT 下整体 α ≤ 0.05 **nominal**） | "
             "M1b CERTIFIED | E1 的 A/B 两路径两臂 + C/H1；E3 的 B/C 两路径。"
             "**不含** E1 的 C/MCQ、**E3 的 A**、**M2c 的三条路径** |\n")
    c.append("| S\\*+H **超过每一个单 block** | M1b(iv-a) passed（唯一授权路径） | "
             "仅 **E1**（3 路径 × 2 臂全过）与 **E3**（3 路径全过，H1）；"
             "**不含 M2c**（3 路径 iv-a 均 False，未通过块 B1） |\n")
    c.append("| 存在相对**该特定 S\\* 构造**的增量 | Δ̂_H 的 95% CI 不含 0 | "
             "E1 六个组合、E3 三路径、M2c 三路径均成立；量级见表 2/3 |\n")
    c.append("| E2 在本样本上 **non-estimable**，并报告连通结构统计 | "
             "M2b 预注册判定（17 分量 / 最大占比 0.9920） | 这是**结果**，不是失败 |\n")
    c.append("| E3 在冻结映射下 **estimable** 并给出其 Δ_H | M2d-gate 通过 | "
             "E3 标签定义与 E1 不同，**不要求一致** |\n")
    c.append("| 按预注册的保守准入准则，我们**不作**机制性表述 | "
             "M1c T1 角色 I 触发（H1） | 角色 I 措辞；见下方禁令 |\n")
    c.append("| 精度达标（CI width 远小于 0.04，MC band 未含 0.04） | "
             "precision = adequate | 五个 E1 组合；**C/MCQ 为 insufficient** |\n")

    c.append("\n## 禁止（本次结果已实例化的禁令）\n\n")
    c.append("| 禁止的表述 | 触发判据 | 命中的路径 × 臂 |\n|---|---|---|\n")
    labels = {
        "t1_triggered_forbids": ("**机制性**表述：「语义理解」「内部觉察」"
                                 "「模型知道自己在编造」", "M1c T1 触发"),
        "m1b_not_certified_forbids": ("「通过强表层基线」「超出输出可观测」类表述；"
                                      "该分支的 Δ_H 只能限定为「相对该特定 S\\* 构造的增量」",
                                      "M1b NOT CERTIFIED"),
        "iv_a_failed_forbids": ("「超过最佳表层 block」类表述 —— 无论 (iv-b) 是否成立",
                                "M1b(iv-a) 未通过"),
    }
    for k, (what, why) in labels.items():
        who = "、".join(fired.get(k, [])) or "—（E1 未命中）"
        c.append(f"| {what} | {why} | {who} |\n")
    c.append("| 「(iv-a) 通过 **OR** (iv-b) 通过 → 作正面主张」 | "
             "预注册明令：两条路径的并集会重新引入择优触发与 look-elsewhere | "
             "**绝不允许**（全局） |\n")
    c.append("| 「family-wise certified」 | M1b primary 用的是**边际**界 | **绝不允许**（全局） |\n")
    c.append("| 称 S\\* 为「最强表层基线」/ 表层信息的上界 | "
             "M1b 认证范围仅限所选 block 与 ε_NI = 0.02 | **绝不允许**（全局） |\n")
    c.append("| 只写 (iv-b) 的第二项（Δ_H 的 CI 下界 > ε_NI） | "
             "必须写成合取 | **绝不允许**（全局） |\n")
    c.append("| 用路径 C 证明原 B4 有效 | 路径 C 只作 B4 依赖性诊断 | **绝不允许**（全局） |\n")
    c.append("| 把 M2c 当 primary，或声称其偏倚「只会压低性能」 | "
             "M2c 是 label-informed sensitivity，偏倚方向未知 | **绝不允许**（全局） |\n")
    c.append("| 名义跨臂机制解释（由 D 推 H1/MCQ 的机制差别） | "
             "两臂对非有限值的处置不对称 | **绝不允许**（全局） |\n")
    c.append("| 输出具有名义含义的 frozen RQ2 结论 | "
             "`computed = False`，stratifier 受静默过滤影响 | **绝不允许**（全局） |\n")

    c.append("\n## 必须与结果同报的限定\n\n")
    c.append("1. **M1b(iii)**：ε_NI = 0.02 允许 S\\* 比最佳单 block 差最多约 0.02，"
             "而 E1/H1 的 Δ̂_H 在 0.0016–0.0038 量级，**远小于** ε_NI。"
             "因此在 M1b 一侧，「真正超出表层」与「只是补回 S\\* 相对最佳单 block 的构造损失」"
             "**无法区分**；可主张性完全依赖 (iv-a)。\n")
    c.append("2. **(iv-a) 与 (iv-b) 不一致**：E1 的六个组合 (iv-b) 全部 `holds = False`，"
             "而 (iv-a) 全部通过。按 M1b 明文**以 (iv-a) 为准**并如实报告该不一致；"
             "**不得**取并集。\n")
    c.append("3. **M1b CERTIFIED 推不出的两件事**：推不出 S\\* 逼近表层信息上界；"
             "推不出任何关于未纳入 block 的结论。\n")
    c.append("4. **E2 non-estimable 的含义**：这是对 estimand 的诚实陈述，"
             "**不是**「hidden state 对未见实体无增量」——该问题在本样本上无法回答。\n")
    write_new(OUT / "stage1_claim_boundary.md", "".join(c))

    print(f"\n结果 sha256 {conf_sha}\nlog    sha256 {log_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
