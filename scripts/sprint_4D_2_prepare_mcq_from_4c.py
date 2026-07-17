"""Stage B：把 4B-3/4C 的 MCQ 资产适配成 v2.3 的 completion-level 记录。

设计权威:docs/paper/preregistration.md v2.3（sha256 16fa43db…）§3 / §7.1 / §7.2。
依据:docs/paper/mcq_asset_audit_and_v2.3_rationale.md。

**只做确定性转换**:不重新采样、不重跑生成、不在此另造 parser。
canonicalization 的输入关系恒为:
    upstream parsed_label + visible candidate_choices → selected option text
不是: raw completion → Stage B 新 parser → parsed label。

population role（硬边界）:
  burned 240（4B-3/4C 用过）→ 允许 adapter 开发 / token 定位 / hidden re-forward /
      smoke CV+bootstrap 跑通;**禁止**作为 confirmatory 结果。
  fresh 1760 → 在 G1/G2/G3 全绿前,完全不进入本项目的 prompt 开发、生成、特征工程、
      模型拟合或评估。（注意措辞:不能说"模型没见过",预训练污染无法排除。）

输出:
  mcq_pilot_completion_records.jsonl   ≤20 条 pilot（population_role=pilot_smoke）
  mcq_burned_ids.json                  240
  mcq_fresh_confirmatory_ids.json      1760
  mcq_v2_3_smoke_manifest.jsonl        pilot 选取清单 + 覆盖理由
  encoding_manifest.json               编码决定与理由（如 one-hot 的等价性说明）
  preflight_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention import mcq_conditional_increment as mci  # noqa: E402
from recover_attention.data_io import read_jsonl, write_json, write_jsonl, write_text  # noqa: E402

SMOKE_MAX = 20
# 特征构造函数绝不可见的字段(只能留在 evaluation record 里)
EVAL_ONLY_FIELDS = ("gold_label", "wrong_label", "is_correct")


def _canonical_by_id(pool_path: Path) -> dict[str, dict]:
    return {r["example_id"]: r for r in read_jsonl(pool_path)}


def _greedy_traces(trace_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in read_jsonl(trace_path):
        if r.get("sample_type") == "greedy":
            out[r["example_id"]] = r
    return out


def _sampled_by_id(trace_path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in read_jsonl(trace_path):
        if r.get("sample_type") != "greedy":
            out.setdefault(r["example_id"], []).append(r)
    return out


def build_record(example_id: str, canon: dict, greedy: dict, sampled: list[dict],
                 source_artifact: str) -> dict:
    """一题一条 completion-level 记录（§3:population 恒为每题一条 greedy）。"""
    choices = canon["candidate_choices"]
    parsed = greedy.get("parsed_label")          # **上游**解析结果,不重新解析
    canon_out = mci.canonicalize_mcq_output(parsed, choices)
    fmt = mci.response_format_of(greedy.get("completion"), parsed)

    # F5 的 self-consistency / letter-agreement 跨该题的 sampled traces（§7.1）
    sampled_labels = [s.get("parsed_label") for s in sampled
                      if isinstance(s.get("parsed_label"), str)]
    cons = _exact_consistency(sampled_labels)
    agree = (sum(l == parsed for l in sampled_labels) / len(sampled_labels)
             if sampled_labels and isinstance(parsed, str) else None)

    row: dict[str, Any] = {
        "schema_version": "4D2_mcq_v2.3_completion_record_v1",
        "example_id": example_id,
        "group_id": canon["group_id"],
        "rendered_prompt": greedy["prompt"],
        "raw_completion": greedy.get("completion"),
        "parsed_label": parsed,
        "parse_status": ("upstream_parse_failure" if greedy.get("parse_failure")
                         else greedy.get("parse_method")),
        "candidate_choices": choices,
        "prompt_only_text": mci.prompt_only_text(canon["question"], choices),
        **canon_out,
        **fmt,
        **mci.mcq_surface_features(canon_out["selected_option_text"], choices, parsed),
        "f5_self_consistency_exact": cons,
        "f5_letter_agreement_rate": agree,
        "n_sampled_traces": len(sampled_labels),
        "population_role": "pilot_smoke",
        "source_artifact": source_artifact,
        "prompt_style": greedy.get("prompt_style"),
        # --- evaluation-only：以下字段是 y,不得进入任何特征构造 ---
        "gold_label": canon.get("gold_label"),
        "wrong_label": (None if greedy.get("parse_failure")
                        else int(greedy.get("parsed_label") != canon.get("gold_label"))),
    }
    mci.assert_semantic_output_not_from_gold(row)
    return row


def _exact_consistency(labels: list[str]) -> float | None:
    if not labels:
        return None
    c = Counter(labels)
    return c.most_common(1)[0][1] / len(labels)


def select_smoke_manifest(records: list[dict], limit: int = SMOKE_MAX) -> tuple[list[dict], dict]:
    """覆盖导向的确定性选取（**label-aware**，仅供工程 smoke）。

    刻意覆盖:正确/错误 completion、A/B/C/D 四种字母、裸字母与已知格式异常、
    不同长度的选项文本、至少一条预期 parse failure。
    这**不是**随机样本,任何由此得到的指标只是工程诊断,不得用于估计性能。
    """
    picked: list[dict] = []
    seen: set[str] = set()

    def take(rows: list[dict], n: int, why: str) -> None:
        for r in sorted(rows, key=lambda x: x["example_id"]):
            if len(picked) >= limit or r["example_id"] in seen:
                continue
            if n <= 0:
                break
            seen.add(r["example_id"])
            picked.append({**r, "_selection_reason": why})
            n -= 1

    ok = [r for r in records if r["canonicalization_status"] == mci.CANON_OK]
    # 1) 必含:上游 parse failure（预期 ineligible 路径）
    take([r for r in records if r["canonicalization_status"] != mci.CANON_OK], 2, "parse_failure_path")
    # 2) 必含:格式异常（非裸字母）
    take([r for r in ok if not r["bare_answer"]], 2, "wrapped_label_format")
    # 3) 四种字母 × 正确/错误 各取
    for letter in mci.MCQ_LETTERS:
        for wrong in (0, 1):
            take([r for r in ok if r["parsed_label"] == letter and r["wrong_label"] == wrong],
                 1, f"letter_{letter}_wrong_{wrong}")
    # 4) 选项文本长度两端
    by_len = sorted(ok, key=lambda r: r["surface_option_chars"])
    take(by_len[:2], 2, "shortest_option_text")
    take(by_len[-2:], 2, "longest_option_text")
    # 5) 补齐到 limit（确定性:按 example_id）
    take(ok, limit - len(picked), "fill_to_limit")

    cov = {
        "n_selected": len(picked),
        "letters": dict(Counter(r["parsed_label"] for r in picked)),
        "wrong_label": dict(Counter(str(r["wrong_label"]) for r in picked)),
        "canonicalization_status": dict(Counter(r["canonicalization_status"] for r in picked)),
        "response_format": dict(Counter(r["response_format"] for r in picked)),
        "selection_reasons": dict(Counter(r["_selection_reason"] for r in picked)),
        "note": ("selection is coverage-oriented and label-aware; "
                 "all resulting metrics are engineering diagnostics only."),
    }
    return picked, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/processed/cyber/cybermetric.jsonl")
    ap.add_argument("--traces",
                    default="outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/"
                            "trace_sampling_manifest.jsonl")
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--output-dir", default="outputs/logs/sprint_4D_2_mcq_v2_3")
    ap.add_argument("--smoke-limit", type=int, default=SMOKE_MAX)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # G2:适配器也受冻结校验约束——协议变了就不该用旧适配产物
    g2 = ci.check_preregistration_frozen(args.prereg, args.lock)
    if not g2["match"]:
        raise SystemExit(f"STOP (G2): preregistration hash mismatch\n  current={g2['current']}\n"
                         f"  locked ={g2['locked']}")

    pool = _canonical_by_id(Path(args.pool))
    greedy = _greedy_traces(Path(args.traces))
    sampled = _sampled_by_id(Path(args.traces))

    # fresh / burned：按 **ID 集合** 校验，不靠行数
    split = mci.split_fresh_confirmatory(sorted(pool), sorted(greedy))
    write_json({"n": split["n_burned"], "ids": split["burned_ids"],
                "role": "exploratory_burned",
                "why": ("used by 4B-3/4C; their CI motivated the v2.3 amendment, so reusing them "
                        "as confirmatory would be deciding the design and the conclusion on the "
                        "same data")},
               out / "mcq_burned_ids.json")
    write_json({"n": split["n_fresh"], "ids": split["fresh_ids"],
                "role": "confirmatory_fresh",
                "status": ("NOT used in this project for prompt development, generation, feature "
                           "engineering, model fitting, or evaluation. Enters the experiment only "
                           "after G1+G2+G3 are all green. (Pretraining contamination cannot be "
                           "excluded and is not claimed otherwise.)")},
               out / "mcq_fresh_confirmatory_ids.json")

    records = [build_record(eid, pool[eid], greedy[eid], sampled.get(eid, []),
                            source_artifact=str(args.traces))
               for eid in sorted(greedy)]
    picked, cov = select_smoke_manifest(records, args.smoke_limit)
    write_jsonl(picked, out / "mcq_pilot_completion_records.jsonl")
    write_jsonl([{k: r[k] for k in ("example_id", "group_id", "parsed_label",
                                    "canonicalization_status", "response_format",
                                    "wrong_label", "_selection_reason")} for r in picked],
                out / "mcq_v2_3_smoke_manifest.jsonl")

    write_json({
        "rung2_answer_letter_only": {
            "encoding": "fixed 4-dimensional one-hot",
            "fixed_category_order": list(mci.MCQ_LETTERS),
            "dimensionality": len(mci.MCQ_LETTERS),
            "rationale": (
                "answer-letter-only 的对象是固定类别集合 {A,B,C,D},统计上是 categorical "
                "feature,不是自然语言文本。4 维 one-hot 是该信息的无损、最直接表示。"
                "§7 冻结的 word/char TF-IDF 是 'full-response-text 实现' 标题下的规定,"
                "不要求 ladder 每个 rung 都过 TF-IDF。且冻结的 word tokenizer 要求 >=2 字符,"
                "单字母产生 0 token -> 词表为空直接抛错。"),
            "invariants": (
                "类别顺序恒为 [A,B,C,D];不按 fold 内观测类别动态排序;"
                "某 fold 缺某字母也不改变维数(该列恒 0)。"),
            "amendment_needed": False,
            "why_no_amendment": (
                "消除实现歧义,未改变:输入信息 / 假设空间实质 / 样本 / 评价协议 / "
                "预注册比较关系。preregistration hash 不变。"),
        },
        "rung3_option_surface_only": {
            "features": list(mci.MCQ_SURFACE_FEATURE_NAMES),
            "excludes": "选项词汇语义(否则 rung3 与 rung4 重合)",
        },
        "rung4_canonical_response_text": {
            "encoding": "word + char_wb TF-IDF over selected-option text (§7 frozen params)",
        },
        "f5_mcq": {"features": list(mci.MCQ_F5_FEATURE_NAMES),
                   "d_F": len(mci.MCQ_F5_FEATURE_NAMES),
                   "note": "d_F 按任务实际维数,不沿用 H1 的 14"},
        "parser_contract": {
            "input": "upstream parsed_label + visible candidate_choices",
            "not": "raw completion -> Stage B parser -> parsed label",
            "parse_failure_only_when": [
                "upstream marked parse failure (parsed_label null)",
                "parsed_label not in {A,B,C,D}",
                "parsed_label has no matching candidate_choice",
            ],
            "format_invariance": ("bare 'D' and wrapped 'Answer! <D>' must map to the same "
                                  "option text when upstream parses both to 'D'; format is "
                                  "recorded via response_format and controlled by the "
                                  "response-format/option-surface shortcut, never turned into "
                                  "no-emission"),
        },
    }, out / "encoding_manifest.json")

    status = Counter(r["canonicalization_status"] for r in records)
    fmt = Counter(r["response_format"] for r in records)
    lines = [
        "# Sprint 4D-2 MCQ Stage B Preflight (v2.3)",
        "",
        f"- preregistration: v2.3 sha256 `{g2['current']}` (lock match: {g2['match']})",
        f"- pool: `{args.pool}` -> {split['n_pool']} unique ids",
        f"- burned (4B-3/4C, exploratory): {split['n_burned']}",
        f"- fresh confirmatory: {split['n_fresh']}",
        f"- id-set checks: intersection={split['intersection_fresh_burned']}, "
        f"union==pool: {split['union_equals_pool']}",
        f"- adapted records (burned only): {len(records)}",
        f"- canonicalization_status: {dict(status)}",
        f"- response_format: {dict(fmt)}",
        f"- pilot smoke selected: {cov['n_selected']} (limit {args.smoke_limit})",
        f"- coverage: {json.dumps(cov, ensure_ascii=False)}",
        "",
        "## Boundaries",
        "",
        "- Deterministic transform only: no resampling, no regeneration, no new parser.",
        "- Canonicalization consumes the upstream `parsed_label`; it never sees `raw_completion`,",
        "  `gold_label`, `wrong_label`, or `is_correct`.",
        "- All records here are `population_role=pilot_smoke` from the burned 240. They are",
        "  engineering diagnostics only and must not be reported as confirmatory results.",
        "- The fresh 1760 have not been used in this project for prompt development, generation,",
        "  feature engineering, model fitting, or evaluation. (Pretraining contamination cannot",
        "  be excluded; no such claim is made.)",
        "- rung2 one-hot is an equivalent encoding, not a design change; preregistration hash",
        "  unchanged. See encoding_manifest.json.",
    ]
    write_text("\n".join(lines) + "\n", out / "preflight_report.md")

    print(f"[stage-b] pool={split['n_pool']} burned={split['n_burned']} fresh={split['n_fresh']} "
          f"(intersection={split['intersection_fresh_burned']}, union_ok={split['union_equals_pool']})")
    print(f"[stage-b] adapted {len(records)} burned records; status={dict(status)} format={dict(fmt)}")
    print(f"[stage-b] pilot smoke: {cov['n_selected']} records -> {out}")


if __name__ == "__main__":
    main()
