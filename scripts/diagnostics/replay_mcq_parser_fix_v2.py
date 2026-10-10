r"""MCQ 答案提取规则的**三方离线重放**：冻结旧规则 / 第一版修复 / 当前规则。

只读已有落盘输出，**不加载模型权重、不重新生成、不改写任何既有产物**。
覆盖 Qwen 侧（G3 / G4 / 生产**原始**记录 10,520 条）与 Llama 侧全部 MCQ 输出。

逐条比较：`parsed_label` / `parse_method` / 答案字符区间 / `eligible_for_primary`
以及派生字段（`canonicalization_status` / `selected_option_text` /
`response_format` / `bare_answer`）。

⚠️ **不以"变化数越少"作为正确性标准。** 本脚本只如实列差异；
口径名变化（例如裸字母回答由 `last_isolated_label` 改称 `bare_letter`）
与**实质变化**（标签 / 区间 / eligibility）**分开计数**，不混为一谈。

用法：
    python scripts/diagnostics/replay_mcq_parser_fix_v2.py \
        --v0 <冻结旧规则的 domain_label_proxy.py> \
        --v1 <第一版修复的 domain_label_proxy.py>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp          # noqa: E402
from recover_attention import mcq_conditional_increment as mci   # noqa: E402

OUT = ROOT / "tmp/stage1_l2_g3/mcq_parser_replay_v2.json"
LETTERS = list(mci.MCQ_LETTERS)

FILES = [
    ("qwen", "G3", "outputs/stage1/g3/mcq_g3_unit_results.jsonl"),
    ("qwen", "G4", "outputs/stage1/g4/mcq_g4_unit_results.jsonl"),
    ("qwen", "生产(原始)", "outputs/stage1/production/mcq_raw_unit_results.jsonl"),
    ("llama", "G3(修复前)", "outputs/stage1_llama_v3/g3/mcq_g3_unit_results.jsonl"),
    ("llama", "G3(16,修复后)", "outputs/stage1_llama_v3/g3_postfix/mcq_g3_unit_results.jsonl"),
    ("llama", "G3(512)", "outputs/stage1_llama_v3/g3_budget512/mcq_g3_unit_results.jsonl"),
    ("llama", "G4(512)", "outputs/stage1_llama_v3/g4_budget512/mcq_g4_unit_results.jsonl"),
    ("llama", "生成长度检查",
     "outputs/stage1_llama_v3/mcq_budget512_check/mcq_budget512_unit_results.jsonl"),
    ("llama", "长度诊断",
     "outputs/stage1_llama_v3/mcq_length_diagnosis/mcq_length_diagnosis_rows.jsonl"),
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _span_of(mod, completion: str, parsed: dict):
    """该版本给出的答案字符区间。v0 不返回区间，按它当时的定位规则重算。"""
    span = parsed.get("answer_char_span")
    if span is not None or parsed["parsed_label"] is None:
        return span
    _, valid = mod._valid_label_set(LETTERS)
    explicit = [m for m in mod._EXPLICIT_ANSWER_RE.finditer(completion)
                if m.group(1).upper() in valid]
    if explicit:
        return list(explicit[-1].span(1))
    if hasattr(mod, "_trailing_isolated_matches"):
        trailing = mod._trailing_isolated_matches(completion, valid)
        if trailing:
            return list(trailing[-1].span(1))
    return None


def _derived(label, row):
    choices = row.get("candidate_choices")
    if not choices:
        return {}
    canon = mci.canonicalize_mcq_output(label, choices)
    fmt = mci.response_format_of(row.get("raw_completion") or "", label)
    return {"eligible_for_primary": canon.get("eligible_for_primary"),
            "canonicalization_status": canon.get("canonicalization_status"),
            "selected_option_text": canon.get("selected_option_text"),
            "response_format": fmt.get("response_format"),
            "bare_answer": fmt.get("bare_answer")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0", type=Path, required=True)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-examples", type=int, default=60)
    args = ap.parse_args()

    v0 = _load(args.v0, "_dlp_v0")
    v1 = _load(args.v1, "_dlp_v1")
    versions = {
        "v0_frozen": {"path": str(args.v0),
                      "sha256": hashlib.sha256(args.v0.read_bytes()).hexdigest()},
        "v1_first_fix": {"path": str(args.v1),
                         "sha256": hashlib.sha256(args.v1.read_bytes()).hexdigest()},
        "v2_current": {"path": "src/recover_attention/domain_label_proxy.py",
                       "sha256": hashlib.sha256(
                           (ROOT / "src/recover_attention/domain_label_proxy.py")
                           .read_bytes()).hexdigest()},
    }
    for k, v in versions.items():
        print(f"[{k:12s}] sha={v['sha256'][:16]}…")

    per_file, diffs = [], []
    totals = Counter()
    span_violations = []

    for side, tag, rel in FILES:
        path = ROOT / rel
        n = 0
        c = Counter()
        transitions = Counter()
        for line in io.open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            completion = row.get("raw_completion")
            if not isinstance(completion, str):
                continue
            n += 1
            p0 = v0.parse_option_answer(completion, LETTERS)
            p1 = v1.parse_option_answer(completion, LETTERS)
            p2 = dlp.parse_option_answer(completion, LETTERS)
            s0 = _span_of(v0, completion, p0)
            s1 = _span_of(v1, completion, p1)
            s2 = p2["answer_char_span"]

            if p2["parsed_label"] is not None:
                if not s2 or completion[s2[0]:s2[1]].upper() != p2["parsed_label"]:
                    span_violations.append({"file": rel, "unit_id": row.get("unit_id"),
                                            "label": p2["parsed_label"], "span": s2})

            d0, d2 = _derived(p0["parsed_label"], row), _derived(p2["parsed_label"], row)
            label_changed = p0["parsed_label"] != p2["parsed_label"]
            span_changed = s0 != s2
            method_changed = p0["parse_method"] != p2["parse_method"]
            elig_changed = d0.get("eligible_for_primary") != d2.get("eligible_for_primary")
            derived_changed = any(d0.get(k) != d2.get(k) for k in
                                  ("canonicalization_status", "selected_option_text",
                                   "response_format", "bare_answer"))
            substantive = label_changed or span_changed or elig_changed or derived_changed

            c["label_changed_vs_v0"] += label_changed
            c["span_changed_vs_v0"] += span_changed
            c["method_changed_vs_v0"] += method_changed
            c["eligibility_changed_vs_v0"] += elig_changed
            c["derived_changed_vs_v0"] += derived_changed
            c["substantive_changed_vs_v0"] += substantive
            c["method_name_only_vs_v0"] += method_changed and not substantive
            c["label_changed_vs_v1"] += p1["parsed_label"] != p2["parsed_label"]
            c["span_changed_vs_v1"] += s1 != s2

            if method_changed or substantive:
                transitions[f'{p0["parse_method"]}→{p2["parse_method"]}'] += 1
            if not substantive and p1["parsed_label"] == p2["parsed_label"] and s1 == s2:
                continue
            diffs.append({
                "side": side, "file": rel, "unit_id": row.get("unit_id"),
                "example_id": row.get("example_id"),
                "v0": {"label": p0["parsed_label"], "method": p0["parse_method"],
                       "span": s0, **d0},
                "v1": {"label": p1["parsed_label"], "method": p1["parse_method"],
                       "span": s1},
                "v2": {"label": p2["parsed_label"], "method": p2["parse_method"],
                       "span": s2, "reason": p2.get("parse_failure_reason"), **d2},
                "substantive_vs_v0": substantive,
                "tail": completion.strip()[-170:],
            })
        totals[side] += n
        per_file.append({"side": side, "tag": tag, "path": rel, "n_rows": n,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         **{k: int(v) for k, v in c.items()},
                         "transitions_vs_v0": dict(transitions)})
        print(f"  {side:5s} {tag:14s} n={n:6d} "
              f"实质变化(vs v0)={c['substantive_changed_vs_v0']:4d} "
              f"仅口径名={c['method_name_only_vs_v0']:5d} "
              f"｜ vs v1 标签变={c['label_changed_vs_v1']:3d} 区间变={c['span_changed_vs_v1']:3d}")

    qwen = [f for f in per_file if f["side"] == "qwen"]
    q_sub = sum(f["substantive_changed_vs_v0"] for f in qwen)
    q_name = sum(f["method_name_only_vs_v0"] for f in qwen)

    doc = {
        "schema": "stage1_mcq_parser_replay_v2",
        "date": "2026-09-15",
        "method": ("三方重放：冻结旧规则 v0 / 第一版修复 v1 / 当前规则 v2。"
                   "只读落盘 completion，**未加载模型权重、未重新生成、"
                   "未改写任何既有产物**。"),
        "not_a_criterion": ("**不以『变化数越少』作为正确性标准。** 正确性由定向正反例"
                            "与逐条审阅决定；这里只如实列差异。"),
        "versions": versions,
        "compared_fields": ["parsed_label", "parse_method", "answer_char_span",
                            "eligible_for_primary", "canonicalization_status",
                            "selected_option_text", "response_format", "bare_answer"],
        "scope": per_file,
        "n_rows_total": sum(totals.values()),
        "n_rows_by_side": dict(totals),
        "qwen": {
            "files": [f["path"] for f in qwen],
            "n_rows": totals["qwen"],
            "n_substantive_changed": q_sub,
            "n_method_name_only": q_name,
            "statement": (
                f"Qwen 侧 {totals['qwen']} 条：**标签 / 字符区间 / eligibility / "
                f"派生字段全部未变**；仅 {q_name} 条的 `parse_method` 名称由 "
                f"`last_isolated_label` 改称 `bare_letter`（同一条规则改了名字，"
                f"判定结果不变）。不改写旧证据，不重跑 Qwen。"
                if q_sub == 0 else
                f"**Qwen 侧出现 {q_sub} 条实质变化** —— 见 diffs，须逐条审阅；"
                f"不得再写『不影响旧 Qwen 结论』，本轮不改写旧证据、不重跑 Qwen。")},
        "span_self_consistency": {"n_violations": len(span_violations),
                                  "detail": span_violations[:10]},
        "n_diffs_total": len(diffs),
        "diffs": diffs[:args.max_examples],
        "all_diff_unit_ids": [d["unit_id"] for d in diffs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"\n[总计] {doc['n_rows_total']} 条（qwen {totals['qwen']} / llama {totals['llama']}）")
    print(f"[Qwen] {doc['qwen']['statement']}")
    print(f"[自洽] 区间与标签不一致 {len(span_violations)} 条")
    print(f"[差异] 列出 {len(diffs)} 条（含与 v1 的差异）")
    print(f"[out  ] {args.out}")
    return 0 if not span_violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
