r"""独立审阅 × 原解析器 的对照，以及**修正后**的答案位置定位。

三件事：
  ① 五类对照：明确作答且被正确识别 / 明确作答但未识别或识别错误 /
     尚未作答却被判成功 / 尚未作答且解析失败 / 无法确定 —— 列出全部分歧
  ② 答案位置：区分**首次**明确选择 / 后续改口 / **最后**明确选择；
     复核旧实现用 m[-1] 是否影响此前报告的 12 个位置
  ③ 修订后的汇总数字

定位基于**已保存的原始生成 token ids** 前缀解码，**不是**文本重编码。
可加载本地 tokenizer，**不加载模型权重**。

用法：
    python scripts/diagnostics/reconcile_mcq_length_diagnosis.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp   # noqa: E402

DIAG = ROOT / "outputs/stage1_llama_v3/mcq_length_diagnosis"
ROWS = DIAG / "mcq_length_diagnosis_rows.jsonl"
REVIEW = DIAG / "claude_text_review.jsonl"
OUT = ROOT / "tmp/stage1_l2_g3/mcq_length_diagnosis_analysis_revised.json"
PARSER_OK = ("explicit_answer_marker", "last_isolated_label")


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in io.open(p, encoding="utf-8") if x.strip()]


def token_index_for_char_end(ids: list[int], end_char: int, tok) -> int | None:
    """原始生成 token ids 的前缀解码到达 end_char 所需的 token 数。"""
    for k in range(1, len(ids) + 1):
        if len(tok.decode(ids[:k], skip_special_tokens=False)) >= end_char:
            return k
    return None


def main() -> int:
    rows = {(r["condition"], r["unit_id"]): r for r in load(ROWS)}
    review = load(REVIEW)
    assert len(rows) == len(review) == 240

    # ── 盲标完成**之后**才做的文本→标签映射 ─────────────────────────────
    # 只用 candidate_choices（选项清单），**不看正确答案**。两条审阅给出选项文本
    # 但未给字母的记录在此落位；这一步在盲标之后进行，不回头影响盲标本身。
    POST_HOC_MAP = {
        ("len128", "cybermetric_004295__sampled_5"):
            ("D", "选项 D = '4'；审阅读出 'minimum of 4 hosts'"),
        ("len128", "cybermetric_008950__sampled_3"):
            ("B", "选项 B = 'Spoofing'；审阅读出 'spoofing best describes'"),
    }

    tok = None
    try:
        from transformers import AutoTokenizer          # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(
            "D:/models/Meta-Llama-3.1-8B-Instruct", local_files_only=True)
    except Exception as exc:                            # noqa: BLE001
        print("tokenizer 不可用，位置列为未知：", str(exc)[:100])

    buckets = Counter()
    disagreements, positions = [], []
    for rv in review:
        r = rows[(rv["condition"], rv["unit_id"])]
        pstat, plabel = r["parse_status"], r["parsed_label"]
        parser_ok = pstat in PARSER_OK
        choice = rv["made_final_choice"]

        if choice == "无法确定":
            bucket = "无法确定"
        elif choice == "是":
            if not parser_ok:
                bucket = "明确作答但未识别"
            elif rv["chosen_label"] in (None, "未知"):
                mapped = POST_HOC_MAP.get((rv["condition"], rv["unit_id"]))
                if mapped is None:
                    bucket = "明确作答但标签未映射"
                else:
                    rv["post_hoc_mapped_label"] = mapped[0]
                    rv["post_hoc_mapping_basis"] = mapped[1]
                    bucket = ("明确作答且被正确识别" if plabel == mapped[0]
                              else "明确作答但识别错误")
            elif plabel == rv["chosen_label"]:
                bucket = "明确作答且被正确识别"
            else:
                bucket = "明确作答但识别错误"
        else:
            bucket = "尚未作答却被判成功" if parser_ok else "尚未作答且解析失败"
        buckets[bucket] += 1

        if bucket != "明确作答且被正确识别" and (parser_ok or choice in ("是", "无法确定")):
            disagreements.append({
                "condition": rv["condition"], "unit_id": rv["unit_id"],
                "bucket": bucket,
                "claude_choice": choice, "claude_label": rv["chosen_label"],
                "claude_option_text": rv["chosen_option_text"],
                "post_hoc_mapped_label": rv.get("post_hoc_mapped_label"),
                "post_hoc_mapping_basis": rv.get("post_hoc_mapping_basis"),
                "claude_evidence": rv["evidence_text"],
                "claude_reason": rv["reason"],
                "parser_status": pstat, "parser_label": plabel,
                "parser_span_context": (r.get("parsed_span") or {}).get("context"),
                "tail": (r["decoded_keep_special_tokens"] or "")[-90:],
            })

        # ── 位置：首次 / 改口 / 最后 ──────────────────────────────────────
        if choice != "是":
            continue
        text = r["decoded_keep_special_tokens"] or ""
        ids = r["generated_ids_raw"] or []
        marks = list(dlp._EXPLICIT_ANSWER_RE.finditer(text))
        ev_span = rv["evidence_char_span"]
        entry = {
            "condition": rv["condition"], "unit_id": rv["unit_id"],
            "claude_label": rv["chosen_label"],
            "n_explicit_answer_markers": len(marks),
            "has_self_correction": rv["flags"]["self_correction"],
            "evidence_is_explicit_marker": bool(marks),
        }
        if marks:
            first_end, last_end = marks[0].end(), marks[-1].end()
            entry.update({
                "first_marker_char_end": first_end,
                "last_marker_char_end": last_end,
                "first_equals_last": first_end == last_end,
                "old_impl_used": "m[-1]（最后一个匹配）",
            })
            if tok:
                entry["tokens_until_first_choice"] = token_index_for_char_end(
                    ids, first_end, tok)
                entry["tokens_until_last_choice"] = token_index_for_char_end(
                    ids, last_end, tok)
                entry["old_reported_value_equals_new_first"] = (
                    entry["tokens_until_first_choice"] == entry["tokens_until_last_choice"])
        else:
            # 无 Answer: 标记的最终选择 —— 用审阅证据片段定位
            entry.update({"evidence_char_span": ev_span,
                          "note": "无 Answer: 标记；位置基于审阅证据片段"})
            if tok and ev_span:
                entry["tokens_until_first_choice"] = token_index_for_char_end(
                    ids, ev_span[1], tok)
                entry["tokens_until_last_choice"] = entry["tokens_until_first_choice"]
                entry["old_impl_would_report"] = None
        positions.append(entry)

    with_marker = [p for p in positions if p["evidence_is_explicit_marker"]]
    changed = [p for p in with_marker if not p.get("first_equals_last", True)]

    def by(cond, st=None):
        return [rv for rv in review if rv["condition"] == cond
                and (st is None or rv["sample_type"] == st)]

    summary = {}
    for cond in ("len16", "len128"):
        for st in ("greedy", "sampled"):
            g = by(cond, st)
            summary[f"{cond}/{st}"] = {
                "n": len(g),
                "明确作答": sum(1 for r in g if r["made_final_choice"] == "是"),
                "未作答": sum(1 for r in g if r["made_final_choice"] == "否"),
                "无法确定": sum(1 for r in g if r["made_final_choice"] == "无法确定"),
                "output_category": dict(Counter(r["output_category"] for r in g)),
            }

    enum_fp = [d for d in disagreements
               if d["bucket"] == "尚未作答却被判成功"
               and d["parser_status"] == "last_isolated_label"]

    doc = {
        "schema": "stage1_mcq_length_diagnosis_revised_analysis_v1",
        "date": "2026-09-14",
        "supersedes": ("tmp/stage1_l2_g3/mcq_length_diagnosis_analysis.json —— "
                       "旧文件**保留不改**，其 human_category 与 '有无最终答案' 均由正则得出，"
                       "且 first_answer_token_index 用 m[-1] 定位的是**最后一个**匹配"),
        "inputs": {
            "rows": {"path": str(ROWS.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()},
            "review": {"path": str(REVIEW.relative_to(ROOT)).replace("\\", "/"),
                       "sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
                       "annotator": "Claude 文本审阅（**不是**独立人类标注）"},
        },
        "coverage": {"rows": len(rows), "review": len(review), "matched": len(review)},
        "revised_summary": summary,
        "reconciliation_buckets": dict(buckets),
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
        "enumeration_or_fragment_false_positives": {
            "n": len(enum_fp),
            "note": ("原解析器在审阅判为『尚未作答』的条目上仍报成功，"
                     "全部来自 last_isolated_label 回退。**这是本批的观察**，"
                     "不写成『必然产生伪阳性』"),
        },
        "answer_positions": {
            "n_final_choices": len(positions),
            "n_with_explicit_marker": len(with_marker),
            "n_without_marker": len(positions) - len(with_marker),
            "m_last_vs_m_first_changed": len(changed),
            "changed_detail": changed,
            "detail": positions,
            "method": ("位置由**已保存的原始生成 token ids** 前缀解码求得；"
                       "**不是**对文本重新编码"),
        },
        "post_hoc_label_mapping": {
            "when": "**盲标完成之后**",
            "basis": "只用 candidate_choices 选项清单，**不看正确答案**",
            "entries": {f"{k[0]}/{k[1]}": v for k, v in POST_HOC_MAP.items()},
        },
        "not_claimed": [
            "本轮修订只影响探索性长度诊断的分析与表述，不改动此前 Qwen/Llama 真实生成产物，"
            "不改变原 Llama G3 未通过的结论。它不构成采用新长度或新解析器的批准。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("buckets:", json.dumps(dict(buckets), ensure_ascii=False))
    print("分歧条数:", len(disagreements))
    print("最终选择:", len(positions), "| 带 Answer: 标记:", len(with_marker),
          "| 无标记:", len(positions) - len(with_marker))
    print("m[-1] 与 m[0] 定位不同的条数:", len(changed))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
