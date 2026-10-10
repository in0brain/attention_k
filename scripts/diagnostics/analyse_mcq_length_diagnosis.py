r"""MCQ 长度诊断的分析（只读诊断产物，不动任何 gate 证据）。

逐题逐 trace 配对比较，**贪心与随机采样分开汇总**。人工文本判定与 tokenizer 定位
**分开记录**，不用重编码冒充原生成 token 序列。

用法：
    python scripts/diagnostics/analyse_mcq_length_diagnosis.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp    # noqa: E402
from recover_attention.generation import backends as bk    # noqa: E402

DIAG = ROOT / "outputs" / "stage1_llama_v3" / "mcq_length_diagnosis"
ROWS = DIAG / "mcq_length_diagnosis_rows.jsonl"
OUT = ROOT / "tmp" / "stage1_l2_g3" / "mcq_length_diagnosis_analysis.json"

LLAMA_EOS_IDS = {128001: "<|end_of_text|>", 128008: "<|eom_id|>", 128009: "<|eot_id|>"}
#: 人工审计的四类（卡 §5：只用于区分输出**形态**，不替换正式解析器）
CATS = ("最终答案", "选项枚举", "未完成分析", "其他输出")


def load() -> list[dict]:
    with io.open(ROWS, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def termination(r: dict) -> dict:
    """记录**事实**，不做唯一归因 —— EOS 与长度上限可能同时成立。"""
    hit_eos = bool(r.get("hit_passed_eos"))
    hit_cap = bool(r.get("reached_max_new_tokens"))
    others = [LLAMA_EOS_IDS[i] for i in (r.get("other_eos_ids_present_in_output") or [])]
    if r.get("error"):
        label = "异常"
    elif hit_eos and hit_cap:
        label = "EOS 与长度上限同时满足（不唯一归因）"
    elif hit_eos:
        label = "提前 EOS"
    elif hit_cap:
        label = "达到长度上限"
    else:
        label = "其他停止（未命中所传 EOS 也未达上限）"
    return {"label": label, "hit_passed_eos": hit_eos, "reached_cap": hit_cap,
            "special_eos_ids_seen": others,
            "last_token": r.get("last_generated_token_decoded")}


def human_category(text: str) -> str:
    """人工审计的四分类 —— **不是**解析器，不参与任何 gate 判定。"""
    if not text:
        return "其他输出"
    if dlp._EXPLICIT_ANSWER_RE.search(text):
        return "最终答案"
    t = text.rstrip()
    # 以选项标号/选项行结尾，且其后只剩标点空白 → 停在枚举里
    if re.search(r"(?:\n|^)\s*[-*•]?\s*\(?[A-D][).:]?\s*$", t) or \
       re.search(r"[-*•]\s*\(?[A-D]\b[^\n]{0,40}$", t):
        return "选项枚举"
    if t.endswith((".", "!", "?")):
        return "其他输出"
    return "未完成分析"


def first_answer_token_index(r: dict) -> dict:
    """答案首次完整出现于第几个生成 token 之后。

    **人工文本判定**：在 decoded_keep_special_tokens 上找 `Answer: X` 的字符位置。
    **tokenizer 定位**：用**原始生成 token ids** 逐个前缀解码求最小 j —— 这是本轮
    生成时真正产出的序列，**不是**重编码。
    """
    text = r.get("decoded_keep_special_tokens") or ""
    m = list(dlp._EXPLICIT_ANSWER_RE.finditer(text))
    if not m:
        return {"text_has_answer": False}
    end_char = m[-1].end()
    ids = r.get("generated_ids_raw") or []
    return {"text_has_answer": True,
            "text_char_end_of_answer": end_char,
            "text_total_chars": len(text),
            "token_index_note": ("下方 j 由**原始生成 token ids** 的前缀解码求得，"
                                 "不是对文本重新编码"),
            "_ids_len": len(ids), "_end_char": end_char}


def resolve_token_index(rows: list[dict], tok) -> None:
    """需要 tokenizer 才能把字符位置映射到生成 token 下标；单独一步，失败即标未知。"""
    for r in rows:
        fa = r["_first_answer"]
        if not fa.get("text_has_answer"):
            continue
        ids, end_char = r.get("generated_ids_raw") or [], fa["_end_char"]
        j = None
        for k in range(1, len(ids) + 1):
            if len(tok.decode(ids[:k], skip_special_tokens=False)) >= end_char:
                j = k
                break
        fa["n_generated_tokens_until_answer_complete"] = j
        fa.pop("_ids_len", None)
        fa.pop("_end_char", None)


def main() -> int:
    if not ROWS.exists():
        print(f"未找到诊断产物：{ROWS}")
        return 1
    rows = load()
    for r in rows:
        r["_term"] = termination(r)
        r["_cat"] = human_category(r.get("decoded_keep_special_tokens")
                                   or r.get("raw_completion") or "")
        r["_first_answer"] = first_answer_token_index(r)

    tok = None
    try:
        from transformers import AutoTokenizer            # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(
            "D:/models/Meta-Llama-3.1-8B-Instruct", local_files_only=True)
        resolve_token_index(rows, tok)
    except Exception as exc:                              # noqa: BLE001
        print("tokenizer 定位不可用：", str(exc)[:120])

    def group(cond, st):
        return [r for r in rows if r["condition"] == cond and r["sample_type"] == st]

    summary = {}
    for cond in ("len16", "len128"):
        for st in ("greedy", "sampled"):
            g = group(cond, st)
            if not g:
                continue
            fa = [r for r in g if r["_first_answer"].get("text_has_answer")]
            summary[f"{cond}/{st}"] = {
                "n": len(g),
                "termination": dict(Counter(r["_term"]["label"] for r in g)),
                "human_category": dict(Counter(r["_cat"] for r in g)),
                "parser_status": dict(Counter(r["parse_status"] for r in g)),
                "n_text_has_explicit_answer": len(fa),
                "tokens_until_answer_complete": sorted(
                    x for x in (r["_first_answer"].get(
                        "n_generated_tokens_until_answer_complete") for r in fa)
                    if x is not None),
                "n_generated_tokens": {
                    "min": min(r["n_generated_tokens"] for r in g),
                    "max": max(r["n_generated_tokens"] for r in g),
                    "mean": round(sum(r["n_generated_tokens"] for r in g) / len(g), 2)},
            }

    # ── 配对：同 unit_id 的两条件 ─────────────────────────────────────────
    by_unit = defaultdict(dict)
    for r in rows:
        by_unit[r["unit_id"]][r["condition"]] = r
    paired, prefix_same, prefix_diff, rescued = [], 0, 0, []
    for uid, d in by_unit.items():
        a, b = d.get("len16"), d.get("len128")
        if not (a and b):
            continue
        ia, ib = a.get("generated_ids_raw") or [], b.get("generated_ids_raw") or []
        same = ib[:len(ia)] == ia
        prefix_same += int(same)
        prefix_diff += int(not same)
        if not same:
            n_common = next((k for k in range(min(len(ia), len(ib)))
                             if ia[k] != ib[k]), min(len(ia), len(ib)))
            prefix_diff_detail = {"unit_id": uid, "first_divergence_token_index": n_common,
                                  "sample_type": a["sample_type"]}
        else:
            prefix_diff_detail = None
        if (not a["_first_answer"].get("text_has_answer")
                and b["_first_answer"].get("text_has_answer")):
            rescued.append(uid)
        paired.append({"unit_id": uid, "sample_type": a["sample_type"],
                       "len16": {"term": a["_term"]["label"], "cat": a["_cat"],
                                 "n_gen": a["n_generated_tokens"],
                                 "parser": a["parse_status"]},
                       "len128": {"term": b["_term"]["label"], "cat": b["_cat"],
                                  "n_gen": b["n_generated_tokens"],
                                  "parser": b["parse_status"]},
                       "prefix_identical": same,
                       "prefix_divergence": prefix_diff_detail})

    # ── 解析器判成功的条目是否只是枚举标号 ──────────────────────────────
    parser_ok = [r for r in rows if r["parse_status"] in
                 ("explicit_answer_marker", "last_isolated_label")]
    enum_fp = []
    for r in parser_ok:
        sp = r.get("parsed_span") or {}
        ctx = sp.get("context", "")
        c = r.get("raw_completion") or ""
        cs = sp.get("char_start")
        if cs is None:
            continue
        before, after = c[:cs], c[sp["char_end"]:]
        if (re.search(r"option", before, re.I)
                and re.search(r"(?:\n\s*|[-*•(]\s*|\*\*|Option\s)$", before)
                and after.strip(" .:;,)]}*\n\t") == ""):
            enum_fp.append({"unit_id": r["unit_id"], "condition": r["condition"],
                            "parsed_label": r["parsed_label"], "context": ctx})

    # ── 明确答案已出现但解析器未识别 ────────────────────────────────────
    missed = [{"unit_id": r["unit_id"], "condition": r["condition"],
               "parse_status": r["parse_status"],
               "tail": (r.get("decoded_keep_special_tokens") or "")[-80:]}
              for r in rows
              if r["_first_answer"].get("text_has_answer")
              and r["parse_status"] not in ("explicit_answer_marker",)]

    doc = {
        "schema": "stage1_mcq_length_diagnosis_analysis_v1",
        "date": "2026-09-14",
        "boundary": ("**探索性诊断的分析**。不替换 outputs/stage1_llama_v3/g3_postfix/ 的"
                     "原结果，不构成原协议 G3 通过。人工四分类**不是**解析器，"
                     "不参与任何 gate 判定；未按答案正确率选择设置。"),
        "rows_path": str(ROWS.relative_to(ROOT)).replace("\\", "/"),
        "n_rows": len(rows),
        "prompt_bos_count_distribution": dict(
            Counter(r.get("prompt_n_bos_128000") for r in rows)),
        "summary_by_condition_and_sample_type": summary,
        "paired": {
            "n_pairs": len(paired),
            "prefix_identical": prefix_same,
            "prefix_different": prefix_diff,
            "answer_only_at_len128": rescued,
            "n_answer_only_at_len128": len(rescued),
            "detail": paired,
        },
        "parser_success_rows": {
            "n": len(parser_ok),
            "n_enumeration_label_false_positive": len(enum_fp),
            "examples": enum_fp[:8],
        },
        "answer_present_but_parser_missed": {"n": len(missed), "examples": missed[:8]},
        "special_eos_ids_seen_anywhere": sorted(
            {i for r in rows for i in (r.get("other_eos_ids_present_in_output") or [])}),
        "human_audit_categories": list(CATS),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("prefix identical/different:", prefix_same, "/", prefix_diff)
    print("answer only at len128:", len(rescued))
    print("parser ok:", len(parser_ok), "| 其中枚举伪阳性:", len(enum_fp))
    print("answer present but parser missed:", len(missed))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
