r"""P2 准备：为 **Qwen / H1** 生成可执行材料（**不前向、不加载权重**）。

产出六样东西，全部是"提取之前就该定死"的：

    1 陈述模板        3 个（attack technique / attack subtechnique / cwe weakness）
    2 陈述清单        U1 全量：每个**不同**主标识符一条，带 trace / group 映射
    3 划分映射        外层折**物化**成 trace_id -> fold（逐 repeat），
                      并与主分析 roster 的 fold_of_sha **逐位比对**
    4 小批名单        32 条，用于计时微基准（**只为计时，不产出任何估计**）
    5 层与词元设置    层号两套编号 / final_norm / BOS-EOS / padding / 读取位置
    6 运行命令        提取脚本的调用形式（**本脚本不调用它**）

⚠️ 防泄漏的三条，写死在代码里：
    · 模板只按 `granularity` 选，而 `granularity` 由**表面形式**决定
      （`h1_identifier.py:126` 是 `"." in normalized`），**不查本体** → 不泄漏标签
    · `ontology_status` / `label` / `parent` **一律不进模板文本**
    · 小批名单按**文本 sha256 排序**取，与标签无关
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from recover_attention import conditional_increment as ci                 # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig       # noqa: E402

PRIMARY_FAMILIES = ci.PRIMARY_FAMILIES
OUT_DIR = ROOT / "outputs/stage1/posthoc_readout_probes/p2_prep"

#: 模型配置（**一次只准备一个模型**；本脚本固定 Qwen）
MODEL = {
    "model_key": "qwen2.5-7b-instruct",
    "records": "outputs/stage1/production/h1_completion_records.jsonl",
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "block_index_zero_based": 19,          # 冻结读出层（0 基 transformer 块）
    "hidden_states_tuple_index": 20,       # = block + 1，见 resolve_hidden_index
}

# ══════════════════════════════════════════════════════════════════════════
# 1 陈述模板
#
# 目标陈述必须满足：**它的真假恰好等于该 mention 的 label**。
#   label == "grounded"    → 该编号在本体里存在 → 陈述为真
#   label == "fabricated"  → 不存在             → 陈述为假
# 因此 y=1（fabricated）对应"陈述为假"。方向在产物里显式登记，不靠约定。
#
# ⚠️ 模板里**只**允许出现 `normalized`（规范化编号）。
#    不放技术名称（虚构编号根本没有名称，放了就是按标签区别对待）、
#    不放 ontology_status、不放 parent、不放模型原文上下文。
# ══════════════════════════════════════════════════════════════════════════
#
# ⚠️ **不能用 "is a valid ... identifier"。** 本项目的 `grounded` 判据是
#    `OntologyIndex.has()` —— 只看**目录里有没有这条记录**，与状态无关。
#    实测 grounded 里就包含 attack `revoked` 870 条、`deprecated` 32 条、
#    cwe `Deprecated` 118 条。把这些叫 "valid" 与标签语义不符。
#    改成"目录中有记录"。
TEMPLATES = {
    "attack_technique": {
        "granularity": "technique", "family": "attack",
        "text": "The MITRE ATT&CK catalog contains a record for technique ID {id}.",
    },
    "attack_subtechnique": {
        "granularity": "subtechnique", "family": "attack",
        "text": "The MITRE ATT&CK catalog contains a record for sub-technique ID {id}.",
    },
    "cwe_weakness": {
        "granularity": "weakness", "family": "cwe",
        "text": "The CWE catalog contains a record for weakness ID {id}.",
    },
}
TEMPLATE_BY_KEY = {(v["family"], v["granularity"]): k for k, v in TEMPLATES.items()}

STATEMENT_TRUTH = {
    "grounded": True,      # 目录中有记录（含 revoked / deprecated）→ 陈述为真 → y = 0
    "fabricated": False,   # 目录中没有记录                        → 陈述为假 → y = 1
}

#: 分词器口径（**已实测确定，不再悬着**）
TOKENIZER_SETTINGS = {
    "add_special_tokens": True,
    "why": ("采用作者默认口径。本地 Qwen2.5 tokenizer 的 `bos_token_id` 为 None，"
            "实测 870 个唯一文本在 True/False 两种设置下 input_ids **完全相同**"
            "（差异 0/870），故取 True 不引入任何额外词元。"),
    "batch_size": 1,
    "padding": False,
    "chat_template": False,
    "chat_template_why": "陈述是独立的判真假输入，不是对话轮次；不套模板",
    "verified_by": "scripts/p2_prepare_h1_statements.py --verify-tokenizer",
}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_statements(records: list[dict]) -> tuple[list[dict], dict]:
    """U1：每个 trace 的每个**不同**主标识符一条陈述。

    ⚠️ 两类异常（未知模板 / 同一编号标签冲突）**一律报错停下**，不计数继续。
       上一版只是 `+= 1; continue` —— 真实材料上这两项恰好都是 0，
       所以从产物看不出来；但那是"碰巧没触发"，不是"处理正确"。
    """
    rows = []
    skipped = {"unknown_template": 0, "label_conflict": 0,
               "policy": "两类异常均 fail-closed（raise），此处恒为 0；非零即代码有误"}
    for rec in records:
        if not bool(rec.get("eligible", rec.get("eligible_for_primary"))):
            continue
        tid, gid = str(rec["trace_id"]), str(rec["group_id"])
        uniq: dict[tuple[str, str], dict] = {}
        for m in rec.get("mentions", []):
            if m.get("family") not in PRIMARY_FAMILIES:
                continue
            if m.get("label") not in ("grounded", "fabricated"):
                continue          # echoed 等一律不进（与主分析过滤一致）
            k = (m["family"], m["normalized"])
            if k in uniq and uniq[k]["label"] != m["label"]:
                # **不静默跳过**：同一编号在同一 trace 内出现相反标签，
                # 说明标签不是 (family, normalized, prompt) 的确定函数，
                # 前提被破坏 —— 必须停下来查，不能计个数继续。
                raise RuntimeError(
                    f"{tid}: 编号 {k} 在同一 trace 内标签冲突"
                    f"（{uniq[k]['label']} vs {m['label']}）—— 停，不落盘。"
                    f"这是**前提被破坏**，不是可以计数忽略的边缘情况。")
            uniq.setdefault(k, m)
        for (fam, norm), m in sorted(uniq.items()):
            tkey = TEMPLATE_BY_KEY.get((fam, m.get("granularity")))
            if tkey is None:
                # 同理：没有模板就是**构造不出陈述**。静默丢掉会让样本流转对不上。
                raise RuntimeError(
                    f"{tid}: (family={fam!r}, granularity={m.get('granularity')!r}) "
                    f"没有对应模板，编号 {norm} —— 停，不落盘。"
                    f"要么补模板并登记，要么把该类显式排除并计入样本流转，"
                    f"**不能静默跳过**。")
            text = TEMPLATES[tkey]["text"].format(id=norm)
            rows.append({
                "statement_id": f"{tid}::{fam}::{norm}",
                "trace_id": tid, "group_id": gid,
                "family": fam, "normalized": norm,
                "raw_surface": m.get("raw"),          # 仅审计，**不进文本**
                "granularity": m.get("granularity"),
                "template": tkey, "text": text, "text_sha256": _sha(text),
                "mention_label": m["label"],
                "statement_is_true": STATEMENT_TRUTH[m["label"]],
                "y": int(m["label"] == "fabricated"),
            })
    return rows, skipped


def materialise_folds(records: list[dict], cfg: V3AnalysisConfig) -> dict:
    """把冻结的外层划分**物化**成 trace_id -> fold（逐 repeat）。

    ⚠️ 只留 `fold_seed(r)` 不够 —— 种子要在样本顺序、分组键、
       `stratified_grouped_folds` 实现三者一字不差时才复现同一划分。
       这里生成并落盘映射，同时算 `fold_of_sha` 供与主分析逐位比对。
    """
    used = [r for r in records
            if bool(r.get("eligible", r.get("eligible_for_primary")))]
    ids = np.asarray([str(r["trace_id"]) for r in used])
    y = np.asarray([int(r["label"]) for r in used], dtype=np.int64)
    groups = np.asarray([str(r["group_id"]) for r in used])
    out = {"n": int(y.size), "n_positive": int(y.sum()),
           "n_groups": int(len(set(groups.tolist()))), "repeats": {}}
    for r in range(cfg.n_repeats):
        folds = ci.stratified_grouped_folds(
            y, groups, n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r),
            max_seed_retries=cfg.outer_max_seed_retries)
        fo = np.full(y.size, -1, dtype=np.int64)
        for f, (_, te) in enumerate(folds):
            fo[te] = f
        if np.any(fo < 0):
            raise RuntimeError(f"repeat{r}: 有样本未分到折 —— 停")
        h = hashlib.sha256(np.ascontiguousarray(fo).tobytes()).hexdigest()[:16]
        out["repeats"][f"repeat{r}"] = {
            "fold_seed": cfg.fold_seed(r),
            "fold_of_sha": h,
            "test_sizes": [int(np.size(te)) for _, te in folds],
            "test_positives": [int(y[te].sum()) for _, te in folds],
            "fold_of": {t: int(v) for t, v in zip(ids.tolist(), fo.tolist())},
        }
    out["ids_sha"] = hashlib.sha256(
        "\x1f".join(ids.tolist()).encode()).hexdigest()[:16]
    return out


def duplicate_audit(rows: list[dict], cfg: V3AnalysisConfig) -> dict:
    """跨折文本重复审计 —— **这是新方案最重要的限制**。

    去掉上下文之后，"同一个编号"在不同 trace 上塌成**完全相同的输入**。
    提示级分组是对的（同一提示的 6 条回答不跨折），但**不同提示**生成同一个编号时，
    该文本仍会同时出现在训练折和测试折。探针因此有机会记住"编号 → 标签"的对应。

    这**不证明**它实际依赖记忆，但结果不能直接解释为"识别未见过的编号"的能力。
    """
    texts = collections.Counter(x["text_sha256"] for x in rows)
    uniq = {}
    for x in rows:
        uniq.setdefault(x["text_sha256"], x)
    lab = collections.defaultdict(set)
    for x in rows:
        lab[x["text_sha256"]].add(x["y"])
    conflict = [k for k, v in lab.items() if len(v) > 1]

    per_repeat = {}
    for r in range(cfg.n_repeats):
        seen_n = seen_p = uns_n = uns_p = 0
        per_fold = []
        for f in range(cfg.n_outer_folds):
            tr = {x["text_sha256"] for x in rows if x[f"fold_r{r}"] != f}
            te = [x for x in rows if x[f"fold_r{r}"] == f]
            s = [x for x in te if x["text_sha256"] in tr]
            u = [x for x in te if x["text_sha256"] not in tr]
            seen_n += len(s); seen_p += sum(x["y"] for x in s)
            uns_n += len(u); uns_p += sum(x["y"] for x in u)
            per_fold.append({"fold": f, "n_test": len(te), "n_seen": len(s),
                             "seen_rate": round(len(s) / max(len(te), 1), 6)})
        per_repeat[f"repeat{r}"] = {
            "n_test_total": seen_n + uns_n,
            "n_text_seen_in_train": seen_n,
            "seen_rate": round(seen_n / max(seen_n + uns_n, 1), 6),
            "seen_positives": seen_p,
            "seen_positive_rate": round(seen_p / max(seen_n, 1), 6),
            "n_text_unseen": uns_n,
            "unseen_rate": round(uns_n / max(seen_n + uns_n, 1), 6),
            "unseen_positives": uns_p,
            "unseen_positive_rate": round(uns_p / max(uns_n, 1), 6),
            "per_fold": per_fold,
        }
    counts = sorted(texts.values())
    return {
        "n_statements": len(rows),
        "n_unique_texts": len(texts),
        "n_unique_positive": sum(v["y"] for v in uniq.values()),
        "unique_positive_rate": round(
            sum(v["y"] for v in uniq.values()) / max(len(uniq), 1), 6),
        "statement_positive_rate": round(
            sum(x["y"] for x in rows) / max(len(rows), 1), 6),
        "repetitions": {"min": counts[0], "median": counts[len(counts) // 2],
                        "max": counts[-1],
                        "n_texts_appearing_once": sum(1 for c in counts if c == 1)},
        "same_text_label_conflicts": len(conflict),
        "per_repeat": per_repeat,
        "cause": ("去掉上下文后，不同提示生成的同一个编号塌成完全相同的输入。"
                  "提示级分组正确，但它管不住**跨提示**的文本重复。"),
        "what_this_means": (
            "探针有机会记住『编号 → 标签』的对应关系；**这不证明**它实际依赖记忆，"
            "但新方案的结果**不能**直接解释为『识别未见过的编号』的能力。"),
        "what_this_does_NOT_mean": (
            "这是**新方案（P2）的限制**，与旧的六条件读出器结果无关 —— "
            "旧方案用的是每条 trace 各自的池化隐藏状态，不同 trace 不会塌成同一个输入。"
            "**不构成**对已落盘结果的质疑。"),
        "unseen_subset_caveat": (
            "『未见文本』子集可以报，但它与『已见』子集的**标签构成明显不同**"
            "（未见子集正例率高得多，因为虚构编号往往只出现一次）。"
            "报它时必须同时报两个子集的规模与正例率，不得只挑一个。"),
        "not_a_fix": (
            "按唯一文本去重再训练**不能**解决这件事 —— 那只是换了训练行的权重，"
            "同一个编号照样同时出现在训练与测试。要真正回答泛化问题，"
            "需要**按编号**划分（而不是按提示），那是另一个 estimand，"
            "本轮**不做**，也不得临时改划分。"),
    }


def unique_text_map(rows: list[dict]) -> dict:
    """唯一文本 → 全部记录的映射。编码只需跑唯一文本，再映射回去。

    ⚠️ 这是**省编码成本**的手段，**不是**对上面那条限制的修复：
       训练行、权重、划分**一律保持原样**，每条记录仍按原样进入训练/评价。
    """
    m: dict[str, dict] = {}
    for x in rows:
        e = m.setdefault(x["text_sha256"], {
            "text": x["text"], "template": x["template"],
            "family": x["family"], "normalized": x["normalized"],
            "y": x["y"], "statement_ids": []})
        if e["y"] != x["y"]:
            raise RuntimeError(f"{x['text_sha256']}: 同一文本出现不同标签 —— 停")
        e["statement_ids"].append(x["statement_id"])
    for e in m.values():
        e["n_records"] = len(e["statement_ids"])
    return m


def pick_microbatch(rows: list[dict], n: int = 32) -> list[dict]:
    """计时小批：按**文本 sha256** 排序取，三个模板各取一段。

    **与标签完全无关** —— 不按标签平衡、不按分数挑。
    它只用于测单次前向耗时与显存，**不产出任何估计量**。
    """
    per = {k: n // len(TEMPLATES) for k in TEMPLATES}
    for k in list(TEMPLATES)[: n - sum(per.values())]:
        per[k] += 1
    picked = []
    for tkey, k in per.items():
        # **按文本去重**：同一个编号会出现在多条 trace 上，文本完全相同。
        # 计时批里放重复文本等于浪费槽位（覆盖不到词元长度的变化），故每个
        # 文本只留一条（留 statement_id 最小的那条，确定性）。
        seen: dict[str, dict] = {}
        for x in sorted((r for r in rows if r["template"] == tkey),
                        key=lambda r: r["statement_id"]):
            seen.setdefault(x["text_sha256"], x)
        cand = sorted(seen.values(), key=lambda x: x["text_sha256"])
        if len(cand) < k:
            raise RuntimeError(f"{tkey}: 去重后只有 {len(cand)} 条，不足 {k} 条")
        picked.extend(cand[:k])
    if len({x["text_sha256"] for x in picked}) != len(picked):
        raise RuntimeError("小批里仍有重复文本 —— 停")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--microbatch", type=int, default=32)
    ap.add_argument("--verify-tokenizer", action="store_true",
                    help="加载**本地分词器**（不是权重、不前向）核验 BOS/EOS 口径")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    t0 = time.perf_counter()
    cfg = V3AnalysisConfig()
    cfg.assert_production()

    recs = [json.loads(x) for x in
            (ROOT / MODEL["records"]).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    rows, skipped = build_statements(recs)
    folds = materialise_folds(recs, cfg)

    # ── 划分核验：与主分析的 roster **逐位比对**，不一致就停 ────────────────
    ref_p = ROOT / "outputs/stage1/posthoc_readout_probes/results_h1.json"
    verify: dict = {"reference": str(ref_p.relative_to(ROOT)), "checked": False}
    if ref_p.exists():
        ref = json.loads(ref_p.read_text(encoding="utf-8"))["roster"]
        bad = []
        for k, v in ref["outer_splits"].items():
            mine = folds["repeats"].get(k)
            if mine is None:
                bad.append(f"{k}: 缺")
                continue
            for field in ("fold_of_sha", "test_sizes", "test_positives"):
                if v[field] != mine[field]:
                    bad.append(f"{k}.{field}: 主分析 {v[field]} ≠ 本次 {mine[field]}")
        for field, mine_v in (("ids_sha", folds["ids_sha"]), ("n", folds["n"]),
                              ("n_positive", folds["n_positive"]),
                              ("n_groups", folds["n_groups"])):
            if ref.get(field) != mine_v:
                bad.append(f"{field}: 主分析 {ref.get(field)} ≠ 本次 {mine_v}")
        if bad:
            raise RuntimeError("划分/样本与主分析不一致 —— **停**，不落盘：\n  "
                               + "\n  ".join(bad))
        verify.update(checked=True, passed=True,
                      fold_of_sha={k: v["fold_of_sha"]
                                   for k, v in folds["repeats"].items()},
                      ids_sha=folds["ids_sha"],
                      note="外层折 / 样本名单 / 标签数 / 组数**逐位相同**")
        print("[verify] 划分与主分析逐位一致：" + "  ".join(
            f"{k}={v['fold_of_sha']}" for k, v in folds["repeats"].items()))
    else:
        raise RuntimeError(f"找不到主分析 roster（{ref_p}）—— 无法核验划分，停")

    # 每条陈述继承其 trace 的折（**不对陈述重新分折**）
    for r in range(cfg.n_repeats):
        fo = folds["repeats"][f"repeat{r}"]["fold_of"]
        for x in rows:
            x[f"fold_r{r}"] = fo[x["trace_id"]]
    missing = [x["statement_id"] for x in rows if x.get("fold_r0") is None]
    if missing:
        raise RuntimeError(f"{len(missing)} 条陈述的 trace 不在划分里 —— 停")

    dup = duplicate_audit(rows, cfg)
    umap = unique_text_map(rows)
    n_tr = len({x["trace_id"] for x in rows})

    tok_check = {"ran": False}
    if args.verify_tokenizer:
        # **只加载分词器**：不是模型权重，不前向，不占显存。
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        texts = sorted({x["text"] for x in rows})
        ndiff, lens = 0, []
        for t in texts:
            a = tk(t, add_special_tokens=True)["input_ids"]
            b = tk(t, add_special_tokens=False)["input_ids"]
            ndiff += int(a != b)
            lens.append(len(b))
        lens.sort()
        tok_check = {
            "ran": True, "model_path": args.model_path,
            "tokenizer_class": type(tk).__name__,
            "bos_token_id": tk.bos_token_id, "eos_token_id": tk.eos_token_id,
            "n_unique_texts": len(texts),
            "n_differing_true_vs_false": ndiff,
            "token_len": {"min": lens[0], "median": lens[len(lens) // 2],
                          "p90": lens[int(0.9 * len(lens))], "max": lens[-1]},
            "conclusion": ("两种设置结果相同 → 取 add_special_tokens=True（作者默认口径）"
                           if ndiff == 0 else
                           "**存在差异 —— 口径必须重新裁定，不得默认取 True**"),
            "loaded": "仅分词器；**未加载模型权重、未前向**",
        }
        print(f"[tokenizer] {tok_check['tokenizer_class']}  bos={tk.bos_token_id}  "
              f"True/False 结果不同 {ndiff}/{len(texts)}  "
              f"词元长度 {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}")
        if ndiff:
            raise RuntimeError("add_special_tokens 两种设置结果不同 —— 停，需重新裁定口径")
    by_t = {k: sum(1 for x in rows if x["template"] == k) for k in TEMPLATES}
    micro = pick_microbatch(rows, args.microbatch)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "h1_statements_u1.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
        encoding="utf-8")
    (args.out / "h1_fold_map.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out / "h1_unique_texts.jsonl").write_text(
        "\n".join(json.dumps({"text_sha256": k, **v}, ensure_ascii=False)
                  for k, v in sorted(umap.items())) + "\n", encoding="utf-8")
    (args.out / "h1_duplicate_audit.json").write_text(
        json.dumps(dup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "h1_microbatch_32.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in micro) + "\n",
        encoding="utf-8")

    manifest = {
        "schema": "p2_prep_h1_v1",
        "scope": "**只准备，不提取**：本脚本不加载权重、不前向、不调用外部 API",
        "model": MODEL,
        "condition": "U1（全部不同主标识符各一条陈述）",
        "statements": {"n": len(rows), "n_traces": n_tr,
                       "n_groups": folds["n_groups"],
                       "n_positive": sum(x["y"] for x in rows),
                       "by_template": by_t, "skipped": skipped},
        "templates": TEMPLATES,
        "template_wording_note": (
            "**不用 'valid'**：grounded 的判据是 `OntologyIndex.has()`，"
            "只看目录里有没有记录，与状态无关 —— 实测含 attack revoked 870 / "
            "deprecated 32、cwe Deprecated 118。故改为『目录中有记录』。"),
        "tokenizer_settings": {**TOKENIZER_SETTINGS,
                               "verification": tok_check},
        "duplicate_audit": dup,
        "unique_texts": {
            "n": len(umap),
            "file": "h1_unique_texts.jsonl",
            "use": ("编码**只需跑这 870 条**，再按 text_sha256 映射回全部 10168 条记录。"
                    "训练行、权重、划分**一律保持原样** —— 这是省成本，"
                    "**不是**对跨折重复那条限制的修复。"),
        },
        "label_direction": {
            "grounded": "编号存在 → 陈述为真 → y=0",
            "fabricated": "编号不存在 → 陈述为假 → y=1",
            "note": "y 与主分析的 `label` 同向（fabricated=1）",
        },
        "leakage_guards": {
            "template_selector": ("granularity，由表面形式决定"
                                  "（h1_identifier.py:126 `'.' in normalized`），"
                                  "**不查本体** → 不泄漏标签"),
            "fields_excluded_from_text": ["label", "ontology_status", "parent",
                                          "sub_id", "raw_surface", "模型原文上下文"],
            "microbatch_selection": "按 text_sha256 排序，**与标签无关**",
        },
        "splits": {
            "materialised": True,
            "grouping_key": "group_id（提示级）",
            "rule": "先在原 trace 总体上分折 → 物化 trace_id→fold → 陈述**继承**",
            "per_repeat": {k: {kk: v[kk] for kk in
                               ("fold_seed", "fold_of_sha", "test_sizes",
                                "test_positives")}
                           for k, v in folds["repeats"].items()},
            "verify_against": ("与主分析 roster 的 outer_splits[repeatR].fold_of_sha "
                               "逐位比对；不一致即停"),
            "verification": verify,
        },
        "encoding_settings": {
            "layer_block_zero_based": MODEL["block_index_zero_based"],
            "layer_hidden_states_index": MODEL["hidden_states_tuple_index"],
            "mapping_rule": "block k → hidden_states[k+1]（[0] 是 embedding）",
            "final_norm_applied": ("否 —— 读的是 hidden_states[20]，不是末项 [28]；"
                                   "若层清单含末项必须单独标注"),
            "read_position": "末词元（last token）",
            "batch": "batch=1 逐条编码（不 padding；如改批处理必须**左** padding）",
            "add_special_tokens": ("**True**（作者默认口径）。实测本地 Qwen2.5 "
                                   "tokenizer 的 bos_token_id 为 None，870 个唯一文本在 "
                                   "True/False 下 input_ids 完全相同（差异 0/870），"
                                   "故取 True 不引入额外词元。生成侧用的是 False，"
                                   "两侧口径**分别登记**，不混称。"),
            "chat_template": "**不套** —— 陈述是独立的判真假输入，不是对话轮次",
            "dtype": "与生成侧同后端同精度；指纹随产物落盘",
            "nonfinite": "逐层逐样本检查；有非有限值即报错，不静默删层/删样本",
        },
        "output_format": {
            "vectors": "p2_hidden_{model}_{layer}.npz  键 = statement_id，值 = (d,) float32",
            "index": "p2_index_{model}.jsonl  每条含 statement_id / trace_id / "
                     "group_id / text_sha256 / 实际词元位置 / 序列长度 / 层号（两套）",
            "fingerprint": "p2_run_fingerprint_{model}.json  模型/后端/精度/层/设置",
        },
        "microbatch": {
            "n": len(micro), "purpose": "**只测耗时与显存，不产出任何估计量**",
            "selection": "按 text_sha256 排序，三模板各取一段",
            "by_template": {k: sum(1 for x in micro if x["template"] == k)
                            for k in TEMPLATES},
            "label_composition_descriptive": {
                "y=1": sum(x["y"] for x in micro),
                "y=0": sum(1 - x["y"] for x in micro)},
        },
        "blocking_for_training": (
            "跨折文本重复（见 duplicate_audit）**不阻塞**小批资源计时，"
            "但**正式训练比较前必须把解释范围写清楚**：结果不得直接解释为"
            "『识别未见过的编号』。"),
        "not_prepared_here": [
            "MCQ 的陈述与题目判读（H1 不依赖它，见运行清单的检查点划分）",
            "探针的选参 / 基线核验（属训练比较前的检查点）",
            "微基准之外的正式提取（需单独授权）",
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_secs": round(time.perf_counter() - t0, 2),
    }
    (args.out / "h1_prep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"模型      {MODEL['model_key']}  L={MODEL['num_hidden_layers']} "
          f"d={MODEL['hidden_size']}  block {MODEL['block_index_zero_based']} "
          f"→ hidden_states[{MODEL['hidden_states_tuple_index']}]")
    print(f"陈述      {len(rows)} 条  来自 {n_tr} 条 trace / "
          f"{folds['n_groups']} 个 group  正例 {sum(x['y'] for x in rows)}")
    print(f"按模板    {by_t}")
    print(f"跳过      {skipped}")
    for k, v in folds["repeats"].items():
        print(f"划分      {k}  seed={v['fold_seed']}  fold_of_sha={v['fold_of_sha']}  "
              f"test_sizes={v['test_sizes']}")
    print(f"唯一文本  {dup['n_unique_texts']} 种（{len(rows)} 条记录）  "
          f"其中 y=1 {dup['n_unique_positive']} "
          f"（唯一文本正例率 {dup['unique_positive_rate']:.1%} vs "
          f"记录级 {dup['statement_positive_rate']:.1%}）")
    print(f"重复次数  min/中位/max = {dup['repetitions']['min']}/"
          f"{dup['repetitions']['median']}/{dup['repetitions']['max']}；"
          f"只出现一次的文本 {dup['repetitions']['n_texts_appearing_once']}")
    for k, v in dup["per_repeat"].items():
        print(f"⚠️ 跨折重复 {k}  测试折文本已在训练折出现 "
              f"{v['n_text_seen_in_train']}/{v['n_test_total']} = {v['seen_rate']:.2%}"
              f"（正例率 {v['seen_positive_rate']:.1%}）；"
              f"未见 {v['n_text_unseen']} 条（正例率 {v['unseen_positive_rate']:.1%}）")
    print(f"小批      {len(micro)} 条  "
          f"{manifest['microbatch']['by_template']}  "
          f"（标签组成仅描述性：y=1 {manifest['microbatch']['label_composition_descriptive']['y=1']}）")
    print("\n示例陈述（每模板一条）：")
    seen = set()
    for x in rows:
        if x["template"] not in seen:
            seen.add(x["template"])
            print(f"  [{x['template']:<20}] {x['text']}   (y={x['y']})")
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
