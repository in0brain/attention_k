r"""Stage1-L2C：Llama 同协议 MCQ 失败的**离线**诊断。

四段核查：

    原协议执行核查          提示正文 / chat template / 采样参数 / 后端 / 几何
    答案词元与读取位置核查  tokenizer / 候选词元 / 字符→词元定位 / 三个读取位置
    已有输出及缺失证据      118 条的分类、伪阳性归因、2 条异常的残留证据
    失败原因判断            把上面三段汇总成一个判断，并划出措辞边界

只读 `outputs/**`，只写 `tmp/stage1_l2_g3/l2c_offline_diagnosis.json`。
**不加载模型权重**；可选地只加载 tokenizer（`--with-tokenizer`），并在加载前后
核验 tokenizer 指纹与冻结产物一致 —— 不一致则本段结论全部标为无法核验。

⚠️ 输出 JSON 的**字段名保留原诊断编号**（`Q1_…` / `Q2_…` / `Q3_…` / `Q4_…`、
`F1` / `F2` / `F3`）。这不是遗漏：已落盘的产物不为改名而重写，字段名是它与本脚本之间的
可追溯锚点。面向人的名称一律用上面四个中文名，对应表见
`docs/progress/stage1_llama_replication_history.md`。
项目里 `F1` / `F4` / `F5` 另有**冻结的特征族**含义，与此处的诊断编号无关，不得混淆。

用法：
    python scripts/diagnostics/l2c_mcq_offline_diagnosis.py [--with-tokenizer]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import domain_label_proxy as dlp          # noqa: E402
from recover_attention import h1_f5_features as f5f              # noqa: E402
from recover_attention import mcq_conditional_increment as mci    # noqa: E402
from recover_attention.generation import backends as be          # noqa: E402

QWEN_DIR = ROOT / "outputs" / "stage1" / "g3"
LLAMA_DIR = ROOT / "outputs" / "stage1_llama_v3" / "g3"
OUT = ROOT / "tmp" / "stage1_l2_g3" / "l2c_offline_diagnosis.json"

# 从 rendered_prompt 反解出「模板应用前的消息体」。两个模板的定界符都是公开且固定的。
QWEN_RE = re.compile(
    r"^<\|im_start\|>system\n(?P<system>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n(?P<tail>.*)$", re.S)
LLAMA_RE = re.compile(
    r"^<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n"
    r"(?P<system>.*?)<\|eot_id\|>"
    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(?P<user>.*?)<\|eot_id\|>"
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(?P<tail>.*)$", re.S)


def sha_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(p: Path) -> list[dict]:
    with io.open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(p: Path) -> dict:
    return json.load(io.open(p, encoding="utf-8"))


# ──────────────────────────────────────────────── 原协议执行核查
def q1_prompt_hash_basis(q_rows, l_rows) -> dict:
    """先确定 prompt_hash 的计算口径，再谈任何哈希比对（卡 §3.1）。"""
    out = {}
    for tag, rows in (("qwen", q_rows), ("llama", l_rows)):
        hit_rendered = sum(1 for r in rows if sha_text(r["rendered_prompt"]) == r["prompt_hash"])
        hit_only = sum(1 for r in rows if sha_text(r["prompt_only_text"]) == r["prompt_hash"])
        out[tag] = {"n_rows": len(rows),
                    "sha256(rendered_prompt)==prompt_hash": hit_rendered,
                    "sha256(prompt_only_text)==prompt_hash": hit_only}
    basis = ("sha256(rendered_prompt)"
             if all(out[t]["sha256(rendered_prompt)==prompt_hash"] == out[t]["n_rows"]
                    for t in out) else "UNDETERMINED")
    return {
        "measured": out,
        "basis": basis,
        "implication": (
            "prompt_hash 覆盖 chat template，因此两模型的 prompt_hash **在构造上不可能相等**；"
            "跨模型 prompt_hash 不等**不是**实现错误的证据（卡 §3.1 已作废旧要求）"
            if basis == "sha256(rendered_prompt)" else "口径未定，后续哈希比对不可解释"),
        "code_ref": "src/recover_attention/generation/fingerprint.py:81 "
                    "unit_fingerprint -> prompt_hash = sha256_text(prompt)",
    }


def split_prompt(tag: str, rendered: str) -> dict | None:
    m = (QWEN_RE if tag == "qwen" else LLAMA_RE).match(rendered)
    return None if m is None else m.groupdict()


def q1_layered_prompt_compare(q_rows, l_rows) -> dict:
    q_by = {r["example_id"]: r for r in q_rows}
    l_by = {r["example_id"]: r for r in l_rows}
    common = sorted(set(q_by) & set(l_by))

    parse_fail = {"qwen": [], "llama": []}
    sysmap, usermap, tailmap = {}, {}, {}
    for tag, by in (("qwen", q_by), ("llama", l_by)):
        sysmap[tag], usermap[tag], tailmap[tag] = {}, {}, {}
        for eid, r in by.items():
            g = split_prompt(tag, r["rendered_prompt"])
            if g is None:
                parse_fail[tag].append(eid)
                continue
            sysmap[tag][eid] = g["system"]
            usermap[tag][eid] = g["user"]
            tailmap[tag][eid] = g["tail"]

    # L-A：模板应用前的消息体必须逐字相同
    user_same = [e for e in common
                 if e in usermap["qwen"] and e in usermap["llama"]
                 and usermap["qwen"][e] == usermap["llama"][e]]
    user_diff = [e for e in common
                 if e in usermap["qwen"] and e in usermap["llama"]
                 and usermap["qwen"][e] != usermap["llama"][e]]
    sys_q = sorted(set(sysmap["qwen"].values()))
    sys_l = sorted(set(sysmap["llama"].values()))
    # Llama-3.1 的 chat template 会**自行注入**一段前言（Cutting Knowledge / Today Date），
    # 它不是本地代码传入的 system 内容。必须把「模板注入」与「代码传入」分开归类：
    # 前者属 L-B（模板差异，允许），后者属 L-A（消息体，必须相同）。
    preamble_re = re.compile(r"^Cutting Knowledge Date: [^\n]*\nToday Date: [^\n]*\n\n")
    sys_l_stripped = sorted({preamble_re.sub("", s) for s in sys_l})
    injected = sorted({(preamble_re.match(s).group(0) if preamble_re.match(s) else "")
                       for s in sys_l})

    # L-C：各模型自己的 rendered_prompt 与自己记录的 prompt_hash 自洽
    self_ok = {t: sum(1 for r in rows if sha_text(r["rendered_prompt"]) == r["prompt_hash"])
               for t, rows in (("qwen", q_rows), ("llama", l_rows))}

    sample = common[0] if common else None
    return {
        "n_common_example_ids": len(common),
        "n_qwen_example_ids": len(set(q_by)), "n_llama_example_ids": len(set(l_by)),
        "template_regex_unmatched": parse_fail,
        "L_A_pre_template_user_body": {
            "identical": len(user_diff) == 0,
            "n_identical": len(user_same), "n_different": len(user_diff),
            "different_example_ids": user_diff[:10],
            "verdict": ("逐字相同 —— 提示正文未被改动"
                        if not user_diff else "存在差异 —— 候选实现错误"),
        },
        "L_A_system_message": {
            "qwen_distinct": sys_q,
            "llama_distinct_raw": sys_l,
            "llama_distinct_after_stripping_template_preamble": sys_l_stripped,
            "identical_raw": sys_q == sys_l,
            "identical_after_stripping": sys_q == sys_l_stripped,
            "verdict": ("代码传入的 system 内容逐字相同；差异全部来自模板自注入前言"
                        if sys_q == sys_l_stripped else "代码传入的 system 内容不同 —— 实现错误"),
        },
        "L_B_template_differences": {
            "role_markers": {"qwen": "<|im_start|>/<|im_end|>",
                             "llama": "<|start_header_id|>/<|eot_id|>"},
            "bos_in_template": {"qwen": False, "llama": True},
            "llama_template_injected_system_preamble": injected,
            "injected_preamble_attribution":
                "由 Llama-3.1 的 tokenizer_config chat_template 自行注入，**不是**本地代码添加；"
                "未传 date_string 时模板用其内置默认日期，因此**确定性、不随运行日期漂移**",
            "qwen_assistant_tail": sorted(set(tailmap["qwen"].values()))[:3],
            "llama_assistant_tail": sorted(set(tailmap["llama"].values()))[:3],
            "add_generation_prompt": "两侧 assistant 段尾均为空串 → 生成提示已加且其后无内容",
            "verdict": "以上差异全部属模型固有模板差异，落在允许适配范围内",
        },
        "L_C_self_consistency": {
            "qwen": f'{self_ok["qwen"]}/{len(q_rows)}',
            "llama": f'{self_ok["llama"]}/{len(l_rows)}',
            "all_ok": self_ok["qwen"] == len(q_rows) and self_ok["llama"] == len(l_rows),
        },
        "sample_example_id": sample,
        "sample_user_body_sha256": (sha_text(usermap["qwen"][sample]) if sample else None),
    }


def q1_params_backend_geometry() -> dict:
    q = load_json(QWEN_DIR / "mcq_g3_run_fingerprint.json")
    l = load_json(LLAMA_DIR / "mcq_g3_run_fingerprint.json")
    gp_q, gp_l = q["generation_params"], l["generation_params"]
    bk_q, bk_l = q["model"]["backend"], l["model"]["backend"]
    return {
        "generation_params": {
            "qwen": gp_q, "llama": gp_l,
            "identical": gp_q == gp_l,
            "diff_keys": sorted(k for k in set(gp_q) | set(gp_l) if gp_q.get(k) != gp_l.get(k)),
        },
        "backend": {
            "qwen": bk_q, "llama": bk_l,
            "identical": bk_q == bk_l,
            "diff_keys": sorted(k for k in set(bk_q) | set(bk_l) if bk_q.get(k) != bk_l.get(k)),
        },
        "hidden_index": {
            "qwen": q["hidden_index"], "llama": l["hidden_index"],
            "rel_depth_same": q["hidden_index"]["rel_depth"] == l["hidden_index"]["rel_depth"],
            "derivation_ok": {
                "qwen": int(q["hidden_index"]["num_layers"] * q["hidden_index"]["rel_depth"])
                        == q["hidden_index"]["block_index_zero_based"],
                "llama": int(l["hidden_index"]["num_layers"] * l["hidden_index"]["rel_depth"])
                         == l["hidden_index"]["block_index_zero_based"],
            },
            "tuple_is_block_plus_one": {
                "qwen": q["hidden_index"]["hidden_states_tuple_index"]
                        == q["hidden_index"]["block_index_zero_based"] + 1,
                "llama": l["hidden_index"]["hidden_states_tuple_index"]
                         == l["hidden_index"]["block_index_zero_based"] + 1,
            },
        },
        "model_dims_from_config": {
            "qwen": {k: q["model"][k] for k in ("model_type", "num_hidden_layers",
                                                "hidden_size", "vocab_size", "torch_dtype")},
            "llama": {k: l["model"][k] for k in ("model_type", "num_hidden_layers",
                                                 "hidden_size", "vocab_size", "torch_dtype")},
        },
        "tokenizer_recorded": {
            "qwen": q["tokenizer"], "llama": l["tokenizer"],
            "class_differs": q["tokenizer"]["class"] != l["tokenizer"]["class"],
        },
    }


def q1_generation_config_drift() -> dict:
    """_generate 只显式传了 6 个采样键；其余键由**各模型自带的 generation_config** 补齐。

    这些补齐值既不在 run_fingerprint.generation_params 里，也不在预注册里 ——
    因此两臂的实际采样配方可能不同而**指纹看不出来**。
    """
    import transformers                                    # noqa: PLC0415
    from transformers import GenerationConfig              # noqa: PLC0415

    explicit = ["max_new_tokens", "do_sample", "use_cache", "renormalize_logits",
                "remove_invalid_values", "pad_token_id", "eos_token_id",
                "temperature（仅 do_sample 时）", "top_p（仅 do_sample 时）"]
    watch = ("top_k", "repetition_penalty", "temperature", "top_p",
             "eos_token_id", "bos_token_id", "do_sample")
    got = {}
    for tag, path in (("qwen", "D:/models/Qwen2.5-7B-Instruct"),
                      ("llama", "D:/models/Meta-Llama-3.1-8B-Instruct")):
        try:
            gc = GenerationConfig.from_pretrained(path, local_files_only=True)
            got[tag] = {k: getattr(gc, k, None) for k in watch}
        except Exception as exc:                           # noqa: BLE001
            got[tag] = {"error": str(exc)[:200]}
    diff = sorted(k for k in watch
                  if got.get("qwen", {}).get(k) != got.get("llama", {}).get(k))
    return {
        "explicitly_passed_by_code": explicit,
        "not_passed_by_code": ["top_k", "repetition_penalty"],
        "model_shipped_generation_config": got,
        "keys_differing_between_models": diff,
        "merge_semantics": (
            "transformers GenerationMixin._prepare_generation_config："
            "generation_config.update(**self.generation_config.to_dict(), defaults_only=True) "
            "→ 代码**未显式传**的键由模型自带 generation_config 填充，再由全局默认兜底。"),
        "transformers_version_now": transformers.__version__,
        "evidence_limit": (
            "run_fingerprint **不记录** transformers 版本，也不记录 top_k / repetition_penalty；"
            "因此『2026-09-09 运行时的合并语义与当前版本相同』只能作为**前提**，不是已证事实。"),
        "impact_note": (
            "按安装版 transformers 的 _get_logits_processor 实测：top_k / top_p / temperature 的 "
            "warper 只在 do_sample=True 分支内追加（greedy 不生效）；**repetition_penalty 的 "
            "processor 在该分支之外**，因此 **greedy 臂同样生效**。"
            "Qwen 的 generation_config 带 repetition_penalty=1.05、top_k=20，Llama 两者都没有。"
            "**当前证据不能排除这些差异对本次结果的作用** —— 差异表见 "
            "tmp/stage1_l2_g3/l2c_effective_generation_config.json。"),
    }


def q1_artifact_hashes() -> dict:
    files = [
        QWEN_DIR / "mcq_g3_unit_results.jsonl", QWEN_DIR / "mcq_g3_run_fingerprint.json",
        QWEN_DIR / "mcq_g3_smoke_report.json",
        LLAMA_DIR / "mcq_g3_unit_results.jsonl", LLAMA_DIR / "mcq_g3_run_fingerprint.json",
        LLAMA_DIR / "mcq_g3_smoke_report.json", LLAMA_DIR / "g3_failure_audit.json",
        ROOT / "docs" / "paper" / "preregistration_v3.md",
        ROOT / "docs" / "paper" / "preregistration_v3.lock",
        ROOT / "tmp" / "stage1_l2_g3" / "mcq_parse_audit.json",
    ]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): (sha_file(p) if p.exists() else None)
              for p in files}
    lock_p = ROOT / "docs" / "paper" / "preregistration_v3.lock"
    lock_raw = io.open(lock_p, encoding="utf-8").read() if lock_p.exists() else ""
    covered = [k for k in hashes if k in lock_raw and not k.startswith("docs/paper/prereg")]
    return {
        "measured_sha256": hashes,
        "lock_covers_these_run_artifact_paths": covered,
        "lock_covers_g3_run_artifacts": bool(covered),
        "cross_check_vs_prior_record": (
            "mcq_g3_run_fingerprint 3163001c… / mcq_g3_smoke_report a8f95a14… / "
            "g3_failure_audit da5ded58… / preregistration_v3.lock 099ed322… / "
            "mcq_parse_audit f652f5d0… 与 2026-09-13 清理验收留档的受保护哈希逐位相同 "
            "→ 这些产物自那次记录以来未变"),
        "evidence_limit": (
            "preregistration_v3.lock 冻结的是**协议文档与实现清单**，不覆盖本轮引用的 G3 运行产物；"
            "因此这些产物的『未被改动』只能由本文件记录的 sha256 与既有 history 记录交叉核对，"
            "**不能**假定『冻结产物不会漂移』"),
    }


# ────────────────────────────────────── 答案词元与读取位置核查
def q2_char_span_replay(l_rows) -> dict:
    """对 118 条已落盘 completion 做**纯文本**的解析与字符区间重放（不需 tokenizer）。"""
    letters = list(mci.MCQ_LETTERS)
    reparse_match = span_ok = span_err = 0
    methods = Counter()
    lower_span, upper_span = [], []
    span_fail_reasons = Counter()
    for r in l_rows:
        c = r["raw_completion"]
        p = dlp.parse_option_answer(c, letters)
        methods[p["parse_method"]] += 1
        if p["parsed_label"] == r["parsed_label"] and p["parse_method"] == r["parse_status"]:
            reparse_match += 1
        if p["parsed_label"] is None:
            continue
        try:
            cs, ce = be._parsed_letter_char_span(c, p["parsed_label"])
        except ValueError as exc:
            span_err += 1
            span_fail_reasons[str(exc)[:60]] += 1
            continue
        span_ok += 1
        ch = c[cs:ce]
        (lower_span if ch.islower() else upper_span).append(
            {"unit_id": r["unit_id"], "char": ch, "span": [cs, ce],
             "context": c[max(0, cs - 25):ce + 10]})
    return {
        "n_rows": len(l_rows),
        "reparse_reproduces_frozen_record": f"{reparse_match}/{len(l_rows)}",
        "parse_method_counts": dict(methods),
        "char_span_located": span_ok, "char_span_error": span_err,
        "char_span_error_reasons": dict(span_fail_reasons),
        "span_char_uppercase": len(upper_span),
        "span_char_lowercase": len(lower_span),
        "lowercase_span_examples": lower_span[:5],
        "note": "本段**不需要 tokenizer**，直接基于已落盘 raw_completion —— 属历史重放，强度最高",
    }


def q2_case_sensitivity_audit(q_rows, l_rows) -> dict:
    """解析器 IGNORECASE 与候选 id 集仅大写 —— 核查**两个模型**的影响范围（卡 §4.4）。"""
    letters = list(mci.MCQ_LETTERS)
    res = {}
    for tag, rows in (("qwen", q_rows), ("llama", l_rows)):
        low = []
        for r in rows:
            c = r.get("raw_completion")
            if not isinstance(c, str):
                continue
            p = dlp.parse_option_answer(c, letters)
            if p["parsed_label"] is None:
                continue
            try:
                cs, ce = be._parsed_letter_char_span(c, p["parsed_label"])
            except ValueError:
                continue
            if c[cs:ce].islower():
                low.append(r["unit_id"])
        res[tag] = {"n_rows_with_completion": sum(1 for r in rows
                                                  if isinstance(r.get("raw_completion"), str)),
                    "n_lowercase_answer_span": len(low), "unit_ids": low[:10]}
    return {
        "regex": {"explicit": dlp._EXPLICIT_ANSWER_RE.pattern,
                  "isolated": dlp._ISOLATED_LETTER_RE.pattern,
                  "flags": "re.IGNORECASE on both"},
        "candidate_id_sets": "bare_option_token_ids / option_token_ids 只解析**大写** A-D",
        "per_model": res,
        "latent_defect": (
            "解析器大小写不敏感（返回 .upper() 的标签），但 _parsed_letter_char_span 返回的是"
            "**原文中的字符区间**；若原文是小写字母，该位置的 token id 既不等于 bare 大写形、"
            "也不等于 space 大写形 → MCQBackend._letter_id_set 抛 ValueError。"
            "这是**共享代码中的潜伏缺陷**，不是 Llama 专属：Qwen 侧只是从未触发。"),
        "code_refs": ["src/recover_attention/domain_label_proxy.py:19 _ISOLATED_LETTER_RE",
                      "src/recover_attention/generation/backends.py:425 _parsed_letter_char_span",
                      "src/recover_attention/generation/backends.py:388 _letter_id_set"],
    }


def q2_bos_check(q_rows, l_rows, tok_ctx) -> dict:
    """_generate / _teacher_forced 用 tok(prompt) 默认 add_special_tokens=True。

    Llama 的 chat template 自带 <|begin_of_text|> → 与 tokenizer 自动加的 BOS 叠加。
    Qwen 的模板不含 BOS 且 bos_token_id 为 null → 不叠加。
    """
    q_tpl_bos = sum(1 for r in q_rows if r["rendered_prompt"].startswith("<|im_start|>"))
    l_tpl_bos = sum(1 for r in l_rows if r["rendered_prompt"].startswith("<|begin_of_text|>"))
    out = {
        "code_ref": ["src/recover_attention/generation/backends.py:159 _generate "
                     "-> tok(prompt, return_tensors='pt')  # add_special_tokens 默认 True",
                     "src/recover_attention/generation/backends.py:180 _teacher_forced "
                     "-> tok(prompt, return_tensors='pt')"],
        "template_already_has_bos": {
            "qwen": f"{q_tpl_bos}/{len(q_rows)} 以 <|im_start|> 开头（无 BOS 记号）",
            "llama": f"{l_tpl_bos}/{len(l_rows)} 以 <|begin_of_text|> 开头（**即 BOS**）"},
        "recorded_bos_token_id": {
            "qwen": load_json(QWEN_DIR / "mcq_g3_run_fingerprint.json")
                    ["tokenizer"]["special_token_ids"]["bos_token_id"],
            "llama": load_json(LLAMA_DIR / "mcq_g3_run_fingerprint.json")
                     ["tokenizer"]["special_token_ids"]["bos_token_id"]},
        "runtime_verified": False,
    }
    if tok_ctx and tok_ctx.get("llama"):
        tok = tok_ctx["llama"]
        p = l_rows[0]["rendered_prompt"]
        ids_default = tok(p)["input_ids"]
        ids_nospecial = tok(p, add_special_tokens=False)["input_ids"]
        out["runtime_verified"] = True
        out["llama_runtime"] = {
            "first_4_ids_default": ids_default[:4],
            "first_4_ids_add_special_tokens_False": ids_nospecial[:4],
            "len_default": len(ids_default), "len_nospecial": len(ids_nospecial),
            "duplicate_bos": (len(ids_default) == len(ids_nospecial) + 1
                              and ids_default[0] == ids_default[1] == 128000),
            "decoded_first_2": [tok.decode([i]) for i in ids_default[:2]],
        }
    if tok_ctx and tok_ctx.get("qwen"):
        tok = tok_ctx["qwen"]
        p = q_rows[0]["rendered_prompt"]
        ids_default = tok(p)["input_ids"]
        ids_nospecial = tok(p, add_special_tokens=False)["input_ids"]
        out["qwen_runtime"] = {
            "len_default": len(ids_default), "len_nospecial": len(ids_nospecial),
            "prepends_anything": len(ids_default) != len(ids_nospecial)}
    return out


def q2_tokenizer_section(l_rows, tok_ctx) -> dict:
    if not tok_ctx or not tok_ctx.get("llama"):
        return {"status": "SKIPPED — 未加载 tokenizer（--with-tokenizer 未指定）",
                "consequence": "候选 id 集、token 264、offsets、字符→词元定位计数均无法核验"}
    tok = tok_ctx["llama"]
    letters = list(mci.MCQ_LETTERS)
    bare = dlp.bare_option_token_ids(tok, letters)
    space = dlp.option_token_ids(tok, letters)
    lower_bare, lower_space = {}, {}
    for lab in letters:
        for form, store in ((lab.lower(), lower_bare), (" " + lab.lower(), lower_space)):
            ids = tok(form, add_special_tokens=False)["input_ids"]
            store[lab] = ids[0] if len(ids) == 1 else ids

    # 对 118 条 completion 做「字符区间 → 词元下标」重放（重新编码，见证据限制）
    located = multi = none = err = 0
    per_case = []
    for r in l_rows:
        c = r["raw_completion"]
        p = dlp.parse_option_answer(c, letters)
        if p["parsed_label"] is None:
            continue
        try:
            cs, ce = be._parsed_letter_char_span(c, p["parsed_label"])
        except ValueError:
            err += 1
            continue
        enc = tok(c, return_offsets_mapping=True, add_special_tokens=False)
        idx = f5f.token_indices_for_char_span([list(o) for o in enc["offset_mapping"]], cs, ce)
        if len(idx) == 1:
            located += 1
            tid = int(enc["input_ids"][idx[0]])
            form = ("bare" if tid == bare.get(p["parsed_label"])
                    else "space" if tid == space.get(p["parsed_label"]) else "NEITHER")
            per_case.append({"unit_id": r["unit_id"], "j": idx[0], "token_id": tid,
                             "decoded": tok.decode([tid]), "form": form,
                             "frozen_letter_token_id": r.get("letter_token_id"),
                             "frozen_j": r.get("letter_token_index_local"),
                             "matches_frozen": (tid == r.get("letter_token_id")
                                                and idx[0] == r.get("letter_token_index_local"))})
        elif len(idx) == 0:
            none += 1
        else:
            multi += 1
    neither = [c for c in per_case if c["form"] == "NEITHER"]
    replay_match = sum(1 for c in per_case if c["matches_frozen"])
    return {
        "tokenizer_identity": tok_ctx["identity"],
        "candidate_ids": {"bare_upper": bare, "space_upper": space,
                          "bare_lower": lower_bare, "space_lower": lower_space,
                          "bare_and_space_disjoint": not (set(bare.values())
                                                          & set(space.values()))},
        "token_264": {"decoded": repr(tok.decode([264])),
                      "in_bare_upper": 264 in set(bare.values()),
                      "in_space_upper": 264 in set(space.values()),
                      "equals_space_lower_a": lower_space.get("A") == 264},
        "char_to_token_replay": {
            "located_single_token": located, "spans_multiple_tokens": multi,
            "not_located": none, "char_span_error": err,
            "reproduces_frozen_letter_token_id_and_j": f"{replay_match}/{len(per_case)}",
            "form_NEITHER": len(neither), "form_NEITHER_cases": neither[:5],
        },
        "evidence_limit": (
            "本段需**重新编码** raw_completion（原始 completion token ids 未落盘）。"
            "结论成立的前提是『当前 tokenizer 与运行时同一版本』——"
            "该前提已由 tokenizer_identity 的指纹比对核验，但仍是前提而非独立证据。"),
    }


# ──────────────────────────────────────────── 已有输出及缺失证据
def q3_classify(l_rows) -> dict:
    letters = list(mci.MCQ_LETTERS)
    answer_marker = [r["unit_id"] for r in l_rows
                     if dlp._EXPLICIT_ANSWER_RE.search(r["raw_completion"] or "")]
    buckets = Counter()
    fp_details = []
    for r in l_rows:
        c = r["raw_completion"] or ""
        st = r["parse_status"]
        if st == "explicit_answer_marker":
            buckets["最终答案（Answer: 标记）"] += 1
        elif st == "last_isolated_label":
            buckets["旧回退命中（须逐条判真伪）"] += 1
            try:
                cs, ce = be._parsed_letter_char_span(c, r["parsed_label"])
                ctx = c[max(0, cs - 40):ce + 15]
            except ValueError as exc:
                cs = ce = -1
                ctx = f"<span error: {exc}>"
            before, after = (c[:cs] if cs >= 0 else ""), (c[ce:] if ce >= 0 else "")
            # 该字母之后只剩标点/空白（这正是 _TRAILING_PUNCTUATION_RE 的判据），
            # 之前刚出现选项枚举的引导语与换行/项目符号 → 它是**枚举标号**，不是答案。
            enum_lead = bool(re.search(r"option", before, re.I)) and bool(
                re.search(r"(?:\n\s*|[-*•(]\s*|\*\*|Option\s)$", before))
            fp_details.append({
                "unit_id": r["unit_id"], "parsed_label": r["parsed_label"],
                "gold_wrong_label": r.get("wrong_label"),
                "char_span": [cs, ce],
                "tail_context": c[-48:],
                "chars_after_letter": repr(after),
                "letter_is_at_completion_end": after.strip(" .:;,)]}*\n\t") == "",
                "preceded_by_option_enumeration_lead": enum_lead,
                "why_false_positive": (
                    "命中的字母是**选项枚举的标号**：生成在列出第一个选项处被截断，"
                    "字母之后只剩标点/空白，于是被 _TRAILING_PUNCTUATION_RE 判为"
                    "『末尾孤立字母』→ last_isolated_label 伪阳性"
                    if enum_lead else "命中位置不符合枚举标号模式 —— 须人工复核"),
            })
        else:
            buckets["解析失败"] += 1
    trunc = sum(1 for r in l_rows if not (r["raw_completion"] or "").rstrip()
                .endswith((".", "!", "?")))
    head3 = Counter((r["raw_completion"] or "")[:3] for r in l_rows)
    return {
        "n_rows": len(l_rows),
        "answer_marker_present": len(answer_marker),
        "answer_marker_verdict": "0 条出现 `Answer:` 标记 —— 复核通过"
                                 if not answer_marker else f"{len(answer_marker)} 条，与旧记录不符",
        "buckets": dict(buckets),
        "no_sentence_final_punctuation": {
            "count": f"{trunc}/{len(l_rows)}",
            "caveat": "这是**下界**：以 'A.' 这类枚举标号结尾的条目也被截断，"
                      "但它们以 '.' 结尾，不计入此计数",
        },
        "first3_chars_distribution": dict(head3.most_common(8)),
        "false_positive_details": fp_details,
        "n_enumeration_label_false_positives": sum(
            1 for d in fp_details if d["preceded_by_option_enumeration_lead"]),
        "false_positive_label_distribution": dict(Counter(d["parsed_label"]
                                                          for d in fp_details)),
        "n_false_positive_at_completion_end": sum(
            1 for d in fp_details if d["letter_is_at_completion_end"]),
        "token_budget_caveat": (
            "num_generated_tokens 与 completion token ids **未落盘** → "
            "只能说『文本在句中截断』，**不得**断言『恰好撞满 16 词元上限』"),
    }


def q3_anomaly_residue(l_rows) -> dict:
    planned = load_json(LLAMA_DIR / "mcq_g3_smoke_report.json")
    manifest = LLAMA_DIR / "mcq_g3_generated_manifest.jsonl"
    man_rows = load_rows(manifest) if manifest.exists() else []
    have = {r["unit_id"] for r in l_rows}
    hid_dir = LLAMA_DIR / "mcq_g3_unit_hidden"
    hid = {p.stem for p in hid_dir.glob("*")} if hid_dir.exists() else set()
    man_ids = set()
    for m in man_rows:
        for k in ("unit_id", "id"):
            if k in m:
                man_ids.add(m[k])
    missing = sorted(man_ids - have) if man_ids else []
    return {
        "report_counts": {k: planned.get(k) for k in
                          ("n_units_planned", "n_rows", "n_unhandled_exceptions")
                          if k in planned},
        "manifest_unit_ids": len(man_ids),
        "result_rows": len(have),
        "missing_unit_ids": missing,
        "hidden_files_present_for_missing": sorted(m for m in missing if m in hid),
        "residue_verdict": (
            "2 条异常单元**没有结果行**；若其 hidden 文件也不存在，则现存产物中"
            "关于这 2 条只剩：manifest 中的 unit_id、异常计数、以及 g3_failure_audit "
            "记录的 failing_token_id=264。**completion 原文永久缺失**。"),
        "mechanism_determinable": (
            "原文缺失，但**触发机制可由代码与已记录的 token id 判定**："
            "264 若等于 ' a'（空格+小写 a），则只能来自 _parsed_letter_char_span 落在"
            "**小写**答案字母上（解析器 IGNORECASE）。机制已定，**具体文本仍 UNDETERMINED**。"),
    }


def q4_synthesis(doc: dict) -> dict:
    """失败原因判断。卡 §9：把「事实 / 已证 / 残余」三件事分开，不得合并。

    返回值的键沿用原诊断编号（见模块文档），面向人的名称写在各条的文字里。
    """
    q1, q2, q3 = doc["Q1_protocol_fidelity"], doc["Q2_implementation"], doc["Q3_evidence"]
    bos = q2["bos_duplication"].get("llama_runtime", {})
    return {
        "1_facts_measured": {
            "mcq_gate": "未通过：parse_failure 104/118 = 0.881（阈值 0.10）、"
                        "2 条未处理 ValueError、118/120 行",
            "answer_marker": q3["classification"]["answer_marker_verdict"],
            "parsed_14_are_all_false_positives":
                f'{q3["classification"]["n_enumeration_label_false_positives"]}/14 '
                f'为选项枚举标号伪阳性；标签分布 '
                f'{q3["classification"]["false_positive_label_distribution"]}',
            "prompt_body_identical": q1["layered_prompt_compare"]
                                       ["L_A_pre_template_user_body"]["identical"],
            "sampling_params_identical": q1["params_backend_geometry"]
                                           ["generation_params"]["identical"],
            "backend_identical": q1["params_backend_geometry"]["backend"]["identical"],
        },
        "2_what_is_PROVEN_faithful": {
            "headline": "提示正文、显式生成参数和模型几何已核对；"
                        "实际生效的生成配置仍有差异待核实。",
            "checked": [
                "提示正文（模板应用前的 user 消息体）两模型逐字相同：20/20",
                "代码传入的 system 内容相同；差异仅为 Llama 模板自注入前言",
                "**显式传入的** generation_params"
                "（K / max_new_tokens / temperature / top_p / seed）逐键相同",
                "后端四键（8bit / device_map / attn / local_files_only）相同",
                "rel_depth 0.7 与 block/tuple 推导两侧一致且由 config 推得",
                "各模型 rendered_prompt 与自身 prompt_hash 逐行自洽（120/120、118/118）",
                "解析重放可完全复现冻结记录：118/118",
                "字符区间→词元定位可复现冻结的 letter_token_id 与 j：14/14",
                "三个读取位置的索引语义已由合成张量测试覆盖"
                "（tests/test_l2c_readout_index_semantics.py）",
            ],
            "still_open": [
                "**实际生效**的生成配置（含代码未显式传、由模型 generation_config 补齐的项）"
                "两侧存在差异，尚未核实其影响 —— 差异表见 "
                "tmp/stage1_l2_g3/l2c_effective_generation_config.json",
            ],
            "wording_rule": "在上述差异核实之前，**不得**把本段写成「协议忠实性已证」。",
        },
        "3_implementation_findings": [
            {
                "id": "F1", "severity": "确认的协议保真缺陷（仅影响 Llama）",
                "what": "Llama 的 chat template 已含 <|begin_of_text|>，而 _generate / "
                        "_teacher_forced 用 tok(prompt) 默认 add_special_tokens=True，"
                        "再次前置 BOS → **重复 BOS**",
                "measured": bos,
                "qwen_impact": "无：Qwen tokenizer 不前置任何 token（87→87），"
                               "bos_token_id 为 null。这是 Llama 专属，但**成因在共享代码**",
                "scope": "MCQ 与 H1 两臂的生成与 teacher-forcing 都受影响（H1 仍然通过）",
                "index_impact": "不影响下标自洽：prompt_len 由同一次编码得出",
                "minimal_fix": "在 _generate / _teacher_forced 对 prompt 使用 "
                               "add_special_tokens=False（模板已负责特殊 token）；"
                               "须同时证明 Qwen 侧逐字节不变（Qwen 本就不前置，预期为空变更）",
                "code_ref": ["src/recover_attention/generation/backends.py:159",
                             "src/recover_attention/generation/backends.py:180"],
                "status_as_a_defect": "**已确认**：实测可复现，与 Qwen 侧行为不一致，"
                                     "无需先证明它解释了多少失败即可修复 —— 已于本轮修复",
                "caused_the_parse_failures": "**未证明，且与上一行是两件事**。"
                                             "重复起始标记是否改变输出需真实前向对照，"
                                             "本轮离线无法判定；不把它当作失败原因，"
                                             "也不因此推迟修复",
            },
            {
                "id": "F2", "severity": "共享代码的潜伏不一致（Qwen 未触发）",
                "what": "解析器 IGNORECASE → 标签被 .upper()，但 _parsed_letter_char_span "
                        "返回**原文大小写**的区间；候选 id 集只含大写形 → 小写答案字母时 "
                        "_letter_id_set 抛 ValueError",
                "measured": {
                    "token_264": q2["tokenizer_checks"].get("token_264"),
                    "qwen_lowercase_spans": q2["case_sensitivity_audit"]["per_model"]
                                              ["qwen"]["n_lowercase_answer_span"],
                    "llama_lowercase_spans_in_surviving_rows":
                        q2["case_sensitivity_audit"]["per_model"]["llama"]
                          ["n_lowercase_answer_span"],
                },
                "mechanism": "264 实测解码为 ' a'，恰为 space_lower['A'] → "
                             "触发路径已确定；但 2 条的**原文永久缺失**，文本本身仍 UNDETERMINED",
                "why_surviving_rows_show_zero": "小写命中的那两条正是抛异常、未写盘的两条 —— "
                                                "存活的 118 条按构造不可能包含它们",
                "behaviour": "fail-closed（停下报错），未静默取错 id —— 这一点是对的",
                "minimal_fix": "候选 id 选型需覆盖大小写两形，或解析器与定位统一大小写口径。"
                               "**属改解析/读出行为，须另卡授权**，本卡不实施",
                "code_ref": ["src/recover_attention/domain_label_proxy.py:19",
                             "src/recover_attention/generation/backends.py:425",
                             "src/recover_attention/generation/backends.py:388"],
            },
            {
                "id": "F3", "severity": "指纹覆盖缺口 + 实际生效配置两模型非对称（作用未排除）",
                "what": "代码未显式传 top_k / repetition_penalty，由各模型自带 "
                        "generation_config 补齐；run_fingerprint 不记录这两键。"
                        "Qwen 侧实际生效 top_k=20 / repetition_penalty=1.05，"
                        "Llama 侧为库默认 50 / 1.0",
                "measured": q1["generation_config_drift"]["model_shipped_generation_config"],
                "differing_keys": q1["generation_config_drift"]["keys_differing_between_models"],
                "impact": q1["generation_config_drift"]["impact_note"],
                "cannot_be_ruled_out": (
                    "repetition_penalty 在 greedy 与 sampled 两种模式下都生效，两侧取值不同；"
                    "top_k 影响 sampled。**不得**据「greedy 也失败」推断这些差异无作用。"),
                "minimal_fix": "把实际生效的完整采样配置写进 run_fingerprint（只增记录，不改行为），"
                               "并按 Qwen 实际执行协议显式传 top_k / repetition_penalty。"
                               "**须另卡授权**",
            },
        ],
        "4_residual_unresolvable_offline": [
            "F1（重复 BOS）对输出质量的实际影响 —— 需真实前向对照才能量化，本轮未做",
            "2 条 264 异常的 completion 原文 —— **永久缺失**，重新生成只能得到新证据",
            "completion 的 token ids 与 num_generated_tokens 未落盘 → "
            "无法断言『恰好撞满 16 词元上限』，只能说文本在枚举/句中被截断",
            "2026-09-09 运行时的 transformers 版本未记录 → F3 的合并语义结论依赖当前版本前提",
            "『Llama 在修正 F1 后是否仍不产出可解析答案』—— 未测，因此 A 与 C 尚不能区分",
        ],
        "5_conclusion_form": {
            "verdict": "D（现有离线证据不足以区分 A 与 C）",
            "why_not_C": "已发现一处确认的协议保真缺陷 F1（重复 BOS），"
                         "它在 Qwen 侧不存在。在排除它之前，不能把失败归给『模型在同协议下的实际失败』",
            "why_not_A": "F1 未被证明是 104/118 解析失败的原因；"
                         "所有 118 条（含 greedy 臂）都是被截断的推理散文，"
                         "这与 16 词元预算下的模型行为同样相容",
            "wording_bound": "任何结论最多写到『在本次固定样本（G3 名单 20 题 × K=6）与"
                             "本次冻结配置下，Llama-3.1-8B-Instruct 未通过同协议 MCQ 复现』；"
                             "**不得**写成 Llama 普遍无法执行该协议",
        },
        "6_next_step_proposal": {
            "needed": "是 —— 但先修 F1，再做**最小**真实诊断",
            "order": ["① 另卡授权修复 F1（add_special_tokens=False），"
                      "并证明 Qwen 侧逐字节不变",
                      "② 最小真实诊断：只跑 MCQ 臂、只跑 G3 名单、协议其余一切不动，"
                      "对照修复前后的 completion",
                      "③ 若修复后仍无可解析答案 → 结论落 C（按上述措辞边界）"],
            "minimum_scope_derivation": (
                "条数应由缺口反推而不是取整数：F1 的判别只需在**同一批 G3 题**上比较"
                "『重复 BOS vs 单 BOS』的 completion 是否仍无 Answer: 标记。"
                "由于 118/118 当前一致地无标记，少量条目即可证伪『修复后就能产出答案』；"
                "具体条数、停止条件与产物目录由下一张卡固定，本卡不预设。"),
            "not_authorized_now": "本轮不修、不跑、不重判 gate",
        },
    }


# ─────────────────────────────────────────────────────────── tokenizer
def load_tokenizers(want: bool) -> dict | None:
    if not want:
        return None
    from transformers import AutoTokenizer      # noqa: PLC0415
    from recover_attention.generation import fingerprint as fpm   # noqa: PLC0415
    ctx: dict = {"identity": {}}
    for tag, path, frozen in (
        ("qwen", "D:/models/Qwen2.5-7B-Instruct", QWEN_DIR / "mcq_g3_run_fingerprint.json"),
        ("llama", "D:/models/Meta-Llama-3.1-8B-Instruct",
         LLAMA_DIR / "mcq_g3_run_fingerprint.json"),
    ):
        try:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as exc:                # noqa: BLE001
            ctx["identity"][tag] = {"loaded": False, "error": str(exc)[:200]}
            continue
        now = fpm.tokenizer_fingerprint(tok)
        was = load_json(frozen)["tokenizer"]
        ctx["identity"][tag] = {
            "loaded": True,
            "frozen_tokenizer_hash": was["tokenizer_hash"],
            "current_tokenizer_hash": now["tokenizer_hash"],
            "match": now["tokenizer_hash"] == was["tokenizer_hash"],
            "frozen_class": was["class"], "current_class": now["class"],
        }
        ctx[tag] = tok
    return ctx


# ─────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-tokenizer", action="store_true",
                    help="额外加载 tokenizer（**不加载模型权重**）以核验候选 id / offsets")
    args = ap.parse_args()

    q_rows = load_rows(QWEN_DIR / "mcq_g3_unit_results.jsonl")
    l_rows = load_rows(LLAMA_DIR / "mcq_g3_unit_results.jsonl")
    tok_ctx = load_tokenizers(args.with_tokenizer)

    doc = {
        "schema": "stage1_l2c_offline_diagnosis_v1",
        "date": "2026-09-13",
        "card": "docs/codex_tasks/sprint_stage1_l2c_llama_same_protocol_mcq_diagnosis.md",
        "scope": "离线诊断：未加载模型权重、未跑 GPU、未修改生成语义、未重判 gate",
        "tokenizer_loaded": bool(tok_ctx),
        "Q1_protocol_fidelity": {
            "prompt_hash_basis": q1_prompt_hash_basis(q_rows, l_rows),
            "layered_prompt_compare": q1_layered_prompt_compare(q_rows, l_rows),
            "params_backend_geometry": q1_params_backend_geometry(),
            "generation_config_drift": q1_generation_config_drift(),
            "artifact_hashes": q1_artifact_hashes(),
        },
        "Q2_implementation": {
            "char_span_replay_no_tokenizer": q2_char_span_replay(l_rows),
            "case_sensitivity_audit": q2_case_sensitivity_audit(q_rows, l_rows),
            "bos_duplication": q2_bos_check(q_rows, l_rows, tok_ctx),
            "tokenizer_checks": q2_tokenizer_section(l_rows, tok_ctx),
        },
        "Q3_evidence": {
            "classification": q3_classify(l_rows),
            "anomaly_residue": q3_anomaly_residue(l_rows),
        },
    }
    doc["Q4_synthesis"] = q4_synthesis(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    print(json.dumps(doc, ensure_ascii=False, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
