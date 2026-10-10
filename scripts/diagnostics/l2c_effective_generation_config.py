r"""实际生效的生成配置差异表（离线，**不加载模型权重**）。

对 greedy 与 sampled 两种 unit 分别列出每个参数的实际来源：

    代码显式传入  → `_generate` 的 kw 字典（唯一权威来源是源码本身）
    模型配置      → 各模型目录下的 generation_config.json
    库默认        → transformers 的 GenerationConfig 默认值
    历史证据      → 已落盘的 run_fingerprint / unit 记录 / run log 里能查到什么

⚠️ **当前磁盘上的 generation_config.json 不等于历史运行时加载的那一份。**
本脚本记录它的 sha256，但**不把它当作已证明的历史配置**；凡是历史产物里查不到的项，
一律标 `historical_evidence: none`。

用法：
    python scripts/diagnostics/l2c_effective_generation_config.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "tmp" / "stage1_l2_g3" / "l2c_effective_generation_config.json"
MODELS = {"qwen": "D:/models/Qwen2.5-7B-Instruct",
          "llama": "D:/models/Meta-Llama-3.1-8B-Instruct"}
FP = {
    ("qwen", "mcq"): ROOT / "outputs/stage1/g3/mcq_g3_run_fingerprint.json",
    ("qwen", "h1"): ROOT / "outputs/stage1/g3/h1_g3_run_fingerprint.json",
    ("llama", "mcq"): ROOT / "outputs/stage1_llama_v3/g3/mcq_g3_run_fingerprint.json",
    ("llama", "h1"): ROOT / "outputs/stage1_llama_v3/g3/h1_g3_run_fingerprint.json",
}

# `_generate` 实际构造的 kw（backends.py:159-175）。greedy 走 do_sample=False 分支，
# 此时 temperature / top_p **不会**被传入 —— 这是差异表里最关键的一格。
CODE_KW_ALWAYS = ["max_new_tokens", "do_sample", "use_cache", "renormalize_logits",
                  "remove_invalid_values", "pad_token_id", "eos_token_id"]
CODE_KW_SAMPLED_ONLY = ["temperature", "top_p"]
WATCH = ["do_sample", "max_new_tokens", "temperature", "top_p", "top_k",
         "repetition_penalty", "eos_token_id", "pad_token_id", "bos_token_id",
         "num_beams", "length_penalty", "no_repeat_ngram_size",
         "renormalize_logits", "remove_invalid_values", "use_cache"]


def sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_json(p: Path) -> dict:
    return json.load(io.open(p, encoding="utf-8"))


def library_defaults() -> dict:
    """transformers 5.x 的 GenerationConfig() 属性多为 None，真实兜底值在

    `_get_default_generation_params()` 里（`_prepare_generation_config` 第二次
    `update(..., defaults_only=True)` 用的就是它）。取这一份才是"库默认"。
    """
    from transformers import GenerationConfig      # noqa: PLC0415
    gc = GenerationConfig()
    resolved = gc._get_default_generation_params()
    return {k: resolved.get(k, getattr(gc, k, None)) for k in WATCH}


def model_config(path: str) -> dict:
    """只读 generation_config.json —— 不实例化模型，不加载权重。"""
    p = Path(path) / "generation_config.json"
    raw = load_json(p) if p.exists() else {}
    return {"path": str(p), "sha256": sha_file(p), "raw": raw,
            "watched": {k: raw.get(k) for k in WATCH if k in raw}}


def historical_evidence() -> dict:
    """历史产物里**实际记录了什么**。没记录的就是没证据，不许用当前配置补。"""
    ev: dict = {}
    for (tag, arm), p in FP.items():
        d = load_json(p)
        ev[f"{tag}_{arm}"] = {
            "run_fingerprint_generation_params": d["generation_params"],
            "tokenizer_special_token_ids": d["tokenizer"]["special_token_ids"],
            "file_sha256": sha_file(p),
        }
    row = json.loads(io.open(
        ROOT / "outputs/stage1_llama_v3/g3/mcq_g3_unit_results.jsonl",
        encoding="utf-8").readline())
    ev["unit_record_fields"] = {k: row.get(k) for k in
                                ("seed", "temperature", "top_p", "max_new_tokens",
                                 "sample_type", "sample_index")}
    log = io.open(ROOT / "outputs/stage1_llama_v3/g3/mcq_run_log.txt",
                  encoding="utf-8", errors="replace").read()
    ev["run_log"] = {
        "records_transformers_version": "transformers" in log and "version" in log.lower(),
        "records_effective_generation_config": False,
        "note": "run log 只有几何 / gate / 计时 / 进度行；"
                "**没有**软件版本，也**没有**实际生效的 generation 配置",
    }
    ev["what_history_does_NOT_record"] = [
        "top_k", "repetition_penalty", "num_beams", "renormalize_logits",
        "remove_invalid_values", "transformers / torch / bitsandbytes 版本",
        "greedy 臂是否收到 temperature/top_p（代码上不传，但历史产物无法独立佐证）",
    ]
    return ev


# 哪些参数在 greedy 下**真正影响输出** —— 直接从安装版 transformers 的
# `GenerationMixin._get_logits_processor` 源码读出，不凭记忆：
#   repetition_penalty 的 processor 在 `if generation_config.do_sample:` **之外**追加
#       → greedy 也生效
#   temperature / top_k / top_p 的 warper 在该分支**之内**
#       → 仅 sampling 生效，greedy 下取值无关
AFFECTS_GREEDY = {"max_new_tokens", "do_sample", "repetition_penalty", "eos_token_id",
                  "pad_token_id", "num_beams", "no_repeat_ngram_size",
                  "remove_invalid_values", "use_cache", "length_penalty"}

HIST = {  # 历史产物里**确有记录**的项（run_fingerprint.generation_params / unit 行）
    "max_new_tokens": "run_fingerprint.generation_params 与 unit 行均记录",
    "temperature": "run_fingerprint 与 unit 行记录 0.7（但未记录 greedy 是否收到）",
    "top_p": "run_fingerprint 与 unit 行记录 0.95（同上）",
    "do_sample": "unit 行的 sample_type 可区分 greedy / sampled",
    "eos_token_id": "run_fingerprint.tokenizer.special_token_ids 记录",
    "pad_token_id": "run_fingerprint.tokenizer.special_token_ids 记录",
    "bos_token_id": "run_fingerprint.tokenizer.special_token_ids 记录",
}

TOKEN_ID_KEYS = {"bos_token_id", "eos_token_id", "pad_token_id"}

ALIGN = {
    "top_k": "对齐到 **20**（Qwen 实际执行协议的取值），并在 _generate 里显式传入；"
             "仅影响 sampled 臂",
    "repetition_penalty": "对齐到 **1.05**（Qwen 实际执行协议的取值），显式传入；"
                          "**greedy 与 sampled 都受影响**",
    "temperature": "greedy 下不生效，无需对齐；sampled 已由代码显式传 0.7，已一致",
    "top_p": "greedy 下不生效，无需对齐；sampled 已由代码显式传 0.95，已一致",
    "bos_token_id": "**模型专属 token id**，按各自 tokenizer 取，不对齐",
    "eos_token_id": "**模型专属 token id**，按各自 tokenizer 取，不对齐",
    "pad_token_id": "**模型专属 token id**，按各自 tokenizer 取，不对齐",
}


def build_rows(defaults: dict, cfgs: dict) -> list[dict]:
    rows = []
    for key in WATCH:
        for mode in ("greedy", "sampled"):
            explicit = (key in CODE_KW_ALWAYS
                        or (mode == "sampled" and key in CODE_KW_SAMPLED_ONLY))
            eff = {}
            for tag in MODELS:
                if explicit:
                    src, val = "code", "见 code_value"
                elif key in cfgs[tag]["watched"]:
                    src, val = "model_generation_config", cfgs[tag]["watched"][key]
                else:
                    src, val = "library_default", defaults.get(key)
                eff[tag] = {"source": src, "value": val}
            same = (eff["qwen"]["source"] == eff["llama"]["source"]
                    and eff["qwen"]["value"] == eff["llama"]["value"])
            affects = key in AFFECTS_GREEDY if mode == "greedy" else True
            rows.append({
                "param": key, "mode": mode,
                "passed_explicitly_by_code": explicit,
                "effective": eff,
                "qwen_llama_consistent": same,
                "affects_output_in_this_mode": affects,
                "historical_evidence": HIST.get(key, "none —— 历史产物未记录此项"),
                "proposed_alignment": ALIGN.get(
                    key, "无需对齐（已由代码显式传入且两侧一致）" if explicit else "待定"),
                # 模型专属 token id 两侧本来就该不同 —— 那是模型固有差异，不是协议偏离
                "model_specific_token_id": key in TOKEN_ID_KEYS,
                "action_required": bool(not same and affects
                                        and key not in TOKEN_ID_KEYS),
            })
    return rows


def main() -> int:
    defaults = library_defaults()
    cfgs = {tag: model_config(path) for tag, path in MODELS.items()}
    rows = build_rows(defaults, cfgs)
    inconsistent = [(r["param"], r["mode"]) for r in rows if not r["qwen_llama_consistent"]]
    action = [(r["param"], r["mode"]) for r in rows if r["action_required"]]

    import transformers                              # noqa: PLC0415
    doc = {
        "schema": "stage1_l2c_effective_generation_config_v1",
        "date": "2026-09-13",
        "scope": "离线核查：不加载模型权重，不跑 GPU，**本轮不修改任何生成参数**",
        "code_value": {
            "max_new_tokens": "MCQ 16 / H1 384（两侧 run_fingerprint 记录一致）",
            "do_sample": "greedy False / sampled True",
            "use_cache": True, "renormalize_logits": True, "remove_invalid_values": True,
            "temperature": "0.7（**仅 sampled 传**）", "top_p": "0.95（**仅 sampled 传**）",
            "eos_token_id": "tok.eos_token_id（标量；覆盖模型配置里的列表）",
            "pad_token_id": "tok.pad_token_id or eos",
            "code_ref": "src/recover_attention/generation/backends.py::_generate",
        },
        "library_defaults": defaults,
        "transformers_version_now": transformers.__version__,
        "model_generation_config": cfgs,
        "historical_evidence": historical_evidence(),
        "rows": rows,
        "inconsistent_param_modes": inconsistent,
        "action_required_param_modes": action,
        "processor_gating_verified_from_installed_source": (
            "GenerationMixin._get_logits_processor：repetition_penalty 的 processor 在 "
            " **之外**追加（greedy 亦生效）；"
            "temperature / top_k / top_p 的 warper 在该分支**之内**（仅 sampling 生效）"),
        "evidence_limits": [
            "磁盘上的 generation_config.json 只能证明**现在**是什么，"
            "不能证明 2026-08-08(Qwen) / 2026-09-09(Llama) 运行时加载的是同一份",
            "run_fingerprint 只记录 K / max_new_tokens / temperature / top_p / seed，"
            "不记录 top_k / repetition_penalty / 库版本",
            "因此凡是 source 为 model_generation_config 或 library_default 的行，"
            "其历史取值一律为**推断**，不是实测",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("inconsistent (param, mode):")
    for p, m in inconsistent:
        print("  ", p, m)
    print("ACTION REQUIRED (不一致且在该模式下确实影响输出):")
    for p, m in action:
        print("  ", p, m)
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
