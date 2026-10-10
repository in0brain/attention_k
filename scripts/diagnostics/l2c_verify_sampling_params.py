r"""显式传入 top_k / repetition_penalty 后，两个模型的**有效生成设置**对照。

只读各模型目录下的 `generation_config.json`，按 transformers 的
`GenerationMixin._prepare_generation_config` 合并语义重算：

    base = 模型自带 generation_config  →  再被 generate(**kwargs) 覆盖

对 greedy 与 sampled 两种 unit 分别给出「旧代码」与「新代码」的合并结果，
并只比较**真正驱动 logits processor 的那些键**。

⚠️ 这是**合并后配置层面**的等价判断，**不是**「真实输出逐字节一致」。
后者需要真实前向才能声称，本脚本不加载权重、不作此断言。

用法：
    python scripts/diagnostics/l2c_verify_sampling_params.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.generation import backends as bk      # noqa: E402

OUT = ROOT / "tmp" / "stage1_l2_g3" / "l2c_sampling_params_verification.json"
MODELS = {"qwen": "D:/models/Qwen2.5-7B-Instruct",
          "llama": "D:/models/Meta-Llama-3.1-8B-Instruct"}

TOP_K = bk._RealBackendBase.QWEN_PROTOCOL_TOP_K
REP_PEN = bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY

#: 只有这些键真正进 `_get_logits_processor` / 采样路径。
#: repetition_penalty 在 `if do_sample:` **之外**追加 → 贪心也生效；
#: temperature / top_k / top_p 在该分支**之内** → 只对随机采样生效。
GREEDY_EFFECTIVE = ("repetition_penalty", "max_new_tokens", "do_sample",
                    "no_repeat_ngram_size", "num_beams")
SAMPLED_EFFECTIVE = GREEDY_EFFECTIVE + ("temperature", "top_p", "top_k")


def kwargs_old(do_sample: bool) -> dict:
    """2026-09-13 之前 `_generate` 实际构造的 kw（只列与本次对照相关的键）。"""
    kw = {"max_new_tokens": 16, "do_sample": do_sample}
    if do_sample:
        kw.update({"temperature": 0.7, "top_p": 0.95})
    return kw


def kwargs_new(do_sample: bool) -> dict:
    """现在的 `_generate`：repetition_penalty always；top_k 仅 sampled。"""
    kw = {"max_new_tokens": 16, "do_sample": do_sample,
          "repetition_penalty": REP_PEN}
    if do_sample:
        kw.update({"temperature": 0.7, "top_p": 0.95, "top_k": TOP_K})
    return kw


def merged(model_cfg: dict, lib_defaults: dict, kw: dict, keys) -> dict:
    """kwargs > 模型 generation_config > 库默认。"""
    out = {}
    for k in keys:
        if k in kw:
            out[k] = kw[k]
        elif k in model_cfg and model_cfg[k] is not None:
            out[k] = model_cfg[k]
        else:
            out[k] = lib_defaults.get(k)
    return out


def main() -> int:
    from transformers import GenerationConfig      # noqa: PLC0415
    import transformers                            # noqa: PLC0415

    lib = GenerationConfig()._get_default_generation_params()
    doc: dict = {
        "schema": "stage1_l2c_sampling_params_verification_v1",
        "date": "2026-09-13",
        "what": "显式传入 top_k / repetition_penalty 后的有效生成设置对照",
        "explicit_values": {"top_k": TOP_K, "repetition_penalty": REP_PEN},
        "provenance": (
            "旧 Qwen 运行里这两个键从未由 Python 传入，由 "
            "Qwen2.5-7B-Instruct/generation_config.json 合并生效（20 / 1.05）。"
            "现在把它显式写死，使其成为协议的一部分。**沿用既有协议，不是新增调参。**"),
        "merge_semantics": ("GenerationMixin._prepare_generation_config："
                            "kwargs > 模型 generation_config > 库默认"),
        "processor_gating": (
            "repetition_penalty 的 processor 在 `if generation_config.do_sample:` 之外 "
            "→ 贪心也生效；temperature / top_k / top_p 的 warper 在该分支之内 "
            "→ 只对随机采样生效"),
        "transformers_version_now": transformers.__version__,
        "NOT_claimed": ("这是**合并后配置层面**的等价，**不是**真实输出逐字节一致；"
                        "本脚本不加载权重、不跑前向"),
        "models": {},
    }
    all_ok = True
    for tag, path in MODELS.items():
        p = Path(path) / "generation_config.json"
        cfg = json.loads(io.open(p, encoding="utf-8").read()) if p.exists() else {}
        modes = {}
        for mode, do_sample, keys in (("greedy", False, GREEDY_EFFECTIVE),
                                      ("sampled", True, SAMPLED_EFFECTIVE)):
            before = merged(cfg, lib, kwargs_old(do_sample), keys)
            after = merged(cfg, lib, kwargs_new(do_sample), keys)
            changed = sorted(k for k in keys if before[k] != after[k])
            modes[mode] = {"effective_before": before, "effective_after": after,
                           "changed_keys": changed, "unchanged": not changed}
        expect_unchanged = tag == "qwen"
        ok = all(m["unchanged"] for m in modes.values()) == expect_unchanged
        all_ok = all_ok and ok
        doc["models"][tag] = {
            "generation_config_path": str(p),
            "generation_config_sha256": (hashlib.sha256(p.read_bytes()).hexdigest()
                                         if p.exists() else None),
            "expectation": ("有效设置**不变**（这两个值本来就在生效）" if expect_unchanged
                            else "有效设置**改变**（对齐到旧 Qwen 的执行协议）"),
            "expectation_met": ok,
            "modes": modes,
        }
    doc["all_expectations_met"] = all_ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for tag, m in doc["models"].items():
        print(f"{tag}: expectation_met={m['expectation_met']}  "
              f"greedy_changed={m['modes']['greedy']['changed_keys']}  "
              f"sampled_changed={m['modes']['sampled']['changed_keys']}")
    print("written:", OUT)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
