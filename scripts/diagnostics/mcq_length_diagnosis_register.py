r"""MCQ 长度诊断：**只读**机制核对 + 诊断设置的事前登记。

不加载模型权重。产出 `tmp/stage1_l2_g3/mcq_length_diagnosis_register.json`，
作为"设置在看到结果之前就已固定"的凭据。

用法：
    python scripts/diagnostics/mcq_length_diagnosis_register.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import cyber_data as cd                       # noqa: E402
from recover_attention import domain_label_proxy as dlp              # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool      # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as bk              # noqa: E402

OUT = ROOT / "tmp" / "stage1_l2_g3" / "mcq_length_diagnosis_register.json"
OUT_DIR = ROOT / "outputs" / "stage1_llama_v3" / "mcq_length_diagnosis"
SAMPLES = ROOT / "outputs" / "stage1" / "samples"
MODEL = Path("D:/models/Meta-Llama-3.1-8B-Instruct")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_layer_readout() -> dict:
    """① rel_depth 只控制 hidden 读取，还是也会改变生成？"""
    gen = inspect.getsource(bk._RealBackendBase._generate)
    tf = inspect.getsource(bk._RealBackendBase._teacher_forced)
    verify = inspect.getsource(bk._RealBackendBase.verify_hidden_index)
    whole = inspect.getsource(bk)
    return {
        "tuple_idx_used_in": {
            "_generate": "tuple_idx" in gen,
            "_teacher_forced": "tuple_idx" in tf,
            "verify_hidden_index": "tuple_idx" in verify,
        },
        "_generate_requests_hidden_states": "output_hidden_states" in gen,
        "_generate_registers_hooks": "register_forward_hook" in gen,
        "hook_sites_in_module": whole.count("register_forward_hook"),
        "hook_is_removed_in_verify": "h.remove()" in verify,
        "verify_uses_a_separate_probe_forward": "probe text for layer check" in verify,
        "verdict": (
            "层读取**不参与生成**：_generate 既不请求 hidden_states 也不注册 hook；"
            "tuple_idx 只在 _teacher_forced 里索引 out.hidden_states；"
            "唯一的 forward hook 在 verify_hidden_index 内，用独立探针前向且当场 remove。"
            "→ rel_depth / block / tuple 的取值**不可能**解释本次生成失败。"),
        "explicitly_NOT_claimed": (
            "「读取层位置正确」**不等于**「该层在两个模型上功能等价」。"
            "block 22 对 Llama 而言只是 floor(0.7*32)，"
            "它与 Qwen 的 block 19 是否承载可比的表征，本卡**不作任何断言**。"),
    }


def audit_generate_config() -> dict:
    """② 真实传给 generate 的完整有效配置与停止条件。"""
    import transformers                                   # noqa: PLC0415
    from transformers import GenerationConfig             # noqa: PLC0415
    gen = inspect.getsource(bk._RealBackendBase._generate)
    prep = inspect.getsource(
        transformers.generation.utils.GenerationMixin._prepare_generated_length)
    model_gc = json.loads((MODEL / "generation_config.json").read_text(encoding="utf-8"))
    lib = GenerationConfig()._get_default_generation_params()
    return {
        "explicitly_passed": ["max_new_tokens", "do_sample", "use_cache",
                              "renormalize_logits", "remove_invalid_values",
                              "pad_token_id", "eos_token_id", "repetition_penalty",
                              "temperature(仅 do_sample)", "top_p(仅 do_sample)",
                              "top_k(仅 do_sample)"],
        "max_length": {
            "passed_by_code": "max_length" in gen,
            "hf_behaviour": ("_prepare_generated_length：max_new_tokens 非 None 时 "
                             "max_length = max_new_tokens + input_ids_length"),
            "source_confirms": "generation_config.max_length = generation_config.max_new_tokens + input_ids_length" in prep,
            "verdict": "max_length **不是**混淆项：它被 max_new_tokens 覆盖重算，"
                       "库默认的 20 不会截断长 prompt",
        },
        "eos": {
            "passed": "tok.eos_token_id（标量）",
            "llama_tokenizer_eos": 128009,
            "model_generation_config_eos_list": model_gc.get("eos_token_id"),
            "consequence": ("代码传的是**单个** 128009(<|eot_id|>)，覆盖了模型配置里的 "
                            "[128001,128008,128009]。若模型发出 <|end_of_text|>(128001) "
                            "或 <|eom_id|>(128008)，**不会**触发停止 —— 属待验证机制，"
                            "本轮逐条记录实际末 token 以判定"),
        },
        "pad": {"expr": "tok.pad_token_id or eos",
                "llama_tokenizer_pad": None,
                "effective": "回退为 eos=128009"},
        "other_stopping": {
            "stop_strings": "未传",
            "min_length / min_new_tokens": "未传",
            "num_beams": f"未传 → 库默认 {lib.get('num_beams')}",
            "no_repeat_ngram_size": f"未传 → 库默认 {lib.get('no_repeat_ngram_size')}",
        },
        "library_defaults_for_unpassed": lib,
        "model_generation_config_sha256": sha(MODEL / "generation_config.json"),
        "transformers_version": transformers.__version__,
    }


def audit_prompt_and_ids() -> dict:
    """③ rendered prompt / 输入 token ids / BOS 数量 / chat template。"""
    gen = inspect.getsource(bk._RealBackendBase._generate)
    build = inspect.getsource(g3.build_arm_records)
    return {
        "prompt_source": "v3_g3_smoke.build_arm_records → tokenizer.apply_chat_template("
                         "cd.build_mcq_chat_messages(record), tokenize=False, "
                         "add_generation_prompt=True)",
        "build_arm_records_uses_apply_chat_template": "apply_chat_template" in build,
        "prompt_tokenised_with_add_special_tokens_False": "add_special_tokens=False" in gen,
        "bos_expectation": ("模板自带 1 个 <|begin_of_text|>(128000)；"
                            "add_special_tokens=False 后不再叠加 → 应恰好 1 个"),
        "checked_at_runtime": "本轮逐条记录 prompt_input_ids，实测首 token 与 128000 的出现次数",
        "instruction_lines_frozen": (
            "Think briefly step by step, then answer with exactly one letter as:"
            in inspect.getsource(cd.build_mcq_prompt)),
    }


def audit_slice_decode_parse() -> dict:
    """④⑤ completion 如何切出、解码，解析器如何识别；是否可能漏掉已生成的答案。"""
    gen = inspect.getsource(bk._RealBackendBase._generate)
    return {
        "slice": "new = ids[0, plen:] —— plen 来自**同一次**编码，切分口径自洽",
        "eos_strip": "只剥**末尾一个**且必须恰等于所传 eos；其它位置的特殊 token 不剥",
        "decode": "tok.decode(new, skip_special_tokens=True)",
        "risk_1_special_tokens_hidden": (
            "skip_special_tokens=True 会**隐藏**特殊 token。若答案被特殊 token 包裹、"
            "或中途出现 <|eot_id|> 而未触发停止，正式文本里看不出来 "
            "→ 本轮同时记录 skip_special_tokens=False 的解码"),
        "risk_2_answer_beyond_budget": (
            "若答案出现在预算之外，正式路径**不可能**看到它 —— 这正是本轮 128 条件要测的"),
        "risk_3_parser_locates_wrong_span": (
            "parse_option_answer：explicit_answer_marker → last_isolated_label。"
            "后者会把**末尾孤立字母**判为答案，已知会命中选项枚举标号 "
            "→ 本轮人工审计与解析器结果分开记录，**不替换解析器**"),
        "parser_signature": str(inspect.signature(dlp.parse_option_answer)),
        "explicit_marker_regex": dlp._EXPLICIT_ANSWER_RE.pattern,
        "isolated_letter_regex": dlp._ISOLATED_LETTER_RE.pattern,
    }


def main() -> int:
    cfg = V3AnalysisConfig()
    mcq_list = SAMPLES / "mcq_g3_questions.jsonl"
    rows = [json.loads(x) for x in mcq_list.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    geo = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=str(MODEL))

    import torch, transformers, bitsandbytes, accelerate   # noqa: PLC0415,E401
    doc = {
        "schema": "stage1_mcq_length_diagnosis_register_v1",
        "date": "2026-09-14",
        "card": "docs/codex_tasks/sprint_stage1_llama_mcq_length_diagnosis.md",
        "status": "**事前登记**：本文件在任何诊断生成之前写入，设置不随结果调整",
        "boundary": {
            "is": "观察失败后开展的**探索性诊断**",
            "is_not": ["替换 outputs/stage1_llama_v3/g3_postfix/ 的原结果",
                       "原协议 G3 通过", "事前冻结设置", "G3/G4/生产的输入"],
            "original_verdict_unchanged": ("原 MCQ G3 未通过的结论不变；"
                                           "g3_postfix/ 全部保留，不覆盖"),
            "no_qwen_rerun": True,
            "not_presupposed": ("**不预先认定「16 token 不够」**。旧产物未记录完整终止信息，"
                                "它只是待验证解释之一"),
        },
        "environment": {
            "host": platform.node(),
            "project_root": str(ROOT),
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "gpu": (torch.cuda.get_device_properties(0).name
                    if torch.cuda.is_available() else None),
            "gpu_total_MiB": (torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
                              if torch.cuda.is_available() else None),
            "backend": dict(bk.BACKEND_8BIT),
            "versions": {"torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "bitsandbytes": bitsandbytes.__version__,
                         "accelerate": accelerate.__version__},
        },
        "code_audit": {
            "1_layer_readout_vs_generation": audit_layer_readout(),
            "2_effective_generate_config": audit_generate_config(),
            "3_prompt_and_input_ids": audit_prompt_and_ids(),
            "4_5_slice_decode_parse": audit_slice_decode_parse(),
            "new_blocking_implementation_error_found": False,
            "note": ("若发现足以改变结论的新实现错误，须记录位置/证据/影响面并**停止**"
                     "真实实验，不把实现修复与长度诊断混在同一轮"),
        },
        "preregistered_design": {
            "samples": {
                "list": str(mcq_list.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha(mcq_list),
                "n_questions": len(rows),
                "selection": "原冻结 G3 MCQ 的**同一批 20 题**，不筛题、不另取样本",
            },
            "k_traces": cfg.k_traces,
            "seed_rule": ("沿用 _stable_seed(self.seed=4242, example_id, sample_type, "
                          "sample_index)；与 max_new_tokens 无关，故两条件逐条同 seed"),
            "unit_split": "1 greedy + 5 sampled（units_for，与 G3 同构）",
            "conditions": [
                {"name": "len16", "max_new_tokens": 16, "note": "原长度"},
                {"name": "len128", "max_new_tokens": 128, "note": "长长度"},
            ],
            "everything_else_identical": [
                "模型与 tokenizer", "原提示与 chat template", "原解析器", "8-bit 后端",
                "temperature=0.7", "top_p=0.95", "top_k=20（仅随机采样）",
                "repetition_penalty=1.05（贪心与随机采样）", "层读取位置",
            ],
            "run_order": ["len16（先）", "len128（后）"],
            "max_traces": len(rows) * cfg.k_traces * 2,
            "will_not": ["根据中途结果增加长度", "换题", "调整提示",
                         "把新生成称为旧运行原文的恢复（即使同 seed）"],
            "out_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
            "artifacts_are_not_gate_inputs": True,
        },
        "geometry_readback": {"rel_depth": geo.rel_depth, "block": geo.block_index,
                              "tuple": geo.tuple_index, "hidden_dim": geo.hidden_dim,
                              "source": geo.source,
                              "note": "仅记录；本轮不读 hidden，不缓存 hidden/attention"},
        "observability_change": {
            "file": "src/recover_attention/generation/backends.py",
            "what": "_generate 增加可选参数 capture: dict | None = None（**纯观测**）",
            "behaviour_when_none": "与从前逐字节相同 —— 返回值与所有中间步骤均未改动",
            "why_minimal": "复用既有生成路径，不另写提示或生成逻辑",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a = doc["code_audit"]
    print("① 层读取参与生成？", a["1_layer_readout_vs_generation"]["_generate_requests_hidden_states"],
          "| hook in _generate:", a["1_layer_readout_vs_generation"]["_generate_registers_hooks"])
    print("② max_length 是混淆项？", not a["2_effective_generate_config"]["max_length"]["source_confirms"])
    print("② 传入的 eos:", a["2_effective_generate_config"]["eos"]["llama_tokenizer_eos"],
          "| 模型配置 eos 列表:", a["2_effective_generate_config"]["eos"]["model_generation_config_eos_list"])
    print("③ prompt add_special_tokens=False:",
          a["3_prompt_and_input_ids"]["prompt_tokenised_with_add_special_tokens_False"])
    print("登记样本 sha256:", doc["preregistered_design"]["samples"]["sha256"][:16], "…")
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
