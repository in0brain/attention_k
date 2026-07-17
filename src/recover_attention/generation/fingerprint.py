"""生成记录的指纹。

reviewer 最容易问的一句是:**你的 hidden state 来自哪个模型状态?**
没有指纹就答不上来,而且换了权重/分词器/prompt 之后旧 cache 会被静默复用 —— 那比答不上来更糟。

每条 record 带 unit 级指纹（prompt_hash / seed / 温度 / max_new_tokens）;
每次 run 带 run 级指纹（model / backend / tokenizer / config）。
两者都进 feature cache,分析入口据此校验一致性。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

FINGERPRINT_SCHEMA = "stage0_fingerprint_v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """对可 JSON 化对象的确定性 hash（键排序，浮点按 repr）。"""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr).encode("utf-8")
    ).hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> dict:
    """分词器指纹。

    只记 name/class/vocab_size 不够:同名分词器可能 special token 不同。故把 special token
    的 id 一并纳入 —— 它们直接决定 prompt 的 token 边界,而边界决定 hidden 的取值位置。
    """
    specials = {}
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        specials[attr] = getattr(tokenizer, attr, None)
    payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "special_token_ids": specials,
    }
    return {**payload, "tokenizer_hash": stable_hash(payload)}


def model_fingerprint(model: Any, model_path: str, backend: dict) -> dict:
    cfg = getattr(model, "config", None)
    payload = {
        "model_path": str(model_path),
        "model_type": getattr(cfg, "model_type", None),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
        "hidden_size": int(getattr(cfg, "hidden_size", -1)),
        "vocab_size": int(getattr(cfg, "vocab_size", -1)),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
        "backend": dict(backend),
    }
    return {**payload, "model_hash": stable_hash(payload)}


def run_fingerprint(*, model_fp: dict, tokenizer_fp: dict, arm: str,
                    hidden_index: dict, gen_params: dict) -> dict:
    """一次 production run 的完整指纹。写进 feature cache 的头部。"""
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "arm": arm,
        "model": model_fp,
        "tokenizer": tokenizer_fp,
        "hidden_index": hidden_index,
        "generation_params": gen_params,
    }
    return {**payload, "run_hash": stable_hash(payload)}


def unit_fingerprint(*, prompt: str, seed: int, temperature: float, top_p: float,
                     max_new_tokens: int, sample_type: str, sample_index: int) -> dict:
    """一条生成 unit 的指纹。prompt_hash 让"prompt 悄悄变了"无所遁形。"""
    payload = {
        "prompt_hash": sha256_text(prompt),
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "sample_type": sample_type,
        "sample_index": int(sample_index),
    }
    return {**payload, "unit_hash": stable_hash(payload)}


def assert_run_fingerprint_matches(cache_fp: dict, current_fp: dict) -> None:
    """复用旧 cache 前必须校验:模型/分词器/层位一变,旧 hidden 就不可比。

    静默复用是最难查的错:数字都在、看着正常,但来自不同的模型状态。
    """
    for key in ("model", "tokenizer", "hidden_index"):
        a, b = cache_fp.get(key), current_fp.get(key)
        if a != b:
            raise ValueError(
                f"fingerprint mismatch on {key!r}: the cache was produced under a different "
                f"model/tokenizer/layer state and its hidden vectors are not comparable.\n"
                f"  cache  : {json.dumps(a, ensure_ascii=False, default=repr)[:300]}\n"
                f"  current: {json.dumps(b, ensure_ascii=False, default=repr)[:300]}")
