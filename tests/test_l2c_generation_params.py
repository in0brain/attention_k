r"""Stage1-L2C：显式传入 top_k=20 / repetition_penalty=1.05 的定向验证。

**假 tokenizer + 假模型 + 只读模型配置文件；不加载权重、不跑 GPU。**

沿用的是**旧 Qwen 运行里已经生效**的两个值（经 generation_config.json 合并生效），
只是从前没有由 Python 显式传入、也没有进指纹。这里钉住四件事：

    1. 两臂（H1 / MCQ）的 generate 调用参数一致
    2. repetition_penalty 贪心与随机采样都传；top_k 只在随机采样传
    3. 对旧 Qwen 配置而言，显式赋值不改变**合并后的有效生成设置**
       —— 这是配置层面的等价，**不是**"真实输出逐字节一致"
    4. 指纹记录了这两个参数，且新旧口径混用会被当场抓住
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.generation import backends as bk  # noqa: E402
from recover_attention.generation import fingerprint as fpm  # noqa: E402

QWEN_CONFIG = Path("D:/models/Qwen2.5-7B-Instruct/generation_config.json")
PROMPT = "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"


class RecordingTokenizer:
    def __init__(self):
        self.eos_token_id = 7
        self.pad_token_id = 3

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        ids = [(ord(c) % 90) + 10 for c in text]
        out = {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [[i, i + 1] for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(list(ids))


class RecordingModel:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kw):
        self.calls.append({k: v for k, v in kw.items() if k != "input_ids"})
        ids = kw["input_ids"]
        return torch.cat([ids, torch.tensor([[11, 12]], dtype=ids.dtype)], dim=-1)


def _backend(cls=bk._RealBackendBase, **kw):
    be = cls.__new__(cls)
    be.ctx = {"torch": torch, "tokenizer": RecordingTokenizer(), "model": RecordingModel()}
    be.seed = 4242
    be.max_new_tokens = kw.get("max_new_tokens", 16)
    be.temperature = 0.7
    be.top_p = 0.95
    be.top_k = kw.get("top_k", bk._RealBackendBase.QWEN_PROTOCOL_TOP_K)
    be.repetition_penalty = kw.get(
        "repetition_penalty", bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY)
    be.tuple_idx = 2
    be._to_device = lambda enc: enc
    return be


# ───────────────────────────────────────── 1 / 2：实际 generate 调用参数
def test_protocol_values_come_from_the_old_qwen_run_not_a_new_choice():
    assert bk._RealBackendBase.QWEN_PROTOCOL_TOP_K == 20
    assert bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY == 1.05


def test_greedy_gets_repetition_penalty_but_not_top_k():
    be = _backend()
    be._generate(PROMPT, do_sample=False, seed=1)
    kw = be.ctx["model"].calls[-1]
    assert kw["do_sample"] is False
    assert kw["repetition_penalty"] == 1.05, "repetition_penalty 贪心也生效，必须传"
    assert "top_k" not in kw, "top_k 只用于随机采样，贪心不传"
    assert "temperature" not in kw and "top_p" not in kw


def test_sampled_gets_both_plus_temperature_and_top_p():
    be = _backend()
    be._generate(PROMPT, do_sample=True, seed=1)
    kw = be.ctx["model"].calls[-1]
    assert kw["do_sample"] is True
    assert kw["repetition_penalty"] == 1.05
    assert kw["top_k"] == 20
    assert kw["temperature"] == 0.7 and kw["top_p"] == 0.95


def test_both_arms_use_the_same_generate_kwargs():
    """H1 与 MCQ 共用 _RealBackendBase，两臂口径必须逐键一致。"""
    seen = {}
    for arm, cls in (("h1", bk.H1Backend), ("mcq", bk.MCQBackend)):
        be = _backend(cls)
        be._generate(PROMPT, do_sample=True, seed=1)
        be._generate(PROMPT, do_sample=False, seed=1)
        seen[arm] = [{k: v for k, v in c.items() if k != "attention_mask"}
                     for c in be.ctx["model"].calls]
    assert seen["h1"] == seen["mcq"]
    assert bk.H1Backend._generate is bk._RealBackendBase._generate
    assert bk.MCQBackend._generate is bk._RealBackendBase._generate


def test_max_new_tokens_stays_arm_specific():
    """两臂共用采样口径，但解码预算按冻结设计分臂 —— 本次改动不碰它。"""
    assert _backend(max_new_tokens=384).max_new_tokens == 384
    assert _backend(max_new_tokens=16).max_new_tokens == 16


# ───────────────────────────────────────── 3：对旧 Qwen 的有效设置没有改变
@pytest.mark.skipif(not QWEN_CONFIG.exists(), reason="本机没有 Qwen 模型目录")
def test_explicit_values_match_what_old_qwen_runs_already_had_in_effect():
    """只读 generation_config.json（**不加载权重**）。

    旧代码不传这两个键 → transformers 用模型自带 generation_config 兜底；
    新代码显式传 20 / 1.05。二者对 Qwen 而言是**同一组有效设置**。

    ⚠️ 这是**合并后配置层面**的等价，**不是**"真实输出逐字节一致"——
    后者需要真实前向才能声称，本测试不作此断言。
    """
    cfg = json.loads(QWEN_CONFIG.read_text(encoding="utf-8"))
    assert cfg["top_k"] == bk._RealBackendBase.QWEN_PROTOCOL_TOP_K
    assert cfg["repetition_penalty"] == bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY


@pytest.mark.skipif(not QWEN_CONFIG.exists(), reason="本机没有 Qwen 模型目录")
def test_top_p_was_already_overridden_by_python_and_stays_overridden():
    """top_p 是反例：模型配置写 0.8，而旧 Python 显式覆盖成 0.95 —— 沿用 0.95。"""
    cfg = json.loads(QWEN_CONFIG.read_text(encoding="utf-8"))
    assert cfg["top_p"] == 0.8
    be = _backend()
    be._generate(PROMPT, do_sample=True, seed=1)
    assert be.ctx["model"].calls[-1]["top_p"] == 0.95


# ───────────────────────────────────────── 4：指纹与新旧口径混用
def _fp(**gen):
    base = {"K": 6, "max_new_tokens": 16, "temperature": 0.7, "top_p": 0.95,
            "seed": 4242, "schema": fpm.GENERATION_PARAMS_SCHEMA,
            "top_k": 20, "repetition_penalty": 1.05}
    base.update(gen)
    return {"model": {"m": 1}, "tokenizer": {"t": 1}, "hidden_index": {"h": 1},
            "generation_params": base}


def test_fingerprint_records_the_two_params_and_their_source():
    be = _backend()
    be.hidden_index = {"num_layers": 28, "rel_depth": 0.7,
                       "block_index_zero_based": 19, "hidden_states_tuple_index": 20}
    be.ctx["model_path"] = "<test>"
    be.backend = dict(bk.BACKEND_8BIT)
    be.ctx["model"].config = type("C", (), {
        "num_hidden_layers": 28, "hidden_size": 3584, "vocab_size": 10,
        "model_type": "qwen2", "torch_dtype": "torch.float16"})()
    gen = be.run_fingerprint("mcq")["generation_params"]
    assert gen["top_k"] == 20 and gen["repetition_penalty"] == 1.05
    assert gen["schema"] == fpm.GENERATION_PARAMS_SCHEMA
    assert "20 / 1.05" in gen["sampling_params_source"]


def test_same_schema_fingerprints_compare_normally():
    assert g3.generation_params_schema_mismatch(_fp(), _fp()) == []


def test_differing_recorded_sampling_params_are_caught():
    a, b = _fp(), _fp(top_k=50)
    assert g3.generation_params_schema_mismatch(a, b) == []      # 口径相同
    assert g3._all_schema_tagged(a, b)
    assert a["generation_params"]["top_k"] != b["generation_params"]["top_k"]


def test_mixing_old_and_new_fingerprints_fails_loudly():
    """旧指纹缺字段**不是**"与任意取值兼容"。"""
    old = _fp()
    old["generation_params"].pop("schema")
    old["generation_params"].pop("top_k")
    old["generation_params"].pop("repetition_penalty")
    issues = g3.generation_params_schema_mismatch(_fp(), old)
    assert issues and "新旧口径混用" in issues[0]
    assert not g3._all_schema_tagged(_fp(), old)


def test_two_legacy_fingerprints_are_not_claimed_to_have_been_compared():
    """双方都是旧口径：允许比较其它键，但**不声称**比过这两个采样参数。"""
    old = _fp()
    for k in ("schema", "top_k", "repetition_penalty"):
        old["generation_params"].pop(k)
    other = json.loads(json.dumps(old))
    assert g3.generation_params_schema_mismatch(old, other) == []
    assert not g3._all_schema_tagged(old, other)


def _runs(h1_fp, mcq_fp):
    return {g3.ARM_H1: type("R", (), {"fingerprint": h1_fp})(),
            g3.ARM_MCQ: type("R", (), {"fingerprint": mcq_fp})()}


def test_backend_invariant_flags_a_mixed_pair():
    mcq = _fp(max_new_tokens=16)
    mcq["generation_params"].pop("schema")
    out = g3.backend_invariant(_runs(_fp(max_new_tokens=384), mcq))
    assert not out["ok"]
    assert any("generation_params.schema" in m for m in out["mismatched_keys"])
    assert out["recorded_sampling_params"]["compared"] is False


def test_backend_invariant_passes_when_both_arms_are_on_the_new_schema():
    out = g3.backend_invariant(_runs(_fp(max_new_tokens=384), _fp(max_new_tokens=16)))
    assert out["ok"], out["mismatched_keys"]
    assert out["recorded_sampling_params"]["compared"] is True
    assert out["recorded_sampling_params"]["params"] == ["top_k", "repetition_penalty"]


def test_backend_invariant_catches_arms_with_different_sampling_params():
    out = g3.backend_invariant(_runs(_fp(max_new_tokens=384),
                                     _fp(max_new_tokens=16, repetition_penalty=1.0)))
    assert not out["ok"]
    assert "generation_params.repetition_penalty" in out["mismatched_keys"]


# ───────────────────────────────────────── BOS 修复未回归
def test_bos_fix_is_still_in_place_for_both_paths():
    class Tok(RecordingTokenizer):
        def __init__(self):
            super().__init__()
            self.flags: list[bool] = []

        def __call__(self, text, return_tensors=None, add_special_tokens=True,
                     return_offsets_mapping=False):
            if text == PROMPT:
                self.flags.append(add_special_tokens)
            return super().__call__(text, return_tensors, add_special_tokens,
                                    return_offsets_mapping)

    be = _backend()
    be.ctx["tokenizer"] = Tok()

    class M(RecordingModel):
        def __call__(self, **kw):
            total = int(kw["input_ids"].shape[-1])
            return type("O", (), {
                "logits": torch.zeros(1, total, 256),
                "hidden_states": tuple(torch.zeros(1, total, 4) for _ in range(3))})()

    be.ctx["model"] = M()
    be._generate(PROMPT, do_sample=False, seed=1)
    be._teacher_forced(PROMPT, "ab")
    assert be.ctx["tokenizer"].flags == [False, False]
