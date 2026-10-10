"""Stage 0 的真实生成后端（H1 / MCQ）。

复用 W0.5-B / W0.5-D 已验证过的代码路径,不另写一套:
  - 8-bit 本地加载（backend invariant 钉死;4-bit 在 H1 长文本上标点退化,4D-1 已验证）
  - completion span 由 **offset** 定位,不用 prompt token 数硬推（BPE 边界会变分词）
  - hidden 取 hidden_states[⌊0.7L⌋+1]（§8;tuple index = block+1）
  - §11 只缓存:pooled hidden、必要 token logprobs、identifier/answer 位置
    **不缓存** raw attention / 全层 hidden / F1 / F4 / J-lens

接口:run_unit(unit, record) -> dict，与 generate_arm 的 resume 循环对接。
每条 unit 独立可重跑 —— 这是 resume 正确性的前提。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention import domain_label_proxy as dlp
from recover_attention import h1_data as hd
from recover_attention import h1_f5_features as f5f
from recover_attention import mcq_conditional_increment as mci
from recover_attention.generation import fingerprint as fp
from recover_attention.h1_identifier import label_completion
from recover_attention.hidden_state_cache import (
    import_torch,
    import_transformers_components,
    infer_model_input_device,
    resolve_torch_dtype,
)

def encode_nonfinite(value: Any) -> Any:
    """把可能非有限的浮点编码成**严格 JSON** 可序列化的结构。

    `json.dumps` 默认会吐出 `NaN` / `Infinity` —— 那不是合法 JSON，下游解析器
    （以及 `allow_nan=False`）会直接拒绝。诊断落盘必须能被任意 JSON 读取器读回，
    所以非有限值改写成 `{value: null, nonfinite_kind: ...}`，有限值原样保留。
    """
    v = float(value)
    if math.isnan(v):
        return {"value": None, "nonfinite_kind": "nan"}
    if math.isinf(v):
        return {"value": None, "nonfinite_kind": "posinf" if v > 0 else "neginf"}
    return v


def sanitize_diagnostic(diagnostic: dict) -> dict:
    """诊断落盘前的净化：**移除 completion 原文**，非有限值结构化。

    completion 是模型输出的正式样本内容，不得进入审计产物；只保留
    char_len / utf8_byte_len / sha256（按原始字符串的 UTF-8 字节计算）。
    """
    out = {k: v for k, v in diagnostic.items() if k != "completion"}
    text = diagnostic.get("completion")
    if text is not None:
        raw = str(text).encode("utf-8")
        out["completion_char_len"] = len(str(text))
        out["completion_utf8_byte_len"] = len(raw)
        out["completion_sha256"] = hashlib.sha256(raw).hexdigest()
    if isinstance(out.get("candidate_logits"), dict):
        out["candidate_logits"] = {k: encode_nonfinite(v)
                                   for k, v in out["candidate_logits"].items()}
    return out


class ForwardNonFiniteError(RuntimeError):
    """teacher-forced forward 在该 step 输出非有限 logits —— 这条 trace 的前向不可信。

    与 letter-token 定位/选形错误(该 raise ValueError,让它进 failed 循环重试、
    持续失败则由人核查)不同:数值层面的事不会因为重跑而消失,必须直接终态排除
    (manifest 的 excluded 状态),不能补 0、不能补 NaN、不能静默跳过。diagnostic
    带全部判据,不许靠 repr(exc)[:300] 猜——那截断过一次真实诊断信息。
    """

    def __init__(self, message: str, *, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


# backend invariant：两臂必须同一后端，否则 §6 的 D = S_MCQ − S_H1 被量化混淆
BACKEND_8BIT = {"load_in_8bit": True, "device_map": "auto",
                "attn_implementation": "eager", "local_files_only": True}


def load_backend(model_path: str) -> dict:
    """8-bit 本地加载。与 4D-1 / W0.5-B / W0.5-D 完全同一条路径。"""
    torch = import_torch()
    comp = import_transformers_components()
    tok = comp["AutoTokenizer"].from_pretrained(model_path, local_files_only=True,
                                                trust_remote_code=False)
    qcfg = comp["BitsAndBytesConfig"](load_in_8bit=True)
    model = comp["AutoModelForCausalLM"].from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        quantization_config=qcfg, device_map="auto", attn_implementation="eager",
        torch_dtype=resolve_torch_dtype("float16", torch))
    model.eval()
    return {"torch": torch, "tokenizer": tok, "model": model, "model_path": model_path,
            "backend": dict(BACKEND_8BIT)}


def _stable_seed(*parts: Any) -> int:
    d = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(d[:16], 16) % (2 ** 31)


class _RealBackendBase:
    """两臂共用的加载、生成与 teacher-forced 前向。"""

    #: 旧 Qwen 运行里**实际生效**的两个采样参数。它们从未由 Python 显式传入,而是由
    #: Qwen2.5-7B-Instruct/generation_config.json 经 transformers 的 generation config
    #: 合并生效(该文件 top_k=20 / repetition_penalty=1.05,2026-04-28 后未改;
    #: backends.py 自首次接入真实后端到 2026-09-08 的五个历史版本都没有覆盖过这两个键)。
    #: Llama 的 generation_config 两个键都没有,落到库默认 50 / 1.0 —— 于是同一套协议
    #: 在两个模型上跑出了不同的采样配方。这里把旧 Qwen 的生效值**显式写死**,
    #: 让它成为协议的一部分而不是某个模型附带文件的副作用。
    #: 这是**沿用既有协议**,不是新增调参;Qwen 侧的有效设置因此不变。
    QWEN_PROTOCOL_TOP_K = 20
    QWEN_PROTOCOL_REPETITION_PENALTY = 1.05

    def __init__(self, model_path: str, *, seed: int = 4242, max_new_tokens: int = 384,
                 temperature: float = 0.7, top_p: float = 0.95,
                 top_k: int = QWEN_PROTOCOL_TOP_K,
                 repetition_penalty: float = QWEN_PROTOCOL_REPETITION_PENALTY,
                 ctx: dict | None = None):
        self.ctx = ctx or load_backend(model_path)
        self.backend = self.ctx["backend"]
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        L = int(self.ctx["model"].config.num_hidden_layers)
        self.hidden_index = ci.resolve_hidden_index(L)
        self.tuple_idx = self.hidden_index["hidden_states_tuple_index"]

    # --------------------------------------------------------------- 指纹
    def run_fingerprint(self, arm: str) -> dict:
        return fp.run_fingerprint(
            model_fp=fp.model_fingerprint(self.ctx["model"], self.ctx["model_path"],
                                          self.backend),
            tokenizer_fp=fp.tokenizer_fingerprint(self.ctx["tokenizer"]),
            arm=arm, hidden_index=self.hidden_index,
            gen_params={"K": 6, "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature, "top_p": self.top_p,
                        "top_k": self.top_k,
                        "repetition_penalty": self.repetition_penalty,
                        "seed": self.seed,
                        # 口径标记 + 来源。旧指纹**没有**这两项 —— 见
                        # v3_g3_smoke.GENERATION_PARAMS_SCHEMA 的说明:缺标记不等于
                        # "与任意取值兼容",跨口径比较一律当场失败。
                        "schema": fp.GENERATION_PARAMS_SCHEMA,
                        "sampling_params_source": (
                            "explicit; top_k / repetition_penalty 取旧 Qwen 运行经 "
                            "generation_config.json 实际生效的 20 / 1.05")})

    def verify_hidden_index(self) -> dict:
        """forward hook 与 hidden_states[tuple_idx] 数值相等（§8 要求，不是只对 shape）。"""
        torch = self.ctx["torch"]
        model = self.ctx["model"]
        block = self.hidden_index["block_index_zero_based"]
        cap: dict[str, Any] = {}
        h = model.model.layers[block].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)
                                            .detach().float().cpu()))
        enc = self._to_device(self.ctx["tokenizer"]("probe text for layer check",
                                                    return_tensors="pt"))
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h.remove()
        hs = out.hidden_states[self.tuple_idx].detach().float().cpu()
        diff = float((cap["h"] - hs).abs().max())
        return {**self.hidden_index, "numeric_match": bool(diff < 1e-3), "max_abs_diff": diff}

    # --------------------------------------------------------------- 工具
    def _to_device(self, enc):
        dev = infer_model_input_device(self.ctx["model"], "auto", self.ctx["torch"])
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}

    def _generate(self, prompt: str, *, do_sample: bool, seed: int,
                  capture: dict[str, Any] | None = None) -> str:
        # `prompt` 一律是**已应用 chat template** 的字符串（H1 走 apply_chat_template，
        # MCQ 用 record["rendered_prompt"]），模板自己负责首尾特殊 token。若再让 tokenizer
        # 按默认 add_special_tokens=True 加一次,Llama 会拿到**重复的 <|begin_of_text|>**
        # (实测 [128000, 128000, ...])；Qwen 的 tokenizer 不前置任何 token,所以同一段代码
        # 在 Qwen 上看不出这个缺陷 —— 这正是它能活到跨模型复现才暴露的原因。
        # 未经模板的普通文本(如 verify_hidden_index 的探针串)不走这里,编码口径不变。
        torch = self.ctx["torch"]; tok = self.ctx["tokenizer"]; model = self.ctx["model"]
        inputs = self._to_device(tok(prompt, return_tensors="pt",
                                     add_special_tokens=False))
        plen = int(inputs["input_ids"].shape[-1])
        eos = getattr(tok, "eos_token_id", None)
        pad = getattr(tok, "pad_token_id", None) or eos
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # repetition_penalty 的 processor 在 transformers 的 `_get_logits_processor` 里是在
        # `if generation_config.do_sample:` **之外**追加的 —— 贪心与随机采样**都生效**,
        # 所以它不在下面的 do_sample 分支里。top_k / top_p / temperature 的 warper 在该
        # 分支**之内**,只对随机采样生效,因此只在 do_sample 时传。
        kw = {**inputs, "max_new_tokens": self.max_new_tokens, "do_sample": do_sample,
              "use_cache": True, "renormalize_logits": True, "remove_invalid_values": True,
              "pad_token_id": pad, "eos_token_id": eos,
              "repetition_penalty": self.repetition_penalty}
        if do_sample:
            kw.update({"temperature": self.temperature, "top_p": self.top_p,
                       "top_k": self.top_k})
        with torch.no_grad():
            ids = model.generate(**kw)
        new = ids[0, plen:].detach().cpu().tolist()
        raw_new = list(new)
        if eos is not None and new and int(new[-1]) == int(eos):
            new = new[:-1]
        text = tok.decode(new, skip_special_tokens=True)
        if capture is not None:
            # **纯观测**：只把这一次调用已经算出来的东西记下来，不改变上面任何一步。
            # 诊断脚本据此判断终止原因；正式路径传 capture=None，行为与从前逐字节相同。
            capture.update({
                "prompt_input_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "prompt_len": plen,
                "effective_generate_kwargs": {k: v for k, v in kw.items()
                                              if k not in ("input_ids", "attention_mask")},
                "eos_token_id": (int(eos) if eos is not None else None),
                "pad_token_id": (int(pad) if pad is not None else None),
                "generated_ids_raw": raw_new,
                "n_generated_tokens": len(raw_new),
                "last_generated_token_id": (int(raw_new[-1]) if raw_new else None),
                "last_token_is_the_eos_we_passed": bool(
                    raw_new and eos is not None and int(raw_new[-1]) == int(eos)),
                "reached_max_new_tokens": len(raw_new) >= int(self.max_new_tokens),
                "completion_ids_after_eos_strip": list(new),
                "decoded_skip_special_true": text,
                "decoded_skip_special_false": tok.decode(raw_new,
                                                         skip_special_tokens=False),
            })
        return text

    def _teacher_forced(self, prompt: str, completion: str) -> dict:
        """一次前向拿齐 completion 的 logprob/entropy/rank + 目标层 hidden。

        completion 单独用 return_offsets_mapping 编码 —— offsets 与 completion 的字符坐标
        严格一致,identifier/answer 的 char span 才能可靠映射到 token。

        prompt 的 add_special_tokens=False 与 `_generate` 同口径(理由见那里)。两处必须
        一致,否则 prompt_len 差一个 token,读取位置整体平移而数值看着仍然正常。
        """
        torch = self.ctx["torch"]; tok = self.ctx["tokenizer"]; model = self.ctx["model"]
        p_ids = tok(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][0].tolist()
        enc = tok(completion, return_offsets_mapping=True, add_special_tokens=False)
        c_ids = enc["input_ids"]
        if not c_ids:
            raise ValueError("empty completion tokenization")
        offsets = [list(o) for o in enc["offset_mapping"]]
        full = torch.tensor([p_ids + c_ids])
        inputs = self._to_device({"input_ids": full,
                                  "attention_mask": torch.ones_like(full)})
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        plen = len(p_ids)
        tlp, tent, trank = {}, {}, {}
        for j, tid in enumerate(c_ids):
            row = lp[plen + j - 1]
            tlp[j] = float(row[tid])
            p = row.exp()
            tent[j] = float(-(p * row).sum())
            trank[j] = int((row > row[tid]).sum()) + 1
        hidden = out.hidden_states[self.tuple_idx][0].detach().float().cpu().numpy()
        return {"prompt_len": plen, "completion_ids": c_ids, "offsets": offsets,
                "token_logprobs": tlp, "token_entropies": tent, "first_token_ranks": trank,
                "hidden": hidden, "logits": logits}


class H1Backend(_RealBackendBase):
    """H1：自由生成 → identifier 标注 → 全 eligible identifier token 的 hidden mean pooling。"""

    arm = "h1"

    def __init__(self, model_path: str, ontology_index, **kw):
        super().__init__(model_path, **kw)
        self.index = ontology_index

    def run_unit(self, unit: dict, record: dict) -> dict:
        prompt = self.ctx["tokenizer"].apply_chat_template(
            hd.build_h1_chat_messages(record), tokenize=False, add_generation_prompt=True)
        seed = _stable_seed(self.seed, record["example_id"], unit["sample_type"],
                            unit["sample_index"])
        completion = self._generate(prompt, do_sample=unit["sample_type"] == "sampled",
                                    seed=seed)
        if not completion.strip():
            raise ValueError("empty completion")
        tf = self._teacher_forced(prompt, completion)

        labels = label_completion(completion, record["question_text"], self.index)
        mentions = []
        for m in labels["mentions"]:
            local = f5f.token_indices_for_char_span(tf["offsets"], int(m["start"]), int(m["end"]))
            mentions.append({**m, "token_indices": local})
        refusal = not any(ci._is_emitted(m) for m in mentions) and _looks_like_refusal(completion)
        lab = ci.completion_label(mentions, refusal=refusal)
        pos = ci.eligible_identifier_token_positions(mentions)

        row: dict[str, Any] = {
            "example_id": record["example_id"],
            # family/route 是 §3 判 population 归属所需(CVE 族按设计不进主 population),
            # 也是 G4 分层与诊断所需。它们来自 prompt 元数据,不是标签。
            "family": record.get("family"),
            "route": record.get("route"),
            "prompt_text": record["question_text"],
            "completion": completion,
            "refusal": refusal,
            "mentions": mentions,
            "id_token_positions_local": pos,
            "id_string_text": ci.id_string_text(mentions),
            **lab,
            **ci.surface_format_features(completion, mentions),
            **fp.unit_fingerprint(prompt=prompt, seed=seed, temperature=self.temperature,
                                  top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                                  sample_type=unit["sample_type"],
                                  sample_index=unit["sample_index"]),
        }
        comp_lps = [tf["token_logprobs"][j] for j in sorted(tf["token_logprobs"])]
        if pos:
            row.update(f5f.mention_logprob_features(tf["token_logprobs"], tf["token_entropies"],
                                                    tf["first_token_ranks"], pos))
            # §8:H = 全部 eligible identifier token 在该层的 mean pooling
            row["hidden"] = ci.pool_hidden_states(
                tf["hidden"], [tf["prompt_len"] + i for i in pos])
            row["hidden_pooled_n_tokens"] = len(set(pos))
        row.update(f5f.sequence_logprob_features(comp_lps, id_token_count=len(pos)))
        row.update(f5f.extract_verbalized_confidence(completion))
        return row


class MCQBackend(_RealBackendBase):
    """MCQ：生成字母 → §7.1 canonicalization → answer-letter token 的 hidden。"""

    arm = "mcq"

    def __init__(self, model_path: str, **kw):
        super().__init__(model_path, **kw)
        # 两套 id 必须都备好,且**按实际发射的 token 形式**选用。裸形 {A:32,…} 与空格形
        # {A:362,…} 在 Qwen 上完全不相交:completion 是 "C" 时发射裸形,是 "Answer: C"
        # 时发射 "ĠC"。读错那套,label margin/entropy 会去读一组没有概率质量的 logit ——
        # 不崩、不缺列、形状检查照过,只是数值无意义。F5 是 O 侧基线,削弱 O 会系统性
        # 抬高 Δ_H,方向正好是 §4 主结论,所以这里宁可 raise 也不猜。
        self.letter_ids_bare = dlp.bare_option_token_ids(self.ctx["tokenizer"],
                                                         list(mci.MCQ_LETTERS))
        self.letter_ids_space = dlp.option_token_ids(self.ctx["tokenizer"],
                                                     list(mci.MCQ_LETTERS))

    def run_unit(self, unit: dict, record: dict) -> dict:
        """生成 → 打分。**生成只发生在这里**；打分全部在 `score_completion` 里。

        拆开的唯一理由：解析规则修订后需要对**已保存的 completion 文本**重新提取
        特征，而那条路径必须与正常生成入口用**同一份实现**。这里保持
        "先 `_generate`、再把结果交给 `score_completion`"，所以两条路走的是同一段代码。
        """
        prompt = record["rendered_prompt"]
        seed = _stable_seed(self.seed, record["example_id"], unit["sample_type"],
                            unit["sample_index"])
        completion = self._generate(prompt, do_sample=unit["sample_type"] == "sampled",
                                    seed=seed)
        return self.score_completion(prompt, completion, record, unit, seed=seed)

    def score_completion(self, prompt: str, completion: str, record: dict, unit: dict,
                         *, seed: int) -> dict:
        """把 (prompt, completion) 打成一行记录：解析 → canonicalization → 表面特征
        → teacher-forced 前向 → F5 与 H。

        **纯函数式**：不调用 `generate`，不采样，输入相同则输出相同。
        `run_unit` 生成完之后调用的就是它；离线特征重提取调用的也是它。
        """
        # 键名是 parsed_label / parse_method / parse_failure（不是 label / method）——
        # 取错键会让 label 恒为 None，每条都变成 parse_failure，而 prompt 完全正常。
        parsed = dlp.parse_option_answer(completion, list(mci.MCQ_LETTERS))
        label = parsed["parsed_label"]
        canon = mci.canonicalize_mcq_output(label, record["candidate_choices"])
        wrong_label = _mcq_wrong_label(label, canon["eligible_for_primary"], record["gold_label"])
        row: dict[str, Any] = {
            "example_id": record["example_id"],
            "rendered_prompt": prompt,
            "raw_completion": completion,
            "parsed_label": label,
            "parse_status": parsed["parse_method"],
            "wrong_label": wrong_label,
            "candidate_choices": record["candidate_choices"],
            "prompt_only_text": mci.prompt_only_text(record["question"],
                                                     record["candidate_choices"]),
            **canon,
            **mci.response_format_of(completion, label),
            **mci.mcq_surface_features(canon["selected_option_text"],
                                       record["candidate_choices"], label),
            **fp.unit_fingerprint(prompt=prompt, seed=seed, temperature=self.temperature,
                                  top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                                  sample_type=unit["sample_type"],
                                  sample_index=unit["sample_index"]),
        }
        if not canon["eligible_for_primary"]:
            return row          # parse failure：不进 population，也就没有 H

        tf = self._teacher_forced(prompt, completion)
        cs, ce = _parsed_letter_char_span(completion, label, parsed)
        local = f5f.token_indices_for_char_span(tf["offsets"], cs, ce)
        if len(local) != 1:
            # §8 说的是"answer-letter token 自身"(单数)。多 token 不临时选一个 —— 停下来核对。
            raise ValueError(
                f"answer letter spans {len(local)} tokens for {record['example_id']}; "
                f"section 8 specifies the answer-letter token itself. Do not pick one. "
                f"completion={completion[:80]!r}")
        j = local[0]
        letter_token_id = int(tf["completion_ids"][j])
        letter_ids, letter_form = self._letter_id_set(letter_token_id, label)
        step = tf["prompt_len"] + j - 1
        row_logits = tf["logits"][step].cpu().numpy()
        opt = {lab: float(row_logits[tid]) for lab, tid in letter_ids.items()}
        # 诊断,不改行为:下游 label_margin 只会说"值不是有限实数",看不出是这 4 个候选
        # 的问题还是整行 logits 坏了(8-bit 下 lm_head 偶发 fp16 溢出)。两者的处置完全
        # 不同 —— 前者是选错 id 集,后者是这条 trace 的前向本身不可信(hidden 也一起废)。
        # 报错当场把判据带出来,否则只能等下一轮 resume 再猜一次。
        if not all(np.isfinite(v) for v in opt.values()):
            hidden_vec = np.asarray(tf["hidden"][tf["prompt_len"] + j], dtype=np.float32)
            row_nonfinite = int(np.count_nonzero(~np.isfinite(row_logits)))
            hidden_nonfinite = int(np.count_nonzero(~np.isfinite(hidden_vec)))
            diagnostic = {
                "reason": "forward_nonfinite",
                "example_id": record["example_id"],
                "group_id": record.get("group_id"),
                "sample_type": unit["sample_type"],
                "sample_index": unit["sample_index"],
                "letter_token_form": letter_form,
                "letter_token_id": letter_token_id,
                "candidate_ids": letter_ids,
                "candidate_logits": opt,
                "row_nonfinite": row_nonfinite,
                "row_size": int(row_logits.size),
                "row_dtype": str(row_logits.dtype),
                "hidden_nonfinite": hidden_nonfinite,
                "hidden_size": int(hidden_vec.size),
                "completion": completion,
            }
            # **原文在此就地净化**：completion 是正式样本内容，不得随异常传播到
            # 任何审计产物或日志；只留 char_len / utf8_byte_len / sha256。
            # 同时把可能非有限的 candidate_logits 结构化，保证严格 JSON 可序列化。
            diagnostic = sanitize_diagnostic(diagnostic)
            raise ForwardNonFiniteError(
                f"non-finite option logits for {record['example_id']} "
                f"(row_nonfinite={row_nonfinite}/{row_logits.size}, "
                f"hidden_nonfinite={hidden_nonfinite}/{hidden_vec.size}, form={letter_form})",
                diagnostic=diagnostic)
        row.update({
            "letter_token_index_local": int(j),
            "letter_token_id": letter_token_id,
            "letter_token_form": letter_form,
            "readout_to_prompt_end_distance": int(step - (tf["prompt_len"] - 1)),
            "f5_label_margin": dlp.label_margin(opt),
            "f5_label_entropy": dlp.label_entropy(opt),
            "f5_full_entropy": dlp.full_entropy(row_logits),
            "f5_letter_token_logprob": tf["token_logprobs"][j],
            "hidden": tf["hidden"][tf["prompt_len"] + j],
        })
        return row

    def _letter_id_set(self, letter_token_id: int, label: str) -> tuple[dict[str, int], str]:
        """选与发射 token 同形的那套 option id（bare / space）。

        既不是裸形也不是空格形 → 说明字母被并进了别的 token（如 "(C" ),此时四个候选
        logit 不在同一可比集合上,margin/entropy 无从解释 → 停,不猜。
        """
        if letter_token_id == self.letter_ids_bare.get(label):
            return self.letter_ids_bare, "bare"
        if letter_token_id == self.letter_ids_space.get(label):
            return self.letter_ids_space, "space"
        raise ValueError(
            f"answer-letter token id {letter_token_id} for label {label!r} matches neither the "
            f"bare form ({self.letter_ids_bare.get(label)}) nor the space form "
            f"({self.letter_ids_space.get(label)}); F5 label margin/entropy would be read off a "
            f"candidate set the model never emitted into.")


def _mcq_wrong_label(parsed_label: str | None, eligible_for_primary: bool,
                     gold_label: str) -> int | None:
    """§7.1/§3 的目标 Y 本身(1 = 解析出的选项 != gold_label),不是判别性度量——与 H1 在
    run_unit 内用 label_completion 计算自己的 label 是同一类事,不受 G4 performance-blind
    约束(该约束挡的是 AUROC/CI/ladder 这类"探针表现如何",不挡"这条样本的真值是什么")。

    gold_label 来自 pool 文件,用的是**同一套 canonical 字母**(已核验:candidate_choices
    里每条 choice 的 label_text 与 gold_label_text 逐条对齐,2000/2000 无一不符),故直接
    按字母比较即可,不需要再走 selected_option_text。

    此前 MCQBackend.run_unit 里从未计算这一列,production 产物写不出可分析的 label——
    G4 quality_check(run_stage0_quality_check.py)早就算过同一个式子(仅用于"两类都有吗"
    这一可用性判断),production 生成路径漏了同一行代码,直到 fresh 1760 第一次真正跑到
    冻结/分析入口(load_arm)才暴露:KeyError('wrong_label')。
    """
    if not eligible_for_primary:
        return None
    return int(parsed_label != gold_label)


def _parsed_letter_char_span(completion: str, parsed_label: str,
                             parsed: dict | None = None) -> tuple[int, int]:
    """答案字母的字符区间 —— **直接取解析器那一次判定给出的区间**。

    2026-09-15 修订：这里原本把解析规则**再实现一遍**，于是"解析选 C、定位取末尾 D"
    这种不一致在结构上就是可能的。现在标签与区间出自同一次 `parse_option_answer`：
    调用方可以把那次的结果用 `parsed=` 传进来（省一次解析），不传则就地重算一次
    **同一个函数**，绝不另写规则。
    """
    label = str(parsed_label).strip().upper()
    parsed = parsed or dlp.parse_option_answer(completion, list(mci.MCQ_LETTERS))
    if parsed.get("parsed_label") != label:
        raise ValueError(
            f"parsed label {label!r} disagrees with the parser's own decision "
            f"{parsed.get('parsed_label')!r} ({parsed.get('parse_method')})")
    span = parsed.get("answer_char_span")
    if not span:
        raise ValueError(f"could not locate parsed answer letter {label!r} in completion")
    return int(span[0]), int(span[1])


def _looks_like_refusal(text: str) -> bool:
    import re
    return bool(re.search(
        r"\b(i\s+cannot|i\s+can't|i\s+do\s+not\s+have|i\s+don't\s+have|no\s+known|"
        r"not\s+aware\s+of|unable\s+to|cannot\s+determine|can't\s+determine|"
        r"insufficient\s+information)\b", text or "", re.I))


def attach_cross_sample_f5(rows: list[dict], arm: str) -> None:
    """F5 的 self-consistency / agreement 跨同一 question 的 K 条 trace 聚合。

    逐条算不出来 —— 必须等一个 question 的全部 trace 生成完。不补的话这两维被静默填 0,
    等于削弱 O 侧基线、人为抬高 Δ_H（H1 侧真实犯过一次）。
    """
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["example_id"], []).append(r)
    for group in by_q.values():
        if arm == "mcq":
            vals = [r.get("parsed_label") for r in group]
        else:
            vals = [next((m["normalized"] for m in r.get("mentions", []) if ci._is_primary(m)),
                         None) for r in group]
        present = [v for v in vals if v]
        cons = f5f.exact_consistency(present)
        for r, own in zip(group, vals):
            r["f5_self_consistency_exact"] = cons["f5_self_consistency_exact"]
            rate = f5f.id_agreement_rate(own, present) if own else None
            r["f5_letter_agreement_rate" if arm == "mcq" else "f5_id_agreement_rate"] = rate
