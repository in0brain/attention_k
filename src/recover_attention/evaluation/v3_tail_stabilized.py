"""tail-stabilized Route-B 正式路径的**纯核心** —— performance-blind，不解盲。

可行性硬闸（`scripts/probe_tail_stabilized_feasibility.py`）已实测确认：

    逐层扫描  32/32 条已知坏 trace 的首次非有限**都且仅在** tuple_index 28；
              block 19 输出（tuple 20）完整 sequence 全有限 32/32
    控制 ①    hook 捕获但原样返回 → 原生 logits / hidden_states[20] 与原 forward
              NaN-aware elementwise 相等 32/32
    控制 ②    block 19 后 cast → tail final hidden / 外部 float32 logits 全有限 64/64，
              无 token 过滤 64/64
    控制 ③    用 **cast 之前**捕获的 block-19 输出做 pooling → 与冻结 H / E3 cache
              精确一致 131/131（32 H1 union + 67 H1 E3 block + 32 MCQ answer-letter）

本模块把那条已验证的路径提成纯函数，供正式 runner 复用。**不新增数值语义**。

一条 trace 一次 forward，同时产出两侧：

    block 19 输出
      ├─ cast 前 fp16 → 冻结 H / E3 pooling 与**精确锚点**
      └─ cast 后 bf16 → blocks 20-27 + final norm
                        → 外部 float32 lm_head/logsumexp（`stable_logits`）
                        → stable B4/O 输入

**实测环境限定**：cast 后残差流被类型提升为 `torch.float32`（bitsandbytes 的 int8
matmul 内部把激活降回 fp16，bf16 + fp16 相加提升为 float32），模型自带的 fp16
`lm_head` 因此**调不动**。正式 logits 一律走外部 float32 重算 —— 这不是可选项。
该 dtype 结论限于当前本机环境实测，不作跨环境断言。
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from . import v3_e3_repool as repool
from ..generation import stable_logits as sl

TAIL_SCHEMA = "stage1_tail_stabilized_v1"

#: 冻结口径（**Qwen2.5-7B-Instruct**）：block 19 的输出 == hidden_states[20]
#: （HF 在进第 20 层**前**记录）。三个常量保留原值不变，它们同时是全部默认几何；
#: 跨模型运行请显式传 `geometry=`，**不要**改这三个值，也不要 monkey-patch。
CAST_BLOCK = 19
FROZEN_TUPLE_IDX = 20
HIDDEN_DIM = 3584

#: 默认几何 = Qwen 冻结几何。省略 `geometry=` 时行为与参数化之前逐项一致。
DEFAULT_GEOMETRY = repool.QWEN_GEOMETRY

assert (DEFAULT_GEOMETRY.block_index == CAST_BLOCK
        and DEFAULT_GEOMETRY.tuple_index == FROZEN_TUPLE_IDX
        and DEFAULT_GEOMETRY.hidden_dim == HIDDEN_DIM)


class TailStop(RuntimeError):
    """逐 trace 硬门失败 —— 在 trace 边界立即停止。

    **工程失败，不是科学排除**：调用方记 `failed` 事件，**不得**标 `excluded`。
    """

    def __init__(self, message: str, *, unit_id: str, gate: str, diagnostic: dict):
        super().__init__(message)
        self.unit_id = unit_id
        self.gate = gate
        self.diagnostic = diagnostic


class TailHook:
    """在 `model.model.layers[geometry.block_index]` 上捕获输出；`cast=True` 换成 bf16。

    用 hook 修改层输出并继续调用模型原 forward —— 避免手工重建 attention mask /
    position ids / rotary embedding。`captured` 始终是 **cast 之前**的那份，
    它才是冻结 H / E3 锚点的来源。

    `geometry` 省略即 Qwen 冻结几何（block 19）。跨模型时必须显式传入 ——
    层号取错不会报错，只会安静地在错误的深度取 hidden。
    """

    def __init__(self, model: Any, torch: Any, *, cast: bool = True,
                 geometry: repool.ModelGeometry | None = None):
        self.torch = torch
        self.cast = cast
        self.geometry = geometry or DEFAULT_GEOMETRY
        self.captured = None
        self.output_was_tuple = None
        n_layers = len(model.model.layers)
        if n_layers != self.geometry.num_layers:
            raise TailStop(
                f"模型实际层数 {n_layers} != geometry.num_layers "
                f"{self.geometry.num_layers} —— 拒绝在错误的深度挂 hook",
                unit_id="<hook_setup>", gate="geometry_num_layers",
                diagnostic={"model_n_layers": n_layers,
                            "geometry": self.geometry.as_dict()})
        self.handle = (model.model.layers[self.geometry.block_index]
                       .register_forward_hook(self._fn))

    def _fn(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        self.output_was_tuple = is_tuple
        hidden = output[0] if is_tuple else output
        self.captured = hidden.detach().clone()          # **cast 之前**
        if not self.cast:
            return None
        new = hidden.to(self.torch.bfloat16)
        return (new,) + tuple(output[1:]) if is_tuple else new

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def encode_pair(tokenizer: Any, prompt: str, completion: str):
    """与冻结生成期同一编码口径。completion 不加 special tokens。"""
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    return p_ids, c_ids


def forward_base_only(model: Any, torch: Any, p_ids, c_ids):
    """只跑 `model.model`（不含 lm_head）。

    tail 模式**必须**用它：cast 后残差流是 float32，模型自带 fp16 lm_head 会抛
    dtype mismatch；而正式 logits 本就要求外部 float32 重算。
    """
    full = torch.tensor([p_ids + c_ids])
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model.model(input_ids=full.to(dev),
                           attention_mask=torch.ones_like(full).to(dev),
                           output_hidden_states=True)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_tail_unit(*, model: Any, tokenizer: Any, torch: Any, head: sl.Float32Head,
                  unit_id: str, prompt: str, completion: str,
                  expected_prompt_hash: str,
                  geometry: repool.ModelGeometry | None = None) -> dict:
    """一条 trace 的 tail-stabilized 前向。**一次 forward，两侧产出。**

    逐 trace 硬门（任一失败即 `TailStop`，在 trace 边界停）：

        prompt SHA 与冻结输入一致
        cast 前 layer-19 **完整 sequence** 全有限
        tail final hidden 全有限
        float32 logits / logsumexp 全有限
        completion token **不过滤**（投影行数 == n_tok）

    返回 `pre_cast_hidden`（float32 numpy，(T, D)）供调用方做冻结 H / E3 锚点，
    以及逐 token 的 `target_logprob` / `entropy` / `rank` 与 float32 logits。
    """
    got_hash = _sha_text(prompt)
    if got_hash != expected_prompt_hash:
        raise TailStop(f"{unit_id}: prompt_hash 与冻结输入不一致", unit_id=unit_id,
                       gate="prompt_hash",
                       diagnostic={"got": got_hash, "expected": expected_prompt_hash})

    p_ids, c_ids = encode_pair(tokenizer, prompt, completion)
    if not c_ids:
        raise TailStop(f"{unit_id}: completion 编码为空", unit_id=unit_id,
                       gate="tokenize", diagnostic={"n_prompt_tokens": len(p_ids)})
    plen, n_tok = len(p_ids), len(c_ids)

    geo = geometry or DEFAULT_GEOMETRY
    with TailHook(model, torch, cast=True, geometry=geo) as hook:
        out = forward_base_only(model, torch, p_ids, c_ids)
        cap = hook.captured
        if cap is None:
            raise TailStop(f"{unit_id}: hook 未捕获 block-{geo.block_index} 输出",
                           unit_id=unit_id, gate="hook_capture", diagnostic={})
        # 门：cast 前 layer-{block} **完整 sequence** 全有限
        if not bool(torch.isfinite(cap).all()):
            n_bad = int((~torch.isfinite(cap)).sum())
            raise TailStop(
                f"{unit_id}: cast 前 layer-{geo.block_index} 输出存在非有限值 —— "
                f"tail-stabilized 前提在该 trace 上不成立", unit_id=unit_id,
                gate="pre_cast_layer",
                diagnostic={"n_nonfinite_values": n_bad, "seq_len": int(cap.shape[1]),
                            "block_index": geo.block_index})

        final_hidden = out.hidden_states[-1][0]
        if not bool(torch.isfinite(final_hidden).all()):
            n_bad = int((~torch.isfinite(final_hidden)).sum())
            raise TailStop(f"{unit_id}: tail final hidden 非有限", unit_id=unit_id,
                           gate="tail_final_hidden",
                           diagnostic={"n_nonfinite_values": n_bad})

        rows_idx = [sl.causal_row_index(plen, j) for j in range(n_tok)]
        try:
            logits32 = head.project(final_hidden[rows_idx])
            lp32 = sl.log_softmax_rows(logits32, torch)
            stats = sl.row_statistics(lp32, c_ids, torch)
        except sl.NonFiniteStop as exc:
            raise TailStop(f"{unit_id}: 外部 float32 投影/logsumexp 非有限",
                           unit_id=unit_id, gate="stable_logits",
                           diagnostic=dict(exc.diagnostic)) from exc

        # 门：completion token 不过滤
        if int(logits32.shape[0]) != n_tok:
            raise TailStop(
                f"{unit_id}: 投影行数 {int(logits32.shape[0])} != completion token 数 "
                f"{n_tok} —— 发生了 token 过滤", unit_id=unit_id,
                gate="no_token_filtering",
                diagnostic={"n_rows": int(logits32.shape[0]), "n_tok": n_tok})

        pre_cast = cap[0].detach().float().cpu().numpy()
        # hidden 宽度**不在这里**再查一次：它已经被两处更强的检查覆盖 ——
        #   ① assert_h1_anchors / assert_mcq_anchor 拿 pooled 向量与冻结向量做
        #      **逐元素精确比对**，宽度不符必然失败，且顺带证明取的是同一个模型；
        #   ② assert_h1_anchors 里的 E3 shape 门按 geometry.hidden_dim 校验。
        # 在此再加一道只比宽度的弱检查，除了让合成替身跑不起来之外没有增量。
        post_cast_dtype = str(final_hidden.dtype)
        letter_logits32 = logits32.detach().float().cpu()
        letter_logprobs32 = lp32.detach().float().cpu()

    del out, cap, logits32, lp32, final_hidden
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "unit_id": unit_id,
        "prompt_len": plen,
        "completion_ids": list(c_ids),
        "n_completion_tokens": n_tok,
        "pre_cast_hidden": pre_cast,            # (T, D) float32 —— 锚点来源
        "target_logprob": stats["target_logprob"],
        "entropy": stats["entropy"],
        "rank": stats["rank"],
        "logits32": letter_logits32,            # (n_tok, V) —— MCQ 候选 logit 用
        "log_probs32": letter_logprobs32,
        "post_cast_residual_dtype": post_cast_dtype,
        "prompt_hash": got_hash,
        "geometry": geo.as_dict(),
    }


def assert_h1_anchors(*, unit_id: str, pre_cast_hidden: np.ndarray, prompt_len: int,
                      row: dict, cfg, frozen_union: np.ndarray,
                      frozen_e3: dict,
                      geometry: repool.ModelGeometry | None = None) -> dict:
    """H1 的冻结锚点：union H 与**每个** E3 block 都必须精确一致（无容差）。"""
    positions = repool.union_positions(row)
    pooled = repool.pool_union(pre_cast_hidden, prompt_len, positions)
    diag = repool.anchor_exact_match(pooled, frozen_union)
    if not diag["exact"]:
        raise TailStop(f"{unit_id}: H1 union H 与冻结向量不一致", unit_id=unit_id,
                       gate="h1_union_anchor", diagnostic=dict(diag))

    assignment = repool.block_assignment(row, cfg)
    repool.check_block_partition(row, cfg)
    blocks = repool.pool_blocks(pre_cast_hidden, prompt_len, assignment)
    for b, vec in blocks.items():
        key = f"block_{b}"
        if key not in frozen_e3:
            raise TailStop(f"{unit_id}: 既有 E3 cache 缺 {key}", unit_id=unit_id,
                           gate="h1_e3_anchor",
                           diagnostic={"missing": key,
                                       "available": sorted(frozen_e3)})
        d = repool.anchor_exact_match(vec, frozen_e3[key])
        if not d["exact"]:
            raise TailStop(f"{unit_id}: E3 {key} 与既有 cache 不一致", unit_id=unit_id,
                           gate="h1_e3_anchor", diagnostic={"block": b, **d})
    geo = geometry or DEFAULT_GEOMETRY
    arrays = {f"block_{b}": v.astype(np.float32) for b, v in blocks.items()}
    for k, v in arrays.items():
        if (v.dtype != np.float32 or v.shape != (geo.hidden_dim,)
                or not np.isfinite(v).all()):
            raise TailStop(
                f"{unit_id}/{k}: 向量不满足 float32 ({geo.hidden_dim},) 全有限",
                unit_id=unit_id, gate="e3_vector_shape",
                diagnostic={"dtype": str(v.dtype), "shape": list(v.shape),
                            "expected_hidden_dim": geo.hidden_dim})
    return {"arrays": arrays, "assignment": assignment,
            "position_digest": repool.position_digest(assignment),
            "n_blocks": len(assignment)}


def assert_mcq_anchor(*, unit_id: str, pre_cast_hidden: np.ndarray, prompt_len: int,
                      letter_token_index_local: int,
                      frozen_letter_hidden: np.ndarray) -> dict:
    """MCQ 的冻结锚点：answer-letter H 必须与 `mcq_hidden.npz` 精确一致。"""
    idx = prompt_len + int(letter_token_index_local)
    if idx >= pre_cast_hidden.shape[0]:
        raise TailStop(f"{unit_id}: answer-letter 位置越界", unit_id=unit_id,
                       gate="mcq_letter_index",
                       diagnostic={"idx": idx, "seq_len": int(pre_cast_hidden.shape[0])})
    got = pre_cast_hidden[idx]
    diag = repool.anchor_exact_match(got, frozen_letter_hidden)
    if not diag["exact"]:
        raise TailStop(f"{unit_id}: MCQ answer-letter H 与冻结向量不一致",
                       unit_id=unit_id, gate="mcq_letter_anchor", diagnostic=dict(diag))
    return {"letter_row_index": idx, **diag}
