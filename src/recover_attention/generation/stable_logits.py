"""稳定 logits 原语 —— **performance-blind**，与冻结生成语义完全隔离。

为什么需要它：冻结的 `backends._teacher_forced` 在 8-bit / fp16 下会产出**整行**非有限的
logits（实测 `partial_nonfinite` 恒为 0），而下游 `finite_float` 把它们**静默**剔除。
本模块提供一条不做任何过滤的数值路径，把 lm_head projection 与 logsumexp / softmax
放到 float32 里完成。

**硬约束**（写进代码，不只写文档）：

* 不对 NaN / Inf 做过滤、替换、填零或跳过；
* 任一预测行、候选 logit 或派生值非有限 → **立即抛错停机**（`NonFiniteStop`）；
* 不生成 completion —— 只对既有 completion 做 teacher-forced 前向；
* **不修改** `backends.py` 的任何冻结语义。

产出只能称 `stable-logit shadow sensitivity` / `protocol-deviation sensitivity`，
**不得**冒充原预注册 primary。
"""

from __future__ import annotations

from typing import Any

import numpy as np

STABLE_PROJECTION_SCHEMA = "stable_logit_projection_f32_v1"

#: 词表维度的分块大小 —— fp32 的整份 lm_head 权重是 152064×3584×4B ≈ 2.18 GiB，
#: 在 12 GiB 卡上与 8-bit 模型共存太紧，默认按词表切块并即时上转。
DEFAULT_VOCAB_CHUNK = 16384


class NonFiniteStop(RuntimeError):
    """稳定路径上出现非有限值 —— 立即停机，**绝不**过滤或替换。"""

    def __init__(self, message: str, *, diagnostic: dict):
        super().__init__(message)
        self.diagnostic = diagnostic


# ------------------------------------------------------------ 结构核实
def inspect_lm_head(model: Any) -> dict:
    """运行时核实 lm_head 结构 —— 不猜 weight tying / bias / 量化跳过。"""
    head = getattr(model, "lm_head", None)
    cfg = getattr(model, "config", None)
    qcfg = getattr(model, "quantization_config", None) or getattr(
        cfg, "quantization_config", None)
    weight = getattr(head, "weight", None)
    bias = getattr(head, "bias", None)
    skip = None
    for attr in ("llm_int8_skip_modules", "skip_modules"):
        if qcfg is not None and hasattr(qcfg, attr):
            skip = getattr(qcfg, attr)
            break
    return {
        "lm_head_type": type(head).__name__ if head is not None else None,
        "lm_head_module_path": "model.lm_head",
        "has_bias": bias is not None,
        "weight_dtype": str(getattr(weight, "dtype", None)),
        "weight_shape": list(getattr(weight, "shape", []) or []),
        "weight_is_quantized_param": type(weight).__name__ if weight is not None else None,
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "quantization_skip_modules": list(skip) if skip else skip,
        "final_norm_module": type(getattr(getattr(model, "model", None), "norm", None)).__name__,
    }


def resolve_final_hidden_convention(model: Any, out: Any, torch: Any) -> dict:
    """实测 `hidden_states[-1]` 是否已过 final norm —— **不靠约定，靠数值比对**。

    判据：直接把 `hidden_states[-1]` 送进 lm_head，与 `out.logits` 比。
    对得上 ⇒ 已过 norm；对不上则再试 `norm(hidden_states[-1])`。
    只在**两侧都有限**的行上比较，避免 NaN 污染判据。
    """
    hs = out.hidden_states
    logits = out.logits[0]
    last = hs[-1][0]
    norm = getattr(getattr(model, "model", None), "norm", None)

    def _max_abs_diff(candidate) -> float | None:
        with torch.no_grad():
            got = model.lm_head(candidate.to(model.lm_head.weight.dtype))
        a = got.float()
        b = logits.float()
        mask = torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1)
        if not bool(mask.any()):
            return None
        return float((a[mask] - b[mask]).abs().max())

    direct = _max_abs_diff(last)
    renormed = _max_abs_diff(norm(last)) if norm is not None else None
    return {
        "n_hidden_states": len(hs),
        "expected_n_hidden_states": int(model.config.num_hidden_layers) + 1,
        "max_abs_diff_lm_head_of_last": direct,
        "max_abs_diff_lm_head_of_norm_of_last": renormed,
        "verdict": (
            "hidden_states[-1] is post-final-norm" if direct is not None and direct < 1e-2
            else "hidden_states[-1] is PRE-final-norm"
            if renormed is not None and renormed < 1e-2
            else "INCONCLUSIVE — neither candidate reproduces out.logits"),
        "comparison_note": "只在两侧都有限的行上比较；容差 1e-2（fp16 lm_head 的量化噪声）",
    }


# --------------------------------------------------------- float32 投影
class Float32Head:
    """把 lm_head projection 放进 float32；按词表分块，避免 2.18 GiB 的整份上转。

    权重来源固定为**模型自己的** `lm_head`（含 bias，若存在），不重建、不猜。
    """

    def __init__(self, model: Any, torch: Any, vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
                 cache_f32_weight: bool = False):
        self.torch = torch
        self.head = model.lm_head
        self.weight = self.head.weight
        self.bias = getattr(self.head, "bias", None)
        self.vocab_chunk = int(vocab_chunk)
        self.vocab_size = int(self.weight.shape[0])
        self.hidden_size = int(self.weight.shape[1])
        # 全量跑时每条 trace 重复上转 2.18 GiB 的权重是主要瓶颈 —— 缓存一份 fp32。
        # 显存不够就退回分块，**只影响速度，不影响数值**。
        self.weight_f32 = None
        self.cache_mode = "chunked"
        if cache_f32_weight:
            try:
                self.weight_f32 = self.weight.to(torch.float32)
                self.bias_f32 = (self.bias.to(torch.float32)
                                 if self.bias is not None else None)
                self.cache_mode = "cached_f32_weight"
            except (RuntimeError, MemoryError):
                self.weight_f32 = None
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def project(self, hidden_rows: Any) -> Any:
        """(R, H) hidden → (R, V) float32 logits。输入非有限即抛错。"""
        torch = self.torch
        h = hidden_rows.to(torch.float32)
        if not bool(torch.isfinite(h).all()):
            bad = (~torch.isfinite(h)).sum(dim=-1)
            raise NonFiniteStop(
                "final hidden 非有限 —— float32 projection 无法补救，停机",
                diagnostic={"stage": "input_hidden",
                            "n_rows": int(h.shape[0]),
                            "n_rows_with_nonfinite": int((bad > 0).sum()),
                            "max_nonfinite_per_row": int(bad.max())})
        with torch.no_grad():
            if self.weight_f32 is not None:
                logits = h @ self.weight_f32.t()
                if self.bias_f32 is not None:
                    logits = logits + self.bias_f32
            else:
                outs = []
                for start in range(0, self.vocab_size, self.vocab_chunk):
                    stop = min(start + self.vocab_chunk, self.vocab_size)
                    w = self.weight[start:stop].to(torch.float32)
                    part = h @ w.t()
                    if self.bias is not None:
                        part = part + self.bias[start:stop].to(torch.float32)
                    outs.append(part)
                    del w
                logits = torch.cat(outs, dim=-1)
        if not bool(torch.isfinite(logits).all()):
            bad = (~torch.isfinite(logits)).sum(dim=-1)
            raise NonFiniteStop(
                "float32 projection 输出非有限 —— 停机，不过滤不替换",
                diagnostic={"stage": "projected_logits",
                            "n_rows": int(logits.shape[0]),
                            "n_rows_with_nonfinite": int((bad > 0).sum()),
                            "max_nonfinite_per_row": int(bad.max())})
        return logits


# ------------------------------------------------------ 派生量（float32）
def log_softmax_rows(logits: Any, torch: Any) -> Any:
    """float32 log_softmax。非有限即停机。"""
    lp = torch.log_softmax(logits.to(torch.float32), dim=-1)
    if not bool(torch.isfinite(lp).all()):
        raise NonFiniteStop("log_softmax 输出非有限 —— 停机",
                            diagnostic={"stage": "log_softmax",
                                        "n_rows": int(lp.shape[0])})
    return lp


def row_statistics(log_probs: Any, target_ids, torch: Any) -> dict:
    """逐行取 target logprob / entropy / rank。**全部**在 float32 完成。

    rank 的定义与冻结实现一致：`int((row > row[tid]).sum()) + 1`。
    差别在于这里的 row 已保证全有限 —— 冻结实现里 NaN 会让所有比较为 False，
    把 rank 静默写成 1（degeneracy a），这里不可能发生。
    """
    lp = log_probs
    idx = torch.as_tensor(list(target_ids), dtype=torch.long, device=lp.device)
    tgt = lp.gather(1, idx.view(-1, 1)).view(-1)
    ent = -(lp.exp() * lp).sum(dim=-1)
    rank = (lp > tgt.view(-1, 1)).sum(dim=-1) + 1
    out = {
        "target_logprob": tgt.detach().cpu().numpy().astype(np.float64),
        "entropy": ent.detach().cpu().numpy().astype(np.float64),
        "rank": rank.detach().cpu().numpy().astype(np.int64),
    }
    for name in ("target_logprob", "entropy"):
        arr = out[name]
        if not bool(np.isfinite(arr).all()):
            raise NonFiniteStop(f"{name} 非有限 —— 停机",
                                diagnostic={"stage": name, "n": int(arr.size)})
    return out


def candidate_logits(log_probs: Any, candidate_ids, torch: Any) -> dict:
    """MCQ 用：取候选字母的 logprob。任一非有限即停机。"""
    lp = log_probs
    out = {}
    for label, tid in candidate_ids.items():
        v = float(lp[0, int(tid)])
        if not np.isfinite(v):
            raise NonFiniteStop(f"候选 logit {label!r} 非有限 —— 停机",
                                diagnostic={"stage": "candidate_logit", "label": label})
        out[label] = v
    return out


# ------------------------------------------------- float64 参考实现（单测用）
def reference_log_softmax(logits: np.ndarray) -> np.ndarray:
    """float64 参考：数值稳定的 log_softmax。只用于单测锁定数学正确性。"""
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def reference_entropy(log_probs: np.ndarray) -> np.ndarray:
    lp = np.asarray(log_probs, dtype=np.float64)
    return -(np.exp(lp) * lp).sum(axis=-1)


def reference_rank(log_probs: np.ndarray, target_ids) -> np.ndarray:
    lp = np.asarray(log_probs, dtype=np.float64)
    ids = np.asarray(list(target_ids), dtype=np.int64)
    tgt = lp[np.arange(lp.shape[0]), ids]
    return (lp > tgt[:, None]).sum(axis=-1) + 1


def causal_row_index(prompt_len: int, j: int) -> int:
    """预测第 j 个 completion token 的那一行。与冻结实现同一口径。"""
    return int(prompt_len) + int(j) - 1
