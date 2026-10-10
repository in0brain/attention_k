"""E3 performance-blind per-block repooling —— **纯函数核心**（本轮不含 production 写盘）。

背景：`h1_unit_hidden/*.npy` 与 `h1_hidden.npz` 都已按 E1 口径 pool 成**单**向量
（over `id_token_positions_local`），per-token hidden 未保留，所以 E3 的 block-specific H
无法由缓存重构，必须 performance-blind 地重做 teacher-forced 前向。

本模块的全部函数都满足两条不变量：

1. **label-blind**：任何路径都不读 `record["label"]` / `mention["label"]` /
   `mention["emitted"]`。块归属只由 `mention.token_indices ∩ id_token_positions_local`
   决定 —— 该交集已由生成期的 `eligible_identifier_token_positions` 冻结，因此"哪些
   mention 进了 H"这个信息**已经编码在 token 位置里**，不需要再读标签。
   `strip_label_fields()` 后重算**块归属与 position digest** 并逐项比较，是这条
   不变量的可执行断言。**注意其证据范围**：它证明的是块归属与位置摘要不依赖 label，
   **不是**「两次模型前向的 hidden 输出逐位相同」——后者没有做过。
2. **无副作用**：不写盘、不建目录、不碰 manifest。本轮授权范围只到
   代码 / 单测 / 64 条锚点 / 32-128 微基准，正式 `--run` 与恢复状态机留给全量 sprint。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import conditional_increment as _ci
from ..conditional_increment import pool_hidden_states
from ..generation.fingerprint import sha256_text, stable_hash
from ..h1_data import build_h1_chat_messages
from .v3_config import V3AnalysisConfig
from .v3_grouping import e3_block_of

REPOOL_SCHEMA = "stage1_e3_repool_v1"
POOLING_SCHEMA = "mean_over_unique_token_positions/v1"
TOKEN_POSITION_SELECTION_RULE = (
    "mention.token_indices INTERSECT id_token_positions_local, nonempty; label-free"
)
HIDDEN_DIM = 3584

#: 只在 `strip_label_fields` 里出现——用于证明本模块从不依赖它们。
LABEL_FIELDS_RECORD = ("label", "wrong_label", "gold_label", "primary_exclusion_reason")
LABEL_FIELDS_MENTION = ("label", "emitted")


# ═════════════════════════════════════════════════════════════════════════════
# 模型几何 —— 跨模型复现的**唯一**几何来源
# ═════════════════════════════════════════════════════════════════════════════
#: 相对深度由预注册 §8「层 ⌊0.7L⌋」冻结，**不是可调参数**。
FROZEN_REL_DEPTH: float = 0.7


class GeometryViolation(ValueError):
    """几何参数违反关系不变量 —— 停机，不回退到默认值。"""


@dataclass(frozen=True)
class ModelGeometry:
    """`(L, block, tuple, d)` 四元组，构造即校验**关系**而不仅仅是取值。

    只检查"等于调用方传进来的值"是同义反复：它无法发现 `tuple != block + 1`、
    `block != ⌊0.7L⌋` 或 `hidden_dim` 与模型 config 不符这三类真实错误。
    因此每一项都对着一个**独立来源**核：rel_depth 对预注册、tuple 对 HF 语义、
    block 对 `resolve_hidden_index`、hidden_dim 对模型 config.json。

    ⚠️ 本对象只承载**静态**几何。运行时 hidden-index 的数值一致性
    （`numeric_match` / `max_abs_diff`）必须由真实前向给出，见
    `backends._RealBackendBase.verify_hidden_index`，**本对象不代表它已通过**。
    """

    num_layers: int
    block_index: int
    tuple_index: int
    hidden_dim: int
    rel_depth: float = FROZEN_REL_DEPTH
    source: str = "explicit"

    def __post_init__(self) -> None:
        for name in ("num_layers", "block_index", "tuple_index", "hidden_dim"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise GeometryViolation(
                    f"{name} 必须是非 bool 的 int，得到 {v!r}（{type(v).__name__}）")
        if self.num_layers <= 0:
            raise GeometryViolation(f"num_layers 必须为正，得到 {self.num_layers}")
        if self.hidden_dim <= 0:
            raise GeometryViolation(f"hidden_dim 必须为正，得到 {self.hidden_dim}")
        if not isinstance(self.rel_depth, float) or isinstance(self.rel_depth, bool):
            raise GeometryViolation(f"rel_depth 必须是 float，得到 {self.rel_depth!r}")
        # rel_depth 由预注册冻结，不接受任意深度 —— 事后选层是被明令禁止的。
        if not math.isclose(self.rel_depth, FROZEN_REL_DEPTH, rel_tol=0.0, abs_tol=1e-12):
            raise GeometryViolation(
                f"rel_depth={self.rel_depth} 与冻结值 {FROZEN_REL_DEPTH} 不符；"
                f"层位由预注册第 8 节冻结为 floor(0.7*L)，不得事后选层")
        if not (0 <= self.block_index < self.num_layers):
            raise GeometryViolation(
                f"block_index={self.block_index} 越界，须落在 [0, {self.num_layers})")
        # HF 语义：hidden_states = (embedding, layer_1, …, layer_L) → tuple = block + 1
        if self.tuple_index != self.block_index + 1:
            raise GeometryViolation(
                f"tuple_index={self.tuple_index} != block_index+1="
                f"{self.block_index + 1}；HF hidden_states 的 index 0 是 embedding，"
                f"第 k 个 block 的输出是 index k，该关系不可自由指定")
        want = _ci.resolve_hidden_index(self.num_layers, self.rel_depth)
        if self.block_index != want["block_index_zero_based"]:
            # 报错文字只用 GBK 可编码字符：本机 Windows 控制台是 GBK，
            # 用 floor(...) 而非数学取整符号，免得异常在打印时被 UnicodeEncodeError 掩盖。
            raise GeometryViolation(
                f"block_index={self.block_index} != floor({self.rel_depth}*"
                f"{self.num_layers})={want['block_index_zero_based']}"
                f"（resolve_hidden_index 独立复算）")

    def as_dict(self) -> dict:
        return {"num_layers": self.num_layers, "rel_depth": self.rel_depth,
                "block_index_zero_based": self.block_index,
                "hidden_states_tuple_index": self.tuple_index,
                "hidden_dim": self.hidden_dim, "source": self.source,
                "static_geometry_verified_from_config": self.source.startswith("config:"),
                "runtime_hidden_index_verified": False,
                "runtime_verification_note": (
                    "numeric_match / max_abs_diff 只能由真实模型前向给出"
                    "（backends.verify_hidden_index）；本对象**不代表**它已通过")}

    @classmethod
    def from_num_layers(cls, num_layers: int, hidden_dim: int, *,
                        rel_depth: float = FROZEN_REL_DEPTH,
                        source: str = "explicit") -> "ModelGeometry":
        idx = _ci.resolve_hidden_index(int(num_layers), rel_depth)
        return cls(num_layers=int(num_layers),
                   block_index=int(idx["block_index_zero_based"]),
                   tuple_index=int(idx["hidden_states_tuple_index"]),
                   hidden_dim=int(hidden_dim), rel_depth=rel_depth, source=source)

    @classmethod
    def from_model_config(cls, model_path: Any, *,
                          rel_depth: float = FROZEN_REL_DEPTH) -> "ModelGeometry":
        """只读 `config.json` 推导几何。**绝不加载权重**，也不 import transformers。"""
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.exists():
            raise GeometryViolation(f"模型 config 不存在：{cfg_path}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        missing = [k for k in ("num_hidden_layers", "hidden_size") if k not in raw]
        if missing:
            raise GeometryViolation(f"{cfg_path}: config 缺字段 {missing}")
        return cls.from_num_layers(int(raw["num_hidden_layers"]),
                                   int(raw["hidden_size"]), rel_depth=rel_depth,
                                   source=f"config:{Path(model_path).name}")

    def assert_matches_model_config(self, model_path: Any) -> dict:
        """与模型 config.json 交叉核对。不符即停，**不静默采用其中之一**。"""
        other = ModelGeometry.from_model_config(model_path, rel_depth=self.rel_depth)
        diff = {k: (getattr(self, k), getattr(other, k))
                for k in ("num_layers", "block_index", "tuple_index", "hidden_dim")
                if getattr(self, k) != getattr(other, k)}
        if diff:
            raise GeometryViolation(
                f"几何与模型 config 不符（字段: (传入, config)）：{diff}；"
                f"model_path={model_path}")
        return {"model_path": str(model_path), "matches_config": True,
                **other.as_dict()}


#: Qwen2.5-7B-Instruct —— Stage 1 confirmatory 的冻结几何，同时是全部默认值。
QWEN_GEOMETRY = ModelGeometry(num_layers=28, block_index=19, tuple_index=20,
                              hidden_dim=HIDDEN_DIM, source="frozen_qwen_default")

#: Llama-3.1-8B-Instruct 的**静态预期**。仍须 `assert_matches_model_config` 实测核对。
LLAMA_31_8B_EXPECTED_GEOMETRY = ModelGeometry(num_layers=32, block_index=22,
                                              tuple_index=23, hidden_dim=4096,
                                              source="expected_llama_3_1_8b")

assert QWEN_GEOMETRY.hidden_dim == HIDDEN_DIM        # 常量与默认几何不得漂移


def add_geometry_cli_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给脚本挂几何参数。**三个参数全省略 = Qwen 冻结默认**，行为与改动前一致。"""
    g = ap.add_argument_group("model geometry")
    g.add_argument("--geometry-from-model-config", action="store_true",
                   help="从 --model-path 的 config.json 推导几何（只读，不加载权重）")
    g.add_argument("--num-layers", type=int, default=None,
                   help="显式层数；与 --geometry-from-model-config 同用时必须一致")
    g.add_argument("--hidden-dim", type=int, default=None, help="显式 hidden 维度")
    return ap


def geometry_from_cli(args: Any, *, model_path: Any = None) -> ModelGeometry:
    """解析几何。冲突即停，**不按优先级静默取其一**。"""
    explicit_n = getattr(args, "num_layers", None)
    explicit_d = getattr(args, "hidden_dim", None)
    from_config = bool(getattr(args, "geometry_from_model_config", False))
    if from_config:
        path = model_path if model_path is not None else getattr(args, "model_path", None)
        if not path:
            raise GeometryViolation("--geometry-from-model-config 需要 --model-path")
        geo = ModelGeometry.from_model_config(path)
        if explicit_n is not None and int(explicit_n) != geo.num_layers:
            raise GeometryViolation(
                f"--num-layers={explicit_n} 与 config 的 {geo.num_layers} 冲突")
        if explicit_d is not None and int(explicit_d) != geo.hidden_dim:
            raise GeometryViolation(
                f"--hidden-dim={explicit_d} 与 config 的 {geo.hidden_dim} 冲突")
        return geo
    if explicit_n is None and explicit_d is None:
        return QWEN_GEOMETRY
    if explicit_n is None or explicit_d is None:
        raise GeometryViolation(
            "--num-layers 与 --hidden-dim 必须同时给出（只给一个会让另一半悄悄用 "
            "Qwen 默认，正是跨模型最容易出的错）")
    return ModelGeometry.from_num_layers(int(explicit_n), int(explicit_d),
                                         source="explicit_cli")


# --------------------------------------------------------------------- 身份
def derive_unit_id(row: Mapping[str, Any]) -> str:
    """与 `h1_unit_hidden/{unit_id}.npy` 的文件名口径一致。"""
    for key in ("example_id", "sample_type", "sample_index"):
        if key not in row:
            raise KeyError(f"row missing {key!r}; cannot derive unit_id")
    return f"{row['example_id']}__{row['sample_type']}_{int(row['sample_index'])}"


def build_prompt_record(row: Mapping[str, Any]) -> dict:
    """finalized record 只有 `prompt_text`，而 `build_h1_chat_messages` 强制读
    `question_text` 并对空值抛错 —— 不加这层适配会全量失败。"""
    text = row.get("prompt_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{row.get('example_id')!r}: prompt_text missing or empty")
    return {**row, "question_text": text}


def render_prompt(row: Mapping[str, Any], tokenizer: Any) -> str:
    """重建生成期实际前向的 prompt 字符串（chat template + generation prompt）。"""
    return tokenizer.apply_chat_template(
        build_h1_chat_messages(build_prompt_record(row)),
        tokenize=False, add_generation_prompt=True)


def strip_label_fields(row: Mapping[str, Any]) -> dict:
    """删去 record 与 mention 上的全部标签字段。本模块任何输出都不得因此改变。"""
    out = {k: v for k, v in row.items() if k not in LABEL_FIELDS_RECORD}
    if "mentions" in row:
        mentions = []
        for m in row.get("mentions") or []:
            if isinstance(m, Mapping):
                m = {k: v for k, v in m.items() if k not in LABEL_FIELDS_MENTION}
            mentions.append(m)
        out["mentions"] = mentions
    return out


# ----------------------------------------------------------------- 块归属
def union_positions(row: Mapping[str, Any]) -> list[int]:
    """冻结的 E1 pooling 位置（completion-local）。"""
    return sorted({int(p) for p in (row.get("id_token_positions_local") or [])})


def block_assignment(row: Mapping[str, Any], cfg: V3AnalysisConfig) -> dict[int, list[int]]:
    """block_id -> 该块的 completion-local token 位置（排序去重）。**label-free**。

    只保留与冻结 `id_token_positions_local` 的交集：echoed / CVE mention 的 token 不在
    该集合内，因此自动被排除，无需读 `family`，更无需读 `label`。
    """
    pos = set(union_positions(row))
    out: dict[int, set[int]] = {}
    for m in row.get("mentions") or []:
        if not isinstance(m, Mapping):
            continue
        ident = m.get("normalized")
        if not ident:
            continue
        inter = {int(t) for t in (m.get("token_indices") or [])} & pos
        if not inter:
            continue
        out.setdefault(e3_block_of(str(ident), cfg), set()).update(inter)
    return {b: sorted(v) for b, v in sorted(out.items())}


def expected_blocks(row: Mapping[str, Any], cfg: V3AnalysisConfig) -> list[int]:
    return sorted(block_assignment(row, cfg))


def position_digest(assignment: Mapping[int, Sequence[int]]) -> str:
    """块->位置 的规范摘要，用于 done 事件与 .npz 自洽校验。"""
    canon = [[int(b), sorted({int(p) for p in assignment[b]})] for b in sorted(assignment)]
    return sha256_text(json.dumps(canon, separators=(",", ":")))


def check_block_partition(row: Mapping[str, Any], cfg: V3AnalysisConfig) -> None:
    """块间不重叠，且并集恰为 `id_token_positions_local`。任一不成立即 fail-closed。"""
    assignment = block_assignment(row, cfg)
    seen: set[int] = set()
    for b, pos in assignment.items():
        dup = seen & set(pos)
        if dup:
            raise ValueError(f"blocks overlap at positions {sorted(dup)} (block {b})")
        seen |= set(pos)
    want = set(union_positions(row))
    if seen != want:
        raise ValueError(
            f"block union {sorted(seen)} != id_token_positions_local {sorted(want)}")


# ------------------------------------------------------- tokenizer-only 校验
def verify_prompt_and_positions(row: Mapping[str, Any], tokenizer: Any) -> dict:
    """**不加载 model** 的前置校验：prompt SHA、completion token 数、位置不越界。

    任一不符即抛错——调用方负责在第一条失败时全量停止，不得继续加载模型。
    """
    unit_id = derive_unit_id(row)
    prompt = render_prompt(row, tokenizer)
    got, want = sha256_text(prompt), row.get("prompt_hash")
    if got != want:
        raise ValueError(f"{unit_id}: prompt_hash mismatch: rebuilt {got} != frozen {want}")

    completion = row.get("completion")
    if not isinstance(completion, str) or not completion:
        raise ValueError(f"{unit_id}: completion missing or empty")
    n_tok = len(tokenizer(completion, add_special_tokens=False)["input_ids"])

    # ⚠️ 不要拿 `f5_completion_token_count` 当序列长度。它的定义是
    # `len(clean)` = **有限 logprob 的 completion token 数**（h1_f5_features.py:94），
    # 实测在 300/300 条上都严格小于重编码 token 数（mean 26.1 vs 81.7）。
    # 冻结记录自身即可自证二者不同口径：存在 max(id_token_positions_local) >
    # f5_completion_token_count 的记录，而这些位置对真实 token 序列并不越界。
    # 这里只把它作为**信息量**记录，不作为断言。
    pos = union_positions(row)
    if not pos:
        raise ValueError(f"{unit_id}: no eligible identifier token positions")
    if pos[0] < 0 or pos[-1] >= n_tok:
        raise ValueError(
            f"{unit_id}: id_token_positions_local out of range for n_tok={n_tok}: "
            f"min {pos[0]} max {pos[-1]}")
    return {"unit_id": unit_id, "prompt_sha256": got, "n_completion_tokens": n_tok,
            "n_union_positions": len(pos),
            "f5_completion_token_count": row.get("f5_completion_token_count"),
            "f5_count_equals_sequence_length": (
                row.get("f5_completion_token_count") == n_tok)}


# ------------------------------------------------------------------ pooling
def pool_union(hidden: Any, prompt_len: int, positions: Sequence[int]) -> np.ndarray:
    """union anchor：对**全部** eligible 位置一次性 mean pooling。

    注意不是"各块向量再平均"——块大小不同，块均值的均值 != 全体均值。
    """
    return pool_hidden_states(hidden, [int(prompt_len) + int(p) for p in positions])


def pool_blocks(hidden: Any, prompt_len: int,
                assignment: Mapping[int, Sequence[int]]) -> dict[int, np.ndarray]:
    return {int(b): pool_hidden_states(hidden, [int(prompt_len) + int(p) for p in pos])
            for b, pos in sorted(assignment.items())}


def anchor_exact_match(pooled: Any, frozen: Any) -> dict:
    """精确一致断言。**不设容差**——不一致只产诊断，容差留给另一份 Recovery Preflight。"""
    a = np.asarray(pooled, dtype=np.float32)
    b = np.asarray(frozen, dtype=np.float32)
    if a.shape != b.shape:
        return {"exact": False, "reason": "shape", "pooled_shape": list(a.shape),
                "frozen_shape": list(b.shape)}
    exact = bool(np.array_equal(a, b))
    diag: dict[str, Any] = {"exact": exact}
    if not exact:
        d = np.abs(a - b)
        i = int(np.argmax(d))
        ai = a[i:i + 1].copy().view(np.int32)[0]
        bi = b[i:i + 1].copy().view(np.int32)[0]
        diag.update({
            "reason": "value",
            "n_differing": int((a != b).sum()),
            "max_abs_diff": float(d.max()),
            "argmax_index": i,
            "pooled_at_argmax": float(a[i]),
            "frozen_at_argmax": float(b[i]),
            "ulp_distance_at_argmax": int(abs(int(ai) - int(bi))),
        })
    return diag


# ------------------------------------------------------------ 锚点确定性选取
#: 分层配额：4 块与 3 块全取（各 2 / 11 条），其余按 stride 补足。
ANCHOR_QUOTA: dict[int, int | None] = {4: None, 3: None, 2: 26, 1: 25}
ANCHOR_SIZE = 64


def select_anchor_units(rows: Sequence[Mapping[str, Any]], cfg: V3AnalysisConfig,
                        size: int = ANCHOR_SIZE) -> list[str]:
    """确定性分层选取：按 unit_id 排序分桶，高块数全取，低块数等距 stride 补足。

    纯确定性（无 RNG），可由任何人用同一份 records 复算。
    """
    buckets: dict[int, list[str]] = {}
    for row in rows:
        n = len(expected_blocks(row, cfg))
        if n:
            buckets.setdefault(n, []).append(derive_unit_id(row))
    for key in buckets:
        buckets[key].sort()

    chosen: list[str] = []
    for n_blocks in sorted(buckets, reverse=True):
        if len(chosen) >= size:
            break
        pool = buckets[n_blocks]
        quota = ANCHOR_QUOTA.get(n_blocks)
        take = len(pool) if quota is None else quota
        take = min(take, len(pool), size - len(chosen))
        if take <= 0:
            continue
        step = max(1, len(pool) // take)
        chosen.extend(pool[::step][:take])
    return sorted(set(chosen))


def anchor_set_sha256(unit_ids: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(unit_ids)))


# ------------------------------------------------------------------ 指纹拆分
#: 跨机迁移允许不同——只记录，不作为拒绝条件。
VARIABLE_FINGERPRINT_KEYS = (
    "model_path", "tokenizer_name_or_path", "model_hash", "tokenizer_hash",
    "gpu_name", "driver_version", "cuda_runtime", "compute_capability",
    "directory_fsync_supported", "device_map_resolved", "total_memory_bytes",
)

#: 迁移兼容性基线所覆盖的软件版本。注意 `h1_run_fingerprint.json` **未记录**它们，
#: 所以本项是**前瞻性** reforward 基线，不追溯声称为原始生成期软件环境。
REFERENCE_LIBRARIES = ("torch", "transformers", "bitsandbytes", "accelerate")


def _identity_payload(fp: Mapping[str, Any], drop: Sequence[str]) -> dict:
    return {k: v for k, v in fp.items() if k not in drop}


def model_identity_hash(model_fp: Mapping[str, Any]) -> str:
    """**路径无关**的模型身份哈希。

    `fingerprint.model_fingerprint` 把 `model_path` 算进了 `model_hash`，因此原始
    `model_hash` 在换机后必然改变，不能作为跨机硬一致项。这里去掉路径后重算。

    该哈希覆盖 config 元数据与 backend 配置，**不覆盖权重字节**——权重同一性的实际
    证据是 64 条精确锚点，不是这个哈希。
    """
    return stable_hash(_identity_payload(model_fp, ("model_path", "model_hash")))


def tokenizer_identity_hash(tok_fp: Mapping[str, Any]) -> str:
    """路径无关的分词器身份哈希（去掉 `name_or_path` 与原始 `tokenizer_hash`）。"""
    return stable_hash(_identity_payload(tok_fp, ("name_or_path", "tokenizer_hash")))


def hard_consistency_payload(run_fp: Mapping[str, Any], cfg: V3AnalysisConfig,
                             source_shas: Mapping[str, str],
                             library_versions: Mapping[str, str]) -> dict:
    """跨机**必须逐键相等**的部分。"""
    model_fp = run_fp["model"]
    return {
        "model_identity_hash": model_identity_hash(model_fp),
        "tokenizer_identity_hash": tokenizer_identity_hash(run_fp["tokenizer"]),
        "hidden_index": dict(run_fp["hidden_index"]),
        "backend": dict(model_fp.get("backend") or {}),
        "torch_dtype": model_fp.get("torch_dtype"),
        "library_versions": {k: str(library_versions.get(k)) for k in REFERENCE_LIBRARIES},
        "e3_n_blocks": int(cfg.e3_n_blocks),
        "e3_block_salt": str(cfg.e3_block_salt),
        "pooling_schema": POOLING_SCHEMA,
        "token_position_selection_rule": TOKEN_POSITION_SELECTION_RULE,
        "source_shas": dict(source_shas),
    }


def compare_hard_consistency(reference: Mapping[str, Any],
                             current: Mapping[str, Any]) -> list[str]:
    """返回不一致键的清单（空 = 通过）。fail-closed：缺键也算不一致。"""
    problems: list[str] = []
    for key in sorted(set(reference) | set(current)):
        a = reference.get(key, "<missing>")
        b = current.get(key, "<missing>")
        if a != b:
            problems.append(f"{key}: reference={a!r} current={b!r}")
    return problems


# --------------------------------------------------------- .npz 自洽校验（纯）
def validate_npz_self_consistency(path: Any, row: Mapping[str, Any],
                                  cfg: V3AnalysisConfig,
                                  source_fingerprint_id: str,
                                  geometry: "ModelGeometry | None" = None) -> dict:
    """**无 done 事件时**的 orphan 校验：没有既有 SHA 可比，只能查自洽性。

    通过后由调用方现算 bytes / SHA256 并补记 done——本轮**不实现**补记（无 manifest、
    无 writer、无 failed 事件），那属于全量 sprint。任一项不过即抛错，调用方须停止并
    另出 Recovery Preflight，**不得自动删除**。
    """
    path = Path(path)
    unit_id = derive_unit_id(row)
    geo = geometry or QWEN_GEOMETRY

    # (1) ZIP 目录与每条目 CRC 完整可读
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"{unit_id}: npz corrupt at member {bad!r}")

    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        arrays = {k: np.asarray(z[k]) for k in names if k.startswith("block_")}
        meta_raw = str(z["meta"].item()) if "meta" in names else None
    meta = json.loads(meta_raw) if meta_raw else None

    # (2) 内嵌身份精确匹配
    if not isinstance(meta, dict):
        raise ValueError(f"{unit_id}: npz missing embedded meta")
    for key, want in (("trace_id", unit_id),
                      ("prompt_hash", row.get("prompt_hash")),
                      ("source_fingerprint_id", source_fingerprint_id)):
        if meta.get(key) != want:
            raise ValueError(f"{unit_id}: meta[{key!r}]={meta.get(key)!r} != {want!r}")

    # (3) block keys 与源记录现算的 expected blocks 精确相等
    assignment = block_assignment(row, cfg)
    want_keys = {f"block_{b}" for b in assignment}
    if set(arrays) != want_keys:
        raise ValueError(
            f"{unit_id}: block keys {sorted(arrays)} != expected {sorted(want_keys)}")

    # (4) positions 精确相等、块间不重叠、并集 == id_token_positions_local
    got_pos = {int(k): [int(p) for p in v]
               for k, v in (meta.get("token_positions") or {}).items()}
    want_pos = {int(b): [int(p) for p in pos] for b, pos in assignment.items()}
    if got_pos != want_pos:
        raise ValueError(f"{unit_id}: token_positions mismatch: {got_pos} != {want_pos}")
    check_block_partition(row, cfg)

    # (5) dtype / shape / 有限性
    for key in sorted(arrays):
        arr = arrays[key]
        if arr.dtype != np.float32:
            raise ValueError(f"{unit_id}/{key}: dtype {arr.dtype} != float32")
        if arr.shape != (geo.hidden_dim,):
            raise ValueError(f"{unit_id}/{key}: shape {arr.shape} != ({geo.hidden_dim},)")
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"{unit_id}/{key}: non-finite values")

    blob = path.read_bytes()
    return {
        "unit_id": unit_id,
        "n_blocks": len(arrays),
        "block_ids": sorted(assignment),
        "position_digest": position_digest(assignment),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_on_disk": os.path.getsize(path),
        "validated": True,
        "note": "self-consistency only; no prior done event existed to compare against",
    }
