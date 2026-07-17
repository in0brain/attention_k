"""Stage 0 特征缓存的载入与校验。

这是 pilot 与 production 之间的关口。所有"形状对但内容空"的坑都在这里堵:
  - F5 静默填 0（_dense_matrix 缺失填 0 → d_F 不变、检查全绿、列是空的）
  - population_role 混用（pilot_smoke 混进 confirmatory）
  - 两臂 backend 不一致（§6 的 D = S_MCQ − S_H1 会被量化混淆）
  - label 泄漏进特征
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from recover_attention import conditional_increment as ci
from recover_attention import mcq_conditional_increment as mci
from recover_attention.evaluation.config import Stage0Config

ARM_H1 = "h1"
ARM_MCQ = "mcq"

# 每臂的 F5 列表与 ladder 构造器
ARM_SPEC = {
    ARM_H1: {"f5_names": ci.F5_FEATURE_NAMES,
             # H1 的 verbalized-confidence 是稀疏指示量,整列 0 属正常(该批无置信短语)
             "sparse_indicator_prefixes": ("f5_confidence_",),
             "build_specs": ci.build_ladder_specs,
             "o_rung": "rung6_O_f5_plus_text",
             "shortcut_rungs": ("rung2_id_string_only", "rung3_surface_only"),
             "full_text_rung": "rung4_full_text"},
    ARM_MCQ: {"f5_names": mci.MCQ_F5_FEATURE_NAMES,
              "sparse_indicator_prefixes": (),
              "build_specs": mci.build_mcq_ladder_specs,
              "o_rung": "rung6_O_f5_plus_canonical_text",
              "shortcut_rungs": ("rung2_answer_letter_only", "rung3_option_surface_only"),
              "full_text_rung": "rung4_canonical_output_text"},
}


@dataclass
class ArmCache:
    """一臂的 completion-level 特征缓存。"""

    arm: str
    records: list[dict]
    hidden: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    backend: dict
    validation: dict

    @property
    def spec(self) -> dict:
        return ARM_SPEC[self.arm]

    def build_ladder_specs(self) -> dict[str, dict]:
        return self.spec["build_specs"](self.records, hidden=self.hidden)


def _f5_completeness(records: Sequence[dict], f5_names: Sequence[str],
                     sparse_prefixes: Sequence[str]) -> dict:
    """逐列查非零/去重值。

    d_F 只反映列数,不保证列有值 —— 缺失被 _dense_matrix 填 0 后形状不变、其余检查照样绿。
    这个洞在 MCQ 侧真实发生过(label_margin/label_entropy/full_entropy 三列 19/19 全 0)。
    稀疏指示量允许整列 0;连续量整列 0 = 静默缺失,必须拦。
    """
    mat = ci._dense_matrix(records, f5_names)
    nonzero = {n: int((mat[:, j] != 0).sum()) for j, n in enumerate(f5_names)}
    distinct = {n: int(len(np.unique(mat[:, j]))) for j, n in enumerate(f5_names)}
    missing = {n: int(sum(1 for r in records if r.get(n) is None)) for n in f5_names}
    continuous = [n for n in f5_names if not any(n.startswith(p) for p in sparse_prefixes)]
    empty = [n for n in continuous if nonzero[n] == 0]
    constant = [n for n in continuous if distinct[n] <= 1]
    return {"d_F": len(f5_names), "nonzero_per_column": nonzero,
            "distinct_per_column": distinct, "missing_per_column": missing,
            "empty_continuous_columns": empty, "constant_continuous_columns": constant,
            "ok": not empty and not constant,
            "note": ("d_F 只反映列数;缺失会被静默填 0 而形状不变。稀疏指示量可整列 0,"
                     "连续量不行。")}


def load_arm(arm: str, records_path: Path, hidden_path: Path, config: Stage0Config,
             label_field: str, id_field: str = "example_id",
             group_field: str = "group_id") -> ArmCache:
    """载入一臂并做全部准入校验。任一不过 → 抛错,不进分析。"""
    if arm not in ARM_SPEC:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_SPEC)}")
    records = [json.loads(l) for l in open(records_path, encoding="utf-8")]
    if not records:
        raise ValueError(f"{arm}: empty record file {records_path}")

    # population_role:production 绝不接受 pilot_smoke
    roles = sorted({str(r.get("population_role")) for r in records})
    pilot = [r for r in records if r.get("population_role") == "pilot_smoke"]
    if pilot and not config.allow_pilot_population:
        raise ValueError(
            f"{arm}: {len(pilot)}/{len(records)} records are population_role=pilot_smoke but "
            f"allow_pilot_population is False. Pilot records are engineering diagnostics from the "
            f"burned set; they must never enter confirmatory results.")

    # 只保留 eligible（H1: eligible / MCQ: eligible_for_primary）
    def _is_eligible(r: dict) -> bool:
        v = r.get("eligible", r.get("eligible_for_primary"))
        return bool(v)

    eligible = [r for r in records if _is_eligible(r)]
    if not eligible:
        raise ValueError(f"{arm}: no eligible records")

    npz = np.load(hidden_path)
    missing_h = [r[id_field] for r in eligible if r[id_field] not in npz.files]
    if missing_h:
        raise ValueError(f"{arm}: {len(missing_h)} eligible records lack hidden vectors, "
                         f"e.g. {missing_h[:3]}")
    hidden = np.vstack([npz[r[id_field]] for r in eligible])
    dims = {int(npz[k].shape[0]) for k in npz.files}
    if len(dims) != 1:
        raise ValueError(f"{arm}: inconsistent hidden dims {sorted(dims)}")

    y = np.array([int(r[label_field]) for r in eligible])
    if len(np.unique(y)) < 2:
        raise ValueError(f"{arm}: labels are single-class; AUROC undefined")
    groups = np.array([r[group_field] for r in eligible])

    f5 = _f5_completeness(eligible, ARM_SPEC[arm]["f5_names"],
                          ARM_SPEC[arm]["sparse_indicator_prefixes"])
    if not f5["ok"]:
        raise ValueError(
            f"{arm}: F5 incomplete — empty continuous columns {f5['empty_continuous_columns']}, "
            f"constant continuous columns {f5['constant_continuous_columns']}. "
            f"Silently-zero F5 weakens the O baseline and inflates Delta_H.")

    # label 不得进特征
    feat_names = set(ARM_SPEC[arm]["f5_names"])
    if arm == ARM_MCQ:
        feat_names |= set(mci.MCQ_SURFACE_FEATURE_NAMES)
        leaked = sorted(feat_names & mci.FORBIDDEN_FEATURE_FIELDS)
        if leaked:
            raise ValueError(f"{arm}: label leakage into feature names: {leaked}")
    else:
        feat_names |= set(ci.SURFACE_FEATURE_NAMES)

    backend = _read_backend(records_path.parent)
    validation = {
        "arm": arm, "n_records": len(records), "n_eligible": len(eligible),
        "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
        "n_groups": int(len(np.unique(groups))),
        "population_roles": roles,
        "pilot_admitted": bool(pilot) and config.allow_pilot_population,
        "hidden_dim": sorted(dims)[0],
        "f5_completeness": f5,
        "label_field": label_field,
    }
    return ArmCache(arm=arm, records=eligible, hidden=hidden, y=y, groups=groups,
                    backend=backend, validation=validation)


def _read_backend(dir_path: Path) -> dict:
    """从同目录的 *_report.json 里找 backend 指纹。"""
    for name in ("mcq_hidden_reforward_report.json", "smoke_report_h1.json",
                 "generation_report.json"):
        p = dir_path / name
        if p.exists():
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            bk = rep.get("backend") or (rep.get("fingerprint") or {}).get("backend")
            if isinstance(bk, dict):
                return bk
    return {}


BACKEND_KEYS = ("load_in_8bit", "device_map", "attn_implementation", "local_files_only")


def check_backend_invariant(arms: dict[str, ArmCache]) -> dict:
    """**两臂必须同一 backend**（否则 §6 的 D = S_MCQ − S_H1 被量化混淆）。

    与 scripts/sprint_4D_2_conditional_increment.py::_check_backend_invariant 同一条约束,
    此处是分析入口的第二道关。实测 4C(4-bit) 与本管线(8-bit) 的 f5_label_margin 最大差
    10.6(秩相关仍 0.86 → 定位对、数值被量化改)。
    """
    seen = {a: {k: c.backend.get(k) for k in BACKEND_KEYS} for a, c in arms.items()}
    missing = [a for a, b in seen.items() if not any(v is not None for v in b.values())]
    if missing:
        return {"ok": False, "reason": f"backend fingerprint missing for arms {missing}",
                "per_arm": seen}
    values = list(seen.values())
    same = all(v == values[0] for v in values)
    eight_bit = values[0].get("load_in_8bit") is True
    return {"ok": bool(same and eight_bit), "same_backend_all_arms": same,
            "is_8bit": eight_bit, "per_arm": seen,
            "why": ("section-6 gate D = S_MCQ - S_H1 confounds observability with quantization "
                    "if arms differ; 8-bit is forced by H1 long-form degeneration (4D-1)")}
