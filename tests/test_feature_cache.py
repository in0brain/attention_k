"""evaluation.feature_cache 的 backend-invariant 读取契约。

之前没有任何测试覆盖这一层：本仓库唯一的"backend invariant"测试
(test_backend_invariant.py) 测的是另一个并行实现
(sprint_4D_2_conditional_increment.py::_check_backend_invariant，G3 阶段用，
读 smoke_report_*.json)，从未触达 run_stage0_analysis.py 生产入口实际调用的
这一个（check_backend_invariant + _read_backend，此文件内）。

结果是这个函数从真实产物写出的第一天起就没在生产路径上跑对过：它找的文件名
（generation_report.json / smoke_report_h1.json / mcq_hidden_reforward_report.json）
和目录（records_path.parent）都不是 run_stage0_generation.py 实际写出的
（{arm}_generation_report.json，写在 outputs/stage0/generation/，不是
outputs/stage0/features/）。真正同源、同目录的文件是
{arm}_run_fingerprint.json（_build_feature_cache 写在 records 同目录），
schema 是 {"model": {"backend": {...}}}，_read_backend 原来完全不认。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation.feature_cache import (
    ArmCache,
    _read_backend,
    check_backend_invariant,
)

EIGHT_BIT = {"load_in_8bit": True, "device_map": "auto",
            "attn_implementation": "eager", "local_files_only": True}


def _write_run_fingerprint(dir_path, arm, backend):
    """复刻 run_stage0_generation.py::_build_feature_cache 实际写出的产物形状：
    与 {arm}_completion_records.jsonl 同目录，文件名 {arm}_run_fingerprint.json，
    schema 顶层是 model/tokenizer/hidden_index/generation_params/run_hash，
    backend 嵌在 model 之下（真实文件的确切形状，非简化版）。"""
    fp = {"schema": "stage0_fingerprint_v1", "arm": arm,
         "model": {"model_path": "D:/models/Qwen2.5-7B-Instruct", "backend": backend},
         "tokenizer": {"tokenizer_hash": "t"}, "hidden_index": {"hidden_states_tuple_index": 20},
         "generation_params": {"K": 6}, "run_hash": "h"}
    (dir_path / f"{arm}_run_fingerprint.json").write_text(json.dumps(fp), encoding="utf-8")


def test_read_backend_finds_the_real_production_run_fingerprint(tmp_path):
    """回归核心：_read_backend 必须认得 production 实际写出的文件名(带 arm 前缀)
    与 schema({"model": {"backend": ...}})，不能只认 smoke 阶段的旧名字。"""
    _write_run_fingerprint(tmp_path, "h1", EIGHT_BIT)
    bk = _read_backend(tmp_path, "h1")
    assert bk == EIGHT_BIT


def test_read_backend_does_not_cross_contaminate_arms(tmp_path):
    """两臂各自的 run_fingerprint 独立存在同一目录（真实生产布局），
    必须按 arm 精确取自己那份，不能读到另一臂的或读到 None。"""
    four_bit = {"load_in_8bit": False, "load_in_4bit": True, "device_map": "auto",
               "attn_implementation": "eager", "local_files_only": True}
    _write_run_fingerprint(tmp_path, "h1", EIGHT_BIT)
    _write_run_fingerprint(tmp_path, "mcq", four_bit)
    assert _read_backend(tmp_path, "h1") == EIGHT_BIT
    assert _read_backend(tmp_path, "mcq") == four_bit


def test_check_backend_invariant_passes_against_real_production_layout(tmp_path):
    """端到端：两臂的 run_fingerprint 都在（真实生产会有），check_backend_invariant
    必须能读到并判定一致、8-bit、ok=True —— 这正是此前空转的那条链路。"""
    _write_run_fingerprint(tmp_path, "h1", EIGHT_BIT)
    _write_run_fingerprint(tmp_path, "mcq", EIGHT_BIT)
    arms = {
        "h1": ArmCache(arm="h1", records=[], hidden=np.zeros((0, 1)), y=np.zeros(0),
                       groups=np.zeros(0), backend=_read_backend(tmp_path, "h1"), validation={}),
        "mcq": ArmCache(arm="mcq", records=[], hidden=np.zeros((0, 1)), y=np.zeros(0),
                        groups=np.zeros(0), backend=_read_backend(tmp_path, "mcq"),
                        validation={}),
    }
    r = check_backend_invariant(arms)
    assert r["ok"] is True, r


def test_check_backend_invariant_still_catches_missing_fingerprint(tmp_path):
    """没有任何可读的报告/fingerprint 文件时，必须继续报 missing，不能悄悄放行。"""
    arms = {
        "h1": ArmCache(arm="h1", records=[], hidden=np.zeros((0, 1)), y=np.zeros(0),
                       groups=np.zeros(0), backend=_read_backend(tmp_path, "h1"), validation={}),
        "mcq": ArmCache(arm="mcq", records=[], hidden=np.zeros((0, 1)), y=np.zeros(0),
                        groups=np.zeros(0), backend=_read_backend(tmp_path, "mcq"),
                        validation={}),
    }
    r = check_backend_invariant(arms)
    assert r["ok"] is False and "missing" in r["reason"]
