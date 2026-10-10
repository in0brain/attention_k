"""`run_arm` 存 hidden 时必须**独立拷贝**，不能留视图（轻量回归，不加载任何模型）。

2026-09-17 的真实故障：MCQ 生产跑了 8 小时，生成进程 Private 从 13.8 GB 涨到 22.3 GB
（≈ **8.4 MB/unit**，线性），系统可用 RAM 从 11.7 GB 掉到 4.7 GB，速率随换页压力从
19 s/unit 恶化到 60 s/unit。外推整臂 10,560 条需要 ~88 GB —— 跑不完。

根因：`MCQBackend.score_completion` 里 `row["hidden"] = tf["hidden"][prompt_len + j]`
是整条 trace `[T, 4096] float32`（~7 MB）的**视图**；`run_arm` 原本写
`np.asarray(vec, dtype=np.float32)`，而 dtype 本就匹配 —— **不拷贝**，原样返回视图。
于是 `run.hidden[unit_id]` 里那个 16 KB 的向量，把背后 7 MB 的整表钉住整臂。

H1 不受影响：它走 `pool_hidden_states` 的 mean pooling，产出新数组。
（实测佐证：H1 那 20 小时 Private 一直平在 19.14 GB。）

**本修复只改数组生命周期**，不改生成、解析、特征定义、采样参数、精度或后端；
数值 / shape / dtype / 落盘内容与修复前逐位相同 —— 下面逐条钉住。
不为此加载 Qwen，也不重跑 G3/G4。
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

T, DIM = 450, 4096          # 一条真实 trace 的 hidden 形状量级


def _big() -> np.ndarray:
    """构造"整条 trace 的 hidden"。

    ⚠️ **不能用 `np.arange(...).reshape(...)`**：reshape 出来的本身就是视图，
    numpy 会把 base 链折叠到底层那个一维数组，于是"大数组可回收"在**有缺陷的
    旧写法下也会通过** —— 测试就没牙了（本轮实测撞到过）。这里造一个真正的基数组。
    """
    a = np.zeros((T, DIM), dtype=np.float32)
    a[:] = np.arange(T, dtype=np.float32)[:, None]
    assert a.base is None, "对照数组本身必须是基数组，否则这组测试失去判别力"
    return a


def test_asarray_on_matching_dtype_keeps_the_view_alive():
    """先钉住这条 numpy 语义 —— 它是缺陷的成因，不是推测。"""
    full = np.zeros((8, 16), dtype=np.float32)
    view = full[3]
    assert view.base is full

    assert np.asarray(view, dtype=np.float32) is view          # 旧写法：原样返回视图
    copied = np.array(view, dtype=np.float32, copy=True)       # 新写法：真拷贝
    assert copied.base is None
    assert np.array_equal(copied, view) and copied.dtype == view.dtype


class _Backend:
    """假后端：`run_unit` 返回**大数组的一行视图**，与 MCQ 生产同形。"""

    arm = "mcq"

    def __init__(self, full: np.ndarray, row_index: int = 3):
        self.full, self.row_index = full, row_index

    def run_fingerprint(self, arm):
        return {"arm": arm, "run_hash": "test"}

    def verify_hidden_index(self):
        return {"block_index_zero_based": 22, "hidden_states_tuple_index": 23,
                "numeric_match": True, "max_abs_diff": 0.0}

    def run_unit(self, unit, record):
        return {"example_id": record["example_id"], "eligible_for_primary": True,
                "hidden": self.full[self.row_index]}


def _run(tmp_path: Path, full: np.ndarray):
    from recover_attention.evaluation import v3_g3_smoke as g3
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    # V3AnalysisConfig 是冻结 dataclass（本来就该冻结），用默认 K=6
    return g3.run_arm(arm="mcq", records=[{"example_id": "e0", "rendered_prompt": "p",
                                           "group_id": "g0"}],
                      backend=_Backend(full), out_dir=tmp_path,
                      cfg=V3AnalysisConfig(), gate="g3")


def test_saved_vector_does_not_share_the_big_array(tmp_path):
    full = _big()
    run = _run(tmp_path, full)

    assert run.hidden, "没存下 hidden"
    for uid, vec in run.hidden.items():
        assert vec.base is None, f"{uid} 仍是视图 —— 整表会被钉住整臂"
        assert not np.shares_memory(vec, full), f"{uid} 仍与大数组共享内存"


def test_big_array_is_released_when_nothing_else_holds_it(tmp_path):
    """原数组无其他引用时必须可回收 —— 这正是修复要达成的效果。"""
    full = _big()
    run = _run(tmp_path, full)
    kept = list(run.hidden.values())
    probe = weakref.ref(full)

    del full, run                      # 放掉大数组与 ArmRun，只留下存下来的向量
    gc.collect()

    assert probe() is None, "大数组没能回收 —— 说明还被存下来的向量牵着"
    assert all(v.base is None for v in kept)


def test_values_shape_dtype_and_npy_bytes_are_unchanged(tmp_path):
    """数值 / shape / dtype / 落盘内容与修复前逐位一致。

    "修复前"的落盘内容用**同一个视图**直接 `np.save` 得到 —— 那正是旧代码写出去的东西。
    """
    full = _big()
    expected = full[3]
    run = _run(tmp_path, full)

    ref = tmp_path / "reference_pre_fix.npy"
    np.save(ref, expected)                          # 旧写法落盘的字节
    ref_bytes = ref.read_bytes()

    saved = sorted((tmp_path / "mcq_g3_unit_hidden").glob("*.npy"))
    assert saved, "没有落盘 .npy"
    for p in saved:
        got = np.load(p)
        assert got.shape == expected.shape == (DIM,)
        assert got.dtype == expected.dtype == np.float32
        assert np.array_equal(got, expected)        # 逐位相同，不是容差
        assert p.read_bytes() == ref_bytes          # **落盘字节**也一致

    for vec in run.hidden.values():
        assert vec.shape == expected.shape and vec.dtype == expected.dtype
        assert np.array_equal(vec, expected)


def test_leak_projection_matches_what_we_measured():
    """把实测数字钉进测试：视图写法下，每 unit 钉住的就是整表大小。"""
    per_unit_mb = T * DIM * 4 / 1024 / 1024
    assert 6.0 < per_unit_mb < 9.0                  # 实测告警 8.4 MB/unit
    assert per_unit_mb * 10_560 / 1024 > 60         # 整臂 > 60 GB —— 必然撑爆


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
