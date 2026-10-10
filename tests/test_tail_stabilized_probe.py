"""tail-stabilized 可行性探针的单测。合成替身，**不加载模型、不碰真实产物**。

覆盖研究者点名的七项：
  * layer 返回**裸张量 / tuple** 两种形态都要正确处理
  * cast=False 不改输出（返回 None，捕获仍发生）
  * cast=True 捕获 **pre-cast** 并返回 bf16
  * scan verdict 非 after_block19 时**拒绝** controls
  * target / .tmp 防覆盖
  * tail 只走 **base model** + 外部 `Float32Head`
  * 任一 pre-cast boundary / final hidden / stable logits 非有限即**失败**

**遥测污染的回归防护**：`test_controls_allowed_only_after_block19_wording` 曾经
只 monkeypatch 了 `SCAN_OUT`/`CTRL_OUT`，但 `run_controls` 里的遥测路径是
`SMOKE / f"tail_stabilized_controls_seg{...}"`（对**真实** `outputs/stage1/smoke/`
现场编号），且 `h1_known_bad()` / `mcq_retained()` 经 `SMOKE` / `PROD` 读真实产物。
一旦 backend 用假 model_path 失败，遥测文件已经在真实目录里创建好了 ——
实测产生了 `tail_stabilized_controls_seg002..006_gpu_telemetry.csv` 五份
表头+1 行的空产物。修法是把 `SMOKE`/`PROD` 也一并 monkeypatch 到 `tmp_path`
并提供最小 fixture，让整条调用链**完全不接触真实目录**。
`_assert_real_smoke_untouched` 对本文件**每一条测试**前后做快照比对，
任何测试只要写了真实 `outputs/stage1/smoke/` 就会在这里挂掉。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.generation import stable_logits as sl  # noqa: E402

_REAL_SMOKE = _REPO_ROOT / "outputs" / "stage1" / "smoke"


def _snapshot_real_smoke() -> dict:
    if not _REAL_SMOKE.exists():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in _REAL_SMOKE.iterdir() if p.is_file()}


@pytest.fixture(autouse=True)
def _assert_real_smoke_untouched():
    """本文件任何测试都不得改动真实 `outputs/stage1/smoke/` 的文件集合或字节。"""
    before = _snapshot_real_smoke()
    yield
    after = _snapshot_real_smoke()
    assert after == before, (
        "本测试写了真实 outputs/stage1/smoke/！\n"
        f"  新增: {sorted(set(after) - set(before))}\n"
        f"  丢失: {sorted(set(before) - set(after))}\n"
        f"  改动: {sorted(k for k in after.keys() & before.keys() if after[k] != before[k])}")


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "_tail_probe", _REPO_ROOT / "scripts" / "probe_tail_stabilized_feasibility.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return load_probe()


# ------------------------------------------------------- hook：两种返回形态
class _Layer(torch.nn.Module):
    """替身 decoder layer。`as_tuple` 控制返回裸张量还是 tuple。"""

    def __init__(self, as_tuple: bool, extra=("aux",)):
        super().__init__()
        self.as_tuple = as_tuple
        self.extra = extra

    def forward(self, x):
        out = x * 2.0
        return (out,) + tuple(self.extra) if self.as_tuple else out


class _Stack(torch.nn.Module):
    def __init__(self, as_tuple: bool, n: int = 28):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer(as_tuple) for _ in range(n))


class _Model(torch.nn.Module):
    def __init__(self, as_tuple: bool):
        super().__init__()
        self.model = _Stack(as_tuple)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_hook_handles_bare_tensor_and_tuple_outputs(probe, as_tuple):
    """依赖**真实 hook 机制**触发（不手工调 _fn）——
    这同时验证 PyTorch 确实采用了 hook 的返回值。"""
    m = _Model(as_tuple)
    hook = probe.Block19Hook(m, torch, cast=True)
    try:
        x = torch.randn(1, 4, 8, dtype=torch.float16)
        got = m.model.layers[probe.CAST_BLOCK](x)        # hook 自动触发
    finally:
        hook.close()
    assert hook.output_was_tuple is as_tuple
    if as_tuple:
        assert isinstance(got, tuple) and got[1:] == ("aux",)
        assert got[0].dtype == torch.bfloat16
    else:
        assert isinstance(got, torch.Tensor) and got.dtype == torch.bfloat16
    assert hook.captured.dtype == torch.float16          # 捕获的是 **pre-cast**


def test_hook_cast_false_leaves_output_untouched(probe):
    m = _Model(False)
    hook = probe.Block19Hook(m, torch, cast=False)
    try:
        x = torch.randn(1, 3, 8, dtype=torch.float16)
        got = m.model.layers[probe.CAST_BLOCK](x)
    finally:
        hook.close()
    assert got.dtype == torch.float16                    # **不改动**输出
    assert torch.equal(got, x * 2.0)                     # 就是层本来的输出
    assert hook.captured is not None                     # 捕获仍然发生
    assert torch.equal(hook.captured, got)
    assert hook.captured.dtype == torch.float16


def test_hook_cast_true_captures_pre_cast_fp16_not_the_bf16(probe):
    m = _Model(False)
    hook = probe.Block19Hook(m, torch, cast=True)
    try:
        x = torch.randn(1, 3, 8, dtype=torch.float16)
        got = m.model.layers[probe.CAST_BLOCK](x)
    finally:
        hook.close()
    assert hook.captured.dtype == torch.float16          # **pre-cast**
    assert got.dtype == torch.bfloat16                   # 下游拿到的是 bf16
    # 捕获的就是 cast 之前那份：数值上等于层本来的输出
    assert torch.equal(hook.captured, (x * 2.0).to(torch.float16))
    assert hook.captured.data_ptr() != got.data_ptr()    # clone，不是别名


def test_hook_registers_on_the_frozen_block_index(probe):
    assert probe.CAST_BLOCK == 19
    assert probe.FROZEN_TUPLE_IDX == 20                  # block 19 的输出 == tuple 20
    m = _Model(False)
    hook = probe.Block19Hook(m, torch, cast=False)
    try:
        assert len(m.model.layers[19]._forward_hooks) == 1
        assert all(not m.model.layers[i]._forward_hooks
                   for i in range(28) if i != 19)
    finally:
        hook.close()
    assert not m.model.layers[19]._forward_hooks         # close 后干净移除


# ------------------------------------------- scan verdict 决定能否跑 controls
@pytest.mark.parametrize("verdict", ["at_block19", "before_block19", "no_nonfinite_found"])
def test_controls_refused_when_scan_verdict_is_not_after_block19(probe, tmp_path,
                                                                 monkeypatch, verdict):
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"verdict_relative_to_block19": verdict}), encoding="utf-8")
    monkeypatch.setattr(probe, "SCAN_OUT", scan)
    monkeypatch.setattr(probe, "CTRL_OUT", tmp_path / "ctrl.json")
    with pytest.raises(probe.ProbeStop, match="不换层"):
        probe.run_controls("unused-model-path")


def test_controls_allowed_only_after_block19_wording(probe, tmp_path, monkeypatch):
    """判定为 after_block19 时不应在**这一关**被拦（后续因无模型另行失败）。

    **必须完全隔离真实目录**：`run_controls` 在到达"假模型加载失败"之前，会先
    (a) 用 SMOKE 现场编号写遥测文件、(b) 用 PROD 读 h1/mcq analyzable 记录。
    只 monkeypatch SCAN_OUT/CTRL_OUT 不够——上一版就是这样把 seg002..006 五份
    空遥测写进了真实 outputs/stage1/smoke/。这里把 SMOKE 与 PROD 都换成
    tmp_path 下的最小 fixture，让整条调用链一步都不碰真实目录。
    """
    smoke_dir = tmp_path / "smoke"
    prod_dir = tmp_path / "prod"
    smoke_dir.mkdir()
    prod_dir.mkdir()

    scan = smoke_dir / "tail_stabilized_layer_scan.json"
    scan.write_text(json.dumps({"verdict_relative_to_block19": "after_block19"}),
                    encoding="utf-8")

    # h1_known_bad() 读 SMOKE/shadow_smoke_report.json；给一条已知非有限的 unit。
    (smoke_dir / "shadow_smoke_report.json").write_text(json.dumps({
        "per_unit": [{"unit_id": "fx_ex_0__greedy_0",
                      "row_counts": {"final_hidden_nonfinite": 3}}]}),
        encoding="utf-8")

    # load_analyzable() 读 PROD/{h1,mcq}_completion_records.jsonl。
    h1_row = {"example_id": "fx_ex_0", "sample_type": "greedy", "sample_index": 0,
              "trace_id": "fx_ex_0__greedy_0", "id_token_positions_local": [0, 1]}
    mcq_row = {"example_id": "fx_mcq_0", "eligible_for_primary": True}
    (prod_dir / "h1_completion_records.jsonl").write_text(
        json.dumps(h1_row), encoding="utf-8")
    (prod_dir / "mcq_completion_records.jsonl").write_text(
        json.dumps(mcq_row), encoding="utf-8")

    monkeypatch.setattr(probe, "SMOKE", smoke_dir)
    monkeypatch.setattr(probe, "PROD", prod_dir)
    monkeypatch.setattr(probe, "SCAN_OUT", scan)
    monkeypatch.setattr(probe, "CTRL_OUT", smoke_dir / "ctrl.json")

    real_smoke_before = _snapshot_real_smoke()
    with pytest.raises(Exception) as exc:
        probe.run_controls("definitely-not-a-model")
    assert "不换层" not in str(exc.value)
    # 双保险：即便上面的隔离有遗漏，也不能让真实目录出现新文件。
    assert _snapshot_real_smoke() == real_smoke_before
    # 隔离生效的证据：遥测确实落在 tmp_path，而不是真实 SMOKE。
    assert list(smoke_dir.glob("tail_stabilized_controls_seg*_gpu_telemetry.csv"))


# ------------------------------------------------------------- 防覆盖
def test_publish_refuses_existing_target(probe, tmp_path):
    p = tmp_path / "out.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(probe.ProbeStop, match="拒绝覆盖"):
        probe.publish_json_exclusive(p, {"a": 1})
    assert p.read_text(encoding="utf-8") == "{}"


def test_publish_refuses_stale_tmp_and_preserves_it(probe, tmp_path):
    p = tmp_path / "out.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text("residue", encoding="utf-8")
    with pytest.raises(probe.ProbeStop, match="陈旧临时文件"):
        probe.publish_json_exclusive(p, {"a": 1})
    assert tmp.read_text(encoding="utf-8") == "residue"
    assert not p.exists()


def test_publish_succeeds_and_leaves_no_tmp(probe, tmp_path):
    p = tmp_path / "out.json"
    probe.publish_json_exclusive(p, {"a": 1, "b": [2, 3]})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}
    assert not p.with_suffix(".json.tmp").exists()


def test_scan_refuses_when_output_exists(probe, tmp_path, monkeypatch):
    p = tmp_path / "scan.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(probe, "SCAN_OUT", p)
    with pytest.raises(probe.ProbeStop, match="拒绝覆盖"):
        probe.run_scan("unused")


def test_telemetry_is_exclusive_create(probe, tmp_path):
    p = tmp_path / "tel.csv"
    p.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        probe.start_telemetry(p)
    assert p.read_text(encoding="utf-8") == "keep"


# ------------------------------------ tail 只走 base model + 外部 Float32Head
def test_tail_forward_uses_base_model_not_causal_lm(probe):
    class _Base:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kw):
            self.calls += 1
            return "base"

    class _Causal:
        def __init__(self, base):
            self.model = base
            self.calls = 0

        def __call__(self, **kw):
            self.calls += 1
            return "causal"

    base = _Base()
    causal = _Causal(base)
    backend = type("B", (), {})()
    backend.ctx = {"torch": torch, "model": causal}
    backend._to_device = lambda d: d

    assert probe._forward(backend, [1, 2], [3], base_only=True) == "base"
    assert (base.calls, causal.calls) == (1, 0)
    assert probe._forward(backend, [1, 2], [3], base_only=False) == "causal"
    assert (base.calls, causal.calls) == (1, 1)


def test_controls_source_uses_base_only_in_tail_mode(probe):
    src = (_REPO_ROOT / "scripts" / "probe_tail_stabilized_feasibility.py").read_text("utf-8")
    assert "base_only=True" in src
    assert "native_logits_unavailable_reason" in src
    assert "post_cast_residual_dtype" in src
    assert "Float32Head" in src


# ------------------------------------------ 非有限即失败（三个位置各一条）
def test_external_stable_logits_stop_on_nonfinite_final_hidden():
    """tail 后 final hidden 非有限 → `Float32Head.project` 立即停机，不过滤不替换。"""
    class _H(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(11, 6))
            self.bias = None

    model = type("M", (), {"lm_head": _H()})()
    head = sl.Float32Head(model, torch, vocab_chunk=5)
    x = torch.randn(3, 6, dtype=torch.float32)
    x[1, 0] = float("nan")
    with pytest.raises(sl.NonFiniteStop) as exc:
        head.project(x)
    assert exc.value.diagnostic["stage"] == "input_hidden"


def test_pre_cast_boundary_finiteness_is_checked_elementwise(probe):
    """pre-cast 边界的有限性判据必须覆盖**完整 sequence**，不是抽样。"""
    good = torch.randn(1, 5, 8, dtype=torch.float16)
    bad = good.clone()
    bad[0, 3, 2] = float("nan")
    assert bool(torch.isfinite(good).all())
    assert not bool(torch.isfinite(bad).all())


def test_anchor_comparison_is_exact_with_no_tolerance():
    from recover_attention.evaluation import v3_e3_repool as repool
    a = np.arange(8, dtype=np.float32)
    assert repool.anchor_exact_match(a, a.copy())["exact"] is True
    b = a.copy()
    b[2] = np.nextafter(b[2], np.float32(np.inf))
    diag = repool.anchor_exact_match(a, b)
    assert diag["exact"] is False and diag["ulp_distance_at_argmax"] == 1


# ------------------------------------------------ 排除 unit 的口径不得漂移
@pytest.mark.local_artifacts('outputs/stage1/production/mcq_forward_nonfinite_diagnostics.jsonl',
                             'outputs/stage1/production/mcq_technical_exclusions.jsonl')
def test_excluded_units_are_labelled_untestable_and_never_replayed(probe):
    note = probe.excluded_units_note()
    assert note["classification"] == "untestable_historical_failure_cases"
    assert note["first_failure_has_no_diagnostic"] is True
    assert note["n_technical_exclusions"] > note["n_with_diagnostic"]
    assert "不重放" in note["policy"] and "不作跨臂推断" in note["policy"]
    src = (_REPO_ROOT / "scripts" / "probe_tail_stabilized_feasibility.py").read_text("utf-8")
    for banned in ("regenerate_completion", "replay_excluded", "re_include"):
        assert banned not in src
