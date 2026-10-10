"""tail-stabilized 正式核心与 runner 的单测。合成替身，**不加载模型、不碰真实产物**。

覆盖：
  * TailHook：裸张量 / tuple 两种返回形态；captured 恒为 **cast 之前**
  * 逐 trace 硬门：prompt SHA / cast 前 layer-19 全序列 / tail final hidden /
    float32 logits / **不过滤 token** —— 每条各一个负控制
  * 冻结锚点：H1 union + 每个 E3 block、MCQ answer-letter，**无容差**
  * TailStop 是**工程失败**语义，不得写成 excluded
  * runner：正式产物必须不存在；输出防覆盖；分段规则算术
  * 确定性分层取样可复算、与输入顺序无关
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation import v3_tail_stabilized as tail  # noqa: E402
from recover_attention.generation import stable_logits as sl  # noqa: E402

_REAL_OUT = _REPO_ROOT / "outputs" / "stage1"


def _snapshot(dirpath: Path) -> dict:
    if not dirpath.exists():
        return {}
    return {str(p.relative_to(dirpath)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dirpath.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _real_outputs_untouched():
    """本文件任何测试都不得改动真实 outputs/stage1/{smoke,benchmark}/。

    `tail_stabilized/` 正式开跑后有数千个 NPZ，逐测试算 SHA 会让整份测试从 10 s
    变成十几分钟 —— 它改由模块级的 `_heavy_outputs_untouched` 守（比文件数 /
    总字节 / 最新 mtime），任何写入同样会被抓到。
    """
    watched = [_REAL_OUT / "smoke", _REAL_OUT / "benchmark"]
    before = {str(d): _snapshot(d) for d in watched}
    yield
    after = {str(d): _snapshot(d) for d in watched}
    assert after == before, f"测试写了真实 outputs！diff={[k for k in after if after[k] != before[k]]}"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_tail_runner", _REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return load_runner()


# ------------------------------------------------------------ 替身模型
V, D, NLAYER = 61, 8, 28


class _Layer(torch.nn.Module):
    """替身 decoder layer。`inject_nan=True` 时**在自己的 forward 里**产出 NaN。

    注入点必须在层内部：hook 是在层调用**期间**触发的，若在 `_Base.forward` 的
    循环里、层返回之后再注入，hook 捕到的仍是干净值，就测不到 pre_cast 那道门。
    """

    def __init__(self, as_tuple: bool, inject_nan: bool = False):
        super().__init__()
        self.as_tuple = as_tuple
        self.inject_nan = inject_nan

    def forward(self, x):
        out = x + 0.5
        if self.inject_nan:
            out = out.clone()
            out[0, 0, 0] = float("nan")
        return (out, "aux") if self.as_tuple else out


class _Base(torch.nn.Module):
    """替身 Qwen2Model：逐层前向并按 HF 口径记录 hidden_states（进层前 append）。"""

    def __init__(self, as_tuple: bool, break_at: int | None = None):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _Layer(as_tuple, inject_nan=(break_at is not None and i == break_at))
            for i in range(NLAYER))
        self.break_at = break_at

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        n = input_ids.shape[1]
        h = torch.ones(1, n, D, dtype=torch.float16) * 0.25
        hs = []
        for layer in self.layers:
            hs.append(h)
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        hs.append(h)                     # post-"norm"
        return type("O", (), {"hidden_states": tuple(hs), "last_hidden_state": h})()


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(V, D, dtype=torch.float32) * 0.1)
        self.bias = None


class _Model(torch.nn.Module):
    def __init__(self, as_tuple: bool = False, break_at: int | None = None):
        super().__init__()
        self.model = _Base(as_tuple, break_at)
        self.lm_head = _Head()


class _Tok:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = list(range(3, 3 + len(text.split())))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


PROMPT = "p p p p"
COMPLETION = "c c c"
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def _head_for(model):
    return sl.Float32Head(model, torch, vocab_chunk=17)


# ------------------------------------------------------------- TailHook
@pytest.mark.parametrize("as_tuple", [False, True])
def test_tailhook_handles_both_output_shapes(as_tuple):
    m = _Model(as_tuple)
    with tail.TailHook(m, torch, cast=True) as hook:
        got = m.model.layers[tail.CAST_BLOCK](torch.ones(1, 2, D, dtype=torch.float16))
    if as_tuple:
        assert isinstance(got, tuple) and got[1] == "aux"
        assert got[0].dtype == torch.bfloat16
    else:
        assert got.dtype == torch.bfloat16
    assert hook.output_was_tuple is as_tuple
    assert hook.captured.dtype == torch.float16           # **cast 之前**


def test_tailhook_cast_false_is_a_pure_observer():
    m = _Model(False)
    x = torch.ones(1, 2, D, dtype=torch.float16)
    with tail.TailHook(m, torch, cast=False) as hook:
        got = m.model.layers[tail.CAST_BLOCK](x)
    assert got.dtype == torch.float16
    assert torch.equal(got, x + 0.5)
    assert torch.equal(hook.captured, got)


def test_tailhook_targets_the_frozen_block_and_cleans_up():
    assert tail.CAST_BLOCK == 19 and tail.FROZEN_TUPLE_IDX == 20
    m = _Model()
    hook = tail.TailHook(m, torch, cast=True)
    assert len(m.model.layers[19]._forward_hooks) == 1
    assert all(not m.model.layers[i]._forward_hooks for i in range(NLAYER) if i != 19)
    hook.close()
    assert not m.model.layers[19]._forward_hooks


# ------------------------------------------------------- run_tail_unit 正路
def test_run_tail_unit_happy_path_returns_pre_cast_and_stats():
    m = _Model()
    res = tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                             unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                             expected_prompt_hash=PROMPT_SHA)
    assert res["prompt_len"] == 4 and res["n_completion_tokens"] == 3
    assert res["pre_cast_hidden"].shape == (7, D)
    assert res["pre_cast_hidden"].dtype == np.float32
    for k in ("target_logprob", "entropy", "rank"):
        assert res[k].shape == (3,)
        assert np.isfinite(np.asarray(res[k], dtype=np.float64)).all()
    assert res["logits32"].shape == (3, V)               # 行数 == n_tok，未过滤
    assert res["prompt_hash"] == PROMPT_SHA


def test_pre_cast_hidden_is_block19_output_not_the_bf16():
    """pre_cast_hidden 必须等于 hidden_states[20]（cast 前），而非 cast 后的值。"""
    m = _Model()
    ref = m.model(input_ids=torch.tensor([[0] * 7]), output_hidden_states=True)
    want = ref.hidden_states[tail.FROZEN_TUPLE_IDX][0].float().numpy()
    res = tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                             unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                             expected_prompt_hash=PROMPT_SHA)
    assert np.array_equal(res["pre_cast_hidden"], want)


# --------------------------------------------------- 逐 trace 硬门负控制
def test_gate_prompt_hash_mismatch():
    m = _Model()
    with pytest.raises(tail.TailStop) as exc:
        tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                           unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                           expected_prompt_hash="0" * 64)
    assert exc.value.gate == "prompt_hash"
    assert exc.value.unit_id == "u1"


def test_gate_pre_cast_layer19_nonfinite():
    """cast 前 layer-19 完整 sequence 有非有限 → 立即停，前提在该 trace 不成立。"""
    m = _Model(break_at=tail.CAST_BLOCK)         # 第 19 层输出注入 NaN
    with pytest.raises(tail.TailStop) as exc:
        tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                           unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                           expected_prompt_hash=PROMPT_SHA)
    # gate id 已改为几何中性的 "pre_cast_layer"（Llama 上是 layer 22，字面量 19 会错）；
    # 断言强度不变，另加断言：诊断必须带上实际 block_index。
    assert exc.value.gate == "pre_cast_layer"
    assert exc.value.diagnostic["n_nonfinite_values"] >= 1
    assert exc.value.diagnostic["block_index"] == tail.CAST_BLOCK


def test_gate_tail_final_hidden_nonfinite():
    """cast 前干净、但尾部（第 25 层）坏 → 在 tail_final_hidden 这一关停。"""
    m = _Model(break_at=25)
    with pytest.raises(tail.TailStop) as exc:
        tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                           unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                           expected_prompt_hash=PROMPT_SHA)
    assert exc.value.gate == "tail_final_hidden"


def test_gate_stable_logits_nonfinite():
    """final hidden 有限但投影权重坏 → 归到 stable_logits 门。"""
    m = _Model()
    with torch.no_grad():
        m.lm_head.weight[0, 0] = float("inf")
    with pytest.raises(tail.TailStop) as exc:
        tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                           unit_id="u1", prompt=PROMPT, completion=COMPLETION,
                           expected_prompt_hash=PROMPT_SHA)
    assert exc.value.gate == "stable_logits"


def test_gate_empty_completion():
    m = _Model()
    with pytest.raises(tail.TailStop) as exc:
        tail.run_tail_unit(model=m, tokenizer=_Tok(), torch=torch, head=_head_for(m),
                           unit_id="u1", prompt=PROMPT, completion="",
                           expected_prompt_hash=PROMPT_SHA)
    assert exc.value.gate == "tokenize"


def test_no_token_filtering_gate_exists_in_source():
    src = (_REPO_ROOT / "src" / "recover_attention" / "evaluation"
           / "v3_tail_stabilized.py").read_text(encoding="utf-8")
    assert 'gate="no_token_filtering"' in src
    assert "logits32.shape[0]) != n_tok" in src


def test_tailstop_is_engineering_failure_not_excluded():
    src = (_REPO_ROOT / "src" / "recover_attention" / "evaluation"
           / "v3_tail_stabilized.py").read_text(encoding="utf-8")
    assert "工程失败，不是科学排除" in src
    assert "不得**标 `excluded`" in src or "不得" in src
    runner_src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    assert "不标 `excluded`" in runner_src
    for banned in ('state": "excluded"', "mark_excluded", "auto_retry"):
        assert banned not in runner_src


# ----------------------------------------------------------- 冻结锚点
def _h1_row(positions=(0, 1)):
    return {"example_id": "ex", "sample_type": "greedy", "sample_index": 0,
            "trace_id": "ex__greedy_0", "prompt_hash": PROMPT_SHA,
            "id_token_positions_local": list(positions),
            "mentions": [{"normalized": "T1001", "token_indices": list(positions),
                          "family": "attack"}]}


def test_h1_anchor_accepts_exact_and_rejects_one_ulp(monkeypatch):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    row = _h1_row()
    hidden = np.arange(7 * repool.HIDDEN_DIM, dtype=np.float32).reshape(7, repool.HIDDEN_DIM)
    plen = 4
    want_union = repool.pool_union(hidden, plen, repool.union_positions(row))
    assignment = repool.block_assignment(row, cfg)
    want_e3 = {f"block_{b}": v for b, v in
               repool.pool_blocks(hidden, plen, assignment).items()}

    ok = tail.assert_h1_anchors(unit_id="u", pre_cast_hidden=hidden, prompt_len=plen,
                                row=row, cfg=cfg, frozen_union=want_union,
                                frozen_e3=want_e3)
    assert ok["n_blocks"] == len(assignment)

    bad = want_union.copy()
    bad[3] = np.nextafter(bad[3], np.float32(np.inf))
    with pytest.raises(tail.TailStop) as exc:
        tail.assert_h1_anchors(unit_id="u", pre_cast_hidden=hidden, prompt_len=plen,
                               row=row, cfg=cfg, frozen_union=bad, frozen_e3=want_e3)
    assert exc.value.gate == "h1_union_anchor"
    assert exc.value.diagnostic["ulp_distance_at_argmax"] == 1


def test_h1_e3_block_mismatch_is_caught():
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    row = _h1_row()
    hidden = np.arange(7 * repool.HIDDEN_DIM, dtype=np.float32).reshape(7, repool.HIDDEN_DIM)
    plen = 4
    want_union = repool.pool_union(hidden, plen, repool.union_positions(row))
    assignment = repool.block_assignment(row, cfg)
    e3 = {f"block_{b}": v.copy() for b, v in
          repool.pool_blocks(hidden, plen, assignment).items()}
    k = next(iter(e3))
    e3[k] = e3[k] + np.float32(1.0)
    with pytest.raises(tail.TailStop) as exc:
        tail.assert_h1_anchors(unit_id="u", pre_cast_hidden=hidden, prompt_len=plen,
                               row=row, cfg=cfg, frozen_union=want_union, frozen_e3=e3)
    assert exc.value.gate == "h1_e3_anchor"


def test_h1_missing_e3_block_in_cache_is_caught():
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    row = _h1_row()
    hidden = np.arange(7 * repool.HIDDEN_DIM, dtype=np.float32).reshape(7, repool.HIDDEN_DIM)
    plen = 4
    want_union = repool.pool_union(hidden, plen, repool.union_positions(row))
    with pytest.raises(tail.TailStop) as exc:
        tail.assert_h1_anchors(unit_id="u", pre_cast_hidden=hidden, prompt_len=plen,
                               row=row, cfg=cfg, frozen_union=want_union, frozen_e3={})
    assert exc.value.gate == "h1_e3_anchor"


def test_mcq_anchor_exact_and_mismatch():
    hidden = np.arange(7 * repool.HIDDEN_DIM, dtype=np.float32).reshape(7, repool.HIDDEN_DIM)
    want = hidden[4 + 1]
    out = tail.assert_mcq_anchor(unit_id="m", pre_cast_hidden=hidden, prompt_len=4,
                                 letter_token_index_local=1, frozen_letter_hidden=want)
    assert out["exact"] is True and out["letter_row_index"] == 5
    with pytest.raises(tail.TailStop) as exc:
        tail.assert_mcq_anchor(unit_id="m", pre_cast_hidden=hidden, prompt_len=4,
                               letter_token_index_local=1,
                               frozen_letter_hidden=want + np.float32(1.0))
    assert exc.value.gate == "mcq_letter_anchor"


def test_mcq_anchor_index_out_of_range():
    hidden = np.zeros((5, repool.HIDDEN_DIM), dtype=np.float32)
    with pytest.raises(tail.TailStop) as exc:
        tail.assert_mcq_anchor(unit_id="m", pre_cast_hidden=hidden, prompt_len=4,
                               letter_token_index_local=99,
                               frozen_letter_hidden=hidden[0])
    assert exc.value.gate == "mcq_letter_index"


# ------------------------------------------------------------- runner
def test_runner_pins_the_formal_scale(runner):
    assert runner.N_H1_EXPECTED == 8664
    assert runner.N_MCQ_EXPECTED == 1750
    assert runner.N_E3_PAIRS_EXPECTED == 9916
    assert runner.N_H1_EXPECTED + runner.N_MCQ_EXPECTED == 10414


def test_runner_refuses_when_formal_artifacts_exist(runner, tmp_path, monkeypatch):
    tdir = tmp_path / "tail_stabilized"
    tdir.mkdir()
    monkeypatch.setattr(runner, "TAIL_DIR", tdir)
    monkeypatch.setattr(runner, "FORBIDDEN_FORMAL", (tdir / "tail_manifest.jsonl",))
    runner.assert_formal_absent()                      # 干净时通过
    (tdir / "tail_manifest.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.ScopeViolation, match="不得创建正式 tail 产物"):
        runner.assert_formal_absent()


def test_runner_refuses_formal_segment_telemetry(runner, tmp_path, monkeypatch):
    tdir = tmp_path / "tail_stabilized"
    tdir.mkdir()
    monkeypatch.setattr(runner, "TAIL_DIR", tdir)
    monkeypatch.setattr(runner, "FORBIDDEN_FORMAL", ())
    (tdir / "tail_seg001_gpu_telemetry.csv").write_text("h\n", encoding="utf-8")
    with pytest.raises(runner.ScopeViolation):
        runner.assert_formal_absent()


def test_runner_publish_refuses_overwrite(runner, tmp_path):
    p = tmp_path / "o.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.ScopeViolation, match="拒绝覆盖"):
        runner.publish_json_exclusive(p, {"a": 1})
    p.unlink()
    stale = p.with_suffix(".json.tmp")
    stale.write_text("x", encoding="utf-8")
    with pytest.raises(runner.ScopeViolation, match="陈旧临时文件"):
        runner.publish_json_exclusive(p, {"a": 1})
    assert stale.read_text(encoding="utf-8") == "x"


def test_runner_telemetry_is_exclusive_create(runner, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.start_telemetry(p)
    assert p.read_text(encoding="utf-8") == "keep"


# --------------------------------------------------------------- 分段规则
def test_segment_plan_respects_time_budget_and_cap(runner):
    # 0.5 s/trace → 1200/0.5 = 2400 < 4000，按时间预算取 2400
    plan = runner._segment_plan("h1", 8664, 0.5)
    assert plan["seg_len"] == 2400
    assert plan["n_segments"] == 4
    assert sum(plan["segment_sizes"]) == 8664
    assert plan["segment_sizes"][-1] == 8664 - 2400 * 3        # 最后一段取余


def test_segment_plan_caps_at_4000(runner):
    # 0.1 s/trace → 12000，被 4000 上限截断
    plan = runner._segment_plan("h1", 8664, 0.1)
    assert plan["seg_len"] == runner.SEG_MAX_TRACES == 4000
    assert plan["n_segments"] == 3
    assert sum(plan["segment_sizes"]) == 8664


def test_segment_plan_single_segment_when_small(runner):
    plan = runner._segment_plan("mcq", 1750, 0.5)
    assert plan["n_segments"] == 1
    assert plan["segment_sizes"] == [1750]


def test_segment_plan_arms_are_not_mixed(runner):
    h1 = runner._segment_plan("h1", 8664, 0.4)
    mcq = runner._segment_plan("mcq", 1750, 0.4)
    assert h1["arm"] == "h1" and mcq["arm"] == "mcq"
    assert h1["n_total"] == 8664 and mcq["n_total"] == 1750


# --------------------------------------------------------- 确定性分层取样
def _fake_h1_rows(n=200):
    rows = []
    for i in range(n):
        rows.append({"example_id": f"e{i:04d}", "sample_type": "greedy",
                     "sample_index": 0, "trace_id": f"e{i:04d}__greedy_0",
                     "completion": "w " * (5 + i % 40),
                     "id_token_positions_local": [0, 1],
                     "mentions": [{"normalized": f"T{1000 + i % 50}",
                                   "token_indices": [0, 1], "family": "attack"}]})
    return rows


def test_h1_sampling_is_deterministic_and_order_independent(runner):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    cfg = V3AnalysisConfig()
    rows = _fake_h1_rows()
    a = runner.h1_strata_sample(rows, cfg, 32)
    b = runner.h1_strata_sample(list(reversed(rows)), cfg, 32)
    assert a["unit_ids"] == b["unit_ids"]
    assert a["selection_sha256"] == b["selection_sha256"]
    assert a["n"] <= 32
    assert "n_e3_blocks" in a["sort_key"]


def test_mcq_sampling_is_deterministic_and_spans_length_range(runner):
    rows = [{"example_id": f"m{i:04d}", "raw_completion": "x" * (1 + i % 30)}
            for i in range(300)]
    a = runner.mcq_strata_sample(rows, 32)
    b = runner.mcq_strata_sample(list(reversed(rows)), 32)
    assert a["example_ids"] == b["example_ids"]
    assert a["n"] == 32
    lens = sorted(len(r["raw_completion"]) for r in rows)
    picked = {r["example_id"]: len(r["raw_completion"]) for r in rows}
    got = sorted(picked[e] for e in a["example_ids"])
    assert got[0] == lens[0] and got[-1] == lens[-1]     # 含首尾


def test_old_e3_benchmark_is_only_corroborating(runner):
    src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    assert "只能作旁证" in src
    assert "不得**直接用于分段" in src


# ══════════════════════════════════════════════════════════ 正式执行路径
# 全部用 tmp_path 下的替身状态，**不加载模型、不碰真实 outputs**。
# `_real_outputs_untouched`（逐测试比字节）与 `_heavy_outputs_untouched`
# （模块级比文件数 / 总字节 / 最新 mtime）一起兜底。

def _heavy_snapshot() -> dict:
    out = {}
    for rel in ("production", "repool", "shadow", "tail_stabilized"):
        d = _REAL_OUT / rel
        if not d.exists():
            out[rel] = None
            continue
        n = total = newest = 0
        stack = [str(d)]
        while stack:
            for entry in os.scandir(stack.pop()):
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                st = entry.stat()
                n += 1
                total += st.st_size
                newest = max(newest, st.st_mtime_ns)
        out[rel] = (n, total, newest)
    return out


@pytest.fixture(scope="module", autouse=True)
def _heavy_outputs_untouched():
    """`production/` `repool/` `shadow/` `tail_stabilized/` 都是大目录，逐测试算
    SHA 太慢 —— 改为模块级比对文件数 / 总字节 / 最新 mtime，任何写入都会改动其中之一。"""
    before = _heavy_snapshot()
    yield
    after = _heavy_snapshot()
    assert after == before, (
        "测试写了真实产物目录！"
        f"{[k for k in after if after[k] != before[k]]}")


def _h1_named(eid: str, positions=(0, 1)) -> dict:
    return {"example_id": eid, "sample_type": "greedy", "sample_index": 0,
            "trace_id": f"{eid}__greedy_0", "prompt_hash": PROMPT_SHA,
            "run_hash": "runhash0", "completion": "c c c",
            "id_token_positions_local": list(positions),
            "mentions": [{"normalized": f"T{1000 + len(eid)}",
                          "token_indices": list(positions), "family": "attack"}]}


def _mcq_named(eid: str) -> dict:
    return {"example_id": eid, "prompt_hash": PROMPT_SHA, "run_hash": "runhash0",
            "rendered_prompt": PROMPT, "raw_completion": "A",
            "letter_token_index_local": 0, "letter_token_id": 32,
            "letter_token_form": "bare", "eligible_for_primary": True}


def _cfg():
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    return V3AnalysisConfig()


#: 与 `_h1_named` 的 positions=(0,1) 一致：3 个 completion token、2 个 identifier 位置。
_H1_NTOK, _H1_NID = 3, 2
_H1_STABLE = {
    "f5_completion_token_count": _H1_NTOK,
    "f5_lengthnorm_logprob": -1.5,
    "f5_completion_perplexity": float(np.exp(1.5)),
    "f5_id_token_ratio": _H1_NID / _H1_NTOK,
    "f5_id_logprob_mean": -1.2,
    "f5_id_logprob_min": -2.0,
    "f5_id_token_entropy_mean": 0.5,
    "f5_first_id_token_rank": 1,
}
_MCQ_STABLE = {"f5_label_margin": 1.0, "f5_label_entropy": 0.5,
               "f5_full_entropy": 2.0, "f5_letter_token_logprob": -0.3}


def _fake_h1_out(r, row: dict, cfg) -> dict:
    """一条 H1 单元**已算好**的结果：E3 arrays 与 stable F5 出自同一次 forward。

    记录本体走**真实**的 `tail_record()`，因此写盘前的 `assert_tail_f5_row` 与
    shadow 冻结验收都在测试里真跑一遍，而不是被一个手搓 dict 绕过去。
    """
    uid = repool.derive_unit_id(row)
    assignment = repool.block_assignment(row, cfg)
    arrays = {f"block_{b}": np.full(repool.HIDDEN_DIM, float(b) + 0.5, dtype=np.float32)
              for b in assignment}
    meta = {"trace_id": uid, "prompt_hash": row["prompt_hash"],
            "source_fingerprint_id": row.get("run_hash"),
            "token_positions": {str(b): list(p) for b, p in assignment.items()}}
    record = r.tail_record(row, "h1", dict(_H1_STABLE), provenance={
        "unit_id": uid, "prompt_hash": row["prompt_hash"],
        "source_run_hash": row.get("run_hash"), "n_completion_tokens": _H1_NTOK,
        "n_identifier_positions": _H1_NID, "n_e3_blocks": len(assignment),
        "position_digest": repool.position_digest(assignment),
        "post_cast_residual_dtype": "torch.float32"})
    return {"unit_id": uid, "arrays": arrays, "npz_meta": meta,
            "n_blocks": len(assignment), "block_ids": sorted(assignment),
            "position_digest": repool.position_digest(assignment),
            "n_completion_tokens": _H1_NTOK, "record": record}


def _fake_mcq_out(r, row: dict) -> dict:
    record = r.tail_record(row, "mcq", dict(_MCQ_STABLE), provenance={
        "example_id": row["example_id"], "prompt_hash": row["prompt_hash"],
        "letter_token_index_local": 0, "letter_token_id": 32,
        "letter_token_form": "bare", "letter_row_index": 4,
        "source_run_hash": row.get("run_hash"), "n_completion_tokens": 1,
        "post_cast_residual_dtype": "torch.float32"})
    return {"unit_id": row["example_id"], "arrays": None, "npz_meta": None,
            "n_completion_tokens": 1, "letter_row_index": 4, "record": record}


@pytest.fixture
def formal(runner, tmp_path, monkeypatch):
    """把 runner 的正式命名空间与规模常量整体重定向到 tmp_path 的替身。"""
    cfg = _cfg()
    h1 = [_h1_named(f"e{i:02d}") for i in range(3)]
    mcq = [_mcq_named(f"m{i:02d}") for i in range(2)]
    monkeypatch.setattr(runner, "TAIL_DIR", tmp_path / "tail_stabilized")
    monkeypatch.setattr(runner, "load_analyzable",
                        lambda arm: h1 if arm == "h1" else mcq)
    monkeypatch.setattr(runner, "N_H1_EXPECTED", len(h1))
    monkeypatch.setattr(runner, "N_MCQ_EXPECTED", len(mcq))
    monkeypatch.setattr(runner, "N_E3_PAIRS_EXPECTED",
                        sum(len(repool.expected_blocks(r, cfg)) for r in h1))
    monkeypatch.setattr(runner, "source_freeze", lambda: {"production/x": "a" * 64})
    return {"runner": runner, "cfg": cfg, "h1": h1, "mcq": mcq,
            "paths": runner.tail_paths()}


def _write_all(f, *, arms=("h1", "mcq"), skip=()):
    """按正式写序落盘指定臂的全部单元。"""
    r, cfg = f["runner"], f["cfg"]
    for arm in arms:
        for row in f[arm]:
            uid = repool.derive_unit_id(row) if arm == "h1" else row["example_id"]
            if uid in skip:
                continue
            out = _fake_h1_out(r, row, cfg) if arm == "h1" else _fake_mcq_out(r, row)
            r.process_unit(arm=arm, uid=uid, out=out, paths=f["paths"], row=row,
                           cfg=cfg, segment="tail_seg001")


# ----------------------------------------------- 一次 forward，两侧同时落盘
def test_h1_unit_writes_stable_f5_and_e3_npz_from_one_forward(formal):
    r, f = formal["runner"], formal
    row = f["h1"][0]
    uid = repool.derive_unit_id(row)
    ev = r.process_unit(arm="h1", uid=uid, out=_fake_h1_out(r, row, f["cfg"]),
                        paths=f["paths"], row=row, cfg=f["cfg"], segment="tail_seg001")

    npz = f["paths"]["e3_dir"] / f"{uid}.npz"
    assert npz.exists(), "E3 NPZ 未写"
    rows = r.read_jsonl_strict(f["paths"]["h1"], what="h1")
    assert len(rows) == 1 and rows[0][0]["trace_id"] == uid, "stable F5 行未写"
    # 同一条 done 事件同时携带两侧摘要 —— 它们出自同一个单元
    assert ev["f5_sha256"] == rows[0][1]
    assert ev["npz_sha256"] == r.npz_digest(npz)["sha256"]
    assert ev["n_blocks"] == len(repool.block_assignment(row, f["cfg"]))
    assert ev["position_digest"]


def test_tail_h1_unit_hands_arrays_to_the_writer(runner):
    """曾经的缺口：`tail_h1_unit` 只回 n_blocks/position_digest，把向量丢了。"""
    src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    assert '"arrays": anch["arrays"]' in src
    assert '"npz_meta": meta' in src


def test_tail_record_keeps_shadow_naming_and_never_claims_primary(runner):
    from recover_attention.evaluation import v3_shadow_f5 as sf
    row = _h1_named("e00")
    rec = runner.tail_record(row, "h1", {"f5_lengthnorm_logprob": -1.0},
                             provenance={"unit_id": "u"})
    assert rec["interpretation"] == sf.SENSITIVITY_LABEL
    assert rec["deviation_class"] == sf.DEVIATION_LABEL
    assert rec["variant"] == runner.TAIL_VARIANT
    assert "不得" in rec["not_primary"]
    assert rec["provenance"]["path"] == runner.TAIL_VARIANT


# ------------------------------------------------------------- 写序与恢复
def test_outputs_written_but_done_missing_is_repaired_without_gpu(formal):
    """崩在最后一步：NPZ + F5 都在、manifest 无 done → 校验后补记，不重算数值。"""
    r, f = formal["runner"], formal
    _write_all(f)
    target = repool.derive_unit_id(f["h1"][-1])
    kept = [ev for ev in r.read_manifest_events(f["paths"]["manifest"])
            if ev.get("unit_id") != target]
    f["paths"]["manifest"].write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in kept),
        encoding="utf-8", newline="")

    plan = r.plan_resume(arm="h1", unit_ids=r.unit_ids_for("h1", f["h1"]),
                         events=r.read_manifest_events(f["paths"]["manifest"]),
                         f5_index=r.read_f5_index(f["paths"]["h1"], "h1"),
                         npz_index=r.npz_index_on_disk(f["paths"]["e3_dir"]))
    assert plan["repair"] == [target] and not plan["problems"]

    rc = r.run_formal("h1", 4, "definitely-not-a-model")     # 无待跑 → 不加载模型
    assert rc == 0
    assert not list(f["paths"]["dir"].glob("tail_seg*")), "不应启动遥测/加载模型"
    last = r.read_manifest_events(f["paths"]["manifest"])[-1]
    assert last["unit_id"] == target and last["status"] == "done"
    assert last["repaired"] is True


def test_orphan_npz_is_rerun_and_compared_bitwise_not_deleted(formal):
    r, f = formal["runner"], formal
    row = f["h1"][0]
    uid = repool.derive_unit_id(row)
    out = _fake_h1_out(r, row, f["cfg"])
    npz = f["paths"]["e3_dir"] / f"{uid}.npz"
    r.npz_atomic(npz, out["arrays"], out["npz_meta"])        # 只有 NPZ，没有 F5 行
    before = npz.read_bytes()

    plan = r.plan_resume(arm="h1", unit_ids=[uid], events=[], f5_index={},
                         npz_index=r.npz_index_on_disk(f["paths"]["e3_dir"]))
    assert plan["rerun_orphan"] == [uid] and not plan["problems"]

    ev = r.process_unit(arm="h1", uid=uid, out=out, paths=f["paths"], row=row,
                        cfg=f["cfg"], segment="tail_seg002", orphan=True)
    assert npz.read_bytes() == before, "孤儿 NPZ 被改写了"
    assert ev["npz_sha256"] == hashlib.sha256(before).hexdigest()


def test_orphan_npz_mismatch_stops_and_keeps_the_file(formal):
    r, f = formal["runner"], formal
    row = f["h1"][0]
    uid = repool.derive_unit_id(row)
    out = _fake_h1_out(r, row, f["cfg"])
    npz = f["paths"]["e3_dir"] / f"{uid}.npz"
    r.npz_atomic(npz, out["arrays"], out["npz_meta"])
    before = npz.read_bytes()

    changed = dict(out)
    changed["arrays"] = {k: v + np.float32(1.0) for k, v in out["arrays"].items()}
    with pytest.raises(tail.TailStop) as exc:
        r.process_unit(arm="h1", uid=uid, out=changed, paths=f["paths"], row=row,
                       cfg=f["cfg"], segment="tail_seg002", orphan=True)
    assert exc.value.gate == "orphan_npz_mismatch"
    assert npz.read_bytes() == before, "失败时不得删除或改写现场"
    assert not f["paths"]["h1"].exists(), "比对失败后不得写 F5 行"


def test_npz_atomic_refuses_overwrite(formal):
    r, f = formal["runner"], formal
    out = _fake_h1_out(r, f["h1"][0], f["cfg"])
    p = f["paths"]["e3_dir"] / "x.npz"
    r.npz_atomic(p, out["arrays"], out["npz_meta"])
    with pytest.raises(r.ScopeViolation, match="拒绝覆盖"):
        r.npz_atomic(p, out["arrays"], out["npz_meta"])


def test_append_only_prefix_is_never_rewritten(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    prefix = f["paths"]["manifest"].read_bytes()
    mcq_rows = f["paths"]["mcq"].read_bytes()
    r.process_unit(arm="h1", uid=repool.derive_unit_id(f["h1"][0]),
                   out=_fake_h1_out(r, f["h1"][0], f["cfg"]), paths=f["paths"],
                   row=f["h1"][0], cfg=f["cfg"], segment="tail_seg002")
    assert f["paths"]["manifest"].read_bytes().startswith(prefix)
    assert f["paths"]["mcq"].read_bytes() == mcq_rows
    assert f["paths"]["manifest"].read_bytes().endswith(b"\n")


# ------------------------------------------------- done 与输出不符即拒绝
@pytest.mark.parametrize("field,bad", [
    ("f5_sha256", "0" * 64), ("npz_sha256", "0" * 64), ("npz_bytes", 1),
    ("n_blocks", 99), ("position_digest", "deadbeef")])
def test_done_event_must_match_outputs(formal, field, bad):
    r, f = formal["runner"], formal
    _write_all(f, arms=("h1",))
    events = r.read_manifest_events(f["paths"]["manifest"])
    events[0][field] = bad
    plan = r.plan_resume(arm="h1", unit_ids=r.unit_ids_for("h1", f["h1"]),
                         events=events,
                         f5_index=r.read_f5_index(f["paths"]["h1"], "h1"),
                         npz_index=r.npz_index_on_disk(f["paths"]["e3_dir"]))
    assert plan["problems"], f"{field} 被篡改却未被发现"
    assert len(plan["done"]) == len(f["h1"]) - 1


def test_done_but_output_missing_is_rejected(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("h1",))
    events = r.read_manifest_events(f["paths"]["manifest"])
    npz_index = r.npz_index_on_disk(f["paths"]["e3_dir"])
    npz_index.pop(events[0]["unit_id"])                      # 模拟 NPZ 丢失
    plan = r.plan_resume(arm="h1", unit_ids=r.unit_ids_for("h1", f["h1"]),
                         events=events,
                         f5_index=r.read_f5_index(f["paths"]["h1"], "h1"),
                         npz_index=npz_index)
    assert any("NPZ 缺失" in p for p in plan["problems"])


def test_f5_row_without_npz_is_a_recovery_stop_not_a_rerun(formal):
    """写序被破坏（有 F5 行、无 NPZ）不能当成待跑 —— 重跑会写出重复 F5 行。"""
    r = formal["runner"]
    plan = r.plan_resume(arm="h1", unit_ids=["u1"], events=[],
                         f5_index={"u1": "a" * 64}, npz_index={})
    assert plan["problems"] and not plan["pending"] and not plan["rerun_orphan"]


# ------------------------------------------------------- failed 不自动重试
def test_failed_blocks_the_next_run_and_is_never_requeued(formal):
    r, f = formal["runner"], formal
    uid = repool.derive_unit_id(f["h1"][-1])
    _write_all(f, arms=("h1",), skip={uid})
    r.durable_append(f["paths"]["manifest"], r.failed_event(
        arm="h1", uid=uid, segment="tail_seg001",
        exc=tail.TailStop("boom", unit_id=uid, gate="pre_cast_layer19",
                          diagnostic={"n_nonfinite_values": 7})))

    plan = r.plan_resume(arm="h1", unit_ids=r.unit_ids_for("h1", f["h1"]),
                         events=r.read_manifest_events(f["paths"]["manifest"]),
                         f5_index=r.read_f5_index(f["paths"]["h1"], "h1"),
                         npz_index=r.npz_index_on_disk(f["paths"]["e3_dir"]))
    assert plan["failed"] == [uid]
    assert uid not in plan["todo_gpu"] and uid not in plan["pending"]

    with pytest.raises(r.TailRecoveryStop, match="不自动重试"):
        r.run_formal("h1", 4, "definitely-not-a-model")
    assert not list(f["paths"]["dir"].glob("tail_seg*")), "拒绝时不得启动遥测"


def test_failed_is_a_sensitivity_path_consistency_failure_not_population_attrition(runner):
    """`failed` 不得被读成 population 的 runtime technical attrition ——
    本路径是 sensitivity，**不改变** confirmatory population。"""
    ev = runner.failed_event(arm="h1", uid="u", segment="tail_seg001",
                             exc=tail.TailStop("x", unit_id="u", gate="stable_logits",
                                               diagnostic={}))
    assert ev["status"] == "failed"
    blob = json.dumps(ev, ensure_ascii=False)
    assert "engineering consistency failure of the tail sensitivity path" in blob
    assert "不改变** confirmatory population" in blob
    assert "不是**预注册允许的科学排除" in blob
    assert "不**新增 population 的 runtime technical attrition" in blob
    assert "excluded" not in blob


# ------------------------------------------- completion report 不得提前出现
def test_completion_report_absent_until_both_arms_close(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("h1",))                              # MCQ 还没跑
    status = r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])
    assert status["complete"] is False
    assert not f["paths"]["report"].exists()
    assert status["arms"]["h1"]["complete"] is True
    assert status["arms"]["mcq"]["complete"] is False

    _write_all(f, arms=("mcq",))
    status = r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])
    assert status["complete"] is True
    assert f["paths"]["report"].exists()
    assert status["e3_trace_block_pairs"] == r.N_E3_PAIRS_EXPECTED
    assert status["interpretation"] == "stable_logit_shadow_sensitivity"


def test_completion_report_blocked_by_a_single_failed_unit(formal):
    r, f = formal["runner"], formal
    last = repool.derive_unit_id(f["h1"][-1])
    _write_all(f, skip={last})
    r.durable_append(f["paths"]["manifest"], r.failed_event(
        arm="h1", uid=last, segment="tail_seg001",
        exc=tail.TailStop("x", unit_id=last, gate="tail_final_hidden", diagnostic={})))
    status = r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])
    assert status["complete"] is False and not f["paths"]["report"].exists()
    assert status["arms"]["h1"]["n_failed"] == 1


# ------------------------------------------------ 源 SHA 漂移在加载前拒绝
def test_source_drift_is_refused_before_the_model_is_loaded(formal):
    r, f = formal["runner"], formal
    f["paths"]["dir"].mkdir(parents=True, exist_ok=True)
    f["paths"]["fingerprint"].write_text(
        json.dumps({"source_freeze": {"production/x": "b" * 64}}), encoding="utf-8")
    with pytest.raises(r.TailRecoveryStop, match="漂移"):
        r.run_formal("h1", 1, "definitely-not-a-model")
    assert not list(f["paths"]["dir"].glob("tail_seg*"))
    assert not f["paths"]["manifest"].exists()


def test_source_freeze_covers_finalized_fingerprints_and_e3_cache(runner, tmp_path,
                                                                  monkeypatch):
    prod, rep = tmp_path / "production", tmp_path / "repool"
    (prod / "h1_unit_hidden").mkdir(parents=True)
    (rep / "h1_e3_block_hidden").mkdir(parents=True)
    for name in runner.FREEZE_PROD_FILES:
        (prod / name).write_bytes(name.encode())
    for name in runner.FREEZE_REPOOL_FILES:
        (rep / name).write_bytes(name.encode())
    (prod / "h1_unit_hidden" / "a.npy").write_bytes(b"x")
    (rep / "h1_e3_block_hidden" / "a.npz").write_bytes(b"y")
    monkeypatch.setattr(runner, "PROD", prod)
    monkeypatch.setattr(runner, "REPOOL", rep)

    fz = runner.source_freeze()
    assert len(fz) == 12                       # 六项 finalized + 两指纹 + 三 E3 + 一树
    for name in ("h1_completion_records.jsonl", "h1_hidden.npz",
                 "h1_finalization_report.json", "mcq_completion_records.jsonl",
                 "mcq_hidden.npz", "mcq_finalization_report.json",
                 "h1_run_fingerprint.json", "mcq_run_fingerprint.json"):
        assert f"production/{name}" in fz
    assert "production/h1_unit_hidden__tree_digest" in fz
    assert "repool/e3_repool_manifest.jsonl" in fz
    assert "repool/e3_repool_report.json" in fz
    assert "repool/h1_e3_block_hidden__tree_digest" in fz

    before = dict(fz)
    (rep / "h1_e3_block_hidden" / "a.npz").write_bytes(b"z")   # 既有 E3 cache 被改
    assert runner.source_freeze() != before


# --------------------------------------------- 目标 / 遥测存在时拒绝覆盖
def test_segment_numbering_increments_and_is_exclusive(formal):
    r, f = formal["runner"], formal
    f["paths"]["dir"].mkdir(parents=True, exist_ok=True)
    assert r.next_segment_path().name == "tail_seg001_gpu_telemetry.csv"
    (f["paths"]["dir"] / "tail_seg001_gpu_telemetry.csv").write_text("h\n")
    (f["paths"]["dir"] / "tail_seg002_gpu_telemetry.csv").write_text("h\n")
    assert r.next_segment_path().name == "tail_seg003_gpu_telemetry.csv"
    with pytest.raises(FileExistsError):
        r.start_telemetry(f["paths"]["dir"] / "tail_seg001_gpu_telemetry.csv")


def test_fingerprint_and_report_are_create_only(formal):
    r, f = formal["runner"], formal
    r.publish_json_exclusive(f["paths"]["fingerprint"], {"a": 1})
    with pytest.raises(r.ScopeViolation, match="拒绝覆盖"):
        r.publish_json_exclusive(f["paths"]["fingerprint"], {"a": 2})
    assert json.loads(f["paths"]["fingerprint"].read_text(encoding="utf-8")) == {"a": 1}


# ------------------------------------------------------------ 截断与重复
def test_truncated_result_tail_is_refused_not_silently_dropped(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    with f["paths"]["mcq"].open("ab") as fh:
        fh.write(b'{"example_id": "m9", "arm"')          # 半行
    with pytest.raises(r.TailRecoveryStop, match="半行"):
        r.read_f5_index(f["paths"]["mcq"], "mcq")


def test_manifest_half_line_is_refused_and_never_appended_after(formal):
    """曾经的**不安全契约**：忽略半行但不截断，随后 append 会焊死一条永久坏行。

    现在两道防线：读取直接停机；`durable_append` 自己也拒绝接在半行后面。
    """
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    with f["paths"]["manifest"].open("ab") as fh:
        fh.write(b'{"kind": "unit"')                     # 崩溃留下的未终止尾行
    before = f["paths"]["manifest"].read_bytes()

    with pytest.raises(r.TailRecoveryStop, match="半行"):
        r.read_manifest_events(f["paths"]["manifest"])
    with pytest.raises(r.TailRecoveryStop, match="拒绝 append"):
        r.durable_append(f["paths"]["manifest"], {"kind": "unit"})
    assert f["paths"]["manifest"].read_bytes() == before, "既不截断也不追加"

    with pytest.raises(r.TailRecoveryStop, match="半行"):
        r.run_formal("mcq", 4, "definitely-not-a-model")
    assert f["paths"]["manifest"].read_bytes() == before


def test_manifest_mid_file_damage_is_refused(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    with f["paths"]["manifest"].open("ab") as fh:
        fh.write(b'{"kind": "unit"\n{"kind": "unit", "arm": "mcq"}\n')
    with pytest.raises(r.TailRecoveryStop, match="损坏"):
        r.read_manifest_events(f["paths"]["manifest"])


def test_duplicate_f5_rows_are_refused(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    dup = r.read_jsonl_strict(f["paths"]["mcq"], what="mcq")[0][0]
    r.durable_append(f["paths"]["mcq"], dup)
    with pytest.raises(r.TailRecoveryStop, match="多行"):
        r.read_f5_index(f["paths"]["mcq"], "mcq")


# ----------------------------------------------------------- 顺序与分段
def test_unit_order_is_sorted_and_independent_of_record_order(runner):
    rows = [_h1_named(f"e{i:02d}") for i in range(5)]
    a = runner.unit_ids_for("h1", rows)
    b = runner.unit_ids_for("h1", list(reversed(rows)))
    assert a == b == sorted(a)
    assert "sorted(unit_id)" in runner.UNIT_ORDER_RULE
    with pytest.raises(runner.TailRecoveryStop, match="重复 unit_id"):
        runner.unit_ids_for("h1", rows + [rows[0]])


@pytest.mark.parametrize("rate", [0.1780, 0.2119])
def test_measured_rates_both_give_the_planned_h1_segments(runner, rate):
    """两种口径（128 条稳态 / 32 条保守）都触发 4,000 上限，分段计划相同。"""
    plan = runner._segment_plan("h1", 8664, rate)
    assert plan["seg_len"] == 4000
    assert plan["segment_sizes"] == [4000, 4000, 664]
    assert plan["seg_len_by_time_budget"] > 4000            # 被上限截断，不是被时间


@pytest.mark.parametrize("rate", [0.2125, 0.2159])
def test_measured_rates_give_a_single_mcq_segment(runner, rate):
    plan = runner._segment_plan("mcq", 1750, rate)
    assert plan["segment_sizes"] == [1750] and plan["n_segments"] == 1


def test_run_requires_arm_and_explicit_segment_size(runner):
    for argv in (["--run"], ["--run", "--arm", "h1"],
                 ["--run", "--arm", "h1", "--benchmark"], []):
        with pytest.raises(SystemExit):
            runner.main(argv)


# ═══════════════════════════════════════════════ correction sprint 的四项
# ① manifest 半行 ② 首段模型身份 ③ 遥测存活 ④ 集合严格闭合 + 恢复路径收紧

# ---------------------------------------------- ② 首段模型身份 vs 冻结指纹
def _fp(model_hash="m", tok_hash="t", tuple_idx=20):
    """一份最小的 run fingerprint 替身（身份哈希由 payload 内容决定）。"""
    return {"model": {"model_path": "D:/models/X", "model_hash": "ignored",
                      "arch": model_hash, "backend": {"load_in_8bit": True}},
            "tokenizer": {"name_or_path": "D:/models/X", "tokenizer_hash": "ignored",
                          "vocab_size": tok_hash},
            "hidden_index": {"num_layers": 28, "block_index_zero_based": 19,
                             "hidden_states_tuple_index": tuple_idx}}


def _live_from(runner, fp, cache_mode="chunked"):
    return {**runner.frozen_identity(fp), "projection_cache_mode": cache_mode}


def test_model_identity_is_pinned_to_both_frozen_production_fingerprints(runner):
    good = _fp()
    live = _live_from(runner, good)
    runner.assert_model_identity(live=live, frozen_h1=good, frozen_mcq=_fp())

    # 传错 --model-path 会换掉 backend/arch → 现场身份对不上冻结值
    wrong = _live_from(runner, _fp(model_hash="OTHER"))
    with pytest.raises(runner.TailRecoveryStop, match="--model-path 传错"):
        runner.assert_model_identity(live=wrong, frozen_h1=good, frozen_mcq=_fp())

    wrong_tok = _live_from(runner, _fp(tok_hash="OTHER"))
    with pytest.raises(runner.TailRecoveryStop, match="tokenizer_identity_hash"):
        runner.assert_model_identity(live=wrong_tok, frozen_h1=good, frozen_mcq=_fp())

    wrong_idx = _live_from(runner, _fp(tuple_idx=21))
    with pytest.raises(runner.TailRecoveryStop, match="hidden_index"):
        runner.assert_model_identity(live=wrong_idx, frozen_h1=good, frozen_mcq=_fp())


def test_two_frozen_fingerprints_must_agree_with_each_other(runner):
    good = _fp()
    with pytest.raises(runner.TailRecoveryStop, match="彼此不一致"):
        runner.assert_model_identity(live=_live_from(runner, good), frozen_h1=good,
                                     frozen_mcq=_fp(tuple_idx=21))


@pytest.mark.local_artifacts('outputs/stage1/production/h1_run_fingerprint.json',
                             'outputs/stage1/production/mcq_run_fingerprint.json')
def test_real_frozen_fingerprints_agree_and_pin_the_frozen_hidden_index():
    """真实的两份 production fingerprint 必须互相一致，且钉在 block 19 / tuple 20。"""
    prod = _REAL_OUT / "production"
    a = json.loads((prod / "h1_run_fingerprint.json").read_text(encoding="utf-8"))
    b = json.loads((prod / "mcq_run_fingerprint.json").read_text(encoding="utf-8"))
    assert repool.model_identity_hash(a["model"]) == repool.model_identity_hash(b["model"])
    assert (repool.tokenizer_identity_hash(a["tokenizer"])
            == repool.tokenizer_identity_hash(b["tokenizer"]))
    assert a["hidden_index"] == b["hidden_index"]
    assert a["hidden_index"]["block_index_zero_based"] == tail.CAST_BLOCK
    assert a["hidden_index"]["hidden_states_tuple_index"] == tail.FROZEN_TUPLE_IDX


# ------------------------------------------------------------ ③ 遥测存活
class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 4242

    def poll(self):
        return self.returncode


def _tel(runner, path, proc):
    return runner.Telemetry(proc, None, Path(path))


def test_wait_for_telemetry_returns_once_the_first_sample_lands(runner, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("header\n", encoding="utf-8")
    tel = _tel(runner, p, _FakeProc())
    calls = {"n": 0}

    def sleep(_s):
        calls["n"] += 1
        if calls["n"] == 2:
            p.write_text("header\nrow1\n", encoding="utf-8")

    assert runner.wait_for_telemetry(tel, sleep=sleep) == 1
    assert calls["n"] == 2


def test_wait_for_telemetry_refuses_when_sampler_dies_before_first_sample(runner, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("header\n", encoding="utf-8")
    tel = _tel(runner, p, _FakeProc(returncode=9))
    with pytest.raises(runner.TailRecoveryStop, match="首个采样点前退出"):
        runner.wait_for_telemetry(tel, sleep=lambda _s: None)


def test_wait_for_telemetry_times_out_rather_than_loading_the_model(runner, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("header\n", encoding="utf-8")
    tel = _tel(runner, p, _FakeProc())
    ticks = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    with pytest.raises(runner.TailRecoveryStop, match="未落下首个采样点"):
        runner.wait_for_telemetry(tel, timeout=1.0, sleep=lambda _s: None,
                                  clock=lambda: next(ticks))


def test_check_telemetry_alive_stops_the_segment_when_the_sampler_exits(runner, tmp_path):
    p = tmp_path / "t.csv"
    runner.check_telemetry_alive(_tel(runner, p, _FakeProc()))     # 活着 → 通过
    with pytest.raises(runner.TailRecoveryStop, match="运行途中退出"):
        runner.check_telemetry_alive(_tel(runner, p, _FakeProc(returncode=1)))


def test_stop_telemetry_uses_only_the_recorded_handle(runner, tmp_path):
    """只按句柄停止，不按进程名批量杀 —— 源码里不得出现 taskkill / pkill。"""
    src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    for banned in ("taskkill", "pkill", "killall", 'Popen(["nvidia-smi"], '):
        assert banned not in src

    class _P(_FakeProc):
        def __init__(self):
            super().__init__()
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    class _F:
        closed = False

        def close(self):
            self.closed = True

    proc, fh = _P(), _F()
    runner.stop_telemetry(runner.Telemetry(proc, fh, tmp_path / "t.csv"))
    assert proc.terminated and fh.closed


# --------------------------------------------------------- ④ 集合严格闭合
def test_extra_f5_row_blocks_completion(formal):
    """多出一行 F5 也必须挡住 complete=true，不能只比数目。"""
    r, f = formal["runner"], formal
    _write_all(f)
    assert r.formal_status(f["cfg"], f["h1"], f["mcq"])["complete"] is True
    ghost = _mcq_named("GHOST")
    r.durable_append(f["paths"]["mcq"], _fake_mcq_out(r, ghost)["record"])
    status = r.formal_status(f["cfg"], f["h1"], f["mcq"])
    assert status["complete"] is False
    assert status["arms"]["mcq"]["f5_key_set_matches_expected"] is False
    assert any("expected 集合之外" in p for p in status["arms"]["mcq"]["problems"])


def test_extra_npz_and_unknown_manifest_unit_are_rejected(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("h1",))
    ids = r.unit_ids_for("h1", f["h1"])
    plan = r.plan_resume(arm="h1", unit_ids=ids,
                         events=r.read_manifest_events(f["paths"]["manifest"])
                         + [{"kind": "unit", "arm": "h1", "unit_id": "GHOST",
                             "status": "done"}],
                         f5_index=r.read_f5_index(f["paths"]["h1"], "h1"),
                         npz_index={**r.npz_index_on_disk(f["paths"]["e3_dir"]),
                                    "GHOST2": {"sha256": "x", "bytes": 1}})
    assert sum("expected 集合之外" in p for p in plan["problems"]) == 2


def test_manifest_unit_event_with_an_unknown_arm_is_rejected(formal):
    r, f = formal["runner"], formal
    _write_all(f, arms=("mcq",))
    r.durable_append(f["paths"]["manifest"],
                     {"kind": "unit", "arm": "h3", "unit_id": "x", "status": "done"})
    assert r.audit_manifest_arms(r.read_manifest_events(f["paths"]["manifest"]))
    assert r.formal_status(f["cfg"], f["h1"], f["mcq"])["complete"] is False
    with pytest.raises(r.TailRecoveryStop, match="非法 unit 事件"):
        r.run_formal("mcq", 4, "definitely-not-a-model")


def test_existing_report_must_match_a_fresh_recompute(formal):
    r, f = formal["runner"], formal
    _write_all(f)
    assert r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])["complete"] is True
    r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])          # 幂等
    doctored = json.loads(f["paths"]["report"].read_text(encoding="utf-8"))
    doctored["e3_trace_block_pairs"] = 99999
    f["paths"]["report"].write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(r.TailRecoveryStop, match="与现场重算不一致"):
        r.maybe_publish_report(f["cfg"], f["h1"], f["mcq"])


def test_existing_fingerprint_must_match_build_fingerprint(formal):
    r, f = formal["runner"], formal
    freeze = r.source_freeze()
    r.publish_json_exclusive(f["paths"]["fingerprint"], r.build_fingerprint(f["cfg"], freeze))
    r.assert_fingerprint_matches(f["cfg"], freeze)               # 一致 → 通过
    doctored = json.loads(f["paths"]["fingerprint"].read_text(encoding="utf-8"))
    doctored["unit_order_rule"] = "whatever"
    f["paths"]["fingerprint"].write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(r.TailRecoveryStop, match="与现场重算不一致"):
        r.assert_fingerprint_matches(f["cfg"], freeze)


# ------------------------------------------------------------ 恢复路径收紧
def test_orphan_npz_meta_is_fully_validated_before_reuse(formal):
    """孤儿 NPZ 不能只比 block arrays —— meta 的身份四项也必须自证。"""
    r, f = formal["runner"], formal
    row = f["h1"][0]
    uid = repool.derive_unit_id(row)
    out = _fake_h1_out(r, row, f["cfg"])
    bad_meta = {**out["npz_meta"], "prompt_hash": "0" * 64}      # 身份被改
    npz = f["paths"]["e3_dir"] / f"{uid}.npz"
    r.npz_atomic(npz, out["arrays"], bad_meta)
    before = npz.read_bytes()
    with pytest.raises(tail.TailStop) as exc:
        r.process_unit(arm="h1", uid=uid, out=out, paths=f["paths"], row=row,
                       cfg=f["cfg"], segment="tail_seg002", orphan=True)
    assert exc.value.gate == "orphan_npz_meta"
    assert npz.read_bytes() == before and not f["paths"]["h1"].exists()


@pytest.mark.parametrize("mutate,pattern", [
    (lambda rec: rec.update(schema="not-the-shadow-schema"), "字段不合格"),
    (lambda rec: rec.update(variant="something-else"), "字段不合格"),
    (lambda rec: rec.update(interpretation="primary"), "字段不合格"),
    (lambda rec: rec["provenance"].update(path="elsewhere"), "provenance.path"),
    (lambda rec: rec["provenance"].update(prompt_hash="0" * 64), "prompt_hash"),
    (lambda rec: rec["stable_fields"].pop("f5_id_logprob_min"), "stable_fields 缺"),
    (lambda rec: rec["stable_fields"].update(f5_id_logprob_mean=float("nan")), "非有限"),
    (lambda rec: rec["stable_fields"].update(f5_id_token_ratio=0.99), "shadow 验收不过"),
])
def test_repair_refuses_an_f5_row_that_is_merely_valid_json(formal, mutate, pattern):
    """`repair` 曾把任意合法 JSON 补记成 done。现在整行都要过校验。"""
    r, f = formal["runner"], formal
    row = f["h1"][0]
    uid = repool.derive_unit_id(row)
    rec = _fake_h1_out(r, row, f["cfg"])["record"]
    mutate(rec)
    with pytest.raises(r.TailRecoveryStop, match=pattern):
        r.assert_tail_f5_row(rec, arm="h1", uid=uid, row=row)


def test_repair_path_runs_the_full_f5_validation(formal):
    r, f = formal["runner"], formal
    _write_all(f)
    target = repool.derive_unit_id(f["h1"][-1])
    events = [ev for ev in r.read_manifest_events(f["paths"]["manifest"])
              if ev.get("unit_id") != target]
    f["paths"]["manifest"].write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in events),
        encoding="utf-8", newline="")
    # 把那一行 F5 的口径字段改坏（重写整个文件，模拟被动过的产物）
    rows = [obj for obj, _sha in r.read_jsonl_strict(f["paths"]["h1"], what="h1")]
    for obj in rows:
        if obj["trace_id"] == target:
            obj["interpretation"] = "primary"
    f["paths"]["h1"].write_text(
        "".join(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n" for o in rows),
        encoding="utf-8", newline="")
    with pytest.raises(r.TailRecoveryStop, match="字段不合格"):
        r.run_formal("h1", 4, "definitely-not-a-model")


class _DummyFh:
    def close(self):
        pass


def _fake_telemetry(r, *, returncode=None, rows=0):
    """把 start_telemetry / stop_telemetry 换成不起子进程的替身。"""
    def fake_start(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        body = "header\n" + "".join(f"row{i}\n" for i in range(rows))
        Path(path).write_text(body, encoding="utf-8")
        return r.Telemetry(_FakeProc(returncode=returncode), _DummyFh(), Path(path))
    return fake_start


def _prime_one_pending(f, arm="mcq"):
    """写满两臂，再抽掉某一臂最后一条的 F5 行与 done 事件 → 恰好一条待跑。"""
    r = f["runner"]
    _write_all(f)
    f["paths"]["dir"].mkdir(parents=True, exist_ok=True)
    if not f["paths"]["fingerprint"].exists():
        f["paths"]["fingerprint"].write_text(
            json.dumps(r.build_fingerprint(f["cfg"], r.source_freeze())),
            encoding="utf-8")
    drop = (repool.derive_unit_id(f[arm][-1]) if arm == "h1"
            else f[arm][-1]["example_id"])
    key = "trace_id" if arm == "h1" else "example_id"
    rows = [o for o, _s in r.read_jsonl_strict(f["paths"][arm], what=arm)
            if o[key] != drop]
    f["paths"][arm].write_text(
        "".join(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n" for o in rows),
        encoding="utf-8", newline="")
    events = [e for e in r.read_manifest_events(f["paths"]["manifest"])
              if e.get("unit_id") != drop]
    f["paths"]["manifest"].write_text(
        "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in events),
        encoding="utf-8", newline="")
    if arm == "h1":                      # 让它变成"纯 pending"而不是 NPZ 孤儿
        (f["paths"]["e3_dir"] / f"{drop}.npz").unlink()
    return drop


def test_unexpected_exception_is_never_recorded_as_a_successful_segment(formal,
                                                                       monkeypatch):
    """P1：`stop_reason` 曾默认为 completed —— 任何未预期异常都会被记成成功收场。"""
    r, f = formal["runner"], formal
    _prime_one_pending(f)
    monkeypatch.setattr(r, "start_telemetry", _fake_telemetry(r, rows=1))
    monkeypatch.setattr(r, "stop_telemetry", lambda tel: tel.fh.close())

    def boom(*_a, **_k):
        raise MemoryError("CUDA out of memory (模拟)")
    monkeypatch.setattr(r, "wait_for_telemetry", boom)

    with pytest.raises(MemoryError):
        r.run_formal("mcq", 4, "definitely-not-a-model")

    close = [e for e in r.read_manifest_events(f["paths"]["manifest"])
             if e.get("kind") == "segment_close"]
    assert len(close) == 1
    assert close[0]["stopped_reason"] == "unexpected:MemoryError"
    assert close[0]["exit_code"] == 5 and close[0]["completed"] == 0
    assert not f["paths"]["report"].exists()


def test_completed_is_written_only_after_the_final_sampler_check():
    """`completed / 0` 只能在 for-else 里、**紧跟**末次采样器存活检查之后写一次。

    这条是结构断言：走到那一行需要真实 backend，行为测试覆盖不到；但"默认失败 +
    唯一成功赋值 + 成功赋值前必须先查采样器"三件事在源码层面是可判定的。
    """
    src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    assert src.count('rc, stop_reason = 5, "aborted_before_completion"') == 1
    assert src.count('rc, stop_reason = 0, "completed"') == 1
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines) if 'rc, stop_reason = 0, "completed"' in ln)
    assert "check_telemetry_alive(tel)" in lines[i - 1]
    assert lines[i - 2].strip() == "else:"
    # 未预期异常必须被记下来并**原样抛出**
    assert 'rc, stop_reason = 5, f"unexpected:{type(exc).__name__}"' in src
    j = next(k for k, ln in enumerate(lines) if 'f"unexpected:' in ln)
    assert any("raise" == lines[k].strip() for k in range(j, j + 4))


def test_segment_close_records_the_outcome_and_telemetry_digest(formal):
    """正常结束与已知失败都要留 segment_close。"""
    r, f = formal["runner"], formal
    _write_all(f)
    f["paths"]["dir"].mkdir(parents=True, exist_ok=True)
    f["paths"]["fingerprint"].write_text(
        json.dumps(r.build_fingerprint(f["cfg"], r.source_freeze())), encoding="utf-8")
    # 让采样器"立刻死掉"：run_formal 会在 wait_for_telemetry 处停，并仍写 segment_close
    seg = f["paths"]["dir"] / "tail_seg001_gpu_telemetry.csv"
    monkey = {"n": 0}

    def fake_start(path):
        monkey["n"] += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("header\n", encoding="utf-8")
        return r.Telemetry(_FakeProc(returncode=127), _DummyFh(), Path(path))

    class _DummyFh:
        def close(self):
            pass

    orig_start, orig_stop = r.start_telemetry, r.stop_telemetry
    r.start_telemetry, r.stop_telemetry = fake_start, lambda tel: tel.fh.close()
    try:
        # 造出一条待跑单元：删掉 mcq 的一条 F5 行与它的 done 事件
        drop = f["mcq"][-1]["example_id"]
        keep_rows = [o for o, _s in r.read_jsonl_strict(f["paths"]["mcq"], what="mcq")
                     if o["example_id"] != drop]
        f["paths"]["mcq"].write_text(
            "".join(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n"
                    for o in keep_rows), encoding="utf-8", newline="")
        keep_ev = [e for e in r.read_manifest_events(f["paths"]["manifest"])
                   if e.get("unit_id") != drop]
        f["paths"]["manifest"].write_text(
            "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n"
                    for e in keep_ev), encoding="utf-8", newline="")
        rc = r.run_formal("mcq", 4, "definitely-not-a-model")
    finally:
        r.start_telemetry, r.stop_telemetry = orig_start, orig_stop

    assert rc == 4
    close = [e for e in r.read_manifest_events(f["paths"]["manifest"])
             if e.get("kind") == "segment_close"]
    assert len(close) == 1
    ev = close[0]
    assert ev["segment"] == seg.stem and ev["planned"] == 1 and ev["completed"] == 0
    assert ev["stopped_reason"].startswith("recovery_stop")
    assert ev["exit_code"] == 4 and ev["telemetry_samples"] == 0
    assert ev["telemetry_sha256"] == hashlib.sha256(seg.read_bytes()).hexdigest()
    for key in ("planned", "completed", "repaired", "failed", "stopped_reason",
                "elapsed_seconds", "telemetry_sha256"):
        assert key in ev


# ══════════════════════════════════════════════════════ 最小串行 driver
def load_driver():
    spec = importlib.util.spec_from_file_location(
        "_tail_driver", _REPO_ROOT / "scripts" / "run_stage1_tail_stabilized_segments.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver():
    return load_driver()


def test_driver_plan_is_the_measured_four_segments(driver):
    assert driver.SEGMENT_PLAN == (("h1", 4000), ("h1", 4000), ("h1", 664),
                                   ("mcq", 1750))
    assert sum(n for a, n in driver.SEGMENT_PLAN if a == "h1") == 8664
    assert sum(n for a, n in driver.SEGMENT_PLAN if a == "mcq") == 1750
    assert driver.MIN_HEADROOM_MIB == 500


def test_driver_has_no_retry_repair_or_delete_path(driver):
    src = (_REPO_ROOT / "scripts"
           / "run_stage1_tail_stabilized_segments.py").read_text("utf-8")
    for banned in ("retry", "unlink", "rmtree", "shutil", "os.remove",
                   "write_text", "write_bytes", "--force"):
        assert banned not in src, banned
    assert "while " not in src                       # 没有任何重试循环


def _snap(*, n_done, n_todo, n_repair, n_failed=0, problems=(), complete=True,
          arm="mcq", freeze=None, blob=b"", segments=()):
    other = "h1" if arm == "mcq" else "mcq"
    one = {"n_expected": n_done + n_todo, "n_done": n_done, "n_todo_gpu": n_todo,
           "n_repair": n_repair, "n_failed": n_failed, "n_remaining": n_todo + n_repair,
           "n_problems": len(problems), "problems": list(problems),
           "f5_key_set_matches_expected": complete,
           "npz_key_set_matches_expected": complete if arm == "h1" else None,
           "complete": complete}
    return {"freeze": dict(freeze or {"production/x": "a"}),
            "status": {"global_problems": [], "arms": {arm: one, other: dict(one)}},
            "bytes": {"manifest": blob, "h1": blob, "mcq": blob},
            "segments": set(segments)}


def _close(**kw):
    ev = {"kind": "segment_close", "arm": "mcq", "segment": "tail_seg001_gpu_telemetry",
          "planned": 10, "completed": 10, "repaired": 0, "failed": 0,
          "stopped_reason": "completed", "elapsed_seconds": 100.0,
          "telemetry_file": "tail_seg001_gpu_telemetry.csv", "telemetry_samples": 20,
          "telemetry_sha256": "s" * 64, "exit_code": 0}
    ev.update(kw)
    return ev


def _telstats(**kw):
    t = {"exists": True, "samples": 20, "sha256": "s" * 64, "span_seconds": 95.0,
         "peak_used_mib": 11415, "n_parsed": 20}
    t.update(kw)
    return t


def _eval(driver, *, before=None, after=None, closes=None, tel=None, proc_rc=0,
          max_new=10, expect_arm_complete=True, gpu_total=12227,
          segment_key_used="tail_seg001_gpu_telemetry"):
    return driver.evaluate_segment(
        arm="mcq", max_new=max_new, proc_rc=proc_rc, segment_key_used=segment_key_used,
        before=before or _snap(n_done=0, n_todo=10, n_repair=0, complete=False,
                               blob=b"old\n"),
        after=after or _snap(n_done=10, n_todo=0, n_repair=0, blob=b"old\nnew\n",
                             segments=("tail_seg001_gpu_telemetry",)),
        closes=[_close()] if closes is None else closes,
        tel=_telstats() if tel is None else tel, gpu_total=gpu_total,
        expect_arm_complete=expect_arm_complete)


def test_driver_happy_path_passes_all_eight_gates(driver):
    assert _eval(driver) == []


@pytest.mark.parametrize("kwargs,needle", [
    ({"proc_rc": 3}, "① 子进程 exit=3"),
    ({"closes": []}, "segment_close 有 0 条"),
    ({"closes": [_close(), _close()]}, "segment_close 有 2 条"),
    ({"closes": [_close(stopped_reason="unexpected:MemoryError")]}, "stopped_reason"),
    ({"closes": [_close(exit_code=5)]}, "exit_code"),
    ({"closes": [_close(failed=1)]}, "failed"),
    ({"closes": [_close(completed=9)]}, "completed"),
    ({"tel": _telstats(sha256="x" * 64)}, "遥测 SHA"),
    ({"tel": _telstats(samples=3)}, "遥测样本数"),
    ({"tel": _telstats(span_seconds=10.0)}, "采样跨度"),
    ({"tel": _telstats(exists=False)}, "段遥测文件不存在"),
    ({"tel": _telstats(peak_used_mib=11900)}, "headroom"),
    ({"tel": _telstats(peak_used_mib=None)}, "peak memory.used"),
])
def test_driver_each_gate_can_fail(driver, kwargs, needle):
    fails = _eval(driver, **kwargs)
    assert any(needle in f for f in fails), fails


def test_driver_requires_exactly_one_new_segment(driver):
    fails = _eval(driver, after=_snap(n_done=10, n_todo=0, n_repair=0,
                                      blob=b"old\nnew\n", segments=()))
    assert any("新增段命名空间" in f for f in fails)
    fails = _eval(driver, after=_snap(n_done=10, n_todo=0, n_repair=0,
                                      blob=b"old\nnew\n",
                                      segments=("tail_seg001_gpu_telemetry",
                                                "tail_seg002_gpu_telemetry")))
    assert any("新增段命名空间" in f for f in fails)


def test_driver_new_done_count_must_be_exact(driver):
    """少跑一条、或多跑一条，都必须被抓到 —— 包含 repair 的贡献。"""
    after = _snap(n_done=9, n_todo=1, n_repair=0, complete=False,
                  blob=b"old\nnew\n", segments=("tail_seg001_gpu_telemetry",))
    fails = _eval(driver, after=after, closes=[_close(completed=9)],
                  expect_arm_complete=False)
    assert any("新增 done 9 != 期望 10" in f for f in fails)

    before = _snap(n_done=0, n_todo=8, n_repair=2, complete=False, blob=b"old\n")
    after_ok = _snap(n_done=10, n_todo=0, n_repair=0, blob=b"old\nnew\n",
                     segments=("tail_seg001_gpu_telemetry",))
    assert _eval(driver, before=before, after=after_ok,
                 closes=[_close(completed=8, repaired=2)]) == []


def test_driver_flags_failed_problems_and_incomplete_arm(driver):
    bad = _snap(n_done=10, n_todo=0, n_repair=0, n_failed=1, blob=b"old\nnew\n",
                segments=("tail_seg001_gpu_telemetry",))
    assert any("failed=1" in f for f in _eval(driver, after=bad))
    bad = _snap(n_done=10, n_todo=0, n_repair=0, problems=("ghost",),
                blob=b"old\nnew\n", segments=("tail_seg001_gpu_telemetry",))
    assert any("problems" in f for f in _eval(driver, after=bad))
    bad = _snap(n_done=10, n_todo=0, n_repair=0, complete=False, blob=b"old\nnew\n",
                segments=("tail_seg001_gpu_telemetry",))
    assert any("本臂应在此段闭合" in f for f in _eval(driver, after=bad))
    glob = _snap(n_done=10, n_todo=0, n_repair=0, blob=b"old\nnew\n",
                 segments=("tail_seg001_gpu_telemetry",))
    glob["status"]["global_problems"] = ["arm 非法"]
    assert any("全局问题" in f for f in _eval(driver, after=glob))


def test_driver_catches_source_freeze_drift_and_prefix_rewrite(driver):
    drifted = _snap(n_done=10, n_todo=0, n_repair=0, blob=b"old\nnew\n",
                    freeze={"production/x": "CHANGED"},
                    segments=("tail_seg001_gpu_telemetry",))
    assert any("source freeze 漂移" in f for f in _eval(driver, after=drifted))
    rewritten = _snap(n_done=10, n_todo=0, n_repair=0, blob=b"DIFFERENT\n",
                      segments=("tail_seg001_gpu_telemetry",))
    fails = _eval(driver, after=rewritten)
    assert sum("append-only 被破坏" in f for f in fails) == 3   # manifest + 两臂


@pytest.mark.local_artifacts('outputs/stage1/benchmark/tail_stabilized_benchmark_gpu_telemetry.csv')
def test_driver_telemetry_stats_parses_the_real_benchmark_csv(driver):
    """用真实的微基准遥测做一次端到端解析核对（只读）。"""
    p = _REAL_OUT / "benchmark" / "tail_stabilized_benchmark_gpu_telemetry.csv"
    st = driver.telemetry_stats(p)
    assert st["exists"] and st["samples"] == 17 and st["n_parsed"] == 17
    assert st["peak_used_mib"] == 11415
    assert 75.0 < st["span_seconds"] < 85.0
    assert st["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert 12227 - st["peak_used_mib"] == 812        # 与 addendum 的订正一致
    assert 12227 - st["peak_used_mib"] >= driver.MIN_HEADROOM_MIB


def test_driver_telemetry_stats_on_a_missing_file(driver, tmp_path):
    st = driver.telemetry_stats(tmp_path / "nope.csv")
    assert st["exists"] is False and st["samples"] == 0 and st["sha256"] is None


def test_driver_segment_key_matches_what_the_runner_actually_records(driver, runner,
                                                                     tmp_path,
                                                                     monkeypatch):
    """**第一次实跑栽在这里**：runner 记的是 `tail_seg001_gpu_telemetry`（`Path.stem`），
    driver 却把 `_gpu_telemetry` 剥掉去查 `tail_seg001` —— 一段跑得完全正常，
    却被判成"segment_close 有 0 条"。两侧口径必须由**同一处**定义。
    """
    monkeypatch.setattr(runner, "TAIL_DIR", tmp_path / "tail_stabilized")
    (tmp_path / "tail_stabilized").mkdir()
    seg = runner.next_segment_path()
    assert seg.name == "tail_seg001_gpu_telemetry.csv"
    key = driver.segment_key(seg)
    assert key == seg.stem == "tail_seg001_gpu_telemetry"
    assert driver.telemetry_path_for(runner, key) == seg
    # runner 源码里写进 manifest 的就是 seg_path.stem，不是别的派生名
    src = (_REPO_ROOT / "scripts" / "run_stage1_tail_stabilized.py").read_text("utf-8")
    assert '"segment": seg_path.stem' in src
    assert 'segment=seg_path.stem' in src


def test_driver_reports_a_key_mismatch_as_its_own_fault_not_a_failed_segment(driver):
    fails = _eval(driver, segment_key_used="tail_seg001")     # 剥掉后缀的旧写法
    assert any("driver 自身口径错误" in f for f in fails)
    assert not any("segment_close 有 0 条" in f for f in fails)


def test_driver_finds_a_real_segment_close_written_by_the_runner(formal, driver):
    """端到端把 runner 写的 segment_close 用 driver 的解析路径查回来。"""
    r, f = formal["runner"], formal
    _prime_one_pending(f)
    seg = r.next_segment_path()
    r.durable_append(f["paths"]["manifest"], {
        "kind": "segment_close", "arm": "mcq", "segment": seg.stem,
        "planned": 1, "completed": 1, "repaired": 0, "failed": 0,
        "stopped_reason": "completed", "elapsed_seconds": 1.0,
        "telemetry_file": seg.name, "telemetry_samples": 1,
        "telemetry_sha256": "x" * 64, "exit_code": 0})
    found = driver.segment_close_events(r, driver.segment_key(seg))
    assert len(found) == 1 and found[0]["segment"] == seg.stem


def test_driver_skips_a_planned_segment_already_covered_by_an_earlier_one(driver):
    """断点续跑：第一段跑完后剩 4,664，后续变成 4,000 + 664，
    计划里的第三项就无活可干 —— 必须跳过而不是判 ② 失败。"""
    complete = _snap(n_done=8664, n_todo=0, n_repair=0, arm="h1")["status"]["arms"]["h1"]
    assert driver.should_skip_segment(complete, 664) == (True, None)

    busy = _snap(n_done=4000, n_todo=4664, n_repair=0, arm="h1",
                 complete=False)["status"]["arms"]["h1"]
    assert driver.should_skip_segment(busy, 4000) == (False, None)

    # 有待补记就不算"无活可干"
    repairs = _snap(n_done=8663, n_todo=0, n_repair=1, arm="h1",
                    complete=False)["status"]["arms"]["h1"]
    assert driver.should_skip_segment(repairs, 664) == (False, None)


def test_driver_refuses_to_skip_when_the_arm_is_not_closed(driver):
    """没活干但本臂没闭合 = 不自洽，必须报错而不是静悄悄跳过。"""
    stuck = _snap(n_done=8000, n_todo=0, n_repair=0, arm="h1", n_failed=1,
                  complete=False)["status"]["arms"]["h1"]
    skip, err = driver.should_skip_segment(stuck, 664)
    assert skip is False and err and "状态不自洽" in err


def test_driver_plan_totals_match_the_pinned_scale(driver, runner):
    assert sum(n for a, n in driver.SEGMENT_PLAN if a == "h1") == runner.N_H1_EXPECTED
    assert sum(n for a, n in driver.SEGMENT_PLAN if a == "mcq") == runner.N_MCQ_EXPECTED


def test_driver_runs_segments_strictly_serially_via_subprocess(driver):
    src = (_REPO_ROOT / "scripts"
           / "run_stage1_tail_stabilized_segments.py").read_text("utf-8")
    assert "subprocess.run(cmd, cwd=str(ROOT), check=False)" in src
    assert "Popen" not in src                        # 不留任何并发启动方式
    assert "--defer-report" in src                   # 段内不发布终态产物
    assert "--finalize" in src                       # 四段全过后才发布
    i = src.index("四段全部通过机械门")
    assert src.index("run_finalize_subprocess()", i) > i
