"""W0.5-F：production generation + feature cache 的护栏测试。

重点是**崩溃语义**与**契约**,不是 happy path:
  - running（上次被杀留下的）必须重新入队,否则丢数据
  - "manifest 说 done 但产物不在盘上" 必须自愈重跑 —— 这是 resume 最阴险的失败模式,
    它把数据丢失伪装成"已完成"(本 sprint 真实发生过:中断后 38 个 done 的 unit 向量全丢)
  - 半行（写到一半断电）不得让读取崩溃
  - fingerprint 不一致时不得静默复用旧 cache
  - 生成端产物必须正好是分析入口读得懂的格式
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.generation import fingerprint as fp
from recover_attention.generation.manifest import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    WorkManifest,
    write_atomic,
)

_spec = importlib.util.spec_from_file_location(
    "run_stage0_generation", ROOT / "scripts" / "run_stage0_generation.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


# ---------------------------------------------------------------- manifest
def test_register_is_idempotent(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    assert m.register(["a", "b"]) == 2
    assert m.register(["a", "b", "c"]) == 1        # 只登记新的
    assert set(m.current_status()) == {"a", "b", "c"}


def test_done_units_are_not_requeued(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a", "b"])
    m.mark("a", STATUS_RUNNING)
    m.mark("a", STATUS_DONE)
    assert m.pending_units(["a", "b"]) == ["b"]


def test_running_is_treated_as_unfinished(tmp_path):
    """上次跑到一半被杀会留下 running。不重跑就会丢数据。"""
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a"])
    m.mark("a", STATUS_RUNNING)
    assert WorkManifest(tmp_path / "m.jsonl").pending_units(["a"]) == ["a"]


def test_failed_is_requeued_and_attempts_counted(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a"])
    m.mark("a", STATUS_RUNNING); m.mark("a", STATUS_FAILED, error="boom")
    assert m.pending_units(["a"]) == ["a"]
    m.mark("a", STATUS_RUNNING); m.mark("a", STATUS_DONE)
    assert m.attempts("a") == 2
    assert m.pending_units(["a"]) == []


def test_torn_last_line_does_not_break_replay(tmp_path):
    """断电会留下半行。append-only 的损坏只可能在末尾 —— 之前的事件仍然有效。"""
    p = tmp_path / "m.jsonl"
    m = WorkManifest(p)
    m.register(["a", "b"])
    m.mark("a", STATUS_RUNNING); m.mark("a", STATUS_DONE)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"unit_id": "b", "sta')            # 半行
    m2 = WorkManifest(p)
    assert m2.current_status()["a"] == STATUS_DONE
    assert m2.pending_units(["a", "b"]) == ["b"]


def test_state_is_last_event_wins(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a"])
    for s in (STATUS_RUNNING, STATUS_FAILED, STATUS_RUNNING, STATUS_DONE):
        m.mark("a", s)
    assert m.current_status()["a"] == STATUS_DONE
    assert m.is_complete(["a"])


def test_invalid_status_rejected(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    with pytest.raises(ValueError, match="invalid status"):
        m.mark("a", "almost_done")


def test_write_atomic_leaves_no_partial_file(tmp_path):
    p = tmp_path / "out.json"
    write_atomic(p, '{"ok": true}')
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------- fingerprint
def test_unit_fingerprint_changes_with_prompt():
    a = fp.unit_fingerprint(prompt="p1", seed=1, temperature=0.7, top_p=0.95,
                            max_new_tokens=384, sample_type="greedy", sample_index=0)
    b = fp.unit_fingerprint(prompt="p2", seed=1, temperature=0.7, top_p=0.95,
                            max_new_tokens=384, sample_type="greedy", sample_index=0)
    assert a["prompt_hash"] != b["prompt_hash"] and a["unit_hash"] != b["unit_hash"]


def test_unit_fingerprint_is_stable_for_same_inputs():
    kw = dict(prompt="p", seed=3, temperature=0.7, top_p=0.95, max_new_tokens=384,
              sample_type="sampled", sample_index=2)
    assert fp.unit_fingerprint(**kw) == fp.unit_fingerprint(**kw)


def test_run_fingerprint_mismatch_blocks_cache_reuse():
    """换了模型/分词器/层位,旧 hidden 就不可比。静默复用是最难查的错。"""
    base = fp.run_fingerprint(model_fp={"model_hash": "m1"}, tokenizer_fp={"tokenizer_hash": "t1"},
                              arm="h1", hidden_index={"hidden_states_tuple_index": 20},
                              gen_params={"K": 6})
    fp.assert_run_fingerprint_matches(base, dict(base))          # 相同 → 放行
    for key, patch in (("model", {"model_hash": "m2"}),
                       ("tokenizer", {"tokenizer_hash": "t2"}),
                       ("hidden_index", {"hidden_states_tuple_index": 21})):
        other = {**base, key: patch}
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            fp.assert_run_fingerprint_matches(base, other)


def test_tokenizer_fingerprint_includes_special_tokens():
    """同名分词器可能 special token 不同,而它们决定 token 边界、边界决定 hidden 取值位置。"""
    class T:
        name_or_path = "x"; vocab_size = 100
        bos_token_id = 1; eos_token_id = 2; pad_token_id = 2; unk_token_id = 0
    a = fp.tokenizer_fingerprint(T())
    T.eos_token_id = 99
    b = fp.tokenizer_fingerprint(T())
    assert a["tokenizer_hash"] != b["tokenizer_hash"]


# ------------------------------------------------- resume / durability 契约
def _records(n, arm):
    return [{"example_id": f"{arm}_{i:05d}", "group_id": f"{arm}_g{i:05d}",
             "question_text": f"q{i}"} for i in range(n)]


def _run(arm, tmp_path, records, backend, **kw):
    run_fp = fp.run_fingerprint(model_fp={"model_hash": "dry"},
                                tokenizer_fp={"tokenizer_hash": "dry"}, arm=arm,
                                hidden_index={"hidden_states_tuple_index": 20},
                                gen_params={"K": 6})
    return gen.generate_arm(arm, records, backend, tmp_path / "gen", tmp_path / "feat",
                            run_fp, progress_every=999, **kw)


def test_done_but_missing_hidden_is_requeued(tmp_path):
    """核心回归:manifest 说 done、hidden 却不在盘上 → 必须重跑,不得写出残缺 cache。

    真实发生过:早期实现把 hidden 攒在内存里最后才 savez,中断后 38 个 done 的 unit
    向量全丢,而 manifest 照样说 done → n_records=12 但 n_hidden=6。
    """
    recs = _records(3, "h1")
    res = _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8, seed=0))
    assert res["status"] == "complete"
    assert res["feature_cache"]["n_records"] == res["feature_cache"]["n_hidden"] == 3

    # 抹掉一条 done unit 的 hidden，模拟"记录说完成、向量丢了"
    victim = tmp_path / "gen" / "h1_unit_hidden" / f"{gen._safe('h1_00001__greedy_0')}.npy"
    assert victim.exists()
    victim.unlink()
    res2 = _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8, seed=1))
    assert res2["status"] == "complete"
    assert res2["feature_cache"]["n_hidden"] == 3, "缺失的 hidden 必须被重新生成"


def test_resume_skips_completed_units(tmp_path):
    recs = _records(4, "h1")
    _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8))
    man = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    before = len(man._events)
    _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8))
    man2 = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    assert len(man2._events) == before, "全部 done 时重跑不应产生新事件"


def test_incomplete_run_does_not_write_feature_cache(tmp_path):
    """未跑完就写 cache = 用残缺数据做分析。必须拒绝。"""
    class Flaky:
        backend = {"load_in_8bit": True}
        def run_unit(self, unit, record):
            if unit["sample_index"] == 3:
                raise RuntimeError("simulated failure")
            return {"completion": "x", "hidden": np.zeros(8, dtype=np.float32)}

    res = _run("h1", tmp_path, _records(2, "h1"), Flaky())
    assert res["status"] == "incomplete"
    assert "feature_cache" not in res
    assert not (tmp_path / "feat" / "h1_completion_records.jsonl").exists()


def test_generation_output_is_readable_by_the_analysis_entry(tmp_path):
    """契约点:生成端产物必须正好是 evaluation.feature_cache 读得懂的格式。

    两条 pipeline 分开写最容易在这里错位 —— 生成完了分析入口读不进去。
    """
    from recover_attention.evaluation.config import Stage0Config
    from recover_attention.evaluation.feature_cache import load_arm

    recs = _records(6, "mcq")
    res = _run("mcq", tmp_path, recs, gen._FakeBackend("mcq", hidden_dim=8, seed=0))
    assert res["status"] == "complete"
    feat = tmp_path / "feat"
    assert (feat / "mcq_completion_records.jsonl").exists()
    assert (feat / "mcq_hidden.npz").exists()

    rows = [json.loads(l) for l in open(feat / "mcq_completion_records.jsonl", encoding="utf-8")]
    for r in rows:                       # 分析入口按这些字段读
        assert {"example_id", "group_id", "population_role", "run_hash"} <= set(r)
        assert r["population_role"] == "confirmatory"   # 绝不能是 pilot_smoke


def test_population_role_is_confirmatory_not_pilot(tmp_path):
    """production 产物必须标 confirmatory —— 分析入口据此拒绝 pilot 混入。"""
    res = _run("h1", tmp_path, _records(2, "h1"), gen._FakeBackend("h1", hidden_dim=8))
    rows = [json.loads(l) for l in
            open(tmp_path / "feat" / "h1_completion_records.jsonl", encoding="utf-8")]
    assert all(r["population_role"] == "confirmatory" for r in rows)


def test_mcq_population_is_greedy_only():
    """§3 v2.3:MCQ population = 每题一条 greedy;5 条 sampled 仅供 F5 self-consistency。"""
    units = gen.mcq_units(_records(3, "mcq"))
    assert len(units) == 3 * gen.K_TRACES
    in_pop = [u for u in units if u["in_population"]]
    assert len(in_pop) == 3
    assert all(u["sample_type"] == "greedy" for u in in_pop)


def test_h1_units_are_k6_per_prompt():
    """§3 冻结:K=6 = 1 greedy + 5 sampled。"""
    units = gen.h1_units(_records(4, "h1"))
    assert len(units) == 4 * 6
    assert sum(u["sample_type"] == "greedy" for u in units) == 4
    assert sum(u["sample_type"] == "sampled" for u in units) == 20
