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
from recover_attention.generation import backends as bk
from recover_attention.generation.manifest import (
    STATUS_DONE,
    STATUS_EXCLUDED,
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


def test_excluded_is_terminal_and_not_requeued(tmp_path):
    """excluded 是终态但不是 done:不参与重试,但计入完成度。

    区别于 failed —— 数值层面的事(如 forward 输出非有限)不会因为重跑而消失,
    重新排队只会原样再失败一次,把已经下过的结论又推翻成"待定"。
    """
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a", "b"])
    m.mark("a", STATUS_RUNNING); m.mark("a", STATUS_EXCLUDED, reason="forward_nonfinite")
    m.mark("b", STATUS_RUNNING); m.mark("b", STATUS_DONE)
    assert m.pending_units(["a", "b"]) == []
    assert m.is_complete(["a", "b"])


def test_summary_counts_excluded_by_reason(tmp_path):
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a", "b", "c"])
    m.mark("a", STATUS_EXCLUDED, reason="forward_nonfinite")
    m.mark("b", STATUS_EXCLUDED, reason="forward_nonfinite")
    m.mark("c", STATUS_DONE)
    s = m.summary(["a", "b", "c"])
    assert s["n_excluded"] == 2
    assert s["excluded_by_reason"] == {"forward_nonfinite": 2}
    assert s["n_done"] == 1
    assert s["complete"] is True
    assert s["excluded_units"] == ["a", "b"]


def test_incomplete_when_units_still_pending_alongside_excluded(tmp_path):
    """excluded 不能让别的、真正还没跑完的 unit 被误判为"整批完成"。"""
    m = WorkManifest(tmp_path / "m.jsonl")
    m.register(["a", "b"])
    m.mark("a", STATUS_EXCLUDED, reason="forward_nonfinite")
    s = m.summary(["a", "b"])
    assert s["complete"] is False


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


def test_mcq_parsed_letter_span_uses_letter_not_whole_completion():
    text = "The best option is not B.\nFinal answer: C"
    cs, ce = bk._parsed_letter_char_span(text, "C")
    assert text[cs:ce] == "C"
    assert (ce - cs) == 1


def test_mcq_parsed_letter_span_follows_trailing_isolated_parser():
    text = "After checking the choices, I choose D."
    cs, ce = bk._parsed_letter_char_span(text, "D")
    assert text[cs:ce] == "D"


def _mcq_backend_with_letter_ids():
    """只装两套 option id 的 MCQBackend —— 不加载模型（`_letter_id_set` 只用这两个属性）。"""
    backend = bk.MCQBackend.__new__(bk.MCQBackend)
    backend.letter_ids_bare = {"A": 32, "B": 33, "C": 34, "D": 35}
    backend.letter_ids_space = {"A": 362, "B": 425, "C": 356, "D": 422}
    return backend


def test_mcq_letter_id_set_follows_the_emitted_token_form():
    """F5 的候选 id 集必须跟发射 token 同形。

    真实风险:letter span 修好后,"Answer: C" 这类 completion 开始进 population,其字母
    token 是空格形 356;若仍按裸形 {32..35} 读 logit,margin/entropy 会落在一组没有概率
    质量的候选上 —— 不崩、不缺列,只是数值无意义,且错在与标签相关的子集上(削弱 O →
    系统性抬高 Δ_H)。
    """
    backend = _mcq_backend_with_letter_ids()
    assert backend._letter_id_set(34, "C") == (backend.letter_ids_bare, "bare")
    assert backend._letter_id_set(356, "C") == (backend.letter_ids_space, "space")


def test_mcq_letter_id_set_refuses_an_unknown_token_form():
    backend = _mcq_backend_with_letter_ids()
    with pytest.raises(ValueError, match="neither the bare form"):
        backend._letter_id_set(9999, "C")


def test_mcq_wrong_label_matches_gold_comparison():
    """§7.1/§3 的目标 Y:1 = 解析出的选项与 gold_label 不同,0 = 相同。"""
    assert bk._mcq_wrong_label("C", True, "C") == 0
    assert bk._mcq_wrong_label("C", True, "D") == 1


def test_mcq_wrong_label_is_none_when_ineligible():
    """parse_failure 没有可比较的选项,不得编出一个 0/1——必须是 None,
    与 canon["eligible_for_primary"]=False 时不进 population 的语义一致。"""
    assert bk._mcq_wrong_label(None, False, "C") is None
    assert bk._mcq_wrong_label("Z", False, "C") is None


def test_done_but_missing_hidden_is_requeued(tmp_path):
    """核心回归:manifest 说 done、hidden 却不在盘上 → 必须重跑,不得写出残缺 cache。

    真实发生过:早期实现把 hidden 攒在内存里最后才 savez,中断后 38 个 done 的 unit
    向量全丢,而 manifest 照样说 done → n_records=12 但 n_hidden=6。
    """
    class HiddenRequiredBackend:
        backend = {"load_in_8bit": True}

        def __init__(self):
            self.n = 0

        def run_unit(self, unit, record):
            self.n += 1
            return {"completion": "T1114",
                    "id_token_positions_local": [0],
                    "eligible": True,
                    "hidden": np.full(8, self.n, dtype=np.float32)}

    recs = _records(3, "h1")
    res = _run("h1", tmp_path, recs, HiddenRequiredBackend())
    assert res["status"] == "complete"
    # H1 的 population 是每条 completion（3 prompts × K=6）。此前这里写死 ==3,
    # 那是把 greedy-only 那个 bug 的行为当成了期望值。
    expected = 3 * gen.K_TRACES
    assert res["feature_cache"]["n_records"] == res["feature_cache"]["n_hidden"] == expected

    # 抹掉一条 done unit 的 hidden，模拟"记录说完成、向量丢了"
    victim = tmp_path / "gen" / "h1_unit_hidden" / f"{gen._safe('h1_00001__greedy_0')}.npy"
    assert victim.exists()
    victim.unlink()
    res2 = _run("h1", tmp_path, recs, HiddenRequiredBackend())
    assert res2["status"] == "complete"
    assert res2["feature_cache"]["n_hidden"] == expected, "缺失的 hidden 必须被重新生成"


def test_h1_done_without_identifier_positions_does_not_require_hidden(tmp_path):
    """H1 hidden is pooled over eligible identifier tokens. If a completion has
    no identifier positions, there is no hidden vector to persist or require.
    """
    class NoIdBackend:
        backend = {"load_in_8bit": True}

        def run_unit(self, unit, record):
            return {"completion": "no identifiers here",
                    "id_token_positions_local": [],
                    "eligible": False,
                    "primary_exclusion_reason": "no_identifier"}

    recs = _records(2, "h1")
    res = _run("h1", tmp_path, recs, NoIdBackend())
    assert res["status"] == "complete"
    assert res["feature_cache"]["n_records"] == 2 * gen.K_TRACES
    assert res["feature_cache"]["n_hidden"] == 0

    man = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    before = len(man._events)
    res2 = _run("h1", tmp_path, recs, NoIdBackend())
    man2 = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    assert res2["status"] == "complete"
    assert len(man2._events) == before


def test_resume_skips_completed_units(tmp_path):
    recs = _records(4, "h1")
    _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8))
    man = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    before = len(man._events)
    _run("h1", tmp_path, recs, gen._FakeBackend("h1", hidden_dim=8))
    man2 = WorkManifest(tmp_path / "gen" / "h1_generated_manifest.jsonl")
    assert len(man2._events) == before, "全部 done 时重跑不应产生新事件"


class _ForwardNonFiniteBackend:
    """模拟真实事故:29/10560 条 trace 在 answer-letter token 那一步整行 logits 非有限。"""

    backend = {"load_in_8bit": True}

    def __init__(self, excluded_unit_ids):
        self.excluded_unit_ids = set(excluded_unit_ids)
        self.rng = np.random.default_rng(0)

    def run_unit(self, unit, record):
        if unit["unit_id"] in self.excluded_unit_ids:
            raise bk.ForwardNonFiniteError(
                f"non-finite option logits for {unit['unit_id']}",
                diagnostic={"reason": "forward_nonfinite", "example_id": record["example_id"],
                           "group_id": record["group_id"], "sample_type": unit["sample_type"],
                           "sample_index": unit["sample_index"], "row_nonfinite": 152064,
                           "row_size": 152064, "hidden_nonfinite": 0, "hidden_size": 3584})
        letter = "ABCD"[unit["sample_index"] % 4]
        return {"raw_completion": letter, "parsed_label": letter, "eligible_for_primary": True,
                "hidden": self.rng.normal(0, 1, 8).astype(np.float32),
                "f5_label_margin": 1.0, "f5_label_entropy": 0.1}


def test_mcq_greedy_forward_nonfinite_excludes_question_from_primary_population(tmp_path):
    """决策(a,2026-07-21 用户确认):greedy 落进 forward_nonfinite 时排除该题,不用 sampled
    顶替 —— 顶替会改写"MCQ 主样本=greedy"这条协议定义,而且"这题可以替、那题不替"本身
    比直接排除更难向审稿人解释。

    真实发生过:MCQ fresh 1760 生成中,29/10560 条 trace 在 answer-letter token 那一步
    teacher-forced forward 输出整行非有限 logits(8-bit lm_head 反量化溢出,集中在
    "Answer: X" 显式风格 completion 上),其中 5 题连 greedy 都在内。若不处理,
    payloads.get(f"{eid}__greedy_0") 返回 None 时 _build_feature_cache 会直接 continue,
    population 从 1760 悄悄变 1755,没有任何排除记录 —— 这条测试钉住"悄悄"必须变成
    "显式记录在 primary_exclusion_report 里"。
    """
    recs = _records(3, "mcq")
    # mcq_00000 的 greedy 排除 → 整题掉出 primary population。
    # mcq_00001 只排除一条 sampled 手足 → greedy 完好,primary population 不受影响,
    # 只是 F5 self-consistency 聚合的手足少了一个。
    excluded = {"mcq_00000__greedy_0", "mcq_00001__sampled_2"}
    backend = _ForwardNonFiniteBackend(excluded)
    res = _run("mcq", tmp_path, recs, backend)

    assert res["status"] == "complete"
    cache = res["feature_cache"]
    assert cache["n_records"] == 2
    assert cache["n_excluded_primary"] == 1

    records_out = [json.loads(l) for l in
                   open(tmp_path / "feat" / "mcq_completion_records.jsonl", encoding="utf-8")]
    assert {r["example_id"] for r in records_out} == {"mcq_00001", "mcq_00002"}

    report = json.loads((tmp_path / "feat" / "mcq_primary_exclusion_report.json")
                        .read_text(encoding="utf-8"))
    assert report["n_nominal_questions"] == 3
    assert report["n_primary_records_written"] == 2
    assert report["n_primary_records_eligible"] == 2
    assert report["n_primary_records_parse_failure"] == 0
    assert report["n_excluded_primary_forward_nonfinite"] == 1
    assert report["excluded_primary"][0]["example_id"] == "mcq_00000"
    assert report["excluded_primary"][0]["reason"] == "forward_nonfinite"
    # mcq_00001 只是手足受损,主样本完好 —— 不得出现在 primary 排除列表里
    assert not any(e["example_id"] == "mcq_00001" for e in report["excluded_primary"])

    manifest = WorkManifest(tmp_path / "gen" / "mcq_generated_manifest.jsonl")
    summ = manifest.summary([u["unit_id"] for u in gen.mcq_units(recs)])
    assert summ["n_excluded"] == 2
    assert summ["excluded_by_reason"] == {"forward_nonfinite": 2}
    assert summ["complete"] is True


def test_excluded_units_are_not_requeued_on_mcq_resume(tmp_path):
    """excluded 是终态:resume 不应该把它们再排一次队(数值层面的事重跑不会变好)。"""
    recs = _records(1, "mcq")
    backend = _ForwardNonFiniteBackend({"mcq_00000__greedy_0"})
    _run("mcq", tmp_path, recs, backend)
    man = WorkManifest(tmp_path / "gen" / "mcq_generated_manifest.jsonl")
    before = len(man._events)
    _run("mcq", tmp_path, recs, backend)
    man2 = WorkManifest(tmp_path / "gen" / "mcq_generated_manifest.jsonl")
    assert len(man2._events) == before, "excluded 不应被重新入队产生新事件"


def test_mcq_wrong_label_is_preserved_into_written_completion_records(tmp_path):
    """回归:evaluation.feature_cache.load_arm(..., label_field='wrong_label') 在真实
    fresh 1760 产物上直接 KeyError('wrong_label')——production 的 MCQBackend.run_unit
    从未计算这一列(G4 quality_check 早就算过同一个式子,production 生成路径漏了,直到
    这次 resume 第一次真正跑到冻结/分析入口才暴露)。这里锁住 _build_feature_cache 这一
    层的契约:只要 run_unit 的返回值带 wrong_label,组装 row 时必须原样保留,不能被
    覆盖或漏掉。"""
    class LabeledBackend:
        backend = {"load_in_8bit": True}

        def __init__(self):
            self.rng = np.random.default_rng(0)

        def run_unit(self, unit, record):
            letter = "ABCD"[unit["sample_index"] % 4]
            return {"raw_completion": letter, "parsed_label": letter,
                    "eligible_for_primary": True,
                    "wrong_label": bk._mcq_wrong_label(letter, True, "A"),
                    "hidden": self.rng.normal(0, 1, 8).astype(np.float32),
                    "f5_label_margin": 1.0, "f5_label_entropy": 0.1}

    recs = _records(2, "mcq")
    res = _run("mcq", tmp_path, recs, LabeledBackend())
    assert res["status"] == "complete"
    rows = [json.loads(l) for l in
            open(tmp_path / "feat" / "mcq_completion_records.jsonl", encoding="utf-8")]
    assert len(rows) == 2
    assert all("wrong_label" in r for r in rows)
    # 两题的 greedy(sample_index=0)都拿到字母 "A" == gold "A" → wrong_label 恒 0，
    # 值本身不是这条测试的重点，重点是这一列必须存在且等于 run_unit 返回的原值。
    assert all(r["wrong_label"] == 0 for r in rows)


class _ParseFailureBackend:
    """模拟真实事故:cybermetric_000718 的模型输出("Answer: T")解析不到合法选项。

    §7.1 冻结:parse_failure 记 eligible_for_primary=False,不进 population,不算 hidden——
    但这条 done unit **理应**没有 hidden,不是数据丢失,不该被当孤儿。
    """

    backend = {"load_in_8bit": True}

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def run_unit(self, unit, record):
        if unit["unit_id"] == "mcq_00000__greedy_0":
            return {"raw_completion": "Answer: T", "parsed_label": None,
                    "eligible_for_primary": False}
        letter = "ABCD"[unit["sample_index"] % 4]
        return {"raw_completion": letter, "parsed_label": letter,
                "eligible_for_primary": True,
                "hidden": self.rng.normal(0, 1, 8).astype(np.float32),
                "f5_label_margin": 1.0, "f5_label_entropy": 0.1}


def test_mcq_parse_failure_primary_is_written_without_hidden_and_not_reorphaned(tmp_path):
    """回归:MCQ 的 `_requires_hidden` 曾经对非 H1 臂恒返回 True,把每一条合法的
    parse_failure(eligible_for_primary=False)都当成"该有 hidden 却没有"的孤儿,每次
    resume 都重新入队、重跑、再判孤儿,自愈判定永不收敛——真实发生过:
    cybermetric_000718 的 greedy/sampled_4 连续两次 resume 被重新排队,直到这次才发现。
    `_build_feature_cache` 的 missing-hidden 硬校验同样漏了 eligible 判据,导致它一旦
    真正跑到这条记录就直接 RuntimeError,MCQ population 永远写不出来。
    """
    recs = _records(2, "mcq")
    backend = _ParseFailureBackend()
    res = _run("mcq", tmp_path, recs, backend)

    assert res["status"] == "complete"
    cache = res["feature_cache"]
    assert cache["n_records"] == 2           # parse_failure 仍写进 records(与 H1 同规则)
    assert cache["n_excluded_primary"] == 0  # 不是 forward_nonfinite,不进 excluded_primary
    assert cache["n_hidden"] == 1            # 只有 eligible 的那题有 hidden

    report = json.loads((tmp_path / "feat" / "mcq_primary_exclusion_report.json")
                        .read_text(encoding="utf-8"))
    assert report["n_primary_records_written"] == 2
    assert report["n_primary_records_eligible"] == 1
    assert report["n_primary_records_parse_failure"] == 1

    man = WorkManifest(tmp_path / "gen" / "mcq_generated_manifest.jsonl")
    before = len(man._events)
    _run("mcq", tmp_path, recs, backend)     # resume:parse_failure 不该被重新入队
    man2 = WorkManifest(tmp_path / "gen" / "mcq_generated_manifest.jsonl")
    assert len(man2._events) == before, "parse_failure 被误判孤儿、resume 又重跑了一次"


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


def test_cross_sample_f5_is_wired_into_h1_feature_cache(tmp_path):
    """回归：_build_feature_cache 曾经从未调用 attach_cross_sample_f5。

    真实发生过：Stage 0 全量 H1（2880 条）跑完后，f5_self_consistency_exact 与
    f5_id_agreement_rate 两列在生产 feature cache 里全部缺失（既非 0 也不是常数——
    是根本不存在），因为组装 rows 时没人调用跨样本聚合函数。F5 completeness 的既有检查
    没抓住它，因为那条检查只看"连续量是否全零/常数"，而这两列在 _dense_matrix 里被填成
    0 后表现为常数列——本该被抓住，但当时的检查实现遗漏了这个具体场景的复现。
    这里直接断言 generate_arm 的产出里这两列存在且并非清一色相同值。
    """
    class VaryingMentionBackend:
        backend = {"load_in_8bit": True}

        def run_unit(self, unit, record):
            # 每个 prompt（example_id 末位数字 i）里,前 i 个 sample_index 的 unit 给出
            # 少数派 identifier，其余给多数派 —— 4 个 prompt 因此产出 4 种不同的一致性比例
            # (6/6, 5/6, 4/6, 3/6),不是"同一个分裂模式在所有 group 里重复"。
            i = int(record["example_id"].rsplit("_", 1)[-1])
            ident = "T2000" if unit["sample_index"] < i else "T1000"
            return {"completion": ident, "eligible": True,
                    "id_token_positions_local": [0],
                    "mentions": [{"family": "attack", "label": "grounded",
                                 "normalized": ident}],
                    "hidden": np.ones(8, dtype=np.float32)}

    recs = _records(4, "h1")
    res = _run("h1", tmp_path, recs, VaryingMentionBackend())
    assert res["status"] == "complete"
    rows = [json.loads(l) for l in
            open(tmp_path / "feat" / "h1_completion_records.jsonl", encoding="utf-8")]
    assert all("f5_self_consistency_exact" in r for r in rows)
    assert all("f5_id_agreement_rate" in r for r in rows)
    cons = {r["f5_self_consistency_exact"] for r in rows}
    agree = {r["f5_id_agreement_rate"] for r in rows}
    assert cons != {None}, "跨样本一致性字段不得整批为 None/缺失"
    assert len(cons) > 1, "不同 prompt 的一致性比例不同，不该被聚合成同一个常数"
    assert len(agree) > 1, "identifier 有分歧时，agreement 不应是常数"


def test_cross_sample_f5_is_wired_into_mcq_feature_cache(tmp_path):
    """同一回归的 MCQ 版本：population 只含 greedy，但一致性要看它与 5 条 sampled 手足
    是否吻合，必须用**全部** K 个 unit 聚合，而不是只用最终进 population 的那一行。"""
    class VaryingLetterBackend:
        backend = {"load_in_8bit": True}

        def run_unit(self, unit, record):
            # 与 H1 版同样的构造：按 example 索引 i 变化少数派比例，避免 4 个 group
            # 产出完全相同的一致性数值。
            i = int(record["example_id"].rsplit("_", 1)[-1])
            letter = "B" if unit["sample_index"] < i else "A"
            return {"raw_completion": letter, "parsed_label": letter,
                    "eligible_for_primary": True,
                    "selected_option_text": f"option {letter}",
                    "semantic_output_text": f"option {letter}",
                    "hidden": np.ones(8, dtype=np.float32)}

    recs = _records(4, "mcq")
    res = _run("mcq", tmp_path, recs, VaryingLetterBackend())
    assert res["status"] == "complete"
    rows = [json.loads(l) for l in
            open(tmp_path / "feat" / "mcq_completion_records.jsonl", encoding="utf-8")]
    assert all("f5_self_consistency_exact" in r for r in rows)
    assert all("f5_letter_agreement_rate" in r for r in rows)
    assert {r["f5_self_consistency_exact"] for r in rows} != {None}
    assert len({r["f5_self_consistency_exact"] for r in rows}) > 1, \
        "不同 prompt 的一致性比例不同，不该被聚合成同一个常数"
    assert len({r["f5_letter_agreement_rate"] for r in rows}) > 1


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


def test_h1_population_is_every_completion_not_just_greedy(tmp_path):
    """§3：H1 的主样本是**每条 completion**，K=6 → n prompts 应产出 6n 条记录。

    回归:早期 _build_feature_cache 对两臂都只取 greedy_0,H1 的 population 会从 ~2760
    静默塌到 480。形状对、分析照跑,只是有效样本量塌掉 → Δ_H 的 CI 宽到装不进 ±ε →
    该臂被判 inconclusive,而原因是代码 bug 不是真实精度限制。
    """
    n = 5
    res = _run("h1", tmp_path, _records(n, "h1"), gen._FakeBackend("h1", hidden_dim=8))
    assert res["status"] == "complete"
    assert res["feature_cache"]["n_records"] == n * gen.K_TRACES == 30
    assert res["feature_cache"]["n_hidden"] == n * gen.K_TRACES
    rows = [json.loads(l) for l in
            open(tmp_path / "feat" / "h1_completion_records.jsonl", encoding="utf-8")]
    # 每条 completion 一条记录，但同一 prompt 的 K 条共享 group（§3：不当独立样本）
    assert len({r["trace_id"] for r in rows}) == n * gen.K_TRACES
    assert len({r["group_id"] for r in rows}) == n


def test_mcq_population_stays_greedy_only_after_the_h1_fix(tmp_path):
    """同一次修改不得把 MCQ 也改成全 trace —— §3 v2.3：MCQ 每题一条 greedy。"""
    n = 5
    res = _run("mcq", tmp_path, _records(n, "mcq"), gen._FakeBackend("mcq", hidden_dim=8))
    assert res["feature_cache"]["n_records"] == n, "MCQ population 应是每题一条"


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
