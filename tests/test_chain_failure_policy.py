"""接续链失败策略的合成故障验证（用户 2026-09-15 有限授权的七条限定）。

这里只测**判据**；端到端的真实 runner 合成故障见
`scripts/diagnostics/verify_chain_failure_policy.py`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chain_failure_policy as pol  # noqa: E402

NONFINITE_SEG = {"stopped_reason": pol.RESUMABLE_STOPPED_REASON, "n_failures": 1}
NORMAL_SEG = {"stopped_reason": "completed", "n_failures": 0}
PLAN2 = {"u1", "u2"}


def _mstate(events, path: Path) -> pol.ManifestState:
    path.write_text("".join(
        json.dumps({"unit_id": u, "status": s, "reason": r}, ensure_ascii=False) + "\n"
        for u, s, r in events), encoding="utf-8")
    return pol.manifest_state(path)


@pytest.fixture
def mk(tmp_path):
    n = {"i": 0}

    def make(events):
        n["i"] += 1
        return _mstate(events, tmp_path / f"m{n['i']}.jsonl")
    return make


# ══════════════════════════════════════════════════════════════════════════════
# 限定 5：完成判据绑定完整计划名单
# ══════════════════════════════════════════════════════════════════════════════
def test_complete_counts_excluded_and_matches_plan(mk):
    m = mk([("u1", "done", None), ("u2", "excluded", "forward_nonfinite")])
    ok, why = pol.arm_complete(m, ["u1"], PLAN2)          # excluded 没有结果行
    assert ok, why


def test_missing_unit_is_not_cancelled_by_manifest_total(mk):
    """残缺 manifest：少了一个 unit。拿 manifest 自己的 total 当预期就会误判完成。"""
    m = mk([("u1", "done", None)])
    assert m.total == 1 and m.all_terminal          # 旧判据在这里会说"完成"
    ok, why = pol.arm_complete(m, ["u1"], PLAN2)
    assert not ok and "与计划名单不符" in why and "缺 1 个" in why


def test_stray_unit_not_in_plan_is_refused(mk):
    m = mk([("u1", "done", None), ("u2", "done", None), ("u9", "done", None)])
    ok, why = pol.arm_complete(m, ["u1", "u2", "u9"], PLAN2)
    assert not ok and "多 1 个" in why


def test_duplicate_raw_rows_are_refused(mk):
    """重复 raw 不得被"数量凑够了"掩盖。"""
    m = mk([("u1", "done", None), ("u2", "done", None)])
    ok, why = pol.arm_complete(m, ["u1", "u1", "u2"], PLAN2)
    assert not ok and "重复行" in why


def test_duplicate_raw_cannot_offset_missing_done(mk):
    """u1 写了两行、u2 根本没行 —— 行数恰好 == done 数，但集合不符。"""
    m = mk([("u1", "done", None), ("u2", "done", None)])
    ok, why = pol.arm_complete(m, ["u1", "u1"], PLAN2)
    assert not ok and ("重复行" in why or "集合不符" in why)


def test_raw_set_must_equal_done_set(mk):
    m = mk([("u1", "done", None), ("u2", "excluded", "forward_nonfinite")])
    ok, why = pol.arm_complete(m, ["u2"], PLAN2)       # raw 里是被排除的那个
    assert not ok and "集合不符" in why


def test_nonterminal_unit_blocks_completion(mk):
    m = mk([("u1", "done", None), ("u2", "running", None)])
    ok, why = pol.arm_complete(m, ["u1"], PLAN2)
    assert not ok and "未进入终态" in why


# ══════════════════════════════════════════════════════════════════════════════
# 限定 1 + 4：退出归类。PID 消失不等于正常结束
# ══════════════════════════════════════════════════════════════════════════════
def test_no_segment_log_is_unconfirmed_not_normal():
    kind, why = pol.classify_exit(segments_new=[], returncode=None)
    assert kind == pol.OUTCOME_UNKNOWN and "PID 消失不等于正常结束" in why


def test_adopted_process_killed_by_oom_is_unconfirmed():
    """被接管的 H1 被 OOM killer 干掉：没有段日志、也没有退出码 —— 只能停。"""
    kind, _ = pol.classify_exit(segments_new=[], returncode=None)
    assert kind == pol.OUTCOME_UNKNOWN


@pytest.mark.parametrize("reason", list(pol.NORMAL_SEGMENT_REASONS))
def test_explicit_normal_segment_end(reason):
    kind, _ = pol.classify_exit(
        segments_new=[{"stopped_reason": reason, "n_failures": 0}], returncode=0)
    assert kind == pol.OUTCOME_NORMAL


def test_normal_reason_with_nonzero_returncode_is_unconfirmed():
    kind, why = pol.classify_exit(segments_new=[NORMAL_SEG], returncode=1)
    assert kind == pol.OUTCOME_UNKNOWN and "矛盾" in why


def test_nonfinite_reason_with_zero_returncode_is_unconfirmed():
    kind, why = pol.classify_exit(segments_new=[NONFINITE_SEG], returncode=0)
    assert kind == pol.OUTCOME_UNKNOWN and "矛盾" in why


@pytest.mark.parametrize("reason", ["unit_failure:OutOfMemoryError",
                                    "unit_failure:torch.cuda.OutOfMemoryError",
                                    "unit_failure:ValueError", "unit_failure:RuntimeError"])
def test_other_failures_classified_as_other(reason):
    kind, _ = pol.classify_exit(
        segments_new=[{"stopped_reason": reason, "n_failures": 1}], returncode=1)
    assert kind == pol.OUTCOME_OTHER


def test_multiple_new_segments_is_unconfirmed():
    kind, _ = pol.classify_exit(segments_new=[NORMAL_SEG, NORMAL_SEG], returncode=0)
    assert kind == pol.OUTCOME_UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# 限定 4：证据必须属于本次运行
# ══════════════════════════════════════════════════════════════════════════════
def test_stale_segment_log_cannot_explain_new_failure(mk):
    """旧段日志里的 forward_nonfinite 不得用来解释本次 OOM / 未知退出。

    边界之后**没有**新段日志 → segments_new 为空 → 无法确认 → 停。
    """
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None)])
    d = pol.decide_autoresume(segments_new=[], mstate_before=before, mstate_after=after,
                              new_diag=[], already_resumed=[], returncode=1)
    assert not d.allow and pol.OUTCOME_UNKNOWN in d.why


def test_diagnostic_prefix_must_be_unchanged():
    with pytest.raises(ValueError, match="诊断历史前缀被改动"):
        pol.new_diagnostics(["a", "b"], ["a", "X", "c"])


def test_diagnostic_shrink_is_refused():
    with pytest.raises(ValueError, match="诊断历史前缀被改动"):
        pol.new_diagnostics(["a", "b"], ["a"])


def test_new_diagnostics_returns_only_the_appended_ones():
    assert pol.new_diagnostics(["a", "b"], ["a", "b", "c"]) == ["c"]


def test_extra_failed_events_block_resume(mk):
    """本段除 forward_nonfinite 外还多了 failed 事件 —— 不放行。"""
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None), ("u2", "excluded", "forward_nonfinite"),
                ("u3", "failed", "RuntimeError")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u2"], already_resumed=[],
                              returncode=1)
    assert not d.allow and "failed 事件" in d.why


def test_diagnostic_row_without_unit_id_is_refused(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"reason": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺 unit_id"):
        pol.diagnostic_unit_ids(p)


# ══════════════════════════════════════════════════════════════════════════════
# 限定 1–3：放行与各种拒绝
# ══════════════════════════════════════════════════════════════════════════════
def test_allows_resume_when_all_three_sources_agree(mk):
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None), ("u2", "excluded", "forward_nonfinite")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u2"], already_resumed=[],
                              returncode=1)
    assert d.allow and d.unit_id == "u2"
    assert d.evidence["manifest_status"] == "excluded"


def test_stops_when_manifest_does_not_show_excluded(mk):
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None), ("u2", "failed", "RuntimeError")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u2"], already_resumed=[],
                              returncode=1)
    assert not d.allow


def test_stops_when_no_new_diagnostic(mk):
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None), ("u2", "excluded", "forward_nonfinite")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=[], already_resumed=[],
                              returncode=1)
    assert not d.allow and "诊断落盘不完整" in d.why


def test_stops_on_repeated_same_unit(mk):
    before = mk([("u2", "excluded", "forward_nonfinite")])
    after = mk([("u2", "excluded", "forward_nonfinite"), ("u3", "done", None)])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u2"],
                              already_resumed=["u2"], returncode=1)
    assert not d.allow and "重复出现同一事件" in d.why


def test_stops_when_no_new_terminal_progress(mk):
    before = mk([("u2", "excluded", "forward_nonfinite")])
    after = mk([("u2", "excluded", "forward_nonfinite")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u2"],
                              already_resumed=["other"], returncode=1)
    assert not d.allow and "没有新增终态进展" in d.why


def test_stops_at_cumulative_cap(mk):
    before = mk([("u1", "done", None)])
    after = mk([("u1", "done", None), ("u9", "excluded", "forward_nonfinite")])
    d = pol.decide_autoresume(segments_new=[NONFINITE_SEG], mstate_before=before,
                              mstate_after=after, new_diag=["u9"],
                              already_resumed=[f"u{i}" for i in range(5)], returncode=1)
    assert not d.allow and "已达上限 5" in d.why


def test_counter_persists_across_arms_and_restarts(tmp_path):
    p = tmp_path / "c.json"
    c = pol.load_counter(p)
    c = pol.record_autoresume(p, c, arm="h1", unit_id="h1u", when="t", evidence={})
    c = pol.record_autoresume(p, c, arm="mcq", unit_id="mq", when="t", evidence={})
    reloaded = pol.load_counter(p)                    # 模拟脚本重启
    assert reloaded["units"] == ["h1u", "mq"]
    assert pol.units_for_arm(reloaded, "h1") == ["h1u"]
    assert pol.units_for_arm(reloaded, "mcq") == ["mq"]


# ══════════════════════════════════════════════════════════════════════════════
# 限定 2 + 3：已排除 unit 消失 / 状态被改 —— 都要拒绝
# ══════════════════════════════════════════════════════════════════════════════
def test_registered_excluded_unit_missing_is_refused(mk):
    m = mk([("u1", "done", None)])
    ok, why = pol.verify_still_excluded(m, ["u2"])
    assert not ok and "消失" in why


def test_registered_excluded_unit_cleared_is_refused(mk):
    m = mk([("u2", "done", None)])
    ok, why = pol.verify_still_excluded(m, ["u2"])
    assert not ok and "被改动" in why


def test_registered_excluded_unit_reason_changed_is_refused(mk):
    m = mk([("u2", "excluded", "some_other_reason")])
    ok, why = pol.verify_still_excluded(m, ["u2"])
    assert not ok and "被改动" in why


def test_registered_excluded_unit_preserved_passes(mk):
    m = mk([("u2", "excluded", "forward_nonfinite")])
    ok, _ = pol.verify_still_excluded(m, ["u2"])
    assert ok


# ══════════════════════════════════════════════════════════════════════════════
# 计划名单规则
# ══════════════════════════════════════════════════════════════════════════════
def test_planned_unit_ids_follows_k_rule():
    got = pol.planned_unit_ids(["e0"], 6)
    assert got == {"e0__greedy_0"} | {f"e0__sampled_{i}" for i in range(1, 6)}
    assert len(pol.planned_unit_ids(["a", "b"], 6)) == 12
