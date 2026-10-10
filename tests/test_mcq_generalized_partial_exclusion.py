"""泛化的形态 B：sampled_j 失败（j∈[1,K-1]），整组不进 F5 / finalization。

    preserved = greedy_0 + sampled_1..j-1   j 条 done + raw（永久保留，但不被分析）
    trigger   = sampled_j                   excluded / forward_nonfinite
    suffix    = sampled_{j+1}..sampled_5    excluded / post_trigger_exclusion_reason

j = K-1 是历史特例（无 suffix，旧式 `excluded_unit_ids` 记录），必须**继续逐字节合法**。
suffix 的 reason 从**审计记录**的 `post_trigger_exclusion_reason` 读取并按白名单校验，
而不是从调用时的代码常量取值 —— 否则日后改常量会把既有审计记录重新解释一遍。
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as gb  # noqa: E402
from recover_attention.generation.manifest import WorkManifest  # noqa: E402

K = 6
PARTIAL_REASON = "partial_group_excluded_due_sampled_forward_nonfinite"
POST_REASON = "group_excluded_due_sampled_forward_nonfinite"


def _units(eid: str) -> list[str]:
    return [f"{eid}__greedy_0"] + [f"{eid}__sampled_{i}" for i in range(1, K)]


def _split(eid: str, j: int) -> tuple[list[str], str, list[str]]:
    u = _units(eid)
    return u[:j], u[j], u[j + 1:]


def _row(eid: str, i: int, sample_type: str) -> dict:
    return {"unit_id": f"{eid}__{sample_type}_{i}", "example_id": eid,
            "sample_type": sample_type, "sample_index": i,
            "eligible_for_primary": True, "parsed_label": "A",
            "f5_label_margin": 0.5, "f5_label_entropy": 0.5, "f5_full_entropy": 0.5,
            "f5_letter_token_logprob": -0.5,
            "f5_self_consistency_exact": 1.0, "f5_letter_agreement_rate": 1.0}


def _diag_bytes(uid: str) -> bytes:
    return json.dumps({"schema_version": "stage1_forward_nonfinite_diagnostic_v1",
                       "arm": "mcq", "unit_id": uid, "reason": "forward_nonfinite",
                       "row_nonfinite": 152064, "row_size": 152064},
                      ensure_ascii=False).encode("utf-8")


def _audit(eid: str, j: int, diag_sha: str, *, legacy: bool = False) -> dict:
    """legacy=True 复刻 cybermetric_000278 的旧式字段形状（只在 j=K-1 合法）。"""
    preserved, trigger, suffix = _split(eid, j)
    rec = {"schema_version": "stage1_technical_exclusion_v1", "arm": "mcq",
           "example_id": eid, "question_hash": "h" * 64, "group_id": f"grp_{eid}",
           "action": prod.ACTION_PARTIAL_GROUP,
           "trigger_unit_id": trigger, "trigger_sample_type": "sampled",
           "trigger_sample_index": j, "trigger_reason": "forward_nonfinite",
           "manifest_pattern": (f"{j}_done_1_excluded" if legacy
                                else f"{j}_done_1_excluded_{len(suffix)}_post_excluded"),
           "preserved_raw_unit_ids": preserved,
           "raw_preserved_not_analyzed": True,
           "exclusion_reason": PARTIAL_REASON,
           "diagnostic_unit_id": trigger, "diagnostic_record_sha256": diag_sha,
           "diagnostic_record_sha256_caliber": prod.DIAGNOSTIC_RECORD_SHA_CALIBER,
           "diagnostics_prefix_bytes_at_recovery": 0,
           "diagnostics_prefix_sha256_at_recovery": "0" * 64,
           "diagnostic_persisted": True,
           "frozen_list_sha256": "a" * 64, "run_fingerprint_sha256": "b" * 64,
           "classification": "runtime_technical_attrition",
           "protocol_deviation": True, "preregistration_files_modified": False}
    if legacy:
        rec["excluded_unit_ids"] = [trigger]
    else:
        rec["trigger_excluded_unit_id"] = trigger
        rec["post_trigger_excluded_unit_ids"] = suffix
        rec["post_trigger_exclusion_reason"] = POST_REASON
    return rec


def _build(tmp_path, *, partials: dict[str, int], n: int = 12, legacy: set | None = None,
           audit_over: dict | None = None, man_over=None):
    """partials = {example_id: j}；其余 example 为完整 K=6 组。"""
    legacy = legacy or set()
    p = prod.paths_for(tmp_path, "mcq")
    eids = [f"q{i:04d}" for i in range(n)]
    assert set(partials) <= set(eids)
    rows = []
    for e in eids:
        keep = range(K) if e not in partials else range(partials[e])
        for i in keep:
            rows.append(_row(e, i, "greedy" if i == 0 else "sampled"))
    p["raw_results"].parent.mkdir(parents=True, exist_ok=True)
    p["raw_results"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    man = WorkManifest(p["manifest"])
    man.register(u for e in eids for u in _units(e))
    for r in rows:
        man.mark(r["unit_id"], "done")
    for eid, j in partials.items():
        _pre, trigger, suffix = _split(eid, j)
        man.mark(trigger, "excluded", reason="forward_nonfinite")
        for uid in suffix:
            man.mark(uid, "excluded", reason=POST_REASON)
    if man_over:
        man_over(man)

    p["diagnostics"].write_bytes(b"".join(
        _diag_bytes(_split(e, j)[1]) + b"\r\n" for e, j in sorted(partials.items())))
    audit = []
    for eid, j in sorted(partials.items()):
        rec = _audit(eid, j, hashlib.sha256(_diag_bytes(_split(eid, j)[1])).hexdigest(),
                     legacy=eid in legacy)
        if audit_over and eid in audit_over:
            over = audit_over[eid]
            rec.update({k: v for k, v in over.items() if v is not prod})
            for k, v in over.items():
                if v is prod:                       # 哨兵：删除该字段
                    rec.pop(k, None)
        audit.append(rec)
    p["technical_exclusions"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit), encoding="utf-8")
    hd = p["hidden_dir"]
    hd.mkdir(parents=True, exist_ok=True)
    for r in rows:
        np.save(hd / f"{r['unit_id']}.npy", np.zeros(3584, dtype=np.float32))
    return p, [{"example_id": e, "group_id": f"grp_{e}"} for e in eids]


def _finalize(tmp_path, records):
    return prod.finalize_arm("mcq", tmp_path, records, cfg=V3AnalysisConfig(),
                             run_fingerprint={"run_hash": "x" * 64})


# ── 通则：j = 1..5 全部合法，三口径正确，F5 只见完整组 ──────────────────────
@pytest.mark.parametrize("j", [1, 2, 3, 4, 5])
def test_generalized_partial_j(tmp_path, monkeypatch, j):
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, partials={"q0003": j}, n=12,
                        legacy={"q0003"} if j == K - 1 else set())
    rep = _finalize(tmp_path, records)
    assert rep["technical_exclusions"] == 1
    assert rep["retained_questions"] == 11
    assert rep["actual_finalized_rows"] == 11
    assert rep["retained_raw_units"] == 11 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == j
    assert rep["expected_total_raw_rows"] == 11 * K + j
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 11}, "F5 只能收到完整 K=6 的 group"
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert len(fin) == 11 and all(r["example_id"] != "q0003" for r in fin)
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in raw if r["example_id"] == "q0003") == j     # 物理保留
    for uid in _split("q0003", j)[0]:
        assert (p["hidden_dir"] / f"{uid}.npy").exists()


def test_two_partial_groups_j1_and_j5(tmp_path, monkeypatch):
    """j=1 与 j=5 同时存在：过滤、F5、三口径计数均正确。"""
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, partials={"q0002": 1, "q0008": 5}, n=14,
                        legacy={"q0008"})
    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 14
    assert rep["technical_exclusions"] == 2
    assert rep["retained_questions"] == 12
    assert rep["actual_finalized_rows"] == 12
    assert rep["retained_raw_units"] == 12 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 6          # 1 + 5
    assert rep["expected_total_raw_rows"] == 12 * K + 6
    assert rep["excluded_example_ids"] == ["q0002", "q0008"]
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 12}
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert not ({r["example_id"] for r in fin} & {"q0002", "q0008"})


def test_three_partial_groups_j5_j1_j1(tmp_path, monkeypatch):
    """三个 partial 共存（j=5 旧式 + 两个 j=1 新式）：过滤、F5 与三口径均正确。

    对应 MCQ 臂的实际状态：000278(j=5, legacy) / 005478(j=1) / 005228(j=1)。
    """
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, partials={"q0002": 5, "q0007": 1, "q0011": 1},
                        n=16, legacy={"q0002"})
    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 16
    assert rep["technical_exclusions"] == 3
    assert rep["retained_questions"] == 13
    assert rep["actual_finalized_rows"] == 13
    assert rep["retained_raw_units"] == 13 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 7        # 5 + 1 + 1
    assert rep["expected_total_raw_rows"] == 13 * K + 7
    assert rep["excluded_example_ids"] == ["q0002", "q0007", "q0011"]
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 13}, "F5 只能收到 13 个完整 K=6 group"
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert not ({r["example_id"] for r in fin} & {"q0002", "q0007", "q0011"})
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    kept = collections.Counter(r["example_id"] for r in raw)
    assert (kept["q0002"], kept["q0007"], kept["q0011"]) == (5, 1, 1)   # 物理保留
    for eid in ("q0007", "q0011"):
        assert (p["hidden_dir"] / f"{eid}__greedy_0.npy").exists()


def test_four_partial_groups_j5_j1_j1_j2(tmp_path, monkeypatch):
    """四个 partial 共存（j=5 旧式 + j=1 + j=1 + j=2）：过滤、F5 与三口径均正确。

    对应 MCQ 臂的实际状态：000278(j=5, legacy) / 005478(j=1) / 005228(j=1) / 003232(j=2)。
    preserved 合计 5 + 1 + 1 + 2 = 9，全部保留在 raw 但都不进入分析。
    """
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    p, records = _build(tmp_path, n=18, legacy={"q0002"},
                        partials={"q0002": 5, "q0007": 1, "q0011": 1, "q0014": 2})
    rep = _finalize(tmp_path, records)
    assert rep["nominal_questions"] == 18
    assert rep["technical_exclusions"] == 4
    assert rep["retained_questions"] == 14
    assert rep["actual_finalized_rows"] == 14
    assert rep["retained_raw_units"] == 14 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == 9        # 5 + 1 + 1 + 2
    assert rep["expected_total_raw_rows"] == 14 * K + 9
    assert rep["excluded_example_ids"] == ["q0002", "q0007", "q0011", "q0014"]
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 14}, "F5 只能收到 14 个完整 K=6 group"
    fin = [json.loads(x) for x in p["records"].read_text(encoding="utf-8").splitlines()]
    assert not ({r["example_id"] for r in fin} & {"q0002", "q0007", "q0011", "q0014"})
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    kept = collections.Counter(r["example_id"] for r in raw)
    assert (kept["q0002"], kept["q0007"], kept["q0011"], kept["q0014"]) == (5, 1, 1, 2)
    for uid in ("q0014__greedy_0", "q0014__sampled_1"):            # j=2 的两条物理保留
        assert (p["hidden_dir"] / f"{uid}.npy").exists()


def test_five_partial_groups_mixed_j(tmp_path, monkeypatch):
    """五个 partial 共存（j = 5 旧式, 1, 1, 2, 1）：preserved 合计 10，三口径正确。

    对应 MCQ 臂的实际状态：000278(j=5, legacy) / 005478(1) / 005228(1) /
    003232(2) / 002391(1)。preserved 5+1+1+2+1 = 10 全部保留在 raw 但都不进入分析。
    """
    seen = {}
    monkeypatch.setattr(gb, "attach_cross_sample_f5",
                        lambda rows, arm: seen.update(rows=list(rows)))
    partials = {"q0002": 5, "q0007": 1, "q0011": 1, "q0014": 2, "q0017": 1}
    p, records = _build(tmp_path, n=20, legacy={"q0002"}, partials=partials)
    rep = _finalize(tmp_path, records)
    assert rep["technical_exclusions"] == 5
    assert rep["retained_questions"] == 15
    assert rep["actual_finalized_rows"] == 15
    assert rep["retained_raw_units"] == 15 * K
    assert rep["preserved_but_not_analyzed_raw_units"] == sum(partials.values()) == 10
    assert rep["expected_total_raw_rows"] == 15 * K + 10
    assert rep["excluded_example_ids"] == sorted(partials)
    sizes = collections.Counter(collections.Counter(
        r["example_id"] for r in seen["rows"]).values())
    assert sizes == {K: 15}, "F5 只能收到 15 个完整 K=6 group"
    raw = [json.loads(x) for x in p["raw_results"].read_text(encoding="utf-8").splitlines()]
    kept = collections.Counter(r["example_id"] for r in raw)
    assert {e: kept[e] for e in partials} == partials      # 每组保留数 == 各自的 j


def test_legacy_j5_record_bytes_are_still_accepted(tmp_path, monkeypatch):
    """旧式 j=K-1 记录（excluded_unit_ids，无 post_* 字段）**逐字节不变**仍通过。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _build(tmp_path, partials={"q0004": 5}, legacy={"q0004"})
    before = p["technical_exclusions"].read_bytes()
    rec = json.loads(before.decode("utf-8").splitlines()[0])
    assert rec["excluded_unit_ids"] == ["q0004__sampled_5"]
    assert "trigger_excluded_unit_id" not in rec
    assert "post_trigger_exclusion_reason" not in rec
    assert _finalize(tmp_path, records)["technical_exclusions"] == 1
    assert p["technical_exclusions"].read_bytes() == before          # finalize 不改审计真源


# ── 拒绝路径 ────────────────────────────────────────────────────────────────
def test_rejects_list_valued_trigger_excluded_unit_id(tmp_path, monkeypatch):
    """`trigger_excluded_unit_id` 必须是标量字符串，单元素列表也拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 1},
                        audit_over={"q0003": {"trigger_excluded_unit_id": ["q0003__sampled_1"]}})
    with pytest.raises(ValueError, match="必须是标量字符串"):
        _finalize(tmp_path, records)


def test_rejects_missing_post_trigger_exclusion_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 1},
                        audit_over={"q0003": {"post_trigger_exclusion_reason": prod}})
    with pytest.raises(ValueError, match="缺 post_trigger_exclusion_reason"):
        _finalize(tmp_path, records)


def test_rejects_non_whitelisted_post_trigger_exclusion_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 1},
                        audit_over={"q0003": {"post_trigger_exclusion_reason": "just_drop_it"}})
    with pytest.raises(ValueError, match="post_trigger_exclusion_reason.*不在白名单"):
        _finalize(tmp_path, records)


def test_rejects_group_level_reason_used_for_suffix(tmp_path, monkeypatch):
    """组级 exclusion_reason 字符串不能拿来当 suffix 的 manifest reason。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 1},
                        audit_over={"q0003": {"post_trigger_exclusion_reason": PARTIAL_REASON}})
    with pytest.raises(ValueError, match="不在白名单"):
        _finalize(tmp_path, records)


@pytest.mark.parametrize("bad_idx", [2, 3, 4, 5])
def test_rejects_suffix_manifest_reason_mismatch(tmp_path, monkeypatch, bad_idx):
    """sampled_2..5 任一 manifest reason 不匹配即拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 1},
        man_over=lambda m: m.mark(f"q0003__sampled_{bad_idx}", "excluded", reason="something_else"))
    with pytest.raises(ValueError, match="manifest latest reason"):
        _finalize(tmp_path, records)


@pytest.mark.parametrize("bad_idx", [2, 3, 4, 5])
def test_rejects_suffix_still_pending(tmp_path, monkeypatch, bad_idx):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 1},
        man_over=lambda m: m.mark(f"q0003__sampled_{bad_idx}", "pending"))
    with pytest.raises(ValueError, match="期望 'excluded'"):
        _finalize(tmp_path, records)


def test_rejects_non_contiguous_preserved_prefix(tmp_path, monkeypatch):
    """preserved 必须是从 greedy_0 起的连续前缀；跳号即拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 3},
        audit_over={"q0003": {"preserved_raw_unit_ids":
                              ["q0003__greedy_0", "q0003__sampled_2", "q0003__sampled_1"]}})
    with pytest.raises(ValueError, match="连续前缀"):
        _finalize(tmp_path, records)


def test_rejects_preserved_without_greedy(tmp_path, monkeypatch):
    """preserved 必须从 greedy_0 起；丢掉 greedy 的前缀即拒绝（三段无交集，隔离该检查）。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 3},
        audit_over={"q0003": {"preserved_raw_unit_ids":
                              ["q0003__sampled_1", "q0003__sampled_2"]}})
    with pytest.raises(ValueError, match="连续前缀"):
        _finalize(tmp_path, records)


def test_rejects_trigger_not_next_sampled(tmp_path, monkeypatch):
    """trigger 必须恰为 sampled_{len(preserved)}（suffix 同步下移以避开集合相交检查）。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 1},
        audit_over={"q0003": {"trigger_unit_id": "q0003__sampled_2",
                              "trigger_excluded_unit_id": "q0003__sampled_2",
                              "post_trigger_excluded_unit_ids":
                                  ["q0003__sampled_3", "q0003__sampled_4", "q0003__sampled_5"]}})
    with pytest.raises(ValueError, match="按 preserved 长度应为"):
        _finalize(tmp_path, records)


def test_rejects_wrong_suffix_set(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(
        tmp_path, partials={"q0003": 1},
        audit_over={"q0003": {"post_trigger_excluded_unit_ids":
                              ["q0003__sampled_2", "q0003__sampled_3"]}})
    with pytest.raises(ValueError, match="post-trigger unit 集合"):
        _finalize(tmp_path, records)


def test_rejects_both_field_shapes(tmp_path, monkeypatch):
    """新旧字段并存 → 授权形状有歧义，拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 1},
                        audit_over={"q0003": {"excluded_unit_ids": ["q0003__sampled_1"]}})
    with pytest.raises(ValueError, match="授权形状有歧义"):
        _finalize(tmp_path, records)


def test_rejects_legacy_shape_with_nonempty_suffix(tmp_path, monkeypatch):
    """旧式记录（无 post_* 字段）在 j < K-1 时欠定义 → 拒绝。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    _, records = _build(tmp_path, partials={"q0003": 2}, legacy={"q0003"})
    with pytest.raises(ValueError, match="post-trigger unit 集合"):
        _finalize(tmp_path, records)


def test_rejects_raw_set_beyond_preserved_prefix(tmp_path, monkeypatch):
    """raw 集合必须精确等于 preserved prefix。"""
    monkeypatch.setattr(gb, "attach_cross_sample_f5", lambda rows, arm: None)
    p, records = _build(tmp_path, partials={"q0003": 1})
    with open(p["raw_results"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row("q0003", 2, "sampled")) + "\n")
    with pytest.raises(ValueError, match="raw 集合与 preserved_raw_unit_ids 不符|尚未跑完"):
        _finalize(tmp_path, records)
