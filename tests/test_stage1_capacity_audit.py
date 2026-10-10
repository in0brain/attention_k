"""Stage 1 容量审计的回归测试。

**不读取 `data/**` 或 `outputs/**`**：全部用合成 fixture（与 v3 其余测试同约定）。
测试保护三类东西：
  1. 冻结规则（E3 分块、MCQ 选择序、题面归一化）的确定性与 salt 的承重性；
  2. 计数口径 —— 尤其 open_gen 的 `"None"` 哨兵不得被计成 ontology 条目；
  3. 审计边界 —— 产物不含任何标签/性能量，且不物化正式样本。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "audit_stage1_capacity", _REPO_ROOT / "scripts" / "audit_stage1_capacity.py"
)
audit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit)

from recover_attention.cyber_data import _normalize_question_for_grouping  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────
def _write_ontology(root: Path, family: str, ids: list[str]) -> None:
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"family": family, "normalized_id": ident, "status": "active",
                    "name": f"n-{ident}", "description": None, "metadata": {}})
        for ident in ids
    ]
    (directory / "ontology_index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_h1_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _recall(family: str, entry_id: str) -> dict:
    return {"family": family, "route": "recall", "source_entry_id": entry_id}


def _open_gen(family: str) -> dict:
    # Stage 0 产物里 open_gen 的 source_entry_id 是**字符串** "None"，不是 JSON null。
    return {"family": family, "route": "open_gen", "source_entry_id": "None"}


def _write_cybermetric(path: Path, questions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"question": q, "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "A"}
        for q in questions
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def h1_tree(tmp_path: Path) -> tuple[Path, Path]:
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack", [f"T{1000 + i}" for i in range(10)])
    _write_ontology(ontology, "cwe", [f"CWE-{i}" for i in range(1, 9)])
    samples = tmp_path / "h1_samples.jsonl"
    _write_h1_samples(samples, [
        _recall("attack", "T1000"), _recall("attack", "T1001"), _recall("attack", "T1002"),
        _recall("cwe", "CWE-1"), _recall("cwe", "CWE-2"),
        _open_gen("attack"), _open_gen("cwe"),          # 不消耗条目
        {"family": "cve", "route": "recall", "source_entry_id": "CVE-2020-1000"},
    ])
    return ontology, samples


# ── 冻结规则 ─────────────────────────────────────────────────────────────────
def test_e3_block_is_deterministic_and_lands_in_range():
    ids = [f"T{1000 + i}" for i in range(200)] + [f"CWE-{i}" for i in range(1, 201)]
    first = [audit.e3_block(i) for i in ids]
    assert first == [audit.e3_block(i) for i in ids]
    assert set(first) <= set(range(audit.E3_F))
    assert len(set(first)) == audit.E3_F, "400 个 id 应覆盖全部 5 块"


def test_e3_block_matches_the_frozen_formula_bit_for_bit():
    import hashlib
    for ident in ("T1059.003", "CWE-79", "T1000"):
        expected = int(
            hashlib.sha256(f"{audit.E3_BLOCK_SALT}|{ident}".encode("utf-8")).hexdigest(), 16
        ) % audit.E3_F
        assert audit.e3_block(ident) == expected


def test_e3_salt_is_load_bearing(monkeypatch):
    ids = [f"T{1000 + i}" for i in range(200)]
    before = [audit.e3_block(i) for i in ids]
    monkeypatch.setattr(audit, "E3_BLOCK_SALT", audit.E3_BLOCK_SALT + "x")
    after = [audit.e3_block(i) for i in ids]
    assert before != after, "换 salt 必须改变分块 —— 否则 salt 没有承重，冻结它就没有意义"


def test_frozen_question_key_matches_the_src_normalizer():
    for question in ("  What   is\tTLS?  ", "MIXED Case\nQuestion", "a b"):
        assert audit.frozen_question_key(question) == _normalize_question_for_grouping(question)


def test_question_hash_collapses_whitespace_and_case():
    assert audit.frozen_question_hash("What  is TLS?") == audit.frozen_question_hash("what is\ttls?")
    assert audit.frozen_question_hash("a") != audit.frozen_question_hash("b")


def test_mcq_sort_key_is_deterministic_and_salted(monkeypatch):
    digest = audit.frozen_question_hash("q")
    assert audit.mcq_sort_key(digest) == audit.mcq_sort_key(digest)
    before = audit.mcq_sort_key(digest)
    monkeypatch.setattr(audit, "MCQ_SELECT_SALT", audit.MCQ_SELECT_SALT + "x")
    assert audit.mcq_sort_key(digest) != before


# ── H1 计数口径 ──────────────────────────────────────────────────────────────
def test_open_gen_sentinel_is_not_counted_as_an_ontology_entry(h1_tree):
    ontology, samples = h1_tree
    report = audit.audit_h1(ontology, samples)
    # attack 有 3 条 recall + 1 条 open_gen；只有 3 个条目被消耗
    assert report["stage0_consumed_entries"] == {"attack": 3, "cwe": 2}
    assert report["stage0_entryless_prompts"] == {"attack:open_gen": 1, "cwe:open_gen": 1}
    assert report["stage1_eligible_pool"] == {"attack": 7, "cwe": 6, "total": 13}


def test_non_primary_family_prompts_are_counted_but_not_consumed(h1_tree):
    ontology, samples = h1_tree
    report = audit.audit_h1(ontology, samples)
    assert report["stage0_consumed_non_primary_prompts"] == {"cve": 1}
    assert report["cve_excluded_from_primary"] is True
    assert report["stage0_prompts"] == 8


def test_pool_is_summarised_by_digest_not_materialised(h1_tree):
    ontology, samples = h1_tree
    report = audit.audit_h1(ontology, samples)
    assert report["pool_materialized"] is False
    assert len(report["eligible_pool_chain_sha256"]) == 64
    serialized = json.dumps(report)
    assert "T1003" not in serialized, "池的成员不得被物化进产物"


def test_capacity_ok_tracks_the_frozen_target(h1_tree, monkeypatch):
    ontology, samples = h1_tree
    monkeypatch.setattr(audit, "N_H1_TARGET", 13)
    ok = audit.audit_h1(ontology, samples)
    assert ok["capacity_ok"] is True and ok["capacity_headroom"] == 0
    assert ok["planned_traces"] == 13 * audit.K_TRACES
    monkeypatch.setattr(audit, "N_H1_TARGET", 14)
    short = audit.audit_h1(ontology, samples)
    assert short["capacity_ok"] is False and short["capacity_headroom"] == -1


def test_e3_distribution_is_recorded_as_non_binding(h1_tree):
    ontology, samples = h1_tree
    block = audit.audit_h1(ontology, samples)["e3_block_distribution_over_pool"]
    assert block["F"] == 5 and block["binding"] is False
    assert sum(block["counts"].values()) == 13


def test_unnormalised_ontology_id_is_rejected(tmp_path):
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack", ["T1000"])
    _write_ontology(ontology, "cwe", ["CWE-0079"])      # 前导零 → 非不动点
    samples = tmp_path / "h1_samples.jsonl"
    _write_h1_samples(samples, [_recall("attack", "T1000")])
    with pytest.raises(ValueError, match="不动点"):
        audit.audit_h1(ontology, samples)


def test_entry_used_by_stage0_but_absent_from_ontology_is_rejected(tmp_path):
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack", ["T1000"])
    _write_ontology(ontology, "cwe", ["CWE-79"])
    samples = tmp_path / "h1_samples.jsonl"
    _write_h1_samples(samples, [_recall("attack", "T9999")])
    with pytest.raises(ValueError, match="不在 ontology index"):
        audit.audit_h1(ontology, samples)


# ── MCQ 计数口径 ─────────────────────────────────────────────────────────────
@pytest.fixture()
def cyber_tree(tmp_path: Path) -> Path:
    cyber = tmp_path / "cybermetric"
    # 10000 池：q0..q9，外加 q0 的一个"仅空白/大小写不同"的重复 → 归一化后重复
    _write_cybermetric(cyber / "CyberMetric-10000-v1.json",
                       [f"question {i}?" for i in range(10)] + ["QUESTION  0?"])
    # 2000 池：q0..q2 在池内，另有 1 题不在 10000 里
    _write_cybermetric(cyber / "CyberMetric-2000-v1.json",
                       [f"question {i}?" for i in range(3)] + ["only in 2000?"])
    _write_cybermetric(cyber / "CyberMetric-500-v1.json", ["question 5?", "question 6?"])
    return cyber


def test_fresh_pool_uses_intra_file_dedup_and_explains_the_naive_number(cyber_tree):
    report = audit.audit_mcq(cyber_tree)
    acc = report["fresh_pool_accounting"]
    assert report["raw_record_counts"]["CyberMetric-10000-v1.json"] == 11
    assert report["distinct_question_counts"]["CyberMetric-10000-v1.json"] == 10
    assert report["intra_file_duplicates"]["CyberMetric-10000-v1.json"] == 1
    assert acc["overlap_with_2000"] == 3
    assert report["fresh_pool_size"] == 7                    # 10 distinct − 3 overlap
    assert acc["naive_without_intra_file_dedup"] == 8        # 11 raw − 3 overlap
    assert acc["authoritative"] == 7
    assert acc["questions_in_2000_absent_from_10000"] == 1


def test_500v1_overlap_is_disclosed_but_not_excluded(cyber_tree):
    secondary = audit.audit_mcq(cyber_tree)["disclosed_secondary"]
    assert secondary["fresh_also_present_in_500v1"] == 2
    assert secondary["pool_if_500v1_also_excluded"] == 5


def test_selection_rule_is_frozen_but_not_materialised(cyber_tree):
    report = audit.audit_mcq(cyber_tree)
    assert report["selection_materialized"] is False
    assert report["selection_rule"]["salt"] == audit.MCQ_SELECT_SALT
    assert report["selection_rule"]["take"] == audit.N_MCQ_TARGET
    serialized = json.dumps(report)
    assert "question 5?" not in serialized, "选中的题面不得出现在产物里"


def test_normalisation_drift_against_src_is_caught(cyber_tree, monkeypatch):
    monkeypatch.setattr(audit, "frozen_question_key", lambda q: q)   # 故意漂移
    with pytest.raises(ValueError, match="冻结规则漂移"):
        audit.audit_mcq(cyber_tree)


# ── 审计边界 ─────────────────────────────────────────────────────────────────
def test_report_declares_and_honours_the_no_label_boundary(h1_tree, cyber_tree, tmp_path):
    ontology, samples = h1_tree
    prereg = tmp_path / "prereg.md"
    prereg.write_text("# fake\n", encoding="utf-8")
    report = audit.build_report(
        ontology_dir=ontology, h1_samples=samples, cyber_dir=cyber_tree, prereg=prereg
    )
    assert report["reads_labels_or_performance"] is False
    assert report["prereg_v3_sha256"] == audit.sha256_file(prereg)

    forbidden = ("auroc", "delta_h", "ci_lo", "ci_hi", "width", "label", "positive")
    keys: Counter = Counter()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                keys[key.lower()] += 1
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(report)
    declaration_keys = {"reads_labels_or_performance"}   # 声明本身允许提到 label
    assert not [
        k for k in keys
        if k not in declaration_keys and any(bad in k for bad in forbidden)
    ]
