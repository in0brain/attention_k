"""Stage 1 sample-builder 的回归测试（G2 + G5 + 原子落盘）。

**全部用合成 fixture**，不读 `data/**`；只有一项例外，读仓库里的真实 prereg/lock
做 G2 接线的连通性检查（不读数据）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_sample_builder as sb  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.evaluation.v3_freeze import V3_LOCK_VERSION, sha256_text_lf  # noqa: E402
from recover_attention.h1_identifier import OntologyEntry  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────
def _entry(family: str, ident: str, name: str = "N", desc: str = "D") -> OntologyEntry:
    return OntologyEntry(family=family, normalized_id=ident, status="active",
                         name=name, description=desc, metadata={})


def _write_ontology(root: Path, family: str, rows: list[tuple[str, str, str]]) -> None:
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"family": family, "normalized_id": i, "status": "active",
                         "name": n, "description": d, "metadata": {}}) for i, n, d in rows]
    (directory / "ontology_index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage0_row(family: str, ident: str, name: str, desc: str) -> dict:
    entry = _entry(family, ident, name, desc)
    record = sb.make_h1_record(f"{family}:{ident}", entry)
    return {"example_id": record["example_id"], "route": "recall", "family": family,
            "prompt_params": record["prompt_params"], "question_text": record["question_text"],
            "source_entry_id": ident, "group_id": record["group_id"], "split": "train",
            "label_space": "open_identifier"}


@pytest.fixture()
def tree(tmp_path: Path):
    """8 个 attack + 8 个 cwe；每族前 4 个已被 Stage 0 烧过 → fresh 4+4，burned 4+4。"""
    ontology = tmp_path / "ontology"
    for family in ("attack", "cwe"):
        prefix = "T" if family == "attack" else "CWE-"
        rows = [(f"{prefix}{1000 + i}", f"name-{family}-{i}", f"desc-{family}-{i}")
                for i in range(8)]
        _write_ontology(ontology, family, rows)

    stage0 = tmp_path / "h1_samples.jsonl"
    burned = []
    for family in ("attack", "cwe"):
        prefix = "T" if family == "attack" else "CWE-"
        for i in range(4):
            burned.append(_stage0_row(family, f"{prefix}{1000 + i}",
                                      f"name-{family}-{i}", f"desc-{family}-{i}"))
    stage0.write_text("\n".join(json.dumps(r) for r in burned) + "\n", encoding="utf-8")
    return ontology, stage0


def _cfg(**over) -> V3AnalysisConfig:
    base = dict(mode="dry_run", n_h1_eligible_pool=8, n_h1_prompts=8, n_mcq_questions=4,
                n_g3_per_arm=2, n_g4_per_arm=4)
    base.update(over)
    return V3AnalysisConfig(**base)


# ── g5_prompt_hash ───────────────────────────────────────────────────────────
def test_g5_prompt_hash_is_not_sha256_of_question_text():
    """裁决明确：不得定义为 sha256(question_text)。它取的是 chat messages 的 canonical JSON。"""
    import hashlib
    record = sb.make_h1_record("attack:T1000", _entry("attack", "T1000"))
    naive = hashlib.sha256(record["question_text"].encode("utf-8")).hexdigest()
    assert record["g5_prompt_hash"] != naive


def test_g5_prompt_hash_covers_the_system_prompt(monkeypatch):
    """只哈希 question_text 会漏掉 system prompt 的变化；本口径不会。"""
    record = sb.make_h1_record("attack:T1000", _entry("attack", "T1000"))
    before = sb.g5_prompt_hash(record)
    monkeypatch.setattr(sb, "build_h1_chat_messages", lambda r: [
        {"role": "system", "content": "DIFFERENT SYSTEM PROMPT"},
        {"role": "user", "content": r["question_text"]}])
    assert sb.g5_prompt_hash(record) != before


def test_g5_prompt_hash_is_line_ending_invariant():
    """Stage 0 与 Stage 1 用同一规则重算 —— 行尾差异不得制造假的"不同 prompt"。"""
    lf = {"question_text": "a\nb\n"}
    crlf = {"question_text": "a\r\nb\r\n"}
    assert sb.g5_prompt_hash(lf) == sb.g5_prompt_hash(crlf)


def test_g5_prompt_hash_still_distinguishes_real_differences():
    assert sb.g5_prompt_hash({"question_text": "a b"}) != sb.g5_prompt_hash({"question_text": "a  b"})


# ── H1 池与分组 ──────────────────────────────────────────────────────────────
def test_pools_split_fresh_and_burned(tree):
    ontology, stage0 = tree
    pools = sb.load_h1_pools(ontology, stage0)
    assert len(pools.fresh_ids) == 8 and len(pools.burned_ids) == 8
    assert not set(pools.fresh_ids) & set(pools.burned_ids)


def test_group_id_comes_from_prompt_identity_not_entry_id():
    """revoked 条目与继任者共享 name+description 时，两行必须落在同一个 group。"""
    a = sb.make_h1_record("attack:T1536", _entry("attack", "T1536", "Same", "Same desc"))
    b = sb.make_h1_record("attack:T1578.004", _entry("attack", "T1578.004", "Same", "Same desc"))
    assert a["source_entry_id"] != b["source_entry_id"]
    assert a["g5_prompt_hash"] == b["g5_prompt_hash"]
    assert a["group_id"] == b["group_id"], "同一条 prompt 必须共享 group，否则会跨 fold"


def _dedup_tree(tmp_path: Path, rows: list[tuple[str, str, str]],
                stage0: list[tuple[str, str, str]]):
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack", rows)
    _write_ontology(ontology, "cwe", [("CWE-1", "n", "d")])
    s0 = tmp_path / "s.jsonl"
    s0.write_text("".join(json.dumps(_stage0_row("attack", i, n, d)) + "\n"
                          for i, n, d in stage0), encoding="utf-8")
    return sb.load_h1_pools(ontology, s0)


def _with_status(tmp_path: Path, rows: list[tuple[str, str, str, str]]):
    """rows = (id, name, desc, status)。用于验证"优先保留非 revoked/deprecated"。"""
    directory = tmp_path / "ontology" / "attack"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ontology_index.jsonl").write_text(
        "".join(json.dumps({"family": "attack", "normalized_id": i, "status": st,
                            "name": n, "description": d, "metadata": {}}) + "\n"
                for i, n, d, st in rows), encoding="utf-8")
    _write_ontology(tmp_path / "ontology", "cwe", [("CWE-1", "n", "d")])
    s0 = tmp_path / "s.jsonl"
    s0.write_text("", encoding="utf-8")
    return sb.load_h1_pools(tmp_path / "ontology", s0)


def test_cross_stage_prompt_collision_is_excluded(tmp_path):
    pools = _dedup_tree(
        tmp_path,
        rows=[("T1000", "shared", "shared d"), ("T2000", "shared", "shared d"),
              ("T3000", "own", "own d")],
        stage0=[("T1000", "shared", "shared d")])
    primary, excluded = sb.select_h1_primary(pools)
    assert "attack:T2000" not in primary, "prompt 与 Stage 0 相同的条目必须被排除"
    assert [e["reason"] for e in excluded] == ["cross_stage_prompt_collision"]
    assert excluded[0]["entry"] == "attack:T2000"


def test_intra_set_duplicate_prefers_the_non_retired_entry(tmp_path):
    pools = _with_status(tmp_path, [
        ("T1536", "same", "same d", "revoked"),
        ("T1578.004", "same", "same d", "active"),
    ])
    primary, excluded = sb.select_h1_primary(pools)
    assert "attack:T1578.004" in primary and "attack:T1536" not in primary
    assert excluded[0]["reason"] == "intra_set_duplicate_prompt"
    assert excluded[0]["kept_instead"] == "attack:T1578.004"


@pytest.mark.parametrize("status", ["revoked", "deprecated", "Deprecated", "REVOKED"])
def test_retired_status_matching_is_case_insensitive(tmp_path, status):
    pools = _with_status(tmp_path, [
        ("T1000", "same", "same d", status),
        ("T2000", "same", "same d", "active"),
    ])
    primary, _ = sb.select_h1_primary(pools)
    assert "attack:T2000" in primary and "attack:T1000" not in primary


def test_tie_between_two_active_entries_falls_back_to_lexicographic(tmp_path):
    pools = _with_status(tmp_path, [
        ("T2000", "same", "same d", "active"),
        ("T1000", "same", "same d", "active"),
    ])
    primary, excluded = sb.select_h1_primary(pools)
    assert "attack:T1000" in primary and "attack:T2000" not in primary
    assert excluded[0]["kept_instead"] == "attack:T1000"


def test_two_retired_entries_also_fall_back_to_lexicographic(tmp_path):
    pools = _with_status(tmp_path, [
        ("T2000", "same", "same d", "revoked"),
        ("T1000", "same", "same d", "deprecated"),
    ])
    primary, _ = sb.select_h1_primary(pools)
    assert "attack:T1000" in primary


def test_retired_entries_are_kept_when_they_are_not_duplicates(tmp_path):
    """revoked/deprecated **不作整体排除** —— estimand 问的是"是否存在于冻结 ontology"。"""
    pools = _with_status(tmp_path, [
        ("T1000", "a", "a d", "revoked"),
        ("T2000", "b", "b d", "deprecated"),
        ("T3000", "c", "c d", "active"),
    ])
    primary, excluded = sb.select_h1_primary(pools)
    assert set(primary) == {"attack:T1000", "attack:T2000", "attack:T3000", "cwe:CWE-1"}
    assert excluded == []


def test_selection_is_order_independent_and_deterministic(tmp_path):
    a = _with_status(tmp_path / "a", [("T1000", "s", "s", "revoked"),
                                      ("T2000", "s", "s", "active")])
    b = _with_status(tmp_path / "b", [("T2000", "s", "s", "active"),
                                      ("T1000", "s", "s", "revoked")])
    assert sb.select_h1_primary(a)[0] == sb.select_h1_primary(b)[0]


def test_entry_without_metadata_is_refused(tmp_path):
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack", [("T1000", "", "")])
    _write_ontology(ontology, "cwe", [("CWE-1", "n", "d")])
    stage0 = tmp_path / "s.jsonl"
    stage0.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="既无 name 也无 description"):
        sb.load_h1_pools(ontology, stage0)


# ── gate 选择 ────────────────────────────────────────────────────────────────
def test_gate_selection_is_deterministic_and_family_balanced(tree):
    ontology, stage0 = tree
    pools = sb.load_h1_pools(ontology, stage0)
    cfg = _cfg()
    first = sb.select_h1_gate(pools, cfg)
    assert first == sb.select_h1_gate(pools, cfg)
    assert len(first["G3"]) == 2 and len(first["G4"]) == 4
    for gate, per_family in (("G3", 1), ("G4", 2)):
        for family in ("attack", "cwe"):
            assert sum(k.startswith(f"{family}:") for k in first[gate]) == per_family
    assert not set(first["G3"]) & set(first["G4"])
    assert set(first["G3"]) | set(first["G4"]) <= set(pools.burned_ids)


def test_gate_salt_is_load_bearing(tree):
    ontology, stage0 = tree
    pools = sb.load_h1_pools(ontology, stage0)
    base = sb.select_h1_gate(pools, _cfg())
    other = sb.select_h1_gate(pools, _cfg(h1_gate_selection_salt="different-salt"))
    assert (base["G3"], base["G4"]) != (other["G3"], other["G4"])


def test_gate_pool_too_small_is_refused(tree):
    ontology, stage0 = tree
    pools = sb.load_h1_pools(ontology, stage0)
    with pytest.raises(ValueError, match="不足"):
        sb.select_h1_gate(pools, _cfg(n_g3_per_arm=4, n_g4_per_arm=8))


# ── MCQ 排序与切片 ───────────────────────────────────────────────────────────
def _write_cyber(root: Path, pool: list[str], stage0: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    def dump(name, qs):
        (root / name).write_text(json.dumps(
            [{"question": q, "answers": {"A": "a", "B": "b"}, "solution": "A"} for q in qs]),
            encoding="utf-8")
    dump("CyberMetric-10000-v1.json", pool)
    dump("CyberMetric-2000-v1.json", stage0)
    dump("CyberMetric-500-v1.json", [])
    return root


def test_mcq_pool_excludes_stage0_and_intra_file_duplicates(tmp_path):
    cyber = _write_cyber(tmp_path / "c",
                         pool=[f"q{i}?" for i in range(10)] + ["Q0?", "q3?"],
                         stage0=["q0?", "q1?"])
    pools = sb.load_mcq_pools(cyber, _cfg())
    assert len(pools.ranked_fresh) == 8            # 10 distinct − 2 与 Stage 0 重叠
    hashes = [r["question_hash"] for r in pools.ranked_fresh]
    assert len(set(hashes)) == 8
    assert not set(hashes) & pools.stage0_hashes


def test_mcq_slices_are_contiguous_by_rank_and_disjoint(tmp_path):
    cyber = _write_cyber(tmp_path / "c", pool=[f"q{i}?" for i in range(20)], stage0=[])
    cfg = _cfg()
    pools = sb.load_mcq_pools(cyber, cfg)
    ranks = [r["rank"] for r in pools.ranked_fresh]
    assert ranks == sorted(ranks) == list(range(20))

    # 切片本身（正式集 → G3 → G4 依名次连续、互不重叠）
    primary, g3, g4 = pools.ranked_fresh[:4], pools.ranked_fresh[4:6], pools.ranked_fresh[6:10]
    assert [r["rank"] for r in primary] == [0, 1, 2, 3]
    assert [r["rank"] for r in g3] == [4, 5]
    assert [r["rank"] for r in g4] == [6, 7, 8, 9]
    ids = [{r["question_hash"] for r in part} for part in (primary, g3, g4)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_mcq_ranking_is_salt_dependent(tmp_path):
    cyber = _write_cyber(tmp_path / "c", pool=[f"q{i}?" for i in range(20)], stage0=[])
    a = [r["question_hash"] for r in sb.load_mcq_pools(cyber, _cfg()).ranked_fresh]
    b = [r["question_hash"] for r in
         sb.load_mcq_pools(cyber, _cfg(mcq_selection_salt="other")).ranked_fresh]
    assert a != b and set(a) == set(b)


# ── 端到端 + 原子性 ──────────────────────────────────────────────────────────
def _full_tree(tmp_path: Path):
    ontology, stage0 = None, None
    ontology = tmp_path / "ontology"
    for family in ("attack", "cwe"):
        prefix = "T" if family == "attack" else "CWE-"
        _write_ontology(ontology, family,
                        [(f"{prefix}{1000 + i}", f"n-{family}-{i}", f"d-{family}-{i}")
                         for i in range(8)])
    stage0 = tmp_path / "h1_samples.jsonl"
    rows = []
    for family in ("attack", "cwe"):
        prefix = "T" if family == "attack" else "CWE-"
        for i in range(4):
            rows.append(_stage0_row(family, f"{prefix}{1000 + i}",
                                    f"n-{family}-{i}", f"d-{family}-{i}"))
    stage0.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    cyber = _write_cyber(tmp_path / "c", pool=[f"q{i}?" for i in range(20)], stage0=["s0?"])
    pools = sb.load_h1_pools(ontology, stage0)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "h1": {"eligible_pool_chain_sha256": sb.chain_digest(pools.fresh_ids)},
        "mcq": {"fresh_pool_size": 20}}), encoding="utf-8")

    prereg = tmp_path / "preregistration_v3.md"
    prereg.write_bytes(b"# frozen design\n")
    lock = tmp_path / "preregistration_v3.lock"
    lock.write_text(
        "file:      docs/paper/preregistration_v3.md\n"
        f"version:   {V3_LOCK_VERSION}\n"
        "hash_mode: text-lf-utf8\n"
        f"sha256:    {sha256_text_lf(prereg)}\nfrozen_on: 2026-08-08\n", encoding="utf-8")
    return dict(ontology_dir=ontology, stage0_samples=stage0, cyber_dir=cyber,
                capacity_audit=audit, prereg=prereg, lock=lock)


def test_end_to_end_builds_all_six_lists_and_a_manifest(tmp_path):
    args = _full_tree(tmp_path)
    out = tmp_path / "samples"
    manifest = sb.build_stage1_samples(out_dir=out, cfg=_cfg(), **args)

    assert manifest["counts"]["h1"] == {"primary": 8, "G3": 2, "G4": 4}
    assert manifest["counts"]["mcq"] == {"primary": 4, "G3": 2, "G4": 4}
    assert manifest["gates"]["G2"]["match"] is True
    assert manifest["prereg"]["hash_mode"] == "text-lf-utf8"
    assert manifest["boundary"]["no_model_was_run"] is True
    for name in ("h1_primary_entries.jsonl", "h1_g3_entries.jsonl", "h1_g4_entries.jsonl",
                 "mcq_primary_questions.jsonl", "mcq_g3_questions.jsonl",
                 "mcq_g4_questions.jsonl", sb.MANIFEST_NAME):
        assert (out / name).exists()
    assert not (out / "_staging").exists()

    gate_rows = [json.loads(l) for l in (out / "h1_g3_entries.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert all(r["gate_only"] and r["previously_burned"] and r["gate"] == "G3" for r in gate_rows)
    primary_rows = [json.loads(l) for l in (out / "h1_primary_entries.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert all("gate" not in r for r in primary_rows)

    mcq_rows = {name: [json.loads(l) for l in (out / f"mcq_{key}_questions.jsonl").read_text(
        encoding="utf-8").splitlines()] for name, key in
        (("primary", "primary"), ("G3", "g3"), ("G4", "g4"))}
    assert [r["rank"] for r in mcq_rows["primary"]] == [0, 1, 2, 3]
    assert [r["rank"] for r in mcq_rows["G3"]] == [4, 5]
    assert [r["rank"] for r in mcq_rows["G4"]] == [6, 7, 8, 9]
    assert all(r["gate_only"] and not r["previously_burned"]
               for r in mcq_rows["G3"] + mcq_rows["G4"])


def test_g2_failure_creates_nothing(tmp_path):
    args = _full_tree(tmp_path)
    args["prereg"].write_bytes(b"# frozen design TAMPERED\n")
    out = tmp_path / "samples"
    with pytest.raises(ValueError, match="G2 失败"):
        sb.build_stage1_samples(out_dir=out, cfg=_cfg(), **args)
    assert not out.exists() or not any(out.iterdir()), "G2 不过时不得创建任何名单"


def test_chain_hash_mismatch_stops_the_build(tmp_path):
    args = _full_tree(tmp_path)
    args["capacity_audit"].write_text(json.dumps({
        "h1": {"eligible_pool_chain_sha256": "0" * 64}, "mcq": {"fresh_pool_size": 20}}),
        encoding="utf-8")
    out = tmp_path / "samples"
    with pytest.raises(ValueError, match="chain hash"):
        sb.build_stage1_samples(out_dir=out, cfg=_cfg(), **args)
    assert not out.exists() or not any(out.iterdir())


def test_frozen_eligible_pool_mismatch_stops_the_build(tmp_path):
    """源数据变动导致 eligible pool 变了 → 立即炸，不得静默顺应。"""
    args = _full_tree(tmp_path)
    out = tmp_path / "samples"
    with pytest.raises(ValueError, match="不得因源数据变动而静默调整"):
        sb.build_stage1_samples(out_dir=out, cfg=_cfg(n_h1_eligible_pool=7), **args)


def test_frozen_primary_size_mismatch_stops_the_build(tmp_path):
    """排除后的数目与冻结的 N_H1 不符 → 也必须炸（报错要给出 eligible − 排除 的账）。"""
    args = _full_tree(tmp_path)
    out = tmp_path / "samples"
    with pytest.raises(ValueError, match=r"排除后正式集 8 != 冻结的 N_H1=7"):
        sb.build_stage1_samples(out_dir=out, cfg=_cfg(n_h1_prompts=7), **args)


def test_failure_midway_leaves_no_half_written_output(tmp_path, monkeypatch):
    """写盘阶段炸掉 —— staging 必须被清掉，正式目录不得出现半成品。"""
    args = _full_tree(tmp_path)
    out = tmp_path / "samples"
    sb.build_stage1_samples(out_dir=out, cfg=_cfg(), **args)          # 先建一份完整的
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    calls = {"n": 0}
    real = sb._write_jsonl

    def boom(path, rows):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        real(path, rows)

    monkeypatch.setattr(sb, "_write_jsonl", boom)
    with pytest.raises(OSError):
        sb.build_stage1_samples(out_dir=out, cfg=_cfg(), **args)
    assert not (out / "_staging").exists()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before, "旧产物不得被半成品覆盖"


# ── G5 的否定用例 ────────────────────────────────────────────────────────────
def test_g5_rejects_a_primary_entry_that_stage0_already_used(tmp_path):
    args = _full_tree(tmp_path)
    pools = sb.load_h1_pools(args["ontology_dir"], args["stage0_samples"])
    mcq_pools = sb.load_mcq_pools(args["cyber_dir"], _cfg())
    h1, mcq, _excluded = sb.build_sample_sets(pools, mcq_pools, _cfg())
    h1["primary"][0]["source_entry_id"] = pools.burned_ids[0].split(":", 1)[1]
    with pytest.raises(ValueError, match="与 Stage 0 条目交集非空"):
        sb.run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)


def test_g5_enforces_records_eq_prompts_eq_groups(tmp_path):
    """v3.1 的核心恒等式：正式集内 prompt 零重复，且每条 prompt 恰好一个 group。"""
    args = _full_tree(tmp_path)
    pools = sb.load_h1_pools(args["ontology_dir"], args["stage0_samples"])
    mcq_pools = sb.load_mcq_pools(args["cyber_dir"], _cfg())
    h1, mcq, _excluded = sb.build_sample_sets(pools, mcq_pools, _cfg())
    h1["primary"][1]["g5_prompt_hash"] = h1["primary"][0]["g5_prompt_hash"]   # 制造重复 prompt
    with pytest.raises(ValueError, match="records = distinct prompts = groups"):
        sb.run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)


def test_g5_rejects_a_gate_entry_drawn_from_the_fresh_pool(tmp_path):
    args = _full_tree(tmp_path)
    pools = sb.load_h1_pools(args["ontology_dir"], args["stage0_samples"])
    mcq_pools = sb.load_mcq_pools(args["cyber_dir"], _cfg())
    h1, mcq, _excluded = sb.build_sample_sets(pools, mcq_pools, _cfg())
    # 用一个既不在正式集也不在已烧池里的 id：这样先命中的正是"gate 必须来自已烧池"这一条
    h1["G3"][0]["source_entry_id"] = "T9999"
    with pytest.raises(ValueError, match="gate 不得消耗正式新池"):
        sb.run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)


def test_g5_rejects_a_gate_entry_that_is_also_in_the_primary_set(tmp_path):
    args = _full_tree(tmp_path)
    pools = sb.load_h1_pools(args["ontology_dir"], args["stage0_samples"])
    mcq_pools = sb.load_mcq_pools(args["cyber_dir"], _cfg())
    h1, mcq, _excluded = sb.build_sample_sets(pools, mcq_pools, _cfg())
    h1["G3"][0]["source_entry_id"] = pools.fresh_ids[0].split(":", 1)[1]
    with pytest.raises(ValueError, match="H1 的 primary 与 G3 交集非空"):
        sb.run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)


def test_g5_rejects_mcq_overlap_between_primary_and_gate(tmp_path):
    args = _full_tree(tmp_path)
    pools = sb.load_h1_pools(args["ontology_dir"], args["stage0_samples"])
    mcq_pools = sb.load_mcq_pools(args["cyber_dir"], _cfg())
    h1, mcq, _excluded = sb.build_sample_sets(pools, mcq_pools, _cfg())
    mcq["G3"][0]["question_hash"] = mcq["primary"][0]["question_hash"]
    with pytest.raises(ValueError, match="MCQ 的 primary 与 G3 交集非空"):
        sb.run_g5(h1, mcq, pools=pools, mcq_pools=mcq_pools)


# ── 诊断模式 ─────────────────────────────────────────────────────────────────
def test_diagnose_reports_cross_stage_collisions_without_raising(tmp_path):
    """把一个 fresh 条目做成与已烧条目同 name+description → 诊断须报告而**不抛错**。"""
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack",
                    [("T1000", "shared", "shared desc"), ("T2000", "shared", "shared desc")])
    _write_ontology(ontology, "cwe", [("CWE-1", "n", "d")])
    stage0 = tmp_path / "s.jsonl"
    stage0.write_text(json.dumps(_stage0_row("attack", "T1000", "shared", "shared desc")) + "\n",
                      encoding="utf-8")
    report = sb.diagnose_g5_conflicts(sb.load_h1_pools(ontology, stage0))
    assert report["blocking"] is True
    assert report["cross_stage_collisions"]["count"] == 1
    assert report["cross_stage_collisions"]["detail"][0]["fresh_entry"] == "attack:T2000"
    assert report["intra_stage_duplicate_prompts"]["count"] == 0


def test_diagnose_reports_intra_stage_duplicates_as_non_blocking(tmp_path):
    ontology = tmp_path / "ontology"
    _write_ontology(ontology, "attack",
                    [("T1000", "same", "same d"), ("T2000", "same", "same d")])
    _write_ontology(ontology, "cwe", [("CWE-1", "n", "d")])
    stage0 = tmp_path / "s.jsonl"
    stage0.write_text("", encoding="utf-8")
    report = sb.diagnose_g5_conflicts(sb.load_h1_pools(ontology, stage0))
    assert report["blocking"] is False
    assert report["intra_stage_duplicate_prompts"]["count"] == 1
    assert report["n_fresh_entries"] == 3 and report["n_distinct_prompts"] == 2
