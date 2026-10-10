"""analyzer / readiness 接线 sprint 的单测。合成 fixture，**不读真实 label**。

锁定：
  * 三条分析路径**冻结**，禁止第四种变体
  * path runner 在 canonical 块集上与冻结 `v3_stacking.run_repeat` **逐元素相同**
  * path C 去掉 B4，其余块不变
  * shadow overlay 硬门逐项 fail-closed；overlay 只在内存构造，不改原 records
  * E3 reader 先校验 completion report
  * readiness builder 确定性；入口**不信任磁盘布尔值**
  * RQ2 / D 的状态与标记
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_adapter as adapter  # noqa: E402
from recover_attention.evaluation import v3_analysis_paths as vpaths  # noqa: E402
from recover_attention.evaluation import v3_readiness as vrd  # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow_f5  # noqa: E402
from recover_attention.evaluation import v3_shadow_overlay as vov  # noqa: E402
from recover_attention.evaluation import v3_stacking as vs  # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fx  # noqa: E402


# --------------------------------------------------------------- 路径冻结
def test_exactly_three_frozen_paths_in_fixed_order():
    assert vpaths.FROZEN_PATHS == (vpaths.PATH_A, vpaths.PATH_B, vpaths.PATH_C)
    assert len(vpaths.FROZEN_PATHS) == 3
    vpaths.assert_frozen_paths(vpaths.FROZEN_PATHS)


def test_fourth_variant_is_refused():
    with pytest.raises(vpaths.PathViolation):
        vpaths.assert_frozen_paths(vpaths.FROZEN_PATHS + ("convenient_extra",))
    with pytest.raises(vpaths.PathViolation):
        vpaths.assert_frozen_paths((vpaths.PATH_B, vpaths.PATH_A, vpaths.PATH_C))
    with pytest.raises(vpaths.PathViolation):
        vpaths.blocks_for_path([], "some_new_path")


def test_each_path_carries_a_mandatory_marker_and_limit():
    for p in vpaths.FROZEN_PATHS:
        m = vpaths.path_markers(p)
        assert m["marker"] and m["interpretation_limit"]
    assert vpaths.path_markers(vpaths.PATH_A)["marker"] == \
        "h1_b4_numeric_integrity_compromised"
    assert "不得" in vpaths.path_markers(vpaths.PATH_A)["interpretation_limit"]
    assert "不得" in vpaths.path_markers(vpaths.PATH_C)["interpretation_limit"]


# ------------------------------------------------- path runner 等价性锁死
@pytest.fixture(scope="module")
def small_arm():
    cfg = fx.synthetic_cfg()
    recs, hid = fx.make_h1(n_prompts=12, seed=0)
    return cfg, adapter.build_arm(recs, hid)


def test_path_runner_matches_frozen_run_repeat_elementwise(small_arm):
    """给 canonical 块集时必须与冻结实现**逐元素相同** —— 防止实现分叉。"""
    cfg, b = small_arm
    a = vs.run_repeat(y=b["y"], groups=b["groups"], blocks=b["blocks"], repeat=0, cfg=cfg)
    c = vpaths.run_repeat_for_path(y=b["y"], groups=b["groups"], blocks=b["blocks"],
                                   repeat=0, cfg=cfg, path=vpaths.PATH_A)
    assert sorted(a["oof"]) == sorted(c["oof"])
    for k in a["oof"]:
        assert np.array_equal(a["oof"][k], c["oof"][k], equal_nan=True), k


def test_path_b_uses_identical_block_structure_to_path_a(small_arm):
    """A 与 B 的**块结构相同** —— 差别只在 B4 的数值。"""
    _cfg, b = small_arm
    assert ([x.name for x in vpaths.blocks_for_path(b["blocks"], vpaths.PATH_A)]
            == [x.name for x in vpaths.blocks_for_path(b["blocks"], vpaths.PATH_B)])


def test_path_c_removes_b4_and_keeps_everything_else(small_arm):
    cfg, b = small_arm
    names = [x.name for x in vpaths.blocks_for_path(b["blocks"], vpaths.PATH_C)]
    assert "B4" not in names
    assert names == ["B1", "B2", "B3", "B5", "H"]
    s, sh = vpaths.names_for_path(vpaths.PATH_C)
    assert s == ["B1", "B2", "B3", "B5"] and sh == ["B1", "B2", "B3", "B5", "H"]
    out = vpaths.run_repeat_for_path(y=b["y"], groups=b["groups"], blocks=b["blocks"],
                                     repeat=0, cfg=cfg, path=vpaths.PATH_C)
    assert "B4" not in out["oof"]
    assert set(out["oof"]) == {"B1", "B2", "B3", "B5", "H", "S*", "S*+H"}


def test_path_runner_refuses_missing_block(small_arm):
    cfg, b = small_arm
    stripped = [x for x in b["blocks"] if x.name != "B2"]
    with pytest.raises(vpaths.PathViolation, match="缺块"):
        vpaths.run_repeat_for_path(y=b["y"], groups=b["groups"], blocks=stripped,
                                   repeat=0, cfg=cfg, path=vpaths.PATH_A)


# --------------------------------------------------------- overlay 硬门
def _shadow_dir(tmp_path, *, complete=True, drop=0, break_field=None,
                dup=False, bad_schema=False):
    prod = tmp_path / "production"
    sh = tmp_path / "shadow"
    prod.mkdir(exist_ok=True)
    sh.mkdir(exist_ok=True)
    h1 = [{"example_id": f"e{i}", "sample_type": "greedy", "sample_index": 0,
           "trace_id": f"e{i}__greedy_0", "eligible": True,
           "id_token_positions_local": [0], "f5_id_token_count": 1,
           "f5_self_consistency_exact": 0.0, "f5_id_agreement_rate": 0.0,
           "f5_confidence_high": 0, "f5_confidence_medium": 0, "f5_confidence_low": 0}
          for i in range(5)]
    mcq = [{"example_id": f"m{i}", "eligible_for_primary": True,
            "f5_self_consistency_exact": 0.0, "f5_letter_agreement_rate": 0.0}
           for i in range(4)]
    (prod / "h1_completion_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in h1), encoding="utf-8")
    (prod / "mcq_completion_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in mcq), encoding="utf-8")

    def rows(recs, arm):
        names = (shadow_f5.H1_STABLE_FIELDS if arm == "h1"
                 else shadow_f5.MCQ_STABLE_FIELDS)
        out = []
        for r in recs:
            stable = {n: (1 if n in ("f5_completion_token_count",
                                     "f5_first_id_token_rank") else 0.5) for n in names}
            if break_field and arm == "h1" and r is recs[0]:
                stable[break_field] = float("nan")
            key = "trace_id" if arm == "h1" else "example_id"
            out.append({"schema": shadow_f5.SHADOW_SCHEMA if not bad_schema else "wrong",
                        "arm": arm, key: r[key],
                        "interpretation": shadow_f5.SENSITIVITY_LABEL,
                        "deviation_class": shadow_f5.DEVIATION_LABEL,
                        "stable_fields": stable,
                        "frozen_passthrough": shadow_f5.frozen_passthrough(r, arm)})
        return out

    h1_rows = rows(h1, "h1")[: len(h1) - drop]
    if dup:
        h1_rows = h1_rows + h1_rows[:1]
    (sh / "h1_shadow_f5.jsonl").write_text(
        "\n".join(json.dumps(r) for r in h1_rows), encoding="utf-8")
    (sh / "mcq_shadow_f5.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows(mcq, "mcq")), encoding="utf-8")
    (sh / "shadow_completion_report.json").write_text(
        json.dumps({"complete": complete}), encoding="utf-8")
    (sh / "shadow_fingerprint.json").write_text(json.dumps({"source_shas": {}}),
                                                encoding="utf-8")
    return sh, prod


def test_overlay_gate_accepts_complete_and_matching(tmp_path):
    sh, prod = _shadow_dir(tmp_path)
    ov = vov.load_shadow_overlay(sh, prod)
    assert ov["gate_passed"] is True
    assert ov["counts"] == {"h1": 5, "mcq": 4}
    assert ov["interpretation"] == shadow_f5.SENSITIVITY_LABEL
    assert "不是" in ov["not_truth"]


def test_overlay_gate_refuses_incomplete_report(tmp_path):
    sh, prod = _shadow_dir(tmp_path, complete=False)
    with pytest.raises(vov.OverlayRefused, match="complete != true"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_missing_report(tmp_path):
    sh, prod = _shadow_dir(tmp_path)
    (sh / "shadow_completion_report.json").unlink()
    with pytest.raises(vov.OverlayRefused, match="必须拒绝读取"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_id_set_mismatch(tmp_path):
    sh, prod = _shadow_dir(tmp_path, drop=1)
    with pytest.raises(vov.OverlayRefused, match="集合不等于 analyzable"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_duplicates(tmp_path):
    sh, prod = _shadow_dir(tmp_path, dup=True)
    with pytest.raises(vov.OverlayRefused, match="重复"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_nonfinite_stable_field(tmp_path):
    sh, prod = _shadow_dir(tmp_path, break_field="f5_lengthnorm_logprob")
    with pytest.raises(vov.OverlayRefused, match="非有限"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_wrong_schema(tmp_path):
    sh, prod = _shadow_dir(tmp_path, bad_schema=True)
    with pytest.raises(vov.OverlayRefused, match="schema 不符"):
        vov.load_shadow_overlay(sh, prod)


def test_overlay_gate_refuses_source_sha_drift(tmp_path):
    sh, prod = _shadow_dir(tmp_path)
    (sh / "shadow_fingerprint.json").write_text(
        json.dumps({"source_shas": {"h1_completion_records.jsonl": "0" * 64}}),
        encoding="utf-8")
    with pytest.raises(vov.OverlayRefused, match="SHA 漂移"):
        vov.load_shadow_overlay(sh, prod)


def test_apply_overlay_is_in_memory_and_does_not_mutate_records(tmp_path):
    sh, prod = _shadow_dir(tmp_path)
    ov = vov.load_shadow_overlay(sh, prod)
    recs = [json.loads(x) for x in
            (prod / "h1_completion_records.jsonl").read_bytes().split(b"\n") if x.strip()]
    before = json.dumps(recs, sort_keys=True)
    new = vov.apply_overlay(recs, ov, "h1")
    assert json.dumps(recs, sort_keys=True) == before        # 原 records 未被修改
    assert new is not recs and new[0] is not recs[0]
    assert new[0]["f5_lengthnorm_logprob"] == 0.5            # 已被 overlay 覆盖
    assert new[0]["f5_id_token_count"] == recs[0]["f5_id_token_count"]   # passthrough 不变


def test_apply_overlay_refuses_unknown_record(tmp_path):
    sh, prod = _shadow_dir(tmp_path)
    ov = vov.load_shadow_overlay(sh, prod)
    with pytest.raises(vov.OverlayRefused, match="无对应 shadow 记录"):
        vov.apply_overlay([{"trace_id": "ghost", "example_id": "g",
                            "sample_type": "greedy", "sample_index": 0}], ov, "h1")


def test_e3_reader_requires_completion_report(tmp_path):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    with pytest.raises(vov.OverlayRefused, match="必须拒绝读取"):
        vov.load_e3_blocks(tmp_path, [], V3AnalysisConfig())


# --------------------------------------------------------------- readiness
def _readiness_kw(tmp_path):
    for sub in ("production", "readiness", "shadow", "repool", "confirmatory"):
        (tmp_path / sub).mkdir(exist_ok=True)
    return {"prod_dir": tmp_path / "production", "readiness_dir": tmp_path / "readiness",
            "shadow_dir": tmp_path / "shadow", "repool_dir": tmp_path / "repool",
            "result_dir": tmp_path / "confirmatory",
            "unblind_log": tmp_path / "unblinding.jsonl",
            "freeze_manifest_path": tmp_path / "readiness" / "missing.json"}


def test_readiness_is_deterministic(tmp_path):
    kw = _readiness_kw(tmp_path)
    a, b = vrd.recompute(**kw), vrd.recompute(**kw)
    assert vrd._comparable(a) == vrd._comparable(b)


def test_readiness_blocks_when_artifacts_missing(tmp_path):
    doc = vrd.recompute(**_readiness_kw(tmp_path))
    assert doc["readiness"]["ready_for_unblinding"] is False
    assert "finalized_freeze_manifest" in doc["readiness"]["blocking_reasons"]
    assert "shadow_overlay_gate" in doc["readiness"]["blocking_reasons"]
    assert "e3_repool_report" in doc["readiness"]["blocking_reasons"]


def test_readiness_always_reports_paths_and_callback_wired(tmp_path):
    doc = vrd.recompute(**_readiness_kw(tmp_path))
    by = {c["name"]: c for c in doc["checks"]}
    assert by["analysis_paths_frozen"]["ok"] is True
    assert by["real_analyze_callback_wired"]["ok"] is True
    assert by["m4_population_declaration_writer"]["ok"] is True
    assert by["estimands_wired"]["ok"] is True


def test_recompute_and_compare_refuses_tampered_disk_boolean(tmp_path):
    kw = _readiness_kw(tmp_path)
    doc = vrd.recompute(**kw)
    p = tmp_path / "readiness.json"
    tampered = json.loads(json.dumps(doc))
    tampered["readiness"]["ready_for_unblinding"] = True
    tampered["readiness"]["blocking_reasons"] = []
    p.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(vrd.ReadinessMismatch, match="拒绝信任磁盘布尔值"):
        vrd.recompute_and_compare(p, **kw)


def test_recompute_and_compare_accepts_faithful_disk_doc(tmp_path):
    kw = _readiness_kw(tmp_path)
    doc = vrd.recompute(**kw)
    p = tmp_path / "readiness.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert vrd.recompute_and_compare(p, **kw)["readiness"] == doc["readiness"]


def test_recompute_and_compare_refuses_missing_disk_doc(tmp_path):
    kw = _readiness_kw(tmp_path)
    with pytest.raises(vrd.ReadinessMismatch, match="不存在"):
        vrd.recompute_and_compare(tmp_path / "nope.json", **kw)


# ---------------------------------------------------------- RQ2 / D 语义
def test_rq2_frozen_status_is_pinned():
    from recover_attention.evaluation import v3_analyze as va
    assert va.RQ2_FROZEN_STATUS == "not_interpretable_due_to_invalid_frozen_stratifier"


def test_analyze_callback_refuses_without_gate():
    from recover_attention.evaluation import v3_analyze as va
    with pytest.raises(vov.OverlayRefused):
        va.make_analyze_callback(cfg=fx.synthetic_cfg(), h1_records=[], h1_hidden={},
                                 mcq_records=[], mcq_hidden={},
                                 shadow_overlay={"gate_passed": False}, e3_blocks={})
    ok_overlay = {"gate_passed": True,
                  "overlay_source": vov.OVERLAY_SOURCE_TAIL}
    with pytest.raises(vov.OverlayRefused, match="E3 blocks"):
        va.make_analyze_callback(cfg=fx.synthetic_cfg(), h1_records=[], h1_hidden={},
                                 mcq_records=[], mcq_hidden={},
                                 shadow_overlay=ok_overlay, e3_blocks=None)
    # 旧 full-BF16 shadow **不得**接 label
    with pytest.raises(vov.OverlayRefused, match="不在被接受集合"):
        va.make_analyze_callback(cfg=fx.synthetic_cfg(), h1_records=[], h1_hidden={},
                                 mcq_records=[], mcq_hidden={},
                                 shadow_overlay={"gate_passed": True,
                                                 "overlay_source":
                                                     vov.OVERLAY_SOURCE_LEGACY_SHADOW},
                                 e3_blocks={"canonical_source": "repool"})
    # E3 必须是 canonical repool
    with pytest.raises(vov.OverlayRefused, match="canonical repool"):
        va.make_analyze_callback(cfg=fx.synthetic_cfg(), h1_records=[], h1_hidden={},
                                 mcq_records=[], mcq_hidden={},
                                 shadow_overlay=ok_overlay,
                                 e3_blocks={"canonical_source": "tail_stabilized"})
