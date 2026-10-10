"""Stage1-L1：Llama 跨模型复现的离线适配测试。

**合成数据 + 假后端 + 轻量 tensor；不加载模型、不跑 GPU、不碰真实 outputs/。**

覆盖 task card §10 的十二条：

    1  Qwen 默认几何与行为兼容（既有常量与默认值一个都没漂）
    2  Llama 几何贯通；错误层号 / 维度 / rel_depth 明确失败
    3  replication 角色从 CLI → finalized 记录 → report → 分析结果全程一致
    4  finalized 输入与输出计数不一致时失败
    5  独立 artifact root 不回退到 Qwen 路径
    6  --repool-only 不进入 full-BF16 shadow 分支
    7  resume 遇到 fingerprint / 输入变化时失败
    8  readiness 拒绝缺失或漂移的产物
    9  已有分析释放记录与结果不得被静默覆盖
    10 数值状态未知时不得标记为通过
    11 复现角色下 RQ2 状态正确
    12 纯进度日志不含性能统计量、且不改变结果

关键拒绝路径一律用**故障注入**验证：先证明正常路径通过，再注入缺陷证明它被抓住。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_analysis_paths as vpaths  # noqa: E402
from recover_attention.evaluation import v3_analyze as vanalyze  # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3  # noqa: E402
from recover_attention.evaluation import v3_g4_quality as g4  # noqa: E402
from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation import v3_readiness as vreadiness  # noqa: E402
from recover_attention.evaluation import v3_tail_stabilized as tail  # noqa: E402

_REAL_OUTPUTS = _REPO_ROOT / "outputs"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_l1_{name}", _REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def entry():
    return _load_script("run_stage1_v3_replication")


@pytest.fixture(scope="module")
def shadow_repool():
    return _load_script("run_stage1_shadow_and_repool")


@pytest.fixture(scope="module")
def seg_driver():
    return _load_script("run_stage1_tail_stabilized_segments")


# ═════════════════════════════════════════════════════════ 真实产物守卫
def _tree_state(root: Path) -> dict:
    """(文件数, 总字节, 最新 mtime)。

    刻意**不**逐文件算 SHA：`outputs/stage1/{tail_stabilized,repool}/` 下有上万个
    NPZ，逐文件哈希会把本文件从秒级拖到十几分钟。既有的
    `test_v3_tail_stabilized._heavy_snapshot` 用的就是这个口径 —— 任何写入都会
    改变三者之一，同样抓得到，而代价是 O(stat) 而非 O(read)。
    """
    if not root.exists():
        return {"exists": False}
    n = total = 0
    newest = 0
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            n += 1
            total += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return {"exists": True, "n_files": n, "total_bytes": total, "newest_mtime_ns": newest}


@pytest.fixture(scope="module", autouse=True)
def _real_outputs_untouched():
    """模块级快照守卫。

    这不是形式主义：本仓库出过一次真实事故 —— `run_controls("definitely-not-a-model")`
    在 backend 构造失败**之前**就按真实 smoke 目录的现场编号建好了遥测文件，
    留下 5 份空产物。快照比对是当时抓住它的机制，这里沿用同一套。
    """
    watched = [_REAL_OUTPUTS / d for d in ("stage1", "stage1_llama_v3", "stage0")]
    before = {str(d): _tree_state(d) for d in watched}
    yield
    after = {str(d): _tree_state(d) for d in watched}
    changed = [k for k in after if after[k] != before[k]]
    assert not changed, f"真实 outputs/ 被本测试文件改动了：{changed}"


@pytest.mark.skipif(
    (_REAL_OUTPUTS / 'stage1_llama_v3/analysis/llama_ac_analysis.json').exists(),
    reason='Historical G3 workspace check; the release contains completed Llama analysis')
def test_no_stage_beyond_g3_has_produced_artifacts():
    """断言的**意图**是"后续阶段尚未开跑"，不是"目录不存在"。

    L1 时该意图等价于 `outputs/stage1_llama_v3/` 不存在；L2 已获授权跑真实 G3
    并合法创建了 `g3/`，故原断言过期。改为逐项检查**后续阶段**的产物一律缺席 ——
    这比原断言更强：它在目录存在之后仍然有效。
    """
    root = _REAL_OUTPUTS / "stage1_llama_v3"
    if not root.exists():
        return                       # L1 语义仍成立
    # G3 之后的每个阶段各自有专属目录 / 产物，一个都不得出现
    for later in ("g4", "production", "tail_stabilized", "repool", "readiness",
                  "analysis", "analysis_release", "confirmatory", "unblinding"):
        assert not (root / later).exists(), f"后续阶段产物已出现：{later}"
    # L2A 的未来运行根同样不得被本轮创建
    # G3 目录里不得出现两臂汇总（MCQ 未通过，按卡不生成）
    assert not (root / "g3" / "g3_summary.json").exists()


# ═════════════════════════════════════════ §10.1 Qwen 默认几何与行为兼容
def test_qwen_frozen_constants_did_not_drift():
    assert (tail.CAST_BLOCK, tail.FROZEN_TUPLE_IDX, tail.HIDDEN_DIM) == (19, 20, 3584)
    assert repool.HIDDEN_DIM == 3584
    q = repool.QWEN_GEOMETRY
    assert (q.num_layers, q.block_index, q.tuple_index, q.hidden_dim) == (28, 19, 20, 3584)
    assert tail.DEFAULT_GEOMETRY == q


def test_gate_criteria_default_to_qwen_geometry():
    for crit in (g3.G3Criteria(), g4.G4Criteria()):
        assert (crit.hidden_dim, crit.block_index, crit.tuple_index) == (3584, 19, 20)
    assert g3.CRITERIA.block_index == 19 and g4.CRITERIA.tuple_index == 20


def test_finalize_arm_and_analyze_default_to_confirmatory():
    """默认角色不变 —— 这是 Qwen 既有产物可复算的前提。"""
    import inspect
    assert (inspect.signature(prod.finalize_arm)
            .parameters["population_role"].default == "confirmatory")
    sig = inspect.signature(vanalyze.make_analyze_callback)
    assert sig.parameters["population_role"].default == vanalyze.ROLE_CONFIRMATORY
    assert sig.parameters["result_schema"].default is None
    assert sig.parameters["numeric_integrity_status"].default is None


def test_path_markers_default_text_is_byte_identical_to_frozen_qwen_wording():
    """省略 numeric_status 时 Path A 的标记与解释文字必须一字不变。"""
    m = vpaths.path_markers(vpaths.PATH_A)
    assert m["marker"] == "h1_b4_numeric_integrity_compromised"
    assert m["interpretation_limit"] == (
        vpaths.PATH_SEMANTICS[vpaths.PATH_A]["interpretation_limit"])
    for p in (vpaths.PATH_B, vpaths.PATH_C):
        sem = vpaths.PATH_SEMANTICS[p]
        got = vpaths.path_markers(p)
        assert got["marker"] == sem["mandatory_marker"]
        assert got["interpretation_limit"] == sem["interpretation_limit"]


def test_geometry_cli_defaults_to_qwen_when_all_flags_omitted():
    import argparse
    ap = repool.add_geometry_cli_args(argparse.ArgumentParser())
    ap.add_argument("--model-path", default="x")
    assert repool.geometry_from_cli(ap.parse_args([])) is repool.QWEN_GEOMETRY


# ═════════════════════════════════════════ §10.2 Llama 几何贯通 / 错误即失败
def test_llama_geometry_from_real_config_matches_expectation():
    path = Path("D:/models/Meta-Llama-3.1-8B-Instruct")
    if not (path / "config.json").exists():
        pytest.skip("本机无 Llama 权重目录")
    geo = repool.ModelGeometry.from_model_config(path)
    assert (geo.num_layers, geo.block_index, geo.tuple_index, geo.hidden_dim) \
        == (32, 22, 23, 4096)
    assert geo.as_dict()["static_geometry_verified_from_config"] is True
    # 静态几何**不**代表运行时已验证 —— 这条断言防的是把两者混为一谈。
    assert geo.as_dict()["runtime_hidden_index_verified"] is False


@pytest.mark.parametrize("kwargs,why", [
    (dict(num_layers=32, block_index=22, tuple_index=24, hidden_dim=4096),
     "tuple != block+1"),
    (dict(num_layers=32, block_index=25, tuple_index=26, hidden_dim=4096),
     "block != floor(0.7L)"),
    (dict(num_layers=32, block_index=32, tuple_index=33, hidden_dim=4096),
     "block 越界"),
    (dict(num_layers=0, block_index=0, tuple_index=1, hidden_dim=4096),
     "层数非正"),
    (dict(num_layers=32, block_index=22, tuple_index=23, hidden_dim=0),
     "维度非正"),
    (dict(num_layers=32, block_index=22, tuple_index=23, hidden_dim=True),
     "bool 冒充 int"),
    (dict(num_layers=32, block_index=22, tuple_index=23, hidden_dim=4096,
          rel_depth=0.5), "rel_depth 非冻结值"),
])
def test_bad_geometry_is_rejected_by_name(kwargs, why):
    """故障注入：每一类错误都必须**具名**失败，不能静默接受。"""
    with pytest.raises(repool.GeometryViolation):
        repool.ModelGeometry(**kwargs)


def test_geometry_cross_check_against_config_rejects_wrong_model():
    path = Path("D:/models/Meta-Llama-3.1-8B-Instruct")
    if not (path / "config.json").exists():
        pytest.skip("本机无 Llama 权重目录")
    # 拿 Qwen 的几何去核 Llama 的 config —— 必须停。
    with pytest.raises(repool.GeometryViolation):
        repool.QWEN_GEOMETRY.assert_matches_model_config(path)


def test_geometry_cli_conflict_and_half_specification_are_rejected():
    import argparse
    ap = repool.add_geometry_cli_args(argparse.ArgumentParser())
    ap.add_argument("--model-path", default="D:/models/Meta-Llama-3.1-8B-Instruct")
    # 只给一半 → 另一半会悄悄用 Qwen 默认，正是跨模型最容易出的错
    with pytest.raises(repool.GeometryViolation):
        repool.geometry_from_cli(ap.parse_args(["--num-layers", "32"]))
    if Path("D:/models/Meta-Llama-3.1-8B-Instruct/config.json").exists():
        with pytest.raises(repool.GeometryViolation):
            repool.geometry_from_cli(ap.parse_args(
                ["--geometry-from-model-config", "--hidden-dim", "3584"]))


def test_tail_hook_refuses_a_model_whose_depth_disagrees_with_geometry():
    """hook 层号取错不会自己报错，只会安静地在错误深度取 hidden —— 必须先拦。"""
    import torch

    class _Layers(list):
        pass

    class _Base:
        def __init__(self, n):
            self.layers = _Layers(torch.nn.Identity() for _ in range(n))

    class _M:
        def __init__(self, n):
            self.model = _Base(n)

    llama_geo = repool.ModelGeometry.from_num_layers(32, 4096)
    with pytest.raises(tail.TailStop) as exc:
        tail.TailHook(_M(28), torch, geometry=llama_geo)      # 28 层的模型 + 32 层几何
    assert exc.value.gate == "geometry_num_layers"
    # 层数相符时正常挂上
    hook = tail.TailHook(_M(32), torch, geometry=llama_geo)
    assert hook.geometry.block_index == 22
    hook.close()


def test_e3_shape_gate_follows_the_run_geometry_not_a_constant():
    llama_geo = repool.ModelGeometry.from_num_layers(32, 4096)
    arr_3584 = np.zeros(3584, dtype=np.float32)
    diag = repool.anchor_exact_match(arr_3584, arr_3584)
    assert diag["exact"] is True
    # 4096 维几何下，3584 的向量必须被拒
    assert llama_geo.hidden_dim == 4096 and repool.QWEN_GEOMETRY.hidden_dim == 3584


# ═════════════════════════════════════════ §10.3 / §10.11 角色与 RQ2 一致性
def _run_replication_probe(progress=None):
    """用复现角色跑一遍**真实的**合成 end-to-end（与 --execute-all 同一路径）。

    ⚠️ 一次约 145 s（production 配置下 B_total=20,000 的 bootstrap 与 stacking 拟合
    占绝大部分，与 n_prompts 关系不大 —— 实测把 n_prompts 30→14 只降到 93 s，
    且会让 E2 从 estimable 变 non_estimable，改变探针语义，故不缩样本）。
    因此本文件把重跑压到**两次**：一次 module fixture、一次开日志的对照。

    刻意**不传** `numeric_integrity_status` —— 这样这一次运行同时验证了
    「省略即 not_audited」这条 fail-closed 默认，不必再多跑一次。
    """
    import warnings
    probe = vanalyze.synthetic_contract_probe()
    pcfg = probe.pop("cfg")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                module=r"sklearn\.model_selection\..*")
        result = vanalyze.make_analyze_callback(
            cfg=pcfg, population_role=vanalyze.ROLE_CROSS_MODEL_REPLICATION,
            result_schema=vanalyze.REPLICATION_ANALYSIS_SCHEMA,
            progress=progress, **probe)()
    return result, pcfg


@pytest.fixture(scope="module")
def replication_result():
    return _run_replication_probe()[0]


@pytest.fixture(scope="module")
def replication_result_with_progress():
    events = []
    result, pcfg = _run_replication_probe(progress=events.append)
    return result, events, pcfg


def test_replication_result_declares_its_role_and_schema(replication_result):
    r = replication_result
    assert r["schema"] == "stage1_v3_cross_model_replication_analysis_v1"
    assert r["population_role"] == "cross_model_replication_non_confirmatory"
    assert r["is_confirmatory"] is False
    assert r["contract"]["ok"] is True


def test_replication_rq2_is_out_of_scope_not_a_finding(replication_result):
    rq2 = replication_result["rq2"]
    assert rq2["status"] == "not_in_stage1_v3_cross_model_replication_scope"
    assert rq2["computed"] is False
    # 不得混入 confirmatory 那套 RQ2 结构（否则会被读成"算过了"）
    assert "frozen_primary" not in rq2 and "stable_shadow" not in rq2


def test_replication_keeps_the_three_frozen_paths_and_statistical_definitions(
        replication_result):
    from recover_attention.evaluation import v3_grouping as grouping
    assert replication_result["frozen_paths"] == list(vpaths.FROZEN_PATHS)
    for path in vpaths.FROZEN_PATHS:
        est = replication_result["estimands"][path]
        assert set(est) == {"E1", "E2", "E3", "M2c"}
        # MCQ 的 E2 / E3 / M2c 按协议为 not applicable，且不得算任何性能量
        for name in ("E2", "E3", "M2c"):
            mcq = est[name]["mcq"]
            assert mcq["estimability"] == grouping.NOT_APPLICABLE
            assert mcq["performance_estimates_computed"] is False
        assert est["E1"]["role"] == "PRIMARY"


def test_replication_schema_cannot_masquerade_as_confirmatory():
    probe = vanalyze.synthetic_contract_probe()
    probe.pop("cfg")
    with pytest.raises(ValueError, match="不得使用 confirmatory 的 schema"):
        vanalyze.make_analyze_callback(
            cfg=vanalyze.V3AnalysisConfig(mode="dry_run"),
            population_role=vanalyze.ROLE_CROSS_MODEL_REPLICATION,
            result_schema=vanalyze.ANALYSIS_SCHEMA, **probe)


def test_unknown_population_role_is_rejected_everywhere():
    with pytest.raises(ValueError):
        vanalyze.make_analyze_callback(
            cfg=None, h1_records=[], h1_hidden={}, mcq_records=[], mcq_hidden={},
            shadow_overlay=None, e3_blocks=None, population_role="whatever")
    assert "whatever" not in prod.ALLOWED_POPULATION_ROLES


# ═════════════════════════════════════════ §10.10 数值状态 fail-closed
def test_missing_numeric_audit_is_never_treated_as_passing():
    """缺失审计 → not_audited，**绝不**落到 verified_clean。"""
    assert vpaths.assert_numeric_status(None) == vpaths.NUMERIC_NOT_AUDITED
    with pytest.raises(vpaths.PathViolation):
        vpaths.assert_numeric_status("passed")            # 不是白名单里的三态之一
    with pytest.raises(vpaths.PathViolation):
        vpaths.assert_numeric_status("clean")


def test_replication_defaults_to_not_audited_when_status_omitted(replication_result):
    """fixture 刻意未传状态 —— 结果必须落到 not_audited，且标记随之如实写出。"""
    r = replication_result
    assert r["numeric_integrity"]["status"] == vpaths.NUMERIC_NOT_AUDITED
    assert r["path_markers"][vpaths.PATH_A]["marker"] == \
        "h1_b4_numeric_integrity_not_audited"
    assert r["contrast_D"][vpaths.PATH_A]["marker"] == \
        "asymmetric_numeric_handling_not_audited"
    assert "fail_closed" in r["numeric_integrity"]["fail_closed_note"] or \
        "not_audited" in r["numeric_integrity"]["fail_closed_note"]


@pytest.mark.parametrize("status,path_a,d_marker", [
    (vpaths.NUMERIC_COMPROMISED, "h1_b4_numeric_integrity_compromised",
     "asymmetric_numeric_handling_compromised"),
    (vpaths.NUMERIC_NOT_AUDITED, "h1_b4_numeric_integrity_not_audited",
     "asymmetric_numeric_handling_not_audited"),
    (vpaths.NUMERIC_VERIFIED_CLEAN, "h1_b4_numeric_integrity_verified_clean",
     "symmetric_numeric_handling_verified_clean"),
])
def test_path_a_and_d_markers_follow_this_runs_audit(status, path_a, d_marker):
    """Qwen 的 compromised 裁定**不自动继承**给另一个模型。"""
    assert vpaths.path_markers(vpaths.PATH_A, status)["marker"] == path_a
    assert vpaths.d_markers(status)["marker"] == d_marker
    # B / C 与冻结 B4 的数值完整性无关，标记不随审计状态变
    for p in (vpaths.PATH_B, vpaths.PATH_C):
        assert (vpaths.path_markers(p, status)["marker"]
                == vpaths.PATH_SEMANTICS[p]["mandatory_marker"])


# ═════════════════════════════════════════ §10.12 纯进度日志
def test_progress_events_carry_no_performance_quantities():
    seen = []
    vpaths.emit_progress(seen.append, "outer_fold_done", path="A", repeat=0,
                         outer_fold=1, elapsed_s=1.5, status="ok")
    assert seen and set(seen[0]) == {"event", "ts", "path", "repeat", "outer_fold",
                                     "elapsed_s", "status"}
    for banned in ("auroc", "delta_h", "ci_lo", "oof", "label", "score"):
        assert banned not in seen[0]


def test_progress_logger_refuses_to_carry_a_performance_field():
    """结构性护栏：就算有人往进度事件里塞 AUROC，也必须当场拒绝。"""
    with pytest.raises(vpaths.PathViolation, match="不得携带性能相关字段"):
        vpaths.emit_progress(lambda e: None, "x", auroc=0.9)
    with pytest.raises(vpaths.PathViolation):
        vpaths.emit_progress(lambda e: None, "x", delta_h=0.01)


def test_progress_logging_does_not_change_any_result(replication_result,
                                                     replication_result_with_progress):
    """开日志不得改变任何数值 —— 证明它没消耗随机数、没改变计算顺序。

    两个 fixture 的输入完全相同，只差一个 `progress=` 回调。
    """
    loud, events, _ = replication_result_with_progress
    assert events, "开了日志却一条事件都没有"
    assert {e["event"] for e in events} >= {"outer_fold_start", "outer_fold_done",
                                            "bootstrap_start", "bootstrap_done"}
    dump = lambda d: json.dumps(d, sort_keys=True, default=str)   # noqa: E731
    assert dump(replication_result["estimands"]) == dump(loud["estimands"])
    assert dump(replication_result["arms"]) == dump(loud["arms"])
    assert dump(replication_result["contrast_D"]) == dump(loud["contrast_D"])


def test_progress_events_are_label_free_and_cover_every_fold(
        replication_result_with_progress):
    """实跑出来的每一条事件都必须不含性能量，且覆盖到 fold 粒度。"""
    _, events, pcfg = replication_result_with_progress
    banned = {"auroc", "delta", "delta_h", "ci", "ci_lo", "ci_hi", "oof", "score",
              "label", "y", "pred", "increment", "width", "point"}
    for e in events:
        assert not (set(k.lower() for k in e) & banned), f"进度事件泄漏了性能量：{e}"
    folds = {(e.get("path"), e.get("arm"), e.get("repeat"), e.get("outer_fold"))
             for e in events if e["event"] == "outer_fold_done"}
    # 期望数由**探针自己的 cfg** 决定，不写死 R=3 —— 合成探针用的是缩小的 cfg。
    expected = len(vpaths.FROZEN_PATHS) * 2 * pcfg.n_repeats * pcfg.n_outer_folds
    assert len(folds) == expected, f"fold 事件数 {len(folds)} != 期望 {expected}"
    assert {f[3] for f in folds} == set(range(pcfg.n_outer_folds))
    assert {f[2] for f in folds} == set(range(pcfg.n_repeats))


# ═════════════════════════════════════════ §10.5 独立目录不回退 Qwen 路径
def test_replication_entry_layout_is_entirely_under_the_artifact_root(entry, tmp_path):
    lay = entry.layout(tmp_path / "llama_root")
    for key, value in lay.items():
        if isinstance(value, Path):
            assert str(tmp_path) in str(value), f"{key} 落在 artifact root 之外：{value}"
            assert "stage1/production" not in str(value).replace("\\", "/") or \
                str(tmp_path) in str(value)


def test_replication_entry_never_references_the_confirmatory_unblinding_log(entry):
    """源码层断言：一次性解盲已用尽，本入口不得触碰那条路径。"""
    src = entry.ENTRY_SOURCE.read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]        # 跳过模块 docstring（其中解释了为何不用）
    assert "record_stage1_unblinding" not in body
    assert "unblinding_log.jsonl" not in body
    assert entry.RELEASE_LOG_NAME == "analysis_release_log.jsonl"


def test_readiness_replication_entry_check_catches_an_unblinding_writer(tmp_path):
    """故障注入：入口若写解盲日志，readiness 必须变红。"""
    good = tmp_path / "good.py"
    good.write_text("print('no unblinding here')\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("ub.record_stage1_unblinding(log)\n", encoding="utf-8")

    def blockers(entry_path):
        doc = vreadiness.recompute_replication(
            prod_dir=tmp_path / "prod", tail_dir=tmp_path / "tail",
            repool_dir=tmp_path / "repool", result_dir=tmp_path / "res",
            release_log=tmp_path / "rel.jsonl",
            freeze_manifest_path=tmp_path / "fz.json",
            analysis_input_manifest_path=tmp_path / "ai.json",
            entry_source_path=entry_path)
        return doc["readiness"]["blocking_reasons"]

    name = "replication_entry_never_writes_unblinding_log"
    assert name not in blockers(good)
    assert name in blockers(bad)


# ═════════════════════════════════════════ §10.8 readiness 拒绝缺失/漂移
def test_readiness_replication_refuses_when_artifacts_are_missing(tmp_path):
    doc = vreadiness.recompute_replication(
        prod_dir=tmp_path / "prod", tail_dir=tmp_path / "tail",
        repool_dir=tmp_path / "repool", result_dir=tmp_path / "res",
        release_log=tmp_path / "rel.jsonl",
        freeze_manifest_path=tmp_path / "missing_fz.json",
        analysis_input_manifest_path=tmp_path / "missing_ai.json",
        entry_source_path=tmp_path / "nonexistent.py")
    assert doc["readiness"]["ready_for_analysis"] is False
    assert "finalized_freeze_manifest" in doc["readiness"]["blocking_reasons"]
    assert "analysis_input_freeze_manifest" in doc["readiness"]["blocking_reasons"]
    assert doc["schema_version"] == "stage1_v3_cross_model_replication_readiness_v1"
    assert doc["is_confirmatory"] is False
    # 判定键必须是 ready_for_analysis，不得混用 confirmatory 的 ready_for_unblinding
    assert "ready_for_unblinding" not in doc["readiness"]


def test_readiness_replication_drops_only_the_legacy_shadow_precondition():
    """角色切换只允许去掉那一条，且『旧 shadow 不得接 label』仍强制。"""
    assert set(vreadiness.REPLICATION_DROPPED_CHECKS) == {
        "legacy_shadow_superseded_and_not_joined"}
    from recover_attention.evaluation import v3_shadow_overlay as ov
    assert ov.OVERLAY_SOURCE_LEGACY_SHADOW not in ov.ACCEPTED_OVERLAY_SOURCES


def test_readiness_replication_does_not_weaken_the_v3_check_set(tmp_path):
    """复现 readiness 的检查集合必须覆盖 v3 的实质项，不能借角色少查。"""
    doc = vreadiness.recompute_replication(
        prod_dir=tmp_path, tail_dir=tmp_path, repool_dir=tmp_path,
        result_dir=tmp_path / "r", release_log=tmp_path / "l.jsonl",
        freeze_manifest_path=tmp_path / "a.json",
        analysis_input_manifest_path=tmp_path / "b.json",
        entry_source_path=tmp_path / "c.py")
    names = {c["name"] for c in doc["checks"]}
    for required in ("preregistration_frozen", "finalized_freeze_manifest",
                     "analysis_input_freeze_manifest", "path_b_tail_overlay_gate",
                     "e3_canonical_repool_report", "analysis_paths_frozen",
                     "replication_analyze_callback_wired",
                     "m4_population_declaration_writer",
                     "estimands_wired_end_to_end",
                     "no_replication_result_targets_exist"):
        assert required in names, f"复现 readiness 少了 {required}"


def test_readiness_replication_rejects_a_hand_edited_disk_boolean(tmp_path):
    """磁盘上的布尔值不可信：手改 ready 为 true 必须在比对处被抓。"""
    doc_path = tmp_path / "readiness.json"
    kw = dict(prod_dir=tmp_path, tail_dir=tmp_path, repool_dir=tmp_path,
              result_dir=tmp_path / "r", release_log=tmp_path / "l.jsonl",
              freeze_manifest_path=tmp_path / "a.json",
              analysis_input_manifest_path=tmp_path / "b.json",
              entry_source_path=tmp_path / "c.py")
    fresh = vreadiness.recompute_replication(**kw)
    fresh["readiness"]["ready_for_analysis"] = True
    fresh["readiness"]["blocking_reasons"] = []
    doc_path.write_text(json.dumps(fresh, ensure_ascii=False, default=str),
                        encoding="utf-8")
    with pytest.raises(vreadiness.ReadinessMismatch):
        vreadiness.recompute_and_compare_replication(doc_path, **kw)


# ═════════════════════════════════════════ §10.9 禁止静默覆盖
def test_release_log_is_create_only(entry, tmp_path):
    log = tmp_path / "analysis_release_log.jsonl"
    entry.record_analysis_release(log, payload={"timestamp": "t1"})
    first = log.read_bytes()
    with pytest.raises(entry.ReplicationRefused, match="禁止覆盖"):
        entry.record_analysis_release(log, payload={"timestamp": "t2"})
    assert log.read_bytes() == first, "被拒绝的第二次写入不得改动既有字节"
    row = json.loads(first.decode("utf-8"))
    assert row["role"] == "cross_model_replication_non_confirmatory"
    assert row["is_confirmatory"] is False


def test_result_writer_refuses_to_overwrite(entry, tmp_path):
    out = tmp_path / "analysis" / entry.RESULT_NAME
    entry.write_result_atomic(out, {"schema": "x", "n": 1})
    before = out.read_bytes()
    with pytest.raises(entry.ReplicationRefused, match="拒绝覆盖"):
        entry.write_result_atomic(out, {"schema": "x", "n": 2})
    assert out.read_bytes() == before
    # 半文件不得残留
    assert not list(out.parent.glob("*.tmp*"))


def test_result_writer_rejects_nonfinite_values(entry, tmp_path):
    """NaN/Infinity 不是合法 JSON —— 必须在落盘前抛错，不写半个文件。"""
    out = tmp_path / "r.json"
    with pytest.raises(ValueError):
        entry.write_result_atomic(out, {"x": float("nan")})
    assert not out.exists()


# ═════════════════════════════════════════ §10.4 计数不一致即失败
def _write_arm(prod_dir: Path, arm: str, *, n: int, role: str,
               report_n: int | None = None, dup: bool = False):
    prod_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        key = "0" if dup and i == 1 else str(i)
        base = {"example_id": f"e{key}", "population_role": role,
                "group_id": f"g{key}"}
        if arm == "h1":
            base.update({"trace_id": f"t{key}", "id_token_positions_local": [0]})
        else:
            base.update({"eligible_for_primary": True})
        rows.append(base)
    (prod_dir / f"{arm}_completion_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    (prod_dir / f"{arm}_finalization_report.json").write_text(json.dumps({
        "n_finalized_records": n if report_n is None else report_n,
        "n_analyzable_records": n if report_n is None else report_n,
        "population_role": role}), encoding="utf-8")


def test_observed_counts_come_from_finalized_inputs(entry, tmp_path):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    prod_dir = tmp_path / "production"
    _write_arm(prod_dir, "h1", n=4, role=entry.ROLE)
    _write_arm(prod_dir, "mcq", n=3, role=entry.ROLE)
    counts = entry.observed_counts({"prod": prod_dir}, V3AnalysisConfig())
    assert counts["h1"]["n_finalized_records"] == 4
    assert counts["mcq"]["n_finalized_records"] == 3
    assert counts["source"] == "finalized_completion_records_and_reports"


def test_counts_mismatch_between_records_and_report_fails(entry, tmp_path):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    prod_dir = tmp_path / "production"
    _write_arm(prod_dir, "h1", n=4, role=entry.ROLE, report_n=5)   # 注入不一致
    _write_arm(prod_dir, "mcq", n=3, role=entry.ROLE)
    with pytest.raises(entry.ReplicationRefused, match="不一致"):
        entry.observed_counts({"prod": prod_dir}, V3AnalysisConfig())


def test_duplicate_keys_in_finalized_records_fail(entry, tmp_path):
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    prod_dir = tmp_path / "production"
    _write_arm(prod_dir, "h1", n=4, role=entry.ROLE, dup=True)
    _write_arm(prod_dir, "mcq", n=3, role=entry.ROLE)
    with pytest.raises(entry.ReplicationRefused, match="重复键"):
        entry.observed_counts({"prod": prod_dir}, V3AnalysisConfig())


def test_confirmatory_records_cannot_be_analysed_as_a_replication(entry, tmp_path):
    """把 Qwen 的 confirmatory 产物喂给复现入口 → 必须停。"""
    from recover_attention.evaluation.v3_config import V3AnalysisConfig
    prod_dir = tmp_path / "production"
    _write_arm(prod_dir, "h1", n=4, role="confirmatory")
    _write_arm(prod_dir, "mcq", n=3, role="confirmatory")
    with pytest.raises(entry.ReplicationRefused, match="population_role"):
        entry.observed_counts({"prod": prod_dir}, V3AnalysisConfig())


def test_finalize_arm_rejects_an_unlisted_role(tmp_path):
    with pytest.raises(ValueError, match="不在白名单"):
        prod.finalize_arm("h1", tmp_path, [], cfg=None, run_fingerprint={},
                          population_role="looks_confirmatory_but_isnt")


# ═════════════════════════════════════════ §10.6 repool-only 不进 shadow
def test_repool_only_mode_exists_and_is_mutually_exclusive(shadow_repool):
    assert hasattr(shadow_repool, "run_repool_only")
    src = (_REPO_ROOT / "scripts" / "run_stage1_shadow_and_repool.py").read_text("utf-8")
    assert "--repool-only" in src


def _code_without_docstring(fn) -> str:
    """取函数**代码**，剥掉 docstring。

    必须剥：`run_repool_only` 的 docstring 恰恰**点名**了那些禁用符号来解释
    "它们不在这里"，直接扫全文会被自己的注释绊倒。要证明的是**代码**里没有调用。
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    body = node.body[1:] if (isinstance(node.body[0], ast.Expr)
                             and isinstance(node.body[0].value, ast.Constant)
                             and isinstance(node.body[0].value.value, str)) else node.body
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_repool_only_never_calls_the_full_bf16_shadow_path(shadow_repool):
    """结构性断言：不是靠开关关掉 shadow，而是那条路径**根本不在函数代码里**。"""
    code = _code_without_docstring(shadow_repool.run_repool_only)
    for banned in ("shadow_h1_unit", "shadow_mcq_unit", "load_route_b_backend",
                   "SHADOW_H1", "SHADOW_MCQ", "SHADOW_MANIFEST", "SHADOW_REPORT",
                   "SHADOW_FP"):
        assert banned not in code, f"repool-only 的代码里出现了 shadow 符号 {banned}"
    assert "REPOOL_MANIFEST" in code and "npz_atomic" in code


def test_repool_only_report_declares_it_wrote_no_shadow(shadow_repool):
    assert shadow_repool.REPOOL_ONLY_SCHEMA == \
        "stage1_e3_canonical_repool_completion_report_v1"
    code = _code_without_docstring(shadow_repool.run_repool_only)
    assert "'writes_full_bf16_shadow': False" in code or \
        '"writes_full_bf16_shadow": False' in code


# ═════════════════════════════════════════ §10.7 resume 遇变化即失败
def test_repool_only_refuses_to_resume_across_a_changed_input(shadow_repool, tmp_path,
                                                              monkeypatch):
    """既有 report 的 source_shas 与本次输入不符 → 拒绝续跑，不混两次运行。"""
    root = tmp_path / "llama_root"
    prod_dir, repool_dir = root / "production", root / "repool"
    prod_dir.mkdir(parents=True)
    repool_dir.mkdir(parents=True)
    (prod_dir / "h1_completion_records.jsonl").write_text("", encoding="utf-8")
    (prod_dir / "h1_run_fingerprint.json").write_text("{}", encoding="utf-8")
    (repool_dir / "e3_repool_report.json").write_text(
        json.dumps({"source_shas": {"h1_completion_records.jsonl": "stale",
                                    "h1_run_fingerprint.json": "stale"}}),
        encoding="utf-8")
    monkeypatch.setattr(shadow_repool, "PROD", prod_dir)
    monkeypatch.setattr(shadow_repool, "REPOOL", repool_dir)
    monkeypatch.setattr(shadow_repool, "REPOOL_REPORT",
                        repool_dir / "e3_repool_report.json")
    monkeypatch.setattr(shadow_repool, "REPOOL_DIR", repool_dir / "h1_e3_block_hidden")
    monkeypatch.setattr(shadow_repool, "REPOOL_MANIFEST",
                        repool_dir / "e3_repool_manifest.jsonl")
    with pytest.raises(shadow_repool.ScopeViolation, match="拒绝跨配置续跑"):
        shadow_repool.run_repool_only("<no-model>")


def test_set_artifact_root_refuses_a_second_call(shadow_repool):
    """跑到一半换根目录会让 append-only 前缀校验失去意义 —— 只允许一次。"""
    import importlib.util
    mod = _load_script("run_stage1_shadow_and_repool")     # 干净副本
    mod.set_artifact_root(Path("x"))
    with pytest.raises(mod.ScopeViolation, match="只能调用一次"):
        mod.set_artifact_root(Path("y"))


# ═════════════════════════════════════════ 分段计划：随实际计数与本机 benchmark
def test_segment_plan_is_rebuilt_from_counts_and_local_benchmark(seg_driver):
    """Qwen 的输入必须复现出 Qwen 那份冻结计划 —— 证明规则同源。"""
    qwen = seg_driver.build_segment_plan({"h1": 8664, "mcq": 1750},
                                         {"h1": 0.2119, "mcq": 0.2159})
    assert qwen == seg_driver.SEGMENT_PLAN
    # 换了计数与速率 → 计划必须跟着变，不能照搬 4000/4000/664/1750
    other = seg_driver.build_segment_plan({"h1": 6500, "mcq": 1700},
                                          {"h1": 0.25, "mcq": 0.26})
    assert other != seg_driver.SEGMENT_PLAN
    assert sum(n for a, n in other if a == "h1") == 6500
    assert sum(n for a, n in other if a == "mcq") == 1700


@pytest.mark.parametrize("counts,rates", [
    ({"h1": 0, "mcq": 10}, {"h1": 0.2, "mcq": 0.2}),
    ({"h1": 10, "mcq": 10}, {"h1": 0.0, "mcq": 0.2}),
    ({"h1": 10, "mcq": 10}, {"h1": -1.0, "mcq": 0.2}),
])
def test_segment_plan_rejects_degenerate_inputs(seg_driver, counts, rates):
    with pytest.raises(seg_driver.DriverStop):
        seg_driver.build_segment_plan(counts, rates)


def test_segment_plan_requires_a_local_benchmark_report(seg_driver, tmp_path):
    """没有本机 benchmark 就没有段长依据 —— 不得沿用另一模型的段长。"""
    class _Fake:
        BENCH_OUT = tmp_path / "absent.json"
    with pytest.raises(seg_driver.DriverStop, match="benchmark"):
        seg_driver.load_benchmark_rates(_Fake())


# ═════════════════════════════════════════ tail runner：计数钉子随 root 变
def test_tail_scale_pin_applies_only_to_the_default_qwen_root():
    runner = _load_script("run_stage1_tail_stabilized")
    assert runner.pinned_scale() == {"h1": 8664, "mcq": 1750, "e3_pairs": 9916}
    runner.set_artifact_root(Path("some") / "llama_root")
    assert runner.pinned_scale() is None, "独立 root 下不得沿用 Qwen 的先验计数"
    # 但输入变化仍会被拦下：source freeze 含两臂 finalized records 的 sha256
    assert "h1_completion_records.jsonl" in runner.FREEZE_PROD_FILES


def test_tail_scale_pin_rejects_a_wrong_count_on_the_default_root():
    runner = _load_script("run_stage1_tail_stabilized")
    with pytest.raises(runner.ScopeViolation, match="冻结钉子"):
        runner.assert_scale({"h1": 8000, "mcq": 1750, "e3_pairs": 9916})
