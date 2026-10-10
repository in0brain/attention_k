"""W0.5-C：production preflight 的护栏测试。

这个 preflight 的价值全在于**它不能骗人**:
  - G1 红时绝不许 launch_allowed
  - 代码没写完时不许把锅推给 CFP（须落在 stage0_blocked_by，而非投稿项）
  - 只读:不得改 G1、不得改 hash、不得加载权重
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import conditional_increment as ci
from recover_attention import stage0_preflight as pf

OK = {"ok": True}
BAD = {"ok": False}


def _gates(g1=True, g2=True, g3=True):
    return {"G1": {"ok": g1, "status": "confirmed" if g1 else "provisional",
                   "missing": [] if g1 else ["page_limit"]},
            "G2": {"ok": g2, "current": "x", "locked": "x"},
            "G3": {"ok": g3, "h1_arm": OK, "mcq_arm": OK, "backend_invariant": OK}}


# ---- verdict：v2.4 起分两类 gate（科学启动 / 投稿就绪）----
def test_cfp_blocks_submission_only():
    """v2.4:G1 是场地事实,不再阻塞实验,但仍阻塞投稿(§2)。"""
    v = pf.launch_readiness(_gates(g1=False), OK, OK, OK, OK, g4=OK)
    assert v["stage0_launch_allowed"] is True
    assert v["submission_allowed"] is False
    assert v["submission_blocked_by"] == ["external_cfp_confirmation"]
    assert "2026-07-29" in v["what_unblocks_submission"]


def test_missing_code_still_blocks_stage0():
    """原意保留:代码没写完不得启动全量 —— 只是不再和 CFP 混为一谈。"""
    v = pf.launch_readiness(_gates(g1=False), BAD, OK, OK, OK, g4=OK)
    assert v["stage0_launch_allowed"] is False
    assert "code_not_ready" in v["stage0_blocked_by"]
    # 且它不该被算进投稿阻塞项
    assert v["submission_blocked_by"] == ["external_cfp_confirmation"]


def test_all_green_allows_launch():
    v = pf.launch_readiness(_gates(), OK, OK, OK, OK, g4=OK)
    assert v["stage0_launch_allowed"] is True and v["stage0_blocked_by"] == []
    assert v["submission_allowed"] is True


def test_each_failure_surfaces_its_own_reason():
    cases = [
        (_gates(g2=False), OK, OK, OK, OK, OK, "preregistration_hash_mismatch"),
        (_gates(g3=False), OK, OK, OK, OK, OK, "model_smoke_not_green"),
        (_gates(), OK, BAD, OK, OK, OK, "input_assets_missing"),
        (_gates(), OK, OK, BAD, OK, OK, "insufficient_disk"),
        (_gates(), OK, OK, OK, BAD, OK, "frozen_constants_drifted"),
        (_gates(), OK, OK, OK, OK, BAD, "production_quality_check_not_passed"),
    ]
    for g, c, a, s, i, g4, expected in cases:
        v = pf.launch_readiness(g, c, a, s, i, g4=g4)
        assert expected in v["stage0_blocked_by"], f"{expected} 未被报出"
        assert v["stage0_launch_allowed"] is False


# ---- 冻结常量 ----
def test_frozen_invariants_match_preregistration():
    inv = pf.check_frozen_invariants()
    assert inv["ok"] is True
    assert inv["eps_auroc"]["value"] == 0.02 and inv["eps_auroc"]["ok"]
    assert inv["delta_rank_biserial"]["value"] == 0.15
    assert inv["c_grid"]["value"] == [0.01, 0.1, 1.0, 10.0]
    assert inv["rel_depth"]["value"] == 0.7
    assert inv["K_traces"]["value"] == 6
    assert inv["precision_max_ci_width"]["value"] == pytest.approx(0.04)


# ---- 规模:纠正"40GB"那个错 ----
def test_scale_projection_is_megabytes_not_gigabytes():
    """§8 的 H 是每 completion 一个 pooled 向量(14 KB),不是每 token 一个。
    早前"≈40GB"按全 token 算 —— 差三个数量级,会误导出没必要的分片设计。"""
    s = pf.check_scale(Path("."))
    assert s["hidden_bytes_per_vector"] == 3584 * 4
    assert s["h1"]["hidden_mb"] == pytest.approx(41.3, abs=0.5)
    assert s["mcq"]["hidden_mb"] == pytest.approx(25.2, abs=0.5)
    assert s["total_hidden_mb"] < 100, "总量应是几十 MB 级"
    assert s["sharding_required"] is False


def test_scale_uses_frozen_population_sizes():
    s = pf.check_scale(Path("."))
    assert s["h1"]["prompts"] == 480 and s["h1"]["K"] == 6
    assert s["h1"]["generation_units"] == 2880
    assert s["mcq"]["questions"] == 1760
    # §3 v2.3:MCQ population = 每题一条 greedy;sampled 只喂 F5
    assert s["mcq"]["population"] == 1760


# ---- 真实仓库状态 ----
def test_real_repo_is_blocked_only_by_external_cfp():
    """对真实仓库跑一遍:应当只差 CFP。若这条红了,说明有别的东西没做完。"""
    gates = pf.check_gates(ROOT / "docs/paper/preregistration.md",
                           ROOT / "docs/paper/preregistration.lock",
                           ROOT / "docs/paper/cfp_record.json",
                           ROOT / "outputs/logs/sprint_4D_2_conditional_increment")
    if not gates["G3"]["ok"]:
        pytest.skip("G3 smoke artifacts not present in this checkout")
    assert gates["G2"]["ok"] is True
    # 不耦合到具体场地：2026-08-19 换目标后缺的事实从 page_limit 变成三项全缺。
    assert gates["G1"]["ok"] is False
    assert gates["G1"]["missing"], "G1 必须因缺事实而红，不得被刷绿"
    inv = pf.check_frozen_invariants()
    assets = pf.check_input_assets(
        ROOT / "data/processed/h1/h1_samples.jsonl",
        ROOT / "data/processed/cyber/cybermetric.jsonl",
        ROOT / "outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/trace_sampling_manifest.jsonl",
        ROOT / "data/raw/ontology", Path("D:/models/Qwen2.5-7B-Instruct"))
    if not assets["ok"]:
        pytest.skip(f"input assets unavailable in this checkout: {assets}")
    assert assets["h1_samples"]["n_unique"] == 480
    assert assets["mcq_split"]["n_fresh"] == 1760
    assert assets["mcq_split"]["intersection"] == 0
    assert assets["model_config"]["hidden_index"]["hidden_states_tuple_index"] == 20
    assert assets["model_config"]["weights_loaded"] is False, "preflight 不得加载权重"


def test_preflight_does_not_mutate_gate_state():
    """只读契约:跑 preflight 前后 cfp_record 与 prereg hash 必须逐字节不变。"""
    import hashlib
    paths = [ROOT / "docs/paper/cfp_record.json", ROOT / "docs/paper/preregistration.md",
             ROOT / "docs/paper/preregistration.lock"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    pf.check_gates(paths[1], paths[2], paths[0],
                   ROOT / "outputs/logs/sprint_4D_2_conditional_increment")
    pf.check_frozen_invariants()
    after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert before == after, "preflight 必须只读 —— 它不得改 G1 或 hash"


def test_check_gates_reads_g1_from_record_content_not_existence(tmp_path):
    """G1 校验内容而非存在性:占位文件不得刷绿。"""
    rec = tmp_path / "cfp.json"
    rec.write_text(json.dumps({"status": "confirmed", "deadline": "d",
                               "archival": False, "page_limit": None}), encoding="utf-8")
    g = pf.check_gates(ROOT / "docs/paper/preregistration.md",
                       ROOT / "docs/paper/preregistration.lock", rec, tmp_path)
    assert g["G1"]["ok"] is False and "page_limit" in g["G1"]["missing"]
