"""correction sprint 六项修正的测试。

**只用 synthetic fixture**：不读真实 finalized 的 feature-label 联合关系、
不解盲、不写 unblinding log、不创建真实分析输出。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_grouping as grouping           # noqa: E402
from recover_attention.evaluation import v3_population_declaration as pdec  # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fixture   # noqa: E402
from recover_attention.evaluation import v3_unblinding_boundary as ub      # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_stage1_v3_confirmatory", _REPO_ROOT / "scripts" / "run_stage1_v3_confirmatory.py")
entry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = entry
_spec.loader.exec_module(entry)


# ══ ① CLI dry-run 必须用 synthetic 配置 ════════════════════════════════════
def test_cli_dry_run_uses_synthetic_cfg_not_production(monkeypatch):
    seen = {}
    monkeypatch.setattr(entry, "run_dry", lambda **kw: seen.update(kw) or {"ok": True})
    assert entry.main(["--arm", "mcq"]) == 0
    cfg = seen["cfg"]
    assert cfg.mode == "dry_run", "CLI 默认路径不得使用 production 配置"
    assert cfg.b_batch * cfg.b_per_batch < 20000
    assert cfg.b_batch * cfg.b_per_batch == fixture.synthetic_cfg().b_total


def test_cli_explicit_dry_run_flag_also_synthetic(monkeypatch):
    seen = {}
    monkeypatch.setattr(entry, "run_dry", lambda **kw: seen.update(kw) or {"ok": True})
    entry.main(["--arm", "h1", "--dry-run"])
    assert seen["cfg"].mode == "dry_run"


def test_production_cfg_would_have_been_20000():
    """对照：production 预算确实是 20,000 —— 说明 ① 修的是真问题。"""
    c = V3AnalysisConfig()
    assert c.mode == "production" and c.b_batch * c.b_per_batch == 20000


# ══ ② M4：validate_declaration 双调用 + 恒等式 + analyzable==0 ══════════════
def _filled(analyzable, pos, neg, rate, **extra):
    """两臂同值的 filled 声明（schema 要求 arms 恰为 {h1, mcq}）。"""
    arm = {"analyzable": analyzable, "n_positive": pos, "n_negative": neg,
           "positive_rate": rate, **extra}
    return {"schema_version": pdec.DECLARATION_SCHEMA,
            "arms": {"h1": dict(arm), "mcq": dict(arm)},
            "label_dependent_fields": list(pdec.LABEL_DEPENDENT_FIELDS),
            "label_dependent_status": "filled", "performance_blind": False,
            "attrition_categories": {}}


def test_m4_schema_source_field_is_analyzable():
    assert pdec.ANALYZABLE_FIELD == "analyzable"
    d = pdec.build_declaration(h1={"analyzable": 8664}, mcq={"analyzable": 1750})
    assert d["arms"]["h1"]["analyzable"] == 8664


def test_m4_filled_with_missing_values_is_rejected():
    with pytest.raises(ValueError, match="不得为 None"):
        pdec.build_declaration(h1={"analyzable": 10}, mcq={"analyzable": 10},
                               include_label_dependent=True)


def test_m4_identity_must_hold():
    pdec.validate_declaration(_filled(10, 4, 6, 0.4))
    with pytest.raises(ValueError, match="!= analyzable"):
        pdec.validate_declaration(_filled(10, 4, 5, 0.4))
    with pytest.raises(ValueError, match="positive_rate"):
        pdec.validate_declaration(_filled(10, 4, 6, 0.5))


def test_m4_analyzable_zero_has_no_division():
    ok = _filled(0, 0, 0, None, undefined_rate_reason="analyzable == 0")
    pdec.validate_declaration(ok)
    with pytest.raises(ValueError, match="必须为 None"):
        pdec.validate_declaration(_filled(0, 0, 0, 0.0,
                                          undefined_rate_reason="analyzable == 0"))
    with pytest.raises(ValueError, match="undefined_rate_reason"):
        pdec.validate_declaration(_filled(0, 0, 0, None))


def test_m4_writer_revalidates_after_tampering(tmp_path):
    """构造后被篡改再写入 —— writer 必须拦住。"""
    d = pdec.build_declaration(h1={"analyzable": 3}, mcq={"analyzable": 3})
    d["arms"]["h1"]["n_positive"] = 2                    # 篡改
    with pytest.raises(ValueError, match="未解盲声明含真实 label 派生值"):
        pdec.write_declaration(tmp_path / "d.json", d)
    assert not (tmp_path / "d.json").exists()


# ══ ③ E3：结构状态更名 + 任何路径都不读 label ══════════════════════════════
class _NoLabelMapping(Mapping):
    """访问正常结构字段合法；一旦触碰 "label"（下标或 .get）立即抛错。"""

    def __init__(self, data):
        self._d = dict(data)

    def _guard(self, key):
        if key == "label":
            raise AssertionError("E3 在 repooling 完成前**不得**读取 label")

    def __getitem__(self, k):
        self._guard(k)
        return self._d[k]

    def get(self, k, default=None):
        self._guard(k)
        return self._d.get(k, default)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


def _probe_records(n=8):
    return [_NoLabelMapping({
        "trace_id": f"t{i:04d}", "group_id": f"p{i}", "label": i % 2,
        "sample_type": "greedy",
        "mentions": [{"family": "attack", "normalized": f"T{1000 + i:04d}", "emitted": True}],
    }) for i in range(n)]


def test_e3_never_touches_label_on_any_path():
    cfg = V3AnalysisConfig()
    recs = _probe_records()
    out = grouping.build_e3_groups(recs, cfg, arm="h1")          # 无 block_index
    assert out["uses_label"] is False
    used = {grouping.e3_block_of(f"T{1000 + i:04d}", cfg) for i in range(8)}
    for b in range(cfg.e3_n_blocks):                              # 每个块都走一遍
        r = grouping.build_e3_groups(recs, cfg, arm="h1", block_index=b)
        assert r["fold_structure_status"] in (grouping.NON_EMPTY, grouping.EMPTY)
        assert r["class_estimability"] == grouping.NOT_EVALUATED
        assert r["wiring_status"] == grouping.NEEDS_DECISION
    assert used


def test_e3_status_fields_are_fixed_on_all_paths():
    cfg = V3AnalysisConfig()
    recs, _h = fixture.make_h1(n_prompts=6)
    for kw in ({}, {"block_index": 0}, {"block_index": cfg.e3_n_blocks - 1}):
        out = grouping.build_e3_groups(recs, cfg, arm="h1", **kw)
        assert out["wiring_status"] == grouping.NEEDS_DECISION
        assert out["class_estimability"] == grouping.NOT_EVALUATED
        assert out["uses_label"] is False
        assert "fold_status" not in out, "旧名 fold_status 应已移除"


def test_e3_empty_block_is_structure_only():
    cfg = V3AnalysisConfig()
    recs, _h = fixture.make_h1(n_prompts=4)
    idents = [f"T{1000 + (p % 90):04d}" for p in range(4)]
    used = {grouping.e3_block_of(i, cfg) for i in idents}
    empty = next(b for b in range(cfg.e3_n_blocks) if b not in used)
    out = grouping.build_e3_groups(recs, cfg, arm="h1", block_index=empty)
    assert out["fold_structure_status"] == grouping.EMPTY
    assert "空 fold" in out["reason"]
    assert out["class_estimability"] == grouping.NOT_EVALUATED


# ══ ④ freeze verifier：畸形输入受控 + hash_matches 并入 ok ══════════════════
def _prod(tmp_path):
    d = tmp_path / "prod"
    d.mkdir(exist_ok=True)
    for n in ub.FINALIZED_SIX:
        (d / n).write_bytes(n.encode())
    return d


def _good(tmp_path):
    return ub.build_stage1_freeze_manifest(_prod(tmp_path)), _prod(tmp_path)


def test_verifier_accepts_good_manifest(tmp_path):
    m, prod = _good(tmp_path)
    v = ub.verify_stage1_freeze_manifest(m, prod)
    assert v["ok"] is True and v["hash_matches"] is True and v["structural_problems"] == []


@pytest.mark.parametrize("mutate,label", [
    (lambda m: m.pop("entries"), "entries 缺失"),
    (lambda m: m.update(entries="notalist"), "entries 非 list"),
    (lambda m: m.update(entries=[1, 2, 3, 4, 5, 6]), "entry 非 dict"),
    (lambda m: m["entries"][0].update(bytes=True), "bytes 为 bool"),
    (lambda m: m["entries"][0].update(bytes=-1), "bytes 负数"),
    (lambda m: m["entries"][0].update(bytes="10"), "bytes 非 int"),
    (lambda m: m["entries"][0].update(sha256="ABC"), "sha256 非 64 位十六进制"),
    (lambda m: m["entries"][0].update(sha256="A" * 64), "sha256 非小写"),
    (lambda m: m["entries"].__setitem__(1, dict(m["entries"][0])), "重复路径"),
    (lambda m: m["entries"][0].update(relative_path="/abs/x.json"), "绝对路径"),
    (lambda m: m["entries"][0].update(relative_path="../x.json"), "含 .."),
    (lambda m: m["entries"][0].update(relative_path="a/b.json"), "含 /"),
    (lambda m: m["entries"][0].update(relative_path="a\\b.json"), "含反斜杠"),
    (lambda m: m.update(n_artifacts=5), "n_artifacts != 6"),
    (lambda m: m["entries"].pop(), "entries 少一条"),
    (lambda m: m.pop("freeze_hash"), "freeze_hash 缺失"),
    (lambda m: m.update(freeze_hash="zz"), "freeze_hash 格式错"),
    (lambda m: m.update(schema_version="wrong"), "schema_version 错"),
])
def test_verifier_returns_ok_false_on_malformed(tmp_path, mutate, label):
    m, prod = _good(tmp_path)
    mutate(m)
    v = ub.verify_stage1_freeze_manifest(m, prod)      # **不得抛异常**
    assert v["ok"] is False, label
    assert v["structural_problems"], label


def test_verifier_non_dict_input_is_controlled(tmp_path):
    _m, prod = _good(tmp_path)
    for bad in (None, [], "x", 3):
        v = ub.verify_stage1_freeze_manifest(bad, prod)
        assert v["ok"] is False and v["structural_problems"]


def test_verifier_ok_includes_hash_matches(tmp_path):
    m, prod = _good(tmp_path)
    m["freeze_hash"] = "0" * 64                        # 结构合法但 hash 不符
    v = ub.verify_stage1_freeze_manifest(m, prod)
    assert v["hash_matches"] is False and v["ok"] is False


# ══ ⑤ D 降级为数学 smoke ═══════════════════════════════════════════════════
def test_contrast_is_math_smoke_not_end_to_end():
    cfg = fixture.synthetic_cfg(b_batch=2, b_per_batch=5)
    out = fixture.contrast_math_smoke(cfg)
    assert out["kind"] == "contrast_math_smoke"
    assert out["is_end_to_end"] is False
    assert out["uses_frozen_O_definition"] is False
    assert not hasattr(fixture, "run_contrast_end_to_end"), "旧名不应保留"


def test_dry_run_scope_excludes_contrast():
    cfg = fixture.synthetic_cfg(b_batch=2, b_per_batch=5)
    r = entry.run_dry(arm="both", estimand="E1", cfg=cfg)
    assert "per-arm E1 only" in r["end_to_end_scope"]
    assert r["contrast_math_smoke"]["is_end_to_end"] is False
    assert "contrast_D" not in r
    single = entry.run_dry(arm="h1", estimand="E1", cfg=cfg)
    assert single["contrast_math_smoke"] is None


# ══ ⑥ 结果 writer 加固 ═════════════════════════════════════════════════════
def test_result_writer_normalizes_numpy(tmp_path):
    out = tmp_path / "r.json"
    entry.write_result_atomic(out, {"a": np.float32(1.5), "b": np.int64(3),
                                    "c": np.array([1.0, 2.0]), "d": np.bool_(True)})
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back == {"a": 1.5, "b": 3, "c": [1.0, 2.0], "d": True}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float32("nan")])
def test_result_writer_rejects_non_finite(tmp_path, bad):
    out = tmp_path / "r.json"
    with pytest.raises(ValueError):
        entry.write_result_atomic(out, {"x": bad})
    assert not out.exists(), "拒绝时不得留下半文件"


def test_result_writer_leaves_no_partial_file(tmp_path):
    out = tmp_path / "r.json"

    class Boom:
        pass

    with pytest.raises(TypeError):
        entry.write_result_atomic(out, {"x": Boom()})
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp")), "临时文件必须被清理"


def test_result_writer_publishes_once(tmp_path):
    """正常路径：一次性写成，无临时残留。"""
    out = tmp_path / "r.json"
    assert entry.write_result_atomic(out, {"v": 1}) == out
    assert json.loads(out.read_text(encoding="utf-8")) == {"v": 1}
    assert not list(tmp_path.glob("*.tmp")), "发布后临时链接必须删除"


def test_result_writer_refuses_a_preexisting_target(tmp_path):
    """目标先前就存在 → 拒绝，且**目标字节不变**。

    旧测试曾要求"第二次写覆盖第一次"（`os.replace` 语义）。那是错的：
    preflight 只能保证分析开始前目标不存在，覆盖会让一次性 confirmatory 产物
    被悄悄替换。这里把契约反过来钉死。
    """
    out = tmp_path / "r.json"
    entry.write_result_atomic(out, {"v": 1})
    before = out.read_bytes()
    with pytest.raises(entry.ResultPublishRefused, match="已存在"):
        entry.write_result_atomic(out, {"v": 2})
    assert out.read_bytes() == before, "既有结果字节必须不变"
    assert not list(tmp_path.glob("*.tmp"))
    # 已解盲就是已解盲：报错必须指向 recovery，而不是暗示可以重写
    with pytest.raises(entry.ResultPublishRefused, match="recovery"):
        entry.write_result_atomic(out, {"v": 3})


def test_result_writer_refuses_a_target_that_appears_during_the_write(tmp_path,
                                                                     monkeypatch):
    """落盘期竞态：目标在 fsync 之后、发布之前出现 → 原子失败，不覆盖。"""
    import os as _os
    out = tmp_path / "r.json"
    real_link = _os.link

    def racing_link(src, dst, *a, **k):
        Path(dst).write_bytes(b'{"other": "writer"}\n')   # 别的进程抢先落地
        return real_link(src, dst, *a, **k)

    # `write_result_atomic` 内部 `import os`，拿到的就是这个模块对象
    monkeypatch.setattr(_os, "link", racing_link)
    with pytest.raises(entry.ResultPublishRefused, match="落盘期间出现"):
        entry.write_result_atomic(out, {"v": 1})
    assert out.read_bytes() == b'{"other": "writer"}\n', "抢先者的字节必须原样保留"
    assert not list(tmp_path.glob("*.tmp"))


def test_result_writer_never_uses_replace():
    """`os.replace` 无条件覆盖 —— 正式结果发布路径里不得出现它。"""
    src = (Path(entry.__file__).read_text(encoding="utf-8")
           if hasattr(entry, "__file__") else "")
    body = src[src.index("def write_result_atomic("):src.index("def run_dry(")]
    assert "os.replace(" not in body, "发布路径不得调用 os.replace"
    assert "os.link(tmp, str(path))" in body
    assert "不退回 os.replace" in body        # 只作说明，不是调用


# ══ 边界：本轮不得产生真实解盲产物 ═════════════════════════════════════════
@pytest.mark.pre_unblinding_workspace
def test_no_real_unblinding_artifacts():
    assert not entry.RESULT_DIR.exists()
    assert not entry.UNBLIND_LOG.exists()
