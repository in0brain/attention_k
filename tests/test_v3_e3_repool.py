"""E3 repool 纯函数核心的单测。全部用合成 fixture，**不加载模型、不碰真实产物**。

覆盖研究者逐条裁定的边界：
  * prompt 适配（finalized record 只有 prompt_text）与 SHA 校验
  * label-blind：删去全部 label 字段后**块归属与 position digest 相同**
    （证据范围仅此，不含 hidden 输出的两次前向比较）
  * 块归属 = token_indices ∩ id_token_positions_local（echoed/CVE 自动出局）
  * union anchor 不等于"块均值再平均"
  * 精确一致，无容差
  * 锚点选取确定性、分层、可复算
  * 指纹拆分：路径进"允许不同"，身份哈希路径无关
  * .npz **自洽**校验（无 done 事件时没有既有 SHA 可比）
本轮**不测**也**不实现** WorkManifest / production writer / failed 事件 / orphan 自动收敛。
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig
from recover_attention.evaluation.v3_grouping import e3_block_of
from recover_attention.generation.fingerprint import sha256_text

CFG = V3AnalysisConfig()
HD = repool.HIDDEN_DIM


# --------------------------------------------------------------- fixtures
class FakeTokenizer:
    """够用的替身：chat template 是确定性字符串拼接，编码按空格切分。"""

    name_or_path = "fake/model"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens=True, **kw):
        return {"input_ids": list(range(len(text.split())))}


def make_row(tokenizer=None, *, idents=("T1001", "T1002"), n_tok_each=2,
             completion=None, unit=("ex_a", "greedy", 0)):
    """构造一条 label-free 可用的合成 record（同时带上 label 字段以便测 label-blind）。"""
    words, mentions, cursor = [], [], 0
    for ident in idents:
        toks = [f"{ident}_{i}" for i in range(n_tok_each)]
        mentions.append({"normalized": ident,
                         "token_indices": list(range(cursor, cursor + n_tok_each)),
                         "family": "attack", "label": "fabricated"})
        words.extend(toks)
        cursor += n_tok_each
    text = completion if completion is not None else " ".join(words)
    positions = sorted({p for m in mentions for p in m["token_indices"]})
    row = {
        "example_id": unit[0], "sample_type": unit[1], "sample_index": unit[2],
        "prompt_text": "What ATT&CK techniques apply?",
        "completion": text,
        "mentions": mentions,
        "id_token_positions_local": positions,
        "f5_completion_token_count": len(text.split()),
        "label": 1, "gold_label": "A",
    }
    if tokenizer is not None:
        row["prompt_hash"] = sha256_text(repool.render_prompt(row, tokenizer))
    return row


# ------------------------------------------------------------ prompt 适配
def test_build_prompt_record_maps_prompt_text_to_question_text():
    row = make_row()
    assert "question_text" not in row
    assert repool.build_prompt_record(row)["question_text"] == row["prompt_text"]


def test_build_prompt_record_rejects_missing_or_blank_prompt_text():
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="prompt_text"):
            repool.build_prompt_record({**make_row(), "prompt_text": bad})


def test_verify_prompt_and_positions_accepts_matching_row():
    tok = FakeTokenizer()
    out = repool.verify_prompt_and_positions(make_row(tok), tok)
    assert out["unit_id"] == "ex_a__greedy_0"
    assert out["n_union_positions"] == 4


def test_verify_prompt_rejects_prompt_hash_drift():
    tok = FakeTokenizer()
    row = {**make_row(tok), "prompt_hash": "deadbeef"}
    with pytest.raises(ValueError, match="prompt_hash mismatch"):
        repool.verify_prompt_and_positions(row, tok)


def test_f5_completion_token_count_is_recorded_but_never_asserted():
    """`f5_completion_token_count` = len(有限 completion logprob)，**不是**序列长度
    （h1_f5_features.py:94）。真实 finalized 上 8,663/8,664 条二者不等，因此它只能作为
    信息量登记；把它当序列长度断言会让全量校验假失败。"""
    tok = FakeTokenizer()
    row = {**make_row(tok), "f5_completion_token_count": 99}
    out = repool.verify_prompt_and_positions(row, tok)          # 不得抛错
    assert out["f5_completion_token_count"] == 99
    assert out["f5_count_equals_sequence_length"] is False
    assert out["n_completion_tokens"] == len(row["completion"].split())


def test_verify_prompt_rejects_out_of_range_positions():
    tok = FakeTokenizer()
    row = make_row(tok)
    row["id_token_positions_local"] = [0, 999]
    with pytest.raises(ValueError, match="out of range"):
        repool.verify_prompt_and_positions(row, tok)


def test_verify_prompt_rejects_empty_positions():
    tok = FakeTokenizer()
    row = make_row(tok)
    row["id_token_positions_local"] = []
    with pytest.raises(ValueError, match="no eligible identifier"):
        repool.verify_prompt_and_positions(row, tok)


# ------------------------------------------------------------- label-blind
def test_block_assignment_is_bit_identical_without_label_fields():
    row = make_row()
    stripped = repool.strip_label_fields(row)
    assert "label" not in stripped
    assert all("label" not in m and "emitted" not in m for m in stripped["mentions"])
    assert repool.block_assignment(row, CFG) == repool.block_assignment(stripped, CFG)
    assert repool.expected_blocks(row, CFG) == repool.expected_blocks(stripped, CFG)
    assert (repool.position_digest(repool.block_assignment(row, CFG))
            == repool.position_digest(repool.block_assignment(stripped, CFG)))


def test_strip_label_fields_leaves_non_label_metadata_intact():
    row = make_row()
    stripped = repool.strip_label_fields(row)
    assert stripped["mentions"][0]["family"] == "attack"
    assert stripped["id_token_positions_local"] == row["id_token_positions_local"]


# --------------------------------------------------------------- 块归属
def test_mentions_outside_frozen_positions_are_dropped():
    """echoed / CVE mention 的 token 不在 id_token_positions_local 内 -> 自动出局。"""
    row = make_row()
    row["mentions"].append({"normalized": "CVE-2021-44228", "token_indices": [50, 51],
                            "family": "cve", "label": "grounded"})
    assignment = repool.block_assignment(row, CFG)
    covered = {p for pos in assignment.values() for p in pos}
    assert covered == set(row["id_token_positions_local"])
    assert not (covered & {50, 51})                  # CVE mention 的 token 一个都没进
    baseline = repool.block_assignment(
        {**row, "mentions": row["mentions"][:-1]}, CFG)
    assert assignment == baseline                    # 加不加这条 CVE mention，结果不变
    repool.check_block_partition(row, CFG)


def test_block_partition_union_equals_frozen_positions():
    row = make_row(idents=("T1001", "T1002", "T1003"))
    assignment = repool.block_assignment(row, CFG)
    union = sorted(p for pos in assignment.values() for p in pos)
    assert union == row["id_token_positions_local"]
    repool.check_block_partition(row, CFG)


def test_check_block_partition_detects_missing_coverage():
    row = make_row()
    row["id_token_positions_local"] = row["id_token_positions_local"] + [99]
    with pytest.raises(ValueError, match="block union"):
        repool.check_block_partition(row, CFG)


def test_same_identifier_twice_collapses_into_one_block():
    row = make_row(idents=("T1001",))
    row["mentions"].append({"normalized": "T1001", "token_indices": [0, 1],
                            "family": "attack", "label": "fabricated"})
    assignment = repool.block_assignment(row, CFG)
    assert len(assignment) == 1
    assert list(assignment.values())[0] == [0, 1]


def test_block_ids_follow_frozen_salt_and_modulus():
    row = make_row(idents=("T1001", "T1002"))
    assert set(repool.block_assignment(row, CFG)) == {
        e3_block_of("T1001", CFG), e3_block_of("T1002", CFG)}
    assert all(0 <= b < CFG.e3_n_blocks for b in repool.expected_blocks(row, CFG))


# --------------------------------------------------------------- pooling
def test_pool_union_is_not_the_mean_of_block_means():
    """块大小不同时，块均值的均值 != 全体均值 —— 锚点必须用 union pooling。"""
    hidden = np.arange(20 * HD, dtype=np.float32).reshape(20, HD)
    assignment = {0: [0], 1: [1, 2, 3]}
    union = repool.pool_union(hidden, 0, [0, 1, 2, 3])
    blocks = repool.pool_blocks(hidden, 0, assignment)
    naive = np.mean([blocks[0], blocks[1]], axis=0)
    assert not np.allclose(union, naive)
    assert np.allclose(union, hidden[[0, 1, 2, 3]].mean(axis=0))


def test_pool_union_offsets_by_prompt_len():
    hidden = np.arange(10 * HD, dtype=np.float32).reshape(10, HD)
    assert np.array_equal(repool.pool_union(hidden, 4, [1, 2]),
                          hidden[[5, 6]].mean(axis=0))


def test_anchor_exact_match_reports_exact():
    v = np.arange(HD, dtype=np.float32)
    assert repool.anchor_exact_match(v, v.copy())["exact"] is True


def test_anchor_exact_match_rejects_tiny_difference_without_tolerance():
    a = np.ones(HD, dtype=np.float32)
    b = a.copy()
    b[7] = np.nextafter(np.float32(1.0), np.float32(2.0))
    diag = repool.anchor_exact_match(a, b)
    assert diag["exact"] is False
    assert diag["n_differing"] == 1
    assert diag["argmax_index"] == 7
    assert diag["ulp_distance_at_argmax"] == 1
    assert 0 < diag["max_abs_diff"] < 1e-6           # 极小，但**仍然判失败**


def test_anchor_exact_match_reports_shape_mismatch():
    diag = repool.anchor_exact_match(np.zeros(HD, np.float32), np.zeros(8, np.float32))
    assert diag["exact"] is False and diag["reason"] == "shape"


# ---------------------------------------------------------------- 锚点选取
def _anchor_corpus():
    rows = []
    for n_blocks, count in ((4, 2), (3, 11), (2, 40), (1, 100)):
        for i in range(count):
            idents = [f"T{2000 + n_blocks * 1000 + i * 10 + k}" for k in range(8)]
            row = make_row(idents=idents, unit=(f"ex_{n_blocks}_{i:03d}", "greedy", 0))
            # 只保留恰好落在 n_blocks 个块里的 mention
            keep, seen = [], set()
            for m in row["mentions"]:
                b = e3_block_of(m["normalized"], CFG)
                if b in seen or len(seen) < n_blocks:
                    seen.add(b)
                    keep.append(m)
            if len(seen) != n_blocks:
                continue
            row["mentions"] = keep
            row["id_token_positions_local"] = sorted(
                {p for m in keep for p in m["token_indices"]})
            rows.append(row)
    return rows


def test_select_anchor_units_is_deterministic_and_stratified():
    rows = _anchor_corpus()
    first = repool.select_anchor_units(rows, CFG)
    second = repool.select_anchor_units(list(reversed(rows)), CFG)
    assert first == second                                    # 与输入顺序无关
    assert len(first) == repool.ANCHOR_SIZE
    by_uid = {repool.derive_unit_id(r): r for r in rows}
    counts = {}
    for uid in first:
        n = len(repool.expected_blocks(by_uid[uid], CFG))
        counts[n] = counts.get(n, 0) + 1
    assert counts[4] == 2 and counts[3] == 11                 # 高块数**全取**
    assert set(counts) == {1, 2, 3, 4}                        # 四层都被覆盖


def test_anchor_set_sha256_is_order_independent():
    ids = ["b", "a", "c"]
    assert repool.anchor_set_sha256(ids) == repool.anchor_set_sha256(reversed(ids))
    assert repool.anchor_set_sha256(ids) == sha256_text("a\nb\nc")


# ---------------------------------------------------------------- 指纹拆分
FROZEN_FP = {
    "model": {"model_path": "D:/models/Qwen2.5-7B-Instruct", "model_type": "qwen2",
              "num_hidden_layers": 28, "hidden_size": 3584, "vocab_size": 152064,
              "torch_dtype": "torch.float16",
              "backend": {"load_in_8bit": True, "device_map": "auto",
                          "attn_implementation": "eager", "local_files_only": True},
              "model_hash": "frozen-hash"},
    "tokenizer": {"class": "Qwen2Tokenizer", "name_or_path": "D:/models/Qwen2.5-7B-Instruct",
                  "vocab_size": 151643, "special_token_ids": {"eos_token_id": 151645},
                  "tokenizer_hash": "frozen-tok-hash"},
    "hidden_index": {"num_layers": 28, "rel_depth": 0.7, "block_index_zero_based": 19,
                     "hidden_states_tuple_index": 20},
}
LIBS = {"torch": "2.7.0", "transformers": "4.51.0",
        "bitsandbytes": "0.45.0", "accelerate": "1.4.0"}
SHAS = {"h1_completion_records.jsonl": "aa", "h1_run_fingerprint.json": "bb",
        "h1_unit_hidden_tree_digest": "cc"}


def test_identity_hashes_are_path_independent():
    """跨机迁移路径必然不同——身份哈希不能把路径算进去，否则远程永远无法通过。"""
    moved = json.loads(json.dumps(FROZEN_FP))
    moved["model"]["model_path"] = "/mnt/remote/Qwen2.5-7B-Instruct"
    moved["model"]["model_hash"] = "different-because-path-changed"
    moved["tokenizer"]["name_or_path"] = "/mnt/remote/Qwen2.5-7B-Instruct"
    moved["tokenizer"]["tokenizer_hash"] = "different-too"
    assert (repool.model_identity_hash(FROZEN_FP["model"])
            == repool.model_identity_hash(moved["model"]))
    assert (repool.tokenizer_identity_hash(FROZEN_FP["tokenizer"])
            == repool.tokenizer_identity_hash(moved["tokenizer"]))


def test_path_and_gpu_keys_are_declared_variable_not_hard():
    for key in ("model_path", "tokenizer_name_or_path", "model_hash", "tokenizer_hash",
                "gpu_name", "driver_version", "compute_capability",
                "directory_fsync_supported"):
        assert key in repool.VARIABLE_FINGERPRINT_KEYS
    hard = repool.hard_consistency_payload(FROZEN_FP, CFG, SHAS, LIBS)
    assert not (set(hard) & set(repool.VARIABLE_FINGERPRINT_KEYS))


def test_hard_consistency_accepts_moved_paths_and_rejects_real_drift():
    ref = repool.hard_consistency_payload(FROZEN_FP, CFG, SHAS, LIBS)
    moved = json.loads(json.dumps(FROZEN_FP))
    moved["model"]["model_path"] = "/mnt/remote/model"
    moved["tokenizer"]["name_or_path"] = "/mnt/remote/model"
    assert repool.compare_hard_consistency(
        ref, repool.hard_consistency_payload(moved, CFG, SHAS, LIBS)) == []

    for mutate in (
        lambda fp: fp["hidden_index"].update({"block_index_zero_based": 18}),
        lambda fp: fp["model"]["backend"].update({"load_in_8bit": False}),
        lambda fp: fp["model"].update({"torch_dtype": "torch.bfloat16"}),
        lambda fp: fp["model"].update({"num_hidden_layers": 32}),
        lambda fp: fp["tokenizer"]["special_token_ids"].update({"eos_token_id": 1}),
    ):
        bad = json.loads(json.dumps(FROZEN_FP))
        mutate(bad)
        assert repool.compare_hard_consistency(
            ref, repool.hard_consistency_payload(bad, CFG, SHAS, LIBS))


def test_hard_consistency_covers_library_versions_salt_and_source_shas():
    ref = repool.hard_consistency_payload(FROZEN_FP, CFG, SHAS, LIBS)
    assert repool.compare_hard_consistency(
        ref, repool.hard_consistency_payload(
            FROZEN_FP, CFG, SHAS, {**LIBS, "torch": "2.8.0"}))
    assert repool.compare_hard_consistency(
        ref, repool.hard_consistency_payload(
            FROZEN_FP, CFG, {**SHAS, "h1_unit_hidden_tree_digest": "dd"}, LIBS))
    assert ref["e3_block_salt"] == CFG.e3_block_salt
    assert ref["pooling_schema"] == repool.POOLING_SCHEMA
    assert ref["token_position_selection_rule"] == repool.TOKEN_POSITION_SELECTION_RULE


def test_compare_hard_consistency_is_fail_closed_on_missing_keys():
    ref = repool.hard_consistency_payload(FROZEN_FP, CFG, SHAS, LIBS)
    assert repool.compare_hard_consistency(ref, {})


# ------------------------------------------------------- .npz 自洽校验
FPID = "fingerprint-id-1"


def write_npz(path, row, cfg=CFG, *, arrays=None, meta_override=None):
    assignment = repool.block_assignment(row, cfg)
    payload = arrays if arrays is not None else {
        f"block_{b}": np.full(HD, float(b) + 1.0, dtype=np.float32) for b in assignment}
    meta = {"trace_id": repool.derive_unit_id(row), "prompt_hash": row.get("prompt_hash"),
            "source_fingerprint_id": FPID,
            "token_positions": {str(b): list(pos) for b, pos in assignment.items()}}
    meta.update(meta_override or {})
    np.savez(path, meta=np.array(json.dumps(meta)), **payload)
    return path


def test_validate_npz_accepts_self_consistent_file(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    p = write_npz(tmp_path / "u.npz", row)
    out = repool.validate_npz_self_consistency(p, row, CFG, FPID)
    assert out["validated"] and out["n_blocks"] == len(repool.expected_blocks(row, CFG))
    assert out["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert out["bytes"] == p.stat().st_size
    assert "no prior done event" in out["note"]


def test_validate_npz_rejects_corrupt_zip(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    p = write_npz(tmp_path / "u.npz", row)
    blob = bytearray(p.read_bytes())
    blob[80] ^= 0xFF                                   # 破坏成员数据 -> CRC 失败
    p.write_bytes(bytes(blob))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)


def test_validate_npz_rejects_wrong_trace_id(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    p = write_npz(tmp_path / "u.npz", row, meta_override={"trace_id": "someone_else"})
    with pytest.raises(ValueError, match="trace_id"):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)


def test_validate_npz_rejects_wrong_prompt_hash_and_fingerprint(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    p = write_npz(tmp_path / "a.npz", row, meta_override={"prompt_hash": "nope"})
    with pytest.raises(ValueError, match="prompt_hash"):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)
    q = write_npz(tmp_path / "b.npz", row, meta_override={"source_fingerprint_id": "x"})
    with pytest.raises(ValueError, match="source_fingerprint_id"):
        repool.validate_npz_self_consistency(q, row, CFG, FPID)


def test_validate_npz_rejects_missing_or_extra_block_keys(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    assignment = repool.block_assignment(row, CFG)
    partial = {f"block_{b}": np.zeros(HD, np.float32) for b in list(assignment)[:-1]}
    p = write_npz(tmp_path / "a.npz", row, arrays=partial)
    with pytest.raises(ValueError, match="block keys"):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)

    extra = {f"block_{b}": np.zeros(HD, np.float32) for b in assignment}
    extra["block_99"] = np.zeros(HD, np.float32)
    q = write_npz(tmp_path / "b.npz", row, arrays=extra)
    with pytest.raises(ValueError, match="block keys"):
        repool.validate_npz_self_consistency(q, row, CFG, FPID)


def test_validate_npz_rejects_position_drift(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    assignment = repool.block_assignment(row, CFG)
    bad = {str(b): [p + 1 for p in pos] for b, pos in assignment.items()}
    p = write_npz(tmp_path / "u.npz", row, meta_override={"token_positions": bad})
    with pytest.raises(ValueError, match="token_positions mismatch"):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)


def test_validate_npz_rejects_bad_dtype_shape_and_nonfinite(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    assignment = repool.block_assignment(row, CFG)
    first = f"block_{sorted(assignment)[0]}"

    arrays = {f"block_{b}": np.zeros(HD, np.float32) for b in assignment}
    arrays[first] = np.zeros(HD, np.float64)
    with pytest.raises(ValueError, match="dtype"):
        repool.validate_npz_self_consistency(
            write_npz(tmp_path / "a.npz", row, arrays=arrays), row, CFG, FPID)

    arrays = {f"block_{b}": np.zeros(HD, np.float32) for b in assignment}
    arrays[first] = np.zeros(7, np.float32)
    with pytest.raises(ValueError, match="shape"):
        repool.validate_npz_self_consistency(
            write_npz(tmp_path / "b.npz", row, arrays=arrays), row, CFG, FPID)

    arrays = {f"block_{b}": np.zeros(HD, np.float32) for b in assignment}
    arrays[first] = np.full(HD, np.nan, np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        repool.validate_npz_self_consistency(
            write_npz(tmp_path / "c.npz", row, arrays=arrays), row, CFG, FPID)


def test_validate_npz_rejects_missing_meta(tmp_path):
    tok = FakeTokenizer()
    row = make_row(tok)
    p = tmp_path / "u.npz"
    np.savez(p, **{f"block_{b}": np.zeros(HD, np.float32)
                   for b in repool.block_assignment(row, CFG)})
    with pytest.raises(ValueError, match="missing embedded meta"):
        repool.validate_npz_self_consistency(p, row, CFG, FPID)


# ------------------------------------------------------------------ 范围
def test_module_declares_no_manifest_or_writer_symbols():
    """本轮不实现 production 恢复状态机——这些名字**不应存在**于模块中。"""
    for banned in ("WorkManifest", "write_npz", "append_done", "mark_failed",
                   "recover_orphans", "run_full", "production_writer"):
        assert not hasattr(repool, banned), f"本轮不应实现 {banned}"


def test_script_has_no_run_subcommand():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_e3_script", __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts" / "run_stage1_e3_repool.py")
    src = spec.origin
    text = __import__("pathlib").Path(src).read_text(encoding="utf-8")
    assert '"--run"' not in text and "'--run'" not in text
    for name in ("h1_e3_block_hidden", "h1_e3_repool_manifest.jsonl",
                 "h1_e3_repool_fingerprint.json", "h1_e3_repool_report.json"):
        # 只允许出现在 FORBIDDEN 断言里，不允许有写入路径
        assert "np.savez" not in text
        assert name in text          # 必须被列入禁止清单


# ============================================================ CLI 安全边界
# A/B/C/D 四项缺口的负控制：修复前可触发，修复后必须被拒。
# 全部不加载模型、不碰真实 outputs/（PROD 与三个目标路径都重定向到 tmp_path）。
import importlib.util  # noqa: E402


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "_e3_cli", _REPO_ROOT / "scripts" / "run_stage1_e3_repool.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cli(tmp_path, monkeypatch):
    """把 CLI 的所有写盘目标重定向到 tmp_path，绝不触碰真实 outputs/。"""
    mod = load_cli()
    prod = tmp_path / "production"
    bench = tmp_path / "benchmark"
    prod.mkdir()
    bench.mkdir()
    monkeypatch.setattr(mod, "PROD", prod)
    monkeypatch.setattr(mod, "BENCH", bench)
    monkeypatch.setattr(mod, "ANCHOR_OUT", bench / "e3_anchor_diagnostic.json")
    monkeypatch.setattr(mod, "BENCH_OUT", bench / "e3_benchmark_report.json")
    monkeypatch.setattr(mod, "TELEMETRY_OUT", bench / "e3_benchmark_gpu_telemetry.csv")
    return mod


# --- A: 遥测必须进硬门，且独占创建 --------------------------------------
def test_A_telemetry_is_gated_whenever_the_model_will_load(cli):
    """遥测在 --anchor 与 --benchmark 两种情况下都会启动，因此两种情况都必须入门。"""
    cli.TELEMETRY_OUT.write_bytes(b"pre-existing telemetry\n")
    before = cli.TELEMETRY_OUT.read_bytes()
    with pytest.raises(cli.ScopeViolation, match="拒绝覆盖"):
        cli.main(["--anchor"])                      # 此前 --anchor 不检查遥测，会截断
    assert cli.TELEMETRY_OUT.read_bytes() == before


def test_A_start_telemetry_refuses_to_truncate_existing_file(cli):
    """`"w"` 会静默截断——必须是 `"x"` 独占创建。"""
    cli.TELEMETRY_OUT.write_bytes(b"do not clobber me\n")
    with pytest.raises(FileExistsError):
        cli.start_telemetry(cli.TELEMETRY_OUT)
    assert cli.TELEMETRY_OUT.read_bytes() == b"do not clobber me\n"


# --- B: --benchmark 必须与 --anchor 同次 --------------------------------
def test_B_benchmark_alone_exits_2_before_loading_the_model(cli, monkeypatch):
    from recover_attention.generation import backends as gb

    def explode(*a, **kw):                          # pragma: no cover - 不应被调用
        raise AssertionError("模型不得在被拒绝的调用里加载")

    monkeypatch.setattr(gb, "_RealBackendBase", explode)
    monkeypatch.setattr(gb, "load_backend", explode)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--benchmark"])
    assert exc.value.code == 2
    assert not cli.BENCH_OUT.exists()
    assert not cli.TELEMETRY_OUT.exists()
    assert not cli.ANCHOR_OUT.exists()


def test_B_no_disk_anchor_reuse_path_exists(cli):
    """刻意不支持"读磁盘旧 anchor 后单跑 benchmark"——那要一整套额外校验。"""
    text = (_REPO_ROOT / "scripts" / "run_stage1_e3_repool.py").read_text(encoding="utf-8")
    assert "--benchmark 必须与 --anchor 同次给出" in text
    for banned in ("load_prior_anchor", "reuse_anchor", "anchor_from_disk"):
        assert not hasattr(cli, banned)


# --- C: 发布只创建不覆盖，fail-closed ------------------------------------
def test_C_publish_refuses_existing_target(cli):
    cli.ANCHOR_OUT.write_bytes(b"{}")
    with pytest.raises(cli.ScopeViolation, match="拒绝覆盖"):
        cli.publish_json_exclusive(cli.ANCHOR_OUT, {"a": 1})
    assert cli.ANCHOR_OUT.read_bytes() == b"{}"


def test_C_publish_refuses_stale_tmp_and_preserves_it(cli):
    tmp = cli.ANCHOR_OUT.with_suffix(cli.ANCHOR_OUT.suffix + ".tmp")
    tmp.write_bytes(b"crash residue")
    with pytest.raises(cli.ScopeViolation, match="陈旧临时文件"):
        cli.publish_json_exclusive(cli.ANCHOR_OUT, {"a": 1})
    assert tmp.read_bytes() == b"crash residue"     # 保留现场
    assert not cli.ANCHOR_OUT.exists()


def test_C_publish_is_atomic_create_not_replace(cli, monkeypatch):
    """目标在写入期间出现时必须原子失败——不得退回 os.replace 覆盖。"""
    calls = []
    monkeypatch.setattr(cli.os, "replace",
                        lambda *a, **kw: calls.append(a))     # 若被调用即失败
    real_link = cli.os.link

    def link_then_conflict(src, dst):
        Path(dst).write_bytes(b"appeared mid-write")
        return real_link(src, dst)                  # 必然 FileExistsError

    monkeypatch.setattr(cli.os, "link", link_then_conflict)
    with pytest.raises(cli.ScopeViolation, match="原子发布失败"):
        cli.publish_json_exclusive(cli.ANCHOR_OUT, {"a": 1})
    assert cli.ANCHOR_OUT.read_bytes() == b"appeared mid-write"
    assert calls == []                              # os.replace 从未被调用


def test_C_publish_fails_closed_when_link_unavailable(cli, monkeypatch):
    def no_link(*a, **kw):
        raise OSError("link unsupported on this filesystem")

    monkeypatch.setattr(cli.os, "link", no_link)
    monkeypatch.setattr(cli.os, "replace",
                        lambda *a, **kw: pytest.fail("不得退回 os.replace"))
    with pytest.raises(cli.ScopeViolation, match="fail-closed"):
        cli.publish_json_exclusive(cli.ANCHOR_OUT, {"a": 1})
    assert not cli.ANCHOR_OUT.exists()
    assert cli.ANCHOR_OUT.with_suffix(".json.tmp").exists()   # 保留现场


def test_C_publish_succeeds_and_leaves_no_tmp(cli):
    cli.publish_json_exclusive(cli.ANCHOR_OUT, {"a": 1, "b": [2, 3]})
    assert json.loads(cli.ANCHOR_OUT.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}
    assert not cli.ANCHOR_OUT.with_suffix(".json.tmp").exists()


# --- D: 段号命名空间也在范围检查内 --------------------------------------
def test_D_scope_check_covers_segment_namespace(cli):
    seg = cli.PROD / "h1_e3_seg001_gpu_telemetry.csv"
    seg.write_bytes(b"segment file")
    with pytest.raises(cli.ScopeViolation, match="段号命名空间"):
        cli.assert_scope_clean()
    assert seg.read_bytes() == b"segment file"


def test_D_cli_refuses_before_loading_model_when_segment_present(cli, monkeypatch):
    from recover_attention.generation import backends as gb

    def explode(*a, **kw):                          # pragma: no cover
        raise AssertionError("模型不得在被拒绝的调用里加载")

    monkeypatch.setattr(gb, "_RealBackendBase", explode)
    seg = cli.PROD / "h1_e3_seg007_partial.npz"
    seg.write_bytes(b"leaked")
    with pytest.raises(cli.ScopeViolation, match="段号命名空间"):
        cli.main(["--anchor", "--benchmark"])
    assert seg.read_bytes() == b"leaked"
    assert not cli.TELEMETRY_OUT.exists()


def test_D_forbidden_paths_still_covers_the_four_fixed_names(cli):
    names = {p.name for p in cli.forbidden_paths()}
    assert names == set(cli.FORBIDDEN_NAMES)
    for name in cli.FORBIDDEN_NAMES:
        target = cli.PROD / name
        target.mkdir() if name == "h1_e3_block_hidden" else target.write_bytes(b"x")
        with pytest.raises(cli.ScopeViolation):
            cli.assert_scope_clean()
        target.rmdir() if name == "h1_e3_block_hidden" else target.unlink()
    cli.assert_scope_clean()                        # 清干净后应通过
