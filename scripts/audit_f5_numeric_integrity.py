"""H1 F5 数值完整性审计 —— **只读、performance-blind、不解盲**。

起因：`_teacher_forced` 为**每个** completion token 都写一项 logprob，而
`sequence_logprob_features` / `mention_logprob_features` 只剔除非有限值
（`finite_float`），因此数值异常会被**静默**吸收进聚合量。

受影响 **8 项**（H1 臂；MCQ 用另一套 6 个字段，与此机制无关）：

    sequence-level   f5_completion_token_count / f5_lengthnorm_logprob /
                     f5_completion_perplexity / f5_id_token_ratio
    identifier-pos   f5_id_logprob_mean / f5_id_logprob_min /
                     f5_id_token_entropy_mean / f5_first_id_token_rank

参照（位置计数，恒有限）：`f5_id_token_count`。
口径：B4 共 **14** 项（`F5_FEATURE_NAMES`），上述 8 项是其中一部分，**不是** B4 本身。

`f5_first_id_token_rank` 有**两种**退化，都写不进冻结记录：

    trank[j] = int((row > row[tid]).sum()) + 1
    (a) row[tid] 非有限 → 所有比较 False → rank **精确写成 1**
    (b) row[tid] 有限但同 row 含 NaN competitor → 这些 competitor 不计入 → rank **被低估**

证据分三级，措辞不得互串：

    1a  纯冻结历史事实（只读记录，零外部依赖）
    1b  **当前兼容环境下的确定性重建**（需 tokenizer；**不是**原始 c_ids 的落盘回读）
    2   **当前环境诊断**（需模型前向；生成期逐 token logprob 从未落盘）

本脚本**不**决定修复口径，**不**裁定特征可用性 —— 那是研究者裁决。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention import h1_f5_features as f5f                     # noqa: E402
from recover_attention.conditional_increment import F5_FEATURE_NAMES     # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool          # noqa: E402
from recover_attention.evaluation import v3_production as prod           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig      # noqa: E402

PROD = ROOT / "outputs/stage1/production"
AUDIT = ROOT / "outputs/stage1/audit"

#: 两个相分别落一个产物 —— `--diagnose` **不覆盖** `--frozen` 的结果。
#: 诊断相会重算 1a/1b，因此两份文件的 1a 段应逐字段相同，这本身就是一道交叉校验。
AUDIT_OUT = AUDIT / "f5_numeric_integrity_audit.json"
DIAGNOSTIC_OUT = AUDIT / "f5_numeric_integrity_diagnostic.json"
TELEMETRY_OUT = AUDIT / "f5_audit_gpu_telemetry.csv"

AUDIT_SCHEMA = "stage1_f5_numeric_integrity_audit_v1"
SELECTION_ALGORITHM_VERSION = "length_stratified_equidistant_v1"

FIELDS_SEQUENCE = ("f5_completion_token_count", "f5_lengthnorm_logprob",
                   "f5_completion_perplexity", "f5_id_token_ratio")
FIELDS_IDENTIFIER = ("f5_id_logprob_mean", "f5_id_logprob_min",
                     "f5_id_token_entropy_mean", "f5_first_id_token_rank")
FIELD_REFERENCE = "f5_id_token_count"
FIELDS_AUDITED = FIELDS_SEQUENCE + FIELDS_IDENTIFIER

#: 预先写死的容差 —— 不用浮点相等。
RATIO_REL_TOL = 1e-9
RATIO_ABS_TOL = 0.0
PPL_LOG_ABS_TOL = 1e-9
#: `sequence_logprob_features` 里的指数保护阈值，用于 None 归因。
PPL_EXP_GUARD = 700.0

GROUP_II_SIZE = 64


class ScopeViolation(RuntimeError):
    """越出本轮授权范围。"""


# ------------------------------------------------------------------ 硬门
# 与 run_stage1_e3_repool.py 同款边界。刻意**复制**而非抽公共模块：
# 本轮授权的新增文件只有本脚本与其测试，加公共模块属范围外改动。
def assert_outputs_absent(paths) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise ScopeViolation("输出已存在，拒绝覆盖：\n  " + "\n  ".join(existing))


def assert_scope_clean() -> None:
    """本审计**只读**：不得在 production 下留下任何新文件。"""
    leaked = [str(p) for p in sorted(PROD.glob("h1_e3_seg*"))] if PROD.exists() else []
    for name in ("h1_e3_block_hidden", "h1_e3_repool_manifest.jsonl",
                 "h1_e3_repool_fingerprint.json", "h1_e3_repool_report.json"):
        if (PROD / name).exists():
            leaked.append(str(PROD / name))
    if leaked:
        raise ScopeViolation("检测到不应存在的产物：\n  " + "\n  ".join(leaked))


def publish_json_exclusive(path: Path, payload: dict) -> None:
    """只创建、绝不覆盖：O_EXCL 临时文件 → os.link 原子发布；失败 fail-closed。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise ScopeViolation(f"发布目标已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ScopeViolation(f"存在陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except FileExistsError as exc:
        raise ScopeViolation(f"发布目标在写入期间出现，拒绝覆盖：{path}") from exc
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(
            f"os.link 不可用或失败，fail-closed（不退回 os.replace）："
            f"{type(exc).__name__}: {exc}；临时文件保留在 {tmp}") from exc
    os.unlink(tmp)


def start_telemetry(path: Path):
    """模型加载**之前**启动；`"x"` 独占创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("x", encoding="utf-8", newline="")
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,"
         "clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
         "--format=csv", "-l", "15"],
        stdout=fh, stderr=subprocess.DEVNULL)
    return proc, fh


def stop_telemetry(handle) -> None:
    proc, fh = handle
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:                                   # noqa: BLE001
        proc.kill()
    finally:
        fh.close()


# ------------------------------------------------------------------ 载入
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records() -> tuple[list[dict], list[dict]]:
    """返回 (finalized 全量, analyzable 子集)。

    **主审计分母 = analyzable**。finalized 只作 secondary accounting，单独报告，
    绝不与主分母混用 —— non-analyzable 的合法缺失会污染结果。
    """
    raw = (PROD / "h1_completion_records.jsonl").read_bytes().split(b"\n")
    finalized = [json.loads(x) for x in raw if x.strip()]
    analyzable = [r for r in finalized if prod.requires_hidden("h1", r)]
    return finalized, analyzable


def _num(value):
    """只接受真正的有限数值；None / 非有限 / 非数一律返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


# ------------------------------------------------- 类别 1a：纯冻结历史事实
def audit_1a(rows: list[dict], cfg: V3AnalysisConfig) -> dict:
    """只读冻结记录，零外部依赖。恒等式用预先写死的容差。"""
    n = len(rows)
    out: dict = {
        "evidence_class": "1a_frozen_record_arithmetic",
        "evidence_caliber": "纯冻结历史事实：只读记录，零外部依赖",
        "denominator": n,
        "tolerances": {"ratio_rel_tol": RATIO_REL_TOL, "ratio_abs_tol": RATIO_ABS_TOL,
                       "perplexity_log_abs_tol": PPL_LOG_ABS_TOL,
                       "note": "perplexity 取对数后比较，不直接比 exp 输出（避免放大误差）"},
    }

    # -- 恒等式 (1) ratio x count == id_token_count -------------------------
    ok = bad = skip = 0
    bad_ids: list[str] = []
    for r in rows:
        ratio, count = _num(r.get("f5_id_token_ratio")), _num(r.get("f5_completion_token_count"))
        idc = _num(r.get(FIELD_REFERENCE))
        if ratio is None or count is None or idc is None:
            skip += 1
            continue
        if math.isclose(ratio * count, idc, rel_tol=RATIO_REL_TOL, abs_tol=RATIO_ABS_TOL):
            ok += 1
        else:
            bad += 1
            if len(bad_ids) < 50:
                bad_ids.append(repool.derive_unit_id(r))
    out["identity_ratio_times_count_eq_id_count"] = {
        "pass": ok, "fail": bad, "not_evaluable_missing_value": skip,
        "fail_unit_ids_first_50": bad_ids}

    # -- 恒等式 (2) log(ppl) == -lengthnorm ---------------------------------
    ok = bad = skip = 0
    bad_ids = []
    for r in rows:
        ppl, ln = _num(r.get("f5_completion_perplexity")), _num(r.get("f5_lengthnorm_logprob"))
        if ppl is None or ln is None or ppl <= 0.0:
            skip += 1
            continue
        if math.isclose(math.log(ppl), -ln, rel_tol=0.0, abs_tol=PPL_LOG_ABS_TOL):
            ok += 1
        else:
            bad += 1
            if len(bad_ids) < 50:
                bad_ids.append(repool.derive_unit_id(r))
    out["identity_log_perplexity_eq_neg_lengthnorm"] = {
        "pass": ok, "fail": bad, "not_evaluable_missing_or_nonpositive": skip,
        "fail_unit_ids_first_50": bad_ids}

    # -- 恒等式 (3) id_token_count == len(id_token_positions_local) --------
    ok = bad = 0
    bad_ids = []
    for r in rows:
        want = len(repool.union_positions(r))
        got = r.get(FIELD_REFERENCE)
        if isinstance(got, int) and not isinstance(got, bool) and got == want:
            ok += 1
        else:
            bad += 1
            if len(bad_ids) < 50:
                bad_ids.append(repool.derive_unit_id(r))
    out["identity_id_token_count_eq_n_positions"] = {
        "pass": ok, "fail": bad, "fail_unit_ids_first_50": bad_ids,
        "comparison": "exact int equality"}

    # -- 红旗 (4) ratio > 1 -------------------------------------------------
    over = [repool.derive_unit_id(r) for r in rows
            if (_num(r.get("f5_id_token_ratio")) or 0.0) > 1.0]
    out["red_flag_ratio_gt_1"] = {
        "count": len(over), "unit_ids_first_50": over[:50],
        "interpretation": "违反 intended ratio 语义的严重红旗",
        "verdict": "NOT_DECIDED_BY_SCRIPT —— 是否仍可用于 confirmatory 由研究者裁决"}

    # -- (5)(6) 聚合量为 None，且存在 identifier 位置 -----------------------
    for field, key in (("f5_id_logprob_mean", "no_finite_identifier_logprob_entered_aggregate"),
                       ("f5_id_token_entropy_mean", "no_finite_identifier_entropy_entered_aggregate")):
        hit = [repool.derive_unit_id(r) for r in rows
               if r.get(field) is None and len(repool.union_positions(r)) > 0]
        out[key] = {
            "field": field, "count": len(hit), "unit_ids_first_50": hit[:50],
            "provable_from_frozen_record": (
                f"{field} is None → 没有有限的 identifier-token 数值进入聚合"),
            "NOT_provable_from_frozen_record": (
                "不能仅凭冻结记录断言原始值『全部非有限』—— "
                "token_logprobs.get(index) 缺键返回 None，会产生完全相同的结果。"
                "结合代码路径与位置在界内等证据后，可以说『与全部非有限一致』，"
                "但那已不是纯 1a 历史事实。"),
        }

    # -- (7) rank == 1 的边际频率（只看边际，不与 label 联表）---------------
    ranks = [r.get("f5_first_id_token_rank") for r in rows]
    finite_ranks = [int(x) for x in ranks
                    if isinstance(x, int) and not isinstance(x, bool)]
    eq1 = sum(1 for x in finite_ranks if x == 1)
    hist: dict[str, int] = {}
    for x in finite_ranks:
        bucket = "1" if x == 1 else ("2-5" if x <= 5 else ("6-50" if x <= 50 else ">50"))
        hist[bucket] = hist.get(bucket, 0) + 1
    out["red_flag_rank_eq_1_marginal"] = {
        "n_with_rank": len(finite_ranks), "n_rank_eq_1": eq1,
        "share": (eq1 / len(finite_ranks)) if finite_ranks else None,
        "bucket_histogram": hist,
        "caveat": ("单条不可判：rank=1 既可能是真 top-1，也可能来自 target 非有限。"
                   "只能作分布层面的红旗；未与 label 联表。"),
    }

    # -- (8) perplexity is None 的两因归因 ---------------------------------
    a = b = c = 0
    for r in rows:
        if r.get("f5_completion_perplexity") is not None:
            continue
        ln = r.get("f5_lengthnorm_logprob")
        if ln is None:
            a += 1
        elif _num(ln) is not None and float(ln) >= PPL_EXP_GUARD:
            b += 1
        else:
            c += 1
    out["perplexity_none_attribution"] = {
        "total_none": a + b + c,
        "cause_a_lengthnorm_is_none_no_finite_logprob": a,
        "cause_b_exponent_guard_lengthnorm_ge_700": b,
        "cause_c_unexplained": c,
        "guard_threshold": PPL_EXP_GUARD,
        "note": "两因语义完全不同，分别计数；不得混成普通恒等式失败",
    }

    # -- (9) 逐字段 None / 缺失 / 非有限 ------------------------------------
    per_field = {}
    for field in FIELDS_AUDITED + (FIELD_REFERENCE,):
        missing = sum(1 for r in rows if field not in r)
        none_ = sum(1 for r in rows if r.get(field) is None) - missing
        nonfinite = sum(1 for r in rows
                        if r.get(field) is not None and _num(r.get(field)) is None
                        and isinstance(r.get(field), (int, float))
                        and not isinstance(r.get(field), bool))
        per_field[field] = {"missing_key": missing, "present_but_none": max(0, none_),
                            "present_but_nonfinite": nonfinite,
                            "denominator": n}
    out["per_field_none_and_missing"] = per_field

    out["observable_lower_bound_caveat"] = (
        "**部分**非有限在冻结聚合值里不可恢复 —— 子集均值与正常均值完全同形。"
        "因此 1a 只给出可观察**下界**：真实受影响比例只会更高，不会更低。")
    return out


# ---------------------------- 类别 1b：当前兼容环境下的确定性重建
def audit_1b(rows: list[dict], tokenizer, tok_fp: dict) -> dict:
    deltas, pairs = [], []
    for r in rows:
        n_tok = len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"])
        stored = _num(r.get("f5_completion_token_count"))
        if stored is None:
            continue
        d = n_tok - int(stored)
        deltas.append(d)
        pairs.append((n_tok, d, len(repool.union_positions(r))))

    deltas_sorted = sorted(deltas)

    def q(p):
        return deltas_sorted[min(len(deltas_sorted) - 1, int(p * len(deltas_sorted)))]

    by_len: dict[str, dict] = {}
    for n_tok, d, _n_id in pairs:
        bucket = ("<=50" if n_tok <= 50 else "51-100" if n_tok <= 100
                  else "101-200" if n_tok <= 200 else ">200")
        e = by_len.setdefault(bucket, {"n": 0, "sum_delta": 0, "sum_ntok": 0})
        e["n"] += 1
        e["sum_delta"] += d
        e["sum_ntok"] += n_tok
    for e in by_len.values():
        e["mean_delta"] = e.pop("sum_delta") / e["n"]
        e["mean_n_tok"] = e.pop("sum_ntok") / e["n"]

    return {
        "evidence_class": "1b_deterministic_reconstruction_in_current_compatible_env",
        "evidence_caliber": (
            "冻结 completion 文本在**当前、与冻结指纹兼容的 tokenizer 环境**下的"
            "确定性重建。**不是**原始 c_ids 的落盘回读，**不得**升格为原始运行日志。"),
        "fingerprint_limitation": (
            "tokenizer_fingerprint 的 payload 只有 class / name_or_path / vocab_size / "
            "special_token_ids，**不哈希 tokenizer 文件内容**。"
            "因此『指纹一致』不等于『分词器字节一致』。"),
        "tokenizer_fingerprint_current": dict(tok_fp),
        "n_evaluated": len(deltas),
        "delta_n_tok_minus_stored_count": {
            "mean": sum(deltas) / len(deltas) if deltas else None,
            "min": deltas_sorted[0] if deltas else None,
            "p50": q(0.5) if deltas else None,
            "p90": q(0.9) if deltas else None,
            "max": deltas_sorted[-1] if deltas else None,
            "n_zero_delta": sum(1 for d in deltas if d == 0),
            "n_negative_delta": sum(1 for d in deltas if d < 0),
        },
        "delta_by_length_bucket": by_len,
        "corroborating_evidence_only": [
            "8,664/8,664 prompt_hash 精确相等（tokenizer-only 校验）",
            "64 条 union-pooling 与冻结 h1_unit_hidden 精确一致",
        ],
    }


# ------------------------------------- 组 II：长度分层确定性选取
def select_group_ii(rows: list[dict], tokenizer, exclude: set[str],
                    size: int = GROUP_II_SIZE) -> dict:
    """按 (n_tok, unit_id) 排序，取**含首尾**的 size 个等距 rank，与组 I 去重后补位。"""
    keyed = []
    for r in rows:
        n_tok = len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"])
        keyed.append((n_tok, repool.derive_unit_id(r)))
    keyed.sort()
    total = len(keyed)

    ranks = [round(i * (total - 1) / (size - 1)) for i in range(size)]
    seen_rank, chosen, fills = set(), [], 0
    for rk in ranks:
        while rk < total and (rk in seen_rank or keyed[rk][1] in exclude):
            rk += 1
            fills += 1
        if rk >= total:
            continue
        seen_rank.add(rk)
        chosen.append(keyed[rk][1])

    idx = 0
    while len(chosen) < size and idx < total:
        if idx not in seen_rank and keyed[idx][1] not in exclude:
            seen_rank.add(idx)
            chosen.append(keyed[idx][1])
            fills += 1
        idx += 1

    n_tok_of = {u: n for n, u in keyed}
    return {
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "sort_key": "(n_tok, unit_id)",
        "requested_size": size,
        "n_selected": len(chosen),
        "n_fill_steps": fills,
        "population_size": total,
        "unit_ids": sorted(chosen),
        "selection_set_sha256": repool.anchor_set_sha256(chosen),
        "n_tok_min": min((n_tok_of[u] for u in chosen), default=None),
        "n_tok_max": max((n_tok_of[u] for u in chosen), default=None),
        "excluded_group_i_size": len(exclude),
    }


# --------------------------------------------- 类别 2：当前环境诊断
def _row_class(values) -> str:
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if bool(finite.all()):
        return "finite"
    return "all_nonfinite" if not bool(finite.any()) else "partial_nonfinite"


def _scalar_class(v) -> str:
    x = float(v)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "posinf" if x > 0 else "neginf"
    return "finite"


def diagnose_unit(backend, row: dict) -> dict:
    """单条 trace 的当前环境诊断。

    **不持久化** completion 文本、raw logits 或逐 vocab 数值 ——
    只保留每 row 的有限性类别、计数与必要 unit_id。
    """
    import numpy as np
    torch = backend.ctx["torch"]

    unit_id = repool.derive_unit_id(row)
    prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
    tf = backend._teacher_forced(prompt, row["completion"])

    logits = tf["logits"]
    lp = torch.log_softmax(logits, dim=-1)
    plen, c_ids = tf["prompt_len"], tf["completion_ids"]
    n_tok = len(c_ids)

    logit_rows = {"finite": 0, "partial_nonfinite": 0, "all_nonfinite": 0}
    lsm_rows = {"finite": 0, "partial_nonfinite": 0, "all_nonfinite": 0}
    target_cls = {"finite": 0, "nan": 0, "posinf": 0, "neginf": 0}
    anomalous_rows: set[int] = set()

    for j, tid in enumerate(c_ids):
        i = plen + j - 1
        lg = logits[i].detach().float().cpu().numpy()
        ls = lp[i].detach().float().cpu().numpy()
        lc, sc = _row_class(lg), _row_class(ls)
        logit_rows[lc] += 1
        lsm_rows[sc] += 1
        tc = _scalar_class(float(ls[int(tid)]))
        target_cls[tc] += 1
        if lc != "finite" or sc != "finite" or tc != "finite":
            anomalous_rows.add(j)
        del lg, ls

    id_pos = repool.union_positions(row)
    id_on_anomalous = sorted(p for p in id_pos if p in anomalous_rows)
    first = id_pos[0] if id_pos else None

    # rank=1 的三分解（当前环境）
    rank_state = None
    if first is not None:
        i = plen + first - 1
        ls = lp[i].detach().float().cpu().numpy()
        row_all_finite = bool(np.isfinite(ls).all())
        target_finite = bool(np.isfinite(ls[int(c_ids[first])]))
        if row_all_finite:
            rank_state = "row_all_finite"
        elif not target_finite:
            rank_state = "target_nonfinite"
        else:
            rank_state = "target_finite_row_has_nonfinite"
        del ls

    # 当前环境重算两组字段（复用**冻结的**函数，保证语义一致）
    comp_lps = [tf["token_logprobs"][j] for j in sorted(tf["token_logprobs"])]
    cur_seq = f5f.sequence_logprob_features(comp_lps, id_token_count=len(id_pos))
    cur_id = (f5f.mention_logprob_features(
        tf["token_logprobs"], tf["token_entropies"], tf["first_token_ranks"], id_pos)
        if id_pos else {})

    def cmp_fields(names):
        out = {}
        for name in names:
            frozen = row.get(name)
            current = cur_seq.get(name, cur_id.get(name))
            if frozen is None or current is None:
                same = (frozen is None and current is None)          # None 精确匹配
            elif isinstance(frozen, int) and isinstance(current, int) \
                    and not isinstance(frozen, bool):
                same = frozen == current                              # 整数精确比较
            else:
                fa, ca = _num(frozen), _num(current)
                same = (fa is not None and ca is not None
                        and math.isclose(fa, ca, rel_tol=1e-9, abs_tol=1e-12))
            out[name] = {"frozen": frozen, "current": current, "match": bool(same)}
        return out

    frozen_count = _num(row.get("f5_completion_token_count"))
    cur_clean = _num(cur_seq.get("f5_completion_token_count"))
    if cur_clean is not None and frozen_count is not None and cur_clean == frozen_count:
        abc = "A"
    elif cur_clean is not None and cur_clean == n_tok and cur_clean != frozen_count:
        abc = "B"
    else:
        abc = "C"

    del tf, logits, lp
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "unit_id": unit_id,
        "n_completion_tokens_current": n_tok,
        "raw_logits_row_classes": logit_rows,
        "log_softmax_row_classes": lsm_rows,
        "target_token_logprob_classes": target_cls,
        "n_anomalous_rows": len(anomalous_rows),
        "n_identifier_positions": len(id_pos),
        "identifier_positions_on_anomalous_rows": id_on_anomalous,
        "first_identifier_rank_state": rank_state,
        "first_identifier_rank_current": cur_id.get("f5_first_id_token_rank"),
        "sequence_level_comparison": cmp_fields(FIELDS_SEQUENCE),
        "identifier_position_comparison": cmp_fields(FIELDS_IDENTIFIER),
        "abc_class": abc,
    }


ABC_MEANING = {
    "A": ("current len(clean) == frozen count → 当前重跑复现冻结的有限-token 计数，"
          "与生成期过滤解释一致；**不证明**原始逐-token 状态"),
    "B": ("current len(clean) == current n_tok != frozen count → 当前环境未复现冻结计数；"
          "来源仍未确定，可能是运行期数值漂移或其他路径差异"),
    "C": ("两者都不满足 → 既不复现冻结计数、也不是全有限；"
          "**停止作统一解释**，逐例报告状态"),
}


def summarize_group(name: str, results: list[dict]) -> dict:
    abc: dict[str, list[str]] = {"A": [], "B": [], "C": []}
    rank_states: dict[str, int] = {}
    field_mismatch: dict[str, int] = {}
    for r in results:
        abc[r["abc_class"]].append(r["unit_id"])
        st = r.get("first_identifier_rank_state")
        if st:
            rank_states[st] = rank_states.get(st, 0) + 1
        for group in ("sequence_level_comparison", "identifier_position_comparison"):
            for field, cmp_ in r[group].items():
                if not cmp_["match"]:
                    field_mismatch[field] = field_mismatch.get(field, 0) + 1
    return {
        "group": name,
        "n_units": len(results),
        "abc_counts": {k: len(v) for k, v in abc.items()},
        "abc_unit_ids": abc,
        "abc_meaning": ABC_MEANING,
        "first_identifier_rank_state_counts": rank_states,
        "field_mismatch_counts_vs_frozen": field_mismatch,
        "n_units_with_any_anomalous_row": sum(1 for r in results if r["n_anomalous_rows"]),
        "n_units_with_identifier_on_anomalous_row":
            sum(1 for r in results if r["identifier_positions_on_anomalous_rows"]),
    }


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", action="store_true",
                    help="只跑类别 1a + 1b（不加载模型）")
    ap.add_argument("--diagnose", action="store_true",
                    help="1a + 1b + 类别 2（加载模型，两组各 64 条）")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    args = ap.parse_args(argv)
    if not (args.frozen or args.diagnose):
        ap.error("至少选择 --frozen 或 --diagnose")

    assert_scope_clean()
    out_path = DIAGNOSTIC_OUT if args.diagnose else AUDIT_OUT
    targets = [out_path] + ([TELEMETRY_OUT] if args.diagnose else [])
    assert_outputs_absent(targets)

    cfg = V3AnalysisConfig()
    finalized, analyzable = load_records()
    print(f"[scope] finalized {len(finalized):,} ｜ **analyzable（主分母）** {len(analyzable):,}")

    payload: dict = {
        "schema": AUDIT_SCHEMA,
        "mode": "diagnose" if args.diagnose else "frozen_only",
        "arm": "h1",
        "scope": {
            "primary_denominator": "analyzable records entering confirmatory B4",
            "n_analyzable": len(analyzable),
            "n_finalized_secondary_accounting_only": len(finalized),
            "warning": ("两个分母**不得混用** —— non-analyzable 的合法缺失会污染结果。"
                        "finalized 只作 secondary accounting，单独报告。"),
            "fields_audited_sequence_level": list(FIELDS_SEQUENCE),
            "fields_audited_identifier_position": list(FIELDS_IDENTIFIER),
            "field_reference_position_count": FIELD_REFERENCE,
            "b4_total_fields": len(F5_FEATURE_NAMES),
            "b4_note": ("B4 共 14 项（F5_FEATURE_NAMES）；本次 8 项是其中**一部分**，"
                        "**不是** B4 本身"),
            "arm_scope_note": ("H1-only：MCQ 用另一套 6 个字段，与本机制零重叠。"
                               "MCQ 的 29/10560 非有限走 ForwardNonFiniteError 显式停机 + "
                               "technical exclusion，**不是**静默过滤，两件事不得混为一谈"),
        },
        "source_shas": {
            "h1_completion_records.jsonl": file_sha256(PROD / "h1_completion_records.jsonl"),
            "h1_run_fingerprint.json": file_sha256(PROD / "h1_run_fingerprint.json"),
        },
        "boundaries": {
            "performance_blind": True,
            "joined_with_label": False,
            "modified_frozen_records": False,
            "unblinded": False,
            "decides_remediation": False,
            "decides_feature_usability": False,
        },
        "rank_degeneracy_mechanism": {
            "code": "trank[j] = int((row > row[tid]).sum()) + 1",
            "degeneracy_a": "row[tid] 非有限 → 所有比较 False → rank **精确写成 1**",
            "degeneracy_b": ("row[tid] 有限但同 row 含非有限 competitor → "
                             "这些 competitor 不计入 → rank **被低估**"),
            "validity_condition": ("f5_first_id_token_rank 只有在整个 log_softmax row "
                                   "**全有限**时才可称数值有效"),
            "direction_claim_allowed": ("异常把编码推向『更小、更像高排名』的方向"),
            "direction_claim_forbidden": ("**不得**声称它必然把 fabrication classifier "
                                          "朝某个预测方向推 —— 那需要 feature-label / "
                                          "performance 分析，本审计不做"),
        },
    }

    print("[1a   ] 冻结记录算术审计（主分母 = analyzable）...")
    payload["class_1a_primary_analyzable"] = audit_1a(analyzable, cfg)
    payload["class_1a_secondary_finalized"] = audit_1a(finalized, cfg)
    payload["class_1a_secondary_finalized"]["denominator_note"] = (
        "secondary accounting only —— 含 non-analyzable 的合法缺失，**不得**与主分母混用")

    telemetry = None
    backend = None
    try:
        from recover_attention.generation import backends as gb
        from recover_attention.generation import fingerprint as fp

        if args.diagnose:
            telemetry = start_telemetry(TELEMETRY_OUT)     # 模型加载**之前**
            print(f"[tele ] pid={telemetry[0].pid} -> {TELEMETRY_OUT.name}")
            backend = gb._RealBackendBase(args.model_path)
            tokenizer = backend.ctx["tokenizer"]
        else:
            comp = gb.import_transformers_components()
            tokenizer = comp["AutoTokenizer"].from_pretrained(
                args.model_path, local_files_only=True, trust_remote_code=False)

        tok_fp = fp.tokenizer_fingerprint(tokenizer)
        print("[1b   ] 当前兼容环境下的确定性重建...")
        payload["class_1b_primary_analyzable"] = audit_1b(analyzable, tokenizer, tok_fp)

        group_i = repool.select_anchor_units(analyzable, cfg)
        group_ii = select_group_ii(analyzable, tokenizer, set(group_i))
        payload["sample_groups"] = {
            "group_i_block_stratified_anchor": {
                "unit_ids": group_i, "n": len(group_i),
                "selection_set_sha256": repool.anchor_set_sha256(group_i),
                "purpose": "与既有 64/64 精确锚点身份连续"},
            "group_ii_length_stratified": {**group_ii,
                                           "purpose": "覆盖完整长度范围 (n_tok min..max)"},
            "union_size_after_dedup": len(set(group_i) | set(group_ii["unit_ids"])),
            "representativeness_caveat": (
                "两组**都不能**估计原始运行期的真实发生率，"
                "只够检查**当前**异常是否明显依赖长度"),
        }

        if args.diagnose:
            by_uid = {repool.derive_unit_id(r): r for r in analyzable}
            cls2: dict = {
                "evidence_class": "2_current_environment_diagnostic",
                "evidence_caliber": (
                    "**当前环境诊断**。生成期的逐 token logprob **从未落盘**，"
                    "因此本节无论结果如何，都只能说明『当前环境、当前指纹下重跑会得到什么』，"
                    "**不能**冒充原始运行时的逐 token 日志，"
                    "也**不能**单独证明生成当刻发生了什么。"),
                "persistence_policy": (
                    "不持久化 completion 文本、raw logits 或逐 vocab 数值；"
                    "只保留每 row 的有限性类别、计数与必要 unit_id"),
                "comparison_policy": ("整数与状态精确比较；浮点 rel_tol=1e-9 / abs_tol=1e-12；"
                                      "None 精确匹配"),
            }
            for name, ids in (("group_i", group_i), ("group_ii", group_ii["unit_ids"])):
                t0 = time.time()
                res = [diagnose_unit(backend, by_uid[u]) for u in ids if u in by_uid]
                cls2[name] = {**summarize_group(name, res),
                              "elapsed_seconds": round(time.time() - t0, 2),
                              "per_unit": res}
                print(f"[cls2 ] {name}: {cls2[name]['abc_counts']} "
                      f"({cls2[name]['elapsed_seconds']}s)")
            merged = cls2["group_i"]["per_unit"] + cls2["group_ii"]["per_unit"]
            seen, uniq = set(), []
            for r in merged:
                if r["unit_id"] not in seen:
                    seen.add(r["unit_id"])
                    uniq.append(r)
            cls2["union_deduped_summary"] = {
                **summarize_group("union_deduped", uniq),
                "note": "两组已分别报告；此处为去重后 union 的汇总，**不替代**分组报告"}
            del cls2["union_deduped_summary"]["abc_meaning"]
            payload["class_2_current_environment_diagnostic"] = cls2

        payload["researcher_rulings_required"] = {
            "note": "以下三问**均由研究者裁决**，本脚本不预选任何一项",
            "q1_sequence_level_four_fields": (
                "sequence-level 四项的语义与数值完整性 —— 是否仍可按冻结定义使用"),
            "q2_identifier_position_four_fields": (
                "identifier-position 四项的数值完整性 —— 尤其 f5_first_id_token_rank "
                "的退化在冻结记录里无签名，是否构成不可接受的方向性偏倚"),
            "q3_whole_b4": (
                "整个 H1 B4（14 项）是否可按冻结定义进入 confirmatory，"
                "还是需要独立的 protocol-deviation 决策"),
            "remediation_options_not_selected": [
                "(a) 重算并回写 —— 需 9,264 条全部重前向，改的是 B4 的输入",
                "(b) 从 B4 剔除 —— 改变冻结的特征集合",
                "(c) 如实报告并保留 —— 在论文中说明该口径",
            ],
        }
        publish_json_exclusive(out_path, payload)
        print(f"[out  ] {out_path}")
    finally:
        if telemetry is not None:
            stop_telemetry(telemetry)
        backend = None
        assert_scope_clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
