r"""MCQ 输出基线覆盖对照：B3 只读"所选选项文本" vs 读**模型可见输出全文**。

### 为什么要做

冻结的 MCQ 映射把 B3 定为 `semantic_output_text`（被选中的选项文本），
并把 `raw_completion` 列入 `FORBIDDEN_IN_BLOCKS`。
这个定义是按 **Qwen 只输出字母**的行为制定的 —— Qwen 的 `raw_completion`
长度中位数 **1 字符**，B3 覆盖已足够。

迁移到 Llama 后没有补:Llama 的 `raw_completion` 长度中位数 **886 字符**
（一整段解释），而 B3 仍只读 **31.5 字符**的选项文本 ——
**100%** 的记录里模型输出比基线读到的多。

于是 Llama MCQ 的 `Δ_H = +0.0596` **不能**被解释为
"隐藏状态提供了超出完整输出所含信息的增量" —— 基线**根本没看过那段解释**。
M1b 只比较已纳入的五块，**发现不了漏掉的文本**。

### 本脚本做什么

在**同一样本、同一划分、同一协议**下，只把 B3 的文本源换掉，跑 Path A：

    frozen          B3 = semantic_output_text                       （复现对照）
    full_output     B3 = raw_completion                              （模型可见输出全文）
    option_plus     B3 = semantic_output_text + "\n" + raw_completion
                    （**选项文本与输出全文联合编码** —— ⚠️ **不是**"严格超集"：
                      拼接后词表、IDF、行 L2 归一化都变，不得要求性能只升不降）

Llama 是待查的那格。Qwen 侧只作**参照**，⚠️ **不是阴性对照** ——
它的 `raw_completion` 就是个字母，换成 full_output 等于**删掉选项语义**，
那一格变化大**不说明入口有 bug**。

### 边界

    · **不改冻结实现**，不写任何冻结产物；新条件**单独报告**
    · 不重新生成、不重新提取隐藏态、不动 H
    · `raw_completion` 是**模型自己的输出**，不含 gold / 正确性字段 ——
      仍逐条断言禁用字段没有进入文本
    · 只跑 Path A。Path B/E3 的硬门不碰
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import conditional_increment as ci             # noqa: E402
from recover_attention.evaluation import v3_adapter_mcq as amcq       # noqa: E402
from recover_attention.evaluation import v3_analysis_paths as paths   # noqa: E402
from recover_attention.evaluation import v3_analyze as va            # noqa: E402
from recover_attention.evaluation.v3_blocks import BlockSpec         # noqa: E402
from recover_attention.evaluation import v3_config as vcfg        # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

PROD = {"qwen": ROOT / "outputs/stage1/production",
        "llama": ROOT / "outputs/stage1_llama_v3/production"}
#: 第一轮（修复前代码）的六格汇总。**只读参照，任何新运行都不得写它。**
FIRST_RUN_SUMMARY = (ROOT / "outputs/stage1/posthoc_readout_probes"
                     / "p3_mcq_output_baseline.json")
#: 新运行一律落在**独立运行目录**下，一次运行一个目录，互不覆盖。
RUNS_DIR = ROOT / "outputs/stage1/posthoc_readout_probes/p3_mcq_baseline_runs"

B3_VARIANTS = {
    "frozen": {
        "field": "semantic_output_text",
        "what": "冻结口径：只读被选中的**选项文本**",
    },
    "full_output": {
        "field": "raw_completion",
        "what": "模型**可见输出全文**（Llama 侧即整段解释）",
    },
    "option_plus_output": {
        "field": None,
        "what": "**选项文本与输出全文联合编码**（两段文本拼接后重新拟合 TF-IDF）",
        "caveat": ("⚠️ **不是『严格超集』**。拼接后词表、IDF 与行 L2 归一化都会变，"
                   "所以**不能**据此要求性能只升不降。"
                   "它只是『补齐输出覆盖』这个问题**最贴近**的一格。"),
    },
}
#: 各格的解释定位（写进产物，避免被读成"应该不变 / 应该更强"）
VARIANT_ROLE = {
    "frozen": "复现对照：验证入口没有引入 B3 之外的差异",
    "full_output": ("只读输出全文。⚠️ 对 **Qwen** 而言这会**删掉选项语义**"
                    "（其 raw_completion 就是个字母，如 'B'），"
                    "所以 Qwen 这一格变化大**不代表入口有 bug**。"),
    "option_plus_output": ("保留选项文本、同时补上输出全文 —— "
                           "**这是『输出覆盖不足』问题最该看的一格**。"),
}
#: 各臂的 **B4 数值完整性状态**。⚠️ 省略即 `path_markers()` 的默认值
#: `compromised` —— 那是 **Qwen/H1** 的既有裁定，**Llama 不得继承**（fail-closed）。
NUMERIC_STATUS_BY_MODEL = {
    "qwen": paths.NUMERIC_COMPROMISED,     # H1 侧既有裁定，原样沿用
    "llama": paths.NUMERIC_NOT_AUDITED,    # 与 run_stage1_llama_ac_analysis.py 同一裁定
}
#: marker 的名字是**冻结登记名**（`h1_b4_*`），说的是 **H1 的 B4**；
#: 本脚本跑的是 **MCQ**。两者的适用范围必须写清，不得互相冒充。
NUMERIC_SCOPE_NOTE = {
    "marker_is_h1_scoped": (
        "`path_markers()` 返回的 marker 与解释边界是**为 H1 的 B4 登记的**"
        "（H1 的 8 项 logit-derived F5 字段）。本脚本跑的是 **MCQ 臂**，"
        "marker 名字里的 `h1_b4_` 是**冻结登记名**，"
        "**不得**读成『本次 MCQ 的 B4 已被审计』。"),
    "mcq_b4_not_separately_audited": (
        "⚠️ **两臂的 MCQ B4 数值完整性都没有单独审计。** "
        "本轮只换 B3 的文本源，**没有**触碰 B4，也**没有**为 MCQ 的 B4 新增任何审计证据。"),
    "per_model": {
        "qwen": ("沿用既有裁定 `compromised`（来自 H1 侧）。"
                 "它**不是**本次 MCQ B4 的审计结论。"),
        "llama": ("`not_audited` —— **fail-closed，不继承 Qwen 的 compromised**，"
                  "也**不因**落盘字段都是有限值就宣布通过。"
                  "与 `run_stage1_llama_ac_analysis.py` 的登记一致。"),
    },
    "unchanged_by_this_run": "本轮**不解除、也不加重**任何既有的数值完整性限制。",
}
#: 绝不允许出现在文本里的字段（沿用冻结禁用表，去掉本次显式启用的 raw_completion）
STILL_FORBIDDEN = tuple(f for f in amcq.FORBIDDEN_IN_BLOCKS if f != "raw_completion")


def _b3_text(rec: dict, variant: str) -> str:
    if variant == "frozen":
        return str(rec.get("semantic_output_text") or "")
    if variant == "full_output":
        return str(rec.get("raw_completion") or "")
    return (str(rec.get("semantic_output_text") or "") + "\n"
            + str(rec.get("raw_completion") or ""))


def load(model: str, variant: str) -> dict:
    recs = [json.loads(x) for x in
            (PROD[model] / "mcq_completion_records.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    used = amcq.filter_analyzable(recs)
    with np.load(PROD[model] / "mcq_hidden.npz") as z:
        hidden = {k: z[k] for k in z.files}
    b = amcq.build_mcq_arm(used, hidden)          # 冻结 adapter，原样调用
    # 只替换 B3 的文本源，其余块**一个字节不动**
    texts = [_b3_text(r, variant) for r in used]
    if any(not t.strip() for t in texts):
        n = sum(1 for t in texts if not t.strip())
        raise RuntimeError(f"{model}/{variant}: {n} 条 B3 文本为空 —— 停，不静默填空串")
    blocks = []
    for blk in b["blocks"]:
        if blk.name == "B3":
            blocks.append(BlockSpec(name="B3", kind="text", text=texts))
        else:
            blocks.append(blk)
    b["blocks"] = blocks
    b["sample_ids"] = np.asarray([str(r[amcq.ID_FIELD]) for r in used])
    b["_records"] = used
    b["_b3_texts"] = texts
    return b


def _rel(p: Path) -> str:
    """仓库内路径写相对路径；仓库外（自检用的临时目录）如实写绝对路径。"""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _fold_map(y, groups, cfg):
    """外层 fold 映射。

    ⚠️ **这是确定性重算，不是运行期记录。** `run_repeat_for_path` 调用的就是同一个
    表达式、同一组 (y, groups, seed)，`StratifiedGroupKFold` 不依赖全局 RNG，
    因此重算与运行期逐元素相同；但它**仍然是重算**，不得称为"运行时落的日志"。
    重算结果会与逐折审计的 `n_test` 对账，对不上就停。
    """
    y = np.asarray(y)
    per_repeat = []
    for r in range(cfg.n_repeats):
        folds = ci.stratified_grouped_folds(
            y, groups, n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r),
            max_seed_retries=cfg.outer_max_seed_retries)
        fold_of = np.full(y.size, -1, dtype=int)
        for f, (_train, test) in enumerate(folds):
            fold_of[test] = f
        if int((fold_of < 0).sum()):
            raise RuntimeError(f"repeat {r}: 有样本未落入任何 test fold")
        per_repeat.append(fold_of)
    meta = {
        "provenance": ("确定性重算（与 run_repeat_for_path 同一表达式、同一 seed），"
                       "**不是**运行期记录"),
        "expression": ("ci.stratified_grouped_folds(y, groups, "
                       "n_splits=cfg.n_outer_folds, seed=cfg.fold_seed(r), "
                       "max_seed_retries=cfg.outer_max_seed_retries)"),
        "fold_seeds": [int(cfg.fold_seed(r)) for r in range(cfg.n_repeats)],
        "n_outer_folds": int(cfg.n_outer_folds),
        "checked_against": "每个 repeat 的 audits[f]['n_test']",
    }
    return per_repeat, meta


def _plan_spec(groups, cfg) -> dict:
    """bootstrap 抽样计划的完整决定式（`_summarise` 不回传 plan 对象）。

    计划只由 (唯一 group 列表, b_batch, b_per_batch, batch_seeds) 决定，与 B3 的
    文本口径无关 —— 所以各条件用的是**同一个抽样计划**，每格内部也是配对的。
    """
    uniq = np.unique(np.asarray(groups).astype(str))
    return {
        "b_batch": int(cfg.b_batch), "b_per_batch": int(cfg.b_per_batch),
        "b_total": int(cfg.b_total),
        "batch_seeds": [int(cfg.batch_seed(j)) for j in range(cfg.b_batch)],
        "n_uniq_groups": int(uniq.size),
        "uniq_groups_sha256": hashlib.sha256(
            "\u0000".join(uniq.tolist()).encode("utf-8")).hexdigest()[:16],
        "pairing": ("同一 draw 内 S* 与 S*+H 用同一组重抽样行 → **每格内部配对**；"
                    "抽样计划与 B3 口径无关 → **跨条件用的是同一计划**。"
                    "⚠️ 但**跨条件差值的区间本脚本没有计算**，不得称已检验过降幅。"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # ⚠️ **没有默认值** —— 防止一条命令无差别重跑六格。必须显式点名要跑哪几格。
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(PROD))
    ap.add_argument("--variants", nargs="+", required=True, choices=list(B3_VARIANTS))
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="独立运行目录；省略则按 UTC 时间戳 + 代码指纹自动生成")
    ap.add_argument("--note", default="", help="写进产物的一句话运行说明")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    code_fp = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.run_dir or (RUNS_DIR / f"run_{stamp}_{code_fp}")).resolve()
    # ── 不覆盖：既不许写第一轮汇总，也不许写进任何非空目录 ────────────────
    if run_dir == FIRST_RUN_SUMMARY.parent.resolve():
        raise SystemExit("拒绝：run-dir 指向第一轮产物目录，会覆盖既有汇总")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"拒绝：{run_dir} 非空 —— 新运行必须用独立空目录，不覆盖旧产物")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "summary.json"

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    t0 = time.perf_counter()
    out = {
        "schema": "p3_mcq_output_baseline_check_v2",
        "run_id": run_dir.name,
        "run_started_utc": stamp,
        "note": args.note,
        "cells_requested": {"models": list(args.models),
                            "variants": list(args.variants)},
        "relation_to_first_run": (
            "这是**新的独立运行**，产物只落在本运行目录。"
            "⚠️ 新运行**只能提供新证据**，**不能**补称第一轮（修复前代码）的运行日志 —— "
            "第一轮的线程 / 库版本 / 机器状态本来就没有留下记录。"
            f"第一轮汇总只作只读参照：{FIRST_RUN_SUMMARY.relative_to(ROOT).as_posix()}"),
        "question": ("MCQ 的输出侧基线只读『所选选项文本』；Llama 的回答是整段解释。"
                     "把 B3 换成**模型可见输出全文**之后，Δ_H 是否还保持？"),
        "why": ("冻结的 B3 定义按 Qwen『只输出字母』制定（raw_completion 中位 1 字符）；"
                "Llama 的 raw_completion 中位 886 字符，B3 仍只读 31.5 字符的选项文本，"
                "**100%** 的记录里输出比基线读到的多。"
                "因此原 Δ_H **不能**解释为『超出完整输出所含信息』。"),
        "boundaries": [
            "**不改冻结实现**，不写任何冻结产物；新条件**单独报告**",
            "不重新生成、不重新提取隐藏态、不动 H",
            "只跑 Path A；Path B / E3 的硬门不碰",
            "这是**事后敏感性分析**，不是预注册条件",
            "结论只对**本次 TF-IDF 编码的输出基线**成立；这是输出侧基线的一个下界，不是上界",
        ],
        "b3_variants": {k: v for k, v in B3_VARIANTS.items() if k in args.variants},
        "variant_roles": {k: v for k, v in VARIANT_ROLE.items() if k in args.variants},
        # ── 运行环境与输入指纹：**开跑之前**记下来，不事后补 ────────────────
        "runtime": {
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS")},
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
            "recorded_before_compute": True,
        },
        "input_fingerprints": {
            f"{m}_{f}": hashlib.sha256((PROD[m] / f).read_bytes()).hexdigest()[:16]
            for m in args.models
            for f in ("mcq_completion_records.jsonl", "mcq_hidden.npz")},
        "code_fingerprint": code_fp,
        "numeric_integrity_scope": NUMERIC_SCOPE_NOTE,
        "frozen_fields_sha256": hashlib.sha256(
            json.dumps(vcfg.FROZEN_FIELDS, sort_keys=True, ensure_ascii=False,
                       default=str).encode("utf-8")).hexdigest()[:16],
        "results": {},
    }

    def flush(done: bool) -> None:
        """**每格跑完就写**，不等全部跑完才落一次盘。"""
        out_path.write_text(
            json.dumps({**out, "partial": not done,
                        "elapsed_secs": round(time.perf_counter() - t0, 1)},
                       ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8")

    flush(False)                              # 开跑之前先把条件与指纹落盘

    for model in args.models:
        out["results"][model] = {}
        for variant in args.variants:
            b = load(model, variant)
            y, groups, blocks = b["y"], b["groups"], b["blocks"]
            texts = b["_b3_texts"]
            lens = sorted(len(t) for t in texts)
            # 禁用字段仍不得出现在文本里（raw_completion 本轮显式启用，其余照禁）
            leaked = []
            for rec, t in zip(b["_records"], texts):
                for f in STILL_FORBIDDEN:
                    v = rec.get(f)
                    if isinstance(v, str) and v and v in t and f != "semantic_output_text":
                        leaked.append((rec[amcq.ID_FIELD], f))
            if leaked:
                raise RuntimeError(f"{model}/{variant}: 禁用字段进入 B3 文本 {leaked[:3]}")
            t1 = time.perf_counter()
            sh = va._summarise(y, groups, blocks, cfg, paths.PATH_A, estimand="E1")

            # ── 逐样本 OOF：**缺就停**，不再静默成功 ───────────────────────
            reps_oof = sh.get("_oof_per_repeat") or []
            if len(reps_oof) != int(cfg.n_repeats):
                raise RuntimeError(
                    f"{model}/{variant}: 拿到 {len(reps_oof)} 组逐样本 OOF，"
                    f"要求 {cfg.n_repeats} 组 —— **停**。没有 OOF 就无法独立复算 "
                    "AUROC 与区间，不允许当成功返回。")
            # ⚠️ 必须核**全部八列**（S* / S*+H / B1..B5 / H），不是只看两条汇总列 ——
            #    只查 S* 与 S*+H 时，删掉 H 列、给 B4 植入 NaN、把 S* 换成 (n,1)
            #    三种坏产物都会静默通过（第二十四轮的反例）。
            _, names_sh = paths.names_for_path(paths.PATH_A)
            want_cols = ["S*", "S*+H", *names_sh]
            n_y = int(np.asarray(y).size)
            for i, o in enumerate(reps_oof):
                got_cols = sorted(map(str, o))
                if got_cols != sorted(want_cols):
                    raise RuntimeError(
                        f"{model}/{variant} r{i}: OOF 列集合不符 —— "
                        f"缺 {sorted(set(want_cols) - set(got_cols))}、"
                        f"多 {sorted(set(got_cols) - set(want_cols))}；要求恰为 {want_cols}")
                for k in want_cols:
                    arr = np.asarray(o[k], dtype=float)
                    if arr.shape != (n_y,):
                        raise RuntimeError(
                            f"{model}/{variant} r{i}/{k}: OOF 形状 {arr.shape}，"
                            f"要求严格 ({n_y},) —— 二维或行数不符一律停")
                    if not np.all(np.isfinite(arr)):
                        bad = int((~np.isfinite(arr)).sum())
                        raise RuntimeError(
                            f"{model}/{variant} r{i}/{k}: OOF 有 {bad} 个非有限值 —— 停")

            # ── 折映射：确定性重算 + 与逐折审计对账 ────────────────────────
            fold_of, fold_meta = _fold_map(y, groups, cfg)
            reps_full = sh.get("_repeats") or []
            # ⚠️ **先要求审计存在，再对账。** 原来写成 `if want and got != want`，
            #    没有审计时直接跳过 —— 删掉整个 `_repeats`、或清空某个 repeat 的
            #    审计，都会静默通过（第二十四轮的反例）。缺审计即**停**。
            if len(reps_full) != int(cfg.n_repeats):
                raise RuntimeError(
                    f"{model}/{variant}: 拿到 {len(reps_full)} 组 repeat 审计，"
                    f"要求 {cfg.n_repeats} 组 —— 停，没有逐折审计就无法对账折映射")
            for i, fo in enumerate(fold_of):
                aud = reps_full[i].get("audits")
                if not isinstance(aud, list) or len(aud) != int(cfg.n_outer_folds):
                    raise RuntimeError(
                        f"{model}/{variant} r{i}: 逐折审计有 "
                        f"{0 if not isinstance(aud, list) else len(aud)} 条，"
                        f"要求 {cfg.n_outer_folds} 条 —— 停")
                missing = [k for k in ("n_test", "n_train", "chosen_C_outer")
                           if any(k not in a for a in aud)]
                if missing:
                    raise RuntimeError(
                        f"{model}/{variant} r{i}: 逐折审计缺字段 {missing} —— 停")
                got = [int((fo == f).sum()) for f in range(cfg.n_outer_folds)]
                want = [int(a["n_test"]) for a in aud]
                if got != want:
                    raise RuntimeError(
                        f"{model}/{variant} r{i}: 重算折大小 {got} 与运行期审计 "
                        f"{want} 不一致 —— 停，折映射不可信")

            oof_path = run_dir / "oof" / f"oof_{model}_{variant}.npz"
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                oof_path,
                sample_id=np.asarray(b["sample_ids"]).astype(str),
                group_id=np.asarray(groups).astype(str),
                y=np.asarray(y),
                b3_text_sha256=np.asarray(
                    [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts]),
                **{f"r{i}_fold_of": fo for i, fo in enumerate(fold_of)},
                **{f"r{i}_{k}": np.asarray(o[k], dtype=float)
                   for i, o in enumerate(reps_oof) for k in o})
            oof_sha = hashlib.sha256(oof_path.read_bytes()).hexdigest()[:16]

            # ── 完整审计：§6 全部必报项 + 逐折 cache 审计，一个不筛 ────────
            r = va._strip_internal(sh)
            audit_path = run_dir / "audit" / f"audit_{model}_{variant}.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps({
                "cell": f"{model}/{variant}",
                "run_id": run_dir.name,
                "code_fingerprint": code_fp,
                "summary_full": r,               # §6 必报项**全量**，不做子集筛选
                "path_markers": paths.path_markers(
                    paths.PATH_A, numeric_status=NUMERIC_STATUS_BY_MODEL[model]),
                "numeric_integrity_scope": {**NUMERIC_SCOPE_NOTE,
                                            "this_model": model,
                                            "status_used": NUMERIC_STATUS_BY_MODEL[model]},
                "per_repeat_fold_audits": [
                    {"repeat": int(rp.get("repeat", i)), "audits": rp.get("audits")}
                    for i, rp in enumerate(reps_full)],
                "fold_map": fold_meta,
                "bootstrap_plan_spec": _plan_spec(groups, cfg),
                "oof_artifact": _rel(oof_path),
                "oof_sha256": oof_sha,
            }, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")

            out["results"][model][variant] = {
                "role": VARIANT_ROLE[variant],
                "oof_artifact": _rel(oof_path),
                "oof_sha256": oof_sha,
                "audit_artifact": _rel(audit_path),
                "n_repeats_oof": len(reps_oof),
                "fold_map_provenance": fold_meta["provenance"],
                "numeric_integrity_status": NUMERIC_STATUS_BY_MODEL[model],
                "b3_field": B3_VARIANTS[variant]["what"],
                "b3_len": {"median": lens[len(lens) // 2], "min": lens[0],
                           "max": lens[-1]},
                "n": int(np.asarray(y).size), "n_positive": int(np.asarray(y).sum()),
                "auroc": r["auroc"], "delta_h_point": r["delta_h_point"],
                "delta_h_ci": r["delta_h_ci"],
                "precision": r.get("precision"),
                "m1b": {k: r[k] for k in r
                        if k.startswith("m1b") and isinstance(r[k], dict)},
                "bootstrap_budget": r.get("bootstrap_budget"),
                "secs": round(time.perf_counter() - t1, 1),
            }
            a = r["auroc"]
            print(f"{model:>6} / {variant:<18} B3中位{lens[len(lens) // 2]:>5}字  "
                  f"S*={a['S*']:.6f} H={a['H']:.6f} S*+H={a['S*+H']:.6f}  "
                  f"Δ_H={r['delta_h_point']:+.6f} "
                  f"[{r['delta_h_ci']['ci_lo']:+.6f}, {r['delta_h_ci']['ci_hi']:+.6f}]"
                  f"  {round(time.perf_counter() - t1, 1)}s", flush=True)
            flush(False)                        # **每格跑完立刻落盘**

    out["total_secs"] = round(time.perf_counter() - t0, 1)
    flush(True)
    print(f"\n[out] {out_path}  用时 {out['total_secs']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
