"""Stage 1 v3 confirmatory 顶层入口。

**默认 `--dry-run`：只跑 synthetic fixture，不读真实 finalized 产物、不解盲。**

正式路径是**一次性** `--execute-all`：`analyze()` 一次算完两臂 × 三路径 ×
全部 estimand，输出**唯一**综合结果文件 `stage1_confirmatory_result.json`。
`--arm/--estimand` 只服务 `--dry-run`。

Path B 的唯一实现是 **tail-stabilized 单次前向**；旧 full-BF16 shadow 是
superseded_before_unblinding / engineering_historical_evidence_only /
not_joined_with_labels，**不接 label**。E3 固定读 canonical `outputs/stage1/repool/`。

`--execute-all` 的调用顺序（**审计顺序，不可调换**）：

    1  冻结链与输入 SHA 校验
    2  readiness.ready_for_unblinding == true 且真实结果目标与 unblinding log 均不存在
    3  record_unblinding                       ← **解盲日志先于任何 label 载入**
    4  analyze_factory()：此刻才把含 label 字段的 records 读进内存
    5  analyze()：首次读取 label 并计算性能
    6  写分析结果

即：解盲日志必须先于任何 label 载入、feature-label association 或性能计算落盘。
第 4 步被推迟到第 3 步之后，是为了让这句话**字面成立** —— 构造回调会加载两臂
completion records，而那些记录本身带 label 字段。
若记录后分析失败，**日志必须保留**并另走 recovery，不得假装尚未解盲。

`--dry-run` 与不带任何标志的默认调用**都不得**调用 `record_unblinding`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_grouping as grouping           # noqa: E402
from recover_attention.evaluation import v3_analysis_paths as vpaths      # noqa: E402
from recover_attention.evaluation import v3_analyze as vanalyze           # noqa: E402
from recover_attention.evaluation import v3_readiness as vreadiness       # noqa: E402
from recover_attention.evaluation import v3_shadow_overlay as voverlay    # noqa: E402
from recover_attention.evaluation import v3_synthetic_fixture as fixture    # noqa: E402
from recover_attention.evaluation import v3_unblinding_boundary as ub       # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig        # noqa: E402
from recover_attention.evaluation.v3_freeze import check_v3_frozen         # noqa: E402

PROD = ROOT / "outputs/stage1/production"
READINESS_V1 = ROOT / "outputs/stage1/readiness/stage1_confirmatory_readiness.json"
#: v2 由 checked-in 的 `v3_readiness` builder 确定性生成；v1 保留不覆盖。
READINESS = ROOT / "outputs/stage1/readiness/stage1_confirmatory_readiness_v2.json"
#: **已被取代**：旧 full-BF16 shadow 只保留为工程史，不接 label。
SHADOW_DIR = ROOT / "outputs/stage1/shadow"
#: Path B 的正式实现来源。
TAIL_DIR = ROOT / "outputs/stage1/tail_stabilized"
#: E3 的 canonical 输入 —— 独立 estimand，**不随 Path B 改变**。
REPOOL_DIR = ROOT / "outputs/stage1/repool"
RESULT_DIR = ROOT / "outputs/stage1/confirmatory"
UNBLIND_DIR = ROOT / "outputs/stage1/unblinding"
UNBLIND_LOG = UNBLIND_DIR / "unblinding_log.jsonl"
FREEZE_MANIFEST = ROOT / "outputs/stage1/readiness/stage1_finalized_freeze_manifest.json"
ANALYSIS_INPUT_MANIFEST = (
    ROOT / "outputs/stage1/readiness/stage1_analysis_input_freeze_manifest.json")
READINESS_V3 = ROOT / "outputs/stage1/readiness/stage1_confirmatory_readiness_v3.json"
ENTRY_SOURCE = Path(__file__).resolve()
FINALIZED_SIX = ub.FINALIZED_SIX
#: 一次性解盲的**唯一**综合结果文件。
RESULT_NAME = "stage1_confirmatory_result.json"


class ExecuteRefused(RuntimeError):
    """fail closed：在加载 label 或计算任何性能量**之前**拒绝。"""


class ReadinessBuilderNotWired(NotImplementedError):
    """保留供旧测试引用；builder 已入库，正常路径不再抛它。"""


def recompute_readiness(readiness_path: Path, prod_dir: Path) -> dict:
    """**checked-in readiness v3 重算入口**。

    从产物现场重算，再与磁盘 JSON 逐项比对 —— 入口**不信任磁盘上的布尔值**。
    任何人手工把 `ready_for_unblinding` 改成 true 都会在比对处被抓出来。
    v2 文档保留不覆盖，但**执行路径只认 v3**。
    """
    return vreadiness.recompute_and_compare_v3(
        readiness_path,
        prod_dir=prod_dir,
        tail_dir=TAIL_DIR,
        shadow_dir=SHADOW_DIR,
        repool_dir=REPOOL_DIR,
        result_dir=RESULT_DIR,
        unblind_log=UNBLIND_LOG,
        freeze_manifest_path=FREEZE_MANIFEST,
        analysis_input_manifest_path=ANALYSIS_INPUT_MANIFEST,
        entry_source_path=ENTRY_SOURCE)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preflight_execute(*, prod_dir: Path = PROD, readiness_path: Path = READINESS_V3,
                      result_dir: Path = RESULT_DIR, unblind_log: Path = UNBLIND_LOG,
                      freeze_manifest_path: Path = FREEZE_MANIFEST,
                      analysis_input_manifest_path: Path = ANALYSIS_INPUT_MANIFEST,
                      tail_dir: Path = TAIL_DIR,
                      repool_dir: Path = REPOOL_DIR) -> dict:
    """`--execute-all` 的 1–2 步。任一不满足 → `ExecuteRefused`，**绝不读 label**。

    第 0 步是 checked-in readiness 重算 —— 它先于任何对磁盘 JSON 的信任。
    """
    # ── 第 0 步：readiness 必须可**重算**，不能只信磁盘上的布尔值 ──────────
    try:
        recompute_readiness(readiness_path, prod_dir)
    except (ReadinessBuilderNotWired, vreadiness.ReadinessMismatch) as exc:
        raise ExecuteRefused(str(exc)) from exc
    fz = check_v3_frozen()
    if not fz.match:
        raise ExecuteRefused("冻结链校验未通过：check_v3_frozen().match 为假")
    if not readiness_path.exists():
        raise ExecuteRefused(f"readiness 报告不存在：{readiness_path}")
    rep = json.loads(readiness_path.read_text(encoding="utf-8"))
    rd = rep.get("readiness", {})
    if not rd.get("ready_for_unblinding"):
        raise ExecuteRefused(
            f"readiness.ready_for_unblinding 为 {rd.get('ready_for_unblinding')!r}；"
            f"blocking_reasons={rd.get('blocking_reasons')}")
    recorded = (rep.get("freeze_chain", {}) or {}).get("finalized_artifacts", {})
    if set(recorded) != set(FINALIZED_SIX):
        raise ExecuteRefused(f"readiness 记录的 finalized 产物集合与预期不符：{sorted(recorded)}")
    drift = []
    for name in FINALIZED_SIX:
        f = prod_dir / name
        if not f.exists():
            drift.append(f"{name}: 文件不存在")
        elif _sha(f) != recorded[name].get("sha256"):
            drift.append(f"{name}: SHA 与 readiness 记录不符")
    if drift:
        raise ExecuteRefused("finalized 产物与 readiness 记录不一致：" + "；".join(drift))
    existing = [str(p) for p in (list(result_dir.glob("*")) if result_dir.exists() else [])]
    if existing:
        raise ExecuteRefused(f"真实结果目标已存在（{len(existing)} 项）；拒绝覆盖")
    if unblind_log.exists():
        raise ExecuteRefused(f"unblinding log 已存在：{unblind_log}；首次解盲只允许一次")
    # 六项 finalized freeze manifest：**input_freeze_hash 的唯一来源**（不是 prereg SHA）
    if not freeze_manifest_path.exists():
        raise ExecuteRefused(f"freeze manifest 不存在：{freeze_manifest_path}")
    man = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    ver = ub.verify_stage1_freeze_manifest(man, prod_dir)
    if not (ver["ok"] and ver["hash_matches"]):
        raise ExecuteRefused("freeze manifest 校验失败：drift={} hash_matches={}".format(
            ver["drift"], ver["hash_matches"]))
    # analysis-input freeze manifest：冻结**实际被分析读取**的 tail F5 与 canonical repool
    if not analysis_input_manifest_path.exists():
        raise ExecuteRefused(
            f"analysis-input freeze manifest 不存在：{analysis_input_manifest_path}；"
            f"六项 finalized 管不住 Path B 实际读的 tail F5 与 E3 实际读的 repool 向量")
    aman = json.loads(analysis_input_manifest_path.read_text(encoding="utf-8"))
    aver = ub.verify_analysis_input_manifest(aman, prod_dir=prod_dir, tail_dir=tail_dir,
                                             repool_dir=repool_dir)
    if not aver["ok"]:
        raise ExecuteRefused(
            "analysis-input manifest 校验失败："
            f"structural={aver['structural_problems'][:3]} drift={aver['drift'][:3]} "
            f"hash_matches={aver['hash_matches']}")
    if aman.get("path_b_overlay_source") != voverlay.OVERLAY_SOURCE_TAIL:
        raise ExecuteRefused("analysis-input manifest 未声明 Path B = tail 实现")
    if aman.get("e3_canonical_source") != "repool":
        raise ExecuteRefused("analysis-input manifest 未声明 E3 canonical = repool")
    return {"input_freeze_hash": man["freeze_hash"],
            "analysis_input_freeze_hash": aman["analysis_input_freeze_hash"],
            "prereg_sha256": fz.current_sha256, "locked_version": fz.locked_version,
            "readiness_sha256": _sha(readiness_path),
            "finalized_sha256": {n: recorded[n]["sha256"] for n in FINALIZED_SIX}}


def run_execute_all(*, cfg: V3AnalysisConfig, analyze_factory, timestamp: str,
                    prod_dir: Path = PROD, readiness_path: Path = READINESS_V3,
                    result_dir: Path = RESULT_DIR, unblind_log: Path = UNBLIND_LOG,
                    freeze_manifest_path: Path = FREEZE_MANIFEST,
                    analysis_input_manifest_path: Path = ANALYSIS_INPUT_MANIFEST,
                    tail_dir: Path = TAIL_DIR, repool_dir: Path = REPOOL_DIR) -> dict:
    """**一次性**执行全部分析并写唯一综合结果文件。

    为什么不再按 arm / estimand 分次：`analyze()` 一次就算**两臂 × 三路径 ×
    全部 estimand**，`--arm/--estimand` 从来只影响文件名与日志；而首次解盲以
    独占创建写 log，第二次调用必然失败。分次调用因此既无实效又会造成
    "只跑了一部分" 的错觉。

    **调用顺序**：preflight → record_unblinding → `analyze_factory()` → analyze() → 写结果。

    收 `analyze_factory` 而不是现成的 `analyze`，是为了让文档里那句
    "日志先于首次读取 label" **字面成立**：构造回调会把两臂 completion records
    读进内存，而那些记录本身带 label 字段。此前 `main()` 先构造再调用，
    等于在写日志之前就把含 label 的记录载入了。现在构造被推迟到日志之后。

    工厂本身抛错时日志**仍然保留** —— 已解盲就是已解盲。这不会把一次解盲浪费在
    可预见的门禁失败上：`preflight_execute` 里的 readiness v3 现场重算已经把
    tail overlay / 旧 shadow supersede / E3 canonical 这几道门跑过一遍了，
    工厂再失败属于意外状态，本就该留痕。
    """
    gate = preflight_execute(prod_dir=prod_dir, readiness_path=readiness_path,
                             result_dir=result_dir, unblind_log=unblind_log,
                             freeze_manifest_path=freeze_manifest_path,
                             analysis_input_manifest_path=analysis_input_manifest_path,
                             tail_dir=tail_dir, repool_dir=repool_dir)
    config_hash = hashlib.sha256(
        json.dumps(cfg.provenance(), sort_keys=True, ensure_ascii=False,
                   default=str).encode("utf-8")).hexdigest()
    # log v2：finalized 六项与 analysis-input **两个** freeze hash 同时记录
    ub.record_stage1_unblinding(
        unblind_log, input_freeze_hash=gate["input_freeze_hash"],
        analysis_input_freeze_hash=gate["analysis_input_freeze_hash"],
        prereg_sha256=gate["prereg_sha256"], readiness_sha256=gate["readiness_sha256"],
        config_hash=config_hash, timestamp=timestamp,
        note="execute-all: 两臂 x 三路径 x 全部 estimand，单次调用")
    try:
        analyze = analyze_factory()              # ← 含 label 的 records 此刻才进内存
        result = analyze()                       # ← 第一次读 label / 算性能
    except Exception as exc:                     # noqa: BLE001
        raise RuntimeError(
            f"解盲后分析失败：{exc}。**unblinding log 保留**，须另走 recovery，"
            f"不得假装尚未解盲") from exc
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / RESULT_NAME
    result = {**result, "execution": {
        "mode": "execute_all_single_shot",
        "scope": "两臂 x 三路径 x 全部 estimand，一次调用",
        "path_b_implementation": voverlay.OVERLAY_SOURCE_TAIL,
        "e3_canonical_source": "repool",
        "legacy_shadow_status": list(voverlay.LEGACY_SHADOW_STATUS),
        "finalized_freeze_hash": gate["input_freeze_hash"],
        "analysis_input_freeze_hash": gate["analysis_input_freeze_hash"],
        "prereg_sha256": gate["prereg_sha256"], "timestamp": timestamp}}
    # 归一化 + 严格 JSON + 原子写：异常或断电都不得留下半文件。
    # 顺序不变 —— 本步仍在 record_stage1_unblinding **之后**。
    write_result_atomic(out, result)
    return {"unblinded": True, "result_path": str(out), "gate": gate}


def json_safe(obj):
    """把 NumPy 标量/数组转成原生类型；非有限值**不在此处静默处理**，交给 allow_nan=False。"""
    import numpy as np
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            # 绝不 str(k) 强转：{1:"a","1":"b"} 会被静默合并成一个键，正式结果不容此损坏。
            raise TypeError(f"结果含非字符串 dict key {bad[:3]}；拒绝写入")
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


class ResultPublishRefused(RuntimeError):
    """发布目标已存在或在落盘瞬间出现 —— 拒绝覆盖。

    到这一步 **unblinding log 已经写好了**：必须走 recovery，不得假装尚未解盲，
    也不得删除或覆盖已有的结果文件。
    """


def write_result_atomic(path: Path, result: dict) -> Path:
    """严格 JSON（allow_nan=False）→ 临时文件 → flush → fsync → **no-clobber 发布**。

    发布用同文件系统的 `os.link`（目标已存在时**原子失败**），成功后删除临时链接。
    **刻意不用 `os.replace`**：它无条件覆盖 —— preflight 只能保证"分析开始前目标
    不存在"，若目标在分析期间出现（并发进程、人工放置、恢复流程），`os.replace`
    会把它悄悄盖掉。confirmatory 结果是一次性产物，不容此损坏。
    """
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(result), ensure_ascii=False, indent=1,
                         allow_nan=False) + "\n"      # NaN/Inf 在此抛错，不落盘
    if path.exists():
        raise ResultPublishRefused(
            f"结果目标已存在，拒绝覆盖：{path}\n"
            f"  若此时 unblinding log 已写入，**已解盲就是已解盲** —— 保留日志，"
            f"另走 recovery，不得覆盖既有结果")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, str(path))                   # ← 原子 no-clobber 发布
        except FileExistsError as exc:
            raise ResultPublishRefused(
                f"结果目标在落盘期间出现，拒绝覆盖：{path}\n"
                f"  unblinding log 若已写入则**保留**，须另走 recovery") from exc
        except (OSError, AttributeError, NotImplementedError) as exc:
            raise ResultPublishRefused(
                f"os.link 不可用或失败，fail-closed（**不退回 os.replace**）："
                f"{type(exc).__name__}: {exc}") from exc
    except BaseException:
        Path(tmp).unlink(missing_ok=True)             # 不留半文件，也不留临时链接
        raise
    Path(tmp).unlink(missing_ok=True)                 # 发布成功后删除临时链接
    return path


def run_dry(*, arm: str, estimand: str, cfg: V3AnalysisConfig | None = None) -> dict:
    """synthetic 路径：**臂内 E1** 走与 `--execute` 相同的计算路径，输入全部合成。

    ⚠️ 两臂 D **不在** end-to-end 覆盖范围内 —— 只有 `contrast_math_smoke`
    （用任意 OOF 分数验证算术与判读；冻结的 output-only O 尚未接线）。
    不读真实产物、不写结果、不调用任何 unblinding 记录。
    `--arm both` 才运行 D 的数学 smoke。
    """
    cfg = cfg or fixture.synthetic_cfg()
    out = {"mode": "dry-run", "synthetic": True, "arm": arm, "estimand": estimand,
           "reads_real_finalized": False, "writes_results": False,
           "record_unblinding_called": False,
           "b_total": int(cfg.b_batch * cfg.b_per_batch), "n_repeats": int(cfg.n_repeats)}
    if estimand in ("E2", "E3") and arm == "mcq":
        return {**out, **grouping.not_applicable(arm, estimand)}
    if estimand == "E3":
        recs, _hid = fixture.make_h1()
        return {**out, "e3": grouping.build_e3_groups(recs, V3AnalysisConfig(), arm="h1")}
    if estimand == "E2":
        recs, _hid = fixture.make_h1()
        return {**out, "e2": grouping.build_e2_groups(recs, arm="h1")}
    arms = ["h1", "mcq"] if arm == "both" else [arm]
    out["arms"] = {a: fixture.run_arm_end_to_end(a, cfg, seed=i) for i, a in enumerate(arms)}
    out["end_to_end_scope"] = "per-arm E1 only（D 不在 end-to-end 覆盖范围内）"
    if arm == "both":                                  # **只有 --arm both 运行 D**
        out["contrast_math_smoke"] = fixture.contrast_math_smoke(cfg)
    else:
        out["contrast_math_smoke"] = None
        out["contrast_note"] = "D 只在 --arm both 时运行"
    return out


def load_real_inputs(prod_dir: Path = PROD) -> dict:
    """加载真实分析输入。**只读 records / hidden / shadow / repool，不读 label 做任何计算。**"""
    import numpy as np
    from recover_attention.evaluation import v3_production as vprod

    def _recs(arm: str) -> list[dict]:
        raw = (prod_dir / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
        rows = [json.loads(x) for x in raw if x.strip()]
        return [r for r in rows if vprod.requires_hidden(arm, r)]

    def _hidden(arm: str) -> dict:
        with np.load(prod_dir / f"{arm}_hidden.npz") as z:
            return {k: np.asarray(z[k]) for k in z.files}

    h1, mcq = _recs("h1"), _recs("mcq")
    # Path B = **tail**；旧 full-BF16 shadow 是 superseded_before_unblinding，
    # 这里连读都不读 —— 同时分析两者会重新形成结果树。
    voverlay.assert_legacy_shadow_superseded(SHADOW_DIR)
    ov = voverlay.load_tail_overlay(TAIL_DIR, prod_dir, REPOOL_DIR)
    # E3 固定 canonical repool：独立 estimand，不随 Path B 改变。
    e3 = voverlay.load_e3_blocks(REPOOL_DIR, h1, V3AnalysisConfig())
    return {"h1_records": h1, "mcq_records": mcq,
            "h1_hidden": _hidden("h1"), "mcq_hidden": _hidden("mcq"),
            "shadow_overlay": ov, "e3_blocks": e3}


def build_real_analyze(cfg: V3AnalysisConfig, prod_dir: Path = PROD):
    """构造真实 analyze 回调。**构造不读 label**；调用它才第一次读。"""
    vpaths.assert_frozen_paths(vpaths.FROZEN_PATHS)
    return vanalyze.make_analyze_callback(cfg=cfg, **load_real_inputs(prod_dir))


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False：否则 argparse 会把旧的 `--execute` 当成 `--execute-all`
    # 的唯一前缀而**静默接受** —— 正式解盲入口不容许这种"手滑也能跑"。
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 allow_abbrev=False)
    ap.add_argument("--arm", choices=["h1", "mcq", "both"], required=True,
                    help="**只影响 --dry-run**；正式路径是一次性 --execute-all")
    ap.add_argument("--estimand", choices=["E1", "E2", "E3"], default="E1",
                    help="**只影响 --dry-run**")
    ap.add_argument("--dry-run", action="store_true",
                    help="默认行为；只跑 synthetic fixture，不解盲")
    ap.add_argument("--execute-all", dest="execute_all", action="store_true",
                    help="真实 confirmatory run，**一次性**跑完两臂 x 三路径 x 全部 estimand")
    ap.add_argument("--confirm-unblinding", action="store_true",
                    help="**一次性解盲**。没有它，--execute-all 只跑闸、不构造回调、不解盲。")
    ap.add_argument("--timestamp", default=None,
                    help="ISO8601 解盲时间戳；--confirm-unblinding 时必填")
    args = ap.parse_args(argv)
    if args.confirm_unblinding and not args.execute_all:
        ap.error("--confirm-unblinding 必须与 --execute-all 同时给出")
    if args.confirm_unblinding and not args.timestamp:
        ap.error("--confirm-unblinding 需显式 --timestamp（ISO8601），不自动取当前时间")
    if not args.execute_all:
        # **dry-run 必须用 synthetic 小预算**：production cfg 会让默认路径实跑
        # B_total = 20,000 个 draw —— 那既不是 fixture，也让"默认即安全"名不副实。
        res = run_dry(arm=args.arm, estimand=args.estimand,
                      cfg=fixture.synthetic_cfg())
        print(json.dumps(json_safe(res), ensure_ascii=True, allow_nan=False))
        return 0
    try:
        preflight_execute()
    except ExecuteRefused as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        print("[REFUSED] 在加载 label 或计算任何性能量之前拒绝；未解盲。", file=sys.stderr)
        return 2
    if not args.confirm_unblinding:
        # 闸通过 ≠ 该解盲。构造回调会加载全部 records / hidden / tail / E3，
        # 代价大且没必要；**必须**显式 --confirm-unblinding 才继续。
        print("[EXECUTE] fail-closed 闸通过（readiness v3 现场重算与磁盘一致）。",
              file=sys.stderr)
        print("[EXECUTE] 未加 --confirm-unblinding：**不构造回调、不解盲**，就此停止。",
              file=sys.stderr)
        return 3
    print("[UNBLIND] --confirm-unblinding 已给出。", file=sys.stderr)
    print(f"[UNBLIND] paths={vpaths.FROZEN_PATHS} "
          f"path_b={voverlay.OVERLAY_SOURCE_TAIL}", file=sys.stderr)
    cfg = V3AnalysisConfig()

    def _analyze_factory():
        """**在 record_unblinding 之后**才被调用 —— 含 label 的 records 此刻才进内存。"""
        print("[UNBLIND] 日志已落盘；开始构造真实 analyze 回调……", file=sys.stderr)
        a = build_real_analyze(cfg)
        vpaths.assert_frozen_paths(a.frozen_paths)
        return a

    res = run_execute_all(cfg=cfg, analyze_factory=_analyze_factory,
                          timestamp=args.timestamp)
    print(json.dumps({"unblinded": True, "result_path": res["result_path"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
