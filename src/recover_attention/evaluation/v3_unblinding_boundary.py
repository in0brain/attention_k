"""Stage 1 解盲边界：finalized freeze manifest + Stage 1 专用 unblinding log。

**外层专用 builder/verifier**：不改 `unblinding.build_freeze_manifest` 的签名，
也不改 Stage 0 产物语义 —— 那是另一套血统，此处独立实现 Stage 1 的六项冻结。

manifest hash 口径（**只覆盖这三项，排序后**）：

    sha256( json([{relative_path, bytes, sha256}, …] sorted by relative_path) )

首次**原子创建**；若已存在，只允许**完全一致**的 no-op，**禁止覆盖**。

Stage 1 unblinding log 用**独立 schema** `stage1_v3_unblinding_log_v1`，
**不继续使用 stage0 schema**；首次记录以**独占创建**（`x` 模式）+ `flush` + `os.fsync`
写入，并记录 `input_freeze_hash` / `prereg_sha256` / `readiness_sha256` / `config_hash`。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

MANIFEST_SCHEMA = "stage1_finalized_freeze_manifest_v1"
#: manifest hash 的口径字面值（锁定，防止悄悄换算法后 manifest 仍"合法"）。
HASH_CALIBER = "sha256 over sorted [{relative_path, bytes, sha256}]"
STAGE1_LOG_SCHEMA = "stage1_v3_unblinding_log_v1"
#: 同时记录 finalized 与 analysis-input 两个 freeze hash 时用 v2。
STAGE1_LOG_SCHEMA_V2 = "stage1_v3_unblinding_log_v2"
#: Stage 1 冻结的六项 finalized 产物（两臂各 records / hidden / finalization_report）。
FINALIZED_SIX = (
    "h1_completion_records.jsonl", "h1_hidden.npz", "h1_finalization_report.json",
    "mcq_completion_records.jsonl", "mcq_hidden.npz", "mcq_finalization_report.json",
)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_entries(prod_dir: Path, names=FINALIZED_SIX) -> list[dict]:
    """按 relative_path 排序的 `{relative_path, bytes, sha256}` 列表。"""
    out = []
    for n in names:
        p = Path(prod_dir) / n
        if not p.exists():
            raise FileNotFoundError(f"finalized 产物缺失：{p}")
        out.append({"relative_path": n, "bytes": p.stat().st_size, "sha256": _sha_file(p)})
    return sorted(out, key=lambda e: e["relative_path"])


def freeze_hash_of(entries: list[dict]) -> str:
    """**只**对排序后的 relative_path / bytes / sha256 求 hash，不含路径前缀或时间。"""
    payload = [{"relative_path": e["relative_path"], "bytes": e["bytes"],
                "sha256": e["sha256"]} for e in entries]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def build_stage1_freeze_manifest(prod_dir: Path, names=FINALIZED_SIX) -> dict:
    entries = manifest_entries(prod_dir, names)
    return {"schema_version": MANIFEST_SCHEMA, "n_artifacts": len(entries),
            "hash_caliber": HASH_CALIBER,
            "entries": entries, "freeze_hash": freeze_hash_of(entries)}


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_stage1_freeze_manifest(path: Path, prod_dir: Path, names=FINALIZED_SIX) -> dict:
    """首次原子创建；已存在则只允许**完全一致**的 no-op，**绝不覆盖**。"""
    path = Path(path)
    fresh = build_stage1_freeze_manifest(prod_dir, names)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != fresh:
            diff = sorted(k for k in set(old) | set(fresh) if old.get(k) != fresh.get(k))
            raise ValueError(
                f"freeze manifest 已存在且与当前状态不一致（差异字段 {diff}）；"
                f"**禁止覆盖** —— 产物在冻结后被动过，须人工介入")
        return {**fresh, "written": False, "reason": "已存在且完全一致 —— no-op"}
    _atomic_write_json(path, fresh)
    return {**fresh, "written": True}


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _structural_problems(manifest) -> list[str]:
    """manifest 自身的结构校验。**任何畸形输入都受控返回问题列表，不抛异常。**"""
    p: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest 非 dict"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        p.append(f"schema_version={manifest.get('schema_version')!r} != {MANIFEST_SCHEMA}")
    fh = manifest.get("freeze_hash")
    if not isinstance(fh, str) or not _HEX64.match(fh):
        p.append("freeze_hash 缺失或不是 64 位小写十六进制")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        p.append("entries 缺失或非 list")
        return p
    n_art = manifest.get("n_artifacts")
    # 必须是**非 bool 的精确 int**：6.0 == 6 为真，float 会溜过去。
    if not isinstance(n_art, int) or isinstance(n_art, bool) or n_art != len(FINALIZED_SIX):
        p.append(f"n_artifacts={n_art!r} 必须是精确等于 {len(FINALIZED_SIX)} 的 int")
    if manifest.get("hash_caliber") != HASH_CALIBER:
        p.append(f"hash_caliber={manifest.get('hash_caliber')!r} != {HASH_CALIBER!r}")
    if len(entries) != len(FINALIZED_SIX):
        p.append(f"entries 有 {len(entries)} 条，应为 {len(FINALIZED_SIX)}")
    if isinstance(n_art, int) and not isinstance(n_art, bool) and len(entries) != n_art:
        p.append(f"len(entries)={len(entries)} != n_artifacts={n_art}")
    seen, paths = set(), []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            p.append(f"entries[{i}] 非 dict")
            continue
        rp = e.get("relative_path")
        if not isinstance(rp, str) or not rp:
            p.append(f"entries[{i}].relative_path 缺失或非字符串")
        else:
            paths.append(rp)
            if rp in seen:
                p.append(f"relative_path 重复：{rp}")
            seen.add(rp)
            if (Path(rp).is_absolute() or ".." in Path(rp).parts
                    or "/" in rp or "\\" in rp):
                p.append(f"relative_path 不安全（只允许纯文件名）：{rp!r}")
        b = e.get("bytes")
        if not isinstance(b, int) or isinstance(b, bool) or b < 0:
            p.append(f"entries[{i}].bytes={b!r} 必须是非负整数")
        sh = e.get("sha256")
        if not isinstance(sh, str) or not _HEX64.match(sh):
            p.append(f"entries[{i}].sha256 不是 64 位小写十六进制")
    if paths and set(paths) != set(FINALIZED_SIX):
        missing = sorted(set(FINALIZED_SIX) - set(paths))
        extra = sorted(set(paths) - set(FINALIZED_SIX))
        p.append(f"relative_path 集合不等于冻结六项（缺 {missing}，多 {extra}）")
    return p


def verify_stage1_freeze_manifest(manifest, prod_dir: Path) -> dict:
    """结构校验 + 逐项重算比对。**`ok` 已包含 `hash_matches`**，调用方无须再查第二个布尔。

    任何畸形输入（缺 entries / 类型错 / 坏字段 / 路径不安全 / 条数不符）都**受控**
    返回 `ok=False`，不会抛 KeyError/TypeError。
    """
    structural = _structural_problems(manifest)
    if structural:
        return {"ok": False, "structural_problems": structural, "drift": [],
                "freeze_hash": (manifest or {}).get("freeze_hash")
                if isinstance(manifest, dict) else None,
                "recomputed_freeze_hash": None, "hash_matches": False}

    drift = []
    for e in manifest["entries"]:
        p = Path(prod_dir) / e["relative_path"]
        if not p.exists():
            drift.append({"relative_path": e["relative_path"], "issue": "missing"})
            continue
        if p.stat().st_size != e["bytes"]:
            drift.append({"relative_path": e["relative_path"], "issue": "bytes changed",
                          "frozen": e["bytes"], "current": p.stat().st_size})
            continue
        cur = _sha_file(p)
        if cur != e["sha256"]:
            drift.append({"relative_path": e["relative_path"], "issue": "sha256 changed",
                          "frozen": e["sha256"][:16], "current": cur[:16]})
    recomputed = None
    if not drift:
        try:
            recomputed = freeze_hash_of(manifest_entries(
                prod_dir, tuple(e["relative_path"] for e in manifest["entries"])))
        except OSError as exc:
            drift.append({"relative_path": "*", "issue": f"recompute failed: {exc}"})
    hash_matches = bool(recomputed) and recomputed == manifest["freeze_hash"]
    return {"ok": (not drift) and hash_matches,          # **hash_matches 已并入 ok**
            "structural_problems": [], "drift": drift,
            "freeze_hash": manifest["freeze_hash"], "recomputed_freeze_hash": recomputed,
            "hash_matches": hash_matches}


# ══════════════════════════════════════════════════════════════════════════════
# analysis-input freeze manifest —— 冻结**实际被分析读取**的输入
# ══════════════════════════════════════════════════════════════════════════════
#: 六项 finalized 只覆盖 production；但 Path B 实际读 tail F5，E3 实际读 canonical
#: repool 的 8,664 个 NPZ。不把它们冻住，`input_freeze_hash` 就管不住真正的输入。
ANALYSIS_INPUT_SCHEMA = "stage1_analysis_input_freeze_manifest_v1"
ANALYSIS_INPUT_HASH_CALIBER = (
    "sha256 over sorted [{relative_path, bytes, sha256} | "
    "{relative_path, kind:tree, n_files, tree_digest}]")

TAIL_INPUT_FILES = ("h1_tail_f5.jsonl", "mcq_tail_f5.jsonl",
                    "tail_completion_report.json", "tail_fingerprint.json")
REPOOL_INPUT_FILES = ("e3_repool_manifest.jsonl", "e3_repool_report.json")
#: E3 的 pooled 向量真源 —— 目录整体摘要，逐文件冻结 8,664 条不现实。
REPOOL_INPUT_TREE = "h1_e3_block_hidden"


def _tree_entry(root: Path, rel: str) -> dict:
    d = Path(root) / rel
    if not d.is_dir():
        raise FileNotFoundError(f"analysis input 目录缺失：{d}")
    names = sorted(os.listdir(d))
    h = hashlib.sha256()
    for name in names:
        blob = (d / name).read_bytes()
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(blob).digest())
    return {"relative_path": f"{rel}/", "kind": "tree", "n_files": len(names),
            "tree_digest": h.hexdigest()}


def analysis_input_entries(*, prod_dir: Path, tail_dir: Path,
                           repool_dir: Path) -> list[dict]:
    """按 `relative_path` 排序的条目。文件给 bytes+sha256，目录给 n_files+tree_digest。"""
    out: list[dict] = []
    for prefix, root, names in (("production", prod_dir, FINALIZED_SIX),
                                ("tail_stabilized", tail_dir, TAIL_INPUT_FILES),
                                ("repool", repool_dir, REPOOL_INPUT_FILES)):
        for n in names:
            p = Path(root) / n
            if not p.exists():
                raise FileNotFoundError(f"analysis input 缺失：{p}")
            out.append({"relative_path": f"{prefix}/{n}", "kind": "file",
                        "bytes": p.stat().st_size, "sha256": _sha_file(p)})
    e = _tree_entry(repool_dir, REPOOL_INPUT_TREE)
    out.append({**e, "relative_path": f"repool/{e['relative_path']}"})
    return sorted(out, key=lambda x: x["relative_path"])


def analysis_input_freeze_hash(entries: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def build_analysis_input_manifest(*, prod_dir: Path, tail_dir: Path,
                                  repool_dir: Path) -> dict:
    entries = analysis_input_entries(prod_dir=prod_dir, tail_dir=tail_dir,
                                     repool_dir=repool_dir)
    return {
        "schema_version": ANALYSIS_INPUT_SCHEMA,
        "hash_caliber": ANALYSIS_INPUT_HASH_CALIBER,
        "n_entries": len(entries),
        "path_b_overlay_source": "tail_stabilized_single_forward",
        "e3_canonical_source": "repool",
        "binds": ("finalized 六项 + tail F5/report/fingerprint + canonical repool"
                  "（manifest / report / h1_e3_block_hidden 树摘要）"),
        "entries": entries,
        "analysis_input_freeze_hash": analysis_input_freeze_hash(entries),
    }


def write_analysis_input_manifest(path: Path, *, prod_dir: Path, tail_dir: Path,
                                  repool_dir: Path) -> dict:
    """首次原子创建；已存在则只允许**完全一致**的 no-op，**绝不覆盖**。"""
    path = Path(path)
    fresh = build_analysis_input_manifest(prod_dir=prod_dir, tail_dir=tail_dir,
                                          repool_dir=repool_dir)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != fresh:
            diff = sorted(k for k in set(old) | set(fresh) if old.get(k) != fresh.get(k))
            raise ValueError(
                f"analysis-input manifest 已存在且与现场不一致（差异字段 {diff}）；"
                f"**禁止覆盖** —— 分析输入在冻结后被动过，须人工介入")
        return {**fresh, "written": False, "reason": "已存在且完全一致 —— no-op"}
    _atomic_write_json(path, fresh)
    return {**fresh, "written": True}


def verify_analysis_input_manifest(manifest, *, prod_dir: Path, tail_dir: Path,
                                   repool_dir: Path) -> dict:
    """结构 + 逐项重算比对。畸形输入受控返回 `ok=False`，不抛异常。"""
    if not isinstance(manifest, dict):
        return {"ok": False, "structural_problems": ["manifest 非 dict"], "drift": [],
                "hash_matches": False, "recomputed": None}
    problems = []
    if manifest.get("schema_version") != ANALYSIS_INPUT_SCHEMA:
        problems.append(f"schema_version={manifest.get('schema_version')!r}")
    if manifest.get("hash_caliber") != ANALYSIS_INPUT_HASH_CALIBER:
        problems.append("hash_caliber 不符")
    fh = manifest.get("analysis_input_freeze_hash")
    if not isinstance(fh, str) or not _HEX64.match(fh):
        problems.append("analysis_input_freeze_hash 不是 64 位小写十六进制")
    if not isinstance(manifest.get("entries"), list):
        problems.append("entries 缺失或非 list")
    if problems:
        return {"ok": False, "structural_problems": problems, "drift": [],
                "hash_matches": False, "recomputed": None}
    try:
        fresh = analysis_input_entries(prod_dir=prod_dir, tail_dir=tail_dir,
                                       repool_dir=repool_dir)
    except (OSError, FileNotFoundError) as exc:
        return {"ok": False, "structural_problems": [], "hash_matches": False,
                "recomputed": None, "drift": [{"relative_path": "*",
                                               "issue": f"recompute failed: {exc}"}]}
    old = {e.get("relative_path"): e for e in manifest["entries"]}
    new = {e["relative_path"]: e for e in fresh}
    drift = []
    for rp in sorted(set(old) | set(new)):
        if old.get(rp) != new.get(rp):
            drift.append({"relative_path": rp,
                          "issue": "missing" if rp not in new else
                                   ("unexpected" if rp not in old else "changed")})
    recomputed = analysis_input_freeze_hash(fresh)
    hash_matches = recomputed == manifest["analysis_input_freeze_hash"]
    return {"ok": (not drift) and hash_matches, "structural_problems": [],
            "drift": drift, "hash_matches": hash_matches, "recomputed": recomputed,
            "n_entries": len(fresh)}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 unblinding log（独立 schema；**不用 stage0 那套**）
# ══════════════════════════════════════════════════════════════════════════════
def record_stage1_unblinding(log_path: Path, *, input_freeze_hash: str, prereg_sha256: str,
                             readiness_sha256: str, config_hash: str, timestamp: str,
                             analysis_input_freeze_hash: str | None = None,
                             note: str | None = None) -> dict:
    """首次解盲记录：**独占创建**（`x`）+ flush + fsync（文件与目录）。

    `input_freeze_hash` 是**六项 finalized freeze manifest 的 hash**；
    `prereg_sha256` 单独成字段，**不占** freeze_hash 的位置。
    给出 `analysis_input_freeze_hash` 时 schema 升到 v2，条目同时携带
    `finalized_freeze_hash` 与 `analysis_input_freeze_hash` —— 前者管 production
    六项，后者管**实际被分析读取**的 tail F5 与 canonical repool。
    日志已存在 → 抛错：一次性解盲只允许一次，重跑须另行裁决。
    """
    log_path = Path(log_path)
    if log_path.exists():
        raise FileExistsError(
            f"unblinding log 已存在：{log_path}；§5 的一次性解盲只允许一次，"
            f"重跑须另行裁决并单独记录，不得追加覆盖")
    entry = {"schema_version": STAGE1_LOG_SCHEMA, "role": "confirmatory",
             "timestamp": timestamp, "input_freeze_hash": input_freeze_hash,
             "prereg_sha256": prereg_sha256, "readiness_sha256": readiness_sha256,
             "config_hash": config_hash, "note": note}
    if analysis_input_freeze_hash is not None:
        entry["schema_version"] = STAGE1_LOG_SCHEMA_V2
        entry["finalized_freeze_hash"] = input_freeze_hash
        entry["analysis_input_freeze_hash"] = analysis_input_freeze_hash
    payload = json.dumps(entry, ensure_ascii=False, sort_keys=True,
                         allow_nan=False) + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "x", encoding="utf-8", newline="\n") as f:   # 独占创建
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    # 目录项落盘：POSIX 上有效；**Windows 不支持打开目录句柄**，静默跳过。
    # 文件本身的 flush + fsync 已在上面完成，是本函数的持久化保证。
    try:
        dfd = os.open(str(log_path.parent), os.O_RDONLY)
    except OSError:
        pass
    else:
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)
    return entry
