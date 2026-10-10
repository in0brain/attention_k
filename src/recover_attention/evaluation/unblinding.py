"""§2.2 一次性揭盲：由可审计记录强制，而非承诺。

预注册的价值全在"跑之前写死、跑之后不改"。若分析可以反复跑到满意为止,前面所有冻结
都失效 —— 而这一点靠自觉是守不住的,必须留痕。

两件事:
  freeze manifest  生成产物在分析前冻结:记录数、hidden 数、run fingerprint、内容 sha256。
                   分析时重算并比对,不一致 → 说明产物在冻结后被动过 → 停。
  unblinding log   每次分析调用都追加一条记录(时间由调用方传入、config hash、输入 freeze
                   hash、输出 hash)。第一次为 confirmatory;之后必须显式标 rerun 并写理由,
                   且不得替换已发布的 confirmatory 结果。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FREEZE_SCHEMA = "stage0_freeze_manifest_v1"
LOG_SCHEMA = "stage0_unblinding_log_v1"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_freeze_manifest(feature_dir: Path, arms: dict[str, dict]) -> dict:
    """在分析**之前**冻结生成产物。

    arms = {"h1": {"records": Path, "hidden": Path, "run_fingerprint": Path}, ...}
    """
    out: dict[str, Any] = {"schema_version": FREEZE_SCHEMA, "feature_dir": str(feature_dir),
                           "arms": {}}
    for arm, paths in arms.items():
        rec, hid = Path(paths["records"]), Path(paths["hidden"])
        if not rec.exists() or not hid.exists():
            raise FileNotFoundError(f"{arm}: missing {rec if not rec.exists() else hid}")
        n_records = sum(1 for line in open(rec, encoding="utf-8") if line.strip())
        fp_path = paths.get("run_fingerprint")
        run_fp = None
        if fp_path and Path(fp_path).exists():
            run_fp = json.loads(Path(fp_path).read_text(encoding="utf-8")).get("run_hash")
        out["arms"][arm] = {
            "records_path": str(rec), "records_sha256": _sha_file(rec), "n_records": n_records,
            "hidden_path": str(hid), "hidden_sha256": _sha_file(hid),
            "run_hash": run_fp,
        }
    out["freeze_hash"] = hashlib.sha256(
        json.dumps(out["arms"], sort_keys=True).encode("utf-8")).hexdigest()
    return out


def verify_freeze(manifest: dict) -> dict:
    """分析时重算并比对。产物在冻结后被动过 → 结果不可信,必须停。"""
    drift = []
    for arm, rec in manifest["arms"].items():
        for kind in ("records", "hidden"):
            p = Path(rec[f"{kind}_path"])
            if not p.exists():
                drift.append({"arm": arm, "kind": kind, "issue": "missing"})
                continue
            cur = _sha_file(p)
            if cur != rec[f"{kind}_sha256"]:
                drift.append({"arm": arm, "kind": kind, "issue": "sha256 changed",
                              "frozen": rec[f"{kind}_sha256"][:16], "current": cur[:16]})
    return {"ok": not drift, "drift": drift,
            "meaning": ("generation artifacts changed after the freeze — the analysis would no "
                        "longer be the preregistered one" if drift else "artifacts unchanged")}


def record_unblinding(log_path: Path, *, freeze_hash: str, config_hash: str,
                      timestamp: str, is_rerun: bool = False, reason: str | None = None) -> dict:
    """追加一条揭盲记录。返回本次的角色（confirmatory / rerun_N）。

    第一次自动记为 confirmatory。之后必须显式 is_rerun=True 且给出 reason ——
    这不阻止重跑(有时确有正当理由),但**让重跑无法隐形**。
    """
    log_path = Path(log_path)
    prior: list[dict] = []
    if log_path.exists():
        prior = [json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip()]

    if prior and not is_rerun:
        raise ValueError(
            f"analysis already unblinded {len(prior)} time(s) (first at "
            f"{prior[0].get('timestamp')}). §2.2 allows ONE confirmatory unblinding. "
            f"To run again you must pass is_rerun=True with a reason; the rerun will be logged "
            f"and must not replace the published confirmatory result.")
    if is_rerun and not reason:
        raise ValueError("a rerun must state a reason (§2.2)")

    role = "confirmatory" if not prior else f"rerun_{len(prior)}"
    entry = {"schema_version": LOG_SCHEMA, "role": role, "timestamp": timestamp,
             "freeze_hash": freeze_hash, "config_hash": config_hash,
             "is_rerun": bool(prior), "reason": reason,
             "n_prior_unblindings": len(prior)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def confirmatory_entry(log_path: Path) -> dict | None:
    """取第一次（confirmatory）的记录 —— 它是唯一可以作为论文结果的那次。"""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    for line in open(log_path, encoding="utf-8"):
        if line.strip():
            return json.loads(line)
    return None
