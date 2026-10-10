r"""G3 模拟指纹解码预算修复的**实际行为**验证（走真实 CLI，读落盘产物）。

不检查构造函数、也不 grep 源码字符串 —— 全部结论来自真实 CLI 的退出码与落盘 JSON。

    ① 分臂 dry-run，读落盘指纹：H1 默认 384 / MCQ 默认 16 / 两臂 CLI 覆盖生效
    ② H1 → MCQ → summarise 模拟贯通，读 g3_summary.json
    ③ 故障注入：把某臂指纹的长度改错，原检查仍须拒绝

**不加载模型。** 模拟产物写独立临时目录；项目 tmp/ 下只留一份小型 JSON 摘要。

用法：
    python scripts/diagnostics/l2c_verify_dry_budget_cli.py [--work-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ENTRY = ROOT / "scripts" / "run_stage1_g3_smoke.py"
OUT = ROOT / "tmp" / "stage1_l2_g3" / "l2c_dry_budget_cli_verification.json"

from recover_attention.evaluation import v3_g3_smoke as g3          # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402


def run_cli(args: list[str]) -> dict:
    cmd = [sys.executable, "-X", "utf8", str(ENTRY), *args]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT), env=env)
    return {"argv": args, "exit_code": p.returncode,
            "stdout_tail": p.stdout.strip().splitlines()[-3:] if p.stdout else [],
            "stderr_tail": p.stderr.strip().splitlines()[-3:] if p.stderr else []}


def budget_of(out_dir: Path, arm: str) -> int | None:
    fp = out_dir / f"{arm}_g3_run_fingerprint.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))["generation_params"]["max_new_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="模拟产物目录；默认用系统临时目录（**不在仓库内**）")
    args = ap.parse_args()
    work = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="l2c_dry_budget_"))
    work.mkdir(parents=True, exist_ok=True)

    cfg = V3AnalysisConfig()
    checks: list[dict] = []
    cases: list[dict] = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ── ① 分臂 dry-run，读落盘指纹 ───────────────────────────────────────
    for label, arm, extra, expect in (
        ("h1_default", "h1", [], cfg.h1_max_new_tokens),
        ("mcq_default", "mcq", [], cfg.mcq_max_new_tokens),
        ("h1_cli_override", "h1", ["--max-new-tokens", "256"], 256),
        ("mcq_cli_override", "mcq", ["--mcq-max-new-tokens", "8"], 8),
    ):
        d = work / label
        r = run_cli(["--arm", arm, "--dry-run", "--out-dir", str(d), *extra])
        got = budget_of(d, arm)
        cases.append({"case": label, "out_dir": str(d), "expected": expect,
                      "landed_max_new_tokens": got, **r})
        check(f"{label}: 落盘指纹 max_new_tokens == {expect}", got == expect,
              {"got": got, "fingerprint": str(d / f"{arm}_g3_run_fingerprint.json")})

    # ── ② H1 → MCQ → summarise 贯通 ─────────────────────────────────────
    chain = work / "chain"
    chain_runs = [run_cli(["--arm", a, "--dry-run", "--out-dir", str(chain)])
                  for a in ("h1", "mcq")]
    chain_runs.append(run_cli(["--summarise", "--out-dir", str(chain)]))
    summary_path = chain / "g3_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    inv = summary.get("backend_invariant", {})
    mism = inv.get("mismatched_keys", ["<summary 缺失>"])

    check("贯通: H1 落盘指纹 == 384", budget_of(chain, "h1") == 384, budget_of(chain, "h1"))
    check("贯通: MCQ 落盘指纹 == 16", budget_of(chain, "mcq") == 16, budget_of(chain, "mcq"))
    check("贯通: backend_invariant 无 max_new_tokens 不一致",
          not any(str(m).startswith("generation_params.max_new_tokens") for m in mism), mism)
    check("贯通: mode == simulated", summary.get("mode") == "simulated", summary.get("mode"))
    check("贯通: real_g3_passed == false", summary.get("real_g3_passed") is False,
          summary.get("real_g3_passed"))
    check("贯通: production_fingerprint_is_now_frozen == false",
          summary.get("production_fingerprint_is_now_frozen") is False,
          summary.get("production_fingerprint_is_now_frozen"))
    check("贯通: refusals 含 dry_run_present",
          any(str(x).startswith("dry_run_present") for x in summary.get("refusals", [])),
          summary.get("refusals"))

    # ── ③ 故障注入：错误长度仍须被原检查拒绝 ────────────────────────────
    fps = {a: json.loads((chain / f"{a}_g3_run_fingerprint.json").read_text(encoding="utf-8"))
           for a in ("h1", "mcq")}
    fps["mcq"]["generation_params"]["max_new_tokens"] = 384       # 注入错误长度
    runs = {a: type("R", (), {"fingerprint": fps[a]})() for a in ("h1", "mcq")}
    injected = g3.backend_invariant(runs, cfg)
    check("故障注入: 错误长度仍被拒绝",
          (not injected["ok"])
          and "generation_params.max_new_tokens[mcq]" in injected["mismatched_keys"],
          injected["mismatched_keys"])

    doc = {
        "schema": "stage1_l2c_dry_budget_cli_verification_v1",
        "date": "2026-09-14",
        "what": "G3 模拟指纹解码预算修复的实际行为验证（真实 CLI + 落盘产物）",
        "not_claimed": "全部为 dry-run 模拟；**不加载模型**，不构成任何真实 gate 证据",
        "entry": str(ENTRY.relative_to(ROOT)).replace("\\", "/"),
        "interpreter": sys.executable,
        "work_dir": str(work),
        "artifact_paths": {
            "per_arm_cases": [c["out_dir"] for c in cases],
            "chain_dir": str(chain),
            "chain_summary": str(summary_path),
            "chain_fingerprints": {a: str(chain / f"{a}_g3_run_fingerprint.json")
                                   for a in ("h1", "mcq")},
        },
        "cli_cases": cases,
        "chain_cli_runs": chain_runs,
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"], "|", c["detail"])
    print("all_ok:", doc["all_ok"])
    print("written:", OUT)
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
