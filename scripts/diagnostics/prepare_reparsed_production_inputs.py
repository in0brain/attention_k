r"""把新证据链（`*_budget512_reparsed/`）补齐成**生产入口可用**的形态，并核验一致性。

MCQ 证据来自本轮重提取；**H1 证据按已核验的来源原样复用，不重跑 gate** ——
本脚本逐文件比对 sha256 后复制，并把来源与哈希写进 provenance 清单。

同时跑一遍生产入口的输入校验（`verify_production_inputs`），确认新链两臂都能通过；
**不加载模型、不创建生产目录**。

用法：
    python scripts/diagnostics/prepare_reparsed_production_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_g3_smoke as g3           # noqa: E402
from recover_attention.evaluation import v3_production as prod       # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402

SAMPLES = ROOT / "outputs/stage1/samples"
PREREG = ROOT / "docs/paper/preregistration_v3.md"
LOCK = ROOT / "docs/paper/preregistration_v3.lock"
G3_SRC = ROOT / "outputs/stage1_llama_v3/g3_budget512"
G4_SRC = ROOT / "outputs/stage1_llama_v3/g4_budget512"
G3_NEW = ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed"
G4_NEW = ROOT / "outputs/stage1_llama_v3/g4_budget512_reparsed"
OUT = ROOT / "tmp/stage1_l2_g3/reparsed_production_inputs.json"

#: H1 侧原样复用的文件（MCQ 解析修复不触及 H1）。
H1_REUSE = [
    (G3_SRC / "h1_g3_smoke_report.json", G3_NEW / "h1_g3_smoke_report.json"),
    (G3_SRC / "h1_g3_run_fingerprint.json", G3_NEW / "h1_g3_run_fingerprint.json"),
    (G3_SRC / "h1_g3_unit_results.jsonl", G3_NEW / "h1_g3_unit_results.jsonl"),
    (G4_SRC / "h1_g4_quality_report.json", G4_NEW / "h1_g4_quality_report.json"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    cfg.assert_production()
    failures: list[str] = []
    log: list[dict] = []

    def ck(label: str, ok: bool, detail=None):
        log.append({"check": label, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    # ── 1. H1 证据：核验后原样复用 ──────────────────────────────────────
    print("[1] H1 证据：核验来源哈希后原样复用（不重跑 gate）")
    reused = {}
    for src, dst in H1_REUSE:
        ck(f"来源存在 {src.name}", src.exists())
        if not src.exists():
            continue
        src_sha = _sha(src)
        if dst.exists() and _sha(dst) == src_sha:
            pass
        else:
            shutil.copy2(src, dst)
        ck(f"{dst.name} 与来源逐字节相同", _sha(dst) == src_sha, src_sha[:16])
        reused[str(dst.relative_to(ROOT)).replace("\\", "/")] = {
            "source": str(src.relative_to(ROOT)).replace("\\", "/"),
            "sha256": src_sha, "kind": "reused_verbatim",
            "why": "MCQ 解析修复不触及 H1；按已核验来源复用，未重跑 gate"}

    # ── 2. 新链自洽 ─────────────────────────────────────────────────────
    print("[2] 新证据链自洽")
    g3s = json.loads((G3_NEW / "g3_summary.json").read_text(encoding="utf-8"))
    g4s = json.loads((G4_NEW / "g4_summary.json").read_text(encoding="utf-8"))
    ck("G3 汇总 passed", g3s.get("passed") is True)
    ck("G3 汇总 real_g3_passed", g3s.get("real_g3_passed") is True)
    ck("G4 汇总 passed 且两臂全过", g4s.get("passed") is True
       and all(g4s.get("arms_passed", {}).values()), g4s.get("arms_passed"))
    mcq3 = json.loads((G3_NEW / "mcq_g3_smoke_report.json").read_text(encoding="utf-8"))
    mcq4 = json.loads((G4_NEW / "mcq_g4_quality_report.json").read_text(encoding="utf-8"))
    h1_3 = json.loads((G3_NEW / "h1_g3_smoke_report.json").read_text(encoding="utf-8"))
    ck("MCQ G3 报告 passed 且带 evidence_sources",
       mcq3.get("passed") is True and "evidence_sources" in mcq3)
    ck("MCQ G4 报告 passed 且带 evidence_sources",
       mcq4.get("passed") is True and "evidence_sources" in mcq4)
    ck("MCQ 报告标注了 supersedes（旧报告保留）",
       "supersedes" in mcq3 and "supersedes" in mcq4)
    ck("两臂 G3 指纹 model/tokenizer/hidden_index 逐键相同",
       all(json.dumps(mcq3["fingerprint"][k], sort_keys=True)
           == json.dumps(h1_3["fingerprint"][k], sort_keys=True)
           for k in ("model", "tokenizer", "hidden_index")))
    ck("逐臂预算 h1=384 / mcq=512",
       (h1_3["fingerprint"]["generation_params"]["max_new_tokens"],
        mcq3["fingerprint"]["generation_params"]["max_new_tokens"]) == (384, 512))

    # 解析实现可追溯：报告里记的 parser sha256 == 当前
    parser_sha = _sha(ROOT / "src/recover_attention/domain_label_proxy.py")
    ck("MCQ 报告记录的 parser sha256 == 当前解析实现",
       mcq3["evidence_sources"]["current_parsing"]["parser_sha256"] == parser_sha,
       parser_sha[:16])

    # 预算登记
    reg = g3.load_budget_registration(
        ROOT / "outputs/stage1_llama_v3/budget512_registration.json",
        cfg=cfg, model_path="D:/models/Meta-Llama-3.1-8B-Instruct")
    ck("预算登记可载入且连到通过的生成长度检查",
       reg.registered_max_new_tokens == 512 and reg.length_check.get("passed") is True)
    ck("G3 汇总记录的登记 sha256 == 当前登记",
       (g3s.get("budget_registration") or {}).get("sha256") == reg.sha256)

    # 特征与记录一致
    for tag, d, name, n_expected in (("G3", G3_NEW, "mcq_g3_unit_results.jsonl", 120),
                                     ("G4", G4_NEW, "mcq_g4_unit_results.jsonl", 240)):
        rows = [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines()
                if x.strip()]
        hid = d / ("mcq_g3_unit_hidden" if tag == "G3" else "mcq_g4_unit_hidden")
        elig = sum(1 for r in rows if r.get("eligible_for_primary"))
        ck(f"{tag} MCQ {n_expected} 行、每行带 _provenance",
           len(rows) == n_expected and all("_provenance" in r for r in rows))
        ck(f"{tag} MCQ hidden 数 == eligible 数",
           len(list(hid.glob('*.npy'))) == elig, f"{len(list(hid.glob('*.npy')))} / {elig}")

    # ── 3. 生产入口输入校验（不加载模型）────────────────────────────────
    print("[3] 生产入口输入校验（不加载模型、不创建生产目录）")
    for arm in (prod.ARM_H1, prod.ARM_MCQ):
        try:
            v = prod.verify_production_inputs(samples_dir=SAMPLES, g3_dir=G3_NEW,
                                              g4_dir=G4_NEW, prereg=PREREG, lock=LOCK,
                                              cfg=cfg, arm=arm)
            ck(f"{arm}: 新链通过 verify_production_inputs",
               v["g3_passed"] and v["g4_passed"],
               f"n_primary={v['n_primary']}")
        except Exception as exc:                          # noqa: BLE001
            ck(f"{arm}: 新链通过 verify_production_inputs", False,
               f"{type(exc).__name__}: {exc}")

    # 负例：旧解析的 MCQ 证据目录必须被拒（口径不同，不得混用）
    print("[4] 负例：旧解析证据链")
    old_mcq3 = json.loads((G3_SRC / "mcq_g3_smoke_report.json").read_text(encoding="utf-8"))
    ck("旧 G3 MCQ 报告**没有** evidence_sources（口径不同，可据此识别）",
       "evidence_sources" not in old_mcq3)
    ck("旧 G3 MCQ 报告的 parse_failure_rate 与新报告不同（0.0333 vs 0.025）",
       old_mcq3["checks"]["parse_failure_rate"] != mcq3["checks"]["parse_failure_rate"],
       f'{old_mcq3["checks"]["parse_failure_rate"]} vs {mcq3["checks"]["parse_failure_rate"]}')

    doc = {
        "schema": "stage1_reparsed_production_inputs_v1", "date": "2026-09-15",
        "what": ("把新证据链补齐成生产入口可用形态：MCQ 用本轮重提取的新证据，"
                 "H1 按已核验来源**原样复用**（未重跑 gate）"),
        "g3_dir": str(G3_NEW.relative_to(ROOT)).replace("\\", "/"),
        "g4_dir": str(G4_NEW.relative_to(ROOT)).replace("\\", "/"),
        "h1_reused_verbatim": reused,
        "parser_sha256": parser_sha,
        "budget_registration_sha256": reg.sha256,
        "checks": log,
        "n_failed": len(failures),
        "failures": failures,
        "do_not_mix": ("旧 g3_budget512/ 与 g4_budget512/ 的 MCQ 证据用的是**旧解析**，"
                       "不得与本链混用；两者的 passed 都成立但证据口径不同。"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n[{'ok   ' if not failures else 'STOP '}] "
          f"{len(log) - len(failures)}/{len(log)} 通过")
    print(f"[out  ] {args.out}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
