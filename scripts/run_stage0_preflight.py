"""Stage 0 production preflight（W0.5-C）。

    python scripts/run_stage0_preflight.py

**只读**:不生成 completion、不加载模型权重、不改 v2.3 hash、不碰 G1 状态。

它回答的是一个问题:**如果 CFP 明天确认，我们能不能按按钮？**
输出 blocked_by 把"缺外部事实"和"缺代码"分开 —— 7/29 前应达到
    blocked_by = ["external_cfp_confirmation"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import stage0_preflight as pf  # noqa: E402

REPORT_SCHEMA = "stage0_production_preflight_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default="docs/paper/preregistration.md")
    ap.add_argument("--lock", default="docs/paper/preregistration.lock")
    ap.add_argument("--cfp-record", default="docs/paper/cfp_record.json")
    ap.add_argument("--smoke-dir", default="outputs/logs/sprint_4D_2_conditional_increment")
    ap.add_argument("--h1-samples", default="data/processed/h1/h1_samples.jsonl")
    ap.add_argument("--mcq-pool", default="data/processed/cyber/cybermetric.jsonl")
    ap.add_argument("--burned-traces",
                    default="outputs/logs/sprint_4B_3_full_f5_baseline_and_site_transfer/"
                            "trace_sampling_manifest.jsonl")
    ap.add_argument("--ontology-dir", default="data/raw/ontology")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--output-root", default="outputs/stage0")
    args = ap.parse_args()

    gates = pf.check_gates(Path(args.prereg), Path(args.lock), Path(args.cfp_record),
                           Path(args.smoke_dir))
    code = pf.check_code_readiness()
    assets = pf.check_input_assets(Path(args.h1_samples), Path(args.mcq_pool),
                                   Path(args.burned_traces), Path(args.ontology_dir),
                                   Path(args.model_path))
    scale = pf.check_scale(Path(args.output_root))
    inv = pf.check_frozen_invariants()
    verdict = pf.launch_readiness(gates, code, assets, scale, inv)

    report = {
        "schema_version": REPORT_SCHEMA,
        "read_only": True,
        "did_not": ["generate completions", "load model weights", "change the v2.3 hash",
                    "touch G1 status"],
        "gates": gates, "code_readiness": code, "input_assets": assets,
        "scale_projection": scale, "frozen_invariants": inv,
        **verdict,
    }
    out = Path(args.output_root) / "stage0_production_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    g = gates
    print("=" * 72)
    print("Stage 0 Production Preflight (read-only)")
    print("=" * 72)
    print(f"  G1 CFP                 : {'OK' if g['G1']['ok'] else 'BLOCKED':8s} "
          f"target={g['G1'].get('target')} status={g['G1'].get('status')} "
          f"missing={g['G1'].get('missing')}")
    print(f"  G2 preregistration     : {'OK' if g['G2']['ok'] else 'BLOCKED':8s} "
          f"sha={g['G2']['current'][:16]}")
    print(f"  G3 model smoke         : {'OK' if g['G3']['ok'] else 'BLOCKED':8s} "
          f"h1={g['G3']['h1_arm']['ok']} mcq={g['G3']['mcq_arm']['ok']} "
          f"backend_invariant={g['G3']['backend_invariant']['ok']}")
    print(f"  code readiness         : {'OK' if code['ok'] else 'NOT READY'}")
    for k, v in code.items():
        if isinstance(v, dict):
            print(f"      {k:34s} {'ok' if v.get('ok') else 'FAIL'}")
    print(f"  input assets           : {'OK' if assets['ok'] else 'NOT READY'}")
    for k, v in assets.items():
        if isinstance(v, dict):
            print(f"      {k:34s} {'ok' if v.get('ok') else 'FAIL'}")
    print(f"  scale projection       : hidden {scale['total_hidden_mb']} MB total "
          f"(h1 {scale['h1']['hidden_mb']} + mcq {scale['mcq']['hidden_mb']}), "
          f"sharding_required={scale['sharding_required']}")
    print(f"  disk free              : {scale['disk_free_gb']} GB "
          f"(need ~{scale['estimated_need_gb']} GB)")
    print("-" * 72)
    print(f"  stage0_launch_allowed  : {verdict['stage0_launch_allowed']}")
    print(f"  blocked_by             : {verdict['blocked_by']}")
    print(f"  readiness              : {verdict['readiness']}")
    if verdict["what_unblocks_it"]:
        print(f"  what unblocks it       : {verdict['what_unblocks_it']}")
    print("=" * 72)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
