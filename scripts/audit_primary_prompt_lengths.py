"""正式集 prompt 的 **只读** token 长度审计（评估显存风险用）。

    python.exe -X utf8 scripts/audit_primary_prompt_lengths.py

身份约束（任一不满足即停）：
    ✔ tokenizer fingerprint 与 G3/G4 冻结的 tokenizer_hash 匹配
    ✔ 输入名单 chain hash 与 sample manifest 匹配
    ✘ 不读取或输出任何标签 / 性能量
    ✘ 不前向、不生成、**不加载模型权重**（只加载 tokenizer）
    ✘ 不拿正式样本做任何试跑

只报告 token 长度分布与 context-window headroom。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention import cyber_data as cd  # noqa: E402
from recover_attention import h1_data as hd  # noqa: E402
from recover_attention.evaluation import v3_production as prod  # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.evaluation.v3_freeze import require_v3_frozen  # noqa: E402
from recover_attention.evaluation.v3_g3_smoke import _read_jsonl, _sha256_file  # noqa: E402
from recover_attention.evaluation.v3_sample_builder import MANIFEST_NAME, chain_digest  # noqa: E402
from recover_attention.generation.manifest import write_atomic  # noqa: E402


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, default=Path("outputs/stage1/samples"))
    ap.add_argument("--g3-dir", type=Path, default=Path("outputs/stage1/g3"))
    ap.add_argument("--prereg", type=Path, default=Path("docs/paper/preregistration_v3.md"))
    ap.add_argument("--lock", type=Path, default=Path("docs/paper/preregistration_v3.lock"))
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/logs/stage1_primary_prompt_length_audit.json"))
    args = ap.parse_args()

    cfg = V3AnalysisConfig()
    freeze = require_v3_frozen(args.prereg, args.lock)

    manifest_path = args.samples_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["prereg"]["sha256"] != freeze.current_sha256:
        raise SystemExit("sample manifest 的 prereg sha256 与当前不一致")

    # 只加载 tokenizer —— **不加载模型权重**
    from recover_attention.hidden_state_cache import import_transformers_components
    tok = import_transformers_components()["AutoTokenizer"].from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False)

    from recover_attention.generation import fingerprint as fp
    tok_fp = fp.tokenizer_fingerprint(tok)
    frozen = json.loads((args.g3_dir / "h1_g3_smoke_report.json").read_text(
        encoding="utf-8"))["fingerprint"]["tokenizer"]
    if tok_fp.get("tokenizer_hash") != frozen.get("tokenizer_hash"):
        raise SystemExit(
            f"tokenizer 指纹与 G3/G4 冻结值不一致：{tok_fp.get('tokenizer_hash')} "
            f"!= {frozen.get('tokenizer_hash')}；停。")

    context_window = int(getattr(tok, "model_max_length", 0) or 0)
    arms: dict[str, dict] = {}
    for arm, filename, key in (
            ("h1", "h1_primary_entries.jsonl", "source_entry_id"),
            ("mcq", "mcq_primary_questions.jsonl", "question_hash")):
        rows = _read_jsonl(args.samples_dir / filename)
        actual = chain_digest(r[key] for r in rows)
        if actual != manifest["chain_sha256"][f"{arm}_primary"]:
            raise SystemExit(f"{filename} 的 chain hash 与 manifest 不符；停。")

        lengths = []
        for row in rows:
            if arm == "h1":
                text = tok.apply_chat_template(hd.build_h1_chat_messages(row),
                                               tokenize=False, add_generation_prompt=True)
            else:
                canonical = cd.to_canonical_cyber_sample(
                    {"question": row["question"], "answers": row["answers"],
                     "solution": row["gold_original_label"]},
                    original_index=int(row["original_index"]), source=row["source"],
                    shuffle_seed=cfg.mcq_shuffle_seed)
                text = tok.apply_chat_template(cd.build_mcq_chat_messages(canonical),
                                               tokenize=False, add_generation_prompt=True)
            lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))

        max_new = cfg.h1_max_new_tokens if arm == "h1" else cfg.mcq_max_new_tokens
        worst = max(lengths) + max_new
        arms[arm] = {
            "n_prompts": len(rows),
            "chain_sha256": actual,
            "prompt_tokens": {"min": min(lengths), "median": _percentile(lengths, 0.5),
                              "p95": _percentile(lengths, 0.95),
                              "p99": _percentile(lengths, 0.99), "max": max(lengths)},
            "max_new_tokens": max_new,
            "worst_case_total_tokens": worst,
            "context_window": context_window,
            "context_headroom_tokens": (context_window - worst) if context_window else None,
        }

    report = {
        "schema_version": "stage1_primary_prompt_length_audit_v1",
        "read_only": True,
        "loaded_model_weights": False,
        "ran_forward_or_generation": False,
        "reads_labels_or_performance": False,
        "prereg_sha256": freeze.current_sha256,
        "lock_version": freeze.locked_version,
        "sample_manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_fingerprint": tok_fp,
        "tokenizer_matches_g3_g4_frozen": True,
        "arms": arms,
        "scope_of_conclusion": (
            "本审计只能证明：计划序列长度远低于 context limit，排除'上下文窗口不足'。"
            "它**不能**证明显存峰值的组成 —— KV cache / activation / 临时 workspace / "
            "量化运行时占用均未分解，须由正式首段的运行期遥测核实。"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, json.dumps(report, ensure_ascii=False, indent=2))

    for arm, data in arms.items():
        t = data["prompt_tokens"]
        print(f"[{arm:4s}] n={data['n_prompts']} tokens min/med/p95/p99/max="
              f"{t['min']}/{t['median']}/{t['p95']}/{t['p99']}/{t['max']} "
              f"worst+gen={data['worst_case_total_tokens']} "
              f"headroom={data['context_headroom_tokens']}")
    print(f"[out  ] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
