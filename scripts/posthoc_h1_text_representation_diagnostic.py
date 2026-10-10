"""POST-HOC：H1 的 rung2>rung4 到底是"任务被字符串解决"还是"TF-IDF 稀释"？

**这不是结果，不改动、不重跑、不替换 confirmatory。** 它只闭合 identification audit
finding 3 里那句"两种解释现有数据分不开"——用**现有 completion 文本**的三种表示跑同一
条 ladder 机制，不引入任何新样本、不动 hidden、不动冻结的 O 定义。

────────────────────────────────────────────────────────────────────────
三种文本表示（都喂进 §7 冻结的同一套 TF-IDF：word 1-2gram + char_wb 3-5gram）

  full        整条 completion（= 冻结的 rung4_full_text 原文，作基线复算）
  window      每个 mention 前后 ±W 字符的局部窗口拼接（去掉几百字无关正文的稀释）
  window_mask 同 window，但把 identifier 字符串本身替换为占位符（切断字符串通道，
              只留"identifier 周围的措辞"）

────────────────────────────────────────────────────────────────────────
判读（写在跑之前）

  full → window 大涨          full-text 确有稀释问题（TF-IDF 被无关正文淹没）
  window_mask 仍强            identifier 的**上下文措辞**本身带信号（模型措辞泄露不确定）
  window_mask 弱、window 强   信号主要在 identifier 字符串本身（string/memorisation channel）
  两者都弱                    rung2 之强更像纯任务/字符串 artifact，上下文无补充信息

同时在两种分组下各跑一遍（A=prompt，C=prompt∪fabricated-id），因为 window_mask 是否
"真的切断了字符串通道"只有在封堵了跨折记忆的 C 分组下才干净可读。

────────────────────────────────────────────────────────────────────────
不做的事：不改 O 的定义、不改 Δ_H、不用本分析论证增量有无、不 bump 预注册。
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import conditional_increment as ci  # noqa: E402
from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: E402
from recover_attention.evaluation.feature_cache import ARM_H1, load_arm  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "leak", PROJECT_ROOT / "scripts" / "posthoc_h1_identifier_leakage_diagnostic.py")
leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak)

MASK_TOKEN = " idmask "


def _mentions(rec):
    return [m for m in rec.get("mentions", [])
            if m.get("start") is not None and m.get("end") is not None]


def text_full(rec) -> str:
    return str(rec.get("completion", ""))


def text_window(rec, w: int, mask: bool) -> str:
    """每个 mention ±w 字符窗口拼接；mask=True 时把 identifier 本身换成占位符。

    窗口按 [start-w, end+w] 取；mask 时先把 [start,end] 替成 MASK_TOKEN 再取窗口，
    确保字符串本身不进 TF-IDF，只留周围措辞。
    """
    comp = str(rec.get("completion", ""))
    ms = _mentions(rec)
    if not ms:
        return ""
    pieces = []
    for m in ms:
        s, e = int(m["start"]), int(m["end"])
        lo, hi = max(0, s - w), min(len(comp), e + w)
        if mask:
            seg = comp[lo:s] + MASK_TOKEN + comp[e:hi]
        else:
            seg = comp[lo:hi]
        pieces.append(seg)
    return " ".join(pieces)


def run_rung(y, groups, text, config) -> dict:
    """单个文本-only rung 的 OOF AUROC（复用 §7 冻结的 block_oof_scores）。"""
    res = ci.block_oof_scores(y, groups, ci.stratified_grouped_folds(
        y, groups, n_splits=config.n_outer_folds, seed=config.seed),
        text=list(text), dense_blocks=None,
        c_grid=config.c_grid, inner_splits=config.n_inner_folds, seed=config.seed)
    return {"auroc": float(res["auroc"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/stage0/features")
    ap.add_argument("--output", default="outputs/stage0/evaluation/"
                                        "posthoc_h1_text_representation_diagnostic.json")
    ap.add_argument("--config", default="configs/stage0.yaml")
    ap.add_argument("--window", type=int, default=60,
                    help="mention 前后各取多少字符")
    ap.add_argument("--timestamp", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    config = Stage0Config(n_outer_folds=cfg.n_outer_folds, n_inner_folds=cfg.n_inner_folds,
                          n_boot=cfg.n_boot, seed=cfg.seed, mode=cfg.mode,
                          allow_pilot_population=False)
    in_dir = Path(args.input_dir)
    cache = load_arm(ARM_H1, in_dir / "h1_completion_records.jsonl", in_dir / "h1_hidden.npz",
                     config, label_field="label", id_field="trace_id")
    y = cache.y

    reps = {
        "full": [text_full(r) for r in cache.records],
        "window": [text_window(r, args.window, mask=False) for r in cache.records],
        "window_mask": [text_window(r, args.window, mask=True) for r in cache.records],
    }
    # 报告文本量级，佐证"稀释"确实存在（full 的字数远大于 window）
    charlens = {k: int(np.median([len(t) for t in v])) for k, v in reps.items()}
    print(f"[textdiag] median chars: {charlens}")

    groupings = {
        "A_prompt": cache.groups,
        "C_prompt_union_fabid": leak.component_groups(
            cache.records, link_prompt=True, link_fabricated=True)[0],
    }

    cells = {}
    for gname, groups in groupings.items():
        cells[gname] = {}
        for rname, text in reps.items():
            r = run_rung(y, groups, text, config)
            cells[gname][rname] = r["auroc"]
            print(f"[textdiag] {gname:22s} {rname:12s} AUROC={r['auroc']:.4f}")

    # 参照：把冻结报告里的 rung2/rung4 一并列出，便于对照
    def read_conf():
        p = in_dir.parent / "evaluation" / "stage0_analysis_report.json"
        if not p.exists():
            return {}
        rep = json.loads(p.read_text(encoding="utf-8"))
        L = rep["ladder"]["h1"]["rungs"]
        return {"rung2_id_string_only": L.get("rung2_id_string_only"),
                "rung4_full_text": L.get("rung4_full_text")}

    report = {
        "schema_version": "posthoc_h1_text_representation_v1",
        "status": ("POST-HOC — not a result; closes the surface-artifact-vs-dilution ambiguity "
                  "in identification_audit finding 3 using existing completions only. Does not "
                  "modify or supersede stage0_analysis_report.json, and is not used to argue an "
                  "increment. O's definition and delta_H are untouched."),
        "timestamp": args.timestamp,
        "window_chars_each_side": args.window,
        "median_text_length_chars": charlens,
        "confirmatory_reference_prompt_grouped": read_conf(),
        "text_representation_auroc": cells,
        "interpretation_key": {
            "full_to_window_large_rise": "full-text has a real dilution problem",
            "window_mask_still_strong": "the wording AROUND identifiers carries signal",
            "window_mask_weak_window_strong": "signal is mainly the identifier string itself",
            "both_weak": "rung2's strength is a task/string artifact; context adds nothing",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    print("\n=== 判读辅助（C 分组是干净口径）===")
    for gname in groupings:
        f, w, wm = (cells[gname][k] for k in ("full", "window", "window_mask"))
        print(f"  {gname:22s} full={f:.4f}  window={w:.4f} ({w-f:+.4f})  "
              f"window_mask={wm:.4f} ({wm-w:+.4f} vs window)")
    print(f"\n[textdiag] report -> {out}")


if __name__ == "__main__":
    main()
