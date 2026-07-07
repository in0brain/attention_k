"""Sprint 2I stage 1: cache original+masked attention summaries for the 500 subset.

Re-runs Qwen2.5-7B (4-bit, eager attention) on the original and masked questions only
(never the recovered question), extracts leakage-free attention summary features, and
merges them with the 2H-C enriched dataset labels/features. No full attention tensors
are saved; only summaries. Resumable.

Slot token indices are located via the tokenizer's own offset mapping (self-consistent
with the attention seq dimension), using span/mask char ranges from the 2G manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention import attention_features as af  # noqa: E402

G2 = PROJECT_ROOT / "outputs" / "logs" / "sprint_2G_full_scale_2000"
HC = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_feature_enrichment_500"
MANIFEST = G2 / "02_hidden_state_cache" / "hidden_state_manifest.jsonl"
ENRICHED_DATASET = HC / "pre_recovery_feature_dataset.jsonl"

CACHE_OUT = PROJECT_ROOT / "outputs" / "logs" / "sprint_2I_attention_cache_500"
FEAT_OUT = PROJECT_ROOT / "outputs" / "logs" / "sprint_2I_attention_features_500"
CACHE_REPORT = CACHE_OUT / "attention_cache_report.json"
FEATURE_DATASET = FEAT_OUT / "attention_feature_dataset.jsonl"
FEATURE_REPORT = FEAT_OUT / "attention_feature_report.json"

MODEL_PATH = r"D:/models/Qwen2.5-7B-Instruct"
# Attention layers to summarize. The final layer (27) is dropped because its attention
# is all-NaN under 4-bit eager attention; layers 0/8/16/24 are clean (rows sum to 1).
LAYER_INDICES = [0, 8, 16, 24]

NUMBER_RE = re.compile(r"(?<![\dA-Za-z_,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\dA-Za-z_,.])\d+(?:\.\d+)?%?")
QFOCUS_RE = re.compile(r"\b(how many|how much|what|which|how)\b", re.IGNORECASE)
OPERATION_RE = re.compile(
    r"\b(total|each|per|more|less|fewer|times|twice|double|half|sum|difference|remaining|left|altogether|combined|plus|minus|every|apiece)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _token_indices_for_char_ranges(offsets, char_ranges, exclude: set[int]) -> list[int]:
    out = []
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:  # special token / empty
            continue
        if ti in exclude:
            continue
        for cs, ce in char_ranges:
            if ts < ce and cs < te:
                out.append(ti)
                break
    return out


def _span_char_ranges(manifest_rec: dict) -> list[list[int]]:
    it = manifest_rec.get("input_type")
    if it == "original":
        return [s["original_char_range"] for s in manifest_rec.get("masked_original_spans", [])
                if s.get("original_char_range")]
    if it == "masked":
        return [r for r in manifest_rec.get("mask_char_ranges", []) if r and len(r) == 2]
    return []


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=qcfg, device_map="auto",
        attn_implementation="eager", dtype=torch.float16,
    )
    model.eval()
    return tok, model


def _forward_attention(tok, model, text: str, char_ranges: list[list[int]]):
    import torch

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    att = af.head_average_selected_layers(out.attentions, LAYER_INDICES)  # [L, seq, seq]
    slot_idx = _token_indices_for_char_ranges(offsets, char_ranges, exclude=set())
    return att, slot_idx, offsets


def stage_cache(limit: int | None):
    enriched = {r["masked_id"]: r for r in read_jsonl(ENRICHED_DATASET)}
    manifest = read_jsonl(MANIFEST)
    man_by_key = {(r["masked_id"], r["input_type"]): r for r in manifest}

    subset_ids = [mid for mid in enriched if enriched[mid]["fragility_bucket"] is not None]
    if limit:
        subset_ids = subset_ids[:limit]

    done = set()
    if FEATURE_DATASET.exists():
        done = {r["masked_id"] for r in read_jsonl(FEATURE_DATASET)}
    pending = [mid for mid in subset_ids if mid not in done]
    print(f"[cache] subset={len(subset_ids)} done={len(done)} pending={len(pending)}")

    tok, model = load_model()
    print("[cache] model loaded")

    seq_mismatch = 0
    slot_missing = 0
    ctx_missing = {"qfocus": 0, "operation": 0, "numctx": 0}
    feature_name_union: set[str] = set()
    t0 = time.time()

    for i, mid in enumerate(pending):
        orig_rec = man_by_key.get((mid, "original"))
        masked_rec = man_by_key.get((mid, "masked"))
        if orig_rec is None or masked_rec is None:
            continue
        orig_text = orig_rec["input_text"]
        masked_text = masked_rec["input_text"]
        orig_ranges = _span_char_ranges(orig_rec)
        masked_ranges = _span_char_ranges(masked_rec)

        orig_att, orig_slot, orig_offsets = _forward_attention(tok, model, orig_text, orig_ranges)
        masked_att, masked_slot, _ = _forward_attention(tok, model, masked_text, masked_ranges)

        if orig_att.shape[1] != len(orig_offsets):
            seq_mismatch += 1
        if not orig_slot or not masked_slot:
            slot_missing += 1
            continue

        # context indices in ORIGINAL input (exclude slot tokens)
        slot_exclude = set(orig_slot)
        num_ranges = [[m.start(), m.end()] for m in NUMBER_RE.finditer(orig_text)]
        qf_ranges = [[m.start(), m.end()] for m in QFOCUS_RE.finditer(orig_text)]
        op_ranges = [[m.start(), m.end()] for m in OPERATION_RE.finditer(orig_text)]
        context = {
            "qfocus": _token_indices_for_char_ranges(orig_offsets, qf_ranges, slot_exclude),
            "operation": _token_indices_for_char_ranges(orig_offsets, op_ranges, slot_exclude),
            "numctx": _token_indices_for_char_ranges(orig_offsets, num_ranges, slot_exclude),
        }
        for cname, cidx in context.items():
            if not cidx:
                ctx_missing[cname] += 1

        built = af.build_attention_features(
            orig_att, orig_slot, masked_att, masked_slot, LAYER_INDICES, context_orig=context
        )
        feats = built["features"]
        feature_name_union.update(feats.keys())

        src = enriched[mid]
        record = {
            "masked_id": mid,
            "source_question_id": src["source_question_id"],
            "span_type": src["span_type"],
            "span_text": src["span_text"],
            "question": src["question"],
            "solution_path_status": src["solution_path_status"],
            "fragility_bucket": src["fragility_bucket"],
            "fragility_bucket_name": src.get("fragility_bucket_name"),
            "risk_strength": src["risk_strength"],
            "layer_indices": src.get("layer_indices"),
            "feature_values": src.get("feature_values", {}),
            "feature_arrays": src.get("feature_arrays", {}),
            "pre_recovery_features": src.get("pre_recovery_features", {}),
            "attention_features": feats,
            "num_orig_slot_tokens": len(orig_slot),
            "num_masked_slot_tokens": len(masked_slot),
        }
        append_jsonl([record], FEATURE_DATASET)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[cache] {i + 1}/{len(pending)} ({rate:.2f}/s)")

    total = read_jsonl(FEATURE_DATASET) if FEATURE_DATASET.exists() else []
    banned = [n for n in feature_name_union for tok_ in af.BANNED_FEATURE_SUBSTRINGS if tok_ in n]
    write_json({
        "sprint": "2I",
        "stage": "attention_cache",
        "model_path": MODEL_PATH,
        "load_in_4bit": True,
        "attn_implementation": "eager",
        "attention_from_quantized_model": True,
        "layer_indices": LAYER_INDICES,
        "num_records_total": len(total),
        "num_pending_this_run": len(pending),
        "seq_len_mismatch": seq_mismatch,
        "slot_missing_skipped": slot_missing,
        "context_missing_counts": ctx_missing,
        "channels_cached": ["original", "masked"],
        "recovered_channel_cached": False,
    }, CACHE_REPORT)
    write_json({
        "sprint": "2I",
        "stage": "attention_features",
        "num_records": len(total),
        "num_attention_features": len(feature_name_union),
        "attention_feature_names_sample": sorted(feature_name_union)[:40],
        "banned_substring_hits": banned,
        "leakage_free": len(banned) == 0,
        "feature_families": {
            "A_slot_mass": len([n for n in feature_name_union if any(k in n for k in ["in_mass", "out_mass", "self_mass", "rank", "in_rel"])]),
            "B_shape": len([n for n in feature_name_union if any(k in n for k in ["entropy", "mass_layer", "top1", "top3", "top5", "edge_count"])]),
            "C_context_to_slot": len([n for n in feature_name_union if "_to_slot" in n]),
            "E_delta": len([n for n in feature_name_union if n.startswith("attn_delta_")]),
        },
    }, FEATURE_REPORT)
    print(f"[cache] complete: {len(total)} records, {len(feature_name_union)} attention features, "
          f"leakage_free={len(banned) == 0}, seq_mismatch={seq_mismatch}, slot_missing={slot_missing}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 2I attention cache")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    FEAT_OUT.mkdir(parents=True, exist_ok=True)
    stage_cache(args.limit)


if __name__ == "__main__":
    main()
