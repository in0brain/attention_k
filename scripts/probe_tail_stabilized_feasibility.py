"""tail-stabilized 可行性硬闸 —— **只做可行性验证**，≤64 traces，严格串行。

不解盲、不读 label、不算性能量、不写正式 Route-B v2 cache、不改 FROZEN_PATHS、
不建 readiness v3、不改任何文档。`backends.py` **一行不动**，只通过 forward hook 介入。

两步：

    --scan       逐层定位：在已知 final hidden 非有限的 trace 上，对**完整 sequence**
                 （prompt + completion，不只 identifier positions）逐层检查有限性，
                 回答"首次非有限出现在冻结 block 19（tuple 20）之前 / 之中 / 之后"。
    --controls   三组硬控制（仅当 block-19 完整输出全有限才有意义）。

样本口径（研究者裁定）：

    H1 已知坏 traces        直接验证 tail-stabilized 对**已观察**非有限问题的修复
    MCQ retained+analyzable 直接验证 no-intervention、冻结 answer-letter H 锚点、
                            cast 前边界有限性、外部 stable logits 有限性
    9 例 MCQ technical exclusions
                            **untestable historical failure cases** ——
                            completion 原文未持久化，本轮不判断 tail-stabilized 是否
                            修复这些历史失败；**不重放、不重新纳入、不作跨臂推断**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool          # noqa: E402
from recover_attention.evaluation import v3_production as prod           # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig      # noqa: E402
from recover_attention.generation import stable_logits as sl             # noqa: E402

PROD = ROOT / "outputs/stage1/production"
REPOOL = ROOT / "outputs/stage1/repool"
SMOKE = ROOT / "outputs/stage1/smoke"

SCAN_OUT = SMOKE / "tail_stabilized_layer_scan.json"
CTRL_OUT = SMOKE / "tail_stabilized_controls.json"
TELEMETRY = SMOKE / "tail_stabilized_gpu_telemetry.csv"

#: 冻结口径：block 19 的输出 == hidden_states[20]（进第 20 层前记录）。
CAST_BLOCK = 19
FROZEN_TUPLE_IDX = 20

N_H1 = 32
N_MCQ = 32


class ProbeStop(RuntimeError):
    """硬控制未通过 —— 立即停止，不换层、不逐 trace 选边界。"""


# ------------------------------------------------------------------ 工具
def publish_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise ProbeStop(f"输出已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ProbeStop(f"陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ProbeStop(f"os.link 失败，fail-closed：{exc}；临时文件保留在 {tmp}") from exc
    os.unlink(tmp)


def start_telemetry(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("x", encoding="utf-8", newline="")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,"
         "clocks.sm,temperature.gpu,clocks_throttle_reasons.active", "--format=csv",
         "-l", "15"], stdout=fh, stderr=subprocess.DEVNULL)
    return proc, fh


def stop_telemetry(handle) -> None:
    proc, fh = handle
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:                                   # noqa: BLE001
        proc.kill()
    finally:
        fh.close()


def load_analyzable(arm: str) -> list[dict]:
    raw = (PROD / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
    rows = [json.loads(x) for x in raw if x.strip()]
    return [r for r in rows if prod.requires_hidden(arm, r)]


def h1_known_bad(n: int = N_H1) -> list[str]:
    """确定性取样：步骤 0 已确认 final hidden 非有限的 trace。"""
    d = json.loads((SMOKE / "shadow_smoke_report.json").read_text(encoding="utf-8"))
    bad = sorted(u["unit_id"] for u in d["per_unit"]
                 if u["row_counts"]["final_hidden_nonfinite"] > 0)
    return bad[:n]


def mcq_retained(n: int = N_MCQ) -> list[str]:
    ids = sorted(r["example_id"] for r in load_analyzable("mcq"))
    step = max(1, len(ids) // n)
    return ids[::step][:n]


def excluded_units_note() -> dict:
    diag = [json.loads(x) for x in
            (PROD / "mcq_forward_nonfinite_diagnostics.jsonl").read_bytes().split(b"\n")
            if x.strip()]
    exc = [json.loads(x) for x in
           (PROD / "mcq_technical_exclusions.jsonl").read_bytes().split(b"\n")
           if x.strip()]
    return {
        "classification": "untestable_historical_failure_cases",
        "n_technical_exclusions": len(exc),
        "n_with_diagnostic": len(diag),
        "unit_ids_with_diagnostic": sorted(d["unit_id"] for d in diag),
        "why_untestable": ("completion 原文**未持久化** —— 失败在写 raw 行之前抛出，"
                           "mcq_raw_unit_results.jsonl 里 0/8 可找到；诊断只留 SHA 与长度。"),
        "first_failure_has_no_diagnostic": True,
        "policy": ("**不重放、不重新纳入**；本轮**不判断** tail-stabilized 是否修复"
                   "这些历史失败；**不作跨臂推断**。"),
    }


# ------------------------------------------------------------- 前向工具
def _encode(backend, prompt: str, completion: str):
    tok = backend.ctx["tokenizer"]
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    return p_ids, c_ids


def _forward(backend, p_ids, c_ids, *, base_only: bool = False):
    """`base_only=True` 时只跑 `model.model`（不含 lm_head）。

    tail 模式**必须**用它：cast 后残差流被类型提升为 float32，模型自带的
    fp16 lm_head 会直接抛 dtype mismatch —— 而正式 logits 本来就要求由
    `stable_logits` 的 float32 lm_head/logsumexp **外部重算**，模型自带的用不上。
    """
    torch = backend.ctx["torch"]
    full = torch.tensor([p_ids + c_ids])
    inputs = backend._to_device({"input_ids": full,
                                 "attention_mask": torch.ones_like(full)})
    module = backend.ctx["model"].model if base_only else backend.ctx["model"]
    with torch.no_grad():
        return module(**inputs, output_hidden_states=True)


class Block19Hook:
    """在 `model.model.layers[19]` 上捕获输出；`cast=True` 时把它换成 bf16。

    **优先用 hook 修改层输出并继续调用模型原 forward** —— 避免手工重建
    attention mask / position ids / rotary embedding。
    """

    def __init__(self, model, torch, cast: bool):
        self.torch = torch
        self.cast = cast
        self.captured = None
        self.output_was_tuple = None
        self.handle = model.model.layers[CAST_BLOCK].register_forward_hook(self._fn)

    def _fn(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        self.output_was_tuple = is_tuple
        hidden = output[0] if is_tuple else output
        self.captured = hidden.detach().clone()          # **cast 之前**的 fp16 输出
        if not self.cast:
            return None                                   # 原样返回，不改动
        new = hidden.to(self.torch.bfloat16)
        return (new,) + tuple(output[1:]) if is_tuple else new

    def close(self):
        self.handle.remove()


def _pool_from_captured(captured, prompt_len: int, positions, torch) -> np.ndarray:
    """与冻结实现同一口径：hidden[tuple_idx][0].detach().float().cpu().numpy() 后 pooling。"""
    hidden = captured[0].detach().float().cpu().numpy()
    return repool.pool_union(hidden, prompt_len, positions)


# ------------------------------------------------------------- 步骤 1
def run_scan(model_path: str) -> int:
    if SCAN_OUT.exists():
        raise ProbeStop(f"输出已存在，拒绝覆盖：{SCAN_OUT}")
    units = h1_known_bad()
    by = {repool.derive_unit_id(r): r for r in load_analyzable("h1")}
    units = [u for u in units if u in by]
    print(f"[scan ] {len(units)} 条已知 final-hidden 非有限的 H1 trace（确定性取样）")

    tel = start_telemetry(TELEMETRY)
    backend = None
    try:
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        torch = backend.ctx["torch"]
        per_unit = []
        for uid in units:
            row = by[uid]
            prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
            p_ids, c_ids = _encode(backend, prompt, row["completion"])
            out = _forward(backend, p_ids, c_ids)
            plen, total = len(p_ids), len(p_ids) + len(c_ids)
            layers = []
            first_bad_idx = None
            for i, hs in enumerate(out.hidden_states):
                h = hs[0]
                finite = torch.isfinite(h).all(dim=-1)     # 逐 token（完整 sequence）
                n_bad = int((~finite).sum())
                entry = {"tuple_index": i, "n_nonfinite_tokens": n_bad,
                         "n_tokens": int(h.shape[0])}
                if n_bad:
                    pos = int((~finite).nonzero()[0, 0])
                    entry.update({
                        "first_nonfinite_token_position": pos,
                        "region": "prompt" if pos < plen else "completion",
                        "n_nonfinite_values_total": int((~torch.isfinite(h)).sum()),
                    })
                    if first_bad_idx is None:
                        first_bad_idx = i
                layers.append(entry)
            per_unit.append({
                "unit_id": uid, "prompt_len": plen, "total_len": total,
                "n_hidden_states": len(out.hidden_states),
                "first_nonfinite_tuple_index": first_bad_idx,
                "block19_output_tuple_index": FROZEN_TUPLE_IDX,
                "block19_full_sequence_all_finite":
                    layers[FROZEN_TUPLE_IDX]["n_nonfinite_tokens"] == 0,
                "layers": layers,
            })
            del out
            torch.cuda.empty_cache()

        firsts = [u["first_nonfinite_tuple_index"] for u in per_unit]
        n_clean19 = sum(1 for u in per_unit if u["block19_full_sequence_all_finite"])
        earliest = min(x for x in firsts if x is not None) if any(
            x is not None for x in firsts) else None
        verdict = ("after_block19" if earliest is not None and earliest > FROZEN_TUPLE_IDX
                   else "at_block19" if earliest == FROZEN_TUPLE_IDX
                   else "before_block19" if earliest is not None else "no_nonfinite_found")
        payload = {
            "schema": "stage1_tail_stabilized_layer_scan_v1",
            "purpose": "逐层定位首次非有限；**完整 sequence**，不只 identifier positions",
            "no_label_or_performance_analysis": True,
            "n_units": len(per_unit),
            "frozen_block_index_zero_based": CAST_BLOCK,
            "frozen_hidden_states_tuple_index": FROZEN_TUPLE_IDX,
            "earliest_nonfinite_tuple_index_across_units": earliest,
            "n_units_with_block19_full_sequence_all_finite": n_clean19,
            "verdict_relative_to_block19": verdict,
            "verdict_meaning": {
                "after_block19": "tail-stabilized **可行**：block 19 输出干净，坏在其后",
                "at_block19": "tail-stabilized **不可行**：block 19 自身已坏",
                "before_block19": "tail-stabilized **不可行**：更早已坏",
            },
            "mcq_technical_exclusions": excluded_units_note(),
            "per_unit": per_unit,
        }
        publish_json_exclusive(SCAN_OUT, payload)
        print(f"[scan ] 最早非有限 tuple_index = {earliest} ｜ 判定 = **{verdict}**")
        print(f"[scan ] block19 完整 sequence 全有限的 unit：{n_clean19}/{len(per_unit)}")
        print(f"[out  ] {SCAN_OUT}")
        return 0 if verdict == "after_block19" else 3
    finally:
        stop_telemetry(tel)
        backend = None


# ------------------------------------------------------------- 步骤 2+3
def run_controls(model_path: str) -> int:
    if CTRL_OUT.exists():
        raise ProbeStop(f"输出已存在，拒绝覆盖：{CTRL_OUT}")
    scan = json.loads(SCAN_OUT.read_text(encoding="utf-8"))
    if scan["verdict_relative_to_block19"] != "after_block19":
        raise ProbeStop(
            f"逐层扫描判定为 {scan['verdict_relative_to_block19']} —— "
            f"tail-stabilized 不可行，按裁定规则停止，**不换层**")

    cfg = V3AnalysisConfig()
    h1_by = {repool.derive_unit_id(r): r for r in load_analyzable("h1")}
    mcq_by = {r["example_id"]: r for r in load_analyzable("mcq")}
    h1_units = [u for u in h1_known_bad() if u in h1_by]
    mcq_units = [e for e in mcq_retained() if e in mcq_by]
    print(f"[ctrl ] H1 已知坏 {len(h1_units)} ｜ MCQ retained+analyzable {len(mcq_units)}")

    tel = start_telemetry(SMOKE / f"tail_stabilized_controls_seg{len(list(SMOKE.glob('tail_stabilized_controls_seg*'))) + 1:03d}_gpu_telemetry.csv")
    backend = None
    try:
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        torch = backend.ctx["torch"]
        head = sl.Float32Head(backend.ctx["model"], torch)
        frozen_mcq_hidden = dict(np.load(PROD / "mcq_hidden.npz"))

        c1, c2, c3 = [], [], []

        # ---------------- 控制 ①：无干预 -------------------------------
        for uid in h1_units[:16] + [f"MCQ::{e}" for e in mcq_units[:16]]:
            is_mcq = uid.startswith("MCQ::")
            if is_mcq:
                row = mcq_by[uid[5:]]
                prompt, completion = row["rendered_prompt"], row["raw_completion"]
            else:
                row = h1_by[uid]
                prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
                completion = row["completion"]
            p_ids, c_ids = _encode(backend, prompt, completion)

            base = _forward(backend, p_ids, c_ids)
            base_logits = base.logits[0].detach().float().cpu().numpy()
            base_h20 = base.hidden_states[FROZEN_TUPLE_IDX][0].detach().float().cpu().numpy()
            del base
            torch.cuda.empty_cache()

            hook = Block19Hook(backend.ctx["model"], torch, cast=False)
            try:
                out = _forward(backend, p_ids, c_ids)
                logits = out.logits[0].detach().float().cpu().numpy()
                h20 = out.hidden_states[FROZEN_TUPLE_IDX][0].detach().float().cpu().numpy()
                cap = hook.captured[0].detach().float().cpu().numpy()
            finally:
                hook.close()
            entry = {"unit_id": uid, "arm": "mcq" if is_mcq else "h1",
                     "output_was_tuple": hook.output_was_tuple,
                     "logits_exact": bool(np.array_equal(base_logits, logits,
                                                         equal_nan=True)),
                     "hidden20_exact": bool(np.array_equal(base_h20, h20, equal_nan=True)),
                     "captured_equals_hidden20": bool(np.array_equal(cap, h20,
                                                                     equal_nan=True))}
            c1.append(entry)
            if not (entry["logits_exact"] and entry["hidden20_exact"]
                    and entry["captured_equals_hidden20"]):
                raise ProbeStop(
                    f"无干预控制失败（hook 或层边界错误）：{entry}")
            del out
            torch.cuda.empty_cache()
        print(f"[ctrl ] ① 无干预：{len(c1)}/{len(c1)} 逐位一致")

        # ---------------- 控制 ②③：尾部稳定 + 锚点 ----------------------
        for uid in h1_units + [f"MCQ::{e}" for e in mcq_units]:
            is_mcq = uid.startswith("MCQ::")
            if is_mcq:
                row = mcq_by[uid[5:]]
                prompt, completion = row["rendered_prompt"], row["raw_completion"]
            else:
                row = h1_by[uid]
                prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
                completion = row["completion"]
            p_ids, c_ids = _encode(backend, prompt, completion)
            plen = len(p_ids)

            hook = Block19Hook(backend.ctx["model"], torch, cast=True)
            try:
                out = _forward(backend, p_ids, c_ids, base_only=True)
                cap_t = hook.captured
                pre_cast_all_finite = bool(torch.isfinite(cap_t).all())
                final_hidden = out.hidden_states[-1][0]
                final_finite = bool(torch.isfinite(final_hidden).all())
                rows_idx = [sl.causal_row_index(plen, j) for j in range(len(c_ids))]
                # **正式** logits 一律由 cast 后 final hidden 外部重算（float32）
                stable_ok, stable_diag = True, {}
                try:
                    lg = head.project(final_hidden[rows_idx])
                    lp = sl.log_softmax_rows(lg, torch)
                    st = sl.row_statistics(lp, c_ids, torch)
                    stable_diag = {"n_rows": int(lg.shape[0]),
                                   "n_tokens_used": len(c_ids),
                                   "no_token_filtering": int(lg.shape[0]) == len(c_ids),
                                   "rank_min": int(st["rank"].min()),
                                   "rank_max": int(st["rank"].max())}
                    del lg, lp
                except sl.NonFiniteStop as exc:
                    stable_ok, stable_diag = False, {"stop": str(exc)}
                # tail 模式下模型自带的 lm_head **调不动**（fp16 权重 vs float32 激活），
                # 因此原生 logits 这项诊断在本模式下**不存在**，如实记为 null 并写明原因。
                native_finite = None
                final_dtype = str(final_hidden.dtype)
                cap_np = cap_t[0].detach().float().cpu().numpy()
            finally:
                hook.close()

            c2.append({"unit_id": uid, "arm": "mcq" if is_mcq else "h1",
                       "pre_cast_block19_all_finite": pre_cast_all_finite,
                       "post_cast_final_hidden_all_finite": final_finite,
                       "external_float32_stable_logits_ok": stable_ok,
                       "native_logits_finite_DIAGNOSTIC_ONLY": native_finite,
                       "native_logits_unavailable_reason": (
                           "tail 模式下残差流被提升为 float32，模型自带 fp16 lm_head "
                           "抛 dtype mismatch；正式 logits 本就要求外部 float32 重算"),
                       "post_cast_residual_dtype": final_dtype,
                       **stable_diag})

            # ---- 控制 ③：锚点用 **cast 之前** 捕获的 block-19 输出 ----
            if is_mcq:
                j = int(row["letter_token_index_local"])
                got = cap_np[plen + j]
                want = frozen_mcq_hidden[row["example_id"]]
                c3.append({"unit_id": uid, "kind": "mcq_answer_letter_H",
                           **repool.anchor_exact_match(got, want)})
            else:
                pos = repool.union_positions(row)
                got = repool.pool_union(cap_np, plen, pos)
                want = np.load(PROD / "h1_unit_hidden" / f"{uid}.npy")
                c3.append({"unit_id": uid, "kind": "h1_union_H",
                           **repool.anchor_exact_match(got, want)})
                assignment = repool.block_assignment(row, cfg)
                blocks = repool.pool_blocks(cap_np, plen, assignment)
                with np.load(REPOOL / "h1_e3_block_hidden" / f"{uid}.npz",
                             allow_pickle=False) as z:
                    for b, v in blocks.items():
                        c3.append({"unit_id": uid, "kind": f"h1_e3_block_{b}",
                                   **repool.anchor_exact_match(v, z[f"block_{b}"])})
            del out
            torch.cuda.empty_cache()

        n_pre = sum(1 for e in c2 if e["pre_cast_block19_all_finite"])
        n_fin = sum(1 for e in c2 if e["post_cast_final_hidden_all_finite"])
        n_stab = sum(1 for e in c2 if e["external_float32_stable_logits_ok"])
        n_nofilt = sum(1 for e in c2 if e.get("no_token_filtering"))
        n_anchor = sum(1 for e in c3 if e["exact"])
        print(f"[ctrl ] ② cast前全有限 {n_pre}/{len(c2)} ｜ cast后 final hidden 全有限 "
              f"{n_fin}/{len(c2)} ｜ 外部 stable logits {n_stab}/{len(c2)} ｜ "
              f"无过滤 {n_nofilt}/{len(c2)}")
        print(f"[ctrl ] ③ 锚点精确一致 {n_anchor}/{len(c3)}")

        all_pass = (len(c1) and all(e["logits_exact"] and e["hidden20_exact"] for e in c1)
                    and n_pre == len(c2) and n_fin == len(c2) and n_stab == len(c2)
                    and n_nofilt == len(c2) and n_anchor == len(c3))
        payload = {
            "schema": "stage1_tail_stabilized_controls_v1",
            "cast_block_index_zero_based": CAST_BLOCK,
            "boundary_policy": ("边界**全臂固定**在 block 19；"
                                "**不逐 trace 选择不同 cast 层**"),
            "logits_policy": ("tail 模式的**正式** logits 一律由 cast 后尾部 final hidden 经 "
                              "stable_logits 的 float32 lm_head/logsumexp **外部重算**；"
                              "原生 out.logits 在 tail 模式下**仅作诊断**。"
                              "只有无干预控制才要求原生 out.logits 与原 forward 精确一致。"),
            "sample_roles": {
                "h1_known_bad": "直接验证 tail-stabilized 对**已观察**非有限问题的修复",
                "mcq_retained_analyzable": ("直接验证 no-intervention、冻结 answer-letter H "
                                            "锚点、cast 前边界有限性、外部 stable logits 有限性"),
            },
            "mcq_technical_exclusions": excluded_units_note(),
            "control_1_no_intervention": {"n": len(c1), "per_unit": c1},
            "control_2_tail_stability": {
                "n": len(c2), "n_pre_cast_all_finite": n_pre,
                "n_post_cast_final_hidden_all_finite": n_fin,
                "n_external_stable_logits_ok": n_stab,
                "n_no_token_filtering": n_nofilt, "per_unit": c2},
            "control_3_frozen_anchors": {"n": len(c3), "n_exact": n_anchor,
                                         "tolerance": "none (exact equality)",
                                         "per_unit": c3},
            "all_controls_passed": bool(all_pass),
            "future_full_run_requirement": (
                "逐 trace 检查 **cast 前** layer-19 输出的**完整 sequence** 全有限；"
                "tail 后 final hidden 与外部 float32 stable logits 全有限；"
                "任一失败**立即停** —— 不换边界、不写 completion report。"),
        }
        publish_json_exclusive(CTRL_OUT, payload)
        print(f"[ctrl ] all_controls_passed = **{all_pass}**")
        print(f"[out  ] {CTRL_OUT}")
        return 0 if all_pass else 3
    finally:
        stop_telemetry(tel)
        backend = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    args = ap.parse_args(argv)
    if args.scan:
        return run_scan(args.model_path)
    if args.controls:
        return run_controls(args.model_path)
    ap.error("选择 --scan 或 --controls")


if __name__ == "__main__":
    raise SystemExit(main())
