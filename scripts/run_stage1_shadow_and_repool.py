"""Stage 1 stable-logit shadow + H1 E3 repool —— **performance-blind，不解盲**。

产物只能称 `stable-logit shadow sensitivity` / `protocol-deviation sensitivity`；
**不得**冒充原预注册 primary。原 14 项 H1 B4 仍是 frozen implementation primary，
保留不覆盖不作废；其中 8 项不能按名义科学含义单独解释。

本文件**不修改** `backends.py` 的任何冻结语义 —— 只调用它的加载与前向。

子模式
    --step0   结构与可行性硬闸（≤32 条）：实测 lm_head 结构、final-hidden 口径，
              并判定路线 A / B。**不写正式产物。**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recover_attention.evaluation import v3_e3_repool as repool      # noqa: E402
from recover_attention.evaluation import v3_production as prod       # noqa: E402
from recover_attention.generation import stable_logits as sl         # noqa: E402
from recover_attention.evaluation import v3_shadow_f5 as shadow_f5   # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention import domain_label_proxy as dlp              # noqa: E402

DEFAULT_ARTIFACT_ROOT = ROOT / "outputs/stage1"
ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT

PROD = ARTIFACT_ROOT / "production"
SHADOW = ARTIFACT_ROOT / "shadow"
REPOOL = ARTIFACT_ROOT / "repool"
SMOKE = ARTIFACT_ROOT / "smoke"

#: 本次运行的模型几何；默认 Qwen 冻结几何，由 CLI 覆盖。
GEOMETRY = repool.QWEN_GEOMETRY
_ROOT_REBOUND = False

STEP0_OUT = SMOKE / "shadow_smoke_report.json"

STEP0_N = 32


class ScopeViolation(RuntimeError):
    """越出本轮授权范围。"""


def assert_outputs_absent(paths) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise ScopeViolation("输出已存在，拒绝覆盖：\n  " + "\n  ".join(existing))


def publish_json_exclusive(path: Path, payload: dict) -> None:
    """只创建、绝不覆盖：O_EXCL 临时文件 → os.link 原子发布；失败 fail-closed。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise ScopeViolation(f"发布目标已存在，拒绝覆盖：{path}")
    if tmp.exists():
        raise ScopeViolation(f"存在陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except FileExistsError as exc:
        raise ScopeViolation(f"发布目标在写入期间出现，拒绝覆盖：{path}") from exc
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(
            f"os.link 不可用或失败，fail-closed（不退回 os.replace）："
            f"{type(exc).__name__}: {exc}；临时文件保留在 {tmp}") from exc
    os.unlink(tmp)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_analyzable(arm: str) -> list[dict]:
    raw = (PROD / f"{arm}_completion_records.jsonl").read_bytes().split(b"\n")
    rows = [json.loads(x) for x in raw if x.strip()]
    return [r for r in rows if prod.requires_hidden(arm, r)]


def library_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate"):
        try:
            out[name] = str(getattr(__import__(name), "__version__", "unknown"))
        except Exception as exc:                       # noqa: BLE001
            out[name] = f"<unavailable: {type(exc).__name__}>"
    return out


def start_telemetry(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("x", encoding="utf-8", newline="")
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,"
         "clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
         "--format=csv", "-l", "15"],
        stdout=fh, stderr=subprocess.DEVNULL)
    return proc, fh


def stop_telemetry(handle) -> None:
    proc, fh = handle
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:                                  # noqa: BLE001
        proc.kill()
    finally:
        fh.close()


# ------------------------------------------------------------------ step 0
def step0_probe_unit(backend, row: dict, torch) -> dict:
    """单条 trace：定位非有限发生在 final hidden 还是 lm_head 之后。

    **不预设唯一原因** —— 整行 NaN 既可能来自非有限的 final hidden，
    也可能来自 fp16 lm_head / kernel 的数值传播。这里只做实测分档。
    """
    prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
    completion = row["completion"]
    tok = backend.ctx["tokenizer"]
    model = backend.ctx["model"]

    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    full = torch.tensor([p_ids + c_ids])
    inputs = backend._to_device({"input_ids": full,
                                 "attention_mask": torch.ones_like(full)})
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    plen = len(p_ids)
    rows_idx = [sl.causal_row_index(plen, j) for j in range(len(c_ids))]
    last_hidden = out.hidden_states[-1][0]
    frozen_logits = out.logits[0]

    counts = {"final_hidden_finite__logits_nonfinite": 0,      # 路线 A 可救
              "final_hidden_nonfinite": 0,                     # 路线 A 无效
              "both_finite": 0,
              "final_hidden_nonfinite__logits_finite": 0}
    for i in rows_idx:
        h_ok = bool(torch.isfinite(last_hidden[i]).all())
        l_ok = bool(torch.isfinite(frozen_logits[i]).all())
        if h_ok and l_ok:
            counts["both_finite"] += 1
        elif h_ok and not l_ok:
            counts["final_hidden_finite__logits_nonfinite"] += 1
        elif (not h_ok) and l_ok:
            counts["final_hidden_nonfinite__logits_finite"] += 1
        else:
            counts["final_hidden_nonfinite"] += 1

    # 冻结层（E3 / H 用）的有限性 —— 应当全有限（H1 全量验收已核过）
    frozen_layer = out.hidden_states[backend.tuple_idx][0]
    layer_finite = bool(torch.isfinite(frozen_layer[[plen + p for p in
                                                     repool.union_positions(row)]]).all())

    # 若 final hidden 全有限，试一次 float32 投影（只在这条 trace 的 completion 行上）
    projection = {"attempted": False}
    if counts["final_hidden_nonfinite"] == 0:
        head = sl.Float32Head(model, torch)
        try:
            logits32 = head.project(last_hidden[rows_idx])
            lp32 = sl.log_softmax_rows(logits32, torch)
            stats = sl.row_statistics(lp32, c_ids, torch)
            projection = {"attempted": True, "ok": True,
                          "n_rows": int(logits32.shape[0]),
                          "all_finite": True,
                          "rank_min": int(stats["rank"].min()),
                          "rank_max": int(stats["rank"].max()),
                          "target_logprob_min": float(stats["target_logprob"].min()),
                          "target_logprob_max": float(stats["target_logprob"].max())}
            del logits32, lp32
        except sl.NonFiniteStop as exc:
            projection = {"attempted": True, "ok": False,
                          "stop_reason": str(exc), "diagnostic": exc.diagnostic}

    del out, last_hidden, frozen_logits, frozen_layer
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"unit_id": repool.derive_unit_id(row),
            "n_completion_tokens": len(c_ids),
            "row_counts": counts,
            "frozen_layer_identifier_positions_finite": layer_finite,
            "float32_projection": projection}


def run_step0(model_path: str) -> int:
    assert_outputs_absent([STEP0_OUT])
    analyzable = load_analyzable("h1")

    # 取诊断中**已知含 NaN logits 行**的 trace —— 直接沿用组 I 的 unit_id，确定性
    diag_path = ROOT / "outputs/stage1/audit/f5_numeric_integrity_diagnostic.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    group_i = diag["sample_groups"]["group_i_block_stratified_anchor"]["unit_ids"]
    with_nan = [u["unit_id"] for u in
                diag["class_2_current_environment_diagnostic"]["group_i"]["per_unit"]
                if u["raw_logits_row_classes"]["all_nonfinite"] > 0]
    picked = [u for u in group_i if u in set(with_nan)][:STEP0_N]
    by_uid = {repool.derive_unit_id(r): r for r in analyzable}
    picked = [u for u in picked if u in by_uid]
    print(f"[step0] 取样 {len(picked)} 条（均为诊断确认含 NaN logits 行的 trace）")

    telemetry = start_telemetry(SMOKE / "shadow_step0_gpu_telemetry.csv")
    print(f"[tele ] pid={telemetry[0].pid}")
    backend = None
    try:
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        torch = backend.ctx["torch"]
        model = backend.ctx["model"]

        structure = sl.inspect_lm_head(model)
        print(f"[step0] lm_head: {structure['lm_head_type']} bias={structure['has_bias']} "
              f"dtype={structure['weight_dtype']} tie={structure['tie_word_embeddings']}")

        # final-hidden 口径：用一条最短的 trace 做数值比对
        probe_row = min((by_uid[u] for u in picked),
                        key=lambda r: len(r["completion"]))
        pr = repool.render_prompt(probe_row, backend.ctx["tokenizer"])
        enc = backend._to_device(backend.ctx["tokenizer"](pr, return_tensors="pt"))
        with torch.no_grad():
            probe_out = model(**enc, output_hidden_states=True)
        convention = sl.resolve_final_hidden_convention(model, probe_out, torch)
        print(f"[step0] final-hidden 口径: {convention['verdict']}")
        del probe_out

        results = [step0_probe_unit(backend, by_uid[u], torch) for u in picked]

        agg = {k: 0 for k in results[0]["row_counts"]}
        for r in results:
            for k, v in r["row_counts"].items():
                agg[k] += v
        n_proj_ok = sum(1 for r in results
                        if r["float32_projection"].get("ok") is True)
        n_proj_fail = sum(1 for r in results
                          if r["float32_projection"].get("ok") is False)
        n_units_hidden_nonfinite = sum(1 for r in results
                                       if r["row_counts"]["final_hidden_nonfinite"] > 0)

        if n_units_hidden_nonfinite == 0 and n_proj_fail == 0:
            route, verdict = "A", (
                "final hidden 在全部受检行上有限，且 float32 projection 全部产出有限值 "
                "→ 路线 A 成立：原 8-bit hidden + float32 lm_head/logsumexp")
        elif agg["final_hidden_nonfinite"] == sum(agg.values()):
            route, verdict = "B", (
                "final hidden 在全部受检行上非有限 → 路线 A 无效，"
                "自动转入已预授权的路线 B（两臂对称 BF16 teacher-forced shadow）")
        else:
            route, verdict = "B", (
                "final hidden 有限/非有限**混合**，或 float32 projection 仍失败 → "
                "路线 A 不可全量依赖，转入已预授权的路线 B（两臂对称 BF16）")

        payload = {
            "schema": "stage1_shadow_step0_v1",
            "purpose": "结构与可行性硬闸；**不写正式产物**，不解盲，不读 label",
            "n_units_probed": len(results),
            "sample_note": "全部取自 F5 诊断中已确认含 NaN logits 行的组 I trace，确定性选取",
            "lm_head_structure": structure,
            "final_hidden_convention": convention,
            "aggregate_row_counts": agg,
            "n_units_with_any_nonfinite_final_hidden": n_units_hidden_nonfinite,
            "float32_projection": {"ok_units": n_proj_ok, "failed_units": n_proj_fail},
            "frozen_layer_identifier_positions_all_finite": all(
                r["frozen_layer_identifier_positions_finite"] for r in results),
            "route_decision": route,
            "route_verdict": verdict,
            "route_definitions": {
                "A": ("原 8-bit teacher-forced final hidden 有限 → 原 backend hidden + "
                      "float32 lm_head/logsumexp；H1 shadow 与 E3 可同一次前向完成"),
                "B": ("final hidden 非有限或 stable projection 仍失败 → 两臂完全对称的 "
                      "BF16 teacher-forced 路径；同一模型/tokenizer、既有 completion、"
                      "无生成、不改 population；记为 protocol-deviation numerical "
                      "sensitivity；H1 与 MCQ **必须一起切换**"),
            },
            "e3_unaffected": ("E3 一律仍用原 8-bit、冻结 layer-20 hidden 口径；"
                              "若 stable F5 走路线 B，允许与 E3 分成两次前向"),
            "environment": {"libraries": library_versions(),
                            "platform": platform.platform(),
                            "python": sys.version.split()[0]},
            "source_shas": {
                "h1_completion_records.jsonl": file_sha256(
                    PROD / "h1_completion_records.jsonl"),
                "h1_run_fingerprint.json": file_sha256(PROD / "h1_run_fingerprint.json"),
            },
            "per_unit": results,
        }
        publish_json_exclusive(STEP0_OUT, payload)
        print(f"[step0] 路线判定 = **{route}**")
        print(f"[step0] {verdict}")
        print(f"[out  ] {STEP0_OUT}")
    finally:
        stop_telemetry(telemetry)
        backend = None
    return 0



# ------------------------------------------------------------ route B smoke
ROUTE_B_OUT = SMOKE / "shadow_route_b_smoke.json"

#: 路线 B 的精确形态：**保持 8-bit 权重**（纯 bf16 全量 15 GB，装不下 12 GiB 卡），
#: 只把激活 / 计算 dtype 从 float16 换成 bfloat16 —— bf16 的指数范围与 fp32 相同。
#: 同模型、同 tokenizer、同量化、同 completion、无生成、不改 population。
ROUTE_B_BACKEND = {"load_in_8bit": True, "device_map": "auto",
                   "attn_implementation": "eager", "local_files_only": True,
                   "activation_dtype": "bfloat16"}


def load_route_b_backend(model_path: str) -> dict:
    """路线 B 专用加载器 —— **不改** backends.load_backend，不复用其 dtype。"""
    from recover_attention.generation import backends as gb
    torch = gb.import_torch()
    comp = gb.import_transformers_components()
    tok = comp["AutoTokenizer"].from_pretrained(model_path, local_files_only=True,
                                                trust_remote_code=False)
    qcfg = comp["BitsAndBytesConfig"](load_in_8bit=True)
    model = comp["AutoModelForCausalLM"].from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        quantization_config=qcfg, device_map="auto", attn_implementation="eager",
        dtype=torch.bfloat16)
    model.eval()
    return {"torch": torch, "tokenizer": tok, "model": model,
            "model_path": model_path, "backend": dict(ROUTE_B_BACKEND)}


def run_route_b_smoke(model_path: str) -> int:
    assert_outputs_absent([ROUTE_B_OUT])
    analyzable = load_analyzable("h1")
    by_uid = {repool.derive_unit_id(r): r for r in analyzable}
    step0 = json.loads(STEP0_OUT.read_text(encoding="utf-8"))
    picked = [u["unit_id"] for u in step0["per_unit"]]
    print(f"[rbsmk] 复用步骤 0 的同一批 {len(picked)} 条（确定性，可直接对照）")

    telemetry = start_telemetry(SMOKE / "shadow_route_b_gpu_telemetry.csv")
    ctx = None
    try:
        ctx = load_route_b_backend(model_path)
        torch, model, tok = ctx["torch"], ctx["model"], ctx["tokenizer"]
        print(f"[rbsmk] loaded; cuda_allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB")

        import recover_attention.conditional_increment as ci
        hidden_index = ci.resolve_hidden_index(int(model.config.num_hidden_layers))
        head = sl.Float32Head(model, torch)

        results, stops = [], []
        for uid in picked:
            row = by_uid[uid]
            prompt = repool.render_prompt(row, tok)
            got = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if got != row.get("prompt_hash"):
                stops.append({"unit_id": uid, "stop": "prompt_hash mismatch"})
                break
            p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
            c_ids = tok(row["completion"], add_special_tokens=False)["input_ids"]
            full = torch.tensor([p_ids + c_ids])
            dev = next(model.parameters()).device
            inputs = {"input_ids": full.to(dev),
                      "attention_mask": torch.ones_like(full).to(dev)}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            plen = len(p_ids)
            rows_idx = [sl.causal_row_index(plen, j) for j in range(len(c_ids))]
            last = out.hidden_states[-1][0]
            n_hidden_nonfinite = int((~torch.isfinite(last[rows_idx])).any(dim=-1).sum())
            entry = {"unit_id": uid, "n_completion_tokens": len(c_ids),
                     "n_rows_final_hidden_nonfinite": n_hidden_nonfinite}
            try:
                logits32 = head.project(last[rows_idx])
                lp32 = sl.log_softmax_rows(logits32, torch)
                stats = sl.row_statistics(lp32, c_ids, torch)
                entry.update({"stable_ok": True,
                              "all_finite": True,
                              "n_rows": int(logits32.shape[0]),
                              "rank_eq_1": int((stats["rank"] == 1).sum()),
                              "rank_max": int(stats["rank"].max()),
                              "mean_target_logprob": float(stats["target_logprob"].mean())})
                del logits32, lp32
            except sl.NonFiniteStop as exc:
                entry.update({"stable_ok": False, "stop_reason": str(exc),
                              "diagnostic": exc.diagnostic})
                stops.append(entry)
            results.append(entry)
            del out, last
            torch.cuda.empty_cache()
            if stops:
                break

        n_ok = sum(1 for r in results if r.get("stable_ok"))
        tot_rows = sum(r.get("n_rows", 0) for r in results if r.get("stable_ok"))
        n_rank1 = sum(r.get("rank_eq_1", 0) for r in results if r.get("stable_ok"))
        payload = {
            "schema": "stage1_shadow_route_b_smoke_v1",
            "route": "B",
            "route_form": ("**8-bit 权重 + bfloat16 激活**。纯 bf16 全量约 15 GB，"
                           "装不下 12,227 MiB 显卡；因此只把激活/计算 dtype 从 fp16 "
                           "换成 bf16（bf16 指数范围与 fp32 相同）。"
                           "同模型、同 tokenizer、同量化、同 completion、无生成、不改 population。"),
            "naming": ("产物只能称 stable-logit shadow sensitivity / "
                       "protocol-deviation numerical sensitivity；"
                       "**不得**冒充原预注册 primary，也不是把 bf16 冒充原 backend"),
            "backend": dict(ROUTE_B_BACKEND),
            "hidden_index": hidden_index,
            "n_units": len(results),
            "n_units_stable_ok": n_ok,
            "n_units_stopped": len(stops),
            "total_rows_projected": tot_rows,
            "rank_eq_1_rows": n_rank1,
            "rank_eq_1_share": (n_rank1 / tot_rows) if tot_rows else None,
            "total_final_hidden_nonfinite_rows": sum(
                r["n_rows_final_hidden_nonfinite"] for r in results),
            "stops": stops,
            "environment": {"libraries": library_versions(),
                            "platform": platform.platform()},
            "source_shas": {
                "h1_completion_records.jsonl": file_sha256(
                    PROD / "h1_completion_records.jsonl")},
            "per_unit": results,
        }
        publish_json_exclusive(ROUTE_B_OUT, payload)
        print(f"[rbsmk] stable_ok {n_ok}/{len(results)} ｜ "
              f"final-hidden 非有限行 {payload['total_final_hidden_nonfinite_rows']} ｜ "
              f"rank==1 {n_rank1}/{tot_rows}")
        print(f"[out  ] {ROUTE_B_OUT}")
        return 0 if not stops else 3
    finally:
        stop_telemetry(telemetry)
        ctx = None



# ================================================================ full run
SHADOW_H1 = SHADOW / "h1_shadow_f5.jsonl"
SHADOW_MCQ = SHADOW / "mcq_shadow_f5.jsonl"
SHADOW_MANIFEST = SHADOW / "shadow_manifest.jsonl"
SHADOW_FP = SHADOW / "shadow_fingerprint.json"
SHADOW_REPORT = SHADOW / "shadow_completion_report.json"

REPOOL_DIR = REPOOL / "h1_e3_block_hidden"
REPOOL_MANIFEST = REPOOL / "e3_repool_manifest.jsonl"
REPOOL_REPORT = REPOOL / "e3_repool_report.json"

HIDDEN_DIM = 3584


def durable_append(path: Path, obj: dict) -> None:
    """append-only + flush + fsync。崩溃后已写前缀完整可读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def manifest_done(path: Path, kind: str) -> set:
    """从 append-only manifest 重建已完成集合 —— manifest 是唯一真相。"""
    if not path.exists():
        return set()
    done = set()
    for raw in path.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue          # 崩溃可能截断最后一行；前缀仍然有效
        if ev.get("kind") == kind and ev.get("state") == "done":
            done.add(ev.get("unit_id"))
    return done


def npz_atomic(path: Path, arrays: dict, meta: dict) -> dict:
    """每 trace 一个 NPZ，原子**创建**（os.link），不覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    if tmp.exists():
        raise ScopeViolation(f"陈旧临时文件，拒绝覆盖并保留现场：{tmp}")
    with tmp.open("wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta, ensure_ascii=False,
                                              sort_keys=True)), **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(str(tmp), str(path))
    except FileExistsError as exc:
        raise ScopeViolation(f"目标已存在，拒绝覆盖：{path}") from exc
    except (OSError, AttributeError, NotImplementedError) as exc:
        raise ScopeViolation(
            f"os.link 不可用或失败，fail-closed：{type(exc).__name__}: {exc}") from exc
    os.unlink(tmp)
    blob = path.read_bytes()
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


# ------------------------------------------------------------- shadow 前向
def _teacher_forced_stable(ctx, head, prompt: str, completion: str):
    """稳定路径的 teacher-forced 前向。**不生成**，只读既有 completion。"""
    torch, tok, model = ctx["torch"], ctx["tokenizer"], ctx["model"]
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if not c_ids:
        raise sl.NonFiniteStop("empty completion tokenization",
                               diagnostic={"stage": "tokenize"})
    full = torch.tensor([p_ids + c_ids])
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=full.to(dev),
                    attention_mask=torch.ones_like(full).to(dev),
                    output_hidden_states=True)
    plen = len(p_ids)
    rows_idx = [sl.causal_row_index(plen, j) for j in range(len(c_ids))]
    logits32 = head.project(out.hidden_states[-1][0][rows_idx])
    lp32 = sl.log_softmax_rows(logits32, torch)
    return {"prompt_len": plen, "completion_ids": c_ids,
            "logits32": logits32, "log_probs": lp32, "out": out,
            "offsets": None}


def shadow_h1_unit(ctx, head, row: dict) -> dict:
    torch = ctx["torch"]
    prompt = repool.render_prompt(row, ctx["tokenizer"])
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row.get("prompt_hash"):
        raise ScopeViolation(f"{repool.derive_unit_id(row)}: prompt_hash 不一致")
    tf = _teacher_forced_stable(ctx, head, prompt, row["completion"])
    stats = sl.row_statistics(tf["log_probs"], tf["completion_ids"], torch)
    n_tok = len(tf["completion_ids"])
    pos = repool.union_positions(row)
    stable = shadow_f5.h1_shadow_row(
        row, n_tok=n_tok, target_logprob=stats["target_logprob"],
        entropy=stats["entropy"], rank=stats["rank"], id_positions=pos)
    rec = shadow_f5.shadow_record(row, "h1", stable, provenance={
        "unit_id": repool.derive_unit_id(row),
        "prompt_hash": row["prompt_hash"], "unit_hash": row.get("unit_hash"),
        "source_run_hash": row.get("run_hash"),
        "n_completion_tokens": n_tok, "n_identifier_positions": len(pos),
        "backend": dict(ROUTE_B_BACKEND)})
    del tf, stats
    torch.cuda.empty_cache()
    return rec


def shadow_mcq_unit(ctx, head, row: dict, letter_ids_bare, letter_ids_space) -> dict:
    torch = ctx["torch"]
    prompt = row["rendered_prompt"]
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row.get("prompt_hash"):
        raise ScopeViolation(f"{row['example_id']}: prompt_hash 不一致")
    tf = _teacher_forced_stable(ctx, head, prompt, row["raw_completion"])
    j = int(row["letter_token_index_local"])
    if j >= len(tf["completion_ids"]):
        raise ScopeViolation(f"{row['example_id']}: letter token index 越界")
    if int(tf["completion_ids"][j]) != int(row["letter_token_id"]):
        raise ScopeViolation(
            f"{row['example_id']}: letter_token_id 与重编码不符 "
            f"({tf['completion_ids'][j]} vs {row['letter_token_id']})")
    ids = letter_ids_bare if row["letter_token_form"] == "bare" else letter_ids_space
    logits_row = tf["logits32"][j]
    opts = {lab: float(logits_row[int(tid)]) for lab, tid in ids.items()}
    letter_lp = float(tf["log_probs"][j, int(row["letter_token_id"])])
    stable = shadow_f5.mcq_shadow_row(row, option_logits=opts,
                                      full_logits_row=logits_row.detach().float().cpu(),
                                      letter_logprob=letter_lp)
    rec = shadow_f5.shadow_record(row, "mcq", stable, provenance={
        "example_id": row["example_id"], "prompt_hash": row["prompt_hash"],
        "letter_token_index_local": j, "letter_token_id": int(row["letter_token_id"]),
        "letter_token_form": row["letter_token_form"],
        "source_run_hash": row.get("run_hash"), "backend": dict(ROUTE_B_BACKEND)})
    del tf
    torch.cuda.empty_cache()
    return rec


# ----------------------------------------------------------------- E3 pass
def e3_unit(backend, row: dict, cfg) -> tuple[dict, dict]:
    """原 8-bit / 冻结 layer-20 口径。含逐 trace union-pooling 精确一致断言。"""
    torch = backend.ctx["torch"]
    unit_id = repool.derive_unit_id(row)
    prompt = repool.render_prompt(row, backend.ctx["tokenizer"])
    tf = backend._teacher_forced(prompt, row["completion"])
    positions = repool.union_positions(row)
    pooled_union = repool.pool_union(tf["hidden"], tf["prompt_len"], positions)
    frozen = np.load(PROD / "h1_unit_hidden" / f"{unit_id}.npy")
    diag = repool.anchor_exact_match(pooled_union, frozen)
    if not diag["exact"]:
        raise ScopeViolation(
            f"{unit_id}: union anchor 不一致（union_anchor_mismatch）：{diag}")
    assignment = repool.block_assignment(row, cfg)
    repool.check_block_partition(row, cfg)
    blocks = repool.pool_blocks(tf["hidden"], tf["prompt_len"], assignment)
    arrays = {f"block_{b}": v.astype(np.float32) for b, v in blocks.items()}
    for k, v in arrays.items():
        if (v.shape != (GEOMETRY.hidden_dim,) or v.dtype != np.float32
                or not np.isfinite(v).all()):
            raise ScopeViolation(
                f"{unit_id}/{k}: 向量不满足 float32 ({GEOMETRY.hidden_dim},) 全有限")
    meta = {"trace_id": unit_id, "prompt_hash": row["prompt_hash"],
            "source_fingerprint_id": row.get("run_hash"),
            "token_positions": {str(b): list(p) for b, p in assignment.items()}}
    del tf
    torch.cuda.empty_cache()
    return arrays, {"meta": meta, "assignment": assignment,
                    "position_digest": repool.position_digest(assignment)}


def run_full(model_path: str) -> int:
    import time
    cfg = V3AnalysisConfig()
    h1 = load_analyzable("h1")
    mcq = load_analyzable("mcq")
    src = {"h1_completion_records.jsonl": file_sha256(PROD / "h1_completion_records.jsonl"),
           "mcq_completion_records.jsonl": file_sha256(PROD / "mcq_completion_records.jsonl"),
           "h1_run_fingerprint.json": file_sha256(PROD / "h1_run_fingerprint.json"),
           "mcq_run_fingerprint.json": file_sha256(PROD / "mcq_run_fingerprint.json")}
    print(f"[full ] H1 analyzable {len(h1):,} ｜ MCQ analyzable {len(mcq):,}")

    n_pairs_expected = sum(len(repool.expected_blocks(r, cfg)) for r in h1)
    print(f"[full ] E3 (trace,block) 现场重建 = {n_pairs_expected:,}")

    # ---------------- phase S：两臂 shadow（路线 B backend）----------------
    done_h1 = manifest_done(SHADOW_MANIFEST, "h1")
    done_mcq = manifest_done(SHADOW_MANIFEST, "mcq")
    todo_h1 = [r for r in h1 if repool.derive_unit_id(r) not in done_h1]
    todo_mcq = [r for r in mcq if r["example_id"] not in done_mcq]
    print(f"[phaseS] 待跑 H1 {len(todo_h1):,} ｜ MCQ {len(todo_mcq):,}"
          f"（已完成 {len(done_h1):,}/{len(done_mcq):,}）")

    telemetry = None
    if todo_h1 or todo_mcq:
        telemetry = start_telemetry(SHADOW / "shadow_seg001_gpu_telemetry.csv"
                                    if not (SHADOW / "shadow_seg001_gpu_telemetry.csv").exists()
                                    else SHADOW / f"shadow_seg{len(list(SHADOW.glob('shadow_seg*'))) + 1:03d}_gpu_telemetry.csv")
        ctx = load_route_b_backend(model_path)
        torch = ctx["torch"]
        head = sl.Float32Head(ctx["model"], torch, cache_f32_weight=True)
        print(f"[phaseS] projection cache_mode={head.cache_mode} ｜ "
              f"cuda={torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB")
        import recover_attention.mcq_conditional_increment as mci
        bare = dlp.bare_option_token_ids(ctx["tokenizer"], list(mci.MCQ_LETTERS))
        space = dlp.option_token_ids(ctx["tokenizer"], list(mci.MCQ_LETTERS))
        try:
            t0 = time.time()
            for i, row in enumerate(todo_h1, 1):
                rec = shadow_h1_unit(ctx, head, row)
                durable_append(SHADOW_H1, rec)
                durable_append(SHADOW_MANIFEST,
                               {"kind": "h1", "unit_id": rec["trace_id"], "state": "done"})
                if i % 500 == 0:
                    print(f"[phaseS] H1 {i}/{len(todo_h1)} "
                          f"({(time.time() - t0) / i:.3f} s/trace)")
            t1 = time.time()
            for i, row in enumerate(todo_mcq, 1):
                rec = shadow_mcq_unit(ctx, head, row, bare, space)
                durable_append(SHADOW_MCQ, rec)
                durable_append(SHADOW_MANIFEST,
                               {"kind": "mcq", "unit_id": row["example_id"], "state": "done"})
                if i % 500 == 0:
                    print(f"[phaseS] MCQ {i}/{len(todo_mcq)} "
                          f"({(time.time() - t1) / i:.3f} s/unit)")
        finally:
            del head
            ctx = None
            if telemetry is not None:
                stop_telemetry(telemetry)
            telemetry = None
    print("[phaseS] done")

    # ---------------- phase E：E3 repool（原 8-bit 冻结口径）---------------
    done_e3 = manifest_done(REPOOL_MANIFEST, "e3")
    todo_e3 = [r for r in h1 if repool.derive_unit_id(r) not in done_e3]
    print(f"[phaseE] 待跑 {len(todo_e3):,}（已完成 {len(done_e3):,}）")
    if todo_e3:
        telemetry = start_telemetry(
            REPOOL / f"e3_seg{len(list(REPOOL.glob('e3_seg*'))) + 1:03d}_gpu_telemetry.csv")
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        try:
            t0 = time.time()
            for i, row in enumerate(todo_e3, 1):
                uid = repool.derive_unit_id(row)
                arrays, info = e3_unit(backend, row, cfg)
                digest = npz_atomic(REPOOL_DIR / f"{uid}.npz", arrays, info["meta"])
                durable_append(REPOOL_MANIFEST, {
                    "kind": "e3", "unit_id": uid, "state": "done",
                    "block_ids": sorted(info["assignment"]),
                    "n_blocks": len(info["assignment"]),
                    "position_digest": info["position_digest"], **digest})
                if i % 500 == 0:
                    print(f"[phaseE] {i}/{len(todo_e3)} "
                          f"({(time.time() - t0) / i:.3f} s/trace)")
        finally:
            backend = None
            if telemetry is not None:
                stop_telemetry(telemetry)
    print("[phaseE] done")

    # ---------------- 指纹与完成报告 -------------------------------------
    fp_payload = {
        "schema": "stage1_shadow_repool_fingerprint_v1",
        "shadow_route": "B",
        "shadow_backend": dict(ROUTE_B_BACKEND),
        "shadow_backend_caveat": (
            "bitsandbytes MatMul8bitLt 会把激活从 bfloat16 内部降回 float16 做量化 —— "
            "**不得**声称全程 bf16；实测该路径下 final hidden 全有限"),
        "e3_backend": "原 8-bit fp16 冻结口径（未改动），冻结 layer-20 hidden",
        "projection_schema": sl.STABLE_PROJECTION_SCHEMA,
        "shadow_schema": shadow_f5.SHADOW_SCHEMA,
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "deviation_class": shadow_f5.DEVIATION_LABEL,
        "not_primary": "原 14 项 H1 B4 仍是 frozen implementation primary，保留不覆盖不作废",
        "source_shas": src,
        "e3_config": {"e3_n_blocks": cfg.e3_n_blocks, "e3_block_salt": cfg.e3_block_salt,
                      "pooling_schema": repool.POOLING_SCHEMA,
                      "token_position_selection_rule": repool.TOKEN_POSITION_SELECTION_RULE},
        "libraries": library_versions(),
        "numeric_dtypes": {"projection": "float32", "pooled_hidden": "float32",
                           "activations": "bfloat16 (int8 matmul 内部转 fp16)"},
        "platform": platform.platform(),
    }
    if not SHADOW_FP.exists():
        publish_json_exclusive(SHADOW_FP, fp_payload)

    n_h1_rows = sum(1 for x in SHADOW_H1.read_bytes().split(b"\n") if x.strip())
    n_mcq_rows = sum(1 for x in SHADOW_MCQ.read_bytes().split(b"\n") if x.strip())
    n_npz = len(list(REPOOL_DIR.glob("*.npz")))
    pairs = 0
    for raw in REPOOL_MANIFEST.read_bytes().split(b"\n"):
        if raw.strip():
            pairs += json.loads(raw).get("n_blocks", 0)
    report = {
        "schema": "stage1_shadow_repool_completion_report_v1",
        "interpretation": shadow_f5.SENSITIVITY_LABEL,
        "h1_shadow_rows": n_h1_rows, "h1_expected": len(h1),
        "mcq_shadow_rows": n_mcq_rows, "mcq_expected": len(mcq),
        "e3_npz_files": n_npz, "e3_expected_traces": len(h1),
        "e3_trace_block_pairs": pairs, "e3_expected_pairs": n_pairs_expected,
        "complete": (n_h1_rows == len(h1) and n_mcq_rows == len(mcq)
                     and n_npz == len(h1) and pairs == n_pairs_expected),
        "source_shas": src,
        "downstream_contract": ("本报告缺失或 complete=false 时，下游**必须拒绝读取** "
                               "shadow / repool 产物"),
    }
    if report["complete"] and not REPOOL_REPORT.exists():
        publish_json_exclusive(REPOOL_REPORT, report)
    if report["complete"] and not SHADOW_REPORT.exists():
        publish_json_exclusive(SHADOW_REPORT, report)
    print(f"[report] {json.dumps({k: v for k, v in report.items() if k != 'source_shas'}, ensure_ascii=False)}")
    return 0 if report["complete"] else 3


def set_artifact_root(root, *, geometry=None) -> dict:
    """整体重定向产物目录。只允许在 CLI 启动时、任何 IO 之前调用一次。"""
    global ARTIFACT_ROOT, PROD, SHADOW, REPOOL, SMOKE, GEOMETRY, _ROOT_REBOUND
    global STEP0_OUT, SHADOW_H1, SHADOW_MCQ, SHADOW_MANIFEST, SHADOW_FP, SHADOW_REPORT
    global REPOOL_DIR, REPOOL_MANIFEST, REPOOL_REPORT
    if _ROOT_REBOUND:
        raise ScopeViolation("set_artifact_root 只能调用一次")
    ARTIFACT_ROOT = Path(root)
    PROD = ARTIFACT_ROOT / "production"
    SHADOW = ARTIFACT_ROOT / "shadow"
    REPOOL = ARTIFACT_ROOT / "repool"
    SMOKE = ARTIFACT_ROOT / "smoke"
    STEP0_OUT = SMOKE / "shadow_smoke_report.json"
    SHADOW_H1 = SHADOW / "h1_shadow_f5.jsonl"
    SHADOW_MCQ = SHADOW / "mcq_shadow_f5.jsonl"
    SHADOW_MANIFEST = SHADOW / "shadow_manifest.jsonl"
    SHADOW_FP = SHADOW / "shadow_fingerprint.json"
    SHADOW_REPORT = SHADOW / "shadow_completion_report.json"
    REPOOL_DIR = REPOOL / "h1_e3_block_hidden"
    REPOOL_MANIFEST = REPOOL / "e3_repool_manifest.jsonl"
    REPOOL_REPORT = REPOOL / "e3_repool_report.json"
    if geometry is not None:
        GEOMETRY = geometry
    _ROOT_REBOUND = True
    return {"artifact_root": str(ARTIFACT_ROOT), "geometry": GEOMETRY.as_dict()}


REPOOL_ONLY_SCHEMA = "stage1_e3_canonical_repool_completion_report_v1"


def run_repool_only(model_path: str) -> int:
    """**只**生成 canonical E3 repool。绝不进入已被取代的 full-BF16 shadow 路径。

    与 `--full` 的差别是结构性的，不是开关：本函数体内没有 shadow_h1_unit /
    shadow_mcq_unit / load_route_b_backend 的任何调用，也不写 SHADOW_* 产物
    （有测试逐符号断言）。E3 用**原 8-bit 冻结口径**，与 Path B 无关 ——
    E3 是独立 estimand。

    断点续跑沿用既有机制：manifest_done 读已完成集合，durable_append
    （flush + fsync）只追加，npz_atomic 原子落盘。
    """
    import time
    cfg = V3AnalysisConfig()
    h1 = load_analyzable("h1")
    n_pairs_expected = sum(len(repool.expected_blocks(r, cfg)) for r in h1)
    src = {"h1_completion_records.jsonl": file_sha256(PROD / "h1_completion_records.jsonl"),
           "h1_run_fingerprint.json": file_sha256(PROD / "h1_run_fingerprint.json")}
    print(f"[repool] artifact_root={ARTIFACT_ROOT}")
    print(f"[repool] geometry block={GEOMETRY.block_index} tuple={GEOMETRY.tuple_index} "
          f"dim={GEOMETRY.hidden_dim} source={GEOMETRY.source}")
    print(f"[repool] H1 analyzable {len(h1):,} | E3 (trace,block) {n_pairs_expected:,}")

    # 既有 report 必须与本次输入一致，否则拒绝 —— 换了输入还续跑等于混两次运行。
    if REPOOL_REPORT.exists():
        old = json.loads(REPOOL_REPORT.read_text(encoding="utf-8"))
        if old.get("source_shas") != src:
            raise ScopeViolation(
                f"既有 {REPOOL_REPORT} 的 source_shas 与本次输入不符 —— "
                f"拒绝跨配置续跑，请另用独立 artifact root")

    done_e3 = manifest_done(REPOOL_MANIFEST, "e3")
    todo_e3 = [r for r in h1 if repool.derive_unit_id(r) not in done_e3]
    print(f"[repool] 待跑 {len(todo_e3):,}（已完成 {len(done_e3):,}）")

    if todo_e3:
        REPOOL.mkdir(parents=True, exist_ok=True)
        REPOOL_DIR.mkdir(parents=True, exist_ok=True)
        seg_n = len(list(REPOOL.glob("e3_seg*"))) + 1
        telemetry = start_telemetry(REPOOL / f"e3_seg{seg_n:03d}_gpu_telemetry.csv")
        from recover_attention.generation import backends as gb
        backend = gb._RealBackendBase(model_path)
        try:
            t0 = time.time()
            for i, row in enumerate(todo_e3, 1):
                uid = repool.derive_unit_id(row)
                arrays, info = e3_unit(backend, row, cfg)
                digest = npz_atomic(REPOOL_DIR / f"{uid}.npz", arrays, info["meta"])
                durable_append(REPOOL_MANIFEST, {
                    "kind": "e3", "unit_id": uid, "state": "done",
                    "block_ids": sorted(info["assignment"]),
                    "n_blocks": len(info["assignment"]),
                    "position_digest": info["position_digest"], **digest})
                if i % 500 == 0:
                    print(f"[repool] {i}/{len(todo_e3)} "
                          f"({(time.time() - t0) / i:.3f} s/trace)")
        finally:
            backend = None
            if telemetry is not None:
                stop_telemetry(telemetry)

    # ---- 完整性与 fingerprint 校验 --------------------------------------
    n_npz = len(list(REPOOL_DIR.glob("*.npz"))) if REPOOL_DIR.exists() else 0
    pairs, seen = 0, set()
    raw_lines = (REPOOL_MANIFEST.read_bytes().split(b"\n")
                 if REPOOL_MANIFEST.exists() else [])
    for raw in raw_lines:
        if raw.strip():
            ev = json.loads(raw)
            pairs += ev.get("n_blocks", 0)
            seen.add(ev["unit_id"])
    want_ids = {repool.derive_unit_id(r) for r in h1}
    report = {
        "schema": REPOOL_ONLY_SCHEMA,
        "mode": "repool_only",
        "writes_full_bf16_shadow": False,
        "artifact_root": str(ARTIFACT_ROOT),
        "geometry": GEOMETRY.as_dict(),
        "e3_backend": "原 8-bit fp16 冻结口径（未改动）",
        "e3_npz_files": n_npz, "e3_expected_traces": len(h1),
        "e3_trace_block_pairs": pairs, "e3_expected_pairs": n_pairs_expected,
        "manifest_unit_ids_match_expected": seen == want_ids,
        "n_missing_units": len(want_ids - seen), "n_extra_units": len(seen - want_ids),
        "e3_config": {"e3_n_blocks": cfg.e3_n_blocks, "e3_block_salt": cfg.e3_block_salt,
                      "pooling_schema": repool.POOLING_SCHEMA,
                      "token_position_selection_rule":
                          repool.TOKEN_POSITION_SELECTION_RULE},
        "libraries": library_versions(),
        "platform": platform.platform(),
        "source_shas": src,
        "complete": (n_npz == len(h1) and pairs == n_pairs_expected
                     and seen == want_ids),
        "downstream_contract": ("本报告缺失或 complete=false 时，下游**必须拒绝读取** "
                                "canonical repool 产物"),
    }
    if report["complete"] and not REPOOL_REPORT.exists():
        publish_json_exclusive(REPOOL_REPORT, report)
    shown = {k: v for k, v in report.items() if k != "source_shas"}
    print(f"[report] {json.dumps(shown, ensure_ascii=False)}")
    return 0 if report["complete"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--route-b-smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--repool-only", action="store_true",
                    help="只跑 canonical E3 repool（跨模型复现用）；"
                         "**不**生成已被取代的 full-BF16 shadow")
    ap.add_argument("--model-path", default="D:/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--artifact-root", type=Path, default=None,
                    help="独立 artifact root；省略即默认 outputs/stage1")
    repool.add_geometry_cli_args(ap)
    args = ap.parse_args(argv)
    if sum(bool(x) for x in (args.step0, args.route_b_smoke, args.full,
                             args.repool_only)) > 1:
        ap.error("--step0 / --route-b-smoke / --full / --repool-only 互斥")
    geo = repool.geometry_from_cli(args, model_path=args.model_path)
    if args.artifact_root is not None or geo is not repool.QWEN_GEOMETRY:
        info = set_artifact_root(args.artifact_root or DEFAULT_ARTIFACT_ROOT,
                                 geometry=geo)
        print(f"[root ] {info['artifact_root']}")
    if args.step0:
        return run_step0(args.model_path)
    if args.route_b_smoke:
        return run_route_b_smoke(args.model_path)
    if args.full:
        return run_full(args.model_path)
    if args.repool_only:
        return run_repool_only(args.model_path)
    ap.error("选择 --step0 / --route-b-smoke / --full / --repool-only")


if __name__ == "__main__":
    raise SystemExit(main())
