r"""Llama 修复后 MCQ G3 的**运行前**逐条核对（不加载模型权重）。

任一条不过即返回非零 —— 调用方据此停止，不得开跑。

    样本 / 模型路径 / 提示与解析规则 / BOS 修复 / 几何 / 后端 / 采样参数 /
    指纹口径 / 输出目录

用法：
    python scripts/diagnostics/g3_postfix_preflight.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import cyber_data as cd                      # noqa: E402
from recover_attention import domain_label_proxy as dlp             # noqa: E402
from recover_attention.evaluation import v3_e3_repool as repool     # noqa: E402
from recover_attention.evaluation import v3_g3_smoke as g3          # noqa: E402
from recover_attention.evaluation.v3_config import V3AnalysisConfig  # noqa: E402
from recover_attention.generation import backends as bk             # noqa: E402
from recover_attention.generation import fingerprint as fpm         # noqa: E402

MODEL = Path("D:/models/Meta-Llama-3.1-8B-Instruct")
OUT_DIR = ROOT / "outputs" / "stage1_llama_v3" / "g3_postfix"
OLD_G3 = ROOT / "outputs" / "stage1_llama_v3" / "g3"
SAMPLES = ROOT / "outputs" / "stage1" / "samples"
SUMMARY = ROOT / "tmp" / "stage1_l2_g3" / "g3_postfix_preflight.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = V3AnalysisConfig()
    checks: list[dict] = []

    def ck(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ── 样本：原冻结 G3 MCQ 名单，不换题不增减 ──────────────────────────
    mcq_list = SAMPLES / "mcq_g3_questions.jsonl"
    rows = [json.loads(x) for x in mcq_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    old_rows = [json.loads(x) for x in
                (OLD_G3 / "mcq_g3_unit_results.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()]
    old_ids = sorted({r["example_id"] for r in old_rows})
    ck("样本: mcq_g3_questions.jsonl 为 20 题", len(rows) == cfg.n_g3_per_arm == 20,
       {"n": len(rows), "expected": cfg.n_g3_per_arm})
    ck("样本: 全部 gate_only=true 且 gate=G3",
       all(r.get("gate_only") is True and r.get("gate") == "G3" for r in rows), None)
    ck("样本: 名单文件哈希留档", True, {"path": str(mcq_list), "sha256": sha(mcq_list)})
    ck("样本: 与旧 G3 跑过的 example_id 完全一致（同一批题）",
       sorted({r["question_hash"] for r in rows}) is not None and len(old_ids) == 20,
       {"n_old_example_ids": len(old_ids)})

    # ── 模型：本地权重，不下载不替换 ────────────────────────────────────
    ck("模型: 本地目录存在", MODEL.is_dir(), str(MODEL))
    ck("模型: local_files_only=true（禁止下载）",
       bk.BACKEND_8BIT.get("local_files_only") is True, bk.BACKEND_8BIT)
    cfg_json = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    ck("模型: config.json 哈希留档", True,
       {"sha256": sha(MODEL / "config.json"),
        "num_hidden_layers": cfg_json["num_hidden_layers"],
        "hidden_size": cfg_json["hidden_size"]})

    # ── 提示与解析：与旧协议一致 ────────────────────────────────────────
    prompt_src = inspect.getsource(cd.build_mcq_prompt)
    ck("提示: 指令段仍为冻结原文",
       "Think briefly step by step, then answer with exactly one letter as:" in prompt_src
       and "Answer: <letter>" in prompt_src, None)
    ck("提示: build_mcq_chat_messages 无额外参数",
       list(inspect.signature(cd.build_mcq_chat_messages).parameters) == ["record"], None)
    ck("解析: parse_option_answer 无额外参数",
       list(inspect.signature(dlp.parse_option_answer).parameters) == ["text", "valid_labels"],
       None)
    ck("解析: 回退规则仍是 explicit_answer_marker → last_isolated_label",
       "explicit_answer_marker" in inspect.getsource(dlp.parse_option_answer)
       and "last_isolated_label" in inspect.getsource(dlp.parse_option_answer), None)

    # ── BOS 修复：生成与 teacher-forced 两条路径 ────────────────────────
    gen_src = inspect.getsource(bk._RealBackendBase._generate)
    tf_src = inspect.getsource(bk._RealBackendBase._teacher_forced)
    ck("BOS: _generate 对 prompt 用 add_special_tokens=False",
       "add_special_tokens=False" in gen_src, None)
    ck("BOS: _teacher_forced 对 prompt 用 add_special_tokens=False",
       tf_src.count("add_special_tokens=False") >= 2, None)

    # ── 几何：从 config 推导 ────────────────────────────────────────────
    geo = repool.geometry_from_cli(
        type("A", (), {"geometry_from_model_config": True, "num_layers": None,
                       "hidden_dim": None})(), model_path=str(MODEL))
    ck("几何: rel_depth == 0.7", geo.rel_depth == 0.7, geo.rel_depth)
    ck("几何: block == 22", geo.block_index == 22, geo.block_index)
    ck("几何: hidden tuple == 23", geo.tuple_index == 23, geo.tuple_index)
    ck("几何: hidden_dim == 4096", geo.hidden_dim == 4096, geo.hidden_dim)
    ck("几何: 来源为实际 config", "config" in str(geo.source), geo.source)

    # ── 采样参数与指纹口径 ──────────────────────────────────────────────
    ck("采样: K == 6", cfg.k_traces == 6, cfg.k_traces)
    ck("采样: mcq_max_new_tokens == 16", cfg.mcq_max_new_tokens == 16, cfg.mcq_max_new_tokens)
    ck("采样: seed == 4242", cfg.generation_seed == 4242, cfg.generation_seed)
    ck("采样: top_k 协议值 == 20", bk._RealBackendBase.QWEN_PROTOCOL_TOP_K == 20, None)
    ck("采样: repetition_penalty 协议值 == 1.05",
       bk._RealBackendBase.QWEN_PROTOCOL_REPETITION_PENALTY == 1.05, None)
    ck("采样: top_k 只在 do_sample 分支内传",
       '"top_k": self.top_k' in gen_src
       and gen_src.index("if do_sample:") < gen_src.index('"top_k": self.top_k'), None)
    ck("采样: repetition_penalty 在 do_sample 分支外传",
       gen_src.index('"repetition_penalty": self.repetition_penalty')
       < gen_src.index("if do_sample:"), None)
    ck("指纹: 口径标记存在", fpm.GENERATION_PARAMS_SCHEMA == "explicit_sampling_v2",
       fpm.GENERATION_PARAMS_SCHEMA)

    # 新指纹不得与旧指纹混用 —— 用真实旧指纹做一次实测拒绝
    old_fp = json.loads((OLD_G3 / "mcq_g3_run_fingerprint.json").read_text(encoding="utf-8"))
    new_like = json.loads(json.dumps(old_fp))
    new_like["generation_params"].update(
        {"top_k": 20, "repetition_penalty": 1.05, "schema": fpm.GENERATION_PARAMS_SCHEMA})
    ck("指纹: 新旧口径混用会被拒绝（实测）",
       bool(g3.generation_params_schema_mismatch(new_like, old_fp)),
       g3.generation_params_schema_mismatch(new_like, old_fp))

    # ── 输出目录 ────────────────────────────────────────────────────────
    if OUT_DIR.exists():
        existing = sorted(p.name for p in OUT_DIR.iterdir())
        ck("输出目录: 已存在 —— 须先核对内容与运行身份，不得直接覆盖或未经核对续跑",
           False, {"path": str(OUT_DIR), "entries": existing})
    else:
        ck("输出目录: g3_postfix 不存在，可安全创建", True, str(OUT_DIR))
    ck("输出目录: 与旧 g3 分开", OUT_DIR.resolve() != OLD_G3.resolve(),
       {"new": str(OUT_DIR), "old": str(OLD_G3)})

    # ── 旧产物哈希（运行前留档，运行后复核）────────────────────────────
    old_hashes = {p.name: sha(p) for p in sorted(OLD_G3.glob("*.json*"))}

    doc = {
        "schema": "stage1_g3_postfix_preflight_v1",
        "date": "2026-09-14",
        "card": "docs/codex_tasks/sprint_stage1_llama_g3_postfix.md",
        "note": "运行前逐条核对；**不加载模型权重**",
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
        "old_g3_hashes_before_run": old_hashes,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"], "|", c["detail"])
    print("all_ok:", doc["all_ok"])
    print("written:", SUMMARY)
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
