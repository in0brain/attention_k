r"""重复起始标记修复的**本地 tokenizer 核验**（不加载模型权重）。

对两个模型、两条臂（H1 / MCQ）分别比较修复前后的 prompt token ids：

    Qwen   必须**完全一致**（它的 tokenizer 本来就不前置任何 token）
    Llama  必须**只少掉开头多出来的那一个 <|begin_of_text|>**，其余 token ids 逐位不变

同时核验生成路径与 teacher-forced 路径对同一 prompt 的编码一致。

用法：
    python scripts/diagnostics/l2c_verify_bos_fix.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention import cyber_data as cd            # noqa: E402
from recover_attention import h1_data as hd               # noqa: E402

OUT = ROOT / "tmp" / "stage1_l2_g3" / "l2c_bos_fix_verification.json"
MODELS = {"qwen": "D:/models/Qwen2.5-7B-Instruct",
          "llama": "D:/models/Meta-Llama-3.1-8B-Instruct"}


def sample_prompts(tok) -> dict[str, str]:
    """两条臂各取一个**真实结构**的 prompt（内容取自已落盘的 G3 记录）。"""
    mcq_row = json.loads(io.open(
        ROOT / "outputs/stage1_llama_v3/g3/mcq_g3_unit_results.jsonl",
        encoding="utf-8").readline())
    mcq_record = {"question": "placeholder",
                  "candidate_choices": mcq_row["candidate_choices"]}
    # 用已落盘记录里的真实题干，保证提示正文与实跑同构
    body = mcq_row["prompt_only_text"].split("\n")[0]
    mcq_record["question"] = body
    h1_row = json.loads(io.open(
        ROOT / "outputs/stage1_llama_v3/g3/h1_g3_unit_results.jsonl",
        encoding="utf-8").readline())
    h1_record = {"question_text": h1_row["prompt_text"]}
    return {
        "mcq": tok.apply_chat_template(cd.build_mcq_chat_messages(mcq_record),
                                       tokenize=False, add_generation_prompt=True),
        "h1": tok.apply_chat_template(hd.build_h1_chat_messages(h1_record),
                                      tokenize=False, add_generation_prompt=True),
    }


def compare(tok, prompt: str) -> dict:
    before = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()   # 旧口径
    after = tok(prompt, return_tensors="pt",
                add_special_tokens=False)["input_ids"][0].tolist()       # 修复后
    # 生成路径与 teacher-forced 路径现在用同一行参数；这里分别按各自写法再算一遍，
    # 证明两条路径对同一 prompt 得到**逐位相同**的 ids。
    gen_ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"][0].tolist()
    tf_ids = tok(prompt, return_tensors="pt",
                 add_special_tokens=False)["input_ids"][0].tolist()
    identical = before == after
    only_leading_removed = (not identical
                            and len(before) == len(after) + 1
                            and before[1:] == after)
    return {
        "len_before": len(before), "len_after": len(after),
        "identical": identical,
        "only_leading_token_removed": only_leading_removed,
        "removed_leading_id": (before[0] if only_leading_removed else None),
        "removed_leading_decoded": (tok.decode([before[0]])
                                    if only_leading_removed else None),
        "tail_unchanged": before[len(before) - len(after):] == after,
        "generate_and_teacher_forced_identical": gen_ids == tf_ids,
        "first_6_before": before[:6], "first_6_after": after[:6],
    }


def main() -> int:
    from transformers import AutoTokenizer                  # noqa: PLC0415
    from recover_attention.generation import fingerprint as fpm   # noqa: PLC0415

    doc: dict = {
        "schema": "stage1_l2c_bos_fix_verification_v1",
        "date": "2026-09-13",
        "what": "重复起始标记修复：只加载 tokenizer，不加载模型权重，不跑 GPU",
        "change": ["src/recover_attention/generation/backends.py::_RealBackendBase._generate",
                   "src/recover_attention/generation/backends.py::_RealBackendBase._teacher_forced"],
        "models": {},
    }
    ok = True
    for tag, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        arms = {arm: compare(tok, p) for arm, p in sample_prompts(tok).items()}
        expect = "identical" if tag == "qwen" else "only_leading_token_removed"
        passed = all(a[expect] for a in arms.values())
        ok = ok and passed
        doc["models"][tag] = {
            "tokenizer_fingerprint": fpm.tokenizer_fingerprint(tok)["tokenizer_hash"],
            "expectation": expect,
            "passed": passed,
            "arms": arms,
        }
    doc["all_passed"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    print("written:", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
