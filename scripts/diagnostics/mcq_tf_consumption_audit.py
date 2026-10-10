r"""静态 + 动态追踪：MCQ 打分实际消费 `_teacher_forced` 的哪些输出。

目的只有一个 —— 找出**算了但没人用**的部分，为"只减少未使用计算"的最小优化提供依据。
**不加载模型权重，不用 GPU**：用一个记录键访问的假 `_teacher_forced` 返回值，
跑真实的 `MCQBackend.score_completion` / `H1Backend` 消费代码，看它到底取了什么。

用法：
    python scripts/diagnostics/mcq_tf_consumption_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recover_attention.generation import backends as bk              # noqa: E402

OUT = ROOT / "tmp/stage1_l2_g3/mcq_tf_consumption_audit.json"

V = 64          # 假词表
D = 8           # 假 hidden 维
LETTERS = ["A", "B", "C", "D"]


class RecordingTF(dict):
    """记录被 `__getitem__` 取过的键；对张量型值再记录被索引的下标数量。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.touched: list[str] = []

    def __getitem__(self, k):
        if k not in self.touched:
            self.touched.append(k)
        return super().__getitem__(k)


def _fake_tf(prompt_len: int, completion: str, label: str) -> RecordingTF:
    """每个字符一个 token —— 字母的 1 字符区间必然只落在单个 token 上。"""
    n = len(completion)
    c_ids = [ord(ch) % V for ch in completion]
    # 每个出现的选项字母都换成"裸形"候选 id，好让 _letter_id_set 认出来
    for i, ch in enumerate(completion):
        if ch in LETTERS:
            c_ids[i] = LETTERS.index(ch)
    offsets = [[i, i + 1] for i in range(n)]
    T = prompt_len + n
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((T, V)).astype(np.float32)
    return RecordingTF({
        "prompt_len": prompt_len,
        "completion_ids": c_ids,
        "offsets": offsets,
        "token_logprobs": {i: float(-i) for i in range(n)},
        "token_entropies": {i: float(i) for i in range(n)},
        "first_token_ranks": {i: i + 1 for i in range(n)},
        "hidden": rng.standard_normal((T, D)).astype(np.float32),
        "logits": _Tensorish(logits),
    })


class _Tensorish:
    """只支持 `x[step].cpu().numpy()` 的最小替身（score_completion 对 logits 就这一种用法）。"""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.rows_taken: list[int] = []

    def __getitem__(self, i):
        self.rows_taken.append(int(i))
        return _Row(self.arr[i])


class _Row:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def main() -> int:
    src = ROOT / "outputs/stage1_llama_v3/g3_budget512_reparsed/mcq_g3_unit_results.jsonl"
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("eligible_for_primary")][:3]

    be = object.__new__(bk.MCQBackend)
    be.seed, be.temperature, be.top_p, be.max_new_tokens = 4242, 0.7, 0.95, 512
    be.letter_ids_bare = {L: i for i, L in enumerate(LETTERS)}
    be.letter_ids_space = {L: 32 + i for i, L in enumerate(LETTERS)}

    results = []
    for r in rows:
        tf = _fake_tf(17, r["raw_completion"], r["parsed_label"])
        be._teacher_forced = lambda *a, **k: tf                       # noqa: ARG005
        rec = {"example_id": r["example_id"], "rendered_prompt": r["rendered_prompt"],
               "candidate_choices": r["candidate_choices"], "question": "",
               "gold_label": r["parsed_label"], "group_id": r.get("group_id")}
        unit = {"sample_type": r["sample_type"], "sample_index": r["sample_index"]}
        out = be.score_completion(r["rendered_prompt"], r["raw_completion"], rec, unit,
                                  seed=1)
        n_tok = len(tf["completion_ids"])
        results.append({
            "example_id": r["example_id"],
            "n_completion_tokens_fake": n_tok,
            "tf_keys_touched": sorted(tf.touched),
            "tf_keys_never_touched": sorted(set(tf) - set(tf.touched)),
            "logits_rows_taken": tf["logits"].rows_taken,
            "letter_token_index_local": out.get("letter_token_index_local"),
            "produced_f5": sorted(k for k in out if k.startswith("f5_")),
        })

    never = sorted(set.intersection(*(set(x["tf_keys_never_touched"]) for x in results)))
    # `token_logprobs` 虽被取，但只取了一个下标 —— 单独标注
    doc = {
        "schema": "mcq_tf_consumption_audit_v1", "date": "2026-09-15",
        "what": ("MCQ 打分对 _teacher_forced 输出的实际消费。假返回值 + 真实 "
                 "score_completion；不加载权重、不用 GPU。"),
        "source_rows": str(src.relative_to(ROOT)).replace("\\", "/"),
        "per_row": results,
        "never_touched_by_mcq": never,
        "touched_but_only_one_index": {
            "token_logprobs": "只取 tf['token_logprobs'][j]（答案字母那一个 token）",
            "completion_ids": "只取 tf['completion_ids'][j]",
            "logits": "只取第 step=prompt_len+j-1 **一行**",
            "hidden": "只取第 prompt_len+j **一行**",
            "offsets": "整表都要用（token_indices_for_char_span 扫描）",
        },
        "unused_computation_in_teacher_forced_for_mcq": [
            "逐 token 的 token_entropies：MCQ 从不读取（全部 N 个 token 白算）",
            "逐 token 的 first_token_ranks：MCQ 从不读取（全部 N 个 token 白算）",
            "逐 token 的 token_logprobs：MCQ 只用 1 个，其余 N-1 个白算",
            "→ 三者同在 _teacher_forced 的那个 for 循环里，每 token 3 次 128k 词表规约 "
            "+ 3 次 GPU→CPU 同步",
        ],
        "h1_note": ("H1 用满 token_logprobs（序列级 F5），但 token_entropies / "
                    "first_token_ranks 只在 identifier 位置用（mention_logprob_features(pos)）。"
                    "**本轮不改 H1** —— H1 正在跑。"),
        "candidate_minimal_optimization": {
            "what": "MCQ 路径跳过整个逐 token 循环，只对 j 这一个 token 算 logprob",
            "why_bit_exact": ("取的是同一个 lp 张量的同一个下标 "
                              "float(lp[prompt_len+j-1][c_ids[j]])，表达式一字不改"),
            "not_touched": "log_softmax / .float() / hidden 的整表搬运一律不动，避免换 kernel",
            "h1_unchanged": "新参数默认给全量统计；H1 不传该参数，行为逐位不变",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for x in results:
        print(f"  {x['example_id']}  touched={x['tf_keys_touched']}")
        print(f"      never={x['tf_keys_never_touched']}  logits_rows={x['logits_rows_taken']}")
    print(f"\n[never touched by MCQ] {never}")
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
