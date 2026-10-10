r"""P1 小规模验证：与作者实现的一致性（**参数 + 分数**两条都验）。

对应待办 P1：
    · 固定作者源码快照与文件哈希（由 `pin_author_snapshot.py` 产出，这里核对）
    · 复用 `get_classifier()` 的分类器构造 —— **直接 import 作者那一份**，
      与本项目构造的分类器比 **get_params()** 与 **逐元素分数**
    · 原文固定配置 / 适配调参 分条件；作者原始激活 / 本项目 z-score+/sqrt(d) 分条件
    · 缺失标签**不填 0**
    · 不采用作者 `run_cv_probing()` / `eval_utils.py` 的区间口径

⚠️ 作者仓库 commit `851b70cb` **无 LICENSE**，故其源码只放在项目外的固定快照，
本脚本按路径 import，**不把源码复制进本仓库**。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_readout_probe_lib as lib                               # noqa: E402

SNAPSHOT = Path("D:/Projects/_vendor/fact-probe_851b70cb")
MANIFEST = ROOT / "outputs/stage1/posthoc_readout_probes/author_snapshot_manifest.json"
OUT = ROOT / "outputs/stage1/posthoc_readout_probes/p1_author_consistency.json"
CHECKS: list[dict] = []


def ck(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return bool(ok)


def load_author_get_classifier() -> tuple[object, str]:
    """只取出作者 `train.py` 里的 `get_classifier`，**用它的原文**执行。

    为什么不整模块 import：`train.py` 顶部 `import pandas`，而本机环境**没装 pandas**，
    且既定约束是**不安装或升级任何依赖**。整模块导入会因为无关依赖失败，
    与"复用作者的分类器构造"这件事无关。

    所以用 `ast` 精确定位那一个 FunctionDef，取其**源码片段原文**编译执行，
    命名空间里只放它用到的 sklearn / xgboost 符号。
    片段原文与其 sha256 一并写进产物，Codex 可据此核对取的是不是同一段。
    """
    import ast as _ast
    import hashlib as _hl
    f = SNAPSHOT / "long_fact_probes" / "train.py"
    src = f.read_text(encoding="utf-8")
    node = next((n for n in _ast.walk(_ast.parse(src))
                 if isinstance(n, _ast.FunctionDef) and n.name == "get_classifier"), None)
    if node is None:
        raise RuntimeError("作者 train.py 里找不到 get_classifier")
    seg = _ast.get_source_segment(src, node)
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    ns: dict = {"LogisticRegression": LogisticRegression,
                "XGBClassifier": XGBClassifier}
    exec(compile(seg, str(f), "exec"), ns)          # noqa: S102 —— 执行的是作者原文
    return ns["get_classifier"], seg, _hl.sha256(seg.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    t0 = time.perf_counter()
    rec: dict = {"schema": "p1_author_consistency_v1"}

    # ── 1 快照与哈希 ──────────────────────────────────────────────────────
    print("═══ 1 作者源码快照")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    import hashlib
    bad = []
    for rel, d in man["files_of_interest"].items():
        f = SNAPSHOT / rel
        h = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        if h != d["sha256"]:
            bad.append(rel)
    ck("快照文件哈希与清单一致", not bad, f"{len(man['files_of_interest'])} 个文件"
       + (f"；不符 {bad}" if bad else ""))
    ck("commit 与登记一致", man["commit"] == lib.AUTHOR_COMMIT, man["commit"][:12])
    ck("已登记该 commit 无 LICENSE，源码未进本仓库",
       man["license"]["file_present"] is False and man["snapshot_is_outside_repo"])

    # 整树摘要：**自己重算**，不抄清单里的数。
    # 稳定文件集合 = 排除 `.git` 与 `__pycache__`（读一次源码就会多出 .pyc，
    # 那不是源码污染；旧算法把它算进去，摘要就无谓地对不上）。
    def _stable_files(root: Path):
        return sorted(p for p in root.rglob("*")
                      if p.is_file() and ".git" not in p.parts
                      and "__pycache__" not in p.parts
                      and p.suffix not in (".pyc", ".pyo"))

    tree = hashlib.sha256()
    files = _stable_files(SNAPSHOT)
    for p in files:
        tree.update(str(p.relative_to(SNAPSHOT)).replace("\\", "/").encode())
        tree.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
    recomputed = tree.hexdigest()
    ck("整树摘要**重算**与清单一致（稳定文件集合：排除 .git / __pycache__ / *.pyc）",
       recomputed == man["tree_digest"] and len(files) == man["tree_n_files"],
       f"{len(files)} 个文件（清单记 {man['tree_n_files']}）  {recomputed[:16]}…")
    excluded = sorted(str(p.relative_to(SNAPSHOT)).replace("\\", "/")
                      for p in SNAPSHOT.rglob("*")
                      if p.is_file() and ".git" not in p.parts
                      and ("__pycache__" in p.parts or p.suffix in (".pyc", ".pyo")))
    if excluded:
        print(f"   排除的非源码文件 {len(excluded)} 个：{excluded[:3]}"
              f"{' …' if len(excluded) > 3 else ''}（导入作者模块时生成，非源码变更）")
    rec["snapshot"] = {"commit": man["commit"],
                       "tree_digest_manifest": man["tree_digest"],
                       "tree_digest_recomputed": recomputed,
                       "stable_file_set": "排除 .git / __pycache__ / *.pyc",
                       "n_stable_files": len(files),
                       "excluded_nonsource": excluded,
                       "hash_mismatch": bad}

    # ── 2 分类器构造：参数逐项比 ──────────────────────────────────────────
    print("\n═══ 2 分类器构造（参数）")
    author_get_classifier, seg, seg_sha = load_author_get_classifier()
    ck("取到作者 get_classifier 的源码片段并可执行", callable(author_get_classifier),
       f"片段 sha256 {seg_sha[:16]}…  {len(seg)} 字节")
    rec["author_get_classifier"] = {"source_sha256": seg_sha, "n_bytes": len(seg),
                                    "how": "ast 定位 FunctionDef → 取源码片段原文 → "
                                           "在只含 sklearn/xgboost 的命名空间里执行"
                                           "（本机无 pandas，且不装依赖）"}
    rng = np.random.default_rng(0)
    n, d = 300, 16
    y = (rng.random(n) > 0.6).astype(int)
    X = rng.normal(size=(n, d)) + 1.2 * y[:, None]
    Xt = rng.normal(size=(120, d))

    # ⚠️ 被比较的一侧必须来自**正式生产入口** `lib._fit_score`。
    #    旧版在这里手写了一个 `o_clf = LogisticRegression(...)` 再和作者比 ——
    #    那比的是"测试作者写得对不对"，正式路径改错了它照样通过。
    #    现在参数取 `info["estimator_params"]`（正式路径实际构造出来的那个），
    #    分数取 `info["raw"] / info["proba"]` 与返回的融合分数。
    #: 已登记的本地适配差异。**逐项列明**，不用宽泛白名单盖掉预算/种子的变化。
    REGISTERED_DIFFS = {
        "n_jobs": "本地适配：显式 1，给生成任务让路（作者未指定，默认吃满核）",
        "nthread": "同上（xgboost 旧别名）",
        "verbosity": "本地适配：静默，不影响数值",
        "max_iter": "事前阶梯 (1000, 5000, 20000) 的**起点等于作者值**；"
                    "未收敛才升档，升档会记 max_iter_escalated",
        "use_label_encoder": "作者显式 False；本机 xgboost 3.2 已移除该参数",
        "device": "xgboost 新增参数，作者版本没有；本轮固定 CPU",
        "silent": "xgboost 旧参数，作者版本没有",
    }

    def _cmp_params(a_par: dict, o_par: dict) -> dict:
        """按**键集合的并集**比 —— 只看交集会把新增/缺失的参数整个漏掉。"""
        out = {}
        for k in sorted(set(a_par) | set(o_par)):
            if k not in a_par:
                # ⚠️ "仅我方有" / "仅作者有" **不是**自动放行的理由。
                #    旧版把这两类的 why 写成说明文字，于是未登记的新增键、
                #    被删掉的键（如 C 不见了）都不会触发失败。
                out[k] = {"author": "<缺失>", "ours": str(o_par[k]),
                          "why": REGISTERED_DIFFS.get(k, "**未登记**（仅我方有）")}
            elif k not in o_par:
                out[k] = {"author": str(a_par[k]), "ours": "<缺失>",
                          "why": REGISTERED_DIFFS.get(k, "**未登记**（仅作者有）")}
            elif str(a_par[k]) != str(o_par[k]):
                out[k] = {"author": str(a_par[k]), "ours": str(o_par[k]),
                          "why": REGISTERED_DIFFS.get(k, "**未登记**")}
        return out

    def _production(kind, params, *, X_tr=X, y_tr=y, X_te=Xt, score_mode):
        """正式入口。返回 (融合分数, info)。"""
        return lib._fit_score(kind, params, X_tr, y_tr, X_te, seed=0,
                              score_mode=score_mode)

    param_cmp, score_cmp = {}, {}
    for kind, our_params, author_call in (
        # 作者签名：get_classifier(classifier_name, C=None, use_gpu=False, random_state=42)
        ("logreg_author", {"C": 0.5},
         dict(classifier_name="logistic_regression", C=0.5, random_state=42)),
        ("xgb", {"max_depth": 6, "n_estimators": 1000, "learning_rate": 0.1},
         dict(classifier_name="xgboost", random_state=42)),
    ):
        a_clf = author_get_classifier(**author_call)
        a_par = a_clf.get_params()
        fus_clip, info = _production(kind, our_params, score_mode="clipped_logodds")
        fus_raw, info_raw = _production(kind, our_params, score_mode="raw_margin")
        o_par = info["estimator_params"]

        diff = _cmp_params(a_par, o_par)
        unregistered = {k: v for k, v in diff.items() if v["why"].startswith("**未登记**")}
        param_cmp[kind] = {"estimator_class": info["estimator_class"],
                           "n_keys_union": len(set(a_par) | set(o_par)),
                           "n_keys_author": len(a_par), "n_keys_ours": len(o_par),
                           "differences": diff,
                           "source": "lib._fit_score → info['estimator_params']"}
        ck(f"{kind}: 参数按**并集**比，差异逐项有登记说明", not unregistered,
           f"并集 {len(set(a_par) | set(o_par))} 键，差异 {sorted(diff)}"
           + (f"；未登记 {sorted(unregistered)}" if unregistered else ""))

        # ── 3 分数：同数据同种子，正式路径 vs 作者实现，逐元素比 ───────────
        a_clf.fit(X, y)
        ap_ = a_clf.predict_proba(Xt)[:, list(a_clf.classes_).index(1)]
        if info["proba"] is None:
            # logreg 的正式路径**刻意不算** predict_proba —— decision_function
            # 本身就是未裁剪的 log-odds，转成概率再转回来只会引入裁剪。
            # 因此这里比的是 sigmoid(正式原始分数) 与作者的正类概率。
            op_ = 1.0 / (1.0 + np.exp(-np.asarray(info["raw"], dtype=float)))
            how = "sigmoid(正式 decision_function)（正式路径不调 predict_proba）"
        else:
            op_, how = info["proba"], "正式路径的 predict_proba"
        d_proba = float(np.max(np.abs(ap_ - op_)))
        ck(f"{kind}: **正类概率**与作者实现一致（≤1e-9）", d_proba <= 1e-9,
           f"最大差 {d_proba:.3e}  [{how}]")

        # 原始分数：L1 比 decision_function，XGB 比 output_margin
        if kind == "logreg_author":
            a_raw = np.asarray(a_clf.decision_function(Xt), dtype=float).ravel()
            rname = "decision_function"
        else:
            a_raw = np.asarray(a_clf.predict(Xt, output_margin=True),
                               dtype=float).ravel()
            rname = "output_margin"
        d_raw = float(np.max(np.abs(a_raw - info["raw"])))
        ck(f"{kind}: **原始分数**（{rname}）与作者实现一致（≤1e-9）", d_raw <= 1e-9,
           f"最大差 {d_raw:.3e}")

        # 融合分数规则：两种 score_mode 下的实际返回值都要对得上定义
        if kind == "logreg_author":
            rule_ok = (np.array_equal(fus_clip, info["raw"])
                       and np.array_equal(fus_raw, info["raw"]))
            rule = "logreg 两种口径都用 decision_function（本身即原始 log-odds）"
        else:
            rule_ok = (np.array_equal(fus_raw, info_raw["raw"])
                       and np.allclose(fus_clip, lib._to_log_odds(info["proba"]),
                                       rtol=0, atol=0))
            rule = "raw_margin → output_margin；clipped_logodds → 裁剪概率的 log-odds"
        ck(f"{kind}: **实际融合分数**符合口径定义", rule_ok, rule)
        score_cmp[kind] = {"max_abs_diff_proba": d_proba, "proba_source": how,
                           "max_abs_diff_raw": d_raw, "raw_kind": rname,
                           "fusion_rule": rule, "fusion_rule_ok": bool(rule_ok)}
    rec["param_comparison"] = param_cmp
    rec["score_comparison"] = score_cmp

    # ── 3b 反向检验：故意改一个**有效**参数，这套检查必须失败 ──────────────
    print("\n═══ 3b 反向检验（一致性测试本身能不能失败）")
    mutated = []
    for kind, base, mut, field in (
        ("logreg_author", {"C": 0.5}, {"C": 5.0}, "C"),
        ("xgb", {"max_depth": 6, "n_estimators": 1000, "learning_rate": 0.1},
         {"max_depth": 2, "n_estimators": 1000, "learning_rate": 0.1}, "max_depth"),
    ):
        a_clf = author_get_classifier(
            **(dict(classifier_name="logistic_regression", C=0.5, random_state=42)
               if kind == "logreg_author"
               else dict(classifier_name="xgboost", random_state=42)))
        a_clf.fit(X, y)
        ap_ = a_clf.predict_proba(Xt)[:, list(a_clf.classes_).index(1)]
        _f, bad_info = _production(kind, mut, score_mode="clipped_logodds")
        d_par = _cmp_params(a_clf.get_params(), bad_info["estimator_params"])
        bp = (bad_info["proba"] if bad_info["proba"] is not None
              else 1.0 / (1.0 + np.exp(-np.asarray(bad_info["raw"], dtype=float))))
        d_sc = float(np.max(np.abs(ap_ - bp)))
        caught_par = field in d_par and d_par[field]["why"] == "**未登记**"
        caught_sc = d_sc > 1e-9
        mutated.append({"kind": kind, "mutated": field,
                        "base": base[field], "to": mut[field],
                        "param_check_fires": caught_par,
                        "score_check_fires": caught_sc,
                        "max_abs_diff_proba": d_sc})
        ck(f"{kind}: 把 {field} 从 {base[field]} 改成 {mut[field]} → "
           f"参数检查与分数检查**都报错**", caught_par and caught_sc,
           f"参数命中={caught_par}  分数最大差={d_sc:.3e}")
    # 键集合层面的两种篡改：删掉一个键 / 增加一个未知键。
    # 这两种旧版**都会放行** —— Codex 实测确认，这里补上。
    a_ref = author_get_classifier(classifier_name="logistic_regression",
                                  C=0.5, random_state=42).get_params()
    _f, ok_info = _production("logreg_author", {"C": 0.5},
                              score_mode="clipped_logodds")
    keyset = []
    for label, mutate in (
        ("删掉作者侧的 C", lambda p: {k: v for k, v in p.items() if k != "C"}),
        ("我方多出未知键 bogus_param",
         lambda p: p),   # 改我方，见下
    ):
        if label.startswith("删掉"):
            d = _cmp_params(mutate(a_ref), ok_info["estimator_params"])
        else:
            d = _cmp_params(a_ref, {**ok_info["estimator_params"],
                                    "bogus_param": 123})
        fired = any(v["why"].startswith("**未登记**") for v in d.values())
        keyset.append({"case": label, "fires": fired,
                       "flagged": sorted(k for k, v in d.items()
                                         if v["why"].startswith("**未登记**"))})
        ck(f"键集合篡改「{label}」→ 参数检查报错", fired,
           f"命中 {keyset[-1]['flagged']}")
    rec["mutation_check"] = {
        "why": "证明这套一致性检查会失败 —— 不会失败的检查等于没检查",
        "value_cases": mutated, "keyset_cases": keyset}

    # ── 4 条件表：输入口径 / 原文固定配置 ────────────────────────────────
    print("\n═══ 4 条件表分条件")
    for a, b, field, val in (("han_l1", "han_l1_authorinput", "preprocess", "raw_activation"),
                             ("han_xgboost", "han_xgboost_authorinput", "preprocess",
                              "raw_activation"),
                             ("han_l1", "han_l1_authorfixed", "author_fixed", True),
                             ("han_xgboost", "han_xgboost_authorfixed", "author_fixed", True)):
        A, B = lib.CONDITION_BY_NAME[a], lib.CONDITION_BY_NAME[b]
        others_same = (A.kind == B.kind and A.score_mode == B.score_mode
                       and (A.candidates == B.candidates if field == "preprocess" else True)
                       and (A.preprocess == B.preprocess if field == "author_fixed" else True))
        ck(f"{b}: 只改 {field}={val}，其余继承 {a}",
           getattr(B, field) == val and others_same)
    for c in ("han_l1_authorfixed", "han_xgboost_authorfixed"):
        C = lib.CONDITION_BY_NAME[c]
        ck(f"{c}: 候选集只有原文那一个配置（不调参）", len(C.candidates) == 1,
           str(lib._jsonable(C.candidates[0])))
    ck("原文参照点确实是所固定的那个",
       lib.CONDITION_BY_NAME["han_l1_authorfixed"].candidates[0]["C"] == 0.5
       and lib.CONDITION_BY_NAME["han_xgboost_authorfixed"].candidates[0][
           "n_estimators"] == 1000)

    # ── 4b 三因素覆盖：**如实登记缺口**，不称"已覆盖完整作者设置" ───────────
    print("\n═══ 4b 三因素覆盖（输入口径 × 参数来源 × 评分口径）")
    print(f"   {'条件':<26}{'输入':<18}{'参数':<10}{'评分':<18}{'K'}")
    cover = {}
    for c in lib.CONDITIONS:
        src = "原文固定" if c.author_fixed else "项目调参"
        cover[c.name] = {"preprocess": c.preprocess, "params": src,
                         "score_mode": c.score_mode, "n_candidates": len(c.candidates)}
        print(f"   {c.name:<26}{c.preprocess:<18}{src:<10}{c.score_mode:<18}"
              f"{len(c.candidates)}")
    han = [c for c in lib.CONDITIONS if c.kind in ("logreg_author", "xgb")]
    full_author = [c.name for c in han
                   if c.preprocess == "raw_activation" and c.author_fixed
                   and (c.kind != "xgb" or c.score_mode == "raw_margin")]
    ck("**如实登记**：目前没有『原始激活 + 作者固定参数 + 原生评分』这一组合 —— "
       "现有的是两个单因素对照，不得称已覆盖完整作者设置",
       not full_author,
       f"缺口条件数 {len(full_author)}；现有 authorinput/authorfixed 各改一个因素")
    xgb_inherit = [c.name for c in han
                   if c.kind == "xgb" and (c.preprocess == "raw_activation"
                                           or c.author_fixed)
                   and c.score_mode == "clipped_logodds"]
    ck("XGB 的 authorinput / authorfixed 仍继承 clipped_logodds，已登记",
       set(xgb_inherit) == {"han_xgboost_authorinput", "han_xgboost_authorfixed"},
       str(sorted(xgb_inherit)))
    rec["factor_coverage"] = {
        "matrix": cover,
        "gap": ("无『raw_activation + author_fixed + raw_margin』组合。"
                "L1 不受评分口径影响（decision_function 本身即原始 log-odds），"
                "故 L1 的缺口只在『原始激活 + 固定 C=0.5』；"
                "XGB 的缺口是三因素同时齐备。"),
        "not_claimed": "两个单因素对照 ≠ 覆盖完整作者设置",
        "next": "最小必要组合与预算见 docs/posthoc/minimal_condition_matrix.md；"
                "运行需另行授权"}

    # ── 5 明确不采用的作者行为 ───────────────────────────────────────────
    print("\n═══ 5 明确不采用的作者行为")
    src = (SNAPSHOT / "long_fact_probes" / "train.py").read_text(
        encoding="utf-8", errors="replace")
    ck("作者 train.py 里确有 run_cv_probing（我们不复用）", "run_cv_probing" in src)
    ck("作者确有把缺失 is_supported 当 0 的写法（我们不采用）",
       "is_supported" in src, "见 manifest.provenance.not_adopted")
    # ⚠️ 不能用字符串匹配：lib 的注释里就写着"不复用 run_cv_probing()"，
    #    子串必然命中。要查的是**真实的 import 与调用**，所以走 AST。
    import ast as _a
    banned_calls, banned_imports = {"run_cv_probing"}, {"eval_utils"}
    hits = []
    for rel in ("scripts/posthoc_readout_probe_lib.py",
                "scripts/posthoc_readout_probes_stage1.py"):
        tree = _a.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in _a.walk(tree):
            if isinstance(node, _a.Call):
                fn = node.func
                nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if nm in banned_calls:
                    hits.append(f"{rel}:{node.lineno} 调用 {nm}")
            elif isinstance(node, (_a.Import, _a.ImportFrom)):
                mods = ([a.name for a in node.names]
                        + ([node.module] if isinstance(node, _a.ImportFrom)
                           and node.module else []))
                for m in mods:
                    if any(b in (m or "") for b in banned_imports):
                        hits.append(f"{rel}:{node.lineno} import {m}")
    ck("本项目代码里没有 run_cv_probing 的调用、也没有 import eval_utils（AST 查，非字符串匹配）",
       not hits, "；".join(hits) if hits else "两个驱动文件均无")
    # 缺失标签：必须报错，不能当负例
    raised = False
    try:
        lib._fit_score("logreg_author", {"C": 1.0}, X,
                       np.where(rng.random(n) < 0.1, np.nan, y), Xt, seed=0)
    except Exception as exc:                                  # noqa: BLE001
        raised = True
        why = f"{type(exc).__name__}"
    ck("标签含缺失时**报错**，不静默当负例", raised, why if raised else "未报错")

    ok = all(c["ok"] for c in CHECKS)
    rec.update(verdict="PASS" if ok else "FAIL", n_checks=len(CHECKS),
               n_failed=sum(1 for c in CHECKS if not c["ok"]), checks=CHECKS,
               total_secs=round(time.perf_counter() - t0, 1),
               scope="只用合成数据核一致性；**不跑任何真实臂，不改已有结果**")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str)
                        + "\n", encoding="utf-8")
    print(f"\n[{rec['verdict']}] {len(CHECKS) - rec['n_failed']}/{len(CHECKS)} 通过")
    print(f"[out] {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
