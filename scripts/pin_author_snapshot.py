r"""固定作者源码快照并生成哈希清单（P1 第一项）。

⚠️ **不把作者源码复制进本项目仓库。** `JThh/fact-probe` 在 commit `851b70cb`
**没有 LICENSE 文件** —— 无显式许可即默认保留全部权利，因此：

    快照位置   D:/Projects/_vendor/fact-probe_851b70cb/   （项目外，不进版本库）
    项目内只存 本脚本产出的**哈希清单**（提交 / 路径 / sha256 / 字节数）
    引用方式   报告里引用 commit 与文件哈希，不转载其源码

清单同时记录三类来源的划分：**直接移植 / 我们修改 / 依据论文自行实现**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("D:/Projects/_vendor/fact-probe_851b70cb")
OUT = ROOT / "outputs/stage1/posthoc_readout_probes/author_snapshot_manifest.json"
COMMIT = "851b70cbdc8a800b135afb9323ff6c4d6996b06c"

#: 本项目**实际读过或复用**的文件。只给这些算哈希，避免清单被无关文件淹没。
FILES_OF_INTEREST = [
    "long_fact_probes/train.py",
    "long_fact_probes/eval.py",
    "long_fact_probes/eval_utils.py",
    "long_fact_probes/predict.py",
    "long_fact_probes/README.md",
    "requirements.txt",
    "scripts/run_sentence.py",
    "README.md",
]

PROVENANCE = {
    "directly_ported": [
        {"what": "`get_classifier()` 的分类器构造与超参数",
         "where": "long_fact_probes/train.py",
         "detail": ("LogisticRegression(penalty='l1', solver='liblinear', C=C, "
                    "max_iter=1000, random_state=random_state)；"
                    "XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=6, "
                    "objective='binary:logistic', use_label_encoder=False, "
                    "eval_metric='auc', random_state=random_state)")},
    ],
    "modified_by_us": [
        {"what": "XGBoost 显式 `n_jobs=1` + CPU",
         "why": "作者未指定 n_jobs（默认吃满核）；本机要给生成任务让路。"
                "作者的 gpu_hist 分支本轮不用。"},
        {"what": "`max_iter` 按事前阶梯 (1000, 5000, 20000) 上调",
         "why": "作者值是参照不是约束；不为保持原样而接受未收敛结果。"
                "用到后面档位会记 max_iter_escalated。"},
        {"what": "新增**原始分数**通道（XGB output_margin / MLP sigmoid 前）",
         "why": "P0：旧口径把概率按 PROB_EPS=1e-6 裁剪后转 log-odds，"
                "实测 H1 MLP 有 47.69% 的预测被边界夹住。旧条件保留，新条件并列报告。"},
    ],
    "not_adopted": [
        {"what": "`run_cv_probing()` 的整套评估流程",
         "why": "无分组 StratifiedKFold、跨层拼接选最佳 layer group、"
                "选完在全量数据重拟合、裸 `except: continue` 会静默跳过失败配置。"
                "四条都与本项目的 grouped nested CV / 共享 S* / 融合协议冲突。"},
        {"what": "缺失 `is_supported` 默认填 0",
         "why": "缺标签当负例会**制造**负例。本项目缺标签一律报错或透明登记。"},
        {"what": "`eval_utils.py` 的区间口径",
         "why": "其独立样本 bootstrap 不处理组相关；`std_err = std / sqrt(B)` "
                "是 bootstrap 均值的蒙特卡洛误差，**不是**估计量的采样标准误。"
                "本项目用配对 cluster bootstrap。"},
        {"what": "requirements.txt 整份安装、自动下载与 FP16 加载入口",
         "why": "`torch>=1.13.0,<2.0.0` 会降级本机 torch 2.11。"
                "隐藏状态一律用本项目既有的本地加载器，精度差异另行登记。"},
    ],
    "implemented_from_paper": [
        {"what": "SAPLMA MLP 256→128→64 + ReLU + sigmoid + Adam",
         "source": "Azaria & Mitchell 2023, Findings of EMNLP, §4",
         "caveat": "**未找到可核实的作者官方实现**，不得归为 Han 作者代码。"
                   "优化器超参、batch size 用 sklearn 默认；epoch 与 alpha 训练折内选参。"},
    ],
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    if not args.snapshot.is_dir():
        print(f"[stop] 快照目录不存在：{args.snapshot}", file=sys.stderr)
        return 2
    head = subprocess.run(["git", "-C", str(args.snapshot), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        print(f"[stop] 快照 HEAD={head!r} 与登记的 commit 不符（{COMMIT}）",
              file=sys.stderr)
        return 2

    files, missing = {}, []
    for rel in FILES_OF_INTEREST:
        f = args.snapshot / rel
        if not f.is_file():
            missing.append(rel)
            continue
        files[rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
        print(f"  {rel:<36} {files[rel]['bytes']:>7} B  {files[rel]['sha256'][:16]}…")
    # 整树摘要：把上面没列进来的文件也锁住，防止"只换了别处"。
    #
    # **稳定文件集合**必须定义清楚：排除 `.git`、`__pycache__` 和 `*.pyc`。
    # 只要导入过作者模块一次，目录里就会多出 .pyc，摘要随即对不上 ——
    # 那是解释器产物，不是源码变更。核验脚本用的是同一套规则。
    stable = sorted(f for f in args.snapshot.rglob("*")
                    if f.is_file() and ".git" not in f.parts
                    and "__pycache__" not in f.parts
                    and f.suffix not in (".pyc", ".pyo"))
    tree = hashlib.sha256()
    for f in stable:
        tree.update(str(f.relative_to(args.snapshot)).replace("\\", "/").encode())
        tree.update(sha256(f).encode())
    n_files = len(stable)

    doc = {
        "schema": "author_snapshot_manifest_v1",
        "repo": "https://github.com/JThh/fact-probe",
        "commit": COMMIT,
        "commit_date": subprocess.run(
            ["git", "-C", str(args.snapshot), "log", "-1", "--format=%aI"],
            capture_output=True, text=True).stdout.strip(),
        "snapshot_path": str(args.snapshot).replace("\\", "/"),
        "snapshot_is_outside_repo": True,
        "license": {
            "file_present": (args.snapshot / "LICENSE").exists(),
            "note": ("该 commit **没有 LICENSE 文件** —— 无显式许可即默认保留全部权利。"
                     "因此**不把其源码复制进本项目仓库**，只在项目内保存本哈希清单，"
                     "报告中引用 commit 与文件哈希，不转载源码。"),
        },
        "files_of_interest": files,
        "missing_files": missing,
        "tree_digest": tree.hexdigest(),
        "tree_n_files": n_files,
        "tree_stable_file_set": "排除 .git / __pycache__ / *.pyc（解释器产物，非源码）",
        "provenance": PROVENANCE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n  整树摘要 {doc['tree_digest'][:16]}…  ({n_files} 个文件)")
    print(f"  LICENSE 存在？ {doc['license']['file_present']}")
    if missing:
        print(f"  ⚠️ 缺文件：{missing}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
