# Sprint 0E：基础验收与工程收尾

## 1. 目标

本 sprint 是 Sprint 0 的收尾检查。

目标是：

```text id="tglgxp"
确认 Sprint 0B / 0C / 0D 的基础工程能力可以在当前 conda 环境稳定运行
检查 requirements.txt 是否包含当前阶段必要依赖
检查 .gitignore 是否覆盖常见临时文件和生成文件
整理 PROGRESS.md
标记 Sprint 0 工程地基完成
```

本 sprint 不实现任何新的实验方法。

---

## 2. 本次允许修改的文件

只允许修改或创建：

```text id="dy5h9o"
.gitignore
requirements.txt
PROGRESS.md
```

运行命令时允许生成或更新：

```text id="f7zhfv"
data/processed/questions.jsonl
outputs/logs/smoke_test_questions.jsonl
```

但不要手动编辑这些生成文件。

---

## 3. 本次禁止修改的文件

不要修改：

```text id="c7c9i0"
README.md
AGENTS.md
docs/skill/SKILL.md
docs/prompts.md
docs/reference/*
docs/codex_tasks/*
src/recover_attention/*.py
scripts/*.py
tests/*.py
configs/*.yaml
data/examples/*
```

如果发现代码文件存在问题，不要直接修改，先停止并报告。

---

## 4. 环境要求

当前正确环境是 conda 环境：

```text id="cxwfuw"
recover_attention
```

安装依赖时必须使用：

```bash id="b3k5re"
python -m pip install -r requirements.txt
```

运行 pytest 时必须使用：

```bash id="hqsr6o"
python -m pytest -q
```

禁止使用：

```text id="b9l88v"
pip install ...
pytest -q
.venv
```

---

## 5. requirements.txt 检查要求

检查 `requirements.txt` 是否至少包含当前阶段必要依赖：

```text id="h7i0w4"
PyYAML
pytest
```

允许写成：

```text id="u5j5o3"
PyYAML>=6.0
pytest>=8.0
```

如果缺少，可以补充。

不要加入未来阶段依赖，例如：

```text id="ha0j11"
torch
transformers
datasets
sentence-transformers
scikit-learn
pandas
numpy
openai
accelerate
```

这些依赖后续需要时再加。

---

## 6. .gitignore 检查要求

检查 `.gitignore` 是否覆盖以下内容。

建议至少包含：

```gitignore id="sgqu52"
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/

# Local outputs
outputs/
data/processed/

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db
```

如果 `.gitignore` 已存在，尽量增量补充，不要删除已有规则。

---

## 7. 必须运行的命令

请按顺序运行：

### 7.1 检查 Python 路径

```bash id="la287e"
python -c "import sys; print(sys.executable); print(sys.version)"
```

期望路径应指向 conda 环境 `recover_attention`。

---

### 7.2 检查 pip 路径

```bash id="psfsw2"
python -m pip -V
```

期望路径应指向 conda 环境 `recover_attention`。

---

### 7.3 安装当前阶段依赖

```bash id="ytpiq6"
python -m pip install -r requirements.txt
```

---

### 7.4 验证 pytest 和 yaml

```bash id="z7cr3i"
python -c "import pytest; print('pytest', pytest.__version__); print(pytest.__file__)"
```

```bash id="acnfpz"
python -c "import yaml; print('pyyaml ok'); print(yaml.__file__)"
```

---

### 7.5 运行 Sprint 0B smoke test

```bash id="hmnik0"
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
```

---

### 7.6 运行 Sprint 0D prepare_data

```bash id="3b7kqn"
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
```

---

### 7.7 运行全部测试

```bash id="x16j8v"
python -m pytest -q
```

---

### 7.8 检查实际修改文件

如果当前目录是 git 仓库，运行：

```bash id="3w0k4d"
git diff --name-only
```

如果不是 git 仓库，在最终回复中说明未检测到 git 仓库。

---

## 8. 验收标准

必须满足：

```text id="rng7c8"
[OK] Python 指向 recover_attention 环境
[OK] pip 指向 recover_attention 环境
[OK] pytest 可以 import
[OK] yaml 可以 import
[OK] requirements.txt 包含 PyYAML 和 pytest
[OK] .gitignore 覆盖常见 Python 缓存、虚拟环境、outputs、data/processed
[OK] Sprint 0B smoke test 通过
[OK] Sprint 0D prepare_data 通过
[OK] python -m pytest -q 通过
[OK] 未修改 src/recover_attention/*.py
[OK] 未修改 scripts/*.py
[OK] 未修改 tests/*.py
[OK] 未修改 README.md / AGENTS.md / docs/skill/SKILL.md / docs/prompts.md / docs/reference/*
[OK] PROGRESS.md 已更新
```

---

## 9. 禁止事项

本次不要做：

```text id="liw6qn"
不要修改实验代码。
不要修改测试代码。
不要新增实验功能。
不要实现 candidate span extraction。
不要实现 ablated question construction。
不要实现 NLI scoring。
不要实现 masked question construction。
不要实现 question recovery。
不要实现 recoverability scoring。
不要实现 label building。
不要实现 evaluation metrics。
不要调用任何模型。
不要下载任何模型。
不要调用外部 API。
不要训练 probe。
不要做 attention guidance。
不要做 hidden states。
不要做 trajectory analysis。
不要做大规模实验。
不要开始 Sprint 1A。
```

---

## 10. PROGRESS.md 更新要求

完成后更新 `PROGRESS.md`，新增或更新一节：

```md id="r1zssu"
### Sprint 0E：基础验收与工程收尾

已完成内容：

- 检查当前 conda 环境。
- 检查 requirements.txt。
- 检查 .gitignore。
- 重新运行 Sprint 0B smoke test。
- 重新运行 Sprint 0D prepare_data。
- 运行全部 pytest。
- 完成 Sprint 0 工程地基验收。

新增或修改文件：

- .gitignore
- requirements.txt
- PROGRESS.md

运行命令：

- python -c "import sys; print(sys.executable); print(sys.version)"
- python -m pip -V
- python -m pip install -r requirements.txt
- python -c "import pytest; print('pytest', pytest.__version__); print(pytest.__file__)"
- python -c "import yaml; print('pyyaml ok'); print(yaml.__file__)"
- python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
- python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
- python -m pytest -q
- git diff --name-only

检查结果：

- xxx

遗留问题：

- xxx

Sprint 0 总结：

- Sprint 0A：项目骨架与文档约束，完成。
- Sprint 0B：jsonl 数据读写 + 样例数据 + smoke test，完成。
- Sprint 0C：schema 校验，完成。
- Sprint 0D：prepare_data 数据准备脚本，完成。
- Sprint 0E：基础验收与工程收尾，完成。

下一步建议：

- Sprint 1A：candidate span extraction
```

不要自动开始 Sprint 1A。
