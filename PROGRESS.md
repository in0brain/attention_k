# 实验进度记录：Recoverability-Guided Attention Allocation

## 0. 当前实验边界

当前只做 v0 / v1 阶段：

candidate span extraction
→ ablated question construction
→ NLI semantic necessity scoring
→ masked question construction
→ question recovery
→ recoverability scoring

暂时不做：

probe 训练
attention guidance
hidden states
trajectory analysis
大规模实验

## 1. 现有进度

### (1) Sprint 0A：最小项目骨架

已完成内容：

- 初始化项目说明、开发规则和进度记录文件。
- 创建 docs、configs、data、outputs、src、scripts、tests 基础目录。
- 创建 recover_attention Python 包结构。
- 为后续实验模块添加最小占位文件。

新增或修改文件：

- README.md
- AGENTS.md
- PROGRESS.md
- requirements.txt
- pyproject.toml
- docs/method.md
- docs/experiment_guide.md
- docs/label_schema.md
- docs/prompts.md
- docs/codex_tasks.md
- src/recover_attention/__init__.py
- src/recover_attention/data_io.py
- src/recover_attention/schemas.py
- src/recover_attention/candidate_extraction.py
- src/recover_attention/question_ablations.py
- src/recover_attention/nli_scoring.py
- src/recover_attention/recover_generation.py
- src/recover_attention/recover_scoring.py
- src/recover_attention/label_builder.py
- src/recover_attention/evaluation.py
- src/recover_attention/utils.py

检查结果：

- 已确认当前目录作为项目根目录，没有嵌套新的 recover_attention_project 目录。
- Sprint 0A 只包含项目骨架和占位内容，未实现实验逻辑。

### Sprint 0A-docs：文档控制框架

已完成内容：

- 建立 docs/skill、docs/reference、docs/codex_tasks 三层文档结构。
- 创建短版 SKILL.md，作为后续 Codex 执行契约。
- 创建 reference 目录，用于保存完整实验指导书和项目方案。
- 创建 sprint_0B_data_io.md，作为下一步任务卡。

新增或修改文件：

- docs/skill/SKILL.md
- docs/reference/README.md
- docs/reference/experiment_guide_full.md
- docs/reference/project_plan_full.md
- docs/reference/progress_template_full.md
- docs/codex_tasks/README.md
- docs/codex_tasks/task_card_template.md
- docs/codex_tasks/sprint_0B_data_io.md

检查结果：

- 文档目录已建立。
- 根目录未发现完整参考文档源文件，已在 docs/reference/ 创建占位参考文件，未删除任何原文件。
- 后续 Codex 执行入口明确为 docs/skill/SKILL.md + 当前 task card + PROGRESS.md。

下一步：

- 执行 docs/codex_tasks/sprint_0B_data_io.md。

### Sprint 0B：jsonl 数据读写 + 样例数据 + smoke test

已完成内容：

- 实现统一 jsonl 读写工具。
- 创建 v0_nli_small.yaml 配置文件。
- 创建 5 条 questions_small.jsonl 样例数据。
- 实现 smoke test。
- 实现 data_io 的 pytest 测试。

新增或修改文件：

- src/recover_attention/data_io.py
- configs/v0_nli_small.yaml
- data/examples/questions_small.jsonl
- scripts/00_smoke_test.py
- tests/test_data_io.py
- PROGRESS.md

运行命令：

- python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
- pytest -q

检查结果：

- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`。
- pytest 已通过，结果为 `4 passed`。
- data/examples/questions_small.jsonl 已创建，包含 5 条样例数据。
- configs/v0_nli_small.yaml 已创建。
- 未修改 README.md、AGENTS.md、docs/skill/SKILL.md 或 docs/reference/*。
- 未开始 Sprint 0C。
- 当前验收环境已确认切换为 conda recover_attention。
- Python 路径：D:\conda\Miniconda3\envs\recover_attention\python.exe
- pip 路径：D:\conda\Miniconda3\envs\recover_attention\lib\site-packages\pip
- 依赖已通过 python -m pip install -r requirements.txt 安装到当前环境。
- smoke test 已在当前 conda 环境通过。
- python -m pytest -q 已在当前 conda 环境通过，结果为 `4 passed`。

遗留问题：

- 无。

下一步建议：

- Sprint 0C：schema 校验

### Sprint 0C：schema 校验

已完成内容：

- 实现 v0/v1 jsonl 中间记录的轻量 schema 校验函数。
- 支持 question、candidate span、ablated question、NLI score、masked question、recover output、recover score 记录校验。
- 校验失败时抛出 ValueError，并在错误信息中说明缺失字段或非法字段。
- 实现 schema 校验 pytest 测试。

新增或修改文件：

- src/recover_attention/schemas.py
- tests/test_schemas.py
- PROGRESS.md

运行命令：

- python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
- python -m pytest -q

检查结果：

- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`。
- python -m pytest -q 已通过，结果为 `19 passed`。
- src/recover_attention/schemas.py 已实现 task card 要求的校验函数。
- tests/test_schemas.py 已覆盖 task card 要求的基础有效与失败场景。
- 未修改 README.md、AGENTS.md、docs/skill/SKILL.md、docs/reference/* 或 docs/codex_tasks/*。
- 未实现 Sprint 0D，未实现 prepare_data.py。

遗留问题：

- 无。

下一步建议：

- Sprint 0D：prepare_data 数据准备脚本

### Sprint 0D：prepare_data 数据准备脚本

已完成内容：

- 实现 question record 规范化函数。
- 实现 prepare_questions 数据准备函数。
- 实现 CLI 脚本 scripts/01_prepare_data.py。
- 实现 prepare_data pytest 测试。
- 生成 data/processed/questions.jsonl。

新增或修改文件：

- src/recover_attention/prepare_data.py
- scripts/01_prepare_data.py
- tests/test_prepare_data.py
- data/processed/questions.jsonl
- PROGRESS.md

运行命令：

- python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
- python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
- python -m pytest -q

检查结果：

- prepare_data 脚本已通过，生成 5 条标准 question records。
- smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`。
- python -m pytest -q 已通过，结果为 `27 passed`。
- data/processed/questions.jsonl 已生成。
- 未修改 README.md、AGENTS.md、docs/skill/SKILL.md、docs/prompts.md、docs/reference/* 或 docs/codex_tasks/*。
- 未修改 data_io.py 和 schemas.py。
- 未实现 Sprint 1A，未实现 candidate span extraction、ablation、NLI 或 recover 逻辑。

遗留问题：

- 无。

下一步建议：

- Sprint 0E：基础验收与工程收尾

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

- Python 指向 conda recover_attention 环境：D:\conda\Miniconda3\envs\recover_attention\python.exe。
- pip 指向 conda recover_attention 环境：D:\conda\Miniconda3\envs\recover_attention\lib\site-packages\pip。
- pytest 可以 import，版本为 9.1.1。
- yaml 可以 import，PyYAML 版本满足当前阶段依赖要求。
- requirements.txt 包含 PyYAML>=6.0 和 pytest>=8.0，未加入未来阶段大依赖。
- .gitignore 覆盖 Python 缓存、虚拟环境、outputs、data/processed、IDE 和 OS 临时文件。
- Sprint 0B smoke test 已通过，输出 `[OK] Sprint 0B smoke test passed.`。
- Sprint 0D prepare_data 已通过，生成 5 条标准 question records。
- python -m pytest -q 已通过，结果为 `27 passed`。
- 未修改 src/recover_attention/*.py、scripts/*.py、tests/*.py。
- 未修改 README.md、AGENTS.md、docs/skill/SKILL.md、docs/prompts.md 或 docs/reference/*。
- 未开始 Sprint 1A。

遗留问题：

- 当前 git 工作区存在此前已有的 docs/codex_tasks/*、docs/prompts.md、src/recover_attention/schemas.py、.idea/recover_attention_project.iml 等改动，本 sprint 未处理这些文件。

Sprint 0 总结：

- Sprint 0A：项目骨架与文档约束，完成。
- Sprint 0B：jsonl 数据读写 + 样例数据 + smoke test，完成。
- Sprint 0C：schema 校验，完成。
- Sprint 0D：prepare_data 数据准备脚本，完成。
- Sprint 0E：基础验收与工程收尾，完成。

下一步建议：

- Sprint 1A：candidate span extraction（暂时还没写）

## 2. 需要下一步干的事项

### (1) Sprint 1A：candidate span extraction（暂时还没写）

目标：

等待后续 task card 明确后，再实现 candidate span extraction。

需要新增或修改的文件：

- 待后续 task card 明确。

验收标准：

- 待后续 task card 明确。
