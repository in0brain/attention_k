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

## 2. 需要下一步干的事项

### (1) Sprint 0C：schema 校验

目标：

根据后续 task card 实现最小 schema 校验。

需要新增或修改的文件：

- 待后续 task card 明确。

验收标准：

- 待后续 task card 明确。
