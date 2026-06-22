# Prompts

本文件保存可复用 Codex 提示词模板。

用途：

```text
减少重复输入
避免 Codex 越界
统一每轮 sprint 执行方式
统一环境检查方式
统一 PROGRESS.md 更新要求
```

---

# 1. 通用规则

给 Codex 的每轮提示词都应包含：

```text
1. 当前项目路径
2. 当前只执行哪一张 task card
3. 必须阅读哪些文件
4. 禁止阅读或扩展哪些 full reference 文档
5. 本轮允许修改哪些文件
6. 本轮禁止修改哪些文件
7. 必须运行哪些命令
8. 必须更新 PROGRESS.md
9. 不要自动开始下一步
```

本文件只保存**通用提示词模板**。

具体 sprint 的详细要求应写在：

```text
docs/codex_tasks/*.md
```

不要在本文件中长期维护某个具体 sprint 的完整执行提示词，避免和 task card 内容不一致。

---

# 2. 标准 Sprint 执行提示词

适用于大多数 sprint。

使用时只需要替换：

```text
<TASK_CARD>
<SPRINT_NAME>
<NEXT_SPRINT_NAME>
```

模板如下：

```text
你现在在 `D:\Projects\recover_attention_project` 项目中工作。

请严格按照以下流程执行。

本次 task card：

`docs/codex_tasks/<TASK_CARD>.md`

本次 sprint：

`<SPRINT_NAME>`

下一步建议：

`<NEXT_SPRINT_NAME>`

# 一、必须先阅读的文件

请按顺序阅读：

1. `AGENTS.md`
2. `PROGRESS.md`
3. `docs/skill/SKILL.md`
4. `docs/codex_tasks/<TASK_CARD>.md`

本轮不要主动阅读或使用：

`docs/reference/*`

`docs/reference/*` 只是长期参考文档，不是本轮执行入口。除非当前 task card 信息缺失到无法执行，否则不要参考 full 文档；如果确实无法执行，请先停止并说明缺什么，不要自行扩展任务。

# 二、执行前必须先输出 Preflight

修改任何文件之前，必须先输出 Preflight。

Preflight 必须包括：

1. 已阅读文件列表
2. 本次允许修改的文件
3. 本次禁止修改的文件
4. 本次必须运行的命令
5. 是否需要参考 `docs/reference/*`
6. 是否发现 task card 与当前提示词冲突

如果当前提示词、task card、AGENTS.md、SKILL.md 发生冲突，优先级为：
当前用户消息 > 当前 task card > AGENTS.md > docs/skill/SKILL.md > docs/prompts.md > docs/reference/*
并在最后在`遗留问题`中给用户报告冲突部分。
如果发现需要修改允许列表之外的文件，先停止并询问用户。

Preflight 输出后，再开始修改。

# 三、硬约束

本次只执行当前 task card。

不要做当前 task card 以外的工作。

尤其不要做：

- 不要实现下一阶段。
- 不要调用任何模型。
- 不要下载任何模型。
- 不要调用外部 API。
- 不要训练 probe。
- 不要做 attention guidance。
- 不要做 hidden states。
- 不要做 trajectory analysis。
- 不要做大规模实验。
- 不要重构项目。
- 不要修改 README.md。
- 不要修改 AGENTS.md。
- 不要修改 docs/skill/SKILL.md。
- 不要修改 docs/reference/*。
- 不要自动开始下一 sprint。

# 四、环境要求

当前正确环境是 conda 环境：

`recover_attention`

安装依赖时必须使用：

`python -m pip install -r requirements.txt`

运行 pytest 时必须使用：

`python -m pytest -q`

不要使用裸命令：

`pip install ...`

不要使用裸命令：

`pytest -q`

不要使用 `.venv`。

# 五、必须运行的命令

请根据 task card 运行其中要求的命令。

如果本轮涉及测试，优先使用：

`python -m pytest -q`

如果本轮涉及 smoke test，使用：

`python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml`

# 六、PROGRESS.md 更新要求

完成后必须更新 `PROGRESS.md`。

更新内容包括：

- 本次 sprint 名称
- 已完成内容
- 新增或修改文件
- 运行命令
- 检查结果
- 遗留问题
- 下一步建议：`<NEXT_SPRINT_NAME>`

不要自动开始下一步。

# 七、完成后的回复格式

完成后请回复：

1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. PROGRESS.md 更新摘要
6. 遗留问题
7. 下一步建议：`<NEXT_SPRINT_NAME>`

不要直接开始下一 sprint。
```

---

# 3. 环境修复 / 重新验收提示词

当发现 Codex 装错环境，或者 `.venv` 和 conda 混用时，使用这个。

```text
你现在在 `D:\Projects\recover_attention_project` 项目中工作。

本次不是新 sprint，不要实现任何新功能。

本次任务只做：修复并验证当前 conda 环境 `recover_attention` 的依赖安装，然后重新运行已有验收命令。

# 一、当前正确环境

当前正确 Python 解释器应为：

`D:\conda\Miniconda3\envs\recover_attention\python.exe`

当前正确 pip 应来自：

`D:\conda\Miniconda3\envs\recover_attention\lib\site-packages\pip`

# 二、严格要求

本次所有命令都必须通过当前 conda 环境的 Python 执行。

优先使用：

`python -m pip ...`

`python -m pytest ...`

禁止使用：

`pip install ...`

`pytest -q`

`.venv\Scripts\python.exe`

`.venv\Scripts\pip.exe`

不要创建新的 `.venv`。
不要激活 `.venv`。
不要删除 `.venv`。
不要修改实验代码。

# 三、本次允许做的事

只允许：

1. 检查当前 Python 解释器。
2. 检查当前 pip 路径。
3. 使用当前 Python 安装 `requirements.txt` 中的依赖。
4. 验证 pytest / pyyaml 是否装在当前 conda 环境。
5. 重新运行 smoke test。
6. 重新运行 pytest。
7. 如果验收通过，更新 PROGRESS.md 中的检查结果或遗留问题说明。

# 四、本次禁止修改的文件

不要修改：

- `README.md`
- `AGENTS.md`
- `docs/skill/SKILL.md`
- `docs/reference/*`
- `docs/codex_tasks/*`
- `src/recover_attention/*.py`
- `scripts/*.py`
- `tests/*.py`
- `configs/*.yaml`
- `data/*`

唯一可以修改：

- `PROGRESS.md`

# 五、请按顺序执行这些命令

1. 确认当前 Python：

`python -c "import sys; print(sys.executable); print(sys.version)"`

2. 确认当前 pip：

`python -m pip -V`

3. 安装依赖到当前 conda 环境：

`python -m pip install -r requirements.txt`

4. 验证 pytest：

`python -c "import pytest; print('pytest', pytest.__version__); print(pytest.__file__)"`

5. 验证 yaml：

`python -c "import yaml; print('pyyaml ok'); print(yaml.__file__)"`

6. 运行 smoke test：

`python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml`

7. 运行测试：

`python -m pytest -q`

# 六、验收标准

必须满足：

- Python 指向 `D:\conda\Miniconda3\envs\recover_attention\python.exe`
- pip 指向 `D:\conda\Miniconda3\envs\recover_attention`
- pytest 可以在当前 Python 中 import
- yaml 可以在当前 Python 中 import
- smoke test 通过
- `python -m pytest -q` 通过
- 没有修改代码文件
- 没有使用 `.venv`

# 七、完成后回复格式

请回复：

1. 当前 Python 路径
2. 当前 pip 路径
3. pytest 安装位置
4. yaml 安装位置
5. smoke test 结果
6. `python -m pytest -q` 结果
7. 是否修改了 PROGRESS.md
8. 是否存在遗留问题

不要开始新的 sprint。
```

---

# 4. 只更新 PROGRESS.md 的提示词

当代码已经人工确认，只想让 Codex 整理进度时使用。

```text
你现在在 `D:\Projects\recover_attention_project` 项目中工作。

本次不要实现任何新功能。
本次只更新 `PROGRESS.md`。

请阅读：

1. `AGENTS.md`
2. `PROGRESS.md`
3. `docs/skill/SKILL.md`

不要阅读或扩展：

`docs/reference/*`

不要修改：

- `README.md`
- `AGENTS.md`
- `docs/skill/SKILL.md`
- `docs/reference/*`
- `docs/codex_tasks/*`
- `src/*`
- `scripts/*`
- `tests/*`
- `configs/*`
- `data/*`

只允许修改：

- `PROGRESS.md`

请把最近完成的工作整理进 `PROGRESS.md`，格式包括：

1. sprint 名称
2. 已完成内容
3. 新增或修改文件
4. 运行命令
5. 检查结果
6. 遗留问题
7. 下一步建议

不要开始下一步。
```

---

# 5. 下一步 Task Card 生成提示词

当需要让 Codex 帮忙草拟下一张 task card 时使用。

注意：只让它草拟 task card，不让它执行。

```text
你现在在 `D:\Projects\recover_attention_project` 项目中工作。

本次不要实现任何实验功能。
本次只草拟下一张 Codex task card。

请阅读：

1. `AGENTS.md`
2. `PROGRESS.md`
3. `docs/skill/SKILL.md`
4. `docs/codex_tasks/task_card_template.md`

不要阅读或扩展：

`docs/reference/*`

本次目标：

创建或更新：

`docs/codex_tasks/<TASK_CARD_NAME>.md`

要求：

1. 只写 task card。
2. 不要修改代码。
3. 不要运行实验。
4. 不要开始执行该 task card。
5. task card 必须包含：
   - 目标
   - 允许修改的文件
   - 禁止修改的文件
   - 输入文件
   - 输出文件
   - 实现要求
   - 验收标准
   - 运行命令
   - 禁止事项
   - PROGRESS.md 更新要求

完成后只回复：

1. 新增/修改的 task card
2. task card 摘要
3. 下一步建议

不要开始执行 task card。
```
