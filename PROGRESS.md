# 实验进度记录：Reasoning-Aware Attention Guidance

## 1. 当前项目状态

当前主线：

```text
Reasoning-Aware Attention Guidance
```

完整研究闭环：

```text
Token / Span Intervention
→ Reasoning Dependency Discovery
→ Attention Importance Hierarchy
→ Attention Steering
→ Stable Reasoning Trajectory
→ Reduced Hallucination
```

当前阶段：

```text
Sprint 0H 已完成。
下一步是 Sprint 1A：Candidate Span Extraction，Baseline CoT / trajectory 相关模块后移到 reasoning-risk 或 trajectory 分析阶段，不作为 Sprint 1 起点。
```

当前不做：

- 真实模型调用
- hidden states 大规模缓存
- trajectory stability
- attention guidance
- probe training
- 大规模实验

## 2. 已完成 Sprint 摘要

| Sprint | 状态 | 摘要 |
|---|---|---|
| Sprint 0A | 完成 | 项目骨架与文档约束 |
| Sprint 0A-docs | 完成 | docs/skill、docs/reference、docs/codex_tasks 结构 |
| Sprint 0B | 完成 | jsonl 读写、样例数据、smoke test |
| Sprint 0C | 完成 | 基础 schema 校验 |
| Sprint 0D | 完成 | prepare_data |
| Sprint 0E | 完成 | 工程基础验收 |
| Sprint 0F | 完成 | 文档主线对齐 |
| Sprint 0G | 完成 | schema 与 Attention Anchor 标签体系对齐 |
| Sprint 0H | 完成 | PROGRESS.md 瘦身与 Sprint 0 历史归档 |

详细历史见：

```text
docs/progress/sprint_0_history.md
```

## 3. 当前可运行命令

```bash
python scripts/00_smoke_test.py --config configs/v0_nli_small.yaml
python scripts/01_prepare_data.py --input data/examples/questions_small.jsonl --output data/processed/questions.jsonl
python -m pytest -q
```

最近一次检查结果：

```text
pytest: 38 passed
smoke test: passed
prepare_data: passed
```

## 4. 当前关键文件状态

已完成：

- src/recover_attention/data_io.py
- src/recover_attention/schemas.py
- src/recover_attention/prepare_data.py
- scripts/00_smoke_test.py
- scripts/01_prepare_data.py
- tests/test_data_io.py
- tests/test_schemas.py
- tests/test_prepare_data.py
- docs/skill/*
- README.md
- AGENTS.md

下一阶段将新增或修改：

- src/recover_attention/baseline_cot.py
- scripts/02_build_baseline_cot.py
- tests/test_baseline_cot.py

具体以 Sprint 1A task card 为准。

## 5. 当前遗留问题

- git 工作区中可能仍存在 Sprint 0F 文档迁移痕迹，需要在提交前检查。
- 如果 `__pycache__` / `.pyc` 出现在 git status 中，需要确认是否被 git 跟踪；若已被跟踪，应单独处理。
- 不要直接进入旧的 candidate span extraction 路线。

## 6. 下一步

下一步建议：

```text
Sprint 1A：Baseline CoT Schema
```

注意：

```text
不要自动开始 Sprint 1A。
必须先有 Sprint 1A task card 或用户明确指令。
```
