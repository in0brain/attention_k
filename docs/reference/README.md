# Reference Documents

本目录保存项目的完整参考文档，包括完整实验指导书、完整项目方案和进度记录模板。

这些文件只作为长期参考，不是 Codex 每轮任务的直接执行入口。

Codex 执行任务时应优先遵守：

1. `AGENTS.md`
2. `PROGRESS.md`
3. `docs/skill/SKILL.md`
4. 当前 sprint 对应的 task card

除非用户明确要求，否则不要根据本目录中的完整文档自行扩展任务。

---

## Archive 说明（重要）

本目录是 **archive / 历史参考**，不是权威接口来源。

```text
1. 本目录中的完整方案、字段示例和 pipeline 顺序可能描述早期设计
   （例如旧版 span-level masked / recover schema、不同的脚本编号）。
2. 这些内容与当前 unit-level 主线可能不一致，且不会随当前接口同步更新。
3. 当前权威 schema 来源是：
   - src/recover_attention/schemas.py 的 REQUIRED_FIELDS / FORBIDDEN_FIELDS
   - docs/skill/<artifact>_interface.md
4. 当前实验流程以 docs/skill/experiment_guide.md 和当前 task card 为准。
```

如果本目录内容与 `docs/skill/*_interface.md` 或 `schemas.py` 冲突，
一律以后者为准，并把本目录视为背景资料，不要据此实现或修改代码。
