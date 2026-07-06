# Sprint 1I：Build Unit Evidence

## 1. 目标

本 sprint 只实现一个数据转换阶段：

```text id="pjb3md"
data/processed/semantic_labels.jsonl
+
data/processed/recover_scores.jsonl
→
data/processed/unit_evidence.jsonl
```

本 sprint 按照已经完成的 `unit_evidence` interface，把 semantic necessity evidence 和 semantic recoverability evidence 聚合到 unit-level evidence record 中。

本 sprint 不修改 `unit_evidence` schema，不修改 interface 文档，不生成 attention anchor label，不生成 guidance action。

---

## 2. 当前阶段定位

当前已经完成：

```text id="ttdyuh"
Sprint 1I-prep-a：Unit Evidence Interface Design
```

因此，当前应已有：

```text id="qjwid9"
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
src/recover_attention/schemas.py 中的 REQUIRED_FIELDS["unit_evidence"]
src/recover_attention/schemas.py 中的 FORBIDDEN_FIELDS["unit_evidence"]
src/recover_attention/schemas.py 中的 validate_unit_evidence_record
src/recover_attention/schemas.py 中的 INTERFACE_DOCS["unit_evidence"]
```

本 sprint 不再重新设计接口。

本 sprint 只实现：

```text id="k2r9kd"
semantic_labels.jsonl + recover_scores.jsonl → unit_evidence.jsonl
```

---

## 3. 本 sprint 的输入与输出

### 3.1 输入文件

```text id="sk59cb"
data/processed/semantic_labels.jsonl
data/processed/recover_scores.jsonl
```

### 3.2 输出文件

```text id="gf0vs4"
data/processed/unit_evidence.jsonl
```

### 3.3 新增代码文件

```text id="p6htet"
src/recover_attention/unit_evidence.py
scripts/10_build_unit_evidence.py
tests/test_unit_evidence.py
```

---

## 4. 核心原则

本 sprint 只做 evidence aggregation。

它只回答：

```text id="m1jws7"
这个 ablation unit 当前有哪些 evidence？
```

它不回答：

```text id="a2wqec"
这个 unit 是否是 final attention anchor？
```

它也不输出：

```text id="exj13b"
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
```

关键边界：

```text id="s3cb59"
1. Semantic necessity 是 evidence，不是 final anchor label。
2. Recoverability 是 evidence，不是 final anchor label。
3. Non-recoverable 不能直接等同于 Strong Anchor。
4. Recoverable 不能直接等同于不重要。
5. Misleading Recovery 不能直接等同于 suppress。
6. 当前 recoverability 来自 oracle_stub_v0 + stub_rule_v0，只能用于管线验证。
7. 本 sprint 不做 trajectory stability。
8. 本 sprint 不做 answer stability。
9. 本 sprint 不做 raw attention pattern。
10. 本 sprint 不做 attention steering effect。
```

---

## 5. 权威接口

本 sprint 必须严格遵守：

```text id="ph6qni"
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
src/recover_attention/schemas.py
```

其中：

```text id="kwzb0y"
REQUIRED_FIELDS["unit_evidence"]
FORBIDDEN_FIELDS["unit_evidence"]
validate_unit_evidence_record
```

是 `unit_evidence` 输出的权威 schema 与 validator。

不要在代码中手抄完整 schema 作为新的事实来源。

---

## 6. 开始前必须读取

开始前必须读取：

```text id="sc3mvx"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
src/recover_attention/data_io.py
src/recover_attention/schemas.py
src/recover_attention/semantic_labels.py
src/recover_attention/recover_scoring.py
scripts/06_build_semantic_labels.py
scripts/09_score_recovery.py
tests/test_semantic_labels.py
tests/test_recover_scoring.py
tests/test_schemas.py
data/processed/semantic_labels.jsonl
data/processed/recover_scores.jsonl
docs/progress/sprint_1_history.md
```

不要读取：

```text id="gd3ojn"
docs/reference/*
```

除非用户另行明确要求。

---

## 7. 允许修改

本 sprint 允许修改：

```text id="u8n52o"
src/recover_attention/unit_evidence.py
scripts/10_build_unit_evidence.py
tests/test_unit_evidence.py
data/processed/unit_evidence.jsonl
PROGRESS.md
docs/progress/sprint_1_history.md
```

其中以下文件如果不存在，可以新建：

```text id="madpex"
src/recover_attention/unit_evidence.py
scripts/10_build_unit_evidence.py
tests/test_unit_evidence.py
data/processed/unit_evidence.jsonl
```

---

## 8. 禁止修改

本 sprint 禁止修改：

```text id="x2xgio"
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/prompts.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/masked_questions_interface.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/ablation_units_interface.md
docs/reference/*
src/recover_attention/schemas.py
src/recover_attention/semantic_labels.py
src/recover_attention/recover_scoring.py
src/recover_attention/recover_generation.py
src/recover_attention/masked_questions.py
src/recover_attention/nli_scoring.py
scripts/06_build_semantic_labels.py
scripts/09_score_recovery.py
scripts/11_build_attention_anchor_labels.py
tests/test_schemas.py
tests/test_interface_consistency.py
tests/test_semantic_labels.py
tests/test_recover_scoring.py
configs/*
requirements.txt
pyproject.toml
.gitignore
```

如果发现必须修改禁止列表中的文件，先停止并报告原因，不要自行修改。

---

## 9. Preflight 要求

修改任何文件前，必须先输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text id="foao1l"
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1I-prep-a 已完成。
6. 当前 docs/reasoning-aware-attention-guidance/unit_evidence_interface.md 是否存在。
7. 当前 schemas.py 是否已有 validate_unit_evidence_record。
8. 当前 schemas.py 是否已有 REQUIRED_FIELDS["unit_evidence"]。
9. 当前 schemas.py 是否已有 INTERFACE_DOCS["unit_evidence"]。
10. 当前 data/processed/semantic_labels.jsonl 是否存在。
11. 当前 data/processed/recover_scores.jsonl 是否存在。
12. semantic_labels.jsonl record 数量。
13. recover_scores.jsonl record 数量。
14. semantic_labels 中唯一 (id, unit_id) 数量。
15. recover_scores 中唯一 (id, unit_id) 数量。
16. 当前 src/recover_attention/unit_evidence.py 是否存在。
17. 当前 scripts/10_build_unit_evidence.py 是否存在。
18. 当前 tests/test_unit_evidence.py 是否存在。
19. 本次是否会修改 schema/interface 文档。
20. 本次是否会生成 attention_anchor_labels.jsonl。
21. 本次必须运行的命令。
22. 是否发现冲突。
```

其中第 19 项必须回答：

```text id="h69h57"
否，除非发现已完成的 1I-prep-a 存在阻塞性错误；若发现，先停止并报告。
```

第 20 项必须回答：

```text id="y5nuxf"
否。
```

---

## 10. Join 设计

### 10.1 主 join key

本 sprint 使用：

```text id="kelfw0"
(id, unit_id)
```

作为主 join key。

原因：

```text id="a4njbu"
semantic_labels.jsonl 对同一个 unit 可能有多条 semantic label record。
recover_scores.jsonl 对同一个 unit 应有一条 recover score record。
```

`recover_scores.jsonl` 中的 `masked_id` 应满足：

```python id="o87150"
masked_id = f"{id}__{unit_id}__mask"
```

### 10.2 分组规则

```text id="pyxhlg"
1. semantic_labels 按 (id, unit_id) 分组。
2. recover_scores 按 (id, unit_id) 建唯一索引。
3. 每个 semantic group 必须找到一个 recover_score。
4. 每个 recover_score 必须能对应一个 semantic group。
5. 每个 (id, unit_id) 输出一条 unit_evidence record。
```

### 10.3 缺失处理

本 sprint 采用 fail-fast：

```text id="i9v0ja"
1. semantic group 找不到 recover_score：raise ValueError。
2. recover_score 找不到 semantic group：raise ValueError。
3. recover_scores 中同一个 (id, unit_id) 出现多条：raise ValueError。
4. semantic group 内 source metadata 不一致：raise ValueError。
5. semantic group 与 recover_score 的 unit metadata 不一致：raise ValueError。
```

不要静默跳过 record。

---

## 11. metadata 一致性要求

同一个 `(id, unit_id)` 下，以下字段必须一致：

```text id="tfbfmh"
id
unit_id
unit_scope
group_type
span_ids
spans
original_question
```

semantic group 与 recover_score 之间也必须一致。

如果不一致，报错信息必须指出：

```text id="t75w0o"
1. 出错的 id。
2. 出错的 unit_id。
3. 不一致的字段名。
```

---

## 12. semantic_evidence 聚合规则

对同一个 `(id, unit_id)` 的 semantic label records 聚合。

### 12.1 输入校验

每条 semantic label record 必须通过已有 validator。

使用：

```python id="2nxm7l"
validate_semantic_label_record(record)
```

### 12.2 semantic_evidence 内容

`semantic_evidence` 是 dict。

建议包含：

```text id="wwxuco"
source_semantic_label_ids
source_nli_ids
source_ablation_ids
ablation_types
semantic_necessity_labels
semantic_necessity_scores
is_semantically_necessary_votes
num_semantic_records
num_semantically_necessary
summary_score
summary_label
semantic_label_backend
nli_backends
language_settings
```

其中：

```text id="agz5zf"
source_semantic_label_ids:
  按稳定顺序排列。

source_nli_ids:
  按 semantic label records 顺序排列。

source_ablation_ids:
  按 semantic label records 顺序排列。

ablation_types:
  按 semantic label records 顺序排列。

semantic_necessity_labels:
  按 semantic label records 顺序排列。

semantic_necessity_scores:
  按 semantic label records 顺序排列。

is_semantically_necessary_votes:
  按 semantic label records 顺序排列。

num_semantic_records:
  semantic label record 数量。

num_semantically_necessary:
  is_semantically_necessary == true 的数量。

summary_score:
  max(semantic_necessity_scores)。

summary_label:
  如果 num_semantically_necessary == 0:
      "no_semantic_necessity_evidence"
  如果 num_semantically_necessary == num_semantic_records:
      "consistent_semantic_necessity_evidence"
  否则:
      "mixed_semantic_necessity_evidence"
```

注意：

```text id="r6h6z9"
summary_label 只放在 semantic_evidence dict 内。
不要新增顶层 enum。
不要把 summary_label 当成 attention_anchor_label。
```

### 12.3 稳定排序

为了输出可复现，semantic label records 建议按以下 key 排序：

```python id="aqew65"
(record["id"], record["unit_id"], record["ablation_type"], record["semantic_label_id"])
```

如果 `semantic_label_id` 不存在，说明输入不符合当前 schema，应由 validator 报错。

---

## 13. recoverability_evidence 聚合规则

`recover_scores.jsonl` 每个 `(id, unit_id)` 应有一条 record。

### 13.1 输入校验

每条 recover score record 必须通过：

```python id="oqmyrg"
validate_recover_score_record(record)
```

### 13.2 recoverability_evidence 内容

`recoverability_evidence` 是 dict。

建议包含：

```text id="zoj6uf"
recover_score_id
masked_id
recovery_backend
score_backend
recoverability_label
recoverability_score
confidence_mean
recovery_consistency
misleading_recovery
num_samples
source_sample_ids
source_recovered_questions
score_evidence
```

其中：

```text id="9b97r7"
source_recovered_questions
```

来自 recover_score record 的：

```text id="96849n"
recovered_questions
```

但注意：

```text id="74vylg"
不要把 recovered_questions 写成 unit_evidence 的顶层字段。
只能放在 recoverability_evidence dict 内。
```

---

## 14. unit_evidence record 构造规则

每个 `(id, unit_id)` 输出一条 unit_evidence record。

顶层字段以：

```text id="soc43h"
REQUIRED_FIELDS["unit_evidence"]
```

为准。

### 14.1 ID 规则

使用：

```python id="m1cyq1"
unit_evidence_id = f"{id}__{unit_id}__evidence_{evidence_backend}"
```

默认 backend：

```text id="qmqtpd"
aggregate_stub_v0
```

### 14.2 available_signal_types

当前固定为：

```python id="9fr9uo"
[
    "semantic_necessity",
    "semantic_recoverability",
]
```

### 14.3 missing_signal_types

当前固定为：

```python id="z49rwu"
[
    "trajectory_stability",
    "answer_stability",
    "raw_attention_pattern",
    "attention_steering_effect",
]
```

### 14.4 evidence_status

当前固定为：

```text id="sjz3f4"
partial_stub_evidence
```

### 14.5 evidence

`evidence` 是 dict。

建议包含：

```text id="dfveva"
source_files
join_key
available_signal_types
missing_signal_types
limitations
notes
```

其中 `limitations` 必须说明：

```text id="54rl90"
1. recoverability comes from oracle_stub_v0 + stub_rule_v0.
2. no trajectory stability yet.
3. no answer stability yet.
4. no raw attention pattern yet.
5. no attention steering effect yet.
6. unit_evidence is not final attention anchor label.
```

### 14.6 输出校验

每条输出 record 写入前必须调用：

```python id="iu5kfn"
validate_unit_evidence_record(record)
```

---

## 15. 实现要求

新增：

```text id="ru9a78"
src/recover_attention/unit_evidence.py
```

建议实现以下常量：

```python id="mgtj3k"
DEFAULT_UNIT_EVIDENCE_BACKEND = "aggregate_stub_v0"
SUPPORTED_UNIT_EVIDENCE_BACKENDS = {"aggregate_stub_v0"}
```

建议实现以下函数：

```python id="wnfy24"
build_semantic_evidence_summary(semantic_label_records: list[dict]) -> dict

build_recoverability_evidence_summary(recover_score_record: dict) -> dict

build_unit_evidence_record(
    semantic_label_records: list[dict],
    recover_score_record: dict,
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> dict

build_unit_evidence_records(
    semantic_label_records: list[dict],
    recover_score_records: list[dict],
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> tuple[list[dict], dict]

build_unit_evidence_file(
    semantic_labels_path: str | Path,
    recover_scores_path: str | Path,
    output_path: str | Path,
    evidence_backend: str = DEFAULT_UNIT_EVIDENCE_BACKEND,
) -> tuple[list[dict], dict]
```

可以新增私有 helper 函数，例如：

```python id="wmu7i2"
_group_semantic_labels_by_unit
_index_recover_scores_by_unit
_validate_backend
_validate_unit_metadata_consistency
_sort_semantic_label_records
_build_stats
```

要求代码风格和已有模块保持一致，例如 `recover_scoring.py`、`semantic_labels.py`、`masked_questions.py` 的风格。

---

## 16. CLI 要求

新增：

```text id="m17td4"
scripts/10_build_unit_evidence.py
```

CLI 参数：

```text id="hi5d8v"
--semantic-labels
--recover-scores
--output
--backend
```

其中：

```text id="ou9qbm"
--semantic-labels  必填
--recover-scores   必填
--output           必填
--backend          默认 aggregate_stub_v0
```

示例命令：

```bash id="gud1xc"
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
```

CLI 行为：

```text id="ddzbed"
1. 读取 semantic_labels.jsonl。
2. 读取 recover_scores.jsonl。
3. 调用 build_unit_evidence_file。
4. 写出 unit_evidence.jsonl。
5. 对每条输出调用 validate_unit_evidence_record。
6. 打印 summary stats。
7. 输入文件缺失时给出明确 FileNotFoundError。
8. backend 不支持时给出明确 Unsupported unit evidence backend。
```

建议打印统计：

```text id="rc9lbn"
num_semantic_labels
num_recover_scores
num_output_unit_evidence
evidence_backend
available_signal_types
missing_signal_types
evidence_status_counts
unit_scope_counts
group_type_counts
semantic_summary_label_counts
recoverability_label_counts
num_misleading_recovery
```

---

## 17. 测试要求

新增：

```text id="zylir1"
tests/test_unit_evidence.py
```

至少覆盖以下测试。

### 17.1 正常构造

```text id="bwr6q9"
1. 单个 semantic label + 单个 recover score → valid unit_evidence record。
2. 同一 unit 多条 semantic labels → 聚合为一条 unit_evidence record。
3. 多个 unit → 输出多条 unit_evidence records。
4. 输出 record 全部通过 validate_unit_evidence_record。
```

### 17.2 semantic_evidence 聚合

```text id="xk3q6d"
1. source_semantic_label_ids 顺序稳定。
2. ablation_types 顺序稳定。
3. summary_score = max(semantic_necessity_scores)。
4. 全部 semantically necessary → consistent_semantic_necessity_evidence。
5. 部分 semantically necessary → mixed_semantic_necessity_evidence。
6. 全部 not necessary → no_semantic_necessity_evidence。
```

### 17.3 recoverability_evidence 聚合

```text id="fptxaq"
1. recover_score_id / masked_id 被正确放入 recoverability_evidence。
2. recoverability_label / recoverability_score 被正确复制。
3. recovered_questions 不作为顶层字段出现。
4. score evidence 被放入 recoverability_evidence["score_evidence"]。
```

### 17.4 join 和错误处理

```text id="o9o3d8"
1. semantic group 找不到 recover_score → ValueError。
2. recover_score 找不到 semantic group → ValueError。
3. recover_scores 中同一 (id, unit_id) 重复 → ValueError。
4. semantic group 内 unit metadata 不一致 → ValueError。
5. semantic group 与 recover_score metadata 不一致 → ValueError。
6. unsupported backend → ValueError。
7. semantic_labels_path 缺失 → FileNotFoundError。
8. recover_scores_path 缺失 → FileNotFoundError。
```

### 17.5 禁止字段

验证输出 record 顶层不包含：

```text id="vgryl0"
span_id
span_text
span_type
sample_id
recovered_question
recoverable
confidence
reason
attention_importance_score
attention_anchor_label
guidance_action
guidance_strength
hidden_states
attention_maps
trajectory_analysis
probe_label
```

### 17.6 CLI smoke test

用 `tmp_path` 构造小输入，运行：

```text id="v5t2ev"
scripts/10_build_unit_evidence.py
```

确认：

```text id="lwr6ww"
1. 输出文件存在。
2. 输出 jsonl 可读。
3. 输出 record 数量正确。
4. 输出 record 通过 validate_unit_evidence_record。
```

---

## 18. 数据产物要求

本 sprint 可以生成：

```text id="d0cxj3"
data/processed/unit_evidence.jsonl
```

但必须注意：

```text id="rsxr93"
1. data/processed/* 可能被 .gitignore 忽略。
2. PROGRESS.md 可以记录生成结果，但不要声称该文件一定提交到 GitHub。
3. unit_evidence 当前只包含 semantic necessity 和 semantic recoverability 两类 early evidence。
4. recoverability 部分来自 oracle_stub_v0 + stub_rule_v0，只用于管线验证。
5. unit_evidence 不是 final attention anchor label。
```

---

## 19. 本 sprint 不做

本 sprint 不做：

```text id="xrejbi"
1. 不修改 unit_evidence schema。
2. 不修改 unit_evidence_interface.md。
3. 不修改 semantic_labels_interface.md。
4. 不修改 recover_scores_interface.md。
5. 不修改 label_schema.md。
6. 不修改 SKILL.md。
7. 不修改 schemas.py。
8. 不新增 attention anchor label schema。
9. 不新增 attention anchor label builder。
10. 不新增 scripts/11_build_attention_anchor_labels.py。
11. 不生成 attention_anchor_labels.jsonl。
12. 不输出 attention_importance_score。
13. 不输出 attention_anchor_label。
14. 不输出 guidance_action。
15. 不输出 guidance_strength。
16. 不调用真实模型。
17. 不做 semantic similarity。
18. 不做 trajectory stability。
19. 不做 answer stability。
20. 不读取 hidden states。
21. 不读取 attention maps。
22. 不训练 probe。
23. 不做 attention guidance。
24. 不自动开始 Sprint 1J-prep。
```

---

## 20. 必须运行命令

至少运行：

```bash id="gq3w6h"
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_interface_consistency.py -q
conda run -n recover_attention python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
conda run -n recover_attention python -m pytest tests/test_unit_evidence.py -q
conda run -n recover_attention python -m pytest -q
```

建议在运行 1I 前确认上游输入仍可生成：

```bash id="jzrse6"
conda run -n recover_attention python scripts/06_build_semantic_labels.py --input data/processed/nli_scores.jsonl --output data/processed/semantic_labels.jsonl --backend rule_v0 --equivalent-threshold 0.70 --directional-entailment-threshold 0.50 --contradiction-threshold 0.50
conda run -n recover_attention python scripts/09_score_recovery.py --input data/processed/recover_outputs.jsonl --output data/processed/recover_scores.jsonl --backend stub_rule_v0
```

如果当前 shell 已明确处于 `recover_attention` 环境，也可以使用：

```bash id="sj5eot"
python scripts/sync_interface_fields.py --check
python -m pytest tests/test_interface_consistency.py -q
python scripts/10_build_unit_evidence.py --semantic-labels data/processed/semantic_labels.jsonl --recover-scores data/processed/recover_scores.jsonl --output data/processed/unit_evidence.jsonl --backend aggregate_stub_v0
python -m pytest tests/test_unit_evidence.py -q
python -m pytest -q
```

但最终回复中必须说明实际使用的是哪一种。

---

## 21. PROGRESS.md 更新要求

更新：

```text id="k05ige"
PROGRESS.md
```

要求：

```text id="uvvhzb"
1. 当前阶段更新为 Sprint 1I 已完成：Build Unit Evidence。
2. 已完成 Sprint 摘要中新增 Sprint 1I。
3. 当前可运行命令中新增 scripts/10_build_unit_evidence.py 命令。
4. 最近一次检查结果中记录 unit evidence build passed。
5. 当前关键文件状态中新增：
   src/recover_attention/unit_evidence.py
   scripts/10_build_unit_evidence.py
   tests/test_unit_evidence.py
   data/processed/unit_evidence.jsonl
6. 遗留问题中说明：
   - unit_evidence 目前只汇总 semantic necessity 与 semantic recoverability early evidence。
   - recoverability 来自 oracle_stub_v0 + stub_rule_v0，只用于管线验证。
   - trajectory stability、answer stability、raw attention pattern、attention steering effect 尚未接入。
   - unit_evidence 不是 final attention anchor label。
   - 尚未实现 attention_anchor_labels builder。
7. 下一步建议：
   Sprint 1J-prep：Attention Anchor Label Interface Alignment
```

不要声称已经实现 attention anchor labeling。

---

## 22. docs/progress/sprint_1_history.md 更新要求

更新：

```text id="hc96dc"
docs/progress/sprint_1_history.md
```

追加小节：

```text id="rj3xx0"
## Sprint 1I：Build Unit Evidence
```

内容包括：

```text id="ctb1ao"
1. 已完成内容。
2. 新增或修改文件。
3. 输入文件：
   data/processed/semantic_labels.jsonl
   data/processed/recover_scores.jsonl
4. 输出文件：
   data/processed/unit_evidence.jsonl
5. 运行命令。
6. 检查结果。
7. unit_evidence 数量统计。
8. semantic summary label 分布。
9. recoverability label 分布。
10. 遗留问题。
11. 下一步建议：Sprint 1J-prep：Attention Anchor Label Interface Alignment。
```

---

## 23. 验收标准

本 sprint 完成后必须满足：

```text id="jqk7yd"
1. src/recover_attention/unit_evidence.py 已实现。
2. scripts/10_build_unit_evidence.py 已实现。
3. tests/test_unit_evidence.py 已实现。
4. data/processed/unit_evidence.jsonl 可由 semantic_labels.jsonl + recover_scores.jsonl 生成。
5. 每个 (id, unit_id) 聚合为一条 unit_evidence record。
6. 输出 record 全部通过 validate_unit_evidence_record。
7. 输出 record 顶层不包含 forbidden fields。
8. semantic_evidence 正确汇总 semantic labels。
9. recoverability_evidence 正确汇总 recover score。
10. available_signal_types 当前为 semantic_necessity + semantic_recoverability。
11. missing_signal_types 当前包含 trajectory_stability、answer_stability、raw_attention_pattern、attention_steering_effect。
12. evidence_status 当前为 partial_stub_evidence。
13. scripts/sync_interface_fields.py --check 通过。
14. tests/test_interface_consistency.py -q 通过。
15. tests/test_unit_evidence.py -q 通过。
16. 全量 pytest 通过。
17. 未修改 schemas.py。
18. 未修改 unit_evidence_interface.md。
19. 未修改 label_schema.md。
20. 未生成 attention_anchor_labels.jsonl。
21. 未实现 attention anchor label builder。
22. 未调用真实模型。
23. 未做 trajectory / answer stability / raw attention / guidance。
24. PROGRESS.md 已更新。
25. docs/progress/sprint_1_history.md 已更新。
```

---

## 24. 完成后回复格式

完成后请按以下格式回复：

```text id="hcxg3t"
1. 本次完成内容
2. 新增/修改文件
3. 运行命令
4. 检查结果
5. 生成数据摘要
6. PROGRESS.md 更新摘要
7. docs/progress/sprint_1_history.md 更新摘要
8. 遗留问题
9. 下一步建议
```

下一步建议必须是：

```text id="tlv6ya"
Sprint 1J-prep：Attention Anchor Label Interface Alignment
```

不要自动开始 Sprint 1J-prep。
