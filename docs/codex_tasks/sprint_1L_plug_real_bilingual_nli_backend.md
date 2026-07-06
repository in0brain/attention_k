# Sprint 1L：Plug Real Bilingual NLI Backend

建议保存路径：

```text id="i3dsft"
docs/codex_tasks/sprint_1L_plug_real_bilingual_nli_backend.md
```

---

## 1. 目标

本 sprint 接入真实 NLI backend，用于替换当前 deterministic `stub_v0` NLI scoring。

当前 NLI 管线已经存在：

```text id="qymhqr"
data/processed/ablated_questions.jsonl
→
data/processed/nli_scores.jsonl
```

本 sprint 的目标是在不改变 `nli_scores.jsonl` 顶层 schema 的前提下，新增真实 HuggingFace NLI backend，并支持：

```text id="g27hbw"
1. 本地模型路径加载。
2. 联网下载 / HuggingFace model id 加载。
3. 英文 NLI。
4. 中文 / 多语言 NLI。
5. auto 双语路由。
```

本 sprint 的核心原则：

```text id="wijhe9"
本地优先，显式允许后才联网。
```

也就是说：

```text id="sk96yx"
1. 如果用户给的是本地路径，并且路径存在：直接加载本地模型。
2. 如果用户给的是 HuggingFace repo id，或者本地路径不存在：
   只有在 --allow-download 打开时才允许联网下载 / 从 HuggingFace Hub 拉取。
3. 如果未打开 --allow-download，模型路径缺失时必须直接报错。
4. pytest 默认不得下载模型。
5. stub_v0 不依赖 torch / transformers。
```

---

## 2. 当前项目已有基础

当前项目已经支持：

```text id="y3x0pe"
--language auto
--language en
--language zh
```

并且 `nli_score` record 已经包含：

```text id="raq3zy"
language
language_setting
forward
backward
bidirectional_entailment_score
contradiction_score
```

当前缺失的是：

```text id="r8s0w9"
真实 NLI backend
```

当前 backend 只有：

```text id="gpu2pi"
stub_v0
```

本 sprint 应新增真实 backend，但不要破坏 stub。

---

## 3. 本 sprint 新增 backend 设计

新增 NLI backends：

```text id="tamczt"
hf_nli_en_v0
hf_nli_zh_v0
hf_nli_auto_v0
```

含义：

```text id="r2rnas"
hf_nli_en_v0:
  使用英文 NLI 模型。
  默认要求 resolved language == en。

hf_nli_zh_v0:
  使用中文 / multilingual NLI 模型。
  默认要求 resolved language == zh。

hf_nli_auto_v0:
  根据 resolved language 自动选择英文或中文模型。
```

注意：

```text id="dv03hq"
hf_nli_auto_v0 不是新的输出 schema。
它只是 backend routing policy。
```

---

## 4. 推荐模型

本 sprint 默认推荐以下模型路径和 HuggingFace repo id。

### 4.1 英文 NLI

本地路径：

```text id="evkqj5"
models/nli/en/roberta-large-mnli
```

远程 repo id：

```text id="krfjf5"
FacebookAI/roberta-large-mnli
```

### 4.2 中文 / 多语言 NLI

本地路径：

```text id="ku8cvy"
models/nli/zh/mdeberta-v3-base-xnli
```

远程 repo id：

```text id="z4uwxk"
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

### 4.3 中文备选对照模型

本地路径：

```text id="qi8cau"
models/nli/zh_alt/erlangshen-roberta-330m-nli
```

远程 repo id：

```text id="j7gctv"
IDEA-CCNL/Erlangshen-Roberta-330M-NLI
```

注意：

```text id="etyqoj"
zh_alt 只作为后续对照模型。
本 sprint 主线默认只使用：
  models/nli/en/roberta-large-mnli
  models/nli/zh/mdeberta-v3-base-xnli
```

---

## 5. 模型加载策略

本 sprint 必须实现强约束模型加载策略。

### 5.1 默认行为

默认：

```text id="q8dfal"
allow_download = False
```

也就是说，如果用户传入：

```text id="h0kg1a"
--en-model models/nli/en/roberta-large-mnli
--zh-model models/nli/zh/mdeberta-v3-base-xnli
```

则程序必须先检查本地路径是否存在。

如果本地路径存在：

```text id="vj5v1x"
加载本地模型。
```

如果本地路径不存在：

```text id="efrs6z"
直接报错，不允许偷偷联网下载。
```

错误信息必须包含缺失路径，例如：

```text id="kg1rjv"
Local NLI model path does not exist: models/nli/en/roberta-large-mnli
Pass --allow-download with a HuggingFace repo id, or download the model into models/nli/ first.
```

### 5.2 显式允许联网

只有用户显式传入：

```text id="usncpr"
--allow-download
```

时，才允许：

```text id="ae6qjc"
1. 从 HuggingFace repo id 下载模型。
2. 使用 transformers 自动从 Hub 拉取模型。
3. 将模型缓存到本地 HuggingFace cache。
```

如果用户传入的是 repo id：

```text id="qny4yv"
--en-model FacebookAI/roberta-large-mnli
--allow-download
```

可以联网下载。

如果用户传入的是本地路径但路径不存在：

```text id="gftcq7"
--en-model models/nli/en/roberta-large-mnli
--allow-download
```

不要猜测 repo id。应报错并提示用户改用 repo id 或先下载模型。

### 5.3 本地优先 + 远程 fallback

允许新增 CLI 参数：

```text id="qac5e9"
--en-model-id
--zh-model-id
```

用于 fallback 下载。

推荐语义：

```text id="kbif4y"
--en-model:
  本地英文模型路径，默认 models/nli/en/roberta-large-mnli。

--zh-model:
  本地中文 / 多语言模型路径，默认 models/nli/zh/mdeberta-v3-base-xnli。

--en-model-id:
  英文 HuggingFace repo id，默认 FacebookAI/roberta-large-mnli。

--zh-model-id:
  中文 / 多语言 HuggingFace repo id，默认 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7。

--allow-download:
  如果本地模型路径不存在，允许用 model id 联网下载 / 加载。
```

加载规则：

```text id="po2eum"
1. 先检查 --en-model / --zh-model 本地路径。
2. 如果路径存在，加载本地路径。
3. 如果路径不存在且 --allow-download=False，报错。
4. 如果路径不存在且 --allow-download=True，使用 --en-model-id / --zh-model-id 从 Hub 加载。
5. 不要在未启用 --allow-download 时使用 repo id 自动联网。
```

---

## 6. 语言规则

本 sprint 必须保留现有语义：

```text id="e1nqsk"
language_setting:
  用户传入的语言设置，取值 auto / en / zh。

language:
  每条 record 最终 resolved language，取值 en / zh。
```

规则：

```text id="qd3fhj"
1. --language en：
   所有 record 的 language = en。

2. --language zh：
   所有 record 的 language = zh。

3. --language auto：
   每条 record 根据 original_question + ablated_question 自动检测。
   含 CJK 字符则 zh，否则 en。
```

真实 backend 选择规则：

```text id="w7r2qg"
backend = hf_nli_en_v0:
  使用英文 NLI model。
  若 resolved language != en，默认 raise ValueError。

backend = hf_nli_zh_v0:
  使用中文 / multilingual NLI model。
  若 resolved language != zh，默认 raise ValueError。

backend = hf_nli_auto_v0:
  resolved language == en → English model。
  resolved language == zh → Chinese / multilingual model。
```

不要在本 sprint 中加入自动翻译。

---

## 7. 本 sprint 的输入与输出

### 7.1 输入文件

```text id="wpsgya"
data/processed/ablated_questions.jsonl
```

### 7.2 输出文件

默认仍然是：

```text id="b5129r"
data/processed/nli_scores.jsonl
```

真实 NLI smoke run 建议输出到独立文件，避免覆盖 stub 产物：

```text id="fq8xpp"
data/processed/nli_scores_real_auto_small.jsonl
data/processed/nli_scores_real_en_small.jsonl
data/processed/nli_scores_real_zh_small.jsonl
```

### 7.3 不新增新的主数据 schema

本 sprint 不新增：

```text id="snvw7d"
real_nli_scores.jsonl
```

仍然复用：

```text id="ns5ua2"
nli_scores.jsonl schema
```

---

## 8. 允许修改

本 sprint 允许修改：

```text id="m5z5xb"
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
tests/test_nli_scoring.py
src/recover_attention/schemas.py
requirements.txt
configs/v0_nli_small.yaml
PROGRESS.md
docs/progress/sprint_1_history.md
```

其中：

```text id="g7trf9"
requirements.txt
```

只允许添加 optional real NLI dependencies 的注释或最小依赖说明。

如果本地环境已经有 `torch` / `transformers`，可以不强行写入 requirements。

---

## 9. 禁止修改

本 sprint 禁止修改：

```text id="eo5xiu"
README.md
AGENTS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/experiment_guide.md
docs/reasoning-aware-attention-guidance/method.md
docs/reasoning-aware-attention-guidance/label_schema.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/ablated_questions_interface.md
docs/reasoning-aware-attention-guidance/recover_outputs_interface.md
docs/reasoning-aware-attention-guidance/recover_scores_interface.md
docs/reasoning-aware-attention-guidance/unit_evidence_interface.md
docs/reasoning-aware-attention-guidance/attention_anchor_labels_interface.md
docs/reasoning-aware-attention-guidance/intervention_manifest_interface.md
docs/reference/*
src/recover_attention/semantic_labels.py
src/recover_attention/question_ablations.py
src/recover_attention/masked_questions.py
src/recover_attention/recover_generation.py
src/recover_attention/recover_scoring.py
src/recover_attention/unit_evidence.py
src/recover_attention/attention_anchor_labels.py
src/recover_attention/intervention_manifest.py
scripts/06_build_semantic_labels.py
scripts/08_run_recovery.py
scripts/09_score_recovery.py
scripts/10_build_unit_evidence.py
scripts/11_build_attention_anchor_labels.py
scripts/12_build_intervention_manifest.py
tests/test_semantic_labels.py
tests/test_recover_generation.py
tests/test_recover_scoring.py
tests/test_unit_evidence.py
tests/test_attention_anchor_labels.py
tests/test_intervention_manifest.py
data/processed/*
pyproject.toml
.gitignore
models/*
```

例外：

```text id="iv5ufl"
data/processed/nli_scores_real_auto_small.jsonl
data/processed/nli_scores_real_en_small.jsonl
data/processed/nli_scores_real_zh_small.jsonl
```

可以由真实 NLI smoke run 生成，但如果 `data/processed` 被 gitignore 忽略，不要求提交。

注意：

```text id="r4qqml"
本 sprint 不允许修改、删除、移动 models/*。
models/* 是本地模型资产，不应提交到 Git。
```

如果发现必须修改禁止列表中的文件，先停止并报告原因。

---

## 10. 开始前必须读取

开始前必须读取：

```text id="lrlp9u"
AGENTS.md
PROGRESS.md
docs/reasoning-aware-attention-guidance/SKILL.md
docs/reasoning-aware-attention-guidance/codex_tasks.md
docs/reasoning-aware-attention-guidance/nli_scores_interface.md
docs/reasoning-aware-attention-guidance/semantic_labels_interface.md
docs/reasoning-aware-attention-guidance/label_schema.md
src/recover_attention/schemas.py
src/recover_attention/nli_scoring.py
scripts/05_run_nli_scoring.py
tests/test_nli_scoring.py
requirements.txt
configs/v0_nli_small.yaml
data/processed/ablated_questions.jsonl
docs/progress/sprint_1_history.md
```

还必须检查，但不要修改：

```text id="hjo6xb"
models/nli/en/roberta-large-mnli
models/nli/zh/mdeberta-v3-base-xnli
models/nli/zh_alt/erlangshen-roberta-330m-nli
```

不要读取：

```text id="z9d8ey"
docs/reference/*
```

除非用户另行明确要求。

---

## 11. Preflight 要求

修改任何文件前，必须输出 Preflight，并等待用户确认。

Preflight 必须包含：

```text id="ah235g"
1. 已阅读文件列表。
2. 本次允许修改的文件。
3. 本次禁止修改的文件。
4. 是否需要读取 docs/reference/*。
5. 当前 PROGRESS.md 是否显示 Sprint 1K 已完成。
6. 当前 nli_scoring.py 是否已有 detect_language。
7. 当前 nli_scoring.py 是否支持 language auto/en/zh。
8. 当前 schemas.py 中 ALLOWED_NLI_BACKENDS 的值。
9. 当前 schemas.py 中 ALLOWED_NLI_LANGUAGES 的值。
10. 当前 schemas.py 中 ALLOWED_NLI_LANGUAGE_SETTINGS 的值。
11. 当前 scripts/05_run_nli_scoring.py 是否已有 --language。
12. 当前 tests/test_nli_scoring.py 是否已有中文 auto 测试。
13. 当前环境是否安装 transformers。
14. 当前环境是否安装 torch。
15. 当前 models/nli/en/roberta-large-mnli 是否存在。
16. 当前 models/nli/zh/mdeberta-v3-base-xnli 是否存在。
17. 当前 models/nli/zh_alt/erlangshen-roberta-330m-nli 是否存在；该项可选。
18. 英文模型目录是否包含 config.json。
19. 中文模型目录是否包含 config.json。
20. 英文模型目录是否包含 tokenizer 相关文件。
21. 中文模型目录是否包含 tokenizer 相关文件。
22. 英文模型目录是否包含 model.safetensors 或 pytorch_model.bin。
23. 中文模型目录是否包含 model.safetensors 或 pytorch_model.bin。
24. 本次是否会修改 nli_scores 顶层 schema。
25. 本次是否会修改 semantic_labels 规则。
26. 本次是否会接入 LLM recovery。
27. 本次是否会调用 attention guidance。
28. 本次是否允许自动联网下载模型。
29. 本次必须运行的命令。
30. 是否发现冲突。
```

第 24 项必须回答：

```text id="exx084"
否。只新增 backend enum 和 backend implementation，不改变 nli_score 顶层字段。
```

第 25 项必须回答：

```text id="erfcc9"
否。
```

第 26 项必须回答：

```text id="al21kn"
否。
```

第 27 项必须回答：

```text id="ehor7h"
否。
```

第 28 项必须回答：

```text id="x3o8wb"
默认否。只有用户显式传 --allow-download 时才允许联网下载或从 Hub 拉取模型。
```

---

## 12. schema 更新要求

只允许在 `src/recover_attention/schemas.py` 中扩展 backend enum。

将：

```python id="mjbm4j"
ALLOWED_NLI_BACKENDS = {"stub_v0"}
```

扩展为：

```python id="jmtaec"
ALLOWED_NLI_BACKENDS = {
    "stub_v0",
    "hf_nli_en_v0",
    "hf_nli_zh_v0",
    "hf_nli_auto_v0",
}
```

不要修改：

```text id="rx3y14"
REQUIRED_FIELDS["nli_score"]
FORBIDDEN_FIELDS["nli_score"]
validate_nli_score_record
ALLOWED_NLI_LABELS
ALLOWED_NLI_LANGUAGES
ALLOWED_NLI_LANGUAGE_SETTINGS
```

---

## 13. nli_scoring.py 实现要求

修改：

```text id="aejwkq"
src/recover_attention/nli_scoring.py
```

### 13.1 保留 stub

必须保留：

```text id="n3mx0o"
stub_v0
score_nli_pair_stub
现有 deterministic stub 行为
现有 tests 不应破坏
```

### 13.2 新增真实 backend 常量

建议新增：

```python id="id9pri"
STUB_BACKEND = "stub_v0"
HF_NLI_EN_BACKEND = "hf_nli_en_v0"
HF_NLI_ZH_BACKEND = "hf_nli_zh_v0"
HF_NLI_AUTO_BACKEND = "hf_nli_auto_v0"

SUPPORTED_BACKENDS = {
    STUB_BACKEND,
    HF_NLI_EN_BACKEND,
    HF_NLI_ZH_BACKEND,
    HF_NLI_AUTO_BACKEND,
}
```

### 13.3 默认模型路径和 repo id

建议新增：

```python id="e6yjpq"
DEFAULT_EN_NLI_MODEL = "models/nli/en/roberta-large-mnli"
DEFAULT_ZH_NLI_MODEL = "models/nli/zh/mdeberta-v3-base-xnli"

DEFAULT_EN_NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"
DEFAULT_ZH_NLI_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

OPTIONAL_ZH_ALT_NLI_MODEL = "models/nli/zh_alt/erlangshen-roberta-330m-nli"
OPTIONAL_ZH_ALT_NLI_MODEL_ID = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
```

主线只使用：

```text id="strgad"
DEFAULT_EN_NLI_MODEL
DEFAULT_ZH_NLI_MODEL
```

不要把 `zh_alt` 自动接入主 pipeline。

### 13.4 backend config 参数

建议支持以下函数参数：

```text id="gucy0g"
en_model
zh_model
en_model_id
zh_model_id
allow_download
device
batch_size
max_length
label_order
```

其中：

```text id="vmjnmu"
en_model:
  英文本地模型路径，默认 models/nli/en/roberta-large-mnli。

zh_model:
  中文 / multilingual 本地模型路径，默认 models/nli/zh/mdeberta-v3-base-xnli。

en_model_id:
  英文远程 repo id，默认 FacebookAI/roberta-large-mnli。

zh_model_id:
  中文 / multilingual 远程 repo id，默认 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7。

allow_download:
  默认 False。
  只有 True 时才允许联网拉取模型。

device:
  auto / cpu / cuda。

batch_size:
  初期可以保留参数，不必真的做高性能批处理。
  默认 4。

max_length:
  tokenizer max_length，默认 512。

label_order:
  默认 auto。
```

### 13.5 依赖导入

真实 backend 需要：

```text id="ob2u17"
transformers
torch
```

但测试不能因为缺少这些库而失败。

要求：

```text id="g8k2vj"
1. 不在模块 import 顶层强制 import transformers / torch。
2. 只在真实 backend 被调用时 lazy import。
3. 如果缺少依赖，raise ImportError，错误信息说明：
   pip install torch transformers
4. stub_v0 不需要 transformers / torch。
5. pytest 默认不下载模型。
```

### 13.6 本地 / 远程模型解析

建议新增：

```python id="xcpbgk"
resolve_model_source(
    local_model_path: str,
    model_id: str,
    allow_download: bool = False,
) -> str
```

语义：

```text id="azjs97"
1. 如果 local_model_path 是存在的本地路径，返回 local_model_path。
2. 如果 local_model_path 不存在且 allow_download=False，raise FileNotFoundError。
3. 如果 local_model_path 不存在且 allow_download=True，返回 model_id。
4. 如果用户传入的 model 参数本身看起来是 repo id：
   - allow_download=True 时允许返回该 repo id。
   - allow_download=False 时 raise ValueError，提示需要 --allow-download。
```

判断 repo id 可以简单使用：

```text id="aci8nj"
包含 "/" 且本地路径不存在
```

但不要过度复杂化。

### 13.7 模型加载

建议新增：

```python id="v3d1gm"
load_hf_nli_model(
    local_model_path: str,
    model_id: str,
    allow_download: bool = False,
    device: str = "auto",
)
```

返回 tokenizer/model/device 或 dataclass。

要求：

```text id="m9kpz5"
1. 支持本地路径。
2. 支持 repo id，但只有 allow_download=True 时允许。
3. 支持 CPU。
4. device=auto 时，如果 torch.cuda.is_available() 则 cuda，否则 cpu。
5. model.eval()。
6. 不在 pytest 中自动下载模型。
7. 使用 local_files_only=not allow_download。
```

建议 transformers 调用：

```python id="c62smq"
AutoTokenizer.from_pretrained(model_source, local_files_only=not allow_download)
AutoModelForSequenceClassification.from_pretrained(model_source, local_files_only=not allow_download)
```

### 13.8 label mapping

真实 HuggingFace 模型的 label 名可能是：

```text id="d49l13"
ENTAILMENT
NEUTRAL
CONTRADICTION
LABEL_0 / LABEL_1 / LABEL_2
```

必须实现 robust mapping。

要求输出统一为：

```text id="ejylhr"
entailment
neutral
contradiction
```

如果无法识别模型 label mapping，raise ValueError，并打印 `model.config.id2label`。

建议支持：

```text id="owlcn5"
1. id2label 中含 entailment / neutral / contradiction 字样。
2. id2label 是 LABEL_0 / LABEL_1 / LABEL_2 时，允许用户通过 --label-order 指定。
```

### 13.9 label_order 参数

为避免不同模型 label 顺序不一致，CLI 增加：

```text id="btyvj1"
--label-order
```

默认：

```text id="b85y1u"
auto
```

支持格式：

```text id="bqfy40"
auto
contradiction,neutral,entailment
entailment,neutral,contradiction
```

如果 `id2label` 无法自动判断，则要求用户显式设置 `--label-order`。

### 13.10 真实 NLI scoring 函数

建议新增：

```python id="ryh6zv"
score_nli_pair_hf(
    premise: str,
    hypothesis: str,
    model_bundle: object,
    max_length: int = 512,
    label_order: str = "auto",
) -> dict
```

输出必须与 stub 方向 record 一致：

```python id="ra536i"
{
    "premise": premise,
    "hypothesis": hypothesis,
    "label": label,
    "scores": {
        "entailment": float,
        "neutral": float,
        "contradiction": float,
    },
}
```

分数必须满足：

```text id="t3g7qe"
0 <= each score <= 1
sum(scores.values()) ~= 1
```

### 13.11 backend routing

`score_ablated_question_record` 应根据 backend 分流：

```text id="xusj15"
backend == stub_v0:
  使用原 stub 逻辑。

backend == hf_nli_en_v0:
  resolved language 必须是 en。
  使用 en_model / en_model_id。

backend == hf_nli_zh_v0:
  resolved language 必须是 zh。
  使用 zh_model / zh_model_id。

backend == hf_nli_auto_v0:
  resolved language == en 使用 en_model / en_model_id。
  resolved language == zh 使用 zh_model / zh_model_id。
```

---

## 14. 函数签名兼容要求

当前函数：

```python id="oyozb5"
score_ablated_question_record(record: dict, backend: str = "stub_v0", language: str = "auto") -> dict
score_ablated_question_records(records: list[dict], backend: str = "stub_v0", language: str = "auto") -> tuple[list[dict], dict]
```

可以扩展参数，但必须保持旧调用不变。

建议扩展为：

```python id="uunwh9"
score_ablated_question_record(
    record: dict,
    backend: str = "stub_v0",
    language: str = "auto",
    en_model: str | None = None,
    zh_model: str | None = None,
    en_model_id: str | None = None,
    zh_model_id: str | None = None,
    allow_download: bool = False,
    device: str = "auto",
    max_length: int = 512,
    label_order: str = "auto",
    _model_cache: dict | None = None,
) -> dict
```

```python id="wz2yew"
score_ablated_question_records(
    records: list[dict],
    backend: str = "stub_v0",
    language: str = "auto",
    en_model: str | None = None,
    zh_model: str | None = None,
    en_model_id: str | None = None,
    zh_model_id: str | None = None,
    allow_download: bool = False,
    device: str = "auto",
    batch_size: int = 4,
    max_length: int = 512,
    label_order: str = "auto",
) -> tuple[list[dict], dict]
```

注意：

```text id="uf0650"
_model_cache 是内部参数，不暴露到 CLI。
```

---

## 15. nli_id 规则

继续使用：

```python id="qpxyhv"
nli_id = f"{ablation_id}__nli_{backend}"
```

不要把 model path 或 model id 写进 `nli_id`，避免路径过长和不可复现差异。

模型信息可以放入 stats，不进入顶层 schema。

---

## 16. CLI 更新要求

修改：

```text id="so67yd"
scripts/05_run_nli_scoring.py
```

当前已有：

```text id="bmr1ly"
--input
--output
--backend
--language
```

新增：

```text id="at60ny"
--en-model
--zh-model
--en-model-id
--zh-model-id
--allow-download
--device
--batch-size
--max-length
--label-order
--limit
```

默认值：

```text id="buaqzf"
--en-model models/nli/en/roberta-large-mnli
--zh-model models/nli/zh/mdeberta-v3-base-xnli
--en-model-id FacebookAI/roberta-large-mnli
--zh-model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
--allow-download False
--device auto
--batch-size 4
--max-length 512
--label-order auto
--limit None
```

`--allow-download` 必须是显式 flag，不要默认打开。

`--limit` 要求：

```text id="kur5kt"
1. 只在 script 层截断 input_records。
2. 不改变 library 层默认行为。
3. 用于小样本真实模型测试。
```

---

## 17. configs 更新要求

更新：

```text id="flhlsl"
configs/v0_nli_small.yaml
```

允许添加真实 NLI 参考配置，但不要破坏旧配置。

建议：

```yaml id="e62gxp"
nli:
  backend: "stub_v0"
  language: "auto"
  real_backends:
    en:
      backend: "hf_nli_en_v0"
      model: "models/nli/en/roberta-large-mnli"
      model_id: "FacebookAI/roberta-large-mnli"
    zh:
      backend: "hf_nli_zh_v0"
      model: "models/nli/zh/mdeberta-v3-base-xnli"
      model_id: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    auto:
      backend: "hf_nli_auto_v0"
      en_model: "models/nli/en/roberta-large-mnli"
      zh_model: "models/nli/zh/mdeberta-v3-base-xnli"
      en_model_id: "FacebookAI/roberta-large-mnli"
      zh_model_id: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
  allow_download: false
  device: "auto"
  batch_size: 4
  max_length: 512
  label_order: "auto"
```

不要让 smoke test 自动使用 real backend。

---

## 18. requirements 更新要求

当前 `requirements.txt` 只包含基础依赖。

本 sprint 有两种可接受方式。

### 方式 A：不改 requirements.txt

通过 lazy import 让真实 backend 在缺少 `torch/transformers` 时明确报错。

错误信息必须包含：

```text id="fdd0jy"
Real NLI backend requires torch and transformers.
Install them in the recover_attention environment before running hf_nli_* backends.
```

### 方式 B：添加注释说明 optional dependencies

允许添加：

```text id="lqtx90"
# Optional for real NLI backends:
# torch
# transformers
# huggingface_hub
```

不要强行把大依赖加入基础测试环境，除非用户确认。

---

## 19. tests/test_nli_scoring.py 更新要求

必须保留所有现有 stub 测试。

新增测试时，不允许下载真实模型。

新增测试分三类。

### 19.1 不依赖 transformers 的测试

必须覆盖：

```text id="cvfz1h"
1. SUPPORTED_BACKENDS 包含 stub_v0 / hf_nli_en_v0 / hf_nli_zh_v0 / hf_nli_auto_v0。
2. schemas.py 的 ALLOWED_NLI_BACKENDS 包含真实 backend。
3. backend=hf_nli_en_v0 且 language=zh 时 raise ValueError。
4. backend=hf_nli_zh_v0 且 language=en 时 raise ValueError。
5. label_order parser 支持 auto。
6. label_order parser 支持 contradiction,neutral,entailment。
7. invalid label_order raises。
8. allow_download 默认 False。
9. local path missing 且 allow_download=False 时 raise FileNotFoundError。
10. repo id 且 allow_download=False 时 raise ValueError。
11. stub_v0 不受 allow_download / model path 影响。
12. --limit 只输出前 N 条。
```

### 19.2 mock model 测试

用 fake tokenizer / fake model 或 monkeypatch 测试：

```text id="tfbp31"
1. score_nli_pair_hf 输出 entailment/neutral/contradiction。
2. scores sum ~= 1。
3. id2label 自动映射能识别 entailment / neutral / contradiction。
4. LABEL_0 / LABEL_1 / LABEL_2 需要 label_order，否则 raise。
5. hf_nli_auto_v0 英文样本调用 en_model。
6. hf_nli_auto_v0 中文样本调用 zh_model。
7. allow_download=True 时允许使用 repo id。
8. allow_download=False 时 from_pretrained 使用 local_files_only=True。
9. allow_download=True 时 from_pretrained 使用 local_files_only=False。
```

### 19.3 CLI 测试

现有 CLI smoke test 必须继续跑：

```bash id="comqb1"
python scripts/05_run_nli_scoring.py --input <tmp> --output <tmp> --backend stub_v0 --language auto
```

新增：

```text id="vwlpd4"
1. --backend stub_v0 --limit 1 只输出 1 条。
2. 真实 backend 缺本地路径且未传 --allow-download 时，报错清楚。
3. 真实 backend 使用 repo id 但未传 --allow-download 时，报错清楚。
```

真实模型 CLI 测试可以在实际环境中手动运行，不要求普通 pytest 下载模型。

---

## 20. 必须运行命令

至少运行：

```bash id="oxwgns"
conda run -n recover_attention python scripts/sync_interface_fields.py --check
conda run -n recover_attention python -m pytest tests/test_nli_scoring.py -q
conda run -n recover_attention python -m pytest tests/test_schemas.py -q
conda run -n recover_attention python -m pytest -q
```

还必须运行 stub CLI：

```bash id="e7a81o"
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores_stub_check.jsonl --backend stub_v0 --language auto --limit 20
```

还必须运行真实 NLI smoke。优先使用本地模型路径：

```bash id="xpgt8j"
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores_real_auto_small.jsonl --backend hf_nli_auto_v0 --language auto --en-model models/nli/en/roberta-large-mnli --zh-model models/nli/zh/mdeberta-v3-base-xnli --device auto --limit 20
```

如果本地模型路径缺失，但环境允许联网，可以使用显式下载模式：

```bash id="jrb15j"
conda run -n recover_attention python scripts/05_run_nli_scoring.py --input data/processed/ablated_questions.jsonl --output data/processed/nli_scores_real_auto_small.jsonl --backend hf_nli_auto_v0 --language auto --en-model models/nli/en/roberta-large-mnli --zh-model models/nli/zh/mdeberta-v3-base-xnli --en-model-id FacebookAI/roberta-large-mnli --zh-model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --allow-download --device auto --limit 20
```

Windows PowerShell 可使用：

```powershell id="iasf3c"
conda run -n recover_attention python scripts/05_run_nli_scoring.py `
  --input data/processed/ablated_questions.jsonl `
  --output data/processed/nli_scores_real_auto_small.jsonl `
  --backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --device auto `
  --limit 20
```

如需显式允许联网：

```powershell id="ntysvr"
conda run -n recover_attention python scripts/05_run_nli_scoring.py `
  --input data/processed/ablated_questions.jsonl `
  --output data/processed/nli_scores_real_auto_small.jsonl `
  --backend hf_nli_auto_v0 `
  --language auto `
  --en-model models/nli/en/roberta-large-mnli `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --en-model-id FacebookAI/roberta-large-mnli `
  --zh-model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 `
  --allow-download `
  --device auto `
  --limit 20
```

如果真实 smoke 因缺少 torch / transformers、模型文件损坏、显存不足等原因不能运行，必须在最终回复中明确说明原因。

---

## 21. 可选中文专项 smoke

如果 `data/processed/ablated_questions.jsonl` 中没有中文样本，不要强行把英文样本用 `hf_nli_zh_v0` 处理。

可以临时构造中文小样本到 tmp 目录，或者只报告：

```text id="fw9evc"
当前 processed 数据没有中文样本，因此 hf_nli_zh_v0 未做真实中文数据 smoke。
hf_nli_auto_v0 的中文 routing 已由 mock test 覆盖。
```

如果有中文样本，可以运行：

```powershell id="azf7ce"
conda run -n recover_attention python scripts/05_run_nli_scoring.py `
  --input data/processed/ablated_questions.jsonl `
  --output data/processed/nli_scores_real_zh_small.jsonl `
  --backend hf_nli_zh_v0 `
  --language zh `
  --zh-model models/nli/zh/mdeberta-v3-base-xnli `
  --device auto `
  --limit 20
```

---

## 22. PROGRESS.md 更新要求

更新：

```text id="e5ohv9"
PROGRESS.md
```

要求：

```text id="kjxdmh"
1. 当前阶段更新为 Sprint 1L 已完成：Plug Real Bilingual NLI Backend。
2. 已完成 Sprint 摘要中新增 Sprint 1L。
3. 当前可运行命令中更新 scripts/05_run_nli_scoring.py 的参数说明，保留 stub 命令，并新增真实 NLI 示例命令。
4. 最近一次检查结果中记录：
   real bilingual NLI backend integration: passed
   stub NLI regression: passed
   local/remote model loading policy: passed
   sync_interface_fields --check: all in sync
5. 当前关键文件状态中说明：
   src/recover_attention/nli_scoring.py 已支持 stub_v0 / hf_nli_en_v0 / hf_nli_zh_v0 / hf_nli_auto_v0。
6. 遗留问题中说明：
   - 真实 NLI backend 需要 torch / transformers。
   - 默认优先使用 models/nli 下的本地模型。
   - 只有显式 --allow-download 时才允许联网下载或从 Hub 加载模型。
   - 若真实 smoke 未运行成功，说明具体原因。
   - LLM recovery 仍未接入。
   - recover_outputs 仍来自 oracle_stub_v0。
   - recover_scores 仍来自 stub_rule_v0。
   - attention_anchor_labels 尚未用真实 NLI 全量重建 downstream。
7. 下一步建议：
   Sprint 1M：Plug Real LLM Recovery Backend
```

---

## 23. docs/progress/sprint_1_history.md 更新要求

更新：

```text id="p8raf1"
docs/progress/sprint_1_history.md
```

追加：

```text id="d9o7qe"
## Sprint 1L：Plug Real Bilingual NLI Backend
```

内容包括：

```text id="y5jkce"
1. 已完成内容。
2. 新增/修改文件。
3. 新增 backend：
   hf_nli_en_v0
   hf_nli_zh_v0
   hf_nli_auto_v0
4. 中英文语言处理方式。
5. 本地模型路径。
6. 远程模型 repo id。
7. allow_download 策略。
8. 是否运行真实模型 smoke。
9. 运行命令。
10. 检查结果。
11. 遗留问题。
12. 下一步建议：Sprint 1M：Plug Real LLM Recovery Backend。
```

---

## 24. 验收标准

本 sprint 完成后必须满足：

```text id="d9f68x"
1. stub_v0 行为保持不变。
2. --language auto/en/zh 行为保持不变。
3. schemas.py 中 ALLOWED_NLI_BACKENDS 包含 hf_nli_en_v0 / hf_nli_zh_v0 / hf_nli_auto_v0。
4. nli_scores 顶层字段未变化。
5. validate_nli_score_record 仍通过。
6. scripts/05_run_nli_scoring.py 支持 --en-model / --zh-model / --en-model-id / --zh-model-id / --allow-download / --device / --batch-size / --max-length / --label-order / --limit。
7. 默认 allow_download=False。
8. 本地模型路径存在时优先加载本地模型。
9. 本地模型路径缺失且 allow_download=False 时必须报错。
10. 只有 allow_download=True 时才允许联网下载或 Hub 加载。
11. hf_nli_en_v0 对英文样本可路由到英文模型。
12. hf_nli_zh_v0 对中文样本可路由到中文 / multilingual 模型。
13. hf_nli_auto_v0 可按 resolved language 路由。
14. 输出 label 统一为 entailment / neutral / contradiction。
15. scores 三类概率和约等于 1。
16. pytest 不下载真实模型。
17. 缺少 torch / transformers 时，真实 backend 报错清晰。
18. stub CLI smoke 通过。
19. 真实 NLI smoke 使用本地模型路径成功运行，或明确说明失败原因。
20. tests/test_nli_scoring.py 通过。
21. tests/test_schemas.py 通过。
22. 全量 pytest 通过。
23. 没有修改 downstream semantic label rule。
24. 没有接入 LLM recovery。
25. 没有执行 attention guidance。
26. 没有做目录重构。
27. 没有修改 models/*。
28. PROGRESS.md 已更新。
29. docs/progress/sprint_1_history.md 已更新。
```

---

## 25. 完成后回复格式

完成后请按以下格式回复：

```text id="bj5hh0"
1. 本次完成内容
2. 新增/修改文件
3. 新增 backend
4. 中英文 NLI 支持方式
5. 本地模型路径与远程 model id
6. allow_download 策略
7. 运行命令
8. 检查结果
9. 是否运行真实模型 smoke；若未运行或失败，说明原因
10. PROGRESS.md 更新摘要
11. docs/progress/sprint_1_history.md 更新摘要
12. 遗留问题
13. 下一步建议
```

下一步建议必须是：

```text id="e7j9m3"
Sprint 1M：Plug Real LLM Recovery Backend
```

不要自动开始 Sprint 1M。
