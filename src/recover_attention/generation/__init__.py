"""Stage 0 production generation + feature cache（W0.5-F）。

与 smoke 的差别不在协议,在**工程**:
  - resume:2880 条跑两小时断电不能全部重来
  - fingerprint:每条记录带 model/backend/tokenizer/prompt/seed/温度/max_new_tokens,
    reviewer 最常问"你的 hidden state 来自哪个模型状态"
  - smoke/production 隔离:pilot 记录绝不能漂进 confirmatory

规模事实（先算清楚,别按错的量级做设计）:
  §8 的 H 是每条 completion **一个 pooled 向量**(3584 维 fp32 = 14 KB),不是每 token 一个。
  H1 2880 条 ≈ 41 MB;MCQ 1760 条 ≈ 25 MB。npz 直接存,无需分片/流式/内存映射。
  §11 明确**不缓存** raw attention / 全层 hidden。
"""

from recover_attention.generation.manifest import (  # noqa: F401
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    WorkManifest,
)
