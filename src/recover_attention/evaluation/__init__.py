"""Stage 0 production evaluation pipeline（W0.5-E）。

把 smoke 版升级成论文版:5 folds / nested C / bootstrap>=1000 / 完整 artifact 链。

分层原则:本包只做**编排**,统计与 schema 的纯函数仍在
`recover_attention.conditional_increment` 与 `recover_attention.mcq_conditional_increment`
里（它们已有单测覆盖,不搬动 —— 重构那些会白白引入风险）。本包负责把它们按冻结协议串成
可审稿的实验系统:

    feature_cache -> ladder -> {O, H, O+H} -> increment(ΔH CI)
                            -> artifact redline
                            -> observability gate(D = S_MCQ - S_H1)
                            -> rq2
设计权威:docs/paper/preregistration.md v2.3（sha256 16fa43db…）。
"""

from recover_attention.evaluation.config import Stage0Config, load_config  # noqa: F401
