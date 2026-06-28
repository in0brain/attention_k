"""Build unit-level recoverability score records from recovery outputs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from recover_attention.data_io import ensure_dir, read_jsonl, write_jsonl
from recover_attention.nli_scoring import (
    DEFAULT_BATCH_SIZE as DEFAULT_NLI_BATCH_SIZE,
    DEFAULT_DEVICE as DEFAULT_NLI_DEVICE,
    DEFAULT_EN_NLI_MODEL,
    DEFAULT_EN_NLI_MODEL_ID,
    DEFAULT_LABEL_ORDER as DEFAULT_NLI_LABEL_ORDER,
    DEFAULT_MAX_LENGTH as DEFAULT_NLI_MAX_LENGTH,
    DEFAULT_ZH_NLI_MODEL,
    DEFAULT_ZH_NLI_MODEL_ID,
    HF_NLI_AUTO_BACKEND,
    HF_NLI_EN_BACKEND,
    HF_NLI_ZH_BACKEND,
    HFNLIModelBundle,
    detect_language,
    load_hf_nli_model,
    parse_label_order,
    score_nli_pair_hf,
)
from recover_attention.schemas import (
    validate_recover_output_record,
    validate_recover_score_record,
)


STUB_RULE_BACKEND = "stub_rule_v0"
NLI_RECOVERY_JUDGE_BACKEND = "nli_recovery_judge_v0"

DEFAULT_SCORE_BACKEND = STUB_RULE_BACKEND
SUPPORTED_SCORE_BACKENDS = {
    STUB_RULE_BACKEND,
    NLI_RECOVERY_JUDGE_BACKEND,
}
SUPPORTED_RECOVER_SCORE_BACKENDS = SUPPORTED_SCORE_BACKENDS

DEFAULT_RECOVERY_NLI_BACKEND = HF_NLI_AUTO_BACKEND
SUPPORTED_RECOVERY_NLI_BACKENDS = {
    HF_NLI_EN_BACKEND,
    HF_NLI_ZH_BACKEND,
    HF_NLI_AUTO_BACKEND,
}
DEFAULT_RECOVERY_LANGUAGE = "auto"
SUPPORTED_RECOVERY_LANGUAGES = {"auto", "en", "zh"}
DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD = 0.70
DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD = 0.50

RECOVERY_SCORING_NEXT_STEP = "Sprint 1P：Rebuild Downstream with Upgraded Recovery Scoring"
RECOVERY_SCORING_KNOWN_LIMITATIONS = [
    "nli_recovery_judge_v0 依赖 question-level NLI，不直接验证每个 masked span。",
    "NLI 等价不等于最终 reasoning usefulness。",
    "该 scorer 不证明 attention guidance 有效。",
    "该 scorer 不使用 hidden states / attention maps / trajectory stability。",
    "下一步需要用 upgraded recovery scores 重建 unit_evidence / attention_anchor_labels / intervention_manifest。",
]

RECOVER_SCORE_SOURCE_FIELDS = (
    "masked_id",
    "id",
    "unit_id",
    "unit_scope",
    "group_type",
    "span_ids",
    "spans",
    "original_question",
    "masked_question",
    "mask_token",
    "mask_backend",
    "mask_strategy",
    "recovery_backend",
)

GROUP_CONSISTENCY_FIELDS = RECOVER_SCORE_SOURCE_FIELDS


def normalize_question(text: str) -> str:
    """Normalize question text for deterministic comparison."""
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    return " ".join(text.strip().split())


def score_recovery_group_stub_rule_v0(recover_output_records: list[dict]) -> dict:
    """Score one masked_id group using deterministic normalized exact match."""
    _validate_non_empty_group(recover_output_records)
    for record in recover_output_records:
        validate_recover_output_record(record)
    _validate_group_consistency(recover_output_records)
    _validate_unique_sample_ids(recover_output_records)

    sorted_records = _sorted_group_records(recover_output_records)
    original_question = sorted_records[0]["original_question"]
    normalized_original = normalize_question(original_question)

    normalized_recoveries = [
        normalize_question(record["recovered_question"]) for record in sorted_records
    ]
    exact_match_count = sum(
        normalized_recovery == normalized_original
        for normalized_recovery in normalized_recoveries
    )
    empty_count = sum(normalized_recovery == "" for normalized_recovery in normalized_recoveries)
    non_empty_mismatch_count = sum(
        normalized_recovery != "" and normalized_recovery != normalized_original
        for normalized_recovery in normalized_recoveries
    )

    num_samples = len(sorted_records)
    recoverability_score = exact_match_count / num_samples
    normalized_counts = Counter(normalized_recoveries)
    recovery_consistency = max(normalized_counts.values()) / num_samples
    misleading_recovery = non_empty_mismatch_count > 0

    if exact_match_count == num_samples:
        recoverability_label = "Recoverable"
    elif exact_match_count > 0:
        recoverability_label = "Partially Recoverable"
    elif non_empty_mismatch_count > 0:
        recoverability_label = "Misleading Recovery"
    else:
        recoverability_label = "Non-recoverable"

    return {
        "num_samples": num_samples,
        "source_sample_ids": [record["sample_id"] for record in sorted_records],
        "recovered_questions": [record["recovered_question"] for record in sorted_records],
        "recoverability_label": recoverability_label,
        "recoverability_score": recoverability_score,
        "confidence_mean": recoverability_score,
        "recovery_consistency": recovery_consistency,
        "misleading_recovery": misleading_recovery,
        "evidence": {
            "rule_name": STUB_RULE_BACKEND,
            "normalization": "strip_and_collapse_whitespace",
            "num_exact_matches": exact_match_count,
            "num_empty_recoveries": empty_count,
            "num_non_empty_mismatches": non_empty_mismatch_count,
            "exact_match_ratio": recoverability_score,
            "unique_normalized_recoveries": dict(sorted(normalized_counts.items())),
            "notes": (
                "Deterministic exact-match stub for pipeline validation only; "
                "not a semantic recovery metric."
            ),
        },
    }


def score_recovery_sample_with_nli(
    original_question: str,
    recovered_question: str,
    mask_token: str = "[MASK]",
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    _nli_model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None = None,
) -> dict:
    """Score one recovery sample with bidirectional question-level NLI."""
    _validate_non_empty_str(original_question, "original_question")
    if not isinstance(recovered_question, str):
        raise ValueError("recovered_question must be a str")
    _validate_non_empty_str(mask_token, "mask_token")
    _validate_nli_backend(nli_backend)
    _validate_recovery_language(language)
    _validate_thresholds(
        recoverable_entailment_threshold,
        partial_entailment_threshold,
        contradiction_threshold,
    )
    _validate_max_length(max_length)
    parse_label_order(label_order)

    normalized_recovered = normalize_question(recovered_question)
    normalized_original = normalize_question(original_question)

    if normalized_recovered == "":
        return _shortcut_sample_evaluation(
            empty_recovery=True,
            mask_remaining=False,
            exact_match=False,
            label="Non-recoverable",
            score=0.0,
            confidence=1.0,
            misleading=False,
        )
    if mask_token in recovered_question:
        return _shortcut_sample_evaluation(
            empty_recovery=False,
            mask_remaining=True,
            exact_match=False,
            label="Non-recoverable",
            score=0.0,
            confidence=1.0,
            misleading=False,
        )
    if normalized_recovered == normalized_original:
        return _shortcut_sample_evaluation(
            empty_recovery=False,
            mask_remaining=False,
            exact_match=True,
            label="Recoverable",
            score=1.0,
            confidence=1.0,
            misleading=False,
        )

    forward = score_recovery_nli_pair(
        premise=original_question,
        hypothesis=recovered_question,
        nli_backend=nli_backend,
        language=language,
        en_model=en_model,
        zh_model=zh_model,
        en_model_id=en_model_id,
        zh_model_id=zh_model_id,
        allow_download=allow_download,
        device=device,
        max_length=max_length,
        label_order=label_order,
        _nli_model_cache=_nli_model_cache,
    )
    backward = score_recovery_nli_pair(
        premise=recovered_question,
        hypothesis=original_question,
        nli_backend=nli_backend,
        language=language,
        en_model=en_model,
        zh_model=zh_model,
        en_model_id=en_model_id,
        zh_model_id=zh_model_id,
        allow_download=allow_download,
        device=device,
        max_length=max_length,
        label_order=label_order,
        _nli_model_cache=_nli_model_cache,
    )

    bidirectional_entailment_score = _round_score(
        min(
            forward["scores"]["entailment"],
            backward["scores"]["entailment"],
        )
    )
    contradiction_score = _round_score(
        max(
            forward["scores"]["contradiction"],
            backward["scores"]["contradiction"],
        )
    )

    label, score, confidence, misleading = _label_nli_recovery_sample(
        bidirectional_entailment_score=bidirectional_entailment_score,
        contradiction_score=contradiction_score,
        recoverable_entailment_threshold=recoverable_entailment_threshold,
        partial_entailment_threshold=partial_entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    return {
        "empty_recovery": False,
        "mask_remaining": False,
        "exact_match": False,
        "forward": forward,
        "backward": backward,
        "bidirectional_entailment_score": bidirectional_entailment_score,
        "contradiction_score": contradiction_score,
        "sample_recoverability_label": label,
        "sample_recoverability_score": score,
        "sample_confidence": confidence,
        "misleading_recovery": misleading,
    }


def score_recovery_nli_pair(
    premise: str,
    hypothesis: str,
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    _nli_model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None = None,
) -> dict:
    """Score one premise/hypothesis pair with a configured HF NLI backend."""
    _validate_non_empty_str(premise, "premise")
    _validate_non_empty_str(hypothesis, "hypothesis")
    _validate_nli_backend(nli_backend)
    resolved_language = _resolve_recovery_language(premise, hypothesis, language)
    model_key = _resolve_recovery_nli_model_key(nli_backend, resolved_language)
    model_bundle = _get_recovery_nli_model_bundle(
        model_key=model_key,
        en_model=en_model,
        zh_model=zh_model,
        en_model_id=en_model_id,
        zh_model_id=zh_model_id,
        allow_download=allow_download,
        device=device,
        model_cache=_nli_model_cache,
    )
    return score_nli_pair_hf(
        premise=premise,
        hypothesis=hypothesis,
        model_bundle=model_bundle,
        max_length=max_length,
        label_order=label_order,
    )


def score_recovery_group_nli_recovery_judge_v0(
    recover_output_records: list[dict],
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    _nli_model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None = None,
) -> dict:
    """Score one masked_id group with bidirectional NLI recovery judging."""
    _validate_non_empty_group(recover_output_records)
    for record in recover_output_records:
        validate_recover_output_record(record)
    _validate_group_consistency(recover_output_records)
    _validate_unique_sample_ids(recover_output_records)

    sorted_records = _sorted_group_records(recover_output_records)
    original_question = sorted_records[0]["original_question"]
    mask_token = sorted_records[0]["mask_token"]
    model_cache = _nli_model_cache if _nli_model_cache is not None else {}
    sample_evaluations = []
    for record in sorted_records:
        evaluation = score_recovery_sample_with_nli(
            original_question=original_question,
            recovered_question=record["recovered_question"],
            mask_token=mask_token,
            nli_backend=nli_backend,
            language=language,
            en_model=en_model,
            zh_model=zh_model,
            en_model_id=en_model_id,
            zh_model_id=zh_model_id,
            allow_download=allow_download,
            device=device,
            max_length=max_length,
            label_order=label_order,
            recoverable_entailment_threshold=recoverable_entailment_threshold,
            partial_entailment_threshold=partial_entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            _nli_model_cache=model_cache,
        )
        sample_evaluations.append({"sample_id": record["sample_id"], **evaluation})

    num_samples = len(sorted_records)
    recoverability_score = _round_score(
        max(evaluation["sample_recoverability_score"] for evaluation in sample_evaluations)
    )
    confidence_mean = _round_score(
        mean(evaluation["sample_confidence"] for evaluation in sample_evaluations)
    )
    normalized_counts = Counter(
        normalize_question(record["recovered_question"]) for record in sorted_records
    )
    recovery_consistency = _round_score(max(normalized_counts.values()) / num_samples)
    recoverability_label = _aggregate_recoverability_label(sample_evaluations)
    misleading_recovery = any(
        evaluation["misleading_recovery"] for evaluation in sample_evaluations
    )

    return {
        "num_samples": num_samples,
        "source_sample_ids": [record["sample_id"] for record in sorted_records],
        "recovered_questions": [record["recovered_question"] for record in sorted_records],
        "recoverability_label": recoverability_label,
        "recoverability_score": recoverability_score,
        "confidence_mean": confidence_mean,
        "recovery_consistency": recovery_consistency,
        "misleading_recovery": misleading_recovery,
        "evidence": {
            "backend": NLI_RECOVERY_JUDGE_BACKEND,
            "scoring_method": "bidirectional_nli_recovery_judge",
            "nli_backend": nli_backend,
            "language": language,
            "rule_parameters": {
                "recoverable_entailment_threshold": recoverable_entailment_threshold,
                "partial_entailment_threshold": partial_entailment_threshold,
                "contradiction_threshold": contradiction_threshold,
            },
            "sample_evaluations": sample_evaluations,
            "aggregation": {
                "recoverability_score": "max_sample_recoverability_score",
                "confidence_mean": "average_sample_confidence",
                "misleading_recovery": "any_sample_misleading_recovery",
                "recovery_consistency": "max_duplicate_normalized_recovery_ratio",
            },
        },
    }


def build_recover_score_record(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    _nli_model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None = None,
) -> dict:
    """Build one validated recover score record for one masked_id group."""
    _validate_score_backend(score_backend)
    _validate_non_empty_group(recover_output_records)

    for record in recover_output_records:
        validate_recover_output_record(record)
    _validate_group_consistency(recover_output_records)
    _validate_unique_sample_ids(recover_output_records)

    first_record = recover_output_records[0]
    score_fields = _score_group(
        recover_output_records,
        score_backend,
        nli_backend=nli_backend,
        language=language,
        en_model=en_model,
        zh_model=zh_model,
        en_model_id=en_model_id,
        zh_model_id=zh_model_id,
        allow_download=allow_download,
        device=device,
        max_length=max_length,
        label_order=label_order,
        recoverable_entailment_threshold=recoverable_entailment_threshold,
        partial_entailment_threshold=partial_entailment_threshold,
        contradiction_threshold=contradiction_threshold,
        _nli_model_cache=_nli_model_cache,
    )
    recover_score_record = {
        field: deepcopy(first_record[field])
        for field in RECOVER_SCORE_SOURCE_FIELDS
    }
    recover_score_record.update(
        {
            "recover_score_id": f"{first_record['masked_id']}__score_{score_backend}",
            "score_backend": score_backend,
            **score_fields,
        }
    )

    validate_recover_score_record(recover_score_record)
    return recover_score_record


def build_recover_score_records(
    recover_output_records: list[dict],
    score_backend: str = DEFAULT_SCORE_BACKEND,
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    batch_size: int = DEFAULT_NLI_BATCH_SIZE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Group recovery outputs by masked_id and build recover score records."""
    _validate_score_backend(score_backend)
    _validate_batch_size(batch_size)

    grouped_records: dict[str, list[dict]] = {}
    for record in recover_output_records:
        validate_recover_output_record(record)
        grouped_records.setdefault(record["masked_id"], []).append(record)

    model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] = {}
    score_records = [
        build_recover_score_record(
            records_for_masked_id,
            score_backend=score_backend,
            nli_backend=nli_backend,
            language=language,
            en_model=en_model,
            zh_model=zh_model,
            en_model_id=en_model_id,
            zh_model_id=zh_model_id,
            allow_download=allow_download,
            device=device,
            max_length=max_length,
            label_order=label_order,
            recoverable_entailment_threshold=recoverable_entailment_threshold,
            partial_entailment_threshold=partial_entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            _nli_model_cache=model_cache,
        )
        for records_for_masked_id in grouped_records.values()
    ]

    stats = _build_stats(
        recover_output_records,
        score_records,
        score_backend,
        nli_backend=nli_backend,
        language=language,
        batch_size=batch_size,
        max_length=max_length,
        allow_download=allow_download,
        device=device,
        label_order=label_order,
    )
    return score_records, stats


def build_recover_score_file(
    input_path: str | Path,
    output_path: str | Path,
    score_backend: str = DEFAULT_SCORE_BACKEND,
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    en_model_id: str = DEFAULT_EN_NLI_MODEL_ID,
    zh_model_id: str = DEFAULT_ZH_NLI_MODEL_ID,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    batch_size: int = DEFAULT_NLI_BATCH_SIZE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Read recovery outputs, build recover scores, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing input: {input_jsonl}\nPlease run Sprint 1G first."
        )

    recover_output_records = read_jsonl(input_jsonl)
    records, stats = build_recover_score_records(
        recover_output_records,
        score_backend=score_backend,
        nli_backend=nli_backend,
        language=language,
        en_model=en_model,
        zh_model=zh_model,
        en_model_id=en_model_id,
        zh_model_id=zh_model_id,
        allow_download=allow_download,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        label_order=label_order,
        recoverable_entailment_threshold=recoverable_entailment_threshold,
        partial_entailment_threshold=partial_entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    write_jsonl(records, output_path)
    return records, stats


def build_recovery_scoring_report(
    recover_output_records: list[dict],
    score_records: list[dict],
    *,
    input_path: str | Path,
    output_path: str | Path,
    score_backend: str,
    nli_backend: str = DEFAULT_RECOVERY_NLI_BACKEND,
    language: str = DEFAULT_RECOVERY_LANGUAGE,
    en_model: str = DEFAULT_EN_NLI_MODEL,
    zh_model: str = DEFAULT_ZH_NLI_MODEL,
    allow_download: bool = False,
    device: str = DEFAULT_NLI_DEVICE,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    label_order: str = DEFAULT_NLI_LABEL_ORDER,
    recoverable_entailment_threshold: float = DEFAULT_RECOVERABLE_ENTAILMENT_THRESHOLD,
    partial_entailment_threshold: float = DEFAULT_PARTIAL_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_RECOVERY_CONTRADICTION_THRESHOLD,
    limit: int | None = None,
) -> dict:
    """Build a JSON-serializable recovery scoring report."""
    sample_evaluations = _collect_sample_evaluations(score_records)
    return {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "score_backend": score_backend,
            "nli_backend": nli_backend,
            "language": language,
            "en_model": en_model,
            "zh_model": zh_model,
            "allow_download": allow_download,
            "device": device,
            "max_length": max_length,
            "label_order": label_order,
            "recoverable_entailment_threshold": recoverable_entailment_threshold,
            "partial_entailment_threshold": partial_entailment_threshold,
            "contradiction_threshold": contradiction_threshold,
            "limit": limit,
        },
        "input_counts": {
            "num_recover_outputs": len(recover_output_records),
            "num_masked_ids": len({record["masked_id"] for record in recover_output_records}),
        },
        "output_counts": {
            "num_recover_scores": len(score_records),
        },
        "backend_counts": dict(
            sorted(Counter(record["score_backend"] for record in score_records).items())
        ),
        "recoverability_label_counts": dict(
            sorted(Counter(record["recoverability_label"] for record in score_records).items())
        ),
        "misleading_recovery_counts": {
            str(value): count
            for value, count in sorted(
                Counter(record["misleading_recovery"] for record in score_records).items()
            )
        },
        "empty_recovery_count": sum(
            1 for evaluation in sample_evaluations if evaluation.get("empty_recovery") is True
        ),
        "mask_remaining_count": sum(
            1 for evaluation in sample_evaluations if evaluation.get("mask_remaining") is True
        ),
        "exact_match_recovery_count": sum(
            1 for evaluation in sample_evaluations if evaluation.get("exact_match") is True
        ),
        "score_distribution": {
            "recoverability_score": _distribution(
                [record["recoverability_score"] for record in score_records]
            ),
            "confidence_mean": _distribution(
                [record["confidence_mean"] for record in score_records]
            ),
            "recovery_consistency": _distribution(
                [record["recovery_consistency"] for record in score_records]
            ),
        },
        "sample_records": _build_report_sample_records(score_records, limit=10),
        "known_limitations": list(RECOVERY_SCORING_KNOWN_LIMITATIONS),
        "next_step_recommendation": RECOVERY_SCORING_NEXT_STEP,
    }


def write_recovery_scoring_report(report: dict, report_path: str | Path) -> None:
    """Write JSON and sibling Markdown reports."""
    path = Path(report_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_recovery_scoring_report_markdown(report, path.with_suffix(".md"))


def write_recovery_scoring_report_markdown(report: dict, report_path: str | Path) -> None:
    """Write a compact Markdown summary for the recovery scoring report."""
    lines = [
        "# Sprint 1O Recovery Scoring Report",
        "",
        "## Run Metadata",
        f"- score_backend: {report['run_metadata']['score_backend']}",
        f"- nli_backend: {report['run_metadata']['nli_backend']}",
        f"- language: {report['run_metadata']['language']}",
        f"- limit: {report['run_metadata']['limit']}",
        "",
        "## Counts",
        f"- num_recover_outputs: {report['input_counts']['num_recover_outputs']}",
        f"- num_recover_scores: {report['output_counts']['num_recover_scores']}",
        f"- empty_recovery_count: {report['empty_recovery_count']}",
        f"- mask_remaining_count: {report['mask_remaining_count']}",
        f"- exact_match_recovery_count: {report['exact_match_recovery_count']}",
        "",
        "## Label Counts",
    ]
    lines.extend(
        f"- {label}: {count}"
        for label, count in report["recoverability_label_counts"].items()
    )
    lines.extend(["", "## Known Limitations"])
    lines.extend(f"- {item}" for item in report["known_limitations"])
    lines.extend(["", f"## Next Step", report["next_step_recommendation"], ""])
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def _score_group(
    recover_output_records: list[dict],
    score_backend: str,
    **kwargs: Any,
) -> dict:
    if score_backend == STUB_RULE_BACKEND:
        return score_recovery_group_stub_rule_v0(recover_output_records)
    if score_backend == NLI_RECOVERY_JUDGE_BACKEND:
        return score_recovery_group_nli_recovery_judge_v0(recover_output_records, **kwargs)
    _validate_score_backend(score_backend)
    raise AssertionError("unreachable score backend validation state")


def _shortcut_sample_evaluation(
    *,
    empty_recovery: bool,
    mask_remaining: bool,
    exact_match: bool,
    label: str,
    score: float,
    confidence: float,
    misleading: bool,
) -> dict:
    return {
        "empty_recovery": empty_recovery,
        "mask_remaining": mask_remaining,
        "exact_match": exact_match,
        "forward": None,
        "backward": None,
        "bidirectional_entailment_score": score,
        "contradiction_score": 0.0,
        "sample_recoverability_label": label,
        "sample_recoverability_score": score,
        "sample_confidence": confidence,
        "misleading_recovery": misleading,
    }


def _label_nli_recovery_sample(
    *,
    bidirectional_entailment_score: float,
    contradiction_score: float,
    recoverable_entailment_threshold: float,
    partial_entailment_threshold: float,
    contradiction_threshold: float,
) -> tuple[str, float, float, bool]:
    if contradiction_score >= contradiction_threshold:
        score = _round_score(max(0.0, 1.0 - contradiction_score))
        return "Misleading Recovery", score, contradiction_score, True
    if bidirectional_entailment_score >= recoverable_entailment_threshold:
        return (
            "Recoverable",
            bidirectional_entailment_score,
            bidirectional_entailment_score,
            False,
        )
    if bidirectional_entailment_score >= partial_entailment_threshold:
        return (
            "Partially Recoverable",
            bidirectional_entailment_score,
            bidirectional_entailment_score,
            False,
        )
    confidence = _round_score(max(1.0 - bidirectional_entailment_score, contradiction_score))
    return "Non-recoverable", bidirectional_entailment_score, confidence, False


def _aggregate_recoverability_label(sample_evaluations: list[dict]) -> str:
    best = max(
        sample_evaluations,
        key=lambda evaluation: (
            evaluation["sample_recoverability_score"],
            -evaluation["sample_id"],
        ),
    )
    if best["sample_recoverability_label"] == "Misleading Recovery":
        recoverable_samples = [
            evaluation
            for evaluation in sample_evaluations
            if evaluation["sample_recoverability_label"] == "Recoverable"
        ]
        if recoverable_samples:
            best = max(
                recoverable_samples,
                key=lambda evaluation: (
                    evaluation["sample_recoverability_score"],
                    -evaluation["sample_id"],
                ),
            )
    return best["sample_recoverability_label"]


def _sorted_group_records(recover_output_records: list[dict]) -> list[dict]:
    return sorted(recover_output_records, key=lambda record: record["sample_id"])


def _validate_score_backend(score_backend: str) -> None:
    if score_backend not in SUPPORTED_SCORE_BACKENDS:
        raise ValueError(f"Unsupported score backend: {score_backend}")


def _validate_nli_backend(nli_backend: str) -> None:
    if nli_backend not in SUPPORTED_RECOVERY_NLI_BACKENDS:
        allowed = ", ".join(sorted(SUPPORTED_RECOVERY_NLI_BACKENDS))
        raise ValueError(f"Unsupported NLI backend: {nli_backend}; allowed: {allowed}")


def _validate_recovery_language(language: str) -> None:
    if language not in SUPPORTED_RECOVERY_LANGUAGES:
        allowed = ", ".join(sorted(SUPPORTED_RECOVERY_LANGUAGES))
        raise ValueError(f"Unsupported language: {language}; allowed: {allowed}")


def _validate_non_empty_group(recover_output_records: list[dict]) -> None:
    if not isinstance(recover_output_records, list) or not recover_output_records:
        raise ValueError("recover_output_records must be a non-empty list")


def _validate_group_consistency(recover_output_records: list[dict]) -> None:
    first_record = recover_output_records[0]
    for record_index, record in enumerate(recover_output_records[1:], start=1):
        for field in GROUP_CONSISTENCY_FIELDS:
            if record[field] != first_record[field]:
                raise ValueError(
                    "recover output records for one masked_id must have consistent "
                    f"{field}; mismatch at record index {record_index}"
                )


def _validate_unique_sample_ids(recover_output_records: list[dict]) -> None:
    sample_ids = [record["sample_id"] for record in recover_output_records]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sample_id values for masked_id group: {duplicates}")


def _validate_non_empty_str(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str")


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be an int >= 1")


def _validate_max_length(max_length: int) -> None:
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be an int >= 1")


def _validate_thresholds(
    recoverable_entailment_threshold: float,
    partial_entailment_threshold: float,
    contradiction_threshold: float,
) -> None:
    for name, value in [
        ("recoverable_entailment_threshold", recoverable_entailment_threshold),
        ("partial_entailment_threshold", partial_entailment_threshold),
        ("contradiction_threshold", contradiction_threshold),
    ]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number")
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if recoverable_entailment_threshold < partial_entailment_threshold:
        raise ValueError(
            "recoverable_entailment_threshold must be >= partial_entailment_threshold"
        )


def _resolve_recovery_language(premise: str, hypothesis: str, language: str) -> str:
    _validate_recovery_language(language)
    if language in {"en", "zh"}:
        return language
    return detect_language(f"{premise}\n{hypothesis}")


def _resolve_recovery_nli_model_key(nli_backend: str, resolved_language: str) -> str:
    if nli_backend == HF_NLI_EN_BACKEND:
        if resolved_language != "en":
            raise ValueError(
                f"{HF_NLI_EN_BACKEND} requires resolved language en; got {resolved_language}"
            )
        return "en"
    if nli_backend == HF_NLI_ZH_BACKEND:
        if resolved_language != "zh":
            raise ValueError(
                f"{HF_NLI_ZH_BACKEND} requires resolved language zh; got {resolved_language}"
            )
        return "zh"
    if nli_backend == HF_NLI_AUTO_BACKEND:
        if resolved_language in {"en", "zh"}:
            return resolved_language
        raise ValueError(f"Unsupported resolved language for auto backend: {resolved_language}")
    _validate_nli_backend(nli_backend)
    raise AssertionError("unreachable NLI backend validation state")


def _get_recovery_nli_model_bundle(
    *,
    model_key: str,
    en_model: str,
    zh_model: str,
    en_model_id: str,
    zh_model_id: str,
    allow_download: bool,
    device: str,
    model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None,
) -> HFNLIModelBundle:
    if model_key == "en":
        local_model_path = en_model
        model_id = en_model_id
    elif model_key == "zh":
        local_model_path = zh_model
        model_id = zh_model_id
    else:
        raise AssertionError(f"Unexpected model key: {model_key}")

    cache_key = (model_key, local_model_path, model_id, allow_download, device)
    if model_cache is not None and cache_key in model_cache:
        return model_cache[cache_key]

    bundle = load_hf_nli_model(
        local_model_path=local_model_path,
        model_id=model_id,
        allow_download=allow_download,
        device=device,
    )
    if model_cache is not None:
        model_cache[cache_key] = bundle
    return bundle


def _round_score(value: float) -> float:
    return round(float(value), 10)


def _build_stats(
    recover_output_records: list[dict],
    score_records: list[dict],
    score_backend: str,
    *,
    nli_backend: str,
    language: str,
    batch_size: int,
    max_length: int,
    allow_download: bool,
    device: str,
    label_order: str,
) -> dict:
    return {
        "num_input_recoveries": len(recover_output_records),
        "num_output_scores": len(score_records),
        "score_backend": score_backend,
        "nli_backend": nli_backend,
        "language": language,
        "allow_download": allow_download,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "label_order": label_order,
        "recovery_backend_counts": dict(
            sorted(Counter(record["recovery_backend"] for record in recover_output_records).items())
        ),
        "backend_counts": dict(
            sorted(Counter(record["score_backend"] for record in score_records).items())
        ),
        "recoverability_label_counts": dict(
            sorted(Counter(record["recoverability_label"] for record in score_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in score_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in score_records).items())
        ),
        "num_misleading_recovery": sum(
            1 for record in score_records if record["misleading_recovery"]
        ),
    }


def _collect_sample_evaluations(score_records: list[dict]) -> list[dict]:
    evaluations = []
    for record in score_records:
        evidence = record.get("evidence", {})
        if isinstance(evidence, dict):
            sample_evaluations = evidence.get("sample_evaluations", [])
            if isinstance(sample_evaluations, list):
                evaluations.extend(
                    evaluation
                    for evaluation in sample_evaluations
                    if isinstance(evaluation, dict)
                )
    return evaluations


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    numeric_values = [float(value) for value in values]
    return {
        "min": _round_score(min(numeric_values)),
        "max": _round_score(max(numeric_values)),
        "mean": _round_score(mean(numeric_values)),
        "median": _round_score(median(numeric_values)),
    }


def _build_report_sample_records(score_records: list[dict], limit: int) -> list[dict]:
    samples = []
    for record in score_records[:limit]:
        samples.append(
            {
                "masked_id": record["masked_id"],
                "id": record["id"],
                "unit_id": record["unit_id"],
                "masked_question": record["masked_question"],
                "original_question": record["original_question"],
                "recovered_questions": list(record["recovered_questions"]),
                "recoverability_label": record["recoverability_label"],
                "recoverability_score": record["recoverability_score"],
                "confidence_mean": record["confidence_mean"],
                "recovery_consistency": record["recovery_consistency"],
                "misleading_recovery": record["misleading_recovery"],
            }
        )
    return samples
