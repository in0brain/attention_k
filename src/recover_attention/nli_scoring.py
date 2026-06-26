"""NLI semantic consistency scoring backends."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import re
from typing import Any

from recover_attention.schemas import (
    validate_ablated_question_record,
    validate_nli_score_record,
)


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
SUPPORTED_LANGUAGES = {"auto", "en", "zh"}
SUPPORTED_ABLATION_TYPES = {"delete", "generalize"}
SUPPORTED_DIRECTIONS = {"forward", "backward"}
NLI_LABELS = ("entailment", "neutral", "contradiction")
LABEL_PRIORITY = {"entailment": 3, "neutral": 2, "contradiction": 1}

DEFAULT_EN_NLI_MODEL = "models/nli/en/roberta-large-mnli"
DEFAULT_ZH_NLI_MODEL = "models/nli/zh/mdeberta-v3-base-xnli"
DEFAULT_EN_NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"
DEFAULT_ZH_NLI_MODEL_ID = (
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)
OPTIONAL_ZH_ALT_NLI_MODEL = "models/nli/zh_alt/erlangshen-roberta-330m-nli"
OPTIONAL_ZH_ALT_NLI_MODEL_ID = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_DEVICE = "auto"
DEFAULT_LABEL_ORDER = "auto"

BASE_STUB_SCORES = {
    "generalize": {
        "forward": {"entailment": 0.75, "neutral": 0.20, "contradiction": 0.05},
        "backward": {"entailment": 0.35, "neutral": 0.60, "contradiction": 0.05},
    },
    "delete": {
        "forward": {"entailment": 0.55, "neutral": 0.40, "contradiction": 0.05},
        "backward": {"entailment": 0.25, "neutral": 0.70, "contradiction": 0.05},
    },
}


@dataclass(frozen=True)
class HFNLIModelBundle:
    """Loaded HuggingFace NLI assets."""

    tokenizer: Any
    model: Any
    device: str
    model_source: str


def detect_language(text: str) -> str:
    """Return zh when text contains CJK unified ideographs, otherwise en."""
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def resolve_record_language(record: dict, language: str = "auto") -> str:
    """Resolve one record's language from the requested setting."""
    _validate_language(language)
    if language in {"en", "zh"}:
        return language

    combined_text = f"{record.get('original_question', '')}\n{record.get('ablated_question', '')}"
    return detect_language(combined_text)


def score_nli_pair_stub(
    premise: str,
    hypothesis: str,
    ablation_type: str,
    num_spans: int,
    direction: str,
) -> dict:
    """Score one NLI direction with deterministic stub_v0 rules."""
    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("premise must be a non-empty str")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty str")
    _validate_ablation_type(ablation_type)
    _validate_direction(direction)
    _validate_num_spans(num_spans)

    scores = dict(BASE_STUB_SCORES[ablation_type][direction])
    scores = _apply_group_penalty(scores, num_spans)
    label = _label_from_scores(scores)
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "scores": scores,
    }


def parse_label_order(label_order: str = DEFAULT_LABEL_ORDER) -> tuple[str, ...] | None:
    """Parse an explicit NLI label order, or return None for auto mapping."""
    if label_order == "auto":
        return None
    if not isinstance(label_order, str) or not label_order.strip():
        raise ValueError("label_order must be 'auto' or a comma-separated label list")

    labels = tuple(part.strip().lower() for part in label_order.split(","))
    if len(labels) != 3 or set(labels) != set(NLI_LABELS):
        allowed_examples = [
            "auto",
            "contradiction,neutral,entailment",
            "entailment,neutral,contradiction",
        ]
        raise ValueError(
            "Unsupported label_order: "
            f"{label_order}. Expected one of: {', '.join(allowed_examples)}"
        )
    return labels


def resolve_label_mapping(
    id2label: dict[Any, Any] | None,
    label_order: str = DEFAULT_LABEL_ORDER,
) -> dict[int, str]:
    """Map model output indices to canonical entailment/neutral/contradiction labels."""
    explicit_order = parse_label_order(label_order)
    if explicit_order is not None:
        return {index: label for index, label in enumerate(explicit_order)}

    if not isinstance(id2label, dict) or not id2label:
        _raise_unrecognized_label_mapping(id2label)

    mapping: dict[int, str] = {}
    for index, raw_label in _sorted_id2label_items(id2label):
        canonical_label = _canonical_nli_label(str(raw_label))
        if canonical_label is None:
            _raise_unrecognized_label_mapping(id2label)
        mapping[index] = canonical_label

    if set(mapping.values()) != set(NLI_LABELS) or len(mapping) != 3:
        _raise_unrecognized_label_mapping(id2label)
    return mapping


def resolve_model_source(
    local_model_path: str,
    model_id: str,
    allow_download: bool = False,
) -> str:
    """Resolve a local NLI model path or an explicitly allowed HuggingFace repo id."""
    if not isinstance(local_model_path, str) or not local_model_path.strip():
        raise ValueError("local_model_path must be a non-empty str")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty str")

    model_path = Path(local_model_path)
    if model_path.exists():
        return local_model_path

    if _looks_like_repo_id(local_model_path):
        if allow_download:
            return local_model_path
        raise ValueError(
            f"HuggingFace NLI model id requires --allow-download: {local_model_path}"
        )

    if allow_download:
        return model_id

    raise FileNotFoundError(
        f"Local NLI model path does not exist: {local_model_path}\n"
        "Pass --allow-download with a HuggingFace repo id, or download the model "
        "into models/nli/ first."
    )


def load_hf_nli_model(
    local_model_path: str,
    model_id: str,
    allow_download: bool = False,
    device: str = DEFAULT_DEVICE,
) -> HFNLIModelBundle:
    """Load a HuggingFace sequence-classification NLI model."""
    model_source = resolve_model_source(
        local_model_path=local_model_path,
        model_id=model_id,
        allow_download=allow_download,
    )
    transformers, torch = _lazy_import_hf_dependencies()
    resolved_device = _resolve_device(torch, device)
    local_files_only = not allow_download

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_files_only,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_source,
        local_files_only=local_files_only,
    )
    if hasattr(model, "to"):
        model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    return HFNLIModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
        model_source=model_source,
    )


def score_nli_pair_hf(
    premise: str,
    hypothesis: str,
    model_bundle: HFNLIModelBundle,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_order: str = DEFAULT_LABEL_ORDER,
) -> dict:
    """Score one NLI direction with a loaded HuggingFace NLI model."""
    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("premise must be a non-empty str")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty str")
    _validate_max_length(max_length)

    encoded = model_bundle.tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = _move_batch_to_device(encoded, model_bundle.device)

    torch = _optional_import("torch")
    context = torch.no_grad() if torch is not None else nullcontext()
    with context:
        output = model_bundle.model(**encoded)

    logits = _extract_logits(output)
    probabilities = _softmax(_as_float_vector(logits))
    id2label = getattr(getattr(model_bundle.model, "config", None), "id2label", None)
    label_mapping = resolve_label_mapping(id2label, label_order=label_order)
    if len(probabilities) != len(label_mapping):
        raise ValueError(
            "Model output size does not match label mapping: "
            f"{len(probabilities)} logits vs {len(label_mapping)} labels"
        )

    scores = {label: 0.0 for label in NLI_LABELS}
    for index, probability in enumerate(probabilities):
        scores[label_mapping[index]] = probability
    scores = _rounded_scores(scores)
    label = _label_from_scores(scores)
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "scores": scores,
    }


def score_ablated_question_record(
    record: dict,
    backend: str = STUB_BACKEND,
    language: str = "auto",
    en_model: str | None = None,
    zh_model: str | None = None,
    en_model_id: str | None = None,
    zh_model_id: str | None = None,
    allow_download: bool = False,
    device: str = DEFAULT_DEVICE,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_order: str = DEFAULT_LABEL_ORDER,
    _model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] | None = None,
) -> dict:
    """Build one validated NLI score record from one ablated question record."""
    validate_ablated_question_record(record)
    _validate_backend(backend)
    resolved_language = resolve_record_language(record, language)

    if backend == STUB_BACKEND:
        num_spans = len(record["span_ids"])
        forward = score_nli_pair_stub(
            premise=record["original_question"],
            hypothesis=record["ablated_question"],
            ablation_type=record["ablation_type"],
            num_spans=num_spans,
            direction="forward",
        )
        backward = score_nli_pair_stub(
            premise=record["ablated_question"],
            hypothesis=record["original_question"],
            ablation_type=record["ablation_type"],
            num_spans=num_spans,
            direction="backward",
        )
    else:
        model_key = _resolve_hf_backend_model_key(backend, resolved_language)
        model_bundle = _get_hf_model_bundle(
            model_key=model_key,
            en_model=en_model or DEFAULT_EN_NLI_MODEL,
            zh_model=zh_model or DEFAULT_ZH_NLI_MODEL,
            en_model_id=en_model_id or DEFAULT_EN_NLI_MODEL_ID,
            zh_model_id=zh_model_id or DEFAULT_ZH_NLI_MODEL_ID,
            allow_download=allow_download,
            device=device,
            model_cache=_model_cache,
        )
        forward = score_nli_pair_hf(
            premise=record["original_question"],
            hypothesis=record["ablated_question"],
            model_bundle=model_bundle,
            max_length=max_length,
            label_order=label_order,
        )
        backward = score_nli_pair_hf(
            premise=record["ablated_question"],
            hypothesis=record["original_question"],
            model_bundle=model_bundle,
            max_length=max_length,
            label_order=label_order,
        )

    nli_record = _build_nli_score_record(
        record=record,
        backend=backend,
        language=language,
        resolved_language=resolved_language,
        forward=forward,
        backward=backward,
    )
    validate_nli_score_record(nli_record)
    return nli_record


def score_ablated_question_records(
    records: list[dict],
    backend: str = STUB_BACKEND,
    language: str = "auto",
    en_model: str | None = None,
    zh_model: str | None = None,
    en_model_id: str | None = None,
    zh_model_id: str | None = None,
    allow_download: bool = False,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_order: str = DEFAULT_LABEL_ORDER,
) -> tuple[list[dict], dict]:
    """Score ablated question records and return records plus summary stats."""
    _validate_backend(backend)
    _validate_language(language)
    _validate_batch_size(batch_size)
    _validate_max_length(max_length)
    parse_label_order(label_order)

    model_cache: dict[tuple[str, str, str, bool, str], HFNLIModelBundle] = {}
    scored_records = [
        score_ablated_question_record(
            record,
            backend=backend,
            language=language,
            en_model=en_model,
            zh_model=zh_model,
            en_model_id=en_model_id,
            zh_model_id=zh_model_id,
            allow_download=allow_download,
            device=device,
            max_length=max_length,
            label_order=label_order,
            _model_cache=model_cache,
        )
        for record in records
    ]
    stats = {
        "num_input_ablations": len(records),
        "num_output_scores": len(scored_records),
        "backend": backend,
        "language_setting": language,
        "language_counts": dict(
            sorted(Counter(record["language"] for record in scored_records).items())
        ),
        "ablation_type_counts": dict(
            sorted(Counter(record["ablation_type"] for record in scored_records).items())
        ),
        "unit_scope_counts": dict(
            sorted(Counter(record["unit_scope"] for record in scored_records).items())
        ),
        "group_type_counts": dict(
            sorted(Counter(record["group_type"] for record in scored_records).items())
        ),
    }
    if backend != STUB_BACKEND:
        stats["model_sources"] = {
            model_key: bundle.model_source
            for (model_key, *_), bundle in sorted(model_cache.items())
        }
        stats["allow_download"] = allow_download
        stats["device"] = device
        stats["batch_size"] = batch_size
        stats["max_length"] = max_length
        stats["label_order"] = label_order
    return scored_records, stats


def _build_nli_score_record(
    record: dict,
    backend: str,
    language: str,
    resolved_language: str,
    forward: dict,
    backward: dict,
) -> dict:
    return {
        "nli_id": f"{record['ablation_id']}__nli_{backend}",
        "ablation_id": record["ablation_id"],
        "id": record["id"],
        "unit_id": record["unit_id"],
        "unit_scope": record["unit_scope"],
        "group_type": record["group_type"],
        "span_ids": list(record["span_ids"]),
        "spans": [dict(span) for span in record["spans"]],
        "ablation_type": record["ablation_type"],
        "original_question": record["original_question"],
        "ablated_question": record["ablated_question"],
        "nli_backend": backend,
        "language": resolved_language,
        "language_setting": language,
        "forward": forward,
        "backward": backward,
        "bidirectional_entailment_score": min(
            forward["scores"]["entailment"],
            backward["scores"]["entailment"],
        ),
        "contradiction_score": max(
            forward["scores"]["contradiction"],
            backward["scores"]["contradiction"],
        ),
    }


def _get_hf_model_bundle(
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


def _resolve_hf_backend_model_key(backend: str, resolved_language: str) -> str:
    if backend == HF_NLI_EN_BACKEND:
        if resolved_language != "en":
            raise ValueError(
                f"{HF_NLI_EN_BACKEND} requires resolved language en; got {resolved_language}"
            )
        return "en"
    if backend == HF_NLI_ZH_BACKEND:
        if resolved_language != "zh":
            raise ValueError(
                f"{HF_NLI_ZH_BACKEND} requires resolved language zh; got {resolved_language}"
            )
        return "zh"
    if backend == HF_NLI_AUTO_BACKEND:
        if resolved_language in {"en", "zh"}:
            return resolved_language
        raise ValueError(f"Unsupported resolved language for auto backend: {resolved_language}")
    _validate_backend(backend)
    raise AssertionError("unreachable backend validation state")


def _apply_group_penalty(scores: dict[str, float], num_spans: int) -> dict[str, float]:
    penalty = min(0.20, 0.05 * (num_spans - 1)) if num_spans > 1 else 0.0
    adjusted = dict(scores)
    adjusted["entailment"] -= penalty
    adjusted["neutral"] += penalty
    return _rounded_scores(adjusted)


def _rounded_scores(scores: dict[str, float]) -> dict[str, float]:
    rounded = {label: round(float(scores[label]), 10) for label in NLI_LABELS}
    total = sum(rounded.values())
    if abs(total - 1.0) > 1e-10:
        rounded["neutral"] = round(rounded["neutral"] + (1.0 - total), 10)
    return rounded


def _label_from_scores(scores: dict[str, float]) -> str:
    return max(NLI_LABELS, key=lambda label: (scores[label], LABEL_PRIORITY[label]))


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")


def _validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")


def _validate_ablation_type(ablation_type: str) -> None:
    if ablation_type not in SUPPORTED_ABLATION_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_ABLATION_TYPES))
        raise ValueError(f"Unsupported ablation_type: {ablation_type}; allowed: {allowed}")


def _validate_direction(direction: str) -> None:
    if direction not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"Unsupported direction: {direction}; allowed: {allowed}")


def _validate_num_spans(num_spans: int) -> None:
    if not isinstance(num_spans, int) or isinstance(num_spans, bool) or num_spans < 1:
        raise ValueError("num_spans must be an int >= 1")


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be an int >= 1")


def _validate_max_length(max_length: int) -> None:
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be an int >= 1")


def _looks_like_repo_id(value: str) -> bool:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith((".", "/", "~")) or ":" in normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    return len(parts) == 2 and all(part not in {".", ".."} for part in parts)


def _lazy_import_hf_dependencies() -> tuple[Any, Any]:
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Real NLI backend requires torch and transformers. "
            "Install them in the recover_attention environment before running "
            "hf_nli_* backends."
        ) from exc
    return transformers, torch


def _optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _resolve_device(torch: Any, device: str) -> Any:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device cuda was requested but torch.cuda.is_available() is False")
    else:
        resolved = device
    return torch.device(resolved)


def _move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    raise ValueError("NLI model output does not contain logits")


def _as_float_vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("logits must be a non-empty vector")
    return [float(item) for item in value]


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exponentials = [math.exp(value - max_value) for value in values]
    total = sum(exponentials)
    if total <= 0:
        raise ValueError("softmax denominator must be positive")
    return [value / total for value in exponentials]


def _sorted_id2label_items(id2label: dict[Any, Any]) -> list[tuple[int, Any]]:
    items: list[tuple[int, Any]] = []
    for raw_index, raw_label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model.config.id2label has non-integer key: {raw_index!r}") from exc
        items.append((index, raw_label))
    return sorted(items)


def _canonical_nli_label(label: str) -> str | None:
    normalized = label.strip().lower()
    if "entail" in normalized:
        return "entailment"
    if "neutral" in normalized:
        return "neutral"
    if "contrad" in normalized:
        return "contradiction"
    return None


def _raise_unrecognized_label_mapping(id2label: Any) -> None:
    raise ValueError(
        "Unable to infer NLI label mapping from model.config.id2label: "
        f"{id2label!r}. Pass --label-order explicitly."
    )
