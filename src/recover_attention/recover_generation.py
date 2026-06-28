"""Build unit-level recovery output records from masked question records."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from recover_attention.data_io import read_jsonl, write_jsonl
from recover_attention.schemas import (
    validate_masked_question_record,
    validate_recover_output_record,
)


ORACLE_STUB_BACKEND = "oracle_stub_v0"
OLLAMA_CHAT_BACKEND = "ollama_chat_v0"

DEFAULT_RECOVERY_BACKEND = ORACLE_STUB_BACKEND
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
FALLBACK_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT = 120
DEFAULT_SEED = 42
SUPPORTED_RECOVERY_BACKENDS = {ORACLE_STUB_BACKEND, OLLAMA_CHAT_BACKEND}
DEFAULT_NUM_SAMPLES = 1

RECOVER_OUTPUT_SOURCE_FIELDS = (
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
)


def build_recovered_question_oracle_stub(masked_question_record: dict) -> str:
    """Return the original question for pipeline verification."""
    return masked_question_record["original_question"]


def build_recovery_prompt(masked_question_record: dict) -> str:
    """Build a leakage-guarded prompt from masked input only."""
    masked_question = masked_question_record["masked_question"]
    mask_token = masked_question_record["mask_token"]
    mask_count = masked_question.count(mask_token)

    return "\n".join(
        [
            f"You are given a question with one or more {mask_token} tokens.",
            "Recover the missing text using only the remaining context in the masked question.",
            "",
            "Rules:",
            "- Do not solve the question.",
            "- Do not explain.",
            "- Do not add reasoning.",
            f"- Replace every {mask_token} token.",
            "- Return only the recovered full question.",
            "",
            f"Mask token: {mask_token}",
            f"Number of masks: {mask_count}",
            "",
            "Masked question:",
            masked_question,
        ]
    )


def call_ollama_chat(
    prompt: str,
    model: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int | None = DEFAULT_SEED,
) -> str:
    """Call local Ollama ``/api/chat`` and return message content."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": options,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if "not found" in body.lower() or exc.code == 404:
            raise RuntimeError(
                f"Ollama model not found or unavailable: {model}. "
                "Install it outside this script; this sprint does not run ollama pull."
            ) from exc
        raise RuntimeError(
            f"Ollama chat request failed with HTTP {exc.code} at {endpoint}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {endpoint}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama chat request timed out after {timeout} seconds at {endpoint}"
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON from {endpoint}") from exc

    if isinstance(response_data, dict) and response_data.get("error"):
        error_text = str(response_data["error"])
        if "not found" in error_text.lower():
            raise RuntimeError(
                f"Ollama model not found or unavailable: {model}. "
                "Install it outside this script; this sprint does not run ollama pull."
            )
        raise RuntimeError(f"Ollama returned an error for model {model}: {error_text}")

    try:
        content = response_data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Ollama response did not contain response['message']['content']"
        ) from exc
    if not isinstance(content, str):
        raise RuntimeError("Ollama response message content must be a string")
    return content


def clean_recovered_question(text: str) -> str:
    """Clean common chat formatting from a recovered question string."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    fence_match = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    prefix_pattern = re.compile(r"^(Recovered question|Answer|Output)\s*:\s*", re.IGNORECASE)
    while True:
        stripped = prefix_pattern.sub("", cleaned, count=1).strip()
        if stripped == cleaned:
            return stripped
        cleaned = stripped


def build_recover_output_record(
    masked_question_record: dict,
    sample_id: int,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int | None = DEFAULT_SEED,
) -> dict:
    """Build one validated recovery output record from one masked question."""
    validate_masked_question_record(masked_question_record)
    _validate_backend(backend)
    _validate_sample_id(sample_id)

    record = {
        field: deepcopy(masked_question_record[field])
        for field in RECOVER_OUTPUT_SOURCE_FIELDS
    }
    record.update(
        {
            "recovered_question": _build_recovered_question(
                masked_question_record,
                backend,
                model=model,
                ollama_base_url=ollama_base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                seed=seed,
            ),
            "recovery_backend": backend,
            "sample_id": sample_id,
        }
    )
    validate_recover_output_record(record)
    return record


def build_recover_output_records(
    masked_question_records: list[dict],
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int | None = DEFAULT_SEED,
) -> tuple[list[dict], dict]:
    """Build recovery output records and return records plus summary stats."""
    _validate_backend(backend)
    _validate_num_samples(num_samples)

    for masked_question_record in masked_question_records:
        validate_masked_question_record(masked_question_record)

    stats = _empty_stats(
        masked_question_records,
        backend,
        num_samples,
        model=model,
        ollama_base_url=ollama_base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        seed=seed,
    )
    output_records: list[dict] = []

    for masked_question_record in masked_question_records:
        for sample_id in range(num_samples):
            output_record = build_recover_output_record(
                masked_question_record,
                sample_id=sample_id,
                backend=backend,
                model=model,
                ollama_base_url=ollama_base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                seed=seed,
            )
            output_records.append(output_record)
            if not output_record["recovered_question"].strip():
                stats["num_empty_recoveries"] += 1

    stats["num_output_recoveries"] = len(output_records)
    stats["unit_scope_counts"] = dict(sorted(stats["unit_scope_counts"].items()))
    stats["group_type_counts"] = dict(sorted(stats["group_type_counts"].items()))
    stats["mask_backend_counts"] = dict(sorted(stats["mask_backend_counts"].items()))
    stats["mask_strategy_counts"] = dict(sorted(stats["mask_strategy_counts"].items()))
    return output_records, stats


def build_recover_output_file(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = DEFAULT_RECOVERY_BACKEND,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int | None = DEFAULT_SEED,
) -> tuple[list[dict], dict]:
    """Read masked questions, build recovery outputs, and write JSONL output."""
    input_jsonl = Path(input_path)
    if not input_jsonl.exists():
        raise FileNotFoundError(
            f"Missing input: {input_jsonl}\nPlease run Sprint 1F first."
        )

    masked_question_records = read_jsonl(input_jsonl)
    records, stats = build_recover_output_records(
        masked_question_records,
        backend=backend,
        num_samples=num_samples,
        model=model,
        ollama_base_url=ollama_base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        seed=seed,
    )
    write_jsonl(records, output_path)
    return records, stats


def _build_recovered_question(
    masked_question_record: dict,
    backend: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    seed: int | None = DEFAULT_SEED,
) -> str:
    if backend == ORACLE_STUB_BACKEND:
        return build_recovered_question_oracle_stub(masked_question_record)
    if backend == OLLAMA_CHAT_BACKEND:
        prompt = build_recovery_prompt(masked_question_record)
        response_text = call_ollama_chat(
            prompt,
            model=model,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
        )
        return clean_recovered_question(response_text)
    _validate_backend(backend)
    raise AssertionError("unreachable backend validation state")


def _empty_stats(
    masked_question_records: list[dict],
    backend: str,
    num_samples: int,
    model: str,
    ollama_base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    seed: int | None,
) -> dict:
    return {
        "num_input_masks": len(masked_question_records),
        "num_output_recoveries": 0,
        "num_samples": num_samples,
        "recovery_backend": backend,
        "model": model,
        "ollama_base_url": ollama_base_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "seed": seed,
        "num_empty_recoveries": 0,
        "unit_scope_counts": Counter(
            record["unit_scope"] for record in masked_question_records
        ),
        "group_type_counts": Counter(
            record["group_type"] for record in masked_question_records
        ),
        "mask_backend_counts": Counter(
            record["mask_backend"] for record in masked_question_records
        ),
        "mask_strategy_counts": Counter(
            record["mask_strategy"] for record in masked_question_records
        ),
    }


def _validate_backend(backend: str) -> None:
    if backend not in SUPPORTED_RECOVERY_BACKENDS:
        raise ValueError(f"Unsupported recovery backend: {backend}")


def _validate_num_samples(num_samples: int) -> None:
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError("num_samples must be an int")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")


def _validate_sample_id(sample_id: int) -> None:
    if not isinstance(sample_id, int) or isinstance(sample_id, bool):
        raise ValueError("sample_id must be an int")
    if sample_id < 0:
        raise ValueError("sample_id must be >= 0")
