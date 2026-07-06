"""Sprint 2H-B task 2: model-generated recovery, replacing template filler.

The 2G template filler (number -> "several", comparison -> "compared", ...) was a
structural leakage source because filler identity encoded span type, which encoded
the weak label. This module replaces it with model-generated recovery.

Hard constraints (see Sprint 2H-B addendum):
- The recovery prompt MUST NOT reveal the span type of the masked slot.
- Sampling MUST use temperature > 0 (a deterministic K=3 run has no instability).
- Every sampling setting is logged into the output records / report.

Recovery outputs are used ONLY for text-level label construction and offline drift
reports. They never become fragility-probe inputs (that would re-introduce the 2G
leakage with model recovery in place of filler).
"""

from __future__ import annotations

from typing import Any, Callable

from recover_attention.recover_generation import call_ollama_chat, clean_recovered_question


PROMPT_VERSION = "leakage_guarded_recovery_v1"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_SAMPLES = 3
DEFAULT_SEED_BASE = 101
DEFAULT_MAX_TOKENS = 160
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120

UNCERTAIN_TOKEN = "uncertain"


def build_recovery_prompt(masked_question: str, mask_token: str) -> str:
    """Leakage-guarded recovery prompt.

    Intentionally omits any hint about the masked slot's type (number / comparison /
    object / negation / ...). Only the surrounding masked context is provided.
    """
    mask_count = masked_question.count(mask_token)
    return "\n".join(
        [
            f"You are given a question that has {mask_count} hidden slot(s) written as {mask_token}.",
            "Recover the most likely original question by replacing every hidden slot.",
            "",
            "Rules:",
            "- Do not solve the problem.",
            "- Do not explain or add reasoning.",
            f"- Replace every {mask_token} with your best guess for the original text.",
            "- If the hidden text cannot be uniquely recovered, reply with the single "
            f"word: {UNCERTAIN_TOKEN}.",
            "- Otherwise return only the recovered full question, nothing else.",
            "",
            "Masked question:",
            masked_question,
        ]
    )


def _is_uncertain(text: str) -> bool:
    stripped = (text or "").strip().lower().strip(".!").strip()
    return stripped == UNCERTAIN_TOKEN


def sample_recoveries(
    masked_question: str,
    mask_token: str,
    *,
    model: str = DEFAULT_MODEL,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed_base: int = DEFAULT_SEED_BASE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    chat_fn: Callable[..., str] = call_ollama_chat,
) -> list[dict[str, Any]]:
    """Draw ``num_samples`` recoveries for one masked question.

    ``chat_fn`` is injectable so tests can run without a live Ollama server.
    Each sample varies the seed so temperature>0 yields genuine sampling spread.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for meaningful recovery instability")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    prompt = build_recovery_prompt(masked_question, mask_token)
    samples: list[dict[str, Any]] = []
    for sample_id in range(num_samples):
        seed = seed_base + sample_id
        raw_text = chat_fn(
            prompt,
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
        )
        recovered_question = clean_recovered_question(raw_text)
        samples.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "raw_text": raw_text,
                "recovered_question": recovered_question,
                "is_uncertain": _is_uncertain(recovered_question) or _is_uncertain(raw_text),
            }
        )
    return samples


def recovery_settings(
    *,
    model: str = DEFAULT_MODEL,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed_base: int = DEFAULT_SEED_BASE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Return the logged sampling settings block (for report + records)."""
    return {
        "recovery_backend": "ollama_chat_v0",
        "prompt_version": PROMPT_VERSION,
        "model_name": model,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "seed_base": seed_base,
        "max_tokens": max_tokens,
    }
