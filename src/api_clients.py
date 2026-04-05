"""Unified interface for calling Claude, Gemini, and OpenAI APIs for text classification.

Supports:
    - claude-opus-4-6 (Anthropic)
    - claude-sonnet-4-6 (Anthropic)
    - claude-haiku-4-5-20251001 (Anthropic)
    - gemini-3-flash-preview (Google)
    - gpt-5.4 (OpenAI)

Usage:
    # Single call
    from src.api_clients import call_model
    response = call_model(
        system_prompt="You are an analyst.",
        user_prompt="Classify this file.",
        model="claude-sonnet-4-6",
    )

    # Batch call (Anthropic only, 50% cheaper)
    from src.api_clients import submit_batch, poll_batch, retrieve_batch_results
    batch_id = submit_batch(requests, model, system_prompt)
    poll_batch(batch_id)
    results = retrieve_batch_results(batch_id)
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Model name -> provider mapping
ANTHROPIC_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}
GEMINI_MODELS = {
    "gemini-3-flash-preview",
}
OPENAI_MODELS = {
    "gpt-5.4",
}
SUPPORTED_MODELS = ANTHROPIC_MODELS | GEMINI_MODELS | OPENAI_MODELS


@dataclass
class ModelResponse:
    """Response from a model call."""
    text: str
    input_tokens: int
    output_tokens: int


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    cache_system_prompt: bool = False,
) -> ModelResponse:
    """Call an Anthropic Claude model.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        model: Anthropic model identifier.
        cache_system_prompt: If True, mark the system prompt block with
            cache_control for prompt caching (ephemeral type).

    Returns:
        ModelResponse with text and token usage.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")

    client = anthropic.Anthropic(api_key=api_key)

    # Build system parameter — use cache_control block if caching requested
    if cache_system_prompt:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = system_prompt

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = message.content[0].text
    usage = message.usage

    return ModelResponse(
        text=response_text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    max_retries: int = 5,
) -> ModelResponse:
    """Call a Google Gemini model with retry logic for transient errors.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        model: Gemini model identifier.
        max_retries: Max retry attempts for 503/429 errors.

    Returns:
        ModelResponse with text and token usage.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment or .env")

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                ),
            )

            response_text = response.text
            usage_metadata = response.usage_metadata

            return ModelResponse(
                text=response_text,
                input_tokens=usage_metadata.prompt_token_count,
                output_tokens=usage_metadata.candidates_token_count,
            )
        except Exception as e:
            error_str = str(e)
            is_retryable = "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str
            if is_retryable and attempt < max_retries:
                wait = min(2 ** attempt * 2, 60)
                print(f"    Gemini {error_str[:80]}... retrying in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> ModelResponse:
    """Call an OpenAI model.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User message content.
        model: OpenAI model identifier.

    Returns:
        ModelResponse with text and token usage.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    choice = response.choices[0]
    usage = response.usage

    return ModelResponse(
        text=choice.message.content,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
    )


def call_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    cache_system_prompt: bool = False,
) -> ModelResponse:
    """Call a language model and return the response with token usage.

    Args:
        system_prompt: System-level instructions sent to the model.
        user_prompt: The user message / classification prompt.
        model: Model identifier. Must be one of the supported models.
        cache_system_prompt: If True and model is Anthropic, enable prompt
            caching on the system prompt block.

    Returns:
        ModelResponse with text, input_tokens, and output_tokens.

    Raises:
        ValueError: If the model is not supported or API key is missing.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Supported: {sorted(SUPPORTED_MODELS)}"
        )

    if model in ANTHROPIC_MODELS:
        return _call_anthropic(
            system_prompt,
            user_prompt,
            model,
            cache_system_prompt=cache_system_prompt,
        )
    elif model in OPENAI_MODELS:
        return _call_openai(system_prompt, user_prompt, model)
    else:
        return _call_gemini(system_prompt, user_prompt, model)


# ---------------------------------------------------------------------------
# Batch API (Anthropic only)
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Return an Anthropic client using the API key from environment."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")
    return anthropic.Anthropic(api_key=api_key)


def submit_batch(
    requests: list[dict],
    model: str,
    system_prompt: str,
    *,
    cache_system_prompt: bool = True,
    max_tokens: int = 1024,
) -> str:
    """Submit a batch of classification requests to the Anthropic Batch API.

    Args:
        requests: List of dicts with 'custom_id' and 'user_prompt' keys.
        model: Anthropic model identifier.
        system_prompt: System prompt shared across all requests.
        cache_system_prompt: Enable prompt caching on system prompt.
        max_tokens: Max output tokens per request.

    Returns:
        Batch ID string for polling and retrieval.
    """
    if model not in ANTHROPIC_MODELS:
        raise ValueError(f"Batch API only supports Anthropic models, got: {model}")

    client = _get_anthropic_client()

    if cache_system_prompt:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = system_prompt

    batch_requests = []
    for req in requests:
        batch_requests.append({
            "custom_id": req["custom_id"],
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [
                    {"role": "user", "content": req["user_prompt"]},
                ],
            },
        })

    batch = client.messages.batches.create(requests=batch_requests)
    return batch.id


def poll_batch(batch_id: str, interval: int = 30) -> dict:
    """Poll a batch until it finishes. Returns the final batch status.

    Args:
        batch_id: The batch ID from submit_batch.
        interval: Seconds between polls (default: 30).

    Returns:
        Dict with 'processing_status' and 'request_counts'.
    """
    client = _get_anthropic_client()

    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts

        print(
            f"  Batch {batch_id[:12]}... "
            f"succeeded={counts.succeeded} "
            f"errored={counts.errored} "
            f"processing={counts.processing}"
        )

        if status.processing_status == "ended":
            return {
                "processing_status": status.processing_status,
                "succeeded": counts.succeeded,
                "failed": counts.errored,
                "expired": counts.expired,
            }

        time.sleep(interval)


@dataclass
class BatchResult:
    """Result for a single request in a batch."""
    custom_id: str
    succeeded: bool
    text: str | None
    input_tokens: int
    output_tokens: int
    error: str | None


def retrieve_batch_results(batch_id: str) -> list[BatchResult]:
    """Retrieve results from a completed batch.

    Args:
        batch_id: The batch ID from submit_batch.

    Returns:
        List of BatchResult objects, one per request.
    """
    client = _get_anthropic_client()

    results = []
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            message = entry.result.message
            text = message.content[0].text if message.content else None
            results.append(BatchResult(
                custom_id=entry.custom_id,
                succeeded=text is not None,
                text=text,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                error=None if text else "empty response",
            ))
        else:
            results.append(BatchResult(
                custom_id=entry.custom_id,
                succeeded=False,
                text=None,
                input_tokens=0,
                output_tokens=0,
                error=str(entry.result.error),
            ))

    return results
