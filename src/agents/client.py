"""API client wrappers for Anthropic and OpenAI-compatible endpoints.

Supports three backends:
- "anthropic": Anthropic API (Claude models) with caching and batch support
- "openai": OpenAI-compatible API (Ollama, vLLM, LM Studio, llama.cpp, etc.)
- "openai-cloud": OpenAI cloud API (GPT models)

This is the ONLY module that imports API SDKs directly.
All other modules depend on the LLMClient protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from src.models import TokenUsage

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of 2026-04)
# Local models have zero cost
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.00,
        "cache_read": 0.08,
        "cache_create": 1.00,
        "batch_discount": 0.50,
    },
    "claude-sonnet-4-6-20250514": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_create": 3.75,
        "batch_discount": 0.50,
    },
}

LOCAL_PRICING: dict[str, float] = {
    "input": 0.0,
    "output": 0.0,
    "cache_read": 0.0,
    "cache_create": 0.0,
    "batch_discount": 1.0,
}

DEFAULT_PRICING: dict[str, float] = {
    "input": 3.00,
    "output": 15.00,
    "cache_read": 0.30,
    "cache_create": 3.75,
    "batch_discount": 0.50,
}


class APIError(Exception):
    """Raised when an API call fails after retries."""

    def __init__(self, message: str, task_id: str = "", condition: str = "") -> None:
        self.task_id = task_id
        self.condition = condition
        super().__init__(message)


def estimate_cost(
    usage: TokenUsage,
    model: str,
    is_batch: bool = False,
    is_local: bool = False,
) -> float:
    """Estimate USD cost for a given token usage."""
    if is_local:
        return 0.0
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (
        usage.input_tokens * pricing["input"] / 1_000_000
        + usage.output_tokens * pricing["output"] / 1_000_000
        + usage.cache_read_tokens * pricing["cache_read"] / 1_000_000
        + usage.cache_creation_tokens * pricing["cache_create"] / 1_000_000
    )
    if is_batch:
        cost *= pricing["batch_discount"]
    return cost


class BaseClient:
    """Shared state tracking and interface for all client implementations."""

    def __init__(self) -> None:
        self._total_cost: float = 0.0
        self._total_usage = TokenUsage()
        self._call_count: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    @property
    def call_count(self) -> int:
        return self._call_count

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, TokenUsage, float]:
        """Make an API call. Implemented by subclasses."""
        raise NotImplementedError

    def _track(self, usage: TokenUsage, cost: float) -> None:
        self._call_count += 1
        self._total_cost += cost
        self._total_usage = TokenUsage(
            input_tokens=self._total_usage.input_tokens + usage.input_tokens,
            output_tokens=self._total_usage.output_tokens + usage.output_tokens,
            cache_read_tokens=(self._total_usage.cache_read_tokens + usage.cache_read_tokens),
            cache_creation_tokens=(
                self._total_usage.cache_creation_tokens + usage.cache_creation_tokens
            ),
        )
        logger.info(
            "API call %d: %d input + %d output tokens, $%.4f",
            self._call_count,
            usage.input_tokens,
            usage.output_tokens,
            cost,
        )


class AnthropicClient(BaseClient):
    """Anthropic API client with retry, caching, and cost tracking."""

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        use_cache: bool = True,
        use_batch: bool = False,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._use_cache = use_cache
        self._use_batch = use_batch
        self._max_retries = max_retries
        self._base_delay = base_delay

    def _build_system_content(self, system: str) -> Any:
        """Build system parameter, optionally with cache control."""
        if self._use_cache:
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, TokenUsage, float]:
        """Make an API call with retry logic."""
        system_content = self._build_system_content(system)
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_output_tokens,
                    temperature=self._temperature,
                    system=system_content,
                    messages=messages,  # type: ignore[arg-type]  # simplified dict form
                )
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                    cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0)
                    or 0,
                )
                cost = estimate_cost(usage, self._model, is_batch=self._use_batch)
                text = ""
                for block in response.content:
                    if block.type == "text":
                        text = str(block.text)
                        break

                self._track(usage, cost)
                return text, usage, cost

            except anthropic.RateLimitError as e:
                last_error = e
                delay = self._base_delay * (2**attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code >= 500:
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs",
                        e.status_code,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise APIError(f"API error {e.status_code}: {e.message}") from e

        raise APIError(f"Failed after {self._max_retries} retries: {last_error}") from last_error


class OpenAICompatibleClient(BaseClient):
    """Client for OpenAI-compatible APIs (Ollama, vLLM, LM Studio, etc.).

    Works with any server that implements the OpenAI chat completions endpoint.
    Local servers have zero cost.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        max_output_tokens: int = 4096,
        context_window: int = 16384,
        temperature: float = 0.0,
        is_local: bool = True,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        super().__init__()
        # Import here to keep openai as an optional dependency
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAI-compatible backends. "
                "Install with: pip install openai"
            ) from e

        # Local models can be slow — use 30 min timeout
        timeout = 1800.0 if is_local else 600.0
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._context_window = context_window
        self._temperature = temperature
        self._is_local = is_local
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, TokenUsage, float]:
        """Make an API call to an OpenAI-compatible endpoint."""
        full_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            *messages,
        ]
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                # Ollama uses num_ctx to set context window size
                extra: dict[str, Any] = {}
                if self._is_local:
                    extra["extra_body"] = {"options": {"num_ctx": self._context_window}}
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=full_messages,  # type: ignore[arg-type]
                    max_tokens=self._max_output_tokens,
                    **extra,
                    temperature=self._temperature,
                )
                text = response.choices[0].message.content or ""

                # Extract usage (some local servers don't report it)
                raw_usage = response.usage
                usage = TokenUsage(
                    input_tokens=raw_usage.prompt_tokens if raw_usage else 0,
                    output_tokens=raw_usage.completion_tokens if raw_usage else 0,
                )
                cost = estimate_cost(usage, self._model, is_local=self._is_local)

                self._track(usage, cost)
                return text, usage, cost

            except Exception as e:
                last_error = e
                delay = self._base_delay * (2**attempt)
                logger.warning(
                    "API error (attempt %d/%d): %s, retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

        raise APIError(f"Failed after {self._max_retries} retries: {last_error}") from last_error


def create_client(
    backend: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    max_output_tokens: int = 4096,
    context_window: int = 16384,
    temperature: float = 0.0,
    use_cache: bool = True,
    use_batch: bool = False,
) -> BaseClient:
    """Factory function to create the appropriate client.

    Args:
        backend: "anthropic", "openai" (local), or "openai-cloud"
        model: Model name/ID
        base_url: API base URL (required for openai backend)
        api_key: API key (optional for local servers)
        max_output_tokens: Maximum output tokens
        context_window: Context window size for local models (Ollama num_ctx)
        temperature: Sampling temperature
        use_cache: Enable prompt caching (Anthropic only)
        use_batch: Enable batch API (Anthropic only)
    """
    if backend == "anthropic":
        return AnthropicClient(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            use_cache=use_cache,
            use_batch=use_batch,
        )
    if backend in ("openai", "openai-cloud"):
        return OpenAICompatibleClient(
            model=model,
            base_url=base_url or "http://localhost:11434/v1",
            api_key=api_key or ("ollama" if backend == "openai" else ""),
            max_output_tokens=max_output_tokens,
            context_window=context_window,
            temperature=temperature,
            is_local=(backend == "openai"),
        )
    raise ValueError(f"Unknown backend: {backend}. Use 'anthropic', 'openai', or 'openai-cloud'.")
