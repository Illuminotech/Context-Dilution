"""Anthropic API client wrapper with batch support, caching, and cost tracking.

This is the ONLY module that imports the anthropic SDK directly.
All other modules use the AgentClient protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic

from src.models import TokenUsage

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of 2026-04)
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
) -> float:
    """Estimate USD cost for a given token usage."""
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


def _extract_usage(response: Any) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _extract_text(response: Any) -> str:
    """Extract text content from an API response."""
    for block in response.content:
        if block.type == "text":
            return str(block.text)
    return ""


class AnthropicClient:
    """Wraps the Anthropic SDK with retry, caching, and cost tracking."""

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 4096,
        use_cache: bool = True,
        use_batch: bool = False,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._use_cache = use_cache
        self._use_batch = use_batch
        self._max_retries = max_retries
        self._base_delay = base_delay
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
        """Make an API call with retry logic.

        Returns (text_output, token_usage, cost_usd).
        """
        system_content = self._build_system_content(system)
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_output_tokens,
                    system=system_content,
                    messages=messages,  # type: ignore[arg-type]  # simplified dict form
                )
                usage = _extract_usage(response)
                cost = estimate_cost(usage, self._model, is_batch=self._use_batch)
                text = _extract_text(response)

                # Track cumulative stats
                self._call_count += 1
                self._total_cost += cost
                self._total_usage = TokenUsage(
                    input_tokens=self._total_usage.input_tokens + usage.input_tokens,
                    output_tokens=self._total_usage.output_tokens + usage.output_tokens,
                    cache_read_tokens=(
                        self._total_usage.cache_read_tokens + usage.cache_read_tokens
                    ),
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

    async def complete_simple(self, system: str, messages: list[dict[str, str]]) -> str:
        """Simplified complete that returns just the text."""
        text, _, _ = await self.complete(system, messages)
        return text
