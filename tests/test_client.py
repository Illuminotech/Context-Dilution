"""Tests for src/agents/client.py — cost estimation and utilities."""

from __future__ import annotations

import pytest

from src.agents.client import estimate_cost
from src.models import TokenUsage


class TestEstimateCost:
    def test_haiku_cost(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost(usage, "claude-haiku-4-5-20251001", is_batch=False)
        expected = 1000 * 0.80 / 1_000_000 + 500 * 4.00 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_batch_discount(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost_normal = estimate_cost(usage, "claude-haiku-4-5-20251001", is_batch=False)
        cost_batch = estimate_cost(usage, "claude-haiku-4-5-20251001", is_batch=True)
        assert cost_batch == pytest.approx(cost_normal * 0.5)

    def test_cache_tokens(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=500,
            cache_creation_tokens=200,
        )
        cost = estimate_cost(usage, "claude-haiku-4-5-20251001")
        assert cost > 0

    def test_zero_usage(self) -> None:
        usage = TokenUsage()
        cost = estimate_cost(usage, "claude-haiku-4-5-20251001")
        assert cost == 0.0

    def test_unknown_model_uses_default(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost(usage, "unknown-model-xyz")
        assert cost > 0
