"""Tests for src/analysis/statistics.py — statistical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.statistics import (
    cliffs_delta,
    cliffs_delta_bootstrap_ci,
    jonckheere_terpstra,
    mann_whitney_pairwise,
    run_full_analysis,
    two_way_analysis,
)


class TestJonckheereTerpstra:
    def test_monotonic_decrease(self) -> None:
        groups = [
            np.array([5.0, 4.5, 4.8, 5.2]),
            np.array([3.5, 3.8, 3.2, 3.6]),
            np.array([2.0, 2.5, 1.8, 2.2]),
            np.array([1.0, 1.5, 0.8, 1.2]),
        ]
        result = jonckheere_terpstra(groups)
        assert result.statistic > 0
        assert result.p_value < 0.05
        assert result.significant is True

    def test_no_trend(self) -> None:
        rng = np.random.default_rng(42)
        groups = [rng.normal(3.0, 1.0, size=20) for _ in range(4)]
        result = jonckheere_terpstra(groups)
        assert result.p_value > 0.01  # Not strongly significant


class TestMannWhitneyPairwise:
    def test_significant_difference(self) -> None:
        groups = {
            "full": np.array([5.0, 4.5, 4.8, 5.2, 4.9]),
            "minimal": np.array([1.0, 1.5, 0.8, 1.2, 1.3]),
        }
        results = mann_whitney_pairwise(groups)
        assert len(results) == 1
        assert results[0].significant is True

    def test_no_difference(self) -> None:
        rng = np.random.default_rng(42)
        groups = {
            "a": rng.normal(3.0, 0.5, size=20),
            "b": rng.normal(3.0, 0.5, size=20),
        }
        results = mann_whitney_pairwise(groups)
        assert len(results) == 1


class TestCliffsDelta:
    def test_perfect_dominance(self) -> None:
        g1 = np.array([10.0, 11.0, 12.0])
        g2 = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(g1, g2) == 1.0

    def test_no_dominance(self) -> None:
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(g1, g2) == 0.0

    def test_bootstrap_ci(self) -> None:
        g1 = np.array([5.0, 4.5, 4.8, 5.2, 4.9])
        g2 = np.array([2.0, 2.5, 1.8, 2.2, 2.1])
        lo, hi = cliffs_delta_bootstrap_ci(g1, g2)
        assert lo > 0  # strong effect, CI should be positive
        assert hi <= 1.0


class TestTwoWayAnalysis:
    def test_kruskal_wallis(self) -> None:
        df = pd.DataFrame(
            {
                "composite_score": [5, 4.5, 4.8, 3, 2.8, 3.2, 1.5, 1.2, 1.8],
                "condition": ["full"] * 3 + ["summarized"] * 3 + ["minimal"] * 3,
                "task_type": ["sequential"] * 9,
            }
        )
        result = two_way_analysis(df)
        assert result.test_name == "Kruskal-Wallis (condition effect)"
        assert result.p_value < 0.05


class TestRunFullAnalysis:
    def test_complete_pipeline(self) -> None:
        rng = np.random.default_rng(42)
        conditions = ["full", "summarized", "partitioned", "minimal"]
        means = [4.5, 3.5, 2.5, 1.5]
        rows = []
        for cond, mean in zip(conditions, means, strict=True):
            for _ in range(10):
                rows.append(
                    {
                        "composite_score": rng.normal(mean, 0.3),
                        "condition": cond,
                        "task_type": "sequential",
                    }
                )
        df = pd.DataFrame(rows)
        results = run_full_analysis(df)
        assert "jonckheere_terpstra" in results
        assert "pairwise" in results
        assert results["jonckheere_terpstra"].significant is True
