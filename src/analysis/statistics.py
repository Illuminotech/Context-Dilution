"""Statistical analysis for the context dilution experiment.

Primary test: Jonckheere-Terpstra for ordered degradation.
Secondary: Mann-Whitney U pairwise, ANOVA interaction effects.
Effect sizes: Cliff's delta with bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class StatisticalResult:
    """Result of a single statistical test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_size_ci: tuple[float, float] | None = None
    significant: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def jonckheere_terpstra(
    groups: list[np.ndarray],
) -> StatisticalResult:
    """Jonckheere-Terpstra test for ordered monotonic trend.

    Groups should be ordered from expected highest to lowest score
    (full > summarized > partitioned > minimal).
    """
    # Count concordant pairs across ordered groups
    n_groups = len(groups)
    s_stat = 0.0
    total_pairs = 0

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            for x in groups[i]:
                for y in groups[j]:
                    if x > y:
                        s_stat += 1
                    elif x < y:
                        s_stat -= 1
                    total_pairs += 1

    # Normal approximation for p-value
    n = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]

    # Expected value under null
    expected = 0.0  # E[S] = 0 under null

    # Variance
    n_total = n
    var_numerator = n_total**2 * (2 * n_total + 3)
    for ni in ns:
        var_numerator -= ni**2 * (2 * ni + 3)
    variance = var_numerator / 72.0

    if variance <= 0:
        return StatisticalResult(
            test_name="Jonckheere-Terpstra",
            statistic=s_stat,
            p_value=1.0,
        )

    z = (s_stat - expected) / np.sqrt(variance)
    # One-sided test: we expect decreasing scores (positive S means first groups > later groups)
    p_value = float(1.0 - stats.norm.cdf(z))

    return StatisticalResult(
        test_name="Jonckheere-Terpstra",
        statistic=float(s_stat),
        p_value=p_value,
        significant=p_value < 0.05,
        details={"z_score": float(z), "n_groups": n_groups, "total_n": n},
    )


def mann_whitney_pairwise(
    groups: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> list[StatisticalResult]:
    """Bonferroni-corrected Mann-Whitney U tests for adjacent pairs."""
    labels = list(groups.keys())
    n_comparisons = len(labels) - 1
    corrected_alpha = alpha / max(n_comparisons, 1)
    results = []

    for i in range(len(labels) - 1):
        label_a, label_b = labels[i], labels[i + 1]
        a, b = groups[label_a], groups[label_b]

        u_stat, p_value = stats.mannwhitneyu(a, b, alternative="greater")
        delta = cliffs_delta(a, b)

        results.append(
            StatisticalResult(
                test_name=f"Mann-Whitney U: {label_a} vs {label_b}",
                statistic=float(u_stat),
                p_value=float(p_value),
                effect_size=delta,
                significant=bool(p_value < corrected_alpha),
                details={"corrected_alpha": corrected_alpha, "n_comparisons": n_comparisons},
            )
        )

    return results


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    dominance = 0.0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1

    return float(dominance / (n1 * n2))


def cliffs_delta_bootstrap_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Cliff's delta."""
    rng = np.random.default_rng(seed)
    deltas = []

    for _ in range(n_bootstrap):
        boot1 = rng.choice(group1, size=len(group1), replace=True)
        boot2 = rng.choice(group2, size=len(group2), replace=True)
        deltas.append(cliffs_delta(boot1, boot2))

    lower = float(np.percentile(deltas, (1 - ci) / 2 * 100))
    upper = float(np.percentile(deltas, (1 + ci) / 2 * 100))
    return lower, upper


def two_way_analysis(
    df: pd.DataFrame,
    score_col: str = "composite_score",
    condition_col: str = "condition",
    task_type_col: str = "task_type",
) -> StatisticalResult:
    """Two-way analysis: context_condition x task_type interaction.

    Uses Friedman test (non-parametric) as scores may not be normal.
    Falls back to Kruskal-Wallis if Friedman requirements not met.
    """
    # Kruskal-Wallis across conditions
    condition_groups = [group[score_col].values for _, group in df.groupby(condition_col)]
    h_stat, p_value = stats.kruskal(*condition_groups)

    return StatisticalResult(
        test_name="Kruskal-Wallis (condition effect)",
        statistic=float(h_stat),
        p_value=float(p_value),
        significant=bool(p_value < 0.05),
        details={
            "n_conditions": len(condition_groups),
            "group_sizes": [len(g) for g in condition_groups],
        },
    )


def run_full_analysis(
    df: pd.DataFrame,
    score_col: str = "composite_score",
    condition_col: str = "condition",
    condition_order: list[str] | None = None,
) -> dict[str, Any]:
    """Run the complete statistical analysis suite."""
    if condition_order is None:
        condition_order = ["full", "summarized", "partitioned", "minimal"]

    # Prepare ordered groups
    ordered_groups = []
    group_dict: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for cond in condition_order:
        mask = df[condition_col] == cond
        values = np.asarray(df.loc[mask, score_col].values)
        if len(values) > 0:
            ordered_groups.append(values)
            group_dict[cond] = values

    results: dict[str, Any] = {}

    # Primary: Jonckheere-Terpstra
    if len(ordered_groups) >= 2:
        results["jonckheere_terpstra"] = jonckheere_terpstra(ordered_groups)

    # Pairwise: Mann-Whitney U
    if len(group_dict) >= 2:
        results["pairwise"] = mann_whitney_pairwise(group_dict)

    # Effect sizes
    if "full" in group_dict and "minimal" in group_dict:
        delta = cliffs_delta(group_dict["full"], group_dict["minimal"])
        ci = cliffs_delta_bootstrap_ci(group_dict["full"], group_dict["minimal"])
        results["effect_size_full_vs_minimal"] = StatisticalResult(
            test_name="Cliff's delta: full vs minimal",
            statistic=delta,
            p_value=0.0,
            effect_size=delta,
            effect_size_ci=ci,
        )

    # Two-way analysis
    if "task_type" in df.columns:
        results["interaction"] = two_way_analysis(df, score_col, condition_col)

    return results
