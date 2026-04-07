"""Composite metric calculation and inter-rater reliability."""

from __future__ import annotations

import itertools

from src.models import AutomatedScores, EvaluationResult, RubricScores


def compute_composite_score(
    automated: AutomatedScores,
    mean_rubric: RubricScores | None,
) -> float:
    """Compute the weighted composite score.

    Weights: 0.15 auto + 0.30 correctness + 0.25 pattern + 0.15 completeness + 0.15 error
    Automated score is scaled from [0,1] to [1,5] range.
    """
    auto_scaled = automated.composite * 4.0 + 1.0  # map [0,1] -> [1,5]
    if mean_rubric is None:
        return auto_scaled
    return (
        0.15 * auto_scaled
        + 0.30 * mean_rubric.correctness
        + 0.25 * mean_rubric.pattern_adherence
        + 0.15 * mean_rubric.completeness
        + 0.15 * mean_rubric.error_avoidance
    )


def krippendorff_alpha_simple(
    ratings: list[list[float]],
) -> float:
    """Compute Krippendorff's alpha for interval-scale data.

    Args:
        ratings: list of raters, each containing a list of scores (one per item).
                 All raters must rate all items.

    Returns:
        Alpha value. 1.0 = perfect agreement, 0.0 = chance, < 0 = worse than chance.
    """
    n_raters = len(ratings)
    if n_raters < 2:
        return 1.0
    n_items = len(ratings[0])
    if n_items < 2:
        return 1.0

    # Within-unit disagreement (Do)
    do = 0.0
    n_pairs_within = 0
    for item_idx in range(n_items):
        item_ratings = [ratings[r][item_idx] for r in range(n_raters)]
        for a, b in itertools.combinations(item_ratings, 2):
            do += (a - b) ** 2
            n_pairs_within += 1

    if n_pairs_within == 0:
        return 1.0
    do /= n_pairs_within

    # Total disagreement (De)
    all_values = [v for rater in ratings for v in rater]
    de = 0.0
    n_pairs_total = 0
    for a, b in itertools.combinations(all_values, 2):
        de += (a - b) ** 2
        n_pairs_total += 1

    if n_pairs_total == 0 or de == 0:
        return 1.0
    de /= n_pairs_total

    return 1.0 - do / de


def check_inter_rater_reliability(
    evaluation_results: list[EvaluationResult],
    min_alpha: float = 0.67,
) -> dict[str, float]:
    """Check inter-rater reliability across all evaluation results.

    Returns dict mapping dimension -> alpha value.
    Raises warning if any dimension < min_alpha.
    """
    alphas: dict[str, float] = {}

    # Collect per-dimension ratings across all items and replicas
    dimensions = ("correctness", "pattern_adherence", "completeness", "error_avoidance")

    for dim in dimensions:
        # Build ratings matrix: [rater_idx][item_idx]
        n_replicas = min(
            (len(r.rubric_scores) for r in evaluation_results if r.rubric_scores),
            default=0,
        )
        if n_replicas < 2:
            alphas[dim] = 1.0
            continue

        ratings: list[list[float]] = [[] for _ in range(n_replicas)]
        for result in evaluation_results:
            if len(result.rubric_scores) >= n_replicas:
                for r_idx in range(n_replicas):
                    ratings[r_idx].append(getattr(result.rubric_scores[r_idx], dim))

        if all(len(r) >= 2 for r in ratings):
            alphas[dim] = krippendorff_alpha_simple(ratings)
        else:
            alphas[dim] = 1.0

    return alphas
