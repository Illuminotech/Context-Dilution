"""Gold set calibration — compares human and LLM judge scores.

Computes agreement metrics between human evaluators and the LLM judge
to quantify judge reliability and identify systematic biases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models import RubricScores

RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "pattern_adherence",
    "completeness",
    "error_avoidance",
)


@dataclass(frozen=True)
class AgreementMetrics:
    """Agreement between human and LLM judge on a single dimension."""

    dimension: str
    n_items: int
    mean_human: float
    mean_llm: float
    mean_absolute_error: float
    bias: float  # positive = LLM scores higher than human
    cohens_kappa: float
    pearson_r: float


@dataclass
class CalibrationReport:
    """Full calibration report across all dimensions."""

    by_dimension: list[AgreementMetrics] = field(default_factory=list)
    overall_mae: float = 0.0
    overall_bias: float = 0.0
    overall_kappa: float = 0.0


def _cohens_kappa(human: list[float], llm: list[float], n_categories: int = 5) -> float:
    """Compute Cohen's kappa for ordinal agreement."""
    if not human or len(human) != len(llm):
        return 0.0

    n = len(human)

    # Build confusion matrix
    matrix = [[0] * n_categories for _ in range(n_categories)]
    for h, a in zip(human, llm, strict=True):
        hi = min(int(h) - 1, n_categories - 1)
        ai = min(int(a) - 1, n_categories - 1)
        matrix[hi][ai] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(n_categories)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(n_categories)]
    col_sums = [sum(matrix[r][c] for r in range(n_categories)) for c in range(n_categories)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(n_categories)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return float(cov / denom)


def compute_calibration(
    human_scores: dict[str, RubricScores],
    llm_scores: dict[str, RubricScores],
) -> CalibrationReport:
    """Compare human and LLM judge scores on shared trial IDs.

    Args:
        human_scores: trial_id -> human RubricScores
        llm_scores: trial_id -> LLM judge mean RubricScores
    """
    shared_ids = sorted(set(human_scores.keys()) & set(llm_scores.keys()))
    if not shared_ids:
        return CalibrationReport()

    dimension_metrics: list[AgreementMetrics] = []
    all_errors: list[float] = []
    all_biases: list[float] = []
    all_kappas: list[float] = []

    for dim in RUBRIC_DIMENSIONS:
        h_vals = [getattr(human_scores[tid], dim) for tid in shared_ids]
        l_vals = [getattr(llm_scores[tid], dim) for tid in shared_ids]

        errors = [abs(h - a) for h, a in zip(h_vals, l_vals, strict=True)]
        biases = [a - h for h, a in zip(h_vals, l_vals, strict=True)]
        mae = sum(errors) / len(errors)
        bias = sum(biases) / len(biases)
        kappa = _cohens_kappa(h_vals, l_vals)
        r = _pearson_r(h_vals, l_vals)

        metrics = AgreementMetrics(
            dimension=dim,
            n_items=len(shared_ids),
            mean_human=sum(h_vals) / len(h_vals),
            mean_llm=sum(l_vals) / len(l_vals),
            mean_absolute_error=mae,
            bias=bias,
            cohens_kappa=kappa,
            pearson_r=r,
        )
        dimension_metrics.append(metrics)
        all_errors.append(mae)
        all_biases.append(bias)
        all_kappas.append(kappa)

    return CalibrationReport(
        by_dimension=dimension_metrics,
        overall_mae=sum(all_errors) / len(all_errors),
        overall_bias=sum(all_biases) / len(all_biases),
        overall_kappa=sum(all_kappas) / len(all_kappas),
    )


def format_calibration_report(report: CalibrationReport) -> str:
    """Format calibration report as markdown."""
    lines = [
        "## Human vs LLM Judge Calibration\n",
        f"Overall MAE: {report.overall_mae:.3f}",
        f"Overall Bias: {report.overall_bias:+.3f} "
        f"({'LLM scores higher' if report.overall_bias > 0 else 'human scores higher'})",
        f"Overall Cohen's kappa: {report.overall_kappa:.3f}\n",
        "| Dimension | N | Human Mean | LLM Mean | MAE | Bias | Kappa | r |",
        "|-----------|---|------------|----------|-----|------|-------|---|",
    ]
    for m in report.by_dimension:
        lines.append(
            f"| {m.dimension} | {m.n_items} | {m.mean_human:.2f} | "
            f"{m.mean_llm:.2f} | {m.mean_absolute_error:.2f} | "
            f"{m.bias:+.2f} | {m.cohens_kappa:.2f} | {m.pearson_r:.2f} |"
        )
    return "\n".join(lines)
