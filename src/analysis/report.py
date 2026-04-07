"""Auto-generate markdown report from experiment results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.statistics import StatisticalResult


def _format_stat_result(result: StatisticalResult) -> str:
    """Format a statistical result as markdown."""
    sig = "**Yes** ✓" if result.significant else "No"
    lines = [
        f"**{result.test_name}**",
        f"- Statistic: {result.statistic:.4f}",
        f"- p-value: {result.p_value:.6f}",
        f"- Significant (alpha=0.05): {sig}",
    ]
    if result.effect_size is not None:
        lines.append(f"- Effect size (Cliff's delta): {result.effect_size:.4f}")
    if result.effect_size_ci is not None:
        lo, hi = result.effect_size_ci
        lines.append(f"- 95% CI: [{lo:.4f}, {hi:.4f}]")
    return "\n".join(lines)


def _descriptive_table(df: pd.DataFrame, score_col: str, condition_col: str) -> str:
    """Generate a descriptive statistics table."""
    stats = df.groupby(condition_col)[score_col].agg(["count", "mean", "std", "median"])
    stats = stats.round(3)
    return stats.to_markdown()


def generate_report(
    df: pd.DataFrame,
    stat_results: dict[str, Any],
    reliability: dict[str, float],
    config_summary: dict[str, Any],
    output_path: Path,
    score_col: str = "composite_score",
    condition_col: str = "condition",
) -> Path:
    """Generate a complete experiment report as markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections: list[str] = []

    # Header
    sections.append(f"""# Context Dilution Experiment Report

**Generated:** {now}
**Experiment:** {config_summary.get("experiment_name", "N/A")}
**Subject Model:** {config_summary.get("subject_model", "N/A")}
**Judge Model:** {config_summary.get("judge_model", "N/A")}
**Trials per cell:** {config_summary.get("trials_per_cell", "N/A")}

---""")

    # Descriptive statistics
    sections.append(f"""## Descriptive Statistics

{_descriptive_table(df, score_col, condition_col)}""")

    # By task type
    if "task_type" in df.columns:
        for task_type in sorted(df["task_type"].unique()):
            subset = df[df["task_type"] == task_type]
            sections.append(f"""### {task_type.title()} Tasks

{_descriptive_table(subset, score_col, condition_col)}""")

    # Statistical tests
    sections.append("## Statistical Analysis\n")

    if "jonckheere_terpstra" in stat_results:
        sections.append(_format_stat_result(stat_results["jonckheere_terpstra"]))
        sections.append("")

    if "pairwise" in stat_results:
        sections.append("### Pairwise Comparisons (Bonferroni-corrected)\n")
        for result in stat_results["pairwise"]:
            sections.append(_format_stat_result(result))
            sections.append("")

    if "effect_size_full_vs_minimal" in stat_results:
        sections.append("### Effect Size\n")
        sections.append(_format_stat_result(stat_results["effect_size_full_vs_minimal"]))
        sections.append("")

    if "interaction" in stat_results:
        sections.append("### Interaction Effect\n")
        sections.append(_format_stat_result(stat_results["interaction"]))
        sections.append("")

    # Inter-rater reliability
    sections.append("## Inter-Rater Reliability (Krippendorff's alpha)\n")
    sections.append("| Dimension | alpha | Adequate (>= 0.67) |")
    sections.append("|-----------|-------|-------------------|")
    for dim, alpha in reliability.items():
        adequate = "Yes" if alpha >= 0.67 else "**No**"
        sections.append(f"| {dim} | {alpha:.3f} | {adequate} |")
    sections.append("")

    # Cost summary
    if "cost_usd" in df.columns:
        total_cost = df["cost_usd"].sum()
        sections.append(f"""## Cost Summary

- **Total cost:** ${total_cost:.2f}
- **Mean cost per trial:** ${df["cost_usd"].mean():.4f}
- **Total trials:** {len(df)}""")

    # Figures
    sections.append("""## Figures

1. `dilution_gradient.png` - Box plots of composite score by context condition
2. `radar_chart.png` - Rubric dimensions per condition
3. `interaction_plot.png` - Mean score x condition, lines per task type
4. `cost_quality.png` - Composite score vs. cost""")

    report = "\n\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path
